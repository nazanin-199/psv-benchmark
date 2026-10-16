import argparse
import copy
import math
import random
import yaml
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from src.utils.seed import seed_everything
from src.data.dataset_curation import download_curated_ec_benchmark
from src.data.parser import PDBStructureParser
from src.data.clustering import SequenceClusterSplitter, generate_kfold_cluster_splits
from src.features.esm_extractor import ESMFeatureExtractor
from src.features.graph_builder import construct_contact_graph
from src.models.sequence_mlp import SequenceOnlyModel
from src.models.hybrid_model import ProteinGraphModel
from src.utils.metrics import compute_classification_metrics

def get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs, num_total_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < num_warmup_epochs:
            return float(current_epoch + 1) / float(max(1, num_warmup_epochs))
        progress = float(current_epoch - num_warmup_epochs) / float(max(1, num_total_epochs - num_warmup_epochs))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

def compute_class_weights(train_data: list, num_classes: int = 6, device: str = "cpu") -> torch.Tensor:
    """Computes balanced inverse-frequency weights to penalize majority class collapse."""
    counts = Counter([item["target"] for item in train_data])
    total = len(train_data)
    weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 0)
        w = total / (num_classes * cnt) if cnt > 0 else 1.0
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32, device=device)

def train_eval_model(model, train_data, val_data, model_type, epochs, lr, wd, device, warmup_epochs=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, epochs)
    
    # Class-weighted CrossEntropy to avoid majority collapse in structure-only models
    class_weights = compute_class_weights(train_data, num_classes=6, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    best_val_f1 = -1.0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        shuffled = train_data.copy()
        random.shuffle(shuffled)
        
        for item in shuffled:
            target = torch.tensor([item["target"]], dtype=torch.long, device=device)
            optimizer.zero_grad()
            if model_type == "seq":
                out = model(item["pooled"].unsqueeze(0).to(device))
            elif model_type in ("struct", "hybrid"):
                nodes = item["nodes"].to(device)
                edge_idx = item["graph"]["edge_index"].to(device)
                edge_attr = item["graph"]["edge_attr"].to(device)
                out = model(nodes, edge_idx, edge_attr)
            
            loss = criterion(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Validation evaluation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for item in val_data:
                target = item["target"]
                if model_type == "seq":
                    out = model(item["pooled"].unsqueeze(0).to(device))
                elif model_type in ("struct", "hybrid"):
                    nodes = item["nodes"].to(device)
                    edge_idx = item["graph"]["edge_index"].to(device)
                    edge_attr = item["graph"]["edge_attr"].to(device)
                    out = model(nodes, edge_idx, edge_attr)
                pred = out.argmax(dim=-1).item()
                y_true.append(target)
                y_pred.append(pred)

        metrics = compute_classification_metrics(y_true, y_pred)
        if metrics["macro_f1"] > best_val_f1:
            best_val_f1 = metrics["macro_f1"]
            best_weights = copy.deepcopy(model.state_dict())

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model

def evaluate_test(model, test_data, model_type, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for item in test_data:
            target = item["target"]
            if model_type == "seq":
                out = model(item["pooled"].unsqueeze(0).to(device))
            elif model_type in ("struct", "hybrid"):
                nodes = item["nodes"].to(device)
                edge_idx = item["graph"]["edge_index"].to(device)
                edge_attr = item["graph"]["edge_attr"].to(device)
                out = model(nodes, edge_idx, edge_attr)
            pred = out.argmax(dim=-1).item()
            y_true.append(target)
            y_pred.append(pred)
    return compute_classification_metrics(y_true, y_pred)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/kaggle/working/protein_benchmark/configs/default_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg["experiment"]["seed"])
    device = cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu"
    print(f"[Benchmark] Execution Device: {device.upper()}")

    manifest_df = download_curated_ec_benchmark(Path(cfg["data"]["manifest_path"]))
    pdb_parser = PDBStructureParser(Path(cfg["data"]["raw_pdb_dir"]))
    
    clean_records = []
    print("[Benchmark] Parsing 3D coordinates...")
    for _, row in manifest_df.iterrows():
        res = pdb_parser.parse_chain(
            row["pdb_id"], row["chain_id"], 
            min_len=cfg["data"]["min_seq_len"], 
            max_len=cfg["data"]["max_seq_len"]
        )
        if res is not None:
            seq, coords = res
            clean_records.append({
                "pdb_id": row["pdb_id"],
                "chain_id": row["chain_id"],
                "ec_class": int(row["ec_class"]) - 1,
                "sequence": seq,
                "coords": coords
            })

    df_clean = pd.DataFrame(clean_records).reset_index(drop=True)
    print(f"[Benchmark] {len(df_clean)} non-redundant structures verified.")

    # Clustering & K-Fold Split
    splitter = SequenceClusterSplitter(cutoff=cfg["data"]["cluster_identity_cutoff"])
    id_matrix = splitter.compute_identity_matrix(df_clean["sequence"].tolist())
    folds = generate_kfold_cluster_splits(
        df_clean, id_matrix, 
        cutoff=cfg["data"]["cluster_identity_cutoff"], 
        k_folds=cfg["data"]["n_folds"], 
        seed=cfg["experiment"]["seed"]
    )

    # Feature Extraction
    esm_ext = ESMFeatureExtractor(cfg["features"]["esm_model_name"], device=device)
    per_res, pooled = esm_ext.extract_and_cache(df_clean, Path(cfg["data"]["cache_dir"]))

    # Graph Extraction with Dihedral Angles
    use_dihedrals = cfg["features"].get("use_dihedral_features", True)
    struct_node_dim = 25 if use_dihedrals else 21
    print(f"[Benchmark] Building contact graphs (Dihedrals: {use_dihedrals}, Node Dim: {struct_node_dim})...")
    
    graphs = {}
    for _, row in df_clean.iterrows():
        pid = f"{row['pdb_id']}_{row['chain_id']}"
        graphs[pid] = construct_contact_graph(
            row["sequence"], row["coords"], 
            cutoff=cfg["features"]["contact_distance_cutoff"],
            use_dihedrals=use_dihedrals
        )

    # Prepare packaged items
    samples = []
    for _, row in df_clean.iterrows():
        pid = f"{row['pdb_id']}_{row['chain_id']}"
        samples.append({
            "pid": pid,
            "target": row["ec_class"],
            "pooled": pooled[pid],
            "graph": graphs[pid],
            "nodes_struct": graphs[pid]["node_features"],
            "nodes_hybrid": per_res[pid]
        })

    fold_metrics = {"seq_f1": [], "struct_f1": [], "hybrid_f1": []}
    warmup_ep = cfg["training"].get("warmup_epochs", 5)

    print("\n" + "=" * 65)
    print("STARTING 5-FOLD CLUSTER-DISJOINT BENCHMARK (<30% Identity)")
    print("=" * 65)

    for f_info in folds:
        k = f_info["fold"]
        train_d = [samples[i] for i in f_info["train"]]
        val_d = [samples[i] for i in f_info["val"]]
        test_d = [samples[i] for i in f_info["test"]]

        max_test_sim = np.max(id_matrix[f_info["test"]][:, f_info["train"]])
        assert max_test_sim < cfg["data"]["cluster_identity_cutoff"], f"Fold {k} leakage!"

        # 1. Model A: Sequence Only
        m_a = SequenceOnlyModel(esm_dim=esm_ext.embed_dim, hidden_dim=cfg["models"]["hidden_dim"]).to(device)
        trained_a = train_eval_model(
            m_a, train_d, val_d, "seq", 
            cfg["training"]["epochs"], cfg["training"]["lr_seq"], cfg["training"]["weight_decay"], 
            device, warmup_epochs=warmup_ep
        )
        res_a = evaluate_test(trained_a, test_d, "seq", device)
        fold_metrics["seq_f1"].append(res_a["macro_f1"])

        # 2. Model B: Structure Only (with Dihedrals & Warmup)
        for s in train_d + val_d + test_d: s["nodes"] = s["nodes_struct"]
        m_b = ProteinGraphModel(input_dim=struct_node_dim, hidden_dim=cfg["models"]["hidden_dim"]).to(device)
        trained_b = train_eval_model(
            m_b, train_d, val_d, "struct", 
            cfg["training"]["epochs"], cfg["training"]["lr_struct"], cfg["training"]["weight_decay"], 
            device, warmup_epochs=warmup_ep
        )
        res_b = evaluate_test(trained_b, test_d, "struct", device)
        fold_metrics["struct_f1"].append(res_b["macro_f1"])

        # 3. Model C: Hybrid
        for s in train_d + val_d + test_d: s["nodes"] = s["nodes_hybrid"]
        m_c = ProteinGraphModel(input_dim=esm_ext.embed_dim, hidden_dim=cfg["models"]["hidden_dim"]).to(device)
        trained_c = train_eval_model(
            m_c, train_d, val_d, "hybrid", 
            cfg["training"]["epochs"], cfg["training"]["lr_hybrid"], cfg["training"]["weight_decay"], 
            device, warmup_epochs=warmup_ep
        )
        res_c = evaluate_test(trained_c, test_d, "hybrid", device)
        fold_metrics["hybrid_f1"].append(res_c["macro_f1"])

        print(f"Fold {k+1}/5 | Test Macro-F1 -> Seq: {res_a['macro_f1']:.3f} | Struct: {res_b['macro_f1']:.3f} | Hybrid: {res_c['macro_f1']:.3f}")

    print("\n" + "=" * 65)
    print("FINAL 5-FOLD CROSS-VALIDATION RESULTS (Mean ± Std):")
    print("=" * 65)
    print(f"Model A (Sequence Only)  : Macro-F1 = {np.mean(fold_metrics['seq_f1']):.3f} ± {np.std(fold_metrics['seq_f1']):.3f}")
    print(f"Model B (Structure Only) : Macro-F1 = {np.mean(fold_metrics['struct_f1']):.3f} ± {np.std(fold_metrics['struct_f1']):.3f}")
    print(f"Model C (Seq-Struct Hyb) : Macro-F1 = {np.mean(fold_metrics['hybrid_f1']):.3f} ± {np.std(fold_metrics['hybrid_f1']):.3f}")
    print("=" * 65)

if __name__ == "__main__":
    main()
