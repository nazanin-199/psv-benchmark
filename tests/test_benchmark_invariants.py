import numpy as np
import pytest
import torch
import pandas as pd
from src.features.graph_builder import construct_contact_graph
from src.data.clustering import SequenceClusterSplitter, generate_kfold_cluster_splits

def test_coordinate_token_alignment():
    seq = "ACDEFGHIKL"
    coords = np.random.randn(len(seq), 3).astype(np.float32)
    graph = construct_contact_graph(seq, coords, cutoff=8.0)
    
    assert graph["num_nodes"] == len(seq)
    assert graph["node_features"].shape[0] == len(seq)
    assert graph["edge_index"].shape[0] == 2

def test_zero_leakage_kfold_splits():
    seqs = [
        "ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWY",
        "WYVTSQPMLKIRHGFEDCA", "WYVTSQPMLKIRHGFEDCA",
        "AAAAAAAAAA", "YYYYYYYYYY"
    ]
    df = pd.DataFrame({"sequence": seqs, "ec_class": [0, 0, 1, 1, 2, 3]})
    splitter = SequenceClusterSplitter(cutoff=0.30)
    id_matrix = splitter.compute_identity_matrix(seqs)
    
    # Test with standard k_folds=3
    folds = generate_kfold_cluster_splits(df, id_matrix, cutoff=0.30, k_folds=3, seed=42)
    
    for fold in folds:
        train_idx = fold["train"]
        test_idx = fold["test"]
        
        assert len(train_idx) > 0, "Train partition cannot be empty."
        assert len(test_idx) > 0, "Test partition cannot be empty."
        
        max_sim = np.max(id_matrix[test_idx][:, train_idx])
        assert max_sim < 0.30, f"Leakage detected: max test similarity {max_sim:.3f} >= 0.30"
