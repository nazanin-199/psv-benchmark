import numpy as np
import pytest
import torch
import pandas as pd
from src.features.graph_builder import construct_contact_graph
from src.data.clustering import SequenceClusterSplitter, generate_kfold_cluster_splits


def test_coordinate_token_alignment():
    """Verify that node features and physical coordinates match sequence length exactly."""
    seq = "ACDEFGHIKL"
    coords = np.random.randn(len(seq), 3).astype(np.float32)
    graph = construct_contact_graph(seq, coords, cutoff=8.0)
    
    assert graph["num_nodes"] == len(seq)
    assert graph["node_features"].shape[0] == len(seq)
    assert graph["edge_index"].shape[0] == 2


def test_zero_leakage_kfold_splits():
    """Ensure that cross-validation folds strictly enforce cluster-disjoint boundaries."""
    # 6 sequences with two distinct homologous pairs and two singletons
    seqs = [
        "ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWY",  # Pair 1 (100% id)
        "WYVTSQPMLKIRHGFEDCA", "WYVTSQPMLKIRHGFEDCA",    # Pair 2 (100% id)
        "AAAAAAAAAA", "YYYYYYYYYY"                       # Singletons
    ]
    df = pd.DataFrame({"sequence": seqs, "ec_class": [0, 0, 1, 1, 2, 3]})
    
    splitter = SequenceClusterSplitter(cutoff=0.30)
    id_matrix = splitter.compute_identity_matrix(seqs)
    folds = generate_kfold_cluster_splits(df, id_matrix, cutoff=0.30, k_folds=2, seed=42)
    
    for fold in folds:
        train_idx = fold["train"]
        test_idx = fold["test"]
        max_sim = np.max(id_matrix[test_idx][:, train_idx])
        assert max_sim < 0.30, f"Cross-fold leakage detected: {max_sim:.3f} >= 0.30"
