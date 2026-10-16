from typing import List, Dict
import networkx as nx
import numpy as np
import pandas as pd

def generate_kfold_cluster_splits(
    df: pd.DataFrame, 
    identity_matrix: np.ndarray, 
    cutoff: float = 0.30, 
    k_folds: int = 5, 
    seed: int = 42
) -> List[Dict[str, List[int]]]:
    """
    Generates k cluster-disjoint folds.
    Guarantees that no sequence in fold i shares >= cutoff identity with any sequence in fold j.
    """
    n = len(df)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if identity_matrix[i, j] >= cutoff:
                g.add_edge(i, j)
                
    clusters = [list(c) for c in nx.connected_components(g)]
    rng = np.random.RandomState(seed)
    rng.shuffle(clusters)
    
    # Distribute clusters across k folds using greedy bin packing
    folds = [[] for _ in range(k_folds)]
    fold_sizes = [0] * k_folds
    
    # Sort clusters descending
    sorted_clusters = sorted(clusters, key=len, reverse=True)
    for c in sorted_clusters:
        idx = int(np.argmin(fold_sizes))
        folds[idx].extend(c)
        fold_sizes[idx] += len(c)
        
    splits = []
    for i in range(k_folds):
        test_indices = folds[i]
        val_indices = folds[(i + 1) % k_folds]
        train_indices = [idx for fold_idx, f in enumerate(folds) if fold_idx not in (i, (i + 1) % k_folds) for idx in f]
        
        splits.append({
            "fold": i,
            "train": train_indices,
            "val": val_indices,
            "test": test_indices
        })
        
    return splits
