from typing import Dict, List
import networkx as nx
import numpy as np
import pandas as pd
from Bio import Align

class SequenceClusterSplitter:
    """Computes pairwise sequence identity and single-linkage clusters."""
    def __init__(self, cutoff: float = 0.30):
        self.cutoff = cutoff

    def compute_identity_matrix(self, sequences: List[str]) -> np.ndarray:
        n = len(sequences)
        id_matrix = np.eye(n, dtype=np.float32)
        aligner = Align.PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 2
        aligner.mismatch_score = -1
        aligner.open_gap_score = -4
        aligner.extend_gap_score = -1

        for i in range(n):
            seq_a, len_a = sequences[i], len(sequences[i])
            for j in range(i + 1, n):
                seq_b, len_b = sequences[j], len(sequences[j])
                max_len = max(len_a, len_b)
                if (min(len_a, len_b) / max_len) < self.cutoff:
                    id_matrix[i, j] = id_matrix[j, i] = 0.10
                    continue

                alignment = aligner.align(seq_a, seq_b)[0]
                matches = sum(
                    ra == rb for ra, rb in zip(alignment[0], alignment[1]) if ra != "-" and rb != "-"
                )
                identity = matches / max_len
                id_matrix[i, j] = id_matrix[j, i] = float(identity)
        return id_matrix


def generate_kfold_cluster_splits(
    df: pd.DataFrame, 
    identity_matrix: np.ndarray, 
    cutoff: float = 0.30, 
    k_folds: int = 5, 
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Distributes single-linkage sequence clusters across k disjoint folds using greedy bin packing.
    Guarantees no sequence in fold i shares >= cutoff identity with fold j.
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

    folds = [[] for _ in range(k_folds)]
    fold_sizes = [0] * k_folds

    # Greedy bin-packing: assign largest clusters first
    sorted_clusters = sorted(clusters, key=len, reverse=True)
    for c in sorted_clusters:
        idx = int(np.argmin(fold_sizes))
        folds[idx].extend(c)
        fold_sizes[idx] += len(c)

    splits = []
    for i in range(k_folds):
        test_indices = folds[i]
        val_indices = folds[(i + 1) % k_folds]
        train_indices = [
            idx for fold_idx, f in enumerate(folds) 
            if fold_idx not in (i, (i + 1) % k_folds) 
            for idx in f
        ]

        splits.append({
            "fold": i,
            "train": train_indices,
            "val": val_indices,
            "test": test_indices
        })

    return splits
