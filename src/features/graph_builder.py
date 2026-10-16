from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F

AA_TO_INDEX = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

def compute_dihedrals(coords: np.ndarray) -> torch.Tensor:
    """
    Computes pseudodihedral approximations or backbone torsions from C-alpha coordinates.
    For C-alpha only traces, computes the pseudo-torsion angle between 4 consecutive C-alphas
    (i-1, i, i+1, i+2) parameterized as [sin(tau), cos(tau)] and pseudo-bond angles [sin(theta), cos(theta)].
    
    Returns:
        torch.Tensor of shape (L, 4)
    """
    L = coords.shape[0]
    feats = np.zeros((L, 4), dtype=np.float32)
    
    if L < 4:
        return torch.tensor(feats, dtype=torch.float32)

    p = coords # (L, 3)
    
    # 1. Pseudo-bond angle at residue i: angle between (p_{i-1} - p_i) and (p_{i+1} - p_i)
    for i in range(1, L - 1):
        v1 = p[i - 1] - p[i]
        v2 = p[i + 1] - p[i]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 > 1e-6 and norm2 > 1e-6:
            cosine = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
            angle = np.arccos(cosine)
            feats[i, 0] = np.sin(angle)
            feats[i, 1] = np.cos(angle)

    # 2. Pseudo-dihedral (torsion) angle for 4 consecutive C-alphas: p_{i-1}, p_i, p_{i+1}, p_{i+2}
    for i in range(1, L - 2):
        b0 = -1.0 * (p[i] - p[i - 1])
        b1 = p[i + 1] - p[i]
        b2 = p[i + 2] - p[i + 1]
        
        # Vector orthogonal to b0 and b1
        v = np.cross(b0, b1)
        # Vector orthogonal to b1 and b2
        w = np.cross(b1, b2)
        
        norm_v = np.linalg.norm(v)
        norm_w = np.linalg.norm(w)
        norm_b1 = np.linalg.norm(b1)
        
        if norm_v > 1e-6 and norm_w > 1e-6 and norm_b1 > 1e-6:
            x = np.dot(v, w) / (norm_v * norm_w)
            y = np.dot(np.cross(b1 / norm_b1, v), w) / (norm_v * norm_w)
            tau = np.arctan2(y, x)
            feats[i, 2] = np.sin(tau)
            feats[i, 3] = np.cos(tau)

    return torch.tensor(feats, dtype=torch.float32)

def construct_contact_graph(
    sequence: str, 
    coords: np.ndarray, 
    cutoff: float = 8.0, 
    use_dihedrals: bool = True
) -> Dict[str, Any]:
    L = len(sequence)
    coords_tensor = torch.tensor(coords, dtype=torch.float32)

    # Base features: One-hot amino acid (20) + normalized position (1)
    aa_indices = torch.tensor([AA_TO_INDEX.get(aa, 0) for aa in sequence], dtype=torch.long)
    one_hot = F.one_hot(aa_indices, num_classes=20).float()
    seq_pos = torch.linspace(0, 1, steps=L, dtype=torch.float32).unsqueeze(-1)
    
    if use_dihedrals:
        dihedrals = compute_dihedrals(coords) # (L, 4)
        node_features = torch.cat([one_hot, seq_pos, dihedrals], dim=-1) # (L, 25)
    else:
        node_features = torch.cat([one_hot, seq_pos], dim=-1) # (L, 21)

    # Distance contacts
    dist_matrix = torch.cdist(coords_tensor, coords_tensor, p=2.0)
    mask = (dist_matrix <= cutoff) & (~torch.eye(L, dtype=torch.bool))
    edge_index = mask.nonzero(as_tuple=False).t()

    src, dst = edge_index[0], edge_index[1]
    spatial_dist = dist_matrix[src, dst].unsqueeze(-1)
    spatial_feat = torch.exp(-0.5 * (spatial_dist / (cutoff / 2)) ** 2)
    seq_dist = (torch.abs(src - dst).float() / L).unsqueeze(-1)
    edge_attr = torch.cat([spatial_feat, seq_dist], dim=-1)

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "num_nodes": L,
    }
