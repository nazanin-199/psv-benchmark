# src/features/graph_builder.py (Enhanced Geometric Node Features)
import torch
import numpy as np
import Bio.PDB

def compute_backbone_torsions(chain: Bio.PDB.Chain.Chain) -> np.ndarray:
    """
    Calculates sine and cosine of backbone dihedral angles (phi, psi) for each residue.
    Returns array of shape (L, 4): [sin(phi), cos(phi), sin(psi), cos(psi)].
    """
    polypeptides = Bio.PDB.PPBuilder().build_peptides(chain)
    if not polypeptides:
        return np.zeros((len(chain), 4), dtype=np.float32)
        
    pp = polypeptides[0]
    phi_psi = pp.get_phi_psi_list()
    
    torsion_feats = []
    for phi, psi in phi_psi:
        phi_val = phi if phi is not None else 0.0
        psi_val = psi if psi is not None else 0.0
        torsion_feats.append([
            np.sin(phi_val), np.cos(phi_val),
            np.sin(psi_val), np.cos(psi_val)
        ])
    return np.array(torsion_feats, dtype=np.float32)
