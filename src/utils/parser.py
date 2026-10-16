import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple
import Bio.PDB
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import numpy as np

warnings.simplefilter("ignore", PDBConstructionWarning)

AA_3_TO_1: Dict[str, str] = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E",
    "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I",
    "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S",
    "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

class PDBStructureParser:
    """Downloads and extracts resolved C-alpha coordinates from PDB files."""
    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.parser = Bio.PDB.PDBParser(QUIET=True)
        self.downloader = Bio.PDB.PDBList(verbose=False)

    def parse_chain(
        self, pdb_id: str, chain_id: str, min_len: int = 30, max_len: int = 512
    ) -> Optional[Tuple[str, np.ndarray]]:
        pdb_id = pdb_id.lower()
        try:
            pdb_file = self.downloader.retrieve_pdb_file(
                pdb_id, pdir=str(self.storage_dir), file_format="pdb"
            )
        except Exception:
            return None

        if not os.path.exists(pdb_file):
            alt_path = self.storage_dir / f"pdb{pdb_id}.ent"
            if alt_path.exists():
                pdb_file = str(alt_path)
            else:
                return None

        try:
            structure = self.parser.get_structure(pdb_id, pdb_file)
            model = structure[0]
            if chain_id not in model:
                return None

            target_chain = model[chain_id]
            seq_letters, ca_coords = [], []

            for residue in target_chain:
                res_name = residue.get_resname().strip()
                if res_name in AA_3_TO_1 and "CA" in residue:
                    coords = residue["CA"].get_coord()
                    if np.isnan(coords).any() or np.isinf(coords).any():
                        continue
                    seq_letters.append(AA_3_TO_1[res_name])
                    ca_coords.append(coords)

            if len(seq_letters) < min_len or len(seq_letters) > max_len:
                return None

            return "".join(seq_letters), np.array(ca_coords, dtype=np.float32)
        except Exception:
            return None
