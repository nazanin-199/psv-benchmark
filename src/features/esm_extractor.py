from pathlib import Path
from typing import Dict, Tuple
import esm
import pandas as pd
import torch
from tqdm.auto import tqdm

class ESMFeatureExtractor:
    """Extracts and stores frozen ESM-2 per-residue and mean-pooled representations."""
    def __init__(self, model_name: str = "esm2_t12_35M_UR50D", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.to(self.device).eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.embed_dim = self.model.embed_dim

    def extract_and_cache(
        self, df: pd.DataFrame, cache_dir: Path
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        res_cache_file = cache_dir / "per_residue_esm.pt"
        pool_cache_file = cache_dir / "pooled_esm.pt"

        if res_cache_file.exists() and pool_cache_file.exists():
            return torch.load(res_cache_file), torch.load(pool_cache_file)

        per_residue, pooled = {}, {}
        with torch.no_grad():
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting ESM-2"):
                pid = f"{row['pdb_id']}_{row['chain_id']}"
                seq = row["sequence"]
                _, _, tokens = self.batch_converter([(pid, seq)])
                tokens = tokens.to(self.device)

                results = self.model(tokens, repr_layers=[self.model.num_layers], return_contacts=False)
                reps = results["representations"][self.model.num_layers]
                res_repr = reps[0, 1 : len(seq) + 1].cpu()
                mean_repr = res_repr.mean(dim=0)

                per_residue[pid] = res_repr
                pooled[pid] = mean_repr

        torch.save(per_residue, res_cache_file)
        torch.save(pooled, pool_cache_file)
        return per_residue, pooled
