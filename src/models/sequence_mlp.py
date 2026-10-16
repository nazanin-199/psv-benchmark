import torch
import torch.nn as nn
from src.models.layers import ClassificationHead

class SequenceOnlyModel(nn.Module):
    """Model A: Global mean-pooled ESM-2 sequence baseline."""
    def __init__(self, esm_dim: int = 480, hidden_dim: int = 128, num_classes: int = 6, dropout: float = 0.2):
        super().__init__()
        self.classifier = ClassificationHead(esm_dim, hidden_dim, num_classes, dropout)

    def forward(self, pooled_seq: torch.Tensor) -> torch.Tensor:
        return self.classifier(pooled_seq)
