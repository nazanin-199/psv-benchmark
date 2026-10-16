import torch
import torch.nn as nn
from src.models.layers import ClassificationHead, EdgeConditionedGNNLayer

class ProteinGraphModel(nn.Module):
    """
    Serves as:
      - Model B (Structure-Only): input_dim = 21 (or 25 with dihedrals)
      - Model C (Sequence-Structure Hybrid): input_dim = 480 (from ESM-2 residue embeddings)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        edge_dim: int = 2,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList([
            EdgeConditionedGNNLayer(hidden_dim, edge_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = ClassificationHead(hidden_dim, hidden_dim, num_classes, dropout)

    def forward(self, nodes: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.node_encoder(nodes)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
            h = self.dropout(h)
        graph_repr = h.mean(dim=0, keepdim=True)
        return self.classifier(graph_repr)
