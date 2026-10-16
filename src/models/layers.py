import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    """Standard 2-layer MLP classifier head with LayerNorm and Dropout."""
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_classes: int = 6, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class EdgeConditionedGNNLayer(nn.Module):
    """
    Message passing layer updating node representations using neighbor nodes and edge attributes.
    m_{j->i} = Linear_n(h_j) + Linear_e(e_{ij})
    h_i' = LayerNorm(h_i + Aggregate(m_{j->i}))
    """
    def __init__(self, node_dim: int, edge_dim: int = 2):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, node_dim, bias=False)
        self.edge_proj = nn.Linear(edge_dim, node_dim, bias=False)
        self.update_gate = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        messages = F.relu(self.node_proj(h[src]) + self.edge_proj(edge_attr))
        aggregated = torch.zeros(h.size(0), h.size(1), device=h.device, dtype=h.dtype)
        aggregated.index_add_(0, dst, messages)
        update_input = torch.cat([h, aggregated], dim=-1)
        return self.norm(h + self.update_gate(update_input))
