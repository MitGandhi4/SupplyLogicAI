"""
Graph neural network for critical failure-point detection and disruption
propagation prediction, built with PyTorch Geometric.

Architecture: a shared GAT (graph attention) encoder produces node
embeddings, feeding two heads:
  - `critical_head`: binary classification — is this node a structural
    critical failure point?
  - `impact_head`: regression — predicted fraction of network capacity lost
    if this node is disrupted (propagation severity).
  - `edge_impact_head`: regression — predicted disruption propagation impact
    along each directed supply-chain edge.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv


class SupplyChainGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        heads: int = 4,
        dropout: float = 0.3,
        edge_channels: int = 4,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.edge_channels = edge_channels

        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout, edge_dim=edge_channels)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, edge_dim=edge_channels)
        self.conv3 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout, edge_dim=edge_channels)

        self.critical_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )
        self.impact_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )
        self.edge_impact_head = nn.Sequential(
            nn.Linear(hidden_channels * 2 + edge_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        if edge_attr is None:
            edge_attr = x.new_zeros((edge_index.shape[1], self.edge_channels))
        h = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.conv2(h, edge_index, edge_attr=edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv3(h, edge_index, edge_attr=edge_attr)
        return h

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encode(x, edge_index, edge_attr)
        critical_logits = self.critical_head(embeddings).squeeze(-1)
        impact_score = torch.sigmoid(self.impact_head(embeddings)).squeeze(-1)
        return critical_logits, impact_score

    def predict_edge_impact(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Estimate normalized disruption propagation impact for each edge."""
        embeddings = self.encode(x, edge_index, edge_attr)
        if edge_attr is None:
            edge_attr = embeddings.new_zeros((edge_index.shape[1], self.edge_channels))
        source, target = edge_index
        edge_embeddings = torch.cat((embeddings[source], embeddings[target], edge_attr), dim=-1)
        return torch.sigmoid(self.edge_impact_head(edge_embeddings)).squeeze(-1)

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        critical_logits, impact_score = self.forward(x, edge_index, edge_attr)
        critical_prob = torch.sigmoid(critical_logits)
        return critical_prob, impact_score
