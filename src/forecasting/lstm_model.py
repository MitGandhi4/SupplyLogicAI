"""PyTorch LSTM for univariate demand forecasting."""
from __future__ import annotations

import torch
from torch import nn


class DemandLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 30,
    ):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, forecast_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, input_size)
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)
