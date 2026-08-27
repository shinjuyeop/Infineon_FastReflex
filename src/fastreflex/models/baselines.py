"""Small raw-IMU baseline models for bounded classification experiments."""

from __future__ import annotations

import torch
from torch import nn


class MLPBaseline(nn.Module):
    """Flattened causal window followed by two small hidden layers."""

    def __init__(self, window_samples: int, input_channels: int = 6) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window_samples * input_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class GRUBaseline(nn.Module):
    """One-layer unidirectional GRU over raw causal IMU samples."""

    def __init__(self, input_channels: int = 6) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )
        self.classifier = nn.Linear(32, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(inputs)
        return self.classifier(hidden[-1])


def build_model(family: str, window_samples: int) -> nn.Module:
    if family == "mlp":
        return MLPBaseline(window_samples)
    if family == "gru":
        return GRUBaseline()
    raise ValueError(f"unsupported model family: {family}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
