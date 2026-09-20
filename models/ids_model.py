"""Deep learning IDS: binary MLP that scores P(attack)."""

from __future__ import annotations

import torch
from torch import nn


class IDSNet(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (128, 64), dropout: float = 0.35):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def attack_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def build_ids_model(input_dim: int, **kwargs) -> IDSNet:
    return IDSNet(input_dim=input_dim, **kwargs)
