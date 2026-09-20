"""Discriminator: real benign traffic vs generator output."""

from __future__ import annotations

import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_discriminator(input_dim: int, hidden: int = 256) -> Discriminator:
    return Discriminator(input_dim=input_dim, hidden=hidden)
