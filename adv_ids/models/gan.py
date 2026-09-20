"""Conditional generator and discriminator for budgeted flow perturbations."""

from __future__ import annotations

import torch
from torch import nn


class Generator(nn.Module):
    """G(x_attack, z) produces a perturbation in [-eps, eps] on every feature."""

    def __init__(self, feature_dim: int, latent_dim: int = 32, hidden: int = 256, eps: float = 0.25):
        super().__init__()
        self.eps = eps
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim + latent_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden // 2, feature_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, z], dim=1)) * self.eps


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


def apply_perturbation(
    x: torch.Tensor,
    delta: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Add a masked perturbation and clip back to the scaled [0, 1] cube."""
    if mask is not None:
        delta = delta * mask.view(1, -1)
    return torch.clamp(x + delta, 0.0, 1.0)


def build_generator(feature_dim: int, latent_dim: int = 32, eps: float = 0.25) -> Generator:
    return Generator(feature_dim=feature_dim, latent_dim=latent_dim, eps=eps)


def build_discriminator(input_dim: int, hidden: int = 256) -> Discriminator:
    return Discriminator(input_dim=input_dim, hidden=hidden)
