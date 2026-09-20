"""Gradient-attack constraint tests on a tiny differentiable model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adv_ids.attacks.fgsm import fgsm_attack
from adv_ids.attacks.pgd import pgd_attack
from adv_ids.models.mlp import MLPIDS


class _LinearIDS(MLPIDS):
    """Single linear unit so FGSM/PGD have a well-defined gradient."""

    def build_module(self, input_dim: int) -> nn.Module:
        return nn.Linear(input_dim, 1, bias=False)


def _toy_model(dim: int = 6) -> MLPIDS:
    model = _LinearIDS(input_dim=dim, device=torch.device("cpu"))
    with torch.no_grad():
        model.module.weight.copy_(torch.ones(1, dim))
    model.module.eval()
    return model


def test_fgsm_respects_mask_and_eps():
    model = _toy_model()
    x = np.full((5, 6), 0.5, dtype=np.float32)
    mask = np.array([0, 1, 1, 0, 1, 0], dtype=np.float32)
    x_adv = fgsm_attack(model, x, mask, eps=0.1)
    np.testing.assert_allclose(x_adv[:, mask < 0.5], x[:, mask < 0.5], atol=1e-6)
    assert np.max(np.abs(x_adv - x)) <= 0.1 + 1e-5
    # Gradient of sum(x) is positive, so evasion steps *down* on open features.
    assert np.all(x_adv[:, mask > 0.5] < x[:, mask > 0.5])


def test_pgd_stays_in_ball_and_unit_cube():
    model = _toy_model()
    x = np.full((4, 6), 0.5, dtype=np.float32)
    mask = np.ones(6, dtype=np.float32)
    x_adv = pgd_attack(model, x, mask, eps=0.12, steps=5, step_size=0.05, random_start=False)
    assert x_adv.min() >= 0.0 - 1e-6
    assert x_adv.max() <= 1.0 + 1e-6
    assert np.max(np.abs(x_adv - x)) <= 0.12 + 1e-5
