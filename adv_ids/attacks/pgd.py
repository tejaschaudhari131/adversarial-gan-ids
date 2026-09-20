"""Constrained projected gradient descent for IDS evasion."""

from __future__ import annotations

import numpy as np
import torch

from adv_ids.attacks.constraints import apply_feature_constraints, project_linf
from adv_ids.models.base import IDSModel
from adv_ids.utils.device import get_device


def pgd_attack(
    model: IDSModel,
    X: np.ndarray,
    mask: np.ndarray,
    *,
    eps: float = 0.15,
    step_size: float | None = None,
    steps: int = 10,
    random_start: bool = True,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    batch_size: int = 512,
) -> np.ndarray:
    """ID-PGD in L_inf with a feature mask (Madry-style, targeted to benign)."""
    if not model.differentiable:
        raise TypeError("PGD requires a differentiable IDS; attack a surrogate instead.")
    if step_size is None:
        step_size = max(eps / max(steps, 1), 1e-4)
    device = getattr(model, "device", None) or get_device()
    model.module.eval()
    mask_t = torch.from_numpy(np.asarray(mask, dtype=np.float32)).to(device)
    X = np.asarray(X, dtype=np.float32)
    outs = []
    for start in range(0, len(X), batch_size):
        x0 = torch.from_numpy(X[start : start + batch_size]).to(device)
        if random_start:
            noise = torch.zeros_like(x0).uniform_(-eps, eps)
            x_adv = apply_feature_constraints(x0, x0 + noise, mask_t, clip_min, clip_max)
        else:
            x_adv = x0.clone()
        for _ in range(steps):
            x_adv = x_adv.detach().requires_grad_(True)
            logits = model.attack_logits_torch(x_adv)
            loss = torch.sigmoid(logits).mean()
            model.module.zero_grad()
            loss.backward()
            grad = x_adv.grad.detach()
            x_next = x_adv.detach() - step_size * grad.sign()
            x_adv = project_linf(x0, x_next, eps, mask_t, clip_min, clip_max)
        outs.append(x_adv.detach().cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)
