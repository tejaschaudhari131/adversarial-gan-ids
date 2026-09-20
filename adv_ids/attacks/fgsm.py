"""Constrained Fast Gradient Sign Method for IDS evasion (targeted to benign)."""

from __future__ import annotations

import numpy as np
import torch

from adv_ids.attacks.constraints import apply_feature_constraints
from adv_ids.models.base import IDSModel
from adv_ids.utils.device import get_device


def fgsm_attack(
    model: IDSModel,
    X: np.ndarray,
    mask: np.ndarray,
    *,
    eps: float = 0.15,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    batch_size: int = 512,
) -> np.ndarray:
    """Decrease P(attack) with a single signed gradient step on modifiable features."""
    if not model.differentiable:
        raise TypeError("FGSM requires a differentiable IDS; attack a surrogate instead.")
    device = getattr(model, "device", None) or get_device()
    model.module.eval()
    mask_t = torch.from_numpy(np.asarray(mask, dtype=np.float32)).to(device)
    outs = []
    X = np.asarray(X, dtype=np.float32)
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start : start + batch_size]).to(device)
        xb.requires_grad_(True)
        logits = model.attack_logits_torch(xb)
        # Minimize P(attack) = sigmoid(logit).
        loss = torch.sigmoid(logits).mean()
        model.module.zero_grad()
        if xb.grad is not None:
            xb.grad.zero_()
        loss.backward()
        grad = xb.grad.detach()
        # Step opposite the attack-score gradient.
        x_adv = xb.detach() - eps * grad.sign()
        x_adv = apply_feature_constraints(xb.detach(), x_adv, mask_t, clip_min, clip_max)
        outs.append(x_adv.cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)
