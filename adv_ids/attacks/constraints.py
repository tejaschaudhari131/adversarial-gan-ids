"""Domain constraints: freeze protocol features and stay inside valid ranges."""

from __future__ import annotations

import numpy as np
import torch


def apply_feature_constraints(
    x: np.ndarray | torch.Tensor,
    x_adv: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
):
    """Keep frozen features equal to ``x`` and clip the rest to ``[clip_min, clip_max]``."""
    if torch.is_tensor(x):
        mask_t = mask if torch.is_tensor(mask) else torch.as_tensor(mask, device=x.device, dtype=x.dtype)
        if mask_t.ndim == 1:
            mask_t = mask_t.view(1, -1)
        out = x + (x_adv - x) * mask_t
        return torch.clamp(out, clip_min, clip_max)
    mask_np = np.asarray(mask, dtype=np.float32)
    out = x + (x_adv - x) * mask_np.reshape(1, -1)
    return np.clip(out, clip_min, clip_max).astype(np.float32)


def project_linf(
    x: np.ndarray | torch.Tensor,
    x_adv: np.ndarray | torch.Tensor,
    eps: float,
    mask: np.ndarray | torch.Tensor,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
):
    """Project onto an L_inf ball, then apply the feature mask and box constraints."""
    if torch.is_tensor(x):
        delta = torch.clamp(x_adv - x, -eps, eps)
        return apply_feature_constraints(x, x + delta, mask, clip_min, clip_max)
    delta = np.clip(x_adv - x, -eps, eps)
    return apply_feature_constraints(x, x + delta, mask, clip_min, clip_max)
