from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from adv_ids.attacks.fgsm import fgsm_attack
from adv_ids.attacks.gan_attack import GANAttack
from adv_ids.attacks.pgd import pgd_attack
from adv_ids.models.base import IDSModel


def generate_adversarial(
    attack: str,
    model: IDSModel,
    X: np.ndarray,
    mask: np.ndarray,
    *,
    gan: GANAttack | None = None,
    **kwargs: Any,
) -> np.ndarray:
    name = attack.strip().lower()
    if name == "fgsm":
        return fgsm_attack(model, X, mask, **{k: v for k, v in kwargs.items() if k in {"eps", "batch_size"}})
    if name == "pgd":
        allowed = {"eps", "step_size", "steps", "random_start", "batch_size"}
        return pgd_attack(model, X, mask, **{k: v for k, v in kwargs.items() if k in allowed})
    if name == "gan":
        if gan is None:
            raise ValueError("GAN attack requires a fitted GANAttack instance.")
        return gan.generate(X, mask)
    raise ValueError(f"Unknown attack '{attack}'. Known: fgsm, pgd, gan")


def build_attack(name: str, feature_dim: int, **kwargs) -> GANAttack | dict:
    key = name.strip().lower()
    if key == "gan":
        return GANAttack(feature_dim=feature_dim, **kwargs)
    return {"name": key, **kwargs}


def maybe_load_gan(artifacts_dir: str | Path) -> GANAttack | None:
    path = Path(artifacts_dir) / "gan.pt"
    if path.is_file():
        return GANAttack.load(path)
    return None
