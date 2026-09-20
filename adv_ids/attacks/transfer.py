"""Black-box transfer: craft on a surrogate, evaluate on a target."""

from __future__ import annotations

from typing import Callable

import numpy as np

from adv_ids.models.base import IDSModel


def transfer_attack(
    X_attack: np.ndarray,
    craft_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Generate adversarial rows using a surrogate-side craft function."""
    return craft_fn(X_attack)


def describe_transfer(surrogate: IDSModel | str, target: IDSModel | str, attack: str) -> dict:
    s = surrogate if isinstance(surrogate, str) else surrogate.name
    t = target if isinstance(target, str) else target.name
    return {
        "surrogate": s,
        "target": t,
        "attack": attack,
        "note": (
            "Examples are crafted with gradients or a GAN trained on the surrogate, "
            "then scored by the target. Transfer is often asymmetric."
        ),
    }
