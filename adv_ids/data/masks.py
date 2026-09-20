"""Modifiable-feature masks that freeze protocol-semantic columns."""

from __future__ import annotations

import numpy as np

from adv_ids.data.schemas import DatasetSpec, get_spec


def frozen_feature_names(dataset: str | DatasetSpec | None = "cicids2017") -> set[str]:
    spec = dataset if isinstance(dataset, DatasetSpec) else get_spec(dataset)
    return set(spec.frozen)


def feature_mask(feature_names, dataset: str | DatasetSpec | None = "cicids2017") -> np.ndarray:
    """Return a float32 mask of shape (n_features,) with 0 on frozen columns."""
    frozen = frozen_feature_names(dataset)
    return np.array([0.0 if name in frozen else 1.0 for name in feature_names], dtype=np.float32)


def assert_mask_respects_frozen(x: np.ndarray, x_adv: np.ndarray, mask: np.ndarray, atol: float = 1e-6) -> None:
    frozen = mask < 0.5
    if frozen.any() and not np.allclose(x[:, frozen], x_adv[:, frozen], atol=atol):
        raise AssertionError("Frozen features changed under the attack mask.")
