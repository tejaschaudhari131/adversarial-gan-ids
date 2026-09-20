"""Train or load any registered IDS model."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from adv_ids.models.mlp import MLPIDS
from adv_ids.models.registry import build_ids
from adv_ids.utils.device import get_device
from adv_ids.utils.io import write_json

logger = logging.getLogger(__name__)


def train_ids_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 12,
    batch_size: int = 256,
    lr: float = 1e-3,
    artifacts_dir: str | Path = "artifacts",
    model_name: str = "mlp",
    **kwargs,
) -> dict:
    """Fit a named IDS and persist it under ``artifacts_dir``."""
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    model = build_ids(model_name, input_dim=X_train.shape[1])
    result = model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        artifacts_dir=artifacts,
        **kwargs,
    )
    # Keep the historical MLP filename so the original evaluate path still works.
    if model.name == "mlp" and model.differentiable:
        legacy = artifacts / "ids_model.pt"
        model.save(legacy)
        write_json(
            artifacts / "ids_train_metrics.json",
            {
                "best_val_acc": result.get("best_val_acc"),
                "arch": model.name,
                "checkpoint": str(legacy),
            },
        )
    return result


def load_ids(artifacts_dir: str | Path = "artifacts", device: torch.device | None = None, model_name: str = "mlp"):
    """Load a checkpoint. MLP also accepts the legacy ``ids_model.pt`` filename."""
    device = device or get_device()
    artifacts = Path(artifacts_dir)
    if model_name in {"mlp", "ids"}:
        legacy = artifacts / "ids_model.pt"
        if legacy.is_file():
            return MLPIDS.load(legacy, device=device).module
        named = artifacts / "mlp.pt"
        if named.is_file():
            return MLPIDS.load(named, device=device).module
    wrapper = build_ids(model_name, input_dim=1)
    suffix = ".joblib" if not wrapper.differentiable else ".pt"
    path = artifacts / f"{wrapper.name}{suffix}"
    loaded = type(wrapper).load(path, device=device) if wrapper.differentiable else type(wrapper).load(path)
    return loaded


def load_ids_wrapper(
    artifacts_dir: str | Path = "artifacts",
    model_name: str = "mlp",
    device: torch.device | None = None,
):
    device = device or get_device()
    artifacts = Path(artifacts_dir)
    if model_name in {"mlp", "ids"}:
        for candidate in (artifacts / "ids_model.pt", artifacts / "mlp.pt"):
            if candidate.is_file():
                return MLPIDS.load(candidate, device=device)
    probe = build_ids(model_name, input_dim=1)
    suffix = ".joblib" if not probe.differentiable else ".pt"
    return type(probe).load(artifacts / f"{probe.name}{suffix}", device=device)
