from __future__ import annotations

from adv_ids.models.base import IDSModel
from adv_ids.models.cnn1d import CNN1DIDS
from adv_ids.models.mlp import DeepMLPIDS, MLPIDS
from adv_ids.models.random_forest import RandomForestIDS

IDS_REGISTRY: dict[str, type[IDSModel]] = {
    "mlp": MLPIDS,
    "deep_mlp": DeepMLPIDS,
    "cnn1d": CNN1DIDS,
    "random_forest": RandomForestIDS,
    "rf": RandomForestIDS,
}


def list_ids_models() -> list[str]:
    return sorted({k for k in IDS_REGISTRY if k != "rf"})


def build_ids(name: str, input_dim: int, **kwargs) -> IDSModel:
    key = name.strip().lower()
    if key not in IDS_REGISTRY:
        raise ValueError(f"Unknown IDS model '{name}'. Known: {list_ids_models()}")
    return IDS_REGISTRY[key](input_dim=input_dim, **kwargs)
