"""Compatibility wrapper — MLP IDS lives in ``adv_ids.models.mlp``."""

from adv_ids.models.mlp import IDSNet, build_ids_model

__all__ = ["IDSNet", "build_ids_model"]
