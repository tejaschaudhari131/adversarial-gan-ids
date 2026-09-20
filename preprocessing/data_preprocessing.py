"""Compatibility wrapper — implementations live in ``adv_ids.data``."""

from adv_ids.data.masks import feature_mask
from adv_ids.data.preprocess import binarize_labels, load_traffic_csv, prepare_dataset
from adv_ids.data.schemas import CICIDS2017_FEATURES, FROZEN_FEATURES_COMPAT as FROZEN_FEATURES
from adv_ids.data.synthetic import generate_synthetic_cicids2017

__all__ = [
    "CICIDS2017_FEATURES",
    "FROZEN_FEATURES",
    "generate_synthetic_cicids2017",
    "load_traffic_csv",
    "binarize_labels",
    "feature_mask",
    "prepare_dataset",
]
