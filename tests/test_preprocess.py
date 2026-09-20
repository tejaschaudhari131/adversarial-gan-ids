"""Unit tests for cleaning, labels, and seeded splits."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adv_ids.data.preprocess import binarize_labels, load_traffic_table, prepare_from_frame
from adv_ids.data.synthetic import generate_synthetic, generate_synthetic_cicids2017


def test_inf_nan_dropped(tmp_path):
    df = generate_synthetic_cicids2017(n_benign=30, n_attack=20, seed=2)
    df.loc[0, "Flow Bytes/s"] = np.inf
    df.loc[1, "Flow Packets/s"] = np.nan
    path = tmp_path / "dirty.csv"
    df.to_csv(path, index=False)
    cleaned = load_traffic_table(path)
    assert np.isfinite(cleaned.select_dtypes(include=[np.number]).to_numpy()).all()
    assert len(cleaned) == len(df) - 2


def test_binary_labels_benign_and_numeric():
    s = pd.Series(["BENIGN", "DoS Hulk", "Benign", "0", "1", "Normal"])
    y = binarize_labels(s)
    assert list(y) == [0, 1, 0, 0, 1, 0]


def test_seeded_split_is_deterministic():
    df = generate_synthetic_cicids2017(n_benign=200, n_attack=100, seed=3)
    a = prepare_from_frame(df, dataset_name="cicids2017", random_state=7, val_size=0.1)
    b = prepare_from_frame(df, dataset_name="cicids2017", random_state=7, val_size=0.1)
    np.testing.assert_array_equal(a.X_train, b.X_train)
    np.testing.assert_array_equal(a.y_test, b.y_test)
    assert len(a.X_val) > 0


def test_unsw_synthetic_multiclass_column():
    df = generate_synthetic("unsw_nb15", n_benign=40, n_attack=20, seed=4)
    bundle = prepare_from_frame(df, dataset_name="unsw_nb15", val_size=0.1, label_mode="binary")
    assert bundle.dataset_name == "unsw_nb15"
    assert bundle.y_multi_train is not None
    assert bundle.X_train.min() >= -1e-6
    assert bundle.X_train.max() <= 1.0 + 1e-6


def test_scaled_unit_cube_and_stratify():
    df = generate_synthetic_cicids2017(n_benign=120, n_attack=80, seed=5)
    bundle = prepare_from_frame(df, val_size=0.15, random_state=5)
    assert 0.2 < bundle.y_train.mean() < 0.6
    assert bundle.X_test.shape[1] == 77
