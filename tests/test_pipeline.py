"""Smoke tests for preprocessing and a tiny end-to-end pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adv_ids.data.preprocess import prepare_dataset
from adv_ids.data.synthetic import generate_synthetic_cicids2017


def test_synthetic_schema():
    df = generate_synthetic_cicids2017(n_benign=20, n_attack=10, seed=0)
    assert "Label" in df.columns
    assert df.shape[1] == 79
    assert (df["Label"] == "BENIGN").sum() == 20


def test_prepare_roundtrip(tmp_path):
    csv_path = tmp_path / "tiny.csv"
    generate_synthetic_cicids2017(n_benign=80, n_attack=40, seed=1, output_path=csv_path)
    data = prepare_dataset(csv_path, artifacts_dir=tmp_path / "art")
    # Duplicate "Fwd Header Length.1" is dropped during cleaning (77 numeric features).
    assert data["X_train"].shape[1] == 77
    assert set(data["y_train"]).issubset({0, 1})
    assert data["modifiable_mask"].shape == (78,)
    assert (tmp_path / "art" / "scaler.joblib").is_file()
