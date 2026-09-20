"""Loaders and prepare() against tiny official-schema fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adv_ids.data.loaders import infer_dataset_name, load_csv_table
from adv_ids.data.preprocess import prepare_dataset, prepare_from_frame
from adv_ids.data.schemas import CICIDS2017_FEATURES, CICIDS2018_FEATURES, UNSW_NB15_NUMERIC

FIXTURES = ROOT / "tests" / "fixtures"


def test_cicids2017_fixture_headers_and_prepare(tmp_path):
    path = FIXTURES / "cicids2017_official_sample.csv"
    raw = path.read_text(encoding="utf-8").splitlines()[0]
    assert " Destination Port" in raw
    assert raw.count("Fwd Header Length") == 2
    df = load_csv_table(path)
    assert infer_dataset_name(df) == "cicids2017"
    assert "Destination Port" in df.columns
    assert "Fwd Header Length.1" in df.columns
    assert "Label" in df.columns
    data = prepare_dataset(path, artifacts_dir=tmp_path, dataset_name="cicids2017", val_size=0.0, test_size=0.25)
    assert data["X_train"].shape[1] == 77
    assert set(data["y_train"]).issubset({0, 1})
    assert 1 in set(data["y_train"]) | set(data["y_test"])
    for name in ("Destination Port", "Flow Duration", "SYN Flag Count"):
        assert name in CICIDS2017_FEATURES


def test_cicids2018_fixture_timestamp_dropped(tmp_path):
    path = FIXTURES / "cicids2018_official_sample.csv"
    df = load_csv_table(path)
    assert infer_dataset_name(df) == "cicids2018"
    assert "Dst Port" in df.columns
    assert "Timestamp" in df.columns
    bundle = prepare_from_frame(df, dataset_name="cicids2018", artifacts_dir=tmp_path, val_size=0.0, test_size=0.25)
    assert "Timestamp" not in bundle.feature_names
    assert "Dst Port" in bundle.feature_names
    assert bundle.X_train.shape[1] == len(CICIDS2018_FEATURES)
    assert bundle.dataset_name == "cicids2018"


def test_unsw_fixture_official_columns(tmp_path):
    path = FIXTURES / "unsw_nb15_official_sample.csv"
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[0] == "id"
    assert header[-2] == "attack_cat"
    assert header[-1].strip() == "label"
    df = load_csv_table(path)
    assert infer_dataset_name(df) == "unsw_nb15"
    bundle = prepare_from_frame(df, dataset_name="unsw_nb15", artifacts_dir=tmp_path, val_size=0.0, test_size=0.25)
    assert bundle.dataset_name == "unsw_nb15"
    assert "id" not in bundle.feature_names
    for col in UNSW_NB15_NUMERIC:
        assert col in bundle.feature_names
    assert bundle.y_multi_train is not None
    assert set(bundle.y_train).issubset({0, 1})
