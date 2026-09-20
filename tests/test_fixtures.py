"""Loaders and prepare() against tiny official-schema fixtures."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adv_ids.data.loaders import infer_dataset_name, load_csv_table
from adv_ids.data.preprocess import official_unsw_split_paths, prepare_dataset, prepare_from_frame
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


def test_official_unsw_split_uses_both_files(tmp_path):
    raw = tmp_path / "unsw-nb15"
    raw.mkdir()
    src = FIXTURES / "unsw_nb15_official_sample.csv"
    shutil.copy(src, raw / "UNSW_NB15_training-set.csv")
    shutil.copy(src, raw / "UNSW_NB15_testing-set.csv")
    assert official_unsw_split_paths(raw) is not None
    data = prepare_dataset(raw, artifacts_dir=tmp_path / "art", dataset_name="unsw_nb15", val_size=0.0)
    assert data["meta"]["official_split"] is True
    assert len(data["X_test"]) == 12
    assert len(data["X_train"]) == 12


def test_engelen_headers_infer_cicids2017():
    df = pd.DataFrame(
        {
            "Dst Port": [80, 443, 80, 22, 80, 443],
            "Total Fwd Packet": [4, 6, 20, 8, 18, 5],
            "FWD Init Win Bytes": [100, 110, 200, 90, 180, 95],
            "Flow Duration": [10, 12, 80, 15, 90, 11],
            "Label": ["BENIGN", "BENIGN", "DDoS", "BENIGN", "DDoS", "BENIGN"],
        }
    )
    assert infer_dataset_name(df) == "cicids2017"
    bundle = prepare_from_frame(df, dataset_name="cicids2017", val_size=0.0, test_size=0.25)
    assert "Destination Port" in bundle.feature_names
    assert "Total Fwd Packets" in bundle.feature_names
    assert "Init_Win_bytes_forward" in bundle.feature_names
