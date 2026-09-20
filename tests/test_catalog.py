"""Dataset catalog, missing-file errors, and check-data CLI."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adv_ids.data.catalog import (
    CATALOG,
    DatasetLayoutError,
    missing_data_message,
    require_dataset_files,
)
from adv_ids.data.setup import check_dataset
from adv_ids.utils.config import load_config
from run import main as run_main


def test_catalog_covers_first_class_datasets():
    for name in ("cicids2017", "cicids2018", "unsw_nb15"):
        entry = CATALOG[name]
        assert entry.homepage.startswith("http")
        assert entry.raw_dir.startswith("data/raw/")
        assert entry.example_files
        assert "sha256" in entry.checksum_notes.lower() or "hash" in entry.checksum_notes.lower()


def test_missing_data_message_is_actionable():
    msg = missing_data_message("unsw_nb15", root=ROOT)
    assert "data/raw/unsw-nb15" in msg
    assert "UNSW_NB15_training-set.csv" in msg
    assert "https://research.unsw.edu.au/projects/unsw-nb15-dataset" in msg
    assert "python run.py setup-data --dataset-name unsw_nb15" in msg
    assert "tests/fixtures/unsw_nb15_official_sample.csv" in msg


def test_require_dataset_files_raises_layout_error(tmp_path):
    with pytest.raises(DatasetLayoutError) as exc:
        require_dataset_files("cicids2017", root=tmp_path)
    text = str(exc.value)
    assert "data/raw/cicids2017" in text
    assert "unb.ca/cic/datasets/ids-2017" in text


def test_require_dataset_files_accepts_existing_csv(tmp_path):
    csv_path = tmp_path / "tiny.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    found = require_dataset_files("cicids2017", csv_path, root=tmp_path)
    assert found == csv_path


def test_check_dataset_reports_not_ready():
    status = check_dataset("cicids2018", root=ROOT)
    assert status["dataset"] == "cicids2018"
    assert status["ready"] is False
    assert "cse-cic-ids2018" in status["hint"]


def test_prepare_without_files_exits_2(capsys):
    # CIC-IDS2018 day files are never auto-fetched (AWS, multi-GB).
    code = run_main(["prepare", "--dataset-name", "cicids2018"])
    assert code == 2
    err = capsys.readouterr().err
    assert "Dataset 'cicids2018' is not on disk" in err
    assert "cse-cic-ids2018" in err


def test_check_data_without_files_exits_2(capsys):
    # CIC day files are never auto-fetched, so this stays missing in CI / this VM.
    code = run_main(["check-data", "--dataset-name", "cicids2018"])
    assert code == 2
    err = capsys.readouterr().err
    assert "cse-cic-ids2018" in err


def test_medium_config_has_multi_seed_matched_eps():
    cfg = load_config(ROOT / "configs" / "medium.yaml")
    assert cfg["name"] == "medium_synthetic"
    assert len(cfg["seeds"]) >= 3
    assert "mlp" in cfg["models"]
    assert "random_forest" in cfg["models"]
    attacks = {(a["name"], float(a["eps"])) for a in cfg["attacks"]}
    assert {("fgsm", 0.15), ("fgsm", 0.25), ("pgd", 0.15), ("pgd", 0.25), ("gan", 0.15), ("gan", 0.25)} <= attacks
    assert cfg["export_tables"] is True
    assert cfg["tables_dir"] == "results/tables"
