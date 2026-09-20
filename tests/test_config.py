from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adv_ids.utils.config import load_config, merge_dict


def test_load_quick_yaml():
    cfg = load_config(ROOT / "configs" / "quick.yaml")
    assert cfg["name"] == "quick_synthetic"
    assert "mlp" in cfg["models"]
    assert any(a["name"] == "gan" for a in cfg["attacks"])


def test_load_medium_yaml():
    cfg = load_config(ROOT / "configs" / "medium.yaml")
    assert cfg["synthetic"] is True
    assert cfg["seeds"] == [42, 43, 44]
    assert {ds["name"] for ds in cfg["datasets"]} == {"cicids2017", "unsw_nb15"}


def test_load_real_long_yamls():
    unsw = load_config(ROOT / "configs" / "unsw_real_long.yaml")
    assert unsw["synthetic"] is False
    assert len(unsw["seeds"]) >= 3
    assert "mlp" in unsw["models"] and "random_forest" in unsw["models"]
    cic = load_config(ROOT / "configs" / "cicids_engelen_long.yaml")
    assert cic["synthetic"] is False
    assert any(ds.get("tag") == "cicids2017_engelen_friday" for ds in cic["datasets"])
    assert any(ds.get("tag") == "cicids2017_engelen_week" for ds in cic["datasets"])
    friday = load_config(ROOT / "configs" / "cicids_engelen_friday_long.yaml")
    assert friday["synthetic"] is False
    assert len(friday["datasets"]) == 1
    assert friday["datasets"][0].get("tag") == "cicids2017_engelen_friday"


def test_merge_dict_nested():
    out = merge_dict({"a": {"b": 1, "c": 2}, "d": 3}, {"a": {"c": 9}, "e": 4})
    assert out["a"]["b"] == 1
    assert out["a"]["c"] == 9
    assert out["e"] == 4
