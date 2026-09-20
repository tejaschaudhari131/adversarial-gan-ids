"""Multi-seed aggregation and matched-eps table export."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adv_ids.experiments.aggregate import (
    aggregate_rows,
    export_suite_tables,
    matched_eps_rows,
)


def _row(seed, dataset, model, attack, eps, evasion, l2, drop=0.1, defense="none"):
    return {
        "seed": seed,
        "dataset": dataset,
        "model": model,
        "attack": attack,
        "defense": defense,
        "eps": eps,
        "clean_accuracy": 0.9,
        "adversarial_accuracy": 0.8,
        "accuracy_drop": drop,
        "evasion_rate": evasion,
        "attack_success_rate": evasion,
        "mean_l2_perturbation": l2,
        "mean_linf_perturbation": eps,
    }


def test_aggregate_mean_and_std():
    rows = [
        _row(42, "cicids2017", "mlp", "pgd", 0.15, 0.40, 1.0),
        _row(43, "cicids2017", "mlp", "pgd", 0.15, 0.50, 1.2),
        _row(44, "cicids2017", "mlp", "pgd", 0.15, 0.60, 1.4),
    ]
    agg = aggregate_rows(rows)
    assert len(agg) == 1
    assert agg[0]["n_seeds"] == 3
    assert abs(agg[0]["evasion_rate_mean"] - 0.5) < 1e-9
    assert abs(agg[0]["evasion_rate_std"] - 0.1) < 1e-9
    assert abs(agg[0]["mean_l2_perturbation_mean"] - 1.2) < 1e-9


def test_matched_eps_rows_side_by_side():
    rows = [
        _row(42, "cicids2017", "mlp", "fgsm", 0.15, 0.70, 0.8),
        _row(42, "cicids2017", "mlp", "pgd", 0.15, 0.55, 1.1),
        _row(42, "cicids2017", "mlp", "gan", 0.15, 0.20, 0.4),
        _row(42, "cicids2017", "transfer:mlp->random_forest", "pgd", 0.15, 0.05, 1.1),
    ]
    matched = matched_eps_rows(rows)
    assert len(matched) == 1
    rec = matched[0]
    assert rec["eps"] == 0.15
    assert rec["n_seeds"] == 1
    assert rec["fgsm_evasion"] == 0.70
    assert rec["pgd_l2"] == 1.1
    assert rec["gan_acc_drop"] == 0.1
    assert "transfer" not in rec["attacks"]


def test_export_suite_tables_writes_csv_and_md(tmp_path):
    rows = [
        _row(42, "unsw_nb15", "mlp", "fgsm", 0.25, 0.3, 0.9),
        _row(43, "unsw_nb15", "mlp", "fgsm", 0.25, 0.4, 1.0),
    ]
    paths = export_suite_tables(rows, tmp_path, "unit", caption="unit test")
    for key in ("per_seed_csv", "aggregate_md", "matched_eps_csv", "readme"):
        assert Path(paths[key]).is_file()
        assert Path(paths[key]).stat().st_size > 0
    text = Path(paths["aggregate_md"]).read_text(encoding="utf-8")
    assert "evasion_rate_mean" in text
    assert "0.3500" in text
