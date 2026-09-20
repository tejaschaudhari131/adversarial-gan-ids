"""Aggregate multi-seed suite rows into mean/std tables (CSV + Markdown)."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

NUMERIC = (
    "clean_accuracy",
    "adversarial_accuracy",
    "accuracy_drop",
    "evasion_rate",
    "attack_success_rate",
    "mean_l2_perturbation",
    "mean_linf_perturbation",
)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def aggregate_rows(
    rows: list[dict[str, Any]],
    group_keys: tuple[str, ...] = ("dataset", "model", "attack", "defense", "eps"),
) -> list[dict[str, Any]]:
    buckets: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        buckets[key].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(buckets.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        rec: dict[str, Any] = {k: v for k, v in zip(group_keys, key)}
        rec["n_seeds"] = len(items)
        rec["seeds"] = ",".join(str(i.get("seed", "")) for i in items)
        for metric in NUMERIC:
            vals = [float(i[metric]) for i in items if _is_number(i.get(metric))]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            rec[f"{metric}_mean"] = mean
            if len(vals) > 1:
                rec[f"{metric}_std"] = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
            else:
                rec[f"{metric}_std"] = 0.0
        out.append(rec)
    return out


def matched_eps_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (dataset, model, defense, eps) with per-attack evasion and L2."""
    grouped: dict[tuple, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("model", "").startswith("transfer:"):
            continue
        key = (row.get("dataset"), row.get("model"), row.get("defense"), row.get("eps"))
        grouped[key][str(row.get("attack"))] = row
    out = []
    for key, attacks in sorted(grouped.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        rec = {
            "dataset": key[0],
            "model": key[1],
            "defense": key[2],
            "eps": key[3],
            "attacks": ",".join(sorted(attacks)),
        }
        for name, row in sorted(attacks.items()):
            rec[f"{name}_evasion"] = row.get("evasion_rate")
            rec[f"{name}_l2"] = row.get("mean_l2_perturbation")
            rec[f"{name}_acc_drop"] = row.get("accuracy_drop")
        out.append(rec)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    cols: list[str] = []
    for row in rows:
        for k in row:
            if k not in cols:
                cols.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return path


def write_markdown(path: Path, rows: list[dict[str, Any]], cols: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("_No rows. This table is only filled from executed runs._\n", encoding="utf-8")
        return path
    if cols is None:
        cols = []
        for row in rows:
            for k in row:
                if k not in cols:
                    cols.append(k)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in rows:
        cells = []
        for c in cols:
            v = row.get(c, "")
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append("" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def export_suite_tables(
    rows: list[dict[str, Any]],
    dest: Path,
    prefix: str,
    caption: str = "",
) -> dict[str, str]:
    dest.mkdir(parents=True, exist_ok=True)
    raw_md_cols = [
        "seed", "dataset", "model", "attack", "defense", "eps",
        "clean_accuracy", "adversarial_accuracy", "accuracy_drop",
        "evasion_rate", "attack_success_rate", "mean_l2_perturbation",
    ]
    agg = aggregate_rows(rows)
    matched = matched_eps_rows(rows)
    paths = {
        "per_seed_csv": str(write_csv(dest / f"{prefix}_per_seed.csv", rows)),
        "per_seed_md": str(write_markdown(dest / f"{prefix}_per_seed.md", rows, raw_md_cols)),
        "aggregate_csv": str(write_csv(dest / f"{prefix}_aggregate.csv", agg)),
        "aggregate_md": str(
            write_markdown(
                dest / f"{prefix}_aggregate.md",
                agg,
                [
                    "dataset", "model", "attack", "defense", "eps", "n_seeds",
                    "evasion_rate_mean", "evasion_rate_std",
                    "mean_l2_perturbation_mean", "mean_l2_perturbation_std",
                    "clean_accuracy_mean", "accuracy_drop_mean",
                ],
            )
        ),
        "matched_eps_csv": str(write_csv(dest / f"{prefix}_matched_eps.csv", matched)),
        "matched_eps_md": str(write_markdown(dest / f"{prefix}_matched_eps.md", matched)),
    }
    note = dest / f"{prefix}_README.md"
    note.write_text(
        (caption.strip() + "\n\n" if caption else "")
        + "These tables were filled from executed runs only. "
        "Empty cells were not measured. Synthetic rows are not CIC/UNSW paper results.\n",
        encoding="utf-8",
    )
    paths["readme"] = str(note)
    return paths
