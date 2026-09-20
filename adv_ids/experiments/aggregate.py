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
    """One row per (dataset, model, defense, eps) with per-attack evasion and L2.

    Per-seed rows are aggregated first so a multi-seed suite does not silently
    keep only the last seed.
    """
    usable = [r for r in rows if not str(r.get("model", "")).startswith("transfer:")]
    if usable and "evasion_rate" in usable[0] and "evasion_rate_mean" not in usable[0]:
        usable = aggregate_rows(usable)
    grouped: dict[tuple, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in usable:
        key = (row.get("dataset"), row.get("model"), row.get("defense"), row.get("eps"))
        grouped[key][str(row.get("attack"))] = row
    out = []
    for key, attacks in sorted(grouped.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        rec = {
            "dataset": key[0],
            "model": key[1],
            "defense": key[2],
            "eps": key[3],
            "n_seeds": next(iter(attacks.values())).get("n_seeds", 1),
            "attacks": ",".join(sorted(attacks)),
        }
        for name, row in sorted(attacks.items()):
            rec[f"{name}_evasion"] = row.get("evasion_rate_mean", row.get("evasion_rate"))
            rec[f"{name}_l2"] = row.get("mean_l2_perturbation_mean", row.get("mean_l2_perturbation"))
            rec[f"{name}_acc_drop"] = row.get("accuracy_drop_mean", row.get("accuracy_drop"))
        out.append(rec)
    return out


def matched_l2_gan_vs_pgd(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Closest mean-L2 GAN vs PGD (and FGSM) pair per dataset/model/defense.

    Sweep cells are compared after multi-seed aggregation. If no pair exists,
    the row is omitted rather than invented. ``l2_gap`` documents mismatch.
    """
    usable = [r for r in rows if not str(r.get("model", "")).startswith("transfer:")]
    if usable and "evasion_rate" in usable[0] and "evasion_rate_mean" not in usable[0]:
        usable = aggregate_rows(usable)
    grouped: dict[tuple, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in usable:
        grouped[(row.get("dataset"), row.get("model"), row.get("defense"))][str(row.get("attack"))].append(row)

    def _l2(row: dict[str, Any]) -> float | None:
        v = row.get("mean_l2_perturbation_mean", row.get("mean_l2_perturbation"))
        return float(v) if _is_number(v) else None

    def _ev(row: dict[str, Any]) -> float | None:
        v = row.get("evasion_rate_mean", row.get("evasion_rate"))
        return float(v) if _is_number(v) else None

    out: list[dict[str, Any]] = []
    for key, attacks in sorted(grouped.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        gans = [r for r in attacks.get("gan", []) if _l2(r) is not None]
        pgds = [r for r in attacks.get("pgd", []) if _l2(r) is not None]
        fgsms = [r for r in attacks.get("fgsm", []) if _l2(r) is not None]
        if not gans or not pgds:
            continue
        best_g, best_p, best_gap = None, None, None
        for g in gans:
            for p in pgds:
                gap = abs(_l2(g) - _l2(p))
                if best_gap is None or gap < best_gap:
                    best_g, best_p, best_gap = g, p, gap
        best_f, f_gap = None, None
        if fgsms:
            for f in fgsms:
                gap = abs(_l2(best_g) - _l2(f))
                if f_gap is None or gap < f_gap:
                    best_f, f_gap = f, gap
        rec = {
            "dataset": key[0],
            "model": key[1],
            "defense": key[2],
            "n_seeds": best_g.get("n_seeds", 1),
            "gan_eps": best_g.get("eps"),
            "gan_l2": _l2(best_g),
            "gan_evasion": _ev(best_g),
            "pgd_eps": best_p.get("eps"),
            "pgd_l2": _l2(best_p),
            "pgd_evasion": _ev(best_p),
            "l2_gap_gan_pgd": best_gap,
            "note": (
                "closest mean-L2 pair from the executed eps sweep; "
                f"L2 gap={best_gap:.4f}"
            ),
        }
        if best_f is not None:
            rec.update({
                "fgsm_eps": best_f.get("eps"),
                "fgsm_l2": _l2(best_f),
                "fgsm_evasion": _ev(best_f),
                "l2_gap_gan_fgsm": f_gap,
            })
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
    matched_l2 = matched_l2_gan_vs_pgd(rows)
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
        "matched_l2_csv": str(write_csv(dest / f"{prefix}_matched_l2.csv", matched_l2)),
        "matched_l2_md": str(
            write_markdown(
                dest / f"{prefix}_matched_l2.md",
                matched_l2,
                [
                    "dataset", "model", "defense", "n_seeds",
                    "gan_eps", "gan_l2", "gan_evasion",
                    "pgd_eps", "pgd_l2", "pgd_evasion", "l2_gap_gan_pgd",
                    "fgsm_eps", "fgsm_l2", "fgsm_evasion", "note",
                ],
            )
        ),
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
