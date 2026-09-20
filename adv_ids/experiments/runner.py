"""Config-driven experiment suite: datasets × models × attacks × defense × transfer."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from adv_ids.attacks.fgsm import fgsm_attack
from adv_ids.attacks.gan_attack import GANAttack
from adv_ids.attacks.pgd import pgd_attack
from adv_ids.attacks.transfer import describe_transfer
from adv_ids.data.catalog import DatasetLayoutError
from adv_ids.data.loaders import resolve_dataset_path
from adv_ids.data.preprocess import prepare_dataset
from adv_ids.data.synthetic import generate_synthetic
from adv_ids.defenses.adv_train import train_adversarial_mlp
from adv_ids.evaluation.plots import (
    save_aggregate_evasion_bars,
    save_attack_comparison,
    save_evasion_vs_l2,
    save_transfer_heatmap,
)
from adv_ids.evaluation.protocol import evaluate_attack_on_model, evaluate_clean
from adv_ids.experiments.aggregate import aggregate_rows, export_suite_tables
from adv_ids.models.base import IDSModel
from adv_ids.models.registry import build_ids
from adv_ids.utils.config import load_config
from adv_ids.utils.io import ensure_dir, write_json
from adv_ids.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _attack_rows(y: np.ndarray) -> np.ndarray:
    return np.where(y == 1)[0]


def _val_split(data: dict) -> tuple[np.ndarray, np.ndarray]:
    X_val, y_val = data.get("X_val"), data.get("y_val")
    if X_val is not None and len(X_val) > 0:
        return X_val, y_val
    return data["X_test"], data["y_test"]


def _normalize_attack_cfg(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        return {"name": item}
    return dict(item)


def _normalize_dataset_cfg(item: Any, defaults: dict[str, Any]) -> dict[str, Any]:
    if isinstance(item, str):
        item = {"name": item}
    cfg = {**defaults, **item}
    cfg["name"] = cfg.get("name") or cfg.get("dataset") or "cicids2017"
    return cfg


def _fit_ids(name: str, data: dict, ids_cfg: dict, artifacts: Path) -> IDSModel:
    X_val, y_val = _val_split(data)
    model = build_ids(name, input_dim=data["X_train"].shape[1])
    model.fit(
        data["X_train"],
        data["y_train"],
        X_val=X_val,
        y_val=y_val,
        epochs=int(ids_cfg.get("epochs", 10)),
        batch_size=int(ids_cfg.get("batch_size", 256)),
        lr=float(ids_cfg.get("lr", 1e-3)),
        artifacts_dir=artifacts,
    )
    return model


def _craft(
    attack_cfg: dict[str, Any],
    model: IDSModel,
    X_attack: np.ndarray,
    mask: np.ndarray,
    data: dict,
    artifacts: Path,
    gan_cache: dict[str, GANAttack],
) -> np.ndarray:
    name = attack_cfg["name"].lower()
    eps = float(attack_cfg.get("eps", 0.15))
    if name == "fgsm":
        if not model.differentiable:
            raise TypeError("fgsm needs a differentiable model")
        return fgsm_attack(model, X_attack, mask, eps=eps)
    if name == "pgd":
        if not model.differentiable:
            raise TypeError("pgd needs a differentiable model")
        return pgd_attack(
            model,
            X_attack,
            mask,
            eps=eps,
            steps=int(attack_cfg.get("steps", 10)),
            step_size=attack_cfg.get("step_size"),
            random_start=bool(attack_cfg.get("random_start", True)),
        )
    if name == "gan":
        key = f"{model.name}:{eps}:{attack_cfg.get('epochs', 8)}"
        if key not in gan_cache:
            gan = GANAttack(feature_dim=X_attack.shape[1], eps=eps)
            gan.fit(
                model,
                data["X_train"],
                data["y_train"],
                mask,
                epochs=int(attack_cfg.get("epochs", 8)),
                batch_size=int(attack_cfg.get("batch_size", 128)),
                lambda_ids=float(attack_cfg.get("lambda_ids", 4.0)),
                artifacts_dir=artifacts / f"gan_{model.name}_{eps}",
            )
            gan_cache[key] = gan
        return gan_cache[key].generate(X_attack, mask)
    raise ValueError(f"Unknown attack {name}")


def run_single_pipeline(
    *,
    dataset_name: str = "cicids2017",
    dataset_path: str | None = None,
    synthetic: bool = True,
    n_benign: int = 1500,
    n_attack: int = 800,
    seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.0,
    ids_epochs: int = 4,
    gan_epochs: int = 8,
    batch_size: int = 128,
    eps: float = 0.25,
    lambda_ids: float = 4.0,
    artifacts_dir: str | Path = "artifacts",
    results_dir: str | Path = "results",
    model_name: str = "mlp",
    label_mode: str = "binary",
) -> dict[str, Any]:
    """Default research pipeline: prepare → IDS → GAN → evaluate."""
    set_seed(seed)
    path = resolve_dataset_path(
        dataset_name,
        dataset_path,
        synthetic=synthetic,
        n_benign=n_benign,
        n_attack=n_attack,
        seed=seed,
    )
    data = prepare_dataset(
        path,
        test_size=test_size,
        random_state=seed,
        artifacts_dir=artifacts_dir,
        dataset_name=dataset_name,
        val_size=val_size,
        label_mode=label_mode,
    )
    X_val, y_val = _val_split(data)
    model = _fit_ids(model_name, data, {"epochs": ids_epochs, "batch_size": batch_size}, Path(artifacts_dir))
    # Historical MLP filename.
    if model.name == "mlp":
        model.save(Path(artifacts_dir) / "ids_model.pt")
    gan = GANAttack(feature_dim=data["X_train"].shape[1], eps=eps)
    gan.fit(
        model,
        data["X_train"],
        data["y_train"],
        data["modifiable_mask"],
        epochs=gan_epochs,
        batch_size=batch_size,
        lambda_ids=lambda_ids,
        artifacts_dir=artifacts_dir,
    )
    attack_idx = _attack_rows(data["y_test"])
    X_adv = gan.generate(data["X_test"][attack_idx], data["modifiable_mask"])
    results = evaluate_attack_on_model(
        model,
        data["X_test"],
        data["y_test"],
        X_adv,
        attack_idx,
        results_dir=results_dir,
        tag="gan",
    )
    # Also write the historical filename.
    write_json(
        Path(results_dir) / "evaluation_metrics.json",
        {
            "clean_accuracy": results["clean_accuracy"],
            "clean_f1": results["clean_f1"],
            "clean_roc_auc": results["clean_roc_auc"],
            "evasion_rate": results["evasion_rate"],
            "adversarial_accuracy": results["adversarial_accuracy"],
            "accuracy_drop": results["accuracy_drop"],
            "mean_abs_perturbation": results["mean_abs_perturbation"],
            "mean_l2_perturbation": results["mean_l2_perturbation"],
            "attack_detected_before": results["attack_detected_before"],
            "attack_detected_after": results["attack_detected_after"],
            "n_attack_test": results["n_attack"],
            "mean_p_attack_after": results["mean_p_attack_after"],
            "dataset": dataset_name,
            "model": model_name,
            "attack": "gan",
            "synthetic": synthetic,
        },
    )
    from adv_ids.evaluation.plots import save_robustness_bars

    save_robustness_bars(
        results["clean_accuracy"],
        results["adversarial_accuracy"],
        results["evasion_rate"],
        Path(results_dir) / "robustness_summary.png",
        title="IDS robustness under adversarial GAN traffic",
    )
    return results


def run_experiment_suite(config: dict[str, Any] | str | Path, root: str | Path = ".") -> dict[str, Any]:
    if not isinstance(config, dict):
        config = load_config(config)
    root = Path(root)
    seeds = [int(s) for s in config.get("seeds") or [config.get("seed", 42)]]
    run_name = config.get("name") or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    out_root = ensure_dir(root / config.get("results_dir", "results") / run_name)
    artifacts_root = ensure_dir(root / config.get("artifacts_dir", "artifacts") / run_name)

    ids_cfg = config.get("ids", {})
    defense_cfg = config.get("defense") or {}
    transfer_cfg = config.get("transfer") or {}
    defaults = {
        "synthetic": bool(config.get("synthetic", True)),
        "n_benign": int(config.get("n_benign", 1500)),
        "n_attack": int(config.get("n_attack", 800)),
        "test_size": float(config.get("test_size", 0.2)),
        "val_size": float(config.get("val_size", 0.1)),
        "label_mode": config.get("label_mode", "binary"),
        "max_samples": config.get("max_samples"),
        "max_files": config.get("max_files"),
    }

    summary_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    write_json(out_root / "config.json", config)

    for seed in seeds:
        set_seed(seed)
        logger.info("===== seed %s =====", seed)
        for raw_ds in config.get("datasets", ["cicids2017"]):
            ds_cfg = _normalize_dataset_cfg(raw_ds, defaults)
            ds_name = ds_cfg["name"]
            logger.info("=== dataset %s ===", ds_name)
            try:
                path = resolve_dataset_path(
                    ds_name,
                    ds_cfg.get("path") or ds_cfg.get("dataset"),
                    synthetic=bool(ds_cfg.get("synthetic", True)),
                    n_benign=int(ds_cfg.get("n_benign", 1500)),
                    n_attack=int(ds_cfg.get("n_attack", 800)),
                    seed=seed,
                    output_dir=root / "data",
                )
            except (FileNotFoundError, DatasetLayoutError) as exc:
                if ds_cfg.get("allow_synthetic_fallback", True):
                    logger.warning("%s — falling back to synthetic %s", exc, ds_name)
                    path = root / "data" / f"synthetic_{ds_name}.csv"
                    generate_synthetic(
                        ds_name,
                        n_benign=int(ds_cfg.get("n_benign", 1500)),
                        n_attack=int(ds_cfg.get("n_attack", 800)),
                        seed=seed,
                        output_path=path,
                    )
                else:
                    logger.warning("Skipping dataset %s:\n%s", ds_name, exc)
                    continue

            ds_art = ensure_dir(artifacts_root / f"seed{seed}" / ds_name)
            data = prepare_dataset(
                path,
                test_size=float(ds_cfg.get("test_size", 0.2)),
                random_state=seed,
                artifacts_dir=ds_art,
                dataset_name=ds_name,
                val_size=float(ds_cfg.get("val_size", 0.1)),
                label_mode=ds_cfg.get("label_mode", "binary"),
                max_files=ds_cfg.get("max_files"),
                max_samples=ds_cfg.get("max_samples"),
            )
            mask = data["modifiable_mask"]
            attack_idx = _attack_rows(data["y_test"])
            X_attack = data["X_test"][attack_idx]
            models: dict[str, IDSModel] = {}
            gan_cache: dict[str, GANAttack] = {}

            for model_name in config.get("models", ["mlp"]):
                logger.info("--- model %s ---", model_name)
                model_art = ensure_dir(ds_art / model_name)
                model = _fit_ids(model_name, data, ids_cfg, model_art)
                models[model.name] = model
                clean = evaluate_clean(
                    model, data["X_test"], data["y_test"],
                    results_dir=out_root / f"seed{seed}" / ds_name / model.name,
                )
                write_json(
                    out_root / f"seed{seed}" / ds_name / model.name / "clean.json",
                    {k: v for k, v in clean.items() if k != "classification_report"},
                )

                for raw_atk in config.get("attacks", ["fgsm", "pgd"]):
                    atk = _normalize_attack_cfg(raw_atk)
                    try:
                        X_adv = _craft(atk, model, X_attack, mask, data, model_art, gan_cache)
                    except TypeError as exc:
                        logger.info("Skip %s on %s: %s", atk["name"], model.name, exc)
                        continue
                    metrics = evaluate_attack_on_model(
                        model, data["X_test"], data["y_test"], X_adv, attack_idx,
                        results_dir=out_root / f"seed{seed}" / ds_name / model.name, tag=atk["name"],
                    )
                    row = {
                        "seed": seed,
                        "dataset": ds_name,
                        "model": model.name,
                        "attack": atk["name"],
                        "defense": "none",
                        "eps": float(atk.get("eps", 0.15)),
                        "clean_accuracy": metrics["clean_accuracy"],
                        "adversarial_accuracy": metrics["adversarial_accuracy"],
                        "accuracy_drop": metrics["accuracy_drop"],
                        "evasion_rate": metrics["evasion_rate"],
                        "attack_success_rate": metrics["attack_success_rate"],
                        "mean_l2_perturbation": metrics["mean_l2_perturbation"],
                        "mean_linf_perturbation": metrics.get("mean_linf_perturbation"),
                        "label": f"{ds_name}/{model.name}/{atk['name']}/ε={atk.get('eps', 0.15)}",
                    }
                    summary_rows.append(row)
                    comparison_rows.append(row)

            if defense_cfg.get("name") in {"adv_train", "adversarial_training", "pgd_train"}:
                logger.info("--- defense adversarial training ---")
                X_val, y_val = _val_split(data)
                defended = train_adversarial_mlp(
                    data["X_train"],
                    data["y_train"],
                    mask,
                    X_val=X_val,
                    y_val=y_val,
                    epochs=int(defense_cfg.get("epochs", ids_cfg.get("epochs", 6))),
                    batch_size=int(ids_cfg.get("batch_size", 128)),
                    eps=float(defense_cfg.get("eps", 0.15)),
                    pgd_steps=int(defense_cfg.get("steps", 5)),
                    artifacts_dir=ds_art / "mlp_advtrain",
                )["model"]
                models[defended.name] = defended
                for raw_atk in config.get("attacks", ["fgsm", "pgd"]):
                    atk = _normalize_attack_cfg(raw_atk)
                    if atk["name"].lower() == "gan":
                        continue
                    try:
                        X_adv = _craft(atk, defended, X_attack, mask, data, ds_art / "mlp_advtrain", gan_cache)
                    except TypeError:
                        continue
                    metrics = evaluate_attack_on_model(
                        defended, data["X_test"], data["y_test"], X_adv, attack_idx,
                        results_dir=out_root / f"seed{seed}" / ds_name / defended.name, tag=atk["name"],
                    )
                    summary_rows.append({
                        "seed": seed,
                        "dataset": ds_name,
                        "model": defended.name,
                        "attack": atk["name"],
                        "defense": "adv_train",
                        "eps": float(atk.get("eps", defense_cfg.get("eps", 0.15))),
                        "clean_accuracy": metrics["clean_accuracy"],
                        "adversarial_accuracy": metrics["adversarial_accuracy"],
                        "accuracy_drop": metrics["accuracy_drop"],
                        "evasion_rate": metrics["evasion_rate"],
                        "attack_success_rate": metrics["attack_success_rate"],
                        "mean_l2_perturbation": metrics["mean_l2_perturbation"],
                        "mean_linf_perturbation": metrics.get("mean_linf_perturbation"),
                        "label": f"{ds_name}/{defended.name}/{atk['name']}/ε={atk.get('eps')}",
                    })

            if transfer_cfg and len(models) >= 2:
                surrogate_name = transfer_cfg.get("surrogate", "mlp")
                attack_name = transfer_cfg.get("attack", "pgd")
                surrogate = models.get(surrogate_name) or models.get("mlp")
                if surrogate is None or not surrogate.differentiable:
                    logger.info("Transfer skipped: no differentiable surrogate.")
                else:
                    atk = {"name": attack_name, **{k: v for k, v in transfer_cfg.items() if k not in {"surrogate", "targets", "attack"}}}
                    X_adv = _craft(atk, surrogate, X_attack, mask, data, ds_art / surrogate.name, gan_cache)
                    targets = transfer_cfg.get("targets") or [n for n in models if n != surrogate.name]
                    for tname in targets:
                        target = models.get(tname)
                        if target is None:
                            continue
                        metrics = evaluate_attack_on_model(
                            target, data["X_test"], data["y_test"], X_adv, attack_idx,
                            results_dir=out_root / f"seed{seed}" / ds_name / "transfer",
                            tag=f"{surrogate.name}_to_{target.name}_{attack_name}",
                        )
                        row = {
                            **describe_transfer(surrogate, target, attack_name),
                            "seed": seed,
                            "dataset": ds_name,
                            "eps": float(transfer_cfg.get("eps", 0.15)),
                            "evasion_rate": metrics["evasion_rate"],
                            "attack_success_rate": metrics["attack_success_rate"],
                            "accuracy_drop": metrics["accuracy_drop"],
                            "clean_accuracy": metrics["clean_accuracy"],
                        }
                        transfer_rows.append(row)
                        summary_rows.append({
                            "seed": seed,
                            "dataset": ds_name,
                            "model": f"transfer:{surrogate.name}->{target.name}",
                            "attack": attack_name,
                            "defense": "none",
                            "eps": float(transfer_cfg.get("eps", 0.15)),
                            "clean_accuracy": metrics["clean_accuracy"],
                            "adversarial_accuracy": metrics["adversarial_accuracy"],
                            "accuracy_drop": metrics["accuracy_drop"],
                            "evasion_rate": metrics["evasion_rate"],
                            "attack_success_rate": metrics["attack_success_rate"],
                            "mean_l2_perturbation": metrics["mean_l2_perturbation"],
                            "mean_linf_perturbation": metrics.get("mean_linf_perturbation"),
                        })

    save_attack_comparison(comparison_rows, out_root / "attack_comparison.png")
    save_evasion_vs_l2(summary_rows, out_root / "evasion_vs_l2.png")
    agg_rows = aggregate_rows(summary_rows)
    save_aggregate_evasion_bars(agg_rows, out_root / "aggregate_evasion.png")
    if transfer_rows:
        row_labels = sorted({f"{r['dataset']}/{r['surrogate']}" for r in transfer_rows})
        col_labels = sorted({r["target"] for r in transfer_rows})
        mat = np.zeros((len(row_labels), len(col_labels)))
        for r in transfer_rows:
            i = row_labels.index(f"{r['dataset']}/{r['surrogate']}")
            j = col_labels.index(r["target"])
            mat[i, j] = r["evasion_rate"]
        save_transfer_heatmap(mat, row_labels, col_labels, out_root / "transfer_heatmap.png")

    table_paths = {}
    if config.get("export_tables", True):
        prefix = config.get("tables_prefix") or run_name
        caption = (
            f"Executed suite `{run_name}` seeds={seeds}. "
            "Numbers are from this run only; do not mix with other papers."
        )
        table_paths = export_suite_tables(summary_rows, out_root, prefix, caption=caption)
        if config.get("tables_dir"):
            public = export_suite_tables(
                summary_rows,
                root / config["tables_dir"],
                prefix,
                caption=caption,
            )
            table_paths.update({f"public_{k}": v for k, v in public.items()})

    payload = {
        "name": run_name,
        "seeds": seeds,
        "n_rows": len(summary_rows),
        "results": summary_rows,
        "transfer": transfer_rows,
        "aggregate": agg_rows,
        "tables": table_paths,
        "note": "Metrics below are from the runs that actually executed. Empty cells were not measured.",
    }
    write_json(out_root / "suite_metrics.json", payload)
    _write_markdown_table(out_root / "suite_metrics.md", summary_rows)
    logger.info("Experiment suite written to %s", out_root)
    return payload


def _write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = [
        "seed", "dataset", "model", "attack", "defense", "eps",
        "clean_accuracy", "adversarial_accuracy", "accuracy_drop",
        "evasion_rate", "attack_success_rate", "mean_l2_perturbation",
    ]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        cells = []
        for c in cols:
            v = row.get(c, "")
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
