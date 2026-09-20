#!/usr/bin/env python3
"""CLI for the adversarial-robustness IDS research pipeline.

Commands: prepare | train-ids | train-attack | train-gan | evaluate | pipeline | experiment-suite
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adv_ids.attacks.fgsm import fgsm_attack
from adv_ids.attacks.gan_attack import GANAttack
from adv_ids.attacks.pgd import pgd_attack
from adv_ids.data.catalog import DatasetLayoutError
from adv_ids.data.loaders import resolve_dataset_path
from adv_ids.data.preprocess import prepare_dataset
from adv_ids.data.setup import check_dataset, print_status, setup_dataset
from adv_ids.evaluation.protocol import evaluate_adversarial, evaluate_attack_on_model, evaluate_clean
from adv_ids.experiments.runner import run_experiment_suite, run_single_pipeline
from adv_ids.training.gan import train_adversarial_gan
from adv_ids.training.ids import load_ids_wrapper, train_ids_model
from adv_ids.utils.seed import set_seed


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )


def _apply_quick(args) -> None:
    if getattr(args, "quick", False):
        args.n_benign = min(getattr(args, "n_benign", 1500), 1500)
        args.n_attack = min(getattr(args, "n_attack", 800), 800)
        args.ids_epochs = min(getattr(args, "ids_epochs", 4), 4)
        if hasattr(args, "gan_epochs"):
            args.gan_epochs = min(args.gan_epochs, 8)
        if hasattr(args, "attack_epochs"):
            args.attack_epochs = min(args.attack_epochs, 8)
        args.batch_size = min(getattr(args, "batch_size", 128), 128)


def _resolve_dataset(args) -> Path:
    return resolve_dataset_path(
        getattr(args, "dataset_name", "cicids2017"),
        getattr(args, "dataset", None),
        synthetic=bool(getattr(args, "synthetic", False)),
        n_benign=getattr(args, "n_benign", 8000),
        n_attack=getattr(args, "n_attack", 4000),
        seed=getattr(args, "seed", 42),
        output_dir=ROOT / "data",
    )


def cmd_prepare(args) -> dict:
    set_seed(args.seed)
    dataset = _resolve_dataset(args)
    return prepare_dataset(
        dataset,
        test_size=args.test_size,
        random_state=args.seed,
        artifacts_dir=args.artifacts,
        dataset_name=args.dataset_name,
        val_size=getattr(args, "val_size", 0.0),
        label_mode=getattr(args, "label_mode", "binary"),
        max_samples=getattr(args, "max_samples", None),
    )


def cmd_train_ids(args) -> None:
    data = cmd_prepare(args)
    train_ids_model(
        data["X_train"],
        data["y_train"],
        X_val=data.get("X_val") if len(data.get("X_val", [])) else data["X_test"],
        y_val=data.get("y_val") if len(data.get("y_val", [])) else data["y_test"],
        epochs=args.ids_epochs,
        batch_size=args.batch_size,
        artifacts_dir=args.artifacts,
        model_name=getattr(args, "ids_model", "mlp"),
    )


def cmd_train_gan(args) -> None:
    data = cmd_prepare(args)
    if not (Path(args.artifacts) / "ids_model.pt").is_file() and not (Path(args.artifacts) / "mlp.pt").is_file():
        logging.getLogger(__name__).info("No IDS checkpoint found — training IDS first.")
        train_ids_model(
            data["X_train"],
            data["y_train"],
            X_val=data["X_test"],
            y_val=data["y_test"],
            epochs=args.ids_epochs,
            batch_size=args.batch_size,
            artifacts_dir=args.artifacts,
            model_name=getattr(args, "ids_model", "mlp"),
        )
    train_adversarial_gan(
        data["X_train"],
        data["y_train"],
        data["modifiable_mask"],
        artifacts_dir=args.artifacts,
        epochs=args.gan_epochs,
        batch_size=args.batch_size,
        eps=args.eps,
        lambda_ids=args.lambda_ids,
    )


def cmd_train_attack(args) -> None:
    """Train or materialise any supported attack (GAN / FGSM / PGD)."""
    data = cmd_prepare(args)
    model = None
    art = Path(args.artifacts)
    if not ((art / "ids_model.pt").is_file() or (art / "mlp.pt").is_file() or (art / f"{args.ids_model}.pt").is_file() or (art / f"{args.ids_model}.joblib").is_file()):
        train_ids_model(
            data["X_train"],
            data["y_train"],
            X_val=data["X_test"],
            y_val=data["y_test"],
            epochs=args.ids_epochs,
            batch_size=args.batch_size,
            artifacts_dir=args.artifacts,
            model_name=args.ids_model,
        )
    model = load_ids_wrapper(args.artifacts, model_name=args.ids_model)
    attack = args.attack.lower()
    if attack == "gan":
        gan = GANAttack(feature_dim=data["X_train"].shape[1], eps=args.eps)
        gan.fit(
            model,
            data["X_train"],
            data["y_train"],
            data["modifiable_mask"],
            epochs=args.attack_epochs,
            batch_size=args.batch_size,
            lambda_ids=args.lambda_ids,
            artifacts_dir=args.artifacts,
        )
        return
    logging.getLogger(__name__).info("%s is example-time; no separate training step. Use evaluate --attack %s.", attack, attack)


def cmd_evaluate(args) -> dict:
    data = cmd_prepare(args)
    if args.clean_only:
        model = load_ids_wrapper(args.artifacts, model_name=args.ids_model)
        return evaluate_clean(model, data["X_test"], data["y_test"], results_dir=args.results)
    attack = getattr(args, "attack", "gan").lower()
    if attack == "gan" and (Path(args.artifacts) / "gan.pt").is_file() and args.ids_model in {"mlp", "ids"}:
        return evaluate_adversarial(
            data["X_test"],
            data["y_test"],
            data["modifiable_mask"],
            artifacts_dir=args.artifacts,
            results_dir=args.results,
        )
    model = load_ids_wrapper(args.artifacts, model_name=args.ids_model)
    attack_idx = (data["y_test"] == 1).nonzero()[0]
    X_attack = data["X_test"][attack_idx]
    if attack == "gan":
        gan = GANAttack.load(Path(args.artifacts) / "gan.pt")
        X_adv = gan.generate(X_attack, data["modifiable_mask"])
    elif attack == "fgsm":
        X_adv = fgsm_attack(model, X_attack, data["modifiable_mask"], eps=args.eps)
    elif attack == "pgd":
        X_adv = pgd_attack(model, X_attack, data["modifiable_mask"], eps=args.eps, steps=args.pgd_steps)
    else:
        raise SystemExit(f"Unknown attack '{attack}'")
    return evaluate_attack_on_model(
        model, data["X_test"], data["y_test"], X_adv, attack_idx, results_dir=args.results, tag=attack,
    )


def cmd_pipeline(args) -> dict:
    results = run_single_pipeline(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset,
        synthetic=bool(args.synthetic),
        n_benign=args.n_benign,
        n_attack=args.n_attack,
        seed=args.seed,
        test_size=args.test_size,
        val_size=getattr(args, "val_size", 0.0),
        ids_epochs=args.ids_epochs,
        gan_epochs=args.gan_epochs,
        batch_size=args.batch_size,
        eps=args.eps,
        lambda_ids=args.lambda_ids,
        artifacts_dir=args.artifacts,
        results_dir=args.results,
        model_name=getattr(args, "ids_model", "mlp"),
        label_mode=getattr(args, "label_mode", "binary"),
    )
    print("\n=== Adversarial robustness vs IDS ===")
    print(f"Clean accuracy:        {results['clean_accuracy']:.4f}")
    print(f"Adversarial accuracy:  {results['adversarial_accuracy']:.4f}")
    print(f"Accuracy drop:         {results['accuracy_drop']:.4f}")
    print(f"Evasion rate:          {results['evasion_rate']:.4f}")
    print(f"Mean |perturbation|:   {results['mean_abs_perturbation']:.4f}")
    print(f"Mean L2 perturbation:  {results['mean_l2_perturbation']:.4f}")
    print(f"Metrics written to:    {Path(args.results) / 'evaluation_metrics.json'}")
    return results


def cmd_experiment_suite(args) -> dict:
    payload = run_experiment_suite(args.config, root=ROOT)
    print(f"\nSuite rows: {payload['n_rows']}")
    print(f"Metrics:    {Path(args.config).resolve()}")
    print(f"Output dir: results/{payload['name']}/suite_metrics.json")
    return payload


def cmd_check_data(args) -> dict:
    status = check_dataset(args.dataset_name, root=ROOT)
    print_status(status)
    if not status["ready"] and not args.synthetic:
        raise DatasetLayoutError(status["hint"])
    return status


def cmd_setup_data(args) -> dict:
    status = setup_dataset(args.dataset_name, root=ROOT, fetch=bool(getattr(args, "fetch", False)))
    print_status(status)
    return status


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", help="Path to a CSV file or a directory of CSVs.")
    common.add_argument(
        "--dataset-name",
        default="cicids2017",
        help="Schema: cicids2017 | cicids2018 | unsw_nb15 | ciciot2023",
    )
    common.add_argument("--synthetic", action="store_true", help="Generate a same-schema synthetic stand-in.")
    common.add_argument("--n-benign", type=int, default=8000)
    common.add_argument("--n-attack", type=int, default=4000)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--test-size", type=float, default=0.2)
    common.add_argument("--val-size", type=float, default=0.0)
    common.add_argument("--label-mode", choices=["binary", "multiclass"], default="binary")
    common.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    common.add_argument("--results", default=str(ROOT / "results"))
    common.add_argument("--ids-model", default="mlp", help="mlp | deep_mlp | cnn1d | random_forest")
    common.add_argument("--ids-epochs", type=int, default=10)
    common.add_argument("--gan-epochs", type=int, default=40)
    common.add_argument("--attack-epochs", type=int, default=40)
    common.add_argument("--attack", default="gan", help="gan | fgsm | pgd")
    common.add_argument("--pgd-steps", type=int, default=10)
    common.add_argument("--batch-size", type=int, default=256)
    common.add_argument("--eps", type=float, default=0.25, help="Max per-feature perturbation in scaled [0,1] space.")
    common.add_argument("--lambda-ids", type=float, default=4.0, help="Weight on fool-the-IDS loss.")
    common.add_argument("--clean-only", action="store_true")
    common.add_argument("--quick", action="store_true", help="Tiny run for smoke tests.")
    common.add_argument("--fetch", action="store_true", help="For setup-data: try public mirrors.")
    common.add_argument("--max-samples", type=int, default=None, help="Optional row cap after cleaning.")
    common.add_argument("-v", "--verbose", action="store_true")

    p = argparse.ArgumentParser(
        description="Adversarial robustness of ML/DL intrusion detection systems (flow features).",
        parents=[common],
    )
    sub = p.add_subparsers(dest="command")
    for name, help_text in [
        ("prepare", "Build scaled train/val/test arrays and persist the scaler."),
        ("train-ids", "Train an IDS model (mlp, deep_mlp, cnn1d, random_forest)."),
        ("train-gan", "Train the adversarial generator against a frozen IDS."),
        ("train-attack", "Train or materialise an attack (GAN; FGSM/PGD are example-time)."),
        ("evaluate", "Measure clean metrics, evasion rate, ASR, and accuracy drop."),
        ("pipeline", "Prepare + train IDS + train GAN + evaluate."),
        ("check-data", "Verify data/raw layout; fail with download instructions if missing."),
        ("setup-data", "Print layout / optionally fetch a public UNSW training CSV."),
    ]:
        sub.add_parser(name, parents=[common], help=help_text, add_help=True)
    suite = sub.add_parser("experiment-suite", parents=[common], help="Run a YAML/JSON experiment sweep.")
    suite.add_argument("--config", default=str(ROOT / "configs" / "quick.yaml"))
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_quick(args)
    _setup_logging(getattr(args, "verbose", False))
    command = args.command or "pipeline"
    try:
        {
            "prepare": cmd_prepare,
            "train-ids": cmd_train_ids,
            "train-gan": cmd_train_gan,
            "train-attack": cmd_train_attack,
            "evaluate": cmd_evaluate,
            "pipeline": cmd_pipeline,
            "experiment-suite": cmd_experiment_suite,
            "check-data": cmd_check_data,
            "setup-data": cmd_setup_data,
        }[command](args)
    except DatasetLayoutError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
