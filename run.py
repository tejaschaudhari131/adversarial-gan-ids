#!/usr/bin/env python3
"""Command-line entry point for the rebuilt adversarial GAN vs IDS pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.data_preprocessing import generate_synthetic_cicids2017, prepare_dataset
from training.train_gan import train_adversarial_gan
from training.train_ids import train_ids_model
from evaluation.evaluate_model import evaluate_adversarial, evaluate_clean
from training.train_ids import load_ids


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )


def _resolve_dataset(args) -> Path:
    if args.dataset:
        path = Path(args.dataset)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path
    if args.synthetic:
        path = ROOT / "data" / "synthetic_cicids2017.csv"
        generate_synthetic_cicids2017(
            n_benign=args.n_benign,
            n_attack=args.n_attack,
            seed=args.seed,
            output_path=path,
        )
        return path
    raise SystemExit("Provide --dataset PATH or pass --synthetic to generate a stand-in CIC-IDS2017 CSV.")


def cmd_prepare(args) -> dict:
    dataset = _resolve_dataset(args)
    return prepare_dataset(dataset, test_size=args.test_size, random_state=args.seed, artifacts_dir=args.artifacts)


def cmd_train_ids(args) -> None:
    data = cmd_prepare(args)
    train_ids_model(
        data["X_train"],
        data["y_train"],
        X_val=data["X_test"],
        y_val=data["y_test"],
        epochs=args.ids_epochs,
        batch_size=args.batch_size,
        artifacts_dir=args.artifacts,
    )


def cmd_train_gan(args) -> None:
    data = cmd_prepare(args)
    if not (Path(args.artifacts) / "ids_model.pt").is_file():
        logging.getLogger(__name__).info("No IDS checkpoint found — training IDS first.")
        train_ids_model(
            data["X_train"],
            data["y_train"],
            X_val=data["X_test"],
            y_val=data["y_test"],
            epochs=args.ids_epochs,
            batch_size=args.batch_size,
            artifacts_dir=args.artifacts,
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


def cmd_evaluate(args) -> dict:
    data = cmd_prepare(args)
    if args.clean_only:
        model = load_ids(args.artifacts)
        return evaluate_clean(model, data["X_test"], data["y_test"], results_dir=args.results)
    return evaluate_adversarial(
        data["X_test"],
        data["y_test"],
        data["modifiable_mask"],
        artifacts_dir=args.artifacts,
        results_dir=args.results,
    )


def cmd_pipeline(args) -> dict:
    data = cmd_prepare(args)
    train_ids_model(
        data["X_train"],
        data["y_train"],
        X_val=data["X_test"],
        y_val=data["y_test"],
        epochs=args.ids_epochs,
        batch_size=args.batch_size,
        artifacts_dir=args.artifacts,
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
    results = evaluate_adversarial(
        data["X_test"],
        data["y_test"],
        data["modifiable_mask"],
        artifacts_dir=args.artifacts,
        results_dir=args.results,
    )
    print("\n=== Adversarial GAN vs IDS ===")
    print(f"Clean accuracy:        {results['clean']['accuracy']:.4f}")
    print(f"Adversarial accuracy:  {results['adversarial_accuracy']:.4f}")
    print(f"Accuracy drop:         {results['accuracy_drop']:.4f}")
    print(f"Evasion rate:          {results['evasion_rate']:.4f}")
    print(f"Mean |perturbation|:   {results['mean_abs_perturbation']:.4f}")
    print(f"Metrics written to:    {Path(args.results) / 'evaluation_metrics.json'}")
    return results


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", help="Path to CIC-IDS2017 CSV (MachineLearningCSV format).")
    common.add_argument("--synthetic", action="store_true", help="Generate a CIC-IDS2017-shaped synthetic dataset.")
    common.add_argument("--n-benign", type=int, default=8000)
    common.add_argument("--n-attack", type=int, default=4000)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--test-size", type=float, default=0.2)
    common.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    common.add_argument("--results", default=str(ROOT / "results"))
    common.add_argument("--ids-epochs", type=int, default=10)
    common.add_argument("--gan-epochs", type=int, default=40)
    common.add_argument("--batch-size", type=int, default=256)
    common.add_argument("--eps", type=float, default=0.25, help="Max per-feature perturbation in scaled [0,1] space.")
    common.add_argument("--lambda-ids", type=float, default=4.0, help="Weight on fool-the-IDS loss.")
    common.add_argument("--clean-only", action="store_true")
    common.add_argument("--quick", action="store_true", help="Tiny run for smoke tests.")
    common.add_argument("-v", "--verbose", action="store_true")

    p = argparse.ArgumentParser(
        description="Adversarial GAN against a deep learning IDS (CIC-IDS2017).",
        parents=[common],
    )
    sub = p.add_subparsers(dest="command")
    for name, help_text in [
        ("prepare", "Build scaled train/test arrays and persist the scaler."),
        ("train-ids", "Train the deep IDS."),
        ("train-gan", "Train the adversarial generator against a frozen IDS."),
        ("evaluate", "Measure clean accuracy, evasion rate, and accuracy drop."),
        ("pipeline", "Prepare + train IDS + train GAN + evaluate."),
    ]:
        sub.add_parser(name, parents=[common], help=help_text, add_help=True)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.quick:
        args.n_benign = min(args.n_benign, 1500)
        args.n_attack = min(args.n_attack, 800)
        args.ids_epochs = min(args.ids_epochs, 4)
        args.gan_epochs = min(args.gan_epochs, 8)
        args.batch_size = min(args.batch_size, 128)
    _setup_logging(args.verbose)
    command = args.command or "pipeline"
    {
        "prepare": cmd_prepare,
        "train-ids": cmd_train_ids,
        "train-gan": cmd_train_gan,
        "evaluate": cmd_evaluate,
        "pipeline": cmd_pipeline,
    }[command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
