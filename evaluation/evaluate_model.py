"""Clean IDS metrics plus adversarial evasion rate and accuracy drop."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)

from models.generator import apply_perturbation
from training.train_gan import load_generator
from training.train_ids import load_ids

logger = logging.getLogger(__name__)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _predict(model, X: np.ndarray, device: torch.device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float().to(device))
        prob = torch.sigmoid(logits).cpu().numpy()
    return (prob >= 0.5).astype(int), prob


def evaluate_clean(model, X_test, y_test, results_dir="results") -> dict:
    device = next(model.parameters()).device
    pred, prob = _predict(model, X_test, device)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)) if len(np.unique(y_test)) == 2 else None,
        "n_samples": int(len(y_test)),
        "classification_report": classification_report(y_test, pred, target_names=["BENIGN", "ATTACK"], zero_division=0),
    }
    logger.info("Clean IDS accuracy=%.4f f1=%.4f auc=%s", metrics["accuracy"], metrics["f1"], metrics["roc_auc"])
    results = Path(results_dir)
    results.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["BENIGN", "ATTACK"])
    ax.set_yticks([0, 1], ["BENIGN", "ATTACK"])
    ax.set_title("Clean IDS confusion matrix")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(results / "confusion_matrix_clean.png", dpi=140)
    plt.close(fig)
    if metrics["roc_auc"] is not None:
        fpr, tpr, _ = roc_curve(y_test, prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
        fig.tight_layout()
        fig.savefig(results / "roc_curve_clean.png", dpi=140)
        plt.close(fig)
    return metrics


def craft_adversarial_attacks(generator, X_attack, mask, latent_dim, device):
    generator.eval()
    with torch.no_grad():
        x = torch.from_numpy(X_attack).float().to(device)
        z = torch.randn(len(X_attack), latent_dim, device=device)
        fake = apply_perturbation(x, generator(x, z), torch.from_numpy(mask).float().to(device))
    return fake.cpu().numpy()


def evaluate_adversarial(X_test, y_test, modifiable_mask, artifacts_dir="artifacts", results_dir="results") -> dict:
    device = _device()
    ids = load_ids(artifacts_dir, device=device)
    G, gan_meta = load_generator(artifacts_dir, device=device)
    clean = evaluate_clean(ids, X_test, y_test, results_dir=results_dir)
    attack_idx = np.where(y_test == 1)[0]
    X_attack = X_test[attack_idx]
    if len(X_attack) == 0:
        raise ValueError("No attack samples in the test set.")
    pred_clean_attack, prob_clean_attack = _predict(ids, X_attack, device)
    X_adv = craft_adversarial_attacks(G, X_attack, modifiable_mask, gan_meta["latent_dim"], device)
    pred_adv, prob_adv = _predict(ids, X_adv, device)
    originally_detected = pred_clean_attack == 1
    evaded = originally_detected & (pred_adv == 0)
    evasion_rate = float(evaded.sum() / max(originally_detected.sum(), 1))
    X_mixed = X_test.copy()
    X_mixed[attack_idx] = X_adv
    pred_mixed, _ = _predict(ids, X_mixed, device)
    adv_accuracy = float(accuracy_score(y_test, pred_mixed))
    results = {
        "clean": clean,
        "attack_detected_before": int(pred_clean_attack.sum()),
        "attack_detected_after": int(pred_adv.sum()),
        "n_attack_test": int(len(X_attack)),
        "mean_p_attack_before": float(prob_clean_attack.mean()),
        "mean_p_attack_after": float(prob_adv.mean()),
        "evasion_rate": evasion_rate,
        "adversarial_accuracy": adv_accuracy,
        "accuracy_drop": float(clean["accuracy"] - adv_accuracy),
        "mean_abs_perturbation": float(np.abs(X_adv - X_attack).mean()),
        "mean_l2_perturbation": float(np.linalg.norm(X_adv - X_attack, axis=1).mean()),
    }
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in results.items() if k != "clean"}
    payload["clean_accuracy"] = clean["accuracy"]
    payload["clean_f1"] = clean["f1"]
    payload["clean_roc_auc"] = clean["roc_auc"]
    with open(out / "evaluation_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Clean accuracy", "Adversarial accuracy", "Evasion rate"]
    values = [clean["accuracy"], adv_accuracy, evasion_rate]
    bars = ax.bar(labels, values, color=["#2a9d8f", "#e76f51", "#264653"])
    ax.set_ylim(0, 1.05)
    ax.set_title("IDS robustness under adversarial GAN traffic")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(out / "robustness_summary.png", dpi=140)
    plt.close(fig)
    logger.info("Evasion rate=%.3f  clean_acc=%.3f  adv_acc=%.3f  drop=%.3f", evasion_rate, clean["accuracy"], adv_accuracy, results["accuracy_drop"])
    return results
