"""Evaluation protocol: clean metrics + constrained-attack robustness."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from adv_ids.attacks.gan_attack import craft_gan_attacks
from adv_ids.evaluation.metrics import (
    classification_metrics,
    evasion_metrics,
    mixed_set_accuracy,
    perturbation_stats,
)
from adv_ids.evaluation.plots import save_confusion_matrix, save_roc_curve, save_robustness_bars
from adv_ids.models.base import IDSModel
from adv_ids.training.gan import load_generator
from adv_ids.training.ids import load_ids
from adv_ids.utils.device import get_device
from adv_ids.utils.io import ensure_dir, write_json

logger = logging.getLogger(__name__)


def evaluate_clean(model, X_test, y_test, results_dir="results") -> dict:
    """Works with an IDSModel wrapper or a raw torch module (legacy)."""
    if isinstance(model, IDSModel):
        scores = model.predict_proba(X_test)
        pred = (scores >= 0.5).astype(int)
    else:
        import torch

        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(np.asarray(X_test, dtype=np.float32)).to(device))
            scores = torch.sigmoid(logits).cpu().numpy()
        pred = (scores >= 0.5).astype(int)
    metrics = classification_metrics(y_test, pred, scores)
    logger.info("Clean IDS accuracy=%.4f f1=%.4f auc=%s", metrics["accuracy"], metrics["f1"], metrics["roc_auc"])
    out = ensure_dir(results_dir)
    save_confusion_matrix(metrics["confusion_matrix"], out / "confusion_matrix_clean.png")
    if metrics["roc_auc"] is not None:
        save_roc_curve(y_test, scores, out / "roc_curve_clean.png", metrics["roc_auc"])
    return metrics


def evaluate_attack_on_model(
    model: IDSModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_adv_attack: np.ndarray,
    attack_index: np.ndarray | None = None,
    results_dir: str | Path | None = None,
    tag: str = "attack",
) -> dict:
    if attack_index is None:
        attack_index = np.where(y_test == 1)[0]
    clean = evaluate_clean(model, X_test, y_test, results_dir=results_dir or "results")
    pred_clean = (model.predict_proba(X_test) >= 0.5).astype(int)
    X_mixed = X_test.copy()
    X_mixed[attack_index] = X_adv_attack
    pred_mixed = (model.predict_proba(X_mixed) >= 0.5).astype(int)
    ev = evasion_metrics(y_test, pred_clean, pred_mixed, attack_index)
    mixed = mixed_set_accuracy(y_test, pred_mixed, clean["accuracy"])
    pert = perturbation_stats(X_test[attack_index], X_adv_attack)
    scores_after = model.predict_proba(X_adv_attack)
    results = {
        "tag": tag,
        "clean": {k: v for k, v in clean.items() if k != "classification_report"},
        "clean_accuracy": clean["accuracy"],
        "clean_f1": clean["f1"],
        "clean_roc_auc": clean["roc_auc"],
        **ev,
        **mixed,
        **pert,
        "mean_p_attack_after": float(scores_after.mean()),
    }
    if results_dir is not None:
        out = ensure_dir(results_dir)
        payload = {k: v for k, v in results.items() if k != "clean"}
        payload["clean"] = results["clean"]
        write_json(out / f"evaluation_{tag}.json", payload)
        save_robustness_bars(
            clean["accuracy"],
            results["adversarial_accuracy"],
            results["evasion_rate"],
            out / f"robustness_{tag}.png",
            title=f"IDS robustness ({tag})",
        )
    logger.info(
        "%s  evasion=%.3f  asr=%.3f  clean=%.3f  adv=%.3f  drop=%.3f  l2=%.4f",
        tag,
        results["evasion_rate"],
        results["attack_success_rate"],
        results["clean_accuracy"],
        results["adversarial_accuracy"],
        results["accuracy_drop"],
        results["mean_l2_perturbation"],
    )
    return results


def evaluate_adversarial(
    X_test,
    y_test,
    modifiable_mask,
    artifacts_dir="artifacts",
    results_dir="results",
) -> dict:
    """Legacy GAN-vs-MLP evaluation used by ``run.py pipeline``."""
    device = get_device()
    ids = load_ids(artifacts_dir, device=device)
    G, gan_meta = load_generator(artifacts_dir, device=device)
    clean = evaluate_clean(ids, X_test, y_test, results_dir=results_dir)
    attack_idx = np.where(y_test == 1)[0]
    X_attack = X_test[attack_idx]
    if len(X_attack) == 0:
        raise ValueError("No attack samples in the test set.")
    import torch

    ids.eval()
    with torch.no_grad():
        logits = ids(torch.from_numpy(X_attack).float().to(device))
        prob_clean_attack = torch.sigmoid(logits).cpu().numpy()
    pred_clean_attack = (prob_clean_attack >= 0.5).astype(int)
    X_adv = craft_gan_attacks(G, X_attack, modifiable_mask, gan_meta["latent_dim"], device)
    with torch.no_grad():
        logits = ids(torch.from_numpy(X_adv).float().to(device))
        prob_adv = torch.sigmoid(logits).cpu().numpy()
    pred_adv = (prob_adv >= 0.5).astype(int)
    originally_detected = pred_clean_attack == 1
    evaded = originally_detected & (pred_adv == 0)
    evasion_rate = float(evaded.sum() / max(originally_detected.sum(), 1))
    X_mixed = X_test.copy()
    X_mixed[attack_idx] = X_adv
    with torch.no_grad():
        logits = ids(torch.from_numpy(X_mixed).float().to(device))
        pred_mixed = (torch.sigmoid(logits).cpu().numpy() >= 0.5).astype(int)
    from sklearn.metrics import accuracy_score

    adv_accuracy = float(accuracy_score(y_test, pred_mixed))
    pert = perturbation_stats(X_attack, X_adv, modifiable_mask)
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
        **pert,
    }
    out = ensure_dir(results_dir)
    payload = {k: v for k, v in results.items() if k != "clean"}
    payload["clean_accuracy"] = clean["accuracy"]
    payload["clean_f1"] = clean["f1"]
    payload["clean_roc_auc"] = clean["roc_auc"]
    write_json(out / "evaluation_metrics.json", payload)
    save_robustness_bars(
        clean["accuracy"],
        adv_accuracy,
        evasion_rate,
        out / "robustness_summary.png",
        title="IDS robustness under adversarial GAN traffic",
    )
    logger.info(
        "Evasion rate=%.3f  clean_acc=%.3f  adv_acc=%.3f  drop=%.3f",
        evasion_rate, clean["accuracy"], adv_accuracy, results["accuracy_drop"],
    )
    return results
