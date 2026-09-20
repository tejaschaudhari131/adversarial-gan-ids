"""Clean and adversarial IDS metrics (no I/O)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return None


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None = None,
    target_names: list[str] | None = None,
) -> dict[str, Any]:
    target_names = target_names or ["BENIGN", "ATTACK"]
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_samples": int(len(y_true)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=target_names[: len(np.unique(y_true))], zero_division=0
        ),
    }
    if scores is not None:
        metrics["roc_auc"] = _safe_auc(y_true, scores)
    else:
        metrics["roc_auc"] = None
    return metrics


def perturbation_stats(x: np.ndarray, x_adv: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    delta = x_adv - x
    if mask is not None:
        frozen = np.asarray(mask) < 0.5
        max_frozen = float(np.max(np.abs(delta[:, frozen]))) if frozen.any() else 0.0
    else:
        max_frozen = 0.0
    l2 = np.linalg.norm(delta, axis=1)
    linf = np.max(np.abs(delta), axis=1)
    return {
        "mean_abs_perturbation": float(np.abs(delta).mean()),
        "mean_l2_perturbation": float(l2.mean()),
        "median_l2_perturbation": float(np.median(l2)),
        "mean_linf_perturbation": float(linf.mean()),
        "max_frozen_feature_change": max_frozen,
    }


def evasion_metrics(
    y_true: np.ndarray,
    pred_clean: np.ndarray,
    pred_adv: np.ndarray,
    attack_index: np.ndarray,
) -> dict[str, Any]:
    """Evasion rate is computed on originally detected attacks only."""
    if len(attack_index) == 0:
        raise ValueError("No attack samples in the evaluation set.")
    pred_clean_a = pred_clean[attack_index]
    pred_adv_a = pred_adv[attack_index]
    originally_detected = pred_clean_a == 1
    evaded = originally_detected & (pred_adv_a == 0)
    n_detected = int(originally_detected.sum())
    evasion_rate = float(evaded.sum() / max(n_detected, 1))
    # Attack success rate: share of all attack rows labelled benign after the attack.
    asr = float((pred_adv_a == 0).mean())
    return {
        "n_attack": int(len(attack_index)),
        "attack_detected_before": n_detected,
        "attack_detected_after": int((pred_adv_a == 1).sum()),
        "evasion_rate": evasion_rate,
        "attack_success_rate": asr,
    }


def mixed_set_accuracy(y_true: np.ndarray, pred_mixed: np.ndarray, clean_accuracy: float) -> dict[str, float]:
    adv_acc = float(accuracy_score(y_true, pred_mixed))
    return {
        "adversarial_accuracy": adv_acc,
        "accuracy_drop": float(clean_accuracy - adv_acc),
    }
