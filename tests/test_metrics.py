"""Metric-definition tests (evasion rate, ASR, accuracy drop)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adv_ids.evaluation.metrics import (
    classification_metrics,
    evasion_metrics,
    mixed_set_accuracy,
    perturbation_stats,
)


def test_evasion_rate_only_uses_originally_detected():
    y = np.array([0, 0, 1, 1, 1, 1, 1])
    pred_clean = np.array([0, 0, 1, 1, 1, 1, 0])  # 4 of 5 attacks detected
    pred_mixed = np.array([0, 0, 0, 0, 1, 1, 0])  # 2 of those 4 evaded
    attack_index = np.array([2, 3, 4, 5, 6])
    ev = evasion_metrics(y, pred_clean, pred_mixed, attack_index)
    assert ev["attack_detected_before"] == 4
    assert ev["evasion_rate"] == 0.5
    assert ev["attack_success_rate"] == 0.6  # 3 of 5 attack rows now benign


def test_accuracy_drop():
    y = np.array([0, 0, 1, 1])
    pred_mixed = np.array([0, 1, 0, 1])
    out = mixed_set_accuracy(y, pred_mixed, clean_accuracy=1.0)
    assert out["adversarial_accuracy"] == 0.5
    assert out["accuracy_drop"] == 0.5


def test_perturbation_stats_and_frozen_change():
    x = np.zeros((3, 4), dtype=np.float32)
    x_adv = np.array([[0.1, 0.0, 0.2, 0.0], [0.0, 0.0, 0.4, 0.0], [0.2, 0.0, 0.0, 0.0]], dtype=np.float32)
    mask = np.array([1, 0, 1, 0], dtype=np.float32)
    stats = perturbation_stats(x, x_adv, mask)
    assert stats["max_frozen_feature_change"] == 0.0
    assert stats["mean_l2_perturbation"] > 0


def test_classification_metrics_binary():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    scores = np.array([0.1, 0.6, 0.9, 0.8])
    m = classification_metrics(y, pred, scores)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert m["roc_auc"] is not None
