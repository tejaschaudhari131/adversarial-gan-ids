"""Matplotlib helpers for IDS robustness figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def save_confusion_matrix(cm, path: str | Path, title: str = "Clean IDS confusion matrix") -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    arr = np.asarray(cm)
    ax.imshow(arr, cmap="Blues")
    ax.set_xticks([0, 1], ["BENIGN", "ATTACK"])
    ax.set_yticks([0, 1], ["BENIGN", "ATTACK"])
    ax.set_title(title)
    for (i, j), v in np.ndenumerate(arr):
        ax.text(j, i, str(int(v)), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_roc_curve(y_true, scores, path: str | Path, auc: float | None) -> None:
    fpr, tpr, _ = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    label = f"AUC={auc:.3f}" if auc is not None else "ROC"
    ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_robustness_bars(clean_acc, adv_acc, evasion_rate, path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Clean accuracy", "Adversarial accuracy", "Evasion rate"]
    values = [clean_acc, adv_acc, evasion_rate]
    bars = ax.bar(labels, values, color=["#2a9d8f", "#e76f51", "#264653"])
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_attack_comparison(rows: list[dict], path: str | Path) -> None:
    if not rows:
        return
    labels = [r.get("label", r.get("attack", "?")) for r in rows]
    values = [float(r.get("evasion_rate", 0.0)) for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(labels)), 4))
    bars = ax.bar(labels, values, color="#264653")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Evasion rate")
    ax.set_title("Constrained attack comparison")
    ax.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_transfer_heatmap(matrix: np.ndarray, row_labels, col_labels, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(1.4 * len(col_labels) + 2, 1.1 * len(row_labels) + 2))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=20)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xlabel("Target")
    ax.set_ylabel("Surrogate")
    ax.set_title("Transfer evasion rate")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
