"""Shared IDS interface so attacks and evaluation can target any model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class IDSModel(ABC):
    """Minimal contract: fit, score P(attack), persist."""

    name: str = "base"
    differentiable: bool = False

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(attack) for each row, shape (n,)."""

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> dict:
        ...

    def attack_logits_torch(self, x):
        """Differentiable attack logit (higher = more attack-like)."""
        raise TypeError(f"{self.name} is not differentiable; use a surrogate for gradient attacks.")

    @abstractmethod
    def save(self, path: str | Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path, **kwargs) -> "IDSModel":
        ...
