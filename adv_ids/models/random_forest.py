"""Sklearn Random Forest IDS (non-differentiable baseline)."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from adv_ids.models.base import IDSModel

logger = logging.getLogger(__name__)


class RandomForestIDS(IDSModel):
    name = "random_forest"
    differentiable = False

    def __init__(
        self,
        input_dim: int | None = None,
        n_estimators: int = 120,
        max_depth: int | None = 20,
        random_state: int = 42,
        n_jobs: int = -1,
        class_weight: str | None = "balanced",
    ):
        self.input_dim = input_dim
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight=class_weight,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, artifacts_dir=None, **kwargs) -> dict:
        self.input_dim = X_train.shape[1]
        self.model.fit(X_train, y_train)
        val_acc = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_acc = float((self.predict(X_val) == y_val).mean())
        logger.info("RandomForest fitted n_estimators=%d val_acc=%s", self.model.n_estimators, val_acc)
        if artifacts_dir is not None:
            self.save(Path(artifacts_dir) / f"{self.name}.joblib")
        return {"best_val_acc": val_acc, "model": self}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.model.predict_proba(X)
        classes = list(self.model.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        return np.zeros(len(X), dtype=np.float64)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "input_dim": self.input_dim, "arch": self.name}, path)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "RandomForestIDS":
        payload = joblib.load(path)
        obj = cls(input_dim=payload.get("input_dim"))
        obj.model = payload["model"]
        return obj
