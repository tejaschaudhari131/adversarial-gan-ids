"""PyTorch IDS helpers shared by MLP and 1D-CNN wrappers."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from adv_ids.models.base import IDSModel
from adv_ids.utils.device import get_device
from adv_ids.utils.io import write_json

logger = logging.getLogger(__name__)


class TorchIDS(IDSModel):
    differentiable = True
    module: nn.Module

    def __init__(self, input_dim: int, device: torch.device | None = None):
        self.input_dim = input_dim
        self.device = device or get_device()
        self.module = self.build_module(input_dim).to(self.device)
        self.history: dict = {"train_loss": [], "val_acc": []}

    def build_module(self, input_dim: int) -> nn.Module:
        raise NotImplementedError

    def extra_state(self) -> dict:
        return {}

    def load_extra_state(self, payload: dict) -> None:
        return

    def attack_logits_torch(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    def predict_proba(self, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        self.module.eval()
        probs = []
        tensor = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            for start in range(0, len(tensor), batch_size):
                xb = tensor[start : start + batch_size].to(self.device)
                logits = self.module(xb)
                probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probs, axis=0).reshape(-1)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 12,
        batch_size: int = 256,
        lr: float = 1e-3,
        artifacts_dir: str | Path | None = None,
        **kwargs,
    ) -> dict:
        device = self.device
        self.module.to(device)
        n_pos = max(int((y_train == 1).sum()), 1)
        n_neg = max(int((y_train == 0).sum()), 1)
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self.module.parameters(), lr=lr)
        train_ds = TensorDataset(
            torch.from_numpy(np.asarray(X_train, dtype=np.float32)),
            torch.from_numpy(np.asarray(y_train, dtype=np.float32)),
        )
        drop_last = len(train_ds) > batch_size
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)

        best_val = -1.0
        best_state = {k: v.detach().cpu().clone() for k, v in self.module.state_dict().items()}
        self.history = {"train_loss": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            self.module.train()
            running = 0.0
            seen = 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = criterion(self.module(xb), yb)
                loss.backward()
                opt.step()
                running += loss.item() * len(xb)
                seen += len(xb)
            train_loss = running / max(seen, 1)
            self.history["train_loss"].append(train_loss)

            val_acc = None
            if X_val is not None and y_val is not None and len(X_val) > 0:
                pred = self.predict(X_val)
                val_acc = float((pred == y_val).mean())
                self.history["val_acc"].append(val_acc)
                if val_acc >= best_val:
                    best_val = val_acc
                    best_state = {k: v.detach().cpu().clone() for k, v in self.module.state_dict().items()}
            logger.info("%s epoch %d/%d loss=%.4f val_acc=%s", self.name, epoch, epochs, train_loss, val_acc)

        if best_val >= 0:
            self.module.load_state_dict(best_state)
        if artifacts_dir is not None:
            ckpt = Path(artifacts_dir) / f"{self.name}.pt"
            self.save(ckpt)
            write_json(
                Path(artifacts_dir) / f"{self.name}_train_metrics.json",
                {
                    "best_val_acc": best_val if best_val >= 0 else None,
                    "final_train_loss": self.history["train_loss"][-1] if self.history["train_loss"] else None,
                    "epochs": epochs,
                    "arch": self.name,
                },
            )
        return {
            "best_val_acc": best_val if best_val >= 0 else None,
            "history": self.history,
            "model": self,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "arch": self.name,
                "input_dim": self.input_dim,
                "model_state": self.module.state_dict(),
                **self.extra_state(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: torch.device | None = None, **kwargs) -> "TorchIDS":
        device = device or get_device()
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        obj = cls(input_dim=int(payload["input_dim"]), device=device, **kwargs)
        obj.load_extra_state(payload)
        obj.module.load_state_dict(payload["model_state"])
        obj.module.to(device)
        obj.module.eval()
        return obj
