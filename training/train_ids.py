"""Train the deep learning IDS on benign vs attack flows."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.ids_model import build_ids_model

logger = logging.getLogger(__name__)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_ids_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 12,
    batch_size: int = 256,
    lr: float = 1e-3,
    artifacts_dir: str | Path = "artifacts",
) -> dict:
    device = _device()
    model = build_ids_model(input_dim=X_train.shape[1]).to(device)

    n_pos = max(int((y_train == 1).sum()), 1)
    n_neg = max(int((y_train == 0).sum()), 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    history = {"train_loss": [], "val_acc": []}
    best_val = -1.0
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    ckpt = artifacts / "ids_model.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        train_loss = running / len(train_ds)
        history["train_loss"].append(train_loss)

        val_acc = None
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                logits = model(torch.from_numpy(X_val).float().to(device))
                preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)
            val_acc = float((preds == y_val).mean())
            history["val_acc"].append(val_acc)
            if val_acc >= best_val:
                best_val = val_acc
                torch.save({"model_state": model.state_dict(), "input_dim": X_train.shape[1]}, ckpt)
        logger.info("IDS epoch %d/%d loss=%.4f val_acc=%s", epoch, epochs, train_loss, val_acc)

    if best_val < 0:
        torch.save({"model_state": model.state_dict(), "input_dim": X_train.shape[1]}, ckpt)

    payload = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(payload["model_state"])
    metrics = {"best_val_acc": best_val if best_val >= 0 else None, "history": history, "checkpoint": str(ckpt)}
    with open(artifacts / "ids_train_metrics.json", "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in metrics.items() if k != "history"} | {"final_train_loss": history["train_loss"][-1]}, fh, indent=2)
    return {"model": model, **metrics}


def load_ids(artifacts_dir: str | Path = "artifacts", device: torch.device | None = None) -> torch.nn.Module:
    device = device or _device()
    payload = torch.load(Path(artifacts_dir) / "ids_model.pt", map_location=device, weights_only=True)
    model = build_ids_model(input_dim=payload["input_dim"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model
