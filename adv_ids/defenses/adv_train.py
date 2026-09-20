"""Madry-style adversarial training on the MLP (PGD on attack rows)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from adv_ids.attacks.pgd import pgd_attack
from adv_ids.models.mlp import MLPIDS
from adv_ids.utils.device import get_device
from adv_ids.utils.io import write_json

logger = logging.getLogger(__name__)


def train_adversarial_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    mask: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    *,
    hidden: tuple[int, ...] = (128, 64),
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    eps: float = 0.15,
    pgd_steps: int = 5,
    adv_ratio: float = 0.5,
    artifacts_dir: str | Path | None = None,
    name: str = "mlp_advtrain",
) -> dict:
    """Train an MLP on a mix of clean flows and PGD-perturbed attacks.

    Only attack-labelled rows are adversarially perturbed (evasion setting).
    Benign rows stay clean so the detector does not drift toward false positives.
    """
    device = get_device()
    model = MLPIDS(input_dim=X_train.shape[1], hidden=hidden, device=device)
    model.name = name
    n_pos = max(int((y_train == 1).sum()), 1)
    n_neg = max(int((y_train == 0).sum()), 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.module.parameters(), lr=lr)
    ds = TensorDataset(
        torch.from_numpy(np.asarray(X_train, dtype=np.float32)),
        torch.from_numpy(np.asarray(y_train, dtype=np.float32)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=len(ds) > batch_size)
    history = {"train_loss": [], "val_acc": []}
    best_val = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.module.train()
        running = 0.0
        seen = 0
        for xb, yb in loader:
            xb_np = xb.numpy()
            yb_np = yb.numpy()
            attack_idx = np.where(yb_np >= 0.5)[0]
            if len(attack_idx) > 0:
                n_adv = max(1, int(len(attack_idx) * adv_ratio))
                chosen = attack_idx[:n_adv]
                xb_adv = pgd_attack(
                    model,
                    xb_np[chosen],
                    mask,
                    eps=eps,
                    steps=pgd_steps,
                    random_start=True,
                    batch_size=max(len(chosen), 1),
                )
                xb_np = xb_np.copy()
                xb_np[chosen] = xb_adv
            xb_t = torch.from_numpy(xb_np).to(device)
            yb_t = yb.to(device)
            opt.zero_grad()
            loss = criterion(model.module(xb_t), yb_t)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
            seen += len(xb)
        train_loss = running / max(seen, 1)
        history["train_loss"].append(train_loss)
        val_acc = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_acc = float((model.predict(X_val) == y_val).mean())
            history["val_acc"].append(val_acc)
            if val_acc >= best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.module.state_dict().items()}
        logger.info("AdvTrain epoch %d/%d loss=%.4f val_acc=%s", epoch, epochs, train_loss, val_acc)

    if best_state is not None:
        model.module.load_state_dict(best_state)
    if artifacts_dir is not None:
        model.save(Path(artifacts_dir) / f"{name}.pt")
        write_json(
            Path(artifacts_dir) / f"{name}_train_metrics.json",
            {
                "best_val_acc": best_val if best_val >= 0 else None,
                "final_train_loss": history["train_loss"][-1],
                "eps": eps,
                "pgd_steps": pgd_steps,
                "defense": "adversarial_training",
            },
        )
    return {"model": model, "history": history, "best_val_acc": best_val if best_val >= 0 else None}
