"""Train an adversarial GAN that perturbs attack flows to evade a frozen IDS."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.discriminator import build_discriminator
from models.generator import apply_perturbation, build_generator
from training.train_ids import load_ids

logger = logging.getLogger(__name__)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_adversarial_gan(
    X_train: np.ndarray,
    y_train: np.ndarray,
    modifiable_mask: np.ndarray,
    artifacts_dir: str | Path = "artifacts",
    latent_dim: int = 32,
    eps: float = 0.15,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-4,
    lambda_ids: float = 2.0,
    lambda_pert: float = 0.5,
    log_interval: int = 5,
) -> dict:
    device = _device()
    artifacts = Path(artifacts_dir)
    ids = load_ids(artifacts, device=device)
    for p in ids.parameters():
        p.requires_grad_(False)
    ids.eval()

    feature_dim = X_train.shape[1]
    G = build_generator(feature_dim, latent_dim=latent_dim, eps=eps).to(device)
    D = build_discriminator(feature_dim).to(device)
    mask = torch.from_numpy(modifiable_mask.astype(np.float32)).to(device)

    benign = torch.from_numpy(X_train[y_train == 0]).float()
    attack = torch.from_numpy(X_train[y_train == 1]).float()
    if len(benign) == 0 or len(attack) == 0:
        raise ValueError("Need both benign and attack rows to train the adversarial GAN.")

    attack_loader = DataLoader(TensorDataset(attack), batch_size=min(batch_size, len(attack)), shuffle=True, drop_last=False)
    bce = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    history = {"d_loss": [], "g_loss": [], "ids_score": []}

    for epoch in range(1, epochs + 1):
        d_run = g_run = ids_run = 0.0
        n_batches = 0
        for (xb,) in attack_loader:
            xb = xb.to(device)
            bs = xb.size(0)
            z = torch.randn(bs, latent_dim, device=device)
            idx = torch.randint(0, len(benign), (bs,))
            real_benign = benign[idx].to(device)
            with torch.no_grad():
                fake = apply_perturbation(xb, G(xb, z), mask)
            opt_d.zero_grad()
            d_real = D(real_benign)
            d_fake = D(fake.detach())
            d_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            d_loss.backward()
            opt_d.step()
            z = torch.randn(bs, latent_dim, device=device)
            opt_g.zero_grad()
            delta = G(xb, z)
            fake = apply_perturbation(xb, delta, mask)
            g_adv = bce(D(fake), torch.ones_like(d_real))
            ids_logits = ids(fake)
            ids_loss = torch.sigmoid(ids_logits).mean()
            pert_loss = (delta * mask).pow(2).mean()
            g_loss = g_adv + lambda_ids * ids_loss + lambda_pert * pert_loss
            g_loss.backward()
            opt_g.step()
            d_run += d_loss.item()
            g_run += g_loss.item()
            ids_run += ids_loss.item()
            n_batches += 1
        history["d_loss"].append(d_run / max(n_batches, 1))
        history["g_loss"].append(g_run / max(n_batches, 1))
        history["ids_score"].append(ids_run / max(n_batches, 1))
        if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
            logger.info("GAN epoch %d/%d  D=%.4f  G=%.4f  IDS_P(attack)=%.4f", epoch, epochs, history["d_loss"][-1], history["g_loss"][-1], history["ids_score"][-1])

    ckpt = artifacts / "gan.pt"
    torch.save({"generator": G.state_dict(), "discriminator": D.state_dict(), "feature_dim": feature_dim, "latent_dim": latent_dim, "eps": eps}, ckpt)
    with open(artifacts / "gan_train_metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"final_d_loss": history["d_loss"][-1], "final_g_loss": history["g_loss"][-1], "final_ids_p_attack": history["ids_score"][-1], "epochs": epochs, "eps": eps, "lambda_ids": lambda_ids}, fh, indent=2)
    logger.info("Saved GAN to %s", ckpt)
    return {"generator": G, "discriminator": D, "history": history, "checkpoint": str(ckpt)}


def load_generator(artifacts_dir: str | Path = "artifacts", device: torch.device | None = None):
    device = device or _device()
    payload = torch.load(Path(artifacts_dir) / "gan.pt", map_location=device, weights_only=True)
    G = build_generator(payload["feature_dim"], latent_dim=payload["latent_dim"], eps=payload["eps"]).to(device)
    G.load_state_dict(payload["generator"])
    G.eval()
    return G, payload
