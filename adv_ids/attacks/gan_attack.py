"""Conditional GAN evasion attack against a frozen differentiable IDS."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from adv_ids.models.base import IDSModel
from adv_ids.models.gan import apply_perturbation, build_discriminator, build_generator
from adv_ids.utils.device import get_device
from adv_ids.utils.io import write_json

logger = logging.getLogger(__name__)


class GANAttack:
    name = "gan"

    def __init__(
        self,
        feature_dim: int,
        latent_dim: int = 32,
        eps: float = 0.15,
        device: torch.device | None = None,
    ):
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.eps = eps
        self.device = device or get_device()
        self.generator = build_generator(feature_dim, latent_dim=latent_dim, eps=eps).to(self.device)
        self.discriminator = build_discriminator(feature_dim).to(self.device)
        self.mask: torch.Tensor | None = None

    def fit(
        self,
        model: IDSModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        mask: np.ndarray,
        *,
        epochs: int = 40,
        batch_size: int = 256,
        lr: float = 1e-4,
        lambda_ids: float = 4.0,
        lambda_pert: float = 0.5,
        artifacts_dir: str | Path | None = None,
        log_interval: int = 5,
    ) -> dict:
        if not model.differentiable:
            raise TypeError("The GAN attack needs a differentiable frozen IDS (use a surrogate for RF).")
        device = self.device
        ids = model.module.to(device)
        ids.eval()
        for p in ids.parameters():
            p.requires_grad_(False)

        self.mask = torch.from_numpy(np.asarray(mask, dtype=np.float32)).to(device)
        benign = torch.from_numpy(X_train[y_train == 0]).float()
        attack = torch.from_numpy(X_train[y_train == 1]).float()
        if len(benign) == 0 or len(attack) == 0:
            raise ValueError("Need both benign and attack rows to train the adversarial GAN.")

        loader = DataLoader(
            TensorDataset(attack),
            batch_size=min(batch_size, len(attack)),
            shuffle=True,
            drop_last=False,
        )
        bce = nn.BCEWithLogitsLoss()
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        history = {"d_loss": [], "g_loss": [], "ids_score": []}

        for epoch in range(1, epochs + 1):
            d_run = g_run = ids_run = 0.0
            n_batches = 0
            for (xb,) in loader:
                xb = xb.to(device)
                bs = xb.size(0)
                z = torch.randn(bs, self.latent_dim, device=device)
                idx = torch.randint(0, len(benign), (bs,))
                real_benign = benign[idx].to(device)
                with torch.no_grad():
                    fake = apply_perturbation(xb, self.generator(xb, z), self.mask)
                opt_d.zero_grad()
                d_real = self.discriminator(real_benign)
                d_fake = self.discriminator(fake.detach())
                d_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
                d_loss.backward()
                opt_d.step()

                z = torch.randn(bs, self.latent_dim, device=device)
                opt_g.zero_grad()
                delta = self.generator(xb, z)
                fake = apply_perturbation(xb, delta, self.mask)
                g_adv = bce(self.discriminator(fake), torch.ones_like(d_real))
                ids_logits = ids(fake)
                ids_loss = torch.sigmoid(ids_logits).mean()
                pert_loss = (delta * self.mask).pow(2).mean()
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
            if epoch % log_interval == 0 or epoch in {1, epochs}:
                logger.info(
                    "GAN epoch %d/%d  D=%.4f  G=%.4f  IDS_P(attack)=%.4f",
                    epoch, epochs, history["d_loss"][-1], history["g_loss"][-1], history["ids_score"][-1],
                )

        if artifacts_dir is not None:
            self.save(Path(artifacts_dir) / "gan.pt")
            write_json(
                Path(artifacts_dir) / "gan_train_metrics.json",
                {
                    "final_d_loss": history["d_loss"][-1],
                    "final_g_loss": history["g_loss"][-1],
                    "final_ids_p_attack": history["ids_score"][-1],
                    "epochs": epochs,
                    "eps": self.eps,
                    "lambda_ids": lambda_ids,
                },
            )
        return {"history": history, "generator": self.generator}

    def generate(self, X_attack: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        self.generator.eval()
        device = self.device
        mask_t = self.mask
        if mask is not None:
            mask_t = torch.from_numpy(np.asarray(mask, dtype=np.float32)).to(device)
        if mask_t is None:
            mask_t = torch.ones(X_attack.shape[1], device=device)
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(X_attack, dtype=np.float32)).to(device)
            z = torch.randn(len(X_attack), self.latent_dim, device=device)
            fake = apply_perturbation(x, self.generator(x, z), mask_t)
        return fake.cpu().numpy().astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "feature_dim": self.feature_dim,
                "latent_dim": self.latent_dim,
                "eps": self.eps,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: torch.device | None = None) -> "GANAttack":
        device = device or get_device()
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        obj = cls(
            feature_dim=int(payload["feature_dim"]),
            latent_dim=int(payload["latent_dim"]),
            eps=float(payload["eps"]),
            device=device,
        )
        obj.generator.load_state_dict(payload["generator"])
        obj.discriminator.load_state_dict(payload["discriminator"])
        obj.generator.eval()
        return obj


def craft_gan_attacks(generator, X_attack, mask, latent_dim, device):
    """Backward-compatible helper used by the original evaluator."""
    generator.eval()
    with torch.no_grad():
        x = torch.from_numpy(X_attack).float().to(device)
        z = torch.randn(len(X_attack), latent_dim, device=device)
        fake = apply_perturbation(x, generator(x, z), torch.from_numpy(mask).float().to(device))
    return fake.cpu().numpy()
