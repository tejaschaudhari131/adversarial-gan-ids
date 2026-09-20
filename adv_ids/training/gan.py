"""Train the conditional adversarial GAN against a frozen IDS."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from adv_ids.attacks.gan_attack import GANAttack
from adv_ids.models.gan import build_generator
from adv_ids.models.mlp import MLPIDS
from adv_ids.training.ids import load_ids_wrapper
from adv_ids.utils.device import get_device


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
    model=None,
) -> dict:
    if model is None:
        model = load_ids_wrapper(artifacts_dir, model_name="mlp")
    elif not hasattr(model, "attack_logits_torch"):
        # Raw nn.Module from the legacy loader.
        wrapper = MLPIDS(input_dim=X_train.shape[1])
        wrapper.module = model
        wrapper.device = next(model.parameters()).device
        model = wrapper
    attack = GANAttack(feature_dim=X_train.shape[1], latent_dim=latent_dim, eps=eps)
    result = attack.fit(
        model,
        X_train,
        y_train,
        modifiable_mask,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lambda_ids=lambda_ids,
        lambda_pert=lambda_pert,
        artifacts_dir=artifacts_dir,
        log_interval=log_interval,
    )
    return {**result, "checkpoint": str(Path(artifacts_dir) / "gan.pt"), "attack": attack}


def load_generator(artifacts_dir: str | Path = "artifacts", device: torch.device | None = None):
    device = device or get_device()
    payload = torch.load(Path(artifacts_dir) / "gan.pt", map_location=device, weights_only=False)
    G = build_generator(payload["feature_dim"], latent_dim=payload["latent_dim"], eps=payload["eps"]).to(device)
    G.load_state_dict(payload["generator"])
    G.eval()
    return G, payload
