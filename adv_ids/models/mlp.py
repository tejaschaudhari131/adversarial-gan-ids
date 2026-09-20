"""MLP and a deeper MLP IDS."""

from __future__ import annotations

from torch import nn

from adv_ids.models.torch_ids import TorchIDS


class IDSNet(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (128, 64), dropout: float = 0.35):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.hidden = hidden
        self.dropout = dropout

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def attack_prob(self, x):
        import torch

        return torch.sigmoid(self.forward(x))


class MLPIDS(TorchIDS):
    name = "mlp"

    def __init__(
        self,
        input_dim: int,
        hidden: tuple[int, ...] = (128, 64),
        dropout: float = 0.35,
        device=None,
    ):
        self.hidden = tuple(hidden)
        self.dropout = float(dropout)
        super().__init__(input_dim=input_dim, device=device)

    def build_module(self, input_dim: int) -> nn.Module:
        return IDSNet(input_dim=input_dim, hidden=self.hidden, dropout=self.dropout)

    def extra_state(self) -> dict:
        return {"hidden": list(self.hidden), "dropout": self.dropout}

    def load_extra_state(self, payload: dict) -> None:
        self.hidden = tuple(payload.get("hidden", self.hidden))
        self.dropout = float(payload.get("dropout", self.dropout))
        self.module = self.build_module(self.input_dim).to(self.device)


class DeepMLPIDS(MLPIDS):
    name = "deep_mlp"

    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (256, 128, 64, 32), dropout: float = 0.30, device=None):
        super().__init__(input_dim=input_dim, hidden=hidden, dropout=dropout, device=device)


def build_ids_model(input_dim: int, **kwargs) -> IDSNet:
    """Backward-compatible constructor used by the original trainer."""
    return IDSNet(input_dim=input_dim, **kwargs)
