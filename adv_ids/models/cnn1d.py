"""1D CNN over the flow-feature vector (channels-first)."""

from __future__ import annotations

from torch import nn

from adv_ids.models.torch_ids import TorchIDS


class CNN1DNet(nn.Module):
    def __init__(self, input_dim: int, channels: tuple[int, ...] = (32, 64), dropout: float = 0.30):
        super().__init__()
        convs: list[nn.Module] = []
        in_ch = 1
        for ch in channels:
            convs.extend(
                [
                    nn.Conv1d(in_ch, ch, kernel_size=5, padding=2),
                    nn.BatchNorm1d(ch),
                    nn.ReLU(),
                ]
            )
            in_ch = ch
        self.conv = nn.Sequential(*convs)
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * 16, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        h = self.conv(x.unsqueeze(1))
        h = self.pool(h)
        return self.fc(h).squeeze(-1)


class CNN1DIDS(TorchIDS):
    name = "cnn1d"

    def __init__(self, input_dim: int, channels: tuple[int, ...] = (32, 64), dropout: float = 0.30, device=None):
        self.channels = tuple(channels)
        self.dropout = float(dropout)
        super().__init__(input_dim=input_dim, device=device)

    def build_module(self, input_dim: int) -> nn.Module:
        return CNN1DNet(input_dim=input_dim, channels=self.channels, dropout=self.dropout)

    def extra_state(self) -> dict:
        return {"channels": list(self.channels), "dropout": self.dropout}

    def load_extra_state(self, payload: dict) -> None:
        self.channels = tuple(payload.get("channels", self.channels))
        self.dropout = float(payload.get("dropout", self.dropout))
        self.module = self.build_module(self.input_dim).to(self.device)
