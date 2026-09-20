"""Deterministic seeding for numpy, Python, and PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and torch (if installed) for reproducible runs."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
