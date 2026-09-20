"""YAML/JSON experiment config loading."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dict(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text) or {}
    elif suffix == ".json":
        import json

        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config type: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data
