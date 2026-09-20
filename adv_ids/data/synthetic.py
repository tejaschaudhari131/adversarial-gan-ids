"""Same-schema synthetic stand-ins so CI and first-run work without multi-GB downloads."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from adv_ids.data.schemas import (
    CIC_ATTACK_NAMES,
    CICIOT2023_FEATURES,
    CICIOT_ATTACK_NAMES,
    CICIDS2017_FEATURES,
    CICIDS2018_FEATURES,
    UNSW_ATTACK_NAMES,
    UNSW_NB15_CATEGORICAL,
    UNSW_NB15_NUMERIC,
    get_spec,
    normalize_dataset_name,
)

logger = logging.getLogger(__name__)


def _write(df: pd.DataFrame, output_path) -> pd.DataFrame:
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Wrote synthetic dataset to %s (%d rows)", path, len(df))
    return df


def generate_synthetic_cicids2017(
    n_benign: int = 8000,
    n_attack: int = 4000,
    seed: int = 42,
    output_path=None,
) -> pd.DataFrame:
    """CIC-IDS2017 MachineLearningCSV schema with a separable attack shift."""
    spec = get_spec("cicids2017")
    rng = np.random.default_rng(seed)
    n_features = len(CICIDS2017_FEATURES)
    frozen_idx = [i for i, name in enumerate(CICIDS2017_FEATURES) if name in spec.frozen]
    benign = rng.normal(loc=0.40, scale=0.16, size=(n_benign, n_features))
    attack = rng.normal(loc=0.40, scale=0.16, size=(n_attack, n_features))
    controllable = [i for i in range(n_features) if i not in frozen_idx]
    attack[:, controllable] += rng.normal(0.18, 0.07, size=(n_attack, len(controllable)))
    port_idx = CICIDS2017_FEATURES.index("Destination Port")
    common_ports = np.array([80, 443, 53, 22, 25, 123, 3389], dtype=float)
    benign[:, port_idx] = rng.choice(common_ports, size=n_benign)
    attack[:, port_idx] = rng.integers(1024, 65535, size=n_attack).astype(float)
    for name in spec.frozen:
        if name == "Destination Port":
            continue
        if name not in CICIDS2017_FEATURES:
            continue
        idx = CICIDS2017_FEATURES.index(name)
        benign[:, idx] = rng.integers(0, 2, size=n_benign).astype(float)
        attack[:, idx] = rng.integers(0, 2, size=n_attack).astype(float)
    X = np.clip(np.vstack([benign, attack]), 0.0, None)
    raw_labels = np.concatenate(
        [np.array(["BENIGN"] * n_benign), rng.choice(np.array(CIC_ATTACK_NAMES), size=n_attack)]
    )
    df = pd.DataFrame(X, columns=CICIDS2017_FEATURES)
    df["Label"] = raw_labels
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return _write(df, output_path)


def generate_synthetic_cicids2018(
    n_benign: int = 8000,
    n_attack: int = 4000,
    seed: int = 42,
    output_path=None,
) -> pd.DataFrame:
    """CSE-CIC-IDS2018 abbreviated CICFlowMeter schema."""
    spec = get_spec("cicids2018")
    rng = np.random.default_rng(seed)
    n_features = len(CICIDS2018_FEATURES)
    frozen_idx = [i for i, name in enumerate(CICIDS2018_FEATURES) if name in spec.frozen]
    benign = rng.normal(loc=0.38, scale=0.15, size=(n_benign, n_features))
    attack = rng.normal(loc=0.38, scale=0.15, size=(n_attack, n_features))
    controllable = [i for i in range(n_features) if i not in frozen_idx]
    attack[:, controllable] += rng.normal(0.16, 0.06, size=(n_attack, len(controllable)))
    port_idx = CICIDS2018_FEATURES.index("Dst Port")
    proto_idx = CICIDS2018_FEATURES.index("Protocol")
    benign[:, port_idx] = rng.choice([80, 443, 53, 22], size=n_benign).astype(float)
    attack[:, port_idx] = rng.integers(1024, 65535, size=n_attack).astype(float)
    benign[:, proto_idx] = rng.choice([6, 17], size=n_benign).astype(float)
    attack[:, proto_idx] = rng.choice([6, 17], size=n_attack).astype(float)
    for name in spec.frozen:
        if name in {"Dst Port", "Protocol"} or name not in CICIDS2018_FEATURES:
            continue
        idx = CICIDS2018_FEATURES.index(name)
        benign[:, idx] = rng.integers(0, 2, size=n_benign).astype(float)
        attack[:, idx] = rng.integers(0, 2, size=n_attack).astype(float)
    X = np.clip(np.vstack([benign, attack]), 0.0, None)
    labels = np.concatenate(
        [np.array(["Benign"] * n_benign), rng.choice(np.array(["DDOS", "DoS", "Bot", "Infilteration", "Brute Force"]), size=n_attack)]
    )
    df = pd.DataFrame(X, columns=CICIDS2018_FEATURES)
    df["Label"] = labels
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return _write(df, output_path)


def generate_synthetic_unsw_nb15(
    n_benign: int = 8000,
    n_attack: int = 4000,
    seed: int = 42,
    output_path=None,
) -> pd.DataFrame:
    """UNSW-NB15 official CSV export schema (numeric + light categoricals)."""
    spec = get_spec("unsw_nb15")
    rng = np.random.default_rng(seed)
    n_features = len(UNSW_NB15_NUMERIC)
    frozen_idx = [i for i, name in enumerate(UNSW_NB15_NUMERIC) if name in spec.frozen]
    benign = np.abs(rng.normal(loc=0.35, scale=0.14, size=(n_benign, n_features)))
    attack = np.abs(rng.normal(loc=0.35, scale=0.14, size=(n_attack, n_features)))
    controllable = [i for i in range(n_features) if i not in frozen_idx]
    attack[:, controllable] += np.abs(rng.normal(0.20, 0.08, size=(n_attack, len(controllable))))
    for name in ("is_ftp_login", "is_sm_ips_ports"):
        idx = UNSW_NB15_NUMERIC.index(name)
        benign[:, idx] = rng.integers(0, 2, size=n_benign).astype(float)
        attack[:, idx] = rng.integers(0, 2, size=n_attack).astype(float)
    df_num = pd.DataFrame(np.clip(np.vstack([benign, attack]), 0.0, None), columns=UNSW_NB15_NUMERIC)
    n = n_benign + n_attack
    proto = rng.choice(["tcp", "udp", "icmp"], size=n)
    service = rng.choice(["-", "http", "dns", "smtp", "ftp", "ssh"], size=n)
    state = rng.choice(["FIN", "INT", "CON", "REQ"], size=n)
    attack_cat = np.concatenate(
        [np.array(["Normal"] * n_benign), rng.choice(np.array(UNSW_ATTACK_NAMES), size=n_attack)]
    )
    label = np.concatenate([np.zeros(n_benign, dtype=int), np.ones(n_attack, dtype=int)])
    df = df_num.copy()
    df["proto"] = proto
    df["service"] = service
    df["state"] = state
    df["attack_cat"] = attack_cat
    df["label"] = label
    order = list(UNSW_NB15_CATEGORICAL) + list(UNSW_NB15_NUMERIC) + ["attack_cat", "label"]
    df = df[order].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return _write(df, output_path)


def generate_synthetic_ciciot2023(
    n_benign: int = 8000,
    n_attack: int = 4000,
    seed: int = 42,
    output_path=None,
) -> pd.DataFrame:
    """Compact CIC-IoT-2023-style numeric stand-in."""
    spec = get_spec("ciciot2023")
    rng = np.random.default_rng(seed)
    n_features = len(CICIOT2023_FEATURES)
    frozen_idx = [i for i, name in enumerate(CICIOT2023_FEATURES) if name in spec.frozen]
    benign = np.abs(rng.normal(loc=0.32, scale=0.12, size=(n_benign, n_features)))
    attack = np.abs(rng.normal(loc=0.32, scale=0.12, size=(n_attack, n_features)))
    controllable = [i for i in range(n_features) if i not in frozen_idx]
    attack[:, controllable] += np.abs(rng.normal(0.22, 0.07, size=(n_attack, len(controllable))))
    for name in spec.frozen:
        if name not in CICIOT2023_FEATURES:
            continue
        idx = CICIOT2023_FEATURES.index(name)
        benign[:, idx] = rng.integers(0, 2, size=n_benign).astype(float)
        attack[:, idx] = rng.integers(0, 2, size=n_attack).astype(float)
    X = np.clip(np.vstack([benign, attack]), 0.0, None)
    labels = np.concatenate(
        [np.array(["Benign"] * n_benign), rng.choice(np.array(CICIOT_ATTACK_NAMES), size=n_attack)]
    )
    df = pd.DataFrame(X, columns=CICIOT2023_FEATURES)
    df["label"] = labels
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return _write(df, output_path)


GENERATORS = {
    "cicids2017": generate_synthetic_cicids2017,
    "cicids2018": generate_synthetic_cicids2018,
    "unsw_nb15": generate_synthetic_unsw_nb15,
    "ciciot2023": generate_synthetic_ciciot2023,
}


def generate_synthetic(
    dataset: str = "cicids2017",
    n_benign: int = 8000,
    n_attack: int = 4000,
    seed: int = 42,
    output_path=None,
) -> pd.DataFrame:
    key = normalize_dataset_name(dataset)
    if key not in GENERATORS:
        raise ValueError(f"No synthetic generator for dataset '{dataset}'")
    return GENERATORS[key](n_benign=n_benign, n_attack=n_attack, seed=seed, output_path=output_path)
