"""CIC-IDS2017 loading, synthetic stand-in data, and train/test preparation."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

CICIDS2017_FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean",
    "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

FROZEN_FEATURES = {
    "Destination Port", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
    "ECE Flag Count", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
}


def generate_synthetic_cicids2017(n_benign=8000, n_attack=4000, seed=42, output_path=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_features = len(CICIDS2017_FEATURES)
    frozen_idx = [i for i, name in enumerate(CICIDS2017_FEATURES) if name in FROZEN_FEATURES]
    benign = rng.normal(loc=0.40, scale=0.16, size=(n_benign, n_features))
    attack = rng.normal(loc=0.40, scale=0.16, size=(n_attack, n_features))
    controllable = [i for i in range(n_features) if i not in frozen_idx]
    attack[:, controllable] += rng.normal(0.18, 0.07, size=(n_attack, len(controllable)))
    port_idx = CICIDS2017_FEATURES.index("Destination Port")
    common_ports = np.array([80, 443, 53, 22, 25, 123, 3389], dtype=float)
    benign[:, port_idx] = rng.choice(common_ports, size=n_benign)
    attack[:, port_idx] = rng.integers(1024, 65535, size=n_attack).astype(float)
    for name in FROZEN_FEATURES:
        if name == "Destination Port":
            continue
        idx = CICIDS2017_FEATURES.index(name)
        benign[:, idx] = rng.integers(0, 2, size=n_benign).astype(float)
        attack[:, idx] = rng.integers(0, 2, size=n_attack).astype(float)
    X = np.clip(np.vstack([benign, attack]), 0.0, None)
    attack_names = np.array(["DoS Hulk", "DDoS", "PortScan", "FTP-Patator", "Bot", "Infiltration"])
    raw_labels = np.concatenate([np.array(["BENIGN"] * n_benign), rng.choice(attack_names, size=n_attack)])
    df = pd.DataFrame(X, columns=CICIDS2017_FEATURES)
    df["Label"] = raw_labels
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Wrote synthetic CIC-IDS2017-like dataset to %s (%d rows)", path, len(df))
    return df


def _normalize_label_column(columns: pd.Index) -> pd.Index:
    stripped = columns.str.strip()
    mapping = {c: ("Label" if c.lower() == "label" else c) for c in stripped}
    return pd.Index([mapping[c] for c in stripped])


def load_traffic_csv(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    if df.empty:
        raise ValueError(f"Dataset is empty: {file_path}")
    df.columns = _normalize_label_column(df.columns)
    if "Label" not in df.columns:
        raise ValueError(f"'Label' column not found. Columns: {list(df.columns)}")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows with NaN/Inf (%.2f%%)", dropped, 100 * dropped / before)
    return df.reset_index(drop=True)


def binarize_labels(labels: pd.Series) -> np.ndarray:
    cleaned = labels.astype(str).str.strip()
    return np.where(cleaned.str.upper() == "BENIGN", 0, 1).astype(np.int64)


def feature_mask(feature_names):
    return np.array([0.0 if name in FROZEN_FEATURES else 1.0 for name in feature_names], dtype=np.float32)


def prepare_dataset(file_path, test_size=0.2, random_state=42, artifacts_dir="artifacts") -> dict[str, Any]:
    df = load_traffic_csv(file_path)
    y = binarize_labels(df["Label"])
    X_df = df.drop(columns=["Label"])
    non_numeric = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        logger.info("Dropping non-numeric columns: %s", non_numeric)
        X_df = X_df.select_dtypes(include=[np.number])
    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=np.float32)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, artifacts / "scaler.joblib")
    meta = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_benign_train": int((y_train == 0).sum()),
        "n_attack_train": int((y_train == 1).sum()),
        "modifiable_mask": feature_mask(feature_names).tolist(),
    }
    with open(artifacts / "dataset_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Prepared dataset: train=%d test=%d features=%d attacks_train=%d", len(X_train), len(X_test), X_train.shape[1], meta["n_attack_train"])
    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "scaler": scaler, "feature_names": feature_names,
        "modifiable_mask": feature_mask(feature_names), "meta": meta,
    }
