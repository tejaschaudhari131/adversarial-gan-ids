"""Load real or synthetic flow-feature tables and infer the dataset family."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from adv_ids.data.catalog import DatasetLayoutError, require_dataset_files
from adv_ids.data.schemas import normalize_dataset_name
from adv_ids.data.synthetic import generate_synthetic

logger = logging.getLogger(__name__)

DEFAULT_RAW_LAYOUT = {
    "cicids2017": "data/raw/cicids2017",
    "cicids2018": "data/raw/cse-cic-ids2018",
    "unsw_nb15": "data/raw/unsw-nb15",
    "ciciot2023": "data/raw/cic-iot-2023",
}

# Recommended CSE-CIC-IDS2018 day files for a tractable subset.
IDS2018_RECOMMENDED_DAYS = [
    "Wednesday-14-02-2018",   # FTP-BruteForce, SSH-Bruteforce
    "Thursday-15-02-2018",    # DoS-GoldenEye, DoS-Slowloris
    "Friday-16-02-2018",      # DoS-SlowHTTPTest, DoS-Hulk
]


def normalize_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """Strip header whitespace and canonicalize the primary label column."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    lower = {c.lower(): c for c in df.columns}
    if "label" in lower:
        src = lower["label"]
        # UNSW official export uses lowercase `label` plus `attack_cat`.
        if "attack_cat" in lower:
            if src != "label":
                df = df.rename(columns={src: "label"})
        elif src != "Label":
            df = df.rename(columns={src: "Label"})
    return df


def infer_dataset_name(df: pd.DataFrame) -> str:
    cols = {c.strip() for c in df.columns.astype(str)}
    lower = {c.lower() for c in cols}
    if "attack_cat" in lower or {"sttl", "dttl", "sload"}.issubset(lower):
        return "unsw_nb15"
    if "header_length" in lower and "protocol type" in lower:
        return "ciciot2023"
    # Engelen / fixed CICFlowMeter uses Dst Port plus singular "Total Fwd Packet".
    if "total fwd packet" in lower or "fwd init win bytes" in lower:
        return "cicids2017"
    if "dst port" in lower or "tot fwd pkts" in lower:
        return "cicids2018"
    if "destination port" in lower or "total fwd packets" in lower:
        return "cicids2017"
    return "cicids2017"


def _iter_csv_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {path}")
    files = sorted(path.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files under {path}")
    return files


def load_csv_table(path: str | Path, max_files: int | None = None) -> pd.DataFrame:
    files = _iter_csv_files(Path(path))
    if max_files is not None:
        files = files[: max(1, int(max_files))]
    frames = []
    for fp in files:
        logger.info("Loading %s", fp)
        part = pd.read_csv(fp, low_memory=False, encoding="utf-8-sig")
        frames.append(normalize_label_column(part))
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")
    return df


def resolve_dataset_path(
    dataset_name: str,
    dataset_path: str | Path | None,
    *,
    synthetic: bool,
    n_benign: int,
    n_attack: int,
    seed: int,
    output_dir: str | Path = "data",
) -> Path:
    """Return a CSV path, generating a synthetic stand-in when requested."""
    name = normalize_dataset_name(dataset_name)
    if synthetic:
        out = Path(output_dir) / f"synthetic_{name}.csv"
        generate_synthetic(name, n_benign=n_benign, n_attack=n_attack, seed=seed, output_path=out)
        return out
    try:
        return require_dataset_files(name, dataset_path)
    except DatasetLayoutError:
        raise
