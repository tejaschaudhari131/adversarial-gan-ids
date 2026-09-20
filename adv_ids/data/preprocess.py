"""Unified cleaning, scaling, and train/val/test splits for all supported datasets."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from adv_ids.data.loaders import infer_dataset_name, load_csv_table, normalize_label_column
from adv_ids.data.masks import feature_mask
from adv_ids.data.schemas import BENIGN_TOKENS, DatasetSpec, get_spec, normalize_dataset_name
from adv_ids.utils.io import ensure_dir, write_json

logger = logging.getLogger(__name__)


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    modifiable_mask: np.ndarray
    scaler: MinMaxScaler
    dataset_name: str
    label_mode: str = "binary"
    class_names: list[str] = field(default_factory=lambda: ["BENIGN", "ATTACK"])
    y_multi_train: np.ndarray | None = None
    y_multi_val: np.ndarray | None = None
    y_multi_test: np.ndarray | None = None
    multi_class_names: list[str] | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    scaler_path: str | None = None

    def as_legacy_dict(self) -> dict[str, Any]:
        """Shape expected by the original train/eval helpers (train + test only)."""
        return {
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
            "X_val": self.X_val,
            "y_val": self.y_val,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "modifiable_mask": self.modifiable_mask,
            "meta": self.meta,
            "bundle": self,
        }


def load_traffic_csv(file_path) -> pd.DataFrame:
    """Backward-compatible single-file loader used by the original tests."""
    return load_traffic_table(file_path)


def load_traffic_table(file_path, max_files: int | None = None) -> pd.DataFrame:
    df = load_csv_table(file_path, max_files=max_files)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows with NaN/Inf (%.2f%%)", dropped, 100 * dropped / before)
    return df.reset_index(drop=True)


def binarize_labels(labels: pd.Series) -> np.ndarray:
    cleaned = labels.astype(str).str.strip()
    benign = cleaned.str.upper().isin({str(t).upper() for t in BENIGN_TOKENS if t != 0})
    # UNSW uses 0/1 already; treat numeric 0 as benign.
    numeric = pd.to_numeric(cleaned, errors="coerce")
    numeric_benign = numeric == 0
    is_benign = benign | numeric_benign.fillna(False)
    return np.where(is_benign.to_numpy(), 0, 1).astype(np.int64)


def encode_multiclass(labels: pd.Series) -> tuple[np.ndarray, list[str]]:
    cleaned = labels.astype(str).str.strip()
    # Keep BENIGN / Normal as class 0 when present.
    classes = sorted(cleaned.unique(), key=lambda s: (s.upper() not in {"BENIGN", "NORMAL", "0"}, s))
    index = {c: i for i, c in enumerate(classes)}
    encoded = cleaned.map(index).to_numpy(dtype=np.int64)
    return encoded, classes


def _drop_duplicate_and_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """CIC-IDS2017 MachineLearningCSV duplicates Fwd Header Length as Fwd Header Length.1."""
    drop = [c for c in df.columns if c.endswith(".1") and c[: -2] in set(df.columns)]
    leakage = [
        c for c in df.columns
        if c.lower() in {
            "timestamp", "flow id", "src ip", "dst ip", "source ip", "destination ip",
            "src port", "source port",
        }
    ]
    to_drop = [c for c in drop + leakage if c in df.columns]
    if to_drop:
        logger.info("Dropping duplicate/leakage columns: %s", to_drop)
        df = df.drop(columns=to_drop)
    return df


def _encode_categoricals(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    df = df.copy()
    for col in spec.categorical:
        if col not in df.columns:
            continue
        codes, _ = pd.factorize(df[col].astype(str), sort=True)
        df[col] = codes.astype(np.float32)
    return df


def _encode_categoricals_pair(
    train: pd.DataFrame, test: pd.DataFrame, spec: DatasetSpec
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Factorize categoricals on train; map unseen test values to -1."""
    train, test = train.copy(), test.copy()
    for col in spec.categorical:
        if col not in train.columns:
            continue
        codes, uniques = pd.factorize(train[col].astype(str), sort=True)
        train[col] = codes.astype(np.float32)
        mapping = {u: i for i, u in enumerate(uniques)}
        if col in test.columns:
            test[col] = test[col].astype(str).map(mapping).fillna(-1).to_numpy(dtype=np.float32)
    return train, test


def _apply_aliases(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    rename = {
        src: dst
        for src, dst in spec.aliases.items()
        if src in df.columns and dst not in df.columns and dst in spec.features
    }
    return df.rename(columns=rename) if rename else df


def official_unsw_split_paths(path: str | Path) -> tuple[Path, Path] | None:
    folder = Path(path)
    if not folder.is_dir():
        return None
    train = folder / "UNSW_NB15_training-set.csv"
    test = folder / "UNSW_NB15_testing-set.csv"
    if train.is_file() and test.is_file():
        return train, test
    return None


def _select_feature_frame(
    df: pd.DataFrame,
    spec: DatasetSpec,
    *,
    encode_categoricals: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    df = normalize_label_column(df)
    df = _apply_aliases(df, spec)
    df = _drop_duplicate_and_leakage_columns(df)
    for col in spec.drop_columns:
        if col in df.columns:
            df = df.drop(columns=[col])

    label_col = spec.label_column if spec.label_column in df.columns else (
        "Label" if "Label" in df.columns else "label"
    )
    if label_col not in df.columns:
        raise ValueError(f"Label column not found. Columns: {list(df.columns)}")

    multi = None
    if spec.multiclass_column and spec.multiclass_column in df.columns:
        multi = df[spec.multiclass_column]
    elif label_col in df.columns:
        multi = df[label_col]

    y_raw = df[label_col]
    drop_y = {label_col}
    if spec.multiclass_column and spec.multiclass_column in df.columns:
        drop_y.add(spec.multiclass_column)
    X_df = df.drop(columns=[c for c in drop_y if c in df.columns])
    if encode_categoricals:
        X_df = _encode_categoricals(X_df, spec)

    # Prefer the documented feature order when columns exist; otherwise keep numeric leftovers.
    available = [c for c in spec.features if c in X_df.columns]
    pending_cat = [] if encode_categoricals else [c for c in spec.categorical if c in X_df.columns and c not in available]
    extra_numeric = [
        c for c in X_df.columns
        if c not in available and c not in pending_cat and pd.api.types.is_numeric_dtype(X_df[c])
    ]
    ordered = available + pending_cat + extra_numeric
    if not ordered:
        raise ValueError("No numeric feature columns remain after cleaning.")
    X_df = X_df[ordered]
    if encode_categoricals:
        non_numeric = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            logger.info("Dropping leftover non-numeric columns: %s", non_numeric)
            X_df = X_df.select_dtypes(include=[np.number])
    else:
        drop_non = [
            c for c in X_df.columns
            if c not in spec.categorical and not pd.api.types.is_numeric_dtype(X_df[c])
        ]
        if drop_non:
            X_df = X_df.drop(columns=drop_non)
    return X_df, y_raw, multi


def _split_arrays(
    X: np.ndarray,
    y: np.ndarray,
    y_multi: np.ndarray | None,
    *,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple:
    strat = y if len(np.unique(y)) > 1 else None
    extra = {"ym": y_multi} if y_multi is not None else {}
    split_payload = {"X": X, "y": y, **extra}
    first = train_test_split(
        *([split_payload["X"], split_payload["y"]] + ([split_payload["ym"]] if extra else [])),
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
    if extra:
        X_rest, X_test, y_rest, y_test, ym_rest, ym_test = first
    else:
        X_rest, X_test, y_rest, y_test = first
        ym_rest = ym_test = None

    if val_size and val_size > 0:
        val_frac = val_size / max(1.0 - test_size, 1e-9)
        strat_rest = y_rest if len(np.unique(y_rest)) > 1 else None
        second = train_test_split(
            *([X_rest, y_rest] + ([ym_rest] if extra else [])),
            test_size=val_frac,
            random_state=random_state,
            stratify=strat_rest,
        )
        if extra:
            X_train, X_val, y_train, y_val, ym_train, ym_val = second
        else:
            X_train, X_val, y_train, y_val = second
            ym_train = ym_val = None
    else:
        X_train, y_train, ym_train = X_rest, y_rest, ym_rest
        X_val = np.empty((0, X.shape[1]), dtype=X.dtype)
        y_val = np.empty((0,), dtype=y.dtype)
        ym_val = None if ym_rest is None else np.empty((0,), dtype=ym_rest.dtype)
    return X_train, X_val, X_test, y_train, y_val, y_test, ym_train, ym_val, ym_test


def prepare_from_frame(
    df: pd.DataFrame,
    *,
    dataset_name: str | None = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    label_mode: str = "binary",
    artifacts_dir: str | Path = "artifacts",
    max_samples: int | None = None,
) -> DatasetBundle:
    df = df.copy()
    inferred = infer_dataset_name(df)
    name = normalize_dataset_name(dataset_name or inferred)
    spec = get_spec(name)
    X_df, y_raw, y_multi_raw = _select_feature_frame(df, spec)
    y_bin = binarize_labels(y_raw)
    y_multi, multi_names = encode_multiclass(y_multi_raw if y_multi_raw is not None else y_raw)

    if max_samples is not None and len(X_df) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_df), size=max_samples, replace=False)
        X_df = X_df.iloc[idx].reset_index(drop=True)
        y_bin = y_bin[idx]
        y_multi = y_multi[idx]

    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=np.float32)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    target = y_bin if label_mode == "binary" else y_multi
    X_train, X_val, X_test, y_train, y_val, y_test, ym_tr, ym_va, ym_te = _split_arrays(
        X_scaled, target, y_multi if label_mode == "binary" else None,
        test_size=test_size, val_size=val_size, random_state=random_state,
    )
    mask = feature_mask(feature_names, spec)
    artifacts = ensure_dir(artifacts_dir)
    scaler_path = artifacts / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    meta = {
        "dataset": name,
        "family": spec.family,
        "label_mode": label_mode,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "n_benign_train": int((y_train == 0).sum()) if label_mode == "binary" else None,
        "n_attack_train": int((y_train == 1).sum()) if label_mode == "binary" else None,
        "modifiable_mask": mask.tolist(),
        "notes": spec.notes,
        "seed": int(random_state),
        "test_size": float(test_size),
        "val_size": float(val_size),
    }
    write_json(artifacts / "dataset_meta.json", meta)
    logger.info(
        "Prepared %s: train=%d val=%d test=%d features=%d",
        name, len(X_train), len(X_val), len(X_test), X_train.shape[1],
    )
    class_names = ["BENIGN", "ATTACK"] if label_mode == "binary" else multi_names
    return DatasetBundle(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        feature_names=feature_names, modifiable_mask=mask, scaler=scaler,
        dataset_name=name, label_mode=label_mode, class_names=class_names,
        y_multi_train=ym_tr, y_multi_val=ym_va, y_multi_test=ym_te,
        multi_class_names=multi_names, meta=meta, scaler_path=str(scaler_path),
    )


def prepare_from_official_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    dataset_name: str = "unsw_nb15",
    val_size: float = 0.1,
    random_state: int = 42,
    label_mode: str = "binary",
    artifacts_dir: str | Path = "artifacts",
    max_samples: int | None = None,
) -> DatasetBundle:
    """Prepare an official train/test export. Scaler and categorical codes fit on train only."""
    spec = get_spec(dataset_name)
    X_tr, y_tr_raw, ym_tr_raw = _select_feature_frame(train_df, spec, encode_categoricals=False)
    X_te, y_te_raw, ym_te_raw = _select_feature_frame(test_df, spec, encode_categoricals=False)
    X_tr, X_te = _encode_categoricals_pair(X_tr, X_te, spec)
    cols = [c for c in X_tr.columns if c in X_te.columns]
    extra = [c for c in X_te.columns if c not in cols and pd.api.types.is_numeric_dtype(X_te[c])]
    if extra:
        logger.info("Dropping test-only numeric columns: %s", extra)
    X_tr = X_tr[cols]
    X_te = X_te[cols]
    y_tr = binarize_labels(y_tr_raw)
    y_te = binarize_labels(y_te_raw)
    ym_tr, multi_names = encode_multiclass(ym_tr_raw if ym_tr_raw is not None else y_tr_raw)
    ym_te, _ = encode_multiclass(ym_te_raw if ym_te_raw is not None else y_te_raw)

    if max_samples is not None:
        rng = np.random.default_rng(random_state)
        if len(X_tr) > max_samples:
            idx = rng.choice(len(X_tr), size=max_samples, replace=False)
            X_tr, y_tr, ym_tr = X_tr.iloc[idx], y_tr[idx], ym_tr[idx]
        test_cap = max(int(0.3 * max_samples), 200)
        if len(X_te) > test_cap:
            idx = rng.choice(len(X_te), size=test_cap, replace=False)
            X_te, y_te, ym_te = X_te.iloc[idx], y_te[idx], ym_te[idx]

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_train_all = scaler.fit_transform(X_tr.to_numpy(dtype=np.float32)).astype(np.float32)
    X_test = scaler.transform(X_te.to_numpy(dtype=np.float32)).astype(np.float32)
    target_tr = y_tr if label_mode == "binary" else ym_tr
    target_te = y_te if label_mode == "binary" else ym_te

    if val_size and val_size > 0:
        strat = target_tr if len(np.unique(target_tr)) > 1 else None
        X_train, X_val, y_train, y_val, ym_train, ym_val = train_test_split(
            X_train_all, target_tr, ym_tr, test_size=val_size, random_state=random_state, stratify=strat,
        )
    else:
        X_train, y_train, ym_train = X_train_all, target_tr, ym_tr
        X_val = np.empty((0, X_train_all.shape[1]), dtype=np.float32)
        y_val = np.empty((0,), dtype=target_tr.dtype)
        ym_val = np.empty((0,), dtype=ym_tr.dtype)

    feature_names = list(cols)
    mask = feature_mask(feature_names, spec)
    artifacts = ensure_dir(artifacts_dir)
    scaler_path = artifacts / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    meta = {
        "dataset": spec.name,
        "family": spec.family,
        "label_mode": label_mode,
        "official_split": True,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "n_benign_train": int((y_train == 0).sum()) if label_mode == "binary" else None,
        "n_attack_train": int((y_train == 1).sum()) if label_mode == "binary" else None,
        "modifiable_mask": mask.tolist(),
        "notes": spec.notes,
        "seed": int(random_state),
        "test_size": "official",
        "val_size": float(val_size),
    }
    write_json(artifacts / "dataset_meta.json", meta)
    logger.info(
        "Prepared official %s: train=%d val=%d test=%d features=%d",
        spec.name, len(X_train), len(X_val), len(X_test), X_train.shape[1],
    )
    class_names = ["BENIGN", "ATTACK"] if label_mode == "binary" else multi_names
    return DatasetBundle(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=target_te,
        feature_names=feature_names, modifiable_mask=mask, scaler=scaler,
        dataset_name=spec.name, label_mode=label_mode, class_names=class_names,
        y_multi_train=ym_train, y_multi_val=ym_val, y_multi_test=ym_te,
        multi_class_names=multi_names, meta=meta, scaler_path=str(scaler_path),
    )


def prepare_dataset(
    file_path,
    test_size: float = 0.2,
    random_state: int = 42,
    artifacts_dir: str | Path = "artifacts",
    *,
    dataset_name: str | None = None,
    val_size: float = 0.0,
    label_mode: str = "binary",
    max_files: int | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Load a CSV/directory, clean, scale, and persist the scaler.

    ``val_size`` defaults to 0 so the original CLI (train + test only) stays
    unchanged. Experiment configs should pass a positive val_size.
    When ``file_path`` is a directory containing both official UNSW train/test
    CSVs, those files are used as the split (scaler fit on train only).
    """
    official = official_unsw_split_paths(file_path)
    name = normalize_dataset_name(dataset_name) if dataset_name else None
    if official and (name in {None, "unsw_nb15"}):
        train_df = load_traffic_table(official[0])
        test_df = load_traffic_table(official[1])
        bundle = prepare_from_official_frames(
            train_df,
            test_df,
            dataset_name="unsw_nb15",
            val_size=val_size,
            random_state=random_state,
            label_mode=label_mode,
            artifacts_dir=artifacts_dir,
            max_samples=max_samples,
        )
        return bundle.as_legacy_dict()
    df = load_traffic_table(file_path, max_files=max_files)
    bundle = prepare_from_frame(
        df,
        dataset_name=dataset_name,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        label_mode=label_mode,
        artifacts_dir=artifacts_dir,
        max_samples=max_samples,
    )
    return bundle.as_legacy_dict()
