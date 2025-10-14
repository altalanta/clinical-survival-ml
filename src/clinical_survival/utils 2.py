"""Utility helpers used across the clinical survival pipeline."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:  # scikit-learn optional in some runtime contexts
    import sklearn  # type: ignore
except Exception:  # pragma: no cover - allow running without sklearn at import time
    sklearn = None  # type: ignore

try:  # xgboost is optional; seed only if present
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None  # type: ignore


@dataclass
class DatasetSplits:
    """Container for train/validation/test splits."""

    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Persist a dictionary as JSON with UTF-8 encoding."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_global_seed(seed: int) -> None:
    """Seed all supported RNGs for reproducibility and configure sklearn output."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if sklearn is not None:  # pragma: no branch - guard for optional import
        sklearn.set_config(transform_output="pandas")
    if xgb is not None:  # pragma: no cover - executed only when xgboost installed
        xgb.set_config(verbosity=0)


def prepare_features(
    df: pd.DataFrame,
    feature_spec: dict[str, Iterable[str]],
    drop_missing: bool = False,
) -> tuple[pd.DataFrame, dict[str, Iterable[str]]]:
    """Select, drop and return covariate features according to configuration."""

    feature_spec = feature_spec.copy()
    drops = feature_spec.get("drop", [])
    if drops:
        df = df.drop(columns=[col for col in drops if col in df.columns], errors="ignore")

    keep_columns = list(feature_spec.get("numeric", [])) + list(feature_spec.get("categorical", []))
    keep_columns = [col for col in keep_columns if col in df.columns]
    features = df[keep_columns].copy()
    if drop_missing:
        features = features.dropna(axis=0)
    return features, feature_spec


def stratified_event_split(
    df: pd.DataFrame,
    event_col: str,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into train/test while approximately preserving event rate."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be within (0, 1)")

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    positive = df[df[event_col] == 1]
    negative = df[df[event_col] == 0]

    n_test = int(len(df) * test_size)
    n_pos_test = int(len(positive) * test_size)
    n_neg_test = n_test - n_pos_test

    test_df = pd.concat([positive.iloc[:n_pos_test], negative.iloc[:n_neg_test]], axis=0).sample(
        frac=1.0, random_state=seed
    )
    train_df = df.drop(test_df.index)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def combine_survival_target(time: pd.Series, event: pd.Series) -> pd.Series:
    """Pack time and event columns into a structured array required by scikit-survival."""

    from sksurv.util import Surv

    return pd.Series(Surv.from_arrays(event.astype(bool), time.astype(float)), index=time.index)
