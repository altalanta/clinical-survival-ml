"""Utility helpers used across the clinical survival pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import yaml


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


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file into a dictionary."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_json(payload: Dict[str, Any], path: str | Path) -> None:
    """Persist a dictionary as JSON with UTF-8 encoding."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_global_seed(seed: int) -> None:
    """Seed numpy and pandas randomness for reproducibility."""

    np.random.seed(seed)


def prepare_features(
    df: pd.DataFrame,
    feature_spec: Dict[str, Iterable[str]],
    drop_missing: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Iterable[str]]]:
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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into train/test while approximately preserving event rate."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be within (0, 1)")

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    positive = df[df[event_col] == 1]
    negative = df[df[event_col] == 0]

    n_test = int(len(df) * test_size)
    n_pos_test = int(len(positive) * test_size)
    n_neg_test = n_test - n_pos_test

    test_df = pd.concat(
        [positive.iloc[:n_pos_test], negative.iloc[:n_neg_test]], axis=0
    ).sample(frac=1.0, random_state=seed)
    train_df = df.drop(test_df.index)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def combine_survival_target(time: pd.Series, event: pd.Series) -> pd.Series:
    """Pack time and event columns into a structured array required by scikit-survival."""

    from sksurv.util import Surv

    return pd.Series(Surv.from_arrays(event.astype(bool), time.astype(float)), index=time.index)
