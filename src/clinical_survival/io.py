"""Dataset loading utilities supporting optional external validation splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from clinical_survival.utils import load_yaml

REQUIRED_COLUMNS = {"time", "event"}


def load_dataset(
    csv_path: str | Path,
    metadata_path: str | Path,
    *,
    time_col: str = "time",
    event_col: str = "event",
    external_config: dict[str, object] | None = None,
) -> tuple[
    tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame] | None, dict[str, object]
]:
    """Load dataset and optional external validation split.

    Returns
    -------
    (X_train, y_train), (X_external, y_external or ``None``), metadata
    """

    csv_path = Path(csv_path)
    metadata_path = Path(metadata_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    metadata = load_yaml(metadata_path)
    external_cfg = external_config.copy() if external_config else {}
    group_column = external_cfg.get("group_column", "group")
    train_value = external_cfg.get("train_value", "train")
    external_value = external_cfg.get("external_value", "external")
    external_csv = external_cfg.get("csv")

    main_df = pd.read_csv(csv_path)
    frames = []

    if external_csv:
        external_path = Path(external_csv)
        if not external_path.exists():
            raise FileNotFoundError(external_path)
        external_df = pd.read_csv(external_path)
        main_df["__split"] = "train"
        external_df["__split"] = "external"
        frames = [main_df, external_df]
    elif group_column in main_df.columns:
        mask_external = main_df[group_column].astype(str) == str(external_value)
        mask_train = (
            main_df[group_column].astype(str) == str(train_value)
            if train_value is not None
            else ~mask_external
        )
        external_df = main_df[mask_external].copy()
        main_df = main_df[mask_train].copy()
        main_df["__split"] = "train"
        external_df["__split"] = "external"
        frames = [main_df, external_df]
    else:
        main_df["__split"] = "train"
        frames = [main_df]
        external_df = None

    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined = _apply_metadata_types(combined, metadata)

    train_df = (
        combined[combined["__split"] == "train"].drop(columns="__split").reset_index(drop=True)
    )
    external_split = None
    external_count = 0
    if "external" in combined["__split"].values:
        ext_df = (
            combined[combined["__split"] == "external"]
            .drop(columns="__split")
            .reset_index(drop=True)
        )
        external_split = _split_features_target(ext_df, time_col, event_col, group_column)
        external_count = len(ext_df)

    train_split = _split_features_target(train_df, time_col, event_col, group_column)
    metadata["n_samples"] = len(train_df)
    metadata["external_present"] = external_split is not None
    metadata["n_external_samples"] = external_count
    return train_split, external_split, metadata


def _apply_metadata_types(df: pd.DataFrame, metadata: dict[str, object]) -> pd.DataFrame:
    column_meta = metadata.get("columns", {})
    df = df.copy()
    categorical_columns = []
    for column, spec in column_meta.items():
        if column not in df.columns:
            continue
        dtype = spec.get("dtype") if isinstance(spec, dict) else None
        if dtype == "category":
            df[column] = df[column].astype("category")
            categorical_columns.append(column)
        elif dtype == "float":
            df[column] = pd.to_numeric(df[column], errors="coerce")
        elif dtype == "int":
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    if categorical_columns:
        categories = {col: df[col].cat.categories for col in categorical_columns}
        for col in categorical_columns:
            df[col] = pd.Categorical(df[col], categories=categories[col])
    return df


def _split_features_target(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    group_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_required = REQUIRED_COLUMNS.difference(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    feature_df = df.drop(
        columns=[col for col in [time_col, event_col, group_col] if col in df.columns]
    )
    target_df = df[[time_col, event_col]].copy()
    target_df[event_col] = target_df[event_col].astype(int)
    target_df[time_col] = target_df[time_col].astype(float)
    return feature_df, target_df
