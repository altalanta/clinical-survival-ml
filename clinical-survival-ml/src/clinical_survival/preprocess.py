"""Preprocessing pipelines for survival modeling."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # noqa: F401 - enable iterative imputer
    from sklearn.experimental import enable_iterative_imputer  # type: ignore
except ImportError:  # pragma: no cover
    enable_iterative_imputer = None  # type: ignore

from sklearn.impute import IterativeImputer


def build_transformer(
    feature_spec: Dict[str, Iterable[str]],
    strategy: str = "iterative",
    max_iter: int = 10,
    initial_strategy: str = "median",
) -> ColumnTransformer:
    """Create a column transformer based on feature specification."""

    numeric_features = list(feature_spec.get("numeric", []))
    categorical_features = list(feature_spec.get("categorical", []))

    if strategy == "iterative":
        numeric_imputer = IterativeImputer(max_iter=max_iter, random_state=0, initial_strategy=initial_strategy)
    else:
        numeric_imputer = SimpleImputer(strategy=initial_strategy)

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", numeric_imputer),
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", sparse=False),
            ),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    return transformer


def fit_transform(
    transformer: ColumnTransformer,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """Fit the transformer and return the transformed frame."""

    transformed = transformer.fit_transform(X, y)
    feature_names = _feature_names(transformer)
    transformed_df = pd.DataFrame(transformed, index=X.index, columns=feature_names)
    return transformed_df, transformer


def transform(transformer: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    """Apply a fitted transformer to new data."""

    transformed = transformer.transform(X)
    feature_names = _feature_names(transformer)
    return pd.DataFrame(transformed, index=X.index, columns=feature_names)


def _feature_names(transformer: ColumnTransformer) -> Iterable[str]:
    try:
        names = transformer.get_feature_names_out()
    except AttributeError:  # pragma: no cover - fallback for older versions
        total_features = 0
        for name, pipeline, columns in transformer.transformers_:
            if columns is None:
                continue
            if hasattr(columns, "__len__"):
                total_features += len(columns)
        names = [f"feature_{i}" for i in range(total_features)]
    return names
