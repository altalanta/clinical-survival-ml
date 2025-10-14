"""Preprocessing pipelines for survival modeling."""

from __future__ import annotations

from collections.abc import Iterable

from sklearn.compose import ColumnTransformer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_transformer(
    feature_spec: dict[str, Iterable[str]],
    *,
    strategy: str = "iterative",
    max_iter: int = 10,
    initial_strategy: str = "median",
    random_state: int | None = None,
) -> ColumnTransformer:
    """Create a column transformer based on feature specification."""

    numeric_features = list(feature_spec.get("numeric", []))
    categorical_features = list(feature_spec.get("categorical", []))

    if strategy == "iterative":
        numeric_imputer = IterativeImputer(
            max_iter=max_iter,
            random_state=random_state,
            initial_strategy=initial_strategy,
        )
    else:
        numeric_imputer = SimpleImputer(strategy=initial_strategy)

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", numeric_imputer),
            ("scale", StandardScaler()),
        ]
    )

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - compatibility with older sklearn
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", encoder),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return transformer


def build_preprocessor(
    feature_spec: dict[str, Iterable[str]],
    missing_cfg: dict[str, object],
    *,
    random_state: int,
) -> ColumnTransformer:
    """Return a column transformer configured for the current fold."""

    return build_transformer(
        feature_spec,
        strategy=str(missing_cfg.get("strategy", "iterative")),
        max_iter=int(missing_cfg.get("max_iter", 10)),
        initial_strategy=str(missing_cfg.get("initial_strategy", "median")),
        random_state=random_state,
    )
