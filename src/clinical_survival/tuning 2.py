"""Model tuning and cross-validation workflow."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from clinical_survival.models import BaseSurvivalModel


@dataclass
class FoldPrediction:
    """Container describing a single outer-fold prediction set."""

    fold: int
    params: dict[str, Any]
    train_indices: np.ndarray
    test_indices: np.ndarray
    risk: np.ndarray
    survival: np.ndarray
    preprocessor_summary: dict[str, np.ndarray]


@dataclass
class NestedCVResult:
    """Result produced by :func:`nested_cv`."""

    model_name: str
    best_params: dict[str, Any]
    estimator: BaseSurvivalModel
    folds: list[FoldPrediction]


def _structured_target(time: pd.Series, event: pd.Series) -> np.ndarray:
    return Surv.from_arrays(event.astype(bool), time.astype(float))


def _parameter_grid(grid: dict[str, Iterable[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [list(grid[key]) for key in keys]
    return [dict(zip(keys, combination, strict=True)) for combination in itertools.product(*values)]


def _extract_preprocessor_summary(estimator: BaseSurvivalModel) -> dict[str, np.ndarray]:
    if not hasattr(estimator, "pipeline"):
        return {}
    preprocessor = estimator.pipeline.named_steps.get("pre")  # type: ignore[attr-defined]
    summary: dict[str, np.ndarray] = {}
    if preprocessor is None:
        return summary
    numeric = getattr(preprocessor, "named_transformers_", {}).get("numeric")
    if numeric is None or not hasattr(numeric, "named_steps"):
        return summary
    imputer = numeric.named_steps.get("impute")
    if imputer is not None and hasattr(imputer, "statistics_"):
        summary["numeric_imputer"] = np.asarray(imputer.statistics_, dtype=float)
    return summary


def nested_cv(
    model_name: str,
    X: pd.DataFrame,
    time: pd.Series,
    event: pd.Series,
    outer_splits: int,
    inner_splits: int,
    param_grid: dict[str, Iterable[Any]],
    eval_times: Iterable[float],
    *,
    random_state: int,
    pipeline_builder: Callable[[dict[str, Any]], BaseSurvivalModel],
) -> NestedCVResult:
    """Perform nested cross-validation returning fold predictions and best estimator."""

    y_struct = _structured_target(time, event)
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    param_candidates = _parameter_grid(param_grid)

    folds: list[FoldPrediction] = []
    best_score = -np.inf
    best_params_global: dict[str, Any] | None = None

    for fold_idx, (train_index, test_index) in enumerate(outer.split(X, event), start=1):
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y_struct[train_index], y_struct[test_index]
        event_train = event.iloc[train_index]

        best_inner_score = -np.inf
        best_params_fold: dict[str, Any] = {}

        for params in param_candidates:
            inner_scores: list[float] = []
            for inner_train_idx, inner_val_idx in inner.split(X_train, event_train):
                X_inner_train = X_train.iloc[inner_train_idx]
                X_inner_val = X_train.iloc[inner_val_idx]
                y_inner_train = y_train[inner_train_idx]
                y_inner_val = y_train[inner_val_idx]

                estimator = pipeline_builder(params)
                estimator.fit(X_inner_train, y_inner_train)
                risk = estimator.predict_risk(X_inner_val)
                cindex = concordance_index_censored(
                    y_inner_val["event"],
                    y_inner_val["time"],
                    -risk,
                )[0]
                inner_scores.append(float(cindex))

            score = float(np.mean(inner_scores)) if inner_scores else -np.inf
            if score > best_inner_score:
                best_inner_score = score
                best_params_fold = dict(params)

        estimator = pipeline_builder(best_params_fold)
        estimator.fit(X_train, y_train)
        risk_valid = estimator.predict_risk(X_valid)
        survival_valid = estimator.predict_survival_function(X_valid, eval_times)
        fold_cindex = concordance_index_censored(
            y_valid["event"],
            y_valid["time"],
            -risk_valid,
        )[0]

        summary = _extract_preprocessor_summary(estimator)

        folds.append(
            FoldPrediction(
                fold=fold_idx,
                params=dict(best_params_fold),
                train_indices=train_index.copy(),
                test_indices=test_index.copy(),
                risk=np.asarray(risk_valid, dtype=float),
                survival=np.asarray(survival_valid, dtype=float),
                preprocessor_summary=summary,
            )
        )

        if float(fold_cindex) > best_score:
            best_score = float(fold_cindex)
            best_params_global = dict(best_params_fold)

    if best_params_global is None:
        best_params_global = {}

    final_estimator = pipeline_builder(best_params_global)
    final_estimator.fit(X, y_struct)

    return NestedCVResult(
        model_name=model_name,
        best_params=best_params_global,
        estimator=final_estimator,
        folds=folds,
    )
