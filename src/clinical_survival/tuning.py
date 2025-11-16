"""Model tuning and cross-validation workflow."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Dict, Any, Callable

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sklearn.model_selection import KFold
from rich.console import Console

import optuna

from clinical_survival.config import ParamsConfig, FeaturesConfig
from clinical_survival.models import make_model
from clinical_survival.preprocess import build_preprocessor
from sklearn.pipeline import Pipeline

console = Console()


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


def _suggest_params(trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
    """Suggests hyperparameters for a trial based on the search space config."""
    params = {}
    for name, config in search_space.items():
        param_type = config.pop("type")
        if param_type == "categorical":
            params[name] = trial.suggest_categorical(name, **config)
        elif param_type == "float":
            params[name] = trial.suggest_float(name, **config)
        elif param_type == "int":
            params[name] = trial.suggest_int(name, **config)
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")
    return params


def _create_objective(
    X: pd.DataFrame,
    y_surv: np.ndarray,
    model_name: str,
    search_space: Dict[str, Any],
    params_config: ParamsConfig,
    features_config: FeaturesConfig,
) -> Callable[[optuna.Trial], float]:
    """Creates the objective function for Optuna study."""

    def objective(trial: optuna.Trial) -> float:
        """The objective function to be minimized/maximized."""
        model_params = _suggest_params(trial, search_space)

        scores = []
        kf = KFold(
            n_splits=params_config.inner_splits,
            shuffle=True,
            random_state=params_config.seed,
        )

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_surv[train_idx], y_surv[val_idx]

            preprocessor = build_preprocessor(
                features_config.model_dump(),
                params_config.missing.model_dump(),
                random_state=params_config.seed,
            )
            model = make_model(model_name, random_state=params_config.seed, **model_params)
            pipeline = Pipeline([("pre", preprocessor), ("est", model)])

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_val)
            
            event_indicator = y_val["event"]
            time_to_event = y_val["time"]
            
            score, _, _, _, _ = concordance_index_censored(
                event_indicator, time_to_event, preds
            )
            scores.append(score)

        return np.mean(scores)

    return objective


def run_tuning(
    X: pd.DataFrame,
    y_surv: np.ndarray,
    model_name: str,
    search_space: Dict[str, Any],
    params_config: ParamsConfig,
    features_config: FeaturesConfig,
) -> Dict[str, Any]:
    """
    Runs hyperparameter tuning for a given model.
    """
    console.print(
        f"🔎 Starting hyperparameter tuning for [bold cyan]{model_name}[/bold cyan]..."
    )

    study = optuna.create_study(direction=params_config.tuning.direction)
    objective = _create_objective(
        X, y_surv, model_name, search_space, params_config, features_config
    )

    study.optimize(objective, n_trials=params_config.tuning.trials, n_jobs=-1)

    console.print(
        f"✅ Tuning complete for [bold cyan]{model_name}[/bold cyan]. "
        f"Best score: {study.best_value:.4f}"
    )
    return study.best_params
