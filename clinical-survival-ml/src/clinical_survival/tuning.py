"""Model tuning and cross-validation workflow."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.util import Surv

from clinical_survival.models import BaseSurvivalModel, make_model


@dataclass
class FoldResult:
    model_name: str
    params: Dict[str, object]
    cindex: float
    ibs: float
    fold: int
    train_size: int
    test_size: int


def _structured_target(time: pd.Series, event: pd.Series) -> np.ndarray:
    return Surv.from_arrays(event.astype(bool), time.astype(float))


def _parameter_grid(grid: Dict[str, Iterable[object]]) -> List[Dict[str, object]]:
    keys = list(grid.keys())
    values = [list(grid[key]) for key in keys]
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)] or [{}]


def nested_cv(
    model_name: str,
    X: pd.DataFrame,
    time: pd.Series,
    event: pd.Series,
    outer_splits: int,
    inner_splits: int,
    param_grid: Dict[str, Iterable[object]],
    eval_times: Iterable[float],
    random_state: int = 42,
) -> Tuple[BaseSurvivalModel, List[FoldResult]]:
    """Perform nested cross-validation returning the best fitted model."""

    y_struct = _structured_target(time, event)
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

    parameter_grid = _parameter_grid(param_grid)
    fold_results: List[FoldResult] = []
    best_global_score = -np.inf
    best_model: BaseSurvivalModel | None = None
    best_params_global: Dict[str, object] | None = None

    for fold_idx, (train_index, test_index) in enumerate(outer.split(X, event), start=1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_struct[train_index], y_struct[test_index]
        best_inner_score = -np.inf
        best_params: Dict[str, object] = {}

        for params in parameter_grid:
            cindex_scores = []
            ibs_scores = []
            for inner_train, inner_valid in inner.split(X_train, event.iloc[train_index]):
                X_inner_train = X_train.iloc[inner_train]
                X_inner_valid = X_train.iloc[inner_valid]
                y_inner_train = y_train[inner_train]
                y_inner_valid = y_train[inner_valid]

                model = make_model(model_name, **params)
                model.fit(X_inner_train, y_inner_train)
                risk = model.predict_risk(X_inner_valid)
                cindex = concordance_index_censored(
                    y_inner_valid["event"],
                    y_inner_valid["time"],
                    -risk,
                )[0]
                surv = model.predict_survival_function(X_inner_valid, eval_times)
                ibs = integrated_brier_score(
                    y_inner_train,
                    y_inner_valid,
                    surv,
                    np.asarray(list(eval_times)),
                )
                cindex_scores.append(cindex)
                ibs_scores.append(ibs)

            mean_cindex = float(np.mean(cindex_scores))
            mean_ibs = float(np.mean(ibs_scores))
            score = mean_cindex - mean_ibs
            if score > best_inner_score:
                best_inner_score = score
                best_params = params

        model = make_model(model_name, **best_params)
        model.fit(X_train, y_train)
        risk = model.predict_risk(X_test)
        fold_cindex = concordance_index_censored(y_test["event"], y_test["time"], -risk)[0]
        surv = model.predict_survival_function(X_test, eval_times)
        fold_ibs = integrated_brier_score(y_train, y_test, surv, np.asarray(list(eval_times)))
        fold_results.append(
            FoldResult(
                model_name=model_name,
                params=best_params,
                cindex=float(fold_cindex),
                ibs=float(fold_ibs),
                fold=fold_idx,
                train_size=len(X_train),
                test_size=len(X_test),
            )
        )
        final_score = fold_cindex - fold_ibs
        if final_score > best_global_score:
            best_global_score = final_score
            best_model = model
            best_params_global = best_params

    if best_model is None:
        raise RuntimeError("Failed to fit model")
    if best_params_global is None:
        best_params_global = {}
    final_model = make_model(model_name, **best_params_global)
    final_model.fit(X, y_struct)
    return final_model, fold_results
