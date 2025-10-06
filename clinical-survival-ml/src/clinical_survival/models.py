"""Model wrappers for survival estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

try:  # pragma: no cover - optional dependency
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None


def _breslow_survival(times: np.ndarray, events: np.ndarray, risks: np.ndarray, eval_times: np.ndarray) -> np.ndarray:
    """Compute baseline survival using Breslow estimator."""

    order = np.argsort(times)
    times = times[order]
    events = events[order]
    risks = risks[order]

    unique_times = np.unique(times[events == 1])
    baseline_hazard = np.zeros_like(unique_times, dtype=float)
    cumulative = 0.0

    for idx, t in enumerate(unique_times):
        event_mask = (times == t) & (events == 1)
        risk_set = risks[times >= t]
        hazard_increment = events[event_mask].sum() / np.clip(risk_set.sum(), a_min=1e-8, a_max=None)
        cumulative += hazard_increment
        baseline_hazard[idx] = cumulative

    baseline_survival = np.exp(-baseline_hazard)
    survival_at_eval = np.ones((len(risks), len(eval_times)))

    for j, r in enumerate(risks):
        surv_values = np.interp(eval_times, unique_times, baseline_survival, left=1.0, right=baseline_survival[-1])
        survival_at_eval[j, :] = np.power(surv_values, r)
    return survival_at_eval


@dataclass
class BaseSurvivalModel:
    """Base wrapper exposing a shared interface."""

    model: object

    def fit(self, X: pd.DataFrame, y: Sequence) -> "BaseSurvivalModel":
        raise NotImplementedError

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_survival_function(self, X: pd.DataFrame, times: Iterable[float]) -> np.ndarray:
        raise NotImplementedError


class CoxPHModel(BaseSurvivalModel):
    """Cox proportional hazards model wrapper."""

    def __init__(self, **params: object) -> None:
        super().__init__(CoxPHSurvivalAnalysis(**params))
        self._baseline_survival_: Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y: Sequence) -> "CoxPHModel":
        self.model.fit(X, y)
        return self

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_survival_function(self, X: pd.DataFrame, times: Iterable[float]) -> np.ndarray:
        surv_funcs = self.model.predict_survival_function(X)
        times = np.asarray(list(times))
        output = np.vstack([fn(times) for fn in surv_funcs])
        return output


class RSFModel(BaseSurvivalModel):
    """Random survival forest wrapper."""

    def __init__(self, **params: object) -> None:
        default_params = {
            "n_estimators": 200,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": 0,
        }
        default_params.update(params)
        super().__init__(RandomSurvivalForest(**default_params))

    def fit(self, X: pd.DataFrame, y: Sequence) -> "RSFModel":
        self.model.fit(X, y)
        return self

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        return -self.model.predict(X)

    def predict_survival_function(self, X: pd.DataFrame, times: Iterable[float]) -> np.ndarray:
        surv_funcs = self.model.predict_survival_function(X)
        times = np.asarray(list(times))
        return np.vstack([fn(times) for fn in surv_funcs])


class XGBCoxModel(BaseSurvivalModel):
    """XGBoost-based Cox model."""

    def __init__(self, **params: object) -> None:
        if xgb is None:  # pragma: no cover
            raise ImportError("xgboost is required for XGBCoxModel")
        default_params = {
            "objective": "survival:cox",
            "eval_metric": "cox-nloglik",
            "tree_method": "hist",
            "learning_rate": 0.05,
            "max_depth": 3,
        }
        default_params.update(params)
        self.params = default_params
        super().__init__(None)
        self._booster: Optional[xgb.Booster] = None
        self._train_risk_: Optional[np.ndarray] = None
        self._train_time_: Optional[np.ndarray] = None
        self._train_event_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: Sequence) -> "XGBCoxModel":
        times = np.asarray([record[1] for record in y], dtype=float)
        events = np.asarray([record[0] for record in y], dtype=float)
        dtrain = xgb.DMatrix(X, label=times, weight=np.maximum(events, 1e-3))
        self._booster = xgb.train(self.params, dtrain, num_boost_round=self.params.get("n_estimators", 200))
        self._train_time_ = times
        self._train_event_ = events
        self._train_risk_ = np.exp(self._booster.predict(dtrain))
        return self

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("Model not fitted")
        dtest = xgb.DMatrix(X)
        return self._booster.predict(dtest)

    def predict_survival_function(self, X: pd.DataFrame, times: Iterable[float]) -> np.ndarray:
        if self._booster is None or self._train_time_ is None or self._train_event_ is None or self._train_risk_ is None:
            raise RuntimeError("Model not fitted")
        eval_times = np.asarray(list(times))
        dtest = xgb.DMatrix(X)
        risks = np.exp(self._booster.predict(dtest))
        surv = _breslow_survival(self._train_time_, self._train_event_, risks, eval_times)
        return surv


class XGBAFTModel(BaseSurvivalModel):
    """XGBoost accelerated failure time model."""

    def __init__(self, **params: object) -> None:
        if xgb is None:  # pragma: no cover
            raise ImportError("xgboost is required for XGBAFTModel")
        default_params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "tree_method": "hist",
            "learning_rate": 0.05,
            "max_depth": 3,
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": 1.0,
        }
        default_params.update(params)
        self.params = default_params
        super().__init__(None)
        self._booster: Optional[xgb.Booster] = None
        self._train_time_: Optional[np.ndarray] = None
        self._train_event_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: Sequence) -> "XGBAFTModel":
        times = np.asarray([record[1] for record in y], dtype=float)
        events = np.asarray([record[0] for record in y], dtype=float)
        lower = np.where(events == 1, times, np.zeros_like(times))
        upper = np.where(events == 1, times, times)
        dtrain = xgb.DMatrix(X)
        dtrain.set_float_info("label_lower_bound", lower)
        dtrain.set_float_info("label_upper_bound", upper)
        self._booster = xgb.train(self.params, dtrain, num_boost_round=self.params.get("n_estimators", 300))
        self._train_time_ = times
        self._train_event_ = events
        return self

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("Model not fitted")
        dtest = xgb.DMatrix(X)
        return -self._booster.predict(dtest)

    def predict_survival_function(self, X: pd.DataFrame, times: Iterable[float]) -> np.ndarray:
        if self._booster is None or self._train_time_ is None or self._train_event_ is None:
            raise RuntimeError("Model not fitted")
        eval_times = np.asarray(list(times))
        dtest = xgb.DMatrix(X)
        preds = self._booster.predict(dtest)
        scale = np.std(self._train_time_) if np.std(self._train_time_) > 0 else 1.0
        survival = np.array([
            np.exp(-np.maximum(0.0, eval_times - pred) / scale) for pred in preds
        ])
        return np.clip(survival, 0.0, 1.0)


def make_model(name: str, **params: object) -> BaseSurvivalModel:
    """Factory method returning a survival model wrapper."""

    name = name.lower()
    if name == "coxph":
        return CoxPHModel(**params)
    if name == "rsf":
        return RSFModel(**params)
    if name == "xgb_cox":
        return XGBCoxModel(**params)
    if name == "xgb_aft":
        return XGBAFTModel(**params)
    raise ValueError(f"Unknown model: {name}")
