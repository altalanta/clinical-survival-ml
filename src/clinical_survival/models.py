"""Model wrappers for survival estimators."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import erf, sqrt

import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis

from clinical_survival.model_plugins import BaseSurvivalModel, ModelRegistry # Import ModelRegistry

try:  # pragma: no cover - optional dependency
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

# Import ensemble methods
try:
    from clinical_survival.ensembles import BaggingEnsemble, DynamicEnsemble, StackingEnsemble
except ImportError:  # pragma: no cover
    BaggingEnsemble = None  # type: ignore
    DynamicEnsemble = None  # type: ignore
    StackingEnsemble = None  # type: ignore

# Import GPU utilities
try:
    from clinical_survival.gpu_utils import create_gpu_accelerator
    GPU_UTILS_AVAILABLE = True
except ImportError:  # pragma: no cover
    GPU_UTILS_AVAILABLE = False


def _breslow_survival(
    times: np.ndarray, events: np.ndarray, risks: np.ndarray, eval_times: np.ndarray
) -> np.ndarray:
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
        hazard_increment = events[event_mask].sum() / np.clip(
            risk_set.sum(), a_min=1e-8, a_max=None
        )
        cumulative += hazard_increment
        baseline_hazard[idx] = cumulative

    baseline_survival = np.exp(-baseline_hazard)
    survival_at_eval = np.ones((len(risks), len(eval_times)))

    for j, r in enumerate(risks):
        surv_values = np.interp(
            eval_times, unique_times, baseline_survival, left=1.0, right=baseline_survival[-1]
        )
        survival_at_eval[j, :] = np.power(surv_values, r)
    return survival_at_eval


def _ensure_dataframe(
    X: pd.DataFrame | np.ndarray, columns: list[str] | None = None
) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    if columns is None:
        columns = [f"feature_{idx}" for idx in range(X.shape[1])]
    return pd.DataFrame(X, columns=columns)


@ModelRegistry.register # Register CoxPHModel
class CoxPHModel(BaseSurvivalModel):
    """Cox proportional hazards model wrapper."""
    name: str = "coxph" # Add name attribute as required by ABC

    def __init__(self, **params: object) -> None:
        self.model = CoxPHSurvivalAnalysis(**params)

    def fit(self, X: pd.DataFrame | np.ndarray, y: Sequence) -> CoxPHModel:
        frame = _ensure_dataframe(X)
        self.feature_names_ = list(frame.columns)
        self.model.fit(frame, y)
        return self

    def predict_risk(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        frame = _ensure_dataframe(X, self.feature_names_)
        return self.model.predict(frame)

    def predict_survival_function(
        self, X: pd.DataFrame | np.ndarray, times: Iterable[float]
    ) -> np.ndarray:
        frame = _ensure_dataframe(X, self.feature_names_)
        surv_funcs = self.model.predict_survival_function(frame)
        times = np.asarray(list(times))
        output = np.vstack([fn(times) for fn in surv_funcs])
        return output


@ModelRegistry.register # Register RSFModel
class RSFModel(BaseSurvivalModel):
    """Random survival forest wrapper."""
    name: str = "rsf" # Add name attribute as required by ABC

    def __init__(self, **params: object) -> None:
        default_params = {
            "n_estimators": 200,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
        }
        default_params.update(params)
        self.model = RandomSurvivalForest(**default_params)

    def fit(self, X: pd.DataFrame | np.ndarray, y: Sequence) -> RSFModel:
        frame = _ensure_dataframe(X)
        self.feature_names_ = list(frame.columns)
        self.model.fit(frame, y)
        return self

    def predict_risk(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        frame = _ensure_dataframe(X, self.feature_names_)
        return -self.model.predict(frame)

    def predict_survival_function(
        self, X: pd.DataFrame | np.ndarray, times: Iterable[float]
    ) -> np.ndarray:
        frame = _ensure_dataframe(X, self.feature_names_)
        surv_funcs = self.model.predict_survival_function(frame)
        times = np.asarray(list(times))
        return np.vstack([fn(times) for fn in surv_funcs])


@ModelRegistry.register # Register XGBCoxModel
class XGBCoxModel(BaseSurvivalModel):
    """XGBoost-based Cox model with GPU acceleration."""

    def __init__(self, use_gpu: bool = True, gpu_id: int = 0, **params: object) -> None:
        if xgb is None:  # pragma: no cover
            raise ImportError("xgboost is required for XGBCoxModel")

        # Initialize GPU accelerator
        self.gpu_accelerator = None
        if GPU_UTILS_AVAILABLE and use_gpu:
            self.gpu_accelerator = create_gpu_accelerator(use_gpu=use_gpu, gpu_id=gpu_id)

        default_params = {
            "objective": "survival:cox",
            "eval_metric": "cox-nloglik",
            "tree_method": "hist",
            "learning_rate": 0.05,
            "max_depth": 3,
        }

        # Add GPU parameters if available
        if self.gpu_accelerator is not None:
            gpu_params = self.gpu_accelerator.get_gpu_params("xgb")
            default_params.update(gpu_params)

        default_params.update(params)
        self.params = default_params
        self.model = None # Set model to None initially as required by ABC
        self.name = "xgb_cox" # Set name attribute as required by ABC
        self._booster: xgb.Booster | None = None
        self._train_risk_: np.ndarray | None = None
        self._train_time_: np.ndarray | None = None
        self._train_event_: np.ndarray | None = None
        self.use_gpu = use_gpu

    def fit(self, X: pd.DataFrame | np.ndarray, y: Sequence) -> XGBCoxModel:
        frame = _ensure_dataframe(X)
        times = np.asarray([record[1] for record in y], dtype=float)
        events = np.asarray([record[0] for record in y], dtype=float)
        self.feature_names_ = list(frame.columns)
        dtrain = xgb.DMatrix(
            frame.to_numpy(),
            feature_names=self.feature_names_,
            label=times,
            weight=np.maximum(events, 1e-3),
        )
        self._booster = xgb.train(
            self.params, dtrain, num_boost_round=self.params.get("n_estimators", 200)
        )
        self.model = self._booster # Set the booster as the model after fitting
        self._train_time_ = times
        self._train_event_ = events
        self._train_risk_ = np.exp(self._booster.predict(dtrain))
        return self

    def predict_risk(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("Model not fitted")
        frame = _ensure_dataframe(X, self.feature_names_)
        dtest = xgb.DMatrix(frame.to_numpy(), feature_names=self.feature_names_)
        return self._booster.predict(dtest)

    def predict_survival_function(
        self, X: pd.DataFrame | np.ndarray, times: Iterable[float]
    ) -> np.ndarray:
        if (
            self._booster is None
            or self._train_time_ is None
            or self._train_event_ is None
            or self._train_risk_ is None
        ):
            raise RuntimeError("Model not fitted")
        eval_times = np.asarray(list(times))
        frame = _ensure_dataframe(X, self.feature_names_)
        dtest = xgb.DMatrix(frame.to_numpy(), feature_names=self.feature_names_)
        risks = np.exp(self._booster.predict(dtest))
        surv = _breslow_survival(self._train_time_, self._train_event_, risks, eval_times)
        return surv


@ModelRegistry.register # Register XGBAFTModel
class XGBAFTModel(BaseSurvivalModel):
    """XGBoost accelerated failure time model with GPU acceleration."""

    def __init__(self, use_gpu: bool = True, gpu_id: int = 0, **params: object) -> None:
        if xgb is None:  # pragma: no cover
            raise ImportError("xgboost is required for XGBAFTModel")

        # Initialize GPU accelerator
        self.gpu_accelerator = None
        if GPU_UTILS_AVAILABLE and use_gpu:
            self.gpu_accelerator = create_gpu_accelerator(use_gpu=use_gpu, gpu_id=gpu_id)

        default_params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "tree_method": "hist",
            "learning_rate": 0.05,
            "max_depth": 3,
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": 1.0,
        }

        # Add GPU parameters if available
        if self.gpu_accelerator is not None:
            gpu_params = self.gpu_accelerator.get_gpu_params("xgb")
            default_params.update(gpu_params)

        default_params.update(params)
        self.params = default_params
        self.model = None # Set model to None initially as required by ABC
        self.name = "xgb_aft" # Set name attribute as required by ABC
        self._booster: xgb.Booster | None = None
        self._train_time_: np.ndarray | None = None
        self._train_event_: np.ndarray | None = None
        self.use_gpu = use_gpu

    def fit(self, X: pd.DataFrame | np.ndarray, y: Sequence) -> XGBAFTModel:
        frame = _ensure_dataframe(X)
        times = np.asarray([record[1] for record in y], dtype=float)
        events = np.asarray([record[0] for record in y], dtype=float)
        lower = np.where(events == 1, times, np.zeros_like(times))
        upper = np.where(events == 1, times, times)
        self.feature_names_ = list(frame.columns)
        dtrain = xgb.DMatrix(frame.to_numpy(), feature_names=self.feature_names_)
        dtrain.set_float_info("label_lower_bound", lower)
        dtrain.set_float_info("label_upper_bound", upper)
        self._booster = xgb.train(
            self.params, dtrain, num_boost_round=self.params.get("n_estimators", 300)
        )
        self.model = self._booster # Set the booster as the model after fitting
        self._train_time_ = times
        self._train_event_ = events
        return self

    def predict_risk(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("Model not fitted")
        frame = _ensure_dataframe(X, self.feature_names_)
        dtest = xgb.DMatrix(frame.to_numpy(), feature_names=self.feature_names_)
        return -self._booster.predict(dtest)

    def predict_survival_function(
        self, X: pd.DataFrame | np.ndarray, times: Iterable[float]
    ) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("Model not fitted")
        distribution = str(self.params.get("aft_loss_distribution", "normal")).lower()
        scale = float(self.params.get("aft_loss_distribution_scale", 1.0))
        if scale <= 0:
            raise ValueError("aft_loss_distribution_scale must be positive")
        frame = _ensure_dataframe(X, self.feature_names_)
        eval_times = np.asarray(list(times), dtype=float)
        eval_times = np.clip(eval_times, a_min=1e-8, a_max=None)
        dtest = xgb.DMatrix(frame.to_numpy(), feature_names=self.feature_names_)
        preds = self._booster.predict(dtest)  # log-location parameter

        log_times = np.log(eval_times)

        def _cdf(z: np.ndarray) -> np.ndarray:
            if distribution == "normal":
                return 0.5 * (1.0 + erf(z / sqrt(2.0)))
            if distribution == "logistic":
                return 1.0 / (1.0 + np.exp(-z))
            raise ValueError(f"Unsupported AFT distribution: {distribution}")

        survival = []
        for pred in preds:
            z = (log_times - pred) / scale
            cdf = _cdf(z)
            survival.append(1.0 - cdf)
        survival_arr = np.clip(np.asarray(survival), 0.0, 1.0)
        survival_arr = np.maximum.accumulate(survival_arr[:, ::-1], axis=1)[:, ::-1]
        return survival_arr


@ModelRegistry.register # Register PipelineModel
class PipelineModel(BaseSurvivalModel):
    """Wrapper around an sklearn pipeline exposing the survival interface."""
    name: str = "pipeline_model" # Add name attribute as required by ABC

    def __init__(self, pipeline) -> None:
        self.model = pipeline
        self.pipeline = pipeline
        self._estimator_step = "est"

    def fit(self, X: pd.DataFrame | np.ndarray, y: Sequence) -> PipelineModel:
        self.pipeline.fit(X, y)
        self._estimator = self.pipeline.named_steps[self._estimator_step]
        self._preprocessor = self.pipeline.named_steps.get("pre")
        return self

    def _transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        if self._preprocessor is None:
            return X
        return self._preprocessor.transform(X)

    def predict_risk(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        transformed = self._transform(X)
        return self._estimator.predict_risk(transformed)

    def predict_survival_function(
        self, X: pd.DataFrame | np.ndarray, times: Iterable[float]
    ) -> np.ndarray:
        transformed = self._transform(X)
        return self._estimator.predict_survival_function(transformed, times)


def make_model(
    name: str, *, random_state: int | None = None, use_gpu: bool = True, gpu_id: int = 0, **params: object
) -> BaseSurvivalModel:
    """Factory method returning a survival model wrapper with GPU support."""

    name = name.lower()

    # Handle ensemble models first, as they use make_model recursively
    if name == "stacking":
        if StackingEnsemble is None:
            raise ImportError("Ensemble methods not available")
        base_models_param = params.pop("base_models", ["coxph", "rsf"])
        base_models = [
            make_model(model_name, random_state=random_state) for model_name in base_models_param
        ]
        return StackingEnsemble(base_models, random_state=random_state, **params)

    if name == "bagging":
        if BaggingEnsemble is None:
            raise ImportError("Ensemble methods not available")
        base_model_name = params.pop("base_model", "rsf")
        base_model = make_model(base_model_name, random_state=random_state)
        return BaggingEnsemble(base_model, random_state=random_state, **params)

    if name == "dynamic":
        if DynamicEnsemble is None:
            raise ImportError("Ensemble methods not available")
        base_models_param = params.pop("base_models", ["coxph", "rsf", "xgb_cox"])
        base_models = [
            make_model(model_name, random_state=random_state) for model_name in base_models_param
        ]
        return DynamicEnsemble(base_models, random_state=random_state, **params)

    # Retrieve model from registry for non-ensemble models
    model_class = ModelRegistry.get_model(name)

    # Prepare params for model instantiation
    model_params = {**params}
    if random_state is not None:
        if name.startswith("xgb"):
            model_params.setdefault("seed", random_state)
        else:
            model_params.setdefault("random_state", random_state)

    # Instantiate the model
    if name.startswith("xgb"):
        return model_class(use_gpu=use_gpu, gpu_id=gpu_id, **model_params)
    else:
        return model_class(**model_params)
