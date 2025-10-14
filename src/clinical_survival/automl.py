"""Automated model selection and hyperparameter optimization for survival models."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sksurv.metrics import concordance_index_censored

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from clinical_survival.gpu_utils import create_gpu_accelerator
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False

from clinical_survival.models import make_model


class AutoSurvivalML:
    """Automated machine learning for survival analysis with Bayesian optimization."""

    def __init__(
        self,
        time_limit: int = 3600,  # 1 hour default
        n_trials: int | None = None,
        metric: str = "concordance",
        random_state: int | None = None,
        cv_folds: int = 3,
        early_stopping_patience: int = 50,
        use_gpu: bool = True,
        gpu_id: int = 0,
    ):
        """Initialize AutoML optimizer.

        Args:
            time_limit: Maximum time in seconds for optimization
            n_trials: Maximum number of trials (None for time-based)
            metric: Primary metric to optimize ('concordance', 'ibs', 'brier')
            random_state: Random seed for reproducibility
            cv_folds: Number of cross-validation folds
            early_stopping_patience: Early stopping patience for optimization
            use_gpu: Whether to use GPU acceleration when available
            gpu_id: GPU device ID to use
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for AutoSurvivalML. "
                "Install with: pip install optuna"
            )

        self.time_limit = time_limit
        self.n_trials = n_trials
        self.metric = metric
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.early_stopping_patience = early_stopping_patience
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id

        # Define search spaces for different models
        self.search_spaces = {
            "coxph": {
                "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                "tol": [1e-7, 1e-5, 1e-3],
            },
            "rsf": {
                "n_estimators": [50, 100, 200, 500],
                "max_depth": [3, 5, 7, None],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 5, 10, 20],
            },
            "xgb_cox": {
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 4, 5, 6, 7],
                "n_estimators": [50, 100, 200, 500],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "reg_alpha": [0, 0.1, 1.0],
                "reg_lambda": [0.1, 1.0, 10.0],
            },
            "xgb_aft": {
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 4, 5, 6, 7],
                "n_estimators": [50, 100, 200, 500],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "reg_alpha": [0, 0.1, 1.0],
                "reg_lambda": [0.1, 1.0, 10.0],
                "aft_loss_distribution": ["normal", "logistic"],
                "aft_loss_distribution_scale": [0.5, 1.0, 2.0],
            },
        }

        # Model architecture search space for ensembles
        self.architecture_space = {
            "stacking": {
                "base_models": [
                    ["coxph", "rsf"],
                    ["coxph", "rsf", "xgb_cox"],
                    ["rsf", "xgb_cox", "xgb_aft"],
                    ["coxph", "rsf", "xgb_cox", "xgb_aft"],
                ],
                "cv_folds": [3, 5],
            },
            "bagging": {
                "base_model": ["rsf", "xgb_cox", "xgb_aft"],
                "n_estimators": [5, 10, 20],
                "max_samples": [0.8, 0.9, 1.0],
            },
        }

    def _objective(
        self,
        trial: optuna.Trial,
        model_type: str,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray | None = None,
    ) -> float:
        """Objective function for Optuna optimization."""
        # Sample model architecture and hyperparameters
        params = self._sample_params(trial, model_type)

        # Create model with GPU support
        model = make_model(
            model_type,
            random_state=self.random_state,
            use_gpu=self.use_gpu,
            gpu_id=self.gpu_id,
            **params
        )

        # Cross-validation scoring
        try:
            if self.metric == "concordance":
                # Custom scorer for concordance index
                def concordance_scorer(estimator, X, y):
                    y_pred = estimator.predict_risk(X)
                    c_index = concordance_index_censored(
                        y[:, 1] == 1, y[:, 0], -y_pred
                    )[0]
                    return c_index

                scores = cross_val_score(
                    model, X, y, cv=self.cv_folds, scoring=concordance_scorer
                )
            else:
                # For other metrics, use negative MSE as a proxy
                scores = cross_val_score(
                    model, X, y, cv=self.cv_folds, scoring="neg_mean_squared_error"
                )

            return np.mean(scores)

        except Exception as e:
            # Return worst score on error
            return -np.inf if self.metric == "concordance" else np.inf

    def _sample_params(self, trial: optuna.Trial, model_type: str) -> dict[str, Any]:
        """Sample hyperparameters for a given model type."""
        params = {}

        if model_type in self.search_spaces:
            for param_name, param_values in self.search_spaces[model_type].items():
                if isinstance(param_values, list):
                    if all(isinstance(x, (int, float)) for x in param_values):
                        # Numeric parameter - use log scale for positive values
                        if all(x > 0 for x in param_values):
                            params[param_name] = trial.suggest_float(
                                param_name,
                                min(param_values),
                                max(param_values),
                                log=True
                            )
                        else:
                            params[param_name] = trial.suggest_categorical(
                                param_name, param_values
                            )
                    else:
                        # Categorical parameter
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_values
                        )
                else:
                    params[param_name] = param_values

        # Add architecture-specific parameters
        if model_type in self.architecture_space:
            for param_name, param_values in self.architecture_space[model_type].items():
                params[param_name] = trial.suggest_categorical(param_name, param_values)

        return params

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model_types: list[str] | None = None,
        groups: np.ndarray | None = None,
    ) -> AutoSurvivalML:
        """Fit AutoML optimizer to find best model and hyperparameters.

        Args:
            X: Feature matrix
            y: Survival target (event, time) pairs
            model_types: List of model types to consider. If None, tries all.
            groups: Group labels for group-based CV

        Returns:
            Self with fitted optimizer
        """
        if model_types is None:
            model_types = ["coxph", "rsf", "xgb_cox", "xgb_aft", "stacking", "bagging"]

        # Convert y to structured array format expected by models
        if isinstance(y, np.ndarray) and y.dtype.names is not None:
            # Already structured array
            y_structured = y
        else:
            # Convert from various formats to structured array
            if isinstance(y, (list, tuple)) and len(y) > 0:
                if isinstance(y[0], (list, tuple)) and len(y[0]) == 2:
                    # List of [event, time] pairs
                    y_structured = np.array(
                        [(bool(event), float(time)) for event, time in y],
                        dtype=[("event", bool), ("time", float)]
                    )
                else:
                    # Assume it's already in the right format
                    y_structured = np.array(y, dtype=[("event", bool), ("time", float)])
            elif isinstance(y, pd.DataFrame):
                # DataFrame with event and time columns
                y_structured = np.array(
                    list(zip(y.iloc[:, 0].astype(bool), y.iloc[:, 1].astype(float))),
                    dtype=[("event", bool), ("time", float)]
                )
            else:
                # Assume it's a 2D array-like with shape (n_samples, 2)
                y_structured = np.array(
                    [(bool(event), float(time)) for event, time in y],
                    dtype=[("event", bool), ("time", float)]
                )

        # Create study
        study = optuna.create_study(
            direction="maximize" if self.metric == "concordance" else "minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )

        # Run optimization
        start_time = time.time()

        def objective_wrapper(trial: optuna.Trial) -> float:
            # Sample model type
            model_type = trial.suggest_categorical("model_type", model_types)

            # Check time limit
            if time.time() - start_time > self.time_limit:
                raise optuna.exceptions.TrialPruned()

            return self._objective(trial, model_type, X, y_structured, groups)

        # Run optimization with early stopping
        best_trial = None
        patience_counter = 0
        best_score = -np.inf if self.metric == "concordance" else np.inf

        for trial in study.ask():
            try:
                score = objective_wrapper(trial)
                study.tell(trial, score)

                # Check for improvement
                current_best = study.best_value
                if self.metric == "concordance":
                    improved = current_best > best_score
                else:
                    improved = current_best < best_score

                if improved:
                    best_score = current_best
                    best_trial = study.best_trial
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at trial {trial.number}")
                    break

            except optuna.exceptions.TrialPruned:
                study.tell(trial, None)

        # Store results
        if study.best_trial is not None:
            self.best_params_ = study.best_params
            self.best_score_ = study.best_value
            self.best_model_type_ = study.best_params["model_type"]
            self.study_ = study
            self.fitted_ = True
        else:
            raise RuntimeError("No successful trials completed")

        return self

    def get_best_model(self, X: pd.DataFrame, y: np.ndarray) -> Any:
        """Get the best model fitted on provided data."""
        if not hasattr(self, "fitted_") or not self.fitted_:
            raise RuntimeError("AutoML must be fitted before getting best model")

        # Create best model
        model_params = self.best_params_.copy()
        model_type = model_params.pop("model_type")

        best_model = make_model(
            model_type,
            random_state=self.random_state,
            use_gpu=self.use_gpu,
            gpu_id=self.gpu_id,
            **model_params
        )

        # Fit the model on provided data
        return best_model.fit(X, y)

    def get_best_params(self) -> dict[str, Any]:
        """Get parameters of the best model."""
        if not hasattr(self, "fitted_") or not self.fitted_:
            raise RuntimeError("AutoML must be fitted before getting best params")

        return self.best_params_

    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame."""
        if not hasattr(self, "study_"):
            raise RuntimeError("No optimization study available")

        return self.study_.trials_dataframe()


def create_automl_study(
    X: pd.DataFrame,
    y: np.ndarray,
    time_limit: int = 1800,  # 30 minutes default
    model_types: list[str] | None = None,
    use_gpu: bool = True,
    gpu_id: int = 0,
    **kwargs: Any,
) -> AutoSurvivalML:
    """Convenience function to create and fit AutoML study.

    Args:
        X: Feature matrix
        y: Survival target array
        time_limit: Time limit in seconds
        model_types: Model types to consider
        use_gpu: Whether to use GPU acceleration
        gpu_id: GPU device ID to use
        **kwargs: Additional arguments for AutoSurvivalML

    Returns:
        Fitted AutoSurvivalML instance
    """
    automl = AutoSurvivalML(
        time_limit=time_limit,
        use_gpu=use_gpu,
        gpu_id=gpu_id,
        **kwargs
    )

    if model_types is None:
        model_types = ["coxph", "rsf", "xgb_cox", "xgb_aft"]

    return automl.fit(X, y, model_types)
