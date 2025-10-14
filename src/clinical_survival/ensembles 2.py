"""Ensemble methods for survival analysis models."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from clinical_survival.models import BaseSurvivalModel


class SurvivalEnsemble(BaseSurvivalModel, ABC):
    """Base class for survival model ensembles."""

    def __init__(self, base_models: list[BaseSurvivalModel], **kwargs):
        super().__init__(None)
        self.base_models = base_models
        self.fitted_models_: list[BaseSurvivalModel] = []
        self.ensemble_weights_: np.ndarray | None = None
        self._meta_learner: BaseEstimator | None = None

    @abstractmethod
    def fit_base_models(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Fit the base survival models."""
        pass

    @abstractmethod
    def predict_base_models(
        self, X: pd.DataFrame, times: Iterable[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get predictions from base models."""
        pass

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> SurvivalEnsemble:
        """Fit the ensemble model."""
        frame = self._ensure_dataframe(X)
        self.feature_names_ = list(frame.columns)

        self.fit_base_models(frame, y)

        # Fit meta-learner if this is a stacking ensemble
        if hasattr(self, "_fit_meta_learner"):
            self._fit_meta_learner(frame, y)

        return self

    def predict_risk(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict risk scores using ensemble."""
        frame = self._ensure_dataframe(X, self.feature_names_)
        base_risks, _ = self.predict_base_models(frame, [0])  # Dummy time for risk
        return self._combine_predictions(base_risks)

    def predict_survival_function(
        self, X: pd.DataFrame | np.ndarray, times: Iterable[float]
    ) -> np.ndarray:
        """Predict survival function using ensemble."""
        frame = self._ensure_dataframe(X, self.feature_names_)
        times_array = np.asarray(list(times))

        base_risks, base_survival = self.predict_base_models(frame, times)

        # For survival functions, we need to combine them appropriately
        if hasattr(self, "_combine_survival_predictions"):
            return self._combine_survival_predictions(base_survival, times_array)
        else:
            # Default: use risk-based combination and convert to survival
            combined_risks = self._combine_predictions(base_risks)
            return self._risk_to_survival(combined_risks, times_array)

    def _combine_predictions(self, base_predictions: np.ndarray) -> np.ndarray:
        """Combine base model predictions."""
        if self.ensemble_weights_ is not None:
            return np.average(base_predictions, axis=1, weights=self.ensemble_weights_)
        else:
            return np.mean(base_predictions, axis=1)

    def _ensure_dataframe(
        self, X: pd.DataFrame | np.ndarray, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Ensure input is a DataFrame."""
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if columns is None:
            columns = [f"feature_{idx}" for idx in range(X.shape[1])]
        return pd.DataFrame(X, columns=columns)


class StackingEnsemble(SurvivalEnsemble):
    """Stacking ensemble for survival models using meta-learning."""

    def __init__(
        self,
        base_models: list[BaseSurvivalModel],
        meta_learner: BaseEstimator | None = None,
        cv_folds: int = 5,
        **kwargs,
    ):
        super().__init__(base_models, **kwargs)
        self.cv_folds = cv_folds
        self.meta_learner = meta_learner or LinearRegression()

    def fit_base_models(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Fit base models using cross-validation."""
        self.fitted_models_ = []

        # Convert survival format if needed
        if isinstance(y, np.ndarray) and len(y.shape) == 1:
            # Assume y is already in structured format
            y_struct = y
        else:
            # Convert from time/event format
            times = np.asarray(
                [record[1] if len(record) > 1 else record[0] for record in y], dtype=float
            )
            events = np.asarray([record[0] if len(record) > 1 else 1 for record in y], dtype=float)
            from sksurv.util import Surv

            y_struct = Surv.from_arrays(events.astype(bool), times)

        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        for model in self.base_models:
            model_fits = []
            for train_idx, val_idx in kf.split(X):
                model_copy = self._clone_model(model)
                model_copy.fit(X.iloc[train_idx], y_struct[train_idx])
                model_fits.append((model_copy, val_idx))

            self.fitted_models_.append(model_fits)

    def _fit_meta_learner(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Fit meta-learner on out-of-fold predictions."""
        # Get out-of-fold predictions from base models
        oof_predictions = []
        oof_targets = []

        # Convert survival format
        if isinstance(y, np.ndarray) and len(y.shape) == 1:
            # Extract times and events (times not used in current implementation)
            events = np.asarray([record[0] if len(record) > 1 else 1 for record in y], dtype=float)
        else:
            # Extract times (not used in current implementation)
            events = y["event"] if hasattr(y, "dtype") else [record[0] for record in y]

        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        for fold_idx, (_train_idx, val_idx) in enumerate(kf.split(X)):
            fold_predictions = []

            for model_fits in self.fitted_models_:
                model_copy, _ = model_fits[fold_idx]
                risk_pred = model_copy.predict_risk(X.iloc[val_idx])
                fold_predictions.append(risk_pred)

            # Average predictions from all base models for this fold
            ensemble_pred = np.mean(fold_predictions, axis=0)

            oof_predictions.append(ensemble_pred)
            oof_targets.append(events[val_idx])  # Use events as target for meta-learner

        # Flatten predictions
        X_meta = np.column_stack(oof_predictions)
        y_meta = np.concatenate(oof_targets)

        # Fit meta-learner
        self.meta_learner.fit(X_meta, y_meta)

    def predict_base_models(
        self, X: pd.DataFrame, times: Iterable[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get predictions from base models."""
        base_risks = []
        base_survival = []

        for model_fits in self.fitted_models_:
            # Use the first fitted model (they should be similar due to CV)
            model_copy, _ = model_fits[0]

            try:
                risk = model_copy.predict_risk(X)
                survival = model_copy.predict_survival_function(X, times)

                base_risks.append(risk.reshape(-1, 1))
                base_survival.append(survival)
            except Exception as e:
                warnings.warn(f"Model prediction failed: {e}", stacklevel=2)
                # Add dummy predictions
                base_risks.append(np.zeros((len(X), 1)))
                base_survival.append(np.ones((len(X), len(times))))

        return np.hstack(base_risks), np.stack(base_survival, axis=-1)

    def _combine_survival_predictions(
        self, base_survival: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        """Combine survival predictions using meta-learner."""
        # For stacking, we'll use the meta-learner to predict risk and convert to survival
        # Use the same X that was passed to the method (stored as instance variable for this context)
        # For now, use a placeholder - this method needs refactoring for proper context
        base_risks, _ = self.predict_base_models(pd.DataFrame(), [0])

        # Use meta-learner to combine risks
        if hasattr(self.meta_learner, "predict"):
            combined_risks = self.meta_learner.predict(base_risks)
        else:
            combined_risks = np.mean(base_risks, axis=1)

        return self._risk_to_survival(combined_risks, times)

    def _risk_to_survival(self, risks: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Convert risk scores to survival probabilities (simplified approach)."""
        # This is a simplified conversion - in practice you'd want more sophisticated methods
        # For now, assume exponential survival function
        survival = np.exp(-risks[:, np.newaxis] * times[np.newaxis, :])
        return np.clip(survival, 0.0, 1.0)

    def _clone_model(self, model: BaseSurvivalModel) -> BaseSurvivalModel:
        """Clone a survival model (simplified version)."""
        # In practice, you'd want deep cloning with proper parameter copying
        return type(model)(**model.__dict__.get("params", {}))


class BaggingEnsemble(SurvivalEnsemble):
    """Bootstrap aggregation ensemble for survival models."""

    def __init__(
        self,
        base_model: BaseSurvivalModel,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        random_state: int | None = None,
        **kwargs,
    ):
        base_models = [base_model] * n_estimators
        super().__init__(base_models, **kwargs)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state

    def fit_base_models(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Fit base models using bootstrap sampling."""
        self.fitted_models_ = []
        rng = np.random.RandomState(self.random_state)

        n_samples = len(X)
        sample_size = int(self.max_samples * n_samples)
        n_features = len(X.columns)
        feature_size = int(self.max_features * n_features)

        for _i in range(self.n_estimators):
            # Bootstrap sampling
            sample_indices = rng.choice(n_samples, size=sample_size, replace=True)
            feature_indices = rng.choice(n_features, size=feature_size, replace=False)

            X_boot = X.iloc[sample_indices]
            y_boot = y[sample_indices] if isinstance(y, np.ndarray) else y.iloc[sample_indices]

            # Sample features if specified
            if feature_size < n_features:
                X_boot = X_boot.iloc[:, feature_indices]

            # Clone and fit model
            model_copy = self._clone_model(self.base_models[0])
            model_copy.fit(X_boot, y_boot)
            self.fitted_models_.append(model_copy)

    def predict_base_models(
        self, X: pd.DataFrame, times: Iterable[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get predictions from all bagged models."""
        base_risks = []
        base_survival = []

        for model in self.fitted_models_:
            try:
                # Need to handle feature selection for prediction
                if hasattr(self, "_feature_indices"):
                    X_pred = X.iloc[:, self._feature_indices]
                else:
                    X_pred = X

                risk = model.predict_risk(X_pred)
                survival = model.predict_survival_function(X_pred, times)

                base_risks.append(risk.reshape(-1, 1))
                base_survival.append(survival)
            except Exception as e:
                warnings.warn(f"Model prediction failed: {e}", stacklevel=2)
                # Add dummy predictions
                base_risks.append(np.zeros((len(X), 1)))
                base_survival.append(np.ones((len(X), len(times))))

        return np.hstack(base_risks), np.stack(base_survival, axis=-1)

    def _combine_survival_predictions(
        self, base_survival: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        """Combine survival predictions using averaging."""
        return np.mean(base_survival, axis=-1)

    def _clone_model(self, model: BaseSurvivalModel) -> BaseSurvivalModel:
        """Clone a survival model."""
        # Simplified cloning - in practice you'd want proper deep copying
        return type(model)(**getattr(model, "params", {}))


class DynamicEnsemble(SurvivalEnsemble):
    """Dynamic ensemble that selects models based on data characteristics."""

    def __init__(
        self, base_models: list[BaseSurvivalModel], selection_method: str = "performance", **kwargs
    ):
        super().__init__(base_models, **kwargs)
        self.selection_method = selection_method
        self.model_performance_: dict[str, float] = {}

    def fit_base_models(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Fit base models and evaluate their performance."""
        self.fitted_models_ = []

        for model in self.base_models:
            model_copy = self._clone_model(model)
            model_copy.fit(X, y)
            self.fitted_models_.append(model_copy)

            # Store model for performance evaluation
            self.model_performance_[model.name] = self._evaluate_model(model_copy, X, y)

    def predict_base_models(
        self, X: pd.DataFrame, times: Iterable[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get predictions from dynamically selected models."""
        # Select best models based on characteristics
        selected_models = self._select_models(X)

        base_risks = []
        base_survival = []

        for model in selected_models:
            try:
                risk = model.predict_risk(X)
                survival = model.predict_survival_function(X, times)

                base_risks.append(risk.reshape(-1, 1))
                base_survival.append(survival)
            except Exception as e:
                warnings.warn(f"Model prediction failed: {e}", stacklevel=2)
                continue

        if not base_risks:
            # Fallback to all models if selection fails
            return super().predict_base_models(X, times)

        return np.hstack(base_risks), np.stack(base_survival, axis=-1)

    def _select_models(self, X: pd.DataFrame) -> list[BaseSurvivalModel]:
        """Select models based on data characteristics."""
        if self.selection_method == "performance":
            # Select top performing models
            sorted_models = sorted(
                self.model_performance_.items(), key=lambda x: x[1], reverse=True
            )
            n_select = max(1, len(sorted_models) // 2)  # Select top 50%
            selected_names = [name for name, _ in sorted_models[:n_select]]
            return [model for model in self.fitted_models_ if model.name in selected_names]

        elif self.selection_method == "diversity":
            # Select diverse models (simplified approach)
            return self.fitted_models_[: len(self.fitted_models_) // 2]

        else:
            # Default to all models
            return self.fitted_models_

    def _evaluate_model(self, model: BaseSurvivalModel, X: pd.DataFrame, y: np.ndarray) -> float:
        """Evaluate model performance (placeholder - would need proper CV)."""
        try:
            # Simple evaluation - in practice you'd want proper cross-validation
            risk_pred = model.predict_risk(X)

            # Simple correlation-based score (not ideal for survival)
            if isinstance(y, np.ndarray) and len(y.shape) > 1:
                events = y[:, 0].astype(float)
                # Extract times (not used in current implementation)
                # Simple score based on event prediction
                score = np.corrcoef(risk_pred, events)[0, 1]
                return abs(score) if not np.isnan(score) else 0.0
            else:
                return 0.5  # Neutral score
        except Exception:
            return 0.0

    def _clone_model(self, model: BaseSurvivalModel) -> BaseSurvivalModel:
        """Clone a survival model."""
        return type(model)(**getattr(model, "params", {}))
