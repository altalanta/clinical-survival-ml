"""
Model inference utilities for loading saved models and running predictions.

This module provides:
- Model loading from saved artifacts
- Batch and single-sample inference
- Survival probability curves
- Risk stratification
- Prediction explanations

Usage:
    from clinical_survival.inference import ModelInference
    
    inference = ModelInference.load("results/artifacts/models/xgb_cox.joblib")
    
    # Single prediction
    risk = inference.predict_risk(patient_features)
    
    # Survival curve
    times, probs = inference.predict_survival(patient_features, times=[365, 730, 1095])
    
    # Batch predictions
    risks = inference.predict_risk_batch(features_df)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from clinical_survival.logging_config import get_logger

# Module logger
logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """Container for a single prediction result."""
    
    patient_id: Optional[str] = None
    risk_score: float = 0.0
    risk_percentile: Optional[float] = None
    risk_category: Optional[str] = None
    survival_probabilities: Dict[int, float] = field(default_factory=dict)
    median_survival_time: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "patient_id": self.patient_id,
            "risk_score": self.risk_score,
            "risk_percentile": self.risk_percentile,
            "risk_category": self.risk_category,
            "survival_probabilities": self.survival_probabilities,
            "median_survival_time": self.median_survival_time,
            "confidence_interval": self.confidence_interval,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass
class BatchPredictionResult:
    """Container for batch prediction results."""
    
    predictions: List[PredictionResult] = field(default_factory=list)
    model_name: str = ""
    model_version: Optional[str] = None
    prediction_timestamp: Optional[str] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        rows = []
        for pred in self.predictions:
            row = {
                "patient_id": pred.patient_id,
                "risk_score": pred.risk_score,
                "risk_percentile": pred.risk_percentile,
                "risk_category": pred.risk_category,
                "median_survival_time": pred.median_survival_time,
            }
            # Add survival probabilities as columns
            for time, prob in pred.survival_probabilities.items():
                row[f"surv_prob_{time}d"] = prob
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "prediction_timestamp": self.prediction_timestamp,
            "n_predictions": len(self.predictions),
            "predictions": [p.to_dict() for p in self.predictions],
        }


class ModelInference:
    """
    Model inference class for survival predictions.
    
    Usage:
        # Load a saved model
        inference = ModelInference.load("models/xgb_cox.joblib")
        
        # Single prediction
        result = inference.predict(patient_df.iloc[0])
        
        # Batch prediction
        results = inference.predict_batch(patients_df)
        
        # Survival curve
        times, probs = inference.predict_survival_curve(patient_df.iloc[0])
    """
    
    # Risk category thresholds
    DEFAULT_RISK_THRESHOLDS = {
        "low": 0.25,
        "moderate": 0.50,
        "high": 0.75,
        "very_high": 1.0,
    }
    
    def __init__(
        self,
        model: Any,
        model_name: str = "unknown",
        model_version: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        risk_thresholds: Optional[Dict[str, float]] = None,
        reference_risks: Optional[np.ndarray] = None,
    ):
        """
        Initialize the inference engine.
        
        Args:
            model: Trained model or pipeline
            model_name: Name of the model
            model_version: Version identifier
            feature_names: Expected feature names
            risk_thresholds: Thresholds for risk categories
            reference_risks: Reference risk scores for percentile calculation
        """
        self.model = model
        self.model_name = model_name
        self.model_version = model_version
        self.feature_names = feature_names
        self.risk_thresholds = risk_thresholds or self.DEFAULT_RISK_THRESHOLDS
        self.reference_risks = reference_risks
        
        # Extract feature names from pipeline if available
        if feature_names is None:
            self._extract_feature_names()
    
    def _extract_feature_names(self) -> None:
        """Try to extract feature names from the model/pipeline."""
        try:
            if hasattr(self.model, "feature_names_"):
                self.feature_names = list(self.model.feature_names_)
            elif hasattr(self.model, "named_steps"):
                # Pipeline
                if "pre" in self.model.named_steps:
                    pre = self.model.named_steps["pre"]
                    if hasattr(pre, "get_feature_names_out"):
                        self.feature_names = list(pre.get_feature_names_out())
        except Exception:
            pass
    
    @classmethod
    def load(
        cls,
        model_path: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None,
    ) -> "ModelInference":
        """
        Load a model from disk.
        
        Args:
            model_path: Path to the saved model (.joblib)
            metadata_path: Optional path to model metadata JSON
            
        Returns:
            ModelInference instance
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        
        # Try to load metadata
        model_name = model_path.stem
        model_version = None
        feature_names = None
        reference_risks = None
        
        if metadata_path is None:
            metadata_path = model_path.with_suffix(".json")
        
        if Path(metadata_path).exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                model_name = metadata.get("model_name", model_name)
                model_version = metadata.get("version")
                feature_names = metadata.get("feature_names")
                
                if "reference_risks" in metadata:
                    reference_risks = np.array(metadata["reference_risks"])
        
        logger.info(f"Model loaded: {model_name} (version: {model_version})")
        
        return cls(
            model=model,
            model_name=model_name,
            model_version=model_version,
            feature_names=feature_names,
            reference_risks=reference_risks,
        )
    
    def save_metadata(self, path: Union[str, Path]) -> None:
        """Save model metadata to JSON."""
        metadata = {
            "model_name": self.model_name,
            "version": self.model_version,
            "feature_names": self.feature_names,
            "risk_thresholds": self.risk_thresholds,
        }
        
        if self.reference_risks is not None:
            metadata["reference_risks"] = self.reference_risks.tolist()
        
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _validate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate and reorder input features."""
        if self.feature_names is None:
            return X
        
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        # Reorder columns to match expected order
        return X[self.feature_names]
    
    def _calculate_risk_category(self, risk_score: float) -> str:
        """Determine risk category from score."""
        if self.reference_risks is not None:
            # Use percentile-based categories
            percentile = np.mean(self.reference_risks <= risk_score)
        else:
            # Normalize to 0-1 range (assuming higher score = higher risk)
            percentile = 1 / (1 + np.exp(-risk_score))  # Sigmoid normalization
        
        for category, threshold in sorted(self.risk_thresholds.items(), key=lambda x: x[1]):
            if percentile <= threshold:
                return category
        
        return "very_high"
    
    def _calculate_percentile(self, risk_score: float) -> Optional[float]:
        """Calculate percentile of risk score."""
        if self.reference_risks is None:
            return None
        
        return float(np.mean(self.reference_risks <= risk_score) * 100)
    
    def predict_risk(
        self,
        X: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
    ) -> float:
        """
        Predict risk score for a single sample.
        
        Args:
            X: Single sample features (DataFrame row, Series, or dict)
            
        Returns:
            Risk score (higher = higher risk)
        """
        # Convert to DataFrame
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, pd.Series):
            X = X.to_frame().T
        
        X = self._validate_features(X)
        
        if hasattr(self.model, "predict_risk"):
            risk = self.model.predict_risk(X)
        else:
            risk = self.model.predict(X)
        
        return float(risk[0])
    
    def predict_risk_batch(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict risk scores for multiple samples.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of risk scores
        """
        X = self._validate_features(X)
        
        if hasattr(self.model, "predict_risk"):
            return self.model.predict_risk(X)
        else:
            return self.model.predict(X)
    
    def predict_survival(
        self,
        X: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
        times: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict survival probabilities at specified times.
        
        Args:
            X: Single sample features
            times: Time points to evaluate (default: [365, 730, 1095])
            
        Returns:
            Tuple of (times array, survival probabilities array)
        """
        if times is None:
            times = [365, 730, 1095, 1825]  # 1, 2, 3, 5 years
        
        times = np.array(times, dtype=float)
        
        # Convert to DataFrame
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, pd.Series):
            X = X.to_frame().T
        
        X = self._validate_features(X)
        
        if hasattr(self.model, "predict_survival_function"):
            surv_probs = self.model.predict_survival_function(X, times)
            return times, surv_probs[0]  # First sample
        else:
            logger.warning("Model does not support survival function prediction")
            return times, np.ones_like(times) * 0.5  # Placeholder
    
    def predict(
        self,
        X: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
        patient_id: Optional[str] = None,
        times: Optional[List[int]] = None,
    ) -> PredictionResult:
        """
        Generate comprehensive prediction for a single sample.
        
        Args:
            X: Single sample features
            patient_id: Optional patient identifier
            times: Time points for survival probabilities
            
        Returns:
            PredictionResult with all predictions
        """
        risk_score = self.predict_risk(X)
        
        result = PredictionResult(
            patient_id=patient_id,
            risk_score=risk_score,
            risk_percentile=self._calculate_percentile(risk_score),
            risk_category=self._calculate_risk_category(risk_score),
        )
        
        # Survival probabilities
        if times:
            times_arr, probs = self.predict_survival(X, times)
            result.survival_probabilities = {
                int(t): float(p) for t, p in zip(times_arr, probs)
            }
            
            # Estimate median survival (when probability = 0.5)
            if len(probs) > 1:
                try:
                    # Linear interpolation to find median
                    idx = np.searchsorted(probs[::-1], 0.5)
                    if 0 < idx < len(probs):
                        result.median_survival_time = float(times_arr[::-1][idx])
                except Exception:
                    pass
        
        return result
    
    def predict_batch(
        self,
        X: pd.DataFrame,
        patient_ids: Optional[List[str]] = None,
        times: Optional[List[int]] = None,
    ) -> BatchPredictionResult:
        """
        Generate predictions for multiple samples.
        
        Args:
            X: Feature DataFrame
            patient_ids: Optional list of patient identifiers
            times: Time points for survival probabilities
            
        Returns:
            BatchPredictionResult with all predictions
        """
        from datetime import datetime
        
        n_samples = len(X)
        
        if patient_ids is None:
            patient_ids = [str(i) for i in range(n_samples)]
        
        logger.info(f"Generating predictions for {n_samples} samples")
        
        predictions = []
        for i in range(n_samples):
            row = X.iloc[i]
            pred = self.predict(row, patient_id=patient_ids[i], times=times)
            predictions.append(pred)
        
        return BatchPredictionResult(
            predictions=predictions,
            model_name=self.model_name,
            model_version=self.model_version,
            prediction_timestamp=datetime.utcnow().isoformat(),
        )
    
    def set_reference_distribution(
        self,
        X: pd.DataFrame,
    ) -> None:
        """
        Set reference risk distribution from a training/reference dataset.
        
        This enables percentile-based risk categorization.
        
        Args:
            X: Reference feature DataFrame
        """
        self.reference_risks = self.predict_risk_batch(X)
        logger.info(f"Reference distribution set from {len(X)} samples")


def load_model(model_path: Union[str, Path]) -> ModelInference:
    """
    Convenience function to load a model.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        ModelInference instance
    """
    return ModelInference.load(model_path)







