"""
Model evaluation module with comprehensive survival metrics calculation.

This module provides:
- Concordance index (C-index) calculation
- Brier score and Integrated Brier Score (IBS)
- Time-dependent AUC
- Calibration metrics
- Confidence intervals via bootstrapping
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sksurv.metrics import (
    concordance_index_censored,
    integrated_brier_score,
    brier_score,
    cumulative_dynamic_auc,
)

from clinical_survival.logging_config import get_logger
from clinical_survival.tracking import MLflowTracker

# Module logger
logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    
    model_name: str
    concordance_index: float
    concordance_index_std: Optional[float] = None
    integrated_brier_score: Optional[float] = None
    brier_scores: Dict[int, float] = field(default_factory=dict)
    time_dependent_auc: Dict[int, float] = field(default_factory=dict)
    mean_auc: Optional[float] = None
    n_samples: int = 0
    n_events: int = 0
    event_rate: float = 0.0
    cv_scores: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "concordance_index": self.concordance_index,
            "concordance_index_std": self.concordance_index_std,
            "integrated_brier_score": self.integrated_brier_score,
            "brier_scores": self.brier_scores,
            "time_dependent_auc": self.time_dependent_auc,
            "mean_auc": self.mean_auc,
            "n_samples": self.n_samples,
            "n_events": self.n_events,
            "event_rate": self.event_rate,
            "cv_scores": self.cv_scores,
        }


def calculate_concordance_index(
    y_true: np.ndarray,
    risk_scores: np.ndarray,
) -> Tuple[float, int, int, int, int]:
    """
    Calculate the concordance index (C-index).
    
    Args:
        y_true: Structured array with 'event' and 'time' fields
        risk_scores: Predicted risk scores (higher = higher risk)
        
    Returns:
        Tuple of (c_index, concordant, discordant, tied_risk, tied_time)
    """
    events = np.array([y[0] for y in y_true])
    times = np.array([y[1] for y in y_true])
    
    return concordance_index_censored(events, times, risk_scores)


def calculate_brier_score(
    y_train: np.ndarray,
    y_test: np.ndarray,
    survival_probs: np.ndarray,
    times: List[int],
) -> Dict[int, float]:
    """
    Calculate Brier scores at specific time points.
    
    Args:
        y_train: Training survival data for estimating censoring distribution
        y_test: Test survival data
        survival_probs: Predicted survival probabilities (n_samples, n_times)
        times: Time points to evaluate
        
    Returns:
        Dictionary mapping time points to Brier scores
    """
    scores = {}
    
    try:
        for i, t in enumerate(times):
            # Get survival probability at time t
            probs_at_t = survival_probs[:, i] if survival_probs.ndim > 1 else survival_probs
            
            # Calculate Brier score
            _, bs = brier_score(y_train, y_test, probs_at_t, t)
            scores[t] = float(bs[-1])  # Last value is the score at time t
    except Exception as e:
        logger.warning(f"Could not calculate Brier score: {e}")
    
    return scores


def calculate_integrated_brier_score(
    y_train: np.ndarray,
    y_test: np.ndarray,
    survival_probs: np.ndarray,
    times: np.ndarray,
) -> Optional[float]:
    """
    Calculate the Integrated Brier Score (IBS).
    
    Args:
        y_train: Training survival data
        y_test: Test survival data
        survival_probs: Predicted survival probabilities (n_samples, n_times)
        times: Time points corresponding to survival_probs columns
        
    Returns:
        IBS value or None if calculation fails
    """
    try:
        ibs = integrated_brier_score(y_train, y_test, survival_probs, times)
        return float(ibs)
    except Exception as e:
        logger.warning(f"Could not calculate IBS: {e}")
        return None


def calculate_time_dependent_auc(
    y_train: np.ndarray,
    y_test: np.ndarray,
    risk_scores: np.ndarray,
    times: List[int],
) -> Tuple[Dict[int, float], Optional[float]]:
    """
    Calculate time-dependent AUC at specific time points.
    
    Args:
        y_train: Training survival data
        y_test: Test survival data
        risk_scores: Predicted risk scores
        times: Time points to evaluate
        
    Returns:
        Tuple of (AUC dict, mean AUC)
    """
    auc_dict = {}
    
    try:
        times_array = np.array(times, dtype=float)
        
        # Filter times to be within data range
        test_times = np.array([y[1] for y in y_test])
        max_time = test_times.max()
        valid_times = times_array[times_array < max_time]
        
        if len(valid_times) == 0:
            logger.warning("No valid time points for AUC calculation")
            return {}, None
        
        auc_values, mean_auc = cumulative_dynamic_auc(
            y_train, y_test, risk_scores, valid_times
        )
        
        for t, auc in zip(valid_times, auc_values):
            auc_dict[int(t)] = float(auc)
        
        return auc_dict, float(mean_auc)
        
    except Exception as e:
        logger.warning(f"Could not calculate time-dependent AUC: {e}")
        return {}, None


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    predictions: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for a metric using bootstrapping.
    
    Args:
        y_true: True survival outcomes
        predictions: Model predictions
        metric_fn: Function that computes the metric
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        random_state: Random seed
        
    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(random_state)
    n_samples = len(y_true)
    
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_boot = y_true[indices]
        pred_boot = predictions[indices]
        
        try:
            score = metric_fn(y_boot, pred_boot)
            bootstrap_scores.append(score)
        except Exception:
            continue
    
    if not bootstrap_scores:
        return np.nan, np.nan, np.nan
    
    bootstrap_scores = np.array(bootstrap_scores)
    point_estimate = np.mean(bootstrap_scores)
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_scores, alpha * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha) * 100)
    
    return point_estimate, lower, upper


def evaluate_survival_model(
    y_train: np.ndarray,
    y_test: np.ndarray,
    risk_scores: np.ndarray,
    survival_probs: Optional[np.ndarray] = None,
    times: Optional[List[int]] = None,
    model_name: str = "model",
    n_bootstrap: int = 200,
) -> EvaluationResult:
    """
    Comprehensive evaluation of a survival model.
    
    Args:
        y_train: Training survival data (for censoring distribution)
        y_test: Test survival data
        risk_scores: Predicted risk scores
        survival_probs: Optional survival probability matrix
        times: Time points for evaluation
        model_name: Name of the model
        n_bootstrap: Number of bootstrap samples for CI
        
    Returns:
        EvaluationResult with all metrics
    """
    logger.info(f"Evaluating model: {model_name}")
    
    # Basic stats
    n_samples = len(y_test)
    events = np.array([y[0] for y in y_test])
    n_events = int(events.sum())
    event_rate = n_events / n_samples if n_samples > 0 else 0.0
    
    # Calculate C-index
    c_index, concordant, discordant, tied_risk, tied_time = calculate_concordance_index(
        y_test, risk_scores
    )
    
    # Bootstrap CI for C-index
    def c_index_metric(y, pred):
        events = np.array([yi[0] for yi in y])
        times = np.array([yi[1] for yi in y])
        return concordance_index_censored(events, times, pred)[0]
    
    _, c_lower, c_upper = bootstrap_confidence_interval(
        y_test, risk_scores, c_index_metric, n_bootstrap=n_bootstrap
    )
    c_std = (c_upper - c_lower) / (2 * 1.96)  # Approximate std from 95% CI
    
    result = EvaluationResult(
        model_name=model_name,
        concordance_index=c_index,
        concordance_index_std=c_std,
        n_samples=n_samples,
        n_events=n_events,
        event_rate=event_rate,
    )
    
    # Time-dependent metrics if times provided
    if times:
        # Time-dependent AUC
        auc_dict, mean_auc = calculate_time_dependent_auc(
            y_train, y_test, risk_scores, times
        )
        result.time_dependent_auc = auc_dict
        result.mean_auc = mean_auc
        
        # Brier scores if survival probabilities provided
        if survival_probs is not None:
            times_array = np.array(times, dtype=float)
            
            result.brier_scores = calculate_brier_score(
                y_train, y_test, survival_probs, times
            )
            
            result.integrated_brier_score = calculate_integrated_brier_score(
                y_train, y_test, survival_probs, times_array
            )
    
    logger.info(
        f"Evaluation complete for {model_name}",
        extra={
            "model": model_name,
            "c_index": round(c_index, 4),
            "ibs": round(result.integrated_brier_score, 4) if result.integrated_brier_score else None,
            "mean_auc": round(mean_auc, 4) if result.mean_auc else None,
        },
    )
    
    return result


def evaluate_predictions(
    oof_preds: np.ndarray,
    y_true: np.ndarray,
    tracker: Optional[MLflowTracker] = None,
    model_name: str = "model",
    times: Optional[List[int]] = None,
) -> EvaluationResult:
    """
    Evaluate out-of-fold predictions and log metrics.
    
    This is the main entry point for the evaluator pipeline step.
    
    Args:
        oof_preds: Out-of-fold risk predictions
        y_true: True survival outcomes
        tracker: Optional MLflow tracker for logging
        model_name: Name of the model
        times: Time points for evaluation
        
    Returns:
        EvaluationResult with all computed metrics
    """
    logger.info(f"Evaluating OOF predictions for {model_name}")
    
    # Calculate C-index from OOF predictions
    events = np.array([y[0] for y in y_true])
    times_arr = np.array([y[1] for y in y_true])
    
    c_index, _, _, _, _ = concordance_index_censored(events, times_arr, oof_preds)
    
    # Bootstrap for confidence interval
    def c_index_fn(y, pred):
        e = np.array([yi[0] for yi in y])
        t = np.array([yi[1] for yi in y])
        return concordance_index_censored(e, t, pred)[0]
    
    _, c_lower, c_upper = bootstrap_confidence_interval(
        y_true, oof_preds, c_index_fn, n_bootstrap=100
    )
    
    result = EvaluationResult(
        model_name=model_name,
        concordance_index=c_index,
        concordance_index_std=(c_upper - c_lower) / (2 * 1.96),
        n_samples=len(y_true),
        n_events=int(events.sum()),
        event_rate=float(events.mean()),
    )
    
    # Log metrics
    metrics = {
        "concordance_index": c_index,
        "concordance_index_lower": c_lower,
        "concordance_index_upper": c_upper,
        "n_samples": len(y_true),
        "n_events": int(events.sum()),
        "event_rate": float(events.mean()),
    }
    
    if tracker is not None:
        tracker.log_metrics(metrics)
    
    logger.info(
        f"OOF evaluation complete",
        extra={
            "model": model_name,
            "c_index": round(c_index, 4),
            "c_index_95ci": f"[{c_lower:.4f}, {c_upper:.4f}]",
        },
    )
    
    return result


def create_evaluation_summary(
    results: List[EvaluationResult],
) -> pd.DataFrame:
    """
    Create a summary DataFrame from multiple evaluation results.
    
    Args:
        results: List of EvaluationResult objects
        
    Returns:
        DataFrame with summary statistics for all models
    """
    rows = []
    for r in results:
        row = {
            "model": r.model_name,
            "c_index": r.concordance_index,
            "c_index_std": r.concordance_index_std,
            "ibs": r.integrated_brier_score,
            "mean_auc": r.mean_auc,
            "n_samples": r.n_samples,
            "n_events": r.n_events,
            "event_rate": r.event_rate,
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values("c_index", ascending=False)
    
    return df
