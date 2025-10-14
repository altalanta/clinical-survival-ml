"""Model monitoring and drift detection for clinical survival models."""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from clinical_survival.utils import ensure_dir, load_json, save_json


@dataclass
class MonitoringMetrics:
    """Container for model monitoring metrics."""

    timestamp: datetime
    model_name: str
    n_samples: int
    concordance: float
    brier_score: float
    feature_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    drift_scores: dict[str, float] = field(default_factory=dict)
    performance_alerts: list[str] = field(default_factory=list)


@dataclass
class DriftAlert:
    """Container for drift detection alerts."""

    timestamp: datetime
    model_name: str
    alert_type: str  # 'concept_drift', 'data_drift', 'performance_degradation'
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    metric_value: float
    threshold: float
    recommendation: str


class ModelMonitor:
    """Monitor model performance and detect drift over time."""

    def __init__(
        self,
        models_dir: str | Path,
        monitoring_dir: str | Path | None = None,
        alert_thresholds: dict[str, float] | None = None,
    ):
        self.models_dir = Path(models_dir)
        self.monitoring_dir = Path(monitoring_dir) if monitoring_dir else self.models_dir / "monitoring"
        self.alert_thresholds = alert_thresholds or self._default_thresholds()
        self.metrics_history: dict[str, list[MonitoringMetrics]] = defaultdict(list)
        self.alerts: list[DriftAlert] = []
        self.baseline_stats: dict[str, dict[str, dict[str, float]]] = {}

    def _default_thresholds(self) -> dict[str, float]:
        """Default alert thresholds for monitoring."""
        return {
            "concordance_drop": 0.05,  # 5% drop in concordance
            "brier_increase": 0.02,    # 2% increase in Brier score
            "feature_drift": 0.1,      # 10% change in feature distribution
            "concept_drift": 0.15,     # 15% change in prediction distribution
        }

    def record_metrics(
        self,
        model_name: str,
        X: pd.DataFrame,
        y_true: pd.DataFrame,
        y_pred: np.ndarray,
        survival_pred: np.ndarray | None = None,
        eval_times: list[float] | None = None,
    ) -> MonitoringMetrics:
        """Record monitoring metrics for a model prediction batch."""

        # Calculate concordance if we have true outcomes
        concordance = self._calculate_concordance(y_true, y_pred)

        # Calculate Brier score if we have survival predictions
        brier_score = self._calculate_brier_score(y_true, survival_pred, eval_times) if survival_pred is not None else 0.0

        # Calculate feature statistics
        feature_stats = self._calculate_feature_stats(X)

        # Calculate drift scores
        drift_scores = self._calculate_drift_scores(X, feature_stats)

        # Create metrics object
        metrics = MonitoringMetrics(
            timestamp=datetime.now(),
            model_name=model_name,
            n_samples=len(X),
            concordance=concordance,
            brier_score=brier_score,
            feature_stats=feature_stats,
            drift_scores=drift_scores,
        )

        # Store metrics
        self.metrics_history[model_name].append(metrics)

        # Check for alerts
        self._check_alerts(model_name, metrics)

        # Keep only recent history (last 1000 records)
        if len(self.metrics_history[model_name]) > 1000:
            self.metrics_history[model_name] = self.metrics_history[model_name][-1000:]

        return metrics

    def _calculate_concordance(self, y_true: pd.DataFrame, y_pred: np.ndarray) -> float:
        """Calculate concordance index between true and predicted risks."""
        try:
            from sksurv.metrics import concordance_index_censored

            # Extract time and event from y_true
            time_col = y_true.columns[y_true.columns.str.contains('time', case=False)][0]
            event_col = y_true.columns[y_true.columns.str.contains('event', case=False)][0]

            times = y_true[time_col].values.astype(float)
            events = y_true[event_col].values.astype(bool)

            # Calculate concordance
            c_index = concordance_index_censored(events, times, y_pred)[0]
            return float(c_index)

        except Exception as e:
            warnings.warn(f"Failed to calculate concordance: {e}")
            return 0.5  # Neutral score

    def _calculate_brier_score(
        self,
        y_true: pd.DataFrame,
        survival_pred: np.ndarray,
        eval_times: list[float],
    ) -> float:
        """Calculate integrated Brier score."""
        try:
            from sksurv.metrics import integrated_brier_score

            # Extract time and event from y_true
            time_col = y_true.columns[y_true.columns.str.contains('time', case=False)][0]
            event_col = y_true.columns[y_true.columns.str.contains('event', case=False)][0]

            times = y_true[time_col].values.astype(float)
            events = y_true[event_col].values.astype(bool)

            # Calculate integrated Brier score
            ibs = integrated_brier_score(events, times, survival_pred, eval_times)
            return float(ibs[0])

        except Exception as e:
            warnings.warn(f"Failed to calculate Brier score: {e}")
            return 0.0

    def _calculate_feature_stats(self, X: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Calculate basic statistics for each feature."""
        stats_dict = {}

        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                stats_dict[col] = {
                    'mean': float(X[col].mean()),
                    'std': float(X[col].std()),
                    'min': float(X[col].min()),
                    'max': float(X[col].max()),
                    'missing_rate': float(X[col].isnull().mean()),
                }
            elif X[col].dtype == 'object' or X[col].dtype.name == 'category':
                # For categorical features, track value counts
                value_counts = X[col].value_counts(normalize=True)
                stats_dict[col] = {
                    'unique_count': float(X[col].nunique()),
                    'missing_rate': float(X[col].isnull().mean()),
                    'top_category_rate': float(value_counts.iloc[0]) if len(value_counts) > 0 else 0.0,
                }

        return stats_dict

    def _calculate_drift_scores(self, X: pd.DataFrame, current_stats: dict[str, dict[str, float]]) -> dict[str, float]:
        """Calculate drift scores by comparing to baseline statistics."""
        drift_scores = {}

        if not self.baseline_stats:
            # Set baseline if not exists
            self.baseline_stats = {col: current_stats[col] for col in current_stats}
            return {col: 0.0 for col in current_stats}

        for col in current_stats:
            if col in self.baseline_stats:
                baseline = self.baseline_stats[col]
                current = current_stats[col]

                # Calculate drift score based on feature type
                if 'mean' in baseline:  # Numeric feature
                    # Use standardized difference in means
                    if baseline['std'] > 0:
                        drift = abs(current['mean'] - baseline['mean']) / baseline['std']
                    else:
                        drift = 0.0

                    # Also consider changes in standard deviation
                    std_drift = abs(current['std'] - baseline['std']) / baseline['std'] if baseline['std'] > 0 else 0.0

                    drift_scores[col] = max(drift, std_drift)

                elif 'unique_count' in baseline:  # Categorical feature
                    # Use Jensen-Shannon divergence for categorical distributions
                    drift_scores[col] = self._calculate_categorical_drift(col, baseline, current)

        return drift_scores

    def _calculate_categorical_drift(
        self,
        col: str,
        baseline: dict[str, float],
        current: dict[str, float]
    ) -> float:
        """Calculate drift for categorical features using Jensen-Shannon divergence."""
        # Simplified approach: compare top category rates and unique counts
        baseline_rate = baseline.get('top_category_rate', 0.0)
        current_rate = current.get('top_category_rate', 0.0)

        # Simple drift measure based on top category change
        drift = abs(current_rate - baseline_rate)

        # Also consider change in unique count
        baseline_unique = baseline.get('unique_count', 0.0)
        current_unique = current.get('unique_count', 0.0)
        unique_drift = abs(current_unique - baseline_unique) / max(baseline_unique, 1.0)

        return max(drift, unique_drift)

    def _check_alerts(self, model_name: str, metrics: MonitoringMetrics) -> None:
        """Check for performance alerts based on thresholds."""

        # Check concordance degradation
        if self._has_recent_history(model_name):
            recent_metrics = self._get_recent_metrics(model_name, days=7)

            if recent_metrics:
                baseline_concordance = np.mean([m.concordance for m in recent_metrics])
                concordance_drop = baseline_concordance - metrics.concordance

                if concordance_drop > self.alert_thresholds["concordance_drop"]:
                    alert = DriftAlert(
                        timestamp=metrics.timestamp,
                        model_name=model_name,
                        alert_type="performance_degradation",
                        severity="medium",
                        message=f"Concordance dropped by {concordance_drop:.3f} from baseline",
                        metric_value=metrics.concordance,
                        threshold=self.alert_thresholds["concordance_drop"],
                        recommendation="Consider model retraining or investigation",
                    )
                    self.alerts.append(alert)

        # Check Brier score increase
        if metrics.brier_score > 0:
            if self._has_recent_history(model_name):
                recent_metrics = self._get_recent_metrics(model_name, days=7)

                if recent_metrics:
                    baseline_brier = np.mean([m.brier_score for m in recent_metrics])
                    brier_increase = metrics.brier_score - baseline_brier

                    if brier_increase > self.alert_thresholds["brier_increase"]:
                        alert = DriftAlert(
                            timestamp=metrics.timestamp,
                            model_name=model_name,
                            alert_type="performance_degradation",
                            severity="medium",
                            message=f"Brier score increased by {brier_increase:.3f} from baseline",
                            metric_value=metrics.brier_score,
                            threshold=self.alert_thresholds["brier_increase"],
                            recommendation="Model calibration may be degrading",
                        )
                        self.alerts.append(alert)

        # Check feature drift
        for feature, drift_score in metrics.drift_scores.items():
            if drift_score > self.alert_thresholds["feature_drift"]:
                alert = DriftAlert(
                    timestamp=metrics.timestamp,
                    model_name=model_name,
                    alert_type="data_drift",
                    severity="low" if drift_score < 0.2 else "medium",
                    message=f"Feature '{feature}' shows significant drift (score: {drift_score:.3f})",
                    metric_value=drift_score,
                    threshold=self.alert_thresholds["feature_drift"],
                    recommendation="Review data collection process or consider retraining",
                )
                self.alerts.append(alert)

    def _has_recent_history(self, model_name: str, days: int = 7) -> bool:
        """Check if we have recent monitoring history."""
        if model_name not in self.metrics_history:
            return False

        cutoff = datetime.now() - timedelta(days=days)
        recent_metrics = [m for m in self.metrics_history[model_name] if m.timestamp > cutoff]

        return len(recent_metrics) >= 3  # Need at least 3 data points for meaningful comparison

    def _get_recent_metrics(self, model_name: str, days: int = 7) -> list[MonitoringMetrics]:
        """Get recent monitoring metrics."""
        if model_name not in self.metrics_history:
            return []

        cutoff = datetime.now() - timedelta(days=days)
        return [m for m in self.metrics_history[model_name] if m.timestamp > cutoff]

    def get_performance_summary(self, model_name: str, days: int = 30) -> dict[str, Any]:
        """Get performance summary for a model over the specified period."""

        if model_name not in self.metrics_history:
            return {"error": "No monitoring data available"}

        metrics = self._get_recent_metrics(model_name, days)

        if not metrics:
            return {"error": "No recent monitoring data"}

        # Calculate summary statistics
        concordances = [m.concordance for m in metrics]
        brier_scores = [m.brier_score for m in metrics if m.brier_score > 0]
        sample_counts = [m.n_samples for m in metrics]

        summary = {
            "model_name": model_name,
            "period_days": days,
            "n_observations": len(metrics),
            "total_samples": sum(sample_counts),
            "concordance": {
                "mean": float(np.mean(concordances)),
                "std": float(np.std(concordances)),
                "min": float(np.min(concordances)),
                "max": float(np.max(concordances)),
                "trend": self._calculate_trend(concordances),
            },
        }

        if brier_scores:
            summary["brier_score"] = {
                "mean": float(np.mean(brier_scores)),
                "std": float(np.std(brier_scores)),
                "min": float(np.min(brier_scores)),
                "max": float(np.max(brier_scores)),
                "trend": self._calculate_trend(brier_scores),
            }

        # Add drift information
        recent_metrics = self._get_recent_metrics(model_name, days=1)  # Most recent
        if recent_metrics:
            latest = recent_metrics[-1]
            summary["latest_drift_scores"] = latest.drift_scores
            summary["latest_alerts"] = len([a for a in self.alerts if a.model_name == model_name])

        return summary

    def _calculate_trend(self, values: list[float]) -> str:
        """Calculate trend direction for a time series."""
        if len(values) < 3:
            return "insufficient_data"

        # Simple linear trend calculation
        x = np.arange(len(values))
        slope, _, r_value, _, _ = stats.linregress(x, values)

        if abs(r_value) < 0.3:  # Weak correlation
            return "stable"
        elif slope > 0:
            return "improving"
        else:
            return "degrading"

    def save_monitoring_data(self) -> None:
        """Save monitoring data to disk."""
        ensure_dir(self.monitoring_dir)

        # Save metrics history
        for model_name, metrics_list in self.metrics_history.items():
            model_dir = ensure_dir(self.monitoring_dir / model_name)

            # Convert metrics to serializable format
            serializable_metrics = []
            for metric in metrics_list:
                serializable_metrics.append({
                    "timestamp": metric.timestamp.isoformat(),
                    "n_samples": metric.n_samples,
                    "concordance": metric.concordance,
                    "brier_score": metric.brier_score,
                    "feature_stats": metric.feature_stats,
                    "drift_scores": metric.drift_scores,
                })

            save_json(serializable_metrics, model_dir / "metrics_history.json")

        # Save alerts
        serializable_alerts = []
        for alert in self.alerts:
            serializable_alerts.append({
                "timestamp": alert.timestamp.isoformat(),
                "model_name": alert.model_name,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "metric_value": alert.metric_value,
                "threshold": alert.threshold,
                "recommendation": alert.recommendation,
            })

        save_json(serializable_alerts, self.monitoring_dir / "alerts.json")

        # Save baseline statistics
        save_json(self.baseline_stats, self.monitoring_dir / "baseline_stats.json")

    def load_monitoring_data(self) -> None:
        """Load monitoring data from disk."""
        if not self.monitoring_dir.exists():
            return

        # Load baseline statistics
        baseline_file = self.monitoring_dir / "baseline_stats.json"
        if baseline_file.exists():
            self.baseline_stats = load_json(baseline_file)

        # Load alerts
        alerts_file = self.monitoring_dir / "alerts.json"
        if alerts_file.exists():
            alert_data = load_json(alerts_file)
            self.alerts = [
                DriftAlert(
                    timestamp=datetime.fromisoformat(alert["timestamp"]),
                    model_name=alert["model_name"],
                    alert_type=alert["alert_type"],
                    severity=alert["severity"],
                    message=alert["message"],
                    metric_value=alert["metric_value"],
                    threshold=alert["threshold"],
                    recommendation=alert["recommendation"],
                )
                for alert in alert_data
            ]

        # Load metrics history for each model
        for model_dir in self.monitoring_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                metrics_file = model_dir / "metrics_history.json"

                if metrics_file.exists():
                    metrics_data = load_json(metrics_file)

                    for metric_data in metrics_data:
                        metric = MonitoringMetrics(
                            timestamp=datetime.fromisoformat(metric_data["timestamp"]),
                            model_name=model_name,
                            n_samples=metric_data["n_samples"],
                            concordance=metric_data["concordance"],
                            brier_score=metric_data["brier_score"],
                            feature_stats=metric_data["feature_stats"],
                            drift_scores=metric_data["drift_scores"],
                        )
                        self.metrics_history[model_name].append(metric)

    def get_recent_alerts(self, model_name: str | None = None, days: int = 7) -> list[DriftAlert]:
        """Get recent alerts for a model or all models."""
        cutoff = datetime.now() - timedelta(days=days)

        if model_name:
            return [alert for alert in self.alerts
                   if alert.model_name == model_name and alert.timestamp > cutoff]
        else:
            return [alert for alert in self.alerts if alert.timestamp > cutoff]

    def reset_baseline(self, model_name: str | None = None) -> None:
        """Reset baseline statistics for a model or all models."""
        if model_name:
            if model_name in self.baseline_stats:
                del self.baseline_stats[model_name]
        else:
            self.baseline_stats.clear()


class PerformanceTracker:
    """Track model performance over time for A/B testing and comparison."""

    def __init__(self, models_dir: str | Path):
        self.models_dir = Path(models_dir)
        self.performance_data: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def track_experiment(
        self,
        experiment_name: str,
        model_names: list[str],
        metrics: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Track an experiment comparing multiple models."""

        experiment_data = {
            "timestamp": datetime.now().isoformat(),
            "experiment_name": experiment_name,
            "models": model_names,
            "metrics": metrics,
            "metadata": metadata or {},
        }

        self.performance_data[experiment_name].append(experiment_data)

        # Save to disk
        self._save_performance_data()

    def _save_performance_data(self) -> None:
        """Save performance data to disk."""
        ensure_dir(self.models_dir / "experiments")

        for experiment_name, data_list in self.performance_data.items():
            save_json(data_list, self.models_dir / "experiments" / f"{experiment_name}.json")

    def get_experiment_results(self, experiment_name: str) -> dict[str, Any]:
        """Get results for a specific experiment."""

        if experiment_name not in self.performance_data:
            return {"error": "Experiment not found"}

        data = self.performance_data[experiment_name]

        # Aggregate results across runs
        model_metrics = defaultdict(list)

        for run in data:
            for model_name, metric_value in run["metrics"].items():
                model_metrics[model_name].append(metric_value)

        # Calculate statistics for each model
        results = {}
        for model_name, values in model_metrics.items():
            results[model_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n_runs": len(values),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        return {
            "experiment_name": experiment_name,
            "n_runs": len(data),
            "model_results": results,
            "best_model": max(results.keys(), key=lambda x: results[x]["mean"]) if results else None,
        }
