# ruff: noqa: S101
"""Tests for model monitoring functionality."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from clinical_survival.monitoring import ModelMonitor, PerformanceTracker


def test_model_monitor_initialization():
    """Test ModelMonitor initialization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        monitor = ModelMonitor(temp_dir)
        assert monitor.models_dir == Path(temp_dir)
        assert monitor.alert_thresholds is not None
        assert "concordance_drop" in monitor.alert_thresholds


def test_monitoring_metrics_creation():
    """Test creation of monitoring metrics."""
    with tempfile.TemporaryDirectory() as temp_dir:
        monitor = ModelMonitor(temp_dir)

        # Create test data
        X = pd.DataFrame({
            "age": np.random.normal(50, 10, 100),
            "sex": np.random.choice(["M", "F"], 100),
        })
        y_true = pd.DataFrame({
            "time": np.random.randint(30, 365, 100),
            "event": np.random.randint(0, 2, 100),
        })
        y_pred = np.random.normal(0, 1, 100)
        survival_pred = np.random.uniform(0.1, 0.9, (100, 3))

        # Record metrics
        metrics = monitor.record_metrics(
            model_name="test_model",
            X=X,
            y_true=y_true,
            y_pred=y_pred,
            survival_pred=survival_pred,
            eval_times=[90, 180, 365],
        )

        assert metrics.model_name == "test_model"
        assert metrics.n_samples == 100
        assert metrics.concordance >= 0 and metrics.concordance <= 1
        assert metrics.timestamp is not None
        assert "age" in metrics.feature_stats
        assert "sex" in metrics.feature_stats


def test_feature_statistics_calculation():
    """Test feature statistics calculation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        monitor = ModelMonitor(temp_dir)

        # Create test data with known statistics
        X = pd.DataFrame({
            "numeric_col": [1.0, 2.0, 3.0, 4.0, 5.0],
            "categorical_col": ["A", "B", "A", "C", "B"],
        })

        feature_stats = monitor._calculate_feature_stats(X)

        # Check numeric column stats
        assert "numeric_col" in feature_stats
        assert feature_stats["numeric_col"]["mean"] == 3.0
        assert feature_stats["numeric_col"]["std"] == pytest.approx(1.581, abs=0.001)
        assert feature_stats["numeric_col"]["min"] == 1.0
        assert feature_stats["numeric_col"]["max"] == 5.0
        assert feature_stats["numeric_col"]["missing_rate"] == 0.0

        # Check categorical column stats
        assert "categorical_col" in feature_stats
        assert feature_stats["categorical_col"]["unique_count"] == 3
        assert feature_stats["categorical_col"]["missing_rate"] == 0.0


def test_drift_detection():
    """Test drift detection between baseline and current data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        monitor = ModelMonitor(temp_dir)

        # Create baseline data
        baseline_X = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.choice(["A", "B", "C"], 100),
        })

        # Record baseline metrics to set baseline stats
        baseline_metrics = monitor.record_metrics(
            model_name="test_model",
            X=baseline_X,
            y_true=pd.DataFrame({"time": [100] * 100, "event": [1] * 100}),
            y_pred=np.random.normal(0, 1, 100),
        )

        # Create current data with drift
        current_X = pd.DataFrame({
            "feature1": np.random.normal(2, 1.5, 100),  # Mean shift
            "feature2": np.random.choice(["A", "B", "C", "D"], 100),  # New category
        })

        # Record current metrics
        current_metrics = monitor.record_metrics(
            model_name="test_model",
            X=current_X,
            y_true=pd.DataFrame({"time": [100] * 100, "event": [1] * 100}),
            y_pred=np.random.normal(0, 1, 100),
        )

        # Should detect drift
        assert any(score > 0 for score in current_metrics.drift_scores.values())


def test_performance_summary():
    """Test performance summary generation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        monitor = ModelMonitor(temp_dir)

        # Create test data and record multiple metrics
        X = pd.DataFrame({"feature": np.random.normal(0, 1, 50)})
        y_true = pd.DataFrame({"time": [100] * 50, "event": [1] * 50})

        # Record several metrics over time
        for i in range(5):
            metrics = monitor.record_metrics(
                model_name="test_model",
                X=X,
                y_true=y_true,
                y_pred=np.random.normal(0, 1, 50),
            )

        # Get performance summary
        summary = monitor.get_performance_summary("test_model", days=1)

        assert summary["model_name"] == "test_model"
        assert "concordance" in summary
        assert summary["concordance"]["mean"] >= 0
        assert summary["n_observations"] == 5


def test_alert_generation():
    """Test alert generation for performance degradation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        monitor = ModelMonitor(temp_dir)

        # Create baseline data with good performance
        X = pd.DataFrame({"feature": np.random.normal(0, 1, 100)})
        y_true = pd.DataFrame({"time": [100] * 100, "event": [1] * 100})

        # Record good baseline metrics
        for _ in range(3):
            monitor.record_metrics(
                model_name="test_model",
                X=X,
                y_true=y_true,
                y_pred=np.random.normal(0, 0.5, 100),  # Good predictions
            )

        # Record degraded performance
        degraded_metrics = monitor.record_metrics(
            model_name="test_model",
            X=X,
            y_true=y_true,
            y_pred=np.random.normal(0, 2.0, 100),  # Poor predictions
        )

        # Should generate alert for performance degradation
        recent_alerts = monitor.get_recent_alerts("test_model", days=1)
        assert len(recent_alerts) > 0


def test_monitoring_data_persistence():
    """Test saving and loading monitoring data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create and populate monitor
        monitor = ModelMonitor(temp_dir)

        X = pd.DataFrame({"feature": np.random.normal(0, 1, 50)})
        y_true = pd.DataFrame({"time": [100] * 50, "event": [1] * 50})

        # Record some metrics
        for i in range(3):
            monitor.record_metrics(
                model_name="test_model",
                X=X,
                y_true=y_true,
                y_pred=np.random.normal(0, 1, 50),
            )

        # Save monitoring data
        monitor.save_monitoring_data()

        # Create new monitor and load data
        new_monitor = ModelMonitor(temp_dir)
        new_monitor.load_monitoring_data()

        # Should have loaded the data
        assert "test_model" in new_monitor.metrics_history
        assert len(new_monitor.metrics_history["test_model"]) == 3


def test_performance_tracker():
    """Test performance tracker for A/B testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = PerformanceTracker(temp_dir)

        # Track an experiment
        tracker.track_experiment(
            experiment_name="model_comparison",
            model_names=["coxph", "rsf"],
            metrics={"coxph": 0.75, "rsf": 0.78},
            metadata={"dataset": "test_data"},
        )

        # Get experiment results
        results = tracker.get_experiment_results("model_comparison")

        assert results["experiment_name"] == "model_comparison"
        assert "coxph" in results["model_results"]
        assert "rsf" in results["model_results"]
        assert results["best_model"] == "rsf"  # Higher concordance


def test_monitoring_with_missing_data():
    """Test monitoring handles missing data correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        monitor = ModelMonitor(temp_dir)

        # Create data with missing values
        X = pd.DataFrame({
            "feature1": [1.0, 2.0, np.nan, 4.0, 5.0],
            "feature2": ["A", "B", np.nan, "D", "E"],
        })
        y_true = pd.DataFrame({"time": [100] * 5, "event": [1] * 5})

        # Should handle missing data gracefully
        metrics = monitor.record_metrics(
            model_name="test_model",
            X=X,
            y_true=y_true,
            y_pred=np.random.normal(0, 1, 5),
        )

        # Check that missing data is properly tracked
        assert metrics.feature_stats["feature1"]["missing_rate"] == 0.2
        assert metrics.feature_stats["feature2"]["missing_rate"] == 0.2


def test_monitoring_edge_cases():
    """Test monitoring handles edge cases correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        monitor = ModelMonitor(temp_dir)

        # Test with very small dataset
        X = pd.DataFrame({"feature": [1.0, 2.0]})
        y_true = pd.DataFrame({"time": [100, 200], "event": [1, 0]})

        metrics = monitor.record_metrics(
            model_name="test_model",
            X=X,
            y_true=y_true,
            y_pred=np.array([0.1, 0.2]),
        )

        assert metrics.n_samples == 2
        assert metrics.concordance >= 0  # Should not crash

        # Test with empty dataset
        empty_X = pd.DataFrame({"feature": []})
        empty_y = pd.DataFrame({"time": [], "event": []})

        # Should handle gracefully (though concordance calculation may fail)
        try:
            empty_metrics = monitor.record_metrics(
                model_name="test_model",
                X=empty_X,
                y_true=empty_y,
                y_pred=np.array([]),
            )
            # If it doesn't crash, that's good
        except Exception:
            # Expected to fail with empty data
            pass

