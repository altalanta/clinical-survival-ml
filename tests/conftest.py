"""
Pytest configuration and shared fixtures for clinical-survival-ml tests.

This module provides reusable fixtures for:
- Test data (DataFrames, survival data)
- Configuration objects
- Mock models and pipelines
- Temporary directories
- Performance assertions
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def toy_csv(project_root: Path) -> Path:
    """Return path to the toy survival dataset."""
    return project_root / "data" / "toy" / "toy_survival.csv"


@pytest.fixture(scope="session")
def metadata_path(project_root: Path) -> Path:
    """Return path to the toy metadata file."""
    return project_root / "data" / "toy" / "metadata.yaml"


@pytest.fixture(scope="session")
def synthetic_csv(project_root: Path) -> Path:
    """Return path to synthetic test data."""
    return project_root / "tests" / "data" / "synthetic_toy.csv"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_output_dir(temp_dir: Path) -> Path:
    """Provide a temporary output directory structure."""
    output_dir = temp_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "artifacts").mkdir()
    (output_dir / "artifacts" / "models").mkdir()
    (output_dir / "artifacts" / "explainability").mkdir()
    return output_dir


# =============================================================================
# Data Fixtures
# =============================================================================


@pytest.fixture()
def toy_frame(toy_csv: Path) -> pd.DataFrame:
    """Load the toy survival dataset."""
    return pd.read_csv(toy_csv)


@pytest.fixture
def sample_survival_df() -> pd.DataFrame:
    """
    Generate a small sample survival DataFrame for testing.
    
    Returns:
        DataFrame with columns: patient_id, time, event, age, biomarker, stage
    """
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        "patient_id": range(1, n_samples + 1),
        "time": np.random.exponential(scale=500, size=n_samples).astype(int) + 1,
        "event": np.random.binomial(1, 0.6, size=n_samples),
        "age": np.random.normal(60, 15, size=n_samples).astype(int),
        "biomarker_a": np.random.uniform(0, 100, size=n_samples),
        "biomarker_b": np.random.normal(50, 10, size=n_samples),
        "stage": np.random.choice(["I", "II", "III", "IV"], size=n_samples),
        "treatment": np.random.choice(["A", "B", "C"], size=n_samples),
    })


@pytest.fixture
def sample_X_y(sample_survival_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Split sample survival data into features and target.
    
    Returns:
        Tuple of (X features DataFrame, y structured array)
    """
    from sksurv.util import Surv
    
    X = sample_survival_df.drop(columns=["patient_id", "time", "event"])
    y = Surv.from_arrays(
        event=sample_survival_df["event"].astype(bool),
        time=sample_survival_df["time"].astype(float),
    )
    return X, y


@pytest.fixture
def sample_train_test_split(
    sample_X_y: Tuple[pd.DataFrame, np.ndarray]
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Provide train/test split of sample data.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    X, y = sample_X_y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def large_sample_df() -> pd.DataFrame:
    """
    Generate a larger sample DataFrame for performance testing.
    
    Returns:
        DataFrame with 1000 rows and multiple feature types
    """
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        "time": np.random.exponential(scale=500, size=n_samples).astype(int) + 1,
        "event": np.random.binomial(1, 0.5, size=n_samples),
    })
    
    # Add numerical features
    for i in range(10):
        df[f"num_{i}"] = np.random.normal(0, 1, size=n_samples)
    
    # Add categorical features
    for i in range(5):
        df[f"cat_{i}"] = np.random.choice([f"a{i}", f"b{i}", f"c{i}"], size=n_samples)
    
    # Add some missing values
    mask = np.random.random(size=n_samples) < 0.05
    df.loc[mask, "num_0"] = np.nan
    
    return df


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def sample_params_dict(temp_output_dir: Path, toy_csv: Path, metadata_path: Path) -> Dict[str, Any]:
    """
    Provide a minimal valid parameters configuration dictionary.
    """
    return {
        "seed": 42,
        "n_splits": 3,
        "inner_splits": 2,
        "test_split": 0.2,
        "time_column": "time",
        "event_column": "event",
        "id_col": "patient_id",
        "models": ["coxph"],
        "scoring": {"primary": "concordance_index_censored", "secondary": []},
        "calibration": {"times_days": [365, 730], "bins": 10},
        "decision_curve": {"times_days": [365], "thresholds": [0.1, 0.2, 0.3]},
        "missing": {"strategy": "simple", "max_iter": 10, "initial_strategy": "mean"},
        "explain": {"shap_samples": 50, "pdp_features": []},
        "monitoring": {
            "alert_thresholds": {
                "concordance_drop": 0.05,
                "brier_increase": 0.1,
                "feature_drift": 0.1,
                "concept_drift": 0.1,
            },
            "baseline_window_days": 30,
            "max_history_records": 1000,
        },
        "paths": {
            "data_csv": str(toy_csv),
            "metadata": str(metadata_path),
            "external_csv": None,
            "outdir": str(temp_output_dir),
            "features": str(metadata_path.parent / "features.yaml"),
        },
        "evaluation": {"bootstrap": 100},
        "external": {
            "label": "external",
            "group_column": "group",
            "train_value": "train",
            "external_value": "external",
        },
        "incremental_learning": {
            "enabled": False,
            "update_frequency_days": 7,
            "min_samples_for_update": 100,
            "max_samples_in_memory": 10000,
            "update_strategy": "sliding_window",
            "drift_detection_enabled": True,
            "create_backup_before_update": True,
            "backup_retention_days": 30,
        },
        "distributed_computing": {
            "enabled": False,
            "cluster_type": "local",
            "n_workers": 2,
            "threads_per_worker": 1,
            "memory_per_worker": "2GB",
            "partition_strategy": "auto",
            "n_partitions": 4,
            "scheduler_address": "",
            "dashboard_address": "",
        },
        "clinical_interpretability": {
            "enabled": False,
            "risk_thresholds": {"low": 0.3, "medium": 0.6, "high": 0.9},
            "clinical_context": {"feature_domains": {}, "risk_categories": {}},
            "explanation_features": {
                "include_shap_values": True,
                "include_clinical_interpretation": True,
                "include_risk_stratification": True,
                "include_recommendations": False,
                "include_confidence_scores": True,
            },
        },
        "mlops": {
            "enabled": False,
            "registry_path": str(temp_output_dir / "registry"),
            "environments": {},
            "triggers": {},
            "deployment_settings": {
                "require_approval_for_production": True,
                "auto_rollback_on_failure": True,
                "max_concurrent_deployments": 1,
                "deployment_timeout_minutes": 30,
            },
        },
        "mlflow_tracking": {
            "enabled": False,
            "experiment_name": "test_experiment",
            "tracking_uri": "mlruns",
        },
        "caching": {"enabled": False, "dir": str(temp_output_dir / "cache")},
        "data_validation": {"enabled": False, "expectation_suite": "clinical_data"},
        "tuning": {
            "enabled": False,
            "trials": 10,
            "metric": "concordance_index_censored",
            "direction": "maximize",
        },
        "counterfactuals": {
            "enabled": False,
            "n_examples": 3,
            "sample_size": 100,
            "features_to_vary": [],
            "output_format": "json",
        },
        "pipeline": [
            "data_loader.load_raw_data",
            "preprocessor.prepare_data",
            "training_loop.run_training_loop",
        ],
        "logging": {"level": "WARNING", "format": "simple"},
    }


@pytest.fixture
def sample_features_dict() -> Dict[str, Any]:
    """Provide a minimal valid features configuration dictionary."""
    return {
        "numerical_cols": ["age", "biomarker_a", "biomarker_b"],
        "categorical_cols": ["stage", "treatment"],
        "binary_cols": [],
        "time_to_event_col": "time",
        "event_col": "event",
        "preprocessing_pipeline": [
            {"component": "StandardScaler", "columns": ["age", "biomarker_a", "biomarker_b"]},
            {"component": "OneHotEncoder", "columns": ["stage", "treatment"]},
        ],
    }


@pytest.fixture
def sample_grid_dict() -> Dict[str, Any]:
    """Provide a minimal model grid configuration dictionary."""
    return {
        "coxph": {"alpha": 0.1},
        "rsf": {"n_estimators": 50, "min_samples_split": 10},
    }


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_mlflow_tracker() -> MagicMock:
    """Provide a mock MLflow tracker."""
    tracker = MagicMock()
    tracker.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    tracker.start_run.return_value.__exit__ = MagicMock(return_value=False)
    tracker.log_params = MagicMock()
    tracker.log_metrics = MagicMock()
    tracker.log_artifact = MagicMock()
    tracker.log_model = MagicMock()
    return tracker


@pytest.fixture
def mock_survival_model() -> MagicMock:
    """
    Provide a mock survival model with the expected interface.
    """
    model = MagicMock()
    model.fit.return_value = model
    model.predict_risk.return_value = np.random.randn(10)
    model.predict_survival_function.return_value = np.random.rand(10, 5)
    return model


@pytest.fixture
def mock_pipeline() -> MagicMock:
    """Provide a mock sklearn pipeline."""
    pipeline = MagicMock()
    pipeline.fit.return_value = pipeline
    pipeline.transform.return_value = np.random.randn(10, 5)
    pipeline.fit_transform.return_value = np.random.randn(10, 5)
    pipeline.get_feature_names_out.return_value = [f"feature_{i}" for i in range(5)]
    pipeline.named_steps = {"pre": MagicMock(), "est": MagicMock()}
    return pipeline


# =============================================================================
# Performance Testing Fixtures
# =============================================================================


@pytest.fixture
def performance_timer():
    """
    Provide a context manager for timing code execution.
    
    Usage:
        with performance_timer() as timer:
            # code to time
        assert timer.elapsed < 1.0  # Should complete in under 1 second
    """
    class Timer:
        def __init__(self):
            self.start_time: float = 0
            self.end_time: float = 0
            self.elapsed: float = 0
        
        def __enter__(self):
            import time
            self.start_time = time.perf_counter()
            return self
        
        def __exit__(self, *args):
            import time
            self.end_time = time.perf_counter()
            self.elapsed = self.end_time - self.start_time
    
    return Timer


@pytest.fixture
def assert_memory_usage():
    """
    Provide a function to assert memory usage is below a threshold.
    
    Usage:
        assert_memory_usage(lambda: some_function(), max_mb=100)
    """
    def _assert_memory(func, max_mb: float):
        import tracemalloc
        
        tracemalloc.start()
        func()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / 1024 / 1024
        assert peak_mb < max_mb, f"Peak memory {peak_mb:.2f}MB exceeded {max_mb}MB limit"
        return peak_mb
    
    return _assert_memory


# =============================================================================
# Cleanup Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state before each test for reproducibility."""
    np.random.seed(42)
    yield


@pytest.fixture
def cleanup_mlflow_runs():
    """Clean up any MLflow runs after tests."""
    yield
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.end_run()
    except Exception:
        pass


# =============================================================================
# Parametrization Helpers
# =============================================================================


def get_model_names() -> List[str]:
    """Return list of supported model names for parametrized tests."""
    return ["coxph", "rsf", "xgb_cox", "xgb_aft"]


def get_preprocessing_strategies() -> List[str]:
    """Return list of preprocessing strategies for parametrized tests."""
    return ["simple", "iterative"]


# =============================================================================
# Markers
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "network: marks tests that require network access"
    )
