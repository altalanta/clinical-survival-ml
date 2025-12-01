"""Performance regression testing with comprehensive type hints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from clinical_survival.config import ParamsConfig, FeaturesConfig
from clinical_survival.io import load_dataset
from clinical_survival.logging_config import get_logger
from clinical_survival.models import make_model
from clinical_survival.preprocess.builder import build_declarative_preprocessor
from clinical_survival.utils import (
    set_global_seed,
    prepare_features,
    combine_survival_target,
    ensure_dir,
)

# Get module logger
logger = get_logger(__name__)

BASELINE_PATH = Path("artifacts/testing/performance_baseline.json")


def run_performance_test(
    params_config: ParamsConfig, features_config: FeaturesConfig
) -> Dict[str, float]:
    """
    Runs a standardized performance test on the benchmark (toy) dataset.
    
    Args:
        params_config: Main parameters configuration
        features_config: Feature engineering configuration

    Returns:
        A dictionary containing performance metrics.
    """
    logger.info("Starting performance regression test")
    
    set_global_seed(params_config.seed)

    # 1. Load benchmark data
    (X, y), _, _ = load_dataset(
        csv_path=params_config.paths.data_csv,
        metadata_path=params_config.paths.metadata,
        time_col=params_config.time_col,
        event_col=params_config.event_col,
    )
    y_surv = combine_survival_target(
        y[params_config.time_col], y[params_config.event_col]
    )
    X, _ = prepare_features(X, features_config.model_dump())

    logger.debug(
        "Benchmark data loaded",
        extra={"n_samples": len(X), "n_features": len(X.columns)},
    )

    # 2. Use a fixed split for consistency
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_surv, test_size=0.3, random_state=params_config.seed, stratify=y_surv["event"]
    )

    # 3. Build and train a default model pipeline (e.g., CoxPH)
    # Using a simple, deterministic model is best for regression testing.
    preprocessor = build_declarative_preprocessor(features_config)
    model = make_model("coxph")
    pipeline = Pipeline([("pre", preprocessor), ("est", model)])
    pipeline.fit(X_train, y_train)

    # 4. Calculate performance metric
    predictions = pipeline.predict(X_test)
    event_times = y_test["time"]
    event_observed = y_test["event"]

    c_index = concordance_index(event_times, -predictions, event_observed)

    logger.info(
        "Performance test complete",
        extra={"concordance_index": c_index},
    )

    return {"concordance_index": float(c_index)}


def update_baseline(metrics: Dict[str, float]) -> None:
    """
    Saves the given metrics as the new performance baseline.
    
    Args:
        metrics: Dictionary of metric names to values
    """
    ensure_dir(BASELINE_PATH.parent)
    with open(BASELINE_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(
        "Performance baseline updated",
        extra={"path": str(BASELINE_PATH), "metrics": metrics},
    )


def load_baseline() -> Optional[Dict[str, float]]:
    """
    Loads the performance baseline from disk.
    
    Returns:
        Baseline metrics dictionary, or None if not found
    """
    if not BASELINE_PATH.exists():
        return None
    
    with open(BASELINE_PATH, "r") as f:
        return json.load(f)


def check_regression(
    new_metrics: Dict[str, float], tolerance: float = 0.02
) -> Dict[str, Any]:
    """
    Compares new metrics against the baseline to check for regression.
    
    Args:
        new_metrics: Dictionary of new metric values
        tolerance: Allowed drop in concordance index before flagging regression

    Returns:
        A dictionary summarizing the regression check.
    """
    logger.info(
        "Checking for performance regression",
        extra={"tolerance": tolerance},
    )
    
    baseline_metrics = load_baseline()
    
    if baseline_metrics is None:
        logger.warning("Baseline not found, cannot check for regression")
        return {"status": "error", "message": "Baseline not found. Please create one first."}

    baseline_c = baseline_metrics.get("concordance_index", 0.0)
    new_c = new_metrics.get("concordance_index", 0.0)

    difference = new_c - baseline_c

    result: Dict[str, Any]
    if difference < -tolerance:
        result = {
            "status": "failed",
            "message": f"Concordance dropped by {abs(difference):.4f} (baseline: {baseline_c:.4f}, new: {new_c:.4f})",
            "baseline": baseline_c,
            "new": new_c,
            "difference": difference,
        }
        logger.warning(
            "Performance regression DETECTED",
            extra={"baseline": baseline_c, "new": new_c, "difference": difference},
        )
    else:
        result = {
            "status": "passed",
            "message": f"Performance is within tolerance (baseline: {baseline_c:.4f}, new: {new_c:.4f})",
            "baseline": baseline_c,
            "new": new_c,
            "difference": difference,
        }
        logger.info(
            "Performance regression check PASSED",
            extra={"baseline": baseline_c, "new": new_c, "difference": difference},
        )

    return result
