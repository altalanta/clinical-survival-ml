from __future__ import annotations
from pathlib import Path
import json
from typing import Any, Dict

import pandas as pd
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


from clinical_survival.config import ParamsConfig, FeaturesConfig
from clinical_survival.io import load_dataset
from clinical_survival.models import make_model
from clinical_survival.preprocess import build_preprocessor
from clinical_survival.utils import (
    set_global_seed,
    prepare_features,
    combine_survival_target,
    ensure_dir,
)

BASELINE_PATH = Path("artifacts/testing/performance_baseline.json")


def run_performance_test(
    params_config: ParamsConfig, features_config: FeaturesConfig
) -> Dict[str, float]:
    """
    Runs a standardized performance test on the benchmark (toy) dataset.

    Returns:
        A dictionary containing performance metrics.
    """
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

    # 2. Use a fixed split for consistency
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_surv, test_size=0.3, random_state=params_config.seed, stratify=y_surv["event"]
    )

    # 3. Build and train a default model pipeline (e.g., CoxPH)
    # Using a simple, deterministic model is best for regression testing.
    preprocessor = build_preprocessor(
        features_config.model_dump(),
        params_config.missing.model_dump(),
        random_state=params_config.seed,
    )
    model = make_model("coxph", random_state=params_config.seed)
    pipeline = Pipeline([("pre", preprocessor), ("est", model)])
    pipeline.fit(X_train, y_train)

    # 4. Calculate performance metric
    predictions = pipeline.predict(X_test)
    event_times = y_test["time"]
    event_observed = y_test["event"]
    
    c_index = concordance_index(event_times, -predictions, event_observed)

    return {"concordance_index": c_index}


def update_baseline(metrics: Dict[str, float]) -> None:
    """Saves the given metrics as the new performance baseline."""
    ensure_dir(BASELINE_PATH.parent)
    with open(BASELINE_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


def check_regression(
    new_metrics: Dict[str, float], tolerance: float
) -> Dict[str, Any]:
    """
    Compares new metrics against the baseline to check for regression.

    Returns:
        A dictionary summarizing the regression check.
    """
    if not BASELINE_PATH.exists():
        return {"status": "error", "message": "Baseline not found. Please create one first."}

    with open(BASELINE_PATH, "r") as f:
        baseline_metrics = json.load(f)
    
    baseline_c = baseline_metrics.get("concordance_index", 0)
    new_c = new_metrics.get("concordance_index", 0)
    
    difference = new_c - baseline_c
    
    if difference < -tolerance:
        status = "failed"
        message = f"Concordance dropped by {abs(difference):.4f} (baseline: {baseline_c:.4f}, new: {new_c:.4f})"
    else:
        status = "passed"
        message = f"Performance is within tolerance (baseline: {baseline_c:.4f}, new: {new_c:.4f})"
        
    return {
        "status": status,
        "message": message,
        "baseline": baseline_c,
        "new": new_c,
        "difference": difference,
    }
