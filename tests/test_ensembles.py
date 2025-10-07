# ruff: noqa: S101
"""Tests for ensemble methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from clinical_survival.ensembles import BaggingEnsemble, DynamicEnsemble, StackingEnsemble
from clinical_survival.models import make_model
from clinical_survival.preprocess import build_preprocessor


def test_stacking_ensemble_basic():
    """Test basic stacking ensemble functionality."""
    # Create simple test data
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=50),
            "x2": rng.normal(size=50),
        }
    )
    times = rng.integers(5, 100, size=50)
    events = rng.integers(0, 2, size=50)
    y = [(bool(e), float(t)) for e, t in zip(events, times, strict=True)]

    # Create base models
    base_models = [
        make_model("coxph", random_state=42),
        make_model("rsf", random_state=42, n_estimators=50),
    ]

    # Create stacking ensemble
    ensemble = StackingEnsemble(base_models=base_models, cv_folds=3, random_state=42)

    # Fit ensemble
    ensemble.fit(X, y)

    # Make predictions
    risk_pred = ensemble.predict_risk(X)
    survival_pred = ensemble.predict_survival_function(X, [30, 60, 90])

    # Check output shapes
    assert risk_pred.shape == (50,)
    assert survival_pred.shape == (50, 3)

    # Check survival probabilities are valid
    assert np.all(survival_pred >= 0) and np.all(survival_pred <= 1)


def test_bagging_ensemble_basic():
    """Test basic bagging ensemble functionality."""
    # Create simple test data
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=50),
            "x2": rng.normal(size=50),
        }
    )
    times = rng.integers(5, 100, size=50)
    events = rng.integers(0, 2, size=50)
    y = [(bool(e), float(t)) for e, t in zip(events, times, strict=True)]

    # Create base model for bagging
    base_model = make_model("rsf", random_state=42, n_estimators=50)

    # Create bagging ensemble
    ensemble = BaggingEnsemble(
        base_model=base_model, n_estimators=5, max_samples=0.8, random_state=42
    )

    # Fit ensemble
    ensemble.fit(X, y)

    # Make predictions
    risk_pred = ensemble.predict_risk(X)
    survival_pred = ensemble.predict_survival_function(X, [30, 60, 90])

    # Check output shapes
    assert risk_pred.shape == (50,)
    assert survival_pred.shape == (50, 3)

    # Check survival probabilities are valid
    assert np.all(survival_pred >= 0) and np.all(survival_pred <= 1)


def test_dynamic_ensemble_basic():
    """Test basic dynamic ensemble functionality."""
    # Create simple test data
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=50),
            "x2": rng.normal(size=50),
        }
    )
    times = rng.integers(5, 100, size=50)
    events = rng.integers(0, 2, size=50)
    y = [(bool(e), float(t)) for e, t in zip(events, times, strict=True)]

    # Create base models
    base_models = [
        make_model("coxph", random_state=42),
        make_model("rsf", random_state=42, n_estimators=50),
        make_model("xgb_cox", random_state=42, n_estimators=50),
    ]

    # Create dynamic ensemble
    ensemble = DynamicEnsemble(
        base_models=base_models, selection_method="performance", random_state=42
    )

    # Fit ensemble
    ensemble.fit(X, y)

    # Make predictions
    risk_pred = ensemble.predict_risk(X)
    survival_pred = ensemble.predict_survival_function(X, [30, 60, 90])

    # Check output shapes
    assert risk_pred.shape == (50,)
    assert survival_pred.shape == (50, 3)

    # Check survival probabilities are valid
    assert np.all(survival_pred >= 0) and np.all(survival_pred <= 1)


def test_ensemble_with_missing_data():
    """Test ensemble methods handle missing data correctly."""
    # Create test data with missing values
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=50),
            "x2": rng.normal(size=50),
        }
    )
    # Introduce missing values
    X.loc[0:5, "x1"] = np.nan
    X.loc[10:15, "x2"] = np.nan

    times = rng.integers(5, 100, size=50)
    events = rng.integers(0, 2, size=50)
    y = [(bool(e), float(t)) for e, t in zip(events, times, strict=True)]

    # Create preprocessing pipeline (not used in this test)

    # Test each ensemble type
    base_models = [make_model("coxph", random_state=42)]

    for ensemble_class in [StackingEnsemble, BaggingEnsemble, DynamicEnsemble]:
        if ensemble_class == StackingEnsemble:
            ensemble = ensemble_class(base_models=[make_model("coxph", random_state=42)])
        elif ensemble_class == BaggingEnsemble:
            ensemble = ensemble_class(
                base_model=make_model("rsf", random_state=42, n_estimators=20)
            )
        else:  # DynamicEnsemble
            ensemble = ensemble_class(base_models=base_models)

        # Should handle missing data gracefully
        ensemble.fit(X, y)
        risk_pred = ensemble.predict_risk(X)
        survival_pred = ensemble.predict_survival_function(X, [30, 60])

        assert risk_pred.shape == (50,)
        assert survival_pred.shape == (50, 2)


def test_ensemble_error_handling():
    """Test ensemble error handling for edge cases."""
    # Test with very small dataset
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=10),
            "x2": rng.normal(size=10),
        }
    )
    times = rng.integers(5, 100, size=10)
    events = rng.integers(0, 2, size=10)
    y = [(bool(e), float(t)) for e, t in zip(events, times, strict=True)]

    # Test stacking ensemble
    base_models = [make_model("coxph", random_state=42)]
    ensemble = StackingEnsemble(base_models=base_models, cv_folds=2)

    # Should handle small datasets gracefully
    ensemble.fit(X, y)
    risk_pred = ensemble.predict_risk(X)
    survival_pred = ensemble.predict_survival_function(X, [30])

    assert risk_pred.shape == (10,)
    assert survival_pred.shape == (10, 1)


def test_ensemble_model_factory_integration():
    """Test that ensemble models work with the model factory."""
    # Test that make_model can create ensemble models
    stacking_model = make_model("stacking", random_state=42)
    assert isinstance(stacking_model, StackingEnsemble)

    bagging_model = make_model("bagging", random_state=42)
    assert isinstance(bagging_model, BaggingEnsemble)

    dynamic_model = make_model("dynamic", random_state=42)
    assert isinstance(dynamic_model, DynamicEnsemble)

    # Test with parameters
    stacking_model = make_model(
        "stacking", random_state=42, base_models=["coxph", "rsf"], cv_folds=3
    )
    assert isinstance(stacking_model, StackingEnsemble)

    bagging_model = make_model("bagging", random_state=42, base_model="rsf", n_estimators=5)
    assert isinstance(bagging_model, BaggingEnsemble)


def test_ensemble_serialization():
    """Test that ensemble models can be serialized/deserialized."""
    import tempfile
    from pathlib import Path

    import joblib

    # Create simple test data
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=30),
            "x2": rng.normal(size=30),
        }
    )
    times = rng.integers(5, 100, size=30)
    events = rng.integers(0, 2, size=30)
    y = [(bool(e), float(t)) for e, t in zip(events, times, strict=True)]

    # Create and fit ensemble
    base_models = [make_model("coxph", random_state=42)]
    ensemble = StackingEnsemble(base_models=base_models, cv_folds=3)
    ensemble.fit(X, y)

    # Test serialization
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        temp_path = f.name

    try:
        # Save model
        joblib.dump(ensemble, temp_path)

        # Load model
        loaded_ensemble = joblib.load(temp_path)

        # Test that loaded model works
        risk_pred = loaded_ensemble.predict_risk(X)
        survival_pred = loaded_ensemble.predict_survival_function(X, [30])

        assert risk_pred.shape == (30,)
        assert survival_pred.shape == (30, 1)

    finally:
        Path(temp_path).unlink()


def test_ensemble_with_preprocessing():
    """Test ensemble methods work with preprocessing pipelines."""
    # Create test data with categorical features
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "age": rng.normal(50, 10, size=50),
            "sex": rng.choice(["M", "F"], size=50),
            "stage": rng.choice([1, 2, 3, 4], size=50),
        }
    )
    times = rng.integers(5, 100, size=50)
    events = rng.integers(0, 2, size=50)
    y = [(bool(e), float(t)) for e, t in zip(events, times, strict=True)]

    # Create feature specification
    feature_spec = {"numeric": ["age"], "categorical": ["sex", "stage"]}

    # Create preprocessing pipeline
    preprocessor = build_preprocessor(feature_spec, {"strategy": "simple"}, random_state=42)

    # Create ensemble with preprocessing
    base_models = [make_model("coxph", random_state=42)]
    ensemble = StackingEnsemble(base_models=base_models)

    # Create full pipeline
    pipeline = Pipeline([("pre", preprocessor), ("ensemble", ensemble)])

    # Fit pipeline
    pipeline.fit(X, y)

    # Make predictions
    risk_pred = pipeline.predict_risk(X)
    survival_pred = pipeline.predict_survival_function(X, [30, 60])

    assert risk_pred.shape == (50,)
    assert survival_pred.shape == (50, 2)
