# ruff: noqa: S101
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from clinical_survival.models import PipelineModel, make_model
from clinical_survival.preprocess import build_preprocessor
from clinical_survival.tuning import nested_cv


def _pipeline_factory(feature_spec, seed):
    def factory(params: dict[str, object]) -> PipelineModel:
        transformer = build_preprocessor(feature_spec, {"strategy": "simple", "initial_strategy": "median"}, random_state=seed)
        estimator = make_model("coxph", random_state=seed, **params)
        pipeline = Pipeline([("pre", transformer), ("est", estimator)])
        wrapped = PipelineModel(pipeline)
        wrapped.name = "coxph"
        return wrapped

    return factory


def _toy_data() -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, list[str]]]:
    data = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 13.0, 14.0],
            "x2": [5.0, np.nan, 7.0, 8.0, 50.0, 52.0, 53.0, 54.0],
        }
    )
    time = pd.Series([5, 6, 7, 8, 9, 10, 11, 12], name="time")
    event = pd.Series([1, 1, 0, 1, 1, 0, 1, 1], name="event")
    feature_spec = {"numeric": ["x1", "x2"], "categorical": []}
    return data, time, event, feature_spec


def test_nested_cv_no_prefit_state():
    X, time, event, feature_spec = _toy_data()
    factory = _pipeline_factory(feature_spec, seed=123)
    result = nested_cv(
        "coxph",
        X,
        time,
        event,
        outer_splits=2,
        inner_splits=2,
        param_grid={},
        eval_times=[5, 10],
        random_state=123,
        pipeline_builder=factory,
    )

    global_transformer = build_preprocessor(feature_spec, {"strategy": "simple", "initial_strategy": "median"}, random_state=123)
    global_transformer.fit(X)
    global_stats = global_transformer.named_transformers_["numeric"].named_steps["impute"].statistics_

    for fold in result.folds:
        fold_stats = fold.preprocessor_summary.get("numeric_imputer")
        assert fold_stats is not None
        # Imputer fitted on fold-train data should differ from global fit statistics
        assert not np.allclose(fold_stats, global_stats)


def test_pipeline_fits_inside_fold():
    X, time, event, feature_spec = _toy_data()
    fit_counts = {"count": 0}

    def factory(params: dict[str, object]) -> PipelineModel:
        transformer = build_preprocessor(feature_spec, {"strategy": "simple", "initial_strategy": "median"}, random_state=7)
        estimator = make_model("coxph", random_state=7, **params)
        pipeline = Pipeline([("pre", transformer), ("est", estimator)])
        wrapped = PipelineModel(pipeline)
        wrapped.name = "coxph"

        original_fit = wrapped.fit

        def tracked_fit(X_train, y_train):
            fit_counts["count"] += 1
            return original_fit(X_train, y_train)

        wrapped.fit = tracked_fit  # type: ignore[assignment]
        return wrapped

    nested_cv(
        "coxph",
        X,
        time,
        event,
        outer_splits=2,
        inner_splits=2,
        param_grid={},
        eval_times=[5, 10],
        random_state=7,
        pipeline_builder=factory,
    )

    # Expected fits: inner loops (outer * (1 + inner_splits)) + outer fits + final fit
    expected_min = 2 * (1 + 2) + 2 + 1
    assert fit_counts["count"] >= expected_min
