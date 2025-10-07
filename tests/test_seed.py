# ruff: noqa: S101
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from clinical_survival.eval import compute_metrics
from clinical_survival.models import PipelineModel, make_model
from clinical_survival.preprocess import build_preprocessor
from clinical_survival.tuning import nested_cv


def _factory(feature_spec, seed):
    def factory(params: dict[str, object]) -> PipelineModel:
        transformer = build_preprocessor(feature_spec, {"strategy": "iterative", "max_iter": 5}, random_state=seed)
        estimator = make_model("coxph", random_state=seed, **params)
        pipeline = Pipeline([("pre", transformer), ("est", estimator)])
        wrapped = PipelineModel(pipeline)
        wrapped.name = "coxph"
        return wrapped

    return factory


def _dataset(seed: int = 123) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, list[str]]]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f1": rng.normal(size=30), "f2": rng.normal(size=30)})
    time = pd.Series(rng.integers(5, 40, size=30), name="time")
    event = pd.Series(rng.integers(0, 2, size=30), name="event")
    feature_spec = {"numeric": ["f1", "f2"], "categorical": []}
    return X, time, event, feature_spec


def test_pipeline_deterministic_seed():
    X, time, event, feature_spec = _dataset()

    def run_once(seed: int):
        result = nested_cv(
            "coxph",
            X,
            time,
            event,
            outer_splits=3,
            inner_splits=2,
            param_grid={},
            eval_times=[10, 20],
            random_state=seed,
            pipeline_builder=_factory(feature_spec, seed),
        )
        oof_risk = np.zeros(len(X))
        oof_surv = np.zeros((len(X), 2))
        for fold in result.folds:
            oof_risk[fold.test_indices] = fold.risk
            oof_surv[fold.test_indices, :] = fold.survival
        metrics = compute_metrics(
            pd.DataFrame({"time": time, "event": event}),
            oof_risk,
            oof_surv,
            [10, 20],
            bootstrap=0,
            seed=seed,
        )
        return oof_risk, oof_surv, metrics

    risk_a, surv_a, metrics_a = run_once(77)
    risk_b, surv_b, metrics_b = run_once(77)

    assert np.allclose(risk_a, risk_b)
    assert np.allclose(surv_a, surv_b)
    for key in metrics_a:
        assert np.isclose(metrics_a[key].estimate, metrics_b[key].estimate)
