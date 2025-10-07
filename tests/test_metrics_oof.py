# ruff: noqa: S101
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sksurv.metrics import brier_score, concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score
from sksurv.util import Surv

from clinical_survival.eval import compute_metrics
from clinical_survival.models import PipelineModel, make_model
from clinical_survival.preprocess import build_preprocessor
from clinical_survival.tuning import nested_cv


def _dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, list[str]]]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "x1": rng.normal(size=40),
        "x2": rng.normal(size=40),
    })
    time = pd.Series(rng.integers(5, 50, size=40), name="time")
    event = pd.Series(rng.integers(0, 2, size=40), name="event")
    feature_spec = {"numeric": ["x1", "x2"], "categorical": []}
    return X, time, event, feature_spec


def _factory(feature_spec, seed):
    def factory(params: dict[str, object]) -> PipelineModel:
        transformer = build_preprocessor(feature_spec, {"strategy": "iterative", "max_iter": 5}, random_state=seed)
        estimator = make_model("rsf", random_state=seed, **params)
        pipeline = Pipeline([("pre", transformer), ("est", estimator)])
        wrapped = PipelineModel(pipeline)
        wrapped.name = "rsf"
        return wrapped

    return factory


def test_oof_metrics_match_manual():
    X, time, event, feature_spec = _dataset()
    result = nested_cv(
        "rsf",
        X,
        time,
        event,
        outer_splits=3,
        inner_splits=2,
        param_grid={"n_estimators": [50]},
        eval_times=[10, 20, 30],
        random_state=99,
        pipeline_builder=_factory(feature_spec, 99),
    )

    n_samples = len(X)
    times = [10.0, 20.0, 30.0]
    oof_risk = np.zeros(n_samples)
    oof_surv = np.zeros((n_samples, len(times)))
    for fold in result.folds:
        oof_risk[fold.test_indices] = fold.risk
        oof_surv[fold.test_indices, :] = fold.survival

    y_frame = pd.DataFrame({"time": time, "event": event})
    metrics = compute_metrics(y_frame, oof_risk, oof_surv, times, bootstrap=0, seed=0)

    surv_struct = Surv.from_dataframe("event", "time", y_frame)
    manual_cindex = concordance_index_censored(event.astype(bool), time.to_numpy(), -oof_risk)[0]
    manual_ibs = integrated_brier_score(surv_struct, surv_struct, oof_surv, np.asarray(times))
    auc_curve, _ = cumulative_dynamic_auc(surv_struct, surv_struct, oof_risk, np.asarray(times))

    assert np.isclose(metrics["concordance"].estimate, manual_cindex)
    assert np.isclose(metrics["ibs"].estimate, manual_ibs)
    for idx, t in enumerate(times):
        bs = brier_score(surv_struct, surv_struct, oof_surv[:, idx], t)[0]
        assert np.isclose(metrics[f"brier@{int(t)}"].estimate, bs)
        assert np.isclose(metrics[f"auc@{int(t)}"].estimate, auc_curve[idx])
