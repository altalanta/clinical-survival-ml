# ruff: noqa: S101
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clinical_survival.models import XGBAFTModel

xgb = pytest.importorskip("xgboost")


def test_survival_function_monotone_and_valid():
    rng = np.random.default_rng(123)
    X = pd.DataFrame({"x1": rng.normal(size=50), "x2": rng.normal(size=50)})
    times = rng.integers(5, 100, size=50)
    events = rng.integers(0, 2, size=50)
    y = [(bool(e), float(t)) for e, t in zip(events, times, strict=True)]

    model = XGBAFTModel(n_estimators=50, learning_rate=0.1, max_depth=2, aft_loss_distribution="logistic", aft_loss_distribution_scale=1.5)
    model.fit(X, y)

    horizons = [5.0, 10.0, 25.0, 50.0, 100.0]
    surv = model.predict_survival_function(X.head(3), horizons)
    assert surv.shape == (3, len(horizons))
    assert np.all(surv <= 1.0 + 1e-6)
    assert np.all(surv >= -1e-6)
    # Survival must be non-increasing in time
    assert np.all(np.diff(surv, axis=1) <= 1e-6)
    # Survival at earliest horizon should be close to one
    assert np.allclose(surv[:, 0], 1.0, atol=0.1)
