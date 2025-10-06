from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sksurv.util import Surv

from clinical_survival.models import make_model
from clinical_survival.preprocess import build_transformer, fit_transform
from clinical_survival.utils import load_yaml, prepare_features


@pytest.fixture(scope="module")
def processed_data(project_root):
    feature_spec = load_yaml(project_root / "configs" / "features.yaml")
    df = pd.read_csv(project_root / "data" / "toy" / "toy_survival.csv")
    features, feature_spec = prepare_features(df, feature_spec)
    transformer = build_transformer(feature_spec)
    X, transformer = fit_transform(transformer, features)
    y = Surv.from_dataframe("event", "time", df)
    return X, y


@pytest.mark.parametrize("model_name", ["coxph", "rsf"])
def test_model_fit_predict(model_name, processed_data):
    X, y = processed_data
    model = make_model(model_name)
    model.fit(X, y)
    risk = model.predict_risk(X.head(10))
    assert np.isfinite(risk).all()
    surv = model.predict_survival_function(X.head(5), [90, 180, 365])
    assert surv.shape == (5, 3)
    assert np.all((surv >= 0) & (surv <= 1))


def test_xgb_models_optional(processed_data):
    pytest.importorskip("xgboost")
    X, y = processed_data
    for model_name in ("xgb_cox", "xgb_aft"):
        model = make_model(model_name, n_estimators=10, max_depth=2)
        model.fit(X.head(50), y[:50])
        risk = model.predict_risk(X.head(10))
        assert np.isfinite(risk).all()
