from __future__ import annotations

import numpy as np
import pandas as pd
from sksurv.util import Surv

from clinical_survival.eval import evaluate
from clinical_survival.models import make_model
from clinical_survival.preprocess import build_transformer, fit_transform
from clinical_survival.utils import ensure_dir, load_yaml, prepare_features


def test_evaluate_outputs(tmp_path, project_root):
    feature_spec = load_yaml(project_root / "configs" / "features.yaml")
    df = pd.read_csv(project_root / "data" / "toy" / "toy_survival.csv")
    features, feature_spec = prepare_features(df, feature_spec)
    transformer = build_transformer(feature_spec)
    X, transformer = fit_transform(transformer, features)
    y = Surv.from_dataframe("event", "time", df)
    model = make_model("coxph")
    model.fit(X, y)
    out_dir = ensure_dir(tmp_path / "metrics")
    result = evaluate(model, "coxph", X, y, X, y, [90, 180, 365], out_dir, label="cv")
    assert "concordance" in result.metrics
    assert result.reliability.shape[0] > 0
    assert result.net_benefit.shape[0] > 0
    assert result.calibration_path.exists()
    assert result.decision_path.exists()
