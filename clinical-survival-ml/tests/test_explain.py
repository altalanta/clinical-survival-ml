from __future__ import annotations

import pandas as pd
from sksurv.util import Surv

from clinical_survival.explain import explain_model
from clinical_survival.models import make_model
from clinical_survival.preprocess import build_transformer, fit_transform
from clinical_survival.utils import ensure_dir, load_yaml, prepare_features


def test_explain_outputs(tmp_path, project_root):
    feature_spec = load_yaml(project_root / "configs" / "features.yaml")
    df = pd.read_csv(project_root / "data" / "toy" / "toy_survival.csv")
    features, feature_spec = prepare_features(df, feature_spec)
    transformer = build_transformer(feature_spec)
    X, transformer = fit_transform(transformer, features)
    y = Surv.from_dataframe("event", "time", df)
    model = make_model("coxph")
    model.fit(X, y)
    out_dir = ensure_dir(tmp_path / "explain")
    outputs = explain_model(model, X.head(20), y[:20], [180, 365], 50, ["age", "sofa"], out_dir)
    assert "permutation" in outputs
    assert outputs["permutation"]
