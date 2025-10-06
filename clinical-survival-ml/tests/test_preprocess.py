from __future__ import annotations

import pandas as pd

from clinical_survival.preprocess import build_transformer, fit_transform, transform
from clinical_survival.utils import load_yaml, prepare_features


def test_transformer_shapes(toy_frame, project_root):
    feature_spec = load_yaml(project_root / "configs" / "features.yaml")
    features, feature_spec = prepare_features(toy_frame, feature_spec)
    transformer = build_transformer(feature_spec)
    transformed, transformer = fit_transform(transformer, features)
    assert transformed.shape[0] == len(features)
    transformed_again = transform(transformer, features)
    assert transformed_again.shape == transformed.shape
