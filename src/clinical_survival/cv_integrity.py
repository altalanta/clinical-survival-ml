from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
from sklearn.pipeline import Pipeline

from clinical_survival.config import ParamsConfig, FeaturesConfig
from clinical_survival.models import make_model
from clinical_survival.preprocess import build_preprocessor
from clinical_survival.utils import combine_survival_target, set_global_seed


def _generate_leakage_data(
    n_samples: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates a synthetic dataset specifically designed to detect leakage."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "feature1": rng.standard_normal(n_samples),
            "feature2": rng.standard_normal(n_samples),
        }
    )

    # A simple linear relationship for survival time
    true_times = 100 * np.exp(
        -0.5 * X["feature1"] + rng.standard_normal(n_samples) * 0.1
    )

    # Add censoring
    censor_times = rng.uniform(0, 200, n_samples)
    event = true_times < censor_times
    time = np.minimum(true_times, censor_times)

    y_surv = combine_survival_target(time, event)

    return X, y_surv


def run_cv_leakage_test(
    params_config: ParamsConfig, features_config: FeaturesConfig
) -> Dict[str, Any]:
    """
    Runs a test to detect data leakage in the cross-validation preprocessing pipeline.

    It works by injecting a feature that is only correlated with the target in the
    test set. A leaky preprocessor will learn this correlation and show an
    artificially high performance gain.
    """
    set_global_seed(params_config.seed)
    n_samples = 500
    leakage_threshold = 0.1  # A C-index jump > 0.1 is a strong signal of leakage

    X, y_surv = _generate_leakage_data(n_samples, params_config.seed)

    kf = KFold(
        n_splits=params_config.n_splits, shuffle=True, random_state=params_config.seed
    )

    scores_with_leakage = []
    scores_without_leakage = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y_surv)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_surv.iloc[train_idx], y_surv.iloc[test_idx]

        # Inject the leakage feature
        X_train_leak = X_train.copy()
        X_test_leak = X_test.copy()

        rng = np.random.default_rng(params_config.seed + fold)
        X_train_leak["leakage_feature"] = rng.standard_normal(len(X_train_leak))
        # In the test set, the feature is highly correlated with the event time
        X_test_leak["leakage_feature"] = -y_test["time"]

        # --- Test WITHOUT leakage feature ---
        preprocessor = build_preprocessor(
            features_config.model_dump(),
            params_config.missing.model_dump(),
            random_state=params_config.seed,
        )
        model = make_model("coxph", random_state=params_config.seed)
        pipeline = Pipeline([("pre", preprocessor), ("est", model)])
        pipeline.fit(X_train, y_train)
        preds_no_leak = pipeline.predict(X_test)
        c_index_no_leak = concordance_index(
            y_test["time"], -preds_no_leak, y_test["event"]
        )
        scores_without_leakage.append(c_index_no_leak)

        # --- Test WITH leakage feature ---
        features_config_leak = features_config.model_copy(deep=True)
        if 'leakage_feature' not in features_config_leak.numeric_features:
            features_config_leak.numeric_features.append("leakage_feature")

        preprocessor_leak = build_preprocessor(
            features_config_leak.model_dump(),
            params_config.missing.model_dump(),
            random_state=params_config.seed,
        )
        model_leak = make_model("coxph", random_state=params_config.seed)
        pipeline_leak = Pipeline([("pre", preprocessor_leak), ("est", model_leak)])
        pipeline_leak.fit(X_train_leak, y_train)
        preds_leak = pipeline_leak.predict(X_test_leak)
        c_index_leak = concordance_index(y_test["time"], -preds_leak, y_test["event"])
        scores_with_leakage.append(c_index_leak)

    # --- Compare results ---
    mean_c_without = np.mean(scores_without_leakage)
    mean_c_with = np.mean(scores_with_leakage)
    difference = mean_c_with - mean_c_without

    result = {
        "mean_concordance_without_leakage": mean_c_without,
        "mean_concordance_with_leakage": mean_c_with,
        "difference": difference,
    }

    if difference > leakage_threshold:
        result["status"] = "failed"
        result[
            "message"
        ] = f"Data leakage DETECTED. Concordance jumped by {difference:.4f}."
    else:
        result["status"] = "passed"
        result[
            "message"
        ] = f"No data leakage detected. Concordance difference was {difference:.4f}."

    return result


