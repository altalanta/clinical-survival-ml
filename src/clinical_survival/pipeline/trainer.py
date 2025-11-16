from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


from clinical_survival.config import ParamsConfig, FeaturesConfig
from clinical_survival.models import make_model
from clinical_survival.preprocess import build_preprocessor


def train_model(
    X: pd.DataFrame,
    y_surv: pd.DataFrame,
    model_name: str,
    params_config: ParamsConfig,
    features_config: FeaturesConfig,
    model_params: Dict[str, Any],
) -> Tuple[Pipeline, np.ndarray]:
    """Trains a model using cross-validation and returns the final model and OOF predictions."""

    oof_preds = np.zeros(len(X))
    kf = KFold(n_splits=params_config.n_splits, shuffle=True, random_state=params_config.seed)

    for _fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y_surv.iloc[train_idx]

        preprocessor = build_preprocessor(
            features_config.model_dump(),
            params_config.missing.model_dump(),
            random_state=params_config.seed,
        )

        model = make_model(model_name, random_state=params_config.seed, **model_params)

        pipeline = Pipeline([("pre", preprocessor), ("est", model)])
        pipeline.fit(X_train, y_train)

        oof_preds[test_idx] = pipeline.predict(X_test)

    # Train final model on full data
    final_pipeline = Pipeline(
        [
            (
                "pre",
                build_preprocessor(
                    features_config.model_dump(),
                    params_config.missing.model_dump(),
                    random_state=params_config.seed,
                ),
            ),
            ("est", make_model(model_name, random_state=params_config.seed, **model_params)),
        ]
    )
    final_pipeline.fit(X, y_surv)

    return final_pipeline, oof_preds

