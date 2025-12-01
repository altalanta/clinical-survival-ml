"""Model training pipeline step with comprehensive type hints."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from clinical_survival.config import ParamsConfig, FeaturesConfig
from clinical_survival.logging_config import get_logger
from clinical_survival.models import make_model
from clinical_survival.preprocess.builder import build_declarative_preprocessor

# Get module logger
logger = get_logger(__name__)


def train_model(
    X: pd.DataFrame,
    y_surv: pd.DataFrame,
    model_name: str,
    params_config: ParamsConfig,
    features_config: FeaturesConfig,
    model_params: Dict[str, Any],
) -> Tuple[Pipeline, np.ndarray]:
    """
    Trains a model using cross-validation and returns the final model and OOF predictions.
    
    Args:
        X: Feature DataFrame
        y_surv: Survival target DataFrame with time and event columns
        model_name: Name of the model to train
        params_config: Main parameters configuration
        features_config: Feature engineering configuration
        model_params: Model hyperparameters
        
    Returns:
        Tuple of (fitted pipeline, out-of-fold predictions array)
    """
    logger.info(
        f"Starting model training: {model_name}",
        extra={
            "model": model_name,
            "n_samples": len(X),
            "n_features": len(X.columns),
            "n_splits": params_config.n_splits,
        },
    )

    oof_preds = np.zeros(len(X))
    kf = KFold(n_splits=params_config.n_splits, shuffle=True, random_state=params_config.seed)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y_surv.iloc[train_idx]

        preprocessor = build_declarative_preprocessor(features_config)
        model = make_model(model_name, **model_params)
        pipeline = Pipeline([("pre", preprocessor), ("est", model)])
        
        pipeline.fit(X_train, y_train)
        oof_preds[test_idx] = pipeline.predict(X_test)

        logger.debug(
            f"Completed fold {fold + 1}/{params_config.n_splits}",
            extra={"fold": fold + 1, "train_size": len(train_idx), "test_size": len(test_idx)},
        )

    # Train final model on full data
    logger.debug("Training final model on full dataset")
    final_preprocessor = build_declarative_preprocessor(features_config)
    final_model = make_model(model_name, **model_params)
    final_pipeline = Pipeline([("pre", final_preprocessor), ("est", final_model)])
    final_pipeline.fit(X, y_surv)

    logger.info(
        f"Model training complete: {model_name}",
        extra={"model": model_name, "oof_preds_shape": oof_preds.shape},
    )

    return final_pipeline, oof_preds
