"""
Data preprocessing pipeline step with schema validation and structured logging.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from clinical_survival.config import ParamsConfig, FeaturesConfig
from clinical_survival.error_handling import wrap_step_errors
from clinical_survival.logging_config import get_logger
from clinical_survival.pipeline.schemas import (
    PreprocessorInput,
    PreprocessorOutput,
    validate_pipeline_step,
)
from clinical_survival.preprocess.builder import build_declarative_preprocessor
from clinical_survival.utils import combine_survival_target

# Get module logger
logger = get_logger(__name__)


@validate_pipeline_step(
    input_schema=PreprocessorInput,
    output_schema=PreprocessorOutput,
)
@wrap_step_errors("preprocessing")
def prepare_data(
    raw_df: pd.DataFrame,
    params_config: ParamsConfig,
    features_config: FeaturesConfig,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """
    Builds the preprocessor from config, splits the data, and applies transformations.
    
    Input Schema (PreprocessorInput):
        - raw_df: pd.DataFrame - Raw input data
        - params_config: ParamsConfig - Main parameters configuration
        - features_config: FeaturesConfig - Feature engineering configuration
        
    Output Schema (PreprocessorOutput):
        - X_train: pd.DataFrame - Preprocessed training features
        - X_test: pd.DataFrame - Preprocessed test features
        - y_train: pd.Series - Training survival target
        - y_test: pd.Series - Test survival target
        - preprocessor: Pipeline - Fitted preprocessor
    
    Args:
        raw_df: Raw input DataFrame
        params_config: Main parameters configuration
        features_config: Feature engineering configuration
        **_kwargs: Additional context (unused)
        
    Returns:
        Dictionary containing processed train/test data and the fitted preprocessor
        
    Raises:
        SchemaValidationError: If input/output validation fails
    """
    logger.info("Starting data preparation")

    # Build the preprocessor using the declarative builder
    logger.debug(
        "Building declarative preprocessor",
        extra={"n_steps": len(features_config.preprocessing_pipeline)},
    )
    preprocessor = build_declarative_preprocessor(features_config)

    # Separate features (X) and survival outcome (y)
    target_cols = [features_config.time_to_event_col, features_config.event_col]
    X = raw_df.drop(columns=target_cols)
    y_surv = combine_survival_target(
        raw_df[features_config.time_to_event_col],
        raw_df[features_config.event_col],
    )

    logger.debug(
        "Feature/target separation complete",
        extra={
            "n_features": len(X.columns),
            "n_samples": len(X),
            "event_rate": float(raw_df[features_config.event_col].mean()),
        },
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_surv,
        test_size=params_config.test_split,
        random_state=params_config.seed,
        stratify=raw_df[features_config.event_col],
    )

    logger.info(
        "Train/test split complete",
        extra={
            "train_size": len(X_train),
            "test_size": len(X_test),
            "test_ratio": round(len(X_test) / len(X), 3),
        },
    )

    # Fit the preprocessor on the training data and transform both sets
    logger.debug("Fitting preprocessor on training data")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Convert processed data back to DataFrame for easier use
    try:
        feature_names = preprocessor.get_feature_names_out()
        X_train = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
        X_test = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)
        logger.debug(
            "Feature names extracted from preprocessor",
            extra={"n_output_features": len(feature_names)},
        )
    except Exception as e:
        logger.warning(
            "Could not retrieve feature names from preprocessor, using numeric indices",
            extra={"error": str(e)},
        )
        X_train = pd.DataFrame(X_train_processed, index=X_train.index)
        X_test = pd.DataFrame(X_test_processed, index=X_test.index)

    logger.info(
        "Data preparation complete",
        extra={
            "train_shape": X_train.shape,
            "test_shape": X_test.shape,
            "n_output_features": X_train.shape[1],
        },
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
    }
