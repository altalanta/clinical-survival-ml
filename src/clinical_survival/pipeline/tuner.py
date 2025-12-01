"""
Hyperparameter tuning pipeline step with schema validation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from clinical_survival.config import ParamsConfig, FeaturesConfig
from clinical_survival.error_handling import wrap_step_errors
from clinical_survival.logging_config import get_logger
from clinical_survival.pipeline.schemas import (
    TunerInput,
    TunerOutput,
    validate_pipeline_step,
)
from clinical_survival.tuning import run_tuning

# Get module logger
logger = get_logger(__name__)


@validate_pipeline_step(
    input_schema=TunerInput,
    output_schema=TunerOutput,
)
@wrap_step_errors("hyperparameter_tuning")
def tune_hyperparameters(
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.DataFrame] = None,
    params_config: Optional[ParamsConfig] = None,
    features_config: Optional[FeaturesConfig] = None,
    grid_config: Optional[Dict[str, Any]] = None,
    # Accept aliases for compatibility
    X: Optional[pd.DataFrame] = None,
    y_surv: Optional[pd.DataFrame] = None,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """
    Runs hyperparameter tuning for all models if enabled in the config.
    
    Input Schema (TunerInput):
        - X_train (or X): pd.DataFrame - Training features
        - y_train (or y_surv): pd.DataFrame - Training target
        - params_config: ParamsConfig - Main parameters configuration
        - features_config: FeaturesConfig - Features configuration
        - grid_config: Dict - Hyperparameter grid configuration
        
    Output Schema (TunerOutput):
        - best_params: Dict[str, Any] - Best hyperparameters for each model
    
    Args:
        X_train: Training features (can also use X alias)
        y_train: Training target (can also use y_surv alias)
        params_config: Main parameters configuration
        features_config: Features configuration
        grid_config: Hyperparameter grid configuration
        X: Alias for X_train
        y_surv: Alias for y_train
        **_kwargs: Additional context
        
    Returns:
        Dictionary with 'best_params' containing best hyperparameters per model
        
    Raises:
        SchemaValidationError: If input/output validation fails
    """
    # Handle aliases
    X_data = X_train if X_train is not None else X
    y_data = y_train if y_train is not None else y_surv
    
    if not params_config.tuning.enabled:
        logger.info("Hyperparameter tuning is disabled, using default parameters")
        return {"best_params": grid_config or {}}

    logger.info(
        "Starting hyperparameter tuning",
        extra={
            "models": params_config.models,
            "n_trials": params_config.tuning.trials,
        },
    )

    best_params_all_models: Dict[str, Any] = {}
    
    for model_name in params_config.models:
        if model_name not in grid_config:
            logger.warning(
                f"No search space defined for '{model_name}', skipping tuning",
                extra={"model": model_name},
            )
            continue

        search_space = grid_config[model_name]
        
        logger.info(
            f"Tuning hyperparameters for {model_name}",
            extra={
                "model": model_name,
                "search_space_params": list(search_space.keys()),
            },
        )
        
        best_params = run_tuning(
            X=X_data,
            y_surv=y_data,
            model_name=model_name,
            search_space=search_space,
            params_config=params_config,
            features_config=features_config,
        )
        best_params_all_models[model_name] = best_params
        
        logger.info(
            f"Tuning complete for {model_name}",
            extra={
                "model": model_name,
                "best_params": best_params,
            },
        )

    logger.info(
        "Hyperparameter tuning complete for all models",
        extra={"n_models_tuned": len(best_params_all_models)},
    )

    return {"best_params": best_params_all_models}
