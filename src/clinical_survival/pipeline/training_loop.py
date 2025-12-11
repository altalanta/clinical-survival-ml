"""
Training loop pipeline step with schema validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from clinical_survival.config import ParamsConfig, FeaturesConfig
from clinical_survival.error_handling import wrap_step_errors
from clinical_survival.logging_config import get_logger
from clinical_survival.pipeline.schemas import (
    TrainingLoopInput,
    TrainingLoopOutput,
    validate_pipeline_step,
)
from clinical_survival.tracking import MLflowTracker
from clinical_survival.utils import ensure_dir, combine_survival_target

from . import trainer, evaluator, explainer, registrar

# Get module logger
logger = get_logger(__name__)


@validate_pipeline_step(
    input_schema=TrainingLoopInput,
    output_schema=TrainingLoopOutput,
)
@wrap_step_errors("training_loop")
def run_training_loop(
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.DataFrame] = None,
    params_config: Optional[ParamsConfig] = None,
    features_config: Optional[FeaturesConfig] = None,
    best_params: Optional[Dict[str, Any]] = None,
    tracker: Optional[MLflowTracker] = None,
    outdir: Optional[Path] = None,
    # Accept aliases for compatibility
    X: Optional[pd.DataFrame] = None,
    y_surv: Optional[pd.DataFrame] = None,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """
    Runs the training loop for each model specified in the config.
    
    Input Schema (TrainingLoopInput):
        - X_train (or X): pd.DataFrame - Training features
        - y_train (or y_surv): pd.DataFrame - Training target
        - params_config: ParamsConfig - Main parameters configuration
        - features_config: FeaturesConfig - Features configuration
        - best_params: Dict - Best hyperparameters from tuning
        - tracker: MLflowTracker - MLflow tracker instance
        - outdir: Path - Output directory
        
    Output Schema (TrainingLoopOutput):
        - final_pipelines: Dict[str, Pipeline] - Trained model pipelines
    
    Args:
        X_train: Training features (can also use X alias)
        y_train: Training target (can also use y_surv alias)
        params_config: Main parameters configuration
        features_config: Features configuration
        best_params: Best hyperparameters from tuning step
        tracker: MLflow tracker instance
        outdir: Output directory path
        X: Alias for X_train
        y_surv: Alias for y_train
        **_kwargs: Additional context
        
    Returns:
        Dictionary with 'final_pipelines' containing trained models
        
    Raises:
        SchemaValidationError: If input/output validation fails
    """
    # Handle aliases
    X_data = X_train if X_train is not None else X
    y_data = y_train if y_train is not None else y_surv
    
    logger.info(
        "Starting training loop",
        extra={
            "models": params_config.models,
            "n_samples": len(X_data),
            "n_features": X_data.shape[1] if hasattr(X_data, "shape") else "unknown",
        },
    )
    
    models_dir = ensure_dir(outdir / "artifacts" / "models")
    metrics_dir = ensure_dir(outdir / "metrics")
    final_pipelines: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    cv_results: Dict[str, Any] = {}
    times = params_config.calibration.times_days if params_config and params_config.calibration else None

    for model_name in params_config.models:
        logger.info(f"Training model: {model_name}", extra={"model": model_name})
        
        with tracker.start_run(f"train_{model_name}"):
            model_params = best_params.get(model_name, {})
            tracker.log_params(model_params)

            # Train
            logger.debug(
                f"Starting training for {model_name}",
                extra={"model": model_name, "params": model_params},
            )
            final_pipeline, oof_preds = trainer.train_model(
                X_data, y_data, model_name, params_config, features_config, model_params
            )

            # Evaluate
            logger.debug(f"Evaluating {model_name}")
            y_struct = combine_survival_target(
                time=y_data[params_config.time_col],
                event=y_data[params_config.event_col],
            ).values
            eval_result = evaluator.evaluate_predictions(
                oof_preds,
                y_struct,
                tracker=tracker,
                model_name=model_name,
                times=times,
            )
            metrics[model_name] = {
                "concordance": eval_result.concordance_index,
                "concordance_std": eval_result.concordance_index_std,
                "ibs": eval_result.integrated_brier_score,
                "mean_auc": eval_result.mean_auc,
            }
            cv_results[model_name] = eval_result.cv_scores

            # Explain
            explain_dir = ensure_dir(outdir / "artifacts" / "explainability" / model_name)
            logger.debug(f"Generating explanations for {model_name}")
            explainer.generate_explanations(final_pipeline, X_data, model_name, explain_dir, tracker)

            # Register
            model_path = models_dir / f"{model_name}.joblib"
            logger.debug(f"Saving and registering {model_name}")
            registrar.save_and_register_model(final_pipeline, model_name, model_path, tracker)
            
            final_pipelines[model_name] = final_pipeline

            logger.info(
                f"Finished training {model_name}",
                extra={
                    "model": model_name,
                    "model_path": str(model_path),
                },
            )

    logger.info(
        "Training loop complete",
        extra={
            "n_models_trained": len(final_pipelines),
            "models": list(final_pipelines.keys()),
        },
    )

    # Persist leaderboard for downstream reporting/comparison
    try:
        import pandas as pd

        rows = []
        for model, vals in metrics.items():
            row = {"model": model}
            row.update(vals)
            rows.append(row)
        leaderboard = pd.DataFrame(rows)
        leaderboard_path = metrics_dir / "leaderboard.csv"
        leaderboard.to_csv(leaderboard_path, index=False)
    except Exception as exc:  # pragma: no cover - logging side-effect
        logger.warning(f"Failed to write leaderboard: {exc}")
        leaderboard_path = None

    return {
        "final_pipelines": final_pipelines,
        "metrics": metrics,
        "cv_results": cv_results,
        "leaderboard_path": leaderboard_path,
        "metrics_dir": metrics_dir,
    }
