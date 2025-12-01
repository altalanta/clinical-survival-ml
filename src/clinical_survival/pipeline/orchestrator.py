"""
Pipeline orchestrator with structured logging and error handling.

This module orchestrates the execution of the modular training pipeline,
using the centralized logging system and unified error handling for
consistent, traceable, and user-friendly output.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

from joblib import Memory

from clinical_survival.config import ParamsConfig, FeaturesConfig
from clinical_survival.error_handling import wrap_step_errors
from clinical_survival.errors import (
    ClinicalSurvivalError,
    ConfigurationError,
    PipelineError,
    StepExecutionError,
    StepNotFoundError,
)
from clinical_survival.logging_config import (
    configure_logging,
    get_logger,
    PipelineLogger,
    set_correlation_id,
)
from clinical_survival.tracking import MLflowTracker
from clinical_survival.utils import set_global_seed, ensure_dir

# Get module logger
logger = get_logger(__name__)


def _get_available_steps() -> List[str]:
    """Get list of available pipeline step modules."""
    # This could be made dynamic by scanning the pipeline directory
    return [
        "data_loader.load_raw_data",
        "data_validator.validate_data",
        "preprocessor.prepare_data",
        "tuner.tune_hyperparameters",
        "training_loop.run_training_loop",
        "evaluator.evaluate_predictions",
        "explainer.generate_explanations",
        "counterfactual_explainer.generate_all_counterfactuals",
        "registrar.register_model",
    ]


def _load_step_function(step: str) -> callable:
    """
    Load a pipeline step function by its module.function path.
    
    Args:
        step: Step identifier in format "module.function"
        
    Returns:
        The callable step function
        
    Raises:
        StepNotFoundError: If the step cannot be loaded
    """
    try:
        module_name, func_name = step.rsplit(".", 1)
    except ValueError:
        raise StepNotFoundError(
            step_name=step,
            available_steps=_get_available_steps(),
        )
    
    try:
        module = importlib.import_module(f"clinical_survival.pipeline.{module_name}")
    except ImportError as e:
        raise StepNotFoundError(
            step_name=step,
            available_steps=_get_available_steps(),
        ) from e
    
    try:
        func = getattr(module, func_name)
    except AttributeError as e:
        raise StepNotFoundError(
            step_name=step,
            available_steps=_get_available_steps(),
        ) from e
    
    return func


def _execute_step(
    step: str,
    func: callable,
    context: Dict[str, Any],
    pipeline_logger: PipelineLogger,
) -> Optional[Dict[str, Any]]:
    """
    Execute a single pipeline step with error handling.
    
    Args:
        step: Step identifier
        func: Step function to execute
        context: Pipeline context dictionary
        pipeline_logger: Logger for step tracking
        
    Returns:
        Step output dictionary, or None
        
    Raises:
        StepExecutionError: If the step fails
    """
    with pipeline_logger.step(step):
        try:
            result = func(**context)
            
            if result:
                logger.debug(
                    f"Step '{step}' completed successfully",
                    extra={"output_keys": list(result.keys())},
                )
            
            return result
            
        except ClinicalSurvivalError:
            # Re-raise our custom exceptions as-is
            raise
        except Exception as e:
            # Wrap other exceptions with context
            raise StepExecutionError(
                message=f"Step '{step}' failed: {e}",
                step_name=step,
                original_error=e,
            ) from e


def run_pipeline(
    params_config: ParamsConfig,
    features_config: FeaturesConfig,
    grid_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Orchestrates the execution of the modular training pipeline.
    
    This function:
    1. Initializes the logging system with configuration from params
    2. Sets up MLflow tracking
    3. Executes each pipeline step in sequence with error handling
    4. Logs detailed timing and status information for each step
    5. Provides user-friendly error messages on failure
    
    Args:
        params_config: Main parameters configuration
        features_config: Feature engineering configuration
        grid_config: Model hyperparameter grid configuration
        
    Returns:
        Final pipeline context dictionary
        
    Raises:
        ConfigurationError: If configuration is invalid
        StepNotFoundError: If a pipeline step cannot be found
        StepExecutionError: If a pipeline step fails
        PipelineError: For other pipeline-related errors
    """
    # Validate configuration
    if not params_config.pipeline:
        raise ConfigurationError(
            "Pipeline configuration is empty",
            config_file="params.yaml",
            key="pipeline",
        )
    
    # 1. Configure logging based on params
    configure_logging(
        level=params_config.logging.level,
        format=params_config.logging.format,
        log_file=params_config.logging.log_file,
    )
    
    # Generate and set correlation ID for this pipeline run
    correlation_id = set_correlation_id()
    
    # Create pipeline logger for step tracking
    pipeline_logger = PipelineLogger("training_pipeline")
    
    logger.info(
        "Starting pipeline execution",
        extra={
            "correlation_id": correlation_id,
            "seed": params_config.seed,
            "models": params_config.models,
            "n_pipeline_steps": len(params_config.pipeline),
            "steps": params_config.pipeline,
        },
    )

    # 2. Setup
    set_global_seed(params_config.seed)
    
    try:
        tracker = MLflowTracker(params_config.mlflow_tracking.model_dump())
    except Exception as e:
        logger.warning(
            f"Failed to initialize MLflow tracker: {e}. Continuing without tracking.",
            extra={"error": str(e)},
        )
        tracker = None

    outdir = ensure_dir(params_config.paths.outdir)

    _memory = None
    if params_config.caching.enabled:
        cache_dir = ensure_dir(params_config.caching.dir)
        _memory = Memory(cache_dir, verbose=0)
        logger.debug(f"Caching enabled at {cache_dir}")

    # Pipeline context to pass data between steps
    context: Dict[str, Any] = {
        "params_config": params_config,
        "features_config": features_config,
        "grid_config": grid_config,
        "tracker": tracker,
        "outdir": outdir,
        "correlation_id": correlation_id,
    }

    # 3. Load all step functions first (fail fast on missing steps)
    step_functions: Dict[str, callable] = {}
    for step in params_config.pipeline:
        step_functions[step] = _load_step_function(step)
    
    logger.debug(
        "All pipeline steps loaded successfully",
        extra={"n_steps": len(step_functions)},
    )

    # 4. Execute pipeline steps
    success = True
    failed_step: Optional[str] = None
    
    try:
        # Use MLflow tracking if available
        tracker_context = tracker.start_run("main_run") if tracker else _null_context()
        
        with tracker_context:
            for step in params_config.pipeline:
                func = step_functions[step]
                
                # Execute step and update context
                result = _execute_step(step, func, context, pipeline_logger)
                if result:
                    context.update(result)

    except ClinicalSurvivalError as e:
        success = False
        failed_step = getattr(e, "step_name", None) or str(e.context.get("step", "unknown"))
        logger.error(
            "Pipeline execution failed",
            extra={
                "error_type": type(e).__name__,
                "failed_step": failed_step,
                "correlation_id": correlation_id,
            },
        )
        raise
    
    except Exception as e:
        success = False
        logger.error(
            "Pipeline execution failed with unexpected error",
            extra={
                "error_type": type(e).__name__,
                "correlation_id": correlation_id,
            },
            exc_info=True,
        )
        raise PipelineError(
            f"Pipeline failed with unexpected error: {e}",
            context={"correlation_id": correlation_id},
        ) from e
    
    finally:
        pipeline_logger.finish(
            success=success,
            models_trained=len(params_config.models),
            failed_step=failed_step,
        )
    
    logger.info(
        "Pipeline execution completed successfully",
        extra={
            "correlation_id": correlation_id,
            "n_context_keys": len(context),
        },
    )
    
    return context


class _null_context:
    """Null context manager for when MLflow is not available."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
