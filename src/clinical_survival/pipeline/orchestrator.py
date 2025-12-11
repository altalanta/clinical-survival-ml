"""
Pipeline orchestrator with structured logging, error handling, and profiling.

This module orchestrates the execution of the modular training pipeline,
using the centralized logging system, unified error handling, and
performance profiling for consistent, traceable, and user-friendly output.

Key features:
- Pre-flight health checks and diagnostics
- Semantic configuration validation
- Artifact manifest generation for reproducibility
- Model comparison and selection
- Performance profiling
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from joblib import Memory

from clinical_survival.artifact_manifest import ManifestManager
from clinical_survival.config import ParamsConfig, FeaturesConfig
from clinical_survival.config_validation import validate_configuration
from clinical_survival.diagnostics import run_health_checks
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
from clinical_survival.model_selection import ModelComparator
from clinical_survival.profiling import PipelineProfiler
from clinical_survival.tracking import MLflowTracker
from clinical_survival.utils import set_global_seed, ensure_dir
from clinical_survival.checkpoint import create_checkpoint_manager, CheckpointManager

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
    profiler: Optional[PipelineProfiler] = None,
) -> Optional[Dict[str, Any]]:
    """
    Execute a single pipeline step with error handling and profiling.
    
    Args:
        step: Step identifier
        func: Step function to execute
        context: Pipeline context dictionary
        pipeline_logger: Logger for step tracking
        profiler: Optional profiler for performance tracking
        
    Returns:
        Step output dictionary, or None
        
    Raises:
        StepExecutionError: If the step fails
    """
    with pipeline_logger.step(step):
        # Use profiler if available, otherwise just execute
        profile_context = profiler.profile_step(step) if profiler else _null_context()
        
        with profile_context:
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
    skip_health_checks: bool = False,
    skip_validation: bool = False,
    enable_checkpoints: bool = True,
    resume: bool = False,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Orchestrates the execution of the modular training pipeline.
    
    This function:
    1. Runs pre-flight health checks and diagnostics
    2. Validates configuration semantically
    3. Initializes the logging system and manifest manager
    4. Sets up MLflow tracking
    5. Executes each pipeline step in sequence with error handling
    6. Compares models and selects the best
    7. Generates artifact manifest for reproducibility
    
    Args:
        params_config: Main parameters configuration
        features_config: Feature engineering configuration
        grid_config: Model hyperparameter grid configuration
        skip_health_checks: Skip pre-flight health checks
        skip_validation: Skip configuration validation
        
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
    
    # 2. Run pre-flight health checks
    if not skip_health_checks:
        logger.info("Running pre-flight health checks...")
        health_result = run_health_checks(params_config, verbose=True)
        
        if not health_result.all_passed:
            raise ConfigurationError(
                "Health checks failed. Fix issues before running pipeline.",
                config_file="environment",
                key="health_checks",
            )
        
        if health_result.has_warnings:
            logger.warning(
                "Health checks passed with warnings",
                extra={"warnings": len(health_result.warning_checks)},
            )
    
    # 3. Validate configuration semantically
    if not skip_validation:
        logger.info("Validating configuration...")
        validation_result = validate_configuration(
            params_config, features_config, grid_config
        )
        
        if validation_result.has_errors:
            logger.error(
                "Configuration validation failed",
                extra={"errors": len(validation_result.errors)},
            )
            raise ConfigurationError(
                f"Configuration validation failed:\n{validation_result.summary()}",
                config_file="params.yaml",
            )
        
        if validation_result.has_warnings:
            logger.warning(
                "Configuration has warnings",
                extra={"warnings": len(validation_result.warnings)},
            )

    # 4. Setup
    set_global_seed(params_config.seed)
    outdir = ensure_dir(params_config.paths.outdir)

    checkpoint_manager: Optional[CheckpointManager] = None
    completed_steps: List[str] = []
    if enable_checkpoints:
        if resume:
            checkpoint_manager = CheckpointManager.get_resumable_run(outdir / "checkpoints")
            if checkpoint_manager:
                completed_steps = checkpoint_manager.get_completed_steps()
                logger.info(
                    "Found resumable run",
                    extra={"run_id": checkpoint_manager.run_id, "completed_steps": completed_steps},
                )
        if checkpoint_manager is None:
            checkpoint_manager = create_checkpoint_manager(outdir, run_id=run_id)
            checkpoint_manager.start_run(
                pipeline_steps=params_config.pipeline,
                correlation_id=correlation_id,
            )
        elif resume:
            # Ensure correlation id recorded for resumed runs
            checkpoint_manager._state.correlation_id = correlation_id
            checkpoint_manager._save_state()

    # Initialize manifest manager for artifact tracking
    manifest_manager = ManifestManager(outdir, run_name=f"pipeline_{correlation_id[:8]}")
    manifest_manager.start_run(
        params_config=params_config,
        features_config=features_config,
        grid_config=grid_config,
        correlation_id=correlation_id,
    )
    
    try:
        tracker = MLflowTracker(params_config.mlflow_tracking.model_dump())
    except Exception as e:
        logger.warning(
            f"Failed to initialize MLflow tracker: {e}. Continuing without tracking.",
            extra={"error": str(e)},
        )
        tracker = None

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
        "manifest_manager": manifest_manager,
    }

    if checkpoint_manager and completed_steps:
        latest_context = checkpoint_manager.load_latest_checkpoint() or {}
        # Preserve critical objects from current process
        latest_context.update(
            {
                "params_config": params_config,
                "features_config": features_config,
                "grid_config": grid_config,
                "tracker": tracker,
                "outdir": outdir,
                "correlation_id": correlation_id,
                "manifest_manager": manifest_manager,
            }
        )
        context = latest_context

    # 5. Load all step functions first (fail fast on missing steps)
    step_functions: Dict[str, callable] = {}
    for step in params_config.pipeline:
        step_functions[step] = _load_step_function(step)
    
    logger.debug(
        "All pipeline steps loaded successfully",
        extra={"n_steps": len(step_functions)},
    )

    # 6. Initialize profiler for performance tracking
    profiler = PipelineProfiler(
        pipeline_name="training_pipeline",
        correlation_id=correlation_id,
        track_memory=True,
    )

    # 7. Execute pipeline steps
    success = True
    failed_step: Optional[str] = None
    
    try:
        # Use MLflow tracking if available
        tracker_context = tracker.start_run("main_run") if tracker else _null_context()
        
        with tracker_context:
            for step in params_config.pipeline:
                if step in completed_steps:
                    logger.info(
                        f"Skipping previously completed step: {step}",
                        extra={"step": step},
                    )
                    continue

                func = step_functions[step]
                
                # Execute step and update context (with profiling)
                result = _execute_step(step, func, context, pipeline_logger, profiler)
                if result:
                    context.update(result)
                    
                    # Record step duration in manifest
                    if step in profiler._profile.step_durations:
                        manifest_manager.add_step_duration(
                            step, 
                            profiler._profile.step_durations.get(step, 0)
                        )

                if checkpoint_manager:
                    checkpoint_manager.save_checkpoint(step, context)
                    checkpoint_manager.mark_step_completed(step)
        
        # 8. Model comparison and selection (if models were trained)
        if "final_pipelines" in context and context["final_pipelines"]:
            logger.info("Comparing trained models...")
            
            # Collect metrics from context (you may need to adapt this based on actual structure)
            metrics = context.get("metrics", {})
            cv_results = context.get("cv_results", {})
            
            if metrics:
                comparator = ModelComparator(metrics, cv_results)
                comparison = comparator.compare(primary_metric="concordance")
                comparator.print_comparison()
                
                best_selection = comparator.select_best()
                context["best_model"] = best_selection.selected_model
                context["model_comparison"] = comparison
                
                # Persist comparison artifacts
                metrics_dir = context.get("metrics_dir") or (outdir / "metrics")
                metrics_dir = ensure_dir(metrics_dir)
                import json
                comparison_path = metrics_dir / "model_comparison.json"
                with comparison_path.open("w", encoding="utf-8") as f:
                    json.dump(comparison, f, indent=2, default=str)

                best_model_path = metrics_dir / "best_model.json"
                with best_model_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "best_model": best_selection.selected_model,
                            "criterion": best_selection.selection_criterion.value,
                        },
                        f,
                        indent=2,
                    )

                manifest_manager.set_best_model(
                    best_selection.selected_model,
                    metrics.get(best_selection.selected_model, {}),
                )
                
                logger.info(
                    f"Best model selected: {best_selection.selected_model}",
                    extra={
                        "best_model": best_selection.selected_model,
                        "criterion": best_selection.selection_criterion.value,
                    },
                )

    except ClinicalSurvivalError as e:
        success = False
        failed_step = getattr(e, "step_name", None) or str(e.context.get("step", "unknown"))
        manifest_manager.add_note(f"Pipeline failed at step: {failed_step}")
        if checkpoint_manager:
            checkpoint_manager.mark_run_failed(str(e))
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
        manifest_manager.add_note(f"Pipeline failed with unexpected error: {e}")
        if checkpoint_manager:
            checkpoint_manager.mark_run_failed(str(e))
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
        if checkpoint_manager and success:
            checkpoint_manager.mark_run_completed()

        pipeline_logger.finish(
            success=success,
            models_trained=len(params_config.models),
            failed_step=failed_step,
        )
        
        # Finish profiling and save report
        profile = profiler.finish()
        
        # Save profiling report
        profile_path = outdir / "artifacts" / "pipeline_profile.json"
        profiler.save_report(profile_path)
        
        # Add profile to manifest
        manifest_manager.add_artifact(
            "pipeline_profile",
            profile_path,
            category="profiling",
            metadata={"total_duration": profile.total_duration_seconds},
        )

        # Add metrics artifacts to manifest
        leaderboard_path = context.get("leaderboard_path")
        if leaderboard_path:
            manifest_manager.add_artifact(
                "leaderboard",
                leaderboard_path,
                category="metrics",
            )

        metrics_dir = context.get("metrics_dir") or (outdir / "metrics")
        comparison_path = metrics_dir / "model_comparison.json"
        best_model_path = metrics_dir / "best_model.json"
        for name, path in [
            ("model_comparison", comparison_path),
            ("best_model_info", best_model_path),
        ]:
            if path.exists():
                manifest_manager.add_artifact(name, path, category="metrics")
        
        # Finalize and save manifest
        manifest_manager.finalize(success=success)
        manifest_path = manifest_manager.save()
        
        logger.info(f"Manifest saved to {manifest_path}")
        
        # Print summary if verbose logging is enabled
        if params_config.logging.level.upper() in ("DEBUG", "INFO"):
            profiler.print_summary()
    
    logger.info(
        "Pipeline execution completed successfully",
        extra={
            "correlation_id": correlation_id,
            "n_context_keys": len(context),
            "total_duration_seconds": profile.total_duration_seconds,
            "peak_memory_mb": profile.peak_memory_mb,
        },
    )
    
    return context


class _null_context:
    """Null context manager for when MLflow is not available."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
