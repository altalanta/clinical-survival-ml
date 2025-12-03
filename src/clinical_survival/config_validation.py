"""
Semantic configuration validation.

This module provides cross-validation of configuration settings beyond schema validation:
- Feature references exist in data
- Model names are registered/available
- Path configurations are consistent
- Hyperparameter ranges are sensible
- Pipeline steps are compatible

Usage:
    from clinical_survival.config_validation import validate_configuration
    
    issues = validate_configuration(params_config, features_config, grid_config, data_df)
    if issues.has_errors:
        print(issues.summary())
        sys.exit(1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from clinical_survival.logging_config import get_logger

# Module logger
logger = get_logger(__name__)


class IssueSeverity(Enum):
    """Severity level of a configuration issue."""
    ERROR = "error"      # Must be fixed before running
    WARNING = "warning"  # Might cause issues
    INFO = "info"        # Suggestion for improvement


@dataclass
class ConfigIssue:
    """A single configuration issue."""
    severity: IssueSeverity
    category: str
    message: str
    config_key: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Aggregate result of configuration validation."""
    issues: List[ConfigIssue] = field(default_factory=list)
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return any(i.severity == IssueSeverity.ERROR for i in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return any(i.severity == IssueSeverity.WARNING for i in self.issues)
    
    @property
    def errors(self) -> List[ConfigIssue]:
        """Get all errors."""
        return [i for i in self.issues if i.severity == IssueSeverity.ERROR]
    
    @property
    def warnings(self) -> List[ConfigIssue]:
        """Get all warnings."""
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]
    
    def add(self, issue: ConfigIssue) -> None:
        """Add an issue."""
        self.issues.append(issue)
    
    def add_error(self, category: str, message: str, **kwargs) -> None:
        """Add an error issue."""
        self.add(ConfigIssue(IssueSeverity.ERROR, category, message, **kwargs))
    
    def add_warning(self, category: str, message: str, **kwargs) -> None:
        """Add a warning issue."""
        self.add(ConfigIssue(IssueSeverity.WARNING, category, message, **kwargs))
    
    def add_info(self, category: str, message: str, **kwargs) -> None:
        """Add an info issue."""
        self.add(ConfigIssue(IssueSeverity.INFO, category, message, **kwargs))
    
    def summary(self) -> str:
        """Generate a summary string."""
        lines = []
        
        if self.errors:
            lines.append("ERRORS:")
            for e in self.errors:
                lines.append(f"  ✗ [{e.category}] {e.message}")
                if e.suggestion:
                    lines.append(f"    → {e.suggestion}")
        
        if self.warnings:
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  ⚠ [{w.category}] {w.message}")
                if w.suggestion:
                    lines.append(f"    → {w.suggestion}")
        
        infos = [i for i in self.issues if i.severity == IssueSeverity.INFO]
        if infos:
            lines.append("INFO:")
            for i in infos:
                lines.append(f"  ℹ [{i.category}] {i.message}")
        
        return "\n".join(lines)


# =============================================================================
# Validators
# =============================================================================


def _validate_feature_references(
    result: ValidationResult,
    features_config: Any,
    data_df: Optional[pd.DataFrame],
) -> None:
    """Validate that referenced features exist in the data."""
    if data_df is None:
        result.add_info(
            "Features",
            "Data not provided - skipping feature existence validation",
        )
        return
    
    data_columns = set(data_df.columns)
    
    # Check numerical columns
    for col in features_config.numerical_cols:
        if col not in data_columns:
            result.add_error(
                "Features",
                f"Numerical column '{col}' not found in data",
                config_key="features.numerical_cols",
                suggestion=f"Remove '{col}' or check column name spelling",
            )
    
    # Check categorical columns
    for col in features_config.categorical_cols:
        if col not in data_columns:
            result.add_error(
                "Features",
                f"Categorical column '{col}' not found in data",
                config_key="features.categorical_cols",
                suggestion=f"Remove '{col}' or check column name spelling",
            )
    
    # Check binary columns
    for col in features_config.binary_cols:
        if col not in data_columns:
            result.add_error(
                "Features",
                f"Binary column '{col}' not found in data",
                config_key="features.binary_cols",
            )
    
    # Check time and event columns
    if features_config.time_to_event_col not in data_columns:
        result.add_error(
            "Features",
            f"Time column '{features_config.time_to_event_col}' not found",
            config_key="features.time_to_event_col",
        )
    
    if features_config.event_col not in data_columns:
        result.add_error(
            "Features",
            f"Event column '{features_config.event_col}' not found",
            config_key="features.event_col",
        )
    
    # Check for columns in data not in config
    all_config_cols = (
        set(features_config.numerical_cols) |
        set(features_config.categorical_cols) |
        set(features_config.binary_cols) |
        {features_config.time_to_event_col, features_config.event_col}
    )
    
    unused_cols = data_columns - all_config_cols
    if unused_cols:
        result.add_info(
            "Features",
            f"Columns in data but not in config: {', '.join(sorted(unused_cols)[:5])}",
            suggestion="These columns will be ignored during training",
        )


def _validate_model_availability(
    result: ValidationResult,
    params_config: Any,
) -> None:
    """Validate that specified models are available."""
    # Known survival models
    known_survival_models = {
        "coxph", "rsf", "xgb_cox", "xgb_aft",
        "stacking", "bagging", "dynamic_ensemble",
    }
    
    # Check each model
    for model_name in params_config.models:
        if model_name not in known_survival_models:
            # Check if it's a plugin
            try:
                from clinical_survival.plugins import model_registry
                if model_name not in model_registry:
                    result.add_warning(
                        "Models",
                        f"Unknown model '{model_name}' - may be a plugin",
                        config_key="models",
                        suggestion=f"Ensure '{model_name}' is registered or use a known model",
                    )
            except ImportError:
                result.add_warning(
                    "Models",
                    f"Unknown model '{model_name}'",
                    config_key="models",
                )
    
    # Check XGBoost availability for xgb models
    xgb_models = [m for m in params_config.models if m.startswith("xgb")]
    if xgb_models:
        try:
            import xgboost
        except ImportError:
            result.add_error(
                "Models",
                f"XGBoost required for models: {xgb_models}",
                suggestion="Install xgboost: pip install xgboost",
            )


def _validate_grid_config(
    result: ValidationResult,
    params_config: Any,
    grid_config: Dict[str, Any],
) -> None:
    """Validate hyperparameter grid configuration."""
    # Check that all models have grid entries
    for model_name in params_config.models:
        if model_name not in grid_config:
            result.add_warning(
                "Grid Config",
                f"No hyperparameters defined for '{model_name}'",
                suggestion="Add default parameters or tuning space to model_grid.yaml",
            )
    
    # Check for unused grid entries
    unused_grids = set(grid_config.keys()) - set(params_config.models)
    if unused_grids:
        result.add_info(
            "Grid Config",
            f"Grid entries for unused models: {unused_grids}",
        )
    
    # Validate hyperparameter ranges
    for model_name, params in grid_config.items():
        if isinstance(params, dict):
            # Check for Optuna-style search spaces
            for param_name, param_value in params.items():
                if isinstance(param_value, dict) and "type" in param_value:
                    param_type = param_value.get("type")
                    
                    if param_type in ("int", "float"):
                        low = param_value.get("low")
                        high = param_value.get("high")
                        
                        if low is not None and high is not None:
                            if low >= high:
                                result.add_error(
                                    "Grid Config",
                                    f"{model_name}.{param_name}: low ({low}) >= high ({high})",
                                    suggestion="Fix the parameter range bounds",
                                )
                            
                            # Check for sensible ranges
                            if param_type == "int" and low < 0:
                                if param_name in ("n_estimators", "max_depth", "min_samples"):
                                    result.add_warning(
                                        "Grid Config",
                                        f"{model_name}.{param_name}: negative value in range",
                                    )


def _validate_pipeline_steps(
    result: ValidationResult,
    params_config: Any,
) -> None:
    """Validate pipeline step configuration."""
    known_steps = {
        "data_loader.load_raw_data",
        "data_validator.validate_data",
        "preprocessor.prepare_data",
        "tuner.tune_hyperparameters",
        "training_loop.run_training_loop",
        "evaluator.evaluate_predictions",
        "explainer.generate_explanations",
        "counterfactual_explainer.generate_all_counterfactuals",
        "registrar.register_model",
    }
    
    pipeline = params_config.pipeline
    
    # Check step order
    required_first = "data_loader.load_raw_data"
    if pipeline and pipeline[0] != required_first:
        result.add_error(
            "Pipeline",
            "Pipeline must start with data_loader.load_raw_data",
            suggestion=f"Add '{required_first}' as the first step",
        )
    
    # Check for unknown steps
    for step in pipeline:
        if step not in known_steps:
            result.add_warning(
                "Pipeline",
                f"Unknown pipeline step: '{step}'",
                suggestion="Verify step name or check if it's a custom step",
            )
    
    # Check dependencies
    step_set = set(pipeline)
    
    # Training loop requires preprocessing
    if "training_loop.run_training_loop" in step_set:
        if "preprocessor.prepare_data" not in step_set:
            result.add_error(
                "Pipeline",
                "training_loop requires preprocessor.prepare_data",
                suggestion="Add preprocessor.prepare_data before training_loop",
            )
    
    # Counterfactuals require training
    if "counterfactual_explainer.generate_all_counterfactuals" in step_set:
        if "training_loop.run_training_loop" not in step_set:
            result.add_error(
                "Pipeline",
                "counterfactual_explainer requires training_loop",
            )


def _validate_paths(
    result: ValidationResult,
    params_config: Any,
) -> None:
    """Validate path configuration."""
    paths = params_config.paths
    
    # Check data file path
    data_path = Path(paths.data_csv)
    if not data_path.exists():
        result.add_error(
            "Paths",
            f"Data file not found: {data_path}",
            config_key="paths.data_csv",
            suggestion="Check the path or run 'dvc pull' if using DVC",
        )
    
    # Check features config path
    features_path = Path(paths.features)
    if not features_path.exists():
        result.add_error(
            "Paths",
            f"Features config not found: {features_path}",
            config_key="paths.features",
        )
    
    # Check output directory is valid
    outdir = Path(paths.outdir)
    try:
        outdir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        result.add_error(
            "Paths",
            f"Cannot create output directory: {e}",
            config_key="paths.outdir",
        )


def _validate_tuning_config(
    result: ValidationResult,
    params_config: Any,
) -> None:
    """Validate tuning configuration."""
    tuning = params_config.tuning
    
    if not tuning.enabled:
        return
    
    # Check trial count
    if tuning.trials < 1:
        result.add_error(
            "Tuning",
            "trials must be at least 1",
            config_key="tuning.trials",
        )
    elif tuning.trials < 10:
        result.add_warning(
            "Tuning",
            f"Only {tuning.trials} trials - results may be suboptimal",
            suggestion="Consider using at least 20-50 trials",
        )
    
    # Check metric
    valid_metrics = {
        "concordance", "concordance_index_censored",
        "ibs", "integrated_brier_score",
        "brier", "brier_score",
    }
    if tuning.metric not in valid_metrics:
        result.add_warning(
            "Tuning",
            f"Unknown tuning metric: '{tuning.metric}'",
            config_key="tuning.metric",
        )
    
    # Check direction
    if tuning.direction not in ("maximize", "minimize"):
        result.add_error(
            "Tuning",
            f"Invalid direction: '{tuning.direction}'",
            config_key="tuning.direction",
            suggestion="Use 'maximize' or 'minimize'",
        )


def _validate_data_quality(
    result: ValidationResult,
    data_df: Optional[pd.DataFrame],
    features_config: Any,
) -> None:
    """Validate data quality aspects."""
    if data_df is None:
        return
    
    # Check sample size
    n_samples = len(data_df)
    if n_samples < 50:
        result.add_warning(
            "Data Quality",
            f"Very small sample size: {n_samples} rows",
            suggestion="Results may be unreliable with < 50 samples",
        )
    elif n_samples < 200:
        result.add_info(
            "Data Quality",
            f"Small sample size: {n_samples} rows",
        )
    
    # Check event rate
    if features_config.event_col in data_df.columns:
        event_rate = data_df[features_config.event_col].mean()
        if event_rate < 0.1:
            result.add_warning(
                "Data Quality",
                f"Low event rate: {event_rate:.1%}",
                suggestion="Consider collecting more events or using specialized methods",
            )
        elif event_rate > 0.9:
            result.add_warning(
                "Data Quality",
                f"Very high event rate: {event_rate:.1%}",
            )
    
    # Check missing data
    missing_pct = data_df.isnull().sum().sum() / data_df.size * 100
    if missing_pct > 20:
        result.add_warning(
            "Data Quality",
            f"High missing data: {missing_pct:.1f}%",
            suggestion="Consider data imputation or excluding high-missing columns",
        )
    elif missing_pct > 5:
        result.add_info(
            "Data Quality",
            f"Missing data: {missing_pct:.1f}%",
        )


# =============================================================================
# Main Validation Function
# =============================================================================


def validate_configuration(
    params_config: Any,
    features_config: Any,
    grid_config: Dict[str, Any],
    data_df: Optional[pd.DataFrame] = None,
) -> ValidationResult:
    """
    Perform comprehensive configuration validation.
    
    Args:
        params_config: ParamsConfig instance
        features_config: FeaturesConfig instance
        grid_config: Model hyperparameter grid dictionary
        data_df: Optional DataFrame for data-aware validation
        
    Returns:
        ValidationResult with all issues found
    """
    result = ValidationResult()
    
    # Run all validators
    _validate_paths(result, params_config)
    _validate_feature_references(result, features_config, data_df)
    _validate_model_availability(result, params_config)
    _validate_grid_config(result, params_config, grid_config)
    _validate_pipeline_steps(result, params_config)
    _validate_tuning_config(result, params_config)
    _validate_data_quality(result, data_df, features_config)
    
    # Log results
    for issue in result.issues:
        log_level = {
            IssueSeverity.ERROR: "error",
            IssueSeverity.WARNING: "warning",
            IssueSeverity.INFO: "info",
        }.get(issue.severity, "info")
        
        getattr(logger, log_level)(
            f"Config validation [{issue.category}]: {issue.message}",
            extra={"category": issue.category, "severity": issue.severity.value},
        )
    
    return result

