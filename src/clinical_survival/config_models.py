"""
Pydantic configuration models with strict validation.

This module provides Pydantic models for loading and validating YAML configuration
files with stricter type checking than the runtime config.py models.

IMPORTANT: This module re-exports and extends the runtime configuration models
from `config.py`. For most use cases, import from `config.py` directly.
This module is intended for:
- CLI configuration parsing with strict validation
- Configuration file schema documentation
- Type-safe configuration loading utilities

For runtime configuration access, use:
    from clinical_survival.config import ParamsConfig, FeaturesConfig

For strict CLI validation, use:
    from clinical_survival.config_models import (
        StrictParamsConfig,
        load_and_validate_config,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, PositiveInt, confloat, field_validator, model_validator

# Re-export runtime config classes for backwards compatibility
from clinical_survival.config import (
    CachingConfig,
    CalibrationConfig,
    ClinicalInterpretabilityConfig,
    CounterfactualsConfig,
    DataValidationConfig,
    DecisionCurveConfig,
    DistributedComputingConfig,
    EvaluationConfig,
    ExplainConfig,
    ExternalConfig,
    FeaturesConfig,
    IncrementalLearningConfig,
    LoggingConfigModel,
    MissingConfig,
    MLflowTrackingConfig,
    MLOpsConfig,
    MonitoringConfig,
    ParamsConfig,
    PathsConfig,
    PreprocessingStep,
    ResilienceConfig,
    ScoringConfig,
    TuningConfig,
)


# =============================================================================
# Strict Validation Models
# =============================================================================


class StrictPathsConfig(BaseModel):
    """
    Paths configuration with strict validation.
    
    Validates that paths exist (for inputs) or can be created (for outputs).
    """
    
    data_csv: Path = Field(..., description="Path to input CSV data file")
    metadata: Path = Field(..., description="Path to metadata YAML file")
    external_csv: Optional[Path] = Field(None, description="Path to external validation CSV")
    outdir: Path = Field(..., description="Output directory for results")
    features: Path = Field(..., description="Path to features configuration YAML")
    
    @field_validator("data_csv", "metadata", "features")
    @classmethod
    def validate_input_paths(cls, v: Path) -> Path:
        """Ensure input paths exist."""
        if not v.exists():
            raise ValueError(f"Input path does not exist: {v}")
        return v
    
    @field_validator("external_csv")
    @classmethod
    def validate_external_csv(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate external CSV if provided."""
        if v is not None and not v.exists():
            raise ValueError(f"External CSV does not exist: {v}")
        return v
    
    @field_validator("outdir", mode="before")
    @classmethod
    def ensure_outdir_is_path(cls, v: Any) -> Path:
        """Convert string to Path and ensure it can be created."""
        path = Path(v) if isinstance(v, str) else v
        # Don't validate existence - will be created
        return path


class StrictScoringConfig(BaseModel):
    """Scoring configuration with strict validation."""
    
    primary: Literal[
        "concordance_index_censored",
        "concordance_index_ipcw",
        "brier_score",
        "integrated_brier_score",
    ] = Field(..., description="Primary scoring metric")
    secondary: List[str] = Field(default_factory=list, description="Secondary metrics")


class StrictCalibrationConfig(BaseModel):
    """Calibration configuration with strict validation."""
    
    times_days: List[PositiveInt] = Field(
        ...,
        min_length=1,
        description="Time points in days for calibration",
    )
    bins: int = Field(ge=5, le=20, description="Number of calibration bins")


class StrictDecisionCurveConfig(BaseModel):
    """Decision curve configuration with strict validation."""
    
    times_days: List[PositiveInt] = Field(
        ...,
        min_length=1,
        description="Time points in days for decision curve",
    )
    thresholds: List[confloat(ge=0, le=1)] = Field(  # type: ignore
        ...,
        min_length=1,
        description="Threshold probabilities for decision curve",
    )


class StrictMissingConfig(BaseModel):
    """Missing data configuration with strict validation."""
    
    strategy: Literal["iterative", "simple"] = Field(
        ...,
        description="Imputation strategy",
    )
    max_iter: PositiveInt = Field(
        ge=1,
        le=50,
        description="Maximum iterations for iterative imputation",
    )
    initial_strategy: Literal["mean", "median", "most_frequent", "constant"] = Field(
        ...,
        description="Initial imputation strategy",
    )


class StrictModelList(BaseModel):
    """Validates model names against supported models."""
    
    __root__: List[Literal["coxph", "rsf", "xgb_cox", "xgb_aft"]]


class StrictFeaturesConfig(BaseModel):
    """
    Features configuration with strict validation.
    
    Ensures preprocessing pipeline steps are valid.
    """
    
    numerical_cols: List[str] = Field(
        default_factory=list,
        description="Numerical feature columns",
    )
    categorical_cols: List[str] = Field(
        default_factory=list,
        description="Categorical feature columns",
    )
    binary_cols: List[str] = Field(
        default_factory=list,
        description="Binary feature columns",
    )
    time_to_event_col: str = Field(..., description="Time-to-event column name")
    event_col: str = Field(..., description="Event indicator column name")
    preprocessing_pipeline: List[PreprocessingStep] = Field(
        default_factory=list,
        description="Preprocessing pipeline steps",
    )
    
    @model_validator(mode="after")
    def validate_has_features(self) -> "StrictFeaturesConfig":
        """Ensure at least some features are defined."""
        total_features = (
            len(self.numerical_cols) +
            len(self.categorical_cols) +
            len(self.binary_cols)
        )
        if total_features == 0:
            raise ValueError("At least one feature column must be defined")
        return self


class StrictParamsConfig(BaseModel):
    """
    Parameters configuration with strict validation.
    
    This model provides stricter validation than ParamsConfig,
    suitable for configuration file loading and CLI validation.
    """
    
    seed: PositiveInt = Field(ge=0, description="Random seed for reproducibility")
    n_splits: int = Field(ge=2, le=10, description="Number of CV folds")
    inner_splits: int = Field(ge=2, le=5, description="Number of inner CV folds")
    test_split: confloat(ge=0.1, le=0.5) = Field(  # type: ignore
        ...,
        description="Test set proportion",
    )
    time_col: str = Field(alias="time_column", description="Time column name")
    event_col: str = Field(alias="event_column", description="Event column name")
    id_col: Optional[str] = Field(None, description="Optional ID column")
    
    scoring: StrictScoringConfig = Field(
        default_factory=lambda: StrictScoringConfig(primary="concordance_index_censored")
    )
    calibration: StrictCalibrationConfig
    decision_curve: StrictDecisionCurveConfig = Field(alias="decision_curve")
    missing: StrictMissingConfig
    models: List[Literal["coxph", "rsf", "xgb_cox", "xgb_aft"]]
    paths: StrictPathsConfig
    evaluation: EvaluationConfig
    external: ExternalConfig
    explain: ExplainConfig
    
    # Optional complex configurations
    monitoring: Optional[MonitoringConfig] = None
    incremental_learning: Optional[IncrementalLearningConfig] = Field(
        None, alias="incremental_learning"
    )
    distributed_computing: Optional[DistributedComputingConfig] = Field(
        None, alias="distributed_computing"
    )
    clinical_interpretability: Optional[ClinicalInterpretabilityConfig] = Field(
        None, alias="clinical_interpretability"
    )
    mlops: Optional[MLOpsConfig] = None
    mlflow_tracking: Optional[MLflowTrackingConfig] = Field(
        None, alias="mlflow_tracking"
    )
    caching: Optional[CachingConfig] = None
    data_validation: Optional[DataValidationConfig] = None
    tuning: Optional[TuningConfig] = None
    counterfactuals: Optional[CounterfactualsConfig] = None
    pipeline: List[str] = Field(default_factory=list)
    logging: LoggingConfigModel = Field(default_factory=LoggingConfigModel)
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)
    
    class Config:
        populate_by_name = True
    
    @model_validator(mode="after")
    def validate_models_not_empty(self) -> "StrictParamsConfig":
        """Ensure at least one model is specified."""
        if not self.models:
            raise ValueError("At least one model must be specified")
        return self


# =============================================================================
# Configuration Loading Utilities
# =============================================================================


def load_and_validate_config(
    params_path: Union[str, Path],
    features_path: Optional[Union[str, Path]] = None,
    strict: bool = False,
) -> Tuple[ParamsConfig, Optional[FeaturesConfig]]:
    """
    Load and validate configuration files.
    
    Args:
        params_path: Path to parameters YAML file
        features_path: Optional path to features YAML file
        strict: If True, use strict validation models
        
    Returns:
        Tuple of (params_config, features_config or None)
        
    Raises:
        ValueError: If validation fails
        FileNotFoundError: If config files don't exist
    """
    import yaml
    
    params_path = Path(params_path)
    if not params_path.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_path}")
    
    with params_path.open() as f:
        params_dict = yaml.safe_load(f)
    
    # Choose model based on strict flag
    config_class = StrictParamsConfig if strict else ParamsConfig
    params_config = config_class.model_validate(params_dict)
    
    # Load features config if path provided
    features_config = None
    if features_path:
        features_path = Path(features_path)
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        with features_path.open() as f:
            features_dict = yaml.safe_load(f)
        
        features_class = StrictFeaturesConfig if strict else FeaturesConfig
        features_config = features_class.model_validate(features_dict)
    
    # If not strict, return ParamsConfig instances
    if strict:
        # Convert strict models back to runtime models
        params_config = ParamsConfig.model_validate(params_config.model_dump(by_alias=True))
        if features_config:
            features_config = FeaturesConfig.model_validate(features_config.model_dump())
    
    return params_config, features_config


def validate_config_dict(config_dict: Dict[str, Any], strict: bool = False) -> ParamsConfig:
    """
    Validate a configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary
        strict: If True, use strict validation
        
    Returns:
        Validated ParamsConfig instance
    """
    config_class = StrictParamsConfig if strict else ParamsConfig
    return config_class.model_validate(config_dict)


def generate_config_schema() -> Dict[str, Any]:
    """
    Generate JSON schema for configuration files.
    
    Returns:
        JSON schema dictionary for StrictParamsConfig
    """
    return StrictParamsConfig.model_json_schema()


# =============================================================================
# Model Grid Configuration
# =============================================================================


class HyperparameterSpace(BaseModel):
    """Defines a hyperparameter search space."""
    
    type: Literal["int", "float", "categorical"] = Field(
        ...,
        description="Parameter type",
    )
    low: Optional[float] = Field(None, description="Lower bound for int/float")
    high: Optional[float] = Field(None, description="Upper bound for int/float")
    choices: Optional[List[Any]] = Field(None, description="Choices for categorical")
    log: bool = Field(False, description="Use log scale for sampling")
    
    @model_validator(mode="after")
    def validate_bounds(self) -> "HyperparameterSpace":
        """Validate that bounds are provided for numeric types."""
        if self.type in ("int", "float"):
            if self.low is None or self.high is None:
                raise ValueError(f"'low' and 'high' required for type '{self.type}'")
            if self.low >= self.high:
                raise ValueError("'low' must be less than 'high'")
        elif self.type == "categorical":
            if not self.choices:
                raise ValueError("'choices' required for categorical type")
        return self


class ModelGridConfig(BaseModel):
    """
    Model hyperparameter grid configuration.
    
    Maps model names to their hyperparameter search spaces.
    """
    
    __root__: Dict[str, Dict[str, Union[HyperparameterSpace, Any]]] = Field(
        default_factory=dict
    )
    
    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """Get hyperparameters for a specific model."""
        return self.__root__.get(model_name, {})


# =============================================================================
# Deprecation Notices
# =============================================================================


# Keep these for backwards compatibility but mark as deprecated
__all__ = [
    # Re-exported from config.py (primary imports)
    "ParamsConfig",
    "FeaturesConfig",
    "PathsConfig",
    "ScoringConfig",
    "CalibrationConfig",
    "DecisionCurveConfig",
    "MissingConfig",
    "EvaluationConfig",
    "ExternalConfig",
    "ExplainConfig",
    # Strict validation models
    "StrictParamsConfig",
    "StrictFeaturesConfig",
    "StrictPathsConfig",
    "StrictScoringConfig",
    "StrictCalibrationConfig",
    "StrictDecisionCurveConfig",
    "StrictMissingConfig",
    # Utilities
    "load_and_validate_config",
    "validate_config_dict",
    "generate_config_schema",
    # Grid configuration
    "ModelGridConfig",
    "HyperparameterSpace",
]
