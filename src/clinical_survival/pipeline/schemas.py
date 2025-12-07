"""
Pipeline step input/output schema definitions and validation.

This module provides:
- Pydantic models defining expected inputs and outputs for each pipeline step
- Validation decorator that checks schemas before/after step execution
- Helpful error messages when validation fails
- Auto-generated documentation from schemas

Usage:
    from clinical_survival.pipeline.schemas import (
        validate_pipeline_step,
        DataLoaderInput,
        DataLoaderOutput,
    )
    
    @validate_pipeline_step(
        input_schema=DataLoaderInput,
        output_schema=DataLoaderOutput,
    )
    def load_raw_data(**context) -> dict:
        ...
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from clinical_survival.config import FeaturesConfig, ParamsConfig
from clinical_survival.errors import SchemaValidationError
from clinical_survival.logging_config import get_logger

# Type variable for decorators
F = TypeVar("F", bound=Callable[..., Any])

# Module logger
logger = get_logger(__name__)

# Console for rich output
console = Console(stderr=True)


# =============================================================================
# Base Schema Classes
# =============================================================================


class PipelineStepInput(BaseModel):
    """
    Base class for pipeline step input schemas.
    
    All input schemas should inherit from this class.
    Allows extra fields by default since context may contain
    additional data from previous steps.
    """
    
    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )


class PipelineStepOutput(BaseModel):
    """
    Base class for pipeline step output schemas.
    
    All output schemas should inherit from this class.
    """
    
    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )


# =============================================================================
# Data Loader Schemas
# =============================================================================


class DataLoaderInput(PipelineStepInput):
    """Input schema for the data loading step."""
    
    params_config: Any = Field(
        ...,
        description="Main parameters configuration object",
    )
    
    @field_validator("params_config")
    @classmethod
    def validate_params_config(cls, v: Any) -> Any:
        if not isinstance(v, ParamsConfig):
            raise ValueError(f"params_config must be ParamsConfig, got {type(v).__name__}")
        return v


class DataLoaderOutput(PipelineStepOutput):
    """Output schema for the data loading step."""
    
    raw_df: Any = Field(
        ...,
        description="Raw DataFrame loaded from the data source",
    )
    
    @field_validator("raw_df")
    @classmethod
    def validate_raw_df(cls, v: Any) -> Any:
        if not isinstance(v, pd.DataFrame):
            raise ValueError(f"raw_df must be a DataFrame, got {type(v).__name__}")
        if len(v) == 0:
            raise ValueError("raw_df cannot be empty")
        return v


# =============================================================================
# Data Validator Schemas
# =============================================================================


class DataValidatorInput(PipelineStepInput):
    """Input schema for the data validation step."""
    
    raw_df: Any = Field(
        ...,
        description="Raw DataFrame to validate",
    )
    params_config: Any = Field(
        ...,
        description="Main parameters configuration",
    )
    features_config: Any = Field(
        ...,
        description="Features configuration",
    )
    
    @field_validator("raw_df")
    @classmethod
    def validate_raw_df(cls, v: Any) -> Any:
        if not isinstance(v, pd.DataFrame):
            raise ValueError(f"raw_df must be a DataFrame, got {type(v).__name__}")
        return v


class DataValidatorOutput(PipelineStepOutput):
    """Output schema for the data validation step (no required outputs)."""
    pass


# =============================================================================
# Preprocessor Schemas
# =============================================================================


class PreprocessorInput(PipelineStepInput):
    """Input schema for the preprocessing step."""
    
    raw_df: Any = Field(
        ...,
        description="Raw DataFrame to preprocess",
    )
    params_config: Any = Field(
        ...,
        description="Main parameters configuration",
    )
    features_config: Any = Field(
        ...,
        description="Features configuration with preprocessing pipeline",
    )
    
    @field_validator("raw_df")
    @classmethod
    def validate_raw_df(cls, v: Any) -> Any:
        if not isinstance(v, pd.DataFrame):
            raise ValueError(f"raw_df must be a DataFrame, got {type(v).__name__}")
        return v
    
    @field_validator("features_config")
    @classmethod
    def validate_features_config(cls, v: Any) -> Any:
        if not isinstance(v, FeaturesConfig):
            raise ValueError(f"features_config must be FeaturesConfig, got {type(v).__name__}")
        return v


class PreprocessorOutput(PipelineStepOutput):
    """Output schema for the preprocessing step."""
    
    X_train: Any = Field(
        ...,
        description="Preprocessed training features",
    )
    X_test: Any = Field(
        ...,
        description="Preprocessed test features",
    )
    y_train: Any = Field(
        ...,
        description="Training survival target",
    )
    y_test: Any = Field(
        ...,
        description="Test survival target",
    )
    preprocessor: Any = Field(
        ...,
        description="Fitted preprocessor pipeline",
    )
    
    @field_validator("X_train", "X_test")
    @classmethod
    def validate_features(cls, v: Any) -> Any:
        if not isinstance(v, (pd.DataFrame, np.ndarray)):
            raise ValueError(f"Features must be DataFrame or ndarray, got {type(v).__name__}")
        return v
    
    @model_validator(mode="after")
    def validate_shapes(self) -> "PreprocessorOutput":
        if hasattr(self.X_train, "__len__") and hasattr(self.y_train, "__len__"):
            if len(self.X_train) != len(self.y_train):
                raise ValueError(
                    f"X_train and y_train must have same length: "
                    f"{len(self.X_train)} vs {len(self.y_train)}"
                )
        if hasattr(self.X_test, "__len__") and hasattr(self.y_test, "__len__"):
            if len(self.X_test) != len(self.y_test):
                raise ValueError(
                    f"X_test and y_test must have same length: "
                    f"{len(self.X_test)} vs {len(self.y_test)}"
                )
        return self


# =============================================================================
# Tuner Schemas
# =============================================================================


class TunerInput(PipelineStepInput):
    """Input schema for the hyperparameter tuning step."""
    
    X_train: Any = Field(
        ...,
        description="Training features (from preprocessor, aliased as X)",
    )
    y_train: Any = Field(
        ...,
        description="Training target (from preprocessor, aliased as y_surv)",
    )
    params_config: Any = Field(
        ...,
        description="Main parameters configuration",
    )
    features_config: Any = Field(
        ...,
        description="Features configuration",
    )
    grid_config: Any = Field(
        ...,
        description="Hyperparameter grid configuration",
    )
    
    # Allow aliases for compatibility
    X: Optional[Any] = Field(None, description="Alias for X_train")
    y_surv: Optional[Any] = Field(None, description="Alias for y_train")
    
    @model_validator(mode="after")
    def handle_aliases(self) -> "TunerInput":
        # Handle X/X_train alias
        if self.X is not None and self.X_train is None:
            object.__setattr__(self, "X_train", self.X)
        # Handle y_surv/y_train alias
        if self.y_surv is not None and self.y_train is None:
            object.__setattr__(self, "y_train", self.y_surv)
        return self


class TunerOutput(PipelineStepOutput):
    """Output schema for the hyperparameter tuning step."""
    
    best_params: Dict[str, Any] = Field(
        ...,
        description="Best hyperparameters for each model",
    )


# =============================================================================
# Training Loop Schemas
# =============================================================================


class TrainingLoopInput(PipelineStepInput):
    """Input schema for the training loop step."""
    
    X_train: Any = Field(
        ...,
        description="Training features",
    )
    y_train: Any = Field(
        ...,
        description="Training target",
    )
    params_config: Any = Field(
        ...,
        description="Main parameters configuration",
    )
    features_config: Any = Field(
        ...,
        description="Features configuration",
    )
    best_params: Dict[str, Any] = Field(
        ...,
        description="Best hyperparameters from tuning",
    )
    tracker: Any = Field(
        ...,
        description="MLflow tracker instance",
    )
    outdir: Any = Field(
        ...,
        description="Output directory path",
    )
    
    # Allow aliases
    X: Optional[Any] = Field(None)
    y_surv: Optional[Any] = Field(None)
    
    @model_validator(mode="after")
    def handle_aliases(self) -> "TrainingLoopInput":
        if self.X is not None and self.X_train is None:
            object.__setattr__(self, "X_train", self.X)
        if self.y_surv is not None and self.y_train is None:
            object.__setattr__(self, "y_train", self.y_surv)
        return self


class TrainingLoopOutput(PipelineStepOutput):
    """Output schema for the training loop step."""
    
    final_pipelines: Dict[str, Any] = Field(
        ...,
        description="Dictionary of trained model pipelines",
    )
    
    @field_validator("final_pipelines")
    @classmethod
    def validate_pipelines(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            raise ValueError(f"final_pipelines must be a dict, got {type(v).__name__}")
        if len(v) == 0:
            raise ValueError("final_pipelines cannot be empty")
        return v


# =============================================================================
# Counterfactual Explainer Schemas
# =============================================================================


class CounterfactualInput(PipelineStepInput):
    """Input schema for the counterfactual explanation step."""
    
    final_pipelines: Dict[str, Any] = Field(
        ...,
        description="Trained model pipelines",
    )
    X_train: Any = Field(
        ...,
        description="Training features for background data",
    )
    params_config: Any = Field(
        ...,
        description="Main parameters configuration",
    )
    features_config: Any = Field(
        ...,
        description="Features configuration",
    )
    outdir: Any = Field(
        ...,
        description="Output directory path",
    )
    
    # Aliases
    X: Optional[Any] = Field(None)


class CounterfactualOutput(PipelineStepOutput):
    """Output schema for counterfactual explanations (optional outputs)."""
    
    counterfactual_results: Optional[Dict[str, Any]] = Field(
        None,
        description="Counterfactual explanation results",
    )


# =============================================================================
# Schema Registry
# =============================================================================


# Map step names to their schemas
STEP_SCHEMAS: Dict[str, Dict[str, Type[BaseModel]]] = {
    "data_loader.load_raw_data": {
        "input": DataLoaderInput,
        "output": DataLoaderOutput,
    },
    "data_validator.validate_data": {
        "input": DataValidatorInput,
        "output": DataValidatorOutput,
    },
    "preprocessor.prepare_data": {
        "input": PreprocessorInput,
        "output": PreprocessorOutput,
    },
    "tuner.tune_hyperparameters": {
        "input": TunerInput,
        "output": TunerOutput,
    },
    "training_loop.run_training_loop": {
        "input": TrainingLoopInput,
        "output": TrainingLoopOutput,
    },
    "counterfactual_explainer.generate_all_counterfactuals": {
        "input": CounterfactualInput,
        "output": CounterfactualOutput,
    },
}


def get_step_schemas(step_name: str) -> Optional[Dict[str, Type[BaseModel]]]:
    """Get the input/output schemas for a pipeline step."""
    return STEP_SCHEMAS.get(step_name)


def list_validated_steps() -> List[str]:
    """List all pipeline steps that have schema validation."""
    return list(STEP_SCHEMAS.keys())


# =============================================================================
# Validation Error Formatting
# =============================================================================


def format_validation_errors(
    step_name: str,
    schema_type: str,  # "input" or "output"
    errors: List[Dict[str, Any]],
) -> Panel:
    """
    Format validation errors into a user-friendly Rich panel.
    
    Args:
        step_name: Name of the pipeline step
        schema_type: Whether this is input or output validation
        errors: List of Pydantic validation errors
        
    Returns:
        Rich Panel with formatted error information
    """
    text = Text()
    text.append(f"Schema validation failed for ", style="red")
    text.append(f"{schema_type} ", style="bold red")
    text.append(f"of step ", style="red")
    text.append(f"'{step_name}'\n\n", style="bold cyan")
    
    # Create a table of errors
    table = Table(show_header=True, header_style="bold")
    table.add_column("Field", style="yellow")
    table.add_column("Error", style="red")
    table.add_column("Input", style="dim")
    
    for error in errors:
        field = ".".join(str(loc) for loc in error.get("loc", ["unknown"]))
        msg = error.get("msg", "Unknown error")
        input_val = str(error.get("input", ""))[:50]
        if len(str(error.get("input", ""))) > 50:
            input_val += "..."
        table.add_row(field, msg, input_val)
    
    # Add suggestions
    suggestions = Text()
    suggestions.append("\nSuggestions:\n", style="bold cyan")
    suggestions.append("  → Check that previous pipeline steps completed successfully\n", style="cyan")
    suggestions.append("  → Verify the step is receiving all required inputs\n", style="cyan")
    suggestions.append("  → Review the schema definition in pipeline/schemas.py\n", style="cyan")
    
    return Panel(
        Text.assemble(text, "\n", table, suggestions),
        title=f"❌ {schema_type.title()} Validation Error",
        border_style="red",
        padding=(1, 2),
    )


# =============================================================================
# Validation Decorator
# =============================================================================


def validate_pipeline_step(
    input_schema: Optional[Type[PipelineStepInput]] = None,
    output_schema: Optional[Type[PipelineStepOutput]] = None,
    strict: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to validate pipeline step inputs and outputs against Pydantic schemas.
    
    Args:
        input_schema: Pydantic model for validating input context
        output_schema: Pydantic model for validating output dictionary
        strict: If True, raise exception on validation failure; if False, log warning
        
    Returns:
        Decorated function with schema validation
        
    Example:
        @validate_pipeline_step(
            input_schema=DataLoaderInput,
            output_schema=DataLoaderOutput,
        )
        def load_raw_data(**context) -> dict:
            ...
    """
    def decorator(func: F) -> F:
        step_name = func.__name__
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Validate input
            if input_schema is not None:
                try:
                    input_schema.model_validate(kwargs)
                    logger.debug(
                        f"Input validation passed for step '{step_name}'",
                        extra={"step": step_name, "schema": input_schema.__name__},
                    )
                except Exception as e:
                    errors = []
                    if hasattr(e, "errors"):
                        errors = e.errors()
                    
                    logger.error(
                        f"Input validation failed for step '{step_name}'",
                        extra={
                            "step": step_name,
                            "schema": input_schema.__name__,
                            "errors": errors,
                        },
                    )
                    
                    if strict:
                        # Show formatted error panel
                        panel = format_validation_errors(step_name, "input", errors)
                        console.print(panel)
                        
                        raise SchemaValidationError(
                            f"Input validation failed for step '{step_name}'",
                            step_name=step_name,
                            errors=errors,
                        ) from e
                    else:
                        logger.warning(
                            f"Input validation failed for '{step_name}', continuing anyway"
                        )
            
            # Execute the function
            result = func(*args, **kwargs)
            
            # Validate output
            if output_schema is not None and result is not None:
                try:
                    output_schema.model_validate(result)
                    logger.debug(
                        f"Output validation passed for step '{step_name}'",
                        extra={"step": step_name, "schema": output_schema.__name__},
                    )
                except Exception as e:
                    errors = []
                    if hasattr(e, "errors"):
                        errors = e.errors()
                    
                    logger.error(
                        f"Output validation failed for step '{step_name}'",
                        extra={
                            "step": step_name,
                            "schema": output_schema.__name__,
                            "errors": errors,
                        },
                    )
                    
                    if strict:
                        panel = format_validation_errors(step_name, "output", errors)
                        console.print(panel)
                        
                        raise SchemaValidationError(
                            f"Output validation failed for step '{step_name}'",
                            step_name=step_name,
                            errors=errors,
                        ) from e
                    else:
                        logger.warning(
                            f"Output validation failed for '{step_name}', continuing anyway"
                        )
            
            return result
        
        # Attach schema info to function for introspection
        wrapper.input_schema = input_schema  # type: ignore
        wrapper.output_schema = output_schema  # type: ignore
        
        return wrapper  # type: ignore
    
    return decorator


def auto_validate_step(step_name: str) -> Callable[[F], F]:
    """
    Decorator that automatically applies schema validation based on step name.
    
    Looks up the schemas in STEP_SCHEMAS registry.
    
    Args:
        step_name: Name of the pipeline step (e.g., "data_loader.load_raw_data")
        
    Returns:
        Decorated function with schema validation
        
    Example:
        @auto_validate_step("data_loader.load_raw_data")
        def load_raw_data(**context) -> dict:
            ...
    """
    schemas = get_step_schemas(step_name)
    
    if schemas is None:
        # No schemas defined, return identity decorator
        def identity(func: F) -> F:
            return func
        return identity
    
    return validate_pipeline_step(
        input_schema=schemas.get("input"),
        output_schema=schemas.get("output"),
    )


# =============================================================================
# Schema Documentation Generation
# =============================================================================


def generate_schema_docs() -> str:
    """
    Generate markdown documentation for all pipeline step schemas.
    
    Returns:
        Markdown string documenting all schemas
    """
    lines = [
        "# Pipeline Step Schemas",
        "",
        "This document describes the input and output schemas for each pipeline step.",
        "",
    ]
    
    for step_name, schemas in STEP_SCHEMAS.items():
        lines.append(f"## `{step_name}`")
        lines.append("")
        
        # Input schema
        input_schema = schemas.get("input")
        if input_schema:
            lines.append("### Input")
            lines.append("")
            lines.append("| Field | Type | Required | Description |")
            lines.append("|-------|------|----------|-------------|")
            
            for field_name, field_info in input_schema.model_fields.items():
                field_type = str(field_info.annotation).replace("typing.", "")
                required = "Yes" if field_info.is_required() else "No"
                description = field_info.description or ""
                lines.append(f"| `{field_name}` | `{field_type}` | {required} | {description} |")
            lines.append("")
        
        # Output schema
        output_schema = schemas.get("output")
        if output_schema:
            lines.append("### Output")
            lines.append("")
            lines.append("| Field | Type | Required | Description |")
            lines.append("|-------|------|----------|-------------|")
            
            for field_name, field_info in output_schema.model_fields.items():
                field_type = str(field_info.annotation).replace("typing.", "")
                required = "Yes" if field_info.is_required() else "No"
                description = field_info.description or ""
                lines.append(f"| `{field_name}` | `{field_type}` | {required} | {description} |")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    return "\n".join(lines)






