"""
Custom exceptions for the clinical-survival-ml pipeline.

This module defines a hierarchy of exceptions that provide:
- Clear error categorization for different failure modes
- Structured error context for debugging
- Consistent error handling across the codebase

Exception Hierarchy:
    ClinicalSurvivalError (base)
    ├── ConfigurationError - Invalid or missing configuration
    ├── DataError - Data loading, validation, or processing issues
    │   ├── DataLoadError - Failed to load data from source
    │   ├── DataValidationError - Data failed quality checks
    │   └── MissingColumnError - Required column not found
    ├── ModelError - Model training or inference issues
    │   ├── ModelNotFittedError - Model used before fitting
    │   ├── ModelTrainingError - Training failed
    │   └── ModelInferenceError - Prediction failed
    ├── PipelineError - Pipeline orchestration issues
    │   ├── StepNotFoundError - Pipeline step not found
    │   └── StepExecutionError - Pipeline step failed
    ├── TypeValidationError - Runtime type validation failed
    ├── SchemaValidationError - Schema validation failed
    └── ReportError - Report generation issues
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ClinicalSurvivalError(Exception):
    """
    Base exception for all custom errors in clinical_survival_ml.
    
    Attributes:
        message: Human-readable error message
        context: Additional context for debugging
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.context = context or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} [{context_str}]"
        return self.message


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(ClinicalSurvivalError):
    """
    Exception raised for issues in configuration files.
    
    Examples:
        - Missing required configuration keys
        - Invalid configuration values
        - Incompatible configuration combinations
    """

    def __init__(
        self,
        message: str,
        config_file: Optional[str] = None,
        key: Optional[str] = None,
        **kwargs: Any,
    ):
        context = {"config_file": config_file, "key": key, **kwargs}
        context = {k: v for k, v in context.items() if v is not None}
        super().__init__(message, context)


# =============================================================================
# Data Errors
# =============================================================================


class DataError(ClinicalSurvivalError):
    """
    Base exception for issues related to data loading or processing.
    """

    pass


class DataLoadError(DataError):
    """
    Exception raised when data cannot be loaded from a source.
    
    Examples:
        - File not found
        - Invalid file format
        - Permission denied
    """

    def __init__(
        self,
        message: str,
        path: Optional[str] = None,
        **kwargs: Any,
    ):
        context = {"path": path, **kwargs}
        context = {k: v for k, v in context.items() if v is not None}
        super().__init__(message, context)


class DataValidationError(DataError):
    """
    Exception raised when data fails validation checks.
    
    Examples:
        - Missing required columns
        - Invalid data types
        - Values out of expected range
    """

    def __init__(
        self,
        message: str,
        validation_errors: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        self.validation_errors = validation_errors or []
        context = {"n_errors": len(self.validation_errors), **kwargs}
        super().__init__(message, context)


class MissingColumnError(DataError):
    """
    Exception raised when a required column is not found in the data.
    """

    def __init__(
        self,
        message: str,
        missing_columns: Optional[List[str]] = None,
        available_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        self.missing_columns = missing_columns or []
        self.available_columns = available_columns or []
        context = {
            "missing": self.missing_columns,
            "available": self.available_columns[:10],  # Truncate for readability
            **kwargs,
        }
        super().__init__(message, context)


# =============================================================================
# Model Errors
# =============================================================================


class ModelError(ClinicalSurvivalError):
    """
    Base exception for issues during model training or evaluation.
    """

    pass


class ModelNotFittedError(ModelError):
    """
    Exception raised when a model is used before being fitted.
    """

    def __init__(self, model_name: str, **kwargs: Any):
        message = f"Model '{model_name}' must be fitted before making predictions"
        context = {"model": model_name, **kwargs}
        super().__init__(message, context)


class ModelTrainingError(ModelError):
    """
    Exception raised when model training fails.
    """

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ):
        context = {"model": model_name, **kwargs}
        context = {k: v for k, v in context.items() if v is not None}
        super().__init__(message, context)


class ModelInferenceError(ModelError):
    """
    Exception raised when model inference fails.
    """

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        n_samples: Optional[int] = None,
        **kwargs: Any,
    ):
        context = {"model": model_name, "n_samples": n_samples, **kwargs}
        context = {k: v for k, v in context.items() if v is not None}
        super().__init__(message, context)


# =============================================================================
# Pipeline Errors
# =============================================================================


class PipelineError(ClinicalSurvivalError):
    """
    Base exception for pipeline orchestration issues.
    """

    pass


class StepNotFoundError(PipelineError):
    """
    Exception raised when a pipeline step cannot be found.
    """

    def __init__(
        self,
        step_name: str,
        available_steps: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        message = f"Pipeline step '{step_name}' not found"
        context = {
            "step": step_name,
            "available": available_steps or [],
            **kwargs,
        }
        super().__init__(message, context)


class StepExecutionError(PipelineError):
    """
    Exception raised when a pipeline step fails during execution.
    """

    def __init__(
        self,
        message: str,
        step_name: str,
        original_error: Optional[Exception] = None,
        **kwargs: Any,
    ):
        self.original_error = original_error
        context = {
            "step": step_name,
            "original_error_type": type(original_error).__name__ if original_error else None,
            **kwargs,
        }
        context = {k: v for k, v in context.items() if v is not None}
        super().__init__(message, context)


# =============================================================================
# Type Validation Errors
# =============================================================================


class TypeValidationError(ClinicalSurvivalError):
    """
    Exception raised when runtime type validation fails.
    """

    def __init__(
        self,
        message: str,
        expected: str,
        received: str,
        param_name: Optional[str] = None,
        **kwargs: Any,
    ):
        self.expected = expected
        self.received = received
        self.param_name = param_name
        context = {
            "expected": expected,
            "received": received,
            "param": param_name,
            **kwargs,
        }
        context = {k: v for k, v in context.items() if v is not None}
        super().__init__(message, context)


class SchemaValidationError(ClinicalSurvivalError):
    """
    Exception raised when pipeline step input/output schema validation fails.
    """

    def __init__(
        self,
        message: str,
        step_name: str,
        errors: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        self.step_name = step_name
        self.errors = errors or []
        context = {
            "step": step_name,
            "n_errors": len(self.errors),
            **kwargs,
        }
        super().__init__(message, context)


# =============================================================================
# Report Errors
# =============================================================================


class ReportError(ClinicalSurvivalError):
    """
    Exception raised for issues during report generation.
    """

    def __init__(
        self,
        message: str,
        template: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs: Any,
    ):
        context = {"template": template, "output_path": output_path, **kwargs}
        context = {k: v for k, v in context.items() if v is not None}
        super().__init__(message, context)
