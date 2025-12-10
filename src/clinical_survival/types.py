"""
Type definitions, protocols, and runtime validation for clinical-survival-ml.

This module provides:
- Protocol classes defining interfaces for models, preprocessors, and pipeline steps
- Type aliases for common data structures
- Runtime validation decorators for enforcing type contracts
- Utility functions for type checking

Usage:
    from clinical_survival.types import (
        SurvivalModelProtocol,
        PipelineStepProtocol,
        validate_input,
        validate_output,
        SurvivalArray,
    )
"""

from __future__ import annotations

from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from clinical_survival.errors import ClinicalSurvivalError
from clinical_survival.logging_config import get_logger

# Get module logger
logger = get_logger(__name__)

# =============================================================================
# Type Aliases
# =============================================================================

# Survival data types
SurvivalArray = np.ndarray  # Structured array with (event, time) dtype
FeatureMatrix = Union[pd.DataFrame, np.ndarray]
RiskScores = np.ndarray
SurvivalProbabilities = np.ndarray  # Shape: (n_samples, n_times)
TimePoints = Union[List[float], np.ndarray, Iterable[float]]

# Configuration types
ConfigDict = Dict[str, Any]
ParamGrid = Dict[str, List[Any]]

# Pipeline context
PipelineContext = Dict[str, Any]


# =============================================================================
# Custom Exceptions
# =============================================================================


class TypeValidationError(ClinicalSurvivalError):
    """Raised when runtime type validation fails."""

    def __init__(self, message: str, expected: str, received: str, param_name: str = ""):
        self.expected = expected
        self.received = received
        self.param_name = param_name
        super().__init__(message)


class SchemaValidationError(ClinicalSurvivalError):
    """Raised when pipeline step input/output schema validation fails."""

    def __init__(self, message: str, step_name: str, errors: List[Dict[str, Any]]):
        self.step_name = step_name
        self.errors = errors
        super().__init__(message)


# =============================================================================
# Protocol Classes (Interfaces)
# =============================================================================


@runtime_checkable
class SurvivalModelProtocol(Protocol):
    """
    Protocol defining the interface for survival models.
    
    All survival models must implement these methods to be compatible
    with the pipeline.
    """

    name: str

    def fit(
        self, X: FeatureMatrix, y: Sequence[Any]
    ) -> "SurvivalModelProtocol":
        """Fit the model to training data."""
        ...

    def predict_risk(self, X: FeatureMatrix) -> RiskScores:
        """Predict risk scores for samples."""
        ...

    def predict_survival_function(
        self, X: FeatureMatrix, times: TimePoints
    ) -> SurvivalProbabilities:
        """Predict survival probabilities at given time points."""
        ...


@runtime_checkable
class PreprocessorProtocol(Protocol):
    """
    Protocol defining the interface for preprocessors.
    
    Preprocessors must implement fit, transform, and fit_transform methods.
    """

    def fit(self, X: FeatureMatrix, y: Optional[Any] = None) -> "PreprocessorProtocol":
        """Fit the preprocessor to training data."""
        ...

    def transform(self, X: FeatureMatrix) -> FeatureMatrix:
        """Transform data using the fitted preprocessor."""
        ...

    def fit_transform(self, X: FeatureMatrix, y: Optional[Any] = None) -> FeatureMatrix:
        """Fit and transform in one step."""
        ...


@runtime_checkable
class PipelineStepProtocol(Protocol):
    """
    Protocol defining the interface for pipeline steps.
    
    Pipeline steps are callable objects that take a context dictionary
    and return an updated context.
    """

    def __call__(self, **context: Any) -> Optional[Dict[str, Any]]:
        """Execute the pipeline step."""
        ...


# =============================================================================
# Type Guards
# =============================================================================


def is_survival_model(obj: Any) -> bool:
    """Check if an object implements the SurvivalModelProtocol."""
    return isinstance(obj, SurvivalModelProtocol)


def is_preprocessor(obj: Any) -> bool:
    """Check if an object implements the PreprocessorProtocol."""
    return isinstance(obj, PreprocessorProtocol)


def is_dataframe(obj: Any) -> bool:
    """Check if an object is a pandas DataFrame."""
    return isinstance(obj, pd.DataFrame)


def is_numpy_array(obj: Any) -> bool:
    """Check if an object is a numpy array."""
    return isinstance(obj, np.ndarray)


def is_feature_matrix(obj: Any) -> bool:
    """Check if an object is a valid feature matrix (DataFrame or ndarray)."""
    return is_dataframe(obj) or is_numpy_array(obj)


# =============================================================================
# Runtime Validation Decorators
# =============================================================================

F = TypeVar("F", bound=Callable[..., Any])


def validate_input(
    param_name: str,
    expected_type: Union[Type, Tuple[Type, ...]],
    allow_none: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to validate function input parameters at runtime.
    
    Args:
        param_name: Name of the parameter to validate
        expected_type: Expected type(s) for the parameter
        allow_none: Whether None is an acceptable value
        
    Returns:
        Decorated function with input validation
        
    Example:
        @validate_input("X", pd.DataFrame)
        @validate_input("y", (np.ndarray, pd.Series))
        def train(X: pd.DataFrame, y: np.ndarray) -> Model:
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the parameter value from args or kwargs
            import inspect

            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            value = None
            if param_name in kwargs:
                value = kwargs[param_name]
            elif param_name in params:
                idx = params.index(param_name)
                if idx < len(args):
                    value = args[idx]

            # Validate
            if value is None:
                if not allow_none:
                    raise TypeValidationError(
                        f"Parameter '{param_name}' cannot be None",
                        expected=str(expected_type),
                        received="None",
                        param_name=param_name,
                    )
            elif not isinstance(value, expected_type):
                raise TypeValidationError(
                    f"Parameter '{param_name}' has invalid type. "
                    f"Expected {expected_type}, got {type(value).__name__}",
                    expected=str(expected_type),
                    received=type(value).__name__,
                    param_name=param_name,
                )

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def validate_output(expected_type: Union[Type, Tuple[Type, ...]]) -> Callable[[F], F]:
    """
    Decorator to validate function return value at runtime.
    
    Args:
        expected_type: Expected type(s) for the return value
        
    Returns:
        Decorated function with output validation
        
    Example:
        @validate_output(pd.DataFrame)
        def load_data(path: str) -> pd.DataFrame:
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            if not isinstance(result, expected_type):
                raise TypeValidationError(
                    f"Function '{func.__name__}' returned invalid type. "
                    f"Expected {expected_type}, got {type(result).__name__}",
                    expected=str(expected_type),
                    received=type(result).__name__,
                )

            return result

        return wrapper  # type: ignore

    return decorator


def validate_dataframe_columns(
    param_name: str, required_columns: List[str]
) -> Callable[[F], F]:
    """
    Decorator to validate that a DataFrame parameter has required columns.
    
    Args:
        param_name: Name of the DataFrame parameter
        required_columns: List of column names that must be present
        
    Returns:
        Decorated function with column validation
        
    Example:
        @validate_dataframe_columns("df", ["time", "event"])
        def process_survival_data(df: pd.DataFrame) -> pd.DataFrame:
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import inspect

            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            df = None
            if param_name in kwargs:
                df = kwargs[param_name]
            elif param_name in params:
                idx = params.index(param_name)
                if idx < len(args):
                    df = args[idx]

            if df is not None and isinstance(df, pd.DataFrame):
                missing = set(required_columns) - set(df.columns)
                if missing:
                    raise TypeValidationError(
                        f"DataFrame '{param_name}' is missing required columns: {missing}",
                        expected=str(required_columns),
                        received=str(list(df.columns)),
                        param_name=param_name,
                    )

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# Pipeline Step Schema Validation
# =============================================================================


class StepInputSchema(BaseModel):
    """Base class for pipeline step input schemas."""

    class Config:
        extra = "allow"  # Allow extra fields in context


class StepOutputSchema(BaseModel):
    """Base class for pipeline step output schemas."""

    class Config:
        extra = "allow"


def validate_step_io(
    input_schema: Optional[Type[StepInputSchema]] = None,
    output_schema: Optional[Type[StepOutputSchema]] = None,
) -> Callable[[F], F]:
    """
    Decorator to validate pipeline step input/output against Pydantic schemas.
    
    Args:
        input_schema: Pydantic model class for validating input context
        output_schema: Pydantic model class for validating output
        
    Returns:
        Decorated function with schema validation
        
    Example:
        class LoadDataInput(StepInputSchema):
            params_config: ParamsConfig
            
        class LoadDataOutput(StepOutputSchema):
            raw_df: pd.DataFrame
            
        @validate_step_io(LoadDataInput, LoadDataOutput)
        def load_raw_data(**context) -> dict:
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            step_name = func.__name__

            # Validate input
            if input_schema is not None:
                try:
                    input_schema.model_validate(kwargs)
                except ValidationError as e:
                    logger.error(
                        f"Input validation failed for step '{step_name}'",
                        extra={"errors": e.errors()},
                    )
                    raise SchemaValidationError(
                        f"Input validation failed for pipeline step '{step_name}'",
                        step_name=step_name,
                        errors=e.errors(),
                    ) from e

            # Execute function
            result = func(*args, **kwargs)

            # Validate output
            if output_schema is not None and result is not None:
                try:
                    output_schema.model_validate(result)
                except ValidationError as e:
                    logger.error(
                        f"Output validation failed for step '{step_name}'",
                        extra={"errors": e.errors()},
                    )
                    raise SchemaValidationError(
                        f"Output validation failed for pipeline step '{step_name}'",
                        step_name=step_name,
                        errors=e.errors(),
                    ) from e

            return result

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# Utility Functions
# =============================================================================


def ensure_dataframe(
    X: FeatureMatrix, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Ensure input is a pandas DataFrame.
    
    Args:
        X: Input feature matrix (DataFrame or ndarray)
        columns: Column names to use if X is an ndarray
        
    Returns:
        pandas DataFrame
    """
    if isinstance(X, pd.DataFrame):
        return X.copy()
    if columns is None:
        columns = [f"feature_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=columns)


def ensure_numpy_array(X: FeatureMatrix) -> np.ndarray:
    """
    Ensure input is a numpy array.
    
    Args:
        X: Input feature matrix (DataFrame or ndarray)
        
    Returns:
        numpy ndarray
    """
    if isinstance(X, np.ndarray):
        return X
    return X.to_numpy()


def check_survival_array(y: Any) -> bool:
    """
    Check if array is a valid structured survival array.
    
    Args:
        y: Array to check
        
    Returns:
        True if valid survival array, False otherwise
    """
    if not isinstance(y, np.ndarray):
        return False
    if y.dtype.names is None:
        return False
    return "event" in y.dtype.names and "time" in y.dtype.names







