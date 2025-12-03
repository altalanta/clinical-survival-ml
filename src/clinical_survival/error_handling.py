"""
Unified error handling utilities for the clinical-survival-ml pipeline.

This module provides:
- Error handling decorators for common patterns
- User-friendly error formatting with suggestions
- Global exception handler for CLI
- Error recovery strategies

Usage:
    from clinical_survival.error_handling import (
        handle_errors,
        cli_exception_handler,
        format_error_for_user,
    )
    
    @handle_errors(default_return=None)
    def risky_operation():
        ...
"""

from __future__ import annotations

import sys
import traceback
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.traceback import Traceback

from clinical_survival.errors import (
    ClinicalSurvivalError,
    ConfigurationError,
    DataError,
    DataLoadError,
    DataValidationError,
    MissingColumnError,
    ModelError,
    ModelNotFittedError,
    ModelTrainingError,
    PipelineError,
    StepExecutionError,
    StepNotFoundError,
    TypeValidationError,
    SchemaValidationError,
    ReportError,
)
from clinical_survival.logging_config import get_logger

# Type variable for decorators
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")

# Module logger
logger = get_logger(__name__)

# Console for rich output
console = Console(stderr=True)


# =============================================================================
# Error Suggestions Database
# =============================================================================

ERROR_SUGGESTIONS: Dict[Type[Exception], List[str]] = {
    ConfigurationError: [
        "Check that your configuration file exists and is valid YAML",
        "Verify all required keys are present in the config",
        "Run 'clinical-ml config validate' to check your configuration",
    ],
    DataLoadError: [
        "Verify the file path is correct and the file exists",
        "Check file permissions",
        "Ensure the file format matches what's expected (CSV, Parquet, etc.)",
        "If using DVC, run 'dvc pull' to fetch the data",
    ],
    DataValidationError: [
        "Review the validation errors above for specific issues",
        "Check that your data matches the expected schema in features.yaml",
        "Ensure there are no unexpected missing values or data types",
    ],
    MissingColumnError: [
        "Check that your data file contains all required columns",
        "Verify column names match exactly (including case)",
        "Update features.yaml if column names have changed",
    ],
    ModelNotFittedError: [
        "Ensure you call model.fit() before making predictions",
        "Check that the training step completed successfully",
        "Verify the model was saved and loaded correctly",
    ],
    ModelTrainingError: [
        "Check that your training data has sufficient samples",
        "Verify there are no NaN or infinite values in the data",
        "Try reducing model complexity or adjusting hyperparameters",
        "Check memory usage - you may need to reduce batch size",
    ],
    StepNotFoundError: [
        "Verify the step name in your pipeline configuration",
        "Check that the module containing the step is installed",
        "Run 'clinical-ml pipeline list-steps' to see available steps",
    ],
    StepExecutionError: [
        "Review the error details above for the root cause",
        "Check that all required inputs are available from previous steps",
        "Verify the step's configuration is correct",
    ],
    TypeValidationError: [
        "Check that you're passing the correct data types",
        "Review the function signature for expected types",
        "Ensure DataFrames have the expected columns and dtypes",
    ],
    SchemaValidationError: [
        "Review the schema definition for the pipeline step",
        "Check that all required fields are provided",
        "Verify data types match the schema expectations",
    ],
    ReportError: [
        "Check that the report template exists",
        "Verify all required data is available for the report",
        "Ensure the output directory is writable",
    ],
    FileNotFoundError: [
        "Verify the file path is correct",
        "Check that the file exists on disk",
        "If using relative paths, ensure you're in the correct directory",
    ],
    PermissionError: [
        "Check file/directory permissions",
        "Try running with appropriate privileges",
        "Verify the path is not read-only",
    ],
    MemoryError: [
        "Try reducing the dataset size or batch size",
        "Close other applications to free memory",
        "Consider using chunked processing for large datasets",
    ],
    KeyboardInterrupt: [
        "Pipeline was interrupted by user",
        "Partial results may be available in the output directory",
    ],
}


# =============================================================================
# Error Formatting
# =============================================================================


def get_suggestions_for_error(error: Exception) -> List[str]:
    """
    Get helpful suggestions for resolving an error.
    
    Args:
        error: The exception to get suggestions for
        
    Returns:
        List of suggestion strings
    """
    # Check for exact type match first
    error_type = type(error)
    if error_type in ERROR_SUGGESTIONS:
        return ERROR_SUGGESTIONS[error_type]
    
    # Check for parent class matches
    for exc_type, suggestions in ERROR_SUGGESTIONS.items():
        if isinstance(error, exc_type):
            return suggestions
    
    # Default suggestions
    return [
        "Check the error message and stack trace for details",
        "Verify your configuration and input data",
        "Run with --verbose flag for more detailed output",
    ]


def format_error_for_user(
    error: Exception,
    show_traceback: bool = False,
    show_suggestions: bool = True,
) -> Panel:
    """
    Format an error into a user-friendly Rich panel.
    
    Args:
        error: The exception to format
        show_traceback: Whether to include the full traceback
        show_suggestions: Whether to include suggestions
        
    Returns:
        Rich Panel object for display
    """
    # Build error message
    error_text = Text()
    
    # Error type and message
    error_type = type(error).__name__
    error_message = str(error)
    
    error_text.append(f"{error_type}\n", style="bold red")
    error_text.append(f"{error_message}\n", style="red")
    
    # Add context if available (for our custom exceptions)
    if isinstance(error, ClinicalSurvivalError) and error.context:
        error_text.append("\nContext:\n", style="bold yellow")
        for key, value in error.context.items():
            error_text.append(f"  • {key}: ", style="yellow")
            error_text.append(f"{value}\n", style="white")
    
    # Add suggestions
    if show_suggestions:
        suggestions = get_suggestions_for_error(error)
        if suggestions:
            error_text.append("\nSuggestions:\n", style="bold cyan")
            for suggestion in suggestions:
                error_text.append(f"  → {suggestion}\n", style="cyan")
    
    # Add traceback if requested
    if show_traceback:
        error_text.append("\nTraceback:\n", style="bold dim")
        tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
        for line in tb_lines:
            error_text.append(line, style="dim")
    
    # Determine panel style based on error severity
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        title = "⚠️  Interrupted"
        border_style = "yellow"
    elif isinstance(error, ClinicalSurvivalError):
        title = "❌ Pipeline Error"
        border_style = "red"
    else:
        title = "❌ Unexpected Error"
        border_style = "red"
    
    return Panel(
        error_text,
        title=title,
        border_style=border_style,
        padding=(1, 2),
    )


def format_error_summary(errors: List[Exception]) -> Panel:
    """
    Format multiple errors into a summary panel.
    
    Args:
        errors: List of exceptions
        
    Returns:
        Rich Panel with error summary
    """
    text = Text()
    text.append(f"Encountered {len(errors)} error(s):\n\n", style="bold red")
    
    for i, error in enumerate(errors, 1):
        error_type = type(error).__name__
        error_msg = str(error)[:100]  # Truncate long messages
        if len(str(error)) > 100:
            error_msg += "..."
        text.append(f"{i}. ", style="bold")
        text.append(f"{error_type}: ", style="red")
        text.append(f"{error_msg}\n", style="white")
    
    return Panel(
        text,
        title="❌ Error Summary",
        border_style="red",
        padding=(1, 2),
    )


# =============================================================================
# Error Handling Decorators
# =============================================================================


def handle_errors(
    *catch_types: Type[Exception],
    default_return: Any = None,
    reraise: bool = False,
    log_level: str = "error",
    message: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Decorator to handle exceptions with logging and optional recovery.
    
    Args:
        *catch_types: Exception types to catch (default: Exception)
        default_return: Value to return on error (if not reraising)
        reraise: Whether to re-raise the exception after handling
        log_level: Log level for the error message
        message: Custom error message prefix
        
    Returns:
        Decorated function
        
    Example:
        @handle_errors(FileNotFoundError, default_return=None)
        def load_optional_config(path):
            return yaml.safe_load(open(path))
            
        @handle_errors(reraise=True, message="Training failed")
        def train_model(X, y):
            ...
    """
    if not catch_types:
        catch_types = (Exception,)
    
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except catch_types as e:
                # Build log message
                log_msg = message or f"Error in {func.__name__}"
                log_msg = f"{log_msg}: {e}"
                
                # Log the error
                log_method = getattr(logger, log_level)
                log_method(
                    log_msg,
                    extra={
                        "function": func.__name__,
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )
                
                if reraise:
                    raise
                return default_return
        
        return wrapper  # type: ignore
    
    return decorator


def wrap_step_errors(step_name: str) -> Callable[[F], F]:
    """
    Decorator to wrap pipeline step errors with context.
    
    Converts generic exceptions into StepExecutionError with
    the step name and original error preserved.
    
    Args:
        step_name: Name of the pipeline step
        
    Returns:
        Decorated function
        
    Example:
        @wrap_step_errors("data_loading")
        def load_raw_data(**context):
            ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except ClinicalSurvivalError:
                # Re-raise our custom exceptions as-is
                raise
            except Exception as e:
                # Wrap other exceptions
                raise StepExecutionError(
                    message=f"Step '{step_name}' failed: {e}",
                    step_name=step_name,
                    original_error=e,
                ) from e
        
        return wrapper  # type: ignore
    
    return decorator


def require_fitted(model_attr: str = "model") -> Callable[[F], F]:
    """
    Decorator to ensure a model is fitted before prediction.
    
    Args:
        model_attr: Name of the model attribute to check
        
    Returns:
        Decorated function
        
    Example:
        class MyModel:
            @require_fitted()
            def predict(self, X):
                return self.model.predict(X)
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            model = getattr(self, model_attr, None)
            if model is None:
                model_name = getattr(self, "name", self.__class__.__name__)
                raise ModelNotFittedError(model_name)
            return func(self, *args, **kwargs)
        
        return wrapper  # type: ignore
    
    return decorator


def validate_file_exists(param_name: str) -> Callable[[F], F]:
    """
    Decorator to validate that a file path parameter exists.
    
    Args:
        param_name: Name of the path parameter
        
    Returns:
        Decorated function
        
    Example:
        @validate_file_exists("config_path")
        def load_config(config_path: Path):
            ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import inspect
            
            # Get parameter value
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            path_value = None
            if param_name in kwargs:
                path_value = kwargs[param_name]
            elif param_name in params:
                idx = params.index(param_name)
                if idx < len(args):
                    path_value = args[idx]
            
            # Validate
            if path_value is not None:
                path = Path(path_value)
                if not path.exists():
                    raise DataLoadError(
                        f"File not found: {path}",
                        path=str(path),
                    )
            
            return func(*args, **kwargs)
        
        return wrapper  # type: ignore
    
    return decorator


# =============================================================================
# CLI Exception Handler
# =============================================================================


class CLIExceptionHandler:
    """
    Global exception handler for CLI commands.
    
    Provides user-friendly error output with suggestions
    and optional debug information.
    
    Usage:
        handler = CLIExceptionHandler(verbose=True)
        
        try:
            run_pipeline()
        except Exception as e:
            handler.handle(e)
            sys.exit(1)
    """
    
    def __init__(
        self,
        verbose: bool = False,
        show_suggestions: bool = True,
        console: Optional[Console] = None,
    ):
        self.verbose = verbose
        self.show_suggestions = show_suggestions
        self.console = console or Console(stderr=True)
    
    def handle(self, error: Exception) -> int:
        """
        Handle an exception and display user-friendly output.
        
        Args:
            error: The exception to handle
            
        Returns:
            Exit code (1 for errors, 130 for interrupts)
        """
        # Log the error
        logger.error(
            f"CLI error: {error}",
            extra={"error_type": type(error).__name__},
            exc_info=self.verbose,
        )
        
        # Handle keyboard interrupt specially
        if isinstance(error, KeyboardInterrupt):
            self.console.print("\n")
            self.console.print(
                Panel(
                    "Operation cancelled by user.",
                    title="⚠️  Interrupted",
                    border_style="yellow",
                )
            )
            return 130
        
        # Format and display the error
        panel = format_error_for_user(
            error,
            show_traceback=self.verbose,
            show_suggestions=self.show_suggestions,
        )
        self.console.print(panel)
        
        # Show help hint
        if not self.verbose:
            self.console.print(
                "\n[dim]Run with --verbose flag for full traceback[/dim]"
            )
        
        return 1
    
    def __enter__(self) -> "CLIExceptionHandler":
        return self
    
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        if exc_val is not None:
            exit_code = self.handle(exc_val)  # type: ignore
            sys.exit(exit_code)
        return False


def cli_exception_handler(verbose: bool = False) -> CLIExceptionHandler:
    """
    Create a CLI exception handler context manager.
    
    Args:
        verbose: Whether to show full tracebacks
        
    Returns:
        CLIExceptionHandler context manager
        
    Example:
        with cli_exception_handler(verbose=True):
            run_pipeline()
    """
    return CLIExceptionHandler(verbose=verbose)


# =============================================================================
# Error Recovery Utilities
# =============================================================================


def safe_cleanup(
    *cleanup_funcs: Callable[[], None],
    ignore_errors: bool = True,
) -> None:
    """
    Execute cleanup functions safely, optionally ignoring errors.
    
    Args:
        *cleanup_funcs: Functions to call for cleanup
        ignore_errors: Whether to continue on errors
    """
    for func in cleanup_funcs:
        try:
            func()
        except Exception as e:
            if ignore_errors:
                logger.warning(f"Cleanup error (ignored): {e}")
            else:
                raise


def retry_on_error(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator to retry a function on failure with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts (seconds)
        backoff: Multiplier for delay after each attempt
        exceptions: Exception types to catch and retry
        
    Returns:
        Decorated function
        
    Example:
        @retry_on_error(max_attempts=3, exceptions=(ConnectionError,))
        def fetch_data(url):
            ...
    """
    import time
    
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception: Optional[Exception] = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}"
                        )
            
            # Re-raise the last exception
            if last_exception:
                raise last_exception
        
        return wrapper  # type: ignore
    
    return decorator



