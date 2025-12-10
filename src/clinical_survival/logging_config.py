"""
Centralized logging configuration for the clinical-survival-ml pipeline.

This module provides:
- Structured JSON logging for production environments
- Rich console logging for development
- Correlation ID support for request tracing
- Consistent log formatting across all modules

Usage:
    from clinical_survival.logging_config import get_logger, LogContext

    logger = get_logger(__name__)

    with LogContext(run_id="abc123", model="coxph"):
        logger.info("Training started", extra={"n_samples": 1000})
"""

from __future__ import annotations

import contextvars
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union
import json
import uuid

from rich.console import Console
from rich.logging import RichHandler

# Context variable to store correlation ID and other context
_log_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "log_context", default={}
)

# Type variable for decorator
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class LoggingConfig:
    """Configuration for the logging system."""

    level: str = "INFO"
    format: str = "structured"  # "structured" for JSON, "rich" for console, "simple" for basic
    log_file: Optional[Path] = None
    include_timestamp: bool = True
    include_correlation_id: bool = True
    include_module: bool = True
    include_function: bool = True
    max_message_length: int = 10000
    sensitive_fields: list = field(default_factory=lambda: ["password", "api_key", "token", "secret"])


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Produces log records in a format suitable for log aggregation systems
    like ELK stack, Datadog, or CloudWatch.
    """

    def __init__(self, config: LoggingConfig):
        super().__init__()
        self.config = config

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "level": record.levelname,
            "message": self._truncate_message(record.getMessage()),
            "logger": record.name,
        }

        if self.config.include_timestamp:
            log_entry["timestamp"] = datetime.now(timezone.utc).isoformat()

        if self.config.include_module:
            log_entry["module"] = record.module

        if self.config.include_function:
            log_entry["function"] = record.funcName
            log_entry["line"] = record.lineno

        # Add correlation ID and other context
        if self.config.include_correlation_id:
            context = _log_context.get()
            if context:
                log_entry["context"] = self._sanitize_context(context)

        # Add any extra fields from the log call
        if hasattr(record, "__dict__"):
            extra_fields = {
                k: v
                for k, v in record.__dict__.items()
                if k not in logging.LogRecord.__dict__
                and not k.startswith("_")
                and k not in ("message", "args", "exc_info", "exc_text", "stack_info")
            }
            if extra_fields:
                log_entry["extra"] = self._sanitize_context(extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)

    def _truncate_message(self, message: str) -> str:
        if len(message) > self.config.max_message_length:
            return message[: self.config.max_message_length] + "...[truncated]"
        return message

    def _sanitize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive fields from context."""
        sanitized = {}
        for key, value in context.items():
            if any(sensitive in key.lower() for sensitive in self.config.sensitive_fields):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        return sanitized


class LogContext:
    """
    Context manager for adding contextual information to log records.
    
    Usage:
        with LogContext(run_id="abc123", model="coxph"):
            logger.info("Training started")  # Will include run_id and model
    """

    def __init__(self, **kwargs: Any):
        self.new_context = kwargs
        self.token: Optional[contextvars.Token] = None

    def __enter__(self) -> "LogContext":
        current = _log_context.get().copy()
        current.update(self.new_context)
        self.token = _log_context.set(current)
        return self

    def __exit__(self, *args: Any) -> None:
        if self.token is not None:
            _log_context.reset(self.token)


def generate_correlation_id() -> str:
    """Generate a unique correlation ID for request tracing."""
    return str(uuid.uuid4())[:8]


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set a correlation ID in the current context.
    
    If no ID is provided, generates a new one.
    Returns the correlation ID that was set.
    """
    if correlation_id is None:
        correlation_id = generate_correlation_id()
    
    current = _log_context.get().copy()
    current["correlation_id"] = correlation_id
    _log_context.set(current)
    return correlation_id


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID from context."""
    return _log_context.get().get("correlation_id")


def log_function_call(logger: logging.Logger) -> Callable[[F], F]:
    """
    Decorator to log function entry and exit with timing.
    
    Usage:
        @log_function_call(logger)
        def my_function(x, y):
            return x + y
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = func.__name__
            logger.debug(
                f"Entering {func_name}",
                extra={"args_count": len(args), "kwargs_keys": list(kwargs.keys())},
            )
            start_time = datetime.now(timezone.utc)
            try:
                result = func(*args, **kwargs)
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                logger.debug(
                    f"Exiting {func_name}",
                    extra={"elapsed_seconds": elapsed, "success": True},
                )
                return result
            except Exception as e:
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                logger.error(
                    f"Exception in {func_name}: {e}",
                    extra={"elapsed_seconds": elapsed, "success": False},
                    exc_info=True,
                )
                raise

        return wrapper  # type: ignore

    return decorator


# Global logger cache
_loggers: Dict[str, logging.Logger] = {}
_configured = False
_current_config: Optional[LoggingConfig] = None


def configure_logging(
    level: str = "INFO",
    format: str = "rich",  # "structured", "rich", or "simple"
    log_file: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> None:
    """
    Configure the logging system.
    
    Should be called once at application startup.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Output format ("structured" for JSON, "rich" for console, "simple" for basic)
        log_file: Optional path to write logs to a file
        **kwargs: Additional configuration options
    """
    global _configured, _current_config

    config = LoggingConfig(
        level=level,
        format=format,
        log_file=Path(log_file) if log_file else None,
        **kwargs,
    )
    _current_config = config

    # Get the root logger for clinical_survival
    root_logger = logging.getLogger("clinical_survival")
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()

    # Add appropriate handler based on format
    if format == "structured":
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter(config))
    elif format == "rich":
        console = Console(stderr=True)
        handler = RichHandler(
            console=console,
            show_time=config.include_timestamp,
            show_path=config.include_module,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
    else:  # simple
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

    root_logger.addHandler(handler)

    # Add file handler if specified
    if config.log_file:
        config.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setFormatter(StructuredFormatter(config))
        root_logger.addHandler(file_handler)

    _configured = True

    # Log that logging is configured
    root_logger.debug(
        "Logging configured",
        extra={"level": level, "format": format, "log_file": str(log_file)},
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module name.
    
    Automatically configures logging if not already done.
    
    Args:
        name: Usually __name__ of the calling module
        
    Returns:
        Configured logger instance
    """
    global _configured

    if not _configured:
        # Auto-configure with sensible defaults
        configure_logging(level="INFO", format="rich")

    if name not in _loggers:
        # Ensure the logger is under the clinical_survival namespace
        if not name.startswith("clinical_survival"):
            name = f"clinical_survival.{name}"
        _loggers[name] = logging.getLogger(name)

    return _loggers[name]


class PipelineLogger:
    """
    Specialized logger for pipeline execution with step tracking.
    
    Usage:
        pipeline_logger = PipelineLogger("training_pipeline")
        
        with pipeline_logger.step("data_loading"):
            # Load data
            pass
            
        with pipeline_logger.step("preprocessing"):
            # Preprocess
            pass
    """

    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.logger = get_logger(f"pipeline.{pipeline_name}")
        self.run_id = generate_correlation_id()
        self.step_count = 0
        self.start_time = datetime.now(timezone.utc)

    def step(self, step_name: str) -> "PipelineStepContext":
        """Create a context manager for a pipeline step."""
        self.step_count += 1
        return PipelineStepContext(
            logger=self.logger,
            pipeline_name=self.pipeline_name,
            step_name=step_name,
            step_number=self.step_count,
            run_id=self.run_id,
        )

    def finish(self, success: bool = True, **metrics: Any) -> None:
        """Log pipeline completion."""
        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        status = "completed" if success else "failed"
        self.logger.info(
            f"Pipeline {self.pipeline_name} {status}",
            extra={
                "run_id": self.run_id,
                "total_steps": self.step_count,
                "elapsed_seconds": elapsed,
                "success": success,
                **metrics,
            },
        )


class PipelineStepContext:
    """Context manager for individual pipeline steps."""

    def __init__(
        self,
        logger: logging.Logger,
        pipeline_name: str,
        step_name: str,
        step_number: int,
        run_id: str,
    ):
        self.logger = logger
        self.pipeline_name = pipeline_name
        self.step_name = step_name
        self.step_number = step_number
        self.run_id = run_id
        self.start_time: Optional[datetime] = None
        self.log_context: Optional[LogContext] = None

    def __enter__(self) -> "PipelineStepContext":
        self.start_time = datetime.now(timezone.utc)
        self.log_context = LogContext(
            run_id=self.run_id,
            pipeline=self.pipeline_name,
            step=self.step_name,
            step_number=self.step_number,
        )
        self.log_context.__enter__()
        
        self.logger.info(
            f"Starting step: {self.step_name}",
            extra={"step_number": self.step_number},
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapsed = (
            (datetime.now(timezone.utc) - self.start_time).total_seconds()
            if self.start_time
            else 0
        )
        
        if exc_type is None:
            self.logger.info(
                f"Completed step: {self.step_name}",
                extra={"step_number": self.step_number, "elapsed_seconds": elapsed},
            )
        else:
            self.logger.error(
                f"Failed step: {self.step_name}",
                extra={
                    "step_number": self.step_number,
                    "elapsed_seconds": elapsed,
                    "error_type": exc_type.__name__ if exc_type else None,
                },
                exc_info=True,
            )
        
        if self.log_context:
            self.log_context.__exit__(exc_type, exc_val, exc_tb)

    def log(self, message: str, level: str = "INFO", **extra: Any) -> None:
        """Log a message within this step context."""
        log_method = getattr(self.logger, level.lower())
        log_method(message, extra=extra)







