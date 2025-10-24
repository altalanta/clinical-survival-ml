"""Logging utilities for clinical survival ML."""

from __future__ import annotations

import logging
import sys
from typing import Any

from pythonjsonlogger import jsonlogger # Import JsonFormatter

# Global logger instance
logger = logging.getLogger("clinical_survival")


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """Setup logging configuration for the application.

    Args:
        verbose: Enable verbose output (INFO level)
        debug: Enable debug output (DEBUG level)
    """
    # Clear any existing handlers
    logger.handlers.clear()

    # Set log level
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logger.setLevel(level)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Create formatter
    if debug:
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(filename)s %(lineno)d %(message)s"
        )
    else:
        formatter = logging.Formatter(fmt="%(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent duplicate messages
    logger.propagate = False


class ProgressIndicator:
    """Simple progress indicator for long-running operations."""

    def __init__(self, operation: str, total: int = 100):
        self.operation = operation
        self.total = total
        self.current = 0

    def update(self, increment: int = 1) -> None:
        """Update progress by increment steps."""
        self.current += increment
        if self.current <= self.total:
            percentage = (self.current / self.total) * 100
            print(f"\r{self.operation}: {percentage:.1f}%", end="", flush=True)

    def finish(self) -> None:
        """Complete the progress indicator."""
        print(f"\r{self.operation}: 100.0% ✅", flush=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:  # No exception occurred
            self.finish()


def log_function_call(func_name: str, args: dict[str, Any] | None = None) -> None:
    """Log function call for debugging purposes."""
    if logger.isEnabledFor(logging.DEBUG):
        args_str = f" with args: {args}" if args else ""
        logger.debug(f"Calling {func_name}{args_str}")


def log_error_with_context(error: Exception, context: str) -> None:
    """Log an error with additional context information."""
    logger.error(f"Error in {context}: {error}")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Full traceback:", exc_info=True)


def format_validation_error(error: str, suggestion: str | None = None) -> str:
    """Format a validation error with optional suggestion."""
    message = f"❌ {error}"
    if suggestion:
        message += f"\n💡 Suggestion: {suggestion}"
    return message


def format_success_message(message: str) -> str:
    """Format a success message."""
    return f"✅ {message}"


def format_warning_message(message: str) -> str:
    """Format a warning message."""
    return f"⚠️  {message}"
