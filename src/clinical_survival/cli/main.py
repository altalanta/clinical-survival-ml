"""Main CLI entry point for clinical survival ML with unified error handling."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from clinical_survival.error_handling import CLIExceptionHandler
from clinical_survival.logging_config import configure_logging, get_logger

from . import training, data, monitoring, mlops, testing, dashboard, api

# Initialize logger
logger = get_logger(__name__)

# Global state for verbose flag (needed for exception handler)
_verbose_mode: bool = False

app = typer.Typer(
    name="clinical-survival-ml",
    help="Reproducible end-to-end survival modeling for tabular clinical outcomes.",
    add_completion=False,
)
console = Console()


def _get_exception_handler() -> CLIExceptionHandler:
    """Get the CLI exception handler with current verbose setting."""
    return CLIExceptionHandler(
        verbose=_verbose_mode,
        show_suggestions=True,
        console=console,
    )


@app.callback()
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose (DEBUG) logging and full tracebacks."
    ),
    log_format: str = typer.Option(
        "rich",
        "--log-format",
        help="Logging format: 'rich' (console), 'structured' (JSON), or 'simple'.",
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        help="Path to write logs to a file (in addition to console).",
    ),
) -> None:
    """
    Clinical Survival ML - Reproducible survival modeling pipeline.
    
    Configure logging options that apply to all subcommands.
    
    Examples:
    
        # Run training with verbose output
        clinical-ml --verbose training run
        
        # Use JSON logging for production
        clinical-ml --log-format structured training run
        
        # Save logs to file
        clinical-ml --log-file results/logs/run.log training run
    """
    global _verbose_mode
    _verbose_mode = verbose
    
    level = "DEBUG" if verbose else "INFO"
    configure_logging(
        level=level,
        format=log_format,
        log_file=str(log_file) if log_file else None,
    )
    logger.debug(
        "CLI initialized",
        extra={"verbose": verbose, "log_format": log_format, "log_file": str(log_file)},
    )


# Register subcommand groups
app.add_typer(training.app, name="training")
app.add_typer(data.app, name="data")
app.add_typer(monitoring.app, name="monitoring")
app.add_typer(mlops.app, name="mlops")
app.add_typer(testing.app, name="testing")
app.add_typer(dashboard.app, name="dashboard")
app.add_typer(api.app, name="api")


def main_with_error_handling() -> None:
    """
    Main entry point with global exception handling.
    
    This wraps the Typer app to provide user-friendly error messages
    for all unhandled exceptions.
    """
    try:
        app()
    except KeyboardInterrupt:
        handler = _get_exception_handler()
        sys.exit(handler.handle(KeyboardInterrupt()))
    except SystemExit:
        # Let SystemExit pass through (normal exit)
        raise
    except Exception as e:
        handler = _get_exception_handler()
        sys.exit(handler.handle(e))


if __name__ == "__main__":
    main_with_error_handling()
