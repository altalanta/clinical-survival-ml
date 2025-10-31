"""Main CLI entry point for clinical survival ML."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from . import training

app = typer.Typer(
    name="clinical-survival-ml",
    help="Reproducible end-to-end survival modeling for tabular clinical outcomes.",
    add_completion=False,
)
console = Console()

app.add_typer(training.app, name="training")

if __name__ == "__main__":
    app()
