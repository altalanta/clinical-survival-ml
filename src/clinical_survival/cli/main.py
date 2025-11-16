"""Main CLI entry point for clinical survival ML."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from . import training, data, monitoring, mlops, testing, dashboard

app = typer.Typer(
    name="clinical-survival-ml",
    help="Reproducible end-to-end survival modeling for tabular clinical outcomes.",
    add_completion=False,
)
console = Console()

app.add_typer(training.app, name="training")
app.add_typer(data.app, name="data")
app.add_typer(monitoring.app, name="monitoring")
app.add_typer(mlops.app, name="mlops")
app.add_typer(testing.app, name="testing")
app.add_typer(dashboard.app, name="dashboard")


if __name__ == "__main__":
    app()
