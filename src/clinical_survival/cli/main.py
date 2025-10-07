"""Main CLI entry point for clinical survival ML."""

from __future__ import annotations

from pathlib import Path

import typer

from clinical_survival.cli.commands import (
    run_evaluate_command,
    run_explain_command,
    run_load_command,
    run_report_command,
    run_run_command,
    run_train_command,
    run_validate_config_command,
    setup_main_callback,
)
from clinical_survival.serve import run_server

app = typer.Typer(help="Clinical survival modeling pipeline")


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", help="Show version and exit"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug output"),
) -> None:
    """Main CLI callback."""
    setup_main_callback(version=version, verbose=verbose, debug=debug)


@app.command()
def load(
    data: Path = typer.Option(Path("data/toy/toy_survival.csv"), exists=True),  # noqa: B008
    meta: Path = typer.Option(Path("data/toy/metadata.yaml"), exists=True),  # noqa: B008
    time_col: str = typer.Option("time"),
    event_col: str = typer.Option("event"),
) -> None:
    """Load and inspect a dataset."""
    run_load_command(data, meta, time_col, event_col)


@app.command()
def train(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
    grid: Path = typer.Option(Path("configs/model_grid.yaml"), exists=True),  # noqa: B008
    features_yaml: Path | None = typer.Option(None, help="Override features YAML"),  # noqa: B008
    seed: int | None = typer.Option(None, help="Override random seed"),
    horizons: list[int]
    | None = typer.Option(None, help="Override evaluation horizons (days)"),  # noqa: B008
    thresholds: list[float]
    | None = typer.Option(None, help="Override decision thresholds"),  # noqa: B008
) -> None:
    """Train survival models with cross-validation."""
    run_train_command(config, grid, features_yaml, seed, horizons, thresholds)


@app.command()
def evaluate_command(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
) -> None:
    """Display evaluation results from previous training."""
    run_evaluate_command(config)


@app.command()
def explain(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
    model_name: str = typer.Option("coxph"),
) -> None:
    """Generate model explanations for the best performing model."""
    run_explain_command(config, model_name)


@app.command()
def report(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
    out: Path = typer.Option(Path("results/report.html")),  # noqa: B008
) -> None:
    """Generate an HTML report with results and visualizations."""
    run_report_command(config, out)


@app.command()
def validate_config(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
    grid: Path = typer.Option(Path("configs/model_grid.yaml"), exists=True),  # noqa: B008
    features: Path = typer.Option(Path("configs/features.yaml"), exists=True),  # noqa: B008
) -> None:
    """Validate configuration files against their schemas."""
    run_validate_config_command(config, grid, features)


@app.command()
def run(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
    grid: Path = typer.Option(Path("configs/model_grid.yaml"), exists=True),  # noqa: B008
) -> None:
    """Run the complete pipeline (train + report)."""
    run_run_command(config, grid)


@app.command()
def serve(
    models_dir: Path = typer.Option(  # noqa: B008
        Path("results/artifacts/models"),
        help="Directory containing trained models",
    ),
    config: Path
    | None = typer.Option(None, help="Configuration file for model metadata"),  # noqa: B008
    host: str = typer.Option("127.0.0.1", help="Host to bind the server to"),
    port: int = typer.Option(8000, help="Port to bind the server to"),
) -> None:
    """Start the REST API server for model predictions."""
    typer.echo("🚀 Starting Clinical Survival ML API server...")
    typer.echo(f"📁 Models directory: {models_dir}")
    typer.echo(f"🌐 Server: http://{host}:{port}")
    typer.echo(f"📚 API documentation: http://{host}:{port}/docs")

    try:
        run_server(models_dir, config, host, port)
    except Exception as e:
        typer.echo(f"❌ Failed to start server: {e}")
        raise typer.Exit(1) from e


if __name__ == "__main__":  # pragma: no cover
    app()
