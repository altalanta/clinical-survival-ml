"""Main CLI entry point for clinical survival ML."""

from __future__ import annotations

from pathlib import Path

import typer

from clinical_survival.cli.commands import (
    run_automl_command,
    run_benchmark_hardware_command,
    run_counterfactual_command,
    run_drift_command,
    run_evaluate_command,
    run_explain_command,
    run_load_command,
    run_monitor_command,
    run_monitoring_status_command,
    run_report_command,
    run_reset_monitoring_command,
    run_run_command,
    run_train_command,
    run_validate_config_command,
    setup_main_callback,
)
from clinical_survival.gpu_utils import create_gpu_accelerator
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


@app.command("evaluate")
def evaluate_cli(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
    report: Path | None = typer.Option(
        None,
        "--report",
        help="Optional path to write the evaluation HTML report",
    ),
    competing_risks: str = typer.Option(
        "none",
        "--competing-risks",
        help="Competing risks adjustment. Choices: none, finegray",
    ),
) -> None:
    """Render evaluation artefacts and optionally rebuild the HTML report."""
    run_evaluate_command(config, report=report, competing_risks=competing_risks)


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
def automl(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
    data: Path = typer.Option(Path("data/toy/toy_survival.csv"), exists=True),  # noqa: B008
    meta: Path = typer.Option(Path("data/toy/metadata.yaml"), exists=True),  # noqa: B008
    time_limit: int = typer.Option(1800, help="Time limit in seconds for optimization"),
    model_types: list[str] = typer.Option(
        ["coxph", "rsf", "xgb_cox", "xgb_aft"],
        help="Model types to optimize"
    ),
    metric: str = typer.Option("concordance", help="Metric to optimize"),
    output_dir: Path = typer.Option(Path("results/automl"), help="Output directory"),
) -> None:
    """Run automated model selection and hyperparameter optimization."""
    run_automl_command(config, data, meta, time_limit, model_types, metric, output_dir)


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


@app.command()
def monitor(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
    data: Path = typer.Option(Path("data/toy/toy_survival.csv"), help="Data file for monitoring"),  # noqa: B008
    meta: Path = typer.Option(Path("data/toy/metadata.yaml"), help="Metadata file"),  # noqa: B008
    model_name: str | None = typer.Option(None, help="Specific model to monitor"),
    batch_size: int = typer.Option(100, help="Batch size for processing"),
    save_monitoring: bool = typer.Option(True, help="Save monitoring data to disk"),
) -> None:
    """Monitor model predictions for drift and performance issues."""
    run_monitor_command(config, data, meta, model_name, batch_size, save_monitoring)


@app.command()
def drift(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
    model_name: str | None = typer.Option(None, help="Specific model to check"),
    days: int = typer.Option(7, help="Number of days to analyze"),
    show_details: bool = typer.Option(False, help="Show detailed drift information"),
) -> None:
    """Check for model drift and performance degradation."""
    run_drift_command(config, model_name, days, show_details)


@app.command()
def monitoring_status(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
) -> None:
    """Show monitoring status dashboard for all models."""
    run_monitoring_status_command(config)


@app.command()
def reset_monitoring(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
    model_name: str | None = typer.Option(None, help="Specific model to reset (or all if not specified)"),
    confirm: bool = typer.Option(False, help="Confirm the reset operation"),
) -> None:
    """Reset monitoring baselines and historical data."""
    run_reset_monitoring_command(config, model_name, confirm)


@app.command()
def benchmark_hardware(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
    data: Path = typer.Option(Path("data/toy/toy_survival.csv"), exists=True),  # noqa: B008
    meta: Path = typer.Option(Path("data/toy/metadata.yaml"), exists=True),  # noqa: B008
    model_type: str = typer.Option("xgb_cox", help="Model type to benchmark"),
    use_gpu: bool = typer.Option(True, help="Whether to test GPU acceleration"),
    gpu_id: int = typer.Option(0, help="GPU device ID to test"),
) -> None:
    """Benchmark hardware performance and check GPU availability."""
    run_benchmark_hardware_command(config, data, meta, model_type, use_gpu, gpu_id)


@app.command()
def counterfactual(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
    data: Path = typer.Option(Path("data/toy/toy_survival.csv"), exists=True),  # noqa: B008
    meta: Path = typer.Option(Path("data/toy/metadata.yaml"), exists=True),  # noqa: B008
    model_name: str = typer.Option("xgb_cox", help="Model to use for explanations"),
    target_risk: float | None = typer.Option(None, help="Target risk value to achieve"),
    target_time: float | None = typer.Option(None, help="Target survival time to achieve"),
    n_counterfactuals: int = typer.Option(3, help="Number of counterfactuals to generate"),
    method: str = typer.Option("gradient", help="Counterfactual generation method"),
    output_dir: Path = typer.Option(Path("results/counterfactuals"), help="Output directory"),
) -> None:
    """Generate counterfactual explanations for model predictions."""
    run_counterfactual_command(
        config, data, meta, model_name, target_risk, target_time,
        n_counterfactuals, method, output_dir
    )


if __name__ == "__main__":  # pragma: no cover
    app()
