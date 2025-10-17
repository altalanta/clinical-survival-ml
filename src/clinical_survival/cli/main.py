"""Main CLI entry point for clinical survival ML."""

from __future__ import annotations

from pathlib import Path

import typer

from clinical_survival.cli.commands import (
    run_automl_command,
    run_benchmark_hardware_command,
    run_configure_distributed_command,
    run_configure_incremental_command,
    run_counterfactual_command,
    run_distributed_benchmark_command,
    run_distributed_evaluate_command,
    run_distributed_train_command,
    run_drift_command,
    run_evaluate_command,
    run_explain_command,
    run_incremental_status_command,
    run_load_command,
    run_monitor_command,
    run_monitoring_status_command,
    run_report_command,
    run_reset_monitoring_command,
    run_run_command,
    run_train_command,
    run_update_models_command,
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


@app.command()
def update_models(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
    data: Path = typer.Option(Path("data/toy/toy_survival.csv"), exists=True),  # noqa: B008
    meta: Path = typer.Option(Path("data/toy/metadata.yaml"), exists=True),  # noqa: B008
    models_dir: Path = typer.Option(Path("results/artifacts/models"), help="Directory containing trained models"),  # noqa: B008
    incremental_config: Path | None = typer.Option(None, help="Incremental learning configuration file"),  # noqa: B008
    model_names: list[str] | None = typer.Option(None, help="Specific models to update (or all if not specified)"),
    force_update: bool = typer.Option(False, help="Force update even if conditions not met"),
) -> None:
    """Update trained models with new data using incremental learning."""
    run_update_models_command(
        config, data, meta, models_dir, incremental_config, model_names, force_update
    )


@app.command()
def incremental_status(
    models_dir: Path = typer.Option(Path("results/artifacts/models"), help="Directory containing trained models"),  # noqa: B008
    model_names: list[str] | None = typer.Option(None, help="Specific models to check (or all if not specified)"),
) -> None:
    """Show status of incremental learning for models."""
    run_incremental_status_command(models_dir, model_names)


@app.command()
def configure_incremental(
    config_path: Path = typer.Option(Path("configs/incremental_config.json"), help="Path to save incremental learning configuration"),  # noqa: B008
    update_frequency_days: int = typer.Option(7, help="How often to check for updates (days)"),
    min_samples_for_update: int = typer.Option(50, help="Minimum new samples before updating"),
    max_samples_in_memory: int = typer.Option(1000, help="Maximum samples to keep in memory"),
    update_strategy: str = typer.Option("online", help="Update strategy: online, batch, or sliding_window"),
    drift_detection_enabled: bool = typer.Option(True, help="Enable drift detection"),
    create_backup_before_update: bool = typer.Option(True, help="Create backup before updating"),
) -> None:
    """Configure incremental learning settings."""
    run_configure_incremental_command(
        config_path, update_frequency_days, min_samples_for_update,
        max_samples_in_memory, update_strategy, drift_detection_enabled,
        create_backup_before_update
    )


@app.command()
def distributed_benchmark(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
    cluster_type: str = typer.Option("local", help="Cluster type: local, dask, ray"),
    n_workers: int = typer.Option(4, help="Number of workers"),
    dataset_sizes: list[int] = typer.Option([1000, 5000, 10000], help="Dataset sizes to benchmark"),
    model_type: str = typer.Option("coxph", help="Model type to benchmark"),
    output_dir: Path = typer.Option(Path("results/distributed_benchmark"), help="Output directory"),  # noqa: B008
) -> None:
    """Benchmark distributed computing performance across different dataset sizes."""
    run_distributed_benchmark_command(
        config, cluster_type, n_workers, dataset_sizes, model_type, output_dir
    )


@app.command()
def distributed_train(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
    data: Path = typer.Option(Path("data/toy/toy_survival.csv"), exists=True),  # noqa: B008
    meta: Path = typer.Option(Path("data/toy/metadata.yaml"), exists=True),  # noqa: B008
    cluster_type: str = typer.Option("local", help="Cluster type: local, dask, ray"),
    n_workers: int = typer.Option(4, help="Number of workers"),
    n_partitions: int = typer.Option(10, help="Number of data partitions"),
    model_type: str = typer.Option("coxph", help="Model type to train"),
    output_dir: Path = typer.Option(Path("results/distributed_training"), help="Output directory"),  # noqa: B008
) -> None:
    """Train model using distributed computing."""
    run_distributed_train_command(
        config, data, meta, cluster_type, n_workers, n_partitions, model_type, output_dir
    )


@app.command()
def distributed_evaluate(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
    data: Path = typer.Option(Path("data/toy/toy_survival.csv"), exists=True),  # noqa: B008
    meta: Path = typer.Option(Path("data/toy/metadata.yaml"), exists=True),  # noqa: B008
    model: Path = typer.Option(..., help="Path to trained model file"),  # noqa: B008
    cluster_type: str = typer.Option("local", help="Cluster type: local, dask, ray"),
    n_workers: int = typer.Option(4, help="Number of workers"),
    n_partitions: int = typer.Option(10, help="Number of data partitions"),
    metrics: list[str] = typer.Option(["concordance", "ibs"], help="Metrics to compute"),
) -> None:
    """Evaluate model using distributed computing."""
    run_distributed_evaluate_command(
        config, data, meta, model, cluster_type, n_workers, n_partitions, metrics
    )


@app.command()
def configure_distributed(
    config_path: Path = typer.Option(Path("configs/distributed_config.json"), help="Path to save distributed computing configuration"),  # noqa: B008
    cluster_type: str = typer.Option("local", help="Cluster type: local, dask, ray"),
    n_workers: int = typer.Option(4, help="Number of workers"),
    threads_per_worker: int = typer.Option(2, help="Threads per worker"),
    memory_per_worker: str = typer.Option("2GB", help="Memory per worker"),
    partition_strategy: str = typer.Option("balanced", help="Partition strategy: balanced, hash, random"),
    n_partitions: int = typer.Option(10, help="Number of partitions"),
) -> None:
    """Configure distributed computing settings."""
    run_configure_distributed_command(
        config_path, cluster_type, n_workers, threads_per_worker,
        memory_per_worker, partition_strategy, n_partitions
    )


if __name__ == "__main__":  # pragma: no cover
    app()
