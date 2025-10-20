"""Main CLI entry point for clinical survival ML."""

from __future__ import annotations

from pathlib import Path

import typer

from clinical_survival.cli.commands import (
    run_ab_test_results_command,
    run_automl_command,
    run_benchmark_hardware_command,
    run_benchmark_suite_command,
    run_check_retraining_triggers_command,
    run_clinical_interpret_command,
    run_configure_distributed_command,
    run_configure_incremental_command,
    run_counterfactual_command,
    run_create_ab_test_command,
    run_cv_integrity_command,
    run_data_cleansing_command,
    run_data_quality_profile_command,
    run_data_validation_command,
    run_deploy_model_command,
    run_distributed_benchmark_command,
    run_distributed_evaluate_command,
    run_distributed_train_command,
    run_drift_command,
    run_evaluate_command,
    run_explain_command,
    run_incremental_status_command,
    run_load_command,
    run_mlops_status_command,
    run_monitor_command,
    run_monitoring_status_command,
    run_performance_regression_command,
    run_register_model_command,
    run_report_command,
    run_reset_monitoring_command,
    run_risk_stratification_command,
    run_rollback_deployment_command,
    run_run_command,
    run_synthetic_data_command,
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


@app.command()
def clinical_interpret(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
    data: Path = typer.Option(Path("data/toy/toy_survival.csv"), exists=True),  # noqa: B008
    meta: Path = typer.Option(Path("data/toy/metadata.yaml"), exists=True),  # noqa: B008
    model: Path = typer.Option(..., help="Path to trained model file"),  # noqa: B008
    output_dir: Path = typer.Option(Path("results/clinical_interpretability"), help="Output directory"),  # noqa: B008
    patient_ids: list[str] | None = typer.Option(None, help="Specific patient IDs to analyze"),
    include_detailed_analysis: bool = typer.Option(True, help="Include detailed feature analysis"),
    output_format: str = typer.Option("html", help="Output format: html or json"),
) -> None:
    """Generate comprehensive clinical interpretability report."""
    run_clinical_interpret_command(
        config, data, meta, model, output_dir, patient_ids, None,
        include_detailed_analysis, output_format
    )


@app.command()
def risk_stratification(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
    data: Path = typer.Option(Path("data/toy/toy_survival.csv"), exists=True),  # noqa: B008
    meta: Path = typer.Option(Path("data/toy/metadata.yaml"), exists=True),  # noqa: B008
    model: Path = typer.Option(..., help="Path to trained model file"),  # noqa: B008
    output_dir: Path = typer.Option(Path("results/risk_stratification"), help="Output directory"),  # noqa: B008
) -> None:
    """Generate risk stratification report."""
    run_risk_stratification_command(
        config, data, meta, model, output_dir, None
    )


@app.command()
def register_model(
    model: Path = typer.Option(..., help="Path to trained model file"),  # noqa: B008
    model_name: str = typer.Option(..., help="Name of the model"),
    version_number: str = typer.Option(..., help="Version number (e.g., 1.0.0)"),
    description: str = typer.Option("", help="Model description"),
    tags: list[str] = typer.Option(None, help="Model tags"),
    registry_path: Path = typer.Option(Path("results/mlops"), help="MLOps registry path"),  # noqa: B008
    created_by: str = typer.Option("system", help="Who created this model"),
) -> None:
    """Register a trained model in the MLOps registry."""
    run_register_model_command(
        model, model_name, version_number, description, tags, registry_path, created_by
    )


@app.command()
def mlops_status(
    registry_path: Path = typer.Option(Path("results/mlops"), help="MLOps registry path"),  # noqa: B008
    model_name: str | None = typer.Option(None, help="Specific model to check"),
) -> None:
    """Show MLOps registry status and model versions."""
    run_mlops_status_command(registry_path, model_name)


@app.command()
def deploy_model(
    version_id: str = typer.Option(..., help="Model version ID to deploy"),
    environment: str = typer.Option(..., help="Target environment (staging, production)"),
    registry_path: Path = typer.Option(Path("results/mlops"), help="MLOps registry path"),  # noqa: B008
    traffic_percentage: float = typer.Option(100.0, help="Traffic percentage for deployment"),
    approved_by: str = typer.Option("system", help="Who approved the deployment"),
) -> None:
    """Deploy a model version to an environment."""
    run_deploy_model_command(
        version_id, environment, registry_path, traffic_percentage, approved_by
    )


@app.command()
def rollback_deployment(
    environment: str = typer.Option(..., help="Environment to rollback"),
    target_version: str = typer.Option(..., help="Target version ID to rollback to"),
    registry_path: Path = typer.Option(Path("results/mlops"), help="MLOps registry path"),  # noqa: B008
    reason: str = typer.Option("manual_rollback", help="Reason for rollback"),
) -> None:
    """Rollback a deployment to a previous version."""
    run_rollback_deployment_command(
        environment, target_version, registry_path, reason
    )


@app.command()
def create_ab_test(
    test_name: str = typer.Option(..., help="Name of the A/B test"),
    model_versions: list[str] = typer.Option(..., help="List of model version IDs to compare"),
    traffic_split: dict[str, float] = typer.Option(..., help="Traffic split as JSON dict"),
    registry_path: Path = typer.Option(Path("results/mlops"), help="MLOps registry path"),  # noqa: B008
    test_duration_days: int = typer.Option(14, help="Test duration in days"),
    success_metrics: list[str] = typer.Option(["concordance", "ibs"], help="Success metrics to evaluate"),
) -> None:
    """Create an A/B test for model versions."""
    run_create_ab_test_command(
        test_name, model_versions, traffic_split, registry_path, test_duration_days, success_metrics
    )


@app.command()
def ab_test_results(
    test_id: str = typer.Option(..., help="A/B test ID"),
    registry_path: Path = typer.Option(Path("results/mlops"), help="MLOps registry path"),  # noqa: B008
) -> None:
    """Get results for an A/B test."""
    run_ab_test_results_command(test_id, registry_path)


@app.command()
def check_retraining_triggers(
    registry_path: Path = typer.Option(Path("results/mlops"), help="MLOps registry path"),  # noqa: B008
    model_name: str | None = typer.Option(None, help="Specific model to check"),
) -> None:
    """Check if any retraining triggers should fire."""
    run_check_retraining_triggers_command(registry_path, model_name)


@app.command()
def data_quality_profile(
    data: Path = typer.Option(Path("data/toy/toy_survival.csv"), exists=True),  # noqa: B008
    meta: Path | None = typer.Option(Path("data/toy/metadata.yaml"), help="Metadata file"),  # noqa: B008
    output_dir: Path = typer.Option(Path("results/data_quality"), help="Output directory"),  # noqa: B008
    include_anomaly_detection: bool = typer.Option(True, help="Include anomaly detection"),
    output_format: str = typer.Option("html", help="Output format: html or json"),
) -> None:
    """Generate comprehensive data quality profile."""
    run_data_quality_profile_command(
        data, meta, output_dir, include_anomaly_detection, None, output_format
    )


@app.command()
def data_validation(
    data: Path = typer.Option(Path("data/toy/toy_survival.csv"), exists=True),  # noqa: B008
    meta: Path | None = typer.Option(Path("data/toy/metadata.yaml"), help="Metadata file"),  # noqa: B008
    validation_rules: str = typer.Option("default", help="Validation rules: default or custom"),
    output_dir: Path = typer.Option(Path("results/data_validation"), help="Output directory"),  # noqa: B008
    strict_mode: bool = typer.Option(False, help="Enable strict validation mode"),
    fail_on_first_error: bool = typer.Option(False, help="Fail on first validation error"),
) -> None:
    """Validate dataset against configured rules."""
    run_data_validation_command(
        data, meta, validation_rules, output_dir, strict_mode, fail_on_first_error
    )


@app.command()
def data_cleansing(
    data: Path = typer.Option(Path("data/toy/toy_survival.csv"), exists=True),  # noqa: B008
    meta: Path | None = typer.Option(Path("data/toy/metadata.yaml"), help="Metadata file"),  # noqa: B008
    output_dir: Path = typer.Option(Path("results/data_cleansing"), help="Output directory"),  # noqa: B008
    remove_duplicates: bool = typer.Option(True, help="Remove duplicate rows"),
    handle_outliers: bool = typer.Option(True, help="Handle outliers"),
    preserve_original: bool = typer.Option(True, help="Preserve original data"),
) -> None:
    """Cleanse dataset based on quality assessment."""
    run_data_cleansing_command(
        data, meta, output_dir, remove_duplicates, handle_outliers, preserve_original
    )


@app.command()
def synthetic_data(
    scenario: str = typer.Option("icu", help="Type of synthetic data to generate (icu, cancer, cardiovascular)"),
    n_samples: int = typer.Option(1000, help="Number of samples to generate"),
    output_dir: Path = typer.Option(Path("data/synthetic"), help="Output directory for generated data"),
    random_state: int = typer.Option(42, help="Random seed for reproducibility")
) -> None:
    """Generate synthetic clinical datasets for testing."""
    run_synthetic_data_command(scenario, n_samples, output_dir, random_state)


@app.command()
def performance_regression(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True, help="Configuration file"),
    baseline_file: Path = typer.Option(Path("tests/baseline_performance.json"), help="Baseline performance file"),
    tolerance: float = typer.Option(0.05, help="Performance tolerance for regression detection"),
    output_dir: Path = typer.Option(Path("tests/performance_regression"), help="Output directory for results")
) -> None:
    """Run automated performance regression testing."""
    run_performance_regression_command(config, baseline_file, tolerance, output_dir)


@app.command()
def cv_integrity(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True, help="Configuration file"),
    cv_folds: int = typer.Option(5, help="Number of CV folds to test"),
    output_dir: Path = typer.Option(Path("tests/cv_integrity"), help="Output directory for results")
) -> None:
    """Check cross-validation integrity and detect data leakage."""
    run_cv_integrity_command(config, cv_folds, output_dir)


@app.command()
def benchmark_suite(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True, help="Configuration file"),
    output_dir: Path = typer.Option(Path("tests/benchmark_results"), help="Output directory for results"),
    include_sksurv: bool = typer.Option(True, help="Include scikit-survival benchmarks"),
    include_lifelines: bool = typer.Option(True, help="Include lifelines benchmarks")
) -> None:
    """Run comprehensive benchmark against other survival libraries."""
    run_benchmark_suite_command(config, output_dir, include_sksurv, include_lifelines)


if __name__ == "__main__":  # pragma: no cover
    app()
