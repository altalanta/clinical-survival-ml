"""
MLOps CLI commands for model registry, deployment, and monitoring.

This module provides CLI commands for:
- Model registry operations (register, promote, approve)
- Model deployment and serving
- Production monitoring and alerting
- Model governance and audit trails
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from clinical_survival.model_registry import (
    get_registry,
    initialize_registry,
    ModelStage,
    ApprovalStatus,
)
from clinical_survival.tracking import (
    MLflowTracker,
    compare_experiments,
    create_performance_dashboard,
)

app = typer.Typer(help="MLOps commands for model registry, deployment, and monitoring.")
console = Console()


@app.callback()
def mlops_callback():
    """MLOps operations for model lifecycle management."""
    pass


@app.command("init-registry")
def init_registry(
    registry_path: Path = typer.Option(
        Path("results/model_registry"),
        "--path",
        help="Path to initialize the model registry"
    ),
    enable_auto_approval: bool = typer.Option(
        False,
        "--auto-approve",
        help="Enable automatic approval for models meeting performance thresholds"
    ),
    require_production_approval: bool = typer.Option(
        True,
        "--require-prod-approval",
        help="Require manual approval for production deployments"
    ),
):
    """Initialize a new model registry."""
    console.print("🔧 Initializing model registry...", style="bold blue")

    registry = initialize_registry(
        registry_root=registry_path,
        enable_auto_approval=enable_auto_approval,
        require_approval_for_production=require_production_approval,
    )

    console.print(f"✅ Model registry initialized at {registry_path}", style="bold green")
    console.print(f"Auto-approval: {'Enabled' if enable_auto_approval else 'Disabled'}")
    console.print(f"Production approval required: {'Yes' if require_production_approval else 'No'}")


@app.command("register")
def register_model(
    model_path: Path = typer.Option(
        ..., "--model", "-m", exists=True,
        help="Path to the trained model file (.joblib)"
    ),
    model_name: str = typer.Option(
        ..., "--name", "-n",
        help="Name of the model"
    ),
    created_by: str = typer.Option(
        ..., "--created-by", "-c",
        help="User or system registering the model"
    ),
    description: str = typer.Option(
        "",
        "--description", "-d",
        help="Description of the model"
    ),
    metrics_file: Optional[Path] = typer.Option(
        None,
        "--metrics",
        help="Path to JSON file containing model metrics"
    ),
    registry_path: Path = typer.Option(
        Path("results/model_registry"),
        "--registry",
        help="Path to the model registry"
    ),
):
    """Register a trained model in the registry."""
    console.print(f"📝 Registering model {model_name}...", style="bold blue")

    # Initialize registry
    registry = initialize_registry(registry_path)

    # Load metrics if provided
    metadata = {}
    if metrics_file and metrics_file.exists():
        import json
        with open(metrics_file, 'r') as f:
            metadata["metrics"] = json.load(f)
        console.print(f"📊 Loaded metrics from {metrics_file}")

    # Register the model
    try:
        version = registry.register_model(
            model_name=model_name,
            model_path=model_path,
            metadata=metadata,
            created_by=created_by,
            description=description,
        )

        console.print("✅ Model registered successfully!", style="bold green")
        console.print(f"Model: {version.model_name} {version.version}")
        console.print(f"Stage: {version.stage.value}")
        console.print(f"Approval Status: {version.approval_status.value}")

        if version.metrics:
            console.print("📈 Metrics:")
            for metric, value in version.metrics.items():
                console.print(f"  {metric}: {value:.4f}")

    except Exception as e:
        console.print(f"❌ Failed to register model: {e}", style="bold red")
        raise typer.Exit(1)


@app.command("promote")
def promote_model(
    model_name: str = typer.Option(
        ..., "--name", "-n",
        help="Name of the model to promote"
    ),
    version: str = typer.Option(
        ..., "--version", "-v",
        help="Version of the model to promote"
    ),
    stage: str = typer.Option(
        ..., "--stage", "-s",
        help="Target stage (development, staging, production, archived)"
    ),
    promoted_by: str = typer.Option(
        ..., "--promoted-by", "-p",
        help="User promoting the model"
    ),
    notes: str = typer.Option(
        "",
        "--notes",
        help="Notes about the promotion"
    ),
    registry_path: Path = typer.Option(
        Path("results/model_registry"),
        "--registry",
        help="Path to the model registry"
    ),
):
    """Promote a model to a different stage."""
    console.print(f"⬆️  Promoting {model_name} {version} to {stage}...", style="bold blue")

    # Validate stage
    try:
        target_stage = ModelStage(stage.lower())
    except ValueError:
        console.print(f"❌ Invalid stage: {stage}. Must be one of: {[s.value for s in ModelStage]}", style="bold red")
        raise typer.Exit(1)

    # Initialize registry
    registry = initialize_registry(registry_path)

    try:
        registry.promote_model(
            model_name=model_name,
            version=version,
            target_stage=target_stage,
            promoted_by=promoted_by,
            notes=notes,
        )

        console.print("✅ Model promoted successfully!", style="bold green")

    except Exception as e:
        console.print(f"❌ Failed to promote model: {e}", style="bold red")
        raise typer.Exit(1)


@app.command("approve")
def approve_model(
    model_name: str = typer.Option(
        ..., "--name", "-n",
        help="Name of the model to approve"
    ),
    version: str = typer.Option(
        ..., "--version", "-v",
        help="Version of the model to approve"
    ),
    approved_by: str = typer.Option(
        ..., "--approved-by", "-a",
        help="User approving the model"
    ),
    notes: str = typer.Option(
        "",
        "--notes",
        help="Approval notes"
    ),
    registry_path: Path = typer.Option(
        Path("results/model_registry"),
        "--registry",
        help="Path to the model registry"
    ),
):
    """Approve a model for promotion."""
    console.print(f"✅ Approving {model_name} {version}...", style="bold green")

    registry = initialize_registry(registry_path)

    try:
        registry.approve_model(
            model_name=model_name,
            version=version,
            approved_by=approved_by,
            notes=notes,
        )

        console.print("✅ Model approved successfully!", style="bold green")

    except Exception as e:
        console.print(f"❌ Failed to approve model: {e}", style="bold red")
        raise typer.Exit(1)


@app.command("reject")
def reject_model(
    model_name: str = typer.Option(
        ..., "--name", "-n",
        help="Name of the model to reject"
    ),
    version: str = typer.Option(
        ..., "--version", "-v",
        help="Version of the model to reject"
    ),
    rejected_by: str = typer.Option(
        ..., "--rejected-by", "-r",
        help="User rejecting the model"
    ),
    notes: str = typer.Option(
        "",
        "--notes",
        help="Rejection notes"
    ),
    registry_path: Path = typer.Option(
        Path("results/model_registry"),
        "--registry",
        help="Path to the model registry"
    ),
):
    """Reject a model for promotion."""
    console.print(f"❌ Rejecting {model_name} {version}...", style="bold red")

    registry = initialize_registry(registry_path)

    try:
        registry.reject_model(
            model_name=model_name,
            version=version,
            rejected_by=rejected_by,
            notes=notes,
        )

        console.print("✅ Model rejected successfully!", style="bold green")

    except Exception as e:
        console.print(f"❌ Failed to reject model: {e}", style="bold red")
        raise typer.Exit(1)


@app.command("list")
def list_models(
    model_name: Optional[str] = typer.Option(
        None,
        "--name", "-n",
        help="Filter by model name"
    ),
    stage: Optional[str] = typer.Option(
        None,
        "--stage", "-s",
        help="Filter by stage (development, staging, production, archived, deprecated)"
    ),
    registry_path: Path = typer.Option(
        Path("results/model_registry"),
        "--registry",
        help="Path to the model registry"
    ),
):
    """List models in the registry."""
    registry = initialize_registry(registry_path)

    # Validate stage filter
    stage_filter = None
    if stage:
        try:
            stage_filter = ModelStage(stage.lower())
        except ValueError:
            console.print(f"❌ Invalid stage: {stage}. Must be one of: {[s.value for s in ModelStage]}", style="bold red")
            raise typer.Exit(1)

    models = registry.list_models(stage=stage_filter, model_name=model_name)

    if not models:
        console.print("No models found matching the criteria.", style="yellow")
        return

    # Display results in a table
    from rich.table import Table

    table = Table(title=f"Model Registry ({len(models)} versions)")
    table.add_column("Model", style="cyan")
    table.add_column("Version", style="magenta")
    table.add_column("Stage", style="green")
    table.add_column("Approval", style="yellow")
    table.add_column("Created", style="blue")
    table.add_column("Created By", style="white")
    table.add_column("Health", style="red")

    for model in models:
        approval_status = model.approval_status.value
        if model.approval_status == ApprovalStatus.AUTO_APPROVED:
            approval_status = "auto"

        health_display = model.health_status
        if model.health_score is not None:
            health_display = ".1f"

        table.add_row(
            model.model_name,
            model.version,
            model.stage.value,
            approval_status,
            model.created_at.strftime("%Y-%m-%d"),
            model.created_by,
            health_display,
        )

    console.print(table)


@app.command("health-check")
def check_model_health(
    model_name: str = typer.Option(
        ..., "--name", "-n",
        help="Name of the model to check"
    ),
    version: str = typer.Option(
        ..., "--version", "-v",
        help="Version of the model to check"
    ),
    registry_path: Path = typer.Option(
        Path("results/model_registry"),
        "--registry",
        help="Path to the model registry"
    ),
):
    """Check the health of a registered model."""
    console.print(f"🏥 Checking health of {model_name} {version}...", style="bold blue")

    registry = initialize_registry(registry_path)

    try:
        health = registry.check_model_health(model_name, version)

        console.print("📊 Health Check Results:", style="bold blue")
        console.print(f"Status: {health['status']}")
        console.print(f"Score: {health['score']:.2f}")
        console.print(f"Last Check: {health['last_check']}")

        if health['issues']:
            console.print("⚠️  Issues Found:", style="yellow")
            for issue in health['issues']:
                console.print(f"  - {issue}")
        else:
            console.print("✅ No issues found", style="green")

    except Exception as e:
        console.print(f"❌ Health check failed: {e}", style="bold red")
        raise typer.Exit(1)


@app.command("report")
def generate_registry_report(
    registry_path: Path = typer.Option(
        Path("results/model_registry"),
        "--registry",
        help="Path to the model registry"
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file for the report (prints to console if not specified)"
    ),
):
    """Generate a comprehensive registry report."""
    console.print("📋 Generating registry report...", style="bold blue")

    registry = initialize_registry(registry_path)

    try:
        report = registry.generate_report()

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            console.print(f"✅ Report saved to {output_file}", style="bold green")
        else:
            console.print(report)

    except Exception as e:
        console.print(f"❌ Failed to generate report: {e}", style="bold red")
        raise typer.Exit(1)


@app.command("cleanup")
def cleanup_registry(
    registry_path: Path = typer.Option(
        Path("results/model_registry"),
        "--registry",
        help="Path to the model registry"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be cleaned up without actually doing it"
    ),
):
    """Clean up old model versions based on retention policy."""
    action = "Analyzing" if dry_run else "Cleaning up"
    console.print(f"🧹 {action} old model versions...", style="bold blue")

    registry = initialize_registry(registry_path)

    try:
        deleted_count = registry.cleanup_old_versions()

        if dry_run:
            console.print(f"Would archive {deleted_count} old versions")
        else:
            console.print(f"✅ Archived {deleted_count} old versions", style="bold green")

    except Exception as e:
        console.print(f"❌ Cleanup failed: {e}", style="bold red")
        raise typer.Exit(1)


@app.command("get-latest")
def get_latest_model(
    model_name: str = typer.Option(
        ..., "--name", "-n",
        help="Name of the model"
    ),
    stage: Optional[str] = typer.Option(
        None,
        "--stage", "-s",
        help="Stage to get latest from (defaults to highest stage)"
    ),
    registry_path: Path = typer.Option(
        Path("results/model_registry"),
        "--registry",
        help="Path to the model registry"
    ),
):
    """Get the latest version of a model."""
    registry = initialize_registry(registry_path)

    # Validate stage
    stage_filter = None
    if stage:
        try:
            stage_filter = ModelStage(stage.lower())
        except ValueError:
            console.print(f"❌ Invalid stage: {stage}. Must be one of: {[s.value for s in ModelStage]}", style="bold red")
            raise typer.Exit(1)

    latest = registry.get_latest_version(model_name, stage_filter)

    if not latest:
        console.print(f"No versions found for model {model_name}", style="yellow")
        if stage:
            console.print(f"in stage {stage}", style="yellow")
        return

    console.print(f"📦 Latest version of {model_name}:", style="bold green")
    console.print(f"Version: {latest.version}")
    console.print(f"Stage: {latest.stage.value}")
    console.print(f"Created: {latest.created_at}")
    console.print(f"Created By: {latest.created_by}")
    console.print(f"Approval: {latest.approval_status.value}")

    if latest.description:
        console.print(f"Description: {latest.description}")

    if latest.metrics:
        console.print("📈 Metrics:")
        for metric, value in latest.metrics.items():
            console.print(f"  {metric}: {value:.4f}")

    if latest.model_path:
        console.print(f"Model Path: {latest.model_path}")


@app.command("compare-experiments")
def compare_experiments_cmd(
    experiment_ids: str = typer.Option(
        ..., "--experiments", "-e",
        help="Comma-separated list of MLflow experiment IDs to compare"
    ),
    metrics: Optional[str] = typer.Option(
        None,
        "--metrics", "-m",
        help="Comma-separated list of metrics to compare (default: all)"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output path for comparison results JSON"
    ),
    plot: Optional[Path] = typer.Option(
        None,
        "--plot", "-p",
        help="Output path for comparison visualization HTML"
    ),
):
    """Compare multiple MLflow experiments with statistical analysis."""
    console.print("🔍 Comparing experiments...", style="bold blue")

    # Parse experiment IDs
    exp_ids = [eid.strip() for eid in experiment_ids.split(",")]

    # Parse metrics
    metric_list = None
    if metrics:
        metric_list = [m.strip() for m in metrics.split(",")]

    try:
        # Initialize tracker
        tracker_config = {}  # Would load from config in real implementation
        tracker = MLflowTracker(tracker_config)

        # Compare experiments
        comparison = compare_experiments(tracker, exp_ids, metrics=metric_list)

        if "error" in comparison:
            console.print(f"❌ Comparison failed: {comparison['error']}", style="bold red")
            raise typer.Exit(1)

        # Display results
        console.print("📊 Comparison Results:", style="bold green")
        console.print(f"Runs compared: {comparison['runs_count']}")
        console.print(f"Experiments: {comparison['experiments_count']}")

        if "summary" in comparison and "best_performers" in comparison["summary"]:
            console.print("\n🏆 Best Performers:")
            for metric, best in comparison["summary"]["best_performers"].items():
                console.print(f"  {metric}: {best['value']:.4f} (run: {best['run_id'][:8]}...)")

        # Save results
        if output:
            import json
            with open(output, 'w') as f:
                json.dump(comparison, f, indent=2, default=str)
            console.print(f"✅ Results saved to {output}", style="green")

        # Generate plot
        if plot:
            from clinical_survival.tracking import ExperimentComparator
            comparator = ExperimentComparator(tracker)
            comparator.plot_comparison(comparison, plot)
            console.print(f"✅ Plot saved to {plot}", style="green")

    except Exception as e:
        console.print(f"❌ Comparison failed: {e}", style="bold red")
        raise typer.Exit(1)


@app.command("dashboard")
def create_experiment_dashboard(
    experiment_id: str = typer.Option(
        ..., "--experiment", "-e",
        help="MLflow experiment ID to create dashboard for"
    ),
    output: Path = typer.Option(
        Path("results/experiment_dashboard.html"),
        "--output", "-o",
        help="Output path for dashboard HTML"
    ),
    metrics: Optional[str] = typer.Option(
        None,
        "--metrics", "-m",
        help="Comma-separated list of metrics to include (default: all)"
    ),
):
    """Create an interactive performance dashboard for an experiment."""
    console.print(f"📊 Creating dashboard for experiment {experiment_id}...", style="bold blue")

    try:
        # Initialize tracker
        tracker_config = {}  # Would load from config in real implementation
        tracker = MLflowTracker(tracker_config)

        # Parse metrics
        metric_list = None
        if metrics:
            metric_list = [m.strip() for m in metrics.split(",")]

        # Create dashboard
        dashboard_path = create_performance_dashboard(
            tracker,
            experiment_id,
            metrics=metric_list,
            output_path=output
        )

        if dashboard_path:
            console.print(f"✅ Dashboard created: {dashboard_path}", style="bold green")
            console.print("Open the HTML file in your browser to view the interactive dashboard.", style="green")
        else:
            console.print("❌ Failed to create dashboard", style="bold red")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"❌ Dashboard creation failed: {e}", style="bold red")
        raise typer.Exit(1)


@app.command("plot-trends")
def plot_metric_trends(
    experiment_ids: str = typer.Option(
        ..., "--experiments", "-e",
        help="Comma-separated list of MLflow experiment IDs"
    ),
    metrics: str = typer.Option(
        ..., "--metrics", "-m",
        help="Comma-separated list of metrics to plot"
    ),
    output: Path = typer.Option(
        Path("results/metric_trends.html"),
        "--output", "-o",
        help="Output path for trends plot HTML"
    ),
):
    """Plot metric trends across multiple experiments."""
    console.print("📈 Plotting metric trends...", style="bold blue")

    # Parse inputs
    exp_ids = [eid.strip() for eid in experiment_ids.split(",")]
    metric_list = [m.strip() for m in metrics.split(",")]

    try:
        # Initialize tracker
        tracker_config = {}
        tracker = MLflowTracker(tracker_config)

        # Create trend plot
        from clinical_survival.tracking import ExperimentVisualizer
        visualizer = ExperimentVisualizer(tracker)

        plot_path = visualizer.plot_metric_trends(exp_ids, metric_list, output)

        if plot_path:
            console.print(f"✅ Trend plot created: {plot_path}", style="bold green")
        else:
            console.print("❌ Failed to create trend plot", style="bold red")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"❌ Trend plotting failed: {e}", style="bold red")
        raise typer.Exit(1)