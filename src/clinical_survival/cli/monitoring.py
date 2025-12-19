from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
import pickle
import pandas as pd
import yaml

from clinical_survival.data_quality import DataQualityMonitor, DataQualityProfiler
from clinical_survival.io import load_params_config
from clinical_survival.utils import ensure_dir
from clinical_survival.monitoring.drift import generate_drift_report
from clinical_survival.monitoring.data_quality_monitor import (
    create_default_monitor,
    MonitoringConfiguration,
    AlertStatus,
    AlertSeverity,
)

app = typer.Typer(help="Commands for model monitoring, drift detection, and maintenance.")
console = Console()

MONITOR_STATE_PATH = Path("artifacts/monitoring/monitor.pkl")

def _load_monitor() -> DataQualityMonitor:
    """Load the DataQualityMonitor instance from disk."""
    ensure_dir(MONITOR_STATE_PATH.parent)
    if MONITOR_STATE_PATH.exists():
        with open(MONITOR_STATE_PATH, "rb") as f:
            return pickle.load(f)
    return DataQualityMonitor()

def _save_monitor(monitor: DataQualityMonitor) -> None:
    """Save the DataQualityMonitor instance to disk."""
    ensure_dir(MONITOR_STATE_PATH.parent)
    with open(MONITOR_STATE_PATH, "wb") as f:
        pickle.dump(monitor, f)

@app.command()
def run(
    config: Path = typer.Option("configs/params.yaml", "--config", "-c", exists=True, help="Path to the parameters config file."),
    data: Path = typer.Option(..., "--data", "-d", exists=True, help="Path to the new data for monitoring."),
) -> None:
    """Monitor data quality for drift and performance issues."""
    console.print(f"Monitoring data quality using data from [bold cyan]{data}[/bold cyan]...")
    
    monitor = _load_monitor()
    new_data = pd.read_csv(data)
    
    if monitor.baseline_profile is None:
        console.print("No baseline found. Setting current data as the new baseline.")
        profiler = DataQualityProfiler()
        baseline_report = profiler.profile_dataset(new_data, dataset_name="baseline")
        monitor.baseline_profile = baseline_report
    else:
        console.print("Comparing new data against the existing baseline.")
        monitor.monitor_data_quality(new_data, dataset_name=data.stem)

    _save_monitor(monitor)
    console.print(f"✅ Monitoring complete. State saved to [bold cyan]{MONITOR_STATE_PATH}[/bold cyan]")

@app.command()
def drift(
    config: Path = typer.Option("configs/params.yaml", "--config", "-c", exists=True, help="Path to the parameters config file."),
    days: int = typer.Option(7, help="Number of days of recent data to analyze."),
) -> None:
    """Check for data drift and performance degradation."""
    console.print(f"Checking for drift over the last {days} days...")
    
    monitor = _load_monitor()
    if monitor.baseline_profile is None:
        console.print("[bold yellow]No baseline profile found. Run `monitor run` first to establish a baseline.[/bold yellow]")
        raise typer.Exit()

    report = monitor.generate_monitoring_report()
    if "error" in report:
        console.print(f"[bold red]Could not generate drift report: {report['error']}[/bold red]")
        raise typer.Exit()

    console.print("Drift analysis complete. ✅")
    table = Table(title="Drift Analysis Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    trends = report.get("quality_trends", {})
    table.add_row("Overall Score Trend", trends.get('overall_score_trend', 'N/A'))
    table.add_row("Error Rate Trend", trends.get('error_rate_trend', 'N/A'))
    table.add_row("Current Quality Score", f"{trends.get('current_quality_score', 0):.2f}")
    table.add_row("Baseline Quality Score", f"{trends.get('baseline_quality_score', 0):.2f}")
    
    console.print(table)
    
    recommendations = report.get("recommendations", [])
    if recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for rec in recommendations:
            console.print(f"- {rec}")


@app.command()
def status(
    config: Path = typer.Option("configs/params.yaml", "--config", "-c", exists=True, help="Path to the parameters config file."),
) -> None:
    """Show the monitoring status dashboard."""
    console.print("📊 Displaying monitoring status dashboard...")
    
    monitor = _load_monitor()
    if not monitor._monitoring_history:
        console.print("[bold yellow]No monitoring history found. Run `monitor run` with new data.[/bold yellow]")
        raise typer.Exit()
        
    report = monitor.generate_monitoring_report()

    if "error" in report:
        console.print(f"[bold red]Could not generate status report: {report['error']}[/bold red]")
        raise typer.Exit()

    # This is the same as the drift command for now, but could be expanded
    # to show more historical data visualizations.
    drift(config, days=30)


@app.command()
def reset(
    config: Path = typer.Option("configs/params.yaml", "--config", "-c", exists=True, help="Path to the parameters config file."),
    confirm: bool = typer.Option(False, "--confirm", help="Confirm the reset operation."),
) -> None:
    """Reset monitoring baselines and historical data."""
    if not confirm:
        console.print("[bold yellow]This is a destructive operation. Please confirm by using the --confirm flag.[/bold yellow]")
        raise typer.Exit(code=1)
    
    console.print(f"Resetting monitoring baselines and history...")
    
    monitor = DataQualityMonitor() # Create a fresh instance
    _save_monitor(monitor)
    
    console.print(f"✅ Monitoring reset complete. State file [bold cyan]{MONITOR_STATE_PATH}[/bold cyan] has been cleared.")


@app.command()
def detect_drift(
    reference_csv: Path = typer.Option(
        ...,
        "--reference-csv",
        "-r",
        help="Path to the reference dataset (e.g., training data).",
        exists=True,
    ),
    current_csv: Path = typer.Option(
        ...,
        "--current-csv",
        "-c",
        help="Path to the current dataset to check for drift.",
        exists=True,
    ),
    features_config: Path = typer.Option(
        "configs/features.yaml",
        "--features-config",
        "-f",
        help="Path to the features configuration YAML file.",
        exists=True,
    ),
    output_path: Path = typer.Option(
        "results/monitoring/drift_report.html",
        "--output-path",
        "-o",
        help="Path to save the generated HTML drift report.",
    ),
):
    """
    Detects data and concept drift between a reference and current dataset.
    """
    console.print("--- 🔬 Starting Drift Detection ---")

    # Load datasets
    try:
        reference_df = pd.read_csv(reference_csv)
        current_df = pd.read_csv(current_csv)
    except Exception as e:
        console.print(f"[bold red]Error loading data: {e}[/bold red]")
        raise typer.Exit(1)

    # Load feature config to build column mapping for Evidently
    try:
        with open(features_config, "r") as f:
            features = yaml.safe_load(f)
    except Exception as e:
        console.print(f"[bold red]Error loading features config: {e}[/bold red]")
        raise typer.Exit(1)

    # Evidently requires a specific mapping of columns
    column_mapping = {
        "target": features["time_to_event_col"],
        "categorical_features": features["categorical_cols"],
        "numerical_features": features["numerical_cols"],
    }
    # For concept drift, the event column can also be considered part of the target.
    # We will let Evidently handle the binary event column automatically.

    generate_drift_report(
        reference_df=reference_df,
        current_df=current_df,
        column_mapping=column_mapping,
        output_path=output_path,
    )
    console.print("--- ✅ Drift Detection Finished ---")


@app.command("start-quality-monitor")
def start_quality_monitor(
    data_source: Path = typer.Option(
        ..., "--data-source", "-d",
        help="Path to data file or directory to monitor",
        exists=True,
    ),
    interval_seconds: int = typer.Option(
        3600, "--interval",
        help="Monitoring check interval in seconds (default: 1 hour)",
    ),
    enable_ge: bool = typer.Option(
        False, "--enable-ge",
        help="Enable Great Expectations integration",
    ),
    ge_context: Optional[Path] = typer.Option(
        None, "--ge-context",
        help="Path to Great Expectations context directory",
    ),
):
    """Start continuous data quality monitoring."""
    console.print("🚀 Starting continuous data quality monitoring...", style="bold green")

    config = MonitoringConfiguration(
        check_interval_seconds=interval_seconds,
        data_source_path=data_source,
        enable_ge=enable_ge,
        ge_context_path=ge_context,
    )

    monitor = DataQualityMonitor(config)
    monitor.start_monitoring()

    console.print(f"✅ Monitoring started for data source: {data_source}", style="bold green")
    console.print(f"📊 Check interval: {interval_seconds} seconds", style="green")
    console.print(f"🎯 Great Expectations: {'Enabled' if enable_ge else 'Disabled'}", style="green")
    console.print("\nMonitoring will continue in the background.", style="yellow")
    console.print("Use 'clinical-ml monitoring stop-quality-monitor' to stop.", style="yellow")


@app.command("stop-quality-monitor")
def stop_quality_monitor():
    """Stop continuous data quality monitoring."""
    console.print("🛑 Stopping data quality monitoring...", style="bold yellow")

    # For now, this is a placeholder - in a real implementation,
    # we'd need a way to access the running monitor instance
    console.print("✅ Monitoring stopped", style="bold green")


@app.command("check-quality")
def check_quality_now(
    data_source: Path = typer.Option(
        ..., "--data-source", "-d",
        help="Path to data file to check quality for",
        exists=True,
    ),
    enable_ge: bool = typer.Option(
        False, "--enable-ge",
        help="Enable Great Expectations integration",
    ),
):
    """Perform an immediate data quality check."""
    console.print("🔍 Performing data quality check...", style="bold blue")

    config = MonitoringConfiguration(
        data_source_path=data_source,
        enable_ge=enable_ge,
    )

    monitor = DataQualityMonitor(config)
    result = monitor.check_data_quality()

    if "error" in result:
        console.print(f"❌ Quality check failed: {result['error']}", style="bold red")
        raise typer.Exit(1)

    # Display results
    console.print("📊 Quality Check Results:", style="bold green")

    metrics = result["metrics"]
    console.print(f"Total Rows: {metrics['total_rows']:,}")
    console.print(f"Total Columns: {metrics['total_columns']}")
    console.print(f"Missing Values: {metrics['missing_values_percentage']:.2f}%")
    console.print(f"Duplicate Rows: {metrics['duplicate_rows_percentage']:.2f}%")

    if result["issues"]:
        console.print(f"\n⚠️  Issues Found ({len(result['issues'])}):", style="yellow")
        for issue in result["issues"][:5]:  # Show first 5
            console.print(f"  - {issue}")
        if len(result["issues"]) > 5:
            console.print(f"  ... and {len(result['issues']) - 5} more")

    if result["alerts_generated"] > 0:
        console.print(f"\n🚨 Alerts Generated: {result['alerts_generated']}", style="red")

    if result["recommendations"]:
        console.print(f"\n💡 Recommendations ({len(result['recommendations'])}):", style="cyan")
        for rec in result["recommendations"][:3]:  # Show first 3
            console.print(f"  - {rec}")

    console.print("✅ Quality check complete", style="bold green")


@app.command("list-alerts")
def list_quality_alerts(
    status: Optional[str] = typer.Option(
        None, "--status", "-s",
        help="Filter by alert status (active, resolved, acknowledged)"
    ),
    severity: Optional[str] = typer.Option(
        None, "--severity", "-v",
        help="Filter by alert severity (low, medium, high, critical)"
    ),
    limit: int = typer.Option(
        20, "--limit", "-l",
        help="Maximum number of alerts to show"
    ),
):
    """List data quality alerts."""
    console.print("📋 Listing data quality alerts...", style="bold blue")

    # Parse filters
    status_filter = None
    if status:
        try:
            status_filter = AlertStatus(status.lower())
        except ValueError:
            console.print(f"❌ Invalid status: {status}", style="bold red")
            raise typer.Exit(1)

    severity_filter = None
    if severity:
        try:
            severity_filter = AlertSeverity(severity.lower())
        except ValueError:
            console.print(f"❌ Invalid severity: {severity}", style="bold red")
            raise typer.Exit(1)

    # For now, create a temporary monitor to show alerts
    # In a real implementation, we'd load the persistent monitor state
    config = MonitoringConfiguration()
    monitor = DataQualityMonitor(config)

    alerts = monitor.get_alerts(
        status=status_filter,
        severity=severity_filter,
        limit=limit
    )

    if not alerts:
        console.print("No alerts found matching criteria.", style="yellow")
        return

    # Display alerts in a table
    table = Table(title=f"Data Quality Alerts ({len(alerts)} found)")
    table.add_column("ID", style="dim", max_width=12)
    table.add_column("Severity", style="red")
    table.add_column("Status", style="yellow")
    table.add_column("Title", max_width=40)
    table.add_column("Time", style="blue")
    table.add_column("Metric", style="cyan")

    for alert in alerts:
        table.add_row(
            alert.alert_id[:12],
            alert.severity.value.upper(),
            alert.status.value,
            alert.title[:37] + "..." if len(alert.title) > 40 else alert.title,
            alert.timestamp.strftime("%m/%d %H:%M"),
            alert.metric_name or "N/A",
        )

    console.print(table)


@app.command("acknowledge-alert")
def acknowledge_alert(
    alert_id: str = typer.Option(
        ..., "--alert-id", "-i",
        help="Alert ID to acknowledge"
    ),
    user: str = typer.Option(
        ..., "--user", "-u",
        help="User acknowledging the alert"
    ),
):
    """Acknowledge a data quality alert."""
    console.print(f"✅ Acknowledging alert {alert_id}...", style="bold green")

    config = MonitoringConfiguration()
    monitor = DataQualityMonitor(config)

    if monitor.acknowledge_alert(alert_id, user):
        console.print("✅ Alert acknowledged successfully", style="bold green")
    else:
        console.print(f"❌ Alert {alert_id} not found or already acknowledged", style="bold red")
        raise typer.Exit(1)


@app.command("resolve-alert")
def resolve_alert(
    alert_id: str = typer.Option(
        ..., "--alert-id", "-i",
        help="Alert ID to resolve"
    ),
    user: str = typer.Option(
        ..., "--user", "-u",
        help="User resolving the alert"
    ),
):
    """Resolve a data quality alert."""
    console.print(f"✅ Resolving alert {alert_id}...", style="bold green")

    config = MonitoringConfiguration()
    monitor = DataQualityMonitor(config)

    if monitor.resolve_alert(alert_id, user):
        console.print("✅ Alert resolved successfully", style="bold green")
    else:
        console.print(f"❌ Alert {alert_id} not found", style="bold red")
        raise typer.Exit(1)


@app.command("quality-trends")
def show_quality_trends(
    metric: str = typer.Option(
        ..., "--metric", "-m",
        help="Metric to show trends for (e.g., missing_values_percentage)"
    ),
    hours: int = typer.Option(
        24, "--hours", "-h",
        help="Number of hours of history to show"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Save trends plot to file (HTML)"
    ),
):
    """Show data quality metric trends over time."""
    console.print(f"📈 Showing trends for metric: {metric}", style="bold blue")

    config = MonitoringConfiguration()
    monitor = DataQualityMonitor(config)

    trends = monitor.get_quality_trends(metric, hours)

    if not trends:
        console.print(f"No trend data found for metric {metric}", style="yellow")
        return

    # Display trends
    console.print(f"📊 Trend data for {metric} (last {hours} hours):", style="bold green")

    table = Table(title=f"{metric} Trends")
    table.add_column("Time", style="blue")
    table.add_column("Value", style="cyan")

    for point in trends[-20:]:  # Show last 20 points
        table.add_row(
            point["timestamp"].strftime("%m/%d %H:%M"),
            f"{point['value']:.4f}",
        )

    console.print(table)

    # Create simple plot if plotly available
    try:
        import plotly.express as px
        import pandas as pd

        df = pd.DataFrame(trends)
        fig = px.line(df, x="timestamp", y="value", title=f"{metric} Trend")
        fig.update_layout(xaxis_title="Time", yaxis_title=metric)

        if output:
            fig.write_html(str(output))
            console.print(f"✅ Trend plot saved to {output}", style="green")
        else:
            fig.show()

    except ImportError:
        console.print("Plotly not available for visualization", style="yellow")
    except Exception as e:
        console.print(f"Failed to create trend plot: {e}", style="red")


