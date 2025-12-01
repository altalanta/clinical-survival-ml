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

