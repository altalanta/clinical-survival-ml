from pathlib import Path

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from rich.console import Console

console = Console()


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    column_mapping: dict,
    output_path: Path,
):
    """
    Generates a data and concept drift report using Evidently.

    Args:
        reference_df: The baseline dataset (e.g., training data).
        current_df: The new dataset to compare (e.g., recent production data).
        column_mapping: A dictionary defining the roles of columns (target, prediction, etc.).
        output_path: The file path to save the HTML report.
    """
    console.print("Generating data and concept drift report...", style="cyan")

    if not all(col in reference_df.columns for col in current_df.columns):
        console.print("[bold red]Error: Reference and current dataframes must have the same columns.[/bold red]")
        return

    # Create a report with two main presets:
    # 1. DataDriftPreset: Checks for drift in input features.
    # 2. TargetDriftPreset: Checks for drift in the target variable (concept drift).
    drift_report = Report(
        metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
        ]
    )

    try:
        drift_report.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=column_mapping,
        )

        # Save the report as an HTML file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        drift_report.save_html(str(output_path))

        console.print(
            f"✅ Drift report saved successfully to [bold green]{output_path}[/bold green]"
        )

    except Exception as e:
        console.print(f"[bold red]Failed to generate drift report: {e}[/bold red]")



