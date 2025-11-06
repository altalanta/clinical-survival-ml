from pathlib import Path
import typer
from rich.console import Console
import pandas as pd
import yaml

from clinical_survival.data.synthetic import simulate_survival
from clinical_survival.data_quality import DataQualityProfiler, save_data_quality_report
from clinical_survival.utils import ensure_dir

app = typer.Typer(help="Commands for data generation and quality assurance.")
console = Console()

@app.command()
def synthetic_data(
    scenario: str = typer.Option("icu", help="Type of synthetic data to generate (icu, cancer, cardiovascular)"),
    n_samples: int = typer.Option(1000, help="Number of samples to generate"),
    output_dir: Path = typer.Option(Path("data/synthetic"), help="Output directory for generated data"),
    random_state: int = typer.Option(42, help="Random seed for reproducibility")
) -> None:
    """Generate synthetic clinical datasets for testing."""
    console.print(f"Generating {n_samples} samples for '{scenario}' scenario...")
    
    ensure_dir(output_dir)
    df, meta = simulate_survival(n=n_samples, seed=random_state)
    
    csv_path = output_dir / f"{scenario}_synthetic_data.csv"
    meta_path = output_dir / f"{scenario}_metadata.yaml"
    
    df.to_csv(csv_path, index=False)
    with open(meta_path, 'w') as f:
        yaml.dump(meta, f)

    console.print(f"✅ Synthetic data saved to [bold cyan]{csv_path}[/bold cyan]")
    console.print(f"✅ Metadata saved to [bold cyan]{meta_path}[/bold cyan]")

@app.command()
def quality_profile(
    data: Path = typer.Option(..., "--data", "-d", exists=True, help="Path to the dataset CSV file."),
    meta: Path = typer.Option(..., "--meta", "-m", exists=True, help="Path to the metadata YAML file."),
    output_dir: Path = typer.Option(Path("results/data_quality"), help="Output directory for the report."),
    output_format: str = typer.Option("html", help="Output format (html, json)."),
) -> None:
    """Generate a comprehensive data quality profile report."""
    console.print(f"Profiling data from [bold cyan]{data}[/bold cyan]...")
    
    df = pd.read_csv(data)
    with open(meta, 'r') as f:
        metadata = yaml.safe_load(f)

    profiler = DataQualityProfiler()
    report = profiler.profile_dataset(df, dataset_name=data.stem, clinical_context=metadata)
    
    ensure_dir(output_dir)
    report_path = output_dir / f"{data.stem}_quality_report.{output_format}"
    save_data_quality_report(report, report_path, format=output_format)

    console.print(f"✅ Data quality report saved to [bold cyan]{report_path}[/bold cyan]")

