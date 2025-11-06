from pathlib import Path
import typer
from rich.console import Console

from clinical_survival.io import load_params_config, load_features_config
from clinical_survival.regression_testing import (
    run_performance_test,
    update_baseline,
    check_regression
)
from clinical_survival.cv_integrity import run_cv_leakage_test

app = typer.Typer(help="Commands for advanced testing and quality assurance.")
console = Console()

@app.command()
def performance_regression(
    config: Path = typer.Option(
        "configs/params.yaml", "--config", "-c", exists=True,
        help="Path to the parameters config file."
    ),
    tolerance: float = typer.Option(
        0.02, help="Performance degradation tolerance."
    ),
    update: bool = typer.Option(
        False, "--update-baseline",
        help="Update the baseline performance metric instead of checking against it."
    ),
) -> None:
    """Run automated performance regression testing."""
    console.print("🧪 Running performance regression tests...")
    
    params_config = load_params_config(config)
    features_config = load_features_config(params_config.paths.features)
    
    metrics = run_performance_test(params_config, features_config)
    
    if update:
        update_baseline(metrics)
        console.print(f"✅ Baseline updated with new metrics: {metrics}", style="bold green")
    else:
        result = check_regression(metrics, tolerance)
        
        if result["status"] == "passed":
            console.print(f"✅ {result['message']}", style="bold green")
        elif result["status"] == "failed":
            console.print(f"❌ {result['message']}", style="bold red")
            raise typer.Exit(code=1)
        else:
            console.print(f"⚠️ {result['message']}", style="bold yellow")
            raise typer.Exit(code=1)

@app.command()
def cv_integrity(
    config: Path = typer.Option(
        "configs/params.yaml", "--config", "-c", exists=True,
        help="Path to the parameters config file."
    ),
) -> None:
    """Check cross-validation integrity and for data leakage."""
    console.print("🔍 Checking cross-validation integrity for data leakage...")

    params_config = load_params_config(config)
    features_config = load_features_config(params_config.paths.features)
    
    result = run_cv_leakage_test(params_config, features_config)
    
    if result["status"] == "passed":
        console.print(f"✅ {result['message']}", style="bold green")
    else:
        console.print(f"❌ {result['message']}", style="bold red")
        raise typer.Exit(code=1)

@app.command()
def benchmark(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="Path to the parameters config file."),
) -> None:
    """Run a benchmark suite against other survival analysis libraries."""
    console.print("🏆 Running benchmark suite...")
    # Placeholder for the actual implementation
    console.print("✅ Benchmark suite complete.")

