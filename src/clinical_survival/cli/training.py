from pathlib import Path
import typer
from rich.console import Console

from clinical_survival.io import load_params_config, load_features_config
from clinical_survival.training import train_and_evaluate
from clinical_survival.report import generate_report
from clinical_survival.utils import load_yaml

app = typer.Typer(help="Core model training and evaluation commands.")
console = Console()

@app.command()
def run(
    config_path: Path = typer.Option(
        "configs/params.yaml", "--config", "-c", exists=True,
        help="Path to the main parameters configuration file."
    ),
    grid_path: Path = typer.Option(
        "configs/model_grid.yaml", "--grid", "-g", exists=True,
        help="Path to the model grid configuration file."
    ),
) -> None:
    """Run the complete pipeline: train, evaluate, and generate a report."""
    console.print("🚀 Starting the full clinical survival analysis pipeline...", style="bold green")

    params_config = load_params_config(config_path)
    features_config = load_features_config(params_config.paths.features)
    grid_config = load_yaml(grid_path)

    console.print("1. [bold blue]Training and evaluating models...[/bold blue]")
    train_and_evaluate(
        params_config=params_config,
        features_config=features_config,
        grid_config=grid_config
    )

    console.print("2. [bold blue]Generating HTML report...[/bold blue]")
    generate_report(params_config)

    console.print(f"✅ Pipeline complete. Results saved to [bold cyan]{params_config.paths.outdir}[/bold cyan]", style="bold green")

# NOTE: The original `train`, `evaluate`, and `report` commands were very thin
# wrappers. The core logic is better encapsulated in the `run` command and the
# underlying library functions. For a more modular CLI, these could be built
# out with more distinct functionality if needed in the future.
