from pathlib import Path
from typing import Optional

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
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from the latest failed run (if checkpoints exist).",
    ),
    disable_checkpoints: bool = typer.Option(
        False,
        "--no-checkpoints",
        help="Disable checkpointing for this run.",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Optional run identifier when resuming or forcing a specific ID.",
    ),
    enable_performance_monitoring: bool = typer.Option(
        False,
        "--performance-monitoring",
        help="Enable detailed performance monitoring and optimization recommendations.",
    ),
    performance_report_path: Optional[Path] = typer.Option(
        None,
        "--performance-report",
        help="Path to save performance monitoring report (default: results/artifacts/performance_report.md).",
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
        grid_config=grid_config,
        resume=resume,
        enable_checkpoints=not disable_checkpoints,
        run_id=run_id,
        enable_performance_monitoring=enable_performance_monitoring,
        performance_report_path=str(performance_report_path) if performance_report_path else None,
    )

    console.print("2. [bold blue]Generating HTML report...[/bold blue]")
    generate_report(params_config)

    console.print(f"✅ Pipeline complete. Results saved to [bold cyan]{params_config.paths.outdir}[/bold cyan]", style="bold green")


@app.command("list-checkpoints")
def list_checkpoints(
    output_dir: Path = typer.Option(
        "results",
        "--outdir",
        help="Base output directory where checkpoints/ are stored.",
    )
) -> None:
    """List available checkpoint runs."""
    from clinical_survival.checkpoint import CheckpointManager
    runs = CheckpointManager.list_runs(output_dir / "checkpoints")
    if not runs:
        console.print("No checkpoint runs found.")
        return
    console.print(f"Found {len(runs)} checkpoint run(s):")
    for run in runs:
        console.print(
            f"- {run.run_id} | status={run.status} | steps={len(run.completed_steps)}/{len(run.pipeline_steps)} | updated={run.updated_at}"
        )


@app.command("resume")
def resume_run(
    config_path: Path = typer.Option(
        "configs/params.yaml", "--config", "-c", exists=True, help="Path to params config."
    ),
    grid_path: Path = typer.Option(
        "configs/model_grid.yaml", "--grid", "-g", exists=True, help="Path to model grid config."
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Run identifier to resume; if omitted, resumes the most recent failed run.",
    ),
    outdir: Path = typer.Option(
        "results",
        "--outdir",
        help="Base output directory where checkpoints/ are stored.",
    ),
) -> None:
    """Resume a failed pipeline run from checkpoints."""
    from clinical_survival.checkpoint import CheckpointManager

    console.print("🔄 Attempting to resume pipeline from checkpoints...", style="bold yellow")
    manager = None
    if run_id:
        # Try to load specific run
        candidates = CheckpointManager.list_runs(outdir / "checkpoints")
        for run in candidates:
            if run.run_id == run_id:
                manager = CheckpointManager(outdir / "checkpoints", run_id=run_id)
                manager._state = run
                break
    else:
        manager = CheckpointManager.get_resumable_run(outdir / "checkpoints")

    if manager is None:
        console.print("[red]No resumable checkpoint run found.[/red]")
        raise typer.Exit(1)

    params_config = load_params_config(config_path)
    features_config = load_features_config(params_config.paths.features)
    grid_config = load_yaml(grid_path)

    train_and_evaluate(
        params_config=params_config,
        features_config=features_config,
        grid_config=grid_config,
        resume=True,
        enable_checkpoints=True,
        run_id=manager.run_id,
    )

    console.print(f"[green]Resumed run completed for {manager.run_id}[/green]")

# NOTE: The original `train`, `evaluate`, and `report` commands were very thin
# wrappers. The core logic is better encapsulated in the `run` command and the
# underlying library functions. For a more modular CLI, these could be built
# out with more distinct functionality if needed in the future.
