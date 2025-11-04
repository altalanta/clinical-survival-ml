from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
import mlflow

from clinical_survival.serve import run_server

app = typer.Typer(help="Commands for MLOps, model registration, and deployment.")
console = Console()

@app.command()
def promote_model(
    model_name: str = typer.Option(..., "--model-name", help="Name of the model to promote."),
    version: int = typer.Option(..., "--version", help="Version of the model to promote."),
    stage: str = typer.Option(..., "--stage", help="Stage to promote the model to (e.g., 'staging', 'production')."),
) -> None:
    """Promote a model version to a new stage in the registry."""
    console.print(f"Promoting model [bold green]{model_name}[/bold green] version [bold cyan]{version}[/bold cyan] to stage [bold yellow]{stage}[/bold yellow]...")
    try:
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage=stage,
            archive_existing_versions=True
        )
        console.print("✅ Model promoted successfully.")
    except Exception as e:
        console.print(f"[bold red]Error promoting model: {e}[/bold red]")
        raise typer.Exit(1)

@app.command()
def deploy_model(
    model_name: str = typer.Option(..., "--model-name", help="Name of the model to deploy."),
    stage: str = typer.Option("production", help="The model stage to deploy (e.g., 'production')."),
    host: str = typer.Option("127.0.0.1", help="Host to bind the server to."),
    port: int = typer.Option(8000, help="Port to bind the server to."),
) -> None:
    """Deploy a model from the registry to a serving environment."""
    console.print(f"Deploying model [bold green]{model_name}[/bold green] from stage [bold yellow]{stage}[/bold yellow]...")
    
    model_uri = f"models:/{model_name}/{stage}"
    
    console.print(f"Starting server with model from [bold cyan]{model_uri}[/bold cyan]...")
    run_server(model_uri=model_uri, host=host, port=port)
