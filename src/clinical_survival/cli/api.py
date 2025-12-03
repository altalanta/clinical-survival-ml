import subprocess
import sys
from pathlib import Path

import typer
import uvicorn
from rich.console import Console

app = typer.Typer(help="Commands for serving the model via a REST API.")
console = Console()


@app.command()
def launch(
    model_path: Path = typer.Option(
        None,
        "--model-path",
        "-m",
        help="Path to a specific model .joblib file to serve. Overrides the default.",
    ),
    host: str = typer.Option("127.0.0.1", help="Host to bind the API server to."),
    port: int = typer.Option(8000, help="Port to run the API server on."),
    reload: bool = typer.Option(
        False, "--reload", help="Enable auto-reloading for development."
    ),
):
    """
    Launch the FastAPI server to provide real-time model inference.
    """
    console.print("🚀 Launching the inference API server...", style="bold green")

    # This is a bit of a workaround to pass the model_path to the FastAPI app.
    # We set an environment variable that the service.py file can read on startup.
    if model_path:
        if not model_path.exists():
            console.print(f"[bold red]Error: Model not found at {model_path}[/bold red]")
            raise typer.Exit(1)
        import os
        os.environ["CLINICAL_SURVIVAL_MODEL_PATH"] = str(model_path.resolve())

    # The entrypoint for uvicorn is specified as 'module:app_instance'
    app_entrypoint = "clinical_survival.api.service:app"

    uvicorn.run(
        app_entrypoint,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )



