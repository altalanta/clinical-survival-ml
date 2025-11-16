import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Commands for visualizing results with an interactive dashboard.")
console = Console()

@app.command()
def launch(
    port: int = typer.Option(8501, help="Port to run the Streamlit dashboard on."),
):
    """
    Launch the interactive Streamlit dashboard to explore results.
    """
    console.print("🚀 Launching the interactive dashboard...", style="bold green")
    
    # Get the path to the dashboard script
    dashboard_script_path = Path(__file__).parent.parent / "dashboard.py"
    
    if not dashboard_script_path.exists():
        console.print(f"[bold red]Error: Dashboard script not found at {dashboard_script_path}[/bold red]")
        raise typer.Exit(1)

    command = [
        "streamlit",
        "run",
        str(dashboard_script_path),
        "--server.port",
        str(port),
        "--server.headless",
        "true", # To prevent Streamlit from opening a browser window automatically
    ]

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        console.print("[bold red]Error: `streamlit` command not found.[/bold red]")
        console.print("Please make sure Streamlit is installed in your environment (`poetry install`).")
        raise typer.Exit(1)
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error launching dashboard: {e}[/bold red]")
        raise typer.Exit(1)
