from rich.console import Console
import great_expectations as gx

console = Console()


def validate_data(**_kwargs) -> None:
    """Runs the Great Expectations checkpoint."""
    console.print("--- Running Data Validation ---")
    context = gx.get_context()
    checkpoint_result = context.run_checkpoint(checkpoint_name="toy_survival_checkpoint")
    if not checkpoint_result["success"]:
        console.print(
            "[bold red]Data validation failed! Please check the Data Docs for details.[/bold red]"
        )
        raise RuntimeError("Data validation failed.")
    console.print("✅ Data validation successful.", style="bold green")
