from pathlib import Path
from typing import Dict, Any

import pandas as pd
from great_expectations.data_context.types.base import (
    DataContextConfig,
    FilesystemStoreBackendDefaults,
)
from great_expectations.data_context import BaseDataContext
from great_expectations.exceptions import DataContextError
from rich.console import Console

console = Console()


def validate_data(
    df: pd.DataFrame, expectation_suite_path: Path, run_name: str = "pipeline_data_run"
) -> bool:
    """
    Validates a DataFrame against a Great Expectations Expectation Suite.

    Args:
        df: The pandas DataFrame to validate.
        expectation_suite_path: Path to the JSON file containing the expectation suite.
        run_name: A name for the validation run.

    Returns:
        True if validation succeeds.

    Raises:
        ValueError: If validation fails.
    """
    if not expectation_suite_path.exists():
        raise FileNotFoundError(f"Expectation suite not found at {expectation_suite_path}")

    console.print(f"🔎 Validating data against suite: [bold cyan]{expectation_suite_path.name}[/bold cyan]")

    # Create an in-memory Data Context
    project_config = DataContextConfig(
        store_backend_defaults=FilesystemStoreBackendDefaults(
            root_directory=expectation_suite_path.parent
        ),
    )
    context = BaseDataContext(project_config=project_config)

    # Create a Datasource and a Batch Request
    datasource_name = "pandas_datasource"
    try:
        datasource = context.sources.add_pandas(name=datasource_name)
    except DataContextError:
        # Datasource already exists
        datasource = context.get_datasource(datasource_name)

    asset_name = "pipeline_data_asset"
    data_asset = datasource.add_dataframe_asset(name=asset_name, dataframe=df)
    batch_request = data_asset.build_batch_request()

    # Create a Checkpoint to run the validation
    checkpoint = context.add_or_update_checkpoint(
        name="pipeline_checkpoint",
        run_name_template=f"{run_name}-%Y%m%d-%H%M%S",
        batch_request=batch_request,
        expectation_suite_name=expectation_suite_path.stem,
    )

    # Run validation
    result = checkpoint.run()

    if not result["success"]:
        console.print("[bold red]Data validation failed![/bold red]")
        # You could log the full `result` here for detailed debugging
        raise ValueError("Input data does not meet the quality standards defined in the expectation suite.")

    console.print("✅ Data validation successful.", style="green")
    return True
