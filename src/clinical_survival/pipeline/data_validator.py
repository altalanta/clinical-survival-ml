from pathlib import Path
import sys

import pandas as pd
from rich.console import Console
from great_expectations.data_context import FileDataContext
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.exceptions import DataContextError

from clinical_survival.config import ParamsConfig, FeaturesConfig

console = Console()

def validate_data(
    raw_df: pd.DataFrame,
    params_config: ParamsConfig,
    features_config: FeaturesConfig,
    **kwargs,
):
    """
    Validates the input dataframe using a Great Expectations suite.
    
    This function acts as a "Data Quality Gate". If validation fails,
    it terminates the pipeline.
    """
    if not params_config.data_validation.enabled:
        console.print("Data validation is disabled. Skipping.", style="yellow")
        return

    try:
        context = FileDataContext.create(project_root_dir=".")
    except DataContextError:
        context = FileDataContext(project_root_dir=".")

    datasource_name = "pandas_datasource"
    if datasource_name not in context.list_datasources():
        context.sources.add_pandas(name=datasource_name)

    expectation_suite_name = params_config.data_validation.expectation_suite
    
    # Create suite if it doesn't exist
    if expectation_suite_name not in context.list_expectation_suite_names():
        console.print(f"Creating new expectation suite: '{expectation_suite_name}'")
        suite = context.add_expectation_suite(expectation_suite_name)
        
        # Add some basic expectations based on the features config
        required_columns = (
            features_config.numerical_features +
            features_config.categorical_features +
            [features_config.time_column, features_config.event_column]
        )
        
        for col in required_columns:
            suite.add_expectation(
                expectation_configuration={
                    "expectation_type": "expect_column_to_exist",
                    "kwargs": {"column": col},
                }
            )
        context.save_expectation_suite(suite)


    batch_request = RuntimeBatchRequest(
        datasource_name=datasource_name,
        data_asset_name="clinical_data_asset",
        runtime_data=raw_df,
        batch_identifiers={"default_identifier": "default_identifier"},
    )
    
    checkpoint_name = "validation_checkpoint"
    checkpoint_config = {
        "name": checkpoint_name,
        "config_version": 1.0,
        "class_name": "SimpleCheckpoint",
        "run_name_template": "%Y%m%d-%H%M%S-validation-run",
        "validations": [
            {
                "batch_request": batch_request,
                "expectation_suite_name": expectation_suite_name,
            }
        ],
    }
    context.add_or_update_checkpoint(**checkpoint_config)

    console.print(f"✅ Running data validation against suite: '{expectation_suite_name}'...")
    results = context.run_checkpoint(checkpoint_name=checkpoint_name)

    if not results["success"]:
        console.print("[bold red]Data validation failed![/bold red]")
        console.print("Review the validation results for more details.")
        # Optionally, you can build and show the data docs here.
        sys.exit(1)
    
    console.print("✅ Data validation successful.", style="green")
