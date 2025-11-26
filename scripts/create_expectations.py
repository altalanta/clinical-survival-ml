import great_expectations as gx
from great_expectations.cli.datasource import sanitize_yaml_and_save_datasource
from great_expectations.core.batch import BatchRequest
from great_expectations.core.yaml_handler import YAMLHandler

yaml = YAMLHandler()

# This script is designed to be run from the root of the project.
data_context_config = gx.data_context.types.base.DataContextConfig(
    config_version=3.0,
    plugins_directory=None,
    evaluation_parameter_store_name="evaluation_parameter_store",
    validations_store_name="validations_store",
    expectations_store_name="expectations_store",
    checkpoint_store_name="checkpoint_store",
    data_docs_sites={
        "local_site": {
            "class_name": "SiteBuilder",
            "store_backend": {
                "class_name": "TupleFilesystemStoreBackend",
                "base_directory": "great_expectations/uncommitted/data_docs/local_site/",
            },
            "site_index_builder": {"class_name": "DefaultSiteIndexBuilder"},
        }
    },
    stores={
        "expectations_store": {
            "class_name": "ExpectationsStore",
            "store_backend": {
                "class_name": "TupleFilesystemStoreBackend",
                "base_directory": "great_expectations/expectations/",
            },
        },
        "validations_store": {
            "class_name": "ValidationsStore",
            "store_backend": {
                "class_name": "TupleFilesystemStoreBackend",
                "base_directory": "great_expectations/uncommitted/validations/",
            },
        },
        "evaluation_parameter_store": {"class_name": "EvaluationParameterStore"},
        "checkpoint_store": {
            "class_name": "CheckpointStore",
            "store_backend": {
                "class_name": "TupleFilesystemStoreBackend",
                "base_directory": "great_expectations/checkpoints/",
            },
        },
    },
    anonymous_usage_statistics={
        "enabled": True,
    },
)

context = gx.get_context(project_config=data_context_config)

# Add a datasource for the toy data
datasource_config = {
    "name": "pandas_datasource",
    "class_name": "Datasource",
    "module_name": "great_expectations.datasource",
    "execution_engine": {
        "module_name": "great_expectations.execution_engine",
        "class_name": "PandasExecutionEngine",
    },
    "data_connectors": {
        "default_inferred_data_connector_name": {
            "class_name": "InferredAssetFilesystemDataConnector",
            "base_directory": "./data/toy",
            "default_regex": {"pattern": "(.*)\\.csv", "group_names": ["data_asset_name"]},
        },
    },
}
context.add_datasource(**datasource_config)

# Create an expectation suite
suite_name = "toy_survival_suite"
context.add_or_update_expectation_suite(expectation_suite_name=suite_name)

batch_request = BatchRequest(
    datasource_name="pandas_datasource",
    data_connector_name="default_inferred_data_connector_name",
    data_asset_name="toy_survival",
)

validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name=suite_name,
)

print("### Creating Expectations ###")

# Core columns
validator.expect_column_to_exist("time")
validator.expect_column_to_exist("event")
validator.expect_column_values_to_not_be_null("time")
validator.expect_column_values_to_not_be_null("event")

# Features
validator.expect_column_values_to_be_in_set("sex", ["male", "female"])
validator.expect_column_mean_to_be_between("age", 30, 80)
validator.expect_column_to_exist("sofa")
validator.expect_column_values_to_not_be_null("sofa")

print("### Saving Expectation Suite ###")
validator.save_expectation_suite(discard_failed_expectations=False)

print("### Creating Checkpoint ###")
checkpoint_config = {
    "name": "toy_survival_checkpoint",
    "config_version": 1,
    "class_name": "SimpleCheckpoint",
    "run_name_template": "%Y%m%d-%H%M%S-my-run-name-template",
    "validations": [
        {
            "batch_request": {
                "datasource_name": "pandas_datasource",
                "data_connector_name": "default_inferred_data_connector_name",
                "data_asset_name": "toy_survival",
            },
            "expectation_suite_name": suite_name,
        }
    ],
}
context.add_or_update_checkpoint(**checkpoint_config)

print("### Running Checkpoint and Building Data Docs ###")
context.run_checkpoint(checkpoint_name="toy_survival_checkpoint")
context.build_data_docs()

print("\n✅ Done. To view the Data Docs, open great_expectations/uncommitted/data_docs/local_site/index.html")









