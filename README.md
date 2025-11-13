# Clinical Survival ML

Reproducible end-to-end survival modeling for tabular clinical outcomes. This toolkit ingests time-to-event data, preprocesses covariates, fits survival models, evaluates them, and renders a publication-ready HTML report.

**📖 [View the full documentation here.](<placeholder_for_github_pages_url>)**

## Overview

This project provides a comprehensive pipeline for clinical survival analysis, including:

- **Data Preprocessing**: Imputation, scaling, and encoding of clinical data.
- **Survival Modeling**: Support for Cox PH, Random Survival Forests, and XGBoost-based models.
- **Rigorous Evaluation**: Cross-validation with metrics like Harrell's C-index, Brier score, and IPCW calibration.
- **Explainability**: SHAP- and PDP-based interpretability to understand model predictions.
- **Automated Reporting**: Generation of a complete, interactive HTML report.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/clinical-survival-ml.git
    cd clinical-survival-ml
    ```

2.  **Install dependencies using Poetry:**
    This project uses [Poetry](https://python-poetry.org/) for dependency management.
    ```bash
    poetry install --all-extras
    ```

3.  **Retrieve the data:**
    This project uses [DVC](https://dvc.org/) to version control data. After installing the dependencies, pull the data from the default remote:
    ```bash
    poetry run dvc pull
    ```

## Quickstart

```bash
# Install dependencies
poetry install --all-extras

# Run the toy example
poetry run clinical-ml run --config configs/params.yaml

# View the results
open results/report.html
```

## Data Validation with Great Expectations

This project uses [Great Expectations](https://greatexpectations.io/) to ensure the quality and validity of input data. Before any training is performed, the pipeline runs a checkpoint to verify that the data conforms to a predefined set of expectations.

### How It Works

1.  **Expectation Suite**: A collection of expectations about the data is defined in a JSON file located in the `great_expectations/expectations` directory. This suite acts as a "unit test for your data."
2.  **Validation Checkpoint**: Before the training pipeline runs, it executes a checkpoint that validates the input data against this suite.
3.  **Data Docs**: If the validation fails, the pipeline will stop, and you can review the detailed validation results in the "Data Docs," which are HTML reports that break down the expectation results.

### Running Validation

To generate the initial Expectation Suite and view the Data Docs, run the following script from the root of the project:

```bash
poetry run python scripts/create_expectations.py
```

This will also build the Data Docs. You can view them by opening `great_expectations/uncommitted/data_docs/local_site/index.html`.

## Caching

To speed up development and experimentation, this project uses `joblib` to cache the results of the expensive data preprocessing step. When you re-run the training pipeline with the same data and configuration, the preprocessed data will be loaded from a local cache, saving significant time.

Caching is enabled by default and can be configured in `configs/params.yaml`:

```yaml
caching:
  enabled: true
  dir: "artifacts/cache"
```

## Experiment Tracking with MLflow

This project is integrated with [MLflow](https://mlflow.org/) for comprehensive experiment tracking and model management. All training runs are automatically logged, allowing you to track parameters, metrics, and artifacts, as well as version and manage your models.

### Viewing Experiments

To view the MLflow UI and compare your experiment runs, execute the following command from the root of the project:

```bash
poetry run mlflow ui
```

This will start the MLflow Tracking UI (by default at `http://127.0.0.1:5000`), where you can:
-   Compare the performance of different models.
-   View the parameters and metrics for each run.
-   See the artifacts that were generated, including plots and the trained model files.

### Model Registry

Trained models are automatically registered in the MLflow Model Registry, providing a central place to manage their lifecycle from development to production. You can use the MLflow UI to transition models between stages (e.g., from `Staging` to `Production`).

## Plugin System for Extensibility

This framework includes a plugin system that allows you to add your own custom models and preprocessors without modifying the core library code. This makes it easy to experiment with new algorithms and components.

For a detailed guide and a working example, please see the [custom model plugin example](./examples/plugins/custom_model/README.md).

## Contributing

Contributions are welcome! Please see the `CONTRIBUTING.md` file for details.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
