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

## Key Features

- **End-to-End Survival Pipeline:** From data preprocessing to model evaluation and explainability.
- **Configurability:** Easily configure all aspects of the pipeline via YAML files, including feature definitions, model parameters, and cross-validation strategy.
- **Data Quality Gate:** Integrated with Great Expectations to ensure the quality and integrity of input data, preventing errors before they impact the pipeline.
- **Automated Hyperparameter Tuning:** Leverages Optuna for efficient and effective hyperparameter searches.
- **Intermediate Caching:** Caches the results of expensive computations (like data preprocessing) to dramatically speed up subsequent runs.
- **Advanced Explainability:** Generates SHAP explanations to understand model predictions at a global and local level.
- **Counterfactual Explanations for Actionable Insights:** Uses DiCE to generate "what-if" scenarios.

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

To accelerate development and experimentation, the pipeline includes a caching mechanism for intermediate artifacts. Expensive, deterministic steps like data preprocessing are cached using `joblib`. On subsequent runs, if the input data and configurations have not changed, the results are loaded from the cache, saving significant computation time.

Caching can be enabled/disabled and the cache directory can be configured in `configs/params.yaml`.

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

## Counterfactual Explanations for Actionable Insights

Beyond understanding *why* a model makes a certain prediction, this framework can generate counterfactual explanations to show *what* would need to change for a different outcome. Using the `DiCE` library, it can answer "what if" questions for instances predicted to have a high risk, providing actionable insights.

For example, a counterfactual explanation might show the minimal changes in a patient's lab values that would flip their prediction from "high-risk" to "low-risk".

This feature is configurable in `configs/params.yaml` and the results are saved to the `results/artifacts/counterfactuals/` directory.

### Data Validation

The pipeline includes a data quality gate powered by Great Expectations. Before any processing, it validates the raw input data against a defined "Expectation Suite." This ensures that the data adheres to a predefined schema and quality standards. If validation fails, the pipeline halts, preventing corrupted or unexpected data from propagating.

You can enable/disable this feature and configure the expectation suite in `configs/params.yaml`.

## Data and Concept Drift Monitoring

To ensure that deployed models maintain their performance over time, this project includes a service for detecting data and concept drift. This is a critical MLOps practice that helps identify when a model might need to be retrained due to changes in the underlying data distribution.

The service uses the `evidently` library to compare a new batch of data (the "current" data) against a baseline (the "reference" data, typically the training set). It generates a detailed HTML report that visualizes changes in feature distributions (data drift) and the target variable's behavior (concept drift).

### Running a Drift Analysis

To run a drift analysis, use the `monitoring detect-drift` command. You will need to provide the reference dataset, the current dataset you want to analyze, and the feature configuration file.

```bash
poetry run clinical-ml monitoring detect-drift \
  --reference-csv data/toy/toy_survival.csv \
  --current-csv data/toy/toy_survival_new_batch.csv \
  --features-config configs/features.yaml \
  --output-path results/monitoring/drift_report.html
```

After running the command, you can open the generated `drift_report.html` in your browser to explore the results interactively.

## Real-Time Inference API

This project includes a REST API built with FastAPI to serve trained models for real-time inference. This allows other applications to get survival predictions by sending a simple HTTP request.

### Launching the API

To launch the API server, run the following command. By default, it will load the `rsf.joblib` model from the `results/artifacts/models/` directory.

```bash
poetry run clinical-ml api launch
```

You can also specify a different model to serve:

```bash
poetry run clinical-ml api launch --model-path /path/to/your/model.joblib
```

The server will be available at `http://127.0.0.1:8000`.

### Making a Prediction

You can send a `POST` request to the `/predict` endpoint with the patient's feature data in the request body. Here is an example using `curl`:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "feature_1": 0.5,
    "feature_2": 1,
    "feature_3": "A"
  }'
```

The API will respond with the predicted risk score:

```json
{
  "risk_score": 0.85
}
```

## Interactive Dashboard for Results Exploration

This project includes an interactive dashboard built with Streamlit to help you explore and understand the results of your analysis. The dashboard provides:
-   High-level performance overviews.
-   Detailed, interactive plots for model comparison.
-   A **Counterfactual Explorer** to investigate "what-if" scenarios for individual patients.

### Launching the Dashboard

To launch the dashboard, run the following command from the root of the project:

```bash
poetry run clinical-ml dashboard launch
```

## Contributing

Contributions are welcome! Please see the `CONTRIBUTING.md` file for details.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
