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

## Structured Logging

The pipeline includes a comprehensive, centralized logging system designed for both development and production use. It provides:

-   **Structured JSON logging**: For production environments, logs are output in JSON format, making them easy to ingest into log aggregation systems like ELK stack, Datadog, or CloudWatch.
-   **Rich console logging**: For development, colorful and readable output with tracebacks.
-   **Correlation IDs**: Every pipeline run is assigned a unique correlation ID, allowing you to trace all log messages from a single execution.
-   **Pipeline step tracking**: Each step of the pipeline is logged with timing information, making it easy to identify bottlenecks.

### Configuration

Logging is configured in `configs/params.yaml`:

```yaml
logging:
  level: "INFO"           # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "rich"          # "rich" (console), "structured" (JSON), "simple"
  log_file: null          # Optional: path to write logs to a file
  include_correlation_id: true
```

### CLI Options

You can also override logging settings via CLI flags:

```bash
# Enable verbose (DEBUG) logging
poetry run clinical-ml --verbose training run

# Use structured JSON logging (useful for production/CI)
poetry run clinical-ml --log-format structured training run

# Write logs to a file
poetry run clinical-ml --log-file results/logs/run.log training run
```

### Using the Logger in Custom Code

If you're extending the pipeline, you can use the centralized logger:

```python
from clinical_survival.logging_config import get_logger, LogContext

logger = get_logger(__name__)

# Basic logging with extra context
logger.info("Training started", extra={"n_samples": 1000, "model": "coxph"})

# Use LogContext to add context to all logs within a block
with LogContext(run_id="abc123", fold=1):
    logger.info("Processing fold")  # Will include run_id and fold
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

## Type Safety and Runtime Validation

This project uses comprehensive type hints and runtime validation to catch errors early and improve code maintainability.

### Type Hints

All public functions and methods include type hints. The project uses:

-   **Protocol classes**: Define interfaces for models, preprocessors, and pipeline steps
-   **Type aliases**: Common types like `FeatureMatrix`, `SurvivalArray`, `RiskScores`
-   **Pydantic models**: Configuration validation with automatic type coercion

### Runtime Validation

The `types` module provides decorators for runtime type checking:

```python
from clinical_survival.types import validate_input, validate_output, validate_dataframe_columns

@validate_input("X", pd.DataFrame)
@validate_dataframe_columns("X", ["age", "bmi"])
@validate_output(np.ndarray)
def predict(X: pd.DataFrame) -> np.ndarray:
    ...
```

### Pipeline Step Schemas

Pipeline steps can define input/output schemas for automatic validation:

```python
from clinical_survival.types import StepInputSchema, StepOutputSchema, validate_step_io

class LoadDataOutput(StepOutputSchema):
    raw_df: pd.DataFrame

@validate_step_io(output_schema=LoadDataOutput)
def load_raw_data(**context) -> dict:
    ...
```

### Running Type Checks

```bash
# Run mypy for static type checking
poetry run mypy src/clinical_survival/

# The project uses strict mode with Pydantic plugin
```

## Unified Error Handling

The pipeline includes a comprehensive error handling system that provides:

-   **User-friendly error messages**: Clear explanations of what went wrong
-   **Actionable suggestions**: Hints on how to fix common problems
-   **Structured error context**: Detailed information for debugging
-   **Consistent error hierarchy**: Well-organized exception types

### Error Categories

The pipeline uses a hierarchy of custom exceptions:

```
ClinicalSurvivalError (base)
├── ConfigurationError - Invalid or missing configuration
├── DataError - Data loading, validation, or processing issues
│   ├── DataLoadError - Failed to load data from source
│   ├── DataValidationError - Data failed quality checks
│   └── MissingColumnError - Required column not found
├── ModelError - Model training or inference issues
│   ├── ModelNotFittedError - Model used before fitting
│   ├── ModelTrainingError - Training failed
│   └── ModelInferenceError - Prediction failed
├── PipelineError - Pipeline orchestration issues
│   ├── StepNotFoundError - Pipeline step not found
│   └── StepExecutionError - Pipeline step failed
└── ReportError - Report generation issues
```

### User-Friendly CLI Output

When errors occur, the CLI displays helpful panels with:

```
❌ Pipeline Error
────────────────────────────────────────
DataLoadError
Data file not found: data/missing.csv

Context:
  • path: data/missing.csv

Suggestions:
  → Verify the file path is correct and the file exists
  → Check file permissions
  → If using DVC, run 'dvc pull' to fetch the data
────────────────────────────────────────
Run with --verbose flag for full traceback
```

### Error Handling Decorators

The `error_handling` module provides decorators for common patterns:

```python
from clinical_survival.error_handling import (
    handle_errors,
    wrap_step_errors,
    require_fitted,
    retry_on_error,
)

# Wrap pipeline steps with automatic error context
@wrap_step_errors("data_loading")
def load_raw_data(**context):
    ...

# Retry transient failures with exponential backoff
@retry_on_error(max_attempts=3, exceptions=(ConnectionError,))
def fetch_external_data(url):
    ...

# Ensure model is fitted before prediction
class MyModel:
    @require_fitted()
    def predict(self, X):
        return self.model.predict(X)
```

## Resilience Patterns for External Services

The pipeline includes comprehensive resilience patterns to handle failures in external services (MLflow, Dask, external APIs) gracefully.

### Retry with Exponential Backoff

Automatically retry transient failures with increasing delays:

```python
from clinical_survival.resilience import retry_with_backoff

@retry_with_backoff(
    max_attempts=3,
    initial_delay=1.0,
    backoff_factor=2.0,
    exceptions=(ConnectionError, TimeoutError),
)
def call_external_service():
    return requests.get(url).json()
```

### Circuit Breaker Pattern

Prevent cascading failures by "opening" the circuit after repeated failures:

```python
from clinical_survival.resilience import circuit_breaker, CircuitBreaker

@circuit_breaker(
    name="external_api",
    failure_threshold=5,
    recovery_timeout=60,
    fallback=lambda: {"status": "unavailable"},
)
def call_external_api():
    return api.fetch_data()

# Check circuit state
breaker = CircuitBreaker.get("external_api")
print(f"Circuit state: {breaker.state}")  # CLOSED, OPEN, or HALF_OPEN
```

### Graceful Degradation

Continue operation with reduced functionality when services are unavailable:

```python
from clinical_survival.resilience import graceful_degradation

@graceful_degradation(default=None)
def log_to_monitoring_service(data):
    monitoring_api.log(data)  # Won't raise even if service is down
```

### MLflow Tracking with Resilience

The MLflow tracker automatically handles failures:

```python
from clinical_survival.tracking import MLflowTracker

tracker = MLflowTracker(config)

# This won't fail even if MLflow is unavailable
with tracker.start_run("training"):
    tracker.log_params({"lr": 0.01})  # Falls back to local storage
    tracker.log_metrics({"accuracy": 0.95})

# Check if tracker is degraded
if tracker.is_degraded:
    print("MLflow unavailable, using local fallback")
```

### Configuration

Resilience settings are configured in `configs/params.yaml`:

```yaml
resilience:
  # Retry settings
  max_retries: 3
  retry_delay: 1.0
  retry_backoff: 2.0
  
  # Circuit breaker
  circuit_failure_threshold: 5
  circuit_recovery_timeout: 60.0
  
  # Timeouts
  mlflow_timeout: 30.0
  external_api_timeout: 60.0
  
  # Fallback
  enable_fallback: true
  fallback_dir: "artifacts/fallback"
```

### Combined Resilience Decorator

For maximum protection, use the combined decorator:

```python
from clinical_survival.resilience import resilient

@resilient(
    max_retries=3,
    circuit_name="api",
    timeout_seconds=30,
    fallback=lambda: {"error": "service unavailable"},
)
def call_critical_service():
    return api.call()
```

## Pipeline Step Schema Validation

The pipeline includes comprehensive input/output schema validation for each step, ensuring data flows correctly between steps and catching configuration errors early.

### How It Works

Each pipeline step has defined schemas for its expected inputs and outputs:

```
data_loader.load_raw_data
  Input:  params_config (ParamsConfig)
  Output: raw_df (DataFrame)
     ↓
preprocessor.prepare_data
  Input:  raw_df (DataFrame), params_config, features_config
  Output: X_train, X_test, y_train, y_test, preprocessor
     ↓
tuner.tune_hyperparameters
  Input:  X_train, y_train, params_config, features_config, grid_config
  Output: best_params (Dict)
     ↓
training_loop.run_training_loop
  Input:  X_train, y_train, best_params, tracker, outdir, ...
  Output: final_pipelines (Dict)
```

### User-Friendly Error Messages

When validation fails, you get helpful error messages:

```
❌ Input Validation Error
────────────────────────────────────────────────────────
Schema validation failed for input of step 'prepare_data'

┌─────────────┬─────────────────────────────────┬───────────┐
│ Field       │ Error                           │ Input     │
├─────────────┼─────────────────────────────────┼───────────┤
│ raw_df      │ raw_df must be a DataFrame      │ None      │
│ params_con… │ Field required                  │           │
└─────────────┴─────────────────────────────────┴───────────┘

Suggestions:
  → Check that previous pipeline steps completed successfully
  → Verify the step is receiving all required inputs
  → Review the schema definition in pipeline/schemas.py
────────────────────────────────────────────────────────
```

### Defining Custom Step Schemas

To add validation to a new pipeline step:

```python
from clinical_survival.pipeline.schemas import (
    PipelineStepInput,
    PipelineStepOutput,
    validate_pipeline_step,
)
from pydantic import Field, field_validator

class MyStepInput(PipelineStepInput):
    """Input schema for my custom step."""
    
    data: Any = Field(..., description="Input data")
    threshold: float = Field(0.5, description="Processing threshold")
    
    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("threshold must be between 0 and 1")
        return v

class MyStepOutput(PipelineStepOutput):
    """Output schema for my custom step."""
    
    processed_data: Any = Field(..., description="Processed output")
    metrics: Dict[str, float] = Field(..., description="Processing metrics")

@validate_pipeline_step(
    input_schema=MyStepInput,
    output_schema=MyStepOutput,
)
def my_custom_step(data, threshold=0.5, **context):
    # Process data...
    return {
        "processed_data": result,
        "metrics": {"accuracy": 0.95},
    }
```

### Listing Validated Steps

To see which steps have schema validation:

```python
from clinical_survival.pipeline.schemas import list_validated_steps

print(list_validated_steps())
# ['data_loader.load_raw_data', 'preprocessor.prepare_data', ...]
```

### Generating Schema Documentation

Auto-generate markdown documentation for all schemas:

```python
from clinical_survival.pipeline.schemas import generate_schema_docs

docs = generate_schema_docs()
print(docs)  # Markdown table of all step schemas
```

## Pipeline Performance Profiling

The pipeline includes built-in performance profiling to track execution time and memory usage for each step.

### Automatic Profiling

Every pipeline run automatically generates a performance profile saved to `artifacts/pipeline_profile.json`:

```json
{
  "pipeline_name": "training_pipeline",
  "total_duration_seconds": 45.23,
  "peak_memory_mb": 512.4,
  "steps": [
    {"name": "data_loader.load_raw_data", "duration_seconds": 1.2, "memory_peak_mb": 128.5},
    {"name": "preprocessor.prepare_data", "duration_seconds": 3.4, "memory_peak_mb": 256.2},
    ...
  ]
}
```

### Using the Profiler Directly

```python
from clinical_survival.profiling import PipelineProfiler, profile_function, timed

# Profile a complete workflow
profiler = PipelineProfiler("my_workflow", track_memory=True)

with profiler.profile_step("load_data"):
    df = load_data(path)

with profiler.profile_step("train_model"):
    model = train(df)

profile = profiler.finish()
profiler.print_summary()  # Rich console output
profiler.save_report("profile.json")

# Simple function timing decorator
@timed
def my_function():
    ...

# Full function profiling with memory
@profile_function(track_memory=True)
def expensive_computation(data):
    ...
```

### Profile Summary Output

```
Pipeline Profile: training_pipeline
Correlation ID: abc123
Status: ✅ Success

┌─────────────────┬──────────┐
│ Metric          │ Value    │
├─────────────────┼──────────┤
│ Total Duration  │ 45.23s   │
│ Peak Memory     │ 512.4 MB │
│ Steps Executed  │ 5        │
└─────────────────┴──────────┘

┌──────────────────────────────┬──────────┬──────────┬─────────────┬────────┐
│ Step                         │ Duration │ % Total  │ Memory Peak │ Status │
├──────────────────────────────┼──────────┼──────────┼─────────────┼────────┤
│ data_loader.load_raw_data    │ 1.20s    │ 2.7%     │ 128.5 MB    │ ✅     │
│ preprocessor.prepare_data    │ 3.40s    │ 7.5%     │ 256.2 MB    │ ✅     │
│ tuner.tune_hyperparameters   │ 25.10s   │ 55.5%    │ 384.1 MB    │ ✅     │
│ training_loop.run_training   │ 15.53s   │ 34.3%    │ 512.4 MB    │ ✅     │
└──────────────────────────────┴──────────┴──────────┴─────────────┴────────┘
```

## API Health Checks and Metrics

The inference API includes comprehensive health check and metrics endpoints for production deployments.

### Endpoints

| Endpoint | Purpose | Response |
|----------|---------|----------|
| `GET /health` | Basic liveness probe | `{"status": "healthy", "uptime_seconds": 123.4}` |
| `GET /ready` | Readiness probe (model loaded?) | `{"ready": true, "model_loaded": true}` |
| `GET /metrics` | Prometheus-compatible metrics | JSON or text format |
| `POST /predict` | Single prediction | `{"risk_score": 0.75}` |
| `POST /predict/batch` | Batch predictions (up to 1000) | `[{"risk_score": 0.75}, ...]` |

### Kubernetes Integration

```yaml
# Kubernetes deployment with probes
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: clinical-api
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
```

### Prometheus Metrics

Request metrics in Prometheus format:

```bash
curl http://localhost:8000/metrics?format=prometheus
```

```
# HELP clinical_survival_requests_total Total number of requests
# TYPE clinical_survival_requests_total counter
clinical_survival_requests_total 1523

# HELP clinical_survival_predictions_total Total number of predictions
# TYPE clinical_survival_predictions_total counter
clinical_survival_predictions_total 1200

# HELP clinical_survival_errors_total Total number of errors
# TYPE clinical_survival_errors_total counter
clinical_survival_errors_total 12

# HELP clinical_survival_request_latency_ms Average request latency
# TYPE clinical_survival_request_latency_ms gauge
clinical_survival_request_latency_ms 45.23
```

## Pre-flight Health Checks

The pipeline includes comprehensive health checks that run before execution to validate your environment:

```python
from clinical_survival.diagnostics import run_health_checks

# Run health checks
results = run_health_checks(params_config)

if not results.all_passed:
    print("Health checks failed!")
    for check in results.failed_checks:
        print(f"  ✗ {check.name}: {check.message}")
        if check.suggestion:
            print(f"    → {check.suggestion}")
```

### What Gets Checked

| Check | Description |
|-------|-------------|
| Python Version | Ensures Python >= 3.10 |
| Core Dependencies | Validates numpy, pandas, sklearn, sksurv |
| Optional Dependencies | Checks for xgboost, mlflow, shap, etc. |
| GPU Availability | Detects CUDA and XGBoost GPU support |
| System Memory | Warns if < 2GB available |
| Disk Space | Warns if < 1GB free |
| Data File | Validates data file exists and is readable |
| Output Directory | Confirms write permissions |
| MLflow Connection | Tests tracking server if configured |

### Sample Output

```
🏥 Health Check Results
┌────────┬─────────────────────┬──────────────────────────────────┐
│ Status │ Check               │ Result                           │
├────────┼─────────────────────┼──────────────────────────────────┤
│   ✓    │ Python Version      │ Python 3.11.5                    │
│   ✓    │ Core Dependencies   │ 7 core packages installed        │
│   ⚠    │ Optional Dep.       │ Some optional features unavail.  │
│   ⚠    │ GPU Availability    │ No GPU detected, using CPU only  │
│   ✓    │ System Memory       │ 12.4GB available of 16.0GB       │
│   ✓    │ Data File           │ Found data.csv (2.5MB, 15 cols)  │
└────────┴─────────────────────┴──────────────────────────────────┘
```

## Configuration Validation

Beyond schema validation, the pipeline performs semantic validation to catch configuration issues early:

```python
from clinical_survival.config_validation import validate_configuration

result = validate_configuration(
    params_config,
    features_config, 
    grid_config,
    data_df=df  # Optional: enables data-aware validation
)

if result.has_errors:
    print("Configuration errors found:")
    print(result.summary())
```

### Validation Categories

- **Feature References**: Ensures columns referenced in config exist in data
- **Model Availability**: Validates model names and dependencies
- **Grid Config**: Checks hyperparameter ranges are sensible
- **Pipeline Steps**: Verifies step order and dependencies
- **Path Validation**: Confirms files and directories exist
- **Data Quality**: Checks sample size, event rate, missing data

## Model Comparison and Selection

The pipeline automatically compares trained models and selects the best:

```python
from clinical_survival.model_selection import ModelComparator, SelectionCriterion

# Compare models
comparator = ModelComparator(
    metrics={
        "coxph": {"concordance": 0.72, "ibs": 0.18},
        "rsf": {"concordance": 0.75, "ibs": 0.16},
        "xgb_cox": {"concordance": 0.77, "ibs": 0.15},
    },
    cv_results={
        "coxph": [0.70, 0.73, 0.71, 0.74, 0.72],
        "rsf": [0.74, 0.76, 0.73, 0.77, 0.75],
        "xgb_cox": [0.75, 0.78, 0.76, 0.79, 0.77],
    },
)

# Get comparison results
comparison = comparator.compare(primary_metric="concordance")
comparator.print_comparison()

# Select best model
best = comparator.select_best(SelectionCriterion.BEST_METRIC)
print(f"Best model: {best.selected_model}")
```

### Selection Criteria

| Criterion | Description |
|-----------|-------------|
| `BEST_METRIC` | Highest primary metric score |
| `BEST_AVERAGE` | Best normalized average across all metrics |
| `MOST_STABLE` | Lowest cross-validation variance |
| `PARETO_OPTIMAL` | Best trade-off between performance and stability |
| `ENSEMBLE_TOP_K` | Recommend ensemble of top K models |

### Comparison Table Output

```
🏆 Model Comparison
┌──────┬─────────┬────────────┬────────┬───────────┐
│ Rank │ Model   │ Concordance│ CV Std │ Stability │
├──────┼─────────┼────────────┼────────┼───────────┤
│  🥇  │ xgb_cox │    0.7700  │ 0.0158 │   0.984   │
│  🥈  │ rsf     │    0.7500  │ 0.0158 │   0.984   │
│  3   │ coxph   │    0.7200  │ 0.0158 │   0.984   │
└──────┴─────────┴────────────┴────────┴───────────┘
```

## Artifact Manifest System

Every pipeline run generates a comprehensive manifest for reproducibility:

```python
from clinical_survival.artifact_manifest import ManifestManager

# Create manifest manager
manager = ManifestManager(Path("results"), run_name="experiment_v1")

# Start run (captures environment, git info, etc.)
manager.start_run(params_config, features_config, grid_config)

# Add artifacts during execution
manager.add_artifact("model", model_path, category="models")
manager.add_metric("concordance", 0.75, model="coxph")
manager.add_step_duration("training_loop", 45.2)

# Finalize and save
manifest = manager.finalize(success=True)
manifest_path = manager.save()
```

### Manifest Contents

The manifest JSON includes:

```json
{
  "manifest_version": "1.0",
  "run_id": "20231203_143052_a1b2c3d4",
  "run_name": "experiment_v1",
  "created_at": "2023-12-03T14:30:52.123Z",
  "completed_at": "2023-12-03T14:45:23.456Z",
  "duration_seconds": 871.333,
  "status": "completed",
  "correlation_id": "abc123",
  "configuration": {
    "params": { "seed": 42, "n_splits": 5, ... },
    "features": { "numerical_cols": [...], ... },
    "grid": { "coxph": { "alpha": 0.1 }, ... }
  },
  "environment": {
    "python_version": "3.11.5",
    "git_commit": "abc123def",
    "git_branch": "main",
    "git_dirty": false,
    "package_versions": { "numpy": "1.24.0", ... }
  },
  "artifacts": [
    {
      "name": "coxph_model",
      "path": "results/artifacts/models/coxph.joblib",
      "category": "models",
      "checksum": "sha256:abc123...",
      "size_bytes": 102400
    }
  ],
  "metrics": {
    "coxph": { "concordance": 0.75, "ibs": 0.18 }
  },
  "best_model": {
    "name": "xgb_cox",
    "metrics": { "concordance": 0.77 }
  }
}
```

### Loading Previous Manifests

```python
# Load a previous manifest
manifest = ManifestManager.load("results/artifacts/manifests/manifest_20231203.json")

print(f"Run ID: {manifest.run_id}")
print(f"Best model: {manifest.best_model}")
print(f"Duration: {manifest.duration_seconds:.1f}s")
```

## Comprehensive Model Evaluation

The pipeline includes a complete evaluation module with survival-specific metrics:

```python
from clinical_survival.pipeline.evaluator import (
    evaluate_survival_model,
    create_evaluation_summary,
)

# Full model evaluation
result = evaluate_survival_model(
    y_train=y_train,
    y_test=y_test,
    risk_scores=predictions,
    survival_probs=surv_probs,
    times=[365, 730, 1095],
    model_name="xgb_cox",
    n_bootstrap=200,
)

print(f"C-index: {result.concordance_index:.4f} ± {result.concordance_index_std:.4f}")
print(f"IBS: {result.integrated_brier_score:.4f}")
print(f"Mean AUC: {result.mean_auc:.4f}")
```

### Metrics Calculated

| Metric | Description |
|--------|-------------|
| Concordance Index | Discrimination ability (with 95% CI) |
| Integrated Brier Score | Overall prediction accuracy |
| Time-dependent Brier | Accuracy at specific time points |
| Time-dependent AUC | Discrimination at specific times |
| Event Rate | Proportion of events in data |

## Model Inference CLI

Run predictions on new data using trained models:

```bash
# Batch predictions
clinical-ml inference predict models/xgb_cox.joblib new_patients.csv -o predictions.csv

# Single patient prediction
clinical-ml inference single models/xgb_cox.joblib '{"age": 65, "stage": 2}'

# Explain a prediction
clinical-ml inference explain models/xgb_cox.joblib patients.csv --index 0

# Compare multiple models
clinical-ml inference compare "models/coxph.joblib,models/rsf.joblib" test_data.csv
```

### Python API

```python
from clinical_survival.inference import ModelInference

# Load model
inference = ModelInference.load("models/xgb_cox.joblib")

# Single prediction with survival curve
result = inference.predict(patient_features, times=[365, 730, 1095])
print(f"Risk: {result.risk_score:.4f}")
print(f"Category: {result.risk_category}")
print(f"1-year survival: {result.survival_probabilities[365]:.1%}")

# Batch predictions
results = inference.predict_batch(patients_df)
predictions_df = results.to_dataframe()
```

## Data Profiling

Automated exploratory data analysis for survival datasets:

```python
from clinical_survival.data_profiling import DataProfiler, profile_data

# Quick profile
profile = profile_data(df, time_col="time", event_col="event")

# Detailed profiler
profiler = DataProfiler(df, time_col="time", event_col="event")
profiler.generate_profile()
profiler.print_summary()  # Rich console output
profiler.save_report("data_profile.html")  # HTML report
```

### Profile Contents

- **Summary Statistics**: Rows, columns, memory usage
- **Column Profiles**: Type, missing values, unique values, distributions
- **Survival Statistics**: Event rate, time distribution, KM estimates
- **Correlations**: Numeric feature correlations
- **Warnings**: High missing values, constant columns, etc.

### Sample Output

```
📊 Dataset Overview
Rows: 10,000  |  Columns: 25  |  Memory: 2.45 MB

Survival Statistics:
  Events: 3,247 (32.5%)
  Censored: 6,753 (67.5%)
  Time range: 1 - 2,556 days
  KM median survival: 892 days
  1-year survival: 78.3%
```

## Metrics & Model Comparison Artifacts

During training, metrics are now persisted for reporting and selection:

- `metrics/leaderboard.csv`: Per-model metrics (C-index, IBS, mean AUC)
- `metrics/model_comparison.json`: Full comparison report
- `metrics/best_model.json`: Selected best model and criterion

These are also added to the artifact manifest for reproducibility.

## Checkpoint and Resume

Resume interrupted pipeline runs from the last successful step:

```python
from clinical_survival.checkpoint import CheckpointManager

manager = CheckpointManager("results/checkpoints")
manager.start_run(pipeline_steps=["load", "preprocess", "train"])

# After each step
manager.save_checkpoint("load", context)
manager.mark_step_completed("load")

# Resume
context = manager.load_latest_checkpoint()
resume_from = manager.get_resume_step()
```

### CLI Usage

```bash
# Run with checkpointing (default on)
clinical-ml training run --config configs/params.yaml --grid configs/model_grid.yaml

# Resume the most recent failed run
clinical-ml training resume --outdir results

# Resume a specific run-id
clinical-ml training resume --outdir results --run-id 20231203_143052_a1b2c3d4

# List available checkpoint runs
clinical-ml training list-checkpoints --outdir results
```

### Features

- **Automatic Checkpointing**: State saved after each pipeline step
- **Smart Resume**: Skip completed steps when resuming
- **Context Preservation**: Pipeline context serialized with IDs
- **Manifest Integration**: Metrics and comparisons recorded for traceability

## Contributing

Contributions are welcome! Please see the `CONTRIBUTING.md` file for details.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
