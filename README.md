# Clinical Survival ML

Reproducible end-to-end survival modelling for tabular clinical outcomes. The toolkit ingests time-to-event data, preprocesses covariates, fits Cox PH, Random Survival Forests, and XGBoost-based survival models, evaluates discrimination and calibration, performs decision-curve analysis, and renders a publication-ready HTML report complete with SHAP- and PDP-based interpretability.

## Quickstart

### Automated Installation (Recommended)

```bash
# Clone the repository and run the automated installer
make install
# or manually:
./install.sh
```

This will automatically handle dependency installation and package setup.

### Conda / mamba

```bash
mamba env create -f env/environment.yml
mamba activate clinical-survival-ml
make install
pre-commit install
```

### Docker

#### Quick Start with Docker Compose (Recommended)

```bash
# Build and start the API server
make deploy-build
make deploy-serve

# Or use the deployment script directly
./deploy.sh build
./deploy.sh serve
```

#### Manual Docker Commands

```bash
# Build the image
docker build -t clinical-survival-ml .

# Run training
docker run --rm -v $(pwd):/workspace clinical-survival-ml run \
  --config configs/params.yaml --grid configs/model_grid.yaml

# Start API server (after training)
docker run -p 8000:8000 -v $(pwd)/results:/workspace/results \
  clinical-survival-ml serve --models-dir results/artifacts/models
```

### Troubleshooting Installation

If you encounter pip compatibility issues, the installer will automatically fall back to alternative methods:

- **requirements.txt**: Uses a traditional requirements file for maximum compatibility
- **Conda/Mamba**: Uses conda-forge packages when available
- **Direct installation**: Installs the package without development dependencies

Then run the toy workflow:

```bash
clinical-ml run --config configs/params.yaml --grid configs/model_grid.yaml
```

## Deployment & API

After training models, you can deploy them as a REST API:

```bash
# Serve the trained models
clinical-ml serve --models-dir results/artifacts/models

# Access the API at http://localhost:8000
# Interactive documentation at http://localhost:8000/docs
```

### API Endpoints

- `GET /` - API information and available endpoints
- `GET /health` - Health check
- `GET /models` - List available models
- `POST /predict` - Make survival predictions

Example prediction request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age": 65,
      "sex": "male",
      "sofa": 8
    },
    "time_horizons": [90, 180, 365]
  }'
```

## Configuring Experiments

- `configs/params.yaml` sets seeds, CV folds, evaluation time grid, missing-data strategy, and I/O paths.
- `configs/model_grid.yaml` defines per-model hyperparameter grids.
- `configs/features.yaml` defines numeric/categorical features and optional drop columns.

### Ensemble Model Configuration

Configure ensemble methods in `configs/model_grid.yaml`:

```yaml
# Stacking ensemble - combines multiple models with meta-learning
stacking:
  base_models: [["coxph", "rsf"], ["coxph", "rsf", "xgb_cox"]]
  cv_folds: [3, 5]

# Bagging ensemble - bootstrap aggregation for stability
bagging:
  base_model: ["rsf", "xgb_cox"]
  n_estimators: [5, 10]
  max_samples: [0.8, 1.0]

# Dynamic ensemble - automatic model selection
dynamic:
  base_models: [["coxph", "rsf", "xgb_cox"]]
  selection_method: ["performance", "diversity"]

### Model Monitoring

Monitor model performance and detect drift:

```bash
# Monitor model predictions for drift detection
clinical-ml monitor --config configs/params.yaml --data data/new_data.csv

# Check for drift and performance issues
clinical-ml drift --config configs/params.yaml --model coxph --days 7

# View monitoring status dashboard
clinical-ml monitoring-status --config configs/params.yaml

# Reset monitoring baselines
clinical-ml reset-monitoring --config configs/params.yaml --confirm
```

The monitoring system tracks:
- **Performance metrics** (concordance, Brier score) over time
- **Feature drift** using statistical tests (Jensen-Shannon divergence)
- **Data quality** indicators (missing rates, distribution changes)
- **Automated alerts** for performance degradation or significant drift
- **Trend analysis** for long-term model reliability assessment
```

### Configuration Validation

Before running experiments, validate your configuration files:

```bash
clinical-ml validate-config
```

This command checks that all configuration files conform to their expected schemas and provides helpful error messages for any issues. Run this before training to catch configuration problems early.

Modify the dataset paths in `params.yaml` to target your clinical dataset (CSV) and accompanying `metadata.yaml` describing column types.

## Automated Model Selection (AutoML)

The toolkit includes advanced automated machine learning capabilities for optimal model selection and hyperparameter tuning:

### AutoML Features

- **Bayesian Optimization**: Uses Optuna for intelligent hyperparameter search
- **Multi-Model Selection**: Automatically compares Cox PH, RSF, XGBoost Cox/AFT, and ensemble models
- **Early Stopping**: Stops optimization when no improvement is detected
- **Cross-Validation**: Proper evaluation during optimization to prevent overfitting

### Installation for AutoML

```bash
# Install with AutoML dependencies
pip install clinical-survival-ml[automl]

# Or install optuna separately
pip install optuna>=3.0
```

### AutoML Usage

```bash
# Run automated optimization for 30 minutes
clinical-ml automl --config configs/params.yaml \
                   --data data/toy/toy_survival.csv \
                   --meta data/toy/metadata.yaml \
                   --time-limit 1800 \
                   --model-types coxph rsf xgb_cox xgb_aft stacking

# Optimize for specific models only
clinical-ml automl --config configs/params.yaml \
                   --data data/your_data.csv \
                   --meta data/metadata.yaml \
                   --time-limit 3600 \
                   --model-types xgb_cox xgb_aft \
                   --metric concordance \
                   --output-dir results/automl

# Use different optimization metric
clinical-ml automl --config configs/params.yaml \
                   --data data/your_data.csv \
                   --meta data/metadata.yaml \
                   --time-limit 1800 \
                   --metric ibs
```

### AutoML Outputs

The AutoML command generates:
- `results/automl/automl_results.json`: Best model parameters and performance
- `results/automl/optuna_study.pkl`: Complete optimization study for analysis
- `results/automl/best_model.pkl`: Trained best model ready for deployment

## GPU Acceleration & Performance

The toolkit includes intelligent GPU acceleration for XGBoost models with automatic hardware detection:

### Hardware Detection

```bash
# Check hardware capabilities and benchmark performance
clinical-ml benchmark-hardware --config configs/params.yaml \
                               --data data/toy/toy_survival.csv \
                               --meta data/toy/metadata.yaml \
                               --model-type xgb_cox

# Test specific GPU device
clinical-ml benchmark-hardware --config configs/params.yaml \
                               --data data/your_data.csv \
                               --meta data/metadata.yaml \
                               --model-type xgb_aft \
                               --gpu-id 1
```

### Automatic GPU Usage

- **Auto-detection**: Automatically detects CUDA-compatible GPUs
- **Intelligent Fallback**: Falls back to CPU parallel processing when GPU unavailable
- **Multi-GPU Support**: Can utilize multiple GPUs when available
- **Performance Monitoring**: Benchmarks both CPU and GPU performance for optimal configuration

### GPU Features

- **XGBoost GPU Support**: Uses `tree_method="gpu_hist"` for XGBoost models
- **Parallel Processing**: Leverages all available CPU cores when GPU unavailable
- **Memory Management**: Efficient memory usage for large datasets
- **Hardware Recommendations**: Provides guidance on optimal hardware configuration

### Performance Benefits

- **5-10x speedup** for XGBoost model training on GPU-equipped systems
- **Parallel cross-validation** for faster hyperparameter optimization
- **Scalable to large datasets** with efficient memory management
- **Future-ready** for deep learning survival models

## Counterfactual Explanations & Causal Inference

The toolkit includes advanced counterfactual explanation capabilities for generating actionable clinical insights:

### What are Counterfactual Explanations?

Counterfactual explanations answer the question: **"What would need to change for this patient to achieve a different outcome?"**

For example:
- "What changes in patient characteristics would reduce their risk from 0.7 to 0.3?"
- "What interventions would extend survival time from 6 months to 12 months?"

### Counterfactual Features

- **🎯 Multiple Optimization Methods**: Gradient descent, genetic algorithms, and random search
- **📏 Distance Constraints**: Control how different counterfactuals can be from original instances
- **🎛️ Flexible Targets**: Target specific risk levels or survival times
- **🔬 Causal Inference**: Treatment effect estimation and feature importance analysis
- **📊 Actionable Insights**: Clear visualization of required changes for clinicians

### Counterfactual Usage

```bash
# Generate counterfactuals for a specific risk target
clinical-ml counterfactual --config configs/params.yaml \
                          --data data/toy/toy_survival.csv \
                          --meta data/toy/metadata.yaml \
                          --model xgb_cox \
                          --target-risk 0.3 \
                          --n-counterfactuals 3

# Generate counterfactuals for survival time target
clinical-ml counterfactual --config configs/params.yaml \
                          --data data/your_data.csv \
                          --meta data/metadata.yaml \
                          --model rsf \
                          --target-time 365 \
                          --method genetic

# Generate multiple counterfactuals with different methods
clinical-ml counterfactual --config configs/params.yaml \
                          --data data/your_data.csv \
                          --meta data/metadata.yaml \
                          --model xgb_aft \
                          --n-counterfactuals 5 \
                          --output-dir results/counterfactuals
```

### Counterfactual Methods

1. **Gradient-based Optimization**: Fast, deterministic optimization using gradients
2. **Genetic Algorithm**: Population-based evolutionary approach for complex optimization landscapes
3. **Random Search**: Simple but effective baseline method

### Counterfactual Outputs

The counterfactual command generates:
- `results/counterfactuals/counterfactual_results.json`: Complete explanation data
- **Interactive Display**: Shows original prediction, target outcomes, and required changes
- **Feature Changes**: Detailed breakdown of what needs to change and by how much
- **Distance Metrics**: Quantifies how different counterfactuals are from original instances

### Example Counterfactual Output

```
🔍 Generating Counterfactual Explanations
==================================================
🎯 Model: xgb_cox
🔢 Counterfactuals: 3
⚙️  Method: gradient
📊 Target risk: 0.3

✅ Counterfactual generation completed!
📈 Original risk: 0.65

🎯 Generated 3 counterfactuals:

  1. Target risk: 0.30
     Distance: 1.23
     Key changes:
       age: ↓ 5.2 years
       sofa: ↓ 2.1 points
       stage: ↓ 0.8 levels

  2. Target risk: 0.30
     Distance: 0.89
     Key changes:
       creatinine: ↓ 0.3 mg/dL
       bilirubin: ↓ 1.1 mg/dL

📊 Summary:
   Total counterfactuals: 3
   Average distance: 1.05
   Min/Max distance: 0.89 / 1.23

💾 Results saved to results/counterfactuals/counterfactual_results.json
🎉 Counterfactual explanation generation completed!
```

### Causal Inference Capabilities

```python
from clinical_survival.counterfactual import CausalInference

# Estimate treatment effects
causal_analyzer = CausalInference(model, feature_names, treatment_features=['treatment'])
treatment_effects = causal_analyzer.estimate_treatment_effect(
    X_test, 'treatment', [0, 1], outcome_type='risk'
)

# Feature importance for causal understanding
importance = causal_analyzer.feature_importance_causal(
    X_test, outcome_type='risk', method='shap'
)
```

## Incremental Learning & Online Model Updates

The toolkit supports incremental learning to continuously improve models as new clinical data becomes available, without requiring full retraining:

### Incremental Learning Features

- **Online Learning**: XGBoost models support true online updates using booster updates
- **Batch Updates**: Process accumulated new data in batches for other model types
- **Sliding Window**: Maintain models on recent data windows for concept drift adaptation
- **Automatic Drift Detection**: Monitor data distribution changes and trigger updates
- **Model Backup**: Automatic backups before updates with configurable retention
- **Performance Monitoring**: Track model performance changes after each update

### Incremental Learning Configuration

Enable and configure incremental learning in your configuration:

```yaml
# In configs/params.yaml
incremental_learning:
  enabled: true
  update_frequency_days: 7        # Check for updates every 7 days
  min_samples_for_update: 50      # Minimum new samples before updating
  max_samples_in_memory: 1000     # Maximum samples to keep in buffer
  update_strategy: "online"       # "online", "batch", or "sliding_window"
  drift_detection_enabled: true   # Enable drift detection
  create_backup_before_update: true
  backup_retention_days: 30
```

### Incremental Learning Usage

```bash
# Configure incremental learning settings
clinical-ml configure-incremental \
  --config-path configs/incremental_config.json \
  --update-frequency-days 7 \
  --min-samples-for-update 50 \
  --max-samples-in-memory 1000 \
  --update-strategy online \
  --drift-detection-enabled \
  --create-backup-before-update

# Update models with new patient data
clinical-ml update-models \
  --config configs/params.yaml \
  --data data/new_patients.csv \
  --meta data/new_metadata.yaml \
  --models-dir results/artifacts/models \
  --model-names coxph xgb_cox \
  --incremental-config configs/incremental_config.json

# Check incremental learning status
clinical-ml incremental-status \
  --models-dir results/artifacts/models \
  --model-names coxph xgb_cox
```

### Incremental Learning Outputs

The incremental learning system generates:
- `results/artifacts/models/incremental_update_history.json`: Complete update history with performance metrics
- Model backups in `results/artifacts/models/` with timestamp suffixes
- Automatic cleanup of old backups based on retention policy

### Integration with Monitoring

The monitoring system automatically triggers incremental updates when:
- **Performance Degradation**: Concordance drops below threshold
- **Data Drift**: Feature distributions change significantly
- **Scheduled Updates**: Based on configured update frequency

This ensures models stay current with evolving clinical data while maintaining performance and reliability.

## Distributed Computing for Large-Scale Datasets

The toolkit supports distributed computing to handle enterprise-scale clinical datasets efficiently using Dask and Ray frameworks:

### Distributed Computing Features

- **Multiple Frameworks**: Support for Dask, Ray, and local computing
- **Intelligent Partitioning**: Automatic data partitioning with balanced, hash, and random strategies
- **Scalable Training**: Distributed model training across multiple workers and nodes
- **Performance Benchmarking**: Built-in benchmarking tools to measure scaling efficiency
- **Fault Tolerance**: Automatic retry of failed tasks and graceful error handling
- **Resource Management**: Configurable memory, CPU, and GPU allocation per worker

### Distributed Computing Configuration

Configure distributed computing in your main configuration:

```yaml
# In configs/params.yaml
distributed_computing:
  enabled: true
  cluster_type: "dask"          # "local", "dask", "ray"
  n_workers: 8                  # Number of worker processes
  threads_per_worker: 2         # Threads per worker
  memory_per_worker: "4GB"      # Memory per worker
  partition_strategy: "balanced" # "balanced", "hash", "random"
  n_partitions: 16              # Number of data partitions
  scheduler_address: "127.0.0.1:8786"
  dashboard_address: "127.0.0.1:8787"
```

Or create a dedicated distributed configuration file:

```json
{
  "cluster_type": "dask",
  "n_workers": 8,
  "threads_per_worker": 2,
  "memory_per_worker": "4GB",
  "partition_strategy": "balanced",
  "n_partitions": 16,
  "scheduler_address": "127.0.0.1:8786",
  "dashboard_address": "127.0.0.1:8787",
  "chunk_size": 1000,
  "optimize_memory": true,
  "use_gpu_if_available": true,
  "retry_failed_tasks": true,
  "max_retries": 3,
  "timeout_minutes": 60,
  "resource_allocation_strategy": "balanced"
}
```

### Distributed Computing Usage

```bash
# Configure distributed computing settings
clinical-ml configure-distributed \
  --config-path configs/distributed_config.json \
  --cluster-type dask \
  --n-workers 8 \
  --memory-per-worker 4GB \
  --partition-strategy balanced \
  --n-partitions 16

# Benchmark scaling performance across dataset sizes
clinical-ml distributed-benchmark \
  --config configs/params.yaml \
  --cluster-type dask \
  --n-workers 8 \
  --dataset-sizes 1000 5000 10000 25000 \
  --model-type xgb_cox \
  --output-dir results/distributed_benchmark

# Train model on large dataset using distributed computing
clinical-ml distributed-train \
  --config configs/params.yaml \
  --data data/large_clinical_dataset.csv \
  --meta data/large_metadata.yaml \
  --cluster-type dask \
  --n-workers 8 \
  --n-partitions 16 \
  --model-type xgb_cox \
  --output-dir results/distributed_training

# Evaluate model performance using distributed computing
clinical-ml distributed-evaluate \
  --config configs/params.yaml \
  --data data/test_dataset.csv \
  --meta data/test_metadata.yaml \
  --model results/artifacts/models/distributed_xgb_cox.pkl \
  --cluster-type dask \
  --n-workers 4 \
  --n-partitions 8 \
  --metrics concordance ibs brier
```

### Distributed Computing Outputs

The distributed computing system generates:
- `results/distributed_benchmark/benchmark_results.json`: Detailed scaling analysis and performance metrics
- `results/distributed_training/distributed_metrics.json`: Training performance and resource usage statistics
- Model files in distributed training output directories
- Dashboard access for monitoring distributed tasks (Dask/Ray dashboards)

### Scaling Performance

The distributed computing system is designed to scale efficiently:
- **Linear Scaling**: Ideal linear scaling for embarrassingly parallel tasks
- **Communication Overhead**: Minimized through intelligent partitioning
- **Memory Efficiency**: Distributed data loading and processing
- **Fault Tolerance**: Automatic retry of failed tasks

### Integration with Other Features

Distributed computing integrates seamlessly with:
- **Incremental Learning**: Distributed updates for large datasets
- **Model Monitoring**: Distributed evaluation and drift detection
- **AutoML**: Distributed hyperparameter optimization
- **GPU Acceleration**: Distributed GPU training when available

This enables processing of enterprise-scale clinical datasets (100k+ patients) while maintaining the same simple interface and comprehensive evaluation capabilities.

## CLI Commands

```
clinical-ml validate-config --config configs/params.yaml --grid configs/model_grid.yaml --features configs/features.yaml
clinical-ml load --data data/toy/toy_survival.csv --meta data/toy/metadata.yaml
clinical-ml train --config configs/params.yaml --grid configs/model_grid.yaml
clinical-ml automl --config configs/params.yaml --data data/toy/toy_survival.csv --meta data/toy/metadata.yaml --time-limit 1800
clinical-ml benchmark-hardware --config configs/params.yaml --data data/toy/toy_survival.csv --meta data/toy/metadata.yaml
clinical-ml counterfactual --config configs/params.yaml --data data/toy/toy_survival.csv --meta data/toy/metadata.yaml --model xgb_cox --target-risk 0.3
clinical-ml update-models --config configs/params.yaml --data data/new_patients.csv --meta data/new_metadata.yaml --models-dir results/artifacts/models
clinical-ml incremental-status --models-dir results/artifacts/models
clinical-ml configure-incremental --config-path configs/incremental_config.json --update-frequency-days 7 --min-samples-for-update 50
clinical-ml distributed-benchmark --config configs/params.yaml --cluster-type dask --n-workers 8 --dataset-sizes 1000 5000 10000
clinical-ml distributed-train --config configs/params.yaml --data data/large_dataset.csv --cluster-type dask --n-workers 8 --model-type xgb_cox
clinical-ml distributed-evaluate --config configs/params.yaml --data data/test_data.csv --model results/artifacts/models/xgb_cox.pkl --cluster-type dask --n-workers 4
clinical-ml configure-distributed --config-path configs/distributed_config.json --cluster-type dask --n-workers 8 --memory-per-worker 4GB
clinical-ml evaluate --config configs/params.yaml
clinical-ml monitor --config configs/params.yaml --data data/toy/toy_survival.csv
clinical-ml drift --config configs/params.yaml --model coxph --days 7
clinical-ml monitoring-status --config configs/params.yaml
clinical-ml explain --config configs/params.yaml --model xgb_cox
clinical-ml report --config configs/params.yaml --out results/report.html
clinical-ml serve --models-dir results/artifacts/models
clinical-ml run --config configs/params.yaml --grid configs/model_grid.yaml
```

## External Validation

Specify an external split either by providing a secondary CSV (`paths.external_csv`) or by tagging rows with a `group` column (`train`/`external`). Adjust `external.label` to version the persisted model artefacts under `artifacts/models/{label}/`.

## Outputs

Running `clinical-ml run` populates `results/` with:

- `artifacts/models/` – serialized transformers and fitted estimators.
- `artifacts/metrics/` – cross-validation summaries, leaderboards, calibration and net benefit plots.
- `artifacts/explain/` – permutation importances, SHAP summaries, PDPs.
- `report.html` – compiled HTML report driven by `configs/report_template.html.j2`.

## Methods Overview

- **Preprocessing** – Iterative imputation for numeric features, one-hot encoding for categorical, scaling for continuous covariates.
- **Models** – Baseline Cox PH (scikit-survival), Random Survival Forest, XGBoost Cox and AFT.
- **Ensemble Methods** – Advanced stacking, bagging, and dynamic model selection for improved accuracy:
  - **Stacking**: Combines multiple models using meta-learning for optimal prediction
  - **Bagging**: Bootstrap aggregation reduces overfitting and improves stability
  - **Dynamic Selection**: Automatically selects the best models based on data characteristics
- **Model Monitoring** – Real-time drift detection and performance tracking:
  - **Concept Drift Detection**: Statistical tests for changes in prediction distributions
  - **Data Quality Monitoring**: Feature distribution and missing data pattern tracking
  - **Performance Alerts**: Automated notifications for model degradation
  - **Baseline Management**: Historical performance comparison and trend analysis
- **Tuning** – Nested cross-validation with stratified outer/inner folds, scored by concordance and IBS.
- **AutoML** – Automated model selection and Bayesian hyperparameter optimization using Optuna for finding optimal model architectures and parameters.
- **GPU Acceleration** – Automatic GPU detection and acceleration for XGBoost models, with intelligent fallback to CPU parallel processing.
- **Evaluation** – Out-of-fold (OOF) Harrell's C-index, time-dependent Brier scores, integrated Brier score, IPCW calibration curves, and censoring-aware decision-curve net benefit.
- **Counterfactual Explanations** – Generate "what-if" scenarios showing minimal feature changes needed to achieve different outcomes, enabling actionable clinical decision support.
- **Explainability** – Permutation importance, SHAP (tree models), partial dependence plots.

## Evaluation Guarantees

- The preprocessing pipeline (imputation, scaling, encoding) is refit inside every CV fold. No transformers are prefit on the full dataset prior to cross-validation.
- Leaderboard metrics are computed exclusively from aggregated OOF predictions and include bootstrap confidence intervals.
- External/holdout evaluation uses the trained model only for prediction; no training data is reused in metric computation.
- Calibration and decision-curve analyses apply inverse probability of censoring weighting (IPCW) at the requested horizons, ensuring censoring-aware reliability and net-benefit estimates.
- A single global seed controls NumPy, Python, scikit-learn, and XGBoost RNGs. Re-running with the same seed produces identical OOF metrics and artifacts.

## Testing & CI

- `make unit` – run the pytest suite (uses synthetic toy data under `data/toy/`).
- `make smoke` – executes the full CLI pipeline on the toy dataset.
- GitHub Actions workflow (`.github/workflows/ci.yml`) provisions the Conda environment, runs linting, unit tests, smoke test, and uploads `results/` artifacts.

## License

MIT — see `LICENSE`.
