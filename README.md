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

## Advanced Clinical Interpretability & Decision Support

The toolkit provides advanced interpretability tools specifically designed for clinical decision support and regulatory compliance:

### Clinical Interpretability Features

- **Enhanced SHAP Integration**: Clinical context-aware SHAP explanations with medical domain knowledge
- **Risk Stratification**: Automated patient categorization with urgency levels and clinical recommendations
- **Clinical Decision Support**: Comprehensive reports with actionable clinical insights
- **Interactive HTML Reports**: Web-based dashboards for clinical review and decision-making
- **Population Insights**: Aggregate analysis of risk factors and treatment patterns
- **Regulatory Compliance**: Documentation and audit trails for healthcare AI requirements

### Clinical Interpretability Configuration

Configure clinical interpretability in your main configuration:

```yaml
# In configs/params.yaml
clinical_interpretability:
  enabled: true
  risk_thresholds:
    low: 0.2
    moderate: 0.4
    high: 0.7
    very_high: 1.0
  clinical_context:
    feature_domains:
      age: "demographics"
      sex: "demographics"
      sofa: "vitals"
      stage: "comorbidities"
      creatinine: "labs"
      bilirubin: "labs"
    risk_categories:
      protective: ["age", "sex"]
      risk_factor: ["sofa", "stage", "creatinine", "bilirubin"]
      neutral: []
  explanation_features:
    include_shap_values: true
    include_clinical_interpretation: true
    include_risk_stratification: true
    include_recommendations: true
    include_confidence_scores: true
```

### Clinical Interpretability Usage

```bash
# Generate comprehensive clinical interpretability report (HTML format)
clinical-ml clinical-interpret \
  --config configs/params.yaml \
  --data data/patient_data.csv \
  --meta data/metadata.yaml \
  --model results/artifacts/models/coxph.pkl \
  --output-dir results/clinical_reports \
  --output-format html \
  --patient-ids patient_001 patient_002 patient_003

# Generate risk stratification report
clinical-ml risk-stratification \
  --config configs/params.yaml \
  --data data/patient_data.csv \
  --meta data/metadata.yaml \
  --model results/artifacts/models/coxph.pkl \
  --output-dir results/risk_analysis
```

### Clinical Interpretability Outputs

The clinical interpretability system generates:
- `results/clinical_interpretability/clinical_interpretability_report.html`: Interactive HTML dashboard with patient-specific explanations
- `results/clinical_interpretability/clinical_interpretability_report.json`: Detailed JSON report for programmatic analysis
- `results/risk_stratification/risk_stratification_results.json`: Patient risk stratification with clinical recommendations

### Enhanced SHAP Explanations

The system provides enhanced SHAP explanations with:
- **Clinical Context**: Medical domain mapping for features (demographics, vitals, labs, comorbidities)
- **Risk Categorization**: Automatic risk level assessment with confidence intervals
- **Feature Importance**: Domain-specific feature importance ranking
- **Clinical Interpretation**: Natural language explanations of model predictions
- **Actionable Insights**: Specific clinical recommendations based on risk factors

### Risk Stratification System

Automated patient risk stratification includes:
- **Risk Categories**: Low, moderate, high, very high with configurable thresholds
- **Urgency Levels**: Routine, urgent, critical based on risk factors and clinical context
- **Clinical Recommendations**: Evidence-based recommendations for each risk category
- **Population Insights**: Aggregate analysis of risk patterns across patient cohorts
- **Confidence Assessment**: Statistical confidence intervals for risk assessments

### Clinical Decision Support Integration

The system integrates with:
- **Model Monitoring**: Automatic triggering of interpretability analysis for high-risk predictions
- **Counterfactual Explanations**: "What-if" scenarios for treatment planning
- **Population Health**: Aggregate insights for quality improvement initiatives
- **Regulatory Compliance**: Documentation for healthcare AI regulatory requirements

### Example Clinical Report Output

```html
<!-- Interactive HTML Report -->
<div class="patient-card risk-high">
    <h3>Patient ID: patient_001</h3>
    <p><strong>Risk Category:</strong> HIGH</p>
    <p><strong>Predicted Risk:</strong> 0.75</p>
    <p><strong>Confidence:</strong> 0.85</p>

    <div class="clinical-interpretation">
        <h4>Clinical Interpretation</h4>
        <p>High risk prediction (0.75) driven by elevated SOFA score and advanced disease stage</p>
    </div>

    <div class="key-findings">
        <h4>Key Findings</h4>
        <ul>
            <li>Elevated SOFA score increases predicted risk</li>
            <li>Advanced disease stage contributes significantly</li>
        </ul>
    </div>

    <div class="recommendations">
        <h4>Clinical Recommendations</h4>
        <ul>
            <li>Monitor organ function closely</li>
            <li>Consider specialist consultation</li>
            <li>Evaluate for clinical trial eligibility</li>
        </ul>
    </div>
</div>
```

This advanced interpretability system transforms ML predictions into clinically actionable insights, enabling safer and more effective use of AI in healthcare settings.

## Automated MLOps Pipeline & Model Lifecycle Management

The toolkit provides a comprehensive MLOps system for production-ready model management, automated retraining, and deployment lifecycle management:

### MLOps Features

- **Model Registry**: Version control with metadata, performance metrics, and approval workflows
- **Automated Retraining**: Scheduled model updates based on performance, drift, or data volume triggers
- **Deployment Management**: Multi-environment deployment with traffic splitting and rollback capabilities
- **A/B Testing Framework**: Statistical comparison of model versions with automated promotion
- **Approval Workflows**: Configurable approval processes for production deployments
- **Audit Trails**: Complete lifecycle tracking for regulatory compliance

### MLOps Configuration

Configure MLOps in your main configuration:

```yaml
# In configs/params.yaml
mlops:
  enabled: true
  registry_path: "results/mlops"
  environments:
    development:
      type: "development"
      auto_rollback: false
      rollback_threshold: 0.1
    staging:
      type: "staging"
      auto_rollback: true
      rollback_threshold: 0.05
    production:
      type: "production"
      auto_rollback: true
      rollback_threshold: 0.02
  triggers:
    daily_retrain:
      enabled: true
      trigger_type: "scheduled"
      schedule_cron: "0 2 * * *"
      require_approval: true
    performance_monitor:
      enabled: true
      trigger_type: "performance"
      performance_threshold: 0.05
      auto_retrain: false
      require_approval: true
    drift_detection:
      enabled: true
      trigger_type: "drift"
      drift_threshold: 0.1
      auto_retrain: false
      require_approval: true
  deployment_settings:
    require_approval_for_production: true
    auto_rollback_on_failure: true
    max_concurrent_deployments: 3
    deployment_timeout_minutes: 30
```

### MLOps Usage

```bash
# Register a trained model in the MLOps registry
clinical-ml register-model \
  --model results/artifacts/models/coxph.pkl \
  --model-name survival_model \
  --version-number 1.0.0 \
  --description "Cox PH survival model for clinical outcomes" \
  --tags clinical,survival,coxph \
  --registry-path results/mlops

# Check MLOps registry status
clinical-ml mlops-status \
  --registry-path results/mlops \
  --model-name survival_model

# Deploy model to staging environment
clinical-ml deploy-model \
  --version-id abc123def456 \
  --environment staging \
  --traffic-percentage 50 \
  --approved-by clinical_lead

# Create A/B test between model versions
clinical-ml create-ab-test \
  --test-name "Model Comparison Study" \
  --model-versions abc123 def456 \
  --traffic-split '{"abc123": 0.7, "def456": 0.3}' \
  --test-duration-days 14 \
  --success-metrics concordance ibs

# Check if retraining triggers should fire
clinical-ml check-retraining-triggers \
  --registry-path results/mlops \
  --model-name survival_model

# Rollback deployment if needed
clinical-ml rollback-deployment \
  --environment production \
  --target-version abc123 \
  --reason performance_degradation
```

### Model Lifecycle Management

The MLOps system manages the complete model lifecycle:

1. **Model Development**: Register initial model versions with metadata
2. **Testing & Validation**: Deploy to staging for validation and A/B testing
3. **Approval Process**: Require approval for production deployments
4. **Production Deployment**: Deploy to production with traffic splitting
5. **Monitoring & Drift Detection**: Continuous monitoring for performance and data drift
6. **Automated Retraining**: Trigger retraining based on configured conditions
7. **Version Management**: Track model lineage and relationships

### Deployment Environments

The system supports multiple deployment environments:

- **Development**: Local testing and experimentation
- **Staging**: Pre-production validation and testing
- **Production**: Live clinical deployment with safety measures

### Automated Retraining Triggers

Models can be automatically retrained based on:

- **Scheduled Triggers**: Time-based retraining (e.g., daily, weekly)
- **Performance Triggers**: When model performance drops below thresholds
- **Drift Triggers**: When data distribution changes significantly
- **Data Volume Triggers**: When sufficient new data becomes available

### A/B Testing Framework

The system provides robust A/B testing capabilities:

- **Statistical Significance**: Automatic calculation of statistical significance
- **Traffic Splitting**: Configurable traffic allocation between versions
- **Performance Monitoring**: Real-time performance comparison
- **Automated Promotion**: Statistical winner determination and promotion

### Integration with Monitoring

The MLOps system integrates seamlessly with:

- **Performance Monitoring**: Automatic trigger activation based on performance metrics
- **Drift Detection**: Automated retraining when data drift is detected
- **Clinical Interpretability**: Enhanced explanations for production models
- **Incremental Learning**: Distributed model updates in production

### MLOps Outputs

The MLOps system generates:
- `results/mlops/models.json`: Model registry metadata
- `results/mlops/versions.json`: Version-specific information
- `results/mlops/deployments/`: Deployment records and logs
- `results/mlops/ab_tests/`: A/B test results and analysis
- `results/mlops/audit_trail.json`: Complete audit trail for compliance

### Example MLOps Workflow

```bash
# 1. Train and register initial model
clinical-ml train --config configs/params.yaml --grid configs/model_grid.yaml
clinical-ml register-model --model results/artifacts/models/coxph.pkl --model-name survival_model --version-number 1.0.0

# 2. Deploy to staging for validation
clinical-ml deploy-model --version-id abc123 --environment staging --approved-by clinical_lead

# 3. Monitor performance and check for drift
clinical-ml monitor --config configs/params.yaml --data data/new_patients.csv
clinical-ml drift --config configs/params.yaml --model survival_model --days 7

# 4. Create A/B test for model improvement
clinical-ml create-ab-test --test-name "Hyperparameter Study" --model-versions abc123 def456 --traffic-split '{"abc123": 0.5, "def456": 0.5}'

# 5. Check triggers and potentially retrain
clinical-ml check-retraining-triggers --model-name survival_model

# 6. Promote best model to production
clinical-ml deploy-model --version-id def456 --environment production --traffic-percentage 100 --approved-by clinical_director
```

This comprehensive MLOps system enables safe, automated, and auditable model lifecycle management for clinical AI applications.

## Comprehensive Data Quality & Validation Framework

The toolkit provides a robust data quality and validation framework specifically designed for clinical datasets, ensuring data reliability and regulatory compliance:

### Data Quality Framework Features

- **Comprehensive Profiling**: Automated analysis of completeness, consistency, accuracy, and clinical validity
- **Anomaly Detection**: Statistical and ML-based detection of data anomalies and outliers
- **Clinical Validation**: Domain-specific validation rules for clinical data standards
- **Quality Monitoring**: Continuous monitoring of data quality with drift detection
- **Automated Cleansing**: Intelligent data cleaning and imputation strategies
- **Quality Scoring**: Standardized quality metrics with clinical grading (A-F scale)

### Data Quality Configuration

Configure data quality assessment in your main configuration:

```yaml
# In configs/params.yaml
data_quality:
  profiling:
    enable_anomaly_detection: true
    enable_clinical_validation: true
    quality_thresholds:
      excellent: 90.0
      good: 80.0
      acceptable: 70.0
      poor: 60.0
  validation:
    validation_rules: "default"
    strict_mode: false
    fail_on_first_error: false
  cleansing:
    remove_duplicates: true
    missing_values_strategy: "auto"
    handle_outliers: true
    outlier_method: "iqr"
    preserve_original: true
  monitoring:
    enable_continuous_monitoring: true
    monitoring_frequency_hours: 24
    drift_detection_enabled: true
    drift_threshold: 0.1
```

### Data Quality Usage

```bash
# Generate comprehensive data quality profile
clinical-ml data-quality-profile \
  --data data/patient_data.csv \
  --meta data/metadata.yaml \
  --output-dir results/data_quality \
  --output-format html \
  --include-anomaly-detection

# Validate dataset against clinical data rules
clinical-ml data-validation \
  --data data/patient_data.csv \
  --meta data/metadata.yaml \
  --validation-rules default \
  --output-dir results/data_validation \
  --strict-mode

# Cleanse dataset based on quality assessment
clinical-ml data-cleansing \
  --data data/patient_data.csv \
  --meta data/metadata.yaml \
  --output-dir results/data_cleansing \
  --remove-duplicates \
  --handle-outliers
```

### Data Quality Profiling

The system provides comprehensive data profiling:

- **Completeness Analysis**: Missing value detection and column completeness scoring
- **Statistical Profiling**: Distribution analysis, outlier detection, and normality testing
- **Clinical Validation**: Domain-specific checks for clinical data validity
- **Anomaly Detection**: ML-based anomaly detection using isolation forests
- **Drift Detection**: Statistical comparison with baseline data distributions

### Validation Rules Engine

Configurable validation rules for clinical data:

- **Completeness Rules**: Ensure required fields meet minimum completeness thresholds
- **Range Rules**: Validate that values fall within clinically meaningful ranges
- **Categorical Rules**: Ensure categorical values are from allowed sets
- **Clinical Rules**: Domain-specific validation for clinical measurements
- **Statistical Rules**: Distribution and normality validation

### Automated Data Cleansing

Intelligent data cleaning pipeline:

- **Duplicate Removal**: Automatic detection and removal of duplicate records
- **Missing Value Imputation**: Context-aware imputation strategies
- **Outlier Handling**: Statistical outlier detection and treatment
- **Clinical Value Correction**: Domain-specific value corrections
- **Data Transformation**: Standardization and normalization

### Quality Monitoring System

Continuous data quality monitoring:

- **Baseline Establishment**: Initial quality assessment as baseline
- **Trend Analysis**: Tracking quality changes over time
- **Drift Detection**: Statistical detection of data distribution changes
- **Alert Generation**: Automated alerts for quality degradation
- **Remediation Tracking**: Monitoring effectiveness of quality improvements

### Data Quality Outputs

The data quality system generates:
- `results/data_quality/data_quality_report.html`: Interactive HTML quality dashboard
- `results/data_quality/data_quality_report.json`: Detailed JSON quality metrics
- `results/data_validation/validation_results.json`: Validation rule results
- `results/data_cleansing/cleansing_summary.json`: Cleansing operation summary
- `results/data_monitoring/`: Continuous monitoring history and trends

### Clinical Data Quality Standards

The framework implements clinical data quality standards:

- **SOFA Score Validation**: Organ dysfunction severity scoring validation
- **Age Range Validation**: Realistic age value checking
- **Sex/Gender Consistency**: Cross-field consistency validation
- **Clinical Value Ranges**: Normal range validation for lab values
- **Data Type Validation**: Appropriate data types for clinical measurements

### Integration with Other Features

Data quality integrates seamlessly with:

- **Model Training**: Quality-gated model training with validation checks
- **Incremental Learning**: Quality monitoring for incremental data updates
- **Model Monitoring**: Data quality drift detection for model performance
- **Clinical Interpretability**: Quality-aware explanation generation
- **MLOps Pipeline**: Quality-based model promotion and deployment

### Example Data Quality Workflow

```bash
# 1. Profile incoming data
clinical-ml data-quality-profile --data data/new_patients.csv --output-format html

# 2. Validate against clinical standards
clinical-ml data-validation --data data/new_patients.csv --strict-mode

# 3. Cleanse data based on quality assessment
clinical-ml data-cleansing --data data/new_patients.csv --remove-duplicates --handle-outliers

# 4. Train model with quality-assured data
clinical-ml train --config configs/params.yaml --grid configs/model_grid.yaml

# 5. Monitor data quality continuously
clinical-ml monitor --config configs/params.yaml --data data/production_patients.csv
```

This comprehensive data quality framework ensures that clinical AI models are built on reliable, validated data, meeting the highest standards for healthcare applications.

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
clinical-ml clinical-interpret --config configs/params.yaml --data data/patients.csv --model results/artifacts/models/coxph.pkl --output-format html
clinical-ml risk-stratification --config configs/params.yaml --data data/patients.csv --model results/artifacts/models/coxph.pkl
clinical-ml register-model --model results/artifacts/models/coxph.pkl --model-name survival_model --version-number 1.0.0 --description "Cox PH survival model"
clinical-ml mlops-status --registry-path results/mlops --model-name survival_model
clinical-ml deploy-model --version-id abc123 --environment staging --traffic-percentage 50 --approved-by clinical_lead
clinical-ml create-ab-test --test-name "Model Comparison" --model-versions abc123 def456 --traffic-split '{"abc123": 0.7, "def456": 0.3}'
clinical-ml check-retraining-triggers --registry-path results/mlops --model-name survival_model
clinical-ml data-quality-profile --data data/patient_data.csv --output-format html
clinical-ml data-validation --data data/patient_data.csv --validation-rules default
clinical-ml data-cleansing --data data/patient_data.csv --remove-duplicates --handle-outliers
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
