# 🚀 Complete Quickstart Tutorial

This comprehensive guide will walk you through everything you need to know to get started with Clinical Survival ML, from installation to your first production deployment.

## 📋 Prerequisites

- **Python 3.10+** (3.11+ recommended)
- **4GB+ RAM** (8GB+ for large datasets)
- **Optional**: NVIDIA GPU for 5-10x faster training

## 🛠️ Installation

### Option 1: Automated Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/clinical-survival-ml.git
cd clinical-survival-ml

# Run the automated installer (handles everything)
make install
```

**What happens**:
- ✅ Installs all dependencies
- ✅ Sets up the package in development mode
- ✅ Configures pre-commit hooks
- ✅ Creates necessary directories

### Option 2: Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Install development dependencies (optional)
pip install -e .[dev]

# Install GPU support (optional)
pip install -e .[gpu]
```

### Option 3: Docker (For Production)

```bash
# Build the Docker image
make deploy-build

# Run training
make deploy-run

# Start the API server
make deploy-serve
```

## 🎯 Your First Analysis

Let's run a complete survival analysis on the included toy dataset:

```bash
# This runs the full pipeline: preprocessing → training → evaluation → reporting
clinical-ml run --config configs/params.yaml
```

**Expected output**:
```
🚀 Starting clinical survival analysis...
✅ Data loaded: 1000 patients, 10 features
🔧 Preprocessing: imputation, scaling, encoding...
🤖 Training models: coxph, rsf, xgb_cox, xgb_aft
📊 Cross-validation: concordance, IBS, Brier scores
📈 Generating calibration and decision curves
📋 Creating HTML report...
✅ Complete! Results saved to results/
```

**View your results**:
```bash
# Open the interactive HTML report
open results/report.html

# Or view metrics in JSON format
cat results/artifacts/metrics/cv_results.json
```

## 📊 Understanding Your Results

The toolkit generates comprehensive evaluation metrics:

### Model Performance (Concordance Index)
- **Cox PH**: 0.75 ± 0.03 (baseline statistical model)
- **Random Survival Forest**: 0.78 ± 0.02 (non-linear patterns)
- **XGBoost Cox**: 0.81 ± 0.02 (best performance)

### Calibration (How well predicted risks match actual outcomes)
- **Brier Score**: Lower is better (perfect = 0)
- **Calibration plots**: Show if model over/under-estimates risk

### Decision Curves
- Show net benefit of using the model vs. treating all/none
- Help determine optimal risk thresholds for clinical decisions

## 🔬 Deep Dive: Model Training

Let's examine what happens during training:

```bash
# 1. Load and validate configuration
clinical-ml validate-config --config configs/params.yaml

# 2. Load your data
clinical-ml load --data data/toy/toy_survival.csv --meta data/toy/metadata.yaml

# 3. Train models with custom grid search
clinical-ml train --config configs/params.yaml --grid configs/model_grid.yaml

# 4. Evaluate on holdout set
clinical-ml evaluate --config configs/params.yaml
```

## 🎛️ Configuration Files Explained

### `configs/params.yaml` - Main Configuration

```yaml
# Basic settings
seed: 42                    # For reproducibility
n_splits: 3                # Cross-validation folds
time_col: "time"           # Survival time column
event_col: "event"         # Event indicator column

# Model selection
models:
  - coxph                  # Cox proportional hazards
  - rsf                    # Random survival forest
  - xgb_cox                # XGBoost Cox model
  - xgb_aft                # XGBoost AFT model

# Data paths
paths:
  data_csv: "data/toy/toy_survival.csv"
  metadata: "data/toy/metadata.yaml"
  outdir: "results"

# Memory optimization (new!)
memory:
  max_memory_gb: 8         # Auto-detected if not set
  use_partitioning: true   # For large datasets
```

### `configs/features.yaml` - Feature Definitions

```yaml
# Define feature types for preprocessing
numeric_features:
  - age
  - sofa_score
  - creatinine

categorical_features:
  - sex
  - disease_stage

# Features to exclude
drop_features:
  - patient_id
```

## 🚀 Advanced Features

### GPU Acceleration

```bash
# Check if GPU acceleration is available
clinical-ml benchmark-hardware --config configs/params.yaml

# Train with GPU acceleration
clinical-ml run --config configs/params.yaml --use-gpu --gpu-id 0
```

**Performance gains**:
- **XGBoost**: 5-10x faster training
- **Memory management**: Automatic partitioning for large datasets
- **Random Survival Forest**: GPU acceleration via cuML

### Automated Model Selection (AutoML)

```bash
# Find the best model and hyperparameters automatically
clinical-ml automl \
  --config configs/params.yaml \
  --time-limit 1800 \
  --model-types coxph rsf xgb_cox xgb_aft \
  --metric concordance
```

### Counterfactual Explanations

```bash
# Generate "what-if" scenarios for clinical decision support
clinical-ml counterfactual \
  --model xgb_cox \
  --target-risk 0.3 \
  --n-counterfactuals 3

# Output: "To reduce risk from 0.65 to 0.30, decrease SOFA score by 2.1 points"
```

### Model Deployment & API

```bash
# 1. Register your trained model
clinical-ml register-model \
  --model results/artifacts/models/xgb_cox.pkl \
  --model-name survival_model \
  --version-number 1.0.0

# 2. Deploy as REST API
clinical-ml serve --models-dir results/artifacts/models

# 3. Make predictions
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": {"age": 65, "sex": "male", "sofa": 8}}'
```

## 🔧 Troubleshooting Common Issues

### Installation Problems

```bash
# If pip fails, try conda
mamba env create -f env/environment.yml
mamba activate clinical-survival-ml

# If you get dependency conflicts
pip install --upgrade pip setuptools wheel
pip install -e . --no-deps
pip install -r requirements.txt
```

### Memory Issues with Large Datasets

```bash
# Enable memory optimization
clinical-ml run --config configs/params.yaml --max-memory-gb 16

# Or use partitioning for very large datasets
clinical-ml run --config configs/params.yaml --use-partitioning --chunk-size 5000
```

### GPU Issues

```bash
# Check GPU availability
clinical-ml benchmark-hardware --config configs/params.yaml

# Common fixes:
# 1. Install GPU drivers
# 2. Update CUDA toolkit
# 3. Check GPU memory usage: nvidia-smi
```

## 📚 Learning Resources

### Documentation
- **[API Reference](https://clinical-survival-ml.readthedocs.io/)** - Complete function documentation
- **[Examples Gallery](examples/)** - Real-world use cases and code samples
- **[Research Papers](docs/papers/)** - Academic references and validation studies

### Community Support
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Q&A and community help
- **Stack Overflow**: Tag with `clinical-survival-ml`

### Next Steps
1. **Try the examples** in the `examples/` directory
2. **Read the research methods** in `docs/methods.md`
3. **Join our community** for support and updates

---

**🎉 Congratulations!** You now have everything you need to build, evaluate, and deploy clinical survival models. Start with the 5-minute quickstart and explore the advanced features as needed!


