# User Guide

This guide provides a detailed overview of how to use the `clinical-survival-ml` toolkit, from configuring experiments to deploying models.

## Configuring Experiments

- `configs/params.yaml` sets seeds, CV folds, evaluation time grid, missing-data strategy, and I/O paths.
- `configs/model_grid.yaml` defines per-model hyperparameter grids.
- `configs/features.yaml` defines numeric/categorical features and optional drop columns.

Before running experiments, validate your configuration files:

```bash
poetry run clinical-ml validate-config
```

## Running the Pipeline

The main entry point for running a complete analysis is the `run` command:

```bash
poetry run clinical-ml training run \
  --config-path configs/params.yaml \
  --grid-path configs/model_grid.yaml
```

This command will:
1.  Load and preprocess the data.
2.  Train all specified models using cross-validation.
3.  Evaluate the models on a hold-out set.
4.  Generate a comprehensive HTML report with all results.

## Advanced Features

This project includes a wide range of advanced features for production-grade survival modeling.

### GPU Acceleration & Memory Optimization

```bash
# Check hardware capabilities
poetry run clinical-ml benchmark-hardware --config configs/params.yaml
```

### Counterfactual Explanations

```bash
# Generate "what-if" scenarios for clinical decision support
poetry run clinical-ml counterfactual --model xgb_cox --target-risk 0.3
```

### Model Monitoring & Drift Detection

```bash
# Monitor model performance over time
poetry run clinical-ml monitor --config configs/params.yaml --days 30
```

*For more details on all available commands and their options, please refer to the API Reference section.*
