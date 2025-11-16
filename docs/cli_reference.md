# CLI Reference

This page provides a complete reference for all CLI commands with practical examples.

## 🚀 Core Workflow Commands

| Command | Description | Example |
|---|---|---|
| **`poetry run clinical-ml run`** | Complete pipeline: train → evaluate → report | `poetry run clinical-ml run --config configs/params.yaml` |
| **`poetry run clinical-ml train`** | Train models only | `poetry run clinical-ml train --config configs/params.yaml --grid configs/model_grid.yaml` |
| **`poetry run clinical-ml evaluate`** | Evaluate trained models | `poetry run clinical-ml evaluate --config configs/params.yaml` |
| **`poetry run clinical-ml report`** | Generate HTML report | `poetry run clinical-ml report --config configs/params.yaml --out results/report.html` |

## 🤖 Model Training & Selection

| Command | Description | Example |
|---|---|---|
| **`poetry run clinical-ml automl`** | Automated model selection | `poetry run clinical-ml automl --config configs/params.yaml --time-limit 1800` |
| **`poetry run clinical-ml benchmark-hardware`** | Check GPU/CPU performance | `poetry run clinical-ml benchmark-hardware --config configs/params.yaml` |
| **`poetry run clinical-ml counterfactual`** | Generate explanations | `poetry run clinical-ml counterfactual --model xgb_cox --target-risk 0.3` |

... and so on for all commands.







