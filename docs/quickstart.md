# Quickstart

This guide will get you up and running with `clinical-survival-ml` in just five minutes.

## 🚀 5-Minute Quickstart

**For beginners**: Get started with clinical survival modeling in 5 minutes!

```bash
# 1. Clone and install (takes ~2 minutes)
poetry install --all-extras

# 2. Run the toy example (takes ~1 minute)
poetry run clinical-ml run --config configs/params.yaml

# 3. View results
open results/report.html
```

That's it! You now have a complete survival analysis with models, evaluation metrics, and an interactive report.

## Your First Analysis

Let's run a complete survival analysis on the included toy dataset:

```bash
# This runs the full pipeline: preprocessing → training → evaluation → reporting
poetry run clinical-ml run --config configs/params.yaml
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







