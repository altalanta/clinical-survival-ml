# Clinical Survival ML

Reproducible end-to-end survival modelling for tabular clinical outcomes. The toolkit ingests time-to-event data, preprocesses covariates, fits Cox PH, Random Survival Forests, and XGBoost-based survival models, evaluates discrimination and calibration, performs decision-curve analysis, and renders a publication-ready HTML report complete with SHAP- and PDP-based interpretability.

## Quickstart

### Conda / mamba

```bash
mamba env create -f env/environment.yml
mamba activate clinical-survival-ml
pip install -e .[dev]
pre-commit install
```

Then run the toy workflow:

```bash
clinical-ml run --config configs/params.yaml --grid configs/model_grid.yaml
```

### Docker

```bash
docker build -t clinical-survival-ml .
docker run --rm -v $(pwd):/workspace clinical-survival-ml run \
  --config configs/params.yaml --grid configs/model_grid.yaml
```

## Configuring Experiments

- `configs/params.yaml` sets seeds, CV folds, evaluation time grid, missing-data strategy, and I/O paths.
- `configs/model_grid.yaml` defines per-model hyperparameter grids.
- `configs/features.yaml` teams numeric/categorical features and optional drop columns.

Modify the dataset paths in `params.yaml` to target your clinical dataset (CSV) and accompanying `metadata.yaml` describing column types.

## CLI Commands

```
clinical-ml load --data data/toy/toy_survival.csv --meta data/toy/metadata.yaml
clinical-ml train --config configs/params.yaml --grid configs/model_grid.yaml
clinical-ml evaluate --config configs/params.yaml
clinical-ml explain --config configs/params.yaml --model xgb_cox
clinical-ml report --config configs/params.yaml --out results/report.html
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
- **Tuning** – Nested cross-validation with stratified outer/inner folds, scored by concordance and IBS.
- **Evaluation** – Out-of-fold (OOF) Harrell's C-index, time-dependent Brier scores, integrated Brier score, IPCW calibration curves, and censoring-aware decision-curve net benefit.
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
