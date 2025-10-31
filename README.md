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

## Quickstart

```bash
# Install dependencies
poetry install --all-extras

# Run the toy example
poetry run clinical-ml run --config configs/params.yaml

# View the results
open results/report.html
```

## Contributing

Contributions are welcome! Please see the `CONTRIBUTING.md` file for details.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
