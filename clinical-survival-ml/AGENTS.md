# Repository Guidelines

## Project Structure & Module Organization
- `src/clinical_survival/`: survival pipeline modules (data loading, preprocessing, model wrappers, tuning, evaluation, explainability, CLI, reporting).
- `configs/`: YAML configuration files (parameters, feature schema, model grids, report template).
- `tests/`: pytest suite, including smoke tests (`test_cli.py`) and external validation checks (`test_external.py`).
- `data/toy/`: synthetic survival dataset and metadata used for CI and smoke runs.
- `artifacts/` (generated): models, metrics, explainability outputs under `results/artifacts/` after running the CLI.

## Build, Test, and Development Commands
```bash
make lint        # ruff + black checks
make unit        # run pytest suite
make smoke       # end-to-end pipeline on toy dataset
clinical-ml run --config configs/params.yaml --grid configs/model_grid.yaml
```
Use `mamba env create -f env/environment.yml` or `pip install -e .[dev]` for local setup.

## Coding Style & Naming Conventions
- Python 3.10+, PEP 8 compliant; 4-space indentation.
- Lint/format via `ruff` and `black` (config in `.pre-commit-config.yaml`, `pyproject.toml`).
- Module names are snake_case; classes in PascalCase; functions and variables in snake_case.

## Testing Guidelines
- Pytest is the primary framework; tests live under `tests/` with `test_*.py` naming.
- Ensure new features include unit coverage and, when relevant, CLI smoke assertions.
- Synthetic data fixtures in `tests/data/` should stay <200 KB for CI friendliness.

## Commit & Pull Request Guidelines
- Prefer conventional commit summaries (`feat:`, `fix:`, `docs:`) when practical.
- PRs should describe scope, reference issues, and note new configs/env changes.
- Include test evidence (`make unit`, `make smoke`) and screenshots for report/UI changes when applicable.

## Security & Configuration Tips
- Secrets and PHI must never enter the repo; use environment variables for protected endpoints.
- External validation can be supplied via `paths.external_csv` or `external.group_column`; verify metadata schemas before sharing datasets.
