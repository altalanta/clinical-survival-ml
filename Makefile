PYTHON ?= python3

.PHONY: help
help:
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

.PHONY: setup
setup: ## create conda env, install pre-commit hooks
	@echo "Use: mamba env create -f env/environment.yml"
	@echo "Then: mamba activate clinical-survival-ml && pip install -e .[dev] && pre-commit install"

.PHONY: lint
lint: ## ruff + black --check
	ruff check src tests
	black --check src tests

.PHONY: format
format: ## apply ruff & black formatting
	ruff check --fix src tests
	black src tests

.PHONY: unit
unit: ## run unit tests
	pytest -q

.PHONY: smoke
smoke: ## end-to-end toy run
	clinical-ml run --config configs/params.yaml

.PHONY: report
report: ## regenerate HTML report
	clinical-ml report --config configs/params.yaml --out results/report.html
