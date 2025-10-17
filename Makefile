PYTHON ?= python3

.PHONY: help
help:
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

.PHONY: install
install: ## install the package and development dependencies
	@if command -v uv >/dev/null 2>&1; then \
		echo "Syncing environment with uv"; \
		uv sync --extra dev --extra docs; \
	else \
		echo "uv not available, falling back to pip"; \
		$(PYTHON) -m pip install --upgrade pip; \
		$(PYTHON) -m pip install -e .[dev]; \
	fi

.PHONY: lint
lint: ## ruff + black --check
	ruff check src tests tests/integration
	black --check src tests tests/integration

.PHONY: format
format: ## apply ruff & black formatting
	ruff check --fix src tests tests/integration
	black src tests tests/integration

.PHONY: unit
unit: ## run unit tests
	pytest -q

.PHONY: integration
integration: ## run integration tests across all workflows
	pytest tests/integration/ -v --tb=short

.PHONY: test
test: unit integration ## run all tests (unit + integration)

.PHONY: smoke
smoke: ## end-to-end toy run
	clinical-ml run --config configs/params.yaml

.PHONY: report
report: ## regenerate HTML report
	clinical-ml report --config configs/params.yaml --out results/report.html
