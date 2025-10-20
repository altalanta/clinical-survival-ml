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

.PHONY: test-quality
test-quality: ## run comprehensive quality assurance tests
	@echo "🔬 Generating synthetic test data..."
	clinical-ml synthetic-data --scenario icu --n-samples 1000 --random-state 42 --output-dir tests/data
	@echo "🧪 Running performance regression tests..."
	clinical-ml performance-regression --config configs/params.yaml --tolerance 0.02 --output-dir tests/performance_regression
	@echo "🔍 Checking CV integrity..."
	clinical-ml cv-integrity --config configs/params.yaml --cv-folds 3 --output-dir tests/cv_integrity
	@echo "🏆 Running benchmark suite..."
	clinical-ml benchmark-suite --config configs/params.yaml --output-dir tests/benchmark_results
	@echo "✅ Quality assurance tests completed!"

.PHONY: test-synthetic
test-synthetic: ## generate synthetic datasets for testing
	clinical-ml synthetic-data --scenario icu --n-samples 500 --random-state 42
	clinical-ml synthetic-data --scenario cancer --n-samples 500 --random-state 42
	clinical-ml synthetic-data --scenario cardiovascular --n-samples 500 --random-state 42

.PHONY: test-regression
test-regression: ## run performance regression tests
	clinical-ml performance-regression --config configs/params.yaml --tolerance 0.05

.PHONY: test-cv-integrity
test-cv-integrity: ## check cross-validation integrity
	clinical-ml cv-integrity --config configs/params.yaml --cv-folds 5

.PHONY: test-benchmark
test-benchmark: ## run benchmark suite
	clinical-ml benchmark-suite --config configs/params.yaml
