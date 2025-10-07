PYTHON ?= python3

.PHONY: help
help:
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

.PHONY: install
install: ## install the package using the automated installer
	./install.sh

.PHONY: setup
setup: ## create conda env, install pre-commit hooks
	@echo "Setting up clinical-survival-ml..."
	@echo "Option 1 - Using the automated installer:"
	@echo "  make install"
	@echo ""
	@echo "Option 2 - Manual installation:"
	@echo "  mamba env create -f env/environment.yml"
	@echo "  mamba activate clinical-survival-ml"
	@echo "  make install"
	@echo ""
	@echo "Then install pre-commit hooks:"
	@echo "  pre-commit install"

.PHONY: lint
lint: ## ruff + black --check
	ruff check src tests
	black --check src tests

.PHONY: format
format: ## apply ruff & black formatting
	ruff check --fix src tests
	black src tests

.PHONY: validate
validate: ## validate configuration files
	clinical-ml validate-config

.PHONY: unit
unit: ## run unit tests
	pytest -q

.PHONY: smoke
smoke: ## end-to-end toy run
	clinical-ml validate-config
	clinical-ml run --config configs/params.yaml

.PHONY: report
report: ## regenerate HTML report
	clinical-ml report --config configs/params.yaml --out results/report.html

.PHONY: deploy-help
deploy-help: ## show deployment options
	./deploy.sh help

.PHONY: deploy-build
deploy-build: ## build Docker image for deployment
	./deploy.sh build

.PHONY: deploy-serve
deploy-serve: ## start API server (requires trained models)
	./deploy.sh serve

.PHONY: deploy-train
deploy-train: ## run training in Docker
	./deploy.sh train
