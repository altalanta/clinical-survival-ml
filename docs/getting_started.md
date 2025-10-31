# Getting Started

This guide will walk you through the installation of `clinical-survival-ml` and get you started with your first analysis.

## Installation

### Poetry (Recommended)

```bash
# 1. Install poetry (if you don't have it)
# https://python-poetry.org/docs/#installation
curl -sSL https://install.python-poetry.org | python3 -

# 2. Clone the repository
git clone https://github.com/artemisfolle/clinical-survival-ml.git
cd clinical-survival-ml

# 3. Install dependencies
poetry install --all-extras

# 4. Activate the virtual environment
poetry shell
```

This method uses `poetry` to create a reproducible environment.

### GPU Acceleration (Optional)

```bash
# Install with GPU support for 5-10x faster training
poetry install --extras "gpu"
```

### Docker

#### Quick Start with Docker Compose (Recommended)

```bash
# Build and start the API server
docker-compose up --build
```

#### Manual Docker Commands

```bash
# Build the image
docker build -t clinical-survival-ml .

# Run training
docker run --rm -v $(pwd):/workspace clinical-survival-ml training run \
  --config-path configs/params.yaml --grid-path configs/model_grid.yaml
```

## 5-Minute Quickstart

Get started with clinical survival modeling in 5 minutes!

```bash
# 1. (If you haven't already) Clone and install
git clone https://github.com/artemisfolle/clinical-survival-ml.git
cd clinical-survival-ml
poetry install --all-extras

# 2. Run the toy example
poetry run clinical-ml training run

# 3. View results
open results/report.html
```

That's it! You now have a complete survival analysis with models, evaluation metrics, and an interactive report.
