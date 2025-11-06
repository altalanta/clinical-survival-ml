# Installation

This page covers the different ways you can install `clinical-survival-ml`.

## Poetry (Recommended)

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

## GPU Acceleration (Optional)

```bash
# Install with GPU support for 5-10x faster training
poetry install --extras "gpu"
```

## Docker

### Quick Start with Docker Compose (Recommended)

```bash
# Build and start the API server
docker-compose up --build
```

### Manual Docker Commands

```bash
# Build the image
docker build -t clinical-survival-ml .

# Run training
docker run --rm -v $(pwd):/workspace clinical-survival-ml run \
  --config configs/params.yaml --grid configs/model_grid.yaml

# Start API server (after training)
docker run -p 8000:8000 -v $(pwd)/results:/workspace/results \
  clinical-survival-ml serve --models-dir results/artifacts/models
```

## Troubleshooting Installation

If you encounter issues, please refer to the Poetry documentation for resolving dependency conflicts.
Then run the toy workflow:

```bash
poetry run clinical-ml run --config configs/params.yaml --grid configs/model_grid.yaml
```



