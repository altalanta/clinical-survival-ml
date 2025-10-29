# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.10

FROM python:${PYTHON_VERSION}-slim AS builder

# Install poetry
RUN pip install poetry

# Copy project files
COPY pyproject.toml poetry.lock README.md ./
COPY src ./src

# Install dependencies
RUN poetry install --no-root --all-extras

FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace

# Create a non-root user
RUN useradd --create-home --system appuser
USER appuser

# Copy virtual environment and source code
COPY --from=builder /root/.cache/pypoetry/virtualenvs /opt/venv
COPY . .

# Set entrypoint
ENTRYPOINT ["poetry", "run", "clinical-ml"]
CMD ["--help"]
