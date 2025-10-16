# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.4.23

FROM python:${PYTHON_VERSION}-slim AS builder
ARG UV_VERSION
ENV UV_SYSTEM_PYTHON=1
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip uv==${UV_VERSION}
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs
COPY examples ./examples
RUN uv sync --frozen --extra dev --extra docs

FROM python:${PYTHON_VERSION}-slim AS runtime
ARG UV_VERSION
ENV UV_SYSTEM_PYTHON=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/clinical-ml/bin:$PATH"
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --system clinicalml
RUN pip install --upgrade pip uv==${UV_VERSION}
COPY --from=builder /workspace/.venv /opt/clinical-ml
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs
COPY examples ./examples
COPY scripts ./scripts
COPY configs/report_template.html.j2 configs/report_template.html.j2
ENV VIRTUAL_ENV="/opt/clinical-ml"
USER clinicalml
HEALTHCHECK --interval=1m --timeout=5s CMD clinical-ml --help >/dev/null 2>&1 || exit 1
ENTRYPOINT ["clinical-ml"]
CMD ["--help"]
