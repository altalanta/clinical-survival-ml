#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

log() {
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"
}

log "Validating configuration"
clinical-ml validate-config \
  --config configs/params.yaml \
  --grid configs/model_grid.yaml \
  --features configs/features.yaml

log "Running end-to-end training"
clinical-ml run --config configs/params.yaml --grid configs/model_grid.yaml

REPORT_PATH=${REPORT_PATH:-results/report.html}
log "Evaluating and building report at ${REPORT_PATH}"
clinical-ml evaluate --config configs/params.yaml --report "$REPORT_PATH"

log "Starting API server"
clinical-ml serve --models-dir results/artifacts/models --host 127.0.0.1 --port 8000 &
SERVER_PID=$!
trap 'log "Stopping API server"; kill ${SERVER_PID} 2>/dev/null || true' EXIT

log "Waiting for API health endpoint"
for attempt in {1..10}; do
  if curl -fsS "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
  if [[ $attempt -eq 10 ]]; then
    log "Server did not become healthy"
    exit 1
  fi
done

log "Running smoke API checks"
./scripts/smoke_api.sh

log "Toy workflow completed"
