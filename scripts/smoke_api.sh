#!/usr/bin/env bash
set -euo pipefail

API_ROOT=${API_ROOT:-http://127.0.0.1:8000}

curl -fsS "${API_ROOT}/health" >/tmp/clinical_ml_health.json
python - <<'PY'
import json
import sys

with open('/tmp/clinical_ml_health.json', 'r', encoding='utf-8') as handle:
    payload = json.load(handle)

if payload.get('status') != 'ok':
    sys.exit('health endpoint did not return status=ok')
PY

curl -fsS "${API_ROOT}/models" >/tmp/clinical_ml_models.json
python - <<'PY'
import json
import sys

with open('/tmp/clinical_ml_models.json', 'r', encoding='utf-8') as handle:
    models = json.load(handle)

if not isinstance(models, list) or not models:
    sys.exit('models endpoint did not return any models')
PY

cat <<'PAYLOAD' > /tmp/clinical_ml_payload.json
{
  "features": {
    "age": 65,
    "sex": "male",
    "sofa": 8,
    "lactate": 2.1,
    "creatinine": 1.0,
    "heart_rate": 92,
    "stage": "III",
    "icu_type": "medical",
    "treatment": "standard"
  },
  "time_horizons": [90, 180, 365]
}
PAYLOAD

curl -fsS -X POST "${API_ROOT}/predict" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/clinical_ml_payload.json >/tmp/clinical_ml_predict.json

python - <<'PY'
import json
import sys

with open('/tmp/clinical_ml_predict.json', 'r', encoding='utf-8') as handle:
    payload = json.load(handle)

if 'survival' not in payload:
    sys.exit('predict endpoint did not return survival probabilities')
PY
