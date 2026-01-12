#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-10000}"

# WEB_CONCURRENCY can be unset or empty on Render; default to 1 safely
W="${WEB_CONCURRENCY:-1}"
if ! [[ "$W" =~ ^[0-9]+$ ]]; then
  W=1
fi
if [[ "$W" -lt 1 ]]; then
  W=1
fi

echo "Starting gunicorn on 0.0.0.0:${PORT} with workers=${W}"
exec gunicorn \
  -k uvicorn.workers.UvicornWorker \
  -w "$W" \
  -b "0.0.0.0:${PORT}" \
  --timeout 120 \
  --graceful-timeout 30 \
  --access-logfile - \
  --error-logfile - \
  football_pred_service:app
