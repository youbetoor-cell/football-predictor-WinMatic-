#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-10000}"

W="${WEB_CONCURRENCY:-1}"
if ! [[ "$W" =~ ^[0-9]+$ ]]; then W=1; fi
if [[ "$W" -lt 1 ]]; then W=1; fi

echo "Starting on 0.0.0.0:${PORT} with workers=${W}"
exec gunicorn -k uvicorn.workers.UvicornWorker \
  -w "$W" \
  -b "0.0.0.0:${PORT}" \
  --timeout 120 \
  --graceful-timeout 30 \
  --access-logfile - \
  --error-logfile - \
  football_pred_service:app
