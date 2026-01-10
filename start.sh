#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"

# WEB_CONCURRENCY may be unset or set to an empty string on Render.
# Force it to a valid integer >=1.
RAW_WEB_CONCURRENCY="${WEB_CONCURRENCY-}"
if [[ -z "${RAW_WEB_CONCURRENCY}" || ! "${RAW_WEB_CONCURRENCY}" =~ ^[0-9]+$ ]]; then
  WEB_CONCURRENCY="1"
else
  WEB_CONCURRENCY="${RAW_WEB_CONCURRENCY}"
fi

# Prefer local venv gunicorn if present, else fall back to PATH
if [ -x ".venv/bin/gunicorn" ]; then
  GUNICORN="./.venv/bin/gunicorn"
elif [ -x "env/bin/gunicorn" ]; then
  GUNICORN="./env/bin/gunicorn"
else
  GUNICORN="gunicorn"
fi

exec "$GUNICORN" \
  -k uvicorn.workers.UvicornWorker \
  -w "$WEB_CONCURRENCY" \
  -b "0.0.0.0:${PORT}" \
  --timeout 120 \
  --graceful-timeout 30 \
  --keep-alive 5 \
  football_pred_service:app
