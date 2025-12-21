#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
WEB_CONCURRENCY="${WEB_CONCURRENCY:-2}"

# Always prefer local venv gunicorn
if [ -x ".venv/bin/gunicorn" ]; then
  GUNICORN="./.venv/bin/gunicorn"
elif [ -x "env/bin/gunicorn" ]; then
  GUNICORN="./env/bin/gunicorn"
else
  echo "ERROR: gunicorn not found. Install it in .venv or env." >&2
  exit 1
fi

exec "$GUNICORN" \
  -k uvicorn.workers.UvicornWorker \
  -w "$WEB_CONCURRENCY" \
  -b "0.0.0.0:${PORT}" \
  --timeout 120 \
  --graceful-timeout 30 \
  --keep-alive 5 \
  football_pred_service:app
