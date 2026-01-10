#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"

# IMPORTANT:
# Render/Gunicorn may have WEB_CONCURRENCY set but empty -> gunicorn crashes (int("") ValueError).
# Normalize it here to a valid integer and export it so gunicorn sees it.
WM_WEB_CONCURRENCY=""
export WEB_CONCURRENCY

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
  -w "$WM_WEB_CONCURRENCY" \
  -b "0.0.0.0:${PORT}" \
  --timeout 120 \
  --graceful-timeout 30 \
  --keep-alive 5 \
  football_pred_service:app
