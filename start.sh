#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"

# Prefer a single worker so in-memory TTL caches actually work.
# If WEB_CONCURRENCY is set but empty/non-numeric, force it to 1.
WC="${WEB_CONCURRENCY:-1}"
if [[ -z "${WC}" || ! "${WC}" =~ ^[0-9]+$ || "${WC}" -lt 1 ]]; then
  WC="1"
fi
export WEB_CONCURRENCY="$WC"

# Always prefer local venv gunicorn
if [ -x ".venv/bin/gunicorn" ]; then
  GUNICORN="./.venv/bin/gunicorn"
elif [ -x "env/bin/gunicorn" ]; then
  GUNICORN="./env/bin/gunicorn"
else
  echo "ERROR: gunicorn not found. Install it in .venv or env." >&2
  exit 1
fi

echo "[start.sh] PORT=$PORT WEB_CONCURRENCY=$WEB_CONCURRENCY" >&2

exec "$GUNICORN" \
  -k uvicorn.workers.UvicornWorker \
  -w "$WEB_CONCURRENCY" \
  -b "0.0.0.0:${PORT}" \
  --timeout 120 \
  --graceful-timeout 30 \
  --keep-alive 5 \
  football_pred_service:app
