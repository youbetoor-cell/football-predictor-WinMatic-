#!/usr/bin/env bash
set -e

# Activate virtualenv if running locally (Render will ignore this)
if [ -d "env" ] && [ -f "env/bin/activate" ]; then
  source env/bin/activate
fi

# Use PORT from environment if set, otherwise default to 8001
PORT="${PORT:-8001}"

# Start Uvicorn
uvicorn football_pred_service:app --host 0.0.0.0 --port "$PORT"
