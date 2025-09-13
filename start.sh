#!/usr/bin/env bash
set -euo pipefail

SERVICE="${SERVICE:-backend}"
PORT="${PORT:-8000}"

if [[ "$SERVICE" == "backend" ]]; then
  cd backend
  if [[ -f requirements.txt ]]; then pip install -r requirements.txt; fi
  exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
elif [[ "$SERVICE" == "frontend" ]]; then
  cd frontend
  if command -v npm >/dev/null 2>&1; then
    npm ci
    npm run build
    npx serve -s dist -l "${PORT:-3000}"
  else
    echo "Node/npm not available"; exit 1
  fi
else
  echo "Unknown SERVICE=$SERVICE"; exit 1
fi
