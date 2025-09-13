#!/usr/bin/env bash
set -euo pipefail

# Detect which service this instance is supposed to run
SERVICE="${SERVICE:-backend}"

if [ "$SERVICE" = "backend" ]; then
  cd backend
  pip install -r requirements.txt
  exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
elif [ "$SERVICE" = "frontend" ]; then
  cd frontend
  npm ci
  npm run build
  npx serve -s dist -l "${PORT:-3000}"
else
  echo "Unknown SERVICE=$SERVICE"
  exit 1
fi
