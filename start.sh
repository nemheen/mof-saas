#!/bin/sh
# POSIX-safe start script for Railway
set -eu  # no pipefail in /bin/sh

SERVICE="${SERVICE:-backend}"
PORT="${PORT:-8000}"

case "$SERVICE" in
  backend)
    cd backend
    if [ -f requirements.txt ]; then
      pip install -r requirements.txt
    fi
    # call uvicorn via python for portability
    exec python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
    ;;

  frontend)
    cd frontend
    if command -v npm >/dev/null 2>&1; then
      # use ci if lockfile exists, otherwise install
      if [ -f package-lock.json ]; then npm ci; else npm install; fi
      npm run build
      # serve static build; -y auto-installs "serve" if missing
      npx -y serve -s dist -l "${PORT:-3000}"
    else
      echo "Node/npm not available" >&2
      exit 1
    fi
    ;;

  *)
    echo "Unknown SERVICE=$SERVICE" >&2
    exit 1
    ;;
esac
