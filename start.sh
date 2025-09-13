#!/bin/sh
# POSIX-safe start script for Railway
set -eu  # no pipefail in /bin/sh

echo "Using shell: $(ps -p $$ -o comm=)"
echo "SERVICE=${SERVICE:-backend}  PORT=${PORT:-8000}"

SERVICE="${SERVICE:-backend}"
PORT="${PORT:-8000}"

case "$SERVICE" in
  backend)
    cd backend || { echo "backend dir missing"; exit 1; }
    if [ -f requirements.txt ]; then
      pip install -r requirements.txt
    fi
    exec python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
    ;;

  frontend)
    cd frontend || { echo "frontend dir missing"; exit 1; }
    if command -v npm >/dev/null 2>&1; then
      if [ -f package-lock.json ]; then npm ci; else npm install; fi
      npm run build
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
