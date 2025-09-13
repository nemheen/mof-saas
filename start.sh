#!/bin/sh
# start.sh — POSIX bootstrap for Railway Shell runtime (no pip, no ps)
set -eu

echo "=== BOOT: start.sh (uv bootstrap) v1 ==="

SERVICE="${SERVICE:-backend}"
PORT="${PORT:-8080}"

need() { command -v "$1" >/dev/null 2>&1; }

ensure_uv() {
  if need uv; then
    echo "uv found: $(uv --version || echo unknown)"
    return
  fi
  echo "uv not found; installing…"

  if need curl; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif need wget; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    echo "ERROR: Neither curl nor wget is available to install uv/Python." >&2
    exit 1
  fi

  # uv installs into ~/.local/bin
  export PATH="$HOME/.local/bin:$PATH"
  if ! need uv; then
    echo "ERROR: uv failed to install (PATH=$PATH)" >&2
    exit 1
  fi
  echo "uv installed: $(uv --version || echo unknown)"
}

run_backend() {
  echo "SERVICE=backend  PORT=$PORT"
  cd backend || { echo "ERROR: backend directory not found"; exit 1; }

  # 1) ensure uv + Python
  ensure_uv
  uv python install 3.11

  # 2) create venv (idempotent) and activate
  uv venv .venv
  . .venv/bin/activate

  # 3) install deps without using pip directly
  if [ -f requirements.txt ]; then
    echo "Installing backend requirements via uv…"
    uv pip install --no-cache-dir -r requirements.txt
  else
    echo "WARNING: requirements.txt not found; continuing"
  fi

  # 4) run app (no pip)
  echo "Starting uvicorn…"
  exec uv run uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
}

run_frontend() {
  echo "SERVICE=frontend  PORT=$PORT"
  cd frontend || { echo "ERROR: frontend directory not found"; exit 1; }
  if ! need npm; then
    echo "ERROR: Node/npm not available in this runtime." >&2
    exit 1
  fi
  if [ -f package-lock.json ]; then npm ci; else npm install; fi
  npm run build
  exec npx -y serve -s dist -l "${PORT:-3000}"
}

case "$SERVICE" in
  backend)  run_backend ;;
  frontend) run_frontend ;;
  *) echo "ERROR: Unknown SERVICE=$SERVICE (use 'backend' or 'frontend')" >&2; exit 1 ;;
esac
