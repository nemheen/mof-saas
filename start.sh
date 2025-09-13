#!/bin/sh
# start.sh — POSIX bootstrap for Railway Shell runtime (no pip/ps)
set -eu
echo "### NEMHEEN SCRIPT ACTIVE: $(date) ###"
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

  ensure_uv
  uv python install 3.11
  uv venv .venv
  . .venv/bin/activate

  if [ -f requirements.txt ]; then
    echo "Installing backend requirements via uv…"
    uv pip install --no-cache-dir -r requirements.txt
  else
    echo "WARNING: requirements.txt not found; continuing"
  fi

  echo "Starting uvicorn…"
  exec uv run uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
}

case "$SERVICE" in
  backend)  run_backend ;;
  frontend) echo "Frontend requires Node in this runtime"; exit 1 ;;
  *) echo "ERROR: Unknown SERVICE=$SERVICE"; exit 1 ;;
esac
