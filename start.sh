#!/bin/sh
set -eu

SERVICE="${SERVICE:-backend}"
PORT="${PORT:-8080}"

echo "### NEMHEEN SCRIPT ACTIVE: $(date) ###"

if [ "$SERVICE" = "backend" ]; then
  cd backend || { echo "ERROR: backend directory not found"; exit 1; }

  echo "Installing Python 3.11 with uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"

  uv python install 3.11
  uv venv .venv
  . .venv/bin/activate

  [ -f requirements.txt ] && uv pip install -r requirements.txt

  echo "Starting uvicorn…"
  exec uv run uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
else
  echo "ERROR: Unknown SERVICE=$SERVICE"
  exit 1
fi
