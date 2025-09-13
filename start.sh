#!/bin/sh
# start.sh — self-contained bootstrap for Railway Shell runtime
set -eu

SERVICE="${SERVICE:-backend}"
PORT="${PORT:-8080}"

# --- helpers ---------------------------------------------------------------
need() {
  command -v "$1" >/dev/null 2>&1
}

ensure_uv() {
  if need uv; then
    return
  fi

  # We need curl or wget to fetch the installer
  if need curl; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif need wget; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    echo "Neither curl nor wget is available to install uv/Python." >&2
    exit 1
  fi

  # uv installs to ~/.local/bin
  export PATH="$HOME/.local/bin:$PATH"

  if ! need uv; then
    echo "uv did not install correctly (PATH=$PATH)" >&2
    exit 1
  fi
}

run_backend() {
  cd backend || { echo "backend directory not found"; exit 1; }

  # 1) ensure uv, then install Python
  ensure_uv
  # install a Python runtime (cached in $HOME by uv)
  uv python install 3.11

  # 2) create and activate a venv
  uv venv .venv
  # shellcheck disable=SC1091
  . .venv/bin/activate

  # 3) install deps
  if [ -f requirements.txt ]; then
    uv pip install --no-cache-dir -r requirements.txt
  fi

  # 4) run the app
  exec uv run uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
}

run_frontend() {
  cd frontend || { echo "frontend directory not found"; exit 1; }

  if ! need npm; then
    echo "Node/npm not available in this runtime. The Shell image doesn't include Node." >&2
    exit 1
  fi

  if [ -f package-lock.json ]; then npm ci; else npm install; fi
  npm run build
  npx -y serve -s dist -l "${PORT:-3000}"
}

# --- entrypoint ------------------------------------------------------------
case "$SERVICE" in
  backend)  run_backend ;;
  frontend) run_frontend ;;
  *) echo "Unknown SERVICE=$SERVICE (use 'backend' or 'frontend')" >&2; exit 1 ;;
esac
