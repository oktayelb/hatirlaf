#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"

# shellcheck source=/dev/null
source "$ROOT/.venv/bin/activate"

HOST="${HATIRLAF_HOST:-127.0.0.1}"
PORT="${HATIRLAF_PORT:-8000}"

echo "→ Starting Hatırlaf on http://$HOST:$PORT"
echo "  UI:       http://$HOST:$PORT/"
echo "  API:      http://$HOST:$PORT/api/"
echo "  Admin:    http://$HOST:$PORT/admin/"
echo

exec python manage.py runserver "$HOST:$PORT"
