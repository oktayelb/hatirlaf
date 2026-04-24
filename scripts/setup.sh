#!/usr/bin/env bash
# One-shot local setup. Safe to re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"

if [ ! -d "$ROOT/.venv" ]; then
  echo "→ Creating virtualenv at $ROOT/.venv"
  python3 -m venv "$ROOT/.venv"
fi

# shellcheck source=/dev/null
source "$ROOT/.venv/bin/activate"

python -m pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Optional ML deps (install one of the Whisper backends for real transcription):"
echo "  pip install faster-whisper                # preferred, light CPU footprint"
echo "  pip install openai-whisper                # reference implementation"
echo "  pip install transformers torch            # enable BERTurk NER"
echo

python manage.py makemigrations diary
python manage.py migrate

echo
echo "Ready. Run: scripts/run.sh"
