#!/usr/bin/env bash
# One-shot local setup. Safe to re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"

MINIMAL_SETUP="${HATIRLAF_SETUP_MINIMAL:-0}"

if [ ! -d "$ROOT/.venv" ]; then
  echo "→ Creating virtualenv at $ROOT/.venv"
  python3 -m venv "$ROOT/.venv"
fi

# shellcheck source=/dev/null
source "$ROOT/.venv/bin/activate"

python -m pip install --upgrade pip
pip install -r requirements.txt

if [ "$MINIMAL_SETUP" != "1" ]; then
  echo
  echo "→ Installing local ML stack (STT, Turkish NER, LLM, SAVYAR bridge)"
  # Install torch first so the downstream packages can resolve against it.
  if ! python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; then
    echo "  ! PyTorch CPU wheels were not available; falling back to default PyPI"
    python -m pip install torch torchvision torchaudio
  fi
  python -m pip install faster-whisper openai-whisper transformers llama-cpp-python
else
  echo
  echo "→ Skipping optional ML stack because HATIRLAF_SETUP_MINIMAL=1"
  echo "  You can install it later with: scripts/install_whisper.sh all"
fi

python manage.py makemigrations diary
python manage.py migrate

echo
echo "Ready. Run: scripts/run.sh"
