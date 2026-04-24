#!/usr/bin/env bash
# Helper for installing optional ML dependencies.
# Usage:
#   scripts/install_whisper.sh            # installs faster-whisper
#   scripts/install_whisper.sh openai     # installs openai-whisper instead
#   scripts/install_whisper.sh berturk    # also installs transformers+torch

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/.venv/bin/activate"

mode="${1:-faster}"
case "$mode" in
  faster)
    pip install faster-whisper
    ;;
  openai)
    pip install openai-whisper
    ;;
  berturk)
    pip install transformers torch
    ;;
  all)
    pip install faster-whisper transformers torch
    ;;
  *)
    echo "Unknown mode '$mode'. Use: faster | openai | berturk | all"
    exit 1
    ;;
esac

echo "Done. Restart the server to pick up the new backend."
