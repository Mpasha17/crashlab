#!/usr/bin/env bash
# Local pre-push check: ruff + offline tests + secret-scan.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${ROOT}/.venv/bin:${PATH}"
echo "== ruff =="
ruff check .
ruff format --check .
echo "== pytest (offline) =="
env -u CRASHLAB_DIR -u GROQ_API_KEY -u GEMINI_API_KEY -u OPENAI_API_KEY \
  pytest -q -o addopts='' -p no:cacheprovider
if [[ -x /workspace/crashlab-live/scan.sh ]]; then
  echo "== secret-scan =="
  /workspace/crashlab-live/scan.sh
elif [[ -x "$ROOT/scripts/scan.sh" ]]; then
  "$ROOT/scripts/scan.sh"
else
  echo "(no scan.sh; grepping common key prefixes)"
  ! git grep -nE 'AQ\.Ab8|gsk_[A-Za-z0-9]{20,}' -- ':!docs/demo/*.gif' || exit 1
fi
echo "OK"
