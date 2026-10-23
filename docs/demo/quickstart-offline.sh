#!/usr/bin/env bash
# Offline README quickstart path — no API keys. Run from repo root in a venv with
# crashlab installed (`pip install -e ".[openai]"`).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export CRASHLAB_DIR="${CRASHLAB_DIR:-$(mktemp -d /tmp/crashlab-demo.XXXX)}"
echo "CRASHLAB_DIR=$CRASHLAB_DIR"
python3 - <<'PY'
import crashlab
print("crashlab", getattr(crashlab, "__version__", "ok"))
PY
shopt -s nullglob
demos=(examples/refund_bot/demo/*.jsonl)
if ((${#demos[@]})); then
  for f in "${demos[@]}"; do
    crashlab import "$f"
  done
  crashlab ls
  echo "next: crashlab show <id> · crashlab check --all · crashlab replay <id>"
else
  echo "no examples/refund_bot/demo/*.jsonl found; see README Quickstart"
  crashlab --help | head -20
fi
