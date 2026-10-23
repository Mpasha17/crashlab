#!/usr/bin/env bash
# Offline README quickstart path — no API keys. Run from a venv with crashlab installed.
set -euo pipefail
export CRASHLAB_DIR="${CRASHLAB_DIR:-$(mktemp -d /tmp/crashlab-demo.XXXX)}"
echo "CRASHLAB_DIR=$CRASHLAB_DIR"
python - <<'PY'
import crashlab
from openai import OpenAI
# use the built-in refund_bot cassette path if present, else a minimal record
print("crashlab", crashlab.__version__ if hasattr(crashlab,"__version__") else "ok")
print("commands: crashlab ls | show | check | replay | diagnose | fix | verify | export-test | report --html")
print("see README Quickstart for the no-key path using examples/refund_bot")
PY
crashlab --help | head -20
