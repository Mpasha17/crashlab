#!/usr/bin/env bash
# Flagship Day 11 demo: smolagents text_to_sql empty-:memory: bug.
# No API keys. Needs: smolagents + sqlalchemy in PYTHONPATH (or the real-agents venv).
set -euo pipefail
ROOT="${REAL_AGENTS:-/workspace/real-agents}"
VENV="${ROOT}/.venv/bin"
export PATH="${VENV}:${PATH}"
cd "${ROOT}/smolagents-sql"

echo "$ crashlab check 01M5H3SCC8   # live failing run (recorded earlier)"
echo "  fail  answer_is_correct  expected /Woodrow Wilson/, got empty-table excuse"
echo
echo "$ python repro.py              # no LLM; smolagents LocalPythonExecutor"
python repro.py 2>/dev/null | grep -E 'upstream|engine_fix|same thread|Woodrow|ERROR|no such'
echo
echo "$ # fix: StaticPool + check_same_thread=False on the example's engine"
echo "$ # upstream → no such table: receipts"
echo "$ # engine_fix → ('Woodrow Wilson',)"
