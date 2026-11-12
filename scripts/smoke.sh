#!/usr/bin/env bash
# End-to-end CLI smoke test: every crashlab command against the installed `crashlab`,
# offline (a local fake OpenAI server, no keys, no network). Needs the openai SDK:
#   pip install -e ".[dev,openai]" && scripts/smoke.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
TMP="$(mktemp -d)"
unset GROQ_API_KEY GEMINI_API_KEY ANTHROPIC_API_KEY PROVIDER
export CRASHLAB_DIR="$TMP/store" CRASHLAB_NO_PYPROJECT=1 COLUMNS=160 NO_COLOR=1
export OPENAI_API_KEY=smoke MODEL=gpt-4o-mini LITELLM_LOCAL_MODEL_COST_MAP=True

"$PY" scripts/smoke_fake_llm.py "$TMP/port" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true; rm -rf "$TMP"' EXIT
for _ in $(seq 50); do [[ -s "$TMP/port" ]] && break; sleep 0.1; done
export OPENAI_BASE_URL="http://127.0.0.1:$(cat "$TMP/port")/v1"

n=0
# step EXPECTED_EXIT GREP_PATTERN -- crashlab args...
step() {
  local want="$1" pat="$2"; shift 3
  n=$((n + 1))
  local got=0
  crashlab "$@" >"$TMP/out" 2>&1 || got=$?
  if [[ "$got" != "$want" ]] || ! grep -qE -- "$pat" "$TMP/out"; then
    cat "$TMP/out"
    echo "FAIL: crashlab $* (exit $got, want $want; pattern: $pat)"
    exit 1
  fi
  echo "ok  crashlab $*"
}
agent() { "$PY" -m examples.refund_bot "$@" >"$TMP/out" 2>&1 || { cat "$TMP/out"; exit 1; }; }

# record: one passing run, two failing runs of the same input (policy_drop fault)
agent "Refund my order A17, it arrived broken"
grep -q "outcome: pass" "$TMP/out"
agent --fault-policy-drop "Refund order C03 please"
agent --fault-policy-drop "Refund order C03 please"
grep -q "outcome: fail" "$TMP/out"
BAD=$(crashlab search C03 --outcome fail --json | "$PY" -c \
  "import json,sys; print(json.load(sys.stdin)['matches'][0]['run_id'])")
GOOD=$(crashlab search A17 --outcome pass --json | "$PY" -c \
  "import json,sys; print(json.load(sys.stdin)['matches'][0]['run_id'])")

step 0 "crashlab [0-9]"        -- --version
step 0 "refund-bot"            -- ls
step 0 "refund_within_policy"  -- show "$BAD"
for flag in --tools --llm --errors --failing; do
  step 0 "refund_within_policy" -- show "$BAD" "$flag"
done
step 0 "issue_refund"          -- show "$BAD" --step 3
step 0 "showing 0 of 5"        -- show "$BAD" --agent nobody
step 0 '"run_id"'              -- show "$BAD" --json
step 0 "issue_refund"          -- search C03 --tool issue_refund --since 1d --outcome fail
step 1 "No runs matching"      -- search no-such-text-anywhere
step 0 "refund_within_policy"  -- cluster
step 0 "2 runs"                -- flakes --all
step 0 "refund-bot"            -- watch --existing --timeout 1
step 0 "outcome: pass"         -- check "$GOOD"
step 1 "fail"                  -- check --all --fail-on error
step 0 "TASK_SPEC_VIOLATION"   -- diagnose "$BAD" --heuristic --save
step 0 "reused 1 stored"       -- triage
step 0 '"diagnoser": "heuristic"' -- triage --json --since 1d --name refund-bot
step 0 "TASK_SPEC_VIOLATION"   -- report "$BAD"
step 0 "wrote"                 -- report "$BAD" --html "$TMP/report.html"
grep -q "timing and cost" "$TMP/report.html"
step 0 "reproduced: 5/5"       -- replay "$BAD"
step 0 "outcome: fail -> pass" -- fork "$BAD" --at 2 --patch examples/refund_bot/restore_policy.json
step 0 "patch"                 -- fork "$BAD" --at 2 --system-append "x" --dry-run
step 0 "refund_within_policy"  -- diff "$BAD" "$GOOD"
step 0 "verdict: (VERIFIED|LIKELY)" -- verify "$BAD" --fix examples/refund_bot/restore_policy.json \
  -k 2 --regress "$GOOD"
step 0 "wrote"                 -- export-test "$BAD" -o "$TMP/tests"
"$PY" -m pytest --crashlab -q -p no:cacheprovider -o addopts='' --rootdir "$TMP/tests" \
  "$TMP/tests" >"$TMP/out" 2>&1 || { cat "$TMP/out"; echo "FAIL: exported test"; exit 1; }
echo "ok  pytest --crashlab (exported regression test)"
step 0 "exported"              -- export "$BAD" -o "$TMP/run.jsonl"
CRASHLAB_DIR="$TMP/store2" step 0 "imported" -- import "$TMP/run.jsonl"
CRASHLAB_DIR="$TMP/store2" step 0 "already there" -- import "$TMP/run.jsonl"
CRASHLAB_DIR="$TMP/store2" step 0 "refund-bot" -- show "${BAD}"
step 0 "spec_drop"             -- bench --offline --runs-dir bench/runs/seed-gemini-flash-lite-latest \
  --label smoke --out-dir "$TMP/bench"
echo "SMOKE OK: $n crashlab commands"
