#!/usr/bin/env bash
# End-to-end CLI smoke test: every crashlab command against the installed `crashlab`,
# offline (a local fake OpenAI server, no keys, no network). Needs the openai SDK
# and pytest (the exported regression test is run with pytest --crashlab):
#   pip install -e ".[dev,openai]" && scripts/smoke.sh
#   # or: pip install "crashlab[openai,pytest]"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
if ! "$PY" -c "import pytest" 2>/dev/null; then
  echo "scripts/smoke.sh needs pytest to run the exported regression test."
  echo 'Install with: pip install pytest   or   pip install "crashlab[pytest]"   or   pip install -e ".[dev,openai]"'
  exit 1
fi
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
step 0 "refund-bot"            -- ls --trace "$BAD"
step 0 "prod"                  -- tag "$GOOD" +prod
step 0 "$GOOD"                 -- ls --tag prod
step 0 '"tags"'                -- tag "$GOOD" --json
step 1 "Pass --yes"            -- rm "$GOOD"
step 0 "would delete|Nothing to delete|kept" -- prune --max-runs 100 --dry-run
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
step 0 "scenario"              -- stress "$GOOD" --scenario tool_empty --json
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

# Wave 1–5 commands, still 100% offline against the local fake server / store
step 0 "runs"                  -- stats
step 0 "wrote"                 -- stats --html "$TMP/stats.html"
grep -q "<html" "$TMP/stats.html"
step 0 "refund"                -- timeline "$BAD"
step 0 "crashlab eval"         -- eval --outcome pass
step 0 "fixed|unchanged|verdict" -- sweep "$BAD" --patch examples/refund_bot/restore_policy.json --at-step 2 --json
step 0 "first_divergence|outcome" -- bisect "$GOOD" "$BAD" --json
step 0 "min_score|passed"      -- score "$GOOD" --min-score 0
step 0 "wrote"                 -- baseline save --out "$TMP/baseline.json"
step 0 "no regressions"        -- baseline check "$TMP/baseline.json"
step 0 "store check"           -- doctor
step 0 "profile"               -- profile "$GOOD"
step 0 "context"               -- context "$GOOD"
step 0 "no schema issues|called:|issue_refund|get_order" -- audit-tools "$GOOD"
step 0 "wrote"                 -- dataset build "$GOOD" --out "$TMP/dataset.jsonl"
step 0 "case"                  -- dataset inspect "$TMP/dataset.jsonl"
step 0 "needs_live_agent|survived|mutation" -- fuzz "$BAD" --mutation tool_poison --max-cases 1
step 0 "wrote html"            -- diff "$BAD" "$GOOD" --html "$TMP/diff.html"
step 0 "wrote markdown"        -- diff "$BAD" "$GOOD" --markdown "$TMP/diff.md"
step 1 "fail|regression"       -- diff "$GOOD" "$BAD" --fail-on-regression
step 0 "bundled"               -- bundle "$BAD" -o "$TMP/run.crashlab.zip"
CRASHLAB_DIR="$TMP/store3" step 0 "imported" -- unbundle "$TMP/run.crashlab.zip"
step 0 "audit-secrets|no secrets found|hit" -- audit-secrets "$GOOD"
step 0 "exported"              -- export "$BAD" --format otel -o "$TMP/run.otel.json"
step 0 "exported"              -- export "$BAD" --format chrome-trace -o "$TMP/run.trace.json"
# Today's features: judges (no model needed for --help), guard-report, trace import
step 0 "judge model" -- judge --help
step 0 "no guard interventions" -- guard-report "$GOOD"
step 0 '"interventions"|no guard interventions' -- guard-report "$GOOD" --json
CRASHLAB_DIR="$TMP/store4" step 0 "otel, 3 steps" -- import tests/fixtures/ingest/otel_genai.json --format otel
CRASHLAB_DIR="$TMP/store4" step 0 "langsmith, 2 steps" -- import tests/fixtures/ingest/langsmith_rag_run.json --format langsmith
CRASHLAB_DIR="$TMP/store4" step 0 "langfuse" -- import tests/fixtures/ingest/langfuse_trace.json --format langfuse
CRASHLAB_DIR="$TMP/store4" step 0 "imported|already there" -- import tests/fixtures/ingest/openinference_phoenix.json --format openinference
CRASHLAB_DIR="$TMP/store4" step 0 "outcome|steps|rag_chain|llm" -- show last
# Nov 27 wave 2: repair, prompt versions, memory/state (help only; colour-safe patterns)
step 0 "candidate patches" -- repair --help
step 0 "prompt versions" -- prompts --help
step 0 "lost context" -- memory --help
step 0 "state snapshots" -- state --help
echo "SMOKE OK: $n crashlab commands"
