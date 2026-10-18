"""Score ``crashlab.diagnose`` on the hand-labelled refund-bot runs in diagnose_cases.json.

    python bench/diagnose_eval.py --heuristic                       # rule-based baseline
    python bench/diagnose_eval.py --model openai/gpt-oss-120b \\
        --base-url https://api.groq.com/openai/v1 --api-key-env GROQ_API_KEY
    python bench/diagnose_eval.py --model gemini/gemini-flash-latest  # via LiteLLM

Each run is imported into a scratch store, re-checked with the current runtime checks and
refund-bot oracles, then diagnosed with the run's fault switches, model and provider
hidden (they would give the answer away). Metrics:

- label accuracy: the label is one of the case's accepted labels (all cases)
- step exact / step in window: on the failure cases, the culprit step equals the expected
  one / lies in the expected window (a correct label is not required)
- false alarms: failure labels on runs labelled NO_FAILURE only
- evidence: quotes that were found verbatim / quotes given; downgrades to UNKNOWN
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import crashlab  # noqa: E402
from crashlab.checks import evaluate, replace_findings, resolve_config  # noqa: E402
from crashlab.diagnose import PROMPT_VERSION, DiagnoseError  # noqa: E402
from crashlab.store import Store, import_run  # noqa: E402
from crashlab.verify.oracles import apply_outcome, run_checks  # noqa: E402

HIDE = ("faults", "model", "provider")


def load_cases(path: Path) -> tuple[list[dict[str, Any]], str]:
    doc = json.loads(path.read_text("utf-8"))
    return doc["cases"], doc["agent"]


def prepare(store: Store, case: dict[str, Any]) -> crashlab.Run:
    run_id, _ = import_run(store, ROOT / case["file"], replace=True)
    run = store.load_run(run_id)
    replace_findings(run, evaluate(run, resolve_config(None, base=run.check_config), store=store))
    apply_outcome(run, run_checks(run))
    run.diagnosis = None
    store.update_run(run)
    return run


def diagnose_with_retry(run: crashlab.Run, args: argparse.Namespace, store: Store, key: Any):
    for attempt in range(args.retries + 1):
        try:
            return crashlab.diagnose(
                run,
                args.model,
                base_url=args.base_url,
                api_key=key,
                heuristic=args.heuristic,
                store=store,
                hide_input=HIDE,
                max_field=args.max_chars,
                cache=not args.no_cache,
            )
        except DiagnoseError as e:
            limited = any(k in str(e) for k in ("429", "RateLimit", "rate limit", "quota"))
            daily = "daily quota" in str(e)  # a daily quota will not recover by waiting
            if attempt == args.retries or not limited or daily:
                raise
            wait = args.backoff * (attempt + 1)
            print(f"  rate limited, waiting {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
    raise AssertionError("unreachable")


def score(case: dict[str, Any], d: crashlab.Diagnosis | None, error: str | None) -> dict[str, Any]:
    clean_only = case["labels"] == ["NO_FAILURE"]
    failure_case = case["step"] is not None
    row: dict[str, Any] = {
        "run": case["run"],
        "expected": case["labels"],
        "expected_step": case["step"],
        "window": case["window"],
        "error": error,
    }
    if d is None:
        row.update(
            label=None,
            step=None,
            label_ok=False,
            step_exact=False,
            step_in_window=False,
            false_alarm=False,
            downgraded=False,
            evidence_valid=0,
            evidence_total=0,
        )
        return row
    step = d.failing_step
    lo, hi = case["window"] or (None, None)
    m = d.method or {}
    row.update(
        label=d.label.value,
        step=step,
        confidence=d.confidence,
        root_cause=d.root_cause,
        evidence=[{"step": e.step_idx, "field": e.field, "quote": e.quote} for e in d.evidence],
        notes=d.notes,
        label_ok=d.label.value in case["labels"],
        step_exact=failure_case and step == case["step"],
        step_in_window=failure_case and step is not None and lo <= step <= hi,
        false_alarm=clean_only and d.is_failure,
        downgraded=any(n.startswith("downgraded") for n in d.notes),
        evidence_valid=m.get("evidence_valid", len(d.evidence)),
        evidence_total=m.get("evidence_total", len(d.evidence)),
        input_tokens=m.get("input_tokens"),
        output_tokens=m.get("output_tokens"),
        cost_usd=m.get("cost_usd"),
        seconds=m.get("seconds"),
        cached=m.get("cached", False),
    )
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    fail_rows = [r for r in rows if r["expected_step"] is not None]
    clean_rows = [r for r in rows if r["expected"] == ["NO_FAILURE"]]
    total_ev = sum(r["evidence_total"] for r in rows)

    def pct(k: int, d: int) -> float | None:
        return round(100 * k / d, 1) if d else None

    return {
        "cases": n,
        "label_accuracy": pct(sum(r["label_ok"] for r in rows), n),
        "failure_cases": len(fail_rows),
        "failure_label_accuracy": pct(sum(r["label_ok"] for r in fail_rows), len(fail_rows)),
        "step_exact": pct(sum(r["step_exact"] for r in fail_rows), len(fail_rows)),
        "step_in_window": pct(sum(r["step_in_window"] for r in fail_rows), len(fail_rows)),
        "clean_cases": len(clean_rows),
        "false_alarms": sum(r["false_alarm"] for r in clean_rows),
        "unknown": sum(r["label"] == "UNKNOWN" for r in rows),
        "downgraded": sum(r["downgraded"] for r in rows),
        "errors": sum(r["error"] is not None for r in rows),
        "evidence_valid": f"{sum(r['evidence_valid'] for r in rows)}/{total_ev}",
        "input_tokens": sum(r.get("input_tokens") or 0 for r in rows),
        "output_tokens": sum(r.get("output_tokens") or 0 for r in rows),
        "cost_usd": round(sum(r.get("cost_usd") or 0 for r in rows), 5),
        "seconds_per_case": round(sum(r.get("seconds") or 0 for r in rows) / max(n, 1), 2),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", help="diagnoser model (default: $CRASHLAB_DIAGNOSER_MODEL)")
    p.add_argument("--base-url", help="OpenAI-compatible endpoint")
    p.add_argument("--api-key-env", help="env var holding the API key (never the key itself)")
    p.add_argument("--heuristic", action="store_true", help="rule-based baseline, no LLM")
    p.add_argument("--cases", default=str(ROOT / "bench" / "diagnose_cases.json"))
    p.add_argument("--only", nargs="*", help="run id prefixes to evaluate")
    p.add_argument("--store", help="store dir to keep (and cache) diagnoses (default: temp)")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--sleep", type=float, default=0.0, help="seconds between cases")
    p.add_argument("--retries", type=int, default=3, help="retries on rate limits")
    p.add_argument("--backoff", type=float, default=20.0, help="seconds, times the attempt")
    p.add_argument("--max-chars", type=int, default=600)
    p.add_argument("--out", help="results JSON (default: bench/results/<name>.json)")
    args = p.parse_args(argv)

    key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not key:
        p.error(f"${args.api_key_env} is not set")
    cases, agent = load_cases(Path(args.cases))
    if args.only:
        cases = [c for c in cases if any(c["run"].startswith(o) for o in args.only)]
    __import__(agent)  # registers the @crashlab.check oracles
    name = (
        "heuristic"
        if args.heuristic
        else (args.model or os.environ.get("CRASHLAB_DIAGNOSER_MODEL") or "?")
    )

    tmp = None if args.store else tempfile.TemporaryDirectory(prefix="crashlab-eval-")
    store = Store(args.store or tmp.name)  # type: ignore[union-attr]
    rows = []
    for i, case in enumerate(cases):
        if i and args.sleep:
            time.sleep(args.sleep)
        run = prepare(store, case)
        d, error = None, None
        try:
            d = diagnose_with_retry(run, args, store, key)
        except DiagnoseError as e:
            error = str(e)[:300]
        row = score(case, d, error)
        rows.append(row)
        mark = "ok " if row["label_ok"] else "BAD"
        got = f"{row['label']}@{row['step']}" if d else f"ERROR {error}"
        print(
            f"{mark} {case['run']}  expected {'|'.join(case['labels'])}@{case['step']}  got {got}"
        )

    summary = {"diagnoser": name, **summarize(rows)}
    print(json.dumps(summary, indent=1))
    slug = name.replace("/", "_").replace(":", "_")
    out = Path(args.out or ROOT / "bench" / "results" / f"{slug}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"summary": summary, "prompt_version": PROMPT_VERSION, "cases": rows}
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", "utf-8")
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    if tmp is not None:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
