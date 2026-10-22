"""``crashlab bench``: run the seed fault-injection benchmark live, or rescore it offline.

Live (:func:`run_live`): for each task x condition (clean + every fault) x episode, run
the agent on the model, then the full loop on the recorded run: oracles -> heuristic and
LLM diagnosis -> fix -> ``verify`` (k patched, k controls, regression on this bench's
clean runs). Runs are tagged (``bench:<id>``, ``task:``, ``cond:``, ``ep:``) and exported,
so :func:`score_runs` can recompute every number from the cassettes alone (offline).
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .._ids import new_ulid
from ..models import Run
from ..verify.stats import wilson
from .catalog import CONDITIONS, FAULTS, TASKS

HIDE = ("faults", "model", "provider")
DAILY = re.compile(r"PerDay|per day|daily quota", re.I)
LIMITED = re.compile(r"429|rate.?limit|quota|RESOURCE_EXHAUSTED|503|overloaded|high demand", re.I)


class QuotaExhausted(RuntimeError):
    """The provider's daily quota is used up: stop the benchmark, keep what ran."""


def _tag(run: Run, prefix: str) -> str | None:
    return next((t.split(":", 1)[1] for t in run.tags if t.startswith(prefix + ":")), None)


# --- live -------------------------------------------------------------------------------


def _retrying(fn: Callable[[], Any], log: Callable[[str], None], *, waits=(60, 120)) -> Any:
    for attempt in range(len(waits) + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - classify provider failures
            text = f"{type(e).__name__}: {e}"
            if DAILY.search(text):
                raise QuotaExhausted(text[:300]) from e
            if attempt == len(waits) or not LIMITED.search(text):
                raise
            log(f"  rate limited, waiting {waits[attempt]}s")
            time.sleep(waits[attempt])
    raise AssertionError("unreachable")


def run_live(
    model: str,
    *,
    base_url: str,
    api_key: str,
    tasks: Iterable[str] = tuple(TASKS),
    conditions: Iterable[str] = CONDITIONS,
    episodes: int = 3,
    k: int = 3,
    verify_per_cell: int = 1,
    regression_runs: int = 2,
    pace_s: float = 4.0,
    store_dir: Any = None,
    log: Callable[[str], None] = print,
) -> tuple[str, list[str], str | None]:
    """Record and process the episodes. Returns ``(bench_id, run_ids, stopped_reason)``.
    The agent uses ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` (set here for this process)."""
    import crashlab
    from crashlab.store import Store

    from ..verify.runner import is_infra_error

    os.environ["OPENAI_BASE_URL"], os.environ["OPENAI_API_KEY"] = base_url, api_key
    if store_dir is not None:
        os.environ["CRASHLAB_DIR"] = str(store_dir)
    store = Store(store_dir)
    bench_id = new_ulid()
    ids: list[str] = []
    clean: dict[str, list[str]] = {}
    verified: dict[tuple[str, str], int] = {}
    stopped: str | None = None
    try:
        for tname in tasks:
            task = TASKS[tname]
            for cond in conditions:
                prompts = task.prompts_for(cond)
                for ep in range(episodes):
                    prompt = prompts[ep % len(prompts)]

                    def once(prompt: str = prompt, cond: str = cond, task: Any = task) -> Run:
                        before = store.load_run("last").run_id if store.list_runs() else None
                        try:
                            task.run(prompt, model, cond)
                        except Exception as e:  # noqa: BLE001
                            # a model-side failure (e.g. Groq tool_use_failed 400) is an
                            # episode outcome, recorded as an errored run; anything that
                            # did not even record a run, or is a quota/5xx, propagates
                            r = store.load_run("last") if store.list_runs() else None
                            if r is None or r.run_id == before or r.status != "error":
                                raise
                            if LIMITED.search(str(e)) or DAILY.search(str(e)):
                                raise
                        r = store.load_run("last")
                        if r.status == "error" and is_infra_error(r.error):
                            raise RuntimeError(f"{r.error.type}: {r.error.message}")
                        return r

                    t0 = time.time()
                    run = _retrying(once, log)
                    run.tags += [
                        "bench",
                        f"bench:{bench_id}",
                        f"task:{tname}",
                        f"cond:{cond}",
                        f"ep:{ep}",
                    ]
                    store.update_run(run)
                    ids.append(run.run_id)
                    log(f"{tname:<12} {cond:<16} ep{ep} {run.run_id[:10]} {run.outcome}")
                    time.sleep(pace_s)
                    d = _retrying(
                        lambda run=run: crashlab.diagnose(
                            run,
                            model,
                            base_url=base_url,
                            api_key=api_key,
                            store=store,
                            hide_input=HIDE,
                        ),
                        log,
                    )
                    log(f"  diagnosis {d.label.value}@{d.failing_step}")
                    if cond == "clean" and run.outcome == "pass":
                        clean.setdefault(tname, []).append(run.run_id)
                    cell = (tname, cond)
                    if (
                        cond != "clean"
                        and run.outcome == "fail"
                        and d.is_failure
                        and verified.get(cell, 0) < verify_per_cell
                    ):
                        verified[cell] = verified.get(cell, 0) + 1
                        time.sleep(pace_s)
                        _fix_and_verify(
                            run,
                            model,
                            base_url,
                            api_key,
                            store,
                            k,
                            clean.get(tname, [])[:regression_runs],
                            pace_s,
                            log,
                        )
                    run = store.load_run(run.run_id)
                    run.env["bench_seconds"] = round(time.time() - t0, 1)
                    store.update_run(run)
                    time.sleep(pace_s)
    except QuotaExhausted as e:
        stopped = f"daily quota exhausted: {e}"[:300]
        log(stopped)
    return bench_id, ids, stopped


def _fix_and_verify(run, model, base_url, api_key, store, k, regress, pace_s, log) -> None:
    import crashlab

    fixes = _retrying(
        lambda: crashlab.propose_fixes(
            run, model, base_url=base_url, api_key=api_key, store=store, hide_input=HIDE
        ),
        log,
    )
    top = next((f for f in fixes if not f.test_only), None)
    if top is None:
        log("  no usable fix proposed")
        return
    time.sleep(pace_s)
    v = crashlab.verify(
        store.load_run(run.run_id),
        top,
        k=k,
        regress=regress or "none",
        store_dir=store.root,
        pace_s=pace_s,
    )
    if any(t.status == "errored" and t.error and DAILY.search(t.error) for t in v.trials):
        log(f"  verify {v.verdict} (some trials hit the daily quota)")
    log(
        f"  verify {top.fix_id}: {v.verdict} patched {v.patched.passes}/{v.patched.n} "
        f"control {v.control.passes if v.control else '-'}/{v.control.n if v.control else '-'}"
    )


# --- scoring (offline; the same code scores a live run and a rescore) --------------------


def _diag_row(d: Any, labels: frozenset[str], step: Any, window: Any) -> dict[str, Any]:
    if d is None:
        return {
            "label": None,
            "step": None,
            "label_ok": False,
            "step_exact": False,
            "in_window": False,
            "is_failure": False,
        }
    lo, hi = window or (None, None)
    s = d.failing_step
    return {
        "label": d.label.value,
        "step": s,
        "label_ok": d.label.value in labels,
        "step_exact": step is not None and s == step,
        "in_window": lo is not None and s is not None and lo <= s <= hi,
        "is_failure": d.is_failure,
    }


def score_run(run: Run) -> dict[str, Any]:
    """One episode's record, from the run alone (its tags, checks, stored diagnosis and
    fix attempts); the heuristic baseline is recomputed (deterministic, no LLM)."""
    import crashlab

    task = TASKS[_tag(run, "task") or run.name]
    cond = _tag(run, "cond") or "clean"
    fault = FAULTS.get(cond)
    labels = fault.labels if fault else frozenset({"NO_FAILURE"})
    triggered = fault is not None and run.outcome == "fail"
    step, window = task.ground_truth(run, cond) if triggered else (None, None)
    llm = run.diagnosis if run.diagnosis is not None and run.diagnosis.model else None
    heur = crashlab.diagnose(run, heuristic=True, save=False, cache=False)
    attempt = run.fix_attempts[-1] if run.fix_attempts else None
    usage = run.usage
    m = (llm.method if llm else {}) or {}
    fm = (llm.fix_method if llm else {}) or {}
    return {
        "run_id": run.run_id,
        "task": task.name,
        "split": task.split,
        "condition": cond,
        "episode": int(_tag(run, "ep") or 0),
        "model": str(run.input.get("model")),
        "outcome": run.outcome,
        "failed_checks": [c.name for c in run.checks if c.status != "pass"],
        "triggered": triggered,
        "flagged_by_findings": any(f.severity in ("error", "critical") for f in run.findings),
        "expected_labels": sorted(labels),
        "expected_step": step,
        "window": list(window) if window else None,
        "llm": _diag_row(llm, labels, step, window),
        "heuristic": _diag_row(heur, labels, step, window),
        "fix": {
            "proposed": len(llm.fixes) if llm else 0,
            "verdict": attempt.get("verdict") if attempt else None,
            "patched": attempt.get("patched") if attempt else None,
            "control": attempt.get("control") if attempt else None,
            "regressions": attempt.get("regressions") if attempt else None,
        },
        "cost_usd": round(
            (usage.cost_usd or 0)
            + (m.get("cost_usd") or 0)
            + (fm.get("cost_usd") or 0)
            + (attempt.get("cost_usd") or 0 if attempt else 0),
            6,
        ),
        "seconds": run.env.get("bench_seconds"),
    }


def _frac(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n) if n else (None, None)
    return {
        "k": k,
        "n": n,
        "rate": round(k / n, 3) if n else None,
        "ci95": [round(lo, 3), round(hi, 3)] if n else None,
    }


def summarize(rows: list[dict[str, Any]], who: str) -> dict[str, Any]:
    faulted = [r for r in rows if r["condition"] != "clean"]
    trig = [r for r in faulted if r["triggered"]]
    clean = [r for r in rows if r["condition"] == "clean" and r["outcome"] == "pass"]
    tried = [r for r in trig if r["fix"]["verdict"]]
    return {
        "faulted_runs": len(faulted),
        "triggered": _frac(len(trig), len(faulted)),
        "detected": _frac(sum(r[who]["is_failure"] for r in trig), len(trig)),
        "flagged_by_findings": _frac(sum(r["flagged_by_findings"] for r in trig), len(trig)),
        "label_ok": _frac(sum(r[who]["label_ok"] for r in trig), len(trig)),
        "step_exact": _frac(sum(r[who]["step_exact"] for r in trig), len(trig)),
        "in_window": _frac(sum(r[who]["in_window"] for r in trig), len(trig)),
        "clean_false_positive": _frac(sum(r[who]["is_failure"] for r in clean), len(clean)),
        "fix_verified": _frac(sum(r["fix"]["verdict"] == "VERIFIED" for r in tried), len(tried)),
        "fix_verified_or_likely": _frac(
            sum(r["fix"]["verdict"] in ("VERIFIED", "LIKELY") for r in tried), len(tried)
        ),
        "fix_regressions": sum(r["fix"]["regressions"] or 0 for r in tried),
        "cost_usd": round(sum(r["cost_usd"] for r in rows), 5),
        "seconds_per_episode": round(
            sum(r["seconds"] or 0 for r in rows)
            / max(1, sum(r["seconds"] is not None for r in rows)),
            1,
        ),
    }


def score_runs(runs: list[Run], *, label: str, stopped: str | None = None) -> dict[str, Any]:
    rows = sorted(
        (score_run(r) for r in runs),
        key=lambda r: (r["task"], CONDITIONS.index(r["condition"]), r["episode"]),
    )
    out: dict[str, Any] = {
        "benchmark": "crashlab seed v0.1",
        "label": label,
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stopped": stopped,
        "faults": {
            f.name: {"description": f.description, "labels": sorted(f.labels)}
            for f in FAULTS.values()
        },
        "splits": {t.name: t.split for t in TASKS.values()},
        "summary": {},
        "episodes": rows,
    }
    for split in ("dev", "test", "all"):
        part = [r for r in rows if split == "all" or r["split"] == split]
        if part:
            out["summary"][split] = {w: summarize(part, w) for w in ("llm", "heuristic")}
    return out


# --- markdown ---------------------------------------------------------------------------


def _cell(f: dict[str, Any]) -> str:
    if not f["n"]:
        return "–"
    lo, hi = f["ci95"]
    return f"{f['k']}/{f['n']} ({100 * f['rate']:.0f}%, {100 * lo:.0f}–{100 * hi:.0f})"


def render_md(res: dict[str, Any]) -> str:
    lines = [f"### {res['label']}", ""]
    if res.get("stopped"):
        lines += [f"> Partial: {res['stopped'][:160]}", ""]
    lines += [
        "| split | diagnoser | fault triggered | detected | label ok | step exact | step in "
        "window | fix VERIFIED | VERIFIED or LIKELY | clean false alarms | cost |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for split, by in res["summary"].items():
        for who, s in by.items():
            name = res["label"].split(" ")[0] if who == "llm" else "heuristic"
            fixes = (
                (_cell(s["fix_verified"]), _cell(s["fix_verified_or_likely"]))
                if who == "llm"
                else ("–", "–")
            )
            lines.append(
                f"| {split} | {name} | {_cell(s['triggered'])} | {_cell(s['detected'])} | "
                f"{_cell(s['label_ok'])} | {_cell(s['step_exact'])} | {_cell(s['in_window'])} | "
                f"{fixes[0]} | {fixes[1]} | {_cell(s['clean_false_positive'])} | "
                + (f"${s['cost_usd']:.4f}" if who == "llm" else "$0")
                + " |"
            )
    lines += [
        "",
        "Per cell (task × fault): triggered / LLM label ok / heuristic label ok / fix verdicts",
        "",
        "| task | condition | runs | triggered | LLM label | heuristic label | fix verdicts |",
        "|---|---|---|---|---|---|---|",
    ]
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in res["episodes"]:
        cells.setdefault((r["task"], r["condition"]), []).append(r)
    for (task, cond), rs in cells.items():
        trig = [r for r in rs if r["triggered"]]
        if cond == "clean":
            ok = f"{sum(not r['llm']['is_failure'] for r in rs)}/{len(rs)} no-failure"
            hk = f"{sum(not r['heuristic']['is_failure'] for r in rs)}/{len(rs)} no-failure"
        else:
            ok = f"{sum(r['llm']['label_ok'] for r in trig)}/{len(trig)}"
            hk = f"{sum(r['heuristic']['label_ok'] for r in trig)}/{len(trig)}"
        verdicts = ", ".join(r["fix"]["verdict"] for r in rs if r["fix"]["verdict"]) or "–"
        n_trig = "–" if cond == "clean" else f"{len(trig)}/{len(rs)}"
        lines.append(f"| {task} | {cond} | {len(rs)} | {n_trig} | {ok} | {hk} | {verdicts} |")
    return "\n".join(lines) + "\n"


def write_results(res: dict[str, Any], out_dir: str | Path, slug: str) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    j, m = out / f"seed_{slug}.json", out / f"seed_{slug}.md"
    j.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", "utf-8")
    m.write_text(render_md(res), "utf-8")
    return j, m


def export_runs(store: Any, run_ids: list[str], runs_dir: str | Path) -> int:
    from ..export import _export

    d = Path(runs_dir)
    d.mkdir(parents=True, exist_ok=True)
    for rid in run_ids:
        _export(store, rid, d / f"{rid}.jsonl")
    return len(run_ids)


def load_runs_dir(runs_dir: str | Path, store_dir: str | Path) -> list[Run]:
    """Import exported bench runs into ``store_dir`` (a scratch store) and load them."""
    from ..store import Store, import_run

    store = Store(store_dir)
    ids = [import_run(store, p, replace=True)[0] for p in sorted(Path(runs_dir).glob("*.jsonl"))]
    return [store.load_run(i) for i in ids]
