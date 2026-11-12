"""``crashlab report RUN --html out.html``: one self-contained HTML file (inline CSS, no
scripts, no external assets) to share a run: outcome, checks, findings, diagnosis with
its quotes, fixes with diffs, verification stats, a per-step timing / cost waterfall and
the step timeline."""

from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from .models import Run

CSS = """
body{font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;margin:2rem auto;max-width:1100px;
color:#1d2330;padding:0 1rem}
h1{font-size:1.4rem;margin:0 0 .3rem}h2{font-size:1.1rem;margin:1.8rem 0 .5rem;
border-bottom:1px solid #dde;padding-bottom:.2rem}
table{border-collapse:collapse;width:100%;margin:.3rem 0}
td,th{border:1px solid #e3e6ee;padding:.3rem .5rem;text-align:left;vertical-align:top}
th{background:#f4f6fa;font-weight:600}
code,pre{font:12.5px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:#f7f8fb;border:1px solid #e3e6ee;padding:.6rem;overflow-x:auto;white-space:pre-wrap}
.pass,.VERIFIED,.LIKELY{color:#137333;font-weight:600}.fail,.error,.REJECTED,.critical{color:#b3261e;font-weight:600}
.unknown,.warn,.INCONCLUSIVE,.errored,.skipped{color:#a15c00;font-weight:600}.info{color:#555}
.badge{display:inline-block;padding:.1rem .5rem;border-radius:.6rem;background:#eef1f7}
.add{color:#137333}.del{color:#b3261e}.hunk{color:#5b6ad0}.muted{color:#6b7280}
.quote{background:#fff7d6;padding:0 .2rem}
.wf{position:relative;height:.9rem;min-width:180px;background:#f4f6fa;border-radius:2px}
.wf span{position:absolute;top:0;bottom:0;border-radius:2px;min-width:2px;background:#5b6ad0}
.wf .tool{background:#2a9d8f}.wf .err{background:#b3261e}.wf .other{background:#8a8fa3}
td.num{text-align:right;white-space:nowrap}
"""


def _e(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def _short(value: Any, width: int = 160) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _local(ts: Any) -> str:
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            return ts
    if not isinstance(ts, datetime):
        return ""
    ts = ts.astimezone()
    return ts.strftime("%Y-%m-%d %H:%M:%S ") + (ts.tzname() or "")


def _cls(word: Any) -> str:
    return f'<span class="{_e(word)}">{_e(word)}</span>'


def _diff_html(diff: str) -> str:
    out = []
    for line in diff.splitlines():
        cls = ""
        if line.startswith("+") and not line.startswith("+++"):
            cls = "add"
        elif line.startswith("-") and not line.startswith("---"):
            cls = "del"
        elif line.startswith("@@"):
            cls = "hunk"
        out.append(f'<span class="{cls}">{_e(line)}</span>' if cls else _e(line))
    return "<pre>" + "\n".join(out) + "</pre>"


def _table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{_e(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{th}</tr>{body}</table>"


def render_html(run: Run, verifications: list[Any] | None = None) -> str:
    from .patch import PatchError, PatchSet

    p: list[str] = []
    p.append(f"<h1>crashlab report: {_e(run.name)}</h1>")
    p.append(
        f'<p><span class="badge">{_cls(run.outcome)}</span> status {_e(run.status)} · '
        f"run <code>{_e(run.run_id)}</code> · {_e(_local(run.started_at))}</p>"
    )
    u = run.usage
    meta = [
        ["reason", _e(run.outcome_reason)],
        ["input", f"<code>{_e(_short(run.input, 400))}</code>"],
        ["output", _e(_short(run.output, 400))],
        [
            "usage",
            _e(
                f"{u.steps} steps · {u.llm_calls} LLM · {u.tool_calls} tool · "
                f"{u.total_tokens} tokens" + (f" · ${u.cost_usd:.4f}" if u.cost_usd else "")
            ),
        ],
    ]
    if run.forked_from:
        meta.append(["forked from", _e(f"{run.forked_from} at step {run.fork_step}")])
    if run.patch_description:
        meta.append(["patch", _e(run.patch_description)])
    p.append(_table(["", ""], meta).replace("<tr><th></th><th></th></tr>", ""))

    if run.checks:
        p.append("<h2>Checks</h2>")
        p.append(
            _table(
                ["check", "status", "message"],
                [[_e(c.name), _cls(c.status), _e(c.message)] for c in run.checks],
            )
        )
    if run.findings:
        p.append("<h2>Runtime findings</h2>")
        rows = [
            [
                _cls(f.severity),
                _e(f.kind),
                _e(", ".join(map(str, f.steps or [f.step]))),
                _e(f.message),
            ]
            for f in run.findings
        ]
        p.append(_table(["severity", "kind", "steps", "message"], rows))

    d = run.diagnosis
    if d is not None:
        p.append("<h2>Diagnosis</h2>")
        rows = [
            ["verdict", _e(f"{run.describe_step(d.failing_step)} · {d.label.value}")],
            ["confidence", _e(f"{d.confidence:.2f}")],
            ["root cause", _e(d.root_cause)],
            ["diagnoser", _e(d.model or "heuristic")],
        ]
        if d.secondary:
            rows.insert(
                1,
                ["also", _e(", ".join(s.value for s in d.secondary))],
            )
        for ev in d.evidence:
            rows.append(
                [
                    _e(f"evidence, step {ev.step_idx}"),
                    f'{_e(ev.field)}: <span class="quote">{_e(ev.quote)}</span>',
                ]
            )
        for n in d.notes:
            rows.append(["note", _e(n)])
        p.append(_table(["", ""], rows).replace("<tr><th></th><th></th></tr>", ""))
        if d.fixes or d.fix_notes:
            p.append("<h2>Proposed fixes</h2>")
        for f in d.fixes:
            p.append(
                f"<h3>{_e(f.fix_id)} · {_e(f.kind)} → {_e(f.target)}"
                + (" <span class='unknown'>(test only)</span>" if f.test_only else "")
                + "</h3>"
            )
            p.append(f"<p>{_e(f.after)}</p>")
            if f.rationale:
                p.append(f"<p><b>why:</b> {_e(f.rationale)}</p>")
            if f.expected_effect:
                p.append(f"<p><b>expect:</b> {_e(f.expected_effect)}</p>")
            for n in f.notes:
                p.append(f'<p class="unknown">note: {_e(n)}</p>')
            if f.as_patch is not None:
                at = (f.tried or {}).get("fork_step") or f.as_patch.target_step() or 0
                try:
                    diff = PatchSet(patches=[f.as_patch]).render_diff(run, fork_step=at)
                    p.append(_diff_html("\n".join(diff.splitlines()[1:]).strip()))
                except PatchError as exc:
                    p.append(f'<p class="muted">diff unavailable: {_e(exc)}</p>')
        for n in d.fix_notes:
            p.append(f'<p class="unknown">{_e(n)}</p>')

    for v in verifications or []:
        p.append(f"<h2>Verification of {_e(v.fix_id)}: {_cls(v.verdict)}</h2>")
        bits = [
            x
            for x in (
                getattr(v, "patch_kind", None),
                getattr(v, "patch_op", None),
                getattr(v, "patch_digest", None),
            )
            if x
        ]
        prov = (" · " + " ".join(_e(x) for x in bits)) if bits else ""
        p.append(
            f'<p class="muted">{_e(v.fix_id)}: {_e(v.patch_description)}{prov} · '
            f"{_e(_local(v.created_at))}</p>"
        )
        rows = []
        for name, a in (("patched", v.patched), ("control (no patch)", v.control)):
            if a is None:
                rows.append([_e(name), "not run", "", "", ""])
                continue
            ci = f"{a.ci95[0]:.2f}–{a.ci95[1]:.2f}" if a.ci95 else ""
            rate = f"{a.rate:.2f}" if a.rate is not None else ""
            pk = f"{a.pass_hat_k:.2f}" if a.pass_hat_k is not None else ""
            rows.append(
                [
                    _e(name),
                    _e(f"{a.passes}/{a.n}"),
                    _e(rate),
                    _e(ci),
                    _e(pk) + (f" · {a.errored} errored" if a.errored else ""),
                ]
            )
        p.append(_table(["arm", "passed", "rate", "Wilson 95% CI", "pass^k"], rows))
        if v.regression:
            p.append(
                _table(
                    ["past good run", "with the patch", "note"],
                    [
                        [f"<code>{_e(r.run_id)}</code>", _cls(r.status), _e(r.note)]
                        for r in v.regression
                    ],
                )
            )
        p.append("<ul>" + "".join(f"<li>{_e(r)}</li>" for r in v.reasons) + "</ul>")
        p.append(
            f'<p class="muted">cost ${v.cost_usd:.4f} · {v.tokens} tokens · '
            f"{len(v.trials)} trials (child runs tagged verify:{_e(v.verification_id)})</p>"
        )

    p.append(_timing_html(run))
    p.append("<h2>Step timeline</h2>")
    rows = []
    for s in run.steps:
        what = s.args if s.operation == "execute_tool" else None
        if s.operation == "chat":
            msg = (s.response or {}).get("message") or (s.response or {})
            res = _short(msg, 220)
        else:
            res = _short(s.result if s.error is None else None, 220)
        rows.append(
            [
                _e(s.idx),
                _e(s.operation),
                _e(s.name),
                _e(f"{s.duration_ms:.0f} ms"),
                _e(s.source),
                f"<code>{_e(_short(what, 160))}</code>" if what else "",
                _e(res)
                if s.error is None
                else f'<span class="fail">{_e(s.error.type)}: '
                f"{_e(_short(s.error.detail, 200))}</span>",
            ]
        )
    p.append(_table(["#", "op", "name", "time", "source", "args", "result / error"], rows))
    p.append(f'<p class="muted">generated by crashlab · {_e(_local(datetime.now()))}</p>')
    title = f"crashlab · {run.name} · {run.run_id[:10]}"
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_e(title)}</title><style>{CSS}</style></head><body>"
        + "\n".join(p)
        + "</body></html>\n"
    )


def _step_usd(step: Any) -> float | None:
    from .pricing import step_cost

    cost = step.attributes.get("crashlab.usage.cost_usd")
    return step_cost(step) if cost is None else float(cost)


def _timing_html(run: Run) -> str:
    """Per-step latency waterfall and cost table, from the recorded ``started_at`` /
    ``duration_ms`` and token / cost attributes. Says so instead of drawing bars when a
    run has no timing data; never estimates missing numbers."""
    out = ["<h2>Step timing and cost</h2>"]
    steps = sorted(run.steps, key=lambda s: s.idx)
    if not steps:
        out.append('<p class="muted">No steps recorded.</p>')
        return "\n".join(out)
    if all(not s.duration_ms for s in steps):
        out.append(
            '<p class="muted">No step durations were recorded for this run (every step has '
            "duration 0), so there is no latency waterfall.</p>"
        )
        return "\n".join(out)
    t0 = min(s.started_at for s in steps)
    offs = [max((s.started_at - t0).total_seconds() * 1000, 0.0) for s in steps]
    span = max(o + max(s.duration_ms, 0.0) for o, s in zip(offs, steps, strict=True)) or 1.0
    rows = []
    llm_ms = tool_ms = 0.0
    total_cost, unknown_cost, in_tok, out_tok = 0.0, 0, 0, 0
    for off, s in zip(offs, steps, strict=True):
        dur = max(s.duration_ms, 0.0)
        if s.error:
            kind = "err"
        elif s.operation in ("chat", "embeddings"):
            kind = ""
        elif s.operation in ("execute_tool", "invoke_agent"):
            kind = "tool"
        else:
            kind = "other"
        if s.operation in ("chat", "embeddings"):
            llm_ms += dur
        elif s.operation in ("execute_tool", "invoke_agent"):
            tool_ms += dur
        left, width = off / span * 100, dur / span * 100
        bar = (
            f'<div class="wf" title="{_e(f"+{off:.0f} ms, {dur:.1f} ms")}">'
            f'<span class="{kind}" style="left:{left:.2f}%;width:{width:.2f}%"></span></div>'
        )
        tokens = cost = ""
        if s.operation == "chat":
            a = s.attributes or {}
            i, o = a.get("gen_ai.usage.input_tokens"), a.get("gen_ai.usage.output_tokens")
            if i is not None or o is not None:
                in_tok, out_tok = in_tok + int(i or 0), out_tok + int(o or 0)
                tokens = f"{int(i or 0)} / {int(o or 0)}"
            usd = _step_usd(s)
            if usd is None:
                unknown_cost += 1
                cost = "?"
            else:
                total_cost += usd
                cost = f"${usd:.4f}" if usd >= 0.0001 or usd == 0 else f"${usd:.6f}"
        rows.append(
            f"<tr><td>{_e(s.idx)}</td><td>{_e(s.operation)}</td><td>{_e(s.name)}</td>"
            f'<td class="num">+{off:,.0f} ms</td><td class="num">{dur:,.1f} ms</td>'
            f'<td class="num">{_e(tokens)}</td><td class="num">{_e(cost)}</td>'
            f"<td>{bar}</td></tr>"
        )
    head = "".join(
        f"<th>{h}</th>"
        for h in (
            "#",
            "op",
            "name",
            "start",
            "duration",
            "tokens in / out",
            "est. cost",
            "waterfall",
        )  # fmt: skip
    )
    out.append(f"<table><tr>{head}</tr>{''.join(rows)}</table>")
    slow = max(steps, key=lambda s: s.duration_ms)
    facts = [
        f"span {span:,.0f} ms from the first step's start",
        f"LLM calls {llm_ms:,.0f} ms",
        f"tools {tool_ms:,.0f} ms",
        f"slowest: step {slow.idx} ({slow.operation} {slow.name}, {slow.duration_ms:,.0f} ms)",
    ]
    if run.usage.wall_time_ms is not None:
        facts.insert(0, f"run wall time {run.usage.wall_time_ms:,.0f} ms")
    if any(s.operation == "chat" for s in steps):
        facts.append(f"{in_tok:,} / {out_tok:,} tokens")
        facts.append(
            f"est. cost ${total_cost:.4f}"
            + (f" + {unknown_cost} call(s) with no known price" if unknown_cost else "")
        )
    out.append(f'<p class="muted">{_e(" · ".join(facts))}</p>')
    notes = [
        "Durations are wall-clock times measured around each call by crashlab, including "
        "network time; overlapping bars are calls that ran concurrently."
    ]
    if any(s.source != "live" for s in steps):
        notes.append(
            "Steps served from a cassette, patched, blocked or faked show the time of this "
            "run (a replay or fork), not of the original call."
        )
    if unknown_cost:
        notes.append(
            "'?' cost: the model has no price in crashlab's table or [tool.crashlab.prices], "
            "or the call reported no usage."
        )
    out.append("".join(f'<p class="muted">{_e(n)}</p>' for n in notes))
    return "\n".join(out)


def write_html(run: Run, path: str | Path, verifications: list[Any] | None = None) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(run, verifications), "utf-8")
    return out
