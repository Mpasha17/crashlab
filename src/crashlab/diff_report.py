"""HTML and Markdown reports for ``crashlab.diff`` / ``crashlab diff``.

``render_diff_html`` writes a self-contained side-by-side comparison (inline CSS
and a small vanilla JS filter, no external assets). ``render_diff_markdown`` is a
GitHub-flavored summary meant for ``gh pr comment``. ``has_regression`` is the CI
gate used by ``crashlab diff --fail-on-regression``.
"""

from __future__ import annotations

import json
from html import escape
from typing import Any

from .compare import RunDiff
from .models import Finding, Run, Step
from .pricing import run_cost, step_cost

_BAD_FINDING = frozenset({"warn", "error", "critical"})
_CHECK_REGRESS = frozenset({"fail", "error"})

_STEP_LABEL = {
    "same": "same",
    "response": "≠ response",
    "request": "≠ request",
    "different": "≠ call",
    "only_a": "only A",
    "only_b": "only B",
}

CSS = """
body{font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;margin:2rem auto;max-width:1200px;
color:#1d2330;padding:0 1rem}
h1{font-size:1.4rem;margin:0 0 .3rem}h2{font-size:1.1rem;margin:1.8rem 0 .5rem;
border-bottom:1px solid #dde;padding-bottom:.2rem}
table{border-collapse:collapse;width:100%;margin:.3rem 0}
td,th{border:1px solid #e3e6ee;padding:.3rem .5rem;text-align:left;vertical-align:top}
th{background:#f4f6fa;font-weight:600}
code,pre{font:12.5px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:#f7f8fb;border:1px solid #e3e6ee;padding:.6rem;overflow-x:auto;white-space:pre-wrap}
.muted{color:#6b7280}.pass,.ok,.same,.improvement{color:#137333;font-weight:600}
.fail,.error,.critical,.regression,.only_a,.only_b,.different{color:#b3261e;font-weight:600}
.unknown,.warn,.request,.response{color:#a15c00;font-weight:600}
.cards{display:flex;flex-wrap:wrap;gap:.75rem;margin:1rem 0}
.card{flex:1 1 9rem;border:1px solid #e3e6ee;border-radius:.5rem;padding:.7rem .9rem;
background:#f4f6fa}
.card .lbl{font-size:.75rem;color:#6b7280;text-transform:uppercase;letter-spacing:.04em}
.card .val{font-size:1.2rem;font-weight:650;margin-top:.15rem}
.card .sub{font-size:.8rem;color:#6b7280;margin-top:.15rem}
.pills{display:flex;flex-wrap:wrap;gap:.4rem;margin:.6rem 0 1rem}
.pill{display:inline-block;padding:.15rem .6rem;border-radius:.8rem;border:1px solid #e3e6ee;
background:#eef1f7;font-size:.85rem}
.pill.regression{background:#fdecea;border-color:#f4c7c3;color:#b3261e;font-weight:600}
.pill.improvement{background:#e6f4ea;border-color:#c4e4cd;color:#137333;font-weight:600}
.filters{display:flex;flex-wrap:wrap;gap:.4rem;margin:.4rem 0 1rem}
.filters button{font:inherit;padding:.2rem .7rem;border-radius:.8rem;border:1px solid #c9cdd8;
background:#fff;cursor:pointer}
.filters button.active{background:#1d2330;color:#fff;border-color:#1d2330}
tr.diverged{background:#fff8e6}tr.regression{background:#fdecea}tr.improvement{background:#eef8f0}
tr.only_a,tr.only_b,tr.different{background:#fdecea}
td.num{text-align:right;white-space:nowrap}
.detail{display:none}tr.open .detail{display:block}
"""

JS = """
(function(){
  var table=document.getElementById("steps");
  if(!table)return;
  var buttons=document.querySelectorAll(".filters [data-filter]");
  function apply(filter){
    buttons.forEach(function(b){b.classList.toggle("active",b.getAttribute("data-filter")===filter);});
    table.querySelectorAll("tr.step").forEach(function(row){
      row.hidden=!(filter==="all"||row.classList.contains(filter));
    });
  }
  buttons.forEach(function(btn){
    btn.addEventListener("click",function(){apply(btn.getAttribute("data-filter"));});
  });
  table.querySelectorAll("tr.step[data-detail]").forEach(function(row){
    row.addEventListener("click",function(){row.classList.toggle("open");});
  });
})();
"""


def _e(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def _short(value: Any, width: int = 120) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _cls(word: Any) -> str:
    return f'<span class="{_e(word)}">{_e(word)}</span>'


def _fmt_cost(value: float | None) -> str:
    if value is None:
        return "?"
    return "$0" if value == 0 else f"${value:.4f}" if value >= 0.0001 else f"${value:.6f}"


def _fmt_dur_ms(value: float | None) -> str:
    if value is None:
        return "-"
    if abs(value) < 1000:
        return f"{value:.0f}ms"
    return f"{value / 1000:.2f}s"


def _fmt_signed(value: float | None, *, kind: str = "num") -> str:
    if value is None or value == 0:
        return ""
    sign = "+" if value > 0 else ""
    if kind == "cost":
        return f"{sign}{_fmt_cost(value)}"
    if kind == "ms":
        return f"{sign}{_fmt_dur_ms(value)}"
    if kind == "s":
        return f"{sign}{value:.2f}s"
    if float(value).is_integer():
        return f"{sign}{int(value)}"
    return f"{sign}{value:g}"


def _run_tokens(run: Run) -> int:
    u = run.usage
    return u.total_tokens or (u.input_tokens + u.output_tokens)


def _run_duration_ms(run: Run) -> float | None:
    if run.usage.wall_time_ms is not None:
        return run.usage.wall_time_ms
    if run.steps:
        return sum(s.duration_ms for s in run.steps)
    return None


def _step_cost(step: Step | None) -> float | None:
    if step is None:
        return None
    stored = (step.attributes or {}).get("crashlab.usage.cost_usd")
    if stored is not None:
        try:
            return float(stored)
        except (TypeError, ValueError):
            return None
    return step_cost(step)


def _finding_id(finding: Finding) -> tuple[str, str]:
    return (finding.kind, finding.key or finding.message)


def _bad_findings(run: Run) -> list[Finding]:
    return [f for f in run.findings if f.severity in _BAD_FINDING]


def regressions(diff_result: RunDiff, run_a: Run, run_b: Run) -> list[str]:
    """Human-readable reasons Run B regressed versus Run A."""
    items: list[str] = []
    if run_a.outcome != "fail" and run_b.outcome == "fail":
        items.append(f"outcome {run_a.outcome} → fail")
    if run_a.status != "error" and run_b.status == "error":
        items.append(f"status {run_a.status} → error")
    for check in diff_result.checks:
        a_st = check.a.status if check.a else None
        b_st = check.b.status if check.b else None
        if a_st == "pass" and b_st in _CHECK_REGRESS:
            items.append(f"check {check.name} pass → {b_st}")
    seen = {_finding_id(f) for f in _bad_findings(run_a)}
    for finding in _bad_findings(run_b):
        if _finding_id(finding) not in seen:
            items.append(f"new {finding.severity} finding: {finding.kind}")
    return items


def improvements(diff_result: RunDiff, run_a: Run, run_b: Run) -> list[str]:
    """Human-readable reasons Run B improved versus Run A."""
    items: list[str] = []
    if run_a.outcome == "fail" and run_b.outcome != "fail":
        items.append(f"outcome fail → {run_b.outcome}")
    if run_a.status == "error" and run_b.status != "error":
        items.append(f"status error → {run_b.status}")
    for check in diff_result.checks:
        a_st = check.a.status if check.a else None
        b_st = check.b.status if check.b else None
        if a_st in _CHECK_REGRESS and b_st == "pass":
            items.append(f"check {check.name} {a_st} → pass")
    seen = {_finding_id(f) for f in _bad_findings(run_b)}
    for finding in _bad_findings(run_a):
        if _finding_id(finding) not in seen:
            items.append(f"resolved {finding.severity} finding: {finding.kind}")
    return items


def has_regression(diff_result: RunDiff) -> bool:
    """True if Run B regressed versus Run A.

    A regression is a status/outcome that became a failure, a new error/warn
    (or critical) finding, or a check that flipped from pass to fail or error.
    """
    return bool(regressions(diff_result, diff_result.a, diff_result.b))


def _md_cell(value: Any) -> str:
    text = "-" if value is None or value == "" else str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip() or "-"


def _metric_rows(diff_result: RunDiff, run_a: Run, run_b: Run) -> list[tuple[str, str, str, str]]:
    ca, cb = run_cost(run_a), run_cost(run_b)
    ta, tb = _run_tokens(run_a), _run_tokens(run_b)
    da, db = _run_duration_ms(run_a), _run_duration_ms(run_b)
    fa, fb = len(run_a.findings), len(run_b.findings)
    cost_delta = "" if ca is None or cb is None else _fmt_signed(cb - ca, kind="cost")
    dur_delta = "" if da is None or db is None else _fmt_signed(db - da, kind="ms")
    checks_a = ", ".join(f"{c.name} {c.a.status}" for c in diff_result.checks if c.a) or "—"
    checks_b = ", ".join(f"{c.name} {c.b.status}" for c in diff_result.checks if c.b) or "—"
    check_delta = ", ".join(
        f"{c.name} {(c.a.status if c.a else '—')} → {(c.b.status if c.b else '—')}"
        for c in diff_result.checks
        if c.changed
    )
    return [
        (
            "Outcome",
            run_a.outcome,
            run_b.outcome,
            "changed" if run_a.outcome != run_b.outcome else "",
        ),
        ("Status", run_a.status, run_b.status, "changed" if run_a.status != run_b.status else ""),
        ("Checks", checks_a, checks_b, check_delta),
        ("Findings", str(fa), str(fb), _fmt_signed(fb - fa)),
        ("Cost", _fmt_cost(ca), _fmt_cost(cb), cost_delta),
        ("Tokens", str(ta), str(tb), _fmt_signed(tb - ta)),
        ("Latency", _fmt_dur_ms(da), _fmt_dur_ms(db), dur_delta),
        (
            "Trajectory edit distance",
            str(diff_result.tool_edit_distance),
            f"similarity {diff_result.trajectory_similarity:.2f}",
            "",
        ),
    ]


def _step_label(step: Step | None) -> str:
    if step is None:
        return "—"
    bits = [step.operation, step.name]
    if step.operation == "execute_tool":
        args = step.request.get("args") if step.request else None
        if args:
            bits.append(_short(args, 80))
    elif step.operation == "chat":
        out = None
        if step.response:
            msgs = step.response.get("messages") or []
            if msgs:
                out = msgs
        bits.append(_short(out if out is not None else (step.response or ""), 80))
    if step.error:
        bits.append(f"error {step.error.type}")
    return " · ".join(str(b) for b in bits if b)


def _step_findings(run: Run, idx: int) -> list[Finding]:
    out = []
    for finding in run.findings:
        steps = set(finding.steps)
        if finding.step is not None:
            steps.add(finding.step)
        if idx in steps:
            out.append(finding)
    return out


def _step_delta(a: Step | None, b: Step | None) -> tuple[str, str]:
    ca, cb = _step_cost(a), _step_cost(b)
    if ca is None and cb is None:
        cost = "—"
    elif ca is None or cb is None:
        cost = f"{_fmt_cost(ca)} → {_fmt_cost(cb)}"
    else:
        cost = _fmt_signed(cb - ca, kind="cost") or "0"
    da = a.duration_ms if a is not None else None
    db = b.duration_ms if b is not None else None
    if da is None and db is None:
        latency = "—"
    elif da is None or db is None:
        latency = f"{_fmt_dur_ms(da)} → {_fmt_dur_ms(db)}"
    else:
        latency = _fmt_signed(db - da, kind="ms") or "0"
    return cost, latency


def _tool_delta(a: Step | None, b: Step | None) -> str:
    def tool_name(step: Step | None) -> str | None:
        if step is None or step.operation != "execute_tool":
            return None
        return step.name

    na, nb = tool_name(a), tool_name(b)
    if na is None and nb is None:
        return ""
    if na == nb:
        aa = (a.request or {}).get("args") if a is not None else None
        bb = (b.request or {}).get("args") if b is not None else None
        if aa == bb:
            return na or ""
        return f"{na} args changed"
    if na and not nb:
        return f"only A: {na}"
    if nb and not na:
        return f"only B: {nb}"
    return f"{na} → {nb}"


def _new_resolved(run_a: Run, run_b: Run, idx: int) -> tuple[list[str], list[str]]:
    def keys(run: Run) -> dict[tuple[str, str], Finding]:
        return {_finding_id(f): f for f in _step_findings(run, idx)}

    ka, kb = keys(run_a), keys(run_b)
    new = [f"{f.severity} {f.kind}" for k, f in kb.items() if k not in ka]
    resolved = [f"{f.severity} {f.kind}" for k, f in ka.items() if k not in kb]
    return new, resolved


def render_diff_html(diff_result: RunDiff, run_a: Run, run_b: Run) -> str:
    """Self-contained HTML/CSS/vanilla JS side-by-side comparison of two runs."""
    regs = regressions(diff_result, run_a, run_b)
    imps = improvements(diff_result, run_a, run_b)
    parts: list[str] = []
    parts.append("<h1>crashlab diff</h1>")
    rel = f'<p class="muted">{_e(diff_result.relation)}</p>' if diff_result.relation else ""
    parts.append(
        f"<p>A <code>{_e(run_a.run_id)}</code> ({_e(run_a.name)}, {_cls(run_a.status)}, "
        f"{_cls(run_a.outcome)}) vs B <code>{_e(run_b.run_id)}</code> "
        f"({_e(run_b.name)}, {_cls(run_b.status)}, {_cls(run_b.outcome)})</p>{rel}"
    )
    ca, cb = run_cost(run_a), run_cost(run_b)
    ta, tb = _run_tokens(run_a), _run_tokens(run_b)
    da, db = _run_duration_ms(run_a), _run_duration_ms(run_b)
    cards = [
        ("Status", f"{_cls(run_a.status)} → {_cls(run_b.status)}", ""),
        (
            "Duration",
            f"{_e(_fmt_dur_ms(da))} → {_e(_fmt_dur_ms(db))}",
            _fmt_signed(None if da is None or db is None else db - da, kind="ms"),
        ),  # fmt: skip
        (
            "Cost",
            f"{_e(_fmt_cost(ca))} → {_e(_fmt_cost(cb))}",
            _fmt_signed(None if ca is None or cb is None else cb - ca, kind="cost"),
        ),  # fmt: skip
        ("Tokens", f"{ta} → {tb}", _fmt_signed(tb - ta)),
        (
            "Findings",
            f"{len(run_a.findings)} → {len(run_b.findings)}",
            _fmt_signed(len(run_b.findings) - len(run_a.findings)),
        ),  # fmt: skip
        (
            "Trajectory",
            f"edit distance {diff_result.tool_edit_distance}",
            f"similarity {diff_result.trajectory_similarity:.2f}",
        ),
    ]
    card_html = []
    for label, value, sub in cards:
        card_html.append(
            f'<div class="card"><div class="lbl">{_e(label)}</div>'
            f'<div class="val">{value}</div>'
            f'<div class="sub">{_e(sub)}</div></div>'
        )
    parts.append('<div class="cards">' + "".join(card_html) + "</div>")

    parts.append("<h2>Regressions and improvements</h2>")
    pills: list[str] = []
    if not regs and not imps:
        pills.append('<span class="pill">No regressions or improvements</span>')
    for item in regs:
        pills.append(f'<span class="pill regression">{_e(item)}</span>')
    for item in imps:
        pills.append(f'<span class="pill improvement">{_e(item)}</span>')
    parts.append('<div class="pills">' + "".join(pills) + "</div>")

    if diff_result.checks:
        parts.append("<h2>Checks</h2>")
        rows = []
        for check in diff_result.checks:
            a_st = check.a.status if check.a else "—"
            b_st = check.b.status if check.b else "—"
            a_msg = check.a.message if check.a else ""
            b_msg = check.b.message if check.b else ""
            rows.append(
                "<tr>"
                f"<td>{_e(check.name)}</td>"
                f"<td>{_cls(a_st)}"
                f"{' · ' + _e(_short(a_msg, 80)) if a_msg else ''}</td>"
                f"<td>{_cls(b_st)}"
                f"{' · ' + _e(_short(b_msg, 80)) if b_msg else ''}</td>"
                f"<td>{_e('changed' if check.changed else '')}</td>"
                "</tr>"
            )
        parts.append(
            "<table><tr><th>check</th><th>A</th><th>B</th><th>Δ</th></tr>"
            + "".join(rows)
            + "</table>"
        )

    if diff_result.findings:
        parts.append("<h2>Findings</h2>")
        rows = []
        for fd in diff_result.findings:

            def side(fs: list[Finding], worst: str | None) -> str:
                if not fs:
                    return "—"
                msg = _short(fs[0].message, 80)
                return f"{len(fs)} {_cls(worst)} · {_e(msg)}"

            rows.append(
                "<tr>"
                f"<td>{_e(fd.kind)}</td>"
                f"<td>{side(fd.a, fd.worst_a)}</td>"
                f"<td>{side(fd.b, fd.worst_b)}</td>"
                f"<td>{_e('changed' if fd.changed else '')}</td>"
                "</tr>"
            )
        parts.append(
            "<table><tr><th>kind</th><th>A</th><th>B</th><th>Δ</th></tr>"
            + "".join(rows)
            + "</table>"
        )

    parts.append("<h2>Steps</h2>")
    parts.append(
        '<div class="filters">'
        '<button type="button" class="active" data-filter="all">All</button>'
        '<button type="button" data-filter="diverged">Diverged</button>'
        '<button type="button" data-filter="regression">Regressions</button>'
        '<button type="button" data-filter="improvement">Improvements</button>'
        "</div>"
    )
    body = []
    for sd in diff_result.steps:
        new, resolved = _new_resolved(run_a, run_b, sd.idx)
        cost, latency = _step_delta(sd.a, sd.b)
        tools = _tool_delta(sd.a, sd.b)
        classes = ["step", sd.status]
        if not sd.same:
            classes.append("diverged")
        b_err = sd.b is not None and sd.b.error and (sd.a is None or not sd.a.error)
        a_err = sd.a is not None and sd.a.error and (sd.b is None or not sd.b.error)
        if new or b_err:
            classes.append("regression")
        if resolved or a_err:
            classes.append("improvement")
        label = _STEP_LABEL[sd.status]
        if sd.fields and sd.status in ("request", "response"):
            label += " (" + ", ".join(sd.fields[:3]) + ")"
        finding_bits = []
        if new:
            finding_bits.append("new: " + ", ".join(new))
        if resolved:
            finding_bits.append("resolved: " + ", ".join(resolved))
        detail = sd.detail()
        detail_html = f'<pre class="detail">{_e(detail)}</pre>' if detail else ""
        body.append(
            f'<tr class="{" ".join(_e(c) for c in classes)}"'
            + (' data-detail="1"' if detail else "")
            + ">"
            f'<td class="num">{sd.idx}{detail_html}</td>'
            f"<td>{_e(_step_label(sd.a))}</td>"
            f"<td>{_e(_step_label(sd.b))}</td>"
            f"<td>{_e(label)}</td>"
            f"<td>{_e(' · '.join(finding_bits))}</td>"
            f"<td>{_e(tools)}</td>"
            f'<td class="num">{_e(cost)}</td>'
            f'<td class="num">{_e(latency)}</td>'
            "</tr>"
        )
    parts.append(
        '<table id="steps"><tr>'
        "<th>#</th><th>A</th><th>B</th><th>status</th>"
        "<th>findings</th><th>tools</th><th>cost Δ</th><th>latency Δ</th>"
        "</tr>" + "".join(body) + "</table>"
    )
    parts.append(f'<p class="muted">{_e(diff_result.summary())}</p>')
    title = f"crashlab diff · {run_a.run_id[:10]} vs {run_b.run_id[:10]}"
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_e(title)}</title><style>{CSS}</style></head><body>"
        + "\n".join(parts)
        + f"<script>{JS}</script></body></html>\n"
    )


def render_diff_markdown(diff_result: RunDiff, run_a: Run, run_b: Run) -> str:
    """GitHub-flavored Markdown summary table, ready for a PR comment."""
    lines = [
        f"# crashlab diff: `{_md_cell(run_a.run_id)}` vs `{_md_cell(run_b.run_id)}`",
        "",
    ]
    if run_a.name or run_b.name:
        lines.append(f"- **A:** {_md_cell(run_a.name)} (`{run_a.status}`, `{run_a.outcome}`)")
        lines.append(f"- **B:** {_md_cell(run_b.name)} (`{run_b.status}`, `{run_b.outcome}`)")
        lines.append("")
    if diff_result.relation:
        lines.append(f"_{_md_cell(diff_result.relation)}_")
        lines.append("")
    lines += [
        "| Metric | A | B | Δ |",
        "| --- | --- | --- | --- |",
    ]
    for label, va, vb, delta in _metric_rows(diff_result, run_a, run_b):
        lines.append(f"| {_md_cell(label)} | {_md_cell(va)} | {_md_cell(vb)} | {_md_cell(delta)} |")
    lines.append("")
    lines.append(f"Trajectory edit distance: **{diff_result.tool_edit_distance}**; "
                 f"similarity **{diff_result.trajectory_similarity:.2f}**.")  # fmt: skip
    lines.append("")
    regs = regressions(diff_result, run_a, run_b)
    imps = improvements(diff_result, run_a, run_b)
    lines.append("## Regressions")
    if regs:
        lines.extend(f"- {_md_cell(item)}" for item in regs)
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Improvements")
    if imps:
        lines.extend(f"- {_md_cell(item)}" for item in imps)
    else:
        lines.append("- None")
    lines.append("")
    lines.append(diff_result.summary())
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "has_regression",
    "improvements",
    "regressions",
    "render_diff_html",
    "render_diff_markdown",
]
