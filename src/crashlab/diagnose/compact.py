"""Compact a recorded run into the numbered step table the diagnoser reads.

Keeps the task input, outcome checks, runtime findings, the system prompt, the tool
schemas and, per step, only what is new: a chat step shows the messages added since the
agent's previous turn, its reasoning, reply and tool calls; a tool step its arguments,
result or error. Long fields are cut with an explicit marker."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from ..models import Run, Step
from .fields import as_text, message_text, reply_parts, tool_calls
from .heuristics import Candidate


def cut(text: Any, limit: int, what: str) -> str:
    text = text if isinstance(text, str) else as_text(text)
    if limit <= 0 or len(text) <= limit:
        return text
    return (
        f"{text[:limit]} …[truncated: {what} is {len(text):,} chars; "
        "only the part above can be quoted]"
    )


def _tool_sig(tool: dict[str, Any]) -> str:
    params = tool.get("parameters") or tool.get("input_schema") or {}
    props = params.get("properties") or {}
    req = set(params.get("required") or [])
    args = ", ".join(
        f"{k}{'' if k in req else '?'}: {v.get('type', 'any')}"
        for k, v in props.items()
        if isinstance(v, dict)
    )
    desc = " ".join(str(tool.get("description") or "").split())
    return f"- {tool.get('name')}({args})" + (f": {desc}" if desc else "")


def _new_messages(msgs: list[dict[str, Any]], first: bool) -> list[dict[str, Any]]:
    """Messages since the agent's previous turn (all of them on the first call)."""
    if first:
        return msgs
    assistant = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
    return msgs[(assistant[-1] + 1 if assistant else 0) :]


def _findings_by_step(run: Run) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for f in run.findings:
        if f.kind == "fork":
            continue
        for i in f.steps or ([f.step] if f.step is not None else []):
            out.setdefault(i, []).append(f"{f.kind} ({f.severity})")
    return out


def compact(
    run: Run,
    *,
    candidates: Iterable[Candidate] = (),
    max_field: int = 600,
    max_system: int = 6000,
    hide_input: Iterable[str] = (),
) -> str:
    """The run as the diagnoser sees it. ``hide_input`` drops run input keys (e.g. fault
    switches that would give the answer away)."""
    hidden = set(hide_input)
    inp = {k: v for k, v in (run.input or {}).items() if k not in hidden and v is not None}
    lines = [
        f"RUN {run.run_id} (agent: {run.name})",
        "TASK INPUT: " + cut(json.dumps(inp, ensure_ascii=False, default=str), 2000, "input"),
    ]
    status = f"status={run.status} outcome={run.outcome}"
    if run.error is not None:
        status += f" error={run.error.type}: {cut(run.error.detail, 300, 'the error')}"
    lines.append("RESULT: " + status)
    if run.output is not None:
        lines.append("FINAL OUTPUT: " + cut(as_text(run.output), max_field, "the output"))

    lines += ["", "CHECKS (@crashlab.check outcome oracles):"]
    lines += [
        f"- {c.status.upper()} {c.name}" + (f": {c.message}" if c.message else "")
        for c in run.checks
    ] or ["- none"]
    lines.append("RUNTIME FINDINGS (built-in detectors):")
    real = [f for f in run.findings if f.kind != "fork"]
    for f in real:
        at = ",".join(map(str, f.steps or ([f.step] if f.step is not None else []))) or "-"
        lines.append(f"- [{f.severity}] {f.kind} at step {at}: {f.message}")
    if not real:
        lines.append("- none")
    cands = list(candidates)
    lines.append("CANDIDATE STEPS (heuristic hints, may be wrong):")
    lines += [f"- step {c.step}: {c.reason}" for c in cands] or ["- none"]

    chats = [s for s in run.steps if s.operation == "chat"]
    system = (chats[0].request or {}).get("system") if chats else None
    lines += ["", "SYSTEM PROMPT (as sent at the first LLM call):"]
    lines.append(cut(as_text(system), max_system, "the system prompt") if system else "(none)")
    tools = (chats[0].request or {}).get("tools") if chats else None
    lines.append("TOOLS:")
    lines += [_tool_sig(t) for t in tools or [] if isinstance(t, dict)] or ["(none)"]

    lines += ["", "STEPS:"]
    has_tool_steps = any(s.operation == "execute_tool" for s in run.steps)
    by_step = _findings_by_step(run)
    cand_steps = {c.step for c in cands}
    first_chat = True
    for s in run.steps:
        lines += _step_lines(s, first_chat, system, has_tool_steps, max_field)
        if s.operation == "chat":
            first_chat = False
        if s.idx in by_step:
            lines.append("    findings: " + ", ".join(by_step[s.idx]))
        if s.idx in cand_steps:
            lines.append("    (candidate step)")
    return "\n".join(lines)


def _step_lines(s: Step, first: bool, system: Any, has_tool_steps: bool, lim: int) -> list[str]:
    tag = "" if s.source in ("live", "cassette") else f"  [source: {s.source}]"
    if s.operation != "chat":
        out = [f"[{s.idx}] tool {s.name}{tag}", "    args: " + cut(as_text(s.args), lim, "args")]
        if s.error is None and s.response is not None:
            out.append("    result: " + cut(as_text(s.result), lim, f"the step {s.idx} result"))
    else:
        fin = (s.response or {}).get("finish_reason")
        head = f"[{s.idx}] chat (LLM call, {s.name})"
        out = [head + (f" finish_reason={fin}" if fin else "") + tag]
        req = s.request or {}
        if not first and req.get("system") and req.get("system") != system:
            out.append("    system prompt changed: " + cut(as_text(req["system"]), lim, "it"))
        msgs = [m for m in req.get("messages") or [] if isinstance(m, dict)]
        skipped = 0
        for m in _new_messages(msgs, first):
            role = m.get("role")
            if role == "tool" and has_tool_steps:
                skipped += 1
                continue
            out.append(f"    new {role} message: " + cut(message_text(m), lim, "the message"))
        if skipped:
            out.append(f"    (+{skipped} tool result message(s): shown at the tool steps)")
        out += [
            "    reasoning: " + cut(r, lim, "the reasoning") for r in reply_parts(s, "reasoning")
        ]
        out += ["    reply: " + cut(t, lim, "the reply") for t in reply_parts(s, "text")]
        for c in tool_calls(s):
            args = cut(as_text(c.get("arguments")), lim, "the arguments")
            out.append(f"    calls tool: {c.get('name')}({args})")
    if s.error is not None:
        out.append(f"    error: {s.error.type}: " + cut(s.error.detail, lim, "the error"))
    return out
