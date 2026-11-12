"""Search the local run store: ``crashlab search`` and :func:`search_runs`.

Plain, offline substring (or regex) search over what was recorded: run ids, agent
names, input / output, run and step errors, chat messages, tool names / arguments /
results, model names, check names and messages, runtime findings and diagnosis labels.
Filters (``tool``, ``model``, ``check``, ``label``, ``errors``, ``since``, run id
prefix ...) narrow the runs first; the query then finds hits inside them.

Chat requests repeat the whole conversation on every call, so a message is reported
once per run, at the first step that sent it. The system prompt is reported once per
run as well.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from ._runscan import ScanStats, iter_runs
from .models import Run, Step
from .store import Store

__all__ = ["SearchHit", "RunMatch", "SearchResult", "search_runs", "run_has_errors"]

SNIPPET_WIDTH = 100

#: fields a query can be limited to with ``fields=`` / ``--in``
FIELDS = (
    "run",  # run id, agent name, tags, input, output, outcome reason
    "error",  # run error and step errors
    "message",  # chat messages (system prompt, user / assistant / tool turns)
    "tool",  # tool names, arguments and results
    "model",  # requested / responded model names
    "check",  # @crashlab.check names and messages
    "finding",  # runtime-check findings
    "diagnosis",  # diagnosis labels and root cause
)


@dataclass
class SearchHit:
    """One place the query matched. ``step`` is None for run-level fields."""

    field: str  # e.g. "run.error", "step.error", "message.user", "tool.result", "check"
    step: int | None
    snippet: str


@dataclass
class RunMatch:
    run_id: str
    name: str
    status: str
    outcome: str
    started_at: str
    hits: list[SearchHit] = field(default_factory=list)
    hits_truncated: int = 0  # hits found beyond ``max_hits_per_run``

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    query: str | None
    matches: list[RunMatch]
    scanned: int
    unreadable: list[str]
    truncated: bool  # stopped at ``limit`` matching runs; older runs were not scanned

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "scanned": self.scanned,
            "unreadable": self.unreadable,
            "truncated": self.truncated,
            "matches": [m.to_dict() for m in self.matches],
        }


# --- text extraction ------------------------------------------------------------------


def _json(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _part_text(part: Any) -> str:
    if not isinstance(part, dict):
        return _json(part)
    kind = part.get("type")
    if kind == "text" or "content" in part and kind not in ("tool_call", "tool_call_response"):
        return _json(part.get("content", ""))
    if kind == "tool_call":
        return f"{part.get('name', '')}({_json(part.get('arguments', {}))})"
    if kind == "tool_call_response":
        return _json(part.get("response", ""))
    return _json({k: v for k, v in part.items() if k != "type"})


def _message_text(msg: Any) -> tuple[str, str]:
    """(role, text) of a normalized message (``{"role", "parts"}``) or a provider dict."""
    if not isinstance(msg, dict):
        return "message", _json(msg)
    role = str(msg.get("role") or "message")
    parts = msg.get("parts")
    if isinstance(parts, list):
        return role, "\n".join(_part_text(p) for p in parts)
    content = msg.get("content")
    if isinstance(content, list):
        return role, "\n".join(_part_text(p) for p in content)
    return role, _json(content if content is not None else msg)


def _flat_key(text: str) -> str:
    return "flat:" + hashlib.sha1(" ".join(text.split()).encode()).hexdigest()


def _models(step: Step) -> list[str]:
    attrs = step.attributes or {}
    out = []
    for key in ("gen_ai.request.model", "gen_ai.response.model"):
        v = attrs.get(key)
        if v and str(v) not in out:
            out.append(str(v))
    if not out and step.operation in ("chat", "embeddings") and step.name:
        out.append(step.name)
    return out


def _fields(run: Run) -> Iterator[tuple[str, str, int | None, str]]:
    """(group, field, step, text) for everything searchable in ``run``."""
    yield "run", "run.id", None, run.run_id
    yield "run", "run.name", None, run.name
    for tag in run.tags:
        yield "run", "run.tag", None, tag
    if run.input:
        yield "run", "run.input", None, _json(run.input)
    if run.output is not None:
        yield "run", "run.output", None, _json(run.output)
    if run.outcome_reason:
        yield "run", "run.outcome_reason", None, run.outcome_reason
    if run.error is not None:
        yield "error", "run.error", None, f"{run.error.type}: {run.error.detail}"
    for c in run.checks:
        yield (
            "check",
            "check",
            None,
            f"{c.name} [{c.status}]" + (f": {c.message}" if c.message else ""),
        )
    for f in run.findings:
        yield "finding", "finding", f.step, f"{f.kind} [{f.severity}]: {f.message}"
    d = run.diagnosis
    if d is not None:
        labels = [d.label.value] + [s.value for s in d.secondary]
        yield "diagnosis", "diagnosis.label", d.failing_step, " ".join(labels)
        if d.root_cause:
            yield "diagnosis", "diagnosis.root_cause", d.failing_step, d.root_cause

    seen: set[str] = set()  # message digests already reported for this run
    seen_models: set[str] = set()
    for s in run.steps:
        for m in _models(s):
            if m not in seen_models:
                seen_models.add(m)
                yield "model", "model", s.idx, m
        if s.error is not None:
            yield "error", "step.error", s.idx, f"{s.error.type}: {s.error.detail}"
        if s.operation == "chat":
            req = s.request or {}
            system = req.get("system")
            if system:
                key = "system:" + hashlib.sha1(_json(system).encode()).hexdigest()
                if key not in seen:
                    seen.add(key)
                    yield "message", "message.system", s.idx, _json(system)
            for msg in req.get("messages") or []:
                role, text = _message_text(msg)
                key = hashlib.sha1(f"{role}\0{text}".encode()).hexdigest()
                if key in seen or _flat_key(text) in seen:
                    continue  # already sent, or a tool result this run already reported
                seen.add(key)
                yield "message", f"message.{role}", s.idx, text
            resp = s.response or {}
            for msg in resp.get("messages") or []:
                role, text = _message_text(msg)
                key = hashlib.sha1(f"{role}\0{text}".encode()).hexdigest()
                seen.add(key)  # the next request repeats it: report it here only
                yield "message", "response", s.idx, text
            # normalized responses carry the tool calls in messages too; older ones may not
            for tc in [] if resp.get("messages") else resp.get("tool_calls") or []:
                text = (
                    _part_text({"type": "tool_call", **tc}) if isinstance(tc, dict) else _json(tc)
                )
                yield "message", "response.tool_call", s.idx, text
        elif s.operation in ("execute_tool", "invoke_agent"):
            yield "tool", "tool.name", s.idx, s.name
            if s.args:
                yield "tool", "tool.args", s.idx, _json(s.args)
            if s.response is not None and s.result is not None:
                text = _json(s.result)
                seen.add(_flat_key(text))  # the next chat request echoes it back
                yield "tool", "tool.result", s.idx, text
        else:
            yield "run", "step.name", s.idx, s.name


def _snippet(text: str, start: int, end: int, width: int = SNIPPET_WIDTH) -> str:
    flat_pre = " ".join(text[:start].split())
    flat_hit = " ".join(text[start:end].split())
    flat_post = " ".join(text[end:].split())
    room = max(width - len(flat_hit), 10)
    before = flat_pre[-(room // 2) :] if flat_pre else ""
    after = flat_post[: room - len(before)] if flat_post else ""
    pre = ("…" if len(before) < len(flat_pre) else "") + before
    post = after + ("…" if len(after) < len(flat_post) else "")
    sep_pre = " " if text[:start] and text[start - 1].isspace() and pre else ""
    sep_post = " " if text[end:] and text[end].isspace() and post else ""
    return f"{pre}{sep_pre}{flat_hit}{sep_post}{post}"


# --- run filters ----------------------------------------------------------------------


def run_has_errors(run: Run) -> bool:
    """The run raised, ended with status error, or any step recorded an error."""
    return run.status == "error" or run.error is not None or any(s.error for s in run.steps)


def _failing_checks(run: Run) -> list[str]:
    return [c.name for c in run.checks if c.status != "pass"]


def _passes_filters(
    run: Run,
    *,
    tool: str | None,
    model: str | None,
    check: str | None,
    label: str | None,
    errors: bool,
    status: str | None,
    outcome: str | None,
) -> bool:
    if status and run.status != status:
        return False
    if outcome and run.outcome != outcome:
        return False
    if errors and not run_has_errors(run):
        return False
    if tool and not any(
        s.name == tool for s in run.steps if s.operation in ("execute_tool", "invoke_agent")
    ):
        return False
    if model:
        want = model.lower()
        if not any(want in m.lower() for s in run.steps for m in _models(s)):
            return False
    if check and check not in _failing_checks(run):
        return False
    if label:
        d = run.diagnosis
        if d is None:
            return False
        labels = {d.label.value, *(s.value for s in d.secondary)}
        if label.upper() not in labels:
            return False
    return True


# --- search ---------------------------------------------------------------------------


def _compile(query: str, *, regex: bool, case_sensitive: bool) -> re.Pattern[str]:
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        return re.compile(query if regex else re.escape(query), flags)
    except re.error as exc:
        raise ValueError(f"bad regular expression {query!r}: {exc}") from None


def search_runs(
    query: str | None = None,
    *,
    store: Store | None = None,
    store_dir: str | None = None,
    tool: str | None = None,
    model: str | None = None,
    check: str | None = None,
    label: str | None = None,
    errors: bool = False,
    status: str | None = None,
    outcome: str | None = None,
    name: str | None = None,
    since: datetime | None = None,
    run_prefix: str | None = None,
    fields: list[str] | tuple[str, ...] | None = None,
    regex: bool = False,
    case_sensitive: bool = False,
    limit: int | None = 50,
    max_hits_per_run: int = 20,
) -> SearchResult:
    """Search stored runs, newest first.

    Filters are ANDed and select runs:

    - ``tool``: the run executed a tool with exactly this name
    - ``model``: a chat / embeddings step requested or got a model whose name contains this
      (case-insensitive)
    - ``check``: an ``@crashlab.check`` with exactly this name did not pass (fail / error)
    - ``label``: the stored diagnosis has this primary or secondary taxonomy label
    - ``errors``: the run raised, ended with status ``error`` or any step recorded an error
    - ``status`` / ``outcome`` / ``name``: exact run status, outcome, agent name
    - ``since``: started at or after this time; ``run_prefix``: run id starts with this

    With a ``query``, only runs with at least one hit are returned, each with its hits
    (case-insensitive substring unless ``case_sensitive`` / ``regex``). ``fields`` limits
    where the query looks (see :data:`FIELDS`). Without a query, every run passing the
    filters is returned with no hits.

    ``limit`` stops after that many matching runs (None = no limit); ``truncated`` then
    says older runs were not scanned.
    """
    if fields:
        bad = [f for f in fields if f not in FIELDS]
        if bad:
            raise ValueError(f"unknown search field(s) {bad}; choose from {', '.join(FIELDS)}")
    pattern = _compile(query, regex=regex, case_sensitive=case_sensitive) if query else None
    store = store or Store(store_dir)
    stats = ScanStats()
    matches: list[RunMatch] = []
    truncated = False
    for run in iter_runs(store, since=since, run_prefix=run_prefix, name=name, stats=stats):
        if limit is not None and len(matches) >= limit:
            truncated = True
            stats.scanned -= 1  # this one was read only to learn there are more
            break
        if not _passes_filters(
            run, tool=tool, model=model, check=check, label=label, errors=errors,
            status=status, outcome=outcome,
        ):  # fmt: skip
            continue
        match = RunMatch(
            run_id=run.run_id,
            name=run.name,
            status=run.status,
            outcome=run.outcome,
            started_at=run.started_at.isoformat(),
        )
        if pattern is not None:
            for group, fname, step, text in _fields(run):
                if fields and group not in fields:
                    continue
                m = pattern.search(text)
                if m is None or m.end() == m.start():
                    continue
                if len(match.hits) >= max_hits_per_run:
                    match.hits_truncated += 1
                    continue
                match.hits.append(SearchHit(fname, step, _snippet(text, m.start(), m.end())))
            if not match.hits and not match.hits_truncated:
                continue
        matches.append(match)
    return SearchResult(
        query=query,
        matches=matches,
        scanned=stats.scanned,
        unreadable=stats.unreadable,
        truncated=truncated,
    )
