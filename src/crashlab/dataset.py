"""Portable golden datasets: turn recorded runs into shareable JSONL test cases."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import Run
from .store import Store
from .store.redact import REDACTED, get_redactor

# Same shape as crashlab.cluster, applied only to a dataset export (not the store).
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

CASE_FIELDS = (
    "id",
    "name",
    "input",
    "system_prompt",
    "expected_tools",
    "expected_outcome",
    "reference_reply",
    "tags",
    "metadata",
)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "text" in value:
            return _text(value.get("text"))
        if "content" in value:
            return _text(value.get("content"))
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(t for t in (_text(v) for v in value) if t)
    return str(value)


def system_prompt_of(run: Run) -> str | None:
    """The first distinct system / developer prompt on ``run``'s chat requests."""
    seen: list[str] = []
    for step in run.steps:
        if step.operation != "chat":
            continue
        req = step.request or {}
        text = _text(req.get("system") or req.get("instructions"))
        if not text:
            for msg in req.get("messages") or []:
                if isinstance(msg, dict) and msg.get("role") in ("system", "developer"):
                    text = _text(msg.get("content"))
                    if text:
                        break
        if text and text not in seen:
            seen.append(text)
    return seen[0] if seen else None


def tools_of(run: Run) -> list[str]:
    """Tool names the run actually called, in first-use order."""
    names: list[str] = []
    for step in run.tool_steps():
        if step.name and step.name not in names:
            names.append(step.name)
    return names


def _scrub_emails(obj: Any) -> Any:
    if isinstance(obj, str):
        return _EMAIL.sub(REDACTED, obj)
    if isinstance(obj, dict):
        return {k: _scrub_emails(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_emails(v) for v in obj]
    if isinstance(obj, tuple):
        return [_scrub_emails(v) for v in obj]
    return obj


def _redact_case(obj: Any) -> Any:
    """Apply the store redactor, then scrub emails so a shared file has no PII."""
    return _scrub_emails(get_redactor().redact(obj))


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def case_from_run(run: Run, *, redact: bool = True, include_tools: bool = True) -> dict[str, Any]:
    """One portable test case derived from a recorded run."""
    prompts = []
    first = system_prompt_of(run)
    # collect the rest for metadata when several agents have their own prompts
    if first:
        seen = [first]
        for step in run.steps:
            if step.operation != "chat":
                continue
            req = step.request or {}
            text = _text(req.get("system") or req.get("instructions"))
            if text and text not in seen:
                seen.append(text)
        prompts = seen
    case = {
        "id": run.run_id,
        "name": run.name,
        "input": run.input if isinstance(run.input, dict) else {"input": run.input},
        "system_prompt": first,
        "expected_tools": tools_of(run) if include_tools else [],
        "expected_outcome": run.outcome,
        "reference_reply": run.output,
        "tags": list(run.tags or []),
        "metadata": {
            "run_id": run.run_id,
            "entrypoint": run.entrypoint,
            "status": run.status,
            "started_at": _iso(run.started_at),
            "ended_at": _iso(run.ended_at),
            "code_version": run.code_version,
            "system_prompts": prompts[1:] if len(prompts) > 1 else [],
        },
    }
    return _redact_case(case) if redact else case


def _as_runs(
    runs: Run | str | Iterable[Run | str],
    store: Store | str | None,
) -> list[Run]:
    if isinstance(runs, (Run, str)):
        items: list[Run | str] = [runs]
    else:
        items = list(runs)
    st: Store | None = None
    if isinstance(store, Store):
        st = store
    elif store is not None:
        st = Store(store)
    out: list[Run] = []
    for item in items:
        if isinstance(item, Run):
            out.append(item)
            continue
        if st is None:
            st = Store()
        out.append(st.load_run(item))
    return out


@dataclass
class DatasetReport:
    """Result of :func:`build_dataset` / ``crashlab dataset build``."""

    path: str | None
    cases: int
    agents: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        data = {
            "path": self.path,
            "cases": self.cases,
            "agents": list(self.agents),
            "tags": list(self.tags),
            "tools": list(self.tools),
        }
        if self.path is None:
            data["records"] = list(self.records)
        return data


@dataclass
class DatasetSummary:
    """Result of :func:`inspect_dataset` / ``crashlab dataset inspect``."""

    path: str
    cases: int
    agents: dict[str, int] = field(default_factory=dict)
    tags: dict[str, int] = field(default_factory=dict)
    tools: dict[str, int] = field(default_factory=dict)
    malformed: int = 0

    @property
    def tool_coverage(self) -> dict[str, int]:
        return self.tools

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "cases": self.cases,
            "agents": dict(self.agents),
            "tags": dict(self.tags),
            "tools": dict(self.tools),
            "tool_coverage": dict(self.tools),
            "malformed": self.malformed,
        }


def _unique(values: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        if value and value not in seen:
            seen[value] = None
    return list(seen)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(dict(rec), ensure_ascii=False, default=str) + "\n")


def build_dataset(
    runs: Run | str | Iterable[Run | str],
    *,
    out_path: str | Path | None = None,
    redact: bool = True,
    include_tools: bool = True,
    store: Store | str | None = None,
) -> DatasetReport:
    """Convert recorded runs into a portable JSONL dataset of test cases.

    Each case is ``{id, name, input, system_prompt, expected_tools,
    expected_outcome, reference_reply, tags, metadata}``. When ``redact`` is
    true (the default), :mod:`crashlab.store.redact` plus an email scrubber run
    over the case before it is written, so API keys, secrets and email
    addresses are not shared.
    """
    loaded = _as_runs(runs, store)
    records = [case_from_run(run, redact=redact, include_tools=include_tools) for run in loaded]
    path: Path | None = Path(out_path) if out_path is not None else None
    if path is not None:
        _write_jsonl(path, records)
    agents = _unique(str(r.get("name") or "") for r in records)
    tags = _unique(t for r in records for t in (r.get("tags") or []) if isinstance(t, str))
    tools = _unique(
        t for r in records for t in (r.get("expected_tools") or []) if isinstance(t, str)
    )
    return DatasetReport(
        path=str(path) if path is not None else None,
        cases=len(records),
        agents=agents,
        tags=tags,
        tools=tools,
        records=records,
    )


def inspect_dataset(path: str | Path) -> DatasetSummary:
    """Count cases, agents, tags, tool coverage and malformed lines in a JSONL dataset."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"no dataset at {target}")
    agents: Counter[str] = Counter()
    tags: Counter[str] = Counter()
    tools: Counter[str] = Counter()
    cases = 0
    malformed = 0
    text = target.read_text("utf-8-sig")
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if not isinstance(rec, dict) or not rec.get("id"):
            malformed += 1
            continue
        cases += 1
        name = rec.get("name")
        if isinstance(name, str) and name:
            agents[name] += 1
        for tag in rec.get("tags") or []:
            if isinstance(tag, str) and tag:
                tags[tag] += 1
        for tool in rec.get("expected_tools") or []:
            if isinstance(tool, str) and tool:
                tools[tool] += 1
    return DatasetSummary(
        path=str(target),
        cases=cases,
        agents=dict(sorted(agents.items(), key=lambda kv: (-kv[1], kv[0]))),
        tags=dict(sorted(tags.items(), key=lambda kv: (-kv[1], kv[0]))),
        tools=dict(sorted(tools.items(), key=lambda kv: (-kv[1], kv[0]))),
        malformed=malformed,
    )


__all__ = [
    "CASE_FIELDS",
    "DatasetReport",
    "DatasetSummary",
    "build_dataset",
    "case_from_run",
    "inspect_dataset",
    "system_prompt_of",
    "tools_of",
]
