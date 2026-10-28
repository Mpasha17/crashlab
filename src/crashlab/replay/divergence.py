"""``DivergenceError``: a replayed run stopped matching its recording, with a readable diff."""

from __future__ import annotations

import difflib
import json
import re
from typing import Any, Literal

DivergenceKind = Literal[
    "request",  # same step, different request (hash mismatch)
    "step",  # a different operation / name than recorded at this index
    "extra_step",  # replay made more steps than were recorded
    "missing_steps",  # replay finished before using every recorded step
    "output",  # same steps, different final output
    "exception",  # replay raised where the recording did not (or vice versa)
    "network",  # replay tried to open a network connection
]

MAX_DIFF_LINES = 80


def _scalar(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _render(value: Any, indent: int = 0) -> list[str]:
    """YAML-like lines: multi-line strings (prompts, tool output) diff line by line."""
    pad = "  " * indent
    if isinstance(value, str) and "\n" in value:
        return [pad + line for line in value.split("\n")]
    if isinstance(value, dict) and value:
        out: list[str] = []
        for key, val in value.items():
            if isinstance(val, str) and "\n" in val:
                out.append(f"{pad}{key}: |")
                out.extend(_render(val, indent + 1))
            elif isinstance(val, (dict, list)) and val:
                out.append(f"{pad}{key}:")
                out.extend(_render(val, indent + 1))
            else:
                out.append(f"{pad}{key}: {_scalar(val)}")
        return out
    if isinstance(value, list) and value:
        out = []
        for item in value:
            if isinstance(item, (dict, list)) and item or isinstance(item, str) and "\n" in item:
                out.append(f"{pad}-")
                out.extend(_render(item, indent + 1))
            else:
                out.append(f"{pad}- {_scalar(item)}")
        return out
    return [pad + _scalar(value)]


def _dump(value: Any) -> list[str]:
    return _render(value)


def _dump_pair(a: Any, b: Any) -> tuple[list[str], list[str]]:
    """Render two values for a line diff; two strings are compared raw, line by line (a
    one-line and a multi-line prompt must not differ just by quoting)."""
    if isinstance(a, str) and isinstance(b, str):
        return a.split("\n"), b.split("\n")
    return _dump(a), _dump(b)


def _differing_paths(expected: Any, actual: Any, prefix: str, out: list[str], limit: int = 12):
    """Collect dotted paths where two JSON-like values differ (first ``limit`` only)."""
    if len(out) >= limit or expected == actual:
        return
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual), key=str):
            if expected.get(key) != actual.get(key):
                path = f"{prefix}.{key}" if prefix else str(key)
                if key not in expected or key not in actual:
                    out.append(path + (" (added)" if key not in expected else " (removed)"))
                else:
                    _differing_paths(expected[key], actual[key], path, out, limit)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        for i in range(max(len(expected), len(actual))):
            if i >= len(expected):
                out.append(f"{prefix}[{i}] (added)")
            elif i >= len(actual):
                out.append(f"{prefix}[{i}] (removed)")
            else:
                _differing_paths(expected[i], actual[i], f"{prefix}[{i}]", out, limit)
            if len(out) >= limit:
                return
        return
    out.append(prefix or "(value)")


def diff_values(
    expected: Any, actual: Any, *, fromfile: str = "recorded", tofile: str = "replay"
) -> tuple[list[str], str]:
    """Return ``(differing field paths, unified diff text)`` for two JSON-like values.

    For dicts (normalized requests) the diff is shown per differing top-level field, so a
    changed system prompt is not buried under an unchanged message list.
    """
    paths: list[str] = []
    _differing_paths(expected, actual, "", paths)
    chunks: list[str] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        keys = [k for k in sorted(set(expected) | set(actual)) if expected.get(k) != actual.get(k)]
        for key in keys:
            left, right = _dump_pair(expected.get(key), actual.get(key))
            chunks.extend(
                difflib.unified_diff(
                    left,
                    right,
                    fromfile=f"{fromfile} {key}",
                    tofile=f"{tofile} {key}",
                    lineterm="",
                    n=2,
                )
            )
    else:
        left, right = _dump_pair(expected, actual)
        chunks.extend(
            difflib.unified_diff(left, right, fromfile=fromfile, tofile=tofile, lineterm="", n=2)
        )
    if len(chunks) > MAX_DIFF_LINES:
        hidden = len(chunks) - MAX_DIFF_LINES
        chunks = chunks[:MAX_DIFF_LINES] + [f"... ({hidden} more diff lines)"]
    return paths, "\n".join(chunks)


_VOLATILE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"  # uuid
    r"|\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?"  # iso
    r"|\b1[6-9]\d{8}(?:\.\d+|\d{3})?\b"  # unix seconds / millis (2020s)
)
NONDETERMINISM_HINT = (
    "hint: the only differences look like UUIDs / timestamps made while the agent runs. "
    "Strict replay needs the same requests: create them in a @crashlab.tool (its result "
    "is served from the cassette on replay) or pass them in as agent input."
)


def nondeterminism_hint(diff: str) -> str | None:
    """The hint above when every changed diff line differs only by uuid/time values."""
    lines = diff.split("\n")
    removed = [x[1:] for x in lines if x.startswith("-") and not x.startswith("---")]
    added = [x[1:] for x in lines if x.startswith("+") and not x.startswith("+++")]
    if not removed or len(removed) != len(added):
        return None
    if not any(_VOLATILE.search(x) for x in removed):
        return None
    norm = [_VOLATILE.sub("<v>", x) for x in removed]
    return NONDETERMINISM_HINT if norm == [_VOLATILE.sub("<v>", x) for x in added] else None


class DivergenceError(Exception):
    """A strict replay did not match its recording.

    Attributes: ``kind`` (see :data:`DivergenceKind`), ``step_idx``, ``fields`` (dotted
    paths that differ), ``diff`` (unified diff, recorded vs replay), ``expected`` /
    ``actual`` (the compared values), ``source_run_id`` and ``replay_run_id``.
    """

    def __init__(
        self,
        kind: DivergenceKind,
        summary: str,
        *,
        step_idx: int | None = None,
        expected: Any = None,
        actual: Any = None,
        fields: list[str] | None = None,
        diff: str = "",
        source_run_id: str | None = None,
        replay_run_id: str | None = None,
    ) -> None:
        self.kind = kind
        self.summary = summary
        self.step_idx = step_idx
        self.expected = expected
        self.actual = actual
        self.fields = list(fields or [])
        self.diff = diff
        self.source_run_id = source_run_id
        self.replay_run_id = replay_run_id
        super().__init__(self._render())

    def _render(self) -> str:
        lines = [f"replay diverged from run {self.source_run_id or '?'}: {self.summary}"]
        if self.fields:
            lines.append("differing fields: " + ", ".join(self.fields))
        if self.diff:
            lines.append(self.diff)
            hint = nondeterminism_hint(self.diff) if self.kind == "request" else None
            if hint:
                lines.append(hint)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self._render()


class SideEffectBlockedError(RuntimeError):
    """A ``side_effects=True`` tool would have run live during a replay; it never does."""


class ReplayNetworkError(ConnectionRefusedError):
    """A replay tried to open a network connection (blocked: replays are offline)."""


class ReplayError(RuntimeError):
    """A run cannot be replayed (no entry point, missing cassette, ...)."""


class ForkError(ReplayError):
    """A run cannot be forked as asked (fork step out of range, patch does not apply ...)."""


class PatchedToolError(RuntimeError):
    """Raised by a tool whose result a :class:`~crashlab.patch.ToolResultPatch` replaced
    with an error."""
