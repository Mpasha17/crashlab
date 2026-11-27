"""Bring traces you already have into crashlab.

``from_otlp`` (OpenTelemetry GenAI, OTLP/JSON), ``from_langfuse``, ``from_openinference``
(OpenInference / Arize Phoenix) and ``from_langsmith`` turn an exported trace into
:class:`crashlab.Run` objects; :func:`save_runs` stores them, after which ``show``,
``diagnose``, ``check``, ``eval``, ``judge``, ``score``, ``diff``, ``stats`` (and
``replay`` with a stand-in entry point) work on them. ``crashlab import PATH`` does both.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..models import Run
from ._core import IngestError

__all__ = [
    "FORMATS",
    "IngestError",
    "detect_format",
    "from_langfuse",
    "from_langsmith",
    "from_openinference",
    "from_otlp",
    "load_path",
    "save_runs",
]

FORMATS = ("otel", "openinference", "langfuse", "langsmith")


def load_json_input(data: Any) -> Any:
    """``data`` as parsed JSON: JSON text / bytes are parsed, a path is read."""
    if isinstance(data, os.PathLike):
        data = Path(data).read_text("utf-8")
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8", "replace")
    if isinstance(data, str):
        try:
            return json.loads(data)
        except ValueError as exc:
            raise IngestError(f"not valid JSON: {exc}") from exc
    return data


def _first_item(data: Any) -> Any:
    if isinstance(data, dict) and isinstance(data.get("data"), list) and data["data"]:
        return data["data"][0]
    if isinstance(data, dict) and isinstance(data.get("runs"), list) and data["runs"]:
        return data["runs"][0]
    if isinstance(data, dict) and isinstance(data.get("spans"), list) and data["spans"]:
        return data["spans"][0]
    if isinstance(data, list) and data:
        return data[0]
    return data


def _has_oi_kind(data: Any) -> bool:
    from ._spans import otlp_spans

    try:
        return any("openinference.span.kind" in sp.attrs for sp in otlp_spans(data))
    except IngestError:
        return False


def detect_format(data: Any) -> str:
    """``"otel"``, ``"openinference"``, ``"langfuse"`` or ``"langsmith"`` for a parsed
    export (or JSON text); raises :class:`IngestError` when it is none of them."""
    data = load_json_input(data)
    item = _first_item(data)
    if isinstance(item, dict):
        if "resourceSpans" in item or "resource_spans" in item:
            return "openinference" if _has_oi_kind(data) else "otel"
        if isinstance(item.get("observations"), list):
            return "langfuse"
        if "run_type" in item:
            return "langsmith"
        attrs = item.get("attributes")
        flat_oi = any(str(k).startswith("attributes.openinference") for k in item)
        if ("context" in item or "span_kind" in item or "context.span_id" in item) and (
            isinstance(attrs, dict) or flat_oi or "span_kind" in item
        ):
            return "openinference"
    raise IngestError(
        "unrecognized trace format: expected OTLP/JSON (resourceSpans), a Langfuse trace "
        "(observations), OpenInference / Phoenix spans or LangSmith runs (run_type)"
    )


def from_otlp(data: Any) -> list[Run]:
    from .otlp import from_otlp as _f

    return _f(data)


def from_openinference(data: Any) -> list[Run]:
    from .openinference import from_openinference as _f

    return _f(data)


def from_langfuse(data: Any) -> list[Run]:
    from .langfuse import from_langfuse as _f

    return _f(data)


def from_langsmith(data: Any) -> list[Run]:
    from .langsmith import from_langsmith as _f

    return _f(data)


from_otlp.__doc__ = "Runs from an OTLP/JSON export (OpenTelemetry GenAI), one per trace id."
from_openinference.__doc__ = "Runs from OpenInference / Arize Phoenix spans, one per trace id."
from_langfuse.__doc__ = "Runs from Langfuse trace exports (trace + observations), one per trace."
from_langsmith.__doc__ = "Runs from LangSmith run exports (run trees), one per trace."

_PARSERS = {
    "otel": from_otlp,
    "otlp": from_otlp,
    "openinference": from_openinference,
    "phoenix": from_openinference,
    "langfuse": from_langfuse,
    "langsmith": from_langsmith,
}


def load_path(path: str | os.PathLike[str], fmt: str = "auto") -> list[Run]:
    """Runs from one ``.json`` file or ``.jsonl`` file (one document per line).

    ``fmt`` is ``auto`` (detected) or one of :data:`FORMATS`."""
    p = Path(path)
    if fmt != "auto" and fmt not in _PARSERS:
        raise IngestError(f"unknown format {fmt!r} (use auto or one of {', '.join(FORMATS)})")
    try:
        text = p.read_text("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise IngestError(f"cannot read {p}: {exc}") from exc
    if p.suffix.lower() == ".jsonl":
        docs: list[Any] = []
        for n, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                docs.append(json.loads(line))
            except ValueError as exc:
                raise IngestError(f"line {n}: not valid JSON: {exc}") from exc
        if not docs:
            raise IngestError("the file is empty")
        if fmt == "auto" and len({detect_format(d) for d in docs}) > 1:
            runs: list[Run] = []
            for d in docs:
                runs.extend(_PARSERS[detect_format(d)](d))
            return runs
        data: Any = []
        for d in docs:  # one trace / span batch per line: parse them as a single export
            data.extend(d) if isinstance(d, list) else data.append(d)
    else:
        data = load_json_input(text)
    name = detect_format(data) if fmt == "auto" else fmt
    return _PARSERS[name](data)


def save_runs(store: Any, runs: list[Run], *, replace: bool = False) -> list[tuple[str, bool]]:
    """Write ``runs`` to ``store`` the way recording does (secrets redacted). Returns
    ``(run_id, added)``; a run already in the store is left alone unless ``replace``."""
    out: list[tuple[str, bool]] = []
    for run in runs:
        if store.index.get_run_json(run.run_id) is not None and not replace:
            out.append((run.run_id, False))
            continue
        cassette = store.cassette_path(run.run_id)
        cassette.parent.mkdir(parents=True, exist_ok=True)
        cassette.write_text("", "utf-8")
        for step in run.steps:
            store.add_step(step)
        store.end_run(run)
        out.append((run.run_id, True))
    return out
