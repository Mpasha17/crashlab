"""Span-shaped input shared by the OTLP and OpenInference importers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ._core import IngestError, maybe_json, parse_time


@dataclass
class FlatSpan:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    start: datetime | None
    end: datetime | None
    attrs: dict[str, Any]
    status_code: str = "UNSET"  # UNSET | OK | ERROR
    status_message: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)  # {name, attrs}

    def exception(self) -> dict[str, Any] | None:
        for ev in self.events:
            if ev.get("name") == "exception":
                return ev.get("attrs") or {}
        return None


def decode_value(v: Any) -> Any:
    """An OTLP ``AnyValue`` (``{"stringValue": ...}`` ...) as a plain value."""
    if not isinstance(v, dict):
        return v
    for key in ("stringValue", "string_value"):
        if key in v:
            return v[key]
    for key in ("intValue", "int_value"):
        if key in v:
            try:
                return int(v[key])
            except (TypeError, ValueError):
                return v[key]
    for key in ("doubleValue", "double_value"):
        if key in v:
            return v[key]
    for key in ("boolValue", "bool_value"):
        if key in v:
            return v[key]
    for key in ("arrayValue", "array_value"):
        if key in v:
            return [decode_value(x) for x in (v[key] or {}).get("values") or []]
    for key in ("kvlistValue", "kvlist_value"):
        if key in v:
            return decode_attrs((v[key] or {}).get("values"))
    for key in ("bytesValue", "bytes_value"):
        if key in v:
            return v[key]
    return v


def decode_attrs(items: Any) -> dict[str, Any]:
    """OTLP attributes (a list of ``{key, value}``) or a plain mapping -> dict."""
    if isinstance(items, dict):
        return dict(items)
    out: dict[str, Any] = {}
    for item in items or []:
        if isinstance(item, dict) and "key" in item:
            out[str(item["key"])] = decode_value(item.get("value"))
    return out


def _pick(d: dict[str, Any], *names: str) -> Any:
    for n in names:
        if d.get(n) is not None:
            return d[n]
    return None


_STATUS = {
    0: "UNSET",
    1: "OK",
    2: "ERROR",
    "STATUS_CODE_UNSET": "UNSET",
    "STATUS_CODE_OK": "OK",
    "STATUS_CODE_ERROR": "ERROR",
    "UNSET": "UNSET",
    "OK": "OK",
    "ERROR": "ERROR",
}


def _status(raw: Any) -> tuple[str, str | None]:
    if isinstance(raw, dict):
        code = _STATUS.get(raw.get("code", 0), _STATUS.get(str(raw.get("code")).upper(), "UNSET"))
        return code, raw.get("message")
    if raw is None:
        return "UNSET", None
    return _STATUS.get(raw, _STATUS.get(str(raw).upper(), "UNSET")), None


def otlp_spans(data: Any) -> list[FlatSpan]:
    """Every span of an OTLP/JSON export (one document or a list of them)."""
    docs = data if isinstance(data, list) else [data]
    out: list[FlatSpan] = []
    for doc in docs:
        if not isinstance(doc, dict):
            raise IngestError("OTLP/JSON must be an object with resourceSpans")
        resources = _pick(doc, "resourceSpans", "resource_spans")
        if not isinstance(resources, list):
            raise IngestError("OTLP/JSON must have a resourceSpans list")
        for res in resources:
            scopes = (
                _pick(
                    res or {},
                    "scopeSpans",
                    "scope_spans",
                    "instrumentationLibrarySpans",
                    "instrumentation_library_spans",
                )
                or []
            )
            for scope in scopes:
                for sp in (scope or {}).get("spans") or []:
                    if not isinstance(sp, dict):
                        continue
                    code, msg = _status(sp.get("status"))
                    events = [
                        {"name": e.get("name"), "attrs": decode_attrs(e.get("attributes"))}
                        for e in sp.get("events") or []
                        if isinstance(e, dict)
                    ]
                    out.append(
                        FlatSpan(
                            trace_id=str(_pick(sp, "traceId", "trace_id") or ""),
                            span_id=str(_pick(sp, "spanId", "span_id") or ""),
                            parent_id=(
                                str(_pick(sp, "parentSpanId", "parent_span_id") or "") or None
                            ),
                            name=str(sp.get("name") or ""),
                            start=parse_time(
                                _pick(sp, "startTimeUnixNano", "start_time_unix_nano")
                            ),
                            end=parse_time(_pick(sp, "endTimeUnixNano", "end_time_unix_nano")),
                            attrs=decode_attrs(sp.get("attributes")),
                            status_code=code,
                            status_message=msg,
                            events=events,
                        )
                    )
    return out


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Nested dicts / lists as dotted keys (``llm.input_messages.0.message.role``)."""
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)) and v and not _is_leaf_json(key, v):
                out.update(flatten(v, key))
            else:
                out[key] = v
    elif isinstance(value, list):
        for i, v in enumerate(value):
            key = f"{prefix}.{i}"
            if isinstance(v, (dict, list)) and v:
                out.update(flatten(v, key))
            else:
                out[key] = v
    return out


def _is_leaf_json(key: str, value: Any) -> bool:
    # a JSON-valued attribute whose own structure is the payload
    return key.endswith(("invocation_parameters", "json_schema", "metadata", "arguments")) or (
        isinstance(value, list) and all(not isinstance(x, (dict, list)) for x in value)
    )


def group_by_trace(spans: list[FlatSpan]) -> dict[str, list[FlatSpan]]:
    groups: dict[str, list[FlatSpan]] = {}
    for sp in spans:
        groups.setdefault(sp.trace_id or "trace", []).append(sp)
    return groups


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


__all__ = ["FlatSpan", "decode_attrs", "flatten", "group_by_trace", "otlp_spans", "maybe_json"]
