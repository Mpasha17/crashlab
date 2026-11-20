"""``wrap_mcp_session``: record MCP ``call_tool`` as crashlab tool steps.

The wrapper is duck-typed: it needs a client with ``call_tool(name, arguments=...)``
(sync or async), not the ``mcp`` package. ``list_tools`` and every other method are
forwarded unchanged. ``mcp`` is never imported, so a missing install does not break
``import crashlab.integrations.mcp``.

Each call is recorded as an ``execute_tool`` step. The tool name is
``mcp:{server}:{name}`` when ``server_name`` is given, otherwise ``name``. Arguments,
the result, duration and errors go through the same redaction and runtime checks as
``@crashlab.tool`` (including ``suspicious_tool_result`` and ``tool_error_streak``).

On ``crashlab.replay(run)`` the recorded result is served (or the recorded exception is
raised) and the MCP server is never contacted. ``side_effects=True`` uses the same
protection as other side-effect tools: a fork will not run a new call live unless
``allow_live_side_effects=True``.

MCP ``CallToolResult``-like values (objects or dicts with ``content`` and ``isError``)
are stored as JSON and rebuilt on replay as :class:`McpCallToolResult`, so
``result.content[0].text`` and ``result.isError`` still work offline.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..capture._steps import begin_step
from ..capture.normalize import to_jsonable
from ..capture.tools import ToolSpec, _serve_replay
from ..replay.rebuild import recancel

_RESULT_KIND = "crashlab.mcp.result_kind"
_CALL_TOOL_RESULT = "call_tool_result"


def _is_async_callable(fn: Any) -> bool:
    if inspect.iscoroutinefunction(fn):
        return True
    inner = getattr(fn, "__func__", None)
    return inspect.iscoroutinefunction(inner) if inner is not None else False


def _tool_name(server_name: str | None, name: str) -> str:
    return f"mcp:{server_name}:{name}" if server_name else name


def _content_item(item: Any) -> Any:
    if item is None or isinstance(item, (str, int, float, bool)):
        return item
    if isinstance(item, dict):
        return to_jsonable(item)
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        try:
            return to_jsonable(dump(mode="json", exclude_none=True))
        except TypeError:
            return to_jsonable(dump())
    data: dict[str, Any] = {}
    for key in ("type", "text", "data", "mimeType", "annotations", "name", "uri", "resource"):
        if hasattr(item, key):
            value = getattr(item, key)
            if value is not None:
                data[key] = to_jsonable(value)
    return data or to_jsonable(item)


def _is_call_tool_result(obj: Any) -> bool:
    if obj is None:
        return False
    if isinstance(obj, McpCallToolResult):
        return True
    if isinstance(obj, dict):
        return "content" in obj and ("isError" in obj or "is_error" in obj)
    return hasattr(obj, "content") and (hasattr(obj, "isError") or hasattr(obj, "is_error"))


def _is_error_flag(obj: Any) -> bool:
    if isinstance(obj, dict):
        flag = obj.get("isError", obj.get("is_error", False))
    else:
        flag = getattr(obj, "isError", getattr(obj, "is_error", False))
    return bool(flag)


def _structured(obj: Any) -> Any:
    if isinstance(obj, dict):
        return obj.get("structuredContent", obj.get("structured_content"))
    return getattr(obj, "structuredContent", getattr(obj, "structured_content", None))


def _content_list(obj: Any) -> list[Any]:
    raw = obj.get("content") if isinstance(obj, dict) else getattr(obj, "content", None)
    if raw is None:
        return []
    if isinstance(raw, list):
        return [_content_item(item) for item in raw]
    return [_content_item(raw)]


def _error_text(content: list[Any], obj: Any) -> str:
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("text"):
            parts.append(str(item["text"]))
        elif isinstance(item, str):
            parts.append(item)
    if parts:
        return "\n".join(parts)
    if isinstance(obj, dict):
        for key in ("error", "message"):
            if obj.get(key):
                return str(obj[key])
    return "tool returned isError=True"


def normalize_mcp_result(result: Any) -> tuple[Any, dict[str, Any] | None]:
    """JSON payload and step attributes for a live MCP ``call_tool`` result."""
    if _is_call_tool_result(result):
        content = _content_list(result)
        is_error = _is_error_flag(result)
        payload: dict[str, Any] = {"content": content, "isError": is_error}
        structured = _structured(result)
        if structured is not None:
            payload["structuredContent"] = to_jsonable(structured)
        # classify_result / tool_error_streak look for an ``error`` key
        if is_error:
            payload["error"] = _error_text(content, result)
        return payload, {_RESULT_KIND: _CALL_TOOL_RESULT}
    return to_jsonable(result), None


@dataclass
class McpContent:
    """A rebuilt MCP content part, typically ``type='text'`` with ``text``."""

    type: str = "text"
    text: str = ""
    data: Any = None
    mimeType: str | None = None
    annotations: Any = None
    name: str | None = None
    uri: str | None = None
    resource: Any = None

    def model_dump(self, **_kw: Any) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        if self.text or self.type == "text":
            out["text"] = self.text
        if self.data is not None:
            out["data"] = self.data
        if self.mimeType is not None:
            out["mimeType"] = self.mimeType
        if self.annotations is not None:
            out["annotations"] = self.annotations
        if self.name is not None:
            out["name"] = self.name
        if self.uri is not None:
            out["uri"] = self.uri
        if self.resource is not None:
            out["resource"] = self.resource
        return out


def _as_content(item: Any) -> Any:
    if isinstance(item, McpContent):
        return item
    if not isinstance(item, dict):
        return item
    known = {k: item[k] for k in ("type", "text", "data", "mimeType", "annotations",
                                   "name", "uri", "resource") if k in item}  # fmt: skip
    if not known:
        return item
    known.setdefault("type", "text")
    known.setdefault("text", "")
    return McpContent(**known)


@dataclass
class McpCallToolResult:
    """Duck-typed MCP ``CallToolResult`` rebuilt from a recorded step on replay."""

    content: list[Any] = field(default_factory=list)
    isError: bool = False
    structuredContent: Any = None

    def __post_init__(self) -> None:
        self.content = [_as_content(item) for item in self.content or []]

    def model_dump(self, **_kw: Any) -> dict[str, Any]:
        content = [
            item.model_dump() if isinstance(item, McpContent) else item for item in self.content
        ]
        out: dict[str, Any] = {"content": content, "isError": self.isError}
        if self.structuredContent is not None:
            out["structuredContent"] = self.structuredContent
        return out


def restore_mcp_result(value: Any, *, kind: str | None = None) -> Any:
    """Rebuild a :class:`McpCallToolResult` from recorded JSON when the live call
    returned a CallToolResult-like object; otherwise return ``value`` as stored."""
    if isinstance(value, McpCallToolResult):
        return value
    if kind == _CALL_TOOL_RESULT or _is_call_tool_result(value):
        if not isinstance(value, dict):
            return value
        return McpCallToolResult(
            content=list(value.get("content") or []),
            isError=bool(value.get("isError", value.get("is_error", False))),
            structuredContent=value.get("structuredContent", value.get("structured_content")),
        )
    return value


def _bind_call(
    name: Any, arguments: Any, args: tuple, kwargs: dict
) -> tuple[str, Any, tuple, dict]:
    extra = dict(kwargs)
    if arguments is None and "arguments" in extra:
        arguments = extra.pop("arguments")
    if arguments is None and args:
        arguments = args[0]
        args = args[1:]
    if arguments is None:
        arguments = {}
    return str(name), arguments, args, extra


def _begin_mcp(recorded_name: str, tool: str, arguments: Any, server_name: str | None,
               side_effects: bool):  # fmt: skip
    call_args = arguments if isinstance(arguments, dict) else {"value": arguments}
    attrs: dict[str, Any] = {
        "gen_ai.tool.name": recorded_name,
        "gen_ai.tool.type": "function",
        "crashlab.tool.side_effects": side_effects,
        "crashlab.mcp.tool": tool,
    }
    if server_name:
        attrs["crashlab.mcp.server"] = server_name
    return begin_step(
        "execute_tool",
        recorded_name,
        {"args": call_args},
        attributes=attrs,
        hash_input={"tool": recorded_name, "args": call_args},
    )


def _spec(recorded_name: str, side_effects: bool) -> ToolSpec:
    return ToolSpec(name=recorded_name, side_effects=side_effects, description=None)


def _run_tool(
    orig: Callable[..., Any],
    pending: Any,
    spec: ToolSpec,
    name: str,
    arguments: Any,
    args: tuple,
    kwargs: dict,
) -> Any:
    served, value, impl = _serve_replay(pending, spec, orig)
    if served:
        kind = pending.attributes.get(_RESULT_KIND)
        return restore_mcp_result(value, kind=kind)
    try:
        with pending.activate():
            result = orig(name, arguments, *args, **kwargs)
    except BaseException as exc:
        pending.fail(exc, is_tool_error=True)
        raise
    payload, extra = normalize_mcp_result(result)
    pending.finish({"result": payload}, attributes=extra)
    return result


async def _arun_tool(
    orig: Callable[..., Any],
    pending: Any,
    spec: ToolSpec,
    name: str,
    arguments: Any,
    args: tuple,
    kwargs: dict,
) -> Any:
    try:
        served, value, impl = _serve_replay(pending, spec, orig)
    except BaseException as exc:
        await recancel(exc)
        raise
    if served:
        kind = pending.attributes.get(_RESULT_KIND)
        return restore_mcp_result(value, kind=kind)
    try:
        with pending.activate():
            result = orig(name, arguments, *args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
    except BaseException as exc:
        pending.fail(exc, is_tool_error=True)
        raise
    payload, extra = normalize_mcp_result(result)
    pending.finish({"result": payload}, attributes=extra)
    return result


class _McpSessionProxy:
    """Forwards every attribute to the wrapped session except ``call_tool``."""

    __crashlab_mcp_wrapped__ = True

    def __init__(self, session: Any, *, server_name: str | None, side_effects: bool) -> None:
        object.__setattr__(self, "_crashlab_session", session)
        object.__setattr__(self, "_crashlab_server_name", server_name)
        object.__setattr__(self, "_crashlab_side_effects", side_effects)
        object.__setattr__(self, "_crashlab_async", _is_async_callable(session.call_tool))

    @property
    def unwrapped(self) -> Any:
        return self._crashlab_session

    def __getattr__(self, name: str) -> Any:
        return getattr(self._crashlab_session, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_crashlab") or name in {"call_tool"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self._crashlab_session, name, value)

    def __enter__(self) -> Any:
        entered = self._crashlab_session.__enter__()
        if entered is self._crashlab_session:
            return self
        return wrap_mcp_session(
            entered, server_name=self._crashlab_server_name,
            side_effects=self._crashlab_side_effects,
        )  # fmt: skip

    def __exit__(self, *exc: Any) -> Any:
        return self._crashlab_session.__exit__(*exc)

    async def __aenter__(self) -> Any:
        entered = await self._crashlab_session.__aenter__()
        if entered is self._crashlab_session:
            return self
        return wrap_mcp_session(
            entered, server_name=self._crashlab_server_name,
            side_effects=self._crashlab_side_effects,
        )  # fmt: skip

    async def __aexit__(self, *exc: Any) -> Any:
        return await self._crashlab_session.__aexit__(*exc)

    def _start(self, name: Any, arguments: Any, args: tuple, kwargs: dict):
        tool, arguments, args, kwargs = _bind_call(name, arguments, args, kwargs)
        recorded = _tool_name(self._crashlab_server_name, tool)
        pending = _begin_mcp(
            recorded, tool, arguments, self._crashlab_server_name, self._crashlab_side_effects
        )
        spec = _spec(recorded, self._crashlab_side_effects)
        return tool, arguments, args, kwargs, pending, spec

    def call_tool(self, name: str, arguments: Any = None, *args: Any, **kwargs: Any) -> Any:
        tool, arguments, args, kwargs, pending, spec = self._start(name, arguments, args, kwargs)
        orig = self._crashlab_session.call_tool
        if pending is None:
            return orig(tool, arguments, *args, **kwargs)
        if self._crashlab_async:
            return _arun_tool(orig, pending, spec, tool, arguments, args, kwargs)
        return _run_tool(orig, pending, spec, tool, arguments, args, kwargs)


def wrap_mcp_session(
    session: Any, *, server_name: str | None = None, side_effects: bool = False
) -> Any:
    """Return a recording proxy for an MCP ``ClientSession`` (or any duck-typed client
    with ``call_tool(name, arguments=...)``).

    ``server_name`` is prepended to the recorded tool name as ``mcp:{server}:{name}``.
    ``side_effects=True`` marks every call as a side-effect tool (writes, sends, payments)
    so a replay or fork never hits the real server for a new call.

    Wrapping is idempotent: an already wrapped session is returned unchanged. Other
    methods (``list_tools``, ``initialize``, context managers) are forwarded.
    """
    if session is None:
        raise TypeError("wrap_mcp_session(): session is required")
    if getattr(session, "__crashlab_mcp_wrapped__", False):
        return session
    if not callable(getattr(session, "call_tool", None)):
        raise TypeError(
            "wrap_mcp_session(): session has no call_tool(name, arguments=...); "
            "pass an MCP ClientSession or a duck-typed client"
        )
    if server_name is not None and (not isinstance(server_name, str) or not server_name.strip()):
        raise ValueError("wrap_mcp_session(): server_name must be a non-empty string")
    return _McpSessionProxy(session, server_name=server_name, side_effects=bool(side_effects))


__all__ = [
    "McpCallToolResult",
    "McpContent",
    "normalize_mcp_result",
    "restore_mcp_result",
    "wrap_mcp_session",
]
