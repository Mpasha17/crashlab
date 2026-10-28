"""Rebuild provider objects and exceptions from recorded (JSON) cassette data.

Replay hands user code the same kinds of objects the live call returned: the response
class recorded at capture time (``crashlab.response.type`` / ``crashlab.stream.chunk_type``)
is imported and validated from the recorded dict (``ChatCompletion``, ``anthropic``
``Message``, ``litellm.ModelResponse`` ...). When that class is not importable or does not
validate, a :class:`ReplayObject` with attribute *and* item access is returned instead.
"""

from __future__ import annotations

import builtins
import contextlib
import functools
import importlib
from typing import Any

from ..models import ErrorInfo

# Used when a recording has no type attribute (recorded before crashlab 0.0.1 day 3).
DEFAULT_RESPONSE_TYPES = {
    "openai.chat.completions": "openai.types.chat.ChatCompletion",
    "openai.responses": "openai.types.responses.Response",
    "openai.responses.retrieve": "openai.types.responses.Response",
    "openai.embeddings": "openai.types.CreateEmbeddingResponse",
    "anthropic.messages": "anthropic.types.Message",
    "anthropic.messages.count_tokens": "anthropic.types.MessageTokensCount",
    "litellm.completion": "litellm.types.utils.ModelResponse",
}
DEFAULT_CHUNK_TYPES = {
    "openai.chat.completions": "openai.types.chat.ChatCompletionChunk",
    "openai.responses": "openai.types.responses.ResponseStreamEvent",
    "anthropic.messages": "anthropic.types.RawMessageStreamEvent",
    "litellm.completion": "litellm.types.utils.ModelResponseStream",
}
# Event unions: streams whose items are a discriminated union, not one class.
_UNION_TYPES = {
    "openai": "openai.types.responses.ResponseStreamEvent",
    "anthropic": "anthropic.types.RawMessageStreamEvent",
}


def type_path(obj: Any) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


@functools.lru_cache(maxsize=256)
def import_path(path: str) -> Any:
    """Import ``pkg.mod.Attr`` (the longest importable module prefix, then attributes).

    A generic model's recorded name (``ParsedChatCompletion[TypeVar]``) imports the
    unparametrized class."""
    if path.endswith("]") and "[" in path:
        path = path[: path.index("[")]
    parts = path.split(".")
    for i in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except AttributeError:
            return None
        return obj
    return None


class ReplayObject:
    """A recorded JSON object with attribute and item access (``r.choices[0].message``)."""

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any]) -> None:
        object.__setattr__(self, "_data", data)

    @staticmethod
    def wrap(value: Any) -> Any:
        if isinstance(value, dict):
            return ReplayObject(value)
        if isinstance(value, list):
            return [ReplayObject.wrap(v) for v in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return ReplayObject.wrap(self._data[name])
        except KeyError:
            if name.startswith("__"):
                raise AttributeError(name) from None
            return None  # SDK objects expose unset optional fields as None

    def __getitem__(self, key: str) -> Any:
        return ReplayObject.wrap(self._data[key])

    def get(self, key: str, default: Any = None) -> Any:
        return ReplayObject.wrap(self._data.get(key, default))

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def keys(self):
        return self._data.keys()

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return self._data

    to_dict = model_dump

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ReplayObject):
            return self._data == other._data
        return self._data == other

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"ReplayObject({self._data!r})"


def _stainless_model(cls: Any) -> bool:
    """A model class of an SDK generated like openai's (its package has ``_models`` with a
    ``BaseModel`` whose ``construct`` builds nested models from a response body)."""
    import sys

    parts = (getattr(cls, "__module__", "") or "").split(".")
    for i in range(len(parts), 0, -1):
        mod = sys.modules.get(".".join(parts[:i]) + "._models")
        base = getattr(mod, "BaseModel", None)
        if isinstance(base, type) and isinstance(cls, type):
            return issubclass(cls, base) and hasattr(base, "construct")
    return False


def _validate(cls: Any, data: dict[str, Any]) -> Any:
    root = (getattr(cls, "__module__", "") or "").split(".")[0]
    attempts = []
    if root == "litellm":  # litellm objects take keyword arguments
        attempts.append(lambda: cls(**data))
    if hasattr(cls, "model_validate"):
        attempts.append(lambda: cls.model_validate(data))
    if root != "litellm":
        attempts.append(lambda: cls(**data))
    if (root in ("openai", "anthropic") or _stainless_model(cls)) and hasattr(cls, "construct"):
        # what the SDKs themselves do with a response body: build the (nested) models
        # without validation, so provider extensions such as Groq's
        # service_tier="on_demand" still give a real ChatCompletion (LangChain checks
        # isinstance) instead of a ReplayObject; the same for the other SDKs generated
        # like them (groq, together, cerebras ...)
        attempts.append(lambda: cls.construct(**data))
    for attempt in attempts:
        try:
            return attempt()
        except Exception:
            continue
    return None


@functools.lru_cache(maxsize=16)
def _type_adapter(path: str) -> Any:
    union = import_path(path)
    if union is None:
        return None
    try:
        from pydantic import TypeAdapter

        return TypeAdapter(union)
    except Exception:
        return None


def rebuild(data: Any, path: str | None) -> Any:
    """Recorded dict -> instance of the recorded class (or a :class:`ReplayObject`).

    ``path`` may name a class or a typing union of event classes.
    """
    if not isinstance(data, dict):
        return data
    target = import_path(path) if path else None
    if isinstance(target, type):
        obj = _validate(target, data)
        if obj is not None:
            return obj
    elif target is not None:
        adapter = _type_adapter(path)
        if adapter is not None:
            try:
                return adapter.validate_python(data)
            except Exception:
                pass
    return ReplayObject(data)


def rebuild_chunk(data: dict[str, Any], path: str | None) -> Any:
    """One stream item. Anthropic / Responses events are validated against the provider's
    event union (the recorded type is only the first event's class)."""
    union = _UNION_TYPES.get((path or "").split(".")[0])
    if union and "type" in data and data.get("object") != "chat.completion.chunk":
        return rebuild(data, union)
    return rebuild(data, path)


# --- exceptions ----------------------------------------------------------------------


class ReplayedException(Exception):
    """Mixin for exceptions re-raised from a recording.

    The class is a dynamic subclass of the original exception type (so ``except
    openai.BadRequestError`` and ``type(exc).__name__`` behave as they did live) and
    ``str(exc)`` is the recorded message, verbatim.
    """

    recorded: ErrorInfo

    def __str__(self) -> str:
        return self.recorded.message


@functools.lru_cache(maxsize=128)
def _replayed_class(type_name: str) -> type[BaseException]:
    original = import_path(type_name) if "." in type_name else getattr(builtins, type_name, None)
    if not (isinstance(original, type) and issubclass(original, BaseException)):
        original = None
    if original is None or original is Exception:
        bases: tuple[type, ...] = (ReplayedException,)
    else:
        bases = (ReplayedException, original)
    name = type_name.rsplit(".", 1)[-1]
    try:
        cls = type(name, bases, {"__module__": original.__module__ if original else __name__})
    except TypeError:  # incompatible layout (rare C-level exception types)
        cls = type(name, (ReplayedException,), {})
    cls.__qualname__ = getattr(original, "__qualname__", name)
    return cls


def rebuild_exception(info: ErrorInfo) -> BaseException:
    """An exception that looks like the recorded one (type name, message, status_code)."""
    cls = _replayed_class(info.type)
    try:  # allocate via the builtin base; skip SDK __init__s that need live HTTP objects
        builtin = next(b for b in cls.__mro__ if b.__module__ == "builtins")
        exc = builtin.__new__(cls)
    except Exception:
        cls = _replayed_class("Exception")
        exc = Exception.__new__(cls)
    exc.args = (info.message,)
    exc.recorded = info  # type: ignore[attr-defined]
    for attr, value in (("message", info.message), ("status_code", info.status_code)):
        try:
            if not isinstance(getattr(type(exc), attr, None), property):
                setattr(exc, attr, value)
        except (AttributeError, TypeError):
            pass
    if isinstance(info.status_code, int):
        _add_http_parts(exc, info)
    return exc


def _http_lib(owner: type) -> Any:
    """The HTTP library the SDK that defines ``owner`` uses: ``httpx2`` (openai>=3,
    anthropic>=1) or ``httpx``; either installed one when that is unknown."""
    import sys

    top = (getattr(owner, "__module__", "") or "").split(".")[0]
    for name in (f"{top}._exceptions", f"{top}._base_client", top):
        mod = sys.modules.get(name) if top else None
        for lib_name in ("httpx2", "httpx"):
            lib = getattr(mod, lib_name, None)
            if lib is not None and hasattr(lib, "Response"):
                return lib
    for lib_name in ("httpx", "httpx2"):
        try:
            return importlib.import_module(lib_name)
        except ImportError:
            continue
    return None


def _add_http_parts(exc: BaseException, info: ErrorInfo) -> None:
    """SDK status errors carry ``response`` / ``request`` / ``body`` (``request_id``,
    ``code`` ...); callers such as LangChain re-raise them as their own error types from
    those. Give the replayed error a minimal HTTP response with the recorded status."""
    httpx = _http_lib(type(exc))
    if httpx is None:  # pragma: no cover - the SDKs depend on httpx / httpx2
        return
    request = httpx.Request("POST", "https://crashlab.replay.invalid/")
    if info.body is not None:
        import json

        response = httpx.Response(info.status_code or 500, request=request,
                                  text=json.dumps(info.body, default=str))  # fmt: skip
    else:
        response = httpx.Response(info.status_code or 500, request=request, text=info.message)
    parts = {"response": response, "request": request, "body": info.body, "request_id": None,
             "code": None, "param": None, "type": None}  # fmt: skip
    for attr, value in parts.items():
        if attr in exc.__dict__ or isinstance(getattr(type(exc), attr, None), property):
            continue
        with contextlib.suppress(AttributeError, TypeError):
            setattr(exc, attr, value)


class ReplayRawResponse:
    """Stand-in for the SDKs' raw-response wrapper (OpenAI ``LegacyAPIResponse``,
    Anthropic ``APIResponse``) returned by ``with_raw_response.create`` during replay.

    LangChain (``raw.parse()``, ``raw.headers``) and CrewAI (``raw.text``,
    ``raw.http_response``) read these; there is no live HTTP exchange to expose, so the
    headers are empty and the body is the recorded response.
    """

    status_code = 200
    is_closed = True
    retries_taken = 0

    def __init__(self, parsed: Any, raw: Any = None) -> None:
        self._parsed = parsed
        self._raw = raw if raw is not None else {}
        self.headers: dict[str, str] = {}

    def parse(self, *, to: Any = None) -> Any:
        return self._parsed

    async def aparse(self) -> Any:  # pragma: no cover - not used by known callers
        return self._parsed

    @property
    def text(self) -> str:
        import json

        return json.dumps(self._raw, ensure_ascii=False, default=str)

    @property
    def content(self) -> bytes:
        return self.text.encode()

    def json(self) -> Any:
        return self._raw

    @property
    def http_response(self) -> Any:
        httpx = _http_lib(type(self._parsed))
        if httpx is None:  # pragma: no cover - every supported SDK depends on httpx(2)
            return None
        return httpx.Response(
            200,
            json=self._raw,
            request=httpx.Request("POST", "https://crashlab.replay.invalid/"),
        )

    def __repr__(self) -> str:
        return f"<ReplayRawResponse {type(self._parsed).__name__}>"
