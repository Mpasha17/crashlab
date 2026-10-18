"""Rebuild provider objects and exceptions from recorded (JSON) cassette data.

Replay hands user code the same kinds of objects the live call returned: the response
class recorded at capture time (``crashlab.response.type`` / ``crashlab.stream.chunk_type``)
is imported and validated from the recorded dict (``ChatCompletion``, ``anthropic``
``Message``, ``litellm.ModelResponse`` ...). When that class is not importable or does not
validate, a :class:`ReplayObject` with attribute *and* item access is returned instead.
"""

from __future__ import annotations

import builtins
import functools
import importlib
from typing import Any

from ..models import ErrorInfo

# Used when a recording has no type attribute (recorded before crashlab 0.0.1 day 3).
DEFAULT_RESPONSE_TYPES = {
    "openai.chat.completions": "openai.types.chat.ChatCompletion",
    "openai.responses": "openai.types.responses.Response",
    "anthropic.messages": "anthropic.types.Message",
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
    """Import ``pkg.mod.Attr`` (the longest importable module prefix, then attributes)."""
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


def _validate(cls: Any, data: dict[str, Any]) -> Any:
    root = (getattr(cls, "__module__", "") or "").split(".")[0]
    attempts = []
    if root == "litellm":  # litellm objects take keyword arguments
        attempts.append(lambda: cls(**data))
    if hasattr(cls, "model_validate"):
        attempts.append(lambda: cls.model_validate(data))
    if root != "litellm":
        attempts.append(lambda: cls(**data))
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
    return exc
