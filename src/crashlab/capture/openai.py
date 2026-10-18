"""Recording proxy for OpenAI (and OpenAI-compatible) clients.

Intercepts ``client.chat.completions.create`` on sync and async clients. Everything else
is delegated untouched. The ``openai`` package is never imported here, so crashlab works
without it (any object with the same shape can be wrapped).

Limitations in v0.0.1: ``stream=True`` calls are passed through and recorded without the
streamed response; ``responses.create`` is not intercepted yet.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any
from urllib.parse import urlparse

from ._steps import begin_step
from .normalize import (
    normalize_openai_chat_request,
    normalize_openai_chat_response,
    to_jsonable,
)

_warned_outside_run = False


def _warn_outside_run() -> None:
    global _warned_outside_run
    if not _warned_outside_run:
        _warned_outside_run = True
        warnings.warn(
            "crashlab: LLM call made outside a recorded run was not recorded. Wrap your entry "
            "function with @crashlab.agent(...) or use `with crashlab.record(...)`.",
            stacklevel=4,
        )


def _server_address(client: Any) -> str | None:
    base = getattr(client, "base_url", None)
    if base is None:
        return None
    try:
        return urlparse(str(base)).hostname
    except ValueError:
        return None


def _request_attributes(kwargs: dict[str, Any], provider: str, server: str | None) -> dict:
    attrs: dict[str, Any] = {"gen_ai.provider.name": provider}
    if server:
        attrs["server.address"] = server
    model = kwargs.get("model")
    if isinstance(model, str):
        attrs["gen_ai.request.model"] = model
    for key, attr in (
        ("temperature", "gen_ai.request.temperature"),
        ("top_p", "gen_ai.request.top_p"),
        ("max_tokens", "gen_ai.request.max_tokens"),
        ("max_completion_tokens", "gen_ai.request.max_tokens"),
        ("seed", "gen_ai.request.seed"),
    ):
        val = kwargs.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            attrs[attr] = val
    if kwargs.get("stream") is True:
        attrs["crashlab.stream"] = True
    return attrs


def _response_attributes(norm: dict[str, Any]) -> dict[str, Any]:
    usage = norm.get("usage") or {}
    attrs: dict[str, Any] = {
        "gen_ai.response.model": norm.get("model"),
        "gen_ai.response.id": norm.get("id"),
        "gen_ai.response.finish_reasons": norm.get("finish_reasons") or [],
        "gen_ai.usage.input_tokens": usage.get("input_tokens"),
        "gen_ai.usage.output_tokens": usage.get("output_tokens"),
    }
    return {k: v for k, v in attrs.items() if v is not None}


class _Recorder:
    """Records one chat.completions.create call."""

    def __init__(self, kwargs: dict[str, Any], provider: str, server: str | None) -> None:
        normalized = normalize_openai_chat_request(kwargs)
        model = kwargs.get("model")
        self.stream = kwargs.get("stream") is True
        self.pending = begin_step(
            "chat",
            str(model) if model else "unknown-model",
            normalized,
            attributes=_request_attributes(kwargs, provider, server),
            raw_request=to_jsonable(kwargs),
        )
        if self.pending is None:
            _warn_outside_run()

    def fail(self, exc: BaseException) -> None:
        if self.pending is not None:
            self.pending.fail(exc)

    def finish(self, resp: Any) -> None:
        if self.pending is None:
            return
        if self.stream:
            self.pending.finish(
                None, attributes={"crashlab.note": "stream=True responses are not recorded yet"}
            )
            return
        norm, raw = normalize_openai_chat_response(resp)
        self.pending.finish(norm, raw_response=raw, attributes=_response_attributes(norm))


class _CompletionsProxy:
    def __init__(self, inner: Any, provider: str, server: str | None) -> None:
        self._inner = inner
        self._provider = provider
        self._server = server

    def create(self, *args: Any, **kwargs: Any) -> Any:
        rec = _Recorder(kwargs, self._provider, self._server)
        try:
            result = self._inner.create(*args, **kwargs)
        except BaseException as exc:
            rec.fail(exc)
            raise
        if inspect.isawaitable(result):

            async def _awaited() -> Any:
                try:
                    resp = await result
                except BaseException as exc:
                    rec.fail(exc)
                    raise
                rec.finish(resp)
                return resp

            return _awaited()
        rec.finish(result)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _ChatProxy:
    def __init__(self, inner: Any, provider: str, server: str | None) -> None:
        self._inner = inner
        self.completions = _CompletionsProxy(inner.completions, provider, server)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class WrappedOpenAI:
    """Thin proxy around an ``openai.OpenAI`` / ``openai.AsyncOpenAI`` (or compatible) client."""

    __crashlab_wrapped__ = True

    def __init__(self, client: Any, *, provider: str = "openai") -> None:
        object.__setattr__(self, "_crashlab_client", client)
        object.__setattr__(self, "_crashlab_provider", provider)
        object.__setattr__(self, "chat", _ChatProxy(client.chat, provider, _server_address(client)))

    @property
    def unwrapped(self) -> Any:
        return self._crashlab_client

    def with_options(self, *args: Any, **kwargs: Any) -> WrappedOpenAI:
        return WrappedOpenAI(
            self._crashlab_client.with_options(*args, **kwargs), provider=self._crashlab_provider
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._crashlab_client, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._crashlab_client, name, value)

    def __repr__(self) -> str:
        return f"crashlab.wrap({self._crashlab_client!r})"


def looks_like_openai(client: Any) -> bool:
    chat = getattr(client, "chat", None)
    completions = getattr(chat, "completions", None)
    return callable(getattr(completions, "create", None))


def wrap_openai(client: Any, *, provider: str = "openai") -> WrappedOpenAI:
    if not looks_like_openai(client):
        raise TypeError(f"{type(client).__name__} has no chat.completions.create method")
    return WrappedOpenAI(client, provider=provider)
