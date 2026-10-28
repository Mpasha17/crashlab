"""Recording proxy for OpenAI (and OpenAI-compatible) clients.

Intercepts ``client.chat.completions.create`` / ``.parse`` and ``client.responses.create`` /
``.parse`` / ``.stream`` on sync and async clients, including ``stream=True`` (chunks pass through
unchanged; the assembled response is recorded when the stream ends). Everything else is
delegated untouched. ``openai`` is imported only lazily to re-run the SDK's parser over a
replayed ``parse()`` result, so any object with the same shape works.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any

from . import normalize as N
from ._chat import ChatAdapter, invoke
from ._proxy import Proxy, WrappedClient, server_address
from .streams import OpenAIChatAssembler, ResponsesAssembler

CHAT = ChatAdapter(
    provider="openai",
    api="openai.chat.completions",
    normalize_request=N.normalize_openai_chat_request,
    normalize_response=N.normalize_openai_chat_response,
    new_assembler=OpenAIChatAssembler,
)
RESPONSES = ChatAdapter(
    provider="openai",
    api="openai.responses",
    normalize_request=N.normalize_openai_responses_request,
    normalize_response=N.normalize_openai_responses_response,
    new_assembler=ResponsesAssembler,
)


def _adapter(base: ChatAdapter, provider: str) -> ChatAdapter:
    if provider == base.provider:
        return base
    return ChatAdapter(
        provider, base.api, base.normalize_request, base.normalize_response, base.new_assembler
    )


def _is_async(resource: Any) -> bool:
    if "Async" in type(resource).__name__:
        return True
    return inspect.iscoroutinefunction(getattr(resource, "create", None))


def _create(inner: Any, adapter: ChatAdapter, server: str | None):
    is_async = _is_async(inner)

    def create(*args: Any, **kwargs: Any) -> Any:
        return invoke(adapter, inner.create, args, kwargs, server=server, is_async=is_async)

    create.__doc__ = getattr(inner.create, "__doc__", None)
    return create


def _reparse(obj: Any, kwargs: dict[str, Any], api: str) -> Any:
    """A replayed ``parse()`` result is rebuilt as a plain completion / response: run the
    SDK's own parser over it so ``.parsed`` / ``.output_parsed`` work as when recorded."""
    if obj is None or type(obj).__name__.startswith("Parsed"):
        return obj
    try:
        from openai import omit
    except ImportError:  # pragma: no cover - older SDKs
        from openai import NOT_GIVEN as omit  # type: ignore[attr-defined,no-redef]
    try:
        if api == "openai.responses":
            from openai.lib._parsing._responses import parse_response

            return parse_response(
                text_format=kwargs.get("text_format", omit),
                input_tools=kwargs.get("tools", omit),
                response=obj,
            )
        from openai.lib._parsing._completions import parse_chat_completion

        return parse_chat_completion(
            response_format=kwargs.get("response_format", omit),
            input_tools=kwargs.get("tools", omit),
            chat_completion=obj,
        )
    except Exception:  # pragma: no cover - SDK internals moved: serve what we rebuilt
        return obj


def _parse(inner: Any, adapter: ChatAdapter, server: str | None):
    """Record ``chat.completions.parse`` / ``responses.parse`` (structured outputs)."""
    is_async = _is_async(inner)

    def parse(*args: Any, **kwargs: Any) -> Any:
        out = invoke(adapter, inner.parse, args, kwargs, server=server, is_async=is_async)
        if is_async:

            async def finish() -> Any:
                return _reparse(await out, kwargs, adapter.api)

            return finish()
        return _reparse(out, kwargs, adapter.api)

    parse.__doc__ = getattr(inner.parse, "__doc__", None)
    return parse


def _bind_stream_helper(proxy: Proxy, inner: Any) -> None:
    """``.stream(...)`` (``ChatCompletionStreamManager``) calls ``self.create(stream=True)``:
    run the SDK's own method with the recording proxy as ``self`` so the stream records."""
    helper = getattr(type(inner), "stream", None)
    if not callable(helper):
        return

    def stream(*args: Any, **kwargs: Any) -> Any:
        return helper(proxy, *args, **kwargs)

    stream.__doc__ = getattr(helper, "__doc__", None)
    object.__setattr__(proxy, "stream", stream)


def _warn_streaming_response(inner: Any, api: str) -> Any:
    """``with_streaming_response.create`` hands back raw HTTP bytes: not recorded. Say so
    once instead of silently leaving a gap in the run."""
    resource = inner.with_streaming_response

    def create(*args: Any, **kwargs: Any) -> Any:
        from ..context import current

        if current() is not None and api not in _WARNED_STREAMING_RESPONSE:
            _WARNED_STREAMING_RESPONSE.add(api)
            warnings.warn(
                f"crashlab: {api}.with_streaming_response.create is not recorded (raw HTTP "
                "body); use create(stream=True) or .stream() to record this call",
                RuntimeWarning,
                stacklevel=2,
            )
        return resource.create(*args, **kwargs)

    return Proxy(resource, create=create)


_WARNED_STREAMING_RESPONSE: set[str] = set()


def _raw_create(inner_completions: Any, adapter: ChatAdapter, server: str | None) -> Any:
    """Record ``with_raw_response.create`` (CrewAI) without losing the raw return value."""
    raw_resource = inner_completions.with_raw_response
    is_async = _is_async(inner_completions)

    def create(*args: Any, **kwargs: Any) -> Any:
        # invoke records via normalize; _parsed_for_record unwraps LegacyAPIResponse
        return invoke(
            adapter, raw_resource.create, args, kwargs, server=server, is_async=is_async, raw=True
        )

    create.__doc__ = getattr(raw_resource.create, "__doc__", None)
    return create


class WrappedOpenAI(WrappedClient):
    """Thin proxy around ``openai.OpenAI`` / ``openai.AsyncOpenAI`` (or compatible)."""

    def __init__(self, client: Any, *, provider: str = "openai") -> None:
        server = server_address(client)
        overrides: dict[str, Any] = {"_crashlab_provider": provider}
        chat = getattr(client, "chat", None)
        if chat is not None and getattr(chat, "completions", None) is not None:
            adapter = _adapter(CHAT, provider)
            create = _create(chat.completions, adapter, server)
            completions_kw: dict[str, Any] = {"create": create}
            if callable(getattr(chat.completions, "parse", None)):
                completions_kw["parse"] = _parse(chat.completions, adapter, server)
            # CrewAI calls with_raw_response.create; bind recording onto that path too
            if getattr(chat.completions, "with_raw_response", None) is not None:
                completions_kw["with_raw_response"] = Proxy(
                    chat.completions.with_raw_response,
                    create=_raw_create(chat.completions, adapter, server),
                )
            if getattr(chat.completions, "with_streaming_response", None) is not None:
                completions_kw["with_streaming_response"] = _warn_streaming_response(
                    chat.completions, adapter.api
                )
            completions = Proxy(chat.completions, **completions_kw)
            _bind_stream_helper(completions, chat.completions)
            overrides["chat"] = Proxy(chat, completions=completions)
        responses = getattr(client, "responses", None)
        if responses is not None and callable(getattr(responses, "create", None)):
            resp_adapter = _adapter(RESPONSES, provider)
            resp_kw: dict[str, Any] = {"create": _create(responses, resp_adapter, server)}
            if callable(getattr(responses, "parse", None)):
                resp_kw["parse"] = _parse(responses, resp_adapter, server)
            if getattr(responses, "with_raw_response", None) is not None:
                resp_kw["with_raw_response"] = Proxy(
                    responses.with_raw_response,
                    create=_raw_create(responses, resp_adapter, server),
                )
            if getattr(responses, "with_streaming_response", None) is not None:
                resp_kw["with_streaming_response"] = _warn_streaming_response(
                    responses, resp_adapter.api
                )
            overrides["responses"] = Proxy(responses, **resp_kw)
            _bind_stream_helper(overrides["responses"], responses)
        super().__init__(client, **overrides)

    def _rewrap(self, client: Any) -> WrappedOpenAI:
        return WrappedOpenAI(client, provider=self._crashlab_provider)


def looks_like_openai(client: Any) -> bool:
    chat = getattr(client, "chat", None)
    completions = getattr(chat, "completions", None)
    if callable(getattr(completions, "create", None)):
        return True
    return callable(getattr(getattr(client, "responses", None), "create", None))


def wrap_openai(client: Any, *, provider: str = "openai") -> WrappedOpenAI:
    if not looks_like_openai(client):
        raise TypeError(
            f"{type(client).__name__} has neither chat.completions.create nor responses.create"
        )
    return WrappedOpenAI(client, provider=provider)
