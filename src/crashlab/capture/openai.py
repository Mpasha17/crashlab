"""Recording proxy for OpenAI (and OpenAI-compatible) clients.

Intercepts ``client.chat.completions.create`` and ``client.responses.create`` on sync and
async clients, including ``stream=True`` (chunks pass through unchanged; the assembled
response is recorded when the stream ends). Everything else is delegated untouched. The
``openai`` package is never imported here, so any object with the same shape works.
"""

from __future__ import annotations

import inspect
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


def _raw_create(inner_completions: Any, adapter: ChatAdapter, server: str | None) -> Any:
    """Record ``with_raw_response.create`` (CrewAI) without losing the raw return value."""
    raw_resource = inner_completions.with_raw_response
    is_async = _is_async(inner_completions)

    def create(*args: Any, **kwargs: Any) -> Any:
        # invoke records via normalize; _parsed_for_record unwraps LegacyAPIResponse
        return invoke(adapter, raw_resource.create, args, kwargs, server=server, is_async=is_async)

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
            # CrewAI calls with_raw_response.create; bind recording onto that path too
            if getattr(chat.completions, "with_raw_response", None) is not None:
                completions_kw["with_raw_response"] = Proxy(
                    chat.completions.with_raw_response,
                    create=_raw_create(chat.completions, adapter, server),
                )
            completions = Proxy(chat.completions, **completions_kw)
            overrides["chat"] = Proxy(chat, completions=completions)
        responses = getattr(client, "responses", None)
        if responses is not None and callable(getattr(responses, "create", None)):
            resp_adapter = _adapter(RESPONSES, provider)
            resp_kw: dict[str, Any] = {"create": _create(responses, resp_adapter, server)}
            if getattr(responses, "with_raw_response", None) is not None:
                resp_kw["with_raw_response"] = Proxy(
                    responses.with_raw_response,
                    create=_raw_create(responses, resp_adapter, server),
                )
            overrides["responses"] = Proxy(responses, **resp_kw)
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
