"""Recording proxy for ``anthropic.Anthropic`` / ``anthropic.AsyncAnthropic``.

Intercepts ``client.messages.create`` (plain and ``stream=True``),
``client.messages.stream(...)`` and ``client.messages.parse(...)`` (structured outputs).
``anthropic`` is imported only lazily, to re-run its parser over a replayed ``parse()``.
Not intercepted yet: ``client.beta.messages``, ``messages.batches``, ``count_tokens``.
"""

from __future__ import annotations

import inspect
from typing import Any

from . import normalize as N
from ._chat import ChatAdapter, ChatCall, invoke, prepare_request
from ._proxy import Proxy, WrappedClient, server_address
from .streams import (
    AnthropicAssembler,
    AsyncRecordingMessageStreamManager,
    RecordingMessageStreamManager,
)

MESSAGES = ChatAdapter(
    provider="anthropic",
    api="anthropic.messages",
    normalize_request=N.normalize_anthropic_request,
    normalize_response=N.normalize_anthropic_response,
    new_assembler=AnthropicAssembler,
)


def _is_async_messages(messages: Any) -> bool:
    if "Async" in type(messages).__name__:
        return True
    return inspect.iscoroutinefunction(getattr(messages, "create", None))


class WrappedAnthropic(WrappedClient):
    def __init__(self, client: Any) -> None:
        server = server_address(client)
        messages = client.messages
        is_async = _is_async_messages(messages)

        def create(*args: Any, **kwargs: Any) -> Any:
            return invoke(MESSAGES, messages.create, args, kwargs, server=server, is_async=is_async)

        def stream(*args: Any, **kwargs: Any) -> Any:
            patched, idx, applied = prepare_request(MESSAGES, dict(kwargs))
            if applied:
                args, kwargs = (), patched
            call = ChatCall(MESSAGES, kwargs, server=server, stream=True, idx=idx,
                            patches=applied)  # fmt: skip
            src = call.replay_source()
            if src is not None:
                return _replay_stream_manager(call, src, is_async)
            try:
                manager = messages.stream(*args, **kwargs)
            except BaseException as exc:
                call.fail(exc)
                raise
            if not call.active:
                return manager
            if is_async or hasattr(manager, "__aenter__") and not hasattr(manager, "__enter__"):
                return AsyncRecordingMessageStreamManager(manager, call)
            return RecordingMessageStreamManager(manager, call)

        messages_kw: dict[str, Any] = {"create": create, "stream": stream}
        if callable(getattr(messages, "parse", None)):

            def parse(*args: Any, **kwargs: Any) -> Any:
                out = invoke(MESSAGES, messages.parse, args, kwargs, server=server,
                             is_async=is_async)  # fmt: skip
                if is_async:

                    async def finish() -> Any:
                        return _reparse(await out, kwargs)

                    return finish()
                return _reparse(out, kwargs)

            parse.__doc__ = getattr(messages.parse, "__doc__", None)
            messages_kw["parse"] = parse
        if getattr(messages, "with_raw_response", None) is not None:
            raw = messages.with_raw_response

            def raw_create(*args: Any, **kwargs: Any) -> Any:
                return invoke(
                    MESSAGES, raw.create, args, kwargs, server=server, is_async=is_async, raw=True
                )

            raw_create.__doc__ = getattr(raw.create, "__doc__", None)
            messages_kw["with_raw_response"] = Proxy(raw, create=raw_create)
        overrides: dict[str, Any] = {"messages": Proxy(messages, **messages_kw)}
        super().__init__(client, **overrides)

    def _rewrap(self, client: Any) -> WrappedAnthropic:
        return WrappedAnthropic(client)


def _reparse(obj: Any, kwargs: dict[str, Any]) -> Any:
    """A replayed ``messages.parse()`` result is rebuilt as a plain ``Message``: run the
    SDK's parser over it so ``.parsed_output`` works as when recorded."""
    if obj is None or type(obj).__name__.startswith("Parsed"):
        return obj
    try:
        from anthropic import omit
        from anthropic.lib._parse._response import parse_response

        return parse_response(response=obj, output_format=kwargs.get("output_format", omit))
    except Exception:  # pragma: no cover - SDK internals moved: serve what we rebuilt
        return obj


def _replay_stream_manager(call: ChatCall, src: Any, is_async: bool) -> Any:
    from ..replay import rebuild as R
    from ..replay import synth

    call.pending.finish_from(src)  # type: ignore[union-attr]
    error = R.rebuild_exception(src.error) if src.error is not None else None
    events = synth.synthesize(MESSAGES.api, src.raw_response, R.DEFAULT_CHUNK_TYPES[MESSAGES.api])
    cls = synth.AsyncReplayMessageStreamManager if is_async else synth.ReplayMessageStreamManager
    return cls(events, src.raw_response, error)


def looks_like_anthropic(client: Any) -> bool:
    return callable(getattr(getattr(client, "messages", None), "create", None))


def wrap_anthropic(client: Any) -> WrappedAnthropic:
    if not looks_like_anthropic(client):
        raise TypeError(f"{type(client).__name__} has no messages.create method")
    return WrappedAnthropic(client)
