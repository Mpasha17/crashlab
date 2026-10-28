"""Recording proxy for ``anthropic.Anthropic`` / ``anthropic.AsyncAnthropic``.

Intercepts ``create`` (plain and ``stream=True``), ``stream(...)`` and ``parse(...)``
(structured outputs) on ``client.messages`` and ``client.beta.messages``. ``anthropic`` is
imported only lazily, to re-run its parser over a replayed ``parse()``.
Also ``count_tokens`` (a ``custom`` step). Not intercepted yet: ``messages.batches``, the beta
``tool_runner``.
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


def _count_tokens_response(resp: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = N.to_jsonable(resp)
    raw = raw if isinstance(raw, dict) else {"input_tokens": raw}
    return {"input_tokens": raw.get("input_tokens")}, raw


# a pre-flight token count (no generation): recorded so replay stays offline; operation
# "custom" keeps it out of LLM-call counts, tokens and cost
COUNT_TOKENS = ChatAdapter(
    provider="anthropic",
    api="anthropic.messages.count_tokens",
    normalize_request=N.normalize_anthropic_request,
    normalize_response=_count_tokens_response,
    new_assembler=AnthropicAssembler,
    operation="custom",
)


def _is_async_messages(messages: Any) -> bool:
    if "Async" in type(messages).__name__:
        return True
    return inspect.iscoroutinefunction(getattr(messages, "create", None))


def _wrap_messages(messages: Any, server: str | None) -> Proxy:
    """``messages`` (or ``beta.messages``) with create / stream / parse /
    with_raw_response.create recorded."""
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
    if callable(getattr(messages, "count_tokens", None)):

        def count_tokens(*args: Any, **kwargs: Any) -> Any:
            return invoke(COUNT_TOKENS, messages.count_tokens, args, kwargs, server=server,
                          is_async=is_async)  # fmt: skip

        count_tokens.__doc__ = getattr(messages.count_tokens, "__doc__", None)
        messages_kw["count_tokens"] = count_tokens
    if getattr(messages, "with_raw_response", None) is not None:
        raw = messages.with_raw_response

        def raw_create(*args: Any, **kwargs: Any) -> Any:
            return invoke(
                MESSAGES, raw.create, args, kwargs, server=server, is_async=is_async, raw=True
            )

        raw_create.__doc__ = getattr(raw.create, "__doc__", None)
        messages_kw["with_raw_response"] = Proxy(raw, create=raw_create)
    return Proxy(messages, **messages_kw)


class WrappedAnthropic(WrappedClient):
    def __init__(self, client: Any) -> None:
        server = server_address(client)
        overrides: dict[str, Any] = {"messages": _wrap_messages(client.messages, server)}
        beta = getattr(client, "beta", None)
        beta_messages = getattr(beta, "messages", None)
        if callable(getattr(beta_messages, "create", None)):
            overrides["beta"] = Proxy(beta, messages=_wrap_messages(beta_messages, server))
        super().__init__(client, **overrides)

    def _rewrap(self, client: Any) -> WrappedAnthropic:
        return WrappedAnthropic(client)


def _already_parsed(obj: Any) -> bool:
    """A live result carries model instances; one rebuilt from a cassette carries the
    recorded JSON (a dict) or nothing."""

    def done(value: Any) -> bool:
        return value is not None and not isinstance(value, dict)

    for choice in getattr(obj, "choices", None) or []:
        if done(getattr(getattr(choice, "message", None), "parsed", None)):
            return True
    try:
        return any(done(getattr(obj, a, None)) for a in ("output_parsed", "parsed_output"))
    except Exception:  # a property over a malformed body
        return False


def _reparse(obj: Any, kwargs: dict[str, Any]) -> Any:
    """A replayed ``messages.parse()`` result is rebuilt from JSON: run the
    SDK's parser over it so ``.parsed_output`` works as when recorded."""
    from ..context import is_replaying

    if obj is None or not is_replaying() or _already_parsed(obj):
        return obj  # a live call (recording, or a fork past its fork step): SDK-parsed
    try:
        from anthropic import omit
        from anthropic.lib._parse import _response as P

        # BetaMessage (client.beta.messages.parse) has its own parser
        parse = P.parse_beta_response if "Beta" in type(obj).__name__ else P.parse_response
        return parse(response=obj, output_format=kwargs.get("output_format", omit))
    except Exception:  # pragma: no cover - SDK internals moved: serve what we rebuilt
        return obj


def _replay_stream_manager(call: ChatCall, src: Any, is_async: bool) -> Any:
    from ..replay import rebuild as R
    from ..replay import synth

    call.pending.finish_from(src)  # type: ignore[union-attr]
    error = R.rebuild_exception(src.error) if src.error is not None else None
    events = synth.synthesize(
        MESSAGES.api,
        src.raw_response,
        R.DEFAULT_CHUNK_TYPES[MESSAGES.api],
        plan=src.attributes.get("crashlab.stream.plan"),
    )
    cls = synth.AsyncReplayMessageStreamManager if is_async else synth.ReplayMessageStreamManager
    return cls(events, src.raw_response, error)


def looks_like_anthropic(client: Any) -> bool:
    return callable(getattr(getattr(client, "messages", None), "create", None))


def wrap_anthropic(client: Any) -> WrappedAnthropic:
    if not looks_like_anthropic(client):
        raise TypeError(f"{type(client).__name__} has no messages.create method")
    return WrappedAnthropic(client)
