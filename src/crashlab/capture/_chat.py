"""Provider-independent recording of one LLM call (plain or streamed).

Each provider supplies a :class:`ChatAdapter`; :func:`invoke` is the single interception
point used by every proxy (OpenAI chat + responses, Anthropic, LiteLLM). In replay mode it
serves the recorded response from the cassette instead of calling ``fn``; in a fork it
applies the fork's patches to the outgoing request first.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..context import current
from ..models import Step
from . import normalize as N
from ._steps import begin_step
from .streams import (
    Assembler,
    AsyncRecordingStream,
    RecordingStream,
)

_warned_outside_run = False


def _warn_outside_run() -> None:
    global _warned_outside_run
    if not _warned_outside_run:
        _warned_outside_run = True
        warnings.warn(
            "crashlab: LLM call made outside a recorded run was not recorded. Wrap your entry "
            "function with @crashlab.agent(...) or use `with crashlab.record(...)`.",
            stacklevel=5,
        )


@dataclass(frozen=True)
class ChatAdapter:
    provider: str  # gen_ai.provider.name
    api: str  # e.g. "openai.chat.completions"
    normalize_request: Callable[[dict[str, Any]], dict[str, Any]]
    normalize_response: Callable[[Any], tuple[dict[str, Any], dict[str, Any]]]
    new_assembler: Callable[[], Assembler]


_PARAM_ATTRS = (
    ("temperature", "gen_ai.request.temperature"),
    ("top_p", "gen_ai.request.top_p"),
    ("top_k", "gen_ai.request.top_k"),
    ("max_tokens", "gen_ai.request.max_tokens"),
    ("seed", "gen_ai.request.seed"),
    ("frequency_penalty", "gen_ai.request.frequency_penalty"),
    ("presence_penalty", "gen_ai.request.presence_penalty"),
)


def _request_attributes(
    adapter: ChatAdapter, normalized: dict[str, Any], server: str | None, stream: bool
) -> dict[str, Any]:
    params = normalized.get("params") or {}
    attrs: dict[str, Any] = {"gen_ai.provider.name": adapter.provider, "crashlab.api": adapter.api}
    if server:
        attrs["server.address"] = server
    if isinstance(params.get("model"), str):
        attrs["gen_ai.request.model"] = params["model"]
    for key, attr in _PARAM_ATTRS:
        val = params.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            attrs[attr] = val
    if params.get("stop"):
        attrs["gen_ai.request.stop_sequences"] = params["stop"]
    if stream:
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
        "gen_ai.usage.cache_read.input_tokens": usage.get("cache_read_input_tokens"),
        "gen_ai.usage.cache_creation.input_tokens": usage.get("cache_creation_input_tokens"),
        "gen_ai.usage.reasoning.output_tokens": usage.get("reasoning_tokens"),
        # provider-reported total; can exceed input+output (e.g. Gemini thinking tokens)
        "crashlab.usage.total_tokens": usage.get("total_tokens"),
    }
    return {k: v for k, v in attrs.items() if v is not None}


class ChatCall:
    """One recorded LLM call."""

    def __init__(
        self,
        adapter: ChatAdapter,
        kwargs: dict[str, Any],
        *,
        server: str | None = None,
        stream: bool | None = None,
        idx: int | None = None,
        patches: list[str] | None = None,
    ) -> None:
        self.adapter = adapter
        self.stream = bool(kwargs.get("stream") is True) if stream is None else stream
        normalized = adapter.normalize_request(kwargs)
        model = (normalized.get("params") or {}).get("model")
        self.pending = begin_step(
            "chat",
            str(model) if model else "unknown-model",
            normalized,
            attributes=_request_attributes(adapter, normalized, server, self.stream),
            raw_request=N.strip_binary(N.to_jsonable(kwargs)),
            idx=idx,
        )
        if self.pending is not None and self.stream:
            self.pending.defer_breaker = True
        if self.pending is None:
            _warn_outside_run()
        elif patches:
            self.pending.source = "patched"
            self.pending.attributes["crashlab.fork.patches"] = list(patches)

    @property
    def active(self) -> bool:
        return self.pending is not None and not self.pending.done

    def fail(self, exc: BaseException) -> None:
        if self.active:
            self.pending.fail(exc)  # type: ignore[union-attr]

    def finish(self, resp: Any) -> None:
        if not self.active:
            return
        norm, raw = self.adapter.normalize_response(resp)
        attrs = _response_attributes(norm)
        attrs["crashlab.response.type"] = f"{type(resp).__module__}.{type(resp).__qualname__}"
        self.pending.finish(  # type: ignore[union-attr]
            norm, raw_response=N.strip_binary(raw), attributes=attrs
        )

    # replay ----------------------------------------------------------------------------
    def replay_source(self) -> Step | None:
        return self.pending.replay_source() if self.pending is not None else None

    def serve_replay(self, src: Step, *, is_async: bool, raw: bool = False) -> Any:
        """Finish this step from the recording and return what the live call returned.

        ``raw``: the call went through ``with_raw_response.create``, so the caller expects
        a raw-response object (``.parse()``, ``.headers``) around the completion.
        """
        from ..replay import rebuild as R
        from ..replay import synth

        self.pending.finish_from(src)  # type: ignore[union-attr]
        error = R.rebuild_exception(src.error) if src.error is not None else None
        api = self.adapter.api
        if self.stream:
            chunk_type = src.attributes.get("crashlab.stream.chunk_type") or (
                R.DEFAULT_CHUNK_TYPES.get(api)
            )
            items = synth.synthesize(api, src.raw_response, chunk_type)
            if error is not None and not items:
                return _raise_later(error) if is_async else _raise(error)
            stream_obj: Any = (
                synth.AsyncReplayStream(items, error)
                if is_async
                else synth.ReplayStream(items, error)
            )
            if raw:
                stream_obj = R.ReplayRawResponse(stream_obj, src.raw_response)
            # `await create(stream=True)` returns the stream
            return _return_later(stream_obj) if is_async else stream_obj
        if error is not None:
            return _raise_later(error) if is_async else _raise(error)
        resp_type = src.attributes.get("crashlab.response.type") or R.DEFAULT_RESPONSE_TYPES.get(
            api
        )
        obj = R.rebuild(src.raw_response or {}, resp_type)
        if raw:
            obj = R.ReplayRawResponse(obj, src.raw_response)
        return _return_later(obj) if is_async else obj

    def finish_stream(
        self, asm: Assembler, *, error: BaseException | None = None, complete: bool = False
    ) -> None:
        if not self.active:
            return
        raw = asm.result()
        norm, raw = self.adapter.normalize_response(raw) if raw else (None, None)
        attrs = _response_attributes(norm) if norm else {}
        attrs["crashlab.stream.chunks"] = asm.chunks
        if asm.chunk_type:
            attrs["crashlab.stream.chunk_type"] = asm.chunk_type
        attrs["crashlab.stream.complete"] = bool(complete and error is None)
        raw = N.strip_binary(raw) if raw else None
        if error is not None and not isinstance(error, GeneratorExit):
            self.pending.fail(  # type: ignore[union-attr]
                error, response=norm, raw_response=raw, attributes=attrs
            )
        else:
            self.pending.finish(norm, raw_response=raw, attributes=attrs)  # type: ignore[union-attr]


def _parsed_for_record(result: Any) -> Any:
    """OpenAI ``with_raw_response.create`` returns a LegacyAPIResponse; record the parsed
    completion while still returning the raw object to the caller (CrewAI needs it)."""
    parse = getattr(result, "parse", None)
    if callable(parse) and hasattr(result, "status_code") and hasattr(result, "headers"):
        try:
            return parse()
        except Exception:  # noqa: BLE001 - fall back to recording the raw object
            return result
    return result


def _wrap_result(call: ChatCall, result: Any, *, is_async: bool) -> Any:
    if call.stream and call.active:
        asm = call.adapter.new_assembler()
        if is_async or (hasattr(result, "__aiter__") and not hasattr(result, "__iter__")):
            return AsyncRecordingStream(result, call, asm)
        return RecordingStream(result, call, asm)
    call.finish(_parsed_for_record(result))
    return result


def _raise(exc: BaseException) -> Any:
    raise exc


async def _raise_later(exc: BaseException) -> Any:
    raise exc


async def _return_later(value: Any) -> Any:
    return value


def prepare_request(
    adapter: ChatAdapter, kwargs: dict[str, Any]
) -> tuple[dict[str, Any], int | None, list[str]]:
    """In a fork past its fork step: allocate the step index and apply the patches to the
    outgoing request. Returns ``(kwargs, idx or None, descriptions of applied patches)``."""
    ctx = current()
    rp = ctx.replayer if ctx is not None else None
    if rp is None or not rp.edits_requests:
        return kwargs, None, []
    idx = ctx.next_idx()  # type: ignore[union-attr]
    new, applied = rp.patch_chat(idx, adapter.api, kwargs)
    return new, idx, applied


def invoke(
    adapter: ChatAdapter,
    fn: Callable[..., Any],
    args: tuple,
    kwargs: dict[str, Any],
    *,
    server: str | None = None,
    bind_to: Callable[..., Any] | None = None,
    is_async: bool | None = None,
    raw: bool = False,
) -> Any:
    """Call ``fn(*args, **kwargs)`` and record it. Works for sync and async callables.

    In replay mode ``fn`` is not called: the recorded response is returned (awaitable when
    ``is_async``; SDK async methods are not detectable as coroutine functions, so the
    proxies pass it explicitly).
    """
    call_kwargs = dict(kwargs)
    if args and bind_to is not None:  # positional args (e.g. litellm.completion(model, msgs))
        try:
            bound = inspect.signature(bind_to).bind_partial(*args, **kwargs)
            call_kwargs = dict(bound.arguments)
            call_kwargs.update(call_kwargs.pop("kwargs", {}) or {})
        except (TypeError, ValueError):
            pass
    call_kwargs, idx, applied = prepare_request(adapter, call_kwargs)
    if applied:  # call the SDK with the patched request
        args, kwargs = (), call_kwargs
    call = ChatCall(adapter, call_kwargs, server=server, idx=idx, patches=applied)
    src = call.replay_source()
    if src is not None:
        if is_async is None:
            is_async = inspect.iscoroutinefunction(fn)
        return call.serve_replay(src, is_async=is_async, raw=raw)
    try:
        result = fn(*args, **kwargs)
    except BaseException as exc:
        call.fail(exc)
        raise
    if inspect.isawaitable(result):

        async def _awaited() -> Any:
            try:
                resp = await result
            except BaseException as exc:
                call.fail(exc)
                raise
            return _wrap_result(call, resp, is_async=True)

        return _awaited()
    return _wrap_result(call, result, is_async=False)
