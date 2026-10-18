"""Provider-independent recording of one LLM call (plain or streamed).

Each provider supplies a :class:`ChatAdapter`; :func:`invoke` is the single interception
point used by every proxy (OpenAI chat + responses, Anthropic, LiteLLM). The replay
engine will plug in here to serve responses from the cassette instead of calling ``fn``.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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
        )
        if self.pending is None:
            _warn_outside_run()

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
        self.pending.finish(  # type: ignore[union-attr]
            norm, raw_response=N.strip_binary(raw), attributes=_response_attributes(norm)
        )

    def finish_stream(
        self, asm: Assembler, *, error: BaseException | None = None, complete: bool = False
    ) -> None:
        if not self.active:
            return
        raw = asm.result()
        norm, raw = self.adapter.normalize_response(raw) if raw else (None, None)
        attrs = _response_attributes(norm) if norm else {}
        attrs["crashlab.stream.chunks"] = asm.chunks
        attrs["crashlab.stream.complete"] = bool(complete and error is None)
        raw = N.strip_binary(raw) if raw else None
        if error is not None and not isinstance(error, GeneratorExit):
            self.pending.fail(  # type: ignore[union-attr]
                error, response=norm, raw_response=raw, attributes=attrs
            )
        else:
            self.pending.finish(norm, raw_response=raw, attributes=attrs)  # type: ignore[union-attr]


def _wrap_result(call: ChatCall, result: Any, *, is_async: bool) -> Any:
    if call.stream and call.active:
        asm = call.adapter.new_assembler()
        if is_async or (hasattr(result, "__aiter__") and not hasattr(result, "__iter__")):
            return AsyncRecordingStream(result, call, asm)
        return RecordingStream(result, call, asm)
    call.finish(result)
    return result


def invoke(
    adapter: ChatAdapter,
    fn: Callable[..., Any],
    args: tuple,
    kwargs: dict[str, Any],
    *,
    server: str | None = None,
    bind_to: Callable[..., Any] | None = None,
) -> Any:
    """Call ``fn(*args, **kwargs)`` and record it. Works for sync and async callables."""
    call_kwargs = dict(kwargs)
    if args and bind_to is not None:  # positional args (e.g. litellm.completion(model, msgs))
        try:
            bound = inspect.signature(bind_to).bind_partial(*args, **kwargs)
            call_kwargs = dict(bound.arguments)
            call_kwargs.update(call_kwargs.pop("kwargs", {}) or {})
        except (TypeError, ValueError):
            pass
    call = ChatCall(adapter, call_kwargs, server=server)
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
