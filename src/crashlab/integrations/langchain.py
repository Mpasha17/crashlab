"""``CrashlabCallbackHandler``: record LangChain / LangGraph calls into the active run.

Zero hard dependency: if ``langchain_core.callbacks.BaseCallbackHandler`` (or the
legacy ``langchain.callbacks.base`` class) is importable, the handler subclasses it so
it can be passed as a callback; otherwise it is a plain duck-typed object with the same
methods. ``import crashlab.integrations.langchain`` never requires LangChain.

The handler captures:

- ``on_chat_model_start`` / ``on_llm_start`` → ``on_llm_end`` / ``on_llm_error``
- ``on_tool_start`` → ``on_tool_end`` / ``on_tool_error``
- ``on_chain_start`` / ``on_chain_end`` / ``on_chain_error``: when
  ``metadata['langgraph_node']`` or the chain name is present, steps recorded inside
  that chain (including by ``crashlab.wrap()``) are labelled with
  :func:`crashlab.agent_scope`.

If the underlying OpenAI / Anthropic client is already wrapped with ``crashlab.wrap()``,
the LLM or tool call is already a step by the time the end callback fires, and the
handler does not record a second copy.

Limits. A callback cannot cancel a provider call, so offline ``crashlab.replay`` still
needs ``crashlab.wrap`` on the client (or a fake). The handler alone is enough to
inspect a live ``crashlab.record()`` run.
"""

from __future__ import annotations

import time
from typing import Any

from ..capture._steps import begin_step
from ..capture.normalize import to_jsonable
from ..context import agent_scope, current
from ..models import utcnow

_GENERIC_CHAINS = frozenset(
    {
        "LangGraph",
        "RunnableSequence",
        "RunnableLambda",
        "RunnableParallel",
        "RunnablePassthrough",
        "RunnableBinding",
        "RunnableEach",
        "CompiledStateGraph",
        "CompiledGraph",
        "AgentExecutor",
        "LLMChain",
        "ChatPromptTemplate",
        "PromptTemplate",
    }
)


def _handler_base() -> type:
    try:
        from langchain_core.callbacks import BaseCallbackHandler

        return BaseCallbackHandler
    except ImportError:
        pass
    try:
        from langchain.callbacks.base import BaseCallbackHandler

        return BaseCallbackHandler
    except ImportError:
        return object


def _ctx() -> Any:
    return current()


def _live_steps(ctx: Any) -> list[Any]:
    # During a run, finished steps live on the context; run.steps is filled when saved.
    return ctx.steps if getattr(ctx, "steps", None) is not None else ctx.run.steps


def _last_idx(ctx: Any) -> int:
    steps = _live_steps(ctx)
    return steps[-1].idx if steps else -1


def _new_ops(ctx: Any, last_idx: int, operation: str) -> list[Any]:
    return [s for s in _live_steps(ctx) if s.idx > last_idx and s.operation == operation]


def _role(message: Any) -> str:
    if isinstance(message, dict):
        typ = message.get("type") or message.get("role") or "user"
    else:
        typ = getattr(message, "type", None) or getattr(message, "role", None) or "user"
    return {"human": "user", "ai": "assistant", "system": "system", "tool": "tool",
            "developer": "system"}.get(str(typ), str(typ))  # fmt: skip


def _text(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", message)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            else:
                parts.append(str(getattr(item, "text", item)))
        return "".join(parts)
    return str(content)


def _as_message(message: Any) -> dict[str, Any]:
    role = _role(message)
    body = _text(message)
    msg: dict[str, Any] = {"role": role, "parts": [{"type": "text", "content": body}]}
    name = message.get("name") if isinstance(message, dict) else getattr(message, "name", None)
    if name:
        msg["name"] = name
    tool_call_id = (
        message.get("tool_call_id")
        if isinstance(message, dict)
        else getattr(message, "tool_call_id", None)
    )
    if tool_call_id:
        msg["tool_call_id"] = tool_call_id
    return msg


def _flatten_messages(messages: Any) -> list[Any]:
    if not messages:
        return []
    # Chat model callbacks pass list[list[BaseMessage]] (one list per generation)
    first = messages[0] if isinstance(messages, list) and messages else None
    if first is not None and isinstance(first, list):
        return list(first)
    if isinstance(messages, list):
        return list(messages)
    return [messages]


def _split_system(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    rest: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            texts = [p.get("content", "") for p in msg.get("parts") or [] if isinstance(p, dict)]
            system_parts.append("".join(texts))
        else:
            rest.append(msg)
    system = "\n\n".join(p for p in system_parts if p) or None
    return system, rest


def _model_of(serialized: Any, invocation_params: Any, response: Any = None) -> str:
    params = invocation_params if isinstance(invocation_params, dict) else {}
    for key in ("model", "model_name", "model_id"):
        value = params.get(key)
        if isinstance(value, str) and value:
            return value
    if isinstance(serialized, dict):
        kwargs = serialized.get("kwargs") if isinstance(serialized.get("kwargs"), dict) else {}
        for key in ("model", "model_name", "model_id"):
            value = kwargs.get(key) or serialized.get(key)
            if isinstance(value, str) and value:
                return value
        ident = serialized.get("id")
        if isinstance(ident, list) and ident and isinstance(ident[-1], str):
            return ident[-1]
    if response is not None:
        out = getattr(response, "llm_output", None) or {}
        if isinstance(out, dict):
            for key in ("model_name", "model"):
                value = out.get(key)
                if isinstance(value, str) and value:
                    return value
    return "langchain"


def _params(invocation_params: Any, model: str) -> dict[str, Any]:
    params: dict[str, Any] = {"model": model}
    if not isinstance(invocation_params, dict):
        return params
    for key, value in invocation_params.items():
        if key in {"model", "model_name", "model_id", "messages", "tools", "prompts"}:
            continue
        if key.startswith("_"):
            continue
        try:
            params[key] = to_jsonable(value)
        except Exception:
            continue
    return params


def _tools(invocation_params: Any) -> list[Any]:
    if not isinstance(invocation_params, dict):
        return []
    tools = invocation_params.get("tools")
    if not tools:
        return []
    try:
        from ..capture.normalize import normalize_tools

        return normalize_tools(tools)
    except Exception:
        return to_jsonable(tools) or []


def _usage(response: Any) -> dict[str, Any]:
    out = getattr(response, "llm_output", None) or {}
    if not isinstance(out, dict):
        return {}
    usage = out.get("token_usage") or out.get("usage") or {}
    if not isinstance(usage, dict):
        return {}
    inp = usage.get("prompt_tokens") or usage.get("input_tokens")
    tok = usage.get("completion_tokens") or usage.get("output_tokens")
    total = usage.get("total_tokens")
    attrs: dict[str, Any] = {}
    if inp is not None:
        attrs["gen_ai.usage.input_tokens"] = int(inp)
    if tok is not None:
        attrs["gen_ai.usage.output_tokens"] = int(tok)
    if total is not None:
        attrs["crashlab.usage.total_tokens"] = int(total)
    return attrs


def _generations(response: Any) -> list[Any]:
    gens = getattr(response, "generations", None) or []
    flat: list[Any] = []
    for row in gens:
        if isinstance(row, list):
            flat.extend(row)
        else:
            flat.append(row)
    return flat


def _tool_calls(message: Any) -> list[dict[str, Any]]:
    raw = getattr(message, "tool_calls", None) if message is not None else None
    if not raw:
        add = getattr(message, "additional_kwargs", None) or {}
        raw = add.get("tool_calls") if isinstance(add, dict) else None
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    for i, call in enumerate(raw):
        if isinstance(call, dict):
            name = call.get("name") or (call.get("function") or {}).get("name")
            args = (
                call.get("args")
                or call.get("arguments")
                or (call.get("function") or {}).get("arguments")
            )
            cid = call.get("id") or call.get("tool_call_id")
        else:
            name = getattr(call, "name", None)
            args = getattr(call, "args", None) or getattr(call, "arguments", None)
            cid = getattr(call, "id", None)
        if isinstance(args, str):
            try:
                import json

                args = json.loads(args)
            except (TypeError, ValueError):
                pass
        entry: dict[str, Any] = {"name": name, "arguments": to_jsonable(args)}
        if cid:
            entry["id"] = cid
        elif name:
            entry["id"] = f"call_{i}"
        out.append(entry)
    return out


def _normalize_llm_response(response: Any, model: str) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    finish = None
    for gen in _generations(response):
        message = getattr(gen, "message", None)
        text = _text(message) if message is not None else getattr(gen, "text", "") or ""
        info = getattr(gen, "generation_info", None) or {}
        if isinstance(info, dict) and info.get("finish_reason"):
            finish = info.get("finish_reason")
        msg: dict[str, Any] = {"role": "assistant", "parts": [{"type": "text", "content": text}]}
        messages.append(msg)
        calls.extend(_tool_calls(message))
    if not messages and hasattr(response, "content"):
        messages = [{"role": "assistant", "parts": [{"type": "text", "content": _text(response)}]}]
    out: dict[str, Any] = {"messages": messages, "model": model}
    if calls:
        out["tool_calls"] = calls
        out["finish_reason"] = finish or "tool_calls"
    elif finish:
        out["finish_reason"] = finish
    usage = getattr(response, "llm_output", None) or {}
    if isinstance(usage, dict) and (usage.get("token_usage") or usage.get("usage")):
        raw = usage.get("token_usage") or usage.get("usage")
        if isinstance(raw, dict):
            out["usage"] = to_jsonable(raw)
    return out


def _chain_label(serialized: Any, metadata: Any, name: Any, kwargs: dict[str, Any]) -> str | None:
    meta = metadata if isinstance(metadata, dict) else {}
    node = meta.get("langgraph_node")
    if isinstance(node, str) and node.strip():
        return node.strip()
    label = name or kwargs.get("name")
    if not label and isinstance(serialized, dict):
        label = serialized.get("name")
        if not label:
            ident = serialized.get("id")
            if isinstance(ident, list) and ident:
                label = ident[-1]
    if isinstance(label, str) and label.strip() and label not in _GENERIC_CHAINS:
        return label.strip()
    return None


def _tool_name(serialized: Any, kwargs: dict[str, Any]) -> str:
    if isinstance(serialized, dict):
        name = serialized.get("name") or (serialized.get("id") or [None])[-1]
        if isinstance(name, str) and name:
            return name
    name = kwargs.get("name")
    return name if isinstance(name, str) and name else "tool"


def _tool_args(input_str: Any, inputs: Any) -> dict[str, Any]:
    if isinstance(inputs, dict):
        return {str(k): to_jsonable(v) for k, v in inputs.items()}
    if isinstance(input_str, dict):
        return {str(k): to_jsonable(v) for k, v in input_str.items()}
    if input_str is None or input_str == "":
        return {}
    return {"input": input_str if isinstance(input_str, str) else to_jsonable(input_str)}


def _tool_result(output: Any) -> Any:
    if output is None or isinstance(output, (str, int, float, bool, dict, list)):
        return to_jsonable(output)
    content = getattr(output, "content", None)
    if content is not None:
        return to_jsonable(content)
    return to_jsonable(output)


class CrashlabCallbackHandler(_handler_base()):
    """LangChain / LangGraph callback handler that records into the active crashlab run."""

    name = "crashlab"

    def __init__(self, **kwargs: Any) -> None:
        base = _handler_base()
        if base is not object:
            try:
                super().__init__(**kwargs)
            except TypeError:
                super().__init__()
        self._pending: dict[str, dict[str, Any]] = {}
        self._scopes: dict[str, Any] = {}

    def _key(self, run_id: Any) -> str:
        return str(run_id)

    def _open_scope(self, run_id: Any, label: str) -> None:
        key = self._key(run_id)
        self._close_scope(run_id)
        cm = agent_scope(label)
        cm.__enter__()
        self._scopes[key] = cm

    def _close_scope(self, run_id: Any) -> None:
        cm = self._scopes.pop(self._key(run_id), None)
        if cm is not None:
            cm.__exit__(None, None, None)

    def _remember(self, run_id: Any, kind: str, **data: Any) -> None:
        ctx = _ctx()
        if ctx is None:
            return
        payload = {
            "kind": kind,
            "last_idx": _last_idx(ctx),
            "t0": time.perf_counter(),
            "started_at": utcnow(),
        }
        payload.update(data)
        self._pending[self._key(run_id)] = payload

    def _take(self, run_id: Any, kind: str) -> dict[str, Any] | None:
        pending = self._pending.pop(self._key(run_id), None)
        if pending is None or pending.get("kind") != kind:
            if pending is not None:
                self._pending[self._key(run_id)] = pending
            return None
        return pending

    def _skip(self, ctx: Any, pending: dict[str, Any], operation: str) -> bool:
        return bool(_new_ops(ctx, pending["last_idx"], operation))

    def on_chat_model_start(
        self,
        serialized: Any,
        messages: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: Any = None,
        metadata: Any = None,
        **kwargs: Any,
    ) -> None:
        self._remember(
            run_id,
            "llm",
            messages=_flatten_messages(messages),
            serialized=serialized,
            invocation_params=kwargs.get("invocation_params"),
        )

    def on_llm_start(
        self,
        serialized: Any,
        prompts: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: Any = None,
        metadata: Any = None,
        **kwargs: Any,
    ) -> None:
        if self._key(run_id) in self._pending:
            return
        self._remember(
            run_id,
            "llm",
            prompts=list(prompts or []),
            serialized=serialized,
            invocation_params=kwargs.get("invocation_params"),
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        pending = self._take(run_id, "llm")
        ctx = _ctx()
        if pending is None or ctx is None or self._skip(ctx, pending, "chat"):
            return
        inv = pending.get("invocation_params")
        model = _model_of(pending.get("serialized"), inv, response)
        if pending.get("messages") is not None:
            msgs = [_as_message(m) for m in pending["messages"]]
            system, messages = _split_system(msgs)
        else:
            system, messages = (
                None,
                [
                    {"role": "user", "parts": [{"type": "text", "content": p}]}
                    for p in pending.get("prompts") or []
                ],
            )
        request = {
            "system": system,
            "messages": messages,
            "tools": _tools(inv),
            "params": _params(inv, model),
        }
        step = begin_step(
            "chat",
            model,
            request,
            attributes={
                "gen_ai.request.model": model,
                "gen_ai.provider.name": "langchain",
                "crashlab.api": "langchain.callback",
            },
        )
        if step is None:
            return
        step.started_at = pending["started_at"]
        step.t0 = pending["t0"]
        attrs = _usage(response)
        step.finish(_normalize_llm_response(response, model), attributes=attrs or None)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        pending = self._take(run_id, "llm")
        ctx = _ctx()
        if pending is None or ctx is None or self._skip(ctx, pending, "chat"):
            return
        inv = pending.get("invocation_params")
        model = _model_of(pending.get("serialized"), inv)
        if pending.get("messages") is not None:
            msgs = [_as_message(m) for m in pending["messages"]]
            system, messages = _split_system(msgs)
        else:
            system, messages = (
                None,
                [
                    {"role": "user", "parts": [{"type": "text", "content": p}]}
                    for p in pending.get("prompts") or []
                ],
            )
        request = {
            "system": system,
            "messages": messages,
            "tools": _tools(inv),
            "params": _params(inv, model),
        }
        step = begin_step(
            "chat",
            model,
            request,
            attributes={
                "gen_ai.request.model": model,
                "gen_ai.provider.name": "langchain",
                "crashlab.api": "langchain.callback",
            },
        )
        if step is None:
            return
        step.started_at = pending["started_at"]
        step.t0 = pending["t0"]
        step.fail(error)

    def on_tool_start(
        self,
        serialized: Any,
        input_str: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: Any = None,
        metadata: Any = None,
        inputs: Any = None,
        **kwargs: Any,
    ) -> None:
        name = _tool_name(serialized, kwargs)
        self._remember(
            run_id,
            "tool",
            name=name,
            args=_tool_args(input_str, inputs),
            serialized=serialized,
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        pending = self._take(run_id, "tool")
        ctx = _ctx()
        if pending is None or ctx is None or self._skip(ctx, pending, "execute_tool"):
            return
        name = pending.get("name") or "tool"
        args = pending.get("args") or {}
        step = begin_step(
            "execute_tool",
            name,
            {"args": args},
            attributes={
                "gen_ai.tool.name": name,
                "gen_ai.tool.type": "function",
                "crashlab.api": "langchain.callback",
            },
            hash_input={"tool": name, "args": args},
        )
        if step is None:
            return
        step.started_at = pending["started_at"]
        step.t0 = pending["t0"]
        step.finish({"result": _tool_result(output)})

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        pending = self._take(run_id, "tool")
        ctx = _ctx()
        if pending is None or ctx is None or self._skip(ctx, pending, "execute_tool"):
            return
        name = pending.get("name") or "tool"
        args = pending.get("args") or {}
        step = begin_step(
            "execute_tool",
            name,
            {"args": args},
            attributes={
                "gen_ai.tool.name": name,
                "gen_ai.tool.type": "function",
                "crashlab.api": "langchain.callback",
            },
            hash_input={"tool": name, "args": args},
        )
        if step is None:
            return
        step.started_at = pending["started_at"]
        step.t0 = pending["t0"]
        step.fail(error, is_tool_error=True)

    def on_chain_start(
        self,
        serialized: Any,
        inputs: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: Any = None,
        metadata: Any = None,
        **kwargs: Any,
    ) -> None:
        label = _chain_label(serialized, metadata, kwargs.get("name"), kwargs)
        if label:
            self._open_scope(run_id, label)

    def on_chain_end(
        self,
        outputs: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        self._close_scope(run_id)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        self._close_scope(run_id)


__all__ = ["CrashlabCallbackHandler"]
