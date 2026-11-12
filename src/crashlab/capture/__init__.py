"""Capture layer: client proxies, tool and agent decorators."""

from __future__ import annotations

from typing import Any

from .agent import agent, child_env, record
from .tools import clear_fakes, fake_tool, tool


def _module_root(obj: Any) -> str:
    return (type(obj).__module__ or "").split(".")[0]


def wrap(client: Any, *, provider: str | None = None, agent: str | None = None) -> Any:
    """Return a recording proxy for an LLM client.

    - ``openai.OpenAI`` / ``AsyncOpenAI`` and any OpenAI-compatible client (Ollama, vLLM,
      LiteLLM proxy, OpenRouter ...): ``chat.completions.create`` and ``responses.create``.
    - ``anthropic.Anthropic`` / ``AsyncAnthropic``: ``messages.create`` and
      ``messages.stream``.
    - the ``litellm`` module: same as ``crashlab.integrations.litellm.patch()``.

    ``provider`` forces the adapter (``"openai"``, ``"anthropic"``, ``"litellm"``); other
    values are treated as OpenAI-compatible and used as ``gen_ai.provider.name``.

    ``agent`` names the agent that owns this client in a multi-agent app (one client per
    agent, as OpenAI Agents SDK, AutoGen, CrewAI and PydanticAI agents allow): every
    call it makes is recorded with ``gen_ai.agent.name``, and tool steps requested by
    those calls inherit it. For one client shared by several agents use
    :func:`crashlab.agent_scope`. Labels only: replay and request hashes are unchanged.
    """
    if agent is not None and (not isinstance(agent, str) or not agent.strip()):
        raise ValueError("crashlab.wrap(): agent= must be a non-empty string")
    if getattr(client, "__crashlab_wrapped__", False):
        if agent is not None and getattr(client, "_crashlab_agent", None) != agent:
            # re-label: wrap the original client again under the new name
            return wrap(client.unwrapped, provider=provider, agent=agent)
        return client
    root = provider or _module_root(client)
    if root == "litellm" or getattr(client, "__name__", None) == "litellm":
        from .litellm import patch

        if agent is not None:
            raise TypeError(
                "crashlab.wrap(litellm, agent=...): litellm is patched module-wide, so its "
                "calls cannot be named per client; use `with crashlab.agent_scope(name):`"
            )
        patch(client if getattr(client, "__name__", None) == "litellm" else None)
        return client

    from .anthropic import looks_like_anthropic, wrap_anthropic
    from .openai import looks_like_openai, wrap_openai

    if root == "anthropic" or (provider is None and not looks_like_openai(client)
                               and looks_like_anthropic(client)):  # fmt: skip
        return wrap_anthropic(client, agent=agent)
    if root == "openai" or provider is not None or looks_like_openai(client):
        return wrap_openai(client, provider=provider or "openai", agent=agent)
    raise TypeError(
        f"crashlab.wrap(): don't know how to wrap {type(client).__module__}."
        f"{type(client).__qualname__}. Supported: OpenAI-compatible clients "
        "(chat.completions.create or responses.create), Anthropic clients "
        "(messages.create), the litellm module."
    )


__all__ = ["agent", "child_env", "clear_fakes", "fake_tool", "record", "tool", "wrap"]
