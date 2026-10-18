"""Capture layer: client proxies, tool and agent decorators."""

from __future__ import annotations

from typing import Any

from .agent import agent, record
from .tools import tool


def _module_root(obj: Any) -> str:
    return (type(obj).__module__ or "").split(".")[0]


def wrap(client: Any, *, provider: str | None = None) -> Any:
    """Return a recording proxy for an LLM client.

    Supported in v0.0.1: ``openai.OpenAI`` / ``openai.AsyncOpenAI`` and any object with an
    OpenAI-compatible ``chat.completions.create`` (e.g. clients pointed at Ollama, vLLM,
    LiteLLM proxy or OpenRouter). Anthropic and LiteLLM module wrappers are not
    implemented yet and raise ``NotImplementedError``.
    """
    if getattr(client, "__crashlab_wrapped__", False):
        return client
    root = provider or _module_root(client)
    if root == "anthropic":
        from .anthropic import wrap_anthropic

        return wrap_anthropic(client)
    if root == "litellm" or getattr(client, "__name__", None) == "litellm":
        from .litellm import wrap_litellm

        return wrap_litellm(client)

    from .openai import looks_like_openai, wrap_openai

    if root == "openai" or provider is not None or looks_like_openai(client):
        return wrap_openai(client, provider=provider or "openai")
    raise TypeError(
        f"crashlab.wrap(): don't know how to wrap {type(client).__module__}."
        f"{type(client).__qualname__}. Supported: OpenAI-compatible clients "
        "(client.chat.completions.create)."
    )


__all__ = ["agent", "record", "tool", "wrap"]
