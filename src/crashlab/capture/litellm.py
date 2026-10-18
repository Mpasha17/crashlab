"""LiteLLM integration (planned; not implemented in v0.0.1)."""

from __future__ import annotations

from typing import Any


def patch() -> None:
    raise NotImplementedError(
        "crashlab LiteLLM support (patching litellm.completion / acompletion) is not "
        "implemented yet in crashlab 0.0.1. Use an OpenAI client pointed at a LiteLLM proxy "
        "or any OpenAI-compatible endpoint and wrap it with crashlab.wrap()."
    )


def wrap_litellm(obj: Any) -> Any:
    patch()
