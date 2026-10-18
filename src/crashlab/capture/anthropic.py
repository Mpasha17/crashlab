"""Anthropic client proxy (planned; not implemented in v0.0.1)."""

from __future__ import annotations

from typing import Any


def wrap_anthropic(client: Any) -> Any:
    raise NotImplementedError(
        "crashlab.wrap(): Anthropic clients are not supported yet in crashlab 0.0.1. "
        "Support for anthropic.Anthropic / AsyncAnthropic messages.create is planned for the "
        "next release. For now you can call Claude through an OpenAI-compatible endpoint and "
        "wrap an openai client instead."
    )
