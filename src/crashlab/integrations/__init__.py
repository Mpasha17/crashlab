"""Integrations with LLM libraries and protocols that are not plain clients.

``crashlab.instrument()`` / ``uninstrument()`` / ``instrumented()`` auto-patch
provider SDKs (OpenAI, Anthropic, LiteLLM, Google GenAI) so frameworks that
construct a client internally are recorded without a private ``_client`` hook.

``crashlab.integrations.litellm`` (``patch`` / ``unpatch``), ``.mcp``
(``wrap_mcp_session``), ``.langchain`` (``CrashlabCallbackHandler``) and
``.frameworks`` (OpenAI Agents SDK, PydanticAI, CrewAI). Optional packages are
imported only when the corresponding integration is used — and the frameworks
adapters never import ``agents``, ``pydantic_ai`` or ``crewai`` at all — so a
missing optional package never breaks ``import crashlab``.
"""

from __future__ import annotations

from typing import Any

from .frameworks import (
    CrewAICallback,
    OpenAIAgentsTracingProcessor,
    PydanticAIAdapter,
    crashlab_tracing_processor,
    wrap_crew,
    wrap_pydantic_ai_agent,
)
from .instrument import instrument, instrumented, uninstrument

__all__ = [
    "CrashlabCallbackHandler",
    "CrewAICallback",
    "OpenAIAgentsTracingProcessor",
    "PydanticAIAdapter",
    "crashlab_tracing_processor",
    "frameworks",
    "instrument",
    "instrumented",
    "langchain",
    "litellm",
    "mcp",
    "uninstrument",
    "wrap_crew",
    "wrap_mcp_session",
    "wrap_pydantic_ai_agent",
]


def __getattr__(name: str) -> Any:
    # Submodules (litellm, mcp, langchain, frameworks) are loaded by the import
    # system, not here: `from . import litellm` inside __getattr__ recurses.
    if name == "wrap_mcp_session":
        from crashlab.integrations.mcp import wrap_mcp_session as fn

        globals()[name] = fn
        return fn
    if name == "CrashlabCallbackHandler":
        from crashlab.integrations.langchain import CrashlabCallbackHandler as cls

        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
