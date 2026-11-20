"""Integrations with LLM libraries and protocols that are not plain clients.

``crashlab.instrument()`` / ``uninstrument()`` / ``instrumented()`` auto-patch
provider SDKs (OpenAI, Anthropic, LiteLLM, Google GenAI) so frameworks that
construct a client internally are recorded without a private ``_client`` hook.

``crashlab.integrations.litellm`` (``patch`` / ``unpatch``), ``.mcp``
(``wrap_mcp_session``) and ``.langchain`` (``CrashlabCallbackHandler``). Optional
packages are imported only when the corresponding integration is used, so a missing
``mcp`` or ``langchain_core`` never breaks ``import crashlab``.
"""

from __future__ import annotations

from typing import Any

from .instrument import instrument, instrumented, uninstrument

__all__ = [
    "CrashlabCallbackHandler",
    "instrument",
    "instrumented",
    "langchain",
    "litellm",
    "mcp",
    "uninstrument",
    "wrap_mcp_session",
]


def __getattr__(name: str) -> Any:
    # Submodules (litellm, mcp, langchain) are loaded by the import system, not here:
    # `from . import litellm` inside __getattr__ recurses.
    if name == "wrap_mcp_session":
        from crashlab.integrations.mcp import wrap_mcp_session as fn

        globals()[name] = fn
        return fn
    if name == "CrashlabCallbackHandler":
        from crashlab.integrations.langchain import CrashlabCallbackHandler as cls

        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
