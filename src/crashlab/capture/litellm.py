"""LiteLLM integration: ``crashlab.integrations.litellm.patch()`` / ``unpatch()``.

``patch()`` replaces ``litellm.completion`` and ``litellm.acompletion`` with recording
wrappers (plain and ``stream=True``). Calls go through the same interception point as the
client proxies, so ``crashlab.replay`` serves them from the cassette too.

Code that did ``from litellm import completion`` *before* ``patch()`` keeps a reference to
the original function and is not recorded; call ``litellm.completion(...)`` or patch first.
"""

from __future__ import annotations

import functools
import threading
from typing import Any

from . import normalize as N
from ._chat import ChatAdapter, invoke
from .streams import OpenAIChatAssembler

_lock = threading.Lock()
_originals: dict[str, Any] = {}
_module: Any = None


def _provider(kwargs: dict[str, Any]) -> str:
    if kwargs.get("custom_llm_provider"):
        return str(kwargs["custom_llm_provider"])
    model = kwargs.get("model")
    if isinstance(model, str) and "/" in model:
        return model.split("/", 1)[0]
    return "litellm"


def _adapter_for(kwargs: dict[str, Any]) -> ChatAdapter:
    return ChatAdapter(
        provider=_provider(kwargs),
        api="litellm.completion",
        normalize_request=N.normalize_openai_chat_request,
        normalize_response=N.normalize_openai_chat_response,
        new_assembler=OpenAIChatAssembler,
    )


def _make(orig: Any, is_async: bool) -> Any:
    if is_async:

        @functools.wraps(orig)
        async def acompletion(*args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model", args[0] if args else None)
            adapter = _adapter_for({**kwargs, "model": model})
            return await invoke(adapter, orig, args, kwargs, bind_to=orig, is_async=True)

        acompletion.__crashlab_patched__ = True  # type: ignore[attr-defined]
        return acompletion

    @functools.wraps(orig)
    def completion(*args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model", args[0] if args else None)
        adapter = _adapter_for({**kwargs, "model": model})
        return invoke(adapter, orig, args, kwargs, bind_to=orig, is_async=False)

    completion.__crashlab_patched__ = True  # type: ignore[attr-defined]
    return completion


def _import_litellm() -> Any:
    try:
        import litellm
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "crashlab LiteLLM integration needs litellm: pip install 'crashlab[litellm]'"
        ) from exc
    return litellm


def patch(module: Any = None) -> None:
    """Start recording ``litellm.completion`` / ``litellm.acompletion`` calls (idempotent)."""
    global _module
    mod = module if module is not None else _import_litellm()
    with _lock:
        if _originals:
            if _module is mod:
                return
            raise RuntimeError("crashlab: a different litellm module is already patched")
        for name, is_async in (("completion", False), ("acompletion", True)):
            orig = getattr(mod, name, None)
            if orig is None:
                continue
            _originals[name] = orig
            setattr(mod, name, _make(orig, is_async))
        _module = mod


def unpatch() -> None:
    """Restore the original litellm functions."""
    global _module
    with _lock:
        if _module is not None:
            for name, orig in _originals.items():
                setattr(_module, name, orig)
        _originals.clear()
        _module = None


def is_patched() -> bool:
    return bool(_originals)
