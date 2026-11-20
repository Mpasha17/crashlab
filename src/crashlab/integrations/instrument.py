"""Zero-config auto-instrumentation of LLM provider SDKs.

``crashlab.instrument()`` patches ``openai.OpenAI`` / ``AsyncOpenAI``,
``anthropic.Anthropic`` / ``AsyncAnthropic``, ``google.genai.Client`` and
``litellm.completion`` / ``acompletion`` so a client constructed *inside* a
framework (Haystack ``warm_up``, AutoGen, CrewAI, LlamaIndex, PydanticAI, the
OpenAI Agents SDK) is recorded without reaching for a private ``_client``.

Optional SDKs are never imported at module import time. ``instrument()`` patches
modules already in ``sys.modules``, tries ``importlib`` for requested providers,
and installs a post-import hook so a later ``import openai`` is patched too.
``uninstrument()`` restores the original ``__init__`` / LiteLLM hooks (reference
counted, so nested ``with crashlab.instrument():`` blocks unwind cleanly).
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import importlib.abc
import importlib.machinery
import sys
import threading
from collections.abc import Iterable
from types import ModuleType
from typing import Any

PROVIDERS: tuple[str, ...] = ("openai", "anthropic", "litellm", "google_genai")

# provider name -> fully-qualified module to patch
_MODULE_NAME: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "litellm": "litellm",
    "google_genai": "google.genai",
}

# class names on each module whose ``__init__`` should wrap the instance in-place
_INIT_CLASSES: dict[str, tuple[str, ...]] = {
    "openai": ("OpenAI", "AsyncOpenAI", "AzureOpenAI", "AsyncAzureOpenAI"),
    "anthropic": ("Anthropic", "AsyncAnthropic"),
    "google_genai": ("Client", "AsyncClient"),
}

_lock = threading.RLock()
_refcount: dict[str, int] = {name: 0 for name in PROVIDERS}
# (providers, agent) for each live instrument() call, innermost last
_sessions: list[tuple[tuple[str, ...], str | None]] = []
# provider -> {class_name: original __init__}
_original_inits: dict[str, dict[str, Any]] = {}
# LiteLLM: only unpatch on the way down if we were the ones who patched
_litellm_owned: bool = False
_hook: _PostImportFinder | None = None


def instrument(
    providers: str | Iterable[str] | None = None,
    *,
    agent: str | None = None,
    store: Any | None = None,
) -> InstrumentHandle:
    """Start recording clients constructed by provider SDKs (idempotent, nestable).

    ``providers`` defaults to all supported names (``openai``, ``anthropic``,
    ``litellm``, ``google_genai``). Pass a subset to patch only those. ``agent``
    is baked into clients constructed while this call is active (the same as
    ``crashlab.wrap(client, agent=...)``); an innermost :func:`crashlab.agent_scope`
    still labels calls when ``agent`` is omitted.

    ``store`` is accepted and kept on the returned handle; recording still starts
    with :func:`crashlab.record` or ``@crashlab.agent`` as usual.

    The return value is a context manager: ``with crashlab.instrument():``
    restores on exit. Calling :func:`uninstrument` (or ``handle.close()``) does
    the same. Nested calls use a per-provider reference count, so an inner
    block exiting does not unpatch an outer one.
    """
    if agent is not None and (not isinstance(agent, str) or not agent.strip()):
        raise ValueError("crashlab.instrument(): agent= must be a non-empty string")
    names = _normalize_providers(providers)
    with _lock:
        _sessions.append((names, agent))
        _install_hook()
        for name in names:
            _refcount[name] += 1
            if _refcount[name] == 1:
                _patch_provider(name)
    return InstrumentHandle(names, agent=agent, store=store)


def uninstrument(providers: str | Iterable[str] | None = None) -> None:
    """Restore original ``__init__`` / LiteLLM hooks for ``providers`` (default: all).

    Safe to call when nothing is instrumented. A provider is only actually
    unpatched when its reference count reaches zero.
    """
    names = _normalize_providers(providers)
    with _lock:
        # drop the innermost session that covers these providers, else just count
        _pop_session(names)
        for name in names:
            if _refcount[name] <= 0:
                _refcount[name] = 0
                continue
            _refcount[name] -= 1
            if _refcount[name] == 0:
                _unpatch_provider(name)
        if not any(_refcount.values()):
            _remove_hook()
            _sessions.clear()


def instrumented() -> dict[str, bool]:
    """Return whether each supported provider is currently auto-instrumented."""
    with _lock:
        return {name: _refcount[name] > 0 for name in PROVIDERS}


class InstrumentHandle:
    """Token returned by :func:`instrument`; closing it undoes that call."""

    def __init__(
        self,
        providers: tuple[str, ...],
        *,
        agent: str | None = None,
        store: Any | None = None,
    ) -> None:
        self.providers = providers
        self.agent = agent
        self.store = store
        self._closed = False

    def close(self) -> None:
        """Undo this ``instrument()`` call (idempotent)."""
        if self._closed:
            return
        self._closed = True
        uninstrument(self.providers)

    def __enter__(self) -> InstrumentHandle:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        state = "closed" if self._closed else "active"
        return f"<crashlab.instrument {', '.join(self.providers)} {state}>"


# --- provider name / session ------------------------------------------------


def _normalize_providers(providers: str | Iterable[str] | None) -> tuple[str, ...]:
    if providers is None:
        return PROVIDERS
    items = (providers,) if isinstance(providers, str) else tuple(providers)
    if not items:
        raise ValueError(
            "crashlab.instrument(): providers= is empty; pass one or more of "
            + ", ".join(PROVIDERS)
        )
    seen: list[str] = []
    unknown: list[str] = []
    for raw in items:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("crashlab.instrument(): provider names must be non-empty strings")
        name = raw.strip()
        if name not in _MODULE_NAME:
            unknown.append(name)
        elif name not in seen:
            seen.append(name)
    if unknown:
        raise ValueError(
            "crashlab.instrument(): unknown provider(s) "
            + ", ".join(repr(n) for n in unknown)
            + "; supported: "
            + ", ".join(PROVIDERS)
        )
    return tuple(seen)


def _pop_session(names: tuple[str, ...]) -> None:
    """Remove one matching session so the agent stack follows nesting."""
    for idx in range(len(_sessions) - 1, -1, -1):
        if _sessions[idx][0] == names:
            _sessions.pop(idx)
            return
    # uninstrument() called without a matching instrument() — just the refcount


def _agent_for(provider: str) -> str | None:
    for names, agent in reversed(_sessions):
        if provider in names:
            return agent
    return None


# --- in-place wrapping ------------------------------------------------------


def _try_setattr(obj: Any, name: str, value: Any) -> bool:
    try:
        object.__setattr__(obj, name, value)
        return True
    except (AttributeError, TypeError):
        try:
            setattr(obj, name, value)
            return True
        except (AttributeError, TypeError):
            return False


def _is_proxy(obj: Any) -> bool:
    from crashlab.capture._proxy import Proxy

    return isinstance(obj, Proxy)


def _has_recording_resources(client: Any, *paths: str) -> bool:
    """True when an in-place wrap (or ``crashlab.wrap``) already sits on a resource.

    A subclass ``__init__`` may call ``super().__init__`` (which wraps) and then
    replace ``chat`` / ``messages`` / ``models``; the marker flag would still be
    set, so we look at the *current* resources before skipping.
    """
    for path in paths:
        obj: Any = client
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        else:
            if _is_proxy(obj):
                return True
    return False


def _copy_overrides(proxy: Any, client: Any) -> None:
    """Copy recording resource proxies from a ``WrappedClient`` onto ``client``."""
    for name, value in vars(proxy).items():
        if name == "_inner":
            continue
        _try_setattr(client, name, value)
    _try_setattr(client, "__crashlab_wrapped__", True)
    _try_setattr(client, "_crashlab_instrumented", True)


def _wrap_openai_inplace(client: Any) -> None:
    if _has_recording_resources(client, "chat.completions", "responses", "embeddings"):
        return
    from crashlab.capture.openai import WrappedOpenAI, looks_like_openai

    if not looks_like_openai(client):
        return
    wrapped = WrappedOpenAI(client, provider="openai", agent=_agent_for("openai"))
    _copy_overrides(wrapped, client)


def _wrap_anthropic_inplace(client: Any) -> None:
    if _has_recording_resources(client, "messages", "beta.messages"):
        return
    from crashlab.capture.anthropic import WrappedAnthropic, looks_like_anthropic

    if not looks_like_anthropic(client):
        return
    wrapped = WrappedAnthropic(client, agent=_agent_for("anthropic"))
    _copy_overrides(wrapped, client)


def _wrap_genai_inplace(client: Any) -> None:
    if _has_recording_resources(client, "models", "chats", "aio"):
        return
    from crashlab.capture.genai import WrappedGenAI, looks_like_genai

    if not looks_like_genai(client):
        return
    wrapped = WrappedGenAI(client, agent=_agent_for("google_genai"))
    _copy_overrides(wrapped, client)


_INPLACE = {
    "openai": _wrap_openai_inplace,
    "anthropic": _wrap_anthropic_inplace,
    "google_genai": _wrap_genai_inplace,
}


def _patch_init(cls: type, provider: str) -> Any:
    """Replace ``cls.__init__`` so the constructed instance is wrapped in-place."""
    orig = cls.__init__
    if getattr(orig, "__crashlab_instrumented_init__", False):
        return orig  # already our wrapper (or a subclass sharing it)

    wrap_inplace = _INPLACE[provider]

    @functools.wraps(orig)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        orig(self, *args, **kwargs)
        wrap_inplace(self)

    __init__.__crashlab_instrumented_init__ = True  # type: ignore[attr-defined]
    __init__.__crashlab_orig_init__ = orig  # type: ignore[attr-defined]
    cls.__init__ = __init__  # type: ignore[method-assign]
    return orig


def _restore_init(cls: type, orig: Any) -> None:
    current = getattr(cls, "__init__", None)
    # only restore if our wrapper is still in place
    if (
        getattr(current, "__crashlab_instrumented_init__", False)
        and getattr(current, "__crashlab_orig_init__", None) is orig
    ):
        cls.__init__ = orig  # type: ignore[method-assign]


# --- patch / unpatch a provider --------------------------------------------


def _module_for(provider: str, *, try_import: bool) -> ModuleType | None:
    name = _MODULE_NAME[provider]
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    if not try_import:
        return None
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _patch_classes(provider: str, mod: ModuleType) -> None:
    saved = _original_inits.setdefault(provider, {})
    for cls_name in _INIT_CLASSES.get(provider, ()):
        cls = getattr(mod, cls_name, None)
        if cls is None or not isinstance(cls, type):
            continue
        if cls_name in saved:
            continue
        orig = _patch_init(cls, provider)
        saved[cls_name] = orig


def _unpatch_classes(provider: str, mod: ModuleType | None) -> None:
    saved = _original_inits.pop(provider, {})
    if mod is None:
        return
    for cls_name, orig in saved.items():
        cls = getattr(mod, cls_name, None)
        if cls is None or not isinstance(cls, type):
            continue
        _restore_init(cls, orig)


def _patch_litellm(mod: ModuleType | None) -> None:
    global _litellm_owned
    from crashlab.capture.litellm import is_patched, patch

    if is_patched():
        return
    if mod is not None:
        patch(mod)
        _litellm_owned = True
        return
    try:
        patch()
        _litellm_owned = True
    except ImportError:
        _litellm_owned = False


def _unpatch_litellm(mod: ModuleType | None) -> None:
    global _litellm_owned
    from crashlab.capture.litellm import unpatch

    if _litellm_owned:
        unpatch()
    _litellm_owned = False


def _patch_provider(name: str) -> None:
    if name == "litellm":
        _patch_litellm(_module_for(name, try_import=True))
        return
    mod = _module_for(name, try_import=True)
    if mod is not None:
        _patch_classes(name, mod)


def _unpatch_provider(name: str) -> None:
    if name == "litellm":
        _unpatch_litellm(sys.modules.get(_MODULE_NAME[name]))
        return
    _unpatch_classes(name, sys.modules.get(_MODULE_NAME[name]))


def _patch_loaded_module(fullname: str, module: ModuleType) -> None:
    """Called after a watched module is imported (and for modules already loaded)."""
    provider = next((p for p, n in _MODULE_NAME.items() if n == fullname), None)
    if provider is None or _refcount.get(provider, 0) <= 0:
        return
    with _lock:
        if _refcount.get(provider, 0) <= 0:
            return
        if provider == "litellm":
            _patch_litellm(module)
        else:
            _patch_classes(provider, module)


# --- post-import hook -------------------------------------------------------


class _LoaderWrapper:
    def __init__(self, loader: Any, fullname: str) -> None:
        object.__setattr__(self, "_loader", loader)
        object.__setattr__(self, "_fullname", fullname)

    def create_module(self, spec: Any) -> ModuleType | None:
        loader = self._loader
        if hasattr(loader, "create_module"):
            return loader.create_module(spec)
        return None

    def exec_module(self, module: ModuleType) -> None:
        self._loader.exec_module(module)
        _patch_loaded_module(self._fullname, module)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """Wrap the real loader of a watched module so we patch it after exec."""

    def __init__(self) -> None:
        self._busy: set[str] = set()

    def find_spec(
        self,
        fullname: str,
        path: Any,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname not in _MODULE_NAME.values() or fullname in self._busy:
            return None
        self._busy.add(fullname)
        try:
            for finder in sys.meta_path:
                if finder is self:
                    continue
                find_spec = getattr(finder, "find_spec", None)
                if find_spec is None:
                    continue
                spec = find_spec(fullname, path, target)
                if spec is None or spec.loader is None:
                    continue
                spec.loader = _LoaderWrapper(spec.loader, fullname)  # type: ignore[assignment]
                return spec
            return None
        finally:
            self._busy.discard(fullname)


def _install_hook() -> None:
    global _hook
    if _hook is not None and _hook in sys.meta_path:
        return
    _hook = _PostImportFinder()
    # in front of the path finders so we see every import of a watched module
    sys.meta_path.insert(0, _hook)


def _remove_hook() -> None:
    global _hook
    hook, _hook = _hook, None
    if hook is None:
        return
    with contextlib.suppress(ValueError):
        sys.meta_path.remove(hook)


def _reset() -> None:
    """Unpatch everything and drop the import hook. Tests only."""
    with _lock:
        for name in list(PROVIDERS):
            if _refcount[name] > 0:
                _unpatch_provider(name)
            _refcount[name] = 0
        _sessions.clear()
        _remove_hook()


__all__ = [
    "InstrumentHandle",
    "PROVIDERS",
    "instrument",
    "instrumented",
    "uninstrument",
]
