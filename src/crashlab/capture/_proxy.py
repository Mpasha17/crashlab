"""Generic attribute-delegating proxy used by the client wrappers."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse


class Proxy:
    """Delegates every attribute to ``inner`` except the ones given as overrides."""

    def __init__(self, inner: Any, **overrides: Any) -> None:
        object.__setattr__(self, "_inner", inner)
        for name, value in overrides.items():
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._inner, name, value)

    def __repr__(self) -> str:
        return f"<crashlab proxy for {self._inner!r}>"


class WrappedClient(Proxy):
    """Base for wrapped LLM clients (``crashlab.wrap(client)``)."""

    __crashlab_wrapped__ = True

    def __init__(self, client: Any, **overrides: Any) -> None:
        super().__init__(client, **overrides)

    @property
    def unwrapped(self) -> Any:
        return self._inner

    def _rewrap(self, client: Any) -> WrappedClient:  # pragma: no cover - overridden
        raise NotImplementedError

    def with_options(self, *args: Any, **kwargs: Any) -> WrappedClient:
        return self._rewrap(self._inner.with_options(*args, **kwargs))

    def copy(self, *args: Any, **kwargs: Any) -> WrappedClient:
        return self._rewrap(self._inner.copy(*args, **kwargs))

    def __repr__(self) -> str:
        return f"crashlab.wrap({self._inner!r})"


def server_address(client: Any) -> str | None:
    base = getattr(client, "base_url", None)
    if base is None:
        return None
    raw = str(base).strip()
    if not raw:
        return None
    if "://" not in raw:
        raw = "https://" + raw  # OpenAI-compatible clients sometimes omit the scheme
    try:
        return urlparse(raw).hostname
    except ValueError:
        return None
