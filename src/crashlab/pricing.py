"""USD cost estimates for LLM calls.

``step_cost(step)`` prices one recorded chat step from its token counts, using (in order)
prices you set (``crashlab.pricing.set_price(...)`` or ``[tool.crashlab.prices]`` in
``pyproject.toml``), a small built-in table of list prices, and LiteLLM's cost map when
LiteLLM is installed. Calls to a local server (Ollama, vLLM, LM Studio on localhost) cost
0. Anything else returns ``None``: an unknown model is never guessed.

These are list-price *estimates*: free tiers, batch discounts, regional surcharges and
negotiated rates are not modelled.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

PRICES_AS_OF = "2026-10-18"


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens. ``cached_input`` applies to cache-read input tokens and
    ``cache_write`` to cache-creation tokens (Anthropic); both default to ``input``."""

    input: float
    output: float
    cached_input: float | None = None
    cache_write: float | None = None
    until: date | None = None  # last day this price applies (inclusive)
    source: str = "builtin"


def _p(
    i: float, o: float, c: float | None = None, w: float | None = None, until: str | None = None
) -> Price:  # noqa: E501
    return Price(i, o, c, w, date.fromisoformat(until) if until else None)


# provider -> model -> [prices, oldest first]; checked against the providers' pricing pages
# on PRICES_AS_OF (OpenAI, Anthropic, Groq, Google AI for Developers).
_GEMINI_FLASH = [_p(0.75, 3.75, 0.075, until="2026-12-31"), _p(1.50, 7.50, 0.15)]
BUILTIN: dict[str, dict[str, list[Price]]] = {
    "openai": {
        "gpt-4o-mini": [_p(0.15, 0.60, 0.075)],
        "gpt-4o": [_p(2.50, 10.00, 1.25)],
        "gpt-4.1": [_p(2.00, 8.00, 0.50)],
        "gpt-4.1-mini": [_p(0.40, 1.60, 0.10)],
        "gpt-4.1-nano": [_p(0.10, 0.40, 0.025)],
        "gpt-5": [_p(1.25, 10.00, 0.125)],
        "gpt-5-mini": [_p(0.25, 2.00, 0.025)],
        "gpt-5-nano": [_p(0.05, 0.40, 0.005)],
        "o3": [_p(2.00, 8.00, 0.50)],
        "o4-mini": [_p(1.10, 4.40, 0.275)],
    },
    "anthropic": {
        "claude-haiku-4-5": [_p(1.00, 5.00, 0.10, 1.25)],
        "claude-sonnet-4-5": [_p(3.00, 15.00, 0.30, 3.75)],
        "claude-sonnet-4-6": [_p(3.00, 15.00, 0.30, 3.75)],
        "claude-opus-4-5": [_p(5.00, 25.00, 0.50, 6.25)],
    },
    "groq": {
        "openai/gpt-oss-120b": [_p(0.15, 0.60, 0.075)],
        "openai/gpt-oss-20b": [_p(0.075, 0.30, 0.0375)],
    },
    "gemini": {
        "gemini-3.8-flash": _GEMINI_FLASH,
        "gemini-3.7-flash": _GEMINI_FLASH,  # deprecated, served by 3.8 Flash
        "gemini-3.6-flash": _GEMINI_FLASH,
    },
}

# OpenAI-compatible hosts -> provider key
_HOSTS = {
    "api.openai.com": "openai",
    "api.anthropic.com": "anthropic",
    "api.groq.com": "groq",
    "generativelanguage.googleapis.com": "gemini",
}
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal")
_LOCAL_PREFIXES = ("ollama/", "ollama_chat/", "lm_studio/", "hosted_vllm/", "llamafile/")
# LiteLLM model prefixes -> provider key
_LITELLM_PREFIX = {"groq": "groq", "gemini": "gemini", "anthropic": "anthropic", "openai": "openai"}

_custom: dict[tuple[str | None, str], Price] = {}
_lock = threading.Lock()


def set_price(
    model: str,
    *,
    input: float,
    output: float,
    cached_input: float | None = None,
    cache_write: float | None = None,
    provider: str | None = None,
) -> None:
    """Set (or override) the USD-per-1M-token price of ``model`` (optionally only when
    served by ``provider``: openai, anthropic, groq, gemini, or a host name)."""
    with _lock:
        _custom[(provider, model)] = Price(input, output, cached_input, cache_write,
                                           source="custom")  # fmt: skip


def clear_prices() -> None:
    with _lock:
        _custom.clear()


def provider_of(attributes: dict[str, Any], model: str) -> str | None:
    """Best guess of who served (and bills) the call: a provider key, "local", or the
    server host when it is not one we know."""
    host = str(attributes.get("server.address") or "").lower()
    if host:
        if host in _HOSTS:
            return _HOSTS[host]
        if host.split(":")[0] in _LOCAL_HOSTS or host.endswith(".local"):
            return "local"
        return host
    if model.startswith(_LOCAL_PREFIXES):
        return "local"
    name = str(attributes.get("gen_ai.provider.name") or "")
    if name == "litellm":
        head = model.split("/", 1)[0]
        return _LITELLM_PREFIX.get(head, head if "/" in model else "openai")
    if name in ("openai", "anthropic"):
        return name  # SDK default endpoint
    return name or None


def _strip_prefix(model: str, provider: str | None) -> str:
    if provider and model.startswith(provider + "/"):
        return model[len(provider) + 1 :]
    return model


def _pick(prices: list[Price], when: date | None) -> Price:
    for p in prices:
        if p.until is None or when is None or when <= p.until:
            return p
    return prices[-1]


def _builtin(provider: str | None, model: str, when: date | None) -> Price | None:
    table = BUILTIN.get(provider or "", {})
    names = [model, _strip_prefix(model, provider)]
    for name in names:
        if name in table:
            return _pick(table[name], when)
        # dated snapshots: gpt-4o-mini-2024-07-18, claude-haiku-4-5-20251001
        for base in sorted(table, key=len, reverse=True):
            rest = name[len(base) :]
            if name.startswith(base + "-") and rest[1:].replace("-", "").isdigit():
                return _pick(table[base], when)
    return None


_litellm_map: dict[str, Any] | None = None


def _litellm_cost_map() -> dict[str, Any]:
    """LiteLLM's bundled cost map, if LiteLLM is installed (never fetched from the
    network by crashlab)."""
    global _litellm_map
    if _litellm_map is not None:
        return _litellm_map
    cost_map: dict[str, Any] = {}
    mod = sys.modules.get("litellm")
    if mod is None and importlib.util.find_spec("litellm") is not None:
        old = os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP")
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"  # use the bundled copy
        try:
            import litellm as mod  # type: ignore[no-redef]
        except Exception:  # pragma: no cover - broken install
            mod = None
        finally:
            if old is None:
                os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
            else:
                os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = old
    if mod is not None:
        cost_map = dict(getattr(mod, "model_cost", None) or {})
    _litellm_map = cost_map
    return cost_map


def _from_litellm(provider: str | None, model: str) -> Price | None:
    cmap = _litellm_cost_map()
    if not cmap:
        return None
    bare = _strip_prefix(model, provider)
    keys = [model] + ([f"{provider}/{bare}", bare] if provider else [])
    for key in keys:
        info = cmap.get(key)
        if not isinstance(info, dict):
            continue
        pin, pout = info.get("input_cost_per_token"), info.get("output_cost_per_token")
        if pin is None or pout is None:
            continue
        lp = info.get("litellm_provider")
        if provider and lp and key == bare and _LITELLM_PREFIX.get(lp, lp) != provider:
            continue  # the same model name from a different provider: not our price
        cache = info.get("cache_read_input_token_cost")
        write = info.get("cache_creation_input_token_cost")
        return Price(
            pin * 1e6,
            pout * 1e6,
            cache * 1e6 if cache is not None else None,
            write * 1e6 if write is not None else None,
            source="litellm",
        )
    return None


def price_for(
    model: str, provider: str | None = None, when: date | datetime | None = None
) -> Price | None:
    """The price that applies to ``model`` served by ``provider`` on ``when``; None if
    unknown. Local servers are free."""
    if isinstance(when, datetime):
        when = when.date()
    if provider == "local":
        return Price(0.0, 0.0, source="local")
    with _lock:
        custom = _custom.get((provider, model)) or _custom.get((None, model))
    if custom is None:
        custom = _pyproject_prices().get((provider, model)) or _pyproject_prices().get(
            (None, model)
        )
    if custom is not None:
        return custom
    return _builtin(provider, model, when) or _from_litellm(provider, model)


def _pyproject_prices() -> dict[tuple[str | None, str], Price]:
    from .checks.config import pyproject_section

    out: dict[tuple[str | None, str], Price] = {}
    for model, spec in (pyproject_section("prices") or {}).items():
        if not isinstance(spec, dict) or "input" not in spec or "output" not in spec:
            continue
        out[(spec.get("provider"), model)] = Price(
            float(spec["input"]),
            float(spec["output"]),
            spec.get("cached_input"),
            spec.get("cache_write"),
            source="pyproject",
        )
    return out


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def step_cost(step: Any) -> float | None:
    """Estimated USD cost of one chat step; None when the price (or usage) is unknown,
    0.0 for calls that reported no tokens (e.g. a failed request)."""
    if getattr(step, "operation", None) != "chat":
        return None
    attrs = step.attributes or {}
    inp = _int(attrs.get("gen_ai.usage.input_tokens"))
    out = _int(attrs.get("gen_ai.usage.output_tokens"))
    total = _int(attrs.get("crashlab.usage.total_tokens"))
    if not (inp or out or total):
        return 0.0 if step.error is not None else None
    model = str(attrs.get("gen_ai.response.model") or attrs.get("gen_ai.request.model")
                or step.name or "")  # fmt: skip
    provider = provider_of(attrs, model)
    price = price_for(model, provider, step.started_at)
    if price is None and attrs.get("gen_ai.request.model") not in (None, model):
        req_model = str(attrs["gen_ai.request.model"])
        price = price_for(req_model, provider_of(attrs, req_model), step.started_at)
    if price is None:
        return None
    cached = _int(attrs.get("gen_ai.usage.cache_read.input_tokens"))
    written = _int(attrs.get("gen_ai.usage.cache_creation.input_tokens"))
    if provider == "anthropic" or str(attrs.get("crashlab.api", "")).startswith("anthropic"):
        fresh = inp  # Anthropic reports cache reads / writes separately from input_tokens
    else:
        fresh = max(inp - cached, 0)  # OpenAI-style input_tokens include cached tokens
        written = 0
    # thinking tokens some providers leave out of output_tokens (Gemini) but bill
    out = max(out, total - inp - cached - written if provider == "anthropic" else total - inp)
    usd = (
        fresh * price.input
        + cached * (price.cached_input if price.cached_input is not None else price.input)
        + written * (price.cache_write if price.cache_write is not None else price.input)
        + out * price.output
    ) / 1e6
    return round(usd, 8)


__all__ = [
    "BUILTIN",
    "PRICES_AS_OF",
    "Price",
    "clear_prices",
    "price_for",
    "run_cost",
    "provider_of",
    "set_price",
    "step_cost",
]


def run_cost(run: Any) -> float | None:
    """A run's estimated cost: the stored total, or (older runs) the sum of its chat
    steps' costs; None if any priced call is unknown."""
    if run.usage.cost_usd is not None:
        return run.usage.cost_usd
    total, any_chat = 0.0, False
    for step in run.steps:
        if step.operation != "chat":
            continue
        any_chat = True
        cost = step.attributes.get("crashlab.usage.cost_usd")
        cost = step_cost(step) if cost is None else cost
        if cost is None:
            return None
        total += cost
    return round(total, 8) if any_chat else None
