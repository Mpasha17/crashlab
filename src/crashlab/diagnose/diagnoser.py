"""``crashlab.diagnose(run, model=...)``: one LLM call over the compacted trace, validated.

The diagnoser model is any client crashlab can wrap: an OpenAI-compatible endpoint
(``base_url=``: Groq, Gemini, Ollama, vLLM, OpenRouter ...), a LiteLLM model string
(``groq/...``, ``gemini/...``, ``ollama/...``), Anthropic (``claude-*``) or OpenAI. Its own
calls are recorded as ``crashlab-diagnoser`` runs in ``<store>/diagnoser/``, a separate
store, so they never appear among the user's runs. Results are cached by a hash of the
prompt version, the model and the exact prompt, so a changed trace, check or finding
invalidates the cache."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Iterable
from typing import Any

from ..context import current
from ..models import Diagnosis, Run
from .compact import compact
from .evidence import validate
from .heuristics import candidates, heuristic_raw
from .prompt import PROMPT_VERSION, messages

ENV_MODEL = "CRASHLAB_DIAGNOSER_MODEL"
ENV_BASE_URL = "CRASHLAB_DIAGNOSER_BASE_URL"
ENV_API_KEY = "CRASHLAB_DIAGNOSER_API_KEY"
DIAGNOSER_RUN_NAME = "crashlab-diagnoser"
# model prefixes routed through LiteLLM when no base_url is given ("openai/" is not one:
# Groq's "openai/gpt-oss-120b" is a model name)
LITELLM_PREFIXES = (
    "groq/",
    "gemini/",
    "vertex_ai/",
    "ollama/",
    "ollama_chat/",
    "anthropic/",
    "openrouter/",
    "together_ai/",
    "mistral/",
    "deepseek/",
    "xai/",
    "bedrock/",
    "azure/",
    "fireworks_ai/",
    "cohere/",
    "huggingface/",
)

Messages = list[dict[str, str]]


class DiagnoseError(RuntimeError):
    """The diagnosis could not be produced (no model configured, the call failed ...)."""


def cache_key(model: str, msgs: Messages) -> str:
    blob = json.dumps({"v": PROMPT_VERSION, "model": model, "messages": msgs}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8", "surrogatepass")).hexdigest()


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_json(text: Any) -> dict[str, Any] | None:
    """The JSON object in a model reply: bare, in a ```json fence, or embedded in prose."""
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    text = _THINK.sub("", text).strip()
    tries = [text, *_FENCE.findall(text)]
    if "{" in text and "}" in text:
        tries.append(text[text.index("{") : text.rindex("}") + 1])
    for t in [*tries, *(_requote(t) for t in tries if "'" in t)]:
        try:
            value = json.loads(t.strip())
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


_CLOSE_SINGLE = re.compile(r"'\s*[,}\]:]")


def _requote(text: str) -> str:
    """Turn Python-style ``'single-quoted'`` string values into JSON strings.

    Gemini flash-lite sometimes quotes a value that itself holds JSON with single quotes
    (``"result": '{"order_id": ...}'``), which ``json.loads`` rejects. A single-quoted
    string ends at a ``'`` followed by ``, } ] :`` so apostrophes inside it survive."""
    out: list[str] = []
    i, n, in_double = 0, len(text), False
    while i < n:
        c = text[i]
        if in_double:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 1
            elif c == '"':
                in_double = False
        elif c == '"':
            in_double = True
            out.append(c)
        elif c == "'":
            m = _CLOSE_SINGLE.search(text, i + 1)
            if m is None:
                return text
            out.append(json.dumps(text[i + 1 : m.start()].replace("\\'", "'")))
            i = m.start()
        else:
            out.append(c)
        i += 1
    return "".join(out)


def _api_of(model: str, base_url: str | None) -> str:
    if base_url:
        return "openai"
    if model.startswith(LITELLM_PREFIXES):
        return "litellm"
    if model.startswith("claude"):
        return "anthropic"
    return "openai"


def _make_client(model: str, base_url: str | None, api_key: str | None) -> tuple[Any, str]:
    from ..capture import wrap

    api = _api_of(model, base_url)
    try:
        if api == "litellm":
            import litellm

            return wrap(litellm), api
        if api == "anthropic":
            import anthropic

            return wrap(anthropic.Anthropic(**({"api_key": api_key} if api_key else {}))), api
        import openai
    except ImportError as e:
        raise DiagnoseError(
            f"the diagnoser model {model!r} needs the {e.name!r} package "
            f"(pip install 'crashlab[{api}]')"
        ) from e
    kwargs: dict[str, Any] = {}
    if base_url:
        kwargs["base_url"] = base_url
        kwargs["api_key"] = api_key or "not-needed"  # local servers ignore it
    elif api_key:
        kwargs["api_key"] = api_key
    return wrap(openai.OpenAI(**kwargs)), api


def _client_api(client: Any) -> str:
    if getattr(client, "__name__", None) == "litellm":
        return "litellm"
    if hasattr(client, "messages") and not hasattr(client, "chat"):
        return "anthropic"
    return "openai"


def _complete(client: Any, api: str, model: str, msgs: Messages, json_mode: bool) -> str:
    if api == "anthropic":
        resp = client.messages.create(
            model=model,
            system=msgs[0]["content"],
            messages=msgs[1:],
            max_tokens=4000,
            temperature=0,
        )
        return "".join(getattr(b, "text", "") or "" for b in resp.content)
    create = client.completion if api == "litellm" else client.chat.completions.create
    kwargs: dict[str, Any] = {"model": model, "messages": msgs, "temperature": 0}
    if json_mode:
        try:
            resp = create(**kwargs, response_format={"type": "json_object"})
        except Exception as e:  # some servers reject response_format: retry without it
            if getattr(e, "status_code", None) not in (400, 422) and "response_format" not in str(
                e
            ):
                raise
            resp = create(**kwargs)
    else:
        resp = create(**kwargs)
    msg = resp.choices[0].message
    return msg.content or ""


def _ask(
    run: Run,
    msgs: Messages,
    model: str,
    *,
    client: Any,
    fn: Callable[[Messages], Any] | None,
    base_url: str | None,
    api_key: str | None,
    json_mode: bool,
    store: Any,
    prompt_version: str = PROMPT_VERSION,
) -> tuple[Any, dict[str, Any]]:
    if current() is not None:
        raise DiagnoseError(
            "crashlab.diagnose() cannot run inside an active crashlab run: its LLM call "
            "would be recorded into that run. Call it after the run has finished."
        )
    t0 = time.perf_counter()
    if fn is not None:
        reply = fn(msgs)
        return reply, {"backend": "fn", "seconds": round(time.perf_counter() - t0, 2)}

    from ..capture import record, wrap
    from ..store import crashlab_dir

    if client is None:
        client, api = _make_client(model, base_url, api_key)
    else:
        api = _client_api(client)
        client = wrap(client)
    root = store.root if store is not None else crashlab_dir()
    try:
        with record(
            DIAGNOSER_RUN_NAME,
            input={"run_id": run.run_id, "model": model, "prompt_version": prompt_version},
            store_dir=root / "diagnoser",
            checks=False,
            tags=[DIAGNOSER_RUN_NAME],
        ) as drun:
            reply = _complete(client, api, model, msgs, json_mode)
            drun.output = reply
    except DiagnoseError:
        raise
    except Exception as e:
        text = str(e)
        hint = " (daily quota exhausted)" if "PerDay" in text or "per day" in text.lower() else ""
        raise DiagnoseError(
            f"the diagnoser call failed{hint}: {type(e).__name__}: {text[:300]}"
        ) from e
    u = drun.usage
    return reply, {
        "backend": api,
        "diagnoser_run_id": drun.run_id,
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cost_usd": u.cost_usd,
        "seconds": round(time.perf_counter() - t0, 2),
    }


def _in_store(store: Any, run: Run) -> bool:
    return store is not None and store.index.get_run_json(run.run_id) is not None


def diagnose(
    run: Run | str,
    model: str | None = None,
    *,
    client: Any = None,
    fn: Callable[[Messages], Any] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    heuristic: bool = False,
    cache: bool = True,
    save: bool = True,
    store: Any = None,
    hide_input: Iterable[str] = (),
    max_field: int = 600,
    json_mode: bool = True,
) -> Diagnosis:
    """Diagnose a recorded run: label, culprit step, root cause and verbatim evidence.

    ``model`` (or ``$CRASHLAB_DIAGNOSER_MODEL``) picks the diagnoser; ``base_url`` (or
    ``$CRASHLAB_DIAGNOSER_BASE_URL``) points it at any OpenAI-compatible endpoint, with
    ``api_key`` or ``$CRASHLAB_DIAGNOSER_API_KEY``. ``client`` passes your own client and
    ``fn(messages) -> reply`` any callable (not recorded). ``heuristic=True`` makes no LLM
    call. The answer is validated: a label outside the taxonomy, a step that does not
    exist or a failure label without a verbatim evidence quote becomes ``UNKNOWN``.
    The diagnosis is stored with the run (``save``) and cached by prompt hash (``cache``).
    """
    from ..store import Store

    if isinstance(run, str):
        store = store or Store()
        run = store.load_run(run)
    elif store is None and (save or cache):
        store = Store()
    keep = save and _in_store(store, run)
    cands = candidates(run)
    hidden = tuple(hide_input)

    if heuristic:
        d = validate(
            heuristic_raw(run, cands),
            run,
            model="heuristic",
            method={"backend": "heuristic", "candidates": [c.as_dict() for c in cands]},
        )
        if keep:
            store.save_diagnosis(run, d)
        else:
            run.diagnosis = d
        return d

    model = model or os.environ.get(ENV_MODEL) or ("custom" if fn is not None else None)
    if not model:
        raise DiagnoseError(
            f"no diagnoser model: pass model=..., set ${ENV_MODEL}, or use heuristic=True"
        )
    base_url = base_url or os.environ.get(ENV_BASE_URL) or None
    api_key = api_key or os.environ.get(ENV_API_KEY) or None
    trace = compact(run, candidates=cands, max_field=max_field, hide_input=hidden)
    msgs = messages(trace)
    key = cache_key(model, msgs)
    if cache and _in_store(store, run):
        hit = store.cached_diagnosis(run.run_id, key)
        if hit is not None:
            hit.method = {**hit.method, "cached": True}
            if keep and (run.diagnosis is None or run.diagnosis.created_at != hit.created_at):
                run.diagnosis = hit
                store.update_run(run)
            run.diagnosis = hit
            return hit

    reply, info = _ask(
        run,
        msgs,
        model,
        client=client,
        fn=fn,
        base_url=base_url,
        api_key=api_key,
        json_mode=json_mode,
        store=store,
    )
    raw = parse_json(reply)
    method: dict[str, Any] = {
        **info,
        "prompt_version": PROMPT_VERSION,
        "prompt_chars": sum(len(m["content"]) for m in msgs),
        "candidates": [c.as_dict() for c in cands],
        "cached": False,
        "cache_key": key,
    }
    if raw is None:
        method["raw_reply"] = str(reply)[:500]
    d = validate(raw, run, model=model, method=method)
    if keep:
        store.save_diagnosis(run, d, key)
    else:
        run.diagnosis = d
    return d
