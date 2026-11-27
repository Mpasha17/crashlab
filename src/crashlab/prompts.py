"""Prompt version tracking and A/B comparison: which prompt change broke or fixed the agent?

Every recorded chat step is stamped with a **prompt id** (``p-3f9a1c``): a short hash of
the normalized system prompt. Normalization makes the id insensitive to whitespace and to
template placeholders (``{name}``, ``{{ name }}``, ``${name}``, ``%(name)s``) and to
volatile values such as ISO dates and UUIDs, so only a real wording change makes a new
version. Old runs recorded before this module existed are fingerprinted lazily from the
stored request, so nothing has to be re-recorded.

Name a version yourself with :func:`prompt_version` (``with crashlab.prompt_version
("support-v3"):`` or ``record(..., prompt_version="support-v3")``). The label is stored next
to the id; both select a version in ``crashlab prompts``.

:func:`list_prompts` summarises each version (runs, pass rate, error and finding rates,
cost, latency, tokens); :func:`compare_prompts` compares two versions on one outcome metric
with a significance test written in pure Python (Fisher's exact test or a two-proportion z
test for pass rate, Mann-Whitney U for cost, latency and judge scores).
"""

from __future__ import annotations

import contextvars
import difflib
import functools
import hashlib
import math
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import Run, Step
    from .store import Store

__all__ = [
    "ABResult",
    "PromptError",
    "PromptVersion",
    "compare_prompts",
    "diff_prompts",
    "fisher_exact",
    "list_prompts",
    "mann_whitney",
    "normalize_prompt",
    "prompt_id",
    "prompt_version",
    "step_prompt",
    "two_proportion_z",
]

ID_ATTR = "crashlab.prompt.id"
LABEL_ATTR = "crashlab.prompt.label"
TEMPLATES_ATTR = "crashlab.prompt.templates"
METRICS = ("pass_rate", "cost", "latency")


class PromptError(ValueError):
    """An unknown prompt version, an unusable metric, or nothing to compare."""


# --- fingerprint -------------------------------------------------------------------------

_PLACEHOLDERS = (
    re.compile(r"\{\{[^{}\n]{0,80}\}\}"),  # {{ name }} jinja / mustache
    re.compile(r"\$\{[^{}\n]{0,80}\}"),  # ${name}
    re.compile(r"\{%[^{}\n]{0,80}%\}"),  # {% if x %}
    re.compile(r"%\([A-Za-z_][\w.]{0,40}\)[sdif]"),  # %(name)s
    re.compile(r"\{[A-Za-z_][\w.\[\]'\"]{0,40}(?:![rsa])?(?::[^{}\n]{0,20})?\}"),  # {name} {0}
    re.compile(r"\{\s*\d{0,3}\s*\}"),  # {} {0}
)
_UUID = re.compile(r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b")
_DATE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"
)
_VOLATILE = ((_UUID, "{uuid}"), (_DATE, "{date}"))
_SPACE = re.compile(r"\s+")


def normalize_prompt(text: str) -> str:
    """The canonical form that is hashed: placeholders and volatile values (ISO dates,
    UUIDs) collapse to a marker and all whitespace runs to one space."""
    out = str(text)
    for pat, repl in _VOLATILE:
        out = pat.sub(repl, out)
    for pat in _PLACEHOLDERS:
        out = pat.sub("{}", out)
    return _SPACE.sub(" ", out).strip()


def prompt_id(text: str | None) -> str | None:
    """``"p-3f9a1c"``: the short id of a prompt text (None for empty / missing text)."""
    if text is None or not str(text).strip():
        return None
    digest = hashlib.sha256(normalize_prompt(str(text)).encode("utf-8")).hexdigest()
    return f"p-{digest[:6]}"


# --- explicit labels ---------------------------------------------------------------------

_scope: contextvars.ContextVar[tuple[str | None, dict[str, str]] | None] = contextvars.ContextVar(
    "crashlab_prompt_version", default=None
)


@contextmanager
def prompt_version(
    label: str | None = None, *, templates: Mapping[str, str] | None = None
) -> Iterator[None]:
    """Label every chat step recorded inside the block with a prompt version name::

        with crashlab.prompt_version("support-v3"):
            answer = run_agent(question)

    ``templates`` maps a name to a prompt template text (a retrieval prompt, a planner
    prompt ...); each is fingerprinted and stored as ``crashlab.prompt.templates``. Works
    in sync and async code and nests (the innermost label wins). Labels never change
    recording, replay or request hashes."""
    if label is not None and (not isinstance(label, str) or not label.strip()):
        raise ValueError("crashlab.prompt_version(): label must be a non-empty string")
    if label is None and not templates:
        raise ValueError("crashlab.prompt_version(): pass a label and / or templates=")
    tpl = {str(k): prompt_id(v) or "" for k, v in (templates or {}).items()}
    outer = _scope.get()
    if outer is not None:
        label = label if label is not None else outer[0]
        tpl = {**outer[1], **tpl}
    token = _scope.set((label.strip() if label else None, tpl))
    try:
        yield
    finally:
        _scope.reset(token)


def stamp_step(ctx: Any, attrs: dict[str, Any], request: Mapping[str, Any]) -> None:
    """Record time: add the prompt id / label to a chat step's attributes. Cheap (one hash
    of the system prompt) and never raises."""
    try:
        pid = prompt_id(request.get("system") if isinstance(request, Mapping) else None)
        if pid:
            attrs.setdefault(ID_ATTR, pid)
        scoped = _scope.get()
        label = scoped[0] if scoped else None
        if label is None:
            env = getattr(getattr(ctx, "run", None), "env", None) or {}
            raw = env.get("prompt_version")
            label = raw if isinstance(raw, str) and raw.strip() else None
        if label:
            attrs.setdefault(LABEL_ATTR, label)
        if scoped and scoped[1]:
            attrs.setdefault(TEMPLATES_ATTR, dict(scoped[1]))
    except Exception:  # fingerprinting must never break the agent
        return


# --- reading prompts back ----------------------------------------------------------------


def _system_text(step: Step) -> str:
    req = step.request or {}
    system = req.get("system")
    if isinstance(system, str):
        return system
    out: list[str] = []  # older / imported shapes: a leading system message
    for msg in req.get("messages") or []:
        if not isinstance(msg, dict) or msg.get("role") not in ("system", "developer"):
            break
        content = msg.get("content")
        if content is None:
            content = "".join(
                p.get("content", "") for p in msg.get("parts") or [] if isinstance(p, dict)
            )
        out.append(content if isinstance(content, str) else str(content))
    return "\n\n".join(out)


def step_prompt(step: Step) -> dict[str, Any] | None:
    """``{"id", "label", "text"}`` for a chat step, from its stamp or computed lazily from
    the stored request; None for a step without a system prompt."""
    if step.operation != "chat":
        return None
    text = _system_text(step)
    attrs = step.attributes or {}
    pid = attrs.get(ID_ATTR) if isinstance(attrs.get(ID_ATTR), str) else None
    pid = pid or prompt_id(text)
    if not pid:
        return None
    label = attrs.get(LABEL_ATTR)
    return {"id": pid, "label": label if isinstance(label, str) and label else None, "text": text}


def _agent_of(step: Step) -> str | None:
    value = (step.attributes or {}).get("gen_ai.agent.name")
    return value if isinstance(value, str) else None


def run_prompts(run: Run, *, agent: str | None = None) -> dict[str, dict[str, Any]]:
    """The prompt versions one run used: ``{id: {"id", "label", "labels", "agents", "text"}}``.
    ``agent`` keeps the run's chat steps of that agent (all of them when the run itself is
    named ``agent``)."""
    out: dict[str, dict[str, Any]] = {}
    for step in run.steps:
        if agent and run.name != agent and _agent_of(step) != agent:
            continue
        info = step_prompt(step)
        if info is None:
            continue
        cur = out.setdefault(info["id"], {**info, "labels": [], "agents": set()})
        cur["agents"].add(_agent_of(step) or run.name)
        if info["label"] and info["label"] not in cur["labels"]:
            cur["labels"].append(info["label"])
    return out


# --- metrics -----------------------------------------------------------------------------


def _outcome_bucket(run: Run) -> str:
    from .stats import _bucket

    return _bucket(run)


def run_metrics(run: Run) -> dict[str, Any]:
    """The per-run numbers every summary and test is built from."""
    from .stats import _cost, _duration_s, _tokens

    bucket = _outcome_bucket(run)
    return {
        "bucket": bucket,
        "pass": 1.0 if bucket == "pass" else 0.0 if bucket in ("fail", "error") else None,
        "error": bucket == "error",
        "finding": bool(run.findings),
        "cost": _cost(run),
        "latency": _duration_s(run),
        "tokens": _tokens(run),
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


@dataclass
class PromptVersion:
    """One prompt version and how the agent did under it."""

    id: str
    text: str = ""
    labels: list[str] = field(default_factory=list)
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    run_ids: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def runs(self) -> int:
        return len(self.run_ids)

    def summary(self) -> dict[str, Any]:
        m = self.metrics
        known = [x["pass"] for x in m if x["pass"] is not None]
        n = len(m)

        def avg(key: str) -> float | None:
            return _mean([float(x[key]) for x in m if x[key] is not None])

        return {
            "id": self.id,
            "labels": list(self.labels),
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "runs": n,
            "pass_rate": _mean(known),
            "error_rate": (sum(1 for x in m if x["error"]) / n) if n else None,
            "finding_rate": (sum(1 for x in m if x["finding"]) / n) if n else None,
            "avg_cost_usd": avg("cost"),
            "avg_latency_s": avg("latency"),
            "avg_tokens": avg("tokens"),
            "agents": list(self.agents),
            "chars": len(self.text),
        }


def list_prompts(
    store: Store,
    *,
    since: datetime | str | None = None,
    agent: str | None = None,
    runs: Iterable[Run] | None = None,
) -> tuple[list[PromptVersion], int]:
    """Every prompt version in the store (newest ``last_seen`` first) and the number of
    scanned runs with no system prompt at all. ``runs`` replaces the store scan."""
    from ._runscan import iter_runs
    from .stats import _parse_since

    if runs is None:
        runs = iter_runs(store, since=_parse_since(since), name=None)
    found: dict[str, PromptVersion] = {}
    without = 0
    for run in runs:
        used = run_prompts(run, agent=agent)
        if not used:
            without += 1
            continue
        metrics = run_metrics(run)
        started = run.started_at
        for pid, info in used.items():
            pv = found.get(pid)
            if pv is None:
                pv = found[pid] = PromptVersion(id=pid, text=info["text"])
            pv.run_ids.append(run.run_id)
            pv.metrics.append(metrics)
            for label in info["labels"]:
                if label not in pv.labels:
                    pv.labels.append(label)
            for name in sorted(info["agents"]):
                if name not in pv.agents:
                    pv.agents.append(name)
            if started is not None:
                if pv.first_seen is None or started < pv.first_seen:
                    pv.first_seen = started
                    pv.text = info["text"] or pv.text
                if pv.last_seen is None or started > pv.last_seen:
                    pv.last_seen = started
    out = sorted(found.values(), key=lambda p: p.last_seen.timestamp() if p.last_seen else 0.0)
    out.reverse()
    return out, without


def resolve_prompt(versions: list[PromptVersion], ref: str) -> list[PromptVersion]:
    """Versions matching an id (``p-3f9a1c`` or just ``3f9a1c``, any unique prefix) or a
    label. A label can name several ids (the text changed under one label)."""
    ref = ref.strip()
    if not ref:
        raise PromptError("empty prompt version")
    by_label = [v for v in versions if ref in v.labels]
    if by_label:
        return by_label
    want = ref if ref.startswith("p-") else f"p-{ref}"
    exact = [v for v in versions if v.id == want]
    if exact:
        return exact
    prefix = [v for v in versions if v.id.startswith(want)]
    if len(prefix) > 1:
        raise PromptError(f"{ref!r} is ambiguous: {', '.join(v.id for v in prefix)}")
    if prefix:
        return prefix
    known = ", ".join(sorted({*(v.id for v in versions), *(x for v in versions for x in v.labels)}))
    raise PromptError(f"no prompt version {ref!r}" + (f" (known: {known})" if known else ""))


def diff_prompts(a: PromptVersion, b: PromptVersion) -> str:
    """Unified diff of two prompt versions' texts (as first recorded)."""
    return "".join(
        difflib.unified_diff(
            a.text.splitlines(keepends=True),
            b.text.splitlines(keepends=True),
            fromfile=f"{a.id}" + (f" ({a.labels[0]})" if a.labels else ""),
            tofile=f"{b.id}" + (f" ({b.labels[0]})" if b.labels else ""),
        )
    )


# --- statistics (pure python) ------------------------------------------------------------


def _norm_sf2(z: float) -> float:
    """Two-sided tail probability of a standard normal."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def two_proportion_z(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """Pooled two-proportion z test: ``(z, two-sided p)`` for ``x/n`` against ``x/n``.
    ``z`` is positive when the second proportion is larger. ``(0, 1)`` when undefined."""
    if n1 <= 0 or n2 <= 0:
        return 0.0, 1.0
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (x2 / n2 - x1 / n1) / se
    return z, _norm_sf2(z)


def fisher_exact(x1: int, n1: int, x2: int, n2: int) -> float:
    """Two-sided Fisher exact p-value for ``x1/n1`` against ``x2/n2`` successes."""
    if n1 <= 0 or n2 <= 0:
        return 1.0
    total, hits = n1 + n2, x1 + x2
    denom = math.comb(total, hits)

    def prob(k: int) -> float:
        return math.comb(n1, k) * math.comb(n2, hits - k) / denom

    lo, hi = max(0, hits - n2), min(n1, hits)
    observed = prob(x1)
    p = sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1 + 1e-9))
    return min(1.0, p)


def _ranks(values: list[float]) -> tuple[list[float], list[int]]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    ties: list[int] = []
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        if j > i:
            ties.append(j - i + 1)
        i = j + 1
    return ranks, ties


def _exact_u_p(u: float, n1: int, n2: int) -> float:
    """Two-sided exact Mann-Whitney p (no ties) by counting U's distribution."""

    @functools.cache
    def count(m: int, n: int) -> tuple[int, ...]:
        if m == 0 or n == 0:
            return (1,)
        a, b = count(m - 1, n), count(m, n - 1)
        size = m * n + 1
        out = [0] * size
        for k, v in enumerate(a):  # the largest value is from sample 1: adds n to U
            if k + n < size:
                out[k + n] += v
        for k, v in enumerate(b):
            if k < size:
                out[k] += v
        return tuple(out)

    dist = count(n1, n2)
    total = sum(dist)
    ui = int(round(u))
    tail = sum(dist[: ui + 1]) if ui <= n1 * n2 / 2 else sum(dist[ui:])
    return min(1.0, 2 * tail / total)


def mann_whitney(a: list[float], b: list[float]) -> tuple[float, float]:
    """Mann-Whitney U test: ``(U of the first sample, two-sided p)``.

    Exact for samples of at most 12 values each with no ties, otherwise the normal
    approximation with tie and continuity corrections (the same as R's
    ``wilcox.test(exact = FALSE)``). ``(U, 1)`` when either sample is empty or all
    values are identical."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    ranks, ties = _ranks([*map(float, a), *map(float, b)])
    u1 = sum(ranks[:n1]) - n1 * (n1 + 1) / 2
    n = n1 + n2
    if not ties and n1 <= 12 and n2 <= 12:
        return u1, _exact_u_p(u1, n1, n2)
    mean = n1 * n2 / 2
    tie_term = sum(t**3 - t for t in ties) / (n * (n - 1))
    var = n1 * n2 / 12 * ((n + 1) - tie_term)
    if var <= 0:
        return u1, 1.0
    z = (abs(u1 - mean) - 0.5) / math.sqrt(var)
    return u1, _norm_sf2(max(z, 0.0))


# --- A/B ---------------------------------------------------------------------------------


@dataclass
class ABResult:
    """``crashlab prompts ab``: prompt version ``a`` (the baseline) against ``b``."""

    metric: str
    a: str
    b: str
    a_ids: list[str]
    b_ids: list[str]
    n_a: int
    n_b: int
    value_a: float | None
    value_b: float | None
    delta: float | None
    p_value: float | None
    test: str
    alpha: float
    higher_is_better: bool
    verdict: str  # better | worse | inconclusive  (b relative to a)
    low_sample: bool = False
    overlap_runs: int = 0
    note: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d.update(d.pop("extra"))
        return d


def _metric_value(
    run: Run, metric: str, judge: Callable[[Run], float | None] | None
) -> float | None:
    if metric == "pass_rate":
        return run_metrics(run)["pass"]
    if metric in ("cost", "latency"):
        return run_metrics(run)[metric]
    assert judge is not None
    return judge(run)


def check_metric(metric: str) -> str:
    if metric in METRICS or (metric.startswith("judge:") and metric[6:].strip()):
        return metric
    raise PromptError(f"unknown metric {metric!r}: use pass_rate, cost, latency or judge:<name>")


def compare_prompts(
    store: Store,
    a: str,
    b: str,
    *,
    metric: str = "pass_rate",
    since: datetime | str | None = None,
    agent: str | None = None,
    alpha: float = 0.05,
    runs: Iterable[Run] | None = None,
    judge_fn: Callable[[Any], str] | None = None,
    judge_model: str | None = None,
) -> ABResult:
    """Compare prompt version ``a`` (id or label) with ``b`` on ``metric``:
    ``pass_rate``, ``cost``, ``latency`` or ``judge:<name>`` (a built-in or registered
    judge; verdicts are cached, ``judge_fn`` replaces the model call).

    A run that used both versions is left out of both groups. The verdict describes ``b``
    against ``a``: ``better`` / ``worse`` when the difference is significant at ``alpha``,
    otherwise ``inconclusive``. Raises :class:`PromptError` for an unknown version or
    metric or when a version has no runs with the metric."""
    check_metric(metric)
    if not 0.0 < alpha < 1.0:
        raise PromptError("alpha must be between 0 and 1")
    from ._runscan import iter_runs
    from .stats import _parse_since

    pool = list(
        runs if runs is not None else iter_runs(store, since=_parse_since(since), name=None)
    )
    versions, _ = list_prompts(store, agent=agent, runs=pool)
    va, vb = resolve_prompt(versions, a), resolve_prompt(versions, b)
    ids_a, ids_b = {v.id for v in va}, {v.id for v in vb}
    if ids_a & ids_b:
        raise PromptError(f"{a!r} and {b!r} are the same prompt version")
    judge = (
        _judge_scorer(metric[6:].strip(), judge_fn, judge_model, store) if ":" in metric else None
    )
    group_a: list[tuple[Run, float | None]] = []
    group_b: list[tuple[Run, float | None]] = []
    overlap = 0
    for run in pool:
        used = set(run_prompts(run, agent=agent))
        in_a, in_b = bool(used & ids_a), bool(used & ids_b)
        if in_a and in_b:
            overlap += 1
        elif in_a:
            group_a.append((run, None))
        elif in_b:
            group_b.append((run, None))
    xs = [v for r, _ in group_a if (v := _metric_value(r, metric, judge)) is not None]
    ys = [v for r, _ in group_b if (v := _metric_value(r, metric, judge)) is not None]
    if not xs or not ys:
        raise PromptError(
            f"no {metric} data: {a!r} has {len(xs)} run(s), {b!r} has {len(ys)} run(s)"
        )
    higher = metric != "cost" and metric != "latency"
    extra: dict[str, Any] = {}
    if metric == "pass_rate":
        x1, x2 = int(sum(xs)), int(sum(ys))
        n1, n2 = len(xs), len(ys)
        pooled = (x1 + x2) / (n1 + n2)
        small = min(n1 * pooled, n1 * (1 - pooled), n2 * pooled, n2 * (1 - pooled)) < 5
        if small:
            p, test = fisher_exact(x1, n1, x2, n2), "fisher_exact"
        else:
            p, test = two_proportion_z(x1, n1, x2, n2)[1], "two_proportion_z"
        va_, vb_ = x1 / n1, x2 / n2
        extra = {"passes_a": x1, "passes_b": x2}
    else:
        _, p = mann_whitney(xs, ys)
        test = "mann_whitney_u"
        va_, vb_ = sum(xs) / len(xs), sum(ys) / len(ys)
        extra = {"median_a": _median(xs), "median_b": _median(ys)}
    delta = vb_ - va_
    if p < alpha and delta != 0:
        verdict = "better" if (delta > 0) == higher else "worse"
    else:
        verdict = "inconclusive"
    low = min(len(xs), len(ys)) < 5
    note = None
    if low and verdict == "inconclusive":
        note = "small sample: record more runs under each version before deciding"
    return ABResult(
        metric=metric,
        a=a,
        b=b,
        a_ids=sorted(ids_a),
        b_ids=sorted(ids_b),
        n_a=len(xs),
        n_b=len(ys),
        value_a=va_,
        value_b=vb_,
        delta=delta,
        p_value=p,
        test=test,
        alpha=alpha,
        higher_is_better=higher,
        verdict=verdict,
        low_sample=low,
        overlap_runs=overlap,
        note=note,
        extra=extra,
    )


def _median(values: list[float]) -> float:
    xs = sorted(values)
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


def _judge_scorer(
    name: str, judge_fn: Callable[[Any], str] | None, model: str | None, store: Store
) -> Callable[[Run], float | None]:
    from .judges import get_judge

    try:
        judge = get_judge(name)
    except ValueError as exc:
        raise PromptError(str(exc)) from exc

    def score(run: Run) -> float | None:
        res = judge.evaluate(run, judge_fn=judge_fn, model=model, store=store)
        return None if res.skipped else res.score

    return score
