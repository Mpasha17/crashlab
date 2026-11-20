"""Compare a passing run and a failing run, and optionally isolate the causal change.

``bisect_runs(good, bad)`` diffs the two runs across git / code version, model and
sampling params, system prompts, declared tool schemas and descriptions, the agent
roster, recorded inputs, and the step-by-step trajectory (``crashlab.diff``). Factors
are ranked by how close they sit to the first diverging step.

``isolate=True`` builds a candidate :class:`~crashlab.patch` for each reversible
difference (revert the failing run's system prompt, model, params, or a tool
description / schema to the passing run's value) and forks the failing run with each
single patch. The first single revert that restores ``outcome == "pass"`` is the
**minimal causal change**. Forks replay matching tool calls from the cassette and go
live (or hit a mock) only for the patched request; they need a replayable entry point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .compare import RunDiff, diff
from .models import Run, Step
from .patch import (
    ModelParamPatch,
    SystemPromptPatch,
    ToolDescriptionPatch,
    ToolSchemaPatch,
)
from .store import Store

_KIND_ORDER = (
    "trajectory",
    "system_prompt",
    "model",
    "model_params",
    "tool_description",
    "tool_schema",
    "input",
    "agents",
    "code_version",
)


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _store_of(store: Store | Path | str | None) -> Store:
    if store is None:
        return Store()
    if isinstance(store, Store):
        return store
    return Store(store)


def _load(ref: Run | str, store: Store) -> Run:
    return ref if isinstance(ref, Run) else store.load_run(ref)


def _chat(run: Run, *, agent: str | None, model: str | None) -> list[Step]:
    steps = run.chat_steps(agent=agent)
    if model is None:
        return steps
    want = model.strip()
    out = []
    for step in steps:
        params = step.request.get("params") or {}
        name = params.get("model") or step.name or ""
        if want in str(name):
            out.append(step)
    return out


def _system(step: Step) -> str | None:
    system = (step.request or {}).get("system")
    return system if isinstance(system, str) and system else system


def _model_and_params(step: Step) -> tuple[str | None, dict[str, Any]]:
    params = dict((step.request or {}).get("params") or {})
    model = params.pop("model", None) or step.name
    for key in ("messages", "tools"):
        params.pop(key, None)
    return (str(model) if model else None), params


def _tools(step: Step) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for tool in (step.request or {}).get("tools") or []:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        out.setdefault(name, tool)
    return out


def _first_by_agent(steps: list[Step]) -> dict[str | None, Step]:
    seen: dict[str | None, Step] = {}
    for step in steps:
        seen.setdefault(step.agent, step)
    return seen


def _q(value: Any, width: int = 70) -> str:
    text = value if isinstance(value, str) else _canon(value)
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


@dataclass
class Factor:
    """One difference between the passing and failing run."""

    kind: str
    summary: str
    good: Any = None
    bad: Any = None
    first_step: int | None = None
    proximity: int = 0
    agent: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "summary": self.summary,
            "good": self.good,
            "bad": self.bad,
            "first_step": self.first_step,
            "proximity": self.proximity,
            "agent": self.agent,
            "details": self.details,
        }


@dataclass
class IsolationTrial:
    """One single-patch fork of the failing run."""

    factor_kind: str
    patch_description: str
    patch: Any = None
    run_id: str | None = None
    outcome: str | None = None
    restored: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        dumped = None
        if self.patch is not None and hasattr(self.patch, "model_dump"):
            dumped = self.patch.model_dump(exclude_none=True)
        return {
            "factor_kind": self.factor_kind,
            "patch_description": self.patch_description,
            "patch": dumped,
            "run_id": self.run_id,
            "outcome": self.outcome,
            "restored": self.restored,
            "error": self.error,
        }


@dataclass
class BisectReport:
    """Result of :func:`bisect_runs`."""

    good: Run
    bad: Run
    diff: RunDiff
    first_divergence: int | None
    factors: list[Factor] = field(default_factory=list)
    trials: list[IsolationTrial] = field(default_factory=list)
    causal: IsolationTrial | None = None
    isolated: bool = False

    def summary(self) -> str:
        parts = [self.diff.summary()]
        if self.factors:
            kinds = [f.kind for f in self.factors]
            parts.append(f"{len(self.factors)} factor{'s' if len(self.factors) != 1 else ''}: "
                         + ", ".join(kinds))  # fmt: skip
        if self.isolated:
            if self.causal is not None:
                parts.append(f"minimal causal change: {self.causal.patch_description}")
            elif self.trials:
                parts.append("no single revert restored outcome=pass")
            else:
                parts.append("no reversible candidates to isolate")
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "good": self.good.run_id,
            "bad": self.bad.run_id,
            "first_divergence": self.first_divergence,
            "summary": self.summary(),
            "relation": self.diff.relation,
            "outcome": [self.good.outcome, self.bad.outcome],
            "factors": [f.to_dict() for f in self.factors],
            "isolated": self.isolated,
            "causal": self.causal.to_dict() if self.causal is not None else None,
            "trials": [t.to_dict() for t in self.trials],
        }


def _proximity(first_step: int | None, divergence: int | None) -> int:
    if divergence is None:
        return 0 if first_step is None else first_step
    if first_step is None:
        return divergence
    return abs(divergence - first_step)


def _compare_configs(
    good: Run, bad: Run, *, agent: str | None, model: str | None, divergence: int | None
) -> list[Factor]:
    factors: list[Factor] = []
    g_steps, b_steps = _chat(good, agent=agent, model=model), _chat(bad, agent=agent, model=model)
    g_first, b_first = _first_by_agent(g_steps), _first_by_agent(b_steps)

    if good.code_version != bad.code_version:
        factors.append(
            Factor(
                kind="code_version",
                summary=f"code version {_q(good.code_version or '-', 40)} vs "
                f"{_q(bad.code_version or '-', 40)}",
                good=good.code_version,
                bad=bad.code_version,
                first_step=None,
                proximity=_proximity(None, divergence),
            )
        )

    if _canon(good.input) != _canon(bad.input):
        factors.append(
            Factor(
                kind="input",
                summary="recorded inputs differ",
                good=good.input,
                bad=bad.input,
                first_step=0 if (g_steps or b_steps) else None,
                proximity=_proximity(0, divergence),
            )
        )

    g_agents, b_agents = good.agents, bad.agents
    if g_agents != b_agents:
        factors.append(
            Factor(
                kind="agents",
                summary=f"agent roster {g_agents} vs {b_agents}",
                good=g_agents,
                bad=b_agents,
                first_step=None,
                proximity=_proximity(None, divergence),
            )
        )

    agents = list(dict.fromkeys([*g_first, *b_first]))
    if agent is not None:
        agents = [a for a in agents if a == agent or (a is not None and a.startswith(agent))]
    if not agents:
        agents = [None]

    for who in agents:
        gs, bs = g_first.get(who), b_first.get(who)
        if gs is None and bs is None:
            continue
        step_idx = (bs or gs).idx
        label = f" (agent {who})" if who else ""

        g_sys = _system(gs) if gs is not None else None
        b_sys = _system(bs) if bs is not None else None
        if g_sys != b_sys:
            factors.append(
                Factor(
                    kind="system_prompt",
                    summary=f"system prompt{label}: {_q(g_sys or '', 40)!r} vs "
                    f"{_q(b_sys or '', 40)!r}",
                    good=g_sys,
                    bad=b_sys,
                    first_step=step_idx,
                    proximity=_proximity(step_idx, divergence),
                    agent=who,
                )
            )

        g_model, g_params = _model_and_params(gs) if gs is not None else (None, {})
        b_model, b_params = _model_and_params(bs) if bs is not None else (None, {})
        if g_model != b_model:
            factors.append(
                Factor(
                    kind="model",
                    summary=f"model{label}: {g_model} vs {b_model}",
                    good=g_model,
                    bad=b_model,
                    first_step=step_idx,
                    proximity=_proximity(step_idx, divergence),
                    agent=who,
                )
            )
        if _canon(g_params) != _canon(b_params):
            factors.append(
                Factor(
                    kind="model_params",
                    summary=f"model params{label} changed",
                    good=g_params,
                    bad=b_params,
                    first_step=step_idx,
                    proximity=_proximity(step_idx, divergence),
                    agent=who,
                    details={"unset": [k for k in b_params if k not in g_params]},
                )
            )

        g_tools = _tools(gs) if gs is not None else {}
        b_tools = _tools(bs) if bs is not None else {}
        for name in list(dict.fromkeys([*g_tools, *b_tools])):
            gt, bt = g_tools.get(name), b_tools.get(name)
            g_desc = (gt or {}).get("description")
            b_desc = (bt or {}).get("description")
            if g_desc != b_desc:
                factors.append(
                    Factor(
                        kind="tool_description",
                        summary=f"tool {name} description{label} changed",
                        good=g_desc,
                        bad=b_desc,
                        first_step=step_idx,
                        proximity=_proximity(step_idx, divergence),
                        agent=who,
                        details={"tool": name},
                    )
                )
            g_schema = (gt or {}).get("parameters")
            b_schema = (bt or {}).get("parameters")
            if _canon(g_schema) != _canon(b_schema):
                factors.append(
                    Factor(
                        kind="tool_schema",
                        summary=f"tool {name} schema{label} changed",
                        good=g_schema,
                        bad=b_schema,
                        first_step=step_idx,
                        proximity=_proximity(step_idx, divergence),
                        agent=who,
                        details={"tool": name},
                    )
                )

    return factors


def _trajectory_factor(rd: RunDiff) -> Factor | None:
    idx = rd.first_divergence
    if idx is None:
        return None
    step = next(s for s in rd.steps if s.idx == idx)
    return Factor(
        kind="trajectory",
        summary=rd.summary(),
        good={"status": step.status, "fields": step.fields},
        bad={"status": step.status, "fields": step.fields},
        first_step=idx,
        proximity=0,
        details={"status": step.status, "fields": step.fields},
    )


def _sort_factors(factors: list[Factor]) -> list[Factor]:
    rank = {k: i for i, k in enumerate(_KIND_ORDER)}
    return sorted(
        factors, key=lambda f: (f.proximity, rank.get(f.kind, 99), f.kind, f.first_step or 0)
    )


def _tighten_proximity(factors: list[Factor], rd: RunDiff) -> None:
    """A config difference that shows up in the diverging step's fields is proximity 0."""
    idx = rd.first_divergence
    if idx is None:
        return
    step = next((s for s in rd.steps if s.idx == idx), None)
    if step is None:
        return
    fields = set(step.fields or [])
    close = {
        "trajectory": True,
        "system_prompt": "system" in fields,
        "model": step.status == "different" or "params" in fields or "name" in fields,
        "model_params": "params" in fields,
        "tool_description": "tools" in fields,
        "tool_schema": "tools" in fields,
    }
    for factor in factors:
        if close.get(factor.kind):
            factor.proximity = 0


def _candidate_patch(factor: Factor) -> Any | None:
    """A single patch that reverts the failing run toward the passing run, or None."""
    agent = factor.agent
    from_step = factor.first_step
    if factor.kind == "system_prompt" and isinstance(factor.good, str):
        return SystemPromptPatch(op="replace", text=factor.good, agent=agent, from_step=from_step)
    if factor.kind == "model" and isinstance(factor.good, str) and factor.good:
        return ModelParamPatch(model=factor.good, agent=agent, from_step=from_step)
    if factor.kind == "model_params" and isinstance(factor.good, dict):
        unset = [k for k in (factor.details or {}).get("unset") or [] if isinstance(k, str)]
        if not factor.good and not unset:
            return None
        return ModelParamPatch(
            params=dict(factor.good), unset=unset, agent=agent, from_step=from_step
        )
    tool = (factor.details or {}).get("tool")
    if factor.kind == "tool_description" and isinstance(tool, str) and isinstance(factor.good, str):
        return ToolDescriptionPatch(
            tool=tool, description=factor.good, agent=agent, from_step=from_step
        )
    if factor.kind == "tool_schema" and isinstance(tool, str) and isinstance(factor.good, dict):
        return ToolSchemaPatch(
            tool=tool, parameters=dict(factor.good), agent=agent, from_step=from_step
        )
    return None


def _fork_step(bad: Run, factor: Factor) -> int:
    n = len(bad.steps)
    idx = factor.first_step
    if idx is None:
        return 0
    if n <= 0:
        return 0
    if idx < 0:
        idx = 0
    if idx >= n:
        return max(n - 1, 0)
    return idx


def _isolate(
    bad: Run, factors: list[Factor], store: Store, *, agent: str | None
) -> tuple[list[IsolationTrial], IsolationTrial | None]:
    from .replay import fork

    trials: list[IsolationTrial] = []
    winners: list[IsolationTrial] = []
    for factor in factors:
        patch = _candidate_patch(factor)
        if patch is None:
            continue
        if agent is not None and getattr(patch, "agent", None) not in (None, agent):
            continue
        trial = IsolationTrial(
            factor_kind=factor.kind,
            patch_description=patch.describe(),
            patch=patch,
        )
        try:
            branched = fork(
                bad,
                at_step=_fork_step(bad, factor),
                patch=patch,
                store_dir=store.root,
                raise_errors=False,
            )
            trial.run_id = branched.run_id
            trial.outcome = branched.outcome
            trial.restored = branched.outcome == "pass"
        except Exception as exc:
            trial.error = f"{type(exc).__name__}: {exc}"
        trials.append(trial)
        if trial.restored:
            winners.append(trial)
    causal = min(winners, key=lambda t: next(
        (i for i, f in enumerate(factors) if f.kind == t.factor_kind), 99
    )) if winners else None  # fmt: skip
    return trials, causal


def bisect_runs(
    good: Run | str,
    bad: Run | str,
    *,
    store: Store | Path | str | None = None,
    isolate: bool = False,
    agent: str | None = None,
    model: str | None = None,
) -> BisectReport:
    """Compare a passing run ``good`` with a failing run ``bad``.

    ``good`` and ``bad`` are :class:`~crashlab.models.Run` objects or run ids (or
    unique prefixes / ``"last"``). ``agent`` and ``model`` restrict the config
    comparison (and isolation patches) to that agent's steps, or to chat steps whose
    model name contains ``model``.

    When ``isolate=True``, each reversible difference is applied on its own as a fork
    of ``bad``. A trial that ends with ``outcome == "pass"`` is a causal candidate;
    :attr:`BisectReport.causal` is the highest-ranked one (closest to the divergence).
    """
    st = _store_of(store)
    good_run, bad_run = _load(good, st), _load(bad, st)
    rd = diff(good_run, bad_run)
    divergence = rd.first_divergence
    factors = _compare_configs(good_run, bad_run, agent=agent, model=model, divergence=divergence)
    traj = _trajectory_factor(rd)
    if traj is not None:
        factors.append(traj)
    _tighten_proximity(factors, rd)
    factors = _sort_factors(factors)

    trials: list[IsolationTrial] = []
    causal: IsolationTrial | None = None
    if isolate:
        trials, causal = _isolate(bad_run, factors, st, agent=agent)

    return BisectReport(
        good=good_run,
        bad=bad_run,
        diff=rd,
        first_divergence=divergence,
        factors=factors,
        trials=trials,
        causal=causal,
        isolated=isolate,
    )


__all__ = ["BisectReport", "Factor", "IsolationTrial", "bisect_runs"]
