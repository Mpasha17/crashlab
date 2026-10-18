"""Compare two runs step by step: ``crashlab.diff(a, b)`` / ``crashlab diff A B``.

Steps are aligned by index. Two steps are *the same* when they are the same operation on
the same model / tool with semantically equal requests and responses. Volatile values that
differ between any two live calls (response ids, tool-call ids, usage, Gemini thought
signatures) are ignored; ``exact`` additionally says whether the request hashes match.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from .models import CheckResult, Run, Step
from .replay.divergence import diff_values

StepStatus = Literal["same", "response", "request", "different", "only_a", "only_b"]

_VOLATILE_KEYS = frozenset(
    {"id", "tool_call_id", "call_id", "signature", "thought_signature", "extra_content"}
)
_VOLATILE_RESPONSE = frozenset({"id", "usage", "model", "provider_finish_reason"})


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in _VOLATILE_KEYS}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def semantic_request(step: Step) -> Any:
    return _strip(step.request)


def semantic_response(step: Step) -> Any:
    resp = {k: v for k, v in (step.response or {}).items() if k not in _VOLATILE_RESPONSE}
    out: dict[str, Any] = {"response": _strip(resp)}
    if step.error is not None:
        out["error"] = {"type": step.error.type, "message": step.error.message}
    return out


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


@dataclass
class StepDiff:
    idx: int
    a: Step | None
    b: Step | None
    status: StepStatus
    fields: list[str] = field(default_factory=list)  # differing request / response fields
    exact: bool = False  # identical request hash (and same response)

    @property
    def same(self) -> bool:
        return self.status == "same"

    def detail(self) -> str:
        """Unified diff of the differing request / response (YAML-like rendering)."""
        a, b = self.a, self.b
        if a is None or b is None or self.status == "same":
            return ""
        if self.status == "different":
            return (f"A: {a.operation} {a.name!r}\nB: {b.operation} {b.name!r}")  # fmt: skip
        if self.status == "request":
            _, diff = diff_values(semantic_request(a), semantic_request(b),
                                  fromfile=f"A step {self.idx} request",
                                  tofile=f"B step {self.idx} request")  # fmt: skip
            return diff
        _, diff = diff_values(semantic_response(a), semantic_response(b),
                              fromfile=f"A step {self.idx} response",
                              tofile=f"B step {self.idx} response")  # fmt: skip
        return diff


@dataclass
class CheckDiff:
    name: str
    a: CheckResult | None
    b: CheckResult | None

    @property
    def changed(self) -> bool:
        return (self.a.status if self.a else None) != (self.b.status if self.b else None)


@dataclass
class RunDiff:
    a: Run
    b: Run
    steps: list[StepDiff]
    checks: list[CheckDiff]

    @property
    def first_divergence(self) -> int | None:
        return next((s.idx for s in self.steps if not s.same), None)

    @property
    def identical_steps(self) -> bool:
        return self.first_divergence is None

    @property
    def relation(self) -> str | None:
        """How the runs are related, e.g. "B is a fork of A at step 0"."""
        a, b = self.a, self.b
        for x, y, xn, yn in ((b, a, "B", "A"), (a, b, "A", "B")):
            if x.forked_from == y.run_id:
                return f"{xn} is a fork of {yn} at step {x.fork_step}"
            if x.replay_of == y.run_id:
                return f"{xn} is a replay of {yn}"
        if a.forked_from and a.forked_from == b.forked_from:
            return f"both are forks of {a.forked_from}"
        return None

    def summary(self) -> str:
        fd = self.first_divergence
        if fd is None:
            return f"all {len(self.steps)} steps are the same"
        sd = next(s for s in self.steps if s.idx == fd)
        what = {
            "response": "same request, different response",
            "request": "different request (" + ", ".join(sd.fields[:4]) + ")",
            "different": "a different call",
            "only_a": "only A has this step",
            "only_b": "only B has this step",
        }[sd.status]
        same = sum(1 for s in self.steps if s.same)
        return f"first divergence at step {fd}: {what}; {same}/{len(self.steps)} steps the same"


def _top_fields(paths: list[str]) -> list[str]:
    out: list[str] = []
    for p in paths:
        top = p.split(".", 1)[0].split("[", 1)[0] or p
        if top not in out:
            out.append(top)
    return out


def compare_steps(a: Step | None, b: Step | None, idx: int) -> StepDiff:
    if a is None:
        return StepDiff(idx, a, b, "only_b")
    if b is None:
        return StepDiff(idx, a, b, "only_a")
    # a chat step is named after its model: a model swap is a request difference
    # (params.model), not a different call
    if a.operation != b.operation or (a.name != b.name and a.operation != "chat"):
        return StepDiff(idx, a, b, "different", ["operation" if a.operation != b.operation
                                                 else "name"])  # fmt: skip
    req_a, req_b = semantic_request(a), semantic_request(b)
    if _canon(req_a) != _canon(req_b):
        paths, _ = diff_values(req_a, req_b)
        return StepDiff(idx, a, b, "request", _top_fields(paths) or ["request"])
    res_a, res_b = semantic_response(a), semantic_response(b)
    exact = a.request_hash == b.request_hash
    if _canon(res_a) != _canon(res_b):
        paths, _ = diff_values(res_a, res_b)
        fields = [p.removeprefix("response.") for p in paths]
        return StepDiff(idx, a, b, "response", _top_fields(fields) or ["response"], exact)
    return StepDiff(idx, a, b, "same", [], exact)


def diff(a: Run, b: Run) -> RunDiff:
    """Step-level comparison of two runs (aligned by step index) plus their checks."""
    sa = {s.idx: s for s in a.steps}
    sb = {s.idx: s for s in b.steps}
    steps = [compare_steps(sa.get(i), sb.get(i), i) for i in sorted(set(sa) | set(sb))]
    ca = {c.name: c for c in a.checks}
    cb = {c.name: c for c in b.checks}
    names = list(ca) + [n for n in cb if n not in ca]
    checks = [CheckDiff(n, ca.get(n), cb.get(n)) for n in names]
    return RunDiff(a, b, steps, checks)


__all__ = ["CheckDiff", "RunDiff", "StepDiff", "compare_steps", "diff"]
