"""Compare two runs step by step: ``crashlab.diff(a, b)`` / ``crashlab diff A B``.

Steps are aligned by index. Two steps are *the same* when they are the same operation on
the same model / tool with semantically equal requests and responses. Volatile values that
differ between any two live calls (response ids, tool-call ids, usage, Gemini thought
signatures) are ignored; ``exact`` additionally says whether the request hashes match.

:class:`RunDiff` also reports a tool-sequence Levenshtein distance
(:attr:`RunDiff.tool_edit_distance`) and a normalized trajectory similarity
(:attr:`RunDiff.trajectory_similarity`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from .models import CheckResult, Finding, Run, Step
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
    fault = step.attributes.get("crashlab.fault")
    if isinstance(fault, dict) and fault:
        out["fault"] = {
            k: fault[k] for k in ("kind", "tool", "mode", "status", "name") if k in fault
        }
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
class FindingDiff:
    """Runtime-check findings of one kind in A and in B."""

    kind: str
    a: list[Finding]
    b: list[Finding]

    @staticmethod
    def _worst(fs: list[Finding]) -> str | None:
        return max(fs, key=lambda f: f.rank).severity if fs else None

    @property
    def worst_a(self) -> str | None:
        return self._worst(self.a)

    @property
    def worst_b(self) -> str | None:
        return self._worst(self.b)

    @property
    def changed(self) -> bool:
        return (len(self.a), self.worst_a) != (len(self.b), self.worst_b)


@dataclass
class RunDiff:
    a: Run
    b: Run
    steps: list[StepDiff]
    checks: list[CheckDiff]
    findings: list[FindingDiff] = field(default_factory=list)

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

    @property
    def tool_sequence_a(self) -> list[str]:
        return _tool_sequence(self.a)

    @property
    def tool_sequence_b(self) -> list[str]:
        return _tool_sequence(self.b)

    @property
    def tool_edit_distance(self) -> int:
        """Levenshtein edit distance between the two tool-name sequences."""
        return _tool_levenshtein(self.tool_sequence_a, self.tool_sequence_b)[0]

    @property
    def trajectory_similarity(self) -> float:
        """``1 - distance / max(len(a), len(b))``, or ``1.0`` when both sequences are empty."""
        a, b = self.tool_sequence_a, self.tool_sequence_b
        dist, _ = _tool_levenshtein(a, b)
        return _trajectory_similarity(len(a), len(b), dist)

    @property
    def tool_alignment(self) -> list[dict[str, Any]]:
        """Equal / replace / delete / insert ops that turn A's tools into B's."""
        return _tool_levenshtein(self.tool_sequence_a, self.tool_sequence_b)[1]

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable summary (the same shape ``crashlab diff --json`` prints)."""

        def finding_side(fs: list[Finding]) -> list[dict[str, Any]]:
            return [{"severity": f.severity, "steps": f.steps, "message": f.message} for f in fs]

        seq_a, seq_b = self.tool_sequence_a, self.tool_sequence_b
        dist, alignment = _tool_levenshtein(seq_a, seq_b)
        return {
            "a": self.a.run_id,
            "b": self.b.run_id,
            "relation": self.relation,
            "outcome": [self.a.outcome, self.b.outcome],
            "status": [self.a.status, self.b.status],
            "first_divergence": self.first_divergence,
            "summary": self.summary(),
            "identical_steps": self.identical_steps,
            "checks": {
                c.name: [c.a.status if c.a else None, c.b.status if c.b else None]
                for c in self.checks
            },
            "steps": [
                {"idx": s.idx, "status": s.status, "fields": s.fields, "exact": s.exact}
                for s in self.steps
            ],
            "findings": [
                {
                    "kind": f.kind,
                    "worst_a": f.worst_a,
                    "worst_b": f.worst_b,
                    "changed": f.changed,
                    "a": finding_side(f.a),
                    "b": finding_side(f.b),
                }
                for f in self.findings
            ],
            "usage": {"a": self.a.usage.model_dump(), "b": self.b.usage.model_dump()},
            "tool_sequence_a": list(seq_a),
            "tool_sequence_b": list(seq_b),
            "tool_edit_distance": dist,
            "trajectory_similarity": _trajectory_similarity(len(seq_a), len(seq_b), dist),
            "tool_alignment": alignment,
        }


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
    kinds = list(dict.fromkeys([f.kind for f in a.findings] + [f.kind for f in b.findings]))
    findings = [
        FindingDiff(
            k, [f for f in a.findings if f.kind == k], [f for f in b.findings if f.kind == k]
        )
        for k in kinds
    ]
    return RunDiff(a, b, steps, checks, findings)


def _tool_sequence(run: Run) -> list[str]:
    return [s.name for s in run.steps if s.operation == "execute_tool"]


def _trajectory_similarity(len_a: int, len_b: int, distance: int) -> float:
    denom = max(len_a, len_b)
    if denom == 0:
        return 1.0
    return 1.0 - (distance / denom)


def _tool_levenshtein(a: list[str], b: list[str]) -> tuple[int, list[dict[str, Any]]]:
    """Return (edit distance, alignment ops) for two tool-name sequences."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    alignment: list[dict[str, Any]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            alignment.append({"op": "equal", "a": a[i - 1], "b": b[j - 1]})
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            alignment.append({"op": "replace", "a": a[i - 1], "b": b[j - 1]})
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append({"op": "delete", "a": a[i - 1], "b": None})
            i -= 1
        else:
            alignment.append({"op": "insert", "a": None, "b": b[j - 1]})
            j -= 1
    alignment.reverse()
    return dp[n][m], alignment


__all__ = ["CheckDiff", "FindingDiff", "RunDiff", "StepDiff", "compare_steps", "diff"]
