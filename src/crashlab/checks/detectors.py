"""Built-in runtime checks. Each detector looks at one finished step at a time (plus what
it has seen so far) and returns findings; the same code runs while recording and on a
stored run (``crashlab check``)."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..models import Evidence, Finding, Run, Step
from .config import ChecksConfig
from .schema import validate

QUOTE = 200  # max characters per evidence quote (always a verbatim slice)


# --- helpers -------------------------------------------------------------------------


def _num(v: Any) -> Any:
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, dict):
        return {k: _num(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_num(x) for x in v]
    return v


def canonical(value: Any) -> str:
    """Stable JSON for comparing arguments / results (120 == 120.0, key order ignored)."""
    return json.dumps(_num(value), sort_keys=True, ensure_ascii=False, default=str)


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8", "surrogatepass")).hexdigest()[:12]


def as_text(value: Any) -> str:
    """How a field is quoted: strings as-is, everything else as compact JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def excerpt(text: str, start: int = 0, width: int = QUOTE) -> str:
    """A verbatim slice of ``text`` around ``start`` (no ellipses added)."""
    start = max(0, min(start, len(text)))
    return text[start : start + width]


def _ev(step: Step, fld: str, value: Any, start: int = 0) -> Evidence:
    return Evidence(step_idx=step.idx, field=fld, quote=excerpt(as_text(value), start))


def _steps_text(idxs: list[int]) -> str:
    return ", ".join(str(i) for i in idxs)


def response_text(step: Step) -> str:
    """The reply's text. Recorded replies are ``{"role", "parts": [{"type": "text",
    "content": ...}]}``; ``content`` strings / lists are accepted too."""
    out: list[str] = []
    for msg in (step.response or {}).get("messages") or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            out += [p.get("text") or "" for p in content
                    if isinstance(p, dict) and p.get("type") == "text"]  # fmt: skip
        for p in msg.get("parts") or []:
            if isinstance(p, dict) and p.get("type") == "text":
                text = p.get("content", p.get("text"))
                out.append(text if isinstance(text, str) else "")
    return "\n".join(t for t in out if t)


def _tool_calls(step: Step) -> list[dict[str, Any]]:
    return [c for c in (step.response or {}).get("tool_calls") or [] if isinstance(c, dict)]


# --- result classification -------------------------------------------------------------

_HTML = re.compile(r"<!doctype\s+html|<html[\s>]", re.I)
_TITLE = re.compile(r"<title>(.*?)</title>", re.I | re.S)
_ERROR_TEXT = re.compile(
    r"^\s*(traceback \(most recent call last\)|internal server error|bad gateway|"
    r"service unavailable|gateway time-?out|too many requests|error\b|exception\b|"
    r"\d{3} [a-z][a-z ]+$|http(/\d(\.\d)?)? [45]\d\d\b)",
    re.I,
)
_ERROR_KEYS = ("error", "errors", "exception", "error_message", "errorMessage", "fault")
_STATUS_KEYS = ("status_code", "statusCode", "http_status", "code", "status")
# a status *word* that means the call did not do what was asked ({"status": "rejected"})
_FAILED_STATUS = frozenset({"error", "failed", "failure", "rejected", "denied", "declined"})
_TRUNC_TAIL = ("...", "…", "[truncated]", "(truncated)", "<truncated>")


@dataclass
class ResultIssue:
    reason: str  # html | error_body | empty | truncated
    severity: str
    detail: str
    start: int = 0  # where the evidence quote starts in the result text
    counts_as_failure: bool = False


def classify_result(result: Any, cfg: Any = None) -> list[ResultIssue]:
    """What looks wrong with a tool result (HTML / error page, error body, empty,
    truncated)."""
    on = (lambda name: getattr(cfg, name, True)) if cfg is not None else (lambda name: True)
    issues: list[ResultIssue] = []
    if isinstance(result, str):
        text = result
        m = _HTML.search(text)
        if m and on("html"):
            title = _TITLE.search(text)
            what = f" ({title.group(1).strip()})" if title else ""
            issues.append(ResultIssue("html", "warn", f"an HTML page{what} instead of data",
                                      title.start() if title else m.start(), True))  # fmt: skip
        elif on("error_bodies") and _ERROR_TEXT.match(text) and len(text) < 4000:
            issues.append(ResultIssue("error_body", "warn", "an error message instead of data",
                                      0, True))  # fmt: skip
        elif on("error_bodies") and text.lstrip()[:1] == "{" and len(text) < 4000:
            # many tools return JSON text, not a dict: {"status": "rejected", "error": ...}
            try:
                obj = json.loads(text)
            except ValueError:
                obj = None
            if isinstance(obj, dict):
                issues += [i for i in classify_result(obj, cfg) if i.reason == "error_body"]
        if not text.strip() and on("empty"):
            issues.append(ResultIssue("empty", "info", "an empty string"))
        elif on("truncated") and not issues:
            stripped = text.rstrip()
            if stripped.endswith(_TRUNC_TAIL):
                issues.append(ResultIssue("truncated", "warn", "text that looks cut off",
                                          max(0, len(stripped) - 80)))  # fmt: skip
            elif stripped[:1] in "{[" and len(stripped) > 1:
                try:
                    json.loads(stripped)
                except ValueError:
                    issues.append(
                        ResultIssue(
                            "truncated",
                            "warn",
                            "JSON that does not parse (cut off?)",
                            max(0, len(stripped) - 80),
                        )
                    )
    elif isinstance(result, dict):
        if on("error_bodies"):
            for key in _ERROR_KEYS:
                if result.get(key) not in (None, "", [], {}, False):
                    issues.append(
                        ResultIssue(
                            "error_body", "info", f"an error object ({key!r} is set)", 0, True
                        )
                    )
                    break
            else:
                for key in _STATUS_KEYS:
                    code = result.get(key)
                    if isinstance(code, str) and code.strip().lower() in _FAILED_STATUS:
                        issues.append(
                            ResultIssue(
                                "error_body", "info", f"a failed status ({key}={code!r})", 0, True
                            )
                        )
                        break
                    if isinstance(code, int) and not isinstance(code, bool) and 400 <= code < 600:
                        issues.append(
                            ResultIssue(
                                "error_body", "info", f"an error status ({key}={code})", 0, True
                            )
                        )
                        break
        if not result and on("empty"):
            issues.append(ResultIssue("empty", "info", "an empty object"))
    elif isinstance(result, list):
        if not result and on("empty"):
            issues.append(ResultIssue("empty", "info", "an empty list"))
    elif result is None and on("empty"):
        issues.append(ResultIssue("empty", "info", "nothing (null)"))
    return issues


def shape_of(result: Any) -> dict[str, str] | None:
    """Top-level field -> JSON type for object results (None for anything else)."""
    if not isinstance(result, dict):
        return None
    out = {}
    for k, v in result.items():
        t = "null" if v is None else "boolean" if isinstance(v, bool) else (
            "number" if isinstance(v, (int, float)) else "string" if isinstance(v, str)
            else "array" if isinstance(v, list) else "object" if isinstance(v, dict)
            else type(v).__name__)  # fmt: skip
        out[str(k)] = t
    return out


# --- state shared by the detectors ----------------------------------------------------


@dataclass
class History:
    run: Run
    steps: list[Step] = field(default_factory=list)
    llm_calls: int = 0
    tool_calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    cost_unknown: bool = False
    elapsed_s: float = 0.0
    baseline: Callable[[str], list[dict[str, str]]] | None = None  # tool -> result shapes

    def tool_steps(self, name: str | None = None) -> list[Step]:
        return [s for s in self.steps
                if s.operation == "execute_tool" and (name is None or s.name == name)]  # fmt: skip

    def chat_steps(self) -> list[Step]:
        return [s for s in self.steps if s.operation == "chat"]


class Detector:
    kind = ""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def on_step(self, step: Step, h: History) -> list[Finding]:
        return []

    def on_end(self, run: Run, h: History) -> list[Finding]:
        return []

    def finding(self, **kw: Any) -> Finding:
        return Finding(kind=self.kind, **kw)


# --- step_repetition --------------------------------------------------------------------

_VOLATILE_PARAMS = {
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "n",
    "user",
    "metadata",
    "stream",
    "stream_options",
}
_ID_KEYS = {"id", "tool_call_id", "call_id", "signature", "thought_signature", "extra_content"}


def _strip_ids(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _strip_ids(x) for k, x in v.items() if k not in _ID_KEYS}
    if isinstance(v, list):
        return [_strip_ids(x) for x in v]
    if isinstance(v, str):
        return " ".join(v.split())
    return v


def near_identical_key(request: dict[str, Any]) -> str:
    """Requests that differ only in sampling params, ids or whitespace get the same key."""
    params = {k: v for k, v in (request.get("params") or {}).items() if k not in _VOLATILE_PARAMS}
    return _digest(_strip_ids({**request, "params": params}))


class StepRepetition(Detector):
    kind = "step_repetition"

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        # per-step memo: every new step is compared with all earlier ones, and
        # re-canonicalizing each earlier request (the whole conversation so far) made a
        # long run quadratic in hashing (a 250-step run spent ~10 s here)
        self._memo: dict[tuple[str, int, int], str] = {}

    def _cached(self, kind: str, s: Step, fn: Any, value: Any) -> str:
        key = (kind, s.idx, id(value))
        got = self._memo.get(key)
        if got is None:
            got = self._memo[key] = fn(value)
        return got

    def on_step(self, step: Step, h: History) -> list[Finding]:
        if step.operation == "execute_tool":
            return self._tool(step, h)
        if step.operation == "chat":
            return self._requests(step, h) + self._outputs(step, h)
        return []

    def _tool(self, step: Step, h: History) -> list[Finding]:
        cfg = self.cfg
        if step.name in cfg.ignore_tools or step.source == "blocked":
            return []
        key = self._cached("args", step, canonical, step.args)
        same = [s for s in h.tool_steps(step.name)
                if self._cached("args", s, canonical, s.args) == key
                and s.idx > step.idx - cfg.window and s.source != "blocked"]  # fmt: skip
        if len(same) < cfg.threshold:
            return []
        outcomes = {canonical(s.error.message if s.error else s.result) for s in same}
        idxs = [s.idx for s in same]
        if len(same) >= cfg.critical_at:
            sev = "critical"
        else:
            sev = "warn" if len(outcomes) == 1 else "info"
        tail = ("and got the same result each time" if len(outcomes) == 1
                else "with different results (polling?)")  # fmt: skip
        args = excerpt(as_text(step.args), 0, 120)
        first = same[0]
        ev = [_ev(first, "args", first.args)]
        ev.append(_ev(first, "error.message" if first.error else "result",
                      first.error.message if first.error else first.result))  # fmt: skip
        return [self.finding(
            severity=sev, step=idxs[0], steps=idxs, key=f"tool:{step.name}:{_digest(step.args)}",
            message=f"{step.name} was called {len(same)} times with the same arguments {args} "
                    f"(steps {_steps_text(idxs)}) {tail}",
            evidence=ev,
        )]  # fmt: skip

    def _requests(self, step: Step, h: History) -> list[Finding]:
        key = self._cached("req", step, near_identical_key, step.request)
        chats = [s for s in h.chat_steps()
                 if self._cached("req", s, near_identical_key, s.request) == key]  # fmt: skip
        if len(chats) < self.cfg.request_threshold:
            return []
        prev = chats[-2]
        if prev.error is not None:  # a retry after a provider error is not a loop
            return []
        idxs = [s.idx for s in chats]
        sev = "critical" if len(chats) >= self.cfg.critical_at else "warn"
        last = (step.request.get("messages") or [None])[-1]
        # tools that ran between the two identical requests: the conversation was rebuilt
        # from scratch (a framework retry / restart), not a model stuck in a loop
        between = [s for s in h.tool_steps() if prev.idx < s.idx < step.idx]
        done = [s for s in between if s.error is None and s.source not in ("blocked", "fake")
                and s.attributes.get("crashlab.tool.side_effects")]  # fmt: skip
        if done:
            sev = "critical" if sev == "critical" else "error"
            what = (f"the run went back to the request of step {prev.idx} at step {step.idx} "
                    f"after {done[0].name} (side effects) succeeded at step {done[0].idx}: a "
                    "restart from here repeats that call (retry the failed LLM call, not the "
                    "whole task)")  # fmt: skip
            idxs = sorted({*idxs, done[0].idx})
        elif between:
            what = (f"the same LLM request was sent {len(chats)} times (steps "
                    f"{_steps_text(idxs)}): the conversation was restarted after "
                    f"{len(between)} tool call(s)")  # fmt: skip
        else:
            what = (
                f"the same LLM request was sent {len(chats)} times "
                f"(steps {_steps_text(idxs)}) with no new information in between"
            )
        return [self.finding(
            severity=sev, step=idxs[0], steps=idxs, key=f"request:{key}", message=what,
            evidence=[_ev(step, "request.messages[-1]", last)] if last is not None else [],
        )]  # fmt: skip

    def _outputs(self, step: Step, h: History) -> list[Finding]:
        text = response_text(step)
        if not text.strip() or step.error is not None:
            return []
        run: list[Step] = []
        for s in reversed(h.chat_steps()):
            if s.error is None and response_text(s) == text:
                run.append(s)
            else:
                break
        if len(run) < self.cfg.llm_output_threshold:
            return []
        idxs = sorted(s.idx for s in run)
        return [self.finding(
            severity="warn", step=idxs[0], steps=idxs, key=f"output:{_digest(text)}",
            message=f"the model gave the same reply {len(run)} times in a row "
                    f"(steps {_steps_text(idxs)})",
            evidence=[_ev(step, "response.text", text)],
        )]  # fmt: skip


# --- tool_error_streak ------------------------------------------------------------------


class ToolErrorStreak(Detector):
    kind = "tool_error_streak"

    def __init__(self, cfg: Any, results_cfg: Any = None) -> None:
        super().__init__(cfg)
        self.results_cfg = results_cfg
        self.streaks: dict[str, list[tuple[Step, str]]] = {}

    def _failure(self, step: Step) -> str | None:
        if step.source == "blocked":
            return None
        if step.error is not None:
            return f"{step.error.type.rsplit('.', 1)[-1]}: {step.error.message}"
        if self.cfg.count_error_results:
            for issue in classify_result(step.result, self.results_cfg):
                if issue.counts_as_failure:
                    return f"returned {issue.detail}"
        return None

    def on_step(self, step: Step, h: History) -> list[Finding]:
        if step.operation != "execute_tool":
            return []
        why = self._failure(step)
        if why is None:
            self.streaks.pop(step.name, None)
            return []
        streak = self.streaks.setdefault(step.name, [])
        streak.append((step, why))
        if len(streak) < self.cfg.threshold:
            return []
        idxs = [s.idx for s, _ in streak]
        reasons = Counter(w for _, w in streak)
        common = reasons.most_common(1)[0][0]
        sev = "critical" if len(streak) >= self.cfg.critical_at else "error"
        ev = []
        for s, _ in streak[:3]:
            if s.error is not None:
                ev.append(_ev(s, "error.message", s.error.message))
            else:
                issue = next(i for i in classify_result(s.result, self.results_cfg)
                             if i.counts_as_failure)  # fmt: skip
                ev.append(_ev(s, "result", s.result, issue.start))
        return [self.finding(
            severity=sev, step=idxs[0], steps=idxs, key=f"{step.name}:{idxs[0]}",
            message=f"{step.name} failed {len(streak)} times in a row "
                    f"(steps {_steps_text(idxs)}): {common[:160]}",
            evidence=ev,
        )]  # fmt: skip


# --- bad_tool_args ----------------------------------------------------------------------


class BadToolArgs(Detector):
    kind = "bad_tool_args"

    def on_step(self, step: Step, h: History) -> list[Finding]:
        if step.operation != "chat" or step.error is not None:
            return []
        tools = {t.get("name"): t for t in step.request.get("tools") or []
                 if isinstance(t, dict) and t.get("type", "function") == "function"}  # fmt: skip
        if not tools:
            return []
        out = []
        for i, call in enumerate(_tool_calls(step)):
            name, args = call.get("name"), call.get("arguments")
            fld = f"response.tool_calls[{i}].arguments"
            if name not in tools:
                problems, sev = (
                    [f"unknown tool {name!r} (declared: {', '.join(map(str, tools))})"],
                    "error",
                )  # noqa: E501
            elif isinstance(args, str):
                problems, sev = ["arguments are not valid JSON"], "error"
            else:
                problems = validate(args, tools[name].get("parameters") or {})
                sev = "warn"
            if not problems:
                continue
            out.append(self.finding(
                severity=sev, step=step.idx, steps=[step.idx], key=f"{step.idx}:{i}",
                message=f"the model called {name} with bad arguments: {'; '.join(problems[:4])}",
                evidence=[_ev(step, fld, args)],
            ))  # fmt: skip
        return out


# --- suspicious_tool_result -------------------------------------------------------------


class SuspiciousToolResult(Detector):
    kind = "suspicious_tool_result"

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.shapes: dict[str, list[dict[str, str]]] = {}

    def on_step(self, step: Step, h: History) -> list[Finding]:
        if step.operation != "execute_tool" or step.error is not None:
            return []
        result = step.result
        out = []
        issues = classify_result(result, self.cfg)
        for issue in issues:
            out.append(self.finding(
                severity=issue.severity, step=step.idx, steps=[step.idx],
                key=f"{issue.reason}:{step.name}",
                message=f"{step.name} returned {issue.detail}",
                evidence=[_ev(step, "result", result, issue.start)],
            ))  # fmt: skip
        if self.cfg.drift and not issues:
            out += self._drift(step, h)
        return out

    def _baseline(self, tool: str, h: History) -> list[dict[str, str]]:
        shapes = list(self.shapes.get(tool, []))
        if self.cfg.baseline_runs and h.baseline is not None:
            shapes += h.baseline(tool)
        return shapes

    def _drift(self, step: Step, h: History) -> list[Finding]:
        shape = shape_of(step.result)
        base = self._baseline(step.name, h)
        if shape is None:
            if base and isinstance(step.result, (str, list)) and step.result:
                kind = "text" if isinstance(step.result, str) else "a list"
                return [self.finding(
                    severity="warn", step=step.idx, steps=[step.idx], key=f"drift:{step.name}",
                    message=f"{step.name} returned {kind} where {len(base)} earlier result(s) "
                            f"were objects",
                    evidence=[_ev(step, "result", step.result)],
                )]  # fmt: skip
            return []
        self.shapes.setdefault(step.name, []).append(shape)
        if not base:
            return []
        n = len(base)
        counts = Counter(k for b in base for k in b)
        expected = {k for k, c in counts.items() if c * 2 > n}
        missing = sorted(expected - set(shape))
        new = sorted(set(shape) - set(counts))
        changed = []
        for k in sorted(set(shape) & expected):
            types = Counter(b[k] for b in base if k in b)
            usual = types.most_common(1)[0][0]
            if shape[k] != usual and "null" not in (shape[k], usual):
                changed.append(f"{k!r} is {shape[k]} (was {usual})")
        if not (missing or new or changed):
            return []
        parts = []
        if missing:
            parts.append("missing " + ", ".join(repr(k) for k in missing))
        if new:
            parts.append("new " + ", ".join(repr(k) for k in new))
        parts += changed
        sev = "warn" if missing or changed else "info"
        text = as_text(step.result)
        anchor = text.find(f'"{new[0]}"') if new else 0
        return [self.finding(
            severity=sev, step=step.idx, steps=[step.idx], key=f"drift:{step.name}",
            message=f"{step.name} result changed shape vs {n} earlier result(s): "
                    + "; ".join(parts),
            evidence=[_ev(step, "result", step.result, max(anchor, 0))],
        )]  # fmt: skip


# --- budget -----------------------------------------------------------------------------

_LIMITS = (
    ("max_steps", "steps", lambda h: len(h.steps), "{:,}"),
    ("max_llm_calls", "LLM calls", lambda h: h.llm_calls, "{:,}"),
    ("max_tool_calls", "tool calls", lambda h: h.tool_calls, "{:,}"),
    ("max_tokens", "tokens", lambda h: h.tokens, "{:,}"),
    ("max_cost_usd", "USD", lambda h: h.cost_usd, "${:.4f}"),
    ("max_seconds", "seconds", lambda h: h.elapsed_s, "{:.1f}s"),
)


class Budget(Detector):
    kind = "budget"

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.noted_unknown_cost = False

    def _check(self, step_idx: int | None, h: History) -> list[Finding]:
        out = []
        for name, what, get, fmt in _LIMITS:
            limit = getattr(self.cfg, name)
            if limit is None:
                continue
            if name == "max_cost_usd" and h.cost_unknown:
                if not self.noted_unknown_cost:
                    self.noted_unknown_cost = True
                    out.append(self.finding(
                        severity="info", step=step_idx, key="max_cost_usd:unknown",
                        message="max_cost_usd is set but the cost of some calls is unknown "
                                "(model not in the price table); it is not enforced",
                    ))  # fmt: skip
                continue
            value = get(h)
            idxs = [step_idx] if step_idx is not None else []
            if value > limit:
                out.append(self.finding(
                    severity="critical", step=step_idx, steps=idxs, key=f"{name}:exceeded",
                    message=f"budget exceeded: {fmt.format(value)} {what} > {name} "
                            f"{fmt.format(limit)}".replace(" USD", "").replace(" seconds", ""),
                ))  # fmt: skip
            elif self.cfg.warn_at and value >= self.cfg.warn_at * limit and value > 0:
                out.append(self.finding(
                    severity="warn", step=step_idx, steps=idxs, key=f"{name}:warn",
                    message=f"{round(100 * value / limit)}% of the budget used: "
                            f"{fmt.format(value)} of {name} {fmt.format(limit)}",
                ))  # fmt: skip
        return out

    def on_step(self, step: Step, h: History) -> list[Finding]:
        return self._check(step.idx, h)

    def on_end(self, run: Run, h: History) -> list[Finding]:
        if run.usage.wall_time_ms is not None:
            h.elapsed_s = max(h.elapsed_s, run.usage.wall_time_ms / 1000)
        return self._check(None, h)


# --- length_cutoff ----------------------------------------------------------------------

_LENGTH = {"length", "max_tokens", "max_output_tokens", "model_context_window_exceeded"}


class LengthCutoff(Detector):
    kind = "length_cutoff"

    def on_step(self, step: Step, h: History) -> list[Finding]:
        if step.operation != "chat" or not step.response:
            return []
        reasons = {str(r).lower() for r in [step.response.get("finish_reason"),
                   step.response.get("provider_finish_reason")] if r}  # fmt: skip
        if not reasons & _LENGTH:
            return []
        text = response_text(step)
        calls = _tool_calls(step)
        broken = [c.get("name") for c in calls if isinstance(c.get("arguments"), str)]
        what = (f"; the {', '.join(map(str, broken))} call arguments are cut off" if broken
                else "; the reply is probably cut off")  # fmt: skip
        ev = [_ev(step, "response.text", text, max(0, len(text) - 120))] if text else []
        return [self.finding(
            severity="warn", step=step.idx, steps=[step.idx], key=f"{step.idx}",
            message=f"the model stopped at its output limit (finish_reason="
                    f"{step.response.get('finish_reason')}){what}",
            evidence=ev,
        )]  # fmt: skip


# --- no_llm_calls -----------------------------------------------------------------------


class NoLlmCalls(Detector):
    """Tools ran but no LLM call was recorded: the framework talks to the model through a
    client crashlab does not see (not wrapped, a private copy, or a native engine, e.g.
    DSPy's default ``lm15`` engine). Replay / fork / diagnose then miss the model."""

    kind = "no_llm_calls"

    def on_end(self, run: Run, h: History) -> list[Finding]:
        ops = [s.operation for s in run.steps]
        tools = [s for s in run.steps if s.operation == "execute_tool"]
        if not tools or "chat" in ops or run.replay_of:
            return []
        first = tools[0]
        return [self.finding(
            severity="warn", step=first.idx, steps=[s.idx for s in tools], key="run",
            message=f"{len(tools)} tool call(s) but no LLM call was recorded: wrap the "
                    "client the framework uses (crashlab.wrap / "
                    "crashlab.integrations.litellm.patch()); see examples/integrations/",
        )]  # fmt: skip


# --- failed_side_effect_claim ------------------------------------------------------------

_SUCCESS = re.compile(
    r"\b(has been placed|have been placed|is placed|are placed|placed (your|the|an?) |"
    r"is confirmed|are confirmed|confirmed (your|the)|successfully|order (id|number)|"
    r"could only (get|order|place)|i('ve| have) (ordered|placed|issued|sent|booked)|"
    r"(is|has been|was) (issued|sent|booked|processed|completed|refunded))",
    re.I,
)
_NEGATED = re.compile(
    r"\b(not|n't|unable|cannot|can't|couldn't|failed|fail|wasn't|weren't|no longer|"
    r"rejected|declined|denied|without)\b",
    re.I,
)


def _claim_sentence(text: str) -> str | None:
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        m = _SUCCESS.search(sentence)
        if m and not _NEGATED.search(sentence) and "could only" not in sentence.lower():
            return sentence.strip()
        if m and "could only" in m.group(0).lower():
            return sentence.strip()
    return None


class FailedSideEffectClaim(Detector):
    """Every call of a ``side_effects=True`` tool failed (exception or error result), but
    the final answer reads like it went through: the customer is told an order exists."""

    kind = "failed_side_effect_claim"

    def __init__(self, cfg: Any, results_cfg: Any = None) -> None:
        super().__init__(cfg)
        self.results_cfg = results_cfg

    def _failed(self, step: Step) -> bool:
        if step.error is not None:
            return True
        return any(i.counts_as_failure for i in classify_result(step.result, self.results_cfg))

    def on_end(self, run: Run, h: History) -> list[Finding]:
        effects = [s for s in run.steps if s.operation == "execute_tool"
                   and s.attributes.get("crashlab.tool.side_effects")]  # fmt: skip
        if not effects or any(s.source in ("blocked", "fake") for s in effects):
            return []
        if not all(self._failed(s) for s in effects):
            return []
        out = run.output if isinstance(run.output, str) else None
        if not out:
            return []
        claim = _claim_sentence(out)
        if claim is None:
            return []
        names = sorted({s.name for s in effects})
        last = effects[-1]
        ev = []
        chats = [s for s in run.steps if s.operation == "chat" and claim in response_text(s)]
        if chats:
            text = response_text(chats[-1])
            ev.append(_ev(chats[-1], "response.text", text, text.find(claim)))
        why = (f"{last.error.type.rsplit('.', 1)[-1]}: {last.error.message}" if last.error
               else as_text(last.result))  # fmt: skip
        return [self.finding(
            severity="error", step=last.idx, steps=[s.idx for s in effects], key="run",
            message=f"every {', '.join(names)} call failed ({len(effects)}; last: "
                    f"{why[:100]}) but the final answer reads like it went through: "
                    f"{claim[:120]!r}",
            evidence=ev,
        )]  # fmt: skip


# --- tool_result_instructions ----------------------------------------------------------

# text in a tool result that addresses the model, as prompt injection does; data rarely
# talks to "the assistant" or asks to hide something from the user
_INJECTION = re.compile(
    r"\b(ignore|disregard|forget|override)\s+(all\s+|any\s+|the\s+|your\s+)?(previous|prior|"
    r"above|earlier|system|original)\s+(instructions?|rules?|prompts?|messages?|guidelines)"
    r"|\b(notice|note|message|instructions?|directive|command)s?\s+(to|for)\s+(the\s+)?"
    r"(assistant|ai|agent|model|llm|chatbot)\b"
    r"|\b(system|admin|administrator|developer)\s+(notice|message|instruction|override|"
    r"prompt|directive)s?\b\s*:?\s*.{0,60}?\b(assistant|ai|agent|model|you)\b"
    r"|\b(do\s+not|don't|never)\s+(tell|mention|reveal|disclose|inform|show)\b.{0,40}?"
    r"\b(user|customer|human|them|this|anyone)\b"
    r"|\bregardless\s+of\s+(the\s+|what\s+the\s+)?(user|customer|request|requested|"
    r"instructions?)\b"
    r"|\byou\s+(must|should|are\s+required\s+to)\s+now\b"
    r"|\bnew\s+instructions?\s*:",
    re.I | re.S,
)


class ToolResultInstructions(Detector):
    """A tool result contains text addressed to the model ("ignore previous instructions",
    "SYSTEM NOTICE TO ASSISTANT: ... do not mention this"): likely prompt injection through
    data. ``error`` when a side-effect tool was called after the model read it."""

    kind = "tool_result_instructions"

    def on_end(self, run: Run, h: History) -> list[Finding]:
        out: list[Finding] = []
        for step in run.steps:
            if step.operation != "execute_tool" or step.result is None:
                continue
            text = as_text(step.result)
            m = _INJECTION.search(text)
            if m is None:
                continue
            after = [s for s in run.steps if s.idx > step.idx and s.operation == "execute_tool"
                     and s.attributes.get("crashlab.tool.side_effects")]  # fmt: skip
            acted = (f"; {after[0].name} (side effects) was called after it, at step "
                     f"{after[0].idx}") if after else ""  # fmt: skip
            out.append(self.finding(
                severity="error" if after else "warn", step=step.idx,
                steps=[step.idx] + [s.idx for s in after], key=f"{step.name}:{step.idx}",
                message=f"{step.name} returned text addressed to the model (possible prompt "
                        f"injection): {m.group(0)[:80]!r}{acted}",
                evidence=[_ev(step, "result", text, max(0, m.start() - 20))],
            ))  # fmt: skip
        return out


# --- unhandled_tool_error --------------------------------------------------------------


class UnhandledToolError(Detector):
    """A tool raised and the exception ended the run: the framework never handed the error
    to the model, so no prompt change can make the agent retry or explain it."""

    kind = "unhandled_tool_error"

    def on_end(self, run: Run, h: History) -> list[Finding]:
        if run.error is None or run.replay_of:
            return []
        failed = [s for s in run.steps if s.operation == "execute_tool" and s.error is not None]
        if not failed:
            return []
        last = failed[-1]
        if any(s.operation == "chat" and s.idx > last.idx for s in run.steps):
            return []  # the model got another turn after it
        short = last.error.type.rsplit(".", 1)[-1]
        if short not in run.error.type and last.error.message not in run.error.message:
            return []
        return [self.finding(
            severity="error", step=last.idx, steps=[last.idx], key="run",
            message=f"{last.name} raised {short} and the exception ended the run: the model "
                    "never saw the error, so a prompt fix cannot help. Catch it in the tool "
                    "(return an error result the model can read, or retry) or turn on the "
                    "framework's tool-error handling (LangGraph: ToolNode(tools, "
                    "handle_tool_errors=True); PydanticAI: raise ModelRetry)",
            evidence=[_ev(last, "error", f"{short}: {last.error.message}")],
        )]  # fmt: skip


def build(cfg: ChecksConfig) -> list[Detector]:
    out: list[Detector] = []
    if not cfg.enabled:
        return out
    if cfg.step_repetition.enabled:
        out.append(StepRepetition(cfg.step_repetition))
    if cfg.tool_error_streak.enabled:
        out.append(ToolErrorStreak(cfg.tool_error_streak, cfg.suspicious_tool_result))
    if cfg.bad_tool_args.enabled:
        out.append(BadToolArgs(cfg.bad_tool_args))
    if cfg.suspicious_tool_result.enabled:
        out.append(SuspiciousToolResult(cfg.suspicious_tool_result))
    if cfg.budget.enabled:
        out.append(Budget(cfg.budget))
    if cfg.length_cutoff.enabled:
        out.append(LengthCutoff(cfg.length_cutoff))
    if cfg.no_llm_calls.enabled:
        out.append(NoLlmCalls(cfg.no_llm_calls))
    if cfg.failed_side_effect_claim.enabled:
        out.append(FailedSideEffectClaim(cfg.failed_side_effect_claim, cfg.suspicious_tool_result))
    if cfg.tool_result_instructions.enabled:
        out.append(ToolResultInstructions(cfg.tool_result_instructions))
    if cfg.unhandled_tool_error.enabled:
        out.append(UnhandledToolError(cfg.unhandled_tool_error))
    return out


KINDS = (
    "step_repetition",
    "tool_error_streak",
    "bad_tool_args",
    "suspicious_tool_result",
    "budget",
    "length_cutoff",
    "no_llm_calls",
    "failed_side_effect_claim",
    "tool_result_instructions",
    "unhandled_tool_error",
)
