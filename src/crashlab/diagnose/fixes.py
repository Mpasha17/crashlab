"""``crashlab.propose_fixes(diagnosis)``: a second LLM call, separate from diagnosis, that
turns a diagnosis into 1-3 ranked typed patches (:mod:`crashlab.patch`), each validated:

- it parses into a known patch type;
- its target exists in the recorded run (a dry run against the run's own requests: the
  tool, the system prompt line, the step);
- it is minimal (size limits; no wholesale prompt replacement);
- it does not leak the test case's answer: request patches that contain literals from
  this run's task data (order ids, amounts, days ...) are rejected unless the proposal
  justifies them or they are already part of the agent's instructions (an overfitting
  guard in the spirit of a leakage critic).

Kept fixes are stored on ``run.diagnosis.fixes`` and saved as patch files under
``<store>/runs/<run_id>/fixes/`` for ``crashlab fork --patch``."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from ..models import Diagnosis, Fix, Run
from ..patch import PatchError, PatchSet, as_patchset, preview_patches
from .compact import compact
from .diagnoser import DiagnoseError, Messages, _ask, cache_key, parse_json
from .heuristics import candidates

FIX_PROMPT_VERSION = "fix-v4"
MAX_FIXES = 3
MAX_PATCH_CHARS = 700  # the patch's own text (prompt lines, descriptions ...)
MAX_PROMPT_LINES = 4
ALLOWED_KINDS = (
    "system_prompt",
    "tool_description",
    "tool_schema",
    "tool_result",
    "model_params",
    "message",
    "inject_message",
)

SYSTEM = """You are crashlab's fix agent. A recorded run of an AI agent failed and was \
diagnosed. Propose 1 to 3 minimal patches, best first, that would make the agent behave \
correctly on this AND similar tasks.

Patch types (JSON objects; "kind" selects the type):
- {"kind": "system_prompt", "op": "append" | "insert_line" | "replace_line" | "substitute", \
"text": "...", "line": N | "after": "<substring of a line>" | "match": "<substring>" | \
"old": "<exact text>", "when": "<optional>"}: edit the system prompt (lines are 1-based). \
In a multi-agent run (the trace shows "system prompt changed"), set "when" to a phrase \
found only in the prompt of the agent to fix, so the other agents' prompts are untouched.
- {"kind": "tool_description", "tool": "<name>", "op": "append" | "replace", \
"description": "..."}: clarify what a tool does or returns.
- {"kind": "tool_schema", "tool": "<name>", "parameters": {<JSON schema>}, "merge": true}: \
tighten a tool's argument schema.
- {"kind": "model_params", "params": {"temperature": 0}}: change sampling parameters.
- {"kind": "message", "step": N, "index": I, "content": "..."}: rewrite one input message \
of LLM step N.
- {"kind": "tool_result", "step": N, "result": ... | "error": "..."}: a counterfactual tool \
result for testing a hypothesis; it proves a cause but is not a fix.

Rules:
1. Fix the cause named in the diagnosis with the smallest change, preferably a general \
rule in the system prompt or a tool description. No wholesale prompt rewrites.
2. Never hard-code this run's specifics (order ids, amounts, dates, names, the expected \
answer): the fix must generalise. If a literal is truly part of a general rule, explain \
it in "literal_justification".
3. Targets must exist: the tools listed in the trace, lines of the system prompt shown.
4. Every check marked PASS in the trace is a requirement too: the fix must keep it \
passing. Do not make the agent refuse or skip a step a passing check expects (e.g. \
"do not place the order" when a check requires an order); prefer a rule that does \
the task within the limit.
5. The agent runs unattended: a rule that makes it ask the user to confirm, or only \
"inform" the user, ends the run without the action. Say what the agent must DO \
(e.g. "order the maximum allowed and tell the customer the limit").
6. Put a new rule next to the related lines of the prompt ("op": "insert_line" with \
"after") rather than appending it after unrelated style rules.
7. expected_effect: what the agent will do differently, in one sentence.

Reply with only this JSON object:
{"fixes": [{"patch": {...}, "rationale": "<why this fixes the cause>", \
"expected_effect": "<one sentence>", "literal_justification": "<optional>"}]}"""

_ID = re.compile(r"\b[A-Z]{1,4}-?\d{2,}\b")
_NUM = re.compile(r"(?<![\w.])\d{2,}(?:\.\d+)?(?![\w.])")


def _walk(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from _walk(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk(v)
    elif value is not None:
        yield str(value)


def _literals(text: str) -> set[str]:
    out = set(_ID.findall(text))
    for n in _NUM.findall(text):
        out.add(n)
        if n.endswith(".0"):
            out.add(n[:-2])
    return out


def instance_literals(run: Run) -> set[str]:
    """Task-instance literals of ``run``: ids and numbers from its input and its tool
    arguments / results, minus those already in its instructions (system prompt, tool
    descriptions), which are policy rather than the test case."""
    data = " ".join(_walk(run.input))
    for s in run.steps:
        if s.operation == "execute_tool":
            data += " " + " ".join(_walk(s.args)) + " " + " ".join(_walk(s.result))
    found = _literals(data)
    chats = [s for s in run.steps if s.operation == "chat"]
    if chats:
        req = chats[0].request or {}
        policy = " ".join(_walk(req.get("system"))) + " " + " ".join(_walk(req.get("tools")))
        found -= _literals(policy)
    return found


def _patch_text(patch: Any) -> str:
    data = patch.model_dump(mode="json", exclude_none=True)
    data.pop("kind", None)
    return " ".join(_walk(data))


def _target(patch: Any) -> str:
    if getattr(patch, "tool", None):
        return str(patch.tool)
    if patch.kind == "system_prompt":
        return "system prompt"
    if getattr(patch, "step", None) is not None:
        return f"step {patch.step}"
    return patch.kind


def _fork_step(patch: Any, run: Run) -> int:
    step = patch.target_step()
    if step is not None:
        return int(step)
    chats = [s.idx for s in run.steps if s.operation == "chat"]
    return chats[0] if chats else 0


#: system_prompt fields each op ignores; a proposal that sets one is confused (e.g. an
#: ``after`` anchor on an ``append``), so it is flagged and the field dropped
_UNUSED_BY_OP = {
    "replace": ("line", "after", "before", "match", "old"),
    "append": ("line", "after", "before", "match", "old"),
    "prepend": ("line", "after", "before", "match", "old"),
    "insert_line": ("match", "old"),
    "replace_line": ("after", "before", "old"),
    "delete_line": ("after", "before", "old"),
    "substitute": ("line", "after", "before", "match"),
}


def stray_fields(patch: Any) -> list[str]:
    """Fields set on ``patch`` that its op ignores (only system_prompt has such fields;
    the other kinds forbid unknown keys outright)."""
    if getattr(patch, "kind", None) != "system_prompt":
        return []
    return [f for f in _UNUSED_BY_OP.get(patch.op, ()) if getattr(patch, f) is not None]


def validate_fix(item: Any, run: Run, rank: int) -> tuple[Fix | None, str | None]:
    """``(Fix, None)`` for an acceptable proposal, else ``(None, reason)``."""
    if not isinstance(item, dict):
        return None, "a proposal is not a JSON object"
    raw = item.get("patch", item if "kind" in item else None)
    if not isinstance(raw, dict):
        return None, "a proposal has no patch object"
    kind = raw.get("kind")
    if kind not in ALLOWED_KINDS:
        return None, f"patch kind {kind!r} is not one of {', '.join(ALLOWED_KINDS)}"
    try:
        patches = as_patchset(raw)
    except (ValueError, TypeError) as e:
        return None, f"{kind} patch does not parse: {' '.join(str(e).split())[:200]}"
    patch = patches.patches[0]
    stray = stray_fields(patch)
    if stray:
        patch = patch.model_copy(update=dict.fromkeys(stray))
    label = patch.describe()
    if patch.kind == "system_prompt" and patch.op in ("replace", "prepend"):
        return None, f"{label}: not minimal (rewrites the whole prompt; edit a line instead)"
    text = _patch_text(patch)
    if len(text) > MAX_PATCH_CHARS:
        return None, f"{label}: not minimal ({len(text)} chars, limit {MAX_PATCH_CHARS})"
    if patch.kind == "system_prompt" and len(patch.text.splitlines()) > MAX_PROMPT_LINES:
        return None, f"{label}: not minimal (more than {MAX_PROMPT_LINES} new lines)"
    tools = {
        t.get("name")
        for s in run.steps
        if s.operation == "chat"
        for t in (s.request or {}).get("tools") or []
        if isinstance(t, dict)
    }
    if getattr(patch, "tool", None) and patch.kind != "tool_result" and patch.tool not in tools:
        return None, f"{label}: the run has no tool {patch.tool!r} (tools: {sorted(tools)})"
    if patch.scope == "step" and patch.step not in {s.idx for s in run.steps}:
        return None, f"{label}: the run has no step {patch.step}"
    at = _fork_step(patch, run)
    try:
        preview = preview_patches(PatchSet(patches=[patch]), run, fork_step=at)
    except PatchError as e:
        return None, f"{label}: does not apply to the recorded run: {e}"
    if not preview and patch.kind != "model_params":
        return None, f"{label}: changes nothing in the recorded run"
    notes: list[str] = []
    if stray:
        notes.append(f"ignored field(s) {stray} that {patch.kind} op {patch.op!r} does not use")
    test_only = patch.kind == "tool_result"
    if not test_only:
        leaked = sorted(_literals(text) & instance_literals(run))
        why = str(item.get("literal_justification") or "").strip()
        if leaked and not why:
            return None, (
                f"{label}: leaks this run's specifics {leaked} (overfits the test case; "
                "a fix must generalise)"
            )
        if leaked:
            notes.append(f"contains run literals {leaked}, justified: {why}")
    else:
        notes.append("counterfactual for testing a hypothesis, not a fix")
    fix = Fix(
        fix_id=f"fix-{rank}",
        rank=rank,
        kind=patch.kind,
        target=_target(patch),
        after=label,
        rationale=str(item.get("rationale") or "").strip(),
        expected_effect=str(item.get("expected_effect") or "").strip(),
        as_patch=patch,
        test_only=test_only,
        notes=notes,
    )
    return fix, None


def _diagnosis_text(d: Diagnosis) -> str:
    lines = [
        f"LABEL: {d.label.value}",
        f"FAILING STEP: {d.failing_step}",
        f"ROOT CAUSE: {d.root_cause or '(none given)'}",
    ]
    lines += [f'EVIDENCE: step {e.step_idx} {e.field}: "{e.quote}"' for e in d.evidence]
    return "\n".join(lines)


def _attempts_text(run: Run) -> str:
    """Fixes already verified on this run, so a new proposal does not repeat a rejected
    one. Empty (and the prompt unchanged) when nothing was verified yet."""
    lines = []
    for a in run.fix_attempts:
        if not a.get("patch") or not a.get("verdict"):
            continue
        stats = f"patched passed {a.get('patched')}"
        if a.get("control"):
            stats += f", control {a['control']}"
        if a.get("regressions"):
            stats += f", {a['regressions']} regression(s)"
        lines.append(f"- {a.get('fix_id')}: {a['patch']} -> {a['verdict']} ({stats})")
    if not lines:
        return ""
    return (
        "\n\nPREVIOUS FIXES (already verified by forking the run; do not propose them "
        "again; a REJECTED one did not work, so try a different target or wording)\n"
        + "\n".join(lines[-6:])
    )


def fix_messages(run: Run, d: Diagnosis, *, max_field: int = 600, hide_input=()) -> Messages:
    trace = compact(run, candidates=candidates(run), max_field=max_field, hide_input=hide_input)
    user = f"DIAGNOSIS\n{_diagnosis_text(d)}\n\nTRACE\n{trace}{_attempts_text(run)}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _attempted_fix_ids(run: Run) -> set[str]:
    return {str(a.get("fix_id")) for a in run.fix_attempts if a.get("fix_id")}


def _number_after_attempts(run: Run, fixes: list[Fix]) -> None:
    """New proposals never reuse the id of a fix that was already verified: its verdict
    (in ``fix_attempts``) and its file stay attached to that patch, not to a new one."""
    nums = [
        int(m.group(1)) for i in _attempted_fix_ids(run) if (m := re.fullmatch(r"fix-(\d+)", i))
    ]
    off = max(nums, default=0)
    for f in fixes:
        f.fix_id = f"fix-{f.rank + off}"


def save_fix_files(store: Any, run: Run, fixes: list[Fix]) -> None:
    folder = Path(store.run_dir(run.run_id)) / "fixes"
    folder.mkdir(parents=True, exist_ok=True)
    keep = _attempted_fix_ids(run)
    for old in folder.glob("fix-*.*"):
        if old.stem not in keep:
            old.unlink()
    try:
        import yaml
    except ImportError:  # pragma: no cover - depends on the environment
        yaml = None
    for f in fixes:
        data = {"patches": [f.as_patch.model_dump(mode="json", exclude_none=True)]}  # type: ignore[union-attr]
        if yaml is not None:
            path = folder / f"{f.fix_id}.yaml"
            head = f"# {f.after}\n# why: {' '.join(f.rationale.split())[:200]}\n"
            path.write_text(head + yaml.safe_dump(data, sort_keys=False), "utf-8")
        else:
            path = folder / f"{f.fix_id}.json"
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", "utf-8")
        f.file = str(path)


def propose_fixes(
    target: Diagnosis | Run | str,
    model: str | None = None,
    *,
    run: Run | None = None,
    client: Any = None,
    fn: Callable[[Messages], Any] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    cache: bool = True,
    save: bool = True,
    store: Any = None,
    max_fixes: int = MAX_FIXES,
    hide_input: Iterable[str] = (),
    max_field: int = 600,
    json_mode: bool = True,
) -> list[Fix]:
    """Propose 1-3 ranked, validated patches for a diagnosed run (a separate LLM call
    from the diagnosis). ``target`` is the diagnosis, the run (its ``run.diagnosis``) or
    a run reference. Rejected proposals are explained in ``diagnosis.fix_notes``.
    Model options are those of :func:`crashlab.diagnose`."""
    import os

    from ..store import Store
    from .diagnoser import ENV_API_KEY, ENV_BASE_URL, ENV_MODEL

    if isinstance(target, Diagnosis):
        d = target
        if run is None:
            store = store or Store()
            run = store.load_run(d.run_id)
    else:
        if isinstance(target, str):
            store = store or Store()
            target = store.load_run(target)
        run = target
        if run.diagnosis is None:
            raise DiagnoseError(f"run {run.run_id} has no diagnosis: run crashlab.diagnose first")
        d = run.diagnosis
    if store is None and (save or cache):
        store = Store()
    in_store = store is not None and store.index.get_run_json(run.run_id) is not None
    if not d.is_failure:
        d.fixes, d.fix_notes = [], [f"no fix proposed: the diagnosis is {d.label.value}"]
        return []

    model = model or os.environ.get(ENV_MODEL) or ("custom" if fn is not None else None)
    if not model:
        raise DiagnoseError(f"no fix model: pass model=..., or set ${ENV_MODEL}")
    msgs = fix_messages(run, d, max_field=max_field, hide_input=tuple(hide_input))
    key = cache_key(f"{FIX_PROMPT_VERSION}:{model}", msgs)
    if cache and d.fixes and d.fix_method.get("cache_key") == key:
        d.fix_method = {**d.fix_method, "cached": True}
        return d.fixes

    reply, info = _ask(
        run,
        msgs,
        model,
        client=client,
        fn=fn,
        base_url=base_url or os.environ.get(ENV_BASE_URL) or None,
        api_key=api_key or os.environ.get(ENV_API_KEY) or None,
        json_mode=json_mode,
        store=store if in_store else None,
        prompt_version=FIX_PROMPT_VERSION,
    )
    raw = parse_json(reply)
    items = raw.get("fixes") if isinstance(raw, dict) else None
    notes: list[str] = []
    if not isinstance(items, list):
        notes.append('the fix model did not return {"fixes": [...]}')
        items = []
    fixes: list[Fix] = []
    for item in items:
        if len(fixes) >= max_fixes:
            notes.append(f"kept the first {max_fixes} valid proposals")
            break
        fix, why = validate_fix(item, run, len(fixes) + 1)
        if fix is None:
            notes.append(f"rejected: {why}")
        elif any(f.as_patch == fix.as_patch for f in fixes):
            notes.append(f"rejected: duplicate of an earlier proposal ({fix.after})")
        else:
            fixes.append(fix)
    _number_after_attempts(run, fixes)
    d.fixes, d.fix_notes = fixes, notes
    d.fix_method = {
        **info,
        "model": model,
        "prompt_version": FIX_PROMPT_VERSION,
        "cache_key": key,
        "proposed": len(items),
        "kept": len(fixes),
        "cached": False,
    }
    if raw is None:
        d.fix_method["raw_reply"] = str(reply)[:500]
    if save and in_store:
        save_fix_files(store, run, fixes)
        store.save_diagnosis(run, d, d.method.get("cache_key"))
    run.diagnosis = d
    return fixes


def try_fix(run: Run, fix: Fix, *, client: Any = None, store: Any = None) -> dict[str, Any]:
    """Fork ``run`` once with ``fix``'s patch, live from the step it applies to, and
    return what the oracles said (full k-rerun verification is separate)."""
    from ..replay import fork

    if fix.as_patch is None:
        raise ValueError(f"{fix.fix_id} has no patch")
    at = _fork_step(fix.as_patch, run)
    root = store.root if store is not None else None
    branch = fork(run, at_step=at, patch=fix.as_patch, client=client, store_dir=root)
    failed = [c.name for c in branch.checks if c.status != "pass"]
    fix.tried = {
        "run_id": branch.run_id,
        "fork_step": at,
        "status": branch.status,
        "outcome": branch.outcome,
        "failed_checks": failed,
        "error": f"{branch.error.type}: {branch.error.message[:200]}" if branch.error else None,
    }
    return fix.tried
