"""Restore proposals: when past passing runs of the same agent exist, diff their system
prompt and tool descriptions against the failing run and propose putting the missing or
changed text back where it was (deterministic, no LLM).

A rule restored at its original position works where the same rule appended at the end
of the prompt often does not, so these proposals are ranked before the fix model's.
Only text that most of the past good runs agree on is restored, so per-run details in a
prompt (a customer name, today's date) are not proposed as fixes."""

from __future__ import annotations

import difflib
from typing import Any

from ..models import Run

#: past good runs looked at (newest first)
MAX_GOOD = 10
#: a good run's prompt must be at least this similar to be the same prompt template
MIN_RATIO = 0.5


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _text(value.get("text") if "text" in value else value.get("content"))
    if isinstance(value, (list, tuple)):
        return "\n".join(t for t in (_text(v) for v in value) if t)
    return str(value)


def _chats(run: Run) -> list[Any]:
    return [s for s in run.steps if s.operation == "chat"]


def system_prompts(run: Run) -> list[str]:
    """The distinct system prompts of ``run``'s chat requests, in first-use order."""
    out: list[str] = []
    for s in _chats(run):
        text = _text((s.request or {}).get("system"))
        if text and text not in out:
            out.append(text)
    return out


def tool_descriptions(run: Run) -> dict[str, str]:
    out: dict[str, str] = {}
    for s in _chats(run):
        for t in (s.request or {}).get("tools") or []:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") if isinstance(t.get("function"), dict) else t
            name = fn.get("name")
            if name and name not in out:
                out[name] = str(fn.get("description") or "")
    return out


def good_runs(store: Any, run: Run, limit: int = MAX_GOOD) -> list[Run]:
    """Past passing runs of the same agent, newest first: recorded runs only (no forks,
    replays or verify trials, whose prompts may carry a patch)."""
    if store is None or not run.name:
        return []
    out: list[Run] = []
    for row in store.list_runs(limit=None, name=run.name):
        rid = row["run_id"]
        if rid == run.run_id or row.get("outcome") != "pass":
            continue
        try:
            past = store.load_run(rid)
        except Exception:  # noqa: BLE001 - a broken old run is just not a candidate
            continue
        if past.forked_from or past.replay_of or any(t.startswith("verify") for t in past.tags):
            continue
        out.append(past)
        if len(out) >= limit:
            break
    return out


def _anchor(lines: list[str], i: int) -> str | None:
    """A short substring of ``lines[i]`` found in no other line (None if none is)."""
    line = lines[i].strip()
    if not line:
        return None
    others = [x for j, x in enumerate(lines) if j != i]
    for n in (40, 60, 90, len(line)):
        cut = line[:n]
        if not any(cut in x for x in others):
            return cut
    return None


def _best_match(prompt: str, candidates: list[str]) -> str | None:
    best, score = None, MIN_RATIO
    for c in candidates:
        r = difflib.SequenceMatcher(None, prompt, c, autojunk=False).ratio()
        if r >= score:
            best, score = c, r
    return best


def _prompt_hunks(failing: str, good: str) -> list[tuple[str, dict[str, Any], str]]:
    """``(key, patch, what)`` for each change that turns ``failing`` back into ``good``."""
    a, b = failing.splitlines(), good.splitlines()
    out: list[tuple[str, dict[str, Any], str]] = []
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        new = [x for x in b[j1:j2] if x.strip()]
        if op == "insert" and new:
            patch: dict[str, Any] = {"kind": "system_prompt", "op": "insert_line",
                                     "text": "\n".join(new)}  # fmt: skip
            anchor = _anchor(a, i1 - 1) if i1 > 0 else None
            if anchor is not None:
                patch["after"] = anchor
            elif i1 < len(a) and (nxt := _anchor(a, i1)) is not None:
                patch["before"] = nxt
            else:
                patch["line"] = i1 + 1
            out.append(("+" + "\n".join(new), patch, "missing"))
        elif op == "replace" and new:
            old = a[i1:i2]
            if len(old) == 1 and (m := _anchor(a, i1)) is not None:
                patch = {"kind": "system_prompt", "op": "replace_line", "match": m,
                         "text": "\n".join(new)}  # fmt: skip
            else:
                patch = {"kind": "system_prompt", "op": "substitute", "old": "\n".join(old),
                         "text": "\n".join(b[j1:j2])}  # fmt: skip
            out.append(("~" + "\n".join(old) + "->" + "\n".join(new), patch, "changed"))
        elif op == "delete":
            for i in range(i1, i2):
                if a[i].strip() and (m := _anchor(a, i)) is not None:
                    patch = {"kind": "system_prompt", "op": "delete_line", "match": m}
                    out.append(("-" + a[i], patch, "added"))
    return out


def restore_proposals(run: Run, good: list[Run]) -> list[dict[str, Any]]:
    """Fix-model-shaped proposals (``{"patch", "rationale", "expected_effect", ...}``)
    restoring the system prompt lines and tool descriptions that the past good runs
    ``good`` agree on and the failing ``run`` lacks. Most supported first."""
    if not good:
        return []
    need = max(1, (len(good) + 1) // 2)  # a majority of the good runs
    found: dict[str, dict[str, Any]] = {}
    prompts = system_prompts(run)
    for g in good:
        gp = system_prompts(g)
        for p in prompts:
            match = _best_match(p, gp)
            if match is None or match == p:
                continue
            for key, patch, what in _prompt_hunks(p, match):
                if len(prompts) > 1:  # several agents/prompts: edit only this one
                    first = next((x for x in p.splitlines() if x.strip()), "")
                    if first and sum(first in q for q in prompts) == 1:
                        patch["when"] = first[:60]
                hit = found.setdefault(key, {"patch": patch, "what": what, "runs": []})
                hit["runs"].append(g.run_id)
    fail_tools = tool_descriptions(run)
    for g in good:
        for name, desc in tool_descriptions(g).items():
            if name in fail_tools and desc and desc != fail_tools[name]:
                patch = {"kind": "tool_description", "tool": name, "description": desc}
                hit = found.setdefault(f"tool:{name}:{desc}",
                                       {"patch": patch, "what": "tool", "runs": []})  # fmt: skip
                hit["runs"].append(g.run_id)
    order = {"missing": 0, "changed": 1, "tool": 2, "added": 3}
    kept = [h for h in found.values() if len(set(h["runs"])) >= need]
    kept.sort(key=lambda h: (-len(set(h["runs"])), order[h["what"]]))
    out = []
    for h in kept:
        runs = sorted(set(h["runs"]))
        seen = f"{len(runs)} of {len(good)} past passing run(s) ({', '.join(runs[:3])})"
        why = {
            "missing": f"{seen} had this text at this position; the failing run's prompt lacks it",
            "changed": f"{seen} had this wording; the failing run's prompt changed it",
            "tool": f"{seen} described this tool this way; the failing run's description differs",
            "added": f"{seen} did not have this line; the failing run's prompt added it",
        }[h["what"]]
        out.append({
            "patch": h["patch"],
            "rationale": f"Restore from past good runs: {why}.",
            "expected_effect": "the agent sees the instructions it had when it passed",
            "literal_justification": "restored verbatim from past passing runs",
            "source": "restore",
            "support": runs,
        })  # fmt: skip
    return out
