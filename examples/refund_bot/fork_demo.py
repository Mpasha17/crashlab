"""Fork demo: "what if the refund-window rule had been in the prompt?"

Takes recorded ``policy_drop`` runs (the 30-day rule missing from the system prompt), forks
each at its first LLM call with a ``SystemPromptPatch`` that restores the rule, runs the
fork live ``k`` times and counts how often the checks flip from fail to pass.

    FAULT_POLICY_DROP=1 python -m examples.refund_bot "Please refund order C03, it fell apart"
    python -m examples.refund_bot.fork_demo last -k 3          # uses $OPENAI_* / $MODEL as usual
    python -m examples.refund_bot.fork_demo last -k 3 --control  # also k unpatched forks

The same fork from the CLI (the patch lives in restore_policy.yaml / .json):

    crashlab fork last --at 0 --patch examples/refund_bot/restore_policy.json
    crashlab diff last

Steps before the fork step come from the cassette; ``get_order`` calls identical to the
parent's are served from it too, and ``issue_refund`` never runs live: an identical refund
gets the recorded result, a different one is blocked (and still fails the checks).
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from typing import Any

import crashlab
from crashlab import patch as P

from . import agent as refund

RESTORE_POLICY = P.SystemPromptPatch(
    op="insert_line",
    after="Always look up the order",
    text=f"- {refund.POLICY_LINES[refund.POLICY_DROP_LINE]}",
)


def first_llm_step(run: crashlab.Run) -> int:
    chats = run.chat_steps()
    if not chats:
        raise SystemExit(f"run {run.run_id} has no LLM calls")
    return chats[0].idx


def fork_k(
    run: crashlab.Run,
    k: int,
    patch: object | None,
    client_factory: Callable[[], Any] | None = None,
) -> list[crashlab.Run]:
    """Fork ``run`` at its first LLM call ``k`` times. Without ``client_factory`` the agent
    builds its own client from the environment, as it did when recording."""
    at = first_llm_step(run)
    out = []
    for i in range(k):
        refund.reset_db()
        client = client_factory() if client_factory else None
        branch = crashlab.fork(run, at_step=at, patch=patch, client=client)
        refunds = [s for s in branch.tool_calls("issue_refund") if s.error is None]
        blocked = [s for s in branch.tool_calls("issue_refund") if s.source == "blocked"]
        what = (
            f"refunded {refunds[0].args.get('amount')}"
            if refunds
            else f"tried to refund {blocked[0].args.get('amount')} (blocked)"
            if blocked
            else "no refund"
        )
        err = branch.error.type if branch.error else branch.status
        status = "" if branch.status == "ok" else f" [{err}]"
        print(f"  fork {i + 1}/{k} {branch.run_id}: {branch.outcome:4} {what}{status}"
              f" | {str(branch.output or '')[:90]!r}")  # fmt: skip
        out.append(branch)
    return out


def tally(branch: crashlab.Run) -> str:
    """pass / fail, but a fork that crashed (rate limit, network) is counted on its own: it
    says nothing about the patch."""
    if branch.status == "ok":
        return branch.outcome
    return f"error ({branch.error.type.rsplit('.', 1)[-1]})" if branch.error else "error"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("runs", nargs="*", default=["last"], help="policy_drop run ids (default: last)")
    ap.add_argument("-k", type=int, default=3, help="forks per run (default 3)")
    ap.add_argument("--control", action="store_true", help="also run k forks without the patch")
    ns = ap.parse_args(argv)
    for ref in ns.runs:
        run = crashlab.load(ref)
        model = run.input.get("model")
        print(f"\n== {run.run_id}  model={model}  faults={run.input.get('faults')}")
        print(f"   parent: {run.outcome} ({run.outcome_reason})")
        if "policy_drop" not in (run.input.get("faults") or []):
            print("   (warning: this run was not recorded with the policy_drop fault)")
        diff = P.PatchSet.of(RESTORE_POLICY).render_diff(run, fork_step=first_llm_step(run))
        print("   patch:\n" + "\n".join("     " + ln for ln in diff.splitlines()))
        groups = [("patched", RESTORE_POLICY)] + ([("control", None)] if ns.control else [])
        for label, patch in groups:
            print(f"   {label} forks:")
            branches = fork_k(run, ns.k, patch)
            counts = Counter(tally(b) for b in branches)
            print(f"   {label}: " + ", ".join(f"{n} {o}" for o, n in sorted(counts.items()))
                  + f" (of {len(branches)})")  # fmt: skip


if __name__ == "__main__":
    main()
