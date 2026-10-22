"""The seed benchmark's fault catalog and tasks.

Labels and culprit-step rules were fixed *before* the first benchmark run (pre-registered
in bench/README.md). ``split``: refund-bot is the **dev** agent (the heuristics and the
diagnosis prompt were developed on its runs); sql-analyst is the held-out **test** agent
(written for Day 10, never used for tuning).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Fault:
    name: str
    description: str
    labels: frozenset[str]  # accepted top-1 labels
    reference_fix: str


FAULTS: dict[str, Fault] = {
    f.name: f
    for f in (
        Fault(
            "spec_drop",
            "a rule is removed from the system prompt (refund window / revenue definition)",
            frozenset({"TASK_SPEC_VIOLATION"}),
            "restore or strengthen the instruction (system_prompt patch)",
        ),
        Fault(
            "tool_5xx",
            "a tool returns an HTML 502 page instead of data, every call",
            frozenset({"EXTERNAL_FAILURE", "TOOL_ERROR_MISHANDLED", "STEP_REPETITION"}),
            "tell the agent to stop and escalate on non-JSON / 5xx results",
        ),
        Fault(
            "schema_drift",
            "a field/column is renamed upstream (total -> amount_total)",
            frozenset({"TOOL_ERROR_MISHANDLED", "BAD_TOOL_ARGUMENTS"}),
            "update the tool description / prompt to the new field name",
        ),
        Fault(
            "misleading_desc",
            "a tool description is wrong (restocking fee / amounts in cents)",
            frozenset({"BAD_TOOL_ARGUMENTS", "TASK_SPEC_VIOLATION"}),
            "correct the tool description (tool_description patch)",
        ),
    )
}


@dataclass(frozen=True)
class Task:
    name: str  # the @crashlab.agent name
    module: str
    split: str  # "dev" or "test"
    prompts: dict[str, list[str]] = field(default_factory=dict)  # condition -> prompts
    default_prompts: tuple[str, ...] = ()

    def load(self) -> Any:
        return importlib.import_module(self.module)

    def prompts_for(self, condition: str) -> list[str]:
        return list(self.prompts.get(condition) or self.default_prompts)

    def agent_faults(self, condition: str) -> list[str]:
        if condition == "clean":
            return []
        mapping = getattr(self.load(), "BENCH_FAULTS", None) or {condition: condition}
        return [mapping[condition]]

    def run(self, prompt: str, model: str, condition: str) -> Any:
        mod = self.load()
        faults = self.agent_faults(condition)
        if self.name == "refund-bot":
            return mod.run_agent(prompt, model=model, provider="openai", faults=faults)
        return mod.run_agent(prompt, model=model, faults=faults)

    def ground_truth(self, run: Any, condition: str) -> tuple[Any, Any]:
        return self.load().ground_truth(run, condition)


A17 = "Hi, I'd like a refund for order A17 please, it arrived broken."
C03 = "Please refund order C03, the shoes fell apart."
SQL_QS = (
    "What was ShopCo's revenue in March 2026?",
    "How much revenue came from customers in Germany in 2026?",
    "Which country generated the most revenue in Q1 2026 (January to March)?",
)

TASKS: dict[str, Task] = {
    "refund-bot": Task(
        "refund-bot",
        "examples.refund_bot.agent",
        "dev",
        prompts={"spec_drop": [C03], "clean": [A17, C03]},
        default_prompts=(A17,),
    ),
    "sql-analyst": Task(
        "sql-analyst", "examples.sql_analyst.agent", "test", default_prompts=SQL_QS
    ),
}
CONDITIONS = ("clean", *FAULTS)
