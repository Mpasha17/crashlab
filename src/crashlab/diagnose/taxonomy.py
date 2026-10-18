"""The failure labels the diagnoser must choose from, with the definitions it is shown."""

from __future__ import annotations

from ..models import TaxonomyLabel

DEFINITIONS: dict[TaxonomyLabel, str] = {
    TaxonomyLabel.TOOL_ERROR_MISHANDLED: (
        "A tool returned an error or garbage (an HTML error page, an error message, empty "
        "or wrong data) and the agent ignored it, misread it or passed it on as if it were "
        "valid (e.g. told the user something the tool never said)."
    ),
    TaxonomyLabel.BAD_TOOL_ARGUMENTS: (
        "A tool was called with malformed, schema-invalid or semantically wrong arguments "
        "(wrong type, missing field, wrong id or amount)."
    ),
    TaxonomyLabel.WRONG_TOOL_SELECTION: (
        "The agent used an inappropriate tool, or did not call a tool that the task needed "
        "(e.g. replied, or claimed an action was done, without calling the tool that does it)."
    ),
    TaxonomyLabel.STEP_REPETITION: (
        "The agent repeated the same or an equivalent action (same tool and arguments, or the "
        "same output) without new information or progress."
    ),
    TaxonomyLabel.TASK_SPEC_VIOLATION: (
        "An action or output violates an explicit instruction or constraint: one in the "
        "agent's context (system prompt, user message, tool description) or a business rule "
        "stated by a failed @crashlab.check oracle."
    ),
    TaxonomyLabel.EXTERNAL_FAILURE: (
        "The root cause is outside the agent (an API or tool down, rate limited, wrong "
        "upstream data, the environment) and the agent behaved reasonably given it."
    ),
    TaxonomyLabel.UNKNOWN: "No label can be supported with verbatim evidence from the trace.",
    TaxonomyLabel.NO_FAILURE: (
        "The agent did what the task asked and nothing in the trace shows a failure."
    ),
}

FAILURE_LABELS = [
    lbl for lbl in TaxonomyLabel if lbl not in (TaxonomyLabel.UNKNOWN, TaxonomyLabel.NO_FAILURE)
]


def parse_label(value: object) -> TaxonomyLabel | None:
    """The label for ``value`` ("task_spec_violation", "TASK SPEC VIOLATION" ...) or None."""
    if isinstance(value, TaxonomyLabel):
        return value
    if not isinstance(value, str):
        return None
    key = value.strip().upper().replace(" ", "_").replace("-", "_")
    try:
        return TaxonomyLabel(key)
    except ValueError:
        return None


def render() -> str:
    return "\n".join(f"- {lbl.value}: {text}" for lbl, text in DEFINITIONS.items())
