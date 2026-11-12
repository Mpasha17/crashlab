"""Models / dataclasses nested in a plain dict or list come back typed on replay.

Live finding (Haystack AgentTool used as a tool): the sub-agent tool returns
``{"messages": [ChatMessage, ...]}``. Replay served the recorded JSON dicts, Haystack's
output handler failed on ``messages[-1].text`` ('dict' object has no attribute 'text'),
fell back to str(), and the orchestrator's next request diverged from the recording."""

from __future__ import annotations

import dataclasses
import enum
from typing import Any

from pydantic import BaseModel

import crashlab
from crashlab.capture.tools import RESULT_TYPES


class Role(enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclasses.dataclass
class Text:
    text: str


@dataclasses.dataclass
class Call:
    tool_name: str
    arguments: dict[str, Any]


@dataclasses.dataclass
class Message:  # shaped like haystack.dataclasses.ChatMessage
    _role: Role
    _content: list[Text | Call]
    _meta: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def text(self) -> str | None:
        return next((c.text for c in self._content if isinstance(c, Text)), None)


class Order(BaseModel):
    id: str
    total: float


@crashlab.tool()
def sub_agent(question: str):  # no annotation, like Haystack's component invoker
    return {
        "messages": [
            Message(Role.USER, [Text(question)]),
            Message(Role.ASSISTANT, [Call("lookup", {"q": "py"})]),
            Message(Role.ASSISTANT, [Text("Python 3.14.0 on 7 October 2025.")]),
        ],
        "steps": 3,
    }


@crashlab.tool()
def orders() -> dict[str, Any]:
    return {"open": [Order(id="A1", total=1.5)], "count": 1, "when": ["x", ("y", 2)]}


@crashlab.agent(name="nested-typed")
def orchestrator(q: str) -> str:
    msgs = sub_agent(q)["messages"]
    last = msgs[-1]
    o = orders()
    return (
        f"{last.text} | {last._role.value} | {type(msgs[1]._content[0]).__name__} | "
        f"{o['open'][0].total} {o['count']}"
    )


EXPECTED = "Python 3.14.0 on 7 October 2025. | assistant | Call | 1.5 1"


def test_nested_models_and_dataclasses_are_typed_on_replay():
    assert orchestrator("latest python?") == EXPECTED
    run = crashlab.load("last")
    types = run.steps[0].attributes[RESULT_TYPES]
    assert [p for p, _ in types] == [["messages", 0], ["messages", 1], ["messages", 2]]
    assert all(ref.endswith(":Message") for _, ref in types)
    assert run.steps[1].attributes[RESULT_TYPES] == [[["open", 0], f"{__name__}:Order"]]
    replayed = crashlab.replay(run)
    assert replayed.status == "ok" and replayed.output == EXPECTED


def test_plain_results_get_no_type_map():
    @crashlab.tool()
    def plain() -> dict:
        return {"a": [1, {"b": 2}], "c": ("d",)}

    @crashlab.agent(name="nested-plain")
    def run_it() -> Any:
        return plain()

    run_it()
    assert RESULT_TYPES not in crashlab.load("last").steps[0].attributes


def test_an_unimportable_type_is_served_as_recorded_json():
    assert orchestrator("q") == EXPECTED
    run = crashlab.load("last")
    step = run.steps[0]
    step.attributes[RESULT_TYPES] = [[["messages", 2], "no_such_module_xyz:Message"]]
    from crashlab.capture.tools import _revive

    value = _revive(step.response["result"], step, sub_agent)
    assert isinstance(value["messages"][2], dict) and value["messages"][2]["_meta"] == {}
    assert value["messages"][2] == step.response["result"]["messages"][2]


def test_recordings_without_the_type_map_replay_as_before():
    assert orchestrator("q") == EXPECTED
    run = crashlab.load("last")
    step = run.steps[0]
    step.attributes.pop(RESULT_TYPES)
    from crashlab.capture.tools import _revive

    value = _revive(step.response["result"], step, sub_agent)
    assert isinstance(value["messages"][0], dict)
