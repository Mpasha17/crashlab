"""Tools that take a framework context object. Found on a live PydanticAI 'agent
delegation' agent (docs pattern ``async def tool(ctx: RunContext[None], ...)``):
@crashlab.tool deep-copied the RunContext through dataclasses.asdict, hit an RLock
inside it and raised TypeError out of the tool, crashing the agent. Recording must never
crash the agent, and a per-run context object (deps, usage, the whole conversation)
does not belong in the recorded tool arguments or the replay hash."""

from __future__ import annotations

import dataclasses
import threading

import pytest

import crashlab


@dataclasses.dataclass
class Deps:
    db: str
    lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)


@dataclasses.dataclass
class Ctx:
    deps: Deps
    retry: int = 0


def test_dataclass_arg_holding_a_lock_does_not_crash_the_tool():
    @crashlab.tool()
    def lookup(ctx: Ctx, order_id: str) -> str:
        return f"{ctx.deps.db}:{order_id}"

    @crashlab.agent(name="dc")
    def run() -> str:
        return lookup(Ctx(Deps("orders")), "A1")

    assert run() == "orders:A1"
    step = crashlab.load("last").tool_calls("lookup")[0]
    assert step.args["order_id"] == "A1"
    assert step.args["ctx"]["deps"]["db"] == "orders"
    assert step.args["ctx"]["retry"] == 0


def test_dataclass_conversion_matches_asdict_for_plain_dataclasses():
    """Hashes of recordings made before the fix stay the same."""
    from crashlab.capture.normalize import to_jsonable

    @dataclasses.dataclass
    class Inner:
        xs: tuple = (1, 2)
        m: dict = dataclasses.field(default_factory=lambda: {"a": [1, {"b": 2}]})

    @dataclasses.dataclass
    class Outer:
        inner: Inner
        items: list

    v = Outer(Inner(), [Inner(xs=(3,))])
    assert to_jsonable(v) == to_jsonable(dataclasses.asdict(v))


def test_unconvertible_argument_is_recorded_as_a_placeholder_not_raised():
    class Weird:
        @property
        def __dict__(self):  # noqa: D401 - a hostile object
            raise RuntimeError("no")

    @crashlab.tool()
    def t(x: object, n: int) -> int:
        return n

    @crashlab.agent(name="weird")
    def run() -> int:
        return t(Weird(), 3)

    assert run() == 3
    step = crashlab.load("last").tool_calls("t")[0]
    assert step.args["n"] == 3
    assert "Weird" in str(step.args["x"])


def test_ignore_args_drops_a_context_parameter_from_args_and_hash():
    @crashlab.tool(ignore_args=("ctx",))
    def lookup(ctx: Ctx, order_id: str) -> str:
        return order_id

    @crashlab.agent(name="ign")
    def run(retry: int) -> str:
        return lookup(Ctx(Deps("orders"), retry=retry), "A1")

    run(0)
    first = crashlab.load("last").tool_calls("lookup")[0]
    assert first.args == {"order_id": "A1"}
    run(5)
    second = crashlab.load("last").tool_calls("lookup")[0]
    assert second.request_hash == first.request_hash


def _fake_pydantic_ai_ctx():
    cls = type("RunContext", (), {"__module__": "pydantic_ai._run_context"})
    obj = cls()
    obj.deps = None
    obj.lock = threading.RLock()
    return obj


@crashlab.tool()
def joke_factory(ctx, count: int) -> list:
    return ["a"] * count


@crashlab.agent(name="pai-ctx")
def pai_ctx_agent() -> list:
    return joke_factory(_fake_pydantic_ai_ctx(), 2)


def test_pydantic_ai_run_context_is_not_recorded():
    assert pai_ctx_agent() == ["a", "a"]
    src = crashlab.load("last")
    assert src.tool_calls("joke_factory")[0].args == {"count": 2}
    crashlab.replay(src)  # a fresh ctx object on replay: same hash, served offline
    rep = crashlab.load("last")
    assert rep.replay_of == src.run_id and rep.status == "ok"


@pytest.mark.requires_providers("pydantic_ai")
def test_real_pydantic_ai_run_context_type_is_detected():
    from pydantic_ai import RunContext

    from crashlab.capture.tools import is_framework_context

    assert is_framework_context(RunContext.__new__(RunContext))
