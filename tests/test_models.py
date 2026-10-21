from __future__ import annotations

from crashlab._ids import new_span_id, new_ulid, ulid_timestamp_ms
from crashlab.models import (
    Diagnosis,
    ErrorInfo,
    Evidence,
    Fix,
    Flag,
    Run,
    Step,
    TaxonomyLabel,
    Usage,
    Verification,
)
from crashlab.patch import PatchSet, SystemPromptPatch, ToolResultPatch


def roundtrip(obj):
    cls = type(obj)
    again = cls.model_validate_json(obj.model_dump_json())
    assert again == obj
    return again


def test_taxonomy_labels():
    assert [lbl.value for lbl in TaxonomyLabel] == [
        "TOOL_ERROR_MISHANDLED",
        "BAD_TOOL_ARGUMENTS",
        "WRONG_TOOL_SELECTION",
        "STEP_REPETITION",
        "TASK_SPEC_VIOLATION",
        "EXTERNAL_FAILURE",
        "UNKNOWN",
        "NO_FAILURE",  # a verdict, not a failure label
    ]


def test_run_roundtrip():
    run = Run(
        run_id=new_ulid(),
        name="refund-bot",
        input={"user_msg": "hi"},
        output="done",
        status="ok",
        flags=[Flag(detector="loop", step_idx=3, message="same call 3x")],
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15, steps=2),
        forked_from="PARENT",
        fork_step=2,
        patch=PatchSet.of(
            SystemPromptPatch(op="append", text="Be careful."),
            ToolResultPatch(step=3, result={"total": 1}),
        ),  # fmt: skip
    )
    again = roundtrip(run)
    assert [p.kind for p in again.patch] == ["system_prompt", "tool_result"]


def test_legacy_fork_fields_and_stub_patches_load():
    run = Run.model_validate(
        {"run_id": "X", "name": "n", "parent_run_id": "P", "fork_at": 3,
         "patch": {"kind": "raw", "data": {"a": 1}}}
    )  # fmt: skip
    assert (run.forked_from, run.fork_step, run.patch) == ("P", 3, None)
    one = Run.model_validate({"run_id": "X", "name": "n", "patch": {"kind": "model_params",
                                                                  "model": "m"}})  # fmt: skip
    assert one.patch.patches[0].model == "m"


def test_step_roundtrip():
    step = Step(
        run_id="R",
        idx=0,
        span_id=new_span_id(),
        operation="execute_tool",
        name="get_order",
        attributes={"gen_ai.tool.name": "get_order"},
        request={"args": {"order_id": "A17"}},
        response={"result": {"total": 49.0}},
        error=ErrorInfo(type="ValueError", message="bad", is_tool_error=True),
        request_hash="0" * 64,
    )
    again = roundtrip(step)
    assert again.args == {"order_id": "A17"}
    assert again.result == {"total": 49.0}


def test_diagnosis_fix_verification_roundtrip():
    fix = Fix(
        fix_id="f1", kind="tool_description", target="get_order", after="Returns JSON.",
        rationale="r",
    )  # fmt: skip
    d = Diagnosis(
        run_id="R",
        failing_step=4,
        step_window=(4, 5),
        label=TaxonomyLabel.TOOL_ERROR_MISHANDLED,
        evidence=[Evidence(step_idx=4, field="response.result", quote="502 Bad Gateway")],
        fixes=[fix],
    )
    roundtrip(d)
    v = Verification(run_id="R", fix_id="f1", k=3, verdict="VERIFIED")
    roundtrip(v)


def test_ulid_sortable_and_unique():
    ids = [new_ulid() for _ in range(2000)]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
    assert all(len(i) == 26 for i in ids)
    assert ulid_timestamp_ms(ids[0]) > 1_700_000_000_000
