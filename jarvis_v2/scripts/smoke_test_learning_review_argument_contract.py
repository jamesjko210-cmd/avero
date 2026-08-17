from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jarvis_v2.agent.receipts import runtime_result_receipt
from jarvis_v2.agent.types import Plan, PlannedAction, ToolResult
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.memory_curator import make_memory_curator_tools


PRIVATE_MARKER = "LEARNING_REVIEW_PRIVATE_SENTINEL_4f91"
MALFORMED_MARKER = "MALFORMED_ARGUMENT_SENTINEL_8c27"


class StaticPlanner:
    def __init__(self, args: Any):
        self.args = args

    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Exercise the learning-review argument boundary.",
            [PlannedAction("learning_review", self.args, "Argument-boundary smoke.")],
            needs_model=False,
        )


def _assert_contract(runtime: Any) -> None:
    contract = runtime.registry.get("learning_review").argument_contract
    if contract is None or contract.allow_unknown is not False:
        raise SystemExit("learning_review does not have a strict argument contract")
    shape = tuple(
        (
            field.name,
            tuple(sorted(argument_type.value for argument_type in field.types)),
            field.required,
        )
        for field in contract.fields
    )
    if shape != (("limit", ("integer",), False),):
        raise SystemExit(f"learning_review argument shape drifted: {shape}")
    if not isinstance(contract.fields, tuple) or any(
        not isinstance(field.types, frozenset) for field in contract.fields
    ):
        raise SystemExit("learning_review argument contract is not structurally immutable")
    try:
        contract.allow_unknown = True  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise SystemExit("learning_review argument contract can be mutated")


def _install_private_read_fixture(runtime: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    tool = runtime.registry.get("learning_review")

    def handler(args: dict[str, Any]) -> ToolResult:
        calls.append(dict(args))
        return ToolResult(
            "learning_review",
            True,
            PRIVATE_MARKER,
            {
                "reads_private_data": True,
                "handler_invoked": True,
                "private_value": PRIVATE_MARKER,
            },
        )

    runtime.registry._tools["learning_review"] = replace(tool, handler=handler)
    return calls


def _assert_valid_inputs(runtime: Any, calls: list[dict[str, Any]]) -> None:
    for args in ({}, {"limit": 1}, {"limit": 200}):
        before = len(calls)
        result = runtime.executor.execute(
            PlannedAction("learning_review", args, "Valid learning-review argument smoke.")
        )
        if not result.ok or calls[before:] != [args]:
            raise SystemExit(f"learning_review rejected compatible arguments: {args} / {result}")


def _assert_malformed_inputs(runtime: Any, calls: list[dict[str, Any]]) -> None:
    malformed: tuple[tuple[Any, str], ...] = (
        ({"limit": "not-a-number"}, "type_mismatch"),
        ({"limit": True}, "type_mismatch"),
        ({"limit": 1.5}, "type_mismatch"),
        ({"unknown": MALFORMED_MARKER}, "unknown_arguments"),
        ({"limit": "bad", "unknown": MALFORMED_MARKER}, "multiple_errors"),
        ([MALFORMED_MARKER], "arguments_not_object"),
        (MALFORMED_MARKER, "arguments_not_object"),
        (None, "arguments_not_object"),
    )
    for index, (args, expected_status) in enumerate(malformed):
        before = len(calls)
        runtime.planner = StaticPlanner(args)
        result = runtime.handle(f"exercise malformed learning-review arguments {index}")
        if len(result.tool_results) != 1:
            raise SystemExit("learning_review runtime lost its argument rejection")
        item = result.tool_results[0]
        trace = result.metadata.get("runtime_trace") or {}
        if (
            item.ok
            or item.metadata.get("failure_kind") != "tool_arguments_invalid"
            or item.metadata.get("argument_validation_status") != expected_status
            or item.metadata.get("handler_invoked") is not False
            or item.metadata.get("executed_handler") is not False
            or trace.get("executed_handler_count") != 0
            or calls[before:]
        ):
            raise SystemExit(f"malformed learning_review arguments reached private reads: {args!r} / {item}")
        expected_public_args = {"<redacted>": "<redacted>"} if args else {}
        if result.plan.actions[0].args != expected_public_args:
            raise SystemExit(
                f"learning_review returned malformed private arguments in its public plan: {result.plan.actions[0].args!r}"
            )
        receipt = runtime_result_receipt(result, pending_approvals=0)
        rendered = (
            f"{item.output}\n{item.metadata}\n{result.response}\n{result.metadata}"
            f"\n{result.plan}\n{receipt}"
        )
        if MALFORMED_MARKER in rendered or PRIVATE_MARKER in rendered:
            raise SystemExit("learning_review argument rejection leaked private or malformed content")


def _assert_planner_aliases(runtime: Any, calls: list[dict[str, Any]], planner: Any) -> None:
    runtime.planner = planner
    for command in (
        "learning review",
        "show learning review",
        "show my learning review",
    ):
        plan = planner.plan(command)
        actual = [(action.tool_name, action.args) for action in plan.actions]
        if actual != [("learning_review", {})]:
            raise SystemExit(f"learning_review planner alias drifted: {command!r} -> {actual}")
        before = len(calls)
        result = runtime.handle(command)
        if (
            len(result.tool_results) != 1
            or not result.tool_results[0].ok
            or calls[before:] != [{}]
        ):
            raise SystemExit(f"learning_review alias missed the typed runtime: {command!r}")


def _assert_direct_handler_compatibility(runtime: Any) -> None:
    learning_review = make_memory_curator_tools(runtime.store, runtime.vault)[8]
    for args in ({"limit": "bad"}, {"limit": True}):
        result = learning_review(args)
        if not result.ok or result.metadata.get("limit") != 12:
            raise SystemExit(f"direct learning_review compatibility drifted: {args} / {result}")


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-learning-review-contract-") as temp:
        runtime = make_temp_runtime(Path(temp))
        planner = runtime.planner
        _assert_contract(runtime)
        _assert_direct_handler_compatibility(runtime)
        calls = _install_private_read_fixture(runtime)
        _assert_valid_inputs(runtime, calls)
        _assert_malformed_inputs(runtime, calls)
        _assert_planner_aliases(runtime, calls, planner)
    print("Learning-review argument contract smoke passed")


if __name__ == "__main__":
    main()
