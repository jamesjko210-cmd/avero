from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from jarvis_v2.agent.types import Plan, PlannedAction, ToolResult
from jarvis_v2.scripts.test_runtime import make_temp_runtime


PERSONAL_KNOWLEDGE_READ_TOOLS = (
    "read_profile",
    "list_preferences",
    "get_preference",
    "list_people",
    "get_person",
    "list_decisions",
    "get_decision",
    "list_goals",
    "goal_status",
    "next_actions",
    "get_memory",
)

EXPECTED_SHAPES = {
    "read_profile": (("max_chars", ("integer",), False),),
    "list_preferences": (
        ("category", ("string",), False),
        ("status", ("string",), False),
        ("limit", ("integer",), False),
    ),
    "get_preference": (
        ("key", ("string",), True),
        ("category", ("string",), False),
    ),
    "list_people": (("limit", ("integer",), False),),
    "get_person": (
        ("name", ("string",), False),
        ("person_id", ("integer",), False),
        ("limit", ("integer",), False),
    ),
    "list_decisions": (
        ("status", ("string",), False),
        ("limit", ("integer",), False),
    ),
    "get_decision": (("decision_id", ("integer",), True),),
    "list_goals": (
        ("status", ("string",), False),
        ("limit", ("integer",), False),
    ),
    "goal_status": (("goal_id", ("integer",), True),),
    "next_actions": (("limit", ("integer",), False),),
    "get_memory": (("memory_id", ("integer",), True),),
}


class StaticPlanner:
    def __init__(self, action: PlannedAction):
        self.action = action

    def plan(self, _user_input: str) -> Plan:
        return Plan("Exercise personal-knowledge read preflight.", [self.action])


def _install_recording_handlers(runtime: Any) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    for name in PERSONAL_KNOWLEDGE_READ_TOOLS:
        tool = runtime.registry.get(name)

        def handler(args: dict[str, Any], *, tool_name: str = name) -> ToolResult:
            calls.append((tool_name, dict(args)))
            return ToolResult(
                tool_name,
                True,
                f"Mocked {tool_name} result.",
                {
                    "reads_private_data": True,
                    "calls_model": False,
                    "queues_approval": False,
                    "executes_side_effect": False,
                    "authorizes_execution": False,
                    "approval_granted": False,
                },
            )

        runtime.registry._tools[name] = replace(tool, handler=handler)
    return calls


def _assert_contract_shapes(runtime: Any) -> None:
    for name, expected in EXPECTED_SHAPES.items():
        contract = runtime.registry.get(name).argument_contract
        if contract is None or contract.allow_unknown is not False:
            raise SystemExit(f"{name} does not have a strict argument contract")
        actual = tuple(
            (
                field.name,
                tuple(sorted(argument_type.value for argument_type in field.types)),
                field.required,
            )
            for field in contract.fields
        )
        if actual != expected:
            raise SystemExit(f"{name} argument shape drifted: {actual}")
        if not isinstance(contract.fields, tuple) or any(
            not isinstance(field.types, frozenset) for field in contract.fields
        ):
            raise SystemExit(f"{name} argument contract is not structurally immutable")
        try:
            contract.allow_unknown = True  # type: ignore[misc]
        except FrozenInstanceError:
            pass
        else:
            raise SystemExit(f"{name} argument contract can be mutated")


def _assert_selectorless_person_fails_before_private_read(runtime: Any) -> None:
    with patch.object(
        runtime.store,
        "get_person",
        side_effect=AssertionError("selectorless get_person reached the private store"),
    ):
        result = runtime.executor.execute(
            PlannedAction(
                "get_person",
                {},
                "Selectorless personal-knowledge semantic preflight smoke.",
            )
        )
    if (
        result.ok
        or result.metadata.get("handler_invoked") is not True
        or result.metadata.get("reason") != "missing_person"
        or result.metadata.get("reads_private_data") is not False
        or "Person name or id is required" not in result.output
    ):
        raise SystemExit(f"selectorless get_person did not fail before private storage: {result}")


def _assert_compatible_inputs(runtime: Any, calls: list[tuple[str, dict[str, Any]]]) -> None:
    cases = (
        ("read_profile", {}),
        ("read_profile", {"max_chars": 1200}),
        ("list_preferences", {}),
        ("list_preferences", {"category": "work", "status": "all", "limit": 5}),
        ("get_preference", {"key": "reply style"}),
        ("get_preference", {"key": "reply style", "category": "communication"}),
        ("list_people", {"limit": 5}),
        ("get_person", {"name": "Maya", "limit": 3}),
        ("get_person", {"person_id": 7}),
        ("list_decisions", {"status": "all", "limit": 5}),
        ("get_decision", {"decision_id": 7}),
        ("list_goals", {"status": "active", "limit": 5}),
        ("goal_status", {"goal_id": 7}),
        ("next_actions", {"limit": 5}),
        ("get_memory", {"memory_id": 7}),
    )
    for name, args in cases:
        before = len(calls)
        result = runtime.executor.execute(
            PlannedAction(name, args, "Personal-knowledge typed compatibility smoke.")
        )
        if (
            not result.ok
            or result.metadata.get("handler_invoked") is not True
            or calls[before:] != [(name, args)]
        ):
            raise SystemExit(f"{name} rejected compatible arguments: {args} / {result}")


def _assert_malformed_preflight(runtime: Any, calls: list[tuple[str, dict[str, Any]]]) -> None:
    malformed: tuple[tuple[str, Any, str, list[str]], ...] = (
        ("read_profile", {"max_chars": False}, "type_mismatch", ["max_chars"]),
        ("read_profile", {"max_chars": "bad"}, "type_mismatch", ["max_chars"]),
        ("read_profile", ["not", "an", "object"], "arguments_not_object", []),
        ("list_preferences", {"status": ["active"]}, "type_mismatch", ["status"]),
        ("list_preferences", {"private_note": "hidden"}, "unknown_arguments", ["<unknown>"]),
        ("get_preference", {}, "missing_required", ["key"]),
        ("get_preference", {"key": 7}, "type_mismatch", ["key"]),
        ("list_people", {"limit": True}, "type_mismatch", ["limit"]),
        ("list_people", {"limit": "many"}, "type_mismatch", ["limit"]),
        ("get_person", {"name": ["Maya"]}, "type_mismatch", ["name"]),
        ("get_person", {"person_id": 1.5}, "type_mismatch", ["person_id"]),
        ("list_decisions", {"status": {"active": True}}, "type_mismatch", ["status"]),
        ("get_decision", {}, "missing_required", ["decision_id"]),
        ("get_decision", {"decision_id": False}, "type_mismatch", ["decision_id"]),
        ("get_decision", {"decision_id": "7"}, "type_mismatch", ["decision_id"]),
        ("list_goals", {"limit": []}, "type_mismatch", ["limit"]),
        ("goal_status", {"goal_id": 1.5}, "type_mismatch", ["goal_id"]),
        ("next_actions", {"unexpected": 1}, "unknown_arguments", ["<unknown>"]),
        ("get_memory", {}, "missing_required", ["memory_id"]),
        ("get_memory", {"memory_id": True}, "type_mismatch", ["memory_id"]),
        ("get_memory", {"memory_id": "7"}, "type_mismatch", ["memory_id"]),
    )
    for index, (name, args, expected_status, expected_keys) in enumerate(malformed):
        before = len(calls)
        runtime.planner = StaticPlanner(
            PlannedAction(name, args, "Personal-knowledge malformed preflight smoke.")  # type: ignore[arg-type]
        )
        result = runtime.handle(f"exercise malformed personal-knowledge read {index}")
        if len(result.tool_results) != 1:
            raise SystemExit(f"{name} runtime preflight lost its rejection result")
        item = result.tool_results[0]
        trace = result.metadata.get("runtime_trace") or {}
        if (
            item.metadata.get("failure_kind") != "tool_arguments_invalid"
            or item.metadata.get("handler_invoked") is not False
            or item.metadata.get("executed_handler") is not False
            or trace.get("executed_handler_count") != 0
            or calls[before:]
        ):
            raise SystemExit(f"{name} malformed arguments reached private reads: {args} / {item}")
        if item.metadata.get("argument_validation_status") != expected_status:
            raise SystemExit(f"{name} returned the wrong preflight status: {args} / {item.metadata}")
        actual_keys = (
            item.metadata.get("unknown_arg_keys")
            if expected_status == "unknown_arguments"
            else item.metadata.get("missing_arg_keys")
            if expected_status == "missing_required"
            else item.metadata.get("type_mismatch_arg_keys")
            if expected_status == "type_mismatch"
            else []
        )
        if actual_keys != expected_keys:
            raise SystemExit(f"{name} returned the wrong rejected keys: {args} / {item.metadata}")
        if "private_note" in str(item.metadata):
            raise SystemExit("unknown private argument name leaked into rejection metadata")


def _assert_planner_aliases(runtime: Any, calls: list[tuple[str, dict[str, Any]]], planner: Any) -> None:
    runtime.planner = planner
    cases = {
        "read my profile": ("read_profile", {}),
        "all preferences": ("list_preferences", {"status": "all"}),
        "show preference reply style": ("get_preference", {"key": "reply style"}),
        "people": ("list_people", {}),
        "person #1 please": ("get_person", {"person_id": 1}),
        "all decisions": ("list_decisions", {"status": "all"}),
        "show decision 1": ("get_decision", {"decision_id": 1}),
        "내 목표 보여줘": ("list_goals", {"status": "active"}),
        "goal 1 status": ("goal_status", {"goal_id": 1}),
        "next actions": ("next_actions", {}),
        "show memory 1": ("get_memory", {"memory_id": 1}),
        "사람 목록": ("list_people", {}),
    }
    for command, expected in cases.items():
        before = len(calls)
        plan = planner.plan(command)
        actual_plan = [(action.tool_name, action.args) for action in plan.actions]
        if actual_plan != [expected]:
            raise SystemExit(f"personal-knowledge planner alias drifted: {command!r} -> {actual_plan}")
        result = runtime.handle(command)
        if (
            len(result.tool_results) != 1
            or result.tool_results[0].metadata.get("handler_invoked") is not True
            or calls[before:] != [expected]
        ):
            raise SystemExit(f"personal-knowledge alias missed typed runtime: {command!r}")


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-personal-knowledge-read-contract-") as temp:
        runtime = make_temp_runtime(Path(temp))
        planner = runtime.planner
        _assert_contract_shapes(runtime)
        _assert_selectorless_person_fails_before_private_read(runtime)
        calls = _install_recording_handlers(runtime)
        _assert_compatible_inputs(runtime, calls)
        _assert_malformed_preflight(runtime, calls)
        _assert_planner_aliases(runtime, calls, planner)
    print("Personal-knowledge read argument contract smoke passed")


if __name__ == "__main__":
    main()
