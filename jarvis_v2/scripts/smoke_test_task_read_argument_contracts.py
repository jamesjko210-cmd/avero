from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jarvis_v2.agent.types import Plan, PlannedAction, ToolResult
from jarvis_v2.scripts.test_runtime import make_temp_runtime


TASK_READ_TOOLS = (
    "list_tasks",
    "inspect_task",
    "search_tasks",
    "overdue_tasks",
    "task_overview",
    "next_task",
    "task_board",
)

EXPECTED_SHAPES = {
    "list_tasks": (
        ("status", ("string",), False),
        ("limit", ("integer", "string"), False),
    ),
    "inspect_task": (("task_id", ("integer", "string"), True),),
    "search_tasks": (
        ("query", ("string",), True),
        ("status", ("string",), False),
        ("limit", ("integer", "string"), False),
    ),
    "overdue_tasks": (("limit", ("integer", "string"), False),),
    "task_overview": (("limit", ("integer", "string"), False),),
    "next_task": (),
    "task_board": (("limit", ("integer", "string"), False),),
}


class StaticPlanner:
    def __init__(self, action: PlannedAction):
        self.action = action

    def plan(self, _user_input: str) -> Plan:
        return Plan("Exercise task read argument preflight.", [self.action])


def _install_recording_handlers(runtime: Any) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    for name in TASK_READ_TOOLS:
        tool = runtime.registry.get(name)

        def handler(args: dict[str, Any], *, tool_name: str = name) -> ToolResult:
            calls.append((tool_name, dict(args)))
            return ToolResult(
                tool_name,
                True,
                f"Mocked {tool_name} result.",
                {
                    "calls_model": False,
                    "executes_tools": False,
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


def _assert_canonical_defaults_and_legacy_inputs(runtime: Any, calls: list[tuple[str, dict[str, Any]]]) -> None:
    cases = (
        ("list_tasks", {"status": "all", "limit": 4}),
        ("list_tasks", {}),
        ("list_tasks", {"limit": "4"}),
        ("inspect_task", {"task_id": 7}),
        ("inspect_task", {"task_id": "7"}),
        ("search_tasks", {"query": "launch", "status": "open", "limit": 3}),
        ("search_tasks", {"query": "launch"}),
        ("search_tasks", {"query": "launch", "limit": "3"}),
        ("overdue_tasks", {}),
        ("overdue_tasks", {"limit": "2"}),
        ("task_overview", {}),
        ("task_overview", {"limit": "2"}),
        ("next_task", {}),
        ("task_board", {}),
        ("task_board", {"limit": "2"}),
    )
    for name, args in cases:
        before = len(calls)
        result = runtime.executor.execute(
            PlannedAction(name, args, "Task read typed-argument compatibility smoke.")
        )
        if (
            not result.ok
            or result.metadata.get("handler_invoked") is not True
            or calls[before:] != [(name, args)]
        ):
            raise SystemExit(f"{name} rejected compatible arguments: {args} / {result}")


def _assert_runtime_preflight(runtime: Any, calls: list[tuple[str, dict[str, Any]]]) -> None:
    malformed: tuple[tuple[str, Any, str, list[str]], ...] = (
        ("list_tasks", {"status": ["open"]}, "type_mismatch", ["status"]),
        ("list_tasks", {"limit": False}, "type_mismatch", ["limit"]),
        ("inspect_task", {}, "missing_required", ["task_id"]),
        ("inspect_task", {"task_id": True}, "type_mismatch", ["task_id"]),
        ("inspect_task", {"task_id": 1.5}, "type_mismatch", ["task_id"]),
        ("search_tasks", {}, "missing_required", ["query"]),
        ("search_tasks", {"query": 7}, "type_mismatch", ["query"]),
        ("search_tasks", {"query": "launch", "limit": True}, "type_mismatch", ["limit"]),
        (
            "search_tasks",
            {"query": "launch", "private": "hidden"},
            "unknown_arguments",
            ["<unknown>"],
        ),
        ("overdue_tasks", {"limit": []}, "type_mismatch", ["limit"]),
        ("task_overview", {"unexpected": 1}, "unknown_arguments", ["<unknown>"]),
        ("next_task", {"limit": 1}, "unknown_arguments", ["<unknown>"]),
        ("task_board", {"limit": True}, "type_mismatch", ["limit"]),
        ("task_board", ["not", "an", "object"], "arguments_not_object", []),
    )
    for index, (name, args, expected_status, expected_keys) in enumerate(malformed):
        before = len(calls)
        runtime.planner = StaticPlanner(
            PlannedAction(name, args, "Task read real-runtime preflight smoke.")  # type: ignore[arg-type]
        )
        result = runtime.handle(f"exercise malformed task read arguments {index}")
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
            raise SystemExit(f"{name} malformed arguments reached execution: {args} / {item}")
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


def _assert_planner_aliases(runtime: Any, calls: list[tuple[str, dict[str, Any]]], planner: Any) -> None:
    runtime.planner = planner
    cases = {
        "my todo list": ("list_tasks", {"status": "open"}),
        "show task 7": ("inspect_task", {"task_id": 7}),
        "search for the task about groceries": (
            "search_tasks",
            {"query": "groceries", "status": "all"},
        ),
        "tasks overdue": ("overdue_tasks", {}),
        "todos summary": ("task_overview", {}),
        "pick next todo": ("next_task", {}),
        "todo kanban": ("task_board", {}),
    }
    for command, expected in cases.items():
        before = len(calls)
        plan = planner.plan(command)
        actual_plan = [(action.tool_name, action.args) for action in plan.actions]
        if actual_plan != [expected]:
            raise SystemExit(f"task read planner alias drifted: {command!r} -> {actual_plan}")
        result = runtime.handle(command)
        if (
            len(result.tool_results) != 1
            or result.tool_results[0].metadata.get("handler_invoked") is not True
            or calls[before:] != [expected]
        ):
            raise SystemExit(f"task read planner alias missed typed runtime: {command!r}")


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-task-read-contract-") as temp:
        runtime = make_temp_runtime(Path(temp))
        planner = runtime.planner
        _assert_contract_shapes(runtime)
        calls = _install_recording_handlers(runtime)
        _assert_canonical_defaults_and_legacy_inputs(runtime, calls)
        _assert_runtime_preflight(runtime, calls)
        _assert_planner_aliases(runtime, calls, planner)
    print("Task read argument contract smoke passed")


if __name__ == "__main__":
    main()
