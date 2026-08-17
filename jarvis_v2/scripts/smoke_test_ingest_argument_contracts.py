from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jarvis_v2.agent.receipts import runtime_result_receipt
from jarvis_v2.agent.types import Plan, PlannedAction, ToolResult
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.registry import tool_argument_contract_summary


INGEST_TOOLS = ("ingest_obsidian_inbox", "recent_file_digest")
EXPECTED_SHAPES = {
    "ingest_obsidian_inbox": (("limit", ("integer",), False, 1, 200),),
    "recent_file_digest": (
        ("hours", ("integer",), False, 1, 720),
        ("limit", ("integer",), False, 1, 100),
    ),
}
PRIVATE_MARKER = "INGEST_PRIVATE_SENTINEL_6d21"
MALFORMED_MARKER = "INGEST_MALFORMED_SENTINEL_91af"


class StaticPlanner:
    def __init__(self, action: PlannedAction):
        self.action = action

    def plan(self, _user_input: str) -> Plan:
        return Plan("Exercise ingest argument preflight.", [self.action], needs_model=False)


def _assert_contract_shapes(runtime: Any) -> None:
    for name, expected in EXPECTED_SHAPES.items():
        tool = runtime.registry.get(name)
        contract = tool.argument_contract
        if contract is None or contract.allow_unknown is not False:
            raise SystemExit(f"{name} does not have a strict argument contract")
        actual = tuple(
            (
                field.name,
                tuple(sorted(argument_type.value for argument_type in field.types)),
                field.required,
                field.minimum,
                field.maximum,
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
        if name == "recent_file_digest" and tool.auto_mutation_contract is not None:
            raise SystemExit(f"read-only digest tool became auto-mutation eligible: {name}")
        summary = tool_argument_contract_summary(tool)
        if name == "ingest_obsidian_inbox" and "limit:integer[1..200]?" not in summary:
            raise SystemExit(f"{name} model contract omitted its bound: {summary}")
        if name == "recent_file_digest" and (
            "hours:integer[1..720]?" not in summary
            or "limit:integer[1..100]?" not in summary
            or "directory" in summary
        ):
            raise SystemExit(f"{name} model contract advertised an unsafe shape: {summary}")


def _install_private_read_fixtures(runtime: Any) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    for name in INGEST_TOOLS:
        tool = runtime.registry.get(name)

        def handler(args: dict[str, Any], *, tool_name: str = name) -> ToolResult:
            # Receipt bindings are private handler custody data, not planner arguments.
            calls.append(
                (
                    tool_name,
                    {
                        key: value
                        for key, value in args.items()
                        if not key.startswith("_auto_mutation_")
                    },
                )
            )
            return ToolResult(
                tool_name,
                True,
                PRIVATE_MARKER,
                {
                    "reads_private_data": True,
                    "writes_files": True,
                    "handler_invoked": True,
                    "private_value": PRIVATE_MARKER,
                },
            )

        runtime.registry._tools[name] = replace(tool, handler=handler)
    return calls


def _assert_valid_inputs(runtime: Any, calls: list[tuple[str, dict[str, Any]]]) -> None:
    cases = (
        ("ingest_obsidian_inbox", {}),
        ("ingest_obsidian_inbox", {"limit": 1}),
        ("ingest_obsidian_inbox", {"limit": 200}),
        ("recent_file_digest", {}),
        ("recent_file_digest", {"hours": 1, "limit": 1}),
        ("recent_file_digest", {"hours": 168, "limit": 100}),
    )
    for name, args in cases:
        before = len(calls)
        result = runtime.executor.execute(
            PlannedAction(name, args, "Valid ingest argument smoke.")
        )
        if not result.ok or calls[before:] != [(name, args)]:
            raise SystemExit(f"{name} rejected compatible arguments: {args} / {result}")


def _assert_malformed_inputs(runtime: Any, calls: list[tuple[str, dict[str, Any]]]) -> None:
    malformed: tuple[tuple[str, Any, str], ...] = (
        ("ingest_obsidian_inbox", {"limit": "50"}, "type_mismatch"),
        ("ingest_obsidian_inbox", {"limit": True}, "type_mismatch"),
        ("ingest_obsidian_inbox", {"limit": 1.5}, "type_mismatch"),
        ("ingest_obsidian_inbox", {"limit": 0}, "type_mismatch"),
        ("ingest_obsidian_inbox", {"limit": -1}, "type_mismatch"),
        ("ingest_obsidian_inbox", {"limit": 201}, "type_mismatch"),
        (
            "ingest_obsidian_inbox",
            {"unknown": MALFORMED_MARKER},
            "unknown_arguments",
        ),
        (
            "ingest_obsidian_inbox",
            {"limit": "bad", "unknown": MALFORMED_MARKER},
            "multiple_errors",
        ),
        (
            "recent_file_digest",
            {"directory": MALFORMED_MARKER},
            "unknown_arguments",
        ),
        ("recent_file_digest", {"hours": "24"}, "type_mismatch"),
        ("recent_file_digest", {"hours": False}, "type_mismatch"),
        ("recent_file_digest", {"hours": 0}, "type_mismatch"),
        ("recent_file_digest", {"hours": 721}, "type_mismatch"),
        ("recent_file_digest", {"limit": 2.5}, "type_mismatch"),
        ("recent_file_digest", {"limit": 0}, "type_mismatch"),
        ("recent_file_digest", {"limit": 101}, "type_mismatch"),
        (
            "recent_file_digest",
            {"private_path": MALFORMED_MARKER},
            "unknown_arguments",
        ),
        ("recent_file_digest", [MALFORMED_MARKER], "arguments_not_object"),
        ("recent_file_digest", MALFORMED_MARKER, "arguments_not_object"),
        ("recent_file_digest", None, "arguments_not_object"),
        ("recent_file_digest", True, "arguments_not_object"),
        ("recent_file_digest", 7, "arguments_not_object"),
    )
    for index, (name, args, expected_status) in enumerate(malformed):
        before = len(calls)
        runtime.planner = StaticPlanner(
            PlannedAction(name, args, "Ingest real-runtime preflight smoke.")  # type: ignore[arg-type]
        )
        result = runtime.handle(f"exercise malformed ingest arguments {index}")
        if len(result.tool_results) != 1:
            raise SystemExit(f"{name} runtime lost its argument rejection")
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
            raise SystemExit(f"malformed {name} arguments reached private reads or writes: {args!r} / {item}")
        expected_public_args = {"<redacted>": "<redacted>"} if args else {}
        if result.plan.actions[0].args != expected_public_args:
            raise SystemExit(
                f"{name} returned malformed private arguments in its public plan: "
                f"{result.plan.actions[0].args!r}"
            )
        receipt = runtime_result_receipt(result, pending_approvals=0)
        rendered = (
            f"{item.output}\n{item.metadata}\n{result.response}\n{result.metadata}"
            f"\n{result.plan}\n{receipt}"
        )
        if MALFORMED_MARKER in rendered or PRIVATE_MARKER in rendered:
            raise SystemExit(f"{name} argument rejection leaked private or malformed content")


def _assert_planner_aliases(
    runtime: Any,
    calls: list[tuple[str, dict[str, Any]]],
    planner: Any,
) -> None:
    runtime.planner = planner
    cases = {
        "ingest inbox": ("ingest_obsidian_inbox", {}),
        "ingest my inbox": ("ingest_obsidian_inbox", {}),
        "recent file digest": ("recent_file_digest", {}),
        "what files changed recently": ("recent_file_digest", {}),
    }
    for command, expected in cases.items():
        plan = planner.plan(command)
        actual = [(action.tool_name, action.args) for action in plan.actions]
        if actual != [expected]:
            raise SystemExit(f"ingest planner alias drifted: {command!r} -> {actual}")
        before = len(calls)
        result = runtime.handle(command)
        if (
            len(result.tool_results) != 1
            or not result.tool_results[0].ok
            or calls[before:] != [expected]
        ):
            raise SystemExit(f"ingest planner alias missed the typed runtime: {command!r}")


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-ingest-contract-") as temp:
        runtime = make_temp_runtime(Path(temp))
        planner = runtime.planner
        _assert_contract_shapes(runtime)
        calls = _install_private_read_fixtures(runtime)
        _assert_valid_inputs(runtime, calls)
        _assert_malformed_inputs(runtime, calls)
        _assert_planner_aliases(runtime, calls, planner)
    print("Ingest argument contract smoke passed")


if __name__ == "__main__":
    main()
