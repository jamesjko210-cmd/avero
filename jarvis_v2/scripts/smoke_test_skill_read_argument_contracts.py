from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable
from unittest.mock import patch

from jarvis_v2.agent.types import Plan, PlannedAction, RiskLevel, ToolResult
from jarvis_v2.scripts.test_runtime import make_temp_runtime


SKILL_READ_TOOLS = (
    "list_skills",
    "search_skills",
    "get_skill",
)

EXPECTED_SHAPES = {
    "list_skills": (("limit", ("integer",), False),),
    "search_skills": (
        ("query", ("string",), True),
        ("limit", ("integer",), False),
    ),
    "get_skill": (("name", ("string",), True),),
}


class StaticPlanner:
    def __init__(self, action: PlannedAction):
        self.action = action

    def plan(self, _user_input: str) -> Plan:
        return Plan("Exercise saved-skill read argument preflight.", [self.action])


def _install_handler_reach_spies(
    runtime: Any,
) -> tuple[
    list[tuple[str, dict[str, Any]]],
    dict[str, Callable[[dict[str, Any]], ToolResult]],
]:
    handler_calls: list[tuple[str, dict[str, Any]]] = []
    original_handlers: dict[str, Callable[[dict[str, Any]], ToolResult]] = {}
    for name in SKILL_READ_TOOLS:
        tool = runtime.registry.get(name)
        original_handlers[name] = tool.handler

        def handler(
            args: dict[str, Any],
            *,
            tool_name: str = name,
            original_handler: Callable[[dict[str, Any]], ToolResult] = tool.handler,
        ) -> ToolResult:
            handler_calls.append((tool_name, dict(args)))
            return original_handler(args)

        runtime.registry._tools[name] = replace(tool, handler=handler)
    return handler_calls, original_handlers


def _assert_contract_inventory_is_exact_and_immutable(runtime: Any) -> None:
    marked_skill_reads = {
        tool.name
        for tool in runtime.registry.list()
        if tool.toolset == "skills"
        and tool.risk is RiskLevel.READ_ONLY
        and tool.argument_contract is not None
    }
    if marked_skill_reads != set(EXPECTED_SHAPES):
        raise SystemExit(
            f"saved-skill read argument contract inventory drifted: {marked_skill_reads}"
        )
    actual_inventory = {
        name: runtime.registry.get(name).argument_contract for name in SKILL_READ_TOOLS
    }

    for name, expected in EXPECTED_SHAPES.items():
        contract = actual_inventory[name]
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
        try:
            contract.fields[0].required = not contract.fields[0].required  # type: ignore[misc]
        except FrozenInstanceError:
            pass
        else:
            raise SystemExit(f"{name} argument field can be mutated")


def _assert_valid_executor_inputs(
    runtime: Any,
    handler_calls: list[tuple[str, dict[str, Any]]],
) -> None:
    cases = (
        ("list_skills", {}),
        ("list_skills", {"limit": 5}),
        ("search_skills", {"query": "launch"}),
        ("search_skills", {"query": "launch", "limit": 4}),
        ("get_skill", {"name": "Deploy Audit"}),
    )
    for name, args in cases:
        before = len(handler_calls)
        result = runtime.executor.execute(
            PlannedAction(name, args, "Saved-skill read typed compatibility smoke.")
        )
        if (
            not result.ok
            or result.metadata.get("handler_invoked") is not True
            or handler_calls[before:] != [(name, args)]
        ):
            raise SystemExit(f"{name} rejected valid typed arguments: {args} / {result}")


def _assert_malformed_routed_args_stop_before_private_reads(
    runtime: Any,
    handler_calls: list[tuple[str, dict[str, Any]]],
) -> None:
    private_sentinel = "private-skill-argument-sentinel"
    malformed: tuple[tuple[str, Any, str, list[str]], ...] = (
        ("list_skills", ["not", "an", "object"], "arguments_not_object", []),
        (
            "list_skills",
            {"private_skill": private_sentinel},
            "unknown_arguments",
            ["<unknown>"],
        ),
        ("list_skills", {"limit": False}, "type_mismatch", ["limit"]),
        ("list_skills", {"limit": "5"}, "type_mismatch", ["limit"]),
        ("list_skills", {"limit": 1.5}, "type_mismatch", ["limit"]),
        ("search_skills", None, "arguments_not_object", []),
        ("search_skills", {}, "missing_required", ["query"]),
        ("search_skills", {"query": 7}, "type_mismatch", ["query"]),
        (
            "search_skills",
            {"query": "launch", "limit": True},
            "type_mismatch",
            ["limit"],
        ),
        (
            "search_skills",
            {"query": "launch", "limit": "4"},
            "type_mismatch",
            ["limit"],
        ),
        (
            "search_skills",
            {"query": "launch", "private_skill": private_sentinel},
            "unknown_arguments",
            ["<unknown>"],
        ),
        ("get_skill", ["not", "an", "object"], "arguments_not_object", []),
        ("get_skill", {}, "missing_required", ["name"]),
        ("get_skill", {"name": False}, "type_mismatch", ["name"]),
        ("get_skill", {"name": 7}, "type_mismatch", ["name"]),
        (
            "get_skill",
            {"name": "Deploy Audit", "private_skill": private_sentinel},
            "unknown_arguments",
            ["<unknown>"],
        ),
    )
    private_store_calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def list_store_spy(*args: Any, **kwargs: Any) -> list[Any]:
        private_store_calls.append(("list_skills", args, kwargs))
        return []

    def search_store_spy(*args: Any, **kwargs: Any) -> list[Any]:
        private_store_calls.append(("search_skills", args, kwargs))
        return []

    def get_store_spy(*args: Any, **kwargs: Any) -> None:
        private_store_calls.append(("get_skill", args, kwargs))
        return None

    with (
        patch.object(runtime.store, "list_skills", side_effect=list_store_spy),
        patch.object(runtime.store, "search_skills", side_effect=search_store_spy),
        patch.object(runtime.store, "get_skill", side_effect=get_store_spy),
    ):
        for index, (name, args, expected_status, expected_keys) in enumerate(malformed):
            before_handlers = len(handler_calls)
            before_store = len(private_store_calls)
            runtime.planner = StaticPlanner(
                PlannedAction(
                    name,
                    args,
                    "Saved-skill malformed routed-argument smoke.",
                )  # type: ignore[arg-type]
            )
            result = runtime.handle(f"exercise malformed saved-skill read {index}")
            if len(result.tool_results) != 1:
                raise SystemExit(f"{name} runtime preflight lost its rejection result")
            item = result.tool_results[0]
            trace = result.metadata.get("runtime_trace") or {}
            if (
                item.metadata.get("failure_kind") != "tool_arguments_invalid"
                or item.metadata.get("handler_invoked") is not False
                or item.metadata.get("executed_handler") is not False
                or trace.get("executed_handler_count") != 0
                or handler_calls[before_handlers:]
                or private_store_calls[before_store:]
            ):
                raise SystemExit(
                    f"{name} malformed arguments reached a handler or private store: {item}"
                )
            if item.metadata.get("argument_validation_status") != expected_status:
                raise SystemExit(
                    f"{name} returned the wrong preflight status: {item.metadata}"
                )
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
                raise SystemExit(
                    f"{name} returned the wrong rejected keys: {item.metadata}"
                )
            rejected_metadata = str(item.metadata)
            if private_sentinel in rejected_metadata or "private_skill" in rejected_metadata:
                raise SystemExit("saved-skill argument rejection leaked private input")


def _assert_command_first_aliases(
    runtime: Any,
    handler_calls: list[tuple[str, dict[str, Any]]],
    planner: Any,
) -> None:
    runtime.planner = planner
    cases = (
        ("list skills", ("list_skills", {})),
        ("show my recent skills please", ("list_skills", {})),
        ("search skills for launch", ("search_skills", {"query": "launch"})),
        ("read skill Deploy Audit", ("get_skill", {"name": "Deploy Audit"})),
        ("내 스킬 보여줘", ("list_skills", {})),
        ("스킬 보여줘", ("list_skills", {})),
        ("스킬 목록", ("list_skills", {})),
        ("내 스킬", ("list_skills", {})),
    )
    for command, expected in cases:
        before = len(handler_calls)
        plan = planner.plan(command)
        actual_plan = [(action.tool_name, action.args) for action in plan.actions]
        if actual_plan != [expected]:
            raise SystemExit(f"saved-skill planner alias drifted: {command!r} -> {actual_plan}")
        result = runtime.handle(command)
        if (
            len(result.tool_results) != 1
            or result.tool_results[0].metadata.get("handler_invoked") is not True
            or handler_calls[before:] != [expected]
        ):
            raise SystemExit(f"saved-skill alias missed the typed runtime: {command!r}")


def _assert_direct_handler_legacy_coercion(
    runtime: Any,
    original_handlers: dict[str, Callable[[dict[str, Any]], ToolResult]],
) -> None:
    list_limits: list[int] = []

    def list_store(limit: int) -> list[Any]:
        list_limits.append(limit)
        return []

    with patch.object(runtime.store, "list_skills", side_effect=list_store):
        string_limit = original_handlers["list_skills"]({"limit": "5"})
        boolean_limit = original_handlers["list_skills"]({"limit": False})
    if not string_limit.ok or not boolean_limit.ok or list_limits != [5, 25]:
        raise SystemExit("list_skills direct-handler legacy limit coercion changed")

    search_calls: list[tuple[str, int]] = []

    def search_store(query: str, limit: int = 20) -> list[Any]:
        search_calls.append((query, limit))
        return []

    with patch.object(runtime.store, "search_skills", side_effect=search_store):
        scalar_query = original_handlers["search_skills"]({"query": 7, "limit": "4"})
        boolean_limit = original_handlers["search_skills"](
            {"query": "launch", "limit": True}
        )
    if (
        not scalar_query.ok
        or not boolean_limit.ok
        or search_calls != [("7", 4), ("launch", 20)]
    ):
        raise SystemExit("search_skills direct-handler legacy coercion changed")

    get_names: list[str] = []

    def get_store(name: str) -> None:
        get_names.append(name)
        return None

    with patch.object(runtime.store, "get_skill", side_effect=get_store):
        scalar_name = original_handlers["get_skill"]({"name": 7})
    if not scalar_name.ok or get_names != ["7"]:
        raise SystemExit("get_skill direct-handler legacy name coercion changed")


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-skill-read-contract-") as temp:
        runtime = make_temp_runtime(Path(temp))
        planner = runtime.planner
        _assert_contract_inventory_is_exact_and_immutable(runtime)
        handler_calls, original_handlers = _install_handler_reach_spies(runtime)
        _assert_valid_executor_inputs(runtime, handler_calls)
        _assert_malformed_routed_args_stop_before_private_reads(runtime, handler_calls)
        _assert_command_first_aliases(runtime, handler_calls, planner)
        _assert_direct_handler_legacy_coercion(runtime, original_handlers)
    print("Saved-skill read argument contract smoke passed")


if __name__ == "__main__":
    main()
