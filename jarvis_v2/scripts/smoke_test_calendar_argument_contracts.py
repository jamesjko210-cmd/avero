from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any
from unittest.mock import patch

from jarvis_v2.agent.executor import Executor
from jarvis_v2.agent.types import PlannedAction, RiskLevel, ToolResult
from jarvis_v2.config import load_config
from jarvis_v2.tools import calendar_connector as calendar_module
from jarvis_v2.tools.permissions import PermissionPolicy
from jarvis_v2.tools.registry import (
    TOOL_ARGUMENT_CONTRACT_VERSION,
    Tool,
    ToolRegistry,
)


CALENDAR_TOOLS = (
    "list_calendars",
    "list_events",
    "check_availability",
    "create_event",
    "update_event",
    "delete_event",
)
CALENDAR_TARGET_TOOLS = CALENDAR_TOOLS[1:]
CALENDAR_MUTATION_TOOLS = (
    "create_event",
    "update_event",
    "delete_event",
)
PRIVATE_SENTINEL = "calendar-contract-private-sentinel"

EXPECTED_SCHEMAS = {
    "list_calendars": (
        RiskLevel.LOCAL_SAFE,
        (),
    ),
    "list_events": (
        RiskLevel.LOCAL_SAFE,
        (
            ("range", ("string",), False, None, None),
            ("calendar_id", ("string",), False, None, None),
            ("max_results", ("integer",), False, 1, 50),
        ),
    ),
    "check_availability": (
        RiskLevel.LOCAL_SAFE,
        (
            ("text", ("string",), False, None, None),
            ("request", ("string",), False, None, None),
            ("start", ("string",), False, None, None),
            ("end", ("string",), False, None, None),
            ("label", ("string",), False, None, None),
            ("calendar_id", ("string",), False, None, None),
        ),
    ),
    "create_event": (
        RiskLevel.HIGH_RISK,
        (
            ("start", ("string",), True, None, None),
            ("title", ("string",), False, None, None),
            ("summary", ("string",), False, None, None),
            ("end", ("string",), False, None, None),
            ("calendar_id", ("string",), False, None, None),
            ("description", ("string",), False, None, None),
            ("location", ("string",), False, None, None),
        ),
    ),
    "update_event": (
        RiskLevel.HIGH_RISK,
        (
            ("event_id", ("string",), True, None, None),
            ("title", ("string",), False, None, None),
            ("summary", ("string",), False, None, None),
            ("start", ("string",), False, None, None),
            ("end", ("string",), False, None, None),
            ("calendar_id", ("string",), False, None, None),
            ("description", ("string",), False, None, None),
            ("location", ("string",), False, None, None),
        ),
    ),
    "delete_event": (
        RiskLevel.HIGH_RISK,
        (
            ("event_id", ("string",), True, None, None),
            ("calendar_id", ("string",), False, None, None),
        ),
    ),
}

EXPECTED_BOUND_SCHEMAS = {
    "create_event": (
        ("start", ("string",), True, None, None),
        ("calendar_id", ("string",), True, None, None),
        ("title", ("string",), False, None, None),
        ("summary", ("string",), False, None, None),
        ("end", ("string",), False, None, None),
        ("description", ("string",), False, None, None),
        ("location", ("string",), False, None, None),
    ),
    "update_event": (
        ("event_id", ("string",), True, None, None),
        ("calendar_id", ("string",), True, None, None),
        ("title", ("string",), False, None, None),
        ("summary", ("string",), False, None, None),
        ("start", ("string",), False, None, None),
        ("end", ("string",), False, None, None),
        ("description", ("string",), False, None, None),
        ("location", ("string",), False, None, None),
    ),
    "delete_event": (
        ("event_id", ("string",), True, None, None),
        ("calendar_id", ("string",), True, None, None),
    ),
}

BASE_ARGS: dict[str, dict[str, Any]] = {
    "list_calendars": {},
    "list_events": {},
    "check_availability": {"text": "tomorrow afternoon"},
    "create_event": {
        "title": "Calendar custody smoke",
        "start": "2030-01-02T10:00:00+00:00",
    },
    "update_event": {"event_id": "event-custody-1", "title": "Updated title"},
    "delete_event": {"event_id": "event-custody-1"},
}


class CountingPermissionPolicy(PermissionPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, bool]] = []

    def check(self, tool: Tool, approved: bool = False):
        self.calls.append((tool.name, approved))
        return super().check(tool, approved=approved)


class FakeExecute:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    def execute(self) -> dict[str, Any]:
        return self.payload


class FakeEvents:
    def __init__(self, probe: dict[str, Any]):
        self.probe = probe

    def list(self, **kwargs: Any) -> FakeExecute:
        self.probe["event_lists"].append(dict(kwargs))
        return FakeExecute({"items": []})


class FakeCalendarList:
    def __init__(self, probe: dict[str, Any]):
        self.probe = probe

    def list(self) -> FakeExecute:
        self.probe["calendar_lists"] += 1
        return FakeExecute(
            {
                "items": [
                    {
                        "id": "owner-calendar@example.com",
                        "primary": True,
                    }
                ]
            }
        )


class FakeCalendarService:
    def __init__(self, probe: dict[str, Any]):
        self._calendar_list = FakeCalendarList(probe)
        self._events = FakeEvents(probe)

    def calendarList(self) -> FakeCalendarList:
        return self._calendar_list

    def events(self) -> FakeEvents:
        return self._events


def _new_probe() -> dict[str, Any]:
    return {
        "handlers": {name: 0 for name in CALENDAR_TOOLS},
        "service": 0,
        "calendar_lists": 0,
        "event_lists": [],
    }


def _service_factory(probe: dict[str, Any]) -> FakeCalendarService:
    probe["service"] += 1
    return FakeCalendarService(probe)


def _make_executor(
    probe: dict[str, Any],
) -> tuple[Executor, ToolRegistry, CountingPermissionPolicy]:
    registry = ToolRegistry()
    for tool in calendar_module.make_calendar_tools(load_config()):
        original_handler = tool.handler

        def handler(
            args: dict[str, Any],
            *,
            tool_name: str = tool.name,
            delegate: Any = original_handler,
        ) -> ToolResult:
            probe["handlers"][tool_name] += 1
            return delegate(args)

        registry.register(replace(tool, handler=handler))
    policy = CountingPermissionPolicy()
    return Executor(registry, policy), registry, policy


def _snapshot(
    probe: dict[str, Any], policy: CountingPermissionPolicy
) -> tuple[Any, ...]:
    return (
        tuple((name, probe["handlers"][name]) for name in CALENDAR_TOOLS),
        probe["service"],
        probe["calendar_lists"],
        tuple(policy.calls),
        tuple(tuple(sorted(row.items())) for row in probe["event_lists"]),
    )


def _contract_shape(contract: Any) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            field.name,
            tuple(sorted(argument_type.value for argument_type in field.types)),
            field.required,
            field.minimum,
            field.maximum,
        )
        for field in contract.fields
    )


def _assert_contract_shapes(registry: ToolRegistry) -> None:
    for name, (expected_risk, expected_fields) in EXPECTED_SCHEMAS.items():
        tool = registry.get(name)
        contract = tool.argument_contract
        if tool.risk is not expected_risk or tool.toolset != "personal":
            raise SystemExit(
                f"{name} risk/toolset drifted: {tool.risk!r} / {tool.toolset!r}"
            )
        if (
            contract is None
            or contract.version != TOOL_ARGUMENT_CONTRACT_VERSION
            or contract.allow_unknown is not False
        ):
            raise SystemExit(f"{name} does not have the expected strict contract")
        actual_fields = _contract_shape(contract)
        if actual_fields != expected_fields:
            raise SystemExit(f"{name} argument schema drifted: {actual_fields}")
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

        if name not in CALENDAR_MUTATION_TOOLS:
            if (
                tool.approval_argument_resolver is not None
                or tool.approval_argument_contract is not None
            ):
                raise SystemExit(f"read-only {name} unexpectedly owns approval binding")
            continue

        bound = tool.approval_argument_contract
        if tool.approval_argument_resolver is None or bound is None:
            raise SystemExit(f"{name} missed its calendar target approval binding")
        if (
            bound.version != TOOL_ARGUMENT_CONTRACT_VERSION
            or bound.allow_unknown is not False
        ):
            raise SystemExit(f"{name} bound contract is not strict")
        actual_bound = _contract_shape(bound)
        if actual_bound != EXPECTED_BOUND_SCHEMAS[name]:
            raise SystemExit(f"{name} bound argument schema drifted: {actual_bound}")


def _assert_contract_rejection(
    result: ToolResult,
    *,
    name: str,
    args: Any,
    expected_status: str,
) -> None:
    if (
        result.ok
        or result.metadata.get("failure_kind") != "tool_arguments_invalid"
        or result.metadata.get("argument_contract_version")
        != TOOL_ARGUMENT_CONTRACT_VERSION
        or result.metadata.get("argument_validation_status") != expected_status
        or result.metadata.get("requires_confirmation") is not False
        or result.metadata.get("handler_invoked") is not False
        or result.metadata.get("executed_handler") is not False
    ):
        raise SystemExit(f"{name} malformed arguments were not stopped: {args!r} / {result}")
    exposed = f"{result.output} {result.metadata}"
    if PRIVATE_SENTINEL in exposed:
        raise SystemExit("calendar argument rejection leaked unknown argument content")


def _assert_malformed_roots_and_unknown_keys(
    executor: Executor,
    probe: dict[str, Any],
    policy: CountingPermissionPolicy,
) -> None:
    for name in CALENDAR_TOOLS:
        for malformed in (None, ["not", "an", "object"]):
            before = _snapshot(probe, policy)
            result = executor.execute(
                PlannedAction(name, malformed, "Calendar malformed-root smoke.")  # type: ignore[arg-type]
            )
            _assert_contract_rejection(
                result,
                name=name,
                args=malformed,
                expected_status="arguments_not_object",
            )
            if _snapshot(probe, policy) != before:
                raise SystemExit(f"{name} malformed root crossed a custody boundary")

        args = dict(BASE_ARGS[name])
        args["private_payload"] = PRIVATE_SENTINEL
        before = _snapshot(probe, policy)
        result = executor.execute(
            PlannedAction(name, args, "Calendar unknown-key smoke.")
        )
        _assert_contract_rejection(
            result,
            name=name,
            args=args,
            expected_status="unknown_arguments",
        )
        if result.metadata.get("unknown_arg_keys") != ["<unknown>"]:
            raise SystemExit(f"{name} did not redact its unknown argument key")
        if _snapshot(probe, policy) != before:
            raise SystemExit(f"{name} unknown key crossed a custody boundary")


def _assert_calendar_id_preflight(
    executor: Executor,
    probe: dict[str, Any],
    policy: CountingPermissionPolicy,
) -> None:
    for name in CALENDAR_TARGET_TOOLS:
        for value in (None, 7, True, ["primary"]):
            args = dict(BASE_ARGS[name], calendar_id=value)
            before = _snapshot(probe, policy)
            result = executor.execute(
                PlannedAction(name, args, "Calendar target type smoke.")
            )
            _assert_contract_rejection(
                result,
                name=name,
                args=args,
                expected_status="type_mismatch",
            )
            if result.metadata.get("type_mismatch_arg_keys") != ["calendar_id"]:
                raise SystemExit(f"{name} rejected the wrong calendar target field")
            if _snapshot(probe, policy) != before:
                raise SystemExit(f"{name} non-string target crossed a custody boundary")


def _assert_blank_calendar_id_preflight(
    executor: Executor,
    probe: dict[str, Any],
    policy: CountingPermissionPolicy,
) -> None:
    for name in CALENDAR_TARGET_TOOLS:
        for value in ("", "   \t"):
            args = dict(BASE_ARGS[name], calendar_id=value)
            before = _snapshot(probe, policy)
            result = executor.execute(
                PlannedAction(name, args, "Calendar blank-target smoke.")
            )
            if (
                result.ok
                or result.metadata.get("reason") != "invalid_calendar_id"
                or probe["service"] != before[1]
            ):
                raise SystemExit(
                    f"{name} blank calendar target was not refused before Google access: {result}"
                )
            after = _snapshot(probe, policy)
            if name in CALENDAR_MUTATION_TOOLS:
                if (
                    result.metadata.get("requires_confirmation") is not False
                    or result.metadata.get("handler_invoked") is not False
                    or result.metadata.get("executed_handler") is not False
                    or after != before
                ):
                    raise SystemExit(f"{name} blank target crossed its approval boundary")
            elif (
                result.metadata.get("handler_invoked") is not True
                or result.metadata.get("executed_handler") is not True
                or probe["handlers"][name] != dict(before[0])[name] + 1
                or len(policy.calls) != len(before[3]) + 1
            ):
                raise SystemExit(f"{name} blank target did not fail locally in its handler")


def _assert_max_results_contract(
    executor: Executor,
    probe: dict[str, Any],
    policy: CountingPermissionPolicy,
) -> None:
    for value in (None, "1", 1.0, True, 0, -1, 51):
        args = {"max_results": value}
        before = _snapshot(probe, policy)
        result = executor.execute(
            PlannedAction("list_events", args, "Calendar max-results rejection smoke.")
        )
        _assert_contract_rejection(
            result,
            name="list_events",
            args=args,
            expected_status="type_mismatch",
        )
        if result.metadata.get("type_mismatch_arg_keys") != ["max_results"]:
            raise SystemExit("list_events rejected the wrong bounded integer field")
        if _snapshot(probe, policy) != before:
            raise SystemExit(f"invalid max_results crossed a custody boundary: {value!r}")

    for value in (1, 50):
        before_handlers = probe["handlers"]["list_events"]
        before_service = probe["service"]
        before_policy = len(policy.calls)
        before_lists = len(probe["event_lists"])
        result = executor.execute(
            PlannedAction(
                "list_events",
                {"max_results": value},
                "Calendar max-results edge smoke.",
            )
        )
        new_lists = probe["event_lists"][before_lists:]
        if (
            not result.ok
            or result.metadata.get("handler_invoked") is not True
            or probe["handlers"]["list_events"] != before_handlers + 1
            or probe["service"] != before_service + 1
            or len(policy.calls) != before_policy + 1
            or len(new_lists) != 1
            or new_lists[0].get("calendarId") != "primary"
            or new_lists[0].get("maxResults") != value
            or new_lists[0].get("singleEvents") is not True
            or new_lists[0].get("orderBy") != "startTime"
            or not new_lists[0].get("timeMin")
        ):
            raise SystemExit(f"list_events rejected valid max_results={value}: {result}")


def _assert_mutation_target_binding(
    executor: Executor,
    probe: dict[str, Any],
    policy: CountingPermissionPolicy,
) -> None:
    for name in CALENDAR_MUTATION_TOOLS:
        for explicit_target, expected_target, expected_source, expected_service_delta in (
            (None, "owner-calendar@example.com", "resolved_primary", 1),
            ("primary", "owner-calendar@example.com", "resolved_primary", 1),
            (
                "team-calendar@example.com",
                "team-calendar@example.com",
                "explicit",
                0,
            ),
        ):
            args = dict(BASE_ARGS[name])
            if explicit_target is not None:
                args["calendar_id"] = explicit_target
            before_handlers = probe["handlers"][name]
            before_service = probe["service"]
            before_calendar_lists = probe["calendar_lists"]
            before_policy = len(policy.calls)
            result = executor.execute(
                PlannedAction(name, args, "Calendar target binding smoke.")
            )
            expected_args = dict(args, calendar_id=expected_target)
            if (
                result.metadata.get("failure_kind") != "approval_required"
                or result.metadata.get("requires_confirmation") is not True
                or result.metadata.get("planned_args") != expected_args
                or result.metadata.get("calendar_target_bound") is not True
                or result.metadata.get("calendar_target_source") != expected_source
                or result.metadata.get("handler_invoked") is True
                or probe["handlers"][name] != before_handlers
                or probe["service"] != before_service + expected_service_delta
                or probe["calendar_lists"]
                != before_calendar_lists + expected_service_delta
                or len(policy.calls) != before_policy + 1
                or policy.calls[-1] != (name, False)
            ):
                raise SystemExit(
                    f"{name} did not bind calendar target before approval: {result}"
                )


def _assert_approved_legacy_targets_fail_before_service(
    executor: Executor,
    probe: dict[str, Any],
) -> None:
    for name in CALENDAR_MUTATION_TOOLS:
        for target_state, value in (
            ("missing", None),
            ("non-string", 7),
            ("blank", "   "),
            ("alias", "primary"),
        ):
            args = dict(BASE_ARGS[name])
            if target_state != "missing":
                args["calendar_id"] = value
            before_service = probe["service"]
            result = executor.execute(
                PlannedAction(name, args, "Legacy approved calendar target smoke."),
                approved=True,
            )
            if result.ok or probe["service"] != before_service:
                raise SystemExit(
                    f"{name} approved {target_state} target reached Google Calendar: {result}"
                )
            if target_state in {"missing", "non-string"} and (
                result.metadata.get("failure_kind") != "tool_arguments_invalid"
                or result.metadata.get("handler_invoked") is not False
            ):
                raise SystemExit(
                    f"{name} approved {target_state} target escaped the bound contract: {result}"
                )
            if target_state in {"blank", "alias"} and (
                result.metadata.get("reason") != "invalid_calendar_id"
                or result.metadata.get("handler_invoked") is not True
            ):
                raise SystemExit(
                    f"{name} approved {target_state} target was not refused locally: {result}"
                )


def _assert_unresolved_primary_fails_closed(
    executor: Executor,
    probe: dict[str, Any],
    policy: CountingPermissionPolicy,
) -> None:
    with patch.object(
        calendar_module,
        "_concrete_primary_calendar_id",
        side_effect=RuntimeError("primary unavailable"),
    ):
        for name in CALENDAR_MUTATION_TOOLS:
            before = _snapshot(probe, policy)
            result = executor.execute(
                PlannedAction(
                    name,
                    dict(BASE_ARGS[name]),
                    "Unresolved primary calendar smoke.",
                )
            )
            if (
                result.ok
                or result.metadata.get("reason") != "primary_calendar_unresolved"
                or result.metadata.get("requires_confirmation") is not False
                or result.metadata.get("handler_invoked") is not False
                or result.metadata.get("executed_handler") is not False
                or _snapshot(probe, policy) != before
            ):
                raise SystemExit(
                    f"{name} unresolved primary target did not fail closed: {result}"
                )


def main() -> None:
    probe = _new_probe()
    executor, registry, policy = _make_executor(probe)
    _assert_contract_shapes(registry)
    with patch.object(
        calendar_module,
        "_get_service",
        side_effect=lambda: _service_factory(probe),
    ), patch.object(
        calendar_module,
        "_get_readonly_service",
        side_effect=lambda: _service_factory(probe),
    ):
        _assert_malformed_roots_and_unknown_keys(executor, probe, policy)
        _assert_calendar_id_preflight(executor, probe, policy)
        _assert_max_results_contract(executor, probe, policy)
        _assert_mutation_target_binding(executor, probe, policy)
        _assert_unresolved_primary_fails_closed(executor, probe, policy)
        _assert_approved_legacy_targets_fail_before_service(executor, probe)
        _assert_blank_calendar_id_preflight(executor, probe, policy)
    print("Calendar argument contract smoke passed")


if __name__ == "__main__":
    main()
