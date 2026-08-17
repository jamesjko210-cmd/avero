"""Offline proof for canonical Google Calendar refusal/setup recovery guidance."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from jarvis_v2.config import load_config
from jarvis_v2.tools import calendar_connector as cc


PRIVATE_MARKERS = ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")


class ServiceProbe:
    def __init__(self) -> None:
        self.calls = 0

    def fail(self) -> None:
        self.calls += 1
        raise RuntimeError("private setup failure near /\x55sers/example/calendar-token")


def _assert_guided_failure(
    result: Any,
    *,
    label: str,
    reason: str,
    action: str,
    commands: list[str] | None = None,
) -> None:
    expected_commands = commands or []
    handoff = result.metadata.get("calendar_mutation_handoff")
    declared_reason = result.metadata.get("reason")
    if declared_reason is None and isinstance(handoff, dict):
        declared_reason = handoff.get("reason")
    if result.ok or declared_reason != reason:
        raise SystemExit(f"{label} did not fail with {reason}: {result}")
    if action not in result.output:
        raise SystemExit(f"{label} omitted its visible recovery action: {result.output}")
    expected_guidance = {
        "version": 1,
        "action": action,
        "commands": expected_commands,
    }
    if result.metadata.get("recovery_guidance") != expected_guidance:
        raise SystemExit(f"{label} canonical recovery declaration drifted: {result.metadata}")
    expected_flags = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
    }
    for key, expected in expected_flags.items():
        if result.metadata.get(key) is not expected:
            raise SystemExit(f"{label} field {key} drifted: {result.metadata}")
    if expected_commands:
        if result.metadata.get("next_command") != expected_commands[0]:
            raise SystemExit(f"{label} next command drifted: {result.metadata}")
        if result.metadata.get("recovery_commands") != expected_commands:
            raise SystemExit(f"{label} recovery commands drifted: {result.metadata}")
    if any(marker in result.output or marker in str(result.metadata) for marker in PRIVATE_MARKERS):
        raise SystemExit(f"{label} exposed private-looking detail: {result}")
    for key in (
        "calls_model",
        "executes_tools",
        "queues_approval",
        "executes_side_effect",
        "external_side_effect",
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "controls_computer",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ):
        if result.metadata.get(key):
            raise SystemExit(f"{label} unexpectedly set {key}: {result.metadata}")


def main() -> None:
    tools = {tool.name: tool for tool in cc.make_calendar_tools(load_config())}
    input_cases = (
        ("list empty calendar", "list_events", {"calendar_id": ""}, "invalid_calendar_id"),
        ("list invalid range", "list_events", {"range": "not-a-calendar-range"}, "invalid_range"),
        ("availability empty calendar", "check_availability", {"calendar_id": ""}, "invalid_calendar_id"),
        ("create missing title", "create_event", {"start": "2030-01-02T10:00:00+09:00"}, "missing_title"),
        ("create missing start", "create_event", {"title": "Test"}, "missing_start"),
        ("create empty calendar", "create_event", {"title": "Test", "start": "2030-01-02", "calendar_id": ""}, "invalid_calendar_id"),
        ("create local text", "create_event", {"title": "/\x55sers/example/private", "start": "2030-01-02", "calendar_id": "calendar@example.com"}, "invalid_event_text"),
        ("update missing event", "update_event", {"title": "Test"}, "missing_event_id"),
        ("update local event", "update_event", {"event_id": "/tmp/private-event", "title": "Test"}, "invalid_event_id"),
        ("update empty calendar", "update_event", {"event_id": "event-1", "title": "Test", "calendar_id": ""}, "invalid_calendar_id"),
        ("update local text", "update_event", {"event_id": "event-1", "title": "/private/title", "calendar_id": "calendar@example.com"}, "invalid_event_text"),
        ("update end only", "update_event", {"event_id": "event-1", "end": "2030-01-02T11:00:00+09:00", "calendar_id": "calendar@example.com"}, "missing_start_for_end_change"),
        ("update no changes", "update_event", {"event_id": "event-1", "calendar_id": "calendar@example.com"}, "missing_changes"),
        ("delete missing event", "delete_event", {}, "missing_event_id"),
        ("delete local event", "delete_event", {"event_id": "/var/folders/private-event"}, "invalid_event_id"),
        ("delete empty calendar", "delete_event", {"event_id": "event-1", "calendar_id": ""}, "invalid_calendar_id"),
    )

    probe = ServiceProbe()
    with patch.object(cc, "_get_service", side_effect=probe.fail):
        for label, tool_name, args, reason in input_cases:
            result = tools[tool_name].handler(args)
            _assert_guided_failure(
                result,
                label=label,
                reason=reason,
                action=cc.CALENDAR_INPUT_RECOVERY_ACTION,
            )
    if probe.calls:
        raise SystemExit(f"calendar input refusals contacted the service {probe.calls} time(s)")

    for tool_name in ("create_event", "update_event", "delete_event"):
        resolver = tools[tool_name].approval_argument_resolver
        if resolver is None:
            raise SystemExit(f"{tool_name} missed its approval argument resolver")
        _assert_guided_failure(
            resolver({"calendar_id": ""}),
            label=f"{tool_name} approval empty calendar",
            reason="invalid_calendar_id",
            action=cc.CALENDAR_INPUT_RECOVERY_ACTION,
        )
        with patch.object(cc, "_concrete_primary_calendar_id", side_effect=RuntimeError("private")):
            _assert_guided_failure(
                resolver({}),
                label=f"{tool_name} unresolved primary",
                reason="primary_calendar_unresolved",
                action=cc.CALENDAR_PRIMARY_TARGET_RECOVERY_ACTION,
                commands=["list calendars"],
            )

    setup_args = {
        "create_event": {
            "title": "Offline setup test",
            "start": "2030-01-02T10:00:00+09:00",
            "calendar_id": "calendar@example.com",
        },
        "update_event": {
            "event_id": "event-1",
            "title": "Offline setup test",
            "calendar_id": "calendar@example.com",
        },
        "delete_event": {
            "event_id": "event-1",
            "calendar_id": "calendar@example.com",
        },
    }
    setup_probe = ServiceProbe()
    with patch.object(cc, "_get_service", side_effect=setup_probe.fail):
        for tool_name, args in setup_args.items():
            result = tools[tool_name].handler(args)
            _assert_guided_failure(
                result,
                label=f"{tool_name} setup failure",
                reason="fetch_error",
                action=cc.CALENDAR_MUTATION_SETUP_RECOVERY_ACTION,
                commands=["setup check"],
            )
            if result.metadata.get("mutation_attempted") is not False:
                raise SystemExit(f"{tool_name} setup failure claimed a mutation attempt: {result.metadata}")
    if setup_probe.calls != len(setup_args):
        raise SystemExit(f"setup failure probe count drifted: {setup_probe.calls}")

    print("Jarvis Calendar refusal/setup error-guidance smoke test passed.")


if __name__ == "__main__":
    main()
