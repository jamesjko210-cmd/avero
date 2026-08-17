from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools import calendar_connector as calendar_module


def _fail_generic_dispatch(_low_command: str) -> bool:
    raise AssertionError("calendar delete routing reached generic risky dispatch")


def test_exact_event_ids_route_directly_before_generic_dispatch() -> None:
    planner = RuleBasedPlanner()
    cases = (
        ("delete event evt123", "evt123"),
        ("cancel event id AbC_123@calendar", "AbC_123@calendar"),
        ("remove calendar event recurring.2026-07-14:09", "recurring.2026-07-14:09"),
        ("delete event trailing.period.", "trailing.period."),
    )
    with patch(
        "jarvis_v2.agent.planner._looks_like_risky_natural_order",
        side_effect=_fail_generic_dispatch,
    ):
        for command, event_id in cases:
            plan = planner.plan(command)
            if [action.tool_name for action in plan.actions] != ["delete_event"]:
                raise SystemExit(f"exact calendar delete missed delete_event: {command!r} -> {plan}")
            if plan.actions[0].args != {"event_id": event_id}:
                raise SystemExit(f"calendar delete changed or supplemented the exact id: {command!r} -> {plan}")
            if plan.metadata:
                raise SystemExit(f"calendar delete added unexpected planner metadata: {command!r} -> {plan.metadata}")


def test_fuzzy_or_missing_event_ids_only_request_clarification() -> None:
    planner = RuleBasedPlanner()
    commands = (
        "delete event",
        "cancel my 3pm meeting",
        "remove the dentist appointment",
        "delete event /tmp/not-an-event-id",
    )
    with patch(
        "jarvis_v2.agent.planner._looks_like_risky_natural_order",
        side_effect=_fail_generic_dispatch,
    ):
        for command in commands:
            plan = planner.plan(command)
            if [action.tool_name for action in plan.actions] != ["respond"]:
                raise SystemExit(f"fuzzy calendar delete should only clarify: {command!r} -> {plan}")
            clarification = plan.actions[0].args.get("text", "").lower()
            if "exact event id" not in clarification:
                raise SystemExit(f"calendar delete clarification did not request an exact id: {command!r} -> {plan}")
            if plan.metadata:
                raise SystemExit(f"calendar clarification added unexpected planner metadata: {command!r} -> {plan.metadata}")


def test_exact_delete_stays_high_risk_and_does_not_run_before_approval() -> None:
    with TemporaryDirectory(prefix="jarvis-calendar-delete-routing-") as temp:
        runtime = make_temp_runtime(Path(temp))
        with patch.object(
            calendar_module,
            "_concrete_primary_calendar_id",
            return_value="owner-calendar@example.com",
        ), patch.object(
            calendar_module,
            "_get_service",
            side_effect=AssertionError("delete_event handler ran before approval"),
        ):
            result = runtime.handle("delete event evt123")

        if len(result.tool_results) != 1:
            raise SystemExit(f"calendar delete should produce one approval result: {result.tool_results}")
        tool_result = result.tool_results[0]
        metadata = tool_result.metadata
        if (
            tool_result.tool_name != "delete_event"
            or tool_result.ok
            or metadata.get("risk_level") != "HIGH_RISK"
            or metadata.get("failure_kind") != "approval_required"
            or metadata.get("executed_handler") is not False
        ):
            raise SystemExit(f"calendar delete bypassed the HIGH_RISK approval boundary: {tool_result}")
        if metadata.get("planned_args") != {
            "event_id": "evt123",
            "calendar_id": "owner-calendar@example.com",
        }:
            raise SystemExit(f"calendar delete approval binding metadata drifted: {metadata}")

        approvals = runtime.store.list_pending_approvals(limit=10)
        if len(approvals) != 1 or dict(approvals[0]).get("tool_name") != "delete_event":
            raise SystemExit(f"calendar delete did not queue one protected approval: {approvals}")


def main() -> None:
    test_exact_event_ids_route_directly_before_generic_dispatch()
    test_fuzzy_or_missing_event_ids_only_request_clarification()
    test_exact_delete_stays_high_risk_and_does_not_run_before_approval()
    print("Calendar delete routing smoke passed")


if __name__ == "__main__":
    main()
