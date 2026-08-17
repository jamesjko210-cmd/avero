"""Smoke tests for the on-this-day history connector (mocked fetch, no network)."""

from __future__ import annotations

from typing import Any

from jarvis_v2.agent.executor import Executor
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import PlannedAction, RiskLevel
from jarvis_v2.config import load_config
from jarvis_v2.tools._http import HttpError
from jarvis_v2.tools import history_connector as hc
from jarvis_v2.tools.permissions import PermissionPolicy
from jarvis_v2.tools.registry import TOOL_ARGUMENT_CONTRACT_VERSION, ToolRegistry

NO_AUTHORITY_FLAGS = (
    "authorizes_execution",
    "authorizes_completion_claim",
    "approval_granted",
)


def _tools():
    return {t.name: t for t in hc.make_history_tools(load_config())}


def _assert_history_recovery_message(output: str, label: str) -> None:
    lowered = output.lower()
    for fragment in [
        "network access to wikipedia on-this-day",
        "setup check",
        "retry in a moment",
    ]:
        if fragment not in lowered:
            raise SystemExit(f"{label} missed actionable recovery guidance {fragment!r}: {output}")
    for forbidden in ["http error", "offline", "traceback", "/users/", "/private/", "/var/folders/", "/tmp/"]:
        if forbidden in lowered:
            raise SystemExit(f"{label} leaked raw backend text {forbidden!r}: {output}")


def _assert_history_handoff(
    metadata: dict[str, Any],
    label: str,
    *,
    status: str,
    reason: str = "",
    count: int = 0,
    retry_safe: bool = False,
    calls_external_service: bool = True,
) -> dict[str, Any]:
    handoff = metadata.get("history_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missing history_handoff: {metadata}")
    if handoff.get("source") != "on_this_day" or handoff.get("status") != status:
        raise SystemExit(f"{label} has wrong source/status: {handoff}")
    if metadata.get("history_handoff_ready") is not True or handoff.get("history_handoff_ready") is not True:
        raise SystemExit(f"{label} should expose flat and nested handoff readiness: {metadata} vs {handoff}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} should expose nested handoff_ready: {handoff}")
    if handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} should be operator-ready: {handoff}")
    if metadata.get("history_ready_for_operator") != handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} ready alias parity failed: {metadata} vs {handoff}")
    if reason and handoff.get("reason") != reason:
        raise SystemExit(f"{label} has wrong reason: {handoff}")
    if handoff.get("state_changed") is not False or handoff.get("changed") != []:
        raise SystemExit(f"{label} should not claim changed state: {handoff}")
    if metadata.get("history_state_changed") != handoff.get("state_changed") or metadata.get("history_changed") != handoff.get("changed"):
        raise SystemExit(f"{label} state alias parity failed: {metadata} vs {handoff}")
    for key in NO_AUTHORITY_FLAGS:
        if metadata.get(key) is not False or handoff.get(key) is not False:
            raise SystemExit(f"{label} should keep {key}=False in flat and nested metadata: {metadata}")
        alias = f"history_{key}"
        if metadata.get(alias) is not False or metadata.get(alias) != handoff.get(key):
            raise SystemExit(f"{label} no-authority alias parity failed for {alias}: {metadata} vs {handoff}")
    if handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} should keep output content out of handoff: {handoff}")
    if metadata.get("history_content_in_handoff") != handoff.get("content_in_handoff"):
        raise SystemExit(f"{label} content-in-handoff alias parity failed: {metadata} vs {handoff}")
    for key in ("month", "day", "date"):
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} should mirror {key}: {handoff} vs {metadata}")
    for key in ("source", "raw_date", "local_path_date"):
        handoff_key = "date_source" if key == "source" else key
        if handoff.get(handoff_key) != metadata.get(key):
            raise SystemExit(f"{label} should mirror {key}: {handoff} vs {metadata}")
    if handoff.get("event_count") != count:
        raise SystemExit(f"{label} has wrong event count: {handoff}")
    if "count" in metadata and handoff.get("event_count") != metadata.get("count"):
        raise SystemExit(f"{label} should mirror count metadata: {handoff} vs {metadata}")
    if handoff.get("exception_type", "") != metadata.get("exception_type", ""):
        raise SystemExit(f"{label} should mirror exception type: {handoff} vs {metadata}")
    if handoff.get("content_in_metadata") is not False:
        raise SystemExit(f"{label} should mark content exclusion: {handoff}")
    if metadata.get("history_content_in_metadata") != handoff.get("content_in_metadata"):
        raise SystemExit(f"{label} content-in-metadata alias parity failed: {metadata} vs {handoff}")
    if handoff.get("retry_safe") is not retry_safe:
        raise SystemExit(f"{label} has wrong retry safety: {handoff}")
    if handoff.get("next_safe_command") != "on this day":
        raise SystemExit(f"{label} has wrong next safe command: {handoff}")
    expected_commands = ["on this day"]
    if handoff.get("next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} has wrong next safe command list: {handoff}")
    if handoff.get("next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} has wrong next safe command count: {handoff}")
    if metadata.get("history_next_safe_command") != handoff.get("next_safe_command"):
        raise SystemExit(f"{label} next-command alias parity failed: {metadata} vs {handoff}")
    if metadata.get("history_next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} next-commands alias parity failed: {metadata} vs {handoff}")
    if metadata.get("history_next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} next-command count alias parity failed: {metadata} vs {handoff}")

    boundaries = handoff.get("boundaries")
    expected = {
        "calls_model": False,
        "calls_external_service": calls_external_service,
        "calls_external_services": calls_external_service,
        "executes_tools": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "reads_personal_data": False,
        "reads_private_data": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": False,
    }
    if boundaries != expected:
        raise SystemExit(f"{label} has wrong boundaries: {boundaries}")
    if metadata.get("history_boundaries") != boundaries:
        raise SystemExit(f"{label} boundary alias parity failed: {metadata} vs {handoff}")
    if metadata.get("calls_external_service") is not calls_external_service or metadata.get("calls_external_services") is not calls_external_service:
        raise SystemExit(f"{label} flat external-call flags should match handoff: {metadata}")
    combined = f"{metadata}\n{handoff}".lower()
    for fragment in ("/users/", "/private/", "/var/folders/", "/tmp/"):
        if fragment in combined:
            raise SystemExit(f"{label} leaked local path fragment {fragment!r}: {metadata} {handoff}")
    return handoff


def test_is_local_safe() -> None:
    if _tools()["on_this_day"].risk != RiskLevel.LOCAL_SAFE:
        raise SystemExit("on_this_day should be LOCAL_SAFE")


def test_history_argument_contract_fences_malformed_input() -> None:
    fetch_calls: list[tuple[int, int]] = []
    original_fetch = hc._fetch

    def mocked_fetch(month: int, day: int) -> dict:
        fetch_calls.append((month, day))
        return {"events": [{"year": 1776, "text": "Declaration adopted."}]}

    try:
        hc._fetch = mocked_fetch  # type: ignore[assignment]
        private_sentinel = "private-history-contract-sentinel"
        tool = _tools()["on_this_day"]
        contract = tool.argument_contract
        actual_shape = (
            tuple(
                (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
                for field in contract.fields
            )
            if contract is not None
            else ()
        )
        if (
            contract is None
            or contract.version != TOOL_ARGUMENT_CONTRACT_VERSION
            or contract.allow_unknown is not False
            or actual_shape
            != (("text", ("string",), False), ("date", ("string",), False))
        ):
            raise SystemExit(f"on_this_day should have an exact strict text/date contract: {contract}")

        registry = ToolRegistry()
        registry.register(tool)
        executor = Executor(registry, PermissionPolicy())
        malformed_args: list[object] = [
            None,
            ["not", "an", "object"],
            {"text": {"private": private_sentinel}},
            {"text": True},
            {"date": 704},
            {"date": [private_sentinel]},
            {"date": "July 4", "unexpected": private_sentinel},
        ]
        rejected_surfaces: list[object] = []
        for args in malformed_args:
            result = executor.execute(
                PlannedAction("on_this_day", args, "history contract smoke")  # type: ignore[arg-type]
            )
            if (
                result.ok
                or result.metadata.get("failure_kind") != "tool_arguments_invalid"
                or result.metadata.get("handler_invoked") is not False
                or result.metadata.get("executed_handler") is not False
                or result.metadata.get("requires_confirmation") is not False
            ):
                raise SystemExit(f"malformed history input passed the executor: {result}")
            rejected_surfaces.append({"output": result.output, "metadata": result.metadata})
        if fetch_calls:
            raise SystemExit(f"malformed history input reached Wikipedia on-this-day: {fetch_calls}")
        if private_sentinel in str(rejected_surfaces):
            raise SystemExit("history argument rejection leaked a supplied private value")

        valid_cases = (
            ({"date": "July 4"}, (7, 4)),
            ({"text": "what happened on December 25"}, (12, 25)),
        )
        for args, expected_date in valid_cases:
            result = executor.execute(
                PlannedAction("on_this_day", args, "valid history contract smoke")
            )
            if not result.ok or fetch_calls[-1] != expected_date:
                raise SystemExit(f"valid history input changed: {args} {result} {fetch_calls}")
    finally:
        hc._fetch = original_fetch  # type: ignore[assignment]


def test_lists_events_sorted_recent_first() -> None:
    hc._fetch = lambda month, day: {  # type: ignore
        "events": [
            {"year": 1900, "text": "Old event."},
            {"year": 2020, "text": "Recent event."},
        ]
    }
    out = _tools()["on_this_day"].handler({})
    if not out.ok or "Recent event." not in out.output or "Old event." not in out.output:
        raise SystemExit(f"history output wrong: {out.output}")
    if not out.metadata.get("month") or not out.metadata.get("day") or not out.metadata.get("date"):
        raise SystemExit(f"history success should preserve attempted date metadata: {out.metadata}")
    if out.metadata.get("source") != "today" or out.metadata.get("raw_date") != "":
        raise SystemExit(f"default history should mark today's date source: {out.metadata}")
    # most recent should appear before older
    if out.output.index("2020") > out.output.index("1900"):
        raise SystemExit("events not sorted recent-first")
    handoff = _assert_history_handoff(out.metadata, "history success", status="ok", count=2)
    rows = handoff.get("preview_rows") or []
    if [row.get("year") for row in rows] != [2020, 1900]:
        raise SystemExit(f"history handoff should preserve sorted event rows: {handoff}")
    if rows[0].get("text_preview") != "Recent event.":
        raise SystemExit(f"history handoff should preserve bounded text preview: {handoff}")


def test_sentinel_phrases_default_to_today_instead_of_refusing() -> None:
    """Real bug found live 2026-07-08: "what happened on this day in history" and
    "what happened today in history" residually extract to a recognized-but-then
    -mishandled sentinel ("this day" / "what happened today") after the date-text
    extractor strips trailing "in history" -- these were being treated as
    unparseable garbage (refused) instead of the same "no specific date, use
    today" case that bare "on this day" already handled correctly."""
    hc._fetch = lambda month, day: {"events": [{"year": 2020, "text": "Recent event."}]}  # type: ignore
    for text in (
        "what happened on this day in history",
        "what happened today in history",
    ):
        out = _tools()["on_this_day"].handler({"text": text})
        if not out.ok or "month and day" in out.output:
            raise SystemExit(f"sentinel phrase should default to today, not refuse: {text!r} -> {out.output}")
        if out.metadata.get("source") != "today" or out.metadata.get("reason") == "invalid_date":
            raise SystemExit(f"sentinel phrase should mark today's date source, not invalid_date: {text!r} -> {out.metadata}")


def test_handles_empty_events() -> None:
    hc._fetch = lambda month, day: {"events": []}  # type: ignore
    out = _tools()["on_this_day"].handler({})
    if not out.ok or "No events found" not in out.output:
        raise SystemExit(f"empty history should be a clean success: {out.output}")
    _assert_history_handoff(
        out.metadata,
        "empty history",
        status="empty",
        reason="no_events",
        retry_safe=True,
    )


def test_handles_error() -> None:
    def boom(month, day):
        raise RuntimeError("offline")

    hc._fetch = boom  # type: ignore
    out = _tools()["on_this_day"].handler({})
    if out.ok or not out.output.strip():
        raise SystemExit(f"error not handled: {out.output}")
    if not out.metadata.get("month") or not out.metadata.get("day") or not out.metadata.get("date"):
        raise SystemExit(f"history error should preserve attempted date metadata: {out.metadata}")
    if "offline" in out.output or out.metadata.get("exception_type") != "RuntimeError":
        raise SystemExit(f"history error should be friendly with bounded exception metadata: {out.output} {out.metadata}")
    _assert_history_recovery_message(out.output, "history generic fetch error")
    _assert_history_handoff(
        out.metadata,
        "history error",
        status="unavailable",
        reason="fetch_error",
        retry_safe=True,
    )

    def server_error(month, day):
        raise HttpError(500, "HTTP Error 500: Internal Server Error")

    hc._fetch = server_error  # type: ignore
    server_out = _tools()["on_this_day"].handler({})
    if server_out.ok or server_out.metadata.get("exception_type") != "HttpError":
        raise SystemExit(f"HTTP 500 should preserve bounded exception metadata: {server_out.output} {server_out.metadata}")
    _assert_history_recovery_message(server_out.output, "history HTTP 500")
    _assert_history_handoff(
        server_out.metadata,
        "history HTTP 500 fetch error",
        status="unavailable",
        reason="fetch_error",
        retry_safe=True,
    )

    def timeout_error(month, day):
        raise TimeoutError("request timed out near /\x55sers/example/private")

    hc._fetch = timeout_error  # type: ignore
    timeout_out = _tools()["on_this_day"].handler({})
    if timeout_out.ok or timeout_out.metadata.get("exception_type") != "TimeoutError":
        raise SystemExit(f"timeout should preserve bounded exception metadata: {timeout_out.output} {timeout_out.metadata}")
    if "timed out" not in timeout_out.output.lower():
        raise SystemExit(f"timeout recovery should preserve timeout cause: {timeout_out.output}")
    _assert_history_recovery_message(timeout_out.output, "history timeout")
    _assert_history_handoff(
        timeout_out.metadata,
        "history timeout fetch error",
        status="unavailable",
        reason="fetch_error",
        retry_safe=True,
    )


def test_explicit_date_from_text() -> None:
    seen = []

    def fake_fetch(month, day):
        seen.append((month, day))
        return {"events": [{"year": 1776, "text": "Declaration adopted."}]}

    hc._fetch = fake_fetch  # type: ignore
    cases = [
        ("what happened on July 4", (7, 4), "07-04"),
        ("July 4 in history", (7, 4), "07-04"),
        ("July 4 history", (7, 4), "07-04"),
        ("history July 4", (7, 4), "07-04"),
        ("history on July 4", (7, 4), "07-04"),
        ("what happened on December 25 in history", (12, 25), "12-25"),
        ("what happened in history on 7/4", (7, 4), "07-04"),
        ("history 7/4", (7, 4), "07-04"),
        ("7/4 history", (7, 4), "07-04"),
    ]
    for text, expected_fetch, expected_date in cases:
        seen.clear()
        out = _tools()["on_this_day"].handler({"text": text})
        if not out.ok or "Declaration adopted" not in out.output:
            raise SystemExit(f"explicit history date should succeed: {text!r} -> {out.output} {out.metadata}")
        if seen != [expected_fetch]:
            raise SystemExit(f"explicit history date fetched wrong date for {text!r}: {seen}")
        if out.metadata.get("date") != expected_date or out.metadata.get("source") != "text":
            raise SystemExit(f"explicit history date should preserve date metadata: {out.metadata}")
        _assert_history_handoff(out.metadata, f"explicit date {text!r}", status="ok", count=1)


def test_invalid_explicit_date_never_fetches() -> None:
    calls = []

    def fail_fetch(month, day):
        calls.append((month, day))
        raise AssertionError("invalid history date should not fetch")

    hc._fetch = fail_fetch  # type: ignore
    for text, local_path in [
        ("what happened on February 30", False),
        ("what happened on 13/40", False),
        ("what happened on /\x55sers/example/private/day in history", True),
    ]:
        out = _tools()["on_this_day"].handler({"text": text})
        if out.ok or "month and day" not in out.output:
            raise SystemExit(f"invalid explicit history date should be rejected locally: {text!r} -> {out.output}")
        if out.metadata.get("reason") != "invalid_date":
            raise SystemExit(f"invalid history date should include reason metadata: {out.metadata}")
        if bool(out.metadata.get("local_path_date")) is not local_path:
            raise SystemExit(f"invalid history date should preserve local path marker: {out.metadata}")
        _assert_history_handoff(
            out.metadata,
            f"invalid explicit date {text!r}",
            status="refused",
            reason="invalid_date",
            calls_external_service=False,
        )
    if calls:
        raise SystemExit(f"invalid history date unexpectedly fetched: {calls}")


def test_planner_routes_history() -> None:
    p = RuleBasedPlanner()
    for q in [
        "on this day",
        "on this day July 4",
        "this day in history",
        "today in history",
        "history today",
        "what happened today in history",
        "what happened on July 4",
        "July 4 in history",
        "July 4 history",
        "history July 4",
        "history on July 4",
        "what happened on December 25 in history",
        "what happened in history on 7/4",
        "history 7/4",
        "7/4 history",
        "historical events today",
        "tell me a historical fact today",
    ]:
        if [a.tool_name for a in p.plan(q).actions] != ["on_this_day"]:
            raise SystemExit(f"history route missed: {q!r}")
    event_actions = p.plan("events today").actions
    if [a.tool_name for a in event_actions] != ["list_events"]:
        raise SystemExit(f"history route should preserve plain calendar events: {event_actions}")


def main() -> None:
    test_is_local_safe()
    test_history_argument_contract_fences_malformed_input()
    test_lists_events_sorted_recent_first()
    test_sentinel_phrases_default_to_today_instead_of_refusing()
    test_handles_empty_events()
    test_handles_error()
    test_explicit_date_from_text()
    test_invalid_explicit_date_never_fetches()
    test_planner_routes_history()
    print("History connector smoke passed")


if __name__ == "__main__":
    main()
