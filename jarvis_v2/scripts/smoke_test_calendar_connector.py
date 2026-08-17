"""Smoke tests for the Google Calendar connector (mocked service, no network)."""

from __future__ import annotations

import os
import sys
import tempfile
import types
from datetime import datetime, timedelta
from pathlib import Path

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.config import load_config
from jarvis_v2.tools import calendar_connector as cc

_ORIGINAL_GET_SERVICE = cc._get_service
_ORIGINAL_GET_READONLY_SERVICE = cc._get_readonly_service
LOCAL_PATH_FRAGMENTS = ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")
NO_AUTHORITY_FLAGS = {
    "authorizes_execution": False,
    "authorizes_completion_claim": False,
    "approval_granted": False,
}


def _leaks_local_path(text: object) -> bool:
    return any(fragment in str(text) for fragment in LOCAL_PATH_FRAGMENTS)


def assert_calendar_recovery_message(output: str, label: str) -> None:
    required = [
        "Google Calendar is having trouble",
        "network access to Google Calendar",
        "setup check",
        "./launch_jarvis_v3_calendar_auth.py readonly",
        "normal macOS Terminal shell prompt",
        "not in Telegram",
        "retry",
    ]
    missing = [piece for piece in required if piece not in output]
    if missing:
        raise SystemExit(f"{label} failure missed recovery guidance {missing}: {output}")
    if "./launch_jarvis_v3_calendar_auth.py full-access" in output:
        raise SystemExit(f"{label} read failure suggested full-access authorization: {output}")


def assert_personal_read_recovery(result, label: str) -> None:
    action = cc.PERSONAL_READ_RECOVERY_ACTION
    if result.ok or action not in result.output:
        raise SystemExit(f"{label} hid the canonical read recovery action: {result}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": action,
        "commands": ["setup check"],
    }:
        raise SystemExit(f"{label} read recovery declaration drifted: {result.metadata}")
    expected = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "next_command": "setup check",
        "recovery_commands": ["setup check"],
    }
    for key, value in expected.items():
        if result.metadata.get(key) != value:
            raise SystemExit(f"{label} read recovery field {key} drifted: {result.metadata}")


def assert_calendar_events_handoff(
    metadata: dict,
    label: str,
    *,
    status: str,
    calls_external_service: bool = True,
) -> dict:
    handoff = metadata.get("calendar_events_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed calendar events handoff: {metadata}")
    if handoff.get("source") != "list_events":
        raise SystemExit(f"{label} handoff source wrong: {handoff}")
    for key, expected_value in NO_AUTHORITY_FLAGS.items():
        if metadata.get(key) is not expected_value:
            raise SystemExit(f"{label} flat {key} should be {expected_value}: {metadata}")
        if handoff.get(key) is not expected_value:
            raise SystemExit(f"{label} handoff {key} should be {expected_value}: {handoff}")
    if handoff.get("status") != status:
        raise SystemExit(f"{label} handoff status wrong: {handoff}")
    if handoff.get("event_count") != metadata.get("count"):
        raise SystemExit(f"{label} handoff count parity failed: {metadata}")
    if handoff.get("max_results") != metadata.get("max_results"):
        raise SystemExit(f"{label} handoff max_results parity failed: {metadata}")
    if handoff.get("calendar_id") != metadata.get("calendar_id"):
        raise SystemExit(f"{label} handoff calendar_id parity failed: {metadata}")
    if handoff.get("label") != metadata.get("label") or handoff.get("range") != metadata.get("range"):
        raise SystemExit(f"{label} handoff range/label parity failed: {metadata}")
    if handoff.get("content_in_metadata") is not False:
        raise SystemExit(f"{label} handoff should declare bounded metadata content: {handoff}")
    if _leaks_local_path(handoff):
        raise SystemExit(f"{label} handoff leaked local path: {handoff}")
    events = handoff.get("events")
    if not isinstance(events, list) or len(events) != handoff.get("event_count"):
        raise SystemExit(f"{label} handoff event rows wrong: {handoff}")
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict):
        raise SystemExit(f"{label} handoff missed boundaries: {handoff}")
    if boundaries.get("calls_external_service") is not calls_external_service:
        raise SystemExit(f"{label} handoff external-service boundary was wrong: {handoff}")
    if boundaries.get("reads_personal_data") is not calls_external_service:
        raise SystemExit(f"{label} handoff personal-data boundary was wrong: {handoff}")
    for key in [
        "calls_model",
        "reads_private_data",
        "executes_side_effect",
        "external_side_effect",
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "queues_approval",
        "controls_computer",
        *NO_AUTHORITY_FLAGS.keys(),
    ]:
        if boundaries.get(key) or metadata.get(key):
            raise SystemExit(f"{label} should not perform {key}: {metadata}")
    return handoff


def assert_calendar_list_handoff(metadata: dict, label: str, *, status: str) -> dict:
    handoff = metadata.get("calendar_list_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed calendar list handoff: {metadata}")
    if handoff.get("source") != "list_calendars":
        raise SystemExit(f"{label} list handoff source wrong: {handoff}")
    for key, expected_value in NO_AUTHORITY_FLAGS.items():
        if metadata.get(key) is not expected_value:
            raise SystemExit(f"{label} flat {key} should be {expected_value}: {metadata}")
        if handoff.get(key) is not expected_value:
            raise SystemExit(f"{label} list handoff {key} should be {expected_value}: {handoff}")
    if handoff.get("status") != status:
        raise SystemExit(f"{label} list handoff status wrong: {handoff}")
    if handoff.get("calendar_count") != metadata.get("count"):
        raise SystemExit(f"{label} list handoff count parity failed: {metadata}")
    calendars = handoff.get("calendars")
    if not isinstance(calendars, list) or len(calendars) != handoff.get("calendar_count"):
        raise SystemExit(f"{label} list handoff calendar rows wrong: {handoff}")
    if handoff.get("calendar_ids") != [row.get("calendar_id") for row in calendars if row.get("calendar_id")]:
        raise SystemExit(f"{label} list handoff ids diverged: {handoff}")
    if handoff.get("content_in_metadata") is not False:
        raise SystemExit(f"{label} list handoff should declare bounded metadata content: {handoff}")
    if _leaks_local_path(handoff):
        raise SystemExit(f"{label} list handoff leaked local path: {handoff}")
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict):
        raise SystemExit(f"{label} list handoff missed boundaries: {handoff}")
    if not boundaries.get("calls_external_service") or not boundaries.get("reads_personal_data"):
        raise SystemExit(f"{label} list handoff must declare external personal calendar read: {handoff}")
    for key in [
        "calls_model",
        "reads_private_data",
        "executes_side_effect",
        "external_side_effect",
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "queues_approval",
        "controls_computer",
        *NO_AUTHORITY_FLAGS.keys(),
    ]:
        if boundaries.get(key) or metadata.get(key):
            raise SystemExit(f"{label} should not perform {key}: {metadata}")
    return handoff


def assert_calendar_availability_handoff(metadata: dict, label: str, *, status: str) -> dict:
    handoff = metadata.get("calendar_availability_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed calendar availability handoff: {metadata}")
    if handoff.get("source") != "check_availability":
        raise SystemExit(f"{label} availability handoff source wrong: {handoff}")
    for key, expected_value in NO_AUTHORITY_FLAGS.items():
        if metadata.get(key) is not expected_value:
            raise SystemExit(f"{label} flat {key} should be {expected_value}: {metadata}")
        if handoff.get(key) is not expected_value:
            raise SystemExit(f"{label} availability handoff {key} should be {expected_value}: {handoff}")
    if handoff.get("status") != status:
        raise SystemExit(f"{label} availability handoff status wrong: {handoff}")
    if handoff.get("conflict_count") != metadata.get("count"):
        raise SystemExit(f"{label} availability handoff count parity failed: {metadata}")
    if handoff.get("calendar_id") != metadata.get("calendar_id", "primary"):
        raise SystemExit(f"{label} availability handoff calendar parity failed: {metadata}")
    if handoff.get("label") != metadata.get("label", handoff.get("label")):
        raise SystemExit(f"{label} availability handoff label parity failed: {metadata}")
    conflicts = handoff.get("conflicts")
    if not isinstance(conflicts, list):
        raise SystemExit(f"{label} availability conflicts should be rows: {handoff}")
    if handoff.get("content_in_metadata") is not False:
        raise SystemExit(f"{label} availability handoff should declare bounded metadata content: {handoff}")
    if _leaks_local_path(handoff):
        raise SystemExit(f"{label} availability handoff leaked local path: {handoff}")
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict):
        raise SystemExit(f"{label} availability handoff missed boundaries: {handoff}")
    if not boundaries.get("calls_external_service") or not boundaries.get("reads_personal_data"):
        raise SystemExit(f"{label} availability must declare external personal calendar read: {handoff}")
    for key in [
        "calls_model",
        "reads_private_data",
        "executes_side_effect",
        "external_side_effect",
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "queues_approval",
        "controls_computer",
        *NO_AUTHORITY_FLAGS.keys(),
    ]:
        if boundaries.get(key) or metadata.get(key):
            raise SystemExit(f"{label} should not perform {key}: {metadata}")
    return handoff


def assert_calendar_mutation_handoff(
    metadata: dict,
    label: str,
    *,
    tool_name: str,
    status: str,
    reason: str = "",
    service_contacted: bool,
    mutation_attempted: bool,
) -> dict:
    specific_key = f"{tool_name}_handoff"
    handoff = metadata.get(specific_key)
    if not isinstance(handoff, dict) or metadata.get("calendar_mutation_handoff") != handoff:
        raise SystemExit(f"{label} missed calendar mutation handoff aliases: {metadata}")
    if metadata.get(f"{tool_name}_handoff_ready") is not True or metadata.get("calendar_mutation_handoff_ready") is not True:
        raise SystemExit(f"{label} missed calendar mutation handoff readiness aliases: {metadata}")
    if handoff.get("source") != tool_name:
        raise SystemExit(f"{label} mutation handoff source wrong: {handoff}")
    if handoff.get("status") != status or handoff.get("reason") != reason:
        raise SystemExit(f"{label} mutation handoff status/reason wrong: {handoff}")
    expected_changed = [f"{tool_name}_mutation_attempt"] if mutation_attempted else []
    for key, value in [
        ("ready_for_operator", True),
        ("state_changed", mutation_attempted),
        ("changed", expected_changed),
        ("content_in_handoff", False),
        ("authorizes_execution", False),
        ("authorizes_completion_claim", False),
        ("approval_granted", False),
    ]:
        if handoff.get(key) != value or metadata.get(key) != value:
            raise SystemExit(f"{label} mutation handoff {key} parity wrong: {handoff} / {metadata}")
    if handoff.get("service_contacted") is not service_contacted:
        raise SystemExit(f"{label} mutation handoff service state wrong: {handoff}")
    if handoff.get("mutation_attempted") is not mutation_attempted:
        raise SystemExit(f"{label} mutation handoff attempted state wrong: {handoff}")
    if handoff.get("content_in_metadata") is not False or handoff.get("description_in_metadata") is not False:
        raise SystemExit(f"{label} mutation handoff should keep event body content out of metadata: {handoff}")
    if handoff.get("approval_required_before_execution") is not True or handoff.get("manual_review_required") is not True:
        raise SystemExit(f"{label} mutation handoff missed high-risk review flags: {handoff}")
    expected_commands = (
        ["what is on my calendar <date>", "recent tool runs"]
        if status == "outcome_unknown"
        else ["setup check"]
        if status in {"failed", "unavailable"}
        else ["recent tool runs"]
        if status == "ok"
        else ["safe next actions"]
    )
    if (
        handoff.get("next_safe_command") != expected_commands[0]
        or handoff.get("next_safe_commands") != expected_commands
        or handoff.get("next_safe_command_count") != len(expected_commands)
    ):
        raise SystemExit(f"{label} mutation handoff recovery commands drifted: {handoff}")
    if _leaks_local_path(handoff):
        raise SystemExit(f"{label} mutation handoff leaked local path: {handoff}")
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict):
        raise SystemExit(f"{label} mutation handoff missed boundaries: {handoff}")
    expected = {
        "calls_model": False,
        "calls_external_service": service_contacted,
        "reads_personal_data": False,
        "reads_private_data": False,
        "executes_side_effect": mutation_attempted,
        "external_side_effect": mutation_attempted,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": mutation_attempted,
        "controls_computer": False,
        **NO_AUTHORITY_FLAGS,
    }
    for key, value in expected.items():
        if boundaries.get(key) is not value:
            raise SystemExit(f"{label} mutation boundary {key} wrong: {handoff}")
        if key in metadata and metadata.get(key) is not value:
            raise SystemExit(f"{label} flat metadata {key} wrong: {metadata}")
    return handoff


def assert_calendar_mutation_outcome_unknown(result, label: str, *, tool_name: str) -> None:
    expected_output = cc._calendar_mutation_outcome_unknown_guidance(tool_name)
    if result.ok or result.output != expected_output:
        raise SystemExit(f"{label} missed fixed outcome-unknown guidance: {result}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": expected_output,
        "commands": ["what is on my calendar <date>", "recent tool runs"],
    }:
        raise SystemExit(f"{label} recovery declaration drifted: {result.metadata}")
    expected = {
        "outcome_known": False,
        "outcome_unknown": True,
        "execution_outcome_unknown": True,
        "side_effect_possible": True,
        "retry_safe": False,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "next_command": "what is on my calendar <date>",
        "recovery_commands": ["what is on my calendar <date>", "recent tool runs"],
    }
    for key, value in expected.items():
        if result.metadata.get(key) != value:
            raise SystemExit(f"{label} outcome/retry field {key} drifted: {result.metadata}")
    if _leaks_local_path(result.metadata):
        raise SystemExit(f"{label} leaked a local path: {result.metadata}")


class FakeExec:
    def __init__(self, payload):
        self._payload = payload

    def execute(self):
        return self._payload


class FailingExec:
    def __init__(self, message):
        self.message = message

    def execute(self):
        raise RuntimeError(self.message)


class FakeEvents:
    def __init__(self, store):
        self.store = store

    def list(self, **kwargs):
        self.store["last_list_kwargs"] = kwargs
        return FakeExec({"items": self.store.get("events", [])})

    def insert(self, calendarId, body):
        self.store["inserted"] = body
        return FakeExec({"id": "evt123", "summary": body["summary"], "start": body["start"]})

    def patch(self, calendarId, eventId, body):
        self.store["patched"] = body
        self.store["patched_id"] = eventId
        payload = {"id": eventId, "summary": body.get("summary", "Updated")}
        if "start" in body:
            payload["start"] = body["start"]
        return FakeExec(payload)

    def delete(self, calendarId, eventId):
        self.store["deleted"] = eventId
        return FakeExec({})


class FakeCalendarList:
    def __init__(self, store):
        self.store = store

    def list(self):
        return FakeExec({"items": self.store.get("calendars", [])})


class FakeService:
    def __init__(self, store):
        self.store = store

    def calendarList(self):
        return FakeCalendarList(self.store)

    def events(self):
        return FakeEvents(self.store)


def _tools(store):
    cc._get_service = lambda: FakeService(store)  # type: ignore
    cc._get_readonly_service = lambda: FakeService(store)  # type: ignore
    return {t.name: t for t in cc.make_calendar_tools(load_config())}


def _tools_with_forbidden_service():
    def forbidden_service():
        raise AssertionError("local validation should not initialize Google Calendar service")

    cc._get_service = forbidden_service  # type: ignore
    cc._get_readonly_service = forbidden_service  # type: ignore
    return {t.name: t for t in cc.make_calendar_tools(load_config())}


def test_list_calendars_formats_output() -> None:
    store = {"calendars": [{"summary": "Primary", "primary": True, "accessRole": "owner", "id": "p@x"}]}
    out = _tools(store)["list_calendars"].handler({})
    if not out.ok or "Primary" not in out.output or "★" not in out.output:
        raise SystemExit(f"list_calendars wrong: {out.output}")
    handoff = assert_calendar_list_handoff(out.metadata, "list_calendars", status="ok")
    if handoff.get("primary_calendar_id") != "p@x":
        raise SystemExit(f"list_calendars handoff missed primary calendar id: {handoff}")
    if handoff["calendars"][0].get("summary") != "Primary" or handoff["calendars"][0].get("access_role") != "owner":
        raise SystemExit(f"list_calendars handoff missed row fields: {handoff}")


def test_calendar_read_outputs_redact_path_shaped_text() -> None:
    calendars = [{
        "summary": "Work near /\x55sers/example/Desktop/Claude code/calendar.md",
        "primary": True,
        "accessRole": "owner",
        "id": "/var/folders/zc/calendar-id",
    }]
    cal_out = _tools({"calendars": calendars})["list_calendars"].handler({})
    if not cal_out.ok or _leaks_local_path(cal_out.output) or "<local-path>" not in cal_out.output:
        raise SystemExit(f"list_calendars should redact path-shaped display text: {cal_out.output}")
    cal_handoff = assert_calendar_list_handoff(cal_out.metadata, "path-shaped list_calendars", status="ok")
    if _leaks_local_path(cal_handoff) or "<local-path>" not in str(cal_handoff):
        raise SystemExit(f"list_calendars handoff should redact path-shaped calendar text: {cal_handoff}")

    events = [{
        "id": "evt-list-123",
        "summary": "Standup near /\x55sers/example/Desktop/Claude code/event.md",
        "location": "/tmp/calendar-room",
        "start": {"dateTime": "2026-06-16T09:00:00+09:00"},
    }]
    event_out = _tools({"events": events})["list_events"].handler({})
    if (
        not event_out.ok
        or _leaks_local_path(event_out.output)
        or "<local-path>" not in event_out.output
        or "id=evt-list-123" not in event_out.output
    ):
        raise SystemExit(f"list_events should redact path-shaped event text: {event_out.output}")
    event_handoff = assert_calendar_events_handoff(event_out.metadata, "path-shaped list_events", status="ok")
    if _leaks_local_path(event_handoff) or "<local-path>" not in str(event_handoff):
        raise SystemExit(f"list_events handoff should redact path-shaped event text: {event_handoff}")

    busy = _tools({"events": events})["check_availability"].handler({"text": "am I free today"})
    if not busy.ok or _leaks_local_path(busy.output) or "<local-path>" not in busy.output:
        raise SystemExit(f"check_availability should redact path-shaped conflict text: {busy.output}")
    busy_handoff = assert_calendar_availability_handoff(busy.metadata, "path-shaped check_availability", status="busy")
    if _leaks_local_path(busy_handoff) or "<local-path>" not in str(busy_handoff):
        raise SystemExit(f"check_availability handoff should redact path-shaped conflict text: {busy_handoff}")


def test_calendar_read_failures_are_clean() -> None:
    def boom():
        raise RuntimeError("raw calendar backend exploded")

    cc._get_readonly_service = boom  # type: ignore
    tools = {t.name: t for t in cc.make_calendar_tools(load_config())}
    cases = [
        ("list_calendars", {}),
        ("list_events", {"max_results": "bad"}),
        ("check_availability", {"text": "am I free today"}),
    ]
    for name, args in cases:
        out = tools[name].handler(args)
        if out.ok or "Google Calendar is having trouble" not in out.output:
            raise SystemExit(f"{name} failure should be friendly: {out.output}")
        assert_personal_read_recovery(out, f"{name} failure")
        assert_calendar_recovery_message(out.output, name)
        if "raw calendar backend exploded" in out.output or "Calendar error" in out.output:
            raise SystemExit(f"{name} failure should not leak raw exceptions: {out.output}")
        if out.metadata.get("exception_type") != "RuntimeError":
            raise SystemExit(f"{name} failure should preserve bounded diagnostic metadata: {out.metadata}")
        if name == "list_calendars":
            handoff = assert_calendar_list_handoff(out.metadata, "list_calendars failure", status="unavailable")
            if handoff.get("reason") != "fetch_error":
                raise SystemExit(f"list_calendars failure handoff should preserve fetch error reason: {handoff}")
        if name == "list_events":
            handoff = assert_calendar_events_handoff(out.metadata, "list_events failure", status="unavailable")
            if handoff.get("reason") != "fetch_error":
                raise SystemExit(f"list_events failure handoff should preserve fetch error reason: {handoff}")
        if name == "check_availability":
            handoff = assert_calendar_availability_handoff(out.metadata, "check_availability failure", status="unavailable")
            if handoff.get("reason") != "fetch_error":
                raise SystemExit(f"check_availability failure handoff should preserve fetch error reason: {handoff}")
    if tools["list_events"].handler({"max_results": "bad"}).metadata.get("raw_max_results") != "bad":
        raise SystemExit("list_events failure should preserve sanitized max_results metadata")


def test_create_event_requires_title_and_start() -> None:
    tools = _tools_with_forbidden_service()
    missing_title = tools["create_event"].handler({"start": "2026-06-20T10:00:00"})
    if missing_title.ok:
        raise SystemExit("create_event accepted missing title")
    assert_calendar_mutation_handoff(
        missing_title.metadata,
        "create_event missing title",
        tool_name="create_event",
        status="refused",
        reason="missing_title",
        service_contacted=False,
        mutation_attempted=False,
    )
    missing_start = tools["create_event"].handler({"title": "Lunch"})
    if missing_start.ok:
        raise SystemExit("create_event accepted missing start")
    handoff = assert_calendar_mutation_handoff(
        missing_start.metadata,
        "create_event missing start",
        tool_name="create_event",
        status="refused",
        reason="missing_start",
        service_contacted=False,
        mutation_attempted=False,
    )
    if handoff.get("title") != "Lunch":
        raise SystemExit(f"create_event missing-start handoff missed title preview: {handoff}")


def test_calendar_mutations_reject_path_shaped_fields_before_service_setup() -> None:
    tools = _tools_with_forbidden_service()

    for field, value in [
        ("title", "/\x55sers/example/Desktop/Claude code/calendar-title"),
        ("description", "/var/folders/zc/calendar-notes"),
        ("location", "/tmp/calendar-location"),
    ]:
        args = {"title": "Planning", "start": "2026-06-20T10:00:00", field: value}
        if field == "title":
            args["title"] = value
        created = tools["create_event"].handler(args)
        if created.ok or created.metadata.get("reason") != "invalid_event_text":
            raise SystemExit(f"create_event should reject path-shaped {field} locally: {created.output} {created.metadata}")
        if _leaks_local_path(created.output) or _leaks_local_path(created.metadata):
            raise SystemExit(f"create_event path-shaped rejection leaked local paths: {created.output} {created.metadata}")
        handoff = assert_calendar_mutation_handoff(
            created.metadata,
            f"create_event path-shaped {field}",
            tool_name="create_event",
            status="refused",
            reason="invalid_event_text",
            service_contacted=False,
            mutation_attempted=False,
        )
        if field != "description" and "<local-path>" not in str(handoff):
            raise SystemExit(f"create_event path-shaped handoff missed redaction marker: {handoff}")

    for event_id in ["/private/tmp/calendar-event-id", "/var/folders/zc/calendar-event-id", "/tmp/calendar-event-id"]:
        updated_id = tools["update_event"].handler({"event_id": event_id, "title": "Renamed"})
        if updated_id.ok or updated_id.metadata.get("reason") != "invalid_event_id" or updated_id.metadata.get("event_id") != "<local-path>":
            raise SystemExit(f"update_event should reject path-shaped event IDs locally: {updated_id.output} {updated_id.metadata}")
        if _leaks_local_path(updated_id.output) or _leaks_local_path(updated_id.metadata):
            raise SystemExit(f"update_event event-id rejection leaked local paths: {updated_id.output} {updated_id.metadata}")
        handoff = assert_calendar_mutation_handoff(
            updated_id.metadata,
            "update_event path-shaped id",
            tool_name="update_event",
            status="refused",
            reason="invalid_event_id",
            service_contacted=False,
            mutation_attempted=False,
        )
        if handoff.get("event_id") != "<local-path>":
            raise SystemExit(f"update_event path-shaped id handoff missed redaction: {handoff}")

    for field, value in [
        ("title", "/\x55sers/example/Desktop/Claude code/calendar-location"),
        ("description", "/var/folders/zc/calendar-location"),
        ("location", "/tmp/calendar-location"),
    ]:
        updated_text = tools["update_event"].handler({"event_id": "evt123", field: value})
        if updated_text.ok or updated_text.metadata.get("reason") != "invalid_event_text":
            raise SystemExit(f"update_event should reject path-shaped {field} locally: {updated_text.output} {updated_text.metadata}")
        if _leaks_local_path(updated_text.output) or _leaks_local_path(updated_text.metadata):
            raise SystemExit(f"update_event text rejection leaked local paths: {updated_text.output} {updated_text.metadata}")
        assert_calendar_mutation_handoff(
            updated_text.metadata,
            f"update_event path-shaped {field}",
            tool_name="update_event",
            status="refused",
            reason="invalid_event_text",
            service_contacted=False,
            mutation_attempted=False,
        )

    for event_id in [
        "/\x55sers/example/Desktop/Claude code/calendar-delete-id",
        "/var/folders/zc/calendar-delete-id",
        "/tmp/calendar-delete-id",
    ]:
        deleted = tools["delete_event"].handler({"event_id": event_id})
        if deleted.ok or deleted.metadata.get("reason") != "invalid_event_id" or deleted.metadata.get("event_id") != "<local-path>":
            raise SystemExit(f"delete_event should reject path-shaped event IDs locally: {deleted.output} {deleted.metadata}")
        if _leaks_local_path(deleted.output) or _leaks_local_path(deleted.metadata):
            raise SystemExit(f"delete_event rejection leaked local paths: {deleted.output} {deleted.metadata}")
        handoff = assert_calendar_mutation_handoff(
            deleted.metadata,
            "delete_event path-shaped id",
            tool_name="delete_event",
            status="refused",
            reason="invalid_event_id",
            service_contacted=False,
            mutation_attempted=False,
        )
        if handoff.get("event_id") != "<local-path>":
            raise SystemExit(f"delete_event path-shaped id handoff missed redaction: {handoff}")


def test_create_event_inserts() -> None:
    store = {}
    out = _tools(store)["create_event"].handler({"title": "Lunch", "start": "2026-06-20T12:00:00", "end": "2026-06-20T13:00:00"})
    if (
        not out.ok
        or store.get("inserted", {}).get("summary") != "Lunch"
        or "id=evt123" not in out.output
    ):
        raise SystemExit(f"create_event did not insert: {out.output} {store}")
    if not out.metadata.get("executes_side_effect"):
        raise SystemExit("create_event must be marked as a side effect")
    handoff = assert_calendar_mutation_handoff(
        out.metadata,
        "create_event success",
        tool_name="create_event",
        status="ok",
        service_contacted=True,
        mutation_attempted=True,
    )
    if handoff.get("event_id") != "evt123" or handoff.get("title") != "Lunch":
        raise SystemExit(f"create_event success handoff missed event id/title: {handoff}")


def test_create_all_day_event_uses_date_fields() -> None:
    store = {}
    out = _tools(store)["create_event"].handler({"title": "PTO", "start": "2026-06-16", "end": "2026-06-17"})
    inserted = store.get("inserted", {})
    if not out.ok or inserted.get("start") != {"date": "2026-06-16"} or inserted.get("end") != {"date": "2026-06-17"}:
        raise SystemExit(f"all-day create_event should use date fields: {out.output} {inserted}")
    if "2026-06-16" not in out.output:
        raise SystemExit(f"all-day create_event output should include date: {out.output}")
    handoff = assert_calendar_mutation_handoff(
        out.metadata,
        "create_event all-day success",
        tool_name="create_event",
        status="ok",
        service_contacted=True,
        mutation_attempted=True,
    )
    if handoff.get("start") != "2026-06-16" or handoff.get("end") != "2026-06-17":
        raise SystemExit(f"all-day create_event handoff missed date-only fields: {handoff}")


def test_create_event_failure_preserves_attempt_metadata() -> None:
    class FailingInsertEvents(FakeEvents):
        def insert(self, calendarId, body):
            self.store["insert_attempted"] = body.get("summary")
            return FailingExec("insert failed")

    class FailingInsertService(FakeService):
        def events(self):
            return FailingInsertEvents(self.store)

    store = {}
    cc._get_service = lambda: FailingInsertService(store)  # type: ignore
    tools = {t.name: t for t in cc.make_calendar_tools(load_config())}

    created = tools["create_event"].handler({"title": "Lunch", "start": "2026-06-20T12:00:00"})
    assert_calendar_mutation_outcome_unknown(
        created,
        "create_event insert failure",
        tool_name="create_event",
    )
    if "insert failed" in created.output or "Calendar error" in created.output:
        raise SystemExit(f"create_event failure should not leak raw exceptions: {created.output}")
    if store.get("insert_attempted") != "Lunch":
        raise SystemExit(f"create_event did not reach mocked insert: {store}")
    if created.metadata.get("title") != "Lunch" or not created.metadata.get("mutation_attempted"):
        raise SystemExit(f"create_event failure should preserve attempted mutation metadata: {created.metadata}")
    if not created.metadata.get("executes_side_effect") or not created.metadata.get("external_side_effect"):
        raise SystemExit(f"create_event failure should mark attempted external side effect: {created.metadata}")
    if created.metadata.get("exception_type") != "RuntimeError":
        raise SystemExit(f"create_event failure should preserve bounded diagnostic metadata: {created.metadata}")
    handoff = assert_calendar_mutation_handoff(
        created.metadata,
        "create_event insert failure",
        tool_name="create_event",
        status="outcome_unknown",
        reason="calendar_mutation_outcome_unknown",
        service_contacted=True,
        mutation_attempted=True,
    )
    if handoff.get("exception_type") != "RuntimeError":
        raise SystemExit(f"create_event failure handoff missed exception type: {handoff}")


def test_update_event_requires_id_and_changes() -> None:
    tools = _tools_with_forbidden_service()
    missing_id = tools["update_event"].handler({"title": "Renamed"})
    if missing_id.ok:
        raise SystemExit("update_event accepted missing event_id")
    assert_calendar_mutation_handoff(
        missing_id.metadata,
        "update_event missing id",
        tool_name="update_event",
        status="refused",
        reason="missing_event_id",
        service_contacted=False,
        mutation_attempted=False,
    )
    missing_changes = tools["update_event"].handler({"event_id": "evt123"})
    if missing_changes.ok:
        raise SystemExit("update_event accepted no changes")
    assert_calendar_mutation_handoff(
        missing_changes.metadata,
        "update_event missing changes",
        tool_name="update_event",
        status="refused",
        reason="missing_changes",
        service_contacted=False,
        mutation_attempted=False,
    )
    missing_start = tools["update_event"].handler({"event_id": "evt123", "end": "2026-06-20T15:00:00"})
    if missing_start.ok:
        raise SystemExit("update_event accepted end change without start")
    assert_calendar_mutation_handoff(
        missing_start.metadata,
        "update_event missing start for end",
        tool_name="update_event",
        status="refused",
        reason="missing_start_for_end_change",
        service_contacted=False,
        mutation_attempted=False,
    )
    if tools["update_event"].risk.name != "HIGH_RISK":
        raise SystemExit("update_event must stay HIGH_RISK")


def test_delete_event_requires_id_before_service_setup() -> None:
    tools = _tools_with_forbidden_service()
    out = tools["delete_event"].handler({"calendar_id": "primary"})
    if out.ok or "event_id" not in out.output:
        raise SystemExit(f"delete_event accepted missing event_id: {out}")
    if tools["delete_event"].risk.name != "HIGH_RISK":
        raise SystemExit("delete_event must stay HIGH_RISK")
    if out.metadata.get("executes_side_effect") or out.metadata.get("external_side_effect"):
        raise SystemExit(f"delete_event validation must not claim a side effect: {out.metadata}")
    assert_calendar_mutation_handoff(
        out.metadata,
        "delete_event missing id",
        tool_name="delete_event",
        status="refused",
        reason="missing_event_id",
        service_contacted=False,
        mutation_attempted=False,
    )


def test_update_event_patches_timed_event() -> None:
    store = {}
    out = _tools(store)["update_event"].handler({
        "event_id": "evt123",
        "title": "Moved lunch",
        "start": "2026-06-20T14:00:00",
        "end": "2026-06-20T15:00:00",
    })
    patched = store.get("patched", {})
    if (
        not out.ok
        or store.get("patched_id") != "evt123"
        or patched.get("summary") != "Moved lunch"
        or "id=evt123" not in out.output
    ):
        raise SystemExit(f"update_event did not patch expected event: {out.output} {store}")
    if patched.get("start", {}).get("dateTime") != "2026-06-20T14:00:00":
        raise SystemExit(f"timed update_event should use dateTime: {patched}")
    if not out.metadata.get("executes_side_effect"):
        raise SystemExit("update_event must be marked as a side effect")
    handoff = assert_calendar_mutation_handoff(
        out.metadata,
        "update_event success",
        tool_name="update_event",
        status="ok",
        service_contacted=True,
        mutation_attempted=True,
    )
    if handoff.get("event_id") != "evt123" or handoff.get("title") != "Moved lunch":
        raise SystemExit(f"update_event success handoff missed event id/title: {handoff}")


def test_update_all_day_event_uses_date_fields() -> None:
    store = {}
    out = _tools(store)["update_event"].handler({"event_id": "evt123", "start": "2026-06-16", "end": "2026-06-17"})
    patched = store.get("patched", {})
    if not out.ok or patched.get("start") != {"date": "2026-06-16"} or patched.get("end") != {"date": "2026-06-17"}:
        raise SystemExit(f"all-day update_event should use date fields: {out.output} {patched}")
    handoff = assert_calendar_mutation_handoff(
        out.metadata,
        "update_event all-day success",
        tool_name="update_event",
        status="ok",
        service_contacted=True,
        mutation_attempted=True,
    )
    if handoff.get("start") != "2026-06-16" or handoff.get("end") != "2026-06-17":
        raise SystemExit(f"all-day update_event handoff missed date-only fields: {handoff}")


def test_calendar_mutation_failures_preserve_attempt_metadata() -> None:
    class FailingEvents(FakeEvents):
        def patch(self, calendarId, eventId, body):
            self.store["patch_attempted"] = eventId
            return FailingExec("patch failed")

        def delete(self, calendarId, eventId):
            self.store["delete_attempted"] = eventId
            return FailingExec("delete failed")

    class FailingService(FakeService):
        def events(self):
            return FailingEvents(self.store)

    store = {}
    cc._get_service = lambda: FailingService(store)  # type: ignore
    tools = {t.name: t for t in cc.make_calendar_tools(load_config())}

    updated = tools["update_event"].handler({"event_id": "evt123", "title": "Renamed"})
    assert_calendar_mutation_outcome_unknown(
        updated,
        "update_event patch failure",
        tool_name="update_event",
    )
    if "patch failed" in updated.output or "Calendar error" in updated.output:
        raise SystemExit(f"update_event failure should not leak raw exceptions: {updated.output}")
    if store.get("patch_attempted") != "evt123":
        raise SystemExit(f"update_event did not reach mocked patch: {store}")
    if updated.metadata.get("event_id") != "evt123" or not updated.metadata.get("mutation_attempted"):
        raise SystemExit(f"update_event failure should preserve attempted mutation metadata: {updated.metadata}")
    if not updated.metadata.get("executes_side_effect") or not updated.metadata.get("external_side_effect"):
        raise SystemExit(f"update_event failure should mark attempted external side effect: {updated.metadata}")
    if updated.metadata.get("exception_type") != "RuntimeError":
        raise SystemExit(f"update_event failure should preserve bounded diagnostic metadata: {updated.metadata}")
    handoff = assert_calendar_mutation_handoff(
        updated.metadata,
        "update_event patch failure",
        tool_name="update_event",
        status="outcome_unknown",
        reason="calendar_mutation_outcome_unknown",
        service_contacted=True,
        mutation_attempted=True,
    )
    if handoff.get("exception_type") != "RuntimeError":
        raise SystemExit(f"update_event failure handoff missed exception type: {handoff}")

    deleted = tools["delete_event"].handler({"event_id": "evt456"})
    assert_calendar_mutation_outcome_unknown(
        deleted,
        "delete_event failure",
        tool_name="delete_event",
    )
    if "delete failed" in deleted.output or "Calendar error" in deleted.output:
        raise SystemExit(f"delete_event failure should not leak raw exceptions: {deleted.output}")
    if store.get("delete_attempted") != "evt456":
        raise SystemExit(f"delete_event did not reach mocked delete: {store}")
    if deleted.metadata.get("event_id") != "evt456" or not deleted.metadata.get("mutation_attempted"):
        raise SystemExit(f"delete_event failure should preserve attempted mutation metadata: {deleted.metadata}")
    if not deleted.metadata.get("executes_side_effect") or not deleted.metadata.get("external_side_effect"):
        raise SystemExit(f"delete_event failure should mark attempted external side effect: {deleted.metadata}")
    if deleted.metadata.get("exception_type") != "RuntimeError":
        raise SystemExit(f"delete_event failure should preserve bounded diagnostic metadata: {deleted.metadata}")
    handoff = assert_calendar_mutation_handoff(
        deleted.metadata,
        "delete_event failure",
        tool_name="delete_event",
        status="outcome_unknown",
        reason="calendar_mutation_outcome_unknown",
        service_contacted=True,
        mutation_attempted=True,
    )
    if handoff.get("exception_type") != "RuntimeError" or handoff.get("event_id") != "evt456":
        raise SystemExit(f"delete_event failure handoff missed exception/event id: {handoff}")


def test_list_events_empty() -> None:
    out = _tools({"events": []})["list_events"].handler({})
    if not out.ok or "No events upcoming" not in out.output:
        raise SystemExit(f"list_events empty wrong: {out.output}")
    if out.metadata.get("max_results") != 10:
        raise SystemExit(f"list_events empty output missed default max_results metadata: {out.metadata}")
    handoff = assert_calendar_events_handoff(out.metadata, "list_events empty", status="empty")
    if handoff.get("reason") != "no_events" or handoff.get("events") != []:
        raise SystemExit(f"list_events empty handoff should preserve empty state: {handoff}")


def test_list_events_sanitizes_bad_max_results() -> None:
    store = {"events": []}
    out = _tools(store)["list_events"].handler({"max_results": "bad"})
    if not out.ok or out.metadata.get("max_results") != 10 or out.metadata.get("raw_max_results") != "bad":
        raise SystemExit(f"list_events should preserve sanitized raw max_results metadata: {out.metadata}")
    if store.get("last_list_kwargs", {}).get("maxResults") != 10:
        raise SystemExit(f"list_events should send sanitized maxResults to Google service: {store}")
    long_store = {"events": []}
    long_out = _tools(long_store)["list_events"].handler({"max_results": "m" * 120})
    if long_out.metadata.get("raw_max_results") != ("m" * 79 + "…"):
        raise SystemExit(f"list_events should bound long raw max_results metadata: {long_out.metadata}")
    clamped_store = {"events": []}
    clamped = _tools(clamped_store)["list_events"].handler({"max_results": 9999})
    if not clamped.ok or clamped.metadata.get("max_results") != 50:
        raise SystemExit(f"list_events should clamp large max_results metadata: {clamped.metadata}")
    if clamped_store.get("last_list_kwargs", {}).get("maxResults") != 50:
        raise SystemExit(f"list_events should send clamped maxResults to Google service: {clamped_store}")
    bool_store = {"events": []}
    bool_out = _tools(bool_store)["list_events"].handler({"max_results": True})
    if not bool_out.ok or bool_out.metadata.get("max_results") != 10 or bool_out.metadata.get("raw_max_results") != "True":
        raise SystemExit(f"list_events should treat boolean max_results as malformed defaults: {bool_out.metadata}")
    if bool_store.get("last_list_kwargs", {}).get("maxResults") != 10:
        raise SystemExit(f"list_events should send default maxResults for boolean input: {bool_store}")
    path_store = {"events": []}
    path_out = _tools(path_store)["list_events"].handler({"max_results": "/var/folders/zc/calendar-limit"})
    if not path_out.ok or path_out.metadata.get("max_results") != 10 or path_out.metadata.get("raw_max_results") != "<local-path>":
        raise SystemExit(f"list_events leaked local path in raw max_results metadata: {path_out.metadata}")
    if path_store.get("last_list_kwargs", {}).get("maxResults") != 10:
        raise SystemExit(f"list_events should send default maxResults for path-shaped input: {path_store}")
    assert_calendar_events_handoff(path_out.metadata, "path-shaped max_results list_events", status="empty")


def test_list_events_range_labels() -> None:
    ev = [{"summary": "Standup", "start": {"dateTime": "2026-06-16T09:00:00+09:00"}}]
    for rng, label in [
        ("today", "today"),
        ("today_morning", "today morning"),
        ("today_afternoon", "today afternoon"),
        ("tomorrow", "tomorrow"),
        ("tomorrow_morning", "tomorrow morning"),
        ("tomorrow_afternoon", "tomorrow afternoon"),
        ("tomorrow_night", "tomorrow evening"),
        ("today_evening", "today evening"),
        ("this_morning", "today morning"),
        ("this_afternoon", "today afternoon"),
        ("this_evening", "today evening"),
        ("this_night", "today evening"),
        ("week", "this week"),
        ("next_week", "next week"),
        ("weekend", "this weekend"),
        ("this_weekend", "this weekend"),
        ("next_weekend", "next weekend"),
        ("month", "this month"),
        ("this_month", "this month"),
        ("next_month", "next month"),
        ("year", "this year"),
        ("this_year", "this year"),
        ("next_year", "next year"),
        ("monday", "Monday"),
        ("monday_morning", "Monday morning"),
        ("monday_afternoon", "Monday afternoon"),
        ("monday_night", "Monday evening"),
        ("next_monday", "next Monday"),
        ("next_monday_morning", "next Monday morning"),
        ("next_monday_night", "next Monday evening"),
        ("2026-07-15", "July 15, 2026"),
    ]:
        store = {"events": ev}
        out = _tools(store)["list_events"].handler({"range": rng})
        if not out.ok or f"Events {label}" not in out.output:
            raise SystemExit(f"range {rng} label wrong: {out.output}")
        handoff = assert_calendar_events_handoff(out.metadata, f"list_events range {rng}", status="ok")
        if handoff.get("label") != label or handoff.get("event_count") != 1:
            raise SystemExit(f"list_events range handoff missed label/count: {handoff}")
        if handoff["events"][0].get("summary") != "Standup" or not handoff["events"][0].get("start"):
            raise SystemExit(f"list_events range handoff missed event row fields: {handoff}")
        if rng == "next_week":
            kwargs = store.get("last_list_kwargs", {})
            start = datetime.fromisoformat(str(kwargs.get("timeMin")))
            end = datetime.fromisoformat(str(kwargs.get("timeMax")))
            if start.weekday() != 0 or start.hour != 0 or start.minute != 0:
                raise SystemExit(f"next_week should start at local Monday midnight: {kwargs}")
            if end - start != timedelta(days=7):
                raise SystemExit(f"next_week should span exactly seven days: {kwargs}")
        if rng in {"weekend", "this_weekend", "next_weekend"}:
            kwargs = store.get("last_list_kwargs", {})
            start = datetime.fromisoformat(str(kwargs.get("timeMin")))
            end = datetime.fromisoformat(str(kwargs.get("timeMax")))
            if start.weekday() != 5 or start.hour != 0 or start.minute != 0:
                raise SystemExit(f"{rng} should start at local Saturday midnight: {kwargs}")
            if end - start != timedelta(days=2):
                raise SystemExit(f"{rng} should span exactly two days: {kwargs}")
        if rng in {"month", "this_month", "next_month"}:
            kwargs = store.get("last_list_kwargs", {})
            start = datetime.fromisoformat(str(kwargs.get("timeMin")))
            end = datetime.fromisoformat(str(kwargs.get("timeMax")))
            if start.day != 1 or start.hour != 0 or start.minute != 0:
                raise SystemExit(f"{rng} should start at local month boundary: {kwargs}")
            if end.day != 1 or end.hour != 0 or end.minute != 0 or end <= start:
                raise SystemExit(f"{rng} should end at following local month boundary: {kwargs}")
            if not 28 <= (end - start).days <= 31:
                raise SystemExit(f"{rng} should span one calendar month: {kwargs}")
        if rng in {"year", "this_year", "next_year"}:
            kwargs = store.get("last_list_kwargs", {})
            start = datetime.fromisoformat(str(kwargs.get("timeMin")))
            end = datetime.fromisoformat(str(kwargs.get("timeMax")))
            if start.month != 1 or start.day != 1 or start.hour != 0 or start.minute != 0:
                raise SystemExit(f"{rng} should start at local year boundary: {kwargs}")
            if end.month != 1 or end.day != 1 or end.hour != 0 or end.minute != 0 or end <= start:
                raise SystemExit(f"{rng} should end at following local year boundary: {kwargs}")
            if not 365 <= (end - start).days <= 366:
                raise SystemExit(f"{rng} should span one calendar year: {kwargs}")
        if rng in {"monday", "next_monday"}:
            kwargs = store.get("last_list_kwargs", {})
            start = datetime.fromisoformat(str(kwargs.get("timeMin")))
            end = datetime.fromisoformat(str(kwargs.get("timeMax")))
            if start.weekday() != 0 or start.hour != 0 or start.minute != 0:
                raise SystemExit(f"{rng} should start at local Monday midnight: {kwargs}")
            if end - start != timedelta(days=1):
                raise SystemExit(f"{rng} should span exactly one day: {kwargs}")
        if rng in {"today_morning", "this_morning", "tomorrow_morning", "monday_morning", "next_monday_morning"}:
            kwargs = store.get("last_list_kwargs", {})
            start = datetime.fromisoformat(str(kwargs.get("timeMin")))
            end = datetime.fromisoformat(str(kwargs.get("timeMax")))
            if start.hour != 8 or end.hour != 12:
                raise SystemExit(f"{rng} should keep morning bounds: {kwargs}")
        if rng in {"today_afternoon", "this_afternoon", "tomorrow_afternoon", "monday_afternoon"}:
            kwargs = store.get("last_list_kwargs", {})
            start = datetime.fromisoformat(str(kwargs.get("timeMin")))
            end = datetime.fromisoformat(str(kwargs.get("timeMax")))
            if start.hour != 12 or end.hour != 18:
                raise SystemExit(f"{rng} should keep afternoon bounds: {kwargs}")
        if rng in {"today_evening", "this_evening", "this_night", "tomorrow_night", "monday_night", "next_monday_night"}:
            kwargs = store.get("last_list_kwargs", {})
            start = datetime.fromisoformat(str(kwargs.get("timeMin")))
            end = datetime.fromisoformat(str(kwargs.get("timeMax")))
            if start.hour != 18 or end.hour != 22:
                raise SystemExit(f"{rng} should keep evening bounds: {kwargs}")
    same_day_next = cc._weekday_range("next monday", datetime.fromisoformat("2026-06-15T10:30:00+09:00"))
    if same_day_next is None or same_day_next[0].date().isoformat() != "2026-06-22" or same_day_next[2] != "next Monday":
        raise SystemExit(f"next weekday event range should skip same-day Monday: {same_day_next}")


def test_absolute_date_ranges_and_invalid_refusal() -> None:
    fixed_now = datetime.fromisoformat("2026-07-14T10:30:00+09:00")
    date_cases = {
        "2026-07-15": ("2026-07-15", "July 15, 2026"),
        "2026/07/15": ("2026-07-15", "July 15, 2026"),
        "07/15/2026": ("2026-07-15", "July 15, 2026"),
        "7/15": ("2026-07-15", "July 15, 2026"),
        "7-15-2026": ("2026-07-15", "July 15, 2026"),
        "Jul 15": ("2026-07-15", "July 15, 2026"),
        "Sept 15": ("2026-09-15", "September 15, 2026"),
        "15 Sept": ("2026-09-15", "September 15, 2026"),
        "July 15th": ("2026-07-15", "July 15, 2026"),
        "15 Jul": ("2026-07-15", "July 15, 2026"),
    }
    for value, (expected_date, expected_label) in date_cases.items():
        window = cc._absolute_date_range(value, fixed_now)
        if window is None:
            raise SystemExit(f"absolute calendar range was not parsed: {value!r}")
        start, end, label = window
        if start.date().isoformat() != expected_date or end - start != timedelta(days=1):
            raise SystemExit(f"absolute calendar range bounds were wrong: {value!r} -> {window}")
        if label != expected_label:
            raise SystemExit(f"absolute calendar range label was wrong: {value!r} -> {label!r}")

    for value in ("2026-02-30", "July 99", "/\x55sers/example/private/date"):
        if cc._absolute_date_range(value, fixed_now) is not None:
            raise SystemExit(f"invalid absolute calendar range parsed: {value!r}")

    store = {"events": []}
    invalid = _tools(store)["list_events"].handler({"range": "2026-02-30"})
    if invalid.ok or invalid.metadata.get("reason") != "invalid_range":
        raise SystemExit(f"invalid calendar range should fail closed: {invalid}")
    if (
        store.get("last_list_kwargs") is not None
        or invalid.metadata.get("calls_external_service") is not False
        or invalid.metadata.get("reads_personal_data") is not False
    ):
        raise SystemExit(f"invalid calendar range contacted Google: {store} {invalid.metadata}")
    handoff = assert_calendar_events_handoff(
        invalid.metadata,
        "invalid list_events range",
        status="invalid",
        calls_external_service=False,
    )
    if handoff.get("reason") != "invalid_range" or handoff.get("boundaries", {}).get("calls_external_service") is not False:
        raise SystemExit(f"invalid calendar range handoff was untruthful: {handoff}")


def test_planner_routes_calendar_event_list_aliases() -> None:
    planner = RuleBasedPlanner()
    for text in [
        "show my calendar today",
        "calendar please",
        "show calendar",
        "show latest calendar",
        "calendar brief please",
        "what is on my calendar today",
        "what is on my schedule today",
        "what's on my agenda tomorrow",
        "calendar today",
        "schedule today",
        "what am I doing today",
        "what am I up to tomorrow",
        "what's my day tomorrow",
        "how does my day look tomorrow",
        "what does my day look like today",
        "what does my day look like tomorrow",
        "what's my day look like tomorrow",
        "how is my day looking tomorrow",
        "how does tomorrow look",
        "what does tomorrow look like",
        "what does my week look like",
        "what does my schedule look like",
        "what does my schedule look like today",
        "what does my schedule look like tomorrow",
        "how does my schedule look tomorrow",
        "what's my schedule look like tomorrow",
        "what's my schedule tomorrow",
        "what is my schedule tomorrow",
        "how's my schedule tomorrow",
        "how is my schedule tomorrow",
        "my schedule tomorrow",
        "what's my schedule",
        "what does my calendar look like tomorrow",
        "what's my calendar tomorrow",
        "my calendar tomorrow",
        "how does my calendar look tomorrow",
        "what does my agenda look like tomorrow",
        "what's my agenda tomorrow",
        "my agenda tomorrow",
        "how does my agenda look tomorrow",
        "what's my schedule next week",
        "my schedule next week",
        "what's my calendar next week",
        "my calendar next week",
        "what's my agenda next week",
        "my agenda next week",
        "open calendar next week",
        "calendar monday",
        "calendar next monday",
        "schedule monday",
        "agenda monday",
        "events monday",
        "events next monday",
        "what do i have monday",
        "what do i have next monday",
        "what's on my calendar monday",
        "what's on my calendar next monday",
        "what's my schedule monday",
        "what's my schedule next monday",
        "my schedule monday",
        "my schedule next monday",
        "open calendar monday",
        "open calendar next monday",
        "calendar tomorrow morning",
        "calendar tomorrow afternoon",
        "calendar tonight",
        "what do i have tomorrow morning",
        "what's on my calendar tomorrow morning",
        "what's my schedule monday morning",
        "my schedule monday afternoon",
        "calendar monday morning",
        "events monday afternoon",
        "open calendar next monday morning",
        "what's my agenda friday evening",
        "calendar this morning",
        "calendar this afternoon",
        "calendar this evening",
        "schedule this morning",
        "agenda this afternoon",
        "events this evening",
        "what do i have this morning",
        "what do i have this afternoon",
        "what's on my calendar this evening",
        "what's my schedule this morning",
        "my schedule this afternoon",
        "open calendar this morning",
        "open calendar this afternoon",
        "open calendar this evening",
        "check my calendar tomorrow",
        "open calendar tomorrow",
        "open calendar this week",
        "view my calendar this week",
        "calendar this year",
        "calendar next year",
        "schedule this year",
        "agenda next year",
        "events this year",
        "what do i have this year",
        "what's on my calendar next year",
        "what's my schedule this year",
        "my schedule next year",
        "open calendar this year",
        "open calendar next year",
    ]:
        plan = planner.plan(text)
        if [a.tool_name for a in plan.actions] != ["list_events"]:
            raise SystemExit(f"planner missed calendar event-list route: {text!r}: {plan}")
    if planner.plan("show my calendar today").actions[0].args.get("range") != "today":
        raise SystemExit("planner should extract today range for show-my-calendar alias")
    korean_reads = {
        "오늘 일정 알려줘": "today",
        "오늘 내 일정 알려주세요": "today",
        "오늘 스케줄 보여줘": "today",
        "오늘 캘린더 확인해 줘": "today",
        "내일 일정 알려줘요": "tomorrow",
        "내일 내 스케줄 보여주세요": "tomorrow",
        "내일 캘린더": "tomorrow",
    }
    for text, expected_range in korean_reads.items():
        actions = planner.plan(text).actions
        if [action.tool_name for action in actions] != ["list_events"]:
            raise SystemExit(f"planner missed Korean calendar-read route: {text!r}: {actions}")
        if actions[0].args != {"range": expected_range}:
            raise SystemExit(
                f"planner extracted wrong Korean calendar range: {text!r}: {actions[0].args}"
            )
    for text in (
        "오늘 일정 잡아줘",
        "내일 일정 추가해줘",
        "내일 회의 만들어줘",
        "오늘 일정 바꿔줘",
    ):
        actions = planner.plan(text).actions
        if any(action.tool_name == "list_events" for action in actions):
            raise SystemExit(f"Korean calendar read route stole a write request: {text!r}: {actions}")
    if planner.plan("calendar today").actions[0].args.get("range") != "today":
        raise SystemExit("planner should extract today range for bare calendar-today alias")
    if planner.plan("schedule today").actions[0].args.get("range") != "today":
        raise SystemExit("planner should extract today range for terse schedule alias")
    if planner.plan("what am I doing today").actions[0].args.get("range") != "today":
        raise SystemExit("planner should extract today range for doing-today alias")
    if planner.plan("what am I up to tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for up-to alias")
    if planner.plan("what's my day tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for my-day alias")
    if planner.plan("how does my day look tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for day-look alias")
    if planner.plan("what does my day look like today").actions[0].args.get("range") != "today":
        raise SystemExit("planner should extract today range for does-my-day-look alias")
    if planner.plan("what does my day look like tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for does-my-day-look alias")
    if planner.plan("what's my day look like tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for whats-my-day-look alias")
    if planner.plan("how is my day looking tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for day-looking alias")
    if planner.plan("how does tomorrow look").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for tomorrow-look alias")
    if planner.plan("what does tomorrow look like").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for tomorrow-look-like alias")
    if planner.plan("what does my week look like").actions[0].args.get("range") != "week":
        raise SystemExit("planner should extract week range for my-week-look alias")
    if planner.plan("what does my schedule look like").actions[0].args:
        raise SystemExit("planner should not invent a range for undated schedule-look alias")
    if planner.plan("what does my schedule look like today").actions[0].args.get("range") != "today":
        raise SystemExit("planner should extract today range for schedule-look alias")
    if planner.plan("what does my schedule look like tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for schedule-look alias")
    if planner.plan("how does my schedule look tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for schedule-how alias")
    if planner.plan("what's my schedule look like tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for schedule-whats alias")
    if planner.plan("what's my schedule tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for abbreviated schedule alias")
    if planner.plan("what is my schedule tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for what-is schedule alias")
    if planner.plan("how's my schedule tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for hows schedule alias")
    if planner.plan("how is my schedule tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for how-is schedule alias")
    if planner.plan("my schedule tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for bare schedule alias")
    if planner.plan("what's my schedule").actions[0].args:
        raise SystemExit("planner should not invent a range for abbreviated undated schedule alias")
    if planner.plan("what does my calendar look like tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for calendar-look alias")
    if planner.plan("what's my calendar tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for abbreviated calendar alias")
    if planner.plan("my calendar tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for bare calendar alias")
    if planner.plan("how does my calendar look tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for calendar-how alias")
    if planner.plan("what does my agenda look like tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for agenda-look alias")
    if planner.plan("what's my agenda tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for abbreviated agenda alias")
    if planner.plan("my agenda tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for bare agenda alias")
    if planner.plan("how does my agenda look tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for agenda-how alias")
    if planner.plan("what's my schedule next week").actions[0].args.get("range") != "next_week":
        raise SystemExit("planner should extract next_week range for abbreviated schedule alias")
    if planner.plan("my schedule next week").actions[0].args.get("range") != "next_week":
        raise SystemExit("planner should extract next_week range for bare schedule alias")
    if planner.plan("what's my calendar next week").actions[0].args.get("range") != "next_week":
        raise SystemExit("planner should extract next_week range for abbreviated calendar alias")
    if planner.plan("my calendar next week").actions[0].args.get("range") != "next_week":
        raise SystemExit("planner should extract next_week range for bare calendar alias")
    if planner.plan("what's my agenda next week").actions[0].args.get("range") != "next_week":
        raise SystemExit("planner should extract next_week range for abbreviated agenda alias")
    if planner.plan("my agenda next week").actions[0].args.get("range") != "next_week":
        raise SystemExit("planner should extract next_week range for bare agenda alias")
    if planner.plan("open calendar next week").actions[0].args.get("range") != "next_week":
        raise SystemExit("planner should extract next_week range for open-calendar alias")
    if planner.plan("calendar this weekend").actions[0].args.get("range") != "weekend":
        raise SystemExit("planner should extract weekend range for calendar-this-weekend alias")
    if planner.plan("calendar weekend").actions[0].args.get("range") != "weekend":
        raise SystemExit("planner should extract weekend range for bare calendar-weekend alias")
    if planner.plan("calendar next weekend").actions[0].args.get("range") != "next_weekend":
        raise SystemExit("planner should extract next_weekend range for calendar-next-weekend alias")
    if planner.plan("schedule this weekend").actions[0].args.get("range") != "weekend":
        raise SystemExit("planner should extract weekend range for schedule-this-weekend alias")
    if planner.plan("agenda next weekend").actions[0].args.get("range") != "next_weekend":
        raise SystemExit("planner should extract next_weekend range for agenda-next-weekend alias")
    if planner.plan("events this weekend").actions[0].args.get("range") != "weekend":
        raise SystemExit("planner should extract weekend range for events-this-weekend alias")
    if planner.plan("what do i have this weekend").actions[0].args.get("range") != "weekend":
        raise SystemExit("planner should extract weekend range for what-do-i-have alias")
    if planner.plan("what's on my calendar this weekend").actions[0].args.get("range") != "weekend":
        raise SystemExit("planner should extract weekend range for on-my-calendar alias")
    if planner.plan("what's my schedule this weekend").actions[0].args.get("range") != "weekend":
        raise SystemExit("planner should extract weekend range for schedule-this-weekend alias")
    if planner.plan("my schedule next weekend").actions[0].args.get("range") != "next_weekend":
        raise SystemExit("planner should extract next_weekend range for bare schedule alias")
    if planner.plan("open calendar this weekend").actions[0].args.get("range") != "weekend":
        raise SystemExit("planner should extract weekend range for open-calendar alias")
    if planner.plan("open calendar next weekend").actions[0].args.get("range") != "next_weekend":
        raise SystemExit("planner should extract next_weekend range for open-calendar alias")
    if planner.plan("calendar this month").actions[0].args.get("range") != "month":
        raise SystemExit("planner should extract month range for calendar-this-month alias")
    if planner.plan("calendar month").actions[0].args.get("range") != "month":
        raise SystemExit("planner should extract month range for bare calendar-month alias")
    if planner.plan("calendar next month").actions[0].args.get("range") != "next_month":
        raise SystemExit("planner should extract next_month range for calendar-next-month alias")
    if planner.plan("schedule this month").actions[0].args.get("range") != "month":
        raise SystemExit("planner should extract month range for schedule-this-month alias")
    if planner.plan("agenda next month").actions[0].args.get("range") != "next_month":
        raise SystemExit("planner should extract next_month range for agenda-next-month alias")
    if planner.plan("events this month").actions[0].args.get("range") != "month":
        raise SystemExit("planner should extract month range for events-this-month alias")
    if planner.plan("what do i have this month").actions[0].args.get("range") != "month":
        raise SystemExit("planner should extract month range for what-do-i-have alias")
    if planner.plan("what's on my calendar next month").actions[0].args.get("range") != "next_month":
        raise SystemExit("planner should extract next_month range for on-my-calendar alias")
    if planner.plan("what's my schedule this month").actions[0].args.get("range") != "month":
        raise SystemExit("planner should extract month range for schedule-this-month alias")
    if planner.plan("my schedule next month").actions[0].args.get("range") != "next_month":
        raise SystemExit("planner should extract next_month range for bare schedule alias")
    if planner.plan("open calendar this month").actions[0].args.get("range") != "month":
        raise SystemExit("planner should extract month range for open-calendar alias")
    if planner.plan("open calendar next month").actions[0].args.get("range") != "next_month":
        raise SystemExit("planner should extract next_month range for open-calendar alias")
    if planner.plan("calendar this year").actions[0].args.get("range") != "year":
        raise SystemExit("planner should extract year range for calendar-this-year alias")
    if planner.plan("calendar year").actions[0].args.get("range") != "year":
        raise SystemExit("planner should extract year range for bare calendar-year alias")
    if planner.plan("calendar next year").actions[0].args.get("range") != "next_year":
        raise SystemExit("planner should extract next_year range for calendar-next-year alias")
    if planner.plan("schedule this year").actions[0].args.get("range") != "year":
        raise SystemExit("planner should extract year range for schedule-this-year alias")
    if planner.plan("agenda next year").actions[0].args.get("range") != "next_year":
        raise SystemExit("planner should extract next_year range for agenda-next-year alias")
    if planner.plan("events this year").actions[0].args.get("range") != "year":
        raise SystemExit("planner should extract year range for events-this-year alias")
    if planner.plan("what do i have this year").actions[0].args.get("range") != "year":
        raise SystemExit("planner should extract year range for what-do-i-have alias")
    if planner.plan("what's on my calendar next year").actions[0].args.get("range") != "next_year":
        raise SystemExit("planner should extract next_year range for on-my-calendar alias")
    if planner.plan("what's my schedule this year").actions[0].args.get("range") != "year":
        raise SystemExit("planner should extract year range for schedule-this-year alias")
    if planner.plan("my schedule next year").actions[0].args.get("range") != "next_year":
        raise SystemExit("planner should extract next_year range for bare schedule alias")
    if planner.plan("open calendar this year").actions[0].args.get("range") != "year":
        raise SystemExit("planner should extract year range for open-calendar alias")
    if planner.plan("open calendar next year").actions[0].args.get("range") != "next_year":
        raise SystemExit("planner should extract next_year range for open-calendar alias")
    if planner.plan("calendar monday").actions[0].args.get("range") != "monday":
        raise SystemExit("planner should extract monday range for bare calendar alias")
    if planner.plan("calendar next monday").actions[0].args.get("range") != "next_monday":
        raise SystemExit("planner should extract next_monday range for bare calendar alias")
    if planner.plan("schedule monday").actions[0].args.get("range") != "monday":
        raise SystemExit("planner should extract monday range for terse schedule alias")
    if planner.plan("agenda monday").actions[0].args.get("range") != "monday":
        raise SystemExit("planner should extract monday range for terse agenda alias")
    if planner.plan("events monday").actions[0].args.get("range") != "monday":
        raise SystemExit("planner should extract monday range for events alias")
    if planner.plan("events next monday").actions[0].args.get("range") != "next_monday":
        raise SystemExit("planner should extract next_monday range for events alias")
    if planner.plan("what do i have monday").actions[0].args.get("range") != "monday":
        raise SystemExit("planner should extract monday range for what-do-i-have alias")
    if planner.plan("what do i have next monday").actions[0].args.get("range") != "next_monday":
        raise SystemExit("planner should extract next_monday range for what-do-i-have alias")
    if planner.plan("what's on my calendar monday").actions[0].args.get("range") != "monday":
        raise SystemExit("planner should extract monday range for on-my-calendar alias")
    # Real gap found live 2026-07-10: "events from 1969" (a historical-year
    # question, not a personal-calendar request) still fell into the calendar
    # events route (the bare word "events" is a broad trigger), and the
    # generic "capture any bare number as max_results" logic then misread the
    # year "1969" as a request to list up to 1969 of the user's own calendar
    # events -- the wrong interpretation entirely, though bounded since
    # list_events itself clamps large max_results down to 50. Fixed by
    # excluding 4-digit year-like numbers (19xx/20xx) from the max_results
    # capture; legitimate small limits ("show 5 events") are unaffected.
    if planner.plan("events from 1969").actions[0].args.get("max_results") is not None:
        raise SystemExit("planner should not treat a year as an events max_results limit")
    if planner.plan("show 5 events").actions[0].args.get("max_results") != 5:
        raise SystemExit("planner should still capture a genuine small events limit")
    if planner.plan("next 3 events").actions[0].args.get("max_results") != 3:
        raise SystemExit("planner should still capture a genuine small events limit for 'next N events'")
    if planner.plan("what's on my calendar next monday").actions[0].args.get("range") != "next_monday":
        raise SystemExit("planner should extract next_monday range for on-my-calendar alias")
    if planner.plan("what's my schedule monday").actions[0].args.get("range") != "monday":
        raise SystemExit("planner should extract monday range for abbreviated schedule alias")
    if planner.plan("what's my schedule next monday").actions[0].args.get("range") != "next_monday":
        raise SystemExit("planner should extract next_monday range for abbreviated schedule alias")
    if planner.plan("my schedule monday").actions[0].args.get("range") != "monday":
        raise SystemExit("planner should extract monday range for bare schedule alias")
    if planner.plan("my schedule next monday").actions[0].args.get("range") != "next_monday":
        raise SystemExit("planner should extract next_monday range for bare schedule alias")
    if planner.plan("open calendar monday").actions[0].args.get("range") != "monday":
        raise SystemExit("planner should extract monday range for open-calendar weekday alias")
    if planner.plan("open calendar next monday").actions[0].args.get("range") != "next_monday":
        raise SystemExit("planner should extract next_monday range for open-calendar weekday alias")
    if planner.plan("calendar tomorrow morning").actions[0].args.get("range") != "tomorrow_morning":
        raise SystemExit("planner should extract tomorrow_morning range for calendar alias")
    if planner.plan("calendar tomorrow afternoon").actions[0].args.get("range") != "tomorrow_afternoon":
        raise SystemExit("planner should extract tomorrow_afternoon range for calendar alias")
    if planner.plan("calendar tonight").actions[0].args.get("range") != "today_evening":
        raise SystemExit("planner should extract today_evening range for calendar-tonight alias")
    if planner.plan("calendar tomorrow night").actions[0].args.get("range") != "tomorrow_evening":
        raise SystemExit("planner should extract tomorrow_evening range for calendar-tomorrow-night alias")
    if planner.plan("what do i have tomorrow night").actions[0].args.get("range") != "tomorrow_evening":
        raise SystemExit("planner should extract tomorrow_evening range for what-do-i-have night alias")
    if planner.plan("what's on my calendar tomorrow night").actions[0].args.get("range") != "tomorrow_evening":
        raise SystemExit("planner should extract tomorrow_evening range for on-my-calendar night alias")
    if planner.plan("my schedule tomorrow night").actions[0].args.get("range") != "tomorrow_evening":
        raise SystemExit("planner should extract tomorrow_evening range for bare schedule night alias")
    if planner.plan("what do i have tomorrow morning").actions[0].args.get("range") != "tomorrow_morning":
        raise SystemExit("planner should extract tomorrow_morning range for what-do-i-have alias")
    if planner.plan("what's on my calendar tomorrow morning").actions[0].args.get("range") != "tomorrow_morning":
        raise SystemExit("planner should extract tomorrow_morning range for on-my-calendar alias")
    if planner.plan("what's my schedule monday morning").actions[0].args.get("range") != "monday_morning":
        raise SystemExit("planner should extract monday_morning range for schedule alias")
    if planner.plan("my schedule monday afternoon").actions[0].args.get("range") != "monday_afternoon":
        raise SystemExit("planner should extract monday_afternoon range for bare schedule alias")
    if planner.plan("calendar monday morning").actions[0].args.get("range") != "monday_morning":
        raise SystemExit("planner should extract monday_morning range for bare calendar alias")
    if planner.plan("calendar monday night").actions[0].args.get("range") != "monday_evening":
        raise SystemExit("planner should extract monday_evening range for bare calendar night alias")
    if planner.plan("events monday afternoon").actions[0].args.get("range") != "monday_afternoon":
        raise SystemExit("planner should extract monday_afternoon range for events alias")
    if planner.plan("events monday night").actions[0].args.get("range") != "monday_evening":
        raise SystemExit("planner should extract monday_evening range for events night alias")
    if planner.plan("open calendar next monday morning").actions[0].args.get("range") != "next_monday_morning":
        raise SystemExit("planner should extract next_monday_morning range for open-calendar alias")
    if planner.plan("open calendar next monday night").actions[0].args.get("range") != "next_monday_evening":
        raise SystemExit("planner should extract next_monday_evening range for open-calendar night alias")
    if planner.plan("what's my agenda friday evening").actions[0].args.get("range") != "friday_evening":
        raise SystemExit("planner should extract friday_evening range for agenda alias")
    if planner.plan("calendar this morning").actions[0].args.get("range") != "today_morning":
        raise SystemExit("planner should extract today_morning range for calendar this-morning alias")
    if planner.plan("calendar this afternoon").actions[0].args.get("range") != "today_afternoon":
        raise SystemExit("planner should extract today_afternoon range for calendar this-afternoon alias")
    if planner.plan("calendar this evening").actions[0].args.get("range") != "today_evening":
        raise SystemExit("planner should extract today_evening range for calendar this-evening alias")
    if planner.plan("schedule this morning").actions[0].args.get("range") != "today_morning":
        raise SystemExit("planner should extract today_morning range for schedule alias")
    if planner.plan("agenda this afternoon").actions[0].args.get("range") != "today_afternoon":
        raise SystemExit("planner should extract today_afternoon range for agenda alias")
    if planner.plan("events this evening").actions[0].args.get("range") != "today_evening":
        raise SystemExit("planner should extract today_evening range for events alias")
    if planner.plan("what do i have this morning").actions[0].args.get("range") != "today_morning":
        raise SystemExit("planner should extract today_morning range for what-do-i-have alias")
    if planner.plan("what do i have this afternoon").actions[0].args.get("range") != "today_afternoon":
        raise SystemExit("planner should extract today_afternoon range for what-do-i-have alias")
    if planner.plan("what's on my calendar this evening").actions[0].args.get("range") != "today_evening":
        raise SystemExit("planner should extract today_evening range for on-my-calendar alias")
    if planner.plan("what's my schedule this morning").actions[0].args.get("range") != "today_morning":
        raise SystemExit("planner should extract today_morning range for schedule alias")
    if planner.plan("my schedule this afternoon").actions[0].args.get("range") != "today_afternoon":
        raise SystemExit("planner should extract today_afternoon range for bare schedule alias")
    if planner.plan("open calendar this morning").actions[0].args.get("range") != "today_morning":
        raise SystemExit("planner should extract today_morning range for open-calendar alias")
    if planner.plan("open calendar this afternoon").actions[0].args.get("range") != "today_afternoon":
        raise SystemExit("planner should extract today_afternoon range for open-calendar alias")
    if planner.plan("open calendar this evening").actions[0].args.get("range") != "today_evening":
        raise SystemExit("planner should extract today_evening range for open-calendar alias")
    if planner.plan("check my calendar tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for check-my-calendar alias")
    if planner.plan("open calendar tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for open-calendar alias")
    if planner.plan("open calendar tomorrow night").actions[0].args.get("range") != "tomorrow_evening":
        raise SystemExit("planner should extract tomorrow_evening range for open-calendar night alias")
    if planner.plan("what's on my agenda tomorrow").actions[0].args.get("range") != "tomorrow":
        raise SystemExit("planner should extract tomorrow range for agenda alias")
    if planner.plan("open calendar this week").actions[0].args.get("range") != "week":
        raise SystemExit("planner should extract week range for open-calendar alias")
    if planner.plan("view my calendar this week").actions[0].args.get("range") != "week":
        raise SystemExit("planner should extract week range for view-my-calendar alias")
    created = planner.plan("schedule dentist tomorrow at 3pm")
    if [a.tool_name for a in created.actions] != ["create_event"]:
        raise SystemExit(f"planner should preserve schedule-create intent: {created}")
    app_open = planner.plan("open calendar app")
    if [a.tool_name for a in app_open.actions] != ["open_application"]:
        raise SystemExit(f"planner should preserve explicit calendar-app opens: {app_open}")
    daily_brief = planner.plan("what does my day look like")
    if [a.tool_name for a in daily_brief.actions] != ["daily_briefing"]:
        raise SystemExit(f"planner should preserve undated day-look as daily brief: {daily_brief}")
    dictionary = planner.plan("what does ephemeral mean")
    if [a.tool_name for a in dictionary.actions] != ["define"]:
        raise SystemExit(f"planner should preserve real dictionary lookup: {dictionary}")
    schedule_dictionary = planner.plan("what does schedule mean")
    if [a.tool_name for a in schedule_dictionary.actions] != ["define"]:
        raise SystemExit(f"planner should preserve schedule dictionary lookup: {schedule_dictionary}")
    year_time = planner.plan("what year is it")
    if [a.tool_name for a in year_time.actions] == ["list_events"]:
        raise SystemExit(f"planner should preserve ordinary year/time questions: {year_time}")


def test_planner_routes_calendar_availability_aliases() -> None:
    planner = RuleBasedPlanner()
    for text in [
        "free today",
        "free tomorrow afternoon",
        "free tomorrow night",
        "busy tonight",
        "available this week",
        "free next week",
        "busy next week",
        "available next week",
        "open next week",
        "free this weekend",
        "busy next weekend",
        "available this weekend",
        "open this weekend",
        "open next weekend",
        "free this month",
        "busy next month",
        "available this month",
        "open this month",
        "open next month",
        "free this year",
        "busy next year",
        "available this year",
        "open this year",
        "open next year",
        "free monday",
        "busy friday morning",
        "available tuesday evening",
        "busy monday night",
        "open monday",
        "open next monday",
        "free next monday afternoon",
        "open tomorrow",
        "open tomorrow afternoon",
        "open this week",
        "availability please",
        "free time today please",
        "do I have any free time today",
        "do I have free time tomorrow",
        "any free time today",
        "any openings today",
        "do I have an opening today",
        "do I have a slot today",
        "free at 3pm",
        "available at 3pm",
        "open at 3pm",
        "booked at 3pm",
        "busy at 3pm",
        "can I meet at 3pm",
        "can I schedule something at 3pm",
        # Real gap found live 2026-07-09: "check my availability tomorrow" and
        # "what's my availability tomorrow" fell through to chat -- the bare
        # "availability"/"free time" alternative was anchored (^...$) with no
        # "check (my)"/"what's my" lead-in, even though "am I free tomorrow"
        # (a sibling phrasing) already worked.
        "check my availability tomorrow",
        "check availability tomorrow",
        "what's my availability tomorrow",
        "what is my availability tomorrow",
    ]:
        plan = planner.plan(text)
        if [a.tool_name for a in plan.actions] != ["check_availability"]:
            raise SystemExit(f"planner missed availability alias {text!r}: {plan}")
        if plan.actions[0].args.get("text") != text:
            raise SystemExit(f"planner should preserve availability text for {text!r}: {plan.actions[0].args}")
    korean_availability = {
        "내일 시간 있어?": "am I free tomorrow",
        "내일 바빠?": "am I busy tomorrow",
        "오늘 오후에 시간 돼?": "am I free today afternoon",
        "오늘 오전에 시간 있어요?": "am I free today morning",
        "내일 저녁에 바빠요?": "am I busy tomorrow evening",
        "내일 밤에 바쁘니?": "am I busy tomorrow evening",
    }
    for text, canonical_text in korean_availability.items():
        plan = planner.plan(text)
        if [action.tool_name for action in plan.actions] != ["check_availability"]:
            raise SystemExit(f"planner missed Korean availability route {text!r}: {plan}")
        if plan.actions[0].args != {"text": canonical_text}:
            raise SystemExit(
                f"planner canonicalized Korean availability incorrectly {text!r}: {plan.actions[0].args}"
            )
    for text in (
        "시간 있어?",
        "내일 3시에 시간 돼?",
        "내일 오후 2시부터 4시까지 시간 돼?",
        "내일 오후에 회의 잡아줘",
        "오늘 일정 추가해줘",
        "내일 약속 만들어줘",
        "오늘 오후 시간 비워줘",
        "내일 회의 취소해줘",
        "내일 시간 있어? 있으면 회의 잡아줘",
    ):
        plan = planner.plan(text)
        if any(action.tool_name == "check_availability" for action in plan.actions):
            raise SystemExit(f"Korean availability route stole unsupported/write request {text!r}: {plan}")
    if planner.plan("free trial").actions:
        raise SystemExit("planner should not steal non-calendar 'free trial'")
    app_open = planner.plan("open Calendar")
    if [a.tool_name for a in app_open.actions] != ["open_application"]:
        raise SystemExit(f"planner should preserve explicit app open requests: {app_open}")
    created = planner.plan("schedule dentist tomorrow at 3pm")
    if [a.tool_name for a in created.actions] != ["create_event"]:
        raise SystemExit(f"planner should preserve explicit calendar event creation: {created}")


def test_exact_tomorrow_0900_availability_query() -> None:
    from jarvis_v2.tools.calendar_connector import _availability_window

    text = "am I free tomorrow at 9:00am?"
    plan = RuleBasedPlanner().plan(text)
    if [action.tool_name for action in plan.actions] != ["check_availability"]:
        raise SystemExit(f"exact availability query routed incorrectly: {plan}")
    if plan.actions[0].args != {"text": text}:
        raise SystemExit(f"exact availability query text was not preserved: {plan.actions[0].args}")

    fixed_now = datetime.fromisoformat("2026-08-04T21:52:00+09:00")
    start, end, label = _availability_window(text, now=fixed_now)
    if datetime.fromisoformat(start) != datetime.fromisoformat("2026-08-05T09:00:00+09:00"):
        raise SystemExit(f"exact availability query should start tomorrow at 09:00 local: {start}")
    if datetime.fromisoformat(end) != datetime.fromisoformat("2026-08-05T10:00:00+09:00"):
        raise SystemExit(f"exact availability query should end tomorrow at 10:00 local: {end}")
    if label != "tomorrow at 09:00":
        raise SystemExit(f"exact availability query label drifted: {label!r}")


def test_availability_window_parses_parts() -> None:
    from jarvis_v2.tools.calendar_connector import _availability_window
    fixed_now = datetime.fromisoformat("2026-06-17T10:30:00+09:00")
    for q, expected in [("am I free tomorrow afternoon", "tomorrow afternoon"),
                        ("am I free tomorrow morning", "tomorrow morning"),
                        ("am I free tomorrow night", "tomorrow evening"),
                        ("am I busy monday night", "Monday evening"),
                        ("am I busy tonight", "today evening")]:
        if _availability_window(q)[2] != expected:
            raise SystemExit(f"window label wrong for {q!r}: {_availability_window(q)[2]}")
    night_start, night_end, night_label = _availability_window("am I free tomorrow night", now=fixed_now)
    night_start_dt = datetime.fromisoformat(night_start)
    night_end_dt = datetime.fromisoformat(night_end)
    if night_label != "tomorrow evening" or night_start_dt.hour != 18 or night_end_dt.hour != 22:
        raise SystemExit(f"tomorrow-night availability should use evening bounds: {(night_start, night_end, night_label)}")
    canonical_windows = {
        "am I free tomorrow": (8, 22, "tomorrow"),
        "am I free today morning": (8, 12, "today morning"),
        "am I free today afternoon": (12, 18, "today afternoon"),
        "am I busy tomorrow evening": (18, 22, "tomorrow evening"),
    }
    for canonical_text, (start_hour, end_hour, expected_label) in canonical_windows.items():
        start, end, label = _availability_window(canonical_text, now=fixed_now)
        if (
            datetime.fromisoformat(start).hour != start_hour
            or datetime.fromisoformat(end).hour != end_hour
            or label != expected_label
        ):
            raise SystemExit(
                f"canonical Korean availability window drifted for {canonical_text!r}: {(start, end, label)}"
            )
    this_start, this_end, this_label = _availability_window("am I available this week", now=fixed_now)
    if this_label != "this week" or datetime.fromisoformat(this_start) != fixed_now:
        raise SystemExit(f"this-week availability should start at now with label this week: {(this_start, this_end, this_label)}")
    if datetime.fromisoformat(this_end) - datetime.fromisoformat(this_start) != timedelta(days=7):
        raise SystemExit(f"this-week availability should span seven days: {(this_start, this_end, this_label)}")
    next_start, next_end, next_label = _availability_window("am I free next week", now=fixed_now)
    next_start_dt = datetime.fromisoformat(next_start)
    next_end_dt = datetime.fromisoformat(next_end)
    if next_label != "next week" or next_start_dt.weekday() != 0 or next_start_dt.hour != 0 or next_start_dt.minute != 0:
        raise SystemExit(f"next-week availability should start at local Monday midnight: {(next_start, next_end, next_label)}")
    if next_end_dt - next_start_dt != timedelta(days=7):
        raise SystemExit(f"next-week availability should span seven days: {(next_start, next_end, next_label)}")
    weekend_start, weekend_end, weekend_label = _availability_window("am I free this weekend", now=fixed_now)
    weekend_start_dt = datetime.fromisoformat(weekend_start)
    weekend_end_dt = datetime.fromisoformat(weekend_end)
    if weekend_label != "this weekend" or weekend_start_dt.weekday() != 5 or weekend_start_dt.hour != 0:
        raise SystemExit(f"this-weekend availability should start at local Saturday midnight: {(weekend_start, weekend_end, weekend_label)}")
    if weekend_end_dt - weekend_start_dt != timedelta(days=2):
        raise SystemExit(f"this-weekend availability should span two days: {(weekend_start, weekend_end, weekend_label)}")
    next_weekend_start, next_weekend_end, next_weekend_label = _availability_window("am I free next weekend", now=fixed_now)
    next_weekend_start_dt = datetime.fromisoformat(next_weekend_start)
    next_weekend_end_dt = datetime.fromisoformat(next_weekend_end)
    if next_weekend_label != "next weekend" or next_weekend_start_dt.weekday() != 5 or next_weekend_start_dt.hour != 0:
        raise SystemExit(f"next-weekend availability should start at local Saturday midnight: {(next_weekend_start, next_weekend_end, next_weekend_label)}")
    if next_weekend_end_dt - next_weekend_start_dt != timedelta(days=2):
        raise SystemExit(f"next-weekend availability should span two days: {(next_weekend_start, next_weekend_end, next_weekend_label)}")
    if next_weekend_start_dt - weekend_start_dt != timedelta(days=7):
        raise SystemExit(f"next weekend should start one week after this weekend: {(weekend_start, next_weekend_start)}")
    month_start, month_end, month_label = _availability_window("am I free this month", now=fixed_now)
    month_start_dt = datetime.fromisoformat(month_start)
    month_end_dt = datetime.fromisoformat(month_end)
    if month_label != "this month" or month_start_dt != fixed_now:
        raise SystemExit(f"this-month availability should start at now: {(month_start, month_end, month_label)}")
    if month_end_dt.day != 1 or month_end_dt.hour != 0 or month_end_dt.month != 7:
        raise SystemExit(f"this-month availability should end at next local month boundary: {(month_start, month_end, month_label)}")
    next_month_start, next_month_end, next_month_label = _availability_window("am I free next month", now=fixed_now)
    next_month_start_dt = datetime.fromisoformat(next_month_start)
    next_month_end_dt = datetime.fromisoformat(next_month_end)
    if next_month_label != "next month" or next_month_start_dt.day != 1 or next_month_start_dt.month != 7 or next_month_start_dt.hour != 0:
        raise SystemExit(f"next-month availability should start at next local month boundary: {(next_month_start, next_month_end, next_month_label)}")
    if next_month_end_dt.day != 1 or next_month_end_dt.month != 8 or next_month_end_dt.hour != 0:
        raise SystemExit(f"next-month availability should end at following local month boundary: {(next_month_start, next_month_end, next_month_label)}")
    year_start, year_end, year_label = _availability_window("am I free this year", now=fixed_now)
    year_start_dt = datetime.fromisoformat(year_start)
    year_end_dt = datetime.fromisoformat(year_end)
    if year_label != "this year" or year_start_dt != fixed_now:
        raise SystemExit(f"this-year availability should start at now: {(year_start, year_end, year_label)}")
    if year_end_dt.month != 1 or year_end_dt.day != 1 or year_end_dt.year != 2027 or year_end_dt.hour != 0:
        raise SystemExit(f"this-year availability should end at next local year boundary: {(year_start, year_end, year_label)}")
    next_year_start, next_year_end, next_year_label = _availability_window("am I free next year", now=fixed_now)
    next_year_start_dt = datetime.fromisoformat(next_year_start)
    next_year_end_dt = datetime.fromisoformat(next_year_end)
    if next_year_label != "next year" or next_year_start_dt.year != 2027 or next_year_start_dt.month != 1 or next_year_start_dt.day != 1 or next_year_start_dt.hour != 0:
        raise SystemExit(f"next-year availability should start at next local year boundary: {(next_year_start, next_year_end, next_year_label)}")
    if next_year_end_dt.year != 2028 or next_year_end_dt.month != 1 or next_year_end_dt.day != 1 or next_year_end_dt.hour != 0:
        raise SystemExit(f"next-year availability should end at following local year boundary: {(next_year_start, next_year_end, next_year_label)}")
    weekday_start, weekday_end, weekday_label = _availability_window("am I free next monday afternoon", now=fixed_now)
    if weekday_label != "next Monday afternoon":
        raise SystemExit(f"next weekday daypart label wrong: {(weekday_start, weekday_end, weekday_label)}")
    weekday_start_dt = datetime.fromisoformat(weekday_start)
    weekday_end_dt = datetime.fromisoformat(weekday_end)
    if weekday_start_dt.weekday() != 0 or weekday_start_dt.hour != 12 or weekday_end_dt.hour != 18:
        raise SystemExit(f"next weekday afternoon bounds wrong: {(weekday_start, weekday_end, weekday_label)}")
    same_day_now = datetime.fromisoformat("2026-06-15T10:30:00+09:00")
    same_day_start, _, same_day_label = _availability_window("am I free next monday", now=same_day_now)
    same_day_start_dt = datetime.fromisoformat(same_day_start)
    if same_day_label != "next Monday" or same_day_start_dt.date().isoformat() != "2026-06-22":
        raise SystemExit(f"explicit next weekday should skip same-day Monday: {(same_day_start, same_day_label)}")


def test_check_availability_free_and_busy() -> None:
    free = _tools({"events": []})["check_availability"].handler({"text": "am I free today"})
    if not free.ok or "free" not in free.output.lower():
        raise SystemExit(f"empty calendar should be free: {free.output}")
    free_handoff = assert_calendar_availability_handoff(free.metadata, "check_availability free", status="free")
    if free_handoff.get("free") is not True or free_handoff.get("conflicts") != []:
        raise SystemExit(f"free availability handoff should carry an empty free result: {free_handoff}")
    if not free_handoff.get("start_iso") or not free_handoff.get("end_iso"):
        raise SystemExit(f"free availability handoff should preserve parsed window: {free_handoff}")
    busy_events = [{"summary": "Standup", "start": {"dateTime": "2026-06-15T09:00:00+09:00"}}]
    busy = _tools({"events": busy_events})["check_availability"].handler({"text": "am I free today"})
    if not busy.ok or "Standup" not in busy.output:
        raise SystemExit(f"busy calendar should list conflicts: {busy.output}")
    busy_handoff = assert_calendar_availability_handoff(busy.metadata, "check_availability busy", status="busy")
    if busy_handoff.get("free") is not False or busy_handoff.get("conflict_count") != 1:
        raise SystemExit(f"busy availability handoff should carry one conflict: {busy_handoff}")
    if busy_handoff["conflicts"][0].get("summary") != "Standup":
        raise SystemExit(f"busy availability handoff should preserve conflict summary: {busy_handoff}")


def test_planner_routes_explicit_event_update() -> None:
    planner = RuleBasedPlanner()
    plan = planner.plan("move event evt123 to June 20 at 2pm")
    if [a.tool_name for a in plan.actions] != ["update_event"]:
        raise SystemExit(f"planner should route explicit event move to update_event: {plan}")
    args = plan.actions[0].args
    if args.get("event_id") != "evt123" or "-06-20T14:00" not in str(args.get("start", "")):
        raise SystemExit(f"planner did not extract event id and start time: {args}")


def test_planner_numeric_date_update_keeps_default_duration() -> None:
    planner = RuleBasedPlanner()
    plan = planner.plan("move event evt123 to 6/20 at 3pm")
    if [a.tool_name for a in plan.actions] != ["update_event"]:
        raise SystemExit(f"planner should route numeric date update to update_event: {plan}")
    args = plan.actions[0].args
    if (
        args.get("event_id") != "evt123"
        or "-06-20T15:00" not in str(args.get("start", ""))
        or "-06-20T16:00" not in str(args.get("end", ""))
    ):
        raise SystemExit(f"numeric date update should not treat 'to 6/20' as an end time: {args}")


def test_planner_routes_explicit_event_rename() -> None:
    planner = RuleBasedPlanner()
    plan = planner.plan("rename event evt123 to Lunch with Sam")
    if [a.tool_name for a in plan.actions] != ["update_event"]:
        raise SystemExit(f"planner should route explicit event rename to update_event: {plan}")
    args = plan.actions[0].args
    if args.get("event_id") != "evt123" or args.get("title") != "Lunch with Sam":
        raise SystemExit(f"planner did not extract event id and title: {args}")


def test_planner_refuses_fuzzy_event_update() -> None:
    planner = RuleBasedPlanner()
    plan = planner.plan("move my 3pm to 4pm")
    if [a.tool_name for a in plan.actions] != ["respond"]:
        raise SystemExit(f"planner should clarify fuzzy calendar edits without routing a side effect: {plan}")
    if "event id" not in plan.actions[0].args.get("text", "").lower():
        raise SystemExit(f"planner clarification should ask for an event id: {plan.actions[0].args}")


def test_planner_routes_all_day_calendar_create() -> None:
    planner = RuleBasedPlanner()
    for text, expected_title in [
        ("add PTO all day tomorrow", "PTO"),
        ("book PTO all day tomorrow", "PTO"),
        ("book an appointment all day tomorrow", "Event"),
    ]:
        plan = planner.plan(text)
        if [a.tool_name for a in plan.actions] != ["create_event"]:
            raise SystemExit(f"planner should route all-day create to create_event: {text!r} -> {plan}")
        args = plan.actions[0].args
        if args.get("title") != expected_title or "T" in str(args.get("start", "")) or "T" in str(args.get("end", "")):
            raise SystemExit(f"all-day planner route should use date-only create args: {text!r} -> {args}")


def test_planner_routes_numeric_date_calendar_create() -> None:
    planner = RuleBasedPlanner()
    plan = planner.plan("schedule dentist 6/20 at 3pm")
    if [a.tool_name for a in plan.actions] != ["create_event"]:
        raise SystemExit(f"planner should route numeric date create to create_event: {plan}")
    args = plan.actions[0].args
    if args.get("title") != "dentist" or "-06-20T15:00" not in str(args.get("start", "")):
        raise SystemExit(f"planner should extract clean numeric date create args: {args}")


def test_planner_dashed_numeric_date_with_year_not_time_range() -> None:
    planner = RuleBasedPlanner()
    plan = planner.plan("schedule dentist 06-20-2026 at 3pm")
    if [a.tool_name for a in plan.actions] != ["create_event"]:
        raise SystemExit(f"planner should route dashed numeric date create to create_event: {plan}")
    args = plan.actions[0].args
    if (
        args.get("title") != "dentist"
        or not str(args.get("start", "")).startswith("2026-06-20T15:00")
        or not str(args.get("end", "")).startswith("2026-06-20T16:00")
    ):
        raise SystemExit(f"dashed numeric date should not become a long 06-20 time range: {args}")


def test_planner_routes_inferred_meridiem_range() -> None:
    planner = RuleBasedPlanner()
    plan = planner.plan("schedule dentist tomorrow 2 to 4pm")
    if [a.tool_name for a in plan.actions] != ["create_event"]:
        raise SystemExit(f"planner should route inferred-meridiem range to create_event: {plan}")
    args = plan.actions[0].args
    if (
        args.get("title") != "dentist"
        or not str(args.get("start", "")).endswith("T14:00:00+09:00")
        or not str(args.get("end", "")).endswith("T16:00:00+09:00")
    ):
        raise SystemExit(f"planner should extract clean inferred-meridiem range args: {args}")


def test_google_credential_paths_read_environment_at_call_time() -> None:
    old_creds = os.environ.get("JARVIS_GOOGLE_CREDS")
    old_token = os.environ.get("JARVIS_GOOGLE_TOKEN")
    temp = Path(tempfile.mkdtemp())
    creds_path = temp / "missing-creds.json"
    token_path = temp / "missing-token.json"
    module_names = [
        "google",
        "google.oauth2",
        "google.oauth2.credentials",
        "google.auth",
        "google.auth.transport",
        "google.auth.transport.requests",
        "google_auth_oauthlib",
        "google_auth_oauthlib.flow",
        "googleapiclient",
        "googleapiclient.discovery",
    ]
    old_modules = {name: sys.modules.get(name) for name in module_names}
    try:
        os.environ["JARVIS_GOOGLE_CREDS"] = str(creds_path)
        os.environ["JARVIS_GOOGLE_TOKEN"] = str(token_path)

        google = types.ModuleType("google")
        oauth2 = types.ModuleType("google.oauth2")
        credentials_mod = types.ModuleType("google.oauth2.credentials")
        auth = types.ModuleType("google.auth")
        transport = types.ModuleType("google.auth.transport")
        requests_mod = types.ModuleType("google.auth.transport.requests")
        flow_pkg = types.ModuleType("google_auth_oauthlib")
        flow_mod = types.ModuleType("google_auth_oauthlib.flow")
        api_pkg = types.ModuleType("googleapiclient")
        discovery_mod = types.ModuleType("googleapiclient.discovery")

        class Credentials:
            valid = False
            expired = False
            refresh_token = None

            @classmethod
            def from_authorized_user_file(cls, path, scopes):
                return cls()

        class Request:
            pass

        class InstalledAppFlow:
            @classmethod
            def from_client_secrets_file(cls, path, scopes):
                raise SystemExit(f"credential file should not be read when missing: {path}")

        credentials_mod.Credentials = Credentials
        requests_mod.Request = Request
        flow_mod.InstalledAppFlow = InstalledAppFlow
        discovery_mod.build = lambda *args, **kwargs: None
        sys.modules.update(
            {
                "google": google,
                "google.oauth2": oauth2,
                "google.oauth2.credentials": credentials_mod,
                "google.auth": auth,
                "google.auth.transport": transport,
                "google.auth.transport.requests": requests_mod,
                "google_auth_oauthlib": flow_pkg,
                "google_auth_oauthlib.flow": flow_mod,
                "googleapiclient": api_pkg,
                "googleapiclient.discovery": discovery_mod,
            }
        )

        try:
            _ORIGINAL_GET_SERVICE()
        except RuntimeError as exc:
            if "full-access token not found" not in str(exc):
                raise SystemExit(f"missing mutation token did not fail closed: {exc}")
        else:
            raise SystemExit("_get_service unexpectedly succeeded with missing mutation token")
    finally:
        for name, module in old_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if old_creds is None:
            os.environ.pop("JARVIS_GOOGLE_CREDS", None)
        else:
            os.environ["JARVIS_GOOGLE_CREDS"] = old_creds
        if old_token is None:
            os.environ.pop("JARVIS_GOOGLE_TOKEN", None)
        else:
            os.environ["JARVIS_GOOGLE_TOKEN"] = old_token


def test_google_credential_paths_fall_back_on_blank_environment() -> None:
    old_creds = os.environ.get("JARVIS_GOOGLE_CREDS")
    old_token = os.environ.get("JARVIS_GOOGLE_TOKEN")
    old_readonly_token = os.environ.get("JARVIS_GOOGLE_READONLY_TOKEN")
    try:
        os.environ["JARVIS_GOOGLE_CREDS"] = "   "
        os.environ["JARVIS_GOOGLE_TOKEN"] = "   "
        os.environ["JARVIS_GOOGLE_READONLY_TOKEN"] = "   "
        if cc._creds_file() != Path.home() / ".jarvis_v3" / "google_credentials.json":
            raise SystemExit(f"blank Google creds env should use default path: {cc._creds_file()}")
        if cc._token_file() != Path.home() / ".jarvis_v3" / "google_token.json":
            raise SystemExit(f"blank Google token env should use default path: {cc._token_file()}")
        if cc._readonly_token_file() != Path.home() / ".jarvis_v3" / "google_calendar_readonly_token.json":
            raise SystemExit(f"blank Google read-only token env should use default path: {cc._readonly_token_file()}")
    finally:
        if old_creds is None:
            os.environ.pop("JARVIS_GOOGLE_CREDS", None)
        else:
            os.environ["JARVIS_GOOGLE_CREDS"] = old_creds
        if old_token is None:
            os.environ.pop("JARVIS_GOOGLE_TOKEN", None)
        else:
            os.environ["JARVIS_GOOGLE_TOKEN"] = old_token
        if old_readonly_token is None:
            os.environ.pop("JARVIS_GOOGLE_READONLY_TOKEN", None)
        else:
            os.environ["JARVIS_GOOGLE_READONLY_TOKEN"] = old_readonly_token


def main() -> None:
    test_list_calendars_formats_output()
    test_calendar_read_outputs_redact_path_shaped_text()
    test_calendar_read_failures_are_clean()
    test_create_event_requires_title_and_start()
    test_calendar_mutations_reject_path_shaped_fields_before_service_setup()
    test_create_event_inserts()
    test_create_all_day_event_uses_date_fields()
    test_create_event_failure_preserves_attempt_metadata()
    test_update_event_requires_id_and_changes()
    test_delete_event_requires_id_before_service_setup()
    test_update_event_patches_timed_event()
    test_update_all_day_event_uses_date_fields()
    test_calendar_mutation_failures_preserve_attempt_metadata()
    test_list_events_empty()
    test_list_events_sanitizes_bad_max_results()
    test_list_events_range_labels()
    test_absolute_date_ranges_and_invalid_refusal()
    test_planner_routes_calendar_event_list_aliases()
    test_planner_routes_calendar_availability_aliases()
    test_exact_tomorrow_0900_availability_query()
    test_availability_window_parses_parts()
    test_check_availability_free_and_busy()
    test_planner_routes_explicit_event_update()
    test_planner_numeric_date_update_keeps_default_duration()
    test_planner_routes_explicit_event_rename()
    test_planner_refuses_fuzzy_event_update()
    test_planner_routes_all_day_calendar_create()
    test_planner_routes_numeric_date_calendar_create()
    test_planner_dashed_numeric_date_with_year_not_time_range()
    test_planner_routes_inferred_meridiem_range()
    test_google_credential_paths_read_environment_at_call_time()
    test_google_credential_paths_fall_back_on_blank_environment()
    print("Calendar connector smoke passed")


if __name__ == "__main__":
    main()
