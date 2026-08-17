"""Smoke tests for owner-Telegram reminders/timers (no network)."""

from __future__ import annotations

import os
import tempfile
import time
import json
from datetime import datetime
from pathlib import Path

from jarvis_v2.agent.failure_guidance import (
    LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
)
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.automations import reminders as rem
from jarvis_v2.automations import telegram_control as tc
from jarvis_v2.config import load_config
from jarvis_v2.scripts.public_release_candidate import structural_public_candidate_profile
from jarvis_v2.tools import apple_reminders as apple_reminders_tool
from jarvis_v2.tools.reminder_tools import _metadata_bool, _reminder_handoff_metadata, make_reminder_tools


OWNER_CHAT_ID = "555001"
REPO_ROOT = Path(__file__).resolve().parents[2]
IS_PUBLIC_CANDIDATE = structural_public_candidate_profile(REPO_ROOT) is not None
FIRST_REMINDER_ID = "11111111-1111-4111-8111-111111111111"
SECOND_REMINDER_ID = "22222222-2222-4222-8222-222222222222"
AMBIGUOUS_FIRST_ID = "abcdef12-1111-4111-8111-111111111111"
AMBIGUOUS_SECOND_ID = "abcdef12-2222-4222-8222-222222222222"
SENDING_REMINDER_ID = "feedface-1111-4111-8111-111111111111"


def _isolate() -> None:
    rem.REMINDERS_FILE = Path(tempfile.mkdtemp()) / "reminders.json"


def _tool():
    return {t.name: t for t in make_reminder_tools(load_config())}["set_reminder"]


def _assert_no_local_path(value: object, label: str) -> None:
    text = str(value)
    for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
        if fragment in text:
            raise SystemExit(f"{label} leaked a local path: {text}")


def _fixture_reminder(
    reminder_id: str,
    message: str,
    due: float,
    *,
    state: str = "pending",
    chat_id: str = OWNER_CHAT_ID,
) -> dict:
    row = {
        "id": reminder_id,
        "due": due,
        "message": message,
        "chat_id": chat_id,
        "transport": "telegram",
        "state": state,
        "attempt_count": 0,
    }
    if state == "sending":
        row.update(
            {
                "attempt_count": 1,
                "token": "synthetic-delivery-token",
                "sending_at": time.time(),
            }
        )
    return row


def _seed_fixed_reminders(rows: list[dict]) -> None:
    _isolate()
    rem._reminders_file().write_text(json.dumps(rows), encoding="utf-8")


def _stored_reminders_by_id() -> dict[str, dict]:
    rows, _ = rem._read_locked(rem._reminders_file())
    return {str(row.get("id")): row for row in rows}


def _assert_no_message_leak(value: object, messages: list[str], label: str) -> None:
    serialized = json.dumps(value, sort_keys=True, default=str)
    for message in messages:
        if message in serialized:
            raise SystemExit(f"{label} leaked raw reminder content: {serialized}")


def _assert_boundary_flags(boundaries: dict, label: str, *, reads_personal_data: bool = False, writes_files: bool = False) -> None:
    expected = {
        "calls_model": False,
        "calls_external_service": False,
        "executes_tools": False,
        "reads_personal_data": reads_personal_data,
        "reads_private_data": False,
        "executes_side_effect": writes_files,
        "external_side_effect": False,
        "writes_files": writes_files,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    for key, value in expected.items():
        if boundaries.get(key) is not value:
            raise SystemExit(f"{label} boundary {key} should be {value}: {boundaries}")


def _assert_reminder_exact_metadata_bool() -> None:
    for value in ("true", "false", "yes", "0", 1, 0, [], ["value"], None, object()):
        if _metadata_bool(value) is not False:
            raise SystemExit(f"reminder metadata bool should fail closed for {value!r}")
    if _metadata_bool(True) is not True:
        raise SystemExit("reminder metadata bool should preserve exact True")
    if _metadata_bool(False) is not False:
        raise SystemExit("reminder metadata bool should preserve exact False")
    if _metadata_bool("false", default=True) is not True:
        raise SystemExit("reminder metadata bool should honor the explicit default")


def _assert_reminder_malformed_handoff_flags() -> None:
    handoff = {
        "source": "set_reminder",
        "state_changed": "false",
        "changed": [],
        "content_in_handoff": "true",
        "next_commands": ["list reminders"],
        "boundaries": {"read_only": True},
    }
    metadata = _reminder_handoff_metadata("set_reminder_handoff", handoff)
    if metadata.get("state_changed") is not False or metadata.get("set_reminder_state_changed") is not False:
        raise SystemExit(f"malformed reminder state_changed should fail closed: {metadata}")
    if metadata.get("content_in_handoff") is not False or metadata.get("set_reminder_content_in_handoff") is not False:
        raise SystemExit(f"malformed reminder content_in_handoff should fail closed: {metadata}")


def test_reminder_docs_do_not_overclaim_phone_delivery() -> None:
    description = _tool().description
    readme = Path("README.md").read_text()
    reminders_doc = rem.__doc__ or ""
    apple_reminders_doc = apple_reminders_tool.__doc__ or ""
    def normalized(text: str) -> str:
        return " ".join(text.split())

    expected_parts = [
        (description, "owner Telegram channel"),
        (readme, "owner-Telegram reminders/timers"),
        (readme, "Telegram API acceptance"),
        (readme, "phone-display delivery remains live-user confirmation"),
        (reminders_doc, "Telegram API acceptance is the automatable proof"),
        (reminders_doc, "phone display remains"),
        (apple_reminders_doc, "Telegram API acceptance is the automatable proof"),
        (apple_reminders_doc, "phone-display delivery remains live-user confirmation"),
    ]
    for text, expected in expected_parts:
        if expected not in normalized(text):
            raise SystemExit(f"reminder docs missed trust-boundary wording: {expected}")
    stale_parts = [
        "delivered to your phone via Telegram",
        "Telegram-delivered reminders/timers",
        "sends them to the owner's phone",
        "delivered over Telegram",
    ]
    combined = normalized("\n".join([description, readme, reminders_doc, apple_reminders_doc]))
    for stale in stale_parts:
        if stale in combined:
            raise SystemExit(f"reminder docs still overclaim phone delivery: {stale}")


def test_finish_plan_documents_set_reminder_without_erasing_macos_create_path() -> None:
    plan_path = REPO_ROOT / "FINISH_PLAN_AUGUST.md"
    personal_py = (REPO_ROOT / "jarvis_v2/tools/personal.py").read_text()

    implementation_evidence = [
        "def create_reminder(args:",
        'tell application "Reminders" to make new reminder',
    ]
    for expected in implementation_evidence:
        if expected not in personal_py:
            raise SystemExit(f"personal.py reminder implementation evidence missing: {expected}")

    if IS_PUBLIC_CANDIDATE:
        if plan_path.exists():
            raise SystemExit(
                "public candidate retained private operational reference: FINISH_PLAN_AUGUST.md"
            )
        return
    plan = plan_path.read_text()

    required_parts = [
        "`set_reminder` live-proven",
        "`create_reminder` in `personal.py`",
        "AppleScript create path is real",
        "cannot satisfy it",
        "approval-gated macOS `create_reminder` path",
    ]
    for expected in required_parts:
        if expected not in plan:
            raise SystemExit(f"finish plan missed reminder path truthfulness wording: {expected}")

    stale_parts = [
        "no such creation path exists",
        "no AppleScript bridge exists",
        "no AppleScript-based reminder CREATE path",
    ]
    for stale in stale_parts:
        if stale in plan:
            raise SystemExit(f"finish plan still erases macOS reminder create path: {stale}")


def _assert_handoff_safe_commands(metadata: dict, handoff: dict, handoff_key: str, label: str) -> None:
    prefix = handoff_key.removesuffix("_handoff")
    raw_next = handoff.get("next_commands") or []
    if isinstance(raw_next, dict):
        expected = [str(value) for value in raw_next.values() if str(value or "").strip()]
    else:
        expected = [str(value) for value in raw_next if str(value or "").strip()]
    first = expected[0] if expected else ""
    for key, value in [
        ("handoff_ready", True),
        ("next_safe_command", first),
        ("next_safe_commands", expected),
        ("next_safe_command_count", len(expected)),
    ]:
        if handoff.get(key) != value:
            raise SystemExit(f"{label} handoff {key} mismatch: {handoff}")
    for key, value in [
        (f"{handoff_key}_ready", True),
        (f"{prefix}_handoff_ready", True),
        ("next_safe_command", first),
        ("next_safe_commands", expected),
        ("next_safe_command_count", len(expected)),
        (f"{prefix}_next_safe_command", first),
        (f"{prefix}_next_safe_commands", expected),
        (f"{prefix}_next_safe_command_count", len(expected)),
        (f"{prefix}_ready_for_operator", True),
        (f"{prefix}_state_changed", handoff.get("state_changed")),
        (f"{prefix}_changed", handoff.get("changed")),
        (f"{prefix}_content_in_handoff", handoff.get("content_in_handoff")),
        (f"{prefix}_authorizes_execution", False),
        (f"{prefix}_authorizes_completion_claim", False),
        (f"{prefix}_approval_granted", False),
    ]:
        if metadata.get(key) != value:
            raise SystemExit(f"{label} metadata {key} mismatch: {metadata}")


def _assert_set_reminder_handoff(metadata: dict, label: str, *, status: str, writes_files: bool = False) -> dict:
    handoff = metadata.get("set_reminder_handoff")
    if not isinstance(handoff, dict) or metadata.get("set_reminder_handoff_ready") is not True:
        raise SystemExit(f"{label} missed set reminder handoff readiness: {metadata}")
    if handoff.get("source") != "set_reminder" or handoff.get("status") != status:
        raise SystemExit(f"{label} set reminder handoff status/source wrong: {handoff}")
    expected_changed = ["reminder_store_write"] if writes_files else []
    for key, value in [
        ("ready_for_operator", True),
        ("state_changed", writes_files),
        ("changed", expected_changed),
        ("content_in_handoff", bool(handoff.get("message"))),
        ("authorizes_execution", False),
        ("authorizes_completion_claim", False),
        ("approval_granted", False),
    ]:
        if handoff.get(key) != value or metadata.get(key) != value:
            raise SystemExit(f"{label} set reminder handoff {key} parity wrong: {handoff} / {metadata}")
    if handoff.get("stored") is not writes_files:
        raise SystemExit(f"{label} set reminder handoff stored/write parity failed: {metadata}")
    if handoff.get("storage_error") is not bool(metadata.get("storage_error")):
        raise SystemExit(f"{label} set reminder handoff storage parity failed: {metadata}")
    if handoff.get("content_is_preview_only") is not True:
        raise SystemExit(f"{label} set reminder handoff should mark preview content: {handoff}")
    _assert_handoff_safe_commands(metadata, handoff, "set_reminder_handoff", label)
    _assert_no_local_path(handoff, f"{label} set reminder handoff")
    _assert_boundary_flags(handoff.get("boundaries") or {}, label, writes_files=writes_files)
    return handoff


def _assert_list_reminders_handoff(metadata: dict, label: str, *, count: int) -> dict:
    handoff = metadata.get("list_reminders_handoff")
    if not isinstance(handoff, dict) or metadata.get("list_reminders_handoff_ready") is not True:
        raise SystemExit(f"{label} missed list reminders handoff readiness: {metadata}")
    if handoff.get("source") != "list_reminders" or handoff.get("status") != "ok":
        raise SystemExit(f"{label} list reminders handoff status/source wrong: {handoff}")
    if handoff.get("count") != count or handoff.get("count") != metadata.get("count"):
        raise SystemExit(f"{label} list reminders handoff count parity failed: {metadata}")
    content_present = count > 0
    for key, value in [
        ("ready_for_operator", True),
        ("state_changed", False),
        ("changed", []),
        ("content_in_handoff", content_present),
        ("authorizes_execution", False),
        ("authorizes_completion_claim", False),
        ("approval_granted", False),
    ]:
        if handoff.get(key) != value or metadata.get(key) != value:
            raise SystemExit(f"{label} list reminders handoff {key} parity wrong: {handoff} / {metadata}")
    if len(handoff.get("reminders") or []) != count:
        raise SystemExit(f"{label} list reminders handoff rows wrong: {handoff}")
    if handoff.get("content_is_preview_only") is not True:
        raise SystemExit(f"{label} list reminders handoff should mark preview content: {handoff}")
    health = handoff.get("delivery_health")
    if (
        not isinstance(health, dict)
        or handoff.get("delivery_health_status") != health.get("status")
        or handoff.get("delivery_health_available") is not (health.get("source_available") is True)
        or health.get("content_in_receipt") is not False
    ):
        raise SystemExit(f"{label} list reminders delivery health was not content-free/truthful: {handoff}")
    _assert_no_local_path(health, f"{label} reminder delivery health")
    _assert_handoff_safe_commands(metadata, handoff, "list_reminders_handoff", label)
    _assert_no_local_path(handoff, f"{label} list reminders handoff")
    _assert_boundary_flags(handoff.get("boundaries") or {}, label, reads_personal_data=True)
    return handoff


def _assert_list_reminders_unavailable(metadata: dict, label: str, *, reason: str) -> dict:
    handoff = metadata.get("list_reminders_handoff")
    if not isinstance(handoff, dict) or metadata.get("list_reminders_handoff_ready") is not True:
        raise SystemExit(f"{label} missed list reminders handoff readiness: {metadata}")
    for key, expected in (
        ("status", "unavailable"),
        ("reason", reason),
        ("source_available", False),
        ("result_complete", False),
        ("count_known", False),
        ("storage_error", True),
        ("state_changed", False),
        ("authorizes_execution", False),
        ("authorizes_completion_claim", False),
        ("approval_granted", False),
    ):
        if handoff.get(key) != expected:
            raise SystemExit(f"{label} unavailable handoff {key} wrong: {handoff}")
    if "count" in handoff or "count" in metadata:
        raise SystemExit(f"{label} unavailable read should not claim a reminder count: {metadata}")
    if handoff.get("reminders") != [] or metadata.get("write_suppressed") is not True:
        raise SystemExit(f"{label} unavailable read should be content-free and write-suppressed: {metadata}")
    health = handoff.get("delivery_health")
    if (
        not isinstance(health, dict)
        or handoff.get("delivery_health_status") != health.get("status")
        or health.get("content_in_receipt") is not False
    ):
        raise SystemExit(f"{label} unavailable list lost content-free delivery health: {handoff}")
    _assert_no_local_path(health, f"{label} unavailable reminder delivery health")
    _assert_handoff_safe_commands(metadata, handoff, "list_reminders_handoff", label)
    _assert_no_local_path(handoff, f"{label} list reminders unavailable handoff")
    return handoff


def _assert_local_productivity_read_recovery(result, label: str) -> None:
    action = LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION
    if result.ok or action not in result.output:
        raise SystemExit(f"{label} hid the canonical local recovery action: {result}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": action,
        "commands": ["setup check"],
    }:
        raise SystemExit(f"{label} local recovery declaration drifted: {result.metadata}")
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
            raise SystemExit(f"{label} local recovery field {key} drifted: {result.metadata}")


def _assert_cancel_reminders_handoff(metadata: dict, label: str, *, status: str, count: int, writes_files: bool = False) -> dict:
    handoff = metadata.get("cancel_reminders_handoff")
    if not isinstance(handoff, dict) or metadata.get("cancel_reminders_handoff_ready") is not True:
        raise SystemExit(f"{label} missed cancel reminders handoff readiness: {metadata}")
    if handoff.get("source") != "cancel_reminders" or handoff.get("status") != status:
        raise SystemExit(f"{label} cancel reminders handoff status/source wrong: {handoff}")
    if handoff.get("count") != count or handoff.get("count") != metadata.get("count"):
        raise SystemExit(f"{label} cancel reminders handoff count parity failed: {metadata}")
    expected_changed = ["reminder_store_write"] if writes_files else []
    content_present = count > 0
    for key, value in [
        ("ready_for_operator", True),
        ("state_changed", writes_files),
        ("changed", expected_changed),
        ("content_in_handoff", content_present),
        ("authorizes_execution", False),
        ("authorizes_completion_claim", False),
        ("approval_granted", False),
    ]:
        if handoff.get(key) != value or metadata.get(key) != value:
            raise SystemExit(f"{label} cancel reminders handoff {key} parity wrong: {handoff} / {metadata}")
    if len(handoff.get("cancelled_reminders") or []) != count:
        raise SystemExit(f"{label} cancel reminders handoff rows wrong: {handoff}")
    if handoff.get("storage_error") is not bool(metadata.get("storage_error")):
        raise SystemExit(f"{label} cancel reminders storage parity failed: {metadata}")
    if handoff.get("content_is_preview_only") is not True:
        raise SystemExit(f"{label} cancel reminders handoff should mark preview content: {handoff}")
    _assert_handoff_safe_commands(metadata, handoff, "cancel_reminders_handoff", label)
    _assert_no_local_path(handoff, f"{label} cancel reminders handoff")
    _assert_boundary_flags(handoff.get("boundaries") or {}, label, reads_personal_data=True, writes_files=writes_files)
    return handoff


def _assert_cancel_selector_refusal(result, label: str, *, private_messages: list[str]) -> dict:
    if result.ok:
        raise SystemExit(f"{label} should refuse without cancellation: {result}")
    for key in ("writes_files", "executes_side_effect", "state_changed"):
        if result.metadata.get(key):
            raise SystemExit(f"{label} claimed a state change at {key}: {result.metadata}")
    handoff = result.metadata.get("cancel_reminders_handoff")
    if not isinstance(handoff, dict) or result.metadata.get("cancel_reminders_handoff_ready") is not True:
        raise SystemExit(f"{label} missed cancel reminder handoff readiness: {result.metadata}")
    if handoff.get("count") != 0 or handoff.get("cancelled_reminders"):
        raise SystemExit(f"{label} claimed cancelled reminder rows: {handoff}")
    if handoff.get("state_changed") is not False or handoff.get("changed") != []:
        raise SystemExit(f"{label} handoff claimed a state change: {handoff}")
    _assert_boundary_flags(handoff.get("boundaries") or {}, label, reads_personal_data=True)
    _assert_no_message_leak(result.output, private_messages, f"{label} output")
    _assert_no_message_leak(result.metadata, private_messages, f"{label} metadata")
    return handoff


def _assert_location_reminder_draft_handoff(metadata: dict, label: str, *, status: str) -> dict:
    handoff = metadata.get("location_reminder_draft_handoff")
    if not isinstance(handoff, dict) or metadata.get("location_reminder_draft_handoff_ready") is not True:
        raise SystemExit(f"{label} missed location reminder draft handoff readiness: {metadata}")
    if handoff.get("source") != "location_reminder_draft" or handoff.get("status") != status:
        raise SystemExit(f"{label} location reminder draft status/source wrong: {handoff}")
    for key, value in [
        ("ready_for_operator", True),
        ("state_changed", False),
        ("changed", []),
        ("authorizes_execution", False),
        ("authorizes_completion_claim", False),
        ("approval_granted", False),
        ("supported_execution_now", False),
        ("requires_review_before_creation", True),
    ]:
        if handoff.get(key) != value or metadata.get(key) != value:
            raise SystemExit(f"{label} location draft handoff {key} parity wrong: {handoff} / {metadata}")
    if handoff.get("content_is_preview_only") is not True:
        raise SystemExit(f"{label} location draft handoff should mark preview content: {handoff}")
    _assert_handoff_safe_commands(metadata, handoff, "location_reminder_draft_handoff", label)
    _assert_no_local_path(handoff, f"{label} location draft handoff")
    _assert_boundary_flags(handoff.get("boundaries") or {}, label)
    return handoff


def test_parse_durations_and_time() -> None:
    now = datetime(2026, 6, 15, 16, 0).astimezone()
    d, m = rem.parse_when("remind me in 30 minutes to stretch", now)
    if d.strftime("%H:%M") != "16:30" or m != "stretch":
        raise SystemExit(f"duration parse wrong: {d} {m}")
    d, m = rem.parse_when("set a timer for 10 minutes", now)
    if d.strftime("%H:%M") != "16:10":
        raise SystemExit(f"timer parse wrong: {d}")
    d, m = rem.parse_when("timer 10 minutes", now)
    if d.strftime("%H:%M") != "16:10" or m != "⏰ Reminder!":
        raise SystemExit(f"terse timer parse wrong: {d} {m}")
    d, m = rem.parse_when("timer 10m", now)
    if d.strftime("%H:%M") != "16:10" or m != "⏰ Reminder!":
        raise SystemExit(f"compact minute timer parse wrong: {d} {m}")
    d, m = rem.parse_when("10m timer", now)
    if d.strftime("%H:%M") != "16:10" or m != "⏰ Reminder!":
        raise SystemExit(f"compact reverse timer parse wrong: {d} {m}")
    d, m = rem.parse_when("remind me in an hour to stretch", now)
    if d.strftime("%H:%M") != "17:00" or m != "stretch":
        raise SystemExit(f"an-hour duration parse wrong: {d} {m}")
    d, m = rem.parse_when("remind me in 2h to stretch", now)
    if d.strftime("%H:%M") != "18:00" or m != "stretch":
        raise SystemExit(f"compact hour reminder parse wrong: {d} {m}")
    d, m = rem.parse_when("remind me in a minute to stretch", now)
    if d.strftime("%H:%M") != "16:01" or m != "stretch":
        raise SystemExit(f"a-minute duration parse wrong: {d} {m}")
    d, m = rem.parse_when("timer for half an hour", now)
    if d.strftime("%H:%M") != "16:30" or m != "⏰ Reminder!":
        raise SystemExit(f"half-hour timer parse wrong: {d} {m}")
    d, m = rem.parse_when("remind me to call Sam in 2 days", now)
    if d.strftime("%Y-%m-%d %H:%M") != "2026-06-17 16:00" or m != "call Sam":
        raise SystemExit(f"day duration parse wrong: {d} {m}")
    d, m = rem.parse_when("2 day timer", now)
    if d.strftime("%Y-%m-%d %H:%M") != "2026-06-17 16:00":
        raise SystemExit(f"day timer parse wrong: {d}")
    d, m = rem.parse_when("remind me to call Sam in 2 weeks", now)
    if d.strftime("%Y-%m-%d %H:%M") != "2026-06-29 16:00" or m != "call Sam":
        raise SystemExit(f"week duration parse wrong: {d} {m}")
    d, m = rem.parse_when("1 week timer", now)
    if d.strftime("%Y-%m-%d %H:%M") != "2026-06-22 16:00":
        raise SystemExit(f"week timer parse wrong: {d}")
    d, m = rem.parse_when("remind me to call Sam in 2 hrs and 30 mins", now)
    if d.strftime("%Y-%m-%d %H:%M") != "2026-06-15 18:30" or m != "call Sam":
        raise SystemExit(f"compound duration parse wrong: {d} {m}")
    d, m = rem.parse_when("remind me to call Sam next monday at 5pm", now)
    if m != "call Sam":
        raise SystemExit(f"next weekday reminder message should be clean: {m!r}")
    d, m = rem.parse_when("remind me to call Sam on monday at 5pm", now)
    if m != "call Sam":
        raise SystemExit(f"on-weekday reminder message should be clean: {m!r}")
    d, m = rem.parse_when("ping me in 10 minutes to stretch", now)
    if d.strftime("%H:%M") != "16:10" or m != "stretch":
        raise SystemExit(f"ping reminder parse wrong: {d} {m}")
    d, m = rem.parse_when("notify me in 15 minutes to check the oven", now)
    if d.strftime("%H:%M") != "16:15" or m != "check the oven":
        raise SystemExit(f"notify reminder parse wrong: {d} {m}")
    d, m = rem.parse_when("wake me up in 30 minutes", now)
    if d.strftime("%H:%M") != "16:30" or m != "⏰ Reminder!":
        raise SystemExit(f"wake reminder parse wrong: {d} {m}")
    d, m = rem.parse_when("set an alarm for 10 minutes", now)
    if d.strftime("%H:%M") != "16:10" or m != "⏰ Reminder!":
        raise SystemExit(f"alarm reminder parse wrong: {d} {m}")
    d, m = rem.parse_when("alarm for 7am", now)
    if d.strftime("%H:%M") != "07:00" or m != "⏰ Reminder!":
        raise SystemExit(f"bare alarm cleanup wrong: {d} {m}")
    if rem.parse_when("hello there", now) is not None:
        raise SystemExit("non-reminder text should not parse")


def test_huge_duration_does_not_crash() -> None:
    now = datetime(2026, 6, 15, 16, 0).astimezone()
    huge = "9" * 80
    if rem.parse_when(f"remind me in {huge} hours to stretch", now) is not None:
        raise SystemExit("overflowing reminder duration should not parse")
    out = _tool().handler({"text": f"set a timer for {huge} minutes"})
    if out.ok or "couldn't work out the time" not in out.output.lower():
        raise SystemExit(f"overflowing timer should ask for a clearer time: {out.output}")


def test_tool_requires_owner_and_stores() -> None:
    _isolate()
    tool = _tool()
    old = os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
    try:
        out = tool.handler({"text": "remind me in 5 minutes to drink water"})
        if out.ok or "JARVIS_OWNER_TELEGRAM" not in out.output:
            raise SystemExit(f"reminder without owner should fail: {out.output}")
        if out.metadata.get("writes_files") is not False:
            raise SystemExit(f"reminder owner refusal should not claim a file write: {out.metadata}")
        missing_owner_handoff = _assert_set_reminder_handoff(out.metadata, "missing owner reminder", status="refused")
        if missing_owner_handoff.get("reason") != "missing_owner_telegram" or missing_owner_handoff.get("owner_configured") is not False:
            raise SystemExit(f"missing owner handoff should preserve refusal reason: {missing_owner_handoff}")
        os.environ["JARVIS_OWNER_TELEGRAM"] = "555001"
        out = tool.handler({"text": "remind me in 5 minutes to drink water"})
        if not out.ok or "drink water" not in out.output:
            raise SystemExit(f"reminder not scheduled: {out.output}")
        if out.metadata.get("writes_files") is not True:
            raise SystemExit(f"scheduled reminder should report the local file write: {out.metadata}")
        scheduled_handoff = _assert_set_reminder_handoff(out.metadata, "scheduled reminder", status="scheduled", writes_files=True)
        if scheduled_handoff.get("message") != "drink water" or scheduled_handoff.get("owner_configured") is not True:
            raise SystemExit(f"scheduled reminder handoff should preserve sanitized message and owner state: {scheduled_handoff}")
        if len(rem._load()) != 1 or rem._load()[0]["chat_id"] != "555001":
            raise SystemExit(f"reminder not stored correctly: {rem._load()}")
    finally:
        if old is not None:
            os.environ["JARVIS_OWNER_TELEGRAM"] = old
        else:
            os.environ.pop("JARVIS_OWNER_TELEGRAM", None)


def test_recurring_reminders_are_explicitly_unsupported_and_inert() -> None:
    _isolate()
    tool = _tool()
    old_owner = os.environ.get("JARVIS_OWNER_TELEGRAM")
    try:
        os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        for text in [
            "remind me every 10 minutes to stretch",
            "remind me to stretch every 10 minutes",
            "remind me every day at 9am to stretch",
            "set a recurring reminder to stand up every hour",
        ]:
            out = tool.handler({"text": text})
            if out.ok or out.metadata.get("reason") != "recurring_reminder_unsupported":
                raise SystemExit(f"recurring reminder should be an explicit unsupported refusal: {out.output} {out.metadata}")
            if out.metadata.get("writes_files"):
                raise SystemExit(f"recurring reminder refusal should not claim file writes: {out.metadata}")
            handoff = _assert_set_reminder_handoff(out.metadata, "recurring reminder refusal", status="refused")
            if handoff.get("reason") != "recurring_reminder_unsupported":
                raise SystemExit(f"recurring reminder handoff should preserve reason: {handoff}")
            if rem._load():
                raise SystemExit(f"recurring reminder refusal should not store reminders: {rem._load()}")
    finally:
        if old_owner is None:
            os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        else:
            os.environ["JARVIS_OWNER_TELEGRAM"] = old_owner


def test_path_shaped_reminder_text_is_rejected_and_redacted() -> None:
    _isolate()
    tool = _tool()
    old_owner = os.environ.get("JARVIS_OWNER_TELEGRAM")
    try:
        os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        for text in [
            "remind me in 5 minutes to /\x55sers/example/private/reminder.txt",
            "remind me in 5 minutes to /var/folders/zc/jarvis/reminder.txt",
            "remind me in 5 minutes to /tmp/jarvis-reminder.txt",
        ]:
            rejected = tool.handler({"text": text})
            if rejected.ok or rejected.metadata.get("reason") != "invalid_reminder_text":
                raise SystemExit(f"path-shaped reminder text should be rejected before owner/storage checks: {rejected.metadata}")
            if rejected.metadata.get("message") != "<local-path>":
                raise SystemExit(f"path-shaped reminder metadata should be redacted: {rejected.metadata}")
            path_handoff = _assert_set_reminder_handoff(rejected.metadata, "path-shaped reminder refusal", status="refused")
            if path_handoff.get("message") != "<local-path>" or path_handoff.get("reason") != "invalid_reminder_text":
                raise SystemExit(f"path-shaped reminder handoff should be redacted and refused: {path_handoff}")
            _assert_no_local_path(rejected.output, "path-shaped reminder refusal output")
            _assert_no_local_path(rejected.metadata, "path-shaped reminder refusal metadata")
            if rem._load():
                raise SystemExit(f"path-shaped reminder should be inert: {rem._load()}")

        if not rem.add_reminder(time.time() + 3600, "/\x55sers/example/private/direct-reminder.txt", "555001"):
            raise SystemExit("direct reminder insert should succeed with sanitized message")
        if not rem.add_reminder(time.time() + 3600, "/var/folders/zc/jarvis/direct-reminder.txt", "555001"):
            raise SystemExit("direct temp-root reminder insert should succeed with sanitized message")
        if not rem.add_reminder(time.time() + 3600, "/tmp/jarvis-direct-reminder.txt", "555001"):
            raise SystemExit("direct /tmp reminder insert should succeed with sanitized message")
        stored = rem._load()
        if [r.get("message") for r in stored] != ["<local-path>", "<local-path>", "<local-path>"]:
            raise SystemExit(f"direct reminder insert should sanitize message text: {stored}")

        rem._reminders_file().write_text(json.dumps([
            {"due": time.time() + 3600, "message": "/private/tmp/legacy-reminder.txt", "chat_id": "555001"},
            {"due": time.time() + 7200, "message": "/var/folders/zc/jarvis/legacy-reminder.txt", "chat_id": "555001"},
            {"due": time.time() + 10800, "message": "/tmp/jarvis-legacy-reminder.txt", "chat_id": "555001"},
        ]), encoding="utf-8")
        os.environ["JARVIS_OWNER_TELEGRAM"] = "555001"
        listed = _all_tools()["list_reminders"].handler({})
        if "<local-path>" not in listed.output:
            raise SystemExit(f"legacy reminder list output should redact local paths: {listed.output}")
        _assert_no_local_path(listed.output, "legacy reminder list output")
    finally:
        if old_owner is None:
            os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        else:
            os.environ["JARVIS_OWNER_TELEGRAM"] = old_owner


def test_tool_reports_storage_failure_without_claiming_write() -> None:
    old_file = rem.REMINDERS_FILE
    old_owner = os.environ.get("JARVIS_OWNER_TELEGRAM")
    blocker = Path(tempfile.mkdtemp()) / "not-a-dir"
    blocker.write_text("block", encoding="utf-8")
    try:
        rem.REMINDERS_FILE = blocker / "reminders.json"
        os.environ["JARVIS_OWNER_TELEGRAM"] = "555001"
        out = _tool().handler({"text": "remind me in 5 minutes to drink water"})
    finally:
        rem.REMINDERS_FILE = old_file
        if old_owner is None:
            os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        else:
            os.environ["JARVIS_OWNER_TELEGRAM"] = old_owner
    if out.ok or "couldn't save" not in out.output:
        raise SystemExit(f"storage failure should be reported cleanly: {out.output}")
    if out.metadata.get("writes_files") or out.metadata.get("storage_error") is not True:
        raise SystemExit(f"storage failure should not claim a write: {out.metadata}")
    storage_handoff = _assert_set_reminder_handoff(out.metadata, "storage failure reminder", status="failed")
    if storage_handoff.get("reason") != "storage_error" or storage_handoff.get("storage_error") is not True:
        raise SystemExit(f"storage failure handoff should preserve failure reason: {storage_handoff}")


def test_tool_reports_post_publication_durability_uncertainty_without_retrying() -> None:
    _isolate()
    old_owner = os.environ.get("JARVIS_OWNER_TELEGRAM")
    original_fsync = rem.os.fsync
    fsync_calls = 0

    def fail_directory_sync(fd: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("synthetic directory sync failure")
        original_fsync(fd)

    try:
        os.environ["JARVIS_OWNER_TELEGRAM"] = "555001"
        rem.os.fsync = fail_directory_sync  # type: ignore[assignment]
        out = _tool().handler({"text": "remind me in 5 minutes to verify uncertain durability"})
    finally:
        rem.os.fsync = original_fsync  # type: ignore[assignment]
        if old_owner is None:
            os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        else:
            os.environ["JARVIS_OWNER_TELEGRAM"] = old_owner

    stored = rem._load()
    if out.ok or len(stored) != 1 or stored[0].get("message") != "verify uncertain durability":
        raise SystemExit(f"post-publication failure lost committed reminder truth: {out} / {stored}")
    for key, expected in {
        "writes_files": True,
        "executes_side_effect": True,
        "storage_error": True,
        "durability_uncertain": True,
        "outcome_known": False,
        "retry_safe": False,
        "authorizes_retry": False,
        "state_changed": True,
    }.items():
        if out.metadata.get(key) is not expected:
            raise SystemExit(f"post-publication reminder truth drifted at {key}: {out.metadata}")
    handoff = _assert_set_reminder_handoff(
        out.metadata,
        "post-publication uncertain reminder",
        status="published_uncertain",
        writes_files=True,
    )
    if (
        handoff.get("reason") != "directory_sync_failed"
        or handoff.get("stored") is not True
        or handoff.get("storage_error") is not True
        or handoff.get("next_safe_command") != "list reminders"
    ):
        raise SystemExit(f"post-publication reminder handoff invites unsafe retry: {handoff}")
    if "do not retry automatically" not in out.output.lower() or "couldn't save" in out.output.lower():
        raise SystemExit(f"post-publication reminder output claimed a definite no-write: {out.output}")

    _isolate()
    fsync_calls = 0
    try:
        rem.os.fsync = fail_directory_sync  # type: ignore[assignment]
        legacy_published = rem.add_reminder(
            time.time() + 300,
            "legacy caller must not retry",
            "555001",
        )
    finally:
        rem.os.fsync = original_fsync  # type: ignore[assignment]
    if legacy_published is not True or len(rem._load()) != 1:
        raise SystemExit("legacy Boolean caller treated a published uncertain reminder as absent")


def test_list_is_read_only_and_cancel_reports_storage_failure() -> None:
    old_file = rem.REMINDERS_FILE
    old_owner = os.environ.get("JARVIS_OWNER_TELEGRAM")
    root = Path(tempfile.mkdtemp())
    missing = root / "missing" / "reminders.json"
    try:
        rem.REMINDERS_FILE = missing
        os.environ["JARVIS_OWNER_TELEGRAM"] = "555001"
        listed = _all_tools()["list_reminders"].handler({})
        if not listed.output.startswith("You have no pending reminders.") or missing.parent.exists():
            raise SystemExit("read-only reminder listing should not create storage or lock files")

        blocker = root / "not-a-dir"
        blocker.write_text("block", encoding="utf-8")
        rem.REMINDERS_FILE = blocker / "reminders.json"
        cancelled = _all_tools()["cancel_reminders"].handler({})
        if cancelled.ok or cancelled.metadata.get("storage_error") is not True:
            raise SystemExit(f"cancel storage failure should be explicit: {cancelled.output} {cancelled.metadata}")
        if cancelled.metadata.get("writes_files"):
            raise SystemExit(f"failed cancel should not claim a file write: {cancelled.metadata}")
        handoff = _assert_cancel_reminders_handoff(
            cancelled.metadata,
            "cancel storage failure",
            status="failed",
            count=0,
        )
        if handoff.get("storage_error") is not True:
            raise SystemExit(f"cancel failure handoff lost storage error: {handoff}")
    finally:
        rem.REMINDERS_FILE = old_file
        if old_owner is None:
            os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        else:
            os.environ["JARVIS_OWNER_TELEGRAM"] = old_owner


def test_pop_due_returns_due_keeps_future() -> None:
    _isolate()
    now = time.time()
    rem.add_reminder(now - 5, "past one", "555001")
    rem.add_reminder(now + 3600, "future one", "555001")
    due = rem.pop_due(now)
    if [d["message"] for d in due] != ["past one"]:
        raise SystemExit(f"pop_due returned wrong items: {due}")
    if [r["message"] for r in rem._load()] != ["future one"]:
        raise SystemExit(f"future reminder was not kept: {rem._load()}")


def test_unreadable_existing_store_fails_closed_without_writes() -> None:
    old_owner = os.environ.get("JARVIS_OWNER_TELEGRAM")
    os.environ["JARVIS_OWNER_TELEGRAM"] = "555001"
    now = time.time()
    cases = [
        ("malformed_json", b"{not-json"),
        ("invalid_encoding", b"\xff\xfe\xfd"),
        ("invalid_schema", json.dumps({"rows": []}).encode("utf-8")),
        (
            "invalid_row",
            json.dumps(
                [
                    {"due": "not-a-number", "message": "bad", "chat_id": "555001"},
                    {"due": now + 3600, "message": "otherwise valid", "chat_id": "555001"},
                ]
            ).encode("utf-8"),
        ),
    ]
    try:
        for reason, payload in cases:
            _isolate()
            reminders_file = rem._reminders_file()
            reminders_file.write_bytes(payload)
            original_bytes = reminders_file.read_bytes()
            original_write = rem._write_locked
            write_calls: list[str] = []

            def forbidden_write(path, items):
                write_calls.append(str(path))
                raise AssertionError("unavailable reminder state reached the writer")

            rem._write_locked = forbidden_write  # type: ignore[assignment]
            try:
                items, available, observed_reason = rem.pending_reminders_status("555001")
                if items or available or observed_reason != reason:
                    raise SystemExit(f"{reason} reminder read truth was wrong: {(items, available, observed_reason)}")
                if rem.add_reminder(now + 60, "new reminder", "555001"):
                    raise SystemExit(f"{reason} reminder state was overwritten by add_reminder")
                if rem._save([{"due": now + 60, "message": "replacement", "chat_id": "555001"}]):
                    raise SystemExit(f"{reason} reminder state was overwritten by _save")
                if rem.claim_due(now, "555001"):
                    raise SystemExit(f"{reason} reminder state produced a delivery claim")
                if rem.begin_delivery("reminder-id", "token"):
                    raise SystemExit(f"{reason} reminder state began delivery")
                if rem.finish_delivery("reminder-id", "token", "accepted", telegram_message_id=1):
                    raise SystemExit(f"{reason} reminder state finalized delivery")
                removed, saved = rem.cancel_pending_items("555001")
                if removed or saved:
                    raise SystemExit(f"{reason} reminder state claimed a successful cancellation: {(removed, saved)}")
                if rem.pop_due(now):
                    raise SystemExit(f"{reason} reminder state popped due rows")

                listed = _all_tools()["list_reminders"].handler({})
                if listed.ok or "can't verify" not in listed.output or "no pending reminders" in listed.output.lower():
                    raise SystemExit(f"{reason} list should be explicitly unavailable: {listed.output}")
                _assert_local_productivity_read_recovery(
                    listed,
                    f"{reason} reminder list",
                )
                _assert_list_reminders_unavailable(listed.metadata, f"{reason} reminder list", reason=reason)

                scheduled = _tool().handler({"text": "remind me in 5 minutes to preserve state"})
                if scheduled.ok or scheduled.metadata.get("storage_error") is not True:
                    raise SystemExit(f"{reason} set reminder should fail without claiming a write: {scheduled}")
                cancelled = _all_tools()["cancel_reminders"].handler({})
                if cancelled.ok or cancelled.metadata.get("write_suppressed") is not True:
                    raise SystemExit(f"{reason} cancel should fail without claiming a write: {cancelled}")
            finally:
                rem._write_locked = original_write  # type: ignore[assignment]

            if write_calls or reminders_file.read_bytes() != original_bytes:
                raise SystemExit(f"{reason} reminder state was not preserved byte-for-byte: {write_calls}")

        _isolate()
        reminders_file = rem._reminders_file()
        reminders_file.write_text("[]", encoding="utf-8")
        original_read_text = Path.read_text

        def failed_read(path, *args, **kwargs):
            if path == reminders_file:
                raise OSError("synthetic read failure with /\x55sers/example/private detail")
            return original_read_text(path, *args, **kwargs)

        Path.read_text = failed_read  # type: ignore[assignment]
        try:
            items, available, reason = rem.pending_reminders_status("555001")
            if items or available or reason != "read_error":
                raise SystemExit(f"read OSError should be bounded and unavailable: {(items, available, reason)}")
            listed = _all_tools()["list_reminders"].handler({})
            _assert_list_reminders_unavailable(listed.metadata, "read-error reminder list", reason="read_error")
            _assert_no_local_path(listed.output, "read-error reminder list output")
            _assert_no_local_path(listed.metadata, "read-error reminder list metadata")
        finally:
            Path.read_text = original_read_text  # type: ignore[assignment]
    finally:
        if old_owner is None:
            os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        else:
            os.environ["JARVIS_OWNER_TELEGRAM"] = old_owner


def test_reminders_file_reads_environment_at_call_time() -> None:
    old_file = rem.REMINDERS_FILE
    old_env = os.environ.get("JARVIS_REMINDERS_FILE")
    temp = Path(tempfile.mkdtemp()) / "env-reminders.json"
    try:
        rem.REMINDERS_FILE = None
        os.environ["JARVIS_REMINDERS_FILE"] = str(temp)
        rem.add_reminder(time.time() + 60, "env reminder", "555001")
        if not temp.exists():
            raise SystemExit("JARVIS_REMINDERS_FILE was not honored at call time")
        if [r["message"] for r in rem._load()] != ["env reminder"]:
            raise SystemExit(f"env reminder file not loaded: {rem._load()}")
    finally:
        rem.REMINDERS_FILE = old_file
        if old_env is None:
            os.environ.pop("JARVIS_REMINDERS_FILE", None)
        else:
            os.environ["JARVIS_REMINDERS_FILE"] = old_env


def test_blank_reminders_file_env_uses_default() -> None:
    old_file = rem.REMINDERS_FILE
    old_env = os.environ.get("JARVIS_REMINDERS_FILE")
    try:
        rem.REMINDERS_FILE = None
        os.environ["JARVIS_REMINDERS_FILE"] = "   "
        expected = Path.home() / ".jarvis_v3" / "telegram_reminders.json"
        if rem._reminders_file() != expected:
            raise SystemExit(f"blank JARVIS_REMINDERS_FILE should use default: {rem._reminders_file()}")
    finally:
        rem.REMINDERS_FILE = old_file
        if old_env is None:
            os.environ.pop("JARVIS_REMINDERS_FILE", None)
        else:
            os.environ["JARVIS_REMINDERS_FILE"] = old_env


def test_bridge_delivers_due_reminder() -> None:
    _isolate()
    old_owner = os.environ.get("JARVIS_OWNER_TELEGRAM")
    os.environ["JARVIS_OWNER_TELEGRAM"] = "555001"
    sent = []
    try:
        rem.add_reminder(time.time() - 1, "stand up", "555001")

        def accepted(cid, text, markup=None):
            sent.append((cid, text))
            return {"ok": True, "result": {"message_id": 101}}

        bridge = tc.TelegramCommandBridge(
            runtime_factory=lambda: None,
            send_func=accepted,
            fetch_func=lambda token, offset, timeout: [],
        )
        bridge._deliver_due_reminders()
    finally:
        if old_owner is None:
            os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        else:
            os.environ["JARVIS_OWNER_TELEGRAM"] = old_owner
    if sent != [("555001", "⏰ stand up")]:
        raise SystemExit(f"due reminder not delivered: {sent}")
    if rem._load():
        raise SystemExit("delivered reminder was not removed from store")


def test_bridge_redacts_legacy_path_shaped_reminder() -> None:
    _isolate()
    old_owner = os.environ.get("JARVIS_OWNER_TELEGRAM")
    os.environ["JARVIS_OWNER_TELEGRAM"] = "555001"
    sent = []
    try:
        rem._reminders_file().write_text(json.dumps([
            {"due": time.time() - 1, "message": "/\x55sers/example/private/bridge-reminder.txt", "chat_id": "555001"},
            {"due": time.time() - 1, "message": "/var/folders/zc/jarvis/bridge-reminder.txt", "chat_id": "555001"},
            {"due": time.time() - 1, "message": "/tmp/jarvis-bridge-reminder.txt", "chat_id": "555001"},
        ]), encoding="utf-8")

        def accepted(cid, text, markup=None):
            sent.append((cid, text))
            return {"ok": True, "result": {"message_id": 100 + len(sent)}}

        bridge = tc.TelegramCommandBridge(
            runtime_factory=lambda: None,
            send_func=accepted,
            fetch_func=lambda token, offset, timeout: [],
        )
        bridge._deliver_due_reminders()
    finally:
        if old_owner is None:
            os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        else:
            os.environ["JARVIS_OWNER_TELEGRAM"] = old_owner
    if sent != [("555001", "⏰ <local-path>"), ("555001", "⏰ <local-path>"), ("555001", "⏰ <local-path>")]:
        raise SystemExit(f"due reminder delivery should redact local paths: {sent}")
    _assert_no_local_path(sent, "due reminder delivery")
    if rem._load():
        raise SystemExit("delivered path-shaped reminder was not removed from store")


def test_planner_routes_reminders() -> None:
    p = RuleBasedPlanner()
    timed_requests = [
        "remind me in 30 minutes to stretch",
        "remind me to call Sam in 10 minutes",
        "remind me to call Sam in 2 days",
        "remind me to call Sam in 2 weeks",
        "remind me to stretch in an hour",
        "remind me to stretch in half an hour",
        "remind me to stretch every 10 minutes",
        "remind me daily to stretch",
        "set a recurring reminder to stand up every hour",
        "remind me to check the oven at 5pm",
        "set a reminder to call Sam at 5pm",
        "set reminder to call Sam at 5pm",
        "create a reminder to stretch tomorrow",
        "add reminder to submit report tomorrow",
        "set a timer for 10 minutes",
        "timer 10 minutes",
        "timer 10m",
        "10m timer",
        "set timer 2h",
        "remind me in 2h to stretch",
        "ping me in 10 minutes to stretch",
        "notify me in 10 minutes to stretch",
        "wake me up in 10 minutes",
        "set an alarm for 10 minutes",
        "alarm in 10 minutes",
        "alarm 7am",
        "alarm at 7am",
        "7am alarm",
        "2 day timer",
        "1 week timer",
        "5 minute timer",
    ]
    for q in timed_requests:
        plan = p.plan(q)
        if [a.tool_name for a in plan.actions] != ["set_reminder"]:
            raise SystemExit(f"planner missed reminder route: {q!r}")
        if plan.actions[0].args.get("text") != q:
            raise SystemExit(f"timed reminder should preserve original text for parser: {plan.actions[0].args}")
    location_requests = [
        "remind me to call mom when I get home",
        "remind me when I get home to call mom",
        "remind me when I leave work to buy milk",
        "remind me when I arrive at the office to check badge",
        "remind me when I get to school to submit form",
    ]
    for q in location_requests:
        plan = p.plan(q)
        if [a.tool_name for a in plan.actions] != ["location_reminder_draft"]:
            raise SystemExit(f"planner missed location reminder draft route: {q!r} -> {plan.actions}")
        if plan.actions[0].args.get("text") != q:
            raise SystemExit(f"location reminder draft should preserve original text: {plan.actions[0].args}")
    if [a.tool_name for a in p.plan("remind me to review approval safety").actions] != ["create_reminder"]:
        raise SystemExit("untimed reminder route should still use create_reminder")
    if [a.tool_name for a in p.plan("set a reminder to review approval safety").actions] != ["create_reminder"]:
        raise SystemExit("untimed set-a-reminder route should still use create_reminder")
    if [a.tool_name for a in p.plan("create a reminder to review approval safety").actions] != ["create_reminder"]:
        raise SystemExit("untimed create-a-reminder route should still use create_reminder")
    # Real gap found live 2026-07-10: "remind me to call mom and then set a
    # timer for 10 minutes" (a compound sentence) swallowed the whole second
    # clause into the reminder title, writing a confusing reminder titled
    # "call mom and then set a timer for 10 minutes" instead of just "call
    # mom" -- and silently dropped the second intent.
    compound_reminder_plan = p.plan("remind me to call mom and then set a timer for 10 minutes")
    if compound_reminder_plan.actions[0].tool_name != "create_reminder" or compound_reminder_plan.actions[0].args.get("title") != "call mom":
        raise SystemExit(f"planner should stop reminder title at a compound-sentence boundary: {compound_reminder_plan.actions}")
    for q in [
        "reminders",
        "reminders please",
        "timers",
        "alarms",
        "list my reminders",
        "show reminders",
        "show alarms",
        "do i have reminders",
        "do i have any alarms",
        "is there a timer running",
        "is there any timer running",
        "timer list",
        "alarm list",
        "reminder list",
        "리마인더",
        "리마인더 목록",
        "리마인더 보여줘",
        "리마인더 보여주세요",
        "리마인더 있어?",
        "내 리마인더",
        "내 리마인더 목록",
        "내 리마인더 보여줘",
        "내 리마인더 뭐야?",
        "알림 목록",
        "알림 보여줘",
        "알림 보여주세요",
        "알림 있어?",
        "내 알림",
        "내 알림 목록",
        "내 알림 보여줘",
        "내 알림 뭐야?",
        "타이머 목록",
        "타이머 보여줘",
        "알람 목록",
        "알람 보여줘",
        "설정된 리마인더",
        "설정된 알림",
    ]:
        if [a.tool_name for a in p.plan(q).actions] != ["list_reminders"]:
            raise SystemExit(f"list reminders route missed: {q!r}")
    for q in [
        "리마인더 설정해줘",
        "알림 추가해줘",
        "내 리마인더 취소해줘",
        "알림 삭제해줘",
        "리마인더 보여줘 그리고 하나 추가해줘",
    ]:
        plan = p.plan(q)
        if any(action.tool_name == "list_reminders" for action in plan.actions):
            raise SystemExit(f"Korean reminder read route captured a mutation or compound request: {q!r} -> {plan.actions}")
    for q in [
        "cancel my reminders",
        "cancel all reminders",
        "clear reminders",
        "delete all reminders",
        "delete every timer",
        "cancel every reminder",
        "stop every alarm",
        "remove reminders",
        "stop timers",
        "stop all timers",
        "stop alarms",
    ]:
        plan = p.plan(q)
        if [a.tool_name for a in plan.actions] != ["cancel_reminders"]:
            raise SystemExit(f"cancel reminders route missed: {q!r}")
        if plan.actions[0].args != {}:
            raise SystemExit(f"broad reminder cancellation should not carry a selector: {q!r} -> {plan}")


def test_planner_routes_uuid_reminder_selectors_and_fails_closed() -> None:
    planner = RuleBasedPlanner()
    for command, reminder_id in (
        ("cancel reminder 11111111", "11111111"),
        ("cancel timer id 11111111", "11111111"),
        ("clear reminder 11111111", "11111111"),
        ("delete reminder 11111111", "11111111"),
        ("remove reminder 11111111", "11111111"),
        ("stop alarm #11111111", "11111111"),
        ("Jarvis, please delete reminder 11111111", "11111111"),
        ("Hey Jarvis, please remove reminder 11111111", "11111111"),
        ("Hey, Jarvis, please delete reminder 11111111", "11111111"),
        ("Hey! Jarvis, please remove reminder 11111111", "11111111"),
        (f"cancel reminder {SECOND_REMINDER_ID}", SECOND_REMINDER_ID),
    ):
        plan = planner.plan(command)
        if [action.tool_name for action in plan.actions] != ["cancel_reminders"]:
            raise SystemExit(f"UUID reminder cancellation route missed: {command!r} -> {plan}")
        if plan.actions[0].args != {"reminder_id": reminder_id}:
            raise SystemExit(f"UUID reminder selector was not preserved exactly: {command!r} -> {plan}")

    for command in (
        "cancel reminder",
        "cancel reminder 1",
        "cancel timer 2",
        "cancel the first reminder",
        "cancel the second alarm",
        "stop timer",
        "stop my timer",
        "stop alarm",
        "cancel reminder tomorrow",
    ):
        plan = planner.plan(command)
        if [action.tool_name for action in plan.actions] != ["respond"]:
            raise SystemExit(f"unsafe singular reminder selector should only clarify: {command!r} -> {plan}")
        clarification = str(plan.actions[0].args.get("text") or "").lower()
        if "list reminders" not in clarification or not any(token in clarification for token in ("id", "prefix", "8")):
            raise SystemExit(f"reminder clarification is not actionable: {command!r} -> {plan}")

    for command in (
        "cancel all but reminder 11111111",
        "cancel reminders except 11111111",
        "cancel some reminders",
    ):
        plan = planner.plan(command)
        if [action.tool_name for action in plan.actions] != ["respond"]:
            raise SystemExit(f"reminder subset language should clarify without mutation: {command!r} -> {plan}")

    for command in (
        "cancel no reminders",
        "tell me how to cancel reminders",
        "can I cancel all reminders?",
        "what happens if I cancel all reminders?",
        "suppose I cancel all reminders",
        "please don't cancel my reminders",
        "I don't want to cancel my reminders",
        "I don't want my reminders cancelled",
    ):
        plan = planner.plan(command)
        if any(action.tool_name == "cancel_reminders" for action in plan.actions):
            raise SystemExit(f"non-imperative reminder language planned cancellation: {command!r} -> {plan}")
        if command != "cancel no reminders" and [action.tool_name for action in plan.actions] != ["respond"]:
            raise SystemExit(f"non-imperative reminder language should stay on a read-only response: {command!r} -> {plan}")

    reminder_mutations = {
        "cancel_reminders",
        "set_reminder",
        "create_reminder",
        "location_reminder_draft",
    }
    for command in (
        "don't cancel my reminders",
        "do not cancel any reminders",
        "never stop my timers",
        "do not cancel reminder 11111111",
        "don't set a timer for 10 minutes",
        "do not remind me in 5 minutes to leave",
    ):
        plan = planner.plan(command)
        planned_mutations = [
            action.tool_name for action in plan.actions if action.tool_name in reminder_mutations
        ]
        if planned_mutations:
            raise SystemExit(f"negated reminder command planned a mutation: {command!r} -> {plan}")

    overlong_withdrawal = "delete reminder 11111111" + "." * 4100 + " no, do not do that"
    overlong_plan = planner.plan(overlong_withdrawal)
    if [action.tool_name for action in overlong_plan.actions] != ["respond"]:
        raise SystemExit(f"truncated reminder cancellation must fail closed: {overlong_plan}")

    for command in (
        "delete file reminder.txt",
        "delete reminder.txt",
        "delete reminder-backup.txt",
        "delete reminder/notes.txt",
        "delete reminder (copy).txt",
        "delete timer.log",
        "delete timer-archive.log",
        "delete the reminder routing test file",
        "email Bob the words cancel reminder 11111111",
    ):
        plan = planner.plan(command)
        if [action.tool_name for action in plan.actions] != ["dispatch_decision_packet"]:
            raise SystemExit(f"non-reminder risky order was stolen by reminder parsing: {command!r} -> {plan}")

    punctuated_bulk_plan = planner.plan("Hi, Jarvis, please clear all reminders")
    if [action.tool_name for action in punctuated_bulk_plan.actions] != ["cancel_reminders"]:
        raise SystemExit(f"punctuated reminder wrapper route missed: {punctuated_bulk_plan}")
    if punctuated_bulk_plan.actions[0].args != {}:
        raise SystemExit(f"punctuated broad reminder cancellation carried a selector: {punctuated_bulk_plan}")


def _all_tools():
    return {t.name: t for t in make_reminder_tools(load_config())}


def test_location_reminder_draft_is_review_only() -> None:
    _isolate()
    tool = _all_tools()["location_reminder_draft"]
    for text, event, location, message in [
        ("remind me to call mom when I get home", "arrive", "home", "call mom"),
        ("remind me when I get home to call mom", "arrive", "home", "call mom"),
        ("remind me when I leave work to buy milk", "leave", "work", "buy milk"),
        ("remind me when I arrive at the office to check badge", "arrive", "the office", "check badge"),
    ]:
        out = tool.handler({"text": text})
        if not out.ok or "Location reminder draft" not in out.output:
            raise SystemExit(f"location reminder draft should succeed without execution: {out.output} {out.metadata}")
        if out.metadata.get("writes_files") or out.metadata.get("executes_side_effect") or out.metadata.get("requires_location_permission") is not True:
            raise SystemExit(f"location reminder draft should be read-only but permission-aware: {out.metadata}")
        handoff = _assert_location_reminder_draft_handoff(out.metadata, "location reminder draft", status="drafted")
        if handoff.get("trigger_event") != event or handoff.get("location_label") != location or handoff.get("message") != message:
            raise SystemExit(f"location reminder draft handoff lost parsed fields: {handoff}")
        if rem._load():
            raise SystemExit(f"location reminder draft should not store reminders: {rem._load()}")

    missing = tool.handler({"text": "remind me when I get home"})
    if missing.ok or missing.metadata.get("reason") != "missing_message":
        raise SystemExit(f"location reminder draft should refuse missing message: {missing.output} {missing.metadata}")
    _assert_location_reminder_draft_handoff(missing.metadata, "missing-message location reminder draft", status="refused")

    path = tool.handler({"text": "remind me to /tmp/secret when I get home"})
    if path.ok or path.metadata.get("reason") != "local_path_input":
        raise SystemExit(f"location reminder draft should refuse path-shaped input: {path.output} {path.metadata}")
    _assert_location_reminder_draft_handoff(path.metadata, "path-shaped location reminder draft", status="refused")
    _assert_no_local_path(path.output, "path-shaped location reminder draft output")
    _assert_no_local_path(path.metadata, "path-shaped location reminder draft metadata")


def test_list_reminders_displays_stable_unique_uuid_prefixes() -> None:
    now = time.time()
    rows = [
        _fixture_reminder(AMBIGUOUS_FIRST_ID, "selector fixture alpha", now + 3600),
        _fixture_reminder(AMBIGUOUS_SECOND_ID, "selector fixture beta", now + 7200),
    ]
    old_owner = os.environ.get("JARVIS_OWNER_TELEGRAM")
    try:
        os.environ["JARVIS_OWNER_TELEGRAM"] = OWNER_CHAT_ID
        _seed_fixed_reminders(rows)
        first = _all_tools()["list_reminders"].handler({})
        second = _all_tools()["list_reminders"].handler({})
    finally:
        if old_owner is None:
            os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        else:
            os.environ["JARVIS_OWNER_TELEGRAM"] = old_owner

    expected_prefixes = ("abcdef121", "abcdef122")
    if not first.ok or first.output != second.output:
        raise SystemExit(f"reminder selector display should be successful and stable: {first} / {second}")
    for prefix in expected_prefixes:
        if first.output.lower().count(prefix) != 1:
            raise SystemExit(f"list reminders missed unique UUID prefix {prefix}: {first.output}")
    handoff = _assert_list_reminders_handoff(first.metadata, "unique selector reminder list", count=2)
    handoff_selectors = {
        str(row.get("reminder_id") or row.get("id_prefix") or row.get("selector") or "").lower()
        for row in handoff.get("reminders", [])
    }
    if handoff_selectors != set(expected_prefixes):
        raise SystemExit(f"list reminder handoff missed stable selectors: {handoff}")


def test_legacy_reminder_selector_is_stable_and_cancellable() -> None:
    old_owner = os.environ.get("JARVIS_OWNER_TELEGRAM")
    try:
        os.environ["JARVIS_OWNER_TELEGRAM"] = OWNER_CHAT_ID
        _seed_fixed_reminders(
            [{"due": 1_900_000_000.0, "message": "legacy selector fixture", "chat_id": OWNER_CHAT_ID}]
        )
        tools = _all_tools()
        first = tools["list_reminders"].handler({})
        second = tools["list_reminders"].handler({})
        first_selector = first.metadata["list_reminders_handoff"]["reminders"][0].get("selector")
        second_selector = second.metadata["list_reminders_handoff"]["reminders"][0].get("selector")
        if not first_selector or first_selector != second_selector:
            raise SystemExit(f"legacy reminder selector was not deterministic: {first} / {second}")
        cancelled = tools["cancel_reminders"].handler({"reminder_id": first_selector})
        if not cancelled.ok or cancelled.metadata.get("count") != 1:
            raise SystemExit(f"stable legacy reminder selector was not cancellable: {cancelled}")
    finally:
        if old_owner is None:
            os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        else:
            os.environ["JARVIS_OWNER_TELEGRAM"] = old_owner


def test_cancel_reminder_by_unique_prefix_and_full_uuid_preserves_sibling() -> None:
    old_owner = os.environ.get("JARVIS_OWNER_TELEGRAM")
    try:
        os.environ["JARVIS_OWNER_TELEGRAM"] = OWNER_CHAT_ID
        for selector, target_id, sibling_id, target_message, sibling_message in (
            ("11111111", FIRST_REMINDER_ID, SECOND_REMINDER_ID, "prefix target", "prefix sibling"),
            (SECOND_REMINDER_ID, SECOND_REMINDER_ID, FIRST_REMINDER_ID, "full target", "full sibling"),
        ):
            now = time.time()
            _seed_fixed_reminders(
                [
                    _fixture_reminder(FIRST_REMINDER_ID, target_message if target_id == FIRST_REMINDER_ID else sibling_message, now + 3600),
                    _fixture_reminder(SECOND_REMINDER_ID, target_message if target_id == SECOND_REMINDER_ID else sibling_message, now + 7200),
                ]
            )
            result = _all_tools()["cancel_reminders"].handler({"reminder_id": selector})
            if not result.ok or result.metadata.get("count") != 1 or "cancelled 1" not in result.output.lower():
                raise SystemExit(f"unique reminder selector did not cancel exactly one row: {result}")
            handoff = _assert_cancel_reminders_handoff(
                result.metadata,
                f"unique reminder cancellation {selector}",
                status="cancelled",
                count=1,
                writes_files=True,
            )
            cancelled_rows = handoff.get("cancelled_reminders") or []
            if [row.get("message") for row in cancelled_rows] != [target_message]:
                raise SystemExit(f"targeted cancellation reported the wrong reminder: {handoff}")

            stored = _stored_reminders_by_id()
            target = stored.get(target_id)
            sibling = stored.get(sibling_id)
            if target is not None and target.get("state") != "cancelled":
                raise SystemExit(f"target reminder remained active after cancellation: {stored}")
            if sibling is None or sibling.get("state") != "pending" or sibling.get("message") != sibling_message:
                raise SystemExit(f"targeted cancellation changed its sibling: {stored}")
            active_owner_ids = {
                reminder_id
                for reminder_id, row in stored.items()
                if row.get("state") in rem._ACTIVE_STATES and row.get("chat_id") == OWNER_CHAT_ID
            }
            if active_owner_ids != {sibling_id}:
                raise SystemExit(f"targeted cancellation changed more than one owner reminder: {stored}")
    finally:
        if old_owner is None:
            os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        else:
            os.environ["JARVIS_OWNER_TELEGRAM"] = old_owner


def test_owner_scoped_cancel_preserves_other_owner_stale_state() -> None:
    old_owner = os.environ.get("JARVIS_OWNER_TELEGRAM")
    try:
        os.environ["JARVIS_OWNER_TELEGRAM"] = OWNER_CHAT_ID
        now = time.time()
        other_owner = _fixture_reminder(
            SECOND_REMINDER_ID,
            "PRIVATE OTHER OWNER CLAIM",
            now + 7200,
            chat_id="different-owner",
        )
        other_owner.update(
            {
                "state": "claimed",
                "token": "other-owner-claim-token",
                "claimed_at": now - rem.DELIVERY_CLAIM_STALE_SECONDS - 60,
            }
        )
        _seed_fixed_reminders(
            [
                _fixture_reminder(FIRST_REMINDER_ID, "owner target", now + 3600),
                other_owner,
            ]
        )
        result = _all_tools()["cancel_reminders"].handler({})
        if not result.ok:
            raise SystemExit(f"owner-scoped cancellation failed: {result}")
        stored_rows = json.loads(rem._reminders_file().read_text(encoding="utf-8"))
        preserved = next(row for row in stored_rows if row.get("id") == SECOND_REMINDER_ID)
        if preserved != other_owner:
            raise SystemExit(f"owner-scoped cancellation mutated another owner's row: {preserved} / {other_owner}")
    finally:
        if old_owner is None:
            os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        else:
            os.environ["JARVIS_OWNER_TELEGRAM"] = old_owner


def test_cancel_reminder_selector_refusals_are_inert_actionable_and_private() -> None:
    old_owner = os.environ.get("JARVIS_OWNER_TELEGRAM")
    try:
        os.environ["JARVIS_OWNER_TELEGRAM"] = OWNER_CHAT_ID
        now = time.time()

        no_match_messages = ["PRIVATE NO MATCH ALPHA", "PRIVATE NO MATCH BETA"]
        _seed_fixed_reminders(
            [
                _fixture_reminder(FIRST_REMINDER_ID, no_match_messages[0], now + 3600),
                _fixture_reminder(SECOND_REMINDER_ID, no_match_messages[1], now + 7200),
            ]
        )
        before = _stored_reminders_by_id()
        no_match = _all_tools()["cancel_reminders"].handler({"reminder_id": "ffffffff"})
        _assert_cancel_selector_refusal(no_match, "no-match reminder selector", private_messages=no_match_messages)
        no_match_output = no_match.output.lower()
        if not any(term in no_match_output for term in ("not found", "couldn't find", "no reminder")) or "list reminders" not in no_match_output:
            raise SystemExit(f"no-match reminder selector was not actionable: {no_match.output}")
        if _stored_reminders_by_id() != before:
            raise SystemExit("no-match reminder selector changed reminder state")

        invalid = _all_tools()["cancel_reminders"].handler({"reminder_id": ""})
        invalid_handoff = _assert_cancel_selector_refusal(
            invalid,
            "empty reminder selector",
            private_messages=no_match_messages,
        )
        if invalid_handoff.get("status") != "invalid_selector" or "list reminders" not in invalid.output.lower():
            raise SystemExit(f"empty reminder selector was not rejected with exact-id guidance: {invalid}")
        if _stored_reminders_by_id() != before:
            raise SystemExit("empty reminder selector fell through to broad cancellation")

        ambiguous_messages = ["PRIVATE AMBIGUOUS ALPHA", "PRIVATE AMBIGUOUS BETA"]
        _seed_fixed_reminders(
            [
                _fixture_reminder(AMBIGUOUS_FIRST_ID, ambiguous_messages[0], now + 3600),
                _fixture_reminder(AMBIGUOUS_SECOND_ID, ambiguous_messages[1], now + 7200),
            ]
        )
        before = _stored_reminders_by_id()
        ambiguous = _all_tools()["cancel_reminders"].handler({"reminder_id": "abcdef12"})
        _assert_cancel_selector_refusal(ambiguous, "ambiguous reminder selector", private_messages=ambiguous_messages)
        ambiguous_output = ambiguous.output.lower()
        if not any(term in ambiguous_output for term in ("ambiguous", "more than one", "multiple")):
            raise SystemExit(f"ambiguous reminder selector was not explicit: {ambiguous.output}")
        if not any(term in ambiguous_output for term in ("longer", "more characters", "list reminders")):
            raise SystemExit(f"ambiguous reminder selector was not actionable: {ambiguous.output}")
        if _stored_reminders_by_id() != before:
            raise SystemExit("ambiguous reminder selector changed reminder state")

        in_flight_messages = ["PRIVATE IN FLIGHT TARGET", "PRIVATE IN FLIGHT SIBLING"]
        _seed_fixed_reminders(
            [
                _fixture_reminder(
                    SENDING_REMINDER_ID,
                    in_flight_messages[0],
                    now + 3600,
                    state="sending",
                ),
                _fixture_reminder(SECOND_REMINDER_ID, in_flight_messages[1], now + 7200),
            ]
        )
        before = _stored_reminders_by_id()
        in_flight = _all_tools()["cancel_reminders"].handler({"reminder_id": "feedface"})
        _assert_cancel_selector_refusal(in_flight, "in-flight reminder selector", private_messages=in_flight_messages)
        in_flight_output = in_flight.output.lower()
        if not any(term in in_flight_output for term in ("in flight", "being sent", "sending", "delivery")):
            raise SystemExit(f"in-flight reminder refusal was not explicit: {in_flight.output}")
        if _stored_reminders_by_id() != before:
            raise SystemExit("in-flight reminder selector changed reminder state")

        other_owner_message = "PRIVATE OTHER OWNER REMINDER"
        _seed_fixed_reminders(
            [
                _fixture_reminder(FIRST_REMINDER_ID, "owner reminder", now + 3600),
                _fixture_reminder(
                    SECOND_REMINDER_ID,
                    other_owner_message,
                    now + 7200,
                    chat_id="different-owner",
                ),
            ]
        )
        before_bytes = rem._reminders_file().read_bytes()
        other_owner_no_match = _all_tools()["cancel_reminders"].handler({"reminder_id": "ffffffff"})
        _assert_cancel_selector_refusal(
            other_owner_no_match,
            "other-owner no-match selector",
            private_messages=[other_owner_message],
        )
        if rem._reminders_file().read_bytes() != before_bytes:
            raise SystemExit("selective refusal mutated another owner's reminder row")
    finally:
        if old_owner is None:
            os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        else:
            os.environ["JARVIS_OWNER_TELEGRAM"] = old_owner


def test_bulk_cancel_reports_in_flight_and_partial_truth() -> None:
    old_owner = os.environ.get("JARVIS_OWNER_TELEGRAM")
    try:
        os.environ["JARVIS_OWNER_TELEGRAM"] = OWNER_CHAT_ID
        now = time.time()
        _seed_fixed_reminders(
            [_fixture_reminder(SENDING_REMINDER_ID, "PRIVATE SENDING ONLY", now + 3600, state="sending")]
        )
        before = rem._reminders_file().read_bytes()
        blocked = _all_tools()["cancel_reminders"].handler({})
        if blocked.ok or blocked.metadata.get("in_flight_count") != 1 or "being sent" not in blocked.output.lower():
            raise SystemExit(f"bulk cancellation hid an in-flight reminder: {blocked}")
        _assert_cancel_reminders_handoff(blocked.metadata, "in-flight bulk cancel", status="in_flight", count=0)
        if rem._reminders_file().read_bytes() != before:
            raise SystemExit("in-flight-only bulk cancellation changed the reminder store")

        stale_sending = _fixture_reminder(
            SENDING_REMINDER_ID,
            "sending sibling",
            now + 7200,
            state="sending",
        )
        stale_sending["sending_at"] = now - rem.DELIVERY_CLAIM_STALE_SECONDS - 60
        _seed_fixed_reminders(
            [
                _fixture_reminder(FIRST_REMINDER_ID, "pending target", now + 3600),
                stale_sending,
            ]
        )
        partial = _all_tools()["cancel_reminders"].handler({})
        if (
            not partial.ok
            or partial.metadata.get("count") != 1
            or partial.metadata.get("in_flight_count") != 1
            or partial.metadata.get("partial") is not True
            or "not cancelled" not in partial.output.lower()
        ):
            raise SystemExit(f"partial bulk cancellation hid its in-flight remainder: {partial}")
        _assert_cancel_reminders_handoff(
            partial.metadata,
            "partial in-flight bulk cancel",
            status="cancelled",
            count=1,
            writes_files=True,
        )
        stored = _stored_reminders_by_id()
        if stored[FIRST_REMINDER_ID].get("state") != "cancelled" or stored[SENDING_REMINDER_ID].get("state") != "sending":
            raise SystemExit(f"partial bulk cancellation crossed the sending boundary: {stored}")
    finally:
        if old_owner is None:
            os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        else:
            os.environ["JARVIS_OWNER_TELEGRAM"] = old_owner


def test_list_and_cancel_refuse_without_owner_configuration() -> None:
    tools = _all_tools()
    old_owner = os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
    try:
        now = time.time()
        _seed_fixed_reminders(
            [
                _fixture_reminder(FIRST_REMINDER_ID, "PRIVATE OWNER ONE", now + 3600),
                _fixture_reminder(
                    SECOND_REMINDER_ID,
                    "PRIVATE OWNER TWO",
                    now + 7200,
                    chat_id="different-owner",
                ),
            ]
        )
        before = rem._reminders_file().read_bytes()
        listed = tools["list_reminders"].handler({})
        cancelled = tools["cancel_reminders"].handler({})
        if listed.ok or cancelled.ok:
            raise SystemExit(f"missing owner configuration did not fail closed: {listed} / {cancelled}")
        if listed.metadata.get("reason") != "missing_owner_configuration":
            raise SystemExit(f"missing-owner list reason drifted: {listed.metadata}")
        _assert_local_productivity_read_recovery(
            listed,
            "missing-owner reminder list",
        )
        if cancelled.metadata.get("reason") != "missing_owner_configuration":
            raise SystemExit(f"missing-owner cancel reason drifted: {cancelled.metadata}")
        if "PRIVATE OWNER" in f"{listed.output}{listed.metadata}{cancelled.output}{cancelled.metadata}":
            raise SystemExit("missing owner configuration disclosed reminder content")
        if rem._reminders_file().read_bytes() != before:
            raise SystemExit("missing owner configuration mutated the reminder store")
        if rem.cancel_pending_item("", "11111111")[0] != "owner_missing":
            raise SystemExit("selective reminder storage API did not reject a missing owner")
        if rem.cancel_pending_items_status("")[3] != "owner_missing":
            raise SystemExit("bulk reminder storage API did not reject a missing owner")
        if rem.pending_reminders_status("")[1:] != (False, "owner_missing"):
            raise SystemExit("reminder read storage API did not reject a missing owner")
    finally:
        if old_owner is not None:
            os.environ["JARVIS_OWNER_TELEGRAM"] = old_owner


def test_cancel_reports_post_publication_durability_uncertainty() -> None:
    old_owner = os.environ.get("JARVIS_OWNER_TELEGRAM")
    original_fsync = rem.os.fsync
    fsync_calls = 0

    def fail_directory_sync(fd: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("synthetic cancellation directory sync failure")
        original_fsync(fd)

    try:
        os.environ["JARVIS_OWNER_TELEGRAM"] = OWNER_CHAT_ID
        _seed_fixed_reminders(
            [
                _fixture_reminder(FIRST_REMINDER_ID, "uncertain cancel target", time.time() + 3600),
                _fixture_reminder(
                    SENDING_REMINDER_ID,
                    "uncertain in-flight sibling",
                    time.time() + 7200,
                    state="sending",
                ),
            ]
        )
        rem.os.fsync = fail_directory_sync  # type: ignore[assignment]
        result = _all_tools()["cancel_reminders"].handler({})
    finally:
        rem.os.fsync = original_fsync  # type: ignore[assignment]
        if old_owner is None:
            os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        else:
            os.environ["JARVIS_OWNER_TELEGRAM"] = old_owner

    if result.ok or "do not retry automatically" not in result.output.lower():
        raise SystemExit(f"uncertain cancellation invited an unsafe retry: {result}")
    for key, expected in {
        "writes_files": True,
        "executes_side_effect": True,
        "storage_error": True,
        "durability_uncertain": True,
        "outcome_known": False,
        "retry_safe": False,
        "authorizes_retry": False,
        "state_changed": True,
        "partial": True,
    }.items():
        if result.metadata.get(key) is not expected:
            raise SystemExit(f"uncertain cancellation metadata drifted at {key}: {result.metadata}")
    handoff = _assert_cancel_reminders_handoff(
        result.metadata,
        "post-publication uncertain cancellation",
        status="published_uncertain",
        count=1,
        writes_files=True,
    )
    if handoff.get("reason") != "directory_sync_failed" or handoff.get("storage_error") is not True:
        raise SystemExit(f"uncertain cancellation handoff lost durability truth: {handoff}")
    if "not cancelled" not in result.output.lower() or result.metadata.get("in_flight_count") != 1:
        raise SystemExit(f"uncertain partial cancellation hid its in-flight remainder: {result}")
    stored = _stored_reminders_by_id()
    if stored[FIRST_REMINDER_ID].get("state") != "cancelled":
        raise SystemExit(f"published uncertain cancellation lost the visible store mutation: {stored}")


def test_list_and_cancel_reminders() -> None:
    _isolate()
    os.environ["JARVIS_OWNER_TELEGRAM"] = "555001"
    tools = _all_tools()
    empty_list = tools["list_reminders"].handler({})
    if not empty_list.output.startswith("You have no pending reminders."):
        raise SystemExit("empty list wrong")
    if empty_list.metadata.get("reads_personal_data") is not True or empty_list.metadata.get("writes_files"):
        raise SystemExit(f"empty reminder list should report personal read without write: {empty_list.metadata}")
    _assert_list_reminders_handoff(empty_list.metadata, "empty reminder list", count=0)
    rem.add_reminder(time.time() + 3600, "drink water", "555001")
    rem.add_reminder(time.time() + 7200, "stand up", "555001")
    listed = tools["list_reminders"].handler({})
    if "drink water" not in listed.output or "stand up" not in listed.output:
        raise SystemExit(f"list did not show reminders: {listed.output}")
    if listed.metadata.get("reads_personal_data") is not True or listed.metadata.get("writes_files"):
        raise SystemExit(f"reminder list should report personal read without write: {listed.metadata}")
    list_handoff = _assert_list_reminders_handoff(listed.metadata, "populated reminder list", count=2)
    if [row.get("message") for row in list_handoff.get("reminders", [])] != ["drink water", "stand up"]:
        raise SystemExit(f"list reminder handoff should preserve display order/messages: {list_handoff}")
    cancelled = tools["cancel_reminders"].handler({})
    if "Cancelled 2" not in cancelled.output:
        raise SystemExit(f"cancel wrong: {cancelled.output}")
    if cancelled.metadata.get("reads_personal_data") is not True or cancelled.metadata.get("writes_files") is not True:
        raise SystemExit(f"cancel should report personal read and file write: {cancelled.metadata}")
    cancel_handoff = _assert_cancel_reminders_handoff(cancelled.metadata, "populated reminder cancel", status="cancelled", count=2, writes_files=True)
    if [row.get("message") for row in cancel_handoff.get("cancelled_reminders", [])] != ["drink water", "stand up"]:
        raise SystemExit(f"cancel handoff should preserve cancelled reminder previews: {cancel_handoff}")
    if rem._load():
        raise SystemExit("reminders not cleared after cancel")
    empty_cancel = tools["cancel_reminders"].handler({})
    if empty_cancel.metadata.get("reads_personal_data") is not True or empty_cancel.metadata.get("writes_files"):
        raise SystemExit(f"empty cancel should report personal read without write: {empty_cancel.metadata}")
    _assert_cancel_reminders_handoff(empty_cancel.metadata, "empty reminder cancel", status="empty", count=0)


def test_list_reminders_surfaces_content_free_delivery_health() -> None:
    _isolate()
    os.environ["JARVIS_OWNER_TELEGRAM"] = OWNER_CHAT_ID
    tools = _all_tools()
    if not rem.record_delivery_health(status="healthy_idle"):
        raise SystemExit("could not seed healthy reminder delivery snapshot")
    healthy = tools["list_reminders"].handler({})
    healthy_handoff = _assert_list_reminders_handoff(
        healthy.metadata,
        "healthy delivery health list",
        count=0,
    )
    if "Reminder delivery health: healthy idle" not in healthy.output:
        raise SystemExit(f"healthy reminder delivery status was not operator-visible: {healthy.output}")
    if (
        healthy.metadata.get("delivery_health_status") != "healthy_idle"
        or healthy_handoff["delivery_health"].get("source_available") is not True
    ):
        raise SystemExit(f"healthy reminder delivery metadata drifted: {healthy.metadata}")

    if not rem.record_delivery_health(
        status="healthy_idle",
        checked_at=time.time() - rem.REMINDER_DELIVERY_HEALTH_STALE_SECONDS - 1,
    ):
        raise SystemExit("could not seed stale reminder delivery snapshot")
    stale = tools["list_reminders"].handler({})
    stale_handoff = _assert_list_reminders_handoff(
        stale.metadata,
        "stale delivery health list",
        count=0,
    )
    if "Reminder delivery health: unknown (stale)." not in stale.output:
        raise SystemExit(f"stale reminder delivery snapshot was trusted: {stale.output}")
    if (
        stale.metadata.get("delivery_health_status") != "unknown"
        or stale_handoff["delivery_health"].get("last_status") != "healthy_idle"
    ):
        raise SystemExit(f"stale reminder health metadata lost bounded history: {stale.metadata}")

    rem._delivery_health_file().write_bytes(b"{private malformed health bytes")
    malformed = tools["list_reminders"].handler({})
    malformed_handoff = _assert_list_reminders_handoff(
        malformed.metadata,
        "malformed delivery health list",
        count=0,
    )
    if "Reminder delivery health: unknown (malformed)." not in malformed.output:
        raise SystemExit(f"malformed reminder delivery snapshot was trusted: {malformed.output}")
    if (
        malformed.metadata.get("delivery_health_status") != "unknown"
        or malformed_handoff["delivery_health"].get("reason") != "malformed"
        or "private malformed" in str(malformed.metadata)
    ):
        raise SystemExit(f"malformed reminder health content leaked: {malformed.metadata}")

    rem._reminders_file().write_bytes(b"{malformed reminder state")
    if not rem.record_delivery_health(status="store_unavailable", reason="malformed_json"):
        raise SystemExit("could not seed unavailable reminder delivery snapshot")
    unavailable = tools["list_reminders"].handler({})
    unavailable_handoff = _assert_list_reminders_unavailable(
        unavailable.metadata,
        "store unavailable delivery health list",
        reason="malformed_json",
    )
    if "Reminder delivery health: store unavailable" not in unavailable.output:
        raise SystemExit(f"store outage health was not operator-visible: {unavailable.output}")
    if unavailable_handoff["delivery_health"].get("status") != "store_unavailable":
        raise SystemExit(f"store outage health metadata drifted: {unavailable.metadata}")
    _assert_no_message_leak(
        {"output": unavailable.output, "metadata": unavailable.metadata},
        ["malformed reminder state", "private malformed health bytes"],
        "reminder delivery health list",
    )


def main() -> None:
    _assert_reminder_exact_metadata_bool()
    _assert_reminder_malformed_handoff_flags()
    test_reminder_docs_do_not_overclaim_phone_delivery()
    test_finish_plan_documents_set_reminder_without_erasing_macos_create_path()
    test_parse_durations_and_time()
    test_huge_duration_does_not_crash()
    test_tool_requires_owner_and_stores()
    test_recurring_reminders_are_explicitly_unsupported_and_inert()
    test_path_shaped_reminder_text_is_rejected_and_redacted()
    test_tool_reports_storage_failure_without_claiming_write()
    test_tool_reports_post_publication_durability_uncertainty_without_retrying()
    test_list_is_read_only_and_cancel_reports_storage_failure()
    test_pop_due_returns_due_keeps_future()
    test_unreadable_existing_store_fails_closed_without_writes()
    test_reminders_file_reads_environment_at_call_time()
    test_blank_reminders_file_env_uses_default()
    test_bridge_delivers_due_reminder()
    test_bridge_redacts_legacy_path_shaped_reminder()
    test_planner_routes_reminders()
    test_planner_routes_uuid_reminder_selectors_and_fails_closed()
    test_location_reminder_draft_is_review_only()
    test_list_reminders_displays_stable_unique_uuid_prefixes()
    test_legacy_reminder_selector_is_stable_and_cancellable()
    test_cancel_reminder_by_unique_prefix_and_full_uuid_preserves_sibling()
    test_owner_scoped_cancel_preserves_other_owner_stale_state()
    test_cancel_reminder_selector_refusals_are_inert_actionable_and_private()
    test_bulk_cancel_reports_in_flight_and_partial_truth()
    test_list_and_cancel_refuse_without_owner_configuration()
    test_cancel_reports_post_publication_durability_uncertainty()
    test_list_and_cancel_reminders()
    test_list_reminders_surfaces_content_free_delivery_health()
    print("Reminders smoke passed")


if __name__ == "__main__":
    main()
