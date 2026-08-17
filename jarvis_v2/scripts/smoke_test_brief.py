"""Smoke tests for the daily brief (mocked weather/calendar/news, no network)."""

from __future__ import annotations

from datetime import datetime

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import RiskLevel
from jarvis_v2.config import load_config
import os

from jarvis_v2.tools import brief_tools as bt
from jarvis_v2.tools import weather_connector as wc
from jarvis_v2.tools import calendar_connector as cc
from jarvis_v2.tools import news_connector as nc
from jarvis_v2.tools import reminder_tools as rt


def _tool():
    return {t.name: t for t in bt.make_brief_tools(load_config())}["daily_briefing"]


def test_greeting_by_hour() -> None:
    if bt._greeting(datetime(2026, 6, 15, 8)) != "Good morning":
        raise SystemExit("morning greeting wrong")
    if bt._greeting(datetime(2026, 6, 15, 14)) != "Good afternoon":
        raise SystemExit("afternoon greeting wrong")
    if bt._greeting(datetime(2026, 6, 15, 20)) != "Good evening":
        raise SystemExit("evening greeting wrong")


def test_is_local_safe() -> None:
    if _tool().risk != RiskLevel.LOCAL_SAFE:
        raise SystemExit("daily_briefing should be LOCAL_SAFE")


def _install_mocks() -> None:
    # Reminder reads require an explicit owner. Keep this mocked brief
    # independent of the operator's real environment and reminder store.
    os.environ["JARVIS_OWNER_TELEGRAM"] = "smoke-owner"
    wc._fetch = lambda location: {
        "current_condition": [{"temp_C": "20", "FeelsLikeC": "20", "humidity": "50",
                               "windspeedKmph": "5", "weatherDesc": [{"value": "Clear"}]}],
        "weather": [{"maxtempC": "25", "mintempC": "15"}],
        "nearest_area": [{"areaName": [{"value": "Seoul"}]}],
    }

    class FakeExec:
        def __init__(self, payload):
            self._p = payload

        def execute(self):
            return self._p

    class FakeService:
        def events(self):
            outer = self

            class E:
                def list(self, **kwargs):
                    return FakeExec({"items": [{"summary": "Standup", "start": {"dateTime": "2026-06-15T09:00:00+09:00"}}]})

            return E()

    cc._get_readonly_service = lambda: FakeService()
    nc._fetch = lambda query: b"<rss><channel><item><title>Big news today - Wire</title></item></channel></rss>"
    # One pending reminder for the owner so the brief's reminders section is deterministic.
    _chat = os.getenv("JARVIS_OWNER_TELEGRAM", "").strip()
    rt._rem.pending_reminders_status = lambda owner_chat_id="": (  # type: ignore
        [
            {
                "id": "00000000-0000-0000-0000-000000000001",
                "due": 99999999999,
                "message": "Call mom",
                "chat_id": owner_chat_id or _chat,
                "transport": "telegram",
                "state": "pending",
                "attempt_count": 0,
            }
        ],
        True,
        "",
    )


def _assert_daily_brief_handoff(metadata: dict, label: str, *, available: list[str], unavailable: list[str]) -> dict:
    handoff = metadata.get("daily_brief_handoff")
    if (
        not isinstance(handoff, dict)
        or metadata.get("daily_brief_handoff_ready") is not True
        or handoff.get("handoff_ready") is not True
    ):
        raise SystemExit(f"{label} missed structured handoff: {metadata}")
    for key, expected in (
        ("ready_for_operator", True),
        ("state_changed", False),
        ("changed", []),
        ("content_in_handoff", False),
        ("authorizes_execution", False),
        ("authorizes_completion_claim", False),
        ("approval_granted", False),
    ):
        if metadata.get(key) != expected or handoff.get(key) != expected:
            raise SystemExit(f"{label} missed flat/nested {key}={expected}: {metadata} / {handoff}")
        prefixed_key = f"daily_brief_{key}"
        if metadata.get(prefixed_key) != expected:
            raise SystemExit(f"{label} missed prefixed {prefixed_key}={expected}: {metadata}")
    if handoff.get("available_sections") != available or handoff.get("unavailable_sections") != unavailable:
        raise SystemExit(f"{label} handoff section status wrong: {handoff}")
    if metadata.get("available_count") != len(available) or metadata.get("unavailable_count") != len(unavailable):
        raise SystemExit(f"{label} metadata missed section counts: {metadata}")
    if handoff.get("available_count") != len(available) or handoff.get("unavailable_count") != len(unavailable):
        raise SystemExit(f"{label} handoff missed section counts: {handoff}")
    expected_recovery = bt._daily_brief_recovery_hints(unavailable)
    if (
        metadata.get("recovery_hints") != expected_recovery
        or metadata.get("daily_brief_recovery_hints") != expected_recovery
        or handoff.get("recovery_hints") != expected_recovery
    ):
        raise SystemExit(f"{label} missed safe recovery hints: {metadata} / {handoff}")
    if (
        metadata.get("recovery_hint_count") != len(expected_recovery)
        or metadata.get("daily_brief_recovery_hint_count") != len(expected_recovery)
        or handoff.get("recovery_hint_count") != len(expected_recovery)
    ):
        raise SystemExit(f"{label} missed recovery hint count: {metadata} / {handoff}")
    if handoff.get("content_in_metadata") is not False or "Standup" in str(handoff) or "Big news today" in str(handoff):
        raise SystemExit(f"{label} handoff should be content-free: {handoff}")
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict):
        raise SystemExit(f"{label} handoff missed boundaries: {handoff}")
    expected_false = [
        "calls_model",
        "executes_tools",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "executes_side_effect",
        "external_side_effect",
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "queues_approval",
        "controls_computer",
    ]
    for key in expected_false:
        if boundaries.get(key) or metadata.get(key):
            raise SystemExit(f"{label} should not perform {key}: {metadata} / {handoff}")
    if not boundaries.get("calls_external_service") or not boundaries.get("reads_personal_data"):
        raise SystemExit(f"{label} handoff should declare external/personal-data reads: {handoff}")
    expected_commands = ["daily briefing"]
    if (
        metadata.get("next_safe_command") != expected_commands[0]
        or handoff.get("next_safe_command") != expected_commands[0]
        or metadata.get("daily_brief_next_safe_command") != expected_commands[0]
    ):
        raise SystemExit(f"{label} handoff missed retry command: {metadata} / {handoff}")
    if (
        metadata.get("next_safe_commands") != expected_commands
        or handoff.get("next_safe_commands") != expected_commands
        or metadata.get("daily_brief_next_safe_commands") != expected_commands
    ):
        raise SystemExit(f"{label} handoff missed safe-command list: {metadata} / {handoff}")
    if (
        metadata.get("next_safe_command_count") != len(expected_commands)
        or handoff.get("next_safe_command_count") != len(expected_commands)
        or metadata.get("daily_brief_next_safe_command_count") != len(expected_commands)
    ):
        raise SystemExit(f"{label} handoff missed safe-command count: {metadata} / {handoff}")
    if handoff.get("retry_safe") is not True:
        raise SystemExit(f"{label} handoff missed retry command: {handoff}")
    return handoff


def test_brief_composes_all_sections() -> None:
    _install_mocks()
    out = _tool().handler({})
    if not out.ok:
        raise SystemExit(f"brief failed: {out.output}")
    for fragment in ["Seoul", "Standup", "Call mom", "Big news today"]:
        if fragment not in out.output:
            raise SystemExit(f"brief missing {fragment!r}: {out.output}")
    if "Good " not in out.output:
        raise SystemExit("brief missing greeting")
    metadata = out.metadata
    _assert_daily_brief_handoff(metadata, "brief", available=["weather", "calendar", "reminders", "news"], unavailable=[])


def test_brief_reports_partial_unavailable_sections() -> None:
    _install_mocks()

    def weather_down(location):
        raise RuntimeError("offline")

    wc._fetch = weather_down  # type: ignore
    out = _tool().handler({})
    if not out.ok:
        raise SystemExit(f"partial brief should still render: {out.output}")
    for fragment in ["Standup", "Call mom", "Big news today", "Unavailable: weather."]:
        if fragment not in out.output:
            raise SystemExit(f"partial brief missing {fragment!r}: {out.output}")
    for fragment in ["Recovery:", "weather: check network access to wttr.in", "setup check", "daily briefing"]:
        if fragment not in out.output:
            raise SystemExit(f"partial brief missed recovery guidance {fragment!r}: {out.output}")
    if "offline" in out.output:
        raise SystemExit(f"partial brief leaked raw backend error: {out.output}")
    _assert_daily_brief_handoff(out.metadata, "partial brief", available=["calendar", "reminders", "news"], unavailable=["weather"])


def test_brief_reports_false_reminder_result_as_unavailable() -> None:
    _install_mocks()
    rt._rem.pending_reminders_status = lambda owner_chat_id="": ([], False, "malformed_json")  # type: ignore
    out = _tool().handler({})
    if not out.ok:
        raise SystemExit(f"brief should remain usable when reminders are unavailable: {out.output}")
    if "Unavailable: reminders." not in out.output or "You have no pending reminders" in out.output:
        raise SystemExit(f"brief hid reminder-store unavailability: {out.output}")
    if "malformed_json" in out.output or "malformed_json" in str(out.metadata):
        raise SystemExit(f"brief should not expose reminder storage internals: {out.output} / {out.metadata}")
    _assert_daily_brief_handoff(
        out.metadata,
        "reminder-unavailable brief",
        available=["weather", "calendar", "news"],
        unavailable=["reminders"],
    )


def test_brief_reports_all_unavailable_recovery_without_leaks() -> None:
    _install_mocks()

    def weather_down(location):
        raise RuntimeError("/\x55sers/example/weather backend SHOULD NOT APPEAR")

    def calendar_down():
        raise RuntimeError("/private/tmp/calendar backend SHOULD NOT APPEAR")

    def reminders_down():
        raise RuntimeError("reminders backend SHOULD NOT APPEAR")

    def news_down(query):
        raise RuntimeError("news backend SHOULD NOT APPEAR")

    wc._fetch = weather_down  # type: ignore
    cc._get_readonly_service = calendar_down  # type: ignore
    rt._rem.pending_reminders_status = lambda owner_chat_id="": reminders_down()  # type: ignore
    nc._fetch = news_down  # type: ignore

    out = _tool().handler({})
    if not out.ok:
        raise SystemExit(f"all-unavailable brief should still render a recovery packet: {out.output}")
    for fragment in [
        "Couldn't reach any daily brief sections right now.",
        "Recovery:",
        "weather: check network access to wttr.in",
        "calendar: check Google Calendar network/auth",
        "./launch_jarvis_v3_calendar_auth.py readonly",
        "normal macOS Terminal shell prompt",
        "not in Telegram",
        "reminders: check the local Jarvis reminders file path is writable",
        "news: check network access to Google News",
        "setup check",
        "daily briefing",
    ]:
        if fragment not in out.output:
            raise SystemExit(f"all-unavailable brief missed {fragment!r}: {out.output}")
    if "./launch_jarvis_v3_calendar_auth.py full-access" in out.output:
        raise SystemExit("read-only daily brief suggested full-access Calendar authorization")
    for forbidden in ["SHOULD NOT APPEAR", "/\x55sers/operator", "/private/tmp", "backend"]:
        if forbidden in out.output or forbidden in str(out.metadata):
            raise SystemExit(f"all-unavailable brief leaked raw diagnostic {forbidden!r}: {out.output} / {out.metadata}")
    _assert_daily_brief_handoff(
        out.metadata,
        "all-unavailable brief",
        available=[],
        unavailable=["weather", "calendar", "reminders", "news"],
    )


def test_planner_routes_brief() -> None:
    p = RuleBasedPlanner()
    for q in [
        "good morning",
        "morning brief",
        "morning briefing please",
        "daily briefing",
        "brief me",
        "brief please",
        "today brief please",
        "today's brief please",
        "what's my brief today",
        "phone brief please",
        "telegram brief please",
        "how's my day look",
    ]:
        if [a.tool_name for a in p.plan(q).actions] != ["daily_briefing"]:
            raise SystemExit(f"planner missed brief route: {q!r}")
    for q in ["send brief to phone", "send brief to telegram", "send brief now"]:
        actions = p.plan(q).actions
        if [a.tool_name for a in actions] != ["run_job_now"] or actions[0].args != {"name": "Morning Brief"}:
            raise SystemExit(f"planner should trigger Morning Brief now for {q!r}: {actions!r}")
    # "daily brief" / "proactive brief" keep the existing context brief tool.
    for q in ["daily brief", "daily brief please", "proactive brief"]:
        if [a.tool_name for a in p.plan(q).actions] != ["daily_brief"]:
            raise SystemExit(f"{q!r} should stay on the context daily_brief tool")


def main() -> None:
    test_greeting_by_hour()
    test_is_local_safe()
    test_brief_composes_all_sections()
    test_brief_reports_partial_unavailable_sections()
    test_brief_reports_false_reminder_result_as_unavailable()
    test_brief_reports_all_unavailable_recovery_without_leaks()
    test_planner_routes_brief()
    print("Brief smoke passed")


if __name__ == "__main__":
    main()
