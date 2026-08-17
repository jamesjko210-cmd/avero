"""A one-shot daily brief: greeting + weather + today's calendar + reminders + headlines."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from jarvis_v2.agent.types import RiskLevel, ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.v3_commands import (
    V3_CALENDAR_AUTH_READONLY_COMMAND,
    V3_CALENDAR_AUTH_TERMINAL_LOCATION,
)


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    base = {
        "calls_model": False,
        "calls_external_service": True,
        "executes_tools": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "reads_personal_data": True,
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
    base.update(extra)
    return base


def _greeting(now: datetime) -> str:
    h = now.hour
    if h < 12:
        return "Good morning"
    if h < 18:
        return "Good afternoon"
    return "Good evening"


_SECTION_RECOVERY_HINTS = {
    "weather": "check network access to wttr.in, run `setup check`, then retry `daily briefing`",
    "calendar": (
        "check Google Calendar network/auth, run `setup check`, and if auth expired run "
        f"`{V3_CALENDAR_AUTH_READONLY_COMMAND}` {V3_CALENDAR_AUTH_TERMINAL_LOCATION}, "
        "then retry `daily briefing`"
    ),
    "reminders": "check the local Jarvis reminders file path is writable, run `setup check`, then retry `daily briefing`",
    "news": "check network access to Google News, run `setup check`, then retry `daily briefing`",
}


def _daily_brief_recovery_hints(unavailable: list[str]) -> dict[str, str]:
    return {section: _SECTION_RECOVERY_HINTS[section] for section in unavailable if section in _SECTION_RECOVERY_HINTS}


def _daily_brief_recovery_line(unavailable: list[str]) -> str:
    hints = _daily_brief_recovery_hints(unavailable)
    if not hints:
        return ""
    details = "; ".join(f"{section}: {hint}" for section, hint in hints.items())
    return f"Recovery: {details}."


def _daily_brief_handoff(*, available: list[str], unavailable: list[str]) -> dict[str, Any]:
    next_safe_commands = ["daily briefing"]
    recovery_hints = _daily_brief_recovery_hints(unavailable)
    return {
        "daily_brief_handoff_ready": True,
        "daily_brief_ready_for_operator": True,
        "daily_brief_state_changed": False,
        "daily_brief_changed": [],
        "daily_brief_content_in_handoff": False,
        "daily_brief_next_safe_command": next_safe_commands[0],
        "daily_brief_next_safe_commands": list(next_safe_commands),
        "daily_brief_next_safe_command_count": len(next_safe_commands),
        "daily_brief_recovery_hints": dict(recovery_hints),
        "daily_brief_recovery_hint_count": len(recovery_hints),
        "daily_brief_authorizes_execution": False,
        "daily_brief_authorizes_completion_claim": False,
        "daily_brief_approval_granted": False,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "next_safe_command": next_safe_commands[0],
        "next_safe_commands": list(next_safe_commands),
        "next_safe_command_count": len(next_safe_commands),
        "recovery_hints": dict(recovery_hints),
        "recovery_hint_count": len(recovery_hints),
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "daily_brief_handoff": {
            "tool": "daily_briefing",
            "handoff_ready": True,
            "ready_for_operator": True,
            "state_changed": False,
            "changed": [],
            "content_in_handoff": False,
            "next_safe_command": next_safe_commands[0],
            "next_safe_commands": list(next_safe_commands),
            "next_safe_command_count": len(next_safe_commands),
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "available_sections": list(available),
            "unavailable_sections": list(unavailable),
            "available_count": len(available),
            "unavailable_count": len(unavailable),
            "recovery_hints": dict(recovery_hints),
            "recovery_hint_count": len(recovery_hints),
            "content_in_metadata": False,
            "retry_safe": True,
            "boundaries": {
                "calls_model": False,
                "calls_external_service": True,
                "executes_tools": False,
                "authorizes_execution": False,
                "authorizes_completion_claim": False,
                "approval_granted": False,
                "reads_personal_data": True,
                "reads_private_data": False,
                "executes_side_effect": False,
                "external_side_effect": False,
                "writes_files": False,
                "writes_database": False,
                "writes_memory": False,
                "writes_notes": False,
                "queues_approval": False,
                "controls_computer": False,
            },
        }
    }


def make_brief_tools(config: JarvisConfig):
    def daily_briefing(args: dict[str, Any]) -> ToolResult:
        from jarvis_v2.tools.weather_connector import make_weather_tools
        from jarvis_v2.tools.calendar_connector import make_calendar_tools
        from jarvis_v2.tools.news_connector import make_news_tools
        from jarvis_v2.tools.reminder_tools import make_reminder_tools

        now = datetime.now().astimezone()
        sections = [f"{_greeting(now)}! Here's your brief for {now.strftime('%A, %B %d')}."]
        available: list[str] = []
        unavailable: list[str] = []

        try:
            weather = {t.name: t for t in make_weather_tools(config)}["get_weather"].handler({})
            if weather.ok:
                sections.append(f"\n🌤  {weather.output}")
                available.append("weather")
            else:
                unavailable.append("weather")
        except Exception:
            unavailable.append("weather")

        try:
            events = {t.name: t for t in make_calendar_tools(config)}["list_events"].handler({"range": "today"})
            if events.ok:
                sections.append(f"\n📅  {events.output}")
                available.append("calendar")
            else:
                unavailable.append("calendar")
        except Exception:
            unavailable.append("calendar")

        try:
            reminders = {t.name: t for t in make_reminder_tools(config)}["list_reminders"].handler({})
            # Only surface reminders when some are pending; "none pending" isn't an outage.
            if reminders.ok and reminders.metadata.get("count"):
                sections.append(f"\n⏰  {reminders.output}")
                available.append("reminders")
            elif not reminders.ok:
                unavailable.append("reminders")
        except Exception:
            unavailable.append("reminders")

        try:
            news = {t.name: t for t in make_news_tools(config)}["get_news"].handler({"limit": 3})
            if news.ok:
                sections.append(f"\n📰  {news.output}")
                available.append("news")
            else:
                unavailable.append("news")
        except Exception:
            unavailable.append("news")

        if len(sections) == 1:
            sections.append("\nCouldn't reach any daily brief sections right now.")
            recovery_line = _daily_brief_recovery_line(unavailable)
            if recovery_line:
                sections.append(recovery_line)
        elif unavailable:
            sections.append(f"\nUnavailable: {', '.join(unavailable)}.")
            recovery_line = _daily_brief_recovery_line(unavailable)
            if recovery_line:
                sections.append(recovery_line)
        return ToolResult(
            "daily_briefing",
            True,
            "\n".join(sections),
            _safe_metadata(
                available_sections=list(available),
                unavailable_sections=list(unavailable),
                available_count=len(available),
                unavailable_count=len(unavailable),
                **_daily_brief_handoff(available=available, unavailable=unavailable),
            ),
        )

    from jarvis_v2.tools.registry import Tool
    return [
        Tool(
            "daily_briefing",
            "A combined daily brief: greeting + weather + today's calendar + pending reminders + top headlines. No args.",
            RiskLevel.LOCAL_SAFE,
            daily_briefing,
            "personal",
        ),
    ]
