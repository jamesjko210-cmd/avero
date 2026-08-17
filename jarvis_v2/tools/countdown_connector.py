"""Days-until / countdown for Jarvis V2 (local date math, no network)."""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import date, timedelta
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    LOCAL_READ_INPUT_RECOVERY_ACTION,
    declare_retryable_local_read_failure,
)
from jarvis_v2.agent.types import RiskLevel, ToolResult
from jarvis_v2.config import JarvisConfig


_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10,
    "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_NAMED = {
    "christmas": (12, 25), "xmas": (12, 25), "christmas eve": (12, 24),
    "new year": (1, 1), "new years": (1, 1), "new year's": (1, 1), "new years day": (1, 1),
    "new years eve": (12, 31), "new year's eve": (12, 31),
    "halloween": (10, 31), "valentine's day": (2, 14), "valentines day": (2, 14), "valentines": (2, 14),
    "april fools": (4, 1), "independence day": (7, 4), "july 4th": (7, 4), "the fourth of july": (7, 4),
    "juneteenth": (6, 19), "veterans day": (11, 11), "veteran's day": (11, 11), "groundhog day": (2, 2),
}
_EASTER_NAMES = {"easter", "easter sunday"}
_MOVABLE_NAMES = {
    "martin luther king jr day": "mlk day",
    "martin luther king day": "mlk day",
    "mlk day": "mlk day",
    "presidents day": "presidents day",
    "president's day": "presidents day",
    "washington's birthday": "presidents day",
    "washingtons birthday": "presidents day",
    "indigenous peoples day": "indigenous peoples day",
    "indigenous people's day": "indigenous peoples day",
    "columbus day": "columbus day",
    "thanksgiving": "thanksgiving",
    "thanksgiving day": "thanksgiving",
    "black friday": "black friday",
    "mother's day": "mother's day",
    "mothers day": "mother's day",
    "mother day": "mother's day",
    "father's day": "father's day",
    "fathers day": "father's day",
    "father day": "father's day",
    "memorial day": "memorial day",
    "labor day": "labor day",
    "labour day": "labor day",
}
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
MAX_TARGET_CHARS = 120


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    base = {
        "calls_model": False,
        "calls_external_service": False,
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
    base.update(extra)
    return base


def _target_metadata(text: str, **extra: Any) -> dict[str, Any]:
    return _safe_metadata(target_length=len(text), local_path_target=bool(LOCAL_PATH_RE.search(text or "")), **extra)


def _display_target(value: Any, *, limit: int = MAX_TARGET_CHARS) -> str:
    text = "" if value is None else str(value).strip()
    text = " ".join(text.split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "..."
    return text


def _countdown_boundaries() -> dict[str, bool]:
    return {
        "calls_model": False,
        "calls_external_service": False,
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


def _countdown_handoff(
    *,
    text: str,
    status: str,
    reason: str = "",
    label: str = "",
    target: date | None = None,
    days: int | None = None,
    unit: str = "days",
    weeks: int | None = None,
    week_remainder_days: int | None = None,
    months: int | None = None,
    month_remainder_days: int | None = None,
) -> dict[str, Any]:
    if days is None:
        timing = "unknown"
    elif days < 0:
        timing = "past"
    elif days == 0:
        timing = "today"
    elif days == 1:
        timing = "tomorrow"
    else:
        timing = "future"
    next_safe_command = "days until <date or event>"
    boundaries = _countdown_boundaries()
    handoff = {
        "source": "days_until",
        "status": status,
        "reason": reason,
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "content_in_handoff": False,
        "target_preview": _display_target(text),
        "target_length": len(text),
        "local_path_target": bool(LOCAL_PATH_RE.search(text or "")),
        "label": _display_target(label),
        "target_date": target.isoformat() if target else "",
        "days": days,
        "unit": unit,
        "weeks": weeks,
        "week_remainder_days": week_remainder_days,
        "months": months,
        "month_remainder_days": month_remainder_days,
        "timing": timing,
        "retry_safe": status != "ok",
        "next_safe_command": next_safe_command,
        "next_safe_commands": [next_safe_command],
        "next_safe_command_count": 1,
        "boundaries": boundaries,
    }
    return {
        "countdown_handoff_ready": True,
        "ready_for_operator": handoff["ready_for_operator"],
        "state_changed": handoff["state_changed"],
        "changed": handoff["changed"],
        "authorizes_execution": handoff["authorizes_execution"],
        "authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "approval_granted": handoff["approval_granted"],
        "content_in_handoff": handoff["content_in_handoff"],
        "countdown_ready_for_operator": handoff["ready_for_operator"],
        "countdown_state_changed": handoff["state_changed"],
        "countdown_changed": handoff["changed"],
        "countdown_content_in_handoff": handoff["content_in_handoff"],
        "countdown_next_safe_command": handoff["next_safe_command"],
        "countdown_next_safe_commands": handoff["next_safe_commands"],
        "countdown_next_safe_command_count": handoff["next_safe_command_count"],
        "countdown_authorizes_execution": handoff["authorizes_execution"],
        "countdown_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "countdown_approval_granted": handoff["approval_granted"],
        "countdown_boundaries": boundaries,
        "countdown_handoff": handoff,
    }


def _next_occurrence(month: int, day: int, today: date) -> date | None:
    year = today.year
    try:
        target = date(year, month, day)
    except ValueError:
        return None
    if target < today:
        target = date(year + 1, month, day)
    return target


def _easter_sunday(year: int) -> date:
    # Anonymous Gregorian computus.
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _next_easter(today: date) -> date:
    target = _easter_sunday(today.year)
    if target < today:
        target = _easter_sunday(today.year + 1)
    return target


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    target = date(year, month, 1)
    offset = (weekday - target.weekday()) % 7
    return target + timedelta(days=offset + (n - 1) * 7)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        target = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        target = date(year, month + 1, 1) - timedelta(days=1)
    return target - timedelta(days=(target.weekday() - weekday) % 7)


def _movable_date(name: str, year: int) -> date:
    if name == "mlk day":
        return _nth_weekday(year, 1, 0, 3)  # third Monday in January
    if name == "presidents day":
        return _nth_weekday(year, 2, 0, 3)  # third Monday in February
    if name in {"columbus day", "indigenous peoples day"}:
        return _nth_weekday(year, 10, 0, 2)  # second Monday in October
    if name == "thanksgiving":
        return _nth_weekday(year, 11, 3, 4)  # fourth Thursday in November
    if name == "black friday":
        return _movable_date("thanksgiving", year) + timedelta(days=1)
    if name == "mother's day":
        return _nth_weekday(year, 5, 6, 2)  # second Sunday in May
    if name == "father's day":
        return _nth_weekday(year, 6, 6, 3)  # third Sunday in June
    if name == "memorial day":
        return _last_weekday(year, 5, 0)  # last Monday in May
    if name == "labor day":
        return _nth_weekday(year, 9, 0, 1)  # first Monday in September
    raise ValueError(f"unknown movable holiday: {name}")


def _next_movable(name: str, today: date) -> date:
    target = _movable_date(name, today.year)
    if target < today:
        target = _movable_date(name, today.year + 1)
    return target


def _plural(value: int, singular: str) -> str:
    return f"{value} {singular}{'' if value == 1 else 's'}"


def _add_calendar_months(start: date, months: int) -> date:
    month_index = start.month - 1 + months
    year = start.year + month_index // 12
    month = month_index % 12 + 1
    day = min(start.day, monthrange(year, month)[1])
    return date(year, month, day)


def _calendar_month_delta(start: date, target: date) -> tuple[int, int]:
    months = (target.year - start.year) * 12 + target.month - start.month
    if months < 0:
        return 0, (target - start).days
    if _add_calendar_months(start, months) > target:
        months -= 1
    anchor = _add_calendar_months(start, max(months, 0))
    return max(months, 0), (target - anchor).days


def _countdown_unit(value: Any) -> str:
    unit = str(value or "").strip().lower()
    if unit in {"week", "weeks"}:
        return "weeks"
    if unit in {"month", "months"}:
        return "months"
    return "days"


def _weeks_parts(days: int) -> tuple[int, int]:
    return divmod(days, 7)


def _format_weeks(days: int) -> str:
    weeks, remainder = _weeks_parts(days)
    exact_days = _plural(days, "day")
    if weeks == 0:
        return f"Less than 1 week ({exact_days})"
    if remainder == 0:
        return f"{_plural(weeks, 'week')} ({exact_days})"
    return f"{_plural(weeks, 'week')} and {_plural(remainder, 'day')} ({exact_days})"


def _format_months(start: date, target: date, days: int) -> tuple[str, int, int]:
    months, remainder = _calendar_month_delta(start, target)
    exact_days = _plural(days, "day")
    if months == 0:
        return f"Less than 1 month ({exact_days})", months, remainder
    if remainder == 0:
        return f"{_plural(months, 'month')} ({exact_days})", months, remainder
    return f"{_plural(months, 'month')} and {_plural(remainder, 'day')} ({exact_days})", months, remainder


def parse_target(text: str, today: date | None = None) -> tuple[date, str] | None:
    today = today or date.today()
    low = " ".join(text.lower().split())

    if any(name in low for name in _EASTER_NAMES):
        return _next_easter(today), "easter"

    for alias, name in _MOVABLE_NAMES.items():
        if alias in low:
            return _next_movable(name, today), name

    for name, (mo, d) in _NAMED.items():
        if name in low:
            target = _next_occurrence(mo, d, today)
            return (target, name) if target else None

    # full ISO date
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", low)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3))), m.group(0)
        except ValueError:
            return None
    # "month day"
    m = re.search(r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2})\b", low)
    if m:
        target = _next_occurrence(_MONTHS[m.group(1)], int(m.group(2)), today)
        return (target, f"{m.group(1).capitalize()} {m.group(2)}") if target else None
    # MM/DD
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})\b", low)
    if m:
        target = _next_occurrence(int(m.group(1)), int(m.group(2)), today)
        return (target, m.group(0)) if target else None
    # bare year
    m = re.search(r"\b(20\d{2})\b", low)
    if m:
        return date(int(m.group(1)), 1, 1), m.group(1)
    return None


def make_countdown_tools(config: JarvisConfig):
    def days_until(args: dict[str, Any]) -> ToolResult:
        text = str(args.get("target") or args.get("text") or "").strip()
        if not text:
            failure_output = (
                "Days until what? e.g. 'days until Christmas'. "
                f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "days_until",
                False,
                failure_output,
                declare_retryable_local_read_failure(
                    _target_metadata(text, reason="missing_target", **_countdown_handoff(text=text, status="refused", reason="missing_target")),
                    output=failure_output,
                    action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                ),
            )
        if LOCAL_PATH_RE.search(text):
            return ToolResult(
                "days_until",
                False,
                "Tell me a date or event, not a local file path.",
                _target_metadata(text, reason="invalid_target", **_countdown_handoff(text=text, status="refused", reason="invalid_target")),
            )
        today = date.today()
        parsed = parse_target(text, today)
        if parsed is None:
            return ToolResult(
                "days_until",
                False,
                "Tell me a date or event, e.g. 'days until Dec 25' or 'days until New Year'.",
                _target_metadata(text, reason="invalid_target", **_countdown_handoff(text=text, status="refused", reason="invalid_target")),
            )
        target, label = parsed
        delta = (target - today).days
        unit = _countdown_unit(args.get("unit"))
        if delta < 0:
            return ToolResult(
                "days_until",
                True,
                f"{label.capitalize()} ({target.strftime('%B %d, %Y')}) has already passed — that was {abs(delta)} days ago.",
                _target_metadata(text, days=delta, unit=unit, **_countdown_handoff(text=text, status="ok", label=label, target=target, days=delta, unit=unit)),
            )
        if delta == 0:
            return ToolResult(
                "days_until",
                True,
                f"{label.capitalize()} is today!",
                _target_metadata(text, days=delta, unit=unit, **_countdown_handoff(text=text, status="ok", label=label, target=target, days=delta, unit=unit)),
            )
        if delta == 1:
            return ToolResult(
                "days_until",
                True,
                f"{label.capitalize()} is tomorrow ({target.isoformat()}).",
                _target_metadata(text, days=delta, unit=unit, **_countdown_handoff(text=text, status="ok", label=label, target=target, days=delta, unit=unit)),
            )
        when = target.strftime("%A, %B %d, %Y")
        weeks = week_remainder_days = months = month_remainder_days = None
        if unit == "weeks":
            weeks, week_remainder_days = _weeks_parts(delta)
            message = f"{_format_weeks(delta)} until {label} ({when})."
        elif unit == "months":
            month_text, months, month_remainder_days = _format_months(today, target, delta)
            message = f"{month_text} until {label} ({when})."
        else:
            message = f"{delta} days until {label} ({when})."
        return ToolResult(
            "days_until",
            True,
            message,
            _target_metadata(
                text,
                days=delta,
                unit=unit,
                weeks=weeks,
                week_remainder_days=week_remainder_days,
                months=months,
                month_remainder_days=month_remainder_days,
                **_countdown_handoff(
                    text=text,
                    status="ok",
                    label=label,
                    target=target,
                    days=delta,
                    unit=unit,
                    weeks=weeks,
                    week_remainder_days=week_remainder_days,
                    months=months,
                    month_remainder_days=month_remainder_days,
                ),
            ),
        )

    from jarvis_v2.tools.registry import Tool
    return [
        Tool(
            "days_until",
            "Count days until a date or holiday. Args: target/text (e.g. 'Christmas', 'Dec 25', '2027-01-01').",
            RiskLevel.READ_ONLY,
            days_until,
            "personal",
        ),
    ]
