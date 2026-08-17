"""Minimal natural-language date/time parser for calendar event creation.

Stdlib only. Handles the common phone phrasings:
  "schedule lunch tomorrow at noon"
  "create an event called dentist friday at 3pm"
  "add meeting with sarah monday 10am"
  "block 2pm to 4pm tomorrow for deep work"

parse_event(text) -> (title, start_iso, end_iso)  on success
                  -> ("__need_time__", "", "")     when a day/title is clear but no time
                  -> None                            when it does not look like an event
All-day events return date-only start/end strings, with end as the next day.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta


_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_CREATE_VERBS = r"^(?:please\s+)?(?:schedule|create|add|set up|setup|book|make|new|put|block)\b"
_EVENT_NOUNS = r"\b(?:an?|the)\s+(?:event|meeting|appointment)\s+(?:called|titled|named)\s+"
_LEAD_NOUNS = r"\b(?:an?|the)\s+(?:event|meeting|appointment)\b"


def _month_day_pattern() -> str:
    month_names = "|".join(sorted(_MONTHS, key=len, reverse=True))
    return rf"\b(?:on\s+)?({month_names})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,\s*(\d{{4}}))?\b"


def _numeric_day_pattern() -> str:
    return r"\b(?:on\s+)?(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b"


def _next_weekday(base: datetime, target: int) -> datetime:
    ahead = (target - base.weekday()) % 7
    return base + timedelta(days=ahead)


def _weekday_from_text(text: str, base: datetime) -> datetime | None:
    for name, idx in _WEEKDAYS.items():
        if not re.search(rf"\b{name}\b", text):
            continue
        day = _next_weekday(base, idx)
        if re.search(rf"\bnext\s+{name}\b", text):
            day += timedelta(days=7)
        return day
    return None


def _parse_time(text: str) -> tuple[int, int] | None:
    """Return (hour, minute) or None. Accepts noon/midnight, 3pm, 3:30 pm, 15:00, at 3."""
    def valid(hour: int, minute: int) -> tuple[int, int] | None:
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour, minute
        return None

    if re.search(r"\bnoon\b", text):
        return 12, 0
    if re.search(r"\bmidnight\b", text):
        return 0, 0
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text)
    if m:
        raw_hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        if not (1 <= raw_hour <= 12):
            return None
        hour = raw_hour % 12
        if m.group(3) == "pm":
            hour += 12
        return valid(hour, minute)
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", text)  # 24h
    if m:
        return valid(int(m.group(1)), int(m.group(2)))
    m = re.search(r"\bat\s+(\d{1,2})\b", text)  # "at 3" -> assume pm for 1-7, am for 8-12
    if m:
        hour = int(m.group(1))
        if not 0 <= hour <= 23:
            return None
        if 1 <= hour <= 7:
            hour += 12
        return valid(hour % 24, 0)
    return None


def _parse_time_range(text: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    time_token = r"(?:\d{1,2}(?::\d{2})?\s*(?:am|pm)?|noon|midnight)"
    match = re.search(
        rf"\b(?:from\s+)?({time_token})\s*(?:to|until|till|-)\s*({time_token})\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    if match.end(2) < len(text) and text[match.end(2)] in "/-":
        return None

    start_text = match.group(1).strip()
    end_text = match.group(2).strip()
    if not re.search(r"\b(?:am|pm|noon|midnight)\b|:", start_text, flags=re.IGNORECASE):
        end_meridiem = re.search(r"\b(am|pm)\b", end_text, flags=re.IGNORECASE)
        if end_meridiem:
            start_text = f"{start_text}{end_meridiem.group(1).lower()}"
        else:
            start_text = f"at {start_text}"
    if not re.search(r"\b(?:am|pm|noon|midnight)\b|:", end_text, flags=re.IGNORECASE):
        start_meridiem = re.search(r"\b(am|pm)\b", start_text, flags=re.IGNORECASE)
        if start_meridiem:
            end_text = f"{end_text}{start_meridiem.group(1).lower()}"
        else:
            end_text = f"at {end_text}"

    start_hm = _parse_time(start_text)
    end_hm = _parse_time(end_text)
    if not start_hm or not end_hm:
        return None
    return start_hm, end_hm


def _is_all_day(text: str) -> bool:
    return bool(re.search(r"\b(all[-\s]?day|for the day|whole day)\b", text, flags=re.IGNORECASE))


def _explicit_month_day(text: str, base: datetime) -> datetime | None:
    match = re.search(
        _month_day_pattern(),
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    month = _MONTHS[match.group(1).lower().rstrip(".")]
    day = int(match.group(2))
    year = int(match.group(3) or base.year)
    try:
        candidate = base.replace(year=year, month=month, day=day)
    except ValueError:
        return None
    if match.group(3) is None and candidate.date() < base.date():
        candidate = candidate.replace(year=year + 1)
    return candidate


def _explicit_numeric_day(text: str, base: datetime) -> datetime | None:
    match = re.search(_numeric_day_pattern(), text, flags=re.IGNORECASE)
    if not match:
        return None
    month = int(match.group(1))
    day = int(match.group(2))
    raw_year = match.group(3)
    if raw_year:
        year = int(raw_year)
        if year < 100:
            year += 2000
    else:
        year = base.year
    try:
        candidate = base.replace(year=year, month=month, day=day)
    except ValueError:
        return None
    if raw_year is None and candidate.date() < base.date():
        candidate = candidate.replace(year=year + 1)
    return candidate


def _extract_title(text: str) -> str:
    t = text.strip()
    t = re.sub(_CREATE_VERBS, "", t, flags=re.IGNORECASE)
    t = re.sub(_EVENT_NOUNS, "", t, flags=re.IGNORECASE)
    t = re.sub(_LEAD_NOUNS, "", t, flags=re.IGNORECASE)
    month_names = "|".join(sorted(_MONTHS, key=len, reverse=True))
    t = re.sub(
        rf"\b(?:on\s+)?(?:{month_names})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*\d{{4}})?\b",
        "",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(_numeric_day_pattern(), "", t, flags=re.IGNORECASE)
    # strip day words
    t = re.sub(r"\b(today|tonight|tomorrow|" + "|".join(_WEEKDAYS) + r")\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bnext\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(all[-\s]?day|for the day|whole day)\b", "", t, flags=re.IGNORECASE)
    # strip time phrases
    time_token = r"(?:\d{1,2}(?::\d{2})?\s*(?:am|pm)?|noon|midnight)"
    t = re.sub(rf"\b(?:from\s+)?{time_token}\s*(?:to|until|till|-)\s*{time_token}\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bfrom\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(at|from|to|until|till|-)\s*\d{1,2}(?::\d{2})?\s*(am|pm)?\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b\d{1,2}(?::\d{2})?\s*(am|pm)\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(noon|midnight)\b", "", t, flags=re.IGNORECASE)
    # strip dangling prepositions / filler left behind
    t = re.sub(r"\b(for|on|at|to|with the)\b\s*$", "", t.strip(), flags=re.IGNORECASE)
    t = re.sub(r"^\s*(for|on|at|to)\b\s+", "", t.strip(), flags=re.IGNORECASE)
    t = re.sub(r"^(a|an|the)\s+", "", t.strip(), flags=re.IGNORECASE)
    t = " ".join(t.split()).strip(" ,.-")
    return t


def parse_event(text: str, now: datetime | None = None) -> tuple[str, str, str] | None:
    now = now or datetime.now().astimezone()
    low = " ".join(text.lower().split())

    # Resolve the day.
    day = now
    day_source = "implicit"
    if re.search(r"\btomorrow\b", low):
        day = now + timedelta(days=1)
        day_source = "relative"
    elif re.search(r"\b(today|tonight)\b", low):
        day = now
        day_source = "explicit_today"
    elif re.search(_month_day_pattern(), low, flags=re.IGNORECASE):
        explicit_day = _explicit_month_day(low, now)
        if explicit_day is None:
            return None
        day = explicit_day
        day_source = "explicit_date"
    elif re.search(_numeric_day_pattern(), low, flags=re.IGNORECASE):
        explicit_day = _explicit_numeric_day(low, now)
        if explicit_day is None:
            return None
        day = explicit_day
        day_source = "explicit_date"
    elif weekday_day := _weekday_from_text(low, now):
        day = weekday_day
        day_source = "weekday"

    # Resolve the time (tonight defaults to 19:00 if unspecified).
    hm = _parse_time(low)
    if hm is None and re.search(r"\btonight\b", low):
        hm = (19, 0)
    title = _extract_title(text)

    if hm is None and _is_all_day(low):
        start = day.date()
        end = (day + timedelta(days=1)).date()
        return (title or "Event", start.isoformat(), end.isoformat())

    if hm is None:
        # Day/title may be clear but we won't invent a time.
        if title:
            return ("__need_time__", "", "")
        return None

    range_hm = _parse_time_range(low)
    if range_hm:
        hm, end_hm = range_hm

    hour, minute = hm
    start = day.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if day_source == "weekday" and start < now:
        start += timedelta(days=7)
        day = start

    # End time: "to/until/till 4pm" or "-4pm"; else +1 hour.
    end = start + timedelta(hours=1)
    if range_hm:
        end = day.replace(hour=end_hm[0], minute=end_hm[1], second=0, microsecond=0)
        if end <= start:
            end = start + timedelta(hours=1)
    elif m_end := re.search(r"\b(?:to|until|till|-)\s*(\d{1,2}(?![/-])(?::\d{2})?\s*(?:am|pm)?|noon|midnight)\b", low):
        end_hm = _parse_time(m_end.group(1) if any(x in m_end.group(1) for x in ("am", "pm", ":", "noon", "midnight")) else "at " + m_end.group(1))
        if end_hm:
            end = day.replace(hour=end_hm[0], minute=end_hm[1], second=0, microsecond=0)
            if end <= start:
                end = start + timedelta(hours=1)

    return (title or "Event", start.isoformat(), end.isoformat())
