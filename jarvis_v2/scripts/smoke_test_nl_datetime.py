"""Smoke tests for the natural-language event date/time parser."""

from __future__ import annotations

from datetime import datetime

from jarvis_v2.tools.nl_datetime import parse_event


# Fixed reference: Monday 2026-06-15 16:00 local.
NOW = datetime(2026, 6, 15, 16, 0).astimezone()
AFTER_JUNE_20 = datetime(2026, 6, 21, 9, 0).astimezone()


def _start(text: str) -> str:
    r = parse_event(text, NOW)
    if not r or r[0] == "__need_time__":
        raise SystemExit(f"expected a parsed event for {text!r}, got {r}")
    return r[1]


def test_tomorrow_noon() -> None:
    title, start, end = parse_event("schedule lunch tomorrow at noon", NOW)
    if title != "lunch" or not start.startswith("2026-06-16T12:00") or not end.startswith("2026-06-16T13:00"):
        raise SystemExit(f"tomorrow noon wrong: {title} {start} {end}")


def test_named_event_weekday_pm() -> None:
    title, start, _ = parse_event("create an event called dentist friday at 3pm", NOW)
    if title != "dentist" or not start.startswith("2026-06-19T15:00"):
        raise SystemExit(f"weekday pm wrong: {title} {start}")


def test_next_weekday_skips_one_week() -> None:
    title, start, _ = parse_event("create an event called dentist next friday at 3pm", NOW)
    if title != "dentist" or not start.startswith("2026-06-26T15:00"):
        raise SystemExit(f"next weekday should skip one week: {title} {start}")


def test_title_with_person() -> None:
    title, start, _ = parse_event("add meeting with sarah monday 10am", NOW)
    if title != "meeting with sarah" or not start.startswith("2026-06-22T10:00"):
        raise SystemExit(f"person title wrong: {title} {start}")


def test_same_weekday_past_time_rolls_forward() -> None:
    title, start, _ = parse_event("schedule standup monday at 3pm", NOW)
    if title != "standup" or not start.startswith("2026-06-22T15:00"):
        raise SystemExit(f"same weekday past time should roll forward: {title} {start}")


def test_explicit_range() -> None:
    title, start, end = parse_event("block 2pm to 4pm tomorrow for deep work", NOW)
    if not start.startswith("2026-06-16T14:00") or not end.startswith("2026-06-16T16:00"):
        raise SystemExit(f"range wrong: {start} {end}")
    if title != "deep work":
        raise SystemExit(f"range title wrong: {title}")


def test_range_infers_start_meridiem_from_end() -> None:
    for text in ["schedule dentist tomorrow 2 to 4pm", "schedule dentist tomorrow 2-4pm"]:
        title, start, end = parse_event(text, NOW)
        if title != "dentist" or not start.startswith("2026-06-16T14:00") or not end.startswith("2026-06-16T16:00"):
            raise SystemExit(f"range should infer start meridiem from end: {text!r} -> {title} {start} {end}")


def test_missing_time_needs_clarification() -> None:
    r = parse_event("add dinner tomorrow", NOW)
    if not r or r[0] != "__need_time__":
        raise SystemExit(f"missing time should ask for clarification: {r}")


def test_default_one_hour_duration() -> None:
    _, start, end = parse_event("schedule a call today at 5pm", NOW)
    if not start.startswith("2026-06-15T17:00") or not end.startswith("2026-06-15T18:00"):
        raise SystemExit(f"default duration wrong: {start} {end}")


def test_explicit_month_day() -> None:
    title, start, end = parse_event("schedule dentist on June 20 at 3pm", NOW)
    if title != "dentist" or not start.startswith("2026-06-20T15:00") or not end.startswith("2026-06-20T16:00"):
        raise SystemExit(f"explicit month day wrong: {title} {start} {end}")


def test_explicit_month_day_rolls_to_next_year() -> None:
    title, start, _ = parse_event("book renewal Jan 3 at 9am", NOW)
    if title != "renewal" or not start.startswith("2027-01-03T09:00"):
        raise SystemExit(f"past explicit month day should roll forward a year: {title} {start}")


def test_explicit_month_day_still_needs_time() -> None:
    r = parse_event("add checkup June 20", NOW)
    if not r or r[0] != "__need_time__":
        raise SystemExit(f"explicit date without time should ask for clarification: {r}")


def test_numeric_month_day() -> None:
    title, start, end = parse_event("schedule dentist 6/20 at 3pm", NOW)
    if title != "dentist" or not start.startswith("2026-06-20T15:00") or not end.startswith("2026-06-20T16:00"):
        raise SystemExit(f"numeric month/day wrong: {title} {start} {end}")


def test_dashed_numeric_date_with_year_not_time_range() -> None:
    title, start, end = parse_event("schedule dentist 06-20-2026 at 3pm", NOW)
    if title != "dentist" or not start.startswith("2026-06-20T15:00") or not end.startswith("2026-06-20T16:00"):
        raise SystemExit(f"dashed numeric date should not be parsed as a time range: {title} {start} {end}")


def test_numeric_month_day_rolls_to_next_year() -> None:
    title, start, _ = parse_event("book renewal 1/3 at 9am", NOW)
    if title != "renewal" or not start.startswith("2027-01-03T09:00"):
        raise SystemExit(f"past numeric month/day should roll forward a year: {title} {start}")


def test_yearless_june_20_rolls_after_date_passes() -> None:
    for text in ["schedule dentist on June 20 at 3pm", "schedule dentist 6/20 at 3pm"]:
        title, start, end = parse_event(text, AFTER_JUNE_20)
        if title != "dentist" or not start.startswith("2027-06-20T15:00") or not end.startswith("2027-06-20T16:00"):
            raise SystemExit(f"past yearless June 20 should roll forward a year: {text!r} -> {title} {start} {end}")


def test_invalid_explicit_month_day_does_not_fall_back_to_today() -> None:
    for text in ["schedule dentist on Feb 31 at 3pm", "book renewal Apr 31 at 9am", "schedule dentist 2/31 at 3pm"]:
        if parse_event(text, NOW) is not None:
            raise SystemExit(f"invalid explicit date should not become a fallback event: {text!r} -> {parse_event(text, NOW)}")


def test_all_day_event_returns_date_only_range() -> None:
    title, start, end = parse_event("schedule PTO all day tomorrow", NOW)
    if title != "PTO" or start != "2026-06-16" or end != "2026-06-17":
        raise SystemExit(f"all-day event should return date-only next-day range: {title} {start} {end}")


def test_all_day_event_without_title_uses_default() -> None:
    title, start, end = parse_event("book an appointment all day tomorrow", NOW)
    if title != "Event" or start != "2026-06-16" or end != "2026-06-17":
        raise SystemExit(f"all-day title fallback wrong: {title} {start} {end}")


def test_invalid_clock_time_needs_clarification() -> None:
    for text in ["schedule call tomorrow at 25:00", "schedule call tomorrow at 13pm", "schedule call tomorrow at 8:99"]:
        r = parse_event(text, NOW)
        if not r or r[0] != "__need_time__":
            raise SystemExit(f"invalid clock time should ask for clarification, not crash: {text!r} -> {r}")


def main() -> None:
    test_tomorrow_noon()
    test_named_event_weekday_pm()
    test_next_weekday_skips_one_week()
    test_title_with_person()
    test_same_weekday_past_time_rolls_forward()
    test_explicit_range()
    test_range_infers_start_meridiem_from_end()
    test_missing_time_needs_clarification()
    test_default_one_hour_duration()
    test_explicit_month_day()
    test_explicit_month_day_rolls_to_next_year()
    test_explicit_month_day_still_needs_time()
    test_numeric_month_day()
    test_dashed_numeric_date_with_year_not_time_range()
    test_numeric_month_day_rolls_to_next_year()
    test_yearless_june_20_rolls_after_date_passes()
    test_invalid_explicit_month_day_does_not_fall_back_to_today()
    test_all_day_event_returns_date_only_range()
    test_all_day_event_without_title_uses_default()
    test_invalid_clock_time_needs_clarification()
    print("NL datetime smoke passed")


if __name__ == "__main__":
    main()
