"""Live Google Calendar connector for Jarvis V2."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    PERSONAL_READ_RECOVERY_ACTION,
    declare_failure_guidance,
    declare_outcome_unknown_failure,
    declare_retryable_personal_read_failure,
)
from jarvis_v2.agent.types import ApprovalArgumentResolution, RiskLevel, ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.v3_commands import (
    V3_CALENDAR_AUTH_FULL_ACCESS_COMMAND,
    V3_CALENDAR_AUTH_READONLY_COMMAND,
    V3_CALENDAR_AUTH_TERMINAL_LOCATION,
)


_SCOPES = ["https://www.googleapis.com/auth/calendar"]
_READONLY_SCOPES = [
    "https://www.googleapis.com/auth/calendar.events.readonly",
    "https://www.googleapis.com/auth/calendar.calendarlist.readonly",
]
_READONLY_AUTH_COMMAND = V3_CALENDAR_AUTH_READONLY_COMMAND
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
CALENDAR_INPUT_RECOVERY_ACTION = (
    "Correct the reported calendar input, then submit a new calendar request through the normal policy."
)
CALENDAR_MUTATION_SETUP_RECOVERY_ACTION = (
    "Run `setup check`, correct the reported Google Calendar setup issue, then submit a new approved calendar request."
)
CALENDAR_PRIMARY_TARGET_RECOVERY_ACTION = (
    "Run `list calendars`, then retry with an exact calendar_id."
)


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    return Path(raw or default).expanduser()


def _creds_file() -> Path:
    return _env_path("JARVIS_GOOGLE_CREDS", Path.home() / ".jarvis_v3" / "google_credentials.json")


def _token_file() -> Path:
    return _env_path("JARVIS_GOOGLE_TOKEN", Path.home() / ".jarvis_v3" / "google_token.json")


def _readonly_token_file() -> Path:
    return _env_path(
        "JARVIS_GOOGLE_READONLY_TOKEN",
        Path.home() / ".jarvis_v3" / "google_calendar_readonly_token.json",
    )


class GoogleCalendarReadonlyAuthError(RuntimeError):
    """Bounded read-lane authorization failure with no credential contents."""


def _calendar_id_arg(args: dict[str, Any]) -> str | None:
    """Return the explicit calendar target, defaulting only when it was omitted."""
    if "calendar_id" not in args:
        return "primary"
    value = args.get("calendar_id")
    if type(value) is not str:
        return None
    normalized = value.strip()
    return normalized or None


def _calendar_mutation_id_arg(args: dict[str, Any]) -> str | None:
    calendar_id = _calendar_id_arg(args)
    if (
        calendar_id is not None
        and "calendar_id" in args
        and calendar_id.casefold() == "primary"
    ):
        return None
    return calendar_id


def _atomic_write_private_text(path: Path, text: str) -> None:
    """Atomically replace a private text file and durably sync its directory."""
    path = Path(path)
    parent = path.parent
    missing: list[Path] = []
    cursor = parent
    while not cursor.exists():
        missing.append(cursor)
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    for directory in reversed(missing):
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
        if not directory.is_dir():
            raise NotADirectoryError(directory)

    fd = -1
    temp_path: Path | None = None
    try:
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
        temp_path = Path(temp_name)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    base = {
        "calls_model": False,
        "calls_external_service": True,
        "executes_tools": False,
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
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    base.update(extra)
    return base


def _mutation_metadata(**extra: Any) -> dict[str, Any]:
    return _safe_metadata(reads_personal_data=False, **extra)


def _known_no_change_failure(
    tool_name: str,
    output: str,
    metadata: dict[str, Any],
    *,
    action: str = CALENDAR_INPUT_RECOVERY_ACTION,
    commands: tuple[str, ...] = (),
) -> ToolResult:
    """Declare an actionable failure that happened before any calendar mutation."""

    public_output = output.strip()
    if action not in public_output:
        public_output = f"{public_output} {action}"
    result_metadata = dict(metadata)
    result_metadata.update(
        {
            "outcome_known": True,
            "outcome_unknown": False,
            "execution_outcome_unknown": False,
            "side_effect_possible": False,
            "retry_safe": True,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
        }
    )
    return ToolResult(
        tool_name,
        False,
        public_output,
        declare_failure_guidance(
            result_metadata,
            output=public_output,
            action=action,
            commands=commands,
        ),
    )


def _calendar_read_boundaries(*, calls_external_service: bool = True) -> dict[str, bool]:
    return {
        "calls_model": False,
        "calls_external_service": calls_external_service,
        "reads_personal_data": calls_external_service,
        "reads_private_data": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "controls_computer": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _calendar_mutation_boundaries(*, service_contacted: bool, mutation_attempted: bool) -> dict[str, bool]:
    return {
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
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _calendar_event_row(ev: dict[str, Any], index: int) -> dict[str, Any]:
    start = ev.get("start", {})
    end = ev.get("end", {})
    start_raw = start.get("dateTime") or start.get("date", "")
    end_raw = end.get("dateTime") or end.get("date", "")
    return {
        "index": index,
        "event_id": _display_text(ev.get("id"), limit=120),
        "summary": _display_text(ev.get("summary", "(no title)"), limit=160),
        "location": _display_text(ev.get("location", ""), limit=120),
        "start": _short_dt(start_raw),
        "end": _short_dt(end_raw),
        "all_day": _is_date_only(str(start_raw)),
    }


def _calendar_row(cal: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "index": index,
        "calendar_id": _display_text(cal.get("id"), limit=120),
        "summary": _display_text(cal.get("summary"), limit=120),
        "access_role": _display_text(cal.get("accessRole", ""), limit=40),
        "primary": bool(cal.get("primary")),
    }


def _calendar_list_handoff(*, rows: list[dict[str, Any]] | None, status: str, reason: str = "") -> dict[str, Any]:
    rows = rows or []
    primary = next((row for row in rows if row.get("primary")), None)
    return {
        "calendar_list_handoff": {
            "source": "list_calendars",
            "ready_for_operator": True,
            "state_changed": False,
            "changed": [],
            "content_in_handoff": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "status": status,
            "reason": reason,
            "calendar_count": len(rows),
            "calendar_ids": [row["calendar_id"] for row in rows if row.get("calendar_id")],
            "primary_calendar_id": primary.get("calendar_id") if primary else "",
            "calendars": rows,
            "content_in_metadata": False,
            "next_safe_command": "list calendars",
            "boundaries": _calendar_read_boundaries(calls_external_service=True),
        }
    }


def _calendar_events_handoff(
    *,
    calendar_id: str,
    label: str,
    range_name: str,
    max_results: int,
    rows: list[dict[str, Any]] | None,
    status: str,
    reason: str = "",
    calls_external_service: bool = True,
) -> dict[str, Any]:
    rows = rows or []
    return {
        "calendar_events_handoff": {
            "source": "list_events",
            "ready_for_operator": True,
            "state_changed": False,
            "changed": [],
            "content_in_handoff": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "calendar_id": _display_text(calendar_id, limit=120),
            "range": range_name,
            "label": label,
            "max_results": max_results,
            "status": status,
            "reason": reason,
            "event_count": len(rows),
            "event_ids": [row["event_id"] for row in rows if row.get("event_id")],
            "events": rows,
            "content_in_metadata": False,
            "next_safe_command": f"what's on my calendar {label}" if label != "upcoming" else "what's on my calendar",
            "boundaries": _calendar_read_boundaries(calls_external_service=calls_external_service),
        }
    }


def _availability_handoff(
    *,
    calendar_id: str,
    label: str,
    start_iso: str,
    end_iso: str,
    rows: list[dict[str, Any]] | None,
    status: str,
    reason: str = "",
) -> dict[str, Any]:
    rows = rows or []
    return {
        "calendar_availability_handoff": {
            "source": "check_availability",
            "ready_for_operator": True,
            "state_changed": False,
            "changed": [],
            "content_in_handoff": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "calendar_id": _display_text(calendar_id, limit=120),
            "label": _display_text(label, limit=120),
            "start_iso": _display_text(start_iso, limit=120),
            "end_iso": _display_text(end_iso, limit=120),
            "status": status,
            "reason": reason,
            "free": status == "free",
            "conflict_count": len(rows),
            "conflict_ids": [row["event_id"] for row in rows if row.get("event_id")],
            "conflicts": rows,
            "content_in_metadata": False,
            "next_safe_command": f"am I free {label}" if label else "am I free",
            "boundaries": _calendar_read_boundaries(calls_external_service=True),
        }
    }


def _mutation_handoff(
    *,
    tool_name: str,
    calendar_id: str = "primary",
    event_id: str = "",
    title: str = "",
    start: str = "",
    end: str = "",
    description: str = "",
    location: str = "",
    status: str,
    reason: str = "",
    service_contacted: bool = False,
    mutation_attempted: bool = False,
    exception_type: str = "",
) -> dict[str, Any]:
    changed = [f"{tool_name}_mutation_attempt"] if mutation_attempted else []
    if status == "outcome_unknown":
        next_safe_commands = [
            "what is on my calendar <date>",
            "recent tool runs",
        ]
    elif status in {"failed", "unavailable"}:
        next_safe_commands = ["setup check"]
    elif status == "ok":
        next_safe_commands = ["recent tool runs"]
    else:
        next_safe_commands = ["safe next actions"]
    handoff = {
        "source": tool_name,
        "status": status,
        "reason": reason,
        "ready_for_operator": True,
        "state_changed": mutation_attempted,
        "changed": changed,
        "calendar_id": _display_text(calendar_id, limit=120),
        "event_id": _display_text(event_id, limit=120),
        "title": _display_text(title, limit=160),
        "title_chars": len(title or ""),
        "start": _display_text(start, limit=120),
        "end": _display_text(end, limit=120),
        "description_chars": len(description or ""),
        "description_in_metadata": False,
        "location": _display_text(location, limit=120),
        "location_chars": len(location or ""),
        "content_in_metadata": False,
        "content_in_handoff": False,
        "service_contacted": service_contacted,
        "mutation_attempted": mutation_attempted,
        "approval_required_before_execution": True,
        "manual_review_required": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "exception_type": exception_type,
        "next_safe_command": next_safe_commands[0],
        "next_safe_commands": next_safe_commands,
        "next_safe_command_count": len(next_safe_commands),
        "boundaries": _calendar_mutation_boundaries(
            service_contacted=service_contacted,
            mutation_attempted=mutation_attempted,
        ),
    }
    return {
        f"{tool_name}_handoff_ready": True,
        "calendar_mutation_handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": mutation_attempted,
        "changed": changed,
        "content_in_handoff": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        f"{tool_name}_handoff": handoff,
        "calendar_mutation_handoff": handoff,
    }


def _calendar_mutation_outcome_unknown_guidance(tool_name: str) -> str:
    action = {
        "create_event": "create",
        "update_event": "update",
        "delete_event": "delete",
    }.get(tool_name, "change")
    return (
        f"Calendar {action} outcome is unknown; it may already have happened. Check "
        "`what is on my calendar <date>` and `recent tool runs`. Do not repeat automatically. "
        f"Submit a fresh approved {action} only after inspecting the calendar."
    )


def _short_raw(value: Any, *, limit: int = 80) -> str:
    text = "" if value is None else str(value).strip()
    text = " ".join(text.split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _display_text(value: Any, *, limit: int = 160) -> str:
    text = "" if value is None else str(value).strip()
    text = " ".join(text.split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _has_local_path(value: Any) -> bool:
    return bool(LOCAL_PATH_RE.search(str(value or "")))


def _bounded_int(value: Any, default: int, low: int, high: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _raw_int_metadata(value: Any, *, key: str, sanitized: int) -> dict[str, Any]:
    if value is None:
        return {key: sanitized}
    if isinstance(value, bool):
        return {key: sanitized, f"raw_{key}": str(value)}
    try:
        int(value)
    except (TypeError, ValueError):
        return {key: sanitized, f"raw_{key}": _short_raw(value)}
    return {key: sanitized}


def _calendar_error(e: Exception, *, mutation: bool = False) -> str:
    if isinstance(e, GoogleCalendarReadonlyAuthError):
        return (
            "Google Calendar read-only access is not connected or needs renewal. "
            f"The operator must run `{_READONLY_AUTH_COMMAND}` "
            f"{V3_CALENDAR_AUTH_TERMINAL_LOCATION}, then retry the read."
        )
    text = str(e).lower()
    if "google api packages not installed" in text:
        return "Google Calendar support is not installed on this Mac. Run setup check for the required Google packages."
    if "google credentials not found" in text:
        return "Google Calendar is not connected yet. Add Google credentials and run setup first."
    if mutation and "google full-access token not found" in text:
        return (
            "Google Calendar mutation access is not connected. The operator must run "
            f"`{V3_CALENDAR_AUTH_FULL_ACCESS_COMMAND}` {V3_CALENDAR_AUTH_TERMINAL_LOCATION}, "
            "review the full Calendar access request, then submit a new approved mutation."
        )
    if "invalid_grant" in text or type(e).__name__ == "RefreshError":
        # The stored OAuth token expired/was revoked AND the refresh token is
        # dead, so no retry will ever succeed until the operator re-consents. Found
        # 2026-07-06: the generic "having trouble" message hid a token that
        # had been dead since 06-17, silently degrading three weeks of
        # morning briefs. Name the exact fix instead of suggesting a retry.
        auth_command = (
            V3_CALENDAR_AUTH_FULL_ACCESS_COMMAND
            if mutation
            else V3_CALENDAR_AUTH_READONLY_COMMAND
        )
        return (
            "Google Calendar login has expired and cannot auto-refresh (invalid_grant). "
            f"Retrying will not help. The operator must run `{auth_command}` "
            f"{V3_CALENDAR_AUTH_TERMINAL_LOCATION} to sign in again, then retry this command."
        )
    auth_command = (
        V3_CALENDAR_AUTH_FULL_ACCESS_COMMAND
        if mutation
        else V3_CALENDAR_AUTH_READONLY_COMMAND
    )
    return (
        "Google Calendar is having trouble right now. Check network access to Google Calendar, "
        "run setup check, and if this keeps happening run "
        f"`{auth_command}` {V3_CALENDAR_AUTH_TERMINAL_LOCATION}, then retry."
    )


def _get_service():
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError:
        raise RuntimeError("Google API packages not installed. Run: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")

    token_file = _token_file()
    if not token_file.is_file():
        raise RuntimeError("Google full-access token not found. Run the interactive full-access reauth first.")
    creds = Credentials.from_authorized_user_file(str(token_file), _SCOPES)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            raise RuntimeError("Google full-access token is invalid. Run the interactive full-access reauth first.")
    return build("calendar", "v3", credentials=creds)


def _get_readonly_service():
    """Build the Calendar read lane without launching OAuth or using the write token."""

    token_file = _readonly_token_file()
    try:
        if token_file.is_symlink():
            raise GoogleCalendarReadonlyAuthError("read-only token cannot be a symlink")
        token_stat = token_file.stat()
    except FileNotFoundError as exc:
        raise GoogleCalendarReadonlyAuthError("read-only token is missing") from exc
    except OSError as exc:
        raise GoogleCalendarReadonlyAuthError("read-only token is unavailable") from exc
    if not stat.S_ISREG(token_stat.st_mode) or stat.S_IMODE(token_stat.st_mode) & 0o077:
        raise GoogleCalendarReadonlyAuthError("read-only token must be an owner-only regular file")

    try:
        payload = json.loads(token_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GoogleCalendarReadonlyAuthError("read-only token is malformed") from exc
    stored_scopes = payload.get("scopes") if isinstance(payload, dict) else None
    if (
        not isinstance(stored_scopes, list)
        or any(type(scope) is not str for scope in stored_scopes)
        or len(stored_scopes) != len(_READONLY_SCOPES)
        or set(stored_scopes) != set(_READONLY_SCOPES)
    ):
        raise GoogleCalendarReadonlyAuthError("read-only token scope mismatch")

    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Google API packages not installed. Run: pip install google-auth "
            "google-auth-oauthlib google-auth-httplib2 google-api-python-client"
        ) from exc

    try:
        creds = Credentials.from_authorized_user_file(str(token_file), _READONLY_SCOPES)
        if not creds.valid:
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                raise GoogleCalendarReadonlyAuthError("read-only token is not refreshable")
        if not creds.valid:
            raise GoogleCalendarReadonlyAuthError("read-only token did not become valid")
    except GoogleCalendarReadonlyAuthError:
        raise
    except Exception as exc:
        raise GoogleCalendarReadonlyAuthError("read-only token refresh failed") from exc
    return build("calendar", "v3", credentials=creds)


def _concrete_primary_calendar_id() -> str:
    """Resolve Google's mutable ``primary`` alias to one concrete calendar ID."""
    result = _get_service().calendarList().list().execute()
    items = result.get("items", []) if isinstance(result, dict) else []
    primary_ids = {
        item.get("id").strip()
        for item in items
        if isinstance(item, dict)
        and item.get("primary") is True
        and type(item.get("id")) is str
        and item.get("id").strip()
    }
    if len(primary_ids) != 1:
        raise RuntimeError("Google Calendar primary target could not be resolved uniquely")
    calendar_id = next(iter(primary_ids))
    if calendar_id.casefold() == "primary":
        raise RuntimeError("Google Calendar returned an unresolved primary alias")
    return calendar_id


def _short_dt(dt_str: str) -> str:
    if not dt_str:
        return ""
    try:
        if "T" in dt_str:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            return dt.astimezone().strftime("%a %b %d %H:%M")
        return dt_str
    except Exception:
        return dt_str


def _is_date_only(value: str) -> bool:
    return bool(value and "T" not in value and len(value) == 10)


def _event_time_fields(start_str: str, end_str: str = "") -> dict[str, Any]:
    if _is_date_only(start_str):
        if not end_str:
            try:
                end_str = (datetime.fromisoformat(start_str) + timedelta(days=1)).date().isoformat()
            except ValueError:
                end_str = start_str
        return {
            "start": {"date": start_str},
            "end": {"date": end_str},
        }
    if not end_str:
        try:
            end_str = (datetime.fromisoformat(start_str.replace("Z", "+00:00")) + timedelta(hours=1)).isoformat()
        except ValueError:
            end_str = start_str
    return {
        "start": {"dateTime": start_str, "timeZone": "Asia/Seoul"},
        "end": {"dateTime": end_str, "timeZone": "Asia/Seoul"},
    }


_WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
_DAYPARTS = {"morning": (8, 12), "afternoon": (12, 18), "evening": (18, 22)}
_DAYPART_ALIASES = {"night": "evening"}


def _weekday_range(which: str, now: datetime) -> tuple[datetime, datetime, str] | None:
    normalized = (which or "").strip().lower().replace("_", " ")
    match = re.fullmatch(r"(?P<next>next\s+)?(?P<day>monday|tuesday|wednesday|thursday|friday|saturday|sunday)", normalized)
    if not match:
        return None
    day_name = match.group("day")
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    ahead = (_WEEKDAYS[day_name] - today.weekday()) % 7
    if match.group("next") and ahead == 0:
        ahead = 7
    start = today + timedelta(days=ahead)
    label = f"next {day_name.capitalize()}" if match.group("next") else day_name.capitalize()
    return start, start + timedelta(days=1), label


def _weekend_range(which: str, now: datetime) -> tuple[datetime, datetime, str] | None:
    normalized = (which or "").strip().lower().replace("_", " ")
    if normalized not in {"weekend", "this weekend", "next weekend"}:
        return None
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if today.weekday() == 6:
        start = today - timedelta(days=1)
    else:
        start = today + timedelta(days=(5 - today.weekday()))
    if normalized == "next weekend":
        start += timedelta(days=7)
    label = "next weekend" if normalized == "next weekend" else "this weekend"
    return start, start + timedelta(days=2), label


def _add_months(start: datetime, months: int) -> datetime:
    month_index = start.month - 1 + months
    year = start.year + month_index // 12
    month = month_index % 12 + 1
    return start.replace(year=year, month=month, day=1)


def _month_range(which: str, now: datetime) -> tuple[datetime, datetime, str] | None:
    normalized = (which or "").strip().lower().replace("_", " ")
    if normalized not in {"month", "this month", "next month"}:
        return None
    this_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    start = _add_months(this_start, 1) if normalized == "next month" else this_start
    label = "next month" if normalized == "next month" else "this month"
    return start, _add_months(start, 1), label


def _year_range(which: str, now: datetime) -> tuple[datetime, datetime, str] | None:
    normalized = (which or "").strip().lower().replace("_", " ")
    if normalized not in {"year", "this year", "next year"}:
        return None
    this_start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    start = this_start.replace(year=this_start.year + 1) if normalized == "next year" else this_start
    label = "next year" if normalized == "next year" else "this year"
    return start, start.replace(year=start.year + 1), label


def _absolute_date_range(which: str, now: datetime) -> tuple[datetime, datetime, str] | None:
    normalized = unicodedata.normalize("NFKC", str(which or "")).strip().casefold()
    if not normalized or len(normalized) > 64 or any(ord(char) < 32 for char in normalized):
        return None
    if LOCAL_PATH_RE.search(normalized) or "\\" in normalized:
        return None
    normalized = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", normalized)
    normalized = re.sub(r"\bsept\b", "sep", normalized)
    normalized = re.sub(r",\s*", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    formats = (
        ("%Y-%m-%d", True),
        ("%Y/%m/%d", True),
        ("%m/%d/%Y", True),
        ("%m/%d", False),
        ("%m-%d-%Y", True),
        ("%m-%d", False),
        ("%B %d %Y", True),
        ("%b %d %Y", True),
        ("%B %d", False),
        ("%b %d", False),
        ("%d %B %Y", True),
        ("%d %b %Y", True),
        ("%d %B", False),
        ("%d %b", False),
    )
    for date_format, has_year in formats:
        try:
            parsed = datetime.strptime(normalized, date_format)
            year = parsed.year if has_year else now.year
            start = now.replace(
                year=year,
                month=parsed.month,
                day=parsed.day,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
        except ValueError:
            continue
        label = f"{start.strftime('%B')} {start.day}, {start.year}"
        return start, start + timedelta(days=1), label
    return None


def _apply_daypart(start: datetime, label: str, daypart: str | None) -> tuple[datetime, datetime, str]:
    if not daypart:
        return start, start + timedelta(days=1), label
    start_hour, end_hour = _DAYPARTS[daypart]
    return start.replace(hour=start_hour), start.replace(hour=end_hour), f"{label} {daypart}"


def _range_window(which: str, now: datetime) -> tuple[datetime, datetime, str] | None:
    normalized = (which or "").strip().lower().replace("_", " ")
    daypart = None
    for candidate in (*_DAYPARTS.keys(), *_DAYPART_ALIASES.keys()):
        if re.search(rf"\b{candidate}$", normalized):
            daypart = _DAYPART_ALIASES.get(candidate, candidate)
            normalized = re.sub(rf"\s*{candidate}$", "", normalized).strip()
            break
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if normalized in {"today", "day", "this"}:
        return _apply_daypart(today, "today", daypart)
    if normalized in {"tonight"}:
        return _apply_daypart(today, "today", "evening")
    if normalized == "tomorrow":
        return _apply_daypart(today + timedelta(days=1), "tomorrow", daypart)
    weekend = _weekend_range(normalized, now)
    if weekend:
        return weekend
    month = _month_range(normalized, now)
    if month:
        return month
    year = _year_range(normalized, now)
    if year:
        return year
    weekday = _weekday_range(normalized, now)
    if weekday:
        start, _end, label = weekday
        return _apply_daypart(start, label, daypart)
    absolute_date = _absolute_date_range(normalized, now)
    if absolute_date:
        return absolute_date
    return None


def _range_bounds(which: str) -> tuple[str, str] | None:
    """Return (timeMin, timeMax) RFC3339 strings for a named range, or None for
    the default 'upcoming' view. Uses the Mac's local timezone."""
    which = (which or "").strip().lower()
    now = datetime.now().astimezone()
    window = _range_window(which, now)
    if window:
        start, end, _label = window
        return start.isoformat(), end.isoformat()
    if which in {"week", "this week"}:
        return now.isoformat(), (now + timedelta(days=7)).isoformat()
    if which in {"next_week", "next week"}:
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start = today + timedelta(days=(7 - today.weekday()))
        return start.isoformat(), (start + timedelta(days=7)).isoformat()
    return None


def _range_label(which: str) -> str:
    window = _range_window(which, datetime.now().astimezone())
    if window:
        return window[2]
    return {
        "today": "today",
        "tomorrow": "tomorrow",
        "week": "this week",
        "this week": "this week",
        "next_week": "next week",
        "next week": "next week",
    }.get((which or "").strip().lower(), "upcoming")


def _availability_window(text: str, now: datetime | None = None) -> tuple[str, str, str]:
    """Parse an availability question into (start_iso, end_iso, label)."""
    import re
    from jarvis_v2.tools.nl_datetime import _parse_time

    now = now or datetime.now().astimezone()
    low = " ".join(text.lower().split())

    weekend = _weekend_range(
        "next weekend" if re.search(r"\bnext\s+weekend\b", low) else "this weekend" if re.search(r"\b(?:this\s+)?weekend\b", low) else "",
        now,
    )
    if weekend:
        start, end, label = weekend
        return start.isoformat(), end.isoformat(), label
    if re.search(r"\bnext\s+month\b", low):
        month = _month_range("next month", now)
        if month:
            start, end, label = month
            return start.isoformat(), end.isoformat(), label
    if re.search(r"\bthis\s+month\b", low):
        this_month = _month_range("this month", now)
        if this_month:
            _start, end, label = this_month
            return now.isoformat(), end.isoformat(), label
    if re.search(r"\bnext\s+year\b", low):
        year = _year_range("next year", now)
        if year:
            start, end, label = year
            return start.isoformat(), end.isoformat(), label
    if re.search(r"\bthis\s+year\b", low):
        this_year = _year_range("this year", now)
        if this_year:
            _start, end, label = this_year
            return now.isoformat(), end.isoformat(), label
    if re.search(r"\bnext\s+week\b", low):
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start = today + timedelta(days=(7 - today.weekday()))
        return start.isoformat(), (start + timedelta(days=7)).isoformat(), "next week"
    if re.search(r"\bthis\s+week\b|\bweek\b", low):
        return now.isoformat(), (now + timedelta(days=7)).isoformat(), "this week"

    day = now
    day_label = "today"
    if re.search(r"\btomorrow\b", low):
        day = now + timedelta(days=1)
        day_label = "tomorrow"
    else:
        for name, idx in _WEEKDAYS.items():
            if re.search(rf"\b{name}\b", low):
                ahead = (idx - now.weekday()) % 7
                if re.search(rf"\bnext\s+{name}\b", low) and ahead == 0:
                    ahead = 7
                day = now + timedelta(days=ahead)
                day_label = f"next {name.capitalize()}" if re.search(rf"\bnext\s+{name}\b", low) else name.capitalize()
                break
    midnight = day.replace(hour=0, minute=0, second=0, microsecond=0)

    hm = _parse_time(low)
    if hm is not None:
        start = midnight.replace(hour=hm[0], minute=hm[1])
        dur = re.search(r"\bfor\s+(\d+)\s*(hours?|hrs?|minutes?|mins?)\b", low)
        if dur:
            n = int(dur.group(1))
            end = start + (timedelta(hours=n) if dur.group(2).startswith(("hour", "hr")) else timedelta(minutes=n))
        else:
            end = start + timedelta(hours=1)
        return start.isoformat(), end.isoformat(), f"{day_label} at {start.strftime('%H:%M')}"
    if re.search(r"\bmorning\b", low):
        return midnight.replace(hour=8).isoformat(), midnight.replace(hour=12).isoformat(), f"{day_label} morning"
    if re.search(r"\bafternoon\b", low):
        return midnight.replace(hour=12).isoformat(), midnight.replace(hour=18).isoformat(), f"{day_label} afternoon"
    if re.search(r"\b(evening|tonight|night)\b", low):
        return midnight.replace(hour=18).isoformat(), midnight.replace(hour=22).isoformat(), f"{day_label} evening"
    # Whole waking day.
    return midnight.replace(hour=8).isoformat(), midnight.replace(hour=22).isoformat(), day_label


def make_calendar_tools(config: JarvisConfig):
    def _calendar_target_refusal(tool_name: str) -> ToolResult:
        return _known_no_change_failure(
            tool_name,
            "calendar_id must be a non-empty calendar ID when provided. Nothing was sent to Google Calendar.",
            _mutation_metadata(
                calls_external_service=False,
                reason="invalid_calendar_id",
                requires_confirmation=False,
                executed_handler=False,
                handler_invoked=False,
            ),
        )

    def _calendar_primary_resolution_refusal(tool_name: str) -> ToolResult:
        return _known_no_change_failure(
            tool_name,
            "I couldn't bind the current primary calendar to a concrete calendar ID, so no approval was queued. "
            f"{CALENDAR_PRIMARY_TARGET_RECOVERY_ACTION}",
            _safe_metadata(
                calls_external_service=True,
                reads_personal_data=True,
                reason="primary_calendar_unresolved",
                requires_confirmation=False,
                executed_handler=False,
                handler_invoked=False,
            ),
            action=CALENDAR_PRIMARY_TARGET_RECOVERY_ACTION,
            commands=("list calendars",),
        )

    def _resolve_calendar_mutation_approval(
        tool_name: str,
        args: dict[str, Any],
    ) -> ApprovalArgumentResolution | ToolResult:
        calendar_id = _calendar_id_arg(args)
        if calendar_id is None:
            return _calendar_target_refusal(tool_name)
        target_source = "explicit"
        if calendar_id.casefold() == "primary":
            try:
                calendar_id = _concrete_primary_calendar_id()
            except Exception:
                return _calendar_primary_resolution_refusal(tool_name)
            target_source = "resolved_primary"
        resolved = dict(args)
        resolved["calendar_id"] = calendar_id
        return ApprovalArgumentResolution(
            resolved,
            {
                "calendar_target_bound": True,
                "calendar_target_source": target_source,
                "calendar_id_display": _display_text(calendar_id, limit=120),
            },
        )

    def resolve_create_event_approval(args: dict[str, Any]) -> ApprovalArgumentResolution | ToolResult:
        return _resolve_calendar_mutation_approval("create_event", args)

    def resolve_update_event_approval(args: dict[str, Any]) -> ApprovalArgumentResolution | ToolResult:
        return _resolve_calendar_mutation_approval("update_event", args)

    def resolve_delete_event_approval(args: dict[str, Any]) -> ApprovalArgumentResolution | ToolResult:
        return _resolve_calendar_mutation_approval("delete_event", args)

    def list_calendars(args: dict[str, Any]) -> ToolResult:
        try:
            service = _get_readonly_service()
            result = service.calendarList().list().execute()
            items = result.get("items", [])
            rows = [_calendar_row(cal, index) for index, cal in enumerate(items, start=1)]
            lines = [f"Google Calendars ({len(items)}):"]
            for cal in items:
                primary = " ★" if cal.get("primary") else ""
                lines.append(f"  • {_display_text(cal.get('summary'), limit=120)}{primary}  [{_display_text(cal.get('accessRole', ''), limit=40)}]  id={_display_text(cal.get('id'), limit=120)}")
            return ToolResult(
                "list_calendars",
                True,
                "\n".join(lines),
                _safe_metadata(
                    count=len(rows),
                    **_calendar_list_handoff(rows=rows, status="ok"),
                ),
            )
        except Exception as e:
            failure_output = (
                f"{_calendar_error(e)} {PERSONAL_READ_RECOVERY_ACTION}"
            )
            return ToolResult(
                "list_calendars",
                False,
                failure_output,
                declare_retryable_personal_read_failure(
                    _safe_metadata(
                        count=0,
                        exception_type=type(e).__name__,
                        **_calendar_list_handoff(rows=[], status="unavailable", reason="fetch_error"),
                    ),
                    output=failure_output,
                    action=PERSONAL_READ_RECOVERY_ACTION,
                    commands=("setup check",),
                ),
            )

    def list_events(args: dict[str, Any]) -> ToolResult:
        calendar_id = _calendar_id_arg(args)
        if calendar_id is None:
            return _known_no_change_failure(
                "list_events",
                "calendar_id must be a non-empty calendar ID when provided.",
                _safe_metadata(
                    calls_external_service=False,
                    reads_personal_data=False,
                    reason="invalid_calendar_id",
                    count=0,
                ),
            )
        max_results = _bounded_int(args.get("max_results"), 10, 1, 50)
        max_results_metadata = _raw_int_metadata(args.get("max_results"), key="max_results", sanitized=max_results)
        which = str(args.get("range") or "").strip().lower()
        bounds = _range_bounds(which)
        if which not in {"", "upcoming"} and bounds is None:
            label = _display_text(which, limit=80)
            return _known_no_change_failure(
                "list_events",
                f"I couldn't understand the calendar range '{label}'. Try today, tomorrow, a weekday, or an exact date.",
                _safe_metadata(
                    count=0,
                    calendar_id=_display_text(calendar_id, limit=120),
                    range=label,
                    label=label,
                    reason="invalid_range",
                    calls_external_service=False,
                    reads_personal_data=False,
                    **max_results_metadata,
                    **_calendar_events_handoff(
                        calendar_id=calendar_id,
                        label=label,
                        range_name=label,
                        max_results=max_results,
                        rows=[],
                        status="invalid",
                        reason="invalid_range",
                        calls_external_service=False,
                    ),
                ),
            )
        try:
            service = _get_readonly_service()
            list_kwargs: dict[str, Any] = {
                "calendarId": calendar_id,
                "maxResults": max_results,
                "singleEvents": True,
                "orderBy": "startTime",
            }
            if bounds:
                list_kwargs["timeMin"], list_kwargs["timeMax"] = bounds
            else:
                list_kwargs["timeMin"] = datetime.now(timezone.utc).isoformat()
            result = service.events().list(**list_kwargs).execute()
            items = result.get("items", [])
            label = _range_label(which)
            if not items:
                return ToolResult(
                    "list_events",
                    True,
                    f"No events {label}.",
                    _safe_metadata(
                        count=0,
                        calendar_id=_display_text(calendar_id, limit=120),
                        range=which or "upcoming",
                        label=label,
                        **max_results_metadata,
                        **_calendar_events_handoff(
                            calendar_id=calendar_id,
                            label=label,
                            range_name=which or "upcoming",
                            max_results=max_results,
                            rows=[],
                            status="empty",
                            reason="no_events",
                        ),
                    ),
                )
            lines = [f"Events {label} ({len(items)}):"]
            rows = [_calendar_event_row(ev, index) for index, ev in enumerate(items, start=1)]
            for row in rows:
                start_str = row["start"]
                summary = row["summary"]
                location = f" @ {row['location']}" if row.get("location") else ""
                event_id = f"  [id={row['event_id']}]" if row.get("event_id") else ""
                lines.append(f"  • {start_str}  {summary}{location}{event_id}")
            return ToolResult(
                "list_events",
                True,
                "\n".join(lines),
                _safe_metadata(
                    count=len(rows),
                    calendar_id=_display_text(calendar_id, limit=120),
                    range=which or "upcoming",
                    label=label,
                    **max_results_metadata,
                    **_calendar_events_handoff(
                        calendar_id=calendar_id,
                        label=label,
                        range_name=which or "upcoming",
                        max_results=max_results,
                        rows=rows,
                        status="ok",
                    ),
                ),
            )
        except Exception as e:
            failure_output = (
                f"{_calendar_error(e)} {PERSONAL_READ_RECOVERY_ACTION}"
            )
            return ToolResult(
                "list_events",
                False,
                failure_output,
                declare_retryable_personal_read_failure(
                    _safe_metadata(
                        exception_type=type(e).__name__,
                        count=0,
                        calendar_id=_display_text(calendar_id, limit=120),
                        range=str(args.get("range") or "").strip().lower() or "upcoming",
                        label=_range_label(str(args.get("range") or "").strip().lower()),
                        reason="fetch_error",
                        **max_results_metadata,
                        **_calendar_events_handoff(
                            calendar_id=calendar_id,
                            label=_range_label(str(args.get("range") or "").strip().lower()),
                            range_name=str(args.get("range") or "").strip().lower() or "upcoming",
                            max_results=max_results,
                            rows=[],
                            status="unavailable",
                            reason="fetch_error",
                        ),
                    ),
                    output=failure_output,
                    action=PERSONAL_READ_RECOVERY_ACTION,
                    commands=("setup check",),
                ),
            )

    def create_event(args: dict[str, Any]) -> ToolResult:
        summary = ""
        start_str = ""
        end_str = ""
        calendar_id = "primary"
        description = ""
        location = ""
        service_contacted = False
        mutation_attempted = False
        try:
            summary = str(args.get("title") or args.get("summary") or "").strip()
            if not summary:
                return _known_no_change_failure(
                    "create_event",
                    "Event title is required.",
                    _mutation_metadata(
                        calls_external_service=False,
                        reason="missing_title",
                        **_mutation_handoff(
                            tool_name="create_event",
                            status="refused",
                            reason="missing_title",
                        ),
                    ),
                )
            start_str = str(args.get("start") or "").strip()
            end_str = str(args.get("end") or "").strip()
            if not start_str:
                return _known_no_change_failure(
                    "create_event",
                    "Start time is required (ISO format).",
                    _mutation_metadata(
                        calls_external_service=False,
                        reason="missing_start",
                        title=_display_text(summary) or None,
                        **_mutation_handoff(
                            tool_name="create_event",
                            title=summary,
                            status="refused",
                            reason="missing_start",
                        ),
                    ),
                )
            resolved_calendar_id = _calendar_mutation_id_arg(args)
            if resolved_calendar_id is None:
                return _calendar_target_refusal("create_event")
            calendar_id = resolved_calendar_id
            description = str(args.get("description") or "").strip()
            location = str(args.get("location") or "").strip()
            if any(_has_local_path(value) for value in (summary, description, location)):
                return _known_no_change_failure(
                    "create_event",
                    "Event title, description, and location cannot be local file paths.",
                    _mutation_metadata(
                        calls_external_service=False,
                        reason="invalid_event_text",
                        title=_display_text(summary) or None,
                        **_mutation_handoff(
                            tool_name="create_event",
                            calendar_id=calendar_id,
                            title=summary,
                            start=start_str,
                            end=end_str,
                            description=description,
                            location=location,
                            status="refused",
                            reason="invalid_event_text",
                        ),
                    ),
                )
            service_contacted = True
            service = _get_service()

            body: dict[str, Any] = {"summary": summary, **_event_time_fields(start_str, end_str)}
            if description:
                body["description"] = description
            if location:
                body["location"] = location

            mutation_attempted = True
            ev = service.events().insert(calendarId=calendar_id, body=body).execute()
            created_start = ev["start"].get("dateTime") or ev["start"].get("date", "")
            created_event_id = _display_text(ev.get("id"), limit=120)
            created_event_suffix = f"  [id={created_event_id}]" if created_event_id else ""
            return ToolResult(
                "create_event", True,
                f"Event created: {_display_text(ev.get('summary'))} on {_short_dt(created_start)}{created_event_suffix}",
                _mutation_metadata(
                    executes_side_effect=True,
                    external_side_effect=True,
                    requires_approval=True,
                    event_id=ev.get("id"),
                    **_mutation_handoff(
                        tool_name="create_event",
                        calendar_id=calendar_id,
                        event_id=str(ev.get("id") or ""),
                        title=summary,
                        start=start_str,
                        end=end_str,
                        description=description,
                        location=location,
                        status="ok",
                        service_contacted=service_contacted,
                        mutation_attempted=mutation_attempted,
                    ),
                ),
            )
        except Exception as e:
            outcome_unknown = mutation_attempted
            failure_output = (
                _calendar_mutation_outcome_unknown_guidance("create_event")
                if outcome_unknown
                else _calendar_error(e, mutation=True)
            )
            failure_metadata = _mutation_metadata(
                title=_display_text(summary) or None,
                mutation_attempted=mutation_attempted,
                service_contacted=service_contacted,
                executes_side_effect=mutation_attempted,
                external_side_effect=mutation_attempted,
                requires_approval=mutation_attempted,
                exception_type=type(e).__name__,
                outcome_known=not outcome_unknown,
                outcome_unknown=outcome_unknown,
                side_effect_possible=outcome_unknown,
                retry_safe=not outcome_unknown,
                automatic_retry_allowed=False,
                authorizes_retry=False,
                **_mutation_handoff(
                    tool_name="create_event",
                    calendar_id=calendar_id,
                    title=summary,
                    start=start_str,
                    end=end_str,
                    description=description,
                    location=location,
                    status="outcome_unknown" if outcome_unknown else "unavailable",
                    reason="calendar_mutation_outcome_unknown" if outcome_unknown else "fetch_error",
                    service_contacted=service_contacted,
                    mutation_attempted=mutation_attempted,
                    exception_type=type(e).__name__,
                ),
            )
            if outcome_unknown:
                failure_metadata = declare_outcome_unknown_failure(
                    failure_metadata,
                    output=failure_output,
                    commands=(
                        "what is on my calendar <date>",
                        "recent tool runs",
                    ),
                )
            else:
                return _known_no_change_failure(
                    "create_event",
                    failure_output,
                    failure_metadata,
                    action=CALENDAR_MUTATION_SETUP_RECOVERY_ACTION,
                    commands=("setup check",),
                )
            return ToolResult(
                "create_event",
                False,
                failure_output,
                failure_metadata,
            )

    def update_event(args: dict[str, Any]) -> ToolResult:
        event_id = ""
        calendar_id = "primary"
        summary = ""
        start_str = ""
        end_str = ""
        description = ""
        location = ""
        service_contacted = False
        mutation_attempted = False
        try:
            event_id = str(args.get("event_id") or "").strip()
            if not event_id:
                return _known_no_change_failure(
                    "update_event",
                    "event_id is required.",
                    _mutation_metadata(
                        calls_external_service=False,
                        reason="missing_event_id",
                        **_mutation_handoff(
                            tool_name="update_event",
                            calendar_id=calendar_id,
                            status="refused",
                            reason="missing_event_id",
                        ),
                    ),
                )
            if _has_local_path(event_id):
                return _known_no_change_failure(
                    "update_event",
                    "event_id cannot be a local file path.",
                    _mutation_metadata(
                        calls_external_service=False,
                        reason="invalid_event_id",
                        event_id="<local-path>",
                        **_mutation_handoff(
                            tool_name="update_event",
                            calendar_id=calendar_id,
                            event_id=event_id,
                            status="refused",
                            reason="invalid_event_id",
                        ),
                    ),
                )
            resolved_calendar_id = _calendar_mutation_id_arg(args)
            if resolved_calendar_id is None:
                return _calendar_target_refusal("update_event")
            calendar_id = resolved_calendar_id
            body: dict[str, Any] = {}
            summary = str(args.get("title") or args.get("summary") or "").strip()
            start_str = str(args.get("start") or "").strip()
            end_str = str(args.get("end") or "").strip()
            description = str(args.get("description") or "").strip()
            location = str(args.get("location") or "").strip()
            if any(_has_local_path(value) for value in (summary, description, location)):
                return _known_no_change_failure(
                    "update_event",
                    "Event title, description, and location cannot be local file paths.",
                    _mutation_metadata(
                        calls_external_service=False,
                        reason="invalid_event_text",
                        event_id=_display_text(event_id),
                        title=_display_text(summary) or None,
                        **_mutation_handoff(
                            tool_name="update_event",
                            calendar_id=calendar_id,
                            event_id=event_id,
                            title=summary,
                            start=start_str,
                            end=end_str,
                            description=description,
                            location=location,
                            status="refused",
                            reason="invalid_event_text",
                        ),
                    ),
                )
            if summary:
                body["summary"] = summary
            if start_str:
                body.update(_event_time_fields(start_str, end_str))
            elif end_str:
                return _known_no_change_failure(
                    "update_event",
                    "Start time is required when changing an end time.",
                    _mutation_metadata(
                        calls_external_service=False,
                        reason="missing_start_for_end_change",
                        event_id=_display_text(event_id),
                        **_mutation_handoff(
                            tool_name="update_event",
                            calendar_id=calendar_id,
                            event_id=event_id,
                            end=end_str,
                            status="refused",
                            reason="missing_start_for_end_change",
                        ),
                    ),
                )
            if description:
                body["description"] = description
            if location:
                body["location"] = location
            if not body:
                return _known_no_change_failure(
                    "update_event",
                    "Give Jarvis a title, start/end time, description, or location to change.",
                    _mutation_metadata(
                        calls_external_service=False,
                        reason="missing_changes",
                        event_id=_display_text(event_id),
                        **_mutation_handoff(
                            tool_name="update_event",
                            calendar_id=calendar_id,
                            event_id=event_id,
                            status="refused",
                            reason="missing_changes",
                        ),
                    ),
                )
            service_contacted = True
            service = _get_service()

            mutation_attempted = True
            ev = service.events().patch(calendarId=calendar_id, eventId=event_id, body=body).execute()
            start = ev.get("start", {})
            moved_to = _short_dt(start.get("dateTime") or start.get("date", ""))
            suffix = f" on {moved_to}" if moved_to else ""
            updated_event_id = _display_text(ev.get("id", event_id), limit=120)
            id_suffix = f"  [id={updated_event_id}]" if updated_event_id else ""
            return ToolResult(
                "update_event", True,
                f"Event updated: {_display_text(ev.get('summary', event_id))}{suffix}{id_suffix}",
                _mutation_metadata(
                    executes_side_effect=True,
                    external_side_effect=True,
                    requires_approval=True,
                    event_id=_display_text(ev.get("id", event_id)),
                    **_mutation_handoff(
                        tool_name="update_event",
                        calendar_id=calendar_id,
                        event_id=str(ev.get("id", event_id)),
                        title=summary,
                        start=start_str,
                        end=end_str,
                        description=description,
                        location=location,
                        status="ok",
                        service_contacted=service_contacted,
                        mutation_attempted=mutation_attempted,
                    ),
                ),
            )
        except Exception as e:
            outcome_unknown = mutation_attempted
            failure_output = (
                _calendar_mutation_outcome_unknown_guidance("update_event")
                if outcome_unknown
                else _calendar_error(e, mutation=True)
            )
            failure_metadata = _mutation_metadata(
                event_id=_display_text(event_id) or None,
                mutation_attempted=mutation_attempted,
                service_contacted=service_contacted,
                executes_side_effect=mutation_attempted,
                external_side_effect=mutation_attempted,
                requires_approval=mutation_attempted,
                exception_type=type(e).__name__,
                outcome_known=not outcome_unknown,
                outcome_unknown=outcome_unknown,
                side_effect_possible=outcome_unknown,
                retry_safe=not outcome_unknown,
                automatic_retry_allowed=False,
                authorizes_retry=False,
                **_mutation_handoff(
                    tool_name="update_event",
                    calendar_id=calendar_id,
                    event_id=event_id,
                    title=summary,
                    start=start_str,
                    end=end_str,
                    description=description,
                    location=location,
                    status="outcome_unknown" if outcome_unknown else "unavailable",
                    reason="calendar_mutation_outcome_unknown" if outcome_unknown else "fetch_error",
                    service_contacted=service_contacted,
                    mutation_attempted=mutation_attempted,
                    exception_type=type(e).__name__,
                ),
            )
            if outcome_unknown:
                failure_metadata = declare_outcome_unknown_failure(
                    failure_metadata,
                    output=failure_output,
                    commands=(
                        "what is on my calendar <date>",
                        "recent tool runs",
                    ),
                )
            else:
                return _known_no_change_failure(
                    "update_event",
                    failure_output,
                    failure_metadata,
                    action=CALENDAR_MUTATION_SETUP_RECOVERY_ACTION,
                    commands=("setup check",),
                )
            return ToolResult(
                "update_event",
                False,
                failure_output,
                failure_metadata,
            )

    def check_availability(args: dict[str, Any]) -> ToolResult:
        calendar_id = _calendar_id_arg(args)
        if calendar_id is None:
            return _known_no_change_failure(
                "check_availability",
                "calendar_id must be a non-empty calendar ID when provided.",
                _safe_metadata(
                    calls_external_service=False,
                    reason="invalid_calendar_id",
                    count=0,
                ),
            )
        try:
            text = str(args.get("text") or args.get("request") or "").strip()
            start_iso = str(args.get("start") or "").strip()
            end_iso = str(args.get("end") or "").strip()
            if not (start_iso and end_iso):
                start_iso, end_iso, label = _availability_window(text or "today")
            else:
                label = str(args.get("label") or "that time")
            service = _get_readonly_service()
            result = service.events().list(
                calendarId=calendar_id,
                timeMin=start_iso,
                timeMax=end_iso,
                singleEvents=True,
                orderBy="startTime",
            ).execute()
            items = result.get("items", [])
            # All-day events have no dateTime; treat only timed events as conflicts.
            conflicts = [ev for ev in items if (ev.get("start") or {}).get("dateTime")]
            rows = [_calendar_event_row(ev, index) for index, ev in enumerate(conflicts, start=1)]
            if not conflicts:
                return ToolResult(
                    "check_availability",
                    True,
                    f"You're free {label}.",
                    _safe_metadata(
                        count=0,
                        calendar_id=_display_text(calendar_id, limit=120),
                        label=_display_text(label, limit=120),
                        start_iso=_display_text(start_iso, limit=120),
                        end_iso=_display_text(end_iso, limit=120),
                        **_availability_handoff(
                            calendar_id=calendar_id,
                            label=label,
                            start_iso=start_iso,
                            end_iso=end_iso,
                            rows=[],
                            status="free",
                        ),
                    ),
                )
            lines = [f"You have {len(conflicts)} thing(s) {label}:"]
            for row in rows[:5]:
                lines.append(f"  • {row['start']}  {row['summary']}")
            return ToolResult(
                "check_availability",
                True,
                "\n".join(lines),
                _safe_metadata(
                    count=len(conflicts),
                    calendar_id=_display_text(calendar_id, limit=120),
                    label=_display_text(label, limit=120),
                    start_iso=_display_text(start_iso, limit=120),
                    end_iso=_display_text(end_iso, limit=120),
                    **_availability_handoff(
                        calendar_id=calendar_id,
                        label=label,
                        start_iso=start_iso,
                        end_iso=end_iso,
                        rows=rows,
                        status="busy",
                    ),
                ),
            )
        except Exception as e:
            failure_output = (
                f"{_calendar_error(e)} {PERSONAL_READ_RECOVERY_ACTION}"
            )
            return ToolResult(
                "check_availability",
                False,
                failure_output,
                declare_retryable_personal_read_failure(
                    _safe_metadata(
                        count=0,
                        exception_type=type(e).__name__,
                        **_availability_handoff(
                            calendar_id=calendar_id,
                            label=str(args.get("label") or "that time"),
                            start_iso=str(args.get("start") or ""),
                            end_iso=str(args.get("end") or ""),
                            rows=[],
                            status="unavailable",
                            reason="fetch_error",
                        ),
                    ),
                    output=failure_output,
                    action=PERSONAL_READ_RECOVERY_ACTION,
                    commands=("setup check",),
                ),
            )

    def delete_event(args: dict[str, Any]) -> ToolResult:
        event_id = ""
        calendar_id = "primary"
        service_contacted = False
        mutation_attempted = False
        try:
            event_id = str(args.get("event_id") or "").strip()
            if not event_id:
                return _known_no_change_failure(
                    "delete_event",
                    "event_id is required.",
                    _mutation_metadata(
                        calls_external_service=False,
                        reason="missing_event_id",
                        **_mutation_handoff(
                            tool_name="delete_event",
                            calendar_id=calendar_id,
                            status="refused",
                            reason="missing_event_id",
                        ),
                    ),
                )
            if _has_local_path(event_id):
                return _known_no_change_failure(
                    "delete_event",
                    "event_id cannot be a local file path.",
                    _mutation_metadata(
                        calls_external_service=False,
                        reason="invalid_event_id",
                        event_id="<local-path>",
                        **_mutation_handoff(
                            tool_name="delete_event",
                            calendar_id=calendar_id,
                            event_id=event_id,
                            status="refused",
                            reason="invalid_event_id",
                        ),
                    ),
                )
            resolved_calendar_id = _calendar_mutation_id_arg(args)
            if resolved_calendar_id is None:
                return _calendar_target_refusal("delete_event")
            calendar_id = resolved_calendar_id
            service_contacted = True
            service = _get_service()
            mutation_attempted = True
            service.events().delete(calendarId=calendar_id, eventId=event_id).execute()
            return ToolResult(
                "delete_event",
                True,
                f"Event {_display_text(event_id)} deleted.",
                _mutation_metadata(
                    executes_side_effect=True,
                    external_side_effect=True,
                    requires_approval=True,
                    event_id=_display_text(event_id),
                    **_mutation_handoff(
                        tool_name="delete_event",
                        calendar_id=calendar_id,
                        event_id=event_id,
                        status="ok",
                        service_contacted=service_contacted,
                        mutation_attempted=mutation_attempted,
                    ),
                ),
            )
        except Exception as e:
            outcome_unknown = mutation_attempted
            failure_output = (
                _calendar_mutation_outcome_unknown_guidance("delete_event")
                if outcome_unknown
                else _calendar_error(e, mutation=True)
            )
            failure_metadata = _mutation_metadata(
                event_id=_display_text(event_id) or None,
                mutation_attempted=mutation_attempted,
                service_contacted=service_contacted,
                executes_side_effect=mutation_attempted,
                external_side_effect=mutation_attempted,
                requires_approval=mutation_attempted,
                exception_type=type(e).__name__,
                outcome_known=not outcome_unknown,
                outcome_unknown=outcome_unknown,
                side_effect_possible=outcome_unknown,
                retry_safe=not outcome_unknown,
                automatic_retry_allowed=False,
                authorizes_retry=False,
                **_mutation_handoff(
                    tool_name="delete_event",
                    calendar_id=calendar_id,
                    event_id=event_id,
                    status="outcome_unknown" if outcome_unknown else "unavailable",
                    reason="calendar_mutation_outcome_unknown" if outcome_unknown else "fetch_error",
                    service_contacted=service_contacted,
                    mutation_attempted=mutation_attempted,
                    exception_type=type(e).__name__,
                ),
            )
            if outcome_unknown:
                failure_metadata = declare_outcome_unknown_failure(
                    failure_metadata,
                    output=failure_output,
                    commands=(
                        "what is on my calendar <date>",
                        "recent tool runs",
                    ),
                )
            else:
                return _known_no_change_failure(
                    "delete_event",
                    failure_output,
                    failure_metadata,
                    action=CALENDAR_MUTATION_SETUP_RECOVERY_ACTION,
                    commands=("setup check",),
                )
            return ToolResult(
                "delete_event",
                False,
                failure_output,
                failure_metadata,
            )

    from jarvis_v2.tools.registry import (
        TOOL_ARGUMENT_CONTRACT_VERSION,
        Tool,
        ToolArgumentContract,
        ToolArgumentSpec,
        ToolArgumentType,
    )

    string_type = frozenset({ToolArgumentType.STRING})
    integer_type = frozenset({ToolArgumentType.INTEGER})

    def argument_contract(
        *,
        required_strings: tuple[str, ...] = (),
        optional_strings: tuple[str, ...] = (),
        optional_integer_ranges: tuple[tuple[str, int, int], ...] = (),
    ) -> ToolArgumentContract:
        return ToolArgumentContract(
            version=TOOL_ARGUMENT_CONTRACT_VERSION,
            fields=tuple(
                [ToolArgumentSpec(name, string_type, True) for name in required_strings]
                + [ToolArgumentSpec(name, string_type, False) for name in optional_strings]
                + [
                    ToolArgumentSpec(name, integer_type, False, minimum, maximum)
                    for name, minimum, maximum in optional_integer_ranges
                ]
            ),
            allow_unknown=False,
        )

    create_initial_contract = argument_contract(
        required_strings=("start",),
        optional_strings=("title", "summary", "end", "calendar_id", "description", "location"),
    )
    create_approved_contract = argument_contract(
        required_strings=("start", "calendar_id"),
        optional_strings=("title", "summary", "end", "description", "location"),
    )
    update_initial_contract = argument_contract(
        required_strings=("event_id",),
        optional_strings=("title", "summary", "start", "end", "calendar_id", "description", "location"),
    )
    update_approved_contract = argument_contract(
        required_strings=("event_id", "calendar_id"),
        optional_strings=("title", "summary", "start", "end", "description", "location"),
    )
    delete_initial_contract = argument_contract(
        required_strings=("event_id",),
        optional_strings=("calendar_id",),
    )
    delete_approved_contract = argument_contract(required_strings=("event_id", "calendar_id"))

    return [
        Tool(
            "list_calendars",
            "List all Google Calendars.",
            RiskLevel.LOCAL_SAFE,
            list_calendars,
            "personal",
            argument_contract=argument_contract(),
        ),
        Tool(
            "list_events",
            "List Google Calendar events. Args: range (today|tomorrow|week, default upcoming), calendar_id, max_results.",
            RiskLevel.LOCAL_SAFE,
            list_events,
            "personal",
            argument_contract=argument_contract(
                optional_strings=("range", "calendar_id"),
                optional_integer_ranges=(("max_results", 1, 50),),
            ),
        ),
        Tool(
            "check_availability",
            "Check if you're free at a time. Args: text (e.g. 'am I free tomorrow afternoon') or start/end ISO.",
            RiskLevel.LOCAL_SAFE,
            check_availability,
            "personal",
            argument_contract=argument_contract(
                optional_strings=("text", "request", "start", "end", "label", "calendar_id"),
            ),
        ),
        Tool(
            "create_event",
            "Create a Google Calendar event. Args: title, start (ISO), end (ISO), calendar_id, description, location.",
            RiskLevel.HIGH_RISK,
            create_event,
            "personal",
            argument_contract=create_initial_contract,
            approval_argument_resolver=resolve_create_event_approval,
            approval_argument_contract=create_approved_contract,
        ),
        Tool(
            "update_event",
            "Update or move a Google Calendar event by event_id. Args: event_id, title/summary, start, end, calendar_id, description, location.",
            RiskLevel.HIGH_RISK,
            update_event,
            "personal",
            argument_contract=update_initial_contract,
            approval_argument_resolver=resolve_update_event_approval,
            approval_argument_contract=update_approved_contract,
        ),
        Tool(
            "delete_event",
            "Delete a Google Calendar event by event_id.",
            RiskLevel.HIGH_RISK,
            delete_event,
            "personal",
            argument_contract=delete_initial_contract,
            approval_argument_resolver=resolve_delete_event_approval,
            approval_argument_contract=delete_approved_contract,
        ),
    ]
