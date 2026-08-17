"""'On this day in history' for Jarvis V2 (Wikipedia feed, free, no key)."""

from __future__ import annotations

import json
import re
import urllib.request
from datetime import datetime
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    EXTERNAL_INFORMATION_RECOVERY_ACTION,
    declare_retryable_external_information_failure,
)
from jarvis_v2.agent.types import RiskLevel, ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.tools._http import http_get, http_get_json, friendly_http_error

LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
MAX_RAW_DATE_CHARS = 80
_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10,
    "november": 11, "nov": 11, "december": 12, "dec": 12,
}


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    base = {
        "calls_model": False,
        "calls_external_service": True,
        "calls_external_services": True,
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
    base["calls_external_services"] = base.get("calls_external_service") is True
    return base


def _history_boundaries(*, calls_external_service: bool = True) -> dict[str, bool]:
    return {
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


def _event_rows(events: list[dict[str, Any]], *, limit: int = 4) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events[:limit]:
        rows.append(
            {
                "year": event.get("year", ""),
                "text_preview": str(event.get("text") or "").strip()[:140],
            }
        )
    return rows


def _history_handoff(
    *,
    date_metadata: dict[str, Any],
    status: str,
    reason: str = "",
    events: list[dict[str, Any]] | None = None,
    exception_type: str = "",
    calls_external_service: bool = True,
) -> dict[str, Any]:
    rows = _event_rows(events or [])
    next_safe_command = "on this day"
    boundaries = _history_boundaries(calls_external_service=calls_external_service)
    handoff = {
        "source": "on_this_day",
        "history_handoff_ready": True,
        "handoff_ready": True,
        "ready_for_operator": True,
        "status": status,
        "reason": reason,
        "state_changed": False,
        "changed": [],
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "content_in_handoff": False,
        "month": date_metadata["month"],
        "day": date_metadata["day"],
        "date": date_metadata["date"],
        "date_source": date_metadata.get("source", ""),
        "raw_date": date_metadata.get("raw_date", ""),
        "local_path_date": date_metadata.get("local_path_date", False),
        "event_count": len(events or []),
        "preview_rows": rows,
        "preview_row_count": len(rows),
        "exception_type": exception_type,
        "content_in_metadata": False,
        "retry_safe": status in {"empty", "unavailable"},
        "next_safe_command": next_safe_command,
        "next_safe_commands": [next_safe_command],
        "next_safe_command_count": 1,
        "boundaries": boundaries,
    }
    return {
        "history_handoff_ready": True,
        "history_ready_for_operator": handoff["ready_for_operator"],
        "history_state_changed": handoff["state_changed"],
        "history_changed": handoff["changed"],
        "history_content_in_handoff": handoff["content_in_handoff"],
        "history_content_in_metadata": handoff["content_in_metadata"],
        "history_next_safe_command": handoff["next_safe_command"],
        "history_next_safe_commands": handoff["next_safe_commands"],
        "history_next_safe_command_count": handoff["next_safe_command_count"],
        "history_authorizes_execution": handoff["authorizes_execution"],
        "history_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "history_approval_granted": handoff["approval_granted"],
        "history_boundaries": boundaries,
        "history_handoff": handoff,
    }


def _fetch(month: int, day: int) -> dict:
    url = f"https://en.wikipedia.org/api/rest_v1/feed/onthisday/events/{month:02d}/{day:02d}"
    return http_get_json(url, headers={"User-Agent": "JarvisV2/1.0 (personal assistant)"})


def _history_recovery_message(prefix: str) -> str:
    return f"{prefix} Check network access to Wikipedia on-this-day, run setup check, then retry in a moment."


def _history_error_message(e: Exception) -> str:
    friendly = friendly_http_error(e, service="Wikipedia")
    lowered = friendly.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return _history_recovery_message("The history request timed out.")
    if friendly.startswith("I couldn't find"):
        return friendly
    if friendly.startswith("Error:") or "having trouble" in lowered or "try again" in lowered:
        return _history_recovery_message("Wikipedia history is having trouble right now.")
    return friendly


def _short_raw_date(value: Any, limit: int = MAX_RAW_DATE_CHARS) -> str:
    text = "" if value is None else str(value)
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "..."
    return text


def _valid_month_day(month: int, day: int) -> bool:
    try:
        datetime(2000, month, day)
    except ValueError:
        return False
    return True


def _extract_date_text(text: str) -> str:
    raw = " ".join(str(text or "").strip().split())
    if not raw:
        return ""
    low = raw.lower()
    patterns = [
        r"\bwhat happened (?:in history )?on (?P<date>[a-z0-9][a-z0-9 /.-]{1,30})(?: in history)?[\?\.!]*$",
        r"\bon this day (?P<date>[a-z0-9][a-z0-9 /.-]{1,30})[\?\.!]*$",
        r"^(?P<date>[a-z0-9][a-z0-9 /.-]{1,30}) in history[\?\.!]*$",
        r"^(?P<date>[a-z0-9][a-z0-9 /.-]{1,30}) history[\?\.!]*$",
        r"^history (?:on )?(?P<date>[a-z0-9][a-z0-9 /.-]{1,30})[\?\.!]*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, low)
        if match:
            value = match.group("date").strip()
            value = re.sub(r"\s+(?:in history|history)$", "", value).strip()
            return value
    return ""


_TODAY_SENTINEL_PHRASES = {
    "today", "today in history", "on this day", "this day",
    "what happened today", "what happened",
}
"""Text extracted from a query that means "no specific date given, use today" --
distinct from genuinely unparseable garbage. `_resolve_history_date` checks this
BEFORE treating a `_parse_month_day` None as an "invalid_date" refusal, so e.g.
"what happened on this day in history" (which residually extracts to "this day"
after the extractor strips the "in history" suffix) defaults to today instead of
wrongly refusing, matching how bare "on this day" already correctly behaves."""


def _parse_month_day(value: str) -> tuple[int, int, str] | None:
    text = " ".join(str(value or "").strip().lower().split())
    if not text or text in _TODAY_SENTINEL_PHRASES:
        return None
    month_names = "|".join(re.escape(name) for name in _MONTHS)
    match = re.search(rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", text)
    if match:
        month = _MONTHS[match.group(1)]
        day = int(match.group(2))
        if _valid_month_day(month, day):
            return month, day, f"{month:02d}-{day:02d}"
        return None
    match = re.search(r"\b(\d{1,2})[/-](\d{1,2})\b", text)
    if match:
        month = int(match.group(1))
        day = int(match.group(2))
        if _valid_month_day(month, day):
            return month, day, f"{month:02d}-{day:02d}"
    return None


def _resolve_history_date(args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    now = datetime.now()
    raw_candidate = str(args.get("date") or "").strip()
    raw_text = str(args.get("text") or "")
    if not raw_candidate:
        local_path_match = LOCAL_PATH_RE.search(raw_text)
        raw_candidate = local_path_match.group(0) if local_path_match else _extract_date_text(raw_text)
    if raw_candidate and raw_candidate.strip().lower() not in _TODAY_SENTINEL_PHRASES:
        local_path = bool(LOCAL_PATH_RE.search(raw_candidate))
        parsed = None if local_path else _parse_month_day(raw_candidate)
        metadata = {
            "month": parsed[0] if parsed else 0,
            "day": parsed[1] if parsed else 0,
            "date": parsed[2] if parsed else "",
            "source": "text",
            "raw_date": _short_raw_date(raw_candidate),
            "local_path_date": local_path,
        }
        if not parsed:
            return metadata, "invalid_date"
        return metadata, ""
    return {"month": now.month, "day": now.day, "date": now.strftime("%m-%d"), "source": "today", "raw_date": "", "local_path_date": False}, ""


def make_history_tools(config: JarvisConfig):
    def on_this_day(args: dict[str, Any]) -> ToolResult:
        date_metadata, refusal_reason = _resolve_history_date(args)
        if refusal_reason:
            return ToolResult(
                "on_this_day",
                False,
                "Tell me a month and day, like 'what happened on July 4' or 'December 25 in history'.",
                _safe_metadata(
                    **date_metadata,
                    calls_external_service=False,
                    calls_external_services=False,
                    reason=refusal_reason,
                    **_history_handoff(
                        date_metadata=date_metadata,
                        status="refused",
                        reason=refusal_reason,
                        calls_external_service=False,
                    ),
                ),
            )
        try:
            data = _fetch(int(date_metadata["month"]), int(date_metadata["day"]))
            events = data.get("events") or []
            if not events:
                return ToolResult(
                    "on_this_day",
                    True,
                    f"No events found for {date_metadata['date']}.",
                    _safe_metadata(
                        **date_metadata,
                        count=0,
                        **_history_handoff(date_metadata=date_metadata, status="empty", reason="no_events"),
                    ),
                )
            events = sorted(events, key=lambda e: e.get("year", 0), reverse=True)[:4]
            month_name = datetime(2000, int(date_metadata["month"]), int(date_metadata["day"])).strftime("%B %d")
            lines = [f"On this day ({month_name}):"]
            for e in events:
                lines.append(f"  • {e.get('year', '?')}: {e.get('text', '').strip()[:140]}")
            return ToolResult(
                "on_this_day",
                True,
                "\n".join(lines),
                _safe_metadata(
                    **date_metadata,
                    count=len(events),
                    **_history_handoff(date_metadata=date_metadata, status="ok", events=events),
                ),
            )
        except Exception as e:
            failure_output = (
                f"{_history_error_message(e)} "
                f"{EXTERNAL_INFORMATION_RECOVERY_ACTION}"
            )
            return ToolResult(
                "on_this_day",
                False,
                failure_output,
                declare_retryable_external_information_failure(
                    _safe_metadata(
                        **date_metadata,
                        exception_type=type(e).__name__,
                        **_history_handoff(
                            date_metadata=date_metadata,
                            status="unavailable",
                            reason="fetch_error",
                            exception_type=type(e).__name__,
                        ),
                    ),
                    output=failure_output,
                    action=EXTERNAL_INFORMATION_RECOVERY_ACTION,
                    commands=("setup check",),
                ),
            )

    from jarvis_v2.tools.registry import Tool, _tool_argument_contract
    return [
        Tool(
            "on_this_day",
            "What happened on this day in history (free). Args: optional text or date.",
            RiskLevel.LOCAL_SAFE,
            on_this_day,
            "personal",
            argument_contract=_tool_argument_contract(optional_strings=("text", "date")),
        ),
    ]
