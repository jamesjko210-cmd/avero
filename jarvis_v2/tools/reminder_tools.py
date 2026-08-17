"""Reminder/timer tool for the owner Telegram channel."""

from __future__ import annotations

import os
import re
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
    declare_failure_guidance,
    declare_retryable_personal_read_failure,
)
from jarvis_v2.agent.types import ApprovalArgumentResolution, RiskLevel, ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.automations.reminders import add_reminder_result, has_local_path, is_recurring_request, parse_when
from jarvis_v2.automations import reminders as _rem


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    base = {
        "calls_model": False,
        "calls_external_service": False,
        "executes_tools": False,
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
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    base.update(extra)
    return base


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _reminder_boundaries(*, reads_personal_data: bool = False, writes_files: bool = False) -> dict[str, bool]:
    # These flags describe the handler result after the registry approval gate.
    # The Tool registration remains the source of truth for required consent.
    return {
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


def _reminder_id_key(value: object) -> str:
    key = str(value or "").replace("-", "").strip().lower()
    return key if re.fullmatch(r"[0-9a-f]{32}", key) else ""


def _with_reminder_selectors(reminders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [_reminder_id_key(row.get("id")) for row in reminders]
    decorated: list[dict[str, Any]] = []
    for index, row in enumerate(reminders):
        key = keys[index]
        length = 8
        while key and length < len(key) and any(
            other_index != index and other.startswith(key[:length])
            for other_index, other in enumerate(keys)
            if other
        ):
            length += 1
        copy = dict(row)
        copy["selector"] = key[:length] if key else ""
        decorated.append(copy)
    return decorated


def _normalize_reminder_handoff(handoff: dict[str, Any]) -> dict[str, Any]:
    raw_next = handoff.get("next_commands") or []
    if isinstance(raw_next, dict):
        next_safe_commands = [str(value) for value in raw_next.values() if str(value or "").strip()]
    else:
        next_safe_commands = [str(value) for value in raw_next if str(value or "").strip()]
    handoff["handoff_ready"] = True
    handoff["next_safe_command"] = next_safe_commands[0] if next_safe_commands else ""
    handoff["next_safe_commands"] = list(next_safe_commands)
    handoff["next_safe_command_count"] = len(next_safe_commands)
    return handoff


def _reminder_handoff_metadata(handoff_key: str, handoff: dict[str, Any]) -> dict[str, Any]:
    _normalize_reminder_handoff(handoff)
    prefix = handoff_key.removesuffix("_handoff")
    return {
        f"{handoff_key}_ready": True,
        f"{prefix}_handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": _metadata_bool(handoff.get("state_changed")),
        "changed": list(handoff.get("changed") or []),
        "content_in_handoff": _metadata_bool(handoff.get("content_in_handoff")),
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "next_safe_command": handoff["next_safe_command"],
        "next_safe_commands": list(handoff["next_safe_commands"]),
        "next_safe_command_count": handoff["next_safe_command_count"],
        f"{prefix}_ready_for_operator": True,
        f"{prefix}_state_changed": _metadata_bool(handoff.get("state_changed")),
        f"{prefix}_changed": list(handoff.get("changed") or []),
        f"{prefix}_content_in_handoff": _metadata_bool(handoff.get("content_in_handoff")),
        f"{prefix}_authorizes_execution": False,
        f"{prefix}_authorizes_completion_claim": False,
        f"{prefix}_approval_granted": False,
        f"{prefix}_next_safe_command": handoff["next_safe_command"],
        f"{prefix}_next_safe_commands": list(handoff["next_safe_commands"]),
        f"{prefix}_next_safe_command_count": handoff["next_safe_command_count"],
        handoff_key: handoff,
    }


def _reminder_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    due_raw = row.get("due", 0)
    try:
        due_iso = datetime.fromtimestamp(float(due_raw)).astimezone().isoformat()
        due_display = datetime.fromtimestamp(float(due_raw)).astimezone().strftime("%a %b %d %H:%M")
    except Exception:
        due_iso = ""
        due_display = "?"
    result = {
        "index": index,
        "due": due_iso,
        "due_display": due_display,
        "message": _rem._display_message(row.get("message", "")),
    }
    selector = str(row.get("selector") or "").strip()
    if selector:
        result["selector"] = selector
    return result


def _set_reminder_handoff(
    *,
    status: str,
    reason: str = "",
    due: datetime | None = None,
    message: str = "",
    owner_configured: bool = False,
    stored: bool = False,
    storage_error: bool = False,
) -> dict[str, Any]:
    changed = ["reminder_store_write"] if stored else []
    handoff = {
        "source": "set_reminder",
        "status": status,
        "reason": reason,
        "ready_for_operator": True,
        "state_changed": stored,
        "changed": changed,
        "due": due.isoformat() if due else "",
        "message": _rem._display_message(message) if message else "",
        "owner_configured": owner_configured,
        "stored": stored,
        "storage_error": storage_error,
        "content_in_metadata": bool(message),
        "content_in_handoff": bool(message),
        "content_is_preview_only": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "next_commands": [
            "list reminders" if stored else "set reminder <when> <message>",
            "cancel reminders",
        ],
        "boundaries": _reminder_boundaries(writes_files=stored),
    }
    return _reminder_handoff_metadata("set_reminder_handoff", handoff)


def _list_reminders_handoff(
    *,
    reminders: list[dict[str, Any]],
    status: str = "ok",
    reason: str = "",
    storage_error: bool = False,
    delivery_health: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = [_reminder_row(row, index) for index, row in enumerate(reminders, start=1)]
    content_present = bool(rows)
    source_available = status == "ok"
    health = dict(delivery_health or _rem.delivery_health_status())
    handoff = {
        "source": "list_reminders",
        "status": status,
        "reason": reason,
        "source_available": source_available,
        "result_complete": source_available,
        "count_known": source_available,
        "storage_error": storage_error,
        "delivery_health": health,
        "delivery_health_status": str(health.get("status") or "unknown"),
        "delivery_health_available": health.get("source_available") is True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "reminders": rows,
        "content_in_metadata": content_present,
        "content_in_handoff": content_present,
        "content_is_preview_only": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "next_commands": [
            "cancel reminders" if rows else "set reminder <when> <message>"
            if source_available
            else "setup check",
            "set reminder <when> <message>" if source_available else "list reminders",
        ],
        "boundaries": _reminder_boundaries(reads_personal_data=True),
    }
    if source_available:
        handoff["count"] = len(rows)
    return _reminder_handoff_metadata("list_reminders_handoff", handoff)


def _delivery_health_summary(health: dict[str, Any]) -> str:
    status = str(health.get("status") or "unknown")
    if health.get("source_available") is not True:
        reason = str(health.get("reason") or "unknown")
        return f"Reminder delivery health: unknown ({reason})."
    age_seconds = max(0, min(int(health.get("age_seconds") or 0), 86_400))
    label = status.replace("_", " ")
    return f"Reminder delivery health: {label} ({age_seconds}s old)."


def _cancel_reminders_handoff(
    *,
    status: str,
    removed: list[dict[str, Any]],
    reason: str = "",
    storage_error: bool = False,
) -> dict[str, Any]:
    rows = [_reminder_row(row, index) for index, row in enumerate(removed, start=1)]
    writes_files = status in {"cancelled", "published_uncertain"}
    changed = ["reminder_store_write"] if writes_files else []
    content_present = bool(rows)
    handoff = {
        "source": "cancel_reminders",
        "status": status,
        "reason": reason,
        "ready_for_operator": True,
        "state_changed": writes_files,
        "changed": changed,
        "count": len(rows),
        "cancelled_reminders": rows,
        "storage_error": storage_error,
        "content_in_metadata": content_present,
        "content_in_handoff": content_present,
        "content_is_preview_only": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "next_commands": [
            "list reminders",
            "set reminder <when> <message>",
        ],
        "boundaries": _reminder_boundaries(reads_personal_data=True, writes_files=writes_files),
    }
    return _reminder_handoff_metadata("cancel_reminders_handoff", handoff)


LOCATION_REMINDER_TRIGGER_PATTERNS = (
    (
        "arrive",
        re.compile(
            r"\b(?:when|once)\s+i\s+(?:get\s+home|arrive\s+home|come\s+home)\b",
            re.IGNORECASE,
        ),
        "home",
    ),
    (
        "arrive",
        re.compile(
            r"\b(?:when|once)\s+i\s+(?:get\s+to|arrive\s+at|arrive\s+to|reach)\s+(?P<location>[a-z][a-z0-9\s'\-]{1,50}?)(?=\s+(?:to|that|about|for)\b|$)",
            re.IGNORECASE,
        ),
        "",
    ),
    (
        "leave",
        re.compile(
            r"\b(?:when|once)\s+i\s+(?:leave|am\s+leaving)\s+(?P<location>[a-z][a-z0-9\s'\-]{1,50}?)(?=\s+(?:to|that|about|for)\b|$)",
            re.IGNORECASE,
        ),
        "",
    ),
)


def _parse_location_reminder_draft(text: str) -> dict[str, str]:
    raw = re.sub(r"\s+", " ", str(text or "").strip())
    if not raw:
        return {"status": "refused", "reason": "missing_text"}
    if has_local_path(raw):
        return {"status": "refused", "reason": "local_path_input", "message": "<local-path>", "location_label": "<local-path>"}
    cleaned = re.sub(r"[\?\.!]+$", "", raw).strip()
    for event, pattern, default_location in LOCATION_REMINDER_TRIGGER_PATTERNS:
        match = pattern.search(cleaned)
        if not match:
            continue
        location = (default_location or match.groupdict().get("location") or "").strip()
        location = re.sub(r"\s+(?:to|and|then)$", "", location, flags=re.IGNORECASE).strip()
        before = cleaned[: match.start()].strip()
        after = cleaned[match.end() :].strip()
        before = re.sub(
            r"^(?:please\s+)?(?:remind me(?:\s+to)?|set\s+(?:a\s+)?reminder(?:\s+to)?|create\s+(?:a\s+)?reminder(?:\s+to)?|add\s+(?:a\s+)?reminder(?:\s+to)?)\s*",
            "",
            before,
            flags=re.IGNORECASE,
        ).strip()
        after = re.sub(r"^(?:to|that|about|for)\s+", "", after, flags=re.IGNORECASE).strip()
        message = before or after
        message = re.sub(r"\s+", " ", message).strip(" :,-")
        location = re.sub(r"\s+", " ", location).strip(" :,-")
        if not location:
            return {"status": "refused", "reason": "missing_location"}
        if not message:
            return {"status": "refused", "reason": "missing_message", "trigger_event": event, "location_label": location}
        if has_local_path(message) or has_local_path(location):
            return {"status": "refused", "reason": "local_path_input", "message": "<local-path>", "location_label": "<local-path>"}
        return {
            "status": "drafted",
            "trigger_event": event,
            "location_label": location,
            "message": _rem._display_message(message),
        }
    return {"status": "refused", "reason": "unsupported_location_trigger"}


def _location_reminder_draft_handoff(*, parsed: dict[str, str]) -> dict[str, Any]:
    status = parsed.get("status") or "refused"
    message = parsed.get("message") or ""
    location_label = parsed.get("location_label") or ""
    trigger_event = parsed.get("trigger_event") or ""
    handoff = {
        "source": "location_reminder_draft",
        "status": status,
        "reason": parsed.get("reason") or "",
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "trigger_event": trigger_event,
        "location_label": location_label,
        "message": message,
        "requires_location_permission": status == "drafted",
        "requires_review_before_creation": True,
        "supported_execution_now": False,
        "content_in_metadata": bool(message or location_label),
        "content_in_handoff": bool(message or location_label),
        "content_is_preview_only": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "next_commands": [
            "set a time-based reminder instead",
            "review location-reminder permissions before enabling geofencing",
        ],
        "boundaries": _reminder_boundaries(),
    }
    return _reminder_handoff_metadata("location_reminder_draft_handoff", handoff)


def _reminder_time_parse_guidance() -> str:
    return "I couldn't work out the time. Try 'in 20 minutes', 'in 2 hours', or 'at 3pm'."


def _reminder_missing_owner_guidance() -> str:
    return "Telegram reminders need JARVIS_OWNER_TELEGRAM set (so I can message you when it's due)."


def _reminder_storage_recovery_guidance() -> str:
    return "I couldn't save that reminder locally. Check that the Jarvis reminders file path is writable."


REMINDER_INPUT_RECOVERY_ACTION = (
    "Correct the reminder details shown above, then submit a new request through the normal approval flow."
)
REMINDER_SETUP_RECOVERY_ACTION = (
    "Run `setup check`, correct the reminder configuration, then submit a new request through the normal approval flow."
)
REMINDER_STORE_RECOVERY_ACTION = (
    "Run `setup check`, inspect or restore the reminder store, then retry through the normal policy."
)
REMINDER_CANCEL_RECOVERY_ACTION = (
    "Run `list reminders`, choose the current reminder id, then submit a new cancellation through the normal approval flow."
)
LOCATION_REMINDER_RECOVERY_ACTION = (
    "Correct the location-reminder draft details shown above, then retry the draft."
)


def _known_no_reminder_change_failure(
    tool_name: str,
    output: str,
    metadata: dict[str, Any],
    *,
    action: str,
    commands: tuple[str, ...] = (),
) -> ToolResult:
    """Declare a rejected reminder operation before any target-state change."""

    public_output = str(output).strip()
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


def make_reminder_tools(config: JarvisConfig):
    def _prepare_reminder_request(text: str) -> tuple[datetime, str, str] | ToolResult:
        if not text:
            return _known_no_reminder_change_failure(
                "set_reminder",
                "Tell me what and when, e.g. 'remind me in 20 minutes to call mom'.",
                _safe_metadata(**_set_reminder_handoff(status="refused", reason="missing_text")),
                action=REMINDER_INPUT_RECOVERY_ACTION,
            )
        if is_recurring_request(text):
            return _known_no_reminder_change_failure(
                "set_reminder",
                "Recurring reminders are not supported yet. I can set a one-time reminder like 'tomorrow at 9am' or 'in 20 minutes'.",
                _safe_metadata(
                    reason="recurring_reminder_unsupported",
                    **_set_reminder_handoff(status="refused", reason="recurring_reminder_unsupported"),
                ),
                action=REMINDER_INPUT_RECOVERY_ACTION,
            )
        parsed = parse_when(text)
        if parsed is None:
            return _known_no_reminder_change_failure(
                "set_reminder",
                _reminder_time_parse_guidance(),
                _safe_metadata(**_set_reminder_handoff(status="refused", reason="unparsed_time")),
                action=REMINDER_INPUT_RECOVERY_ACTION,
            )
        due, message = parsed
        if has_local_path(message):
            return _known_no_reminder_change_failure(
                "set_reminder",
                "Reminder text cannot be a local file path.",
                _safe_metadata(
                    reason="invalid_reminder_text",
                    message="<local-path>",
                    **_set_reminder_handoff(
                        status="refused",
                        reason="invalid_reminder_text",
                        due=due,
                        message="<local-path>",
                    ),
                ),
                action=REMINDER_INPUT_RECOVERY_ACTION,
            )
        chat_id = os.getenv("JARVIS_OWNER_TELEGRAM", "").strip()
        if not chat_id:
            return _known_no_reminder_change_failure(
                "set_reminder",
                _reminder_missing_owner_guidance(),
                _safe_metadata(
                    **_set_reminder_handoff(
                        status="refused",
                        reason="missing_owner_telegram",
                        due=due,
                        message=message,
                    )
                ),
                action=REMINDER_SETUP_RECOVERY_ACTION,
                commands=("setup check",),
            )
        return due, message, chat_id

    def _owner_fingerprint(chat_id: str, bot_token: str) -> str:
        return hmac.new(
            bot_token.encode("utf-8"),
            chat_id.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def resolve_set_reminder_approval(args: dict[str, Any]) -> ApprovalArgumentResolution | ToolResult:
        text = str(args.get("text") or "").strip()
        prepared = _prepare_reminder_request(text)
        if isinstance(prepared, ToolResult):
            prepared.metadata.update(
                {
                    "executed_handler": False,
                    "handler_invoked": False,
                }
            )
            return prepared
        due, message, chat_id = prepared
        low = " ".join(text.casefold().split())
        if re.search(r"\b(?:\d+|an?|half(?:\s+an?)?)\s*(?:seconds?|secs?|s)\b", low):
            if due.microsecond:
                due += timedelta(seconds=1)
            due = due.replace(microsecond=0)
        else:
            if due.second or due.microsecond:
                due += timedelta(minutes=1)
            due = due.replace(second=0, microsecond=0)
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not bot_token:
            refused = _known_no_reminder_change_failure(
                "set_reminder",
                "Telegram reminders need TELEGRAM_BOT_TOKEN configured before I can queue one for approval.",
                _safe_metadata(
                    reason="missing_telegram_bot_token",
                    executed_handler=False,
                    handler_invoked=False,
                    **_set_reminder_handoff(
                        status="refused",
                        reason="missing_telegram_bot_token",
                    ),
                ),
                action=REMINDER_SETUP_RECOVERY_ACTION,
                commands=("setup check",),
            )
            return refused
        return ApprovalArgumentResolution(
            {
                "due_epoch": due.timestamp(),
                "message": message,
                "owner_fingerprint": _owner_fingerprint(chat_id, bot_token),
            },
            {
                "reminder_approval_binding": "absolute_due_owner_fingerprint",
                "reminder_due": due.isoformat(),
                "reminder_owner_bound": True,
            },
        )

    def set_reminder(args: dict[str, Any]) -> ToolResult:
        if {"due_epoch", "message", "owner_fingerprint"}.issubset(args):
            chat_id = os.getenv("JARVIS_OWNER_TELEGRAM", "").strip()
            bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
            message = str(args.get("message") or "").strip()
            owner_fingerprint = str(args.get("owner_fingerprint") or "").strip()
            try:
                due = datetime.fromtimestamp(float(args.get("due_epoch"))).astimezone()
            except (TypeError, ValueError, OverflowError, OSError):
                due = datetime.now().astimezone()
                owner_fingerprint = ""
            if (
                not chat_id
                or not bot_token
                or not owner_fingerprint
                or not hmac.compare_digest(
                    _owner_fingerprint(chat_id, bot_token),
                    owner_fingerprint,
                )
                or has_local_path(message)
            ):
                return _known_no_reminder_change_failure(
                    "set_reminder",
                    "The approved reminder binding is no longer valid, so nothing was scheduled. Submit the reminder again for a fresh approval.",
                    _safe_metadata(
                        reason="stale_reminder_binding",
                        **_set_reminder_handoff(status="refused", reason="stale_reminder_binding"),
                    ),
                    action=REMINDER_INPUT_RECOVERY_ACTION,
                )
            if due <= datetime.now().astimezone():
                return _known_no_reminder_change_failure(
                    "set_reminder",
                    "That approved reminder time has already passed, so nothing was scheduled. Submit it again for a fresh approval.",
                    _safe_metadata(
                        reason="stale_reminder_due",
                        **_set_reminder_handoff(status="refused", reason="stale_reminder_due"),
                    ),
                    action=REMINDER_INPUT_RECOVERY_ACTION,
                )
        else:
            text = str(args.get("text") or args.get("request") or "").strip()
            prepared = _prepare_reminder_request(text)
            if isinstance(prepared, ToolResult):
                return prepared
            due, message, chat_id = prepared
        write_result = add_reminder_result(due.timestamp(), message, chat_id)
        if write_result.status == "failed":
            return _known_no_reminder_change_failure(
                "set_reminder",
                _reminder_storage_recovery_guidance(),
                _safe_metadata(
                    due=due.isoformat(),
                    message=message,
                    storage_error=True,
                    **_set_reminder_handoff(
                        status="failed",
                        reason="storage_error",
                        due=due,
                        message=message,
                        owner_configured=True,
                        storage_error=True,
                    ),
                ),
                action=REMINDER_STORE_RECOVERY_ACTION,
                commands=("setup check",),
            )
        if write_result.status == "published_uncertain":
            return ToolResult(
                "set_reminder",
                False,
                "The reminder appears in Jarvis's local store, but disk durability could not be confirmed. "
                "Do not retry automatically; run `list reminders` to verify the published reminder.",
                _safe_metadata(
                    due=due.isoformat(),
                    message=message,
                    writes_files=True,
                    executes_side_effect=True,
                    storage_error=True,
                    durability_uncertain=True,
                    outcome_known=False,
                    retry_safe=False,
                    authorizes_retry=False,
                    **_set_reminder_handoff(
                        status="published_uncertain",
                        reason="directory_sync_failed",
                        due=due,
                        message=message,
                        owner_configured=True,
                        stored=True,
                        storage_error=True,
                    ),
                ),
            )
        when = due.strftime("%a %b %d %H:%M") if (due.date() != due.today().date()) else due.strftime("%H:%M")
        return ToolResult(
            "set_reminder", True,
            f"Got it — I'll message you at {when}: {message}",
            _safe_metadata(
                due=due.isoformat(),
                message=message,
                writes_files=True,
                executes_side_effect=True,
                **_set_reminder_handoff(
                    status="scheduled",
                    due=due,
                    message=message,
                    owner_configured=True,
                    stored=True,
                ),
            ),
        )

    def location_reminder_draft(args: dict[str, Any]) -> ToolResult:
        text = str(args.get("text") or args.get("request") or "").strip()
        parsed = _parse_location_reminder_draft(text)
        metadata = _safe_metadata(
            location_reminder_status=parsed.get("status") or "refused",
            reason=parsed.get("reason") or "",
            trigger_event=parsed.get("trigger_event") or "",
            location_label=parsed.get("location_label") or "",
            message=parsed.get("message") or "",
            requires_location_permission=parsed.get("status") == "drafted",
            requires_review_before_creation=True,
            supported_execution_now=False,
            **_location_reminder_draft_handoff(parsed=parsed),
        )
        if parsed.get("status") != "drafted":
            reason = parsed.get("reason") or "unsupported_location_trigger"
            if reason == "missing_message":
                output = "I can draft a location reminder, but I need what to remind you about."
            elif reason == "local_path_input":
                output = "Location reminder drafts cannot include local file paths."
            else:
                output = "I can draft location reminders for arrivals like 'when I get home' or departures like 'when I leave work'."
            return _known_no_reminder_change_failure(
                "location_reminder_draft",
                output,
                metadata,
                action=LOCATION_REMINDER_RECOVERY_ACTION,
            )

        event_label = "arriving at" if parsed.get("trigger_event") == "arrive" else "leaving"
        output = (
            f"Location reminder draft: when {event_label} {parsed.get('location_label')}, remind you to {parsed.get('message')}.\n"
            "This is only a draft. Jarvis is not monitoring location or creating a geofence reminder yet."
        )
        return ToolResult("location_reminder_draft", True, output, metadata)

    def list_reminders(args: dict[str, Any]) -> ToolResult:
        chat_id = os.getenv("JARVIS_OWNER_TELEGRAM", "").strip()
        delivery_health = _rem.delivery_health_status()
        health_summary = _delivery_health_summary(delivery_health)
        if not chat_id:
            failure_output = (
                "Telegram reminder access needs JARVIS_OWNER_TELEGRAM configured. "
                "I did not read the reminder store. "
                f"{LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION}\n{health_summary}"
            )
            return ToolResult(
                "list_reminders",
                False,
                failure_output,
                declare_retryable_personal_read_failure(
                    _safe_metadata(
                        reason="missing_owner_configuration",
                        delivery_health=delivery_health,
                        delivery_health_status=delivery_health.get("status"),
                        source_available=False,
                        result_complete=False,
                        count_known=False,
                        write_suppressed=True,
                        **_list_reminders_handoff(
                            reminders=[],
                            status="unavailable",
                            reason="missing_owner_configuration",
                            delivery_health=delivery_health,
                        ),
                    ),
                    output=failure_output,
                    action=LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
                    commands=("setup check",),
                ),
            )
        items, source_available, reason = _rem.pending_reminders_status(chat_id)
        if not source_available:
            failure_output = (
                "The local reminder store is unavailable, so I can't verify whether reminders are pending. "
                "I did not change it. Run `setup check`, inspect or restore the reminders file, then retry "
                f"`list reminders`. {LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION}\n{health_summary}"
            )
            return ToolResult(
                "list_reminders",
                False,
                failure_output,
                declare_retryable_personal_read_failure(
                    _safe_metadata(
                        reads_personal_data=True,
                        storage_error=True,
                        delivery_health=delivery_health,
                        delivery_health_status=delivery_health.get("status"),
                        failure_kind="reminder_store_unavailable",
                        source_available=False,
                        result_complete=False,
                        count_known=False,
                        write_suppressed=True,
                        **_list_reminders_handoff(
                            reminders=[],
                            status="unavailable",
                            reason=reason,
                            storage_error=True,
                            delivery_health=delivery_health,
                        ),
                    ),
                    output=failure_output,
                    action=LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
                    commands=("setup check",),
                ),
            )
        if not items:
            return ToolResult(
                "list_reminders",
                True,
                "You have no pending reminders.\n" + health_summary,
                _safe_metadata(
                    reads_personal_data=True,
                    count=0,
                    delivery_health=delivery_health,
                    delivery_health_status=delivery_health.get("status"),
                    **_list_reminders_handoff(
                        reminders=[],
                        delivery_health=delivery_health,
                    ),
                ),
            )
        items = _with_reminder_selectors(items)
        lines = [f"Pending reminders ({len(items)}):"]
        for r in items:
            try:
                when = datetime.fromtimestamp(float(r.get("due", 0))).astimezone().strftime("%a %b %d %H:%M")
            except Exception:
                when = "?"
            selector = str(r.get("selector") or "")
            lines.append(f"  • [{selector}] {when}: {_rem._display_message(r.get('message', ''))}")
        lines.extend(["", health_summary])
        return ToolResult(
            "list_reminders",
            True,
            "\n".join(lines),
            _safe_metadata(
                reads_personal_data=True,
                count=len(items),
                delivery_health=delivery_health,
                delivery_health_status=delivery_health.get("status"),
                **_list_reminders_handoff(
                    reminders=items,
                    delivery_health=delivery_health,
                ),
            ),
        )

    def cancel_reminders(args: dict[str, Any]) -> ToolResult:
        chat_id = os.getenv("JARVIS_OWNER_TELEGRAM", "").strip()
        if not chat_id:
            return _known_no_reminder_change_failure(
                "cancel_reminders",
                "Telegram reminder cancellation needs JARVIS_OWNER_TELEGRAM configured. I did not read or change the reminder store.",
                _safe_metadata(
                    count=0,
                    reason="missing_owner_configuration",
                    write_suppressed=True,
                    **_cancel_reminders_handoff(
                        status="refused",
                        reason="missing_owner_configuration",
                        removed=[],
                    ),
                ),
                action=REMINDER_SETUP_RECOVERY_ACTION,
                commands=("setup check",),
            )
        reminder_id = str(args.get("reminder_id") or "").strip()
        selective = "reminder_id" in args
        if selective:
            status, removed_items, saved = _rem.cancel_pending_item(chat_id, reminder_id)
            in_flight_count = 0
        else:
            removed_items, saved, in_flight_count, status = _rem.cancel_pending_items_status(chat_id)
        removed = len(removed_items)
        if status == "published_uncertain":
            if selective and removed_items:
                selector = reminder_id.lower().lstrip("#").replace("-", "")
                removed_items = [dict(removed_items[0], selector=selector)]
            else:
                removed_items = _with_reminder_selectors(removed_items)
            uncertainty_output = (
                "The cancellation appears in the local reminder store, but disk durability could not be confirmed. "
                "Do not retry automatically; run `list reminders` to verify the current state."
            )
            if in_flight_count:
                uncertainty_output += (
                    f" {in_flight_count} reminder(s) were already being sent and were not cancelled."
                )
            return ToolResult(
                "cancel_reminders",
                False,
                uncertainty_output,
                _safe_metadata(
                    reads_personal_data=True,
                    count=removed,
                    in_flight_count=in_flight_count,
                    writes_files=True,
                    executes_side_effect=True,
                    storage_error=True,
                    durability_uncertain=True,
                    outcome_known=False,
                    retry_safe=False,
                    authorizes_retry=False,
                    partial=bool(in_flight_count),
                    **_cancel_reminders_handoff(
                        status="published_uncertain",
                        reason="directory_sync_failed",
                        removed=removed_items,
                        storage_error=True,
                    ),
                ),
            )
        if not saved:
            return _known_no_reminder_change_failure(
                "cancel_reminders",
                "I couldn't safely confirm the reminder update. Run `list reminders` before retrying so I don't cancel the wrong item.",
                _safe_metadata(
                    reads_personal_data=True,
                    count=0,
                    storage_error=True,
                    outcome_known=False,
                    retry_safe=False,
                    write_suppressed=True,
                    **_cancel_reminders_handoff(
                        status="failed",
                        reason="reminder_store_unavailable",
                        removed=[],
                        storage_error=True,
                    ),
                ),
                action=REMINDER_STORE_RECOVERY_ACTION,
                commands=("setup check",),
            )
        if selective and status != "cancelled":
            messages = {
                "invalid_selector": "I need the reminder id shown by `list reminders`. Say `cancel reminder <id>`.",
                "not_found": "I couldn't find that pending reminder. Run `list reminders` and retry `cancel reminder <id>`.",
                "ambiguous": "That reminder id prefix matches more than one reminder. Use the longer id shown by `list reminders`.",
                "in_flight": "That reminder is already being sent, so I did not cancel it. Run `list reminders` to check its state.",
            }
            return _known_no_reminder_change_failure(
                "cancel_reminders",
                messages.get(status, messages["not_found"]),
                _safe_metadata(
                    reads_personal_data=True,
                    count=0,
                    reason=status,
                    write_suppressed=True,
                    **_cancel_reminders_handoff(status=status, reason=status, removed=[]),
                ),
                action=REMINDER_CANCEL_RECOVERY_ACTION,
                commands=("list reminders",),
            )
        if not selective and status == "in_flight":
            return _known_no_reminder_change_failure(
                "cancel_reminders",
                f"{in_flight_count} reminder(s) are already being sent, so I did not cancel them. "
                "Run `list reminders` to check their state.",
                _safe_metadata(
                    reads_personal_data=True,
                    count=0,
                    in_flight_count=in_flight_count,
                    reason="in_flight",
                    write_suppressed=True,
                    **_cancel_reminders_handoff(status="in_flight", reason="in_flight", removed=[]),
                ),
                action=REMINDER_CANCEL_RECOVERY_ACTION,
                commands=("list reminders",),
            )
        if status == "empty":
            return ToolResult(
                "cancel_reminders",
                True,
                "You have no pending reminders to cancel.",
                _safe_metadata(reads_personal_data=True, count=0, **_cancel_reminders_handoff(status="empty", removed=[])),
            )
        if selective:
            selector = reminder_id.lower().lstrip("#").replace("-", "")
            removed_items = [dict(removed_items[0], selector=selector)]
            output = f"Cancelled 1 reminder [{selector}]."
        else:
            removed_items = _with_reminder_selectors(removed_items)
            output = f"Cancelled {removed} reminder(s)."
            if in_flight_count:
                output += f" {in_flight_count} reminder(s) were already being sent and were not cancelled."
        return ToolResult(
            "cancel_reminders",
            True,
            output,
            _safe_metadata(
                reads_personal_data=True,
                count=removed,
                in_flight_count=in_flight_count,
                partial=bool(in_flight_count),
                writes_files=True,
                executes_side_effect=True,
                **_cancel_reminders_handoff(status="cancelled", removed=removed_items),
            ),
        )

    from jarvis_v2.tools.registry import (
        TOOL_ARGUMENT_CONTRACT_VERSION,
        Tool,
        ToolArgumentContract,
        ToolArgumentSpec,
        ToolArgumentType,
    )
    return [
        Tool(
            "set_reminder",
            "Set a reminder or timer stored locally and later sent through the owner Telegram channel. Args: text (e.g. 'remind me in 20 minutes to stretch', 'set a timer for 10 minutes').",
            RiskLevel.EXTERNAL_SIDE_EFFECT,
            set_reminder,
            "personal",
            argument_contract=ToolArgumentContract(
                version=TOOL_ARGUMENT_CONTRACT_VERSION,
                fields=(
                    ToolArgumentSpec(
                        "text",
                        frozenset({ToolArgumentType.STRING}),
                        True,
                    ),
                ),
                allow_unknown=False,
            ),
            approval_argument_resolver=resolve_set_reminder_approval,
            approval_argument_contract=ToolArgumentContract(
                version=TOOL_ARGUMENT_CONTRACT_VERSION,
                fields=(
                    ToolArgumentSpec("due_epoch", frozenset({ToolArgumentType.NUMBER}), True),
                    ToolArgumentSpec("message", frozenset({ToolArgumentType.STRING}), True),
                    ToolArgumentSpec("owner_fingerprint", frozenset({ToolArgumentType.STRING}), True),
                ),
                allow_unknown=False,
            ),
        ),
        Tool(
            "list_reminders",
            "List your pending reminders/timers. No args.",
            RiskLevel.READ_ONLY,
            list_reminders,
            "personal",
        ),
        Tool(
            "location_reminder_draft",
            "Draft a location-triggered reminder packet without monitoring location or creating a reminder. Args: text.",
            RiskLevel.READ_ONLY,
            location_reminder_draft,
            "personal",
        ),
        Tool(
            "cancel_reminders",
            "Cancel all pending reminders, or exactly one by the stable id shown by list_reminders. Optional arg: reminder_id (string).",
            RiskLevel.LOCAL_SAFE,
            cancel_reminders,
            "personal",
            argument_contract=ToolArgumentContract(
                version=TOOL_ARGUMENT_CONTRACT_VERSION,
                fields=(
                    ToolArgumentSpec(
                        "reminder_id",
                        frozenset({ToolArgumentType.STRING}),
                        False,
                    ),
                ),
                allow_unknown=False,
            ),
        ),
    ]
