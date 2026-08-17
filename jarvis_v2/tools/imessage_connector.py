"""iMessage send connector for Jarvis V2 (macOS only, uses osascript)."""

from __future__ import annotations

import re
import subprocess
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    KNOWN_NOT_SENT_SEND_RECOVERY_ACTION,
    PERSONAL_READ_RECOVERY_ACTION,
    declare_known_not_sent_failure,
    declare_outcome_unknown_failure,
    declare_retryable_personal_read_failure,
)
from jarvis_v2.agent.types import RiskLevel, ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.tools import contacts_connector
from jarvis_v2.tools import contacts_fuzzy


MAX_RECIPIENT_CHARS = 160
MAX_MESSAGE_CHARS = 2000
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)


def _post_attempt_send_recovery_guidance() -> str:
    return (
        "The iMessage delivery outcome is unknown. Check the exact conversation; do not resend "
        "automatically. If the message is absent, run `setup check`, repair Messages setup and "
        "macOS Automation permission, then submit a new approved send."
    )


def _send_outcome_is_known_not_sent(stage: str) -> bool:
    return stage in {
        "macos_automation_permission",
        "messages_service_unavailable",
        "imessage_recipient_unavailable",
        "messages_unavailable",
    }


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
    if value is True:
        return True
    if value is False:
        return False
    return default


def _short(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text[:limit] if len(text) > limit else text


def _short_raw(value: Any, limit: int = 80) -> str:
    text = "" if value is None else str(value).strip()
    text = " ".join(text.split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _message_preview(value: Any, limit: int = 120) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.strip().split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _imessage_send_boundaries(*, send_attempted: bool, contact_lookup_attempted: bool = False) -> dict[str, bool]:
    return {
        "calls_model": False,
        "calls_external_service": False,
        "executes_tools": False,
        "reads_personal_data": contact_lookup_attempted,
        "reads_private_data": False,
        "executes_side_effect": send_attempted,
        "external_side_effect": send_attempted,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": send_attempted,
        "controls_computer": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _imessage_send_handoff(
    *,
    recipient: str,
    message_chars: int,
    status: str,
    reason: str = "",
    send_attempted: bool = False,
    original_to: str = "",
    original_recipient: str = "",
    contact_lookup_attempted: bool = False,
    contact_resolution_status: str = "skipped",
    contact_match_count: int = 0,
    contact_candidates: list[str] | None = None,
) -> dict[str, Any]:
    changed = ["imessage_send_attempt"] if send_attempted else []
    boundaries = _imessage_send_boundaries(
        send_attempted=send_attempted,
        contact_lookup_attempted=contact_lookup_attempted,
    )
    if status == "outcome_unknown":
        next_safe_commands = ["recent tool runs", "execution recovery"]
    elif status == "failed":
        next_safe_commands = ["setup check"]
    elif status == "sender_confirmed":
        next_safe_commands = ["recent tool runs"]
    else:
        next_safe_commands = ["safe next actions"]
    next_safe_command = next_safe_commands[0]
    handoff = {
        "source": "send_imessage",
        "status": status,
        "reason": reason,
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": send_attempted,
        "changed": changed,
        "content_in_handoff": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "to": _message_preview(recipient, limit=MAX_RECIPIENT_CHARS),
        "original_to": _message_preview(original_to or original_recipient or recipient, limit=MAX_RECIPIENT_CHARS),
        "message_chars": message_chars,
        "contact_lookup_attempted": contact_lookup_attempted,
        "contact_resolution_status": contact_resolution_status,
        "contact_match_count": contact_match_count,
        "contact_candidates": [
            _message_preview(candidate, limit=MAX_RECIPIENT_CHARS) for candidate in (contact_candidates or [])
        ],
        "content_in_metadata": False,
        "send_attempted": send_attempted,
        "sender_side_send_confirmed": status == "sender_confirmed",
        "recipient_delivery_confirmed": False,
        "delivery_confirmed": False,
        "delivery_evidence": "sender_side_only" if status == "sender_confirmed" else "none",
        "recipient_confirmation_required": status == "sender_confirmed",
        "approval_required_before_execution": True,
        "manual_review_required": True,
        "next_safe_command": next_safe_command,
        "next_safe_commands": next_safe_commands,
        "next_safe_command_count": len(next_safe_commands),
        "boundaries": boundaries,
    }
    return {
        "imessage_send_handoff_ready": True,
        "imessage_send_ready_for_operator": handoff["ready_for_operator"],
        "imessage_send_state_changed": handoff["state_changed"],
        "imessage_send_changed": handoff["changed"],
        "imessage_send_content_in_handoff": handoff["content_in_handoff"],
        "imessage_send_content_in_metadata": handoff["content_in_metadata"],
        "imessage_send_next_safe_command": handoff["next_safe_command"],
        "imessage_send_next_safe_commands": handoff["next_safe_commands"],
        "imessage_send_next_safe_command_count": handoff["next_safe_command_count"],
        "imessage_send_authorizes_execution": handoff["authorizes_execution"],
        "imessage_send_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "imessage_send_approval_granted": handoff["approval_granted"],
        "imessage_send_boundaries": boundaries,
        "ready_for_operator": handoff["ready_for_operator"],
        "state_changed": handoff["state_changed"],
        "changed": handoff["changed"],
        "content_in_handoff": handoff["content_in_handoff"],
        "authorizes_execution": handoff["authorizes_execution"],
        "authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "approval_granted": handoff["approval_granted"],
        "imessage_send_handoff": handoff,
    }


def _resolve_send_recipient(raw_recipient: str) -> tuple[str | None, dict[str, Any], str]:
    if contacts_connector.looks_like_handle(raw_recipient):
        return raw_recipient, {
            "original_to": _message_preview(raw_recipient, limit=MAX_RECIPIENT_CHARS),
            "contact_lookup_attempted": False,
            "contact_resolution_status": "skipped",
            "contact_match_count": 0,
            "contact_candidates": [],
        }, ""
    try:
        matches = contacts_connector.resolve_contact(raw_recipient)
    except Exception:
        return None, {
            "original_to": _message_preview(raw_recipient, limit=MAX_RECIPIENT_CHARS),
            "contact_lookup_attempted": True,
            "contact_resolution_status": "unavailable",
            "contact_match_count": 0,
            "contact_candidates": [],
        }, "I couldn't access Contacts to resolve that iMessage recipient. Check Contacts access and retry."
    candidate_names = [match.name for match in matches]
    base = {
        "original_to": _message_preview(raw_recipient, limit=MAX_RECIPIENT_CHARS),
        "contact_lookup_attempted": True,
        "contact_match_count": len(matches),
        "contact_candidates": [_message_preview(name, limit=MAX_RECIPIENT_CHARS) for name in candidate_names],
    }
    if len(matches) == 1 and matches[0].handle:
        return matches[0].handle, {**base, "contact_resolution_status": "resolved"}, ""
    if len(matches) > 1:
        names = " or ".join(_message_preview(name, limit=MAX_RECIPIENT_CHARS) for name in candidate_names)
        return None, {**base, "contact_resolution_status": "ambiguous"}, (
            f"I found {len(matches)} people named \"{_message_preview(raw_recipient, limit=MAX_RECIPIENT_CHARS)}\" — {names}?"
        )
    return None, {**base, "contact_resolution_status": "not_found"}, (
        f"I couldn't find a contact matching \"{_message_preview(raw_recipient, limit=MAX_RECIPIENT_CHARS)}\"."
        + contacts_fuzzy.suggestion_clause(raw_recipient)
    )


def _imessage_read_boundaries() -> dict[str, bool]:
    return {
        "calls_model": False,
        "calls_external_service": False,
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
        "controls_computer": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _imessage_recent_row(row: tuple[Any, Any, Any, Any], index: int) -> dict[str, Any]:
    text, is_from_me, sender, date = row
    return {
        "index": index,
        "direction": "sent" if is_from_me else "received",
        "sender": "Me" if is_from_me else _message_preview(sender or "Unknown", limit=MAX_RECIPIENT_CHARS),
        "preview": _message_preview(text),
        "date": _short_raw(date, limit=80),
    }


def _imessage_recent_handoff(
    *,
    limit: int,
    rows: list[dict[str, Any]] | None,
    status: str,
    reason: str = "",
) -> dict[str, Any]:
    rows = rows or []
    boundaries = _imessage_read_boundaries()
    next_safe_command = f"read recent imessages limit {limit}"
    handoff = {
        "source": "read_recent_imessages",
        "status": status,
        "reason": reason,
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "limit": limit,
        "message_count": len(rows),
        "messages": rows,
        "content_in_metadata": True,
        "content_is_preview_only": True,
        "next_safe_command": next_safe_command,
        "next_safe_commands": [next_safe_command],
        "next_safe_command_count": 1,
        "boundaries": boundaries,
    }
    return {
        "imessage_recent_handoff_ready": True,
        "imessage_recent_ready_for_operator": handoff["ready_for_operator"],
        "imessage_recent_state_changed": handoff["state_changed"],
        "imessage_recent_changed": handoff["changed"],
        "imessage_recent_content_in_handoff": handoff["content_in_handoff"],
        "imessage_recent_content_in_metadata": handoff["content_in_metadata"],
        "imessage_recent_next_safe_command": handoff["next_safe_command"],
        "imessage_recent_next_safe_commands": handoff["next_safe_commands"],
        "imessage_recent_next_safe_command_count": handoff["next_safe_command_count"],
        "imessage_recent_authorizes_execution": handoff["authorizes_execution"],
        "imessage_recent_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "imessage_recent_approval_granted": handoff["approval_granted"],
        "imessage_recent_boundaries": boundaries,
        "ready_for_operator": handoff["ready_for_operator"],
        "state_changed": handoff["state_changed"],
        "changed": handoff["changed"],
        "content_in_handoff": handoff["content_in_handoff"],
        "authorizes_execution": handoff["authorizes_execution"],
        "authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "approval_granted": handoff["approval_granted"],
        "imessage_recent_handoff": handoff,
    }


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


def _applescript_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _imessage_error(operation: str) -> str:
    if operation == "send":
        return "iMessage could not send that message right now. Check Messages setup and try again."
    return "iMessage could not read recent messages right now. Check Full Disk Access for the Jarvis Python process."


class IMessageSendError(RuntimeError):
    def __init__(self, stage: str, detail: str = "") -> None:
        self.stage = _short_raw(stage, limit=80)
        self.detail = _short_raw(detail, limit=220)
        suffix = f": {self.detail}" if self.detail else ""
        super().__init__(f"iMessage automation failed at {self.stage}{suffix}")


def _imessage_error_stage(stderr: str) -> tuple[str, str]:
    detail = _short_raw(stderr, limit=220)
    low = stderr.lower()
    if (
        "not authorized to send apple events" in low
        or "not permitted to send apple events" in low
        or "not allowed to send apple events" in low
        or "(-1743)" in low
    ):
        return (
            "macos_automation_permission",
            "In System Settings > Privacy & Security > Automation, allow the Jarvis Python process "
            "to control Messages, then retry.",
        )
    if "no service" in low or "can't get service" in low or "cannot get service" in low:
        return "messages_service_unavailable", "Open Messages and confirm that an iMessage account is signed in, then retry."
    if "can't get buddy" in low or "cannot get buddy" in low or "recipient is not registered" in low:
        return (
            "imessage_recipient_unavailable",
            "Confirm the resolved phone number or email is registered for iMessage, then retry.",
        )
    if "application isn't running" in low or "application is not running" in low:
        return "messages_unavailable", "Open Messages, confirm it is ready, then retry."
    return "osascript_failed", detail


def _imessage_send_failure(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, IMessageSendError):
        if exc.stage in {
            "macos_automation_permission",
            "messages_service_unavailable",
            "imessage_recipient_unavailable",
            "messages_unavailable",
        }:
            return exc.stage, f"iMessage send stopped at {exc.stage}: {exc.detail}"
        return exc.stage, _imessage_error("send")
    if isinstance(exc, subprocess.TimeoutExpired):
        return "automation_timeout", "iMessage automation timed out before Jarvis could confirm the send; nothing was retried."
    return "send_error", _imessage_error("send")


def _send_imessage(recipient: str, message: str) -> str:
    safe_recipient = _applescript_string(recipient)
    safe_message = _applescript_string(message)
    script = f'''tell application "Messages"
    set targetService to 1st service whose service type = iMessage
    set targetBuddy to buddy "{safe_recipient}" of targetService
    send "{safe_message}" to targetBuddy
end tell'''
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or "osascript failed"
        stage, detail = _imessage_error_stage(stderr)
        raise IMessageSendError(stage, detail)
    return result.stdout.strip()


def make_imessage_tools(config: JarvisConfig):
    def send_imessage(args: dict[str, Any]) -> ToolResult:
        recipient = _short(args.get("to") or args.get("recipient") or args.get("contact"), MAX_RECIPIENT_CHARS)
        message = _short(args.get("message") or args.get("body") or args.get("text"), MAX_MESSAGE_CHARS)

        if not recipient:
            return ToolResult(
                "send_imessage",
                False,
                "Recipient ('to') is required.",
                _safe_metadata(
                    message_chars=len(message),
                    **_imessage_send_handoff(
                        recipient="",
                        message_chars=len(message),
                        status="refused",
                        reason="missing_recipient",
                    ),
                ),
            )
        if not message:
            return ToolResult(
                "send_imessage",
                False,
                "Message text is required.",
                _safe_metadata(
                    to=_message_preview(recipient, limit=MAX_RECIPIENT_CHARS),
                    message_chars=0,
                    **_imessage_send_handoff(
                        recipient=recipient,
                        message_chars=0,
                        status="refused",
                        reason="missing_message",
                    ),
                ),
            )

        resolved_recipient, resolution_metadata, resolution_message = _resolve_send_recipient(recipient)
        contact_lookup_attempted = _metadata_bool(resolution_metadata.get("contact_lookup_attempted"))
        resolution_metadata = {
            **resolution_metadata,
            "contact_lookup_attempted": contact_lookup_attempted,
        }
        if not resolved_recipient:
            return ToolResult(
                "send_imessage",
                False,
                resolution_message,
                _safe_metadata(
                    reads_personal_data=contact_lookup_attempted,
                    to=_message_preview(recipient, limit=MAX_RECIPIENT_CHARS),
                    message_chars=len(message),
                    reason=resolution_metadata.get("contact_resolution_status"),
                    **resolution_metadata,
                    **_imessage_send_handoff(
                        recipient=recipient,
                        message_chars=len(message),
                        status="refused",
                        reason=str(resolution_metadata.get("contact_resolution_status") or "contact_resolution_failed"),
                        **resolution_metadata,
                    ),
                ),
            )

        try:
            _send_imessage(resolved_recipient, message)
            return ToolResult(
                "send_imessage", True,
                (
                    "Messages accepted the send action to "
                    f"{_message_preview(resolved_recipient, limit=MAX_RECIPIENT_CHARS)}; "
                    "recipient delivery is not confirmed."
                ),
                _safe_metadata(
                    reads_personal_data=contact_lookup_attempted,
                    to=_message_preview(recipient, limit=MAX_RECIPIENT_CHARS),
                    resolved_to=_message_preview(resolved_recipient, limit=MAX_RECIPIENT_CHARS),
                    message_chars=len(message),
                    send_attempted=True,
                    sender_side_send_confirmed=True,
                    recipient_delivery_confirmed=False,
                    delivery_confirmed=False,
                    delivery_evidence="sender_side_only",
                    recipient_confirmation_required=True,
                    requires_approval=True,
                    executes_side_effect=True,
                    external_side_effect=True,
                    **resolution_metadata,
                    **_imessage_send_handoff(
                        recipient=recipient,
                        message_chars=len(message),
                        status="sender_confirmed",
                        send_attempted=True,
                        **resolution_metadata,
                    ),
                ),
            )
        except Exception as e:
            failure_stage, classified_output = _imessage_send_failure(e)
            outcome_unknown = not _send_outcome_is_known_not_sent(failure_stage)
            failure_output = (
                _post_attempt_send_recovery_guidance()
                if outcome_unknown
                else f"{classified_output} {KNOWN_NOT_SENT_SEND_RECOVERY_ACTION}"
            )
            failure_metadata = _safe_metadata(
                to=_message_preview(recipient, limit=MAX_RECIPIENT_CHARS),
                resolved_to=_message_preview(resolved_recipient, limit=MAX_RECIPIENT_CHARS),
                reads_personal_data=contact_lookup_attempted,
                message_chars=len(message),
                send_attempted=True,
                requires_approval=True,
                executes_side_effect=True,
                external_side_effect=True,
                failure_stage=failure_stage,
                imessage_send_stage=failure_stage,
                exception_type=type(e).__name__,
                outcome_known=not outcome_unknown,
                outcome_unknown=outcome_unknown,
                side_effect_possible=outcome_unknown,
                retry_safe=not outcome_unknown,
                automatic_retry_allowed=False,
                authorizes_retry=False,
                **resolution_metadata,
                **_imessage_send_handoff(
                    recipient=recipient,
                    message_chars=len(message),
                    status="outcome_unknown" if outcome_unknown else "failed",
                    reason="send_outcome_unknown" if outcome_unknown else "send_error",
                    send_attempted=True,
                    **resolution_metadata,
                ),
            )
            if outcome_unknown:
                failure_metadata = declare_outcome_unknown_failure(
                    failure_metadata,
                    output=failure_output,
                    commands=("setup check",),
                )
            else:
                failure_metadata = declare_known_not_sent_failure(
                    failure_metadata,
                    output=failure_output,
                    action=KNOWN_NOT_SENT_SEND_RECOVERY_ACTION,
                    commands=("setup check",),
                )
            return ToolResult(
                "send_imessage",
                False,
                failure_output,
                failure_metadata,
            )

    def read_recent_imessages(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 10, 1, 50)
        limit_metadata = _raw_int_metadata(args.get("limit"), key="limit", sanitized=limit)
        try:
            import sqlite3, os
            db_path = os.path.expanduser("~/Library/Messages/chat.db")
            conn = sqlite3.connect(db_path)
            try:
                rows = conn.execute(
                    """
                    SELECT m.text, m.is_from_me, COALESCE(h.id,''), m.date
                    FROM message m
                    LEFT JOIN handle h ON m.handle_id = h.ROWID
                    WHERE m.text IS NOT NULL AND m.text != ''
                    ORDER BY m.ROWID DESC LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            finally:
                conn.close()
            if not rows:
                return ToolResult(
                    "read_recent_imessages",
                    True,
                    "No messages found.",
                    _safe_metadata(
                        reads_personal_data=True,
                        executes_side_effect=False,
                        external_side_effect=False,
                        count=0,
                        **limit_metadata,
                        **_imessage_recent_handoff(limit=limit, rows=[], status="empty"),
                    ),
                )
            lines = [f"Recent {len(rows)} iMessages:"]
            message_rows = [_imessage_recent_row(row, index) for index, row in enumerate(reversed(rows), start=1)]
            for row in message_rows:
                who = row["sender"]
                preview = row["preview"]
                lines.append(f"  [{who}] {preview}")
            return ToolResult(
                "read_recent_imessages",
                True,
                "\n".join(lines),
                _safe_metadata(
                    reads_personal_data=True,
                    executes_side_effect=False,
                    external_side_effect=False,
                    count=len(message_rows),
                    **limit_metadata,
                    **_imessage_recent_handoff(limit=limit, rows=message_rows, status="ok"),
                ),
            )
        except Exception as e:
            failure_output = (
                f"{_imessage_error('read')} {PERSONAL_READ_RECOVERY_ACTION}"
            )
            return ToolResult(
                "read_recent_imessages",
                False,
                failure_output,
                declare_retryable_personal_read_failure(
                    _safe_metadata(
                        reads_personal_data=True,
                        executes_side_effect=False,
                        external_side_effect=False,
                        count=0,
                        exception_type=type(e).__name__,
                        **limit_metadata,
                        **_imessage_recent_handoff(limit=limit, rows=[], status="unavailable", reason="read_error"),
                    ),
                    output=failure_output,
                    action=PERSONAL_READ_RECOVERY_ACTION,
                    commands=("setup check",),
                ),
            )

    from jarvis_v2.tools.registry import Tool
    return [
        Tool(
            "send_imessage",
            "Send an iMessage. Args: to (phone number or email), message.",
            RiskLevel.HIGH_RISK,
            send_imessage,
            "personal",
        ),
        Tool(
            "read_recent_imessages",
            "Read recent iMessages from chat.db. Args: limit (default 10, max 50). Requires Full Disk Access.",
            RiskLevel.LOCAL_SAFE,
            read_recent_imessages,
            "personal",
        ),
    ]
