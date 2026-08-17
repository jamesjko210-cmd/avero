"""Live Gmail connector for Jarvis V2 (SMTP send + IMAP read via app password)."""

from __future__ import annotations

import email as email_lib
import html
import imaplib
import os
import re
import smtplib
import ssl
from email.header import decode_header, make_header
from email.mime.text import MIMEText
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


MAX_SUBJECT_CHARS = 300
MAX_BODY_CHARS = 8000
MAX_TO_CHARS = 200
MAX_SEARCH_CHARS = 200
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
IMAP_SEARCH_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
EMAIL_SEARCH_ARG_NAMES = ("from", "sender", "subject", "text", "query", "about")


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    base = {
        "calls_model": False,
        "calls_external_service": True,
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


def _declare_known_not_sent_input_failure(
    metadata: dict[str, Any], *, output: str
) -> dict[str, Any]:
    return declare_known_not_sent_failure(
        metadata,
        output=output,
        action=KNOWN_NOT_SENT_SEND_RECOVERY_ACTION,
        commands=("setup check",),
    )


def _declare_personal_read_input_failure(
    metadata: dict[str, Any], *, output: str
) -> dict[str, Any]:
    return declare_retryable_personal_read_failure(
        metadata,
        output=output,
        action=PERSONAL_READ_RECOVERY_ACTION,
        commands=("setup check",),
    )


def _short(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text[:limit] if len(text) > limit else text


def _redact_local_paths(value: Any) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", str(value or ""))


def _display_text(value: Any, limit: int) -> str:
    text = " ".join(_redact_local_paths(value).strip().split())
    return text[:limit] if len(text) > limit else text


def _short_raw(value: Any, limit: int) -> str:
    text = _short(value, limit)
    return _redact_local_paths(text)


def _has_local_path(value: Any) -> bool:
    return bool(LOCAL_PATH_RE.search(str(value or "")))


def _metadata_bool(value: Any) -> bool:
    return value is True


def _email_send_boundaries(
    *, external_attempted: bool, send_attempted: bool, contact_lookup_attempted: bool = False
) -> dict[str, bool]:
    requires_approval = external_attempted or send_attempted
    return {
        "calls_model": False,
        "calls_external_service": external_attempted,
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
        "requires_approval": requires_approval,
        "controls_computer": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _email_send_handoff(
    *,
    to: str,
    subject: str,
    body_chars: int,
    status: str,
    reason: str = "",
    external_attempted: bool = False,
    send_attempted: bool = False,
    retry_safe: bool = False,
    outcome_known: bool = True,
    original_to: str = "",
    contact_lookup_attempted: bool = False,
    contact_resolution_status: str = "skipped",
    contact_match_count: int = 0,
    contact_candidates: list[str] | None = None,
) -> dict[str, Any]:
    boundaries = _email_send_boundaries(
        external_attempted=external_attempted,
        send_attempted=send_attempted,
        contact_lookup_attempted=contact_lookup_attempted,
    )
    if status == "outcome_unknown":
        next_safe_commands = ["recent tool runs", "execution recovery"]
    elif status in {"failed", "unavailable"}:
        next_safe_commands = ["setup check"]
    elif status == "sent":
        next_safe_commands = ["recent tool runs"]
    else:
        next_safe_commands = ["safe next actions"]
    next_safe_command = next_safe_commands[0]
    handoff = {
        "source": "send_email",
        "status": status,
        "reason": reason,
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": send_attempted,
        "changed": ["email_send_attempt"] if send_attempted else [],
        "content_in_handoff": False,
        "to": _display_text(to, MAX_TO_CHARS),
        "original_to": _display_text(original_to or to, MAX_TO_CHARS),
        "subject": _display_text(subject, MAX_SUBJECT_CHARS),
        "body_chars": body_chars,
        "contact_lookup_attempted": contact_lookup_attempted,
        "contact_resolution_status": contact_resolution_status,
        "contact_match_count": contact_match_count,
        "contact_candidates": [_display_text(candidate, MAX_TO_CHARS) for candidate in (contact_candidates or [])],
        "content_in_metadata": False,
        "external_attempted": external_attempted,
        "send_attempted": send_attempted,
        "retry_safe": retry_safe,
        "outcome_known": outcome_known,
        "approval_required_before_execution": True,
        "manual_review_required": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "next_safe_command": next_safe_command,
        "next_safe_commands": next_safe_commands,
        "next_safe_command_count": len(next_safe_commands),
        "boundaries": boundaries,
    }
    return {
        "email_send_handoff_ready": True,
        "email_send_ready_for_operator": handoff["ready_for_operator"],
        "email_send_state_changed": handoff["state_changed"],
        "email_send_changed": handoff["changed"],
        "email_send_content_in_handoff": handoff["content_in_handoff"],
        "email_send_content_in_metadata": handoff["content_in_metadata"],
        "email_send_retry_safe": handoff["retry_safe"],
        "email_send_outcome_known": handoff["outcome_known"],
        "email_send_next_safe_command": handoff["next_safe_command"],
        "email_send_next_safe_commands": handoff["next_safe_commands"],
        "email_send_next_safe_command_count": handoff["next_safe_command_count"],
        "email_send_authorizes_execution": handoff["authorizes_execution"],
        "email_send_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "email_send_approval_granted": handoff["approval_granted"],
        "email_send_boundaries": boundaries,
        "ready_for_operator": handoff["ready_for_operator"],
        "state_changed": handoff["state_changed"],
        "changed": handoff["changed"],
        "content_in_handoff": handoff["content_in_handoff"],
        "authorizes_execution": handoff["authorizes_execution"],
        "authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "approval_granted": handoff["approval_granted"],
        "email_send_handoff": handoff,
    }


def _resolve_send_recipient(raw_recipient: str) -> tuple[str | None, dict[str, Any], str]:
    if contacts_connector.looks_like_handle(raw_recipient):
        return raw_recipient, {
            "original_to": _display_text(raw_recipient, MAX_TO_CHARS),
            "contact_lookup_attempted": False,
            "contact_resolution_status": "skipped",
            "contact_match_count": 0,
            "contact_candidates": [],
        }, ""
    matches = contacts_connector.resolve_contact(raw_recipient)
    candidate_names = [match.name for match in matches]
    base = {
        "original_to": _display_text(raw_recipient, MAX_TO_CHARS),
        "contact_lookup_attempted": True,
        "contact_match_count": len(matches),
        "contact_candidates": [_display_text(name, MAX_TO_CHARS) for name in candidate_names],
    }
    if len(matches) == 1 and matches[0].handle:
        return matches[0].handle, {**base, "contact_resolution_status": "resolved"}, ""
    if len(matches) > 1:
        names = " or ".join(_display_text(name, MAX_TO_CHARS) for name in candidate_names)
        return None, {**base, "contact_resolution_status": "ambiguous"}, (
            f"I found {len(matches)} people named \"{_display_text(raw_recipient, MAX_TO_CHARS)}\" — {names}?"
        )
    return None, {**base, "contact_resolution_status": "not_found"}, (
        f"I couldn't find a contact matching \"{_display_text(raw_recipient, MAX_TO_CHARS)}\"."
    )


def _email_read_boundaries(*, external_attempted: bool) -> dict[str, bool]:
    return {
        "calls_model": False,
        "calls_external_service": external_attempted,
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


def _email_read_handoff(
    *,
    limit: int,
    unread_only: bool,
    status: str,
    reason: str = "",
    count: int = 0,
    rows: list[dict[str, str]] | None = None,
    external_attempted: bool = False,
    retry_safe: bool = False,
) -> dict[str, Any]:
    bounded_rows = rows or []
    boundaries = _email_read_boundaries(external_attempted=external_attempted)
    next_safe_command = "read emails" if retry_safe else "read email body"
    handoff = {
        "source": "read_emails",
        "status": status,
        "reason": reason,
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "limit": limit,
        "unread_only": unread_only,
        "count": count,
        "row_count": len(bounded_rows),
        "rows": bounded_rows,
        "headers_only": True,
        "body_in_metadata": False,
        "external_attempted": external_attempted,
        "retry_safe": retry_safe,
        "next_safe_command": next_safe_command,
        "next_safe_commands": [next_safe_command],
        "next_safe_command_count": 1,
        "boundaries": boundaries,
    }
    return {
        "email_read_handoff_ready": True,
        "email_read_ready_for_operator": handoff["ready_for_operator"],
        "email_read_state_changed": handoff["state_changed"],
        "email_read_changed": handoff["changed"],
        "email_read_content_in_handoff": handoff["content_in_handoff"],
        "email_read_body_in_metadata": handoff["body_in_metadata"],
        "email_read_next_safe_command": handoff["next_safe_command"],
        "email_read_next_safe_commands": handoff["next_safe_commands"],
        "email_read_next_safe_command_count": handoff["next_safe_command_count"],
        "email_read_authorizes_execution": handoff["authorizes_execution"],
        "email_read_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "email_read_approval_granted": handoff["approval_granted"],
        "email_read_boundaries": boundaries,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "email_read_handoff": handoff,
    }


def _email_search_boundaries(*, external_attempted: bool) -> dict[str, bool]:
    return {
        "calls_model": False,
        "calls_external_service": external_attempted,
        "executes_tools": False,
        "reads_personal_data": True,
        "reads_private_data": True,
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


def _email_search_handoff(
    *,
    criterion: str,
    limit: int,
    status: str,
    reason: str = "",
    count: int = 0,
    rows: list[dict[str, str]] | None = None,
    external_attempted: bool = False,
    retry_safe: bool = False,
) -> dict[str, Any]:
    bounded_rows = rows or []
    boundaries = _email_search_boundaries(external_attempted=external_attempted)
    next_safe_command = "search emails" if retry_safe else "read email body"
    handoff = {
        "source": "search_emails",
        "status": status,
        "reason": reason,
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "criterion": _display_text(criterion, 240),
        "limit": limit,
        "count": count,
        "row_count": len(bounded_rows),
        "rows": bounded_rows,
        "headers_only": True,
        "body_in_metadata": False,
        "external_attempted": external_attempted,
        "retry_safe": retry_safe,
        "next_safe_command": next_safe_command,
        "next_safe_commands": [next_safe_command],
        "next_safe_command_count": 1,
        "boundaries": boundaries,
    }
    return {
        "email_search_handoff_ready": True,
        "email_search_ready_for_operator": handoff["ready_for_operator"],
        "email_search_state_changed": handoff["state_changed"],
        "email_search_changed": handoff["changed"],
        "email_search_content_in_handoff": handoff["content_in_handoff"],
        "email_search_body_in_metadata": handoff["body_in_metadata"],
        "email_search_next_safe_command": handoff["next_safe_command"],
        "email_search_next_safe_commands": handoff["next_safe_commands"],
        "email_search_next_safe_command_count": handoff["next_safe_command_count"],
        "email_search_authorizes_execution": handoff["authorizes_execution"],
        "email_search_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "email_search_approval_granted": handoff["approval_granted"],
        "email_search_boundaries": boundaries,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "email_search_handoff": handoff,
    }


def _email_body_boundaries(*, external_attempted: bool) -> dict[str, bool]:
    return {
        "calls_model": False,
        "calls_external_service": external_attempted,
        "executes_tools": False,
        "reads_personal_data": True,
        "reads_private_data": True,
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


def _email_body_handoff(
    *,
    criterion: str,
    limit: int,
    selected_index: int,
    status: str,
    reason: str = "",
    count: int = 0,
    body_chars: int = 0,
    message: dict[str, str] | None = None,
    external_attempted: bool = False,
    retry_safe: bool = False,
) -> dict[str, Any]:
    safe_message = message or {}
    boundaries = _email_body_boundaries(external_attempted=external_attempted)
    next_safe_command = "read email body" if retry_safe else "search emails"
    handoff = {
        "source": "read_email_body",
        "status": status,
        "reason": reason,
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "criterion": _display_text(criterion, 240),
        "limit": limit,
        "selected_index": selected_index,
        "count": count,
        "message": safe_message,
        "body_chars": body_chars,
        "body_in_metadata": False,
        "external_attempted": external_attempted,
        "retry_safe": retry_safe,
        "next_safe_command": next_safe_command,
        "next_safe_commands": [next_safe_command],
        "next_safe_command_count": 1,
        "boundaries": boundaries,
    }
    return {
        "email_body_handoff_ready": True,
        "email_body_ready_for_operator": handoff["ready_for_operator"],
        "email_body_state_changed": handoff["state_changed"],
        "email_body_changed": handoff["changed"],
        "email_body_content_in_handoff": handoff["content_in_handoff"],
        "email_body_body_in_metadata": handoff["body_in_metadata"],
        "email_body_next_safe_command": handoff["next_safe_command"],
        "email_body_next_safe_commands": handoff["next_safe_commands"],
        "email_body_next_safe_command_count": handoff["next_safe_command_count"],
        "email_body_authorizes_execution": handoff["authorizes_execution"],
        "email_body_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "email_body_approval_granted": handoff["approval_granted"],
        "email_body_boundaries": boundaries,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "email_body_handoff": handoff,
    }


def _credential_status() -> dict[str, Any]:
    gmail_address = os.getenv("GMAIL_ADDRESS", "").strip()
    app_password = os.getenv("GMAIL_APP_PASSWORD", "").strip()
    address_configured = bool(gmail_address)
    password_configured = bool(app_password)
    address_valid = (
        address_configured
        and not _has_local_path(gmail_address)
        and not any(sep in gmail_address for sep in ("/", "\\"))
        and bool(re.fullmatch(r"[^@\s/\\]+@[^@\s/\\]+\.[^@\s/\\]+", gmail_address))
    )
    password_valid = (
        password_configured
        and not _has_local_path(app_password)
        and not any(sep in app_password for sep in ("/", "\\"))
        and any(ch.isalnum() for ch in app_password)
    )
    return {
        "gmail_address": gmail_address,
        "app_password": app_password,
        "address_configured": address_configured,
        "password_configured": password_configured,
        "address_valid": address_valid,
        "password_valid": password_valid,
        "valid": address_valid and password_valid,
        "reason": "missing_credentials"
        if not address_configured or not password_configured
        else "invalid_credentials",
    }


def _credential_metadata(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "credentials_configured": status["address_configured"] and status["password_configured"],
        "gmail_address_configured": status["address_configured"],
        "gmail_address_valid": status["address_valid"],
        "gmail_app_password_configured": status["password_configured"],
        "gmail_app_password_valid": status["password_valid"],
        "reason": status["reason"],
        "calls_external_service": False,
        "external_attempted": False,
    }


def _credentials_error(status: dict[str, Any]) -> str:
    if status["reason"] == "missing_credentials":
        return "GMAIL_ADDRESS and GMAIL_APP_PASSWORD must be set in .env"
    return "Gmail credentials are invalid; check GMAIL_ADDRESS and GMAIL_APP_PASSWORD in .env (values hidden)."


def _decode(raw_header: str) -> str:
    """Decode a MIME-encoded header (e.g. =?UTF-8?B?...?= used for Korean subjects)."""
    try:
        return str(make_header(decode_header(raw_header or "")))
    except Exception:
        return raw_header or ""


def _bounded_int(value: Any, default: int, low: int = 1, high: int = 20) -> int:
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
        return {key: sanitized, f"raw_{key}": _short_raw(value, 80)}
    return {key: sanitized}


def _is_gmail_auth_error(exc: Exception | None) -> bool:
    if exc is None:
        return False
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        return True
    if isinstance(exc, imaplib.IMAP4.error) and "AUTHENTICATIONFAILED" in str(exc).upper():
        return True
    return False


def _gmail_error(operation: str, exc: Exception | None = None) -> str:
    action = {
        "send": "send that message",
        "read": "read the inbox",
        "search": "search mail",
        "body": "read that message body",
    }.get(operation, "complete that")
    if _is_gmail_auth_error(exc):
        return (
            f"Gmail could not {action} -- the app password was rejected (authentication failed). "
            "Retrying will not help. the operator: generate a fresh Gmail App Password "
            "(myaccount.google.com/apppasswords) and update GMAIL_APP_PASSWORD in .env, then retry."
        )
    return (
        f"Gmail could not {action} right now. Check network access to Gmail, confirm "
        "GMAIL_ADDRESS and GMAIL_APP_PASSWORD are present in .env, run setup check, then retry."
    )


def _gmail_uncertain_send_error() -> str:
    return (
        "Gmail's send outcome is unknown. The message may already have been sent, "
        "so do not retry blindly. Check Gmail Sent for the recipient and subject first. "
        "If absent, run `setup check`, repair Gmail access, then submit a new approved send."
    )


class _IMAPProtocolError(RuntimeError):
    def __init__(self, operation: str, status: str):
        super().__init__(f"imap_{operation}_{status}")
        self.operation = operation
        self.status = status


def _imap_status(value: object) -> str:
    if isinstance(value, bytes):
        text = value.decode("ascii", errors="replace")
    else:
        text = str(value or "")
    normalized = text.strip().upper()
    return normalized if normalized in {"OK", "NO", "BAD", "BYE", "PREAUTH"} else "MALFORMED"


def _imap_data(response: object, operation: str) -> object:
    if not isinstance(response, (tuple, list)) or len(response) != 2:
        raise _IMAPProtocolError(operation, "MALFORMED")
    status = _imap_status(response[0])
    if status != "OK":
        raise _IMAPProtocolError(operation, status)
    return response[1]


def _imap_search_ids(data: object) -> list[bytes]:
    if not isinstance(data, (tuple, list)) or not data:
        raise _IMAPProtocolError("search", "MALFORMED")
    value = data[0]
    if value in (None, b"", ""):
        return []
    if isinstance(value, str):
        value = value.encode("ascii", errors="ignore")
    if not isinstance(value, bytes):
        raise _IMAPProtocolError("search", "MALFORMED")
    return value.split()


def _imap_fetch_bytes(data: object) -> bytes:
    if not isinstance(data, (tuple, list)) or not data:
        raise _IMAPProtocolError("fetch", "MALFORMED")
    first = data[0]
    if (
        not isinstance(first, (tuple, list))
        or len(first) < 2
        or not isinstance(first[1], bytes)
        or not first[1]
    ):
        raise _IMAPProtocolError("fetch", "MALFORMED")
    return first[1]


def _imap_failure_metadata(exc: Exception) -> dict[str, Any]:
    if not isinstance(exc, _IMAPProtocolError):
        return {}
    return {
        "imap_operation": exc.operation,
        "imap_status": exc.status,
    }


def _strip_html(value: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value or "")
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</p\s*>", "\n", text)
    text = re.sub(r"(?is)<.*?>", " ", text)
    return html.unescape(re.sub(r"[ \t\r\f\v]+", " ", text)).strip()


def _message_body(parsed: email_lib.message.Message) -> str:
    candidates: list[tuple[str, bytes, str]] = []
    if parsed.is_multipart():
        for part in parsed.walk():
            if part.get_content_maintype() == "multipart":
                continue
            disposition = (part.get("Content-Disposition") or "").lower()
            if "attachment" in disposition:
                continue
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            candidates.append((part.get_content_type(), payload, part.get_content_charset() or "utf-8"))
    else:
        payload = parsed.get_payload(decode=True)
        if payload is not None:
            candidates.append((parsed.get_content_type(), payload, parsed.get_content_charset() or "utf-8"))

    plain_parts: list[str] = []
    html_parts: list[str] = []
    for content_type, payload, charset in candidates:
        try:
            text = payload.decode(charset, errors="replace")
        except LookupError:
            text = payload.decode("utf-8", errors="replace")
        if content_type == "text/plain":
            plain_parts.append(text.strip())
        elif content_type == "text/html":
            html_parts.append(_strip_html(text))
    body = "\n\n".join(part for part in plain_parts if part) or "\n\n".join(part for part in html_parts if part)
    return _short(_redact_local_paths(body), MAX_BODY_CHARS)


def _search_criterion(args: dict[str, Any]) -> str:
    def quoted(value: Any) -> str:
        text = _short(value, MAX_SEARCH_CHARS)
        return text.replace("\\", "\\\\").replace('"', '\\"')

    sender = quoted(args.get("from") or args.get("sender"))
    subject = quoted(args.get("subject"))
    text = quoted(args.get("text") or args.get("query") or args.get("about"))
    parts: list[str] = []
    if sender:
        parts.append(f'FROM "{sender}"')
    if subject:
        parts.append(f'SUBJECT "{subject}"')
    if text:
        parts.append(f'TEXT "{text}"')
    return " ".join(parts) if parts else "ALL"


def _imap_search(
    imap: Any,
    charset: str | None,
    criteria: tuple[str | bytes, ...],
    *,
    literal: bytes | None = None,
) -> object:
    if literal is not None:
        imap.literal = literal
    try:
        return imap.search(charset, *criteria)
    except imaplib.IMAP4.error as exc:
        raise _IMAPProtocolError("search", "BAD") from exc
    finally:
        if literal is not None:
            imap.literal = None


def _imap_search_ids_for_args(imap: Any, criterion: str, args: dict[str, Any]) -> list[bytes]:
    """Search Gmail with UTF-8 literals and intersect structured conditions."""
    if criterion.isascii():
        return _imap_search_ids(_imap_data(_imap_search(imap, None, (criterion,)), "search"))

    terms = (
        ("FROM", args.get("from") or args.get("sender")),
        ("SUBJECT", args.get("subject")),
        ("TEXT", args.get("text") or args.get("query") or args.get("about")),
    )
    matched_ids: list[bytes] | None = None
    for search_key, raw_value in terms:
        value = _short(raw_value, MAX_SEARCH_CHARS)
        if not value:
            continue
        if value.isascii():
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            response = _imap_search(imap, None, (f'{search_key} "{escaped}"',))
        else:
            response = _imap_search(
                imap,
                "UTF-8",
                (search_key.encode("ascii"),),
                literal=value.encode("utf-8"),
            )
        ids = _imap_search_ids(_imap_data(response, "search"))
        if matched_ids is None:
            matched_ids = ids
        else:
            retained = set(ids)
            matched_ids = [message_id for message_id in matched_ids if message_id in retained]
        if not matched_ids:
            return []
    return matched_ids or []


def _invalid_imap_search_term(args: dict[str, Any]) -> str:
    for key in EMAIL_SEARCH_ARG_NAMES:
        value = args.get(key)
        if value is None:
            continue
        if _has_local_path(value):
            return "<local-path>"
        if IMAP_SEARCH_CONTROL_RE.search(str(value)):
            return "<invalid-search-term>"
    return ""


def make_email_tools(config: JarvisConfig):
    def send_email(args: dict[str, Any]) -> ToolResult:
        credentials = _credential_status()
        gmail_address = credentials["gmail_address"]
        app_password = credentials["app_password"]
        if not credentials["valid"]:
            to = _short(args.get("to") or args.get("recipient"), MAX_TO_CHARS)
            subject = _short(args.get("subject") or args.get("title") or "(no subject)", MAX_SUBJECT_CHARS)
            body = _short(args.get("body") or args.get("message") or args.get("content") or "", MAX_BODY_CHARS)
            failure_output = (
                f"{_credentials_error(credentials)} {KNOWN_NOT_SENT_SEND_RECOVERY_ACTION}"
            )
            return ToolResult(
                "send_email", False,
                failure_output,
                _declare_known_not_sent_input_failure(
                    _safe_metadata(
                        body_chars=len(body),
                        **_credential_metadata(credentials),
                        **_email_send_handoff(
                            to=to,
                            subject=subject,
                            body_chars=len(body),
                            status="unavailable",
                            reason=credentials["reason"],
                        ),
                    ),
                    output=failure_output,
                ),
            )

        to = _short(args.get("to") or args.get("recipient"), MAX_TO_CHARS)
        subject = _short(args.get("subject") or args.get("title") or "(no subject)", MAX_SUBJECT_CHARS)
        body = _short(args.get("body") or args.get("message") or args.get("content") or "", MAX_BODY_CHARS)

        if not to:
            failure_output = (
                f"Recipient ('to') is required. {KNOWN_NOT_SENT_SEND_RECOVERY_ACTION}"
            )
            return ToolResult(
                "send_email",
                False,
                failure_output,
                _declare_known_not_sent_input_failure(
                    _safe_metadata(
                        subject=_display_text(subject, MAX_SUBJECT_CHARS),
                        body_chars=len(body),
                        **_email_send_handoff(
                            to="",
                            subject=subject,
                            body_chars=len(body),
                            status="refused",
                            reason="missing_recipient",
                        ),
                    ),
                    output=failure_output,
                ),
            )
        if not body:
            failure_output = f"Email body is required. {KNOWN_NOT_SENT_SEND_RECOVERY_ACTION}"
            return ToolResult(
                "send_email",
                False,
                failure_output,
                _declare_known_not_sent_input_failure(
                    _safe_metadata(
                        to=_display_text(to, MAX_TO_CHARS),
                        subject=_display_text(subject, MAX_SUBJECT_CHARS),
                        body_chars=0,
                        **_email_send_handoff(
                            to=to,
                            subject=subject,
                            body_chars=0,
                            status="refused",
                            reason="missing_body",
                        ),
                    ),
                    output=failure_output,
                ),
            )

        resolved_to, resolution_metadata, resolution_message = _resolve_send_recipient(to)
        resolution_metadata = dict(resolution_metadata)
        contact_lookup_attempted = _metadata_bool(resolution_metadata.get("contact_lookup_attempted"))
        resolution_metadata["contact_lookup_attempted"] = contact_lookup_attempted
        if not resolved_to:
            failure_output = f"{resolution_message} {KNOWN_NOT_SENT_SEND_RECOVERY_ACTION}"
            return ToolResult(
                "send_email",
                False,
                failure_output,
                _declare_known_not_sent_input_failure(
                    _safe_metadata(
                        reads_personal_data=contact_lookup_attempted,
                        to=_display_text(to, MAX_TO_CHARS),
                        subject=_display_text(subject, MAX_SUBJECT_CHARS),
                        body_chars=len(body),
                        reason=resolution_metadata.get("contact_resolution_status"),
                        external_attempted=False,
                        send_attempted=False,
                        **resolution_metadata,
                        **_email_send_handoff(
                            to=to,
                            subject=subject,
                            body_chars=len(body),
                            status="refused",
                            reason=str(
                                resolution_metadata.get("contact_resolution_status")
                                or "contact_resolution_failed"
                            ),
                            **resolution_metadata,
                        ),
                    ),
                    output=failure_output,
                ),
            )

        smtp_attempted = False
        send_attempted = False
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = gmail_address
            msg["To"] = resolved_to
            ctx = ssl.create_default_context()
            smtp_attempted = True
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as smtp:
                smtp.login(gmail_address, app_password)
                send_attempted = True
                smtp.sendmail(gmail_address, [resolved_to], msg.as_string())
            return ToolResult(
                "send_email", True,
                f"Email sent to {_display_text(resolved_to, MAX_TO_CHARS)}: \"{_display_text(subject, MAX_SUBJECT_CHARS)}\"",
                _safe_metadata(
                    reads_personal_data=contact_lookup_attempted,
                    executes_side_effect=True,
                    external_side_effect=True,
                    requires_approval=True,
                    to=_display_text(to, MAX_TO_CHARS),
                    resolved_to=_display_text(resolved_to, MAX_TO_CHARS),
                    subject=_display_text(subject, MAX_SUBJECT_CHARS),
                    body_chars=len(body),
                    external_attempted=True,
                    send_attempted=True,
                    **resolution_metadata,
                    **_email_send_handoff(
                        to=to,
                        subject=subject,
                        body_chars=len(body),
                        status="sent",
                        external_attempted=True,
                        send_attempted=True,
                        **resolution_metadata,
                    ),
                ),
            )
        except Exception as e:
            outcome_uncertain = send_attempted
            retry_safe = not outcome_uncertain and not _is_gmail_auth_error(e)
            status = "outcome_unknown" if outcome_uncertain else "failed"
            reason = "send_outcome_unknown" if outcome_uncertain else "send_error"
            failure_output = (
                _gmail_uncertain_send_error()
                if outcome_uncertain
                else f"{_gmail_error('send', e)} {KNOWN_NOT_SENT_SEND_RECOVERY_ACTION}"
            )
            failure_metadata = _safe_metadata(
                to=_display_text(to, MAX_TO_CHARS),
                resolved_to=_display_text(resolved_to, MAX_TO_CHARS),
                subject=_display_text(subject, MAX_SUBJECT_CHARS),
                body_chars=len(body),
                external_attempted=smtp_attempted,
                send_attempted=send_attempted,
                retry_safe=retry_safe,
                outcome_known=not outcome_uncertain,
                outcome_unknown=outcome_uncertain,
                side_effect_possible=outcome_uncertain,
                automatic_retry_allowed=False,
                authorizes_retry=False,
                reads_personal_data=contact_lookup_attempted,
                requires_approval=smtp_attempted or send_attempted,
                executes_side_effect=send_attempted,
                external_side_effect=send_attempted,
                exception_type=type(e).__name__,
                **resolution_metadata,
                **_email_send_handoff(
                    to=to,
                    subject=subject,
                    body_chars=len(body),
                    status=status,
                    reason=reason,
                    external_attempted=smtp_attempted,
                    send_attempted=send_attempted,
                    retry_safe=retry_safe,
                    outcome_known=not outcome_uncertain,
                    **resolution_metadata,
                ),
            )
            if outcome_uncertain:
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
                "send_email",
                False,
                failure_output,
                failure_metadata,
            )

    def read_emails(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 5)
        credentials = _credential_status()
        gmail_address = credentials["gmail_address"]
        app_password = credentials["app_password"]
        unread_only = _metadata_bool(args.get("unread"))
        if not credentials["valid"]:
            failure_output = (
                f"{_credentials_error(credentials)} {PERSONAL_READ_RECOVERY_ACTION}"
            )
            return ToolResult(
                "read_emails", False,
                failure_output,
                declare_retryable_personal_read_failure(
                    _safe_metadata(
                        reads_personal_data=True,
                        **_credential_metadata(credentials),
                        **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
                        **_email_read_handoff(
                            limit=limit,
                            unread_only=unread_only,
                            status="unavailable",
                            reason=credentials["reason"],
                        ),
                    ),
                    output=failure_output,
                    action=PERSONAL_READ_RECOVERY_ACTION,
                    commands=("setup check",),
                ),
            )
        imap_attempted = False
        try:
            imap_attempted = True
            imap = imaplib.IMAP4_SSL("imap.gmail.com", 993)
            try:
                _imap_data(imap.login(gmail_address, app_password), "login")
                _imap_data(imap.select("INBOX", readonly=True), "select")  # readonly: never marks mail as read
                criterion = "UNSEEN" if unread_only else "ALL"
                data = _imap_data(imap.search(None, criterion), "search")
                ids = _imap_search_ids(data)
                if not ids:
                    label = "unread emails" if unread_only else "emails"
                    return ToolResult(
                        "read_emails",
                        True,
                        f"No {label} in your inbox.",
                        _safe_metadata(
                            reads_personal_data=True,
                            count=0,
                            external_attempted=True,
                            **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
                            **_email_read_handoff(
                                limit=limit,
                                unread_only=unread_only,
                                status="empty",
                                count=0,
                                external_attempted=True,
                            ),
                        ),
                    )
                latest = ids[-limit:][::-1]
                lines = [f"{'Unread' if unread_only else 'Latest'} emails ({len(latest)}):"]
                rows: list[dict[str, str]] = []
                for mid in latest:
                    # PEEK keeps messages unread; fetch headers only, never bodies.
                    msg_data = _imap_data(
                        imap.fetch(mid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])"),
                        "fetch",
                    )
                    raw = _imap_fetch_bytes(msg_data)
                    parsed = email_lib.message_from_bytes(raw)
                    frm = _decode(parsed.get("From", ""))
                    subj = _decode(parsed.get("Subject", "(no subject)"))
                    safe_from = _display_text(frm, 60)
                    safe_subject = _display_text(subj, 90)
                    rows.append({"from": safe_from, "subject": safe_subject})
                    lines.append(f"  • {safe_from} — {safe_subject}")
                return ToolResult(
                    "read_emails",
                    True,
                    "\n".join(lines),
                    _safe_metadata(
                        reads_personal_data=True,
                        count=len(latest),
                        external_attempted=True,
                        **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
                        **_email_read_handoff(
                            limit=limit,
                            unread_only=unread_only,
                            status="ok",
                            count=len(latest),
                            rows=rows,
                            external_attempted=True,
                        ),
                    ),
                )
            finally:
                try:
                    imap.logout()
                except Exception:
                    pass
        except Exception as e:
            failure_output = (
                f"{_gmail_error('read', e)} {PERSONAL_READ_RECOVERY_ACTION}"
            )
            return ToolResult(
                "read_emails",
                False,
                failure_output,
                declare_retryable_personal_read_failure(
                    _safe_metadata(
                        reads_personal_data=True,
                        external_attempted=imap_attempted,
                        exception_type=type(e).__name__,
                        **_imap_failure_metadata(e),
                        **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
                        **_email_read_handoff(
                            limit=limit,
                            unread_only=unread_only,
                            status="failed",
                            reason="read_error",
                            external_attempted=imap_attempted,
                            retry_safe=True,
                        ),
                    ),
                    output=failure_output,
                    action=PERSONAL_READ_RECOVERY_ACTION,
                    commands=("setup check",),
                )
            )

    def search_emails(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 10)
        invalid_criterion = _invalid_imap_search_term(args)
        if invalid_criterion:
            failure_output = (
                "Email search terms cannot contain local file paths or control characters. "
                f"{PERSONAL_READ_RECOVERY_ACTION}"
            )
            return ToolResult(
                "search_emails",
                False,
                failure_output,
                _declare_personal_read_input_failure(
                    _safe_metadata(
                        reads_personal_data=True,
                        reads_private_data=True,
                        calls_external_service=False,
                        reason="invalid_search_term",
                        criterion=invalid_criterion,
                        **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
                        **_email_search_handoff(
                            criterion=invalid_criterion,
                            limit=limit,
                            status="refused",
                            reason="invalid_search_term",
                            retry_safe=True,
                        ),
                    ),
                    output=failure_output,
                ),
            )
        credentials = _credential_status()
        gmail_address = credentials["gmail_address"]
        app_password = credentials["app_password"]
        criterion = _search_criterion(args)
        if not credentials["valid"]:
            failure_output = (
                f"{_credentials_error(credentials)} {PERSONAL_READ_RECOVERY_ACTION}"
            )
            return ToolResult(
                "search_emails", False,
                failure_output,
                declare_retryable_personal_read_failure(
                    _safe_metadata(
                        reads_personal_data=True,
                        reads_private_data=True,
                        **_credential_metadata(credentials),
                        **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
                        **_email_search_handoff(
                            criterion=criterion,
                            limit=limit,
                            status="unavailable",
                            reason=credentials["reason"],
                        ),
                    ),
                    output=failure_output,
                    action=PERSONAL_READ_RECOVERY_ACTION,
                    commands=("setup check",),
                ),
            )
        imap_attempted = False
        try:
            imap_attempted = True
            imap = imaplib.IMAP4_SSL("imap.gmail.com", 993)
            try:
                _imap_data(imap.login(gmail_address, app_password), "login")
                _imap_data(imap.select("INBOX", readonly=True), "select")
                ids = _imap_search_ids_for_args(imap, criterion, args)
                if not ids:
                    return ToolResult(
                        "search_emails",
                        True,
                        "No matching emails found.",
                        _safe_metadata(
                            reads_personal_data=True,
                            reads_private_data=True,
                            external_attempted=True,
                            count=0,
                            criterion=_display_text(criterion, 240),
                            **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
                            **_email_search_handoff(
                                criterion=criterion,
                                limit=limit,
                                status="empty",
                                count=0,
                                external_attempted=True,
                            ),
                        ),
                    )
                latest = ids[-limit:][::-1]
                lines = [f"Matching emails ({len(latest)}):"]
                rows: list[dict[str, str]] = []
                for mid in latest:
                    msg_data = _imap_data(
                        imap.fetch(mid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])"),
                        "fetch",
                    )
                    raw = _imap_fetch_bytes(msg_data)
                    parsed = email_lib.message_from_bytes(raw)
                    frm = _decode(parsed.get("From", ""))
                    subj = _decode(parsed.get("Subject", "(no subject)"))
                    date = _decode(parsed.get("Date", ""))
                    safe_date = _display_text(date, 40)
                    safe_from = _display_text(frm, 60)
                    safe_subject = _display_text(subj, 90)
                    rows.append({"date": safe_date, "from": safe_from, "subject": safe_subject})
                    lines.append(f"  • {safe_date} — {safe_from} — {safe_subject}")
                return ToolResult(
                    "search_emails",
                    True,
                    "\n".join(lines),
                    _safe_metadata(
                        reads_personal_data=True,
                        reads_private_data=True,
                        external_attempted=True,
                        count=len(latest),
                        criterion=_display_text(criterion, 240),
                        **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
                        **_email_search_handoff(
                            criterion=criterion,
                            limit=limit,
                            status="ok",
                            count=len(latest),
                            rows=rows,
                            external_attempted=True,
                        ),
                    ),
                )
            finally:
                try:
                    imap.logout()
                except Exception:
                    pass
        except Exception as e:
            failure_output = (
                f"{_gmail_error('search', e)} {PERSONAL_READ_RECOVERY_ACTION}"
            )
            return ToolResult(
                "search_emails",
                False,
                failure_output,
                declare_retryable_personal_read_failure(
                    _safe_metadata(
                        reads_personal_data=True,
                        reads_private_data=True,
                        external_attempted=imap_attempted,
                        exception_type=type(e).__name__,
                        **_imap_failure_metadata(e),
                        **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
                        **_email_search_handoff(
                            criterion=criterion,
                            limit=limit,
                            status="failed",
                            reason="search_error",
                            external_attempted=imap_attempted,
                            retry_safe=True,
                        ),
                    ),
                    output=failure_output,
                    action=PERSONAL_READ_RECOVERY_ACTION,
                    commands=("setup check",),
                ),
            )

    def read_email_body(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 10)
        index = _bounded_int(args.get("index"), 1, low=1, high=limit)
        raw_index = args.get("index")
        requested_index: int | None = None
        if not isinstance(raw_index, bool):
            try:
                requested_index = int(raw_index)
            except (TypeError, ValueError):
                pass
        if requested_index is not None and requested_index > limit:
            failure_output = (
                "Email message index cannot exceed the requested result limit. "
                f"{PERSONAL_READ_RECOVERY_ACTION}"
            )
            return ToolResult(
                "read_email_body",
                False,
                failure_output,
                _declare_personal_read_input_failure(
                    _safe_metadata(
                        reads_personal_data=True,
                        reads_private_data=True,
                        calls_external_service=False,
                        reason="index_exceeds_limit",
                        selected_index=requested_index,
                        **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
                        **_email_body_handoff(
                            criterion="<not-evaluated>",
                            limit=limit,
                            selected_index=requested_index,
                            status="refused",
                            reason="index_exceeds_limit",
                            retry_safe=True,
                        ),
                    ),
                    output=failure_output,
                ),
            )
        invalid_criterion = _invalid_imap_search_term(args)
        if invalid_criterion:
            failure_output = (
                "Email body search terms cannot contain local file paths or control characters. "
                f"{PERSONAL_READ_RECOVERY_ACTION}"
            )
            return ToolResult(
                "read_email_body",
                False,
                failure_output,
                _declare_personal_read_input_failure(
                    _safe_metadata(
                        reads_personal_data=True,
                        reads_private_data=True,
                        calls_external_service=False,
                        reason="invalid_search_term",
                        criterion=invalid_criterion,
                        selected_index=index,
                        **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
                        **_email_body_handoff(
                            criterion=invalid_criterion,
                            limit=limit,
                            selected_index=index,
                            status="refused",
                            reason="invalid_search_term",
                            retry_safe=True,
                        ),
                    ),
                    output=failure_output,
                ),
            )
        credentials = _credential_status()
        gmail_address = credentials["gmail_address"]
        app_password = credentials["app_password"]
        criterion = _search_criterion(args)
        if not credentials["valid"]:
            failure_output = (
                f"{_credentials_error(credentials)} {PERSONAL_READ_RECOVERY_ACTION}"
            )
            return ToolResult(
                "read_email_body", False,
                failure_output,
                declare_retryable_personal_read_failure(
                    _safe_metadata(
                        reads_personal_data=True,
                        reads_private_data=True,
                        **_credential_metadata(credentials),
                        **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
                        **_email_body_handoff(
                            criterion=criterion,
                            limit=limit,
                            selected_index=index,
                            status="unavailable",
                            reason=credentials["reason"],
                        ),
                    ),
                    output=failure_output,
                    action=PERSONAL_READ_RECOVERY_ACTION,
                    commands=("setup check",),
                ),
            )
        imap_attempted = False
        try:
            imap_attempted = True
            imap = imaplib.IMAP4_SSL("imap.gmail.com", 993)
            try:
                _imap_data(imap.login(gmail_address, app_password), "login")
                _imap_data(imap.select("INBOX", readonly=True), "select")
                ids = _imap_search_ids_for_args(imap, criterion, args)
                if not ids:
                    return ToolResult(
                        "read_email_body",
                        True,
                        "No matching emails found.",
                        _safe_metadata(
                            reads_personal_data=True,
                            reads_private_data=True,
                            external_attempted=True,
                            count=0,
                            criterion=_display_text(criterion, 240),
                            selected_index=index,
                            **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
                            **_email_body_handoff(
                                criterion=criterion,
                                limit=limit,
                                selected_index=index,
                                status="empty",
                                count=0,
                                external_attempted=True,
                            ),
                        ),
                    )
                latest = ids[-limit:][::-1]
                if index > len(latest):
                    failure_output = (
                        f"Email message index {index} is unavailable; the search returned "
                        f"{len(latest)} message(s). {PERSONAL_READ_RECOVERY_ACTION}"
                    )
                    return ToolResult(
                        "read_email_body",
                        False,
                        failure_output,
                        _declare_personal_read_input_failure(
                            _safe_metadata(
                                reads_personal_data=True,
                                reads_private_data=True,
                                external_attempted=True,
                                count=len(latest),
                                criterion=_display_text(criterion, 240),
                                selected_index=index,
                                reason="index_not_found",
                                **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
                                **_email_body_handoff(
                                    criterion=criterion,
                                    limit=limit,
                                    selected_index=index,
                                    status="refused",
                                    reason="index_not_found",
                                    count=len(latest),
                                    external_attempted=True,
                                    retry_safe=True,
                                ),
                            ),
                            output=failure_output,
                        ),
                    )
                selected = latest[index - 1]
                msg_data = _imap_data(imap.fetch(selected, "(BODY.PEEK[])"), "fetch")
                raw = _imap_fetch_bytes(msg_data)
                parsed = email_lib.message_from_bytes(raw)
                frm = _decode(parsed.get("From", ""))
                subj = _decode(parsed.get("Subject", "(no subject)"))
                date = _decode(parsed.get("Date", ""))
                body = _message_body(parsed) or "(No readable body found.)"
                message = {
                    "date": _display_text(date, 40),
                    "from": _display_text(frm, 60),
                    "subject": _display_text(subj, 90),
                }
                output = "\n".join(
                    [
                        f"From: {_display_text(frm, 120)}",
                        f"Date: {_display_text(date, 80)}",
                        f"Subject: {_display_text(subj, 160)}",
                        "",
                        body,
                    ]
                )
                return ToolResult(
                    "read_email_body",
                    True,
                    output,
                    _safe_metadata(
                        reads_personal_data=True,
                        reads_private_data=True,
                        external_attempted=True,
                        criterion=_display_text(criterion, 240),
                        selected_index=index,
                        body_chars=len(body),
                        **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
                        **_email_body_handoff(
                            criterion=criterion,
                            limit=limit,
                            selected_index=index,
                            status="ok",
                            count=len(latest),
                            body_chars=len(body),
                            message=message,
                            external_attempted=True,
                        ),
                    ),
                )
            finally:
                try:
                    imap.logout()
                except Exception:
                    pass
        except Exception as e:
            failure_output = (
                f"{_gmail_error('body', e)} {PERSONAL_READ_RECOVERY_ACTION}"
            )
            return ToolResult(
                "read_email_body",
                False,
                failure_output,
                declare_retryable_personal_read_failure(
                    _safe_metadata(
                        reads_personal_data=True,
                        reads_private_data=True,
                        external_attempted=imap_attempted,
                        exception_type=type(e).__name__,
                        **_imap_failure_metadata(e),
                        **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
                        **_email_body_handoff(
                            criterion=criterion,
                            limit=limit,
                            selected_index=index,
                            status="failed",
                            reason="body_error",
                            external_attempted=imap_attempted,
                            retry_safe=True,
                        ),
                    ),
                    output=failure_output,
                    action=PERSONAL_READ_RECOVERY_ACTION,
                    commands=("setup check",),
                ),
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
    boolean_type = frozenset({ToolArgumentType.BOOLEAN})
    search_fields = tuple(
        ToolArgumentSpec(name, string_type, False)
        for name in ("from", "sender", "subject", "text", "query", "about")
    )
    read_emails_contract = ToolArgumentContract(
        version=TOOL_ARGUMENT_CONTRACT_VERSION,
        fields=(
            ToolArgumentSpec("limit", integer_type, False, 1, 20),
            ToolArgumentSpec("unread", boolean_type, False),
        ),
        allow_unknown=False,
    )
    search_emails_contract = ToolArgumentContract(
        version=TOOL_ARGUMENT_CONTRACT_VERSION,
        fields=search_fields + (ToolArgumentSpec("limit", integer_type, False, 1, 20),),
        allow_unknown=False,
    )
    read_email_body_contract = ToolArgumentContract(
        version=TOOL_ARGUMENT_CONTRACT_VERSION,
        fields=search_fields
        + (
            ToolArgumentSpec("index", integer_type, False, 1, 20),
            ToolArgumentSpec("limit", integer_type, False, 1, 20),
        ),
        allow_unknown=False,
    )

    return [
        Tool(
            "send_email",
            "Send an email via Gmail. Args: to (address), subject, body. Requires GMAIL_ADDRESS and GMAIL_APP_PASSWORD in env.",
            RiskLevel.HIGH_RISK,
            send_email,
            "personal",
        ),
        Tool(
            "read_emails",
            "Read recent email headers via Gmail IMAP (metadata only: sender + subject, no bodies). Args: limit (default 5), unread (true for unread only).",
            RiskLevel.LOCAL_SAFE,
            read_emails,
            "personal",
            argument_contract=read_emails_contract,
        ),
        Tool(
            "search_emails",
            "Search Gmail via IMAP and return matching email metadata. Args: from/sender, subject, text/query/about, limit.",
            RiskLevel.PERSONAL_DATA,
            search_emails,
            "personal",
            argument_contract=search_emails_contract,
        ),
        Tool(
            "read_email_body",
            "Read one Gmail message body via IMAP. Args: from/sender, subject, text/query/about, index, limit.",
            RiskLevel.PERSONAL_DATA,
            read_email_body,
            "personal",
            argument_contract=read_email_body_contract,
        ),
    ]
