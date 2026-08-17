"""Best-effort KakaoTalk sender for Jarvis V2.

This connector uses macOS GUI automation because KakaoTalk does not provide a
stable personal-use messaging API. It requires the KakaoTalk Mac app to be
installed and Accessibility permission for the Python process that runs Jarvis.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import subprocess
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    KNOWN_NOT_SENT_SEND_RECOVERY_ACTION,
    declare_known_not_sent_failure,
    declare_outcome_unknown_failure,
)
from jarvis_v2.agent.types import ApprovalArgumentResolution, RiskLevel, ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.tools import clipboard_safety, contacts_connector


MAX_RECIPIENT_CHARS = 120
MAX_MESSAGE_CHARS = 2000
KAKAO_APPROVAL_BINDING_KEY = "_kakao_send_binding"
KAKAO_TARGET_MODE_KEY = "target_mode"
KAKAO_SEARCH_EXACT_TITLE_MODE = "search_exact_title"
KAKAO_PREOPENED_EXACT_CHAT_MODE = "preopened_exact_chat"
KAKAO_TARGET_MODES = frozenset(
    {KAKAO_SEARCH_EXACT_TITLE_MODE, KAKAO_PREOPENED_EXACT_CHAT_MODE}
)
_KAKAO_APPROVAL_BINDING_DOMAIN = b"jarvis:kakao-send-approval:v1\x00"
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)


def _post_attempt_send_recovery_guidance() -> str:
    return (
        "The KakaoTalk delivery outcome is unknown. Check the exact conversation; do not resend "
        "automatically. If the message is absent, run `setup check`, repair login and macOS "
        "Automation/Accessibility permissions, then submit a new approved send."
    )


def _send_outcome_is_known_not_sent(stage: str) -> bool:
    return stage.startswith("recipient_phase_") or stage in {
        "guard_stopped",
        "macos_automation_permission",
        "macos_accessibility_permission",
        "kakaotalk_unavailable",
        "applescript_compile_failed",
        "clipboard_inspection",
        "clipboard_read",
        "clipboard_rich_content",
        "clipboard_custody_before_enter",
    }


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    base = {
        "calls_model": False,
        "calls_external_service": False,
        "executes_tools": False,
        "reads_personal_data": False,
        "reads_private_data": False,
        "reads_clipboard": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    base.update(extra)
    return base


def _short(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text[:limit] if len(text) > limit else text


def _contains_control_characters(value: str) -> bool:
    return any(ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F for char in value)


def _kakao_raw_argument_failure(recipient: Any, message: Any) -> tuple[str, str]:
    """Validate the exact values before they can become approval authority."""
    if type(recipient) is not str or not recipient.strip():
        return "missing_recipient", "Kakao recipient ('to') is required."
    if len(recipient) > MAX_RECIPIENT_CHARS:
        return "recipient_too_long", f"Kakao recipient must be at most {MAX_RECIPIENT_CHARS} characters."
    if _contains_control_characters(recipient):
        return "recipient_has_control_characters", "Kakao recipient cannot contain control characters."
    if recipient.strip().startswith("@"):
        return "recipient_handle_not_allowed", "Use the exact KakaoTalk display name, not an @handle."
    if type(message) is not str or not message.strip():
        return "missing_message", "Kakao message text is required."
    if len(message) > MAX_MESSAGE_CHARS:
        return "message_too_long", f"Kakao message must be at most {MAX_MESSAGE_CHARS} characters."
    if _contains_control_characters(message):
        return "message_has_control_characters", "Kakao message cannot contain control characters."
    return "", ""


def _kakao_target_mode(value: Any) -> tuple[str | None, str, str]:
    """Return one exact execution mode without silently accepting near-misses."""
    if value is None:
        return KAKAO_SEARCH_EXACT_TITLE_MODE, "", ""
    if type(value) is not str or value not in KAKAO_TARGET_MODES:
        return (
            None,
            "target_mode_invalid",
            "Kakao target mode must be exactly `search_exact_title` or `preopened_exact_chat`.",
        )
    return value, "", ""


def _kakao_approval_binding(
    recipient: str,
    message: str,
    target_mode: str = KAKAO_SEARCH_EXACT_TITLE_MODE,
) -> str:
    payload = json.dumps(
        {"message": message, "target_mode": target_mode, "to": recipient},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(_KAKAO_APPROVAL_BINDING_DOMAIN + payload).hexdigest()


def _display_text(value: Any, limit: int = MAX_RECIPIENT_CHARS) -> str:
    text = " ".join(str(value or "").strip().split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    return text[:limit] if len(text) > limit else text


def _metadata_bool(value: Any) -> bool:
    return value is True


def _kakao_send_boundaries(
    *,
    send_attempted: bool,
    controls_computer: bool,
    contact_lookup_attempted: bool = False,
    reads_clipboard: bool = False,
) -> dict[str, bool]:
    return {
        "calls_model": False,
        "calls_external_service": False,
        "executes_tools": False,
        "reads_personal_data": contact_lookup_attempted or reads_clipboard,
        "reads_private_data": reads_clipboard,
        "reads_clipboard": reads_clipboard,
        "executes_side_effect": send_attempted,
        "external_side_effect": send_attempted,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": send_attempted,
        "controls_computer": controls_computer,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _kakao_send_handoff(
    *,
    recipient: str,
    message_chars: int,
    status: str,
    reason: str = "",
    send_attempted: bool = False,
    controls_computer: bool = False,
    original_to: str = "",
    contact_lookup_attempted: bool = False,
    contact_resolution_status: str = "skipped",
    contact_match_count: int = 0,
    contact_candidates: list[str] | None = None,
    target_content_verified: bool = False,
    delivery_verified: bool = False,
    gui_automation_completed: bool = False,
    reads_clipboard: bool = False,
    kakao_recipient_bound_before_approval: bool = False,
    kakao_message_preserved_before_approval: bool = False,
    kakao_target_mode_bound_before_approval: bool = False,
    target_mode: str = KAKAO_SEARCH_EXACT_TITLE_MODE,
) -> dict[str, Any]:
    changed = ["kakao_send_attempt"] if send_attempted else []
    boundaries = _kakao_send_boundaries(
        send_attempted=send_attempted,
        controls_computer=controls_computer,
        contact_lookup_attempted=contact_lookup_attempted,
        reads_clipboard=reads_clipboard,
    )
    if status == "outcome_unknown":
        next_safe_commands = ["recent tool runs", "execution recovery"]
    elif status == "failed":
        next_safe_commands = ["setup check"]
    elif status == "requested":
        next_safe_commands = ["recent tool runs"]
    else:
        next_safe_commands = ["safe next actions"]
    next_safe_command = next_safe_commands[0]
    handoff = {
        "source": "send_kakao",
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
        "to": _display_text(recipient),
        "original_to": _display_text(original_to or recipient),
        "message_chars": message_chars,
        "target_mode": target_mode,
        "preopened_exact_chat_required": (
            target_mode == KAKAO_PREOPENED_EXACT_CHAT_MODE
        ),
        "contact_lookup_attempted": contact_lookup_attempted,
        "contact_resolution_status": contact_resolution_status,
        "contact_match_count": contact_match_count,
        "contact_candidates": [_display_text(candidate) for candidate in (contact_candidates or [])],
        "recipient_bound_before_approval": kakao_recipient_bound_before_approval,
        "message_preserved_before_approval": kakao_message_preserved_before_approval,
        "target_mode_bound_before_approval": kakao_target_mode_bound_before_approval,
        "reads_clipboard": reads_clipboard,
        "content_in_metadata": False,
        "send_attempted": send_attempted,
        "send_requested": send_attempted and status in {"requested", "outcome_unknown"},
        "gui_automation_completed": gui_automation_completed,
        "target_content_verified": target_content_verified,
        "delivery_verified": delivery_verified,
        "confirmation_required": send_attempted
        and status in {"requested", "outcome_unknown"}
        and not delivery_verified,
        "approval_required_before_execution": True,
        "manual_review_required": True,
        "next_safe_command": next_safe_command,
        "next_safe_commands": next_safe_commands,
        "next_safe_command_count": len(next_safe_commands),
        "boundaries": boundaries,
    }
    return {
        "kakao_send_handoff_ready": True,
        "kakao_send_ready_for_operator": handoff["ready_for_operator"],
        "kakao_send_state_changed": handoff["state_changed"],
        "kakao_send_changed": handoff["changed"],
        "kakao_send_content_in_handoff": handoff["content_in_handoff"],
        "kakao_send_content_in_metadata": handoff["content_in_metadata"],
        "kakao_send_next_safe_command": handoff["next_safe_command"],
        "kakao_send_next_safe_commands": handoff["next_safe_commands"],
        "kakao_send_next_safe_command_count": handoff["next_safe_command_count"],
        "kakao_send_authorizes_execution": handoff["authorizes_execution"],
        "kakao_send_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "kakao_send_approval_granted": handoff["approval_granted"],
        "kakao_send_target_content_verified": handoff["target_content_verified"],
        "kakao_send_delivery_verified": handoff["delivery_verified"],
        "kakao_send_gui_automation_completed": handoff["gui_automation_completed"],
        "kakao_send_boundaries": boundaries,
        "ready_for_operator": handoff["ready_for_operator"],
        "state_changed": handoff["state_changed"],
        "changed": handoff["changed"],
        "content_in_handoff": handoff["content_in_handoff"],
        "authorizes_execution": handoff["authorizes_execution"],
        "authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "approval_granted": handoff["approval_granted"],
        "kakao_send_handoff": handoff,
    }


def _resolve_send_recipient(raw_recipient: str) -> tuple[str | None, dict[str, Any], str]:
    """Resolve the *name string* Jarvis will type into KakaoTalk's own friend search.

    KakaoTalk has no messaging API, so a send is GUI automation: type a string into
    Kakao's search box and message the top result. That search is name-based, so
    Kakao never needs a phone/email handle the way iMessage does. Apple Contacts is
    therefore only a disambiguation aid here — never a hard gate:

    - already a handle  -> pass through unchanged,
    - exactly one match -> type that contact's canonical NAME (not a phone number),
    - multiple matches  -> ask which person is meant (the one real refusal),
    - no match          -> fall through with the RAW name, so a Kakao-only friend
                           who isn't in Apple Contacts (e.g. "Fixture Example") is still
                           reachable. The recipient name on the approval screen is
                           the verification step; nothing sends without approval.

    Returns (search_string, metadata, refusal_message). refusal_message is only
    non-empty for genuine ambiguity.
    """
    if contacts_connector.looks_like_handle(raw_recipient):
        return raw_recipient, {
            "original_to": _display_text(raw_recipient),
            "contact_lookup_attempted": False,
            "contact_resolution_status": "skipped",
            "contact_match_count": 0,
            "contact_candidates": [],
        }, ""
    try:
        matches = contacts_connector.resolve_contact(raw_recipient)
    except Exception:
        return raw_recipient, {
            "original_to": _display_text(raw_recipient),
            "contact_lookup_attempted": True,
            "contact_resolution_status": "unavailable",
            "contact_match_count": 0,
            "contact_candidates": [],
        }, ""
    candidate_names = [match.name for match in matches]
    base = {
        "original_to": _display_text(raw_recipient),
        "contact_lookup_attempted": True,
        "contact_match_count": len(matches),
        "contact_candidates": [_display_text(name) for name in candidate_names],
    }
    if len(matches) == 1:
        return matches[0].name, {**base, "contact_resolution_status": "resolved"}, ""
    if len(matches) > 1:
        names = " or ".join(_display_text(name) for name in candidate_names)
        return None, {**base, "contact_resolution_status": "ambiguous"}, (
            f"I found {len(matches)} people named \"{_display_text(raw_recipient)}\" — {names}?"
        )
    return raw_recipient, {**base, "contact_resolution_status": "unverified_name"}, ""


def _kakao_approval_refusal(
    *,
    recipient: str,
    message_chars: int,
    reason: str,
    output: str,
    resolution_metadata: dict[str, Any] | None = None,
) -> ToolResult:
    resolution = dict(resolution_metadata or {})
    contact_lookup_attempted = _metadata_bool(
        resolution.get("contact_lookup_attempted")
    )
    resolution["contact_lookup_attempted"] = contact_lookup_attempted
    return ToolResult(
        "send_kakao",
        False,
        output,
        _safe_metadata(
            controls_computer=False,
            reads_personal_data=contact_lookup_attempted,
            to=_display_text(recipient),
            message_chars=message_chars,
            reason=reason,
            failure_kind="kakao_approval_arguments_invalid",
            requires_confirmation=False,
            executed_handler=False,
            handler_invoked=False,
            **resolution,
            **_kakao_send_handoff(
                recipient=recipient,
                message_chars=message_chars,
                status="refused",
                reason=reason,
                **resolution,
            ),
        ),
    )


def resolve_kakao_send_approval(
    args: dict[str, Any],
) -> ApprovalArgumentResolution | ToolResult:
    """Bind one immutable Kakao display name before approval is queued.

    The initial tool contract prevents callers from supplying the private binding
    field. The resolver validates the unmodified message, resolves the name once,
    and returns the exact values the approved handler will later execute.
    """
    raw_recipient = args.get("to")
    raw_message = args.get("message")
    target_mode, mode_reason, mode_output = _kakao_target_mode(
        args.get(KAKAO_TARGET_MODE_KEY)
    )
    reason, output = _kakao_raw_argument_failure(raw_recipient, raw_message)
    recipient_display = raw_recipient if type(raw_recipient) is str else ""
    message_chars = len(raw_message) if type(raw_message) is str else 0
    if reason:
        return _kakao_approval_refusal(
            recipient=recipient_display,
            message_chars=message_chars,
            reason=reason,
            output=output,
        )
    if target_mode is None:
        return _kakao_approval_refusal(
            recipient=recipient_display,
            message_chars=message_chars,
            reason=mode_reason,
            output=mode_output,
        )

    assert type(raw_recipient) is str
    assert type(raw_message) is str
    if (
        target_mode == KAKAO_PREOPENED_EXACT_CHAT_MODE
        and raw_recipient != raw_recipient.strip()
    ):
        return _kakao_approval_refusal(
            recipient=raw_recipient,
            message_chars=len(raw_message),
            reason="preopened_recipient_not_exact",
            output=(
                "The preopened Kakao recipient must exactly match the displayed chat title "
                "without surrounding spaces."
            ),
        )

    recipient_query = raw_recipient.strip()
    if contacts_connector.looks_like_handle(recipient_query):
        return _kakao_approval_refusal(
            recipient=recipient_query,
            message_chars=len(raw_message),
            reason="recipient_handle_not_allowed",
            output="Use the exact KakaoTalk display name, not a phone number, email address, or @handle.",
        )

    if target_mode == KAKAO_PREOPENED_EXACT_CHAT_MODE:
        resolved_recipient = raw_recipient
        resolution = {
            "original_to": _display_text(raw_recipient),
            "contact_lookup_attempted": False,
            "contact_resolution_status": "preopened_exact_name",
            "contact_match_count": 0,
            "contact_candidates": [],
        }
        resolution_message = ""
    else:
        resolved_recipient, resolution, resolution_message = _resolve_send_recipient(
            recipient_query
        )
    resolution = {
        **resolution,
        "contact_lookup_attempted": _metadata_bool(
            resolution.get("contact_lookup_attempted")
        ),
    }
    if not resolved_recipient:
        return _kakao_approval_refusal(
            recipient=recipient_query,
            message_chars=len(raw_message),
            reason=str(
                resolution.get("contact_resolution_status")
                or "contact_resolution_failed"
            ),
            output=resolution_message,
            resolution_metadata=resolution,
        )

    if type(resolved_recipient) is not str:
        return _kakao_approval_refusal(
            recipient=recipient_query,
            message_chars=len(raw_message),
            reason="resolved_recipient_invalid",
            output="The resolved Kakao recipient is invalid, so no approval was queued.",
            resolution_metadata=resolution,
        )
    canonical_recipient = resolved_recipient.strip()
    canonical_reason, canonical_output = _kakao_raw_argument_failure(
        canonical_recipient,
        raw_message,
    )
    if canonical_reason or contacts_connector.looks_like_handle(canonical_recipient):
        return _kakao_approval_refusal(
            recipient=recipient_query,
            message_chars=len(raw_message),
            reason=canonical_reason or "resolved_recipient_handle_not_allowed",
            output=canonical_output
            or "The resolved Kakao recipient is not an exact display name, so no approval was queued.",
            resolution_metadata=resolution,
        )

    return ApprovalArgumentResolution(
        {
            "to": canonical_recipient,
            "message": raw_message,
            KAKAO_TARGET_MODE_KEY: target_mode,
            KAKAO_APPROVAL_BINDING_KEY: _kakao_approval_binding(
                canonical_recipient,
                raw_message,
                target_mode,
            ),
        },
        {
            **resolution,
            "kakao_recipient_bound_before_approval": True,
            "kakao_message_preserved_before_approval": True,
            "kakao_target_mode_bound_before_approval": True,
        },
    )


def _as_script_string(value: str) -> str:
    # ensure_ascii=False is load-bearing: with the default, Korean text becomes
    # \uXXXX escapes, which are a SYNTAX ERROR in AppleScript string literals —
    # every script with a Korean recipient would fail to compile.
    return json.dumps(value, ensure_ascii=False)


def _kakao_error() -> str:
    return (
        "KakaoTalk could not send that message right now. Check that KakaoTalk is installed, logged in, "
        "and Accessibility permission is granted for the Jarvis Python process."
    )


_GUARD_MARKER = "JARVIS_GUARD:"


class KakaoSendError(RuntimeError):
    def __init__(
        self,
        stage: str,
        detail: str = "",
        *,
        clipboard_metadata: dict[str, Any] | None = None,
        enter_pressed: bool = False,
    ) -> None:
        self.stage = _display_text(stage, 80)
        self.detail = _display_text(detail, 220)
        self.clipboard_metadata = dict(clipboard_metadata or {})
        self.enter_pressed = enter_pressed is True
        suffix = f": {self.detail}" if self.detail else ""
        super().__init__(f"KakaoTalk automation failed at {self.stage}{suffix}")


def _kakao_error_stage(stderr: str) -> tuple[str, str]:
    detail = _display_text(stderr, 220)
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
            "to control System Events and KakaoTalk, then retry.",
        )
    if "assistive access" in low or "accessibility" in low or ("not allowed" in low and "system events" in low):
        return (
            "macos_accessibility_permission",
            "In System Settings > Privacy & Security > Accessibility, enable the exact Python process "
            "running Jarvis, then retry.",
        )
    if "application isn't running" in low or "application is not running" in low or "application not found" in low:
        return "kakaotalk_unavailable", "Open KakaoTalk, confirm it is logged in, then retry."
    if "syntax error" in low or "a identifier can't go after" in low or "(-2741)" in low:
        return "applescript_compile_failed", "Jarvis could not compile the KakaoTalk automation script."
    return "osascript_failed", detail


def _kakao_failure(exc: Exception) -> tuple[str, str]:
    guard = guard_message(exc)
    if guard:
        return "guard_stopped", f"KakaoTalk send stopped safely: {guard}"
    if isinstance(exc, KakaoSendError):
        if exc.stage == "clipboard_custody_before_enter":
            return (
                exc.stage,
                "KakaoTalk stopped before Enter because clipboard custody changed; no message was sent.",
            )
        if exc.stage.startswith("recipient_phase_"):
            original_stage = exc.stage.removeprefix("recipient_phase_")
            if original_stage in {
                "macos_automation_permission",
                "macos_accessibility_permission",
                "kakaotalk_unavailable",
            }:
                return exc.stage, f"KakaoTalk send stopped at {original_stage}: {exc.detail}"
            return exc.stage, _kakao_error()
        if exc.stage == "clipboard_rich_content":
            return (
                exc.stage,
                "KakaoTalk send stopped before changing the clipboard. Copy plain text or clear "
                "the clipboard, then submit a new approved send.",
            )
        if exc.stage in {"clipboard_inspection", "clipboard_read"}:
            return (
                exc.stage,
                "KakaoTalk send stopped before acting because Jarvis could not safely snapshot "
                "the clipboard. Run `setup check`, repair clipboard access, then retry.",
            )
        if exc.stage in {"macos_automation_permission", "macos_accessibility_permission", "kakaotalk_unavailable"}:
            return exc.stage, f"KakaoTalk send stopped at {exc.stage}: {exc.detail}"
        return exc.stage, _kakao_error()
    if isinstance(exc, subprocess.TimeoutExpired):
        return "automation_timeout", "KakaoTalk automation timed out before Jarvis could verify the send; nothing was retried."
    return "send_error", _kakao_error()


def guard_message(exc: Exception) -> str:
    """Extract the safe, self-authored guard diagnostic from an AppleScript error.

    Guard aborts (wrong window focused, wrong search result) raise AppleScript
    errors prefixed with JARVIS_GUARD:. They never expose uncontrolled window or
    chat titles. A mismatched result is normalized into a recovery step using
    only the requested recipient; other messages are static self-authored text.
    Returns '' for non-guard errors.
    """
    text = str(exc)
    if _GUARD_MARKER not in text:
        return ""
    guard = text.split(_GUARD_MARKER, 1)[1]
    guard = guard.split(" (-", 1)[0]  # trim osascript's trailing error code
    guard = _display_text(guard, 220).strip()
    mismatch_prefix = "top_result_mismatch:"
    if guard.startswith(mismatch_prefix):
        recipient = _display_text(guard.removeprefix(mismatch_prefix), MAX_RECIPIENT_CHARS)
        if recipient:
            return (
                f"KakaoTalk's top search result did not match {recipient}, so no message was typed. "
                "Confirm that person's exact KakaoTalk display name, then retry."
            )
        return "KakaoTalk's top search result did not match the requested recipient, so no message was typed."
    return guard


def open_kakao_chat_fragment(recipient: str) -> str:
    """AppleScript fragment: raise Kakao's MAIN window, search `recipient`, open
    the top result, and verify the front window is really that chat.

    Two live failures drove this design:
    1. Cmd+F is context-dependent — with a chat window frontmost it opens the
       IN-CHAT message search, and everything typed afterwards lands in search
       fields. So the main list window is raised first (Cmd+2 targets the Chats
       tab of the main window; AXRaise on the main window is the fallback) and
       the flow ABORTS if the main window still isn't frontmost.
    2. Enter on a search result doesn't always open a chat. So after Down+Enter
       the front window title must (case-insensitively) match the recipient
       before anything else is typed — otherwise ABORT. This is what prevents a
       message from being typed into a search bar or sent to the wrong chat
       (e.g. a group that merely contains the friend).

    The fragment leaves the verified chat window focused with `chatTitle` set;
    callers append their own action (type message / click call button) and must
    close both `tell` blocks (and should restore `savedClip`).

    Text entry uses clipboard-paste, never `keystroke <text>`: keystroke cannot
    type Korean (Hangul has no key-event mapping on this layout), and the operator's
    Kakao contacts are frequently Korean names. The user's clipboard is saved
    into `savedClip` first so callers can restore it.
    """
    q = _as_script_string(recipient)
    return f"""
set savedClip to ""
try
    set savedClip to (the clipboard as text)
end try
tell application "KakaoTalk" to reopen
tell application "KakaoTalk" to activate
delay 0.8
tell application "System Events"
    tell process "KakaoTalk"
        set frontmost to true
        set mainTitles to {{"KakaoTalk", "카카오톡"}}
        keystroke "2" using {{command down}}
        delay 0.6
        set frontTitle to ""
        try
            set frontTitle to name of front window
        end try
        if frontTitle is missing value then set frontTitle to ""
        if mainTitles does not contain frontTitle then
            repeat with mainName in mainTitles
                try
                    perform action "AXRaise" of window mainName
                    exit repeat
                end try
            end repeat
            delay 0.4
            try
                set frontTitle to name of front window
            end try
            if frontTitle is missing value then set frontTitle to ""
        end if
        if mainTitles does not contain frontTitle then
            error "{_GUARD_MARKER} KakaoTalk's main window is not focused, so nothing was typed. Open KakaoTalk's Chats view and retry"
        end if
        -- Kakao's first search result can be an unrelated group before the exact
        -- individual. Try a small, fixed number of candidates, verifying the
        -- opened chat title after each attempt. No message text is pasted until
        -- an exact match is found.
        set chatMatched to false
        set maxSearchResults to 5
        repeat with candidateIndex from 1 to maxSearchResults
            if candidateIndex is not 1 then
                keystroke "2" using {{command down}}
                delay 0.6
                set frontTitle to ""
                try
                    set frontTitle to name of front window
                end try
                if frontTitle is missing value then set frontTitle to ""
                if mainTitles does not contain frontTitle then
                    repeat with mainName in mainTitles
                        try
                            perform action "AXRaise" of window mainName
                            exit repeat
                        end try
                    end repeat
                    delay 0.4
                    try
                        set frontTitle to name of front window
                    end try
                    if frontTitle is missing value then set frontTitle to ""
                end if
                if mainTitles does not contain frontTitle then
                    error "{_GUARD_MARKER} KakaoTalk's main window is not focused, so nothing was typed. Open KakaoTalk's Chats view and retry"
                end if
            end if
            keystroke "f" using {{command down}}
            delay 0.4
            set the clipboard to {q}
            delay 0.2
            keystroke "v" using {{command down}}
            delay 1.0
            repeat with selectionStep from 1 to candidateIndex
                key code 125
            end repeat
            delay 0.3
            key code 36
            delay 1.2
            set chatTitle to ""
            try
                set chatTitle to name of front window
            end try
            if chatTitle is missing value then set chatTitle to ""
            if chatTitle is not "" and mainTitles does not contain chatTitle then
                set titlesMatch to false
                ignoring case
                    set titlesMatch to (chatTitle is {q})
                end ignoring
                if titlesMatch then
                    set chatMatched to true
                    exit repeat
                end if
            end if
        end repeat
        if chatMatched is false then
            error "{_GUARD_MARKER} no exact KakaoTalk chat matched " & {q} & " in the first 5 search results, so no message was typed. Confirm the exact KakaoTalk display name and retry"
        end if
"""


def _kakao_send_recipient_phase_script(recipient: str) -> str:
    """Open one exact Kakao chat while the transaction owns recipient text."""
    q = _as_script_string(recipient)
    return f"""
tell application "KakaoTalk" to reopen
tell application "KakaoTalk" to activate
delay 0.8
tell application "System Events"
    tell process "KakaoTalk"
        set frontmost to true
        set mainTitles to {{"KakaoTalk", "카카오톡"}}
        set chatMatched to false
        set maxSearchResults to 5
        repeat with candidateIndex from 1 to maxSearchResults
            keystroke "2" using {{command down}}
            delay 0.6
            set frontTitle to ""
            try
                set frontTitle to name of front window
            end try
            if frontTitle is missing value then set frontTitle to ""
            if mainTitles does not contain frontTitle then
                repeat with mainName in mainTitles
                    try
                        perform action "AXRaise" of window mainName
                        exit repeat
                    end try
                end repeat
                delay 0.4
                try
                    set frontTitle to name of front window
                end try
                if frontTitle is missing value then set frontTitle to ""
            end if
            if mainTitles does not contain frontTitle then
                error "{_GUARD_MARKER} KakaoTalk's main window is not focused, so nothing was typed. Open KakaoTalk's Chats view and retry"
            end if
            keystroke "f" using {{command down}}
            delay 0.4
            set stagedRecipient to ""
            try
                set stagedRecipient to (the clipboard as text)
            on error
                error "{_GUARD_MARKER} clipboard custody was unreadable before recipient paste, so no message was typed"
            end try
            if stagedRecipient is not {q} then
                error "{_GUARD_MARKER} clipboard ownership changed before recipient paste, so no message was typed"
            end if
            keystroke "v" using {{command down}}
            delay 1.0
            repeat with selectionStep from 1 to candidateIndex
                key code 125
            end repeat
            delay 0.3
            key code 36
            delay 1.2
            set chatTitle to ""
            try
                set chatTitle to name of front window
            end try
            if chatTitle is missing value then set chatTitle to ""
            if chatTitle is not "" and mainTitles does not contain chatTitle then
                set titlesMatch to false
                ignoring case
                    set titlesMatch to (chatTitle is {q})
                end ignoring
                if titlesMatch then
                    set chatMatched to true
                    exit repeat
                end if
            end if
        end repeat
        if chatMatched is false then
            error "{_GUARD_MARKER} no exact KakaoTalk chat matched " & {q} & " in the first 5 search results, so no message was typed. Confirm the exact KakaoTalk display name and retry"
        end if
    end tell
end tell
return "CHAT_VERIFIED"
"""


def _kakao_preopened_recipient_phase_script(recipient: str) -> str:
    """Verify only the already-open Kakao chat; never navigate or search.

    The operator visually establishes the one-to-one identity before returning to
    Terminal. Execution may bring KakaoTalk to the front, but it must preserve the
    existing front window and exact title. Multiple same-title windows are
    ambiguous and stop before any clipboard replacement or message paste.
    """
    q = _as_script_string(recipient)
    return f"""
tell application "System Events"
    if not (exists process "KakaoTalk") then
        error "{_GUARD_MARKER} KakaoTalk is not already open, so no message was pasted or sent"
    end if
    tell process "KakaoTalk"
        set frontmost to true
        delay 0.4
        set chatTitle to ""
        try
            set chatTitle to name of front window
        end try
        if chatTitle is missing value then set chatTitle to ""
        set titlesMatch to false
        considering case, diacriticals
            set titlesMatch to (chatTitle is {q})
        end considering
        if titlesMatch is false then
            error "{_GUARD_MARKER} the preopened exact KakaoTalk chat is not focused, so no message was pasted or sent"
        end if
        set exactTitleWindowCount to 0
        repeat with candidateWindow in windows
            set candidateTitle to ""
            try
                set candidateTitle to name of candidateWindow
            end try
            if candidateTitle is missing value then set candidateTitle to ""
            set candidateMatches to false
            considering case, diacriticals
                set candidateMatches to (candidateTitle is {q})
            end considering
            if candidateMatches then set exactTitleWindowCount to exactTitleWindowCount + 1
        end repeat
        if exactTitleWindowCount is not 1 then
            error "{_GUARD_MARKER} the preopened KakaoTalk target is ambiguous, so no message was pasted or sent"
        end if
    end tell
end tell
return "CHAT_VERIFIED"
"""


def _kakao_send_message_phase_script(
    recipient: str,
    message: str,
    *,
    preopened_exact_chat: bool = False,
) -> str:
    """Revalidate target and staged message immediately before paste and Enter."""
    q_recipient = _as_script_string(recipient)
    q_message = _as_script_string(message)
    if preopened_exact_chat:
        activation = f"""
tell application "System Events"
    if not (exists process "KakaoTalk") then
        error "{_GUARD_MARKER} KakaoTalk is no longer open, so no message was pasted or sent"
    end if
end tell
"""
    else:
        activation = """
tell application "KakaoTalk" to activate
delay 0.2
"""
    return f"""
{activation}
tell application "System Events"
    tell process "KakaoTalk"
        set frontmost to true
        set mainTitles to {{"KakaoTalk", "카카오톡"}}
        set chatTitle to ""
        try
            set chatTitle to name of front window
        end try
        if chatTitle is missing value then set chatTitle to ""
        set titlesMatch to false
        if chatTitle is not "" and mainTitles does not contain chatTitle then
            considering case, diacriticals
                set titlesMatch to (chatTitle is {q_recipient})
            end considering
        end if
        if titlesMatch is false then
            error "{_GUARD_MARKER} the exact KakaoTalk chat is no longer focused, so no message was pasted or sent"
        end if
        set exactTitleWindowCountBeforePaste to 0
        repeat with candidateWindow in windows
            set candidateTitle to ""
            try
                set candidateTitle to name of candidateWindow
            end try
            if candidateTitle is missing value then set candidateTitle to ""
            set candidateMatches to false
            considering case, diacriticals
                set candidateMatches to (candidateTitle is {q_recipient})
            end considering
            if candidateMatches then set exactTitleWindowCountBeforePaste to exactTitleWindowCountBeforePaste + 1
        end repeat
        if exactTitleWindowCountBeforePaste is not 1 then
            error "{_GUARD_MARKER} the exact KakaoTalk target became ambiguous before message paste, so no message was sent"
        end if
        set stagedMessage to ""
        try
            set stagedMessage to (the clipboard as text)
        on error
            error "{_GUARD_MARKER} clipboard custody was unreadable before message paste, so no message was sent"
        end try
        if stagedMessage is not {q_message} then
            error "{_GUARD_MARKER} clipboard ownership changed before message paste, so no message was sent"
        end if
        keystroke "v" using {{command down}}
        delay 0.3
        set stagedMessageBeforeEnter to ""
        try
            set stagedMessageBeforeEnter to (the clipboard as text)
        on error
            error "{_GUARD_MARKER} clipboard custody was unreadable before Enter, so no message was sent"
        end try
        if stagedMessageBeforeEnter is not {q_message} then
            error "{_GUARD_MARKER} clipboard ownership changed before Enter, so no message was sent"
        end if
        set chatTitleBeforeEnter to ""
        try
            set chatTitleBeforeEnter to name of front window
        end try
        if chatTitleBeforeEnter is missing value then set chatTitleBeforeEnter to ""
        set titlesMatchBeforeEnter to false
        if chatTitleBeforeEnter is not "" and mainTitles does not contain chatTitleBeforeEnter then
            considering case, diacriticals
                set titlesMatchBeforeEnter to (chatTitleBeforeEnter is {q_recipient})
            end considering
        end if
        if titlesMatchBeforeEnter is false then
            error "{_GUARD_MARKER} the exact KakaoTalk chat changed before Enter, so no message was sent"
        end if
        set exactTitleWindowCountBeforeEnter to 0
        repeat with candidateWindow in windows
            set candidateTitle to ""
            try
                set candidateTitle to name of candidateWindow
            end try
            if candidateTitle is missing value then set candidateTitle to ""
            set candidateMatchesBeforeEnter to false
            considering case, diacriticals
                set candidateMatchesBeforeEnter to (candidateTitle is {q_recipient})
            end considering
            if candidateMatchesBeforeEnter then set exactTitleWindowCountBeforeEnter to exactTitleWindowCountBeforeEnter + 1
        end repeat
        if exactTitleWindowCountBeforeEnter is not 1 then
            error "{_GUARD_MARKER} the exact KakaoTalk target became ambiguous before Enter, so no message was sent"
        end if
        key code 36
    end tell
end tell
return "ENTER_PRESSED"
"""


def _run_with_clipboard_restored(run):
    """Run a GUI-automation callable, restoring the user's clipboard afterwards.

    The AppleScript saves/restores the clipboard itself on the HAPPY path, but a
    guard abort (`error "JARVIS_GUARD: ..."`) exits the script before its restore
    line — without this, a failed send leaves Jarvis's message text sitting in
    the user's clipboard."""
    try:
        saved = clipboard_safety.snapshot_plain_text_clipboard(subprocess.run)
    except clipboard_safety.ClipboardSafetyError as exc:
        raise KakaoSendError(
            exc.stage,
            "Jarvis stopped before changing a rich, non-text, or unreadable clipboard.",
            clipboard_metadata=clipboard_safety.privacy_boundary_metadata(
                snapshot_attempted=True,
                snapshot_captured=False,
                replacement_attempted=False,
                text_restored=None,
            ),
        ) from exc
    pending_error: Exception | None = None
    result = None
    try:
        result = run()
    except Exception as exc:
        pending_error = exc
    finally:
        try:
            clipboard_safety.restore_plain_text_clipboard(saved, subprocess.run)
        except clipboard_safety.ClipboardSafetyError as exc:
            raise KakaoSendError(
                "clipboard_restore_failed",
                "KakaoTalk did not confirm clipboard restoration after computer control.",
                clipboard_metadata=clipboard_safety.privacy_boundary_metadata(
                    snapshot_attempted=True,
                    snapshot_captured=True,
                    replacement_attempted=True,
                    text_restored=False,
                ),
            ) from exc
    if pending_error is not None:
        if isinstance(pending_error, KakaoSendError):
            pending_error.clipboard_metadata = (
                clipboard_safety.privacy_boundary_metadata(
                    snapshot_attempted=True,
                    snapshot_captured=True,
                    replacement_attempted=True,
                    text_restored=True,
                )
            )
        raise pending_error
    return result


def _run_kakao_send_phase(
    script: str,
    *,
    phase: str,
    expected_marker: str,
) -> str:
    try:
        result = subprocess.run(
            ["osascript", "-"],
            input=script,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        stage = "recipient_phase_timeout" if phase == "recipient" else "automation_timeout"
        raise KakaoSendError(stage, "KakaoTalk automation timed out.") from exc
    if result.returncode != 0:
        stderr = result.stderr.strip() or "KakaoTalk AppleScript failed"
        stage, detail = _kakao_error_stage(stderr)
        if phase == "recipient" and not guard_message(KakaoSendError(stage, detail)):
            stage = f"recipient_phase_{stage}"
        raise KakaoSendError(stage, detail)
    marker = result.stdout.strip()
    if marker != expected_marker:
        stage = (
            "recipient_phase_receipt_invalid"
            if phase == "recipient"
            else "message_phase_receipt_invalid"
        )
        raise KakaoSendError(
            stage,
            "KakaoTalk automation did not return the exact bounded phase receipt.",
        )
    return marker


def _send_kakao(recipient: str, message: str) -> dict[str, Any]:
    """Send with two guarded GUI phases and one ownership-checked transaction."""
    transaction = clipboard_safety.PlainTextClipboardTransaction(
        run=subprocess.run
    )
    recipient_phase_completed = False
    message_phase_completed = False
    enter_pressed = False
    message_phase_error: KakaoSendError | None = None
    try:
        with transaction:
            transaction.stage(recipient)
            _run_kakao_send_phase(
                _kakao_send_recipient_phase_script(recipient),
                phase="recipient",
                expected_marker="CHAT_VERIFIED",
            )
            recipient_phase_completed = True
            transaction.restage(message)
            try:
                _run_kakao_send_phase(
                    _kakao_send_message_phase_script(recipient, message),
                    phase="message",
                    expected_marker="ENTER_PRESSED",
                )
            except KakaoSendError as phase_error:
                message_phase_error = phase_error
                raise
            message_phase_completed = True
            enter_pressed = True
    except clipboard_safety.ClipboardSafetyError as exc:
        custody_metadata = transaction.privacy_metadata()
        phase_proves_no_enter = bool(
            message_phase_error is not None
            and (
                guard_message(message_phase_error)
                or _send_outcome_is_known_not_sent(message_phase_error.stage)
            )
        )
        stage = "clipboard_custody_before_enter"
        if (
            custody_metadata.get("clipboard_replacement_attempted") is not True
            and exc.stage in {
                "clipboard_inspection",
                "clipboard_read",
                "clipboard_rich_content",
            }
        ):
            stage = exc.stage
        elif enter_pressed or (message_phase_error is not None and not phase_proves_no_enter):
            stage = "clipboard_custody_after_enter"
        raise KakaoSendError(
            stage,
            "KakaoTalk could not preserve exclusive clipboard custody.",
            clipboard_metadata=custody_metadata,
            enter_pressed=enter_pressed,
        ) from exc
    except KakaoSendError as exc:
        if not exc.clipboard_metadata:
            exc.clipboard_metadata = transaction.privacy_metadata()
        raise

    return {
        **transaction.privacy_metadata(),
        "kakao_recipient_phase_completed": recipient_phase_completed,
        "kakao_message_phase_completed": message_phase_completed,
        "kakao_enter_pressed": enter_pressed,
    }


def _send_kakao_preopened_exact_chat(recipient: str, message: str) -> dict[str, Any]:
    """Send only in the operator-preopened, uniquely titled front chat window."""
    recipient_phase_completed = False
    message_phase_completed = False
    enter_pressed = False

    # This phase does not navigate, search, or replace the clipboard. A missing,
    # changed, or ambiguous preopened target is therefore a known no-send stop.
    _run_kakao_send_phase(
        _kakao_preopened_recipient_phase_script(recipient),
        phase="recipient",
        expected_marker="CHAT_VERIFIED",
    )
    recipient_phase_completed = True

    transaction = clipboard_safety.PlainTextClipboardTransaction(
        run=subprocess.run
    )
    message_phase_error: KakaoSendError | None = None
    try:
        with transaction:
            transaction.stage(message)
            try:
                _run_kakao_send_phase(
                    _kakao_send_message_phase_script(
                        recipient,
                        message,
                        preopened_exact_chat=True,
                    ),
                    phase="message",
                    expected_marker="ENTER_PRESSED",
                )
            except KakaoSendError as phase_error:
                message_phase_error = phase_error
                raise
            message_phase_completed = True
            enter_pressed = True
    except clipboard_safety.ClipboardSafetyError as exc:
        custody_metadata = transaction.privacy_metadata()
        phase_proves_no_enter = bool(
            message_phase_error is not None
            and (
                guard_message(message_phase_error)
                or _send_outcome_is_known_not_sent(message_phase_error.stage)
            )
        )
        stage = "clipboard_custody_before_enter"
        if (
            custody_metadata.get("clipboard_replacement_attempted") is not True
            and exc.stage in {
                "clipboard_inspection",
                "clipboard_read",
                "clipboard_rich_content",
            }
        ):
            stage = exc.stage
        elif enter_pressed or (message_phase_error is not None and not phase_proves_no_enter):
            stage = "clipboard_custody_after_enter"
        raise KakaoSendError(
            stage,
            "KakaoTalk could not preserve exclusive clipboard custody.",
            clipboard_metadata=custody_metadata,
            enter_pressed=enter_pressed,
        ) from exc
    except KakaoSendError as exc:
        if not exc.clipboard_metadata:
            exc.clipboard_metadata = transaction.privacy_metadata()
        raise

    return {
        **transaction.privacy_metadata(),
        "kakao_recipient_phase_completed": recipient_phase_completed,
        "kakao_message_phase_completed": message_phase_completed,
        "kakao_enter_pressed": enter_pressed,
        "kakao_preopened_navigation_performed": False,
        "kakao_preopened_search_performed": False,
    }


def make_kakao_tools(config: JarvisConfig):
    def send_kakao(args: dict[str, Any]) -> ToolResult:
        binding = args.get(KAKAO_APPROVAL_BINDING_KEY)
        bound_for_approval = binding is not None
        target_mode, mode_reason, mode_output = _kakao_target_mode(
            args.get(KAKAO_TARGET_MODE_KEY)
        )
        if bound_for_approval:
            raw_recipient = args.get("to")
            raw_message = args.get("message")
        else:
            raw_recipient = (
                args.get("to")
                if "to" in args
                else args.get("recipient", args.get("contact"))
            )
            raw_message = (
                args.get("message")
                if "message" in args
                else args.get("body", args.get("text"))
            )

        reason, output = _kakao_raw_argument_failure(raw_recipient, raw_message)
        recipient_display = raw_recipient if type(raw_recipient) is str else ""
        message_chars = len(raw_message) if type(raw_message) is str else 0
        if reason:
            return _kakao_approval_refusal(
                recipient=recipient_display,
                message_chars=message_chars,
                reason=reason,
                output=output,
            )
        if target_mode is None:
            return _kakao_approval_refusal(
                recipient=recipient_display,
                message_chars=message_chars,
                reason=mode_reason,
                output=mode_output,
            )

        assert type(raw_recipient) is str
        assert type(raw_message) is str
        if bound_for_approval:
            expected_binding = _kakao_approval_binding(
                raw_recipient,
                raw_message,
                target_mode,
            )
            if (
                type(binding) is not str
                or not hmac.compare_digest(binding, expected_binding)
            ):
                return _kakao_approval_refusal(
                    recipient=raw_recipient,
                    message_chars=len(raw_message),
                    reason="approval_binding_invalid",
                    output=(
                        "The approved Kakao recipient or message binding is invalid, so nothing ran. "
                        "Submit a fresh request for review."
                    ),
                )
            recipient = raw_recipient
            message = raw_message
            resolved_recipient = recipient
            resolution_metadata = {
                "original_to": _display_text(recipient),
                "contact_lookup_attempted": False,
                "contact_resolution_status": "bound_before_approval",
                "contact_match_count": 0,
                "contact_candidates": [],
                "kakao_recipient_bound_before_approval": True,
                "kakao_message_preserved_before_approval": True,
                "kakao_target_mode_bound_before_approval": True,
            }
            resolution_message = ""
        else:
            # Direct-handler compatibility remains deterministic, but real runtime
            # execution always uses the resolver-bound branch above.
            recipient = (
                raw_recipient
                if target_mode == KAKAO_PREOPENED_EXACT_CHAT_MODE
                else raw_recipient.strip()
            )
            message = raw_message.strip()
            if target_mode == KAKAO_PREOPENED_EXACT_CHAT_MODE:
                if recipient != recipient.strip():
                    return _kakao_approval_refusal(
                        recipient=recipient,
                        message_chars=len(message),
                        reason="preopened_recipient_not_exact",
                        output=(
                            "The preopened Kakao recipient must exactly match the displayed chat "
                            "title without surrounding spaces."
                        ),
                    )
                resolved_recipient = recipient
                resolution_metadata = {
                    "original_to": _display_text(recipient),
                    "contact_lookup_attempted": False,
                    "contact_resolution_status": "preopened_exact_name",
                    "contact_match_count": 0,
                    "contact_candidates": [],
                }
                resolution_message = ""
            else:
                resolved_recipient, resolution_metadata, resolution_message = _resolve_send_recipient(recipient)
        resolution_metadata = {
            **resolution_metadata,
            "contact_lookup_attempted": _metadata_bool(resolution_metadata.get("contact_lookup_attempted")),
        }
        if not resolved_recipient:
            return ToolResult(
                "send_kakao",
                False,
                resolution_message,
                _safe_metadata(
                    controls_computer=False,
                    reads_personal_data=_metadata_bool(resolution_metadata.get("contact_lookup_attempted")),
                    to=_display_text(recipient),
                    message_chars=len(message),
                    reason=resolution_metadata.get("contact_resolution_status"),
                    **resolution_metadata,
                    **_kakao_send_handoff(
                        recipient=recipient,
                        message_chars=len(message),
                        status="refused",
                        reason=str(resolution_metadata.get("contact_resolution_status") or "contact_resolution_failed"),
                        **resolution_metadata,
                    ),
                ),
            )
        try:
            if target_mode == KAKAO_PREOPENED_EXACT_CHAT_MODE:
                send_custody = _send_kakao_preopened_exact_chat(
                    resolved_recipient,
                    message,
                )
            else:
                send_custody = _send_kakao(resolved_recipient, message)
            if not isinstance(send_custody, dict):
                send_custody = clipboard_safety.privacy_boundary_metadata(
                    snapshot_attempted=True,
                    snapshot_captured=True,
                    replacement_attempted=True,
                    text_restored=True,
                )
            requested_note = (
                f"KakaoTalk send was requested for {_display_text(resolved_recipient)}. "
                "GUI automation reached the send-key step, but Jarvis could not verify a "
                "one-to-one target or delivery. Treat the outcome as unknown, inspect the "
                "already-open conversation, and do not retry automatically."
            )
            return ToolResult(
                "send_kakao",
                True,
                requested_note,
                _safe_metadata(
                    reads_personal_data=True,
                    reads_private_data=True,
                    reads_clipboard=True,
                    executes_side_effect=True,
                    external_side_effect=True,
                    requires_approval=True,
                    to=_display_text(recipient),
                    resolved_to=_display_text(resolved_recipient),
                    message_chars=len(message),
                    target_mode=target_mode,
                    preopened_exact_chat_required=(
                        target_mode == KAKAO_PREOPENED_EXACT_CHAT_MODE
                    ),
                    preopened_navigation_performed=(
                        False
                        if target_mode == KAKAO_PREOPENED_EXACT_CHAT_MODE
                        else None
                    ),
                    preopened_search_performed=(
                        False
                        if target_mode == KAKAO_PREOPENED_EXACT_CHAT_MODE
                        else None
                    ),
                    send_attempted=True,
                    send_requested=True,
                    recipient_window_title_verified=True,
                    one_to_one_chat_verified=False,
                    send_key_requested=True,
                    gui_automation_completed=True,
                    target_content_verified=False,
                    delivery_verified=False,
                    confirmation_required=True,
                    operator_confirmation_required=True,
                    execution_outcome_unknown=True,
                    outcome_known=False,
                    side_effect_possible=True,
                    retry_safe=False,
                    automatic_retry_allowed=False,
                    authorizes_retry=False,
                    **send_custody,
                    **resolution_metadata,
                    **_kakao_send_handoff(
                        recipient=recipient,
                        message_chars=len(message),
                        status="outcome_unknown",
                        reason="delivery_and_one_to_one_target_unverified",
                        send_attempted=True,
                        controls_computer=True,
                        gui_automation_completed=True,
                        target_content_verified=False,
                        delivery_verified=False,
                        reads_clipboard=True,
                        target_mode=target_mode,
                        **resolution_metadata,
                    ),
                ),
            )
        except Exception as e:
            failure_stage, classified_output = _kakao_failure(e)
            outcome_unknown = not _send_outcome_is_known_not_sent(failure_stage)
            clipboard_metadata = dict(
                getattr(e, "clipboard_metadata", {}) or {}
            )
            failure_output = (
                _post_attempt_send_recovery_guidance()
                if outcome_unknown
                else f"{classified_output} {KNOWN_NOT_SENT_SEND_RECOVERY_ACTION}"
            )
            failure_metadata = _safe_metadata(
                guard_stopped=failure_stage == "guard_stopped",
                failure_stage=failure_stage,
                kakao_send_stage=failure_stage,
                executes_side_effect=True,
                external_side_effect=True,
                requires_approval=True,
                to=_display_text(recipient),
                resolved_to=_display_text(resolved_recipient),
                reads_personal_data=True,
                reads_private_data=True,
                reads_clipboard=True,
                message_chars=len(message),
                target_mode=target_mode,
                preopened_exact_chat_required=(
                    target_mode == KAKAO_PREOPENED_EXACT_CHAT_MODE
                ),
                send_attempted=True,
                exception_type=type(e).__name__,
                outcome_known=not outcome_unknown,
                outcome_unknown=outcome_unknown,
                side_effect_possible=outcome_unknown,
                retry_safe=not outcome_unknown,
                automatic_retry_allowed=False,
                authorizes_retry=False,
                **clipboard_metadata,
                **resolution_metadata,
                **_kakao_send_handoff(
                    recipient=recipient,
                    message_chars=len(message),
                    status="outcome_unknown" if outcome_unknown else "failed",
                    reason="send_outcome_unknown" if outcome_unknown else "send_error",
                    send_attempted=True,
                    controls_computer=True,
                    reads_clipboard=True,
                    target_mode=target_mode,
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
                "send_kakao",
                False,
                failure_output,
                failure_metadata,
            )

    from jarvis_v2.tools.registry import Tool, _tool_argument_contract

    return [
        Tool(
            "send_kakao",
            "Send a KakaoTalk message through the Mac app GUI. Args: to, message. Requires Accessibility permission.",
            RiskLevel.HIGH_RISK,
            send_kakao,
            "personal",
            argument_contract=_tool_argument_contract(
                required_strings=("to", "message"),
                optional_strings=(KAKAO_TARGET_MODE_KEY,),
            ),
            approval_argument_resolver=resolve_kakao_send_approval,
            approval_argument_contract=_tool_argument_contract(
                required_strings=(
                    "to",
                    "message",
                    KAKAO_TARGET_MODE_KEY,
                    KAKAO_APPROVAL_BINDING_KEY,
                ),
            ),
        )
    ]
