"""Best-effort Instagram DM sender for Jarvis V2.

Instagram does not offer a stable personal-use DM API. This connector drives a
logged-in Chrome session through macOS GUI automation, which is fragile and may
conflict with Instagram's automation rules. Keep it approval-gated and use it
only for explicit, reviewed sends.
"""

from __future__ import annotations

import json
import hashlib
import hmac
import re
import subprocess
import time
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
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
INSTAGRAM_FOLLOWING_LABELS = ("Following", "팔로잉")
INSTAGRAM_FOLLOW_LABELS = ("Follow", "팔로우")
INSTAGRAM_MESSAGES_SECTION_LABELS = ("Messages", "메시지")
INSTAGRAM_MORE_ACCOUNTS_SECTION_LABELS = ("More accounts", "계정 더 보기")
INSTAGRAM_USERNAME_RE = re.compile(r"^@[A-Za-z0-9._]{1,30}$")
INSTAGRAM_APPROVAL_BINDING_KEY = "_instagram_send_binding"
_INSTAGRAM_APPROVAL_BINDING_DOMAIN = b"jarvis:instagram-send-approval:v1\x00"
INSTAGRAM_COMPOSER_SELECTOR = (
    '[data-lexical-editor="true"][contenteditable="true"], '
    '[role="textbox"][contenteditable="true"]'
)


def _post_attempt_send_recovery_guidance() -> str:
    return (
        "The Instagram delivery outcome is unknown. Check the exact conversation; do not resend "
        "automatically. If the message is absent, run `setup check`, repair login and macOS "
        "Automation/Accessibility permissions, then submit a new approved send."
    )


def _send_outcome_is_known_not_sent(exc: Exception, stage: str) -> bool:
    return (
        isinstance(exc, InstagramWebError)
        and not exc.send_attempted
        and stage != "instagram_send_unconfirmed"
    )


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
        "controls_computer": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "automation_warning": "Instagram DM automation is best-effort GUI control and may violate platform automation limits.",
    }
    base.update(extra)
    return base


def _short(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text[:limit] if len(text) > limit else text


def _display_text(value: Any, limit: int = MAX_RECIPIENT_CHARS) -> str:
    text = " ".join(str(value or "").strip().split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    return text[:limit] if len(text) > limit else text


def _metadata_bool(value: Any) -> bool:
    return value is True


def _instagram_approval_binding(
    recipient: str,
    message: str,
    allow_new_recipient: bool,
) -> str:
    payload = json.dumps(
        {
            "allow_new_recipient": allow_new_recipient,
            "message": message,
            "to": recipient,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(_INSTAGRAM_APPROVAL_BINDING_DOMAIN + payload).hexdigest()


def _instagram_dm_boundaries(
    *, send_attempted: bool, controls_computer: bool, contact_lookup_attempted: bool = False
) -> dict[str, bool]:
    return {
        "calls_model": False,
        "calls_external_service": True,
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
        "controls_computer": controls_computer,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _instagram_dm_handoff(
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
    allow_new_recipient: bool = False,
    instagram_recipient_bound_before_approval: bool = False,
    instagram_message_bound_before_approval: bool = False,
    instagram_new_recipient_mode_bound_before_approval: bool = False,
    instagram_identity_kind: str = "",
) -> dict[str, Any]:
    changed = ["instagram_dm_send_attempt"] if send_attempted else []
    boundaries = _instagram_dm_boundaries(
        send_attempted=send_attempted,
        controls_computer=controls_computer,
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
        "source": "send_instagram_dm",
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
        "contact_lookup_attempted": contact_lookup_attempted,
        "contact_resolution_status": contact_resolution_status,
        "contact_match_count": contact_match_count,
        "contact_candidates": [_display_text(candidate) for candidate in (contact_candidates or [])],
        "allow_new_recipient": allow_new_recipient,
        "recipient_policy": "explicit_new_allowed" if allow_new_recipient else "existing_or_followed",
        "recipient_bound_before_approval": instagram_recipient_bound_before_approval,
        "message_bound_before_approval": instagram_message_bound_before_approval,
        "new_recipient_mode_bound_before_approval": instagram_new_recipient_mode_bound_before_approval,
        "identity_kind": instagram_identity_kind,
        "content_in_metadata": False,
        "send_attempted": send_attempted,
        "sender_side_send_confirmed": status == "sender_confirmed",
        "recipient_delivery_confirmed": False,
        "delivery_confirmed": False,
        "delivery_evidence": "sender_side_only" if status == "sender_confirmed" else "none",
        "recipient_confirmation_required": status == "sender_confirmed",
        "approval_required_before_execution": True,
        "manual_review_required": True,
        "automation_warning": "Instagram DM automation is best-effort GUI control and may violate platform automation limits.",
        "next_safe_command": next_safe_command,
        "next_safe_commands": next_safe_commands,
        "next_safe_command_count": len(next_safe_commands),
        "boundaries": boundaries,
    }
    return {
        "instagram_dm_handoff_ready": True,
        "instagram_dm_ready_for_operator": handoff["ready_for_operator"],
        "instagram_dm_state_changed": handoff["state_changed"],
        "instagram_dm_changed": handoff["changed"],
        "instagram_dm_content_in_handoff": handoff["content_in_handoff"],
        "instagram_dm_content_in_metadata": handoff["content_in_metadata"],
        "instagram_dm_next_safe_command": handoff["next_safe_command"],
        "instagram_dm_next_safe_commands": handoff["next_safe_commands"],
        "instagram_dm_next_safe_command_count": handoff["next_safe_command_count"],
        "instagram_dm_authorizes_execution": handoff["authorizes_execution"],
        "instagram_dm_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "instagram_dm_approval_granted": handoff["approval_granted"],
        "instagram_dm_boundaries": boundaries,
        "ready_for_operator": handoff["ready_for_operator"],
        "state_changed": handoff["state_changed"],
        "changed": handoff["changed"],
        "content_in_handoff": handoff["content_in_handoff"],
        "authorizes_execution": handoff["authorizes_execution"],
        "authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "approval_granted": handoff["approval_granted"],
        "instagram_dm_handoff": handoff,
    }


def _resolve_send_recipient(raw_recipient: str) -> tuple[str | None, dict[str, Any], str]:
    """Resolve a unique saved contact to the exact name Instagram should match.

    Instagram has no personal messaging API, and its browser UI only exposes
    displayed thread names. Contacts is therefore a disambiguation aid: a unique
    ``Fixture`` can become ``Fixture Example`` before the exact-thread guard runs;
    Instagram-only contacts still fall through as typed; ambiguous people stop
    before browser automation begins.
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
    names = [match.name for match in matches]
    base = {
        "original_to": _display_text(raw_recipient),
        "contact_lookup_attempted": True,
        "contact_match_count": len(matches),
        "contact_candidates": [_display_text(name) for name in names],
    }
    if len(matches) == 1:
        return matches[0].name, {**base, "contact_resolution_status": "resolved"}, ""
    if len(matches) > 1:
        choices = " or ".join(_display_text(name) for name in names)
        return None, {**base, "contact_resolution_status": "ambiguous"}, (
            f'I found {len(matches)} people named "{_display_text(raw_recipient)}" — {choices}?'
        )
    return raw_recipient, {**base, "contact_resolution_status": "unverified_name"}, ""


def _instagram_approval_refusal(
    *,
    recipient: str,
    message_chars: int,
    reason: str,
    output: str,
    allow_new_recipient: bool = False,
) -> ToolResult:
    return ToolResult(
        "send_instagram_dm",
        False,
        output,
        _safe_metadata(
            controls_computer=False,
            to=_display_text(recipient),
            message_chars=message_chars,
            reason=reason,
            failure_kind="instagram_approval_arguments_invalid",
            requires_confirmation=False,
            executed_handler=False,
            handler_invoked=False,
            **_instagram_dm_handoff(
                recipient=recipient,
                message_chars=message_chars,
                status="refused",
                reason=reason,
                allow_new_recipient=allow_new_recipient,
            ),
        ),
    )


def resolve_instagram_send_approval(
    args: dict[str, Any],
) -> ApprovalArgumentResolution | ToolResult:
    """Bind one exact Instagram username and message before approval review.

    Display names and Apple Contacts aliases are intentionally refused here:
    neither is a stable Instagram account identity.  The approved execution
    receives the exact ``@username`` the operator reviewed plus a binding over
    the message and new-recipient mode, so later contact changes cannot retarget
    the one-shot send.
    """
    raw_recipient = args.get("to")
    raw_message = args.get("message")
    raw_allow_new = args.get("allow_new_recipient", False)
    recipient = raw_recipient.strip() if type(raw_recipient) is str else ""
    message = raw_message.strip() if type(raw_message) is str else ""
    allow_new_recipient = raw_allow_new is True

    if not recipient:
        return _instagram_approval_refusal(
            recipient="",
            message_chars=len(message),
            reason="missing_recipient",
            output="Instagram recipient ('to') is required.",
            allow_new_recipient=allow_new_recipient,
        )
    if len(raw_recipient) > MAX_RECIPIENT_CHARS:
        return _instagram_approval_refusal(
            recipient=recipient,
            message_chars=len(message),
            reason="recipient_too_long",
            output=f"Instagram recipient must be at most {MAX_RECIPIENT_CHARS} characters.",
            allow_new_recipient=allow_new_recipient,
        )
    if not _is_exact_instagram_handle(recipient):
        return _instagram_approval_refusal(
            recipient=recipient,
            message_chars=len(message),
            reason="exact_username_required_before_approval",
            output=(
                "Use the exact Instagram @username before approval. Display names and Contacts "
                "aliases are not stable account identities, so no approval was queued."
            ),
            allow_new_recipient=allow_new_recipient,
        )
    if not message:
        return _instagram_approval_refusal(
            recipient=recipient,
            message_chars=0,
            reason="missing_message",
            output="Instagram DM message text is required.",
            allow_new_recipient=allow_new_recipient,
        )
    if len(raw_message) > MAX_MESSAGE_CHARS:
        return _instagram_approval_refusal(
            recipient=recipient,
            message_chars=len(raw_message),
            reason="message_too_long",
            output=f"Instagram DM message must be at most {MAX_MESSAGE_CHARS} characters.",
            allow_new_recipient=allow_new_recipient,
        )
    if type(raw_allow_new) is not bool:
        return _instagram_approval_refusal(
            recipient=recipient,
            message_chars=len(message),
            reason="allow_new_recipient_invalid",
            output="Instagram new-recipient mode must be true or false.",
        )

    return ApprovalArgumentResolution(
        {
            "to": recipient,
            "message": message,
            "allow_new_recipient": allow_new_recipient,
            INSTAGRAM_APPROVAL_BINDING_KEY: _instagram_approval_binding(
                recipient,
                message,
                allow_new_recipient,
            ),
        },
        {
            "instagram_recipient_bound_before_approval": True,
            "instagram_message_bound_before_approval": True,
            "instagram_new_recipient_mode_bound_before_approval": True,
            "instagram_identity_kind": "exact_username",
        },
    )


def _as_script_string(value: str) -> str:
    # ensure_ascii=False is load-bearing: \uXXXX escapes are an AppleScript
    # SYNTAX ERROR, so Korean text must pass through raw.
    return json.dumps(value, ensure_ascii=False)


def _instagram_error() -> str:
    return (
        "Instagram DM could not send right now. Check that Chrome is logged into Instagram and "
        "Accessibility permission is granted for the Jarvis Python process."
    )


class InstagramWebError(RuntimeError):
    def __init__(
        self,
        stage: str,
        detail: str = "",
        *,
        clipboard_metadata: dict[str, Any] | None = None,
        send_attempted: bool = False,
    ) -> None:
        self.stage = _display_text(stage, 80)
        self.detail = _display_text(detail, 220)
        self.clipboard_metadata = dict(clipboard_metadata or {})
        self.send_attempted = send_attempted is True
        suffix = f": {self.detail}" if self.detail else ""
        super().__init__(f"Instagram Web failed at {self.stage}{suffix}")


def _instagram_error_stage(stderr: str) -> tuple[str, str]:
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
            "to control Google Chrome and System Events, then retry.",
        )
    if "assistive access" in low or "accessibility" in low:
        return (
            "macos_accessibility_permission",
            "In System Settings > Privacy & Security > Accessibility, allow the process running Jarvis, then retry.",
        )
    if "allow javascript from apple events" in low or "javascript through applescript" in low:
        return "chrome_javascript_permission", "In Chrome, enable View > Developer > Allow JavaScript from Apple Events, then retry."
    return "osascript_failed", detail


def _instagram_recovery_guidance(stage: str) -> str:
    return {
        "macos_automation_permission": (
            "In System Settings > Privacy & Security > Automation, allow the Jarvis Python process "
            "to control Google Chrome and System Events, then retry."
        ),
        "macos_accessibility_permission": (
            "In System Settings > Privacy & Security > Accessibility, allow the process running Jarvis, then retry."
        ),
        "chrome_javascript_permission": (
            "In Chrome, enable View > Developer > Allow JavaScript from Apple Events, then retry."
        ),
        "instagram_not_logged_in": "Log into Instagram in Chrome, then retry.",
        "instagram_not_loaded": "Refresh Instagram's DM inbox in Chrome, wait for it to load, then retry.",
        "instagram_thread_not_found": "Open the exact conversation in Chrome or use the exact displayed name, then retry.",
        "instagram_recipient_not_known": (
            "Default Instagram sends are limited to existing conversations or accounts Instagram confirms you follow. "
            "To contact a new account, explicitly say "
            "`send a new Instagram DM to @username saying ...`."
        ),
        "instagram_recipient_not_followed": (
            "Instagram did not confirm that you follow that exact account, so Jarvis stopped before selecting it. "
            "If this is intentionally a new person, explicitly say `send a new Instagram DM to @username saying ...`."
        ),
        "instagram_exact_handle_required": (
            "More than one Instagram account can use the same display name. For a followed account without an existing "
            "one-to-one thread, use its exact `@username`; Jarvis will verify the profile says Following before selecting it."
        ),
        "instagram_follow_state_unavailable": (
            "Instagram did not expose a verifiable Following state for that exact profile, so Jarvis stopped before selecting it."
        ),
        "instagram_new_message_unavailable": "Instagram's New message screen was not ready. Refresh the inbox, then retry.",
        "instagram_new_recipient_not_found": "Instagram could not find one exact new account, so nothing was typed or sent.",
        "instagram_thread_not_opened": "Refresh the inbox and open the intended conversation once, then retry.",
        "instagram_compose_failed": "Instagram's message composer was not ready. Refresh the conversation, then retry.",
        "instagram_clipboard_rich_content": (
            "Jarvis stopped before changing the clipboard. Copy plain text or clear the clipboard, "
            "then submit a new approved Instagram send."
        ),
        "instagram_clipboard_inspection": (
            "Jarvis could not verify that the clipboard was safe to replace. Run `setup check`, "
            "repair clipboard access, then retry."
        ),
        "instagram_clipboard_read": (
            "Jarvis could not snapshot the plain-text clipboard, so nothing was pasted or sent."
        ),
        "instagram_send_button_missing": "Instagram did not show a Send button. Refresh the conversation, then retry.",
        "instagram_send_unconfirmed": (
            "Instagram did not confirm the send. Check the conversation in Chrome before deciding whether to send again."
        ),
        "instagram_confirmation_unavailable": (
            "Instagram could not establish a safe before-send confirmation baseline, so nothing was typed or sent."
        ),
        "instagram_active_call": (
            "An active Instagram call is visible, so Jarvis stopped before focusing Chrome, navigating, searching, typing, or sending."
        ),
        "instagram_call_state_unavailable": (
            "Jarvis could not prove that Instagram was free of an active call, so it stopped before changing the page."
        ),
        "instagram_call_button_missing": "Instagram did not show the requested call button. Check the conversation, then retry.",
    }.get(stage, "")


def _instagram_dm_failure(exc: Exception) -> tuple[str, str]:
    """Return a recovery-safe failure surface without exposing page or thread data."""
    if isinstance(exc, InstagramWebError):
        guidance = _instagram_recovery_guidance(exc.stage)
        if guidance:
            return exc.stage, f"Instagram DM stopped at {exc.stage}: {guidance}"
        return exc.stage, _instagram_error()
    if isinstance(exc, subprocess.TimeoutExpired):
        return "automation_timeout", "Instagram DM automation timed out before Jarvis could confirm the send; nothing was retried."
    return "send_error", _instagram_error()


# Instagram Web flow, inspected live in Chrome on 2026-07-02:
# - The DM inbox renders threads as role="button" divs (no anchors); each thread
#   shows the person's name as an exact <span>. Unlike Telegram, Instagram's
#   React handlers ACCEPT synthetic pointer/mouse events, so the whole flow can
#   run through Chrome's JavaScript bridge — Unicode-safe (no keystroke typing,
#   which cannot produce Hangul) and verifiable at every step.
# - The composer is a Lexical contenteditable ([aria-placeholder="Message..."]);
#   execCommand('insertText') works once it is clicked+focused. A "Send" button
#   appears when text is staged.

# The wrapper finds the Instagram tab BY URL on every call and executes the JS
# on that tab directly — never "active tab of front window", which silently runs
# against whatever tab happens to be frontmost (live Telegram failure mode).
_CHROME_JS_WRAPPER = """on run argv
    tell application "Google Chrome"
        repeat with w in windows
            repeat with t in tabs of w
                if URL of t contains "instagram.com" then
                    return execute t javascript (item 1 of argv)
                end if
            end repeat
        end repeat
    end tell
    error "no Instagram tab is open in Chrome"
end run"""

_ACTIVE_INSTAGRAM_CALL_JS = """(function() {
    var labels = [
        'end call', 'end video call', 'leave call', 'hang up',
        '통화 종료', '영상 통화 종료', '통화 나가기', '전화 끊기'
    ];
    var nodes = Array.prototype.slice.call(
        document.querySelectorAll('button,[role="button"],[aria-label],[title]')
    );
    var active = nodes.some(function(x) {
        var r = x.getBoundingClientRect();
        if (r.width <= 0 || r.height <= 0) return false;
        var label = (
            x.getAttribute('aria-label') ||
            x.getAttribute('title') ||
            x.innerText ||
            x.textContent ||
            ''
        ).trim().toLowerCase();
        return labels.some(function(expected) {
            return label === expected || label.indexOf(expected) >= 0;
        });
    });
    return active ? 'ACTIVE' : 'CLEAR';
})()"""

_CHROME_INSTAGRAM_CALL_GUARD_WRAPPER = """on run argv
    tell application "Google Chrome"
        repeat with w in windows
            repeat with t in tabs of w
                if URL of t contains "instagram.com" then
                    set callState to execute t javascript (item 1 of argv)
                    if callState is "ACTIVE" then return "ACTIVE"
                end if
            end repeat
        end repeat
    end tell
    return "CLEAR"
end run"""


def _chrome_js(js: str, timeout: int = 20) -> str:
    """Run JavaScript in Chrome's Instagram tab (located by URL each call). The
    JS travels via argv, so no AppleScript escaping and raw Korean passes through."""
    result = subprocess.run(["osascript", "-e", _CHROME_JS_WRAPPER, js], capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "Chrome JavaScript bridge failed"
        stage, detail = _instagram_error_stage(stderr)
        raise InstagramWebError(stage, detail)
    return result.stdout.strip()


def _guard_no_active_instagram_call() -> None:
    """Fail closed before focus/navigation if any Instagram tab shows call controls."""
    result = subprocess.run(
        ["osascript", "-e", _CHROME_INSTAGRAM_CALL_GUARD_WRAPPER, _ACTIVE_INSTAGRAM_CALL_JS],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or "could not inspect Instagram call state"
        stage, detail = _instagram_error_stage(stderr)
        raise InstagramWebError(stage, detail)
    state = result.stdout.strip()
    if state == "ACTIVE":
        raise InstagramWebError("instagram_active_call")
    if state != "CLEAR":
        raise InstagramWebError("instagram_call_state_unavailable")


def _focus_instagram_tab() -> None:
    """Bring an Instagram tab to the front without discarding an open DM."""
    script = """
tell application "Google Chrome"
    activate
    set found to false
    repeat with w in windows
        set i to 1
        repeat with t in tabs of w
            if URL of t contains "instagram.com" then
                set active tab index of w to i
                set index of w to 1
                set found to true
                exit repeat
            end if
            set i to i + 1
        end repeat
        if found then exit repeat
    end repeat
    if not found then
        if (count of windows) is 0 then make new window
        tell front window to make new tab with properties {URL:"https://www.instagram.com/direct/inbox/"}
    end if
end tell
"""
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=20)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "could not focus an Instagram tab in Chrome"
        stage, detail = _instagram_error_stage(stderr)
        raise InstagramWebError(stage, detail)


def _navigate_instagram_inbox() -> None:
    """Load the DM inbox only after no verified open conversation was found."""
    _chrome_js(
        "(function(){location.href='https://www.instagram.com/direct/inbox/';"
        "return 'NAVIGATING';})()"
    )


def _open_thread_matches(recipient: str) -> bool:
    """Return true only when the current DM URL has an exact visible header match."""
    recipient_js = json.dumps(recipient, ensure_ascii=False)
    state = _chrome_js(
        "(function() {"
        " if (location.pathname.indexOf('/direct/t/') !== 0) return 'NOT_THREAD';"
        f" var raw = {recipient_js}.trim().toLowerCase();"
        " var q = raw.charAt(0)==='@' ? raw.slice(1) : raw;"
        " var main = document.querySelector('[role=\"main\"]') || document.querySelector('main') || document.body;"
        " var header = main.querySelector('header') || document.querySelector('header');"
        " function visible(x){var r=x.getBoundingClientRect();return r.width>0&&r.height>0;}"
        " var headings=Array.prototype.slice.call(main.querySelectorAll('h1,h2,h3,[role=\"heading\"]')).filter(function(h){"
        "  return visible(h)&&!h.closest('nav,[role=\"navigation\"]');});"
        " if(raw.charAt(0)==='@'){"
        "  var expected='/'+q+'/';"
        "  var headerLinked=!!header&&Array.prototype.some.call(header.querySelectorAll('a[href]'),function(a){"
        "   try{return new URL(a.href,location.origin).pathname.toLowerCase()===expected;}catch(_e){return false;}});"
        "  var headingLinked=headings.some(function(h){var a=h.closest('a[href]');"
        "   if(!a)return false;try{return new URL(a.href,location.origin).pathname.toLowerCase()===expected;}catch(_e){return false;}});"
        "  return headerLinked||headingLinked ? 'MATCH' : 'NOT_MATCH';"
        " }"
        " var headerMatched=!!header&&Array.prototype.some.call(header.querySelectorAll('span,h1,h2,h3,[role=\"heading\"]'),function(x){"
        "  return visible(x)&&(x.textContent||'').trim().toLowerCase()===q;});"
        " var headingMatched=headings.some(function(h){return (h.textContent||'').trim().toLowerCase()===q;});"
        " var matched=headerMatched||headingMatched;"
        " return matched ? 'MATCH' : 'NOT_MATCH'; })()"
    )
    return state == "MATCH"


def _start_instagram_thread_search(recipient: str) -> str:
    """Filter the inbox with Instagram's own search input, without selecting."""
    recipient_js = json.dumps(recipient, ensure_ascii=False)
    activated = _chrome_js(
        "(function() {"
        " var input = Array.prototype.find.call(document.querySelectorAll('input'), function(x) {"
        "  var p=(x.getAttribute('placeholder')||'').trim().toLowerCase();"
        "  return p.indexOf('search')===0 || p.indexOf('검색')===0; });"
        " if (!input) return 'NO_SEARCH_INPUT';"
        " var el=input;"
        f" {_CLICK_SEQUENCE_JS}"
        " input.focus();"
        " return 'SEARCH_ACTIVATED'; })()"
    )
    if activated != "SEARCH_ACTIVATED":
        return activated
    # Instagram replaces the input when its search overlay opens. Reacquire it
    # after React commits that transition; changing the detached input appears
    # successful but never produces visible search results.
    time.sleep(0.1)
    return _chrome_js(
        "(function() {"
        " var input = Array.prototype.find.call(document.querySelectorAll('input'), function(x) {"
        "  var p=(x.getAttribute('placeholder')||'').trim().toLowerCase();"
        "  return p.indexOf('search')===0 || p.indexOf('검색')===0; });"
        " if (!input) return 'NO_SEARCH_INPUT';"
        " input.focus();"
        " var setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;"
        f" setter.call(input, {recipient_js});"
        " input.dispatchEvent(new InputEvent('input', {bubbles:true, inputType:'insertText', data:" + recipient_js + "}));"
        " input.dispatchEvent(new Event('change', {bubbles:true}));"
        " return input.value === " + recipient_js + " ? 'SEARCHING' : 'SEARCH_FAILED'; })()"
    )


def _instagram_thread_finder_js(
    recipient: str,
    *,
    require_following: bool = False,
    existing_only: bool = False,
) -> str:
    """Build a fail-closed existing/new recipient selector.

    Only one exact visible name may be selected. Prefix expansion is forbidden:
    a label such as ``Fixture Example and Fixture Alternate`` must never satisfy ``Fixture``.
    The selected displayed name is returned for downstream verification.
    """
    recipient_js = json.dumps(recipient, ensure_ascii=False)
    following_labels_js = json.dumps([label.casefold() for label in INSTAGRAM_FOLLOWING_LABELS], ensure_ascii=False)
    messages_labels_js = json.dumps(
        [label.casefold() for label in INSTAGRAM_MESSAGES_SECTION_LABELS],
        ensure_ascii=False,
    )
    more_accounts_labels_js = json.dumps(
        [label.casefold() for label in INSTAGRAM_MORE_ACCOUNTS_SECTION_LABELS],
        ensure_ascii=False,
    )
    require_following_js = "true" if require_following else "false"
    existing_only_js = "true" if existing_only else "false"
    return (
        "(function() {"
        f" var raw = {recipient_js}.trim().toLowerCase();"
        f" var requireFollowing = {require_following_js};"
        f" var existingOnly = {existing_only_js};"
        f" var followingLabels = {following_labels_js};"
        f" var messagesLabels = {messages_labels_js};"
        f" var moreAccountsLabels = {more_accounts_labels_js};"
        " var q = raw.charAt(0) === '@' ? raw.slice(1) : raw;"
        " var scope = document.body;"
        " function visible(x) { var b=x.getBoundingClientRect(); return b.width>0 && b.height>0; }"
        " var sectionNodes = Array.prototype.slice.call(scope.querySelectorAll('span,div,h1,h2,h3,h4,h5,h6')).filter(function(x) {"
        "  var t=(x.textContent||'').trim().toLowerCase();"
        "  return visible(x) && (messagesLabels.indexOf(t)>=0 || moreAccountsLabels.indexOf(t)>=0); });"
        " var messageHeadings=sectionNodes.filter(function(x){"
        "  return messagesLabels.indexOf((x.textContent||'').trim().toLowerCase())>=0;});"
        " var accountHeadings=sectionNodes.filter(function(x){"
        "  return moreAccountsLabels.indexOf((x.textContent||'').trim().toLowerCase())>=0;});"
        " function inMessagesSection(s) {"
        "  var top=s.getBoundingClientRect().top;"
        "  var prior=messageHeadings.filter(function(h){return h.getBoundingClientRect().bottom<=top;});"
        "  if(!prior.length)return false;"
        "  prior.sort(function(a,b){return b.getBoundingClientRect().bottom-a.getBoundingClientRect().bottom;});"
        "  var start=prior[0].getBoundingClientRect().bottom;"
        "  return !accountHeadings.some(function(h){var y=h.getBoundingClientRect().top;return y>=start&&y<=top;});"
        " }"
        " function inAccountSection(s) {"
        "  var top=s.getBoundingClientRect().top;"
        "  return accountHeadings.some(function(h){return h.getBoundingClientRect().bottom<=top;});"
        " }"
        " function inUnfilteredInbox() {"
        "  if(location.pathname.indexOf('/direct/inbox/')!==0)return false;"
        "  var input=Array.prototype.find.call(document.querySelectorAll('input'),function(x){"
        "   var p=(x.getAttribute('placeholder')||'').trim().toLowerCase();"
        "   return p.indexOf('search')===0||p.indexOf('검색')===0;});"
        "  return !!input && !(input.value||'').trim() && !messageHeadings.length && !accountHeadings.length;"
        " }"
        " var spans = Array.prototype.slice.call(scope.querySelectorAll('span')).filter(function(s) {"
        "  var b=s.getBoundingClientRect(); var t=(s.textContent||'').trim();"
        "  return b.width>0 && b.height>0 && t && t.length<=120; });"
        " function isMessageSearchSummary(lines) {"
        "  var detail=lines.slice(1).join(' ').trim().toLowerCase();"
        "  return /^\\d+\\s+matched messages?$/.test(detail)"
        "   || (/\\d+/.test(detail) && /메시지/.test(detail) && /일치/.test(detail));"
        " }"
        " function entry(s) {"
        "  var thread=s.closest('a[href*=\"/direct/t/\"]');"
        "  var root=thread || s.closest('button,[role=\"button\"]') || s;"
        "  var lines=(root.innerText||root.textContent||'').split('\\n').map(function(x){return x.trim();}).filter(Boolean);"
        "  return {node:root,name:(s.textContent||'').trim(),primary:lines.length?lines[0]:'',"
        "   existing:!!thread||inUnfilteredInbox()||(inMessagesSection(s)&&!isMessageSearchSummary(lines)),"
        "   account:inAccountSection(s),messageSummary:isMessageSearchSummary(lines)}; }"
        " function unique(entries) { var roots=[]; return entries.filter(function(e) {"
        "  if(roots.indexOf(e.node)>=0)return false; roots.push(e.node); return true; }); }"
        " function normalized(t) { var x=t.toLowerCase(); return x.charAt(0)==='@' ? x.slice(1) : x; }"
        " var exact = unique(spans.filter(function(s) { return normalized((s.textContent||'').trim())===q; }).map(entry));"
        " var existing = exact.filter(function(e) { return e.existing && normalized(e.primary)===q; });"
        " var accounts = exact.filter(function(e) { return e.account; });"
        " var candidates = existingOnly ? existing : (accountHeadings.length ? accounts : exact.filter(function(e){return !e.messageSummary;}));"
        " if (candidates.length===0) return 'MISS';"
        " if (candidates.length!==1) return 'AMBIGUOUS';"
        " if (requireFollowing) {"
        "  var rowText=(candidates[0].node.innerText||candidates[0].node.textContent||'').trim().toLowerCase();"
        "  var rowTokens=rowText.split(/\\s+/).filter(Boolean);"
        "  var isFollowing=followingLabels.some(function(label){return rowTokens.indexOf(label)>=0;});"
        "  if(!isFollowing) return 'NOT_FOLLOWING';"
        " }"
        " var el=candidates[0].node;"
        f" {_CLICK_SEQUENCE_JS}"
        " return 'CLICKED:' + candidates[0].name; })()"
    )


def _clicked_instagram_name(result: str, fallback: str) -> str:
    def normalized(value: str) -> str:
        return value.strip().removeprefix("@").casefold()

    if result.startswith("CLICKED:"):
        matched = result.partition(":")[2].strip()
        if matched and normalized(matched) == normalized(fallback):
            # Instagram's account picker commonly renders a bare username even
            # when the operator supplied an exact @handle. Preserve that exact
            # identity marker so opened-thread verification uses the profile
            # link guard instead of treating the username as a display name.
            if _is_exact_instagram_handle(fallback):
                return fallback.strip()
            return matched
    return ""


def _is_exact_instagram_handle(recipient: str) -> bool:
    return INSTAGRAM_USERNAME_RE.fullmatch(recipient.strip()) is not None


def _verify_followed_instagram_profile(recipient: str) -> None:
    """Require positive Following/팔로잉 proof on one exact @username profile."""
    if not _is_exact_instagram_handle(recipient):
        raise InstagramWebError("instagram_exact_handle_required")
    username = recipient.strip()[1:]
    profile_url_js = json.dumps(f"https://www.instagram.com/{username}/", ensure_ascii=False)
    expected_path_js = json.dumps(f"/{username.casefold()}/", ensure_ascii=False)
    following_labels_js = json.dumps([label.casefold() for label in INSTAGRAM_FOLLOWING_LABELS], ensure_ascii=False)
    follow_labels_js = json.dumps([label.casefold() for label in INSTAGRAM_FOLLOW_LABELS], ensure_ascii=False)
    _chrome_js(
        f"(function(){{location.href={profile_url_js};return 'NAVIGATING';}})()"
    )
    check_js = (
        "(function(){"
        "if(document.querySelector('input[name=\"username\"]'))return 'LOGIN';"
        f"var expected={expected_path_js};"
        "if(location.pathname.toLowerCase()!==expected)return 'LOADING';"
        "var main=document.querySelector('main,[role=\"main\"]');"
        "if(!main)return 'LOADING';"
        "var header=main.querySelector('header');"
        "if(!header)return 'UNKNOWN';"
        "var nodes=Array.from(header.querySelectorAll('button,[role=\"button\"]')).filter(function(x){"
        " var r=x.getBoundingClientRect();return r.width>0&&r.height>0;});"
        "function labelsFor(x){var values=[x.innerText||x.textContent||'',x.getAttribute('aria-label')||''];"
        " Array.from(x.querySelectorAll('[aria-label]')).forEach(function(y){values.push(y.getAttribute('aria-label')||'');});"
        " return values.map(function(v){return v.replace(/\\s+/g,' ').trim().toLowerCase();}).filter(Boolean);};"
        "var labels=[].concat.apply([],nodes.map(labelsFor));"
        f"var following={following_labels_js};var follow={follow_labels_js};"
        "if(labels.some(function(x){return following.indexOf(x)>=0;}))return 'FOLLOWING';"
        "if(labels.some(function(x){return follow.indexOf(x)>=0;}))return 'NOT_FOLLOWING';"
        "return 'UNKNOWN';})()"
    )
    last_state = "LOADING"
    for _ in range(20):
        time.sleep(0.5)
        last_state = _chrome_js(check_js)
        if last_state == "FOLLOWING":
            return
        if last_state == "NOT_FOLLOWING":
            raise InstagramWebError("instagram_recipient_not_followed")
        if last_state == "LOGIN":
            raise InstagramWebError("instagram_not_logged_in")
    raise InstagramWebError("instagram_follow_state_unavailable", last_state)


def _open_new_instagram_recipient(recipient: str, *, require_following: bool) -> str:
    """Open one exact account through Instagram's compose UI.

    Normal sends require a positive Following/팔로잉 marker in the same result
    row. Explicit new-person sends retain exact/unique identity checks but may
    bypass only that relationship requirement.
    """
    profile_following_verified = False
    if require_following and _is_exact_instagram_handle(recipient):
        _verify_followed_instagram_profile(recipient)
        profile_following_verified = True

    _chrome_js(
        "(function(){location.href='https://www.instagram.com/direct/new/';"
        "return 'NAVIGATING';})()"
    )
    for _ in range(15):
        time.sleep(0.5)
        state = _start_instagram_thread_search(recipient)
        if state == "SEARCHING":
            break
    else:
        raise InstagramWebError("instagram_new_message_unavailable")

    finder = _instagram_thread_finder_js(
        recipient,
        require_following=require_following and not profile_following_verified,
    )
    found = "MISS"
    for _ in range(12):
        time.sleep(0.5)
        found = _chrome_js(finder)
        if _clicked_instagram_name(found, recipient):
            break
        if found == "AMBIGUOUS":
            raise InstagramWebError("instagram_new_recipient_not_found")
        if found == "NOT_FOLLOWING":
            stage = "instagram_exact_handle_required" if require_following else "instagram_recipient_not_followed"
            raise InstagramWebError(stage)
    matched_name = _clicked_instagram_name(found, recipient)
    if not matched_name:
        raise InstagramWebError("instagram_new_recipient_not_found")

    advanced = False
    thread_opened = False
    for _ in range(10):
        if _chrome_js("location.pathname.indexOf('/direct/t/') === 0 ? 'OPEN' : 'WAIT'") == "OPEN":
            thread_opened = True
            if _open_thread_matches(matched_name):
                return matched_name
            time.sleep(0.5)
            continue
        if not advanced:
            advanced = _chrome_js(
                "(function(){var buttons=Array.prototype.slice.call(document.querySelectorAll('button,[role=\"button\"]'));"
                "var b=buttons.find(function(x){var t=(x.textContent||'').trim();"
                "return t==='Chat'||t==='Next'||t==='채팅'||t==='다음';});"
                "if(!b)return 'NO_BUTTON';" + _CLICK_SEQUENCE_JS + " return 'CLICKED';})()"
            ) == "CLICKED"
        time.sleep(0.5)
    if thread_opened:
        raise InstagramWebError("instagram_thread_not_found")
    raise InstagramWebError("instagram_thread_not_opened")


def _trusted_enter() -> None:
    """Fallback: one real Enter keystroke (layout-independent) if the synthetic
    Send-button click is ever ignored."""
    result = subprocess.run(
        ["osascript", "-e", 'tell application "System Events" to key code 36'],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or "could not press Enter"
        stage, detail = _instagram_error_stage(stderr)
        raise InstagramWebError(stage, detail)


def _trusted_paste_with_clipboard_restored(message: str) -> dict[str, Any]:
    """Paste Unicode text under verified, ownership-guarded clipboard custody."""
    transaction = clipboard_safety.PlainTextClipboardTransaction(
        run=subprocess.run
    )
    try:
        with transaction:
            transaction.stage(message)
            result = subprocess.run(
                [
                    "osascript",
                    "-e",
                    """
tell application "Google Chrome" to activate
delay 0.2
tell application "System Events"
    tell process "Google Chrome"
        set frontmost to true
        key code 9 using {command down}
    end tell
end tell
delay 0.35
""",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                stderr = result.stderr.strip() or "could not paste into Instagram"
                stage, detail = _instagram_error_stage(stderr)
                raise InstagramWebError(stage, detail)
    except clipboard_safety.ClipboardSafetyError as exc:
        clipboard_metadata = transaction.privacy_metadata()
        replacement_started = (
            clipboard_metadata.get("clipboard_replacement_attempted") is True
        )
        raise InstagramWebError(
            (
                "instagram_send_unconfirmed"
                if replacement_started
                else f"instagram_{exc.stage}"
            ),
            (
                "Instagram did not confirm clipboard custody after staging the message."
                if replacement_started
                else "Jarvis stopped before changing a rich, non-text, or unstable clipboard."
            ),
            clipboard_metadata=clipboard_metadata,
        ) from exc
    except Exception as exc:
        if isinstance(exc, InstagramWebError):
            exc.clipboard_metadata = transaction.privacy_metadata()
        raise
    return transaction.privacy_metadata()


_CLICK_SEQUENCE_JS = (
    "var rect = el.getBoundingClientRect();"
    " var o = {bubbles: true, cancelable: true, view: window,"
    "  clientX: rect.x + rect.width/2, clientY: rect.y + rect.height/2, button: 0};"
    " ['pointerdown','mousedown','pointerup','mouseup','click'].forEach(function(t) {"
    "  el.dispatchEvent(t.indexOf('pointer') === 0 ? new PointerEvent(t, o) : new MouseEvent(t, o)); });"
)


def open_instagram_thread(
    recipient: str,
    *,
    allow_new_recipient: bool = False,
    allow_followed_recipient: bool = False,
) -> str:
    """Open the DM thread whose displayed name EXACTLY matches `recipient`.

    Raises InstagramWebError (stage, detail) if the inbox never loads, the name
    is not in the thread list (the detail lists the names that ARE visible), or
    the thread never opens. Used by both send_instagram_dm and call_instagram.
    """
    _guard_no_active_instagram_call()
    _focus_instagram_tab()
    if _open_thread_matches(recipient):
        return recipient
    _navigate_instagram_inbox()

    for _ in range(15):
        state = _chrome_js(
            "(function() {"
            " if (document.querySelector('input[name=\"username\"]')) return 'login';"
            " var rows = document.querySelectorAll('[role=\"button\"] img');"
            " var search=Array.prototype.find.call(document.querySelectorAll('input'),function(x){"
            "  var p=(x.getAttribute('placeholder')||'').trim().toLowerCase();"
            "  return p.indexOf('search')===0||p.indexOf('검색')===0;});"
            " return rows.length > 0 && search ? 'ready' : 'booting'; })()"
        )
        if state == "ready":
            break
        if state == "login":
            raise InstagramWebError("instagram_not_logged_in", "Chrome is not logged into Instagram — log in once and retry.")
        time.sleep(1)
    else:
        raise InstagramWebError("instagram_not_loaded", "The Instagram DM inbox never finished loading.")

    find_and_click = _instagram_thread_finder_js(recipient, existing_only=True)
    found = _chrome_js(find_and_click)

    def search_and_select(query: str) -> str:
        search_state = "NO_SEARCH_INPUT"
        for _ in range(6):
            search_state = _start_instagram_thread_search(query)
            if search_state == "SEARCHING":
                break
            if search_state not in {"NO_SEARCH_INPUT", "SEARCH_FAILED"}:
                return "MISS"
            time.sleep(0.3)
        if search_state == "SEARCHING":
            for _ in range(12):
                time.sleep(0.5)
                search_result = _chrome_js(find_and_click)
                if _clicked_instagram_name(search_result, recipient) or search_result == "AMBIGUOUS":
                    return search_result
        return "MISS"

    if not _clicked_instagram_name(found, recipient):
        # Instagram's empty, focused inbox-search overlay can expose exact
        # one-to-one conversations that are absent from the virtualized recent
        # thread list. Check that deterministic surface before entering a text
        # query, whose "Messages" results may instead be message-content hits.
        found = search_and_select("")
    if not _clicked_instagram_name(found, recipient) and found != "AMBIGUOUS":
        found = search_and_select(recipient)
    matched_name = _clicked_instagram_name(found, recipient)
    if not matched_name:
        if allow_new_recipient and found != "AMBIGUOUS":
            return _open_new_instagram_recipient(recipient, require_following=False)
        if allow_followed_recipient and found != "AMBIGUOUS":
            return _open_new_instagram_recipient(recipient, require_following=True)
        stage = "instagram_recipient_not_known" if found != "AMBIGUOUS" else "instagram_thread_not_found"
        raise InstagramWebError(stage)

    advanced = False
    thread_opened = False
    for _ in range(10):
        time.sleep(0.5)
        if _chrome_js("location.pathname.indexOf('/direct/t/') === 0 ? 'OPEN' : 'LIST'") == "OPEN":
            thread_opened = True
            if _open_thread_matches(matched_name):
                return matched_name
            continue
        if not advanced:
            advanced = _chrome_js(
                "(function(){var buttons=Array.prototype.slice.call(document.querySelectorAll('button,[role=\"button\"]'));"
                "var b=buttons.find(function(x){var t=(x.textContent||'').trim();"
                "return t==='Chat'||t==='Next'||t==='채팅'||t==='다음';});"
                "if(!b)return 'NO_BUTTON';" + _CLICK_SEQUENCE_JS + " return 'CLICKED';})()"
            ) == "CLICKED"
    if thread_opened:
        raise InstagramWebError("instagram_thread_not_found")
    raise InstagramWebError("instagram_thread_not_opened", f"The thread for '{recipient}' never opened, so nothing was typed.")


def _revalidate_instagram_send_target(recipient: str) -> None:
    """Recheck the exact direct thread and its sole visible composer before Cmd-V."""
    recipient_js = json.dumps(recipient, ensure_ascii=False)
    selector_js = json.dumps(INSTAGRAM_COMPOSER_SELECTOR)
    state = _chrome_js(
        "(function() {"
        " if (location.pathname.indexOf('/direct/t/') !== 0) return 'NOT_THREAD';"
        f" var raw={recipient_js}.trim().toLowerCase();"
        " var q=raw.charAt(0)==='@' ? raw.slice(1) : raw;"
        " var main=document.querySelector('[role=\"main\"]')||document.querySelector('main')||document.body;"
        " var header=main.querySelector('header')||document.querySelector('header');"
        " function visibleNode(x){var r=x.getBoundingClientRect();return r.width>0&&r.height>0;}"
        " var headings=Array.prototype.slice.call(main.querySelectorAll('h1,h2,h3,[role=\"heading\"]'))"
        "  .filter(function(h){return visibleNode(h)&&!h.closest('nav,[role=\"navigation\"]');});"
        " var matched=false;"
        " if(raw.charAt(0)==='@'){"
        "  var expected='/'+q+'/';"
        "  var linked=function(a){try{return new URL(a.href,location.origin).pathname.toLowerCase()===expected;}"
        "   catch(_e){return false;}};"
        "  matched=(!!header&&Array.prototype.some.call(header.querySelectorAll('a[href]'),linked))"
        "   ||headings.some(function(h){var a=h.closest('a[href]');return !!a&&linked(a);});"
        " }else{"
        "  matched=(!!header&&Array.prototype.some.call(header.querySelectorAll('span,h1,h2,h3,[role=\"heading\"]'),"
        "   function(x){return visibleNode(x)&&(x.textContent||'').trim().toLowerCase()===q;}))"
        "   ||headings.some(function(h){return (h.textContent||'').trim().toLowerCase()===q;});"
        " }"
        " if(!matched) return 'NOT_MATCH';"
        f" var candidates=Array.prototype.slice.call(document.querySelectorAll({selector_js}));"
        " var visible=candidates.filter(visibleNode);"
        " if (visible.length!==1) return visible.length ? 'AMBIGUOUS_COMPOSER' : 'NO_COMPOSER';"
        " var el=visible[0];"
        " if (!document.hasFocus() || document.activeElement!==el) return 'COMPOSER_UNFOCUSED';"
        " return 'SEND_TARGET_READY'; })()"
    )
    if state in {"NOT_THREAD", "NOT_MATCH"}:
        raise InstagramWebError(
            "instagram_thread_not_found",
            "The exact Instagram conversation changed before the trusted paste, so nothing was pasted or sent.",
        )
    if state != "SEND_TARGET_READY":
        raise InstagramWebError(
            "instagram_compose_failed",
            "Instagram's exact direct-thread composer changed before the trusted paste, so nothing was pasted or sent.",
        )


def _send_instagram_dm(recipient: str, message: str, *, allow_new_recipient: bool = False) -> str:
    matched_recipient = open_instagram_thread(
        recipient,
        allow_new_recipient=allow_new_recipient,
        allow_followed_recipient=not allow_new_recipient,
    )
    message_js = json.dumps(message.strip(), ensure_ascii=False)
    count_message_js = (
        "(function() {"
        f" var expected = {message_js};"
        " var main = document.querySelector('[role=\"main\"]') || document.body;"
        " var matches = Array.prototype.filter.call(main.querySelectorAll('div,span'), function(x) {"
        "  if (x.closest('[contenteditable=\"true\"]')) return false;"
        "  var r=x.getBoundingClientRect();"
        "  if (r.width<=0 || r.height<=0 || (x.textContent||'').trim()!==expected) return false;"
        "  return !Array.prototype.some.call(x.children, function(c) {"
        "   return (c.textContent||'').trim()===expected;"
        "  });"
        " });"
        " return String(matches.length); })()"
    )
    baseline_text = _chrome_js(count_message_js)
    try:
        baseline_count = int(baseline_text)
    except (TypeError, ValueError) as exc:
        raise InstagramWebError(
            "instagram_confirmation_unavailable",
            "Instagram did not return a valid delivery baseline.",
        ) from exc

    # Instagram's current Lexical editor ignores synthetic insertText events.
    # Focus and select its contents through JS, then use one trusted Cmd-V. This
    # remains Unicode-safe and the clipboard is restored immediately afterwards.
    composer_selector_js = json.dumps(INSTAGRAM_COMPOSER_SELECTOR)
    prepared = _chrome_js(
        "(function() {"
        " var candidates = Array.prototype.slice.call(document.querySelectorAll("
        f"  {composer_selector_js}));"
        " var visible=candidates.filter(function(x){var r=x.getBoundingClientRect();return r.width>0&&r.height>0;});"
        " if (visible.length!==1) return visible.length ? 'AMBIGUOUS_COMPOSER' : 'NO_COMPOSER';"
        " var el=visible[0];"
        f" {_CLICK_SEQUENCE_JS}"
        " el.focus();"
        " if (!document.hasFocus() || document.activeElement !== el) return 'COMPOSER_UNFOCUSED';"
        " var range=document.createRange(); range.selectNodeContents(el);"
        " var sel=window.getSelection(); sel.removeAllRanges(); sel.addRange(range);"
        " return 'COMPOSER_FOCUSED'; })()"
    )
    if prepared != "COMPOSER_FOCUSED":
        raise InstagramWebError(
            "instagram_compose_failed",
            "Instagram's message composer could not be focused, so nothing was pasted or sent.",
        )
    _revalidate_instagram_send_target(matched_recipient)
    _trusted_paste_with_clipboard_restored(message)
    time.sleep(0.2)
    staged = _chrome_js(
        "(function() {"
        f" var candidates=Array.prototype.slice.call(document.querySelectorAll({composer_selector_js}));"
        " var visible=candidates.filter(function(x){var r=x.getBoundingClientRect();return r.width>0&&r.height>0;});"
        " var el=visible.length===1 ? visible[0] : null;"
        " return el ? 'TEXT:' + el.textContent : 'NO_COMPOSER'; })()"
    )
    if not staged.startswith("TEXT:") or staged[5:].strip() != message.strip():
        raise InstagramWebError(
            "instagram_compose_failed",
            f"Could not stage the message in the composer (got {_display_text(staged, 80)!r}), so nothing was sent.",
        )

    # Instagram increasingly ignores synthetic click events for its Send
    # control. Use one trusted Enter only after proving that Chrome and the
    # exact composer own keyboard focus; this avoids a hanging synthetic click
    # while retaining the fail-closed target guard.
    focused = _chrome_js(
        "(function() {"
        f" var candidates=Array.prototype.slice.call(document.querySelectorAll({composer_selector_js}));"
        " var visible=candidates.filter(function(x){var r=x.getBoundingClientRect();return r.width>0&&r.height>0;});"
        " var el=visible.length===1 ? visible[0] : null;"
        " if (el) el.focus();"
        " if (!document.hasFocus()) return 'WINDOW_UNFOCUSED';"
        " return (document.activeElement === el) ? 'COMPOSER_FOCUSED' : 'WRONG_ELEMENT'; })()"
    )
    if focused != "COMPOSER_FOCUSED":
        raise InstagramWebError(
            "instagram_compose_failed",
            "Instagram's message composer lost keyboard focus, so nothing was sent.",
        )
    try:
        _trusted_enter()
    except Exception as exc:
        raise InstagramWebError(
            "instagram_send_unconfirmed",
            "Instagram could not confirm whether the trusted Enter action sent the staged message.",
            clipboard_metadata=dict(
                getattr(exc, "clipboard_metadata", {}) or {}
            ),
            send_attempted=True,
        ) from exc

    check_js = (
        "(function() {"
        f" var candidates=Array.prototype.slice.call(document.querySelectorAll({composer_selector_js}));"
        " var visible=candidates.filter(function(x){var r=x.getBoundingClientRect();return r.width>0&&r.height>0;});"
        " var composer=visible.length===1 ? visible[0] : null;"
        " if (!composer || composer.textContent.trim() !== '') return 'FULL';"
        f" var expected = {message_js};"
        f" var baseline = {baseline_count};"
        " var main = document.querySelector('[role=\"main\"]') || document.body;"
        " var visible = Array.prototype.filter.call(main.querySelectorAll('div,span'), function(x) {"
        "  if (x.closest('[contenteditable=\"true\"]')) return false;"
        "  var r=x.getBoundingClientRect();"
        "  if (r.width<=0 || r.height<=0 || (x.textContent||'').trim()!==expected) return false;"
        "  return !Array.prototype.some.call(x.children, function(c) {"
        "   return (c.textContent||'').trim()===expected;"
        "  });"
        " }).length;"
        " return visible>baseline ? 'CONFIRMED' : 'EMPTY_UNCONFIRMED'; })()"
    )
    for _ in range(10):
        time.sleep(0.5)
        try:
            confirmation_state = _chrome_js(check_js)
        except Exception as exc:
            raise InstagramWebError(
                "instagram_send_unconfirmed",
                "Instagram delivery confirmation failed after the trusted Enter action.",
                send_attempted=True,
            ) from exc
        if confirmation_state == "CONFIRMED":
            return matched_recipient
    raise InstagramWebError(
        "instagram_send_unconfirmed",
        "The message was staged in the right thread but Instagram did not show it in the conversation.",
        send_attempted=True,
    )


def make_instagram_tools(config: JarvisConfig):
    def send_instagram_dm(args: dict[str, Any]) -> ToolResult:
        recipient = _short(args.get("to") or args.get("recipient") or args.get("handle"), MAX_RECIPIENT_CHARS)
        message = _short(args.get("message") or args.get("body") or args.get("text"), MAX_MESSAGE_CHARS)
        allow_new_recipient = args.get("allow_new_recipient") is True
        approval_binding = args.get(INSTAGRAM_APPROVAL_BINDING_KEY)
        bound_for_approval = approval_binding is not None
        if not recipient:
            return ToolResult(
                "send_instagram_dm",
                False,
                "Instagram recipient ('to') is required.",
                _safe_metadata(
                    controls_computer=False,
                    message_chars=len(message),
                    **_instagram_dm_handoff(
                        recipient="",
                        message_chars=len(message),
                        status="refused",
                        reason="missing_recipient",
                        allow_new_recipient=allow_new_recipient,
                    ),
                ),
            )
        if not message:
            return ToolResult(
                "send_instagram_dm",
                False,
                "Instagram DM message text is required.",
                _safe_metadata(
                    controls_computer=False,
                    to=_display_text(recipient),
                    message_chars=0,
                    **_instagram_dm_handoff(
                        recipient=recipient,
                        message_chars=0,
                        status="refused",
                        reason="missing_message",
                        allow_new_recipient=allow_new_recipient,
                    ),
                ),
            )
        if bound_for_approval:
            expected_binding = _instagram_approval_binding(
                recipient,
                message,
                allow_new_recipient,
            )
            if (
                not _is_exact_instagram_handle(recipient)
                or type(approval_binding) is not str
                or not hmac.compare_digest(approval_binding, expected_binding)
            ):
                return _instagram_approval_refusal(
                    recipient=recipient,
                    message_chars=len(message),
                    reason="approval_binding_invalid",
                    output=(
                        "The approved Instagram username, message, or recipient-mode binding is invalid, "
                        "so nothing ran. Submit a fresh exact-@username request for review."
                    ),
                    allow_new_recipient=allow_new_recipient,
                )
            resolved_recipient = recipient
            resolution_metadata = {
                "original_to": _display_text(recipient),
                "contact_lookup_attempted": False,
                "contact_resolution_status": "bound_exact_username_before_approval",
                "contact_match_count": 0,
                "contact_candidates": [],
                "instagram_recipient_bound_before_approval": True,
                "instagram_message_bound_before_approval": True,
                "instagram_new_recipient_mode_bound_before_approval": True,
                "instagram_identity_kind": "exact_username",
            }
            resolution_message = ""
        else:
            # Preserve deterministic direct-handler compatibility for isolated
            # connector tests. Real runtime sends always use the resolver-bound
            # exact-username branch above.
            resolved_recipient, resolution_metadata, resolution_message = _resolve_send_recipient(recipient)
        resolution_metadata = {
            **resolution_metadata,
            "contact_lookup_attempted": _metadata_bool(resolution_metadata.get("contact_lookup_attempted")),
        }
        if not resolved_recipient:
            return ToolResult(
                "send_instagram_dm",
                False,
                resolution_message,
                _safe_metadata(
                    controls_computer=False,
                    reads_personal_data=_metadata_bool(resolution_metadata.get("contact_lookup_attempted")),
                    to=_display_text(recipient),
                    message_chars=len(message),
                    reason=resolution_metadata.get("contact_resolution_status"),
                    **resolution_metadata,
                    **_instagram_dm_handoff(
                        recipient=recipient,
                        message_chars=len(message),
                        status="refused",
                        reason=str(resolution_metadata.get("contact_resolution_status") or "contact_resolution_failed"),
                        allow_new_recipient=allow_new_recipient,
                        **resolution_metadata,
                    ),
                ),
            )
        try:
            sent_recipient = _send_instagram_dm(
                resolved_recipient,
                message,
                allow_new_recipient=allow_new_recipient,
            )
            sent_note = (
                f"Instagram showed the outgoing message in the exact conversation for "
                f"{_display_text(sent_recipient)}; recipient delivery is not confirmed."
            )
            if allow_new_recipient:
                sent_note = (
                    f"Instagram showed the outgoing message for {_display_text(sent_recipient)} in explicitly "
                    "approved new-recipient mode; recipient delivery is not confirmed."
                )
            return ToolResult(
                "send_instagram_dm",
                True,
                sent_note,
                _safe_metadata(
                    reads_personal_data=_metadata_bool(resolution_metadata.get("contact_lookup_attempted")),
                    executes_side_effect=True,
                    external_side_effect=True,
                    requires_approval=True,
                    to=_display_text(recipient),
                    resolved_to=_display_text(sent_recipient),
                    allow_new_recipient=allow_new_recipient,
                    recipient_policy="explicit_new_allowed" if allow_new_recipient else "existing_or_followed",
                    message_chars=len(message),
                    send_attempted=True,
                    sender_side_send_confirmed=True,
                    recipient_delivery_confirmed=False,
                    delivery_confirmed=False,
                    delivery_evidence="sender_side_only",
                    recipient_confirmation_required=True,
                    **clipboard_safety.privacy_boundary_metadata(
                        snapshot_attempted=True,
                        snapshot_captured=True,
                        replacement_attempted=True,
                        text_restored=True,
                    ),
                    **resolution_metadata,
                    **_instagram_dm_handoff(
                        recipient=recipient,
                        message_chars=len(message),
                        status="sender_confirmed",
                        send_attempted=True,
                        controls_computer=True,
                        allow_new_recipient=allow_new_recipient,
                        **resolution_metadata,
                    ),
                ),
            )
        except Exception as e:
            failure_stage, classified_output = _instagram_dm_failure(e)
            outcome_unknown = not _send_outcome_is_known_not_sent(e, failure_stage)
            clipboard_metadata = dict(
                getattr(e, "clipboard_metadata", {}) or {}
            )
            if (
                not clipboard_metadata
                and failure_stage == "instagram_send_unconfirmed"
            ):
                clipboard_metadata = clipboard_safety.privacy_boundary_metadata(
                    snapshot_attempted=True,
                    snapshot_captured=True,
                    replacement_attempted=True,
                    text_restored=True,
                )
            failure_output = (
                _post_attempt_send_recovery_guidance()
                if outcome_unknown
                else f"{classified_output} {KNOWN_NOT_SENT_SEND_RECOVERY_ACTION}"
            )
            failure_metadata = _safe_metadata(
                reads_personal_data=_metadata_bool(resolution_metadata.get("contact_lookup_attempted")),
                executes_side_effect=True,
                external_side_effect=True,
                requires_approval=True,
                to=_display_text(recipient),
                resolved_to=_display_text(resolved_recipient),
                allow_new_recipient=allow_new_recipient,
                recipient_policy="explicit_new_allowed" if allow_new_recipient else "existing_or_followed",
                message_chars=len(message),
                send_attempted=True,
                failure_stage=failure_stage,
                instagram_send_stage=failure_stage,
                error_type=type(e).__name__,
                exception_type=type(e).__name__,
                outcome_known=not outcome_unknown,
                outcome_unknown=outcome_unknown,
                side_effect_possible=outcome_unknown,
                retry_safe=not outcome_unknown,
                automatic_retry_allowed=False,
                authorizes_retry=False,
                **clipboard_metadata,
                **resolution_metadata,
                **_instagram_dm_handoff(
                    recipient=recipient,
                    message_chars=len(message),
                    status="outcome_unknown" if outcome_unknown else "failed",
                    reason="send_outcome_unknown" if outcome_unknown else "send_error",
                    send_attempted=True,
                    controls_computer=True,
                    allow_new_recipient=allow_new_recipient,
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
                "send_instagram_dm",
                False,
                failure_output,
                failure_metadata,
            )

    from jarvis_v2.tools.registry import Tool, _tool_argument_contract

    return [
        Tool(
            "send_instagram_dm",
            "Send an Instagram DM through a logged-in Chrome session. Args: to, message, optional allow_new_recipient. "
            "The approval target must be an exact @username; display names and Contacts aliases are refused before "
            "approval. Default is existing conversations or exact accounts visibly marked Following; "
            "allow_new_recipient must come from explicit user wording. Best-effort GUI automation.",
            RiskLevel.HIGH_RISK,
            send_instagram_dm,
            "personal",
            argument_contract=_tool_argument_contract(
                required_strings=("to", "message"),
                optional_booleans=("allow_new_recipient",),
            ),
            approval_argument_resolver=resolve_instagram_send_approval,
            approval_argument_contract=_tool_argument_contract(
                required_strings=("to", "message", INSTAGRAM_APPROVAL_BINDING_KEY),
                required_booleans=("allow_new_recipient",),
            ),
        )
    ]
