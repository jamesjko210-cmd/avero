"""Best-effort calling for Jarvis V2 across FaceTime/phone, KakaoTalk, and Instagram.

Design mirrors the send connectors (kakao/imessage/instagram):

- **call_contact** — the RELIABLE native path. Resolves a name to a phone/email
  handle via macOS Contacts, then places a FaceTime video, FaceTime audio, or
  phone (Continuity → iPhone) call via the `facetime://` / `facetime-audio://` /
  `tel://` URL schemes. Needs a handle, exactly like iMessage.
- **call_kakao / call_instagram** — BEST-EFFORT GUI automation. These apps expose
  no calling API, so Jarvis opens the person's chat and tries to click the
  voice/video call button. Name-based (types into the app's own search). Fragile
  by nature; the approval screen shows exactly who will be called.
- **call_telegram** — DEFERRED. Telegram Desktop is not installed, and the Bot API
  cannot place calls, so this refuses with install guidance (parity with send).

All call tools are HIGH_RISK, so the executor's approval gate stops them until the
owner approves. The handler runs (places the call) only on the approved rerun.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import unicodedata
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    KNOWN_NOT_SENT_SEND_RECOVERY_ACTION,
    declare_failure_guidance,
    declare_known_not_sent_failure,
    declare_outcome_unknown_failure,
)
from jarvis_v2.agent.types import RiskLevel, ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.tools import clipboard_safety, contacts_connector

MAX_RECIPIENT_CHARS = 120
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)

# Accepted call modes, normalized. "phone" routes through Continuity to the iPhone.
# A plain "call X" defaults to a PHONE call; FaceTime only when explicitly asked.
_VALID_MODES = {"video", "audio", "phone"}
_DEFAULT_MODE = "phone"
_OWNER_SELF_RECIPIENTS = {
    "me",
    "myself",
    "self",
    "my telegram",
    "나",
    "나에게",
    "나한테",
    "내게",
    "저에게",
    "저한테",
    "본인",
}


def _short(value: Any, limit: int = MAX_RECIPIENT_CHARS) -> str:
    text = str(value or "").strip()
    return text[:limit] if len(text) > limit else text


def _display_text(value: Any, limit: int = MAX_RECIPIENT_CHARS) -> str:
    text = " ".join(str(value or "").strip().split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    return text[:limit] if len(text) > limit else text


class TelegramWebSendError(RuntimeError):
    def __init__(
        self,
        stage: str,
        detail: str = "",
        *,
        outcome_unknown: bool = False,
    ) -> None:
        self.stage = _display_text(stage, 80)
        self.detail = _display_text(detail, 220)
        self.outcome_unknown = outcome_unknown is True
        suffix = f": {self.detail}" if self.detail else ""
        super().__init__(f"Telegram Web send failed at {self.stage}{suffix}")


class TelegramOwnerSendError(RuntimeError):
    def __init__(
        self,
        stage: str,
        detail: str = "",
        *,
        outcome_unknown: bool = False,
    ) -> None:
        self.stage = _display_text(stage, 80)
        self.detail = _display_text(detail, 220)
        self.outcome_unknown = outcome_unknown is True
        suffix = f": {self.detail}" if self.detail else ""
        super().__init__(f"Owner Telegram send failed at {self.stage}{suffix}")


class InstagramCallError(RuntimeError):
    """Bound one Instagram call failure to whether activation may have run."""

    def __init__(
        self,
        stage: str,
        detail: str = "",
        *,
        outcome_unknown: bool,
        call_attempted: bool,
    ) -> None:
        if outcome_unknown is False and call_attempted is True:
            raise ValueError("a completed Instagram call attempt cannot be known-not-started")
        self.stage = _display_text(stage, 80)
        self.detail = _display_text(detail, 220)
        self.outcome_unknown = outcome_unknown is True
        self.call_attempted = call_attempted is True
        suffix = f": {self.detail}" if self.detail else ""
        super().__init__(f"Instagram call failed at {self.stage}{suffix}")


def _is_owner_self_recipient(value: Any) -> bool:
    normalized = _normalized_recipient_identity(value)
    return normalized in _OWNER_SELF_RECIPIENTS


def _normalized_recipient_identity(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", _short(value)).casefold()
    return " ".join(normalized.split())


def _send_owner_telegram(message: str) -> None:
    from jarvis_v2.automations import telegram_control

    owner = telegram_control._owner_chat_id()
    if not owner:
        raise TelegramOwnerSendError(
            "owner_not_configured",
            "Configure the owner Telegram channel, then retry.",
        )
    result = telegram_control.send_message(owner, message)
    if not isinstance(result, dict):
        raise TelegramOwnerSendError(
            "telegram_delivery_outcome_unknown",
            "Telegram returned no authoritative delivery acknowledgement.",
            outcome_unknown=True,
        )
    # `ok: true` alone is not an authoritative sender receipt. Telegram's
    # sendMessage success payload must identify both the created message and the
    # chat that accepted it. Fail closed on wrappers/partial batches too: one
    # accepted chunk followed by an error cannot be safely retried as a whole.
    if "results" in result:
        raise TelegramOwnerSendError(
            "telegram_delivery_outcome_unknown",
            "Telegram returned a partial or wrapped delivery acknowledgement.",
            outcome_unknown=True,
        )
    if result.get("ok") is True:
        payload = result.get("result")
        message_id = payload.get("message_id") if isinstance(payload, dict) else None
        chat = payload.get("chat") if isinstance(payload, dict) else None
        returned_chat_id = chat.get("id") if isinstance(chat, dict) else None
        valid_message_id = type(message_id) is int and message_id > 0
        valid_returned_chat_id = (
            not isinstance(returned_chat_id, bool)
            and isinstance(returned_chat_id, (int, str))
            and str(returned_chat_id) == owner
        )
        if valid_message_id and valid_returned_chat_id:
            return
        raise TelegramOwnerSendError(
            "telegram_delivery_outcome_unknown",
            "Telegram returned no authoritative delivery acknowledgement for the configured owner channel.",
            outcome_unknown=True,
        )
    if result.get("ok") is False:
        raise TelegramOwnerSendError(
            "telegram_api_rejected",
            "Check the Telegram bot connection and owner configuration, then retry.",
        )
    raise TelegramOwnerSendError(
        "telegram_delivery_outcome_unknown",
        "Telegram returned no authoritative delivery acknowledgement.",
        outcome_unknown=True,
    )


def _telegram_error_stage(stderr: str) -> tuple[str, str]:
    detail = _display_text(stderr, 220)
    low = stderr.lower()
    if "assistive access" in low or "accessibility" in low or "not allowed" in low and "system events" in low:
        return "macos_accessibility_permission", "Grant Accessibility permission to the process running Jarvis, then retry."
    if "allow javascript from apple events" in low or "execute javascript" in low:
        return "chrome_javascript_permission", "In Chrome, enable View > Developer > Allow JavaScript from Apple Events, then retry."
    if "no chat opened" in low:
        return "telegram_chat_not_opened", detail
    return "osascript_failed", detail


def _telegram_web_recovery_guidance(stage: str) -> str:
    return {
        "macos_accessibility_permission": (
            "In System Settings > Privacy & Security > Accessibility, allow the process running Jarvis, then retry."
        ),
        "chrome_javascript_permission": (
            "In Chrome, enable View > Developer > Allow JavaScript from Apple Events, then retry."
        ),
        "telegram_tab_hidden": "Bring the Telegram Web tab to the foreground, then retry.",
        "telegram_not_loaded": "Open web.telegram.org in Chrome, confirm you are logged in, then retry.",
        "telegram_chat_not_found": "Open or create the exact Telegram chat in Chrome, then retry.",
        "telegram_chat_not_opened": "Refresh Telegram Web and open the intended chat once, then retry.",
        "telegram_wrong_chat": "Jarvis stopped before acting because the opened chat was not the intended one.",
        "telegram_compose_failed": "Telegram's message composer was not ready. Refresh the chat, then retry.",
        "telegram_window_not_focused": "Bring the Telegram Web window to the foreground, then retry.",
        "telegram_enter_did_not_send": "Check Telegram's send-with-Enter setting, then retry.",
        "telegram_call_menu_missing": "Telegram did not show the chat menu. Refresh the chat, then retry.",
        "telegram_call_item_missing": "Telegram did not offer the requested call option for this chat.",
        "telegram_call_menu_not_open": "Telegram's call menu did not open. Refresh the chat, then retry.",
        "telegram_call_click_failed": "Telegram did not activate the requested call option. Check the chat, then retry.",
    }.get(stage, "")


def _telegram_action_failure(exc: Exception, *, label: str, fallback: str) -> tuple[str, str]:
    """Surface a bounded recovery step without exposing Telegram page state."""
    from jarvis_v2.tools.kakao_connector import guard_message

    guard = guard_message(exc)
    if guard:
        return "guard_stopped", f"{label} stopped safely: {guard}"
    if isinstance(exc, TelegramWebSendError):
        guidance = _telegram_web_recovery_guidance(exc.stage)
        if guidance:
            return exc.stage, f"{label} stopped at {exc.stage}: {guidance}"
        return exc.stage, fallback
    if isinstance(exc, TelegramOwnerSendError):
        guidance = {
            "owner_not_configured": "Configure the owner Telegram channel, then retry.",
            "telegram_api_rejected": "Check the Telegram bot connection and owner configuration, then retry.",
        }.get(exc.stage, "")
        if guidance:
            return exc.stage, f"{label} stopped at {exc.stage}: {guidance}"
        return exc.stage, fallback
    if isinstance(exc, subprocess.TimeoutExpired):
        return "automation_timeout", f"{label} automation timed out before Jarvis could confirm the action; nothing was retried."
    return "action_error", fallback


def _normalize_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"facetime", "face time", "video call", "videocall"}:
        return "video"
    if mode in {"voice", "voice call", "audio call", "facetime audio"}:
        return "audio"
    if mode in {"phone", "phone call", "cell", "mobile", "call"}:
        return "phone"
    return mode if mode in _VALID_MODES else _DEFAULT_MODE


def call_handle_for_mode(
    match: contacts_connector.ContactMatch,
    mode: Any,
) -> str:
    normalized_mode = _normalize_mode(mode)
    if normalized_mode == "phone":
        phones = match.all_phones()
        return phones[0] if phones else ""
    return match.handle


def call_handle_is_valid(mode: Any, handle: Any) -> bool:
    normalized_mode = _normalize_mode(mode)
    if normalized_mode == "phone":
        return contacts_connector.looks_like_phone_handle(handle)
    return contacts_connector.looks_like_handle(handle)


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
        "controls_computer": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    base.update(extra)
    return base


def _telegram_send_handoff(
    *,
    recipient: str,
    message_chars: int,
    status: str,
    reason: str = "",
    owner_self: bool = False,
    send_attempted: bool = False,
) -> dict[str, Any]:
    if status == "outcome_unknown":
        next_safe_commands = ["recent tool runs", "execution recovery"]
    elif status in {"failed", "known_not_started"}:
        next_safe_commands = ["setup check"]
    elif status == "sender_confirmed":
        next_safe_commands = ["recent tool runs"]
    else:
        next_safe_commands = ["safe next actions"]
    handoff = {
        "source": "send_telegram",
        "status": status,
        "reason": reason,
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": send_attempted,
        "changed": ["telegram_send_attempt"] if send_attempted else [],
        "content_in_handoff": False,
        "content_in_metadata": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "to": _display_text(recipient),
        "message_chars": message_chars,
        "send_attempted": send_attempted,
        "sender_side_send_confirmed": status == "sender_confirmed",
        "recipient_delivery_confirmed": False,
        "delivery_confirmed": False,
        "delivery_evidence": "sender_side_only" if status == "sender_confirmed" else "none",
        "recipient_confirmation_required": status == "sender_confirmed",
        "delivery_path": "owner_bot_api" if owner_self else "telegram_web",
        "approval_required_before_execution": True,
        "manual_review_required": True,
        "next_safe_command": next_safe_commands[0],
        "next_safe_commands": next_safe_commands,
    }
    return {
        "telegram_send_handoff_ready": True,
        "telegram_send_handoff": handoff,
        "ready_for_operator": True,
        "state_changed": send_attempted,
        "changed": handoff["changed"],
        "content_in_handoff": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _post_attempt_telegram_send_recovery_guidance(owner_self: bool) -> str:
    path = "owner Bot API" if owner_self else "Telegram Web"
    repair = (
        "bot connection and owner configuration"
        if owner_self
        else "Telegram Web login and macOS permissions"
    )
    return (
        f"The {path} delivery outcome is unknown. Check the exact conversation; do not resend "
        "automatically. If the message is absent, run `setup check`, repair the "
        f"{repair}, then submit a new approved send."
    )


def _call_handoff(
    *,
    channel: str,
    recipient: str,
    mode: str,
    status: str,
    reason: str = "",
    call_attempted: bool = False,
    controls_computer: bool = False,
    contact_lookup_attempted: bool = False,
    resolved_to: str = "",
    reads_clipboard: bool = False,
) -> dict[str, Any]:
    if status == "outcome_unknown":
        next_safe_commands = ["recent tool runs", "execution recovery"]
    elif status == "failed":
        next_safe_commands = ["setup check"]
    elif status == "call_requested":
        next_safe_commands = ["recent tool runs"]
    else:
        next_safe_commands = ["safe next actions"]
    boundaries = {
        "calls_model": False,
        "calls_external_service": False,
        "executes_tools": False,
        "reads_personal_data": contact_lookup_attempted or reads_clipboard,
        "reads_private_data": reads_clipboard,
        "reads_clipboard": reads_clipboard,
        "executes_side_effect": call_attempted,
        "external_side_effect": call_attempted,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": call_attempted,
        "controls_computer": controls_computer,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    handoff = {
        "source": f"call_{channel}",
        "status": status,
        "reason": reason,
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": call_attempted,
        "changed": [f"{channel}_call_attempt"] if call_attempted else [],
        "content_in_handoff": False,
        "content_in_metadata": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "channel": channel,
        "to": _display_text(recipient),
        "resolved_to": _display_text(resolved_to or recipient),
        "mode": mode,
        "contact_lookup_attempted": contact_lookup_attempted,
        "reads_clipboard": reads_clipboard,
        "call_attempted": call_attempted,
        "approval_required_before_execution": True,
        "manual_review_required": True,
        "next_safe_command": next_safe_commands[0],
        "next_safe_commands": next_safe_commands,
        "boundaries": boundaries,
    }
    return {
        "call_handoff_ready": True,
        "call_handoff": handoff,
        "ready_for_operator": True,
        "state_changed": call_attempted,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "call_handoff_boundaries": boundaries,
    }


def _call_attempt_recovery_guidance(channel: str) -> str:
    repair = {
        "FaceTime/phone": "FaceTime setup and sign-in",
        "KakaoTalk": "login and macOS Automation and Accessibility permissions",
        "Instagram": "login and macOS Automation and Accessibility permissions",
        "Telegram": "Telegram Web login and macOS permissions",
    }[channel]
    return (
        f"The {channel} call outcome is unknown. Check recent calls; do not retry automatically. "
        f"If none started, run `setup check`, repair {repair}, then submit a new approved call request."
    )


def _instagram_call_no_surface_recovery_guidance() -> str:
    return (
        "No verified Instagram call popup or outgoing controls appeared after the click, so the outcome is unknown. "
        "Check recent calls; do not retry automatically. If none started, run `setup check`, then submit a new approved call request."
    )


def _call_attempt_failure_metadata(
    metadata: dict[str, Any],
    *,
    output: str,
) -> dict[str, Any]:
    metadata.update(
        {
            "outcome_known": False,
            "outcome_unknown": True,
            "execution_outcome_unknown": True,
            "side_effect_possible": True,
            "retry_safe": False,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
        }
    )
    return declare_failure_guidance(
        metadata,
        output=output,
        action=output,
        commands=("setup check",),
    )


# --- FaceTime / phone (reliable native path) ---------------------------------


def _facetime_url(mode: str, handle: str) -> str:
    scheme = {"video": "facetime", "audio": "facetime-audio", "phone": "tel"}[mode]
    return f"{scheme}://{handle}"


def _place_native_call(mode: str, handle: str) -> str:
    """Open the FaceTime/phone URL scheme. Raises on non-zero exit."""
    url = _facetime_url(mode, handle)
    result = subprocess.run(["open", url], capture_output=True, text=True, timeout=20)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"could not open {mode} call")
    return result.stdout.strip()


def _resolve_call_contact(
    raw_recipient: str,
    mode: str,
) -> tuple[str | None, dict[str, Any], str]:
    """Resolve a name to a phone/email handle for a FaceTime/phone call.

    Native calls need a real handle (like iMessage), so a bare name must resolve to
    exactly one contact with a phone/email. Returns (handle, metadata, refusal).
    """
    if contacts_connector.looks_like_handle(raw_recipient):
        if not call_handle_is_valid(mode, raw_recipient):
            return None, {
                "contact_lookup_attempted": False,
                "contact_resolution_status": "invalid_phone_handle",
            }, (
                "A phone call requires a phone number; an email address cannot be dialed. "
                "Give me a phone number or request FaceTime audio/video instead."
            )
        return raw_recipient, {"contact_lookup_attempted": False, "contact_resolution_status": "skipped"}, ""
    matches = contacts_connector.resolve_contact(raw_recipient)
    base = {"contact_lookup_attempted": True, "contact_match_count": len(matches)}
    if len(matches) == 1:
        handle = call_handle_for_mode(matches[0], mode)
        if handle:
            return handle, {**base, "contact_resolution_status": "resolved"}, ""
        if _normalize_mode(mode) == "phone":
            return None, {**base, "contact_resolution_status": "missing_phone_handle"}, (
                f"I found \"{_display_text(matches[0].name)}\", but that contact has no phone number. "
                "Add a phone number or request FaceTime audio/video instead."
            )
    if len(matches) > 1:
        names = " or ".join(_display_text(m.name) for m in matches)
        return None, {**base, "contact_resolution_status": "ambiguous"}, (
            f"I found {len(matches)} people named \"{_display_text(raw_recipient)}\" — {names}? "
            "Tell me which one to call."
        )
    return None, {**base, "contact_resolution_status": "not_found"}, (
        f"I couldn't find a contact matching \"{_display_text(raw_recipient)}\", so I did not place the call. "
        "Add them to Contacts or give me a phone number."
    )


# --- KakaoTalk / Instagram (best-effort GUI automation) ----------------------


def _as_script_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _place_kakao_call(recipient: str, mode: str) -> str:
    """Open the KakaoTalk chat for `recipient` and try to click the call button.

    Best-effort: Kakao exposes no API. The chat is opened through the shared
    guarded fragment (raises the MAIN window, verifies the front window title
    matches the recipient, aborts otherwise), then the voice/video call button is
    clicked by its accessibility description.
    """
    from jarvis_v2.tools.kakao_connector import open_kakao_chat_fragment

    want_video = "true" if mode == "video" else "false"
    script = (
        open_kakao_chat_fragment(recipient)
        + f"""
        set wantVideo to {want_video}
        set clicked to false
        try
            set allButtons to every button of front window
            repeat with b in allButtons
                set d to ""
                try
                    set d to (description of b) as text
                end try
                if d is not "" then
                    if wantVideo then
                        if d contains "video" or d contains "페이스톡" or d contains "영상" then
                            click b
                            set clicked to true
                            exit repeat
                        end if
                    else
                        if d contains "call" or d contains "voice" or d contains "보이스톡" or d contains "통화" then
                            click b
                            set clicked to true
                            exit repeat
                        end if
                    end if
                end if
            end repeat
        end try
    end tell
end tell
try
    set the clipboard to savedClip
end try
return clicked
"""
    )
    from jarvis_v2.tools.kakao_connector import KakaoSendError, _kakao_error_stage, _run_with_clipboard_restored

    def run():
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=45)
        if result.returncode != 0:
            stage, detail = _kakao_error_stage(result.stderr.strip() or "KakaoTalk call automation failed")
            raise KakaoSendError(stage, detail)
        return result.stdout.strip()

    return _run_with_clipboard_restored(run)


# Telegram Web flow, verified live in Chrome on 2026-07-02:
# - Blind keystroke automation failed twice here: `keystroke` cannot type Korean
#   (가상연락처이), and nothing verified which field had focus. This flow drives
#   Telegram Web with JavaScript through Chrome's Apple Events bridge instead —
#   Unicode-safe and verifiable at every step.
# - Telegram-K ignores synthetic clicks/Enter on the send path (isTrusted
#   checks) and ignores programmatic hashchange, but honors the peer hash on
#   page BOOT. So: find the peer id in the chat list by exact title, set
#   location.hash, reload, verify the chat header IS the recipient, stage the
#   message via JS, then fire ONE real Enter (key code 36 — layout-independent,
#   works regardless of language) and confirm the composer cleared.

# The wrapper finds the Telegram tab BY URL on every call and executes the JS
# on that tab directly. Never use "active tab of front window": the operator often has
# two Telegram tabs across windows, and the front-window shortcut ran the whole
# flow against the wrong one (live failure — stale chat list, missing chats).
_CHROME_JS_WRAPPER = """on run argv
    tell application "Google Chrome"
        repeat with w in windows
            repeat with t in tabs of w
                if URL of t starts with "https://web.telegram.org" then
                    return execute t javascript (item 1 of argv)
                end if
            end repeat
        end repeat
    end tell
    error "no Telegram Web tab is open in Chrome"
end run"""


def _chrome_js(js: str, timeout: int = 20) -> str:
    """Run JavaScript in Chrome's Telegram tab (located by URL each call). The
    JS travels via argv, so no AppleScript escaping and raw Korean passes through."""
    result = subprocess.run(["osascript", "-e", _CHROME_JS_WRAPPER, js], capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "Chrome JavaScript bridge failed"
        stage, detail = _telegram_error_stage(stderr)
        raise TelegramWebSendError(stage, detail)
    return result.stdout.strip()


def _focus_telegram_tab() -> None:
    """Bring an existing Telegram Web tab to the front (or open one)."""
    script = """
tell application "Google Chrome"
    activate
    set found to false
    repeat with w in windows
        set i to 1
        repeat with t in tabs of w
            if URL of t starts with "https://web.telegram.org" then
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
        tell front window to make new tab with properties {URL:"https://web.telegram.org/k/"}
    end if
end tell
"""
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=20)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "could not focus a Telegram Web tab in Chrome"
        stage, detail = _telegram_error_stage(stderr)
        raise TelegramWebSendError(stage, detail)


def _trusted_enter() -> None:
    """One REAL Enter keystroke — Telegram requires a trusted event to send, and
    key code 36 is keyboard-layout-independent (safe for any language)."""
    result = subprocess.run(
        ["osascript", "-e", 'tell application "System Events" to key code 36'],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or "could not press Enter"
        stage, detail = _telegram_error_stage(stderr)
        raise TelegramWebSendError(stage, detail)


def _open_telegram_chat(recipient: str) -> str:
    """Shared verified chat-open flow: focus tab → fresh reload → exact-title
    peer lookup → hash+reload open → header verification. Returns the verified
    header. Raises TelegramWebSendError with a specific stage otherwise."""
    recipient_js = json.dumps(recipient, ensure_ascii=False)

    _focus_telegram_tab()

    # 1. Reload for a FRESH chat list. A long-open Telegram tab keeps a stale
    #    virtualized list (live failure: BotFather existed but wasn't in the
    #    stale render); a fresh boot renders the full recent list. Telegram-K
    #    only mounts while visible, so self-heal if the tab reports hidden.
    boot_js = (
        "(function() { if (document.hidden) return 'hidden';"
        " return document.querySelector('.chatlist a[data-peer-id]') ? 'ready' : 'booting'; })()"
    )
    for attempt in range(3):
        state = _chrome_js(boot_js)
        if state != "hidden":
            break
        _focus_telegram_tab()
        time.sleep(1)
    else:
        raise TelegramWebSendError(
            "telegram_tab_hidden",
            "Could not bring the Telegram tab to the foreground — Telegram Web only renders while visible.",
        )
    _chrome_js("location.reload(); 'reloading'")
    for _ in range(20):
        time.sleep(1)
        try:
            if _chrome_js(boot_js) == "ready":
                break
        except TelegramWebSendError:
            continue  # bridge can error mid-reload; keep polling
    else:
        raise TelegramWebSendError(
            "telegram_not_loaded",
            "Telegram Web never showed a chat list — check that Chrome is logged in at web.telegram.org.",
        )

    # 2. Find the peer id by EXACT chat title (case-insensitive). A miss returns
    #    the visible titles so the failure tells the operator what names ARE available.
    find_js = (
        "(function() {"
        " function normalized(v){return String(v||'').normalize('NFKC').trim().toLocaleLowerCase();}"
        f" var q = normalized({recipient_js});"
        " var els = Array.prototype.slice.call(document.querySelectorAll('.chatlist a[data-peer-id]'));"
        " var m = els.find(function(el) {"
        "   var t = el.querySelector('.user-title, .peer-title');"
        "   return t && normalized(t.textContent) === q; });"
        " if (m) return 'PEER:' + m.getAttribute('data-peer-id');"
        " var titles = els.map(function(el) {"
        "   var t = el.querySelector('.user-title, .peer-title');"
        "   return t ? t.textContent.trim() : ''; }).filter(Boolean).slice(0, 12);"
        " return 'MISS:' + titles.join(' | ');"
        "})()"
    )
    found = _chrome_js(find_js)
    if not found.startswith("PEER:"):
        titles = found[5:] if found.startswith("MISS:") else ""
        raise TelegramWebSendError(
            "telegram_chat_not_found",
            f"No Telegram chat titled '{recipient}'. Chats I can see: {titles}. Use the exact chat title.",
        )
    peer_id = found[5:].strip()

    # 3. Open the chat via the hash router — honored on page boot, so reload.
    _chrome_js(f"location.hash = '#{peer_id}'; location.reload(); 'reloading'")

    # 4. Wait for the chat to mount, then VERIFY the header IS the recipient.
    header = ""
    for _ in range(15):
        time.sleep(1)
        try:
            state = _chrome_js(
                "(function() { var h = document.querySelector('#column-center .peer-title');"
                " var c = document.querySelector('.input-message-input[contenteditable=\"true\"]');"
                " if (!h || !c) return 'PENDING'; return 'HEADER:' + h.textContent.trim(); })()"
            )
        except TelegramWebSendError:
            continue  # the JS bridge can error mid-reload; keep polling
        if state.startswith("HEADER:"):
            header = state[7:].strip()
            break
    if not header:
        raise TelegramWebSendError(
            "telegram_chat_not_opened",
            f"The chat for '{recipient}' never finished opening, so nothing was sent.",
        )
    if _normalized_recipient_identity(header) != _normalized_recipient_identity(recipient):
        raise TelegramWebSendError(
            "telegram_wrong_chat",
            f"The opened chat is '{header}', not '{recipient}' — stopped before sending.",
        )
    return header


def _place_telegram_send(recipient: str, message: str) -> str:
    message_js = json.dumps(message, ensure_ascii=False)
    header = _open_telegram_chat(recipient)

    # 5. Stage the message via JS (Unicode-safe; no keystroke typing) and confirm
    #    Telegram armed its send button.
    set_js = (
        "(function() {"
        " var c = document.querySelector('.input-message-input[contenteditable=\"true\"]');"
        " if (!c) return 'NO_COMPOSER';"
        " c.focus();"
        f" c.textContent = {message_js};"
        " c.dispatchEvent(new InputEvent('input', {bubbles: true}));"
        " var b = document.querySelector('.btn-send');"
        " return (b && b.classList.contains('send')) ? 'ARMED' : 'NOT_READY';"
        "})()"
    )
    armed = _chrome_js(set_js)
    if armed != "ARMED":
        raise TelegramWebSendError(
            "telegram_compose_failed",
            f"Could not stage the message in the verified chat ({armed}), so nothing was sent.",
        )

    # 6. Before the trusted keystroke: confirm the Telegram tab actually has
    #    system keyboard focus AND the composer is the active element — a real
    #    Enter goes to whatever is focused system-wide, so fail closed here.
    focused = _chrome_js(
        "(function() { var c = document.querySelector('.input-message-input[contenteditable=\"true\"]');"
        " if (c) c.focus();"
        " if (!document.hasFocus()) return 'WINDOW_UNFOCUSED';"
        " return (document.activeElement === c) ? 'COMPOSER_FOCUSED' : 'WRONG_ELEMENT'; })()"
    )
    if focused != "COMPOSER_FOCUSED":
        _focus_telegram_tab()
        time.sleep(0.8)
        focused = _chrome_js(
            "(function() { var c = document.querySelector('.input-message-input[contenteditable=\"true\"]');"
            " if (c) c.focus();"
            " if (!document.hasFocus()) return 'WINDOW_UNFOCUSED';"
            " return (document.activeElement === c) ? 'COMPOSER_FOCUSED' : 'WRONG_ELEMENT'; })()"
        )
    if focused != "COMPOSER_FOCUSED":
        raise TelegramWebSendError(
            "telegram_window_not_focused",
            f"The message is staged but the Telegram window does not have keyboard focus ({focused}) — "
            "refusing to press Enter into the wrong window.",
        )

    # 7. The one trusted keystroke sends; then confirm the composer cleared.
    try:
        _trusted_enter()
    except Exception as exc:
        raise TelegramWebSendError(
            "telegram_send_unconfirmed",
            "The trusted Enter action completed without a delivery acknowledgement.",
            outcome_unknown=True,
        ) from exc
    for _ in range(8):
        time.sleep(0.5)
        try:
            check = _chrome_js(
                "(function() { var c = document.querySelector('.input-message-input[contenteditable=\"true\"]');"
                " return (c && c.textContent.trim() === '') ? 'CLEARED' : 'STILL_TYPED'; })()"
            )
        except Exception as exc:
            raise TelegramWebSendError(
                "telegram_send_unconfirmed",
                "Telegram delivery confirmation failed after the trusted Enter action.",
                outcome_unknown=True,
            ) from exc
        if check == "CLEARED":
            return header
    raise TelegramWebSendError(
        "telegram_enter_did_not_send",
        "The message was staged in the right chat but delivery could not be confirmed.",
        outcome_unknown=True,
    )


def _trusted_click_at(x: int, y: int) -> None:
    """One real click at absolute screen coordinates (System Events; trusted)."""
    result = subprocess.run(
        ["osascript", "-e", f'tell application "System Events" to click at {{{x}, {y}}}'],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or "could not click"
        stage, detail = _telegram_error_stage(stderr)
        raise TelegramWebSendError(stage, detail)


_INSTAGRAM_CALL_SURFACE_JS = r"""(function() {
    var labels = [
        'outgoing call', 'calling', 'ringing', 'connecting',
        'end call', 'end video call', 'leave call', 'hang up',
        '통화 중', '전화 거는 중', '연결 중',
        '통화 종료', '영상 통화 종료', '통화 나가기', '전화 끊기'
    ];
    var nodes = Array.prototype.slice.call(document.querySelectorAll(
        'button,[role="button"],[role="heading"],[aria-label],[title]'
    ));
    var visible = nodes.some(function(x) {
        var r = x.getBoundingClientRect();
        if (r.width <= 0 || r.height <= 0) return false;
        var label = (
            x.getAttribute('aria-label') || x.getAttribute('title') ||
            x.innerText || x.textContent || ''
        ).trim().toLowerCase();
        return labels.some(function(expected) {
            return label === expected || label.indexOf(expected) >= 0;
        });
    });
    if (visible) return 'CALL_SURFACE';
    var path = (location.pathname || '').toLowerCase();
    return /\/(?:call|calls)(?:\/|$)/.test(path) || path.indexOf('/direct/call/') === 0
        ? 'CALL_URL' : 'CLEAR';
})()"""


_CHROME_INSTAGRAM_CALL_SURFACE_WRAPPER = """on run argv
    set instagramTabs to 0
    set instagramWindows to 0
    set callSurfaces to 0
    set callUrlTabs to 0
    tell application "Google Chrome"
        repeat with w in windows
            set windowHasInstagram to false
            repeat with t in tabs of w
                if URL of t contains "instagram.com" then
                    set instagramTabs to instagramTabs + 1
                    set windowHasInstagram to true
                    set surfaceState to execute t javascript (item 1 of argv)
                    if surfaceState is "CALL_SURFACE" then set callSurfaces to callSurfaces + 1
                    if surfaceState is "CALL_URL" then set callUrlTabs to callUrlTabs + 1
                end if
            end repeat
            if windowHasInstagram then set instagramWindows to instagramWindows + 1
        end repeat
    end tell
    return (instagramTabs as text) & "," & (instagramWindows as text) & "," & (callSurfaces as text) & "," & (callUrlTabs as text)
end run"""


def _instagram_call_surface_snapshot(timeout: int = 10) -> tuple[int, int, int, int]:
    """Return bounded counts: Instagram tabs/windows and call-specific surfaces."""
    from jarvis_v2.tools.instagram_connector import InstagramWebError, _instagram_error_stage

    result = subprocess.run(
        ["osascript", "-e", _CHROME_INSTAGRAM_CALL_SURFACE_WRAPPER, _INSTAGRAM_CALL_SURFACE_JS],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or "could not inspect Instagram call windows"
        stage, detail = _instagram_error_stage(stderr)
        raise InstagramWebError(stage, detail)
    try:
        values = tuple(int(value) for value in result.stdout.strip().split(","))
    except (TypeError, ValueError) as exc:
        raise InstagramWebError("instagram_call_state_unavailable") from exc
    if len(values) != 4 or any(value < 0 for value in values):
        raise InstagramWebError("instagram_call_state_unavailable")
    return values


def _instagram_call_surface_started(
    before: tuple[int, int, int, int],
    after: tuple[int, int, int, int],
) -> bool:
    # Raw Instagram tab/window growth is not proof: only a new call-specific URL
    # or outgoing/ringing/end-call UI can confirm the requested call surface.
    return after[2] > before[2] or after[3] > before[3]


def _wait_for_instagram_call_surface(
    before: tuple[int, int, int, int],
    *,
    attempts: int = 6,
    delay: float = 0.5,
) -> bool:
    for _ in range(attempts):
        after = _instagram_call_surface_snapshot()
        if _instagram_call_surface_started(before, after):
            return True
        time.sleep(delay)
    return False


def _trusted_instagram_click_at(x: int, y: int) -> None:
    """Issue one trusted click after the exact thread/button is revalidated."""
    from jarvis_v2.tools.instagram_connector import InstagramWebError, _instagram_error_stage

    result = subprocess.run(
        [
            "osascript",
            "-e",
            f'tell application "Google Chrome" to activate\n'
            'delay 0.15\n'
            f'tell application "System Events" to click at {{{x}, {y}}}',
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or "could not click the Instagram call button"
        stage, detail = _instagram_error_stage(stderr)
        raise InstagramWebError(stage, detail)


def _place_telegram_call(recipient: str, mode: str) -> str:
    """Start a Telegram voice/video call via the chat's ⋮ menu.

    Verified live: the ⋮ menu toggle ACCEPTS synthetic clicks (unlike Telegram's
    send path), and the open menu lists text-labeled items 'Call' / 'Video Call'.
    The item click is layered: synthetic first; if the menu stays open (gated),
    one trusted System Events click at the item's real screen coordinates.
    """
    label = "Video Call" if mode == "video" else "Call"
    _open_telegram_chat(recipient)

    toggled = _chrome_js(
        "(function() {"
        " var el = document.querySelector('#column-center .chat-utils .btn-menu-toggle:not(.hide)');"
        " if (!el) return 'NO_TOGGLE';"
        " var rect = el.getBoundingClientRect();"
        " var o = {bubbles: true, cancelable: true, view: window,"
        "  clientX: rect.x + rect.width/2, clientY: rect.y + rect.height/2, button: 0};"
        " ['pointerdown','mousedown','pointerup','mouseup','click'].forEach(function(t) {"
        "  el.dispatchEvent(t.indexOf('pointer') === 0 ? new PointerEvent(t, o) : new MouseEvent(t, o)); });"
        " return 'TOGGLED'; })()"
    )
    if toggled != "TOGGLED":
        raise TelegramWebSendError("telegram_call_menu_missing", "The chat menu button was not found.")

    # Find the call item, synthetically click it, and report its SCREEN coords
    # for the trusted-click fallback.
    click_item_js = (
        "(function() {"
        # Scope to the chat ⋮ menu only: prefer the menu nested in the toggle
        # itself; otherwise visible menus EXCLUDING the send-options and
        # playback menus (they sit in the DOM with nonzero width and polluted
        # the item scan in live testing).
        " var items = Array.prototype.slice.call(document.querySelectorAll("
        "  '#column-center .chat-utils .btn-menu-toggle .btn-menu-item'));"
        " if (!items.length) {"
        "  var menus = Array.prototype.slice.call(document.querySelectorAll('.btn-menu'))"
        "   .filter(function(m) { return m.getBoundingClientRect().width > 0"
        "    && !m.classList.contains('menu-send') && !m.classList.contains('playback-rate-menu'); });"
        "  menus.forEach(function(m) {"
        "   items = items.concat(Array.prototype.slice.call(m.querySelectorAll('.btn-menu-item'))); }); }"
        " if (!items.length) return 'MENU_NOT_OPEN';"
        f" var el = items.find(function(i) {{ return i.textContent.trim() === '{label}'; }});"
        " if (!el) return 'NO_CALL_ITEM:' + items.map(function(i) { return i.textContent.trim(); }).join(' | ');"
        " var rect = el.getBoundingClientRect();"
        " var sx = Math.round(window.screenX + rect.left + rect.width/2);"
        " var sy = Math.round(window.screenY + (window.outerHeight - window.innerHeight) + rect.top + rect.height/2);"
        " var o = {bubbles: true, cancelable: true, view: window,"
        "  clientX: rect.x + rect.width/2, clientY: rect.y + rect.height/2, button: 0};"
        " ['pointerdown','mousedown','pointerup','mouseup','click'].forEach(function(t) {"
        "  el.dispatchEvent(t.indexOf('pointer') === 0 ? new PointerEvent(t, o) : new MouseEvent(t, o)); });"
        " return 'CLICKED:' + sx + ',' + sy; })()"
    )
    clicked = ""
    for _ in range(6):
        time.sleep(0.5)
        try:
            clicked = _chrome_js(click_item_js)
        except Exception as exc:
            stage = (
                exc.stage
                if isinstance(exc, TelegramWebSendError)
                else "telegram_call_confirmation_failed"
            )
            raise TelegramWebSendError(
                stage,
                "Telegram call activation may have completed, but confirmation failed.",
                outcome_unknown=True,
            ) from exc
        if clicked != "MENU_NOT_OPEN":
            break
    if clicked.startswith("NO_CALL_ITEM:"):
        raise TelegramWebSendError(
            "telegram_call_item_missing",
            f"The chat menu has no '{label}' option (calls may not be available for this chat). Menu items: {clicked[13:]}",
        )
    if not clicked.startswith("CLICKED:"):
        raise TelegramWebSendError("telegram_call_menu_not_open", f"The chat menu never opened ({clicked}).")

    menu_open_js = (
        "(function() { var open = Array.prototype.slice.call(document.querySelectorAll('.btn-menu'))"
        " .some(function(m) { return m.getBoundingClientRect().width > 0 && m.textContent.indexOf('Call') !== -1; });"
        " return open ? 'OPEN' : 'CLOSED'; })()"
    )
    try:
        time.sleep(1)
        if _chrome_js(menu_open_js) == "OPEN":
            # Synthetic item click was gated — one trusted click at the real coords.
            sx, sy = (int(v) for v in clicked[8:].split(","))
            _trusted_click_at(sx, sy)
            time.sleep(1)
            if _chrome_js(menu_open_js) == "OPEN":
                raise TelegramWebSendError(
                    "telegram_call_click_failed",
                    f"The '{label}' menu item would not activate — the menu is still open.",
                    outcome_unknown=True,
                )
    except Exception as exc:
        if isinstance(exc, TelegramWebSendError) and exc.outcome_unknown:
            raise
        stage = (
            exc.stage
            if isinstance(exc, TelegramWebSendError)
            else "telegram_call_confirmation_failed"
        )
        raise TelegramWebSendError(
            stage,
            "Telegram call activation may have completed, but confirmation failed.",
            outcome_unknown=True,
        ) from exc
    return recipient


def _place_instagram_call(recipient: str, mode: str) -> str:
    """Request an Instagram call and verify a new call surface before success.

    Instagram's React handler accepts synthetic events, but Chrome can deny the
    call popup because those events have no trusted user activation. JavaScript
    therefore performs only read-only target/occlusion checks; one trusted
    System Events click is the sole activation, followed by read-only proof.
    """
    from jarvis_v2.tools.instagram_connector import (
        InstagramWebError,
        _chrome_js as _ig_js,
        open_instagram_thread,
    )

    label = "Video call" if mode == "video" else "Audio call"

    def fail(
        exc: Exception,
        *,
        outcome_unknown: bool,
        call_attempted: bool,
    ) -> InstagramCallError:
        stage = getattr(exc, "stage", "instagram_call_error")
        detail = getattr(exc, "detail", "")
        if isinstance(exc, subprocess.TimeoutExpired):
            stage = "instagram_call_confirmation_timeout" if outcome_unknown else "automation_timeout"
        return InstagramCallError(
            stage,
            detail,
            outcome_unknown=outcome_unknown,
            call_attempted=call_attempted,
        )

    def button_target() -> tuple[int, int]:
        target = _ig_js(
            "(function() {"
            f" var svg = document.querySelector('svg[aria-label=\"{label}\"]');"
            " if (!svg) return 'NO_CALL_BUTTON';"
            " var el = svg.closest('[role=\"button\"], button') || svg.parentElement;"
            " var rect = el.getBoundingClientRect();"
            " if (rect.width <= 0 || rect.height <= 0) return 'NO_CALL_BUTTON';"
            " var cx = rect.left + rect.width/2, cy = rect.top + rect.height/2;"
            " if (cx < 0 || cy < 0 || cx > window.innerWidth || cy > window.innerHeight)"
            "  return 'NO_CALL_BUTTON';"
            " var top = document.elementFromPoint(cx, cy);"
            " if (!top || !(top === el || el.contains(top))) return 'CALL_BUTTON_OCCLUDED';"
            " var sx = Math.round(window.screenX + rect.left + rect.width/2);"
            " var sy = Math.round(window.screenY + (window.outerHeight-window.innerHeight)"
            "  + rect.top + rect.height/2);"
            " return 'TARGET:' + sx + ',' + sy; })()"
        )
        if not target.startswith("TARGET:"):
            raise InstagramWebError(
                "instagram_call_button_missing",
                f"The verified thread did not expose its visible {label} button.",
            )
        try:
            coordinates = tuple(int(value) for value in target[7:].split(","))
        except (TypeError, ValueError) as exc:
            raise InstagramWebError("instagram_call_button_missing") from exc
        if len(coordinates) != 2 or any(abs(value) > 100_000 for value in coordinates):
            raise InstagramWebError("instagram_call_button_missing")
        return coordinates

    try:
        open_instagram_thread(recipient)
        baseline = _instagram_call_surface_snapshot()
        if baseline[2] > 0 or baseline[3] > 0:
            raise InstagramWebError("instagram_active_call")
        trusted_x, trusted_y = button_target()
    except Exception as exc:
        raise fail(exc, outcome_unknown=False, call_attempted=False) from exc

    try:
        _trusted_instagram_click_at(trusted_x, trusted_y)
        if _wait_for_instagram_call_surface(baseline):
            return recipient
    except Exception as exc:
        raise fail(exc, outcome_unknown=True, call_attempted=True) from exc

    raise InstagramCallError(
        "instagram_call_popup_blocked_or_missing",
        "No new Instagram call window or outgoing call controls appeared after the trusted click.",
        outcome_unknown=True,
        call_attempted=True,
    )


def _gui_call_failure(channel: str, exc: Exception, error_hint: str) -> tuple[str, str]:
    """Classify known GUI-call failures without exposing app/browser page detail."""
    from jarvis_v2.tools.kakao_connector import KakaoSendError, guard_message

    guard = guard_message(exc)
    if guard:
        return "guard_stopped", f"The {channel} call stopped safely: {guard}"
    if channel == "instagram":
        from jarvis_v2.tools.instagram_connector import InstagramWebError, _instagram_recovery_guidance

        if isinstance(exc, InstagramCallError):
            return exc.stage, exc.detail or error_hint
        if isinstance(exc, InstagramWebError):
            guidance = _instagram_recovery_guidance(exc.stage)
            if guidance:
                return exc.stage, f"The Instagram call stopped at {exc.stage}: {guidance}"
            return exc.stage, error_hint
    if channel == "kakao" and isinstance(exc, KakaoSendError):
        if exc.stage in {
            "macos_automation_permission",
            "macos_accessibility_permission",
            "kakaotalk_unavailable",
            "applescript_compile_failed",
        }:
            return exc.stage, f"The KakaoTalk call stopped at {exc.stage}: {exc.detail}"
        return exc.stage, error_hint
    if isinstance(exc, subprocess.TimeoutExpired):
        return "automation_timeout", f"The {channel} call automation timed out before Jarvis could confirm it; nothing was retried."
    return "call_error", error_hint


# --- Tool factory -------------------------------------------------------------


def make_call_tools(config: JarvisConfig):
    from jarvis_v2.tools.registry import (
        TOOL_ARGUMENT_CONTRACT_VERSION,
        Tool,
        ToolArgumentContract,
        ToolArgumentSpec,
        ToolArgumentType,
    )

    string_type = frozenset({ToolArgumentType.STRING})

    def _call_argument_contract(*names: str) -> ToolArgumentContract:
        """Fail closed on unrecognized or non-text call arguments.

        Recipient aliases remain optional so a missing-recipient request still reaches
        the existing user-facing clarification instead of becoming a schema error.
        The executor nevertheless rejects injected fields and non-string values before
        it can create an approval packet or invoke GUI/native call handling.
        """

        return ToolArgumentContract(
            version=TOOL_ARGUMENT_CONTRACT_VERSION,
            fields=tuple(
                ToolArgumentSpec(name, string_type, False)
                for name in names
            ),
            allow_unknown=False,
        )

    contact_call_contract = _call_argument_contract(
        "to", "recipient", "contact", "name", "mode", "kind", "type"
    )
    social_call_contract = _call_argument_contract(
        "to", "recipient", "contact", "name", "mode", "kind", "type"
    )
    telegram_call_contract = _call_argument_contract(
        "to", "recipient", "name", "mode"
    )

    def _require_recipient(tool_name: str, channel: str, recipient: str, mode: str) -> ToolResult | None:
        if recipient:
            return None
        return ToolResult(
            tool_name,
            False,
            f"Who should I call on {channel}? Give me a name" + (" or number." if channel != "instagram" else " or handle."),
            _safe_metadata(
                reason="missing_recipient",
                **_call_handoff(channel=channel, recipient="", mode=mode, status="refused", reason="missing_recipient"),
            ),
        )

    def call_contact(args: dict[str, Any]) -> ToolResult:
        recipient = _short(args.get("to") or args.get("recipient") or args.get("contact") or args.get("name"))
        mode = _normalize_mode(args.get("mode") or args.get("kind") or args.get("type"))
        missing = _require_recipient("call_contact", "FaceTime", recipient, mode)
        if missing:
            return missing
        handle, meta, refusal = _resolve_call_contact(recipient, mode)
        if not handle:
            return ToolResult(
                "call_contact",
                False,
                refusal,
                _safe_metadata(
                    reads_personal_data=bool(meta.get("contact_lookup_attempted")),
                    to=_display_text(recipient),
                    mode=mode,
                    reason=meta.get("contact_resolution_status"),
                    **_call_handoff(
                        channel="contact",
                        recipient=recipient,
                        mode=mode,
                        status="refused",
                        reason=str(meta.get("contact_resolution_status") or "contact_resolution_failed"),
                        contact_lookup_attempted=bool(meta.get("contact_lookup_attempted")),
                    ),
                ),
            )
        try:
            _place_native_call(mode, handle)
            label = {"video": "FaceTime video", "audio": "FaceTime audio", "phone": "phone"}[mode]
            return ToolResult(
                "call_contact",
                True,
                f"Requested a {label} call to {_display_text(recipient)}; confirm it is ringing.",
                _safe_metadata(
                    reads_personal_data=bool(meta.get("contact_lookup_attempted")),
                    executes_side_effect=True,
                    external_side_effect=True,
                    requires_approval=True,
                    controls_computer=True,
                    to=_display_text(recipient),
                    resolved_to=_display_text(handle),
                    mode=mode,
                    **_call_handoff(
                        channel="contact",
                        recipient=recipient,
                        mode=mode,
                        status="call_requested",
                        call_attempted=True,
                        controls_computer=True,
                        contact_lookup_attempted=bool(meta.get("contact_lookup_attempted")),
                        resolved_to=handle,
                    ),
                ),
            )
        except Exception as e:
            failure_output = _call_attempt_recovery_guidance("FaceTime/phone")
            return ToolResult(
                "call_contact",
                False,
                failure_output,
                _call_attempt_failure_metadata(
                    _safe_metadata(
                        executes_side_effect=True,
                        external_side_effect=True,
                        requires_approval=True,
                        controls_computer=True,
                        to=_display_text(recipient),
                        mode=mode,
                        exception_type=type(e).__name__,
                        **_call_handoff(
                            channel="contact",
                            recipient=recipient,
                            mode=mode,
                            status="outcome_unknown",
                            reason="call_outcome_unknown",
                            call_attempted=True,
                            controls_computer=True,
                        ),
                    ),
                    output=failure_output,
                ),
            )

    def _gui_call_tool(tool_name: str, channel: str, placer_name: str, error_hint: str):
        def handler(args: dict[str, Any]) -> ToolResult:
            recipient = _short(args.get("to") or args.get("recipient") or args.get("contact") or args.get("name"))
            mode = _normalize_mode(args.get("mode") or args.get("kind") or args.get("type"))
            if mode == "phone":  # these apps only do voice/video
                mode = "audio"
            missing = _require_recipient(tool_name, channel, recipient, mode)
            if missing:
                return missing
            try:
                # Look the placer up at call time (via module globals) so it stays
                # patchable in tests and swappable at runtime.
                placer = globals()[placer_name]
                placement_result = placer(recipient, mode)
                if channel == "kakao" and str(placement_result).strip().lower() != "true":
                    from jarvis_v2.tools.kakao_connector import KakaoSendError

                    raise KakaoSendError(
                        "kakao_call_button_missing",
                        "KakaoTalk did not confirm that the requested call button was clicked.",
                        clipboard_metadata=clipboard_safety.privacy_boundary_metadata(
                            snapshot_attempted=True,
                            snapshot_captured=True,
                            replacement_attempted=True,
                            text_restored=True,
                        ),
                    )
                label = "video" if mode == "video" else "voice"
                return ToolResult(
                    tool_name,
                    True,
                    f"Requested a {channel} {label} call to {_display_text(recipient)}; confirm it is ringing.",
                    _safe_metadata(
                        reads_personal_data=channel == "kakao",
                        reads_private_data=channel == "kakao",
                        reads_clipboard=channel == "kakao",
                        executes_side_effect=True,
                        external_side_effect=True,
                        requires_approval=True,
                        controls_computer=True,
                        to=_display_text(recipient),
                        mode=mode,
                        **(
                            clipboard_safety.privacy_boundary_metadata(
                                snapshot_attempted=True,
                                snapshot_captured=True,
                                replacement_attempted=True,
                                text_restored=True,
                            )
                            if channel == "kakao"
                            else {}
                        ),
                        **_call_handoff(
                            channel=channel,
                            recipient=recipient,
                            mode=mode,
                            status="call_requested",
                            call_attempted=True,
                            controls_computer=True,
                            reads_clipboard=channel == "kakao",
                        ),
                    ),
                )
            except Exception as e:
                failure_stage, classified_output = _gui_call_failure(channel, e, error_hint)
                display_channel = "KakaoTalk" if channel == "kakao" else "Instagram"
                known_not_started = (
                    channel == "instagram"
                    and isinstance(e, InstagramCallError)
                    and e.outcome_unknown is False
                    and e.call_attempted is False
                )
                call_attempted = (
                    e.call_attempted
                    if channel == "instagram" and isinstance(e, InstagramCallError)
                    else True
                )
                known_not_started_action = (
                    "Run `setup check`, correct the Instagram browser or permission issue, then submit a new approved call request."
                )
                failure_output = (
                    f"The Instagram call did not start at {failure_stage}. {known_not_started_action}"
                    if known_not_started
                    else _instagram_call_no_surface_recovery_guidance()
                    if channel == "instagram" and failure_stage == "instagram_call_popup_blocked_or_missing"
                    else _call_attempt_recovery_guidance(display_channel)
                )
                clipboard_metadata = (
                    dict(getattr(e, "clipboard_metadata", {}) or {})
                    if channel == "kakao"
                    else {}
                )
                failure_metadata = _safe_metadata(
                            reads_personal_data=channel == "kakao",
                            reads_private_data=channel == "kakao",
                            reads_clipboard=channel == "kakao",
                            guard_stopped=failure_stage == "guard_stopped",
                            executes_side_effect=call_attempted,
                            external_side_effect=call_attempted,
                            requires_approval=True,
                            controls_computer=True,
                            to=_display_text(recipient),
                            mode=mode,
                            failure_stage=failure_stage,
                            **{f"{channel}_call_stage": failure_stage},
                            exception_type=type(e).__name__,
                            **clipboard_metadata,
                            outcome_known=known_not_started,
                            outcome_unknown=not known_not_started,
                            execution_outcome_unknown=not known_not_started,
                            side_effect_possible=not known_not_started,
                            retry_safe=known_not_started,
                            automatic_retry_allowed=False,
                            authorizes_retry=False,
                            **_call_handoff(
                                channel=channel,
                                recipient=recipient,
                                mode=mode,
                                status="failed" if known_not_started else "outcome_unknown",
                                reason="call_not_started" if known_not_started else "call_outcome_unknown",
                                call_attempted=call_attempted,
                                controls_computer=True,
                                reads_clipboard=channel == "kakao",
                            ),
                        )
                if known_not_started:
                    failure_metadata = declare_failure_guidance(
                        failure_metadata,
                        output=failure_output,
                        action=known_not_started_action,
                        commands=("setup check",),
                    )
                    return ToolResult(tool_name, False, failure_output, failure_metadata)
                return ToolResult(
                    tool_name,
                    False,
                    failure_output,
                    _call_attempt_failure_metadata(
                        failure_metadata,
                        output=failure_output,
                    ),
                )

        return handler

    call_kakao = _gui_call_tool(
        "call_kakao",
        "kakao",
        "_place_kakao_call",
        "The KakaoTalk call could not be started. Check that KakaoTalk is installed, logged in, and Accessibility is granted.",
    )
    call_instagram = _gui_call_tool(
        "call_instagram",
        "instagram",
        "_place_instagram_call",
        "The Instagram call could not be started. Check that Chrome is logged into Instagram and Accessibility is granted.",
    )

    def call_telegram(args: dict[str, Any]) -> ToolResult:
        recipient = _short(args.get("to") or args.get("recipient") or args.get("name"))
        mode = _normalize_mode(args.get("mode"))
        if mode == "phone":  # Telegram only does voice/video
            mode = "audio"
        missing = _require_recipient("call_telegram", "Telegram", recipient, mode)
        if missing:
            return missing
        try:
            _place_telegram_call(recipient, mode)
            label = "video" if mode == "video" else "voice"
            return ToolResult(
                "call_telegram",
                True,
                f"Requested a Telegram {label} call to {_display_text(recipient)} via web.telegram.org; "
                "confirm it is ringing.",
                _safe_metadata(
                    executes_side_effect=True,
                    external_side_effect=True,
                    requires_approval=True,
                    controls_computer=True,
                    to=_display_text(recipient),
                    mode=mode,
                    **_call_handoff(
                        channel="telegram", recipient=recipient, mode=mode, status="call_requested",
                        call_attempted=True, controls_computer=True,
                    ),
                ),
            )
        except Exception as e:
            failure_stage, classified_output = _telegram_action_failure(
                e,
                label="The Telegram call",
                fallback="The Telegram call could not be started. Check that Chrome is logged in at web.telegram.org.",
            )
            known_not_started = (
                getattr(e, "outcome_unknown", False) is not True
                and failure_stage in {
                "guard_stopped",
                "macos_accessibility_permission",
                "chrome_javascript_permission",
                "telegram_tab_hidden",
                "telegram_not_loaded",
                "telegram_chat_not_found",
                "telegram_chat_not_opened",
                "telegram_wrong_chat",
                "telegram_call_menu_missing",
                "telegram_call_item_missing",
                "telegram_call_menu_not_open",
                }
            )
            failure_output = (
                classified_output
                if known_not_started
                else _call_attempt_recovery_guidance("Telegram")
            )
            failure_metadata = _safe_metadata(
                executes_side_effect=True,
                external_side_effect=True,
                requires_approval=True,
                controls_computer=True,
                to=_display_text(recipient),
                mode=mode,
                failure_stage=failure_stage,
                telegram_call_stage=failure_stage,
                error_type=type(e).__name__,
                exception_type=type(e).__name__,
                outcome_known=known_not_started,
                outcome_unknown=not known_not_started,
                side_effect_possible=not known_not_started,
                retry_safe=known_not_started,
                automatic_retry_allowed=False,
                authorizes_retry=False,
                **_call_handoff(
                    channel="telegram",
                    recipient=recipient,
                    mode=mode,
                    status="failed" if known_not_started else "outcome_unknown",
                    reason="call_error" if known_not_started else "call_outcome_unknown",
                    call_attempted=True,
                    controls_computer=True,
                ),
            )
            if not known_not_started:
                failure_metadata = _call_attempt_failure_metadata(
                    failure_metadata,
                    output=failure_output,
                )
            return ToolResult(
                "call_telegram",
                False,
                failure_output,
                failure_metadata,
            )

    def send_telegram(args: dict[str, Any]) -> ToolResult:
        recipient = _short(args.get("to") or args.get("recipient") or args.get("name"))
        message = _short(args.get("message") or args.get("text") or args.get("msg"), 500)
        if not recipient:
            return ToolResult(
                "send_telegram",
                False,
                "Who should I send a Telegram message to? Give me a name or username.",
                _safe_metadata(
                    reason="missing_recipient",
                    to=_display_text(recipient),
                    **_telegram_send_handoff(
                        recipient=recipient,
                        message_chars=len(message),
                        status="refused",
                        reason="missing_recipient",
                    ),
                ),
            )
        if not message:
            return ToolResult(
                "send_telegram",
                False,
                "What should I send? The message is empty.",
                _safe_metadata(
                    reason="missing_message",
                    to=_display_text(recipient),
                    **_telegram_send_handoff(
                        recipient=recipient,
                        message_chars=0,
                        status="refused",
                        reason="missing_message",
                    ),
                ),
            )
        owner_self = _is_owner_self_recipient(recipient)
        try:
            if owner_self:
                _send_owner_telegram(message)
            else:
                _place_telegram_send(recipient, message)
            return ToolResult(
                "send_telegram",
                True,
                (
                    "Telegram Bot API accepted the owner-chat send; recipient delivery is not confirmed."
                    if owner_self
                    else (
                        f"Telegram Web showed the outgoing message for {_display_text(recipient)}; "
                        "recipient delivery is not confirmed."
                    )
                ),
                _safe_metadata(
                    executes_side_effect=True,
                    external_side_effect=True,
                    requires_approval=True,
                    calls_external_service=owner_self,
                    controls_computer=not owner_self,
                    to=_display_text(recipient),
                    message_chars=len(message),
                    telegram_delivery_path="owner_bot_api" if owner_self else "telegram_web",
                    send_attempted=True,
                    sender_side_send_confirmed=True,
                    recipient_delivery_confirmed=False,
                    delivery_confirmed=False,
                    delivery_evidence="sender_side_only",
                    recipient_confirmation_required=True,
                    **_telegram_send_handoff(
                        recipient=recipient,
                        message_chars=len(message),
                        status="sender_confirmed",
                        owner_self=owner_self,
                        send_attempted=True,
                    ),
                ),
            )
        except Exception as e:
            label = "Owner Telegram send" if owner_self else "Telegram Web send"
            fallback = (
                "Owner Telegram send could not complete. Check the bot connection and owner configuration."
                if owner_self
                else "Telegram Web send could not complete. Check that Chrome is installed and you're logged into Telegram Web."
            )
            failure_stage, classified_output = _telegram_action_failure(
                e,
                label=label,
                fallback=fallback,
            )
            typed_delivery_error = isinstance(
                e,
                (TelegramOwnerSendError, TelegramWebSendError),
            )
            outcome_unknown = (
                getattr(e, "outcome_unknown", False) is True
                or (not typed_delivery_error and failure_stage != "guard_stopped")
            )
            failure_output = (
                _post_attempt_telegram_send_recovery_guidance(owner_self)
                if outcome_unknown
                else f"{classified_output} {KNOWN_NOT_SENT_SEND_RECOVERY_ACTION}"
            )
            failure_metadata = _safe_metadata(
                executes_side_effect=True,
                external_side_effect=True,
                requires_approval=True,
                calls_external_service=owner_self,
                controls_computer=not owner_self,
                guard_stopped=failure_stage == "guard_stopped",
                to=_display_text(recipient),
                message_chars=len(message),
                send_attempted=True,
                failure_stage=failure_stage,
                telegram_send_stage=failure_stage,
                telegram_delivery_path="owner_bot_api" if owner_self else "telegram_web",
                error_type=type(e).__name__,
                exception_type=type(e).__name__,
                outcome_known=not outcome_unknown,
                outcome_unknown=outcome_unknown,
                side_effect_possible=outcome_unknown,
                retry_safe=not outcome_unknown,
                automatic_retry_allowed=False,
                authorizes_retry=False,
                **_telegram_send_handoff(
                    recipient=recipient,
                    message_chars=len(message),
                    status="outcome_unknown" if outcome_unknown else "failed",
                    reason="send_outcome_unknown" if outcome_unknown else "send_error",
                    owner_self=owner_self,
                    send_attempted=True,
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
                "send_telegram",
                False,
                failure_output,
                failure_metadata,
            )

    return [
        Tool(
            "call_contact",
            "Call a person via FaceTime video, FaceTime audio, or phone (through your iPhone). "
            "Args: to (name or number), mode (video|audio|phone). Resolves the name in Contacts; approval-gated.",
            RiskLevel.HIGH_RISK,
            call_contact,
            "personal",
            argument_contract=contact_call_contract,
        ),
        Tool(
            "call_kakao",
            "Start a KakaoTalk voice or video call with a friend by name. Args: to, mode (audio|video). "
            "Best-effort GUI automation; approval-gated.",
            RiskLevel.HIGH_RISK,
            call_kakao,
            "personal",
            argument_contract=social_call_contract,
        ),
        Tool(
            "call_instagram",
            "Start an Instagram voice or video call with someone by handle. Args: to, mode (audio|video). "
            "Best-effort GUI automation through Chrome; approval-gated.",
            RiskLevel.HIGH_RISK,
            call_instagram,
            "personal",
            argument_contract=social_call_contract,
        ),
        Tool(
            "call_telegram",
            "Start a Telegram voice or video call via Chrome at web.telegram.org (chat menu → Call). "
            "Args: to (exact chat title), mode (audio|video). Best-effort GUI automation; approval-gated.",
            RiskLevel.HIGH_RISK,
            call_telegram,
            "personal",
            argument_contract=telegram_call_contract,
        ),
        Tool(
            "send_telegram",
            "Send a Telegram message to a person via Chrome to web.telegram.org. Args: to (name or username), message. Best-effort GUI automation; approval-gated.",
            RiskLevel.HIGH_RISK,
            send_telegram,
            "personal",
        ),
    ]
