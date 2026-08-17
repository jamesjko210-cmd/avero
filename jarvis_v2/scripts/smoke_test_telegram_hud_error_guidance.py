"""Focused offline proof for Telegram and HUD public failure guidance."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from jarvis_v2.automations import telegram_control as telegram
from jarvis_v2.scripts.hud import HudCore


class _Unreadable:
    def __str__(self) -> str:
        raise RuntimeError("SHOULD NOT APPEAR /\x55sers/example/private/input.txt")


def _bridge_for(update: dict, sent: list[tuple[str, str, object]]) -> telegram.TelegramCommandBridge:
    return telegram.TelegramCommandBridge(
        runtime_factory=lambda: SimpleNamespace(),
        send_func=lambda chat_id, text, markup=None: sent.append((chat_id, text, markup)) or {"ok": True},
        fetch_func=lambda _token, _offset, _timeout: [update],
        answer_callback_func=lambda _callback_id, _text: {"ok": True},
        edit_message_func=lambda _chat_id, _message_id, _text, _markup=None: {"ok": True},
        chat_action_func=lambda _chat_id, _action: {"ok": True},
    )


def _prepare_polling() -> None:
    os.environ["TELEGRAM_BOT_TOKEN"] = "offline:test-token"
    os.environ["JARVIS_OWNER_TELEGRAM"] = "555001"
    telegram.STATE_FILE = Path(tempfile.mkdtemp()) / "telegram-offset.json"
    telegram._save_offset(0)


def test_unreadable_voice_error_has_bounded_recovery() -> None:
    _prepare_polling()
    sent: list[tuple[str, str, object]] = []
    update = {
        "update_id": 1,
        "message": {"chat": {"id": 555001}, "voice": {"file_id": "voice-id"}},
    }
    original = telegram.transcribe_voice_message
    try:
        telegram.transcribe_voice_message = lambda _token, _message: ("", _Unreadable())
        processed = _bridge_for(update, sent).process_once(poll_timeout=0)
    finally:
        telegram.transcribe_voice_message = original
    if processed != 1 or len(sent) != 1:
        raise SystemExit(f"unreadable voice result was not handled once: {processed} {sent}")
    surfaced = sent[0][1]
    for expected in ("couldn't transcribe", "`voice setup check`", "type the command instead"):
        if expected not in surfaced:
            raise SystemExit(f"voice recovery missed {expected!r}: {surfaced!r}")
    if "SHOULD NOT APPEAR" in surfaced or "/\x55sers/" in surfaced:
        raise SystemExit(f"voice recovery leaked private input: {surfaced!r}")


def test_unreadable_ocr_error_suppresses_private_caption() -> None:
    _prepare_polling()
    sent: list[tuple[str, str, object]] = []
    update = {
        "update_id": 2,
        "message": {
            "chat": {"id": 555001},
            "photo": [{"file_id": "image-id", "file_size": 10}],
            "caption": "PRIVATE CAPTION",
        },
    }
    original = telegram.extract_photo_text
    try:
        telegram.extract_photo_text = lambda _token, _message: ("", _Unreadable())
        processed = _bridge_for(update, sent).process_once(poll_timeout=0)
    finally:
        telegram.extract_photo_text = original
    if processed != 1 or len(sent) != 1:
        raise SystemExit(f"unreadable OCR result was not handled once: {processed} {sent}")
    surfaced = sent[0][1]
    for expected in ("couldn't read text", "`setup check`", "type the text instead"):
        if expected not in surfaced:
            raise SystemExit(f"OCR recovery missed {expected!r}: {surfaced!r}")
    for forbidden in ("PRIVATE CAPTION", "SHOULD NOT APPEAR", "/\x55sers/"):
        if forbidden in surfaced:
            raise SystemExit(f"OCR failure leaked {forbidden!r}: {surfaced!r}")


def test_callback_toast_is_sanitized_and_refusals_name_recovery() -> None:
    private_path = "/\x55sers/example/private/callback.txt"

    class _Runtime:
        def handle(self, _text: str):
            return SimpleNamespace(response=f"Approval result at {private_path}", tool_results=[])

    answered: list[str] = []
    edited: list[str] = []
    bridge = telegram.TelegramCommandBridge(
        runtime_factory=_Runtime,
        send_func=lambda _chat_id, _text, _markup=None: {"ok": True},
        fetch_func=lambda _token, _offset, _timeout: [],
        answer_callback_func=lambda _callback_id, text: answered.append(text) or {"ok": True},
        edit_message_func=lambda _chat_id, _message_id, text, _markup=None: edited.append(text) or {"ok": True},
        chat_action_func=lambda _chat_id, _action: {"ok": True},
    )
    callback = {
        "id": "callback-1",
        "from": {"id": 555001},
        "data": "approve:7",
        "message": {"message_id": 9, "chat": {"id": 555001}},
    }
    if not bridge._process_callback(callback, "555001") or not answered or not edited:
        raise SystemExit(f"valid callback was not surfaced: {answered} {edited}")
    combined = "\n".join(answered + edited)
    if "<local-path>" not in combined or private_path in combined or "/\x55sers/" in combined:
        raise SystemExit(f"callback egress was not sanitized: {combined!r}")

    for value in ("retry:7", "approve:not-a-number", "deny:0"):
        refusal = bridge._handle_approval_callback(value)
        for expected in ("approval", "fresh packet", "Approve or Deny"):
            if expected not in refusal:
                raise SystemExit(f"callback refusal missed {expected!r}: {value!r} -> {refusal!r}")


def test_hud_failures_use_stable_private_safe_codes_and_recovery() -> None:
    init_reply = HudCore(
        runtime_factory=lambda: (_ for _ in ()).throw(
            RuntimeError("SHOULD NOT APPEAR /\x55sers/example/private/init.sqlite")
        )
    ).submit("hello")
    if init_reply.error != "runtime_initialization_failed":
        raise SystemExit(f"HUD initialization code drifted: {init_reply}")
    for expected in ("bootstrap_memory --check", "repair", "retry"):
        if expected not in init_reply.response:
            raise SystemExit(f"HUD initialization recovery missed {expected!r}: {init_reply}")

    class _FailingRuntime:
        def handle(self, _text: str):
            raise RuntimeError("SHOULD NOT APPEAR /\x55sers/example/private/command.txt")

    command_reply = HudCore(runtime_factory=_FailingRuntime).submit("risky command")
    if command_reply.error != "runtime_command_outcome_unknown":
        raise SystemExit(f"HUD command code drifted: {command_reply}")
    for expected in ("Do not retry automatically", "execution health report", "approval chain"):
        if expected not in command_reply.response:
            raise SystemExit(f"HUD command recovery missed {expected!r}: {command_reply}")
    surfaced = f"{init_reply.response}\n{init_reply.error}\n{command_reply.response}\n{command_reply.error}"
    for forbidden in ("SHOULD NOT APPEAR", "/\x55sers/", "RuntimeError"):
        if forbidden in surfaced:
            raise SystemExit(f"HUD failure leaked {forbidden!r}: {surfaced!r}")


def main() -> None:
    test_unreadable_voice_error_has_bounded_recovery()
    test_unreadable_ocr_error_suppresses_private_caption()
    test_callback_toast_is_sanitized_and_refusals_name_recovery()
    test_hud_failures_use_stable_private_safe_codes_and_recovery()
    print("Telegram/HUD error guidance smoke passed")


if __name__ == "__main__":
    main()
