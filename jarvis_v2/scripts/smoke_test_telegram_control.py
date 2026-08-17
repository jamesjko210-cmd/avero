"""Smoke tests for the Jarvis V2 owner-locked Telegram control bridge.

Uses fake fetch/send functions so it runs without a real bot token or network.
"""

from __future__ import annotations

import os
import pwd
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from jarvis_v2.automations import telegram_control as tc
from jarvis_v2.automations import telegram_offset_state
from jarvis_v2.automations import scheduler as scheduler_module
from jarvis_v2.memory import obsidian as obsidian_module
from jarvis_v2.memory import store as store_module
from jarvis_v2.scripts import run_telegram_control as telegram_launcher
from jarvis_v2.scripts.daemon_gate import V3_DAEMON_ENABLE_ENV, V3_SCHEDULER_ENABLE_ENV
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools import call_connector as call_connector_module


class FakeRuntime:
    def __init__(self, replies):
        self.replies = list(replies)
        self.inputs = []
        self.request_tokens = []

    def handle(self, text, request_token=None):
        self.inputs.append(text)
        self.request_tokens.append(request_token)
        reply = self.replies.pop(0) if self.replies else "ok"
        if isinstance(reply, tuple):
            response, approval_id = reply
            return SimpleNamespace(
                response=response,
                tool_results=[
                    SimpleNamespace(
                        metadata={"requires_confirmation": True, "approval_id": approval_id}
                    )
                ],
            )
        return SimpleNamespace(response=reply, tool_results=[])


class FakeRuntimeTrace:
    def __init__(self, result):
        self.result = result
        self.inputs = []
        self.request_tokens = []

    def handle(self, text, request_token=None):
        self.inputs.append(text)
        self.request_tokens.append(request_token)
        return self.result


def _fresh_state() -> None:
    tc.STATE_FILE = Path(tempfile.mkdtemp()) / "telegram.json"


def _seed_offset(value: int) -> None:
    _fresh_state()
    _setup_env()
    tc._save_offset(value)


def _update(update_id: int, chat_id, text: str) -> dict:
    return {"update_id": update_id, "message": {"chat": {"id": chat_id}, "text": text}}


def _callback_update(update_id: int, user_id, data: str, message_id: int = 90, chat_id=None) -> dict:
    if chat_id is None:
        chat_id = user_id
    return {
        "update_id": update_id,
        "callback_query": {
            "id": f"callback-{update_id}",
            "from": {"id": user_id},
            "data": data,
            "message": {"message_id": message_id, "chat": {"id": chat_id}},
        },
    }


def _setup_env(owner: str = "555001") -> None:
    os.environ["TELEGRAM_BOT_TOKEN"] = "test:token"
    os.environ["JARVIS_OWNER_TELEGRAM"] = owner


def _no_chat_action(chat_id: str, action: str) -> dict:
    return {"ok": True}


class _StopTicker(BaseException):
    pass


class _SynchronousThread:
    def __init__(self, *, target, name, daemon):
        self.target = target
        self.name = name
        self.daemon = daemon

    def start(self) -> None:
        self.target()


def test_telegram_only_activation_starts_no_scheduler_or_jobs() -> None:
    events = []
    config = SimpleNamespace(model_provider="openai")

    class FakeBridge:
        def __init__(self):
            events.append("bridge.construct")

        def run_forever(self):
            events.append("bridge.run_forever")
            raise KeyboardInterrupt

    def forbidden_thread(*args, **kwargs):
        events.append("scheduler.thread")
        raise AssertionError("Telegram-only activation started the scheduler ticker")

    with (
        mock.patch.dict(
            os.environ,
            {V3_DAEMON_ENABLE_ENV: "1"},
            clear=False,
        ),
        mock.patch.object(telegram_launcher, "load_config", return_value=config),
        mock.patch.object(telegram_launcher, "_bot_token", return_value="synthetic-token"),
        mock.patch.object(telegram_launcher, "_owner_chat_id", return_value="555001"),
        mock.patch.object(
            telegram_launcher,
            "_load_offset_snapshot",
            side_effect=lambda *_args: events.append("state.validate"),
        ),
        mock.patch.object(telegram_launcher, "_warm_model", side_effect=lambda _: events.append("model.warm")),
        mock.patch.object(telegram_launcher, "TelegramCommandBridge", FakeBridge),
        mock.patch.object(threading, "Thread", forbidden_thread),
    ):
        os.environ.pop(V3_SCHEDULER_ENABLE_ENV, None)
        telegram_launcher.main()

    if events != ["state.validate", "model.warm", "bridge.construct", "bridge.run_forever"]:
        raise SystemExit(f"Telegram-only activation crossed scheduler custody: {events}")


def test_invalid_offset_custody_blocks_model_scheduler_and_bridge_startup() -> None:
    events: list[str] = []
    config = SimpleNamespace(model_provider="openai")

    with (
        mock.patch.dict(os.environ, {V3_DAEMON_ENABLE_ENV: "1"}, clear=False),
        mock.patch.object(telegram_launcher, "load_config", return_value=config),
        mock.patch.object(telegram_launcher, "_bot_token", return_value="synthetic-token"),
        mock.patch.object(telegram_launcher, "_owner_chat_id", return_value="555001"),
        mock.patch.object(
            telegram_launcher,
            "_load_offset_snapshot",
            side_effect=telegram_offset_state.TelegramOffsetStateError("state_malformed"),
        ),
        mock.patch.object(
            telegram_launcher,
            "_warm_model",
            side_effect=lambda *_args: events.append("model.warm"),
        ),
        mock.patch.object(
            telegram_launcher,
            "_start_scheduler_ticker",
            side_effect=lambda *_args: events.append("scheduler.start"),
        ),
        mock.patch.object(
            telegram_launcher,
            "TelegramCommandBridge",
            side_effect=lambda: events.append("bridge.construct"),
        ),
    ):
        try:
            telegram_launcher.main()
        except SystemExit as exc:
            if exc.code != 1:
                raise SystemExit("invalid offset custody returned the wrong startup status")
        else:
            raise SystemExit("invalid offset custody did not stop Telegram startup")

    if events:
        raise SystemExit(f"invalid offset custody allowed startup work: {events}")


def test_embedded_scheduler_initializes_dependencies_before_first_tick() -> None:
    events = []
    config = SimpleNamespace(db_path="fake.sqlite", obsidian_vault="fake-vault", obsidian_root="root")

    class FakeStore:
        def __init__(self, path):
            events.append(("store.construct", path))

        def init(self):
            events.append(("store.init",))

    class FakeVault:
        def __init__(self, vault, root):
            events.append(("vault.construct", vault, root))

        def init(self):
            events.append(("vault.init",))

    class FakeScheduler:
        def __init__(self, store, vault, received_config):
            events.append(("scheduler.construct",))

        def run_due_jobs(self):
            events.append(("scheduler.run_due_jobs",))
            raise _StopTicker

    with mock.patch.dict(
             os.environ,
             {V3_DAEMON_ENABLE_ENV: "1", V3_SCHEDULER_ENABLE_ENV: "1"},
         ), \
         mock.patch.object(threading, "Thread", _SynchronousThread), \
         mock.patch.object(store_module, "MemoryStore", FakeStore), \
         mock.patch.object(obsidian_module, "ObsidianVault", FakeVault), \
         mock.patch.object(scheduler_module, "Scheduler", FakeScheduler):
        try:
            telegram_launcher._start_scheduler_ticker(config)
        except _StopTicker:
            pass
        else:
            raise SystemExit("mock ticker did not reach its first due-job run")

    expected = [
        ("store.construct", "fake.sqlite"),
        ("vault.construct", "fake-vault", "root"),
        ("store.init",),
        ("vault.init",),
        ("scheduler.construct",),
        ("scheduler.run_due_jobs",),
    ]
    if events != expected:
        raise SystemExit(f"embedded scheduler startup order was wrong: {events}")


def test_embedded_scheduler_setup_failure_retries_then_recovers() -> None:
    events = []
    logs = []
    config = SimpleNamespace(db_path="fake.sqlite", obsidian_vault="fake-vault", obsidian_root="root")

    class FailFirstStore:
        attempts = 0

        def __init__(self, path):
            FailFirstStore.attempts += 1
            self.attempt = FailFirstStore.attempts
            events.append(("store.construct", self.attempt))

        def init(self):
            events.append(("store.init", self.attempt))
            if self.attempt == 1:
                raise RuntimeError("/\x55sers/example/private/setup SHOULD NOT APPEAR")

    class FakeVault:
        def __init__(self, vault, root):
            events.append(("vault.construct", FailFirstStore.attempts))

        def init(self):
            events.append(("vault.init", FailFirstStore.attempts))

    class FakeScheduler:
        def __init__(self, store, vault, received_config):
            events.append(("scheduler.construct", FailFirstStore.attempts))

        def run_due_jobs(self):
            events.append(("scheduler.run_due_jobs", FailFirstStore.attempts))
            raise _StopTicker

    def fake_sleep(seconds):
        events.append(("sleep", seconds))

    def capture_print(*parts, **_kwargs):
        logs.append(" ".join(str(part) for part in parts))

    with mock.patch.dict(
             os.environ,
             {V3_DAEMON_ENABLE_ENV: "1", V3_SCHEDULER_ENABLE_ENV: "1"},
         ), \
         mock.patch.object(threading, "Thread", _SynchronousThread), \
         mock.patch.object(store_module, "MemoryStore", FailFirstStore), \
         mock.patch.object(obsidian_module, "ObsidianVault", FakeVault), \
         mock.patch.object(scheduler_module, "Scheduler", FakeScheduler), \
         mock.patch("time.sleep", fake_sleep), \
         mock.patch("builtins.print", capture_print):
        try:
            telegram_launcher._start_scheduler_ticker(config)
        except _StopTicker:
            pass
        else:
            raise SystemExit("recovered scheduler ticker did not reach its first due-job run")

    expected = [
        ("store.construct", 1),
        ("vault.construct", 1),
        ("store.init", 1),
        ("sleep", 60),
        ("store.construct", 2),
        ("vault.construct", 2),
        ("store.init", 2),
        ("vault.init", 2),
        ("scheduler.construct", 2),
        ("scheduler.run_due_jobs", 2),
    ]
    if events != expected:
        raise SystemExit(f"scheduler setup retry order was wrong: {events}")
    expected_logs = [
        "Scheduler ticker initialization unavailable; retrying in 60 seconds. If this persists, run `setup check` and repair local storage access.",
        "Scheduler ticker initialization recovered.",
        "Scheduler ticker running (checks due jobs every 60s).",
    ]
    if logs != expected_logs:
        raise SystemExit(f"scheduler setup retry transitions were wrong: {logs}")
    if any(
        forbidden in "\n".join(logs)
        for forbidden in ("/\x55sers/operator", "private", "SHOULD NOT APPEAR", "RuntimeError")
    ):
        raise SystemExit(f"scheduler setup retry leaked exception content: {logs}")


def test_embedded_scheduler_tick_failure_retries_then_recovers() -> None:
    events = []
    logs = []
    config = SimpleNamespace(db_path="fake.sqlite", obsidian_vault="fake-vault", obsidian_root="root")

    class FakeStore:
        def __init__(self, path):
            events.append("store.construct")

        def init(self):
            events.append("store.init")

    class FakeVault:
        def __init__(self, vault, root):
            events.append("vault.construct")

        def init(self):
            events.append("vault.init")

    class FailOnceScheduler:
        ticks = 0

        def __init__(self, store, vault, received_config):
            events.append("scheduler.construct")

        def run_due_jobs(self):
            FailOnceScheduler.ticks += 1
            events.append(("scheduler.run_due_jobs", FailOnceScheduler.ticks))
            if FailOnceScheduler.ticks == 1:
                raise RuntimeError("/\x55sers/example/private/tick SHOULD NOT APPEAR")
            if FailOnceScheduler.ticks == 2:
                return "No jobs due."
            raise _StopTicker

    def fake_sleep(seconds):
        events.append(("sleep", seconds))

    def capture_print(*parts, **_kwargs):
        logs.append(" ".join(str(part) for part in parts))

    with mock.patch.dict(
             os.environ,
             {V3_DAEMON_ENABLE_ENV: "1", V3_SCHEDULER_ENABLE_ENV: "1"},
         ), \
         mock.patch.object(threading, "Thread", _SynchronousThread), \
         mock.patch.object(store_module, "MemoryStore", FakeStore), \
         mock.patch.object(obsidian_module, "ObsidianVault", FakeVault), \
         mock.patch.object(scheduler_module, "Scheduler", FailOnceScheduler), \
         mock.patch("time.sleep", fake_sleep), \
         mock.patch("builtins.print", capture_print):
        try:
            telegram_launcher._start_scheduler_ticker(config)
        except _StopTicker:
            pass
        else:
            raise SystemExit("scheduler ticker did not continue after a tick failure")

    expected = [
        "store.construct",
        "vault.construct",
        "store.init",
        "vault.init",
        "scheduler.construct",
        ("scheduler.run_due_jobs", 1),
        ("sleep", 60),
        ("scheduler.run_due_jobs", 2),
        ("sleep", 60),
        ("scheduler.run_due_jobs", 3),
    ]
    if events != expected:
        raise SystemExit(f"scheduler tick retry order was wrong: {events}")
    expected_logs = [
        "Scheduler ticker running (checks due jobs every 60s).",
        "Scheduler ticker tick unavailable; retrying in 60 seconds. If this persists, run `list scheduled jobs` and `setup check`, then repair the reported scheduler or storage issue.",
        "Scheduler ticker tick recovered.",
    ]
    if logs != expected_logs:
        raise SystemExit(f"scheduler tick retry transitions were wrong: {logs}")
    if any(
        forbidden in "\n".join(logs)
        for forbidden in ("/\x55sers/operator", "private", "SHOULD NOT APPEAR", "RuntimeError")
    ):
        raise SystemExit(f"scheduler tick retry leaked exception content: {logs}")


def test_owner_command_runs_and_offset_advances() -> None:
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(["It is 2 PM."])
    sent = []
    fetched = {"called": False}

    def fake_fetch(token, offset, timeout):
        if fetched["called"]:
            return []
        fetched["called"] = True
        return [_update(10, 555001, "what time is it")]

    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=fake_fetch,
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 1:
        raise SystemExit("owner command was not processed")
    if bridge.process_once(poll_timeout=0) != 0:
        raise SystemExit("offset did not advance; update reprocessed")
    if rt.inputs != ["what time is it"]:
        raise SystemExit(f"command not passed through: {rt.inputs}")
    if rt.request_tokens != ["telegram-update:v1:10"]:
        raise SystemExit(f"owner command missed its stable update token: {rt.request_tokens}")
    if sent != [("555001", "It is 2 PM.", None)]:
        raise SystemExit(f"reply not sent to owner: {sent}")


def _voice_update(update_id: int, chat_id, file_id: str = "vfile", file_size: int = 2048) -> dict:
    return {
        "update_id": update_id,
        "message": {"chat": {"id": chat_id}, "voice": {"file_id": file_id, "file_size": file_size}},
    }


def test_voice_memo_is_transcribed_and_routed() -> None:
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(["It is 2 PM."])
    sent = []
    original = tc.transcribe_voice_message
    tc.transcribe_voice_message = lambda token, message: ("what time is it", "")
    try:
        bridge = tc.TelegramCommandBridge(
            runtime_factory=lambda: rt,
            send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
            fetch_func=lambda token, offset, timeout: [_voice_update(10, 555001)],
            chat_action_func=_no_chat_action,
        )
        if bridge.process_once(poll_timeout=0) != 1:
            raise SystemExit("voice memo was not processed")
    finally:
        tc.transcribe_voice_message = original
    if rt.inputs != ["what time is it"]:
        raise SystemExit(f"transcript not routed as a command: {rt.inputs}")
    if rt.request_tokens != ["telegram-update:v1:10"]:
        raise SystemExit(f"voice transcript missed its stable update token: {rt.request_tokens}")
    if not any("🎙 Heard: what time is it" in t for _cid, t, _m in sent):
        raise SystemExit(f"transcript echo not sent: {sent}")
    if not any(t == "It is 2 PM." for _cid, t, _m in sent):
        raise SystemExit(f"command reply not sent: {sent}")


def test_voice_memo_korean_status_shortcut_routes_directly() -> None:
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(["jobs listed"])
    sent = []
    original = tc.transcribe_voice_message
    tc.transcribe_voice_message = lambda token, message: ("브리핑 확인", "")
    try:
        bridge = tc.TelegramCommandBridge(
            runtime_factory=lambda: rt,
            send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
            fetch_func=lambda token, offset, timeout: [_voice_update(11, 555001)],
            chat_action_func=_no_chat_action,
        )
        if bridge.process_once(poll_timeout=0) != 1:
            raise SystemExit("Korean status voice memo was not processed")
    finally:
        tc.transcribe_voice_message = original
    if rt.inputs != ["list scheduled jobs"]:
        raise SystemExit(f"Korean status transcript should route to the read-only shortcut: {rt.inputs}")
    if not any("🎙 Heard: 브리핑 확인" in t for _cid, t, _m in sent):
        raise SystemExit(f"Korean status transcript echo not sent: {sent}")
    if not any(t == "jobs listed" and m is None for _cid, t, m in sent):
        raise SystemExit(f"Korean status shortcut reply should be read-only without buttons: {sent}")


def test_voice_memo_preserves_bilingual_transcript_and_approval_buttons() -> None:
    _seed_offset(0)
    _setup_env("555001")
    transcript = "send 가상연락처이 a telegram saying hello 안녕하세요"
    rt = FakeRuntime([("Queued approval for send_telegram.", 42)])
    sent = []
    original = tc.transcribe_voice_message
    tc.transcribe_voice_message = lambda token, message: (transcript, "")
    try:
        bridge = tc.TelegramCommandBridge(
            runtime_factory=lambda: rt,
            send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
            fetch_func=lambda token, offset, timeout: [_voice_update(12, 555001)],
            chat_action_func=_no_chat_action,
        )
        if bridge.process_once(poll_timeout=0) != 1:
            raise SystemExit("bilingual voice memo was not processed")
    finally:
        tc.transcribe_voice_message = original
    if rt.inputs != [transcript]:
        raise SystemExit(f"bilingual transcript not routed exactly: {rt.inputs!r}")
    if not any(f"🎙 Heard: {transcript}" in t for _cid, t, _m in sent):
        raise SystemExit(f"bilingual transcript echo not sent intact: {sent}")
    approval_replies = [(t, m) for _cid, t, m in sent if "Queued approval" in t]
    if len(approval_replies) != 1:
        raise SystemExit(f"expected exactly one approval reply after bilingual voice command: {sent}")
    markup = approval_replies[0][1] or {}
    buttons = markup.get("inline_keyboard", [[]])[0]
    labels = [button.get("text") for button in buttons]
    callbacks = [button.get("callback_data") for button in buttons]
    if labels != ["✅ Approve", "❌ Deny"] or callbacks != ["approve:42", "deny:42"]:
        raise SystemExit(f"approval buttons not attached to bilingual voice command: {markup}")


def test_voice_transcription_failure_is_reported_without_routing() -> None:
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(["should not run"])
    sent = []
    original = tc.transcribe_voice_message
    tc.transcribe_voice_message = lambda token, message: ("", "I couldn't transcribe that voice note just now.")
    try:
        bridge = tc.TelegramCommandBridge(
            runtime_factory=lambda: rt,
            send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
            fetch_func=lambda token, offset, timeout: [_voice_update(13, 555001)],
            chat_action_func=_no_chat_action,
        )
        bridge.process_once(poll_timeout=0)
    finally:
        tc.transcribe_voice_message = original
    if rt.inputs:
        raise SystemExit(f"failed transcription must not route a command: {rt.inputs}")
    if not any("couldn't transcribe" in t for _cid, t, _m in sent):
        raise SystemExit(f"transcription error not reported: {sent}")


def test_media_failure_guidance_names_safe_recovery() -> None:
    voice_message = {"voice": {"file_id": "voice-id", "file_size": 128}}
    image_message = {
        "document": {
            "file_id": "image-id",
            "file_size": 128,
            "mime_type": "image/png",
        }
    }

    voice_errors = [tc.transcribe_voice_message("token", {})[1]]
    with mock.patch(
        "jarvis_v2.tools.voice._audio_file_transcriber",
        return_value=lambda _path: "unused",
    ), mock.patch.object(tc, "_download_telegram_file", return_value=None):
        voice_errors.append(tc.transcribe_voice_message("token", voice_message)[1])
    with mock.patch(
        "jarvis_v2.tools.voice._audio_file_transcriber",
        return_value=lambda _path: (_ for _ in ()).throw(
            RuntimeError("/\x55sers/example/private/voice SHOULD NOT APPEAR")
        ),
    ), mock.patch.object(tc, "_download_telegram_file", return_value=Path("voice.oga")):
        voice_errors.append(tc.transcribe_voice_message("token", voice_message)[1])
    with mock.patch(
        "jarvis_v2.tools.voice._audio_file_transcriber",
        return_value=lambda _path: "",
    ), mock.patch.object(tc, "_download_telegram_file", return_value=Path("voice.oga")):
        voice_errors.append(tc.transcribe_voice_message("token", voice_message)[1])

    image_errors = [tc.extract_photo_text("token", {})[1]]
    with mock.patch("jarvis_v2.tools.ocr.ocr_available", return_value=False):
        image_errors.append(tc.extract_photo_text("token", image_message)[1])
    with mock.patch(
        "jarvis_v2.tools.ocr.ocr_available",
        return_value=True,
    ), mock.patch.object(tc, "_download_telegram_file", return_value=None):
        image_errors.append(tc.extract_photo_text("token", image_message)[1])
    with mock.patch(
        "jarvis_v2.tools.ocr.ocr_available",
        return_value=True,
    ), mock.patch.object(
        tc,
        "_download_telegram_file",
        return_value=Path("image.png"),
    ), mock.patch(
        "jarvis_v2.tools.ocr.extract_text_from_image",
        side_effect=RuntimeError("/\x55sers/example/private/image SHOULD NOT APPEAR"),
    ):
        image_errors.append(tc.extract_photo_text("token", image_message)[1])
    with mock.patch(
        "jarvis_v2.tools.ocr.ocr_available",
        return_value=True,
    ), mock.patch.object(
        tc,
        "_download_telegram_file",
        return_value=Path("image.png"),
    ), mock.patch(
        "jarvis_v2.tools.ocr.extract_text_from_image",
        return_value="",
    ):
        image_errors.append(tc.extract_photo_text("token", image_message)[1])

    voice_text = "\n".join(voice_errors)
    image_text = "\n".join(image_errors)
    for expected in (
        "under 20MB",
        "Check Telegram network access",
        "`voice setup check`",
        "type the command instead",
    ):
        if expected not in voice_text:
            raise SystemExit(f"Telegram voice failure guidance missed {expected!r}: {voice_errors}")
    for expected in (
        "PNG/JPG",
        "`setup check`",
        "`swift --version`",
        "Check Telegram network access",
        "type the text instead",
    ):
        if expected not in image_text:
            raise SystemExit(f"Telegram image failure guidance missed {expected!r}: {image_errors}")
    combined = voice_text + "\n" + image_text
    for forbidden in ("/\x55sers/", "private/voice", "private/image", "SHOULD NOT APPEAR", "RuntimeError"):
        if forbidden in combined:
            raise SystemExit(f"Telegram media failure guidance leaked private detail: {combined}")


def test_telegram_https_uses_verified_certifi_context_and_hides_tls_failures() -> None:
    real_context = tc._telegram_ssl_context()
    if (
        real_context.check_hostname is not True
        or real_context.verify_mode != tc.ssl.CERT_REQUIRED
    ):
        raise SystemExit("Telegram TLS context must verify hostnames and certificates")

    sentinel_context = object()

    class _Response:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, limit=None):
            if limit is None:
                return self.payload
            return self.payload[:limit]

    with mock.patch.object(
        tc,
        "_telegram_ssl_context",
        return_value=sentinel_context,
    ), mock.patch.object(
        tc.urllib.request,
        "urlopen",
        return_value=_Response(b'{"ok":true,"result":{"id":1}}'),
    ) as api_open:
        result = tc._api_call("synthetic-token", "getMe", {}, timeout=3)
    if result.get("ok") is not True:
        raise SystemExit(f"Telegram verified-context API fixture failed: {result}")
    if api_open.call_args.kwargs.get("context") is not sentinel_context:
        raise SystemExit("Telegram Bot API request omitted the verified SSL context")

    private_tls_error = tc.urllib.error.URLError(
        tc.ssl.SSLCertVerificationError(
            "PRIVATE /\x55sers/example/certificates/telegram.pem SHOULD NOT APPEAR"
        )
    )
    with mock.patch.object(
        tc,
        "_telegram_ssl_context",
        return_value=sentinel_context,
    ), mock.patch.object(
        tc.urllib.request,
        "urlopen",
        side_effect=private_tls_error,
    ):
        failed = tc._api_call("synthetic-token", "getMe", {}, timeout=3)
    if failed != {"error": "network_outcome_unknown"}:
        raise SystemExit(f"Telegram TLS failure classification drifted: {failed}")
    if any(
        fragment in str(failed)
        for fragment in ("PRIVATE", "/\x55sers/", "telegram.pem", "SHOULD NOT APPEAR")
    ):
        raise SystemExit(f"Telegram TLS failure leaked private detail: {failed}")

    with tempfile.TemporaryDirectory(prefix="jarvis-tg-tls-file-") as temp:
        file_info = {
            "ok": True,
            "result": {"file_path": "voice/file.oga"},
        }
        with mock.patch.object(
            tc,
            "_api_call",
            return_value=file_info,
        ), mock.patch.object(
            tc,
            "_telegram_ssl_context",
            return_value=sentinel_context,
        ), mock.patch.object(
            tc.urllib.request,
            "urlopen",
            return_value=_Response(b"audio"),
        ) as file_open:
            downloaded = tc._download_telegram_file(
                "synthetic-token",
                "synthetic-file",
                temp,
            )
        if downloaded is None or downloaded.read_bytes() != b"audio":
            raise SystemExit("Telegram file download fixture did not persist the mocked bytes")
        if file_open.call_args.kwargs.get("context") is not sentinel_context:
            raise SystemExit("Telegram file request omitted the verified SSL context")

        with mock.patch.object(
            tc,
            "_api_call",
            return_value=file_info,
        ), mock.patch.object(
            tc,
            "_telegram_ssl_context",
            return_value=sentinel_context,
        ), mock.patch.object(
            tc.urllib.request,
            "urlopen",
            side_effect=private_tls_error,
        ):
            unavailable = tc._download_telegram_file(
                "synthetic-token",
                "synthetic-file",
                temp,
            )
        if unavailable is not None:
            raise SystemExit("Telegram file TLS failure should return no local path")


def test_unreadable_voice_transcription_result_fails_closed() -> None:
    _seed_offset(0)
    _setup_env("555001")
    leak_marker = "SHOULD NOT APPEAR"

    class HostileText:
        def __str__(self) -> str:
            raise RuntimeError(f"{leak_marker} /\x55sers/hostile/voice-transcript")

    rt = FakeRuntime(["should not run"])
    sent = []
    original = tc.transcribe_voice_message
    tc.transcribe_voice_message = lambda token, message: (HostileText(), "")
    try:
        bridge = tc.TelegramCommandBridge(
            runtime_factory=lambda: rt,
            send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
            fetch_func=lambda token, offset, timeout: [_voice_update(14, 555001)],
            chat_action_func=_no_chat_action,
        )
        if bridge.process_once(poll_timeout=0) != 1:
            raise SystemExit("unreadable voice transcript should still be processed")
    finally:
        tc.transcribe_voice_message = original
    if rt.inputs:
        raise SystemExit(f"unreadable voice transcript must not route a command: {rt.inputs}")
    if len(sent) != 1 or "Resend a clearer voice note" not in sent[0][1] or "type the command instead" not in sent[0][1]:
        raise SystemExit(f"unreadable voice transcript should use safe fallback: {sent}")
    for _cid, text, _markup in sent:
        for leaked in (leak_marker, "/\x55sers/hostile", "voice-transcript", "RuntimeError"):
            if leaked in text:
                raise SystemExit(f"unreadable voice transcript leaked hostile details: {sent}")


def test_unreadable_voice_error_fails_closed() -> None:
    _seed_offset(0)
    _setup_env("555001")
    leak_marker = "SHOULD NOT APPEAR"

    class HostileText:
        def __str__(self) -> str:
            raise RuntimeError(f"{leak_marker} /private/hostile-voice-error")

    rt = FakeRuntime(["should not run"])
    sent = []
    original = tc.transcribe_voice_message
    tc.transcribe_voice_message = lambda token, message: ("", HostileText())
    try:
        bridge = tc.TelegramCommandBridge(
            runtime_factory=lambda: rt,
            send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
            fetch_func=lambda token, offset, timeout: [_voice_update(15, 555001)],
            chat_action_func=_no_chat_action,
        )
        if bridge.process_once(poll_timeout=0) != 1:
            raise SystemExit("unreadable voice error should still be processed")
    finally:
        tc.transcribe_voice_message = original
    if rt.inputs:
        raise SystemExit(f"unreadable voice error must not route a command: {rt.inputs}")
    if len(sent) != 1 or "`voice setup check`" not in sent[0][1] or "type the command instead" not in sent[0][1]:
        raise SystemExit(f"unreadable voice error should use safe fallback: {sent}")
    for _cid, text, _markup in sent:
        for leaked in (leak_marker, "/private/hostile", "hostile-voice-error", "RuntimeError"):
            if leaked in text:
                raise SystemExit(f"unreadable voice error leaked hostile details: {sent}")


def test_oversized_voice_file_id_is_skipped() -> None:
    if tc._get_voice_file_id({"voice": {"file_id": "x", "file_size": tc.MAX_VOICE_FILE_BYTES + 1}}) != "":
        raise SystemExit("oversized voice file should be rejected")
    if tc._get_voice_file_id({"voice": {"file_id": "ok", "file_size": 1000}}) != "ok":
        raise SystemExit("normal voice file_id should be returned")


def _photo_update(update_id: int, chat_id) -> dict:
    return {
        "update_id": update_id,
        "message": {"chat": {"id": chat_id}, "photo": [
            {"file_id": "small", "file_size": 1000, "width": 90},
            {"file_id": "big", "file_size": 50000, "width": 1280},
        ]},
    }


def test_photo_is_ocred_and_surfaced_not_routed() -> None:
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(["should not run"])
    sent = []
    original = tc.extract_photo_text
    tc.extract_photo_text = lambda token, message: ("Receipt total 42.50 USD", "")
    try:
        bridge = tc.TelegramCommandBridge(
            runtime_factory=lambda: rt,
            send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
            fetch_func=lambda token, offset, timeout: [_photo_update(20, 555001)],
            chat_action_func=_no_chat_action,
        )
        if bridge.process_once(poll_timeout=0) != 1:
            raise SystemExit("photo was not processed")
    finally:
        tc.extract_photo_text = original
    if rt.inputs:
        raise SystemExit(f"OCR text must NOT be routed as a command: {rt.inputs}")
    if not any("Receipt total 42.50 USD" in t for _cid, t, _m in sent):
        raise SystemExit(f"extracted text not surfaced: {sent}")


def test_photo_ocr_error_is_reported() -> None:
    _seed_offset(0)
    _setup_env("555001")
    sent = []
    original = tc.extract_photo_text
    tc.extract_photo_text = lambda token, message: ("", "I didn't find any readable text in that image.")
    try:
        bridge = tc.TelegramCommandBridge(
            runtime_factory=lambda: FakeRuntime([]),
            send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
            fetch_func=lambda token, offset, timeout: [_photo_update(21, 555001)],
            chat_action_func=_no_chat_action,
        )
        bridge.process_once(poll_timeout=0)
    finally:
        tc.extract_photo_text = original
    if not any("readable text" in t for _cid, t, _m in sent):
        raise SystemExit(f"OCR error not reported: {sent}")


def test_photo_ocr_failure_omits_private_caption_and_guides_recovery() -> None:
    _seed_offset(0)
    _setup_env("555001")
    private_caption = "PRIVATE CAPTION SHOULD NOT APPEAR"
    sent = []
    original = tc.extract_photo_text
    tc.extract_photo_text = lambda token, message: (
        "",
        tc._telegram_image_recovery_guidance("read"),
    )
    update = _photo_update(25, 555001)
    update["message"]["caption"] = private_caption
    try:
        bridge = tc.TelegramCommandBridge(
            runtime_factory=lambda: FakeRuntime([]),
            send_func=lambda cid, text, markup=None: sent.append((cid, text, markup)),
            fetch_func=lambda token, offset, timeout: [update],
            chat_action_func=_no_chat_action,
        )
        if bridge.process_once(poll_timeout=0) != 1:
            raise SystemExit("captioned OCR failure was not processed")
    finally:
        tc.extract_photo_text = original
    if len(sent) != 1 or "`setup check`" not in sent[0][1] or "PNG/JPG" not in sent[0][1]:
        raise SystemExit(f"captioned OCR failure missed recovery guidance: {sent}")
    if private_caption in sent[0][1]:
        raise SystemExit(f"captioned OCR failure repeated private caption content: {sent}")


def test_photo_caption_is_surfaced_with_ocr_text() -> None:
    _seed_offset(0)
    _setup_env("555001")
    sent = []
    original = tc.extract_photo_text
    tc.extract_photo_text = lambda token, message: ("invoice 99", "")
    update = _photo_update(22, 555001)
    update["message"]["caption"] = "keep this receipt"
    try:
        bridge = tc.TelegramCommandBridge(
            runtime_factory=lambda: FakeRuntime([]),
            send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
            fetch_func=lambda token, offset, timeout: [update],
            chat_action_func=_no_chat_action,
        )
        bridge.process_once(poll_timeout=0)
    finally:
        tc.extract_photo_text = original
    joined = " ".join(t for _cid, t, _m in sent)
    if "keep this receipt" not in joined or "invoice 99" not in joined:
        raise SystemExit(f"caption + OCR text should both be surfaced: {sent}")


def test_unreadable_photo_ocr_result_fails_closed() -> None:
    _seed_offset(0)
    _setup_env("555001")
    leak_marker = "SHOULD NOT APPEAR"

    class HostileText:
        def __str__(self) -> str:
            raise RuntimeError(f"{leak_marker} /\x55sers/hostile/photo-ocr")

    sent = []
    original = tc.extract_photo_text
    tc.extract_photo_text = lambda token, message: (HostileText(), "")
    update = _photo_update(23, 555001)
    update["message"]["caption"] = HostileText()
    try:
        bridge = tc.TelegramCommandBridge(
            runtime_factory=lambda: FakeRuntime([]),
            send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
            fetch_func=lambda token, offset, timeout: [update],
            chat_action_func=_no_chat_action,
        )
        if bridge.process_once(poll_timeout=0) != 1:
            raise SystemExit("unreadable OCR result should still be processed")
    finally:
        tc.extract_photo_text = original
    if len(sent) != 1 or "PNG/JPG" not in sent[0][1] or "type the text instead" not in sent[0][1]:
        raise SystemExit(f"unreadable OCR result should use safe fallback: {sent}")
    for _cid, text, _markup in sent:
        for leaked in (leak_marker, "/\x55sers/hostile", "photo-ocr", "RuntimeError"):
            if leaked in text:
                raise SystemExit(f"unreadable OCR result leaked hostile details: {sent}")


def test_unreadable_photo_ocr_error_fails_closed() -> None:
    _seed_offset(0)
    _setup_env("555001")
    leak_marker = "SHOULD NOT APPEAR"

    class HostileText:
        def __str__(self) -> str:
            raise RuntimeError(f"{leak_marker} /private/hostile-ocr-error")

    sent = []
    original = tc.extract_photo_text
    tc.extract_photo_text = lambda token, message: ("", HostileText())
    try:
        bridge = tc.TelegramCommandBridge(
            runtime_factory=lambda: FakeRuntime([]),
            send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
            fetch_func=lambda token, offset, timeout: [_photo_update(24, 555001)],
            chat_action_func=_no_chat_action,
        )
        if bridge.process_once(poll_timeout=0) != 1:
            raise SystemExit("unreadable OCR error should still be processed")
    finally:
        tc.extract_photo_text = original
    if len(sent) != 1 or "`setup check`" not in sent[0][1] or "type the text instead" not in sent[0][1]:
        raise SystemExit(f"unreadable OCR error should use safe fallback: {sent}")
    for _cid, text, _markup in sent:
        for leaked in (leak_marker, "/private/hostile", "hostile-ocr-error", "RuntimeError"):
            if leaked in text:
                raise SystemExit(f"unreadable OCR error leaked hostile details: {sent}")


def test_image_file_id_picks_largest_within_cap() -> None:
    msg = {"photo": [
        {"file_id": "small", "file_size": 1000, "width": 90},
        {"file_id": "big", "file_size": 50000, "width": 1280},
        {"file_id": "huge", "file_size": tc.MAX_VOICE_FILE_BYTES + 1, "width": 4000},
    ]}
    if tc._get_image_file_id(msg) != "big":
        raise SystemExit("should pick the largest photo within the size cap")
    doc_img = {"document": {"file_id": "d1", "mime_type": "image/png", "file_size": 1234}}
    if tc._get_image_file_id(doc_img) != "d1":
        raise SystemExit("image document should be accepted")
    doc_pdf = {"document": {"file_id": "d2", "mime_type": "application/pdf", "file_size": 1234}}
    if tc._get_image_file_id(doc_pdf) != "":
        raise SystemExit("non-image document must be ignored")


def test_non_owner_is_ignored() -> None:
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(["should not run"])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [_update(11, 999999, "delete everything")],
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 0:
        raise SystemExit("non-owner message must be ignored")
    if rt.inputs or sent:
        raise SystemExit(f"non-owner message leaked: {rt.inputs} {sent}")


def test_non_owner_media_is_ignored_before_private_processing() -> None:
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(["should not run"])
    sent = []
    called = {"voice": 0, "photo": 0}
    original_voice = tc.transcribe_voice_message
    original_photo = tc.extract_photo_text

    def fake_voice(token, message):
        called["voice"] += 1
        return "delete everything", ""

    def fake_photo(token, message):
        called["photo"] += 1
        return "private receipt text", ""

    tc.transcribe_voice_message = fake_voice
    tc.extract_photo_text = fake_photo
    try:
        bridge = tc.TelegramCommandBridge(
            runtime_factory=lambda: rt,
            send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
            fetch_func=lambda token, offset, timeout: [
                _voice_update(12, 999999),
                _photo_update(13, 999999),
            ],
            chat_action_func=_no_chat_action,
        )
        if bridge.process_once(poll_timeout=0) != 0:
            raise SystemExit("non-owner media must not count as processed")
    finally:
        tc.transcribe_voice_message = original_voice
        tc.extract_photo_text = original_photo
    if called != {"voice": 0, "photo": 0}:
        raise SystemExit(f"non-owner media reached private processing: {called}")
    if rt.inputs or sent:
        raise SystemExit(f"non-owner media leaked into runtime or replies: {rt.inputs} {sent}")


def test_fresh_install_drains_backlog() -> None:
    _fresh_state()  # no offset = fresh install
    _setup_env("555001")
    rt = FakeRuntime(["should never run"])
    sent = []

    def fake_fetch(token, offset, timeout):
        if offset is None:
            return [_update(40, 555001, "stale setup message")]
        return []

    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=fake_fetch,
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 0 or rt.inputs or sent:
        raise SystemExit("fresh install executed backlog instead of draining it")
    if tc._load_offset() != 41:
        raise SystemExit(f"backlog not drained past update_id: offset={tc._load_offset()}")


def test_fresh_install_ignores_malformed_update_ids_when_draining() -> None:
    _fresh_state()
    _setup_env("555001")
    rt = FakeRuntime(["should never run"])
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: None,
        fetch_func=lambda token, offset, timeout: [
            {"update_id": "bad", "message": {"chat": {"id": 555001}, "text": "stale bad"}},
            _update(42, 555001, "stale valid"),
        ],
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 0 or rt.inputs:
        raise SystemExit("fresh install should drain backlog without executing commands")
    if tc._load_offset() != 43:
        raise SystemExit(f"fresh install should advance past valid update id only: {tc._load_offset()}")


def test_malformed_update_id_is_ignored_during_polling() -> None:
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(["ok"])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            {"update_id": "bad", "message": {"chat": {"id": 555001}, "text": "bad should not run"}},
            _update(12, 555001, "good command"),
        ],
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 1:
        raise SystemExit("malformed update id should be skipped while valid owner command runs")
    if rt.inputs != ["good command"] or sent != [("555001", "ok", None)]:
        raise SystemExit(f"malformed update id handling wrong: inputs={rt.inputs} sent={sent}")
    if tc._load_offset() != 13:
        raise SystemExit(f"offset should advance past valid update id: {tc._load_offset()}")


def test_transport_update_tokens_are_exact_stable_and_content_free() -> None:
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(["first", "replay", "distinct"])
    original_advance_offset = tc._advance_offset
    try:
        tc._advance_offset = lambda offset, expected, token=None, owner=None: expected
        first = tc.TelegramCommandBridge(
            runtime_factory=lambda: rt,
            send_func=lambda cid, text, markup=None: None,
            fetch_func=lambda token, offset, timeout: [_update(77, 555001, "private command text")],
            chat_action_func=_no_chat_action,
        )
        restarted = tc.TelegramCommandBridge(
            runtime_factory=lambda: rt,
            send_func=lambda cid, text, markup=None: None,
            fetch_func=lambda token, offset, timeout: [_update(77, 555001, "changed replay text")],
            chat_action_func=_no_chat_action,
        )
        if first.process_once(poll_timeout=0) != 1 or restarted.process_once(poll_timeout=0) != 1:
            raise SystemExit("same Telegram transport update did not replay through the bridge")
    finally:
        tc._advance_offset = original_advance_offset

    third = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, text, markup=None: None,
        fetch_func=lambda token, offset, timeout: [_update(78, 555001, "private command text")],
        chat_action_func=_no_chat_action,
    )
    if third.process_once(poll_timeout=0) != 1:
        raise SystemExit("distinct Telegram transport update did not run")
    expected = [
        "telegram-update:v1:77",
        "telegram-update:v1:77",
        "telegram-update:v1:78",
    ]
    if rt.request_tokens != expected:
        raise SystemExit(f"Telegram transport tokens were not deterministic and distinct: {rt.request_tokens}")
    if any("private" in token or "command" in token or len(token) > 64 for token in rt.request_tokens):
        raise SystemExit(f"Telegram transport token contained command content or was unbounded: {rt.request_tokens}")


def test_malformed_transport_update_id_types_fail_closed() -> None:
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(["ok"])
    invalid_ids = [
        True,
        1.0,
        "1",
        -1,
        tc.MAX_TRANSPORT_UPDATE_ID + 1,
        None,
    ]
    updates = [
        {"update_id": value, "message": {"chat": {"id": 555001}, "text": "must not run"}}
        for value in invalid_ids
    ]
    updates.append(_update(19, 555001, "valid command"))
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, text, markup=None: None,
        fetch_func=lambda token, offset, timeout: updates,
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 1:
        raise SystemExit("malformed Telegram transport identities affected valid update processing")
    if rt.inputs != ["valid command"] or rt.request_tokens != ["telegram-update:v1:19"]:
        raise SystemExit(
            f"malformed Telegram identities reached runtime: {rt.inputs} / {rt.request_tokens}"
        )


def test_malformed_update_fields_do_not_break_polling() -> None:
    _seed_offset(0)
    _setup_env("555001")
    leak_marker = "SHOULD NOT APPEAR"

    class HostileField:
        def __str__(self) -> str:
            raise RuntimeError(f"{leak_marker} /\x55sers/hostile/update-field")

        def __int__(self) -> int:
            raise RuntimeError(f"{leak_marker} /private/hostile-update-id")

    rt = FakeRuntime(["ok"])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            {
                "update_id": HostileField(),
                "message": {"chat": {"id": 555001}, "text": "bad id should not run"},
            },
            {
                "update_id": 13,
                "message": {"chat": {"id": HostileField()}, "text": "bad chat should not run"},
            },
            {
                "update_id": 14,
                "message": {"chat": {"id": 555001}, "text": HostileField()},
            },
            {"update_id": 15, "message": HostileField()},
            {"update_id": 16, "callback_query": HostileField()},
            _update(17, 555001, "good command"),
        ],
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 1:
        raise SystemExit("malformed update fields should be skipped while valid owner command runs")
    if rt.inputs != ["good command"] or sent != [("555001", "ok", None)]:
        raise SystemExit(f"malformed update field handling wrong: inputs={rt.inputs} sent={sent}")
    if tc._load_offset() != 18:
        raise SystemExit(f"offset should advance past valid update id: {tc._load_offset()}")
    for _cid, text, _markup in sent:
        for leaked in (leak_marker, "/\x55sers/hostile", "/private/hostile", "update-field"):
            if leaked in text:
                raise SystemExit(f"malformed update field leaked hostile details: {sent}")


def test_missing_token_or_owner_disables() -> None:
    _seed_offset(0)
    os.environ.pop("TELEGRAM_BOT_TOKEN", None)
    os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: FakeRuntime(["x"]),
        send_func=lambda cid, t, markup=None: None,
        fetch_func=lambda token, offset, timeout: [_update(1, 1, "hi")],
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 0:
        raise SystemExit("bridge must not run without token and owner")


def test_path_shaped_token_and_owner_are_rejected_locally() -> None:
    _seed_offset(0)
    raw_token = "/\x55sers/example/private/telegram-token"
    raw_owner = "/private/tmp/telegram-owner"
    os.environ["TELEGRAM_BOT_TOKEN"] = raw_token
    os.environ["JARVIS_OWNER_TELEGRAM"] = raw_owner
    rt = FakeRuntime(["should not run"])
    calls = []
    original_api_call = tc._api_call
    try:
        tc._api_call = lambda token, method, params, timeout: calls.append((token, method, params)) or (_ for _ in ()).throw(AssertionError("Telegram API should not start"))
        bridge = tc.TelegramCommandBridge(
            runtime_factory=lambda: rt,
            send_func=lambda cid, t, markup=None: calls.append(("send", cid, t)),
            fetch_func=lambda token, offset, timeout: calls.append(("fetch", token, offset)) or (_ for _ in ()).throw(AssertionError("Telegram polling should not start")),
            chat_action_func=_no_chat_action,
        )
        if bridge.process_once(poll_timeout=0) != 0:
            raise SystemExit("bridge should not process with path-shaped Telegram config")
        results = [
            tc.send_chat_action("555001"),
            tc.send_message("555001", "hello"),
            tc.answer_callback_query("callback-1", "hello"),
            tc.edit_message_text("555001", 1, "hello"),
        ]
        start_message = tc.start_telegram_control()
    finally:
        tc._api_call = original_api_call
    if rt.inputs or calls:
        raise SystemExit(f"path-shaped Telegram config reached runtime or network: inputs={rt.inputs} calls={calls}")
    for result in results:
        if result.get("ok") is not False or result.get("error") != "invalid token":
            raise SystemExit(f"path-shaped token should be rejected locally: {result}")
        if raw_token in str(result) or raw_owner in str(result):
            raise SystemExit(f"path-shaped Telegram config leaked through helper result: {result}")
    if "invalid" not in start_message or "value hidden" not in start_message:
        raise SystemExit(f"startup should report hidden invalid Telegram token: {start_message}")
    if raw_token in start_message or raw_owner in start_message:
        raise SystemExit(f"startup leaked path-shaped Telegram config: {start_message}")


def test_start_and_help_commands() -> None:
    _seed_offset(0)
    _setup_env("555001")
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: FakeRuntime([]),
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [_update(70, 555001, "/start")],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if not sent or "online" not in sent[0][1].lower():
        raise SystemExit(f"/start did not return an online message: {sent}")


def test_phone_status_shortcuts_map_to_read_only_commands() -> None:
    """"what broke" / "brief status" / "voice" are phone shortcuts for existing
    read-only planner phrases; "cockpit" / "freeze status" call the read-only
    cockpit tool directly. None of them may grant new authority or hit send paths."""
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(
        [
            "status listed",
            "voice cockpit listed",
            "voice setup listed",
            "voice stop listed",
            "capabilities listed",
            "safety listed",
            "safety listed",
            "safety listed",
            "safety listed",
            "safety listed",
            "safety listed",
            "safety listed",
            "safety listed",
            "safety listed",
            "safety listed",
            "safety listed",
            "privacy listed",
            "risk listed",
            "risk listed",
            "risk listed",
            "risk listed",
            "runs listed",
            "jobs listed",
            "channels listed",
            "approvals listed",
            "readiness listed",
            "memory stats listed",
            "learning review listed",
            "setup listed",
            "doctor listed",
            "messages capabilities listed",
            "productivity capabilities listed",
            "voice cockpit listed 2",
            "runs listed 2",
            "info capabilities listed",
            "markets capabilities listed",
            "utilities capabilities listed",
            "research capabilities listed",
            "writing capabilities listed",
            "memory capabilities listed",
            "notes capabilities listed",
            "safety listed 2",
            "risk listed 2",
        ]
    )
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(75, 555001, "status"),
            _update(76, 555001, "/voice"),
            _update(77, 555001, "/voice-setup"),
            _update(78, 555001, "/voice-stop"),
            _update(79, 555001, "capabilities"),
            _update(80, 555001, "/safety"),
            _update(81, 555001, "is Jarvis safe to use?"),
            _update(82, 555001, "안전하게 써도 돼?"),
            _update(83, 555001, "what are your limits?"),
            _update(84, 555001, "what can you not do?"),
            _update(85, 555001, "자비스 한계"),
            _update(86, 555001, "뭐 못해"),
            _update(87, 555001, "what requires approval?"),
            _update(88, 555001, "what tools require approval?"),
            _update(89, 555001, "승인 필요한 것"),
            _update(90, 555001, "어떤 명령이 승인 필요"),
            _update(91, 555001, "/privacy"),
            _update(92, 555001, "/risk"),
            _update(93, 555001, "which commands are risky?"),
            _update(94, 555001, "which tools are high risk?"),
            _update(95, 555001, "고위험 도구"),
            _update(96, 555001, "what broke"),
            _update(97, 555001, "brief status"),
            _update(98, 555001, "channels"),
            _update(99, 555001, "approvals"),
            _update(100, 555001, "readiness"),
            _update(101, 555001, "memory status"),
            _update(102, 555001, "learning status"),
            _update(103, 555001, "setup"),
            _update(104, 555001, "doctor"),
            _update(105, 555001, "can you send messages?"),
            _update(106, 555001, "can you check email?"),
            _update(107, 555001, "can you use voice?"),
            _update(108, 555001, "what is broken?"),
            _update(109, 555001, "can you check weather?"),
            _update(110, 555001, "can you check stock prices?"),
            _update(111, 555001, "can you translate?"),
            _update(112, 555001, "can you research the web?"),
            _update(113, 555001, "can you write text?"),
            _update(114, 555001, "can you remember things?"),
            _update(115, 555001, "can you take notes?"),
            _update(116, 555001, "can you read files?"),
            _update(117, 555001, "can you run commands?"),
            _update(118, 555001, "after action learning status"),
            _update(119, 555001, "what did Jarvis learn from the last failure"),
            _update(120, 555001, "is learning debt closed"),
            _update(121, 555001, "what learning debt is open"),
            _update(122, 555001, "recovery closure status"),
            _update(123, 555001, "is recovery debt closed"),
            _update(124, 555001, "repeated failure status"),
            _update(125, 555001, "failure learning status"),
            _update(126, 555001, "학습 루프 상태"),
            _update(127, 555001, "학습 부채 상태"),
            _update(128, 555001, "복구 학습 상태"),
            _update(129, 555001, "반복 실패 상태"),
            _update(130, 555001, "실패 학습 상태"),
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs != [
        "jarvis status",
        "voice command cockpit",
        "voice setup check",
        "voice stop intent: stop listening",
        "capability map",
        "safety status",
        "safety status",
        "safety status",
        "safety status",
        "safety status",
        "safety status",
        "safety status",
        "safety status",
        "safety status",
        "safety status",
        "safety status",
        "privacy report",
        "risk matrix",
        "risk matrix",
        "risk matrix",
        "risk matrix",
        "recent tool runs",
        "list scheduled jobs",
        "channel health",
        "pending approvals",
        "readiness report",
        "memory stats",
        "learning review",
        "setup check",
        "jarvis doctor",
        "capability map messages",
        "capability map productivity",
        "voice command cockpit",
        "recent tool runs",
        "capability map info",
        "capability map markets",
        "capability map utilities",
        "capability map research",
        "capability map writing",
        "capability map memory",
        "capability map notes",
        "safety status",
        "risk matrix",
        "after-action learning packet",
        "after-action learning packet",
        "execution learning closure",
        "execution learning closure",
        "recovery closure checklist",
        "recovery closure checklist",
        "repeated failure clusters",
        "failure learning cockpit",
        "learning review",
        "execution learning closure",
        "recovery closure checklist",
        "repeated failure clusters",
        "failure learning cockpit",
    ]:
        raise SystemExit(f"status shortcuts mapped wrong: {rt.inputs}")

    class CockpitTool:
        def handler(self, args):
            return SimpleNamespace(output="Capability cockpit (derived...)", metadata={})

    class CockpitRegistry:
        def get(self, name):
            if name != "capability_cockpit":
                raise KeyError(name)
            return CockpitTool()

    rt2 = FakeRuntime([])
    rt2.registry = CockpitRegistry()
    sent2 = []
    bridge2 = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt2,
        send_func=lambda cid, t, markup=None: sent2.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(90, 555001, "cockpit"),
            _update(91, 555001, "freeze status"),
            _update(92, 555001, "what is frozen"),
            _update(93, 555001, "동결 상태"),
            _update(94, 555001, "what should I test?"),
            _update(95, 555001, "test matrix status"),
            _update(96, 555001, "what live proofs are pending?"),
            _update(97, 555001, "라이브 테스트 뭐 해야 해?"),
            _update(98, 555001, "테스트 매트릭스 상태"),
            _update(99, 555001, "라이브 테스트 상태"),
            _update(100, 555001, "뭘 테스트해야 해?"),
            _update(101, 555001, "what can Codex touch?"),
            _update(102, 555001, "what should you not edit?"),
            _update(103, 555001, "proofs pending"),
            _update(104, 555001, "수정 금지 파일"),
            _update(105, 555001, "show guardrails"),
            _update(106, 555001, "what is the live test matrix?"),
            _update(107, 555001, "why frozen?"),
            _update(108, 555001, "safe lane"),
            _update(109, 555001, "프리즈 상태"),
            _update(110, 555001, "동결 파일"),
            _update(111, 555001, "검증 대기 뭐야?"),
            _update(112, 555001, "테스트 매트릭스 보여줘"),
            _update(113, 555001, "live freeze list"),
            _update(114, 555001, "can Codex edit planner?"),
            _update(115, 555001, "can Codex edit the send code?"),
            _update(116, 555001, "can you edit the planner?"),
            _update(117, 555001, "can you edit the send code?"),
            _update(118, 555001, "what should Codex avoid?"),
            _update(119, 555001, "what should Codex leave alone?"),
            _update(120, 555001, "what should Codex not edit?"),
            _update(121, 555001, "what should Codex not touch?"),
            _update(122, 555001, "what should you not touch?"),
        ],
        chat_action_func=_no_chat_action,
    )
    bridge2.process_once(poll_timeout=0)
    if rt2.inputs:
        raise SystemExit("cockpit/guardrail shortcuts must not route through runtime.handle")
    if len(sent2) != 33 or any(markup is not None for _cid, _text, markup in sent2):
        raise SystemExit(f"cockpit/guardrail shortcuts should be read-only without buttons: {sent2}")
    matrix_replies = [text for _cid, text, _markup in sent2 if "Pending live-proof matrix" in text]
    cockpit_replies = [text for _cid, text, _markup in sent2 if "capability cockpit" in text.lower()]
    if len(matrix_replies) != 10 or len(cockpit_replies) != 23:
        raise SystemExit(f"cockpit/live-matrix shortcuts routed to the wrong packets: {sent2}")
    for text in matrix_replies:
        for expected in (
            "KakaoTalk",
            "Telegram",
            "Instagram",
            "iMessage",
            "Phone:",
            "FaceTime:",
            "KakaoTalk call:",
            "Telegram call:",
            "Instagram call:",
            "Call proof gate (all fields are required)",
            "approval card's channel, target, and effective mode match",
            "bounded approval ID and tool-run ID",
            "call_requested / known_not_started / outcome_unknown",
            "`call_requested` alone is not live proof",
            "Only ringing or connected closes",
            "never retry automatically",
            "fresh request and approval",
            "Report format",
            "channel:",
            "result: pass / fail / blocked",
            "approval card shown",
            "approved target + mode matched",
            "call placement outcome",
            "recipient confirmation",
            "last visible stage/error",
            "do not include secrets",
            "message content",
            "does not approve, send, call, run live_check",
        ):
            if expected not in text:
                raise SystemExit(f"live matrix reply lost {expected!r}: {text!r}")
        for forbidden in ("Fixture", "가상연락처일", "가상연락처이", "BotFather"):
            if forbidden in text:
                raise SystemExit(f"live matrix reply should not echo target names: {text!r}")


def test_phone_control_shortcuts_share_real_runtime_state() -> None:
    _seed_offset(0)
    _setup_env("555001")
    with tempfile.TemporaryDirectory(prefix="jarvis-phone-control-state-") as temp:
        root = Path(temp)
        seeded_runtime = make_temp_runtime(root)
        seeded_runtime.store.log_tool_run(
            session_id=seeded_runtime.session_id,
            tool_name="send_telegram",
            risk="HIGH_RISK",
            ok=False,
            approved=True,
            output="Owner Telegram delivery failed at the mocked transport stage.",
            metadata={
                "failure_kind": "transport_error",
                "telegram_send_stage": "mocked_transport",
                "external_side_effect": True,
            },
        )
        held = seeded_runtime.handle(
            "send a telegram to me saying phone control state rehearsal"
        )
        pending = seeded_runtime.store.list_pending_approvals(limit=5)
        if len(pending) != 1 or pending[0]["tool_name"] != "send_telegram":
            raise SystemExit(f"phone-control state rehearsal did not queue one approval: {held}")
        approval_id = int(pending[0]["id"])

        sent = []
        bridge = tc.TelegramCommandBridge(
            runtime_factory=lambda: make_temp_runtime(root),
            send_func=lambda cid, text, markup=None: sent.append((cid, text, markup)),
            fetch_func=lambda token, offset, timeout: [
                _update(200, 555001, "what broke"),
                _update(201, 555001, "approvals"),
                _update(202, 555001, "cockpit summary"),
            ],
            chat_action_func=_no_chat_action,
        )
        if bridge.process_once(poll_timeout=0) != 3:
            raise SystemExit("real-state phone-control shortcuts were not all processed")
        if len(sent) != 3 or any(markup is not None for _cid, _text, markup in sent):
            raise SystemExit(f"real-state phone-control shortcuts should be read-only: {sent}")

        what_broke, approvals, cockpit = [text for _cid, text, _markup in sent]
        if "send_telegram" not in what_broke or "failed" not in what_broke.lower():
            raise SystemExit(f"what broke lost the seeded failure: {what_broke!r}")
        if (
            f"#{approval_id}" not in approvals
            or "send_telegram" not in approvals
            or f"approval packet {approval_id}" not in approvals
            or f"approve approval {approval_id}" not in approvals
        ):
            raise SystemExit(f"approvals lost the real pending review commands: {approvals!r}")
        if "Cockpit summary" not in cockpit:
            raise SystemExit(f"cockpit summary lost its heading: {cockpit!r}")
        if "approval-held" not in cockpit.lower() or "attention" not in cockpit.lower():
            raise SystemExit(f"cockpit summary lost failure/approval-held state: {cockpit!r}")

        verification_runtime = make_temp_runtime(root)
        if verification_runtime.store.get_pending_approval(approval_id, status="pending") is None:
            raise SystemExit("read-only phone-control shortcuts mutated the pending approval")


def test_personal_proof_matrix_shortcuts_are_read_only_and_private() -> None:
    """Personal-proof report prompts should return a privacy-safe checklist, not
    route through runtime or touch personal integrations."""
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime([])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(113, 555001, "personal proof matrix"),
            _update(114, 555001, "personal proof result format"),
            _update(115, 555001, "what personal proofs should I test?"),
            _update(116, 555001, "what personal integrations should I test?"),
            _update(117, 555001, "how should I report personal proof results?"),
            _update(118, 555001, "개인 증명 매트릭스"),
            _update(119, 555001, "개인 증명 결과 형식"),
            _update(120, 555001, "개인 증명 뭐 해야 해?"),
            _update(121, 555001, "개인 연동 뭐 테스트해?"),
            _update(122, 555001, "개인 증명 어떻게 보고해?"),
            _update(123, 555001, "what calendar write should I test"),
            _update(124, 555001, "how should I report calendar write proof"),
            _update(125, 555001, "calendar create update delete proof matrix"),
            _update(126, 555001, "calendar write result format"),
            _update(127, 555001, "what email read should I test"),
            _update(128, 555001, "what email search should I test"),
            _update(129, 555001, "what email send should I test"),
            _update(130, 555001, "how should I report email read proof"),
            _update(131, 555001, "how should I report email search proof"),
            _update(132, 555001, "how should I report email send proof"),
            _update(133, 555001, "email read proof matrix"),
            _update(134, 555001, "email search proof matrix"),
            _update(135, 555001, "email send proof matrix"),
            _update(136, 555001, "what reminder proof should I test"),
            _update(137, 555001, "what set reminder proof should I test"),
            _update(138, 555001, "how should I report reminder proof"),
            _update(139, 555001, "how should I report set reminder proof"),
            _update(140, 555001, "reminder proof matrix"),
            _update(141, 555001, "set reminder proof matrix"),
            _update(142, 555001, "what contact lookup should I test"),
            _update(143, 555001, "how should I report contact lookup proof"),
            _update(144, 555001, "contact lookup proof matrix"),
            _update(145, 555001, "calendar write proof matrix"),
            _update(146, 555001, "personal integrations result format"),
            _update(147, 555001, "personal integrations proof matrix"),
            _update(148, 555001, "personal integration result format"),
            _update(149, 555001, "personal integration proof matrix"),
            _update(150, 555001, "캘린더 쓰기 증명 매트릭스"),
            _update(151, 555001, "이메일 읽기 증명"),
            _update(152, 555001, "이메일 검색 증명"),
            _update(153, 555001, "이메일 보내기 증명"),
            _update(154, 555001, "리마인더 증명 매트릭스"),
            _update(155, 555001, "연락처 조회 증명 매트릭스"),
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs:
        raise SystemExit(f"personal-proof matrix shortcuts must not route through runtime.handle: {rt.inputs}")
    if len(sent) != 43 or any(markup is not None for _cid, _text, markup in sent):
        raise SystemExit(f"personal-proof matrix replies should be read-only without buttons: {sent}")
    for _cid, text, _markup in sent:
        for expected in (
            "Pending personal-proof matrix",
            "Calendar write cycle",
            "Email read/search",
            "Email send",
            "Reminder delivery",
            "Contacts lookup",
            "Report format",
            "area: calendar / email-read / email-send / reminder / contacts",
            "result: pass / fail / blocked",
            "approval card shown: yes / no / n/a",
            "last visible stage/error",
            "optional audit row id",
            "Privacy boundary",
            "do not include email contents",
            "contact handles",
            "does not access accounts, read email",
            "claim personal proof done",
        ):
            if expected not in text:
                raise SystemExit(f"personal-proof matrix reply lost {expected!r}: {text!r}")
        for forbidden in ("Fixture", "가상연락처일", "가상연락처이", "synthetic_owner_handle", "message body", "email body"):
            if forbidden in text:
                raise SystemExit(f"personal-proof matrix reply leaked private target/report detail: {text!r}")


def test_scheduler_proof_matrix_shortcuts_are_read_only_and_private() -> None:
    """Scheduled-job proof prompts should return a privacy-safe checklist, not
    run jobs, restart daemons, or touch live delivery channels."""
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime([])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(123, 555001, "scheduler proof matrix"),
            _update(124, 555001, "scheduler proof result format"),
            _update(125, 555001, "scheduled jobs proof matrix"),
            _update(126, 555001, "what scheduled jobs should I test?"),
            _update(127, 555001, "how should I report scheduler proof?"),
            _update(128, 555001, "7-day scheduler proof"),
            _update(129, 555001, "스케줄 증명 매트릭스"),
            _update(130, 555001, "스케줄 증명 결과 형식"),
            _update(131, 555001, "스케줄 증명 뭐 해야 해?"),
            _update(132, 555001, "예약 작업 뭐 테스트해?"),
            _update(133, 555001, "예약 작업 증명 형식"),
            _update(134, 555001, "morning brief delivery proof matrix"),
            _update(135, 555001, "jobs 7 day proof"),
            _update(136, 555001, "jobs seven day proof"),
            _update(137, 555001, "scheduled jobs 7 day proof"),
            _update(138, 555001, "scheduled jobs streak"),
            _update(139, 555001, "scheduled job streak status"),
            _update(140, 555001, "scheduler streak status"),
            _update(141, 555001, "did scheduled jobs run for 7 days?"),
            _update(142, 555001, "are scheduled jobs running daily?"),
            _update(143, 555001, "what is the 7 day scheduler proof?"),
            _update(144, 555001, "what scheduler streak should I test?"),
            _update(145, 555001, "how should I report scheduled job streak proof?"),
            _update(146, 555001, "daily streak proof"),
            _update(147, 555001, "daily job streak proof"),
            _update(148, 555001, "job streak proof matrix"),
            _update(149, 555001, "scheduled delivery proof matrix"),
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs:
        raise SystemExit(f"scheduler-proof matrix shortcuts must not route through runtime.handle: {rt.inputs}")
    if len(sent) != 27 or any(markup is not None for _cid, _text, markup in sent):
        raise SystemExit(f"scheduler-proof matrix replies should be read-only without buttons: {sent}")
    for _cid, text, _markup in sent:
        for expected in (
            "Pending scheduler-proof matrix",
            "Morning Brief delivery",
            "Scheduler freshness",
            "7-day streak",
            "Silent-failure check",
            "Recovery check",
            "Report format",
            "area: morning-brief / freshness / 7-day-streak / silent-failure / recovery",
            "result: pass / fail / blocked",
            "date range: YYYY-MM-DD..YYYY-MM-DD",
            "enabled jobs: count only",
            "last visible stage/error",
            "live_check row",
            "Privacy boundary",
            "do not include calendar details",
            "full job payloads",
            "does not run jobs, send Telegram messages",
            "restart daemons",
            "claim scheduler proof done",
        ):
            if expected not in text:
                raise SystemExit(f"scheduler-proof matrix reply lost {expected!r}: {text!r}")
        for forbidden in ("Fixture", "가상연락처일", "가상연락처이", "synthetic_owner_handle", "/\x55sers/", "/private/"):
            if forbidden in text:
                raise SystemExit(f"scheduler-proof matrix reply leaked private target/report detail: {text!r}")


def test_daily_value_proof_matrix_shortcuts_are_read_only_and_private() -> None:
    """Daily-value proof prompts should return a privacy-safe checklist, not
    compose/send briefs, run jobs, access accounts, or expose brief contents."""
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime([])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(203, 555001, "daily value proof matrix"),
            _update(204, 555001, "daily value proof result format"),
            _update(205, 555001, "morning brief proof matrix"),
            _update(206, 555001, "morning brief proof result format"),
            _update(207, 555001, "brief delivery proof matrix"),
            _update(208, 555001, "what daily value should I test?"),
            _update(209, 555001, "what morning brief should I test?"),
            _update(210, 555001, "how should I report daily value proof?"),
            _update(211, 555001, "일일 가치 증명 매트릭스"),
            _update(212, 555001, "아침 브리핑 증명 형식"),
            _update(213, 555001, "브리핑 전송 증명"),
            _update(214, 555001, "브리핑 뭐 테스트해?"),
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs:
        raise SystemExit(f"daily-value proof matrix shortcuts must not route through runtime.handle: {rt.inputs}")
    if len(sent) != 12 or any(markup is not None for _cid, _text, markup in sent):
        raise SystemExit(f"daily-value proof matrix replies should be read-only without buttons: {sent}")
    for _cid, text, _markup in sent:
        for expected in (
            "Pending daily-value proof matrix",
            "Morning Brief usefulness",
            "Delivery cadence",
            "On-demand retry",
            "Brief status",
            "Recovery check",
            "Report format",
            "area: usefulness / cadence / on-demand / status / recovery",
            "result: pass / fail / blocked",
            "date: YYYY-MM-DD",
            "sent to owner: yes / no / n/a",
            "unavailable sections: count only, or n/a",
            "last visible stage/error",
            "optional live_check row: daily brief / scheduled jobs / jobs 7-day proof",
            "Privacy boundary",
            "do not include full brief text",
            "calendar titles",
            "does not compose or send a brief, run scheduled jobs, call APIs",
            "claim daily-value proof done",
        ):
            if expected not in text:
                raise SystemExit(f"daily-value proof matrix reply lost {expected!r}: {text!r}")
        for forbidden in (
            "Fixture",
            "가상연락처일",
            "가상연락처이",
            "synthetic_owner_handle",
            "/\x55sers/",
            "/private/",
            "full transcript",
            "calendar title:",
            "email subject:",
            "reminder body:",
            "chat_id",
            "555001",
            "999999",
        ):
            if forbidden in text:
                raise SystemExit(f"daily-value proof matrix reply leaked private target/report detail: {text!r}")


def test_approval_proof_matrix_shortcuts_are_read_only_and_private() -> None:
    """Phone approval-flow proof prompts should return a privacy-safe checklist,
    not create approvals, execute tools, or expose planned args."""
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime([])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(134, 555001, "approval proof matrix"),
            _update(135, 555001, "approval proof result format"),
            _update(136, 555001, "phone approval proof"),
            _update(137, 555001, "what approval flow should I test?"),
            _update(138, 555001, "how should I report approval proof?"),
            _update(139, 555001, "승인 증명 매트릭스"),
            _update(140, 555001, "승인 결과 형식"),
            _update(141, 555001, "승인 증명 뭐 해야 해?"),
            _update(142, 555001, "승인 흐름 뭐 테스트해?"),
            _update(143, 555001, "approval buttons proof matrix"),
            _update(144, 555001, "phone approval buttons proof matrix"),
            _update(145, 555001, "approval callback proof matrix"),
            _update(146, 555001, "how should I report approval callback proof?"),
            _update(147, 555001, "what approval buttons should I test?"),
            _update(148, 555001, "폰 승인 버튼 증명"),
            _update(149, 555001, "텔레그램 승인 증명"),
            _update(150, 555001, "승인 콜백 증명"),
            _update(151, 555001, "승인 거절 증명"),
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs:
        raise SystemExit(f"approval-proof matrix shortcuts must not route through runtime.handle: {rt.inputs}")
    if len(sent) != 18 or any(markup is not None for _cid, _text, markup in sent):
        raise SystemExit(f"approval-proof matrix replies should be read-only without buttons: {sent}")
    for _cid, text, _markup in sent:
        for expected in (
            "Pending approval-proof matrix",
            "Risky action prompt",
            "Approval last-look",
            "Owner approve path",
            "Owner deny path",
            "Non-owner guard",
            "Recovery check",
            "Report format",
            "area: prompt / last-look / approve / deny / non-owner / recovery",
            "result: pass / fail / blocked",
            "approval id: number only, or n/a",
            "callback shown: approve / deny / both / none",
            "last visible stage/error",
            "live_check row",
            "Privacy boundary",
            "do not include message contents",
            "full planned args",
            "does not create approvals, approve, dismiss, execute tools",
            "claim approval proof done",
        ):
            if expected not in text:
                raise SystemExit(f"approval-proof matrix reply lost {expected!r}: {text!r}")
        for forbidden in (
            "Fixture",
            "가상연락처일",
            "가상연락처이",
            "synthetic_owner_handle",
            "/\x55sers/",
            "/private/",
            "approve:7",
            "deny:7",
            "555001",
            "999999",
        ):
            if forbidden in text:
                raise SystemExit(f"approval-proof matrix reply leaked private target/report detail: {text!r}")


def test_conversation_research_proof_matrix_shortcuts_are_read_only_and_private() -> None:
    """Conversation/research proof prompts should return a privacy-safe checklist,
    not run research, call models, execute tools, or expose transcripts/prompts."""
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime([])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(215, 555001, "conversation research proof matrix"),
            _update(216, 555001, "conversation research proof result format"),
            _update(217, 555001, "research proof matrix"),
            _update(218, 555001, "web research proof matrix"),
            _update(219, 555001, "section A proof matrix"),
            _update(220, 555001, "what conversation research should I test?"),
            _update(221, 555001, "what research proof should I test?"),
            _update(222, 555001, "how should I report conversation research proof?"),
            _update(223, 555001, "대화 연구 증명 매트릭스"),
            _update(224, 555001, "연구 증명 매트릭스"),
            _update(225, 555001, "검색 증명 매트릭스"),
            _update(226, 555001, "연구 증명 어떻게 보고해?"),
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs:
        raise SystemExit(f"conversation-research proof matrix shortcuts must not route through runtime.handle: {rt.inputs}")
    if len(sent) != 12 or any(markup is not None for _cid, _text, markup in sent):
        raise SystemExit(f"conversation-research proof matrix replies should be read-only without buttons: {sent}")
    for _cid, text, _markup in sent:
        for expected in (
            "Pending conversation-research proof matrix",
            "Research answer",
            "Wiki/web fallback",
            "Ordinary chat",
            "Mixed conversation",
            "Recovery check",
            "Report format",
            "area: research / wiki-fallback / chat-model / mixed-conversation / recovery",
            "result: pass / fail / blocked",
            "source count: number only, or n/a",
            "chat p95 ms: number only, or n/a",
            "model path: model / fallback / n/a",
            "last visible stage/error",
            "optional live_check row: research / mixed conversation / model routing status",
            "Privacy boundary",
            "do not include transcript contents",
            "full prompts",
            "does not run research, fetch pages, call the model",
            "claim conversation-research proof done",
        ):
            if expected not in text:
                raise SystemExit(f"conversation-research proof matrix reply lost {expected!r}: {text!r}")
        for forbidden in (
            "Fixture",
            "가상연락처일",
            "가상연락처이",
            "synthetic_owner_handle",
            "/\x55sers/",
            "/private/",
            "full transcript",
            "calendar title",
            "task body",
            "email body",
            "raw prompt",
            "555001",
            "999999",
        ):
            if forbidden in text:
                raise SystemExit(f"conversation-research proof matrix reply leaked private target/report detail: {text!r}")


def test_mixed_conversation_proof_matrix_shortcuts_are_read_only_and_private() -> None:
    """Mixed-conversation proof prompts should return a privacy-safe checklist,
    not run conversations, call models, tune config, or expose transcripts."""
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime([])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(143, 555001, "mixed conversation proof matrix"),
            _update(144, 555001, "mixed conversation proof result format"),
            _update(145, 555001, "chat latency proof matrix"),
            _update(146, 555001, "chat latency proof result format"),
            _update(147, 555001, "conversation proof matrix"),
            _update(148, 555001, "conversation proof result format"),
            _update(149, 555001, "what mixed conversation should I test?"),
            _update(150, 555001, "what conversation should I test?"),
            _update(151, 555001, "what chat proof should I run?"),
            _update(152, 555001, "how should I report mixed conversation proof?"),
            _update(153, 555001, "how should I report conversation proof?"),
            _update(154, 555001, "how should I report chat latency proof?"),
            _update(155, 555001, "대화 증명 매트릭스"),
            _update(156, 555001, "대화 증명 결과 형식"),
            _update(157, 555001, "대화 증명 뭐 해야 해?"),
            _update(158, 555001, "대화 증명 어떻게 보고해?"),
            _update(159, 555001, "혼합 대화 증명"),
            _update(160, 555001, "혼합 대화 뭐 테스트해?"),
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs:
        raise SystemExit(f"mixed-conversation proof matrix shortcuts must not route through runtime.handle: {rt.inputs}")
    if len(sent) != 18 or any(markup is not None for _cid, _text, markup in sent):
        raise SystemExit(f"mixed-conversation proof matrix replies should be read-only without buttons: {sent}")
    for _cid, text, _markup in sent:
        for expected in (
            "Pending mixed-conversation proof matrix",
            "10+ turn run",
            "Coherence check",
            "Latency check",
            "chat p95 is <= 8000ms",
            "Model path check",
            "Recovery check",
            "Report format",
            "area: turn-mix / coherence / latency / model-path / recovery",
            "result: pass / fail / blocked",
            "turn count: number only",
            "chat p95 ms: number only, or n/a",
            "slowest stage/error",
            "optional live_check row: mixed conversation",
            "Privacy boundary",
            "do not include transcript contents",
            "full model prompts",
            "does not run a conversation, call the model, change tuning knobs",
            "claim the mixed-conversation proof done",
        ):
            if expected not in text:
                raise SystemExit(f"mixed-conversation proof matrix reply lost {expected!r}: {text!r}")
        for forbidden in (
            "Fixture",
            "가상연락처일",
            "가상연락처이",
            "synthetic_owner_handle",
            "/\x55sers/",
            "/private/",
            "full transcript",
            "calendar title",
            "task body",
            "email body",
            "555001",
            "999999",
        ):
            if forbidden in text:
                raise SystemExit(f"mixed-conversation proof matrix reply leaked private target/report detail: {text!r}")


def test_voice_proof_matrix_shortcuts_are_read_only_and_private() -> None:
    """Voice-proof report prompts should return a privacy-safe checklist, not
    access the microphone, download/transcribe voice files, speak, or route commands."""
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime([])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(152, 555001, "voice proof matrix"),
            _update(153, 555001, "voice proof result format"),
            _update(154, 555001, "voice live proof format"),
            _update(155, 555001, "telegram voice proof matrix"),
            _update(156, 555001, "local voice proof matrix"),
            _update(157, 555001, "korean voice proof matrix"),
            _update(158, 555001, "telegram voice proof result format"),
            _update(159, 555001, "local voice proof result format"),
            _update(160, 555001, "korean voice proof result format"),
            _update(161, 555001, "push to talk proof matrix"),
            _update(162, 555001, "push-to-talk proof matrix"),
            _update(163, 555001, "talk.py proof matrix"),
            _update(164, 555001, "spoken reply proof matrix"),
            _update(165, 555001, "voice speak proof matrix"),
            _update(166, 555001, "what voice should I test?"),
            _update(167, 555001, "what voice proofs should I test?"),
            _update(168, 555001, "what local voice should I test?"),
            _update(169, 555001, "what telegram voice should I test?"),
            _update(170, 555001, "what Korean voice should I test?"),
            _update(171, 555001, "how should I report voice proof?"),
            _update(172, 555001, "how should I report local voice proof?"),
            _update(173, 555001, "how should I report telegram voice proof?"),
            _update(174, 555001, "how should I report Korean voice proof?"),
            _update(175, 555001, "음성 증명 매트릭스"),
            _update(176, 555001, "음성 증명 결과 형식"),
            _update(177, 555001, "음성 뭐 테스트해?"),
            _update(178, 555001, "음성 증명 뭐 해야 해?"),
            _update(179, 555001, "음성 증명 어떻게 보고해?"),
            _update(180, 555001, "텔레그램 음성 증명"),
            _update(181, 555001, "로컬 음성 증명"),
            _update(182, 555001, "한국어 음성 증명 형식"),
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs:
        raise SystemExit(f"voice-proof matrix shortcuts must not route through runtime.handle: {rt.inputs}")
    if len(sent) != 31 or any(markup is not None for _cid, _text, markup in sent):
        raise SystemExit(f"voice-proof matrix replies should be read-only without buttons: {sent}")
    for _cid, text, _markup in sent:
        for expected in (
            "Pending voice-proof matrix",
            "Telegram voice note",
            "Local push-to-talk",
            "Spoken reply",
            "Korean voice",
            "Recovery check",
            "Report format",
            "area: telegram-voice / local-talk / speak / korean-transcription / recovery",
            "result: pass / fail / blocked",
            "language: en / ko / mixed / n/a",
            "approval card shown: yes / no / n/a",
            "last visible stage/error",
            "optional live_check row: telegram voice / local talk / voice setup",
            "Privacy boundary",
            "do not include transcript contents",
            "audio file ids",
            "does not access the microphone, record audio, download voice files",
            "claim the voice proof done",
        ):
            if expected not in text:
                raise SystemExit(f"voice-proof matrix reply lost {expected!r}: {text!r}")
        for forbidden in (
            "Fixture",
            "가상연락처일",
            "가상연락처이",
            "synthetic_owner_handle",
            "/\x55sers/",
            "/private/",
            "full transcript",
            "voice_file_id",
            "file_id",
            "chat_id",
            "555001",
            "999999",
        ):
            if forbidden in text:
                raise SystemExit(f"voice-proof matrix reply leaked private target/report detail: {text!r}")


def test_phone_control_proof_matrix_shortcuts_are_read_only_and_private() -> None:
    """Phone-control proof prompts should return a privacy-safe checklist, not
    inspect payloads, create approvals, execute tools, or expose private state."""
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime([])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(163, 555001, "phone control proof matrix"),
            _update(164, 555001, "phone control proof result format"),
            _update(165, 555001, "phone control live proof format"),
            _update(166, 555001, "what broke proof matrix"),
            _update(167, 555001, "cockpit proof matrix"),
            _update(168, 555001, "cockpit approvals proof matrix"),
            _update(169, 555001, "what phone control should I test?"),
            _update(170, 555001, "what phone shortcuts should I test?"),
            _update(171, 555001, "how should I report phone control proof?"),
            _update(172, 555001, "how should I report phone shortcut proof?"),
            _update(173, 555001, "폰컨트롤 증명 매트릭스"),
            _update(174, 555001, "폰 제어 증명 형식"),
            _update(175, 555001, "뭐가 고장났어 증명 형식"),
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs:
        raise SystemExit(f"phone-control proof matrix shortcuts must not route through runtime.handle: {rt.inputs}")
    if len(sent) != 13 or any(markup is not None for _cid, _text, markup in sent):
        raise SystemExit(f"phone-control proof matrix replies should be read-only without buttons: {sent}")
    for _cid, text, _markup in sent:
        for expected in (
            "Pending phone-control proof matrix",
            "What broke",
            "Cockpit summary",
            "Cockpit attention",
            "Approvals lane",
            "Owner-only guard",
            "Recovery check",
            "Report format",
            "area: what-broke / cockpit-summary / attention / approvals / owner-guard / recovery",
            "result: pass / fail / blocked",
            "approval-held shown: yes / no / n/a",
            "failure lane shown: yes / no / n/a",
            "last visible stage/error",
            "optional live_check row: phone control / channel health / phone approvals",
            "Privacy boundary",
            "do not include message contents",
            "run payloads",
            "does not inspect private payloads, create approvals, approve, dismiss, execute tools",
            "claim phone-control proof done",
        ):
            if expected not in text:
                raise SystemExit(f"phone-control proof matrix reply lost {expected!r}: {text!r}")
        for forbidden in (
            "Fixture",
            "가상연락처일",
            "가상연락처이",
            "synthetic_owner_handle",
            "/\x55sers/",
            "/private/",
            "full transcript",
            "planned args:",
            "run payload:",
            "chat_id",
            "approve:7",
            "deny:7",
            "555001",
            "999999",
        ):
            if forbidden in text:
                raise SystemExit(f"phone-control proof matrix reply leaked private target/report detail: {text!r}")


def test_reboot_proof_matrix_shortcuts_are_read_only_and_private() -> None:
    """Reboot/daemon proof prompts should return a privacy-safe checklist, not
    restart daemons, run launchctl, change networking, or expose private logs."""
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime([])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(176, 555001, "reboot proof matrix"),
            _update(177, 555001, "reboot proof result format"),
            _update(178, 555001, "daemon proof matrix"),
            _update(179, 555001, "daemon proof result format"),
            _update(180, 555001, "daemon startup proof matrix"),
            _update(181, 555001, "launchagent proof matrix"),
            _update(182, 555001, "network recovery proof matrix"),
            _update(183, 555001, "network loss proof matrix"),
            _update(184, 555001, "what reboot proof should I test?"),
            _update(185, 555001, "what daemon proof should I test?"),
            _update(186, 555001, "how should I report reboot proof?"),
            _update(187, 555001, "how should I report daemon proof?"),
            _update(188, 555001, "재부팅 증명 매트릭스"),
            _update(189, 555001, "재부팅 증명 형식"),
            _update(190, 555001, "데몬 증명 매트릭스"),
            _update(191, 555001, "네트워크 복구 증명"),
            _update(216, 555001, "post reboot proof matrix"),
            _update(217, 555001, "post-reboot proof matrix"),
            _update(218, 555001, "launchd proof matrix"),
            _update(219, 555001, "telegram control daemon status"),
            _update(220, 555001, "telegram control proof matrix"),
            _update(221, 555001, "status server proof matrix"),
            _update(222, 555001, "scheduler daemon proof matrix"),
            _update(223, 555001, "network loss proof result format"),
            _update(224, 555001, "network recovery proof result format"),
            _update(225, 555001, "how should I report network recovery proof?"),
            _update(226, 555001, "what network recovery should I test?"),
            _update(227, 555001, "what should I test after reboot?"),
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs:
        raise SystemExit(f"reboot-proof matrix shortcuts must not route through runtime.handle: {rt.inputs}")
    if len(sent) != 28 or any(markup is not None for _cid, _text, markup in sent):
        raise SystemExit(f"reboot-proof matrix replies should be read-only without buttons: {sent}")
    for _cid, text, _markup in sent:
        for expected in (
            "Pending reboot-proof matrix",
            "Post-reboot status",
            "LaunchAgent install",
            "Scheduler freshness",
            "Telegram control",
            "Network recovery",
            "Recovery check",
            "Report format",
            "area: post-reboot / launchagent / scheduler / telegram-control / network-recovery / recovery",
            "result: pass / fail / blocked",
            "daemon: telegram-control / scheduler / status-server / all / n/a",
            "reboot observed: yes / no / n/a",
            "network-loss observed: yes / no / n/a",
            "last visible stage/error",
            "optional live_check row: daemon startup / scheduled jobs / phone control / channel health",
            "Privacy boundary",
            "do not include plist paths",
            "logs with private text",
            "does not run launchctl, install or edit LaunchAgents, restart daemons",
            "claim reboot proof done",
        ):
            if expected not in text:
                raise SystemExit(f"reboot-proof matrix reply lost {expected!r}: {text!r}")
        for forbidden in (
            "Fixture",
            "가상연락처일",
            "가상연락처이",
            "synthetic_owner_handle",
            "/\x55sers/",
            "/private/",
            "chat_id",
            "launchctl load",
            "launchctl unload",
            "launchctl bootout",
            "approve:7",
            "deny:7",
            "555001",
            "999999",
        ):
            if forbidden in text:
                raise SystemExit(f"reboot-proof matrix reply leaked private target/report detail: {text!r}")


def test_error_guidance_proof_matrix_shortcuts_are_read_only_and_private() -> None:
    """Error-guidance proof prompts should return a privacy-safe checklist, not
    trigger failing operations, run live_check, or expose private error payloads."""
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime([])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(192, 555001, "error guidance proof matrix"),
            _update(193, 555001, "error guidance proof result format"),
            _update(194, 555001, "error message proof matrix"),
            _update(195, 555001, "error message proof result format"),
            _update(196, 555001, "recovery guidance proof matrix"),
            _update(197, 555001, "user error proof matrix"),
            _update(198, 555001, "what error messages should I test?"),
            _update(199, 555001, "how should I report error guidance proof?"),
            _update(200, 555001, "오류 안내 증명 매트릭스"),
            _update(201, 555001, "오류 안내 형식"),
            _update(202, 555001, "에러 안내 증명"),
            _update(203, 555001, "do error messages name the fix"),
            _update(204, 555001, "do user errors name the fix"),
            _update(205, 555001, "are error messages actionable"),
            _update(206, 555001, "is error guidance proven"),
            _update(207, 555001, "error messages status"),
            _update(208, 555001, "error recovery status"),
            _update(209, 555001, "error recovery proof"),
            _update(210, 555001, "what error messages should I test"),
            _update(211, 555001, "how should I report error messages"),
            _update(212, 555001, "how should I report error recovery proof"),
            _update(213, 555001, "what recovery guidance should I test"),
            _update(214, 555001, "what user-facing errors need proof"),
            _update(215, 555001, "user-visible error proof matrix"),
            _update(216, 555001, "user visible error proof matrix"),
            _update(217, 555001, "connector error proof matrix"),
            _update(218, 555001, "connector recovery proof matrix"),
            _update(219, 555001, "recovery guidance status"),
            _update(220, 555001, "show recovery guidance"),
            _update(221, 555001, "show error guidance"),
            _update(222, 555001, "오류 메시지 상태"),
            _update(223, 555001, "오류 복구 상태"),
            _update(224, 555001, "오류 안내 증명"),
            _update(225, 555001, "복구 안내 증명"),
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs:
        raise SystemExit(f"error-guidance proof matrix shortcuts must not route through runtime.handle: {rt.inputs}")
    if len(sent) != 34 or any(markup is not None for _cid, _text, markup in sent):
        raise SystemExit(f"error-guidance proof matrix replies should be read-only without buttons: {sent}")
    for _cid, text, _markup in sent:
        for expected in (
            "Pending error-guidance proof matrix",
            "Calendar/Gmail setup",
            "Reminders",
            "Voice/Whisper",
            "Model/Ollama",
            "Generic connector recovery",
            "Recovery check",
            "Report format",
            "area: calendar / gmail / reminders / voice / model / connector / recovery",
            "result: pass / fail / blocked",
            "fix named: yes / no / wrong",
            "raw internals leaked: yes / no",
            "last visible stage/error",
            "optional live_check row: error guidance",
            "Privacy boundary",
            "do not include email subjects",
            "full tracebacks",
            "does not access accounts, trigger failing operations, call models",
            "claim error-guidance proof done",
        ):
            if expected not in text:
                raise SystemExit(f"error-guidance proof matrix reply lost {expected!r}: {text!r}")
        for forbidden in (
            "Fixture",
            "가상연락처일",
            "가상연락처이",
            "synthetic_owner_handle",
            "/\x55sers/",
            "/private/",
            "Traceback",
            "chat_id",
            "approve:7",
            "deny:7",
            "555001",
            "999999",
        ):
            if forbidden in text:
                raise SystemExit(f"error-guidance proof matrix reply leaked private target/report detail: {text!r}")


def test_agent_research_shortcuts_route_to_work_queue_read_only() -> None:
    """Agent-landscape research phrases should be one-step phone shortcuts to
    the existing read-only work queue guidance, not a live research or send path."""
    _seed_offset(0)
    _setup_env("555001")
    reply = (
        "Jarvis work queue:\n"
        "Agent landscape research guidance:\n"
        "- Zoey/OpenClaw/Hermes research points Jarvis toward a visible control plane, not companion personas.\n"
        "- OpenAI Agents SDK/LangGraph/CrewAI/n8n point to traces, evals, guardrails, and human-in-the-loop checks.\n"
        "- OpenClaw/Hermes always-on agents warn against persistent prompt-injection and broad ungated integrations.\n"
        "- Other Jarvis-style systems reinforce visibility before broad autonomy.\n"
        "- Keep one trusted coordinator and use internal subagents only when parallel work helps."
    )
    phrases = [
        "agent research note",
        "AI agent landscape",
        "every AI agent research",
        "three month plan",
        "three months of work",
        "Jarvis three month plan",
        "what are the three months of work",
        "what is the three month plan",
        "what is the three month plan for Jarvis",
        "what should Jarvis focus on for the next three months",
        "how do we unlock the moat",
        "what unlocks Jarvis moat",
        "what unlocks the moat",
        "what should Jarvis build before integrations",
        "when should Jarvis add integrations",
        "should Jarvis add integrations now",
        "other Jarvis models",
        "what did you find about Zoey?",
        "what did you find about other Jarvis models",
        "what should Jarvis copy from OpenClaw",
        "what should Jarvis avoid from Zoey",
        "what is the Jarvis strategy",
        "Jarvis product strategy",
        "what is the right move for Jarvis",
        "what should we do after Zoey research?",
        "should Jarvis use companions",
        "should Jarvis copy Zoey?",
        "what is Jarvis moat",
        "why no companion personas",
        "Lindy research",
        "Manus research",
        "조이 조사 결과",
        "모든 AI 에이전트 조사",
        "다른 자비스 모델",
        "자비스 차별점",
        "자비스 3개월 계획",
        "자비스 세 달 계획",
        "자비스 다음 3개월 뭐 해",
        "자비스 해자 어떻게 열어",
        "해자 어떻게 열어",
        "통합 지금 추가해도 돼",
        "통합 언제 추가해",
        "자비스 통합 언제 추가해",
        "자비스 전략",
        "자비스 방향",
        "조이 이후 뭐 만들까",
        "자비스 조이 이후 뭐 만들어",
        "조이 따라해야 해",
        "자비스 컴패니언 해야 해",
        "왜 컴패니언 안 해",
        "work queue",
    ]
    rt = FakeRuntime([reply] * len(phrases))
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(92 + index, 555001, phrase) for index, phrase in enumerate(phrases)
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs != ["work queue"] * len(phrases):
        raise SystemExit(f"agent research shortcuts should route to work queue: {rt.inputs}")
    if len(sent) != len(phrases) or any(markup is not None for _cid, _text, markup in sent):
        raise SystemExit(f"agent research shortcuts should be read-only without buttons: {sent}")
    for _cid, text, _markup in sent:
        for expected in (
            "Jarvis work queue",
            "Zoey/OpenClaw/Hermes research",
            "OpenAI Agents SDK",
            "human-in-the-loop",
            "persistent prompt-injection",
            "broad ungated integrations",
            "Other Jarvis-style systems",
            "visible control plane",
        ):
            if expected not in text:
                raise SystemExit(f"agent research shortcut lost {expected!r}: {text!r}")


def test_build_progress_and_agi_shortcuts_route_read_only() -> None:
    """Owner-phone build/AGI progress questions should route to existing
    read-only progress surfaces instead of drifting into chat or execution."""
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(
        [
            "Jarvis build progress report: current work state visible.",
            "Jarvis build progress report: current work state visible.",
            "Jarvis roadmap: next assistant layers visible.",
            "AGI gates: readiness still has proof debt.",
            "Active goals: keep Jarvis build visible.",
            "Jarvis build progress report: current work state visible.",
            "Jarvis roadmap: next assistant layers visible.",
            "AGI gates: readiness still has proof debt.",
            "Active goals: keep Jarvis build visible.",
        ]
    )
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(130, 555001, "AGI progress"),
            _update(131, 555001, "what did you build?"),
            _update(132, 555001, "roadmap"),
            _update(133, 555001, "AGI status"),
            _update(134, 555001, "goals"),
            _update(135, 555001, "AGI 진행상황"),
            _update(136, 555001, "로드맵"),
            _update(137, 555001, "AGI 상태"),
            _update(138, 555001, "목표 상태"),
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs != [
        "build progress",
        "build progress",
        "roadmap",
        "agi gates",
        "list goals",
        "build progress",
        "roadmap",
        "agi gates",
        "list goals",
    ]:
        raise SystemExit(f"build/AGI progress shortcuts routed wrong: {rt.inputs}")
    if len(sent) != 9 or any(markup is not None for _cid, _text, markup in sent):
        raise SystemExit(f"build/AGI progress shortcuts should be read-only without buttons: {sent}")
    for expected in ("build progress report", "roadmap", "AGI gates", "Active goals"):
        if not any(expected in text for _cid, text, _markup in sent):
            raise SystemExit(f"build/AGI progress shortcut lost {expected!r}: {sent}")


def test_phone_chat_latency_shortcuts_route_to_model_status_read_only() -> None:
    """Owner-phone latency questions should open the existing read-only model
    routing status packet, not chat or any tuning/execution path."""
    _seed_offset(0)
    _setup_env("555001")
    reply = (
        "Jarvis model routing status:\n"
        "Conversation latency proof:\n"
        "- Mixed-conversation acceptance target: chat p95 <= 8000ms.\n"
        "- Opt-in tuning knobs: JARVIS_CHAT_MAX_REPLY_TOKENS and JARVIS_CHAT_MAX_HISTORY_MESSAGES.\n"
        "- After any tuning, rerun the mixed-conversation proof and live_check before claiming the latency item is done."
    )
    phrases = [
        "chat latency status",
        "chat latency acceptance status",
        "chat acceptance latency",
        "chat p95",
        "chat p95 status",
        "chat p95 target",
        "why is chat slow?",
        "conversation latency",
        "conversation acceptance latency",
        "conversation p95 status",
        "why is Jarvis slow?",
        "8 second chat target",
        "8 second conversation target",
        "8s chat target",
        "8s conversation target",
        "did chat meet the 8 second target",
        "did chat pass latency",
        "did mixed conversation pass latency",
        "how do I make chat faster?",
        "is chat under 8 seconds",
        "is Jarvis under 8 seconds",
        "is mixed conversation under 8 seconds",
        "make Jarvis faster",
        "speed up chat",
        "chat speed settings",
        "chat speed acceptance status",
        "chat token cap",
        "chat reply token cap",
        "chat history window",
        "chat history messages",
        "chat max history messages",
        "chat max reply tokens",
        "chat response length status",
        "chat tuning knobs",
        "reply length status",
        "response length status",
        "reduce reply tokens",
        "reduce chat history",
        "lower reply token cap",
        "lower chat token cap",
        "lower chat history window",
        "should we lower reply tokens",
        "should we lower chat history",
        "what are the chat speed knobs?",
        "what are the chat tuning knobs?",
        "make Jarvis replies shorter",
        "conversation speed status",
        "Jarvis speed status",
        "mixed conversation p95",
        "mixed conversation p95 status",
        "mixed conversation acceptance latency",
        "mixed conversation acceptance status",
        "p95 chat target",
        "p95 latency status",
        "under 8 seconds status",
        "대화 지연 상태",
        "대화 p95 상태",
        "대화 8초 목표",
        "8초 대화 상태",
        "8초 채팅 상태",
        "채팅 p95 상태",
        "채팅 8초 목표",
        "답변 길이 상태",
        "응답 길이 상태",
        "대화 기록 상태",
        "채팅 기록 창",
        "채팅 히스토리 상태",
        "채팅 토큰 상태",
        "채팅 느려",
        "자비스 속도 상태",
        "채팅 속도 설정",
        "채팅 튜닝",
        "혼합 대화 p95 상태",
        "혼합 대화 지연 상태",
        "혼합 대화 속도 상태",
        "자비스 왜 느려",
    ]
    rt = FakeRuntime([reply for _phrase in phrases])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(340 + index, 555001, phrase) for index, phrase in enumerate(phrases)
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs != ["model routing status"] * len(phrases):
        raise SystemExit(f"chat latency shortcuts routed incorrectly: {rt.inputs}")
    if len(sent) != len(phrases) or any(markup is not None for _cid, _text, markup in sent):
        raise SystemExit(f"chat latency shortcuts should be read-only without buttons: {sent}")
    for _cid, text, _markup in sent:
        for expected in (
            "Jarvis model routing status",
            "Conversation latency proof",
            "Mixed-conversation acceptance target",
            "JARVIS_CHAT_MAX_REPLY_TOKENS",
            "JARVIS_CHAT_MAX_HISTORY_MESSAGES",
            "live_check",
        ):
            if expected not in text:
                raise SystemExit(f"chat latency shortcut lost {expected!r}: {text!r}")


def test_handoff_and_change_shortcuts_route_read_only() -> None:
    """Claude/Codex handoff and change-review phrases should be one-step
    owner-phone shortcuts to existing read-only status commands."""
    _seed_offset(0)
    _setup_env("555001")
    handoff_reply = "Jarvis handoff brief: Claude left CODEX_TASKS and the live-proof freeze in force."
    change_reply = "Jarvis change report: latest Codex patch touched read-only command discovery tests."
    phrases = [
        ("what did Claude leave for Codex?", "handoff brief", handoff_reply, "handoff brief"),
        ("what Claude said", "handoff brief", handoff_reply, "handoff brief"),
        ("check what Claude has left", "handoff brief", handoff_reply, "handoff brief"),
        ("show handoff notice", "handoff brief", handoff_reply, "handoff brief"),
        ("클로드 인수인계", "handoff brief", handoff_reply, "handoff brief"),
        ("클로드가 뭐 남겼어", "handoff brief", handoff_reply, "handoff brief"),
        ("check codex.md tasks", "work queue", "Jarvis work queue: current instructions visible.", "work queue"),
        ("what does CODEX_TASKS say?", "work queue", "Jarvis work queue: current instructions visible.", "work queue"),
        ("코덱스 작업 뭐야", "work queue", "Jarvis work queue: current instructions visible.", "work queue"),
        ("작업 큐 보여줘", "work queue", "Jarvis work queue: current instructions visible.", "work queue"),
        ("현재 작업목록", "work queue", "Jarvis work queue: current instructions visible.", "work queue"),
        ("what did Codex modify?", "what changed in Jarvis", change_reply, "change report"),
        ("what changed after Codex?", "what changed in Jarvis", change_reply, "change report"),
        ("뭐 수정했어", "what changed in Jarvis", change_reply, "change report"),
        ("최근 변경사항", "what changed in Jarvis", change_reply, "change report"),
    ]
    rt = FakeRuntime([reply for _phrase, _command, reply, _expected in phrases])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(120 + index, 555001, phrase) for index, (phrase, _command, _reply, _expected) in enumerate(phrases)
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    expected_inputs = [command for _phrase, command, _reply, _expected in phrases]
    if rt.inputs != expected_inputs:
        raise SystemExit(f"handoff/change shortcuts routed incorrectly: {rt.inputs}")
    if len(sent) != len(phrases) or any(markup is not None for _cid, _text, markup in sent):
        raise SystemExit(f"handoff/change shortcuts should be read-only without buttons: {sent}")
    for (_cid, text, _markup), (_phrase, _command, _reply, expected) in zip(sent, phrases, strict=True):
        if expected not in text:
            raise SystemExit(f"handoff/change shortcut lost {expected!r}: {text!r}")


def test_completion_status_shortcuts_route_to_claim_gate_read_only() -> None:
    """Phone questions about whether Jarvis is done should route to the
    read-only completion claim gate, never to chat optimism or approval."""
    _seed_offset(0)
    _setup_env("555001")
    reply = (
        "Completion claim gate:\n"
        "- state: BLOCKED\n"
        "- reason: live channel proofs and learning/recovery proof debt remain."
    )
    phrases = (
        "is Jarvis done?",
        "completion status",
        "can Jarvis claim completion?",
        "자비스 끝났어?",
        "why isn't Jarvis done?",
        "what blocks completion?",
        "what is blocking completion?",
        "when is Jarvis done?",
        "완료 뭐 막혀",
        "자비스 왜 아직 안 끝났어?",
    )
    rt = FakeRuntime([reply for _phrase in phrases])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(96 + index, 555001, phrase) for index, phrase in enumerate(phrases)
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    expected_inputs = ["completion claim gate" for _phrase in phrases]
    if rt.inputs != expected_inputs:
        raise SystemExit(f"completion shortcuts should route to the claim gate: {rt.inputs}")
    if len(sent) != len(phrases) or any(markup is not None for _cid, _text, markup in sent):
        raise SystemExit(f"completion shortcuts should be read-only without buttons: {sent}")
    for _cid, text, _markup in sent:
        for expected in ("Completion claim gate", "BLOCKED", "proof debt"):
            if expected not in text:
                raise SystemExit(f"completion shortcut lost {expected!r}: {text!r}")


def test_next_cockpit_action_shortcuts_are_read_only() -> None:
    """The owner-phone "next action" shortcut should surface the cockpit queue
    without executing the queued command or creating approval buttons."""
    _seed_offset(0)
    _setup_env("555001")

    class CockpitTool:
        def handler(self, args):
            return SimpleNamespace(
                output="Capability cockpit (derived...)",
                metadata={
                    "next_command": "channel health",
                    "next_command_queue": [
                        {
                            "command": "channel health",
                            "kind": "diagnostic",
                            "lane_title": "Messaging (KR/EN)",
                            "status": "attention",
                            "last_failure": "send_telegram @ 2026-07-06T05:20:00Z",
                            "last_failure_kind": "transport_error",
                        },
                        {
                            "command": "approval readiness 7",
                            "kind": "approval_review",
                            "lane_title": "Approvals",
                            "status": "awaiting approval",
                            "last_approval_hold": "send_telegram @ 2026-07-06T05:21:00Z",
                            "approval_next_commands": [
                                "approval readiness 7",
                                "approval packet 7",
                                "approval chain proof 7",
                            ],
                        },
                    ],
                },
            )

    class CockpitRegistry:
        def get(self, name):
            if name != "capability_cockpit":
                raise KeyError(name)
            return CockpitTool()

    rt = FakeRuntime(["should not run"])
    rt.registry = CockpitRegistry()
    phrases = [
        "next cockpit action",
        "what should I do next?",
        "what is Jarvis next step",
        "what is next for Jarvis",
        "what's next for Jarvis",
        "what should I run now",
        "what command should I run now",
        "what should I check now",
        "what should I check next",
        "what should I run next",
        "what command should I run next",
        "what should Jarvis do next",
        "다음 행동",
        "다음 뭐 해야 해",
        "자비스 다음 뭐 해",
        "자비스 다음 단계",
        "자비스 뭐부터 해",
        "다음에 뭘 실행해",
        "무슨 명령 실행해",
    ]
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(92 + index, 555001, phrase) for index, phrase in enumerate(phrases)
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs:
        raise SystemExit(f"next cockpit action must not route through runtime.handle: {rt.inputs}")
    if len(sent) != len(phrases) or any(markup is not None for _cid, _text, markup in sent):
        raise SystemExit(f"next cockpit action replies should be read-only without buttons: {sent}")
    for _cid, text, _markup in sent:
        if "Next cockpit action" not in text or "channel health" not in text:
            raise SystemExit(f"next cockpit action did not surface the queued command: {text!r}")
        if "Messaging (KR/EN)" not in text or "diagnostic" not in text:
            raise SystemExit(f"next cockpit action lost queue context: {text!r}")
        if "approval readiness 7" not in text:
            raise SystemExit(f"next cockpit action should include the second queued review command: {text!r}")
        for expected in (
            "last failure: send_telegram @ 2026-07-06T05:20:00Z (transport_error)",
            "approval hold: send_telegram @ 2026-07-06T05:21:00Z",
            "review: approval readiness 7, approval packet 7",
        ):
            if expected not in text:
                raise SystemExit(f"next cockpit action should include health context {expected!r}: {text!r}")
        if "does not approve" not in text or "execute" not in text:
            raise SystemExit(f"next cockpit action should declare the read-only boundary: {text!r}")


def test_next_cockpit_action_metadata_fails_closed() -> None:
    """Malformed queued cockpit commands should not break or leak through the
    owner-phone next-action shortcut."""
    _seed_offset(0)
    _setup_env("555001")

    leak_marker = "SHOULD NOT APPEAR"

    class HostileText:
        def __str__(self) -> str:
            raise RuntimeError(f"{leak_marker} /\x55sers/hostile/cockpit-next-action")

    class CockpitTool:
        def handler(self, args):
            return SimpleNamespace(
                output="Capability cockpit (derived...)",
                metadata={
                    "next_command": f"{leak_marker} /\x55sers/hostile/top-next-command",
                    "next_command_queue": [
                        {
                            "command": f"{leak_marker} /\x55sers/hostile/cockpit-next-command",
                            "kind": HostileText(),
                            "lane_title": f"{leak_marker} /TMP/hostile-next-lane",
                            "status": "attention",
                            "last_failure": HostileText(),
                            "last_failure_kind": f"{leak_marker} /private/hostile-next-kind",
                            "last_approval_hold": f"{leak_marker} /var/folders/hostile-next-hold",
                            "approval_next_commands": [
                                HostileText(),
                                f"{leak_marker} /\x55sers/hostile/approval-next-command",
                            ],
                        },
                        {
                            "command": "channel health",
                            "kind": "diagnostic",
                            "lane_title": "Messaging (KR/EN)",
                            "status": "attention",
                        },
                    ],
                },
            )

    class CockpitRegistry:
        def get(self, name):
            if name != "capability_cockpit":
                raise KeyError(name)
            return CockpitTool()

    rt = FakeRuntime(["should not run"])
    rt.registry = CockpitRegistry()
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [_update(95, 555001, "next action")],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs:
        raise SystemExit(f"malformed next action must not route through runtime.handle: {rt.inputs}")
    if len(sent) != 1 or sent[0][2] is not None:
        raise SystemExit(f"malformed next action should be read-only without buttons: {sent}")
    text = sent[0][1]
    for expected in (
        "Next cockpit action",
        "1. <redacted-local-path>",
        "<unreadable>",
        "channel health",
        "Messaging (KR/EN)",
        "This is read-only",
    ):
        if expected not in text:
            raise SystemExit(f"malformed next action lost safe context {expected!r}: {text!r}")
    for leaked in (
        leak_marker,
        "/\x55sers/hostile",
        "/TMP/hostile",
        "/private/hostile",
        "/var/folders/hostile",
        "cockpit-next-action",
    ):
        if leaked in text:
            raise SystemExit(f"malformed next action leaked hostile queue metadata: {text!r}")


def test_cockpit_attention_shortcuts_are_read_only() -> None:
    """The owner-phone attention shortcut should show cockpit lanes needing
    review without executing their diagnostics or approval-review commands."""
    _seed_offset(0)
    _setup_env("555001")

    class CockpitTool:
        def handler(self, args):
            return SimpleNamespace(
                output="Capability cockpit (derived...)",
                metadata={
                    "capability_lanes": [
                        {
                            "title": "Messaging (KR/EN)",
                            "status": "attention",
                            "next_command": "channel health",
                            "attention_reasons": ["latest failure: transport_error"],
                            "last_failure": "send_telegram @ 2026-07-06T12:00:00+09:00",
                            "last_failure_kind": "transport_error",
                            "trust_summary": "4/4 checks ready",
                            "trust_checklist": [
                                {"label": "registered tools", "ready": True},
                                {"label": "aggregate smoke coverage", "ready": True},
                                {"label": "approval boundary", "ready": True},
                                {"label": "next command", "ready": True},
                            ],
                        },
                        {
                            "title": "Approvals",
                            "status": "awaiting approval",
                            "next_command": "approval readiness 7",
                            "last_approval_hold": "send_telegram @ 2026-07-06T12:01:00+09:00",
                            "approval_next_commands": ["approval readiness 7", "approval packet 7"],
                        },
                        {
                            "title": "Scheduler",
                            "status": "ok",
                            "next_command": "list scheduled jobs",
                        },
                    ]
                },
            )

    class CockpitRegistry:
        def get(self, name):
            if name != "capability_cockpit":
                raise KeyError(name)
            return CockpitTool()

    rt = FakeRuntime(["should not run"])
    rt.registry = CockpitRegistry()
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(95, 555001, "cockpit attention"),
            _update(96, 555001, "what needs attention?"),
            _update(97, 555001, "주의 상태"),
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs:
        raise SystemExit(f"cockpit attention must not route through runtime.handle: {rt.inputs}")
    if len(sent) != 3 or any(markup is not None for _cid, _text, markup in sent):
        raise SystemExit(f"cockpit attention replies should be read-only without buttons: {sent}")
    for _cid, text, _markup in sent:
        if "Cockpit attention" not in text or "Messaging (KR/EN)" not in text:
            raise SystemExit(f"attention shortcut did not surface the attention lane: {text!r}")
        if "Needs attention:" not in text or "Approval-held review:" not in text:
            raise SystemExit(f"attention shortcut should separate failures from approval-held review: {text!r}")
        for expected in (
            "Approval-held means Jarvis is waiting for owner review",
            "not counted as a tool failure",
        ):
            if expected not in text:
                raise SystemExit(f"attention shortcut lost approval-held safety semantics {expected!r}: {text!r}")
        if "channel health" not in text or "transport_error" not in text:
            raise SystemExit(f"attention shortcut lost diagnostic context: {text!r}")
        if "Approvals" not in text or "approval readiness 7" not in text:
            raise SystemExit(f"attention shortcut should include approval-review context: {text!r}")
        if "Scheduler" in text:
            raise SystemExit(f"attention shortcut should omit healthy lanes from compact view: {text!r}")
        if "does not approve" not in text or "execute" not in text:
            raise SystemExit(f"attention shortcut should declare the read-only boundary: {text!r}")


def test_cockpit_attention_metadata_fails_closed() -> None:
    """Malformed cockpit attention metadata should not break or leak through
    the owner-phone attention shortcut."""
    _seed_offset(0)
    _setup_env("555001")

    leak_marker = "SHOULD NOT APPEAR"

    class HostileText:
        def __str__(self) -> str:
            raise RuntimeError(f"{leak_marker} /\x55sers/hostile/cockpit-attention")

    class CockpitTool:
        def handler(self, args):
            return SimpleNamespace(
                output="Capability cockpit (derived...)",
                metadata={
                    "capability_lanes": [
                        {
                            "title": f"{leak_marker} /\x55sers/hostile/attention-title",
                            "status": "attention",
                            "next_command": f"{leak_marker} /TMP/hostile-attention-command",
                            "attention_reasons": [
                                HostileText(),
                                f"{leak_marker} /private/hostile-attention-reason",
                            ],
                            "last_failure": HostileText(),
                            "last_failure_kind": f"{leak_marker} /var/folders/hostile-attention-kind",
                        },
                        {
                            "title": "Approvals",
                            "status": "awaiting approval",
                            "next_command": "approval readiness 7",
                            "last_approval_hold": f"{leak_marker} /\x55sers/hostile/attention-hold",
                            "approval_next_commands": [
                                HostileText(),
                                f"{leak_marker} /TMP/hostile-attention-review",
                            ],
                        },
                        {
                            "title": "Diagnostics",
                            "status": "attention",
                            "next_command": "jarvis doctor",
                            "attention_reasons": ["Normal visible-control reason"],
                        },
                    ]
                },
            )

    class CockpitRegistry:
        def get(self, name):
            if name != "capability_cockpit":
                raise KeyError(name)
            return CockpitTool()

    rt = FakeRuntime(["should not run"])
    rt.registry = CockpitRegistry()
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [_update(98, 555001, "cockpit attention")],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs:
        raise SystemExit(f"malformed cockpit attention must not route through runtime.handle: {rt.inputs}")
    if len(sent) != 1 or sent[0][2] is not None:
        raise SystemExit(f"malformed cockpit attention should be read-only without buttons: {sent}")
    text = sent[0][1]
    for expected in (
        "Cockpit attention",
        "Needs attention:",
        "Approval-held review:",
        "<redacted-local-path>",
        "<unreadable>",
        "jarvis doctor",
        "Normal visible-control reason",
        "This is read-only",
    ):
        if expected not in text:
            raise SystemExit(f"malformed cockpit attention lost safe context {expected!r}: {text!r}")
    for leaked in (
        leak_marker,
        "/\x55sers/hostile",
        "/TMP/hostile",
        "/private/hostile",
        "/var/folders/hostile",
        "cockpit-attention",
    ):
        if leaked in text:
            raise SystemExit(f"malformed cockpit attention leaked hostile metadata: {text!r}")


def test_cockpit_lane_shortcuts_are_read_only() -> None:
    """Single-lane cockpit views should be compact and read-only, and must not
    steal bare full-cockpit commands like "guardrails"."""
    _seed_offset(0)
    _setup_env("555001")

    class CockpitTool:
        def handler(self, args):
            return SimpleNamespace(
                output="Capability cockpit: full cockpit output with Build Guardrails.",
                metadata={
                    "capability_lanes": [
                        {
                            "key": "messaging",
                            "title": "Messaging (KR/EN)",
                            "status": "attention",
                            "risk": "HIGH_RISK",
                            "approval_required": True,
                            "tool_coverage": "registered",
                            "smoke_coverage": "registered",
                            "next_command": "channel health",
                            "next_command_kind": "diagnostic",
                            "example_command": "send 가상연락처이 a telegram saying hello",
                            "attention_reasons": ["latest failure: transport_error"],
                            "last_failure": "send_telegram @ 2026-07-06T12:00:00+09:00",
                            "last_failure_kind": "transport_error",
                            "trust_summary": "4/4 checks ready",
                            "trust_checklist": [
                                {"label": "registered tools", "ready": True},
                                {"label": "aggregate smoke coverage", "ready": True},
                                {"label": "approval boundary", "ready": True},
                                {"label": "next command", "ready": True},
                            ],
                        },
                        {
                            "key": "approvals",
                            "title": "Approvals",
                            "status": "awaiting approval",
                            "risk": "HIGH_RISK",
                            "approval_required": True,
                            "tool_coverage": "registered",
                            "smoke_coverage": "registered",
                            "next_command": "approval readiness 7",
                            "next_command_kind": "approval_review",
                            "last_approval_hold": "send_telegram @ 2026-07-06T12:01:00+09:00",
                            "approval_next_commands": ["approval readiness 7", "approval packet 7"],
                        },
                        {
                            "key": "calls",
                            "title": "Calls",
                            "status": "attention",
                            "risk": "HIGH_RISK",
                            "approval_required": True,
                            "tool_coverage": "registered",
                            "smoke_coverage": "registered",
                            "next_command": "channel health",
                            "next_command_kind": "diagnostic",
                            "example_command": "call 가상연락처일 on telegram",
                            "attention_reasons": ["live proof pending"],
                            "proof_points": [
                                "call tools remain HIGH_RISK and approval-gated",
                                "channel health reports call delivery state without dialing",
                            ],
                        },
                        {
                            "key": "voice",
                            "title": "Voice",
                            "status": "ok",
                            "risk": "READ_ONLY",
                            "approval_required": False,
                            "tool_coverage": "registered",
                            "smoke_coverage": "registered",
                            "next_command": "voice setup check",
                            "next_command_kind": "example",
                        },
                        {
                            "key": "writing",
                            "title": "Writing",
                            "status": "ready",
                            "risk": "HIGH_RISK",
                            "approval_required": True,
                            "tool_coverage": "registered",
                            "smoke_coverage": "registered",
                            "next_command": "write a paragraph about Jarvis and type it",
                            "next_command_kind": "example",
                            "example_command": "write a paragraph about Jarvis and type it",
                            "proof_points": [
                                "compose_and_write stops before typing or pasting when model generation fails",
                                "human_write and paste_text remain HIGH_RISK and approval-gated before computer control",
                            ],
                            "guardrail_notes": [
                                "draft-to-write flows are inspectable, approval-gated, and covered by mocked writer/compose smokes"
                            ],
                            "trust_summary": "5/5 checks ready",
                            "trust_checklist": [
                                {"label": "registered tools", "ready": True},
                                {"label": "aggregate smoke coverage", "ready": True},
                                {"label": "approval boundary", "ready": True},
                                {"label": "next command", "ready": True},
                                {"label": "explicit proof points", "ready": True},
                            ],
                        },
                        {
                            "key": "morning_brief",
                            "title": "Morning Brief",
                            "status": "ready",
                            "risk": "EXTERNAL_SIDE_EFFECT",
                            "approval_required": True,
                            "tool_coverage": "registered",
                            "smoke_coverage": "registered",
                            "next_command": "run job morning brief now",
                            "next_command_kind": "example",
                            "example_command": "run job morning brief now",
                            "proof_points": ["on-demand Morning Brief is covered by mocked owner-phone smokes"],
                        },
                        {
                            "key": "contacts",
                            "title": "Contacts & People",
                            "status": "ready",
                            "risk": "LOCAL_SAFE",
                            "approval_required": False,
                            "tool_coverage": "registered",
                            "smoke_coverage": "registered",
                            "next_command": "find contact fixture",
                            "next_command_kind": "example",
                            "example_command": "who is 가상연락처이",
                            "proof_points": ["contact lookup is covered by fuzzy and connector smokes"],
                        },
                        {
                            "key": "calendar_email",
                            "title": "Calendar & Email",
                            "status": "ready",
                            "risk": "HIGH_RISK",
                            "approval_required": True,
                            "tool_coverage": "registered",
                            "smoke_coverage": "registered",
                            "next_command": "read my recent emails",
                            "next_command_kind": "example",
                            "example_command": "read my recent emails",
                            "proof_points": ["calendar and email connectors expose mocked setup/recovery smokes"],
                        },
                        {
                            "key": "personal_proofs",
                            "title": "Personal Proofs",
                            "status": "ready",
                            "risk": "HIGH_RISK",
                            "approval_required": True,
                            "tool_coverage": "registered",
                            "smoke_coverage": "registered",
                            "next_command": "personal proofs status",
                            "next_command_kind": "example",
                            "example_command": "personal proofs status",
                            "proof_points": [
                                "personal proofs are still live-acceptance evidence, not mocked completion claims",
                                "set_reminder proof uses Jarvis's local Telegram reminder path",
                            ],
                            "guardrail_notes": [
                                "read-only proof lane for WS2: calendar writes, email read/search/send, set_reminder delivery, and contact lookup",
                                "run opt-in live_check with operator present for actual proof; this lane does not access accounts or send anything",
                            ],
                        },
                        {
                            "key": "markets",
                            "title": "Markets & Weather",
                            "status": "ready",
                            "risk": "LOCAL_SAFE",
                            "approval_required": False,
                            "tool_coverage": "registered",
                            "smoke_coverage": "registered",
                            "next_command": "markets overview",
                            "next_command_kind": "example",
                            "example_command": "markets overview",
                            "proof_points": ["markets, weather, and currency connectors expose mocked recovery smokes"],
                        },
                        {
                            "key": "research_web",
                            "title": "Research & Web",
                            "status": "ready",
                            "risk": "LOCAL_SAFE",
                            "approval_required": False,
                            "tool_coverage": "registered",
                            "smoke_coverage": "registered",
                            "next_command": "research Zoey OS",
                            "next_command_kind": "example",
                            "example_command": "research Zoey OS",
                            "proof_points": [
                                "research and web_lookup emit no-authority handoff packets with content excluded from metadata",
                                "live_check verifies research/web registration, risk gates, and route shape without fetching the web or calling models",
                            ],
                            "guardrail_notes": [
                                "research may call external search/page services and the local model only when explicitly run",
                                "the cockpit lane itself is read-only and does not fetch pages, synthesize answers, write notes, or queue approvals",
                            ],
                        },
                        {
                            "key": "memory",
                            "title": "Memory & Notes",
                            "status": "ready",
                            "risk": "LOCAL_SAFE",
                            "approval_required": False,
                            "tool_coverage": "registered",
                            "smoke_coverage": "registered",
                            "next_command": "memory stats",
                            "next_command_kind": "diagnostic",
                            "example_command": "remember that the operator prefers direct answers",
                            "proof_points": ["core memory and memory stats are covered by mocked storage smokes"],
                        },
                        {
                            "key": "learning_loop",
                            "title": "Learning Loop",
                            "status": "ready",
                            "risk": "LOCAL_SAFE",
                            "approval_required": False,
                            "tool_coverage": "registered",
                            "smoke_coverage": "registered",
                            "next_command": "learning review",
                            "next_command_kind": "diagnostic",
                            "example_command": "learning review",
                            "proof_points": [
                                "learning_review surfaces feedback, memory health, preferences, skills, and execution learning debt without saving",
                                "after_action_learning_packet and execution_learning_closure_packet bind approved runs to verification, audit, recovery, and learning evidence",
                                "feedback_actions and failure_learning_cockpit turn repeated failures into reviewable fixes instead of silent behavior drift",
                            ],
                            "guardrail_notes": [
                                "learning is a visible review loop, not autonomous self-modification",
                                "saving reviews or queueing learning tasks remains local-safe and non-authorizing; risky execution still needs approvals",
                            ],
                        },
                        {
                            "key": "diagnostics",
                            "title": "Diagnostics",
                            "status": "ready",
                            "risk": "READ_ONLY",
                            "approval_required": False,
                            "tool_coverage": "registered",
                            "smoke_coverage": "registered",
                            "next_command": "jarvis doctor",
                            "next_command_kind": "diagnostic",
                            "example_command": "what broke",
                            "proof_points": ["doctor, audit, and channel-health smokes expose recovery state without side effects"],
                        },
                        {
                            "key": "operator_workflow_evals",
                            "title": "Operator Workflow Evals",
                            "status": "ready",
                            "risk": "HIGH_RISK",
                            "approval_required": True,
                            "tool_coverage": "registered",
                            "smoke_coverage": "registered",
                            "next_command": "channel health",
                            "next_command_kind": "diagnostic",
                            "example_command": "push today's brief to my phone",
                            "proof_points": [
                                "Korean Telegram send stops at approval with Hangul recipient/body intact",
                                "Morning Brief can be pushed on demand to the owner phone channel",
                            ],
                        },
                        {
                            "key": "scheduler",
                            "title": "Scheduled Jobs",
                            "status": "ready",
                            "risk": "EXTERNAL_SIDE_EFFECT",
                            "approval_required": True,
                            "tool_coverage": "registered",
                            "smoke_coverage": "registered",
                            "next_command": "list scheduled jobs",
                            "next_command_kind": "example",
                            "example_command": "list scheduled jobs",
                            "proof_points": ["scheduler visibility is covered by mocked scheduled-job smokes"],
                        },
                        {
                            "key": "build_guardrails",
                            "title": "Build Guardrails",
                            "status": "ready",
                            "risk": "READ_ONLY",
                            "approval_required": False,
                            "tool_coverage": "registered",
                            "smoke_coverage": "registered",
                            "next_command": "readiness report",
                            "next_command_kind": "example",
                            "guardrail_notes": ["live-proof freeze active"],
                        },
                        {
                            "key": "orchestration",
                            "title": "Internal Orchestration",
                            "status": "ready",
                            "risk": "READ_ONLY",
                            "approval_required": False,
                            "tool_coverage": "registered",
                            "smoke_coverage": "registered",
                            "next_command": "jarvis status",
                            "next_command_kind": "example",
                            "example_command": "jarvis status",
                            "proof_points": [
                                "subagent_fleet_status exposes worker readiness and capacity without spawning tasks",
                                "subagents stay internal orchestration capacity, not user-facing companion personas",
                                "subagent smoke coverage pins readiness without granting autonomy, approvals, or tool execution",
                            ],
                            "guardrail_notes": [
                                "internal subagents are workers, not companion personas",
                                "worker orchestration does not approve, dispatch, or run risky actions",
                            ],
                        },
                    ]
                },
            )

    class CockpitRegistry:
        def get(self, name):
            if name != "capability_cockpit":
                raise KeyError(name)
            return CockpitTool()

    rt = FakeRuntime(["should not run"])
    rt.registry = CockpitRegistry()
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(98, 555001, "messaging lane"),
            _update(99, 555001, "voice lane"),
            _update(100, 555001, "메시지 상태"),
            _update(101, 555001, "가드레일 레인"),
            _update(102, 555001, "approvals lane"),
            _update(103, 555001, "승인 상태"),
            _update(104, 555001, "승인 레인"),
            _update(105, 555001, "agent status"),
            _update(106, 555001, "worker status"),
            _update(107, 555001, "에이전트 상태"),
            _update(108, 555001, "워커 상태"),
            _update(109, 555001, "guardrails"),
            _update(110, 555001, "writing lane"),
            _update(111, 555001, "typing status"),
            _update(112, 555001, "글쓰기 상태"),
            _update(113, 555001, "morning brief lane"),
            _update(114, 555001, "contacts lane"),
            _update(115, 555001, "calendar email lane"),
            _update(116, 555001, "scheduler lane"),
            _update(117, 555001, "브리핑 레인"),
            _update(118, 555001, "연락처 레인"),
            _update(119, 555001, "캘린더 이메일 레인"),
            _update(120, 555001, "스케줄 레인"),
            _update(121, 555001, "calls lane"),
            _update(122, 555001, "통화 레인"),
            _update(123, 555001, "전화 레인"),
            _update(124, 555001, "음성 레인"),
            _update(125, 555001, "마이크 레인"),
            _update(126, 555001, "markets lane"),
            _update(127, 555001, "weather lane"),
            _update(128, 555001, "시장 레인"),
            _update(129, 555001, "날씨 레인"),
            _update(130, 555001, "research lane"),
            _update(131, 555001, "web lane"),
            _update(132, 555001, "검색 레인"),
            _update(133, 555001, "연구 레인"),
            _update(134, 555001, "memory lane"),
            _update(135, 555001, "diagnostics lane"),
            _update(136, 555001, "operator evals lane"),
            _update(137, 555001, "workflow evals lane"),
            _update(138, 555001, "기억 레인"),
            _update(139, 555001, "진단 레인"),
            _update(140, 555001, "제임스 평가 레인"),
            _update(141, 555001, "워크플로우 평가 레인"),
            _update(142, 555001, "personal proofs lane"),
            _update(143, 555001, "개인 증명 레인"),
            _update(144, 555001, "learning lane"),
            _update(145, 555001, "학습 레인"),
            _update(146, 555001, "orchestration lane"),
            _update(147, 555001, "worker lane"),
            _update(148, 555001, "워커 레인"),
            _update(149, 555001, "에이전트 레인"),
            _update(150, 555001, "subagent health"),
            _update(151, 555001, "subagent fleet readiness"),
            _update(152, 555001, "agent health"),
            _update(153, 555001, "are my agents ready"),
            _update(154, 555001, "ready agent count"),
            _update(155, 555001, "tool orchestration status"),
            _update(156, 555001, "parallel agents status"),
            _update(157, 555001, "worker health"),
            _update(158, 555001, "worker readiness"),
            _update(159, 555001, "내부 워커 상태"),
            _update(160, 555001, "에이전트 준비 상태"),
            _update(161, 555001, "워커 준비 상태"),
            _update(162, 555001, "병렬 에이전트 상태"),
            _update(163, 555001, "서브에이전트 준비 상태"),
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs:
        raise SystemExit(f"cockpit lane shortcuts must not route through runtime.handle: {rt.inputs}")
    if len(sent) != 66 or any(markup is not None for _cid, _text, markup in sent):
        raise SystemExit(f"cockpit lane replies should be read-only without buttons: {sent}")
    messaging_reply = sent[0][1]
    if "Cockpit lane: Messaging (KR/EN)" not in messaging_reply or "channel health" not in messaging_reply:
        raise SystemExit(f"messaging lane did not show its cockpit detail: {messaging_reply!r}")
    if "HIGH_RISK" not in messaging_reply or "transport_error" not in messaging_reply:
        raise SystemExit(f"messaging lane lost risk/failure context: {messaging_reply!r}")
    if "trust: 4/4 checks ready" not in messaging_reply or "approval boundary: ok" not in messaging_reply:
        raise SystemExit(f"messaging lane lost earned-trust checklist context: {messaging_reply!r}")
    voice_reply = sent[1][1]
    if "Cockpit lane: Voice" not in voice_reply or "voice setup check" not in voice_reply:
        raise SystemExit(f"voice lane did not show its cockpit detail: {voice_reply!r}")
    for korean_voice_reply in (sent[26][1], sent[27][1]):
        for expected in ("Cockpit lane: Voice", "READ_ONLY", "voice setup check", "This is read-only"):
            if expected not in korean_voice_reply:
                raise SystemExit(f"Korean voice lane shortcut lost context {expected!r}: {korean_voice_reply!r}")
    korean_reply = sent[2][1]
    if "Cockpit lane: Messaging (KR/EN)" not in korean_reply:
        raise SystemExit(f"Korean lane alias should map to messaging: {korean_reply!r}")
    guardrail_lane_reply = sent[3][1]
    if "Cockpit lane: Build Guardrails" not in guardrail_lane_reply or "live-proof freeze active" not in guardrail_lane_reply:
        raise SystemExit(f"guardrail lane did not show guardrail context: {guardrail_lane_reply!r}")
    approvals_reply = sent[4][1]
    for expected in (
        "Cockpit lane: Approvals",
        "approval hold: send_telegram @ 2026-07-06T12:01:00+09:00",
        "approval readiness 7",
        "not counted as a tool failure",
    ):
        if expected not in approvals_reply:
            raise SystemExit(f"approvals lane lost approval-held context {expected!r}: {approvals_reply!r}")
    for korean_approvals_reply in (sent[5][1], sent[6][1]):
        for expected in (
            "Cockpit lane: Approvals",
            "approval readiness 7",
            "not counted as a tool failure",
        ):
            if expected not in korean_approvals_reply:
                raise SystemExit(
                    f"Korean approvals lane lost approval-held context {expected!r}: {korean_approvals_reply!r}"
                )
    orchestration_replies = [sent[index][1] for index in [7, 8, 9, 10, *range(52, 66)]]
    for orchestration_reply in orchestration_replies:
        for expected in (
            "Cockpit lane: Internal Orchestration",
            "READ_ONLY",
            "jarvis status",
            "internal subagents are workers",
        ):
            if expected not in orchestration_reply:
                raise SystemExit(
                    f"orchestration lane lost read-only worker context {expected!r}: {orchestration_reply!r}"
                )
        for expected in (
            "worker readiness and capacity without spawning tasks",
            "not user-facing companion personas",
            "without granting autonomy, approvals, or tool execution",
            "does not approve, dispatch, or run risky actions",
        ):
            if expected not in orchestration_reply:
                raise SystemExit(
                    f"orchestration lane lost proof/boundary context {expected!r}: {orchestration_reply!r}"
                )
    bare_guardrails_reply = sent[11][1]
    if "full cockpit output" not in bare_guardrails_reply or "Cockpit lane:" in bare_guardrails_reply:
        raise SystemExit(f"bare guardrails should still return the full cockpit: {bare_guardrails_reply!r}")
    for writing_reply in (sent[12][1], sent[13][1], sent[14][1]):
        for expected in (
            "Cockpit lane: Writing",
            "HIGH_RISK",
            "approval required: yes",
            "write a paragraph about Jarvis and type it",
            "compose_and_write stops before typing",
            "approval boundary: ok",
            "This is read-only",
        ):
            if expected not in writing_reply:
                raise SystemExit(f"writing lane shortcut lost context {expected!r}: {writing_reply!r}")
    everyday_expectations = (
        (sent[15][1], "Cockpit lane: Morning Brief", "run job morning brief now", "EXTERNAL_SIDE_EFFECT"),
        (sent[16][1], "Cockpit lane: Contacts & People", "find contact fixture", "LOCAL_SAFE"),
        (sent[17][1], "Cockpit lane: Calendar & Email", "read my recent emails", "HIGH_RISK"),
        (sent[18][1], "Cockpit lane: Scheduled Jobs", "list scheduled jobs", "EXTERNAL_SIDE_EFFECT"),
        (sent[19][1], "Cockpit lane: Morning Brief", "run job morning brief now", "EXTERNAL_SIDE_EFFECT"),
        (sent[20][1], "Cockpit lane: Contacts & People", "find contact fixture", "LOCAL_SAFE"),
        (sent[21][1], "Cockpit lane: Calendar & Email", "read my recent emails", "HIGH_RISK"),
        (sent[22][1], "Cockpit lane: Scheduled Jobs", "list scheduled jobs", "EXTERNAL_SIDE_EFFECT"),
    )
    for reply, title, command, risk in everyday_expectations:
        for expected in (title, command, risk, "smoke: registered", "This is read-only"):
            if expected not in reply:
                raise SystemExit(f"everyday lane shortcut lost context {expected!r}: {reply!r}")
    for calls_reply in (sent[23][1], sent[24][1], sent[25][1]):
        for expected in (
            "Cockpit lane: Calls",
            "HIGH_RISK",
            "approval required: yes",
            "channel health",
            "live proof pending",
            "call tools remain HIGH_RISK",
            "This is read-only",
        ):
            if expected not in calls_reply:
                raise SystemExit(f"calls lane shortcut lost context {expected!r}: {calls_reply!r}")
    for markets_reply in (sent[28][1], sent[29][1], sent[30][1], sent[31][1]):
        for expected in (
            "Cockpit lane: Markets & Weather",
            "LOCAL_SAFE",
            "approval required: no",
            "markets overview",
            "markets, weather, and currency connectors expose mocked recovery smokes",
            "This is read-only",
        ):
            if expected not in markets_reply:
                raise SystemExit(f"markets/weather lane shortcut lost context {expected!r}: {markets_reply!r}")
    for research_reply in (sent[32][1], sent[33][1], sent[34][1], sent[35][1]):
        for expected in (
            "Cockpit lane: Research & Web",
            "LOCAL_SAFE",
            "approval required: no",
            "research Zoey OS",
            "research and web_lookup emit no-authority handoff packets",
            "does not fetch pages, synthesize answers, write notes, or queue approvals",
            "This is read-only",
        ):
            if expected not in research_reply:
                raise SystemExit(f"research/web lane shortcut lost context {expected!r}: {research_reply!r}")
    for memory_reply in (sent[36][1], sent[40][1]):
        for expected in (
            "Cockpit lane: Memory & Notes",
            "LOCAL_SAFE",
            "approval required: no",
            "memory stats",
            "core memory and memory stats are covered by mocked storage smokes",
            "This is read-only",
        ):
            if expected not in memory_reply:
                raise SystemExit(f"memory lane shortcut lost context {expected!r}: {memory_reply!r}")
    for diagnostics_reply in (sent[37][1], sent[41][1]):
        for expected in (
            "Cockpit lane: Diagnostics",
            "READ_ONLY",
            "approval required: no",
            "jarvis doctor",
            "doctor, audit, and channel-health smokes expose recovery state",
            "This is read-only",
        ):
            if expected not in diagnostics_reply:
                raise SystemExit(f"diagnostics lane shortcut lost context {expected!r}: {diagnostics_reply!r}")
    for eval_reply in (sent[38][1], sent[39][1], sent[42][1], sent[43][1]):
        for expected in (
            "Cockpit lane: Operator Workflow Evals",
            "HIGH_RISK",
            "approval required: yes",
            "channel health",
            "Korean Telegram send stops at approval",
            "Morning Brief can be pushed on demand",
            "This is read-only",
        ):
            if expected not in eval_reply:
                raise SystemExit(f"workflow evals lane shortcut lost context {expected!r}: {eval_reply!r}")
    for personal_reply in (sent[44][1], sent[45][1]):
        for expected in (
            "Cockpit lane: Personal Proofs",
            "HIGH_RISK",
            "approval required: yes",
            "personal proofs status",
            "personal proofs are still live-acceptance evidence",
            "set_reminder proof uses Jarvis's local Telegram reminder path",
            "does not access accounts or send anything",
            "This is read-only",
        ):
            if expected not in personal_reply:
                raise SystemExit(f"personal proofs lane shortcut lost context {expected!r}: {personal_reply!r}")
    for learning_reply in (sent[46][1], sent[47][1]):
        for expected in (
            "Cockpit lane: Learning Loop",
            "LOCAL_SAFE",
            "approval required: no",
            "learning review",
            "learning_review surfaces feedback",
            "bind approved runs to verification, audit, recovery, and learning evidence",
            "not autonomous self-modification",
            "This is read-only",
        ):
            if expected not in learning_reply:
                raise SystemExit(f"learning loop lane shortcut lost context {expected!r}: {learning_reply!r}")
    for _cid, orchestration_lane_reply, _markup in sent[48:52]:
        for expected in (
            "Cockpit lane: Internal Orchestration",
            "READ_ONLY",
            "approval required: no",
            "jarvis status",
            "worker readiness and capacity without spawning tasks",
            "not user-facing companion personas",
            "without granting autonomy, approvals, or tool execution",
            "does not approve, dispatch, or run risky actions",
            "This is read-only",
        ):
            if expected not in orchestration_lane_reply:
                raise SystemExit(
                    f"orchestration lane shortcut lost context {expected!r}: {orchestration_lane_reply!r}"
                )
    for _cid, text, _markup in sent[:11] + sent[12:]:
        if "does not approve" not in text or "execute" not in text:
            raise SystemExit(f"lane shortcut should declare the read-only boundary: {text!r}")


def test_acceptance_harness_shortcuts_open_lane_read_only() -> None:
    """Acceptance/live_check phone wording should open the compact Acceptance
    Harness lane, not the broad freeze dump or runtime command path."""
    _seed_offset(0)
    _setup_env("555001")

    class CockpitTool:
        def handler(self, args):
            return SimpleNamespace(
                output="Capability cockpit: full cockpit output with Acceptance Harness.",
                metadata={
                    "capability_lanes": [
                        {
                            "key": "acceptance_harness",
                            "title": "Acceptance Harness",
                            "status": "ready",
                            "risk": "READ_ONLY",
                            "approval_required": False,
                            "tool_coverage": "registered",
                            "smoke_coverage": "registered",
                            "next_command": "live proof status",
                            "next_command_kind": "diagnostic",
                            "example_command": "live proof status",
                            "proof_points": [
                                "live_check publishes one-screen acceptance rows for config, contacts, send resolution, voice, research, daily brief, chat, channels, and phone control",
                                "acceptance coverage, acceptance gaps, acceptance next, aggregate smoke, operator evals, jobs 7-day proof, and phone approvals rows keep open-work and next-proof status visible",
                                "acceptance gaps points to acceptance next for prioritized proof guidance without exposing checklist text",
                                "daemon startup row checks launcher contracts, LaunchAgent templates, scheduler ticker ownership, and restart documentation without process control",
                                "readiness rows do not replace live delivery, approval, phone, reboot, or daily-streak proof",
                            ],
                            "guardrail_notes": [
                                "diagnostic only; run opt-in live_check with operator present for real acceptance proof",
                                "readiness rows do not replace live delivery, approval, phone, reboot, or daily-streak proof",
                            ],
                            "trust_summary": "5/5 checks ready",
                            "trust_checklist": [
                                {"label": "registered tools", "ready": True},
                                {"label": "aggregate smoke coverage", "ready": True},
                                {"label": "read-only boundary", "ready": True},
                                {"label": "next command", "ready": True},
                                {"label": "explicit proof points", "ready": True},
                            ],
                        }
                    ]
                },
            )

    class CockpitRegistry:
        def get(self, name):
            if name != "capability_cockpit":
                raise KeyError(name)
            return CockpitTool()

    rt = FakeRuntime(["should not run"])
    rt.registry = CockpitRegistry()
    updates = [
        _update(113, 555001, "live proof status"),
        _update(114, 555001, "acceptance harness status"),
        _update(115, 555001, "live_check status"),
        _update(128, 555001, "live_check coverage"),
        _update(129, 555001, "live_check rows"),
        _update(130, 555001, "live_check missing rows"),
        _update(131, 555001, "one screen live_check table"),
        _update(132, 555001, "does live_check cover every checklist section"),
        _update(116, 555001, "definition of done"),
        _update(117, 555001, "what is the finish line?"),
        _update(149, 555001, "what rows are in live_check"),
        _update(150, 555001, "what rows does live_check show"),
        _update(151, 555001, "acceptance coverage status"),
        _update(152, 555001, "acceptance coverage drift"),
        _update(118, 555001, "라이브 증명 상태"),
        _update(119, 555001, "인수 상태"),
        _update(120, 555001, "완료 기준"),
        _update(121, 555001, "검수 체크리스트"),
        _update(153, 555001, "라이브체크 커버리지"),
        _update(154, 555001, "라이브 체크 표"),
        _update(155, 555001, "라이브체크 항목"),
        _update(156, 555001, "원스크린 인수 표"),
        _update(157, 555001, "인수 커버리지"),
        _update(122, 555001, "acceptance gaps"),
        _update(123, 555001, "what's left to finish?"),
        _update(124, 555001, "what remains before Jarvis is done?"),
        _update(166, 555001, "what is missing"),
        _update(167, 555001, "what is still missing"),
        _update(168, 555001, "what is unfinished"),
        _update(169, 555001, "what remains unfinished"),
        _update(125, 555001, "remaining acceptance gaps"),
        _update(126, 555001, "남은 기준"),
        _update(127, 555001, "뭐 남았어?"),
        _update(133, 555001, "뭐 증명해야 해?"),
        _update(134, 555001, "smoke status"),
        _update(135, 555001, "are tests green?"),
        _update(136, 555001, "aggregate smoke status"),
        _update(160, 555001, "full smoke status"),
        _update(161, 555001, "latest aggregate smoke proof"),
        _update(162, 555001, "smoke suite status"),
        _update(163, 555001, "suite lock status"),
        _update(164, 555001, "smoke lock status"),
        _update(165, 555001, "smoke suite lock status"),
        _update(137, 555001, "스모크 상태"),
        _update(138, 555001, "테스트 초록"),
        _update(139, 555001, "reboot status"),
        _update(140, 555001, "will Jarvis survive reboot?"),
        _update(141, 555001, "daemon startup status"),
        _update(142, 555001, "재부팅 상태"),
        _update(143, 555001, "데몬 상태"),
        _update(144, 555001, "voice proof status"),
        _update(145, 555001, "is Korean voice tested?"),
        _update(146, 555001, "telegram voice proof status"),
        _update(147, 555001, "음성 증명 상태"),
        _update(148, 555001, "한국어 음성 증명"),
    ]
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: updates,
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs:
        raise SystemExit(f"acceptance shortcuts must not route through runtime.handle: {rt.inputs}")
    if len(sent) != len(updates) or any(markup is not None for _cid, _text, markup in sent):
        raise SystemExit(f"acceptance replies should be read-only without buttons: {sent}")
    for _cid, text, _markup in sent:
        for expected in (
            "Cockpit lane: Acceptance Harness",
            "READ_ONLY",
            "approval required: no",
            "live proof status",
            "live_check publishes one-screen acceptance rows",
            "acceptance coverage, acceptance gaps, acceptance next",
            "acceptance gaps points to acceptance next for prioritized proof guidance",
            "aggregate smoke",
            "daemon startup row checks launcher contracts",
            "diagnostic only; run opt-in live_check with operator present",
            "readiness rows do not replace live delivery",
            "read-only boundary: ok",
            "This is read-only",
        ):
            if expected not in text:
                raise SystemExit(f"acceptance shortcut lost context {expected!r}: {text!r}")
        if "full cockpit output" in text or "should not run" in text:
            raise SystemExit(f"acceptance shortcut should not return broad cockpit/runtime output: {text!r}")
        if "does not approve" not in text or "execute" not in text:
            raise SystemExit(f"acceptance shortcut should declare the read-only boundary: {text!r}")


def test_acceptance_next_proof_phone_shortcuts_use_next_proof_packet() -> None:
    """Explicit next-proof wording should return one actionable proof packet,
    while broader acceptance wording stays on the cockpit lane."""
    _seed_offset(0)
    _setup_env("555001")

    phrases = [
        "acceptance next proof",
        "next acceptance proof",
        "next acceptance test",
        "next live proof",
        "next live test",
        "next proof",
        "next proof to run",
        "what is the next acceptance proof",
        "what proof is next?",
        "what proof should I run next?",
        "what should I prove next?",
        "what should I test next?",
        "what should operator test next?",
        "which proof is next?",
        "다음 증명",
        "다음 라이브 테스트",
        "다음에 뭐 테스트해",
        "다음에 뭐 증명해",
    ]
    rt = FakeRuntime(["Jarvis completion next proof packet:\nNext safe proof lane."] * len(phrases))
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(200 + index, 555001, phrase) for index, phrase in enumerate(phrases)
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs != ["completion_next_proof_packet"] * len(phrases):
        raise SystemExit(f"next-proof shortcuts should run the next-proof packet only: {rt.inputs}")
    if len(sent) != len(phrases) or any(markup is not None for _cid, _text, markup in sent):
        raise SystemExit(f"next-proof replies should be read-only without buttons: {sent}")
    for _cid, text, _markup in sent:
        if "Jarvis completion next proof packet" not in text or "Next safe proof lane" not in text:
            raise SystemExit(f"next-proof shortcut lost packet context: {text!r}")
        for wrong in ("Capability cockpit", "Did you mean", "Wikipedia", "Safety receipt"):
            if wrong in text:
                raise SystemExit(f"next-proof shortcut drifted into stale output {wrong!r}: {text!r}")


def test_cockpit_lane_metadata_fails_closed() -> None:
    """Malformed single-lane cockpit metadata should not break or leak through
    the owner-phone lane shortcut."""
    _seed_offset(0)
    _setup_env("555001")

    leak_marker = "SHOULD NOT APPEAR"

    class HostileText:
        def __str__(self) -> str:
            raise RuntimeError(f"{leak_marker} /\x55sers/hostile/cockpit-lane")

    class CockpitTool:
        def handler(self, args):
            return SimpleNamespace(
                output="Capability cockpit (derived...)",
                metadata={
                    "capability_lanes": [
                        {
                            "key": "voice",
                            "title": HostileText(),
                            "status": "ok",
                            "next_command": "voice setup check",
                        },
                        {
                            "key": "messaging",
                            "title": f"{leak_marker} /\x55sers/hostile/lane-title",
                            "status": HostileText(),
                            "risk": f"{leak_marker} /private/hostile-lane-risk",
                            "approval_required": True,
                            "tool_coverage": f"{leak_marker} /TMP/hostile-lane-tools",
                            "smoke_coverage": "registered",
                            "next_command": "channel health",
                            "next_command_kind": f"{leak_marker} /var/folders/hostile-lane-kind",
                            "example_command": f"{leak_marker} /\x55sers/hostile/lane-example",
                            "attention_reasons": [
                                HostileText(),
                                f"{leak_marker} /private/hostile-lane-reason",
                                "Normal lane reason",
                            ],
                            "proof_points": [
                                HostileText(),
                                f"{leak_marker} /TMP/hostile-lane-proof",
                                "Normal proof point",
                            ],
                            "last_success": f"{leak_marker} /\x55sers/hostile/lane-success",
                            "last_failure": HostileText(),
                            "last_failure_kind": f"{leak_marker} /var/folders/hostile-lane-failure-kind",
                            "last_approval_hold": f"{leak_marker} /private/hostile-lane-hold",
                            "approval_next_commands": [
                                HostileText(),
                                f"{leak_marker} /\x55sers/hostile/lane-review",
                                "approval packet 7",
                            ],
                            "guardrail_notes": [
                                HostileText(),
                                "Normal guardrail note",
                            ],
                        },
                    ]
                },
            )

    class CockpitRegistry:
        def get(self, name):
            if name != "capability_cockpit":
                raise KeyError(name)
            return CockpitTool()

    rt = FakeRuntime(["should not run"])
    rt.registry = CockpitRegistry()
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [_update(110, 555001, "messaging lane")],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs:
        raise SystemExit(f"malformed cockpit lane must not route through runtime.handle: {rt.inputs}")
    if len(sent) != 1 or sent[0][2] is not None:
        raise SystemExit(f"malformed cockpit lane should be read-only without buttons: {sent}")
    text = sent[0][1]
    for expected in (
        "Cockpit lane:",
        "<redacted-local-path>",
        "<unreadable>",
        "channel health",
        "registered",
        "approval packet 7",
        "Normal lane reason",
        "Normal proof point",
        "Normal guardrail note",
        "This is read-only",
    ):
        if expected not in text:
            raise SystemExit(f"malformed cockpit lane lost safe context {expected!r}: {text!r}")
    for leaked in (
        leak_marker,
        "/\x55sers/hostile",
        "/TMP/hostile",
        "/private/hostile",
        "/var/folders/hostile",
        "cockpit-lane",
    ):
        if leaked in text:
            raise SystemExit(f"malformed cockpit lane leaked hostile metadata: {text!r}")


def test_trust_test_shortcuts_open_operator_workflow_eval_lane_read_only() -> None:
    """Trust/eval-pack phone wording should land on the Operator Workflow Evals
    cockpit lane, because that is the visible proof surface for operator-real tasks."""
    _seed_offset(0)
    _setup_env("555001")

    class CockpitTool:
        def handler(self, args):
            return SimpleNamespace(
                output="Capability cockpit: full cockpit output with Operator Workflow Evals.",
                metadata={
                    "capability_lanes": [
                        {
                            "key": "operator_workflow_evals",
                            "title": "Operator Workflow Evals",
                            "status": "ready",
                            "risk": "HIGH_RISK",
                            "approval_required": True,
                            "tool_coverage": "registered",
                            "smoke_coverage": "registered",
                            "next_command": "channel health",
                            "next_command_kind": "diagnostic",
                            "example_command": "push today's brief to my phone",
                            "proof_points": [
                                "Korean Telegram send stops at approval with Hangul recipient/body intact",
                                "Morning Brief can be pushed on demand to the owner phone channel",
                                "Phone control shortcuts expose status, failures, channels, approvals, cockpit, lanes, and guardrails",
                                "Clean-state cockpit has zero false attention lanes",
                            ],
                            "attention_reasons": [
                                "pins Korean sends, phone brief, channel diagnostics, contacts, markets"
                            ],
                            "trust_summary": "5/5 checks ready",
                            "trust_checklist": [
                                {"label": "registered tools", "ready": True},
                                {"label": "aggregate smoke coverage", "ready": True},
                                {"label": "approval boundary", "ready": True},
                                {"label": "next command", "ready": True},
                                {"label": "explicit proof points", "ready": True},
                            ],
                        }
                    ]
                },
            )

    class CockpitRegistry:
        def get(self, name):
            if name != "capability_cockpit":
                raise KeyError(name)
            return CockpitTool()

    rt = FakeRuntime(["should not run"])
    rt.registry = CockpitRegistry()
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(103, 555001, "trust tests"),
            _update(104, 555001, "operator real workflow tests"),
            _update(105, 555001, "eval pack plan"),
            _update(106, 555001, "operator eval pack plan"),
            _update(107, 555001, "what is eval pack"),
            _update(108, 555001, "what is the eval pack"),
            _update(109, 555001, "what real workflows should Jarvis prove"),
            _update(110, 555001, "what should Jarvis prove live"),
            _update(111, 555001, "한국어 메시지 평가"),
            _update(112, 555001, "모닝브리핑 평가"),
            _update(113, 555001, "is Korean messaging tested"),
            _update(114, 555001, "Korean message proof"),
            _update(115, 555001, "can Jarvis safely send Korean messages"),
            _update(116, 555001, "평가팩 계획"),
            _update(117, 555001, "제임스 평가팩 계획"),
            _update(118, 555001, "실제 워크플로우 뭐 증명해"),
            _update(119, 555001, "자비스 뭐 증명해야 해"),
            _update(120, 555001, "한국어 메시지 증명"),
            _update(121, 555001, "한글 전송 증명"),
            _update(122, 555001, "가상연락처이 전송 테스트"),
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs:
        raise SystemExit(f"trust/eval shortcuts must not route through runtime.handle: {rt.inputs}")
    if len(sent) != 20 or any(markup is not None for _cid, _text, markup in sent):
        raise SystemExit(f"trust/eval replies should be read-only without buttons: {sent}")
    for _cid, text, _markup in sent:
        if "Cockpit lane: Operator Workflow Evals" not in text:
            raise SystemExit(f"trust/eval shortcut did not open the eval lane: {text!r}")
        for expected in (
            "HIGH_RISK",
            "approval required: yes",
            "smoke: registered",
            "Korean sends",
            "proofs:",
            "Korean Telegram send stops at approval",
            "Morning Brief can be pushed",
            "Clean-state cockpit has zero false attention lanes",
            "trust: 5/5 checks ready",
            "explicit proof points: ok",
        ):
            if expected not in text:
                raise SystemExit(f"trust/eval lane lost {expected!r}: {text!r}")
        if "does not approve" not in text or "execute" not in text:
            raise SystemExit(f"trust/eval lane should declare the read-only boundary: {text!r}")


def test_cockpit_summary_shortcuts_are_read_only() -> None:
    """Compact cockpit summary shortcuts should show the phone control-center
    overview without executing any suggested command."""
    _seed_offset(0)
    _setup_env("555001")

    class CockpitTool:
        def handler(self, args):
            return SimpleNamespace(
                output="Capability cockpit (derived...)",
                metadata={
                    "lane_count": 5,
                    "attention_lane_count": 1,
                    "direction_summary": "One Jarvis coordinator, visible capability cockpit, internal workers only.",
                    "direction_principles": [
                        "Show capabilities, health, risk, approvals, smoke coverage, and next commands before adding autonomy.",
                        "Keep subagents/internal workers invisible as personas; expose them only as inspectable orchestration capacity.",
                        "Do not add Zoey-style companion identities unless a real workflow proves the need.",
                    ],
                    "coverage_summary": {
                        "lane_count": 5,
                        "tool_ready_count": 5,
                        "smoke_ready_count": 5,
                        "approval_required_count": 2,
                        "auto_run_safe_count": 3,
                        "coverage_ready": True,
                        "summary": "tools 5/5, smokes 5/5, approval-gated 2, auto-run safe 3",
                    },
                    "trust_summary": {
                        "lane_count": 5,
                        "trust_ready_lane_count": 5,
                        "trust_partial_lane_count": 0,
                        "trust_check_count": 21,
                        "trust_ready_check_count": 21,
                        "trust_ready": True,
                        "trust_non_authorizing": True,
                        "summary": "trust checklist 5/5 lanes, 21/21 checks ready",
                    },
                    "proof_lane_count": 3,
                    "proof_point_count": 9,
                    "proof_summary": [
                        {
                            "lane_key": "operator_workflow_evals",
                            "lane_title": "Operator Workflow Evals",
                            "proof_count": 4,
                            "sample_proofs": [
                                "Korean Telegram send stops at approval with Hangul recipient/body intact",
                                "Morning Brief can be pushed on demand to the owner phone channel",
                            ],
                        }
                    ],
                    "proof_summary_hidden_lane_count": 2,
                    "proof_summary_hidden_lanes": [
                        {"lane_key": "learning_loop", "lane_title": "Learning Loop", "proof_count": 4},
                        {
                            "lane_key": "internal_orchestration",
                            "lane_title": "Internal Orchestration",
                            "proof_count": 3,
                        },
                    ],
                    "status_counts": {"attention": 1, "ok": 1, "ready": 3},
                    "next_command": "channel health",
                    "next_command_queue": [
                        {
                            "command": "channel health",
                            "kind": "diagnostic",
                            "lane_title": "Messaging (KR/EN)",
                            "status": "attention",
                        },
                        {
                            "command": "approval readiness 7",
                            "kind": "approval_review",
                            "lane_title": "Approvals",
                            "status": "awaiting approval",
                        },
                    ],
                    "capability_lanes": [
                        {
                            "key": "messaging",
                            "title": "Messaging (KR/EN)",
                            "status": "attention",
                            "next_command": "channel health",
                            "last_failure": "telegram_enter_did_not_send",
                            "last_failure_kind": "transport_error",
                        },
                        {
                            "key": "voice",
                            "title": "Voice",
                            "status": "ok",
                            "next_command": "voice setup check",
                            "last_success": "voice_command_cockpit @ 2026-07-07T00:58:00+09:00",
                        },
                        {
                            "key": "scheduler",
                            "title": "Scheduler",
                            "status": "ready",
                            "next_command": "list scheduled jobs",
                            "last_success": "list_scheduled_jobs @ 2026-07-07T00:59:00+09:00",
                        },
                        {
                            "key": "operator_workflow_evals",
                            "title": "Operator Workflow Evals",
                            "status": "ready",
                            "next_command": "channel health",
                            "proof_points": [
                                "Korean Telegram send stops at approval with Hangul recipient/body intact",
                                "Morning Brief can be pushed on demand to the owner phone channel",
                                "Phone control shortcuts expose status, failures, channels, approvals, cockpit, lanes, and guardrails",
                                "Clean-state cockpit has zero false attention lanes",
                            ],
                        },
                        {
                            "key": "approvals",
                            "title": "Approvals",
                            "status": "awaiting approval",
                            "next_command": "approval readiness 7",
                            "last_approval_hold": "send_telegram held at approval gate",
                            "approval_next_commands": [
                                "approval readiness 7",
                                "approval packet 7",
                                "approval chain proof 7",
                            ],
                        },
                    ],
                },
            )

    class CockpitRegistry:
        def get(self, name):
            if name != "capability_cockpit":
                raise KeyError(name)
            return CockpitTool()

    rt = FakeRuntime(["should not run"])
    rt.registry = CockpitRegistry()
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(103, 555001, "cockpit summary"),
            _update(104, 555001, "control plane summary"),
            _update(105, 555001, "콕핏 요약"),
            _update(106, 555001, "can I trust Jarvis?"),
            _update(107, 555001, "trust report"),
            _update(108, 555001, "is Jarvis reliable?"),
            _update(109, 555001, "자비스 믿어도 돼?"),
            _update(110, 555001, "믿을만해"),
            _update(111, 555001, "trust checklist"),
            _update(112, 555001, "earned trust checklist"),
            _update(113, 555001, "why should I trust Jarvis?"),
            _update(114, 555001, "what makes Jarvis trustworthy?"),
            _update(115, 555001, "신뢰 체크리스트"),
            _update(116, 555001, "자비스 신뢰 체크리스트"),
            _update(117, 555001, "왜 자비스 믿어도 돼"),
            _update(118, 555001, "믿어도 되는 이유"),
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs:
        raise SystemExit(f"cockpit summary must not route through runtime.handle: {rt.inputs}")
    if len(sent) != 16 or any(markup is not None for _cid, _text, markup in sent):
        raise SystemExit(f"cockpit summary replies should be read-only without buttons: {sent}")
    for _cid, text, _markup in sent:
        if "Cockpit summary" not in text:
            raise SystemExit(f"summary shortcut did not render the summary heading: {text!r}")
        for expected in ("5 lane(s)", "1 needing attention", "1 approval-held", "attention: 1", "ok: 1", "ready: 3"):
            if expected not in text:
                raise SystemExit(f"summary shortcut lost {expected!r}: {text!r}")
        if "coverage: tools 5/5, smokes 5/5, approval-gated 2, auto-run safe 3" not in text:
            raise SystemExit(f"summary shortcut lost coverage context: {text!r}")
        if "trust: trust checklist 5/5 lanes, 21/21 checks ready" not in text:
            raise SystemExit(f"summary shortcut lost earned-trust context: {text!r}")
        for expected in (
            "direction: One Jarvis coordinator, visible capability cockpit, internal workers only.",
            "visible capability cockpit",
            "internal workers invisible as personas",
            "Do not add Zoey-style companion identities",
        ):
            if expected not in text:
                raise SystemExit(f"summary shortcut lost research direction {expected!r}: {text!r}")
        for expected in (
            "proofs: 3 lane(s), 9 proof point(s)",
            "Operator Workflow Evals",
            "Korean Telegram send stops at approval",
            "Morning Brief can be pushed",
            "2 more proof lane(s) available",
            "Learning Loop",
            "Internal Orchestration",
            "ask for `<lane> lane`",
        ):
            if expected not in text:
                raise SystemExit(f"summary shortcut lost proof context {expected!r}: {text!r}")
        for expected in ("channel health", "Messaging (KR/EN)", "approval readiness 7"):
            if expected not in text:
                raise SystemExit(f"summary shortcut lost queue/lane context {expected!r}: {text!r}")
        for expected in (
            "last failure: telegram_enter_did_not_send (transport_error)",
            "approval hold: send_telegram held at approval gate",
            "review: approval readiness 7, approval packet 7",
            "last success: voice_command_cockpit @ 2026-07-07T00:58:00+09:00",
            "last success: list_scheduled_jobs @ 2026-07-07T00:59:00+09:00",
            "Approval-held means Jarvis is waiting for owner review",
            "not counted as a tool failure",
        ):
            if expected not in text:
                raise SystemExit(f"summary shortcut lost health detail {expected!r}: {text!r}")
        if "Ask for" not in text or "cockpit attention" not in text:
            raise SystemExit(f"summary shortcut should point to detailed cockpit views: {text!r}")
        if "does not approve" not in text or "execute" not in text:
            raise SystemExit(f"summary shortcut should declare the read-only boundary: {text!r}")


def test_cockpit_summary_direction_metadata_fails_closed() -> None:
    """Malformed direction metadata should not break the owner-phone cockpit
    summary or leak local paths/error text."""
    _seed_offset(0)
    _setup_env("555001")

    leak_marker = "SHOULD NOT APPEAR"

    class HostileText:
        def __str__(self) -> str:
            raise RuntimeError(f"{leak_marker} /\x55sers/hostile/cockpit-direction")

    class CockpitTool:
        def handler(self, args):
            return SimpleNamespace(
                output="Capability cockpit (derived...)",
                metadata={
                    "lane_count": 1,
                    "attention_lane_count": 0,
                    "direction_summary": f"{leak_marker} /\x55sers/hostile/cockpit-direction",
                    "direction_principles": [
                        HostileText(),
                        f"{leak_marker} /TMP/hostile-cockpit-principle",
                        "Normal visible-control principle",
                    ],
                    "status_counts": {"ready": 1},
                    "capability_lanes": [
                        {
                            "key": "diagnostics",
                            "title": "Diagnostics",
                            "status": "ready",
                            "next_command": "jarvis doctor",
                        }
                    ],
                },
            )

    class CockpitRegistry:
        def get(self, name):
            if name != "capability_cockpit":
                raise KeyError(name)
            return CockpitTool()

    rt = FakeRuntime(["should not run"])
    rt.registry = CockpitRegistry()
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [_update(111, 555001, "cockpit summary")],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs:
        raise SystemExit(f"malformed cockpit summary must not route through runtime.handle: {rt.inputs}")
    if len(sent) != 1 or sent[0][2] is not None:
        raise SystemExit(f"malformed cockpit summary should be read-only without buttons: {sent}")
    text = sent[0][1]
    for expected in (
        "Cockpit summary",
        "direction: <redacted-local-path>",
        "<unreadable>",
        "Normal visible-control principle",
        "This is read-only",
    ):
        if expected not in text:
            raise SystemExit(f"malformed cockpit summary lost safe placeholder {expected!r}: {text!r}")
    for leaked in (leak_marker, "/\x55sers/hostile", "/TMP/hostile", "cockpit-direction"):
        if leaked in text:
            raise SystemExit(f"malformed cockpit summary leaked hostile direction metadata: {text!r}")


def test_cockpit_summary_metadata_fails_closed() -> None:
    """Malformed summary counts, queue, proof, and lane metadata should stay
    bounded in the owner-phone cockpit summary."""
    _seed_offset(0)
    _setup_env("555001")

    leak_marker = "SHOULD NOT APPEAR"

    class HostileText:
        def __str__(self) -> str:
            raise RuntimeError(f"{leak_marker} /\x55sers/hostile/cockpit-summary")

    class CockpitTool:
        def handler(self, args):
            return SimpleNamespace(
                output="Capability cockpit (derived...)",
                metadata={
                    "lane_count": HostileText(),
                    "attention_lane_count": f"{leak_marker} /private/hostile-attention-count",
                    "coverage_summary": {"summary": f"{leak_marker} /\x55sers/hostile/coverage-summary"},
                    "proof_lane_count": f"{leak_marker} /\x55sers/hostile/proof-lane-count",
                    "proof_point_count": HostileText(),
                    "status_counts": {
                        HostileText(): 2,
                        f"{leak_marker} /TMP/hostile-status-key": 3,
                        "ready": 1,
                    },
                    "proof_summary": [
                        {
                            "lane_title": HostileText(),
                            "proof_count": f"{leak_marker} /\x55sers/hostile/proof-count",
                            "sample_proofs": [
                                HostileText(),
                                f"{leak_marker} /var/folders/hostile-proof",
                            ],
                        },
                        {
                            "lane_key": "operator_workflow_evals",
                            "proof_count": 1,
                            "sample_proofs": ["Normal proof point"],
                        },
                    ],
                    "proof_summary_hidden_lane_count": 2,
                    "proof_summary_hidden_lanes": [
                        {"lane_title": HostileText(), "proof_count": f"{leak_marker} /tmp/hidden-proof"},
                        {"lane_title": f"{leak_marker} /\x55sers/hostile/hidden-proof-lane"},
                    ],
                    "next_command": f"{leak_marker} /\x55sers/hostile/next-command",
                    "next_command_queue": [
                        {
                            "command": HostileText(),
                            "kind": f"{leak_marker} /private/hostile-kind",
                            "lane_title": f"{leak_marker} /\x55sers/hostile/queue-lane",
                        },
                        {
                            "command": "channel health",
                            "kind": "diagnostic",
                            "lane_title": "Diagnostics",
                        },
                    ],
                    "capability_lanes": [
                        {
                            "key": HostileText(),
                            "title": HostileText(),
                            "status": f"{leak_marker} /\x55sers/hostile/lane-status",
                            "next_command": f"{leak_marker} /private/hostile-lane-command",
                            "last_failure": HostileText(),
                            "last_failure_kind": f"{leak_marker} /TMP/hostile-kind",
                            "last_approval_hold": f"{leak_marker} /var/folders/hostile-hold",
                            "approval_next_commands": [
                                HostileText(),
                                f"{leak_marker} /\x55sers/hostile/review-command",
                            ],
                        },
                        {
                            "key": "diagnostics",
                            "title": "Diagnostics",
                            "status": "ready",
                            "next_command": "jarvis doctor",
                            "last_success": "doctor checked cleanly",
                        },
                    ],
                },
            )

    class CockpitRegistry:
        def get(self, name):
            if name != "capability_cockpit":
                raise KeyError(name)
            return CockpitTool()

    rt = FakeRuntime(["should not run"])
    rt.registry = CockpitRegistry()
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [_update(112, 555001, "cockpit summary")],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs:
        raise SystemExit(f"malformed cockpit summary must not route through runtime.handle: {rt.inputs}")
    if len(sent) != 1 or sent[0][2] is not None:
        raise SystemExit(f"malformed cockpit summary should be read-only without buttons: {sent}")
    text = sent[0][1]
    for expected in (
        "Cockpit summary",
        "<unreadable>",
        "<redacted-local-path>",
        "ready: 1",
        "channel health",
        "Diagnostics",
        "Normal proof point",
        "2 more proof lane(s)",
        "jarvis doctor",
        "last success: doctor checked cleanly",
        "This is read-only",
    ):
        if expected not in text:
            raise SystemExit(f"malformed cockpit summary lost safe context {expected!r}: {text!r}")
    for leaked in (
        leak_marker,
        "/\x55sers/hostile",
        "/private/hostile",
        "/TMP/hostile",
        "/var/folders/hostile",
        "cockpit-summary",
        "coverage-summary",
        "queue-lane",
    ):
        if leaked in text:
            raise SystemExit(f"malformed cockpit summary leaked hostile metadata: {text!r}")


def test_cockpit_proof_shortcuts_are_read_only() -> None:
    """Proof/evidence phone wording should open the compact cockpit proof view
    directly, without routing through runtime.handle or executing a suggested command."""
    _seed_offset(0)
    _setup_env("555001")

    class CockpitTool:
        def handler(self, args):
            return SimpleNamespace(
                output="Capability cockpit (derived...)",
                metadata={
                    "proof_lane_count": 3,
                    "proof_point_count": 9,
                    "proof_summary": [
                        {
                            "lane_key": "operator_workflow_evals",
                            "lane_title": "Operator Workflow Evals",
                            "proof_count": 4,
                            "sample_proofs": [
                                "Korean Telegram send stops at approval with Hangul recipient/body intact",
                                "Morning Brief can be pushed on demand to the owner phone channel",
                            ],
                        }
                    ],
                    "proof_summary_hidden_lane_count": 2,
                    "proof_summary_hidden_lanes": [
                        {"lane_key": "learning_loop", "lane_title": "Learning Loop", "proof_count": 4},
                        {
                            "lane_key": "internal_orchestration",
                            "lane_title": "Internal Orchestration",
                            "proof_count": 3,
                        },
                    ],
                    "capability_lanes": [
                        {
                            "key": "operator_workflow_evals",
                            "title": "Operator Workflow Evals",
                            "proof_points": [
                                "Korean Telegram send stops at approval with Hangul recipient/body intact",
                                "Morning Brief can be pushed on demand to the owner phone channel",
                                "Phone control shortcuts expose status, failures, channels, approvals, cockpit, lanes, and guardrails",
                                "Clean-state cockpit has zero false attention lanes",
                            ],
                        }
                    ],
                },
            )

    class CockpitRegistry:
        def get(self, name):
            if name != "capability_cockpit":
                raise KeyError(name)
            return CockpitTool()

    rt = FakeRuntime(["should not run"])
    rt.registry = CockpitRegistry()
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(106, 555001, "proofs"),
            _update(107, 555001, "evidence"),
            _update(108, 555001, "what proof do we have?"),
            _update(109, 555001, "증거 상태"),
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs:
        raise SystemExit(f"proof shortcuts must not route through runtime.handle: {rt.inputs}")
    if len(sent) != 4 or any(markup is not None for _cid, _text, markup in sent):
        raise SystemExit(f"proof shortcut replies should be read-only without buttons: {sent}")
    for _cid, text, _markup in sent:
        for expected in (
            "Cockpit proofs:",
            "coverage: 3 lane(s), 9 proof point(s)",
            "Operator Workflow Evals",
            "Korean Telegram send stops at approval",
            "Morning Brief can be pushed",
            "2 more proof lane(s) available",
            "Learning Loop",
            "Internal Orchestration",
            "ask for `<lane> lane`",
        ):
            if expected not in text:
                raise SystemExit(f"proof shortcut lost {expected!r}: {text!r}")
        if "does not approve" not in text or "execute" not in text:
            raise SystemExit(f"proof shortcut should declare the read-only boundary: {text!r}")


def test_cockpit_proof_metadata_fails_closed() -> None:
    """Malformed proof metadata should not break or leak through the owner-phone
    proof/evidence shortcut."""
    _seed_offset(0)
    _setup_env("555001")

    leak_marker = "SHOULD NOT APPEAR"

    class HostileText:
        def __str__(self) -> str:
            raise RuntimeError(f"{leak_marker} /\x55sers/hostile/cockpit-proof")

    class CockpitTool:
        def handler(self, args):
            return SimpleNamespace(
                output="Capability cockpit (derived...)",
                metadata={
                    "proof_lane_count": f"{leak_marker} /\x55sers/hostile/proof-lane-count",
                    "proof_point_count": f"{leak_marker} /private/hostile-proof-point-count",
                    "proof_summary": [
                        {
                            "lane_title": HostileText(),
                            "proof_count": f"{leak_marker} /TMP/hostile-proof-count",
                            "sample_proofs": [
                                HostileText(),
                                f"{leak_marker} /var/folders/hostile-proof",
                                "Normal proof point",
                            ],
                        },
                        {
                            "lane_key": "operator_workflow_evals",
                            "proof_count": 1,
                            "sample_proofs": ["Korean Telegram send stops at approval"],
                        },
                    ],
                    "proof_summary_hidden_lane_count": 2,
                    "proof_summary_hidden_lanes": [
                        {"lane_title": HostileText(), "proof_count": f"{leak_marker} /tmp/hidden-count"},
                        {
                            "lane_key": f"{leak_marker} /private/hostile-hidden-key",
                            "lane_title": f"{leak_marker} /\x55sers/hostile-hidden-title",
                        },
                    ],
                    "capability_lanes": [
                        {
                            "key": "fallback_should_not_run",
                            "title": f"{leak_marker} /\x55sers/hostile/fallback-title",
                            "proof_points": [f"{leak_marker} /private/hostile-fallback-proof"],
                        }
                    ],
                },
            )

    class CockpitRegistry:
        def get(self, name):
            if name != "capability_cockpit":
                raise KeyError(name)
            return CockpitTool()

    rt = FakeRuntime(["should not run"])
    rt.registry = CockpitRegistry()
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [_update(110, 555001, "proofs")],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs:
        raise SystemExit(f"malformed cockpit proofs must not route through runtime.handle: {rt.inputs}")
    if len(sent) != 1 or sent[0][2] is not None:
        raise SystemExit(f"malformed cockpit proofs should be read-only without buttons: {sent}")
    text = sent[0][1]
    for expected in (
        "Cockpit proofs:",
        "coverage: <redacted-local-path> lane(s), <redacted-local-path> proof point(s)",
        "<unreadable>",
        "<redacted-local-path>",
        "Normal proof point",
        "Korean Telegram send stops at approval",
        "2 more proof lane(s)",
        "This is read-only",
    ):
        if expected not in text:
            raise SystemExit(f"malformed cockpit proofs lost safe context {expected!r}: {text!r}")
    for leaked in (
        leak_marker,
        "/\x55sers/hostile",
        "/TMP/hostile",
        "/private/hostile",
        "/var/folders/hostile",
        "cockpit-proof",
        "fallback_should_not_run",
    ):
        if leaked in text:
            raise SystemExit(f"malformed cockpit proofs leaked hostile metadata: {text!r}")


def test_korean_phone_status_shortcuts_map_to_read_only_commands() -> None:
    """Korean owner-phone shortcuts should run the same read-only status tools
    directly instead of returning a two-step command suggestion."""
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(
        [
            "status listed",
            "capabilities listed",
            "runs listed",
            "execution health listed",
            "jobs listed",
            "channels listed",
            "channels listed 2",
            "approvals listed",
            "readiness listed",
            "setup listed",
            "doctor listed",
            "safety listed",
            "privacy listed",
            "risk listed",
            "messages capabilities listed",
            "messages capabilities listed 2",
            "productivity capabilities listed",
            "productivity capabilities listed 2",
            "voice cockpit listed",
            "info capabilities listed",
            "markets capabilities listed",
            "utilities capabilities listed",
            "research capabilities listed",
            "writing capabilities listed",
            "memory capabilities listed",
            "notes capabilities listed",
            "safety listed 2",
            "risk listed 2",
        ]
    )
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(92, 555001, "상태 확인"),
            _update(93, 555001, "기능 알려줘"),
            _update(94, 555001, "뭐가 고장났어"),
            _update(95, 555001, "최근 실패"),
            _update(96, 555001, "브리핑 확인"),
            _update(97, 555001, "채널 상태"),
            _update(98, 555001, "채널 헬스"),
            _update(99, 555001, "승인 대기"),
            _update(100, 555001, "준비 상태"),
            _update(101, 555001, "설정 확인"),
            _update(102, 555001, "진단"),
            _update(103, 555001, "안전"),
            _update(104, 555001, "개인정보"),
            _update(105, 555001, "위험"),
            _update(106, 555001, "메시지 보낼 수 있어"),
            _update(107, 555001, "전화 걸 수 있어"),
            _update(108, 555001, "이메일 확인 가능해"),
            _update(109, 555001, "아침 브리핑 가능해"),
            _update(110, 555001, "음성 가능해"),
            _update(111, 555001, "날씨 확인 가능해"),
            _update(112, 555001, "주식 확인 가능해"),
            _update(113, 555001, "번역 가능해"),
            _update(114, 555001, "웹 검색 가능해"),
            _update(115, 555001, "글쓰기 가능해"),
            _update(116, 555001, "기억할 수 있어"),
            _update(117, 555001, "메모 가능해"),
            _update(118, 555001, "파일 읽을 수 있어"),
            _update(119, 555001, "명령 실행 가능해"),
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs != [
        "jarvis status",
        "capability map",
        "recent tool runs",
        "execution health report",
        "list scheduled jobs",
        "channel health",
        "channel health",
        "pending approvals",
        "readiness report",
        "setup check",
        "jarvis doctor",
        "safety status",
        "privacy report",
        "risk matrix",
        "capability map messages",
        "capability map messages",
        "capability map productivity",
        "capability map productivity",
        "voice command cockpit",
        "capability map info",
        "capability map markets",
        "capability map utilities",
        "capability map research",
        "capability map writing",
        "capability map memory",
        "capability map notes",
        "safety status",
        "risk matrix",
    ]:
        raise SystemExit(f"Korean status shortcuts mapped wrong: {rt.inputs}")
    if not sent or any(markup is not None for _, _, markup in sent):
        raise SystemExit(f"Korean read-only shortcuts must not create approval buttons: {sent}")


def test_phone_failure_error_shortcuts_route_to_read_only_diagnostics() -> None:
    """English/Korean owner-phone failure/error shortcuts should open diagnostics,
    never chat fallback, sends, or approval buttons."""
    _seed_offset(0)
    _setup_env("555001")
    cases = [
        ("last error", "recent tool runs"),
        ("show last error", "recent tool runs"),
        ("show last failure", "recent tool runs"),
        ("latest failure", "recent tool runs"),
        ("what errors happened", "recent tool runs"),
        ("anything failing", "execution health report"),
        ("show me failures", "execution health report"),
        ("failure details", "execution health report"),
        ("what failed last", "recent tool runs"),
        ("what was the last failure", "recent tool runs"),
        ("마지막 오류", "recent tool runs"),
        ("무슨 오류 있어", "recent tool runs"),
        ("최근 에러", "recent tool runs"),
        ("마지막 실패", "execution health report"),
        ("최근 실패", "execution health report"),
        ("recovery status", "recovery closure checklist"),
        ("what needs recovery", "recovery closure checklist"),
        ("복구 상태", "recovery closure checklist"),
    ]
    rt = FakeRuntime([f"diagnostic listed {index}" for index, _case in enumerate(cases, start=1)])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(313 + index, 555001, phrase) for index, (phrase, _expected) in enumerate(cases)
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    expected_inputs = [expected for _phrase, expected in cases]
    if rt.inputs != expected_inputs:
        raise SystemExit(f"failure/error shortcuts mapped wrong: {rt.inputs}")
    if not sent or any(markup is not None for _, _, markup in sent):
        raise SystemExit(f"failure/error diagnostics must not create approval buttons: {sent}")


def test_phone_audit_history_shortcuts_route_to_read_only_audit_tools() -> None:
    """Owner-phone audit/history shortcuts should open audit receipts directly,
    never chat fallback, sends, or approval buttons."""
    _seed_offset(0)
    _setup_env("555001")
    cases = [
        ("audit trail", "recent tool runs"),
        ("show audit trail", "recent tool runs"),
        ("execution audit", "recent tool runs"),
        ("show execution audit", "recent tool runs"),
        ("last tool run", "recent tool runs"),
        ("latest tool run", "recent tool runs"),
        ("recent tool run", "recent tool runs"),
        ("tool run history", "recent tool runs"),
        ("tool execution history", "recent tool runs"),
        ("what ran last", "recent tool runs"),
        ("what tool ran last", "recent tool runs"),
        ("what did Jarvis run last", "recent tool runs"),
        ("runtime trace latest", "runtime trace receipt"),
        ("last execution receipt", "verification receipt"),
        ("latest execution receipt", "verification receipt"),
        ("감사 로그", "recent tool runs"),
        ("감사 상태", "recent tool runs"),
        ("실행 기록", "recent tool runs"),
        ("최근 실행 기록", "recent tool runs"),
        ("마지막 실행", "recent tool runs"),
        ("마지막 도구 실행", "recent tool runs"),
        ("최근 도구 실행", "recent tool runs"),
        ("도구 실행 기록", "recent tool runs"),
        ("런타임 추적", "runtime trace receipt"),
        ("실행 영수증", "verification receipt"),
    ]
    rt = FakeRuntime([f"audit listed {index}" for index, _case in enumerate(cases, start=1)])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(413 + index, 555001, phrase) for index, (phrase, _expected) in enumerate(cases)
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    expected_inputs = [expected for _phrase, expected in cases]
    if rt.inputs != expected_inputs:
        raise SystemExit(f"audit/history shortcuts mapped wrong: {rt.inputs}")
    if not sent or any(markup is not None for _, _, markup in sent):
        raise SystemExit(f"audit/history diagnostics must not create approval buttons: {sent}")


def test_english_phone_delivery_status_questions_route_to_channel_health() -> None:
    """English owner-phone "did it send?" questions should open read-only
    channel health directly, without creating sends or approval buttons."""
    _seed_offset(0)
    _setup_env("555001")
    phrases = [
        "did it send?",
        "did it go through?",
        "was it sent?",
        "did the message send?",
        "message sent?",
        "did telegram send?",
        "was telegram sent?",
        "telegram delivered?",
        "did kakao send?",
        "did kakaotalk go through?",
        "did imessage send?",
        "did instagram go through?",
        "did insta send?",
        "did the call go through?",
        "was call connected?",
    ]
    rt = FakeRuntime([f"channel health listed {index}" for index, _phrase in enumerate(phrases, start=1)])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(213 + index, 555001, phrase) for index, phrase in enumerate(phrases)
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs != ["channel health"] * len(phrases):
        raise SystemExit(f"English delivery-status questions mapped wrong: {rt.inputs}")
    if not sent or any(markup is not None for _, _, markup in sent):
        raise SystemExit(f"English delivery-status checks must not create approval buttons: {sent}")


def test_korean_phone_delivery_status_questions_route_to_channel_health() -> None:
    """Korean owner-phone "did it send?" questions should open the read-only
    channel-health control plane directly, not drift into chat or send paths."""
    _seed_offset(0)
    _setup_env("555001")
    phrases = [
        "메시지 보내졌어?",
        "텔레그램 보내졌어?",
        "카톡 보내졌어?",
        "아이메시지 갔어?",
        "인스타그램 보냈어?",
        "메시지 보내졌나요?",
        "메세지 갔나요?",
        "문자 보냈나요?",
        "전송됐나요?",
        "텔레그램 보내졌나요?",
        "카카오 갔나요?",
        "카톡 갔나요?",
        "아이메시지 보내졌나요?",
        "인스타그램 갔나요?",
        "인스타 보냈나요?",
        "전화 연결됐어?",
        "전화 연결됬나요?",
        "전화 걸렸어?",
        "통화 연결됐어?",
        "통화 됬나요?",
        "페이스타임 연결됐어?",
        "페이스타임 됬나요?",
    ]
    rt = FakeRuntime([f"channel health listed {index}" for index, _phrase in enumerate(phrases, start=1)])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(113 + index, 555001, phrase) for index, phrase in enumerate(phrases)
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs != ["channel health"] * len(phrases):
        raise SystemExit(f"Korean delivery-status questions mapped wrong: {rt.inputs}")
    if not sent or any(markup is not None for _, _, markup in sent):
        raise SystemExit(f"Korean delivery-status checks must not create approval buttons: {sent}")

    class CockpitTool:
        def handler(self, args):
            return SimpleNamespace(output="Capability cockpit (derived...)", metadata={})

    class CockpitRegistry:
        def get(self, name):
            if name != "capability_cockpit":
                raise KeyError(name)
            return CockpitTool()

    rt2 = FakeRuntime([])
    rt2.registry = CockpitRegistry()
    sent2 = []
    bridge2 = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt2,
        send_func=lambda cid, t, markup=None: sent2.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(105, 555001, "콕핏"),
            _update(106, 555001, "가드레일 상태"),
            _update(107, 555001, "capability cockpit plan"),
            _update(108, 555001, "phone control center plan"),
            _update(109, 555001, "what is capability cockpit plan"),
            _update(110, 555001, "what is phone control center"),
            _update(111, 555001, "능력 콕핏 계획"),
            _update(112, 555001, "폰 컨트롤 센터 계획"),
        ],
        chat_action_func=_no_chat_action,
    )
    bridge2.process_once(poll_timeout=0)
    if rt2.inputs:
        raise SystemExit("Korean cockpit/guardrail shortcuts must not route through runtime.handle")
    if len(sent2) != 8 or not all("capability cockpit" in item[1].lower() for item in sent2):
        raise SystemExit(f"Korean cockpit/guardrail shortcuts did not reply with the cockpit: {sent2}")


def test_korean_phone_voice_shortcuts_map_to_read_only_commands() -> None:
    """Korean voice-management shortcuts are read-only inspection/intent packets;
    they must not touch the microphone or create approval buttons."""
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(
        [
            "voice cockpit listed",
            "voice setup listed",
            "voice stop listed",
            "voice cockpit listed 2",
            "voice setup listed 2",
            "voice stop listed 2",
        ]
    )
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [
            _update(107, 555001, "음성"),
            _update(108, 555001, "마이크 확인"),
            _update(109, 555001, "음성 중지"),
            _update(110, 555001, "음성 명령 상태"),
            _update(111, 555001, "음성 설정 확인"),
            _update(112, 555001, "다시 녹음"),
        ],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    if rt.inputs != [
        "voice command cockpit",
        "voice setup check",
        "voice stop intent: stop listening",
        "voice command cockpit",
        "voice setup check",
        "voice stop intent: stop listening",
    ]:
        raise SystemExit(f"Korean voice shortcuts mapped wrong: {rt.inputs}")
    if not sent or any(markup is not None for _, _, markup in sent):
        raise SystemExit(f"Korean voice shortcuts must not create approval buttons: {sent}")


def test_help_mentions_status_shortcuts() -> None:
    _seed_offset(0)
    _setup_env("555001")
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: FakeRuntime([]),
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [_update(78, 555001, "/help")],
        chat_action_func=_no_chat_action,
    )
    bridge.process_once(poll_timeout=0)
    help_text = sent[0][1].lower() if sent else ""
    for expected in (
        "status",
        "voice",
        "voice setup",
        "voice stop",
        "voice lane",
        "voice capability health",
        "voice proof status",
        "open voice acceptance proof gaps",
        "voice proof matrix",
        "voice live-proof report format",
        "no transcript contents",
        "capabilities",
        "approval-gated",
        "live-proof pending",
        "agent research",
        "handoff notice",
        "what does codex_tasks say?",
        "what changed after codex?",
        "worker status",
        "completion status",
        "why isn't jarvis done?",
        "completion blockers and proof debt",
        "agi progress",
        "roadmap",
        "agi status",
        "chat latency status",
        "chat p95 status",
        "mixed-conversation latency proof",
        "tuning knobs",
        "goals",
        "trust checklist",
        "earned trust checklist",
        "trust tests",
        "proofs",
        "what broke",
        "worker status",
        "orchestration lane",
        "worker lane",
        "internal worker orchestration health",
        "워커 레인",
        "에이전트 레인",
        "cockpit summary",
        "next action",
        "attention",
        "approvals lane",
        "pending approval review health",
        "messaging lane",
        "calls lane",
        "call capability health",
        "morning brief lane",
        "contacts lane",
        "calendar email lane",
        "personal proofs lane",
        "personal integration proof gaps",
        "markets lane",
        "weather lane",
        "markets/weather capability health",
        "research lane",
        "web lane",
        "research/web lookup capability health",
        "memory lane",
        "memory/notes capability health",
        "learning lane",
        "learning-loop/recovery proof health",
        "diagnostics lane",
        "local diagnostics/recovery health",
        "operator evals lane",
        "workflow evals lane",
        "operator workflow eval proof lane",
        "scheduler lane",
        "guardrails",
        "freeze status",
        "live-proof freeze",
        "live proof status",
        "compact acceptance harness readiness",
        "acceptance harness status",
        "one-screen acceptance proof/gap view",
        "acceptance gaps",
        "open august acceptance gaps",
        "acceptance next for prioritized proof guidance",
        "definition of done",
        "august acceptance checklist status",
        "acceptance next proof",
        "next safe proof lane",
        "smoke status",
        "latest aggregate smoke proof",
        "reboot status",
        "daemon startup and reboot proof readiness",
        "conversation research proof matrix",
        "conversation/research proof report format",
        "mixed conversation proof matrix",
        "10+ turn chat/research/tool proof report format",
        "no transcript contents",
        "what should i test?",
        "pending live-proof matrix",
        "live test result format",
        "pass/fail",
        "stage/error",
        "message content",
        "personal proof matrix",
        "personal integration proof report format",
        "no private account details",
        "scheduler proof matrix",
        "scheduled-job 7-day proof report format",
        "no private payloads",
        "daily value proof matrix",
        "morning brief usefulness/delivery proof report format",
        "no private brief contents",
        "approval proof matrix",
        "phone approval-flow proof report format",
        "no private planned args",
        "phone control proof matrix",
        "what broke/cockpit/approvals proof report format",
        "no private run payloads",
        "reboot proof matrix",
        "daemon/reboot/network recovery proof report format",
        "no private logs",
        "safety",
        "privacy",
        "risk",
        "readiness",
        "memory status",
        "learning status",
        "setup",
        "doctor",
        "brief status",
        "channels",
        "did it send?",
        "delivery/channel diagnostics",
        "전화 연결됐어?",
        "approvals",
    ):
        if expected not in help_text:
            raise SystemExit(f"/help must mention the {expected!r} status shortcut")
    if "setup/readiness" in help_text:
        raise SystemExit("/help should describe phone shortcuts plainly, not with slash-heavy wording")
    for stale in (
        "agent status — internal subagent/worker readiness",
        "• agent status",
    ):
        if stale in help_text:
            raise SystemExit(f"/help should not promote stale external agent wording: {stale!r}")
    for expected in (
        "상태",
        "음성",
        "마이크 확인",
        "음성 중지",
        "음성 레인",
        "마이크 레인",
        "음성 증명 상태",
        "한국어 음성 증명",
        "음성 증명 매트릭스",
        "음성 증명 결과 형식",
        "기능 알려줘",
        "조이 조사 결과",
        "인수 상태",
        "완료 기준",
        "검수 체크리스트",
        "남은 기준",
        "뭐 남았어",
        "다음 증명",
        "다음 라이브 증명",
        "스모크 상태",
        "테스트 초록",
        "재부팅 상태",
        "데몬 상태",
        "코덱스 작업 뭐야",
        "작업 큐 보여줘",
        "뭐 수정했어",
        "워커 상태",
        "자비스 끝났어?",
        "완료 뭐 막혀",
        "agi 진행상황",
        "로드맵",
        "agi 상태",
        "대화 지연 상태",
        "채팅 느려",
        "대화 증명 매트릭스",
        "대화 증명 결과 형식",
        "목표 상태",
        "신뢰 체크리스트",
        "믿어도 되는 이유",
        "신뢰 테스트",
        "증거 상태",
        "뭐가 고장났어",
        "콕핏 요약",
        "다음 행동",
        "주의 상태",
        "승인 레인",
        "메시지 상태",
        "승인 필요/라이브 검증 대기",
        "통화 레인",
        "전화 레인",
        "브리핑 레인",
        "연락처 레인",
        "캘린더 이메일 레인",
        "개인 증명 레인",
        "개인 증명 매트릭스",
        "개인 증명 결과 형식",
        "스케줄 증명 매트릭스",
        "예약 작업 증명 형식",
        "승인 증명 매트릭스",
        "승인 결과 형식",
        "시장 레인",
        "날씨 레인",
        "검색 레인",
        "연구 레인",
        "기억 레인",
        "학습 레인",
        "진단 레인",
        "제임스 평가 레인",
        "워크플로우 평가 레인",
        "스케줄 레인",
        "동결 상태",
        "라이브 증명 상태",
        "라이브 테스트 뭐 해야 해",
        "라이브 결과 형식",
        "기억 상태",
        "학습 상태",
        "브리핑 상태",
        "채널 상태",
        "채널 헬스",
        "텔레그램 보내졌어?",
        "카톡 갔나요?",
        "승인",
    ):
        if expected not in help_text:
            raise SystemExit(f"/help must mention the Korean {expected!r} shortcut")
    if "에이전트 상태" in help_text:
        raise SystemExit("/help should not promote Korean agent wording for internal workers")


def test_pending_approval_sends_inline_buttons() -> None:
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime([("Approval queued.", 7)])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [_update(80, 555001, "send an email saying hi")],
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 1:
        raise SystemExit("approval-producing command was not processed")
    if rt.inputs != ["send an email saying hi"]:
        raise SystemExit(f"unexpected runtime inputs: {rt.inputs}")
    if len(sent) != 1:
        raise SystemExit(f"approval reply not sent: {sent}")
    markup = sent[0][2]
    buttons = (markup or {}).get("inline_keyboard", [[]])[0]
    callback_data = [button.get("callback_data") for button in buttons]
    if callback_data != ["approve:7", "deny:7"]:
        raise SystemExit(f"approval buttons missing or malformed: {markup}")


def test_invalid_metadata_approval_id_does_not_send_inline_buttons() -> None:
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime([("Approval queued with bad metadata.", -7)])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [_update(81, 555001, "send an email saying hi")],
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 1:
        raise SystemExit("invalid approval metadata command was not processed")
    if rt.inputs != ["send an email saying hi"]:
        raise SystemExit(f"unexpected runtime inputs: {rt.inputs}")
    if sent != [("555001", "Approval queued with bad metadata.", None)]:
        raise SystemExit(f"invalid approval id metadata should not attach buttons: {sent}")


def test_invalid_trace_approval_id_does_not_send_inline_buttons() -> None:
    _seed_offset(0)
    _setup_env("555001")
    result = SimpleNamespace(
        response="Approval queued with bad trace metadata.",
        tool_results=[],
        metadata={
            "runtime_trace": {
                "tool_results": [
                    {"metadata": {"requires_confirmation": True, "approval_id": 0}},
                    {"metadata": {"requires_confirmation": True, "approval_id": "-3"}},
                ]
            }
        },
    )
    rt = FakeRuntimeTrace(result)
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [_update(82, 555001, "send a telegram saying hi")],
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 1:
        raise SystemExit("invalid trace approval metadata command was not processed")
    if rt.inputs != ["send a telegram saying hi"]:
        raise SystemExit(f"unexpected runtime inputs: {rt.inputs}")
    if sent != [("555001", "Approval queued with bad trace metadata.", None)]:
        raise SystemExit(f"invalid trace approval ids should not attach buttons: {sent}")


def test_text_approval_receipt_sends_inline_buttons() -> None:
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(
        [
            "'send_email' is risk level HIGH_RISK; explicit approval required.\n\n"
            "Safety receipt: queued as approval #42.\n"
            "Review: pending approvals"
        ]
    )
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [_update(81, 555001, "send an email to alex saying hi")],
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 1:
        raise SystemExit("text approval receipt command was not processed")
    markup = sent[0][2] if sent else None
    buttons = (markup or {}).get("inline_keyboard", [[]])[0]
    callback_data = [button.get("callback_data") for button in buttons]
    if callback_data != ["approve:42", "deny:42"]:
        raise SystemExit(f"text approval receipt did not get buttons: {sent}")


def test_invalid_text_approval_receipt_does_not_send_inline_buttons() -> None:
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(["Safety receipt: queued as approval #0."])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [_update(82, 555001, "send an email to alex saying hi")],
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 1:
        raise SystemExit("invalid text approval receipt command was not processed")
    if sent != [("555001", "Safety receipt: queued as approval #0.", None)]:
        raise SystemExit(f"invalid text approval receipt should not get buttons: {sent}")


def test_generic_approval_text_does_not_get_inline_buttons() -> None:
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(["Approval history:\n- approval #42 was already dismissed."])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [_update(82, 555001, "approval history")],
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 1:
        raise SystemExit("approval history command was not processed")
    if sent != [("555001", "Approval history:\n- approval #42 was already dismissed.", None)]:
        raise SystemExit(f"generic approval text should not get buttons: {sent}")


def test_runtime_exception_reply_is_non_leaky() -> None:
    _seed_offset(0)
    _setup_env("555001")
    sent = []

    class FailingRuntime:
        def handle(self, text, request_token=None):
            raise RuntimeError("secret stack path /\x55sers/example/private and /var/folders/zc/telegram-stack.log plus /tmp/telegram-stack.log")

    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: FailingRuntime(),
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [_update(83, 555001, "do something")],
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 1:
        raise SystemExit("runtime exception command should still be processed")
    if len(sent) != 1:
        raise SystemExit(f"runtime exception reply was not sent: {sent}")
    reply = sent[0][1]
    if "internal error" not in reply or "RuntimeError" not in reply:
        raise SystemExit(f"runtime exception reply should be friendly with bounded type: {reply}")
    if any(fragment in reply for fragment in ["secret stack path", "Telegram control error", "/\x55sers/", "/var/folders/", "/tmp/"]):
        raise SystemExit(f"runtime exception reply leaked raw details: {reply}")


def test_runtime_reply_redacts_local_paths_before_send() -> None:
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(["Saved report at /\x55sers/example/private/report.md and /private/tmp/jarvis-proof.txt plus /var/folders/zc/jarvis-proof.txt and /tmp/jarvis-proof.txt"])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [_update(84, 555001, "where is the report")],
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 1:
        raise SystemExit("path-shaped runtime reply command should be processed")
    if len(sent) != 1:
        raise SystemExit(f"path-shaped runtime reply was not sent once: {sent}")
    reply = sent[0][1]
    if any(fragment in reply for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]) or "<local-path>" not in reply:
        raise SystemExit(f"Telegram runtime reply did not redact local paths: {reply}")


def test_outbound_path_redaction_preserves_slash_separated_prose() -> None:
    prose = (
        "Safety review: stale/broad/private/surprising; "
        "approve/dismiss and input/output stay visible."
    )
    if tc._safe_outbound_text(prose) != prose:
        raise SystemExit("Telegram outbound redaction corrupted ordinary slash-separated prose")
    if tc._clean_cockpit_display("stale/broad/private/surprising") != "stale/broad/private/surprising":
        raise SystemExit("Telegram cockpit redaction corrupted ordinary slash-separated prose")

    path_cases = (
        "/private/tmp/jarvis-proof.txt",
        "Saved at /\x55sers/example/private/report.md",
        "Saved at (/var/folders/zc/jarvis-proof.txt)",
        'Saved at "/tmp/jarvis-proof.txt"',
    )
    for value in path_cases:
        redacted = tc._safe_outbound_text(value)
        if "<local-path>" not in redacted:
            raise SystemExit(f"Telegram outbound redaction missed a token-boundary local path: {value!r}")
    if tc._clean_cockpit_display("(/private/tmp/jarvis-proof.txt)") != "<redacted-local-path>":
        raise SystemExit("Telegram cockpit redaction missed a parenthesized local path")


def test_unreadable_runtime_reply_fails_closed() -> None:
    _seed_offset(0)
    _setup_env("555001")
    leak_marker = "SHOULD NOT APPEAR"

    class HostileText:
        def __str__(self) -> str:
            raise RuntimeError(f"{leak_marker} /\x55sers/hostile/runtime-reply")

    rt = FakeRuntime([HostileText()])
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: [_update(85, 555001, "show status")],
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 1:
        raise SystemExit("unreadable runtime reply command should still be processed")
    if rt.inputs != ["show status"]:
        raise SystemExit(f"unexpected runtime inputs: {rt.inputs}")
    if len(sent) != 1 or sent[0][2] is not None:
        raise SystemExit(f"unreadable runtime reply should be sent without buttons: {sent}")
    reply = sent[0][1]
    if "couldn't safely display" not in reply or "local Jarvis logs" not in reply:
        raise SystemExit(f"unreadable runtime reply did not use safe fallback: {reply!r}")
    for leaked in (leak_marker, "/\x55sers/hostile", "runtime-reply", "RuntimeError"):
        if leaked in reply:
            raise SystemExit(f"unreadable runtime reply leaked hostile details: {reply!r}")


def test_owner_approval_callback_without_readiness_fails_closed() -> None:
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(
        [
            "Approval #7 needs a current readiness receipt before its last-look packet.",
            "Readiness review required. No action ran.",
        ]
    )
    answered = []
    edited = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: None,
        fetch_func=lambda token, offset, timeout: [_callback_update(90, 555001, "approve:7")],
        answer_callback_func=lambda callback_id, text: answered.append((callback_id, text)),
        edit_message_func=lambda cid, mid, text, markup=None: edited.append((cid, mid, text, markup)),
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 1:
        raise SystemExit("owner approval callback was not processed")
    if rt.inputs != ["approval packet 7", "approve approval 7"]:
        raise SystemExit(f"approval command sequence was wrong: {rt.inputs}")
    if rt.request_tokens != [None, None]:
        raise SystemExit(f"approval callbacks must preserve one-shot semantics without ingress tokens: {rt.request_tokens}")
    if answered != [("callback-90", "Readiness review required. No action ran.")]:
        raise SystemExit(f"callback was not answered: {answered}")
    if not edited or edited[0][:3] != (
        "555001",
        90,
        "Readiness review required. No action ran.",
    ):
        raise SystemExit(f"approval message did not show the held outcome: {edited}")
    surfaced = "\n".join([text for _cid, text in answered] + [text for _cid, _mid, text, _markup in edited])
    for false_claim in ("approved approval #7", "sent."):
        if false_claim in surfaced.lower():
            raise SystemExit(f"one-click callback falsely reported approval or delivery: {surfaced!r}")


def test_owner_approval_callback_recovers_after_restart_and_executes_once() -> None:
    _seed_offset(0)
    _setup_env("555001")
    with tempfile.TemporaryDirectory(prefix="jarvis-telegram-callback-restart-") as temp:
        root = Path(temp)
        command = "send a telegram to me saying callback recovery proof"
        first_runtime = make_temp_runtime(root)
        held = first_runtime.handle(command)
        pending = first_runtime.store.list_pending_approvals(limit=5)
        if len(pending) != 1 or pending[0]["tool_name"] != "send_telegram":
            raise SystemExit(f"callback rehearsal did not queue one Telegram approval: {held}")
        approval_id = int(pending[0]["id"])

        owner_sends = []
        answers = []
        edits = []
        original_owner_send = call_connector_module._send_owner_telegram
        try:
            call_connector_module._send_owner_telegram = lambda message: owner_sends.append(message)
            bridge = tc.TelegramCommandBridge(
                runtime_factory=lambda: make_temp_runtime(root),
                send_func=lambda cid, text, markup=None: None,
                fetch_func=lambda token, offset, timeout: [
                    _callback_update(190, 555001, f"approve:{approval_id}")
                ],
                answer_callback_func=lambda callback_id, text: answers.append((callback_id, text)),
                edit_message_func=lambda cid, mid, text, markup=None: edits.append(
                    (cid, mid, text, markup)
                ),
                chat_action_func=_no_chat_action,
            )
            if bridge.process_once(poll_timeout=0) != 1:
                raise SystemExit("owner approval callback after restart was not processed")
            if owner_sends != ["callback recovery proof"]:
                raise SystemExit(f"owner approval callback did not execute exactly once: {owner_sends}")
            if not edits or "Telegram Bot API accepted the owner-chat send" not in edits[0][2]:
                raise SystemExit(f"owner approval callback did not surface the approved result: {edits}")
            if not answers or not answers[0][1]:
                raise SystemExit(f"owner approval callback was not acknowledged: {answers}")

            _seed_offset(191)
            replay_edits = []
            replay_bridge = tc.TelegramCommandBridge(
                runtime_factory=lambda: make_temp_runtime(root),
                send_func=lambda cid, text, markup=None: None,
                fetch_func=lambda token, offset, timeout: [
                    _callback_update(191, 555001, f"approve:{approval_id}")
                ],
                answer_callback_func=lambda callback_id, text: None,
                edit_message_func=lambda cid, mid, text, markup=None: replay_edits.append(
                    (cid, mid, text, markup)
                ),
                chat_action_func=_no_chat_action,
            )
            if replay_bridge.process_once(poll_timeout=0) != 1:
                raise SystemExit("replayed owner approval callback was not handled safely")
            if owner_sends != ["callback recovery proof"]:
                raise SystemExit(f"replayed callback repeated the side effect: {owner_sends}")
            if not replay_edits or "already been claimed" not in replay_edits[0][2]:
                raise SystemExit(f"replayed callback did not surface one-shot custody: {replay_edits}")
        finally:
            call_connector_module._send_owner_telegram = original_owner_send


def test_owner_deny_callback_dismisses_and_edits_message() -> None:
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(["Dismissed approval #8."])
    edited = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: None,
        fetch_func=lambda token, offset, timeout: [_callback_update(91, 555001, "deny:8")],
        answer_callback_func=lambda callback_id, text: None,
        edit_message_func=lambda cid, mid, text, markup=None: edited.append((cid, mid, text, markup)),
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 1:
        raise SystemExit("owner deny callback was not processed")
    if rt.inputs != ["dismiss approval 8"]:
        raise SystemExit(f"deny command sequence was wrong: {rt.inputs}")
    if not edited or "Dismissed approval #8." not in edited[0][2]:
        raise SystemExit(f"deny message was not edited to outcome: {edited}")


def test_owner_deny_callback_after_restart_dismisses_without_execution() -> None:
    _seed_offset(0)
    _setup_env("555001")
    with tempfile.TemporaryDirectory(prefix="jarvis-telegram-deny-restart-") as temp:
        root = Path(temp)
        command = "send a telegram to me saying deny callback proof"
        first_runtime = make_temp_runtime(root)
        first_runtime.handle(command)
        pending = first_runtime.store.list_pending_approvals(limit=5)
        if len(pending) != 1 or pending[0]["tool_name"] != "send_telegram":
            raise SystemExit(f"deny callback rehearsal did not queue one Telegram approval: {pending}")
        approval_id = int(pending[0]["id"])

        owner_sends = []
        edits = []
        original_owner_send = call_connector_module._send_owner_telegram
        try:
            call_connector_module._send_owner_telegram = lambda message: owner_sends.append(message)
            bridge = tc.TelegramCommandBridge(
                runtime_factory=lambda: make_temp_runtime(root),
                send_func=lambda cid, text, markup=None: None,
                fetch_func=lambda token, offset, timeout: [
                    _callback_update(192, 555001, f"deny:{approval_id}")
                ],
                answer_callback_func=lambda callback_id, text: None,
                edit_message_func=lambda cid, mid, text, markup=None: edits.append(
                    (cid, mid, text, markup)
                ),
                chat_action_func=_no_chat_action,
            )
            if bridge.process_once(poll_timeout=0) != 1:
                raise SystemExit("owner deny callback after restart was not processed")
            if owner_sends:
                raise SystemExit(f"owner deny callback executed the blocked side effect: {owner_sends}")
            dismissed = make_temp_runtime(root).store.get_approval(approval_id)
            if dismissed is None or str(dismissed["status"]).lower() != "dismissed":
                raise SystemExit(f"owner deny callback did not persist dismissal: {dismissed}")
            if not edits or f"Dismissed approval #{approval_id}." not in edits[0][2]:
                raise SystemExit(f"owner deny callback did not surface durable dismissal: {edits}")
        finally:
            call_connector_module._send_owner_telegram = original_owner_send


def test_unreadable_approval_callback_reply_fails_closed() -> None:
    _seed_offset(0)
    _setup_env("555001")
    leak_marker = "SHOULD NOT APPEAR"

    class HostileText:
        def __str__(self) -> str:
            raise RuntimeError(f"{leak_marker} /private/hostile-approval-callback")

    rt = FakeRuntime([HostileText(), HostileText()])
    answered = []
    edited = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: None,
        fetch_func=lambda token, offset, timeout: [_callback_update(93, 555001, "approve:7")],
        answer_callback_func=lambda callback_id, text: answered.append((callback_id, text)),
        edit_message_func=lambda cid, mid, text, markup=None: edited.append((cid, mid, text, markup)),
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 1:
        raise SystemExit("unreadable approval callback reply should still be processed")
    if rt.inputs != ["approval packet 7", "approve approval 7"]:
        raise SystemExit(f"approval command sequence was wrong: {rt.inputs}")
    if not answered or "couldn't safely display" not in answered[0][1]:
        raise SystemExit(f"callback answer should use safe fallback: {answered}")
    if not edited or "couldn't safely display" not in edited[0][2]:
        raise SystemExit(f"edited callback message should use safe fallback: {edited}")
    combined = "\n".join([text for _cid, text in answered] + [text for _cid, _mid, text, _markup in edited])
    for leaked in (leak_marker, "/private/hostile", "hostile-approval-callback", "RuntimeError"):
        if leaked in combined:
            raise SystemExit(f"unreadable approval callback leaked hostile details: {combined!r}")


def test_callback_toast_and_edit_redact_runtime_paths() -> None:
    private_path = "/\x55sers/example/private/callback-result.txt"

    class PathRuntime:
        def handle(self, _text: str, request_token=None):
            return SimpleNamespace(response=f"Approval result stored at {private_path}", tool_results=[])

    answered = []
    edited = []
    sent = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=PathRuntime,
        send_func=lambda cid, text, markup=None: sent.append((cid, text, markup)),
        fetch_func=lambda token, offset, timeout: [],
        answer_callback_func=lambda callback_id, text: answered.append((callback_id, text)),
        edit_message_func=lambda cid, mid, text, markup=None: edited.append((cid, mid, text, markup)),
        chat_action_func=_no_chat_action,
    )
    callback = _callback_update(193, 555001, "approve:7")["callback_query"]
    if not bridge._process_callback(callback, "555001"):
        raise SystemExit("path-shaped owner callback was not processed")
    surfaced = "\n".join(
        [text for _callback_id, text in answered]
        + [text for _cid, _mid, text, _markup in edited]
        + [text for _cid, text, _markup in sent]
    )
    if not answered or not edited or "<local-path>" not in surfaced:
        raise SystemExit(f"callback result did not use path redaction on every egress: {answered} {edited}")
    if private_path in surfaced or "/\x55sers/" in surfaced:
        raise SystemExit(f"callback result leaked a private path: {surfaced!r}")


def test_non_owner_callback_is_ignored() -> None:
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(["should not run"])
    answered = []
    edited = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: None,
        fetch_func=lambda token, offset, timeout: [_callback_update(92, 999999, "approve:7")],
        answer_callback_func=lambda callback_id, text: answered.append((callback_id, text)),
        edit_message_func=lambda cid, mid, text, markup=None: edited.append((cid, mid, text, markup)),
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 0:
        raise SystemExit("non-owner callback must be ignored")
    if rt.inputs or answered or edited:
        raise SystemExit(f"non-owner callback leaked: {rt.inputs} {answered} {edited}")


def test_owner_callback_in_non_owner_chat_is_ignored() -> None:
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(["should not run"])
    answered = []
    edited = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: None,
        fetch_func=lambda token, offset, timeout: [
            _callback_update(94, 555001, "approve:7", chat_id=-100123456)
        ],
        answer_callback_func=lambda callback_id, text: answered.append((callback_id, text)),
        edit_message_func=lambda cid, mid, text, markup=None: edited.append((cid, mid, text, markup)),
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 0:
        raise SystemExit("owner callback from a non-owner chat must be ignored")
    if rt.inputs or answered or edited:
        raise SystemExit(
            f"owner callback from non-owner chat leaked: {rt.inputs} {answered} {edited}"
        )


def test_owner_callback_without_message_chat_is_ignored() -> None:
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(["should not run"])
    answered = []
    edited = []
    sent = []
    malformed_updates = [
        {
            "update_id": 95,
            "callback_query": {
                "id": "callback-95",
                "from": {"id": 555001},
                "data": "approve:7",
            },
        },
        {
            "update_id": 96,
            "callback_query": {
                "id": "callback-96",
                "from": {"id": 555001},
                "data": "approve:7",
                "message": {"message_id": 90},
            },
        },
    ]
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: malformed_updates,
        answer_callback_func=lambda callback_id, text: answered.append((callback_id, text)),
        edit_message_func=lambda cid, mid, text, markup=None: edited.append((cid, mid, text, markup)),
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 0:
        raise SystemExit("owner callback without a verifiable chat must be ignored")
    if rt.inputs or answered or edited or sent:
        raise SystemExit(
            "owner callback without message chat leaked: "
            f"{rt.inputs} {answered} {edited} {sent}"
        )


def test_malformed_callback_fields_fail_closed() -> None:
    _seed_offset(0)
    _setup_env("555001")
    leak_marker = "SHOULD NOT APPEAR"

    class HostileField:
        def __str__(self) -> str:
            raise RuntimeError(f"{leak_marker} /\x55sers/hostile/callback-field")

        def __int__(self) -> int:
            raise RuntimeError(f"{leak_marker} /private/hostile-message-id")

    rt = FakeRuntime(["should not run"])
    answered = []
    edited = []
    sent = []
    updates = [
        {
            "update_id": 97,
            "callback_query": {
                "id": HostileField(),
                "from": {"id": HostileField()},
                "data": "approve:7",
                "message": {"message_id": 90, "chat": {"id": 555001}},
            },
        },
        {
            "update_id": 98,
            "callback_query": {
                "id": HostileField(),
                "from": {"id": 555001},
                "data": HostileField(),
                "message": {"message_id": 90, "chat": {"id": 555001}},
            },
        },
        {
            "update_id": 99,
            "callback_query": {
                "id": "callback-99",
                "from": {"id": 555001},
                "data": "approve:7",
                "message": {"message_id": 90, "chat": {"id": HostileField()}},
            },
        },
        {
            "update_id": 100,
            "callback_query": {
                "id": "callback-100",
                "from": {"id": 555001},
                "data": "approve:-1",
                "message": {"message_id": HostileField(), "chat": {"id": 555001}},
            },
        },
    ]
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: sent.append((cid, t, markup)),
        fetch_func=lambda token, offset, timeout: updates,
        answer_callback_func=lambda callback_id, text: answered.append((callback_id, text)),
        edit_message_func=lambda cid, mid, text, markup=None: edited.append((cid, mid, text, markup)),
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 1:
        raise SystemExit("only the verified owner callback with bad message_id should be handled")
    if rt.inputs:
        raise SystemExit(f"malformed callback fields reached approval runtime: {rt.inputs}")
    if len(answered) != 1 or answered[0][0] != "callback-100" or "fresh packet" not in answered[0][1]:
        raise SystemExit(f"malformed callback safe refusal answer wrong: {answered}")
    if edited:
        raise SystemExit(f"malformed callback with unreadable message id should not edit: {edited}")
    if len(sent) != 1 or sent[0][0] != "555001" or "fresh packet" not in sent[0][1] or sent[0][2] is not None:
        raise SystemExit(f"malformed callback safe refusal send wrong: {sent}")
    if tc._load_offset() != 101:
        raise SystemExit(f"offset should advance past handled callback: {tc._load_offset()}")
    combined = "\n".join([text for _cid, text in answered] + [text for _cid, text, _markup in sent])
    for leaked in (leak_marker, "/\x55sers/hostile", "/private/hostile", "callback-field"):
        if leaked in combined:
            raise SystemExit(f"malformed callback fields leaked hostile details: {combined!r}")


def test_invalid_owner_callback_id_is_refused_before_runtime() -> None:
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(["should not run"])
    answered = []
    edited = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: None,
        fetch_func=lambda token, offset, timeout: [_callback_update(93, 555001, "approve:-1")],
        answer_callback_func=lambda callback_id, text: answered.append((callback_id, text)),
        edit_message_func=lambda cid, mid, text, markup=None: edited.append((cid, mid, text, markup)),
        chat_action_func=_no_chat_action,
    )
    if bridge.process_once(poll_timeout=0) != 1:
        raise SystemExit("invalid owner callback should be handled with a refusal")
    if rt.inputs:
        raise SystemExit(f"invalid callback id reached approval runtime: {rt.inputs}")
    if len(answered) != 1 or answered[0][0] != "callback-93" or "fresh packet" not in answered[0][1]:
        raise SystemExit(f"invalid callback was not answered safely: {answered}")
    if not edited or edited[0][0] != "555001" or edited[0][1] != 90 or "Invalid approval action." not in edited[0][2]:
        raise SystemExit(f"invalid callback message was not edited to refusal: {edited}")


def test_typing_indicator_sent_before_owner_command() -> None:
    _seed_offset(0)
    _setup_env("555001")
    rt = FakeRuntime(["done"])
    events = []
    bridge = tc.TelegramCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda cid, t, markup=None: events.append(("send", cid, t)),
        fetch_func=lambda token, offset, timeout: [_update(100, 555001, "summarize my day")],
        chat_action_func=lambda cid, action: events.append(("action", cid, action)),
    )
    if bridge.process_once(poll_timeout=0) != 1:
        raise SystemExit("owner command was not processed")
    if events[:2] != [("action", "555001", "typing"), ("send", "555001", "done")]:
        raise SystemExit(f"typing indicator was not sent before reply: {events}")


def test_typing_indicator_refreshes_while_thinking() -> None:
    """Telegram's typing status expires after ~5s; for a slow command the bridge
    must keep re-sending it, not just once, so it never looks offline."""
    import time

    _seed_offset(0)
    _setup_env("555001")
    typing_calls = []

    class SlowRuntime:
        inputs: list = []

        def handle(self, text, request_token=None):
            SlowRuntime.inputs.append(text)
            time.sleep(0.25)  # simulate a model that takes a while
            return SimpleNamespace(response="done", tool_results=[])

    original_refresh = tc.TYPING_REFRESH_SECONDS
    tc.TYPING_REFRESH_SECONDS = 0.05  # speed up refresh for the test
    try:
        bridge = tc.TelegramCommandBridge(
            runtime_factory=lambda: SlowRuntime(),
            send_func=lambda cid, t, markup=None: None,
            fetch_func=lambda token, offset, timeout: [_update(101, 555001, "think hard about my week")],
            chat_action_func=lambda cid, action: typing_calls.append(action),
        )
        if bridge.process_once(poll_timeout=0) != 1:
            raise SystemExit("slow command was not processed")
    finally:
        tc.TYPING_REFRESH_SECONDS = original_refresh
    # 1 synchronous ping + several refreshes over 0.25s at 0.05s interval.
    if len(typing_calls) < 2:
        raise SystemExit(f"typing was not kept alive during a slow command: {typing_calls}")


def test_send_message_chunks_long_replies_and_keeps_markup_once() -> None:
    _setup_env("555001")
    calls = []
    original_api_call = tc._api_call
    try:
        tc._api_call = lambda token, method, params, timeout: calls.append((method, params)) or {"ok": True}
        markup = {"inline_keyboard": [[{"text": "Approve", "callback_data": "approve:1"}]]}
        result = tc.send_message("555001", "x" * 8200, reply_markup=markup)
    finally:
        tc._api_call = original_api_call
    if not result.get("ok"):
        raise SystemExit(f"chunked send did not report success: {result}")
    if len(calls) != 3:
        raise SystemExit(f"long reply should have been chunked into three sends: {len(calls)}")
    if any(method != "sendMessage" for method, _ in calls):
        raise SystemExit(f"unexpected Telegram methods: {calls}")
    if any(len(params["text"]) > tc.TELEGRAM_TEXT_LIMIT for _, params in calls):
        raise SystemExit(f"chunk exceeded Telegram limit: {[len(params['text']) for _, params in calls]}")
    markup_counts = sum(1 for _, params in calls if "reply_markup" in params)
    if markup_counts != 1 or "reply_markup" not in calls[0][1]:
        raise SystemExit(f"reply markup should be attached only to the first chunk: {calls}")


def test_send_message_redacts_local_paths() -> None:
    _setup_env("555001")
    calls = []
    original_api_call = tc._api_call
    try:
        tc._api_call = lambda token, method, params, timeout: calls.append((method, params)) or {"ok": True}
        result = tc.send_message("555001", "Proof: /\x55sers/example/private/report.md\nTemp: /private/tmp/jarvis-proof.txt\nVar: /var/folders/zc/jarvis-proof.txt\nTmp: /tmp/jarvis-proof.txt")
    finally:
        tc._api_call = original_api_call
    if not result.get("ok") or len(calls) != 1:
        raise SystemExit(f"path-redacted send should succeed once: result={result} calls={calls}")
    sent_text = calls[0][1].get("text", "")
    if any(fragment in sent_text for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]) or "<local-path>" not in sent_text:
        raise SystemExit(f"send_message should redact local paths before Telegram API: {calls}")


def test_parse_mode_is_passed_to_send_and_edit() -> None:
    _setup_env("555001")
    calls = []
    original_api_call = tc._api_call
    try:
        tc._api_call = lambda token, method, params, timeout: calls.append((method, params)) or {"ok": True}
        tc.send_message("555001", "*hello*", parse_mode="Markdown")
        tc.edit_message_text("555001", 9, "*updated*", parse_mode="Markdown")
    finally:
        tc._api_call = original_api_call
    if calls != [
        ("sendMessage", {"chat_id": "555001", "text": "*hello*", "parse_mode": "Markdown"}),
        (
            "editMessageText",
            {"chat_id": "555001", "message_id": 9, "text": "*updated*", "parse_mode": "Markdown"},
        ),
    ]:
        raise SystemExit(f"parse_mode was not passed through: {calls}")


def test_state_file_reads_environment_at_call_time() -> None:
    old_file = tc.STATE_FILE
    old_env = os.environ.get("JARVIS_TELEGRAM_STATE")
    temp = Path(tempfile.mkdtemp()) / "env-telegram.json"
    try:
        tc.STATE_FILE = None
        os.environ["JARVIS_TELEGRAM_STATE"] = str(temp)
        tc._save_offset(123)
        if not temp.exists():
            raise SystemExit("JARVIS_TELEGRAM_STATE was not honored at save time")
        if tc._load_offset() != 123:
            raise SystemExit(f"JARVIS_TELEGRAM_STATE was not honored at load time: {tc._load_offset()}")
    finally:
        tc.STATE_FILE = old_file
        if old_env is None:
            os.environ.pop("JARVIS_TELEGRAM_STATE", None)
        else:
            os.environ["JARVIS_TELEGRAM_STATE"] = old_env


def test_blank_state_file_env_uses_default() -> None:
    old_file = tc.STATE_FILE
    old_env = os.environ.get("JARVIS_TELEGRAM_STATE")
    try:
        tc.STATE_FILE = None
        os.environ["JARVIS_TELEGRAM_STATE"] = "   "
        expected = Path(pwd.getpwuid(os.geteuid()).pw_dir) / ".jarvis_v3" / "telegram_control.json"
        if tc._state_file() != expected:
            raise SystemExit(f"blank JARVIS_TELEGRAM_STATE should use default: {tc._state_file()}")
    finally:
        tc.STATE_FILE = old_file
        if old_env is None:
            os.environ.pop("JARVIS_TELEGRAM_STATE", None)
        else:
            os.environ["JARVIS_TELEGRAM_STATE"] = old_env


def main() -> None:
    test_telegram_only_activation_starts_no_scheduler_or_jobs()
    test_invalid_offset_custody_blocks_model_scheduler_and_bridge_startup()
    test_embedded_scheduler_initializes_dependencies_before_first_tick()
    test_embedded_scheduler_setup_failure_retries_then_recovers()
    test_embedded_scheduler_tick_failure_retries_then_recovers()
    test_owner_command_runs_and_offset_advances()
    test_voice_memo_is_transcribed_and_routed()
    test_voice_memo_korean_status_shortcut_routes_directly()
    test_voice_memo_preserves_bilingual_transcript_and_approval_buttons()
    test_voice_transcription_failure_is_reported_without_routing()
    test_media_failure_guidance_names_safe_recovery()
    test_telegram_https_uses_verified_certifi_context_and_hides_tls_failures()
    test_unreadable_voice_transcription_result_fails_closed()
    test_unreadable_voice_error_fails_closed()
    test_oversized_voice_file_id_is_skipped()
    test_photo_is_ocred_and_surfaced_not_routed()
    test_photo_ocr_error_is_reported()
    test_photo_ocr_failure_omits_private_caption_and_guides_recovery()
    test_photo_caption_is_surfaced_with_ocr_text()
    test_unreadable_photo_ocr_result_fails_closed()
    test_unreadable_photo_ocr_error_fails_closed()
    test_image_file_id_picks_largest_within_cap()
    test_non_owner_is_ignored()
    test_non_owner_media_is_ignored_before_private_processing()
    test_fresh_install_drains_backlog()
    test_fresh_install_ignores_malformed_update_ids_when_draining()
    test_malformed_update_id_is_ignored_during_polling()
    test_transport_update_tokens_are_exact_stable_and_content_free()
    test_malformed_transport_update_id_types_fail_closed()
    test_malformed_update_fields_do_not_break_polling()
    test_missing_token_or_owner_disables()
    test_path_shaped_token_and_owner_are_rejected_locally()
    test_start_and_help_commands()
    test_phone_status_shortcuts_map_to_read_only_commands()
    test_phone_control_shortcuts_share_real_runtime_state()
    test_personal_proof_matrix_shortcuts_are_read_only_and_private()
    test_scheduler_proof_matrix_shortcuts_are_read_only_and_private()
    test_daily_value_proof_matrix_shortcuts_are_read_only_and_private()
    test_approval_proof_matrix_shortcuts_are_read_only_and_private()
    test_conversation_research_proof_matrix_shortcuts_are_read_only_and_private()
    test_mixed_conversation_proof_matrix_shortcuts_are_read_only_and_private()
    test_voice_proof_matrix_shortcuts_are_read_only_and_private()
    test_phone_control_proof_matrix_shortcuts_are_read_only_and_private()
    test_reboot_proof_matrix_shortcuts_are_read_only_and_private()
    test_error_guidance_proof_matrix_shortcuts_are_read_only_and_private()
    test_agent_research_shortcuts_route_to_work_queue_read_only()
    test_build_progress_and_agi_shortcuts_route_read_only()
    test_phone_chat_latency_shortcuts_route_to_model_status_read_only()
    test_handoff_and_change_shortcuts_route_read_only()
    test_completion_status_shortcuts_route_to_claim_gate_read_only()
    test_next_cockpit_action_shortcuts_are_read_only()
    test_next_cockpit_action_metadata_fails_closed()
    test_cockpit_attention_shortcuts_are_read_only()
    test_cockpit_attention_metadata_fails_closed()
    test_cockpit_lane_shortcuts_are_read_only()
    test_acceptance_harness_shortcuts_open_lane_read_only()
    test_acceptance_next_proof_phone_shortcuts_use_next_proof_packet()
    test_cockpit_lane_metadata_fails_closed()
    test_trust_test_shortcuts_open_operator_workflow_eval_lane_read_only()
    test_cockpit_summary_shortcuts_are_read_only()
    test_cockpit_summary_direction_metadata_fails_closed()
    test_cockpit_summary_metadata_fails_closed()
    test_cockpit_proof_shortcuts_are_read_only()
    test_cockpit_proof_metadata_fails_closed()
    test_korean_phone_status_shortcuts_map_to_read_only_commands()
    test_phone_failure_error_shortcuts_route_to_read_only_diagnostics()
    test_phone_audit_history_shortcuts_route_to_read_only_audit_tools()
    test_english_phone_delivery_status_questions_route_to_channel_health()
    test_korean_phone_delivery_status_questions_route_to_channel_health()
    test_korean_phone_voice_shortcuts_map_to_read_only_commands()
    test_help_mentions_status_shortcuts()
    test_pending_approval_sends_inline_buttons()
    test_invalid_metadata_approval_id_does_not_send_inline_buttons()
    test_invalid_trace_approval_id_does_not_send_inline_buttons()
    test_text_approval_receipt_sends_inline_buttons()
    test_invalid_text_approval_receipt_does_not_send_inline_buttons()
    test_generic_approval_text_does_not_get_inline_buttons()
    test_runtime_exception_reply_is_non_leaky()
    test_runtime_reply_redacts_local_paths_before_send()
    test_outbound_path_redaction_preserves_slash_separated_prose()
    test_unreadable_runtime_reply_fails_closed()
    test_owner_approval_callback_without_readiness_fails_closed()
    test_owner_approval_callback_recovers_after_restart_and_executes_once()
    test_owner_deny_callback_dismisses_and_edits_message()
    test_owner_deny_callback_after_restart_dismisses_without_execution()
    test_unreadable_approval_callback_reply_fails_closed()
    test_callback_toast_and_edit_redact_runtime_paths()
    test_non_owner_callback_is_ignored()
    test_owner_callback_in_non_owner_chat_is_ignored()
    test_owner_callback_without_message_chat_is_ignored()
    test_malformed_callback_fields_fail_closed()
    test_invalid_owner_callback_id_is_refused_before_runtime()
    test_typing_indicator_sent_before_owner_command()
    test_typing_indicator_refreshes_while_thinking()
    test_send_message_chunks_long_replies_and_keeps_markup_once()
    test_send_message_redacts_local_paths()
    test_parse_mode_is_passed_to_send_and_edit()
    test_state_file_reads_environment_at_call_time()
    test_blank_state_file_env_uses_default()
    print("Telegram control smoke passed")


if __name__ == "__main__":
    main()
