"""Smoke tests for the local push-to-talk entrypoint (no mic, no network, no whisper)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import wave
from pathlib import Path

from jarvis_v2.agent.types import ToolResult
from jarvis_v2.scripts import talk


def _reset_voice_warmup_for_smoke() -> None:
    with talk._VOICE_WARMUP_LOCK:
        talk._VOICE_WARMUP_STARTED = False


def _commands(*available: str):
    names = set(available)
    return lambda name: f"synthetic-{name}" if name in names else None


def _no_modules(_name: str):
    return None


def test_readiness_reports_missing_ffmpeg_without_audio_access() -> None:
    snapshot = talk._voice_readiness_snapshot(
        environ={},
        command_finder=_commands("whisper", "say"),
        module_finder=_no_modules,
    )
    if snapshot["ready"] is not False or snapshot["ffmpeg_ready"] is not False:
        raise SystemExit(f"missing ffmpeg must hold voice readiness: {snapshot}")
    if "ffmpeg is unavailable" not in snapshot["issues"]:
        raise SystemExit(f"missing ffmpeg needs bounded recovery: {snapshot}")
    if snapshot["microphone_opened"] or snapshot["audio_recorded"] or snapshot["private_data_read"]:
        raise SystemExit(f"readiness check crossed its privacy boundary: {snapshot}")


def test_readiness_reports_missing_transcriber() -> None:
    snapshot = talk._voice_readiness_snapshot(
        environ={},
        command_finder=_commands("ffmpeg", "say"),
        module_finder=_no_modules,
    )
    if snapshot["ready"] is not False or snapshot["transcriber_ready"] is not False:
        raise SystemExit(f"missing transcriber must hold voice readiness: {snapshot}")
    if "no local transcriber is available" not in snapshot["issues"]:
        raise SystemExit(f"missing transcriber needs an explicit safe diagnosis: {snapshot}")


def test_readiness_rejects_invalid_configuration_without_path_leakage() -> None:
    secret = "VOICE-PRIVATE-PATH-8df50"
    snapshot = talk._voice_readiness_snapshot(
        environ={
            "JARVIS_VOICE_WHISPER_CLI": f"/not-present/{secret}/whisper",
            "JARVIS_VOICE_WHISPER_CLI_MODEL": "unsupported-private-model",
            "JARVIS_VOICE_WHISPER_CLI_LANGUAGE": "not-a-language",
            "JARVIS_VOICE_MAX_SECONDS": "99999",
            "JARVIS_VOICE_MIC": "private-device-name",
        },
        command_finder=_commands("ffmpeg", "say"),
        module_finder=_no_modules,
    )
    report = talk._format_voice_readiness(snapshot)
    if snapshot["configuration_valid"] is not False or snapshot["ready"] is not False:
        raise SystemExit(f"invalid settings must hold voice readiness: {snapshot}")
    if secret in repr(snapshot) or secret in report or "unsupported-private-model" in report:
        raise SystemExit(f"voice readiness leaked configured private values: {report}")
    if len(snapshot["issues"]) > 6 or len(snapshot["recovery"]) > 4 or len(report) > 2400:
        raise SystemExit(f"voice recovery must remain bounded: {snapshot} {len(report)}")
    for required in ("needs attention", "Microphone access", "does not list microphones", "Overall: NOT READY"):
        if required not in report:
            raise SystemExit(f"voice readiness report omitted {required!r}: {report}")


def test_readiness_loads_only_voice_settings_from_selected_v3_environment() -> None:
    private_token_key = "TELEGRAM_" + "BOT_TOKEN"
    private_token = "PRIVATE-CREDENTIAL-MUST-NOT-RETURN-71d2"
    with tempfile.TemporaryDirectory(prefix="jarvis-voice-readiness-") as tmp:
        root = Path(tmp)
        cli = root / "whisper"
        cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        cli.chmod(0o700)
        selected_env = root / "runtime.env"
        selected_env.write_text(
            "\n".join(
                [
                    f"JARVIS_VOICE_WHISPER_CLI={cli}",
                    "JARVIS_VOICE_WHISPER_CLI_MODEL=base",
                    f"{private_token_key}={private_token}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        selected_env.chmod(0o600)
        env, issue = talk._voice_readiness_environment({"JARVIS_V3_ENV": str(selected_env)})
        if issue:
            raise SystemExit(f"valid selected V3 voice settings should load: {issue}")
        if env.get("JARVIS_VOICE_WHISPER_CLI") != str(cli):
            raise SystemExit("selected V3 env voice command was not loaded")
        if private_token_key in env or private_token in repr(env):
            raise SystemExit("voice readiness returned a non-voice credential from the selected environment")
        snapshot = talk._voice_readiness_snapshot(
            environ=env,
            command_finder=_commands("ffmpeg", "say"),
            module_finder=_no_modules,
        )
        report = talk._format_voice_readiness(snapshot)
        if not snapshot["ready"] or private_token in report or str(root) in report:
            raise SystemExit(f"selected-env readiness was not ready or leaked private values: {report}")


def test_readiness_bounds_selected_environment_failure() -> None:
    secret = "VOICE-ENV-PATH-MUST-NOT-LEAK-992f"
    env, issue = talk._voice_readiness_environment({"JARVIS_V3_ENV": f"/missing/{secret}/runtime.env"})
    snapshot = talk._voice_readiness_snapshot(
        environ=env,
        command_finder=_commands("ffmpeg", "whisper", "say"),
        module_finder=_no_modules,
        environment_issue=issue,
    )
    report = talk._format_voice_readiness(snapshot)
    if snapshot["ready"] or snapshot["configuration_valid"]:
        raise SystemExit(f"unreadable selected environment must hold readiness: {snapshot}")
    if secret in issue or secret in report or len(report) > 2400:
        raise SystemExit(f"selected environment failure leaked or was unbounded: {report}")


def test_check_command_never_lists_or_opens_microphones() -> None:
    original_snapshot = talk._voice_readiness_snapshot
    original_list = talk._list_audio_devices
    original_record = talk._record_clip
    original_argv = sys.argv

    def fail_audio_access(*_args, **_kwargs):
        raise SystemExit("readiness command must not access an audio device")

    talk._voice_readiness_snapshot = lambda **_kwargs: {  # type: ignore[assignment]
        "ready": True,
        "ffmpeg_ready": True,
        "transcriber_ready": True,
        "transcriber_source": "Whisper command",
        "speech_ready": True,
        "configuration_valid": True,
        "issues": (),
        "recovery": ("Start push-to-talk yourself when ready.",),
        "microphone_opened": False,
        "audio_recorded": False,
        "private_data_read": False,
    }
    talk._list_audio_devices = fail_audio_access  # type: ignore[assignment]
    talk._record_clip = fail_audio_access  # type: ignore[assignment]
    sys.argv = ["talk", "--check"]
    try:
        talk.main()
    finally:
        sys.argv = original_argv
        talk._record_clip = original_record  # type: ignore[assignment]
        talk._list_audio_devices = original_list  # type: ignore[assignment]
        talk._voice_readiness_snapshot = original_snapshot  # type: ignore[assignment]


def test_real_launch_holds_before_microphone_when_readiness_fails() -> None:
    original_environment = talk._voice_readiness_environment
    original_snapshot = talk._voice_readiness_snapshot
    original_list = talk._list_audio_devices
    original_default = talk._default_mic_index
    original_record = talk._record_clip
    original_which = talk.shutil.which
    original_argv = sys.argv
    touched: list[str] = []

    def fail_audio_access(*_args, **_kwargs):
        touched.append("audio")
        raise SystemExit("unready launch must not inspect or open an audio device")

    talk._voice_readiness_environment = lambda: ({}, "")  # type: ignore[assignment]
    talk._voice_readiness_snapshot = lambda **_kwargs: {  # type: ignore[assignment]
        "ready": False,
        "ffmpeg_ready": True,
        "transcriber_ready": False,
        "transcriber_source": "not configured",
        "speech_ready": True,
        "configuration_valid": True,
        "issues": ("no local transcriber is available",),
        "recovery": ("Install a supported local transcriber, then check again.",),
        "microphone_opened": False,
        "audio_recorded": False,
        "private_data_read": False,
    }
    talk._list_audio_devices = fail_audio_access  # type: ignore[assignment]
    talk._default_mic_index = fail_audio_access  # type: ignore[assignment]
    talk._record_clip = fail_audio_access  # type: ignore[assignment]
    talk.shutil.which = lambda name: f"synthetic-{name}"  # type: ignore[assignment]
    sys.argv = ["talk"]
    try:
        try:
            talk.main()
        except SystemExit as exc:
            if exc.code != 2:
                raise SystemExit(f"unready voice launch should exit 2, got {exc.code!r}")
        else:
            raise SystemExit("unready voice launch should stop before microphone access")
    finally:
        sys.argv = original_argv
        talk.shutil.which = original_which  # type: ignore[assignment]
        talk._record_clip = original_record  # type: ignore[assignment]
        talk._default_mic_index = original_default  # type: ignore[assignment]
        talk._list_audio_devices = original_list  # type: ignore[assignment]
        talk._voice_readiness_snapshot = original_snapshot  # type: ignore[assignment]
        talk._voice_readiness_environment = original_environment  # type: ignore[assignment]

    if touched:
        raise SystemExit(f"unready voice launch crossed the audio boundary: {touched}")


def test_parses_audio_device_list() -> None:
    stderr = (
        "[AVFoundation indev] AVFoundation video devices:\n"
        "[AVFoundation indev] [0] FaceTime HD Camera\n"
        "[AVFoundation indev] AVFoundation audio devices:\n"
        "[AVFoundation indev] [0] Someone's iPhone Microphone\n"
        "[AVFoundation indev] [1] MacBook Pro Microphone\n"
    )
    original = talk.subprocess.run
    talk.subprocess.run = lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="", stderr=stderr)  # type: ignore
    try:
        devices = talk._list_audio_devices()
    finally:
        talk.subprocess.run = original  # type: ignore
    if devices != [(0, "Someone's iPhone Microphone"), (1, "MacBook Pro Microphone")]:
        raise SystemExit(f"audio device parsing wrong: {devices}")


def test_default_mic_prefers_builtin_over_iphone() -> None:
    import os

    os.environ.pop("JARVIS_VOICE_MIC", None)
    original = talk._list_audio_devices
    talk._list_audio_devices = lambda: [(0, "Someone's iPhone Microphone"), (1, "MacBook Pro Microphone")]  # type: ignore
    try:
        if talk._default_mic_index() != "1":
            raise SystemExit("should prefer the MacBook mic over the iPhone mic")
        os.environ["JARVIS_VOICE_MIC"] = "0"
        if talk._default_mic_index() != "0":
            raise SystemExit("explicit JARVIS_VOICE_MIC should win")
    finally:
        talk._list_audio_devices = original  # type: ignore
        os.environ.pop("JARVIS_VOICE_MIC", None)


def test_run_turn_routes_transcript_through_runtime() -> None:
    seen = {}

    class FakeRuntime:
        def handle(self, text):
            seen["text"] = text
            return SimpleNamespace(response="It is 2 PM.")

    talk.run_turn(FakeRuntime(), "what time is it", speak=False)
    if seen.get("text") != "what time is it":
        raise SystemExit(f"transcript not routed through runtime: {seen}")


def test_transcribe_passes_explicit_language_to_voice_layer() -> None:
    import jarvis_v2.tools.voice as voice

    seen = {}
    original = voice._audio_file_transcriber

    def fake_transcriber(*, language=None):
        seen["language"] = language
        return lambda _path: "현재 시간 알려줘"

    voice._audio_file_transcriber = fake_transcriber  # type: ignore[assignment]
    try:
        transcript = talk._transcribe(Path("ignored.wav"), language="ko")
    finally:
        voice._audio_file_transcriber = original  # type: ignore[assignment]
    if transcript != "현재 시간 알려줘" or seen.get("language") != "ko":
        raise SystemExit(f"talk should pass an explicit Korean ASR preference: {seen}, {transcript!r}")


def test_run_turn_does_not_auto_approve_side_effects(capsys=None) -> None:
    import io
    import contextlib

    class ApprovalRuntime:
        def handle(self, text):
            return SimpleNamespace(
                response="Safety receipt: queued as approval #7",
                tool_results=[SimpleNamespace(metadata={"requires_confirmation": True, "approval_id": 7})],
            )

    spoken = {}

    # Voice output must be the non-acting message, not a blind confirmation.
    import jarvis_v2.tools.voice as voice
    original_speak = voice.speak_text
    def capture_speak(text, *_args, **_kwargs):
        spoken["text"] = text
        return ToolResult("speak", True, "Spoke safely.")

    voice.speak_text = capture_speak  # type: ignore[assignment]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            talk.run_turn(ApprovalRuntime(), "text fixture saying hi", speak=True)
    finally:
        voice.speak_text = original_speak  # type: ignore
    out = buf.getvalue()
    if "#7" not in out or "approval" not in out.lower():
        raise SystemExit(f"approval-needed turn should surface the approval id: {out!r}")
    if "approve" not in spoken.get("text", "").lower():
        raise SystemExit(f"voice should say it needs approval, not auto-confirm: {spoken}")


def test_run_turn_surfaces_speech_failure() -> None:
    import contextlib
    import io

    class FakeRuntime:
        def handle(self, _text):
            return SimpleNamespace(response="A spoken reply.")

    import jarvis_v2.tools.voice as voice

    original_speak = voice.speak_text
    voice.speak_text = lambda *_args, **_kwargs: ToolResult(
        "speak",
        False,
        "Speech is unavailable. Check macOS speech settings and try again.",
    )  # type: ignore[assignment]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            talk.run_turn(FakeRuntime(), "say hello", speak=True)
    finally:
        voice.speak_text = original_speak  # type: ignore[assignment]

    out = buf.getvalue()
    if "Speech output failed" not in out or "Check macOS speech settings" not in out:
        raise SystemExit(f"speech failure should be visible with safe recovery guidance: {out!r}")


def test_run_turn_does_not_report_successful_speech_as_failure() -> None:
    import contextlib
    import io

    class FakeRuntime:
        def handle(self, _text):
            return SimpleNamespace(response="A spoken reply.")

    import jarvis_v2.tools.voice as voice

    original_speak = voice.speak_text
    voice.speak_text = lambda *_args, **_kwargs: ToolResult("speak", True, "Spoke safely.")  # type: ignore[assignment]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            talk.run_turn(FakeRuntime(), "say hello", speak=True)
    finally:
        voice.speak_text = original_speak  # type: ignore[assignment]

    if "Speech output failed" in buf.getvalue():
        raise SystemExit(f"successful speech should not report a failure: {buf.getvalue()!r}")


def test_run_turn_uses_text_selected_korean_voice_for_localized_time() -> None:
    class FakeRuntime:
        def handle(self, _text):
            return SimpleNamespace(response="현재 시간은 2026년 8월 3일 월요일 오후 11시 7분입니다.")

    import jarvis_v2.tools.voice as voice

    original_speak = voice.speak_text
    spoken = {}

    def capture_speak(text, *, voice="", **_kwargs):
        dry = original_speak(text, voice=voice, dry_run=True)
        spoken.update(text=text, requested_voice=voice, selected_voice=dry.metadata.get("voice"))
        return dry

    voice.speak_text = capture_speak  # type: ignore[assignment]
    try:
        talk.run_turn(FakeRuntime(), "지금 몇 시에요?", speak=True)
    finally:
        voice.speak_text = original_speak  # type: ignore[assignment]

    if spoken != {
        "text": "현재 시간은 2026년 8월 3일 월요일 오후 11시 7분입니다.",
        "requested_voice": "",
        "selected_voice": voice.MACOS_KOREAN_VOICE,
    }:
        raise SystemExit(f"Localized Korean time reply should select Yuna by text: {spoken}")


def test_run_turn_does_not_force_korean_voice_for_english_only_reply() -> None:
    class FakeRuntime:
        def handle(self, _text):
            return SimpleNamespace(response="I found no reminders.")

    import jarvis_v2.tools.voice as voice

    original_speak = voice.speak_text
    spoken = {}

    def capture_speak(text, *, voice="", **_kwargs):
        dry = original_speak(text, voice=voice, dry_run=True)
        spoken.update(text=text, requested_voice=voice, selected_voice=dry.metadata.get("voice"))
        return dry

    voice.speak_text = capture_speak  # type: ignore[assignment]
    try:
        talk.run_turn(FakeRuntime(), "알림 보여 주세요", speak=True)
    finally:
        voice.speak_text = original_speak  # type: ignore[assignment]

    if spoken != {
        "text": "I found no reminders.",
        "requested_voice": "",
        "selected_voice": "",
    }:
        raise SystemExit(f"English-only reply should retain the system voice: {spoken}")


def test_run_turn_surfaces_speech_exception_without_leaking_it() -> None:
    import contextlib
    import io

    class FakeRuntime:
        def handle(self, _text):
            return SimpleNamespace(response="A spoken reply.")

    import jarvis_v2.tools.voice as voice

    original_speak = voice.speak_text

    def fail_speak(*_args, **_kwargs):
        raise RuntimeError("/private/tmp/voice SECRET SHOULD NOT APPEAR")

    voice.speak_text = fail_speak  # type: ignore[assignment]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            talk.run_turn(FakeRuntime(), "say hello", speak=True)
    finally:
        voice.speak_text = original_speak  # type: ignore[assignment]

    out = buf.getvalue()
    if "Speech output failed. Check macOS speech settings, then try again." not in out:
        raise SystemExit(f"speech exception should return bounded recovery guidance: {out!r}")
    for forbidden in ("/private/tmp", "SECRET SHOULD NOT APPEAR", "RuntimeError"):
        if forbidden in out:
            raise SystemExit(f"speech exception leaked implementation detail {forbidden!r}: {out!r}")


def test_max_record_seconds_bounds() -> None:
    cases = {"": 120, "30": 30, "abc": 120, "0": 120, "5000": 120, "600": 600, "5": 5}
    for raw, expected in cases.items():
        if raw:
            os.environ["JARVIS_VOICE_MAX_SECONDS"] = raw
        else:
            os.environ.pop("JARVIS_VOICE_MAX_SECONDS", None)
        if talk._max_record_seconds() != expected:
            raise SystemExit(f"max record seconds for {raw!r} should be {expected}, got {talk._max_record_seconds()}")
    os.environ.pop("JARVIS_VOICE_MAX_SECONDS", None)


def test_voice_warmup_is_disabled_by_default() -> None:
    _reset_voice_warmup_for_smoke()
    os.environ.pop("JARVIS_VOICE_WARMUP", None)

    def fail_thread_factory(**_kwargs):
        raise SystemExit("warmup should not create a thread when disabled")

    if talk.start_voice_warmup_if_enabled(thread_factory=fail_thread_factory):
        raise SystemExit("warmup should return False when disabled")


def test_voice_warmup_starts_once_in_background() -> None:
    _reset_voice_warmup_for_smoke()
    os.environ["JARVIS_VOICE_WARMUP"] = "1"
    created = []
    started = []

    class FakeThread:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def start(self):
            started.append(True)

    try:
        if not talk.start_voice_warmup_if_enabled(source="dashboard ui", thread_factory=FakeThread):
            raise SystemExit("enabled warmup should start one background thread")
        if talk.start_voice_warmup_if_enabled(source="dashboard ui", thread_factory=FakeThread):
            raise SystemExit("warmup should be one-shot per process")
    finally:
        os.environ.pop("JARVIS_VOICE_WARMUP", None)
        _reset_voice_warmup_for_smoke()

    if len(created) != 1 or len(started) != 1:
        raise SystemExit(f"warmup should create/start exactly one thread: created={created}, started={started}")
    kwargs = created[0]
    if kwargs.get("target") is not talk._run_voice_warmup:
        raise SystemExit("warmup thread should target _run_voice_warmup")
    if kwargs.get("daemon") is not True:
        raise SystemExit("warmup thread must be daemonized so UI startup is not held open")
    if kwargs.get("name") != "jarvis-voice-warmup-dashboard-ui":
        raise SystemExit(f"warmup thread name should be source-scoped: {kwargs.get('name')!r}")


def test_voice_warmup_transcribes_silent_wav_without_mic() -> None:
    original_list_audio_devices = talk._list_audio_devices
    original_record_clip = talk._record_clip
    seen = {}

    def fail_list_audio_devices():
        raise SystemExit("warmup must not inspect microphones")

    def fail_record_clip(_mic_index, _dest):
        raise SystemExit("warmup must not record from the microphone")

    def fake_transcriber(path):
        seen["exists_during_transcribe"] = path.exists()
        with wave.open(str(path), "rb") as wav:
            if wav.getnchannels() != 1:
                raise SystemExit("warmup wav should be mono")
            if wav.getsampwidth() != 2:
                raise SystemExit("warmup wav should be 16-bit PCM")
            if wav.getframerate() != 16000:
                raise SystemExit("warmup wav should use 16 kHz audio")
            if wav.getnframes() != 4800:
                raise SystemExit(f"warmup wav should be 0.3s at 16 kHz, got {wav.getnframes()} frames")
            frames = wav.readframes(wav.getnframes())
            if frames != b"\x00\x00" * 4800:
                raise SystemExit("warmup wav should contain only silence")
        seen["path_after_check"] = path
        return ""

    talk._list_audio_devices = fail_list_audio_devices  # type: ignore[assignment]
    talk._record_clip = fail_record_clip  # type: ignore[assignment]
    try:
        talk._run_voice_warmup(transcriber=fake_transcriber)
    finally:
        talk._list_audio_devices = original_list_audio_devices  # type: ignore[assignment]
        talk._record_clip = original_record_clip  # type: ignore[assignment]

    if not seen.get("exists_during_transcribe"):
        raise SystemExit("warmup transcriber should receive an existing silent wav")
    if seen["path_after_check"].exists():
        raise SystemExit("warmup temp wav should be deleted after transcription")


def main() -> None:
    test_readiness_reports_missing_ffmpeg_without_audio_access()
    test_readiness_reports_missing_transcriber()
    test_readiness_rejects_invalid_configuration_without_path_leakage()
    test_readiness_loads_only_voice_settings_from_selected_v3_environment()
    test_readiness_bounds_selected_environment_failure()
    test_check_command_never_lists_or_opens_microphones()
    test_real_launch_holds_before_microphone_when_readiness_fails()
    test_parses_audio_device_list()
    test_max_record_seconds_bounds()
    test_default_mic_prefers_builtin_over_iphone()
    test_run_turn_routes_transcript_through_runtime()
    test_transcribe_passes_explicit_language_to_voice_layer()
    test_run_turn_does_not_auto_approve_side_effects()
    test_run_turn_surfaces_speech_failure()
    test_run_turn_does_not_report_successful_speech_as_failure()
    test_run_turn_uses_text_selected_korean_voice_for_localized_time()
    test_run_turn_does_not_force_korean_voice_for_english_only_reply()
    test_run_turn_surfaces_speech_exception_without_leaking_it()
    test_voice_warmup_is_disabled_by_default()
    test_voice_warmup_starts_once_in_background()
    test_voice_warmup_transcribes_silent_wav_without_mic()
    print("Talk smoke passed")


if __name__ == "__main__":
    main()
