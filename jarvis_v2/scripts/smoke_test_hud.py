"""Headless smoke tests for the Jarvis HUD core (no Tk window is created)."""

from __future__ import annotations

from types import SimpleNamespace

from jarvis_v2.scripts.hud import HudCore, HudReply


class _FakeRuntime:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def handle(self, text: str):
        self.commands.append(text)
        if text.startswith("approve approval"):
            return SimpleNamespace(response=f"Approved: {text.split()[-1]}", tool_results=[])
        if text == "risky send":
            return SimpleNamespace(
                response="Queued for approval.",
                tool_results=[SimpleNamespace(metadata={"requires_confirmation": True, "approval_id": 42})],
            )
        return SimpleNamespace(response=f"echo: {text}", tool_results=[])


def test_submit_routes_through_runtime() -> None:
    fake = _FakeRuntime()
    core = HudCore(runtime_factory=lambda: fake)
    reply = core.submit("what time is it")
    if reply.response != "echo: what time is it" or reply.error or reply.approval_ids:
        raise SystemExit(f"HUD submit should return the runtime response: {reply}")
    if fake.commands != ["what time is it"]:
        raise SystemExit(f"HUD should pass the command verbatim: {fake.commands}")


def test_empty_command_is_rejected_without_runtime() -> None:
    core = HudCore(runtime_factory=lambda: (_ for _ in ()).throw(AssertionError("must not init")))
    reply = core.submit("   ")
    if reply.error != "empty_command":
        raise SystemExit(f"empty input should not touch the runtime: {reply}")


def test_approvals_surface_with_ids() -> None:
    fake = _FakeRuntime()
    core = HudCore(runtime_factory=lambda: fake)
    reply = core.submit("risky send")
    if reply.approval_ids != [42]:
        raise SystemExit(f"HUD should surface pending approval ids: {reply}")
    approved = core.approve(42)
    if approved.response != "Approved: 42" or fake.commands[-1] != "approve approval 42":
        raise SystemExit(f"HUD approve should route through the runtime approval command: {approved} {fake.commands}")


def test_runtime_init_failure_is_friendly() -> None:
    core = HudCore(runtime_factory=lambda: (_ for _ in ()).throw(RuntimeError("db locked")))
    reply = core.submit("hello")
    if (
        reply.error != "runtime_initialization_failed"
        or "could not start" not in reply.response
        or "bootstrap_memory --check" not in reply.response
    ):
        raise SystemExit(f"runtime init failure should be friendly: {reply}")


def test_runtime_command_failure_is_private_safe_and_non_retrying() -> None:
    private_marker = "/\x55sers/example/private/hud-command.txt"

    class FailingRuntime:
        def handle(self, _text: str):
            raise RuntimeError(f"SHOULD NOT APPEAR {private_marker}")

    reply = HudCore(runtime_factory=FailingRuntime).submit("risky command")
    if reply.error != "runtime_command_outcome_unknown":
        raise SystemExit(f"runtime command failure code drifted: {reply}")
    for expected in ("could not confirm", "Do not retry automatically", "`execution health report`"):
        if expected not in reply.response:
            raise SystemExit(f"runtime command failure missed {expected!r}: {reply}")
    for forbidden in ("SHOULD NOT APPEAR", private_marker, "RuntimeError"):
        if forbidden in f"{reply.response}\n{reply.error}":
            raise SystemExit(f"runtime command failure leaked {forbidden!r}: {reply}")


def test_runtime_is_lazy_and_cached() -> None:
    calls = []
    fake = _FakeRuntime()

    def factory():
        calls.append(1)
        return fake

    core = HudCore(runtime_factory=factory)
    if calls:
        raise SystemExit("runtime must be lazy (not built at HudCore construction)")
    core.submit("a")
    core.submit("b")
    if len(calls) != 1:
        raise SystemExit(f"runtime must be built exactly once: {len(calls)}")


def test_hud_reply_shape() -> None:
    reply = HudReply(response="hi")
    if reply.approval_ids != [] or reply.error != "":
        raise SystemExit(f"HudReply defaults wrong: {reply}")


def test_recorder_start_builds_ffmpeg_mic_command() -> None:
    from jarvis_v2.scripts import hud, talk

    captured = {}

    class FakeProc:
        def communicate(self, input=None, timeout=None):
            captured["stop_input"] = input
            return ("", "")

    orig_popen = hud.subprocess.Popen
    orig_mic = talk._default_mic_index
    try:
        talk._default_mic_index = lambda: "3"
        hud.subprocess.Popen = lambda cmd, **kw: captured.update(cmd=cmd) or FakeProc()
        rec = hud.HudRecorder()
        if rec.recording:
            raise SystemExit("recorder must start idle")
        if not rec.start():
            raise SystemExit("recorder start should succeed with mocked Popen")
        if not rec.recording:
            raise SystemExit("recorder should report recording after start")
    finally:
        hud.subprocess.Popen = orig_popen
        talk._default_mic_index = orig_mic
    cmd = captured["cmd"]
    for part in ["ffmpeg", "avfoundation", ":3", "16000"]:
        if not any(part in str(c) for c in cmd):
            raise SystemExit(f"ffmpeg mic command missing {part!r}: {cmd}")


def test_recorder_stop_sends_q_transcribes_and_cleans_up() -> None:
    from jarvis_v2.scripts import hud, talk

    captured = {}

    class FakeProc:
        def communicate(self, input=None, timeout=None):
            captured["stop_input"] = input
            return ("", "")

    orig_popen = hud.subprocess.Popen
    orig_mic = talk._default_mic_index
    orig_transcribe = talk._transcribe
    try:
        talk._default_mic_index = lambda: "0"
        talk._transcribe = lambda path: captured.update(transcribed=str(path)) or "  hello jarvis  "
        hud.subprocess.Popen = lambda cmd, **kw: FakeProc()
        rec = hud.HudRecorder()
        rec.start()
        wav = rec._path
        wav.write_bytes(b"RIFFfakewav" + b"\x00" * 2000)  # >1000 bytes to pass size check
        text, error = rec.stop_and_transcribe()
    finally:
        hud.subprocess.Popen = orig_popen
        talk._default_mic_index = orig_mic
        talk._transcribe = orig_transcribe
    if text != "hello jarvis" or error:
        raise SystemExit(f"recorder should return (transcript, ''): ({text!r}, {error!r})")
    if captured.get("stop_input") != b"q":
        raise SystemExit(f"recorder must stop ffmpeg cleanly with 'q': {captured.get('stop_input')!r}")
    if rec.recording:
        raise SystemExit("recorder should be idle after stop")
    if wav.exists():
        raise SystemExit("recorder should delete the temp clip after transcription")


def test_recorder_stop_without_start_is_empty() -> None:
    from jarvis_v2.scripts import hud

    rec = hud.HudRecorder()
    text, error = rec.stop_and_transcribe()
    if text != "" or error == "":
        raise SystemExit(f"stop without start should return ('', error): ({text!r}, {error!r})")


def test_recorder_empty_clip_skips_transcription() -> None:
    from jarvis_v2.scripts import hud, talk

    class FakeProc:
        def communicate(self, input=None, timeout=None):
            return ("", "")

    orig_popen = hud.subprocess.Popen
    orig_mic = talk._default_mic_index
    orig_transcribe = talk._transcribe
    try:
        talk._default_mic_index = lambda: "0"
        talk._transcribe = lambda path: (_ for _ in ()).throw(AssertionError("must not transcribe empty clip"))
        hud.subprocess.Popen = lambda cmd, **kw: FakeProc()
        rec = hud.HudRecorder()
        rec.start()  # temp file exists but stays 0 bytes (silence/failure)
        text, error = rec.stop_and_transcribe()
    finally:
        hud.subprocess.Popen = orig_popen
        talk._default_mic_index = orig_mic
        talk._transcribe = orig_transcribe
    if text != "" or "no audio detected" not in error:
        raise SystemExit(f"empty clip must yield ('', error with 'no audio detected'): ({text!r}, {error!r})")


def main() -> None:
    test_submit_routes_through_runtime()
    test_empty_command_is_rejected_without_runtime()
    test_approvals_surface_with_ids()
    test_runtime_init_failure_is_friendly()
    test_runtime_command_failure_is_private_safe_and_non_retrying()
    test_runtime_is_lazy_and_cached()
    test_hud_reply_shape()
    test_recorder_start_builds_ffmpeg_mic_command()
    test_recorder_stop_sends_q_transcribes_and_cleans_up()
    test_recorder_stop_without_start_is_empty()
    test_recorder_empty_clip_skips_transcription()
    print("HUD smoke passed")


if __name__ == "__main__":
    main()
