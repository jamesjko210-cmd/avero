"""Offline proof for canonical voice configuration/read/operation failures."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.tools import voice as voice_tools
from jarvis_v2.tools.voice import (
    LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
    VOICE_FILE_READ_RECOVERY_ACTION,
    VOICE_SPEECH_OUTCOME_UNKNOWN_ACTION,
    VOICE_TRANSCRIPTION_RECOVERY_ACTION,
    list_voices,
    speak_text,
    voice_audio_file_gate_packet,
    voice_audio_file_transcription_preview,
    voice_file_transcription_plan,
)


PRIVATE_MARKERS = (
    "/\x55sers/",
    "/private/",
    "/var/folders/",
    "sk_" + "live_",
    "ghp" + "_",
)


def _assert_guidance(result, label: str, *, action: str, commands: list[str]) -> None:
    if result.ok:
        raise SystemExit(f"{label} unexpectedly succeeded: {result}")
    guidance = result.metadata.get("recovery_guidance")
    expected = {"version": 1, "action": action, "commands": commands}
    if guidance != expected:
        raise SystemExit(f"{label} recovery guidance mismatch: {guidance!r}")
    if action not in result.output:
        raise SystemExit(f"{label} output omitted its recovery action: {result.output!r}")
    if result.metadata.get("automatic_retry_allowed") is not False:
        raise SystemExit(f"{label} allowed an automatic retry: {result.metadata}")
    if result.metadata.get("authorizes_retry") is not False:
        raise SystemExit(f"{label} authorized a retry: {result.metadata}")
    public_proof = f"{result.output}\n{guidance}"
    for marker in PRIVATE_MARKERS:
        if marker.lower() in public_proof.lower():
            raise SystemExit(f"{label} leaked private guidance detail: {public_proof}")


def _assert_known_local_read(result, label: str, *, action: str) -> None:
    _assert_guidance(result, label, action=action, commands=[])
    expected = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
    }
    for key, value in expected.items():
        if result.metadata.get(key) is not value:
            raise SystemExit(f"{label} {key} truth mismatch: {result.metadata}")


def _assert_unknown_speech(result, label: str) -> None:
    _assert_guidance(
        result,
        label,
        action=VOICE_SPEECH_OUTCOME_UNKNOWN_ACTION,
        commands=["voice setup check"],
    )
    expected = {
        "outcome_known": False,
        "outcome_unknown": True,
        "execution_outcome_unknown": True,
        "side_effect_possible": True,
        "retry_safe": False,
        "speaks_audio": False,
        "executes_tools": True,
    }
    for key, value in expected.items():
        if result.metadata.get(key) is not value:
            raise SystemExit(f"{label} {key} truth mismatch: {result.metadata}")


def main() -> None:
    pre_read_cases = [
        (
            voice_file_transcription_plan({"path": "x" * 2001}),
            "file plan oversized path",
        ),
        (voice_audio_file_gate_packet({}), "file gate missing path"),
        (
            voice_audio_file_gate_packet({"path": "x" * 2001, "consent": "true"}),
            "file gate oversized path",
        ),
        (voice_audio_file_transcription_preview({}), "transcription preview missing path"),
        (
            voice_audio_file_transcription_preview(
                {"path": "x" * 2001, "consent": "true", "receipt_id": "voice-file-test"}
            ),
            "transcription preview oversized path",
        ),
    ]
    for result, label in pre_read_cases:
        _assert_known_local_read(result, label, action=VOICE_FILE_READ_RECOVERY_ACTION)
        if result.metadata.get("reads_audio_file") or result.metadata.get("transcribes_audio"):
            raise SystemExit(f"{label} overstated audio access: {result.metadata}")

    original_transcriber_factory = voice_tools._audio_file_transcriber
    with TemporaryDirectory(prefix="jarvis-voice-guidance-") as temp:
        audio_path = Path(temp) / "offline-sample.m4a"
        audio_path.write_bytes(b"offline smoke metadata only")
        try:
            voice_tools._audio_file_transcriber = lambda **_kwargs: None  # type: ignore[assignment]
            held = voice_audio_file_transcription_preview(
                {"path": str(audio_path), "consent": "true", "receipt_id": "voice-file-test"}
            )

            def failing_transcriber(_path: Path) -> str:
                raise RuntimeError("private failure near /\x55sers/example/recording.m4a")

            voice_tools._audio_file_transcriber = lambda **_kwargs: failing_transcriber  # type: ignore[assignment]
            failed_transcription = voice_audio_file_transcription_preview(
                {"path": str(audio_path), "consent": "true", "receipt_id": "voice-file-test"}
            )
        finally:
            voice_tools._audio_file_transcriber = original_transcriber_factory  # type: ignore[assignment]

    _assert_known_local_read(held, "held transcription preview", action=VOICE_FILE_READ_RECOVERY_ACTION)
    if held.metadata.get("reads_audio_file") or held.metadata.get("transcribes_audio"):
        raise SystemExit(f"held transcription preview touched audio: {held.metadata}")

    _assert_guidance(
        failed_transcription,
        "attempted transcription failure",
        action=VOICE_TRANSCRIPTION_RECOVERY_ACTION,
        commands=["voice setup check"],
    )
    transcription_truth = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "reads_private_data": True,
        "reads_personal_data": True,
        "reads_audio_file": True,
        "transcribes_audio": True,
        "saves_transcript": False,
    }
    for key, value in transcription_truth.items():
        if failed_transcription.metadata.get(key) is not value:
            raise SystemExit(f"attempted transcription {key} truth mismatch: {failed_transcription.metadata}")

    original_run = voice_tools.subprocess.run

    class FailedProcess:
        returncode = 1
        stdout = ""
        stderr = "private failure near /\x55sers/example/speech"

    try:
        voice_tools.subprocess.run = lambda *_args, **_kwargs: FailedProcess()  # type: ignore[assignment]
        speech_nonzero = speak_text("Offline speech failure proof.", dry_run=False)

        def raise_speech(*_args, **_kwargs):
            raise RuntimeError("private failure near /\x55sers/example/speech")

        voice_tools.subprocess.run = raise_speech  # type: ignore[assignment]
        speech_exception = speak_text("Offline speech exception proof.", dry_run=False)
        voices_exception = list_voices({})

        voice_tools.subprocess.run = lambda *_args, **_kwargs: FailedProcess()  # type: ignore[assignment]
        voices_nonzero = list_voices({})
    finally:
        voice_tools.subprocess.run = original_run  # type: ignore[assignment]

    _assert_unknown_speech(speech_nonzero, "speech nonzero")
    _assert_unknown_speech(speech_exception, "speech exception")
    for result, label in (
        (voices_exception, "voice list exception"),
        (voices_nonzero, "voice list nonzero"),
    ):
        _assert_guidance(
            result,
            label,
            action=LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
            commands=["setup check"],
        )
        if result.metadata.get("outcome_known") is not True or result.metadata.get("retry_safe") is not True:
            raise SystemExit(f"{label} read truth mismatch: {result.metadata}")
        if result.metadata.get("side_effect_possible") is not False:
            raise SystemExit(f"{label} overstated possible side effects: {result.metadata}")

    print("Voice operational failure-guidance smoke passed: 11 offline branches (9 converted, 2 retained).")


if __name__ == "__main__":
    main()
