"""Offline proof for canonical recovery across the remaining voice failure surfaces."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.failure_guidance import (
    LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
    LOCAL_READ_INPUT_RECOVERY_ACTION,
)
from jarvis_v2.tools import voice as voice_tools
from jarvis_v2.tools.voice import (
    MAX_TRANSCRIPT_CHARS,
    VOICE_FILE_READ_RECOVERY_ACTION,
    VOICE_SPEECH_OUTCOME_UNKNOWN_ACTION,
    VOICE_STOP_INPUT_RECOVERY_ACTION,
    VOICE_TRANSCRIPTION_RECOVERY_ACTION,
    list_voices,
    make_voice_action_audit_packet,
    make_voice_command_cockpit,
    make_voice_confirmation_audit_ledger,
    make_voice_confirmation_receipt,
    make_voice_cycle_ledger,
    make_voice_execution_handoff_packet,
    make_voice_post_run_closure_packet,
    make_voice_route_gate_packet,
    make_voice_route_proof_bundle,
    make_voice_runtime_bridge_packet,
    speak_text,
    voice_audio_file_transcription_preview,
    voice_stop_intent_packet,
)


PRIVATE_MARKERS = (
    "/\x55sers/",
    "/private/",
    "/var/folders/",
    "/tmp/",
    "sk_" + "live_",
    "ghp" + "_",
    "traceback",
)


def _unused_tool_lookup(name: str):
    raise AssertionError(f"rejected voice input reached tool lookup: {name}")


def _assert_public(result, label: str) -> None:
    public = f"{result.output}\n{result.metadata.get('recovery_guidance')}".lower()
    for marker in PRIVATE_MARKERS:
        if marker.lower() in public:
            raise SystemExit(f"{label} leaked private recovery detail: {public}")


def _assert_known_input(result, label: str, reason: str) -> None:
    if result.ok:
        raise SystemExit(f"{label} unexpectedly succeeded: {result}")
    expected = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "reason": reason,
    }
    for key, value in expected.items():
        if result.metadata.get(key) != value:
            raise SystemExit(f"{label} {key} truth mismatch: {result.metadata}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": LOCAL_READ_INPUT_RECOVERY_ACTION,
        "commands": [],
    }:
        raise SystemExit(f"{label} recovery declaration mismatch: {result.metadata}")
    if LOCAL_READ_INPUT_RECOVERY_ACTION not in result.output:
        raise SystemExit(f"{label} omitted its public recovery action: {result.output!r}")
    for flag in (
        "calls_model",
        "executes_tools",
        "authorizes_execution",
        "approval_granted",
        "reads_private_data",
        "reads_audio_file",
        "records_audio",
        "starts_listener",
        "transcribes_audio",
        "saves_transcript",
        "external_side_effect",
        "speaks_audio",
    ):
        if result.metadata.get(flag):
            raise SystemExit(f"{label} crossed voice boundary {flag}: {result.metadata}")
    _assert_public(result, label)


def _assert_declared(result, label: str, *, action: str, commands: list[str]) -> None:
    if result.ok:
        raise SystemExit(f"{label} unexpectedly succeeded: {result}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": action,
        "commands": commands,
    }:
        raise SystemExit(f"{label} recovery declaration mismatch: {result.metadata}")
    if action not in result.output:
        raise SystemExit(f"{label} omitted its recovery action: {result.output!r}")
    if result.metadata.get("automatic_retry_allowed") is not False:
        raise SystemExit(f"{label} allowed automatic retry: {result.metadata}")
    if result.metadata.get("authorizes_retry") is not False:
        raise SystemExit(f"{label} authorized replay: {result.metadata}")
    _assert_public(result, label)


def main() -> None:
    factories = (
        make_voice_confirmation_receipt,
        make_voice_route_gate_packet,
        make_voice_route_proof_bundle,
        make_voice_runtime_bridge_packet,
        make_voice_confirmation_audit_ledger,
        make_voice_command_cockpit,
        make_voice_action_audit_packet,
        make_voice_execution_handoff_packet,
        make_voice_post_run_closure_packet,
        make_voice_cycle_ledger,
    )
    packet_cases = []
    for factory in factories:
        handler = factory(_unused_tool_lookup)
        packet_cases.extend(
            (
                (handler({}), f"{factory.__name__} missing", "missing_transcript"),
                (
                    handler({"transcript": "x" * (MAX_TRANSCRIPT_CHARS + 1)}),
                    f"{factory.__name__} oversized",
                    "oversized_transcript",
                ),
            )
        )
    if len(packet_cases) != 20:
        raise SystemExit(f"voice packet input scope drifted: {len(packet_cases)}/20")
    for result, label, reason in packet_cases:
        _assert_known_input(result, label, reason)

    unclear_stop = voice_stop_intent_packet({"phrase": "/\x55sers/example/private/voice.txt"})
    _assert_declared(unclear_stop, "unclear stop intent", action=VOICE_STOP_INPUT_RECOVERY_ACTION, commands=[])
    if unclear_stop.metadata.get("outcome_known") is not True or unclear_stop.metadata.get("side_effect_possible") is not False:
        raise SystemExit(f"unclear stop intent truth mismatch: {unclear_stop.metadata}")
    if unclear_stop.metadata.get("records_audio") or unclear_stop.metadata.get("routes_actions"):
        raise SystemExit(f"unclear stop intent crossed read-only boundary: {unclear_stop.metadata}")

    original_run = voice_tools.subprocess.run
    original_transcriber_factory = voice_tools._audio_file_transcriber

    class FailedProcess:
        returncode = 1
        stdout = ""
        stderr = "private failure near /\x55sers/example/private/voice.txt"

    try:
        voice_tools.subprocess.run = lambda *_args, **_kwargs: FailedProcess()  # type: ignore[assignment]
        speech_failure = speak_text("Offline speech failure proof.", dry_run=False)
        voice_list_failure = list_voices({})
    finally:
        voice_tools.subprocess.run = original_run  # type: ignore[assignment]

    _assert_declared(
        speech_failure,
        "speech attempt",
        action=VOICE_SPEECH_OUTCOME_UNKNOWN_ACTION,
        commands=["voice setup check"],
    )
    if speech_failure.metadata.get("outcome_unknown") is not True or speech_failure.metadata.get("retry_safe") is not False:
        raise SystemExit(f"speech attempt overstated outcome truth: {speech_failure.metadata}")
    if speech_failure.metadata.get("side_effect_possible") is not True or speech_failure.metadata.get("executes_tools") is not True:
        raise SystemExit(f"speech attempt hid possible execution: {speech_failure.metadata}")

    _assert_declared(
        voice_list_failure,
        "voice-list read",
        action=LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
        commands=["setup check"],
    )
    if voice_list_failure.metadata.get("outcome_known") is not True or voice_list_failure.metadata.get("side_effect_possible") is not False:
        raise SystemExit(f"voice-list read truth mismatch: {voice_list_failure.metadata}")

    with TemporaryDirectory(prefix="jarvis-voice-guidance-full-") as temp:
        audio_path = Path(temp) / "offline.m4a"
        audio_path.write_bytes(b"metadata-only offline fixture")

        def failing_transcriber(_path: Path) -> str:
            raise RuntimeError("private failure near /\x55sers/example/private/voice.txt")

        try:
            voice_tools._audio_file_transcriber = lambda **_kwargs: failing_transcriber  # type: ignore[assignment]
            transcription_failure = voice_audio_file_transcription_preview(
                {"path": str(audio_path), "consent": "true", "receipt_id": "voice-file-offline"}
            )
        finally:
            voice_tools._audio_file_transcriber = original_transcriber_factory  # type: ignore[assignment]

    _assert_declared(
        transcription_failure,
        "approved transcription read",
        action=VOICE_TRANSCRIPTION_RECOVERY_ACTION,
        commands=["voice setup check"],
    )
    for key, expected in {
        "outcome_known": True,
        "side_effect_possible": False,
        "reads_private_data": True,
        "reads_audio_file": True,
        "transcribes_audio": True,
        "records_audio": False,
        "starts_listener": False,
        "saves_transcript": False,
    }.items():
        if transcription_failure.metadata.get(key) is not expected:
            raise SystemExit(f"approved transcription {key} truth mismatch: {transcription_failure.metadata}")

    held = voice_audio_file_transcription_preview({})
    _assert_declared(held, "pre-read transcription hold", action=VOICE_FILE_READ_RECOVERY_ACTION, commands=[])
    if held.metadata.get("reads_audio_file") or held.metadata.get("transcribes_audio"):
        raise SystemExit(f"pre-read transcription hold touched audio: {held.metadata}")

    print("Voice full failure-guidance smoke passed: 25 offline branches; no microphone, recording, playback, or production state.")


if __name__ == "__main__":
    main()
