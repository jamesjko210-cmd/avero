"""Focused offline proof for rejected voice text-input recovery guidance."""

from __future__ import annotations

from jarvis_v2.agent.failure_guidance import LOCAL_READ_INPUT_RECOVERY_ACTION
from jarvis_v2.tools.voice import (
    MAX_SPEECH_CHARS,
    MAX_TRANSCRIPT_CHARS,
    make_spoken_turn_rehearsal,
    make_voice_command_lifecycle,
    make_voice_confirmation_packet,
    make_voice_transcript_review,
    voice_reply_preview,
)


PRIVATE_MARKERS = (
    "/\x55sers/example/private/voice.txt",
    "/private/voice.txt",
    "/var/folders/voice.txt",
    "/tmp/voice.txt",
    "sk_" + "live_SUPERSECRET123",
    "traceback",
)


def _unused_tool_lookup(name: str):
    raise AssertionError(f"rejected voice input reached tool lookup: {name}")


def _assert_rejected_input(result, label: str, reason: str) -> None:
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
    }
    for key, value in expected.items():
        if result.metadata.get(key) is not value:
            raise SystemExit(f"{label} outcome field {key} drifted: {result.metadata}")
    if result.metadata.get("reason") != reason:
        raise SystemExit(f"{label} reason drifted: {result.metadata}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": LOCAL_READ_INPUT_RECOVERY_ACTION,
        "commands": [],
    }:
        raise SystemExit(f"{label} recovery declaration drifted: {result.metadata}")
    if LOCAL_READ_INPUT_RECOVERY_ACTION not in result.output:
        raise SystemExit(f"{label} hid the public recovery action: {result.output}")
    if result.metadata.get("recovery_commands") or result.metadata.get("next_command"):
        raise SystemExit(f"{label} accidentally authorized a recovery command: {result.metadata}")
    for flag in (
        "calls_model",
        "executes_tools",
        "authorizes_execution",
        "approval_granted",
        "reads_private_data",
        "reads_personal_data",
        "external_side_effect",
        "queues_approval",
        "speaks_audio",
    ):
        if result.metadata.get(flag):
            raise SystemExit(f"{label} unsafe boundary {flag}: {result.metadata}")
    public_proof = f"{result.output}\n{result.metadata.get('recovery_guidance')}".lower()
    for marker in PRIVATE_MARKERS:
        if marker.lower() in public_proof:
            raise SystemExit(f"{label} leaked private guidance detail: {public_proof}")


def main() -> None:
    transcript_review = make_voice_transcript_review(_unused_tool_lookup)
    confirmation_packet = make_voice_confirmation_packet(_unused_tool_lookup)
    command_lifecycle = make_voice_command_lifecycle(_unused_tool_lookup)
    spoken_rehearsal = make_spoken_turn_rehearsal(_unused_tool_lookup)
    cases = (
        (voice_reply_preview({}), "reply preview missing", "missing_preview_text"),
        (
            voice_reply_preview({"text": "x" * (MAX_SPEECH_CHARS + 1)}),
            "reply preview oversized",
            "oversized_preview_text",
        ),
        (spoken_rehearsal({}), "spoken rehearsal missing", "missing_rehearsal_message"),
        (transcript_review({}), "transcript review missing", "missing_transcript"),
        (
            transcript_review({"transcript": "x" * (MAX_TRANSCRIPT_CHARS + 1)}),
            "transcript review oversized",
            "oversized_transcript",
        ),
        (confirmation_packet({}), "confirmation packet missing", "missing_transcript"),
        (
            confirmation_packet({"transcript": "x" * (MAX_TRANSCRIPT_CHARS + 1)}),
            "confirmation packet oversized",
            "oversized_transcript",
        ),
        (
            command_lifecycle({"transcript": "x" * (MAX_TRANSCRIPT_CHARS + 1)}),
            "command lifecycle oversized",
            "oversized_transcript",
        ),
    )
    if len(cases) != 8:
        raise SystemExit(f"voice input guidance scope drifted: {len(cases)}/8")
    for result, label, reason in cases:
        _assert_rejected_input(result, label, reason)
    print("Voice input error-guidance smoke passed: 8 offline rejection branches.")


if __name__ == "__main__":
    main()
