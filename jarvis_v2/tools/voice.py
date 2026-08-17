from __future__ import annotations

import hashlib
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from jarvis_v2.agent.failure_guidance import (
    LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
    LOCAL_READ_INPUT_RECOVERY_ACTION,
    declare_outcome_unknown_failure,
    declare_retryable_local_read_failure,
    declare_retryable_personal_read_failure,
)
from jarvis_v2.agent.types import ApprovalArgumentResolution, ToolResult


MAX_SPEECH_CHARS = 5000
MAX_TRANSCRIPT_CHARS = 20000
MAX_VOICE_MODE_CHARS = 80
MAX_AUDIO_TRANSCRIPTION_BYTES = 50 * 1024 * 1024
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
COMMON_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm", ".mp4", ".mov"}
VOICE_WARMUP_TRUE_VALUES = {"1", "true", "yes", "on"}
SAFE_WHISPER_CLI_MODELS = {
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
    "turbo",
}
SUPPORTED_TRANSCRIPTION_LANGUAGES = {"en", "ko"}
KOREAN_TEXT_RE = re.compile(
    r"[\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\uac00-\ud7af\ud7b0-\ud7ff]"
)
MACOS_KOREAN_VOICE = "Yuna"
VOICE_FILE_READ_RECOVERY_ACTION = (
    "Correct the reported file, consent, receipt, or local transcription setup, "
    "then request a fresh approved transcription."
)
VOICE_TRANSCRIPTION_RECOVERY_ACTION = (
    "Run `voice setup check`, correct the reported audio-file or transcription issue, "
    "then request a fresh approved transcription."
)
VOICE_SPEECH_OUTCOME_UNKNOWN_ACTION = (
    "Could not confirm speech completion; the outcome is unknown. Check System Settings > Accessibility > "
    "Spoken Content and System Settings > Sound > Output, run `voice setup check`, then retry only after "
    "checking recent audio."
)
VOICE_STOP_INPUT_RECOVERY_ACTION = (
    "Say or type `stop listening`, `cancel`, `pause`, or `rerecord`, then request a fresh stop-intent review."
)
_AUDIO_FILE_TRANSCRIBER: Callable[[Path], str] | None = None
_LOCAL_WHISPER_MODEL: Any | None = None
_LOCAL_WHISPER_MODEL_CACHE_KEY = ""
_LOCAL_FASTER_WHISPER_MODEL: Any | None = None
_LOCAL_FASTER_WHISPER_MODEL_CACHE_KEY = ""


class VoiceSetupError(RuntimeError):
    """Deliberately human-readable, safe-to-show setup/config error (e.g. missing
    whisper CLI, misconfigured model-path env var). Distinct from a bare RuntimeError
    so `_voice_error` can tell "written for the user" apart from an arbitrary
    internal exception that happens to be a RuntimeError -- only the former is
    safe to surface verbatim."""


def _local_voice_model_path(env_name: str) -> Path | None:
    raw = str(os.environ.get(env_name) or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def _whisper_cli_path() -> str:
    """Path to the whisper CLI binary, if available (override via JARVIS_VOICE_WHISPER_CLI)."""
    override = str(os.environ.get("JARVIS_VOICE_WHISPER_CLI") or "").strip()
    if override:
        return override if Path(override).expanduser().is_file() else ""
    return shutil.which("whisper") or ""


def _voice_local_transcriber_source() -> str:
    whisper_path = _local_voice_model_path("JARVIS_VOICE_WHISPER_MODEL_PATH")
    if whisper_path and whisper_path.is_file() and importlib.util.find_spec("whisper") is not None:
        return "local_whisper_model_path"
    faster_path = _local_voice_model_path("JARVIS_VOICE_FASTER_WHISPER_MODEL_PATH")
    if faster_path and faster_path.exists() and importlib.util.find_spec("faster_whisper") is not None:
        return "local_faster_whisper_model_path"
    if _AUDIO_FILE_TRANSCRIBER is not None:
        return "in_process_transcriber"
    if _whisper_cli_path():
        return "local_whisper_cli"
    return "not_configured"


def _local_audio_transcriber_configured() -> bool:
    return _voice_local_transcriber_source() != "not_configured"


def _voice_warmup_enabled() -> bool:
    return str(os.environ.get("JARVIS_VOICE_WARMUP") or "").strip().lower() in VOICE_WARMUP_TRUE_VALUES


def _whisper_cli_model_label() -> str:
    """Expose only an allowlisted CLI model name for setup diagnostics."""
    model = str(os.environ.get("JARVIS_VOICE_WHISPER_CLI_MODEL") or "base").strip().lower() or "base"
    return model if model in SAFE_WHISPER_CLI_MODELS else "custom"


def _whisper_cli_language_label() -> str:
    """Expose a bounded language mode without echoing arbitrary environment text."""
    language = str(os.environ.get("JARVIS_VOICE_WHISPER_CLI_LANGUAGE") or "").strip().lower()
    if not language:
        return "automatic detection"
    if re.fullmatch(r"[a-z]{2,3}", language):
        return language
    return "custom"


def _normalized_transcription_language(language: str | None) -> str:
    value = str(language or "").strip().lower()
    return value if value in SUPPORTED_TRANSCRIPTION_LANGUAGES else ""


def _macos_voice_for_text(text: str, requested_voice: str = "") -> str:
    """Choose the macOS Korean voice when the caller did not choose a voice."""
    requested_voice = str(requested_voice or "").strip()
    if requested_voice:
        return requested_voice
    return MACOS_KOREAN_VOICE if KOREAN_TEXT_RE.search(text) else ""


def _transcribe_with_local_whisper(audio_path: Path, *, language: str | None = None) -> str:
    global _LOCAL_WHISPER_MODEL, _LOCAL_WHISPER_MODEL_CACHE_KEY
    model_path = _local_voice_model_path("JARVIS_VOICE_WHISPER_MODEL_PATH")
    if not model_path or not model_path.is_file():
        raise VoiceSetupError("JARVIS_VOICE_WHISPER_MODEL_PATH must point to a local Whisper model file.")
    cache_key = str(model_path)
    if _LOCAL_WHISPER_MODEL is None or _LOCAL_WHISPER_MODEL_CACHE_KEY != cache_key:
        import whisper  # type: ignore[import-not-found]

        _LOCAL_WHISPER_MODEL = whisper.load_model(cache_key)
        _LOCAL_WHISPER_MODEL_CACHE_KEY = cache_key
    options: dict[str, Any] = {"fp16": False}
    if selected_language := _normalized_transcription_language(language):
        options["language"] = selected_language
    result = _LOCAL_WHISPER_MODEL.transcribe(str(audio_path), **options)
    return str((result or {}).get("text") or "").strip()


def _transcribe_with_local_faster_whisper(audio_path: Path, *, language: str | None = None) -> str:
    global _LOCAL_FASTER_WHISPER_MODEL, _LOCAL_FASTER_WHISPER_MODEL_CACHE_KEY
    model_path = _local_voice_model_path("JARVIS_VOICE_FASTER_WHISPER_MODEL_PATH")
    if not model_path or not model_path.exists():
        raise VoiceSetupError("JARVIS_VOICE_FASTER_WHISPER_MODEL_PATH must point to a local faster-whisper model directory.")
    cache_key = str(model_path)
    if _LOCAL_FASTER_WHISPER_MODEL is None or _LOCAL_FASTER_WHISPER_MODEL_CACHE_KEY != cache_key:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]

        _LOCAL_FASTER_WHISPER_MODEL = WhisperModel(cache_key, device="cpu", compute_type="int8")
        _LOCAL_FASTER_WHISPER_MODEL_CACHE_KEY = cache_key
    options: dict[str, Any] = {}
    if selected_language := _normalized_transcription_language(language):
        options["language"] = selected_language
    segments, _info = _LOCAL_FASTER_WHISPER_MODEL.transcribe(str(audio_path), **options)
    return " ".join(str(segment.text).strip() for segment in segments).strip()


def _transcribe_with_whisper_cli(audio_path: Path, *, language: str | None = None) -> str:
    """Transcribe via the whisper CLI binary, which self-manages model download.

    Used as a fallback when no in-process transcriber or model-path env is set, so
    the voice flow works out of the box wherever the `whisper` CLI is installed.
    """
    cli = _whisper_cli_path()
    if not cli:
        raise VoiceSetupError("whisper CLI not found; set JARVIS_VOICE_WHISPER_CLI or install whisper.")
    model = str(os.environ.get("JARVIS_VOICE_WHISPER_CLI_MODEL") or "base").strip() or "base"
    with tempfile.TemporaryDirectory(prefix="jarvis-whisper-") as out_dir:
        cmd = [
            cli, str(audio_path),
            "--model", model,
            "--output_format", "txt",
            "--output_dir", out_dir,
            "--task", "transcribe",
            "--fp16", "False",
            "--verbose", "False",
        ]
        lang = _normalized_transcription_language(language) or str(
            os.environ.get("JARVIS_VOICE_WHISPER_CLI_LANGUAGE") or ""
        ).strip()
        if lang:
            cmd += ["--language", lang]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise VoiceSetupError(result.stderr.strip() or "whisper CLI failed")
        txt_path = Path(out_dir) / (audio_path.stem + ".txt")
        if txt_path.is_file():
            return txt_path.read_text(encoding="utf-8", errors="replace").strip()
        # Fall back to stdout if the txt file wasn't written.
        return (result.stdout or "").strip()


def _audio_file_transcriber(*, language: str | None = None) -> Callable[[Path], str] | None:
    if _AUDIO_FILE_TRANSCRIBER is not None:
        return _AUDIO_FILE_TRANSCRIBER
    source = _voice_local_transcriber_source()
    selected_language = _normalized_transcription_language(language)
    if source == "local_whisper_model_path":
        if selected_language:
            return lambda path: _transcribe_with_local_whisper(path, language=selected_language)
        return _transcribe_with_local_whisper
    if source == "local_faster_whisper_model_path":
        if selected_language:
            return lambda path: _transcribe_with_local_faster_whisper(path, language=selected_language)
        return _transcribe_with_local_faster_whisper
    if source == "local_whisper_cli":
        if selected_language:
            return lambda path: _transcribe_with_whisper_cli(path, language=selected_language)
        return _transcribe_with_whisper_cli
    return None


def _bounded_int(value: Any, default: int, low: int, high: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _voice_metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _voice_metadata_all_bool(*values: Any) -> bool:
    return all(_voice_metadata_bool(value) for value in values)


def _short_text(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _has_local_path(value: Any) -> bool:
    return bool(LOCAL_PATH_RE.search(str(value or "")))


def _voice_metadata(**extra: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "calls_model": False,
        "calls_chat_model": False,
        "executes_tools": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "reads_audio_file": False,
        "records_audio": False,
        "starts_listener": False,
        "transcribes_audio": False,
        "saves_transcript": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "external_side_effect": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": False,
        "speaks": False,
        "speaks_audio": False,
        "completes_tasks": False,
    }
    metadata.update(extra)
    return metadata


def _voice_readonly_boundaries(**extra: bool) -> dict[str, bool]:
    boundaries = {
        "read_only": True,
        "calls_model": False,
        "calls_chat_model": False,
        "executes_tools": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "reads_audio_file": False,
        "records_audio": False,
        "starts_listener": False,
        "transcribes_audio": False,
        "saves_transcript": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "external_side_effect": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": False,
        "speaks": False,
        "speaks_audio": False,
        "completes_tasks": False,
    }
    boundaries.update(extra)
    return boundaries


def _voice_transcription_boundaries(**extra: bool) -> dict[str, bool]:
    boundaries = _voice_readonly_boundaries(
        read_only=False,
        reads_private_data=True,
        reads_personal_data=True,
        reads_audio_file=True,
        transcribes_audio=True,
        requires_approval=True,
    )
    boundaries.update(extra)
    return boundaries


def _voice_error(action: str, exc: Exception | None = None) -> str:
    # VoiceSetupError marks a deliberately human-readable, already-safe-to-show
    # message (missing CLI, misconfigured model path env var, etc.) -- surface
    # it instead of a generic permissions message that can point the user at the
    # wrong fix (e.g. "check permissions" when the real problem is a missing
    # whisper CLI install). Any OTHER exception (including a bare RuntimeError
    # from somewhere unexpected) keeps the generic message -- it was not written
    # with user-display in mind and may not be safe to show verbatim.
    if isinstance(exc, VoiceSetupError) and str(exc).strip():
        return f"Could not {action}: {_short_text(exc, 200)}"
    detail = f" ({type(exc).__name__})" if exc is not None else ""
    if "transcribe" in action.lower():
        recovery = "Check audio-file permissions and ASR dependencies, run `voice setup check`, then retry"
    else:
        recovery = (
            "Check System Settings > Accessibility > Spoken Content and System Settings > Sound > Output, "
            "run `voice setup check`, then retry"
        )
    return f"Could not {action}. {recovery}.{detail}"


def _voice_input_failure(tool_name: str, message: str, **metadata: Any) -> ToolResult:
    """Return one private-safe, side-effect-free voice input rejection."""

    failure_output = f"{message} {LOCAL_READ_INPUT_RECOVERY_ACTION}"
    return ToolResult(
        tool_name,
        False,
        failure_output,
        declare_retryable_local_read_failure(
            _voice_metadata(**metadata),
            output=failure_output,
            action=LOCAL_READ_INPUT_RECOVERY_ACTION,
        ),
    )


def _voice_file_read_failure(tool_name: str, message: str, **metadata: Any) -> ToolResult:
    """Return one private-safe failure before any audio content was read."""

    failure_output = f"{message} {VOICE_FILE_READ_RECOVERY_ACTION}"
    return ToolResult(
        tool_name,
        False,
        failure_output,
        declare_retryable_local_read_failure(
            _voice_metadata(**metadata),
            output=failure_output,
            action=VOICE_FILE_READ_RECOVERY_ACTION,
        ),
    )


def _voice_speech_outcome_unknown(**metadata: Any) -> ToolResult:
    """Return an honest result after speech execution may have started."""

    return ToolResult(
        "speak",
        False,
        VOICE_SPEECH_OUTCOME_UNKNOWN_ACTION,
        declare_outcome_unknown_failure(
            _voice_metadata(**metadata),
            output=VOICE_SPEECH_OUTCOME_UNKNOWN_ACTION,
            commands=("voice setup check",),
        ),
    )


def _voice_next_commands(transcript: str, approval_required: bool) -> list[str]:
    if approval_required:
        return [
            f"send confirmed transcript: {transcript}",
            "pending approvals",
            "approval readiness latest",
            "approval packet latest",
            "approval chain proof latest",
            "approve approval latest",
        ]
    return [f"send confirmed transcript: {transcript}"]


def _voice_receipt_next_commands(transcript: str, approval_required: bool, confirmed: bool) -> list[str]:
    if confirmed:
        return _voice_next_commands(transcript, approval_required)
    return [
        f"voice transcript review: {transcript}",
        f"voice confirmation: {transcript}",
        f"voice confirmation receipt: {transcript} confirmed=true",
        "rerecord voice command",
    ]


def _transcript_hash(transcript: str) -> str:
    return hashlib.sha256(transcript.encode("utf-8")).hexdigest()[:16]


def _transcript_sha256(transcript: str) -> str:
    return hashlib.sha256(str(transcript or "").encode("utf-8")).hexdigest()


def _voice_text_sha256(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _voice_confirmation_receipt_command(transcript: str) -> str:
    return f"voice confirmation receipt: {transcript} confirmed=true" if transcript else "voice confirmation receipt: <transcript> confirmed=true"


def _voice_receipt_nonce(receipt_id: str, transcript_hash: str) -> str:
    return hashlib.sha256(f"{receipt_id}|{transcript_hash}|voice-confirmation-receipt".encode("utf-8")).hexdigest()[:16]


def _voice_command_intake_contract_sha256(
    *,
    transcript: str,
    transcript_sha256: str,
    privacy_receipt_id: str,
    confirmation_receipt_id: str,
    confirmation_receipt_nonce: str,
    route_gate_state: str,
    route_proof_bundle_state: str,
    runtime_bridge_state: str,
    proof_commands: list[str],
    authorizes_action_now: bool = False,
    authorizes_model_call: bool = False,
    authorizes_tool_execution: bool = False,
    authorizes_approval: bool = False,
    authorizes_routing: bool = False,
    authorizes_transcript_mutation: bool = False,
    authorizes_personal_data_read: bool = False,
    authorizes_external_side_effect: bool = False,
    reusable_for_next_voice_review: bool = False,
) -> str:
    return _voice_text_sha256(
        "|".join(
            [
                "voice_command_intake_contract_v1",
                transcript,
                transcript_sha256,
                privacy_receipt_id,
                confirmation_receipt_id,
                confirmation_receipt_nonce,
                route_gate_state,
                route_proof_bundle_state,
                runtime_bridge_state,
                repr(list(proof_commands)),
                f"authorizes_action_now={authorizes_action_now}",
                f"authorizes_model_call={authorizes_model_call}",
                f"authorizes_tool_execution={authorizes_tool_execution}",
                f"authorizes_approval={authorizes_approval}",
                f"authorizes_routing={authorizes_routing}",
                f"authorizes_transcript_mutation={authorizes_transcript_mutation}",
                f"authorizes_personal_data_read={authorizes_personal_data_read}",
                f"authorizes_external_side_effect={authorizes_external_side_effect}",
                f"reusable_for_next_voice_review={reusable_for_next_voice_review}",
            ]
        )
    )


def _voice_evidence_present(args: dict[str, Any], *keys: str) -> bool:
    evidence = " ".join(str(args.get(key) or "") for key in ("evidence", "proof", "review", "verification", "audit", "health", "learning")).lower()
    return any(str(args.get(key) or "").strip().lower() in {"true", "yes", "1", "reviewed", "verified", "passed", "complete", "ok"} for key in keys) or all(
        key.replace("_", " ") in evidence for key in keys
    )


def _voice_command_intake_contract_metadata(
    *,
    transcript: str,
    privacy_receipt_id: str,
    confirmation_receipt_id: str,
    confirmation_receipt_nonce: str,
    route_gate_state: str,
    route_proof_bundle_state: str,
    runtime_bridge_state: str,
    proof_commands: list[str],
) -> dict[str, Any]:
    transcript_sha256 = _transcript_sha256(transcript)
    contract_sha256 = _voice_command_intake_contract_sha256(
        transcript=transcript,
        transcript_sha256=transcript_sha256,
        privacy_receipt_id=privacy_receipt_id,
        confirmation_receipt_id=confirmation_receipt_id,
        confirmation_receipt_nonce=confirmation_receipt_nonce,
        route_gate_state=route_gate_state,
        route_proof_bundle_state=route_proof_bundle_state,
        runtime_bridge_state=runtime_bridge_state,
        proof_commands=proof_commands,
    )
    return {
        "transcript_sha256": transcript_sha256,
        "voice_command_intake_contract_sha256": contract_sha256,
        "voice_command_intake_contract_privacy_receipt_id": privacy_receipt_id,
        "voice_command_intake_contract_confirmation_receipt_id": confirmation_receipt_id,
        "voice_command_intake_contract_confirmation_receipt_nonce": confirmation_receipt_nonce,
        "voice_command_intake_contract_route_gate_state": route_gate_state,
        "voice_command_intake_contract_route_proof_bundle_state": route_proof_bundle_state,
        "voice_command_intake_contract_runtime_bridge_state": runtime_bridge_state,
        "voice_command_intake_contract_proof_commands": list(proof_commands),
        "voice_command_intake_contract_present": _looks_like_sha256(contract_sha256),
        "voice_command_intake_contract_ready": _looks_like_sha256(contract_sha256),
        "voice_command_intake_contract_authorizes_action_now": False,
        "voice_command_intake_contract_authorizes_model_call": False,
        "voice_command_intake_contract_authorizes_tool_execution": False,
        "voice_command_intake_contract_authorizes_approval": False,
        "voice_command_intake_contract_authorizes_routing": False,
        "voice_command_intake_contract_authorizes_transcript_mutation": False,
        "voice_command_intake_contract_authorizes_personal_data_read": False,
        "voice_command_intake_contract_authorizes_external_side_effect": False,
        "voice_command_intake_contract_reusable_for_next_voice_review": False,
        "next_voice_review_requires_fresh_command_intake_contract": True,
    }


_VOICE_COMMAND_INTAKE_CONTRACT_FIELDS = [
    "transcript_sha256",
    "voice_command_intake_contract_sha256",
    "voice_command_intake_contract_privacy_receipt_id",
    "voice_command_intake_contract_confirmation_receipt_id",
    "voice_command_intake_contract_confirmation_receipt_nonce",
    "voice_command_intake_contract_route_gate_state",
    "voice_command_intake_contract_route_proof_bundle_state",
    "voice_command_intake_contract_runtime_bridge_state",
    "voice_command_intake_contract_proof_commands",
    "voice_command_intake_contract_present",
    "voice_command_intake_contract_ready",
    "voice_command_intake_contract_authorizes_action_now",
    "voice_command_intake_contract_authorizes_model_call",
    "voice_command_intake_contract_authorizes_tool_execution",
    "voice_command_intake_contract_authorizes_approval",
    "voice_command_intake_contract_authorizes_routing",
    "voice_command_intake_contract_authorizes_transcript_mutation",
    "voice_command_intake_contract_authorizes_personal_data_read",
    "voice_command_intake_contract_authorizes_external_side_effect",
    "voice_command_intake_contract_reusable_for_next_voice_review",
    "next_voice_review_requires_fresh_command_intake_contract",
]


def _voice_command_intake_contract_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: metadata.get(key) for key in _VOICE_COMMAND_INTAKE_CONTRACT_FIELDS if key in metadata}


def _looks_like_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(value or "").strip()))


def _voice_native_microphone_gate_ready(metadata: dict[str, Any]) -> bool:
    mode = str(metadata.get("mode") or "")
    permission_receipt_id = str(metadata.get("permission_receipt_id") or "")
    gate_state = str(metadata.get("gate_state") or "")
    missing_checks = metadata.get("missing_checks")
    required_before_capture = metadata.get("required_before_capture")
    required_after_transcript = metadata.get("required_after_transcript")
    capture_gate_receipt_id = str(metadata.get("capture_gate_receipt_id") or "")
    if mode not in {"native-push-to-talk", "browser-push-to-talk"}:
        return False
    if not re.fullmatch(r"voice-native-[a-f0-9]{12}", capture_gate_receipt_id):
        return False
    if not isinstance(missing_checks, list) or metadata.get("missing_check_count") != len(missing_checks):
        return False
    if not isinstance(required_before_capture, list) or not isinstance(required_after_transcript, list):
        return False
    if len(required_before_capture) != 3 or len(required_after_transcript) != 5:
        return False
    if "voice setup check" not in required_before_capture:
        return False
    if "voice capture privacy: native-push-to-talk" not in required_before_capture:
        return False
    if not any(str(command).startswith("voice native microphone gate:") for command in required_before_capture):
        return False
    for command in (
        "voice transcript review: <transcript>",
        "voice confirmation: <transcript>",
        "voice confirmation receipt: <transcript> confirmed=true",
        "voice route proof bundle: <transcript> confirmed=true",
        "voice command cockpit: <transcript> confirmed=true",
    ):
        if command not in required_after_transcript:
            return False
    if gate_state == "NATIVE_MIC_HELD_FOR_PERMISSION_RECEIPT":
        if not missing_checks:
            return False
        if permission_receipt_id and "visible microphone permission receipt" in missing_checks:
            return False
        if metadata.get("next_command") != "voice native microphone gate: mode=native-push-to-talk permission_receipt_id=<receipt> visible_state=reviewed":
            return False
    elif gate_state == "NATIVE_MIC_READY_FOR_APPROVED_CAPTURE_RUNNER":
        if missing_checks or not permission_receipt_id:
            return False
        if metadata.get("next_command") != "voice transcript review: <transcript>":
            return False
    else:
        return False
    expected_flags = {
        "native_microphone_capture_enabled": False,
        "native_capture_runner_separate": True,
        "capture_runner_requires_approval": True,
        "raw_audio_transient_required": True,
        "transcript_preview_required": True,
        "transcript_confirmation_required": True,
        "confirmation_receipt_required": True,
        "receipt_nonce_required": True,
        "route_proof_bundle_required": True,
        "command_cockpit_required": True,
        "post_run_closure_required": True,
        "records_audio": False,
        "starts_listener": False,
        "transcribes_audio": False,
        "reads_audio_file": False,
        "executes_tools": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "queues_approval": False,
        "external_side_effect": False,
        "controls_computer": False,
    }
    if metadata.get("capture_runner_risk_level") != "PERSONAL_DATA":
        return False
    if any(metadata.get(key) is not expected for key, expected in expected_flags.items()):
        return False
    return True


def _voice_sha256_arg(args: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(args.get(key) or "").strip()
        if value:
            return value
    return ""


def _voice_post_run_closure_token_sha256(
    *,
    transcript_hash: str,
    closure_state: str,
    receipt_nonce: str,
    handoff_state: str,
    handoff_ready: bool,
    command_intake_contract_sha256: str,
    verification_receipt_sha256: str,
    execution_health_sha256: str,
    execution_audit_sha256: str,
    after_action_learning_sha256: str,
    next_review_start_command: str,
    authorizes_action_now: bool = False,
    authorizes_model_call: bool = False,
    authorizes_tool_execution: bool = False,
    authorizes_approval: bool = False,
    authorizes_routing: bool = False,
    authorizes_transcript_mutation: bool = False,
    authorizes_personal_data_read: bool = False,
    authorizes_external_side_effect: bool = False,
    authorizes_next_voice_review: bool = False,
    authorizes_receipt_reuse: bool = False,
    reusable_for_next_voice_review: bool = False,
) -> str:
    return _voice_text_sha256(
        "|".join(
            [
                transcript_hash,
                closure_state,
                receipt_nonce,
                handoff_state,
                f"handoff_ready={handoff_ready}",
                command_intake_contract_sha256,
                verification_receipt_sha256,
                execution_health_sha256,
                execution_audit_sha256,
                after_action_learning_sha256,
                next_review_start_command,
                f"authorizes_action_now={authorizes_action_now}",
                f"authorizes_model_call={authorizes_model_call}",
                f"authorizes_tool_execution={authorizes_tool_execution}",
                f"authorizes_approval={authorizes_approval}",
                f"authorizes_routing={authorizes_routing}",
                f"authorizes_transcript_mutation={authorizes_transcript_mutation}",
                f"authorizes_personal_data_read={authorizes_personal_data_read}",
                f"authorizes_external_side_effect={authorizes_external_side_effect}",
                f"authorizes_next_voice_review={authorizes_next_voice_review}",
                f"authorizes_receipt_reuse={authorizes_receipt_reuse}",
                f"reusable_for_next_voice_review={reusable_for_next_voice_review}",
                "voice_post_run_closure_proof_only",
            ]
        )
    )


def _voice_post_run_closure_token_boundary_rows(*, token_sha256: str, source: str) -> list[dict[str, Any]]:
    rows = [
        {
            "item": "voice_post_run_closure_token",
            "source": source,
            "status": "present" if _looks_like_sha256(token_sha256) else "missing",
            "token_sha256": token_sha256,
        },
        {
            "item": "prior_spoken_command_scope",
            "source": source,
            "status": "proof_only_for_completed_voice_turn",
            "token_sha256": token_sha256,
        },
        {
            "item": "next_voice_review_boundary",
            "source": source,
            "status": "fresh_confirmation_receipt_required",
            "token_sha256": token_sha256,
        },
    ]
    for row in rows:
        row.update(
            {
                "authorizes_action_now": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_approval": False,
                "authorizes_routing": False,
                "authorizes_transcript_mutation": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "authorizes_next_voice_review": False,
                "authorizes_receipt_reuse": False,
                "reusable_for_next_voice_review": False,
            }
        )
    return rows


def _voice_confirmation_audit_token_sha256(
    *,
    transcript_hash: str,
    audit_ledger_state: str,
    privacy_receipt_id: str,
    confirmation_receipt_id: str,
    confirmation_receipt_nonce: str,
    supplied_privacy_receipt_id: str,
    supplied_confirmation_receipt_id: str,
    supplied_confirmation_receipt_nonce: str,
    route_gate_state: str,
    route_proof_bundle_state: str,
    runtime_bridge_state: str,
    ready_for_command_intake_proof: bool,
    stage_count: int,
    required_command_count: int,
    authorizes_action_now: bool = False,
    authorizes_model_call: bool = False,
    authorizes_tool_execution: bool = False,
    authorizes_approval: bool = False,
    authorizes_routing: bool = False,
    authorizes_transcript_mutation: bool = False,
    authorizes_personal_data_read: bool = False,
    authorizes_external_side_effect: bool = False,
    authorizes_receipt_reuse: bool = False,
    reusable_for_next_voice_review: bool = False,
) -> str:
    return _voice_text_sha256(
        "|".join(
            [
                transcript_hash,
                audit_ledger_state,
                privacy_receipt_id,
                confirmation_receipt_id,
                confirmation_receipt_nonce,
                supplied_privacy_receipt_id,
                supplied_confirmation_receipt_id,
                supplied_confirmation_receipt_nonce,
                route_gate_state,
                route_proof_bundle_state,
                runtime_bridge_state,
                f"ready_for_command_intake_proof={ready_for_command_intake_proof}",
                f"stage_count={stage_count}",
                f"required_command_count={required_command_count}",
                f"authorizes_action_now={authorizes_action_now}",
                f"authorizes_model_call={authorizes_model_call}",
                f"authorizes_tool_execution={authorizes_tool_execution}",
                f"authorizes_approval={authorizes_approval}",
                f"authorizes_routing={authorizes_routing}",
                f"authorizes_transcript_mutation={authorizes_transcript_mutation}",
                f"authorizes_personal_data_read={authorizes_personal_data_read}",
                f"authorizes_external_side_effect={authorizes_external_side_effect}",
                f"authorizes_receipt_reuse={authorizes_receipt_reuse}",
                f"reusable_for_next_voice_review={reusable_for_next_voice_review}",
                "voice_confirmation_audit_proof_only",
            ]
        )
    )


def _voice_confirmation_audit_token_boundary_rows(*, token_sha256: str, source: str) -> list[dict[str, Any]]:
    rows = [
        {
            "item": "voice_confirmation_audit_token",
            "source": source,
            "status": "present" if _looks_like_sha256(token_sha256) else "missing",
            "token_sha256": token_sha256,
        },
        {
            "item": "confirmed_transcript_scope",
            "source": source,
            "status": "proof_only_for_exact_transcript",
            "token_sha256": token_sha256,
        },
        {
            "item": "receipt_reuse_boundary",
            "source": source,
            "status": "fresh_confirmation_receipt_required",
            "token_sha256": token_sha256,
        },
    ]
    for row in rows:
        row.update(
            {
                "authorizes_action_now": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_approval": False,
                "authorizes_routing": False,
                "authorizes_transcript_mutation": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "authorizes_receipt_reuse": False,
                "reusable_for_next_voice_review": False,
            }
        )
    return rows


def _voice_confirmation_audit_token_boundary_ready(
    rows: list[dict[str, Any]],
    *,
    token_sha256: str,
    source: str = "voice_confirmation_audit_ledger",
) -> bool:
    expected_status = {
        "voice_confirmation_audit_token": "present",
        "confirmed_transcript_scope": "proof_only_for_exact_transcript",
        "receipt_reuse_boundary": "fresh_confirmation_receipt_required",
    }
    if not _looks_like_sha256(token_sha256):
        return False
    if len(rows) != len(expected_status):
        return False
    if {str(row.get("item") or "") for row in rows} != set(expected_status):
        return False
    for row in rows:
        item = str(row.get("item") or "")
        if (
            row.get("source") != source
            or row.get("status") != expected_status.get(item)
            or row.get("token_sha256") != token_sha256
        ):
            return False
        for key in (
            "authorizes_action_now",
            "authorizes_model_call",
            "authorizes_tool_execution",
            "authorizes_approval",
            "authorizes_routing",
            "authorizes_transcript_mutation",
            "authorizes_personal_data_read",
            "authorizes_external_side_effect",
            "authorizes_receipt_reuse",
            "reusable_for_next_voice_review",
        ):
            if row.get(key) is not False:
                return False
    return True


def _voice_post_run_closure_token_boundary_ready(
    rows: list[dict[str, Any]],
    *,
    token_sha256: str,
    source: str = "voice_post_run_closure",
) -> bool:
    expected_status = {
        "voice_post_run_closure_token": "present",
        "prior_spoken_command_scope": "proof_only_for_completed_voice_turn",
        "next_voice_review_boundary": "fresh_confirmation_receipt_required",
    }
    if not _looks_like_sha256(token_sha256):
        return False
    if len(rows) != len(expected_status):
        return False
    if {str(row.get("item") or "") for row in rows} != set(expected_status):
        return False
    for row in rows:
        item = str(row.get("item") or "")
        if (
            row.get("source") != source
            or row.get("status") != expected_status.get(item)
            or row.get("token_sha256") != token_sha256
        ):
            return False
        for key in (
            "authorizes_action_now",
            "authorizes_model_call",
            "authorizes_tool_execution",
            "authorizes_approval",
            "authorizes_routing",
            "authorizes_transcript_mutation",
            "authorizes_personal_data_read",
            "authorizes_external_side_effect",
            "authorizes_next_voice_review",
            "authorizes_receipt_reuse",
            "reusable_for_next_voice_review",
        ):
            if row.get(key) is not False:
                return False
    return True


def _voice_cycle_preflight_scorecard_rows(
    *,
    privacy_match: bool,
    receipt_freshness_match: bool,
    route_bundle_ready: bool,
    runtime_bridge_ready: bool,
    action_audit_ready: bool,
    handoff_ready: bool,
    post_run_artifact_hashes_present: bool,
    cycle_ready: bool,
) -> list[dict[str, Any]]:
    checks = [
        ("visible_privacy_receipt", 15, privacy_match),
        ("fresh_confirmation_receipt", 15, receipt_freshness_match),
        ("route_proof_bundle", 15, route_bundle_ready),
        ("command_intake_bridge", 15, runtime_bridge_ready),
        ("action_audit_boundary", 10, action_audit_ready),
        ("execution_handoff_packet", 10, handoff_ready),
        ("post_run_artifact_hashes", 10, post_run_artifact_hashes_present),
        ("fresh_review_cycle_boundary", 10, cycle_ready),
    ]
    return [
        {
            "item": item,
            "points": max_points if ready else 0,
            "max_points": max_points,
            "ready": ready,
            "required_before_next_voice_review": True,
            "authorizes_action": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_approval": False,
            "authorizes_routing": False,
            "authorizes_transcript_mutation": False,
            "authorizes_personal_data_read": False,
            "authorizes_external_side_effect": False,
            "authorizes_next_voice_review": False,
            "authorizes_receipt_reuse": False,
            "reusable_for_next_voice_review": False,
        }
        for item, max_points, ready in checks
    ]


def _voice_cycle_preflight_scorecard_ready(rows: list[dict[str, Any]]) -> bool:
    if len(rows) != len(_VOICE_CYCLE_PREFLIGHT_ITEMS):
        return False
    if {str(row.get("item") or "") for row in rows} != _VOICE_CYCLE_PREFLIGHT_ITEMS:
        return False
    total_max = 0
    non_authority_fields = (
        "authorizes_action",
        "authorizes_model_call",
        "authorizes_tool_execution",
        "authorizes_approval",
        "authorizes_routing",
        "authorizes_transcript_mutation",
        "authorizes_personal_data_read",
        "authorizes_external_side_effect",
        "authorizes_next_voice_review",
        "authorizes_receipt_reuse",
        "reusable_for_next_voice_review",
    )
    for row in rows:
        if row.get("ready") is not True:
            return False
        if row.get("required_before_next_voice_review") is not True:
            return False
        try:
            points = int(row.get("points"))
            max_points = int(row.get("max_points"))
        except (TypeError, ValueError):
            return False
        if max_points <= 0 or points != max_points:
            return False
        if any(row.get(field) is not False for field in non_authority_fields):
            return False
        total_max += max_points
    return total_max == 100


def _voice_cycle_ledger_token_sha256(
    *,
    transcript: str,
    transcript_hash: str,
    cycle_state: str,
    stage_rows: list[dict[str, Any]],
    voice_preflight_scorecard_rows: list[dict[str, Any]],
    fresh_review_preflight_queue: list[str],
    fresh_review_contract_rows: list[dict[str, Any]],
    required_commands: list[str],
    command_intake_contract_sha256: str,
    voice_post_run_closure_token_sha256: str,
    verification_receipt_sha256: str,
    execution_health_sha256: str,
    execution_audit_sha256: str,
    after_action_learning_sha256: str,
    next_command: str,
    authorizes_action_now: bool = False,
    authorizes_model_call: bool = False,
    authorizes_tool_execution: bool = False,
    authorizes_approval: bool = False,
    authorizes_routing: bool = False,
    authorizes_transcript_mutation: bool = False,
    authorizes_personal_data_read: bool = False,
    authorizes_external_side_effect: bool = False,
    authorizes_next_voice_review: bool = False,
    reusable_for_next_voice_review: bool = False,
) -> str:
    def row_entries(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[str]:
        return ["|".join(str(row.get(key) or "") for key in keys) for row in rows]

    return _voice_text_sha256(
        "\n".join(
            [
                "voice_cycle_ledger_v1",
                transcript,
                transcript_hash,
                cycle_state,
                "stage_rows",
                *row_entries(
                    stage_rows,
                    (
                        "stage",
                        "state",
                        "ready",
                        "proof",
                        "authorizes_action",
                        "authorizes_model_call",
                        "authorizes_tool_execution",
                        "authorizes_approval",
                        "authorizes_routing",
                        "authorizes_transcript_mutation",
                        "authorizes_personal_data_read",
                        "authorizes_external_side_effect",
                        "authorizes_next_voice_review",
                        "authorizes_receipt_reuse",
                        "reusable_for_next_voice_review",
                    ),
                ),
                "preflight_scorecard_rows",
                *row_entries(
                    voice_preflight_scorecard_rows,
                    (
                        "item",
                        "points",
                        "max_points",
                        "ready",
                        "required_before_next_voice_review",
                        "authorizes_action",
                        "authorizes_model_call",
                        "authorizes_tool_execution",
                        "authorizes_approval",
                        "authorizes_routing",
                        "authorizes_transcript_mutation",
                        "authorizes_personal_data_read",
                        "authorizes_external_side_effect",
                        "authorizes_next_voice_review",
                        "authorizes_receipt_reuse",
                        "reusable_for_next_voice_review",
                    ),
                ),
                "fresh_review_preflight_queue",
                *fresh_review_preflight_queue,
                "fresh_review_contract_rows",
                *row_entries(
                    fresh_review_contract_rows,
                    (
                        "item",
                        "required",
                        "status",
                        "proof_only",
                        "authorizes_action",
                        "authorizes_model_call",
                        "authorizes_tool_execution",
                        "authorizes_approval",
                        "authorizes_routing",
                        "authorizes_transcript_mutation",
                        "authorizes_personal_data_read",
                        "authorizes_external_side_effect",
                        "reusable_prior_artifact",
                    ),
                ),
                "required_commands",
                *required_commands,
                "command_intake_contract",
                command_intake_contract_sha256,
                "voice_post_run_closure_token",
                voice_post_run_closure_token_sha256,
                "post_run_artifact_hashes",
                verification_receipt_sha256,
                execution_health_sha256,
                execution_audit_sha256,
                after_action_learning_sha256,
                next_command,
                f"authorizes_action_now={authorizes_action_now}",
                f"authorizes_model_call={authorizes_model_call}",
                f"authorizes_tool_execution={authorizes_tool_execution}",
                f"authorizes_approval={authorizes_approval}",
                f"authorizes_routing={authorizes_routing}",
                f"authorizes_transcript_mutation={authorizes_transcript_mutation}",
                f"authorizes_personal_data_read={authorizes_personal_data_read}",
                f"authorizes_external_side_effect={authorizes_external_side_effect}",
                f"authorizes_next_voice_review={authorizes_next_voice_review}",
                f"reusable_for_next_voice_review={reusable_for_next_voice_review}",
                "voice_cycle_ledger_proof_only",
                "fresh_voice_review_required",
            ]
        )
    )


_VOICE_CYCLE_STAGE_NAMES = {
    "privacy_boundary",
    "confirmation_receipt",
    "route_gate",
    "route_proof_bundle",
    "runtime_bridge",
    "command_cockpit",
    "action_audit",
    "execution_handoff",
    "post_run_closure",
}
_VOICE_CYCLE_PREFLIGHT_ITEMS = {
    "visible_privacy_receipt",
    "fresh_confirmation_receipt",
    "route_proof_bundle",
    "command_intake_bridge",
    "action_audit_boundary",
    "execution_handoff_packet",
    "post_run_artifact_hashes",
    "fresh_review_cycle_boundary",
}
_VOICE_FRESH_REVIEW_CONTRACT_ITEMS = {
    "visible_capture_privacy_receipt",
    "confirmed_transcript_receipt",
    "receipt_nonce_freshness",
    "route_proof_bundle",
    "command_intake_bridge",
    "execution_handoff_packet",
    "post_run_closure_artifacts",
    "voice_cycle_ledger_review_token",
}


def _voice_cycle_stage_rows_ready(stage_rows: list[dict[str, Any]]) -> bool:
    if len(stage_rows) != len(_VOICE_CYCLE_STAGE_NAMES):
        return False
    if {str(row.get("stage") or "") for row in stage_rows} != _VOICE_CYCLE_STAGE_NAMES:
        return False
    stage_non_authority_fields = (
        "authorizes_action",
        "authorizes_model_call",
        "authorizes_tool_execution",
        "authorizes_approval",
        "authorizes_routing",
        "authorizes_transcript_mutation",
        "authorizes_personal_data_read",
        "authorizes_external_side_effect",
        "authorizes_next_voice_review",
        "authorizes_receipt_reuse",
        "reusable_for_next_voice_review",
    )
    for row in stage_rows:
        if row.get("ready") is not True:
            return False
        if not str(row.get("state") or "") or not str(row.get("proof") or ""):
            return False
        if any(row.get(field) is not False for field in stage_non_authority_fields):
            return False
    return True


def _voice_cycle_ledger_ready(
    *,
    cycle_ready: bool,
    stage_rows: list[dict[str, Any]],
    preflight_scorecard_rows: list[dict[str, Any]],
    fresh_review_contract_rows: list[dict[str, Any]],
    post_run_artifact_hashes_present: bool,
    command_intake_contract_sha256: str,
    voice_post_run_closure_token_sha256: str,
    voice_cycle_ledger_token_sha256: str,
    verification_receipt_sha256: str,
    execution_health_sha256: str,
    execution_audit_sha256: str,
    after_action_learning_sha256: str,
) -> bool:
    if not cycle_ready or not post_run_artifact_hashes_present:
        return False
    if not _voice_cycle_stage_rows_ready(stage_rows):
        return False
    if len(preflight_scorecard_rows) != len(_VOICE_CYCLE_PREFLIGHT_ITEMS):
        return False
    if {str(row.get("item") or "") for row in preflight_scorecard_rows} != _VOICE_CYCLE_PREFLIGHT_ITEMS:
        return False
    if len(fresh_review_contract_rows) != len(_VOICE_FRESH_REVIEW_CONTRACT_ITEMS):
        return False
    if {str(row.get("item") or "") for row in fresh_review_contract_rows} != _VOICE_FRESH_REVIEW_CONTRACT_ITEMS:
        return False
    for token in (
        command_intake_contract_sha256,
        voice_post_run_closure_token_sha256,
        voice_cycle_ledger_token_sha256,
        verification_receipt_sha256,
        execution_health_sha256,
        execution_audit_sha256,
        after_action_learning_sha256,
    ):
        if not _looks_like_sha256(token):
            return False
    scorecard_non_authority_fields = (
        "authorizes_action",
        "authorizes_model_call",
        "authorizes_tool_execution",
        "authorizes_approval",
        "authorizes_routing",
        "authorizes_transcript_mutation",
        "authorizes_personal_data_read",
        "authorizes_external_side_effect",
        "authorizes_next_voice_review",
        "authorizes_receipt_reuse",
        "reusable_for_next_voice_review",
    )
    contract_non_authority_fields = (
        "authorizes_action",
        "authorizes_model_call",
        "authorizes_tool_execution",
        "authorizes_approval",
        "authorizes_routing",
        "authorizes_transcript_mutation",
        "authorizes_personal_data_read",
        "authorizes_external_side_effect",
        "reusable_prior_artifact",
    )
    for row in preflight_scorecard_rows:
        if row.get("ready") is not True or row.get("points") != row.get("max_points"):
            return False
        if row.get("required_before_next_voice_review") is not True:
            return False
        if any(row.get(field) is not False for field in scorecard_non_authority_fields):
            return False
    for row in fresh_review_contract_rows:
        if row.get("required") is not True or row.get("status") != "fresh_required" or row.get("proof_only") is not True:
            return False
        if any(row.get(field) is not False for field in contract_non_authority_fields):
            return False
    return True


def voice_capture_privacy_packet(args: dict[str, Any]) -> ToolResult:
    mode = _short_text(args.get("mode") or "browser-push-to-talk", MAX_VOICE_MODE_CHARS) or "browser-push-to-talk"
    transcript_source = _short_text(args.get("transcript_source") or "browser SpeechRecognition", 120) or "browser SpeechRecognition"
    max_duration_seconds = _bounded_int(args.get("max_duration_seconds"), 30, 5, 120)
    receipt_material = "|".join([mode, transcript_source, str(max_duration_seconds), "privacy-visible"])
    privacy_receipt_id = "voice-privacy-" + hashlib.sha256(receipt_material.encode("utf-8")).hexdigest()[:12]
    composer_state_sequence = ["idle", "armed", "listening", "confirm", "confirming", "routing", "stopped", "error"]
    route_guard_commands = [
        "voice transcript review: <transcript>",
        "voice confirmation: <transcript>",
        "voice confirmation receipt: <transcript> confirmed=true",
        "voice route gate: <transcript> confirmed=true",
    ]
    lines = [
        "Jarvis voice capture privacy packet:",
        "This is read-only. It arms the privacy boundary for one visible push-to-talk capture without requesting microphone access, recording audio, starting a listener, transcribing speech, saving transcripts, executing tools, routing actions, or queuing approvals.",
        "",
        f"Privacy receipt id: {privacy_receipt_id}",
        f"Capture mode: {mode}",
        f"Transcript source: {transcript_source}",
        f"Hard maximum duration: {max_duration_seconds} seconds",
        "",
        "Visible indicators required before live capture:",
        "- microphone state must show idle, listening, confirm, routing, stopped, or error",
        "- listen button must show pressed state only while capture is active",
        "- transcript stays in the composer until the operator presses Send",
        "- Send creates an auditable confirmation receipt before normal routing",
        "- Mic rerecord clears only the local draft, not memory, approvals, or prior messages",
        "",
        "Privacy boundary:",
        "- This packet does not request microphone access; the browser prompt can appear only when the operator presses the visible Mic control.",
        "- Browser microphone access is requested only by the visible push-to-talk control.",
        "- Audio remains browser/transient; Jarvis receives only supplied transcript text after SpeechRecognition returns it.",
        "- Transcript saving is not automatic and must be a separate explicit local-safe action.",
        "- Bystander speech, private speech, or ambiguous transcripts should be discarded or rerecorded instead of routed.",
        "",
        "Composer route contract:",
        f"- voice privacy state sequence: {' -> '.join(composer_state_sequence)}",
        "- Send is allowed to route a spoken transcript only after creating a voice confirmation receipt.",
        "- Send without a confirmation receipt must keep the transcript in the composer and block routing.",
        "- Mic rerecord clears only the local draft and does not delete memory, approvals, notes, files, or prior messages.",
        "- Stop capture only stops listening or preserves the draft; it never approves, deletes, rewrites, routes, or executes.",
        "- Route guard commands: voice transcript review, voice confirmation, voice confirmation receipt, then voice route gate.",
        "",
        "Next safe commands:",
        *[f"- {command}" for command in route_guard_commands[:3]],
    ]
    return ToolResult(
        "voice_capture_privacy_packet",
        True,
        "\n".join(lines),
        _voice_metadata(
            mode=mode,
            transcript_source=transcript_source,
            max_duration_seconds=max_duration_seconds,
            privacy_receipt_id=privacy_receipt_id,
            microphone_access_requested=False,
            visible_capture_indicator_required=True,
            push_to_talk_required=True,
            transcript_confirmation_required=True,
            confirmation_receipt_required=True,
            composer_state_sequence=composer_state_sequence,
            composer_state_count=len(composer_state_sequence),
            voice_privacy_state_element="jarvis-voice-privacy-state",
            voice_input_state_element="jarvis-input-state",
            voice_mic_button_element="jarvis-speech-toggle",
            voice_send_button_element="jarvis-command-submit",
            send_requires_confirmation_receipt=True,
            send_without_receipt_routes=False,
            voice_draft_persists_until_confirmation=True,
            rerecord_clears_local_draft_only=True,
            stop_capture_only=True,
            stop_intent_routes=False,
            stop_intent_deletes=False,
            stop_intent_approves=False,
            route_guard_commands=route_guard_commands,
            route_guard_command_count=len(route_guard_commands),
            browser_transient_audio=True,
            raw_audio_stored=False,
            recommended_next_commands=route_guard_commands[:3],
        ),
    )


def speak_text(text: str, voice: str = "", rate: int | None = None, wait: bool = True, dry_run: bool = False) -> ToolResult:
    text = text.strip()
    if not text:
        failure_output = f"No text to speak. {LOCAL_READ_INPUT_RECOVERY_ACTION}"
        return ToolResult(
            "speak",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _voice_metadata(reason="missing_speech_text"),
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )
    if _has_local_path(text):
        failure_output = (
            "Speech text cannot be a local file path. "
            f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return ToolResult(
            "speak",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _voice_metadata(reason="invalid_speech_text", text=_short_text(text, MAX_SPEECH_CHARS), chars=len(text)),
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )
    if len(text) > MAX_SPEECH_CHARS:
        failure_output = (
            f"Refusing to speak oversized text: {len(text)} chars exceeds {MAX_SPEECH_CHARS}. "
            f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return ToolResult(
            "speak",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _voice_metadata(reason="oversized_speech_text", chars=len(text), max_chars=MAX_SPEECH_CHARS),
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )

    voice = _macos_voice_for_text(text, voice)
    command = ["say"]
    if voice:
        command.extend(["-v", voice])
    sanitized_rate = None
    if rate is not None:
        sanitized_rate = _bounded_int(rate, 175, 80, 360)
        command.extend(["-r", str(sanitized_rate)])
    command.append(text)

    if dry_run:
        return ToolResult(
            "speak",
            True,
            "Speech command ready: " + " ".join(command[:-1]) + " <text>",
            _voice_metadata(chars=len(text), voice=voice, rate=sanitized_rate, dry_run=True),
        )

    try:
        if wait:
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                return _voice_speech_outcome_unknown(
                    chars=len(text),
                    voice=voice,
                    rate=sanitized_rate,
                    dry_run=False,
                    returncode=result.returncode,
                    executes_tools=True,
                )
        else:
            subprocess.Popen(command)
        return ToolResult(
            "speak",
            True,
            f"Spoke {len(text)} chars.",
            _voice_metadata(chars=len(text), voice=voice, rate=sanitized_rate, dry_run=False, speaks=True, speaks_audio=True, executes_tools=True),
        )
    except Exception as exc:
        return _voice_speech_outcome_unknown(
            chars=len(text),
            voice=voice,
            rate=sanitized_rate,
            dry_run=False,
            exception_type=type(exc).__name__,
            executes_tools=True,
        )


def resolve_speak_approval(args: dict[str, Any]) -> ApprovalArgumentResolution | ToolResult:
    text = str(args.get("text") or "").strip()
    voice = str(args.get("voice") or "").strip()
    if _has_local_path(text) or _has_local_path(voice):
        refused = _voice_input_failure(
            "speak",
            "Speech text and voice cannot contain a local file path.",
            reason="invalid_speech_input",
        )
        refused.metadata.update({"executed_handler": False, "handler_invoked": False})
        return refused
    if len(text) > MAX_SPEECH_CHARS:
        refused = _voice_input_failure(
            "speak",
            f"Refusing to speak oversized text: {len(text)} chars exceeds {MAX_SPEECH_CHARS}.",
            reason="oversized_speech_text",
            chars=len(text),
            max_chars=MAX_SPEECH_CHARS,
        )
        refused.metadata.update({"executed_handler": False, "handler_invoked": False})
        return refused
    resolved = {"text": text}
    for key in ("voice", "rate", "wait", "dry_run"):
        if key in args:
            resolved[key] = voice if key == "voice" else args[key]
    return ApprovalArgumentResolution(resolved, {})


def speak(args: dict[str, Any]) -> ToolResult:
    rate = args.get("rate")
    return speak_text(
        str(args.get("text") or ""),
        voice=str(args.get("voice") or "").strip(),
        rate=_bounded_int(rate, 175, 80, 360) if rate is not None else None,
        wait=bool(args.get("wait", True)),
        dry_run=bool(args.get("dry_run", False)),
    )


def voice_reply_preview(args: dict[str, Any]) -> ToolResult:
    text = re.sub(r"\s+", " ", str(args.get("text") or "").strip())
    if not text:
        return _voice_input_failure(
            "voice_reply_preview",
            "Text is required. Try `voice reply preview: Jarvis is ready.`",
            reason="missing_preview_text",
        )
    if len(text) > MAX_SPEECH_CHARS:
        return _voice_input_failure(
            "voice_reply_preview",
            f"Text is too large for speech preview ({len(text)} chars). Limit is {MAX_SPEECH_CHARS}.",
            reason="oversized_preview_text",
            text_chars=len(text),
            max_chars=MAX_SPEECH_CHARS,
        )
    display_text = _short_text(text, MAX_SPEECH_CHARS)
    voice = str(args.get("voice") or "").strip()
    rate = _bounded_int(args.get("rate"), 175, 80, 360)
    words = re.findall(r"\S+", text)
    estimated_seconds = max(1, round((len(words) / rate) * 60))
    command = ["say"]
    if voice:
        command.extend(["-v", voice])
    command.extend(["-r", str(rate), "<text>"])
    lines = [
        "Jarvis voice reply preview:",
        "This is read-only. It prepares a spoken-response packet without speaking, requesting microphone access, recording audio, starting a listener, saving transcripts, executing tools, writing memory, or queuing approvals.",
        "",
        "Text to speak:",
        f"- {display_text}",
        "",
        "Speech settings:",
        f"- voice: {voice or 'system default'}",
        f"- rate: {rate} words per minute",
        f"- words: {len(words)}",
        f"- estimated duration: {estimated_seconds} seconds",
        f"- command preview: {' '.join(command)}",
        "",
        "Safety boundary:",
        "- Speaking text can reveal private content to nearby people, so Jarvis should preview before speaking sensitive text.",
        "- Actual `speak ...` is a separate local-safe action and should be used only after the operator intentionally asks.",
        "- Voice output does not bypass planner, ToolRegistry, PermissionPolicy, approval queue, or audit logging for actions.",
    ]
    return ToolResult(
        "voice_reply_preview",
        True,
        "\n".join(lines),
        _voice_metadata(text_chars=len(text), words=len(words), voice=voice, rate=rate, estimated_seconds=estimated_seconds),
    )


def make_spoken_turn_rehearsal(get_tool: Callable[[str], Any]):
    def spoken_turn_rehearsal(args: dict[str, Any]) -> ToolResult:
        message = re.sub(r"\s+", " ", str(args.get("message") or "").strip())
        if not message:
            return _voice_input_failure(
                "spoken_turn_rehearsal",
                "Message is required. Try `spoken turn rehearsal: what should Jarvis do next?`.",
                reason="missing_rehearsal_message",
            )

        from jarvis_v2.agent.planner import RuleBasedPlanner

        plan = RuleBasedPlanner().plan(message)
        planned_actions: list[dict[str, Any]] = []
        action_lines: list[str] = []
        approval_required = False
        mode = "chat" if plan.needs_model and not plan.actions else "tool"

        if mode == "chat":
            action_lines.append("- Route to ChatBrain for a conversational answer; no tool would run from this rehearsal.")
            spoken_candidate = "I can answer that conversationally, then preview the spoken response before speaking."
        elif not plan.actions:
            action_lines.append("- No registered tool action was planned.")
            spoken_candidate = "I do not have a clear action for that yet, so I would ask a clarifying question."
        else:
            for index, action in enumerate(plan.actions, start=1):
                try:
                    tool = get_tool(action.tool_name)
                    requires_approval = tool.risk.name in {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"}
                    approval_required = approval_required or requires_approval
                    planned_actions.append(
                        {
                            "tool": action.tool_name,
                            "risk": tool.risk.name,
                            "toolset": tool.toolset,
                            "requires_approval": requires_approval,
                            "args": action.args,
                        }
                    )
                    approval_text = "approval required" if requires_approval else "would be allowed by default"
                    action_lines.append(f"- {index}. {action.tool_name} [{tool.toolset}, {tool.risk.name}]: {approval_text}.")
                    if action.args:
                        action_lines.append(f"  Planned args: {action.args}")
                except KeyError:
                    approval_required = True
                    planned_actions.append(
                        {
                            "tool": action.tool_name,
                            "risk": "UNKNOWN",
                            "toolset": "unknown",
                            "requires_approval": True,
                            "args": action.args,
                        }
                    )
                    action_lines.append(f"- {index}. {action.tool_name} [unknown]: approval required because the tool is not registered.")
            if approval_required:
                spoken_candidate = "That request needs approval before I do anything. I can show the safety receipt and last-look packet first."
            else:
                spoken_candidate = "I can do that as a low-risk action, then report what happened."

        preview = voice_reply_preview({"text": spoken_candidate})
        lines = [
            "Jarvis spoken turn rehearsal:",
            "This is read-only. It previews whether a user message becomes natural chat, a tool action, or an approval-gated action before any speech or execution.",
            "",
            "User message:",
            f"- {message}",
            "",
            "Route preview:",
            f"- mode: {mode}",
            f"- goal: {plan.goal}",
            *action_lines,
            "",
            "Spoken-response preview:",
            preview.output,
            "",
            "Safety boundary:",
            "- This rehearsal does not call a chat model, speak audio, request microphone access, record audio, start a listener, save transcripts, execute tools, write memory, or queue approvals.",
            "- If the route becomes a real action later, ToolRegistry, PermissionPolicy, approval queue, and audit logging still apply.",
        ]
        return ToolResult(
            "spoken_turn_rehearsal",
            True,
            "\n".join(lines),
            _voice_metadata(
                message=message,
                mode=mode,
                planned_goal=plan.goal,
                planned_actions=planned_actions,
                approval_required=approval_required,
                spoken_candidate=spoken_candidate,
            ),
        )

    return spoken_turn_rehearsal


def voice_input_plan(args: dict[str, Any]) -> ToolResult:
    mode = _short_text(args.get("mode") or "push-to-talk", MAX_VOICE_MODE_CHARS) or "push-to-talk"
    lines = [
        "Jarvis voice input boundary plan:",
        f"Preferred mode: {mode}",
        "",
        "Current state:",
        "- Voice output is available through macOS text-to-speech.",
        "- Wake-word, ASR, always-listening microphone input, and background recording are not active in V2.",
        "- This plan is read-only and does not request microphone access, record audio, transcribe speech, or start a listener.",
        "",
        "Safe input modes:",
        "- Push-to-talk: the operator explicitly starts and stops one recording window.",
        "- File import: the operator provides an audio file intentionally, and Jarvis transcribes only that file.",
        "- Wake-word later: only after a visible on/off switch, local processing preference, and clear recording indicator exist.",
        "",
        "Privacy boundaries:",
        "- Microphone access is PERSONAL_DATA and must be opt-in.",
        "- Raw audio should stay transient by default and should not be written to memory or Obsidian unless the operator explicitly asks.",
        "- Transcripts may contain private speech, names, rooms, meetings, or background voices, so transcript saving needs an explicit local-safe save action.",
        "- Do not listen in the background, infer bystanders, or summarize private conversations without the operator starting the capture.",
        "",
        "Recommended implementation stages:",
        "1. Add a read-only setup check for ASR dependencies and microphone permission state.",
        "2. Add offline transcription for user-provided audio files before live microphone capture.",
        "3. Add push-to-talk capture with a hard maximum duration and visible start/stop status.",
        "4. Add transcript preview; route actions only after the operator confirms the transcript is correct.",
        "5. Consider wake-word only after push-to-talk is stable, auditable, and easy to disable.",
        "",
        "Action boundary:",
        "- Transcribed commands still go through the normal planner, ToolRegistry, PermissionPolicy, approval queue, and audit log.",
        "- Shell/code, screenshots, clipboard reads, reminders, personal connectors, destructive changes, and computer control remain approval-gated even when requested by voice.",
        "",
        "Good first build target:",
        "- `voice input plan push-to-talk`, then a future `voice setup check` before any recording or ASR tool.",
    ]
    return ToolResult(
        "voice_input_plan",
        True,
        "\n".join(lines),
        _voice_metadata(mode=mode),
    )


def voice_setup_check(_: dict[str, Any]) -> ToolResult:
    packages = {
        "sounddevice": importlib.util.find_spec("sounddevice") is not None,
        "pyaudio": importlib.util.find_spec("pyaudio") is not None,
        "speech_recognition": importlib.util.find_spec("speech_recognition") is not None,
        "whisper": importlib.util.find_spec("whisper") is not None,
        "faster_whisper": importlib.util.find_spec("faster_whisper") is not None,
    }
    commands = {
        "say": shutil.which("say") is not None,
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "sox": shutil.which("sox") is not None,
    }
    asr_ready = bool(
        commands["ffmpeg"]
        and (
            packages["whisper"]
            or packages["faster_whisper"]
            or packages["speech_recognition"]
            or _audio_file_transcriber() is not None
        )
    )
    local_audio_transcriber_configured = _local_audio_transcriber_configured()
    local_audio_transcriber_source = _voice_local_transcriber_source()
    whisper_cli_model = _whisper_cli_model_label()
    whisper_cli_language = _whisper_cli_language_label()
    whisper_cli_active = local_audio_transcriber_source == "local_whisper_cli"
    approved_transcription_runner_ready = bool(asr_ready and local_audio_transcriber_configured)
    capture_ready = bool(packages["sounddevice"] or packages["pyaudio"] or commands["sox"])
    voice_warmup_enabled = _voice_warmup_enabled()
    voice_warmup_ready = bool(voice_warmup_enabled and local_audio_transcriber_configured)
    if voice_warmup_ready:
        voice_warmup_readiness = "ready to run in the background at UI start"
    elif voice_warmup_enabled:
        voice_warmup_readiness = "enabled but held until a local audio transcriber is configured"
    else:
        voice_warmup_readiness = "disabled"
    lines = [
        "Jarvis voice setup check:",
        "",
        "Platform:",
        f"- system: {platform.system()} {platform.release()}",
        f"- local speech output command `say`: {'available' if commands['say'] else 'missing'}",
        "",
        "Audio and ASR dependencies:",
    ]
    for name, available in sorted(packages.items()):
        lines.append(f"- Python package {name}: {'available' if available else 'missing'}")
    for name, available in sorted(commands.items()):
        if name != "say":
            lines.append(f"- command {name}: {'available' if available else 'missing'}")

    lines.extend(
        [
            "",
            "Whisper CLI configuration:",
            f"- active source: {'yes' if whisper_cli_active else 'no'}",
            f"- model: {whisper_cli_model}",
            f"- language: {whisper_cli_language}",
            "- fixed language applies to every CLI transcription; keep automatic detection for mixed Korean/English voice notes.",
        ]
    )

    lines.extend(
        [
            "",
            "Readiness:",
            f"- offline/file transcription readiness: {'ready to prototype' if asr_ready else 'not ready yet'}",
            f"- Jarvis local audio transcriber: {'configured' if local_audio_transcriber_configured else 'missing'}",
            f"- Jarvis local audio transcriber source: {local_audio_transcriber_source}",
            f"- approved audio-file transcription runner: {'ready after approval' if approved_transcription_runner_ready else 'held until local transcriber is configured'}",
            f"- Whisper warm start (`JARVIS_VOICE_WARMUP=1`): {'enabled' if voice_warmup_enabled else 'disabled'}",
            f"- Whisper warm start readiness: {voice_warmup_readiness}",
            "- Whisper warm start microphone access: not requested; uses a generated 0.3s silent WAV",
            f"- live capture readiness: {'dependencies present, still approval-gated' if capture_ready else 'not ready yet'}",
            "- microphone permission state: not requested or probed by this check",
            "",
            "Safety boundary:",
            "- This setup check is read-only and does not request microphone access, record audio, transcribe speech, start a listener, or save transcripts.",
            "- Microphone capture remains PERSONAL_DATA and must be opt-in with visible start/stop state and a hard duration limit.",
            "- Voice transcripts should preview first; routing a transcript into actions still uses planner, ToolRegistry, PermissionPolicy, approval queue, and audit log.",
            "",
            "Safe next build order:",
            "1. User-provided audio-file transcription with transcript preview.",
            "2. Push-to-talk capture with short maximum duration and visible recording state.",
            "3. Wake-word only after push-to-talk is stable, auditable, and easy to disable.",
        ]
    )
    return ToolResult(
        "voice_setup_check",
        True,
        "\n".join(lines),
        _voice_metadata(
            packages=packages,
            commands=commands,
            asr_ready=asr_ready,
            local_audio_transcriber_configured=local_audio_transcriber_configured,
            local_audio_transcriber_source=local_audio_transcriber_source,
            whisper_cli_model=whisper_cli_model,
            whisper_cli_language=whisper_cli_language,
            whisper_cli_active=whisper_cli_active,
            approved_transcription_runner_ready=approved_transcription_runner_ready,
            capture_ready=capture_ready,
            voice_warmup_enabled=voice_warmup_enabled,
            voice_warmup_ready=voice_warmup_ready,
            voice_warmup_uses_silent_wav=True,
            voice_warmup_touches_microphone=False,
        ),
    )


def voice_native_microphone_gate_packet(args: dict[str, Any]) -> ToolResult:
    mode = _short_text(args.get("mode") or "native-push-to-talk", MAX_VOICE_MODE_CHARS) or "native-push-to-talk"
    permission_receipt_id = _short_text(args.get("permission_receipt_id") or args.get("receipt_id") or "", 120)
    visible_state = str(args.get("visible_state") or args.get("visible_capture_state") or "").strip().lower() in {
        "true",
        "yes",
        "1",
        "visible",
        "armed",
        "ready",
        "reviewed",
    }
    duration_seconds = _bounded_int(args.get("max_duration_seconds"), 30, 5, 120)
    capture_receipt_material = "|".join([mode, permission_receipt_id or "no-permission-receipt", str(duration_seconds), "native-mic-gate"])
    capture_gate_receipt_id = "voice-native-" + hashlib.sha256(capture_receipt_material.encode("utf-8")).hexdigest()[:12]
    missing_checks: list[str] = []
    if not permission_receipt_id:
        missing_checks.append("visible microphone permission receipt")
    if not visible_state:
        missing_checks.append("visible capture state reviewed")
    if mode not in {"native-push-to-talk", "browser-push-to-talk"}:
        missing_checks.append("push-to-talk mode")
    gate_state = "NATIVE_MIC_HELD_FOR_PERMISSION_RECEIPT" if missing_checks else "NATIVE_MIC_READY_FOR_APPROVED_CAPTURE_RUNNER"
    required_before_capture = [
        "voice setup check",
        "voice capture privacy: native-push-to-talk",
        "voice native microphone gate: mode=native-push-to-talk permission_receipt_id=<receipt> visible_state=reviewed",
    ]
    required_after_transcript = [
        "voice transcript review: <transcript>",
        "voice confirmation: <transcript>",
        "voice confirmation receipt: <transcript> confirmed=true",
        "voice route proof bundle: <transcript> confirmed=true",
        "voice command cockpit: <transcript> confirmed=true",
    ]
    next_command = (
        "voice transcript review: <transcript>"
        if gate_state == "NATIVE_MIC_READY_FOR_APPROVED_CAPTURE_RUNNER"
        else "voice native microphone gate: mode=native-push-to-talk permission_receipt_id=<receipt> visible_state=reviewed"
    )
    lines = [
        "Jarvis native microphone gate packet:",
        "This is read-only. It checks whether a future local microphone capture runner has the minimum permission, visibility, and transcript-confirmation contract before any native/offline capture can start; it does not request microphone access, record audio, start a listener, transcribe speech, save transcripts, execute tools, or queue approvals.",
        "",
        "Capture gate receipt:",
        f"- receipt id: {capture_gate_receipt_id}",
        f"- requested mode: {mode}",
        f"- permission receipt supplied: {'yes' if permission_receipt_id else 'no'}",
        f"- visible capture state reviewed: {'yes' if visible_state else 'no'}",
        f"- hard maximum duration: {duration_seconds} seconds",
        "",
        "Gate decision:",
        f"- gate state: {gate_state}",
        f"- missing checks: {', '.join(missing_checks) if missing_checks else 'none'}",
        f"- next safe command: `{next_command}`",
        "",
        "Required before any native capture runner:",
        *[f"- {command}" for command in required_before_capture],
        "",
        "Required after transcript creation:",
        *[f"- {command}" for command in required_after_transcript],
        "",
        "Native capture boundary:",
        "- Native/offline microphone capture remains PERSONAL_DATA and cannot run from this read-only packet.",
        "- A future capture runner must be a separate approval-gated action with visible armed/listening/stopped/error state and a hard duration limit.",
        "- Raw audio must be transient by default; Jarvis should receive only a transcript preview until the operator confirms the text.",
        "- A transcript from native capture cannot route, approve, execute, write memory, or continue to another voice review without the confirmation receipt, receipt nonce, route proof bundle, cockpit, action audit, handoff, and post-run closure chain.",
    ]
    return ToolResult(
        "voice_native_microphone_gate_packet",
        True,
        "\n".join(lines),
        _voice_metadata(
            mode=mode,
            permission_receipt_id=permission_receipt_id,
            visible_capture_state_reviewed=visible_state,
            max_duration_seconds=duration_seconds,
            capture_gate_receipt_id=capture_gate_receipt_id,
            native_microphone_capture_enabled=False,
            native_capture_runner_separate=True,
            capture_runner_requires_approval=True,
            capture_runner_risk_level="PERSONAL_DATA",
            raw_audio_transient_required=True,
            transcript_preview_required=True,
            transcript_confirmation_required=True,
            confirmation_receipt_required=True,
            receipt_nonce_required=True,
            route_proof_bundle_required=True,
            command_cockpit_required=True,
            post_run_closure_required=True,
            gate_state=gate_state,
            missing_checks=missing_checks,
            missing_check_count=len(missing_checks),
            required_before_capture=required_before_capture,
            required_after_transcript=required_after_transcript,
            next_command=next_command,
        ),
    )


def voice_file_transcription_plan(args: dict[str, Any]) -> ToolResult:
    path_text = str(args.get("path") or "").strip()
    if len(path_text) > 2000:
        return _voice_file_read_failure(
            "voice_file_transcription_plan",
            "Audio path is too long to review safely.",
            path_chars=len(path_text),
        )
    audio_path = Path(path_text).expanduser() if path_text else None
    suffix = audio_path.suffix.lower() if audio_path else ""
    packages = {
        "speech_recognition": importlib.util.find_spec("speech_recognition") is not None,
        "whisper": importlib.util.find_spec("whisper") is not None,
        "faster_whisper": importlib.util.find_spec("faster_whisper") is not None,
    }
    commands = {
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }
    asr_ready = bool(
        commands["ffmpeg"]
        and (
            packages["whisper"]
            or packages["faster_whisper"]
            or packages["speech_recognition"]
            or _audio_file_transcriber() is not None
        )
    )
    local_audio_transcriber_configured = _local_audio_transcriber_configured()
    local_audio_transcriber_source = _voice_local_transcriber_source()
    approved_transcription_runner_ready = bool(asr_ready and local_audio_transcriber_configured)
    exists = bool(audio_path and audio_path.exists())
    is_file = bool(audio_path and audio_path.is_file())
    size_bytes = audio_path.stat().st_size if is_file else None
    recognized_extension = suffix in COMMON_AUDIO_EXTENSIONS if suffix else False
    path_hash = hashlib.sha256(str(audio_path).encode("utf-8")).hexdigest()[:16] if audio_path else ""
    dependency_rows = [
        {"name": "ffmpeg", "available": commands["ffmpeg"]},
        {"name": "whisper", "available": packages["whisper"]},
        {"name": "faster_whisper", "available": packages["faster_whisper"]},
        {"name": "speech_recognition", "available": packages["speech_recognition"]},
        {"name": "jarvis_audio_file_transcriber", "available": local_audio_transcriber_configured},
    ]
    handoff = {
        "source": "voice_file_transcription_plan",
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "path_supplied": bool(audio_path),
        "path_hash": path_hash,
        "exists": exists,
        "is_file": is_file,
        "suffix": suffix,
        "recognized_extension": recognized_extension,
        "size_bytes": size_bytes,
        "max_size_bytes": MAX_AUDIO_TRANSCRIPTION_BYTES,
        "asr_ready": asr_ready,
        "local_audio_transcriber_configured": local_audio_transcriber_configured,
        "local_audio_transcriber_source": local_audio_transcriber_source,
        "approved_transcription_runner_ready": approved_transcription_runner_ready,
        "dependency_rows": dependency_rows,
        "dependency_row_count": len(dependency_rows),
        "future_transcription_requires_approval": True,
        "next_safe_command": "voice audio file gate: <path> consent=true",
        "boundaries": _voice_readonly_boundaries(),
    }

    lines = [
        "Jarvis audio-file transcription plan:",
        "",
        "Purpose:",
        "- Prepare a safe file-import transcription flow before Jarvis has live microphone capture.",
        "- This plan is read-only and does not open the audio file, decode audio, transcribe speech, request microphone access, record audio, start a listener, save transcripts, execute tools, store memory, or queue approvals.",
        "",
        "Requested file:",
    ]
    if audio_path:
        lines.extend(
            [
                f"- path: {audio_path}",
                f"- exists: {'yes' if exists else 'no'}",
                f"- file: {'yes' if is_file else 'no'}",
                f"- extension: {suffix or 'none'}",
                f"- recognized audio extension: {'yes' if recognized_extension else 'no'}",
            ]
        )
        if size_bytes is not None:
            lines.append(f"- size bytes: {size_bytes}")
    else:
        lines.append("- none supplied; try `voice file transcription plan: /path/to/audio.m4a`")

    lines.extend(
        [
            "",
            "Local readiness:",
            f"- ffmpeg: {'available' if commands['ffmpeg'] else 'missing'}",
            f"- whisper: {'available' if packages['whisper'] else 'missing'}",
            f"- faster_whisper: {'available' if packages['faster_whisper'] else 'missing'}",
            f"- speech_recognition: {'available' if packages['speech_recognition'] else 'missing'}",
            f"- offline/file transcription readiness: {'ready to prototype' if asr_ready else 'not ready yet'}",
            f"- Jarvis local audio transcriber: {'configured' if local_audio_transcriber_configured else 'missing'}",
            f"- Jarvis local audio transcriber source: {local_audio_transcriber_source}",
            f"- approved audio-file transcription runner: {'ready after approval' if approved_transcription_runner_ready else 'held until local transcriber is configured'}",
            "",
            "Safe future flow:",
            "1. the operator supplies an audio file path intentionally.",
            "2. Jarvis checks metadata and asks before reading or decoding the file.",
            "3. Jarvis transcribes into a temporary preview only.",
            "4. the operator confirms the transcript is accurate and intended.",
            "5. Jarvis runs `voice transcript review: ...`, then `voice confirmation: ...`, then `voice command lifecycle: ...`.",
            "6. Any real action from the transcript still goes through ToolRegistry, PermissionPolicy, approval queue, and audit log.",
            "",
            "Privacy boundary:",
            "- Audio files can contain the operator, bystanders, meetings, locations, or private background speech.",
            "- Raw audio should stay transient by default and should not be copied to memory or Obsidian unless the operator explicitly asks.",
            "- Transcript saving should be a separate local-safe action after the operator reviews the text.",
            "",
            "Implementation guardrails for the future transcriber:",
            "- require an explicit file path from the operator",
            "- reject directories and unknown extensions by default",
            "- use a short maximum duration or size limit before decoding",
            "- show transcript preview before planner routing",
            "- never start live microphone capture from this file-import path",
        ]
    )
    return ToolResult(
        "voice_file_transcription_plan",
        True,
        "\n".join(lines),
        _voice_metadata(
            voice_file_transcription_plan_handoff_ready=True,
            voice_file_transcription_plan_handoff=handoff,
            voice_file_transcription_plan_ready_for_operator=handoff["ready_for_operator"],
            voice_file_transcription_plan_state_changed=handoff["state_changed"],
            voice_file_transcription_plan_changed=handoff["changed"],
            voice_file_transcription_plan_content_in_handoff=handoff["content_in_handoff"],
            voice_file_transcription_plan_boundaries=handoff["boundaries"],
            voice_file_transcription_plan_next_safe_command=handoff["next_safe_command"],
            ready_for_operator=True,
            state_changed=False,
            changed=[],
            content_in_handoff=False,
            path=str(audio_path) if audio_path else "",
            path_hash=path_hash,
            exists=exists,
            is_file=is_file,
            size_bytes=size_bytes,
            max_size_bytes=MAX_AUDIO_TRANSCRIPTION_BYTES,
            recognized_extension=recognized_extension,
            asr_ready=asr_ready,
            local_audio_transcriber_configured=local_audio_transcriber_configured,
            local_audio_transcriber_source=local_audio_transcriber_source,
            approved_transcription_runner_ready=approved_transcription_runner_ready,
            dependency_rows=dependency_rows,
            dependency_row_count=len(dependency_rows),
            future_transcription_requires_approval=True,
        ),
    )


def voice_audio_file_gate_packet(args: dict[str, Any]) -> ToolResult:
    path_text = str(args.get("path") or "").strip()
    consent = str(args.get("consent") or args.get("reviewed") or args.get("confirmed") or "").strip().lower() in {
        "true",
        "yes",
        "1",
        "approved",
        "confirmed",
        "reviewed",
    }
    if not path_text:
        return _voice_file_read_failure(
            "voice_audio_file_gate_packet",
            "Give Jarvis an audio path to gate, for example: `voice audio file gate: /path/to/audio.m4a consent=true`.",
            consent=consent,
        )
    if len(path_text) > 2000:
        return _voice_file_read_failure(
            "voice_audio_file_gate_packet",
            "Audio path is too long to gate safely.",
            path_chars=len(path_text),
            consent=consent,
        )

    audio_path = Path(path_text).expanduser()
    suffix = audio_path.suffix.lower()
    packages = {
        "speech_recognition": importlib.util.find_spec("speech_recognition") is not None,
        "whisper": importlib.util.find_spec("whisper") is not None,
        "faster_whisper": importlib.util.find_spec("faster_whisper") is not None,
    }
    commands = {"ffmpeg": shutil.which("ffmpeg") is not None}
    asr_ready = bool(
        commands["ffmpeg"]
        and (
            packages["whisper"]
            or packages["faster_whisper"]
            or packages["speech_recognition"]
            or _audio_file_transcriber() is not None
        )
    )
    local_audio_transcriber_configured = _local_audio_transcriber_configured()
    local_audio_transcriber_source = _voice_local_transcriber_source()
    approved_transcription_runner_ready = bool(asr_ready and local_audio_transcriber_configured)
    exists = audio_path.exists()
    is_file = audio_path.is_file() if exists else False
    size_bytes = audio_path.stat().st_size if is_file else None
    recognized_extension = suffix in COMMON_AUDIO_EXTENSIONS if suffix else False
    size_allowed = bool(size_bytes is not None and size_bytes <= MAX_AUDIO_TRANSCRIPTION_BYTES)
    missing_checks: list[str] = []
    if not consent:
        missing_checks.append("operator consent=true")
    if not exists:
        missing_checks.append("file exists")
    if exists and not is_file:
        missing_checks.append("path is a file")
    if not recognized_extension:
        missing_checks.append("recognized audio extension")
    if size_bytes is None:
        missing_checks.append("file size known")
    elif not size_allowed:
        missing_checks.append("file below size limit")
    if not asr_ready:
        missing_checks.append("offline transcription dependency ready")
    if not local_audio_transcriber_configured:
        missing_checks.append("local audio transcriber configured")

    gate_state = "AUDIO_FILE_READY_FOR_APPROVED_TRANSCRIPTION" if not missing_checks else "AUDIO_FILE_HELD_FOR_REVIEW"
    next_command = (
        f"voice file transcription plan: {audio_path}"
        if gate_state == "AUDIO_FILE_READY_FOR_APPROVED_TRANSCRIPTION"
        else f"voice audio file gate: {audio_path} consent=true"
    )
    path_hash = hashlib.sha256(str(audio_path).encode("utf-8")).hexdigest()[:16]
    file_receipt_id = "voice-file-" + hashlib.sha256(
        "|".join([str(audio_path), str(size_bytes), "consent" if consent else "no-consent"]).encode("utf-8")
    ).hexdigest()[:12]
    dependency_rows = [
        {"name": "ffmpeg", "available": commands["ffmpeg"]},
        {"name": "whisper", "available": packages["whisper"]},
        {"name": "faster_whisper", "available": packages["faster_whisper"]},
        {"name": "speech_recognition", "available": packages["speech_recognition"]},
        {"name": "jarvis_audio_file_transcriber", "available": local_audio_transcriber_configured},
    ]
    handoff = {
        "source": "voice_audio_file_gate_packet",
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "receipt_id": file_receipt_id,
        "path_hash": path_hash,
        "consent": consent,
        "exists": exists,
        "is_file": is_file,
        "suffix": suffix,
        "recognized_extension": recognized_extension,
        "size_bytes": size_bytes,
        "max_size_bytes": MAX_AUDIO_TRANSCRIPTION_BYTES,
        "size_allowed": size_allowed,
        "asr_ready": asr_ready,
        "local_audio_transcriber_configured": local_audio_transcriber_configured,
        "local_audio_transcriber_source": local_audio_transcriber_source,
        "approved_transcription_runner_ready": approved_transcription_runner_ready,
        "gate_state": gate_state,
        "missing_checks": list(missing_checks),
        "missing_check_count": len(missing_checks),
        "dependency_rows": dependency_rows,
        "dependency_row_count": len(dependency_rows),
        "can_transcribe_now": False,
        "future_transcription_requires_approval": True,
        "next_safe_command": next_command,
        "boundaries": _voice_readonly_boundaries(),
    }

    lines = [
        "Jarvis voice audio-file gate packet:",
        "This is read-only. It checks whether a user-supplied audio file is ready for a future approved transcription step without opening audio, decoding speech, transcribing audio, saving transcripts, executing tools, or queuing approvals.",
        "",
        "File receipt:",
        f"- receipt id: {file_receipt_id}",
        f"- path hash: `{path_hash}`",
        f"- consent supplied: {'yes' if consent else 'no'}",
        "",
        "File checks:",
        f"- path: {audio_path}",
        f"- exists: {'yes' if exists else 'no'}",
        f"- file: {'yes' if is_file else 'no'}",
        f"- extension: {suffix or 'none'}",
        f"- recognized audio extension: {'yes' if recognized_extension else 'no'}",
        f"- size bytes: {size_bytes if size_bytes is not None else 'unknown'}",
        f"- max size bytes: {MAX_AUDIO_TRANSCRIPTION_BYTES}",
        "",
        "Dependency checks:",
        f"- ffmpeg: {'available' if commands['ffmpeg'] else 'missing'}",
        f"- whisper: {'available' if packages['whisper'] else 'missing'}",
        f"- faster_whisper: {'available' if packages['faster_whisper'] else 'missing'}",
        f"- speech_recognition: {'available' if packages['speech_recognition'] else 'missing'}",
        f"- Jarvis local audio transcriber: {'configured' if local_audio_transcriber_configured else 'missing'}",
        f"- Jarvis local audio transcriber source: {local_audio_transcriber_source}",
        "",
        "Gate decision:",
        f"- gate state: {gate_state}",
        f"- missing checks: {', '.join(missing_checks) if missing_checks else 'none'}",
        f"- next safe command: `{next_command}`",
        "",
        "Approved future flow:",
        "- Only a separate future transcription runner may open or decode the audio file, and that runner must keep raw audio transient by default.",
        "- The first output after transcription must be a transcript preview, followed by `voice transcript review`, `voice confirmation`, `voice confirmation receipt`, and `voice route gate` before normal action routing.",
        "- Transcript saving remains a separate explicit local-safe action after the operator reviews the text.",
        "",
        "Safety boundary:",
        "- This packet inspects path metadata only; it does not read the audio file contents, request microphone access, record audio, transcribe speech, save transcripts, execute tools, call models, approve requests, write memory, or queue approvals.",
    ]

    return ToolResult(
        "voice_audio_file_gate_packet",
        True,
        "\n".join(lines),
        _voice_metadata(
            voice_audio_file_gate_handoff_ready=True,
            voice_audio_file_gate_handoff=handoff,
            voice_audio_file_gate_ready_for_operator=handoff["ready_for_operator"],
            voice_audio_file_gate_state_changed=handoff["state_changed"],
            voice_audio_file_gate_changed=handoff["changed"],
            voice_audio_file_gate_content_in_handoff=handoff["content_in_handoff"],
            voice_audio_file_gate_boundaries=handoff["boundaries"],
            voice_audio_file_gate_next_safe_command=handoff["next_safe_command"],
            ready_for_operator=True,
            state_changed=False,
            changed=[],
            content_in_handoff=False,
            path=str(audio_path),
            path_hash=path_hash,
            consent=consent,
            receipt_id=file_receipt_id,
            exists=exists,
            is_file=is_file,
            suffix=suffix,
            recognized_extension=recognized_extension,
            size_bytes=size_bytes,
            max_size_bytes=MAX_AUDIO_TRANSCRIPTION_BYTES,
            size_allowed=size_allowed,
            asr_ready=asr_ready,
            local_audio_transcriber_configured=local_audio_transcriber_configured,
            local_audio_transcriber_source=local_audio_transcriber_source,
            approved_transcription_runner_ready=approved_transcription_runner_ready,
            gate_state=gate_state,
            missing_checks=missing_checks,
            missing_check_count=len(missing_checks),
            dependency_rows=dependency_rows,
            dependency_row_count=len(dependency_rows),
            can_transcribe_now=False,
            future_transcription_requires_approval=True,
            next_command=next_command,
        ),
    )


def voice_audio_file_transcription_preview(args: dict[str, Any]) -> ToolResult:
    path_text = str(args.get("path") or "").strip()
    consent = str(args.get("consent") or args.get("reviewed") or args.get("confirmed") or "").strip().lower() in {
        "true",
        "yes",
        "1",
        "approved",
        "confirmed",
        "reviewed",
    }
    receipt_id = str(args.get("receipt_id") or args.get("file_receipt_id") or "").strip()
    if not path_text:
        return _voice_file_read_failure(
            "voice_audio_file_transcription_preview",
            "Give Jarvis an audio path to transcribe, for example: `voice audio file transcribe: /path/to/audio.m4a consent=true receipt_id=voice-file-...`.",
            consent=consent,
            receipt_id=receipt_id,
            requires_approval=True,
        )
    if len(path_text) > 2000:
        return _voice_file_read_failure(
            "voice_audio_file_transcription_preview",
            "Audio path is too long to transcribe safely.",
            path_chars=len(path_text),
            consent=consent,
            receipt_id=receipt_id,
            requires_approval=True,
        )

    audio_path = Path(path_text).expanduser()
    suffix = audio_path.suffix.lower()
    exists = audio_path.exists()
    is_file = audio_path.is_file() if exists else False
    size_bytes = audio_path.stat().st_size if is_file else None
    recognized_extension = suffix in COMMON_AUDIO_EXTENSIONS if suffix else False
    size_allowed = bool(size_bytes is not None and size_bytes <= MAX_AUDIO_TRANSCRIPTION_BYTES)
    path_hash = hashlib.sha256(str(audio_path).encode("utf-8")).hexdigest()[:16]
    missing_checks: list[str] = []
    if not consent:
        missing_checks.append("operator consent=true")
    if not receipt_id.startswith("voice-file-"):
        missing_checks.append("voice file gate receipt_id")
    if not exists:
        missing_checks.append("file exists")
    if exists and not is_file:
        missing_checks.append("path is a file")
    if not recognized_extension:
        missing_checks.append("recognized audio extension")
    if size_bytes is None:
        missing_checks.append("file size known")
    elif not size_allowed:
        missing_checks.append("file below size limit")
    transcriber = _audio_file_transcriber()
    local_audio_transcriber_source = _voice_local_transcriber_source()
    if transcriber is None:
        missing_checks.append("local audio transcriber configured")

    if missing_checks:
        failure_output = "\n".join(
            [
                "Jarvis audio-file transcription preview is held.",
                "No audio was opened, decoded, transcribed, saved, routed, or executed.",
                f"- path hash: `{path_hash}`",
                f"- missing checks: {', '.join(missing_checks)}",
                "- Run the read-only gate first: `voice audio file gate: <path> consent=true`",
                "- Then approve the queued PERSONAL_DATA transcription request only if the path and receipt are still trusted.",
                f"- {VOICE_FILE_READ_RECOVERY_ACTION}",
            ]
        )
        return ToolResult(
            "voice_audio_file_transcription_preview",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _voice_metadata(
                    path=str(audio_path),
                    path_hash=path_hash,
                    consent=consent,
                    receipt_id=receipt_id,
                    exists=exists,
                    is_file=is_file,
                    suffix=suffix,
                    recognized_extension=recognized_extension,
                    size_bytes=size_bytes,
                    max_size_bytes=MAX_AUDIO_TRANSCRIPTION_BYTES,
                    size_allowed=size_allowed,
                    local_audio_transcriber_source=local_audio_transcriber_source,
                    missing_checks=missing_checks,
                    missing_check_count=len(missing_checks),
                    requires_approval=True,
                    reads_audio_file=False,
                    transcribes_audio=False,
                    content_in_handoff=False,
                    state_changed=False,
                    changed=[],
                ),
                output=failure_output,
                action=VOICE_FILE_READ_RECOVERY_ACTION,
            ),
        )

    try:
        transcript = str(transcriber(audio_path)).strip()
    except Exception as exc:
        failure_output = (
            f"{_voice_error('transcribe the approved audio file', exc)} "
            f"{VOICE_TRANSCRIPTION_RECOVERY_ACTION}"
        )
        return ToolResult(
            "voice_audio_file_transcription_preview",
            False,
            failure_output,
            declare_retryable_personal_read_failure(
                _voice_metadata(
                    path=str(audio_path),
                    path_hash=path_hash,
                    consent=consent,
                    receipt_id=receipt_id,
                    exists=exists,
                    is_file=is_file,
                    suffix=suffix,
                    recognized_extension=recognized_extension,
                    size_bytes=size_bytes,
                    max_size_bytes=MAX_AUDIO_TRANSCRIPTION_BYTES,
                    size_allowed=size_allowed,
                    local_audio_transcriber_source=local_audio_transcriber_source,
                    requires_approval=True,
                    reads_private_data=True,
                    reads_personal_data=True,
                    reads_audio_file=True,
                    transcribes_audio=True,
                    saves_transcript=False,
                    exception_type=type(exc).__name__,
                    state_changed=False,
                    changed=[],
                ),
                output=failure_output,
                action=VOICE_TRANSCRIPTION_RECOVERY_ACTION,
                commands=("voice setup check",),
            ),
        )

    transcript = transcript[:MAX_TRANSCRIPT_CHARS]
    transcript_sha256 = _transcript_sha256(transcript)
    handoff = {
        "source": "voice_audio_file_transcription_preview",
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "receipt_id": receipt_id,
        "path_hash": path_hash,
        "size_bytes": size_bytes,
        "transcript_chars": len(transcript),
        "transcript_sha256": transcript_sha256,
        "transcript_preview": transcript,
        "local_audio_transcriber_source": local_audio_transcriber_source,
        "next_safe_command": f"voice transcript review: {transcript}",
        "boundaries": _voice_transcription_boundaries(),
    }
    lines = [
        "Jarvis audio-file transcription preview:",
        "",
        "Transcript preview:",
        transcript or "(empty transcript)",
        "",
        "Next safe steps:",
        f"- `voice transcript review: {transcript}`",
        "- `voice confirmation: <reviewed transcript>`",
        "- `voice confirmation receipt: <reviewed transcript> confirmed=true`",
        "- `voice route gate: <reviewed transcript> confirmed=true`",
        "",
        "Safety boundary:",
        "- This approved runner reads and transcribes only the supplied audio file.",
        "- It does not save the transcript, route actions, execute tools, write memory, start a listener, record microphone audio, call models, or approve any follow-up action.",
    ]
    return ToolResult(
        "voice_audio_file_transcription_preview",
        True,
        "\n".join(lines),
        _voice_metadata(
            voice_audio_file_transcription_handoff_ready=True,
            voice_audio_file_transcription_handoff=handoff,
            voice_audio_file_transcription_ready_for_operator=handoff["ready_for_operator"],
            voice_audio_file_transcription_state_changed=handoff["state_changed"],
            voice_audio_file_transcription_changed=handoff["changed"],
            voice_audio_file_transcription_content_in_handoff=handoff["content_in_handoff"],
            voice_audio_file_transcription_boundaries=handoff["boundaries"],
            voice_audio_file_transcription_next_safe_command=handoff["next_safe_command"],
            voice_audio_file_transcription_transcript_chars=handoff["transcript_chars"],
            voice_audio_file_transcription_transcript_sha256=handoff["transcript_sha256"],
            ready_for_operator=True,
            state_changed=False,
            changed=[],
            content_in_handoff=True,
            path=str(audio_path),
            path_hash=path_hash,
            consent=consent,
            receipt_id=receipt_id,
            exists=exists,
            is_file=is_file,
            suffix=suffix,
            recognized_extension=recognized_extension,
            size_bytes=size_bytes,
            max_size_bytes=MAX_AUDIO_TRANSCRIPTION_BYTES,
            size_allowed=size_allowed,
            local_audio_transcriber_source=local_audio_transcriber_source,
            transcript_chars=len(transcript),
            transcript_sha256=transcript_sha256,
            requires_approval=True,
            reads_private_data=True,
            reads_personal_data=True,
            reads_audio_file=True,
            transcribes_audio=True,
            saves_transcript=False,
            writes_files=False,
            writes_database=False,
            writes_memory=False,
            writes_notes=False,
            executes_tools=False,
            state="TRANSCRIPT_PREVIEW_READY",
            next_command=f"voice transcript review: {transcript}",
        ),
    )


def make_voice_transcript_review(get_tool: Callable[[str], Any]):
    def voice_transcript_review(args: dict[str, Any]) -> ToolResult:
        transcript = str(args.get("transcript") or "").strip()
        if not transcript:
            return _voice_input_failure(
                "voice_transcript_review",
                "Give Jarvis a transcript to review, for example: `voice transcript review: run command python3 --version`.",
                reason="missing_transcript",
            )
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            return _voice_input_failure(
                "voice_transcript_review",
                f"Transcript is too large ({len(transcript)} chars). Limit is {MAX_TRANSCRIPT_CHARS}.",
                reason="oversized_transcript",
                transcript_chars=len(transcript),
                max_chars=MAX_TRANSCRIPT_CHARS,
            )
        normalized = re.sub(r"\s+", " ", transcript).strip()
        risk_terms = {
            "computer control": ("click", "type", "screen", "screenshot", "mouse", "keyboard", "desktop"),
            "shell/code": ("run command", "execute", "terminal", "script", "python", "npm", "git"),
            "personal data": ("clipboard", "email", "calendar", "message", "contact", "password"),
            "external side effect": ("send", "post", "buy", "purchase", "call", "remind me"),
            "destructive change": ("delete", "remove", "overwrite", "erase", "reset"),
        }
        low = normalized.lower()
        detected = []
        for label, terms in risk_terms.items():
            for term in terms:
                pattern = r"\b" + re.escape(term).replace(r"\ ", r"\s+") + r"\b"
                if re.search(pattern, low):
                    detected.append(label)
                    break

        from jarvis_v2.agent.planner import RuleBasedPlanner

        plan = RuleBasedPlanner().plan(normalized)
        action_lines: list[str] = []
        planned_actions: list[dict[str, Any]] = []
        approval_required = False
        if plan.needs_model and not plan.actions:
            action_lines.append("- Route to chat for a conversational reply; no tool would run from this review.")
        elif not plan.actions:
            action_lines.append("- No tool action was planned.")
        else:
            for index, action in enumerate(plan.actions, start=1):
                try:
                    tool = get_tool(action.tool_name)
                    requires_approval = tool.risk.name in {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"}
                    approval_required = approval_required or requires_approval
                    planned_actions.append(
                        {
                            "tool": action.tool_name,
                            "risk": tool.risk.name,
                            "toolset": tool.toolset,
                            "requires_approval": requires_approval,
                            "args": action.args,
                        }
                    )
                    approval_text = "approval required" if requires_approval else "would be allowed by default"
                    action_lines.append(f"- {index}. {action.tool_name} [{tool.toolset}, {tool.risk.name}]: {approval_text}.")
                    if action.args:
                        action_lines.append(f"  Planned args: {action.args}")
                except KeyError:
                    approval_required = True
                    planned_actions.append(
                        {
                            "tool": action.tool_name,
                            "risk": "UNKNOWN",
                            "toolset": "unknown",
                            "requires_approval": True,
                            "args": action.args,
                        }
                    )
                    action_lines.append(f"- {index}. {action.tool_name} [unknown]: approval required because the tool is not registered.")

        lines = [
            "Jarvis voice transcript review:",
            "",
            "Transcript:",
            f"- {normalized}",
            "",
            "Recognition check:",
            "- This review assumes the operator supplied the transcript intentionally.",
            "- Jarvis should show the transcript back before routing it into actions.",
            "- If the transcript is ambiguous, private, or contains bystander speech, ask for confirmation instead of acting.",
            "",
            "Risk cues:",
        ]
        if detected:
            lines.extend(f"- {label}" for label in detected)
        else:
            lines.append("- none detected by keyword scan")
        lines.extend(
            [
                "",
                "Planned action preview:",
                f"- Goal: {plan.goal}",
                *action_lines,
                "",
                "Next safe commands after the operator confirms:",
                *[f"- {command}" for command in _voice_next_commands(normalized, approval_required)],
                "",
                "Auditable confirmation handoff:",
                f"- transcript hash: `{_transcript_hash(normalized)}`",
                f"- receipt command: `{_voice_confirmation_receipt_command(normalized)}`",
                "- route blocker: transcript must have a confirmed receipt before a spoken order is sent as a normal command.",
                "",
                "Safety boundary:",
                "- This review is read-only and does not execute tools, approve requests, request microphone access, record audio, transcribe speech, start a listener, save transcripts, or call external services.",
            ]
        )
        if approval_required:
            lines.append("- If the operator confirms this transcript and runs it for real, at least one planned action would still require explicit approval.")
        else:
            lines.append("- If the operator confirms this transcript and runs it for real, the planned action stays within the default auto-run risk boundary.")

        return ToolResult(
            "voice_transcript_review",
            True,
            "\n".join(lines),
            _voice_metadata(
                transcript=normalized,
                detected_risk_cues=detected,
                planned_goal=plan.goal,
                planned_actions=planned_actions,
                approval_required=approval_required,
                transcript_hash=_transcript_hash(normalized),
                confirmation_receipt_command=_voice_confirmation_receipt_command(normalized),
                route_blocked_until_confirmation_receipt=True,
                can_route_after_confirmation_receipt=False,
                recommended_next_commands=_voice_next_commands(normalized, approval_required),
            ),
        )

    return voice_transcript_review


def make_voice_confirmation_packet(get_tool: Callable[[str], Any]):
    transcript_review = make_voice_transcript_review(get_tool)

    def voice_confirmation_packet(args: dict[str, Any]) -> ToolResult:
        transcript = str(args.get("transcript") or "").strip()
        if not transcript:
            return _voice_input_failure(
                "voice_confirmation_packet",
                "Give Jarvis a transcript to package, for example: `voice confirmation: run command python3 --version`.",
                reason="missing_transcript",
            )
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            return _voice_input_failure(
                "voice_confirmation_packet",
                f"Transcript is too large ({len(transcript)} chars). Limit is {MAX_TRANSCRIPT_CHARS}.",
                reason="oversized_transcript",
                transcript_chars=len(transcript),
                max_chars=MAX_TRANSCRIPT_CHARS,
            )
        review = transcript_review({"transcript": transcript})
        if not review.ok:
            return ToolResult("voice_confirmation_packet", False, review.output, review.metadata)

        normalized = str(review.metadata.get("transcript") or transcript).strip()
        planned_actions = list(review.metadata.get("planned_actions") or [])
        approval_required = _voice_metadata_bool(review.metadata.get("approval_required"))
        detected = list(review.metadata.get("detected_risk_cues") or [])
        if approval_required:
            confirmation_step = (
                "If the operator confirms this transcript and sends it as a normal command, Jarvis should stop at the approval queue before any risky action; use `approval readiness #ID`, `approval packet #ID`, and `approval chain proof #ID` before approving it."
            )
        else:
            confirmation_step = (
                "If the operator confirms this transcript and sends it as a normal command, Jarvis may run it under the default read-only/local-safe boundary."
            )

        action_lines = []
        if planned_actions:
            for index, action in enumerate(planned_actions, start=1):
                approval_text = "approval required" if action.get("requires_approval") else "default allowed"
                action_lines.append(
                    f"- {index}. {action.get('tool')} [{action.get('toolset')}, {action.get('risk')}]: {approval_text}"
                )
        else:
            action_lines.append("- No tool action was planned.")

        lines = [
            "Jarvis voice confirmation packet:",
            "",
            "Transcript to confirm:",
            f"- {normalized}",
            "",
            "Risk cues:",
        ]
        if detected:
            lines.extend(f"- {item}" for item in detected)
        else:
            lines.append("- none detected by keyword scan")
        lines.extend(
            [
                "",
                "Planned tools:",
                *action_lines,
                "",
                "Confirmation path:",
                f"- {confirmation_step}",
                f"- Exact command to send after the operator confirms: `{normalized}`",
                "- If the transcript is wrong, private, from a bystander, or unclear, do not run it; ask the operator to restate it.",
                "",
                "Next safe commands after the operator confirms:",
                *[f"- {command}" for command in _voice_next_commands(normalized, approval_required)],
                "",
                "Auditable confirmation handoff:",
                f"- transcript hash: `{_transcript_hash(normalized)}`",
                f"- receipt command: `{_voice_confirmation_receipt_command(normalized)}`",
                "- route blocker: this packet previews the transcript only; a confirmed receipt is required before sending it as a command.",
                "",
                "Safety boundary:",
                "- This packet is read-only and does not execute tools, approve requests, request microphone access, record audio, transcribe speech, start a listener, save transcripts, store memory, or queue approvals.",
            ]
        )
        return ToolResult(
            "voice_confirmation_packet",
            True,
            "\n".join(lines),
            _voice_metadata(
                transcript=normalized,
                detected_risk_cues=detected,
                planned_actions=planned_actions,
                approval_required=approval_required,
                next_command=normalized,
                transcript_hash=_transcript_hash(normalized),
                confirmation_receipt_command=_voice_confirmation_receipt_command(normalized),
                route_blocked_until_confirmation_receipt=True,
                can_route_after_confirmation_receipt=False,
                recommended_next_commands=_voice_next_commands(normalized, approval_required),
            ),
        )

    return voice_confirmation_packet


def make_voice_command_lifecycle(get_tool: Callable[[str], Any]):
    confirmation_packet = make_voice_confirmation_packet(get_tool)

    def voice_command_lifecycle(args: dict[str, Any]) -> ToolResult:
        transcript = str(args.get("transcript") or "").strip()
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            return _voice_input_failure(
                "voice_command_lifecycle",
                f"Transcript is too large ({len(transcript)} chars). Limit is {MAX_TRANSCRIPT_CHARS}.",
                reason="oversized_transcript",
                transcript_chars=len(transcript),
                max_chars=MAX_TRANSCRIPT_CHARS,
            )
        confirmation: ToolResult | None = None
        planned_actions: list[dict[str, Any]] = []
        approval_required = False
        normalized = ""
        if transcript:
            confirmation = confirmation_packet({"transcript": transcript})
            if not confirmation.ok:
                return ToolResult("voice_command_lifecycle", False, confirmation.output, confirmation.metadata)
            normalized = str(confirmation.metadata.get("transcript") or transcript).strip()
            planned_actions = list(confirmation.metadata.get("planned_actions") or [])
            approval_required = _voice_metadata_bool(confirmation.metadata.get("approval_required"))

        action_lines = []
        if planned_actions:
            for index, action in enumerate(planned_actions, start=1):
                approval_text = "approval required" if action.get("requires_approval") else "default allowed"
                action_lines.append(
                    f"- {index}. {action.get('tool')} [{action.get('toolset')}, {action.get('risk')}]: {approval_text}"
                )
                if action.get("args"):
                    action_lines.append(f"  Planned args: {action.get('args')}")
        elif transcript:
            action_lines.append("- No tool action was planned for the supplied transcript.")
        else:
            action_lines.append("- Add a transcript to preview planned tools, risk, and approval requirements.")

        command_line = f"`{normalized}`" if normalized else "`voice command lifecycle: run command python3 --version`"
        receipt_result: ToolResult | None = None
        receipt_id = ""
        receipt_nonce = ""
        transcript_hash = _transcript_hash(normalized) if normalized else ""
        if normalized:
            receipt_result = make_voice_confirmation_receipt(get_tool)({"transcript": normalized, "confirmed": "true"})
            if not receipt_result.ok:
                return ToolResult("voice_command_lifecycle", False, receipt_result.output, receipt_result.metadata)
            receipt_id = str(receipt_result.metadata.get("receipt_id") or "")
            receipt_nonce = str(receipt_result.metadata.get("receipt_nonce") or "")
        receipt_suffix = f" receipt_id={receipt_id} receipt_nonce={receipt_nonce}" if normalized and receipt_id and receipt_nonce else ""
        route_gate_command = f"voice route gate: {normalized} confirmed=true{receipt_suffix}" if normalized else "voice route gate: <transcript> confirmed=true receipt_id=<receipt_id> receipt_nonce=<receipt_nonce>"
        route_proof_bundle_command = f"voice route proof bundle: {normalized} confirmed=true{receipt_suffix}" if normalized else "voice route proof bundle: <transcript> confirmed=true receipt_id=<receipt_id> receipt_nonce=<receipt_nonce>"
        runtime_bridge_command = f"voice runtime bridge: {normalized} confirmed=true{receipt_suffix}" if normalized else "voice runtime bridge: <transcript> confirmed=true receipt_id=<receipt_id> receipt_nonce=<receipt_nonce>"
        cockpit_command = f"voice command cockpit: {normalized} confirmed=true{receipt_suffix}" if normalized else "voice command cockpit: <transcript> confirmed=true receipt_id=<receipt_id> receipt_nonce=<receipt_nonce>"
        action_audit_command = f"voice action audit: {normalized} confirmed=true{receipt_suffix}" if normalized else "voice action audit: <transcript> confirmed=true receipt_id=<receipt_id> receipt_nonce=<receipt_nonce>"
        execution_handoff_command = f"voice execution handoff: {normalized} confirmed=true{receipt_suffix}" if normalized else "voice execution handoff: <transcript> confirmed=true receipt_id=<receipt_id> receipt_nonce=<receipt_nonce>"
        post_run_hash_suffix = "verification_receipt_sha256=<hash>; execution_health_sha256=<hash>; execution_audit_sha256=<hash>; after_action_learning_sha256=<hash>"
        post_run_closure_command = (
            f"voice post-run closure: {normalized}; confirmed=true; receipt_id={receipt_id}; receipt_nonce={receipt_nonce}; verification reviewed; post health reviewed; post audit reviewed; learning reviewed; {post_run_hash_suffix}"
            if normalized
            else f"voice post-run closure: <transcript>; confirmed=true; receipt_id=<receipt_id>; receipt_nonce=<receipt_nonce>; verification reviewed; post health reviewed; post audit reviewed; learning reviewed; {post_run_hash_suffix}"
        )
        proof_commands = [
            f"voice transcript review: {normalized}" if normalized else "voice transcript review: <transcript>",
            f"voice confirmation: {normalized}" if normalized else "voice confirmation: <transcript>",
            _voice_confirmation_receipt_command(normalized),
            route_gate_command,
            route_proof_bundle_command,
            runtime_bridge_command,
            cockpit_command,
            action_audit_command,
            execution_handoff_command,
            post_run_closure_command,
        ]
        next_commands = _voice_next_commands(normalized, approval_required) if normalized else ["voice transcript review: <transcript>", "voice confirmation: <transcript>"]
        if normalized:
            next_commands = [
                _voice_confirmation_receipt_command(normalized),
                route_gate_command,
                route_proof_bundle_command,
                runtime_bridge_command,
                cockpit_command,
            ]
            if approval_required:
                next_commands.extend(["pending approvals", "approval readiness latest", "approval packet latest", "approval chain proof latest"])
        lines = [
            "Jarvis voice command lifecycle:",
            "",
            "Purpose:",
            "- Show the full safe path for a spoken command before Jarvis has live microphone input.",
            "- This lifecycle is read-only and does not request microphone access, record audio, start a listener, transcribe speech, save transcripts, execute tools, store memory, or queue approvals.",
            "",
            "Stages:",
            "1. Setup",
            "   - Run `voice setup check` to inspect local ASR/capture dependencies without touching microphone permissions.",
            "2. Capture boundary",
            "   - Future capture starts only through operator-controlled push-to-talk or an explicitly supplied audio file.",
            "   - Live capture needs visible recording state, hard maximum duration, and easy stop/cancel controls.",
            "3. Transcript preview",
            "   - Show the transcript back to the operator before routing it into planner actions.",
            "   - If bystander speech, private speech, or ambiguity appears, ask the operator to restate instead of acting.",
            "4. Confirmation packet",
            f"   - Build `voice confirmation: ...` so the operator can see risk cues, planned tools, and the Exact command to send: {command_line}.",
            "5. Confirmation receipt",
            "   - Build `voice confirmation receipt: ... confirmed=true` only after the operator accepts the transcript; unconfirmed receipts must not route or suggest sending the transcript.",
            "   - The receipt id and receipt nonce bind to the exact transcript hash; any changed transcript needs a fresh confirmation receipt.",
            "6. Route gate",
            "   - Build `voice route gate: ... confirmed=true receipt_id=... receipt_nonce=...` to prove the confirmed transcript can enter only normal Jarvis routing, not direct execution.",
            "7. Route proof bundle",
            "   - Build `voice route proof bundle: ... confirmed=true receipt_id=... receipt_nonce=...` to join the privacy receipt, confirmation receipt, route gate, receipt freshness, transcript-hash match, approval boundary, and required proof commands.",
            "8. Runtime bridge",
            "   - Build `voice runtime bridge: ... confirmed=true` so the exact confirmed transcript enters command intake only and cannot emit an executable tool action.",
            "9. Command cockpit and action audit",
            "   - Build `voice command cockpit: ... confirmed=true`, then `voice action audit: ... confirmed=true`, before any dispatch, readiness, approval, or execution handoff review.",
            "10. Execution handoff and post-run closure",
            "   - Build `voice execution handoff: ... confirmed=true` before a later real run, then `voice post-run closure: ...` with verification, execution health, execution audit, and after-action learning evidence before the next voice review.",
            "11. Approval gate",
            "   - Real execution happens only after the operator confirms the transcript and the proof chain reaches command intake.",
            "   - Shell/code, personal data, computer control, destructive actions, reminders, and outside-world effects still stop at approval gates.",
            "   - Use `approval readiness #ID`, `approval packet #ID`, and `approval chain proof #ID` before approving any queued risky action.",
            "12. Audit",
            "   - If a real command runs later, Jarvis records tool results and blocked approvals in the normal audit trail.",
            "",
            "Current transcript:",
        ]
        if normalized:
            lines.append(f"- {normalized}")
        else:
            lines.append("- none supplied")
        lines.extend(
            [
                "",
                "Planned action preview:",
                *action_lines,
                "",
                "Required voice proof chain:",
                *[f"- `{command}`" for command in proof_commands],
                "",
                "Next safe commands after the operator confirms:",
                *[f"- {command}" for command in next_commands],
                "",
                "Auditable confirmation handoff:",
                f"- transcript hash: `{transcript_hash}`" if normalized else "- transcript hash: none until transcript is supplied",
                f"- confirmation receipt id: `{receipt_id}`" if normalized else "- confirmation receipt id: none until transcript is confirmed",
                f"- confirmation receipt nonce: `{receipt_nonce}`" if normalized else "- confirmation receipt nonce: none until transcript is confirmed",
                f"- receipt command: `{_voice_confirmation_receipt_command(normalized)}`",
                f"- route gate command: `{route_gate_command}`",
                f"- route proof bundle command: `{route_proof_bundle_command}`",
                f"- runtime bridge command: `{runtime_bridge_command}`",
                "- route blocker: lifecycle preview does not send spoken commands; only a fresh confirmed receipt id and nonce for this exact transcript can unlock normal routing.",
                "",
                "Safety boundary:",
                "- This lifecycle does not request microphone access, record audio, start a listener, save transcripts, execute tools, store memory, or queue approvals.",
            ]
        )
        if approval_required:
            lines.append("- For this transcript, at least one planned action would still require explicit approval before real execution.")
        elif transcript:
            lines.append("- For this transcript, the planned action stays inside the default auto-run boundary after the operator confirms.")
        else:
            lines.append("- Add transcript text to see whether a future spoken command would need approval.")

        return ToolResult(
            "voice_command_lifecycle",
            True,
            "\n".join(lines),
            _voice_metadata(
                transcript=normalized,
                planned_actions=planned_actions,
                approval_required=approval_required,
                transcript_hash=transcript_hash,
                confirmation_receipt_id=receipt_id,
                confirmation_receipt_nonce=receipt_nonce,
                receipt_nonce_required=True,
                receipt_freshness_required=True,
                stale_receipt_blocks_routing=True,
                confirmation_receipt_command=_voice_confirmation_receipt_command(normalized),
                route_gate_command=route_gate_command,
                route_proof_bundle_command=route_proof_bundle_command,
                runtime_bridge_command=runtime_bridge_command,
                command_cockpit_command=cockpit_command,
                action_audit_command=action_audit_command,
                execution_handoff_command=execution_handoff_command,
                post_run_closure_command=post_run_closure_command,
                proof_commands=proof_commands,
                proof_command_count=len(proof_commands),
                route_blocked_until_confirmation_receipt=bool(normalized),
                can_route_after_confirmation_receipt=False,
                command_intake_only=True,
                can_emit_executable_tool_action=False,
                recommended_next_commands=next_commands,
            ),
        )

    return voice_command_lifecycle


def voice_stop_intent_packet(args: dict[str, Any]) -> ToolResult:
    phrase = _short_text(args.get("phrase") or args.get("transcript") or args.get("intent") or "stop listening", 240)
    draft = _short_text(args.get("draft") or args.get("current_transcript") or "", 500)
    low = phrase.lower()
    is_stop = any(token in low for token in ("stop", "cancel", "pause", "abort", "nevermind", "never mind", "rerecord", "re-record"))
    is_rerecord = any(token in low for token in ("rerecord", "re-record", "try again", "start over"))
    stop_state = "VOICE_STOP_CAPTURE_ONLY_READY" if is_stop else "VOICE_STOP_INTENT_UNCLEAR"
    next_commands = [
        "voice transcript review: <new transcript>",
        "voice confirmation: <new transcript>",
        "voice command lifecycle: <new transcript>",
    ]
    if is_rerecord:
        next_commands.insert(0, "rerecord voice command")
    lines = [
        "Jarvis voice stop intent packet:",
        "This is read-only. It separates stop/cancel/rerecord speech from transcript routing so a stop phrase never becomes a command to execute, delete memory, dismiss approvals, or rewrite prior messages.",
        "",
        "Stop intent:",
        f"- phrase: {phrase or 'missing'}",
        f"- state: {stop_state}",
        f"- capture action: {'stop current capture only' if is_stop else 'hold for clarification'}",
        f"- rerecord requested: {'yes' if is_rerecord else 'no'}",
        f"- draft transcript preserved for review: {'yes' if draft else 'none supplied'}",
        "",
        "Non-destructive boundary:",
        "- does not send confirmed transcript",
        "- does not route the stop phrase into command intake",
        "- does not delete memory, approvals, notes, files, or prior messages",
        "- does not rewrite the current transcript unless the operator explicitly chooses rerecord in the visible UI",
        "- does not queue approvals, execute tools, record audio, or start a listener",
        "",
        "Safe next commands:",
    ]
    lines.extend(f"- `{command}`" for command in next_commands)
    lines.extend(
        [
            "",
            "Operator rule:",
            "- Stop speech is a brake. It can stop capture or hold the draft locally, but it is not permission to perform any other action.",
        ]
    )
    if not is_stop:
        lines.extend(["", "Recovery:", f"- {VOICE_STOP_INPUT_RECOVERY_ACTION}"])
    output = "\n".join(lines)
    metadata = _voice_metadata(
        phrase=phrase,
        current_transcript=draft,
        stop_state=stop_state,
        stop_intent_detected=bool(is_stop),
        capture_stop_only=bool(is_stop),
        rerecord_requested=bool(is_rerecord),
        transcript_preserved=bool(draft),
        routes_actions=False,
        routes_stop_phrase_to_command_intake=False,
        sends_confirmed_transcript=False,
        deletes_memory=False,
        deletes_approvals=False,
        deletes_messages=False,
        rewrites_transcript=False,
        next_commands=next_commands,
    )
    if not is_stop:
        metadata = declare_retryable_local_read_failure(
            metadata,
            output=output,
            action=VOICE_STOP_INPUT_RECOVERY_ACTION,
        )
    return ToolResult(
        "voice_stop_intent_packet",
        bool(is_stop),
        output,
        metadata,
    )


def make_voice_confirmation_receipt(get_tool: Callable[[str], Any]):
    confirmation_packet = make_voice_confirmation_packet(get_tool)

    def voice_confirmation_receipt(args: dict[str, Any]) -> ToolResult:
        transcript = str(args.get("transcript") or "").strip()
        confirmed = str(args.get("confirmed") or args.get("reviewed") or "").strip().lower() in {"true", "yes", "1", "confirmed", "reviewed"}
        if not transcript:
            return _voice_input_failure(
                "voice_confirmation_receipt",
                "Give Jarvis a transcript to receipt, for example: `voice confirmation receipt: run command python3 --version`.",
                reason="missing_transcript",
            )
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            return _voice_input_failure(
                "voice_confirmation_receipt",
                f"Transcript is too large ({len(transcript)} chars). Limit is {MAX_TRANSCRIPT_CHARS}.",
                reason="oversized_transcript",
                transcript_chars=len(transcript),
                max_chars=MAX_TRANSCRIPT_CHARS,
            )

        packet = confirmation_packet({"transcript": transcript})
        if not packet.ok:
            return ToolResult("voice_confirmation_receipt", False, packet.output, packet.metadata)

        normalized = str(packet.metadata.get("transcript") or transcript).strip()
        planned_actions = list(packet.metadata.get("planned_actions") or [])
        approval_required = _voice_metadata_bool(packet.metadata.get("approval_required"))
        detected = list(packet.metadata.get("detected_risk_cues") or [])
        receipt_material = "|".join(
            [
                normalized,
                "confirmed" if confirmed else "unconfirmed",
                ",".join(str(action.get("tool") or "") for action in planned_actions),
                ",".join(str(action.get("risk") or "") for action in planned_actions),
            ]
        )
        receipt_id = "voice-" + hashlib.sha256(receipt_material.encode("utf-8")).hexdigest()[:12]
        transcript_hash = _transcript_hash(normalized)
        receipt_nonce = _voice_receipt_nonce(receipt_id, transcript_hash)
        route_state = "CONFIRMED_FOR_ROUTING" if confirmed else "WAITING_FOR_CONFIRMATION"
        next_commands = _voice_receipt_next_commands(normalized, approval_required, confirmed)

        action_lines: list[str] = []
        if planned_actions:
            for index, action in enumerate(planned_actions, start=1):
                approval_text = "approval required" if action.get("requires_approval") else "default allowed after confirmation"
                action_lines.append(
                    f"- {index}. {action.get('tool')} [{action.get('toolset')}, {action.get('risk')}]: {approval_text}"
                )
        else:
            action_lines.append("- No tool action was planned; conversational routing can continue after confirmation.")

        lines = [
            "Jarvis voice confirmation receipt:",
            "This is read-only. It creates an auditable spoken-transcript confirmation receipt without recording audio, saving transcripts, executing tools, approving requests, routing actions, or queuing approvals.",
            "",
            f"Receipt id: {receipt_id}",
            f"Route state: {route_state}",
            f"the operator confirmed transcript: {'yes' if confirmed else 'no'}",
            "",
            "Transcript snapshot:",
            f"- {normalized}",
            "",
            "Risk cues:",
        ]
        if detected:
            lines.extend(f"- {item}" for item in detected)
        else:
            lines.append("- none detected by keyword scan")
        lines.extend(
            [
                "",
                "Planned tools:",
                *action_lines,
                "",
                "Routing decision:",
            ]
        )
        if confirmed and approval_required:
            lines.extend(
                [
                    "- Transcript may be routed only far enough to create or surface the normal approval receipt.",
                    "- Risky execution still requires `approval readiness #ID`, `approval packet #ID`, `approval chain proof #ID`, and explicit approval before running.",
                ]
            )
        elif confirmed:
            lines.append("- Transcript may be sent as a normal command under the default read-only/local-safe boundary.")
        else:
            lines.append("- Do not route this transcript yet; show it to the operator and wait for explicit confirmation.")
        lines.extend(
            [
                "",
                "Audit fields for a future real run:",
                "- receipt_id, transcript_hash, receipt_nonce, confirmed, detected_risk_cues, planned_actions, approval_required, route_state, and next safe commands.",
                "- If the transcript is wrong, private, from a bystander, or unclear, discard this receipt and ask the operator to restate it.",
                "- The route gate must bind the exact transcript hash and receipt nonce; stale receipts from another transcript must hold before routing.",
                "",
                "Next safe commands:",
                *[f"- {command}" for command in next_commands],
                "",
                "Safety boundary:",
                "- This receipt does not request microphone access, record audio, start a listener, save transcripts, execute tools, call models, approve requests, write memory, or queue approvals.",
            ]
        )

        return ToolResult(
            "voice_confirmation_receipt",
            True,
            "\n".join(lines),
            _voice_metadata(
                receipt_id=receipt_id,
                transcript=normalized,
                transcript_hash=transcript_hash,
                receipt_nonce=receipt_nonce,
                freshness_nonce=receipt_nonce,
                receipt_nonce_required=True,
                confirmed=confirmed,
                route_state=route_state,
                confirmation_receipt_command=_voice_confirmation_receipt_command(normalized),
                route_blocked_until_confirmation_receipt=not confirmed,
                can_route_after_confirmation_receipt=confirmed,
                detected_risk_cues=detected,
                planned_actions=planned_actions,
                approval_required=approval_required,
                recommended_next_commands=next_commands,
                auditable=True,
                routes_actions=False,
            ),
        )

    return voice_confirmation_receipt


def make_voice_route_gate_packet(get_tool: Callable[[str], Any]):
    confirmation_receipt = make_voice_confirmation_receipt(get_tool)

    def voice_route_gate_packet(args: dict[str, Any]) -> ToolResult:
        transcript = str(args.get("transcript") or "").strip()
        confirmed = str(args.get("confirmed") or args.get("reviewed") or "").strip().lower() in {"true", "yes", "1", "confirmed", "reviewed"}
        supplied_receipt_id = _short_text(args.get("receipt_id") or args.get("confirmation_receipt_id") or "", 120)
        supplied_receipt_nonce = _short_text(args.get("receipt_nonce") or args.get("freshness_nonce") or "", 120)
        if not transcript:
            return _voice_input_failure(
                "voice_route_gate_packet",
                "Give Jarvis a transcript to gate, for example: `voice route gate: run command python3 --version confirmed=true`.",
                reason="missing_transcript",
            )
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            return _voice_input_failure(
                "voice_route_gate_packet",
                f"Transcript is too large ({len(transcript)} chars). Limit is {MAX_TRANSCRIPT_CHARS}.",
                reason="oversized_transcript",
                transcript_chars=len(transcript),
                max_chars=MAX_TRANSCRIPT_CHARS,
            )

        receipt = confirmation_receipt({"transcript": transcript, "confirmed": "true" if confirmed else ""})
        if not receipt.ok:
            return ToolResult("voice_route_gate_packet", False, receipt.output, receipt.metadata)

        metadata = receipt.metadata
        normalized = str(metadata.get("transcript") or transcript).strip()
        planned_actions = list(metadata.get("planned_actions") or [])
        approval_required = _voice_metadata_bool(metadata.get("approval_required"))
        receipt_id = str(metadata.get("receipt_id") or "")
        transcript_hash = str(metadata.get("transcript_hash") or _transcript_hash(normalized))
        receipt_nonce = str(metadata.get("receipt_nonce") or _voice_receipt_nonce(receipt_id, transcript_hash))
        receipt_id_match = not supplied_receipt_id or supplied_receipt_id == receipt_id
        receipt_nonce_match = not supplied_receipt_nonce or supplied_receipt_nonce == receipt_nonce
        receipt_freshness_match = _voice_metadata_all_bool(receipt_id_match, receipt_nonce_match)
        route_state = "VOICE_READY_FOR_PLANNER" if confirmed and receipt_freshness_match else ("VOICE_HELD_FOR_STALE_RECEIPT" if confirmed else "VOICE_HELD_FOR_CONFIRMATION")
        runtime_route = "approval_review" if confirmed and receipt_freshness_match and approval_required else ("planner_ready" if confirmed and receipt_freshness_match else "voice_confirmation_hold")
        can_enter_runtime = _voice_metadata_all_bool(confirmed, receipt_freshness_match)
        can_auto_execute = _voice_metadata_all_bool(confirmed, receipt_freshness_match, not approval_required)
        next_commands = list(_voice_receipt_next_commands(normalized, approval_required, confirmed))
        if confirmed and approval_required:
            next_commands = [
                f"send confirmed transcript: {normalized}",
                "pending approvals",
                "approval readiness latest",
                "approval packet latest",
                "approval chain proof latest",
            ]
        elif confirmed:
            next_commands = [f"send confirmed transcript: {normalized}"]

        action_lines = []
        if planned_actions:
            for index, action in enumerate(planned_actions, start=1):
                action_lines.append(
                    f"- {index}. {action.get('tool')} [{action.get('toolset')}, {action.get('risk')}], approval required: {'yes' if action.get('requires_approval') else 'no'}"
                )
        else:
            action_lines.append("- no tool action planned; conversational routing can continue after confirmation")

        lines = [
            "Jarvis voice route gate packet:",
            "This is read-only. It decides whether a supplied spoken transcript is allowed to enter the normal Jarvis runtime without recording audio, saving transcripts, executing tools, approving requests, or queuing approvals.",
            "",
            "Transcript receipt:",
            f"- receipt id: {receipt_id}",
            f"- transcript hash: `{transcript_hash}`",
            f"- receipt nonce: `{receipt_nonce}`",
            f"- supplied receipt id: {supplied_receipt_id or 'none supplied; generated from current confirmation receipt'}",
            f"- supplied receipt nonce: {supplied_receipt_nonce or 'none supplied; generated from current confirmation receipt'}",
            f"- receipt freshness match: {'yes' if receipt_freshness_match else 'no'}",
            f"- confirmed by the operator: {'yes' if confirmed else 'no'}",
            "",
            "Route gate:",
            f"- route state: {route_state}",
            f"- runtime route after gate: {runtime_route}",
            f"- can enter runtime now: {'yes' if can_enter_runtime else 'no'}",
            f"- can auto-execute now: {'yes' if can_auto_execute else 'no'}",
            f"- approval required after routing: {'yes' if approval_required else 'no'}",
            "",
            "Planned action preview:",
            *action_lines,
            "",
            "Next safe commands:",
            *[f"- {command}" for command in next_commands],
            "",
            "Safety boundary:",
            "- Voice routing requires a confirmed transcript receipt whose transcript matches the message exactly.",
            "- Receipt id and receipt nonce must match the exact transcript hash; a stale confirmation receipt cannot be reused for a changed spoken order.",
            "- A confirmed transcript only unlocks normal routing; shell/code, computer control, personal data, destructive actions, and external side effects still stop at approval readiness, last-look approval packet, approval chain proof, and explicit approval.",
            "- This packet does not request microphone access, record audio, transcribe speech, save transcripts, execute tools, call models, approve requests, write memory, or queue approvals.",
        ]
        if not confirmed:
            lines.append("- Because the transcript is unconfirmed, keep it in the composer and ask the operator to confirm or rerecord.")

        return ToolResult(
            "voice_route_gate_packet",
            True,
            "\n".join(lines),
            _voice_metadata(
                receipt_id=receipt_id,
                supplied_receipt_id=supplied_receipt_id,
                transcript=normalized,
                transcript_hash=transcript_hash,
                receipt_nonce=receipt_nonce,
                supplied_receipt_nonce=supplied_receipt_nonce,
                receipt_nonce_required=True,
                receipt_id_match=receipt_id_match,
                receipt_nonce_match=receipt_nonce_match,
                receipt_freshness_match=receipt_freshness_match,
                confirmation_receipt_command=_voice_confirmation_receipt_command(normalized),
                confirmed=confirmed,
                route_state=route_state,
                runtime_route_after_gate=runtime_route,
                can_enter_runtime=can_enter_runtime,
                can_auto_execute_now=can_auto_execute,
                route_blocked_until_confirmation_receipt=not confirmed,
                can_route_after_confirmation_receipt=confirmed,
                approval_required_after_routing=approval_required,
                planned_actions=planned_actions,
                planned_action_count=len(planned_actions),
                recommended_next_commands=next_commands,
                auditable=True,
                routes_actions=False,
            ),
        )

    return voice_route_gate_packet


def make_voice_route_proof_bundle(get_tool: Callable[[str], Any]):
    confirmation_receipt = make_voice_confirmation_receipt(get_tool)
    route_gate = make_voice_route_gate_packet(get_tool)

    def voice_route_proof_bundle(args: dict[str, Any]) -> ToolResult:
        transcript = str(args.get("transcript") or "").strip()
        confirmed = str(args.get("confirmed") or args.get("reviewed") or "").strip().lower() in {"true", "yes", "1", "confirmed", "reviewed"}
        mode = _short_text(args.get("mode") or "browser-push-to-talk", MAX_VOICE_MODE_CHARS) or "browser-push-to-talk"
        supplied_privacy_receipt_id = _short_text(args.get("privacy_receipt_id") or args.get("capture_receipt_id") or "", 120)
        supplied_receipt_id = _short_text(args.get("receipt_id") or args.get("confirmation_receipt_id") or "", 120)
        supplied_receipt_nonce = _short_text(args.get("receipt_nonce") or args.get("freshness_nonce") or "", 120)
        if not transcript:
            return _voice_input_failure(
                "voice_route_proof_bundle",
                "Give Jarvis a transcript to bundle, for example: `voice route proof bundle: run command python3 --version confirmed=true`.",
                reason="missing_transcript",
            )
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            return _voice_input_failure(
                "voice_route_proof_bundle",
                f"Transcript is too large ({len(transcript)} chars). Limit is {MAX_TRANSCRIPT_CHARS}.",
                reason="oversized_transcript",
                transcript_chars=len(transcript),
                max_chars=MAX_TRANSCRIPT_CHARS,
            )

        privacy = voice_capture_privacy_packet({"mode": mode})
        expected_privacy_receipt_id = str(privacy.metadata.get("privacy_receipt_id") or "")
        privacy_receipt_match = not supplied_privacy_receipt_id or supplied_privacy_receipt_id == expected_privacy_receipt_id
        receipt = confirmation_receipt({"transcript": transcript, "confirmed": "true" if confirmed else ""})
        if not receipt.ok:
            return ToolResult("voice_route_proof_bundle", False, receipt.output, receipt.metadata)
        gate = route_gate(
            {
                "transcript": transcript,
                "confirmed": "true" if confirmed else "",
                "receipt_id": supplied_receipt_id,
                "receipt_nonce": supplied_receipt_nonce,
            }
        )
        if not gate.ok:
            return ToolResult("voice_route_proof_bundle", False, gate.output, gate.metadata)

        receipt_metadata = receipt.metadata
        gate_metadata = gate.metadata
        normalized = str(gate_metadata.get("transcript") or receipt_metadata.get("transcript") or transcript).strip()
        transcript_hash = _transcript_hash(normalized)
        receipt_hash = str(receipt_metadata.get("transcript_hash") or "")
        gate_hash = str(gate_metadata.get("transcript_hash") or "")
        hash_match = receipt_hash == transcript_hash and gate_hash == transcript_hash
        receipt_id = str(receipt_metadata.get("receipt_id") or "")
        receipt_nonce = str(receipt_metadata.get("receipt_nonce") or _voice_receipt_nonce(receipt_id, transcript_hash))
        receipt_id_match = _voice_metadata_bool(gate_metadata.get("receipt_id_match"))
        receipt_nonce_match = _voice_metadata_bool(gate_metadata.get("receipt_nonce_match"))
        receipt_freshness_match = _voice_metadata_bool(gate_metadata.get("receipt_freshness_match"))
        approval_required = _voice_metadata_bool(gate_metadata.get("approval_required_after_routing"))
        can_enter_runtime = _voice_metadata_bool(gate_metadata.get("can_enter_runtime"))
        can_auto_execute = _voice_metadata_bool(gate_metadata.get("can_auto_execute_now"))
        route_ready = bool(confirmed and can_enter_runtime and hash_match and privacy_receipt_match)
        bundle_state = "VOICE_ROUTE_PROOF_READY" if route_ready else "VOICE_ROUTE_PROOF_HELD"
        bundle_missing: list[str] = []
        if not confirmed:
            bundle_missing.append("confirmed transcript receipt")
        if not hash_match:
            bundle_missing.append("matching transcript hash across receipt and route gate")
        if not privacy_receipt_match:
            bundle_missing.append("matching privacy receipt from visible capture boundary")
        if not receipt_freshness_match:
            bundle_missing.append("fresh confirmation receipt id and nonce for this exact transcript")
        if not can_enter_runtime:
            bundle_missing.append("route gate allows runtime entry")

        proof_commands = [
            "voice capture privacy",
            f"voice transcript review: {normalized}",
            f"voice confirmation: {normalized}",
            _voice_confirmation_receipt_command(normalized),
            f"voice route gate: {normalized} confirmed=true",
        ]
        command_intake_contract = _voice_command_intake_contract_metadata(
            transcript=normalized,
            privacy_receipt_id=expected_privacy_receipt_id,
            confirmation_receipt_id=receipt_id,
            confirmation_receipt_nonce=receipt_nonce,
            route_gate_state=str(gate_metadata.get("route_state") or ""),
            route_proof_bundle_state=bundle_state,
            runtime_bridge_state="not_reviewed_yet",
            proof_commands=proof_commands,
        )
        next_commands = list(gate_metadata.get("recommended_next_commands") or _voice_receipt_next_commands(normalized, approval_required, confirmed))

        lines = [
            "Jarvis voice route proof bundle:",
            "This read-only bundle proves the spoken-order chain before a transcript can enter normal Jarvis routing. It does not record audio, read audio files, route actions, execute tools, approve requests, or queue approvals.",
            "",
            "Bundle state:",
            f"- state: {bundle_state}",
            f"- missing proof: {', '.join(bundle_missing) if bundle_missing else 'none'}",
            f"- can enter runtime: {'yes' if can_enter_runtime else 'no'}",
            f"- can auto-execute: {'yes' if can_auto_execute else 'no'}",
            f"- approval required after routing: {'yes' if approval_required else 'no'}",
            "",
            "Privacy proof:",
            f"- privacy receipt id: {expected_privacy_receipt_id}",
            f"- supplied privacy receipt id: {supplied_privacy_receipt_id or 'none supplied; generated from current capture boundary'}",
            f"- privacy receipt match: {'yes' if privacy_receipt_match else 'no'}",
            f"- capture mode: {privacy.metadata.get('mode')}",
            "- visible push-to-talk and transcript confirmation are required before routing",
            "",
            "Transcript identity:",
            f"- transcript: {normalized}",
            f"- transcript hash: `{transcript_hash}`",
            f"- transcript sha256: `{command_intake_contract['transcript_sha256']}`",
            f"- command-intake contract sha256: `{command_intake_contract['voice_command_intake_contract_sha256']}`",
            f"- receipt hash match: {'yes' if receipt_hash == transcript_hash else 'no'}",
            f"- route-gate hash match: {'yes' if gate_hash == transcript_hash else 'no'}",
            "",
            "Confirmation and route proof:",
            f"- confirmation receipt id: {receipt_metadata.get('receipt_id')}",
            f"- confirmation receipt nonce: `{receipt_nonce}`",
            f"- supplied confirmation receipt id: {supplied_receipt_id or 'none supplied; generated from current receipt'}",
            f"- supplied confirmation receipt nonce: {supplied_receipt_nonce or 'none supplied; generated from current receipt'}",
            f"- receipt freshness match: {'yes' if receipt_freshness_match else 'no'}",
            f"- receipt route state: {receipt_metadata.get('route_state')}",
            f"- route gate state: {gate_metadata.get('route_state')}",
            f"- runtime route after gate: {gate_metadata.get('runtime_route_after_gate')}",
            "",
            "Required proof commands:",
            *[f"- `{command}`" for command in proof_commands],
            "",
            "Next safe commands:",
            *[f"- {command}" for command in next_commands],
            "",
            "Safety boundary:",
            "- A ready voice route proof only unlocks normal routing of the exact confirmed transcript.",
            "- Shell/code, computer control, personal data, destructive changes, and external side effects still require approval readiness, last-look approval packet, approval chain proof, explicit approval, audit, verification, recovery, and learning evidence.",
            "- This bundle does not request microphone access, record audio, read audio files, transcribe speech, save transcripts, execute tools, call models, approve requests, write memory, or queue approvals.",
        ]
        return ToolResult(
            "voice_route_proof_bundle",
            True,
            "\n".join(lines),
            _voice_metadata(
                transcript=normalized,
                transcript_hash=transcript_hash,
                **command_intake_contract,
                confirmed=confirmed,
                mode=mode,
                bundle_state=bundle_state,
                bundle_ready=route_ready,
                bundle_missing=bundle_missing,
                bundle_missing_count=len(bundle_missing),
                privacy_receipt_id=expected_privacy_receipt_id,
                supplied_privacy_receipt_id=supplied_privacy_receipt_id,
                privacy_receipt_match=privacy_receipt_match,
                confirmation_receipt_id=receipt_id,
                supplied_confirmation_receipt_id=supplied_receipt_id,
                confirmation_receipt_nonce=receipt_nonce,
                supplied_confirmation_receipt_nonce=supplied_receipt_nonce,
                receipt_nonce_required=True,
                receipt_id_match=receipt_id_match,
                receipt_nonce_match=receipt_nonce_match,
                receipt_freshness_match=receipt_freshness_match,
                confirmation_receipt_command=_voice_confirmation_receipt_command(normalized),
                receipt_route_state=receipt_metadata.get("route_state"),
                route_gate_state=gate_metadata.get("route_state"),
                runtime_route_after_gate=gate_metadata.get("runtime_route_after_gate"),
                receipt_hash_match=receipt_hash == transcript_hash,
                route_gate_hash_match=gate_hash == transcript_hash,
                can_enter_runtime=can_enter_runtime,
                can_auto_execute_now=can_auto_execute,
                route_blocked_until_confirmation_receipt=not confirmed,
                can_route_after_confirmation_receipt=confirmed,
                approval_required_after_routing=approval_required,
                planned_actions=gate_metadata.get("planned_actions") or [],
                planned_action_count=gate_metadata.get("planned_action_count") or 0,
                proof_commands=proof_commands,
                proof_command_count=len(proof_commands),
                recommended_next_commands=next_commands,
                auditable=True,
                routes_actions=False,
            ),
        )

    return voice_route_proof_bundle


def make_voice_runtime_bridge_packet(get_tool: Callable[[str], Any]):
    route_proof_bundle = make_voice_route_proof_bundle(get_tool)

    def voice_runtime_bridge_packet(args: dict[str, Any]) -> ToolResult:
        transcript = str(args.get("transcript") or "").strip()
        confirmed = str(args.get("confirmed") or args.get("reviewed") or "").strip().lower() in {"true", "yes", "1", "confirmed", "reviewed"}
        mode = _short_text(args.get("mode") or "browser-push-to-talk", MAX_VOICE_MODE_CHARS) or "browser-push-to-talk"
        privacy_receipt_id = _short_text(args.get("privacy_receipt_id") or args.get("capture_receipt_id") or "", 120)
        receipt_id = _short_text(args.get("receipt_id") or args.get("confirmation_receipt_id") or "", 120)
        receipt_nonce = _short_text(args.get("receipt_nonce") or args.get("freshness_nonce") or "", 120)
        if not transcript:
            return _voice_input_failure(
                "voice_runtime_bridge_packet",
                "Give Jarvis a transcript to bridge, for example: `voice runtime bridge: run command python3 --version confirmed=true`.",
                reason="missing_transcript",
            )
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            return _voice_input_failure(
                "voice_runtime_bridge_packet",
                f"Transcript is too large ({len(transcript)} chars). Limit is {MAX_TRANSCRIPT_CHARS}.",
                reason="oversized_transcript",
                transcript_chars=len(transcript),
                max_chars=MAX_TRANSCRIPT_CHARS,
            )

        proof = route_proof_bundle(
            {
                "transcript": transcript,
                "confirmed": "true" if confirmed else "",
                "mode": mode,
                "privacy_receipt_id": privacy_receipt_id,
                "receipt_id": receipt_id,
                "receipt_nonce": receipt_nonce,
            }
        )
        if not proof.ok:
            return ToolResult("voice_runtime_bridge_packet", False, proof.output, proof.metadata)

        proof_metadata = proof.metadata
        normalized = str(proof_metadata.get("transcript") or transcript).strip()
        transcript_hash = str(proof_metadata.get("transcript_hash") or _transcript_hash(normalized))
        approval_required = _voice_metadata_bool(proof_metadata.get("approval_required_after_routing"))
        bundle_ready = _voice_metadata_bool(proof_metadata.get("bundle_ready"))
        can_enter_runtime = _voice_metadata_all_bool(
            confirmed,
            bundle_ready,
            proof_metadata.get("can_enter_runtime"),
        )
        bridge_state = "VOICE_READY_FOR_COMMAND_INTAKE" if can_enter_runtime else "VOICE_HELD_BEFORE_COMMAND_INTAKE"
        runtime_entry_mode = "command_intake_only" if can_enter_runtime else "held_for_voice_confirmation"

        proof_commands = list(proof_metadata.get("proof_commands") or [])
        runtime_proof_commands = [
            f"voice route proof bundle: {normalized} confirmed=true",
            f"command intake: {normalized}",
            f"dispatch decision: {normalized}",
            f"execution readiness matrix: {normalized}",
            f"verification packet: {normalized}",
        ]
        if approval_required:
            runtime_proof_commands.extend(["approval readiness latest", "approval packet latest", "approval chain proof latest"])
        all_proof_commands = proof_commands + [command for command in runtime_proof_commands if command not in proof_commands]
        command_intake_contract = _voice_command_intake_contract_metadata(
            transcript=normalized,
            privacy_receipt_id=str(proof_metadata.get("privacy_receipt_id") or ""),
            confirmation_receipt_id=str(proof_metadata.get("confirmation_receipt_id") or ""),
            confirmation_receipt_nonce=str(proof_metadata.get("confirmation_receipt_nonce") or ""),
            route_gate_state=str(proof_metadata.get("route_gate_state") or ""),
            route_proof_bundle_state=str(proof_metadata.get("bundle_state") or ""),
            runtime_bridge_state=bridge_state,
            proof_commands=all_proof_commands,
        )

        bridge_missing: list[str] = []
        if not confirmed:
            bridge_missing.append("confirmed transcript receipt")
        if not bundle_ready:
            bridge_missing.append("ready voice route proof bundle")
        if not _voice_metadata_bool(proof_metadata.get("can_enter_runtime")):
            bridge_missing.append("route proof allows runtime entry")
        if proof_metadata.get("privacy_receipt_match") is not True:
            bridge_missing.append("matching privacy receipt from visible capture boundary")
        if proof_metadata.get("receipt_freshness_match") is not True:
            bridge_missing.append("fresh confirmation receipt id and nonce for this exact transcript")

        next_commands = [f"command intake: {normalized}"] if can_enter_runtime else [
            f"voice transcript review: {normalized}",
            f"voice confirmation: {normalized}",
            _voice_confirmation_receipt_command(normalized),
            f"voice route proof bundle: {normalized} confirmed=true",
        ]
        if can_enter_runtime and approval_required:
            next_commands.extend(["dispatch decision: " + normalized, "approval readiness latest", "approval packet latest", "approval chain proof latest"])
        elif can_enter_runtime:
            next_commands.extend(["dispatch decision: " + normalized, "execution readiness matrix: " + normalized, "verification packet: " + normalized])

        planned_actions = list(proof_metadata.get("planned_actions") or [])
        action_lines: list[str] = []
        if planned_actions:
            for index, action in enumerate(planned_actions, start=1):
                action_lines.append(
                    f"- {index}. {action.get('tool')} [{action.get('toolset')}, {action.get('risk')}], approval required: {'yes' if action.get('requires_approval') else 'no'}"
                )
        else:
            action_lines.append("- no tool action planned; command intake may become chat or safe-tool routing after proof review")

        lines = [
            "Jarvis voice runtime bridge packet:",
            "This is read-only. It bridges an exact confirmed spoken transcript into Jarvis command-intake proof only; it does not route actions, execute tools, approve requests, queue approvals, save transcripts, write memory, record audio, or request microphone access.",
            "",
            "Bridge state:",
            f"- state: {bridge_state}",
            f"- runtime entry mode: {runtime_entry_mode}",
            f"- missing proof: {', '.join(bridge_missing) if bridge_missing else 'none'}",
            f"- can enter runtime as command intake: {'yes' if can_enter_runtime else 'no'}",
            "- can auto-execute now: no",
            f"- approval required after bridge: {'yes' if approval_required else 'no'}",
            "",
            "Voice proof carried forward:",
            f"- transcript: {normalized}",
            f"- transcript hash: `{transcript_hash}`",
            f"- transcript sha256: `{command_intake_contract['transcript_sha256']}`",
            f"- command-intake contract sha256: `{command_intake_contract['voice_command_intake_contract_sha256']}`",
            f"- route proof bundle state: {proof_metadata.get('bundle_state')}",
            f"- confirmation receipt id: {proof_metadata.get('confirmation_receipt_id')}",
            f"- confirmation receipt nonce: `{proof_metadata.get('confirmation_receipt_nonce')}`",
            f"- receipt freshness match: {'yes' if _voice_metadata_bool(proof_metadata.get('receipt_freshness_match')) else 'no'}",
            f"- route gate state: {proof_metadata.get('route_gate_state')}",
            f"- privacy receipt id: {proof_metadata.get('privacy_receipt_id')}",
            "",
            "Command-intake bridge:",
            f"- command intake: `{normalized}`",
            "- dispatch decision: required after intake and before any action route",
            "- execution readiness matrix: required before trusting auto-safe or approval-gated execution",
            "- verification packet: required before any completion claim",
            "- risk preflight: required for risky wording or any planned action beyond read-only/local-safe",
            "",
            "Planned action preview:",
            *action_lines,
            "",
            "Required proof commands:",
            *[f"- `{command}`" for command in all_proof_commands],
            "",
            "Next safe commands:",
            *[f"- {command}" for command in next_commands],
            "",
            "Safety boundary:",
            "- The bridge never sends the transcript as an executable tool action.",
            "- The exact transcript can only enter the normal command-intake lane, where ToolRegistry, PermissionPolicy, dispatch decision, execution readiness matrix, approval gates, audit, recovery, and verification still apply.",
            "- Shell/code, computer control, personal data, destructive changes, and external side effects still require approval readiness, last-look approval packet, approval chain proof, and explicit approval before real execution.",
        ]

        return ToolResult(
            "voice_runtime_bridge_packet",
            True,
            "\n".join(lines),
            _voice_metadata(
                transcript=normalized,
                transcript_hash=transcript_hash,
                **command_intake_contract,
                confirmed=confirmed,
                mode=mode,
                bridge_state=bridge_state,
                runtime_entry_mode=runtime_entry_mode,
                bridge_missing=bridge_missing,
                bridge_missing_count=len(bridge_missing),
                can_enter_runtime=can_enter_runtime,
                can_auto_execute_now=False,
                can_emit_executable_tool_action=False,
                route_blocked_until_confirmation_receipt=not confirmed,
                can_route_after_confirmation_receipt=confirmed,
                approval_required_after_bridge=approval_required,
                command_intake_required=True,
                dispatch_decision_required=True,
                execution_readiness_matrix_required=True,
                verification_packet_required=True,
                risk_preflight_required=approval_required,
                route_proof_bundle_state=proof_metadata.get("bundle_state"),
                route_proof_bundle_ready=bundle_ready,
                confirmation_receipt_id=proof_metadata.get("confirmation_receipt_id"),
                confirmation_receipt_command=_voice_confirmation_receipt_command(normalized),
                route_gate_state=proof_metadata.get("route_gate_state"),
                privacy_receipt_id=proof_metadata.get("privacy_receipt_id"),
                supplied_privacy_receipt_id=proof_metadata.get("supplied_privacy_receipt_id") or "",
                privacy_receipt_match=proof_metadata.get("privacy_receipt_match"),
                confirmation_receipt_nonce=proof_metadata.get("confirmation_receipt_nonce"),
                supplied_confirmation_receipt_id=proof_metadata.get("supplied_confirmation_receipt_id") or "",
                supplied_confirmation_receipt_nonce=proof_metadata.get("supplied_confirmation_receipt_nonce") or "",
                receipt_nonce_required=True,
                receipt_id_match=proof_metadata.get("receipt_id_match"),
                receipt_nonce_match=proof_metadata.get("receipt_nonce_match"),
                receipt_freshness_match=proof_metadata.get("receipt_freshness_match"),
                planned_actions=planned_actions,
                planned_action_count=len(planned_actions),
                proof_commands=all_proof_commands,
                proof_command_count=len(all_proof_commands),
                recommended_next_commands=next_commands,
                auditable=True,
                routes_actions=False,
            ),
        )

    return voice_runtime_bridge_packet


def make_voice_confirmation_audit_ledger(get_tool: Callable[[str], Any]):
    confirmation_receipt = make_voice_confirmation_receipt(get_tool)
    route_gate = make_voice_route_gate_packet(get_tool)
    route_proof_bundle = make_voice_route_proof_bundle(get_tool)
    runtime_bridge = make_voice_runtime_bridge_packet(get_tool)

    def voice_confirmation_audit_ledger(args: dict[str, Any]) -> ToolResult:
        transcript = str(args.get("transcript") or "").strip()
        confirmed = str(args.get("confirmed") or args.get("reviewed") or "").strip().lower() in {"true", "yes", "1", "confirmed", "reviewed"}
        mode = _short_text(args.get("mode") or "browser-push-to-talk", MAX_VOICE_MODE_CHARS) or "browser-push-to-talk"
        privacy_receipt_id = _short_text(args.get("privacy_receipt_id") or args.get("capture_receipt_id") or "", 120)
        receipt_id = _short_text(args.get("receipt_id") or args.get("confirmation_receipt_id") or "", 120)
        receipt_nonce = _short_text(args.get("receipt_nonce") or args.get("freshness_nonce") or "", 120)
        if not transcript:
            return _voice_input_failure(
                "voice_confirmation_audit_ledger",
                "Give Jarvis a transcript and supplied receipt to audit, for example: `voice confirmation audit ledger: run command python3 --version; confirmed=true; privacy_receipt_id=<receipt>; receipt_id=<receipt>; receipt_nonce=<nonce>`.",
                reason="missing_transcript",
            )
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            return _voice_input_failure(
                "voice_confirmation_audit_ledger",
                f"Transcript is too large ({len(transcript)} chars). Limit is {MAX_TRANSCRIPT_CHARS}.",
                reason="oversized_transcript",
                transcript_chars=len(transcript),
                max_chars=MAX_TRANSCRIPT_CHARS,
            )

        common_args = {
            "transcript": transcript,
            "confirmed": "true" if confirmed else "",
            "mode": mode,
            "privacy_receipt_id": privacy_receipt_id,
            "receipt_id": receipt_id,
            "receipt_nonce": receipt_nonce,
        }
        receipt = confirmation_receipt(common_args)
        if not receipt.ok:
            return ToolResult("voice_confirmation_audit_ledger", False, receipt.output, receipt.metadata)
        gate = route_gate(common_args)
        if not gate.ok:
            return ToolResult("voice_confirmation_audit_ledger", False, gate.output, gate.metadata)
        bundle = route_proof_bundle(common_args)
        if not bundle.ok:
            return ToolResult("voice_confirmation_audit_ledger", False, bundle.output, bundle.metadata)
        bridge = runtime_bridge(common_args)
        if not bridge.ok:
            return ToolResult("voice_confirmation_audit_ledger", False, bridge.output, bridge.metadata)

        receipt_meta = receipt.metadata
        gate_meta = gate.metadata
        bundle_meta = bundle.metadata
        bridge_meta = bridge.metadata
        command_intake_contract = _voice_command_intake_contract_from_metadata(bridge_meta)
        normalized = str(bridge_meta.get("transcript") or bundle_meta.get("transcript") or receipt_meta.get("transcript") or transcript).strip()
        transcript_hash = _transcript_hash(normalized)
        expected_receipt_id = str(receipt_meta.get("receipt_id") or "")
        expected_receipt_nonce = str(receipt_meta.get("receipt_nonce") or "")
        supplied_receipt_id_match = bool(receipt_id and receipt_id == expected_receipt_id)
        supplied_receipt_nonce_match = bool(receipt_nonce and receipt_nonce == expected_receipt_nonce)
        supplied_privacy_receipt_match = bool(privacy_receipt_id and bundle_meta.get("privacy_receipt_match") is True and bridge_meta.get("privacy_receipt_match") is True)
        hash_match = (
            str(receipt_meta.get("transcript_hash") or "") == transcript_hash
            and str(gate_meta.get("transcript_hash") or "") == transcript_hash
            and str(bundle_meta.get("transcript_hash") or "") == transcript_hash
            and str(bridge_meta.get("transcript_hash") or "") == transcript_hash
        )
        bridge_ready = bridge_meta.get("bridge_state") == "VOICE_READY_FOR_COMMAND_INTAKE"
        ledger_ready = bool(
            confirmed
            and supplied_receipt_id_match
            and supplied_receipt_nonce_match
            and supplied_privacy_receipt_match
            and hash_match
            and gate_meta.get("can_enter_runtime")
            and bundle_meta.get("bundle_ready")
            and bridge_ready
            and command_intake_contract.get("voice_command_intake_contract_ready") is True
            and bridge_meta.get("can_emit_executable_tool_action") is False
            and bridge_meta.get("can_auto_execute_now") is False
        )
        ledger_state = "VOICE_CONFIRMATION_AUDIT_LEDGER_READY" if ledger_ready else "VOICE_CONFIRMATION_AUDIT_LEDGER_HELD"
        missing: list[str] = []
        if not confirmed:
            missing.append("explicit confirmed=true")
        if not privacy_receipt_id:
            missing.append("supplied privacy receipt id from visible capture boundary")
        elif not supplied_privacy_receipt_match:
            missing.append("matching supplied privacy receipt")
        if not receipt_id:
            missing.append("supplied confirmation receipt id")
        elif not supplied_receipt_id_match:
            missing.append("matching supplied confirmation receipt id")
        if not receipt_nonce:
            missing.append("supplied confirmation receipt nonce")
        elif not supplied_receipt_nonce_match:
            missing.append("matching supplied confirmation receipt nonce")
        if not hash_match:
            missing.append("matching transcript hash across confirmation artifacts")
        if not gate_meta.get("can_enter_runtime"):
            missing.append("route gate allows runtime entry")
        if not bundle_meta.get("bundle_ready"):
            missing.append("ready voice route proof bundle")
        if not bridge_ready:
            missing.append("runtime bridge ready for command intake")
        if command_intake_contract.get("voice_command_intake_contract_ready") is not True:
            missing.append("ready command-intake contract hash")
        if bridge_meta.get("can_emit_executable_tool_action") is not False or bridge_meta.get("can_auto_execute_now") is not False:
            missing.append("bridge proves no direct executable action")
        missing = list(dict.fromkeys(missing))

        stage_rows = [
            {"stage": "confirmation_receipt", "ready": bool(confirmed and _voice_metadata_bool(receipt_meta.get("can_route_after_confirmation_receipt"))), "state": receipt_meta.get("route_state"), "proof": _voice_confirmation_receipt_command(normalized)},
            {"stage": "supplied_receipt_binding", "ready": bool(supplied_receipt_id_match and supplied_receipt_nonce_match), "state": "BOUND" if supplied_receipt_id_match and supplied_receipt_nonce_match else "HELD", "proof": f"receipt_id={expected_receipt_id} receipt_nonce={expected_receipt_nonce}"},
            {"stage": "privacy_receipt_binding", "ready": supplied_privacy_receipt_match, "state": "BOUND" if supplied_privacy_receipt_match else "HELD", "proof": f"privacy_receipt_id={bundle_meta.get('privacy_receipt_id') or '<missing>'}"},
            {"stage": "route_gate", "ready": _voice_metadata_bool(gate_meta.get("can_enter_runtime")), "state": gate_meta.get("route_state"), "proof": f"voice route gate: {normalized} confirmed=true receipt_id={expected_receipt_id} receipt_nonce={expected_receipt_nonce}"},
            {"stage": "route_proof_bundle", "ready": _voice_metadata_bool(bundle_meta.get("bundle_ready")), "state": bundle_meta.get("bundle_state"), "proof": f"voice route proof bundle: {normalized} confirmed=true receipt_id={expected_receipt_id} receipt_nonce={expected_receipt_nonce}"},
            {"stage": "runtime_bridge", "ready": bridge_ready, "state": bridge_meta.get("bridge_state"), "proof": f"voice runtime bridge: {normalized} confirmed=true receipt_id={expected_receipt_id} receipt_nonce={expected_receipt_nonce}"},
        ]
        required_commands = [
            f"voice capture privacy: {mode}",
            f"voice transcript review: {normalized}",
            f"voice confirmation: {normalized}",
            _voice_confirmation_receipt_command(normalized),
            f"voice confirmation audit ledger: {normalized}; confirmed=true; privacy_receipt_id={privacy_receipt_id or '<privacy_receipt_id>'}; receipt_id={receipt_id or '<receipt_id>'}; receipt_nonce={receipt_nonce or '<receipt_nonce>'}",
            f"voice route gate: {normalized} confirmed=true receipt_id={expected_receipt_id} receipt_nonce={expected_receipt_nonce}",
            f"voice route proof bundle: {normalized} confirmed=true privacy_receipt_id={privacy_receipt_id or '<privacy_receipt_id>'} receipt_id={expected_receipt_id} receipt_nonce={expected_receipt_nonce}",
            f"voice runtime bridge: {normalized} confirmed=true privacy_receipt_id={privacy_receipt_id or '<privacy_receipt_id>'} receipt_id={expected_receipt_id} receipt_nonce={expected_receipt_nonce}",
        ]
        next_command = f"command intake: {normalized}" if ledger_ready else (
            f"voice confirmation receipt: {normalized} confirmed=true"
            if not confirmed or not receipt_id or not receipt_nonce
            else f"voice capture privacy: {mode}"
            if not privacy_receipt_id
            else f"voice confirmation audit ledger: {normalized}; confirmed=true; privacy_receipt_id=<matching>; receipt_id=<matching>; receipt_nonce=<matching>"
        )
        voice_confirmation_audit_token_sha256 = _voice_confirmation_audit_token_sha256(
            transcript_hash=transcript_hash,
            audit_ledger_state=ledger_state,
            privacy_receipt_id=str(bundle_meta.get("privacy_receipt_id") or ""),
            confirmation_receipt_id=expected_receipt_id,
            confirmation_receipt_nonce=expected_receipt_nonce,
            supplied_privacy_receipt_id=privacy_receipt_id,
            supplied_confirmation_receipt_id=receipt_id,
            supplied_confirmation_receipt_nonce=receipt_nonce,
            route_gate_state=str(gate_meta.get("route_state") or ""),
            route_proof_bundle_state=str(bundle_meta.get("bundle_state") or ""),
            runtime_bridge_state=str(bridge_meta.get("bridge_state") or ""),
            ready_for_command_intake_proof=ledger_ready,
            stage_count=len(stage_rows),
            required_command_count=len(required_commands),
        )
        voice_confirmation_audit_token_boundary_rows = _voice_confirmation_audit_token_boundary_rows(
            token_sha256=voice_confirmation_audit_token_sha256,
            source="voice_confirmation_audit_ledger",
        )
        voice_confirmation_audit_token_boundary_ready = _voice_confirmation_audit_token_boundary_ready(
            voice_confirmation_audit_token_boundary_rows,
            token_sha256=voice_confirmation_audit_token_sha256,
            source="voice_confirmation_audit_ledger",
        )

        lines = [
            "Jarvis voice confirmation audit ledger:",
            "This is read-only. It proves a spoken transcript has an externally supplied confirmation receipt id, receipt nonce, and visible-capture privacy receipt before command-intake proof can count.",
            "",
            "Audit state:",
            f"- state: {ledger_state}",
            f"- transcript: {normalized}",
            f"- transcript hash: `{transcript_hash}`",
            f"- confirmed by the operator: {'yes' if confirmed else 'no'}",
            f"- supplied privacy receipt required: yes",
            f"- supplied confirmation receipt id required: yes",
            f"- supplied confirmation receipt nonce required: yes",
            f"- supplied privacy receipt match: {'yes' if supplied_privacy_receipt_match else 'no'}",
            f"- supplied receipt id match: {'yes' if supplied_receipt_id_match else 'no'}",
            f"- supplied receipt nonce match: {'yes' if supplied_receipt_nonce_match else 'no'}",
            f"- route-time generated receipt acceptable as external proof: no",
            f"- ready for command intake proof: {'yes' if ledger_ready else 'no'}",
            f"- command-intake contract sha256: `{command_intake_contract.get('voice_command_intake_contract_sha256') or 'missing'}`",
            "- executable tool action emitted: no",
            "- can auto-execute now: no",
            f"- voice confirmation audit token sha256: `{voice_confirmation_audit_token_sha256}`",
            f"- voice confirmation audit token boundary rows: {len(voice_confirmation_audit_token_boundary_rows)}",
            f"- missing blockers: {', '.join(missing) if missing else 'none'}",
            f"- next command: `{next_command}`",
            "",
            "Audit token boundary:",
            *[
                f"- {row['item']}: {row['status']}; proof only; authorizes action/model/tool/approval/routing/private/external no; reusable next voice review no"
                for row in voice_confirmation_audit_token_boundary_rows
            ],
            "",
            "Ledger stages:",
            *[f"- {row['stage']}: {'ready' if row['ready'] else 'held'} ({row['state']}) via `{row['proof']}`" for row in stage_rows],
            "",
            "Required proof commands:",
            *[f"- `{command}`" for command in required_commands],
            "",
            "Replay boundary:",
            "- This ledger rejects missing receipt ids/nonces even if Jarvis can regenerate a matching preview receipt internally.",
            "- A later spoken order needs a fresh confirmation receipt, fresh nonce, fresh route proof, and fresh command-intake bridge.",
            "- Prior confirmation receipts are proof artifacts only; they are not reusable permission for future voice reviews.",
            "",
            "Safety boundary:",
            "- This ledger does not request microphone access, record audio, read audio files, transcribe speech, save transcripts, route actions, execute tools, emit executable tool calls, approve requests, write memory, or queue approvals.",
        ]

        return ToolResult(
            "voice_confirmation_audit_ledger",
            True,
            "\n".join(lines),
            _voice_metadata(
                transcript=normalized,
                transcript_hash=transcript_hash,
                confirmed=confirmed,
                mode=mode,
                audit_ledger_state=ledger_state,
                audit_ledger_ready=ledger_ready,
                ready_for_command_intake_proof=ledger_ready,
                supplied_privacy_receipt_required=True,
                supplied_confirmation_receipt_id_required=True,
                supplied_confirmation_receipt_nonce_required=True,
                route_time_generated_receipt_acceptable_as_external_proof=False,
                privacy_receipt_id=bundle_meta.get("privacy_receipt_id"),
                supplied_privacy_receipt_id=privacy_receipt_id,
                supplied_privacy_receipt_match=supplied_privacy_receipt_match,
                confirmation_receipt_id=expected_receipt_id,
                supplied_confirmation_receipt_id=receipt_id,
                supplied_confirmation_receipt_id_match=supplied_receipt_id_match,
                confirmation_receipt_nonce=expected_receipt_nonce,
                supplied_confirmation_receipt_nonce=receipt_nonce,
                supplied_confirmation_receipt_nonce_match=supplied_receipt_nonce_match,
                receipt_nonce_required=True,
                receipt_freshness_match=bool(supplied_receipt_id_match and supplied_receipt_nonce_match),
                hash_match=hash_match,
                route_gate_state=gate_meta.get("route_state"),
                route_proof_bundle_state=bundle_meta.get("bundle_state"),
                runtime_bridge_state=bridge_meta.get("bridge_state"),
                can_enter_runtime=bridge_meta.get("can_enter_runtime") if ledger_ready else False,
                can_auto_execute_now=False,
                can_emit_executable_tool_action=False,
                executable_tool_action_emitted=False,
                action_allowed_now=False,
                voice_confirmation_audit_token_sha256=voice_confirmation_audit_token_sha256,
                voice_confirmation_audit_token_present=_looks_like_sha256(voice_confirmation_audit_token_sha256),
                voice_confirmation_audit_token_boundary_rows=voice_confirmation_audit_token_boundary_rows,
                voice_confirmation_audit_token_boundary_row_count=len(voice_confirmation_audit_token_boundary_rows),
                voice_confirmation_audit_token_boundary_ready=voice_confirmation_audit_token_boundary_ready,
                voice_confirmation_audit_token_authorizes_action_now=False,
                voice_confirmation_audit_token_authorizes_model_call=False,
                voice_confirmation_audit_token_authorizes_tool_execution=False,
                voice_confirmation_audit_token_authorizes_approval=False,
                voice_confirmation_audit_token_authorizes_routing=False,
                voice_confirmation_audit_token_authorizes_transcript_mutation=False,
                voice_confirmation_audit_token_authorizes_personal_data_read=False,
                voice_confirmation_audit_token_authorizes_external_side_effect=False,
                voice_confirmation_audit_token_authorizes_receipt_reuse=False,
                voice_confirmation_audit_token_reusable_for_next_voice_review=False,
                **command_intake_contract,
                command_intake_required=True,
                dispatch_decision_required=True,
                execution_readiness_matrix_required=True,
                verification_packet_required=True,
                next_review_requires_fresh_confirmation_receipt=True,
                next_review_requires_fresh_receipt_nonce=True,
                previous_confirmation_receipt_reusable_for_next_voice_review=False,
                previous_transcript_hash_reusable_for_next_voice_review=False,
                missing_blockers=missing,
                missing_blocker_count=len(missing),
                stage_rows=stage_rows,
                stage_count=len(stage_rows),
                required_commands=required_commands,
                required_command_count=len(required_commands),
                next_command=next_command,
                receipt_metadata=receipt_meta,
                route_gate_metadata=gate_meta,
                route_proof_bundle_metadata=bundle_meta,
                runtime_bridge_metadata=bridge_meta,
                auditable=True,
                routes_actions=False,
            ),
        )

    return voice_confirmation_audit_ledger


def make_voice_command_cockpit(get_tool: Callable[[str], Any]):
    confirmation_receipt = make_voice_confirmation_receipt(get_tool)
    route_gate = make_voice_route_gate_packet(get_tool)
    route_proof_bundle = make_voice_route_proof_bundle(get_tool)
    runtime_bridge = make_voice_runtime_bridge_packet(get_tool)

    def voice_command_cockpit(args: dict[str, Any]) -> ToolResult:
        transcript = str(args.get("transcript") or "").strip()
        confirmed = str(args.get("confirmed") or args.get("reviewed") or "").strip().lower() in {"true", "yes", "1", "confirmed", "reviewed"}
        mode = _short_text(args.get("mode") or "browser-push-to-talk", MAX_VOICE_MODE_CHARS) or "browser-push-to-talk"
        privacy_receipt_id = _short_text(args.get("privacy_receipt_id") or args.get("capture_receipt_id") or "", 120)
        receipt_id = _short_text(args.get("receipt_id") or args.get("confirmation_receipt_id") or "", 120)
        receipt_nonce = _short_text(args.get("receipt_nonce") or args.get("freshness_nonce") or "", 120)
        if not transcript:
            return _voice_input_failure(
                "voice_command_cockpit",
                "Give Jarvis a transcript to inspect, for example: `voice command cockpit: run command python3 --version confirmed=true`.",
                reason="missing_transcript",
            )
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            return _voice_input_failure(
                "voice_command_cockpit",
                f"Transcript is too large ({len(transcript)} chars). Limit is {MAX_TRANSCRIPT_CHARS}.",
                reason="oversized_transcript",
                transcript_chars=len(transcript),
                max_chars=MAX_TRANSCRIPT_CHARS,
            )

        common_args = {
            "transcript": transcript,
            "confirmed": "true" if confirmed else "",
            "mode": mode,
            "privacy_receipt_id": privacy_receipt_id,
            "receipt_id": receipt_id,
            "receipt_nonce": receipt_nonce,
        }
        receipt = confirmation_receipt(common_args)
        if not receipt.ok:
            return ToolResult("voice_command_cockpit", False, receipt.output, receipt.metadata)
        gate = route_gate(common_args)
        if not gate.ok:
            return ToolResult("voice_command_cockpit", False, gate.output, gate.metadata)
        bundle = route_proof_bundle(common_args)
        if not bundle.ok:
            return ToolResult("voice_command_cockpit", False, bundle.output, bundle.metadata)
        bridge = runtime_bridge(common_args)
        if not bridge.ok:
            return ToolResult("voice_command_cockpit", False, bridge.output, bridge.metadata)

        receipt_meta = receipt.metadata
        gate_meta = gate.metadata
        bundle_meta = bundle.metadata
        bridge_meta = bridge.metadata
        command_intake_contract = _voice_command_intake_contract_from_metadata(bridge_meta)
        normalized = str(bridge_meta.get("transcript") or bundle_meta.get("transcript") or receipt_meta.get("transcript") or transcript).strip()
        transcript_hash = _transcript_hash(normalized)
        hash_match = (
            str(receipt_meta.get("transcript_hash") or "") == transcript_hash
            and str(gate_meta.get("transcript_hash") or "") == transcript_hash
            and str(bundle_meta.get("transcript_hash") or "") == transcript_hash
            and str(bridge_meta.get("transcript_hash") or "") == transcript_hash
        )
        cockpit_ready = bool(
            confirmed
            and receipt_meta.get("can_route_after_confirmation_receipt")
            and gate_meta.get("can_enter_runtime")
            and bundle_meta.get("bundle_ready")
            and bridge_meta.get("bridge_state") == "VOICE_READY_FOR_COMMAND_INTAKE"
            and hash_match
            and bundle_meta.get("privacy_receipt_match") is True
            and bridge_meta.get("privacy_receipt_match") is True
            and bundle_meta.get("receipt_freshness_match") is True
            and bridge_meta.get("receipt_freshness_match") is True
            and command_intake_contract.get("voice_command_intake_contract_ready") is True
            and bridge_meta.get("can_emit_executable_tool_action") is False
        )
        cockpit_state = "VOICE_COMMAND_COCKPIT_READY_FOR_COMMAND_INTAKE" if cockpit_ready else "VOICE_COMMAND_COCKPIT_HELD"
        missing: list[str] = []
        if not confirmed:
            missing.append("confirmed transcript receipt")
        if not hash_match:
            missing.append("matching transcript hash across voice proof artifacts")
        if not receipt_meta.get("can_route_after_confirmation_receipt"):
            missing.append("receipt allows routing")
        if not gate_meta.get("can_enter_runtime"):
            missing.append("route gate allows runtime entry")
        if not bundle_meta.get("bundle_ready"):
            missing.append("ready voice route proof bundle")
        if bundle_meta.get("privacy_receipt_match") is not True or bridge_meta.get("privacy_receipt_match") is not True:
            missing.append("matching privacy receipt from visible capture boundary")
        if bundle_meta.get("receipt_freshness_match") is not True or bridge_meta.get("receipt_freshness_match") is not True:
            missing.append("fresh confirmation receipt id and nonce for this exact transcript")
        if bridge_meta.get("bridge_state") != "VOICE_READY_FOR_COMMAND_INTAKE":
            missing.append("runtime bridge ready for command intake")
        if command_intake_contract.get("voice_command_intake_contract_ready") is not True:
            missing.append("ready command-intake contract hash")
        if bridge_meta.get("can_emit_executable_tool_action") is not False:
            missing.append("runtime bridge blocks executable tool emission")
        missing = list(dict.fromkeys(missing))

        bridge_commands = list(bridge_meta.get("proof_commands") or []) if cockpit_ready else []
        required_commands = list(dict.fromkeys(
            list(bundle_meta.get("proof_commands") or [])
            + bridge_commands
            + [f"voice command cockpit: {normalized} confirmed=true"]
        ))
        next_command = f"command intake: {normalized}" if cockpit_ready else (
            _voice_confirmation_receipt_command(normalized)
            if not confirmed
            else f"voice route proof bundle: {normalized} confirmed=true"
            if not bundle_meta.get("bundle_ready")
            else f"voice runtime bridge: {normalized} confirmed=true"
        )

        lines = [
            "Jarvis voice command cockpit:",
            "This is read-only. It consolidates transcript receipt, route gate, route proof bundle, and runtime bridge before a spoken command can enter command intake; it does not record audio, save transcripts, route actions, execute tools, approve requests, or queue approvals.",
            "",
            "Cockpit state:",
            f"- state: {cockpit_state}",
            f"- transcript: {normalized}",
            f"- transcript hash: `{transcript_hash}`",
            f"- confirmed by the operator: {'yes' if confirmed else 'no'}",
            f"- receipt route state: {receipt_meta.get('route_state')}",
            f"- route gate state: {gate_meta.get('route_state')}",
            f"- route proof bundle state: {bundle_meta.get('bundle_state')}",
            f"- runtime bridge state: {bridge_meta.get('bridge_state')}",
            f"- runtime entry mode: {bridge_meta.get('runtime_entry_mode')}",
            f"- hash match across artifacts: {'yes' if hash_match else 'no'}",
            f"- receipt freshness match: {'yes' if bundle_meta.get('receipt_freshness_match') and bridge_meta.get('receipt_freshness_match') else 'no'}",
            f"- ready for command intake: {'yes' if cockpit_ready else 'no'}",
            f"- command-intake contract sha256: `{command_intake_contract.get('voice_command_intake_contract_sha256') or 'missing'}`",
            "- can auto-execute now: no",
            f"- approval required after bridge: {'yes' if bridge_meta.get('approval_required_after_bridge') else 'no'}",
            f"- missing blockers: {', '.join(missing) if missing else 'none'}",
            "",
            "Required proof chain:",
            f"- next command: `{next_command}`",
            f"- command count: {len(required_commands)}",
            *[f"- `{command}`" for command in required_commands],
            "",
            "Safety boundary:",
            "- A ready voice cockpit only permits command-intake review of the exact confirmed transcript.",
            "- It never emits an executable tool action; dispatch decision, execution readiness matrix, approval gates, audit, recovery, and verification still apply.",
            "- Shell/code, computer control, personal data, destructive changes, and external side effects remain approval-gated.",
        ]
        return ToolResult(
            "voice_command_cockpit",
            True,
            "\n".join(lines),
            _voice_metadata(
                transcript=normalized,
                transcript_hash=transcript_hash,
                confirmation_receipt_command=_voice_confirmation_receipt_command(normalized),
                confirmed=confirmed,
                mode=mode,
                cockpit_state=cockpit_state,
                ready_for_command_intake=cockpit_ready,
                hash_match=hash_match,
                receipt_id=receipt_meta.get("receipt_id"),
                receipt_route_state=receipt_meta.get("route_state"),
                route_gate_state=gate_meta.get("route_state"),
                route_proof_bundle_state=bundle_meta.get("bundle_state"),
                runtime_bridge_state=bridge_meta.get("bridge_state"),
                runtime_entry_mode=bridge_meta.get("runtime_entry_mode"),
                privacy_receipt_id=bundle_meta.get("privacy_receipt_id"),
                supplied_privacy_receipt_id=bundle_meta.get("supplied_privacy_receipt_id") or "",
                privacy_receipt_match=bool(_voice_metadata_bool(bundle_meta.get("privacy_receipt_match")) and _voice_metadata_bool(bridge_meta.get("privacy_receipt_match"))),
                confirmation_receipt_nonce=bundle_meta.get("confirmation_receipt_nonce"),
                supplied_confirmation_receipt_id=bundle_meta.get("supplied_confirmation_receipt_id") or "",
                supplied_confirmation_receipt_nonce=bundle_meta.get("supplied_confirmation_receipt_nonce") or "",
                receipt_nonce_required=True,
                receipt_id_match=bool(_voice_metadata_bool(bundle_meta.get("receipt_id_match")) and _voice_metadata_bool(bridge_meta.get("receipt_id_match"))),
                receipt_nonce_match=bool(_voice_metadata_bool(bundle_meta.get("receipt_nonce_match")) and _voice_metadata_bool(bridge_meta.get("receipt_nonce_match"))),
                receipt_freshness_match=bool(_voice_metadata_bool(bundle_meta.get("receipt_freshness_match")) and _voice_metadata_bool(bridge_meta.get("receipt_freshness_match"))),
                can_enter_runtime=bridge_meta.get("can_enter_runtime"),
                can_auto_execute_now=False,
                can_emit_executable_tool_action=False,
                route_blocked_until_confirmation_receipt=not confirmed,
                can_route_after_confirmation_receipt=confirmed,
                approval_required_after_bridge=bridge_meta.get("approval_required_after_bridge"),
                **command_intake_contract,
                missing_blockers=missing,
                missing_blocker_count=len(missing),
                next_command=next_command,
                required_commands=required_commands,
                required_command_count=len(required_commands),
                receipt_metadata=receipt_meta,
                route_gate_metadata=gate_meta,
                route_proof_bundle_metadata=bundle_meta,
                runtime_bridge_metadata=bridge_meta,
                receipt_output=receipt.output,
                route_gate_output=gate.output,
                route_proof_bundle_output=bundle.output,
                runtime_bridge_output=bridge.output,
                auditable=True,
                routes_actions=False,
            ),
        )

    return voice_command_cockpit


def make_voice_action_audit_packet(get_tool: Callable[[str], Any]):
    command_cockpit = make_voice_command_cockpit(get_tool)

    def voice_action_audit_packet(args: dict[str, Any]) -> ToolResult:
        transcript = str(args.get("transcript") or "").strip()
        confirmed = str(args.get("confirmed") or args.get("reviewed") or "").strip().lower() in {"true", "yes", "1", "confirmed", "reviewed"}
        mode = _short_text(args.get("mode") or "browser-push-to-talk", MAX_VOICE_MODE_CHARS) or "browser-push-to-talk"
        privacy_receipt_id = _short_text(args.get("privacy_receipt_id") or args.get("capture_receipt_id") or "", 120)
        receipt_id = _short_text(args.get("receipt_id") or args.get("confirmation_receipt_id") or "", 120)
        receipt_nonce = _short_text(args.get("receipt_nonce") or args.get("freshness_nonce") or "", 120)
        if not transcript:
            return _voice_input_failure(
                "voice_action_audit_packet",
                "Give Jarvis a transcript to audit, for example: `voice action audit: run command python3 --version confirmed=true`.",
                reason="missing_transcript",
            )
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            return _voice_input_failure(
                "voice_action_audit_packet",
                f"Transcript is too large ({len(transcript)} chars). Limit is {MAX_TRANSCRIPT_CHARS}.",
                reason="oversized_transcript",
                transcript_chars=len(transcript),
                max_chars=MAX_TRANSCRIPT_CHARS,
            )

        cockpit = command_cockpit(
            {
                "transcript": transcript,
                "confirmed": "true" if confirmed else "",
                "mode": mode,
                "privacy_receipt_id": privacy_receipt_id,
                "receipt_id": receipt_id,
                "receipt_nonce": receipt_nonce,
            }
        )
        if not cockpit.ok:
            return ToolResult("voice_action_audit_packet", False, cockpit.output, cockpit.metadata)

        cockpit_meta = cockpit.metadata
        command_intake_contract = _voice_command_intake_contract_from_metadata(cockpit_meta)
        normalized = str(cockpit_meta.get("transcript") or transcript).strip()
        transcript_hash = str(cockpit_meta.get("transcript_hash") or _transcript_hash(normalized))
        approval_required = _voice_metadata_bool(cockpit_meta.get("approval_required_after_bridge"))
        ready_for_command_intake = _voice_metadata_bool(cockpit_meta.get("ready_for_command_intake"))
        hash_match = _voice_metadata_bool(cockpit_meta.get("hash_match"))
        audit_ready = bool(
            confirmed
            and ready_for_command_intake
            and hash_match
            and cockpit_meta.get("privacy_receipt_match") is True
            and cockpit_meta.get("receipt_freshness_match") is True
            and command_intake_contract.get("voice_command_intake_contract_ready") is True
            and cockpit_meta.get("can_emit_executable_tool_action") is False
            and cockpit_meta.get("can_auto_execute_now") is False
        )
        audit_state = "VOICE_ACTION_AUDIT_READY_FOR_OPERATOR_REVIEW" if audit_ready else "VOICE_ACTION_AUDIT_HELD"

        missing: list[str] = []
        if not confirmed:
            missing.append("confirmed transcript receipt")
        if not hash_match:
            missing.append("matching transcript hash across voice artifacts")
        if not ready_for_command_intake:
            missing.append("voice command cockpit ready for command intake")
        if cockpit_meta.get("privacy_receipt_match") is not True:
            missing.append("matching privacy receipt from visible capture boundary")
        if cockpit_meta.get("receipt_freshness_match") is not True:
            missing.append("fresh confirmation receipt id and nonce for this exact transcript")
        if command_intake_contract.get("voice_command_intake_contract_ready") is not True:
            missing.append("ready command-intake contract hash")
        if cockpit_meta.get("can_emit_executable_tool_action") is not False:
            missing.append("voice bridge blocks executable tool emission")
        if cockpit_meta.get("can_auto_execute_now") is not False:
            missing.append("voice cockpit cannot auto-execute")
        missing.extend(str(item) for item in cockpit_meta.get("missing_blockers") or [])
        missing = list(dict.fromkeys(item for item in missing if item))

        required_commands = list(cockpit_meta.get("required_commands") or [])
        action_review_commands = [
            f"voice action audit: {normalized} confirmed=true",
            f"command intake: {normalized}",
            f"dispatch decision: {normalized}",
            f"execution readiness matrix: {normalized}",
            f"verification packet: {normalized}",
            f"completion audit: spoken command {transcript_hash}",
        ]
        if approval_required:
            action_review_commands.extend(["approval readiness latest", "approval packet latest", "approval chain proof latest"])
        all_required_commands = list(dict.fromkeys(required_commands + action_review_commands)) if audit_ready else list(
            dict.fromkeys(required_commands + [f"voice action audit: {normalized} confirmed=true"])
        )
        next_command = f"command intake: {normalized}" if audit_ready else (
            _voice_confirmation_receipt_command(normalized)
            if not confirmed
            else f"voice command cockpit: {normalized} confirmed=true"
        )

        lines = [
            "Jarvis voice action audit packet:",
            "This is read-only. It is the last transcript-confirmation audit before any spoken order can move into command intake; it does not route actions, emit executable tool calls, execute tools, approve requests, queue approvals, save transcripts, record audio, or request microphone access.",
            "",
            "Audit state:",
            f"- state: {audit_state}",
            f"- transcript: {normalized}",
            f"- transcript hash: `{transcript_hash}`",
            f"- confirmed by the operator: {'yes' if confirmed else 'no'}",
            f"- cockpit state: {cockpit_meta.get('cockpit_state')}",
            f"- runtime bridge state: {cockpit_meta.get('runtime_bridge_state')}",
            f"- runtime entry mode: {cockpit_meta.get('runtime_entry_mode')}",
            f"- hash match across artifacts: {'yes' if hash_match else 'no'}",
            f"- receipt freshness match: {'yes' if cockpit_meta.get('receipt_freshness_match') else 'no'}",
            f"- ready for command intake review: {'yes' if audit_ready else 'no'}",
            f"- command-intake contract sha256: `{command_intake_contract.get('voice_command_intake_contract_sha256') or 'missing'}`",
            "- action allowed now: no",
            "- executable tool action emitted: no",
            f"- approval required after command intake: {'yes' if approval_required else 'no'}",
            f"- missing blockers: {', '.join(missing) if missing else 'none'}",
            "",
            "Operator review boundary:",
            f"- next command: `{next_command}`",
            "- command intake, dispatch decision, execution readiness matrix, and verification packet are still required before any tool can act.",
            "- Risky spoken orders still require approval readiness, last-look approval packet, approval chain proof, and explicit approval after normal routing creates an approval request.",
            "",
            "Required proof chain:",
            f"- command count: {len(all_required_commands)}",
            *[f"- `{command}`" for command in all_required_commands],
            "",
            "Safety boundary:",
            "- This packet proves transcript confirmation and the no-direct-execution bridge only.",
            "- It never treats speech as permission to bypass ToolRegistry, PermissionPolicy, approval gates, audit, recovery, learning, or verification.",
        ]

        return ToolResult(
            "voice_action_audit_packet",
            True,
            "\n".join(lines),
            _voice_metadata(
                transcript=normalized,
                transcript_hash=transcript_hash,
                confirmation_receipt_command=_voice_confirmation_receipt_command(normalized),
                confirmed=confirmed,
                mode=mode,
                audit_state=audit_state,
                audit_ready_for_operator_review=audit_ready,
                ready_for_command_intake=ready_for_command_intake,
                route_blocked_until_confirmation_receipt=not confirmed,
                can_route_after_confirmation_receipt=confirmed,
                action_allowed_now=False,
                executable_tool_action_emitted=False,
                can_emit_executable_tool_action=False,
                can_auto_execute_now=False,
                approval_required_after_command_intake=approval_required,
                hash_match=hash_match,
                cockpit_state=cockpit_meta.get("cockpit_state"),
                runtime_bridge_state=cockpit_meta.get("runtime_bridge_state"),
                runtime_entry_mode=cockpit_meta.get("runtime_entry_mode"),
                privacy_receipt_id=cockpit_meta.get("privacy_receipt_id"),
                supplied_privacy_receipt_id=cockpit_meta.get("supplied_privacy_receipt_id") or "",
                privacy_receipt_match=cockpit_meta.get("privacy_receipt_match"),
                confirmation_receipt_nonce=cockpit_meta.get("confirmation_receipt_nonce"),
                supplied_confirmation_receipt_id=cockpit_meta.get("supplied_confirmation_receipt_id") or "",
                supplied_confirmation_receipt_nonce=cockpit_meta.get("supplied_confirmation_receipt_nonce") or "",
                receipt_nonce_required=True,
                receipt_id_match=cockpit_meta.get("receipt_id_match"),
                receipt_nonce_match=cockpit_meta.get("receipt_nonce_match"),
                receipt_freshness_match=cockpit_meta.get("receipt_freshness_match"),
                **command_intake_contract,
                missing_blockers=missing,
                missing_blocker_count=len(missing),
                next_command=next_command,
                required_commands=all_required_commands,
                required_command_count=len(all_required_commands),
                cockpit_metadata=cockpit_meta,
                cockpit_output=cockpit.output,
                auditable=True,
                routes_actions=False,
            ),
        )

    return voice_action_audit_packet


def make_voice_execution_handoff_packet(get_tool: Callable[[str], Any]):
    action_audit = make_voice_action_audit_packet(get_tool)

    def voice_execution_handoff_packet(args: dict[str, Any]) -> ToolResult:
        transcript = str(args.get("transcript") or "").strip()
        confirmed = str(args.get("confirmed") or args.get("reviewed") or "").strip().lower() in {"true", "yes", "1", "confirmed", "reviewed"}
        mode = _short_text(args.get("mode") or "browser-push-to-talk", MAX_VOICE_MODE_CHARS) or "browser-push-to-talk"
        privacy_receipt_id = _short_text(args.get("privacy_receipt_id") or args.get("capture_receipt_id") or "", 120)
        receipt_id = _short_text(args.get("receipt_id") or args.get("confirmation_receipt_id") or "", 120)
        receipt_nonce = _short_text(args.get("receipt_nonce") or args.get("freshness_nonce") or "", 120)
        if not transcript:
            return _voice_input_failure(
                "voice_execution_handoff_packet",
                "Give Jarvis a transcript to hand off, for example: `voice execution handoff: run command python3 --version confirmed=true`.",
                reason="missing_transcript",
            )
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            return _voice_input_failure(
                "voice_execution_handoff_packet",
                f"Transcript is too large ({len(transcript)} chars). Limit is {MAX_TRANSCRIPT_CHARS}.",
                reason="oversized_transcript",
                transcript_chars=len(transcript),
                max_chars=MAX_TRANSCRIPT_CHARS,
            )

        audit = action_audit(
            {
                "transcript": transcript,
                "confirmed": "true" if confirmed else "",
                "mode": mode,
                "privacy_receipt_id": privacy_receipt_id,
                "receipt_id": receipt_id,
                "receipt_nonce": receipt_nonce,
            }
        )
        if not audit.ok:
            return ToolResult("voice_execution_handoff_packet", False, audit.output, audit.metadata)

        audit_meta = audit.metadata
        command_intake_contract = _voice_command_intake_contract_from_metadata(audit_meta)
        normalized = str(audit_meta.get("transcript") or transcript).strip()
        transcript_hash = str(audit_meta.get("transcript_hash") or _transcript_hash(normalized))
        approval_required = _voice_metadata_bool(audit_meta.get("approval_required_after_command_intake"))
        audit_ready = _voice_metadata_bool(audit_meta.get("audit_ready_for_operator_review"))
        handoff_ready = bool(
            confirmed
            and audit_ready
            and audit_meta.get("privacy_receipt_match") is True
            and audit_meta.get("receipt_freshness_match") is True
            and command_intake_contract.get("voice_command_intake_contract_ready") is True
            and audit_meta.get("action_allowed_now") is False
            and audit_meta.get("executable_tool_action_emitted") is False
            and audit_meta.get("can_emit_executable_tool_action") is False
            and audit_meta.get("can_auto_execute_now") is False
        )
        handoff_state = "VOICE_EXECUTION_HANDOFF_READY_FOR_COMMAND_INTAKE_PACKET" if handoff_ready else "VOICE_EXECUTION_HANDOFF_HELD"

        missing: list[str] = []
        if not confirmed:
            missing.append("confirmed transcript receipt")
        if not audit_ready:
            missing.append("ready voice action audit")
        if audit_meta.get("privacy_receipt_match") is not True:
            missing.append("matching privacy receipt from visible capture boundary")
        if audit_meta.get("receipt_freshness_match") is not True:
            missing.append("fresh confirmation receipt id and nonce for this exact transcript")
        if command_intake_contract.get("voice_command_intake_contract_ready") is not True:
            missing.append("ready command-intake contract hash")
        if audit_meta.get("action_allowed_now") is not False:
            missing.append("action remains blocked before command intake")
        if audit_meta.get("executable_tool_action_emitted") is not False:
            missing.append("no executable tool action emitted")
        if audit_meta.get("can_emit_executable_tool_action") is not False:
            missing.append("voice bridge blocks executable tool emission")
        if audit_meta.get("can_auto_execute_now") is not False:
            missing.append("voice handoff cannot auto-execute")
        missing.extend(str(item) for item in audit_meta.get("missing_blockers") or [])
        missing = list(dict.fromkeys(item for item in missing if item))

        intake_commands = [
            f"command intake: {normalized}",
            f"dispatch decision: {normalized}",
            f"execution readiness matrix: {normalized}",
            f"verification packet: {normalized}",
        ]
        if approval_required:
            intake_commands.extend(["approval readiness latest", "approval packet latest", "approval chain proof latest"])
        post_run_commands = [
            f"verification receipt latest: spoken command {transcript_hash}",
            f"execution audit: spoken command {transcript_hash}",
            f"after action learning: spoken command {transcript_hash}",
        ]
        if handoff_ready:
            required_commands = list(dict.fromkeys(list(audit_meta.get("required_commands") or []) + [f"voice execution handoff: {normalized} confirmed=true"] + intake_commands))
        else:
            required_commands = list(dict.fromkeys(list(audit_meta.get("required_commands") or []) + [f"voice execution handoff: {normalized} confirmed=true"]))
        next_command = f"command intake: {normalized}" if handoff_ready else (
            _voice_confirmation_receipt_command(normalized)
            if not confirmed
            else f"voice action audit: {normalized} confirmed=true"
        )

        lines = [
            "Jarvis voice execution handoff packet:",
            "This is read-only. It packages the exact confirmed spoken transcript for command-intake handoff after voice action audit; it does not route actions, emit executable tool calls, execute tools, approve requests, save transcripts, record audio, request microphone access, or queue approvals.",
            "",
            "Handoff state:",
            f"- state: {handoff_state}",
            f"- transcript: {normalized}",
            f"- transcript hash: `{transcript_hash}`",
            f"- confirmed by the operator: {'yes' if confirmed else 'no'}",
            f"- voice action audit state: {audit_meta.get('audit_state')}",
            f"- receipt freshness match: {'yes' if audit_meta.get('receipt_freshness_match') else 'no'}",
            f"- ready for command intake packet: {'yes' if handoff_ready else 'no'}",
            f"- command-intake contract sha256: `{command_intake_contract.get('voice_command_intake_contract_sha256') or 'missing'}`",
            "- action allowed now: no",
            "- executable tool action emitted: no",
            f"- approval required after command intake: {'yes' if approval_required else 'no'}",
            f"- missing blockers: {', '.join(missing) if missing else 'none'}",
            "",
            "Command-intake handoff:",
            f"- next command: `{next_command}`",
            "- command intake, dispatch decision, execution readiness matrix, and verification packet are required before any spoken order can act.",
            "- Risky spoken orders still require approval readiness, last-look approval packet, approval chain proof, and explicit approval after normal routing creates an approval request.",
            "",
            "Required pre-run proof chain:",
            f"- command count: {len(required_commands)}",
            *[f"- `{command}`" for command in required_commands],
            "",
            "Post-run proof required after any approved spoken action:",
            *[f"- `{command}`" for command in post_run_commands],
            "",
            "Safety boundary:",
            "- This handoff never treats speech as permission to bypass ToolRegistry, PermissionPolicy, approval gates, audit, recovery, learning, or verification.",
            "- It proves only that a confirmed transcript is ready for command-intake review; real execution stays downstream and approval-gated where required.",
        ]

        return ToolResult(
            "voice_execution_handoff_packet",
            True,
            "\n".join(lines),
            _voice_metadata(
                transcript=normalized,
                transcript_hash=transcript_hash,
                confirmation_receipt_command=_voice_confirmation_receipt_command(normalized),
                confirmed=confirmed,
                mode=mode,
                handoff_state=handoff_state,
                ready_for_command_intake_packet=handoff_ready,
                audit_state=audit_meta.get("audit_state"),
                audit_ready_for_operator_review=audit_ready,
                ready_for_command_intake=audit_meta.get("ready_for_command_intake"),
                privacy_receipt_id=audit_meta.get("privacy_receipt_id"),
                supplied_privacy_receipt_id=audit_meta.get("supplied_privacy_receipt_id") or "",
                privacy_receipt_match=audit_meta.get("privacy_receipt_match"),
                confirmation_receipt_nonce=audit_meta.get("confirmation_receipt_nonce"),
                supplied_confirmation_receipt_id=audit_meta.get("supplied_confirmation_receipt_id") or "",
                supplied_confirmation_receipt_nonce=audit_meta.get("supplied_confirmation_receipt_nonce") or "",
                receipt_nonce_required=True,
                receipt_id_match=audit_meta.get("receipt_id_match"),
                receipt_nonce_match=audit_meta.get("receipt_nonce_match"),
                receipt_freshness_match=audit_meta.get("receipt_freshness_match"),
                route_blocked_until_confirmation_receipt=not confirmed,
                can_route_after_confirmation_receipt=confirmed,
                action_allowed_now=False,
                executable_tool_action_emitted=False,
                can_emit_executable_tool_action=False,
                can_auto_execute_now=False,
                approval_required_after_command_intake=approval_required,
                command_intake_required=True,
                dispatch_decision_required=True,
                execution_readiness_matrix_required=True,
                verification_packet_required=True,
                **command_intake_contract,
                missing_blockers=missing,
                missing_blocker_count=len(missing),
                next_command=next_command,
                required_commands=required_commands,
                required_command_count=len(required_commands),
                post_run_commands=post_run_commands,
                post_run_command_count=len(post_run_commands),
                action_audit_metadata=audit_meta,
                action_audit_output=audit.output,
                auditable=True,
                routes_actions=False,
            ),
        )

    return voice_execution_handoff_packet


def make_voice_post_run_closure_packet(get_tool: Callable[[str], Any]):
    execution_handoff = make_voice_execution_handoff_packet(get_tool)

    def voice_post_run_closure_packet(args: dict[str, Any]) -> ToolResult:
        transcript = str(args.get("transcript") or "").strip()
        confirmed = str(args.get("confirmed") or args.get("reviewed") or "").strip().lower() in {"true", "yes", "1", "confirmed", "reviewed"}
        mode = _short_text(args.get("mode") or "browser-push-to-talk", MAX_VOICE_MODE_CHARS) or "browser-push-to-talk"
        privacy_receipt_id = _short_text(args.get("privacy_receipt_id") or args.get("capture_receipt_id") or "", 120)
        receipt_id = _short_text(args.get("receipt_id") or args.get("confirmation_receipt_id") or "", 120)
        receipt_nonce = _short_text(args.get("receipt_nonce") or args.get("freshness_nonce") or "", 120)
        if not transcript:
            return _voice_input_failure(
                "voice_post_run_closure_packet",
                "Give Jarvis a transcript to close, for example: `voice post-run closure: run command python3 --version; confirmed=true; verification reviewed; verification_receipt_sha256=<hash>; post health reviewed; execution_health_sha256=<hash>; post audit reviewed; execution_audit_sha256=<hash>; learning reviewed; after_action_learning_sha256=<hash>`.",
                reason="missing_transcript",
            )
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            return _voice_input_failure(
                "voice_post_run_closure_packet",
                f"Transcript is too large ({len(transcript)} chars). Limit is {MAX_TRANSCRIPT_CHARS}.",
                reason="oversized_transcript",
                transcript_chars=len(transcript),
                max_chars=MAX_TRANSCRIPT_CHARS,
            )

        handoff = execution_handoff(
            {
                "transcript": transcript,
                "confirmed": "true" if confirmed else "",
                "mode": mode,
                "privacy_receipt_id": privacy_receipt_id,
                "receipt_id": receipt_id,
                "receipt_nonce": receipt_nonce,
            }
        )
        if not handoff.ok:
            return ToolResult("voice_post_run_closure_packet", False, handoff.output, handoff.metadata)

        handoff_meta = handoff.metadata
        command_intake_contract = _voice_command_intake_contract_from_metadata(handoff_meta)
        normalized = str(handoff_meta.get("transcript") or transcript).strip()
        transcript_hash = str(handoff_meta.get("transcript_hash") or _transcript_hash(normalized))
        handoff_ready = _voice_metadata_bool(handoff_meta.get("ready_for_command_intake_packet"))
        verification_reviewed = _voice_evidence_present(args, "verification") or "verification receipt" in str(args.get("evidence") or "").lower()
        post_health_reviewed = _voice_evidence_present(args, "post_health") or "execution health" in str(args.get("evidence") or "").lower()
        post_audit_reviewed = _voice_evidence_present(args, "post_audit") or "execution audit" in str(args.get("evidence") or "").lower()
        learning_reviewed = _voice_evidence_present(args, "learning") or "after-action learning" in str(args.get("evidence") or "").lower() or "after action learning" in str(args.get("evidence") or "").lower()
        verification_receipt_sha256 = _voice_sha256_arg(args, "verification_receipt_sha256", "verification_sha256", "receipt_sha256")
        execution_health_sha256 = _voice_sha256_arg(args, "execution_health_sha256", "post_health_sha256", "health_sha256")
        execution_audit_sha256 = _voice_sha256_arg(args, "execution_audit_sha256", "post_audit_sha256", "audit_sha256")
        after_action_learning_sha256 = _voice_sha256_arg(args, "after_action_learning_sha256", "learning_sha256")
        verification_receipt_hash_present = _looks_like_sha256(verification_receipt_sha256)
        execution_health_hash_present = _looks_like_sha256(execution_health_sha256)
        execution_audit_hash_present = _looks_like_sha256(execution_audit_sha256)
        after_action_learning_hash_present = _looks_like_sha256(after_action_learning_sha256)
        post_run_artifact_hashes_present = all(
            [
                verification_receipt_hash_present,
                execution_health_hash_present,
                execution_audit_hash_present,
                after_action_learning_hash_present,
            ]
        )

        missing: list[str] = []
        if not confirmed:
            missing.append("confirmed transcript receipt")
        if not handoff_ready:
            missing.append("ready voice execution handoff")
        if handoff_meta.get("privacy_receipt_match") is not True:
            missing.append("matching privacy receipt from visible capture boundary")
        if handoff_meta.get("receipt_freshness_match") is not True:
            missing.append("fresh confirmation receipt id and nonce for this exact transcript")
        if command_intake_contract.get("voice_command_intake_contract_ready") is not True:
            missing.append("ready command-intake contract hash")
        if not verification_reviewed:
            missing.append("post-run verification receipt evidence")
        if not verification_receipt_hash_present:
            missing.append("valid verification receipt sha256")
        if not post_health_reviewed:
            missing.append("post-run execution health evidence")
        if not execution_health_hash_present:
            missing.append("valid execution health sha256")
        if not post_audit_reviewed:
            missing.append("post-run execution audit evidence")
        if not execution_audit_hash_present:
            missing.append("valid execution audit sha256")
        if not learning_reviewed:
            missing.append("after-action learning evidence")
        if not after_action_learning_hash_present:
            missing.append("valid after-action learning sha256")
        missing.extend(str(item) for item in handoff_meta.get("missing_blockers") or [])
        missing = list(dict.fromkeys(item for item in missing if item))

        closure_ready = not missing
        closure_state = "VOICE_POST_RUN_CLOSURE_READY_FOR_NEXT_VOICE_REVIEW" if closure_ready else "VOICE_POST_RUN_CLOSURE_HELD"
        next_voice_review_state = "FRESH_VOICE_REVIEW_UNLOCKED" if closure_ready else "VOICE_REVIEW_HELD_FOR_POST_RUN_PROOF"
        next_review_start_command = "voice transcript review: <next transcript>" if closure_ready else ""
        next_review_requires_fresh_confirmation_receipt = True
        next_review_requires_fresh_receipt_nonce = True
        previous_confirmation_receipt_reusable_for_next_voice_review = False
        previous_transcript_hash_reusable_for_next_voice_review = False
        previous_handoff_reusable_for_next_voice_review = False
        prior_spoken_command_proof_only = True
        required_commands = list(dict.fromkeys(
            list(handoff_meta.get("required_commands") or [])
            + [f"voice execution handoff: {normalized} confirmed=true"]
            + list(handoff_meta.get("post_run_commands") or [])
            + [f"voice post-run closure: {normalized} confirmed=true"]
        ))
        next_command = next_review_start_command if closure_ready else (
            f"voice post-run closure: {normalized} confirmed=true; verification reviewed; verification_receipt_sha256=<hash>; "
            "post health reviewed; execution_health_sha256=<hash>; post audit reviewed; execution_audit_sha256=<hash>; "
            "learning reviewed; after_action_learning_sha256=<hash>"
        )
        voice_post_run_closure_token_sha256 = _voice_post_run_closure_token_sha256(
            transcript_hash=transcript_hash,
            closure_state=closure_state,
            receipt_nonce=str(handoff_meta.get("confirmation_receipt_nonce") or ""),
            handoff_state=str(handoff_meta.get("handoff_state") or ""),
            handoff_ready=handoff_ready,
            command_intake_contract_sha256=str(command_intake_contract.get("voice_command_intake_contract_sha256") or ""),
            verification_receipt_sha256=verification_receipt_sha256,
            execution_health_sha256=execution_health_sha256,
            execution_audit_sha256=execution_audit_sha256,
            after_action_learning_sha256=after_action_learning_sha256,
            next_review_start_command=next_review_start_command,
        )
        voice_post_run_closure_token_boundary_rows = _voice_post_run_closure_token_boundary_rows(
            token_sha256=voice_post_run_closure_token_sha256,
            source="voice_post_run_closure",
        )
        voice_post_run_closure_token_boundary_ready = _voice_post_run_closure_token_boundary_ready(
            voice_post_run_closure_token_boundary_rows,
            token_sha256=voice_post_run_closure_token_sha256,
            source="voice_post_run_closure",
        )

        lines = [
            "Jarvis voice post-run closure packet:",
            "This is read-only. It closes the proof loop after a confirmed spoken command has gone through command intake and any approved execution, before Jarvis treats the next spoken command as review-ready.",
            "",
            "Closure state:",
            f"- state: {closure_state}",
            f"- transcript: {normalized}",
            f"- transcript hash: `{transcript_hash}`",
            f"- voice execution handoff ready: {'yes' if handoff_ready else 'no'}",
            f"- command-intake contract sha256: `{command_intake_contract.get('voice_command_intake_contract_sha256') or 'missing'}`",
            f"- receipt freshness match: {'yes' if handoff_meta.get('receipt_freshness_match') else 'no'}",
            f"- post-run verification reviewed: {'yes' if verification_reviewed else 'no'}",
            f"- post-run execution health reviewed: {'yes' if post_health_reviewed else 'no'}",
            f"- post-run execution audit reviewed: {'yes' if post_audit_reviewed else 'no'}",
            f"- after-action learning reviewed: {'yes' if learning_reviewed else 'no'}",
            f"- verification receipt sha256 present: {'yes' if verification_receipt_hash_present else 'no'}",
            f"- execution health sha256 present: {'yes' if execution_health_hash_present else 'no'}",
            f"- execution audit sha256 present: {'yes' if execution_audit_hash_present else 'no'}",
            f"- after-action learning sha256 present: {'yes' if after_action_learning_hash_present else 'no'}",
            f"- post-run artifact hashes present: {'yes' if post_run_artifact_hashes_present else 'no'}",
            f"- ready for next voice review: {'yes' if closure_ready else 'no'}",
            "- action allowed now: no",
            "- executable tool action emitted: no",
            f"- missing blockers: {', '.join(missing) if missing else 'none'}",
            "",
            "Continuation boundary:",
            f"- next voice review state: {next_voice_review_state}",
            f"- next review start command: `{next_review_start_command or next_command}`",
            "- fresh confirmation receipt required before next spoken order: yes",
            "- fresh receipt nonce required before next spoken order: yes",
            "- previous confirmation receipt reusable for next voice review: no",
            "- previous transcript hash reusable for next voice review: no",
            "- previous execution handoff reusable for next voice review: no",
            "- prior spoken command proof only: yes",
            f"- voice post-run closure token sha256: `{voice_post_run_closure_token_sha256}`",
            f"- voice post-run closure token boundary rows: {len(voice_post_run_closure_token_boundary_rows)}",
            *[
                f"- {row['item']}: {row['status']}; proof only; authorizes action/model/tool/approval/routing/private/external no; reusable next voice review no"
                for row in voice_post_run_closure_token_boundary_rows
            ],
            "",
            "Required closure proof chain:",
            f"- command count: {len(required_commands)}",
            *[f"- `{command}`" for command in required_commands],
            "",
            "Next safe command:",
            f"- `{next_command}`",
            "",
            "Safety boundary:",
            "- This closure does not record audio, save transcripts, route actions, emit executable tool calls, execute tools, approve requests, write memory, or queue approvals.",
            "- It only proves that a spoken command's post-run verification, audit, health, and learning evidence were reviewed before the next spoken command review starts.",
        ]

        return ToolResult(
            "voice_post_run_closure_packet",
            True,
            "\n".join(lines),
            _voice_metadata(
                transcript=normalized,
                transcript_hash=transcript_hash,
                confirmation_receipt_command=_voice_confirmation_receipt_command(normalized),
                confirmed=confirmed,
                mode=mode,
                route_blocked_until_confirmation_receipt=not confirmed,
                can_route_after_confirmation_receipt=confirmed,
                closure_state=closure_state,
                ready_for_next_voice_review=closure_ready,
                next_voice_review_state=next_voice_review_state,
                next_review_start_command=next_review_start_command,
                next_review_requires_fresh_confirmation_receipt=next_review_requires_fresh_confirmation_receipt,
                next_review_requires_fresh_receipt_nonce=next_review_requires_fresh_receipt_nonce,
                previous_confirmation_receipt_reusable_for_next_voice_review=previous_confirmation_receipt_reusable_for_next_voice_review,
                previous_transcript_hash_reusable_for_next_voice_review=previous_transcript_hash_reusable_for_next_voice_review,
                previous_handoff_reusable_for_next_voice_review=previous_handoff_reusable_for_next_voice_review,
                prior_spoken_command_proof_only=prior_spoken_command_proof_only,
                voice_post_run_closure_token_sha256=voice_post_run_closure_token_sha256,
                voice_post_run_closure_token_present=_looks_like_sha256(voice_post_run_closure_token_sha256),
                voice_post_run_closure_token_binds_handoff_state=True,
                voice_post_run_closure_token_binds_handoff_ready=True,
                voice_post_run_closure_token_binds_command_intake_contract=True,
                voice_post_run_closure_token_boundary_rows=voice_post_run_closure_token_boundary_rows,
                voice_post_run_closure_token_boundary_row_count=len(voice_post_run_closure_token_boundary_rows),
                voice_post_run_closure_token_boundary_ready=voice_post_run_closure_token_boundary_ready,
                voice_post_run_closure_token_authorizes_action_now=False,
                voice_post_run_closure_token_authorizes_model_call=False,
                voice_post_run_closure_token_authorizes_tool_execution=False,
                voice_post_run_closure_token_authorizes_approval=False,
                voice_post_run_closure_token_authorizes_routing=False,
                voice_post_run_closure_token_authorizes_transcript_mutation=False,
                voice_post_run_closure_token_authorizes_personal_data_read=False,
                voice_post_run_closure_token_authorizes_external_side_effect=False,
                voice_post_run_closure_token_authorizes_next_voice_review=False,
                voice_post_run_closure_token_authorizes_receipt_reuse=False,
                voice_post_run_closure_token_reusable_for_next_voice_review=False,
                **command_intake_contract,
                handoff_state=handoff_meta.get("handoff_state"),
                handoff_ready=handoff_ready,
                privacy_receipt_id=handoff_meta.get("privacy_receipt_id"),
                supplied_privacy_receipt_id=handoff_meta.get("supplied_privacy_receipt_id") or "",
                privacy_receipt_match=handoff_meta.get("privacy_receipt_match"),
                confirmation_receipt_nonce=handoff_meta.get("confirmation_receipt_nonce"),
                supplied_confirmation_receipt_id=handoff_meta.get("supplied_confirmation_receipt_id") or "",
                supplied_confirmation_receipt_nonce=handoff_meta.get("supplied_confirmation_receipt_nonce") or "",
                receipt_nonce_required=True,
                receipt_id_match=handoff_meta.get("receipt_id_match"),
                receipt_nonce_match=handoff_meta.get("receipt_nonce_match"),
                receipt_freshness_match=handoff_meta.get("receipt_freshness_match"),
                post_run_verification_reviewed=verification_reviewed,
                post_run_execution_health_reviewed=post_health_reviewed,
                post_run_execution_audit_reviewed=post_audit_reviewed,
                after_action_learning_reviewed=learning_reviewed,
                verification_receipt_sha256=verification_receipt_sha256,
                execution_health_sha256=execution_health_sha256,
                execution_audit_sha256=execution_audit_sha256,
                after_action_learning_sha256=after_action_learning_sha256,
                verification_receipt_hash_present=verification_receipt_hash_present,
                execution_health_hash_present=execution_health_hash_present,
                execution_audit_hash_present=execution_audit_hash_present,
                after_action_learning_hash_present=after_action_learning_hash_present,
                post_run_artifact_hashes_present=post_run_artifact_hashes_present,
                post_run_artifact_hashes={
                    "verification_receipt_sha256": verification_receipt_sha256,
                    "execution_health_sha256": execution_health_sha256,
                    "execution_audit_sha256": execution_audit_sha256,
                    "after_action_learning_sha256": after_action_learning_sha256,
                },
                action_allowed_now=False,
                executable_tool_action_emitted=False,
                can_emit_executable_tool_action=False,
                can_auto_execute_now=False,
                missing_blockers=missing,
                missing_blocker_count=len(missing),
                next_command=next_command,
                required_commands=required_commands,
                required_command_count=len(required_commands),
                handoff_metadata=handoff_meta,
                handoff_output=handoff.output,
                auditable=True,
                routes_actions=False,
            ),
        )

    return voice_post_run_closure_packet


def make_voice_cycle_ledger(get_tool: Callable[[str], Any]):
    post_run_closure = make_voice_post_run_closure_packet(get_tool)

    def voice_cycle_ledger(args: dict[str, Any]) -> ToolResult:
        transcript = str(args.get("transcript") or "").strip()
        if not transcript:
            return _voice_input_failure(
                "voice_cycle_ledger",
                "Give Jarvis a transcript to ledger, for example: `voice cycle ledger: run command python3 --version; confirmed=true; verification reviewed; verification_receipt_sha256=<hash>; post health reviewed; execution_health_sha256=<hash>; post audit reviewed; execution_audit_sha256=<hash>; learning reviewed; after_action_learning_sha256=<hash>`.",
                reason="missing_transcript",
            )
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            return _voice_input_failure(
                "voice_cycle_ledger",
                f"Transcript is too large ({len(transcript)} chars). Limit is {MAX_TRANSCRIPT_CHARS}.",
                reason="oversized_transcript",
                transcript_chars=len(transcript),
                max_chars=MAX_TRANSCRIPT_CHARS,
            )

        closure = post_run_closure(args)
        if not closure.ok:
            return ToolResult("voice_cycle_ledger", False, closure.output, closure.metadata)

        closure_meta = closure.metadata
        command_intake_contract = _voice_command_intake_contract_from_metadata(closure_meta)
        handoff_meta = closure_meta.get("handoff_metadata") if isinstance(closure_meta.get("handoff_metadata"), dict) else {}
        action_audit_meta = handoff_meta.get("action_audit_metadata") if isinstance(handoff_meta.get("action_audit_metadata"), dict) else {}
        cockpit_meta = action_audit_meta.get("cockpit_metadata") if isinstance(action_audit_meta.get("cockpit_metadata"), dict) else {}
        receipt_meta = cockpit_meta.get("receipt_metadata") if isinstance(cockpit_meta.get("receipt_metadata"), dict) else {}
        route_gate_meta = cockpit_meta.get("route_gate_metadata") if isinstance(cockpit_meta.get("route_gate_metadata"), dict) else {}
        route_bundle_meta = cockpit_meta.get("route_proof_bundle_metadata") if isinstance(cockpit_meta.get("route_proof_bundle_metadata"), dict) else {}
        bridge_meta = cockpit_meta.get("runtime_bridge_metadata") if isinstance(cockpit_meta.get("runtime_bridge_metadata"), dict) else {}

        normalized = str(closure_meta.get("transcript") or transcript).strip()
        transcript_hash = str(closure_meta.get("transcript_hash") or _transcript_hash(normalized))
        confirmed = _voice_metadata_bool(closure_meta.get("confirmed"))
        privacy_match = closure_meta.get("privacy_receipt_match") is True
        receipt_freshness_match = closure_meta.get("receipt_freshness_match") is True
        post_run_artifact_hashes_present = closure_meta.get("post_run_artifact_hashes_present") is True

        stage_rows = [
            {
                "stage": "privacy_boundary",
                "state": "PRIVACY_RECEIPT_READY" if route_bundle_meta.get("privacy_receipt_match") is True else "PRIVACY_RECEIPT_HELD",
                "ready": route_bundle_meta.get("privacy_receipt_match") is True,
                "proof": route_bundle_meta.get("privacy_receipt_id") or cockpit_meta.get("privacy_receipt_id") or "",
            },
            {
                "stage": "confirmation_receipt",
                "state": receipt_meta.get("route_state") or "CONFIRMATION_RECEIPT_UNKNOWN",
                "ready": receipt_meta.get("can_route_after_confirmation_receipt") is True and confirmed,
                "proof": receipt_meta.get("receipt_id") or cockpit_meta.get("receipt_id") or "",
            },
            {
                "stage": "route_gate",
                "state": route_gate_meta.get("route_state") or "VOICE_ROUTE_GATE_UNKNOWN",
                "ready": route_gate_meta.get("can_enter_runtime") is True,
                "proof": route_gate_meta.get("runtime_route_after_gate") or "",
            },
            {
                "stage": "route_proof_bundle",
                "state": route_bundle_meta.get("bundle_state") or "VOICE_ROUTE_PROOF_UNKNOWN",
                "ready": route_bundle_meta.get("bundle_ready") is True,
                "proof": route_bundle_meta.get("confirmation_receipt_id") or "",
            },
            {
                "stage": "runtime_bridge",
                "state": bridge_meta.get("bridge_state") or "VOICE_RUNTIME_BRIDGE_UNKNOWN",
                "ready": bridge_meta.get("bridge_state") == "VOICE_READY_FOR_COMMAND_INTAKE",
                "proof": bridge_meta.get("runtime_entry_mode") or "",
            },
            {
                "stage": "command_cockpit",
                "state": cockpit_meta.get("cockpit_state") or "VOICE_COMMAND_COCKPIT_UNKNOWN",
                "ready": cockpit_meta.get("ready_for_command_intake") is True,
                "proof": cockpit_meta.get("next_command") or "",
            },
            {
                "stage": "action_audit",
                "state": action_audit_meta.get("audit_state") or "VOICE_ACTION_AUDIT_UNKNOWN",
                "ready": action_audit_meta.get("audit_ready_for_operator_review") is True,
                "proof": action_audit_meta.get("next_command") or "",
            },
            {
                "stage": "execution_handoff",
                "state": handoff_meta.get("handoff_state") or "VOICE_EXECUTION_HANDOFF_UNKNOWN",
                "ready": handoff_meta.get("ready_for_command_intake_packet") is True,
                "proof": handoff_meta.get("next_command") or "",
            },
            {
                "stage": "post_run_closure",
                "state": closure_meta.get("closure_state") or "VOICE_POST_RUN_CLOSURE_UNKNOWN",
                "ready": closure_meta.get("ready_for_next_voice_review") is True and post_run_artifact_hashes_present,
                "proof": closure_meta.get("next_review_start_command") or "",
            },
        ]
        for row in stage_rows:
            row.update(
                {
                    "authorizes_action": False,
                    "authorizes_model_call": False,
                    "authorizes_tool_execution": False,
                    "authorizes_approval": False,
                    "authorizes_routing": False,
                    "authorizes_transcript_mutation": False,
                    "authorizes_personal_data_read": False,
                    "authorizes_external_side_effect": False,
                    "authorizes_next_voice_review": False,
                    "authorizes_receipt_reuse": False,
                    "reusable_for_next_voice_review": False,
                }
            )
        missing = list(closure_meta.get("missing_blockers") or [])
        for row in stage_rows:
            if not row["ready"]:
                missing.append(f"{row['stage']} not ready")
        missing = list(dict.fromkeys(str(item) for item in missing if item))
        cycle_ready = bool(
            closure_meta.get("ready_for_next_voice_review") is True
            and confirmed
            and privacy_match
            and receipt_freshness_match
            and command_intake_contract.get("voice_command_intake_contract_ready") is True
            and post_run_artifact_hashes_present
            and all(row["ready"] for row in stage_rows)
        )
        cycle_state = "VOICE_CYCLE_LEDGER_READY_FOR_FRESH_REVIEW" if cycle_ready else "VOICE_CYCLE_LEDGER_HELD"
        route_bundle_ready = route_bundle_meta.get("bundle_ready") is True
        runtime_bridge_ready = bridge_meta.get("bridge_state") == "VOICE_READY_FOR_COMMAND_INTAKE"
        action_audit_ready = action_audit_meta.get("audit_ready_for_operator_review") is True
        handoff_ready = handoff_meta.get("ready_for_command_intake_packet") is True
        preflight_scorecard_rows = _voice_cycle_preflight_scorecard_rows(
            privacy_match=privacy_match,
            receipt_freshness_match=receipt_freshness_match,
            route_bundle_ready=route_bundle_ready,
            runtime_bridge_ready=runtime_bridge_ready,
            action_audit_ready=action_audit_ready,
            handoff_ready=handoff_ready,
            post_run_artifact_hashes_present=post_run_artifact_hashes_present,
            cycle_ready=cycle_ready,
        )
        preflight_score = sum(int(row["points"]) for row in preflight_scorecard_rows)
        preflight_max_score = sum(int(row["max_points"]) for row in preflight_scorecard_rows)
        preflight_required_rows_ready = all(bool(row["ready"]) for row in preflight_scorecard_rows)
        preflight_scorecard_ready = _voice_cycle_preflight_scorecard_ready(preflight_scorecard_rows)
        voice_cycle_stage_rows_ready = _voice_cycle_stage_rows_ready(stage_rows)
        next_review_start_command = str(closure_meta.get("next_review_start_command") or "")
        required_commands = list(dict.fromkeys(list(closure_meta.get("required_commands") or []) + [f"voice cycle ledger: {normalized} confirmed=true"]))
        next_command = next_review_start_command if cycle_ready else (
            f"voice cycle ledger: {normalized}; confirmed=true; verification reviewed; verification_receipt_sha256=<hash>; "
            "post health reviewed; execution_health_sha256=<hash>; post audit reviewed; execution_audit_sha256=<hash>; "
            "learning reviewed; after_action_learning_sha256=<hash>"
        )
        fresh_review_preflight_queue = [
            "voice capture privacy",
            "voice transcript review: <next transcript>",
            "voice confirmation receipt: <next transcript> confirmed=true",
            "voice confirmation audit ledger: <next transcript>; confirmed=true; privacy_receipt_id=<fresh>; receipt_id=<fresh>; receipt_nonce=<fresh>",
            "voice route proof bundle: <next transcript> confirmed=true",
            "voice command cockpit: <next transcript> confirmed=true",
            "voice execution handoff: <next transcript> confirmed=true",
            f"voice cycle ledger: {normalized} confirmed=true post-run proof reviewed",
        ]
        fresh_review_contract_rows = [
            {
                "item": "visible_capture_privacy_receipt",
                "required": True,
                "status": "fresh_required",
                "proof_only": True,
                "authorizes_action": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_approval": False,
                "authorizes_routing": False,
                "authorizes_transcript_mutation": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "reusable_prior_artifact": False,
            },
            {
                "item": "confirmed_transcript_receipt",
                "required": True,
                "status": "fresh_required",
                "proof_only": True,
                "authorizes_action": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_approval": False,
                "authorizes_routing": False,
                "authorizes_transcript_mutation": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "reusable_prior_artifact": False,
            },
            {
                "item": "receipt_nonce_freshness",
                "required": True,
                "status": "fresh_required",
                "proof_only": True,
                "authorizes_action": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_approval": False,
                "authorizes_routing": False,
                "authorizes_transcript_mutation": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "reusable_prior_artifact": False,
            },
            {
                "item": "route_proof_bundle",
                "required": True,
                "status": "fresh_required",
                "proof_only": True,
                "authorizes_action": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_approval": False,
                "authorizes_routing": False,
                "authorizes_transcript_mutation": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "reusable_prior_artifact": False,
            },
            {
                "item": "command_intake_bridge",
                "required": True,
                "status": "fresh_required",
                "proof_only": True,
                "authorizes_action": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_approval": False,
                "authorizes_routing": False,
                "authorizes_transcript_mutation": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "reusable_prior_artifact": False,
            },
            {
                "item": "execution_handoff_packet",
                "required": True,
                "status": "fresh_required",
                "proof_only": True,
                "authorizes_action": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_approval": False,
                "authorizes_routing": False,
                "authorizes_transcript_mutation": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "reusable_prior_artifact": False,
            },
            {
                "item": "post_run_closure_artifacts",
                "required": True,
                "status": "fresh_required",
                "proof_only": True,
                "authorizes_action": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_approval": False,
                "authorizes_routing": False,
                "authorizes_transcript_mutation": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "reusable_prior_artifact": False,
            },
            {
                "item": "voice_cycle_ledger_review_token",
                "required": True,
                "status": "fresh_required",
                "proof_only": True,
                "authorizes_action": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_approval": False,
                "authorizes_routing": False,
                "authorizes_transcript_mutation": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "reusable_prior_artifact": False,
            },
        ]
        all_prior_artifacts_non_authorizing = all(
            row["proof_only"] is True
            and row["authorizes_action"] is False
            and row["authorizes_model_call"] is False
            and row["authorizes_tool_execution"] is False
            and row["authorizes_approval"] is False
            and row["authorizes_routing"] is False
            and row["authorizes_transcript_mutation"] is False
            and row["authorizes_personal_data_read"] is False
            and row["authorizes_external_side_effect"] is False
            and row["reusable_prior_artifact"] is False
            for row in fresh_review_contract_rows
        )
        voice_post_run_closure_token_boundary_rows = _voice_post_run_closure_token_boundary_rows(
            token_sha256=str(closure_meta.get("voice_post_run_closure_token_sha256") or ""),
            source="voice_cycle_ledger",
        )
        voice_post_run_closure_token_boundary_ready = _voice_post_run_closure_token_boundary_ready(
            voice_post_run_closure_token_boundary_rows,
            token_sha256=str(closure_meta.get("voice_post_run_closure_token_sha256") or ""),
            source="voice_cycle_ledger",
        )
        voice_cycle_ledger_token_sha256 = _voice_cycle_ledger_token_sha256(
            transcript=normalized,
            transcript_hash=transcript_hash,
            cycle_state=cycle_state,
            stage_rows=stage_rows,
            voice_preflight_scorecard_rows=preflight_scorecard_rows,
            fresh_review_preflight_queue=fresh_review_preflight_queue,
            fresh_review_contract_rows=fresh_review_contract_rows,
            required_commands=required_commands,
            command_intake_contract_sha256=str(command_intake_contract.get("voice_command_intake_contract_sha256") or ""),
            voice_post_run_closure_token_sha256=str(closure_meta.get("voice_post_run_closure_token_sha256") or ""),
            verification_receipt_sha256=str(closure_meta.get("verification_receipt_sha256") or ""),
            execution_health_sha256=str(closure_meta.get("execution_health_sha256") or ""),
            execution_audit_sha256=str(closure_meta.get("execution_audit_sha256") or ""),
            after_action_learning_sha256=str(closure_meta.get("after_action_learning_sha256") or ""),
            next_command=next_command,
        )
        voice_cycle_ledger_ready = _voice_cycle_ledger_ready(
            cycle_ready=cycle_ready,
            stage_rows=stage_rows,
            preflight_scorecard_rows=preflight_scorecard_rows,
            fresh_review_contract_rows=fresh_review_contract_rows,
            post_run_artifact_hashes_present=post_run_artifact_hashes_present,
            command_intake_contract_sha256=str(command_intake_contract.get("voice_command_intake_contract_sha256") or ""),
            voice_post_run_closure_token_sha256=str(closure_meta.get("voice_post_run_closure_token_sha256") or ""),
            voice_cycle_ledger_token_sha256=voice_cycle_ledger_token_sha256,
            verification_receipt_sha256=str(closure_meta.get("verification_receipt_sha256") or ""),
            execution_health_sha256=str(closure_meta.get("execution_health_sha256") or ""),
            execution_audit_sha256=str(closure_meta.get("execution_audit_sha256") or ""),
            after_action_learning_sha256=str(closure_meta.get("after_action_learning_sha256") or ""),
        )

        lines = [
            "Jarvis voice cycle ledger:",
            "This is read-only. It binds the full confirmed-speech lifecycle into one prior-command proof ledger before any next spoken command review can start; it does not record audio, save transcripts, route actions, emit executable tool calls, execute tools, approve requests, write memory, or queue approvals.",
            "",
            "Cycle state:",
            f"- state: {cycle_state}",
            f"- transcript: {normalized}",
            f"- transcript hash: `{transcript_hash}`",
            f"- confirmed by the operator: {'yes' if confirmed else 'no'}",
            f"- privacy receipt match: {'yes' if privacy_match else 'no'}",
            f"- receipt freshness match: {'yes' if receipt_freshness_match else 'no'}",
            f"- command-intake contract sha256: `{command_intake_contract.get('voice_command_intake_contract_sha256') or 'missing'}`",
            f"- post-run artifact hashes present: {'yes' if post_run_artifact_hashes_present else 'no'}",
            f"- ready for fresh next voice review: {'yes' if cycle_ready else 'no'}",
            f"- voice cycle ledger contract ready: {'yes' if voice_cycle_ledger_ready else 'no'}",
            f"- preflight score: {preflight_score}/{preflight_max_score}",
            f"- preflight rows ready: {'yes' if preflight_required_rows_ready else 'no'}",
            f"- preflight scorecard ready: {'yes' if preflight_scorecard_ready else 'no'}",
            "- action allowed now: no",
            "- executable tool action emitted: no",
            f"- missing blockers: {', '.join(missing) if missing else 'none'}",
            "",
            "Cycle stages:",
            *[f"- {row['stage']}: {row['state']} ({'ready' if row['ready'] else 'held'}; proof: {row['proof'] or 'none'})" for row in stage_rows],
            "",
            "Measured preflight scorecard:",
            *[
                f"- {row['item']}: {row['points']}/{row['max_points']} ({'ready' if row['ready'] else 'held'}; does not authorize action, model call, tool execution, approval, or routing)"
                for row in preflight_scorecard_rows
            ],
            "",
            "Fresh-review boundary:",
            f"- next review start command: `{next_review_start_command or next_command}`",
            "- fresh confirmation receipt required before next spoken order: yes",
            "- fresh receipt nonce required before next spoken order: yes",
            "- previous confirmation receipt reusable for next voice review: no",
            "- previous transcript hash reusable for next voice review: no",
            "- previous execution handoff reusable for next voice review: no",
            "- prior spoken command proof only: yes",
            f"- carried post-run closure token sha256: `{closure_meta.get('voice_post_run_closure_token_sha256') or 'missing'}`",
            f"- carried post-run closure token boundary rows: {len(voice_post_run_closure_token_boundary_rows)}",
            *[
                f"- {row['item']}: {row['status']}; proof only; authorizes action/model/tool/approval/routing/private/external no; reusable next voice review no"
                for row in voice_post_run_closure_token_boundary_rows
            ],
            "- all prior voice artifacts non-authorizing: yes",
            "- next voice review requires full preflight: yes",
            f"- voice cycle ledger token sha256: `{voice_cycle_ledger_token_sha256}`",
            "- voice cycle ledger token authorizes action/model/tool/approval/routing/private/external/next-review no",
            "- voice cycle ledger token reusable for next voice review: no",
            "",
            "Fresh-review preflight queue:",
            f"- command count: {len(fresh_review_preflight_queue)}",
            *[f"- `{command}`" for command in fresh_review_preflight_queue],
            "",
            "Fresh-review contract rows:",
            f"- row count: {len(fresh_review_contract_rows)}",
            *[
                f"- {row['item']}: {row['status']}; proof only; does not authorize action, model call, tool execution, approval, routing, transcript mutation, private data, or external side effects; prior artifact reusable: no"
                for row in fresh_review_contract_rows
            ],
            "",
            "Required cycle proof chain:",
            f"- command count: {len(required_commands)}",
            *[f"- `{command}`" for command in required_commands],
            "",
            "Next safe command:",
            f"- `{next_command}`",
            "",
            "Safety boundary:",
            "- This ledger never treats speech as permission to bypass ToolRegistry, PermissionPolicy, approval gates, audit, recovery, learning, or verification.",
            "- It is proof bookkeeping only; a new spoken order must begin with a fresh transcript review and fresh confirmation receipt.",
        ]

        return ToolResult(
            "voice_cycle_ledger",
            True,
            "\n".join(lines),
            _voice_metadata(
                transcript=normalized,
                transcript_hash=transcript_hash,
                confirmation_receipt_command=_voice_confirmation_receipt_command(normalized),
                confirmed=confirmed,
                route_blocked_until_confirmation_receipt=not confirmed,
                can_route_after_confirmation_receipt=confirmed,
                mode=closure_meta.get("mode"),
                cycle_state=cycle_state,
                ready_for_fresh_next_voice_review=cycle_ready,
                voice_cycle_ledger_ready=voice_cycle_ledger_ready,
                ready_for_next_voice_review=closure_meta.get("ready_for_next_voice_review"),
                next_voice_review_state=closure_meta.get("next_voice_review_state"),
                next_review_start_command=next_review_start_command,
                next_review_requires_fresh_confirmation_receipt=True,
                next_review_requires_fresh_receipt_nonce=True,
                previous_confirmation_receipt_reusable_for_next_voice_review=False,
                previous_transcript_hash_reusable_for_next_voice_review=False,
                previous_handoff_reusable_for_next_voice_review=False,
                prior_spoken_command_proof_only=True,
                voice_post_run_closure_token_sha256=closure_meta.get("voice_post_run_closure_token_sha256"),
                voice_post_run_closure_token_present=closure_meta.get("voice_post_run_closure_token_present"),
                voice_post_run_closure_token_boundary_rows=voice_post_run_closure_token_boundary_rows,
                voice_post_run_closure_token_boundary_row_count=len(voice_post_run_closure_token_boundary_rows),
                voice_post_run_closure_token_boundary_ready=voice_post_run_closure_token_boundary_ready,
                voice_post_run_closure_token_authorizes_action_now=False,
                voice_post_run_closure_token_authorizes_model_call=False,
                voice_post_run_closure_token_authorizes_tool_execution=False,
                voice_post_run_closure_token_authorizes_approval=False,
                voice_post_run_closure_token_authorizes_routing=False,
                voice_post_run_closure_token_authorizes_transcript_mutation=False,
                voice_post_run_closure_token_authorizes_personal_data_read=False,
                voice_post_run_closure_token_authorizes_external_side_effect=False,
                voice_post_run_closure_token_authorizes_next_voice_review=False,
                voice_post_run_closure_token_authorizes_receipt_reuse=False,
                voice_post_run_closure_token_reusable_for_next_voice_review=False,
                **command_intake_contract,
                all_prior_artifacts_non_authorizing=all_prior_artifacts_non_authorizing,
                next_voice_review_requires_full_preflight=True,
                voice_cycle_ledger_token_sha256=voice_cycle_ledger_token_sha256,
                voice_cycle_ledger_token_present=_looks_like_sha256(voice_cycle_ledger_token_sha256),
                voice_cycle_ledger_token_authorizes_action_now=False,
                voice_cycle_ledger_token_authorizes_model_call=False,
                voice_cycle_ledger_token_authorizes_tool_execution=False,
                voice_cycle_ledger_token_authorizes_approval=False,
                voice_cycle_ledger_token_authorizes_routing=False,
                voice_cycle_ledger_token_authorizes_transcript_mutation=False,
                voice_cycle_ledger_token_authorizes_personal_data_read=False,
                voice_cycle_ledger_token_authorizes_external_side_effect=False,
                voice_cycle_ledger_token_authorizes_next_voice_review=False,
                voice_cycle_ledger_token_reusable_for_next_voice_review=False,
                next_voice_review_requires_new_cycle_ledger_token=True,
                fresh_review_preflight_queue=fresh_review_preflight_queue,
                fresh_review_preflight_queue_count=len(fresh_review_preflight_queue),
                fresh_review_next_preflight_command="voice capture privacy",
                fresh_review_contract_rows=fresh_review_contract_rows,
                fresh_review_contract_count=len(fresh_review_contract_rows),
                action_allowed_now=False,
                executable_tool_action_emitted=False,
                can_emit_executable_tool_action=False,
                can_auto_execute_now=False,
                routes_actions=False,
                privacy_receipt_id=closure_meta.get("privacy_receipt_id"),
                privacy_receipt_match=privacy_match,
                confirmation_receipt_nonce=closure_meta.get("confirmation_receipt_nonce"),
                receipt_nonce_required=True,
                receipt_id_match=closure_meta.get("receipt_id_match"),
                receipt_nonce_match=closure_meta.get("receipt_nonce_match"),
                receipt_freshness_match=receipt_freshness_match,
                post_run_verification_reviewed=closure_meta.get("post_run_verification_reviewed"),
                post_run_execution_health_reviewed=closure_meta.get("post_run_execution_health_reviewed"),
                post_run_execution_audit_reviewed=closure_meta.get("post_run_execution_audit_reviewed"),
                after_action_learning_reviewed=closure_meta.get("after_action_learning_reviewed"),
                verification_receipt_sha256=closure_meta.get("verification_receipt_sha256"),
                execution_health_sha256=closure_meta.get("execution_health_sha256"),
                execution_audit_sha256=closure_meta.get("execution_audit_sha256"),
                after_action_learning_sha256=closure_meta.get("after_action_learning_sha256"),
                verification_receipt_hash_present=closure_meta.get("verification_receipt_hash_present"),
                execution_health_hash_present=closure_meta.get("execution_health_hash_present"),
                execution_audit_hash_present=closure_meta.get("execution_audit_hash_present"),
                after_action_learning_hash_present=closure_meta.get("after_action_learning_hash_present"),
                post_run_artifact_hashes_present=post_run_artifact_hashes_present,
                voice_preflight_score=preflight_score,
                voice_preflight_max_score=preflight_max_score,
                voice_preflight_scorecard_rows=preflight_scorecard_rows,
                voice_preflight_scorecard_row_count=len(preflight_scorecard_rows),
                voice_preflight_required_rows_ready=preflight_required_rows_ready,
                voice_preflight_scorecard_ready=preflight_scorecard_ready,
                post_run_artifact_hashes=closure_meta.get("post_run_artifact_hashes"),
                missing_blockers=missing,
                missing_blocker_count=len(missing),
                next_command=next_command,
                required_commands=required_commands,
                required_command_count=len(required_commands),
                stage_rows=stage_rows,
                stage_count=len(stage_rows),
                voice_cycle_stage_rows_ready=voice_cycle_stage_rows_ready,
                receipt_metadata=receipt_meta,
                route_gate_metadata=route_gate_meta,
                route_proof_bundle_metadata=route_bundle_meta,
                runtime_bridge_metadata=bridge_meta,
                cockpit_metadata=cockpit_meta,
                action_audit_metadata=action_audit_meta,
                handoff_metadata=handoff_meta,
                post_run_closure_metadata=closure_meta,
                post_run_closure_output=closure.output,
                auditable=True,
            ),
        )

    return voice_cycle_ledger


def list_voices(_: dict[str, Any]) -> ToolResult:
    try:
        result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10)
    except Exception as exc:
        failure_output = (
            f"{_voice_error('list voices', exc)} "
            f"{LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION}"
        )
        return ToolResult(
            "list_voices",
            False,
            failure_output,
            declare_retryable_personal_read_failure(
                _voice_metadata(exception_type=type(exc).__name__),
                output=failure_output,
                action=LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
                commands=("setup check",),
            ),
        )
    if result.returncode != 0:
        failure_output = (
            f"{_voice_error('list voices')} "
            f"{LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION}"
        )
        return ToolResult(
            "list_voices",
            False,
            failure_output,
            declare_retryable_personal_read_failure(
                _voice_metadata(returncode=result.returncode),
                output=failure_output,
                action=LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
                commands=("setup check",),
            ),
        )
    voices = [line for line in result.stdout.splitlines() if line.strip()]
    if not voices:
        return ToolResult("list_voices", True, "No voices reported by macOS.")
    return ToolResult(
        "list_voices",
        True,
        "\n".join(voices[:80]),
        _voice_metadata(count=len(voices), shown=min(len(voices), 80), executes_tools=True),
    )
