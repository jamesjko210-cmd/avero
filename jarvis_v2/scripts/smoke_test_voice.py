from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools import voice as voice_tools
from jarvis_v2.tools.voice import (
    MAX_SPEECH_CHARS,
    MAX_TRANSCRIPT_CHARS,
    _voice_cycle_ledger_token_sha256,
    _voice_cycle_preflight_scorecard_ready,
    _voice_cycle_stage_rows_ready,
    _voice_confirmation_audit_token_sha256,
    _voice_confirmation_audit_token_boundary_ready,
    _voice_command_intake_contract_sha256,
    _voice_post_run_closure_token_boundary_ready,
    _voice_post_run_closure_token_sha256,
    _voice_native_microphone_gate_ready,
    _voice_metadata_all_bool,
    _voice_metadata_bool,
    _transcript_sha256,
    speak_text,
    voice_capture_privacy_packet,
    voice_file_transcription_plan,
    voice_input_plan,
    list_voices,
    voice_native_microphone_gate_packet,
    voice_reply_preview,
)


VOICE_SAFE_FALSE_FLAGS = [
    "calls_model",
    "calls_chat_model",
    "executes_tools",
    "authorizes_execution",
    "authorizes_completion_claim",
    "approval_granted",
    "reads_private_data",
    "reads_personal_data",
    "reads_audio_file",
    "records_audio",
    "starts_listener",
    "transcribes_audio",
    "saves_transcript",
    "writes_files",
    "writes_database",
    "writes_memory",
    "writes_notes",
    "external_side_effect",
    "queues_approval",
    "requires_approval",
    "controls_computer",
    "speaks",
    "speaks_audio",
    "completes_tasks",
]

VOICE_VERIFICATION_RECEIPT_SHA256 = "a" * 64
VOICE_EXECUTION_HEALTH_SHA256 = "b" * 64
VOICE_EXECUTION_AUDIT_SHA256 = "c" * 64
VOICE_AFTER_ACTION_LEARNING_SHA256 = "d" * 64
VOICE_POST_RUN_HASH_SUFFIX = (
    f"verification_receipt_sha256={VOICE_VERIFICATION_RECEIPT_SHA256}; "
    f"execution_health_sha256={VOICE_EXECUTION_HEALTH_SHA256}; "
    f"execution_audit_sha256={VOICE_EXECUTION_AUDIT_SHA256}; "
    f"after_action_learning_sha256={VOICE_AFTER_ACTION_LEARNING_SHA256}"
)


def assert_known_local_recovery(result, label: str, *, action: str, commands: list[str]) -> None:
    if result.ok or action not in result.output:
        raise SystemExit(f"{label} hid the canonical local recovery action: {result}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": action,
        "commands": commands,
    }:
        raise SystemExit(f"{label} recovery declaration drifted: {result.metadata}")
    expected = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
    }
    if commands:
        expected["next_command"] = commands[0]
        expected["recovery_commands"] = commands
    for key, value in expected.items():
        if result.metadata.get(key) != value:
            raise SystemExit(f"{label} recovery field {key} drifted: {result.metadata}")


def _env_truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def assert_voice_metadata_bool_is_exact() -> None:
    if _voice_metadata_bool(True) is not True or _voice_metadata_bool(False) is not False:
        raise SystemExit("Voice metadata bool helper should preserve exact bools.")
    for value in ("true", "false", "yes", "0", 1, 0, ["true"], {"value": True}, None):
        if _voice_metadata_bool(value):
            raise SystemExit(f"Voice metadata bool helper accepted malformed truthy value: {value!r}")
    for value in ("false", 0, None):
        if _voice_metadata_bool(value, default=True) is not True:
            raise SystemExit(f"Voice metadata bool helper should preserve conservative default for malformed value: {value!r}")
    if _voice_metadata_all_bool(True, "true"):
        raise SystemExit("Voice metadata all-bool helper accepted malformed truthy value.")
    if _voice_metadata_all_bool(True, True) is not True or _voice_metadata_all_bool(True, False) is not False:
        raise SystemExit("Voice metadata all-bool helper should preserve exact boolean composition.")


def assert_voice_runtime_bridge_rejects_truthy_route_metadata() -> None:
    original_factory = voice_tools.make_voice_route_proof_bundle

    def fake_route_proof_bundle_factory(_get_tool):
        def fake_route_proof_bundle(args: dict) -> voice_tools.ToolResult:
            transcript = str(args.get("transcript") or "run command python3 --version").strip()
            return voice_tools.ToolResult(
                "voice_route_proof_bundle",
                True,
                "mocked truthy-string route proof",
                voice_tools._voice_metadata(
                    transcript=transcript,
                    transcript_hash=voice_tools._transcript_hash(transcript),
                    confirmed=True,
                    bundle_state="VOICE_ROUTE_PROOF_READY",
                    bundle_ready="true",
                    can_enter_runtime="true",
                    approval_required_after_routing="true",
                    privacy_receipt_match=True,
                    receipt_freshness_match=True,
                    privacy_receipt_id="voice-privacy-mock",
                    confirmation_receipt_id="voice-receipt-mock",
                    confirmation_receipt_nonce="voice-nonce-mock",
                    route_gate_state="VOICE_READY_FOR_PLANNER",
                    proof_commands=[],
                    planned_actions=[],
                ),
            )

        return fake_route_proof_bundle

    try:
        voice_tools.make_voice_route_proof_bundle = fake_route_proof_bundle_factory  # type: ignore[assignment]
        bridge = voice_tools.make_voice_runtime_bridge_packet(lambda _name: None)
        result = bridge({"transcript": "run command python3 --version", "confirmed": "true"})
    finally:
        voice_tools.make_voice_route_proof_bundle = original_factory  # type: ignore[assignment]

    metadata = result.metadata
    if metadata.get("bridge_state") != "VOICE_HELD_BEFORE_COMMAND_INTAKE" or metadata.get("can_enter_runtime") is not False:
        raise SystemExit(f"Voice runtime bridge accepted truthy-string route readiness: {metadata}")
    if metadata.get("route_proof_bundle_ready") is not False or metadata.get("approval_required_after_bridge") is not False:
        raise SystemExit(f"Voice runtime bridge projected malformed route booleans as true: {metadata}")
    if "ready voice route proof bundle" not in metadata.get("bridge_missing", []):
        raise SystemExit(f"Voice runtime bridge missed malformed bundle blocker: {metadata}")


def assert_no_local_path(value: object, label: str) -> None:
    text = str(value)
    for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/"):
        if fragment in text:
            raise SystemExit(f"{label} leaked a local path: {text}")


def assert_voice_approval_chain(metadata: dict, label: str) -> None:
    commands = metadata.get("recommended_next_commands", [])
    for command in ["approval readiness latest", "approval packet latest", "approval chain proof latest"]:
        if command not in commands:
            raise SystemExit(f"{label} missed approval-chain next command {command}.")


def assert_voice_file_handoff(metadata: dict, key: str, label: str) -> dict:
    handoff = metadata.get(key)
    if not isinstance(handoff, dict) or metadata.get(f"{key}_ready") is not True:
        raise SystemExit(f"{label} missed {key}: {metadata}")
    for field, expected in (
        ("ready_for_operator", True),
        ("state_changed", False),
        ("changed", []),
        ("content_in_handoff", False),
        ("authorizes_execution", False),
        ("authorizes_completion_claim", False),
        ("approval_granted", False),
    ):
        if metadata.get(field) != expected or handoff.get(field) != expected:
            raise SystemExit(f"{label} missed flat/nested {field}={expected}: {metadata} / {handoff}")
    if not handoff.get("path_hash") and handoff.get("path_supplied", True):
        raise SystemExit(f"{label} missed path hash: {handoff}")
    if handoff.get("path_hash") != metadata.get("path_hash"):
        raise SystemExit(f"{label} path hash parity failed: {metadata} / {handoff}")
    rows = handoff.get("dependency_rows") or []
    if handoff.get("dependency_row_count") != len(rows) or metadata.get("dependency_row_count") != len(metadata.get("dependency_rows") or []):
        raise SystemExit(f"{label} dependency row counts diverged: {metadata} / {handoff}")
    if handoff.get("future_transcription_requires_approval") is not True or metadata.get("future_transcription_requires_approval") is not True:
        raise SystemExit(f"{label} should keep future transcription approval-gated: {metadata} / {handoff}")
    boundaries = handoff.get("boundaries") or {}
    for flag in VOICE_SAFE_FALSE_FLAGS:
        if boundaries.get(flag) is not False:
            raise SystemExit(f"{label} boundary {flag} should be false: {handoff}")
    if boundaries.get("read_only") is not True:
        raise SystemExit(f"{label} boundary should stay read-only: {handoff}")
    return handoff


def assert_voice_confirmation_handoff(metadata: dict, label: str, transcript: str, *, confirmed: bool = False) -> None:
    if not metadata.get("transcript_hash") or len(str(metadata.get("transcript_hash"))) != 16:
        raise SystemExit(f"{label} missed bounded transcript hash: {metadata}")
    expected_command = f"voice confirmation receipt: {transcript} confirmed=true"
    if metadata.get("confirmation_receipt_command") != expected_command:
        raise SystemExit(f"{label} missed confirmation receipt command: {metadata}")
    if metadata.get("can_route_after_confirmation_receipt") is not confirmed:
        raise SystemExit(f"{label} exposed wrong route-after-receipt state: {metadata}")
    if metadata.get("route_blocked_until_confirmation_receipt") is not (not confirmed):
        raise SystemExit(f"{label} exposed wrong route blocker state: {metadata}")


def assert_voice_privacy_receipt_chain(metadata: dict, label: str) -> None:
    if not metadata.get("privacy_receipt_id"):
        raise SystemExit(f"{label} missed privacy receipt id: {metadata}")
    if metadata.get("privacy_receipt_match") is not True:
        raise SystemExit(f"{label} should prove matching privacy receipt: {metadata}")


def assert_voice_native_microphone_gate(metadata: dict, label: str, *, expected_gate_state: str) -> None:
    if metadata.get("gate_state") != expected_gate_state:
        raise SystemExit(f"{label} exposed wrong gate state: {metadata}")
    if not _voice_native_microphone_gate_ready(metadata):
        raise SystemExit(f"{label} failed production native microphone gate validator: {metadata}")
    tampered = dict(metadata)
    tampered["native_microphone_capture_enabled"] = True
    if _voice_native_microphone_gate_ready(tampered):
        raise SystemExit(f"{label} validator accepted native capture enablement tampering: {tampered}")
    tampered = dict(metadata)
    tampered["required_after_transcript"] = list(metadata.get("required_after_transcript") or [])
    tampered["required_after_transcript"][0] = "voice transcript review: altered"
    if _voice_native_microphone_gate_ready(tampered):
        raise SystemExit(f"{label} validator accepted transcript proof-chain tampering: {tampered}")
    tampered = dict(metadata)
    tampered["capture_runner_requires_approval"] = False
    if _voice_native_microphone_gate_ready(tampered):
        raise SystemExit(f"{label} validator accepted approval-boundary tampering: {tampered}")


def assert_voice_post_run_closure_token_boundary(metadata: dict, label: str, *, expected_source: str) -> None:
    token = str(metadata.get("voice_post_run_closure_token_sha256") or "")
    if len(token) != 64:
        raise SystemExit(f"{label} missed voice post-run closure token: {metadata}")
    if metadata.get("voice_post_run_closure_token_present") is not True:
        raise SystemExit(f"{label} missed voice post-run closure token presence: {metadata}")
    rows = metadata.get("voice_post_run_closure_token_boundary_rows") or []
    if metadata.get("voice_post_run_closure_token_boundary_row_count") != 3:
        raise SystemExit(f"{label} missed voice post-run closure token boundary rows: {metadata}")
    if metadata.get("voice_post_run_closure_token_boundary_ready") is not True:
        raise SystemExit(f"{label} did not mark voice post-run closure token boundary ready: {metadata}")
    if not _voice_post_run_closure_token_boundary_ready(rows, token_sha256=token, source=expected_source):
        raise SystemExit(f"{label} production post-run closure token boundary validator rejected rows: {metadata}")
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["status"] = "present_and_authorizing"
    if _voice_post_run_closure_token_boundary_ready(tampered_rows, token_sha256=token, source=expected_source):
        raise SystemExit(f"{label} production post-run closure validator accepted status tampering: {tampered_rows}")
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["authorizes_next_voice_review"] = True
    if _voice_post_run_closure_token_boundary_ready(tampered_rows, token_sha256=token, source=expected_source):
        raise SystemExit(f"{label} production post-run closure validator accepted authority tampering: {tampered_rows}")
    for key in [
        "voice_post_run_closure_token_authorizes_action_now",
        "voice_post_run_closure_token_authorizes_model_call",
        "voice_post_run_closure_token_authorizes_tool_execution",
        "voice_post_run_closure_token_authorizes_approval",
        "voice_post_run_closure_token_authorizes_routing",
        "voice_post_run_closure_token_authorizes_transcript_mutation",
        "voice_post_run_closure_token_authorizes_personal_data_read",
        "voice_post_run_closure_token_authorizes_external_side_effect",
        "voice_post_run_closure_token_authorizes_next_voice_review",
        "voice_post_run_closure_token_authorizes_receipt_reuse",
        "voice_post_run_closure_token_reusable_for_next_voice_review",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should report {key}=False: {metadata}")


def assert_voice_command_intake_contract(metadata: dict, label: str) -> None:
    contract = str(metadata.get("voice_command_intake_contract_sha256") or "")
    if len(contract) != 64:
        raise SystemExit(f"{label} missed voice command-intake contract sha256: {metadata}")
    if metadata.get("transcript_sha256") != _transcript_sha256(str(metadata.get("transcript") or "")):
        raise SystemExit(f"{label} missed exact transcript sha256 binding: {metadata}")
    if metadata.get("voice_command_intake_contract_present") is not True or metadata.get("voice_command_intake_contract_ready") is not True:
        raise SystemExit(f"{label} missed ready command-intake contract flags: {metadata}")
    if metadata.get("next_voice_review_requires_fresh_command_intake_contract") is not True:
        raise SystemExit(f"{label} should require a fresh command-intake contract next review: {metadata}")
    for key in [
        "voice_command_intake_contract_authorizes_action_now",
        "voice_command_intake_contract_authorizes_model_call",
        "voice_command_intake_contract_authorizes_tool_execution",
        "voice_command_intake_contract_authorizes_approval",
        "voice_command_intake_contract_authorizes_routing",
        "voice_command_intake_contract_authorizes_transcript_mutation",
        "voice_command_intake_contract_authorizes_personal_data_read",
        "voice_command_intake_contract_authorizes_external_side_effect",
        "voice_command_intake_contract_reusable_for_next_voice_review",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} command-intake contract should report {key}=False: {metadata}")
    recomputed = _voice_command_intake_contract_sha256(
        transcript=str(metadata.get("transcript") or ""),
        transcript_sha256=str(metadata.get("transcript_sha256") or ""),
        privacy_receipt_id=str(metadata.get("voice_command_intake_contract_privacy_receipt_id") or ""),
        confirmation_receipt_id=str(metadata.get("voice_command_intake_contract_confirmation_receipt_id") or ""),
        confirmation_receipt_nonce=str(metadata.get("voice_command_intake_contract_confirmation_receipt_nonce") or ""),
        route_gate_state=str(metadata.get("voice_command_intake_contract_route_gate_state") or ""),
        route_proof_bundle_state=str(metadata.get("voice_command_intake_contract_route_proof_bundle_state") or ""),
        runtime_bridge_state=str(metadata.get("voice_command_intake_contract_runtime_bridge_state") or ""),
        proof_commands=list(metadata.get("voice_command_intake_contract_proof_commands") or []),
    )
    if recomputed != contract:
        raise SystemExit(f"{label} command-intake contract should be reproducible: {metadata}")
    tampered = _voice_command_intake_contract_sha256(
        transcript=str(metadata.get("transcript") or ""),
        transcript_sha256=str(metadata.get("transcript_sha256") or ""),
        privacy_receipt_id=str(metadata.get("voice_command_intake_contract_privacy_receipt_id") or ""),
        confirmation_receipt_id=str(metadata.get("voice_command_intake_contract_confirmation_receipt_id") or ""),
        confirmation_receipt_nonce=str(metadata.get("voice_command_intake_contract_confirmation_receipt_nonce") or ""),
        route_gate_state=str(metadata.get("voice_command_intake_contract_route_gate_state") or ""),
        route_proof_bundle_state=str(metadata.get("voice_command_intake_contract_route_proof_bundle_state") or ""),
        runtime_bridge_state=str(metadata.get("voice_command_intake_contract_runtime_bridge_state") or ""),
        proof_commands=list(metadata.get("voice_command_intake_contract_proof_commands") or []),
        authorizes_routing=True,
    )
    if tampered == contract:
        raise SystemExit(f"{label} command-intake contract should bind routing authority: {metadata}")


def assert_voice_post_run_closure_token_hash_binds_authority(metadata: dict, label: str) -> None:
    token = str(metadata.get("voice_post_run_closure_token_sha256") or "")
    recomputed = _voice_post_run_closure_token_sha256(
        transcript_hash=str(metadata.get("transcript_hash") or ""),
        closure_state=str(metadata.get("closure_state") or ""),
        receipt_nonce=str(metadata.get("confirmation_receipt_nonce") or ""),
        handoff_state=str(metadata.get("handoff_state") or ""),
        handoff_ready=bool(metadata.get("handoff_ready")),
        command_intake_contract_sha256=str(metadata.get("voice_command_intake_contract_sha256") or ""),
        verification_receipt_sha256=str(metadata.get("verification_receipt_sha256") or ""),
        execution_health_sha256=str(metadata.get("execution_health_sha256") or ""),
        execution_audit_sha256=str(metadata.get("execution_audit_sha256") or ""),
        after_action_learning_sha256=str(metadata.get("after_action_learning_sha256") or ""),
        next_review_start_command=str(metadata.get("next_review_start_command") or ""),
    )
    if recomputed != token:
        raise SystemExit(f"{label} voice post-run closure token should be reproducible: {metadata}")
    tampered = _voice_post_run_closure_token_sha256(
        transcript_hash=str(metadata.get("transcript_hash") or ""),
        closure_state=str(metadata.get("closure_state") or ""),
        receipt_nonce=str(metadata.get("confirmation_receipt_nonce") or ""),
        handoff_state=str(metadata.get("handoff_state") or ""),
        handoff_ready=bool(metadata.get("handoff_ready")),
        command_intake_contract_sha256=str(metadata.get("voice_command_intake_contract_sha256") or ""),
        verification_receipt_sha256=str(metadata.get("verification_receipt_sha256") or ""),
        execution_health_sha256=str(metadata.get("execution_health_sha256") or ""),
        execution_audit_sha256=str(metadata.get("execution_audit_sha256") or ""),
        after_action_learning_sha256=str(metadata.get("after_action_learning_sha256") or ""),
        next_review_start_command=str(metadata.get("next_review_start_command") or ""),
        authorizes_personal_data_read=True,
    )
    if tampered == token:
        raise SystemExit(f"{label} voice post-run closure token should bind personal-data authority: {metadata}")
    tampered_handoff = _voice_post_run_closure_token_sha256(
        transcript_hash=str(metadata.get("transcript_hash") or ""),
        closure_state=str(metadata.get("closure_state") or ""),
        receipt_nonce=str(metadata.get("confirmation_receipt_nonce") or ""),
        handoff_state="VOICE_EXECUTION_HANDOFF_TAMPERED",
        handoff_ready=bool(metadata.get("handoff_ready")),
        command_intake_contract_sha256=str(metadata.get("voice_command_intake_contract_sha256") or ""),
        verification_receipt_sha256=str(metadata.get("verification_receipt_sha256") or ""),
        execution_health_sha256=str(metadata.get("execution_health_sha256") or ""),
        execution_audit_sha256=str(metadata.get("execution_audit_sha256") or ""),
        after_action_learning_sha256=str(metadata.get("after_action_learning_sha256") or ""),
        next_review_start_command=str(metadata.get("next_review_start_command") or ""),
    )
    if tampered_handoff == token:
        raise SystemExit(f"{label} voice post-run closure token should bind handoff state: {metadata}")
    tampered_contract = _voice_post_run_closure_token_sha256(
        transcript_hash=str(metadata.get("transcript_hash") or ""),
        closure_state=str(metadata.get("closure_state") or ""),
        receipt_nonce=str(metadata.get("confirmation_receipt_nonce") or ""),
        handoff_state=str(metadata.get("handoff_state") or ""),
        handoff_ready=bool(metadata.get("handoff_ready")),
        command_intake_contract_sha256="0" * 64,
        verification_receipt_sha256=str(metadata.get("verification_receipt_sha256") or ""),
        execution_health_sha256=str(metadata.get("execution_health_sha256") or ""),
        execution_audit_sha256=str(metadata.get("execution_audit_sha256") or ""),
        after_action_learning_sha256=str(metadata.get("after_action_learning_sha256") or ""),
        next_review_start_command=str(metadata.get("next_review_start_command") or ""),
    )
    if metadata.get("voice_command_intake_contract_sha256") != "0" * 64 and tampered_contract == token:
        raise SystemExit(f"{label} voice post-run closure token should bind command-intake contract: {metadata}")
    if metadata.get("voice_post_run_closure_token_binds_handoff_state") is not True:
        raise SystemExit(f"{label} should report post-run token binds handoff state: {metadata}")
    if metadata.get("voice_post_run_closure_token_binds_handoff_ready") is not True:
        raise SystemExit(f"{label} should report post-run token binds handoff readiness: {metadata}")
    if metadata.get("voice_post_run_closure_token_binds_command_intake_contract") is not True:
        raise SystemExit(f"{label} should report post-run token binds command-intake contract: {metadata}")


def assert_voice_confirmation_audit_token_boundary(metadata: dict, label: str) -> None:
    token = str(metadata.get("voice_confirmation_audit_token_sha256") or "")
    if len(token) != 64:
        raise SystemExit(f"{label} missed voice confirmation audit token: {metadata}")
    if metadata.get("voice_confirmation_audit_token_present") is not True:
        raise SystemExit(f"{label} missed voice confirmation audit token presence: {metadata}")
    rows = metadata.get("voice_confirmation_audit_token_boundary_rows") or []
    if metadata.get("voice_confirmation_audit_token_boundary_ready") is not True:
        raise SystemExit(f"{label} missed ready voice confirmation audit token boundary flag: {metadata}")
    if not _voice_confirmation_audit_token_boundary_ready(rows, token_sha256=token):
        raise SystemExit(f"{label} voice confirmation audit token boundary failed production validator: {metadata}")
    tampered_rows = [dict(row) for row in rows]
    if tampered_rows:
        tampered_rows[0]["status"] = "tampered"
    if _voice_confirmation_audit_token_boundary_ready(tampered_rows, token_sha256=token):
        raise SystemExit(f"{label} voice confirmation audit token boundary accepted status tampering: {metadata}")
    tampered_rows = [dict(row) for row in rows]
    if tampered_rows:
        tampered_rows[0]["authorizes_routing"] = True
    if _voice_confirmation_audit_token_boundary_ready(tampered_rows, token_sha256=token):
        raise SystemExit(f"{label} voice confirmation audit token boundary accepted routing authority: {metadata}")
    expected_rows = {
        "voice_confirmation_audit_token": "present",
        "confirmed_transcript_scope": "proof_only_for_exact_transcript",
        "receipt_reuse_boundary": "fresh_confirmation_receipt_required",
    }
    if metadata.get("voice_confirmation_audit_token_boundary_row_count") != 3 or len(rows) != 3:
        raise SystemExit(f"{label} missed voice confirmation audit token boundary rows: {metadata}")
    if {row.get("item") for row in rows} != set(expected_rows):
        raise SystemExit(f"{label} voice confirmation audit token boundary items diverged: {metadata}")
    for row in rows:
        item = row.get("item")
        if row.get("status") != expected_rows.get(item):
            raise SystemExit(f"{label} voice confirmation audit token boundary status diverged: {row}")
        if row.get("source") != "voice_confirmation_audit_ledger":
            raise SystemExit(f"{label} voice confirmation audit token boundary source diverged: {row}")
        if row.get("token_sha256") != token:
            raise SystemExit(f"{label} voice confirmation audit token boundary hash diverged: {row}")
        if (
            row.get("authorizes_action_now") is not False
            or row.get("authorizes_model_call") is not False
            or row.get("authorizes_tool_execution") is not False
            or row.get("authorizes_approval") is not False
            or row.get("authorizes_routing") is not False
            or row.get("authorizes_transcript_mutation") is not False
            or row.get("authorizes_personal_data_read") is not False
            or row.get("authorizes_external_side_effect") is not False
            or row.get("authorizes_receipt_reuse") is not False
            or row.get("reusable_for_next_voice_review") is not False
        ):
            raise SystemExit(f"{label} voice confirmation audit token boundary row should be non-authorizing: {row}")
    for key in [
        "voice_confirmation_audit_token_authorizes_action_now",
        "voice_confirmation_audit_token_authorizes_model_call",
        "voice_confirmation_audit_token_authorizes_tool_execution",
        "voice_confirmation_audit_token_authorizes_approval",
        "voice_confirmation_audit_token_authorizes_routing",
        "voice_confirmation_audit_token_authorizes_transcript_mutation",
        "voice_confirmation_audit_token_authorizes_personal_data_read",
        "voice_confirmation_audit_token_authorizes_external_side_effect",
        "voice_confirmation_audit_token_authorizes_receipt_reuse",
        "voice_confirmation_audit_token_reusable_for_next_voice_review",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should report {key}=False: {metadata}")


def assert_voice_confirmation_audit_token_hash_binds_authority(metadata: dict, label: str) -> None:
    token = str(metadata.get("voice_confirmation_audit_token_sha256") or "")
    recomputed = _voice_confirmation_audit_token_sha256(
        transcript_hash=str(metadata.get("transcript_hash") or ""),
        audit_ledger_state=str(metadata.get("audit_ledger_state") or ""),
        privacy_receipt_id=str(metadata.get("privacy_receipt_id") or ""),
        confirmation_receipt_id=str(metadata.get("confirmation_receipt_id") or ""),
        confirmation_receipt_nonce=str(metadata.get("confirmation_receipt_nonce") or ""),
        supplied_privacy_receipt_id=str(metadata.get("supplied_privacy_receipt_id") or ""),
        supplied_confirmation_receipt_id=str(metadata.get("supplied_confirmation_receipt_id") or ""),
        supplied_confirmation_receipt_nonce=str(metadata.get("supplied_confirmation_receipt_nonce") or ""),
        route_gate_state=str(metadata.get("route_gate_state") or ""),
        route_proof_bundle_state=str(metadata.get("route_proof_bundle_state") or ""),
        runtime_bridge_state=str(metadata.get("runtime_bridge_state") or ""),
        ready_for_command_intake_proof=bool(metadata.get("ready_for_command_intake_proof")),
        stage_count=int(metadata.get("stage_count") or 0),
        required_command_count=int(metadata.get("required_command_count") or 0),
    )
    if recomputed != token:
        raise SystemExit(f"{label} voice confirmation audit token should be reproducible: {metadata}")
    tampered = _voice_confirmation_audit_token_sha256(
        transcript_hash=str(metadata.get("transcript_hash") or ""),
        audit_ledger_state=str(metadata.get("audit_ledger_state") or ""),
        privacy_receipt_id=str(metadata.get("privacy_receipt_id") or ""),
        confirmation_receipt_id=str(metadata.get("confirmation_receipt_id") or ""),
        confirmation_receipt_nonce=str(metadata.get("confirmation_receipt_nonce") or ""),
        supplied_privacy_receipt_id=str(metadata.get("supplied_privacy_receipt_id") or ""),
        supplied_confirmation_receipt_id=str(metadata.get("supplied_confirmation_receipt_id") or ""),
        supplied_confirmation_receipt_nonce=str(metadata.get("supplied_confirmation_receipt_nonce") or ""),
        route_gate_state=str(metadata.get("route_gate_state") or ""),
        route_proof_bundle_state=str(metadata.get("route_proof_bundle_state") or ""),
        runtime_bridge_state=str(metadata.get("runtime_bridge_state") or ""),
        ready_for_command_intake_proof=bool(metadata.get("ready_for_command_intake_proof")),
        stage_count=int(metadata.get("stage_count") or 0),
        required_command_count=int(metadata.get("required_command_count") or 0),
        authorizes_tool_execution=True,
    )
    if tampered == token:
        raise SystemExit(f"{label} voice confirmation audit token should bind tool-execution authority: {metadata}")


def assert_voice_cycle_ledger_token_hash_binds_authority(metadata: dict, label: str) -> None:
    token = str(metadata.get("voice_cycle_ledger_token_sha256") or "")
    if len(token) != 64:
        raise SystemExit(f"{label} missed voice cycle ledger token: {metadata}")
    if metadata.get("voice_cycle_ledger_token_present") is not True:
        raise SystemExit(f"{label} missed voice cycle ledger token presence: {metadata}")
    for key in [
        "voice_cycle_ledger_token_authorizes_action_now",
        "voice_cycle_ledger_token_authorizes_model_call",
        "voice_cycle_ledger_token_authorizes_tool_execution",
        "voice_cycle_ledger_token_authorizes_approval",
        "voice_cycle_ledger_token_authorizes_routing",
        "voice_cycle_ledger_token_authorizes_transcript_mutation",
        "voice_cycle_ledger_token_authorizes_personal_data_read",
        "voice_cycle_ledger_token_authorizes_external_side_effect",
        "voice_cycle_ledger_token_authorizes_next_voice_review",
        "voice_cycle_ledger_token_reusable_for_next_voice_review",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should report {key}=False: {metadata}")
    if metadata.get("next_voice_review_requires_new_cycle_ledger_token") is not True:
        raise SystemExit(f"{label} should require a fresh cycle-ledger token next review: {metadata}")

    recomputed = _voice_cycle_ledger_token_sha256(
        transcript=str(metadata.get("transcript") or ""),
        transcript_hash=str(metadata.get("transcript_hash") or ""),
        cycle_state=str(metadata.get("cycle_state") or ""),
        stage_rows=list(metadata.get("stage_rows") or []),
        voice_preflight_scorecard_rows=list(metadata.get("voice_preflight_scorecard_rows") or []),
        fresh_review_preflight_queue=list(metadata.get("fresh_review_preflight_queue") or []),
        fresh_review_contract_rows=list(metadata.get("fresh_review_contract_rows") or []),
        required_commands=list(metadata.get("required_commands") or []),
        command_intake_contract_sha256=str(metadata.get("voice_command_intake_contract_sha256") or ""),
        voice_post_run_closure_token_sha256=str(metadata.get("voice_post_run_closure_token_sha256") or ""),
        verification_receipt_sha256=str(metadata.get("verification_receipt_sha256") or ""),
        execution_health_sha256=str(metadata.get("execution_health_sha256") or ""),
        execution_audit_sha256=str(metadata.get("execution_audit_sha256") or ""),
        after_action_learning_sha256=str(metadata.get("after_action_learning_sha256") or ""),
        next_command=str(metadata.get("next_command") or ""),
    )
    if recomputed != token:
        raise SystemExit(f"{label} voice cycle ledger token should be reproducible: {metadata}")

    tampered_hash = _voice_cycle_ledger_token_sha256(
        transcript=str(metadata.get("transcript") or ""),
        transcript_hash=str(metadata.get("transcript_hash") or ""),
        cycle_state=str(metadata.get("cycle_state") or ""),
        stage_rows=list(metadata.get("stage_rows") or []),
        voice_preflight_scorecard_rows=list(metadata.get("voice_preflight_scorecard_rows") or []),
        fresh_review_preflight_queue=list(metadata.get("fresh_review_preflight_queue") or []),
        fresh_review_contract_rows=list(metadata.get("fresh_review_contract_rows") or []),
        required_commands=list(metadata.get("required_commands") or []),
        command_intake_contract_sha256=str(metadata.get("voice_command_intake_contract_sha256") or ""),
        voice_post_run_closure_token_sha256=str(metadata.get("voice_post_run_closure_token_sha256") or ""),
        verification_receipt_sha256="0" * 64,
        execution_health_sha256=str(metadata.get("execution_health_sha256") or ""),
        execution_audit_sha256=str(metadata.get("execution_audit_sha256") or ""),
        after_action_learning_sha256=str(metadata.get("after_action_learning_sha256") or ""),
        next_command=str(metadata.get("next_command") or ""),
    )
    if metadata.get("verification_receipt_sha256") != "0" * 64 and tampered_hash == token:
        raise SystemExit(f"{label} voice cycle ledger token should bind post-run hashes: {metadata}")

    tampered_stage_rows = [dict(row) for row in list(metadata.get("stage_rows") or [])]
    if tampered_stage_rows:
        tampered_stage_rows[0]["authorizes_tool_execution"] = True
    tampered_stage = _voice_cycle_ledger_token_sha256(
        transcript=str(metadata.get("transcript") or ""),
        transcript_hash=str(metadata.get("transcript_hash") or ""),
        cycle_state=str(metadata.get("cycle_state") or ""),
        stage_rows=tampered_stage_rows,
        voice_preflight_scorecard_rows=list(metadata.get("voice_preflight_scorecard_rows") or []),
        fresh_review_preflight_queue=list(metadata.get("fresh_review_preflight_queue") or []),
        fresh_review_contract_rows=list(metadata.get("fresh_review_contract_rows") or []),
        required_commands=list(metadata.get("required_commands") or []),
        command_intake_contract_sha256=str(metadata.get("voice_command_intake_contract_sha256") or ""),
        voice_post_run_closure_token_sha256=str(metadata.get("voice_post_run_closure_token_sha256") or ""),
        verification_receipt_sha256=str(metadata.get("verification_receipt_sha256") or ""),
        execution_health_sha256=str(metadata.get("execution_health_sha256") or ""),
        execution_audit_sha256=str(metadata.get("execution_audit_sha256") or ""),
        after_action_learning_sha256=str(metadata.get("after_action_learning_sha256") or ""),
        next_command=str(metadata.get("next_command") or ""),
    )
    if tampered_stage_rows and tampered_stage == token:
        raise SystemExit(f"{label} voice cycle ledger token should bind stage authority flags: {metadata}")

    tampered_scorecard = [dict(row) for row in list(metadata.get("voice_preflight_scorecard_rows") or [])]
    if tampered_scorecard:
        tampered_scorecard[0]["points"] = 0
    tampered_score = _voice_cycle_ledger_token_sha256(
        transcript=str(metadata.get("transcript") or ""),
        transcript_hash=str(metadata.get("transcript_hash") or ""),
        cycle_state=str(metadata.get("cycle_state") or ""),
        stage_rows=list(metadata.get("stage_rows") or []),
        voice_preflight_scorecard_rows=tampered_scorecard,
        fresh_review_preflight_queue=list(metadata.get("fresh_review_preflight_queue") or []),
        fresh_review_contract_rows=list(metadata.get("fresh_review_contract_rows") or []),
        required_commands=list(metadata.get("required_commands") or []),
        command_intake_contract_sha256=str(metadata.get("voice_command_intake_contract_sha256") or ""),
        voice_post_run_closure_token_sha256=str(metadata.get("voice_post_run_closure_token_sha256") or ""),
        verification_receipt_sha256=str(metadata.get("verification_receipt_sha256") or ""),
        execution_health_sha256=str(metadata.get("execution_health_sha256") or ""),
        execution_audit_sha256=str(metadata.get("execution_audit_sha256") or ""),
        after_action_learning_sha256=str(metadata.get("after_action_learning_sha256") or ""),
        next_command=str(metadata.get("next_command") or ""),
    )
    if tampered_scorecard and tampered_score == token:
        raise SystemExit(f"{label} voice cycle ledger token should bind preflight scorecard rows: {metadata}")

    tampered_queue = list(metadata.get("fresh_review_preflight_queue") or [])
    if tampered_queue:
        tampered_queue[0] = "voice capture privacy: tampered"
    tampered_queue_token = _voice_cycle_ledger_token_sha256(
        transcript=str(metadata.get("transcript") or ""),
        transcript_hash=str(metadata.get("transcript_hash") or ""),
        cycle_state=str(metadata.get("cycle_state") or ""),
        stage_rows=list(metadata.get("stage_rows") or []),
        voice_preflight_scorecard_rows=list(metadata.get("voice_preflight_scorecard_rows") or []),
        fresh_review_preflight_queue=tampered_queue,
        fresh_review_contract_rows=list(metadata.get("fresh_review_contract_rows") or []),
        required_commands=list(metadata.get("required_commands") or []),
        command_intake_contract_sha256=str(metadata.get("voice_command_intake_contract_sha256") or ""),
        voice_post_run_closure_token_sha256=str(metadata.get("voice_post_run_closure_token_sha256") or ""),
        verification_receipt_sha256=str(metadata.get("verification_receipt_sha256") or ""),
        execution_health_sha256=str(metadata.get("execution_health_sha256") or ""),
        execution_audit_sha256=str(metadata.get("execution_audit_sha256") or ""),
        after_action_learning_sha256=str(metadata.get("after_action_learning_sha256") or ""),
        next_command=str(metadata.get("next_command") or ""),
    )
    if tampered_queue and tampered_queue_token == token:
        raise SystemExit(f"{label} voice cycle ledger token should bind fresh-review queue: {metadata}")


def assert_voice_cycle_preflight_scorecard_ready_flag(metadata: dict, label: str, *, expected_ready: bool) -> None:
    rows = list(metadata.get("voice_preflight_scorecard_rows") or [])
    if metadata.get("voice_preflight_scorecard_row_count") != len(rows):
        raise SystemExit(f"{label} preflight scorecard row count drifted: {metadata}")
    if metadata.get("voice_preflight_scorecard_ready") is not expected_ready:
        raise SystemExit(f"{label} preflight scorecard ready flag drifted: {metadata}")
    if _voice_cycle_preflight_scorecard_ready(rows) is not expected_ready:
        raise SystemExit(f"{label} preflight scorecard production validator drifted: {metadata}")
    if not rows:
        raise SystemExit(f"{label} missed preflight scorecard rows: {metadata}")

    tampered_points = [dict(row) for row in rows]
    tampered_points[0]["points"] = 0
    if _voice_cycle_preflight_scorecard_ready(tampered_points):
        raise SystemExit(f"{label} preflight scorecard validator accepted point tampering: {tampered_points}")

    tampered_authority = [dict(row) for row in rows]
    tampered_authority[0]["authorizes_tool_execution"] = True
    if _voice_cycle_preflight_scorecard_ready(tampered_authority):
        raise SystemExit(f"{label} preflight scorecard validator accepted authority tampering: {tampered_authority}")

    tampered_requirement = [dict(row) for row in rows]
    tampered_requirement[0]["required_before_next_voice_review"] = False
    if _voice_cycle_preflight_scorecard_ready(tampered_requirement):
        raise SystemExit(f"{label} preflight scorecard validator accepted requirement tampering: {tampered_requirement}")


def assert_voice_cycle_stage_rows_ready_flag(metadata: dict, label: str, *, expected_ready: bool) -> None:
    rows = list(metadata.get("stage_rows") or [])
    if metadata.get("stage_count") != len(rows):
        raise SystemExit(f"{label} stage row count drifted: {metadata}")
    if metadata.get("voice_cycle_stage_rows_ready") is not expected_ready:
        raise SystemExit(f"{label} stage rows ready flag drifted: {metadata}")
    if _voice_cycle_stage_rows_ready(rows) is not expected_ready:
        raise SystemExit(f"{label} stage row production validator drifted: {metadata}")
    if not rows:
        raise SystemExit(f"{label} missed stage rows: {metadata}")

    tampered_ready = [dict(row) for row in rows]
    tampered_ready[0]["ready"] = False
    if _voice_cycle_stage_rows_ready(tampered_ready):
        raise SystemExit(f"{label} stage row validator accepted readiness tampering: {tampered_ready}")

    tampered_proof = [dict(row) for row in rows]
    tampered_proof[0]["proof"] = ""
    if _voice_cycle_stage_rows_ready(tampered_proof):
        raise SystemExit(f"{label} stage row validator accepted proof tampering: {tampered_proof}")

    tampered_authority = [dict(row) for row in rows]
    tampered_authority[0]["authorizes_tool_execution"] = True
    if _voice_cycle_stage_rows_ready(tampered_authority):
        raise SystemExit(f"{label} stage row validator accepted authority tampering: {tampered_authority}")


def test_whisper_cli_fallback_transcribes(tmp_text: str = "hello from whisper cli") -> None:
    # Mock the CLI so no real whisper/network is needed; prove the fallback wiring.
    import subprocess as _sp

    original_which = voice_tools.shutil.which
    original_run = voice_tools.subprocess.run
    voice_tools._AUDIO_FILE_TRANSCRIBER = None  # type: ignore[assignment]
    try:
        voice_tools.shutil.which = lambda name: "/usr/bin/whisper" if name == "whisper" else None  # type: ignore[assignment]
        os.environ.pop("JARVIS_VOICE_WHISPER_CLI", None)
        if voice_tools._voice_local_transcriber_source() != "local_whisper_cli":
            raise SystemExit("whisper CLI should be detected as the transcriber source")

        def fake_run(cmd, capture_output=True, text=True, timeout=None):
            # cmd: [cli, audio, --model, .., --output_dir, OUT, ..]
            out_dir = cmd[cmd.index("--output_dir") + 1]
            audio = cmd[1]
            from pathlib import Path as _P
            (_P(out_dir) / (_P(audio).stem + ".txt")).write_text(tmp_text, encoding="utf-8")
            return _sp.CompletedProcess(cmd, 0, stdout="", stderr="")

        voice_tools.subprocess.run = fake_run  # type: ignore[assignment]
        from pathlib import Path as _P
        result = voice_tools._transcribe_with_whisper_cli(_P("/tmp/whatever.oga"))
        if result != tmp_text:
            raise SystemExit(f"whisper CLI transcript wrong: {result!r}")
    finally:
        voice_tools.shutil.which = original_which  # type: ignore[assignment]
        voice_tools.subprocess.run = original_run  # type: ignore[assignment]


def test_voice_error_surfaces_deliberate_setup_error_text() -> None:
    """VoiceSetupError text is deliberately human-readable (missing CLI, bad model
    path env var, etc.) -- it should reach the user instead of a generic message."""
    msg = voice_tools._voice_error(
        "transcribe the approved audio file",
        voice_tools.VoiceSetupError("whisper CLI not found; set JARVIS_VOICE_WHISPER_CLI or install whisper."),
    )
    if "whisper CLI not found" not in msg:
        raise SystemExit(f"_voice_error dropped the specific, actionable message: {msg!r}")
    if "Check macOS voice/speech permissions" in msg:
        raise SystemExit(f"_voice_error should not send permissions advice for a missing-CLI error: {msg!r}")


def test_voice_error_falls_back_for_non_runtime_errors() -> None:
    """Exceptions that are not VoiceSetupError (unexpected/internal, including a
    bare RuntimeError from somewhere else) keep the generic message -- only the
    deliberately-raised VoiceSetupErrors are treated as safe to show verbatim."""
    msg = voice_tools._voice_error("speak text", TimeoutError("boom"))
    for expected in ["System Settings > Accessibility > Spoken Content", "System Settings > Sound > Output", "voice setup check", "then retry"]:
        if expected not in msg:
            raise SystemExit(f"_voice_error generic speech fallback missed actionable guidance {expected}: {msg!r}")
    if "local Jarvis logs" in msg:
        raise SystemExit(f"_voice_error should keep the generic fallback for non-setup errors: {msg!r}")
    if "boom" in msg:
        raise SystemExit(f"_voice_error should not leak raw non-setup exception text: {msg!r}")

    # A bare RuntimeError (not VoiceSetupError) must NOT be treated as safe to show --
    # this is the exact regression this test guards: an arbitrary RuntimeError from an
    # unexpected internal failure (e.g. a mocked subprocess call blowing up) should stay
    # generic, same as the "speech exception" case exercised elsewhere in this file.
    bare_msg = voice_tools._voice_error("speak text", RuntimeError("say missing near /\x55sers/example/private/speech"))
    if "say missing" in bare_msg or "/\x55sers/operator" in bare_msg:
        raise SystemExit(f"_voice_error should not surface a bare RuntimeError's raw text: {bare_msg!r}")
    for expected in ["System Settings > Accessibility > Spoken Content", "System Settings > Sound > Output", "voice setup check", "then retry"]:
        if expected not in bare_msg:
            raise SystemExit(f"_voice_error bare RuntimeError fallback missed actionable guidance {expected}: {bare_msg!r}")

    transcribe_msg = voice_tools._voice_error(
        "transcribe the approved audio file",
        TimeoutError("audio denied near /\x55sers/example/private/voice-note.m4a"),
    )
    for expected in ["audio-file permissions", "ASR dependencies", "voice setup check", "then retry"]:
        if expected not in transcribe_msg:
            raise SystemExit(f"_voice_error transcription fallback missed actionable guidance {expected}: {transcribe_msg!r}")
    if "audio denied" in transcribe_msg or "/\x55sers/operator" in transcribe_msg:
        raise SystemExit(f"_voice_error transcription fallback leaked raw exception text: {transcribe_msg!r}")


def test_voice_error_redacts_local_paths_in_setup_error_text() -> None:
    msg = voice_tools._voice_error(
        "transcribe the approved audio file",
        voice_tools.VoiceSetupError("model file missing at /\x55sers/the operator/.jarvis/whisper/model.bin"),
    )
    if "/\x55sers/" in msg:
        raise SystemExit(f"_voice_error should redact local paths even in VoiceSetupError text: {msg!r}")


def test_faster_whisper_honors_explicit_language_preference() -> None:
    original_path = os.environ.get("JARVIS_VOICE_FASTER_WHISPER_MODEL_PATH")
    original_model = voice_tools._LOCAL_FASTER_WHISPER_MODEL
    original_cache_key = voice_tools._LOCAL_FASTER_WHISPER_MODEL_CACHE_KEY
    seen: dict[str, object] = {}

    class FakeModel:
        def transcribe(self, _path, **kwargs):
            seen.update(kwargs)
            return [SimpleNamespace(text="현재 시간 알려줘")], SimpleNamespace(language="ko")

    try:
        with TemporaryDirectory() as tmp:
            model_path = Path(tmp)
            os.environ["JARVIS_VOICE_FASTER_WHISPER_MODEL_PATH"] = str(model_path)
            voice_tools._LOCAL_FASTER_WHISPER_MODEL = FakeModel()
            voice_tools._LOCAL_FASTER_WHISPER_MODEL_CACHE_KEY = str(model_path)
            transcript = voice_tools._transcribe_with_local_faster_whisper(model_path / "clip.wav", language="ko")
    finally:
        if original_path is None:
            os.environ.pop("JARVIS_VOICE_FASTER_WHISPER_MODEL_PATH", None)
        else:
            os.environ["JARVIS_VOICE_FASTER_WHISPER_MODEL_PATH"] = original_path
        voice_tools._LOCAL_FASTER_WHISPER_MODEL = original_model
        voice_tools._LOCAL_FASTER_WHISPER_MODEL_CACHE_KEY = original_cache_key

    if transcript != "현재 시간 알려줘" or seen.get("language") != "ko":
        raise SystemExit(f"faster-whisper should receive the explicit Korean preference: {seen}, {transcript!r}")


def main() -> None:
    for env_name in ("JARVIS_VOICE_WHISPER_MODEL_PATH", "JARVIS_VOICE_FASTER_WHISPER_MODEL_PATH"):
        os.environ.pop(env_name, None)

    test_voice_error_surfaces_deliberate_setup_error_text()
    test_voice_error_falls_back_for_non_runtime_errors()
    test_voice_error_redacts_local_paths_in_setup_error_text()
    test_faster_whisper_honors_explicit_language_preference()
    test_whisper_cli_fallback_transcribes()
    # Keep the rest of the suite deterministic regardless of whether the whisper CLI
    # happens to be installed on this machine: exercise the "not configured" branches
    # with CLI auto-detection neutralized.
    voice_tools._whisper_cli_path = lambda: ""  # type: ignore[assignment]

    assert_voice_metadata_bool_is_exact()
    assert_voice_runtime_bridge_rejects_truthy_route_metadata()

    dry = speak_text("Jarvis voice dry run.", dry_run=True)
    print(f"[{'ok' if dry.ok else 'failed'}] dry voice")
    print(dry.output)
    print()
    if dry.metadata.get("speaks_audio") or dry.metadata.get("records_audio") or dry.metadata.get("starts_listener"):
        raise SystemExit("Dry-run speech should not speak, record, or start a listener.")
    for key in VOICE_SAFE_FALSE_FLAGS:
        if dry.metadata.get(key):
            raise SystemExit(f"Dry-run speech unsafe metadata {key}: {dry.metadata}")

    korean_dry = speak_text("현재 시간은 오후 11시 7분입니다.", dry_run=True)
    if not korean_dry.ok or korean_dry.metadata.get("voice") != "Yuna":
        raise SystemExit(f"Korean speech should select the macOS Korean voice: {korean_dry}")
    if "say -v Yuna <text>" not in korean_dry.output:
        raise SystemExit(f"Korean speech command should use Yuna: {korean_dry.output!r}")
    explicit_korean_dry = speak_text("현재 시간입니다.", voice="Explicit Voice", dry_run=True)
    if explicit_korean_dry.metadata.get("voice") != "Explicit Voice":
        raise SystemExit(f"Explicit voice should override Korean auto-selection: {explicit_korean_dry.metadata}")

    oversized_speech = speak_text("x" * (MAX_SPEECH_CHARS + 1), dry_run=True)
    if oversized_speech.ok or "oversized" not in oversized_speech.output:
        raise SystemExit("speak_text should refuse oversized speech text.")
    for key in VOICE_SAFE_FALSE_FLAGS:
        if oversized_speech.metadata.get(key):
            raise SystemExit(f"Oversized speech unsafe metadata {key}: {oversized_speech.metadata}")
    assert_known_local_recovery(
        oversized_speech,
        "oversized speech",
        action=voice_tools.LOCAL_READ_INPUT_RECOVERY_ACTION,
        commands=[],
    )

    missing_speech = speak_text("", dry_run=True)
    assert_known_local_recovery(
        missing_speech,
        "missing speech text",
        action=voice_tools.LOCAL_READ_INPUT_RECOVERY_ACTION,
        commands=[],
    )
    for key in VOICE_SAFE_FALSE_FLAGS:
        if missing_speech.metadata.get(key):
            raise SystemExit(f"Missing speech text unsafe metadata {key}: {missing_speech.metadata}")

    for path_text in [
        "/\x55sers/example/private/speech.txt",
        "/private/tmp/voice-speech.txt",
        "/var/folders/zc/jarvis/voice-speech.txt",
        "/tmp/jarvis-voice-speech.txt",
    ]:
        original_subprocess_run = voice_tools.subprocess.run
        try:
            def raise_if_speaks_path(*_args, **_kwargs):
                raise AssertionError("speak_text should reject path-shaped text before invoking say")

            voice_tools.subprocess.run = raise_if_speaks_path  # type: ignore[assignment]
            path_speech = speak_text(path_text, dry_run=False)
        finally:
            voice_tools.subprocess.run = original_subprocess_run  # type: ignore[assignment]
        if path_speech.ok or path_speech.metadata.get("reason") != "invalid_speech_text":
            raise SystemExit(f"speak_text should reject path-shaped speech text: {path_speech.metadata}")
        if path_speech.metadata.get("text") != "<local-path>" or path_speech.metadata.get("executes_tools"):
            raise SystemExit(f"path-shaped speech refusal should be local and redacted: {path_speech.metadata}")
        assert_no_local_path(path_speech.output, "path-shaped speech refusal output")
        assert_no_local_path(path_speech.metadata, "path-shaped speech refusal metadata")
        assert_known_local_recovery(
            path_speech,
            "path-shaped speech text",
            action=voice_tools.LOCAL_READ_INPUT_RECOVERY_ACTION,
            commands=[],
        )

    for preview_text in [
        "Please read /private/tmp/voice-preview.txt aloud",
        "Please read /var/folders/zc/jarvis/voice-preview.txt aloud",
        "Please read /tmp/jarvis-voice-preview.txt aloud",
    ]:
        path_preview = voice_reply_preview({"text": preview_text})
        if not path_preview.ok or "<local-path>" not in path_preview.output:
            raise SystemExit(f"voice reply preview should redact path-shaped display text: {path_preview.output}")
        assert_no_local_path(path_preview.output, "voice reply preview output")

    bad_rate = speak_text("Jarvis rate clamp dry run.", rate=9999, dry_run=True)
    if bad_rate.metadata.get("rate") != 360:
        raise SystemExit("speak_text should clamp oversized speech rates.")
    bool_rate = speak_text("Jarvis boolean rate dry run.", rate=True, dry_run=True)
    if bool_rate.metadata.get("rate") != 175:
        raise SystemExit(f"speak_text should treat boolean speech rates as malformed defaults: {bool_rate.metadata}")

    class FakeSpeechFailure:
        returncode = 1
        stdout = ""
        stderr = "say denied /\x55sers/example/private/speech"

    try:
        voice_tools.subprocess.run = lambda *_args, **_kwargs: FakeSpeechFailure()  # type: ignore[assignment]
        failed_speech = speak_text("Jarvis real speech failure.", dry_run=False)
        failed_voice_list = list_voices({})
    finally:
        voice_tools.subprocess.run = original_subprocess_run  # type: ignore[assignment]
    for label, result in {"speech": failed_speech, "list voices": failed_voice_list}.items():
        if result.ok or "Could not" not in result.output:
            raise SystemExit(f"{label} failure should return friendly output: {result.output}")
        for expected in ["System Settings > Accessibility > Spoken Content", "System Settings > Sound > Output", "voice setup check", "then retry"]:
            if expected not in result.output:
                raise SystemExit(f"{label} failure missed actionable guidance {expected}: {result.output}")
        if "/\x55sers/operator" in result.output or "denied" in result.output:
            raise SystemExit(f"{label} failure leaked raw stderr: {result.output}")
        if result.metadata.get("returncode") != 1:
            raise SystemExit(f"{label} failure missed returncode metadata: {result.metadata}")
    if failed_speech.metadata.get("executes_tools") is not True or failed_speech.metadata.get("speaks_audio"):
        raise SystemExit(f"speak_text failure should report attempted tool execution without successful speech: {failed_speech.metadata}")
    assert_known_local_recovery(
        failed_voice_list,
        "list voices nonzero",
        action=voice_tools.LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
        commands=["setup check"],
    )

    try:
        def raise_voice_failure(*_args, **_kwargs):
            raise RuntimeError("say missing near /\x55sers/example/private/speech")

        voice_tools.subprocess.run = raise_voice_failure  # type: ignore[assignment]
        exception_speech = speak_text("Jarvis exception speech failure.", dry_run=False)
        exception_voice_list = list_voices({})
    finally:
        voice_tools.subprocess.run = original_subprocess_run  # type: ignore[assignment]
    for label, result in {"speech exception": exception_speech, "list voices exception": exception_voice_list}.items():
        if result.ok or "Could not" not in result.output:
            raise SystemExit(f"{label} should return friendly output: {result.output}")
        for expected in ["System Settings > Accessibility > Spoken Content", "System Settings > Sound > Output", "voice setup check", "then retry"]:
            if expected not in result.output:
                raise SystemExit(f"{label} missed actionable guidance {expected}: {result.output}")
        if "/\x55sers/operator" in result.output or "say missing" in result.output:
            raise SystemExit(f"{label} leaked raw exception text: {result.output}")
        if result.metadata.get("exception_type") != "RuntimeError":
            raise SystemExit(f"{label} missed exception metadata: {result.metadata}")
    assert_known_local_recovery(
        exception_voice_list,
        "list voices exception",
        action=voice_tools.LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
        commands=["setup check"],
    )

    with TemporaryDirectory(prefix="jarvis-voice-") as temp:
        runtime = make_temp_runtime(Path(temp))
        sample_audio = Path(temp) / "sample.m4a"
        sample_audio.write_bytes(b"not real audio, metadata gate only")
        empty_voice_packet_args = {
            "transcript": "",
            "confirmed": "",
            "mode": "",
            "privacy_receipt_id": "",
            "receipt_id": "",
            "receipt_nonce": "",
        }
        route_cases = {
            "voice setup please": ("voice_setup_check", {}),
            "voice setup check please": ("voice_setup_check", {}),
            "can Jarvis listen yet": ("voice_setup_check", {}),
            "can Jarvis use the microphone": ("voice_setup_check", {}),
            "is voice input ready": ("voice_setup_check", {}),
            "is microphone input ready": ("voice_setup_check", {}),
            "microphone check please": ("voice_setup_check", {}),
            "mic check please": ("voice_setup_check", {}),
            "voice check please": ("voice_setup_check", {}),
            "speech check please": ("voice_setup_check", {}),
            "is speech input ready": ("voice_setup_check", {}),
            "is ASR ready": ("voice_setup_check", {}),
            "ASR check please": ("voice_setup_check", {}),
            "how does voice input work": ("voice_input_plan", {"mode": ""}),
            "how does spoken command work": ("voice_input_plan", {"mode": ""}),
            "can I talk to Jarvis": ("voice_input_plan", {"mode": ""}),
            "can I speak to Jarvis": ("voice_input_plan", {"mode": ""}),
            "can I use voice with Jarvis": ("voice_input_plan", {"mode": ""}),
            "how do I talk to Jarvis": ("voice_input_plan", {"mode": ""}),
            "how do I use voice with Jarvis": ("voice_input_plan", {"mode": ""}),
            "start voice input": ("voice_input_plan", {"mode": ""}),
            "start listening": ("voice_input_plan", {"mode": ""}),
            "listen to me": ("voice_input_plan", {"mode": ""}),
            "start recording voice": ("voice_input_plan", {"mode": ""}),
            "turn on microphone": ("voice_input_plan", {"mode": ""}),
            "open microphone": ("voice_input_plan", {"mode": ""}),
            "what is the voice input plan": ("voice_input_plan", {"mode": ""}),
            "voice cockpit please": ("voice_command_cockpit", empty_voice_packet_args),
            "voice command cockpit please": ("voice_command_cockpit", empty_voice_packet_args),
            "show voice cockpit": ("voice_command_cockpit", empty_voice_packet_args),
            "show latest voice cockpit": ("voice_command_cockpit", empty_voice_packet_args),
            "voice command lifecycle please": ("voice_command_lifecycle", {"transcript": ""}),
            "voice lifecycle please": ("voice_command_lifecycle", {"transcript": ""}),
            "show voice lifecycle": ("voice_command_lifecycle", {"transcript": ""}),
            "what happens after I speak to Jarvis": ("voice_command_lifecycle", {"transcript": ""}),
            "what is the voice proof chain": ("voice_command_lifecycle", {"transcript": ""}),
            "voice proof chain please": ("voice_command_lifecycle", {"transcript": ""}),
            "voice privacy please": ("voice_capture_privacy_packet", {"mode": "browser-push-to-talk"}),
            "voice privacy packet please": ("voice_capture_privacy_packet", {"mode": "browser-push-to-talk"}),
            "microphone privacy please": ("voice_capture_privacy_packet", {"mode": "browser-push-to-talk"}),
            "voice stop intent please": ("voice_stop_intent_packet", {"phrase": ""}),
            "voice stop please": ("voice_stop_intent_packet", {"phrase": ""}),
            "stop voice input": ("voice_stop_intent_packet", {"phrase": ""}),
            "cancel voice input": ("voice_stop_intent_packet", {"phrase": ""}),
            "rerecord voice input": ("voice_stop_intent_packet", {"phrase": ""}),
            "voice command audit please": ("voice_action_audit_packet", empty_voice_packet_args),
            "voice action audit please": ("voice_action_audit_packet", empty_voice_packet_args),
            "show latest voice action audit": ("voice_action_audit_packet", empty_voice_packet_args),
            "spoken turn rehearsal please": ("spoken_turn_rehearsal", {"message": "what should Jarvis do next"}),
            "spoken rehearsal please": ("spoken_turn_rehearsal", {"message": "what should Jarvis do next"}),
            "voice rehearsal please": ("spoken_turn_rehearsal", {"message": "what should Jarvis do next"}),
            "voice turn preview please": ("spoken_turn_rehearsal", {"message": "what should Jarvis do next"}),
            "preview spoken turn": ("spoken_turn_rehearsal", {"message": "what should Jarvis do next"}),
            "spoken turn rehearsal: can we talk about memory please": (
                "spoken_turn_rehearsal",
                {"message": "can we talk about memory please"},
            ),
            "voice cockpit: run command python3 --version please confirmed=true": (
                "voice_command_cockpit",
                {
                    "transcript": "run command python3 --version please",
                    "confirmed": "true",
                    "mode": "",
                    "privacy_receipt_id": "",
                    "receipt_id": "",
                    "receipt_nonce": "",
                },
            ),
        }
        for command, (expected_tool, expected_args) in route_cases.items():
            plan = runtime.planner.plan(command)
            actual = [(action.tool_name, action.args) for action in plan.actions]
            if actual != [(expected_tool, expected_args)]:
                raise SystemExit(f"Voice planner route mismatch for {command!r}: {actual}")

        for case in [
            "list tools voice",
            "voice setup check",
            "voice native microphone gate",
            "voice capture privacy",
            "voice reply preview: Jarvis is ready to continue safely",
            "spoken turn rehearsal: run command python3 --version",
            "spoken turn rehearsal: can we just talk about memory?",
            "voice transcript review: run command python3 --version",
            "voice transcript review: remember that voice transcripts are previewed first",
            "voice confirmation: run command python3 --version",
            "voice confirmation: remember that confirmed voice packets stay read only",
            "voice confirmation receipt: run command python3 --version confirmed=true",
            "voice confirmation receipt: remember that confirmed voice receipts stay read only",
            "voice confirmation audit ledger: run command python3 --version; confirmed=true",
            "voice route gate: run command python3 --version confirmed=true",
            "voice route gate: remember that route gate waits for confirmation",
            "voice route proof bundle: run command python3 --version confirmed=true",
            "voice route proof bundle: remember that route proof waits for confirmation",
            "voice runtime bridge: run command python3 --version confirmed=true",
            "voice command intake bridge: remember that runtime bridge waits for confirmation",
            "voice command cockpit: run command python3 --version confirmed=true",
            "voice command cockpit: remember that voice cockpit waits for confirmation",
            "voice action audit: run command python3 --version confirmed=true",
            "voice action audit: remember that voice action audits wait for confirmation",
            "voice execution handoff: run command python3 --version confirmed=true",
            "voice command handoff: remember that voice handoffs wait for confirmation",
            f"voice post-run closure: run command python3 --version; confirmed=true; verification reviewed; post health reviewed; post audit reviewed; learning reviewed; {VOICE_POST_RUN_HASH_SUFFIX}",
            "voice post-run closure: remember that voice closures wait for proof; confirmed=true; verification reviewed",
            f"voice cycle ledger: run command python3 --version; confirmed=true; verification reviewed; post health reviewed; post audit reviewed; learning reviewed; {VOICE_POST_RUN_HASH_SUFFIX}",
            "voice cycle ledger: remember that voice cycle ledgers wait for proof; confirmed=true; verification reviewed",
            "voice stop intent: stop listening",
            "voice rerecord intent: rerecord this command",
            "send confirmed transcript: run command python3 --version",
            "voice command lifecycle",
            "voice command lifecycle: run command python3 --version",
            "voice file transcription plan",
            "voice file transcription plan: /tmp/jarvis-test-audio.m4a",
            "voice note plan",
            "voice note plan: /tmp/jarvis-test-audio.m4a",
            f"voice audio file gate: {sample_audio} consent=true",
            f"voice note gate: {sample_audio} consent=true",
            f"voice audio file gate: {sample_audio}",
            "voice audio file gate: /tmp/jarvis-test-audio.txt consent=true",
            "voice input plan push-to-talk",
            "wake-word plan",
            "list voices",
        ]:
            result = runtime.handle(case)
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:1600])
            print()
            if case == "list tools voice":
                required = [
                    "voice_command_lifecycle",
                    "proof chain",
                    "confirmation receipt",
                    "runtime bridge",
                    "handoff",
                    "closure",
                    "voice_cycle_ledger",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Voice tool catalog missing lifecycle proof-chain text: {missing}")
            if case == "voice setup check":
                required = [
                    "Jarvis voice setup check",
                    "Audio and ASR dependencies",
                    "Whisper CLI configuration",
                    "- active source:",
                    "- model: base",
                    "- language: automatic detection",
                    "Readiness",
                    "Whisper warm start",
                    "generated 0.3s silent WAV",
                    "microphone permission state: not requested",
                    "read-only",
                    "does not request microphone access",
                    "record audio",
                    "Safe next build order",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Voice setup check missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                if metadata.get("records_audio") or metadata.get("starts_listener"):
                    raise SystemExit("Voice setup check must not record audio or start a listener.")
                if metadata.get("voice_warmup_uses_silent_wav") is not True or metadata.get("voice_warmup_touches_microphone") is not False:
                    raise SystemExit(f"Voice setup check missed warmup microphone boundary: {metadata}")
                if metadata.get("voice_warmup_enabled") is not _env_truthy(os.environ.get("JARVIS_VOICE_WARMUP")):
                    raise SystemExit(f"Voice setup check did not reflect JARVIS_VOICE_WARMUP: {metadata}")
                for key in VOICE_SAFE_FALSE_FLAGS:
                    if metadata.get(key):
                        raise SystemExit(f"Voice setup check unsafe metadata {key}: {metadata}")
            if case == "voice native microphone gate":
                required = [
                    "Jarvis native microphone gate packet",
                    "Capture gate receipt",
                    "Gate decision",
                    "NATIVE_MIC_HELD_FOR_PERMISSION_RECEIPT",
                    "Required before any native capture runner",
                    "Required after transcript creation",
                    "Native/offline microphone capture remains PERSONAL_DATA",
                    "does not request microphone access",
                    "record audio",
                    "queue approvals",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Voice native microphone gate missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                assert_voice_native_microphone_gate(
                    metadata,
                    "Voice native microphone gate held case",
                    expected_gate_state="NATIVE_MIC_HELD_FOR_PERMISSION_RECEIPT",
                )
                if metadata.get("native_microphone_capture_enabled") is not False:
                    raise SystemExit(f"Native microphone gate must keep native capture disabled: {metadata}")
                if metadata.get("capture_runner_requires_approval") is not True or metadata.get("capture_runner_risk_level") != "PERSONAL_DATA":
                    raise SystemExit(f"Native microphone gate missed approval/risk boundary: {metadata}")
                if metadata.get("transcript_confirmation_required") is not True or metadata.get("receipt_nonce_required") is not True or metadata.get("route_proof_bundle_required") is not True or metadata.get("post_run_closure_required") is not True:
                    raise SystemExit(f"Native microphone gate missed transcript proof chain: {metadata}")
                if "visible microphone permission receipt" not in metadata.get("missing_checks", []):
                    raise SystemExit(f"Native microphone gate should hold without a permission receipt: {metadata}")
                for key in VOICE_SAFE_FALSE_FLAGS:
                    if metadata.get(key):
                        raise SystemExit(f"Voice native microphone gate unsafe metadata {key}: {metadata}")
                ready_metadata = voice_native_microphone_gate_packet(
                    {
                        "mode": "native-push-to-talk",
                        "permission_receipt_id": "visible-permission-receipt-1",
                        "visible_state": "reviewed",
                    }
                ).metadata
                assert_voice_native_microphone_gate(
                    ready_metadata,
                    "Voice native microphone gate ready-for-approved-runner case",
                    expected_gate_state="NATIVE_MIC_READY_FOR_APPROVED_CAPTURE_RUNNER",
                )
                if ready_metadata.get("native_microphone_capture_enabled") is not False or ready_metadata.get("records_audio") is not False:
                    raise SystemExit(f"Ready native microphone gate must still keep capture disabled: {ready_metadata}")
            if case == "voice capture privacy":
                required = [
                    "Jarvis voice capture privacy packet",
                    "read-only",
                    "Privacy receipt id",
                    "Capture mode",
                    "Visible indicators required before live capture",
                    "transcript stays in the composer until the operator presses Send",
                    "Send creates an auditable confirmation receipt before normal routing",
                    "Browser microphone access is requested only by the visible push-to-talk control",
                    "Audio remains browser/transient",
                    "Composer route contract",
                    "voice privacy state sequence: idle -> armed -> listening -> confirm -> confirming -> routing -> stopped -> error",
                    "Send without a confirmation receipt must keep the transcript in the composer and block routing",
                    "Mic rerecord clears only the local draft",
                    "Stop capture only stops listening or preserves the draft",
                    "Route guard commands",
                    "does not request microphone access",
                    "recording audio",
                    "queuing approvals",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Voice capture privacy missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                if not metadata.get("privacy_receipt_id") or not metadata.get("visible_capture_indicator_required"):
                    raise SystemExit(f"Voice capture privacy missed receipt/indicator metadata: {metadata}")
                direct_bool_duration = voice_capture_privacy_packet({"max_duration_seconds": True})
                if not direct_bool_duration.ok or direct_bool_duration.metadata.get("max_duration_seconds") != 30:
                    raise SystemExit(f"Voice capture privacy should treat boolean durations as malformed defaults: {direct_bool_duration.metadata}")
                if metadata.get("microphone_access_requested") or metadata.get("raw_audio_stored"):
                    raise SystemExit(f"Voice capture privacy must not request mic access or store raw audio: {metadata}")
                if not metadata.get("push_to_talk_required") or not metadata.get("transcript_confirmation_required") or not metadata.get("confirmation_receipt_required"):
                    raise SystemExit(f"Voice capture privacy missed route gates: {metadata}")
                expected_states = ["idle", "armed", "listening", "confirm", "confirming", "routing", "stopped", "error"]
                if metadata.get("composer_state_sequence") != expected_states:
                    raise SystemExit(f"Voice capture privacy missed composer state sequence: {metadata}")
                if metadata.get("composer_state_count") != len(expected_states):
                    raise SystemExit(f"Voice capture privacy missed composer state count: {metadata}")
                if (
                    metadata.get("send_requires_confirmation_receipt") is not True
                    or metadata.get("send_without_receipt_routes") is not False
                    or metadata.get("voice_draft_persists_until_confirmation") is not True
                    or metadata.get("rerecord_clears_local_draft_only") is not True
                    or metadata.get("stop_capture_only") is not True
                    or metadata.get("stop_intent_routes") is not False
                    or metadata.get("stop_intent_deletes") is not False
                    or metadata.get("stop_intent_approves") is not False
                ):
                    raise SystemExit(f"Voice capture privacy missed composer route contract flags: {metadata}")
                expected_guard_commands = [
                    "voice transcript review: <transcript>",
                    "voice confirmation: <transcript>",
                    "voice confirmation receipt: <transcript> confirmed=true",
                    "voice route gate: <transcript> confirmed=true",
                ]
                if metadata.get("route_guard_commands") != expected_guard_commands:
                    raise SystemExit(f"Voice capture privacy missed route guard commands: {metadata}")
                for key in VOICE_SAFE_FALSE_FLAGS:
                    if metadata.get(key):
                        raise SystemExit(f"Voice capture privacy unsafe metadata {key}: {metadata}")
            if case.startswith("voice reply preview"):
                required = [
                    "Jarvis voice reply preview",
                    "read-only",
                    "without speaking",
                    "Text to speak",
                    "Jarvis is ready to continue safely",
                    "Speech settings",
                    "estimated duration",
                    "command preview",
                    "Safety boundary",
                    "preview before speaking sensitive text",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Voice reply preview missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                for key in VOICE_SAFE_FALSE_FLAGS:
                    if metadata.get(key):
                        raise SystemExit(f"Voice reply preview unsafe metadata {key}: {metadata}")
                direct_bad_rate = voice_reply_preview({"text": "Jarvis rate preview", "rate": "bad"})
                if not direct_bad_rate.ok or direct_bad_rate.metadata.get("rate") != 175:
                    raise SystemExit("Voice reply preview should sanitize bad rates.")
                direct_bool_rate = voice_reply_preview({"text": "Jarvis rate preview", "rate": False})
                if not direct_bool_rate.ok or direct_bool_rate.metadata.get("rate") != 175:
                    raise SystemExit(f"Voice reply preview should treat boolean rates as malformed defaults: {direct_bool_rate.metadata}")
                direct_oversized = voice_reply_preview({"text": "x" * (MAX_SPEECH_CHARS + 1)})
                if direct_oversized.ok or "too large" not in direct_oversized.output:
                    raise SystemExit("Voice reply preview should refuse oversized text.")
            if case.startswith("spoken turn rehearsal"):
                required = [
                    "Jarvis spoken turn rehearsal",
                    "read-only",
                    "User message",
                    "Route preview",
                    "Spoken-response preview",
                    "Jarvis voice reply preview",
                    "without speaking",
                    "Safety boundary",
                    "does not call a chat model",
                    "execute tools",
                    "queue approvals",
                ]
                if "run command" in case:
                    required.extend(["mode: tool", "run_shell_command", "HIGH_RISK", "approval required", "last-look packet"])
                else:
                    required.extend(["mode: chat", "Route to ChatBrain"])
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Spoken turn rehearsal missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                if "run command" in case and not metadata.get("approval_required"):
                    raise SystemExit("Spoken turn rehearsal should flag approval for high-risk actions.")
                if "can we just talk" in case and metadata.get("mode") != "chat":
                    raise SystemExit("Spoken turn rehearsal should route natural questions to chat.")
                for key in VOICE_SAFE_FALSE_FLAGS:
                    if metadata.get(key):
                        raise SystemExit(f"Spoken turn rehearsal unsafe metadata {key}: {metadata}")
            if case.startswith("voice transcript review"):
                required = [
                    "Jarvis voice transcript review",
                    "Transcript",
                    "Recognition check",
                    "Planned action preview",
                    "Next safe commands after the operator confirms",
                    "Auditable confirmation handoff",
                    "transcript hash",
                    "receipt command",
                    "route blocker",
                    "Safety boundary",
                    "does not execute tools",
                    "record audio",
                    "start a listener",
                ]
                if "run command" in case:
                    required.extend(["run_shell_command", "HIGH_RISK", "approval required", "shell/code", "approval readiness latest", "approval packet latest", "approval chain proof latest"])
                if "remember" in case:
                    required.extend(["remember", "LOCAL_SAFE", "would be allowed by default"])
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Voice transcript review missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                if "run command" in case:
                    assert_voice_approval_chain(metadata, "Voice transcript review")
                    assert_voice_confirmation_handoff(metadata, "Voice transcript review", "run command python3 --version")
                if "remember" in case:
                    assert_voice_confirmation_handoff(metadata, "Voice transcript review", "remember that voice transcripts are previewed first")
                for key in VOICE_SAFE_FALSE_FLAGS:
                    if metadata.get(key):
                        raise SystemExit(f"Voice transcript review unsafe metadata {key}: {metadata}")
            if case.startswith("voice route gate") or case.startswith("send confirmed transcript"):
                required = [
                    "Jarvis voice route gate packet",
                    "read-only",
                    "Transcript receipt",
                    "Route gate",
                    "can enter runtime now",
                    "can auto-execute now",
                    "approval required after routing",
                    "Planned action preview",
                    "Next safe commands",
                    "confirmed transcript receipt",
                    "Safety boundary",
                    "does not request microphone access",
                    "record audio",
                    "execute tools",
                    "queue approvals",
                ]
                if "run command" in case:
                    required.extend(["VOICE_READY_FOR_PLANNER", "approval_review", "run_shell_command", "HIGH_RISK", "approval readiness latest", "approval packet latest", "approval chain proof latest"])
                if "remember" in case:
                    required.extend(["VOICE_HELD_FOR_CONFIRMATION", "voice_confirmation_hold", "keep it in the composer", "voice confirmation receipt: remember that route gate waits for confirmation confirmed=true"])
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Voice route gate missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                if "run command" in case:
                    assert_voice_confirmation_handoff(metadata, "Voice route gate", "run command python3 --version", confirmed=True)
                    if metadata.get("runtime_route_after_gate") != "approval_review" or metadata.get("can_enter_runtime") is not True:
                        raise SystemExit(f"Voice route gate missed approved routing state: {metadata}")
                    if metadata.get("can_auto_execute_now") is not False or metadata.get("approval_required_after_routing") is not True:
                        raise SystemExit(f"Voice route gate should not auto-execute risky voice command: {metadata}")
                if "remember" in case:
                    assert_voice_confirmation_handoff(metadata, "Voice route gate", "remember that route gate waits for confirmation")
                    if metadata.get("can_enter_runtime") or metadata.get("can_route_after_confirmation_receipt"):
                        raise SystemExit(f"Unconfirmed voice route gate should block routing: {metadata}")
                for key in VOICE_SAFE_FALSE_FLAGS:
                    if metadata.get(key):
                        raise SystemExit(f"Voice route gate unsafe metadata {key}: {metadata}")
            if case.startswith("voice route proof bundle"):
                required = [
                    "Jarvis voice route proof bundle",
                    "read-only",
                    "Bundle state",
                    "Privacy proof",
                    "Transcript identity",
                    "Confirmation and route proof",
                    "Required proof commands",
                    "Next safe commands",
                    "Safety boundary",
                    "does not request microphone access",
                    "record audio",
                    "read audio files",
                    "execute tools",
                    "queue approvals",
                ]
                if "run command" in case:
                    required.extend([
                        "VOICE_ROUTE_PROOF_READY",
                        "missing proof: none",
                        "approval required after routing: yes",
                        "privacy receipt match: yes",
                        "receipt hash match: yes",
                        "route-gate hash match: yes",
                        "approval readiness latest",
                        "approval packet latest",
                        "approval chain proof latest",
                    ])
                if "remember" in case:
                    required.extend([
                        "VOICE_ROUTE_PROOF_HELD",
                        "confirmed transcript receipt",
                        "VOICE_HELD_FOR_CONFIRMATION",
                        "voice confirmation receipt: remember that route proof waits for confirmation confirmed=true",
                    ])
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Voice route proof bundle missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                if not metadata.get("auditable") or metadata.get("routes_actions"):
                    raise SystemExit(f"Voice route proof bundle should be auditable without routing actions: {metadata}")
                if metadata.get("proof_command_count") != 5 or len(metadata.get("proof_commands") or []) != 5:
                    raise SystemExit(f"Voice route proof bundle missed proof commands: {metadata}")
                if "run command" in case:
                    assert_voice_approval_chain(metadata, "Voice route proof bundle")
                    assert_voice_confirmation_handoff(metadata, "Voice route proof bundle", "run command python3 --version", confirmed=True)
                    assert_voice_privacy_receipt_chain(metadata, "Voice route proof bundle")
                    assert_voice_command_intake_contract(metadata, "Voice route proof bundle")
                    if metadata.get("bundle_state") != "VOICE_ROUTE_PROOF_READY" or metadata.get("bundle_ready") is not True:
                        raise SystemExit(f"Voice route proof bundle missed ready state: {metadata}")
                    if metadata.get("runtime_route_after_gate") != "approval_review" or metadata.get("can_enter_runtime") is not True:
                        raise SystemExit(f"Voice route proof bundle missed approval-review routing: {metadata}")
                    if metadata.get("can_auto_execute_now") is not False or metadata.get("approval_required_after_routing") is not True:
                        raise SystemExit(f"Voice route proof bundle should hold risky execution behind approval: {metadata}")
                    if metadata.get("receipt_hash_match") is not True or metadata.get("route_gate_hash_match") is not True:
                        raise SystemExit(f"Voice route proof bundle missed hash match proof: {metadata}")
                if "remember" in case:
                    assert_voice_confirmation_handoff(metadata, "Voice route proof bundle", "remember that route proof waits for confirmation")
                    if metadata.get("bundle_state") != "VOICE_ROUTE_PROOF_HELD" or metadata.get("bundle_ready"):
                        raise SystemExit(f"Unconfirmed voice proof bundle should be held: {metadata}")
                    if metadata.get("can_enter_runtime") or metadata.get("can_route_after_confirmation_receipt"):
                        raise SystemExit(f"Unconfirmed voice proof bundle should block routing: {metadata}")
                for key in VOICE_SAFE_FALSE_FLAGS:
                    if metadata.get(key):
                        raise SystemExit(f"Voice route proof bundle unsafe metadata {key}: {metadata}")
            if case.startswith("voice runtime bridge") or case.startswith("voice command intake bridge"):
                required = [
                    "Jarvis voice runtime bridge packet",
                    "read-only",
                    "command-intake proof only",
                    "Bridge state",
                    "runtime entry mode",
                    "Voice proof carried forward",
                    "Command-intake bridge",
                    "command intake",
                    "dispatch decision",
                    "execution readiness matrix",
                    "verification packet",
                    "risk preflight",
                    "Required proof commands",
                    "Next safe commands",
                    "Safety boundary",
                    "never sends the transcript as an executable tool action",
                    "does not route actions",
                    "execute tools",
                    "queue approvals",
                    "request microphone access",
                ]
                if "run command" in case:
                    required.extend([
                        "VOICE_READY_FOR_COMMAND_INTAKE",
                        "command_intake_only",
                        "missing proof: none",
                        "approval required after bridge: yes",
                        "run_shell_command",
                        "HIGH_RISK",
                        "approval readiness latest",
                        "approval packet latest",
                        "approval chain proof latest",
                    ])
                if "remember" in case:
                    required.extend([
                        "VOICE_HELD_BEFORE_COMMAND_INTAKE",
                        "held_for_voice_confirmation",
                        "confirmed transcript receipt",
                        "voice confirmation receipt: remember that runtime bridge waits for confirmation confirmed=true",
                    ])
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Voice runtime bridge missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                if not metadata.get("auditable") or metadata.get("routes_actions") or metadata.get("can_emit_executable_tool_action"):
                    raise SystemExit(f"Voice runtime bridge should be auditable without executable action routing: {metadata}")
                if metadata.get("can_auto_execute_now") is not False:
                    raise SystemExit(f"Voice runtime bridge must never auto-execute: {metadata}")
                for expected_flag in ["command_intake_required", "dispatch_decision_required", "execution_readiness_matrix_required", "verification_packet_required"]:
                    if metadata.get(expected_flag) is not True:
                        raise SystemExit(f"Voice runtime bridge missed required flag {expected_flag}: {metadata}")
                if "run command" in case:
                    assert_voice_confirmation_handoff(metadata, "Voice runtime bridge", "run command python3 --version", confirmed=True)
                    assert_voice_privacy_receipt_chain(metadata, "Voice runtime bridge")
                    assert_voice_command_intake_contract(metadata, "Voice runtime bridge")
                    if metadata.get("bridge_state") != "VOICE_READY_FOR_COMMAND_INTAKE" or metadata.get("runtime_entry_mode") != "command_intake_only":
                        raise SystemExit(f"Voice runtime bridge missed command-intake ready state: {metadata}")
                    if metadata.get("can_enter_runtime") is not True or metadata.get("approval_required_after_bridge") is not True:
                        raise SystemExit(f"Voice runtime bridge missed approval-required runtime state: {metadata}")
                    if metadata.get("risk_preflight_required") is not True:
                        raise SystemExit(f"Risky voice runtime bridge should require risk preflight: {metadata}")
                    for command in [
                        "voice route proof bundle: run command python3 --version confirmed=true",
                        "command intake: run command python3 --version",
                        "dispatch decision: run command python3 --version",
                        "execution readiness matrix: run command python3 --version",
                        "verification packet: run command python3 --version",
                        "approval readiness latest",
                        "approval packet latest",
                        "approval chain proof latest",
                    ]:
                        if command not in metadata.get("proof_commands", []):
                            raise SystemExit(f"Voice runtime bridge missed proof command {command!r}: {metadata}")
                if "remember" in case:
                    assert_voice_confirmation_handoff(metadata, "Voice runtime bridge", "remember that runtime bridge waits for confirmation")
                    if metadata.get("bridge_state") != "VOICE_HELD_BEFORE_COMMAND_INTAKE" or metadata.get("runtime_entry_mode") != "held_for_voice_confirmation":
                        raise SystemExit(f"Unconfirmed voice runtime bridge should be held: {metadata}")
                    if metadata.get("can_enter_runtime") or metadata.get("can_route_after_confirmation_receipt"):
                        raise SystemExit(f"Unconfirmed voice runtime bridge should block runtime entry: {metadata}")
                    if any(str(command).startswith("command intake:") for command in metadata.get("recommended_next_commands", [])):
                        raise SystemExit(f"Unconfirmed voice runtime bridge must not suggest command intake yet: {metadata}")
                for key in VOICE_SAFE_FALSE_FLAGS:
                    if metadata.get(key):
                        raise SystemExit(f"Voice runtime bridge unsafe metadata {key}: {metadata}")
            if case.startswith("voice command cockpit"):
                required = [
                    "Jarvis voice command cockpit",
                    "read-only",
                    "Cockpit state",
                    "transcript hash",
                    "receipt route state",
                    "route gate state",
                    "route proof bundle state",
                    "runtime bridge state",
                    "runtime entry mode",
                    "hash match across artifacts",
                    "ready for command intake",
                    "can auto-execute now: no",
                    "Required proof chain",
                    "Safety boundary",
                    "never emits an executable tool action",
                    "dispatch decision",
                    "execution readiness matrix",
                    "approval gates",
                    "verification",
                ]
                if "run command" in case:
                    required.extend([
                        "VOICE_COMMAND_COCKPIT_READY_FOR_COMMAND_INTAKE",
                        "VOICE_READY_FOR_COMMAND_INTAKE",
                        "command_intake_only",
                        "missing blockers: none",
                        "approval required after bridge: yes",
                        "command intake: run command python3 --version",
                    ])
                if "remember" in case:
                    required.extend([
                        "VOICE_COMMAND_COCKPIT_HELD",
                        "VOICE_HELD_BEFORE_COMMAND_INTAKE",
                        "confirmed transcript receipt",
                        "voice confirmation receipt: remember that voice cockpit waits for confirmation confirmed=true",
                    ])
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Voice command cockpit missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                if not metadata.get("auditable") or metadata.get("routes_actions") or metadata.get("can_emit_executable_tool_action"):
                    raise SystemExit(f"Voice command cockpit should be auditable without executable action routing: {metadata}")
                if metadata.get("can_auto_execute_now") is not False:
                    raise SystemExit(f"Voice command cockpit must never auto-execute: {metadata}")
                for nested_key in ["receipt_metadata", "route_gate_metadata", "route_proof_bundle_metadata", "runtime_bridge_metadata"]:
                    if not isinstance(metadata.get(nested_key), dict):
                        raise SystemExit(f"Voice command cockpit missed nested {nested_key}: {metadata}")
                if "run command" in case:
                    assert_voice_confirmation_handoff(metadata, "Voice command cockpit", "run command python3 --version", confirmed=True)
                    assert_voice_privacy_receipt_chain(metadata, "Voice command cockpit")
                    assert_voice_command_intake_contract(metadata, "Voice command cockpit")
                    if metadata.get("cockpit_state") != "VOICE_COMMAND_COCKPIT_READY_FOR_COMMAND_INTAKE" or metadata.get("ready_for_command_intake") is not True:
                        raise SystemExit(f"Voice command cockpit missed command-intake ready state: {metadata}")
                    if metadata.get("runtime_bridge_state") != "VOICE_READY_FOR_COMMAND_INTAKE" or metadata.get("runtime_entry_mode") != "command_intake_only":
                        raise SystemExit(f"Voice command cockpit missed runtime bridge ready state: {metadata}")
                    if metadata.get("hash_match") is not True or metadata.get("approval_required_after_bridge") is not True:
                        raise SystemExit(f"Voice command cockpit missed hash/approval metadata: {metadata}")
                if "remember" in case:
                    assert_voice_confirmation_handoff(metadata, "Voice command cockpit", "remember that voice cockpit waits for confirmation")
                    if metadata.get("cockpit_state") != "VOICE_COMMAND_COCKPIT_HELD" or metadata.get("ready_for_command_intake"):
                        raise SystemExit(f"Unconfirmed voice command cockpit should be held: {metadata}")
                    if any(str(command).startswith("command intake:") for command in metadata.get("required_commands", [])):
                        raise SystemExit(f"Unconfirmed voice command cockpit must not include command intake proof yet: {metadata}")
                for key in VOICE_SAFE_FALSE_FLAGS:
                    if metadata.get(key):
                        raise SystemExit(f"Voice command cockpit unsafe metadata {key}: {metadata}")
            if case.startswith("voice action audit"):
                required = [
                    "Jarvis voice action audit packet",
                    "read-only",
                    "last transcript-confirmation audit",
                    "Audit state",
                    "transcript hash",
                    "cockpit state",
                    "runtime bridge state",
                    "runtime entry mode",
                    "hash match across artifacts",
                    "ready for command intake review",
                    "action allowed now: no",
                    "executable tool action emitted: no",
                    "Operator review boundary",
                    "command intake, dispatch decision, execution readiness matrix, and verification packet are still required",
                    "Required proof chain",
                    "Safety boundary",
                    "never treats speech as permission to bypass ToolRegistry",
                ]
                if "run command" in case:
                    required.extend([
                        "VOICE_ACTION_AUDIT_READY_FOR_OPERATOR_REVIEW",
                        "VOICE_COMMAND_COCKPIT_READY_FOR_COMMAND_INTAKE",
                        "VOICE_READY_FOR_COMMAND_INTAKE",
                        "command_intake_only",
                        "missing blockers: none",
                        "approval required after command intake: yes",
                        "approval readiness latest",
                        "approval packet latest",
                        "approval chain proof latest",
                    ])
                if "remember" in case:
                    required.extend([
                        "VOICE_ACTION_AUDIT_HELD",
                        "VOICE_COMMAND_COCKPIT_HELD",
                        "confirmed transcript receipt",
                        "voice confirmation receipt: remember that voice action audits wait for confirmation confirmed=true",
                    ])
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Voice action audit missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                if not metadata.get("auditable") or metadata.get("routes_actions"):
                    raise SystemExit(f"Voice action audit should be auditable without routing actions: {metadata}")
                if metadata.get("action_allowed_now") or metadata.get("executable_tool_action_emitted") or metadata.get("can_emit_executable_tool_action") or metadata.get("can_auto_execute_now"):
                    raise SystemExit(f"Voice action audit must never allow direct speech execution: {metadata}")
                if "run command" in case:
                    assert_voice_confirmation_handoff(metadata, "Voice action audit", "run command python3 --version", confirmed=True)
                    assert_voice_privacy_receipt_chain(metadata, "Voice action audit")
                    assert_voice_command_intake_contract(metadata, "Voice action audit")
                    if metadata.get("audit_state") != "VOICE_ACTION_AUDIT_READY_FOR_OPERATOR_REVIEW" or metadata.get("audit_ready_for_operator_review") is not True:
                        raise SystemExit(f"Voice action audit missed ready state: {metadata}")
                    for command in [
                        "voice action audit: run command python3 --version confirmed=true",
                        "command intake: run command python3 --version",
                        "dispatch decision: run command python3 --version",
                        "execution readiness matrix: run command python3 --version",
                        "verification packet: run command python3 --version",
                        "approval readiness latest",
                        "approval packet latest",
                        "approval chain proof latest",
                    ]:
                        if command not in metadata.get("required_commands", []):
                            raise SystemExit(f"Voice action audit missed proof command {command!r}: {metadata}")
                if "remember" in case:
                    assert_voice_confirmation_handoff(metadata, "Voice action audit", "remember that voice action audits wait for confirmation")
                    if metadata.get("audit_state") != "VOICE_ACTION_AUDIT_HELD" or metadata.get("audit_ready_for_operator_review"):
                        raise SystemExit(f"Unconfirmed voice action audit should be held: {metadata}")
                    if any(str(command).startswith("command intake:") for command in metadata.get("required_commands", [])):
                        raise SystemExit(f"Unconfirmed voice action audit must not include command intake proof yet: {metadata}")
                for key in VOICE_SAFE_FALSE_FLAGS:
                    if metadata.get(key):
                        raise SystemExit(f"Voice action audit unsafe metadata {key}: {metadata}")
            if case.startswith("voice execution handoff") or case.startswith("voice command handoff"):
                required = [
                    "Jarvis voice execution handoff packet",
                    "read-only",
                    "Command-intake handoff",
                    "Handoff state",
                    "transcript hash",
                    "voice action audit state",
                    "ready for command intake packet",
                    "action allowed now: no",
                    "executable tool action emitted: no",
                    "Required pre-run proof chain",
                    "Post-run proof required after any approved spoken action",
                    "verification receipt latest",
                    "execution audit",
                    "after action learning",
                    "Safety boundary",
                    "never treats speech as permission to bypass ToolRegistry",
                ]
                if "run command" in case:
                    required.extend([
                        "VOICE_EXECUTION_HANDOFF_READY_FOR_COMMAND_INTAKE_PACKET",
                        "VOICE_ACTION_AUDIT_READY_FOR_OPERATOR_REVIEW",
                        "missing blockers: none",
                        "approval required after command intake: yes",
                        "approval readiness latest",
                        "approval packet latest",
                        "approval chain proof latest",
                    ])
                if "remember" in case:
                    required.extend([
                        "VOICE_EXECUTION_HANDOFF_HELD",
                        "confirmed transcript receipt",
                        "voice confirmation receipt: remember that voice handoffs wait for confirmation confirmed=true",
                    ])
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Voice execution handoff missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                if not metadata.get("auditable") or metadata.get("routes_actions"):
                    raise SystemExit(f"Voice execution handoff should be auditable without routing actions: {metadata}")
                if metadata.get("action_allowed_now") or metadata.get("executable_tool_action_emitted") or metadata.get("can_emit_executable_tool_action") or metadata.get("can_auto_execute_now"):
                    raise SystemExit(f"Voice execution handoff must never allow direct speech execution: {metadata}")
                if "run command" in case:
                    assert_voice_confirmation_handoff(metadata, "Voice execution handoff", "run command python3 --version", confirmed=True)
                    assert_voice_privacy_receipt_chain(metadata, "Voice execution handoff")
                    assert_voice_command_intake_contract(metadata, "Voice execution handoff")
                    if metadata.get("handoff_state") != "VOICE_EXECUTION_HANDOFF_READY_FOR_COMMAND_INTAKE_PACKET" or metadata.get("ready_for_command_intake_packet") is not True:
                        raise SystemExit(f"Voice execution handoff missed ready state: {metadata}")
                    for command in [
                        "voice execution handoff: run command python3 --version confirmed=true",
                        "command intake: run command python3 --version",
                        "dispatch decision: run command python3 --version",
                        "execution readiness matrix: run command python3 --version",
                        "verification packet: run command python3 --version",
                        "approval readiness latest",
                        "approval packet latest",
                        "approval chain proof latest",
                    ]:
                        if command not in metadata.get("required_commands", []):
                            raise SystemExit(f"Voice execution handoff missed proof command {command!r}: {metadata}")
                    for command in metadata.get("post_run_commands", []):
                        if "spoken command" not in str(command):
                            raise SystemExit(f"Voice execution handoff post-run command missed spoken command hash: {metadata}")
                if "remember" in case:
                    assert_voice_confirmation_handoff(metadata, "Voice execution handoff", "remember that voice handoffs wait for confirmation")
                    if metadata.get("handoff_state") != "VOICE_EXECUTION_HANDOFF_HELD" or metadata.get("ready_for_command_intake_packet"):
                        raise SystemExit(f"Unconfirmed voice execution handoff should be held: {metadata}")
                    if any(str(command).startswith("command intake:") for command in metadata.get("required_commands", [])):
                        raise SystemExit(f"Unconfirmed voice execution handoff must not include command intake proof yet: {metadata}")
                for key in VOICE_SAFE_FALSE_FLAGS:
                    if metadata.get(key):
                        raise SystemExit(f"Voice execution handoff unsafe metadata {key}: {metadata}")
            if case.startswith("voice post-run closure"):
                required = [
                    "Jarvis voice post-run closure packet",
                    "read-only",
                    "closes the proof loop",
                    "Closure state",
                    "transcript hash",
                    "voice execution handoff ready",
                    "post-run verification reviewed",
                    "post-run execution health reviewed",
                    "post-run execution audit reviewed",
                    "after-action learning reviewed",
                    "verification receipt sha256 present",
                    "execution health sha256 present",
                    "execution audit sha256 present",
                    "after-action learning sha256 present",
                    "post-run artifact hashes present",
                    "ready for next voice review",
                    "action allowed now: no",
                    "executable tool action emitted: no",
                    "Continuation boundary",
                    "next voice review state",
                    "fresh confirmation receipt required before next spoken order: yes",
                    "fresh receipt nonce required before next spoken order: yes",
                    "previous confirmation receipt reusable for next voice review: no",
                    "previous transcript hash reusable for next voice review: no",
                    "previous execution handoff reusable for next voice review: no",
                    "prior spoken command proof only: yes",
                    "Required closure proof chain",
                    "Next safe command",
                    "Safety boundary",
                    "does not record audio",
                    "execute tools",
                    "queue approvals",
                ]
                if "run command" in case:
                    required.extend(
                        [
                            "VOICE_POST_RUN_CLOSURE_READY_FOR_NEXT_VOICE_REVIEW",
                            "verification receipt sha256 present: yes",
                            "execution health sha256 present: yes",
                            "execution audit sha256 present: yes",
                            "after-action learning sha256 present: yes",
                            "post-run artifact hashes present: yes",
                            "missing blockers: none",
                            "voice execution handoff: run command python3 --version confirmed=true",
                            "verification receipt latest",
                            "execution audit",
                            "after action learning",
                        ]
                    )
                if "remember" in case:
                    required.extend(
                        [
                            "VOICE_POST_RUN_CLOSURE_HELD",
                            "verification receipt sha256 present: no",
                            "execution health sha256 present: no",
                            "execution audit sha256 present: no",
                            "after-action learning sha256 present: no",
                            "post-run artifact hashes present: no",
                            "post-run execution health evidence",
                            "post-run execution audit evidence",
                            "after-action learning evidence",
                        ]
                    )
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Voice post-run closure missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                if not metadata.get("auditable") or metadata.get("routes_actions"):
                    raise SystemExit(f"Voice post-run closure should be auditable without routing actions: {metadata}")
                if metadata.get("action_allowed_now") or metadata.get("executable_tool_action_emitted") or metadata.get("can_emit_executable_tool_action") or metadata.get("can_auto_execute_now"):
                    raise SystemExit(f"Voice post-run closure must never allow direct speech execution: {metadata}")
                if "run command" in case:
                    assert_voice_confirmation_handoff(metadata, "Voice post-run closure", "run command python3 --version", confirmed=True)
                    assert_voice_privacy_receipt_chain(metadata, "Voice post-run closure")
                    assert_voice_command_intake_contract(metadata, "Voice post-run closure")
                    assert_voice_post_run_closure_token_boundary(
                        metadata,
                        "Voice post-run closure",
                        expected_source="voice_post_run_closure",
                    )
                    assert_voice_post_run_closure_token_hash_binds_authority(metadata, "Voice post-run closure")
                    if metadata.get("closure_state") != "VOICE_POST_RUN_CLOSURE_READY_FOR_NEXT_VOICE_REVIEW" or metadata.get("ready_for_next_voice_review") is not True:
                        raise SystemExit(f"Voice post-run closure missed ready state: {metadata}")
                    for flag in [
                        "handoff_ready",
                        "post_run_verification_reviewed",
                        "post_run_execution_health_reviewed",
                        "post_run_execution_audit_reviewed",
                        "after_action_learning_reviewed",
                        "verification_receipt_hash_present",
                        "execution_health_hash_present",
                        "execution_audit_hash_present",
                        "after_action_learning_hash_present",
                        "post_run_artifact_hashes_present",
                    ]:
                        if metadata.get(flag) is not True:
                            raise SystemExit(f"Voice post-run closure missed proof flag {flag}: {metadata}")
                    expected_hashes = {
                        "verification_receipt_sha256": VOICE_VERIFICATION_RECEIPT_SHA256,
                        "execution_health_sha256": VOICE_EXECUTION_HEALTH_SHA256,
                        "execution_audit_sha256": VOICE_EXECUTION_AUDIT_SHA256,
                        "after_action_learning_sha256": VOICE_AFTER_ACTION_LEARNING_SHA256,
                    }
                    for key, value in expected_hashes.items():
                        if metadata.get(key) != value:
                            raise SystemExit(f"Voice post-run closure missed {key}: {metadata}")
                    if metadata.get("post_run_artifact_hashes") != expected_hashes:
                        raise SystemExit(f"Voice post-run closure missed post-run artifact hash map: {metadata}")
                    if (
                        metadata.get("next_voice_review_state") != "FRESH_VOICE_REVIEW_UNLOCKED"
                        or not str(metadata.get("next_review_start_command", "")).startswith("voice transcript review:")
                        or metadata.get("next_review_requires_fresh_confirmation_receipt") is not True
                        or metadata.get("next_review_requires_fresh_receipt_nonce") is not True
                        or metadata.get("previous_confirmation_receipt_reusable_for_next_voice_review") is not False
                        or metadata.get("previous_transcript_hash_reusable_for_next_voice_review") is not False
                        or metadata.get("previous_handoff_reusable_for_next_voice_review") is not False
                        or metadata.get("prior_spoken_command_proof_only") is not True
                    ):
                        raise SystemExit(f"Voice post-run closure missed next-review freshness boundary: {metadata}")
                if "remember" in case:
                    assert_voice_confirmation_handoff(metadata, "Voice post-run closure", "remember that voice closures wait for proof", confirmed=True)
                    assert_voice_post_run_closure_token_boundary(
                        metadata,
                        "Incomplete voice post-run closure",
                        expected_source="voice_post_run_closure",
                    )
                    assert_voice_post_run_closure_token_hash_binds_authority(metadata, "Incomplete voice post-run closure")
                    if metadata.get("closure_state") != "VOICE_POST_RUN_CLOSURE_HELD" or metadata.get("ready_for_next_voice_review"):
                        raise SystemExit(f"Incomplete voice post-run closure should be held: {metadata}")
                    if (
                        metadata.get("next_voice_review_state") != "VOICE_REVIEW_HELD_FOR_POST_RUN_PROOF"
                        or metadata.get("next_review_start_command")
                        or metadata.get("next_review_requires_fresh_confirmation_receipt") is not True
                        or metadata.get("next_review_requires_fresh_receipt_nonce") is not True
                        or metadata.get("previous_confirmation_receipt_reusable_for_next_voice_review") is not False
                        or metadata.get("previous_transcript_hash_reusable_for_next_voice_review") is not False
                        or metadata.get("previous_handoff_reusable_for_next_voice_review") is not False
                        or metadata.get("prior_spoken_command_proof_only") is not True
                    ):
                        raise SystemExit(f"Incomplete voice post-run closure missed held next-review boundary: {metadata}")
                    for blocker in [
                        "post-run execution health evidence",
                        "post-run execution audit evidence",
                        "after-action learning evidence",
                        "valid verification receipt sha256",
                        "valid execution health sha256",
                        "valid execution audit sha256",
                        "valid after-action learning sha256",
                    ]:
                        if blocker not in metadata.get("missing_blockers", []):
                            raise SystemExit(f"Incomplete voice post-run closure missed blocker {blocker!r}: {metadata}")
                for key in VOICE_SAFE_FALSE_FLAGS:
                    if metadata.get(key):
                        raise SystemExit(f"Voice post-run closure unsafe metadata {key}: {metadata}")
            if case.startswith("voice cycle ledger"):
                required = [
                    "Jarvis voice cycle ledger",
                    "read-only",
                    "full confirmed-speech lifecycle",
                    "Cycle state",
                    "transcript hash",
                    "privacy receipt match",
                    "receipt freshness match",
                    "post-run artifact hashes present",
                    "ready for fresh next voice review",
                    "voice cycle ledger contract ready",
                    "action allowed now: no",
                    "executable tool action emitted: no",
                    "Cycle stages",
                    "privacy_boundary",
                    "confirmation_receipt",
                    "route_gate",
                    "route_proof_bundle",
                    "runtime_bridge",
                    "command_cockpit",
                    "action_audit",
                    "execution_handoff",
                    "post_run_closure",
                    "Measured preflight scorecard",
                    "preflight score",
                    "preflight rows ready",
                    "visible_privacy_receipt",
                    "fresh_confirmation_receipt",
                    "command_intake_bridge",
                    "Fresh-review boundary",
                    "fresh confirmation receipt required before next spoken order: yes",
                    "fresh receipt nonce required before next spoken order: yes",
                    "previous confirmation receipt reusable for next voice review: no",
                    "previous transcript hash reusable for next voice review: no",
                    "previous execution handoff reusable for next voice review: no",
                    "prior spoken command proof only: yes",
                    "all prior voice artifacts non-authorizing: yes",
                    "next voice review requires full preflight: yes",
                    "Fresh-review preflight queue",
                    "voice capture privacy",
                    "voice transcript review: <next transcript>",
                    "voice confirmation audit ledger: <next transcript>",
                    "Fresh-review contract rows",
                    "visible_capture_privacy_receipt",
                    "voice_cycle_ledger_review_token",
                    "Required cycle proof chain",
                    "Safety boundary",
                    "new spoken order must begin with a fresh transcript review",
                ]
                if "run command" in case:
                    required.extend(
                        [
                            "VOICE_CYCLE_LEDGER_READY_FOR_FRESH_REVIEW",
                            "missing blockers: none",
                            "post-run artifact hashes present: yes",
                            "voice cycle ledger: run command python3 --version confirmed=true",
                            "voice transcript review: <next transcript>",
                        ]
                    )
                if "remember" in case:
                    required.extend(
                        [
                            "VOICE_CYCLE_LEDGER_HELD",
                            "post-run execution health evidence",
                            "post-run execution audit evidence",
                            "after-action learning evidence",
                        ]
                    )
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Voice cycle ledger missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                if not metadata.get("auditable") or metadata.get("routes_actions"):
                    raise SystemExit(f"Voice cycle ledger should be auditable without routing actions: {metadata}")
                if metadata.get("action_allowed_now") or metadata.get("executable_tool_action_emitted") or metadata.get("can_emit_executable_tool_action") or metadata.get("can_auto_execute_now"):
                    raise SystemExit(f"Voice cycle ledger must never allow direct speech execution: {metadata}")
                assert_voice_confirmation_handoff(
                    metadata,
                    "Voice cycle ledger",
                    "run command python3 --version" if "run command" in case else "remember that voice cycle ledgers wait for proof",
                    confirmed=True,
                )
                assert_voice_privacy_receipt_chain(metadata, "Voice cycle ledger")
                assert_voice_command_intake_contract(metadata, "Voice cycle ledger")
                assert_voice_post_run_closure_token_boundary(
                    metadata,
                    "Voice cycle ledger",
                    expected_source="voice_cycle_ledger",
                )
                assert_voice_post_run_closure_token_hash_binds_authority(
                    metadata.get("post_run_closure_metadata") or metadata,
                    "Voice cycle ledger",
                )
                assert_voice_cycle_ledger_token_hash_binds_authority(metadata, "Voice cycle ledger")
                if metadata.get("stage_count") != 9 or len(metadata.get("stage_rows", [])) != 9:
                    raise SystemExit(f"Voice cycle ledger missed stage rows: {metadata}")
                if any(
                    row.get("authorizes_action") is not False
                    or row.get("authorizes_model_call") is not False
                    or row.get("authorizes_tool_execution") is not False
                    or row.get("authorizes_approval") is not False
                    or row.get("authorizes_routing") is not False
                    or row.get("authorizes_transcript_mutation") is not False
                    or row.get("authorizes_personal_data_read") is not False
                    or row.get("authorizes_external_side_effect") is not False
                    or row.get("authorizes_next_voice_review") is not False
                    or row.get("authorizes_receipt_reuse") is not False
                    or row.get("reusable_for_next_voice_review") is not False
                    for row in metadata.get("stage_rows", [])
                ):
                    raise SystemExit(f"Voice cycle ledger stage rows must stay proof-only/non-authorizing: {metadata}")
                scorecard_rows = metadata.get("voice_preflight_scorecard_rows") or []
                expected_scorecard_items = {
                    "visible_privacy_receipt",
                    "fresh_confirmation_receipt",
                    "route_proof_bundle",
                    "command_intake_bridge",
                    "action_audit_boundary",
                    "execution_handoff_packet",
                    "post_run_artifact_hashes",
                    "fresh_review_cycle_boundary",
                }
                if metadata.get("voice_preflight_scorecard_row_count") != 8 or len(scorecard_rows) != 8:
                    raise SystemExit(f"Voice cycle ledger missed preflight scorecard rows: {metadata}")
                if {row.get("item") for row in scorecard_rows} != expected_scorecard_items:
                    raise SystemExit(f"Voice cycle ledger missed preflight scorecard items: {metadata}")
                if sum(int(row.get("max_points", 0)) for row in scorecard_rows) != 100:
                    raise SystemExit(f"Voice cycle ledger scorecard should total 100 points: {metadata}")
                if metadata.get("voice_preflight_score") != sum(int(row.get("points", 0)) for row in scorecard_rows):
                    raise SystemExit(f"Voice cycle ledger score should equal scorecard points: {metadata}")
                if metadata.get("voice_preflight_max_score") != sum(int(row.get("max_points", 0)) for row in scorecard_rows):
                    raise SystemExit(f"Voice cycle ledger max score should equal scorecard max points: {metadata}")
                if any(
                    row.get("required_before_next_voice_review") is not True
                    or row.get("authorizes_action") is not False
                    or row.get("authorizes_model_call") is not False
                    or row.get("authorizes_tool_execution") is not False
                    or row.get("authorizes_approval") is not False
                    or row.get("authorizes_routing") is not False
                    or row.get("authorizes_transcript_mutation") is not False
                    or row.get("authorizes_personal_data_read") is not False
                    or row.get("authorizes_external_side_effect") is not False
                    or row.get("authorizes_next_voice_review") is not False
                    or row.get("authorizes_receipt_reuse") is not False
                    or row.get("reusable_for_next_voice_review") is not False
                    for row in scorecard_rows
                ):
                    raise SystemExit(f"Voice cycle ledger scorecard rows must stay required/non-authorizing: {metadata}")
                if "run command" in case:
                    if metadata.get("cycle_state") != "VOICE_CYCLE_LEDGER_READY_FOR_FRESH_REVIEW" or metadata.get("ready_for_fresh_next_voice_review") is not True:
                        raise SystemExit(f"Voice cycle ledger missed ready state: {metadata}")
                    if metadata.get("voice_cycle_ledger_ready") is not True:
                        raise SystemExit(f"Voice cycle ledger missed shape-aware ready flag: {metadata}")
                    assert_voice_cycle_stage_rows_ready_flag(
                        metadata,
                        "Voice cycle ledger",
                        expected_ready=True,
                    )
                    if any(row.get("ready") is not True for row in metadata.get("stage_rows", [])):
                        raise SystemExit(f"Voice cycle ledger ready case has held stages: {metadata}")
                    if metadata.get("voice_preflight_score") != 100 or metadata.get("voice_preflight_required_rows_ready") is not True:
                        raise SystemExit(f"Voice cycle ledger ready case missed complete preflight scorecard: {metadata}")
                    assert_voice_cycle_preflight_scorecard_ready_flag(
                        metadata,
                        "Voice cycle ledger",
                        expected_ready=True,
                    )
                    if any(row.get("ready") is not True or row.get("points") != row.get("max_points") for row in scorecard_rows):
                        raise SystemExit(f"Voice cycle ledger ready case has held scorecard rows: {metadata}")
                    for flag in [
                        "post_run_verification_reviewed",
                        "post_run_execution_health_reviewed",
                        "post_run_execution_audit_reviewed",
                        "after_action_learning_reviewed",
                        "verification_receipt_hash_present",
                        "execution_health_hash_present",
                        "execution_audit_hash_present",
                        "after_action_learning_hash_present",
                        "post_run_artifact_hashes_present",
                    ]:
                        if metadata.get(flag) is not True:
                            raise SystemExit(f"Voice cycle ledger missed proof flag {flag}: {metadata}")
                    expected_hashes = {
                        "verification_receipt_sha256": VOICE_VERIFICATION_RECEIPT_SHA256,
                        "execution_health_sha256": VOICE_EXECUTION_HEALTH_SHA256,
                        "execution_audit_sha256": VOICE_EXECUTION_AUDIT_SHA256,
                        "after_action_learning_sha256": VOICE_AFTER_ACTION_LEARNING_SHA256,
                    }
                    for key, value in expected_hashes.items():
                        if metadata.get(key) != value:
                            raise SystemExit(f"Voice cycle ledger missed {key}: {metadata}")
                    if metadata.get("post_run_artifact_hashes") != expected_hashes:
                        raise SystemExit(f"Voice cycle ledger missed post-run artifact hash map: {metadata}")
                    nested_closure = metadata.get("post_run_closure_metadata") or {}
                    if nested_closure.get("post_run_artifact_hashes") != expected_hashes:
                        raise SystemExit(f"Voice cycle ledger nested closure missed artifact hash map: {metadata}")
                    if (
                        metadata.get("next_review_requires_fresh_confirmation_receipt") is not True
                        or metadata.get("next_review_requires_fresh_receipt_nonce") is not True
                        or metadata.get("previous_confirmation_receipt_reusable_for_next_voice_review") is not False
                        or metadata.get("previous_transcript_hash_reusable_for_next_voice_review") is not False
                        or metadata.get("previous_handoff_reusable_for_next_voice_review") is not False
                        or metadata.get("prior_spoken_command_proof_only") is not True
                        or metadata.get("all_prior_artifacts_non_authorizing") is not True
                        or metadata.get("next_voice_review_requires_full_preflight") is not True
                    ):
                        raise SystemExit(f"Voice cycle ledger missed next-review freshness boundary: {metadata}")
                    fresh_review_queue = metadata.get("fresh_review_preflight_queue") or []
                    if metadata.get("fresh_review_preflight_queue_count") != len(fresh_review_queue):
                        raise SystemExit(f"Voice cycle ledger missed fresh-review queue count: {metadata}")
                    if metadata.get("fresh_review_next_preflight_command") != "voice capture privacy":
                        raise SystemExit(f"Voice cycle ledger missed next fresh-review command: {metadata}")
                    for expected_command in [
                        "voice capture privacy",
                        "voice transcript review: <next transcript>",
                        "voice confirmation receipt: <next transcript> confirmed=true",
                        "voice confirmation audit ledger: <next transcript>; confirmed=true; privacy_receipt_id=<fresh>; receipt_id=<fresh>; receipt_nonce=<fresh>",
                        "voice route proof bundle: <next transcript> confirmed=true",
                        "voice command cockpit: <next transcript> confirmed=true",
                        "voice execution handoff: <next transcript> confirmed=true",
                        "voice cycle ledger: run command python3 --version confirmed=true post-run proof reviewed",
                    ]:
                        if expected_command not in fresh_review_queue:
                            raise SystemExit(f"Voice cycle ledger missed fresh-review command {expected_command!r}: {metadata}")
                    fresh_review_contract = metadata.get("fresh_review_contract_rows") or []
                    if metadata.get("fresh_review_contract_count") != len(fresh_review_contract):
                        raise SystemExit(f"Voice cycle ledger missed fresh-review contract count: {metadata}")
                    expected_contract_items = {
                        "visible_capture_privacy_receipt",
                        "confirmed_transcript_receipt",
                        "receipt_nonce_freshness",
                        "route_proof_bundle",
                        "command_intake_bridge",
                        "execution_handoff_packet",
                        "post_run_closure_artifacts",
                        "voice_cycle_ledger_review_token",
                    }
                    if {row.get("item") for row in fresh_review_contract} != expected_contract_items:
                        raise SystemExit(f"Voice cycle ledger missed fresh-review contract items: {metadata}")
                    if any(
                        row.get("proof_only") is not True
                        or row.get("authorizes_action") is not False
                        or row.get("authorizes_model_call") is not False
                        or row.get("authorizes_tool_execution") is not False
                        or row.get("authorizes_approval") is not False
                        or row.get("authorizes_routing") is not False
                        or row.get("authorizes_transcript_mutation") is not False
                        or row.get("authorizes_personal_data_read") is not False
                        or row.get("authorizes_external_side_effect") is not False
                        or row.get("reusable_prior_artifact") is not False
                        for row in fresh_review_contract
                    ):
                        raise SystemExit(f"Voice cycle ledger contract should keep prior artifacts non-authorizing: {metadata}")
                if "remember" in case:
                    if metadata.get("cycle_state") != "VOICE_CYCLE_LEDGER_HELD" or metadata.get("ready_for_fresh_next_voice_review"):
                        raise SystemExit(f"Incomplete voice cycle ledger should be held: {metadata}")
                    if metadata.get("voice_cycle_ledger_ready") is not False:
                        raise SystemExit(f"Incomplete voice cycle ledger should not be shape-ready: {metadata}")
                    assert_voice_cycle_stage_rows_ready_flag(
                        metadata,
                        "Incomplete voice cycle ledger",
                        expected_ready=False,
                    )
                    if metadata.get("voice_preflight_score", 100) >= 100 or metadata.get("voice_preflight_required_rows_ready") is not False:
                        raise SystemExit(f"Incomplete voice cycle ledger should hold preflight scorecard: {metadata}")
                    assert_voice_cycle_preflight_scorecard_ready_flag(
                        metadata,
                        "Incomplete voice cycle ledger",
                        expected_ready=False,
                    )
                    for blocker in [
                        "post-run execution health evidence",
                        "post-run execution audit evidence",
                        "after-action learning evidence",
                        "valid verification receipt sha256",
                        "valid execution health sha256",
                        "valid execution audit sha256",
                        "valid after-action learning sha256",
                        "post_run_closure not ready",
                    ]:
                        if blocker not in metadata.get("missing_blockers", []):
                            raise SystemExit(f"Incomplete voice cycle ledger missed blocker {blocker!r}: {metadata}")
                for key in VOICE_SAFE_FALSE_FLAGS:
                    if metadata.get(key):
                        raise SystemExit(f"Voice cycle ledger unsafe metadata {key}: {metadata}")
            if case.startswith("voice stop intent") or case.startswith("voice rerecord intent"):
                required = [
                    "Jarvis voice stop intent packet",
                    "read-only",
                    "Stop intent",
                    "VOICE_STOP_CAPTURE_ONLY_READY",
                    "stop current capture only",
                    "Non-destructive boundary",
                    "does not send confirmed transcript",
                    "does not route the stop phrase into command intake",
                    "does not delete memory, approvals, notes, files, or prior messages",
                    "does not queue approvals, execute tools, record audio, or start a listener",
                    "Stop speech is a brake",
                ]
                if case.startswith("voice rerecord intent"):
                    required.extend(["rerecord requested: yes", "rerecord voice command"])
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Voice stop intent missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                if metadata.get("stop_state") != "VOICE_STOP_CAPTURE_ONLY_READY" or metadata.get("capture_stop_only") is not True:
                    raise SystemExit(f"Voice stop intent missed stop-only metadata: {metadata}")
                if metadata.get("routes_stop_phrase_to_command_intake") or metadata.get("sends_confirmed_transcript") or metadata.get("deletes_memory") or metadata.get("deletes_approvals") or metadata.get("deletes_messages") or metadata.get("rewrites_transcript"):
                    raise SystemExit(f"Voice stop intent must be non-destructive: {metadata}")
                for key in VOICE_SAFE_FALSE_FLAGS:
                    if metadata.get(key):
                        raise SystemExit(f"Voice stop intent unsafe metadata {key}: {metadata}")
            if case.startswith("voice confirmation"):
                if case.startswith("voice confirmation audit ledger"):
                    required = [
                        "Jarvis voice confirmation audit ledger",
                        "VOICE_CONFIRMATION_AUDIT_LEDGER_HELD",
                        "supplied privacy receipt required: yes",
                        "supplied confirmation receipt id required: yes",
                        "route-time generated receipt acceptable as external proof: no",
                    ]
                    missing = [item for item in required if item not in result.response]
                    if missing:
                        raise SystemExit(f"Voice confirmation audit ledger missing expected text: {missing}")
                    metadata = result.tool_results[0].metadata
                    if metadata.get("audit_ledger_state") != "VOICE_CONFIRMATION_AUDIT_LEDGER_HELD" or metadata.get("audit_ledger_ready"):
                        raise SystemExit(f"Voice confirmation audit ledger should hold missing supplied receipts: {metadata}")
                    for blocker in [
                        "supplied privacy receipt id from visible capture boundary",
                        "supplied confirmation receipt id",
                        "supplied confirmation receipt nonce",
                    ]:
                        if blocker not in metadata.get("missing_blockers", []):
                            raise SystemExit(f"Voice confirmation audit ledger missed blocker {blocker!r}: {metadata}")
                    for key in VOICE_SAFE_FALSE_FLAGS:
                        if metadata.get(key):
                            raise SystemExit(f"Voice confirmation audit ledger unsafe metadata {key}: {metadata}")
                    continue
                if case.startswith("voice confirmation receipt"):
                    required = [
                        "Jarvis voice confirmation receipt",
                        "read-only",
                        "Receipt id",
                        "Route state",
                        "Transcript snapshot",
                        "Risk cues",
                        "Planned tools",
                        "Routing decision",
                        "Audit fields for a future real run",
                        "Next safe commands",
                        "transcript_hash",
                        "Safety boundary",
                        "does not request microphone access",
                        "record audio",
                        "execute tools",
                        "queue approvals",
                    ]
                    if "run command" in case:
                        required.extend(["CONFIRMED_FOR_ROUTING", "run_shell_command", "HIGH_RISK", "approval required", "approval readiness #ID", "approval packet #ID", "approval chain proof #ID", "approval readiness latest", "approval packet latest", "approval chain proof latest"])
                    if "remember" in case:
                        required.extend(["WAITING_FOR_CONFIRMATION", "remember", "LOCAL_SAFE", "default allowed after confirmation"])
                    missing = [item for item in required if item not in result.response]
                    if missing:
                        raise SystemExit(f"Voice confirmation receipt missing expected text: {missing}")
                    metadata = result.tool_results[0].metadata
                    if not metadata.get("auditable") or metadata.get("routes_actions"):
                        raise SystemExit(f"Voice confirmation receipt should be auditable without routing actions: {metadata}")
                    if "run command" in case and (not metadata.get("confirmed") or metadata.get("route_state") != "CONFIRMED_FOR_ROUTING"):
                        raise SystemExit(f"Voice confirmation receipt missed confirmed routing state: {metadata}")
                    if "remember" in case and (metadata.get("confirmed") or metadata.get("route_state") != "WAITING_FOR_CONFIRMATION"):
                        raise SystemExit(f"Voice confirmation receipt should wait when not explicitly confirmed: {metadata}")
                    if "run command" in case:
                        assert_voice_approval_chain(metadata, "Voice confirmation receipt")
                        assert_voice_confirmation_handoff(metadata, "Voice confirmation receipt", "run command python3 --version", confirmed=True)
                        if not any(str(command).startswith("send confirmed transcript:") for command in metadata.get("recommended_next_commands", [])):
                            raise SystemExit(f"Confirmed voice receipt should expose a send-confirmed-transcript command: {metadata}")
                    if "remember" in case:
                        assert_voice_confirmation_handoff(metadata, "Voice confirmation receipt", "remember that confirmed voice receipts stay read only")
                        commands = metadata.get("recommended_next_commands", [])
                        if any(str(command).startswith("send confirmed transcript:") for command in commands):
                            raise SystemExit(f"Unconfirmed voice receipt must not suggest sending a confirmed transcript: {metadata}")
                        for command in [
                            "voice transcript review: remember that confirmed voice receipts stay read only",
                            "voice confirmation: remember that confirmed voice receipts stay read only",
                            "voice confirmation receipt: remember that confirmed voice receipts stay read only confirmed=true",
                            "rerecord voice command",
                        ]:
                            if command not in commands:
                                raise SystemExit(f"Unconfirmed voice receipt missed safe next command {command!r}: {metadata}")
                    for key in VOICE_SAFE_FALSE_FLAGS:
                        if metadata.get(key):
                            raise SystemExit(f"Voice confirmation receipt unsafe metadata {key}: {metadata}")
                else:
                    required = [
                        "Jarvis voice confirmation packet",
                        "Transcript to confirm",
                        "Risk cues",
                        "Planned tools",
                        "Confirmation path",
                        "Exact command to send after the operator confirms",
                        "Next safe commands after the operator confirms",
                        "Auditable confirmation handoff",
                        "transcript hash",
                        "receipt command",
                        "route blocker",
                        "Safety boundary",
                        "does not execute tools",
                        "queue approvals",
                    ]
                    if "run command" in case:
                        required.extend(["run_shell_command", "HIGH_RISK", "approval required", "approval readiness #ID", "approval packet #ID", "approval chain proof #ID", "approval readiness latest", "approval packet latest", "approval chain proof latest"])
                    if "remember" in case:
                        required.extend(["remember", "LOCAL_SAFE", "default allowed"])
                    missing = [item for item in required if item not in result.response]
                    if missing:
                        raise SystemExit(f"Voice confirmation packet missing expected text: {missing}")
                    metadata = result.tool_results[0].metadata
                    if "run command" in case:
                        assert_voice_approval_chain(metadata, "Voice confirmation packet")
                        assert_voice_confirmation_handoff(metadata, "Voice confirmation packet", "run command python3 --version")
                    if "remember" in case:
                        assert_voice_confirmation_handoff(metadata, "Voice confirmation packet", "remember that confirmed voice packets stay read only")
                    for key in VOICE_SAFE_FALSE_FLAGS:
                        if metadata.get(key):
                            raise SystemExit(f"Voice confirmation packet unsafe metadata {key}: {metadata}")
            if case.startswith("voice command lifecycle"):
                required = [
                    "Jarvis voice command lifecycle",
                    "Setup",
                    "Capture boundary",
                    "Transcript preview",
                    "Confirmation packet",
                    "Confirmation receipt",
                    "Route gate",
                    "Route proof bundle",
                    "Runtime bridge",
                    "Command cockpit and action audit",
                    "Execution handoff and post-run closure",
                    "Approval gate",
                    "Audit",
                    "Required voice proof chain",
                    "Next safe commands after the operator confirms",
                    "Auditable confirmation handoff",
                    "receipt command",
                    "route gate command",
                    "route proof bundle command",
                    "runtime bridge command",
                    "does not request microphone access",
                    "record audio",
                    "start a listener",
                    "queue approvals",
                ]
                if "run command" in case:
                    required.extend([
                        "run_shell_command",
                        "HIGH_RISK",
                        "approval required",
                        "Exact command",
                        "approval readiness #ID",
                        "approval packet #ID",
                        "approval chain proof #ID",
                        "approval readiness latest",
                        "approval packet latest",
                        "approval chain proof latest",
                        "receipt nonce",
                        "receipt freshness",
                        "fresh confirmed receipt id and nonce",
                        "voice route proof bundle: run command python3 --version confirmed=true receipt_id=",
                        "voice runtime bridge: run command python3 --version confirmed=true receipt_id=",
                        "voice command cockpit: run command python3 --version confirmed=true receipt_id=",
                    ])
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Voice command lifecycle missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                if "run command" in case:
                    assert_voice_approval_chain(metadata, "Voice command lifecycle")
                    assert_voice_confirmation_handoff(metadata, "Voice command lifecycle", "run command python3 --version")
                    if (
                        not metadata.get("confirmation_receipt_id")
                        or not metadata.get("confirmation_receipt_nonce")
                        or metadata.get("receipt_nonce_required") is not True
                        or metadata.get("receipt_freshness_required") is not True
                        or metadata.get("stale_receipt_blocks_routing") is not True
                    ):
                        raise SystemExit(f"Voice command lifecycle missed freshness metadata: {metadata}")
                    expected_lifecycle_commands = {
                        "route_gate_command": "voice route gate: run command python3 --version confirmed=true",
                        "route_proof_bundle_command": "voice route proof bundle: run command python3 --version confirmed=true",
                        "runtime_bridge_command": "voice runtime bridge: run command python3 --version confirmed=true",
                        "command_cockpit_command": "voice command cockpit: run command python3 --version confirmed=true",
                        "action_audit_command": "voice action audit: run command python3 --version confirmed=true",
                        "execution_handoff_command": "voice execution handoff: run command python3 --version confirmed=true",
                    }
                    for key, expected_prefix in expected_lifecycle_commands.items():
                        expected_command = metadata.get(key) or ""
                        if not str(expected_command).startswith(expected_prefix):
                            raise SystemExit(f"Voice command lifecycle missed metadata {key}: {metadata}")
                        if f"receipt_id={metadata['confirmation_receipt_id']}" not in expected_command or f"receipt_nonce={metadata['confirmation_receipt_nonce']}" not in expected_command:
                            raise SystemExit(f"Voice command lifecycle missed receipt freshness in {key}: {metadata}")
                        if expected_command not in metadata.get("proof_commands", []):
                            raise SystemExit(f"Voice command lifecycle missed proof command {expected_command!r}: {metadata}")
                    if metadata.get("proof_command_count") != len(metadata.get("proof_commands", [])):
                        raise SystemExit(f"Voice command lifecycle proof command count diverged: {metadata}")
                    if metadata.get("proof_command_count", 0) < 10:
                        raise SystemExit(f"Voice command lifecycle proof chain is too short: {metadata}")
                    if metadata.get("command_intake_only") is not True or metadata.get("can_emit_executable_tool_action") is not False:
                        raise SystemExit(f"Voice command lifecycle missed command-intake-only boundary: {metadata}")
                for key in VOICE_SAFE_FALSE_FLAGS:
                    if metadata.get(key):
                        raise SystemExit(f"Voice command lifecycle unsafe metadata {key}: {metadata}")
            if case.startswith(("voice file transcription plan", "voice note plan", "telegram voice note plan", "voice memo plan")):
                required = [
                    "Jarvis audio-file transcription plan",
                    "Requested file",
                    "Local readiness",
                    "Safe future flow",
                    "Privacy boundary",
                    "Implementation guardrails",
                    "read-only",
                    "does not open the audio file",
                    "request microphone access",
                    "record audio",
                    "start a listener",
                    "save transcripts",
                    "queue approvals",
                ]
                if "/tmp/jarvis-test-audio.m4a" in case:
                    required.extend(["/tmp/jarvis-test-audio.m4a", "recognized audio extension: yes"])
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Voice file transcription plan missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                handoff = assert_voice_file_handoff(metadata, "voice_file_transcription_plan_handoff", "Voice file transcription plan")
                if handoff.get("source") != "voice_file_transcription_plan":
                    raise SystemExit(f"Voice file transcription plan handoff source wrong: {handoff}")
                if "/tmp/jarvis-test-audio.m4a" in case and (
                    handoff.get("recognized_extension") is not True or handoff.get("suffix") != ".m4a"
                ):
                    raise SystemExit(f"Voice file transcription plan handoff missed file metadata: {handoff}")
                for key in VOICE_SAFE_FALSE_FLAGS:
                    if metadata.get(key):
                        raise SystemExit(f"Voice file transcription plan unsafe metadata {key}: {metadata}")
            if case.startswith(("voice audio file gate", "voice note gate", "telegram voice note gate", "voice memo gate")):
                required = [
                    "Jarvis voice audio-file gate packet",
                    "File receipt",
                    "path hash",
                    "File checks",
                    "Dependency checks",
                    "Gate decision",
                    "next safe command",
                    "Approved future flow",
                    "Safety boundary",
                    "does not read the audio file contents",
                    "record audio",
                    "save transcripts",
                    "queue approvals",
                ]
                if str(sample_audio) in case and "consent=true" in case:
                    required.extend(
                        [
                            "Jarvis local audio transcriber",
                            "recognized audio extension: yes",
                        ]
                    )
                elif str(sample_audio) in case:
                    required.extend(["AUDIO_FILE_HELD_FOR_REVIEW", "operator consent=true"])
                else:
                    required.extend(["AUDIO_FILE_HELD_FOR_REVIEW", "recognized audio extension: no"])
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Voice audio file gate missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                handoff = assert_voice_file_handoff(metadata, "voice_audio_file_gate_handoff", "Voice audio file gate")
                if handoff.get("source") != "voice_audio_file_gate_packet":
                    raise SystemExit(f"Voice audio file gate handoff source wrong: {handoff}")
                for field in ("gate_state", "missing_checks", "missing_check_count", "receipt_id", "consent"):
                    if handoff.get(field) != metadata.get(field):
                        raise SystemExit(f"Voice audio file gate handoff {field} parity failed: {metadata} / {handoff}")
                if handoff.get("can_transcribe_now") is not False or metadata.get("can_transcribe_now") is not False:
                    raise SystemExit(f"Voice audio file gate should never transcribe in the gate packet: {metadata} / {handoff}")
                if str(sample_audio) in case and "consent=true" in case:
                    if metadata.get("local_audio_transcriber_configured") and metadata.get("approved_transcription_runner_ready"):
                        if metadata.get("gate_state") != "AUDIO_FILE_READY_FOR_APPROVED_TRANSCRIPTION" or metadata.get("missing_checks") != []:
                            raise SystemExit(f"Voice audio file gate should be ready only for approved transcription when local runner is configured: {metadata}")
                        if handoff.get("gate_state") != "AUDIO_FILE_READY_FOR_APPROVED_TRANSCRIPTION" or handoff.get("missing_checks") != []:
                            raise SystemExit(f"Voice audio file gate handoff should be ready only for approved transcription when local runner is configured: {handoff}")
                    else:
                        if metadata.get("gate_state") != "AUDIO_FILE_HELD_FOR_REVIEW" or "local audio transcriber configured" not in metadata.get("missing_checks", []):
                            raise SystemExit(f"Voice audio file gate should hold until local transcriber is configured: {metadata}")
                        if handoff.get("gate_state") != "AUDIO_FILE_HELD_FOR_REVIEW" or "local audio transcriber configured" not in handoff.get("missing_checks", []):
                            raise SystemExit(f"Voice audio file gate handoff should hold until local transcriber is configured: {handoff}")
                        if metadata.get("local_audio_transcriber_configured") is not False or metadata.get("approved_transcription_runner_ready") is not False:
                            raise SystemExit(f"Voice audio file gate overstated transcriber readiness: {metadata}")
                        if handoff.get("local_audio_transcriber_configured") is not False or handoff.get("approved_transcription_runner_ready") is not False:
                            raise SystemExit(f"Voice audio file gate handoff overstated transcriber readiness: {handoff}")
                if str(sample_audio) in case and "consent=true" not in case:
                    if metadata.get("gate_state") != "AUDIO_FILE_HELD_FOR_REVIEW" or "operator consent=true" not in metadata.get("missing_checks", []):
                        raise SystemExit(f"Voice audio file gate should hold without consent: {metadata}")
                    if handoff.get("gate_state") != "AUDIO_FILE_HELD_FOR_REVIEW" or "operator consent=true" not in handoff.get("missing_checks", []):
                        raise SystemExit(f"Voice audio file gate handoff should hold without consent: {handoff}")
                if "jarvis-test-audio.txt" in case and "recognized audio extension" not in metadata.get("missing_checks", []):
                    raise SystemExit(f"Voice audio file gate should hold unknown extension: {metadata}")
                if "jarvis-test-audio.txt" in case and "recognized audio extension" not in handoff.get("missing_checks", []):
                    raise SystemExit(f"Voice audio file gate handoff should hold unknown extension: {handoff}")
                for key in VOICE_SAFE_FALSE_FLAGS:
                    if metadata.get(key):
                        raise SystemExit(f"Voice audio file gate unsafe metadata {key}: {metadata}")
            if case in {"voice input plan push-to-talk", "wake-word plan"}:
                required = [
                    "Jarvis voice input boundary plan",
                    "Current state",
                    "Wake-word, ASR",
                    "read-only",
                    "does not request microphone access",
                    "Privacy boundaries",
                    "Microphone access is PERSONAL_DATA",
                    "Recommended implementation stages",
                    "Action boundary",
                    "approval queue",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Voice input plan missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                for key in VOICE_SAFE_FALSE_FLAGS:
                    if metadata.get(key):
                        raise SystemExit(f"Voice input plan unsafe metadata {key}: {metadata}")
        oversized_review = runtime.registry.get("voice_transcript_review").handler({"transcript": "x" * (MAX_TRANSCRIPT_CHARS + 1)})
        if oversized_review.ok or "too large" not in oversized_review.output:
            raise SystemExit("Voice transcript review should refuse oversized transcripts.")
        oversized_confirmation = runtime.registry.get("voice_confirmation_packet").handler({"transcript": "x" * (MAX_TRANSCRIPT_CHARS + 1)})
        if oversized_confirmation.ok or "too large" not in oversized_confirmation.output:
            raise SystemExit("Voice confirmation packet should refuse oversized transcripts.")
        oversized_receipt = runtime.registry.get("voice_confirmation_receipt").handler({"transcript": "x" * (MAX_TRANSCRIPT_CHARS + 1)})
        if oversized_receipt.ok or "too large" not in oversized_receipt.output:
            raise SystemExit("Voice confirmation receipt should refuse oversized transcripts.")
        oversized_lifecycle = runtime.registry.get("voice_command_lifecycle").handler({"transcript": "x" * (MAX_TRANSCRIPT_CHARS + 1)})
        if oversized_lifecycle.ok or "too large" not in oversized_lifecycle.output:
            raise SystemExit("Voice command lifecycle should refuse oversized transcripts.")
        oversized_closure = runtime.registry.get("voice_post_run_closure_packet").handler({"transcript": "x" * (MAX_TRANSCRIPT_CHARS + 1)})
        if oversized_closure.ok or "too large" not in oversized_closure.output:
            raise SystemExit("Voice post-run closure should refuse oversized transcripts.")
        oversized_cycle = runtime.registry.get("voice_cycle_ledger").handler({"transcript": "x" * (MAX_TRANSCRIPT_CHARS + 1)})
        if oversized_cycle.ok or "too large" not in oversized_cycle.output:
            raise SystemExit("Voice cycle ledger should refuse oversized transcripts.")
        oversized_bridge = runtime.registry.get("voice_runtime_bridge_packet").handler({"transcript": "x" * (MAX_TRANSCRIPT_CHARS + 1)})
        if oversized_bridge.ok or "too large" not in oversized_bridge.output:
            raise SystemExit("Voice runtime bridge should refuse oversized transcripts.")
        oversized_cockpit = runtime.registry.get("voice_command_cockpit").handler({"transcript": "x" * (MAX_TRANSCRIPT_CHARS + 1)})
        if oversized_cockpit.ok or "too large" not in oversized_cockpit.output:
            raise SystemExit("Voice command cockpit should refuse oversized transcripts.")
        privacy = voice_capture_privacy_packet({"mode": "browser-push-to-talk"})
        privacy_receipt_id = privacy.metadata.get("privacy_receipt_id")
        matched_proof = runtime.registry.get("voice_route_proof_bundle").handler(
            {
                "transcript": "run command python3 --version",
                "confirmed": "true",
                "privacy_receipt_id": privacy_receipt_id,
            }
        )
        if not matched_proof.ok or matched_proof.metadata.get("bundle_state") != "VOICE_ROUTE_PROOF_READY":
            raise SystemExit(f"Voice route proof should accept the matching privacy receipt: {matched_proof.metadata}")
        receipt = runtime.registry.get("voice_confirmation_receipt").handler(
            {"transcript": "run command python3 --version", "confirmed": "true"}
        )
        if (
            not receipt.ok
            or not receipt.metadata.get("receipt_id")
            or not receipt.metadata.get("receipt_nonce")
            or receipt.metadata.get("receipt_nonce_required") is not True
        ):
            raise SystemExit(f"Voice confirmation receipt missed freshness nonce: {receipt.metadata}")
        strict_ledger = runtime.registry.get("voice_confirmation_audit_ledger").handler(
            {
                "transcript": "run command python3 --version",
                "confirmed": "true",
                "privacy_receipt_id": privacy_receipt_id,
                "receipt_id": receipt.metadata.get("receipt_id"),
                "receipt_nonce": receipt.metadata.get("receipt_nonce"),
            }
        )
        if (
            not strict_ledger.ok
            or strict_ledger.metadata.get("audit_ledger_state") != "VOICE_CONFIRMATION_AUDIT_LEDGER_READY"
            or strict_ledger.metadata.get("audit_ledger_ready") is not True
            or strict_ledger.metadata.get("supplied_privacy_receipt_match") is not True
            or strict_ledger.metadata.get("supplied_confirmation_receipt_id_match") is not True
            or strict_ledger.metadata.get("supplied_confirmation_receipt_nonce_match") is not True
            or strict_ledger.metadata.get("route_time_generated_receipt_acceptable_as_external_proof") is not False
            or strict_ledger.metadata.get("can_emit_executable_tool_action") is not False
            or strict_ledger.metadata.get("can_auto_execute_now") is not False
        ):
            raise SystemExit(f"Voice confirmation audit ledger should accept supplied matching receipts: {strict_ledger.metadata}")
        if strict_ledger.metadata.get("stage_count") != len(strict_ledger.metadata.get("stage_rows") or []):
            raise SystemExit(f"Voice confirmation audit ledger missed stage count: {strict_ledger.metadata}")
        if any(row.get("ready") is not True for row in strict_ledger.metadata.get("stage_rows", [])):
            raise SystemExit(f"Voice confirmation audit ledger ready case has held stages: {strict_ledger.metadata}")
        assert_voice_confirmation_audit_token_boundary(strict_ledger.metadata, "voice confirmation audit ledger")
        assert_voice_confirmation_audit_token_hash_binds_authority(strict_ledger.metadata, "voice confirmation audit ledger")
        assert_voice_command_intake_contract(strict_ledger.metadata, "voice confirmation audit ledger")
        stale_ledger = runtime.registry.get("voice_confirmation_audit_ledger").handler(
            {
                "transcript": "run command python3 --version",
                "confirmed": "true",
                "privacy_receipt_id": privacy_receipt_id,
                "receipt_id": receipt.metadata.get("receipt_id"),
                "receipt_nonce": "stale-voice-receipt-nonce",
            }
        )
        if (
            not stale_ledger.ok
            or stale_ledger.metadata.get("audit_ledger_state") != "VOICE_CONFIRMATION_AUDIT_LEDGER_HELD"
            or stale_ledger.metadata.get("audit_ledger_ready")
            or stale_ledger.metadata.get("supplied_confirmation_receipt_nonce_match") is not False
            or "matching supplied confirmation receipt nonce" not in stale_ledger.metadata.get("missing_blockers", [])
        ):
            raise SystemExit(f"Voice confirmation audit ledger should hold stale receipt nonce: {stale_ledger.metadata}")
        stale_gate = runtime.registry.get("voice_route_gate_packet").handler(
            {
                "transcript": "run command python3 --version",
                "confirmed": "true",
                "receipt_id": receipt.metadata.get("receipt_id"),
                "receipt_nonce": "stale-voice-receipt-nonce",
            }
        )
        if (
            not stale_gate.ok
            or stale_gate.metadata.get("route_state") != "VOICE_HELD_FOR_STALE_RECEIPT"
            or stale_gate.metadata.get("can_enter_runtime")
            or stale_gate.metadata.get("receipt_freshness_match") is not False
            or stale_gate.metadata.get("receipt_nonce_match") is not False
        ):
            raise SystemExit(f"Voice route gate should hold stale receipt nonce: {stale_gate.metadata}")
        stale_proof = runtime.registry.get("voice_route_proof_bundle").handler(
            {
                "transcript": "run command python3 --version",
                "confirmed": "true",
                "privacy_receipt_id": privacy_receipt_id,
                "receipt_id": receipt.metadata.get("receipt_id"),
                "receipt_nonce": "stale-voice-receipt-nonce",
            }
        )
        if (
            not stale_proof.ok
            or stale_proof.metadata.get("bundle_state") != "VOICE_ROUTE_PROOF_HELD"
            or stale_proof.metadata.get("bundle_ready")
            or stale_proof.metadata.get("receipt_freshness_match") is not False
            or "fresh confirmation receipt id and nonce for this exact transcript" not in stale_proof.metadata.get("bundle_missing", [])
        ):
            raise SystemExit(f"Voice route proof should hold stale receipt nonce: {stale_proof.metadata}")
        stale_cockpit = runtime.registry.get("voice_command_cockpit").handler(
            {
                "transcript": "run command python3 --version",
                "confirmed": "true",
                "privacy_receipt_id": privacy_receipt_id,
                "receipt_id": receipt.metadata.get("receipt_id"),
                "receipt_nonce": "stale-voice-receipt-nonce",
            }
        )
        if (
            not stale_cockpit.ok
            or stale_cockpit.metadata.get("cockpit_state") != "VOICE_COMMAND_COCKPIT_HELD"
            or stale_cockpit.metadata.get("ready_for_command_intake")
            or stale_cockpit.metadata.get("receipt_freshness_match") is not False
            or "fresh confirmation receipt id and nonce for this exact transcript" not in stale_cockpit.metadata.get("missing_blockers", [])
        ):
            raise SystemExit(f"Voice command cockpit should hold stale receipt nonce: {stale_cockpit.metadata}")
        mismatched_proof = runtime.registry.get("voice_route_proof_bundle").handler(
            {
                "transcript": "run command python3 --version",
                "confirmed": "true",
                "privacy_receipt_id": "voice-privacy-mismatch",
            }
        )
        if (
            not mismatched_proof.ok
            or mismatched_proof.metadata.get("bundle_state") != "VOICE_ROUTE_PROOF_HELD"
            or mismatched_proof.metadata.get("bundle_ready")
            or mismatched_proof.metadata.get("privacy_receipt_match") is not False
            or "matching privacy receipt from visible capture boundary" not in mismatched_proof.metadata.get("bundle_missing", [])
        ):
            raise SystemExit(f"Voice route proof should hold mismatched privacy receipts: {mismatched_proof.metadata}")
        mismatched_bridge = runtime.registry.get("voice_runtime_bridge_packet").handler(
            {
                "transcript": "run command python3 --version",
                "confirmed": "true",
                "privacy_receipt_id": "voice-privacy-mismatch",
            }
        )
        if mismatched_bridge.metadata.get("bridge_state") != "VOICE_HELD_BEFORE_COMMAND_INTAKE" or mismatched_bridge.metadata.get("can_enter_runtime"):
            raise SystemExit(f"Voice runtime bridge should hold mismatched privacy receipts: {mismatched_bridge.metadata}")
        command_cockpit = runtime.handle(
            f"voice command cockpit: run command python3 --version; confirmed=true; privacy_receipt_id={privacy_receipt_id}"
        )
        if (
            not command_cockpit.verified
            or command_cockpit.tool_results[0].metadata.get("cockpit_state") != "VOICE_COMMAND_COCKPIT_READY_FOR_COMMAND_INTAKE"
            or command_cockpit.tool_results[0].metadata.get("privacy_receipt_match") is not True
        ):
            raise SystemExit(f"Voice command parser should pass matching privacy receipts into cockpit: {command_cockpit.response}")
        command_cockpit_mismatch = runtime.handle(
            "voice command cockpit: run command python3 --version; confirmed=true; privacy_receipt_id=voice-privacy-mismatch"
        )
        if (
            not command_cockpit_mismatch.verified
            or command_cockpit_mismatch.tool_results[0].metadata.get("cockpit_state") != "VOICE_COMMAND_COCKPIT_HELD"
            or command_cockpit_mismatch.tool_results[0].metadata.get("privacy_receipt_match") is not False
        ):
            raise SystemExit(f"Voice command parser should hold mismatched privacy receipts: {command_cockpit_mismatch.response}")
        command_audit_ledger = runtime.handle(
            f"voice confirmation audit ledger: run command python3 --version; confirmed=true; privacy_receipt_id={privacy_receipt_id}; receipt_id={receipt.metadata.get('receipt_id')}; receipt_nonce={receipt.metadata.get('receipt_nonce')}"
        )
        if (
            not command_audit_ledger.verified
            or command_audit_ledger.tool_results[0].metadata.get("audit_ledger_state") != "VOICE_CONFIRMATION_AUDIT_LEDGER_READY"
            or command_audit_ledger.tool_results[0].metadata.get("audit_ledger_ready") is not True
        ):
            raise SystemExit(f"Voice confirmation audit parser should pass supplied receipts into ledger: {command_audit_ledger.response}")
        command_closure_mismatch = runtime.handle(
            "voice post-run closure: run command python3 --version; confirmed=true; privacy_receipt_id=voice-privacy-mismatch; verification reviewed; post health reviewed; post audit reviewed; learning reviewed"
        )
        if (
            not command_closure_mismatch.verified
            or command_closure_mismatch.tool_results[0].metadata.get("closure_state") != "VOICE_POST_RUN_CLOSURE_HELD"
            or "matching privacy receipt from visible capture boundary" not in command_closure_mismatch.tool_results[0].metadata.get("missing_blockers", [])
        ):
            raise SystemExit(f"Voice post-run closure parser should hold mismatched privacy receipts: {command_closure_mismatch.response}")
        oversized_path = voice_file_transcription_plan({"path": "x" * 2001})
        if oversized_path.ok or "too long" not in oversized_path.output:
            raise SystemExit("Voice file transcription plan should refuse oversized paths.")
        gate_for_transcription = runtime.registry.get("voice_audio_file_gate_packet").handler({"path": str(sample_audio), "consent": "true"})
        receipt_id = gate_for_transcription.metadata.get("receipt_id")
        fake_whisper_model = Path(temp) / "local-whisper-model.pt"
        fake_whisper_model.write_bytes(b"fake local whisper checkpoint for readiness only")
        original_find_spec = voice_tools.importlib.util.find_spec
        original_which = voice_tools.shutil.which
        original_warmup = os.environ.get("JARVIS_VOICE_WARMUP")
        try:
            os.environ["JARVIS_VOICE_WHISPER_MODEL_PATH"] = str(fake_whisper_model)
            os.environ["JARVIS_VOICE_WARMUP"] = "1"

            def fake_find_spec(name: str, *args, **kwargs):
                if name == "whisper":
                    return object()
                return original_find_spec(name, *args, **kwargs)

            def fake_which(name: str, *args, **kwargs):
                if name == "ffmpeg":
                    return "/usr/local/bin/ffmpeg"
                return original_which(name, *args, **kwargs)

            voice_tools.importlib.util.find_spec = fake_find_spec  # type: ignore[assignment]
            voice_tools.shutil.which = fake_which  # type: ignore[assignment]
            env_setup = runtime.registry.get("voice_setup_check").handler({})
            env_plan = runtime.registry.get("voice_file_transcription_plan").handler({"path": str(sample_audio)})
            env_gate = runtime.registry.get("voice_audio_file_gate_packet").handler({"path": str(sample_audio), "consent": "true"})
        finally:
            os.environ.pop("JARVIS_VOICE_WHISPER_MODEL_PATH", None)
            if original_warmup is None:
                os.environ.pop("JARVIS_VOICE_WARMUP", None)
            else:
                os.environ["JARVIS_VOICE_WARMUP"] = original_warmup
            voice_tools.importlib.util.find_spec = original_find_spec  # type: ignore[assignment]
            voice_tools.shutil.which = original_which  # type: ignore[assignment]
        for label, result in {"env setup": env_setup, "env plan": env_plan, "env gate": env_gate}.items():
            if result.metadata.get("local_audio_transcriber_configured") is not True:
                raise SystemExit(f"{label} should detect configured local Whisper model path: {result.metadata}")
            if result.metadata.get("local_audio_transcriber_source") != "local_whisper_model_path":
                raise SystemExit(f"{label} missed local Whisper transcriber source: {result.metadata}")
            if result.metadata.get("approved_transcription_runner_ready") is not True:
                raise SystemExit(f"{label} should mark approved runner ready when ffmpeg and local transcriber are present: {result.metadata}")
        if env_setup.metadata.get("voice_warmup_enabled") is not True or env_setup.metadata.get("voice_warmup_ready") is not True:
            raise SystemExit(f"env setup should mark opt-in warmup ready with a local transcriber: {env_setup.metadata}")
        if "Whisper warm start (`JARVIS_VOICE_WARMUP=1`): enabled" not in env_setup.output:
            raise SystemExit(f"env setup should render enabled warmup state: {env_setup.output}")
        if env_gate.metadata.get("gate_state") != "AUDIO_FILE_READY_FOR_APPROVED_TRANSCRIPTION" or env_gate.metadata.get("missing_checks"):
            raise SystemExit(f"voice audio file gate should become ready with a configured local transcriber: {env_gate.metadata}")
        if env_gate.metadata.get("reads_audio_file") or env_gate.metadata.get("transcribes_audio"):
            raise SystemExit(f"env-configured voice audio gate should remain read-only: {env_gate.metadata}")
        original_transcriber = voice_tools._AUDIO_FILE_TRANSCRIBER
        voice_tools._AUDIO_FILE_TRANSCRIBER = lambda path: "run command python3 --version"  # type: ignore[assignment]
        try:
            transcription_preview = runtime.registry.get("voice_audio_file_transcription_preview").handler(
                {"path": str(sample_audio), "consent": "true", "receipt_id": receipt_id}
            )
        finally:
            voice_tools._AUDIO_FILE_TRANSCRIBER = original_transcriber
        if not transcription_preview.ok or "Jarvis audio-file transcription preview" not in transcription_preview.output:
            raise SystemExit(f"Voice audio transcription preview should return a mocked transcript preview: {transcription_preview}")
        preview_metadata = transcription_preview.metadata
        preview_handoff = preview_metadata.get("voice_audio_file_transcription_handoff")
        if not preview_metadata.get("voice_audio_file_transcription_handoff_ready") or not isinstance(preview_handoff, dict):
            raise SystemExit(f"Voice audio transcription preview missed handoff: {preview_metadata}")
        if preview_handoff.get("handoff_ready") is not True:
            raise SystemExit(f"Voice audio transcription preview handoff should mark itself ready: {preview_handoff}")
        for field, expected in (
            ("content_in_handoff", True),
            ("state_changed", False),
            ("changed", []),
            ("authorizes_execution", False),
            ("authorizes_completion_claim", False),
            ("approval_granted", False),
        ):
            if preview_metadata.get(field) != expected or preview_handoff.get(field) != expected:
                raise SystemExit(f"Voice audio transcription preview missed {field}: {preview_metadata} / {preview_handoff}")
        for field in (
            "ready_for_operator",
            "state_changed",
            "changed",
            "content_in_handoff",
            "boundaries",
            "next_safe_command",
        ):
            metadata_key = f"voice_audio_file_transcription_{field}"
            if preview_metadata.get(metadata_key) != preview_handoff.get(field):
                raise SystemExit(
                    f"Voice audio transcription preview missed flat handoff alias {metadata_key}: "
                    f"{preview_metadata} / {preview_handoff}"
                )
        for key in ("reads_private_data", "reads_personal_data", "reads_audio_file", "transcribes_audio", "requires_approval"):
            if preview_metadata.get(key) is not True:
                raise SystemExit(f"Voice audio transcription preview should mark {key}: {preview_metadata}")
        for key in ("saves_transcript", "writes_files", "writes_database", "writes_memory", "writes_notes", "executes_tools", "records_audio", "starts_listener", "queues_approval"):
            if preview_metadata.get(key):
                raise SystemExit(f"Voice audio transcription preview unsafe metadata {key}: {preview_metadata}")
        if preview_handoff.get("transcript_sha256") != preview_metadata.get("transcript_sha256"):
            raise SystemExit(f"Voice audio transcription preview hash parity failed: {preview_metadata} / {preview_handoff}")
        if preview_metadata.get("voice_audio_file_transcription_transcript_sha256") != preview_handoff.get("transcript_sha256"):
            raise SystemExit(f"Voice audio transcription preview flat hash alias failed: {preview_metadata} / {preview_handoff}")
        if preview_metadata.get("voice_audio_file_transcription_transcript_chars") != preview_handoff.get("transcript_chars"):
            raise SystemExit(f"Voice audio transcription preview flat char-count alias failed: {preview_metadata} / {preview_handoff}")
        if preview_metadata.get("next_command") != preview_handoff.get("next_safe_command"):
            raise SystemExit(f"Voice audio transcription preview next command parity failed: {preview_metadata} / {preview_handoff}")
        boundaries = preview_handoff.get("boundaries") or {}
        if boundaries.get("read_only") is not False or boundaries.get("reads_audio_file") is not True or boundaries.get("transcribes_audio") is not True:
            raise SystemExit(f"Voice audio transcription preview boundaries should describe approved personal-data read: {preview_handoff}")
        held_without_transcriber = runtime.registry.get("voice_audio_file_transcription_preview").handler(
            {"path": str(sample_audio), "consent": "true", "receipt_id": receipt_id}
        )
        if held_without_transcriber.ok or "local audio transcriber configured" not in held_without_transcriber.output:
            raise SystemExit(f"Voice audio transcription preview should hold when no local transcriber is configured: {held_without_transcriber}")
        if held_without_transcriber.metadata.get("reads_audio_file") or held_without_transcriber.metadata.get("transcribes_audio"):
            raise SystemExit(f"Voice audio transcription preview should not touch audio when held: {held_without_transcriber.metadata}")
        approval_runtime = make_temp_runtime(Path(temp) / "ApprovalRuntime")
        queued = approval_runtime.handle(f"voice audio file transcribe: {sample_audio} consent=true receipt_id={receipt_id}")
        pending = approval_runtime.store.list_pending_approvals(limit=10)
        if queued.verified or len(pending) != 1:
            raise SystemExit(f"Voice audio transcription command should queue approval before handler execution: {queued.response}")
        queued_metadata = queued.tool_results[0].metadata
        if (
            queued_metadata.get("failure_kind") != "approval_required"
            or queued_metadata.get("risk_level") != "PERSONAL_DATA"
            or queued_metadata.get("executed_handler") is not False
            or queued_metadata.get("requires_confirmation") is not True
        ):
            raise SystemExit(f"Voice audio transcription command missed approval gate metadata: {queued_metadata}")
        if pending[0]["tool_name"] != "voice_audio_file_transcription_preview":
            raise SystemExit(f"Voice audio transcription queued the wrong tool: {dict(pending[0])}")
        alias_approval_runtime = make_temp_runtime(Path(temp) / "AliasApprovalRuntime")
        alias_queued = alias_approval_runtime.handle(f"voice note transcribe: {sample_audio} consent=true receipt_id={receipt_id}")
        alias_pending = alias_approval_runtime.store.list_pending_approvals(limit=10)
        if alias_queued.verified or len(alias_pending) != 1:
            raise SystemExit(f"Voice note transcription alias should queue approval before handler execution: {alias_queued.response}")
        alias_metadata = alias_queued.tool_results[0].metadata
        if (
            alias_metadata.get("failure_kind") != "approval_required"
            or alias_metadata.get("risk_level") != "PERSONAL_DATA"
            or alias_metadata.get("executed_handler") is not False
            or alias_metadata.get("requires_confirmation") is not True
        ):
            raise SystemExit(f"Voice note transcription alias missed approval gate metadata: {alias_metadata}")
        if alias_pending[0]["tool_name"] != "voice_audio_file_transcription_preview":
            raise SystemExit(f"Voice note transcription alias queued the wrong tool: {dict(alias_pending[0])}")
        direct_input = voice_input_plan({"mode": "push-to-talk"})
        if direct_input.metadata.get("queues_approval") or direct_input.metadata.get("records_audio"):
            raise SystemExit("Voice input plan should stay read-only.")
        bounded_input = voice_input_plan({"mode": "x" * 200})
        if len(bounded_input.metadata.get("mode", "")) != 80:
            raise SystemExit("Voice input plan should bound mode text.")
        for key in VOICE_SAFE_FALSE_FLAGS:
            if direct_input.metadata.get(key):
                raise SystemExit(f"Direct voice input plan unsafe metadata {key}: {direct_input.metadata}")
        if runtime.store.list_pending_approvals(limit=10):
            raise SystemExit("Voice transcript review should not queue pending approvals.")
        memory_result = runtime.handle("search memory for voice transcripts are previewed first")
        if "No memories found" not in memory_result.response:
            raise SystemExit("Voice transcript review should not store previewed memories.")
        packet_memory_result = runtime.handle("search memory for confirmed voice packets stay read only")
        if "No memories found" not in packet_memory_result.response:
            raise SystemExit("Voice confirmation packet should not store previewed memories.")
        receipt_memory_result = runtime.handle("search memory for confirmed voice receipts stay read only")
        if "No memories found" not in receipt_memory_result.response:
            raise SystemExit("Voice confirmation receipt should not store previewed memories.")


if __name__ == "__main__":
    main()
