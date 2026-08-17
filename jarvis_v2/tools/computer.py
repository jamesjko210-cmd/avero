from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    declare_failure_guidance,
    declare_outcome_unknown_failure,
    declare_retryable_local_read_failure,
)
from jarvis_v2.agent.types import ToolResult


_enabled = False
_SCREEN_VISION_REVIEWER: Any | None = None
MAX_OBJECTIVE_CHARS = 240
MAX_EXPECTATION_CHARS = 220
MAX_OBSERVATION_CHARS = 600
MAX_TYPED_TEXT_CHARS = 1000
MAX_FILENAME_CHARS = 120
MAX_COORDINATE = 10000
DEFAULT_OBSERVATION_MAX_AGE_SECONDS = 300
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")
OPERATOR_LIMIT_RULE = (
    "the operator's explicit stop times, work windows, pause commands, and newer instructions "
    "override computer-control readiness, approved-looking actions, and priority goals."
)
COMPUTER_BOUNDARY = (
    "\n\nOperator limit:\n"
    f"- {OPERATOR_LIMIT_RULE}"
)
COMPUTER_INPUT_RECOVERY_ACTION = (
    "Correct the reported input, then submit a fresh request through the normal approval policy."
)
COMPUTER_READ_RECOVERY_ACTION = (
    "Run `computer control readiness`, correct Accessibility or Screen Recording access, then "
    "retry through the normal approval policy."
)
COMPUTER_CONTROL_RECOVERY_ACTION = (
    "Inspect the current screen and computer state before deciding whether to submit a fresh "
    "approved request; do not retry automatically."
)


def _local_screen_vision_command_path() -> Path | None:
    raw = str(os.getenv("JARVIS_OAV_VISION_REVIEWER_COMMAND") or "").strip()
    if not raw:
        return None
    command = shlex.split(raw)[0] if raw else ""
    if not command:
        return None
    path = Path(command).expanduser()
    if not path.exists() or not path.is_file():
        return None
    return path


def _screen_vision_reviewer_source() -> str:
    if _SCREEN_VISION_REVIEWER is not None:
        return "in_process_screen_vision_reviewer"
    if _local_screen_vision_command_path() is not None:
        return "local_vision_reviewer_command"
    return "none"


def _screen_vision_reviewer_configured() -> bool:
    return _screen_vision_reviewer_source() != "none"


def _screen_vision_reviewer() -> Any | None:
    if _SCREEN_VISION_REVIEWER is not None:
        return _SCREEN_VISION_REVIEWER
    raw = str(os.getenv("JARVIS_OAV_VISION_REVIEWER_COMMAND") or "").strip()
    command_path = _local_screen_vision_command_path()
    if not raw or command_path is None:
        return None
    parts = shlex.split(raw)
    parts[0] = str(command_path)

    def run_local_command(image_path: Path, prompt_sections: dict[str, str]) -> str:
        payload = json.dumps({"image_path": str(image_path), "prompt_sections": prompt_sections}, sort_keys=True)
        completed = subprocess.run(
            [*parts, str(image_path)],
            input=payload,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "local vision reviewer failed").strip()
            raise RuntimeError(detail[:500])
        return (completed.stdout or "").strip()

    return run_local_command


def _clean_text(value: Any, *, limit: int) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _bounded_coordinate(value: Any) -> int | None:
    if not _is_int_like(value):
        return None
    return max(0, min(MAX_COORDINATE, int(value)))


def _bounded_age_seconds(value: Any, default: int = DEFAULT_OBSERVATION_MAX_AGE_SECONDS) -> int:
    if not _is_int_like(value):
        return default
    return max(1, min(3600, int(value)))


def _parse_iso_datetime(value: str) -> datetime | None:
    text = _clean_text(value, limit=80)
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "calls_model": False,
        "calls_external_service": False,
        "executes_tools": False,
        "reads_personal_data": False,
        "reads_private_data": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "requires_approval": False,
        "observes_screen": False,
        "takes_screenshot": False,
        "controls_computer": False,
        "reads_clipboard": False,
        "runs_shell": False,
        "writes_files": False,
        "speaks": False,
        "completes_tasks": False,
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
    }
    metadata.update(extra)
    return metadata


def _screen_observation_freshness_ready_from_metadata(metadata: dict[str, Any]) -> bool:
    receipt_metadata = metadata.get("approved_screen_observation_receipt_metadata")
    missing = metadata.get("missing_freshness_proof")
    age_seconds = metadata.get("observation_age_seconds")
    max_age_seconds = metadata.get("max_age_seconds")
    if not isinstance(receipt_metadata, dict):
        return False
    if not isinstance(missing, list) or metadata.get("missing_freshness_proof_count") != len(missing):
        return False
    if missing:
        return False
    if not isinstance(age_seconds, int) or not isinstance(max_age_seconds, int):
        return False
    if age_seconds < 0 or age_seconds > max_age_seconds:
        return False
    expected_true = {
        "durable_receipt_ready",
        "real_screenshot_adapter_ready",
        "timestamp_parse_ready",
        "observation_not_future",
        "within_freshness_window",
        "freshness_ready",
        "fresh_enough_for_one_primitive",
        "ready_for_oav_proof_review",
        "fresh_receipt_required_for_each_primitive",
    }
    for key in expected_true:
        if metadata.get(key) is not True:
            return False
    if metadata.get("freshness_state") != "SCREEN_OBSERVATION_FRESHNESS_READY":
        return False
    if receipt_metadata.get("receipt_state") != "APPROVED_SCREEN_OBSERVATION_RECEIPT_READY":
        return False
    if receipt_metadata.get("real_screenshot_adapter_ready") is not True:
        return False
    if receipt_metadata.get("fresh_receipt_required_for_each_primitive") is not True:
        return False
    for key in (
        "calls_model",
        "calls_external_service",
        "executes_tools",
        "reads_personal_data",
        "reads_private_data",
        "executes_side_effect",
        "external_side_effect",
        "writes_files",
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "requires_approval",
        "observes_screen",
        "takes_screenshot",
        "controls_computer",
        "reads_clipboard",
        "runs_shell",
    ):
        if metadata.get(key) is not False:
            return False
    return True


def _oav_final_review_ready_from_metadata(metadata: dict[str, Any]) -> bool:
    cockpit_metadata = metadata.get("cockpit_metadata")
    missing = metadata.get("missing_blockers")
    required_commands = metadata.get("required_commands")
    if not isinstance(cockpit_metadata, dict):
        return False
    if not isinstance(missing, list) or metadata.get("missing_blocker_count") != len(missing):
        return False
    if missing:
        return False
    if not isinstance(required_commands, list) or metadata.get("required_command_count") != len(required_commands):
        return False
    for command in ("execution audit gate", "execution health report", "human final review"):
        if command not in required_commands:
            return False
    expected_true = {
        "ready_for_human_decision",
        "oav_final_review_ready",
        "final_route_review_ready",
        "execution_audit_evidence_ready",
        "execution_health_evidence_ready",
        "operator_final_review_evidence_ready",
        "real_execution_approval_gated",
        "approval_required_for_real_use",
    }
    for key in expected_true:
        if metadata.get(key) is not True:
            return False
    if metadata.get("final_review_state") != "OAV_FINAL_REVIEW_READY_FOR_HUMAN_DECISION":
        return False
    if metadata.get("cockpit_state") != "OAV_COCKPIT_READY_FOR_FINAL_REVIEW":
        return False
    if cockpit_metadata.get("cockpit_state") != "OAV_COCKPIT_READY_FOR_FINAL_REVIEW":
        return False
    if cockpit_metadata.get("final_route_review_ready") is not True:
        return False
    if cockpit_metadata.get("missing_blockers") != []:
        return False
    if metadata.get("natural_language_routing_enabled") is not False:
        return False
    if metadata.get("natural_language_computer_control_routing_enabled") is not False:
        return False
    if cockpit_metadata.get("natural_language_computer_control_routing_enabled") is not False:
        return False
    if metadata.get("action_allowed_now") is not False:
        return False
    if metadata.get("computer_control_enabled") is not False:
        return False
    if cockpit_metadata.get("computer_control_enabled") is not False:
        return False
    for key in (
        "calls_model",
        "calls_external_service",
        "executes_tools",
        "reads_personal_data",
        "reads_private_data",
        "executes_side_effect",
        "external_side_effect",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "writes_files",
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "requires_approval",
        "observes_screen",
        "takes_screenshot",
        "controls_computer",
        "reads_clipboard",
        "runs_shell",
        "speaks",
        "completes_tasks",
    ):
        if metadata.get(key) is not False:
            return False
    return True


def _oav_action_audit_ready_from_metadata(metadata: dict[str, Any]) -> bool:
    final_review_metadata = metadata.get("final_review_metadata")
    missing = metadata.get("missing_blockers")
    required_commands = metadata.get("required_commands")
    primitive_command = str(metadata.get("exact_primitive_command") or "").strip()
    if not isinstance(final_review_metadata, dict):
        return False
    if not _oav_final_review_ready_from_metadata(final_review_metadata):
        return False
    if not isinstance(missing, list) or metadata.get("missing_blocker_count") != len(missing):
        return False
    if missing:
        return False
    if not isinstance(required_commands, list) or metadata.get("required_command_count") != len(required_commands):
        return False
    for command in (
        "observe act verify action audit",
        primitive_command,
        "verification receipt <approved run id>",
        "execution health report",
        "after-action learning packet <approved run id>",
    ):
        if command not in required_commands:
            return False
    if not primitive_command or "<" in primitive_command:
        return False
    expected_true = {
        "ready_for_approval_decision",
        "oav_action_audit_ready",
        "ready_for_human_decision",
        "approval_decision_required",
        "real_execution_approval_gated",
        "approval_required_for_real_use",
    }
    for key in expected_true:
        if metadata.get(key) is not True:
            return False
    if metadata.get("action_audit_state") != "OAV_ACTION_AUDIT_READY_FOR_APPROVAL_DECISION":
        return False
    if metadata.get("final_review_state") != "OAV_FINAL_REVIEW_READY_FOR_HUMAN_DECISION":
        return False
    if metadata.get("action_allowed_now") is not False:
        return False
    if metadata.get("computer_control_enabled") is not False:
        return False
    if metadata.get("natural_language_routing_enabled") is not False:
        return False
    if metadata.get("natural_language_computer_control_routing_enabled") is not False:
        return False
    for key in (
        "calls_model",
        "calls_external_service",
        "executes_tools",
        "reads_personal_data",
        "reads_private_data",
        "executes_side_effect",
        "external_side_effect",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "writes_files",
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "requires_approval",
        "observes_screen",
        "takes_screenshot",
        "controls_computer",
        "reads_clipboard",
        "runs_shell",
        "speaks",
        "completes_tasks",
    ):
        if metadata.get(key) is not False:
            return False
    return True


def _oav_execution_handoff_ready_from_metadata(metadata: dict[str, Any]) -> bool:
    action_audit_metadata = metadata.get("action_audit_metadata")
    missing = metadata.get("missing_blockers")
    required_commands = metadata.get("required_commands")
    post_run_commands = metadata.get("post_run_commands")
    primitive_command = str(metadata.get("exact_primitive_command") or "").strip()
    approval_readiness_command = str(metadata.get("approval_readiness_command") or "").strip()
    approval_packet_command = str(metadata.get("approval_packet_command") or "").strip()
    approval_chain_command = str(metadata.get("approval_chain_command") or "").strip()
    if not isinstance(action_audit_metadata, dict):
        return False
    if not _oav_action_audit_ready_from_metadata(action_audit_metadata):
        return False
    if not isinstance(missing, list) or metadata.get("missing_blocker_count") != len(missing):
        return False
    if missing:
        return False
    if not isinstance(required_commands, list) or metadata.get("required_command_count") != len(required_commands):
        return False
    if not isinstance(post_run_commands, list) or metadata.get("post_run_command_count") != len(post_run_commands):
        return False
    if not primitive_command or "<" in primitive_command:
        return False
    if not approval_readiness_command.startswith("approval readiness <approval id for `"):
        return False
    if not approval_packet_command.startswith("approval packet <approval id for `"):
        return False
    if not approval_chain_command.startswith("approval chain proof <approval id for `"):
        return False
    if primitive_command not in approval_readiness_command or primitive_command not in approval_packet_command or primitive_command not in approval_chain_command:
        return False
    expected_post_run_commands = [
        "verification receipt <approved run id>",
        "execution health report",
        "execution audit gate",
        "after-action learning packet <approved run id>",
    ]
    if post_run_commands != expected_post_run_commands:
        return False
    for command in [
        "observe act verify execution handoff",
        approval_readiness_command,
        approval_packet_command,
        primitive_command,
        approval_chain_command,
        *expected_post_run_commands,
    ]:
        if command not in required_commands:
            return False
    expected_true = {
        "ready_for_approval_decision",
        "oav_execution_handoff_ready",
        "approval_decision_required",
        "real_execution_approval_gated",
        "approval_required_for_real_use",
    }
    for key in expected_true:
        if metadata.get(key) is not True:
            return False
    if metadata.get("handoff_state") != "OAV_EXECUTION_HANDOFF_READY_FOR_APPROVAL_PACKET":
        return False
    if metadata.get("action_audit_state") != "OAV_ACTION_AUDIT_READY_FOR_APPROVAL_DECISION":
        return False
    if metadata.get("final_review_state") != "OAV_FINAL_REVIEW_READY_FOR_HUMAN_DECISION":
        return False
    if metadata.get("action_allowed_now") is not False:
        return False
    if metadata.get("computer_control_enabled") is not False:
        return False
    if metadata.get("natural_language_routing_enabled") is not False:
        return False
    if metadata.get("natural_language_computer_control_routing_enabled") is not False:
        return False
    for key in (
        "calls_model",
        "calls_external_service",
        "executes_tools",
        "reads_personal_data",
        "reads_private_data",
        "executes_side_effect",
        "external_side_effect",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "writes_files",
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "requires_approval",
        "observes_screen",
        "takes_screenshot",
        "controls_computer",
        "reads_clipboard",
        "runs_shell",
        "speaks",
        "completes_tasks",
    ):
        if metadata.get(key) is not False:
            return False
    return True


def _oav_post_run_closure_ready_from_metadata(metadata: dict[str, Any]) -> bool:
    handoff_metadata = metadata.get("execution_handoff_metadata")
    if not isinstance(handoff_metadata, dict):
        handoff_metadata = metadata.get("handoff_metadata")
    missing = metadata.get("missing_blockers")
    required_commands = metadata.get("required_commands")
    approved_run_id = metadata.get("approved_run_id")
    verification_run_id = metadata.get("verification_run_id")
    if not isinstance(handoff_metadata, dict):
        return False
    if not _oav_execution_handoff_ready_from_metadata(handoff_metadata):
        return False
    if not isinstance(missing, list) or metadata.get("missing_blocker_count") != len(missing):
        return False
    if missing:
        return False
    if not isinstance(required_commands, list) or metadata.get("required_command_count") != len(required_commands):
        return False
    if not isinstance(approved_run_id, int) or not isinstance(verification_run_id, int):
        return False
    if approved_run_id != verification_run_id:
        return False
    expected_true = {
        "ready_for_next_primitive_review",
        "oav_post_run_closure_ready",
        "next_review_requires_fresh_observation",
        "next_review_requires_new_route_lock",
        "next_review_requires_new_approval_bridge",
        "next_review_requires_new_final_review",
        "next_review_requires_new_action_audit",
        "next_review_requires_new_execution_handoff",
        "prior_primitive_proof_only",
        "receipt_binding_ready",
        "post_run_verification_ready",
        "post_run_health_ready",
        "post_run_audit_ready",
        "after_action_learning_ready",
        "real_execution_approval_gated",
        "approval_required_for_real_use",
    }
    for key in expected_true:
        if metadata.get(key) is not True:
            return False
    for key in (
        "previous_approval_reusable_for_next_primitive",
        "previous_observation_reusable_for_next_primitive",
        "previous_verification_reusable_for_next_primitive",
        "previous_approved_run_reusable_for_next_primitive",
        "action_allowed_now",
        "natural_language_routing_enabled",
        "natural_language_computer_control_routing_enabled",
    ):
        if metadata.get(key) is not False:
            return False
    if metadata.get("closure_state") != "OAV_POST_RUN_CLOSURE_READY_FOR_NEXT_PRIMITIVE_REVIEW":
        return False
    if metadata.get("next_primitive_review_state") != "FRESH_OAV_REVIEW_UNLOCKED":
        return False
    if metadata.get("handoff_state") != "OAV_EXECUTION_HANDOFF_READY_FOR_APPROVAL_PACKET":
        return False
    if metadata.get("action_audit_state") != "OAV_ACTION_AUDIT_READY_FOR_APPROVAL_DECISION":
        return False
    if metadata.get("final_review_state") != "OAV_FINAL_REVIEW_READY_FOR_HUMAN_DECISION":
        return False
    if not str(metadata.get("next_review_start_command") or "").startswith("observe act verify cockpit:"):
        return False
    expected_commands = {
        f"verification receipt {approved_run_id}",
        "execution health report",
        "execution audit gate",
        f"after-action learning packet {approved_run_id}",
        "observe act verify post-run closure",
    }
    if not expected_commands.issubset(set(required_commands)):
        return False
    for key in (
        "calls_model",
        "calls_external_service",
        "executes_tools",
        "reads_personal_data",
        "reads_private_data",
        "executes_side_effect",
        "external_side_effect",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "writes_files",
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "requires_approval",
        "observes_screen",
        "takes_screenshot",
        "controls_computer",
        "reads_clipboard",
        "runs_shell",
        "speaks",
        "completes_tasks",
    ):
        if metadata.get(key) is not False:
            return False
    return True


def _with_operator_limit(body: str) -> str:
    return f"{body}{COMPUTER_BOUNDARY}"


def _computer_error(action: str, exc: Exception | None = None) -> str:
    return (
        f"Could not {action}. Check System Settings > Privacy & Security > Accessibility and Screen "
        "Recording."
    )


def _computer_input_failure(
    tool_name: str,
    message: str,
    *,
    commands: tuple[str, ...] = (),
    **metadata: Any,
) -> ToolResult:
    output = f"{message} {COMPUTER_INPUT_RECOVERY_ACTION}"
    controls_computer = metadata.pop("controls_computer", True)
    declared = declare_retryable_local_read_failure(
        _safe_metadata(
            controls_computer=controls_computer,
            computer_control_enabled=_enabled,
            **metadata,
        ),
        output=output,
        action=COMPUTER_INPUT_RECOVERY_ACTION,
        commands=commands,
    )
    return ToolResult(tool_name, False, _with_operator_limit(output), declared)


def _computer_read_failure(
    tool_name: str,
    action: str,
    exc: Exception,
    **metadata: Any,
) -> ToolResult:
    output = f"{_computer_error(action)} {COMPUTER_READ_RECOVERY_ACTION}"
    declared = declare_retryable_local_read_failure(
        _safe_metadata(exception_type=type(exc).__name__, **metadata),
        output=output,
        action=COMPUTER_READ_RECOVERY_ACTION,
        commands=("computer control readiness",),
    )
    return ToolResult(tool_name, False, _with_operator_limit(output), declared)


def _computer_attempt_failure(
    tool_name: str,
    action: str,
    exc: Exception,
    **metadata: Any,
) -> ToolResult:
    canonical_output = (
        f"Could not {action}. The outcome is unknown. {COMPUTER_CONTROL_RECOVERY_ACTION}"
    )
    output = (
        f"{canonical_output} Check System Settings > Privacy & Security > Accessibility and Screen "
        "Recording, then run `computer control readiness`."
    )
    declared = declare_outcome_unknown_failure(
        _safe_metadata(
            controls_computer=True,
            computer_control_enabled=_enabled,
            exception_type=type(exc).__name__,
            **metadata,
        ),
        output=canonical_output,
    )
    return ToolResult(tool_name, False, _with_operator_limit(output), declared)


def _oav_cycle_ledger_token_sha256(
    *,
    action: str,
    x: Any,
    y: Any,
    expectation: str,
    approved_run_id: Any,
    verification_run_id: Any,
    cycle_state: str,
    fresh_review_contract_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
) -> str:
    contract_payload = [
        {
            "item": str(row.get("item") or ""),
            "source": str(row.get("source") or ""),
            "required": row.get("required") is True,
            "prior_artifact_reusable": row.get("prior_artifact_reusable") is True,
            "authorizes_action_now": row.get("authorizes_action_now") is True,
            "authorizes_computer_control": row.get("authorizes_computer_control") is True,
            "authorizes_screenshot": row.get("authorizes_screenshot") is True,
            "authorizes_approval": row.get("authorizes_approval") is True,
            "authorizes_route_unlock": row.get("authorizes_route_unlock") is True,
            "authorizes_verification_shortcut": row.get("authorizes_verification_shortcut") is True,
            "authorizes_model_call": row.get("authorizes_model_call") is True,
            "authorizes_tool_execution": row.get("authorizes_tool_execution") is True,
            "authorizes_personal_data_read": row.get("authorizes_personal_data_read") is True,
            "authorizes_external_side_effect": row.get("authorizes_external_side_effect") is True,
        }
        for row in fresh_review_contract_rows
    ]
    stage_payload = [
        {
            "stage": str(row.get("stage") or ""),
            "state": str(row.get("state") or ""),
            "ready": row.get("ready") is True,
            "reusable_for_next_primitive": row.get("reusable_for_next_primitive") is True,
        }
        for row in stage_rows
    ]
    payload = {
        "kind": "observe_act_verify_cycle_ledger_token_v1",
        "action": str(action or ""),
        "x": x,
        "y": y,
        "expectation": str(expectation or ""),
        "approved_run_id": approved_run_id,
        "verification_run_id": verification_run_id,
        "cycle_state": str(cycle_state or ""),
        "fresh_review_contract_rows": contract_payload,
        "stage_rows": stage_payload,
        "authorizes_action_now": False,
        "authorizes_computer_control": False,
        "authorizes_screenshot": False,
        "authorizes_approval": False,
        "authorizes_route_unlock": False,
        "authorizes_model_call": False,
        "authorizes_tool_execution": False,
        "authorizes_personal_data_read": False,
        "authorizes_external_side_effect": False,
        "reusable_for_next_primitive": False,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _oav_cycle_ledger_token_boundary_ready(token_sha256: Any, rows: Any) -> bool:
    token = str(token_sha256 or "")
    if len(token) != 64 or any(char not in "0123456789abcdef" for char in token.lower()):
        return False
    if not isinstance(rows, list) or len(rows) != 4:
        return False
    expected = [
        ("oav_cycle_ledger_token", "present", "observe_act_verify_cycle_ledger"),
        ("fresh_review_contract_boundary", "prior_primitive_proof_only_not_reusable", "fresh_review_contract_rows"),
        ("stage_row_boundary", "prior_stage_proof_only_not_reusable", "stage_rows"),
        ("next_primitive_review_boundary", "fresh_oav_review_required", "observe_act_verify_cycle_ledger"),
    ]
    for row, (item, status, source) in zip(rows, expected):
        if not isinstance(row, dict):
            return False
        if row.get("item") != item or row.get("status") != status or row.get("source") != source:
            return False
        if row.get("token_sha256") != token:
            return False
        if row.get("proof_only") is not True or row.get("reusable_for_next_primitive") is not False:
            return False
        for key in [
            "authorizes_action_now",
            "authorizes_computer_control",
            "authorizes_screenshot",
            "authorizes_approval",
            "authorizes_route_unlock",
            "authorizes_model_call",
            "authorizes_tool_execution",
            "authorizes_personal_data_read",
            "authorizes_external_side_effect",
        ]:
            if row.get(key) is not False:
                return False
    return True


def _oav_cycle_ledger_ready_from_metadata(metadata: dict[str, Any]) -> bool:
    closure_metadata = metadata.get("post_run_closure_metadata")
    missing = metadata.get("missing_blockers")
    required_commands = metadata.get("required_commands")
    stage_rows = metadata.get("stage_rows")
    fresh_review_queue = metadata.get("fresh_review_preflight_queue")
    fresh_review_contract_rows = metadata.get("fresh_review_contract_rows")
    token_sha256 = metadata.get("oav_cycle_ledger_token_sha256")
    token_rows = metadata.get("oav_cycle_ledger_token_boundary_rows")
    if not isinstance(closure_metadata, dict):
        return False
    if not _oav_post_run_closure_ready_from_metadata(closure_metadata):
        return False
    if not isinstance(missing, list) or metadata.get("missing_blocker_count") != len(missing):
        return False
    if missing:
        return False
    if not isinstance(required_commands, list) or metadata.get("required_command_count") != len(required_commands):
        return False
    if not isinstance(stage_rows, list) or metadata.get("stage_count") != len(stage_rows):
        return False
    expected_stages = {
        "screen_observation_confidence",
        "screen_verification_contract",
        "observe_act_verify_proof",
        "route_lock",
        "approval_bridge",
        "cockpit",
        "final_review",
        "action_audit",
        "execution_handoff",
        "post_run_closure",
    }
    if {str(row.get("stage") or "") for row in stage_rows if isinstance(row, dict)} != expected_stages:
        return False
    for row in stage_rows:
        if not isinstance(row, dict):
            return False
        if row.get("ready") is not True or row.get("reusable_for_next_primitive") is not False:
            return False
        for key in (
            "authorizes_action_now",
            "authorizes_computer_control",
            "authorizes_screenshot",
            "authorizes_approval",
            "authorizes_route_unlock",
            "authorizes_model_call",
            "authorizes_tool_execution",
            "authorizes_personal_data_read",
            "authorizes_external_side_effect",
        ):
            if row.get(key) is not False:
                return False
    if not isinstance(fresh_review_queue, list) or metadata.get("fresh_review_preflight_queue_count") != len(fresh_review_queue):
        return False
    expected_fresh_review_queue = [
        "screen observation freshness: <next primitive approved observation>",
        "observe act verify proof: action <next primitive>; expectation <expected screen change>; observation <fresh approved observation>; source approved_screenshot",
        "observe act verify route lock: action <next primitive>; approval <new approval id>",
        "observe act verify approval bridge: action <next primitive>; approval <new approval id>; approved run <new approved run id>; verification receipt <new receipt id>",
        "observe act verify final review: action <next primitive>",
        "observe act verify action audit: action <next primitive>",
        "observe act verify execution handoff: action <next primitive>",
        "observe act verify post-run closure: action <next primitive>",
    ]
    if fresh_review_queue != expected_fresh_review_queue:
        return False
    if metadata.get("fresh_review_next_preflight_command") != expected_fresh_review_queue[0]:
        return False
    if not isinstance(fresh_review_contract_rows, list) or metadata.get("fresh_review_contract_row_count") != len(fresh_review_contract_rows):
        return False
    expected_contract_items = {
        "fresh_screen_observation",
        "new_route_lock",
        "new_approval_bridge",
        "new_final_review",
        "new_action_audit",
        "new_execution_handoff",
        "new_post_run_closure",
        "fresh_cycle_ledger_review",
    }
    if len(fresh_review_contract_rows) != len(expected_contract_items):
        return False
    if {str(row.get("item") or "") for row in fresh_review_contract_rows if isinstance(row, dict)} != expected_contract_items:
        return False
    if metadata.get("fresh_review_contract_ready") is not True:
        return False
    for row in fresh_review_contract_rows:
        if not isinstance(row, dict):
            return False
        if row.get("required") is not True or row.get("prior_artifact_reusable") is not False:
            return False
        for key in (
            "authorizes_action_now",
            "authorizes_computer_control",
            "authorizes_screenshot",
            "authorizes_approval",
            "authorizes_route_unlock",
            "authorizes_verification_shortcut",
            "authorizes_model_call",
            "authorizes_tool_execution",
            "authorizes_personal_data_read",
            "authorizes_external_side_effect",
        ):
            if row.get(key) is not False:
                return False
    expected_true = {
        "ready_for_fresh_next_primitive_review",
        "oav_cycle_ledger_ready",
        "oav_cycle_ledger_token_present",
        "oav_cycle_ledger_token_boundary_ready",
        "next_review_requires_fresh_observation",
        "prior_primitive_proof_only",
        "receipt_binding_ready",
        "real_execution_approval_gated",
        "approval_required_for_real_use",
        "next_primitive_requires_new_oav_cycle_ledger_token",
    }
    for key in expected_true:
        if metadata.get(key) is not True:
            return False
    for key in (
        "action_allowed_now",
        "computer_control_enabled",
        "natural_language_routing_enabled",
        "natural_language_computer_control_routing_enabled",
        "previous_approval_reusable_for_next_primitive",
        "previous_observation_reusable_for_next_primitive",
        "previous_verification_reusable_for_next_primitive",
        "previous_approved_run_reusable_for_next_primitive",
        "oav_cycle_ledger_token_authorizes_action_now",
        "oav_cycle_ledger_token_authorizes_computer_control",
        "oav_cycle_ledger_token_authorizes_screenshot",
        "oav_cycle_ledger_token_authorizes_approval",
        "oav_cycle_ledger_token_authorizes_route_unlock",
        "oav_cycle_ledger_token_authorizes_model_call",
        "oav_cycle_ledger_token_authorizes_tool_execution",
        "oav_cycle_ledger_token_authorizes_personal_data_read",
        "oav_cycle_ledger_token_authorizes_external_side_effect",
        "oav_cycle_ledger_token_reusable_for_next_primitive",
        "prior_primitive_proof_authorizes_new_action",
        "prior_primitive_proof_authorizes_computer_control",
        "prior_primitive_proof_authorizes_screenshot",
        "prior_primitive_proof_authorizes_model_call",
        "prior_primitive_proof_authorizes_tool_execution",
        "prior_primitive_proof_authorizes_personal_data_read",
        "prior_primitive_proof_authorizes_external_side_effect",
        "prior_primitive_proof_authorizes_approval",
        "prior_primitive_proof_authorizes_route_unlock",
        "prior_primitive_proof_authorizes_verification_shortcut",
    ):
        if metadata.get(key) is not False:
            return False
    if metadata.get("cycle_state") != "OAV_CYCLE_LEDGER_READY_FOR_FRESH_REVIEW":
        return False
    if metadata.get("oav_cycle_ledger_token_boundary_row_count") != 4:
        return False
    if not _oav_cycle_ledger_token_boundary_ready(token_sha256, token_rows):
        return False
    expected_commands = {
        "observe act verify cycle ledger",
        "completion audit: improve AGI gate vision observe-act-verify",
        "evidence ledger",
        "completion claim gate: improve AGI gate vision observe-act-verify",
    }
    if not expected_commands.issubset(set(required_commands)):
        return False
    for key in (
        "calls_model",
        "calls_external_service",
        "executes_tools",
        "reads_personal_data",
        "reads_private_data",
        "executes_side_effect",
        "external_side_effect",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "writes_files",
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "requires_approval",
        "observes_screen",
        "takes_screenshot",
        "controls_computer",
        "reads_clipboard",
        "runs_shell",
        "speaks",
        "completes_tasks",
    ):
        if metadata.get(key) is not False:
            return False
    return True


def _pyautogui():
    import pyautogui

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.08
    return pyautogui


def enable_computer_control(_: dict[str, Any]) -> ToolResult:
    global _enabled
    _enabled = True
    return ToolResult(
        "enable_computer_control",
        True,
        _with_operator_limit("Computer control enabled. Move the mouse to the top-left corner to abort."),
        _safe_metadata(computer_control_enabled=True),
    )


def disable_computer_control(_: dict[str, Any]) -> ToolResult:
    global _enabled
    _enabled = False
    return ToolResult(
        "disable_computer_control",
        True,
        _with_operator_limit("Computer control disabled."),
        _safe_metadata(computer_control_enabled=False),
    )


def computer_control_status(_: dict[str, Any]) -> ToolResult:
    return ToolResult(
        "computer_control_status",
        True,
        "enabled" if _enabled else "disabled",
        _safe_metadata(enabled=_enabled),
    )


def computer_control_readiness(args: dict[str, Any]) -> ToolResult:
    objective = _clean_text(args.get("objective"), limit=MAX_OBJECTIVE_CHARS)
    if not objective:
        objective = "use computer control safely"
    low = objective.lower()
    likely_sensitive = any(
        word in low
        for word in (
            "password",
            "token",
            "secret",
            "bank",
            "payment",
            "email",
            "message",
            "delete",
            "send",
            "submit",
            "purchase",
            "share",
        )
    )
    prerequisites = [
        ("specific objective", bool(objective), "Name the exact app, visible target, and expected finished state."),
        ("computer-control status checked", True, "This check reports the current enable state without changing it."),
        ("read-only task plan prepared", False, f"Run `computer task plan: {objective}` before any screen observation."),
        ("action rehearsal prepared", False, f"Run `action rehearsal: {objective}` if the task might touch files, shell, private data, or outside-world effects."),
        ("screen observation approved", False, "Only run `observe screen` after the operator is ready for visible screen contents to be captured."),
        ("single primitive action chosen", False, "Approve only one click, type, or mouse move at a time with coordinates/text and an expectation."),
        ("stop conditions accepted", False, "Stop on wrong app/account, sensitive content, ambiguity, failed verification, or side effects."),
    ]
    ready_now = False
    lines = [
        "Computer-control readiness:",
        f"Objective: {objective}",
        f"Current control state: {'enabled' if _enabled else 'disabled'}",
        "",
        "Readiness verdict:",
        "- Not ready to operate yet. This is the correct default until a task plan, approvals, observation boundary, and one-step action are explicit.",
        "",
        "Preflight checklist:",
    ]
    for label, passed, guidance in prerequisites:
        marker = "ok" if passed else "needed"
        lines.append(f"- {label}: {marker}. {guidance}")
    lines.extend(
        [
            "",
            "Allowed now:",
            "- Read status, draft plans, rehearse risk, and explain what approval would be needed.",
            "",
            "Still blocked until explicit approval:",
            "- Screenshots, screen observation, clicking, typing, moving the mouse, clipboard reads, shell/code, file writes/deletes, submissions, sends, purchases, sharing, and account actions.",
            "",
            "Safe next command:",
            f"- `computer task plan: {objective}`",
            "",
            "Operator limit:",
            f"- {OPERATOR_LIMIT_RULE}",
        ]
    )
    if likely_sensitive:
        lines.extend(
            [
                "",
                "Extra caution:",
                "- This objective appears to involve sensitive data or an outside-world/destructive side effect. Add `action rehearsal` and require a separate approval receipt before the side effect.",
            ]
        )
    return ToolResult(
        "computer_control_readiness",
        True,
        "\n".join(lines),
        _safe_metadata(
            objective=objective,
            ready_to_operate=ready_now,
            computer_control_enabled=_enabled,
            likely_sensitive_or_side_effecting=likely_sensitive,
        ),
    )


def computer_task_plan(args: dict[str, Any]) -> ToolResult:
    objective = _clean_text(args.get("objective"), limit=MAX_OBJECTIVE_CHARS)
    if not objective:
        objective = "complete a desktop task safely"
    low = objective.lower()
    likely_actions: list[str] = []
    if any(word in low for word in ("click", "button", "select", "open", "menu", "tab")):
        likely_actions.append("click")
    if any(word in low for word in ("type", "write", "enter", "fill", "paste", "search")):
        likely_actions.append("type_text")
    if any(word in low for word in ("drag", "move", "position")):
        likely_actions.append("move_mouse")
    if not likely_actions:
        likely_actions.append("observe_screen")

    lines = [
        "Computer task plan:",
        f"Objective: {objective}",
        "",
        "Safety boundary:",
        "- This plan is read-only and does not observe the screen, click, type, move the mouse, read clipboard, or run shell commands.",
        "- Screenshots, observe-act-verify runs, clicks, typing, and clipboard reads require explicit approval.",
        "- Enable computer control only after reviewing the target app, private windows, and the exact next action.",
        "",
        "Observe-act-verify loop:",
        "1. Observe: request one approved `observe screen` or `verify screen ...` step only when the operator is ready for screen contents to be captured.",
        "2. Decide: identify one small primitive action from the observation; do not batch multiple clicks or typing operations.",
        "3. Act: request approval for one `observe act verify` action with exact coordinates/text and an expectation.",
        "4. Verify: compare the after screenshot to the expected state; stop if the result is ambiguous.",
        "5. Record: summarize what changed in memory or a reflection only after the operator confirms it is useful.",
        "",
        "Likely primitive actions:",
    ]
    for action in likely_actions:
        if action == "observe_screen":
            lines.append("- `observe screen` or `verify screen expectation ...`")
        elif action == "click":
            lines.append("- `observe act verify action click x <x> y <y> expectation <what should change>`")
        elif action == "type_text":
            lines.append("- `observe act verify action type_text text <exact text> expectation <where text should appear>`")
        elif action == "move_mouse":
            lines.append("- `observe act verify action move_mouse x <x> y <y> expectation <pointer location>`")

    lines.extend(
        [
            "",
            "Stop conditions:",
            "- Stop if the visible app, account, file, or target is not clearly the one the operator intended.",
            "- Stop if private messages, passwords, tokens, payment pages, or sensitive documents are visible.",
            "- Stop at the operator's explicit stop times, work windows, pause commands, or newer instructions.",
            "- Stop after any failed or uncertain verification; ask the operator before trying a different action.",
            "- Stop before sending, purchasing, deleting, sharing, or submitting forms unless that exact side effect has its own approval receipt.",
            "",
            "Good next command:",
            f"- `autonomy plan: {objective}` for broader risks, then `observe screen` only when the operator explicitly approves screen observation.",
        ]
    )
    return ToolResult(
        "computer_task_plan",
        True,
        "\n".join(lines),
        _safe_metadata(objective=objective, likely_actions=likely_actions, computer_control_enabled=_enabled),
    )


def computer_action_packet(args: dict[str, Any]) -> ToolResult:
    spec = _clean_text(args.get("spec"), limit=MAX_OBJECTIVE_CHARS)
    action = _clean_text(args.get("action"), limit=40).lower()
    expectation = _clean_text(args.get("expectation"), limit=MAX_EXPECTATION_CHARS)
    text = _clean_text(args.get("text"), limit=MAX_TYPED_TEXT_CHARS)
    x_raw = args.get("x")
    y_raw = args.get("y")
    if spec:
        low = spec.lower()
        if not action:
            if "type" in low:
                action = "type_text"
            elif "move" in low:
                action = "move_mouse"
            elif "click" in low or "press" in low:
                action = "click"
        if not expectation:
            marker = " expectation "
            if marker in low:
                index = low.index(marker)
                expectation = spec[index + len(marker):].strip(" :")
                spec = spec[:index].strip()
        if not text:
            text_match = _extract_after_keyword(spec, (" text ", " type ", " value "))
            if text_match:
                text = _clean_text(text_match, limit=MAX_TYPED_TEXT_CHARS)
        if x_raw is None:
            x_raw = _extract_int_after_keyword(spec, (" x ", " at x ", " coordinate x "))
        if y_raw is None:
            y_raw = _extract_int_after_keyword(spec, (" y ", " at y ", " coordinate y "))

    if action in {"type", "typing"}:
        action = "type_text"
    if action in {"move", "move mouse"}:
        action = "move_mouse"
    supported = {"click", "type_text", "move_mouse"}
    missing: list[str] = []
    if action not in supported:
        missing.append("supported action: click, type_text, or move_mouse")
    if action in {"click", "move_mouse"}:
        if x_raw is None:
            missing.append("x coordinate")
        if y_raw is None:
            missing.append("y coordinate")
    if action == "type_text" and not text:
        missing.append("exact text")
    if not expectation:
        missing.append("expected after-state")

    lines = [
        "Computer action packet:",
        f"Requested action: {action or 'unknown'}",
        "",
        "Safety boundary:",
        "- This packet is read-only and does not observe the screen, click, type, move the mouse, read clipboard, run shell/code, write files, or queue approvals.",
        "- It prepares one auditable observe-act-verify step only.",
        "",
        "Parsed step:",
    ]
    if action in {"click", "move_mouse"}:
        bounded_x = _bounded_coordinate(x_raw)
        bounded_y = _bounded_coordinate(y_raw)
        lines.append(f"- coordinates: x={bounded_x if bounded_x is not None else '<missing>'}, y={bounded_y if bounded_y is not None else '<missing>'}")
    if action == "type_text":
        lines.append(f"- text: {text if text else '<missing exact text>'}")
    lines.append(f"- expectation: {expectation if expectation else '<missing expected after-state>'}")
    lines.extend(
        [
            "",
            "Approval-ready command:",
        ]
    )
    if missing:
        lines.append("- Not ready: add " + ", ".join(missing) + ".")
    elif action == "click":
        lines.append(f"- `observe act verify action click x {_bounded_coordinate(x_raw)} y {_bounded_coordinate(y_raw)} expectation {expectation}`")
    elif action == "move_mouse":
        lines.append(f"- `observe act verify action move_mouse x {_bounded_coordinate(x_raw)} y {_bounded_coordinate(y_raw)} expectation {expectation}`")
    elif action == "type_text":
        lines.append(f"- `observe act verify action type_text text {text} expectation {expectation}`")
    lines.extend(
        [
            "",
            "Screen confidence before action:",
            f"- expected screen evidence: {expectation if expectation else '<missing expected after-state>'}",
            "- minimum confidence before action: medium confidence from a fresh approved screen observation",
            f"- confidence command: `screen observation confidence: expectation {expectation or '<expected after-state>'}; observation <approved screen observation>; source approved_screenshot; action {action or '<primitive action>'}`",
            "- do not approve the primitive action from memory, stale screenshots, or assumed success",
            "",
            "Before approval:",
            "- Run `computer readiness: <objective>` and `computer task plan: <objective>` first.",
            "- Confirm a fresh screen observation shows the intended app/account/field.",
            "- Approve only this one primitive step.",
            "- Stop at the operator's explicit stop times, work windows, pause commands, or newer instructions.",
            "- Stop on ambiguous target, private content, failed verification, or any send/delete/purchase/share/submit side effect.",
        ]
    )
    return ToolResult(
        "computer_action_packet",
        True,
        "\n".join(lines),
        _safe_metadata(
            action=action,
            x=_bounded_coordinate(x_raw),
            y=_bounded_coordinate(y_raw),
            text_chars=len(text),
            expectation=expectation,
            approval_ready=not missing,
            missing=missing,
            expected_screen_evidence=expectation,
            observation_required_before_action=True,
            minimum_confidence_before_action="medium",
            screen_confidence_command=f"screen observation confidence: expectation {expectation or '<expected after-state>'}; observation <approved screen observation>; source approved_screenshot; action {action or '<primitive action>'}",
            computer_control_enabled=_enabled,
        ),
    )


def approved_screen_observation_receipt(args: dict[str, Any]) -> ToolResult:
    raw_spec = str(args.get("spec") or "")
    expectation = _clean_text(args.get("expectation"), limit=MAX_EXPECTATION_CHARS)
    observation = _clean_text(args.get("observation") or args.get("observed") or args.get("evidence"), limit=MAX_OBSERVATION_CHARS)
    action = _clean_text(args.get("action"), limit=MAX_OBJECTIVE_CHARS)
    source = _clean_text(args.get("source") or args.get("observation_source") or "", limit=80).lower()
    observation_id = _clean_text(args.get("observation_id") or args.get("receipt_id"), limit=80)
    screenshot_path = _clean_text(args.get("screenshot_path") or args.get("path"), limit=MAX_FILENAME_CHARS)
    screenshot_sha256 = _clean_text(args.get("screenshot_sha256") or args.get("sha256") or args.get("hash"), limit=80)
    captured_at = _clean_text(args.get("captured_at") or args.get("timestamp") or args.get("time"), limit=80)
    redaction_review = _clean_text(args.get("redaction_review") or args.get("redaction") or args.get("privacy"), limit=120).lower()
    approval_id = _bounded_coordinate(args.get("approval_id"))

    if raw_spec:
        if not expectation:
            expectation = _clean_text(_extract_field(raw_spec, ("expectation", "expected", "expect")), limit=MAX_EXPECTATION_CHARS)
        if not observation:
            observation = _clean_text(_extract_field(raw_spec, ("observation", "observed", "evidence", "screen")), limit=MAX_OBSERVATION_CHARS)
        if not action:
            action = _clean_text(_extract_field(raw_spec, ("action", "step")), limit=MAX_OBJECTIVE_CHARS)
        if not source:
            source = _clean_text(_extract_field(raw_spec, ("observation source", "source")), limit=80).lower()
        if not observation_id:
            observation_id = _clean_text(_extract_field(raw_spec, ("observation id", "receipt id", "screen receipt")), limit=80)
        if not screenshot_path:
            screenshot_path = _clean_text(_extract_field(raw_spec, ("screenshot path", "path")), limit=MAX_FILENAME_CHARS)
        if not screenshot_sha256:
            screenshot_sha256 = _clean_text(_extract_field(raw_spec, ("screenshot sha256", "sha256", "hash")), limit=80)
        if not captured_at:
            captured_at = _clean_text(_extract_field(raw_spec, ("captured at", "timestamp", "time")), limit=80)
        if not redaction_review:
            redaction_review = _clean_text(_extract_field(raw_spec, ("redaction review", "redaction", "privacy")), limit=120).lower()
        if approval_id is None:
            approval_id = _extract_int_after_keyword(raw_spec, (" approval ", " approval id "))

    source_is_current = source in {"approved_screenshot", "approved screen observation", "current approved screen", "current_screenshot"}
    redaction_reviewed = any(marker in redaction_review for marker in ("reviewed", "safe", "no private", "redacted", "approved"))
    durable_artifact_present = bool(screenshot_path or screenshot_sha256)
    receipt_ready = bool(
        source_is_current
        and observation
        and (observation_id or approval_id is not None)
        and durable_artifact_present
        and captured_at
        and redaction_reviewed
    )
    source_only_ready = bool(source_is_current and observation)

    missing: list[str] = []
    if not source_is_current:
        missing.append("current approved screenshot source")
    if not observation:
        missing.append("observation summary")
    if not observation_id and approval_id is None:
        missing.append("observation receipt id or approval id")
    if not durable_artifact_present:
        missing.append("screenshot path or sha256")
    if not captured_at:
        missing.append("capture timestamp")
    if not redaction_reviewed:
        missing.append("privacy/redaction review")

    if receipt_ready:
        receipt_state = "APPROVED_SCREEN_OBSERVATION_RECEIPT_READY"
    elif source_only_ready:
        receipt_state = "APPROVED_SCREEN_OBSERVATION_SOURCE_ONLY"
    else:
        receipt_state = "NEEDS_APPROVED_SCREEN_OBSERVATION_RECEIPT"

    lines = [
        "Approved screen observation receipt:",
        "This is read-only. It validates supplied screenshot-observation receipt metadata before that observation can be treated as real OAV evidence; it does not observe the screen, take screenshots, click, type, move the mouse, read clipboard contents, approve requests, run shell/code, write files, or queue approvals.",
        "",
        "Receipt fields:",
        f"- action: {action or '<not provided>'}",
        f"- expectation: {expectation or '<not provided>'}",
        f"- source: {source or '<missing>'}",
        f"- observation id: {observation_id or '<missing>'}",
        f"- approval id: {approval_id if approval_id is not None else '<missing>'}",
        f"- screenshot path: {screenshot_path or '<missing>'}",
        f"- screenshot sha256: {screenshot_sha256 or '<missing>'}",
        f"- captured at: {captured_at or '<missing>'}",
        f"- redaction review: {redaction_review or '<missing>'}",
        "",
        "Receipt verdict:",
        f"- receipt state: {receipt_state}",
        f"- source-only observation usable for prototype confidence: {'yes' if source_only_ready else 'no'}",
        f"- durable screenshot adapter ready: {'yes' if receipt_ready else 'no'}",
        f"- missing receipt proof: {', '.join(missing) if missing else 'none'}",
        "",
        "Real-execution rule:",
        "- Source-only observation text may support a prototype confidence packet, but real screenshot adapters need a receipt id or approval id, durable screenshot artifact reference, capture time, and privacy/redaction review.",
        "- Do not promote natural-language desktop control from source-only observation text.",
        "- A receipt proves only the current observation; every next primitive needs a fresh receipt.",
        "",
        "Operator limit:",
        f"- {OPERATOR_LIMIT_RULE}",
    ]
    return ToolResult(
        "approved_screen_observation_receipt",
        True,
        "\n".join(lines),
        _safe_metadata(
            action=action,
            expectation=expectation,
            observation_source=source,
            source_is_current_approved_observation=source_is_current,
            observation_chars=len(observation),
            observation_id=observation_id,
            approval_id=approval_id,
            screenshot_path=screenshot_path,
            screenshot_sha256_present=bool(screenshot_sha256),
            captured_at=captured_at,
            redaction_reviewed=redaction_reviewed,
            durable_artifact_present=durable_artifact_present,
            source_only_observation_ready=source_only_ready,
            real_screenshot_adapter_ready=receipt_ready,
            receipt_state=receipt_state,
            missing_receipt_proof=missing,
            missing_receipt_proof_count=len(missing),
            observation_receipt_required_for_real_execution=True,
            fresh_receipt_required_for_each_primitive=True,
        ),
    )


def screen_observation_freshness_packet(args: dict[str, Any]) -> ToolResult:
    raw_spec = str(args.get("spec") or "")
    expectation = _clean_text(args.get("expectation"), limit=MAX_EXPECTATION_CHARS)
    observation = _clean_text(args.get("observation") or args.get("observed") or args.get("evidence"), limit=MAX_OBSERVATION_CHARS)
    action = _clean_text(args.get("action"), limit=MAX_OBJECTIVE_CHARS)
    source = _clean_text(args.get("source") or args.get("observation_source") or "", limit=80).lower()
    observation_id = _clean_text(args.get("observation_id") or args.get("receipt_id"), limit=80)
    screenshot_path = _clean_text(args.get("screenshot_path") or args.get("path"), limit=MAX_FILENAME_CHARS)
    screenshot_sha256 = _clean_text(args.get("screenshot_sha256") or args.get("sha256") or args.get("hash"), limit=80)
    captured_at = _clean_text(args.get("captured_at") or args.get("timestamp") or args.get("time"), limit=80)
    current_time = _clean_text(args.get("current_time") or args.get("reviewed_at") or args.get("now"), limit=80)
    redaction_review = _clean_text(args.get("redaction_review") or args.get("redaction") or args.get("privacy"), limit=120).lower()
    approval_id = _bounded_coordinate(args.get("approval_id"))
    max_age_seconds = _bounded_age_seconds(args.get("max_age_seconds") or args.get("max_age") or args.get("ttl"))

    if raw_spec:
        if not expectation:
            expectation = _clean_text(_extract_field(raw_spec, ("expectation", "expected", "expect")), limit=MAX_EXPECTATION_CHARS)
        if not observation:
            observation = _clean_text(_extract_field(raw_spec, ("observation", "observed", "evidence", "screen")), limit=MAX_OBSERVATION_CHARS)
        if not action:
            action = _clean_text(_extract_field(raw_spec, ("action", "step")), limit=MAX_OBJECTIVE_CHARS)
        if not source:
            source = _clean_text(_extract_field(raw_spec, ("observation source", "source")), limit=80).lower()
        if not observation_id:
            observation_id = _clean_text(_extract_field(raw_spec, ("observation id", "receipt id", "screen receipt")), limit=80)
        if not screenshot_path:
            screenshot_path = _clean_text(_extract_field(raw_spec, ("screenshot path", "path")), limit=MAX_FILENAME_CHARS)
        if not screenshot_sha256:
            screenshot_sha256 = _clean_text(_extract_field(raw_spec, ("screenshot sha256", "sha256", "hash")), limit=80)
        if not captured_at:
            captured_at = _clean_text(_extract_field(raw_spec, ("captured at", "timestamp", "time")), limit=80)
        if not current_time:
            current_time = _clean_text(_extract_field(raw_spec, ("current time", "reviewed at", "now")), limit=80)
        if not redaction_review:
            redaction_review = _clean_text(_extract_field(raw_spec, ("redaction review", "redaction", "privacy")), limit=120).lower()
        if approval_id is None:
            approval_id = _extract_int_after_keyword(raw_spec, (" approval ", " approval id "))
        max_age_match = re.search(r"\b(?:max age|max_age|ttl)\s*:?\s*(?P<value>\d+)\b", raw_spec, re.IGNORECASE)
        if max_age_match:
            max_age_seconds = _bounded_age_seconds(max_age_match.group("value"))

    receipt_result = approved_screen_observation_receipt(
        {
            "expectation": expectation,
            "observation": observation,
            "action": action,
            "source": source,
            "observation_id": observation_id,
            "screenshot_path": screenshot_path,
            "screenshot_sha256": screenshot_sha256,
            "captured_at": captured_at,
            "redaction_review": redaction_review,
            "approval_id": approval_id,
        }
    )
    receipt_meta = receipt_result.metadata
    captured_dt = _parse_iso_datetime(captured_at)
    current_dt = _parse_iso_datetime(current_time)
    age_seconds = None
    if captured_dt is not None and current_dt is not None:
        age_seconds = int((current_dt - captured_dt).total_seconds())

    time_parse_ready = captured_dt is not None and current_dt is not None
    not_future = age_seconds is not None and age_seconds >= 0
    within_window = age_seconds is not None and 0 <= age_seconds <= max_age_seconds
    receipt_ready = receipt_meta.get("real_screenshot_adapter_ready") is True
    freshness_ready = bool(receipt_ready and time_parse_ready and not_future and within_window)

    missing: list[str] = []
    missing.extend(str(item) for item in (receipt_meta.get("missing_receipt_proof") or []) if str(item).strip())
    if captured_dt is None:
        missing.append("parseable capture timestamp")
    if not current_time:
        missing.append("current review time")
    if age_seconds is not None and age_seconds < 0:
        missing.append("capture timestamp is in the future")
    if age_seconds is not None and age_seconds > max_age_seconds:
        missing.append("observation receipt is stale")
    missing = list(dict.fromkeys(missing))

    if freshness_ready:
        state = "SCREEN_OBSERVATION_FRESHNESS_READY"
    elif age_seconds is not None and age_seconds > max_age_seconds:
        state = "SCREEN_OBSERVATION_STALE"
    else:
        state = "SCREEN_OBSERVATION_FRESHNESS_HELD"
    lines = [
        "Screen observation freshness packet:",
        "This is read-only. It checks whether a supplied approved screenshot-observation receipt is fresh enough for one OAV primitive without observing the screen, taking screenshots, clicking, typing, moving the mouse, reading clipboard contents, approving requests, running shell/code, writing files, or queuing approvals.",
        "",
        "Freshness scope:",
        f"- action: {action or '<not provided>'}",
        f"- expectation: {expectation or '<not provided>'}",
        f"- observation id: {observation_id or '<missing>'}",
        f"- approval id: {approval_id if approval_id is not None else '<missing>'}",
        f"- captured at: {captured_at or '<missing>'}",
        f"- current review time: {current_time or '<missing>'}",
        f"- max age seconds: {max_age_seconds}",
        "",
        "Freshness verdict:",
        f"- state: {state}",
        f"- durable receipt ready: {'yes' if receipt_ready else 'no'}",
        f"- timestamp parse ready: {'yes' if time_parse_ready else 'no'}",
        f"- observation age seconds: {age_seconds if age_seconds is not None else '<unknown>'}",
        f"- within freshness window: {'yes' if within_window else 'no'}",
        f"- ready for OAV proof review: {'yes' if freshness_ready else 'no'}",
        f"- missing freshness proof: {', '.join(missing) if missing else 'none'}",
        "",
        "Freshness rules:",
        "- A fresh packet applies only to this one primitive action, expectation, observation id, approval id, and screenshot artifact.",
        "- A stale, future-dated, source-only, or unparseable observation cannot unlock real computer-control review.",
        "- Any changed screen, app, coordinate, text, expectation, observation receipt, approval id, or screenshot artifact needs a fresh packet.",
        "",
        "Operator limit:",
        f"- {OPERATOR_LIMIT_RULE}",
    ]
    metadata = _safe_metadata(
        action=action,
        expectation=expectation,
        observation_id=observation_id,
        approval_id=approval_id,
        captured_at=captured_at,
        current_time=current_time,
        max_age_seconds=max_age_seconds,
        observation_age_seconds=age_seconds,
        receipt_state=receipt_meta.get("receipt_state"),
        durable_receipt_ready=receipt_ready,
        real_screenshot_adapter_ready=receipt_meta.get("real_screenshot_adapter_ready"),
        timestamp_parse_ready=time_parse_ready,
        observation_not_future=not_future,
        within_freshness_window=within_window,
        freshness_state=state,
        freshness_ready=freshness_ready,
        fresh_enough_for_one_primitive=freshness_ready,
        stale_observation=state == "SCREEN_OBSERVATION_STALE",
        ready_for_oav_proof_review=freshness_ready,
        missing_freshness_proof=missing,
        missing_freshness_proof_count=len(missing),
        approved_screen_observation_receipt_metadata=receipt_meta,
        fresh_receipt_required_for_each_primitive=True,
    )
    freshness_ready = _screen_observation_freshness_ready_from_metadata(metadata)
    metadata.update(
        freshness_ready=freshness_ready,
        fresh_enough_for_one_primitive=freshness_ready,
        ready_for_oav_proof_review=freshness_ready,
        screen_observation_freshness_ready=freshness_ready,
    )
    return ToolResult(
        "screen_observation_freshness_packet",
        True,
        "\n".join(lines),
        metadata,
    )


def screen_observation_confidence_packet(args: dict[str, Any]) -> ToolResult:
    expectation = _clean_text(args.get("expectation"), limit=MAX_EXPECTATION_CHARS)
    observation = _clean_text(args.get("observation") or args.get("observed") or args.get("evidence"), limit=MAX_OBSERVATION_CHARS)
    action = _clean_text(args.get("action"), limit=MAX_OBJECTIVE_CHARS)
    source = _clean_text(args.get("source") or args.get("observation_source") or "", limit=80).lower()
    if not expectation:
        return ToolResult(
            "screen_observation_confidence_packet",
            False,
            "Expectation is required.",
            _safe_metadata(),
        )

    source_is_current = source in {"approved_screenshot", "approved screen observation", "current approved screen", "current_screenshot"}
    expected_terms = _significant_terms(expectation)
    observed_low = observation.lower()
    matched_terms = [term for term in expected_terms if term in observed_low]
    missing_terms = [term for term in expected_terms if term not in observed_low]
    match_ratio = (len(matched_terms) / len(expected_terms)) if expected_terms else 0.0
    receipt_result = approved_screen_observation_receipt(
        {
            "expectation": expectation,
            "observation": observation,
            "source": source,
            "action": action,
            "observation_id": args.get("observation_id") or args.get("receipt_id"),
            "approval_id": args.get("approval_id"),
            "screenshot_path": args.get("screenshot_path") or args.get("path"),
            "screenshot_sha256": args.get("screenshot_sha256") or args.get("sha256") or args.get("hash"),
            "captured_at": args.get("captured_at") or args.get("timestamp") or args.get("time"),
            "redaction_review": args.get("redaction_review") or args.get("redaction") or args.get("privacy"),
        }
    )
    receipt_meta = receipt_result.metadata
    if not observation:
        confidence = "none"
        proof_state = "NEEDS_APPROVED_OBSERVATION"
        ready_for_action_review = False
    elif source_is_current and match_ratio >= 1.0:
        confidence = "high"
        proof_state = "OBSERVATION_CONFIDENCE_READY"
        ready_for_action_review = True
    elif source_is_current and match_ratio >= 0.5:
        confidence = "medium"
        proof_state = "PARTIAL_OBSERVATION_CONFIDENCE"
        ready_for_action_review = False
    elif matched_terms:
        confidence = "low"
        proof_state = "STALE_OR_WEAK_OBSERVATION"
        ready_for_action_review = False
    else:
        confidence = "none"
        proof_state = "NOT_OBSERVED"
        ready_for_action_review = False

    lines = [
        "Screen observation confidence packet:",
        "This is read-only. It scores supplied screen evidence before any observe-act-verify primitive can be trusted; it does not observe the screen, take screenshots, click, type, move the mouse, read clipboard contents, run shell/code, write files, or queue approvals.",
        "",
        f"Action under review: {action or '<not provided>'}",
        f"Expectation: {expectation}",
        f"Observation source: {source or '<missing>'}",
        f"Observation evidence: {observation or '<missing>'}",
        "",
        "Confidence result:",
        f"- proof state: {proof_state}",
        f"- confidence: {confidence}",
        f"- ready for action review: {'yes' if ready_for_action_review else 'no'}",
        f"- observation receipt state: {receipt_meta.get('receipt_state')}",
        f"- durable screenshot adapter ready: {'yes' if receipt_meta.get('real_screenshot_adapter_ready') else 'no'}",
        f"- matched terms: {', '.join(matched_terms) if matched_terms else 'none'}",
        f"- missing terms: {', '.join(missing_terms) if missing_terms else 'none'}",
        "",
        "Action-review rule:",
        "- A computer action packet can move toward approval only after a fresh approved screenshot or screen observation reaches at least high confidence for the exact expectation.",
        "- Manual descriptions, stale screenshots, partial matches, and missing observations are not enough to approve clicking, typing, mouse movement, or screen verification.",
        "- If the proof state is not OBSERVATION_CONFIDENCE_READY, stop and request a new approved observation instead of trying another primitive action.",
        "",
        "Operator limit:",
        f"- {OPERATOR_LIMIT_RULE}",
    ]
    return ToolResult(
        "screen_observation_confidence_packet",
        True,
        "\n".join(lines),
        _safe_metadata(
            action=action,
            expectation=expectation,
            observation_source=source,
            source_is_current_approved_observation=source_is_current,
            observation_chars=len(observation),
            expected_terms=expected_terms,
            matched_terms=matched_terms,
            missing_terms=missing_terms,
            match_ratio=round(match_ratio, 3),
            confidence=confidence,
            proof_state=proof_state,
            ready_for_action_review=ready_for_action_review,
            observation_receipt_state=receipt_meta.get("receipt_state"),
            observation_receipt_required_for_real_execution=True,
            source_only_observation_ready=receipt_meta.get("source_only_observation_ready"),
            real_screenshot_adapter_ready=receipt_meta.get("real_screenshot_adapter_ready"),
            missing_receipt_proof=receipt_meta.get("missing_receipt_proof"),
            approved_screen_observation_receipt_metadata=receipt_meta,
            observation_required_before_action=True,
            minimum_confidence_before_action="high",
        ),
    )


def screen_vision_prompt_preview(args: dict[str, Any]) -> ToolResult:
    spec = _clean_text(args.get("spec"), limit=MAX_OBSERVATION_CHARS)
    expectation = _clean_text(args.get("expectation"), limit=MAX_EXPECTATION_CHARS)
    observation = _clean_text(args.get("observation") or args.get("observed") or args.get("evidence"), limit=MAX_OBSERVATION_CHARS)
    source = _clean_text(args.get("source") or args.get("observation_source") or "", limit=80).lower()
    action = _clean_text(args.get("action"), limit=80)
    observation_id = _clean_text(args.get("observation_id") or args.get("receipt_id"), limit=80)
    approval_id = _clean_text(args.get("approval_id"), limit=40)
    screenshot_path = _clean_text(args.get("screenshot_path") or args.get("path"), limit=MAX_FILENAME_CHARS)
    captured_at = _clean_text(args.get("captured_at") or args.get("timestamp") or args.get("time"), limit=80)
    redaction_review = _clean_text(args.get("redaction_review") or args.get("redaction") or args.get("privacy"), limit=120)

    if spec:
        if not expectation:
            expectation = _clean_text(_extract_field(spec, ("expectation", "expected", "expect")), limit=MAX_EXPECTATION_CHARS)
        if not observation:
            observation = _clean_text(_extract_field(spec, ("observation", "observed", "evidence", "screen")), limit=MAX_OBSERVATION_CHARS)
        if not source:
            source = _clean_text(_extract_field(spec, ("observation source", "source")), limit=80).lower()
        if not action:
            action = _clean_text(_extract_field(spec, ("action", "step")), limit=80)
        if not observation_id:
            observation_id = _clean_text(_extract_field(spec, ("observation id", "receipt id", "screen receipt")), limit=80)
        if not screenshot_path:
            screenshot_path = _clean_text(_extract_field(spec, ("screenshot path", "path")), limit=MAX_FILENAME_CHARS)
        if not captured_at:
            captured_at = _clean_text(_extract_field(spec, ("captured at", "timestamp", "time")), limit=80)
        if not redaction_review:
            redaction_review = _clean_text(_extract_field(spec, ("redaction review", "redaction", "privacy")), limit=120)
        if not approval_id:
            approval_match = re.search(r"\bapproval(?:\s+id|\s+#)?\s+(?P<value>\d+)\b", spec, re.IGNORECASE)
            if approval_match:
                approval_id = approval_match.group("value")

    if not expectation:
        return ToolResult(
            "screen_vision_prompt_preview",
            False,
            "Expectation is required.",
            _safe_metadata(),
        )

    receipt_result = approved_screen_observation_receipt(
        {
            "expectation": expectation,
            "observation": observation,
            "source": source,
            "action": action,
            "observation_id": observation_id,
            "approval_id": approval_id,
            "screenshot_path": screenshot_path,
            "captured_at": captured_at,
            "redaction_review": redaction_review,
        }
    )
    confidence_result = screen_observation_confidence_packet(
        {
            "expectation": expectation,
            "observation": observation,
            "source": source,
            "action": action,
            "observation_id": observation_id,
            "approval_id": approval_id,
            "screenshot_path": screenshot_path,
            "captured_at": captured_at,
            "redaction_review": redaction_review,
        }
    )
    receipt_meta = receipt_result.metadata
    confidence_meta = confidence_result.metadata
    prompt_sections = {
        "system": (
            "You are a visual verification reviewer for Jarvis observe-act-verify. "
            "Use only the supplied approved observation summary and metadata. "
            "Do not request computer control, approvals, tool calls, private-data reads, or external actions."
        ),
        "user": "\n".join(
            [
                f"Action under review: {action or '<not provided>'}",
                f"Expected screen state: {expectation}",
                f"Approved observation summary: {observation or '<missing>'}",
                f"Observation source: {source or '<missing>'}",
                f"Observation id: {observation_id or '<missing>'}",
                f"Approval id: {approval_id or '<missing>'}",
                f"Screenshot artifact reference: {screenshot_path or '<not provided>'}",
                f"Captured at: {captured_at or '<unknown>'}",
                f"Redaction/privacy review: {redaction_review or '<not provided>'}",
                "Return a visual verification assessment only. If evidence is incomplete, say what fresh approved observation is needed.",
            ]
        ),
        "response_contract": (
            "Return JSON with: verdict, confidence, matched_visual_evidence, missing_visual_evidence, "
            "privacy_or_safety_concerns, and next_safe_command. The verdict must not approve, execute, "
            "complete, or continue any computer-control action."
        ),
    }
    prompt_section_keys = sorted(prompt_sections)
    handoff_ready = bool(observation and source in {"approved_screenshot", "approved screen observation", "current approved screen", "current_screenshot"})
    next_safe_command = (
        f"screen observation confidence: expectation {expectation}; observation <fresh approved screen observation>; "
        "source approved_screenshot"
    )
    if handoff_ready:
        next_safe_command = f"screen verification contract: expectation {expectation}; observation <vision model assessment>"
    handoff = {
        "source": "screen_vision_prompt_preview",
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "prompt_ready": handoff_ready,
        "receipt_state": receipt_meta.get("receipt_state"),
        "confidence_precheck_state": confidence_meta.get("proof_state"),
        "confidence_precheck": confidence_meta.get("confidence"),
        "prompt_section_keys": prompt_section_keys,
        "prompt_section_count": len(prompt_section_keys),
        "future_model_scope": "visual_verification_only",
        "next_safe_command": next_safe_command,
        "boundaries": {
            "read_only": True,
            "calls_model": False,
            "reads_image_file": False,
            "observes_screen": False,
            "takes_screenshot": False,
            "controls_computer": False,
            "queues_approval": False,
            "requires_approval": False,
            "writes_files": False,
            "writes_memory": False,
            "writes_notes": False,
            "executes_tools": False,
            "authorizes_model_call": False,
            "authorizes_screenshot": False,
            "authorizes_computer_control": False,
            "authorizes_approval": False,
            "authorizes_tool_execution": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
        },
    }

    lines = [
        "Screen vision prompt preview:",
        "This is read-only. It packages supplied approved screen-observation evidence for a future vision-model reviewer without observing the screen, taking screenshots, reading image files, calling a model, clicking, typing, moving the mouse, reading clipboard contents, approving requests, running shell/code, writing files, or queuing approvals.",
        "",
        "Evidence boundary:",
        f"- action under review: {action or '<not provided>'}",
        f"- expectation: {expectation}",
        f"- observation source: {source or '<missing>'}",
        f"- observation id: {observation_id or '<missing>'}",
        f"- approval id: {approval_id or '<missing>'}",
        f"- screenshot artifact reference: {screenshot_path or '<not provided>'}",
        f"- observation supplied: {'yes' if bool(observation) else 'no'}",
        f"- receipt state: {receipt_meta.get('receipt_state')}",
        f"- confidence precheck: {confidence_meta.get('proof_state')} ({confidence_meta.get('confidence')})",
        "",
        "Prompt sections:",
        f"[system]\n{prompt_sections['system']}",
        f"[user]\n{prompt_sections['user']}",
        f"[response_contract]\n{prompt_sections['response_contract']}",
        "",
        "Vision handoff rules:",
        "- This preview does not call a model and does not read screenshots or image files.",
        "- A later model review may only assess the supplied approved visual evidence; it cannot approve, execute, or continue a computer-control primitive.",
        "- Missing, stale, source-only, or private observations must stop the OAV path until the operator approves a fresh observation.",
        f"- next safe command: `{next_safe_command}`",
        "",
        "Operator limit:",
        f"- {OPERATOR_LIMIT_RULE}",
    ]
    return ToolResult(
        "screen_vision_prompt_preview",
        True,
        "\n".join(lines),
        _safe_metadata(
            action=action,
            expectation=expectation,
            observation_source=source,
            observation_id=observation_id,
            approval_id=approval_id,
            screenshot_artifact_reference=screenshot_path,
            captured_at=captured_at,
            redaction_review=redaction_review,
            observation_chars=len(observation),
            receipt_state=receipt_meta.get("receipt_state"),
            confidence_precheck_state=confidence_meta.get("proof_state"),
            confidence_precheck=confidence_meta.get("confidence"),
            screen_vision_prompt_handoff_ready=True,
            screen_vision_prompt_handoff=handoff,
            screen_vision_prompt_ready_for_operator=handoff["ready_for_operator"],
            screen_vision_prompt_state_changed=handoff["state_changed"],
            screen_vision_prompt_changed=handoff["changed"],
            screen_vision_prompt_content_in_handoff=handoff["content_in_handoff"],
            screen_vision_prompt_boundaries=handoff["boundaries"],
            screen_vision_prompt_next_safe_command=handoff["next_safe_command"],
            screen_vision_prompt_section_keys=prompt_section_keys,
            screen_vision_prompt_section_count=len(prompt_section_keys),
            vision_prompt_preview_ready=handoff_ready,
            model_handoff_ready=handoff_ready,
            future_model_scope="visual_verification_only",
            prompt_sections=prompt_sections,
            next_safe_command=next_safe_command,
            reads_image_file=False,
            authorizes_model_call=False,
            authorizes_screenshot=False,
            authorizes_computer_control=False,
            authorizes_approval=False,
            authorizes_tool_execution=False,
            requires_fresh_approved_observation=not handoff_ready,
            approved_screen_observation_receipt_metadata=receipt_meta,
            screen_observation_confidence_metadata=confidence_meta,
        ),
    )


def screen_vision_model_review_preview(args: dict[str, Any]) -> ToolResult:
    spec = _clean_text(args.get("spec"), limit=MAX_OBSERVATION_CHARS)
    expectation = _clean_text(args.get("expectation"), limit=MAX_EXPECTATION_CHARS)
    observation = _clean_text(args.get("observation") or args.get("observed") or args.get("evidence"), limit=MAX_OBSERVATION_CHARS)
    source = _clean_text(args.get("source") or args.get("observation_source") or "", limit=80).lower()
    action = _clean_text(args.get("action"), limit=80)
    observation_id = _clean_text(args.get("observation_id") or args.get("receipt_id"), limit=80)
    approval_id = _clean_text(args.get("approval_id"), limit=40)
    screenshot_path = _clean_text(args.get("screenshot_path") or args.get("path"), limit=MAX_FILENAME_CHARS)
    captured_at = _clean_text(args.get("captured_at") or args.get("timestamp") or args.get("time"), limit=80)
    redaction_review = _clean_text(args.get("redaction_review") or args.get("redaction") or args.get("privacy"), limit=120)
    consent = str(args.get("consent") or args.get("reviewed") or args.get("confirmed") or "").strip().lower() in {
        "true",
        "yes",
        "1",
        "approved",
        "confirmed",
        "reviewed",
    }

    if spec:
        if not expectation:
            expectation = _clean_text(_extract_field(spec, ("expectation", "expected", "expect")), limit=MAX_EXPECTATION_CHARS)
        if not observation:
            observation = _clean_text(_extract_field(spec, ("observation", "observed", "evidence", "screen")), limit=MAX_OBSERVATION_CHARS)
        if not source:
            source = _clean_text(_extract_field(spec, ("observation source", "source")), limit=80).lower()
        if not action:
            action = _clean_text(_extract_field(spec, ("action", "step")), limit=80)
        if not observation_id:
            observation_id = _clean_text(_extract_field(spec, ("observation id", "receipt id", "screen receipt")), limit=80)
        if not screenshot_path:
            screenshot_path = _clean_text(_extract_field(spec, ("screenshot path", "path")), limit=MAX_FILENAME_CHARS)
        if not captured_at:
            captured_at = _clean_text(_extract_field(spec, ("captured at", "timestamp", "time")), limit=80)
        if not redaction_review:
            redaction_review = _clean_text(_extract_field(spec, ("redaction review", "redaction", "privacy")), limit=120)
        if not consent:
            consent_text = _extract_field(spec, ("consent", "reviewed", "confirmed"))
            consent = consent_text.strip().lower() in {"true", "yes", "1", "approved", "confirmed", "reviewed"}
        if not approval_id:
            approval_match = re.search(r"\bapproval(?:\s+id|\s+#)?\s+(?P<value>\d+)\b", spec, re.IGNORECASE)
            if approval_match:
                approval_id = approval_match.group("value")

    reviewer = _screen_vision_reviewer()
    reviewer_source = _screen_vision_reviewer_source()
    image_path = Path(str(args.get("screenshot_path") or args.get("path") or screenshot_path or "")).expanduser()
    exists = image_path.exists() if screenshot_path else False
    is_file = image_path.is_file() if exists else False
    size_bytes = image_path.stat().st_size if is_file else None
    path_hash = hashlib.sha256(str(image_path).encode("utf-8")).hexdigest()[:16] if screenshot_path else ""

    prompt_result = screen_vision_prompt_preview(
        {
            "expectation": expectation,
            "observation": observation,
            "source": source,
            "action": action,
            "observation_id": observation_id,
            "approval_id": approval_id,
            "screenshot_path": screenshot_path,
            "captured_at": captured_at,
            "redaction_review": redaction_review,
        }
    )
    prompt_meta = prompt_result.metadata
    prompt_sections = prompt_meta.get("prompt_sections") if prompt_result.ok else {}
    receipt_state = str(prompt_meta.get("receipt_state") or "")
    missing_checks: list[str] = []
    if not consent:
        missing_checks.append("operator consent=true")
    if receipt_state != "APPROVED_SCREEN_OBSERVATION_RECEIPT_READY":
        missing_checks.append("approved screen observation receipt ready")
    if not screenshot_path:
        missing_checks.append("screenshot path")
    if screenshot_path and not exists:
        missing_checks.append("screenshot file exists")
    if exists and not is_file:
        missing_checks.append("screenshot path is a file")
    if not reviewer:
        missing_checks.append("local vision reviewer configured")

    if missing_checks:
        return ToolResult(
            "screen_vision_model_review_preview",
            False,
            "\n".join(
                [
                    "Screen vision model review preview is held.",
                    "No screenshot was opened, no image was analyzed, no model was called, and no computer action was approved.",
                    f"- path hash: `{path_hash or '<missing>'}`",
                    f"- receipt state: {receipt_state or '<missing>'}",
                    f"- local vision reviewer source: {reviewer_source}",
                    f"- missing checks: {', '.join(missing_checks)}",
                    "- Run `screen vision prompt preview: ...` first, then approve this PERSONAL_DATA review only if the receipt and redaction review are still trusted.",
                ]
            ),
            _safe_metadata(
                requires_approval=True,
                consent=consent,
                path_hash=path_hash,
                screenshot_path=screenshot_path,
                exists=exists,
                is_file=is_file,
                size_bytes=size_bytes,
                local_vision_reviewer_configured=reviewer is not None,
                local_vision_reviewer_source=reviewer_source,
                receipt_state=receipt_state,
                missing_checks=missing_checks,
                missing_check_count=len(missing_checks),
                reads_image_file=False,
                analyzes_image=False,
                calls_model=False,
                content_in_handoff=False,
                state_changed=False,
                changed=[],
                screen_vision_prompt_preview_metadata=prompt_meta,
            ),
        )

    try:
        assessment = str(reviewer(image_path, prompt_sections)).strip()
    except Exception as exc:
        return ToolResult(
            "screen_vision_model_review_preview",
            False,
            _computer_error("run the approved local screen vision reviewer", exc),
            _safe_metadata(
                requires_approval=True,
                consent=consent,
                path_hash=path_hash,
                screenshot_path=screenshot_path,
                exists=exists,
                is_file=is_file,
                size_bytes=size_bytes,
                local_vision_reviewer_configured=True,
                local_vision_reviewer_source=reviewer_source,
                receipt_state=receipt_state,
                reads_private_data=True,
                reads_personal_data=True,
                reads_image_file=True,
                analyzes_image=True,
                calls_model=True,
                exception_type=type(exc).__name__,
                state_changed=False,
                changed=[],
            ),
        )

    assessment = _clean_text(assessment, limit=MAX_OBSERVATION_CHARS)
    assessment_sha256 = hashlib.sha256(assessment.encode("utf-8")).hexdigest()
    next_safe_command = f"screen verification contract: expectation {expectation}; observation {assessment}"
    handoff = {
        "source": "screen_vision_model_review_preview",
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "expectation": expectation,
        "assessment_preview": assessment,
        "assessment_sha256": assessment_sha256,
        "path_hash": path_hash,
        "receipt_state": receipt_state,
        "local_vision_reviewer_source": reviewer_source,
        "next_safe_command": next_safe_command,
        "boundaries": {
            "read_only": False,
            "requires_approval": True,
            "reads_private_data": True,
            "reads_personal_data": True,
            "reads_image_file": True,
            "analyzes_image": True,
            "calls_model": True,
            "observes_screen": False,
            "takes_screenshot": False,
            "controls_computer": False,
            "queues_approval": False,
            "writes_files": False,
            "writes_memory": False,
            "writes_notes": False,
            "executes_tools": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
        },
    }
    lines = [
        "Screen vision model review preview:",
        "",
        "Assessment preview:",
        assessment or "(empty assessment)",
        "",
        "Next safe step:",
        f"- `{next_safe_command}`",
        "",
        "Safety boundary:",
        "- This approved review analyzes only the supplied screenshot artifact using the configured local vision reviewer.",
        "- It does not observe the screen, take screenshots, click, type, move the mouse, route actions, approve requests, write memory, or complete the OAV primitive.",
        "",
        "Operator limit:",
        f"- {OPERATOR_LIMIT_RULE}",
    ]
    return ToolResult(
        "screen_vision_model_review_preview",
        True,
        "\n".join(lines),
        _safe_metadata(
            screen_vision_model_review_handoff_ready=True,
            screen_vision_model_review_handoff=handoff,
            screen_vision_model_review_ready_for_operator=handoff["ready_for_operator"],
            screen_vision_model_review_state_changed=handoff["state_changed"],
            screen_vision_model_review_changed=handoff["changed"],
            screen_vision_model_review_content_in_handoff=handoff["content_in_handoff"],
            screen_vision_model_review_boundaries=handoff["boundaries"],
            screen_vision_model_review_next_safe_command=handoff["next_safe_command"],
            screen_vision_model_review_assessment_chars=len(assessment),
            screen_vision_model_review_assessment_sha256=handoff["assessment_sha256"],
            ready_for_operator=True,
            requires_approval=True,
            consent=consent,
            path_hash=path_hash,
            screenshot_path=screenshot_path,
            exists=exists,
            is_file=is_file,
            size_bytes=size_bytes,
            local_vision_reviewer_configured=True,
            local_vision_reviewer_source=reviewer_source,
            receipt_state=receipt_state,
            expectation=expectation,
            assessment_chars=len(assessment),
            assessment_sha256=assessment_sha256,
            content_in_handoff=True,
            state_changed=False,
            changed=[],
            reads_private_data=True,
            reads_personal_data=True,
            reads_image_file=True,
            analyzes_image=True,
            calls_model=True,
            observes_screen=False,
            takes_screenshot=False,
            controls_computer=False,
            queues_approval=False,
            writes_files=False,
            writes_memory=False,
            writes_notes=False,
            executes_tools=False,
            next_command=next_safe_command,
        ),
    )


def screen_verification_contract(args: dict[str, Any]) -> ToolResult:
    expectation = _clean_text(args.get("expectation"), limit=MAX_EXPECTATION_CHARS)
    observation = _clean_text(args.get("observation") or args.get("observed") or args.get("evidence"), limit=MAX_OBSERVATION_CHARS)
    action = _clean_text(args.get("action"), limit=MAX_OBJECTIVE_CHARS)
    if not expectation:
        return ToolResult(
            "screen_verification_contract",
            False,
            "Expectation is required.",
            _safe_metadata(),
        )

    expected_terms = _significant_terms(expectation)
    observed_low = observation.lower()
    matched_terms = [term for term in expected_terms if term in observed_low]
    missing_terms = [term for term in expected_terms if term not in observed_low]
    if not observation:
        verdict = "NEEDS_OBSERVATION"
        confidence = "none"
    elif expected_terms and len(matched_terms) == len(expected_terms):
        verdict = "LIKELY_VERIFIED"
        confidence = "medium"
    elif matched_terms:
        verdict = "PARTIAL_MATCH"
        confidence = "low"
    else:
        verdict = "NOT_VERIFIED"
        confidence = "low"

    lines = [
        "Screen verification contract:",
        "This is read-only. It compares a stated expectation against supplied observation text without observing the screen, taking screenshots, clicking, typing, moving the mouse, reading clipboard contents, running shell/code, writing files, or queuing approvals.",
        "",
        f"Action under review: {action or '<not provided>'}",
        f"Expectation: {expectation}",
        f"Observation evidence: {observation or '<missing>'}",
        "",
        "Verification result:",
        f"- verdict: {verdict}",
        f"- confidence: {confidence}",
        f"- matched terms: {', '.join(matched_terms) if matched_terms else 'none'}",
        f"- missing terms: {', '.join(missing_terms) if missing_terms else 'none'}",
        "",
        "Use in observe-act-verify:",
        "- Before action: create `computer action packet` with exact primitive action and expected after-state.",
        "- During action: capture or describe the approved observation only after the operator approves screen observation.",
        "- After action: use this contract to decide whether the expected state is proven, partial, or failed.",
        "- Stop on `NEEDS_OBSERVATION`, `PARTIAL_MATCH`, or `NOT_VERIFIED` unless the operator explicitly approves another observation/action.",
        "",
        "Hard stops:",
            "- Do not treat this as visual proof if the observation was not captured from the current approved screen state.",
            "- Do not continue clicking or typing when verification is partial, missing, or ambiguous.",
            "- Stop at the operator's explicit stop times, work windows, pause commands, or newer instructions.",
            "- Do not include private screen contents in memory or notes unless the operator explicitly asks to save a summary.",
        ]
    return ToolResult(
        "screen_verification_contract",
        True,
        "\n".join(lines),
        _safe_metadata(
            action=action,
            expectation=expectation,
            observation_chars=len(observation),
            expected_terms=expected_terms,
            matched_terms=matched_terms,
            missing_terms=missing_terms,
            verdict=verdict,
            confidence=confidence,
        ),
    )


def observe_act_verify_proof_packet(args: dict[str, Any]) -> ToolResult:
    spec = _clean_text(args.get("spec"), limit=MAX_OBJECTIVE_CHARS)
    action = _clean_text(args.get("action"), limit=40).lower()
    expectation = _clean_text(args.get("expectation"), limit=MAX_EXPECTATION_CHARS)
    observation = _clean_text(args.get("observation") or args.get("observed") or args.get("evidence"), limit=MAX_OBSERVATION_CHARS)
    after_observation = _clean_text(
        args.get("after_observation") or args.get("after") or args.get("verification_observation"),
        limit=MAX_OBSERVATION_CHARS,
    )
    source = _clean_text(args.get("source") or args.get("observation_source"), limit=80).lower()
    text = _clean_text(args.get("text"), limit=MAX_TYPED_TEXT_CHARS)
    x_raw = args.get("x")
    y_raw = args.get("y")

    if spec:
        low = spec.lower()
        if not action:
            if "type" in low:
                action = "type_text"
            elif "move" in low:
                action = "move_mouse"
            elif "click" in low or "press" in low:
                action = "click"
        if not expectation:
            expectation_match = _extract_field(spec, ("expectation", "expected", "expect"))
            if expectation_match:
                expectation = _clean_text(expectation_match, limit=MAX_EXPECTATION_CHARS)
        if not observation:
            observation_match = _extract_field(spec, ("observation", "observed", "evidence", "screen"))
            if observation_match:
                observation = _clean_text(observation_match, limit=MAX_OBSERVATION_CHARS)
        if not after_observation:
            after_match = _extract_field(spec, ("after", "after observation", "verification", "verified"))
            if after_match:
                after_observation = _clean_text(after_match, limit=MAX_OBSERVATION_CHARS)
        if not source:
            source_match = _extract_field(spec, ("source", "observation source"))
            if source_match:
                source = _clean_text(source_match, limit=80).lower()
        if not text:
            text_match = _extract_after_keyword(spec, (" text ", " type ", " value "))
            if text_match:
                text = _clean_text(text_match, limit=MAX_TYPED_TEXT_CHARS)
        if x_raw is None:
            x_raw = _extract_int_after_keyword(spec, (" x ", " at x ", " coordinate x "))
        if y_raw is None:
            y_raw = _extract_int_after_keyword(spec, (" y ", " at y ", " coordinate y "))

    if action in {"type", "typing"}:
        action = "type_text"
    if action in {"move", "move mouse"}:
        action = "move_mouse"

    action_result = computer_action_packet(
        {"action": action, "x": x_raw, "y": y_raw, "text": text, "expectation": expectation}
    )
    confidence_result = screen_observation_confidence_packet(
        {"expectation": expectation, "observation": observation, "source": source, "action": action}
    )
    receipt_result = approved_screen_observation_receipt(
        {"spec": raw_spec if 'raw_spec' in locals() else spec, "expectation": expectation, "observation": observation, "source": source, "action": action}
    )
    verification_result = screen_verification_contract(
        {"expectation": expectation, "observation": after_observation or observation, "action": action}
    )

    action_meta = action_result.metadata
    confidence_meta = confidence_result.metadata
    receipt_meta = receipt_result.metadata
    verification_meta = verification_result.metadata
    missing = list(action_meta.get("missing") or [])
    if not observation:
        missing.append("fresh approved screen observation")
    if not source:
        missing.append("observation source")
    if not after_observation:
        missing.append("after-action observation for verification")

    pre_action_confidence_ready = confidence_meta.get("proof_state") == "OBSERVATION_CONFIDENCE_READY"
    verification_likely = verification_meta.get("verdict") == "LIKELY_VERIFIED"
    primitive_ready = bool(action_meta.get("approval_ready")) and pre_action_confidence_ready
    proof_state = "OAV_PROOF_READY_FOR_APPROVAL_REVIEW" if primitive_ready else "OAV_PROOF_INCOMPLETE"
    if primitive_ready and verification_likely:
        proof_state = "OAV_PROOF_READY_WITH_AFTER_VERIFICATION"
    elif primitive_ready and not after_observation:
        proof_state = "OAV_READY_FOR_SINGLE_STEP_APPROVAL"

    lines = [
        "Observe-act-verify proof packet:",
        "This is read-only. It combines the primitive action packet, fresh observation-confidence gate, and screen verification contract without observing the screen, taking screenshots, clicking, typing, moving the mouse, reading clipboard contents, running shell/code, writing files, controlling the computer, or queuing approvals.",
        "",
        "Primitive under review:",
        f"- action: {action or '<missing>'}",
        f"- coordinates: x={action_meta.get('x') if action in {'click', 'move_mouse'} else '<not used>'}, y={action_meta.get('y') if action in {'click', 'move_mouse'} else '<not used>'}",
        f"- text chars: {action_meta.get('text_chars', 0) if action == 'type_text' else '<not used>'}",
        f"- expectation: {expectation or '<missing>'}",
        "",
        "Pre-action observation gate:",
        f"- source: {source or '<missing>'}",
        f"- proof state: {confidence_meta.get('proof_state')}",
        f"- confidence: {confidence_meta.get('confidence')}",
        f"- ready for action review: {'yes' if pre_action_confidence_ready else 'no'}",
        f"- observation receipt state: {receipt_meta.get('receipt_state')}",
        f"- durable screenshot adapter ready: {'yes' if receipt_meta.get('real_screenshot_adapter_ready') else 'no'}",
        "",
        "After-action verification gate:",
        f"- verdict: {verification_meta.get('verdict')}",
        f"- confidence: {verification_meta.get('confidence')}",
        f"- after observation supplied: {'yes' if bool(after_observation) else 'no'}",
        "",
        "Proof verdict:",
        f"- proof state: {proof_state}",
        f"- primitive ready for approval review: {'yes' if primitive_ready else 'no'}",
        f"- missing proof: {', '.join(dict.fromkeys(missing)) if missing else 'none'}",
        "",
        "Approval boundary:",
        "- This packet never approves or runs the action. It only says whether the next single primitive has enough evidence to be reviewed.",
        "- Computer control must remain disabled until the operator explicitly approves the exact one-step action through the normal approval queue.",
        "- A ready packet allows at most one primitive action; it does not grant a session-wide screen-control permission.",
        "- If after-action verification is missing, the next required evidence after approval is a fresh approved verification observation.",
        "",
        "Next safe commands:",
    ]
    if not primitive_ready:
        lines.append(f"- `screen observation confidence: expectation {expectation or '<expected state>'}; observation <fresh approved screen observation>; source approved_screenshot; action {action or '<primitive>'}`")
    else:
        lines.append("- Review the normal approval receipt for the exact `observe act verify action ...` command; do not broaden it.")
    if not verification_likely:
        lines.append(f"- `screen verification contract: expectation {expectation or '<expected state>'}; observation <after-action approved observation>; action {action or '<primitive>'}`")
    lines.extend(
        [
            "",
            "Operator limit:",
            f"- {OPERATOR_LIMIT_RULE}",
        ]
    )
    return ToolResult(
        "observe_act_verify_proof_packet",
        True,
        "\n".join(lines),
        _safe_metadata(
            action=action,
            x=action_meta.get("x"),
            y=action_meta.get("y"),
            text_chars=action_meta.get("text_chars", 0),
            expectation=expectation,
            observation_source=source,
            observation_chars=len(observation),
            after_observation_chars=len(after_observation),
            action_approval_ready=action_meta.get("approval_ready") is True,
            pre_action_confidence=confidence_meta.get("confidence"),
            pre_action_proof_state=confidence_meta.get("proof_state"),
            pre_action_confidence_ready=pre_action_confidence_ready,
            observation_receipt_state=receipt_meta.get("receipt_state"),
            observation_receipt_required_for_real_execution=True,
            source_only_observation_ready=receipt_meta.get("source_only_observation_ready"),
            real_screenshot_adapter_ready=receipt_meta.get("real_screenshot_adapter_ready"),
            missing_receipt_proof=receipt_meta.get("missing_receipt_proof"),
            approved_screen_observation_receipt_metadata=receipt_meta,
            verification_verdict=verification_meta.get("verdict"),
            verification_confidence=verification_meta.get("confidence"),
            after_action_verification_ready=verification_likely,
            primitive_ready_for_approval_review=primitive_ready,
            proof_state=proof_state,
            missing_proof=list(dict.fromkeys(missing)),
            requires_exact_single_step_approval=True,
            computer_control_enabled=_enabled,
        ),
    )


def observe_act_verify_route_lock(args: dict[str, Any]) -> ToolResult:
    raw_spec = str(args.get("spec") or "")
    spec = _clean_text(raw_spec, limit=MAX_OBJECTIVE_CHARS)
    action = _clean_text(args.get("action"), limit=40).lower()
    expectation = _clean_text(args.get("expectation"), limit=MAX_EXPECTATION_CHARS)
    observation = _clean_text(args.get("observation") or args.get("observed") or args.get("evidence"), limit=MAX_OBSERVATION_CHARS)
    after_observation = _clean_text(
        args.get("after_observation") or args.get("after") or args.get("verification_observation"),
        limit=MAX_OBSERVATION_CHARS,
    )
    source = _clean_text(args.get("source") or args.get("observation_source"), limit=80).lower()
    text = _clean_text(args.get("text"), limit=MAX_TYPED_TEXT_CHARS)
    x_raw = args.get("x")
    y_raw = args.get("y")

    if spec:
        low = spec.lower()
        if not action:
            if "type" in low:
                action = "type_text"
            elif "move" in low:
                action = "move_mouse"
            elif "click" in low or "press" in low:
                action = "click"
        if not expectation:
            expectation = _clean_text(_extract_field(spec, ("expectation", "expected", "expect")), limit=MAX_EXPECTATION_CHARS)
        if not observation:
            observation = _clean_text(_extract_field(spec, ("observation", "observed", "evidence", "screen")), limit=MAX_OBSERVATION_CHARS)
        if not after_observation:
            after_observation = _clean_text(_extract_field(spec, ("after observation", "after", "verification", "verified")), limit=MAX_OBSERVATION_CHARS)
        if not source:
            source = _clean_text(_extract_field(spec, ("observation source", "source")), limit=80).lower()
        if not text:
            text = _clean_text(_extract_after_keyword(spec, (" text ", " type ", " value ")), limit=MAX_TYPED_TEXT_CHARS)
        if x_raw is None:
            x_raw = _extract_int_after_keyword(spec, (" x ", " at x ", " coordinate x "))
        if y_raw is None:
            y_raw = _extract_int_after_keyword(spec, (" y ", " at y ", " coordinate y "))

    evidence_text = " ".join(
        str(value or "")
        for value in [
            raw_spec,
            args.get("approval"),
            args.get("approval_evidence"),
            args.get("approval_packet_evidence"),
            args.get("approval_chain_evidence"),
            args.get("readiness"),
            args.get("packet"),
            args.get("chain"),
        ]
    ).lower()
    approval_readiness_present = bool(args.get("approval_readiness_evidence")) or "approval readiness" in evidence_text
    approval_packet_present = bool(args.get("approval_packet_evidence")) or "approval packet" in evidence_text
    approval_chain_present = bool(args.get("approval_chain_evidence")) or "approval chain proof" in evidence_text
    approved_rerun_present = bool(args.get("approved_rerun_evidence")) or "approved rerun" in evidence_text or "approved run" in evidence_text

    proof_args = {
        "spec": spec,
        "action": action,
        "expectation": expectation,
        "observation": observation,
        "after_observation": after_observation,
        "source": source,
        "text": text,
        "x": x_raw,
        "y": y_raw,
    }
    proof_result = observe_act_verify_proof_packet(proof_args)
    proof_meta = proof_result.metadata
    proof_state = str(proof_meta.get("proof_state") or "OAV_PROOF_INCOMPLETE")
    primitive_ready = bool(proof_meta.get("primitive_ready_for_approval_review"))
    after_ready = bool(proof_meta.get("after_action_verification_ready"))

    missing: list[str] = []
    if proof_state != "OAV_PROOF_READY_WITH_AFTER_VERIFICATION":
        missing.append("OAV proof packet with after-action verification")
    if not approval_readiness_present:
        missing.append("approval readiness evidence")
    if not approval_packet_present:
        missing.append("approval packet evidence")
    if not approval_chain_present:
        missing.append("approval chain proof evidence")
    if not approved_rerun_present:
        missing.append("approved rerun verification evidence")

    route_unlock_candidate = not missing
    route_lock_state = "OAV_ROUTE_LOCK_READY_FOR_EXPLICIT_REVIEW" if route_unlock_candidate else "OAV_ROUTE_LOCK_HELD"
    action_label = action or "<primitive>"
    proof_command = (
        f"observe act verify proof: action {action_label}; "
        f"x {proof_meta.get('x') if proof_meta.get('x') is not None else '<x>'}; "
        f"y {proof_meta.get('y') if proof_meta.get('y') is not None else '<y>'}; "
        f"expectation {expectation or '<expected state>'}; observation <fresh approved screen observation>; "
        "source approved_screenshot; after <after-action approved observation>"
    )
    approval_command = f"observe act verify action {action_label}"
    if action in {"click", "move_mouse"}:
        approval_command += f" x {proof_meta.get('x') if proof_meta.get('x') is not None else '<x>'} y {proof_meta.get('y') if proof_meta.get('y') is not None else '<y>'}"
    if action == "type_text":
        approval_command += " text <exact text>"
    approval_command += f" expectation {expectation or '<expected state>'}"
    route_proof_queue = [
        proof_command,
        f"approval readiness <approval id for `{approval_command}`>",
        f"approval packet <approval id for `{approval_command}`>",
        f"approval chain proof <approval id for `{approval_command}`>",
        f"verification receipt <approved run id from approval chain proof>",
        "execution health report",
    ]
    next_route_proof_command = "execution health report" if route_unlock_candidate else route_proof_queue[0]

    lines = [
        "Observe-act-verify route lock:",
        "This is read-only. It decides whether a future computer-control primitive may leave proof-only mode without observing the screen, taking screenshots, clicking, typing, moving the mouse, approving requests, queuing approvals, writing files, or controlling the computer.",
        "",
        "Primitive route:",
        f"- action: {action_label}",
        f"- expectation: {expectation or '<missing>'}",
        f"- proof packet state: {proof_state}",
        f"- primitive ready for approval review: {'yes' if primitive_ready else 'no'}",
        f"- after-action verification ready: {'yes' if after_ready else 'no'}",
        "",
        "Route posture:",
        f"- route lock state: {route_lock_state}",
        f"- route unlock candidate: {'yes, explicit review still required' if route_unlock_candidate else 'no'}",
        "- natural-language computer-control routing: disabled",
        "- real computer-control execution: approval-gated",
        f"- missing blockers: {', '.join(missing) if missing else 'none'}",
        "",
        "Route lock rules:",
        "- OAV proof readiness is not approval and does not run the action.",
        "- Unlock applies only to one exact primitive action, coordinate/text payload, expectation, observation source, and after-action verification.",
        "- Natural-language desktop-control dispatch remains disabled until an explicit route review confirms the exact proof queue.",
        "- Any changed app, screen, coordinate, text, expectation, observation, approval id, or verification receipt needs a fresh route lock packet.",
        "",
        "Route proof queue:",
        f"- next route required: `{next_route_proof_command}`",
        f"- route proof queue count: {len(route_proof_queue)}",
        "- route proof queue: " + ", ".join(f"`{command}`" for command in route_proof_queue),
        "",
        "Hard stops:",
        "- Stop if screen contents include private messages, passwords, tokens, payment pages, or sensitive documents.",
        "- Stop if natural-language routing could control the computer before the explicit route lock review.",
        "- Stop if approval readiness, approval packet, approval chain proof, audit, verification, or rollback/stop evidence can be bypassed.",
        "",
        "Operator limit:",
        f"- {OPERATOR_LIMIT_RULE}",
    ]

    return ToolResult(
        "observe_act_verify_route_lock",
        True,
        "\n".join(lines),
        _safe_metadata(
            action=action,
            x=proof_meta.get("x"),
            y=proof_meta.get("y"),
            text_chars=proof_meta.get("text_chars", 0),
            expectation=expectation,
            proof_state=proof_state,
            route_lock_state=route_lock_state,
            route_unlock_candidate=route_unlock_candidate,
            natural_language_routing_enabled=False,
            natural_language_computer_control_routing_enabled=False,
            real_execution_approval_gated=True,
            approval_required_for_real_use=True,
            primitive_ready_for_approval_review=primitive_ready,
            after_action_verification_ready=after_ready,
            approval_readiness_evidence=approval_readiness_present,
            approval_packet_evidence=approval_packet_present,
            approval_chain_evidence=approval_chain_present,
            approved_rerun_evidence=approved_rerun_present,
            missing_blockers=missing,
            missing_blocker_count=len(missing),
            next_route_proof_command=next_route_proof_command,
            next_route_required_command=next_route_proof_command,
            next_required_command=next_route_proof_command,
            route_proof_queue=route_proof_queue,
            route_proof_queue_count=len(route_proof_queue),
            oav_proof_queue=route_proof_queue,
            oav_proof_queue_count=len(route_proof_queue),
            oav_next_proof_command=next_route_proof_command,
            oav_next_required_command=next_route_proof_command,
            computer_control_enabled=_enabled,
        ),
    )


def observe_act_verify_approval_bridge(args: dict[str, Any]) -> ToolResult:
    raw_spec = str(args.get("spec") or "")
    spec = _clean_text(raw_spec, limit=MAX_OBJECTIVE_CHARS)
    action = _clean_text(args.get("action"), limit=40).lower()
    expectation = _clean_text(args.get("expectation"), limit=MAX_EXPECTATION_CHARS)
    observation = _clean_text(args.get("observation") or args.get("observed") or args.get("evidence"), limit=MAX_OBSERVATION_CHARS)
    after_observation = _clean_text(
        args.get("after_observation") or args.get("after") or args.get("verification_observation"),
        limit=MAX_OBSERVATION_CHARS,
    )
    source = _clean_text(args.get("source") or args.get("observation_source"), limit=80).lower()
    text = _clean_text(args.get("text"), limit=MAX_TYPED_TEXT_CHARS)
    x_raw = args.get("x")
    y_raw = args.get("y")

    if spec:
        if not action:
            action = _clean_text(_extract_field(spec, ("action", "step")), limit=40).lower()
        if not expectation:
            expectation = _clean_text(_extract_field(spec, ("expectation", "expected", "expect")), limit=MAX_EXPECTATION_CHARS)
        if not observation:
            observation = _clean_text(_extract_field(spec, ("observation", "observed", "evidence", "screen")), limit=MAX_OBSERVATION_CHARS)
        if not after_observation:
            after_observation = _clean_text(_extract_field(spec, ("after observation", "after", "verification", "verified")), limit=MAX_OBSERVATION_CHARS)
        if not source:
            source = _clean_text(_extract_field(spec, ("observation source", "source")), limit=80).lower()
        if not text:
            text = _clean_text(_extract_after_keyword(spec, (" text ", " type ", " value ")), limit=MAX_TYPED_TEXT_CHARS)
        if x_raw is None:
            x_raw = _extract_int_after_keyword(spec, (" x ", " at x ", " coordinate x "))
        if y_raw is None:
            y_raw = _extract_int_after_keyword(spec, (" y ", " at y ", " coordinate y "))

    if action in {"type", "typing"}:
        action = "type_text"
    if action in {"move", "move mouse"}:
        action = "move_mouse"

    evidence_text = " ".join(
        str(value or "")
        for value in [
            raw_spec,
            args.get("approval"),
            args.get("approval_evidence"),
            args.get("approval_packet_evidence"),
            args.get("approval_chain_evidence"),
            args.get("readiness"),
            args.get("packet"),
            args.get("chain"),
            args.get("verification"),
            args.get("receipt"),
        ]
    ).lower()

    approval_id = _bounded_coordinate(args.get("approval_id"))
    approved_run_id = _bounded_coordinate(args.get("approved_run_id"))
    verification_run_id = _bounded_coordinate(args.get("verification_run_id"))
    if approval_id is None:
        approval_match = re.search(r"\bapproval(?:\s+id|\s+#)?\s+(\d+)\b", evidence_text)
        approval_id = int(approval_match.group(1)) if approval_match else None
    if approved_run_id is None:
        approved_match = re.search(r"\b(?:approved\s+run|approved\s+run\s+id|approved\s+run\s+#|rerun|rerun\s+id)\s+(\d+)\b", evidence_text)
        approved_run_id = int(approved_match.group(1)) if approved_match else None
    if verification_run_id is None:
        verification_match = re.search(r"\b(?:verification\s+receipt|verification\s+run\s+id|verified\s+run)\s+(\d+)\b", evidence_text)
        verification_run_id = int(verification_match.group(1)) if verification_match else None

    route_args = {
        "spec": spec,
        "action": action,
        "expectation": expectation,
        "observation": observation,
        "after_observation": after_observation,
        "source": source,
        "text": text,
        "x": x_raw,
        "y": y_raw,
        "approval": evidence_text,
    }
    route_result = observe_act_verify_route_lock(route_args)
    route_meta = route_result.metadata
    route_ready = bool(route_meta.get("route_unlock_candidate"))
    receipt_binding_ready = (
        approval_id is not None
        and approved_run_id is not None
        and verification_run_id is not None
        and approved_run_id == verification_run_id
    )

    missing = list(route_meta.get("missing_blockers") or [])
    if approval_id is None:
        missing.append("numeric approval id")
    if approved_run_id is None:
        missing.append("numeric approved run id")
    if verification_run_id is None:
        missing.append("numeric verification receipt run id")
    if approved_run_id is not None and verification_run_id is not None and approved_run_id != verification_run_id:
        missing.append("verification receipt must match approved run id")
    missing = list(dict.fromkeys(missing))

    bridge_ready = route_ready and receipt_binding_ready
    bridge_state = "OAV_APPROVAL_BRIDGE_READY_FOR_FINAL_REVIEW" if bridge_ready else "OAV_APPROVAL_BRIDGE_HELD"
    action_label = action or "<primitive>"
    approval_command = f"observe act verify action {action_label}"
    if action in {"click", "move_mouse"}:
        approval_command += f" x {route_meta.get('x') if route_meta.get('x') is not None else '<x>'} y {route_meta.get('y') if route_meta.get('y') is not None else '<y>'}"
    if action == "type_text":
        approval_command += " text <exact text>"
    approval_command += f" expectation {expectation or '<expected state>'}"
    route_queue = route_meta.get("route_proof_queue") or ["observe act verify proof"]
    required_commands = [
        str(route_queue[0]),
        f"approval readiness {approval_id if approval_id is not None else '<approval id>'}",
        f"approval packet {approval_id if approval_id is not None else '<approval id>'}",
        f"approval chain proof {approval_id if approval_id is not None else '<approval id>'}",
        f"verification receipt {approved_run_id if approved_run_id is not None else '<approved run id>'}",
        "execution audit gate",
        "execution health report",
    ]
    next_command = "execution audit gate" if bridge_ready else required_commands[0]

    lines = [
        "Observe-act-verify approval bridge:",
        "This is read-only. It binds OAV proof, route-lock posture, exact approval id, approved rerun id, and verification receipt before any computer-control route can be treated as ready for final review.",
        "",
        "Primitive binding:",
        f"- action: {action_label}",
        f"- expectation: {expectation or '<missing>'}",
        f"- route lock state: {route_meta.get('route_lock_state')}",
        f"- route unlock candidate: {'yes' if route_ready else 'no'}",
        f"- approval command: `{approval_command}`",
        f"- approval id: {approval_id if approval_id is not None else '<missing>'}",
        f"- approved run id: {approved_run_id if approved_run_id is not None else '<missing>'}",
        f"- verification receipt run id: {verification_run_id if verification_run_id is not None else '<missing>'}",
        f"- receipt binding: {'matched' if receipt_binding_ready else 'not proven'}",
        "",
        "Bridge verdict:",
        f"- bridge state: {bridge_state}",
        f"- final route review ready: {'yes, explicit human review still required' if bridge_ready else 'no'}",
        f"- missing blockers: {', '.join(missing) if missing else 'none'}",
        "",
        "Required command chain:",
        f"- next command: `{next_command}`",
        f"- command count: {len(required_commands)}",
        "- " + "\n- ".join(f"`{command}`" for command in required_commands),
        "",
        "Safety boundary:",
        "- This bridge does not approve, rerun, observe, click, type, move the mouse, queue approvals, or unlock natural-language desktop control.",
        "- It only verifies that proof artifacts point to the same one-shot primitive route before the operator performs final review.",
        "- Any changed coordinate, text, expectation, observation, approval id, approved run id, or verification receipt needs a fresh bridge packet.",
        "",
        "Operator limit:",
        f"- {OPERATOR_LIMIT_RULE}",
    ]
    return ToolResult(
        "observe_act_verify_approval_bridge",
        True,
        "\n".join(lines),
        _safe_metadata(
            action=action,
            x=route_meta.get("x"),
            y=route_meta.get("y"),
            text_chars=route_meta.get("text_chars", 0),
            expectation=expectation,
            route_lock_state=route_meta.get("route_lock_state"),
            route_unlock_candidate=route_ready,
            approval_id=approval_id,
            approved_run_id=approved_run_id,
            verification_run_id=verification_run_id,
            receipt_binding_ready=receipt_binding_ready,
            bridge_state=bridge_state,
            final_route_review_ready=bridge_ready,
            natural_language_routing_enabled=False,
            natural_language_computer_control_routing_enabled=False,
            real_execution_approval_gated=True,
            approval_required_for_real_use=True,
            missing_blockers=missing,
            missing_blocker_count=len(missing),
            required_commands=required_commands,
            required_command_count=len(required_commands),
            next_command=next_command,
            oav_bridge_required_commands=required_commands,
            oav_bridge_next_command=next_command,
            computer_control_enabled=_enabled,
        ),
    )


def observe_act_verify_cockpit(args: dict[str, Any]) -> ToolResult:
    raw_spec = str(args.get("spec") or "")
    spec = _clean_text(raw_spec, limit=MAX_OBJECTIVE_CHARS)
    action = _clean_text(args.get("action"), limit=40).lower()
    expectation = _clean_text(args.get("expectation"), limit=MAX_EXPECTATION_CHARS)
    observation = _clean_text(args.get("observation") or args.get("observed") or args.get("evidence"), limit=MAX_OBSERVATION_CHARS)
    after_observation = _clean_text(
        args.get("after_observation") or args.get("after") or args.get("verification_observation"),
        limit=MAX_OBSERVATION_CHARS,
    )
    source = _clean_text(args.get("source") or args.get("observation_source"), limit=80).lower()
    text = _clean_text(args.get("text"), limit=MAX_TYPED_TEXT_CHARS)
    x_raw = args.get("x")
    y_raw = args.get("y")
    approval_id = args.get("approval_id")
    approved_run_id = args.get("approved_run_id")
    verification_run_id = args.get("verification_run_id")
    approval_evidence = _clean_text(args.get("approval_evidence") or args.get("readiness"), limit=MAX_OBSERVATION_CHARS)
    approval_packet_evidence = _clean_text(args.get("approval_packet_evidence") or args.get("packet"), limit=MAX_OBSERVATION_CHARS)
    approval_chain_evidence = _clean_text(args.get("approval_chain_evidence") or args.get("chain"), limit=MAX_OBSERVATION_CHARS)

    common_args = {
        "spec": raw_spec,
        "action": action,
        "expectation": expectation,
        "observation": observation,
        "after_observation": after_observation,
        "source": source,
        "text": text,
        "x": x_raw,
        "y": y_raw,
    }
    proof_result = observe_act_verify_proof_packet(common_args)
    route_result = observe_act_verify_route_lock(
        {
            **common_args,
            "approval": " ".join(
                value for value in [approval_evidence, approval_packet_evidence, approval_chain_evidence] if value
            ),
        }
    )
    bridge_result = observe_act_verify_approval_bridge(
        {
            **common_args,
            "approval_id": approval_id,
            "approved_run_id": approved_run_id,
            "verification_run_id": verification_run_id,
            "approval_evidence": approval_evidence,
            "approval_packet_evidence": approval_packet_evidence,
            "approval_chain_evidence": approval_chain_evidence,
        }
    )
    proof_meta = proof_result.metadata
    route_meta = route_result.metadata
    bridge_meta = bridge_result.metadata
    cockpit_ready = bool(bridge_meta.get("final_route_review_ready"))
    cockpit_state = "OAV_COCKPIT_READY_FOR_FINAL_REVIEW" if cockpit_ready else "OAV_COCKPIT_HELD"
    missing = list(
        dict.fromkeys(
            list(proof_meta.get("missing_proof") or [])
            + list(route_meta.get("missing_blockers") or [])
            + list(bridge_meta.get("missing_blockers") or [])
        )
    )
    required_commands = list(bridge_meta.get("required_commands") or [])
    next_command = str(bridge_meta.get("next_command") or route_meta.get("next_route_proof_command") or "")

    lines = [
        "Observe-act-verify cockpit:",
        "This is read-only. It consolidates OAV proof, route lock, approval bridge, receipt binding, final review readiness, blockers, and required commands without observing the screen, taking screenshots, clicking, typing, moving the mouse, approving requests, rerunning actions, writing files, running shell/code, or queuing approvals.",
        "",
        "Cockpit state:",
        f"- cockpit state: {cockpit_state}",
        f"- proof state: {proof_meta.get('proof_state')}",
        f"- route lock state: {route_meta.get('route_lock_state')}",
        f"- bridge state: {bridge_meta.get('bridge_state')}",
        f"- primitive ready for approval review: {'yes' if proof_meta.get('primitive_ready_for_approval_review') else 'no'}",
        f"- route unlock candidate: {'yes' if route_meta.get('route_unlock_candidate') else 'no'}",
        f"- receipt binding ready: {'yes' if bridge_meta.get('receipt_binding_ready') else 'no'}",
        f"- final route review ready: {'yes, explicit human review still required' if cockpit_ready else 'no'}",
        f"- natural-language computer-control routing enabled: {'yes' if bridge_meta.get('natural_language_computer_control_routing_enabled') else 'no'}",
        f"- real execution approval-gated: {'yes' if bridge_meta.get('real_execution_approval_gated') else 'no'}",
        f"- missing blockers: {', '.join(missing) if missing else 'none'}",
        "",
        "Required command chain:",
        f"- next command: `{next_command}`" if next_command else "- next command: <none>",
        f"- command count: {len(required_commands)}",
    ]
    lines.extend(f"- `{command}`" for command in required_commands)
    lines.extend(
        [
            "",
            "Proof surfaces:",
            "- proof packet: action, coordinates/text, expectation, fresh approved observation confidence, and after-action verification",
            "- route lock: natural-language desktop routing stays disabled until proof, approvals, and approved-rerun verification are present",
            "- approval bridge: approval id, approved run id, and verification receipt must bind to the same one-shot primitive",
            "",
            "Safety boundary:",
            "- A ready cockpit is still not approval. It only means the final human review packet is internally consistent.",
            "- Any changed app, screen, coordinate, text, expectation, observation, approval id, approved run id, or verification receipt needs a fresh cockpit packet.",
            "",
            "Operator limit:",
            f"- {OPERATOR_LIMIT_RULE}",
        ]
    )
    return ToolResult(
        "observe_act_verify_cockpit",
        proof_result.ok and route_result.ok and bridge_result.ok,
        "\n".join(lines),
        _safe_metadata(
            action=bridge_meta.get("action") or route_meta.get("action") or proof_meta.get("action"),
            x=bridge_meta.get("x") if bridge_meta.get("x") is not None else route_meta.get("x"),
            y=bridge_meta.get("y") if bridge_meta.get("y") is not None else route_meta.get("y"),
            text_chars=bridge_meta.get("text_chars") or route_meta.get("text_chars") or proof_meta.get("text_chars", 0),
            expectation=bridge_meta.get("expectation") or route_meta.get("expectation") or proof_meta.get("expectation"),
            cockpit_state=cockpit_state,
            proof_state=proof_meta.get("proof_state"),
            route_lock_state=route_meta.get("route_lock_state"),
            bridge_state=bridge_meta.get("bridge_state"),
            primitive_ready_for_approval_review=proof_meta.get("primitive_ready_for_approval_review"),
            route_unlock_candidate=route_meta.get("route_unlock_candidate"),
            receipt_binding_ready=bridge_meta.get("receipt_binding_ready"),
            final_route_review_ready=bridge_meta.get("final_route_review_ready"),
            natural_language_routing_enabled=bridge_meta.get("natural_language_routing_enabled"),
            natural_language_computer_control_routing_enabled=bridge_meta.get("natural_language_computer_control_routing_enabled"),
            real_execution_approval_gated=bridge_meta.get("real_execution_approval_gated"),
            approval_required_for_real_use=True,
            missing_blockers=missing,
            missing_blocker_count=len(missing),
            next_command=next_command,
            required_commands=required_commands,
            required_command_count=len(required_commands),
            proof_metadata=proof_meta,
            route_lock_metadata=route_meta,
            approval_bridge_metadata=bridge_meta,
            proof_output=proof_result.output,
            route_lock_output=route_result.output,
            approval_bridge_output=bridge_result.output,
            computer_control_enabled=_enabled,
        ),
    )


def observe_act_verify_final_review(args: dict[str, Any]) -> ToolResult:
    raw_spec = str(args.get("spec") or "")
    spec = _clean_text(raw_spec, limit=MAX_OBJECTIVE_CHARS)
    action = _clean_text(args.get("action"), limit=40).lower()
    expectation = _clean_text(args.get("expectation"), limit=MAX_EXPECTATION_CHARS)
    observation = _clean_text(args.get("observation") or args.get("observed") or args.get("evidence"), limit=MAX_OBSERVATION_CHARS)
    after_observation = _clean_text(
        args.get("after_observation") or args.get("after") or args.get("verification_observation"),
        limit=MAX_OBSERVATION_CHARS,
    )
    source = _clean_text(args.get("source") or args.get("observation_source"), limit=80).lower()
    text = _clean_text(args.get("text"), limit=MAX_TYPED_TEXT_CHARS)
    x_raw = args.get("x")
    y_raw = args.get("y")
    approval_id = args.get("approval_id")
    approved_run_id = args.get("approved_run_id")
    verification_run_id = args.get("verification_run_id")
    approval_evidence = _clean_text(args.get("approval_evidence") or args.get("readiness"), limit=MAX_OBSERVATION_CHARS)
    approval_packet_evidence = _clean_text(args.get("approval_packet_evidence") or args.get("packet"), limit=MAX_OBSERVATION_CHARS)
    approval_chain_evidence = _clean_text(args.get("approval_chain_evidence") or args.get("chain"), limit=MAX_OBSERVATION_CHARS)
    audit_evidence = _clean_text(args.get("audit_evidence") or args.get("audit"), limit=MAX_OBSERVATION_CHARS)
    health_evidence = _clean_text(args.get("health_evidence") or args.get("health"), limit=MAX_OBSERVATION_CHARS)
    operator_review = _clean_text(args.get("operator_review") or args.get("review"), limit=MAX_OBSERVATION_CHARS)

    if spec:
        if not action:
            action = _clean_text(_extract_field(spec, ("action", "step")), limit=40).lower()
        if not expectation:
            expectation = _clean_text(_extract_field(spec, ("expectation", "expected", "expect")), limit=MAX_EXPECTATION_CHARS)
        if not observation:
            observation = _clean_text(_extract_field(spec, ("observation", "observed", "evidence", "screen")), limit=MAX_OBSERVATION_CHARS)
        if not after_observation:
            after_observation = _clean_text(_extract_field(spec, ("after observation", "after", "verification", "verified")), limit=MAX_OBSERVATION_CHARS)
        if not source:
            source = _clean_text(_extract_field(spec, ("observation source", "source")), limit=80).lower()
        if not text:
            text = _clean_text(_extract_after_keyword(spec, (" text ", " type ", " value ")), limit=MAX_TYPED_TEXT_CHARS)
        if x_raw is None:
            x_raw = _extract_int_after_keyword(spec, (" x ", " at x ", " coordinate x "))
        if y_raw is None:
            y_raw = _extract_int_after_keyword(spec, (" y ", " at y ", " coordinate y "))
        if approval_id is None:
            approval_id = _extract_int_after_keyword(spec, (" approval ", " approval id "))
        if approved_run_id is None:
            approved_run_id = _extract_int_after_keyword(spec, (" approved run ", " approved run id "))
        if verification_run_id is None:
            verification_run_id = _extract_int_after_keyword(spec, (" verification receipt ", " verification run id "))
        if not approval_evidence:
            approval_evidence = _clean_text(_extract_field(spec, ("approval evidence", "approval readiness", "readiness")), limit=MAX_OBSERVATION_CHARS)
        if not approval_packet_evidence:
            approval_packet_evidence = _clean_text(_extract_field(spec, ("approval packet evidence", "approval packet", "packet")), limit=MAX_OBSERVATION_CHARS)
        if not approval_chain_evidence:
            approval_chain_evidence = _clean_text(_extract_field(spec, ("approval chain evidence", "approval chain proof", "chain")), limit=MAX_OBSERVATION_CHARS)
        if not audit_evidence:
            audit_evidence = _clean_text(_extract_field(spec, ("audit evidence", "execution audit", "audit")), limit=MAX_OBSERVATION_CHARS)
        if not health_evidence:
            health_evidence = _clean_text(_extract_field(spec, ("health evidence", "execution health", "health")), limit=MAX_OBSERVATION_CHARS)
        if not operator_review:
            operator_review = _clean_text(_extract_field(spec, ("operator review", "human review", "review")), limit=MAX_OBSERVATION_CHARS)

    cockpit_result = observe_act_verify_cockpit(
        {
            "spec": raw_spec,
            "action": action,
            "expectation": expectation,
            "observation": observation,
            "after_observation": after_observation,
            "source": source,
            "text": text,
            "x": x_raw,
            "y": y_raw,
            "approval_id": approval_id,
            "approved_run_id": approved_run_id,
            "verification_run_id": verification_run_id,
            "approval_evidence": approval_evidence,
            "approval_packet_evidence": approval_packet_evidence,
            "approval_chain_evidence": approval_chain_evidence,
        }
    )
    cockpit_meta = cockpit_result.metadata
    final_evidence_text = " ".join([raw_spec, audit_evidence, health_evidence, operator_review]).lower()
    audit_ready = bool(audit_evidence) and "audit" in audit_evidence.lower() or any(
        marker in final_evidence_text for marker in ("execution audit passed", "audit passed", "execution audit reviewed")
    )
    health_ready = bool(health_evidence) and ("health" in health_evidence.lower() or "passed" in health_evidence.lower()) or any(
        marker in final_evidence_text for marker in ("execution health passed", "health passed", "execution health reviewed")
    )
    operator_review_ready = bool(operator_review) and any(
        marker in operator_review.lower() for marker in ("reviewed", "final review", "human review", "operator review")
    ) or any(
        marker in final_evidence_text for marker in ("operator reviewed", "human reviewed", "final packet reviewed", "final review passed")
    )

    missing = list(cockpit_meta.get("missing_blockers") or [])
    if cockpit_meta.get("cockpit_state") != "OAV_COCKPIT_READY_FOR_FINAL_REVIEW":
        missing.append("ready OAV cockpit")
    if not audit_ready:
        missing.append("execution audit evidence")
    if not health_ready:
        missing.append("execution health evidence")
    if not operator_review_ready:
        missing.append("explicit operator final-review evidence")
    if cockpit_meta.get("natural_language_computer_control_routing_enabled") is not False:
        missing.append("natural-language computer-control routing disabled proof")
    if cockpit_meta.get("computer_control_enabled") is not False:
        missing.append("computer control disabled before final review")
    if cockpit_meta.get("real_execution_approval_gated") is not True:
        missing.append("real execution approval-gated proof")
    missing = list(dict.fromkeys(missing))

    required_commands = list(cockpit_meta.get("required_commands") or [])
    final_review_commands = list(dict.fromkeys(required_commands + [
        "execution audit gate",
        "execution health report",
        "human final review",
    ]))
    candidate_final_ready = not missing
    candidate_final_review_state = "OAV_FINAL_REVIEW_READY_FOR_HUMAN_DECISION" if candidate_final_ready else "OAV_FINAL_REVIEW_HELD"
    candidate_metadata = _safe_metadata(
        action=cockpit_meta.get("action"),
        x=cockpit_meta.get("x"),
        y=cockpit_meta.get("y"),
        text_chars=cockpit_meta.get("text_chars", 0),
        expectation=cockpit_meta.get("expectation"),
        final_review_state=candidate_final_review_state,
        ready_for_human_decision=candidate_final_ready,
        oav_final_review_ready=candidate_final_ready,
        cockpit_state=cockpit_meta.get("cockpit_state"),
        final_route_review_ready=cockpit_meta.get("final_route_review_ready"),
        execution_audit_evidence_ready=audit_ready,
        execution_health_evidence_ready=health_ready,
        operator_final_review_evidence_ready=operator_review_ready,
        action_allowed_now=False,
        natural_language_routing_enabled=cockpit_meta.get("natural_language_routing_enabled"),
        natural_language_computer_control_routing_enabled=cockpit_meta.get("natural_language_computer_control_routing_enabled"),
        real_execution_approval_gated=cockpit_meta.get("real_execution_approval_gated"),
        approval_required_for_real_use=True,
        missing_blockers=missing,
        missing_blocker_count=len(missing),
        required_commands=final_review_commands,
        required_command_count=len(final_review_commands),
        cockpit_metadata=cockpit_meta,
        cockpit_output=cockpit_result.output,
        computer_control_enabled=_enabled,
    )
    final_ready = _oav_final_review_ready_from_metadata(candidate_metadata)
    if candidate_final_ready and not final_ready:
        missing = list(dict.fromkeys(missing + ["final review production readiness"]))
    final_review_state = "OAV_FINAL_REVIEW_READY_FOR_HUMAN_DECISION" if final_ready else "OAV_FINAL_REVIEW_HELD"
    next_command = "human final review" if final_ready else (
        "execution audit gate"
        if cockpit_meta.get("final_route_review_ready") and not audit_ready
        else "execution health report"
        if cockpit_meta.get("final_route_review_ready") and audit_ready and not health_ready
        else "human final review"
        if cockpit_meta.get("final_route_review_ready") and audit_ready and health_ready and not operator_review_ready
        else str(cockpit_meta.get("next_command") or "observe act verify cockpit")
    )

    lines = [
        "Observe-act-verify final review:",
        "This is read-only. It turns the OAV cockpit into a final human-decision packet by checking audit evidence, execution-health evidence, operator final-review evidence, and the approval boundary without observing the screen, taking screenshots, clicking, typing, moving the mouse, approving requests, rerunning actions, writing files, running shell/code, or queuing approvals.",
        "",
        "Final review state:",
        f"- final review state: {final_review_state}",
        f"- cockpit state: {cockpit_meta.get('cockpit_state')}",
        f"- final route review ready: {'yes, explicit human review still required' if cockpit_meta.get('final_route_review_ready') else 'no'}",
        f"- execution audit evidence: {'present' if audit_ready else 'missing'}",
        f"- execution health evidence: {'present' if health_ready else 'missing'}",
        f"- operator final-review evidence: {'present' if operator_review_ready else 'missing'}",
        f"- natural-language computer-control routing enabled: {'yes' if cockpit_meta.get('natural_language_computer_control_routing_enabled') else 'no'}",
        f"- real execution approval-gated: {'yes' if cockpit_meta.get('real_execution_approval_gated') else 'no'}",
        f"- ready for operator decision: {'yes, still not approval' if final_ready else 'no'}",
        f"- missing blockers: {', '.join(missing) if missing else 'none'}",
        "",
        "Final review command chain:",
        f"- next command: `{next_command}`",
        f"- command count: {len(final_review_commands)}",
    ]
    lines.extend(f"- `{command}`" for command in final_review_commands)
    lines.extend(
        [
            "",
            "Decision boundary:",
            "- A ready final review packet is still not permission to act. It only says the evidence is coherent enough for the operator to approve or reject one exact primitive.",
            "- Any changed app, screen, coordinate, text, expectation, observation, approval id, run id, receipt, audit result, or health result needs a fresh final review packet.",
            "- Real computer control remains approval-gated and single-primitive only.",
            "",
            "Operator limit:",
            f"- {OPERATOR_LIMIT_RULE}",
        ]
    )
    return ToolResult(
        "observe_act_verify_final_review",
        cockpit_result.ok,
        "\n".join(lines),
        _safe_metadata(
            action=cockpit_meta.get("action"),
            x=cockpit_meta.get("x"),
            y=cockpit_meta.get("y"),
            text_chars=cockpit_meta.get("text_chars", 0),
            expectation=cockpit_meta.get("expectation"),
            final_review_state=final_review_state,
            ready_for_human_decision=final_ready,
            oav_final_review_ready=final_ready,
            cockpit_state=cockpit_meta.get("cockpit_state"),
            final_route_review_ready=cockpit_meta.get("final_route_review_ready"),
            execution_audit_evidence_ready=audit_ready,
            execution_health_evidence_ready=health_ready,
            operator_final_review_evidence_ready=operator_review_ready,
            action_allowed_now=False,
            natural_language_routing_enabled=cockpit_meta.get("natural_language_routing_enabled"),
            natural_language_computer_control_routing_enabled=cockpit_meta.get("natural_language_computer_control_routing_enabled"),
            real_execution_approval_gated=cockpit_meta.get("real_execution_approval_gated"),
            approval_required_for_real_use=True,
            missing_blockers=missing,
            missing_blocker_count=len(missing),
            next_command=next_command,
            required_commands=final_review_commands,
            required_command_count=len(final_review_commands),
            cockpit_metadata=cockpit_meta,
            cockpit_output=cockpit_result.output,
            computer_control_enabled=_enabled,
        ),
    )


def observe_act_verify_action_audit(args: dict[str, Any]) -> ToolResult:
    final_result = observe_act_verify_final_review(args)
    final_meta = final_result.metadata
    action = str(final_meta.get("action") or "").strip()
    expectation = str(final_meta.get("expectation") or "").strip()
    x_value = final_meta.get("x")
    y_value = final_meta.get("y")

    primitive_command = f"observe act verify action {action or '<primitive>'}"
    if action in {"click", "move_mouse"}:
        primitive_command += f" x {x_value if x_value is not None else '<x>'} y {y_value if y_value is not None else '<y>'}"
    if action == "type_text":
        primitive_command += " text <exact text>"
    primitive_command += f" expectation {expectation or '<expected state>'}"

    final_ready = final_meta.get("ready_for_human_decision") is True
    computer_control_disabled = final_meta.get("computer_control_enabled") is False
    natural_routing_disabled = final_meta.get("natural_language_computer_control_routing_enabled") is False
    approval_gated = final_meta.get("real_execution_approval_gated") is True and final_meta.get("approval_required_for_real_use") is True
    audit_ready = bool(final_ready and computer_control_disabled and natural_routing_disabled and approval_gated)

    missing = list(final_meta.get("missing_blockers") or [])
    if not final_ready:
        missing.append("ready OAV final review")
    if not computer_control_disabled:
        missing.append("computer control disabled before approval decision")
    if not natural_routing_disabled:
        missing.append("natural-language computer-control routing disabled proof")
    if not approval_gated:
        missing.append("real execution approval-gated proof")
    if not primitive_command or "<" in primitive_command:
        missing.append("exact primitive command without placeholders")
    missing = list(dict.fromkeys(missing))

    required_commands = list(final_meta.get("required_commands") or [])
    action_audit_commands = list(
        dict.fromkeys(
            required_commands
            + [
                "observe act verify action audit",
                primitive_command,
                "verification receipt <approved run id>",
                "execution health report",
                "after-action learning packet <approved run id>",
            ]
        )
    )
    candidate_action_audit_ready = audit_ready and not missing
    candidate_action_audit_state = "OAV_ACTION_AUDIT_READY_FOR_APPROVAL_DECISION" if candidate_action_audit_ready else "OAV_ACTION_AUDIT_HELD"
    candidate_metadata = _safe_metadata(
        action=action,
        x=x_value,
        y=y_value,
        text_chars=final_meta.get("text_chars", 0),
        expectation=expectation,
        action_audit_state=candidate_action_audit_state,
        ready_for_approval_decision=candidate_action_audit_ready,
        oav_action_audit_ready=candidate_action_audit_ready,
        ready_for_human_decision=final_ready,
        action_allowed_now=False,
        approval_decision_required=True,
        exact_primitive_command=primitive_command,
        final_review_state=final_meta.get("final_review_state"),
        computer_control_enabled=final_meta.get("computer_control_enabled") is True,
        natural_language_routing_enabled=final_meta.get("natural_language_routing_enabled"),
        natural_language_computer_control_routing_enabled=final_meta.get("natural_language_computer_control_routing_enabled"),
        real_execution_approval_gated=final_meta.get("real_execution_approval_gated"),
        approval_required_for_real_use=True,
        missing_blockers=missing,
        missing_blocker_count=len(missing),
        required_commands=action_audit_commands,
        required_command_count=len(action_audit_commands),
        final_review_metadata=final_meta,
        final_review_output=final_result.output,
    )
    audit_ready = _oav_action_audit_ready_from_metadata(candidate_metadata)
    if candidate_action_audit_ready and not audit_ready:
        missing = list(dict.fromkeys(missing + ["action audit production readiness"]))
    action_audit_state = "OAV_ACTION_AUDIT_READY_FOR_APPROVAL_DECISION" if audit_ready else "OAV_ACTION_AUDIT_HELD"
    next_command = primitive_command if audit_ready else str(final_meta.get("next_command") or "observe act verify final review")

    lines = [
        "Observe-act-verify action audit:",
        "This is read-only. It is the last proof packet before the operator may decide on one exact approved computer-control primitive; it does not observe the screen, take screenshots, click, type, move the mouse, approve requests, rerun actions, write files, run shell/code, or queue approvals.",
        "",
        "Action audit state:",
        f"- action audit state: {action_audit_state}",
        f"- final review state: {final_meta.get('final_review_state')}",
        f"- ready for operator decision: {'yes, still not approval' if final_ready else 'no'}",
        f"- exact primitive command: `{primitive_command}`",
        f"- computer control currently enabled: {'yes' if final_meta.get('computer_control_enabled') else 'no'}",
        f"- natural-language computer-control routing enabled: {'yes' if final_meta.get('natural_language_computer_control_routing_enabled') else 'no'}",
        f"- real execution approval-gated: {'yes' if final_meta.get('real_execution_approval_gated') else 'no'}",
        "- action allowed now: no",
        "- approval decision required: yes",
        f"- missing blockers: {', '.join(missing) if missing else 'none'}",
        "",
        "One-primitive boundary:",
        "- Approval can cover only the exact primitive command shown above.",
        "- The next required evidence after any approved run is a matching verification receipt, execution health report, and after-action learning packet.",
        "- Any changed screen, app, coordinate, text, expectation, approval id, run id, receipt, audit result, health result, or operator review needs a fresh action audit.",
        "",
        "Required command chain:",
        f"- next command: `{next_command}`",
        f"- command count: {len(action_audit_commands)}",
    ]
    lines.extend(f"- `{command}`" for command in action_audit_commands)
    lines.extend(
        [
            "",
            "Operator limit:",
            f"- {OPERATOR_LIMIT_RULE}",
        ]
    )
    return ToolResult(
        "observe_act_verify_action_audit",
        final_result.ok,
        "\n".join(lines),
        _safe_metadata(
            action=action,
            x=x_value,
            y=y_value,
            text_chars=final_meta.get("text_chars", 0),
            expectation=expectation,
            action_audit_state=action_audit_state,
            ready_for_approval_decision=audit_ready,
            oav_action_audit_ready=audit_ready,
            ready_for_human_decision=final_ready,
            action_allowed_now=False,
            approval_decision_required=True,
            exact_primitive_command=primitive_command,
            final_review_state=final_meta.get("final_review_state"),
            computer_control_enabled=final_meta.get("computer_control_enabled") is True,
            natural_language_routing_enabled=final_meta.get("natural_language_routing_enabled"),
            natural_language_computer_control_routing_enabled=final_meta.get("natural_language_computer_control_routing_enabled"),
            real_execution_approval_gated=final_meta.get("real_execution_approval_gated"),
            approval_required_for_real_use=True,
            missing_blockers=missing,
            missing_blocker_count=len(missing),
            next_command=next_command,
            required_commands=action_audit_commands,
            required_command_count=len(action_audit_commands),
            final_review_metadata=final_meta,
            final_review_output=final_result.output,
        ),
    )


def observe_act_verify_execution_handoff(args: dict[str, Any]) -> ToolResult:
    audit_result = observe_act_verify_action_audit(args)
    audit_meta = audit_result.metadata
    primitive_command = str(audit_meta.get("exact_primitive_command") or "").strip()
    ready_for_decision = audit_meta.get("ready_for_approval_decision") is True
    approval_decision_required = audit_meta.get("approval_decision_required") is True
    action_allowed_now = False
    handoff_state = "OAV_EXECUTION_HANDOFF_READY_FOR_APPROVAL_PACKET" if ready_for_decision else "OAV_EXECUTION_HANDOFF_HELD"

    missing = list(audit_meta.get("missing_blockers") or [])
    if not ready_for_decision:
        missing.append("ready OAV action audit")
    if not primitive_command or "<" in primitive_command:
        missing.append("exact primitive command without placeholders")
    if not approval_decision_required:
        missing.append("approval decision boundary")
    missing = list(dict.fromkeys(missing))
    if missing:
        handoff_state = "OAV_EXECUTION_HANDOFF_HELD"

    approval_packet_command = f"approval packet <approval id for `{primitive_command or 'observe act verify action <primitive>'}`>"
    approval_readiness_command = f"approval readiness <approval id for `{primitive_command or 'observe act verify action <primitive>'}`>"
    approval_chain_command = f"approval chain proof <approval id for `{primitive_command or 'observe act verify action <primitive>'}`>"
    post_run_commands = [
        "verification receipt <approved run id>",
        "execution health report",
        "execution audit gate",
        "after-action learning packet <approved run id>",
    ]
    handoff_commands = list(
        dict.fromkeys(
            list(audit_meta.get("required_commands") or [])
            + [
                "observe act verify execution handoff",
                approval_readiness_command,
                approval_packet_command,
                primitive_command or "observe act verify action <primitive>",
                approval_chain_command,
            ]
            + post_run_commands
        )
    )
    candidate_handoff_ready = handoff_state.endswith("READY_FOR_APPROVAL_PACKET") and not missing
    candidate_metadata = _safe_metadata(
        action=audit_meta.get("action"),
        x=audit_meta.get("x"),
        y=audit_meta.get("y"),
        text_chars=audit_meta.get("text_chars", 0),
        expectation=audit_meta.get("expectation"),
        handoff_state=handoff_state,
        ready_for_approval_decision=candidate_handoff_ready,
        oav_execution_handoff_ready=candidate_handoff_ready,
        action_allowed_now=action_allowed_now,
        approval_decision_required=approval_decision_required,
        exact_primitive_command=primitive_command,
        approval_readiness_command=approval_readiness_command,
        approval_packet_command=approval_packet_command,
        approval_chain_command=approval_chain_command,
        post_run_commands=post_run_commands,
        post_run_command_count=len(post_run_commands),
        required_commands=handoff_commands,
        required_command_count=len(handoff_commands),
        action_audit_state=audit_meta.get("action_audit_state"),
        final_review_state=audit_meta.get("final_review_state"),
        computer_control_enabled=audit_meta.get("computer_control_enabled") is True,
        natural_language_routing_enabled=audit_meta.get("natural_language_routing_enabled"),
        natural_language_computer_control_routing_enabled=audit_meta.get("natural_language_computer_control_routing_enabled"),
        real_execution_approval_gated=audit_meta.get("real_execution_approval_gated"),
        approval_required_for_real_use=True,
        missing_blockers=missing,
        missing_blocker_count=len(missing),
        action_audit_metadata=audit_meta,
        action_audit_output=audit_result.output,
    )
    handoff_ready = _oav_execution_handoff_ready_from_metadata(candidate_metadata)
    if candidate_handoff_ready and not handoff_ready:
        missing = list(dict.fromkeys(missing + ["execution handoff production readiness"]))
        handoff_state = "OAV_EXECUTION_HANDOFF_HELD"
    next_command = approval_readiness_command if handoff_ready else str(audit_meta.get("next_command") or "observe act verify action audit")

    lines = [
        "Observe-act-verify execution handoff:",
        "This is read-only. It packages the final one-primitive computer-control handoff after action audit without observing the screen, taking screenshots, clicking, typing, moving the mouse, approving requests, rerunning actions, writing files, running shell/code, or queuing approvals.",
        "",
        "Handoff state:",
        f"- handoff state: {handoff_state}",
        f"- action audit state: {audit_meta.get('action_audit_state')}",
        f"- ready for approval decision: {'yes' if ready_for_decision else 'no'}",
        f"- action allowed now: {'yes' if action_allowed_now else 'no'}",
        f"- approval decision required: {'yes' if approval_decision_required else 'no'}",
        f"- exact primitive command: `{primitive_command or '<missing>'}`",
        f"- computer control currently enabled: {'yes' if audit_meta.get('computer_control_enabled') else 'no'}",
        f"- natural-language computer-control routing enabled: {'yes' if audit_meta.get('natural_language_computer_control_routing_enabled') else 'no'}",
        f"- real execution approval-gated: {'yes' if audit_meta.get('real_execution_approval_gated') else 'no'}",
        f"- missing blockers: {', '.join(missing) if missing else 'none'}",
        "",
        "Approval handoff:",
        f"- next command: `{next_command}`",
        f"- approval readiness: `{approval_readiness_command}`",
        f"- last-look packet: `{approval_packet_command}`",
        f"- approval chain proof: `{approval_chain_command}`",
        "- the operator must approve or reject the pending approval explicitly; this packet does not approve anything.",
        "",
        "Post-run proof required after any approved primitive:",
    ]
    lines.extend(f"- `{command}`" for command in post_run_commands)
    lines.extend(
        [
            "",
            "One-shot execution boundary:",
            "- The handoff applies only to the exact primitive command shown above.",
            "- It does not allow natural-language desktop control, multi-step clicking, typed text changes, new coordinates, or new screen assumptions.",
            "- Any changed screen, app, coordinate, text, expectation, approval id, run id, receipt, audit result, health result, or operator review needs a fresh handoff.",
            "",
            "Required command chain:",
            f"- command count: {len(handoff_commands)}",
        ]
    )
    lines.extend(f"- `{command}`" for command in handoff_commands)
    lines.extend(
        [
            "",
            "Operator limit:",
            f"- {OPERATOR_LIMIT_RULE}",
        ]
    )
    return ToolResult(
        "observe_act_verify_execution_handoff",
        audit_result.ok,
        "\n".join(lines),
        _safe_metadata(
            action=audit_meta.get("action"),
            x=audit_meta.get("x"),
            y=audit_meta.get("y"),
            text_chars=audit_meta.get("text_chars", 0),
            expectation=audit_meta.get("expectation"),
            handoff_state=handoff_state,
            ready_for_approval_decision=handoff_ready,
            oav_execution_handoff_ready=handoff_ready,
            action_allowed_now=action_allowed_now,
            approval_decision_required=approval_decision_required,
            exact_primitive_command=primitive_command,
            approval_readiness_command=approval_readiness_command,
            approval_packet_command=approval_packet_command,
            approval_chain_command=approval_chain_command,
            post_run_commands=post_run_commands,
            post_run_command_count=len(post_run_commands),
            next_command=next_command,
            required_commands=handoff_commands,
            required_command_count=len(handoff_commands),
            action_audit_state=audit_meta.get("action_audit_state"),
            final_review_state=audit_meta.get("final_review_state"),
            computer_control_enabled=audit_meta.get("computer_control_enabled") is True,
            natural_language_routing_enabled=audit_meta.get("natural_language_routing_enabled"),
            natural_language_computer_control_routing_enabled=audit_meta.get("natural_language_computer_control_routing_enabled"),
            real_execution_approval_gated=audit_meta.get("real_execution_approval_gated"),
            approval_required_for_real_use=True,
            missing_blockers=missing,
            missing_blocker_count=len(missing),
            action_audit_metadata=audit_meta,
            action_audit_output=audit_result.output,
        ),
    )


def observe_act_verify_post_run_closure(args: dict[str, Any]) -> ToolResult:
    raw_spec = str(args.get("spec") or "")
    spec = _clean_text(raw_spec, limit=MAX_OBJECTIVE_CHARS)
    action = _clean_text(args.get("action"), limit=40).lower()
    expectation = _clean_text(args.get("expectation"), limit=MAX_EXPECTATION_CHARS)
    observation = _clean_text(args.get("observation") or args.get("observed") or args.get("evidence"), limit=MAX_OBSERVATION_CHARS)
    after_observation = _clean_text(
        args.get("after_observation") or args.get("after") or args.get("verification_observation"),
        limit=MAX_OBSERVATION_CHARS,
    )
    source = _clean_text(args.get("source") or args.get("observation_source"), limit=80).lower()
    text = _clean_text(args.get("text"), limit=MAX_TYPED_TEXT_CHARS)
    x_raw = args.get("x")
    y_raw = args.get("y")
    approval_id = args.get("approval_id")
    approved_run_id = args.get("approved_run_id")
    verification_run_id = args.get("verification_run_id") or args.get("verification_receipt")
    approval_evidence = _clean_text(args.get("approval_evidence") or args.get("readiness"), limit=MAX_OBSERVATION_CHARS)
    approval_packet_evidence = _clean_text(args.get("approval_packet_evidence") or args.get("packet"), limit=MAX_OBSERVATION_CHARS)
    approval_chain_evidence = _clean_text(args.get("approval_chain_evidence") or args.get("chain"), limit=MAX_OBSERVATION_CHARS)
    audit_evidence = _clean_text(args.get("audit_evidence") or args.get("audit"), limit=MAX_OBSERVATION_CHARS)
    health_evidence = _clean_text(args.get("health_evidence") or args.get("health"), limit=MAX_OBSERVATION_CHARS)
    operator_review = _clean_text(args.get("operator_review") or args.get("review"), limit=MAX_OBSERVATION_CHARS)
    post_verification = _clean_text(
        args.get("post_verification") or args.get("verification") or args.get("verification_evidence"),
        limit=MAX_OBSERVATION_CHARS,
    )
    post_health = _clean_text(args.get("post_health") or args.get("post_health_evidence"), limit=MAX_OBSERVATION_CHARS)
    post_audit = _clean_text(args.get("post_audit") or args.get("post_audit_evidence"), limit=MAX_OBSERVATION_CHARS)
    learning_evidence = _clean_text(
        args.get("learning_evidence") or args.get("learning") or args.get("after_action_learning"),
        limit=MAX_OBSERVATION_CHARS,
    )

    if spec:
        if not action:
            action = _clean_text(_extract_field(spec, ("action", "step")), limit=40).lower()
        if not expectation:
            expectation = _clean_text(_extract_field(spec, ("expectation", "expected", "expect")), limit=MAX_EXPECTATION_CHARS)
        if not observation:
            observation = _clean_text(_extract_field(spec, ("observation", "observed", "evidence", "screen")), limit=MAX_OBSERVATION_CHARS)
        if not after_observation:
            after_observation = _clean_text(_extract_field(spec, ("after observation", "after", "after screen")), limit=MAX_OBSERVATION_CHARS)
        if not source:
            source = _clean_text(_extract_field(spec, ("observation source", "source")), limit=80).lower()
        if not text:
            text = _clean_text(_extract_after_keyword(spec, (" text ", " type ", " value ")), limit=MAX_TYPED_TEXT_CHARS)
        if x_raw is None:
            x_raw = _extract_int_after_keyword(spec, (" x ", " at x ", " coordinate x "))
        if y_raw is None:
            y_raw = _extract_int_after_keyword(spec, (" y ", " at y ", " coordinate y "))
        if approval_id is None:
            approval_id = _extract_int_after_keyword(spec, (" approval ", " approval id "))
        if approved_run_id is None:
            approved_run_id = _extract_int_after_keyword(spec, (" approved run ", " approved run id "))
        if verification_run_id is None:
            verification_run_id = _extract_int_after_keyword(spec, (" verification receipt ", " verification run id "))
        if not approval_evidence:
            approval_evidence = _clean_text(_extract_field(spec, ("approval evidence", "approval readiness", "readiness")), limit=MAX_OBSERVATION_CHARS)
        if not approval_packet_evidence:
            approval_packet_evidence = _clean_text(_extract_field(spec, ("approval packet evidence", "approval packet", "packet")), limit=MAX_OBSERVATION_CHARS)
        if not approval_chain_evidence:
            approval_chain_evidence = _clean_text(_extract_field(spec, ("approval chain evidence", "approval chain proof", "chain")), limit=MAX_OBSERVATION_CHARS)
        if not audit_evidence:
            audit_evidence = _clean_text(_extract_field(spec, ("audit evidence", "execution audit", "audit")), limit=MAX_OBSERVATION_CHARS)
        if not health_evidence:
            health_evidence = _clean_text(_extract_field(spec, ("health evidence", "execution health", "health")), limit=MAX_OBSERVATION_CHARS)
        if not operator_review:
            operator_review = _clean_text(_extract_field(spec, ("operator review", "human review", "review")), limit=MAX_OBSERVATION_CHARS)
        if not post_verification:
            post_verification = _clean_text(_extract_field(spec, ("post verification", "verification evidence", "verification")), limit=MAX_OBSERVATION_CHARS)
        if not post_health:
            post_health = _clean_text(_extract_field(spec, ("post health", "post-run health")), limit=MAX_OBSERVATION_CHARS)
        if not post_audit:
            post_audit = _clean_text(_extract_field(spec, ("post audit", "post-run audit")), limit=MAX_OBSERVATION_CHARS)
        if not learning_evidence:
            learning_evidence = _clean_text(_extract_field(spec, ("learning evidence", "after action learning", "learning")), limit=MAX_OBSERVATION_CHARS)

    handoff_result = observe_act_verify_execution_handoff(
        {
            "spec": raw_spec,
            "action": action,
            "expectation": expectation,
            "observation": observation,
            "after_observation": after_observation,
            "source": source,
            "text": text,
            "x": x_raw,
            "y": y_raw,
            "approval_id": approval_id,
            "approved_run_id": approved_run_id,
            "verification_run_id": verification_run_id,
            "approval_evidence": approval_evidence,
            "approval_packet_evidence": approval_packet_evidence,
            "approval_chain_evidence": approval_chain_evidence,
            "audit_evidence": audit_evidence,
            "health_evidence": health_evidence,
            "operator_review": operator_review,
        }
    )
    handoff_meta = handoff_result.metadata
    approved_run = _bounded_coordinate(approved_run_id)
    verification_run = _bounded_coordinate(verification_run_id)
    if approved_run is None:
        approved_run = handoff_meta.get("approved_run_id")
    if verification_run is None:
        verification_run = handoff_meta.get("verification_run_id")
    action_audit_meta = handoff_meta.get("action_audit_metadata") if isinstance(handoff_meta.get("action_audit_metadata"), dict) else {}
    final_review_meta = action_audit_meta.get("final_review_metadata") if isinstance(action_audit_meta.get("final_review_metadata"), dict) else {}
    cockpit_meta = final_review_meta.get("cockpit_metadata") if isinstance(final_review_meta.get("cockpit_metadata"), dict) else {}
    bridge_meta = cockpit_meta.get("approval_bridge_metadata") if isinstance(cockpit_meta.get("approval_bridge_metadata"), dict) else {}
    if approved_run is None:
        approved_run = bridge_meta.get("approved_run_id")
    if verification_run is None:
        verification_run = bridge_meta.get("verification_run_id")

    post_text = " ".join([raw_spec, post_verification, post_health, post_audit, learning_evidence]).lower()
    verification_ready = (
        bool(post_verification)
        and ("verification" in post_verification.lower() or "receipt" in post_verification.lower() or "verified" in post_verification.lower())
    ) or "verification receipt" in post_text
    health_ready = bool(post_health) and ("health" in post_health.lower() or "passed" in post_health.lower()) or any(
        marker in post_text for marker in ("execution health passed", "health passed", "execution health reviewed")
    )
    audit_ready = bool(post_audit) and ("audit" in post_audit.lower() or "passed" in post_audit.lower()) or any(
        marker in post_text for marker in ("execution audit passed", "audit passed", "execution audit reviewed")
    )
    learning_ready = bool(learning_evidence) and any(
        marker in learning_evidence.lower()
        for marker in ("after-action", "after action", "learning", "reviewed", "no learning needed")
    ) or any(marker in post_text for marker in ("after-action learning", "after action learning", "learning reviewed"))
    receipt_binding_ready = approved_run is not None and verification_run is not None and approved_run == verification_run

    missing = list(handoff_meta.get("missing_blockers") or [])
    if handoff_meta.get("handoff_state") != "OAV_EXECUTION_HANDOFF_READY_FOR_APPROVAL_PACKET":
        missing.append("ready OAV execution handoff")
    if approved_run is None:
        missing.append("approved primitive run id")
    if verification_run is None:
        missing.append("verification receipt run id")
    if approved_run is not None and verification_run is not None and approved_run != verification_run:
        missing.append("verification receipt must match approved primitive run id")
    if not verification_ready:
        missing.append("post-run verification receipt evidence")
    if not health_ready:
        missing.append("post-run execution health evidence")
    if not audit_ready:
        missing.append("post-run execution audit evidence")
    if not learning_ready:
        missing.append("after-action learning evidence")
    if handoff_meta.get("natural_language_computer_control_routing_enabled") is not False:
        missing.append("natural-language computer-control routing disabled proof")
    missing = list(dict.fromkeys(missing))

    next_review_start_command = "observe act verify cockpit: action <next primitive>; expectation <expected screen change>; observation <fresh approved observation>; source approved_screenshot"
    closure_commands = list(
        dict.fromkeys(
            list(handoff_meta.get("required_commands") or [])
            + [
                f"verification receipt {approved_run if approved_run is not None else '<approved run id>'}",
                "execution health report",
                "execution audit gate",
                f"after-action learning packet {approved_run if approved_run is not None else '<approved run id>'}",
                "observe act verify post-run closure",
            ]
        )
    )
    candidate_closure_ready = not missing
    candidate_closure_state = "OAV_POST_RUN_CLOSURE_READY_FOR_NEXT_PRIMITIVE_REVIEW" if candidate_closure_ready else "OAV_POST_RUN_CLOSURE_HELD"
    candidate_next_primitive_review_state = "FRESH_OAV_REVIEW_UNLOCKED" if candidate_closure_ready else "FRESH_OAV_REVIEW_HELD"
    candidate_metadata = _safe_metadata(
        action=handoff_meta.get("action"),
        x=handoff_meta.get("x"),
        y=handoff_meta.get("y"),
        text_chars=handoff_meta.get("text_chars", 0),
        expectation=handoff_meta.get("expectation"),
        closure_state=candidate_closure_state,
        ready_for_next_primitive_review=candidate_closure_ready,
        oav_post_run_closure_ready=candidate_closure_ready,
        next_primitive_review_state=candidate_next_primitive_review_state,
        next_review_start_command=next_review_start_command,
        next_review_requires_fresh_observation=True,
        next_review_requires_new_route_lock=True,
        next_review_requires_new_approval_bridge=True,
        next_review_requires_new_final_review=True,
        next_review_requires_new_action_audit=True,
        next_review_requires_new_execution_handoff=True,
        previous_approval_reusable_for_next_primitive=False,
        previous_observation_reusable_for_next_primitive=False,
        previous_verification_reusable_for_next_primitive=False,
        previous_approved_run_reusable_for_next_primitive=False,
        prior_primitive_proof_only=True,
        action_allowed_now=False,
        approved_run_id=approved_run,
        verification_run_id=verification_run,
        receipt_binding_ready=receipt_binding_ready,
        post_run_verification_ready=verification_ready,
        post_run_health_ready=health_ready,
        post_run_audit_ready=audit_ready,
        after_action_learning_ready=learning_ready,
        handoff_state=handoff_meta.get("handoff_state"),
        action_audit_state=handoff_meta.get("action_audit_state"),
        final_review_state=handoff_meta.get("final_review_state"),
        natural_language_routing_enabled=False,
        natural_language_computer_control_routing_enabled=False,
        real_execution_approval_gated=True,
        approval_required_for_real_use=True,
        missing_blockers=missing,
        missing_blocker_count=len(missing),
        next_command=next_review_start_command,
        required_commands=closure_commands,
        required_command_count=len(closure_commands),
        execution_handoff_metadata=handoff_meta,
        handoff_output=handoff_result.output,
    )
    closure_ready = _oav_post_run_closure_ready_from_metadata(candidate_metadata)
    if candidate_closure_ready and not closure_ready:
        missing = list(dict.fromkeys(missing + ["post-run closure production readiness"]))
    closure_state = "OAV_POST_RUN_CLOSURE_READY_FOR_NEXT_PRIMITIVE_REVIEW" if closure_ready else "OAV_POST_RUN_CLOSURE_HELD"
    next_primitive_review_state = "FRESH_OAV_REVIEW_UNLOCKED" if closure_ready else "FRESH_OAV_REVIEW_HELD"
    next_command = (
        next_review_start_command
        if closure_ready
        else f"verification receipt {approved_run if approved_run is not None else '<approved run id>'}"
        if not verification_ready
        else "execution health report"
        if not health_ready
        else "execution audit gate"
        if not audit_ready
        else f"after-action learning packet {approved_run if approved_run is not None else '<approved run id>'}"
    )

    lines = [
        "Observe-act-verify post-run closure:",
        "This is read-only. It closes the proof loop after one approved computer-control primitive by binding the approved run, matching verification receipt, execution health, execution audit, and after-action learning before any next primitive review.",
        "",
        "Closure state:",
        f"- closure state: {closure_state}",
        f"- ready for next primitive review: {'yes' if closure_ready else 'no'}",
        "- action allowed now: no",
        "- natural-language computer-control routing enabled: no",
        f"- approved primitive run id: {approved_run if approved_run is not None else '<missing>'}",
        f"- verification receipt run id: {verification_run if verification_run is not None else '<missing>'}",
        f"- receipt binding ready: {'yes' if receipt_binding_ready else 'no'}",
        f"- post-run verification evidence: {'present' if verification_ready else 'missing'}",
        f"- post-run health evidence: {'present' if health_ready else 'missing'}",
        f"- post-run audit evidence: {'present' if audit_ready else 'missing'}",
        f"- after-action learning evidence: {'present' if learning_ready else 'missing'}",
        f"- missing blockers: {', '.join(missing) if missing else 'none'}",
        "",
        "Next command:",
        f"- `{next_command}`",
        "",
        "Continuation boundary:",
        "- A ready closure does not authorize the next click, type, mouse move, screenshot, or natural-language desktop-control route.",
        "- It only says the previous one-shot primitive has enough post-run proof for Jarvis to start a fresh OAV cockpit review for the next primitive.",
        "- Any next primitive needs fresh observation, route lock, approval bridge, final review, action audit, execution handoff, approval, and post-run closure.",
        "- The previous observation, approval id, approved run, and verification receipt are proof of the prior primitive only; they cannot be reused as permission for the next primitive.",
        f"- next primitive review state: {next_primitive_review_state}",
        "- fresh observation required before next primitive review: yes",
        "- previous primitive approval reusable for next primitive: no",
        "- previous primitive observation reusable for next primitive: no",
        "- previous primitive verification reusable for next primitive: no",
        "",
        "Closure command chain:",
        f"- command count: {len(closure_commands)}",
    ]
    lines.extend(f"- `{command}`" for command in closure_commands)
    lines.extend(
        [
            "",
            "Operator limit:",
            f"- {OPERATOR_LIMIT_RULE}",
        ]
    )

    return ToolResult(
        "observe_act_verify_post_run_closure",
        handoff_result.ok,
        "\n".join(lines),
        _safe_metadata(
            action=handoff_meta.get("action"),
            x=handoff_meta.get("x"),
            y=handoff_meta.get("y"),
            text_chars=handoff_meta.get("text_chars", 0),
            expectation=handoff_meta.get("expectation"),
            closure_state=closure_state,
            ready_for_next_primitive_review=closure_ready,
            oav_post_run_closure_ready=closure_ready,
            next_primitive_review_state=next_primitive_review_state,
            next_review_start_command=next_review_start_command,
            next_review_requires_fresh_observation=True,
            next_review_requires_new_route_lock=True,
            next_review_requires_new_approval_bridge=True,
            next_review_requires_new_final_review=True,
            next_review_requires_new_action_audit=True,
            next_review_requires_new_execution_handoff=True,
            previous_approval_reusable_for_next_primitive=False,
            previous_observation_reusable_for_next_primitive=False,
            previous_verification_reusable_for_next_primitive=False,
            previous_approved_run_reusable_for_next_primitive=False,
            prior_primitive_proof_only=True,
            action_allowed_now=False,
            approved_run_id=approved_run,
            verification_run_id=verification_run,
            receipt_binding_ready=receipt_binding_ready,
            post_run_verification_ready=verification_ready,
            post_run_health_ready=health_ready,
            post_run_audit_ready=audit_ready,
            after_action_learning_ready=learning_ready,
            handoff_state=handoff_meta.get("handoff_state"),
            action_audit_state=handoff_meta.get("action_audit_state"),
            final_review_state=handoff_meta.get("final_review_state"),
            natural_language_routing_enabled=False,
            natural_language_computer_control_routing_enabled=False,
            real_execution_approval_gated=True,
            approval_required_for_real_use=True,
            missing_blockers=missing,
            missing_blocker_count=len(missing),
            next_command=next_command,
            required_commands=closure_commands,
            required_command_count=len(closure_commands),
            handoff_metadata=handoff_meta,
            handoff_output=handoff_result.output,
        ),
    )


def observe_act_verify_cycle_ledger(args: dict[str, Any]) -> ToolResult:
    closure_result = observe_act_verify_post_run_closure(args)
    closure_meta = closure_result.metadata
    handoff_meta = closure_meta.get("handoff_metadata") if isinstance(closure_meta.get("handoff_metadata"), dict) else {}
    action_audit_meta = handoff_meta.get("action_audit_metadata") if isinstance(handoff_meta.get("action_audit_metadata"), dict) else {}
    final_review_meta = action_audit_meta.get("final_review_metadata") if isinstance(action_audit_meta.get("final_review_metadata"), dict) else {}
    cockpit_meta = final_review_meta.get("cockpit_metadata") if isinstance(final_review_meta.get("cockpit_metadata"), dict) else {}
    proof_meta = cockpit_meta.get("proof_metadata") if isinstance(cockpit_meta.get("proof_metadata"), dict) else {}
    bridge_meta = cockpit_meta.get("approval_bridge_metadata") if isinstance(cockpit_meta.get("approval_bridge_metadata"), dict) else {}
    route_lock_state = bridge_meta.get("route_lock_state") or proof_meta.get("route_lock_state") or ""

    stage_tuples = [
        ("screen_observation_confidence", proof_meta.get("pre_action_proof_state"), proof_meta.get("pre_action_confidence_ready") is True),
        ("screen_verification_contract", proof_meta.get("verification_verdict"), proof_meta.get("after_action_verification_ready") is True),
        ("observe_act_verify_proof", proof_meta.get("proof_state"), proof_meta.get("primitive_ready_for_approval_review") is True),
        ("route_lock", route_lock_state, bridge_meta.get("route_unlock_candidate") is True),
        ("approval_bridge", bridge_meta.get("bridge_state"), bridge_meta.get("final_route_review_ready") is True),
        ("cockpit", cockpit_meta.get("cockpit_state"), cockpit_meta.get("final_route_review_ready") is True),
        ("final_review", final_review_meta.get("final_review_state"), final_review_meta.get("ready_for_human_decision") is True),
        ("action_audit", action_audit_meta.get("action_audit_state"), action_audit_meta.get("ready_for_approval_decision") is True),
        ("execution_handoff", handoff_meta.get("handoff_state"), handoff_meta.get("ready_for_approval_decision") is True),
        ("post_run_closure", closure_meta.get("closure_state"), closure_meta.get("ready_for_next_primitive_review") is True),
    ]
    missing = list(closure_meta.get("missing_blockers") or [])
    for stage, _state, ready in stage_tuples:
        if ready is not True:
            missing.append(stage)
    missing = list(dict.fromkeys(str(item) for item in missing if str(item).strip()))

    cycle_ready = closure_meta.get("ready_for_next_primitive_review") is True and not missing
    cycle_state = "OAV_CYCLE_LEDGER_READY_FOR_FRESH_REVIEW" if cycle_ready else "OAV_CYCLE_LEDGER_HELD"
    required_commands = list(dict.fromkeys(list(closure_meta.get("required_commands") or []) + [
        "observe act verify cycle ledger",
        "completion audit: improve AGI gate vision observe-act-verify",
        "evidence ledger",
        "completion claim gate: improve AGI gate vision observe-act-verify",
    ]))
    next_command = (
        closure_meta.get("next_review_start_command")
        if cycle_ready
        else closure_meta.get("next_command") or "observe act verify post-run closure"
    )
    fresh_review_preflight_queue = [
        "screen observation freshness: <next primitive approved observation>",
        "observe act verify proof: action <next primitive>; expectation <expected screen change>; observation <fresh approved observation>; source approved_screenshot",
        "observe act verify route lock: action <next primitive>; approval <new approval id>",
        "observe act verify approval bridge: action <next primitive>; approval <new approval id>; approved run <new approved run id>; verification receipt <new receipt id>",
        "observe act verify final review: action <next primitive>",
        "observe act verify action audit: action <next primitive>",
        "observe act verify execution handoff: action <next primitive>",
        "observe act verify post-run closure: action <next primitive>",
    ]
    fresh_review_contract_rows = [
        {
            "item": "fresh_screen_observation",
            "source": "screen observation freshness packet",
            "required": True,
            "prior_artifact_reusable": False,
            "authorizes_action_now": False,
            "authorizes_computer_control": False,
            "authorizes_screenshot": False,
            "authorizes_approval": False,
            "authorizes_route_unlock": False,
            "authorizes_verification_shortcut": False,
        },
        {
            "item": "new_route_lock",
            "source": "observe-act-verify route lock",
            "required": True,
            "prior_artifact_reusable": False,
            "authorizes_action_now": False,
            "authorizes_computer_control": False,
            "authorizes_screenshot": False,
            "authorizes_approval": False,
            "authorizes_route_unlock": False,
            "authorizes_verification_shortcut": False,
        },
        {
            "item": "new_approval_bridge",
            "source": "observe-act-verify approval bridge",
            "required": True,
            "prior_artifact_reusable": False,
            "authorizes_action_now": False,
            "authorizes_computer_control": False,
            "authorizes_screenshot": False,
            "authorizes_approval": False,
            "authorizes_route_unlock": False,
            "authorizes_verification_shortcut": False,
        },
        {
            "item": "new_final_review",
            "source": "observe-act-verify final review",
            "required": True,
            "prior_artifact_reusable": False,
            "authorizes_action_now": False,
            "authorizes_computer_control": False,
            "authorizes_screenshot": False,
            "authorizes_approval": False,
            "authorizes_route_unlock": False,
            "authorizes_verification_shortcut": False,
        },
        {
            "item": "new_action_audit",
            "source": "observe-act-verify action audit",
            "required": True,
            "prior_artifact_reusable": False,
            "authorizes_action_now": False,
            "authorizes_computer_control": False,
            "authorizes_screenshot": False,
            "authorizes_approval": False,
            "authorizes_route_unlock": False,
            "authorizes_verification_shortcut": False,
        },
        {
            "item": "new_execution_handoff",
            "source": "observe-act-verify execution handoff",
            "required": True,
            "prior_artifact_reusable": False,
            "authorizes_action_now": False,
            "authorizes_computer_control": False,
            "authorizes_screenshot": False,
            "authorizes_approval": False,
            "authorizes_route_unlock": False,
            "authorizes_verification_shortcut": False,
        },
        {
            "item": "new_post_run_closure",
            "source": "observe-act-verify post-run closure",
            "required": True,
            "prior_artifact_reusable": False,
            "authorizes_action_now": False,
            "authorizes_computer_control": False,
            "authorizes_screenshot": False,
            "authorizes_approval": False,
            "authorizes_route_unlock": False,
            "authorizes_verification_shortcut": False,
        },
        {
            "item": "fresh_cycle_ledger_review",
            "source": "observe-act-verify cycle ledger",
            "required": True,
            "prior_artifact_reusable": False,
            "authorizes_action_now": False,
            "authorizes_computer_control": False,
            "authorizes_screenshot": False,
            "authorizes_approval": False,
            "authorizes_route_unlock": False,
            "authorizes_verification_shortcut": False,
        },
    ]
    for row in fresh_review_contract_rows:
        row.update(
            {
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
            }
        )
    fresh_review_contract_ready = all(
        row["required"] is True
        and row["prior_artifact_reusable"] is False
        and row["authorizes_action_now"] is False
        and row["authorizes_computer_control"] is False
        and row["authorizes_screenshot"] is False
        and row["authorizes_approval"] is False
        and row["authorizes_route_unlock"] is False
        and row["authorizes_verification_shortcut"] is False
        and row["authorizes_model_call"] is False
        and row["authorizes_tool_execution"] is False
        and row["authorizes_personal_data_read"] is False
        and row["authorizes_external_side_effect"] is False
        for row in fresh_review_contract_rows
    )
    stage_rows = [
        {
            "stage": stage,
            "state": state or "",
            "ready": ready,
            "reusable_for_next_primitive": False,
            "authorizes_action_now": False,
            "authorizes_computer_control": False,
            "authorizes_screenshot": False,
            "authorizes_approval": False,
            "authorizes_route_unlock": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_personal_data_read": False,
            "authorizes_external_side_effect": False,
        }
        for stage, state, ready in stage_tuples
    ]
    oav_cycle_ledger_token_sha256 = _oav_cycle_ledger_token_sha256(
        action=str(closure_meta.get("action") or ""),
        x=closure_meta.get("x"),
        y=closure_meta.get("y"),
        expectation=str(closure_meta.get("expectation") or ""),
        approved_run_id=closure_meta.get("approved_run_id"),
        verification_run_id=closure_meta.get("verification_run_id"),
        cycle_state=cycle_state,
        fresh_review_contract_rows=fresh_review_contract_rows,
        stage_rows=stage_rows,
    )
    oav_cycle_ledger_token_boundary_rows = [
        {
            "item": "oav_cycle_ledger_token",
            "status": "present",
            "source": "observe_act_verify_cycle_ledger",
            "token_sha256": oav_cycle_ledger_token_sha256,
        },
        {
            "item": "fresh_review_contract_boundary",
            "status": "prior_primitive_proof_only_not_reusable",
            "source": "fresh_review_contract_rows",
            "token_sha256": oav_cycle_ledger_token_sha256,
        },
        {
            "item": "stage_row_boundary",
            "status": "prior_stage_proof_only_not_reusable",
            "source": "stage_rows",
            "token_sha256": oav_cycle_ledger_token_sha256,
        },
        {
            "item": "next_primitive_review_boundary",
            "status": "fresh_oav_review_required",
            "source": "observe_act_verify_cycle_ledger",
            "token_sha256": oav_cycle_ledger_token_sha256,
        },
    ]
    for row in oav_cycle_ledger_token_boundary_rows:
        row.update(
            {
                "proof_only": True,
                "authorizes_action_now": False,
                "authorizes_computer_control": False,
                "authorizes_screenshot": False,
                "authorizes_approval": False,
                "authorizes_route_unlock": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "reusable_for_next_primitive": False,
            }
        )
    oav_cycle_ledger_token_boundary_ready = _oav_cycle_ledger_token_boundary_ready(
        oav_cycle_ledger_token_sha256,
        oav_cycle_ledger_token_boundary_rows,
    )

    lines = [
        "Observe-act-verify cycle ledger:",
        "This is read-only. It binds one computer-control primitive from approved screen-observation evidence through proof, route lock, approval bridge, cockpit, final review, action audit, execution handoff, and post-run closure.",
        "",
        "Cycle state:",
        f"- cycle state: {cycle_state}",
        f"- ready for fresh next primitive review: {'yes' if cycle_ready else 'no'}",
        "- action allowed now: no",
        "- computer control enabled: no",
        "- natural-language computer-control routing enabled: no",
        f"- approved primitive run id: {closure_meta.get('approved_run_id') if closure_meta.get('approved_run_id') is not None else '<missing>'}",
        f"- verification receipt run id: {closure_meta.get('verification_run_id') if closure_meta.get('verification_run_id') is not None else '<missing>'}",
        f"- receipt binding ready: {'yes' if closure_meta.get('receipt_binding_ready') else 'no'}",
        f"- missing blockers: {', '.join(missing) if missing else 'none'}",
        f"- next command: `{next_command}`",
        "",
        "Cycle stages:",
    ]
    for row in stage_rows:
        lines.append(f"- {row['stage']}: {'ready' if row['ready'] else 'held'}; state {row['state'] or 'unknown'}")
    lines.extend(
        [
            "",
            "Fresh-review boundary:",
            "- A ready cycle ledger does not authorize the next click, type, mouse move, screenshot, or natural-language desktop-control route.",
            "- It only proves the previous one-shot primitive is closed enough to begin a fresh OAV cockpit review.",
            "- The next primitive still needs a fresh observation, new route lock, new approval bridge, new final review, new action audit, new execution handoff, explicit approval, and new post-run closure.",
            "- Previous observation, approval, approved run, and verification receipt are prior-primitive proof only.",
            "- Prior primitive proof cannot authorize a new action, model call, tool execution, screenshot, computer-control primitive, personal-data read, external side effect, approval, route unlock, or verification shortcut.",
            "",
            "OAV cycle ledger token boundary:",
            f"- token sha256: {oav_cycle_ledger_token_sha256}",
            "- token is proof-only: yes",
            "- reusable for next primitive: no",
            "- authorizes action/computer control/screenshot/approval/route unlock/model/tool/personal-data/external side effect: no",
            "",
            "Fresh-review preflight queue:",
            f"- command count: {len(fresh_review_preflight_queue)}",
            *[f"- `{command}`" for command in fresh_review_preflight_queue],
            "",
            "Fresh-review contract rows:",
            f"- row count: {len(fresh_review_contract_rows)}",
            *[
                f"- {row['item']}: required yes; prior reusable no; authorizes action no; authorizes model call no; authorizes tool execution no; authorizes computer control no; authorizes screenshot no; authorizes personal-data read no; authorizes external side effect no; authorizes approval no; authorizes route unlock no; authorizes verification shortcut no"
                for row in fresh_review_contract_rows
            ],
            "",
            "Proof queue:",
            f"- command count: {len(required_commands)}",
        ]
    )
    lines.extend(f"- `{command}`" for command in required_commands)
    lines.extend(
        [
            "",
            "Operator limit:",
            f"- {OPERATOR_LIMIT_RULE}",
        ]
    )

    candidate_metadata = _safe_metadata(
            action=closure_meta.get("action"),
            x=closure_meta.get("x"),
            y=closure_meta.get("y"),
            text_chars=closure_meta.get("text_chars", 0),
            expectation=closure_meta.get("expectation"),
            cycle_state=cycle_state,
            ready_for_fresh_next_primitive_review=cycle_ready,
            oav_cycle_ledger_ready=cycle_ready,
            action_allowed_now=False,
            computer_control_enabled=False,
            natural_language_routing_enabled=False,
            natural_language_computer_control_routing_enabled=False,
            real_execution_approval_gated=True,
            approval_required_for_real_use=True,
            approved_run_id=closure_meta.get("approved_run_id"),
            verification_run_id=closure_meta.get("verification_run_id"),
            receipt_binding_ready=closure_meta.get("receipt_binding_ready") is True,
            next_review_start_command=closure_meta.get("next_review_start_command"),
            next_review_requires_fresh_observation=True,
            previous_approval_reusable_for_next_primitive=False,
            previous_observation_reusable_for_next_primitive=False,
            previous_verification_reusable_for_next_primitive=False,
            previous_approved_run_reusable_for_next_primitive=False,
            prior_primitive_proof_only=True,
            oav_cycle_ledger_token_sha256=oav_cycle_ledger_token_sha256,
            oav_cycle_ledger_token_present=True,
            oav_cycle_ledger_token_boundary_rows=oav_cycle_ledger_token_boundary_rows,
            oav_cycle_ledger_token_boundary_row_count=len(oav_cycle_ledger_token_boundary_rows),
            oav_cycle_ledger_token_boundary_ready=oav_cycle_ledger_token_boundary_ready,
            oav_cycle_ledger_token_authorizes_action_now=False,
            oav_cycle_ledger_token_authorizes_computer_control=False,
            oav_cycle_ledger_token_authorizes_screenshot=False,
            oav_cycle_ledger_token_authorizes_approval=False,
            oav_cycle_ledger_token_authorizes_route_unlock=False,
            oav_cycle_ledger_token_authorizes_model_call=False,
            oav_cycle_ledger_token_authorizes_tool_execution=False,
            oav_cycle_ledger_token_authorizes_personal_data_read=False,
            oav_cycle_ledger_token_authorizes_external_side_effect=False,
            oav_cycle_ledger_token_reusable_for_next_primitive=False,
            next_primitive_requires_new_oav_cycle_ledger_token=True,
            prior_primitive_proof_authorizes_new_action=False,
            prior_primitive_proof_authorizes_computer_control=False,
            prior_primitive_proof_authorizes_screenshot=False,
            prior_primitive_proof_authorizes_model_call=False,
            prior_primitive_proof_authorizes_tool_execution=False,
            prior_primitive_proof_authorizes_personal_data_read=False,
            prior_primitive_proof_authorizes_external_side_effect=False,
            prior_primitive_proof_authorizes_approval=False,
            prior_primitive_proof_authorizes_route_unlock=False,
            prior_primitive_proof_authorizes_verification_shortcut=False,
            fresh_review_preflight_queue=fresh_review_preflight_queue,
            fresh_review_preflight_queue_count=len(fresh_review_preflight_queue),
            fresh_review_next_preflight_command=fresh_review_preflight_queue[0],
            fresh_review_contract_rows=fresh_review_contract_rows,
            fresh_review_contract_row_count=len(fresh_review_contract_rows),
            fresh_review_contract_ready=fresh_review_contract_ready,
            missing_blockers=missing,
            missing_blocker_count=len(missing),
            next_command=next_command,
            required_commands=required_commands,
            required_command_count=len(required_commands),
            stage_rows=stage_rows,
            stage_count=len(stage_rows),
            proof_metadata=proof_meta,
            approval_bridge_metadata=bridge_meta,
            cockpit_metadata=cockpit_meta,
            final_review_metadata=final_review_meta,
            action_audit_metadata=action_audit_meta,
            handoff_metadata=handoff_meta,
            post_run_closure_metadata=closure_meta,
            post_run_closure_output=closure_result.output,
        )
    cycle_ready = _oav_cycle_ledger_ready_from_metadata(candidate_metadata)
    candidate_metadata["ready_for_fresh_next_primitive_review"] = cycle_ready
    candidate_metadata["oav_cycle_ledger_ready"] = cycle_ready
    if not cycle_ready:
        candidate_metadata["cycle_state"] = "OAV_CYCLE_LEDGER_HELD"
        candidate_metadata["next_command"] = closure_meta.get("next_command") or "observe act verify post-run closure"

    return ToolResult(
        "observe_act_verify_cycle_ledger",
        closure_result.ok,
        "\n".join(lines),
        candidate_metadata,
    )


def _significant_terms(text: str) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "be",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
    terms: list[str] = []
    for raw in text.lower().replace("/", " ").replace("-", " ").split():
        term = raw.strip(".,:;!?()[]{}'\"")
        if len(term) < 3 or term in stopwords or term in terms:
            continue
        terms.append(term)
    return terms[:12]


def _extract_int_after_keyword(text: str, keywords: tuple[str, ...]) -> int | None:
    padded = f" {text} "
    low = padded.lower()
    for keyword in keywords:
        index = low.find(keyword)
        if index == -1:
            continue
        tail = padded[index + len(keyword):].strip()
        token = tail.split(maxsplit=1)[0].strip(" ,.:;")
        try:
            return int(token)
        except ValueError:
            continue
    return None


def _extract_after_keyword(text: str, keywords: tuple[str, ...]) -> str:
    padded = f" {text} "
    low = padded.lower()
    for keyword in keywords:
        index = low.find(keyword)
        if index == -1:
            continue
        return padded[index + len(keyword):].strip(" :")
    return ""


def _extract_field(text: str, field_names: tuple[str, ...]) -> str:
    for field_name in sorted(field_names, key=len, reverse=True):
        pattern = rf"(?:^|;)\s*{re.escape(field_name)}\s*:?\s*(?P<value>[^;]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group("value").strip()
    return ""


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


def _require_enabled(tool_name: str) -> ToolResult | None:
    if not _enabled:
        return _computer_input_failure(
            tool_name,
            "Computer control is disabled. Run `computer control readiness` first.",
            commands=("computer control readiness",),
            controls_computer=False,
        )
    return None


def _locate_target_center(args: dict[str, Any]) -> tuple[int, int] | None:
    target_image = _clean_text(args.get("target_image") or args.get("image") or args.get("template"), limit=MAX_FILENAME_CHARS)
    if not target_image:
        return None
    path = Path(target_image).expanduser()
    gui = _pyautogui()
    try:
        confidence = float(args.get("confidence", 0.8))
    except (TypeError, ValueError):
        confidence = 0.8
    confidence = max(0.1, min(1.0, confidence))
    try:
        center = gui.locateCenterOnScreen(str(path), confidence=confidence)
    except TypeError:
        center = gui.locateCenterOnScreen(str(path))
    if center is None:
        return None
    x = _bounded_coordinate(getattr(center, "x", center[0] if isinstance(center, tuple) else None))
    y = _bounded_coordinate(getattr(center, "y", center[1] if isinstance(center, tuple) else None))
    if x is None or y is None:
        return None
    return x, y


def screen_size(_: dict[str, Any]) -> ToolResult:
    try:
        size = _pyautogui().size()
        return ToolResult("screen_size", True, _with_operator_limit(f"{size.width}x{size.height}"), _safe_metadata(width=size.width, height=size.height, observes_screen=True))
    except Exception as exc:
        try:
            result = subprocess.run(
                ["osascript", "-e", 'tell application "Finder" to get bounds of window of desktop'],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                parts = [int(part.strip()) for part in result.stdout.strip().split(",")]
                if len(parts) == 4:
                    width = parts[2] - parts[0]
                    height = parts[3] - parts[1]
                    return ToolResult("screen_size", True, _with_operator_limit(f"{width}x{height}"), _safe_metadata(width=width, height=height, observes_screen=True))
        except Exception:
            pass
        return _computer_read_failure("screen_size", "read screen size", exc, observes_screen=True)


def mouse_position(_: dict[str, Any]) -> ToolResult:
    try:
        point = _pyautogui().position()
        return ToolResult("mouse_position", True, f"{point.x},{point.y}", _safe_metadata(x=point.x, y=point.y, controls_computer=True))
    except Exception as exc:
        return _computer_read_failure("mouse_position", "read mouse position", exc, controls_computer=True)


def move_mouse(args: dict[str, Any]) -> ToolResult:
    if err := _require_enabled("move_mouse"):
        return err
    try:
        x = _bounded_coordinate(args.get("x"))
        y = _bounded_coordinate(args.get("y"))
        if x is None or y is None:
            return _computer_input_failure("move_mouse", "Valid x and y coordinates are required.")
        _pyautogui().moveTo(x, y, duration=0.2)
        return ToolResult("move_mouse", True, _with_operator_limit(f"Moved mouse to {x},{y}."), _safe_metadata(x=x, y=y, controls_computer=True, computer_control_enabled=_enabled))
    except Exception as exc:
        return _computer_attempt_failure("move_mouse", "move mouse", exc)


def click(args: dict[str, Any]) -> ToolResult:
    if err := _require_enabled("click"):
        return err
    try:
        x = _bounded_coordinate(args.get("x"))
        y = _bounded_coordinate(args.get("y"))
        if x is None or y is None:
            return _computer_input_failure("click", "Valid x and y coordinates are required.")
        button = _clean_text(args.get("button") or "left", limit=16)
        if button not in {"left", "right", "middle"}:
            return _computer_input_failure("click", "Button must be left, right, or middle.", x=x, y=y)
        _pyautogui().click(x, y, button=button)
        return ToolResult("click", True, _with_operator_limit(f"Clicked {x},{y}."), _safe_metadata(x=x, y=y, button=button, controls_computer=True, computer_control_enabled=_enabled))
    except Exception as exc:
        return _computer_attempt_failure("click", "click", exc)


def type_text(args: dict[str, Any]) -> ToolResult:
    if err := _require_enabled("type_text"):
        return err
    text = args.get("text")
    if not isinstance(text, str) or text == "":
        return _computer_input_failure("type_text", "Text is required.")
    if len(text) > MAX_TYPED_TEXT_CHARS:
        return _computer_input_failure(
            "type_text",
            f"Text exceeds the {MAX_TYPED_TEXT_CHARS}-character limit and was not typed.",
            text_chars=len(text),
            max_text_chars=MAX_TYPED_TEXT_CHARS,
        )
    try:
        _pyautogui().write(text, interval=0.02)
        return ToolResult("type_text", True, _with_operator_limit(f"Typed {len(text)} chars."), _safe_metadata(text_chars=len(text), controls_computer=True, computer_control_enabled=_enabled))
    except Exception as exc:
        return _computer_attempt_failure("type_text", "type text", exc, text_chars=len(text))


def screenshot(args: dict[str, Any]) -> ToolResult:
    try:
        filename = _clean_text(args.get("filename") or "jarvis_screen.png", limit=MAX_FILENAME_CHARS)
        path = Path(filename).name
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = Path("~/Desktop").expanduser() / path
        image = _pyautogui().screenshot()
        image.save(path)
        return ToolResult("screenshot", True, _with_operator_limit(f"Screenshot saved to {path}"), _safe_metadata(path=str(path), observes_screen=True, takes_screenshot=True, writes_files=True))
    except Exception as exc:
        return _computer_attempt_failure(
            "screenshot",
            "take screenshot",
            exc,
            observes_screen=True,
            takes_screenshot=True,
            writes_files=True,
        )


def observe_screen(args: dict[str, Any]) -> ToolResult:
    try:
        shot = screenshot({"filename": args.get("filename", "jarvis_observe.png")})
        size = screen_size({})
        mouse = mouse_position({})
        output = "\n".join(
            [
                "Screen observation:",
                f"- screenshot: {shot.output}",
                f"- screen size: {size.output}",
                f"- mouse: {mouse.output}",
            ]
        )
        return ToolResult(
            "observe_screen",
            shot.ok and size.ok and mouse.ok,
            _with_operator_limit(output),
            _safe_metadata(screenshot=shot.metadata.get("path"), screen_size=size.metadata, mouse=mouse.metadata, observes_screen=True, takes_screenshot=True, writes_files=True, controls_computer=True),
        )
    except Exception as exc:
        return _computer_attempt_failure(
            "observe_screen",
            "observe screen",
            exc,
            observes_screen=True,
            takes_screenshot=True,
            writes_files=True,
        )


def verify_screen(args: dict[str, Any]) -> ToolResult:
    expectation = _clean_text(args.get("expectation"), limit=MAX_EXPECTATION_CHARS)
    obs = observe_screen({"filename": args.get("filename", "jarvis_verify.png")})
    if not expectation:
        return ToolResult("verify_screen", obs.ok, _with_operator_limit(obs.output), obs.metadata)
    output = (
        f"{obs.output}\n\n"
        f"Expectation to verify manually or with a future vision model: {expectation}"
    )
    return ToolResult("verify_screen", obs.ok, _with_operator_limit(output), obs.metadata)


def observe_act_verify(args: dict[str, Any]) -> ToolResult:
    """
    Minimal OAV scaffold. It records the before/after state and executes one
    primitive action. A vision model can later replace the manual verification.
    """
    if err := _require_enabled("observe_act_verify"):
        return err
    action = _clean_text(args.get("action"), limit=40)
    before = observe_screen({"filename": "jarvis_oav_before.png"})
    if not before.ok:
        return ToolResult(
            "observe_act_verify",
            False,
            _with_operator_limit(
                "Before observation failed. The requested one-step action was not attempted.\n\n"
                + before.output
            ),
            _safe_metadata(
                before=before.metadata,
                before_observation_ok=False,
                stopped_before_action=True,
                action_attempted=False,
                action=action,
                observes_screen=True,
                takes_screenshot=before.metadata.get("takes_screenshot") is True,
                writes_files=before.metadata.get("writes_files") is True,
                controls_computer=False,
                computer_control_enabled=_enabled,
            ),
        )
    action_args = dict(args)
    located: tuple[int, int] | None = None
    if action in {"click", "move_mouse"} and (action_args.get("x") is None or action_args.get("y") is None):
        located = _locate_target_center(action_args)
        if located is None and (action_args.get("target_image") or action_args.get("image") or action_args.get("template")):
            return ToolResult(
                "observe_act_verify",
                False,
                _with_operator_limit("Could not locate the target image for the requested one-step action."),
                _safe_metadata(
                    before=before.metadata,
                    action=action,
                    target_image=_clean_text(action_args.get("target_image") or action_args.get("image") or action_args.get("template"), limit=MAX_FILENAME_CHARS),
                    target_located=False,
                    observes_screen=True,
                    takes_screenshot=True,
                    writes_files=True,
                    controls_computer=True,
                    computer_control_enabled=_enabled,
                ),
            )
        if located is not None:
            action_args["x"], action_args["y"] = located
    if action == "move_mouse":
        acted = move_mouse(action_args)
    elif action == "click":
        acted = click(action_args)
    elif action == "type_text":
        acted = type_text(action_args)
    else:
        return ToolResult("observe_act_verify", False, _with_operator_limit(f"Unsupported OAV action: {action}"), _safe_metadata(controls_computer=True, observes_screen=True, takes_screenshot=True, writes_files=True))
    after = verify_screen({"filename": "jarvis_oav_after.png", "expectation": args.get("expectation", "")})
    ok = before.ok and acted.ok and after.ok
    output = "\n\n".join(
        [
            "Before:\n" + before.output,
            "Action:\n" + acted.output,
            "After:\n" + after.output,
        ]
    )
    return ToolResult(
        "observe_act_verify",
        ok,
        _with_operator_limit(output),
        _safe_metadata(
            before=before.metadata,
            action=acted.metadata,
            after=after.metadata,
            target_located=located is not None,
            located_x=located[0] if located else None,
            located_y=located[1] if located else None,
            observes_screen=True,
            takes_screenshot=True,
            writes_files=True,
            controls_computer=True,
            computer_control_enabled=_enabled,
        ),
    )
