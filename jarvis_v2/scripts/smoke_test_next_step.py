from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from jarvis_v2.automations.compaction import COMPACTION_JOB_NAME
from jarvis_v2.config import JarvisConfig
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools import next_step as next_step_module
from jarvis_v2.tools.next_step import _metadata_bool, make_next_step_tools
from jarvis_v2.tools.storage import (
    BOOTSTRAP_CHECK_COMMAND,
    BOOTSTRAP_WRITE_COMMAND,
    STORAGE_RECOVERY_CHECK_API,
    STORAGE_RECOVERY_CHECK_COMMAND,
    STORAGE_RECOVERY_PLAN_COMMAND,
)
from jarvis_v2.tools.harness import _agi_gate_summary, _selected_agi_target_readiness
from jarvis_v2.tools.continuity import (
    _AUTONOMY_STEP_CLOSURE_SCORECARD_ITEMS,
    _CHECKPOINT_RECOVERY_EXECUTE_HANDOFF_FALSE_FLAGS,
    _RECOVERY_STEP_APPROVAL_BOUNDARY_FALSE_FLAGS,
    _approval_boundary_rows_for_risky_work,
    _awake_guard_boundary_ready_from_metadata,
    _awake_guard_token_sha256,
    _autonomy_continuation_execution_ready_from_metadata,
    _autonomy_continuation_readiness_scorecard_ready,
    _autonomy_cycle_stage_rows_ready,
    _autonomy_cycle_preflight_scorecard_ready,
    _autonomy_cycle_ledger_ready_from_metadata,
    _autonomy_scorecard_ready,
    _autonomy_expected_post_step_proof_queue,
    _autonomy_post_step_proof_queue_ready,
    _autonomy_step_closure_gate,
    _autonomy_cycle_ledger_token_metadata_ready,
    _autonomy_cycle_ledger_token_sha256,
    _autonomy_resume_gate_ready_from_metadata,
    _autonomy_resume_readiness_scorecard_ready,
    _autonomy_step_closure_ready,
    _autonomy_step_closure_readiness_scorecard_ready,
    _autonomy_step_closure_receipt_token_sha256,
    _step_closure_receipt_metadata_ready,
    _carried_checkpoint_route_boundary_ready,
    _carried_one_step_execution_contract_ready,
    _carried_operator_timebox_review_contract_ready,
    _carried_operator_supersession_token_boundary_ready,
    _carried_local_safe_recovery_execution_token_boundary_ready,
    _carried_step_closure_receipt_boundary_ready,
    _carried_recovery_execution_readiness_token_boundary_ready,
    _carried_recovery_followthrough_token_boundary_ready,
    _checkpoint_recovery_execute_handoff_ready,
    _checkpoint_recovery_execute_handoff_token_sha256,
    _checkpoint_recovery_followthrough_ready,
    _checkpoint_route_boundary_ready,
    _checkpoint_route_token_sha256,
    _continuation_review_token_metadata_ready,
    _fresh_continuation_review_contract_ready,
    _fresh_continuation_review_boundary_token_sha256,
    _fresh_review_boundary_metadata_ready,
    _fresh_review_boundary_token_ready,
    _fresh_review_boundary_token_sha256,
    _local_safe_recovery_execution_token_boundary_ready,
    _local_safe_recovery_execution_token_metadata_ready,
    _local_safe_recovery_execution_token_sha256,
    _metadata_flag_disabled,
    _metadata_flag_ready,
    _next_review_start_command_token_sha256,
    _next_review_start_command_boundary_ready,
    _next_review_start_command_metadata_ready,
    _next_step_approval_boundary_as_prior_proof,
    _next_step_approval_boundary_metadata_ready,
    _next_step_approval_boundary_ready,
    _one_step_execution_contract_ready,
    _one_step_execution_contract_token_sha256,
    _operator_supersession_contract_ready,
    _operator_supersession_token_boundary_ready,
    _operator_supersession_token_sha256,
    _operator_timebox_receipt_sha256,
    _operator_timebox_review_contract_ready,
    _prior_cycle_ledger_token_metadata_ready,
    _recovery_cockpit_readiness_scorecard_ready,
    _recovery_execution_scorecard_metadata_ready,
    _recovery_execution_readiness_token_boundary_ready,
    _recovery_execution_readiness_token_metadata_ready,
    _recovery_followthrough_token_boundary_ready,
    _recovery_followthrough_token_metadata_ready,
    _recovery_followthrough_token_sha256,
    _recovery_execution_readiness_scorecard_ready,
    _recovery_execution_readiness_token_sha256,
    _recovery_step_approval_boundary_ready,
    _risky_next_step_approval_boundary_token_sha256,
    _risky_recovery_step_approval_boundary_token_sha256,
    _scorecard_required_rows_ready,
)


READ_ONLY_FALSE_FLAGS = [
    "calls_model",
    "executes_tools",
    "queues_approval",
    "requires_approval",
    "approves_request",
    "dismisses_request",
    "reads_personal_data",
    "reads_private_data",
    "executes_side_effect",
    "external_side_effect",
    "writes_files",
    "writes_database",
    "writes_memory",
    "writes_notes",
    "edits_files",
    "controls_computer",
    "authorizes_execution",
    "authorizes_completion_claim",
    "approval_granted",
    "speaks",
    "completes_tasks",
]


EXPECTED_APPROVAL_CHAIN = [
    "approval readiness 1",
    "approval packet 1",
    "approval chain proof 1",
    "verification receipt <approved run id from approval chain proof 1>",
]

CONTINUITY_TOOL_NAMES = {
    "operator_timebox_contract",
    "operator_instruction_supersession_packet",
    "checkpoint_recovery_followthrough_packet",
    "checkpoint_recovery_execute_packet",
    "autonomy_resume_gate",
    "autonomy_continuation_execution_packet",
    "autonomy_step_closure_packet",
    "autonomy_cycle_ledger",
}


def assert_sha256(value: object, label: str) -> None:
    text = str(value or "")
    if len(text) != 64 or any(char not in "0123456789abcdefABCDEF" for char in text):
        raise SystemExit(f"{label} should be a sha256 hex digest: {value!r}")


def assert_no_future_authority(metadata: dict, label: str) -> None:
    for key in ("authorizes_execution", "authorizes_completion_claim", "approval_granted"):
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should not grant future authority via {key}: {metadata}")


def assert_next_step_exact_metadata_bool() -> None:
    if _metadata_bool(True) is not True:
        raise SystemExit("next_step exact bool helper should preserve True.")
    if _metadata_bool(False, default=True) is not False:
        raise SystemExit("next_step exact bool helper should preserve False.")
    for value in ("true", "false", "yes", "0", 1, 0, [], ["ready"], None):
        if _metadata_bool(value) is not False:
            raise SystemExit(f"next_step exact bool helper should reject malformed readiness: {value!r}")
    if _metadata_bool("fallback", default=True) is not True:
        raise SystemExit("next_step exact bool helper should honor explicit malformed-value default.")


def assert_continuation_packet_handoff(metadata: dict, label: str, *, objective: str | None = None) -> None:
    handoff = metadata.get("continuation_packet_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed continuation_packet_handoff: {metadata}")
    if metadata.get("continuation_packet_handoff_ready") is not True or handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} should report handoff ready: {metadata}")
    if metadata.get("continuation_packet_ready_for_operator") is not True or handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} should report ready_for_operator: {metadata}")
    if metadata.get("continuation_packet_state_changed") is not False or handoff.get("state_changed") is not False:
        raise SystemExit(f"{label} should report unchanged state: {metadata}")
    if metadata.get("continuation_packet_changed") != [] or handoff.get("changed") != []:
        raise SystemExit(f"{label} should report no changed resources: {metadata}")
    if metadata.get("continuation_packet_content_in_handoff") is not False or handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} should keep content out of handoff: {metadata}")
    if objective and handoff.get("objective") != objective:
        raise SystemExit(f"{label} handoff objective diverged: {metadata}")
    if handoff.get("pending_approvals") != metadata.get("pending_approvals"):
        raise SystemExit(f"{label} pending approval count diverged: {metadata}")
    if handoff.get("open_tasks") != metadata.get("open_tasks") or handoff.get("active_goals") != metadata.get("active_goals"):
        raise SystemExit(f"{label} task/goal counts diverged: {metadata}")
    if handoff.get("scheduler_next_command") != metadata.get("scheduler_next_command"):
        raise SystemExit(f"{label} scheduler command diverged: {metadata}")
    if handoff.get("execution_health_next_commands") != metadata.get("execution_health_next_commands"):
        raise SystemExit(f"{label} execution health queue diverged: {metadata}")
    if handoff.get("execution_health_next_command_count") != metadata.get("execution_health_next_command_count"):
        raise SystemExit(f"{label} execution health queue count diverged: {metadata}")
    if handoff.get("checkpoint_recovery_queue") != metadata.get("checkpoint_recovery_queue"):
        raise SystemExit(f"{label} checkpoint recovery queue diverged: {metadata}")
    if handoff.get("checkpoint_recovery_queue_count") != metadata.get("checkpoint_recovery_queue_count"):
        raise SystemExit(f"{label} checkpoint recovery queue count diverged: {metadata}")
    if metadata.get("continuation_packet_next_commands") != handoff.get("next_commands"):
        raise SystemExit(f"{label} next-command handoff diverged: {metadata}")
    if metadata.get("continuation_packet_next_safe_commands") != handoff.get("next_safe_commands"):
        raise SystemExit(f"{label} next-safe-command handoff diverged: {metadata}")
    if handoff.get("next_safe_commands") != handoff.get("next_commands"):
        raise SystemExit(f"{label} should mirror safe commands to next commands: {metadata}")
    if metadata.get("continuation_packet_next_command_count") != len(handoff.get("next_commands") or []):
        raise SystemExit(f"{label} next-command count diverged: {metadata}")
    if metadata.get("continuation_packet_next_safe_command_count") != len(handoff.get("next_safe_commands") or []):
        raise SystemExit(f"{label} next-safe-command count diverged: {metadata}")
    if not any(str(command).startswith("acceptance gate:") for command in handoff.get("next_commands") or []):
        raise SystemExit(f"{label} missed acceptance gate next command: {metadata}")
    boundaries = handoff.get("boundaries") or {}
    if metadata.get("continuation_packet_boundaries") != boundaries:
        raise SystemExit(f"{label} boundary packet diverged: {metadata}")
    if boundaries.get("read_only") is not True:
        raise SystemExit(f"{label} boundary should report read_only=True: {metadata}")
    for flat_key, boundary_key in [
        ("continuation_packet_authorizes_execution", "authorizes_execution"),
        ("continuation_packet_authorizes_completion_claim", "authorizes_completion_claim"),
        ("continuation_packet_approval_granted", "approval_granted"),
    ]:
        if metadata.get(flat_key) is not False or boundaries.get(boundary_key) is not False:
            raise SystemExit(f"{label} should expose no-authority flat parity for {flat_key}: {metadata}")
    for key in READ_ONLY_FALSE_FLAGS:
        if key in boundaries and boundaries.get(key) is not False:
            raise SystemExit(f"{label} boundary should report {key}=False: {metadata}")
    handoff_text = str(handoff)
    for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
        if forbidden in handoff_text:
            raise SystemExit(f"{label} handoff leaked local path fragment {forbidden!r}: {metadata}")


def assert_build_target_packet_handoff(metadata: dict, label: str, *, objective: str | None = None) -> None:
    handoff = metadata.get("build_target_packet_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed build_target_packet_handoff: {metadata}")
    if metadata.get("build_target_packet_handoff_ready") is not True or handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} should report handoff ready: {metadata}")
    if metadata.get("build_target_packet_ready_for_operator") is not True or handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} should report ready_for_operator: {metadata}")
    if metadata.get("build_target_packet_state_changed") is not False or handoff.get("state_changed") is not False:
        raise SystemExit(f"{label} should report unchanged state: {metadata}")
    if metadata.get("build_target_packet_changed") != [] or handoff.get("changed") != []:
        raise SystemExit(f"{label} should report no changed resources: {metadata}")
    if metadata.get("build_target_packet_content_in_handoff") is not False or handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} should keep content out of handoff: {metadata}")
    if objective and handoff.get("objective") != objective:
        raise SystemExit(f"{label} handoff objective diverged: {metadata}")
    if handoff.get("target_kind") != metadata.get("target_kind"):
        raise SystemExit(f"{label} target kind diverged: {metadata}")
    if handoff.get("target_file_integrity_status") != metadata.get("target_file_integrity_status"):
        raise SystemExit(f"{label} target integrity status diverged: {metadata}")
    if handoff.get("target_files_checked") != metadata.get("target_files_checked"):
        raise SystemExit(f"{label} target file count diverged: {metadata}")
    if handoff.get("target_files_exist") != metadata.get("target_files_exist"):
        raise SystemExit(f"{label} target file existence diverged: {metadata}")
    if handoff.get("missing_target_files") != metadata.get("missing_target_files"):
        raise SystemExit(f"{label} missing target files diverged: {metadata}")
    if handoff.get("likely_files") != metadata.get("likely_file_paths"):
        raise SystemExit(f"{label} likely files diverged: {metadata}")
    if handoff.get("focused_verification_commands") != metadata.get("focused_verification_commands"):
        raise SystemExit(f"{label} focused verification commands diverged: {metadata}")
    if handoff.get("aggregate_verification_command") != metadata.get("aggregate_verification_command"):
        raise SystemExit(f"{label} aggregate verification command diverged: {metadata}")
    if handoff.get("acceptance_gate_command") != metadata.get("acceptance_gate_command"):
        raise SystemExit(f"{label} acceptance gate command diverged: {metadata}")
    if handoff.get("pending_approvals") != metadata.get("pending_approvals"):
        raise SystemExit(f"{label} pending approval count diverged: {metadata}")
    if handoff.get("open_tasks") != metadata.get("open_tasks") or handoff.get("active_goals") != metadata.get("active_goals"):
        raise SystemExit(f"{label} task/goal counts diverged: {metadata}")
    if handoff.get("scheduler_next_command") != metadata.get("scheduler_next_command"):
        raise SystemExit(f"{label} scheduler command diverged: {metadata}")
    if handoff.get("execution_health_next_commands") != metadata.get("execution_health_next_commands"):
        raise SystemExit(f"{label} execution health queue diverged: {metadata}")
    if metadata.get("build_target_packet_next_commands") != handoff.get("next_commands"):
        raise SystemExit(f"{label} next-command handoff diverged: {metadata}")
    if metadata.get("build_target_packet_next_safe_commands") != handoff.get("next_safe_commands"):
        raise SystemExit(f"{label} next-safe-command handoff diverged: {metadata}")
    if handoff.get("next_safe_commands") != handoff.get("next_commands"):
        raise SystemExit(f"{label} should mirror safe commands to next commands: {metadata}")
    if metadata.get("build_target_packet_next_command_count") != len(handoff.get("next_commands") or []):
        raise SystemExit(f"{label} next-command count diverged: {metadata}")
    if metadata.get("build_target_packet_next_safe_command_count") != len(handoff.get("next_safe_commands") or []):
        raise SystemExit(f"{label} next-safe-command count diverged: {metadata}")
    if not any(str(command).startswith("acceptance gate:") for command in handoff.get("next_commands") or []):
        raise SystemExit(f"{label} missed acceptance gate next command: {metadata}")
    boundaries = handoff.get("boundaries") or {}
    if metadata.get("build_target_packet_boundaries") != boundaries:
        raise SystemExit(f"{label} boundary packet diverged: {metadata}")
    if boundaries.get("read_only") is not True:
        raise SystemExit(f"{label} boundary should report read_only=True: {metadata}")
    for flat_key, boundary_key in [
        ("build_target_packet_authorizes_execution", "authorizes_execution"),
        ("build_target_packet_authorizes_completion_claim", "authorizes_completion_claim"),
        ("build_target_packet_approval_granted", "approval_granted"),
    ]:
        if metadata.get(flat_key) is not False or boundaries.get(boundary_key) is not False:
            raise SystemExit(f"{label} should expose no-authority flat parity for {flat_key}: {metadata}")
    for key in READ_ONLY_FALSE_FLAGS:
        if key in boundaries and boundaries.get(key) is not False:
            raise SystemExit(f"{label} boundary should report {key}=False: {metadata}")
    handoff_text = str(handoff)
    for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
        if forbidden in handoff_text:
            raise SystemExit(f"{label} handoff leaked local path fragment {forbidden!r}: {metadata}")


def assert_work_queue_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("work_queue_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed work_queue_handoff: {metadata}")
    if metadata.get("work_queue_handoff_ready") is not True or handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} should report handoff ready: {metadata}")
    if metadata.get("work_queue_ready_for_operator") is not True or handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} should report ready_for_operator: {metadata}")
    if metadata.get("work_queue_state_changed") is not False or handoff.get("state_changed") is not False:
        raise SystemExit(f"{label} should report unchanged state: {metadata}")
    if metadata.get("work_queue_changed") != [] or handoff.get("changed") != []:
        raise SystemExit(f"{label} should report no changed resources: {metadata}")
    if metadata.get("work_queue_content_in_handoff") is not False or handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} should keep content out of handoff: {metadata}")
    for key in [
        "pending_approvals",
        "open_tasks",
        "active_goals",
        "active_decisions",
        "enabled_jobs",
        "state_snapshot_jobs",
        "enabled_state_snapshot_jobs",
        "disabled_state_snapshot_jobs",
        "scheduler_next_command",
        "failed_action_runs",
        "approval_held_action_runs",
        "execution_health_approval_held_action_runs",
        "execution_health_review_required",
        "execution_health_next_command",
        "execution_health_next_commands",
        "execution_health_next_command_count",
        "execution_health_blocker_categories",
        "execution_health_blocker_count",
        "execution_health_verification_coverage",
        "doctor_next_audit_command",
        "doctor_completion_claim_state",
        "doctor_completion_claim_ready",
        "doctor_completion_blockers",
        "doctor_completion_blocker_count",
        "doctor_audit_readability_review_required",
        "doctor_audit_readability_review_commands",
        "doctor_audit_readability_review_command_count",
        "doctor_audit_readability_review_next_command",
        "doctor_unreadable_recent_tool_run_rows",
        "doctor_recovery_closure_state",
        "doctor_recovery_closure_next_required_command",
        "doctor_recovery_closure_next_proof_command",
        "doctor_execution_learning_state",
        "doctor_execution_learning_next_required_command",
        "doctor_execution_learning_next_proof_command",
        "doctor_agi_next_gate",
        "doctor_agi_next_selection_source",
        "doctor_agi_next_selection_reason",
        "doctor_agi_next_canonical_selector_command",
        "doctor_agi_next_deliberate_focus_override",
        "doctor_agi_next_target_title",
        "doctor_agi_next_build_command",
        "doctor_agi_next_real_execution_gap_count",
        "doctor_agi_next_real_execution_gaps_by_gate",
        "doctor_agi_next_selected_real_execution_gap",
        "approval_handoff_pending_count",
        "approval_handoff_first_id",
        "approval_handoff_next_command",
        "approval_handoff_readiness_command",
        "approval_handoff_last_look_command",
        "approval_handoff_proof_command",
        "approval_handoff_proof_chain_commands",
        "limit",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} handoff {key} diverged from metadata: {metadata}")
    for flat_key, nested_key in [
        ("work_queue_start_kind", "start_kind"),
        ("work_queue_start_command", "start_command"),
        ("work_queue_start_label", "start_label"),
        ("work_queue_order", "queue_order"),
        ("work_queue_next_commands", "next_commands"),
        ("work_queue_next_safe_commands", "next_safe_commands"),
    ]:
        if metadata.get(flat_key) != handoff.get(nested_key):
            raise SystemExit(f"{label} {flat_key} diverged from nested handoff: {metadata}")
    if metadata.get("work_queue_next_command_count") != len(handoff.get("next_commands") or []):
        raise SystemExit(f"{label} next-command count diverged: {metadata}")
    if handoff.get("next_safe_commands") != handoff.get("next_commands"):
        raise SystemExit(f"{label} safe next commands should mirror read-only next commands: {metadata}")
    if metadata.get("work_queue_next_safe_command_count") != len(handoff.get("next_safe_commands") or []):
        raise SystemExit(f"{label} next-safe-command count diverged: {metadata}")
    for command in ["handoff brief", "catch me up", "safety status", "focus brief", "safe next actions", "save handoff brief"]:
        if command not in handoff.get("next_commands", []):
            raise SystemExit(f"{label} missed next command {command!r}: {metadata}")
    if metadata.get("pending_approvals") and not any(str(command).startswith("approval readiness") for command in handoff.get("next_commands") or []):
        raise SystemExit(f"{label} missed approval readiness next command: {metadata}")
    boundaries = handoff.get("boundaries") or {}
    if metadata.get("work_queue_boundaries") != boundaries:
        raise SystemExit(f"{label} boundary packet diverged: {metadata}")
    if boundaries.get("read_only") is not True:
        raise SystemExit(f"{label} boundary should report read_only=True: {metadata}")
    for flat_key, boundary_key in [
        ("work_queue_authorizes_execution", "authorizes_execution"),
        ("work_queue_authorizes_completion_claim", "authorizes_completion_claim"),
        ("work_queue_approval_granted", "approval_granted"),
    ]:
        if metadata.get(flat_key) is not False or boundaries.get(boundary_key) is not False:
            raise SystemExit(f"{label} should expose no-authority flat parity for {flat_key}: {metadata}")
    for key in READ_ONLY_FALSE_FLAGS:
        if key in boundaries and boundaries.get(key) is not False:
            raise SystemExit(f"{label} boundary should report {key}=False: {metadata}")
    handoff_text = json.dumps(handoff, sort_keys=True)
    for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
        if forbidden in handoff_text:
            raise SystemExit(f"{label} handoff leaked local path fragment {forbidden!r}: {metadata}")


def assert_safe_next_actions_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("safe_next_actions_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed safe_next_actions_handoff: {metadata}")
    if metadata.get("safe_next_actions_handoff_ready") is not True or handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} should report handoff ready: {metadata}")
    if metadata.get("safe_next_actions_ready_for_operator") is not True or handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} should report ready_for_operator: {metadata}")
    if metadata.get("safe_next_actions_state_changed") is not False or handoff.get("state_changed") is not False:
        raise SystemExit(f"{label} should report unchanged state: {metadata}")
    if metadata.get("safe_next_actions_changed") != [] or handoff.get("changed") != []:
        raise SystemExit(f"{label} should report no changed resources: {metadata}")
    if metadata.get("safe_next_actions_content_in_handoff") is not False or handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} should keep content out of handoff: {metadata}")
    for key in [
        "pending_approvals",
        "open_tasks",
        "active_goals",
        "active_decisions",
        "active_preferences",
        "enabled_jobs",
        "state_snapshot_jobs",
        "enabled_state_snapshot_jobs",
        "disabled_state_snapshot_jobs",
        "scheduler_next_command",
        "failed_action_runs",
        "approval_held_action_runs",
        "execution_health_approval_held_action_runs",
        "execution_health_review_required",
        "execution_health_next_command",
        "execution_health_next_commands",
        "execution_health_next_command_count",
        "execution_health_blocker_categories",
        "execution_health_blocker_count",
        "execution_health_verification_coverage",
        "doctor_next_audit_command",
        "doctor_completion_claim_state",
        "doctor_completion_claim_ready",
        "doctor_completion_blockers",
        "doctor_completion_blocker_count",
        "doctor_audit_readability_review_required",
        "doctor_audit_readability_review_commands",
        "doctor_audit_readability_review_command_count",
        "doctor_audit_readability_review_next_command",
        "doctor_unreadable_recent_tool_run_rows",
        "doctor_recovery_closure_state",
        "doctor_recovery_closure_next_required_command",
        "doctor_recovery_closure_next_proof_command",
        "doctor_execution_learning_state",
        "doctor_execution_learning_next_required_command",
        "doctor_execution_learning_next_proof_command",
        "doctor_agi_next_gate",
        "doctor_agi_next_selection_source",
        "doctor_agi_next_selection_reason",
        "doctor_agi_next_canonical_selector_command",
        "doctor_agi_next_deliberate_focus_override",
        "doctor_agi_next_target_title",
        "doctor_agi_next_build_command",
        "doctor_agi_next_real_execution_gap_count",
        "doctor_agi_next_real_execution_gaps_by_gate",
        "doctor_agi_next_selected_real_execution_gap",
        "approval_handoff_pending_count",
        "approval_handoff_first_id",
        "approval_handoff_next_command",
        "approval_handoff_readiness_command",
        "approval_handoff_last_look_command",
        "approval_handoff_proof_command",
        "approval_handoff_proof_chain_commands",
        "limit",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} handoff {key} diverged from metadata: {metadata}")
    for flat_key, nested_key in [
        ("safe_next_actions_start_kind", "start_kind"),
        ("safe_next_actions_start_command", "start_command"),
        ("safe_next_actions_start_label", "start_label"),
        ("safe_next_actions_review_order", "review_order"),
        ("safe_next_actions_next_commands", "next_commands"),
        ("safe_next_actions_next_safe_commands", "next_safe_commands"),
    ]:
        if metadata.get(flat_key) != handoff.get(nested_key):
            raise SystemExit(f"{label} {flat_key} diverged from nested handoff: {metadata}")
    if metadata.get("safe_next_actions_next_command_count") != len(handoff.get("next_commands") or []):
        raise SystemExit(f"{label} next-command count diverged: {metadata}")
    if handoff.get("next_safe_commands") != handoff.get("next_commands"):
        raise SystemExit(f"{label} safe next commands should mirror read-only next commands: {metadata}")
    if metadata.get("safe_next_actions_next_safe_command_count") != len(handoff.get("next_safe_commands") or []):
        raise SystemExit(f"{label} next-safe-command count diverged: {metadata}")
    for command in ["catch me up", "safety status", "work queue", "focus brief", "build target packet", "save handoff brief"]:
        if command not in handoff.get("next_commands", []):
            raise SystemExit(f"{label} missed next command {command!r}: {metadata}")
    if metadata.get("pending_approvals") and not any(str(command).startswith("approval readiness") for command in handoff.get("next_commands") or []):
        raise SystemExit(f"{label} missed approval readiness next command: {metadata}")
    if metadata.get("execution_health_review_required") and metadata.get("execution_health_next_command") not in handoff.get("next_commands", []):
        raise SystemExit(f"{label} missed execution health next command: {metadata}")
    boundaries = handoff.get("boundaries") or {}
    if metadata.get("safe_next_actions_boundaries") != boundaries:
        raise SystemExit(f"{label} boundary packet diverged: {metadata}")
    if boundaries.get("read_only") is not True:
        raise SystemExit(f"{label} boundary should report read_only=True: {metadata}")
    for flat_key, boundary_key in [
        ("safe_next_actions_authorizes_execution", "authorizes_execution"),
        ("safe_next_actions_authorizes_completion_claim", "authorizes_completion_claim"),
        ("safe_next_actions_approval_granted", "approval_granted"),
    ]:
        if metadata.get(flat_key) is not False or boundaries.get(boundary_key) is not False:
            raise SystemExit(f"{label} should expose no-authority flat parity for {flat_key}: {metadata}")
    for key in READ_ONLY_FALSE_FLAGS:
        if key in boundaries and boundaries.get(key) is not False:
            raise SystemExit(f"{label} boundary should report {key}=False: {metadata}")
    handoff_text = json.dumps(handoff, sort_keys=True)
    for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
        if forbidden in handoff_text:
            raise SystemExit(f"{label} handoff leaked local path fragment {forbidden!r}: {metadata}")


def assert_next_action_packet_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("next_action_packet_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed next_action_packet_handoff: {metadata}")
    if metadata.get("next_action_packet_handoff_ready") is not True or handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} should report handoff ready: {metadata}")
    if metadata.get("next_action_packet_ready_for_operator") is not True or handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} should report ready_for_operator: {metadata}")
    if metadata.get("next_action_packet_state_changed") is not False or handoff.get("state_changed") is not False:
        raise SystemExit(f"{label} should report unchanged state: {metadata}")
    if metadata.get("next_action_packet_changed") != [] or handoff.get("changed") != []:
        raise SystemExit(f"{label} should report no changed resources: {metadata}")
    if metadata.get("next_action_packet_content_in_handoff") is not False or handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} should keep content out of handoff: {metadata}")
    for key in [
        "action_kind",
        "pending_approvals",
        "open_tasks",
        "active_goals",
        "active_decisions",
        "active_preferences",
        "enabled_jobs",
        "state_snapshot_jobs",
        "enabled_state_snapshot_jobs",
        "disabled_state_snapshot_jobs",
        "scheduler_next_command",
        "failed_action_runs",
        "approval_held_action_runs",
        "execution_health_approval_held_action_runs",
        "execution_health_review_required",
        "execution_health_next_command",
        "execution_health_next_commands",
        "execution_health_next_command_count",
        "execution_health_blocker_categories",
        "execution_health_blocker_count",
        "execution_health_verification_coverage",
        "doctor_next_audit_command",
        "doctor_completion_claim_state",
        "doctor_completion_claim_ready",
        "doctor_completion_blockers",
        "doctor_completion_blocker_count",
        "doctor_audit_readability_review_required",
        "doctor_audit_readability_review_commands",
        "doctor_audit_readability_review_command_count",
        "doctor_audit_readability_review_next_command",
        "doctor_unreadable_recent_tool_run_rows",
        "doctor_recovery_closure_state",
        "doctor_recovery_closure_next_required_command",
        "doctor_recovery_closure_next_proof_command",
        "doctor_execution_learning_state",
        "doctor_execution_learning_next_required_command",
        "doctor_execution_learning_next_proof_command",
        "doctor_agi_next_gate",
        "doctor_agi_next_selection_source",
        "doctor_agi_next_selection_reason",
        "doctor_agi_next_canonical_selector_command",
        "doctor_agi_next_deliberate_focus_override",
        "doctor_agi_next_target_title",
        "doctor_agi_next_build_command",
        "approval_handoff_pending_count",
        "approval_handoff_first_id",
        "approval_handoff_next_command",
        "approval_handoff_readiness_command",
        "approval_handoff_last_look_command",
        "approval_handoff_proof_command",
        "approval_handoff_proof_chain_commands",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} handoff {key} diverged from metadata: {metadata}")
    for flat_key, nested_key in [
        ("next_action_packet_action_kind", "action_kind"),
        ("next_action_packet_command", "command"),
        ("next_action_packet_review_order", "review_order"),
        ("next_action_packet_next_commands", "next_commands"),
        ("next_action_packet_next_safe_commands", "next_safe_commands"),
    ]:
        if metadata.get(flat_key) != handoff.get(nested_key):
            raise SystemExit(f"{label} {flat_key} diverged from nested handoff: {metadata}")
    if metadata.get("next_action_packet_next_command_count") != len(handoff.get("next_commands") or []):
        raise SystemExit(f"{label} next-command count diverged: {metadata}")
    if handoff.get("next_safe_commands") != handoff.get("next_commands"):
        raise SystemExit(f"{label} safe next commands should mirror read-only next commands: {metadata}")
    if metadata.get("next_action_packet_next_safe_command_count") != len(handoff.get("next_safe_commands") or []):
        raise SystemExit(f"{label} next-safe-command count diverged: {metadata}")
    if metadata.get("action_command") != handoff.get("command"):
        raise SystemExit(f"{label} action command should mirror handoff command: {metadata}")
    for command in ["catch me up", "safety status", "safe next actions", "work queue", "priority stack", "focus brief", "build target packet"]:
        if command not in handoff.get("next_commands", []):
            raise SystemExit(f"{label} missed next command {command!r}: {metadata}")
    if metadata.get("pending_approvals") and not any(str(command).startswith("approval readiness") for command in handoff.get("next_commands") or []):
        raise SystemExit(f"{label} missed approval readiness next command: {metadata}")
    boundaries = handoff.get("boundaries") or {}
    if metadata.get("next_action_packet_boundaries") != boundaries:
        raise SystemExit(f"{label} boundary packet diverged: {metadata}")
    if boundaries.get("read_only") is not True:
        raise SystemExit(f"{label} boundary should report read_only=True: {metadata}")
    for flat_key, boundary_key in [
        ("next_action_packet_authorizes_execution", "authorizes_execution"),
        ("next_action_packet_authorizes_completion_claim", "authorizes_completion_claim"),
        ("next_action_packet_approval_granted", "approval_granted"),
    ]:
        if metadata.get(flat_key) is not False or boundaries.get(boundary_key) is not False:
            raise SystemExit(f"{label} should expose no-authority flat parity for {flat_key}: {metadata}")
    for key in READ_ONLY_FALSE_FLAGS:
        if key in boundaries and boundaries.get(key) is not False:
            raise SystemExit(f"{label} boundary should report {key}=False: {metadata}")
    handoff_text = json.dumps(handoff, sort_keys=True)
    for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
        if forbidden in handoff_text:
            raise SystemExit(f"{label} handoff leaked local path fragment {forbidden!r}: {metadata}")


def assert_priority_stack_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("priority_stack_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed priority_stack_handoff: {metadata}")
    if metadata.get("priority_stack_handoff_ready") is not True or handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} should report handoff ready: {metadata}")
    if metadata.get("priority_stack_ready_for_operator") is not True or handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} should report ready_for_operator: {metadata}")
    if metadata.get("priority_stack_state_changed") is not False or handoff.get("state_changed") is not False:
        raise SystemExit(f"{label} should report unchanged state: {metadata}")
    if metadata.get("priority_stack_changed") != [] or handoff.get("changed") != []:
        raise SystemExit(f"{label} should report no changed resources: {metadata}")
    if metadata.get("priority_stack_content_in_handoff") is not False or handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} should keep content out of handoff: {metadata}")
    for key in [
        "top_kind",
        "pending_approvals",
        "open_tasks",
        "active_goals",
        "active_decisions",
        "active_preferences",
        "enabled_jobs",
        "state_snapshot_jobs",
        "enabled_state_snapshot_jobs",
        "disabled_state_snapshot_jobs",
        "scheduler_next_command",
        "failed_action_runs",
        "approval_held_action_runs",
        "execution_health_approval_held_action_runs",
        "execution_health_review_required",
        "execution_health_next_command",
        "execution_health_next_commands",
        "execution_health_next_command_count",
        "execution_health_blocker_categories",
        "execution_health_blocker_count",
        "execution_health_verification_coverage",
        "doctor_next_audit_command",
        "doctor_completion_claim_state",
        "doctor_completion_claim_ready",
        "doctor_completion_blockers",
        "doctor_completion_blocker_count",
        "doctor_audit_readability_review_required",
        "doctor_audit_readability_review_commands",
        "doctor_audit_readability_review_command_count",
        "doctor_audit_readability_review_next_command",
        "doctor_unreadable_recent_tool_run_rows",
        "doctor_recovery_closure_state",
        "doctor_recovery_closure_next_required_command",
        "doctor_recovery_closure_next_proof_command",
        "doctor_execution_learning_state",
        "doctor_execution_learning_next_required_command",
        "doctor_execution_learning_next_proof_command",
        "doctor_agi_next_gate",
        "doctor_agi_next_selection_source",
        "doctor_agi_next_selection_reason",
        "doctor_agi_next_canonical_selector_command",
        "doctor_agi_next_deliberate_focus_override",
        "doctor_agi_next_target_title",
        "doctor_agi_next_build_command",
        "approval_handoff_pending_count",
        "approval_handoff_first_id",
        "approval_handoff_next_command",
        "approval_handoff_readiness_command",
        "approval_handoff_last_look_command",
        "approval_handoff_proof_command",
        "approval_handoff_proof_chain_commands",
        "limit",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} handoff {key} diverged from metadata: {metadata}")
    for flat_key, nested_key in [
        ("priority_stack_top_kind", "top_kind"),
        ("priority_stack_top_command", "top_command"),
        ("priority_stack_rows", "stack_rows"),
        ("priority_stack_count", "stack_count"),
        ("priority_stack_ranking_order", "ranking_order"),
        ("priority_stack_next_commands", "next_commands"),
        ("priority_stack_next_safe_commands", "next_safe_commands"),
    ]:
        if metadata.get(flat_key) != handoff.get(nested_key):
            raise SystemExit(f"{label} {flat_key} diverged from nested handoff: {metadata}")
    if metadata.get("items") != handoff.get("stack_count"):
        raise SystemExit(f"{label} item count diverged from handoff: {metadata}")
    if metadata.get("priority_stack_next_command_count") != len(handoff.get("next_commands") or []):
        raise SystemExit(f"{label} next-command count diverged: {metadata}")
    if handoff.get("next_safe_commands") != handoff.get("next_commands"):
        raise SystemExit(f"{label} safe next commands should mirror read-only next commands: {metadata}")
    if metadata.get("priority_stack_next_safe_command_count") != len(handoff.get("next_safe_commands") or []):
        raise SystemExit(f"{label} next-safe-command count diverged: {metadata}")
    if not handoff.get("stack_rows") or not handoff.get("top_command"):
        raise SystemExit(f"{label} should expose stack rows and top command: {metadata}")
    for command in ["catch me up", "safety status", "safe next actions", "work queue", "focus brief", "build target packet"]:
        if command not in handoff.get("next_commands", []):
            raise SystemExit(f"{label} missed next command {command!r}: {metadata}")
    if metadata.get("pending_approvals") and not any(str(command).startswith("approval readiness") for command in handoff.get("next_commands") or []):
        raise SystemExit(f"{label} missed approval readiness next command: {metadata}")
    boundaries = handoff.get("boundaries") or {}
    if metadata.get("priority_stack_boundaries") != boundaries:
        raise SystemExit(f"{label} boundary packet diverged: {metadata}")
    if boundaries.get("read_only") is not True:
        raise SystemExit(f"{label} boundary should report read_only=True: {metadata}")
    for flat_key, boundary_key in [
        ("priority_stack_authorizes_execution", "authorizes_execution"),
        ("priority_stack_authorizes_completion_claim", "authorizes_completion_claim"),
        ("priority_stack_approval_granted", "approval_granted"),
    ]:
        if metadata.get(flat_key) is not False or boundaries.get(boundary_key) is not False:
            raise SystemExit(f"{label} should expose no-authority flat parity for {flat_key}: {metadata}")
    for key in READ_ONLY_FALSE_FLAGS:
        if key in boundaries and boundaries.get(key) is not False:
            raise SystemExit(f"{label} boundary should report {key}=False: {metadata}")
    handoff_text = json.dumps(handoff, sort_keys=True)
    for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
        if forbidden in handoff_text:
            raise SystemExit(f"{label} handoff leaked local path fragment {forbidden!r}: {metadata}")


def assert_vault_relative_save_receipt(result, *, expected_path: Path, expected_display: str, label: str) -> None:
    metadata = result.tool_results[0].metadata
    response = result.response
    first_line = response.split("\n", 1)[0]
    if str(expected_path) not in str(metadata.get("path") or ""):
        raise SystemExit(f"{label} metadata should preserve the exact saved path: {metadata}")
    if metadata.get("path_display") != expected_display:
        raise SystemExit(f"{label} metadata should include vault-relative path_display={expected_display!r}: {metadata}")
    if expected_display not in first_line:
        raise SystemExit(f"{label} receipt should include vault-relative path: {first_line!r}")
    if str(expected_path) in first_line:
        raise SystemExit(f"{label} receipt should not expose the absolute local path: {first_line!r}")


def assert_next_step_approval_boundary_token(metadata: dict, label: str, *, prefix: str) -> None:
    token_key = f"{prefix}_approval_boundary_token_sha256"
    present_key = f"{prefix}_approval_boundary_token_present"
    queue_key = f"{prefix}_approval_proof_queue"
    rows_key = f"{prefix}_approval_boundary_rows"
    token = str(metadata.get(token_key) or "")
    assert_sha256(token, f"{label} approval boundary token")
    if metadata.get(present_key) is not True:
        raise SystemExit(f"{label} missed approval boundary token presence: {metadata}")
    rows = metadata.get(rows_key) or []
    for row in rows:
        for key in (
            "authorizes_action_now",
            "authorizes_risky_work",
            "authorizes_unreviewed_followthrough",
            "authorizes_approval",
            "authorizes_model_call",
            "authorizes_tool_execution",
            "authorizes_personal_data_read",
            "authorizes_external_side_effect",
            "authorizes_timebox_reuse",
            "reusable_for_next_review",
            "reusable_for_recovery_review",
        ):
            if row.get(key) is not False:
                raise SystemExit(f"{label} approval boundary row should report {key}=False: {row}")
    risk_signals: list[str] = []
    for row in rows:
        if row.get("item") == "risk_classification":
            risk_signals = [str(signal) for signal in (row.get("risk_signals") or [])]
            break
    recomputed = _risky_next_step_approval_boundary_token_sha256(
        proposed_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
        proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
        risk_signals=risk_signals,
        approval_proof_queue=[str(command) for command in (metadata.get(queue_key) or [])],
        approval_boundary_rows=rows,
    )
    if recomputed != token:
        raise SystemExit(f"{label} approval boundary token should be reproducible: {metadata}")
    proof_queue = [str(command) for command in (metadata.get(queue_key) or [])]
    if _next_step_approval_boundary_ready(
        token_sha256=token,
        proposed_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
        proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
        approval_proof_queue=proof_queue,
        approval_boundary_rows=rows,
    ) is not bool(metadata.get(f"{prefix}_approval_boundary_ready")):
        raise SystemExit(f"{label} approval boundary ready should use strict production validator: {metadata}")
    if metadata.get(f"{prefix}_approval_boundary_ready") is True:
        tampered_ready_rows = [dict(row) for row in rows]
        tampered_ready_rows[0]["status"] = "held"
        if _next_step_approval_boundary_ready(
            token_sha256=token,
            proposed_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
            proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
            approval_proof_queue=proof_queue,
            approval_boundary_rows=tampered_ready_rows,
        ):
            raise SystemExit(f"{label} approval boundary validator should reject status tampering: {metadata}")
    if not _next_step_approval_boundary_as_prior_proof(
        token_sha256=token,
        proposed_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
        proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
        approval_proof_queue=proof_queue,
        approval_boundary_rows=rows,
    ):
        raise SystemExit(f"{label} approval boundary should be strict prior-proof ready: {metadata}")
    followup_authority_key = (
        f"{prefix}_approval_boundary_authorizes_unreviewed_followthrough"
        if prefix == "carried_next_step"
        else f"{prefix}_approval_boundary_authorizes_followup_without_closure"
    )
    if not _next_step_approval_boundary_metadata_ready(
        metadata,
        prefix=prefix,
        followup_authority_key=followup_authority_key,
    ):
        raise SystemExit(f"{label} approval boundary failed metadata mirror validator: {metadata}")
    for key in [
        f"{prefix}_approval_boundary_authorizes_action_now",
        f"{prefix}_approval_boundary_authorizes_risky_work",
        f"{prefix}_approval_boundary_authorizes_approval",
        f"{prefix}_approval_boundary_authorizes_model_call",
        f"{prefix}_approval_boundary_authorizes_tool_execution",
        f"{prefix}_approval_boundary_authorizes_personal_data_read",
        f"{prefix}_approval_boundary_authorizes_external_side_effect",
        f"{prefix}_approval_boundary_authorizes_timebox_reuse",
        followup_authority_key,
        f"{prefix}_approval_boundary_reusable_for_next_review",
        f"{prefix}_approval_boundary_reusable_for_recovery_review",
    ]:
        if key not in metadata:
            raise SystemExit(f"{label} approval boundary metadata missed required non-authority mirror {key}: {metadata}")
        missing_metadata = dict(metadata)
        missing_metadata.pop(key, None)
        if _next_step_approval_boundary_metadata_ready(
            missing_metadata,
            prefix=prefix,
            followup_authority_key=followup_authority_key,
        ):
            raise SystemExit(
                f"{label} approval boundary metadata validator accepted missing required non-authority mirror {key}: "
                f"{missing_metadata}"
            )
    for key, value in [
        (f"{prefix}_approval_proof_queue_count", 999),
        (f"{prefix}_approval_boundary_row_count", 999),
        (f"{prefix}_approval_boundary_token_present", False),
        (f"{prefix}_approval_boundary_ready", not bool(metadata.get(f"{prefix}_approval_boundary_ready"))),
        (f"{prefix}_approval_boundary_as_prior_proof", False),
        (f"{prefix}_approval_required_before_review", not bool(proof_queue)),
        (f"{prefix}_next_approval_proof_command", "stale approval proof command"),
        (f"{prefix}_approval_boundary_authorizes_action_now", True),
        (f"{prefix}_approval_boundary_authorizes_approval", True),
        (f"{prefix}_approval_boundary_authorizes_model_call", True),
        (f"{prefix}_approval_boundary_authorizes_tool_execution", True),
        (f"{prefix}_approval_boundary_authorizes_personal_data_read", True),
        (f"{prefix}_approval_boundary_authorizes_external_side_effect", True),
        (f"{prefix}_approval_boundary_authorizes_timebox_reuse", True),
        (followup_authority_key, True),
        (f"{prefix}_approval_boundary_reusable_for_next_review", True),
        (f"{prefix}_approval_boundary_reusable_for_recovery_review", True),
    ]:
        if key not in metadata:
            continue
        tampered_metadata = dict(metadata)
        tampered_metadata[key] = value
        if _next_step_approval_boundary_metadata_ready(
            tampered_metadata,
            prefix=prefix,
            followup_authority_key=followup_authority_key,
        ):
            raise SystemExit(f"{label} approval boundary metadata validator accepted tampered {key}: {tampered_metadata}")
    tampered = _risky_next_step_approval_boundary_token_sha256(
        proposed_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
        proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
        risk_signals=risk_signals,
        approval_proof_queue=[str(command) for command in (metadata.get(queue_key) or [])],
        approval_boundary_rows=rows,
        authorizes_action_now=True,
    )
    if tampered == token:
        raise SystemExit(f"{label} approval boundary token should bind action authority: {metadata}")
    tampered_reference = _risky_next_step_approval_boundary_token_sha256(
        proposed_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
        proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
        risk_signals=risk_signals,
        approval_proof_queue=[str(command) for command in (metadata.get(queue_key) or [])],
        approval_boundary_rows=rows,
        approval_reference="different approval reference",
    )
    if tampered_reference == token:
        raise SystemExit(f"{label} approval boundary token should bind approval reference: {metadata}")
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["authorizes_tool_execution"] = True
    tampered_row_token = _risky_next_step_approval_boundary_token_sha256(
        proposed_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
        proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
        risk_signals=risk_signals,
        approval_proof_queue=[str(command) for command in (metadata.get(queue_key) or [])],
        approval_boundary_rows=tampered_rows,
    )
    if tampered_row_token == token:
        raise SystemExit(f"{label} approval boundary token should bind non-authorizing row flags: {metadata}")


def assert_recovery_step_approval_boundary_token(metadata: dict, label: str) -> None:
    token = str(metadata.get("recovery_step_approval_boundary_token_sha256") or "")
    assert_sha256(token, f"{label} approval boundary token")
    if metadata.get("recovery_step_approval_boundary_token_present") is not True:
        raise SystemExit(f"{label} missed approval boundary token presence: {metadata}")
    for flag in _RECOVERY_STEP_APPROVAL_BOUNDARY_FALSE_FLAGS:
        if metadata.get(flag) is not False:
            raise SystemExit(f"{label} approval boundary flag should report {flag}=False: {metadata}")
    rows = metadata.get("recovery_step_approval_boundary_rows") or []
    for row in rows:
        for key in (
            "authorizes_action_now",
            "authorizes_risky_work",
            "authorizes_unreviewed_followthrough",
            "authorizes_approval",
            "authorizes_model_call",
            "authorizes_tool_execution",
            "authorizes_personal_data_read",
            "authorizes_external_side_effect",
            "authorizes_timebox_reuse",
            "reusable_for_next_review",
            "reusable_for_recovery_review",
        ):
            if row.get(key) is not False:
                raise SystemExit(f"{label} approval boundary row should report {key}=False: {row}")
    risk_signals: list[str] = []
    for row in rows:
        if row.get("item") == "risk_classification":
            risk_signals = [str(signal) for signal in (row.get("risk_signals") or [])]
            break
    recomputed = _risky_recovery_step_approval_boundary_token_sha256(
        recovery_step_sha256=str(metadata.get("recovery_step_sha256") or ""),
        recovery_verification_sha256=str(metadata.get("recovery_verification_sha256") or ""),
        risk_signals=risk_signals,
        approval_proof_queue=[str(command) for command in (metadata.get("recovery_step_approval_proof_queue") or [])],
        approval_boundary_rows=rows,
        approval_reference=str(metadata.get("approval_reference") or "") if metadata.get("approval_reference_provided") else "",
    )
    if recomputed != token:
        raise SystemExit(f"{label} approval boundary token should be reproducible: {metadata}")
    proof_queue = [str(command) for command in (metadata.get("recovery_step_approval_proof_queue") or [])]
    approval_reference = str(metadata.get("approval_reference") or "") if metadata.get("approval_reference_provided") else ""
    if _recovery_step_approval_boundary_ready(
        token_sha256=token,
        recovery_step_sha256=str(metadata.get("recovery_step_sha256") or ""),
        recovery_verification_sha256=str(metadata.get("recovery_verification_sha256") or ""),
        approval_proof_queue=proof_queue,
        approval_boundary_rows=rows,
        approval_reference=approval_reference,
    ) is not bool(metadata.get("recovery_step_approval_boundary_ready")):
        raise SystemExit(f"{label} approval boundary ready should use strict production validator: {metadata}")
    if metadata.get("recovery_step_approval_boundary_as_prior_proof") is True and not _recovery_step_approval_boundary_ready(
        token_sha256=token,
        recovery_step_sha256=str(metadata.get("recovery_step_sha256") or ""),
        recovery_verification_sha256=str(metadata.get("recovery_verification_sha256") or ""),
        approval_proof_queue=proof_queue,
        approval_boundary_rows=rows,
        approval_reference=approval_reference,
        require_prior_proof=True,
    ):
        raise SystemExit(f"{label} approval boundary prior proof should use strict production validator: {metadata}")
    tampered_ready_rows = [dict(row) for row in rows]
    tampered_ready_rows[0]["status"] = "not_required" if tampered_ready_rows[0].get("status") == "held" else "held"
    if _recovery_step_approval_boundary_ready(
        token_sha256=token,
        recovery_step_sha256=str(metadata.get("recovery_step_sha256") or ""),
        recovery_verification_sha256=str(metadata.get("recovery_verification_sha256") or ""),
        approval_proof_queue=proof_queue,
        approval_boundary_rows=tampered_ready_rows,
        approval_reference=approval_reference,
    ):
        raise SystemExit(f"{label} approval boundary validator should reject status tampering: {metadata}")
    tampered = _risky_recovery_step_approval_boundary_token_sha256(
        recovery_step_sha256=str(metadata.get("recovery_step_sha256") or ""),
        recovery_verification_sha256=str(metadata.get("recovery_verification_sha256") or ""),
        risk_signals=risk_signals,
        approval_proof_queue=[str(command) for command in (metadata.get("recovery_step_approval_proof_queue") or [])],
        approval_boundary_rows=rows,
        approval_reference=str(metadata.get("approval_reference") or "") if metadata.get("approval_reference_provided") else "",
        authorizes_action_now=True,
    )
    if tampered == token:
        raise SystemExit(f"{label} approval boundary token should bind action authority: {metadata}")
    if metadata.get("approval_reference_provided"):
        tampered_reference = _risky_recovery_step_approval_boundary_token_sha256(
            recovery_step_sha256=str(metadata.get("recovery_step_sha256") or ""),
            recovery_verification_sha256=str(metadata.get("recovery_verification_sha256") or ""),
            risk_signals=risk_signals,
            approval_proof_queue=[str(command) for command in (metadata.get("recovery_step_approval_proof_queue") or [])],
            approval_boundary_rows=rows,
            approval_reference="different approval reference",
        )
        if tampered_reference == token:
            raise SystemExit(f"{label} approval boundary token should bind approval reference: {metadata}")
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["authorizes_personal_data_read"] = True
    tampered_row_token = _risky_recovery_step_approval_boundary_token_sha256(
        recovery_step_sha256=str(metadata.get("recovery_step_sha256") or ""),
        recovery_verification_sha256=str(metadata.get("recovery_verification_sha256") or ""),
        risk_signals=risk_signals,
        approval_proof_queue=[str(command) for command in (metadata.get("recovery_step_approval_proof_queue") or [])],
        approval_boundary_rows=tampered_rows,
        approval_reference=str(metadata.get("approval_reference") or "") if metadata.get("approval_reference_provided") else "",
    )
    if tampered_row_token == token:
        raise SystemExit(f"{label} approval boundary token should bind non-authorizing row flags: {metadata}")


def assert_one_step_execution_contract_token(metadata: dict, label: str, *, carried: bool = False) -> None:
    token = str(metadata.get("one_step_execution_contract_token_sha256") or "")
    assert_sha256(token, f"{label} one-step execution contract token")
    if metadata.get("one_step_execution_contract_token_present") is not True:
        raise SystemExit(f"{label} missed one-step execution contract token presence: {metadata}")
    if carried and metadata.get("one_step_execution_contract_token_as_prior_proof") is not True:
        raise SystemExit(f"{label} should carry one-step execution contract token as prior proof: {metadata}")
    for key in [
        "one_step_execution_contract_token_authorizes_action_now",
        "one_step_execution_contract_token_authorizes_risky_work",
        "one_step_execution_contract_token_authorizes_unreviewed_followthrough",
        "one_step_execution_contract_token_authorizes_batching",
        "one_step_execution_contract_token_reusable_for_next_step",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should report {key}=False: {metadata}")
    if metadata.get("next_step_requires_new_one_step_execution_contract_token") is not True:
        raise SystemExit(f"{label} missed fresh one-step contract token requirement: {metadata}")
    rows = metadata.get("one_step_execution_contract_rows") or []
    if metadata.get("one_step_execution_contract_row_count") != len(rows) or len(rows) != 6:
        raise SystemExit(f"{label} one-step contract row count diverged: {metadata}")
    expected_statuses = {
        "ready_resume_gate": "ready",
        "exact_next_step": "ready",
        "post_step_verification_target": "ready",
        "stop_condition": "ready",
        "risky_work_approval_boundary": "ready",
        "fresh_post_step_closure": "fresh_required_after_step",
    }
    if {str(row.get("item") or "") for row in rows} != set(expected_statuses):
        raise SystemExit(f"{label} one-step contract items diverged: {metadata}")
    for row in rows:
        item = str(row.get("item") or "")
        if row.get("status") != expected_statuses[item] or row.get("required") is not True:
            raise SystemExit(f"{label} one-step contract row shape diverged: {metadata}")
    if metadata.get("one_step_execution_contract_ready") is not True:
        raise SystemExit(f"{label} missed strict one-step execution contract readiness: {metadata}")
    if metadata.get("one_step_execution_contract_all_local_safe_step_limited") is not True:
        raise SystemExit(f"{label} missed one-local-safe-step-limited contract summary: {metadata}")
    if any(row.get("authorizes_only_one_local_safe_step") is not True for row in rows):
        raise SystemExit(f"{label} one-step contract rows should be limited to one local-safe step: {metadata}")
    timebox_review_contract_rows = metadata.get("timebox_review_contract_rows") or []
    post_step_proof_queue = [
        str(command)
        for command in (metadata.get("post_step_proof_queue") or metadata.get("continuation_post_step_proof_queue") or [])
    ]
    expected_post_step_proof_queue = _autonomy_expected_post_step_proof_queue(
        proposed_next_step=str(metadata.get("proposed_next_step") or ""),
        proposed_verification=str(metadata.get("proposed_verification") or ""),
    )
    if post_step_proof_queue != expected_post_step_proof_queue:
        raise SystemExit(f"{label} post-step proof queue diverged from expected closure ladder: {metadata}")
    if not _autonomy_post_step_proof_queue_ready(metadata):
        raise SystemExit(f"{label} production post-step proof queue validator rejected exact queue: {metadata}")
    risk_signals = [str(signal) for signal in (metadata.get("proposed_next_step_risk_signals") or [])]
    recomputed = _one_step_execution_contract_token_sha256(
        objective=str(metadata.get("objective") or ""),
        continuation_state=str(metadata.get("continuation_state") or ""),
        resume_gate_state=str(metadata.get("resume_gate_state") or ""),
        timebox_receipt_sha256=str(metadata.get("timebox_receipt_sha256") or ""),
        awake_guard_token_sha256=str(metadata.get("awake_guard_token_sha256") or ""),
        supersession_token_sha256=str(metadata.get("supersession_token_sha256") or ""),
        checkpoint_route_token_sha256=str(metadata.get("checkpoint_route_token_sha256") or ""),
        recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
        local_safe_recovery_execution_token_sha256=str(metadata.get("local_safe_recovery_execution_token_sha256") or ""),
        prior_cycle_ledger_token_sha256=str(metadata.get("prior_cycle_ledger_token_sha256") or ""),
        continuation_review_token_sha256=str(metadata.get("continuation_review_token_sha256") or ""),
        proposed_next_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
        proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
        risk_signals=risk_signals,
        one_step_execution_contract_rows=rows,
        timebox_review_contract_rows=timebox_review_contract_rows,
        post_step_proof_queue=post_step_proof_queue,
    )
    if recomputed != token:
        raise SystemExit(f"{label} one-step execution contract token should be reproducible: {metadata}")
    if not _one_step_execution_contract_ready(
        objective=str(metadata.get("objective") or ""),
        continuation_state=str(metadata.get("continuation_state") or ""),
        resume_gate_state=str(metadata.get("resume_gate_state") or ""),
        continuation_ready=metadata.get("one_step_execution_contract_ready"),
        one_step_execution_contract_rows=rows,
        one_step_execution_contract_token_sha256=token,
        timebox_receipt_sha256=str(metadata.get("timebox_receipt_sha256") or ""),
        awake_guard_token_sha256=str(metadata.get("awake_guard_token_sha256") or ""),
        supersession_token_sha256=str(metadata.get("supersession_token_sha256") or ""),
        checkpoint_route_token_sha256=str(metadata.get("checkpoint_route_token_sha256") or ""),
        recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
        local_safe_recovery_execution_token_sha256=str(metadata.get("local_safe_recovery_execution_token_sha256") or ""),
        prior_cycle_ledger_token_sha256=str(metadata.get("prior_cycle_ledger_token_sha256") or ""),
        continuation_review_token_sha256=str(metadata.get("continuation_review_token_sha256") or ""),
        proposed_next_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
        proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
        risk_signals=risk_signals,
        timebox_review_contract_ready=metadata.get("timebox_review_contract_ready"),
        timebox_review_contract_rows=timebox_review_contract_rows,
        post_step_proof_queue=post_step_proof_queue,
    ):
        raise SystemExit(f"{label} production one-step execution contract readiness should accept the exact token: {metadata}")
    malformed_timebox_ready_metadata = json.loads(json.dumps(metadata))
    malformed_timebox_ready_metadata["timebox_review_contract_ready"] = "false"
    if _one_step_execution_contract_ready(
        objective=str(malformed_timebox_ready_metadata.get("objective") or ""),
        continuation_state=str(malformed_timebox_ready_metadata.get("continuation_state") or ""),
        resume_gate_state=str(malformed_timebox_ready_metadata.get("resume_gate_state") or ""),
        continuation_ready=malformed_timebox_ready_metadata.get("one_step_execution_contract_ready"),
        one_step_execution_contract_rows=list(malformed_timebox_ready_metadata.get("one_step_execution_contract_rows") or []),
        one_step_execution_contract_token_sha256=str(malformed_timebox_ready_metadata.get("one_step_execution_contract_token_sha256") or ""),
        timebox_receipt_sha256=str(malformed_timebox_ready_metadata.get("timebox_receipt_sha256") or ""),
        awake_guard_token_sha256=str(malformed_timebox_ready_metadata.get("awake_guard_token_sha256") or ""),
        supersession_token_sha256=str(malformed_timebox_ready_metadata.get("supersession_token_sha256") or ""),
        checkpoint_route_token_sha256=str(malformed_timebox_ready_metadata.get("checkpoint_route_token_sha256") or ""),
        recovery_followthrough_token_sha256=str(malformed_timebox_ready_metadata.get("recovery_followthrough_token_sha256") or ""),
        local_safe_recovery_execution_token_sha256=str(
            malformed_timebox_ready_metadata.get("local_safe_recovery_execution_token_sha256") or ""
        ),
        prior_cycle_ledger_token_sha256=str(malformed_timebox_ready_metadata.get("prior_cycle_ledger_token_sha256") or ""),
        continuation_review_token_sha256=str(malformed_timebox_ready_metadata.get("continuation_review_token_sha256") or ""),
        proposed_next_step_sha256=str(malformed_timebox_ready_metadata.get("proposed_next_step_sha256") or ""),
        proposed_verification_sha256=str(malformed_timebox_ready_metadata.get("proposed_verification_sha256") or ""),
        risk_signals=[
            str(signal) for signal in (malformed_timebox_ready_metadata.get("proposed_next_step_risk_signals") or [])
        ],
        timebox_review_contract_ready=malformed_timebox_ready_metadata.get("timebox_review_contract_ready"),
        timebox_review_contract_rows=list(malformed_timebox_ready_metadata.get("timebox_review_contract_rows") or []),
        post_step_proof_queue=[
            str(command)
            for command in (
                malformed_timebox_ready_metadata.get("post_step_proof_queue")
                or malformed_timebox_ready_metadata.get("continuation_post_step_proof_queue")
                or []
            )
        ],
    ):
        raise SystemExit(f"{label} one-step contract accepted malformed timebox readiness mirror: {malformed_timebox_ready_metadata}")
    if carried:
        if not _carried_one_step_execution_contract_ready(metadata):
            raise SystemExit(f"{label} carried one-step execution contract readiness should accept exact mirrors: {metadata}")
        for key, value in [
            ("one_step_execution_contract_token_present", False),
            ("one_step_execution_contract_token_as_prior_proof", False),
            ("one_step_execution_contract_row_count", 999),
            ("one_step_execution_contract_ready", False),
            ("one_step_execution_contract_binds_awake_guard", "false"),
            ("one_step_execution_contract_binds_operator_supersession", "false"),
            ("one_step_execution_contract_binds_timebox_review_contract", "false"),
            ("one_step_execution_contract_all_local_safe_step_limited", "false"),
            ("one_step_execution_contract_all_non_reusable", "false"),
            ("one_step_execution_contract_all_risky_work_gated", "false"),
            ("one_step_execution_contract_requires_fresh_closure", "false"),
            ("timebox_review_contract_ready", "false"),
            ("next_step_requires_new_one_step_execution_contract_token", False),
            ("one_step_execution_contract_token_authorizes_action_now", True),
            ("one_step_execution_contract_token_reusable_for_next_step", True),
        ]:
            tampered_metadata = json.loads(json.dumps(metadata))
            tampered_metadata[key] = value
            if _carried_one_step_execution_contract_ready(tampered_metadata):
                raise SystemExit(f"{label} carried one-step contract accepted stale mirror {key}: {tampered_metadata}")
        stale_queue_metadata = json.loads(json.dumps(metadata))
        stale_queue = list(expected_post_step_proof_queue)
        stale_queue[-1] = "completion claim gate: stale"
        stale_queue_metadata["continuation_post_step_proof_queue"] = stale_queue
        stale_queue_metadata["continuation_post_step_proof_queue_count"] = len(stale_queue)
        stale_queue_metadata["continuation_post_step_next_proof_command"] = stale_queue[0]
        if _autonomy_post_step_proof_queue_ready(stale_queue_metadata):
            raise SystemExit(f"{label} post-step proof queue validator accepted stale carried queue: {stale_queue_metadata}")
        if _carried_one_step_execution_contract_ready(stale_queue_metadata):
            raise SystemExit(f"{label} carried one-step contract accepted stale post-step proof queue: {stale_queue_metadata}")
    forged_token = "f" * 64 if token != "f" * 64 else "e" * 64
    if _one_step_execution_contract_ready(
        objective=str(metadata.get("objective") or ""),
        continuation_state=str(metadata.get("continuation_state") or ""),
        resume_gate_state=str(metadata.get("resume_gate_state") or ""),
        continuation_ready=metadata.get("one_step_execution_contract_ready"),
        one_step_execution_contract_rows=rows,
        one_step_execution_contract_token_sha256=forged_token,
        timebox_receipt_sha256=str(metadata.get("timebox_receipt_sha256") or ""),
        awake_guard_token_sha256=str(metadata.get("awake_guard_token_sha256") or ""),
        supersession_token_sha256=str(metadata.get("supersession_token_sha256") or ""),
        checkpoint_route_token_sha256=str(metadata.get("checkpoint_route_token_sha256") or ""),
        recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
        local_safe_recovery_execution_token_sha256=str(metadata.get("local_safe_recovery_execution_token_sha256") or ""),
        prior_cycle_ledger_token_sha256=str(metadata.get("prior_cycle_ledger_token_sha256") or ""),
        continuation_review_token_sha256=str(metadata.get("continuation_review_token_sha256") or ""),
        proposed_next_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
        proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
        risk_signals=risk_signals,
        timebox_review_contract_ready=metadata.get("timebox_review_contract_ready"),
        timebox_review_contract_rows=timebox_review_contract_rows,
        post_step_proof_queue=post_step_proof_queue,
    ):
        raise SystemExit(f"{label} production one-step execution contract readiness should reject forged tokens: {metadata}")
    tampered = _one_step_execution_contract_token_sha256(
        objective=str(metadata.get("objective") or ""),
        continuation_state=str(metadata.get("continuation_state") or ""),
        resume_gate_state=str(metadata.get("resume_gate_state") or ""),
        timebox_receipt_sha256=str(metadata.get("timebox_receipt_sha256") or ""),
        awake_guard_token_sha256=str(metadata.get("awake_guard_token_sha256") or ""),
        supersession_token_sha256=str(metadata.get("supersession_token_sha256") or ""),
        checkpoint_route_token_sha256=str(metadata.get("checkpoint_route_token_sha256") or ""),
        recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
        local_safe_recovery_execution_token_sha256=str(metadata.get("local_safe_recovery_execution_token_sha256") or ""),
        prior_cycle_ledger_token_sha256=str(metadata.get("prior_cycle_ledger_token_sha256") or ""),
        continuation_review_token_sha256=str(metadata.get("continuation_review_token_sha256") or ""),
        proposed_next_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
        proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
        risk_signals=[str(signal) for signal in (metadata.get("proposed_next_step_risk_signals") or [])],
        one_step_execution_contract_rows=rows,
        timebox_review_contract_rows=metadata.get("timebox_review_contract_rows") or [],
        post_step_proof_queue=[str(command) for command in (metadata.get("post_step_proof_queue") or metadata.get("continuation_post_step_proof_queue") or [])],
        authorizes_action_now=True,
    )
    if tampered == token:
        raise SystemExit(f"{label} one-step execution contract token should bind action authority: {metadata}")
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["authorizes_only_one_local_safe_step"] = False
    tampered_row_token = _one_step_execution_contract_token_sha256(
        objective=str(metadata.get("objective") or ""),
        continuation_state=str(metadata.get("continuation_state") or ""),
        resume_gate_state=str(metadata.get("resume_gate_state") or ""),
        timebox_receipt_sha256=str(metadata.get("timebox_receipt_sha256") or ""),
        awake_guard_token_sha256=str(metadata.get("awake_guard_token_sha256") or ""),
        supersession_token_sha256=str(metadata.get("supersession_token_sha256") or ""),
        checkpoint_route_token_sha256=str(metadata.get("checkpoint_route_token_sha256") or ""),
        recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
        local_safe_recovery_execution_token_sha256=str(metadata.get("local_safe_recovery_execution_token_sha256") or ""),
        prior_cycle_ledger_token_sha256=str(metadata.get("prior_cycle_ledger_token_sha256") or ""),
        continuation_review_token_sha256=str(metadata.get("continuation_review_token_sha256") or ""),
        proposed_next_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
        proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
        risk_signals=[str(signal) for signal in (metadata.get("proposed_next_step_risk_signals") or [])],
        one_step_execution_contract_rows=tampered_rows,
        timebox_review_contract_rows=metadata.get("timebox_review_contract_rows") or [],
        post_step_proof_queue=[str(command) for command in (metadata.get("post_step_proof_queue") or metadata.get("continuation_post_step_proof_queue") or [])],
    )
    if tampered_row_token == token:
        raise SystemExit(f"{label} one-step execution contract token should bind local-safe-step row limits: {metadata}")
    tampered_awake_token = _one_step_execution_contract_token_sha256(
        objective=str(metadata.get("objective") or ""),
        continuation_state=str(metadata.get("continuation_state") or ""),
        resume_gate_state=str(metadata.get("resume_gate_state") or ""),
        timebox_receipt_sha256=str(metadata.get("timebox_receipt_sha256") or ""),
        awake_guard_token_sha256="0" * 64,
        supersession_token_sha256=str(metadata.get("supersession_token_sha256") or ""),
        checkpoint_route_token_sha256=str(metadata.get("checkpoint_route_token_sha256") or ""),
        recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
        local_safe_recovery_execution_token_sha256=str(metadata.get("local_safe_recovery_execution_token_sha256") or ""),
        prior_cycle_ledger_token_sha256=str(metadata.get("prior_cycle_ledger_token_sha256") or ""),
        continuation_review_token_sha256=str(metadata.get("continuation_review_token_sha256") or ""),
        proposed_next_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
        proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
        risk_signals=[str(signal) for signal in (metadata.get("proposed_next_step_risk_signals") or [])],
        one_step_execution_contract_rows=rows,
        timebox_review_contract_rows=metadata.get("timebox_review_contract_rows") or [],
        post_step_proof_queue=[str(command) for command in (metadata.get("post_step_proof_queue") or metadata.get("continuation_post_step_proof_queue") or [])],
    )
    if tampered_awake_token == token:
        raise SystemExit(f"{label} one-step execution contract token should bind awake guard proof: {metadata}")
    tampered_supersession_token = _one_step_execution_contract_token_sha256(
        objective=str(metadata.get("objective") or ""),
        continuation_state=str(metadata.get("continuation_state") or ""),
        resume_gate_state=str(metadata.get("resume_gate_state") or ""),
        timebox_receipt_sha256=str(metadata.get("timebox_receipt_sha256") or ""),
        awake_guard_token_sha256=str(metadata.get("awake_guard_token_sha256") or ""),
        supersession_token_sha256="0" * 64,
        checkpoint_route_token_sha256=str(metadata.get("checkpoint_route_token_sha256") or ""),
        recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
        local_safe_recovery_execution_token_sha256=str(metadata.get("local_safe_recovery_execution_token_sha256") or ""),
        prior_cycle_ledger_token_sha256=str(metadata.get("prior_cycle_ledger_token_sha256") or ""),
        continuation_review_token_sha256=str(metadata.get("continuation_review_token_sha256") or ""),
        proposed_next_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
        proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
        risk_signals=[str(signal) for signal in (metadata.get("proposed_next_step_risk_signals") or [])],
        one_step_execution_contract_rows=rows,
        timebox_review_contract_rows=metadata.get("timebox_review_contract_rows") or [],
        post_step_proof_queue=[str(command) for command in (metadata.get("post_step_proof_queue") or metadata.get("continuation_post_step_proof_queue") or [])],
    )
    if tampered_supersession_token == token:
        raise SystemExit(f"{label} one-step execution contract token should bind operator supersession proof: {metadata}")
    tampered_timebox_rows = [dict(row) for row in (metadata.get("timebox_review_contract_rows") or [])]
    if tampered_timebox_rows:
        tampered_timebox_rows[0]["authorizes_timebox_reuse"] = True
        tampered_timebox_token = _one_step_execution_contract_token_sha256(
            objective=str(metadata.get("objective") or ""),
            continuation_state=str(metadata.get("continuation_state") or ""),
            resume_gate_state=str(metadata.get("resume_gate_state") or ""),
            timebox_receipt_sha256=str(metadata.get("timebox_receipt_sha256") or ""),
            awake_guard_token_sha256=str(metadata.get("awake_guard_token_sha256") or ""),
            supersession_token_sha256=str(metadata.get("supersession_token_sha256") or ""),
            checkpoint_route_token_sha256=str(metadata.get("checkpoint_route_token_sha256") or ""),
            recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
            local_safe_recovery_execution_token_sha256=str(metadata.get("local_safe_recovery_execution_token_sha256") or ""),
            prior_cycle_ledger_token_sha256=str(metadata.get("prior_cycle_ledger_token_sha256") or ""),
            continuation_review_token_sha256=str(metadata.get("continuation_review_token_sha256") or ""),
            proposed_next_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
            proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
            risk_signals=[str(signal) for signal in (metadata.get("proposed_next_step_risk_signals") or [])],
            one_step_execution_contract_rows=rows,
            timebox_review_contract_rows=tampered_timebox_rows,
            post_step_proof_queue=[str(command) for command in (metadata.get("post_step_proof_queue") or metadata.get("continuation_post_step_proof_queue") or [])],
        )
        if tampered_timebox_token == token:
            raise SystemExit(f"{label} one-step execution contract token should bind timebox review row authority: {metadata}")
    if metadata.get("one_step_execution_contract_binds_awake_guard") is not True:
        raise SystemExit(f"{label} missed one-step awake-guard binding flag: {metadata}")
    if metadata.get("one_step_execution_contract_binds_operator_supersession") is not True:
        raise SystemExit(f"{label} missed one-step operator-supersession binding flag: {metadata}")
    if metadata.get("one_step_execution_contract_binds_timebox_review_contract") is not True:
        raise SystemExit(f"{label} missed one-step timebox-review binding flag: {metadata}")


def file_sha256(path: object) -> str:
    return hashlib.sha256(Path(str(path)).read_bytes()).hexdigest()


def assert_timebox_review_contract(metadata: dict, label: str) -> None:
    rows = metadata.get("timebox_review_contract_rows") or []
    expected_items = {
        "explicit_stop_window",
        "newer_instruction_check",
        "resume_gate_review",
        "one_step_continuation_limit",
        "post_step_closure_required",
        "next_timebox_freshness",
    }
    if metadata.get("timebox_receipt_present") is not True:
        raise SystemExit(f"{label} missed timebox receipt presence: {metadata}")
    assert_sha256(metadata.get("timebox_receipt_sha256"), f"{label} timebox receipt")
    if metadata.get("timebox_review_contract_ready") is not True:
        raise SystemExit(f"{label} missed ready timebox review contract: {metadata}")
    if _operator_timebox_review_contract_ready(
        list(rows),
        timebox_state=str(metadata.get("timebox_state") or ""),
        stop_at=str(metadata.get("stop_at") or ""),
        current_time=str(metadata.get("current_time") or ""),
        timezone_label=str(metadata.get("timezone") or ""),
    ) is not bool(metadata.get("timebox_review_contract_ready")):
        raise SystemExit(f"{label} timebox contract should use strict production readiness: {metadata}")
    if not _carried_operator_timebox_review_contract_ready(metadata):
        raise SystemExit(f"{label} carried timebox mirror contract should be production-ready: {metadata}")
    if metadata.get("timebox_review_contract_row_count") != 6 or len(rows) != 6:
        raise SystemExit(f"{label} missed timebox review contract rows: {metadata}")
    if {row.get("item") for row in rows} != expected_items:
        raise SystemExit(f"{label} timebox review contract items diverged: {metadata}")
    for key in [
        "timebox_authorizes_execution",
        "timebox_authorizes_local_safe_step",
        "timebox_authorizes_risky_work",
        "timebox_authorizes_approval",
        "timebox_authorizes_timebox_reuse",
        "timebox_authorizes_model_call",
        "timebox_authorizes_tool_execution",
        "timebox_authorizes_personal_data_read",
        "timebox_authorizes_external_side_effect",
        "timebox_reusable_for_next_step",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should report {key}=False: {metadata}")
    if metadata.get("next_step_requires_fresh_timebox") is not True:
        raise SystemExit(f"{label} missed fresh timebox requirement: {metadata}")
    if any(
        row.get("fresh_required") is not True
        or row.get("reusable_for_next_step") is not False
        or row.get("authorizes_execution") is not False
        or row.get("authorizes_local_safe_step") is not False
        or row.get("authorizes_risky_work") is not False
        or row.get("authorizes_approval") is not False
        or row.get("authorizes_timebox_reuse") is not False
        or row.get("authorizes_model_call") is not False
        or row.get("authorizes_tool_execution") is not False
        or row.get("authorizes_personal_data_read") is not False
        or row.get("authorizes_external_side_effect") is not False
        for row in rows
    ):
        raise SystemExit(f"{label} timebox review contract must stay fresh and non-authorizing: {metadata}")
    for key, value in [
        ("timebox_state", "UNSUPPORTED_TIMEBOX_STATE"),
        ("can_continue_now", not bool(metadata.get("can_continue_now"))),
        ("should_stop_now", not bool(metadata.get("should_stop_now"))),
        ("missing_timebox_proof_count", 999),
        ("timebox_receipt_present", False),
        ("timebox_receipt_sha256", "0" * 64),
        ("timebox_review_contract_row_count", 999),
        ("timebox_review_contract_ready", False),
        ("next_step_requires_fresh_timebox", False),
        ("timebox_authorizes_execution", True),
        ("timebox_authorizes_timebox_reuse", True),
        ("timebox_reusable_for_next_step", True),
    ]:
        tampered = dict(metadata)
        tampered[key] = value
        if _carried_operator_timebox_review_contract_ready(tampered):
            raise SystemExit(f"{label} carried timebox contract accepted stale mirror {key}: {tampered}")
    tampered_timebox_rows = [dict(row) for row in rows]
    if tampered_timebox_rows:
        tampered_timebox_rows[0]["state"] = "tampered"
        if _operator_timebox_review_contract_ready(
            tampered_timebox_rows,
            timebox_state=str(metadata.get("timebox_state") or ""),
            stop_at=str(metadata.get("stop_at") or ""),
            current_time=str(metadata.get("current_time") or ""),
            timezone_label=str(metadata.get("timezone") or ""),
        ):
            raise SystemExit(f"{label} timebox contract should reject row-shape tampering: {metadata}")
    if all(metadata.get(key) is not None for key in ["objective", "stop_at", "current_time", "timebox_state"]):
        recomputed = _operator_timebox_receipt_sha256(
            objective=str(metadata.get("objective") or ""),
            stop_at=str(metadata.get("stop_at") or ""),
            current_time=str(metadata.get("current_time") or ""),
            timezone_label=str(metadata.get("timezone") or "Asia/Seoul"),
            timebox_state=str(metadata.get("timebox_state") or ""),
        )
        if recomputed != metadata.get("timebox_receipt_sha256"):
            raise SystemExit(f"{label} timebox receipt should be reproducible: {metadata}")
        tampered = _operator_timebox_receipt_sha256(
            objective=str(metadata.get("objective") or ""),
            stop_at=str(metadata.get("stop_at") or ""),
            current_time=str(metadata.get("current_time") or ""),
            timezone_label=str(metadata.get("timezone") or "Asia/Seoul"),
            timebox_state=str(metadata.get("timebox_state") or ""),
            authorizes_local_safe_step=True,
        )
        if tampered == metadata.get("timebox_receipt_sha256"):
            raise SystemExit(f"{label} timebox receipt should bind local-safe-step authority: {metadata}")
        tampered_metadata = dict(metadata)
        tampered_metadata["timebox_receipt_sha256"] = tampered
        if _carried_operator_timebox_review_contract_ready(tampered_metadata):
            raise SystemExit(f"{label} carried timebox contract accepted an authority-bound stale receipt: {tampered_metadata}")


def assert_awake_guard_boundary(metadata: dict, label: str, *, expected_requested: bool) -> None:
    rows = metadata.get("awake_guard_boundary_rows") or []
    if metadata.get("awake_guard_requested") is not expected_requested:
        raise SystemExit(f"{label} awake guard request detection mismatch: {metadata}")
    if metadata.get("awake_guard_token_present") is not True:
        raise SystemExit(f"{label} missed awake guard token presence: {metadata}")
    assert_sha256(metadata.get("awake_guard_token_sha256"), f"{label} awake guard token")
    if metadata.get("awake_guard_boundary_row_count") != 3 or len(rows) != 3:
        raise SystemExit(f"{label} missed awake guard boundary rows: {metadata}")
    expected_statuses = {
        "awake_request_detected": "requested" if expected_requested else "not_requested",
        "os_power_management_boundary": "approval_gated_if_shell_or_system_change",
        "next_awake_guard_review": "fresh_review_required",
    }
    if {row.get("item") for row in rows} != set(expected_statuses):
        raise SystemExit(f"{label} awake guard boundary items diverged: {metadata}")
    if {row.get("item"): row.get("status") for row in rows} != expected_statuses:
        raise SystemExit(f"{label} awake guard boundary statuses diverged: {metadata}")
    request_row = next(row for row in rows if row.get("item") == "awake_request_detected")
    if (
        metadata.get("stop_at") is not None
        and metadata.get("current_time") is not None
        and (request_row.get("stop_at") != metadata.get("stop_at") or request_row.get("current_time") != metadata.get("current_time"))
    ):
        raise SystemExit(f"{label} awake guard request row should bind stop/current time: {metadata}")
    for key in [
        "awake_guard_authorizes_os_wake_lock",
        "awake_guard_authorizes_shell_execution",
        "awake_guard_authorizes_computer_control",
        "awake_guard_authorizes_approval",
        "awake_guard_authorizes_model_call",
        "awake_guard_authorizes_tool_execution",
        "awake_guard_authorizes_personal_data_read",
        "awake_guard_authorizes_external_side_effect",
        "awake_guard_reusable_for_next_timebox",
        "awake_guard_caffeinate_command_authorized",
        "awake_guard_keep_awake_command_authorized",
        "awake_guard_authorizes_unattended_execution",
        "awake_guard_authorizes_continuation_window",
        "awake_guard_reusable_as_execution_permission",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should report {key}=False: {metadata}")
    for key in [
        "awake_guard_requires_separate_operator_request",
        "awake_guard_requires_separate_shell_approval",
        "awake_guard_os_wake_lock_boundary_ready",
    ]:
        if metadata.get(key) is not True:
            raise SystemExit(f"{label} should report {key}=True: {metadata}")
    if metadata.get("next_awake_guard_requires_fresh_review") is not True:
        raise SystemExit(f"{label} missed fresh awake guard review requirement: {metadata}")
    validator_objective = str(metadata.get("objective") or metadata.get("request") or "")
    if validator_objective:
        if not _awake_guard_boundary_ready_from_metadata(metadata, default_objective=validator_objective):
            raise SystemExit(f"{label} awake guard metadata failed production boundary validator: {metadata}")
        for key, value in [
            ("awake_guard_requested", "true"),
            ("awake_guard_token_present", False),
            ("awake_guard_boundary_row_count", 999),
            ("awake_guard_os_wake_lock_boundary_ready", False),
            ("next_awake_guard_requires_fresh_review", False),
            ("awake_guard_authorizes_os_wake_lock", True),
            ("awake_guard_authorizes_shell_execution", True),
            ("awake_guard_reusable_for_next_timebox", True),
            ("awake_guard_requires_separate_operator_request", False),
            ("awake_guard_requires_separate_shell_approval", False),
        ]:
            tampered_metadata = dict(metadata)
            tampered_metadata[key] = value
            if _awake_guard_boundary_ready_from_metadata(tampered_metadata, default_objective=validator_objective):
                raise SystemExit(f"{label} awake guard validator accepted stale mirror {key}: {tampered_metadata}")
    if any(
        row.get("authorizes_os_wake_lock") is not False
        or row.get("authorizes_shell_execution") is not False
        or row.get("authorizes_computer_control") is not False
        or row.get("authorizes_approval") is not False
        or row.get("authorizes_model_call") is not False
        or row.get("authorizes_tool_execution") is not False
        or row.get("authorizes_personal_data_read") is not False
        or row.get("authorizes_external_side_effect") is not False
        or row.get("reusable_for_next_timebox") is not False
        for row in rows
    ):
        raise SystemExit(f"{label} awake guard rows should stay non-authorizing: {metadata}")
    if all(metadata.get(key) is not None for key in ["objective", "stop_at", "current_time"]):
        recomputed = _awake_guard_token_sha256(
            objective=str(metadata.get("objective") or ""),
            stop_at=str(metadata.get("stop_at") or ""),
            current_time=str(metadata.get("current_time") or ""),
            timezone_label=str(metadata.get("timezone") or "Asia/Seoul"),
            requested=expected_requested,
        )
        if recomputed != metadata.get("awake_guard_token_sha256"):
            raise SystemExit(f"{label} awake guard token should be reproducible: {metadata}")
        tampered = _awake_guard_token_sha256(
            objective=str(metadata.get("objective") or ""),
            stop_at=str(metadata.get("stop_at") or ""),
            current_time=str(metadata.get("current_time") or ""),
            timezone_label=str(metadata.get("timezone") or "Asia/Seoul"),
            requested=expected_requested,
            authorizes_os_wake_lock=True,
        )
        if tampered == metadata.get("awake_guard_token_sha256"):
            raise SystemExit(f"{label} awake guard token should bind OS wake-lock authority: {metadata}")


def assert_operator_supersession_token_boundary(metadata: dict, label: str, *, expected_source: str) -> None:
    token_sha256 = metadata.get("supersession_token_sha256")
    assert_sha256(token_sha256, f"{label} operator supersession token")
    if metadata.get("supersession_token_present") is not True:
        raise SystemExit(f"{label} missed operator supersession token presence: {metadata}")
    rows = metadata.get("supersession_token_boundary_rows") or []
    if metadata.get("supersession_token_boundary_row_count") != 3 or len(rows) != 3:
        raise SystemExit(f"{label} missed operator supersession token boundary rows: {metadata}")
    if metadata.get("supersession_token_boundary_ready") is not True:
        raise SystemExit(f"{label} missed operator supersession boundary ready flag: {metadata}")
    if not _operator_supersession_token_boundary_ready(
        str(token_sha256),
        rows,
        expected_source=expected_source,
    ):
        raise SystemExit(f"{label} production operator supersession boundary validator rejected rows: {metadata}")
    if not _carried_operator_supersession_token_boundary_ready(
        metadata,
        expected_source=expected_source,
    ):
        raise SystemExit(f"{label} carried operator supersession validator rejected valid metadata: {metadata}")
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["source"] = "tampered_source"
    if _operator_supersession_token_boundary_ready(
        str(token_sha256),
        tampered_rows,
        expected_source=expected_source,
    ):
        raise SystemExit(f"{label} production operator supersession boundary validator accepted tampered source")
    for key, value in [
        ("supersession_token_present", False),
        ("supersession_token_boundary_row_count", 999),
        ("supersession_token_boundary_ready", False),
        ("next_supersession_requires_fresh_latest_instruction_review", False),
        ("supersession_token_authorizes_local_safe_step", True),
        ("supersession_token_authorizes_timebox_override", True),
        ("supersession_token_reusable_for_next_review", True),
    ]:
        tampered_metadata = dict(metadata)
        tampered_metadata[key] = value
        if _carried_operator_supersession_token_boundary_ready(
            tampered_metadata,
            expected_source=expected_source,
        ):
            raise SystemExit(f"{label} carried operator supersession validator accepted stale mirror {key}: {tampered_metadata}")
    expected_rows = {
        "operator_supersession_token": "present",
        "newest_instruction_scope": "proof_only_for_current_operator_instruction_review",
        "next_supersession_review": "fresh_latest_instruction_review_required",
    }
    if {row.get("item") for row in rows} != set(expected_rows):
        raise SystemExit(f"{label} operator supersession token boundary items diverged: {metadata}")
    for key in [
        "supersession_token_authorizes_execution",
        "supersession_token_authorizes_local_safe_step",
        "supersession_token_authorizes_risky_work",
        "supersession_token_authorizes_approval",
        "supersession_token_authorizes_recovery_followthrough",
        "supersession_token_authorizes_timebox_override",
        "supersession_token_authorizes_goal_override",
        "supersession_token_authorizes_model_call",
        "supersession_token_authorizes_tool_execution",
        "supersession_token_authorizes_personal_data_read",
        "supersession_token_authorizes_external_side_effect",
        "supersession_token_reusable_for_next_review",
        "supersession_token_reusable_for_next_timebox",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should report {key}=False: {metadata}")
    if metadata.get("next_supersession_requires_fresh_latest_instruction_review") is not True:
        raise SystemExit(f"{label} missed fresh latest-instruction review requirement: {metadata}")
    for row in rows:
        item = row.get("item")
        if row.get("status") != expected_rows.get(item):
            raise SystemExit(f"{label} operator supersession boundary status diverged: {row}")
        if row.get("source") != expected_source:
            raise SystemExit(f"{label} operator supersession boundary source diverged: {row}")
        if row.get("token_sha256") != token_sha256:
            raise SystemExit(f"{label} operator supersession boundary hash diverged: {row}")
        if (
            row.get("authorizes_execution") is not False
            or row.get("authorizes_local_safe_step") is not False
            or row.get("authorizes_risky_work") is not False
            or row.get("authorizes_approval") is not False
            or row.get("authorizes_recovery_followthrough") is not False
            or row.get("authorizes_timebox_override") is not False
            or row.get("authorizes_goal_override") is not False
            or row.get("authorizes_model_call") is not False
            or row.get("authorizes_tool_execution") is not False
            or row.get("authorizes_personal_data_read") is not False
            or row.get("authorizes_external_side_effect") is not False
            or row.get("reusable_for_next_review") is not False
            or row.get("reusable_for_next_timebox") is not False
        ):
            raise SystemExit(f"{label} operator supersession boundary rows should be proof-only: {row}")
    if metadata.get("previous_instruction") is not None and metadata.get("latest_instruction") is not None:
        explicit_timebox_row = next(
            (
                row
                for row in metadata.get("timebox_review_contract_rows") or []
                if isinstance(row, dict) and row.get("item") == "explicit_stop_window"
            ),
            {},
        )
        recomputed = _operator_supersession_token_sha256(
            objective=str(metadata.get("objective") or ""),
            previous_instruction=str(metadata.get("previous_instruction") or ""),
            latest_instruction=str(metadata.get("latest_instruction") or ""),
            stop_at=str(metadata.get("stop_at") or explicit_timebox_row.get("stop_at") or ""),
            current_time=str(metadata.get("current_time") or explicit_timebox_row.get("current_time") or ""),
            timezone_label=str(metadata.get("timezone") or explicit_timebox_row.get("timezone") or "Asia/Seoul"),
            supersession_state=str(metadata.get("supersession_state") or ""),
            timebox_receipt_sha256=str(metadata.get("timebox_receipt_sha256") or ""),
        )
        if recomputed != token_sha256:
            raise SystemExit(f"{label} operator supersession token should be reproducible: {metadata}")
        tampered = _operator_supersession_token_sha256(
            objective=str(metadata.get("objective") or ""),
            previous_instruction=str(metadata.get("previous_instruction") or ""),
            latest_instruction=str(metadata.get("latest_instruction") or ""),
            stop_at=str(metadata.get("stop_at") or explicit_timebox_row.get("stop_at") or ""),
            current_time=str(metadata.get("current_time") or explicit_timebox_row.get("current_time") or ""),
            timezone_label=str(metadata.get("timezone") or explicit_timebox_row.get("timezone") or "Asia/Seoul"),
            supersession_state=str(metadata.get("supersession_state") or ""),
            timebox_receipt_sha256=str(metadata.get("timebox_receipt_sha256") or ""),
            authorizes_timebox_override=True,
        )
        if tampered == token_sha256:
            raise SystemExit(f"{label} operator supersession token should bind timebox-override authority: {metadata}")
        tampered_metadata = dict(metadata)
        tampered_metadata["supersession_token_sha256"] = tampered
        tampered_metadata["supersession_token_boundary_rows"] = [
            {**dict(row), "token_sha256": tampered} for row in rows
        ]
        if _carried_operator_supersession_token_boundary_ready(
            tampered_metadata,
            expected_source=expected_source,
        ):
            raise SystemExit(f"{label} carried operator supersession validator accepted authority-bound stale token: {tampered_metadata}")


def assert_checkpoint_route_boundary(
    metadata: dict,
    label: str,
    *,
    expected_source: str,
    expected_route_status: str = "fresh_checkpoint_bound_for_review",
) -> None:
    token_sha256 = metadata.get("checkpoint_route_token_sha256")
    assert_sha256(token_sha256, f"{label} checkpoint route token")
    if metadata.get("checkpoint_route_token_present") is not True:
        raise SystemExit(f"{label} missed checkpoint route token presence: {metadata}")
    rows = metadata.get("checkpoint_route_boundary_rows") or []
    if metadata.get("checkpoint_route_boundary_row_count") != 3 or len(rows) != 3:
        raise SystemExit(f"{label} missed checkpoint route boundary rows: {metadata}")
    if metadata.get("checkpoint_route_boundary_ready") is not True:
        raise SystemExit(f"{label} missed checkpoint route boundary ready flag: {metadata}")
    if not _checkpoint_route_boundary_ready(
        str(token_sha256),
        rows,
        expected_source=expected_source,
        expected_checkpoint_freshness=str(metadata.get("checkpoint_freshness") or ""),
        expected_checkpoint_needs_review=metadata.get("checkpoint_needs_review"),
    ):
        raise SystemExit(f"{label} production checkpoint route boundary validator rejected rows: {metadata}")
    if not _carried_checkpoint_route_boundary_ready(metadata):
        raise SystemExit(f"{label} carried checkpoint route mirror contract should be production-ready: {metadata}")
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["source"] = "tampered_source"
    if _checkpoint_route_boundary_ready(
        str(token_sha256),
        tampered_rows,
        expected_source=expected_source,
        expected_checkpoint_freshness=str(metadata.get("checkpoint_freshness") or ""),
        expected_checkpoint_needs_review=metadata.get("checkpoint_needs_review"),
    ):
        raise SystemExit(f"{label} production checkpoint route boundary validator accepted tampered source")
    tampered_need_rows = [dict(row) for row in rows]
    tampered_need_rows[0]["checkpoint_needs_review"] = "false"
    if _checkpoint_route_boundary_ready(
        str(token_sha256),
        tampered_need_rows,
        expected_source=expected_source,
        expected_checkpoint_freshness=str(metadata.get("checkpoint_freshness") or ""),
        expected_checkpoint_needs_review="false",
    ):
        raise SystemExit(f"{label} production checkpoint route boundary validator accepted malformed needs-review mirror")
    expected_rows = {
        "checkpoint_route_token": "present",
        "stale_or_missing_checkpoint_route": expected_route_status,
        "next_checkpoint_recovery_review": "fresh_checkpoint_recovery_review_required",
    }
    if {row.get("item") for row in rows} != set(expected_rows):
        raise SystemExit(f"{label} checkpoint route boundary items diverged: {metadata}")
    for key in [
        "checkpoint_route_authorizes_continuation",
        "checkpoint_route_authorizes_local_safe_step",
        "checkpoint_route_authorizes_risky_work",
        "checkpoint_route_authorizes_approval",
        "checkpoint_route_authorizes_recovery_followthrough",
        "checkpoint_route_authorizes_checkpoint_reuse",
        "checkpoint_route_authorizes_model_call",
        "checkpoint_route_authorizes_tool_execution",
        "checkpoint_route_authorizes_personal_data_read",
        "checkpoint_route_authorizes_external_side_effect",
        "checkpoint_route_reusable_for_next_review",
        "checkpoint_route_reusable_for_next_checkpoint",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should report {key}=False: {metadata}")
    if metadata.get("next_checkpoint_route_requires_fresh_recovery_review") is not True:
        raise SystemExit(f"{label} missed fresh checkpoint route review requirement: {metadata}")
    for key, value in [
        ("checkpoint_route_token_present", False),
        ("checkpoint_route_boundary_row_count", 999),
        ("checkpoint_route_boundary_ready", False),
        ("next_checkpoint_route_requires_fresh_recovery_review", False),
        ("checkpoint_route_authorizes_continuation", True),
        ("checkpoint_route_authorizes_checkpoint_reuse", True),
        ("checkpoint_route_reusable_for_next_review", True),
        ("checkpoint_needs_review", "false"),
    ]:
        tampered = dict(metadata)
        tampered[key] = value
        if _carried_checkpoint_route_boundary_ready(tampered):
            raise SystemExit(f"{label} carried checkpoint route contract accepted stale mirror {key}: {tampered}")
    route_queue = [str(command) for command in (metadata.get("checkpoint_route_recovery_proof_queue") or [])]
    recovery_queue = [str(command) for command in (metadata.get("checkpoint_recovery_proof_queue") or [])]
    if not route_queue:
        raise SystemExit(f"{label} missed route-specific checkpoint recovery proof queue: {metadata}")
    if route_queue != recovery_queue:
        raise SystemExit(f"{label} route recovery proof queue diverged from checkpoint recovery queue: {metadata}")
    if metadata.get("checkpoint_route_recovery_proof_queue_count") != len(route_queue):
        raise SystemExit(f"{label} route recovery proof queue count diverged: {metadata}")
    if metadata.get("checkpoint_route_next_recovery_command") != route_queue[0]:
        raise SystemExit(f"{label} route next recovery command should be first proof command: {metadata}")
    if metadata.get("checkpoint_recovery_next_proof_command") != route_queue[0]:
        raise SystemExit(f"{label} checkpoint recovery next proof command should mirror route next command: {metadata}")
    for row in rows:
        item = row.get("item")
        if row.get("status") != expected_rows.get(item):
            raise SystemExit(f"{label} checkpoint route boundary status diverged: {row}")
        if row.get("source") != expected_source:
            raise SystemExit(f"{label} checkpoint route boundary source diverged: {row}")
        if row.get("token_sha256") != token_sha256:
            raise SystemExit(f"{label} checkpoint route boundary hash diverged: {row}")
        if (
            row.get("authorizes_continuation") is not False
            or row.get("authorizes_local_safe_step") is not False
            or row.get("authorizes_risky_work") is not False
            or row.get("authorizes_approval") is not False
            or row.get("authorizes_recovery_followthrough") is not False
            or row.get("authorizes_checkpoint_reuse") is not False
            or row.get("authorizes_model_call") is not False
            or row.get("authorizes_tool_execution") is not False
            or row.get("authorizes_personal_data_read") is not False
            or row.get("authorizes_external_side_effect") is not False
            or row.get("reusable_for_next_review") is not False
            or row.get("reusable_for_next_checkpoint") is not False
        ):
            raise SystemExit(f"{label} checkpoint route boundary rows should be proof-only: {row}")
    if metadata.get("objective") is not None:
        recomputed = _checkpoint_route_token_sha256(
            objective=str(metadata.get("objective") or ""),
            latest_checkpoint_path=str(metadata.get("latest_checkpoint_path") or ""),
            latest_checkpoint_sha256=str(metadata.get("latest_checkpoint_sha256") or ""),
            checkpoint_freshness=str(metadata.get("checkpoint_freshness") or ""),
            checkpoint_needs_review=metadata.get("checkpoint_needs_review"),
        )
        if recomputed != token_sha256:
            raise SystemExit(f"{label} checkpoint route token should be reproducible: {metadata}")
        tampered = _checkpoint_route_token_sha256(
            objective=str(metadata.get("objective") or ""),
            latest_checkpoint_path=str(metadata.get("latest_checkpoint_path") or ""),
            latest_checkpoint_sha256=str(metadata.get("latest_checkpoint_sha256") or ""),
            checkpoint_freshness=str(metadata.get("checkpoint_freshness") or ""),
            checkpoint_needs_review=metadata.get("checkpoint_needs_review"),
            authorizes_continuation=True,
        )
        if tampered == token_sha256:
            raise SystemExit(f"{label} checkpoint route token should bind continuation authority: {metadata}")


def assert_recovery_cockpit_scorecard(metadata: dict, label: str, *, strict_ready: bool) -> None:
    rows = metadata.get("recovery_cockpit_scorecard_rows") or []
    if metadata.get("recovery_cockpit_scorecard_row_count") != len(rows) or len(rows) != 8:
        raise SystemExit(f"{label} missed recovery cockpit scorecard rows: {metadata}")
    if metadata.get("recovery_cockpit_scorecard_ready") is not True:
        raise SystemExit(f"{label} missed recovery cockpit scorecard shape-ready flag: {metadata}")
    if metadata.get("recovery_cockpit_scorecard_strict_ready") is not strict_ready:
        raise SystemExit(f"{label} strict recovery cockpit scorecard flag diverged: {metadata}")
    if not _recovery_cockpit_readiness_scorecard_ready(rows):
        raise SystemExit(f"{label} production recovery cockpit scorecard validator rejected rows: {metadata}")
    if _recovery_cockpit_readiness_scorecard_ready(rows, require_all_ready=True) is not strict_ready:
        raise SystemExit(f"{label} production recovery cockpit strict validator diverged: {metadata}")
    if _scorecard_required_rows_ready(rows) is not strict_ready:
        raise SystemExit(f"{label} exact recovery cockpit row readiness diverged: {metadata}")
    apply_rows = [row for row in rows if row.get("item") == "apply_boundary_intact"]
    if len(apply_rows) != 1:
        raise SystemExit(f"{label} recovery cockpit scorecard missed apply boundary row: {metadata}")
    apply_row = apply_rows[0]
    recovery_apply_queue = apply_row.get("recovery_apply_proof_queue") or []
    if (
        apply_row.get("apply_requires_approval") is not True
        or apply_row.get("recovery_apply_queue_ready") is not True
        or not recovery_apply_queue
        or apply_row.get("recovery_apply_proof_queue_count") != len(recovery_apply_queue)
        or apply_row.get("recovery_apply_next_proof_command") != recovery_apply_queue[0]
    ):
        raise SystemExit(f"{label} recovery cockpit apply boundary row missed command queue contract: {apply_row}")
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["item"] = "timebox_maybe_active"
    if _recovery_cockpit_readiness_scorecard_ready(tampered_rows):
        raise SystemExit(f"{label} recovery cockpit scorecard validator accepted item tampering: {metadata}")
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["authorizes_execution"] = True
    if _recovery_cockpit_readiness_scorecard_ready(tampered_rows):
        raise SystemExit(f"{label} recovery cockpit scorecard validator accepted authority tampering: {metadata}")
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["points"] = 1 if not tampered_rows[0].get("ready") else 0
    if _recovery_cockpit_readiness_scorecard_ready(tampered_rows):
        raise SystemExit(f"{label} recovery cockpit scorecard validator accepted point/ready tampering: {metadata}")
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["ready"] = "false"
    if _scorecard_required_rows_ready(tampered_rows):
        raise SystemExit(f"{label} exact row readiness accepted string false: {metadata}")
    tampered_rows = [dict(row) for row in rows]
    for row in tampered_rows:
        if row.get("item") == "apply_boundary_intact":
            row["recovery_apply_next_proof_command"] = "wrong next proof"
    if _recovery_cockpit_readiness_scorecard_ready(tampered_rows):
        raise SystemExit(f"{label} recovery cockpit scorecard validator accepted apply next-command tampering: {metadata}")


def assert_autonomy_resume_scorecard(metadata: dict, label: str, *, ready: bool) -> None:
    rows = metadata.get("resume_readiness_scorecard_rows") or []
    if metadata.get("resume_readiness_scorecard_row_count") != len(rows) or len(rows) != 9:
        raise SystemExit(f"{label} missed resume readiness scorecard rows: {metadata}")
    if metadata.get("resume_readiness_scorecard_ready") is not ready:
        raise SystemExit(f"{label} resume readiness ready flag diverged: {metadata}")
    if _autonomy_resume_readiness_scorecard_ready(rows) is not ready:
        raise SystemExit(f"{label} production resume scorecard validator diverged: {metadata}")
    if _scorecard_required_rows_ready(rows) is not ready:
        raise SystemExit(f"{label} exact resume row readiness diverged: {metadata}")
    if ready:
        one_step_rows = [row for row in rows if row.get("item") == "one_step_boundary_intact"]
        if len(one_step_rows) != 1:
            raise SystemExit(f"{label} missed one-step boundary row: {metadata}")
        one_step_row = one_step_rows[0]
        resume_queue = one_step_row.get("resume_proof_queue") or []
        if not resume_queue:
            raise SystemExit(f"{label} one-step boundary missed resume proof queue: {metadata}")
        if one_step_row.get("resume_proof_queue_count") != len(resume_queue):
            raise SystemExit(f"{label} one-step boundary missed resume proof queue count: {metadata}")
        if one_step_row.get("resume_next_proof_command") != resume_queue[0]:
            raise SystemExit(f"{label} one-step boundary missed next proof command: {metadata}")
        if one_step_row.get("resume_one_step_boundary_ready") is not True:
            raise SystemExit(f"{label} one-step boundary should be ready: {metadata}")
        for field in [
            "authorizes_local_safe_step",
            "authorizes_approval",
            "authorizes_model_call",
            "authorizes_tool_execution",
            "authorizes_personal_data_read",
            "authorizes_external_side_effect",
        ]:
            if one_step_row.get(field) is not False:
                raise SystemExit(f"{label} one-step boundary should report {field}=False: {metadata}")
        tampered_rows = [dict(row) for row in rows]
        tampered_rows[0]["item"] = "operator_timebox_maybe_active"
        if _autonomy_resume_readiness_scorecard_ready(tampered_rows):
            raise SystemExit(f"{label} resume scorecard validator accepted item tampering: {metadata}")
        tampered_rows = [dict(row) for row in rows]
        tampered_rows[0]["authorizes_unreviewed_continuation"] = True
        if _autonomy_resume_readiness_scorecard_ready(tampered_rows):
            raise SystemExit(f"{label} resume scorecard validator accepted authority tampering: {metadata}")
        tampered_rows = [dict(row) for row in rows]
        tampered_rows[0]["points"] = 0
        if _autonomy_resume_readiness_scorecard_ready(tampered_rows):
            raise SystemExit(f"{label} resume scorecard validator accepted point tampering: {metadata}")
        tampered_rows = [dict(row) for row in rows]
        tampered_rows[0]["ready"] = "false"
        if _scorecard_required_rows_ready(tampered_rows):
            raise SystemExit(f"{label} exact resume row readiness accepted string false: {metadata}")
        tampered_rows = [dict(row) for row in rows]
        for row in tampered_rows:
            if row.get("item") == "one_step_boundary_intact":
                row["resume_next_proof_command"] = "wrong next proof"
        if _autonomy_resume_readiness_scorecard_ready(tampered_rows):
            raise SystemExit(f"{label} resume scorecard validator accepted one-step next-command tampering: {metadata}")
        tampered_rows = [dict(row) for row in rows]
        for row in tampered_rows:
            if row.get("item") == "one_step_boundary_intact":
                row["resume_proof_queue_count"] = 999
        if _autonomy_resume_readiness_scorecard_ready(tampered_rows):
            raise SystemExit(f"{label} resume scorecard validator accepted one-step queue-count tampering: {metadata}")
        tampered_rows = [dict(row) for row in rows]
        for row in tampered_rows:
            if row.get("item") == "one_step_boundary_intact":
                row["authorizes_local_safe_step"] = True
        if _autonomy_resume_readiness_scorecard_ready(tampered_rows):
            raise SystemExit(f"{label} resume scorecard validator accepted one-step authority tampering: {metadata}")


def assert_autonomy_continuation_scorecard(metadata: dict, label: str, *, ready: bool) -> None:
    rows = metadata.get("continuation_readiness_scorecard_rows") or []
    if metadata.get("continuation_readiness_scorecard_row_count") != len(rows) or len(rows) != 9:
        raise SystemExit(f"{label} missed continuation readiness scorecard rows: {metadata}")
    if metadata.get("continuation_readiness_scorecard_ready") is not ready:
        raise SystemExit(f"{label} continuation readiness ready flag diverged: {metadata}")
    if _autonomy_continuation_readiness_scorecard_ready(rows) is not ready:
        raise SystemExit(f"{label} production continuation scorecard validator diverged: {metadata}")
    if _scorecard_required_rows_ready(rows) is not ready:
        raise SystemExit(f"{label} exact continuation row readiness diverged: {metadata}")
    if ready:
        tampered_rows = [dict(row) for row in rows]
        tampered_rows[0]["item"] = "resume_gate_maybe_ready"
        if _autonomy_continuation_readiness_scorecard_ready(tampered_rows):
            raise SystemExit(f"{label} continuation scorecard validator accepted item tampering: {metadata}")
        tampered_rows = [dict(row) for row in rows]
        tampered_rows[0]["authorizes_batching"] = True
        if _autonomy_continuation_readiness_scorecard_ready(tampered_rows):
            raise SystemExit(f"{label} continuation scorecard validator accepted batching authority: {metadata}")
        tampered_rows = [dict(row) for row in rows]
        tampered_rows[0]["points"] = 0
        if _autonomy_continuation_readiness_scorecard_ready(tampered_rows):
            raise SystemExit(f"{label} continuation scorecard validator accepted point tampering: {metadata}")
        tampered_rows = [dict(row) for row in rows]
        tampered_rows[0]["ready"] = "false"
        if _scorecard_required_rows_ready(tampered_rows):
            raise SystemExit(f"{label} exact continuation row readiness accepted string false: {metadata}")


def assert_autonomy_step_closure_scorecard(metadata: dict, label: str, *, ready: bool, prefix: str = "step_closure") -> None:
    key_prefix = f"{prefix}_readiness"
    rows = metadata.get(f"{key_prefix}_scorecard_rows") or []
    if metadata.get(f"{key_prefix}_scorecard_row_count") != len(rows) or len(rows) != 9:
        raise SystemExit(f"{label} missed step-closure readiness scorecard rows: {metadata}")
    if metadata.get(f"{key_prefix}_scorecard_ready") is not ready:
        raise SystemExit(f"{label} step-closure readiness ready flag diverged: {metadata}")
    if _autonomy_step_closure_readiness_scorecard_ready(rows) is not ready:
        raise SystemExit(f"{label} production step-closure scorecard validator diverged: {metadata}")
    if _scorecard_required_rows_ready(rows) is not ready:
        raise SystemExit(f"{label} exact step-closure row readiness diverged: {metadata}")
    if ready:
        tampered_rows = [dict(row) for row in rows]
        tampered_rows[0]["item"] = "pre_step_permission_maybe_ready"
        if _autonomy_step_closure_readiness_scorecard_ready(tampered_rows):
            raise SystemExit(f"{label} step-closure scorecard validator accepted item tampering: {metadata}")
        tampered_rows = [dict(row) for row in rows]
        tampered_rows[0]["authorizes_followup_without_fresh_review"] = True
        if _autonomy_step_closure_readiness_scorecard_ready(tampered_rows):
            raise SystemExit(f"{label} step-closure scorecard validator accepted authority tampering: {metadata}")
        tampered_rows = [dict(row) for row in rows]
        tampered_rows[0]["points"] = 0
        if _autonomy_step_closure_readiness_scorecard_ready(tampered_rows):
            raise SystemExit(f"{label} step-closure scorecard validator accepted point tampering: {metadata}")
        tampered_rows = [dict(row) for row in rows]
        tampered_rows[0]["ready"] = "false"
        if _scorecard_required_rows_ready(tampered_rows):
            raise SystemExit(f"{label} exact step-closure row readiness accepted string false: {metadata}")


def assert_autonomy_cycle_preflight_scorecard(metadata: dict, label: str, *, ready: bool) -> None:
    rows = metadata.get("autonomy_preflight_scorecard_rows") or []
    if metadata.get("autonomy_preflight_scorecard_row_count") != len(rows) or len(rows) != 9:
        raise SystemExit(f"{label} missed autonomy preflight scorecard rows: {metadata}")
    if metadata.get("autonomy_preflight_scorecard_ready") is not ready:
        raise SystemExit(f"{label} autonomy preflight ready flag diverged: {metadata}")
    if _autonomy_cycle_preflight_scorecard_ready(rows) is not ready:
        raise SystemExit(f"{label} production cycle preflight scorecard validator diverged: {metadata}")
    if _scorecard_required_rows_ready(rows) is not ready:
        raise SystemExit(f"{label} exact cycle preflight row readiness diverged: {metadata}")
    if ready:
        tampered_rows = [dict(row) for row in rows]
        tampered_rows[0]["item"] = "pre_step_continuation_maybe_ready"
        if _autonomy_cycle_preflight_scorecard_ready(tampered_rows):
            raise SystemExit(f"{label} cycle preflight scorecard validator accepted item tampering: {metadata}")
        tampered_rows = [dict(row) for row in rows]
        tampered_rows[0]["authorizes_action"] = True
        if _autonomy_cycle_preflight_scorecard_ready(tampered_rows):
            raise SystemExit(f"{label} cycle preflight scorecard validator accepted action authority: {metadata}")
        tampered_rows = [dict(row) for row in rows]
        tampered_rows[0]["points"] = 0
        if _autonomy_cycle_preflight_scorecard_ready(tampered_rows):
            raise SystemExit(f"{label} cycle preflight scorecard validator accepted point tampering: {metadata}")
        tampered_rows = [dict(row) for row in rows]
        tampered_rows[0]["ready"] = "false"
        if _scorecard_required_rows_ready(tampered_rows):
            raise SystemExit(f"{label} exact cycle preflight row readiness accepted string false: {metadata}")


def assert_local_safe_recovery_execution_token(metadata: dict, label: str, *, expected_source: str) -> None:
    token_sha256 = metadata.get("local_safe_recovery_execution_token_sha256")
    assert_sha256(token_sha256, f"{label} local-safe recovery execution token")
    if metadata.get("local_safe_recovery_execution_token_present") is not True:
        raise SystemExit(f"{label} missed local-safe recovery execution token presence: {metadata}")
    rows = metadata.get("local_safe_recovery_execution_token_boundary_rows") or []
    if metadata.get("local_safe_recovery_execution_token_boundary_row_count") != len(rows) or len(rows) != 3:
        raise SystemExit(f"{label} missed local-safe recovery execution token boundary rows: {metadata}")
    if metadata.get("local_safe_recovery_execution_token_boundary_ready") is not True:
        raise SystemExit(f"{label} missed local-safe recovery execution token boundary ready flag: {metadata}")
    if not _local_safe_recovery_execution_token_boundary_ready(
        str(token_sha256),
        rows,
        expected_source=expected_source,
    ):
        raise SystemExit(f"{label} local-safe recovery execution token boundary helper rejected valid rows: {metadata}")
    if not _carried_local_safe_recovery_execution_token_boundary_ready(
        metadata,
        expected_source=expected_source,
    ):
        raise SystemExit(f"{label} carried local-safe recovery execution token helper rejected valid metadata: {metadata}")
    if not _local_safe_recovery_execution_token_metadata_ready(
        metadata,
        expected_source=expected_source,
    ):
        raise SystemExit(f"{label} local-safe recovery execution token metadata helper rejected valid metadata: {metadata}")
    coordinated_tampered_metadata = json.loads(json.dumps(metadata))
    coordinated_tampered_metadata["local_safe_recovery_execution_token_boundary_rows"] = [
        {**row, "source": "stale_local_safe_recovery_execution_source"} for row in rows
    ]
    if _carried_local_safe_recovery_execution_token_boundary_ready(
        coordinated_tampered_metadata,
        expected_source=expected_source,
    ):
        raise SystemExit(
            f"{label} carried local-safe recovery execution token helper accepted coordinated stale source"
        )
    if _local_safe_recovery_execution_token_metadata_ready(
        coordinated_tampered_metadata,
        expected_source=expected_source,
    ):
        raise SystemExit(
            f"{label} local-safe recovery execution token metadata helper accepted coordinated stale source"
        )
    expected_rows = {
        "local_safe_recovery_execution_token": "present",
        "recovery_execution_scope": "proof_only_for_reviewed_local_safe_recovery",
        "next_recovery_execution_review": "fresh_local_safe_token_required",
    }
    if {row.get("item") for row in rows} != set(expected_rows):
        raise SystemExit(f"{label} local-safe recovery execution token boundary items diverged: {metadata}")
    for row in rows:
        item = row.get("item")
        if row.get("status") != expected_rows.get(item):
            raise SystemExit(f"{label} local-safe recovery execution token boundary status diverged: {row}")
        if row.get("source") != expected_source:
            raise SystemExit(f"{label} local-safe recovery execution token boundary source diverged: {row}")
        if row.get("token_sha256") != token_sha256:
            raise SystemExit(f"{label} local-safe recovery execution token boundary hash diverged: {row}")
        if (
            row.get("authorizes_resume_gate") is not False
            or row.get("authorizes_next_step") is not False
            or row.get("authorizes_risky_work") is not False
            or row.get("authorizes_approval") is not False
            or row.get("authorizes_model_call") is not False
            or row.get("authorizes_tool_execution") is not False
            or row.get("authorizes_personal_data_read") is not False
            or row.get("authorizes_external_side_effect") is not False
            or row.get("reusable_for_future_recovery") is not False
        ):
            raise SystemExit(f"{label} local-safe recovery execution token boundary row should be non-authorizing: {row}")
    for key in [
        "local_safe_recovery_execution_token_authorizes_resume_gate",
        "local_safe_recovery_execution_token_authorizes_next_step",
        "local_safe_recovery_execution_token_authorizes_risky_work",
        "local_safe_recovery_execution_token_authorizes_approval",
        "local_safe_recovery_execution_token_authorizes_model_call",
        "local_safe_recovery_execution_token_authorizes_tool_execution",
        "local_safe_recovery_execution_token_authorizes_personal_data_read",
        "local_safe_recovery_execution_token_authorizes_external_side_effect",
        "local_safe_recovery_execution_token_reusable_for_future_recovery",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should report {key}=False: {metadata}")
    if metadata.get("next_recovery_execution_requires_new_local_safe_token") is not True:
        raise SystemExit(f"{label} missed next local-safe recovery execution token requirement: {metadata}")
    for key, value in [
        ("local_safe_recovery_execution_token_present", False),
        ("local_safe_recovery_execution_token_boundary_row_count", 999),
        ("local_safe_recovery_execution_token_boundary_ready", False),
        ("next_recovery_execution_requires_new_local_safe_token", False),
        ("local_safe_recovery_execution_token_authorizes_next_step", True),
        ("local_safe_recovery_execution_token_reusable_for_future_recovery", True),
        ("previous_local_safe_recovery_execution_token_reusable_for_next_review", True),
    ]:
        if key == "previous_local_safe_recovery_execution_token_reusable_for_next_review" and key not in metadata:
            continue
        tampered = dict(metadata)
        tampered[key] = value
        if _carried_local_safe_recovery_execution_token_boundary_ready(
            tampered,
            expected_source=expected_source,
        ):
            raise SystemExit(
                f"{label} carried local-safe recovery execution token helper accepted stale mirror {key}: {tampered}"
            )
        if _local_safe_recovery_execution_token_metadata_ready(
            tampered,
            expected_source=expected_source,
        ):
            raise SystemExit(
                f"{label} local-safe recovery execution token metadata helper accepted stale mirror {key}: {tampered}"
            )
    if all(
        metadata.get(key) is not None
        for key in [
            "objective",
            "reviewed_step",
            "verification",
            "receipt_sha256",
            "checkpoint_sha256",
            "recovery_followthrough_token_sha256",
            "stop_condition",
        ]
    ):
        recomputed = _local_safe_recovery_execution_token_sha256(
            objective=str(metadata.get("objective") or ""),
            reviewed_step=str(metadata.get("reviewed_step") or ""),
            verification=str(metadata.get("verification") or ""),
            receipt_sha256=str(metadata.get("receipt_sha256") or ""),
            checkpoint_sha256=str(metadata.get("checkpoint_sha256") or ""),
            recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
            stop_condition=str(metadata.get("stop_condition") or ""),
        )
        if recomputed != token_sha256:
            raise SystemExit(f"{label} local-safe recovery execution token should be reproducible: {metadata}")
        tampered = _local_safe_recovery_execution_token_sha256(
            objective=str(metadata.get("objective") or ""),
            reviewed_step=str(metadata.get("reviewed_step") or ""),
            verification=str(metadata.get("verification") or ""),
            receipt_sha256=str(metadata.get("receipt_sha256") or ""),
            checkpoint_sha256=str(metadata.get("checkpoint_sha256") or ""),
            recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
            stop_condition=str(metadata.get("stop_condition") or ""),
            authorizes_next_step=True,
        )
        if tampered == token_sha256:
            raise SystemExit(f"{label} local-safe recovery execution token should bind next-step authority: {metadata}")
        stale_metadata = json.loads(json.dumps(metadata))
        stale_token = _local_safe_recovery_execution_token_sha256(
            objective=str(metadata.get("objective") or ""),
            reviewed_step=f"{metadata.get('reviewed_step') or ''} stale",
            verification=str(metadata.get("verification") or ""),
            receipt_sha256=str(metadata.get("receipt_sha256") or ""),
            checkpoint_sha256=str(metadata.get("checkpoint_sha256") or ""),
            recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
            stop_condition=str(metadata.get("stop_condition") or ""),
        )
        stale_metadata["local_safe_recovery_execution_token_sha256"] = stale_token
        stale_metadata["local_safe_recovery_execution_token_boundary_rows"] = [
            {**row, "token_sha256": stale_token} for row in rows
        ]
        if _local_safe_recovery_execution_token_metadata_ready(
            stale_metadata,
            expected_source=expected_source,
        ):
            raise SystemExit(
                f"{label} local-safe recovery execution token metadata helper accepted stale token binding"
            )
        tampered_rows = [dict(row) for row in rows]
        tampered_rows[0]["source"] = "tampered_source"
        if _local_safe_recovery_execution_token_boundary_ready(
            str(token_sha256),
            tampered_rows,
            expected_source=expected_source,
        ):
            raise SystemExit(f"{label} local-safe recovery execution boundary should bind source: {metadata}")


def assert_recovery_execution_readiness_token(metadata: dict, label: str, *, expected_source: str) -> None:
    token_sha256 = metadata.get("recovery_execution_readiness_token_sha256")
    assert_sha256(token_sha256, f"{label} recovery execution readiness token")
    if metadata.get("recovery_execution_readiness_token_present") is not True:
        raise SystemExit(f"{label} missed recovery execution readiness token presence: {metadata}")
    rows = metadata.get("recovery_execution_readiness_token_boundary_rows") or []
    if metadata.get("recovery_execution_readiness_token_boundary_row_count") != len(rows) or len(rows) != 3:
        raise SystemExit(f"{label} missed recovery execution readiness token boundary rows: {metadata}")
    if metadata.get("recovery_execution_readiness_token_boundary_ready") is not True:
        raise SystemExit(f"{label} missed recovery execution readiness token boundary ready flag: {metadata}")
    if not _recovery_execution_readiness_token_boundary_ready(
        str(token_sha256),
        rows,
        expected_source=expected_source,
    ):
        raise SystemExit(f"{label} production recovery execution readiness boundary validator rejected rows: {metadata}")
    if not _carried_recovery_execution_readiness_token_boundary_ready(metadata):
        raise SystemExit(f"{label} carried recovery execution readiness validator rejected metadata: {metadata}")
    if not _carried_recovery_execution_readiness_token_boundary_ready(metadata, expected_source=expected_source):
        raise SystemExit(f"{label} expected-source carried recovery execution readiness validator rejected metadata: {metadata}")
    if not _recovery_execution_readiness_token_metadata_ready(metadata):
        raise SystemExit(f"{label} recovery execution readiness metadata validator rejected metadata: {metadata}")
    if not _recovery_execution_readiness_token_metadata_ready(metadata, expected_source=expected_source):
        raise SystemExit(f"{label} expected-source recovery execution readiness metadata validator rejected metadata: {metadata}")
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["source"] = "tampered_source"
    if _recovery_execution_readiness_token_boundary_ready(
        str(token_sha256),
        tampered_rows,
        expected_source=expected_source,
    ):
        raise SystemExit(f"{label} production recovery execution readiness boundary validator accepted tampered source")
    tampered_metadata = dict(metadata)
    tampered_metadata["recovery_execution_readiness_token_boundary_rows"] = tampered_rows
    if _carried_recovery_execution_readiness_token_boundary_ready(tampered_metadata):
        raise SystemExit(f"{label} carried recovery execution readiness validator accepted tampered metadata")
    if _recovery_execution_readiness_token_metadata_ready(tampered_metadata):
        raise SystemExit(f"{label} recovery execution readiness metadata validator accepted tampered rows")
    coordinated_tampered_metadata = json.loads(json.dumps(metadata))
    coordinated_tampered_rows = [
        {**row, "source": "stale_recovery_execution_readiness_source"}
        for row in coordinated_tampered_metadata.get("recovery_execution_readiness_token_boundary_rows", [])
    ]
    coordinated_tampered_metadata["recovery_execution_readiness_token_boundary_rows"] = coordinated_tampered_rows
    if _carried_recovery_execution_readiness_token_boundary_ready(
        coordinated_tampered_metadata,
        expected_source=expected_source,
    ):
        raise SystemExit(
            f"{label} expected-source carried recovery execution readiness validator accepted coordinated stale source"
        )
    if _recovery_execution_readiness_token_metadata_ready(
        coordinated_tampered_metadata,
        expected_source=expected_source,
    ):
        raise SystemExit(
            f"{label} expected-source recovery execution readiness metadata validator accepted coordinated stale source"
        )
    for key, value in [
        ("recovery_execution_readiness_token_present", False),
        ("recovery_execution_readiness_token_boundary_row_count", 999),
        ("recovery_execution_readiness_token_boundary_ready", False),
        ("next_recovery_execution_requires_new_readiness_token", False),
        ("recovery_execution_readiness_token_authorizes_next_step", True),
        ("recovery_execution_readiness_token_reusable_for_future_recovery", True),
    ]:
        tampered_metadata = dict(metadata)
        tampered_metadata[key] = value
        if _carried_recovery_execution_readiness_token_boundary_ready(tampered_metadata, expected_source=expected_source):
            raise SystemExit(f"{label} carried recovery execution readiness validator accepted stale mirror {key}")
        if _recovery_execution_readiness_token_metadata_ready(tampered_metadata, expected_source=expected_source):
            raise SystemExit(f"{label} recovery execution readiness metadata validator accepted stale mirror {key}")
    expected_rows = {
        "recovery_execution_readiness_token": "present",
        "recovery_execution_readiness_scope": "proof_only_for_scorecard_and_contract",
        "next_recovery_execution_readiness_review": "fresh_readiness_token_required",
    }
    if {row.get("item") for row in rows} != set(expected_rows):
        raise SystemExit(f"{label} recovery execution readiness token boundary items diverged: {metadata}")
    for row in rows:
        item = row.get("item")
        if row.get("status") != expected_rows.get(item):
            raise SystemExit(f"{label} recovery execution readiness token boundary status diverged: {row}")
        if row.get("source") != expected_source:
            raise SystemExit(f"{label} recovery execution readiness token boundary source diverged: {row}")
        if row.get("token_sha256") != token_sha256:
            raise SystemExit(f"{label} recovery execution readiness token boundary hash diverged: {row}")
        if (
            row.get("authorizes_resume_gate") is not False
            or row.get("authorizes_next_step") is not False
            or row.get("authorizes_risky_work") is not False
            or row.get("authorizes_approval") is not False
            or row.get("authorizes_model_call") is not False
            or row.get("authorizes_tool_execution") is not False
            or row.get("authorizes_personal_data_read") is not False
            or row.get("authorizes_external_side_effect") is not False
            or row.get("authorizes_unreviewed_followthrough") is not False
            or row.get("reusable_for_future_recovery") is not False
        ):
            raise SystemExit(f"{label} recovery execution readiness token boundary row should be non-authorizing: {row}")
    for key in [
        "recovery_execution_readiness_token_authorizes_resume_gate",
        "recovery_execution_readiness_token_authorizes_next_step",
        "recovery_execution_readiness_token_authorizes_risky_work",
        "recovery_execution_readiness_token_authorizes_approval",
        "recovery_execution_readiness_token_authorizes_model_call",
        "recovery_execution_readiness_token_authorizes_tool_execution",
        "recovery_execution_readiness_token_authorizes_personal_data_read",
        "recovery_execution_readiness_token_authorizes_external_side_effect",
        "recovery_execution_readiness_token_authorizes_unreviewed_followthrough",
        "recovery_execution_readiness_token_reusable_for_future_recovery",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should report {key}=False: {metadata}")
    if metadata.get("next_recovery_execution_requires_new_readiness_token") is not True:
        raise SystemExit(f"{label} missed fresh readiness token requirement: {metadata}")
    recomputed = _recovery_execution_readiness_token_sha256(
        objective=str(metadata.get("objective") or ""),
        reviewed_step=str(metadata.get("reviewed_step") or ""),
        verification=str(metadata.get("verification_target") or ""),
        receipt_sha256=str(metadata.get("receipt_sha256") or ""),
        receipt_file_sha256=str(metadata.get("receipt_file_sha256") or ""),
        receipt_hash_matches_file=bool(metadata.get("receipt_hash_matches_file")),
        checkpoint_sha256=str(metadata.get("checkpoint_sha256") or ""),
        checkpoint_file_sha256=str(metadata.get("checkpoint_file_sha256") or ""),
        checkpoint_hash_matches_file=bool(metadata.get("checkpoint_hash_matches_file")),
        recovery_step_approval_boundary_token_sha256=str(metadata.get("recovery_step_approval_boundary_token_sha256") or ""),
        recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
        local_safe_recovery_execution_token_sha256=str(metadata.get("local_safe_recovery_execution_token_sha256") or ""),
        recovery_execution_scorecard_rows=list(metadata.get("recovery_execution_scorecard_rows") or []),
        recovery_execution_contract_fields=list(metadata.get("recovery_execution_contract_fields") or []),
        stop_condition=str(metadata.get("stop_condition") or ""),
        risk_signals=list(metadata.get("risky_recovery_signals") or []),
        approval_reference_present=bool(metadata.get("approval_reference_provided")),
    )
    if recomputed != token_sha256:
        raise SystemExit(f"{label} recovery execution readiness token should be reproducible: {metadata}")
    tampered = _recovery_execution_readiness_token_sha256(
        objective=str(metadata.get("objective") or ""),
        reviewed_step=str(metadata.get("reviewed_step") or ""),
        verification=str(metadata.get("verification_target") or ""),
        receipt_sha256=str(metadata.get("receipt_sha256") or ""),
        receipt_file_sha256=str(metadata.get("receipt_file_sha256") or ""),
        receipt_hash_matches_file=bool(metadata.get("receipt_hash_matches_file")),
        checkpoint_sha256=str(metadata.get("checkpoint_sha256") or ""),
        checkpoint_file_sha256=str(metadata.get("checkpoint_file_sha256") or ""),
        checkpoint_hash_matches_file=bool(metadata.get("checkpoint_hash_matches_file")),
        recovery_step_approval_boundary_token_sha256=str(metadata.get("recovery_step_approval_boundary_token_sha256") or ""),
        recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
        local_safe_recovery_execution_token_sha256=str(metadata.get("local_safe_recovery_execution_token_sha256") or ""),
        recovery_execution_scorecard_rows=list(metadata.get("recovery_execution_scorecard_rows") or []),
        recovery_execution_contract_fields=list(metadata.get("recovery_execution_contract_fields") or []),
        stop_condition=str(metadata.get("stop_condition") or ""),
        risk_signals=list(metadata.get("risky_recovery_signals") or []),
        approval_reference_present=bool(metadata.get("approval_reference_provided")),
        authorizes_next_step=True,
    )
    if tampered == token_sha256:
        raise SystemExit(f"{label} recovery execution readiness token should bind next-step authority: {metadata}")
    stale_metadata = json.loads(json.dumps(metadata))
    stale_token = _recovery_execution_readiness_token_sha256(
        objective=str(metadata.get("objective") or ""),
        reviewed_step=f"{metadata.get('reviewed_step') or ''} stale",
        verification=str(metadata.get("verification_target") or ""),
        receipt_sha256=str(metadata.get("receipt_sha256") or ""),
        receipt_file_sha256=str(metadata.get("receipt_file_sha256") or ""),
        receipt_hash_matches_file=bool(metadata.get("receipt_hash_matches_file")),
        checkpoint_sha256=str(metadata.get("checkpoint_sha256") or ""),
        checkpoint_file_sha256=str(metadata.get("checkpoint_file_sha256") or ""),
        checkpoint_hash_matches_file=bool(metadata.get("checkpoint_hash_matches_file")),
        recovery_step_approval_boundary_token_sha256=str(metadata.get("recovery_step_approval_boundary_token_sha256") or ""),
        recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
        local_safe_recovery_execution_token_sha256=str(metadata.get("local_safe_recovery_execution_token_sha256") or ""),
        recovery_execution_scorecard_rows=list(metadata.get("recovery_execution_scorecard_rows") or []),
        recovery_execution_contract_fields=list(metadata.get("recovery_execution_contract_fields") or []),
        stop_condition=str(metadata.get("stop_condition") or ""),
        risk_signals=list(metadata.get("risky_recovery_signals") or []),
        approval_reference_present=bool(metadata.get("approval_reference_provided")),
    )
    stale_metadata["recovery_execution_readiness_token_sha256"] = stale_token
    stale_metadata["recovery_execution_readiness_token_boundary_rows"] = [
        {**row, "token_sha256": stale_token} for row in rows
    ]
    if _recovery_execution_readiness_token_metadata_ready(
        stale_metadata,
        expected_source=expected_source,
    ):
        raise SystemExit(f"{label} recovery execution readiness metadata helper accepted stale token binding")


def assert_checkpoint_recovery_execute_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("checkpoint_recovery_execute_handoff") or {}
    if handoff.get("handoff_ready") is not True or metadata.get("checkpoint_recovery_execute_handoff_ready") is not True:
        raise SystemExit(f"{label} missed held recovery execute handoff readiness: {metadata}")
    if not _checkpoint_recovery_execute_handoff_ready(metadata):
        raise SystemExit(f"{label} held recovery execute handoff failed production validator: {metadata}")
    precontinuation_mirror_tampers = [
        ("operator_timebox_precontinuation_contract_ready", False),
        ("operator_timebox_precontinuation_contract_state", "TAMPERED_TIMEBOX_STATE"),
        ("operator_timebox_precontinuation_allows_continuation", not bool(metadata.get("operator_timebox_precontinuation_allows_continuation"))),
        ("operator_timebox_precontinuation_blocks_continuation", not bool(metadata.get("operator_timebox_precontinuation_blocks_continuation"))),
        ("operator_timebox_precontinuation_requires_stop", not bool(metadata.get("operator_timebox_precontinuation_requires_stop"))),
        ("operator_timebox_precontinuation_requires_fresh_timebox_review", False),
        ("operator_timebox_precontinuation_allowed_states", ["STOP_WINDOW_ACTIVE"]),
        ("operator_supersession_precontinuation_contract_ready", False),
        ("operator_supersession_precontinuation_contract_state", "TAMPERED_SUPERSESSION_STATE"),
        ("operator_supersession_precontinuation_token_boundary_ready", False),
        ("operator_supersession_precontinuation_allows_continuation", not bool(metadata.get("operator_supersession_precontinuation_allows_continuation"))),
        ("operator_supersession_precontinuation_blocks_continuation", not bool(metadata.get("operator_supersession_precontinuation_blocks_continuation"))),
        ("operator_supersession_precontinuation_requires_stop", not bool(metadata.get("operator_supersession_precontinuation_requires_stop"))),
        ("operator_supersession_precontinuation_requires_fresh_latest_instruction_review", False),
        ("operator_supersession_precontinuation_allowed_states", ["LATEST_INSTRUCTION_READY_TO_GOVERN_CONTINUATION"]),
    ]
    for key, tampered_value in precontinuation_mirror_tampers:
        missing_metadata = dict(metadata)
        missing_metadata.pop(key, None)
        if _checkpoint_recovery_execute_handoff_ready(missing_metadata):
            raise SystemExit(f"{label} held recovery execute handoff validator accepted missing pre-continuation mirror {key}")
        tampered_metadata = dict(metadata)
        tampered_metadata[key] = tampered_value
        if _checkpoint_recovery_execute_handoff_ready(tampered_metadata):
            raise SystemExit(f"{label} held recovery execute handoff validator accepted tampered pre-continuation mirror {key}")
    handoff_token = metadata.get("checkpoint_recovery_execute_handoff_token_sha256")
    assert_sha256(handoff_token, f"{label} held recovery execute handoff token")
    if handoff.get("handoff_token_sha256") != handoff_token:
        raise SystemExit(f"{label} held recovery execute handoff token diverged from nested handoff: {metadata}")
    if handoff.get("handoff_token_present") is not True or metadata.get("checkpoint_recovery_execute_handoff_token_present") is not True:
        raise SystemExit(f"{label} held recovery execute handoff token presence flags were not true: {metadata}")
    recomputed_handoff_token = _checkpoint_recovery_execute_handoff_token_sha256(handoff)
    if recomputed_handoff_token != handoff_token:
        raise SystemExit(f"{label} held recovery execute handoff token failed recompute: {metadata}")
    for tamper_key, tamper_value in [
        ("checkpoint_recovery_execute_handoff_token_sha256", "0" * 64),
        ("checkpoint_recovery_execute_handoff_token_present", False),
    ]:
        tampered_token_metadata = dict(metadata)
        tampered_token_metadata[tamper_key] = tamper_value
        if _checkpoint_recovery_execute_handoff_ready(tampered_token_metadata):
            raise SystemExit(f"{label} held recovery execute handoff validator accepted tampered {tamper_key}")
    for tamper_key, tamper_value in [
        ("handoff_token_sha256", "0" * 64),
        ("handoff_token_present", False),
    ]:
        tampered_token_metadata = dict(metadata)
        tampered_handoff = dict(handoff)
        tampered_handoff[tamper_key] = tamper_value
        tampered_token_metadata["checkpoint_recovery_execute_handoff"] = tampered_handoff
        if _checkpoint_recovery_execute_handoff_ready(tampered_token_metadata):
            raise SystemExit(f"{label} held recovery execute handoff validator accepted tampered nested {tamper_key}")
    for key, value in [
        ("normal_followthrough_allowed", True),
        ("contains_raw_step", True),
        ("state", "READY_FOR_NORMAL_FOLLOWTHROUGH_REVIEW"),
        ("recovery_step_approval_boundary_token_sha256", "0" * 64),
        ("recovery_closure_proof_queue_ready", False),
    ]:
        tampered_metadata = dict(metadata)
        tampered_handoff = dict(handoff)
        tampered_handoff[key] = value
        tampered_metadata["checkpoint_recovery_execute_handoff"] = tampered_handoff
        if _checkpoint_recovery_execute_handoff_ready(tampered_metadata):
            raise SystemExit(f"{label} held recovery execute handoff validator accepted tampered {key}: {tampered_handoff}")
    if handoff.get("state") != "HELD_BEFORE_REVIEW" or metadata.get("checkpoint_recovery_execute_handoff_state") != "HELD_BEFORE_REVIEW":
        raise SystemExit(f"{label} held recovery execute handoff state diverged: {metadata}")
    if handoff.get("reviewed") != metadata.get("checkpoint_recovery_execute_handoff_reviewed"):
        raise SystemExit(f"{label} held recovery execute handoff reviewed flag diverged: {metadata}")
    if handoff.get("reviewed") != metadata.get("reviewed"):
        raise SystemExit(f"{label} held recovery execute handoff reviewed mirror diverged: {metadata}")
    approval_reference = str(metadata.get("approval_reference") or "")
    if metadata.get("approval_reference_provided") is True and not approval_reference:
        raise SystemExit(f"{label} held recovery execute handoff missed approval reference metadata: {metadata}")
    if handoff.get("approval_reference") != approval_reference:
        raise SystemExit(f"{label} held recovery execute handoff approval reference diverged: {metadata}")
    missing_nested_approval_reference = dict(metadata)
    missing_nested_handoff = dict(handoff)
    missing_nested_handoff.pop("approval_reference", None)
    missing_nested_approval_reference["checkpoint_recovery_execute_handoff"] = missing_nested_handoff
    if _checkpoint_recovery_execute_handoff_ready(missing_nested_approval_reference):
        raise SystemExit(f"{label} held recovery execute handoff validator accepted missing nested approval reference")
    tampered_nested_approval_reference = dict(metadata)
    tampered_nested_handoff = dict(handoff)
    tampered_nested_handoff["approval_reference"] = "tampered approval reference"
    tampered_nested_approval_reference["checkpoint_recovery_execute_handoff"] = tampered_nested_handoff
    if _checkpoint_recovery_execute_handoff_ready(tampered_nested_approval_reference):
        raise SystemExit(f"{label} held recovery execute handoff validator accepted tampered nested approval reference")
    if handoff.get("recovery_followthrough_gate_state") != metadata.get("recovery_followthrough_gate_state"):
        raise SystemExit(f"{label} held recovery execute handoff follow-through gate diverged: {metadata}")
    if metadata.get("checkpoint_recovery_execute_handoff_recovery_followthrough_gate_state") != handoff.get("recovery_followthrough_gate_state"):
        raise SystemExit(f"{label} held recovery execute handoff flat follow-through gate diverged: {metadata}")
    missing = metadata.get("missing_fields") or []
    if handoff.get("missing_fields") != missing or metadata.get("checkpoint_recovery_execute_handoff_missing_fields") != missing:
        raise SystemExit(f"{label} held recovery execute handoff missing fields diverged: {metadata}")
    if handoff.get("missing_field_count") != len(missing) or metadata.get("checkpoint_recovery_execute_handoff_missing_field_count") != len(missing):
        raise SystemExit(f"{label} held recovery execute handoff missing count diverged: {metadata}")
    stale_missing_metadata = dict(metadata)
    stale_missing = list(missing)
    stale_missing[0] = "stale checkpoint recovery missing field"
    stale_missing_handoff = dict(handoff)
    stale_missing_handoff["missing_fields"] = stale_missing
    stale_missing_handoff["handoff_token_sha256"] = _checkpoint_recovery_execute_handoff_token_sha256(
        stale_missing_handoff
    )
    stale_missing_metadata["missing_fields"] = stale_missing
    stale_missing_metadata["checkpoint_recovery_execute_handoff_missing_fields"] = stale_missing
    stale_missing_metadata["checkpoint_recovery_execute_handoff"] = stale_missing_handoff
    stale_missing_metadata["checkpoint_recovery_execute_handoff_token_sha256"] = stale_missing_handoff[
        "handoff_token_sha256"
    ]
    if _checkpoint_recovery_execute_handoff_ready(stale_missing_metadata):
        raise SystemExit(
            f"{label} held recovery execute handoff validator accepted coordinated stale missing fields"
        )
    closure_missing = metadata.get("recovery_closure_missing") or []
    if handoff.get("recovery_closure_missing") != closure_missing:
        raise SystemExit(f"{label} held recovery execute handoff closure missing diverged: {metadata}")
    if metadata.get("checkpoint_recovery_execute_handoff_recovery_closure_missing") != closure_missing:
        raise SystemExit(f"{label} held recovery execute handoff flat closure missing diverged: {metadata}")
    if handoff.get("recovery_closure_missing_count") != len(closure_missing):
        raise SystemExit(f"{label} held recovery execute handoff closure missing count diverged: {metadata}")
    if metadata.get("recovery_closure_missing_count") != len(closure_missing):
        raise SystemExit(f"{label} held recovery execute handoff metadata closure missing count diverged: {metadata}")
    if metadata.get("checkpoint_recovery_execute_handoff_recovery_closure_missing_count") != len(closure_missing):
        raise SystemExit(f"{label} held recovery execute handoff flat closure missing count diverged: {metadata}")
    if handoff.get("next_safe_command") != metadata.get("recovery_closure_next_safe_command"):
        raise SystemExit(f"{label} held recovery execute handoff next command diverged: {metadata}")
    if metadata.get("checkpoint_recovery_execute_handoff_next_safe_command") != handoff.get("next_safe_command"):
        raise SystemExit(f"{label} held recovery execute handoff flat next command diverged: {metadata}")
    required_evidence = metadata.get("recovery_closure_required_evidence") or []
    if not required_evidence or "approval boundary" not in required_evidence:
        raise SystemExit(f"{label} held recovery execute handoff missed closure required evidence: {metadata}")
    if handoff.get("recovery_closure_required_evidence") != required_evidence:
        raise SystemExit(f"{label} held recovery execute handoff required evidence diverged: {metadata}")
    if metadata.get("checkpoint_recovery_execute_handoff_recovery_closure_required_evidence") != required_evidence:
        raise SystemExit(f"{label} held recovery execute handoff flat required evidence diverged: {metadata}")
    if handoff.get("recovery_closure_required_evidence_count") != len(required_evidence):
        raise SystemExit(f"{label} held recovery execute handoff required evidence count diverged: {metadata}")
    if metadata.get("recovery_closure_required_evidence_count") != len(required_evidence):
        raise SystemExit(f"{label} held recovery execute handoff metadata required evidence count diverged: {metadata}")
    if metadata.get("checkpoint_recovery_execute_handoff_recovery_closure_required_evidence_count") != len(required_evidence):
        raise SystemExit(f"{label} held recovery execute handoff flat required evidence count diverged: {metadata}")
    stale_required_evidence_metadata = dict(metadata)
    stale_required_evidence = list(required_evidence)
    stale_required_evidence[-2] = "stale recovery closure evidence"
    stale_required_handoff = dict(handoff)
    stale_required_handoff["recovery_closure_required_evidence"] = stale_required_evidence
    stale_required_handoff["handoff_token_sha256"] = _checkpoint_recovery_execute_handoff_token_sha256(stale_required_handoff)
    stale_required_evidence_metadata["recovery_closure_required_evidence"] = stale_required_evidence
    stale_required_evidence_metadata[
        "checkpoint_recovery_execute_handoff_recovery_closure_required_evidence"
    ] = stale_required_evidence
    stale_required_evidence_metadata["checkpoint_recovery_execute_handoff"] = stale_required_handoff
    stale_required_evidence_metadata[
        "checkpoint_recovery_execute_handoff_token_sha256"
    ] = stale_required_handoff["handoff_token_sha256"]
    if _checkpoint_recovery_execute_handoff_ready(stale_required_evidence_metadata):
        raise SystemExit(
            f"{label} held recovery execute handoff validator accepted coordinated stale required evidence"
        )
    proof_queue = metadata.get("recovery_closure_proof_queue") or []
    if not proof_queue:
        raise SystemExit(f"{label} held recovery execute handoff missed recovery closure proof queue: {metadata}")
    if handoff.get("recovery_closure_proof_queue") != proof_queue:
        raise SystemExit(f"{label} held recovery execute handoff proof queue diverged: {metadata}")
    if metadata.get("checkpoint_recovery_execute_handoff_recovery_closure_proof_queue") != proof_queue:
        raise SystemExit(f"{label} held recovery execute handoff flat proof queue diverged: {metadata}")
    if handoff.get("recovery_closure_proof_queue_count") != len(proof_queue):
        raise SystemExit(f"{label} held recovery execute handoff proof queue count diverged: {metadata}")
    if metadata.get("recovery_closure_proof_queue_count") != len(proof_queue):
        raise SystemExit(f"{label} held recovery execute handoff metadata proof queue count diverged: {metadata}")
    if metadata.get("checkpoint_recovery_execute_handoff_recovery_closure_proof_queue_count") != len(proof_queue):
        raise SystemExit(f"{label} held recovery execute handoff flat proof queue count diverged: {metadata}")
    if metadata.get("recovery_closure_next_proof_command") != proof_queue[0]:
        raise SystemExit(f"{label} held recovery execute handoff next proof command diverged: {metadata}")
    if handoff.get("recovery_closure_next_proof_command") != proof_queue[0]:
        raise SystemExit(f"{label} held recovery execute handoff nested next proof command diverged: {metadata}")
    if metadata.get("checkpoint_recovery_execute_handoff_recovery_closure_next_proof_command") != proof_queue[0]:
        raise SystemExit(f"{label} held recovery execute handoff flat next proof command diverged: {metadata}")
    if metadata.get("recovery_closure_next_safe_command") != proof_queue[0]:
        raise SystemExit(f"{label} held recovery execute handoff proof queue should start with next safe command: {metadata}")
    if handoff.get("recovery_closure_proof_queue_ready") is not True:
        raise SystemExit(f"{label} held recovery execute handoff missed nested proof queue ready flag: {metadata}")
    if metadata.get("recovery_closure_proof_queue_ready") is not True:
        raise SystemExit(f"{label} held recovery execute handoff missed metadata proof queue ready flag: {metadata}")
    if metadata.get("checkpoint_recovery_execute_handoff_recovery_closure_proof_queue_ready") is not True:
        raise SystemExit(f"{label} held recovery execute handoff missed flat proof queue ready flag: {metadata}")
    missing_flat_ready = dict(metadata)
    missing_flat_ready.pop("checkpoint_recovery_execute_handoff_recovery_closure_proof_queue_ready", None)
    if _checkpoint_recovery_execute_handoff_ready(missing_flat_ready):
        raise SystemExit(f"{label} held recovery execute handoff validator accepted missing flat proof queue ready flag")
    stale_top_proof_queue = dict(metadata)
    stale_queue = list(proof_queue)
    stale_queue[-1] = "checkpoint recovery follow-through: stale closure proof"
    stale_top_proof_queue["recovery_closure_proof_queue"] = stale_queue
    if _checkpoint_recovery_execute_handoff_ready(stale_top_proof_queue):
        raise SystemExit(f"{label} held recovery execute handoff validator accepted stale top-level proof queue")
    stale_flat_proof_queue = dict(metadata)
    stale_flat_proof_queue["checkpoint_recovery_execute_handoff_recovery_closure_proof_queue"] = stale_queue
    if _checkpoint_recovery_execute_handoff_ready(stale_flat_proof_queue):
        raise SystemExit(f"{label} held recovery execute handoff validator accepted stale flat proof queue")
    stale_nested_proof_queue = dict(metadata)
    stale_nested_handoff = dict(handoff)
    stale_nested_handoff["recovery_closure_proof_queue"] = stale_queue
    stale_nested_proof_queue["checkpoint_recovery_execute_handoff"] = stale_nested_handoff
    if _checkpoint_recovery_execute_handoff_ready(stale_nested_proof_queue):
        raise SystemExit(f"{label} held recovery execute handoff validator accepted stale nested proof queue")
    boundary_rows = metadata.get("recovery_step_approval_boundary_rows") or []
    if handoff.get("recovery_step_approval_boundary_rows") != boundary_rows:
        raise SystemExit(f"{label} held recovery execute handoff approval boundary rows diverged: {metadata}")
    if (
        metadata.get("checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_rows")
        != boundary_rows
    ):
        raise SystemExit(f"{label} held recovery execute handoff flat approval boundary rows diverged: {metadata}")
    for top_key in [
        "missing_fields",
        "reviewed",
        "recovery_followthrough_gate_state",
        "recovery_closure_missing",
        "recovery_closure_missing_count",
        "recovery_closure_required_evidence",
        "recovery_closure_proof_queue",
        "recovery_closure_proof_queue_count",
        "recovery_closure_proof_queue_ready",
        "recovery_closure_next_proof_command",
        "recovery_closure_approval_boundary",
        "recovery_closure_next_safe_command",
        "risky_recovery_signals",
        "risky_recovery_signal_count",
        "approval_reference_provided",
        "recovery_step_approval_required_before_recovery",
        "recovery_step_approval_boundary_rows",
        "recovery_step_approval_boundary_row_count",
        "recovery_step_approval_boundary_ready",
        "recovery_step_approval_proof_queue",
        "recovery_step_approval_proof_queue_count",
        "recovery_step_next_approval_proof_command",
        "recovery_step_approval_boundary_token_sha256",
        "recovery_step_approval_boundary_token_present",
        "checkpoint_recovery_execute_handoff_token_sha256",
        "checkpoint_recovery_execute_handoff_token_present",
        "recovery_step_approval_boundary_as_prior_proof",
        *_RECOVERY_STEP_APPROVAL_BOUNDARY_FALSE_FLAGS,
        "recovery_step_sha256",
        "recovery_verification_sha256",
    ]:
        missing_top_contract = dict(metadata)
        missing_top_contract.pop(top_key, None)
        if _checkpoint_recovery_execute_handoff_ready(missing_top_contract):
            raise SystemExit(f"{label} held recovery execute handoff validator accepted missing top-level contract field {top_key}")
    for flat_key in [
        "checkpoint_recovery_execute_handoff_state",
        "checkpoint_recovery_execute_handoff_reviewed",
        "checkpoint_recovery_execute_handoff_normal_followthrough_allowed",
        "checkpoint_recovery_execute_handoff_recovery_followthrough_gate_state",
        "checkpoint_recovery_execute_handoff_missing_fields",
        "checkpoint_recovery_execute_handoff_missing_field_count",
        "checkpoint_recovery_execute_handoff_recovery_closure_missing",
        "checkpoint_recovery_execute_handoff_recovery_closure_missing_count",
        "checkpoint_recovery_execute_handoff_recovery_closure_required_evidence",
        "checkpoint_recovery_execute_handoff_recovery_closure_required_evidence_count",
        "checkpoint_recovery_execute_handoff_recovery_closure_proof_queue",
        "checkpoint_recovery_execute_handoff_recovery_closure_proof_queue_count",
        "checkpoint_recovery_execute_handoff_recovery_closure_next_proof_command",
        "checkpoint_recovery_execute_handoff_recovery_closure_approval_boundary",
        "checkpoint_recovery_execute_handoff_next_safe_command",
        "checkpoint_recovery_execute_handoff_risky_recovery_signals",
        "checkpoint_recovery_execute_handoff_risky_recovery_signal_count",
        "checkpoint_recovery_execute_handoff_approval_reference_provided",
        "checkpoint_recovery_execute_handoff_recovery_step_approval_required_before_recovery",
        "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_rows",
        "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_row_count",
        "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_ready",
        "checkpoint_recovery_execute_handoff_recovery_step_approval_proof_queue",
        "checkpoint_recovery_execute_handoff_recovery_step_approval_proof_queue_count",
        "checkpoint_recovery_execute_handoff_recovery_step_next_approval_proof_command",
        "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_token_sha256",
        "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_token_present",
        "checkpoint_recovery_execute_handoff_token_sha256",
        "checkpoint_recovery_execute_handoff_token_present",
        "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_as_prior_proof",
        *[
            f"checkpoint_recovery_execute_handoff_{flag}"
            for flag in _RECOVERY_STEP_APPROVAL_BOUNDARY_FALSE_FLAGS
        ],
        "checkpoint_recovery_execute_handoff_recovery_step_sha256",
        "checkpoint_recovery_execute_handoff_recovery_verification_sha256",
        "checkpoint_recovery_execute_handoff_authorizes_execution",
    ]:
        missing_flat_contract = dict(metadata)
        missing_flat_contract.pop(flat_key, None)
        if _checkpoint_recovery_execute_handoff_ready(missing_flat_contract):
            raise SystemExit(f"{label} held recovery execute handoff validator accepted missing flat contract field {flat_key}")
    for flag in _CHECKPOINT_RECOVERY_EXECUTE_HANDOFF_FALSE_FLAGS:
        missing_top_level_flag = dict(metadata)
        missing_top_level_flag.pop(flag, None)
        if _checkpoint_recovery_execute_handoff_ready(missing_top_level_flag):
            raise SystemExit(f"{label} held recovery execute handoff validator accepted missing top-level safety flag {flag}")
        tampered_top_level_flag = dict(metadata)
        tampered_top_level_flag[flag] = True
        if _checkpoint_recovery_execute_handoff_ready(tampered_top_level_flag):
            raise SystemExit(f"{label} held recovery execute handoff validator accepted true top-level safety flag {flag}")
        missing_nested_flag = dict(metadata)
        missing_nested_handoff = dict(handoff)
        missing_nested_handoff.pop(flag, None)
        missing_nested_flag["checkpoint_recovery_execute_handoff"] = missing_nested_handoff
        if _checkpoint_recovery_execute_handoff_ready(missing_nested_flag):
            raise SystemExit(f"{label} held recovery execute handoff validator accepted missing nested safety flag {flag}")
        missing_flat_flag = dict(metadata)
        missing_flat_flag.pop(f"checkpoint_recovery_execute_handoff_{flag}", None)
        if _checkpoint_recovery_execute_handoff_ready(missing_flat_flag):
            raise SystemExit(f"{label} held recovery execute handoff validator accepted missing flat safety flag {flag}")
    for key, value in [
        ("reviewed", not bool(metadata.get("reviewed"))),
        ("checkpoint_recovery_execute_handoff_reviewed", not bool(metadata.get("checkpoint_recovery_execute_handoff_reviewed"))),
        ("recovery_followthrough_gate_state", "tampered_gate"),
        ("checkpoint_recovery_execute_handoff_recovery_followthrough_gate_state", "tampered_gate"),
        ("risky_recovery_signal_count", 999),
        ("checkpoint_recovery_execute_handoff_risky_recovery_signal_count", 999),
        ("recovery_closure_missing_count", 999),
        ("checkpoint_recovery_execute_handoff_recovery_closure_missing_count", 999),
        ("recovery_closure_required_evidence_count", 999),
        ("checkpoint_recovery_execute_handoff_recovery_closure_required_evidence_count", 999),
        ("recovery_closure_proof_queue_count", 999),
        ("checkpoint_recovery_execute_handoff_recovery_closure_proof_queue_count", 999),
        ("recovery_closure_proof_queue_ready", False),
        ("checkpoint_recovery_execute_handoff_recovery_closure_proof_queue_ready", False),
        ("recovery_closure_approval_boundary", "tampered explicit approval boundary"),
        (
            "checkpoint_recovery_execute_handoff_recovery_closure_approval_boundary",
            "tampered explicit approval boundary",
        ),
        ("recovery_closure_next_safe_command", "tampered checkpoint recovery execute command"),
        ("checkpoint_recovery_execute_handoff_next_safe_command", "tampered checkpoint recovery execute command"),
        ("approval_reference_provided", not bool(metadata.get("approval_reference_provided"))),
        ("checkpoint_recovery_execute_handoff_approval_reference_provided", not bool(metadata.get("checkpoint_recovery_execute_handoff_approval_reference_provided"))),
        ("recovery_step_approval_required_before_recovery", not bool(metadata.get("recovery_step_approval_required_before_recovery"))),
        (
            "checkpoint_recovery_execute_handoff_recovery_step_approval_required_before_recovery",
            not bool(metadata.get("checkpoint_recovery_execute_handoff_recovery_step_approval_required_before_recovery")),
        ),
        (
            "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_rows",
            [{"item": "tampered approval boundary row"}],
        ),
        ("recovery_step_approval_boundary_row_count", 999),
        ("checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_row_count", 999),
        ("recovery_step_approval_boundary_ready", not bool(metadata.get("recovery_step_approval_boundary_ready"))),
        (
            "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_ready",
            not bool(metadata.get("checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_ready")),
        ),
        ("recovery_step_next_approval_proof_command", "tampered approval proof command"),
        (
            "checkpoint_recovery_execute_handoff_recovery_step_next_approval_proof_command",
            "tampered approval proof command",
        ),
        ("recovery_step_approval_boundary_token_present", False),
        ("checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_token_present", False),
        ("recovery_step_approval_boundary_token_sha256", "0" * 64),
        ("checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_token_sha256", "0" * 64),
        ("recovery_step_approval_boundary_as_prior_proof", not bool(metadata.get("recovery_step_approval_boundary_as_prior_proof"))),
        (
            "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_as_prior_proof",
            not bool(metadata.get("checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_as_prior_proof")),
        ),
        *[(flag, True) for flag in _RECOVERY_STEP_APPROVAL_BOUNDARY_FALSE_FLAGS],
        *[
            (f"checkpoint_recovery_execute_handoff_{flag}", True)
            for flag in _RECOVERY_STEP_APPROVAL_BOUNDARY_FALSE_FLAGS
        ],
        ("recovery_step_sha256", "0" * 64),
        ("checkpoint_recovery_execute_handoff_recovery_step_sha256", "0" * 64),
        ("recovery_verification_sha256", "0" * 64),
        ("checkpoint_recovery_execute_handoff_recovery_verification_sha256", "0" * 64),
    ]:
        tampered_metadata = dict(metadata)
        tampered_metadata[key] = value
        if _checkpoint_recovery_execute_handoff_ready(tampered_metadata):
            raise SystemExit(f"{label} held recovery execute handoff validator accepted stale mirror {key}")
    tampered_boundary_metadata = dict(metadata)
    tampered_rows = [dict(row) for row in (metadata.get("recovery_step_approval_boundary_rows") or [])]
    tampered_rows[0]["status"] = "not_required" if tampered_rows[0].get("status") == "held" else "held"
    tampered_boundary_metadata["recovery_step_approval_boundary_rows"] = tampered_rows
    if _checkpoint_recovery_execute_handoff_ready(tampered_boundary_metadata):
        raise SystemExit(f"{label} held recovery execute handoff validator accepted tampered approval boundary rows")
    tampered_nested_boundary_metadata = dict(metadata)
    tampered_nested_handoff = dict(handoff)
    tampered_nested_rows = [dict(row) for row in (handoff.get("recovery_step_approval_boundary_rows") or [])]
    tampered_nested_rows[0]["status"] = "not_required" if tampered_nested_rows[0].get("status") == "held" else "held"
    tampered_nested_handoff["recovery_step_approval_boundary_rows"] = tampered_nested_rows
    tampered_nested_boundary_metadata["checkpoint_recovery_execute_handoff"] = tampered_nested_handoff
    if _checkpoint_recovery_execute_handoff_ready(tampered_nested_boundary_metadata):
        raise SystemExit(f"{label} held recovery execute handoff validator accepted tampered nested approval boundary rows")
    tampered_nested_row_count_metadata = dict(metadata)
    tampered_nested_row_count_handoff = dict(handoff)
    tampered_nested_row_count_handoff["recovery_step_approval_boundary_row_count"] = 999
    tampered_nested_row_count_metadata["checkpoint_recovery_execute_handoff"] = tampered_nested_row_count_handoff
    if _checkpoint_recovery_execute_handoff_ready(tampered_nested_row_count_metadata):
        raise SystemExit(f"{label} held recovery execute handoff validator accepted stale nested approval boundary row count")
    tampered_closure_missing_metadata = dict(metadata)
    tampered_closure_missing = list(closure_missing)
    tampered_closure_missing.append("tampered closure blocker")
    tampered_closure_missing_metadata["recovery_closure_missing"] = tampered_closure_missing
    if _checkpoint_recovery_execute_handoff_ready(tampered_closure_missing_metadata):
        raise SystemExit(f"{label} held recovery execute handoff validator accepted tampered closure missing list")
    tampered_required_evidence_metadata = dict(metadata)
    tampered_required_evidence = list(required_evidence)
    tampered_required_evidence.append("tampered required evidence")
    tampered_required_evidence_metadata["recovery_closure_required_evidence"] = tampered_required_evidence
    if _checkpoint_recovery_execute_handoff_ready(tampered_required_evidence_metadata):
        raise SystemExit(f"{label} held recovery execute handoff validator accepted tampered required evidence list")
    tampered_proof_queue_metadata = dict(metadata)
    tampered_proof_queue = list(proof_queue)
    tampered_proof_queue.append("tampered recovery closure proof")
    tampered_proof_queue_metadata["recovery_closure_proof_queue"] = tampered_proof_queue
    if _checkpoint_recovery_execute_handoff_ready(tampered_proof_queue_metadata):
        raise SystemExit(f"{label} held recovery execute handoff validator accepted tampered proof queue")
    approval_boundary = metadata.get("recovery_closure_approval_boundary") or ""
    if "explicit approval" not in approval_boundary:
        raise SystemExit(f"{label} held recovery execute handoff missed approval boundary text: {metadata}")
    if handoff.get("recovery_closure_approval_boundary") != approval_boundary:
        raise SystemExit(f"{label} held recovery execute handoff approval boundary diverged: {metadata}")
    if metadata.get("checkpoint_recovery_execute_handoff_recovery_closure_approval_boundary") != approval_boundary:
        raise SystemExit(f"{label} held recovery execute handoff flat approval boundary diverged: {metadata}")
    if handoff.get("normal_followthrough_allowed") is not False or metadata.get("checkpoint_recovery_execute_handoff_normal_followthrough_allowed") is not False:
        raise SystemExit(f"{label} held recovery execute handoff should not allow follow-through: {metadata}")
    if handoff.get("recovery_step_approval_proof_queue") != metadata.get("recovery_step_approval_proof_queue"):
        raise SystemExit(f"{label} held recovery execute handoff approval queue diverged: {metadata}")
    if handoff.get("recovery_step_approval_proof_queue_count") != metadata.get("recovery_step_approval_proof_queue_count"):
        raise SystemExit(f"{label} held recovery execute handoff approval queue count diverged: {metadata}")
    proof_queue = list(metadata.get("recovery_step_approval_proof_queue") or [])
    stale_approval_queue = list(proof_queue) or ["stale recovery approval proof"]
    stale_approval_queue[-1] = "stale checkpoint recovery approval proof"
    stale_approval_rows = _approval_boundary_rows_for_risky_work(
        risk_signals=[str(signal) for signal in (metadata.get("risky_recovery_signals") or [])],
        step_sha256=str(metadata.get("recovery_step_sha256") or ""),
        verification_sha256=str(metadata.get("recovery_verification_sha256") or ""),
        proof_queue=stale_approval_queue,
        required_field="required_before_risky_recovery_step",
    )
    stale_approval_reference = (
        str(metadata.get("approval_reference") or "")
        if metadata.get("approval_reference_provided")
        else ""
    )
    stale_approval_token = _risky_recovery_step_approval_boundary_token_sha256(
        recovery_step_sha256=str(metadata.get("recovery_step_sha256") or ""),
        recovery_verification_sha256=str(metadata.get("recovery_verification_sha256") or ""),
        risk_signals=[str(signal) for signal in (metadata.get("risky_recovery_signals") or [])],
        approval_proof_queue=stale_approval_queue,
        approval_boundary_rows=stale_approval_rows,
        approval_reference=stale_approval_reference,
    )
    stale_approval_ready = _recovery_step_approval_boundary_ready(
        token_sha256=stale_approval_token,
        recovery_step_sha256=str(metadata.get("recovery_step_sha256") or ""),
        recovery_verification_sha256=str(metadata.get("recovery_verification_sha256") or ""),
        approval_proof_queue=stale_approval_queue,
        approval_boundary_rows=stale_approval_rows,
        approval_reference=stale_approval_reference,
    )
    stale_approval_prior = _recovery_step_approval_boundary_ready(
        token_sha256=stale_approval_token,
        recovery_step_sha256=str(metadata.get("recovery_step_sha256") or ""),
        recovery_verification_sha256=str(metadata.get("recovery_verification_sha256") or ""),
        approval_proof_queue=stale_approval_queue,
        approval_boundary_rows=stale_approval_rows,
        approval_reference=stale_approval_reference,
        require_prior_proof=True,
    )
    stale_approval_metadata = dict(metadata)
    stale_approval_handoff = dict(handoff)
    stale_approval_next = stale_approval_queue[0] if stale_approval_queue else ""
    stale_approval_updates = {
        "recovery_step_approval_proof_queue": stale_approval_queue,
        "recovery_step_approval_proof_queue_count": len(stale_approval_queue),
        "recovery_step_next_approval_proof_command": stale_approval_next,
        "recovery_step_approval_boundary_rows": stale_approval_rows,
        "recovery_step_approval_boundary_row_count": len(stale_approval_rows),
        "recovery_step_approval_boundary_ready": stale_approval_ready,
        "recovery_step_approval_boundary_token_sha256": stale_approval_token,
        "recovery_step_approval_boundary_token_present": True,
        "recovery_step_approval_boundary_as_prior_proof": stale_approval_prior,
    }
    stale_approval_flat_updates = {
        "checkpoint_recovery_execute_handoff_recovery_step_approval_proof_queue": stale_approval_queue,
        "checkpoint_recovery_execute_handoff_recovery_step_approval_proof_queue_count": len(stale_approval_queue),
        "checkpoint_recovery_execute_handoff_recovery_step_next_approval_proof_command": stale_approval_next,
        "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_rows": stale_approval_rows,
        "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_row_count": len(stale_approval_rows),
        "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_ready": stale_approval_ready,
        "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_token_sha256": stale_approval_token,
        "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_token_present": True,
        "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_as_prior_proof": stale_approval_prior,
    }
    stale_approval_metadata.update(stale_approval_updates)
    stale_approval_metadata.update(stale_approval_flat_updates)
    stale_approval_handoff.update(stale_approval_updates)
    stale_approval_handoff["handoff_token_sha256"] = _checkpoint_recovery_execute_handoff_token_sha256(
        stale_approval_handoff
    )
    stale_approval_metadata["checkpoint_recovery_execute_handoff"] = stale_approval_handoff
    stale_approval_metadata["checkpoint_recovery_execute_handoff_token_sha256"] = stale_approval_handoff[
        "handoff_token_sha256"
    ]
    if _checkpoint_recovery_execute_handoff_ready(stale_approval_metadata):
        raise SystemExit(
            f"{label} held recovery execute handoff validator accepted coordinated stale approval proof queue"
        )
    if handoff.get("recovery_step_approval_boundary_token_sha256") != metadata.get("recovery_step_approval_boundary_token_sha256"):
        raise SystemExit(f"{label} held recovery execute handoff boundary token diverged: {metadata}")
    if handoff.get("recovery_step_approval_boundary_as_prior_proof") is not metadata.get("recovery_step_approval_boundary_as_prior_proof"):
        raise SystemExit(f"{label} held recovery execute handoff boundary prior-proof diverged: {metadata}")
    for key in _RECOVERY_STEP_APPROVAL_BOUNDARY_FALSE_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} held recovery execute handoff nested boundary flag should report {key}=False: {metadata}")
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} held recovery execute handoff metadata boundary flag should report {key}=False: {metadata}")
        flat_key = f"checkpoint_recovery_execute_handoff_{key}"
        if metadata.get(flat_key) is not False:
            raise SystemExit(f"{label} held recovery execute handoff flat boundary flag should report {flat_key}=False: {metadata}")
        missing_nested_boundary_flag = dict(metadata)
        missing_nested_handoff = dict(handoff)
        missing_nested_handoff.pop(key, None)
        missing_nested_boundary_flag["checkpoint_recovery_execute_handoff"] = missing_nested_handoff
        if _checkpoint_recovery_execute_handoff_ready(missing_nested_boundary_flag):
            raise SystemExit(f"{label} held recovery execute handoff validator accepted missing nested boundary flag {key}")
        tampered_nested_boundary_flag = dict(metadata)
        tampered_nested_handoff = dict(handoff)
        tampered_nested_handoff[key] = True
        tampered_nested_boundary_flag["checkpoint_recovery_execute_handoff"] = tampered_nested_handoff
        if _checkpoint_recovery_execute_handoff_ready(tampered_nested_boundary_flag):
            raise SystemExit(f"{label} held recovery execute handoff validator accepted stale nested boundary flag {key}")
        missing_top_boundary_flag = dict(metadata)
        missing_top_boundary_flag.pop(key, None)
        if _checkpoint_recovery_execute_handoff_ready(missing_top_boundary_flag):
            raise SystemExit(f"{label} held recovery execute handoff validator accepted missing metadata boundary flag {key}")
        tampered_top_boundary_flag = dict(metadata)
        tampered_top_boundary_flag[key] = True
        if _checkpoint_recovery_execute_handoff_ready(tampered_top_boundary_flag):
            raise SystemExit(f"{label} held recovery execute handoff validator accepted stale metadata boundary flag {key}")
        missing_flat_boundary_flag = dict(metadata)
        missing_flat_boundary_flag.pop(flat_key, None)
        if _checkpoint_recovery_execute_handoff_ready(missing_flat_boundary_flag):
            raise SystemExit(f"{label} held recovery execute handoff validator accepted missing flat boundary flag {flat_key}")
        tampered_flat_boundary_flag = dict(metadata)
        tampered_flat_boundary_flag[flat_key] = True
        if _checkpoint_recovery_execute_handoff_ready(tampered_flat_boundary_flag):
            raise SystemExit(f"{label} held recovery execute handoff validator accepted stale flat boundary flag {flat_key}")
    missing_nested_prior_proof = dict(metadata)
    missing_nested_prior_handoff = dict(handoff)
    missing_nested_prior_handoff.pop("recovery_step_approval_boundary_as_prior_proof", None)
    missing_nested_prior_proof["checkpoint_recovery_execute_handoff"] = missing_nested_prior_handoff
    if _checkpoint_recovery_execute_handoff_ready(missing_nested_prior_proof):
        raise SystemExit(f"{label} held recovery execute handoff validator accepted missing nested boundary prior-proof")
    tampered_nested_prior_proof = dict(metadata)
    tampered_nested_prior_handoff = dict(handoff)
    tampered_nested_prior_handoff["recovery_step_approval_boundary_as_prior_proof"] = not bool(
        metadata.get("recovery_step_approval_boundary_as_prior_proof")
    )
    tampered_nested_prior_proof["checkpoint_recovery_execute_handoff"] = tampered_nested_prior_handoff
    if _checkpoint_recovery_execute_handoff_ready(tampered_nested_prior_proof):
        raise SystemExit(f"{label} held recovery execute handoff validator accepted stale nested boundary prior-proof")
    missing_nested_boundary_token_present = dict(metadata)
    missing_nested_token_handoff = dict(handoff)
    missing_nested_token_handoff.pop("recovery_step_approval_boundary_token_present", None)
    missing_nested_boundary_token_present["checkpoint_recovery_execute_handoff"] = missing_nested_token_handoff
    if _checkpoint_recovery_execute_handoff_ready(missing_nested_boundary_token_present):
        raise SystemExit(f"{label} held recovery execute handoff validator accepted missing nested boundary token-present flag")
    tampered_nested_boundary_token_present = dict(metadata)
    tampered_nested_token_handoff = dict(handoff)
    tampered_nested_token_handoff["recovery_step_approval_boundary_token_present"] = False
    tampered_nested_boundary_token_present["checkpoint_recovery_execute_handoff"] = tampered_nested_token_handoff
    if _checkpoint_recovery_execute_handoff_ready(tampered_nested_boundary_token_present):
        raise SystemExit(f"{label} held recovery execute handoff validator accepted stale nested boundary token-present flag")
    if handoff.get("recovery_step_sha256") != metadata.get("recovery_step_sha256"):
        raise SystemExit(f"{label} held recovery execute handoff step hash diverged: {metadata}")
    if handoff.get("recovery_verification_sha256") != metadata.get("recovery_verification_sha256"):
        raise SystemExit(f"{label} held recovery execute handoff verification hash diverged: {metadata}")
    for key in [
        "contains_raw_step",
        "contains_raw_verification",
        "changed_state",
        "writes_notes",
        "writes_files",
        "calls_model",
        "executes_tools",
        "queues_approval",
        "controls_computer",
        "reads_private_data",
        "reads_personal_data",
        "executes_side_effect",
        "external_side_effect",
        "authorizes_execution",
        "authorizes_recovery_followthrough",
        "authorizes_local_safe_step",
        "authorizes_risky_work",
        "authorizes_approval",
        "authorizes_model_call",
        "authorizes_tool_execution",
        "authorizes_personal_data_read",
        "authorizes_external_side_effect",
        "reusable_for_recovery_review",
    ]:
        flat_key = f"checkpoint_recovery_execute_handoff_{key}"
        if handoff.get(key) is not False or metadata.get(flat_key) is not False:
            raise SystemExit(f"{label} held recovery execute handoff should report {key}=False: {metadata}")


def assert_carried_recovery_execution_readiness_token(
    metadata: dict,
    label: str,
    *,
    expected_source: str,
) -> None:
    token_sha256 = metadata.get("recovery_execution_readiness_token_sha256")
    assert_sha256(token_sha256, f"{label} carried recovery execution readiness token")
    if metadata.get("recovery_execution_readiness_token_present") is not True:
        raise SystemExit(f"{label} missed carried recovery execution readiness token presence: {metadata}")
    rows = metadata.get("recovery_execution_readiness_token_boundary_rows") or []
    if metadata.get("recovery_execution_readiness_token_boundary_row_count") != len(rows) or len(rows) != 3:
        raise SystemExit(f"{label} missed carried recovery execution readiness token rows: {metadata}")
    if metadata.get("recovery_execution_readiness_token_boundary_ready") is not True:
        raise SystemExit(f"{label} missed carried recovery execution readiness token boundary ready flag: {metadata}")
    if not _recovery_execution_readiness_token_boundary_ready(
        str(token_sha256),
        rows,
        expected_source=expected_source,
    ):
        raise SystemExit(f"{label} production carried recovery execution readiness boundary validator rejected rows: {metadata}")
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["source"] = "tampered_source"
    if _recovery_execution_readiness_token_boundary_ready(
        str(token_sha256),
        tampered_rows,
        expected_source=expected_source,
    ):
        raise SystemExit(f"{label} production carried recovery execution readiness boundary validator accepted tampered source")
    tampered_metadata = dict(metadata)
    tampered_metadata["recovery_execution_readiness_token_boundary_rows"] = tampered_rows
    if _carried_recovery_execution_readiness_token_boundary_ready(tampered_metadata):
        raise SystemExit(f"{label} carried recovery execution readiness boundary validator accepted tampered metadata")
    coordinated_tampered_metadata = json.loads(json.dumps(metadata))
    coordinated_tampered_rows = [
        {**row, "source": "stale_recovery_execution_readiness_source"}
        for row in coordinated_tampered_metadata.get("recovery_execution_readiness_token_boundary_rows", [])
    ]
    coordinated_tampered_metadata["recovery_execution_readiness_token_boundary_rows"] = coordinated_tampered_rows
    if _carried_recovery_execution_readiness_token_boundary_ready(
        coordinated_tampered_metadata,
        expected_source=expected_source,
    ):
        raise SystemExit(
            f"{label} expected-source carried recovery execution readiness boundary validator accepted coordinated stale source"
        )
    for key, value in [
        ("recovery_execution_readiness_token_present", False),
        ("recovery_execution_readiness_token_boundary_row_count", 999),
        ("recovery_execution_readiness_token_boundary_ready", False),
        ("next_recovery_execution_requires_new_readiness_token", False),
        ("recovery_execution_readiness_token_authorizes_next_step", True),
        ("recovery_execution_readiness_token_reusable_for_future_recovery", True),
    ]:
        tampered_metadata = dict(metadata)
        tampered_metadata[key] = value
        if _carried_recovery_execution_readiness_token_boundary_ready(tampered_metadata, expected_source=expected_source):
            raise SystemExit(
                f"{label} carried recovery execution readiness boundary validator accepted stale mirror {key}"
            )
    expected_rows = {
        "recovery_execution_readiness_token": "present",
        "recovery_execution_readiness_scope": "proof_only_for_scorecard_and_contract",
        "next_recovery_execution_readiness_review": "fresh_readiness_token_required",
    }
    if {row.get("item") for row in rows} != set(expected_rows):
        raise SystemExit(f"{label} carried recovery execution readiness token items diverged: {metadata}")
    for row in rows:
        item = row.get("item")
        if row.get("status") != expected_rows.get(item):
            raise SystemExit(f"{label} carried recovery execution readiness token status diverged: {row}")
        if row.get("source") != expected_source:
            raise SystemExit(f"{label} carried recovery execution readiness token source diverged: {row}")
        if row.get("token_sha256") != token_sha256:
            raise SystemExit(f"{label} carried recovery execution readiness token hash diverged: {row}")
        if (
            row.get("authorizes_resume_gate") is not False
            or row.get("authorizes_next_step") is not False
            or row.get("authorizes_risky_work") is not False
            or row.get("authorizes_approval") is not False
            or row.get("authorizes_model_call") is not False
            or row.get("authorizes_tool_execution") is not False
            or row.get("authorizes_personal_data_read") is not False
            or row.get("authorizes_external_side_effect") is not False
            or row.get("authorizes_unreviewed_followthrough") is not False
            or row.get("reusable_for_future_recovery") is not False
        ):
            raise SystemExit(f"{label} carried recovery execution readiness token row should be non-authorizing: {row}")
    for key in [
        "recovery_execution_readiness_token_authorizes_resume_gate",
        "recovery_execution_readiness_token_authorizes_next_step",
        "recovery_execution_readiness_token_authorizes_risky_work",
        "recovery_execution_readiness_token_authorizes_approval",
        "recovery_execution_readiness_token_authorizes_model_call",
        "recovery_execution_readiness_token_authorizes_tool_execution",
        "recovery_execution_readiness_token_authorizes_personal_data_read",
        "recovery_execution_readiness_token_authorizes_external_side_effect",
        "recovery_execution_readiness_token_authorizes_unreviewed_followthrough",
        "recovery_execution_readiness_token_reusable_for_future_recovery",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should report carried {key}=False: {metadata}")
    if metadata.get("next_recovery_execution_requires_new_readiness_token") is not True:
        raise SystemExit(f"{label} missed next carried recovery execution readiness token requirement: {metadata}")


def assert_recovery_followthrough_token_boundary(metadata: dict, label: str, *, expected_source: str) -> None:
    token_sha256 = metadata.get("recovery_followthrough_token_sha256")
    assert_sha256(token_sha256, f"{label} recovery follow-through token")
    if metadata.get("recovery_followthrough_token_present") is not True:
        raise SystemExit(f"{label} missed recovery follow-through token presence: {metadata}")
    rows = metadata.get("recovery_followthrough_token_boundary_rows") or []
    if metadata.get("recovery_followthrough_token_boundary_row_count") != 3 or len(rows) != 3:
        raise SystemExit(f"{label} missed recovery follow-through token boundary rows: {metadata}")
    if metadata.get("recovery_followthrough_token_boundary_ready") is not True:
        raise SystemExit(f"{label} missed recovery follow-through token boundary ready flag: {metadata}")
    if not _recovery_followthrough_token_boundary_ready(
        str(token_sha256),
        rows,
        expected_source=expected_source,
    ):
        raise SystemExit(f"{label} production recovery follow-through boundary validator rejected rows: {metadata}")
    if not _carried_recovery_followthrough_token_boundary_ready(metadata):
        raise SystemExit(f"{label} carried recovery follow-through boundary validator rejected metadata: {metadata}")
    if not _carried_recovery_followthrough_token_boundary_ready(metadata, expected_source=expected_source):
        raise SystemExit(f"{label} expected-source carried recovery follow-through validator rejected metadata: {metadata}")
    if not _recovery_followthrough_token_metadata_ready(metadata):
        raise SystemExit(f"{label} recovery follow-through metadata validator rejected metadata: {metadata}")
    if not _recovery_followthrough_token_metadata_ready(metadata, expected_source=expected_source):
        raise SystemExit(f"{label} expected-source recovery follow-through metadata validator rejected metadata: {metadata}")
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["source"] = "tampered_source"
    if _recovery_followthrough_token_boundary_ready(
        str(token_sha256),
        tampered_rows,
        expected_source=expected_source,
    ):
        raise SystemExit(f"{label} production recovery follow-through boundary validator accepted tampered source")
    tampered_metadata = dict(metadata)
    tampered_metadata["recovery_followthrough_token_boundary_rows"] = tampered_rows
    if _carried_recovery_followthrough_token_boundary_ready(tampered_metadata):
        raise SystemExit(f"{label} carried recovery follow-through boundary validator accepted tampered metadata")
    if _recovery_followthrough_token_metadata_ready(tampered_metadata):
        raise SystemExit(f"{label} recovery follow-through metadata validator accepted tampered rows")
    coordinated_tampered_metadata = json.loads(json.dumps(metadata))
    coordinated_tampered_rows = [
        {**row, "source": "stale_recovery_followthrough_source"}
        for row in coordinated_tampered_metadata.get("recovery_followthrough_token_boundary_rows", [])
    ]
    coordinated_tampered_metadata["recovery_followthrough_token_boundary_rows"] = coordinated_tampered_rows
    if _carried_recovery_followthrough_token_boundary_ready(
        coordinated_tampered_metadata,
        expected_source=expected_source,
    ):
        raise SystemExit(
            f"{label} expected-source carried recovery follow-through validator accepted coordinated stale source"
        )
    if _recovery_followthrough_token_metadata_ready(
        coordinated_tampered_metadata,
        expected_source=expected_source,
    ):
        raise SystemExit(
            f"{label} expected-source recovery follow-through metadata validator accepted coordinated stale source"
        )
    for key, value in [
        ("recovery_followthrough_token_present", False),
        ("recovery_followthrough_token_boundary_row_count", 999),
        ("recovery_followthrough_token_boundary_ready", False),
        ("next_recovery_followthrough_requires_new_token", False),
        ("recovery_followthrough_token_authorizes_next_step", True),
        ("recovery_followthrough_token_reusable_for_future_recovery", True),
        ("previous_recovery_followthrough_token_reusable_for_next_review", True),
    ]:
        if key == "previous_recovery_followthrough_token_reusable_for_next_review" and key not in metadata:
            continue
        tampered_metadata = dict(metadata)
        tampered_metadata[key] = value
        if _carried_recovery_followthrough_token_boundary_ready(tampered_metadata, expected_source=expected_source):
            raise SystemExit(f"{label} carried recovery follow-through boundary validator accepted stale mirror {key}")
        if _recovery_followthrough_token_metadata_ready(tampered_metadata, expected_source=expected_source):
            raise SystemExit(f"{label} recovery follow-through metadata validator accepted stale mirror {key}")
    if "post_step_proof_queue" in metadata or "continuation_post_step_proof_queue" in metadata:
        stale_queue_metadata = json.loads(json.dumps(metadata))
        queue_key = (
            "post_step_proof_queue"
            if "post_step_proof_queue" in stale_queue_metadata
            else "continuation_post_step_proof_queue"
        )
        count_key = f"{queue_key}_count"
        next_key = (
            "post_step_next_proof_command"
            if queue_key == "post_step_proof_queue"
            else "continuation_post_step_next_proof_command"
        )
        stale_queue = list(stale_queue_metadata.get(queue_key) or [])
        if stale_queue:
            stale_queue[-1] = "completion claim gate: stale recovery follow-through proof"
            stale_queue_metadata[queue_key] = stale_queue
            stale_queue_metadata[count_key] = len(stale_queue)
            stale_queue_metadata[next_key] = stale_queue[0]
            if _recovery_followthrough_token_metadata_ready(stale_queue_metadata, expected_source=expected_source):
                raise SystemExit(
                    f"{label} recovery follow-through metadata validator accepted stale post-step proof queue"
                )
    expected_rows = {
        "recovery_followthrough_token": "present",
        "reviewed_recovery_scope": "proof_only_for_reviewed_recovery_followthrough",
        "next_recovery_followthrough_review": "fresh_followthrough_token_required",
    }
    if {row.get("item") for row in rows} != set(expected_rows):
        raise SystemExit(f"{label} recovery follow-through token boundary items diverged: {metadata}")
    for row in rows:
        item = row.get("item")
        if row.get("status") != expected_rows.get(item):
            raise SystemExit(f"{label} recovery follow-through token boundary status diverged: {row}")
        if row.get("source") != expected_source:
            raise SystemExit(f"{label} recovery follow-through token boundary source diverged: {row}")
        if row.get("token_sha256") != token_sha256:
            raise SystemExit(f"{label} recovery follow-through token boundary hash diverged: {row}")
        if (
            row.get("authorizes_resume_gate") is not False
            or row.get("authorizes_next_step") is not False
            or row.get("authorizes_risky_work") is not False
            or row.get("authorizes_approval") is not False
            or row.get("authorizes_model_call") is not False
            or row.get("authorizes_tool_execution") is not False
            or row.get("authorizes_personal_data_read") is not False
            or row.get("authorizes_external_side_effect") is not False
            or row.get("authorizes_unreviewed_followthrough") is not False
            or row.get("reusable_for_future_recovery") is not False
            or row.get("reusable_for_next_review") is not False
        ):
            raise SystemExit(f"{label} recovery follow-through token boundary row should be non-authorizing: {row}")
    for key in [
        "recovery_followthrough_token_reusable_for_future_recovery",
        "recovery_followthrough_token_authorizes_resume_gate",
        "recovery_followthrough_token_authorizes_next_step",
        "recovery_followthrough_token_authorizes_risky_work",
        "recovery_followthrough_token_authorizes_approval",
        "recovery_followthrough_token_authorizes_model_call",
        "recovery_followthrough_token_authorizes_tool_execution",
        "recovery_followthrough_token_authorizes_personal_data_read",
        "recovery_followthrough_token_authorizes_external_side_effect",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should report {key}=False: {metadata}")
    if metadata.get("next_recovery_followthrough_requires_new_token") is not True:
        raise SystemExit(f"{label} missed next recovery follow-through token requirement: {metadata}")
    if all(
        metadata.get(key) is not None
        for key in [
            "objective",
            "reviewed_step",
            "verification",
            "receipt_path",
            "receipt_sha256",
            "receipt_file_sha256",
            "receipt_hash_matches_file",
            "checkpoint_path",
            "checkpoint_sha256",
            "checkpoint_file_sha256",
            "checkpoint_hash_matches_file",
            "stop_condition",
        ]
    ):
        recomputed = _recovery_followthrough_token_sha256(
            objective=str(metadata.get("objective") or ""),
            reviewed_step=str(metadata.get("reviewed_step") or ""),
            verification=str(metadata.get("verification") or ""),
            receipt_path=str(metadata.get("receipt_path") or ""),
            receipt_sha256=str(metadata.get("receipt_sha256") or ""),
            receipt_file_sha256=str(metadata.get("receipt_file_sha256") or ""),
            receipt_hash_matches_file=bool(metadata.get("receipt_hash_matches_file")),
            checkpoint_path=str(metadata.get("checkpoint_path") or ""),
            checkpoint_sha256=str(metadata.get("checkpoint_sha256") or ""),
            checkpoint_file_sha256=str(metadata.get("checkpoint_file_sha256") or ""),
            checkpoint_hash_matches_file=bool(metadata.get("checkpoint_hash_matches_file")),
            stop_condition=str(metadata.get("stop_condition") or ""),
        )
        if recomputed != token_sha256:
            raise SystemExit(f"{label} recovery follow-through token should be reproducible: {metadata}")
        tampered = _recovery_followthrough_token_sha256(
            objective=str(metadata.get("objective") or ""),
            reviewed_step=str(metadata.get("reviewed_step") or ""),
            verification=str(metadata.get("verification") or ""),
            receipt_path=str(metadata.get("receipt_path") or ""),
            receipt_sha256=str(metadata.get("receipt_sha256") or ""),
            receipt_file_sha256=str(metadata.get("receipt_file_sha256") or ""),
            receipt_hash_matches_file=bool(metadata.get("receipt_hash_matches_file")),
            checkpoint_path=str(metadata.get("checkpoint_path") or ""),
            checkpoint_sha256=str(metadata.get("checkpoint_sha256") or ""),
            checkpoint_file_sha256=str(metadata.get("checkpoint_file_sha256") or ""),
            checkpoint_hash_matches_file=bool(metadata.get("checkpoint_hash_matches_file")),
            stop_condition=str(metadata.get("stop_condition") or ""),
            authorizes_resume_gate=True,
        )
        if tampered == token_sha256:
            raise SystemExit(f"{label} recovery follow-through token should bind resume-gate authority: {metadata}")
        stale_metadata = json.loads(json.dumps(metadata))
        stale_token = _recovery_followthrough_token_sha256(
            objective=str(metadata.get("objective") or ""),
            reviewed_step=f"{metadata.get('reviewed_step') or ''} stale",
            verification=str(metadata.get("verification") or ""),
            receipt_path=str(metadata.get("receipt_path") or ""),
            receipt_sha256=str(metadata.get("receipt_sha256") or ""),
            receipt_file_sha256=str(metadata.get("receipt_file_sha256") or ""),
            receipt_hash_matches_file=bool(metadata.get("receipt_hash_matches_file")),
            checkpoint_path=str(metadata.get("checkpoint_path") or ""),
            checkpoint_sha256=str(metadata.get("checkpoint_sha256") or ""),
            checkpoint_file_sha256=str(metadata.get("checkpoint_file_sha256") or ""),
            checkpoint_hash_matches_file=bool(metadata.get("checkpoint_hash_matches_file")),
            stop_condition=str(metadata.get("stop_condition") or ""),
        )
        stale_metadata["recovery_followthrough_token_sha256"] = stale_token
        stale_metadata["recovery_followthrough_token_boundary_rows"] = [
            {**row, "token_sha256": stale_token} for row in rows
        ]
        if _recovery_followthrough_token_metadata_ready(
            stale_metadata,
            expected_source=expected_source,
        ):
            raise SystemExit(f"{label} recovery follow-through metadata helper accepted stale token binding")


def assert_prior_cycle_ledger_token_boundary(metadata: dict, label: str, *, expected_source: str) -> None:
    token_sha256 = str(metadata.get("prior_cycle_ledger_token_sha256") or "")
    rows = metadata.get("prior_cycle_ledger_token_boundary_rows") or []
    expected_rows = {
        "prior_cycle_ledger_token": "present" if token_sha256 else "not_supplied",
        "current_review_scope": "proof_only_for_current_review",
        "next_cycle_ledger_review": "fresh_cycle_ledger_token_required",
    }
    if metadata.get("prior_cycle_ledger_token_boundary_row_count") != 3 or len(rows) != 3:
        raise SystemExit(f"{label} missed prior cycle ledger token boundary rows: {metadata}")
    if metadata.get("prior_cycle_ledger_token_boundary_ready") is not True:
        raise SystemExit(f"{label} missed prior cycle ledger token boundary ready flag: {metadata}")
    reusable_key = {
        "autonomy_continuation_execution": "prior_cycle_ledger_token_reusable_for_this_review",
        "autonomy_step_closure": "prior_cycle_ledger_token_reusable_for_this_closure",
        "autonomy_cycle_ledger": "prior_cycle_ledger_token_reusable_for_this_cycle",
    }.get(expected_source)
    authority_action_key = {
        "autonomy_continuation_execution": "prior_cycle_ledger_proof_authorizes_action_now",
        "autonomy_step_closure": "prior_cycle_ledger_proof_authorizes_post_step_closure",
        "autonomy_cycle_ledger": "prior_cycle_ledger_proof_authorizes_new_action",
    }.get(expected_source)
    if not reusable_key or not authority_action_key:
        raise SystemExit(f"{label} has unknown prior cycle ledger source {expected_source}")
    if not _prior_cycle_ledger_token_metadata_ready(
        metadata,
        reusable_key=reusable_key,
        authority_action_key=authority_action_key,
        expected_source=expected_source,
    ):
        raise SystemExit(f"{label} prior cycle ledger token failed metadata mirror validator: {metadata}")
    if {row.get("item") for row in rows} != set(expected_rows):
        raise SystemExit(f"{label} prior cycle ledger token boundary items diverged: {metadata}")
    for row in rows:
        item = row.get("item")
        if row.get("status") != expected_rows.get(item):
            raise SystemExit(f"{label} prior cycle ledger token boundary status diverged: {row}")
        if row.get("source") != expected_source:
            raise SystemExit(f"{label} prior cycle ledger token boundary source diverged: {row}")
        if row.get("token_sha256") != token_sha256:
            raise SystemExit(f"{label} prior cycle ledger token boundary hash diverged: {row}")
        if (
            row.get("authorizes_action_now") is not False
            or row.get("authorizes_local_safe_step") is not False
            or row.get("authorizes_risky_work") is not False
            or row.get("authorizes_post_step_closure") is not False
            or row.get("authorizes_new_action") is not False
            or row.get("authorizes_unreviewed_followthrough") is not False
            or row.get("authorizes_timebox_reuse") is not False
            or row.get("authorizes_checkpoint_reuse") is not False
            or row.get("authorizes_token_reuse") is not False
            or row.get("authorizes_model_call") is not False
            or row.get("authorizes_tool_execution") is not False
            or row.get("authorizes_personal_data_read") is not False
            or row.get("authorizes_external_side_effect") is not False
            or row.get("reusable_for_this_review") is not False
            or row.get("reusable_for_this_closure") is not False
            or row.get("reusable_for_this_cycle") is not False
            or row.get("reusable_for_next_review") is not False
        ):
            raise SystemExit(f"{label} prior cycle ledger token boundary row should be non-authorizing: {row}")
    required_false_keys = [
        reusable_key,
        authority_action_key,
        "prior_cycle_ledger_token_reusable_for_this_review",
        "prior_cycle_ledger_token_reusable_for_this_closure",
        "prior_cycle_ledger_token_reusable_for_this_cycle",
        "prior_cycle_ledger_proof_authorizes_action_now",
        "prior_cycle_ledger_proof_authorizes_post_step_closure",
        "prior_cycle_ledger_proof_authorizes_new_action",
        "prior_cycle_ledger_proof_authorizes_model_call",
        "prior_cycle_ledger_proof_authorizes_tool_execution",
        "prior_cycle_ledger_proof_authorizes_personal_data_read",
        "prior_cycle_ledger_proof_authorizes_external_side_effect",
    ]
    for key in required_false_keys:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should report {key}=False: {metadata}")
        missing = dict(metadata)
        missing.pop(key, None)
        if _prior_cycle_ledger_token_metadata_ready(
            missing,
            reusable_key=reusable_key,
            authority_action_key=authority_action_key,
            expected_source=expected_source,
        ):
            raise SystemExit(f"{label} prior cycle ledger metadata validator accepted missing {key}: {missing}")
    for key, value in [
        ("prior_cycle_ledger_token_present", not bool(token_sha256)),
        ("prior_cycle_ledger_token_boundary_row_count", 999),
        ("prior_cycle_ledger_token_boundary_ready", False),
        (reusable_key, True),
        (authority_action_key, True),
        ("prior_cycle_ledger_token_reusable_for_this_review", True),
        ("prior_cycle_ledger_token_reusable_for_this_closure", True),
        ("prior_cycle_ledger_token_reusable_for_this_cycle", True),
        ("prior_cycle_ledger_proof_authorizes_action_now", True),
        ("prior_cycle_ledger_proof_authorizes_post_step_closure", True),
        ("prior_cycle_ledger_proof_authorizes_new_action", True),
        ("prior_cycle_ledger_proof_authorizes_model_call", True),
        ("prior_cycle_ledger_proof_authorizes_tool_execution", True),
        ("prior_cycle_ledger_proof_authorizes_personal_data_read", True),
        ("prior_cycle_ledger_proof_authorizes_external_side_effect", True),
    ]:
        tampered = dict(metadata)
        tampered[key] = value
        if _prior_cycle_ledger_token_metadata_ready(
            tampered,
            reusable_key=reusable_key,
            authority_action_key=authority_action_key,
            expected_source=expected_source,
        ):
            raise SystemExit(f"{label} prior cycle ledger metadata validator accepted tampered {key}: {tampered}")


def assert_autonomy_cycle_ledger_token_boundary(metadata: dict, label: str, *, expected_source: str) -> None:
    token_sha256 = str(metadata.get("autonomy_cycle_ledger_token_sha256") or "")
    rows = metadata.get("autonomy_cycle_ledger_token_boundary_rows") or []
    stage_rows = metadata.get("stage_rows") or []
    expected_rows = {
        "autonomy_cycle_ledger_token": "present",
        "current_cycle_closure_scope": "proof_only_for_completed_cycle",
        "next_cycle_review_boundary": "fresh_cycle_ledger_token_required",
    }
    if metadata.get("autonomy_cycle_ledger_token_boundary_row_count") != 3 or len(rows) != 3:
        raise SystemExit(f"{label} missed autonomy cycle ledger token boundary rows: {metadata}")
    if metadata.get("autonomy_cycle_ledger_token_boundary_ready") is not True:
        raise SystemExit(f"{label} missed autonomy cycle ledger token boundary ready flag: {metadata}")
    if not _autonomy_cycle_ledger_token_metadata_ready(metadata, expected_source=expected_source):
        raise SystemExit(f"{label} autonomy cycle ledger token failed mirror validator: {metadata}")
    coordinated_stale_source_metadata = dict(metadata)
    coordinated_stale_source_metadata["autonomy_cycle_ledger_token_boundary_rows"] = [
        {**row, "source": "stale_autonomy_cycle_ledger_source"} for row in rows
    ]
    if _autonomy_cycle_ledger_token_metadata_ready(
        coordinated_stale_source_metadata,
        expected_source=expected_source,
    ):
        raise SystemExit(
            f"{label} autonomy cycle ledger token accepted coordinated stale-source boundary rows: "
            f"{coordinated_stale_source_metadata}"
        )
    stale_previous_token_reuse_metadata = dict(metadata)
    stale_previous_token_reuse_metadata["previous_cycle_ledger_token_reusable_for_next_review"] = True
    if _autonomy_cycle_ledger_token_metadata_ready(
        stale_previous_token_reuse_metadata,
        expected_source=expected_source,
    ):
        raise SystemExit(
            f"{label} autonomy cycle ledger token accepted previous-cycle token reuse mirror: "
            f"{stale_previous_token_reuse_metadata}"
        )
    missing_previous_token_reuse_metadata = dict(metadata)
    missing_previous_token_reuse_metadata.pop("previous_cycle_ledger_token_reusable_for_next_review", None)
    if _autonomy_cycle_ledger_token_metadata_ready(
        missing_previous_token_reuse_metadata,
        expected_source=expected_source,
    ):
        raise SystemExit(
            f"{label} autonomy cycle ledger token accepted missing previous-cycle token reuse mirror: "
            f"{missing_previous_token_reuse_metadata}"
        )
    if {row.get("item") for row in rows} != set(expected_rows):
        raise SystemExit(f"{label} autonomy cycle ledger token boundary items diverged: {metadata}")
    for row in rows:
        item = row.get("item")
        if row.get("status") != expected_rows.get(item):
            raise SystemExit(f"{label} autonomy cycle ledger token boundary status diverged: {row}")
        if row.get("source") != expected_source:
            raise SystemExit(f"{label} autonomy cycle ledger token boundary source diverged: {row}")
        if row.get("token_sha256") != token_sha256:
            raise SystemExit(f"{label} autonomy cycle ledger token boundary hash diverged: {row}")
        if (
            row.get("authorizes_action_now") is not False
            or row.get("authorizes_local_safe_step") is not False
            or row.get("authorizes_risky_work") is not False
            or row.get("authorizes_new_cycle") is not False
            or row.get("authorizes_unreviewed_followthrough") is not False
            or row.get("authorizes_timebox_reuse") is not False
            or row.get("authorizes_checkpoint_reuse") is not False
            or row.get("authorizes_token_reuse") is not False
            or row.get("authorizes_model_call") is not False
            or row.get("authorizes_tool_execution") is not False
            or row.get("authorizes_personal_data_read") is not False
            or row.get("authorizes_external_side_effect") is not False
            or row.get("reusable_for_next_review") is not False
            or row.get("reusable_for_next_cycle") is not False
        ):
            raise SystemExit(f"{label} autonomy cycle ledger token boundary row should be non-authorizing: {row}")
    for key in [
        "autonomy_cycle_ledger_token_authorizes_action_now",
        "autonomy_cycle_ledger_token_authorizes_local_safe_step",
        "autonomy_cycle_ledger_token_authorizes_risky_work",
        "autonomy_cycle_ledger_token_authorizes_new_cycle",
        "autonomy_cycle_ledger_token_authorizes_unreviewed_followthrough",
        "autonomy_cycle_ledger_token_authorizes_timebox_reuse",
        "autonomy_cycle_ledger_token_authorizes_checkpoint_reuse",
        "autonomy_cycle_ledger_token_authorizes_token_reuse",
        "autonomy_cycle_ledger_token_authorizes_model_call",
        "autonomy_cycle_ledger_token_authorizes_tool_execution",
        "autonomy_cycle_ledger_token_authorizes_personal_data_read",
        "autonomy_cycle_ledger_token_authorizes_external_side_effect",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should report {key}=False: {metadata}")
    if not stage_rows:
        raise SystemExit(f"{label} missed autonomy cycle stage rows: {metadata}")
    for row in stage_rows:
        if (
            row.get("authorizes_action_now") is not False
            or row.get("authorizes_local_safe_step") is not False
            or row.get("authorizes_risky_work") is not False
            or row.get("authorizes_unreviewed_followthrough") is not False
            or row.get("authorizes_timebox_reuse") is not False
            or row.get("authorizes_checkpoint_reuse") is not False
            or row.get("authorizes_token_reuse") is not False
            or row.get("authorizes_approval") is not False
            or row.get("authorizes_model_call") is not False
            or row.get("authorizes_tool_execution") is not False
            or row.get("authorizes_personal_data_read") is not False
            or row.get("authorizes_external_side_effect") is not False
            or row.get("reusable_for_next_review") is not False
            or row.get("reusable_for_next_cycle") is not False
        ):
            raise SystemExit(f"{label} autonomy cycle stage row should be proof-only: {row}")
    if metadata.get("stage_rows") is not None:
        recomputed = _autonomy_cycle_ledger_token_sha256(
            objective=str(metadata.get("objective") or ""),
            ledger_state=str(metadata.get("cycle_state") or ""),
            stage_rows=list(metadata.get("stage_rows") or []),
            timebox_receipt_sha256=str(metadata.get("timebox_receipt_sha256") or ""),
            awake_guard_token_sha256=str(metadata.get("awake_guard_token_sha256") or ""),
            supersession_token_sha256=str(metadata.get("supersession_token_sha256") or ""),
            checkpoint_route_token_sha256=str(metadata.get("checkpoint_route_token_sha256") or ""),
            latest_checkpoint_sha256=str(metadata.get("latest_checkpoint_sha256") or ""),
            recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
            local_safe_recovery_execution_token_sha256=str(metadata.get("local_safe_recovery_execution_token_sha256") or ""),
            one_step_execution_contract_token_sha256=str(metadata.get("one_step_execution_contract_token_sha256") or ""),
            continuation_review_token_sha256=str(metadata.get("continuation_review_token_sha256") or ""),
            prior_cycle_ledger_token_sha256=str(metadata.get("prior_cycle_ledger_token_sha256") or ""),
            proposed_next_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
            completed_step_sha256=str(metadata.get("completed_step_sha256") or ""),
            proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
            post_step_verification_sha256=str(metadata.get("post_step_verification_sha256") or ""),
            receipt_file_sha256=str(metadata.get("receipt_file_sha256") or ""),
            recovery_checkpoint_file_sha256=str(metadata.get("recovery_checkpoint_file_sha256") or ""),
            post_step_receipt_file_sha256=str(metadata.get("post_step_receipt_file_sha256") or ""),
            post_step_checkpoint_file_sha256=str(metadata.get("post_step_checkpoint_file_sha256") or ""),
            next_review_start_command=str(metadata.get("next_review_start_command") or ""),
            autonomy_preflight_scorecard_rows=list(metadata.get("autonomy_preflight_scorecard_rows") or []),
            carried_step_closure_readiness_scorecard_rows=list(metadata.get("carried_step_closure_readiness_scorecard_rows") or []),
            carried_recovery_execution_scorecard_rows=list(metadata.get("carried_recovery_execution_scorecard_rows") or []),
            carried_next_step_approval_boundary_rows=list(metadata.get("carried_next_step_approval_boundary_rows") or []),
        )
        if recomputed != token_sha256:
            raise SystemExit(f"{label} autonomy cycle ledger token should be reproducible: {metadata}")
        tampered = _autonomy_cycle_ledger_token_sha256(
            objective=str(metadata.get("objective") or ""),
            ledger_state=str(metadata.get("cycle_state") or ""),
            stage_rows=list(metadata.get("stage_rows") or []),
            timebox_receipt_sha256=str(metadata.get("timebox_receipt_sha256") or ""),
            awake_guard_token_sha256=str(metadata.get("awake_guard_token_sha256") or ""),
            supersession_token_sha256=str(metadata.get("supersession_token_sha256") or ""),
            checkpoint_route_token_sha256=str(metadata.get("checkpoint_route_token_sha256") or ""),
            latest_checkpoint_sha256=str(metadata.get("latest_checkpoint_sha256") or ""),
            recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
            local_safe_recovery_execution_token_sha256=str(metadata.get("local_safe_recovery_execution_token_sha256") or ""),
            one_step_execution_contract_token_sha256=str(metadata.get("one_step_execution_contract_token_sha256") or ""),
            continuation_review_token_sha256=str(metadata.get("continuation_review_token_sha256") or ""),
            prior_cycle_ledger_token_sha256=str(metadata.get("prior_cycle_ledger_token_sha256") or ""),
            proposed_next_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
            completed_step_sha256=str(metadata.get("completed_step_sha256") or ""),
            proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
            post_step_verification_sha256=str(metadata.get("post_step_verification_sha256") or ""),
            receipt_file_sha256=str(metadata.get("receipt_file_sha256") or ""),
            recovery_checkpoint_file_sha256=str(metadata.get("recovery_checkpoint_file_sha256") or ""),
            post_step_receipt_file_sha256=str(metadata.get("post_step_receipt_file_sha256") or ""),
            post_step_checkpoint_file_sha256=str(metadata.get("post_step_checkpoint_file_sha256") or ""),
            next_review_start_command=str(metadata.get("next_review_start_command") or ""),
            autonomy_preflight_scorecard_rows=list(metadata.get("autonomy_preflight_scorecard_rows") or []),
            carried_step_closure_readiness_scorecard_rows=list(metadata.get("carried_step_closure_readiness_scorecard_rows") or []),
            carried_recovery_execution_scorecard_rows=list(metadata.get("carried_recovery_execution_scorecard_rows") or []),
            carried_next_step_approval_boundary_rows=list(metadata.get("carried_next_step_approval_boundary_rows") or []),
            authorizes_new_cycle=True,
        )
        if tampered == token_sha256:
            raise SystemExit(f"{label} autonomy cycle ledger token should bind new-cycle authority: {metadata}")
        tampered_awake_guard_token = _autonomy_cycle_ledger_token_sha256(
            objective=str(metadata.get("objective") or ""),
            ledger_state=str(metadata.get("cycle_state") or ""),
            stage_rows=list(metadata.get("stage_rows") or []),
            timebox_receipt_sha256=str(metadata.get("timebox_receipt_sha256") or ""),
            awake_guard_token_sha256="0" * 64,
            supersession_token_sha256=str(metadata.get("supersession_token_sha256") or ""),
            checkpoint_route_token_sha256=str(metadata.get("checkpoint_route_token_sha256") or ""),
            latest_checkpoint_sha256=str(metadata.get("latest_checkpoint_sha256") or ""),
            recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
            local_safe_recovery_execution_token_sha256=str(metadata.get("local_safe_recovery_execution_token_sha256") or ""),
            one_step_execution_contract_token_sha256=str(metadata.get("one_step_execution_contract_token_sha256") or ""),
            continuation_review_token_sha256=str(metadata.get("continuation_review_token_sha256") or ""),
            prior_cycle_ledger_token_sha256=str(metadata.get("prior_cycle_ledger_token_sha256") or ""),
            proposed_next_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
            completed_step_sha256=str(metadata.get("completed_step_sha256") or ""),
            proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
            post_step_verification_sha256=str(metadata.get("post_step_verification_sha256") or ""),
            receipt_file_sha256=str(metadata.get("receipt_file_sha256") or ""),
            recovery_checkpoint_file_sha256=str(metadata.get("recovery_checkpoint_file_sha256") or ""),
            post_step_receipt_file_sha256=str(metadata.get("post_step_receipt_file_sha256") or ""),
            post_step_checkpoint_file_sha256=str(metadata.get("post_step_checkpoint_file_sha256") or ""),
            next_review_start_command=str(metadata.get("next_review_start_command") or ""),
            autonomy_preflight_scorecard_rows=list(metadata.get("autonomy_preflight_scorecard_rows") or []),
            carried_step_closure_readiness_scorecard_rows=list(metadata.get("carried_step_closure_readiness_scorecard_rows") or []),
            carried_recovery_execution_scorecard_rows=list(metadata.get("carried_recovery_execution_scorecard_rows") or []),
            carried_next_step_approval_boundary_rows=list(metadata.get("carried_next_step_approval_boundary_rows") or []),
        )
        if tampered_awake_guard_token == token_sha256:
            raise SystemExit(f"{label} autonomy cycle ledger token should bind carried awake guard token: {metadata}")
        tampered_stage_rows = [dict(row) for row in stage_rows]
        tampered_stage_rows[0]["authorizes_tool_execution"] = True
        tampered_stage_row = _autonomy_cycle_ledger_token_sha256(
            objective=str(metadata.get("objective") or ""),
            ledger_state=str(metadata.get("cycle_state") or ""),
            stage_rows=tampered_stage_rows,
            timebox_receipt_sha256=str(metadata.get("timebox_receipt_sha256") or ""),
            awake_guard_token_sha256=str(metadata.get("awake_guard_token_sha256") or ""),
            supersession_token_sha256=str(metadata.get("supersession_token_sha256") or ""),
            checkpoint_route_token_sha256=str(metadata.get("checkpoint_route_token_sha256") or ""),
            latest_checkpoint_sha256=str(metadata.get("latest_checkpoint_sha256") or ""),
            recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
            local_safe_recovery_execution_token_sha256=str(metadata.get("local_safe_recovery_execution_token_sha256") or ""),
            one_step_execution_contract_token_sha256=str(metadata.get("one_step_execution_contract_token_sha256") or ""),
            continuation_review_token_sha256=str(metadata.get("continuation_review_token_sha256") or ""),
            prior_cycle_ledger_token_sha256=str(metadata.get("prior_cycle_ledger_token_sha256") or ""),
            proposed_next_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
            completed_step_sha256=str(metadata.get("completed_step_sha256") or ""),
            proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
            post_step_verification_sha256=str(metadata.get("post_step_verification_sha256") or ""),
            receipt_file_sha256=str(metadata.get("receipt_file_sha256") or ""),
            recovery_checkpoint_file_sha256=str(metadata.get("recovery_checkpoint_file_sha256") or ""),
            post_step_receipt_file_sha256=str(metadata.get("post_step_receipt_file_sha256") or ""),
            post_step_checkpoint_file_sha256=str(metadata.get("post_step_checkpoint_file_sha256") or ""),
            next_review_start_command=str(metadata.get("next_review_start_command") or ""),
            autonomy_preflight_scorecard_rows=list(metadata.get("autonomy_preflight_scorecard_rows") or []),
            carried_step_closure_readiness_scorecard_rows=list(metadata.get("carried_step_closure_readiness_scorecard_rows") or []),
            carried_recovery_execution_scorecard_rows=list(metadata.get("carried_recovery_execution_scorecard_rows") or []),
            carried_next_step_approval_boundary_rows=list(metadata.get("carried_next_step_approval_boundary_rows") or []),
        )
        if tampered_stage_row == token_sha256:
            raise SystemExit(f"{label} autonomy cycle ledger token should bind stage-row authority flags: {metadata}")
        tampered_scorecard_rows = [dict(row) for row in (metadata.get("carried_step_closure_readiness_scorecard_rows") or [])]
        if tampered_scorecard_rows:
            tampered_scorecard_rows[0]["authorizes_risky_work"] = True
            tampered_scorecard_token = _autonomy_cycle_ledger_token_sha256(
                objective=str(metadata.get("objective") or ""),
                ledger_state=str(metadata.get("cycle_state") or ""),
                stage_rows=list(metadata.get("stage_rows") or []),
            timebox_receipt_sha256=str(metadata.get("timebox_receipt_sha256") or ""),
            awake_guard_token_sha256=str(metadata.get("awake_guard_token_sha256") or ""),
            supersession_token_sha256=str(metadata.get("supersession_token_sha256") or ""),
            checkpoint_route_token_sha256=str(metadata.get("checkpoint_route_token_sha256") or ""),
                latest_checkpoint_sha256=str(metadata.get("latest_checkpoint_sha256") or ""),
                recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
                local_safe_recovery_execution_token_sha256=str(metadata.get("local_safe_recovery_execution_token_sha256") or ""),
                one_step_execution_contract_token_sha256=str(metadata.get("one_step_execution_contract_token_sha256") or ""),
                continuation_review_token_sha256=str(metadata.get("continuation_review_token_sha256") or ""),
            prior_cycle_ledger_token_sha256=str(metadata.get("prior_cycle_ledger_token_sha256") or ""),
                proposed_next_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
                completed_step_sha256=str(metadata.get("completed_step_sha256") or ""),
                proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
                post_step_verification_sha256=str(metadata.get("post_step_verification_sha256") or ""),
                receipt_file_sha256=str(metadata.get("receipt_file_sha256") or ""),
                recovery_checkpoint_file_sha256=str(metadata.get("recovery_checkpoint_file_sha256") or ""),
                post_step_receipt_file_sha256=str(metadata.get("post_step_receipt_file_sha256") or ""),
                post_step_checkpoint_file_sha256=str(metadata.get("post_step_checkpoint_file_sha256") or ""),
                next_review_start_command=str(metadata.get("next_review_start_command") or ""),
                autonomy_preflight_scorecard_rows=list(metadata.get("autonomy_preflight_scorecard_rows") or []),
                carried_step_closure_readiness_scorecard_rows=tampered_scorecard_rows,
                carried_recovery_execution_scorecard_rows=list(metadata.get("carried_recovery_execution_scorecard_rows") or []),
                carried_next_step_approval_boundary_rows=list(metadata.get("carried_next_step_approval_boundary_rows") or []),
            )
            if tampered_scorecard_token == token_sha256:
                raise SystemExit(f"{label} autonomy cycle ledger token should bind carried scorecard authority flags: {metadata}")
        tampered_boundary_rows = [dict(row) for row in (metadata.get("carried_next_step_approval_boundary_rows") or [])]
        if tampered_boundary_rows:
            tampered_boundary_rows[0]["authorizes_tool_execution"] = True
            tampered_boundary_token = _autonomy_cycle_ledger_token_sha256(
                objective=str(metadata.get("objective") or ""),
                ledger_state=str(metadata.get("cycle_state") or ""),
                stage_rows=list(metadata.get("stage_rows") or []),
            timebox_receipt_sha256=str(metadata.get("timebox_receipt_sha256") or ""),
            awake_guard_token_sha256=str(metadata.get("awake_guard_token_sha256") or ""),
            supersession_token_sha256=str(metadata.get("supersession_token_sha256") or ""),
            checkpoint_route_token_sha256=str(metadata.get("checkpoint_route_token_sha256") or ""),
                latest_checkpoint_sha256=str(metadata.get("latest_checkpoint_sha256") or ""),
                recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
                local_safe_recovery_execution_token_sha256=str(metadata.get("local_safe_recovery_execution_token_sha256") or ""),
                one_step_execution_contract_token_sha256=str(metadata.get("one_step_execution_contract_token_sha256") or ""),
                continuation_review_token_sha256=str(metadata.get("continuation_review_token_sha256") or ""),
            prior_cycle_ledger_token_sha256=str(metadata.get("prior_cycle_ledger_token_sha256") or ""),
                proposed_next_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
                completed_step_sha256=str(metadata.get("completed_step_sha256") or ""),
                proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
                post_step_verification_sha256=str(metadata.get("post_step_verification_sha256") or ""),
                receipt_file_sha256=str(metadata.get("receipt_file_sha256") or ""),
                recovery_checkpoint_file_sha256=str(metadata.get("recovery_checkpoint_file_sha256") or ""),
                post_step_receipt_file_sha256=str(metadata.get("post_step_receipt_file_sha256") or ""),
                post_step_checkpoint_file_sha256=str(metadata.get("post_step_checkpoint_file_sha256") or ""),
                next_review_start_command=str(metadata.get("next_review_start_command") or ""),
                autonomy_preflight_scorecard_rows=list(metadata.get("autonomy_preflight_scorecard_rows") or []),
                carried_step_closure_readiness_scorecard_rows=list(metadata.get("carried_step_closure_readiness_scorecard_rows") or []),
                carried_recovery_execution_scorecard_rows=list(metadata.get("carried_recovery_execution_scorecard_rows") or []),
                carried_next_step_approval_boundary_rows=tampered_boundary_rows,
            )
            if tampered_boundary_token == token_sha256:
                raise SystemExit(f"{label} autonomy cycle ledger token should bind carried next-step approval boundary flags: {metadata}")
        tampered_reuse_rows = [dict(row) for row in stage_rows]
        tampered_reuse_rows[0]["reusable_for_next_cycle"] = True
        tampered_reuse_token = _autonomy_cycle_ledger_token_sha256(
            objective=str(metadata.get("objective") or ""),
            ledger_state=str(metadata.get("cycle_state") or ""),
            stage_rows=tampered_reuse_rows,
            timebox_receipt_sha256=str(metadata.get("timebox_receipt_sha256") or ""),
            awake_guard_token_sha256=str(metadata.get("awake_guard_token_sha256") or ""),
            supersession_token_sha256=str(metadata.get("supersession_token_sha256") or ""),
            checkpoint_route_token_sha256=str(metadata.get("checkpoint_route_token_sha256") or ""),
            latest_checkpoint_sha256=str(metadata.get("latest_checkpoint_sha256") or ""),
            recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
            local_safe_recovery_execution_token_sha256=str(metadata.get("local_safe_recovery_execution_token_sha256") or ""),
            one_step_execution_contract_token_sha256=str(metadata.get("one_step_execution_contract_token_sha256") or ""),
            continuation_review_token_sha256=str(metadata.get("continuation_review_token_sha256") or ""),
            prior_cycle_ledger_token_sha256=str(metadata.get("prior_cycle_ledger_token_sha256") or ""),
            proposed_next_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
            completed_step_sha256=str(metadata.get("completed_step_sha256") or ""),
            proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
            post_step_verification_sha256=str(metadata.get("post_step_verification_sha256") or ""),
            receipt_file_sha256=str(metadata.get("receipt_file_sha256") or ""),
            recovery_checkpoint_file_sha256=str(metadata.get("recovery_checkpoint_file_sha256") or ""),
            post_step_receipt_file_sha256=str(metadata.get("post_step_receipt_file_sha256") or ""),
            post_step_checkpoint_file_sha256=str(metadata.get("post_step_checkpoint_file_sha256") or ""),
            next_review_start_command=str(metadata.get("next_review_start_command") or ""),
            autonomy_preflight_scorecard_rows=list(metadata.get("autonomy_preflight_scorecard_rows") or []),
            carried_step_closure_readiness_scorecard_rows=list(metadata.get("carried_step_closure_readiness_scorecard_rows") or []),
            carried_recovery_execution_scorecard_rows=list(metadata.get("carried_recovery_execution_scorecard_rows") or []),
            carried_next_step_approval_boundary_rows=list(metadata.get("carried_next_step_approval_boundary_rows") or []),
        )
        if tampered_reuse_token == token_sha256:
            raise SystemExit(f"{label} autonomy cycle ledger token should bind stage-row reuse flags: {metadata}")


def assert_fresh_continuation_review_boundary_token(metadata: dict, label: str, *, expected_source: str) -> None:
    token_sha256 = str(metadata.get("fresh_continuation_review_boundary_token_sha256") or "")
    assert_sha256(token_sha256, f"{label} fresh continuation review boundary token")
    if metadata.get("fresh_continuation_review_boundary_token_present") is not True:
        raise SystemExit(f"{label} missed fresh continuation review boundary token presence: {metadata}")
    rows = metadata.get("fresh_continuation_review_boundary_token_rows") or []
    expected_rows = {
        "fresh_review_boundary_token": "present",
        "next_continuation_preflight": "fresh_operator_timebox_checkpoint_and_review_required",
        "prior_cycle_reuse_boundary": "prior_cycle_proof_only_not_reusable",
    }
    if metadata.get("fresh_continuation_review_boundary_token_row_count") != 3 or len(rows) != 3:
        raise SystemExit(f"{label} missed fresh continuation review boundary rows: {metadata}")
    if metadata.get("fresh_continuation_review_boundary_token_ready") is not True:
        raise SystemExit(f"{label} missed fresh continuation review boundary readiness: {metadata}")
    if not _fresh_review_boundary_token_ready(token_sha256, list(rows)):
        raise SystemExit(f"{label} fresh continuation review boundary failed production validator: {metadata}")
    if not _fresh_review_boundary_metadata_ready(
        metadata,
        token_prefix="fresh_continuation_review_boundary_token",
        authority_prefix="fresh_continuation_review_boundary",
        next_fresh_token_key="next_review_requires_new_fresh_continuation_review_boundary_token",
        expected_source=expected_source,
    ):
        raise SystemExit(f"{label} fresh continuation review boundary mirror validator rejected exact metadata: {metadata}")
    for key, value in [
        ("fresh_continuation_review_boundary_token_present", False),
        ("fresh_continuation_review_boundary_token_row_count", 999),
        ("fresh_continuation_review_boundary_token_ready", False),
        ("next_review_requires_new_fresh_continuation_review_boundary_token", False),
        ("fresh_continuation_review_boundary_authorizes_action_now", True),
        ("fresh_continuation_review_boundary_reusable_for_next_review", True),
        ("fresh_continuation_review_boundary_reusable_for_next_cycle", True),
    ]:
        tampered_metadata = json.loads(json.dumps(metadata))
        tampered_metadata[key] = value
        if _fresh_review_boundary_metadata_ready(
            tampered_metadata,
            token_prefix="fresh_continuation_review_boundary_token",
            authority_prefix="fresh_continuation_review_boundary",
            next_fresh_token_key="next_review_requires_new_fresh_continuation_review_boundary_token",
            expected_source=expected_source,
        ):
            raise SystemExit(f"{label} fresh continuation review boundary accepted stale mirror {key}: {tampered_metadata}")
    for key in [
        "fresh_continuation_review_boundary_authorizes_action_now",
        "fresh_continuation_review_boundary_authorizes_model_call",
        "fresh_continuation_review_boundary_reusable_for_next_review",
    ]:
        tampered_metadata = json.loads(json.dumps(metadata))
        tampered_metadata.pop(key, None)
        if _fresh_review_boundary_metadata_ready(
            tampered_metadata,
            token_prefix="fresh_continuation_review_boundary_token",
            authority_prefix="fresh_continuation_review_boundary",
            next_fresh_token_key="next_review_requires_new_fresh_continuation_review_boundary_token",
            expected_source=expected_source,
        ):
            raise SystemExit(f"{label} fresh continuation review boundary accepted missing mirror {key}: {tampered_metadata}")
    tampered_ready_rows = [dict(row) for row in rows]
    tampered_ready_rows[0]["authorizes_tool_execution"] = True
    if _fresh_review_boundary_token_ready(token_sha256, tampered_ready_rows):
        raise SystemExit(f"{label} fresh continuation review boundary should reject tool-execution authority: {metadata}")
    tampered_ready_rows = [dict(row) for row in rows]
    tampered_ready_rows[0]["fresh_required"] = False
    if _fresh_review_boundary_token_ready(token_sha256, tampered_ready_rows):
        raise SystemExit(f"{label} fresh continuation review boundary should reject stale review rows: {metadata}")
    if {row.get("item") for row in rows} != set(expected_rows):
        raise SystemExit(f"{label} fresh continuation review boundary items diverged: {metadata}")
    for row in rows:
        item = row.get("item")
        if row.get("status") != expected_rows.get(item):
            raise SystemExit(f"{label} fresh continuation review boundary status diverged: {row}")
        if row.get("source") != expected_source:
            raise SystemExit(f"{label} fresh continuation review boundary source diverged: {row}")
        if row.get("token_sha256") != token_sha256:
            raise SystemExit(f"{label} fresh continuation review boundary hash diverged: {row}")
        if (
            row.get("authorizes_action_now") is not False
            or row.get("authorizes_local_safe_step") is not False
            or row.get("authorizes_risky_work") is not False
            or row.get("authorizes_unreviewed_followthrough") is not False
            or row.get("authorizes_timebox_reuse") is not False
            or row.get("authorizes_checkpoint_reuse") is not False
            or row.get("authorizes_token_reuse") is not False
            or row.get("authorizes_model_call") is not False
            or row.get("authorizes_tool_execution") is not False
            or row.get("authorizes_personal_data_read") is not False
            or row.get("authorizes_external_side_effect") is not False
            or row.get("reusable_for_next_review") is not False
            or row.get("reusable_for_next_cycle") is not False
        ):
            raise SystemExit(f"{label} fresh continuation review boundary row should be non-authorizing: {row}")
    for key in [
        "fresh_continuation_review_boundary_authorizes_action_now",
        "fresh_continuation_review_boundary_authorizes_local_safe_step",
        "fresh_continuation_review_boundary_authorizes_risky_work",
        "fresh_continuation_review_boundary_authorizes_unreviewed_followthrough",
        "fresh_continuation_review_boundary_authorizes_timebox_reuse",
        "fresh_continuation_review_boundary_authorizes_checkpoint_reuse",
        "fresh_continuation_review_boundary_authorizes_token_reuse",
        "fresh_continuation_review_boundary_authorizes_model_call",
        "fresh_continuation_review_boundary_authorizes_tool_execution",
        "fresh_continuation_review_boundary_authorizes_personal_data_read",
        "fresh_continuation_review_boundary_authorizes_external_side_effect",
        "fresh_continuation_review_boundary_reusable_for_next_review",
        "fresh_continuation_review_boundary_reusable_for_next_cycle",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should report {key}=False: {metadata}")
    if metadata.get("next_review_requires_new_fresh_continuation_review_boundary_token") is not True:
        raise SystemExit(f"{label} missed next fresh continuation review boundary token requirement: {metadata}")


def assert_fresh_continuation_review_hash_binds_authority(metadata: dict, label: str) -> None:
    rows = [dict(row) for row in metadata.get("fresh_continuation_review_contract_rows") or []]
    if not rows:
        raise SystemExit(f"{label} missed fresh continuation review contract rows: {metadata}")
    original = str(metadata.get("fresh_continuation_review_boundary_token_sha256") or "")
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["authorizes_tool_execution"] = True
    tampered = _fresh_continuation_review_boundary_token_sha256(
        objective=str(metadata.get("objective") or ""),
        closure_state=str(metadata.get("closure_state") or ""),
        next_safe_command=str(metadata.get("next_safe_command") or ""),
        continuation_review_token_sha256=str(metadata.get("continuation_review_token_sha256") or ""),
        fresh_continuation_review_contract_rows=tampered_rows,
    )
    if tampered == original:
        raise SystemExit(f"{label} boundary token should change when a prior artifact authorizes tool execution.")


def assert_step_closure_receipt_token(metadata: dict, label: str, *, prefix: str = "step_closure") -> None:
    token_key = f"{prefix}_receipt_token_sha256"
    present_key = f"{prefix}_receipt_token_present"
    rows_key = f"{prefix}_receipt_boundary_rows"
    row_count_key = f"{prefix}_receipt_boundary_row_count"
    ready_key = f"{prefix}_receipt_boundary_ready"
    token_sha256 = str(metadata.get(token_key) or "")
    assert_sha256(token_sha256, f"{label} receipt token")
    if metadata.get(present_key) is not True:
        raise SystemExit(f"{label} missed receipt token presence: {metadata}")
    rows = metadata.get(rows_key) or []
    expected_rows = {
        "step_closure_receipt_token": "present",
        "post_step_artifact_binding": "receipt_checkpoint_and_verification_bound",
        "next_continuation_review_boundary": "fresh_review_required_before_followup",
    }
    if metadata.get(row_count_key) != 3 or len(rows) != 3:
        raise SystemExit(f"{label} missed receipt boundary rows: {metadata}")
    if metadata.get(ready_key) is not True:
        raise SystemExit(f"{label} missed receipt boundary ready flag: {metadata}")
    carried_metadata = dict(metadata)
    if prefix == "carried_step_closure":
        carried_metadata["step_closure_receipt_token_sha256"] = metadata.get(token_key)
        carried_metadata["step_closure_receipt_boundary_rows"] = rows
    if not _carried_step_closure_receipt_boundary_ready(carried_metadata):
        raise SystemExit(f"{label} receipt boundary failed carried production validator: {metadata}")
    if not _step_closure_receipt_metadata_ready(
        metadata,
        token_prefix=f"{prefix}_receipt_token",
        boundary_prefix=f"{prefix}_receipt_boundary",
        authority_prefix=f"{prefix}_receipt",
    ):
        raise SystemExit(f"{label} receipt boundary failed metadata mirror validator: {metadata}")
    if {row.get("item") for row in rows} != set(expected_rows):
        raise SystemExit(f"{label} receipt boundary items diverged: {metadata}")
    for row in rows:
        item = row.get("item")
        if row.get("status") != expected_rows.get(item):
            raise SystemExit(f"{label} receipt boundary status diverged: {row}")
        if row.get("token_sha256") != token_sha256:
            raise SystemExit(f"{label} receipt boundary token hash diverged: {row}")
        if (
            row.get("authorizes_action_now") is not False
            or row.get("authorizes_local_safe_step") is not False
            or row.get("authorizes_risky_work") is not False
            or row.get("authorizes_new_cycle") is not False
            or row.get("authorizes_unreviewed_followthrough") is not False
            or row.get("authorizes_timebox_reuse") is not False
            or row.get("authorizes_checkpoint_reuse") is not False
            or row.get("authorizes_model_call") is not False
            or row.get("authorizes_tool_execution") is not False
            or row.get("authorizes_personal_data_read") is not False
            or row.get("authorizes_external_side_effect") is not False
            or row.get("reusable_for_next_review") is not False
            or row.get("reusable_for_next_cycle") is not False
        ):
            raise SystemExit(f"{label} receipt boundary row should be non-authorizing: {row}")
    for suffix in [
        "authorizes_action_now",
        "authorizes_local_safe_step",
        "authorizes_risky_work",
        "authorizes_new_cycle",
        "authorizes_unreviewed_followthrough",
        "authorizes_timebox_reuse",
        "authorizes_checkpoint_reuse",
        "authorizes_model_call",
        "authorizes_tool_execution",
        "authorizes_personal_data_read",
        "authorizes_external_side_effect",
        "reusable_for_next_review",
        "reusable_for_next_cycle",
    ]:
        key = f"{prefix}_receipt_{suffix}"
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should report {key}=False: {metadata}")
    tampered_metadata = dict(carried_metadata)
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["authorizes_tool_execution"] = True
    tampered_metadata[rows_key] = tampered_rows
    if _carried_step_closure_receipt_boundary_ready(tampered_metadata):
        raise SystemExit(f"{label} carried receipt boundary accepted executable authority: {tampered_rows}")
    for key, value in [
        (present_key, False),
        (row_count_key, 999),
        (ready_key, False),
        ("next_review_requires_new_step_closure_receipt_token", False),
        (f"{prefix}_receipt_authorizes_action_now", True),
        (f"{prefix}_receipt_reusable_for_next_review", True),
    ]:
        tampered_metadata = dict(metadata)
        tampered_metadata[key] = value
        if _step_closure_receipt_metadata_ready(
            tampered_metadata,
            token_prefix=f"{prefix}_receipt_token",
            boundary_prefix=f"{prefix}_receipt_boundary",
            authority_prefix=f"{prefix}_receipt",
        ):
            raise SystemExit(f"{label} receipt metadata validator accepted tampered {key}: {tampered_metadata}")


def assert_step_closure_receipt_hash_binds_evidence(metadata: dict, label: str, *, prefix: str = "step_closure") -> None:
    original = str(metadata.get(f"{prefix}_receipt_token_sha256") or "")
    scorecard_key = (
        "carried_step_closure_readiness_scorecard_rows"
        if prefix == "carried_step_closure"
        else "step_closure_readiness_scorecard_rows"
    )
    rows = [dict(row) for row in metadata.get(scorecard_key) or []]
    if not rows:
        raise SystemExit(f"{label} missed step closure scorecard rows: {metadata}")
    recomputed = _autonomy_step_closure_receipt_token_sha256(
        objective=str(metadata.get("objective") or ""),
        closure_state=str(metadata.get("closure_state") or ""),
        completed_step_sha256=str(metadata.get("completed_step_sha256") or ""),
        post_step_verification_sha256=str(metadata.get("post_step_verification_sha256") or ""),
        post_step_receipt_sha256=str(metadata.get("post_step_receipt_sha256") or ""),
        post_step_receipt_file_sha256=str(metadata.get("post_step_receipt_file_sha256") or ""),
        post_step_checkpoint_sha256=str(metadata.get("post_step_checkpoint_sha256") or ""),
        post_step_checkpoint_file_sha256=str(metadata.get("post_step_checkpoint_file_sha256") or ""),
        execution_health=str(metadata.get("execution_health") or ""),
        execution_audit=str(metadata.get("execution_audit") or ""),
        after_action_learning=str(metadata.get("after_action_learning") or ""),
        fresh_continuation_review_boundary_token_sha256=str(
            metadata.get("fresh_continuation_review_boundary_token_sha256") or ""
        ),
        step_closure_scorecard_rows=rows,
    )
    if recomputed != original:
        raise SystemExit(f"{label} receipt token should be reproducible: {metadata}")
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["authorizes_followup_without_fresh_review"] = True
    tampered = _autonomy_step_closure_receipt_token_sha256(
        objective=str(metadata.get("objective") or ""),
        closure_state=str(metadata.get("closure_state") or ""),
        completed_step_sha256=str(metadata.get("completed_step_sha256") or ""),
        post_step_verification_sha256=str(metadata.get("post_step_verification_sha256") or ""),
        post_step_receipt_sha256=str(metadata.get("post_step_receipt_sha256") or ""),
        post_step_receipt_file_sha256=str(metadata.get("post_step_receipt_file_sha256") or ""),
        post_step_checkpoint_sha256=str(metadata.get("post_step_checkpoint_sha256") or ""),
        post_step_checkpoint_file_sha256=str(metadata.get("post_step_checkpoint_file_sha256") or ""),
        execution_health=str(metadata.get("execution_health") or ""),
        execution_audit=str(metadata.get("execution_audit") or ""),
        after_action_learning=str(metadata.get("after_action_learning") or ""),
        fresh_continuation_review_boundary_token_sha256=str(
            metadata.get("fresh_continuation_review_boundary_token_sha256") or ""
        ),
        step_closure_scorecard_rows=tampered_rows,
    )
    if tampered == original:
        raise SystemExit(f"{label} receipt token should bind non-authorizing closure rows.")


def assert_fresh_review_hash_binds_authority(metadata: dict, label: str) -> None:
    rows = [dict(row) for row in metadata.get("fresh_review_contract_rows") or []]
    if not rows:
        raise SystemExit(f"{label} missed fresh-review contract rows: {metadata}")
    original = str(metadata.get("fresh_review_boundary_token_sha256") or "")
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["authorizes_personal_data_read"] = True
    tampered = _fresh_review_boundary_token_sha256(
        objective=str(metadata.get("objective") or ""),
        ledger_state=str(metadata.get("cycle_state") or ""),
        fresh_review_preflight_queue=list(metadata.get("fresh_review_preflight_queue") or []),
        fresh_review_contract_rows=tampered_rows,
        autonomy_cycle_ledger_token_sha256=str(metadata.get("autonomy_cycle_ledger_token_sha256") or ""),
        next_review_start_command=str(metadata.get("next_review_start_command") or ""),
    )
    if tampered == original:
        raise SystemExit(f"{label} boundary token should change when a prior artifact authorizes personal-data read.")


def assert_next_review_start_command_boundary(metadata: dict, label: str, *, expected_source: str) -> None:
    token_sha256 = str(metadata.get("next_review_start_command_token_sha256") or "")
    assert_sha256(token_sha256, f"{label} next review start command token")
    if metadata.get("next_review_start_command_token_present") is not True:
        raise SystemExit(f"{label} missed next review start command token presence: {metadata}")
    rows = metadata.get("next_review_start_command_boundary_rows") or []
    expected_rows = {
        "next_review_start_command_token": "present",
        "next_review_pointer_scope": "proof_only_pointer_not_permission",
        "next_review_preflight_required": "fresh_timebox_checkpoint_and_continuation_review_required",
    }
    if metadata.get("next_review_start_command_boundary_row_count") != 3 or len(rows) != 3:
        raise SystemExit(f"{label} missed next review start command boundary rows: {metadata}")
    if metadata.get("next_review_start_command_boundary_ready") is not True:
        raise SystemExit(f"{label} missed next review start command boundary ready flag: {metadata}")
    if not _next_review_start_command_boundary_ready(token_sha256, list(rows)):
        raise SystemExit(f"{label} next review start command boundary failed production validator: {metadata}")
    if not _next_review_start_command_metadata_ready(metadata, expected_source=expected_source):
        raise SystemExit(f"{label} next review start command boundary failed mirror validator: {metadata}")
    tampered_ready_rows = [dict(row) for row in rows]
    tampered_ready_rows[0]["authorizes_approval"] = True
    if _next_review_start_command_boundary_ready(token_sha256, tampered_ready_rows):
        raise SystemExit(f"{label} next review start command boundary should reject approval authority: {metadata}")
    tampered_ready_rows = [dict(row) for row in rows]
    tampered_ready_rows[0]["status"] = "permission_granted"
    if _next_review_start_command_boundary_ready(token_sha256, tampered_ready_rows):
        raise SystemExit(f"{label} next review start command boundary should reject status tampering: {metadata}")
    if {row.get("item") for row in rows} != set(expected_rows):
        raise SystemExit(f"{label} next review start command boundary items diverged: {metadata}")
    for row in rows:
        item = row.get("item")
        if row.get("status") != expected_rows.get(item):
            raise SystemExit(f"{label} next review start command boundary status diverged: {row}")
        if row.get("source") != expected_source:
            raise SystemExit(f"{label} next review start command boundary source diverged: {row}")
        if row.get("token_sha256") != token_sha256:
            raise SystemExit(f"{label} next review start command boundary hash diverged: {row}")
        if (
            row.get("authorizes_action_now") is not False
            or row.get("authorizes_local_safe_step") is not False
            or row.get("authorizes_risky_work") is not False
            or row.get("authorizes_unreviewed_followthrough") is not False
            or row.get("authorizes_timebox_reuse") is not False
            or row.get("authorizes_checkpoint_reuse") is not False
            or row.get("authorizes_token_reuse") is not False
            or row.get("authorizes_model_call") is not False
            or row.get("authorizes_tool_execution") is not False
            or row.get("authorizes_personal_data_read") is not False
            or row.get("authorizes_external_side_effect") is not False
            or row.get("authorizes_approval") is not False
            or row.get("reusable_for_next_review") is not False
            or row.get("reusable_for_next_cycle") is not False
        ):
            raise SystemExit(f"{label} next review start command boundary row should be non-authorizing: {row}")
    for key in [
        "next_review_start_command_authorizes_action_now",
        "next_review_start_command_authorizes_local_safe_step",
        "next_review_start_command_authorizes_risky_work",
        "next_review_start_command_authorizes_unreviewed_followthrough",
        "next_review_start_command_authorizes_timebox_reuse",
        "next_review_start_command_authorizes_checkpoint_reuse",
        "next_review_start_command_authorizes_token_reuse",
        "next_review_start_command_authorizes_model_call",
        "next_review_start_command_authorizes_tool_execution",
        "next_review_start_command_authorizes_personal_data_read",
        "next_review_start_command_authorizes_external_side_effect",
        "next_review_start_command_authorizes_approval",
        "next_review_start_command_reusable_for_next_review",
        "next_review_start_command_reusable_for_next_cycle",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should report {key}=False: {metadata}")
    if metadata.get("next_review_start_command_requires_fresh_preflight") is not True:
        raise SystemExit(f"{label} missed fresh-preflight requirement for next review command: {metadata}")
    recomputed = _next_review_start_command_token_sha256(
        objective=str(metadata.get("objective") or ""),
        ledger_state=str(metadata.get("cycle_state") or ""),
        next_review_start_command=str(metadata.get("next_review_start_command") or ""),
        fresh_review_preflight_queue=list(metadata.get("fresh_review_preflight_queue") or []),
        autonomy_cycle_ledger_token_sha256=str(metadata.get("autonomy_cycle_ledger_token_sha256") or ""),
    )
    if recomputed != token_sha256:
        raise SystemExit(f"{label} next review start command token should be reproducible: {metadata}")
    tampered = _next_review_start_command_token_sha256(
        objective=str(metadata.get("objective") or ""),
        ledger_state=str(metadata.get("cycle_state") or ""),
        next_review_start_command=str(metadata.get("next_review_start_command") or ""),
        fresh_review_preflight_queue=list(metadata.get("fresh_review_preflight_queue") or []),
        autonomy_cycle_ledger_token_sha256=str(metadata.get("autonomy_cycle_ledger_token_sha256") or ""),
        authorizes_local_safe_step=True,
    )
    if tampered == token_sha256:
        raise SystemExit(f"{label} next review start command token should bind local-safe-step authority: {metadata}")


def assert_recovery_execution_boundary(metadata: dict, label: str, *, prefix: str = "recovery_execution_readiness") -> None:
    metadata_prefix = "recovery_execution" if metadata.get("recovery_execution_scorecard_rows") else "carried_recovery_execution"
    rows = metadata.get("recovery_execution_scorecard_rows") or metadata.get("carried_recovery_execution_scorecard_rows") or []
    readiness_key = (
        "recovery_execution_required_rows_ready"
        if metadata.get("recovery_execution_scorecard_rows")
        else "carried_recovery_execution_required_rows_ready"
    )
    scorecard_ready_key = (
        "recovery_execution_scorecard_ready"
        if metadata.get("recovery_execution_scorecard_rows")
        else "carried_recovery_execution_scorecard_ready"
    )
    prior_proof_key = (
        "recovery_execution_readiness_as_prior_proof"
        if metadata.get("recovery_execution_scorecard_rows")
        else "carried_recovery_execution_as_prior_proof"
    )
    scorecard_ready = _recovery_execution_readiness_scorecard_ready(list(rows))
    if _recovery_execution_scorecard_metadata_ready(
        metadata,
        prefix=metadata_prefix,
        prior_proof_key=prior_proof_key,
    ) is not scorecard_ready:
        raise SystemExit(f"{label} recovery execution scorecard metadata validator diverged: {metadata}")
    if bool(metadata.get(readiness_key)) is not scorecard_ready:
        raise SystemExit(f"{label} recovery execution readiness should use strict scorecard validator: {metadata}")
    if bool(metadata.get(scorecard_ready_key)) is not scorecard_ready:
        raise SystemExit(f"{label} recovery execution scorecard ready flag should use strict scorecard validator: {metadata}")
    if metadata.get(prior_proof_key) is True and not scorecard_ready:
        raise SystemExit(f"{label} recovery execution prior proof should require strict scorecard readiness: {metadata}")
    if scorecard_ready:
        tampered_rows = [dict(row) for row in rows]
        tampered_rows[0]["item"] = "tampered"
        if _recovery_execution_readiness_scorecard_ready(tampered_rows):
            raise SystemExit(f"{label} recovery execution scorecard should reject item tampering: {metadata}")
        tampered_metadata = dict(metadata)
        tampered_metadata[f"{metadata_prefix}_scorecard_rows"] = tampered_rows
        if _recovery_execution_scorecard_metadata_ready(
            tampered_metadata,
            prefix=metadata_prefix,
            prior_proof_key=prior_proof_key,
        ):
            raise SystemExit(f"{label} recovery execution scorecard metadata accepted row tampering: {metadata}")
        for key, value in [
            (f"{metadata_prefix}_scorecard_row_count", 999),
            (f"{metadata_prefix}_score", 0),
            (f"{metadata_prefix}_max_score", 999),
            (readiness_key, False),
            (scorecard_ready_key, False),
        ]:
            tampered_metadata = dict(metadata)
            tampered_metadata[key] = value
            if _recovery_execution_scorecard_metadata_ready(
                tampered_metadata,
                prefix=metadata_prefix,
                prior_proof_key=prior_proof_key,
            ):
                raise SystemExit(f"{label} recovery execution scorecard metadata accepted stale mirror {key}: {metadata}")
    for suffix in [
        "authorizes_model_call",
        "authorizes_tool_execution",
        "authorizes_personal_data_read",
        "authorizes_external_side_effect",
    ]:
        key = f"{prefix}_{suffix}"
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should report {key}=False: {metadata}")
    if any(
        row.get("authorizes_model_call") is not False
        or row.get("authorizes_tool_execution") is not False
        or row.get("authorizes_personal_data_read") is not False
        or row.get("authorizes_external_side_effect") is not False
        for row in rows
    ):
        raise SystemExit(f"{label} recovery execution rows should block model/tool/private/external authority: {metadata}")


def assert_execution_health_proof_chain(metadata: dict, label: str) -> None:
    if metadata.get("execution_health_approval_proof_chains", {}).get("1") != EXPECTED_APPROVAL_CHAIN:
        raise SystemExit(f"{label} missed execution-health approval proof-chain metadata: {metadata}")
    if metadata.get("execution_health_approval_proof_chain_count", 0) < 1:
        raise SystemExit(f"{label} missed execution-health approval proof-chain count: {metadata}")
    for command in EXPECTED_APPROVAL_CHAIN:
        if command not in metadata.get("execution_health_next_commands", []):
            raise SystemExit(f"{label} missed proof-chain command in execution health queue: {command} in {metadata}")
    if metadata.get("execution_health_proof_queue") != metadata.get("execution_health_next_commands"):
        raise SystemExit(f"{label} execution-health proof queue should mirror next commands: {metadata}")
    if metadata.get("execution_health_proof_queue_count") != len(metadata.get("execution_health_proof_queue") or []):
        raise SystemExit(f"{label} execution-health proof queue count diverged: {metadata}")
    if metadata.get("execution_health_next_proof_command") != metadata.get("execution_health_next_command"):
        raise SystemExit(f"{label} execution-health next proof command diverged: {metadata}")
    for key in [
        "execution_health_recovery_closure_proof_queue",
        "execution_health_recovery_closure_proof_queue_count",
        "execution_health_recovery_closure_next_proof_command",
    ]:
        if key not in metadata:
            raise SystemExit(f"{label} missed explicit recovery-closure proof alias {key}: {metadata}")
    closure_commands = metadata.get("execution_health_recovery_closure_required_commands") or []
    if metadata.get("execution_health_recovery_closure_proof_queue") != closure_commands:
        raise SystemExit(f"{label} recovery-closure proof queue should mirror required commands: {metadata}")
    if metadata.get("execution_health_recovery_closure_proof_queue_count") != len(closure_commands):
        raise SystemExit(f"{label} recovery-closure proof queue count diverged: {metadata}")
    if closure_commands and metadata.get("execution_health_recovery_closure_next_proof_command") != closure_commands[0]:
        raise SystemExit(f"{label} missed next recovery-closure proof command: {metadata}")


def assert_execution_health_learning_handoff(metadata: dict, label: str) -> None:
    target_run_id = metadata.get("execution_health_learning_target_run_id")
    target_tool = metadata.get("execution_health_learning_target_tool")
    verify_command = metadata.get("execution_health_verification_handoff_command")
    recovery_command = metadata.get("execution_health_recovery_handoff_command")
    learning_command = metadata.get("execution_health_learning_handoff_command")
    learning_closure_command = metadata.get("execution_health_learning_closure_command")
    execution_learning_closure_command = metadata.get("execution_learning_closure_command")
    closure_state = metadata.get("execution_health_recovery_closure_state")
    closure_missing = metadata.get("execution_health_recovery_closure_missing")
    closure_required_commands = metadata.get("execution_health_recovery_closure_required_commands")
    if metadata.get("execution_health_learning_target_present") is not True:
        raise SystemExit(f"{label} should mark execution-health learning target present: {metadata}")
    if not target_run_id or not target_tool:
        raise SystemExit(f"{label} missed execution-health learning target: {metadata}")
    expected_verify = f"verification receipt {target_run_id}"
    expected_recovery = f"execution recovery packet {target_run_id}"
    expected_learning_closure = f"execution learning closure {target_run_id}"
    expected_learning = f"after-action learning packet {target_run_id}"
    if verify_command != expected_verify:
        raise SystemExit(f"{label} missed verification handoff command {expected_verify}: {metadata}")
    if recovery_command != expected_recovery:
        raise SystemExit(f"{label} missed recovery handoff command {expected_recovery}: {metadata}")
    if learning_command != expected_learning:
        raise SystemExit(f"{label} missed learning handoff command {expected_learning}: {metadata}")
    if learning_closure_command != expected_learning_closure:
        raise SystemExit(f"{label} missed execution learning closure command {expected_learning_closure}: {metadata}")
    if execution_learning_closure_command != expected_learning_closure:
        raise SystemExit(f"{label} execution learning closure command diverged from health closure command: {metadata}")
    for command in [verify_command, recovery_command, learning_closure_command, learning_command]:
        if command not in metadata.get("execution_health_next_commands", []):
            raise SystemExit(f"{label} missed learning handoff command in execution-health queue: {command} in {metadata}")
    if not closure_state or not isinstance(closure_missing, list) or not isinstance(closure_required_commands, list):
        raise SystemExit(f"{label} missed execution-health recovery closure metadata: {metadata}")
    if metadata.get("execution_health_recovery_closure_missing_count") != len(closure_missing):
        raise SystemExit(f"{label} recovery closure missing count diverged: {metadata}")
    if metadata.get("failed_action_runs", 0):
        if metadata.get("execution_health_recovery_closure_ready_to_retry") is not False:
            raise SystemExit(f"{label} should not be ready to retry while seeded closure blockers remain: {metadata}")
        if metadata.get("execution_health_recovery_closure_blocks_auto_execution") is not True:
            raise SystemExit(f"{label} should block auto-run while seeded recovery closure is incomplete: {metadata}")
        if metadata.get("execution_health_recovery_closure_checklist_command") != "recovery closure checklist":
            raise SystemExit(f"{label} missed recovery closure checklist command: {metadata}")
        if metadata.get("execution_health_recovery_closure_should_open_checklist") is not True:
            raise SystemExit(f"{label} should recommend opening the recovery closure checklist: {metadata}")
        if metadata.get("execution_learning_state") != "LEARNING_DEBT_AFTER_FAILURE":
            raise SystemExit(f"{label} missed execution learning debt state: {metadata}")
    elif metadata.get("execution_learning_state") not in {"LEARNING_REVIEW_REQUIRED", "LEARNING_LOOP_HAS_RECENT_ACTION_CONTEXT"}:
        raise SystemExit(f"{label} should keep non-failed execution learning state separate from failure debt: {metadata}")
    if metadata.get("execution_learning_blocks_completion_claim") is not True:
        raise SystemExit(f"{label} should block completion on execution learning debt: {metadata}")
    if metadata.get("execution_learning_target_run_id") != target_run_id:
        raise SystemExit(f"{label} execution learning target diverged from health target: {metadata}")
    for key in [
        "execution_learning_proof_queue",
        "execution_learning_proof_queue_count",
        "execution_learning_next_proof_command",
    ]:
        if key not in metadata:
            raise SystemExit(f"{label} missed explicit execution learning proof alias {key}: {metadata}")
    if metadata.get("execution_learning_missing_count", 0) < 1 or not metadata.get("execution_learning_missing"):
        raise SystemExit(f"{label} missed execution learning missing proofs: {metadata}")
    learning_commands = metadata.get("execution_learning_required_commands") or []
    if expected_learning_closure not in learning_commands:
        raise SystemExit(f"{label} missed execution learning closure command in execution learning queue: {metadata}")
    if not any(str(command).startswith("after-action learning packet") for command in learning_commands):
        raise SystemExit(f"{label} missed after-action learning command in execution learning queue: {metadata}")
    if learning_commands and metadata.get("execution_learning_next_required_command") != learning_commands[0]:
        raise SystemExit(f"{label} missed first execution learning proof command: {metadata}")
    if metadata.get("execution_learning_proof_queue") != learning_commands:
        raise SystemExit(f"{label} execution learning proof queue diverged from required commands: {metadata}")
    if metadata.get("execution_learning_proof_queue_count") != len(learning_commands):
        raise SystemExit(f"{label} execution learning proof queue count diverged: {metadata}")
    if metadata.get("execution_learning_next_proof_command") != metadata.get("execution_learning_next_required_command"):
        raise SystemExit(f"{label} execution learning next proof alias diverged: {metadata}")
    for command in learning_commands:
        if command not in metadata.get("execution_health_next_commands", []):
            raise SystemExit(f"{label} missed execution learning command in health queue: {command} in {metadata}")
    if closure_required_commands and metadata.get("execution_health_next_command") != closure_required_commands[0]:
        raise SystemExit(f"{label} should route next command to first recovery-closure proof: {metadata}")
    if closure_required_commands and metadata.get("execution_health_recovery_closure_next_required_command") != closure_required_commands[0]:
        raise SystemExit(f"{label} missed explicit first recovery-closure proof command: {metadata}")
    if closure_required_commands and metadata.get("execution_health_next_commands", [])[: len(closure_required_commands)] != closure_required_commands:
        raise SystemExit(f"{label} should place recovery-closure proof commands first in the queue: {metadata}")
    for command in closure_required_commands:
        if command and command not in metadata.get("execution_health_next_commands", []):
            raise SystemExit(f"{label} missed closure required command in execution-health queue: {command} in {metadata}")
    for key in [
        "execution_health_target_verification_receipts",
        "execution_health_target_recovery_packets",
        "execution_health_target_after_action_learning_packets",
    ]:
        if key not in metadata:
            raise SystemExit(f"{label} missed target closure count {key}: {metadata}")
    if metadata.get("execution_health_repeated_failure_count", 0) > 0:
        queue = metadata.get("execution_health_failure_promotion_queue") or []
        for key in [
            "execution_health_failure_promotion_command",
            "execution_health_failure_implementation_command",
            "execution_health_failure_apply_contract_command",
        ]:
            command = metadata.get(key)
            if not command:
                raise SystemExit(f"{label} missed execution-health failure promotion key {key}: {metadata}")
            if command not in queue or command not in metadata.get("execution_health_next_commands", []):
                raise SystemExit(f"{label} missed failure promotion command in queues: {command} in {metadata}")
        if metadata.get("execution_health_failure_promotion_queue_count") != len(queue):
            raise SystemExit(f"{label} missed failure promotion queue count: {metadata}")


def assert_doctor_handoff(metadata: dict, label: str) -> None:
    for key in [
        "doctor_next_audit_command",
        "doctor_completion_claim_state",
        "doctor_completion_claim_ready",
        "doctor_completion_blockers",
        "doctor_completion_blocker_count",
        "doctor_recent_failed_runs",
        "doctor_recent_approval_held_runs",
        "doctor_recent_verification_runs",
        "doctor_recent_after_action_learning_runs",
        "doctor_audit_readability_review_required",
        "doctor_audit_readability_review_commands",
        "doctor_audit_readability_review_command_count",
        "doctor_audit_readability_review_next_command",
        "doctor_unreadable_recent_tool_run_rows",
        "doctor_recovery_closure_state",
        "doctor_recovery_closure_missing",
        "doctor_recovery_closure_missing_count",
        "doctor_recovery_closure_required_commands",
        "doctor_recovery_closure_next_required_command",
        "doctor_recovery_closure_proof_queue",
        "doctor_recovery_closure_proof_queue_count",
        "doctor_recovery_closure_next_proof_command",
        "doctor_recovery_closure_ready_to_retry",
        "doctor_recovery_closure_blocks_completion_claim",
        "doctor_recovery_closure_checklist_command",
        "doctor_recovery_closure_should_open_checklist",
        "doctor_execution_learning_state",
        "doctor_execution_learning_missing",
        "doctor_execution_learning_missing_count",
        "doctor_execution_learning_required_commands",
        "doctor_execution_learning_next_required_command",
        "doctor_execution_learning_proof_queue",
        "doctor_execution_learning_proof_queue_count",
        "doctor_execution_learning_next_proof_command",
        "doctor_execution_learning_actionable_required_commands",
        "doctor_execution_learning_actionable_required_command_count",
        "doctor_execution_learning_actionable_next_required_command",
        "doctor_execution_learning_actionable_proof_queue",
        "doctor_execution_learning_actionable_proof_queue_count",
        "doctor_execution_learning_actionable_next_proof_command",
        "doctor_execution_learning_next_evidence_command",
        "doctor_execution_learning_blocks_completion_claim",
        "doctor_completion_proof_queue",
        "doctor_completion_proof_queue_count",
        "doctor_completion_next_proof_command",
        "doctor_agi_next_gate",
        "doctor_agi_next_selection_source",
        "doctor_agi_next_selection_reason",
        "doctor_agi_next_canonical_selector_command",
        "doctor_agi_next_deliberate_focus_override",
        "doctor_agi_next_target_title",
        "doctor_agi_next_build_command",
        "doctor_agi_next_evidence_closure_commands",
        "doctor_agi_next_evidence_closure_command_count",
        "doctor_agi_next_focused_verification_commands",
        "doctor_agi_next_focused_verification_command_count",
        "doctor_agi_next_likely_files",
        "doctor_agi_next_likely_file_count",
        "doctor_agi_next_target_file_integrity_status",
        "doctor_agi_next_target_files_checked",
        "doctor_agi_next_target_files_exist",
        "doctor_agi_next_missing_target_files",
        "doctor_agi_next_missing_target_file_count",
        "doctor_agi_next_target_integrity_blocks_start",
        "doctor_agi_next_acceptance_checks",
        "doctor_agi_next_acceptance_check_count",
        "doctor_agi_next_build_packet_ready_for_review",
        "doctor_agi_next_real_execution_gap_count",
        "doctor_agi_next_real_execution_gaps_by_gate",
        "doctor_agi_next_selected_real_execution_gap",
    ]:
        if key not in metadata:
            raise SystemExit(f"{label} missed doctor handoff key {key}: {metadata}")
    if not metadata.get("doctor_next_audit_command"):
        raise SystemExit(f"{label} missed doctor next audit command: {metadata}")
    if metadata.get("doctor_completion_claim_ready") is not False:
        raise SystemExit(f"{label} should block completion while seeded blockers remain: {metadata}")
    if metadata.get("doctor_completion_claim_state") != "BLOCKED":
        raise SystemExit(f"{label} missed blocked completion claim state: {metadata}")
    if metadata.get("doctor_completion_blocker_count", 0) < 1 or not metadata.get("doctor_completion_blockers"):
        raise SystemExit(f"{label} missed completion blockers: {metadata}")
    recovery_commands = metadata.get("doctor_recovery_closure_required_commands") or []
    recovery_proof_queue = metadata.get("doctor_recovery_closure_proof_queue") or []
    if metadata.get("failed_action_runs", 0):
        if metadata.get("doctor_recovery_closure_blocks_completion_claim") is not True:
            raise SystemExit(f"{label} missed doctor recovery closure completion blocker: {metadata}")
        if metadata.get("doctor_recovery_closure_checklist_command") != "recovery closure checklist":
            raise SystemExit(f"{label} doctor missed recovery closure checklist command: {metadata}")
        if metadata.get("doctor_recovery_closure_should_open_checklist") is not True:
            raise SystemExit(f"{label} doctor should recommend opening recovery closure checklist: {metadata}")
        if not recovery_commands:
            raise SystemExit(f"{label} missed doctor recovery closure proof queue: {metadata}")
        if metadata.get("doctor_recovery_closure_next_required_command") != recovery_commands[0]:
            raise SystemExit(f"{label} missed doctor recovery closure next proof command: {metadata}")
        if recovery_proof_queue != recovery_commands:
            raise SystemExit(f"{label} doctor recovery closure proof queue diverged: {metadata}")
    elif metadata.get("doctor_recovery_closure_blocks_completion_claim") is not False:
        raise SystemExit(f"{label} should not mark approval-held-only state as recovery closure debt: {metadata}")
    if metadata.get("failed_action_runs", 0):
        if metadata.get("doctor_recovery_closure_proof_queue_count") != len(recovery_proof_queue):
            raise SystemExit(f"{label} doctor recovery closure proof queue count diverged: {metadata}")
        if metadata.get("doctor_recovery_closure_next_proof_command") != recovery_proof_queue[0]:
            raise SystemExit(f"{label} doctor missed next recovery closure proof alias: {metadata}")
        if metadata.get("doctor_recovery_closure_missing_count", 0) < 1 or not metadata.get("doctor_recovery_closure_missing"):
            raise SystemExit(f"{label} missed doctor recovery closure missing proofs: {metadata}")
        for command in ["verification receipt", "execution recovery packet", "after-action learning packet"]:
            if not any(str(item).startswith(command) for item in recovery_commands):
                raise SystemExit(f"{label} doctor recovery queue missed {command}: {metadata}")
        if not any("recovery closure" in str(blocker) for blocker in metadata.get("doctor_completion_blockers", [])):
            raise SystemExit(f"{label} doctor completion blockers missed recovery closure: {metadata}")
    if metadata.get("doctor_execution_learning_blocks_completion_claim") is not True:
        raise SystemExit(f"{label} doctor missed execution learning completion blocker: {metadata}")
    if metadata.get("doctor_execution_learning_missing_count", 0) < 1 or not metadata.get("doctor_execution_learning_missing"):
        raise SystemExit(f"{label} doctor missed execution learning missing proofs: {metadata}")
    learning_commands = metadata.get("doctor_execution_learning_required_commands") or []
    learning_proof_queue = metadata.get("doctor_execution_learning_proof_queue") or []
    learning_actionable_commands = metadata.get("doctor_execution_learning_actionable_required_commands") or []
    learning_actionable_queue = metadata.get("doctor_execution_learning_actionable_proof_queue") or []
    if not learning_commands or metadata.get("doctor_execution_learning_next_required_command") != learning_commands[0]:
        raise SystemExit(f"{label} doctor missed execution learning proof queue: {metadata}")
    if learning_proof_queue != learning_commands:
        raise SystemExit(f"{label} doctor execution learning proof queue diverged: {metadata}")
    if metadata.get("doctor_execution_learning_proof_queue_count") != len(learning_proof_queue):
        raise SystemExit(f"{label} doctor execution learning proof queue count diverged: {metadata}")
    if metadata.get("doctor_execution_learning_next_proof_command") != learning_proof_queue[0]:
        raise SystemExit(f"{label} doctor execution learning next proof alias diverged: {metadata}")
    if not learning_actionable_commands:
        raise SystemExit(f"{label} doctor missed execution learning actionable queue: {metadata}")
    if learning_actionable_queue != learning_actionable_commands:
        raise SystemExit(f"{label} doctor execution learning actionable queue diverged: {metadata}")
    if metadata.get("doctor_execution_learning_actionable_required_command_count") != len(learning_actionable_commands):
        raise SystemExit(f"{label} doctor execution learning actionable required count diverged: {metadata}")
    if metadata.get("doctor_execution_learning_actionable_proof_queue_count") != len(learning_actionable_queue):
        raise SystemExit(f"{label} doctor execution learning actionable proof count diverged: {metadata}")
    if metadata.get("doctor_execution_learning_actionable_next_required_command") != learning_actionable_queue[0]:
        raise SystemExit(f"{label} doctor execution learning actionable next required diverged: {metadata}")
    if metadata.get("doctor_execution_learning_actionable_next_proof_command") != learning_actionable_queue[0]:
        raise SystemExit(f"{label} doctor execution learning actionable next proof diverged: {metadata}")
    if metadata.get("doctor_execution_learning_next_evidence_command") != learning_actionable_queue[0]:
        raise SystemExit(f"{label} doctor execution learning next evidence command diverged: {metadata}")
    after_action_indexes = [
        index
        for index, command in enumerate(learning_actionable_queue)
        if str(command).startswith("after-action learning packet")
    ]
    closure_indexes = [
        index
        for index, command in enumerate(learning_actionable_queue)
        if str(command).startswith("execution learning closure")
    ]
    if after_action_indexes and closure_indexes and after_action_indexes[0] > closure_indexes[0]:
        raise SystemExit(f"{label} doctor actionable queue should put after-action evidence before closure: {metadata}")
    if not any("execution learning debt" in str(blocker) for blocker in metadata.get("doctor_completion_blockers", [])):
        raise SystemExit(f"{label} doctor blockers missed execution learning debt: {metadata}")
    completion_proof_queue = metadata.get("doctor_completion_proof_queue") or []
    if not completion_proof_queue:
        raise SystemExit(f"{label} doctor missed completion proof queue: {metadata}")
    if metadata.get("doctor_completion_proof_queue_count") != len(completion_proof_queue):
        raise SystemExit(f"{label} doctor completion proof queue count diverged: {metadata}")
    if metadata.get("doctor_completion_next_proof_command") != completion_proof_queue[0]:
        raise SystemExit(f"{label} doctor missed first completion proof command: {metadata}")
    audit_readability_commands = metadata.get("doctor_audit_readability_review_commands") or []
    if metadata.get("doctor_audit_readability_review_required"):
        expected_audit_readability = ["storage status", "recent tool runs", "execution health report"]
        if audit_readability_commands != expected_audit_readability:
            raise SystemExit(f"{label} doctor audit-readability queue diverged: {metadata}")
        if metadata.get("doctor_audit_readability_review_next_command") != audit_readability_commands[0]:
            raise SystemExit(f"{label} doctor missed audit-readability next command: {metadata}")
        if metadata.get("doctor_unreadable_recent_tool_run_rows", 0) < 1:
            raise SystemExit(f"{label} doctor missed unreadable recent tool-run count: {metadata}")
        if not any("unreadable recent tool run" in str(blocker) for blocker in metadata.get("doctor_completion_blockers", [])):
            raise SystemExit(f"{label} doctor blockers missed audit-readability issue: {metadata}")
    else:
        if audit_readability_commands or metadata.get("doctor_audit_readability_review_next_command"):
            raise SystemExit(f"{label} doctor exposed audit-readability commands without required flag: {metadata}")
    if metadata.get("doctor_audit_readability_review_command_count") != len(audit_readability_commands):
        raise SystemExit(f"{label} doctor audit-readability queue count diverged: {metadata}")
    for command in audit_readability_commands:
        if command not in completion_proof_queue:
            raise SystemExit(f"{label} doctor completion proof queue missed audit-readability command {command!r}: {metadata}")
    for command in recovery_proof_queue + learning_actionable_queue:
        if command not in completion_proof_queue:
            raise SystemExit(f"{label} doctor completion proof queue missed proof command {command!r}: {metadata}")
    after_action_indexes = [
        index
        for index, command in enumerate(completion_proof_queue)
        if str(command).startswith("after-action learning packet")
    ]
    closure_indexes = [
        index
        for index, command in enumerate(completion_proof_queue)
        if str(command).startswith("execution learning closure")
    ]
    if after_action_indexes and closure_indexes and after_action_indexes[0] > closure_indexes[0]:
        raise SystemExit(f"{label} doctor completion proof queue should put after-action evidence before closure: {metadata}")
    approval_indexes = [
        index
        for index, command in enumerate(completion_proof_queue)
        if str(command).startswith("approval readiness")
    ]
    if metadata.get("pending_approvals") and approval_indexes and after_action_indexes and approval_indexes[0] > after_action_indexes[0]:
        raise SystemExit(f"{label} doctor completion proof queue should put pending approval review before learning evidence: {metadata}")
    for command in ["harness completion", "completion audit", "evidence ledger", "completion claim gate"]:
        if command not in completion_proof_queue:
            raise SystemExit(f"{label} doctor completion proof queue missed {command}: {metadata}")
    if not metadata.get("doctor_agi_next_target_title"):
        raise SystemExit(f"{label} missed doctor selected AGI target title: {metadata}")
    if not str(metadata.get("doctor_agi_next_build_command") or "").startswith("agi next build move: "):
        raise SystemExit(f"{label} missed doctor selected AGI build command: {metadata}")
    if metadata.get("doctor_agi_next_gate") not in str(metadata.get("doctor_agi_next_build_command") or ""):
        raise SystemExit(f"{label} doctor selected AGI gate diverged from build command: {metadata}")
    if metadata.get("doctor_agi_next_canonical_selector_command") != metadata.get("doctor_agi_next_build_command"):
        raise SystemExit(f"{label} doctor selected AGI canonical selector command diverged: {metadata}")
    if metadata.get("doctor_agi_next_deliberate_focus_override") is not False:
        raise SystemExit(f"{label} doctor selected AGI should not claim operator focus override: {metadata}")
    if metadata.get("doctor_agi_next_selection_source") not in {
        "next_step_doctor_default_gate",
        "harness_dynamic_registry",
        "next_step_doctor_default_gate_after_selector_error",
    }:
        raise SystemExit(f"{label} doctor selected AGI source is not recognized: {metadata}")
    if metadata.get("doctor_agi_next_likely_file_count") != len(metadata.get("doctor_agi_next_likely_files", [])):
        raise SystemExit(f"{label} doctor selected AGI file count diverged: {metadata}")
    if metadata.get("doctor_agi_next_evidence_closure_command_count") != len(metadata.get("doctor_agi_next_evidence_closure_commands", [])):
        raise SystemExit(f"{label} doctor selected AGI closure count diverged: {metadata}")
    if metadata.get("doctor_agi_next_focused_verification_command_count") != len(metadata.get("doctor_agi_next_focused_verification_commands", [])):
        raise SystemExit(f"{label} doctor selected AGI verification count diverged: {metadata}")
    if metadata.get("doctor_agi_next_acceptance_check_count") != len(metadata.get("doctor_agi_next_acceptance_checks", [])):
        raise SystemExit(f"{label} doctor selected AGI acceptance count diverged: {metadata}")
    if bool(metadata.get("doctor_agi_next_target_integrity_blocks_start")) is metadata.get("doctor_agi_next_target_files_exist"):
        raise SystemExit(f"{label} doctor selected AGI target-integrity blocker diverged: {metadata}")
    expected_ready = bool(
        metadata.get("doctor_agi_next_target_files_exist")
        and metadata.get("doctor_agi_next_focused_verification_commands")
        and metadata.get("doctor_agi_next_acceptance_checks")
    )
    if metadata.get("doctor_agi_next_build_packet_ready_for_review") is not expected_ready:
        raise SystemExit(f"{label} doctor selected AGI readiness flag diverged: {metadata}")
    if metadata.get("doctor_agi_next_real_execution_gap_count") != len(metadata.get("doctor_agi_next_real_execution_gaps_by_gate") or {}):
        raise SystemExit(f"{label} doctor selected AGI real-execution gap count diverged: {metadata}")
    if metadata.get("doctor_agi_next_real_execution_gap_count", 0) <= 0:
        raise SystemExit(f"{label} should expose remaining AGI real-execution gaps: {metadata}")
    if metadata.get("doctor_agi_next_gate") not in (metadata.get("doctor_agi_next_real_execution_gaps_by_gate") or {}):
        raise SystemExit(f"{label} selected AGI gate missing from real-execution gaps: {metadata}")
    if not metadata.get("doctor_agi_next_selected_real_execution_gap"):
        raise SystemExit(f"{label} missed selected AGI real-execution gap detail: {metadata}")


def assert_doctor_learning_prose(response: str, label: str) -> None:
    actionable = "execution learning actionable next required"
    ordered = "execution learning ordered gate required"
    actionable_queue = "execution learning actionable proof queue"
    legacy_queue = "execution learning proof queue"
    for item in [actionable, ordered, actionable_queue, legacy_queue]:
        if item not in response:
            raise SystemExit(f"{label} doctor handoff missed visible {item!r}.")
    for stale in [
        "recovery closure next proof:",
        "execution learning actionable next proof:",
        "execution learning ordered gate proof:",
        "next actionable learning proof:",
        "ordered learning gate proof:",
        "next proof command:",
    ]:
        if stale in response:
            raise SystemExit(f"{label} doctor handoff should not use ambiguous {stale!r} prose.")
    if "recovery closure next required:" not in response:
        raise SystemExit(f"{label} doctor handoff missed recovery closure next required prose.")
    actionable_index = response.find(actionable)
    ordered_index = response.find(ordered)
    if actionable_index == -1 or ordered_index == -1 or actionable_index > ordered_index:
        raise SystemExit(f"{label} doctor handoff should show actionable learning proof before ordered gate proof.")
    actionable_queue_index = response.find(actionable_queue)
    legacy_queue_index = response.find(legacy_queue)
    if actionable_queue_index == -1 or legacy_queue_index == -1 or actionable_queue_index > legacy_queue_index:
        raise SystemExit(f"{label} doctor handoff should show actionable learning queue before legacy proof queue.")


def assert_approval_handoff(metadata: dict, label: str) -> None:
    expected = {
        "approval_handoff_pending_count": 1,
        "approval_handoff_first_id": 1,
        "approval_handoff_next_command": "approval readiness 1",
        "approval_handoff_readiness_command": "approval readiness 1",
        "approval_handoff_last_look_command": "approval packet 1",
        "approval_handoff_proof_command": "approval chain proof 1",
        "approval_handoff_verification_command": "verification receipt <approved run id from approval chain proof 1>",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise SystemExit(f"{label} missed approval handoff {key}={value}: {metadata}")
    if metadata.get("approval_handoff_first_tool_name") != "run_shell_command":
        raise SystemExit(f"{label} missed first approval tool name: {metadata}")
    if metadata.get("approval_handoff_first_staleness") not in {"fresh", "review_again", "stale", "unknown"}:
        raise SystemExit(f"{label} missed first approval staleness: {metadata}")
    if metadata.get("approval_handoff_proof_chain_commands") != EXPECTED_APPROVAL_CHAIN:
        raise SystemExit(f"{label} missed approval handoff proof chain: {metadata}")
    decision_matrix = metadata.get("approval_handoff_decision_matrix", [])
    for expected in ["exact stored request", "verifiable", "dismiss if stale"]:
        if not any(expected in item for item in decision_matrix):
            raise SystemExit(f"{label} missed approval handoff decision criterion {expected}: {metadata}")


def assert_diagnostic_only_runs_do_not_create_learning_debt(root: Path) -> None:
    runtime = make_temp_runtime(root)
    diagnostic_tools = [
        "storage_status",
        "storage_recovery_check",
        "readiness_report",
        "completion_claim_gate",
        "harness_readiness_digest",
        "jarvis_doctor",
    ]
    for tool_name in diagnostic_tools:
        runtime.store.log_tool_run(
            runtime.session_id,
            tool_name,
            "READ_ONLY",
            True,
            False,
            f"{tool_name} diagnostic proof",
            metadata={"diagnostic_only": True},
        )

    safe_next_actions, _next_action_packet, _priority_stack, _continuation_packet, _build_target_packet, _harness_build_slice, _save_build_target_packet, _export_mission_control, work_queue, _save_work_queue = make_next_step_tools(
        runtime.store,
        runtime.vault,
    )
    cases = [
        ("diagnostic-only safe next actions", safe_next_actions({"limit": 5})),
        ("diagnostic-only work queue", work_queue({"limit": 5})),
    ]
    for label, result in cases:
        metadata = result.metadata
        if metadata.get("execution_learning_state") != "NO_RECENT_ACTION_RUNS":
            raise SystemExit(f"{label} counted diagnostic rows as execution learning debt: {metadata}")
        if metadata.get("execution_learning_blocks_completion_claim") is not False:
            raise SystemExit(f"{label} should not block completion on diagnostic-only rows: {metadata}")
        if metadata.get("execution_learning_recent_action_runs") != 0:
            raise SystemExit(f"{label} should report zero recent action runs: {metadata}")
        if metadata.get("execution_learning_target_run_id") is not None:
            raise SystemExit(f"{label} should not pick a diagnostic learning target: {metadata}")
        if metadata.get("execution_health_learning_target_present") is not False:
            raise SystemExit(f"{label} should mark execution-health learning target absent: {metadata}")
        if metadata.get("execution_health_learning_handoff_command") not in {"", None}:
            raise SystemExit(f"{label} should not expose target-specific learning handoff command: {metadata}")
        if metadata.get("execution_health_learning_closure_command") not in {"", None}:
            raise SystemExit(f"{label} should not expose target-specific learning closure command: {metadata}")
        if metadata.get("execution_health_execution_learning_closure_command") not in {"", None}:
            raise SystemExit(f"{label} should not expose target-specific execution learning closure command: {metadata}")
        if metadata.get("execution_learning_next_proof_command") not in {"", None}:
            raise SystemExit(f"{label} should not require execution learning proof: {metadata}")
        if "- target: none in reviewed window" not in result.output:
            raise SystemExit(f"{label} should make the no-target handoff explicit: {result.output}")
        if "latest relevant action run" in result.output:
            raise SystemExit(f"{label} should not imply a relevant action target exists: {result.output}")
        if "close learning gate: `execution learning closure`" in result.output:
            raise SystemExit(f"{label} should not describe target-specific learning closure without a target: {result.output}")
        if any(str(command).startswith("execution learning closure") for command in metadata.get("execution_health_next_commands", [])):
            raise SystemExit(f"{label} leaked execution learning closure into health queue: {metadata}")
        if metadata.get("doctor_execution_learning_state") != "NO_RECENT_ACTION_RUNS":
            raise SystemExit(f"{label} doctor handoff counted diagnostic rows as learning debt: {metadata}")
        if metadata.get("doctor_execution_learning_blocks_completion_claim") is not False:
            raise SystemExit(f"{label} doctor handoff should not block on diagnostic-only rows: {metadata}")
        if metadata.get("doctor_execution_learning_next_proof_command") not in {"", None}:
            raise SystemExit(f"{label} doctor handoff should not require learning proof: {metadata}")


def assert_next_step_doctor_separates_approval_held_runs(root: Path) -> None:
    runtime = make_temp_runtime(root)
    failed_run_id = runtime.store.log_tool_run(
        runtime.session_id,
        "send_kakao",
        "HIGH_RISK",
        False,
        True,
        "transport timed out before delivery",
        metadata={"failure_kind": "transport_error", "failure_stage": "transport_timeout"},
    )
    approval_held_run_id = runtime.store.log_tool_run(
        runtime.session_id,
        "send_telegram",
        "HIGH_RISK",
        False,
        False,
        "raw approval held marker should stay private",
        approval_id=7,
        metadata={"failure_kind": "approval-gate", "requires_confirmation": True},
    )
    (
        safe_next_actions,
        next_action_packet,
        priority_stack,
        _continuation_packet,
        _build_target_packet,
        _harness_build_slice,
        _save_build_target_packet,
        _export_mission_control,
        work_queue,
        _save_work_queue,
    ) = make_next_step_tools(
        runtime.store,
        runtime.vault,
    )
    cases = [
        ("approval-held safe next actions", safe_next_actions({"limit": 5})),
        ("approval-held next action packet", next_action_packet({"limit": 5})),
        ("approval-held priority stack", priority_stack({"limit": 5})),
        ("approval-held work queue", work_queue({"limit": 5})),
    ]
    for label, result in cases:
        metadata = result.metadata
        if metadata.get("doctor_recent_failed_runs") != 1:
            raise SystemExit(f"{label} should count only the true transport failure as failed: {metadata}")
        if metadata.get("doctor_recent_approval_held_runs") != 1:
            raise SystemExit(f"{label} should count approval-held run separately: {metadata}")
        if metadata.get("failed_action_runs") != 1:
            raise SystemExit(f"{label} execution health should count only the true failed action: {metadata}")
        if metadata.get("approval_held_action_runs") != 1:
            raise SystemExit(f"{label} should expose approval-held action count: {metadata}")
        if metadata.get("execution_health_approval_held_action_runs") != 1:
            raise SystemExit(f"{label} missed execution-health approval-held count: {metadata}")
        if metadata.get("execution_health_learning_target_run_id") != failed_run_id:
            raise SystemExit(f"{label} should keep recovery learning target on true failure: {metadata}")
        if metadata.get("execution_health_recovery_closure_target_run_id") != failed_run_id:
            raise SystemExit(f"{label} should keep recovery closure target on true failure: {metadata}")
        next_commands = metadata.get("execution_health_next_commands") or []
        if f"execution recovery packet {approval_held_run_id}" in next_commands:
            raise SystemExit(f"{label} should not create recovery packet for approval-held run: {metadata}")
        if f"verification receipt {approval_held_run_id}" in next_commands:
            raise SystemExit(f"{label} should not verify approval-held run as recovery target: {metadata}")
        if f"execution recovery packet {failed_run_id}" not in next_commands:
            raise SystemExit(f"{label} should keep true failure recovery command: {metadata}")
        blockers = metadata.get("doctor_completion_blockers") or []
        if "1 recent failed/blocked run(s)" not in blockers:
            raise SystemExit(f"{label} missed true failure blocker: {metadata}")
        if "1 recent approval-held run(s) need approval review" not in blockers:
            raise SystemExit(f"{label} missed approval-held blocker: {metadata}")
        if "2 recent failed/blocked run(s)" in blockers:
            raise SystemExit(f"{label} mislabeled approval-held run as failed: {metadata}")
        proof_queue = metadata.get("doctor_completion_proof_queue") or []
        for command in [
            "approval readiness 7",
            "approval packet 7",
            "approval chain proof 7",
            "verification receipt <approved run id from approval chain proof 7>",
        ]:
            if command not in proof_queue:
                raise SystemExit(f"{label} missed approval-held proof command {command!r}: {metadata}")
        if "recent failed/blocked runs: 1" not in result.output:
            raise SystemExit(f"{label} output missed failed-run count: {result.output}")
        if "recent approval-held runs: 1" not in result.output:
            raise SystemExit(f"{label} output missed approval-held-run count: {result.output}")
        if "Review recovery for run #2: send_telegram" in result.output:
            raise SystemExit(f"{label} should not recommend approval-held run as recovery target: {result.output}")
        if "approval-held action runs: 1" not in result.output and label in {
            "approval-held next action packet",
            "approval-held priority stack",
        }:
            raise SystemExit(f"{label} output missed approval-held action count: {result.output}")
        if "raw approval held marker should stay private" in result.output:
            raise SystemExit(f"{label} leaked raw approval-held audit output: {result.output}")


def assert_registry_aware_doctor_agi_selection(root: Path) -> None:
    class FakeTool:
        def __init__(self, name: str) -> None:
            self.name = name

    tool_names = {
        "brain_think",
        "chat_prompt_preview",
        "architecture_map",
    }
    expected = _selected_agi_target_readiness(_agi_gate_summary(tool_names))
    if expected.get("gate_name") == "personal integrations":
        raise SystemExit(f"Registry-aware selector fixture should not collapse to the doctor fallback: {expected}")

    runtime = make_temp_runtime(root)
    (
        safe_next_actions,
        _next_action_packet,
        _priority_stack,
        _continuation_packet,
        _build_target_packet,
        _harness_build_slice,
        _save_build_target_packet,
        _export_mission_control,
        work_queue,
        _save_work_queue,
    ) = make_next_step_tools(
        runtime.store,
        runtime.vault,
        lambda: [FakeTool(name) for name in sorted(tool_names)],
    )
    for label, result in [
        ("registry-aware safe next actions", safe_next_actions({"limit": 5})),
        ("registry-aware work queue", work_queue({"limit": 5})),
    ]:
        metadata = result.metadata
        expected_command = expected["closure_commands"][0]
        if metadata.get("doctor_agi_next_selection_source") != "harness_dynamic_registry":
            raise SystemExit(f"{label} did not use dynamic registry selection: {metadata}")
        if metadata.get("doctor_agi_next_gate") != expected.get("gate_name"):
            raise SystemExit(f"{label} selected AGI gate diverged from harness selector: {metadata}")
        if metadata.get("doctor_agi_next_build_command") != expected_command:
            raise SystemExit(f"{label} selected AGI build command diverged from harness selector: {metadata}")
        if expected_command not in metadata.get("doctor_agi_next_evidence_closure_commands", []):
            raise SystemExit(f"{label} closure commands missed selected AGI build command: {metadata}")
        if metadata.get("doctor_agi_next_real_execution_gap_count") != 6:
            raise SystemExit(f"{label} missed AGI real-execution gap count: {metadata}")
        if "AGI real-execution gaps still tracked: 6" not in result.output:
            raise SystemExit(f"{label} output missed AGI real-execution gap count: {result.output}")


def assert_storage_fallback_drives_next_step_completion_proof(root: Path) -> None:
    runtime = make_temp_runtime(root)

    def storage_fallback() -> dict[str, object]:
        return {
            "reason": "primary_storage_not_writable",
            "exception_type": "OperationalError",
            "db_path_display": "workspace-local fallback database",
            "vault_path_display": "workspace-local fallback notes",
        }

    (
        safe_next_actions,
        _next_action_packet,
        _priority_stack,
        _continuation_packet,
        _build_target_packet,
        _harness_build_slice,
        _save_build_target_packet,
        _export_mission_control,
        work_queue,
        _save_work_queue,
    ) = make_next_step_tools(
        runtime.store,
        runtime.vault,
        None,
        storage_fallback,
        runtime.config,
    )
    expected_commands = [
        "storage status",
        STORAGE_RECOVERY_CHECK_COMMAND,
        "storage status",
    ]
    for label, result in [
        ("storage fallback safe next actions", safe_next_actions({"limit": 5})),
        ("storage fallback work queue", work_queue({"limit": 5})),
    ]:
        metadata = result.metadata
        if metadata.get("doctor_storage_runtime_fallback_active") is not True:
            raise SystemExit(f"{label} missed storage fallback active metadata: {metadata}")
        if metadata.get("doctor_storage_readiness_blocks_completion_claim") is not True:
            raise SystemExit(f"{label} should block completion on storage fallback: {metadata}")
        if metadata.get("doctor_storage_recovery_required") is not True:
            raise SystemExit(f"{label} missed storage recovery required flag: {metadata}")
        if metadata.get("doctor_storage_recovery_mode") != "restart_runtime_to_configured_storage":
            raise SystemExit(f"{label} missed restart-only storage recovery mode: {metadata}")
        if metadata.get("doctor_storage_recovery_restart_required") is not True:
            raise SystemExit(f"{label} missed storage restart-required flag: {metadata}")
        if "restart or reload Jarvis with the configured durable storage envs" not in metadata.get("doctor_storage_recovery_next_operator_action", ""):
            raise SystemExit(f"{label} missed restart-only operator action: {metadata}")
        if metadata.get("doctor_storage_readiness_next_commands") != expected_commands:
            raise SystemExit(f"{label} storage recovery queue diverged: {metadata}")
        if metadata.get("doctor_storage_readiness_next_command_count") != len(expected_commands):
            raise SystemExit(f"{label} storage recovery queue count diverged: {metadata}")
        assert_doctor_storage_handoff_contract(metadata, label, expected_commands)
        if metadata.get("doctor_storage_recovery_check_tool_command") != expected_commands[1]:
            raise SystemExit(f"{label} missed storage recovery check tool command: {metadata}")
        if metadata.get("doctor_storage_recovery_check_command") != BOOTSTRAP_CHECK_COMMAND:
            raise SystemExit(f"{label} missed storage recovery shell check command: {metadata}")
        if metadata.get("doctor_storage_recovery_check_api") != STORAGE_RECOVERY_CHECK_API:
            raise SystemExit(f"{label} missed storage recovery check API: {metadata}")
        if metadata.get("doctor_storage_recovery_command") != BOOTSTRAP_WRITE_COMMAND:
            raise SystemExit(f"{label} missed storage recovery write command: {metadata}")
        if BOOTSTRAP_CHECK_COMMAND in expected_commands or BOOTSTRAP_WRITE_COMMAND in expected_commands:
            raise SystemExit(f"{label} restart-only storage queue should not require bootstrap commands: {metadata}")
        proof_queue = metadata.get("doctor_completion_proof_queue") or []
        if proof_queue[: len(expected_commands)] != expected_commands:
            raise SystemExit(f"{label} completion proof queue should preserve full storage recovery ladder: {metadata}")
        if metadata.get("doctor_completion_next_proof_command") != "storage status":
            raise SystemExit(f"{label} next completion proof should be storage status: {metadata}")
        if "runtime is using workspace-local fallback storage" not in " ".join(str(item) for item in metadata.get("doctor_completion_blockers", [])):
            raise SystemExit(f"{label} missed storage blocker in doctor completion blockers: {metadata}")
        handoff = metadata.get("safe_next_actions_handoff") or metadata.get("work_queue_handoff") or {}
        next_commands = handoff.get("next_commands") or []
        if "storage status" not in next_commands:
            raise SystemExit(f"{label} operator handoff missed storage status: {metadata}")
        if STORAGE_RECOVERY_CHECK_COMMAND not in next_commands:
            raise SystemExit(f"{label} operator handoff missed native storage recovery check: {metadata}")
        if next_commands.index("storage status") > next_commands.index(STORAGE_RECOVERY_CHECK_COMMAND):
            raise SystemExit(f"{label} storage status should precede native storage recovery check: {metadata}")
        if BOOTSTRAP_CHECK_COMMAND in next_commands or BOOTSTRAP_WRITE_COMMAND in next_commands:
            raise SystemExit(f"{label} restart-only operator handoff should not require bootstrap commands: {metadata}")
        if next_commands.index("storage status") > next_commands.index("safe next actions") if "safe next actions" in next_commands else False:
            raise SystemExit(f"{label} storage status should not trail safe-next refresh commands: {metadata}")
        if label.endswith("safe next actions") and metadata.get("safe_next_actions_start_kind") != "storage_recovery":
            raise SystemExit(f"{label} should make storage recovery the start kind: {metadata}")
        if label.endswith("work queue"):
            if metadata.get("work_queue_start_kind") != "storage_recovery" or metadata.get("work_queue_start_command") != "storage status":
                raise SystemExit(f"{label} should make storage recovery the start command: {metadata}")
            if metadata.get("work_queue_next_commands", [None, None, None])[3] != "storage status":
                raise SystemExit(f"{label} should put storage status immediately after context/safety review: {metadata}")
            if metadata.get("work_queue_next_commands", [None, None, None, None, None])[4] != STORAGE_RECOVERY_CHECK_COMMAND:
                raise SystemExit(f"{label} should put native storage check after storage status: {metadata}")
            if "safe next actions" not in metadata.get("work_queue_next_commands", []):
                raise SystemExit(f"{label} should retain safe-next refresh after restart-only storage proof: {metadata}")
        if "storage recovery queue" not in result.output:
            raise SystemExit(f"{label} output should expose storage recovery queue: {result.output}")
        if "restart or reload Jarvis with the configured durable storage envs" not in result.output:
            raise SystemExit(f"{label} output should expose restart-only operator action: {result.output}")

    def storage_fallback_with_primary_diagnostics() -> dict[str, object]:
        return {
            "reason": "primary_storage_not_writable",
            "exception_type": "OperationalError",
            "db_path_display": "workspace-local fallback database",
            "vault_path_display": "workspace-local fallback notes",
            "primary_storage_diagnostics": {
                "available": False,
                "status": "needs attention",
                "issues": [
                    "database parent is not writable",
                    "database file is not writable",
                    "Obsidian vault is not writable",
                ],
                "metadata_only": True,
            },
        }

    (
        repair_safe_next_actions,
        _repair_next_action_packet,
        _repair_priority_stack,
        _repair_continuation_packet,
        _repair_build_target_packet,
        _repair_harness_build_slice,
        _repair_save_build_target_packet,
        _repair_export_mission_control,
        repair_work_queue,
        _repair_save_work_queue,
    ) = make_next_step_tools(
        runtime.store,
        runtime.vault,
        None,
        storage_fallback_with_primary_diagnostics,
        runtime.config,
    )
    repair_expected_commands = [
        "storage status",
        STORAGE_RECOVERY_PLAN_COMMAND,
        STORAGE_RECOVERY_CHECK_COMMAND,
        BOOTSTRAP_CHECK_COMMAND,
        BOOTSTRAP_WRITE_COMMAND,
    ]
    for label, result in [
        ("storage fallback repair safe next actions", repair_safe_next_actions({"limit": 5})),
        ("storage fallback repair work queue", repair_work_queue({"limit": 5})),
    ]:
        metadata = result.metadata
        if metadata.get("doctor_storage_recovery_mode") != "repair_configured_storage_then_restart_runtime":
            raise SystemExit(f"{label} should use repair mode from preserved primary diagnostics: {metadata}")
        if metadata.get("doctor_storage_readiness_next_commands") != repair_expected_commands:
            raise SystemExit(f"{label} repair-mode storage queue diverged: {metadata}")
        assert_doctor_storage_handoff_contract(metadata, label, repair_expected_commands)
        if metadata.get("doctor_storage_recovery_next_operator_action") != "review the storage recovery plan, point Jarvis at writable durable storage, run the no-write storage check, then restart or reload Jarvis":
            raise SystemExit(f"{label} repair-mode operator action drifted: {metadata}")
        proof_queue = metadata.get("doctor_completion_proof_queue") or []
        if proof_queue[: len(repair_expected_commands)] != repair_expected_commands:
            raise SystemExit(f"{label} proof queue should preserve repair-mode storage ladder: {metadata}")
        handoff = metadata.get("safe_next_actions_handoff") or metadata.get("work_queue_handoff") or {}
        next_commands = handoff.get("next_commands") or []
        for command in repair_expected_commands:
            if command not in next_commands:
                raise SystemExit(f"{label} operator handoff missed repair command {command}: {metadata}")


def assert_unhealthy_configured_storage_blocks_next_step_completion_proof(root: Path) -> None:
    runtime = make_temp_runtime(root)
    unhealthy_config = JarvisConfig(
        data_dir=root / "missing-data" / "child",
        db_path=root / "missing-data" / "child" / "jarvis.sqlite",
        obsidian_vault=root / "missing-vault" / "child",
        obsidian_root="Jarvis",
        use_model_planner=False,
    )
    (
        safe_next_actions,
        _next_action_packet,
        _priority_stack,
        _continuation_packet,
        _build_target_packet,
        _harness_build_slice,
        _save_build_target_packet,
        _export_mission_control,
        work_queue,
        _save_work_queue,
    ) = make_next_step_tools(
        runtime.store,
        runtime.vault,
        None,
        None,
        unhealthy_config,
    )
    expected_commands = [
        "storage status",
        STORAGE_RECOVERY_PLAN_COMMAND,
        STORAGE_RECOVERY_CHECK_COMMAND,
        BOOTSTRAP_CHECK_COMMAND,
        BOOTSTRAP_WRITE_COMMAND,
    ]
    for label, result in [
        ("unhealthy configured storage safe next actions", safe_next_actions({"limit": 5})),
        ("unhealthy configured storage work queue", work_queue({"limit": 5})),
    ]:
        metadata = result.metadata
        if metadata.get("doctor_storage_runtime_fallback_active") is not False:
            raise SystemExit(f"{label} should not report runtime fallback: {metadata}")
        if metadata.get("doctor_storage_readiness_blocks_completion_claim") is not True:
            raise SystemExit(f"{label} should block completion on unhealthy configured storage: {metadata}")
        if metadata.get("doctor_storage_recovery_required") is not True:
            raise SystemExit(f"{label} missed configured storage recovery requirement: {metadata}")
        if metadata.get("doctor_storage_recovery_mode") != "repair_configured_storage":
            raise SystemExit(f"{label} should use configured-storage repair mode: {metadata}")
        if metadata.get("doctor_storage_recovery_restart_required") is not False:
            raise SystemExit(f"{label} should not require restart before configured storage is repaired: {metadata}")
        if "writable durable storage" not in metadata.get("doctor_storage_recovery_next_operator_action", ""):
            raise SystemExit(f"{label} missed configured-storage next operator action: {metadata}")
        if metadata.get("doctor_storage_readiness_next_commands") != expected_commands:
            raise SystemExit(f"{label} configured-storage recovery queue diverged: {metadata}")
        if metadata.get("doctor_storage_readiness_next_command_count") != len(expected_commands):
            raise SystemExit(f"{label} configured-storage recovery queue count diverged: {metadata}")
        assert_doctor_storage_handoff_contract(metadata, label, expected_commands)
        if metadata.get("doctor_storage_recovery_check_tool_command") != STORAGE_RECOVERY_CHECK_COMMAND:
            raise SystemExit(f"{label} missed native configured-storage check command: {metadata}")
        proof_queue = metadata.get("doctor_completion_proof_queue") or []
        if proof_queue[: len(expected_commands)] != expected_commands:
            raise SystemExit(f"{label} should put configured-storage recovery first in completion proof queue: {metadata}")
        if metadata.get("doctor_completion_next_proof_command") != "storage status":
            raise SystemExit(f"{label} should make storage status the next proof command: {metadata}")
        if "storage recovery queue" not in result.output:
            raise SystemExit(f"{label} output should expose storage recovery queue: {result.output}")


def assert_malformed_storage_ready_flag_blocks_next_step_completion_proof(root: Path) -> None:
    runtime = make_temp_runtime(root)
    original_configured_storage_diagnostics = next_step_module._configured_storage_diagnostics

    def fake_configured_storage_diagnostics(
        config: JarvisConfig | None,
        fallback: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str]:
        return {
            "available": "true",
            "status": "ready",
            "issues": [],
            "metadata_only": "true",
        }, "mocked"

    try:
        next_step_module._configured_storage_diagnostics = fake_configured_storage_diagnostics  # type: ignore[assignment]
        (
            safe_next_actions,
            _next_action_packet,
            _priority_stack,
            _continuation_packet,
            _build_target_packet,
            _harness_build_slice,
            _save_build_target_packet,
            _export_mission_control,
            work_queue,
            _save_work_queue,
        ) = make_next_step_tools(
            runtime.store,
            runtime.vault,
            None,
            None,
            runtime.config,
        )
        results = [
            ("malformed storage ready safe next actions", safe_next_actions({"limit": 5})),
            ("malformed storage ready work queue", work_queue({"limit": 5})),
        ]
    finally:
        next_step_module._configured_storage_diagnostics = original_configured_storage_diagnostics  # type: ignore[assignment]

    expected_commands = [
        "storage status",
        STORAGE_RECOVERY_PLAN_COMMAND,
        STORAGE_RECOVERY_CHECK_COMMAND,
        BOOTSTRAP_CHECK_COMMAND,
        BOOTSTRAP_WRITE_COMMAND,
    ]
    for label, result in results:
        metadata = result.metadata
        if metadata.get("doctor_storage_runtime_fallback_active") is not False:
            raise SystemExit(f"{label} should not report runtime fallback: {metadata}")
        if metadata.get("doctor_storage_readiness_blocks_completion_claim") is not True:
            raise SystemExit(f"{label} should block completion on malformed configured storage readiness: {metadata}")
        if metadata.get("doctor_storage_recovery_required") is not True:
            raise SystemExit(f"{label} should require storage recovery from malformed readiness: {metadata}")
        if metadata.get("doctor_storage_recovery_mode") != "repair_configured_storage":
            raise SystemExit(f"{label} should route malformed readiness to configured-storage repair: {metadata}")
        if metadata.get("doctor_storage_readiness_next_commands") != expected_commands:
            raise SystemExit(f"{label} malformed readiness recovery queue diverged: {metadata}")
        assert_doctor_storage_handoff_contract(metadata, label, expected_commands)
        if metadata.get("doctor_storage_readiness_blocker") != "configured storage needs attention":
            raise SystemExit(f"{label} should use conservative default storage blocker: {metadata}")
        proof_queue = metadata.get("doctor_completion_proof_queue") or []
        if proof_queue[: len(expected_commands)] != expected_commands:
            raise SystemExit(f"{label} should put storage recovery first in completion proof queue: {metadata}")
        handoff = metadata.get("safe_next_actions_handoff") or metadata.get("work_queue_handoff") or {}
        next_commands = handoff.get("next_commands") or []
        for command in expected_commands:
            if command not in next_commands:
                raise SystemExit(f"{label} operator handoff missed malformed-readiness recovery command {command}: {metadata}")


def assert_doctor_storage_handoff_contract(metadata: dict[str, Any], label: str, expected_commands: list[str]) -> None:
    if metadata.get("doctor_storage_readiness_next_required_command") != expected_commands[0]:
        raise SystemExit(f"{label} missed canonical storage next required command: {metadata}")
    if metadata.get("doctor_storage_readiness_next_proof_command") != expected_commands[0]:
        raise SystemExit(f"{label} missed canonical storage next proof command: {metadata}")
    if metadata.get("doctor_storage_readiness_proof_queue") != expected_commands:
        raise SystemExit(f"{label} storage proof queue diverged: {metadata}")
    if metadata.get("doctor_storage_readiness_proof_queue_count") != len(expected_commands):
        raise SystemExit(f"{label} storage proof queue count diverged: {metadata}")
    if metadata.get("doctor_storage_readiness_first_proof_command") != expected_commands[0]:
        raise SystemExit(f"{label} storage first proof command diverged: {metadata}")
    if metadata.get("doctor_storage_next_proof_command") != metadata.get("doctor_storage_readiness_next_proof_command"):
        raise SystemExit(f"{label} legacy storage proof alias diverged from canonical alias: {metadata}")
    handoff = metadata.get("doctor_storage_handoff")
    if not isinstance(handoff, dict) or not handoff:
        raise SystemExit(f"{label} missed compact doctor storage handoff: {metadata}")
    if handoff.get("source") != "next_step_storage":
        raise SystemExit(f"{label} doctor storage handoff source diverged: {handoff}")
    for key in ["handoff_ready", "storage_handoff_ready", "ready_for_operator"]:
        if handoff.get(key) is not True:
            raise SystemExit(f"{label} doctor storage handoff should report {key}=True: {handoff}")
    for key in ["state_changed", "authorizes_execution", "authorizes_completion_claim", "approval_granted"]:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} doctor storage handoff should report {key}=False: {handoff}")
    boundaries = handoff.get("boundaries") or {}
    if boundaries.get("metadata_only") is not True:
        raise SystemExit(f"{label} doctor storage handoff should be metadata-only: {handoff}")
    for key in [
        "reads_database_file",
        "reads_db_file_contents",
        "reads_vault_files",
        "scans_obsidian_vault",
        "writes_files",
        "writes_database",
        "writes_notes",
        "writes_memory",
        "queues_approval",
        "approves_requests",
        "dismisses_approvals",
        "controls_computer",
        "calls_external_service",
        "executes_tools",
        "authorizes_execution",
        "authorizes_completion_claim",
    ]:
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} doctor storage handoff boundary {key} should be false: {handoff}")
    for top_key, handoff_key in [
        ("doctor_storage_readiness_blocks_completion_claim", "storage_readiness_blocks_completion_claim"),
        ("doctor_storage_recovery_required", "storage_recovery_required"),
        ("doctor_storage_recovery_mode", "storage_recovery_mode"),
        ("doctor_storage_recovery_next_operator_action", "storage_recovery_next_operator_action"),
        ("doctor_storage_recovery_restart_required", "storage_recovery_restart_required"),
        ("doctor_storage_recovery_check_tool_command", "storage_recovery_check_tool_command"),
        ("doctor_storage_recovery_check_command", "storage_recovery_check_command"),
        ("doctor_storage_recovery_command", "storage_recovery_command"),
        ("doctor_storage_readiness_next_required_command", "storage_readiness_next_required_command"),
        ("doctor_storage_readiness_next_proof_command", "storage_readiness_next_proof_command"),
        ("doctor_storage_readiness_proof_queue", "storage_readiness_proof_queue"),
        ("doctor_storage_readiness_first_proof_command", "storage_readiness_first_proof_command"),
        ("doctor_storage_issue_count", "storage_issue_count"),
    ]:
        if metadata.get(top_key) != handoff.get(handoff_key):
            raise SystemExit(f"{label} doctor storage handoff field {handoff_key} diverged from {top_key}: {handoff}")


def assert_unreadable_tool_run_rows_surface_in_next_step_handoffs(root: Path) -> None:
    runtime = make_temp_runtime(root)
    runtime.store.log_tool_run(
        runtime.session_id,
        "",
        "READ_ONLY",
        True,
        False,
        "fixture unreadable recent run",
        metadata={},
    )
    runtime.store.log_tool_run(
        runtime.session_id,
        "run_shell_command",
        "HIGH_RISK",
        False,
        False,
        "fixture failed action",
        metadata={},
    )
    (
        safe_next_actions,
        next_action_packet,
        priority_stack,
        _continuation_packet,
        _build_target_packet,
        _harness_build_slice,
        _save_build_target_packet,
        _export_mission_control,
        work_queue,
        _save_work_queue,
    ) = make_next_step_tools(runtime.store, runtime.vault)
    expected_queue = ["storage status", "recent tool runs", "execution health report"]
    cases = [
        ("unreadable audit safe next actions", safe_next_actions({"limit": 5}), "safe_next_actions_handoff"),
        ("unreadable audit work queue", work_queue({"limit": 5}), "work_queue_handoff"),
        ("unreadable audit next action packet", next_action_packet({}), "next_action_packet_handoff"),
        ("unreadable audit priority stack", priority_stack({}), "priority_stack_handoff"),
    ]
    for label, result, handoff_key in cases:
        metadata = result.metadata
        assert_doctor_handoff(metadata, label)
        if "audit readability review" not in result.output:
            raise SystemExit(f"{label} missed audit-readability prose: {result.output}")
        if metadata.get("doctor_audit_readability_review_required") is not True:
            raise SystemExit(f"{label} missed required audit-readability review flag: {metadata}")
        if metadata.get("doctor_audit_readability_review_commands") != expected_queue:
            raise SystemExit(f"{label} audit-readability queue diverged: {metadata}")
        if metadata.get("doctor_audit_readability_review_next_command") != expected_queue[0]:
            raise SystemExit(f"{label} missed next audit-readability command: {metadata}")
        if metadata.get("doctor_unreadable_recent_tool_run_rows") != 1:
            raise SystemExit(f"{label} missed unreadable recent tool-run row count: {metadata}")
        handoff = metadata.get(handoff_key) or {}
        for key in [
            "doctor_audit_readability_review_required",
            "doctor_audit_readability_review_commands",
            "doctor_audit_readability_review_command_count",
            "doctor_audit_readability_review_next_command",
            "doctor_unreadable_recent_tool_run_rows",
        ]:
            if handoff.get(key) != metadata.get(key):
                raise SystemExit(f"{label} handoff {key} diverged from metadata: {metadata}")


def main() -> None:
    assert_next_step_exact_metadata_bool()
    with TemporaryDirectory(prefix="jarvis-next-step-") as temp:
        runtime = make_temp_runtime(Path(temp))
        assert_diagnostic_only_runs_do_not_create_learning_debt(Path(temp) / "diagnostic-only-next-step")
        assert_next_step_doctor_separates_approval_held_runs(Path(temp) / "approval-held-next-step")
        assert_registry_aware_doctor_agi_selection(Path(temp) / "registry-aware-doctor-agi")
        assert_storage_fallback_drives_next_step_completion_proof(Path(temp) / "storage-fallback-next-step")
        assert_unhealthy_configured_storage_blocks_next_step_completion_proof(
            Path(temp) / "unhealthy-configured-storage-next-step"
        )
        assert_malformed_storage_ready_flag_blocks_next_step_completion_proof(
            Path(temp) / "malformed-storage-ready-next-step"
        )
        assert_unreadable_tool_run_rows_surface_in_next_step_handoffs(
            Path(temp) / "unreadable-tool-run-next-step"
        )
        mission_control = Path(temp) / "Vault" / "Jarvis" / "Automations" / "Mission Control.md"
        work_queue_note = Path(temp) / "Vault" / "Jarvis" / "Automations" / "Work Queue.md"
        build_target_note = Path(temp) / "Vault" / "Jarvis" / "Automations" / "Build Target Packet.md"
        fixture_recovery_receipt_path = Path(temp) / "recovery-receipt.md"
        fixture_recovery_receipt_path.write_text("fixture recovery receipt\n", encoding="utf-8")
        fixture_recovery_receipt_sha256 = file_sha256(fixture_recovery_receipt_path)
        fixture_recovery_checkpoint_path = Path(temp) / "work-block-checkpoint.md"
        fixture_recovery_checkpoint_path.write_text("fixture work-block checkpoint\n", encoding="utf-8")
        fixture_recovery_checkpoint_sha256 = file_sha256(fixture_recovery_checkpoint_path)
        cases = [
            "add task review Jarvis safe next actions priority high",
            "create goal Build safe assistant because continue toward AGI safely",
            "add step to goal 1: verify safety gates before autonomy",
            "record decision Jarvis suggests safe next actions before risky work because full assistant needs guardrails impact safer autonomy",
            "set preference planning style to safety first category work",
            "schedule assistant basics",
            "run command python3 --version",
            "what should Jarvis do next?",
            "what should I work on next for Jarvis",
            "what should I do now?",
            "what should I work on today?",
            "what's the next thing I should do?",
            "what's my next move?",
            "send me my next move",
            "send me a priority card",
            "next action packet",
            "priority stack",
            "resume Jarvis work safely",
            "continuation packet: continue building Jarvis V2 safely",
            "operator timebox: keep my computer awake while working on Jarvis stop_at=2099-01-01T00:00:00+09:00 current_time=2026-06-09T03:00:00+09:00 timezone=Asia/Seoul",
            "operator timebox: continue Jarvis until 2099-01-01T00:00:00+09:00 current_time=2026-06-09T03:00:00+09:00 timezone=Asia/Seoul",
            "operator timebox: continue Jarvis until 2026-06-09T02:00:00+09:00 current_time=2026-06-09T03:00:00+09:00 timezone=Asia/Seoul",
            "operator instruction supersession: continue Jarvis safely previous=work until 9:30 latest=stop now it is past 9:30 stop_at=2026-06-09T09:30:00+09:00 current_time=2026-06-09T10:00:00+09:00 timezone=Asia/Seoul",
            "operator instruction supersession: continue Jarvis safely previous=stop at 9:30 latest=continue until 8pm stop_at=2026-06-09T20:00:00+09:00 current_time=2026-06-09T10:00:00+09:00 timezone=Asia/Seoul",
            f"checkpoint recovery follow-through: step=reviewed local-safe continuity metadata verification=smoke_test_next_step passed receipt={fixture_recovery_receipt_path} receipt_sha256={fixture_recovery_receipt_sha256} checkpoint={fixture_recovery_checkpoint_path} checkpoint_sha256={fixture_recovery_checkpoint_sha256} stop=stop if verification drifts blockers=none",
            "checkpoint recovery execute reviewed=true step=reviewed local-safe continuity metadata verification=smoke_test_next_step passed files=jarvis_v2/tools/continuity.py blockers=none",
            f"autonomy resume gate: continue Jarvis safely stop_at=2099-01-01T00:00:00+09:00 current_time=2026-06-09T03:00:00+09:00 timezone=Asia/Seoul step=reviewed local-safe continuity metadata verification=smoke_test_next_step passed receipt={fixture_recovery_receipt_path} receipt_sha256={fixture_recovery_receipt_sha256} checkpoint={fixture_recovery_checkpoint_path} checkpoint_sha256={fixture_recovery_checkpoint_sha256} stop_condition=stop if verification drifts blockers=none",
            f"autonomy continuation execution: continue Jarvis safely stop_at=2099-01-01T00:00:00+09:00 current_time=2026-06-09T03:00:00+09:00 timezone=Asia/Seoul step=reviewed local-safe continuity metadata verification=smoke_test_next_step passed receipt={fixture_recovery_receipt_path} receipt_sha256={fixture_recovery_receipt_sha256} checkpoint={fixture_recovery_checkpoint_path} checkpoint_sha256={fixture_recovery_checkpoint_sha256} stop_condition=stop if verification drifts blockers=none next_step=update local continuity smoke metadata next_verification=smoke_test_next_step passed",
            f"autonomy step closure: continue Jarvis safely stop_at=2099-01-01T00:00:00+09:00 current_time=2026-06-09T03:00:00+09:00 timezone=Asia/Seoul reviewed_step=reviewed local-safe continuity metadata recovery_verification=smoke_test_next_step passed recovery_receipt_path={fixture_recovery_receipt_path} recovery_receipt_sha256={fixture_recovery_receipt_sha256} recovery_checkpoint_path={fixture_recovery_checkpoint_path} recovery_checkpoint_sha256={fixture_recovery_checkpoint_sha256} stop_condition=stop if verification drifts blockers=none next_step=update local continuity smoke metadata next_verification=smoke_test_next_step passed step=update local continuity smoke metadata",
            f"autonomy cycle ledger: continue Jarvis safely stop_at=2099-01-01T00:00:00+09:00 current_time=2026-06-09T03:00:00+09:00 timezone=Asia/Seoul reviewed_step=reviewed local-safe continuity metadata recovery_verification=smoke_test_next_step passed recovery_receipt_path={fixture_recovery_receipt_path} recovery_receipt_sha256={fixture_recovery_receipt_sha256} recovery_checkpoint_path={fixture_recovery_checkpoint_path} recovery_checkpoint_sha256={fixture_recovery_checkpoint_sha256} stop_condition=stop if verification drifts blockers=none next_step=update local continuity smoke metadata next_verification=smoke_test_next_step passed step=update local continuity smoke metadata",
            "what is the next build target for Jarvis",
            "build target packet: continue building Jarvis V2 safely",
            "pick next harness slice for Jarvis V2",
            "harness build slice: personal connector readiness",
            "harness build slice: dashboard interface polish",
            "continue building Jarvis V2 as an agent harness",
            "save build target packet: continue building Jarvis V2 safely",
            "work queue",
            "save work queue",
            "mission control",
            "help continuity",
        ]
        for case in cases:
            result = runtime.handle(case)
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:2000])
            print()
            if result.tool_results and result.tool_results[0].tool_name in CONTINUITY_TOOL_NAMES:
                assert_no_future_authority(
                    result.tool_results[0].metadata,
                    result.tool_results[0].tool_name,
                )
            if case in {
                "what should Jarvis do next?",
                "what should I work on next for Jarvis",
                "what should I do now?",
                "what should I work on today?",
                "what's the next thing I should do?",
                "what's my next move?",
                "send me my next move",
                "send me a priority card",
            }:
                if case == "what should Jarvis do next?":
                    required = [
                        "Safe next actions:",
                        "Priority goal:",
                        "Finish Jarvis V2 as an AI agent harness",
                        "pending approvals",
                        "approval readiness",
                        "Execution health",
                        "execution recovery packet",
                        "Blocker categories",
                        "Verification coverage",
                        "Recovery queue",
                        "recovery closure checklist",
                        "Verification-to-learning handoff",
                        "Execution learning debt",
                        "next actionable learning required",
                        "actionable learning proof queue",
                        "ordered learning gate required",
                        "learning proof queue",
                        "Approval readiness handoff",
                        "staleness",
                        "last look",
                        "verify after approval",
                        "approve only if exact and verifiable",
                        "dismiss stale",
                        "Approval proof chain handoff",
                        "Doctor handoff",
                        "completion claim state",
                        "completion blockers",
                        "execution learning actionable next required",
                        "execution learning ordered gate required",
                        "execution learning actionable proof queue",
                        "approval chain proof 1",
                        "verification receipt <approved run id from approval chain proof 1>",
                        "Approval #",
                        "review Jarvis safe next actions",
                        "Build safe assistant",
                        "schedule",
                        "safety first",
                        "Do not auto-run shell",
                    ]
                else:
                    required = [
                        "Jarvis next action packet",
                        "read-only",
                        "Priority goal:",
                        "Finish Jarvis V2 as an AI agent harness",
                        "Recommended move",
                        "review_approval",
                        "approval readiness",
                        "Why this move",
                        "Execution health",
                        "execution recovery packet",
                        "blocker categories",
                        "verification coverage",
                        "recovery queue",
                        "recovery closure checklist",
                        "Execution learning debt",
                        "next actionable learning required",
                        "actionable learning proof queue",
                        "Approval readiness handoff",
                        "verify after approval",
                        "approve only if exact and verifiable",
                        "Approval proof chain handoff",
                        "Doctor handoff",
                        "completion claim state",
                        "completion blockers",
                        "execution learning actionable next required",
                        "execution learning ordered gate required",
                        "execution learning actionable proof queue",
                        "completion proof queue",
                        "approval chain proof 1",
                        "approval packet",
                        "does not complete tasks",
                        "approve blocked work",
                    ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Safe next actions missing expected context: {missing}")
                if "next learning proof: `execution learning closure" in result.response:
                    raise SystemExit("Safe next actions should show actionable learning required before the closure recheck.")
                assert_doctor_learning_prose(result.response, "Safe next actions")
                metadata = result.tool_results[0].metadata
                expected_tool = "safe_next_actions" if case == "what should Jarvis do next?" else "next_action_packet"
                if result.tool_results[0].tool_name != expected_tool:
                    raise SystemExit(f"Natural next-action route should use {expected_tool}: {result.tool_results[0].tool_name}")
                if metadata.get("failed_action_runs", 0) + metadata.get("approval_held_action_runs", 0) < 1:
                    raise SystemExit(f"Safe next actions missed execution health metadata: {metadata}")
                if metadata.get("failed_action_runs", 0) and metadata.get("execution_health_review_required") is not True:
                    raise SystemExit(f"Safe next actions missed failed-run review metadata: {metadata}")
                if metadata.get("execution_health_next_command") not in metadata.get("execution_health_next_commands", []):
                    raise SystemExit(f"Safe next actions missed execution health recovery queue: {metadata}")
                if not {"failed_or_blocked", "approval_held"} & set(metadata.get("execution_health_blocker_categories", [])):
                    raise SystemExit(f"Safe next actions missed execution health blocker categories: {metadata}")
                if metadata.get("execution_health_verification_coverage") not in {"present", "missing"}:
                    raise SystemExit(f"Safe next actions missed verification coverage: {metadata}")
                assert_execution_health_proof_chain(metadata, "Safe next actions")
                assert_execution_health_learning_handoff(metadata, "Safe next actions")
                assert_doctor_handoff(metadata, "Safe next actions")
                assert_approval_handoff(metadata, "Safe next actions")
                if case == "what should Jarvis do next?":
                    assert_safe_next_actions_handoff(metadata, "Safe next actions")
                for key in READ_ONLY_FALSE_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Safe next actions should report {key}=False.")
            if case == "next action packet":
                required = [
                    "Jarvis next action packet",
                    "read-only",
                    "Priority goal:",
                    "Finish Jarvis V2 as an AI agent harness",
                    "Recommended move",
                    "review_approval",
                    "approval readiness",
                    "approval packet",
                    "Why this move",
                    "Execution health",
                    "failed action runs",
                    "blocker categories",
                    "recovery queue",
                    "recovery closure checklist",
                    "Verification-to-learning handoff",
                    "Execution learning debt",
                    "next actionable learning required",
                    "actionable learning proof queue",
                    "ordered learning gate required",
                    "learning proof queue",
                    "Approval readiness handoff",
                    "staleness",
                    "verify after approval",
                    "approve only if exact and verifiable",
                    "Approval proof chain handoff",
                    "Doctor handoff",
                    "completion claim state",
                    "completion blockers",
                    "execution learning actionable next required",
                    "execution learning ordered gate required",
                    "execution learning actionable proof queue",
                    "completion proof queue",
                    "approval chain proof 1",
                    "Verification",
                    "Guardrails",
                    "does not complete tasks",
                    "approve blocked work",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Next action packet missing expected context: {missing}")
                if "next learning proof: `execution learning closure" in result.response:
                    raise SystemExit("Next action packet should show actionable learning proof before the closure recheck.")
                assert_doctor_learning_prose(result.response, "Next action packet")
                metadata = result.tool_results[0].metadata
                if metadata.get("action_kind") != "review_approval":
                    raise SystemExit("Next action packet should prioritize pending approval review.")
                if metadata.get("failed_action_runs", 0) + metadata.get("approval_held_action_runs", 0) < 1:
                    raise SystemExit(f"Next action packet missed execution health metadata: {metadata}")
                if metadata.get("failed_action_runs", 0) and metadata.get("execution_health_review_required") is not True:
                    raise SystemExit(f"Next action packet missed failed-run review metadata: {metadata}")
                if metadata.get("execution_health_next_command") not in metadata.get("execution_health_next_commands", []):
                    raise SystemExit(f"Next action packet missed execution health recovery queue: {metadata}")
                assert_execution_health_proof_chain(metadata, "Next action packet")
                assert_execution_health_learning_handoff(metadata, "Next action packet")
                assert_doctor_handoff(metadata, "Next action packet")
                assert_approval_handoff(metadata, "Next action packet")
                assert_next_action_packet_handoff(metadata, "Next action packet")
                for key in READ_ONLY_FALSE_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Next action packet unsafe metadata {key}: {metadata}")
            if case == "work queue":
                required = [
                    "Jarvis work queue:",
                    "Rule: do safe review",
                    "1. Start safely",
                    "2. Approval blockers",
                    "Approval #",
                    "approval readiness",
                    "approval packet",
                    "approve approval",
                    "dismiss approval",
                    "3. Execution health",
                    "execution recovery packet",
                    "Blocker categories",
                    "Verification coverage",
                    "Recovery queue",
                    "recovery closure checklist",
                    "Verification-to-learning handoff",
                    "Execution learning debt",
                    "next actionable learning required",
                    "actionable learning proof queue",
                    "ordered learning gate required",
                    "learning proof queue",
                    "Approval readiness handoff",
                    "staleness",
                    "last look",
                    "verify after approval",
                    "approve only if exact and verifiable",
                    "Approval proof chain handoff",
                    "Doctor handoff",
                    "completion claim state",
                    "completion blockers",
                    "execution learning actionable next required",
                    "execution learning ordered gate required",
                    "execution learning actionable proof queue",
                    "completion proof queue",
                    "approval chain proof 1",
                    "verification receipt <approved run id from approval chain proof 1>",
                    "4. Open tasks",
                    "review Jarvis safe next actions",
                    "complete task",
                    "5. Goal steps",
                    "Build safe assistant",
                    "complete goal step",
                    "6. Background upkeep",
                    "save handoff brief",
                    "7. Agent landscape research guidance",
                    "Zoey/OpenClaw/Hermes research",
                    "OpenAI Agents SDK",
                    "LangGraph",
                    "human-in-the-loop",
                    "OpenHands/Devin",
                    "persistent prompt-injection",
                    "Zoey/Lindy/Zapier",
                    "broad ungated integrations",
                    "visible control plane",
                    "not companion personas",
                    "operator-real evals",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Work queue missing expected context: {missing}")
                if "next learning proof: `execution learning closure" in result.response:
                    raise SystemExit("Work queue should show actionable learning proof before the closure recheck.")
                assert_doctor_learning_prose(result.response, "Work queue")
                metadata = result.tool_results[0].metadata
                if metadata.get("failed_action_runs", 0) + metadata.get("approval_held_action_runs", 0) < 1:
                    raise SystemExit(f"Work queue missed execution health metadata: {metadata}")
                if metadata.get("failed_action_runs", 0) and metadata.get("execution_health_review_required") is not True:
                    raise SystemExit(f"Work queue missed failed-run review metadata: {metadata}")
                if metadata.get("execution_health_next_command") not in metadata.get("execution_health_next_commands", []):
                    raise SystemExit(f"Work queue missed execution health recovery queue: {metadata}")
                if not {"failed_or_blocked", "approval_held"} & set(metadata.get("execution_health_blocker_categories", [])):
                    raise SystemExit(f"Work queue missed execution health blocker categories: {metadata}")
                assert_execution_health_proof_chain(metadata, "Work queue")
                assert_execution_health_learning_handoff(metadata, "Work queue")
                assert_doctor_handoff(metadata, "Work queue")
                assert_approval_handoff(metadata, "Work queue")
                assert_work_queue_handoff(metadata, "Work queue")
                if metadata.get("agent_landscape_research_requires_control_plane") is not True:
                    raise SystemExit(f"Work queue missed agent research control-plane metadata: {metadata}")
                if metadata.get("agent_landscape_research_requires_human_in_loop") is not True:
                    raise SystemExit(f"Work queue missed agent research human-in-loop metadata: {metadata}")
                if metadata.get("agent_landscape_research_warns_persistent_agent_risk") is not True:
                    raise SystemExit(f"Work queue missed persistent-agent risk metadata: {metadata}")
                if metadata.get("agent_landscape_research_defers_broad_integrations") is not True:
                    raise SystemExit(f"Work queue missed integration-deferral metadata: {metadata}")
                if metadata.get("agent_landscape_research_next_build_focus") != [
                    "capability_cockpit",
                    "phone_control_center",
                    "operator_workflow_evals",
                ]:
                    raise SystemExit(f"Work queue missed agent research build-focus metadata: {metadata}")
                for key in READ_ONLY_FALSE_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Work queue should report {key}=False.")
            if case == "priority stack":
                required = [
                    "Jarvis priority stack:",
                    "read-only",
                    "Priority goal:",
                    "Finish Jarvis V2 as an AI agent harness",
                    "Ranking rule",
                    "Pending risky approvals",
                    "Execution-health recovery",
                    "High-priority concrete tasks",
                    "Active goal steps",
                    "Stack:",
                    "approval_review",
                    "approval readiness",
                    "approval packet",
                    "review Jarvis safe next actions",
                    "Build safe assistant",
                    "Context signals",
                    "execution health next command",
                    "execution health blockers",
                    "execution health recovery queue",
                    "recovery closure checklist",
                    "Verification-to-learning handoff",
                    "Execution learning debt",
                    "learning proof queue",
                    "Approval readiness handoff",
                    "staleness",
                    "verify after approval",
                    "approve only if exact and verifiable",
                    "Approval proof chain handoff",
                    "Doctor handoff",
                    "completion claim state",
                    "completion blockers",
                    "completion proof queue",
                    "approval chain proof 1",
                    "Guardrails",
                    "does not complete tasks",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Priority stack missing expected context: {missing}")
                metadata = result.tool_results[0].metadata
                if metadata.get("top_kind") != "approval_review":
                    raise SystemExit("Priority stack should rank pending approval review first.")
                if metadata.get("failed_action_runs", 0) + metadata.get("approval_held_action_runs", 0) < 1:
                    raise SystemExit(f"Priority stack missed execution health metadata: {metadata}")
                if metadata.get("failed_action_runs", 0) and metadata.get("execution_health_review_required") is not True:
                    raise SystemExit(f"Priority stack missed failed-run review metadata: {metadata}")
                if metadata.get("execution_health_next_command") not in metadata.get("execution_health_next_commands", []):
                    raise SystemExit(f"Priority stack missed execution health recovery queue: {metadata}")
                assert_execution_health_proof_chain(metadata, "Priority stack")
                assert_execution_health_learning_handoff(metadata, "Priority stack")
                assert_doctor_handoff(metadata, "Priority stack")
                assert_approval_handoff(metadata, "Priority stack")
                assert_priority_stack_handoff(metadata, "Priority stack")
                for key in READ_ONLY_FALSE_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Priority stack unsafe metadata {key}: {metadata}")
            if case.startswith("continuation packet") or case == "resume Jarvis work safely":
                required = [
                    "Jarvis continuation packet",
                    "read-only",
                    "Objective:",
                    "Priority goal:",
                    "Finish Jarvis V2 as an AI agent harness",
                    "Recommended focus",
                    "execution health report",
                    "failed action runs",
                    "Loop:",
                    "Approval boundaries",
                    "Stop conditions",
                    "Verification ladder",
                    "acceptance gate:",
                    "evidence <receipt>",
                    "tests <verification>",
                    "recovery <rollback or stop condition>",
                    "Context signals",
                    "approval readiness",
                    "approval packet",
                    "execution health blockers",
                    "execution health recovery queue",
                    "Checkpoint recovery contract",
                    "checkpoint:",
                    "checkpoint freshness",
                    "checkpoint age minutes",
                    "recovery required before normal follow-through",
                    "next checkpoint proof",
                    "checkpoint recovery apply",
                    "checkpoint recovery execute reviewed=true",
                    "verification target",
                    "stop condition",
                    "follow-through gate",
                    "checkpoint recovery follow-through packet",
                    "Recovery closure gate",
                    "Verification-to-learning handoff",
                    "Approval proof chain handoff",
                    "approval chain proof 1",
                    "Do not approve pending requests",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Continuation packet missing expected context: {missing}")
                if result.tool_results[0].tool_name != "continuation_packet":
                    raise SystemExit(f"Natural continuation should route to continuation_packet: {result.tool_results[0].tool_name}")
                metadata = result.tool_results[0].metadata
                if metadata.get("failed_action_runs", 0) + metadata.get("approval_held_action_runs", 0) < 1:
                    raise SystemExit(f"Continuation packet missed execution health metadata: {metadata}")
                if metadata.get("failed_action_runs", 0) and metadata.get("execution_health_review_required") is not True:
                    raise SystemExit(f"Continuation packet missed failed-run review metadata: {metadata}")
                if metadata.get("execution_health_next_command") not in metadata.get("execution_health_next_commands", []):
                    raise SystemExit(f"Continuation packet missed execution health recovery queue: {metadata}")
                checkpoint_queue = metadata.get("checkpoint_recovery_queue") or []
                if metadata.get("checkpoint_recovery_queue_count") != len(checkpoint_queue):
                    raise SystemExit(f"Continuation packet missed checkpoint recovery queue count: {metadata}")
                if not checkpoint_queue or metadata.get("checkpoint_recovery_next_command") != checkpoint_queue[0]:
                    raise SystemExit(f"Continuation packet missed checkpoint recovery next command: {metadata}")
                for expected_command in [
                    "checkpoint recovery execute reviewed=true step=<reviewed local-safe step> verification=<evidence>",
                    "work block checkpoint",
                ]:
                    if expected_command not in checkpoint_queue:
                        raise SystemExit(f"Continuation packet missed checkpoint recovery command {expected_command}: {metadata}")
                if not metadata.get("checkpoint_recovery_verification_target") or not metadata.get("checkpoint_recovery_stop_condition"):
                    raise SystemExit(f"Continuation packet missed checkpoint verification target or stop condition: {metadata}")
                if "Recovery closure gate" not in metadata.get("checkpoint_recovery_followthrough_gate", ""):
                    raise SystemExit(f"Continuation packet missed checkpoint follow-through gate metadata: {metadata}")
                if metadata.get("checkpoint_freshness") not in {"missing", "fresh", "review_again", "stale", "unknown"}:
                    raise SystemExit(f"Continuation packet missed checkpoint freshness metadata: {metadata}")
                if "checkpoint_recovery_required" not in metadata:
                    raise SystemExit(f"Continuation packet missed checkpoint recovery-required metadata: {metadata}")
                checkpoint_path = str(metadata.get("checkpoint_path") or "")
                checkpoint_display = str(metadata.get("checkpoint_path_display") or "")
                if metadata.get("checkpoint_found"):
                    if not checkpoint_path:
                        raise SystemExit(f"Continuation packet missed exact checkpoint path metadata: {metadata}")
                    if not checkpoint_display.startswith("Reflections/") or checkpoint_display not in result.response:
                        raise SystemExit(f"Continuation packet missed safe checkpoint display metadata/output: {metadata}")
                    if checkpoint_path in result.response or str(Path(temp)) in result.response or "/private/" in result.response or "/\x55sers/" in result.response:
                        raise SystemExit(f"Continuation packet leaked a local checkpoint path in output: {result.response}")
                elif checkpoint_display:
                    raise SystemExit(f"Continuation packet should not show checkpoint display without a checkpoint: {metadata}")
                assert_execution_health_proof_chain(metadata, "Continuation packet")
                assert_execution_health_learning_handoff(metadata, "Continuation packet")
                assert_continuation_packet_handoff(metadata, "Continuation packet")
                for key in READ_ONLY_FALSE_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Continuation packet should report {key}=False.")
            if case.startswith("operator timebox"):
                expected_timebox_state = "STOP_TIME_REACHED" if "2026-06-09T02:00:00+09:00" in case else "STOP_WINDOW_ACTIVE"
                expected_awake_requested = "awake" in case.lower()
                required = [
                    "Jarvis operator timebox contract",
                    "read-only",
                    expected_timebox_state,
                    "Awake guard boundary",
                    f"awake requested: {'yes' if expected_awake_requested else 'no'}",
                    "a work timebox does not authorize OS wake locks",
                    "authorizes shell execution no",
                    "Non-authorizing timebox review contract",
                    "Next safe command",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Operator timebox contract missing expected context: {missing}")
                if result.tool_results[0].tool_name != "operator_timebox_contract":
                    raise SystemExit(f"Natural operator timebox should route to operator_timebox_contract: {result.tool_results[0].tool_name}")
                metadata = result.tool_results[0].metadata
                if metadata.get("timebox_state") != expected_timebox_state:
                    raise SystemExit(f"Operator timebox missed expected state {expected_timebox_state}: {metadata}")
                if expected_timebox_state == "STOP_WINDOW_ACTIVE" and metadata.get("can_continue_now") is not True:
                    raise SystemExit(f"Operator timebox missed active continuation metadata: {metadata}")
                if expected_timebox_state == "STOP_TIME_REACHED" and (
                    metadata.get("can_continue_now") is not False or metadata.get("should_stop_now") is not True
                ):
                    raise SystemExit(f"Operator timebox missed stop-reached metadata: {metadata}")
                if "until " in case and metadata.get("missing_timebox_proof"):
                    raise SystemExit(f"Operator timebox natural until should extract stop_at proof: {metadata}")
                assert_timebox_review_contract(metadata, "Operator timebox")
                assert_awake_guard_boundary(metadata, "Operator timebox", expected_requested=expected_awake_requested)
                for key in READ_ONLY_FALSE_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Operator timebox should report {key}=False.")
            if case.startswith("operator instruction supersession"):
                required = [
                    "Jarvis operator instruction supersession packet",
                    "read-only",
                    "Instruction precedence",
                    "latest instruction supersedes previous: yes",
                    "Supersession gate",
                    "Bound timebox",
                    "newest explicit user instruction overrides older goals",
                    "Supersession contract",
                    "authorizes execution now: no",
                    "authorizes risky work: no",
                    "authorizes approval: no",
                    "authorizes recovery follow-through: no",
                    "authorizes timebox override: no",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Operator instruction supersession missing expected context: {missing}")
                if result.tool_results[0].tool_name != "operator_instruction_supersession_packet":
                    raise SystemExit(f"Natural supersession should route to operator_instruction_supersession_packet: {result.tool_results[0].tool_name}")
                metadata = result.tool_results[0].metadata
                if metadata.get("latest_instruction_supersedes_previous") is not True:
                    raise SystemExit(f"Supersession packet missed latest-over-previous proof: {metadata}")
                if metadata.get("newest_instruction_overrides_automation") is not True or metadata.get("newest_instruction_overrides_goal") is not True:
                    raise SystemExit(f"Supersession packet missed override metadata: {metadata}")
                contract_rows = metadata.get("supersession_contract_rows") or []
                if metadata.get("supersession_contract_row_count") != len(contract_rows) or len(contract_rows) != 5:
                    raise SystemExit(f"Supersession packet missed contract rows: {metadata}")
                if metadata.get("supersession_contract_ready") is not True:
                    raise SystemExit(f"Supersession packet missed contract ready flag: {metadata}")
                contract_latest_is_stop = bool(metadata.get("latest_instruction_is_stop"))
                if not _operator_supersession_contract_ready(contract_rows, latest_is_stop=contract_latest_is_stop):
                    raise SystemExit(f"Supersession packet contract rows failed strict validator: {metadata}")
                tampered_contract_rows = [dict(row) for row in contract_rows]
                tampered_contract_rows[0]["source"] = "older operator instruction"
                if _operator_supersession_contract_ready(tampered_contract_rows, latest_is_stop=contract_latest_is_stop):
                    raise SystemExit(f"Supersession contract validator accepted tampered source: {tampered_contract_rows}")
                tampered_contract_rows = [dict(row) for row in contract_rows]
                tampered_required_index = 4 if contract_latest_is_stop else 2
                tampered_contract_rows[tampered_required_index]["required"] = not bool(
                    tampered_contract_rows[tampered_required_index].get("required")
                )
                if _operator_supersession_contract_ready(tampered_contract_rows, latest_is_stop=contract_latest_is_stop):
                    raise SystemExit(f"Supersession contract validator accepted tampered required flag: {tampered_contract_rows}")
                expected_contract_items = {
                    "newest_instruction",
                    "active_timebox",
                    "resume_gate_review",
                    "one_step_local_safe_review",
                    "stop_or_pause_brake",
                }
                if {row.get("item") for row in contract_rows} != expected_contract_items:
                    raise SystemExit(f"Supersession packet contract items diverged: {metadata}")
                if any(
                    row.get("authorizes_execution") is not False
                    or row.get("authorizes_risky_work") is not False
                    or row.get("authorizes_approval") is not False
                    or row.get("authorizes_recovery_followthrough") is not False
                    or row.get("authorizes_timebox_override") is not False
                    for row in contract_rows
                ):
                    raise SystemExit(f"Supersession contract rows should be non-authorizing: {metadata}")
                assert_sha256(metadata.get("supersession_token_sha256"), "Operator supersession token")
                if metadata.get("supersession_token_present") is not True:
                    raise SystemExit(f"Supersession packet missed token presence: {metadata}")
                token_rows = metadata.get("supersession_token_boundary_rows") or []
                if metadata.get("supersession_token_boundary_row_count") != 3 or len(token_rows) != 3:
                    raise SystemExit(f"Supersession packet missed token boundary rows: {metadata}")
                expected_token_rows = {
                    "operator_supersession_token": "present",
                    "newest_instruction_scope": "proof_only_for_current_operator_instruction_review",
                    "next_supersession_review": "fresh_latest_instruction_review_required",
                }
                if {row.get("item") for row in token_rows} != set(expected_token_rows):
                    raise SystemExit(f"Supersession packet token boundary items diverged: {metadata}")
                for row in token_rows:
                    item = row.get("item")
                    if row.get("status") != expected_token_rows.get(item):
                        raise SystemExit(f"Supersession token boundary status diverged: {row}")
                    if row.get("source") != "operator_instruction_supersession":
                        raise SystemExit(f"Supersession token boundary source diverged: {row}")
                    if row.get("token_sha256") != metadata.get("supersession_token_sha256"):
                        raise SystemExit(f"Supersession token boundary hash diverged: {row}")
                    if (
                        row.get("authorizes_execution") is not False
                        or row.get("authorizes_local_safe_step") is not False
                        or row.get("authorizes_risky_work") is not False
                        or row.get("authorizes_approval") is not False
                        or row.get("authorizes_recovery_followthrough") is not False
                        or row.get("authorizes_timebox_override") is not False
                        or row.get("authorizes_goal_override") is not False
                        or row.get("authorizes_model_call") is not False
                        or row.get("authorizes_tool_execution") is not False
                        or row.get("authorizes_personal_data_read") is not False
                        or row.get("authorizes_external_side_effect") is not False
                        or row.get("reusable_for_next_review") is not False
                        or row.get("reusable_for_next_timebox") is not False
                    ):
                        raise SystemExit(f"Supersession token boundary rows should be proof-only: {row}")
                for key in [
                    "supersession_authorizes_execution",
                    "supersession_authorizes_risky_work",
                    "supersession_authorizes_approval",
                    "supersession_authorizes_recovery_followthrough",
                    "supersession_authorizes_timebox_override",
                    "supersession_token_authorizes_execution",
                    "supersession_token_authorizes_local_safe_step",
                    "supersession_token_authorizes_risky_work",
                    "supersession_token_authorizes_approval",
                    "supersession_token_authorizes_recovery_followthrough",
                    "supersession_token_authorizes_timebox_override",
                    "supersession_token_authorizes_goal_override",
                    "supersession_token_authorizes_model_call",
                    "supersession_token_authorizes_tool_execution",
                    "supersession_token_authorizes_personal_data_read",
                    "supersession_token_authorizes_external_side_effect",
                    "supersession_token_reusable_for_next_review",
                    "supersession_token_reusable_for_next_timebox",
                ]:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Supersession packet should report {key}=False: {metadata}")
                if metadata.get("next_supersession_requires_fresh_latest_instruction_review") is not True:
                    raise SystemExit(f"Supersession packet missed fresh latest-instruction requirement: {metadata}")
                if "stop now" in case:
                    if metadata.get("supersession_state") != "NEWER_STOP_OR_PAUSE_OVERRIDES_AUTONOMY":
                        raise SystemExit(f"Stop supersession should override autonomy: {metadata}")
                    if metadata.get("can_continue_under_latest_instruction") is not False or metadata.get("stop_or_pause_blocks_autonomy") is not True:
                        raise SystemExit(f"Stop supersession should block continuation: {metadata}")
                    if metadata.get("timebox_state") != "STOP_TIME_REACHED":
                        raise SystemExit(f"Stop supersession should prove stop time reached: {metadata}")
                else:
                    if metadata.get("supersession_state") != "LATEST_INSTRUCTION_READY_TO_GOVERN_CONTINUATION":
                        raise SystemExit(f"Continue supersession should be ready: {metadata}")
                    if metadata.get("can_continue_under_latest_instruction") is not True or metadata.get("latest_instruction_is_continue") is not True:
                        raise SystemExit(f"Continue supersession should allow latest instruction to govern: {metadata}")
                    if metadata.get("timebox_state") != "STOP_WINDOW_ACTIVE":
                        raise SystemExit(f"Continue supersession should prove active timebox: {metadata}")
                for key in READ_ONLY_FALSE_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Operator instruction supersession should report {key}=False.")
            if case.startswith("checkpoint recovery follow-through"):
                required = [
                    "Jarvis checkpoint recovery follow-through packet",
                    "read-only",
                    "Follow-through state",
                    "RECOVERY_FOLLOWTHROUGH_READY",
                    "normal follow-through allowed: yes",
                    "Closure evidence",
                    "approval boundary",
                    "Measured recovery execution readiness proof",
                    "score: 100/100",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Checkpoint recovery follow-through missing expected context: {missing}")
                if result.tool_results[0].tool_name != "checkpoint_recovery_followthrough_packet":
                    raise SystemExit(f"Natural follow-through should route to checkpoint_recovery_followthrough_packet: {result.tool_results[0].tool_name}")
                metadata = result.tool_results[0].metadata
                if metadata.get("followthrough_state") != "RECOVERY_FOLLOWTHROUGH_READY":
                    raise SystemExit(f"Follow-through packet should be ready: {metadata}")
                if metadata.get("normal_followthrough_allowed") is not True:
                    raise SystemExit(f"Follow-through packet should allow normal follow-through: {metadata}")
                if metadata.get("recovery_closure_missing") != [] or metadata.get("recovery_closure_missing_count") != 0:
                    raise SystemExit(f"Follow-through packet should have complete closure proof: {metadata}")
                scorecard_rows = metadata.get("recovery_execution_scorecard_rows") or []
                expected_scorecard_items = {
                    "reviewed_step_permission",
                    "verification_target",
                    "receipt_hash_binding",
                    "checkpoint_hash_binding",
                    "stop_condition",
                    "risky_recovery_approval_boundary",
                    "followthrough_packet_ready",
                    "fresh_followthrough_token",
                }
                if metadata.get("recovery_execution_scorecard_row_count") != 8 or len(scorecard_rows) != 8:
                    raise SystemExit(f"Follow-through packet missed recovery execution scorecard rows: {metadata}")
                if {row.get("item") for row in scorecard_rows} != expected_scorecard_items:
                    raise SystemExit(f"Follow-through packet recovery execution scorecard items diverged: {metadata}")
                if metadata.get("recovery_execution_score") != 100 or metadata.get("recovery_execution_max_score") != 100:
                    raise SystemExit(f"Follow-through packet recovery execution proof should be complete: {metadata}")
                if metadata.get("recovery_execution_required_rows_ready") is not True:
                    raise SystemExit(f"Follow-through packet recovery execution rows should be ready: {metadata}")
                if metadata.get("recovery_execution_scorecard_ready") is not True:
                    raise SystemExit(f"Follow-through packet recovery execution scorecard should be ready: {metadata}")
                if metadata.get("recovery_execution_readiness_as_prior_proof") is not True:
                    raise SystemExit(f"Follow-through packet should carry recovery execution readiness as prior proof: {metadata}")
                if not _recovery_execution_scorecard_metadata_ready(
                    metadata,
                    prefix="recovery_execution",
                    prior_proof_key="recovery_execution_readiness_as_prior_proof",
                ):
                    raise SystemExit(f"Follow-through packet scorecard metadata validator rejected metadata: {metadata}")
                if not _checkpoint_recovery_followthrough_ready(metadata):
                    raise SystemExit(f"Follow-through packet should pass the production readiness validator: {metadata}")
                tampered_scorecard_rows = [dict(row) for row in scorecard_rows]
                tampered_scorecard_rows[0]["item"] = "tampered"
                tampered_scorecard_metadata = dict(metadata)
                tampered_scorecard_metadata["recovery_execution_scorecard_rows"] = tampered_scorecard_rows
                if _recovery_execution_scorecard_metadata_ready(
                    tampered_scorecard_metadata,
                    prefix="recovery_execution",
                    prior_proof_key="recovery_execution_readiness_as_prior_proof",
                ):
                    raise SystemExit(f"Follow-through scorecard metadata validator accepted row tampering: {metadata}")
                for key, value in [
                    ("recovery_execution_scorecard_row_count", 999),
                    ("recovery_execution_score", 0),
                    ("recovery_execution_max_score", 999),
                    ("recovery_execution_required_rows_ready", False),
                    ("recovery_execution_scorecard_ready", False),
                ]:
                    tampered_scorecard_metadata = dict(metadata)
                    tampered_scorecard_metadata[key] = value
                    if _recovery_execution_scorecard_metadata_ready(
                        tampered_scorecard_metadata,
                        prefix="recovery_execution",
                        prior_proof_key="recovery_execution_readiness_as_prior_proof",
                    ):
                        raise SystemExit(f"Follow-through scorecard metadata validator accepted stale mirror {key}: {metadata}")
                for key, value in [
                    ("normal_followthrough_allowed", False),
                    ("followthrough_state", "RECOVERY_FOLLOWTHROUGH_HELD"),
                    ("recovery_followthrough_gate_state", "blocked_missing_reviewed_step"),
                    ("recovery_closure_missing", ["receipt hash missing"]),
                    ("recovery_closure_missing_count", 1),
                    ("recovery_closure_required_evidence", ["reviewed local-safe step"]),
                    ("recovery_closure_required_evidence_count", 999),
                    ("approval_boundary_provided", False),
                    ("approval_boundary", "future risky recovery steps need review"),
                    ("receipt_hash_matches_file", False),
                    ("checkpoint_hash_matches_file", False),
                    ("recovery_followthrough_token_sha256", "0" * 64),
                    ("local_safe_recovery_execution_token_boundary_ready", False),
                    ("recovery_execution_scorecard_ready", False),
                    ("recovery_execution_readiness_token_authorizes_unreviewed_followthrough", True),
                    ("authorizes_execution", True),
                    ("recovery_followthrough_token_boundary_row_count", 999),
                    ("local_safe_recovery_execution_token_boundary_row_count", 999),
                    ("recovery_execution_readiness_token_boundary_row_count", 999),
                    ("next_recovery_followthrough_requires_new_token", False),
                    ("next_recovery_execution_requires_new_local_safe_token", False),
                    ("next_recovery_execution_requires_new_readiness_token", False),
                ]:
                    tampered = dict(metadata)
                    tampered[key] = value
                    if _checkpoint_recovery_followthrough_ready(tampered):
                        raise SystemExit(f"Follow-through readiness validator accepted tampered {key}: {tampered}")
                for missing_closure_key in [
                    "recovery_closure_missing",
                    "recovery_closure_missing_count",
                    "recovery_closure_required_evidence",
                    "recovery_closure_required_evidence_count",
                    "approval_boundary_provided",
                    "approval_boundary",
                    "recovery_followthrough_token_boundary_row_count",
                    "local_safe_recovery_execution_token_boundary_row_count",
                    "recovery_execution_readiness_token_boundary_row_count",
                    "next_recovery_followthrough_requires_new_token",
                    "next_recovery_execution_requires_new_local_safe_token",
                    "next_recovery_execution_requires_new_readiness_token",
                ]:
                    tampered = dict(metadata)
                    tampered.pop(missing_closure_key, None)
                    if _checkpoint_recovery_followthrough_ready(tampered):
                        raise SystemExit(
                            f"Follow-through readiness validator accepted missing explicit contract field "
                            f"{missing_closure_key}: {tampered}"
                        )
                tampered_rows = [dict(row) for row in scorecard_rows]
                tampered_rows[0]["authorizes_risky_work"] = True
                tampered = dict(metadata)
                tampered["recovery_execution_scorecard_rows"] = tampered_rows
                if _checkpoint_recovery_followthrough_ready(tampered):
                    raise SystemExit(f"Follow-through readiness validator accepted authority-bearing scorecard rows: {tampered}")
                if (
                    metadata.get("recovery_execution_readiness_authorizes_action_now") is not False
                    or metadata.get("recovery_execution_readiness_authorizes_risky_work") is not False
                    or metadata.get("recovery_execution_readiness_authorizes_unreviewed_followthrough") is not False
                ):
                    raise SystemExit(f"Follow-through packet recovery execution proof should be non-authorizing: {metadata}")
                assert_recovery_execution_boundary(metadata, "Follow-through packet")
                if any(
                    row.get("ready") is not True
                    or row.get("points") != row.get("max_points")
                    or row.get("required_before_normal_followthrough") is not True
                    or row.get("authorizes_risky_work") is not False
                    or row.get("authorizes_unreviewed_followthrough") is not False
                    for row in scorecard_rows
                ):
                    raise SystemExit(f"Follow-through packet scorecard rows should be ready and non-authorizing: {metadata}")
                for key in READ_ONLY_FALSE_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Follow-through packet should report {key}=False.")
            if case.startswith("checkpoint recovery execute"):
                required = [
                    "Jarvis checkpoint recovery executor",
                    "Reviewed local-safe recovery step recorded",
                    "Recovery closure gate",
                    "Measured recovery execution readiness scorecard",
                    "score: 100/100",
                    "required rows ready: yes",
                    "Recovery follow-through packet",
                    "RECOVERY_FOLLOWTHROUGH_READY",
                    "normal follow-through allowed: yes",
                    "fresh checkpoint",
                    "receipt sha256",
                    "checkpoint sha256",
                    "artifact hashes present: yes",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Checkpoint recovery execute missing expected context: {missing}")
                metadata = result.tool_results[0].metadata
                if not metadata.get("receipt_sha256") or not metadata.get("checkpoint_sha256"):
                    raise SystemExit(f"Executor missed recovery artifact hashes: {metadata}")
                if metadata.get("recovery_artifact_hashes_present") is not True:
                    raise SystemExit(f"Executor should report recovery artifact hashes present: {metadata}")
                contract = metadata.get("recovery_execution_contract") or {}
                if contract.get("receipt_sha256") != metadata.get("receipt_sha256"):
                    raise SystemExit(f"Executor contract missed receipt hash binding: {metadata}")
                if contract.get("checkpoint_sha256") != metadata.get("checkpoint_sha256"):
                    raise SystemExit(f"Executor contract missed checkpoint hash binding: {metadata}")
                if metadata.get("recovery_followthrough_packet_state") != "RECOVERY_FOLLOWTHROUGH_READY":
                    raise SystemExit(f"Executor missed ready follow-through packet state: {metadata}")
                if metadata.get("recovery_followthrough_packet_ready") is not True:
                    raise SystemExit(f"Executor follow-through packet should be ready: {metadata}")
                if metadata.get("recovery_followthrough_packet_artifact_hashes_present") is not True:
                    raise SystemExit(f"Executor follow-through packet missed artifact hash binding: {metadata}")
                if metadata.get("operator_timebox_precontinuation_contract_ready") is not True:
                    raise SystemExit(f"Executor missed pre-continuation timebox contract readiness: {metadata}")
                if metadata.get("operator_timebox_precontinuation_contract_state") != metadata.get("timebox_state"):
                    raise SystemExit(f"Executor pre-continuation timebox state mirror diverged: {metadata}")
                if metadata.get("operator_timebox_precontinuation_allows_continuation") != metadata.get("can_continue_now"):
                    raise SystemExit(f"Executor pre-continuation allow mirror diverged: {metadata}")
                if metadata.get("operator_timebox_precontinuation_blocks_continuation") != (
                    not bool(metadata.get("can_continue_now"))
                ):
                    raise SystemExit(f"Executor pre-continuation block mirror diverged: {metadata}")
                if metadata.get("operator_timebox_precontinuation_requires_stop") != metadata.get("should_stop_now"):
                    raise SystemExit(f"Executor pre-continuation stop mirror diverged: {metadata}")
                if metadata.get("operator_timebox_precontinuation_requires_fresh_timebox_review") is not True:
                    raise SystemExit(f"Executor pre-continuation timebox should require fresh review: {metadata}")
                if metadata.get("timebox_state") != "HELD_FOR_PARSEABLE_TIMEBOX":
                    raise SystemExit(f"Executor without a stop_at should hold the pre-continuation timebox: {metadata}")
                if metadata.get("operator_timebox_precontinuation_allows_continuation") is not False:
                    raise SystemExit(f"Executor missing timebox should not allow next continuation: {metadata}")
                if metadata.get("operator_timebox_precontinuation_blocks_continuation") is not True:
                    raise SystemExit(f"Executor missing timebox should explicitly block next continuation: {metadata}")
                if metadata.get("operator_supersession_precontinuation_contract_ready") is not True:
                    raise SystemExit(f"Executor missed pre-continuation supersession contract readiness: {metadata}")
                if metadata.get("operator_supersession_precontinuation_contract_state") != metadata.get("supersession_state"):
                    raise SystemExit(f"Executor pre-continuation supersession state mirror diverged: {metadata}")
                if metadata.get("operator_supersession_precontinuation_allows_continuation") != metadata.get("can_continue_under_latest_instruction"):
                    raise SystemExit(f"Executor pre-continuation supersession allow mirror diverged: {metadata}")
                if metadata.get("operator_supersession_precontinuation_blocks_continuation") != (
                    not bool(metadata.get("can_continue_under_latest_instruction"))
                ):
                    raise SystemExit(f"Executor pre-continuation supersession block mirror diverged: {metadata}")
                if metadata.get("operator_supersession_precontinuation_requires_stop") != metadata.get("latest_instruction_is_stop"):
                    raise SystemExit(f"Executor pre-continuation supersession stop mirror diverged: {metadata}")
                if metadata.get("operator_supersession_precontinuation_requires_fresh_latest_instruction_review") is not True:
                    raise SystemExit(f"Executor pre-continuation supersession should require fresh latest-instruction review: {metadata}")
                if metadata.get("operator_supersession_precontinuation_allows_continuation") is not False:
                    raise SystemExit(f"Executor without prior/latest instruction review should not allow supersession continuation: {metadata}")
                if metadata.get("operator_supersession_precontinuation_blocks_continuation") is not True:
                    raise SystemExit(f"Executor without prior/latest instruction review should explicitly block supersession continuation: {metadata}")
                if metadata.get("operator_supersession_precontinuation_contract_state") != "SUPERVISION_HELD_FOR_NEWEST_INSTRUCTION_REVIEW":
                    raise SystemExit(f"Executor should hold supersession until newest-instruction review is complete: {metadata}")
                if metadata.get("recovery_followthrough_packet_receipt_sha256") != metadata.get("receipt_sha256"):
                    raise SystemExit(f"Executor follow-through receipt hash diverged: {metadata}")
                if metadata.get("recovery_followthrough_packet_checkpoint_sha256") != metadata.get("checkpoint_sha256"):
                    raise SystemExit(f"Executor follow-through checkpoint hash diverged: {metadata}")
                assert_sha256(metadata.get("recovery_followthrough_token_sha256"), "Executor recovery follow-through token")
                if metadata.get("recovery_followthrough_token_reusable_for_future_recovery") is not False:
                    raise SystemExit(f"Executor should mark recovery follow-through token non-reusable: {metadata}")
                assert_recovery_followthrough_token_boundary(
                    metadata,
                    "Executor",
                    expected_source="checkpoint_recovery_execute",
                )
                if metadata.get("next_recovery_followthrough_requires_new_token") is not True:
                    raise SystemExit(f"Executor missed next recovery follow-through token boundary: {metadata}")
                if metadata.get("recovery_followthrough_packet_token_sha256") != metadata.get("recovery_followthrough_token_sha256"):
                    raise SystemExit(f"Executor follow-through token diverged from packet token: {metadata}")
                assert_local_safe_recovery_execution_token(
                    metadata,
                    "Checkpoint recovery executor",
                    expected_source="checkpoint_recovery_execute",
                )
                assert_recovery_execution_readiness_token(
                    metadata,
                    "Checkpoint recovery executor",
                    expected_source="checkpoint_recovery_execute",
                )
                if metadata.get("recovery_followthrough_packet_local_safe_execution_token_sha256") != metadata.get("local_safe_recovery_execution_token_sha256"):
                    raise SystemExit(f"Executor local-safe recovery execution token diverged from follow-through packet token: {metadata}")
                if metadata.get("normal_followthrough_allowed") is not True:
                    raise SystemExit(f"Executor should allow normal follow-through after closure: {metadata}")
                if metadata.get("recovery_closure_missing") != [] or metadata.get("recovery_closure_missing_count") != 0:
                    raise SystemExit(f"Executor missed explicit clean recovery-closure blocker contract: {metadata}")
                scorecard_rows = metadata.get("recovery_execution_scorecard_rows") or []
                expected_scorecard_items = {
                    "reviewed_step_permission",
                    "verification_target",
                    "receipt_hash_binding",
                    "checkpoint_hash_binding",
                    "stop_condition",
                    "risky_recovery_approval_boundary",
                    "followthrough_packet_ready",
                    "fresh_followthrough_token",
                }
                if metadata.get("recovery_execution_scorecard_row_count") != 8 or len(scorecard_rows) != 8:
                    raise SystemExit(f"Executor missed recovery execution scorecard rows: {metadata}")
                if {row.get("item") for row in scorecard_rows} != expected_scorecard_items:
                    raise SystemExit(f"Executor recovery execution scorecard items diverged: {metadata}")
                if metadata.get("recovery_execution_score") != 100 or metadata.get("recovery_execution_max_score") != 100:
                    raise SystemExit(f"Executor recovery execution score should be complete: {metadata}")
                if metadata.get("recovery_execution_required_rows_ready") is not True:
                    raise SystemExit(f"Executor recovery execution scorecard should be ready: {metadata}")
                if metadata.get("recovery_execution_scorecard_ready") is not True:
                    raise SystemExit(f"Executor recovery execution scorecard ready flag should be true: {metadata}")
                if any(
                    row.get("ready") is not True
                    or row.get("points") != row.get("max_points")
                    or row.get("required_before_normal_followthrough") is not True
                    or row.get("authorizes_risky_work") is not False
                    or row.get("authorizes_unreviewed_followthrough") is not False
                    for row in scorecard_rows
                ):
                    raise SystemExit(f"Executor recovery execution scorecard rows should be ready and non-authorizing: {metadata}")
                assert_recovery_execution_boundary(metadata, "Checkpoint recovery executor")
            if case.startswith("autonomy resume gate"):
                required = [
                    "Jarvis autonomy resume gate",
                    "read-only",
                    "Resume gate",
                    "AUTONOMY_RESUME_HELD_FOR_RECOVERY_COCKPIT",
                    "normal autonomous follow-through allowed: no",
                    "tools executed: no",
                    "approvals queued: no",
                    "Measured resume readiness scorecard",
                    "required rows ready: no",
                    "recovery_cockpit_ready",
                    "Bound proof",
                    "timebox state: STOP_WINDOW_ACTIVE",
                    "follow-through state: RECOVERY_FOLLOWTHROUGH_READY",
                    "checkpoint_recovery_cockpit_not_ready",
                    "approval readiness",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Autonomy resume gate missing expected context: {missing}")
                if result.tool_results[0].tool_name != "autonomy_resume_gate":
                    raise SystemExit(f"Natural autonomy resume should route to autonomy_resume_gate: {result.tool_results[0].tool_name}")
                metadata = result.tool_results[0].metadata
                if metadata.get("resume_gate_state") != "AUTONOMY_RESUME_HELD_FOR_RECOVERY_COCKPIT":
                    raise SystemExit(f"Autonomy resume gate should hold on seeded pending approval/cockpit blockers: {metadata}")
                if metadata.get("normal_autonomous_followthrough_allowed") is not False:
                    raise SystemExit(f"Autonomy resume gate should not resume with cockpit blockers: {metadata}")
                if metadata.get("timebox_state") != "STOP_WINDOW_ACTIVE":
                    raise SystemExit(f"Autonomy resume gate missed active timebox proof: {metadata}")
                if metadata.get("followthrough_state") != "RECOVERY_FOLLOWTHROUGH_READY":
                    raise SystemExit(f"Autonomy resume gate missed ready follow-through proof: {metadata}")
                assert_awake_guard_boundary(metadata, "Autonomy resume gate", expected_requested=False)
                if metadata.get("proof_queue_count") != len(metadata.get("proof_queue") or []):
                    raise SystemExit(f"Autonomy resume gate missed proof queue count: {metadata}")
                scorecard_rows = metadata.get("resume_readiness_scorecard_rows") or []
                expected_scorecard_items = {
                    "operator_timebox_active",
                    "recovery_cockpit_ready",
                    "latest_checkpoint_path_binding",
                    "latest_checkpoint_hash_binding",
                    "followthrough_closure_ready",
                    "artifact_hashes_match_files",
                    "risky_recovery_approval_boundary",
                    "fresh_followthrough_token",
                    "one_step_boundary_intact",
                }
                if metadata.get("resume_readiness_scorecard_row_count") != 9 or len(scorecard_rows) != 9:
                    raise SystemExit(f"Autonomy resume gate missed readiness scorecard rows: {metadata}")
                if {row.get("item") for row in scorecard_rows} != expected_scorecard_items:
                    raise SystemExit(f"Autonomy resume gate readiness scorecard items diverged: {metadata}")
                if metadata.get("resume_readiness_max_score") != 100:
                    raise SystemExit(f"Autonomy resume gate missed readiness max score: {metadata}")
                if not 0 <= int(metadata.get("resume_readiness_score") or -1) < 100:
                    raise SystemExit(f"Autonomy resume gate held path should score below complete: {metadata}")
                if metadata.get("resume_readiness_required_rows_ready") is not False:
                    raise SystemExit(f"Autonomy resume gate held path should report held readiness rows: {metadata}")
                if metadata.get("resume_readiness_scorecard_ready") is not False:
                    raise SystemExit(f"Autonomy resume gate held path should report scorecard not ready: {metadata}")
                assert_autonomy_resume_scorecard(metadata, "Autonomy resume gate held path", ready=False)
                assert_recovery_execution_boundary(metadata, "Autonomy resume gate", prefix="carried_recovery_execution")
                assert_carried_recovery_execution_readiness_token(
                    metadata,
                    "Autonomy resume gate",
                    expected_source="checkpoint_recovery_followthrough",
                )
                if any(
                    row.get("required_before_autonomy_resume") is not True
                    or row.get("authorizes_risky_work") is not False
                    or row.get("authorizes_unreviewed_continuation") is not False
                    for row in scorecard_rows
                ):
                    raise SystemExit(f"Autonomy resume gate readiness rows should be bounded and non-authorizing: {metadata}")
                for key in READ_ONLY_FALSE_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Autonomy resume gate should report {key}=False.")
            if case.startswith("autonomy continuation execution"):
                required = [
                    "Jarvis autonomy continuation execution packet",
                    "read-only",
                    "Continuation state",
                    "AUTONOMY_CONTINUATION_HELD",
                    "one local-safe step allowed: no",
                    "checkpoint_recovery_cockpit_not_ready",
                    "Measured continuation readiness scorecard",
                    "required rows ready: no",
                    "ready_resume_gate",
                    "Post-step proof queue",
                    "Approval boundary",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Autonomy continuation execution packet missing expected context: {missing}")
                if result.tool_results[0].tool_name != "autonomy_continuation_execution_packet":
                    raise SystemExit(f"Natural autonomy continuation should route to autonomy_continuation_execution_packet: {result.tool_results[0].tool_name}")
                metadata = result.tool_results[0].metadata
                if metadata.get("continuation_state") != "AUTONOMY_CONTINUATION_HELD":
                    raise SystemExit(f"Autonomy continuation should hold with seeded cockpit blockers: {metadata}")
                if metadata.get("one_local_safe_step_allowed") is not False:
                    raise SystemExit(f"Autonomy continuation should not allow a step with cockpit blockers: {metadata}")
                assert_awake_guard_boundary(metadata, "Autonomy continuation", expected_requested=False)
                if metadata.get("post_step_proof_queue_count") != len(metadata.get("post_step_proof_queue") or []):
                    raise SystemExit(f"Autonomy continuation missed post-step proof queue count: {metadata}")
                scorecard_rows = metadata.get("continuation_readiness_scorecard_rows") or []
                expected_scorecard_items = {
                    "ready_resume_gate",
                    "exact_next_step",
                    "post_step_verification_target",
                    "stop_condition",
                    "risky_work_approval_boundary",
                    "fresh_continuation_review_token",
                    "prior_cycle_proof_non_authorizing",
                    "one_step_contract_non_reusable",
                    "post_step_proof_queue_ready",
                }
                if metadata.get("continuation_readiness_scorecard_row_count") != 9 or len(scorecard_rows) != 9:
                    raise SystemExit(f"Autonomy continuation missed readiness scorecard rows: {metadata}")
                if {row.get("item") for row in scorecard_rows} != expected_scorecard_items:
                    raise SystemExit(f"Autonomy continuation readiness scorecard items diverged: {metadata}")
                if metadata.get("continuation_readiness_max_score") != 100:
                    raise SystemExit(f"Autonomy continuation missed readiness max score: {metadata}")
                if not 0 <= int(metadata.get("continuation_readiness_score") or -1) < 100:
                    raise SystemExit(f"Autonomy continuation held path should score below complete: {metadata}")
                if metadata.get("continuation_readiness_required_rows_ready") is not False:
                    raise SystemExit(f"Autonomy continuation held path should report held readiness rows: {metadata}")
                if metadata.get("continuation_readiness_scorecard_ready") is not False:
                    raise SystemExit(f"Autonomy continuation held path should report scorecard not ready: {metadata}")
                assert_autonomy_continuation_scorecard(
                    metadata,
                    "Autonomy continuation held path",
                    ready=False,
                )
                assert_recovery_execution_boundary(metadata, "Autonomy continuation", prefix="carried_recovery_execution")
                assert_carried_recovery_execution_readiness_token(
                    metadata,
                    "Autonomy continuation",
                    expected_source="checkpoint_recovery_followthrough",
                )
                if any(
                    row.get("required_before_one_local_safe_step") is not True
                    or row.get("authorizes_risky_work") is not False
                    or row.get("authorizes_batching") is not False
                    or row.get("authorizes_followup_without_closure") is not False
                    or row.get("prior_cycle_ledger_token_authorizes_action") is not False
                    for row in scorecard_rows
                ):
                    raise SystemExit(f"Autonomy continuation readiness rows should be bounded and non-authorizing: {metadata}")
                for key in READ_ONLY_FALSE_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Autonomy continuation execution should report {key}=False.")
            if case.startswith("autonomy step closure"):
                required = [
                    "Jarvis autonomy step closure packet",
                    "read-only",
                    "Closure state",
                    "AUTONOMY_STEP_CLOSURE_HELD",
                    "ready for next continuation review: no",
                    "post-step verification evidence",
                    "Required commands",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Autonomy step closure packet missing expected context: {missing}")
                if result.tool_results[0].tool_name != "autonomy_step_closure_packet":
                    raise SystemExit(f"Natural autonomy step closure should route to autonomy_step_closure_packet: {result.tool_results[0].tool_name}")
                metadata = result.tool_results[0].metadata
                if metadata.get("closure_state") != "AUTONOMY_STEP_CLOSURE_HELD":
                    raise SystemExit(f"Autonomy step closure should hold without closure proof: {metadata}")
                if metadata.get("ready_for_next_continuation_review") is not False:
                    raise SystemExit(f"Autonomy step closure should block the next review: {metadata}")
                assert_autonomy_step_closure_scorecard(
                    metadata,
                    "Autonomy step closure held path",
                    ready=False,
                )
                assert_awake_guard_boundary(metadata, "Autonomy step closure", expected_requested=False)
                assert_recovery_execution_boundary(metadata, "Autonomy step closure", prefix="carried_recovery_execution")
                assert_carried_recovery_execution_readiness_token(
                    metadata,
                    "Autonomy step closure",
                    expected_source="checkpoint_recovery_followthrough",
                )
                for key in READ_ONLY_FALSE_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Autonomy step closure should report {key}=False.")
            if case.startswith("autonomy cycle ledger"):
                required = [
                    "Jarvis autonomy cycle ledger",
                    "read-only",
                    "Cycle state",
                    "AUTONOMY_CYCLE_LEDGER_HELD",
                    "ready for fresh next continuation review: no",
                    "Cycle stages",
                    "Required cycle proof chain",
                    "Fresh-review boundary",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Autonomy cycle ledger missing expected context: {missing}")
                if result.tool_results[0].tool_name != "autonomy_cycle_ledger":
                    raise SystemExit(f"Natural autonomy cycle ledger should route to autonomy_cycle_ledger: {result.tool_results[0].tool_name}")
                metadata = result.tool_results[0].metadata
                if metadata.get("cycle_state") != "AUTONOMY_CYCLE_LEDGER_HELD":
                    raise SystemExit(f"Autonomy cycle ledger should hold without closure proof: {metadata}")
                if metadata.get("ready_for_fresh_next_continuation_review") is not False:
                    raise SystemExit(f"Autonomy cycle ledger should block fresh review without proof: {metadata}")
                if metadata.get("autonomy_cycle_ledger_ready") is not False:
                    raise SystemExit(f"Held autonomy cycle ledger should not expose all-up readiness: {metadata}")
                if _autonomy_cycle_ledger_ready_from_metadata(metadata):
                    raise SystemExit(f"Held autonomy cycle ledger passed all-up production validator: {metadata}")
                required_commands = metadata.get("required_commands") or []
                if not required_commands:
                    raise SystemExit(f"Autonomy cycle ledger missed required proof commands: {metadata}")
                if metadata.get("required_command_count") != len(required_commands):
                    raise SystemExit(f"Autonomy cycle ledger required command count diverged: {metadata}")
                if metadata.get("proof_queue") != required_commands:
                    raise SystemExit(f"Autonomy cycle ledger proof queue should mirror required commands: {metadata}")
                if metadata.get("proof_queue_count") != len(required_commands):
                    raise SystemExit(f"Autonomy cycle ledger proof queue count diverged: {metadata}")
                if metadata.get("next_required_command") != required_commands[0]:
                    raise SystemExit(f"Autonomy cycle ledger missed next required command: {metadata}")
                if metadata.get("next_proof_command") != required_commands[0]:
                    raise SystemExit(f"Autonomy cycle ledger missed next proof command: {metadata}")
                if metadata.get("stage_count") != len(metadata.get("stage_rows") or []):
                    raise SystemExit(f"Autonomy cycle ledger missed stage count: {metadata}")
                assert_autonomy_cycle_preflight_scorecard(
                    metadata,
                    "Autonomy cycle ledger held path",
                    ready=False,
                )
                assert_autonomy_step_closure_scorecard(
                    metadata,
                    "Autonomy cycle ledger held carried closure",
                    ready=False,
                    prefix="carried_step_closure",
                )
                assert_awake_guard_boundary(metadata, "Autonomy cycle ledger", expected_requested=False)
                assert_recovery_execution_boundary(metadata, "Autonomy cycle ledger", prefix="carried_recovery_execution")
                assert_carried_recovery_execution_readiness_token(
                    metadata,
                    "Autonomy cycle ledger",
                    expected_source="checkpoint_recovery_followthrough",
                )
                for key in READ_ONLY_FALSE_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Autonomy cycle ledger should report {key}=False.")
            if case.startswith("build target packet") or case == "what is the next build target for Jarvis":
                required = [
                    "Jarvis build target packet",
                    "read-only",
                    "Objective:",
                    "Priority goal:",
                    "Finish Jarvis V2 as an AI agent harness",
                    "Selected target",
                    "approval_visibility",
                    "readiness visibility",
                    "Likely owning files",
                    "Target integrity",
                    "file check: TARGETS_EXIST",
                    "missing files: none",
                    "Smoke-test target",
                    "Acceptance checks",
                    "execution health report",
                    "Verification-to-learning handoff",
                    "acceptance gate:",
                    "evidence <receipt>",
                    "tests <verification>",
                    "recovery <rollback or stop condition>",
                    "First boundary",
                    "Do not approve approval",
                    "Stop conditions",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Build target packet missing expected context: {missing}")
                if result.tool_results[0].tool_name != "build_target_packet":
                    raise SystemExit(f"Natural build target should route to build_target_packet: {result.tool_results[0].tool_name}")
                metadata = result.tool_results[0].metadata
                if metadata.get("target_kind") != "approval_visibility":
                    raise SystemExit("Build target packet should prioritize approval visibility when an approval is pending.")
                if metadata.get("target_file_integrity_status") != "TARGETS_EXIST":
                    raise SystemExit(f"Build target packet should prove target files exist: {metadata}")
                if metadata.get("target_files_exist") is not True or metadata.get("missing_target_files") != [] or metadata.get("missing_target_file_count") != 0:
                    raise SystemExit(f"Build target packet reported stale target files: {metadata}")
                if metadata.get("target_files_checked") != metadata.get("likely_files") or metadata.get("target_files_checked") != len(metadata.get("likely_file_paths", [])):
                    raise SystemExit(f"Build target packet target file count diverged: {metadata}")
                if any(not row.get("exists") for row in metadata.get("target_file_rows", [])):
                    raise SystemExit(f"Build target packet target file rows include missing files: {metadata}")
                expected_focused_verification = [
                    "python3 -m jarvis_v2.scripts.smoke_test_next_step",
                    "python3 -m compileall jarvis_v2",
                ]
                if metadata.get("focused_verification_commands") != expected_focused_verification:
                    raise SystemExit(f"Build target packet missed focused verification commands: {metadata}")
                if metadata.get("focused_verification_command_count") != len(expected_focused_verification):
                    raise SystemExit(f"Build target packet missed focused verification command count: {metadata}")
                if metadata.get("aggregate_verification_command") != "python3 -m jarvis_v2.scripts.smoke_test_all":
                    raise SystemExit(f"Build target packet missed aggregate verification command: {metadata}")
                if metadata.get("acceptance_gate_command") != "acceptance gate: <changed behavior>; evidence <receipt>; tests <verification>; recovery <rollback or stop condition>":
                    raise SystemExit(f"Build target packet missed machine-readable acceptance gate command: {metadata}")
                if metadata.get("failed_action_runs", 0) + metadata.get("approval_held_action_runs", 0) < 1:
                    raise SystemExit(f"Build target packet missed execution health metadata: {metadata}")
                if metadata.get("failed_action_runs", 0) and metadata.get("execution_health_review_required") is not True:
                    raise SystemExit(f"Build target packet missed failed-run review metadata: {metadata}")
                if metadata.get("execution_health_next_command") not in metadata.get("execution_health_next_commands", []):
                    raise SystemExit(f"Build target packet missed execution health recovery queue: {metadata}")
                assert_execution_health_proof_chain(metadata, "Build target packet")
                assert_execution_health_learning_handoff(metadata, "Build target packet")
                assert_build_target_packet_handoff(metadata, "Build target packet")
                for key in READ_ONLY_FALSE_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Build target packet should report {key}=False.")
            if case.startswith("harness build slice") or case in {"continue building Jarvis V2 as an agent harness", "pick next harness slice for Jarvis V2"}:
                required = [
                    "Jarvis harness build slice",
                    "read-only",
                    "Priority goal:",
                    "Finish Jarvis V2 as an AI agent harness",
                    "Selected slice",
                    "Owning files to inspect first",
                    "Target integrity",
                    "file check: TARGETS_EXIST",
                    "missing files: none",
                    "Build loop",
                    "Focused verification",
                    "Starter harness commands",
                    "Acceptance checks",
                    "Completion proof handoff",
                    "selected slice packet",
                    "evidence ledger",
                    "completion claim gate",
                    "Verification-to-learning handoff",
                    "acceptance gate:",
                    "evidence <receipt>",
                    "tests <verification>",
                    "recovery <rollback or stop condition>",
                    "Approval boundary",
                    "does not approve",
                    "Context signals",
                    "execution learning debt visible: yes",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Harness build slice missing expected context: {missing}")
                if result.tool_results[0].tool_name != "harness_build_slice":
                    raise SystemExit(f"Jarvis build request should route to harness_build_slice: {result.tool_results[0].tool_name}")
                metadata = result.tool_results[0].metadata
                if "personal connector" in case and metadata.get("target_kind") != "personal_integration_harness":
                    raise SystemExit(f"Harness build slice should pick personal integration when requested: {metadata}")
                if "personal connector" in case and "integration execution matrix: email" not in metadata.get("starter_command_names", []):
                    raise SystemExit(f"Harness build slice should seed integration execution matrix for personal connectors: {metadata}")
                if "personal connector" in case and "integration adapter acceptance: email" not in metadata.get("starter_command_names", []):
                    raise SystemExit(f"Harness build slice should seed adapter acceptance for personal connectors: {metadata}")
                if "personal connector" in case and "integration dry run contract: email -> search mailbox metadata; target selected source; time selected window; data metadata-only; verification fake rows only" not in metadata.get("starter_command_names", []):
                    raise SystemExit(f"Harness build slice should seed dry-run row-contract proof before metadata preview: {metadata}")
                if "personal connector" in case and "integration enablement gate: email -> search mailbox metadata; target selected source; time selected window; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed" not in metadata.get("starter_command_names", []):
                    raise SystemExit(f"Harness build slice should seed connector enablement gate after adapter acceptance: {metadata}")
                if "personal connector" in case and "integration proof bundle: email -> search mailbox metadata; target selected source; time selected window; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed" not in metadata.get("starter_command_names", []):
                    raise SystemExit(f"Harness build slice should seed connector proof bundle before implementation review: {metadata}")
                if "personal connector" in case and "integration implementation review: email -> search mailbox metadata; target selected source; time selected window; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed; status status api smoke passed" not in metadata.get("starter_command_names", []):
                    raise SystemExit(f"Harness build slice should seed connector implementation review after proof bundle: {metadata}")
                if metadata.get("target_file_integrity_status") != "TARGETS_EXIST":
                    raise SystemExit(f"Harness build slice should prove target files exist: {metadata}")
                if metadata.get("target_files_exist") is not True or metadata.get("missing_target_files") != [] or metadata.get("missing_target_file_count") != 0:
                    raise SystemExit(f"Harness build slice reported stale target files: {metadata}")
                if metadata.get("target_files_checked") != metadata.get("likely_files") or metadata.get("target_files_checked") != len(metadata.get("likely_file_paths", [])):
                    raise SystemExit(f"Harness build slice target file count diverged: {metadata}")
                if any(not row.get("exists") for row in metadata.get("target_file_rows", [])):
                    raise SystemExit(f"Harness build slice target file rows include missing files: {metadata}")
                if "personal connector" in case:
                    expected_closure = [
                        "harness build slice: personal connector readiness; focus email",
                        "completion audit: improve AGI gate personal integrations",
                        "evidence ledger",
                        "completion claim gate: improve AGI gate personal integrations",
                    ]
                    if metadata.get("evidence_closure_commands") != expected_closure:
                        raise SystemExit(f"Harness build slice missed personal connector evidence closure commands: {metadata}")
                    if metadata.get("evidence_closure_command_count") != len(expected_closure):
                        raise SystemExit(f"Harness build slice missed evidence closure count: {metadata}")
                    if metadata.get("focused_verification_commands") != [
                        "python3 -m py_compile jarvis_v2/tools/personal.py jarvis_v2/tools/registry.py jarvis_v2/agent/planner.py jarvis_v2/tools/help.py jarvis_v2/tools/capabilities.py jarvis_v2/tools/harness.py jarvis_v2/ui/status_server.py",
                        "python3 -m jarvis_v2.scripts.smoke_test_personal",
                        "python3 -m jarvis_v2.scripts.smoke_test_harness",
                        "python3 -m jarvis_v2.scripts.smoke_test_status_server",
                        "python3 -m compileall -q jarvis_v2",
                    ]:
                        raise SystemExit(f"Harness build slice missed focused verification commands: {metadata}")
                    acceptance_names = metadata.get("acceptance_check_names", [])
                    for expected in [
                        "`integration dry run contract: email` proves the metadata row contract before metadata preview or promotion.",
                        "`integration proof bundle: email` passes metadata preview, disabled adapter acceptance, metadata row contract, enablement gate, and rehearsal receipt before implementation review.",
                        "`integration promotion gate: email` carries metadata row contract proof, row limit, blocked payload fields, tests, audit, rollback, and approval boundaries before implementation spec.",
                        "`integration implementation review: email` requires proof bundle, metadata row contract, preflight row-contract proof, implementation spec, status/API smoke evidence, audit, verification, and rollback before code review.",
                        "`integration adapter acceptance: email` passes metadata happy path, row limit, full-content blocked, side-effect blocked, and missing-scope blocked cases.",
                        "`integration enablement gate` reports disabled adapter acceptance proof before review is allowed.",
                        "Natural-language connector auto-routing remains disabled until focused personal and status smoke tests pass.",
                    ]:
                        if expected not in acceptance_names:
                            raise SystemExit(f"Harness build slice missed personal connector acceptance check: {expected} in {metadata}")
                if "dashboard" in case and metadata.get("target_kind") != "command_first_dashboard":
                    raise SystemExit(f"Harness build slice should pick dashboard when requested: {metadata}")
                if metadata.get("likely_files", 0) < 2 or metadata.get("focused_tests", 0) < 2 or metadata.get("starter_commands", 0) < 2 or metadata.get("acceptance_checks", 0) < 3:
                    raise SystemExit(f"Harness build slice should expose files and tests: {metadata}")
                if metadata.get("execution_health_next_command") not in metadata.get("execution_health_next_commands", []):
                    raise SystemExit(f"Harness build slice missed execution health recovery queue: {metadata}")
                if metadata.get("failed_action_runs", 0) and metadata.get("recovery_debt_visible") is not True:
                    raise SystemExit(f"Harness build slice should expose visible recovery debt: {metadata}")
                if metadata.get("execution_learning_debt_visible") is not True:
                    raise SystemExit(f"Harness build slice should expose visible execution learning debt: {metadata}")
                if metadata.get("recovery_or_learning_debt_visible") is not True:
                    raise SystemExit(f"Harness build slice should expose combined recovery-or-learning debt visibility: {metadata}")
                if metadata.get("failed_action_runs", 0) + metadata.get("approval_held_action_runs", 0) < 1:
                    raise SystemExit(f"Harness build slice should expose execution-health attention: {metadata}")
                if metadata.get("read_only_slice_allowed_with_recovery_debt") is not True:
                    raise SystemExit(f"Harness build slice should allow the read-only selector while recovery debt is visible: {metadata}")
                if metadata.get("recovery_debt_blocks_current_slice") is not False:
                    raise SystemExit(f"Harness build slice should not mark the read-only selector blocked by recovery debt: {metadata}")
                assert_execution_health_proof_chain(metadata, "Harness build slice")
                assert_execution_health_learning_handoff(metadata, "Harness build slice")
                for key in READ_ONLY_FALSE_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Harness build slice should report {key}=False.")
            if case.startswith("save build target packet"):
                required = [
                    "Build target packet saved:",
                    "Jarvis build target packet",
                    "Priority goal:",
                    "Finish Jarvis V2 as an AI agent harness",
                    "Selected target",
                    "approval_visibility",
                    "Do not approve approval",
                    "Stop conditions",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Saved build target packet response missing expected context: {missing}")
                metadata = result.tool_results[0].metadata
                if metadata.get("target_kind") != "approval_visibility":
                    raise SystemExit("Saved build target packet should preserve the selected target kind.")
                if not metadata.get("writes_notes") or not metadata.get("writes_files"):
                    raise SystemExit("Saved build target packet should report writes_notes=True and writes_files=True.")
                assert_vault_relative_save_receipt(
                    result,
                    expected_path=build_target_note,
                    expected_display="Automations/Build Target Packet.md",
                    label="Saved build target packet",
                )
                for key in [flag for flag in READ_ONLY_FALSE_FLAGS if flag not in {"writes_files", "writes_notes"}]:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Saved build target packet should report {key}=False.")
                if not build_target_note.exists():
                    raise SystemExit("Build Target Packet note was not written.")
                note_text = build_target_note.read_text(encoding="utf-8")
                for expected in [
                    "# Build Target Packet",
                    "## Receipt",
                    "Generated:",
                    "Objective: continue building Jarvis V2 safely",
                    "Target kind: approval_visibility",
                    "Pending approvals seen:",
                    "Save action: local Obsidian note write only.",
                    "does not call a model",
                    "## Resume Commands",
                    "continuation packet: continue Jarvis V2 safely",
                    "pending approvals",
                    "## Packet",
                    "Jarvis build target packet",
                    "approval_visibility",
                    "Do not approve approval",
                    "Smoke-test target",
                    "acceptance gate:",
                ]:
                    if expected not in note_text:
                        raise SystemExit(f"Build Target Packet note missing expected context: {expected}")
            if case == "save work queue":
                assert_vault_relative_save_receipt(
                    result,
                    expected_path=work_queue_note,
                    expected_display="Automations/Work Queue.md",
                    label="Saved work queue",
                )
                if not work_queue_note.exists():
                    raise SystemExit("Work Queue note was not written.")
                note_text = work_queue_note.read_text(encoding="utf-8")
                required = [
                    "# Work Queue",
                    "Jarvis work queue:",
                    "Priority goal:",
                    "Finish Jarvis V2 as an AI agent harness",
                    "Approval #",
                    "approval readiness",
                    "approval packet",
                    "review Jarvis safe next actions",
                    "Build safe assistant",
                    "save handoff brief",
                ]
                missing = [item for item in required if item not in note_text]
                if missing:
                    raise SystemExit(f"Work Queue note missing expected context: {missing}")
            if case == "mission control":
                if not mission_control.exists():
                    raise SystemExit("Mission Control note was not written.")
                assert_vault_relative_save_receipt(
                    result,
                    expected_path=mission_control,
                    expected_display="Automations/Mission Control.md",
                    label="mission control",
                )
                note_text = mission_control.read_text(encoding="utf-8")
                required = [
                    "# Mission Control",
                    "Safe next actions:",
                    "Approval #",
                    "approval readiness",
                    "review Jarvis safe next actions",
                    "Build safe assistant",
                    "Do not auto-run shell",
                ]
                missing = [item for item in required if item not in note_text]
                if missing:
                    raise SystemExit(f"Mission Control note missing expected context: {missing}")
            if case == "help continuity":
                for expected in ["safe next actions", "next action packet", "priority stack", "continuation packet", "autonomy cycle ledger", "build target packet", "harness build slice", "save build target packet", "what should Jarvis do next?", "work queue", "save work queue", "mission control", "receipt_sha256=<hash>", "checkpoint_sha256=<hash>"]:
                    if expected not in result.response:
                        raise SystemExit(f"Continuity help missing: {expected}")

        paused_state_runtime = make_temp_runtime(Path(temp) / "paused-state-snapshot-next-step")
        scheduled_state = paused_state_runtime.handle("schedule assistant basics")
        if not scheduled_state.verified:
            raise SystemExit("Paused State Snapshot fixture could not schedule assistant basics.")
        paused_state = paused_state_runtime.handle("pause job State Snapshot")
        if not paused_state.verified:
            raise SystemExit("Paused State Snapshot fixture could not pause State Snapshot.")
        paused_cases = [
            ("safe next actions", "safe_next_actions"),
            ("next action packet", "next_action_packet"),
            ("work queue", "work_queue"),
            ("readiness report", "readiness_report"),
            ("morning startup", "morning_startup"),
            ("priority stack", "priority_stack"),
            ("continuation packet", "continuation_packet"),
            ("build target packet", "build_target_packet"),
        ]
        for command_text, expected_tool in paused_cases:
            paused_result = paused_state_runtime.handle(command_text)
            print(f"[ok] paused State Snapshot recommendation: {command_text}")
            print(paused_result.response[:1200])
            print()
            if not paused_result.verified or paused_result.tool_results[0].tool_name != expected_tool:
                raise SystemExit(f"Paused State Snapshot command routed unexpectedly: {command_text}")
            if "resume job State Snapshot" not in paused_result.response:
                raise SystemExit(f"Paused State Snapshot command should recommend resume: {command_text}")
            if "schedule assistant basics" in paused_result.response and command_text != "readiness report":
                raise SystemExit(f"Paused State Snapshot command should not prefer rescheduling defaults: {command_text}")
            paused_metadata = paused_result.tool_results[0].metadata
            for key in READ_ONLY_FALSE_FLAGS:
                if paused_metadata.get(key) is not False:
                    raise SystemExit(f"Paused State Snapshot recommendation should stay read-only for {key}: {paused_metadata}")
            if command_text in {"safe next actions", "next action packet", "work queue", "priority stack", "continuation packet", "build target packet"}:
                if paused_metadata.get("scheduler_next_command") != "resume job State Snapshot":
                    raise SystemExit(f"Paused State Snapshot next-step metadata missed resume command: {paused_metadata}")
                if paused_metadata.get("disabled_state_snapshot_jobs") != 1 or paused_metadata.get("enabled_state_snapshot_jobs") != 0:
                    raise SystemExit(f"Paused State Snapshot counters were wrong: {paused_metadata}")
            if command_text == "build target packet" and "duplicate default jobs" not in paused_result.response:
                raise SystemExit("Build target packet should explain why paused State Snapshot needs resume instead of duplicate defaults.")
            if command_text == "next action packet" and paused_metadata.get("action_kind") != "resume_state_snapshot":
                raise SystemExit(f"Paused State Snapshot next action should select resume_state_snapshot: {paused_metadata}")
            if command_text == "next action packet":
                assert_next_action_packet_handoff(paused_metadata, "paused State Snapshot next action packet")
            if command_text == "safe next actions":
                assert_safe_next_actions_handoff(paused_metadata, "paused State Snapshot safe next actions")
            if command_text == "work queue":
                assert_work_queue_handoff(paused_metadata, "paused State Snapshot work queue")
            if command_text == "priority stack":
                assert_priority_stack_handoff(paused_metadata, "paused State Snapshot priority stack")
            if command_text == "readiness report":
                if paused_metadata.get("state_snapshot_next_command") != "resume job State Snapshot":
                    raise SystemExit(f"Readiness report missed paused State Snapshot resume command: {paused_metadata}")
                if paused_metadata.get("disabled_state_snapshot_jobs") != 1 or paused_metadata.get("enabled_state_snapshot_jobs") != 0:
                    raise SystemExit(f"Readiness report paused State Snapshot counters were wrong: {paused_metadata}")

        missing_compaction_runtime = make_temp_runtime(Path(temp) / "missing-compaction-next-step")
        missing_compaction_runtime.store.upsert_job("State Snapshot", 360, "state_snapshot", "2099-01-01T00:00:00")
        for command_text, expected_tool in [
            ("safe next actions", "safe_next_actions"),
            ("next action packet", "next_action_packet"),
            ("work queue", "work_queue"),
            ("priority stack", "priority_stack"),
            ("continuation packet", "continuation_packet"),
            ("build target packet", "build_target_packet"),
        ]:
            missing_result = missing_compaction_runtime.handle(command_text)
            print(f"[ok] missing Conversation Compaction recommendation: {command_text}")
            print(missing_result.response[:1200])
            print()
            if not missing_result.verified or missing_result.tool_results[0].tool_name != expected_tool:
                raise SystemExit(f"Missing Conversation Compaction command routed unexpectedly: {command_text}")
            should_print_scheduler_command = command_text in {
                "next action packet",
                "priority stack",
                "continuation packet",
                "build target packet",
            }
            if should_print_scheduler_command and "schedule assistant basics" not in missing_result.response:
                raise SystemExit(f"Missing Conversation Compaction command should recommend assistant basics: {command_text}")
            missing_metadata = missing_result.tool_results[0].metadata
            for key in READ_ONLY_FALSE_FLAGS:
                if missing_metadata.get(key) is not False:
                    raise SystemExit(f"Missing Conversation Compaction recommendation should stay read-only for {key}: {missing_metadata}")
            if missing_metadata.get("scheduler_next_command") != "schedule assistant basics":
                raise SystemExit(f"Missing Conversation Compaction metadata missed assistant basics: {missing_metadata}")
            if command_text == "next action packet" and missing_metadata.get("action_kind") != "schedule_basics":
                raise SystemExit(f"Missing Conversation Compaction next action should select schedule_basics: {missing_metadata}")
            if command_text == "build target packet" and "Conversation Compaction" not in missing_result.response:
                raise SystemExit("Build target packet should explain the missing Conversation Compaction job.")

        paused_compaction_runtime = make_temp_runtime(Path(temp) / "paused-compaction-next-step")
        scheduled_compaction = paused_compaction_runtime.handle("schedule assistant basics")
        if not scheduled_compaction.verified:
            raise SystemExit("Paused Conversation Compaction fixture could not schedule assistant basics.")
        paused_compaction = paused_compaction_runtime.handle(f"pause job {COMPACTION_JOB_NAME}")
        if not paused_compaction.verified:
            raise SystemExit("Paused Conversation Compaction fixture could not pause Conversation Compaction.")
        for command_text, expected_tool in [
            ("safe next actions", "safe_next_actions"),
            ("next action packet", "next_action_packet"),
            ("work queue", "work_queue"),
            ("priority stack", "priority_stack"),
            ("continuation packet", "continuation_packet"),
            ("build target packet", "build_target_packet"),
        ]:
            paused_compaction_result = paused_compaction_runtime.handle(command_text)
            print(f"[ok] paused Conversation Compaction recommendation: {command_text}")
            print(paused_compaction_result.response[:1200])
            print()
            if not paused_compaction_result.verified or paused_compaction_result.tool_results[0].tool_name != expected_tool:
                raise SystemExit(f"Paused Conversation Compaction command routed unexpectedly: {command_text}")
            should_print_scheduler_command = command_text in {
                "next action packet",
                "priority stack",
                "continuation packet",
                "build target packet",
            }
            if should_print_scheduler_command and f"resume job {COMPACTION_JOB_NAME}" not in paused_compaction_result.response:
                raise SystemExit(f"Paused Conversation Compaction command should recommend resume: {command_text}")
            paused_compaction_metadata = paused_compaction_result.tool_results[0].metadata
            for key in READ_ONLY_FALSE_FLAGS:
                if paused_compaction_metadata.get(key) is not False:
                    raise SystemExit(f"Paused Conversation Compaction recommendation should stay read-only for {key}: {paused_compaction_metadata}")
            if paused_compaction_metadata.get("scheduler_next_command") != f"resume job {COMPACTION_JOB_NAME}":
                raise SystemExit(f"Paused Conversation Compaction metadata missed resume command: {paused_compaction_metadata}")
            if command_text == "next action packet" and paused_compaction_metadata.get("action_kind") != "resume_conversation_compaction":
                raise SystemExit(f"Paused Conversation Compaction next action should select resume_conversation_compaction: {paused_compaction_metadata}")
            if command_text == "build target packet" and "Conversation Compaction" not in paused_compaction_result.response:
                raise SystemExit("Build target packet should explain the paused Conversation Compaction job.")

        risky_recovery_execute = runtime.registry.get("checkpoint_recovery_execute").handler(
            {
                "objective": "continue Jarvis safely",
                "reviewed": "true",
                "step": "run python script to inspect local recovery state",
                "verification": "verify shell command output",
                "files": "local python script",
                "outcome": "pending risky recovery approval",
                "blockers": "none",
            }
        )
        print("[ok] direct risky checkpoint recovery execute held for approval proof")
        print(risky_recovery_execute.output[:1800])
        print()
        if risky_recovery_execute.ok:
            raise SystemExit("Risky checkpoint recovery execute should hold without approval reference.")
        risky_recovery_metadata = risky_recovery_execute.metadata
        expected_risky_recovery_queue = [
            "approval readiness <id>",
            "approval packet <id>",
            "approval chain proof <id>",
            "verification receipt <approved run id from approval chain proof <id>>",
            "checkpoint recovery execute: <reviewed local-safe recovery step after approval proof>",
        ]
        expected_approval_boundary_items = {
            "risk_classification",
            "exact_arguments_or_step",
            "approval_readiness_packet",
            "last_look_approval_packet",
            "approval_chain_proof",
            "post_approval_verification_receipt",
            "fresh_local_safe_review_after_approval",
        }
        if "Risky recovery approval proof queue" not in risky_recovery_execute.output:
            raise SystemExit("Risky checkpoint recovery execute output missed approval proof queue.")
        if risky_recovery_metadata.get("risky_recovery_without_approval") is not True:
            raise SystemExit(f"Risky checkpoint recovery execute should require approval reference: {risky_recovery_metadata}")
        if risky_recovery_metadata.get("recovery_step_approval_proof_queue") != expected_risky_recovery_queue:
            raise SystemExit(f"Risky checkpoint recovery execute approval queue diverged: {risky_recovery_metadata}")
        if risky_recovery_metadata.get("recovery_step_approval_proof_queue_count") != len(expected_risky_recovery_queue):
            raise SystemExit(f"Risky checkpoint recovery execute missed approval queue count: {risky_recovery_metadata}")
        if risky_recovery_metadata.get("recovery_step_next_approval_proof_command") != "approval readiness <id>":
            raise SystemExit(f"Risky checkpoint recovery execute missed next approval command: {risky_recovery_metadata}")
        if risky_recovery_metadata.get("recovery_step_approval_required_before_recovery") is not True:
            raise SystemExit(f"Risky checkpoint recovery execute missed approval-before-recovery flag: {risky_recovery_metadata}")
        risky_recovery_rows = risky_recovery_metadata.get("recovery_step_approval_boundary_rows") or []
        if risky_recovery_metadata.get("recovery_step_approval_boundary_row_count") != 7 or len(risky_recovery_rows) != 7:
            raise SystemExit(f"Risky checkpoint recovery execute missed approval boundary rows: {risky_recovery_metadata}")
        if {row.get("item") for row in risky_recovery_rows} != expected_approval_boundary_items:
            raise SystemExit(f"Risky checkpoint recovery execute approval boundary rows diverged: {risky_recovery_metadata}")
        if risky_recovery_metadata.get("recovery_step_approval_boundary_ready") is not False:
            raise SystemExit(f"Risky checkpoint recovery execute approval boundary should stay held: {risky_recovery_metadata}")
        if any(risky_recovery_metadata.get(flag) is not False for flag in _RECOVERY_STEP_APPROVAL_BOUNDARY_FALSE_FLAGS):
            raise SystemExit(f"Risky checkpoint recovery execute boundary should stay non-authorizing: {risky_recovery_metadata}")
        assert_recovery_step_approval_boundary_token(risky_recovery_metadata, "Risky checkpoint recovery execute")
        assert_checkpoint_recovery_execute_handoff(risky_recovery_metadata, "Risky checkpoint recovery execute")
        if any(
            row.get("required_before_risky_recovery_step") is not True
            or row.get("status") != "held"
            or row.get("authorizes_action_now") is not False
            or row.get("authorizes_risky_work") is not False
            or row.get("authorizes_unreviewed_followthrough") is not False
            for row in risky_recovery_rows
        ):
            raise SystemExit(f"Risky checkpoint recovery execute rows should be held and non-authorizing: {risky_recovery_metadata}")
        if risky_recovery_metadata.get("normal_followthrough_allowed") is not False:
            raise SystemExit(f"Risky checkpoint recovery execute must not allow follow-through: {risky_recovery_metadata}")
        for key in READ_ONLY_FALSE_FLAGS:
            if risky_recovery_metadata.get(key) is not False:
                raise SystemExit(f"Risky checkpoint recovery execute should report {key}=False.")

        referenced_risky_recovery_execute = runtime.registry.get("checkpoint_recovery_execute").handler(
            {
                "objective": "continue Jarvis safely",
                "reviewed": "true",
                "step": "run python script to inspect local recovery state",
                "approval": "approval chain proof 1",
                "blockers": "none",
            }
        )
        print("[ok] direct referenced risky checkpoint recovery execute remains held with bound approval reference")
        print(referenced_risky_recovery_execute.output[:1800])
        print()
        if referenced_risky_recovery_execute.ok:
            raise SystemExit("Referenced risky checkpoint recovery execute should remain held without verification.")
        referenced_risky_metadata = referenced_risky_recovery_execute.metadata
        referenced_risky_handoff = referenced_risky_metadata.get("checkpoint_recovery_execute_handoff") or {}
        if referenced_risky_metadata.get("approval_reference_provided") is not False:
            raise SystemExit(f"Referenced risky recovery should reject incomplete approval-chain proof: {referenced_risky_metadata}")
        if referenced_risky_handoff.get("approval_reference"):
            raise SystemExit(f"Referenced risky recovery handoff should not bind incomplete approval proof: {referenced_risky_metadata}")
        if referenced_risky_metadata.get("recovery_step_approval_required_before_recovery") is not True:
            raise SystemExit(f"Referenced risky recovery should still require complete approval proof: {referenced_risky_metadata}")
        assert_checkpoint_recovery_execute_handoff(
            referenced_risky_metadata,
            "Referenced risky checkpoint recovery execute",
        )

        incomplete_approval_risky_recovery_execute = runtime.registry.get("checkpoint_recovery_execute").handler(
            {
                "objective": "continue Jarvis safely",
                "reviewed": "true",
                "step": "run python script to inspect local recovery state",
                "verification": "verify shell command output",
                "files": "local python script",
                "outcome": "incomplete approval proof should stay held",
                "approval": "approval chain proof 1",
                "blockers": "none",
            }
        )
        print("[ok] direct incomplete-approval risky checkpoint recovery execute remains held")
        print(incomplete_approval_risky_recovery_execute.output[:1800])
        print()
        if incomplete_approval_risky_recovery_execute.ok:
            raise SystemExit("Incomplete approval proof should not record a risky recovery step.")
        incomplete_approval_metadata = incomplete_approval_risky_recovery_execute.metadata
        if incomplete_approval_metadata.get("approval_reference_provided") is not False:
            raise SystemExit(f"Incomplete approval proof should not satisfy risky recovery approval: {incomplete_approval_metadata}")
        if incomplete_approval_metadata.get("recovery_step_approval_required_before_recovery") is not True:
            raise SystemExit(f"Incomplete approval proof should keep approval-before-recovery required: {incomplete_approval_metadata}")
        if "approval_reference_for_risky_recovery_step" not in incomplete_approval_metadata.get("missing_fields", []):
            raise SystemExit(f"Incomplete approval proof should be reported as missing approval proof: {incomplete_approval_metadata}")
        assert_checkpoint_recovery_execute_handoff(
            incomplete_approval_metadata,
            "Incomplete-approval risky checkpoint recovery execute",
        )

        approved_risky_recovery_execute = runtime.registry.get("checkpoint_recovery_execute").handler(
            {
                "objective": "continue Jarvis safely",
                "reviewed": "true",
                "step": "run python script to inspect local recovery state",
                "verification": "verify shell command output",
                "files": "local python script",
                "outcome": "approved risky recovery proof recorded",
                "approval": "approval readiness 1; approval packet 1; approval chain proof 1",
                "blockers": "none",
            }
        )
        print("[ok] direct approved risky checkpoint recovery execute records boundary proof")
        print(approved_risky_recovery_execute.output[:1800])
        print()
        if not approved_risky_recovery_execute.ok:
            raise SystemExit("Approved risky checkpoint recovery execute should record after approval reference.")
        approved_risky_metadata = approved_risky_recovery_execute.metadata
        assert_recovery_step_approval_boundary_token(
            approved_risky_metadata,
            "Approved risky checkpoint recovery execute",
        )
        if approved_risky_metadata.get("recovery_step_approval_boundary_as_prior_proof") is not True:
            raise SystemExit(f"Approved risky recovery should carry boundary token as prior proof: {approved_risky_metadata}")
        if approved_risky_metadata.get("recovery_step_requires_approval") is not True:
            raise SystemExit(f"Approved risky recovery should keep requires-approval evidence: {approved_risky_metadata}")
        if approved_risky_metadata.get("approval_reference_provided") is not True:
            raise SystemExit(f"Approved risky recovery missed approval reference evidence: {approved_risky_metadata}")
        if approved_risky_metadata.get("normal_followthrough_allowed") is not True:
            raise SystemExit(f"Approved risky recovery should allow normal follow-through after proof: {approved_risky_metadata}")

        missing_checkpoint_runtime = make_temp_runtime(Path(temp) / "missing-checkpoint-route")
        missing_checkpoint_cockpit = missing_checkpoint_runtime.registry.get("checkpoint_recovery_cockpit").handler(
            {
                "objective": "continue Jarvis safely after missing checkpoint",
                "stop_at": "2026-06-09T08:10:00+09:00",
                "current_time": "2026-06-08T20:55:00+09:00",
                "timezone": "Asia/Seoul",
            }
        )
        print("[ok] missing checkpoint routes to recovery review")
        print(missing_checkpoint_cockpit.output[:1800])
        print()
        missing_checkpoint_metadata = missing_checkpoint_cockpit.metadata
        if missing_checkpoint_metadata.get("can_resume_local_safe_review") is not False:
            raise SystemExit(f"Missing checkpoint should block local-safe resume review: {missing_checkpoint_metadata}")
        if missing_checkpoint_metadata.get("checkpoint_needs_review") is not True:
            raise SystemExit(f"Missing checkpoint should require recovery review: {missing_checkpoint_metadata}")
        if missing_checkpoint_metadata.get("checkpoint_freshness") != "missing":
            raise SystemExit(f"Missing checkpoint should expose missing freshness: {missing_checkpoint_metadata}")
        if "checkpoint missing" not in missing_checkpoint_metadata.get("blockers", []):
            raise SystemExit(f"Missing checkpoint should appear in cockpit blockers: {missing_checkpoint_metadata}")
        assert_checkpoint_route_boundary(
            missing_checkpoint_metadata,
            "Missing checkpoint cockpit",
            expected_source="checkpoint_recovery_cockpit",
            expected_route_status="route_to_recovery_review",
        )
        assert_recovery_cockpit_scorecard(
            missing_checkpoint_metadata,
            "Missing checkpoint cockpit",
            strict_ready=False,
        )

        stale_checkpoint_runtime = make_temp_runtime(Path(temp) / "stale-checkpoint-route")
        stale_checkpoint = stale_checkpoint_runtime.registry.get("save_work_block_checkpoint").handler(
            {"objective": "stale autonomy checkpoint route smoke", "limit": 3}
        )
        stale_checkpoint_path = Path(str(stale_checkpoint.metadata.get("path") or ""))
        stale_mtime = time.time() - (8 * 60 * 60)
        os.utime(stale_checkpoint_path, (stale_mtime, stale_mtime))
        stale_checkpoint_cockpit = stale_checkpoint_runtime.registry.get("checkpoint_recovery_cockpit").handler(
            {
                "objective": "continue Jarvis safely after stale checkpoint",
                "stop_at": "2026-06-09T08:10:00+09:00",
                "current_time": "2026-06-08T20:55:00+09:00",
                "timezone": "Asia/Seoul",
            }
        )
        print("[ok] stale checkpoint routes to recovery review")
        print(stale_checkpoint_cockpit.output[:1800])
        print()
        stale_checkpoint_metadata = stale_checkpoint_cockpit.metadata
        if stale_checkpoint_metadata.get("can_resume_local_safe_review") is not False:
            raise SystemExit(f"Stale checkpoint should block local-safe resume review: {stale_checkpoint_metadata}")
        if stale_checkpoint_metadata.get("checkpoint_needs_review") is not True:
            raise SystemExit(f"Stale checkpoint should require recovery review: {stale_checkpoint_metadata}")
        if stale_checkpoint_metadata.get("checkpoint_freshness") != "stale":
            raise SystemExit(f"Stale checkpoint should expose stale freshness: {stale_checkpoint_metadata}")
        if "checkpoint stale" not in stale_checkpoint_metadata.get("blockers", []):
            raise SystemExit(f"Stale checkpoint should appear in cockpit blockers: {stale_checkpoint_metadata}")
        assert_checkpoint_route_boundary(
            stale_checkpoint_metadata,
            "Stale checkpoint cockpit",
            expected_source="checkpoint_recovery_cockpit",
            expected_route_status="route_to_recovery_review",
        )
        assert_recovery_cockpit_scorecard(
            stale_checkpoint_metadata,
            "Stale checkpoint cockpit",
            strict_ready=False,
        )

        review_again_checkpoint_runtime = make_temp_runtime(Path(temp) / "review-again-checkpoint-route")
        review_again_checkpoint = review_again_checkpoint_runtime.registry.get("save_work_block_checkpoint").handler(
            {"objective": "review-again autonomy checkpoint route smoke", "limit": 3}
        )
        review_again_checkpoint_path = Path(str(review_again_checkpoint.metadata.get("path") or ""))
        review_again_mtime = time.time() - (3 * 60 * 60)
        os.utime(review_again_checkpoint_path, (review_again_mtime, review_again_mtime))
        review_again_checkpoint_cockpit = review_again_checkpoint_runtime.registry.get("checkpoint_recovery_cockpit").handler(
            {
                "objective": "continue Jarvis safely after review-again checkpoint",
                "stop_at": "2026-06-09T08:10:00+09:00",
                "current_time": "2026-06-08T20:55:00+09:00",
                "timezone": "Asia/Seoul",
            }
        )
        print("[ok] review-again checkpoint routes to recovery review")
        print(review_again_checkpoint_cockpit.output[:1800])
        print()
        review_again_checkpoint_metadata = review_again_checkpoint_cockpit.metadata
        if review_again_checkpoint_metadata.get("can_resume_local_safe_review") is not False:
            raise SystemExit(f"Review-again checkpoint should block local-safe resume review: {review_again_checkpoint_metadata}")
        if review_again_checkpoint_metadata.get("checkpoint_needs_review") is not True:
            raise SystemExit(f"Review-again checkpoint should require recovery review: {review_again_checkpoint_metadata}")
        if review_again_checkpoint_metadata.get("checkpoint_freshness") != "review_again":
            raise SystemExit(f"Review-again checkpoint should expose review_again freshness: {review_again_checkpoint_metadata}")
        if "checkpoint review_again" not in review_again_checkpoint_metadata.get("blockers", []):
            raise SystemExit(f"Review-again checkpoint should appear in cockpit blockers: {review_again_checkpoint_metadata}")
        assert_checkpoint_route_boundary(
            review_again_checkpoint_metadata,
            "Review-again checkpoint cockpit",
            expected_source="checkpoint_recovery_cockpit",
            expected_route_status="route_to_recovery_review",
        )
        assert_recovery_cockpit_scorecard(
            review_again_checkpoint_metadata,
            "Review-again checkpoint cockpit",
            strict_ready=False,
        )

        ready_runtime = make_temp_runtime(Path(temp) / "ready-autonomy-resume")
        checkpoint = ready_runtime.registry.get("save_work_block_checkpoint").handler(
            {"objective": "ready autonomy resume smoke checkpoint", "limit": 3}
        )
        checkpoint_path = checkpoint.metadata.get("path")
        checkpoint_continuation = ready_runtime.registry.get("continuation_packet").handler(
            {"objective": "continue Jarvis safely after ready checkpoint"}
        )
        checkpoint_continuation_metadata = checkpoint_continuation.metadata
        expected_checkpoint_display = "Reflections/" + Path(str(checkpoint_path)).name
        if checkpoint_continuation_metadata.get("checkpoint_path") != checkpoint_path:
            raise SystemExit(
                f"Continuation packet should preserve exact checkpoint path metadata: {checkpoint_continuation_metadata}"
            )
        if checkpoint_continuation_metadata.get("checkpoint_path_display") != expected_checkpoint_display:
            raise SystemExit(
                f"Continuation packet should include vault-relative checkpoint display metadata: {checkpoint_continuation_metadata}"
            )
        if expected_checkpoint_display not in checkpoint_continuation.output:
            raise SystemExit(f"Continuation packet output missed safe checkpoint display: {checkpoint_continuation.output}")
        if str(Path(temp)) in checkpoint_continuation.output or "/private/" in checkpoint_continuation.output or "/\x55sers/" in checkpoint_continuation.output:
            raise SystemExit(f"Continuation packet leaked local checkpoint path in output: {checkpoint_continuation.output}")
        assert_continuation_packet_handoff(
            checkpoint_continuation_metadata,
            "Ready checkpoint continuation packet",
            objective="continue Jarvis safely after ready checkpoint",
        )
        ready_checkpoint_cockpit = ready_runtime.registry.get("checkpoint_recovery_cockpit").handler(
            {
                "objective": "continue Jarvis safely after ready checkpoint",
                "stop_at": "2026-06-09T08:10:00+09:00",
                "current_time": "2026-06-08T20:55:00+09:00",
                "timezone": "Asia/Seoul",
            }
        )
        print("[ok] ready checkpoint cockpit scorecard")
        print(ready_checkpoint_cockpit.output[:1800])
        print()
        ready_checkpoint_cockpit_metadata = ready_checkpoint_cockpit.metadata
        if ready_checkpoint_cockpit_metadata.get("can_resume_local_safe_review") is not True:
            raise SystemExit(f"Ready checkpoint cockpit should allow local-safe review: {ready_checkpoint_cockpit_metadata}")
        assert_checkpoint_route_boundary(
            ready_checkpoint_cockpit_metadata,
            "Ready checkpoint cockpit",
            expected_source="checkpoint_recovery_cockpit",
        )
        assert_recovery_cockpit_scorecard(
            ready_checkpoint_cockpit_metadata,
            "Ready checkpoint cockpit",
            strict_ready=True,
        )
        ready_recovery_apply = ready_runtime.registry.get("checkpoint_recovery_apply_packet").handler(
            {"objective": "continue Jarvis safely after ready checkpoint"}
        )
        ready_recovery_apply_metadata = ready_recovery_apply.metadata
        ready_recovery_apply_queue = ready_recovery_apply_metadata.get("checkpoint_recovery_proof_queue") or []
        if not ready_recovery_apply_queue:
            raise SystemExit(f"Ready checkpoint recovery apply packet missed proof queue: {ready_recovery_apply_metadata}")
        if ready_recovery_apply_metadata.get("checkpoint_recovery_proof_queue_count") != len(ready_recovery_apply_queue):
            raise SystemExit(f"Ready checkpoint recovery apply packet missed proof queue count: {ready_recovery_apply_metadata}")
        if ready_recovery_apply_metadata.get("checkpoint_recovery_next_proof_command") != ready_recovery_apply_queue[0]:
            raise SystemExit(f"Ready checkpoint recovery apply packet missed next proof command: {ready_recovery_apply_metadata}")
        if ready_recovery_apply_metadata.get("recovery_apply_proof_queue") != ready_recovery_apply_queue:
            raise SystemExit(f"Ready checkpoint recovery apply packet missed apply-specific proof queue parity: {ready_recovery_apply_metadata}")
        if ready_recovery_apply_metadata.get("recovery_apply_proof_queue_count") != len(ready_recovery_apply_queue):
            raise SystemExit(f"Ready checkpoint recovery apply packet missed apply-specific proof queue count: {ready_recovery_apply_metadata}")
        if ready_recovery_apply_metadata.get("recovery_apply_next_proof_command") != ready_recovery_apply_queue[0]:
            raise SystemExit(f"Ready checkpoint recovery apply packet missed apply-specific next proof command: {ready_recovery_apply_metadata}")
        recovery_receipt_path = Path(temp) / "ready-recovery-receipt.md"
        recovery_receipt_path.write_text("ready autonomy recovery receipt\n", encoding="utf-8")
        recovery_receipt_sha256 = file_sha256(recovery_receipt_path)
        mismatched_recovery_receipt_sha256 = "0" * 64
        recovery_checkpoint_sha256 = file_sha256(checkpoint_path)
        mismatched_recovery_checkpoint_sha256 = "1" * 64
        post_step_receipt_path = Path(temp) / "post-step-receipt.md"
        post_step_receipt_path.write_text("post-step autonomy receipt\n", encoding="utf-8")
        post_step_receipt_sha256 = file_sha256(post_step_receipt_path)
        mismatched_post_step_receipt_sha256 = "2" * 64
        post_step_checkpoint_path = Path(temp) / "post-step-checkpoint.md"
        post_step_checkpoint_path.write_text("post-step autonomy checkpoint\n", encoding="utf-8")
        post_step_checkpoint_sha256 = file_sha256(post_step_checkpoint_path)
        mismatched_post_step_checkpoint_sha256 = "3" * 64
        ready_gate = ready_runtime.registry.get("autonomy_resume_gate").handler(
            {
                "objective": "continue Jarvis safely",
                "stop_at": "2099-01-01T00:00:00+09:00",
                "current_time": "2026-06-09T03:00:00+09:00",
                "timezone": "Asia/Seoul",
                "reviewed_step": "reviewed local-safe continuity metadata",
                "verification": "smoke_test_next_step passed",
                "receipt_path": str(recovery_receipt_path),
                "receipt_sha256": recovery_receipt_sha256,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": recovery_checkpoint_sha256,
                "stop_condition": "stop if verification drifts",
                "blockers": "none",
            }
        )
        print("[ok] direct ready autonomy resume gate")
        print(ready_gate.output[:1800])
        print()
        if not ready_gate.ok:
            raise SystemExit("Ready autonomy resume gate should run.")
        ready_metadata = ready_gate.metadata
        if ready_metadata.get("resume_gate_state") != "AUTONOMY_RESUME_READY_FOR_LOCAL_SAFE_CONTINUATION":
            raise SystemExit(f"Ready autonomy resume gate should permit local-safe continuation: {ready_metadata}")
        if ready_metadata.get("normal_autonomous_followthrough_allowed") is not True:
            raise SystemExit(f"Ready autonomy resume gate missed normal follow-through permission: {ready_metadata}")
        if ready_metadata.get("autonomy_resume_gate_ready") is not True:
            raise SystemExit(f"Ready autonomy resume gate missed all-up ready flag: {ready_metadata}")
        if not _autonomy_resume_gate_ready_from_metadata(ready_metadata):
            raise SystemExit(f"Ready autonomy resume gate failed all-up production validator: {ready_metadata}")
        if ready_metadata.get("timebox_state") != "STOP_WINDOW_ACTIVE" or ready_metadata.get("followthrough_state") != "RECOVERY_FOLLOWTHROUGH_READY":
            raise SystemExit(f"Ready autonomy resume gate missed bound timebox/follow-through proof: {ready_metadata}")
        assert_autonomy_resume_scorecard(ready_metadata, "Ready autonomy resume gate", ready=True)
        assert_timebox_review_contract(ready_metadata, "Ready autonomy resume gate")
        assert_operator_supersession_token_boundary(
            ready_metadata,
            "Ready autonomy resume gate",
            expected_source="operator_instruction_supersession",
        )
        assert_checkpoint_route_boundary(
            ready_metadata,
            "Ready autonomy resume gate",
            expected_source="checkpoint_recovery_cockpit",
        )
        if ready_metadata.get("checkpoint_path_matches_latest") is not True:
            raise SystemExit(f"Ready autonomy resume gate missed latest checkpoint path binding: {ready_metadata}")
        if ready_metadata.get("checkpoint_hash_matches_latest") is not True:
            raise SystemExit(f"Ready autonomy resume gate missed latest checkpoint hash binding: {ready_metadata}")
        if ready_metadata.get("latest_checkpoint_sha256") != recovery_checkpoint_sha256:
            raise SystemExit(f"Ready autonomy resume gate latest checkpoint hash diverged: {ready_metadata}")
        if ready_metadata.get("recovery_artifact_hashes_present") is not True:
            raise SystemExit(f"Ready autonomy resume gate missed recovery artifact hash binding: {ready_metadata}")
        if ready_metadata.get("receipt_file_sha256") != recovery_receipt_sha256:
            raise SystemExit(f"Ready autonomy resume gate missed recovery receipt file hash: {ready_metadata}")
        if ready_metadata.get("receipt_hash_matches_file") is not True:
            raise SystemExit(f"Ready autonomy resume gate missed recovery receipt file binding: {ready_metadata}")
        if ready_metadata.get("recovery_checkpoint_hash_matches_file") is not True:
            raise SystemExit(f"Ready autonomy resume gate missed recovery checkpoint file binding: {ready_metadata}")
        if ready_metadata.get("receipt_sha256") != recovery_receipt_sha256 or ready_metadata.get("checkpoint_sha256") != recovery_checkpoint_sha256:
            raise SystemExit(f"Ready autonomy resume gate hash values diverged: {ready_metadata}")
        assert_sha256(ready_metadata.get("recovery_followthrough_token_sha256"), "Ready autonomy resume recovery token")
        if ready_metadata.get("recovery_followthrough_token_reusable_for_future_recovery") is not False:
            raise SystemExit(f"Ready autonomy resume gate should mark recovery token non-reusable: {ready_metadata}")
        assert_recovery_followthrough_token_boundary(
            ready_metadata,
            "Ready autonomy resume gate",
            expected_source="checkpoint_recovery_followthrough",
        )
        if ready_metadata.get("next_recovery_followthrough_requires_new_token") is not True:
            raise SystemExit(f"Ready autonomy resume gate missed recovery token boundary: {ready_metadata}")
        assert_local_safe_recovery_execution_token(
            ready_metadata,
            "Ready autonomy resume gate",
            expected_source="autonomy_resume_gate",
        )
        assert_carried_recovery_execution_readiness_token(
            ready_metadata,
            "Ready autonomy resume gate",
            expected_source="checkpoint_recovery_followthrough",
        )
        carried_recovery_rows = ready_metadata.get("carried_recovery_execution_scorecard_rows") or []
        expected_carried_recovery_items = {
            "reviewed_step_permission",
            "verification_target",
            "receipt_hash_binding",
            "checkpoint_hash_binding",
            "stop_condition",
            "risky_recovery_approval_boundary",
            "followthrough_packet_ready",
            "fresh_followthrough_token",
        }
        if ready_metadata.get("carried_recovery_execution_scorecard_row_count") != 8 or len(carried_recovery_rows) != 8:
            raise SystemExit(f"Ready autonomy resume gate missed carried recovery execution rows: {ready_metadata}")
        if {row.get("item") for row in carried_recovery_rows} != expected_carried_recovery_items:
            raise SystemExit(f"Ready autonomy resume gate carried recovery rows diverged: {ready_metadata}")
        if ready_metadata.get("carried_recovery_execution_score") != 100 or ready_metadata.get("carried_recovery_execution_max_score") != 100:
            raise SystemExit(f"Ready autonomy resume gate carried recovery score should be complete: {ready_metadata}")
        if ready_metadata.get("carried_recovery_execution_required_rows_ready") is not True:
            raise SystemExit(f"Ready autonomy resume gate carried recovery rows should be ready: {ready_metadata}")
        if ready_metadata.get("carried_recovery_execution_scorecard_ready") is not True:
            raise SystemExit(f"Ready autonomy resume gate carried recovery scorecard should be ready: {ready_metadata}")
        if ready_metadata.get("carried_recovery_execution_as_prior_proof") is not True:
            raise SystemExit(f"Ready autonomy resume gate should carry recovery execution as prior proof: {ready_metadata}")
        if (
            ready_metadata.get("carried_recovery_execution_authorizes_action_now") is not False
            or ready_metadata.get("carried_recovery_execution_authorizes_risky_work") is not False
            or ready_metadata.get("carried_recovery_execution_authorizes_unreviewed_followthrough") is not False
        ):
            raise SystemExit(f"Ready autonomy resume gate carried recovery proof should be non-authorizing: {ready_metadata}")
        if any(
            row.get("ready") is not True
            or row.get("points") != row.get("max_points")
            or row.get("required_before_normal_followthrough") is not True
            or row.get("authorizes_risky_work") is not False
            or row.get("authorizes_unreviewed_followthrough") is not False
            for row in carried_recovery_rows
        ):
            raise SystemExit(f"Ready autonomy resume gate carried recovery rows should be ready and non-authorizing: {ready_metadata}")
        if ready_metadata.get("blockers"):
            raise SystemExit(f"Ready autonomy resume gate should have no blockers: {ready_metadata}")
        for key, value in [
            ("blockers", ["stale_blocker"]),
            ("blocker_count", 1),
            ("normal_autonomous_followthrough_allowed", False),
            ("next_safe_command", "continue without review"),
            ("proof_queue", []),
            ("proof_queue_count", 999),
            ("next_proof_command", "wrong next proof"),
            ("resume_readiness_scorecard_ready", False),
            ("resume_readiness_required_rows_ready", False),
            ("recovery_artifact_hashes_match_files", False),
            ("checkpoint_path_matches_latest", False),
            ("checkpoint_hash_matches_latest", False),
            ("tools_executed", True),
            ("queues_approval", True),
        ]:
            tampered_metadata = json.loads(json.dumps(ready_metadata))
            tampered_metadata[key] = value
            if _autonomy_resume_gate_ready_from_metadata(tampered_metadata):
                raise SystemExit(f"Ready autonomy resume validator accepted stale {key}: {tampered_metadata}")
        tampered_metadata = json.loads(json.dumps(ready_metadata))
        tampered_metadata.pop("blockers", None)
        if _autonomy_resume_gate_ready_from_metadata(tampered_metadata):
            raise SystemExit(f"Ready autonomy resume validator accepted missing blockers: {tampered_metadata}")
        tampered_metadata = json.loads(json.dumps(ready_metadata))
        tampered_metadata.pop("next_proof_command", None)
        if _autonomy_resume_gate_ready_from_metadata(tampered_metadata):
            raise SystemExit(f"Ready autonomy resume validator accepted missing next proof command: {tampered_metadata}")
        for key in READ_ONLY_FALSE_FLAGS:
            if ready_metadata.get(key) is not False:
                raise SystemExit(f"Ready autonomy resume gate should report {key}=False.")

        mismatched_checkpoint_gate = ready_runtime.registry.get("autonomy_resume_gate").handler(
            {
                "objective": "continue Jarvis safely",
                "stop_at": "2099-01-01T00:00:00+09:00",
                "current_time": "2026-06-09T03:00:00+09:00",
                "timezone": "Asia/Seoul",
                "reviewed_step": "reviewed local-safe continuity metadata",
                "verification": "smoke_test_next_step passed",
                "receipt_path": str(recovery_receipt_path),
                "receipt_sha256": recovery_receipt_sha256,
                "checkpoint_path": "/tmp/not-the-latest-checkpoint.md",
                "checkpoint_sha256": recovery_checkpoint_sha256,
                "stop_condition": "stop if verification drifts",
                "blockers": "none",
            }
        )
        print("[ok] direct mismatched checkpoint autonomy resume gate")
        print(mismatched_checkpoint_gate.output[:1800])
        print()
        mismatch_metadata = mismatched_checkpoint_gate.metadata
        if mismatch_metadata.get("resume_gate_state") != "AUTONOMY_RESUME_HELD_FOR_CHECKPOINT_BINDING":
            raise SystemExit(f"Mismatched checkpoint should hold resume gate: {mismatch_metadata}")
        if mismatch_metadata.get("normal_autonomous_followthrough_allowed") is not False:
            raise SystemExit(f"Mismatched checkpoint should block normal follow-through: {mismatch_metadata}")
        if mismatch_metadata.get("autonomy_resume_gate_ready") is not False:
            raise SystemExit(f"Mismatched checkpoint should not expose resume readiness: {mismatch_metadata}")
        if _autonomy_resume_gate_ready_from_metadata(mismatch_metadata):
            raise SystemExit(f"Mismatched checkpoint should fail all-up resume validator: {mismatch_metadata}")
        if mismatch_metadata.get("checkpoint_path_matches_latest") is not False:
            raise SystemExit(f"Mismatched checkpoint should report failed latest binding: {mismatch_metadata}")
        if "checkpoint_path_not_latest_recovery_checkpoint" not in mismatch_metadata.get("blockers", []):
            raise SystemExit(f"Mismatched checkpoint should expose checkpoint binding blocker: {mismatch_metadata}")
        for key in READ_ONLY_FALSE_FLAGS:
            if mismatch_metadata.get(key) is not False:
                raise SystemExit(f"Mismatched checkpoint resume gate should report {key}=False.")

        mismatched_checkpoint_hash_gate = ready_runtime.registry.get("autonomy_resume_gate").handler(
            {
                "objective": "continue Jarvis safely",
                "stop_at": "2099-01-01T00:00:00+09:00",
                "current_time": "2026-06-09T03:00:00+09:00",
                "timezone": "Asia/Seoul",
                "reviewed_step": "reviewed local-safe continuity metadata",
                "verification": "smoke_test_next_step passed",
                "receipt_path": str(recovery_receipt_path),
                "receipt_sha256": recovery_receipt_sha256,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": mismatched_recovery_checkpoint_sha256,
                "stop_condition": "stop if verification drifts",
                "blockers": "none",
            }
        )
        print("[ok] direct mismatched checkpoint hash autonomy resume gate")
        print(mismatched_checkpoint_hash_gate.output[:1800])
        print()
        mismatch_hash_metadata = mismatched_checkpoint_hash_gate.metadata
        if mismatch_hash_metadata.get("resume_gate_state") != "AUTONOMY_RESUME_HELD_FOR_CHECKPOINT_HASH_BINDING":
            raise SystemExit(f"Mismatched checkpoint hash should hold resume gate: {mismatch_hash_metadata}")
        if mismatch_hash_metadata.get("normal_autonomous_followthrough_allowed") is not False:
            raise SystemExit(f"Mismatched checkpoint hash should block normal follow-through: {mismatch_hash_metadata}")
        if mismatch_hash_metadata.get("checkpoint_path_matches_latest") is not True:
            raise SystemExit(f"Mismatched checkpoint hash should still bind latest path: {mismatch_hash_metadata}")
        if mismatch_hash_metadata.get("checkpoint_hash_matches_latest") is not False:
            raise SystemExit(f"Mismatched checkpoint hash should report failed hash binding: {mismatch_hash_metadata}")
        if "checkpoint_hash_not_latest_recovery_checkpoint" not in mismatch_hash_metadata.get("blockers", []):
            raise SystemExit(f"Mismatched checkpoint hash should expose hash binding blocker: {mismatch_hash_metadata}")
        for key in READ_ONLY_FALSE_FLAGS:
            if mismatch_hash_metadata.get(key) is not False:
                raise SystemExit(f"Mismatched checkpoint hash resume gate should report {key}=False.")

        invalid_hash_gate = ready_runtime.registry.get("autonomy_resume_gate").handler(
            {
                "objective": "continue Jarvis safely",
                "stop_at": "2099-01-01T00:00:00+09:00",
                "current_time": "2026-06-09T03:00:00+09:00",
                "timezone": "Asia/Seoul",
                "reviewed_step": "reviewed local-safe continuity metadata",
                "verification": "smoke_test_next_step passed",
                "receipt_path": str(recovery_receipt_path),
                "receipt_sha256": "not-a-sha",
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": "also-not-a-sha",
                "stop_condition": "stop if verification drifts",
                "blockers": "none",
            }
        )
        print("[ok] direct invalid-hash autonomy resume gate")
        print(invalid_hash_gate.output[:1800])
        print()
        invalid_hash_metadata = invalid_hash_gate.metadata
        if invalid_hash_metadata.get("resume_gate_state") != "AUTONOMY_RESUME_HELD_FOR_CHECKPOINT_HASH_BINDING":
            raise SystemExit(f"Invalid recovery hashes should hold resume gate: {invalid_hash_metadata}")
        if invalid_hash_metadata.get("recovery_artifact_hashes_present") is not False:
            raise SystemExit(f"Invalid recovery hashes should not count as present: {invalid_hash_metadata}")
        for expected_blocker in ["checkpoint_hash_not_latest_recovery_checkpoint", "followthrough_closure_not_ready"]:
            if expected_blocker not in invalid_hash_metadata.get("blockers", []):
                raise SystemExit(f"Invalid recovery hashes should expose blocker {expected_blocker}: {invalid_hash_metadata}")
        for blocker in ["valid checkpoint recovery receipt sha256", "valid fresh work-block checkpoint sha256"]:
            if blocker not in invalid_hash_metadata.get("recovery_closure_missing", []):
                raise SystemExit(f"Invalid recovery hashes missed blocker {blocker!r}: {invalid_hash_metadata}")
        for key in READ_ONLY_FALSE_FLAGS:
            if invalid_hash_metadata.get(key) is not False:
                raise SystemExit(f"Invalid-hash resume gate should report {key}=False.")

        ready_execution = ready_runtime.registry.get("autonomy_continuation_execution_packet").handler(
            {
                "objective": "continue Jarvis safely",
                "stop_at": "2099-01-01T00:00:00+09:00",
                "current_time": "2026-06-09T03:00:00+09:00",
                "timezone": "Asia/Seoul",
                "reviewed_step": "reviewed local-safe continuity metadata",
                "verification": "smoke_test_next_step passed",
                "receipt_path": str(recovery_receipt_path),
                "receipt_sha256": recovery_receipt_sha256,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": recovery_checkpoint_sha256,
                "stop_condition": "stop if verification drifts",
                "blockers": "none",
                "next_step": "update local continuity smoke metadata",
                "next_verification": "smoke_test_next_step passed",
            }
        )
        print("[ok] direct ready autonomy continuation execution packet")
        print(ready_execution.output[:1800])
        print()
        if not ready_execution.ok:
            raise SystemExit("Ready autonomy continuation execution packet should run.")
        execution_metadata = ready_execution.metadata
        if execution_metadata.get("continuation_state") != "AUTONOMY_CONTINUATION_READY_FOR_ONE_LOCAL_SAFE_STEP":
            raise SystemExit(f"Ready autonomy continuation should permit one local-safe step: {execution_metadata}")
        assert_timebox_review_contract(execution_metadata, "Ready autonomy continuation")
        assert_operator_supersession_token_boundary(
            execution_metadata,
            "Ready autonomy continuation",
            expected_source="operator_instruction_supersession",
        )
        assert_checkpoint_route_boundary(
            execution_metadata,
            "Ready autonomy continuation",
            expected_source="checkpoint_recovery_cockpit",
        )
        if execution_metadata.get("one_local_safe_step_allowed") is not True:
            raise SystemExit(f"Ready autonomy continuation missed one-step permission: {execution_metadata}")
        if execution_metadata.get("autonomy_continuation_execution_ready") is not True:
            raise SystemExit(f"Ready autonomy continuation missed all-up readiness flag: {execution_metadata}")
        if not _autonomy_continuation_execution_ready_from_metadata(execution_metadata):
            raise SystemExit(f"Ready autonomy continuation all-up validator rejected metadata: {execution_metadata}")
        for tampered_key, tampered_value in [
            ("continuation_state", "AUTONOMY_CONTINUATION_HELD"),
            ("one_local_safe_step_allowed", False),
            ("normal_autonomous_followthrough_allowed", False),
            ("missing_blockers", ["tampered"]),
            ("missing_blocker_count", 1),
            ("next_safe_command", "tampered"),
            ("post_step_proof_queue_count", 999),
            ("post_step_next_proof_command", "tampered"),
            ("continuation_readiness_score", 95),
            ("continuation_readiness_required_rows_ready", False),
            ("continuation_readiness_scorecard_ready", False),
            ("resume_gate_state", "AUTONOMY_RESUME_HELD"),
            ("timebox_review_contract_ready", False),
            ("awake_guard_os_wake_lock_boundary_ready", False),
            ("can_continue_under_latest_instruction", False),
            ("supersession_token_boundary_ready", False),
            ("checkpoint_route_boundary_ready", False),
            ("checkpoint_path_matches_latest", False),
            ("checkpoint_hash_matches_latest", False),
            ("recovery_artifact_hashes_match_files", False),
            ("recovery_followthrough_token_boundary_ready", False),
            ("local_safe_recovery_execution_token_boundary_ready", False),
            ("recovery_execution_readiness_token_boundary_ready", False),
            ("carried_recovery_execution_score", 95),
            ("carried_recovery_execution_scorecard_ready", False),
            ("continuation_review_token_present", False),
            ("prior_cycle_ledger_token_present", True),
            ("prior_cycle_ledger_token_boundary_ready", False),
            ("prior_cycle_ledger_token_reusable_for_this_review", True),
            ("prior_cycle_ledger_token_reusable_for_this_closure", True),
            ("prior_cycle_ledger_token_reusable_for_this_cycle", True),
            ("prior_cycle_ledger_proof_authorizes_post_step_closure", True),
            ("prior_cycle_ledger_proof_authorizes_new_action", True),
            ("prior_cycle_ledger_proof_authorizes_model_call", True),
            ("prior_cycle_ledger_proof_authorizes_tool_execution", True),
            ("prior_cycle_ledger_proof_authorizes_personal_data_read", True),
            ("prior_cycle_ledger_proof_authorizes_external_side_effect", True),
            ("one_step_execution_contract_ready", False),
            ("one_step_execution_contract_token_as_prior_proof", True),
            ("one_step_execution_contract_binds_awake_guard", "false"),
            ("one_step_execution_contract_binds_operator_supersession", "false"),
            ("one_step_execution_contract_binds_timebox_review_contract", "false"),
            ("one_step_execution_contract_all_local_safe_step_limited", "false"),
            ("one_step_execution_contract_all_non_reusable", "false"),
            ("one_step_execution_contract_all_risky_work_gated", "false"),
            ("one_step_execution_contract_requires_fresh_closure", "false"),
            ("proposed_next_step_requires_approval", True),
            ("proposed_next_step_risk_signals", ["shell/code"]),
            ("proposed_next_step_approval_proof_queue_count", 999),
            ("proposed_next_step_approval_boundary_row_count", 999),
            ("proposed_next_step_approval_boundary_token_present", False),
            ("proposed_next_step_approval_boundary_ready", False),
            ("proposed_next_step_approval_boundary_as_prior_proof", False),
            ("proposed_next_step_approval_required_before_review", True),
            ("proposed_next_step_next_approval_proof_command", "tampered"),
            ("proposed_next_step_approval_boundary_authorizes_approval", True),
            ("proposed_next_step_approval_boundary_authorizes_model_call", True),
            ("proposed_next_step_approval_boundary_authorizes_tool_execution", True),
            ("proposed_next_step_approval_boundary_authorizes_personal_data_read", True),
            ("proposed_next_step_approval_boundary_authorizes_external_side_effect", True),
            ("proposed_next_step_approval_boundary_authorizes_timebox_reuse", True),
            ("proposed_next_step_approval_boundary_reusable_for_recovery_review", True),
            ("queues_approval", True),
            ("controls_computer", True),
        ]:
            tampered = dict(execution_metadata)
            tampered[tampered_key] = tampered_value
            if _autonomy_continuation_execution_ready_from_metadata(tampered):
                raise SystemExit(
                    f"Ready autonomy continuation all-up validator accepted tampered {tampered_key}: {tampered}"
                )
        assert_autonomy_continuation_scorecard(
            execution_metadata,
            "Ready autonomy continuation",
            ready=True,
        )
        if execution_metadata.get("checkpoint_path_matches_latest") is not True:
            raise SystemExit(f"Ready autonomy continuation missed checkpoint binding proof: {execution_metadata}")
        if execution_metadata.get("checkpoint_hash_matches_latest") is not True:
            raise SystemExit(f"Ready autonomy continuation missed checkpoint hash binding proof: {execution_metadata}")
        if execution_metadata.get("latest_checkpoint_path") != checkpoint_path:
            raise SystemExit(f"Ready autonomy continuation missed latest checkpoint path: {execution_metadata}")
        if execution_metadata.get("latest_checkpoint_sha256") != recovery_checkpoint_sha256:
            raise SystemExit(f"Ready autonomy continuation missed latest checkpoint hash: {execution_metadata}")
        if execution_metadata.get("recovery_artifact_hashes_present") is not True:
            raise SystemExit(f"Ready autonomy continuation missed recovery artifact hash binding: {execution_metadata}")
        if execution_metadata.get("receipt_hash_matches_file") is not True:
            raise SystemExit(f"Ready autonomy continuation missed recovery receipt file binding: {execution_metadata}")
        if execution_metadata.get("recovery_checkpoint_hash_matches_file") is not True:
            raise SystemExit(f"Ready autonomy continuation missed recovery checkpoint file binding: {execution_metadata}")
        if execution_metadata.get("receipt_sha256") != recovery_receipt_sha256 or execution_metadata.get("checkpoint_sha256") != recovery_checkpoint_sha256:
            raise SystemExit(f"Ready autonomy continuation hash values diverged: {execution_metadata}")
        assert_sha256(execution_metadata.get("recovery_followthrough_token_sha256"), "Ready autonomy continuation recovery token")
        if execution_metadata.get("recovery_followthrough_token_sha256") != ready_metadata.get("recovery_followthrough_token_sha256"):
            raise SystemExit(f"Ready autonomy continuation recovery token diverged from resume gate: {execution_metadata}")
        if execution_metadata.get("recovery_followthrough_token_reusable_for_future_recovery") is not False:
            raise SystemExit(f"Ready autonomy continuation should mark recovery token non-reusable: {execution_metadata}")
        assert_recovery_followthrough_token_boundary(
            execution_metadata,
            "Ready autonomy continuation",
            expected_source="checkpoint_recovery_followthrough",
        )
        if execution_metadata.get("next_recovery_followthrough_requires_new_token") is not True:
            raise SystemExit(f"Ready autonomy continuation missed recovery token boundary: {execution_metadata}")
        assert_local_safe_recovery_execution_token(
            execution_metadata,
            "Ready autonomy continuation",
            expected_source="autonomy_continuation_execution",
        )
        if execution_metadata.get("local_safe_recovery_execution_token_sha256") != ready_metadata.get("local_safe_recovery_execution_token_sha256"):
            raise SystemExit(f"Ready autonomy continuation local-safe recovery token diverged from resume gate: {execution_metadata}")
        assert_carried_recovery_execution_readiness_token(
            execution_metadata,
            "Ready autonomy continuation",
            expected_source="checkpoint_recovery_followthrough",
        )
        carried_recovery_rows = execution_metadata.get("carried_recovery_execution_scorecard_rows") or []
        expected_carried_recovery_items = {
            "reviewed_step_permission",
            "verification_target",
            "receipt_hash_binding",
            "checkpoint_hash_binding",
            "stop_condition",
            "risky_recovery_approval_boundary",
            "followthrough_packet_ready",
            "fresh_followthrough_token",
        }
        if execution_metadata.get("carried_recovery_execution_scorecard_row_count") != 8 or len(carried_recovery_rows) != 8:
            raise SystemExit(f"Ready autonomy continuation missed carried recovery execution rows: {execution_metadata}")
        if {row.get("item") for row in carried_recovery_rows} != expected_carried_recovery_items:
            raise SystemExit(f"Ready autonomy continuation carried recovery rows diverged: {execution_metadata}")
        if execution_metadata.get("carried_recovery_execution_score") != 100 or execution_metadata.get("carried_recovery_execution_max_score") != 100:
            raise SystemExit(f"Ready autonomy continuation carried recovery score should be complete: {execution_metadata}")
        if execution_metadata.get("carried_recovery_execution_required_rows_ready") is not True:
            raise SystemExit(f"Ready autonomy continuation carried recovery rows should be ready: {execution_metadata}")
        if execution_metadata.get("carried_recovery_execution_scorecard_ready") is not True:
            raise SystemExit(f"Ready autonomy continuation carried recovery scorecard should be ready: {execution_metadata}")
        if execution_metadata.get("carried_recovery_execution_as_prior_proof") is not True:
            raise SystemExit(f"Ready autonomy continuation should carry recovery execution as prior proof: {execution_metadata}")
        if (
            execution_metadata.get("carried_recovery_execution_authorizes_action_now") is not False
            or execution_metadata.get("carried_recovery_execution_authorizes_risky_work") is not False
            or execution_metadata.get("carried_recovery_execution_authorizes_unreviewed_followthrough") is not False
        ):
            raise SystemExit(f"Ready autonomy continuation carried recovery proof should be non-authorizing: {execution_metadata}")
        if any(
            row.get("ready") is not True
            or row.get("points") != row.get("max_points")
            or row.get("required_before_normal_followthrough") is not True
            or row.get("authorizes_risky_work") is not False
            or row.get("authorizes_unreviewed_followthrough") is not False
            for row in carried_recovery_rows
        ):
            raise SystemExit(f"Ready autonomy continuation carried recovery rows should be ready and non-authorizing: {execution_metadata}")
        if execution_metadata.get("proposed_next_step_requires_approval") is not False or execution_metadata.get("proposed_next_step_risk_signals"):
            raise SystemExit(f"Ready autonomy continuation should remain local-safe: {execution_metadata}")
        if execution_metadata.get("proposed_next_step_approval_proof_queue_count") != 0 or execution_metadata.get("proposed_next_step_approval_proof_queue") != []:
            raise SystemExit(f"Ready autonomy continuation should not create an approval queue for local-safe work: {execution_metadata}")
        if execution_metadata.get("proposed_next_step_next_approval_proof_command") != "":
            raise SystemExit(f"Ready autonomy continuation should not expose an approval proof command: {execution_metadata}")
        if execution_metadata.get("proposed_next_step_approval_required_before_review") is not False:
            raise SystemExit(f"Ready autonomy continuation should not require approval before local-safe review: {execution_metadata}")
        if execution_metadata.get("proposed_next_step_approval_boundary_row_count") != 7:
            raise SystemExit(f"Ready autonomy continuation missed approval boundary row count: {execution_metadata}")
        if execution_metadata.get("proposed_next_step_approval_boundary_ready") is not True:
            raise SystemExit(f"Ready autonomy continuation approval boundary should be ready when no risk is present: {execution_metadata}")
        if (
            execution_metadata.get("proposed_next_step_approval_boundary_authorizes_action_now") is not False
            or execution_metadata.get("proposed_next_step_approval_boundary_authorizes_risky_work") is not False
            or execution_metadata.get("proposed_next_step_approval_boundary_authorizes_followup_without_closure") is not False
        ):
            raise SystemExit(f"Ready autonomy continuation approval boundary should stay non-authorizing: {execution_metadata}")
        approval_boundary_rows = execution_metadata.get("proposed_next_step_approval_boundary_rows") or []
        expected_approval_boundary_items = {
            "risk_classification",
            "exact_arguments_or_step",
            "approval_readiness_packet",
            "last_look_approval_packet",
            "approval_chain_proof",
            "post_approval_verification_receipt",
            "fresh_local_safe_review_after_approval",
        }
        if {row.get("item") for row in approval_boundary_rows} != expected_approval_boundary_items:
            raise SystemExit(f"Ready autonomy continuation approval boundary rows diverged: {execution_metadata}")
        if any(
            row.get("authorizes_action_now") is not False
            or row.get("authorizes_risky_work") is not False
            or row.get("authorizes_unreviewed_followthrough") is not False
            for row in approval_boundary_rows
        ):
            raise SystemExit(f"Ready autonomy continuation approval boundary rows should be non-authorizing: {execution_metadata}")
        assert_next_step_approval_boundary_token(
            execution_metadata,
            "Ready autonomy continuation",
            prefix="proposed_next_step",
        )
        if len(str(execution_metadata.get("proposed_next_step_sha256") or "")) != 64:
            raise SystemExit(f"Ready autonomy continuation missed proposed step hash: {execution_metadata}")
        if len(str(execution_metadata.get("proposed_verification_sha256") or "")) != 64:
            raise SystemExit(f"Ready autonomy continuation missed proposed verification hash: {execution_metadata}")
        if len(str(execution_metadata.get("continuation_review_token_sha256") or "")) != 64:
            raise SystemExit(f"Ready autonomy continuation missed review token hash: {execution_metadata}")
        if execution_metadata.get("continuation_review_token_reusable_for_next_review") is not False:
            raise SystemExit(f"Ready autonomy continuation should mark review token non-reusable: {execution_metadata}")
        if execution_metadata.get("previous_continuation_review_token_reusable_for_next_review") is not False:
            raise SystemExit(f"Ready autonomy continuation should carry prior review-token non-reuse mirror: {execution_metadata}")
        if execution_metadata.get("next_review_requires_new_continuation_review_token") is not True:
            raise SystemExit(f"Ready autonomy continuation missed next-review token boundary: {execution_metadata}")
        if not _continuation_review_token_metadata_ready(
            execution_metadata,
            reusable_key="continuation_review_token_reusable_for_next_review",
        ):
            raise SystemExit(f"Ready autonomy continuation review token failed mirror validator: {execution_metadata}")
        for missing_key in [
            "continuation_review_token_reusable_for_next_review",
            "previous_continuation_review_token_reusable_for_next_review",
        ]:
            missing_metadata = dict(execution_metadata)
            missing_metadata.pop(missing_key, None)
            if _continuation_review_token_metadata_ready(
                missing_metadata,
                reusable_key="continuation_review_token_reusable_for_next_review",
            ):
                raise SystemExit(
                    f"Ready autonomy continuation review token validator accepted missing mirror {missing_key}: "
                    f"{missing_metadata}"
                )
        if execution_metadata.get("prior_cycle_ledger_token_sha256") != "":
            raise SystemExit(f"Ready autonomy continuation should not invent a prior cycle token: {execution_metadata}")
        if execution_metadata.get("prior_cycle_ledger_token_present") is not False:
            raise SystemExit(f"Ready autonomy continuation should report no prior cycle token by default: {execution_metadata}")
        if execution_metadata.get("prior_cycle_ledger_token_reusable_for_this_review") is not False:
            raise SystemExit(f"Ready autonomy continuation should not reuse prior cycle token: {execution_metadata}")
        if execution_metadata.get("prior_cycle_ledger_proof_authorizes_action_now") is not False:
            raise SystemExit(f"Ready autonomy continuation should keep prior cycle proof non-authorizing: {execution_metadata}")
        assert_prior_cycle_ledger_token_boundary(
            execution_metadata,
            "Ready autonomy continuation",
            expected_source="autonomy_continuation_execution",
        )
        if execution_metadata.get("next_step_requires_fresh_cycle_ledger_token") is not True:
            raise SystemExit(f"Ready autonomy continuation missed fresh cycle-token requirement: {execution_metadata}")
        if execution_metadata.get("one_step_execution_contract_ready") is not True:
            raise SystemExit(f"Ready autonomy continuation missed one-step execution contract readiness: {execution_metadata}")
        if execution_metadata.get("one_step_execution_contract_row_count") != 6:
            raise SystemExit(f"Ready autonomy continuation missed one-step contract row count: {execution_metadata}")
        contract_rows = execution_metadata.get("one_step_execution_contract_rows") or []
        expected_contract_items = {
            "ready_resume_gate",
            "exact_next_step",
            "post_step_verification_target",
            "stop_condition",
            "risky_work_approval_boundary",
            "fresh_post_step_closure",
        }
        if {row.get("item") for row in contract_rows} != expected_contract_items:
            raise SystemExit(f"Ready autonomy continuation missed one-step contract items: {execution_metadata}")
        if any(
            row.get("authorizes_risky_work") is not False
            or row.get("reusable_for_next_step") is not False
            for row in contract_rows
        ):
            raise SystemExit(f"Ready autonomy continuation one-step contract should not authorize risky/reused work: {execution_metadata}")
        if execution_metadata.get("one_step_execution_contract_all_non_reusable") is not True:
            raise SystemExit(f"Ready autonomy continuation missed non-reusable contract summary: {execution_metadata}")
        if execution_metadata.get("one_step_execution_contract_all_risky_work_gated") is not True:
            raise SystemExit(f"Ready autonomy continuation missed risky-work gate summary: {execution_metadata}")
        if execution_metadata.get("one_step_execution_contract_all_local_safe_step_limited") is not True:
            raise SystemExit(f"Ready autonomy continuation missed one-local-safe-step-limited summary: {execution_metadata}")
        if execution_metadata.get("one_step_execution_contract_requires_fresh_closure") is not True:
            raise SystemExit(f"Ready autonomy continuation missed fresh closure requirement: {execution_metadata}")
        if execution_metadata.get("one_step_execution_contract_authorizes_batching") is not False:
            raise SystemExit(f"Ready autonomy continuation must not authorize batching: {execution_metadata}")
        if execution_metadata.get("one_step_execution_contract_authorizes_followup_without_closure") is not False:
            raise SystemExit(f"Ready autonomy continuation must not authorize follow-up without closure: {execution_metadata}")
        assert_one_step_execution_contract_token(execution_metadata, "Ready autonomy continuation")
        for expected_summary in [
            "one_step_only",
            "local_safe_only",
            "risky_work_approval_gated",
            "fresh_post_step_closure_required",
            "not_reusable_for_next_step",
        ]:
            if expected_summary not in execution_metadata.get("one_step_execution_contract_summary", []):
                raise SystemExit(f"Ready autonomy continuation missed one-step contract summary {expected_summary}: {execution_metadata}")
        post_step_proof_queue = execution_metadata.get("post_step_proof_queue") or []
        if not post_step_proof_queue:
            raise SystemExit(f"Ready autonomy continuation missed post-step proof queue: {execution_metadata}")
        if execution_metadata.get("post_step_proof_queue_count") != len(post_step_proof_queue):
            raise SystemExit(f"Ready autonomy continuation missed proof queue count: {execution_metadata}")
        if execution_metadata.get("post_step_next_proof_command") != post_step_proof_queue[0]:
            raise SystemExit(f"Ready autonomy continuation missed next post-step proof command: {execution_metadata}")
        expected_post_step_proof_queue = _autonomy_expected_post_step_proof_queue(
            proposed_next_step=str(execution_metadata.get("proposed_next_step") or ""),
            proposed_verification=str(execution_metadata.get("proposed_verification") or ""),
        )
        if post_step_proof_queue != expected_post_step_proof_queue:
            raise SystemExit(f"Ready autonomy continuation post-step proof queue diverged from exact closure ladder: {execution_metadata}")
        stale_post_step_metadata = json.loads(json.dumps(execution_metadata))
        stale_post_step_queue = list(expected_post_step_proof_queue)
        stale_post_step_queue[-1] = "completion claim gate: stale"
        stale_post_step_metadata["post_step_proof_queue"] = stale_post_step_queue
        stale_post_step_metadata["post_step_proof_queue_count"] = len(stale_post_step_queue)
        stale_post_step_metadata["post_step_next_proof_command"] = stale_post_step_queue[0]
        stale_post_step_metadata["next_safe_command"] = stale_post_step_queue[0]
        if _autonomy_post_step_proof_queue_ready(stale_post_step_metadata):
            raise SystemExit(f"Ready autonomy continuation post-step proof queue validator accepted stale queue: {stale_post_step_metadata}")
        if _autonomy_continuation_execution_ready_from_metadata(stale_post_step_metadata):
            raise SystemExit(f"Ready autonomy continuation all-up validator accepted stale post-step proof queue: {stale_post_step_metadata}")
        for key in READ_ONLY_FALSE_FLAGS:
            if execution_metadata.get(key) is not False:
                raise SystemExit(f"Ready autonomy continuation execution packet should report {key}=False.")

        risky_execution = ready_runtime.registry.get("autonomy_continuation_execution_packet").handler(
            {
                "objective": "continue Jarvis safely",
                "stop_at": "2099-01-01T00:00:00+09:00",
                "current_time": "2026-06-09T03:00:00+09:00",
                "timezone": "Asia/Seoul",
                "reviewed_step": "reviewed local-safe continuity metadata",
                "verification": "smoke_test_next_step passed",
                "receipt_path": str(recovery_receipt_path),
                "receipt_sha256": recovery_receipt_sha256,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": recovery_checkpoint_sha256,
                "stop_condition": "stop if verification drifts",
                "blockers": "none",
                "next_step": "run python script to inspect local execution status",
                "next_verification": "verify shell command output",
            }
        )
        print("[ok] direct risky autonomy continuation execution packet held for approval proof")
        print(risky_execution.output[:1800])
        print()
        if not risky_execution.ok:
            raise SystemExit("Risky autonomy continuation execution packet should return a held packet.")
        risky_metadata = risky_execution.metadata
        if risky_metadata.get("continuation_state") != "AUTONOMY_CONTINUATION_HELD":
            raise SystemExit(f"Risky autonomy continuation should be held: {risky_metadata}")
        if risky_metadata.get("one_local_safe_step_allowed") is not False or risky_metadata.get("one_step_execution_contract_ready") is not False:
            raise SystemExit(f"Risky autonomy continuation must not permit one-step execution: {risky_metadata}")
        if "risky next step needs approval packet" not in (risky_metadata.get("missing_blockers") or []):
            raise SystemExit(f"Risky autonomy continuation missed approval blocker: {risky_metadata}")
        if risky_metadata.get("proposed_next_step_requires_approval") is not True:
            raise SystemExit(f"Risky autonomy continuation should require approval: {risky_metadata}")
        if "shell/code" not in (risky_metadata.get("proposed_next_step_risk_signals") or []):
            raise SystemExit(f"Risky autonomy continuation missed shell/code signal: {risky_metadata}")
        expected_risky_queue = [
            "approval readiness <id>",
            "approval packet <id>",
            "approval chain proof <id>",
            "verification receipt <approved run id from approval chain proof <id>>",
            "autonomy continuation execution: <reviewed local-safe step after approval proof>",
        ]
        if risky_metadata.get("proposed_next_step_approval_proof_queue") != expected_risky_queue:
            raise SystemExit(f"Risky autonomy continuation approval proof queue diverged: {risky_metadata}")
        if risky_metadata.get("proposed_next_step_approval_proof_queue_count") != len(expected_risky_queue):
            raise SystemExit(f"Risky autonomy continuation missed approval proof queue count: {risky_metadata}")
        if risky_metadata.get("proposed_next_step_next_approval_proof_command") != "approval readiness <id>":
            raise SystemExit(f"Risky autonomy continuation missed next approval proof command: {risky_metadata}")
        if risky_metadata.get("next_safe_command") != "approval readiness <id>":
            raise SystemExit(f"Risky autonomy continuation should point at approval readiness next: {risky_metadata}")
        if risky_metadata.get("proposed_next_step_approval_required_before_review") is not True:
            raise SystemExit(f"Risky autonomy continuation missed approval-before-review flag: {risky_metadata}")
        if risky_metadata.get("proposed_next_step_approval_boundary_row_count") != 7:
            raise SystemExit(f"Risky autonomy continuation missed approval boundary row count: {risky_metadata}")
        risky_boundary_rows = risky_metadata.get("proposed_next_step_approval_boundary_rows") or []
        if {row.get("item") for row in risky_boundary_rows} != expected_approval_boundary_items:
            raise SystemExit(f"Risky autonomy continuation approval boundary rows diverged: {risky_metadata}")
        if risky_metadata.get("proposed_next_step_approval_boundary_ready") is not False:
            raise SystemExit(f"Risky autonomy continuation approval boundary should stay held: {risky_metadata}")
        if (
            risky_metadata.get("proposed_next_step_approval_boundary_authorizes_action_now") is not False
            or risky_metadata.get("proposed_next_step_approval_boundary_authorizes_risky_work") is not False
            or risky_metadata.get("proposed_next_step_approval_boundary_authorizes_followup_without_closure") is not False
        ):
            raise SystemExit(f"Risky autonomy continuation approval boundary should stay non-authorizing: {risky_metadata}")
        if any(
            row.get("required_before_risky_next_step") is not True
            or row.get("status") != "held"
            or row.get("authorizes_action_now") is not False
            or row.get("authorizes_risky_work") is not False
            or row.get("authorizes_unreviewed_followthrough") is not False
            for row in risky_boundary_rows
        ):
            raise SystemExit(f"Risky autonomy continuation boundary rows should be held and non-authorizing: {risky_metadata}")
        assert_next_step_approval_boundary_token(
            risky_metadata,
            "Risky autonomy continuation",
            prefix="proposed_next_step",
        )
        if "Risky next-step approval proof queue" not in risky_execution.output:
            raise SystemExit("Risky autonomy continuation output missed approval proof queue section.")
        for key in READ_ONLY_FALSE_FLAGS:
            if risky_metadata.get(key) is not False:
                raise SystemExit(f"Risky autonomy continuation execution packet should report {key}=False.")

        risky_closure = ready_runtime.registry.get("autonomy_step_closure_packet").handler(
            {
                "objective": "continue Jarvis safely",
                "stop_at": "2099-01-01T00:00:00+09:00",
                "current_time": "2026-06-09T03:00:00+09:00",
                "timezone": "Asia/Seoul",
                "reviewed_step": "reviewed local-safe continuity metadata",
                "recovery_verification": "smoke_test_next_step passed",
                "recovery_receipt_path": str(recovery_receipt_path),
                "recovery_receipt_sha256": recovery_receipt_sha256,
                "recovery_checkpoint_path": checkpoint_path,
                "recovery_checkpoint_sha256": recovery_checkpoint_sha256,
                "stop_condition": "stop if verification drifts",
                "blockers": "none",
                "next_step": "run python script to inspect local execution status",
                "next_verification": "verify shell command output",
                "completed_step": "run python script to inspect local execution status",
                "post_step_verification": "verify shell command output",
                "post_step_receipt_path": str(post_step_receipt_path),
                "post_step_receipt_sha256": post_step_receipt_sha256,
                "post_step_checkpoint_path": str(post_step_checkpoint_path),
                "post_step_checkpoint_sha256": post_step_checkpoint_sha256,
                "execution_health": "execution health reviewed",
                "execution_audit": "execution audit reviewed",
                "after_action_learning": "learning reviewed",
            }
        )
        print("[ok] risky autonomy step closure carries approval proof hold")
        print(risky_closure.output[:1800])
        print()
        if not risky_closure.ok:
            raise SystemExit("Risky autonomy step closure packet should return a held packet.")
        risky_closure_metadata = risky_closure.metadata
        if risky_closure_metadata.get("closure_state") != "AUTONOMY_STEP_CLOSURE_HELD":
            raise SystemExit(f"Risky autonomy step closure should be held: {risky_closure_metadata}")
        if risky_closure_metadata.get("pre_step_one_local_safe_step_allowed") is not False:
            raise SystemExit(f"Risky autonomy step closure must not inherit one-step permission: {risky_closure_metadata}")
        if risky_closure_metadata.get("carried_next_step_approval_proof_queue") != expected_risky_queue:
            raise SystemExit(f"Risky autonomy step closure missed carried approval queue: {risky_closure_metadata}")
        if risky_closure_metadata.get("carried_next_step_approval_proof_queue_count") != len(expected_risky_queue):
            raise SystemExit(f"Risky autonomy step closure missed carried approval queue count: {risky_closure_metadata}")
        if risky_closure_metadata.get("carried_next_step_next_approval_proof_command") != "approval readiness <id>":
            raise SystemExit(f"Risky autonomy step closure missed carried next approval command: {risky_closure_metadata}")
        if risky_closure_metadata.get("carried_next_step_approval_required_before_review") is not True:
            raise SystemExit(f"Risky autonomy step closure missed carried approval-required flag: {risky_closure_metadata}")
        if risky_closure_metadata.get("carried_next_step_approval_boundary_row_count") != 7:
            raise SystemExit(f"Risky autonomy step closure missed carried approval boundary rows: {risky_closure_metadata}")
        if risky_closure_metadata.get("carried_next_step_approval_boundary_ready") is not False:
            raise SystemExit(f"Risky autonomy step closure should carry held approval boundary: {risky_closure_metadata}")
        if risky_closure_metadata.get("carried_next_step_approval_boundary_as_prior_proof") is not True:
            raise SystemExit(f"Risky autonomy step closure should carry non-authorizing approval boundary as prior proof: {risky_closure_metadata}")
        if (
            risky_closure_metadata.get("carried_next_step_approval_boundary_authorizes_action_now") is not False
            or risky_closure_metadata.get("carried_next_step_approval_boundary_authorizes_risky_work") is not False
            or risky_closure_metadata.get("carried_next_step_approval_boundary_authorizes_unreviewed_followthrough") is not False
        ):
            raise SystemExit(f"Risky autonomy step closure carried approval proof should be non-authorizing: {risky_closure_metadata}")
        carried_risky_boundary_rows = risky_closure_metadata.get("carried_next_step_approval_boundary_rows") or []
        if {row.get("item") for row in carried_risky_boundary_rows} != expected_approval_boundary_items:
            raise SystemExit(f"Risky autonomy step closure carried approval boundary rows diverged: {risky_closure_metadata}")
        if any(
            row.get("required_before_risky_next_step") is not True
            or row.get("status") != "held"
            or row.get("authorizes_action_now") is not False
            or row.get("authorizes_risky_work") is not False
            or row.get("authorizes_unreviewed_followthrough") is not False
            for row in carried_risky_boundary_rows
        ):
            raise SystemExit(f"Risky autonomy step closure carried rows should remain held and non-authorizing: {risky_closure_metadata}")
        assert_next_step_approval_boundary_token(
            risky_closure_metadata,
            "Risky autonomy step closure",
            prefix="carried_next_step",
        )
        if "Carried risky next-step approval proof" not in risky_closure.output:
            raise SystemExit("Risky autonomy step closure output missed carried approval proof text.")
        for key in READ_ONLY_FALSE_FLAGS:
            if risky_closure_metadata.get(key) is not False:
                raise SystemExit(f"Risky autonomy step closure packet should report {key}=False.")

        risky_ledger = ready_runtime.registry.get("autonomy_cycle_ledger").handler(
            {
                "objective": "continue Jarvis safely",
                "stop_at": "2099-01-01T00:00:00+09:00",
                "current_time": "2026-06-09T03:00:00+09:00",
                "timezone": "Asia/Seoul",
                "reviewed_step": "reviewed local-safe continuity metadata",
                "recovery_verification": "smoke_test_next_step passed",
                "recovery_receipt_path": str(recovery_receipt_path),
                "recovery_receipt_sha256": recovery_receipt_sha256,
                "recovery_checkpoint_path": checkpoint_path,
                "recovery_checkpoint_sha256": recovery_checkpoint_sha256,
                "stop_condition": "stop if verification drifts",
                "blockers": "none",
                "next_step": "run python script to inspect local execution status",
                "next_verification": "verify shell command output",
                "completed_step": "run python script to inspect local execution status",
                "post_step_verification": "verify shell command output",
                "post_step_receipt_path": str(post_step_receipt_path),
                "post_step_receipt_sha256": post_step_receipt_sha256,
                "post_step_checkpoint_path": str(post_step_checkpoint_path),
                "post_step_checkpoint_sha256": post_step_checkpoint_sha256,
                "execution_health": "execution health reviewed",
                "execution_audit": "execution audit reviewed",
                "after_action_learning": "learning reviewed",
            }
        )
        print("[ok] risky autonomy cycle ledger carries approval proof hold")
        print(risky_ledger.output[:1800])
        print()
        if not risky_ledger.ok:
            raise SystemExit("Risky autonomy cycle ledger should return a held ledger.")
        risky_ledger_metadata = risky_ledger.metadata
        if risky_ledger_metadata.get("cycle_state") != "AUTONOMY_CYCLE_LEDGER_HELD":
            raise SystemExit(f"Risky autonomy cycle ledger should be held: {risky_ledger_metadata}")
        if risky_ledger_metadata.get("ready_for_fresh_next_continuation_review") is not False:
            raise SystemExit(f"Risky autonomy cycle ledger must not allow fresh review: {risky_ledger_metadata}")
        if risky_ledger_metadata.get("carried_next_step_approval_proof_queue") != expected_risky_queue:
            raise SystemExit(f"Risky autonomy cycle ledger missed carried approval queue: {risky_ledger_metadata}")
        if risky_ledger_metadata.get("carried_next_step_approval_proof_queue_count") != len(expected_risky_queue):
            raise SystemExit(f"Risky autonomy cycle ledger missed carried approval queue count: {risky_ledger_metadata}")
        if risky_ledger_metadata.get("carried_next_step_next_approval_proof_command") != "approval readiness <id>":
            raise SystemExit(f"Risky autonomy cycle ledger missed carried next approval command: {risky_ledger_metadata}")
        if risky_ledger_metadata.get("carried_next_step_approval_required_before_review") is not True:
            raise SystemExit(f"Risky autonomy cycle ledger missed carried approval-required flag: {risky_ledger_metadata}")
        if risky_ledger_metadata.get("carried_next_step_approval_boundary_row_count") != 7:
            raise SystemExit(f"Risky autonomy cycle ledger missed carried approval boundary rows: {risky_ledger_metadata}")
        if risky_ledger_metadata.get("carried_next_step_approval_boundary_ready") is not False:
            raise SystemExit(f"Risky autonomy cycle ledger should carry held approval boundary: {risky_ledger_metadata}")
        if risky_ledger_metadata.get("carried_next_step_approval_boundary_as_prior_proof") is not True:
            raise SystemExit(f"Risky autonomy cycle ledger should carry non-authorizing approval boundary as prior proof: {risky_ledger_metadata}")
        if (
            risky_ledger_metadata.get("carried_next_step_approval_boundary_authorizes_action_now") is not False
            or risky_ledger_metadata.get("carried_next_step_approval_boundary_authorizes_risky_work") is not False
            or risky_ledger_metadata.get("carried_next_step_approval_boundary_authorizes_unreviewed_followthrough") is not False
        ):
            raise SystemExit(f"Risky autonomy cycle ledger carried approval proof should be non-authorizing: {risky_ledger_metadata}")
        carried_ledger_boundary_rows = risky_ledger_metadata.get("carried_next_step_approval_boundary_rows") or []
        if {row.get("item") for row in carried_ledger_boundary_rows} != expected_approval_boundary_items:
            raise SystemExit(f"Risky autonomy cycle ledger carried approval boundary rows diverged: {risky_ledger_metadata}")
        if any(
            row.get("required_before_risky_next_step") is not True
            or row.get("status") != "held"
            or row.get("authorizes_action_now") is not False
            or row.get("authorizes_risky_work") is not False
            or row.get("authorizes_unreviewed_followthrough") is not False
            for row in carried_ledger_boundary_rows
        ):
            raise SystemExit(f"Risky autonomy cycle ledger carried rows should remain held and non-authorizing: {risky_ledger_metadata}")
        assert_next_step_approval_boundary_token(
            risky_ledger_metadata,
            "Risky autonomy cycle ledger",
            prefix="carried_next_step",
        )
        if "Carried risky next-step approval proof" not in risky_ledger.output:
            raise SystemExit("Risky autonomy cycle ledger output missed carried approval proof text.")
        for key in READ_ONLY_FALSE_FLAGS:
            if risky_ledger_metadata.get(key) is not False:
                raise SystemExit(f"Risky autonomy cycle ledger should report {key}=False.")

        ready_closure = ready_runtime.registry.get("autonomy_step_closure_packet").handler(
            {
                "objective": "continue Jarvis safely",
                "stop_at": "2099-01-01T00:00:00+09:00",
                "current_time": "2026-06-09T03:00:00+09:00",
                "timezone": "Asia/Seoul",
                "reviewed_step": "reviewed local-safe continuity metadata",
                "recovery_verification": "smoke_test_next_step passed",
                "recovery_receipt_path": str(recovery_receipt_path),
                "recovery_receipt_sha256": recovery_receipt_sha256,
                "recovery_checkpoint_path": checkpoint_path,
                "recovery_checkpoint_sha256": recovery_checkpoint_sha256,
                "stop_condition": "stop if verification drifts",
                "blockers": "none",
                "next_step": "update local continuity smoke metadata",
                "next_verification": "smoke_test_next_step passed",
                "completed_step": "update local continuity smoke metadata",
                "post_step_verification": "smoke_test_next_step passed",
                "post_step_receipt_path": str(post_step_receipt_path),
                "post_step_receipt_sha256": post_step_receipt_sha256,
                "post_step_checkpoint_path": str(post_step_checkpoint_path),
                "post_step_checkpoint_sha256": post_step_checkpoint_sha256,
                "execution_health": "execution health reviewed",
                "execution_audit": "execution audit reviewed",
                "after_action_learning": "learning reviewed",
            }
        )
        print("[ok] direct ready autonomy step closure packet")
        print(ready_closure.output[:1800])
        print()
        if not ready_closure.ok:
            raise SystemExit("Ready autonomy step closure packet should run.")
        closure_metadata = ready_closure.metadata
        if closure_metadata.get("closure_state") != "AUTONOMY_STEP_CLOSURE_READY_FOR_NEXT_CONTINUATION_REVIEW":
            raise SystemExit(f"Ready autonomy step closure should permit next review: {closure_metadata}")
        assert_timebox_review_contract(closure_metadata, "Ready autonomy step closure")
        assert_operator_supersession_token_boundary(
            closure_metadata,
            "Ready autonomy step closure",
            expected_source="operator_instruction_supersession",
        )
        assert_checkpoint_route_boundary(
            closure_metadata,
            "Ready autonomy step closure",
            expected_source="checkpoint_recovery_cockpit",
        )
        if closure_metadata.get("ready_for_next_continuation_review") is not True or closure_metadata.get("missing_blockers"):
            raise SystemExit(f"Ready autonomy step closure missed complete proof: {closure_metadata}")
        if closure_metadata.get("missing_blocker_count") != 0:
            raise SystemExit(f"Ready autonomy step closure missed explicit zero missing-blocker count: {closure_metadata}")
        expected_step_closure_next_command = "autonomy continuation execution: <next reviewed local-safe step>"
        if closure_metadata.get("next_safe_command") != expected_step_closure_next_command:
            raise SystemExit(f"Ready autonomy step closure missed exact next-safe review command: {closure_metadata}")
        closure_required_commands = closure_metadata.get("required_commands") or []
        if not closure_required_commands:
            raise SystemExit(f"Ready autonomy step closure missed required commands: {closure_metadata}")
        if closure_metadata.get("required_command_count") != len(closure_required_commands):
            raise SystemExit(f"Ready autonomy step closure required command count diverged: {closure_metadata}")
        if closure_metadata.get("proof_queue") != closure_required_commands:
            raise SystemExit(f"Ready autonomy step closure proof queue should mirror required commands: {closure_metadata}")
        if closure_metadata.get("proof_queue_count") != len(closure_required_commands):
            raise SystemExit(f"Ready autonomy step closure proof queue count diverged: {closure_metadata}")
        if expected_step_closure_next_command not in closure_required_commands:
            raise SystemExit(f"Ready autonomy step closure required commands missed next review command: {closure_metadata}")
        if closure_metadata.get("next_required_command") != expected_step_closure_next_command:
            raise SystemExit(f"Ready autonomy step closure missed next required command: {closure_metadata}")
        if closure_metadata.get("next_proof_command") != expected_step_closure_next_command:
            raise SystemExit(f"Ready autonomy step closure missed next proof command: {closure_metadata}")
        if closure_metadata.get("step_closure_ready_contract_ready") is not True:
            raise SystemExit(f"Ready autonomy step closure missed all-up ready contract flag: {closure_metadata}")
        if not _autonomy_step_closure_ready(closure_metadata):
            raise SystemExit(f"Ready autonomy step closure failed production ready validator: {closure_metadata}")
        malformed_permission_gate = _autonomy_step_closure_gate(
            continuation_ready="false",
            continued_step=str(closure_metadata.get("completed_step") or "completed local-safe step"),
            completed_step_matches_proposed=True,
            verification_matches_proposed=True,
            verification_evidence=str(closure_metadata.get("post_step_verification") or "post-step verification"),
            post_step_receipt_path=str(closure_metadata.get("post_step_receipt_path") or "post-step receipt"),
            post_step_receipt_sha256=str(closure_metadata.get("post_step_receipt_sha256") or ""),
            post_step_receipt_file_sha256=str(closure_metadata.get("post_step_receipt_file_sha256") or ""),
            post_step_receipt_hash_matches_file=True,
            post_step_checkpoint_path=str(closure_metadata.get("post_step_checkpoint_path") or "post-step checkpoint"),
            post_step_checkpoint_sha256=str(closure_metadata.get("post_step_checkpoint_sha256") or ""),
            post_step_checkpoint_file_sha256=str(closure_metadata.get("post_step_checkpoint_file_sha256") or ""),
            post_step_checkpoint_hash_matches_file=True,
            execution_health=str(closure_metadata.get("execution_health") or "execution health reviewed"),
            execution_audit=str(closure_metadata.get("execution_audit") or "execution audit reviewed"),
            after_action_learning=str(closure_metadata.get("after_action_learning") or "learning reviewed"),
            blockers="none",
        )
        if malformed_permission_gate.get("ready_for_next_continuation_review") is not False:
            raise SystemExit(f"Step closure gate accepted malformed continuation permission: {malformed_permission_gate}")
        if "ready autonomy continuation execution packet" not in (malformed_permission_gate.get("missing") or []):
            raise SystemExit(f"Step closure gate missed malformed-permission blocker: {malformed_permission_gate}")
        for missing_blocker_key in ["missing_blockers", "missing_blocker_count"]:
            missing_blocker_metadata = dict(closure_metadata)
            missing_blocker_metadata.pop(missing_blocker_key, None)
            if _autonomy_step_closure_ready(missing_blocker_metadata):
                raise SystemExit(
                    f"Autonomy step closure ready validator accepted missing blocker contract field {missing_blocker_key}: "
                    f"{missing_blocker_metadata}"
                )
        for key, tampered_value in [
            ("next_safe_command", "autonomy step closure: stale"),
            ("required_command_count", 999),
            ("proof_queue", ["stale proof command"]),
            ("proof_queue_count", 999),
            ("next_required_command", "stale required command"),
            ("next_proof_command", "stale proof command"),
            ("carried_next_step_approval_proof_queue_count", 999),
            ("carried_next_step_approval_boundary_row_count", 999),
            ("carried_next_step_approval_boundary_token_present", False),
            ("carried_next_step_approval_boundary_ready", False),
            ("carried_next_step_approval_boundary_as_prior_proof", False),
            ("carried_next_step_approval_required_before_review", True),
            ("carried_next_step_next_approval_proof_command", "stale approval proof command"),
            ("carried_next_step_approval_boundary_authorizes_approval", True),
            ("carried_next_step_approval_boundary_authorizes_model_call", True),
            ("carried_next_step_approval_boundary_authorizes_tool_execution", True),
            ("carried_next_step_approval_boundary_authorizes_personal_data_read", True),
            ("carried_next_step_approval_boundary_authorizes_external_side_effect", True),
            ("carried_next_step_approval_boundary_authorizes_timebox_reuse", True),
            ("carried_next_step_approval_boundary_reusable_for_recovery_review", True),
        ]:
            tampered = json.loads(json.dumps(closure_metadata))
            tampered[key] = tampered_value
            if _autonomy_step_closure_ready(tampered):
                raise SystemExit(f"Autonomy step closure ready validator accepted stale command alias {key}: {tampered}")
        for key, tampered_value in [
            ("closure_state", "AUTONOMY_STEP_CLOSURE_HELD"),
            ("ready_for_next_continuation_review", False),
            ("timebox_receipt_present", False),
            ("timebox_review_contract_row_count", 999),
            ("timebox_review_contract_ready", False),
            ("next_step_requires_fresh_timebox", False),
            ("timebox_authorizes_timebox_reuse", True),
            ("checkpoint_route_token_present", False),
            ("checkpoint_route_boundary_row_count", 999),
            ("checkpoint_route_boundary_ready", False),
            ("next_checkpoint_route_requires_fresh_recovery_review", False),
            ("checkpoint_route_authorizes_checkpoint_reuse", True),
            ("one_step_execution_contract_token_present", False),
            ("one_step_execution_contract_token_as_prior_proof", False),
            ("one_step_execution_contract_row_count", 999),
            ("one_step_execution_contract_ready", False),
            ("next_step_requires_new_one_step_execution_contract_token", False),
            ("one_step_execution_contract_token_authorizes_action_now", True),
            ("one_step_execution_contract_token_reusable_for_next_step", True),
            ("continuation_review_token_present", False),
            ("continuation_review_token_reusable_for_next_review", True),
            ("previous_continuation_review_token_reusable_for_next_review", "false"),
            ("next_review_requires_new_continuation_review_token", False),
            ("post_step_receipt_hash_matches_file", False),
            ("post_step_checkpoint_hash_matches_file", False),
            ("fresh_continuation_review_boundary_token_present", False),
            ("fresh_continuation_review_boundary_token_row_count", 999),
            ("fresh_continuation_review_boundary_token_ready", False),
            ("next_review_requires_new_fresh_continuation_review_boundary_token", False),
            ("fresh_continuation_review_boundary_reusable_for_next_review", True),
            ("fresh_continuation_review_boundary_reusable_for_next_cycle", True),
            ("step_closure_receipt_token_present", False),
            ("step_closure_receipt_boundary_row_count", 999),
            ("step_closure_receipt_boundary_ready", False),
            ("next_review_requires_new_step_closure_receipt_token", False),
            ("step_closure_receipt_authorizes_action_now", True),
            ("step_closure_receipt_reusable_for_next_review", True),
            ("prior_cycle_ledger_token_present", True),
            ("prior_cycle_ledger_token_boundary_row_count", 999),
            ("prior_cycle_ledger_token_boundary_ready", False),
            ("prior_cycle_ledger_token_reusable_for_this_closure", True),
            ("prior_cycle_ledger_proof_authorizes_post_step_closure", True),
            ("action_allowed_now", True),
            ("authorizes_execution", True),
        ]:
            tampered = dict(closure_metadata)
            tampered[key] = tampered_value
            if _autonomy_step_closure_ready(tampered):
                raise SystemExit(f"Autonomy step closure ready validator accepted tampered {key}: {tampered}")
        for missing_false_key in [
            "action_allowed_now",
            "fresh_continuation_review_boundary_authorizes_action_now",
            "step_closure_receipt_authorizes_new_cycle",
            "prior_cycle_ledger_proof_authorizes_model_call",
            "calls_model",
            "authorizes_execution",
            "approval_granted",
        ]:
            missing_false_metadata = dict(closure_metadata)
            missing_false_metadata.pop(missing_false_key, None)
            if _autonomy_step_closure_ready(missing_false_metadata):
                raise SystemExit(
                    f"Autonomy step closure ready validator accepted missing false-authority field {missing_false_key}"
                )
        tampered = dict(closure_metadata)
        tampered_rows = [dict(row) for row in (closure_metadata.get("step_closure_readiness_scorecard_rows") or [])]
        tampered_rows[0]["authorizes_risky_work"] = True
        tampered["step_closure_readiness_scorecard_rows"] = tampered_rows
        if _autonomy_step_closure_ready(tampered):
            raise SystemExit(f"Autonomy step closure ready validator accepted authority-bearing scorecard rows: {tampered}")
        closure_scorecard = closure_metadata.get("step_closure_readiness_scorecard_rows") or []
        if "Measured step closure readiness scorecard" not in ready_closure.output:
            raise SystemExit("Ready autonomy step closure output missed measured scorecard text.")
        if closure_metadata.get("step_closure_readiness_scorecard_row_count") != len(closure_scorecard):
            raise SystemExit(f"Ready autonomy step closure missed scorecard row count: {closure_metadata}")
        if closure_metadata.get("step_closure_readiness_scorecard_row_count") != 9:
            raise SystemExit(f"Ready autonomy step closure scorecard should have nine rows: {closure_metadata}")
        if closure_metadata.get("step_closure_readiness_score") != 100 or closure_metadata.get("step_closure_readiness_max_score") != 100:
            raise SystemExit(f"Ready autonomy step closure should score 100/100: {closure_metadata}")
        if closure_metadata.get("step_closure_readiness_required_rows_ready") is not True:
            raise SystemExit(f"Ready autonomy step closure should mark scorecard rows ready: {closure_metadata}")
        if closure_metadata.get("step_closure_readiness_scorecard_ready") is not True:
            raise SystemExit(f"Ready autonomy step closure should mark scorecard shape ready: {closure_metadata}")
        assert_autonomy_step_closure_scorecard(
            closure_metadata,
            "Ready autonomy step closure",
            ready=True,
        )
        expected_closure_scorecard_items = {
            "pre_step_permission_ready",
            "completed_step_matches_allowed",
            "post_step_verification_matches_target",
            "post_step_receipt_hash_binding",
            "post_step_checkpoint_hash_binding",
            "execution_health_supplied",
            "execution_audit_supplied",
            "after_action_learning_supplied",
            "fresh_review_contract_non_reusable",
        }
        if {row.get("item") for row in closure_scorecard} != expected_closure_scorecard_items:
            raise SystemExit(f"Ready autonomy step closure scorecard items diverged: {closure_metadata}")
        if not all(
            row.get("required_before_next_continuation_review") is True
            and row.get("authorizes_action") is False
            and row.get("authorizes_risky_work") is False
            and row.get("authorizes_followup_without_fresh_review") is False
            for row in closure_scorecard
        ):
            raise SystemExit(f"Ready autonomy step closure scorecard should stay non-authorizing: {closure_metadata}")
        if closure_metadata.get("checkpoint_path_matches_latest") is not True:
            raise SystemExit(f"Ready autonomy step closure missed checkpoint binding proof: {closure_metadata}")
        if closure_metadata.get("checkpoint_hash_matches_latest") is not True:
            raise SystemExit(f"Ready autonomy step closure missed checkpoint hash binding proof: {closure_metadata}")
        if closure_metadata.get("timebox_state") != "STOP_WINDOW_ACTIVE":
            raise SystemExit(f"Ready autonomy step closure missed timebox state propagation: {closure_metadata}")
        if closure_metadata.get("cockpit_state") != "RECOVERY_COCKPIT_READY_FOR_LOCAL_SAFE_REVIEW":
            raise SystemExit(f"Ready autonomy step closure missed cockpit state propagation: {closure_metadata}")
        if closure_metadata.get("resume_gate_state") != "AUTONOMY_RESUME_READY_FOR_LOCAL_SAFE_CONTINUATION":
            raise SystemExit(f"Ready autonomy step closure missed resume gate state propagation: {closure_metadata}")
        if closure_metadata.get("followthrough_state") != "RECOVERY_FOLLOWTHROUGH_READY":
            raise SystemExit(f"Ready autonomy step closure missed follow-through state propagation: {closure_metadata}")
        if closure_metadata.get("recovery_artifact_hashes_present") is not True:
            raise SystemExit(f"Ready autonomy step closure missed recovery artifact hash binding: {closure_metadata}")
        if closure_metadata.get("receipt_hash_matches_file") is not True:
            raise SystemExit(f"Ready autonomy step closure missed recovery receipt file binding: {closure_metadata}")
        if closure_metadata.get("recovery_checkpoint_hash_matches_file") is not True:
            raise SystemExit(f"Ready autonomy step closure missed recovery checkpoint file binding: {closure_metadata}")
        if closure_metadata.get("receipt_sha256") != recovery_receipt_sha256 or closure_metadata.get("checkpoint_sha256") != recovery_checkpoint_sha256:
            raise SystemExit(f"Ready autonomy step closure hash values diverged: {closure_metadata}")
        assert_sha256(closure_metadata.get("recovery_followthrough_token_sha256"), "Ready autonomy step closure recovery token")
        if closure_metadata.get("recovery_followthrough_token_sha256") != execution_metadata.get("recovery_followthrough_token_sha256"):
            raise SystemExit(f"Ready autonomy step closure recovery token diverged from continuation: {closure_metadata}")
        if closure_metadata.get("recovery_followthrough_token_reusable_for_future_recovery") is not False:
            raise SystemExit(f"Ready autonomy step closure should mark recovery token non-reusable: {closure_metadata}")
        assert_recovery_followthrough_token_boundary(
            closure_metadata,
            "Ready autonomy step closure",
            expected_source="checkpoint_recovery_followthrough",
        )
        if closure_metadata.get("next_recovery_followthrough_requires_new_token") is not True:
            raise SystemExit(f"Ready autonomy step closure missed recovery token boundary: {closure_metadata}")
        assert_local_safe_recovery_execution_token(
            closure_metadata,
            "Ready autonomy step closure",
            expected_source="autonomy_step_closure",
        )
        if closure_metadata.get("local_safe_recovery_execution_token_sha256") != execution_metadata.get("local_safe_recovery_execution_token_sha256"):
            raise SystemExit(f"Ready autonomy step closure local-safe recovery token diverged from continuation: {closure_metadata}")
        assert_carried_recovery_execution_readiness_token(
            closure_metadata,
            "Ready autonomy step closure",
            expected_source="checkpoint_recovery_followthrough",
        )
        assert_one_step_execution_contract_token(closure_metadata, "Ready autonomy step closure", carried=True)
        if closure_metadata.get("one_step_execution_contract_token_sha256") != execution_metadata.get("one_step_execution_contract_token_sha256"):
            raise SystemExit(f"Ready autonomy step closure one-step contract token diverged from continuation: {closure_metadata}")
        carried_recovery_rows = closure_metadata.get("carried_recovery_execution_scorecard_rows") or []
        expected_recovery_execution_items = {
            "reviewed_step_permission",
            "verification_target",
            "receipt_hash_binding",
            "checkpoint_hash_binding",
            "stop_condition",
            "risky_recovery_approval_boundary",
            "followthrough_packet_ready",
            "fresh_followthrough_token",
        }
        if "Carried recovery execution readiness proof" not in ready_closure.output:
            raise SystemExit("Ready autonomy step closure output missed carried recovery execution proof text.")
        if closure_metadata.get("carried_recovery_execution_scorecard_row_count") != 8 or len(carried_recovery_rows) != 8:
            raise SystemExit(f"Ready autonomy step closure missed carried recovery execution rows: {closure_metadata}")
        if {row.get("item") for row in carried_recovery_rows} != expected_recovery_execution_items:
            raise SystemExit(f"Ready autonomy step closure carried recovery rows diverged: {closure_metadata}")
        if closure_metadata.get("carried_recovery_execution_score") != 100 or closure_metadata.get("carried_recovery_execution_max_score") != 100:
            raise SystemExit(f"Ready autonomy step closure carried recovery score should be complete: {closure_metadata}")
        if closure_metadata.get("carried_recovery_execution_required_rows_ready") is not True:
            raise SystemExit(f"Ready autonomy step closure carried recovery rows should be ready: {closure_metadata}")
        if closure_metadata.get("carried_recovery_execution_scorecard_ready") is not True:
            raise SystemExit(f"Ready autonomy step closure carried recovery scorecard should be ready: {closure_metadata}")
        if closure_metadata.get("carried_recovery_execution_as_prior_proof") is not True:
            raise SystemExit(f"Ready autonomy step closure should carry recovery execution as prior proof: {closure_metadata}")
        if (
            closure_metadata.get("carried_recovery_execution_authorizes_action_now") is not False
            or closure_metadata.get("carried_recovery_execution_authorizes_risky_work") is not False
            or closure_metadata.get("carried_recovery_execution_authorizes_unreviewed_followthrough") is not False
        ):
            raise SystemExit(f"Ready autonomy step closure carried recovery proof should be non-authorizing: {closure_metadata}")
        if any(
            row.get("ready") is not True
            or row.get("points") != row.get("max_points")
            or row.get("required_before_normal_followthrough") is not True
            or row.get("authorizes_action_now") is not False
            or row.get("authorizes_risky_work") is not False
            or row.get("authorizes_unreviewed_followthrough") is not False
            for row in carried_recovery_rows
        ):
            raise SystemExit(f"Ready autonomy step closure carried recovery rows should be ready and non-authorizing: {closure_metadata}")
        if closure_metadata.get("post_step_artifact_hashes_present") is not True:
            raise SystemExit(f"Ready autonomy step closure missed post-step artifact hashes: {closure_metadata}")
        if closure_metadata.get("post_step_artifact_hashes_match_files") is not True:
            raise SystemExit(f"Ready autonomy step closure missed post-step artifact file binding: {closure_metadata}")
        if closure_metadata.get("post_step_receipt_sha256") != post_step_receipt_sha256 or closure_metadata.get("post_step_checkpoint_sha256") != post_step_checkpoint_sha256:
            raise SystemExit(f"Ready autonomy step closure post-step hash values diverged: {closure_metadata}")
        if closure_metadata.get("post_step_receipt_file_sha256") != post_step_receipt_sha256:
            raise SystemExit(f"Ready autonomy step closure missed actual receipt file hash: {closure_metadata}")
        if closure_metadata.get("post_step_receipt_hash_matches_file") is not True:
            raise SystemExit(f"Ready autonomy step closure missed post-step receipt file binding: {closure_metadata}")
        if closure_metadata.get("post_step_checkpoint_file_sha256") != post_step_checkpoint_sha256:
            raise SystemExit(f"Ready autonomy step closure missed actual checkpoint file hash: {closure_metadata}")
        if closure_metadata.get("post_step_checkpoint_hash_matches_file") is not True:
            raise SystemExit(f"Ready autonomy step closure missed post-step checkpoint file binding: {closure_metadata}")
        if closure_metadata.get("completed_step_matches_proposed") is not True:
            raise SystemExit(f"Ready autonomy step closure missed step identity binding: {closure_metadata}")
        if (
            len(str(closure_metadata.get("proposed_next_step_sha256") or "")) != 64
            or closure_metadata.get("proposed_next_step_sha256") != closure_metadata.get("completed_step_sha256")
        ):
            raise SystemExit(f"Ready autonomy step closure step hashes diverged: {closure_metadata}")
        if closure_metadata.get("post_step_verification_matches_proposed") is not True:
            raise SystemExit(f"Ready autonomy step closure missed verification target binding: {closure_metadata}")
        if (
            len(str(closure_metadata.get("proposed_verification_sha256") or "")) != 64
            or closure_metadata.get("proposed_verification_sha256") != closure_metadata.get("post_step_verification_sha256")
        ):
            raise SystemExit(f"Ready autonomy step closure verification hashes diverged: {closure_metadata}")
        if closure_metadata.get("continuation_review_token_sha256") != execution_metadata.get("continuation_review_token_sha256"):
            raise SystemExit(f"Ready autonomy step closure review token diverged: {closure_metadata}")
        if closure_metadata.get("continuation_review_token_reusable_for_next_review") is not False:
            raise SystemExit(f"Ready autonomy step closure should mark review token non-reusable: {closure_metadata}")
        if closure_metadata.get("next_review_requires_new_continuation_review_token") is not True:
            raise SystemExit(f"Ready autonomy step closure missed next-review token boundary: {closure_metadata}")
        if not _continuation_review_token_metadata_ready(
            closure_metadata,
            reusable_key="continuation_review_token_reusable_for_next_review",
        ):
            raise SystemExit(f"Ready autonomy step closure review token failed mirror validator: {closure_metadata}")
        for missing_key in [
            "continuation_review_token_reusable_for_next_review",
            "previous_continuation_review_token_reusable_for_next_review",
        ]:
            missing_metadata = dict(closure_metadata)
            missing_metadata.pop(missing_key, None)
            if _continuation_review_token_metadata_ready(
                missing_metadata,
                reusable_key="continuation_review_token_reusable_for_next_review",
            ):
                raise SystemExit(
                    f"Ready autonomy step closure review token validator accepted missing mirror {missing_key}: "
                    f"{missing_metadata}"
                )
        fresh_contract = closure_metadata.get("fresh_continuation_review_contract_rows") or []
        if closure_metadata.get("fresh_continuation_review_contract_row_count") != len(fresh_contract):
            raise SystemExit(f"Ready autonomy step closure missed fresh contract row count: {closure_metadata}")
        if closure_metadata.get("fresh_continuation_review_contract_row_count") != 6:
            raise SystemExit(f"Ready autonomy step closure fresh contract should have six rows: {closure_metadata}")
        expected_fresh_contract_items = {
            "fresh_operator_timebox",
            "fresh_checkpoint",
            "fresh_recovery_cockpit",
            "fresh_local_safe_step",
            "fresh_continuation_review_token",
            "prior_step_closure",
        }
        if {row.get("item") for row in fresh_contract} != expected_fresh_contract_items:
            raise SystemExit(f"Ready autonomy step closure fresh contract items diverged: {closure_metadata}")
        if not all(
            row.get("prior_artifact_reusable") is False
            and row.get("authorizes_action_now") is False
            and row.get("authorizes_risky_work") is False
            for row in fresh_contract
        ):
            raise SystemExit(f"Ready autonomy step closure fresh contract should keep prior artifacts non-authorizing: {closure_metadata}")
        fresh_required_items = {row.get("item") for row in fresh_contract if row.get("fresh_required") is True}
        if "prior_step_closure" in fresh_required_items or len(fresh_required_items) != 5:
            raise SystemExit(f"Ready autonomy step closure fresh-required rows diverged: {closure_metadata}")
        if closure_metadata.get("fresh_continuation_review_contract_enforced") is not True:
            raise SystemExit(f"Ready autonomy step closure missed fresh contract enforcement flag: {closure_metadata}")
        if not _fresh_continuation_review_contract_ready(fresh_contract):
            raise SystemExit(f"Ready autonomy step closure fresh contract failed strict validator: {closure_metadata}")
        tampered_fresh_contract = [dict(row) for row in fresh_contract]
        tampered_fresh_contract[0]["source"] = "prior operator timebox"
        if _fresh_continuation_review_contract_ready(tampered_fresh_contract):
            raise SystemExit(f"Ready autonomy step closure fresh contract validator accepted tampered source: {tampered_fresh_contract}")
        tampered_fresh_contract = [dict(row) for row in fresh_contract]
        tampered_fresh_contract[-1]["fresh_required"] = True
        if _fresh_continuation_review_contract_ready(tampered_fresh_contract):
            raise SystemExit(f"Ready autonomy step closure fresh contract validator accepted tampered fresh-required flag: {tampered_fresh_contract}")
        assert_fresh_continuation_review_boundary_token(
            closure_metadata,
            "Ready autonomy step closure",
            expected_source="autonomy_step_closure",
        )
        assert_fresh_continuation_review_hash_binds_authority(
            closure_metadata,
            "Ready autonomy step closure",
        )
        assert_step_closure_receipt_token(
            closure_metadata,
            "Ready autonomy step closure",
        )
        assert_step_closure_receipt_hash_binds_evidence(
            closure_metadata,
            "Ready autonomy step closure",
        )
        if "fresh continuation review boundary token sha256" not in ready_closure.output:
            raise SystemExit("Ready autonomy step closure output missed fresh continuation boundary token.")
        if "step closure receipt token sha256" not in ready_closure.output:
            raise SystemExit("Ready autonomy step closure output missed step closure receipt token.")
        for expected_summary in [
            "fresh_operator_timebox_required",
            "fresh_checkpoint_required",
            "fresh_recovery_cockpit_required",
            "fresh_local_safe_step_required",
            "fresh_continuation_review_token_required",
            "prior_step_closure_proof_only",
            "no_followup_without_new_review",
        ]:
            if expected_summary not in closure_metadata.get("fresh_continuation_review_contract_summary", []):
                raise SystemExit(f"Ready autonomy step closure fresh contract summary missed {expected_summary}: {closure_metadata}")
        for key in [
            "next_continuation_requires_fresh_operator_timebox",
            "next_continuation_requires_fresh_checkpoint",
            "next_continuation_requires_fresh_recovery_cockpit",
            "next_continuation_requires_fresh_local_safe_step",
            "next_continuation_requires_fresh_review_token",
        ]:
            if closure_metadata.get(key) is not True:
                raise SystemExit(f"Ready autonomy step closure missed {key}: {closure_metadata}")
        if closure_metadata.get("prior_step_closure_authorizes_followup") is not False:
            raise SystemExit(f"Ready autonomy step closure should not authorize follow-up: {closure_metadata}")
        if closure_metadata.get("prior_step_closure_reusable_for_next_step") is not False:
            raise SystemExit(f"Ready autonomy step closure should not be reusable for next step: {closure_metadata}")
        assert_prior_cycle_ledger_token_boundary(
            closure_metadata,
            "Ready autonomy step closure",
            expected_source="autonomy_step_closure",
        )
        if closure_metadata.get("proof_queue_count") != len(closure_metadata.get("proof_queue") or []):
            raise SystemExit(f"Ready autonomy step closure missed proof queue count: {closure_metadata}")
        carried_post_step_queue = closure_metadata.get("continuation_post_step_proof_queue") or []
        if not carried_post_step_queue:
            raise SystemExit(f"Ready autonomy step closure missed carried post-step proof queue: {closure_metadata}")
        if closure_metadata.get("continuation_post_step_proof_queue_count") != len(carried_post_step_queue):
            raise SystemExit(f"Ready autonomy step closure missed carried post-step proof queue count: {closure_metadata}")
        if closure_metadata.get("continuation_post_step_next_proof_command") != carried_post_step_queue[0]:
            raise SystemExit(f"Ready autonomy step closure missed carried next post-step proof command: {closure_metadata}")
        for key in READ_ONLY_FALSE_FLAGS:
            if closure_metadata.get(key) is not False:
                raise SystemExit(f"Ready autonomy step closure packet should report {key}=False.")

        mismatched_step_closure = ready_runtime.registry.get("autonomy_step_closure_packet").handler(
            {
                "objective": "continue Jarvis safely",
                "stop_at": "2099-01-01T00:00:00+09:00",
                "current_time": "2026-06-09T03:00:00+09:00",
                "timezone": "Asia/Seoul",
                "reviewed_step": "reviewed local-safe continuity metadata",
                "recovery_verification": "smoke_test_next_step passed",
                "recovery_receipt_path": str(recovery_receipt_path),
                "recovery_receipt_sha256": recovery_receipt_sha256,
                "recovery_checkpoint_path": checkpoint_path,
                "recovery_checkpoint_sha256": recovery_checkpoint_sha256,
                "stop_condition": "stop if verification drifts",
                "blockers": "none",
                "next_step": "update local continuity smoke metadata",
                "next_verification": "smoke_test_next_step passed",
                "completed_step": "update different local continuity metadata",
                "post_step_verification": "smoke_test_next_step passed after step",
                "post_step_receipt_path": str(post_step_receipt_path),
                "post_step_receipt_sha256": post_step_receipt_sha256,
                "post_step_checkpoint_path": str(post_step_checkpoint_path),
                "post_step_checkpoint_sha256": post_step_checkpoint_sha256,
                "execution_health": "execution health reviewed",
                "execution_audit": "execution audit reviewed",
                "after_action_learning": "learning reviewed",
            }
        )
        print("[ok] direct mismatched-step autonomy step closure")
        print(mismatched_step_closure.output[:1800])
        print()
        mismatched_step_metadata = mismatched_step_closure.metadata
        if mismatched_step_metadata.get("closure_state") != "AUTONOMY_STEP_CLOSURE_HELD":
            raise SystemExit(f"Mismatched completed step should hold closure: {mismatched_step_metadata}")
        mismatched_step_scorecard = mismatched_step_metadata.get("step_closure_readiness_scorecard_rows") or []
        if mismatched_step_metadata.get("step_closure_readiness_max_score") != 100:
            raise SystemExit(f"Mismatched step closure scorecard max should be 100: {mismatched_step_metadata}")
        if mismatched_step_metadata.get("step_closure_readiness_required_rows_ready") is not False:
            raise SystemExit(f"Mismatched step closure scorecard should hold required rows: {mismatched_step_metadata}")
        if not any(row.get("item") == "completed_step_matches_allowed" and row.get("ready") is False for row in mismatched_step_scorecard):
            raise SystemExit(f"Mismatched step closure scorecard missed completed-step blocker: {mismatched_step_metadata}")
        if mismatched_step_metadata.get("completed_step_matches_proposed") is not False:
            raise SystemExit(f"Mismatched completed step should fail step identity binding: {mismatched_step_metadata}")
        if "completed step matches allowed continuation step" not in mismatched_step_metadata.get("missing_blockers", []):
            raise SystemExit(f"Mismatched completed step missed blocker: {mismatched_step_metadata}")
        if mismatched_step_metadata.get("proposed_next_step_sha256") == mismatched_step_metadata.get("completed_step_sha256"):
            raise SystemExit(f"Mismatched completed step hashes should differ: {mismatched_step_metadata}")
        for key in READ_ONLY_FALSE_FLAGS:
            if mismatched_step_metadata.get(key) is not False:
                raise SystemExit(f"Mismatched-step closure should report {key}=False.")

        mismatched_verification_closure = ready_runtime.registry.get("autonomy_step_closure_packet").handler(
            {
                "objective": "continue Jarvis safely",
                "stop_at": "2099-01-01T00:00:00+09:00",
                "current_time": "2026-06-09T03:00:00+09:00",
                "timezone": "Asia/Seoul",
                "reviewed_step": "reviewed local-safe continuity metadata",
                "recovery_verification": "smoke_test_next_step passed",
                "recovery_receipt_path": str(recovery_receipt_path),
                "recovery_receipt_sha256": recovery_receipt_sha256,
                "recovery_checkpoint_path": checkpoint_path,
                "recovery_checkpoint_sha256": recovery_checkpoint_sha256,
                "stop_condition": "stop if verification drifts",
                "blockers": "none",
                "next_step": "update local continuity smoke metadata",
                "next_verification": "smoke_test_next_step passed",
                "completed_step": "update local continuity smoke metadata",
                "post_step_verification": "smoke_test_next_step passed after swapped verification",
                "post_step_receipt_path": str(post_step_receipt_path),
                "post_step_receipt_sha256": post_step_receipt_sha256,
                "post_step_checkpoint_path": str(post_step_checkpoint_path),
                "post_step_checkpoint_sha256": post_step_checkpoint_sha256,
                "execution_health": "execution health reviewed",
                "execution_audit": "execution audit reviewed",
                "after_action_learning": "learning reviewed",
            }
        )
        print("[ok] direct mismatched-verification autonomy step closure")
        print(mismatched_verification_closure.output[:1800])
        print()
        mismatched_verification_metadata = mismatched_verification_closure.metadata
        if mismatched_verification_metadata.get("closure_state") != "AUTONOMY_STEP_CLOSURE_HELD":
            raise SystemExit(f"Mismatched post-step verification should hold closure: {mismatched_verification_metadata}")
        mismatched_verification_scorecard = mismatched_verification_metadata.get("step_closure_readiness_scorecard_rows") or []
        if not any(row.get("item") == "post_step_verification_matches_target" and row.get("ready") is False for row in mismatched_verification_scorecard):
            raise SystemExit(f"Mismatched verification closure scorecard missed verification blocker: {mismatched_verification_metadata}")
        if mismatched_verification_metadata.get("post_step_verification_matches_proposed") is not False:
            raise SystemExit(f"Mismatched post-step verification should fail target binding: {mismatched_verification_metadata}")
        if "post-step verification matches allowed verification target" not in mismatched_verification_metadata.get("missing_blockers", []):
            raise SystemExit(f"Mismatched post-step verification missed blocker: {mismatched_verification_metadata}")
        if mismatched_verification_metadata.get("proposed_verification_sha256") == mismatched_verification_metadata.get("post_step_verification_sha256"):
            raise SystemExit(f"Mismatched post-step verification hashes should differ: {mismatched_verification_metadata}")
        for key in READ_ONLY_FALSE_FLAGS:
            if mismatched_verification_metadata.get(key) is not False:
                raise SystemExit(f"Mismatched-verification closure should report {key}=False.")

        mismatched_post_step_receipt_closure = ready_runtime.registry.get("autonomy_step_closure_packet").handler(
            {
                "objective": "continue Jarvis safely",
                "stop_at": "2099-01-01T00:00:00+09:00",
                "current_time": "2026-06-09T03:00:00+09:00",
                "timezone": "Asia/Seoul",
                "reviewed_step": "reviewed local-safe continuity metadata",
                "recovery_verification": "smoke_test_next_step passed",
                "recovery_receipt_path": str(recovery_receipt_path),
                "recovery_receipt_sha256": recovery_receipt_sha256,
                "recovery_checkpoint_path": checkpoint_path,
                "recovery_checkpoint_sha256": recovery_checkpoint_sha256,
                "stop_condition": "stop if verification drifts",
                "blockers": "none",
                "next_step": "update local continuity smoke metadata",
                "next_verification": "smoke_test_next_step passed",
                "completed_step": "update local continuity smoke metadata",
                "post_step_verification": "smoke_test_next_step passed",
                "post_step_receipt_path": str(post_step_receipt_path),
                "post_step_receipt_sha256": mismatched_post_step_receipt_sha256,
                "post_step_checkpoint_path": str(post_step_checkpoint_path),
                "post_step_checkpoint_sha256": post_step_checkpoint_sha256,
                "execution_health": "execution health reviewed",
                "execution_audit": "execution audit reviewed",
                "after_action_learning": "learning reviewed",
            }
        )
        print("[ok] direct mismatched post-step receipt hash autonomy step closure")
        print(mismatched_post_step_receipt_closure.output[:1800])
        print()
        mismatched_post_receipt_metadata = mismatched_post_step_receipt_closure.metadata
        if mismatched_post_receipt_metadata.get("closure_state") != "AUTONOMY_STEP_CLOSURE_HELD":
            raise SystemExit(f"Mismatched post-step receipt hash should hold closure: {mismatched_post_receipt_metadata}")
        if mismatched_post_receipt_metadata.get("post_step_receipt_file_sha256") != post_step_receipt_sha256:
            raise SystemExit(f"Mismatched post-step receipt hash should expose actual file hash: {mismatched_post_receipt_metadata}")
        if mismatched_post_receipt_metadata.get("post_step_receipt_hash_matches_file") is not False:
            raise SystemExit(f"Mismatched post-step receipt hash should fail file binding: {mismatched_post_receipt_metadata}")
        if "post-step verification receipt sha256 matches file" not in mismatched_post_receipt_metadata.get("missing_blockers", []):
            raise SystemExit(f"Mismatched post-step receipt hash missed file-binding blocker: {mismatched_post_receipt_metadata}")
        for key in READ_ONLY_FALSE_FLAGS:
            if mismatched_post_receipt_metadata.get(key) is not False:
                raise SystemExit(f"Mismatched post-step receipt closure should report {key}=False.")

        mismatched_post_step_checkpoint_closure = ready_runtime.registry.get("autonomy_step_closure_packet").handler(
            {
                "objective": "continue Jarvis safely",
                "stop_at": "2099-01-01T00:00:00+09:00",
                "current_time": "2026-06-09T03:00:00+09:00",
                "timezone": "Asia/Seoul",
                "reviewed_step": "reviewed local-safe continuity metadata",
                "recovery_verification": "smoke_test_next_step passed",
                "recovery_receipt_path": str(recovery_receipt_path),
                "recovery_receipt_sha256": recovery_receipt_sha256,
                "recovery_checkpoint_path": checkpoint_path,
                "recovery_checkpoint_sha256": recovery_checkpoint_sha256,
                "stop_condition": "stop if verification drifts",
                "blockers": "none",
                "next_step": "update local continuity smoke metadata",
                "next_verification": "smoke_test_next_step passed",
                "completed_step": "update local continuity smoke metadata",
                "post_step_verification": "smoke_test_next_step passed",
                "post_step_receipt_path": str(post_step_receipt_path),
                "post_step_receipt_sha256": post_step_receipt_sha256,
                "post_step_checkpoint_path": str(post_step_checkpoint_path),
                "post_step_checkpoint_sha256": mismatched_post_step_checkpoint_sha256,
                "execution_health": "execution health reviewed",
                "execution_audit": "execution audit reviewed",
                "after_action_learning": "learning reviewed",
            }
        )
        print("[ok] direct mismatched post-step checkpoint hash autonomy step closure")
        print(mismatched_post_step_checkpoint_closure.output[:1800])
        print()
        mismatched_post_checkpoint_metadata = mismatched_post_step_checkpoint_closure.metadata
        if mismatched_post_checkpoint_metadata.get("closure_state") != "AUTONOMY_STEP_CLOSURE_HELD":
            raise SystemExit(f"Mismatched post-step checkpoint hash should hold closure: {mismatched_post_checkpoint_metadata}")
        if mismatched_post_checkpoint_metadata.get("post_step_checkpoint_file_sha256") != post_step_checkpoint_sha256:
            raise SystemExit(f"Mismatched post-step checkpoint hash should expose actual file hash: {mismatched_post_checkpoint_metadata}")
        if mismatched_post_checkpoint_metadata.get("post_step_checkpoint_hash_matches_file") is not False:
            raise SystemExit(f"Mismatched post-step checkpoint hash should fail file binding: {mismatched_post_checkpoint_metadata}")
        if "fresh post-step checkpoint sha256 matches file" not in mismatched_post_checkpoint_metadata.get("missing_blockers", []):
            raise SystemExit(f"Mismatched post-step checkpoint hash missed file-binding blocker: {mismatched_post_checkpoint_metadata}")
        for key in READ_ONLY_FALSE_FLAGS:
            if mismatched_post_checkpoint_metadata.get(key) is not False:
                raise SystemExit(f"Mismatched post-step checkpoint closure should report {key}=False.")

        invalid_post_step_closure = ready_runtime.registry.get("autonomy_step_closure_packet").handler(
            {
                "objective": "continue Jarvis safely",
                "stop_at": "2099-01-01T00:00:00+09:00",
                "current_time": "2026-06-09T03:00:00+09:00",
                "timezone": "Asia/Seoul",
                "reviewed_step": "reviewed local-safe continuity metadata",
                "recovery_verification": "smoke_test_next_step passed",
                "recovery_receipt_path": str(recovery_receipt_path),
                "recovery_receipt_sha256": recovery_receipt_sha256,
                "recovery_checkpoint_path": checkpoint_path,
                "recovery_checkpoint_sha256": recovery_checkpoint_sha256,
                "stop_condition": "stop if verification drifts",
                "blockers": "none",
                "next_step": "update local continuity smoke metadata",
                "next_verification": "smoke_test_next_step passed",
                "completed_step": "update local continuity smoke metadata",
                "post_step_verification": "smoke_test_next_step passed",
                "post_step_receipt_path": str(post_step_receipt_path),
                "post_step_receipt_sha256": "invalid-post-receipt",
                "post_step_checkpoint_path": str(post_step_checkpoint_path),
                "post_step_checkpoint_sha256": "invalid-post-checkpoint",
                "execution_health": "execution health reviewed",
                "execution_audit": "execution audit reviewed",
                "after_action_learning": "learning reviewed",
            }
        )
        print("[ok] direct invalid post-step hash autonomy step closure")
        print(invalid_post_step_closure.output[:1800])
        print()
        invalid_post_metadata = invalid_post_step_closure.metadata
        if invalid_post_metadata.get("closure_state") != "AUTONOMY_STEP_CLOSURE_HELD":
            raise SystemExit(f"Invalid post-step hashes should hold closure: {invalid_post_metadata}")
        if invalid_post_metadata.get("post_step_artifact_hashes_present") is not False:
            raise SystemExit(f"Invalid post-step hashes should not count as present: {invalid_post_metadata}")
        for blocker in ["valid post-step verification receipt sha256", "valid fresh post-step checkpoint sha256"]:
            if blocker not in invalid_post_metadata.get("missing_blockers", []):
                raise SystemExit(f"Invalid post-step hashes missed blocker {blocker!r}: {invalid_post_metadata}")
        for key in READ_ONLY_FALSE_FLAGS:
            if invalid_post_metadata.get(key) is not False:
                raise SystemExit(f"Invalid post-step closure should report {key}=False.")

        ready_ledger = ready_runtime.registry.get("autonomy_cycle_ledger").handler(
            {
                "objective": "continue Jarvis safely",
                "stop_at": "2099-01-01T00:00:00+09:00",
                "current_time": "2026-06-09T03:00:00+09:00",
                "timezone": "Asia/Seoul",
                "reviewed_step": "reviewed local-safe continuity metadata",
                "recovery_verification": "smoke_test_next_step passed",
                "recovery_receipt_path": str(recovery_receipt_path),
                "recovery_receipt_sha256": recovery_receipt_sha256,
                "recovery_checkpoint_path": checkpoint_path,
                "recovery_checkpoint_sha256": recovery_checkpoint_sha256,
                "stop_condition": "stop if verification drifts",
                "blockers": "none",
                "next_step": "update local continuity smoke metadata",
                "next_verification": "smoke_test_next_step passed",
                "completed_step": "update local continuity smoke metadata",
                "post_step_verification": "smoke_test_next_step passed",
                "post_step_receipt_path": str(post_step_receipt_path),
                "post_step_receipt_sha256": post_step_receipt_sha256,
                "post_step_checkpoint_path": str(post_step_checkpoint_path),
                "post_step_checkpoint_sha256": post_step_checkpoint_sha256,
                "execution_health": "execution health reviewed",
                "execution_audit": "execution audit reviewed",
                "after_action_learning": "learning reviewed",
            }
        )
        print("[ok] direct ready autonomy cycle ledger")
        print(ready_ledger.output[:1800])
        print()
        if not ready_ledger.ok:
            raise SystemExit("Ready autonomy cycle ledger should run.")
        ledger_metadata = ready_ledger.metadata
        if ledger_metadata.get("cycle_state") != "AUTONOMY_CYCLE_LEDGER_READY_FOR_FRESH_REVIEW":
            raise SystemExit(f"Ready autonomy cycle ledger should permit a fresh review: {ledger_metadata}")
        assert_timebox_review_contract(ledger_metadata, "Ready autonomy cycle ledger")
        assert_operator_supersession_token_boundary(
            ledger_metadata,
            "Ready autonomy cycle ledger",
            expected_source="operator_instruction_supersession",
        )
        assert_checkpoint_route_boundary(
            ledger_metadata,
            "Ready autonomy cycle ledger",
            expected_source="checkpoint_recovery_cockpit",
        )
        if ledger_metadata.get("ready_for_fresh_next_continuation_review") is not True or ledger_metadata.get("ready_for_next_continuation_review") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed fresh review permission: {ledger_metadata}")
        if ledger_metadata.get("missing_blockers") != [] or ledger_metadata.get("missing_blocker_count") != 0:
            raise SystemExit(f"Ready autonomy cycle ledger missed explicit zero-blocker contract: {ledger_metadata}")
        required_commands = ledger_metadata.get("required_commands") or []
        if not required_commands:
            raise SystemExit(f"Ready autonomy cycle ledger missed required proof commands: {ledger_metadata}")
        if ledger_metadata.get("required_command_count") != len(required_commands):
            raise SystemExit(f"Ready autonomy cycle ledger required command count diverged: {ledger_metadata}")
        if ledger_metadata.get("proof_queue") != required_commands:
            raise SystemExit(f"Ready autonomy cycle ledger proof queue should mirror required commands: {ledger_metadata}")
        if ledger_metadata.get("proof_queue_count") != len(required_commands):
            raise SystemExit(f"Ready autonomy cycle ledger proof queue count diverged: {ledger_metadata}")
        if ledger_metadata.get("next_required_command") != required_commands[0]:
            raise SystemExit(f"Ready autonomy cycle ledger missed next required command: {ledger_metadata}")
        if ledger_metadata.get("next_proof_command") != required_commands[0]:
            raise SystemExit(f"Ready autonomy cycle ledger missed next proof command: {ledger_metadata}")
        carried_post_step_queue = ledger_metadata.get("post_step_proof_queue") or []
        if not carried_post_step_queue:
            raise SystemExit(f"Ready autonomy cycle ledger missed carried post-step proof queue: {ledger_metadata}")
        if ledger_metadata.get("post_step_proof_queue_count") != len(carried_post_step_queue):
            raise SystemExit(f"Ready autonomy cycle ledger missed carried post-step proof queue count: {ledger_metadata}")
        if ledger_metadata.get("post_step_next_proof_command") != carried_post_step_queue[0]:
            raise SystemExit(f"Ready autonomy cycle ledger missed carried next post-step proof command: {ledger_metadata}")
        for carried_closure_bool_key in [
            "recovery_artifact_hashes_present",
            "recovery_artifact_hashes_match_files",
            "completed_step_matches_proposed",
            "post_step_verification_matches_proposed",
            "post_step_artifact_hashes_present",
            "post_step_artifact_hashes_match_files",
            "carried_step_closure_readiness_required_rows_ready",
            "carried_step_closure_receipt_token_present",
            "carried_step_closure_receipt_boundary_ready",
            "recovery_execution_readiness_token_present",
            "recovery_execution_readiness_token_boundary_ready",
            "fresh_continuation_review_boundary_token_present",
            "fresh_continuation_review_boundary_token_ready",
            "timebox_receipt_present",
            "awake_guard_token_present",
            "awake_guard_os_wake_lock_boundary_ready",
            "supersession_token_present",
            "supersession_token_boundary_ready",
            "checkpoint_route_token_present",
            "checkpoint_route_boundary_ready",
            "recovery_followthrough_token_present",
            "recovery_followthrough_token_boundary_ready",
            "local_safe_recovery_execution_token_present",
            "local_safe_recovery_execution_token_boundary_ready",
            "one_step_execution_contract_ready",
            "one_step_execution_contract_token_as_prior_proof",
            "one_step_execution_contract_binds_awake_guard",
            "one_step_execution_contract_binds_operator_supersession",
            "one_step_execution_contract_binds_timebox_review_contract",
            "one_step_execution_contract_all_local_safe_step_limited",
            "one_step_execution_contract_all_non_reusable",
            "one_step_execution_contract_all_risky_work_gated",
            "one_step_execution_contract_requires_fresh_closure",
            "continuation_review_token_present",
        ]:
            if not _metadata_flag_ready(ledger_metadata, carried_closure_bool_key):
                raise SystemExit(
                    f"Ready autonomy cycle ledger should carry exact True for {carried_closure_bool_key}: "
                    f"{ledger_metadata}"
                )
            tampered_ledger_metadata = json.loads(json.dumps(ledger_metadata))
            tampered_ledger_metadata[carried_closure_bool_key] = "false"
            if _metadata_flag_ready(tampered_ledger_metadata, carried_closure_bool_key):
                raise SystemExit(
                    f"Exact metadata flag helper accepted string false for {carried_closure_bool_key}: "
                    f"{tampered_ledger_metadata}"
                )
            if _autonomy_cycle_ledger_ready_from_metadata(tampered_ledger_metadata):
                raise SystemExit(
                    f"Ready autonomy cycle ledger accepted malformed carried closure bool {carried_closure_bool_key}: "
                    f"{tampered_ledger_metadata}"
                )
        if ledger_metadata.get("autonomy_cycle_ledger_ready") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed all-up readiness flag: {ledger_metadata}")
        if not _autonomy_cycle_ledger_ready_from_metadata(ledger_metadata):
            raise SystemExit(f"Ready autonomy cycle ledger failed all-up production validator: {ledger_metadata}")
        if ledger_metadata.get("next_review_start_command") != "autonomy continuation execution: <next reviewed local-safe step>":
            raise SystemExit(f"Ready autonomy cycle ledger missed exact next-review start command: {ledger_metadata}")
        if ledger_metadata.get("next_safe_command") != ledger_metadata.get("next_review_start_command"):
            raise SystemExit(f"Ready autonomy cycle ledger next-safe command drifted from next-review start command: {ledger_metadata}")
        if (
            (ledger_metadata.get("cycle_state") == "AUTONOMY_CYCLE_LEDGER_READY_FOR_FRESH_REVIEW")
            is not ledger_metadata.get("autonomy_cycle_ledger_ready")
            or ledger_metadata.get("ready_for_fresh_next_continuation_review")
            is not ledger_metadata.get("autonomy_cycle_ledger_ready")
        ):
            raise SystemExit(f"Ready autonomy cycle ledger public state drifted from all-up readiness: {ledger_metadata}")
        for missing_blocker_key in ["missing_blockers", "missing_blocker_count"]:
            tampered_ledger_metadata = json.loads(json.dumps(ledger_metadata))
            tampered_ledger_metadata.pop(missing_blocker_key, None)
            if _autonomy_cycle_ledger_ready_from_metadata(tampered_ledger_metadata):
                raise SystemExit(
                    f"Ready autonomy cycle ledger accepted missing blocker contract field {missing_blocker_key}: "
                    f"{tampered_ledger_metadata}"
                )
        for key, value in [
            ("required_commands", ["autonomy cycle ledger: stale"]),
            ("required_command_count", 999),
            ("proof_queue", ["autonomy cycle ledger: stale"]),
            ("proof_queue_count", 999),
            ("next_required_command", "autonomy cycle ledger: stale"),
            ("next_proof_command", "autonomy cycle ledger: stale"),
            ("timebox_receipt_present", False),
            ("timebox_review_contract_row_count", 999),
            ("timebox_review_contract_ready", False),
            ("next_step_requires_fresh_timebox", False),
            ("timebox_authorizes_timebox_reuse", True),
            ("checkpoint_route_token_present", False),
            ("checkpoint_route_boundary_row_count", 999),
            ("checkpoint_route_boundary_ready", False),
            ("next_checkpoint_route_requires_fresh_recovery_review", False),
            ("checkpoint_route_authorizes_checkpoint_reuse", True),
            ("one_step_execution_contract_token_present", False),
            ("one_step_execution_contract_row_count", 999),
            ("one_step_execution_contract_ready", False),
            ("next_step_requires_new_one_step_execution_contract_token", False),
            ("one_step_execution_contract_token_authorizes_action_now", True),
            ("one_step_execution_contract_token_reusable_for_next_step", True),
            ("continuation_review_token_present", False),
            ("previous_continuation_permission_reusable_for_next_review", True),
            ("previous_continuation_review_token_reusable_for_next_review", True),
            ("continuation_review_token_reusable_for_next_review", True),
            ("continuation_review_token_reusable_for_next_review", "false"),
            ("next_review_requires_new_continuation_review_token", False),
            ("previous_post_step_receipt_reusable_for_next_review", True),
            ("previous_post_step_receipt_hash_reusable_for_next_review", True),
            ("previous_post_step_checkpoint_hash_reusable_for_next_review", True),
            ("previous_recovery_followthrough_token_reusable_for_next_review", True),
            ("previous_local_safe_recovery_execution_token_reusable_for_next_review", True),
            ("autonomy_cycle_ledger_token_present", False),
            ("autonomy_cycle_ledger_token_boundary_row_count", 999),
            ("autonomy_cycle_ledger_token_boundary_ready", False),
            ("autonomy_cycle_ledger_token_authorizes_action_now", True),
            ("next_review_requires_new_cycle_ledger_token", False),
            ("fresh_review_boundary_token_present", False),
            ("fresh_review_boundary_token_row_count", 999),
            ("fresh_review_boundary_token_ready", False),
            ("next_review_requires_new_fresh_review_boundary_token", False),
            ("fresh_review_boundary_token_authorizes_action_now", True),
            ("fresh_review_boundary_token_reusable_for_next_review", True),
            ("fresh_review_boundary_token_reusable_for_next_cycle", True),
            ("next_review_start_command", ""),
            ("next_review_start_command", "autonomy continuation execution: stale"),
            ("next_review_start_command_token_present", False),
            ("next_review_start_command_boundary_row_count", 999),
            ("next_review_start_command_boundary_ready", False),
            ("next_review_start_command_requires_fresh_preflight", False),
            ("next_review_start_command_reusable_for_next_review", True),
            ("next_review_start_command_reusable_for_next_cycle", True),
            ("carried_step_closure_receipt_token_present", False),
            ("carried_step_closure_receipt_boundary_row_count", 999),
            ("carried_step_closure_receipt_boundary_ready", False),
            ("next_review_requires_new_step_closure_receipt_token", False),
            ("carried_step_closure_receipt_authorizes_action_now", True),
            ("carried_step_closure_receipt_reusable_for_next_review", True),
            ("prior_cycle_ledger_token_present", True),
            ("prior_cycle_ledger_token_boundary_row_count", 999),
            ("prior_cycle_ledger_token_boundary_ready", False),
            ("prior_cycle_ledger_token_reusable_for_this_cycle", True),
            ("prior_cycle_ledger_proof_authorizes_new_action", True),
            ("next_safe_command", "autonomy cycle ledger: stale"),
            ("fresh_review_preflight_queue", ["operator timebox contract: stop_at=<ISO> current_time=<ISO>"]),
            ("fresh_review_preflight_queue_count", 999),
            ("fresh_review_next_preflight_command", "stale"),
            ("carried_next_step_approval_proof_queue_count", 999),
            ("carried_next_step_approval_boundary_row_count", 999),
            ("carried_next_step_approval_boundary_token_present", False),
            ("carried_next_step_approval_boundary_ready", False),
            ("carried_next_step_approval_boundary_as_prior_proof", False),
            ("carried_next_step_approval_required_before_review", True),
            ("carried_next_step_next_approval_proof_command", "stale approval proof command"),
            ("carried_next_step_approval_boundary_authorizes_approval", True),
            ("carried_next_step_approval_boundary_authorizes_model_call", True),
            ("carried_next_step_approval_boundary_authorizes_tool_execution", True),
            ("carried_next_step_approval_boundary_authorizes_personal_data_read", True),
            ("carried_next_step_approval_boundary_authorizes_external_side_effect", True),
            ("carried_next_step_approval_boundary_authorizes_timebox_reuse", True),
            ("carried_next_step_approval_boundary_reusable_for_recovery_review", True),
        ]:
            tampered_ledger_metadata = json.loads(json.dumps(ledger_metadata))
            tampered_ledger_metadata[key] = value
            if _autonomy_cycle_ledger_ready_from_metadata(tampered_ledger_metadata):
                raise SystemExit(
                    f"Ready autonomy cycle ledger accepted stale next-review/preflight field {key}: "
                    f"{tampered_ledger_metadata}"
                )
        tampered_ledger_metadata = json.loads(json.dumps(ledger_metadata))
        tampered_ledger_metadata["autonomy_preflight_scorecard_rows"][0]["authorizes_risky_work"] = True
        if _autonomy_cycle_ledger_ready_from_metadata(tampered_ledger_metadata):
            raise SystemExit("Ready autonomy cycle ledger accepted a tampered preflight authority row.")
        tampered_ledger_metadata = json.loads(json.dumps(ledger_metadata))
        tampered_ledger_metadata["autonomy_cycle_ledger_token_boundary_rows"][0]["reusable_for_next_cycle"] = True
        if _autonomy_cycle_ledger_ready_from_metadata(tampered_ledger_metadata):
            raise SystemExit("Ready autonomy cycle ledger accepted a reusable cycle-ledger token boundary.")
        tampered_ledger_metadata = json.loads(json.dumps(ledger_metadata))
        tampered_ledger_metadata["stage_rows"][0]["ready"] = False
        if _autonomy_cycle_ledger_ready_from_metadata(tampered_ledger_metadata):
            raise SystemExit("Ready autonomy cycle ledger accepted a tampered stage readiness row.")
        tampered_ledger_metadata = json.loads(json.dumps(ledger_metadata))
        tampered_ledger_metadata["step_closure_metadata"]["step_closure_ready_contract_ready"] = False
        if _autonomy_cycle_ledger_ready_from_metadata(tampered_ledger_metadata):
            raise SystemExit("Ready autonomy cycle ledger accepted a tampered nested step-closure contract.")
        if ledger_metadata.get("stage_count") != 8 or ledger_metadata.get("stage_count") != len(ledger_metadata.get("stage_rows") or []):
            raise SystemExit(f"Ready autonomy cycle ledger missed stage rows: {ledger_metadata}")
        if not all(row.get("ready") is True for row in ledger_metadata.get("stage_rows") or []):
            raise SystemExit(f"Ready autonomy cycle ledger should mark every stage ready: {ledger_metadata}")
        if ledger_metadata.get("autonomy_cycle_stage_rows_ready") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed strict stage-row readiness flag: {ledger_metadata}")
        if not _autonomy_cycle_stage_rows_ready(list(ledger_metadata.get("stage_rows") or [])):
            raise SystemExit(f"Ready autonomy cycle ledger stage rows failed production validator: {ledger_metadata}")
        tampered_stage_rows = [dict(row) for row in (ledger_metadata.get("stage_rows") or [])]
        tampered_stage_rows[0]["authorizes_tool_execution"] = True
        if _autonomy_cycle_stage_rows_ready(tampered_stage_rows):
            raise SystemExit(f"Ready autonomy cycle ledger accepted executable stage-row authority: {tampered_stage_rows}")
        tampered_stage_rows = [dict(row) for row in (ledger_metadata.get("stage_rows") or [])]
        tampered_stage_rows[0]["proof"] = ""
        if _autonomy_cycle_stage_rows_ready(tampered_stage_rows):
            raise SystemExit(f"Ready autonomy cycle ledger accepted stage row without proof: {tampered_stage_rows}")
        scorecard_rows = ledger_metadata.get("autonomy_preflight_scorecard_rows") or []
        expected_scorecard_items = {
            "pre_step_continuation_permission",
            "cycle_stage_readiness",
            "recovery_artifact_file_binding",
            "completed_step_identity",
            "post_step_verification_identity",
            "post_step_artifact_file_binding",
            "fresh_continuation_contract",
            "prior_artifacts_non_authorizing",
            "fresh_review_cycle_boundary",
        }
        if ledger_metadata.get("autonomy_preflight_scorecard_row_count") != 9 or len(scorecard_rows) != 9:
            raise SystemExit(f"Ready autonomy cycle ledger missed preflight scorecard rows: {ledger_metadata}")
        if {row.get("item") for row in scorecard_rows} != expected_scorecard_items:
            raise SystemExit(f"Ready autonomy cycle ledger missed preflight scorecard items: {ledger_metadata}")
        if sum(int(row.get("max_points", 0)) for row in scorecard_rows) != 100:
            raise SystemExit(f"Ready autonomy cycle ledger scorecard should total 100 points: {ledger_metadata}")
        if ledger_metadata.get("autonomy_preflight_score") != 100 or ledger_metadata.get("autonomy_preflight_max_score") != 100:
            raise SystemExit(f"Ready autonomy cycle ledger missed complete preflight score: {ledger_metadata}")
        if ledger_metadata.get("autonomy_preflight_required_rows_ready") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed ready preflight flag: {ledger_metadata}")
        if ledger_metadata.get("autonomy_preflight_scorecard_ready") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed ready preflight scorecard shape: {ledger_metadata}")
        assert_autonomy_cycle_preflight_scorecard(
            ledger_metadata,
            "Ready autonomy cycle ledger",
            ready=True,
        )
        if any(
            row.get("ready") is not True
            or row.get("points") != row.get("max_points")
            or row.get("required_before_next_continuation_review") is not True
            or row.get("authorizes_action") is not False
            or row.get("authorizes_risky_work") is not False
            for row in scorecard_rows
        ):
            raise SystemExit(f"Ready autonomy cycle ledger scorecard rows should be ready and non-authorizing: {ledger_metadata}")
        carried_closure_scorecard = ledger_metadata.get("carried_step_closure_readiness_scorecard_rows") or []
        if "Carried step-closure readiness proof" not in ready_ledger.output:
            raise SystemExit("Ready autonomy cycle ledger output missed carried closure readiness proof text.")
        if ledger_metadata.get("carried_step_closure_readiness_scorecard_row_count") != len(carried_closure_scorecard):
            raise SystemExit(f"Ready autonomy cycle ledger missed carried closure scorecard row count: {ledger_metadata}")
        if ledger_metadata.get("carried_step_closure_readiness_scorecard_row_count") != 9:
            raise SystemExit(f"Ready autonomy cycle ledger should carry nine closure scorecard rows: {ledger_metadata}")
        if ledger_metadata.get("carried_step_closure_readiness_score") != 100 or ledger_metadata.get("carried_step_closure_readiness_max_score") != 100:
            raise SystemExit(f"Ready autonomy cycle ledger should carry 100/100 closure score: {ledger_metadata}")
        if ledger_metadata.get("carried_step_closure_readiness_required_rows_ready") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger should carry ready closure rows: {ledger_metadata}")
        if ledger_metadata.get("carried_step_closure_readiness_scorecard_ready") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger should carry ready closure scorecard shape: {ledger_metadata}")
        assert_autonomy_step_closure_scorecard(
            ledger_metadata,
            "Ready autonomy cycle ledger carried closure",
            ready=True,
            prefix="carried_step_closure",
        )
        if ledger_metadata.get("carried_step_closure_readiness_as_prior_proof") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger should carry closure readiness as prior proof: {ledger_metadata}")
        if (
            ledger_metadata.get("carried_step_closure_readiness_authorizes_action_now") is not False
            or ledger_metadata.get("carried_step_closure_readiness_authorizes_risky_work") is not False
            or ledger_metadata.get("carried_step_closure_readiness_authorizes_followup_without_fresh_review") is not False
        ):
            raise SystemExit(f"Ready autonomy cycle ledger carried closure proof should not authorize action: {ledger_metadata}")
        if {row.get("item") for row in carried_closure_scorecard} != {
            "pre_step_permission_ready",
            "completed_step_matches_allowed",
            "post_step_verification_matches_target",
            "post_step_receipt_hash_binding",
            "post_step_checkpoint_hash_binding",
            "execution_health_supplied",
            "execution_audit_supplied",
            "after_action_learning_supplied",
            "fresh_review_contract_non_reusable",
        }:
            raise SystemExit(f"Ready autonomy cycle ledger carried closure scorecard items diverged: {ledger_metadata}")
        if any(
            row.get("ready") is not True
            or row.get("points") != row.get("max_points")
            or row.get("required_before_next_continuation_review") is not True
            or row.get("authorizes_action") is not False
            or row.get("authorizes_risky_work") is not False
            or row.get("authorizes_followup_without_fresh_review") is not False
            for row in carried_closure_scorecard
        ):
            raise SystemExit(f"Ready autonomy cycle ledger carried closure rows should be ready and non-authorizing: {ledger_metadata}")
        if not _autonomy_scorecard_ready(
            carried_closure_scorecard,
            expected_items=_AUTONOMY_STEP_CLOSURE_SCORECARD_ITEMS,
            required_field="required_before_next_continuation_review",
        ):
            raise SystemExit(f"Ready autonomy cycle ledger carried closure scorecard failed strict validator: {ledger_metadata}")
        tampered_carried_closure_scorecard = [dict(row) for row in carried_closure_scorecard]
        tampered_carried_closure_scorecard[0]["item"] = "prior_step_permission_ready"
        if _autonomy_scorecard_ready(
            tampered_carried_closure_scorecard,
            expected_items=_AUTONOMY_STEP_CLOSURE_SCORECARD_ITEMS,
            required_field="required_before_next_continuation_review",
        ):
            raise SystemExit(f"Ready autonomy cycle ledger carried closure scorecard accepted tampered item: {tampered_carried_closure_scorecard}")
        tampered_carried_closure_scorecard = [dict(row) for row in carried_closure_scorecard]
        tampered_carried_closure_scorecard[0]["required_before_next_continuation_review"] = False
        if _autonomy_scorecard_ready(
            tampered_carried_closure_scorecard,
            expected_items=_AUTONOMY_STEP_CLOSURE_SCORECARD_ITEMS,
            required_field="required_before_next_continuation_review",
        ):
            raise SystemExit(f"Ready autonomy cycle ledger carried closure scorecard accepted tampered required field: {tampered_carried_closure_scorecard}")
        assert_step_closure_receipt_token(
            ledger_metadata,
            "Ready autonomy cycle ledger",
            prefix="carried_step_closure",
        )
        assert_step_closure_receipt_hash_binds_evidence(
            ledger_metadata,
            "Ready autonomy cycle ledger",
            prefix="carried_step_closure",
        )
        carried_recovery_rows = ledger_metadata.get("carried_recovery_execution_scorecard_rows") or []
        expected_recovery_execution_items = {
            "reviewed_step_permission",
            "verification_target",
            "receipt_hash_binding",
            "checkpoint_hash_binding",
            "stop_condition",
            "risky_recovery_approval_boundary",
            "followthrough_packet_ready",
            "fresh_followthrough_token",
        }
        if "Carried recovery execution readiness proof" not in ready_ledger.output:
            raise SystemExit("Ready autonomy cycle ledger output missed carried recovery execution proof text.")
        if ledger_metadata.get("carried_recovery_execution_scorecard_row_count") != 8 or len(carried_recovery_rows) != 8:
            raise SystemExit(f"Ready autonomy cycle ledger missed carried recovery execution rows: {ledger_metadata}")
        if {row.get("item") for row in carried_recovery_rows} != expected_recovery_execution_items:
            raise SystemExit(f"Ready autonomy cycle ledger carried recovery rows diverged: {ledger_metadata}")
        if ledger_metadata.get("carried_recovery_execution_score") != 100 or ledger_metadata.get("carried_recovery_execution_max_score") != 100:
            raise SystemExit(f"Ready autonomy cycle ledger carried recovery score should be complete: {ledger_metadata}")
        if ledger_metadata.get("carried_recovery_execution_required_rows_ready") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger carried recovery rows should be ready: {ledger_metadata}")
        if ledger_metadata.get("carried_recovery_execution_scorecard_ready") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger carried recovery scorecard should be ready: {ledger_metadata}")
        if ledger_metadata.get("carried_recovery_execution_as_prior_proof") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger should carry recovery execution as prior proof: {ledger_metadata}")
        if (
            ledger_metadata.get("carried_recovery_execution_authorizes_action_now") is not False
            or ledger_metadata.get("carried_recovery_execution_authorizes_risky_work") is not False
            or ledger_metadata.get("carried_recovery_execution_authorizes_unreviewed_followthrough") is not False
        ):
            raise SystemExit(f"Ready autonomy cycle ledger carried recovery proof should be non-authorizing: {ledger_metadata}")
        if any(
            row.get("ready") is not True
            or row.get("points") != row.get("max_points")
            or row.get("required_before_normal_followthrough") is not True
            or row.get("authorizes_action_now") is not False
            or row.get("authorizes_risky_work") is not False
            or row.get("authorizes_unreviewed_followthrough") is not False
            for row in carried_recovery_rows
        ):
            raise SystemExit(f"Ready autonomy cycle ledger carried recovery rows should be ready and non-authorizing: {ledger_metadata}")
        stage_states = {row.get("stage"): row.get("state") for row in ledger_metadata.get("stage_rows") or []}
        expected_stage_states = {
            "operator_timebox": "STOP_WINDOW_ACTIVE",
            "operator_supersession": "LATEST_INSTRUCTION_READY_TO_GOVERN_CONTINUATION",
            "checkpoint_route": "fresh",
            "recovery_cockpit": "RECOVERY_COCKPIT_READY_FOR_LOCAL_SAFE_REVIEW",
            "resume_gate": "AUTONOMY_RESUME_READY_FOR_LOCAL_SAFE_CONTINUATION",
            "one_step_continuation": "AUTONOMY_CONTINUATION_READY_FOR_ONE_LOCAL_SAFE_STEP",
            "recovery_followthrough": "RECOVERY_FOLLOWTHROUGH_READY",
            "post_step_closure": "AUTONOMY_STEP_CLOSURE_READY_FOR_NEXT_CONTINUATION_REVIEW",
        }
        if stage_states != expected_stage_states:
            raise SystemExit(f"Ready autonomy cycle ledger should preserve precise stage states: {ledger_metadata}")
        for key, expected in [
            ("timebox_state", "STOP_WINDOW_ACTIVE"),
            ("cockpit_state", "RECOVERY_COCKPIT_READY_FOR_LOCAL_SAFE_REVIEW"),
            ("resume_gate_state", "AUTONOMY_RESUME_READY_FOR_LOCAL_SAFE_CONTINUATION"),
            ("followthrough_state", "RECOVERY_FOLLOWTHROUGH_READY"),
        ]:
            if ledger_metadata.get(key) != expected:
                raise SystemExit(f"Ready autonomy cycle ledger missed {key}: {ledger_metadata}")
        if ledger_metadata.get("next_review_requires_fresh_operator_timebox") is not True or ledger_metadata.get("next_review_requires_fresh_checkpoint") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed fresh-review requirements: {ledger_metadata}")
        fresh_review_queue = ledger_metadata.get("fresh_review_preflight_queue") or []
        if ledger_metadata.get("fresh_review_preflight_queue_count") != len(fresh_review_queue):
            raise SystemExit(f"Ready autonomy cycle ledger missed fresh-review preflight queue count: {ledger_metadata}")
        if ledger_metadata.get("fresh_review_next_preflight_command") != "operator timebox contract: stop_at=<ISO> current_time=<ISO>":
            raise SystemExit(f"Ready autonomy cycle ledger missed first fresh-review preflight command: {ledger_metadata}")
        for expected_command in [
            "operator timebox contract: stop_at=<ISO> current_time=<ISO>",
            "work block checkpoint",
            "checkpoint recovery cockpit",
            "autonomy continuation execution: <next reviewed local-safe step>",
        ]:
            if expected_command not in fresh_review_queue:
                raise SystemExit(f"Ready autonomy cycle ledger missed fresh-review preflight command {expected_command!r}: {ledger_metadata}")
        fresh_review_contract = ledger_metadata.get("fresh_review_contract_rows") or []
        if ledger_metadata.get("fresh_review_contract_count") != len(fresh_review_contract):
            raise SystemExit(f"Ready autonomy cycle ledger missed fresh-review contract count: {ledger_metadata}")
        if ledger_metadata.get("fresh_review_contract_count") != 5:
            raise SystemExit(f"Ready autonomy cycle ledger should expose five fresh-review contract rows: {ledger_metadata}")
        expected_contract_items = {
            "operator_timebox",
            "work_block_checkpoint",
            "checkpoint_recovery_cockpit",
            "one_step_continuation_review",
            "prior_cycle_ledger_token",
        }
        if {row.get("item") for row in fresh_review_contract} != expected_contract_items:
            raise SystemExit(f"Ready autonomy cycle ledger missed fresh-review contract items: {ledger_metadata}")
        if any(
            row.get("fresh_required") is not True
            or row.get("prior_artifact_reusable") is not False
            or row.get("authorizes_action_now") is not False
            or row.get("authorizes_local_safe_step") is not False
            or row.get("authorizes_risky_work") is not False
            or row.get("authorizes_unreviewed_followthrough") is not False
            or row.get("authorizes_timebox_reuse") is not False
            or row.get("authorizes_checkpoint_reuse") is not False
            or row.get("authorizes_token_reuse") is not False
            or row.get("authorizes_model_call") is not False
            or row.get("authorizes_tool_execution") is not False
            or row.get("authorizes_personal_data_read") is not False
            or row.get("authorizes_external_side_effect") is not False
            for row in fresh_review_contract
        ):
            raise SystemExit(f"Ready autonomy cycle ledger fresh-review contract should keep prior proof non-authorizing: {ledger_metadata}")
        if ledger_metadata.get("fresh_review_contract_ready") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed fresh-review contract ready flag: {ledger_metadata}")
        for key in [
            "prior_artifacts_authorize_local_safe_step",
            "prior_artifacts_authorize_risky_work",
            "prior_artifacts_authorize_unreviewed_followthrough",
            "prior_artifacts_authorize_timebox_reuse",
            "prior_artifacts_authorize_checkpoint_reuse",
            "prior_artifacts_authorize_token_reuse",
            "prior_artifacts_authorize_model_call",
            "prior_artifacts_authorize_tool_execution",
            "prior_artifacts_authorize_personal_data_read",
            "prior_artifacts_authorize_external_side_effect",
        ]:
            if ledger_metadata.get(key) is not False:
                raise SystemExit(f"Ready autonomy cycle ledger should report {key}=False: {ledger_metadata}")
        if ledger_metadata.get("all_prior_artifacts_non_authorizing") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed non-authorizing prior-artifact summary: {ledger_metadata}")
        closure_fresh_contract = ledger_metadata.get("closure_fresh_continuation_review_contract_rows") or []
        if ledger_metadata.get("closure_fresh_continuation_review_contract_row_count") != len(closure_fresh_contract):
            raise SystemExit(f"Ready autonomy cycle ledger missed closure fresh contract row count: {ledger_metadata}")
        if ledger_metadata.get("closure_fresh_continuation_review_contract_row_count") != 6:
            raise SystemExit(f"Ready autonomy cycle ledger should carry six closure fresh-contract rows: {ledger_metadata}")
        if {row.get("item") for row in closure_fresh_contract} != {
            "fresh_operator_timebox",
            "fresh_checkpoint",
            "fresh_recovery_cockpit",
            "fresh_local_safe_step",
            "fresh_continuation_review_token",
            "prior_step_closure",
        }:
            raise SystemExit(f"Ready autonomy cycle ledger closure fresh contract items diverged: {ledger_metadata}")
        if ledger_metadata.get("closure_fresh_continuation_review_contract_enforced") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed closure fresh contract enforcement: {ledger_metadata}")
        if not _fresh_continuation_review_contract_ready(closure_fresh_contract):
            raise SystemExit(f"Ready autonomy cycle ledger closure fresh contract failed strict validator: {ledger_metadata}")
        tampered_closure_fresh_contract = [dict(row) for row in closure_fresh_contract]
        tampered_closure_fresh_contract[1]["source"] = "stale work block checkpoint"
        if _fresh_continuation_review_contract_ready(tampered_closure_fresh_contract):
            raise SystemExit(f"Ready autonomy cycle ledger closure fresh contract validator accepted tampered source: {tampered_closure_fresh_contract}")
        tampered_closure_fresh_contract = [dict(row) for row in closure_fresh_contract]
        tampered_closure_fresh_contract[-1]["fresh_required"] = True
        if _fresh_continuation_review_contract_ready(tampered_closure_fresh_contract):
            raise SystemExit(f"Ready autonomy cycle ledger closure fresh contract validator accepted tampered fresh-required flag: {tampered_closure_fresh_contract}")
        for key in [
            "closure_next_continuation_requires_fresh_operator_timebox",
            "closure_next_continuation_requires_fresh_checkpoint",
            "closure_next_continuation_requires_fresh_recovery_cockpit",
            "closure_next_continuation_requires_fresh_local_safe_step",
            "closure_next_continuation_requires_fresh_review_token",
        ]:
            if ledger_metadata.get(key) is not True:
                raise SystemExit(f"Ready autonomy cycle ledger missed {key}: {ledger_metadata}")
            tampered_fresh_requirement_metadata = json.loads(json.dumps(ledger_metadata))
            tampered_fresh_requirement_metadata[key] = "false"
            if _metadata_flag_ready(tampered_fresh_requirement_metadata, key):
                raise SystemExit(
                    f"Exact metadata flag helper accepted string false for carried fresh requirement {key}: "
                    f"{tampered_fresh_requirement_metadata}"
                )
            if _autonomy_cycle_ledger_ready_from_metadata(tampered_fresh_requirement_metadata):
                raise SystemExit(
                    f"Ready autonomy cycle ledger accepted malformed carried fresh requirement {key}: "
                    f"{tampered_fresh_requirement_metadata}"
                )
        if ledger_metadata.get("closure_prior_step_authorizes_followup") is not False:
            raise SystemExit(f"Ready autonomy cycle ledger should not carry authorizing closure proof: {ledger_metadata}")
        if ledger_metadata.get("closure_prior_step_reusable_for_next_step") is not False:
            raise SystemExit(f"Ready autonomy cycle ledger should not carry reusable closure proof: {ledger_metadata}")
        for key in [
            "closure_prior_step_authorizes_followup",
            "closure_prior_step_reusable_for_next_step",
        ]:
            if not _metadata_flag_disabled(ledger_metadata, key):
                raise SystemExit(
                    f"Ready autonomy cycle ledger should carry exact False for {key}: {ledger_metadata}"
                )
            tampered_prior_step_metadata = json.loads(json.dumps(ledger_metadata))
            tampered_prior_step_metadata[key] = "false"
            if _metadata_flag_disabled(tampered_prior_step_metadata, key):
                raise SystemExit(
                    f"Exact metadata disabled helper accepted string false for carried prior-step closure {key}: "
                    f"{tampered_prior_step_metadata}"
                )
            if _autonomy_cycle_ledger_ready_from_metadata(tampered_prior_step_metadata):
                raise SystemExit(
                    f"Ready autonomy cycle ledger accepted malformed carried prior-step closure {key}: "
                    f"{tampered_prior_step_metadata}"
                )
        if ledger_metadata.get("next_review_requires_full_preflight") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed full-preflight requirement: {ledger_metadata}")
        if ledger_metadata.get("checkpoint_path_matches_latest") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed checkpoint binding proof: {ledger_metadata}")
        if ledger_metadata.get("checkpoint_hash_matches_latest") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed checkpoint hash binding proof: {ledger_metadata}")
        if ledger_metadata.get("recovery_artifact_hashes_present") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed recovery artifact hash binding: {ledger_metadata}")
        if ledger_metadata.get("receipt_hash_matches_file") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed recovery receipt file binding: {ledger_metadata}")
        if ledger_metadata.get("recovery_checkpoint_hash_matches_file") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed recovery checkpoint file binding: {ledger_metadata}")
        if ledger_metadata.get("receipt_sha256") != recovery_receipt_sha256 or ledger_metadata.get("checkpoint_sha256") != recovery_checkpoint_sha256:
            raise SystemExit(f"Ready autonomy cycle ledger hash values diverged: {ledger_metadata}")
        assert_sha256(ledger_metadata.get("recovery_followthrough_token_sha256"), "Ready autonomy cycle ledger recovery token")
        if ledger_metadata.get("recovery_followthrough_token_sha256") != execution_metadata.get("recovery_followthrough_token_sha256"):
            raise SystemExit(f"Ready autonomy cycle ledger recovery token diverged from continuation: {ledger_metadata}")
        if ledger_metadata.get("previous_recovery_followthrough_token_reusable_for_next_review") is not False:
            raise SystemExit(f"Ready autonomy cycle ledger should not reuse recovery token: {ledger_metadata}")
        for carried_recovery_mirror in [
            "previous_recovery_followthrough_token_reusable_for_next_review",
            "previous_local_safe_recovery_execution_token_reusable_for_next_review",
        ]:
            missing_carried_recovery_mirror = json.loads(json.dumps(ledger_metadata))
            missing_carried_recovery_mirror.pop(carried_recovery_mirror, None)
            if _autonomy_cycle_ledger_ready_from_metadata(missing_carried_recovery_mirror):
                raise SystemExit(
                    "Ready autonomy cycle ledger accepted missing carried recovery non-reuse mirror "
                    f"{carried_recovery_mirror}: {missing_carried_recovery_mirror}"
                )
        assert_recovery_followthrough_token_boundary(
            ledger_metadata,
            "Ready autonomy cycle ledger",
            expected_source="checkpoint_recovery_followthrough",
        )
        assert_local_safe_recovery_execution_token(
            ledger_metadata,
            "Ready autonomy cycle ledger",
            expected_source="autonomy_cycle_ledger",
        )
        if ledger_metadata.get("local_safe_recovery_execution_token_sha256") != execution_metadata.get("local_safe_recovery_execution_token_sha256"):
            raise SystemExit(f"Ready autonomy cycle ledger local-safe recovery token diverged from continuation: {ledger_metadata}")
        if ledger_metadata.get("previous_local_safe_recovery_execution_token_reusable_for_next_review") is not False:
            raise SystemExit(f"Ready autonomy cycle ledger should not reuse local-safe recovery execution token: {ledger_metadata}")
        assert_carried_recovery_execution_readiness_token(
            ledger_metadata,
            "Ready autonomy cycle ledger",
            expected_source="checkpoint_recovery_followthrough",
        )
        assert_one_step_execution_contract_token(ledger_metadata, "Ready autonomy cycle ledger", carried=True)
        if ledger_metadata.get("one_step_execution_contract_token_sha256") != execution_metadata.get("one_step_execution_contract_token_sha256"):
            raise SystemExit(f"Ready autonomy cycle ledger one-step contract token diverged from continuation: {ledger_metadata}")
        if ledger_metadata.get("next_recovery_followthrough_requires_new_token") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed recovery token boundary: {ledger_metadata}")
        if ledger_metadata.get("post_step_artifact_hashes_present") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed post-step artifact hash binding: {ledger_metadata}")
        if ledger_metadata.get("post_step_artifact_hashes_match_files") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed post-step artifact file binding: {ledger_metadata}")
        if ledger_metadata.get("post_step_receipt_sha256") != post_step_receipt_sha256 or ledger_metadata.get("post_step_checkpoint_sha256") != post_step_checkpoint_sha256:
            raise SystemExit(f"Ready autonomy cycle ledger post-step hash values diverged: {ledger_metadata}")
        if ledger_metadata.get("post_step_receipt_file_sha256") != post_step_receipt_sha256:
            raise SystemExit(f"Ready autonomy cycle ledger missed actual receipt file hash: {ledger_metadata}")
        if ledger_metadata.get("post_step_receipt_hash_matches_file") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed post-step receipt file binding: {ledger_metadata}")
        if ledger_metadata.get("post_step_checkpoint_file_sha256") != post_step_checkpoint_sha256:
            raise SystemExit(f"Ready autonomy cycle ledger missed actual checkpoint file hash: {ledger_metadata}")
        if ledger_metadata.get("post_step_checkpoint_hash_matches_file") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed post-step checkpoint file binding: {ledger_metadata}")
        if ledger_metadata.get("completed_step_matches_proposed") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed step identity binding: {ledger_metadata}")
        if ledger_metadata.get("proposed_next_step_sha256") != ledger_metadata.get("completed_step_sha256"):
            raise SystemExit(f"Ready autonomy cycle ledger step hashes diverged: {ledger_metadata}")
        if ledger_metadata.get("post_step_verification_matches_proposed") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed verification target binding: {ledger_metadata}")
        if ledger_metadata.get("proposed_verification_sha256") != ledger_metadata.get("post_step_verification_sha256"):
            raise SystemExit(f"Ready autonomy cycle ledger verification hashes diverged: {ledger_metadata}")
        if ledger_metadata.get("previous_continuation_permission_reusable_for_next_review") is not False or ledger_metadata.get("previous_post_step_receipt_reusable_for_next_review") is not False:
            raise SystemExit(f"Ready autonomy cycle ledger should not reuse prior permissions: {ledger_metadata}")
        if ledger_metadata.get("previous_post_step_receipt_hash_reusable_for_next_review") is not False or ledger_metadata.get("previous_post_step_checkpoint_hash_reusable_for_next_review") is not False:
            raise SystemExit(f"Ready autonomy cycle ledger should not reuse prior post-step hashes: {ledger_metadata}")
        if ledger_metadata.get("continuation_review_token_sha256") != execution_metadata.get("continuation_review_token_sha256"):
            raise SystemExit(f"Ready autonomy cycle ledger review token diverged: {ledger_metadata}")
        if ledger_metadata.get("previous_continuation_review_token_reusable_for_next_review") is not False:
            raise SystemExit(f"Ready autonomy cycle ledger should not reuse prior review token: {ledger_metadata}")
        if ledger_metadata.get("next_review_requires_new_continuation_review_token") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed next-review token boundary: {ledger_metadata}")
        if not _continuation_review_token_metadata_ready(
            ledger_metadata,
            reusable_key="previous_continuation_review_token_reusable_for_next_review",
        ):
            raise SystemExit(f"Ready autonomy cycle ledger review token failed mirror validator: {ledger_metadata}")
        for missing_key in [
            "continuation_review_token_reusable_for_next_review",
            "previous_continuation_review_token_reusable_for_next_review",
        ]:
            missing_metadata = dict(ledger_metadata)
            missing_metadata.pop(missing_key, None)
            if _continuation_review_token_metadata_ready(
                missing_metadata,
                reusable_key="previous_continuation_review_token_reusable_for_next_review",
            ):
                raise SystemExit(
                    f"Ready autonomy cycle ledger review token validator accepted missing mirror {missing_key}: "
                    f"{missing_metadata}"
                )
        assert_sha256(ledger_metadata.get("autonomy_cycle_ledger_token_sha256"), "Ready autonomy cycle ledger token")
        if ledger_metadata.get("autonomy_cycle_ledger_token_present") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed token presence: {ledger_metadata}")
        assert_autonomy_cycle_ledger_token_boundary(
            ledger_metadata,
            "Ready autonomy cycle ledger",
            expected_source="autonomy_cycle_ledger",
        )
        if ledger_metadata.get("previous_cycle_ledger_token_reusable_for_next_review") is not False:
            raise SystemExit(f"Ready autonomy cycle ledger should not reuse cycle token: {ledger_metadata}")
        if ledger_metadata.get("next_review_requires_new_cycle_ledger_token") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed next cycle-token boundary: {ledger_metadata}")
        assert_sha256(ledger_metadata.get("fresh_review_boundary_token_sha256"), "Ready autonomy fresh-review boundary token")
        if ledger_metadata.get("fresh_review_boundary_token_present") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed fresh-review boundary token presence: {ledger_metadata}")
        fresh_review_boundary_rows = ledger_metadata.get("fresh_review_boundary_token_rows") or []
        if ledger_metadata.get("fresh_review_boundary_token_row_count") != 3 or len(fresh_review_boundary_rows) != 3:
            raise SystemExit(f"Ready autonomy cycle ledger missed fresh-review boundary rows: {ledger_metadata}")
        if ledger_metadata.get("fresh_review_boundary_token_ready") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed fresh-review boundary readiness: {ledger_metadata}")
        expected_fresh_review_boundary_rows = {
            "fresh_review_boundary_token": "present",
            "next_continuation_preflight": "fresh_operator_timebox_checkpoint_and_review_required",
            "prior_cycle_reuse_boundary": "prior_cycle_proof_only_not_reusable",
        }
        if {row.get("item") for row in fresh_review_boundary_rows} != set(expected_fresh_review_boundary_rows):
            raise SystemExit(f"Ready autonomy cycle ledger fresh-review boundary items diverged: {ledger_metadata}")
        for row in fresh_review_boundary_rows:
            item = row.get("item")
            if row.get("status") != expected_fresh_review_boundary_rows.get(item):
                raise SystemExit(f"Ready autonomy cycle ledger fresh-review boundary status diverged: {row}")
            if row.get("source") != "autonomy_cycle_ledger":
                raise SystemExit(f"Ready autonomy cycle ledger fresh-review boundary source diverged: {row}")
            if row.get("token_sha256") != ledger_metadata.get("fresh_review_boundary_token_sha256"):
                raise SystemExit(f"Ready autonomy cycle ledger fresh-review boundary token hash diverged: {row}")
            if (
                row.get("fresh_required") is not True
                or row.get("authorizes_action_now") is not False
                or row.get("authorizes_local_safe_step") is not False
                or row.get("authorizes_risky_work") is not False
                or row.get("authorizes_unreviewed_followthrough") is not False
                or row.get("authorizes_timebox_reuse") is not False
                or row.get("authorizes_checkpoint_reuse") is not False
                or row.get("authorizes_token_reuse") is not False
                or row.get("authorizes_model_call") is not False
                or row.get("authorizes_tool_execution") is not False
                or row.get("authorizes_personal_data_read") is not False
                or row.get("authorizes_external_side_effect") is not False
                or row.get("reusable_for_next_review") is not False
                or row.get("reusable_for_next_cycle") is not False
            ):
                raise SystemExit(f"Ready autonomy cycle ledger fresh-review boundary row should stay non-authorizing: {row}")
        for key in [
            "fresh_review_boundary_token_authorizes_action_now",
            "fresh_review_boundary_token_authorizes_local_safe_step",
            "fresh_review_boundary_token_authorizes_risky_work",
            "fresh_review_boundary_token_authorizes_unreviewed_followthrough",
            "fresh_review_boundary_token_authorizes_timebox_reuse",
            "fresh_review_boundary_token_authorizes_checkpoint_reuse",
            "fresh_review_boundary_token_authorizes_token_reuse",
            "fresh_review_boundary_token_authorizes_model_call",
            "fresh_review_boundary_token_authorizes_tool_execution",
            "fresh_review_boundary_token_authorizes_personal_data_read",
            "fresh_review_boundary_token_authorizes_external_side_effect",
            "fresh_review_boundary_token_reusable_for_next_review",
            "fresh_review_boundary_token_reusable_for_next_cycle",
        ]:
            if ledger_metadata.get(key) is not False:
                raise SystemExit(f"Ready autonomy fresh-review boundary token should report {key}=False: {ledger_metadata}")
        if ledger_metadata.get("next_review_requires_new_fresh_review_boundary_token") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed next fresh-review boundary token requirement: {ledger_metadata}")
        assert_fresh_review_hash_binds_authority(
            ledger_metadata,
            "Ready autonomy cycle ledger",
        )
        assert_next_review_start_command_boundary(
            ledger_metadata,
            "Ready autonomy cycle ledger",
            expected_source="autonomy_cycle_ledger",
        )
        if ledger_metadata.get("prior_continuation_step_proof_only") is not True:
            raise SystemExit(f"Ready autonomy cycle ledger missed prior-proof boundary: {ledger_metadata}")
        assert_prior_cycle_ledger_token_boundary(
            ledger_metadata,
            "Ready autonomy cycle ledger",
            expected_source="autonomy_cycle_ledger",
        )
        if not isinstance(ledger_metadata.get("step_closure_metadata"), dict):
            raise SystemExit(f"Ready autonomy cycle ledger missed nested closure metadata: {ledger_metadata}")
        for key in READ_ONLY_FALSE_FLAGS:
            if ledger_metadata.get(key) is not False:
                raise SystemExit(f"Ready autonomy cycle ledger should report {key}=False.")
        for key in ["action_allowed_now", "executable_tool_action_emitted", "can_emit_executable_tool_action", "can_auto_execute_now"]:
            if ledger_metadata.get(key) is not False:
                raise SystemExit(f"Ready autonomy cycle ledger should report {key}=False.")

        bool_priority = runtime.registry.get("priority_stack").handler({"limit": True})
        if not bool_priority.ok or bool_priority.metadata.get("limit") != 8:
            raise SystemExit(f"priority_stack should treat boolean limits as malformed defaults: {bool_priority.metadata}")
        for key in READ_ONLY_FALSE_FLAGS:
            if bool_priority.metadata.get(key) is not False:
                raise SystemExit(f"priority_stack boolean limit should report {key}=False.")


if __name__ == "__main__":
    main()
