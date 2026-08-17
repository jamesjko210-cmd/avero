from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.automations.compaction import COMPACTION_JOB_NAME, COMPACTION_JOB_TYPE
from jarvis_v2.scripts.test_runtime import make_temp_runtime
import jarvis_v2.tools.focus as focus_module
from jarvis_v2.tools.focus import MAX_OBJECTIVE_CHARS, _metadata_bool, make_focus_tools


def test_planner_routes_focus_brief_aliases() -> None:
    # Real gaps found live 2026-07-09: "give me a focus brief" / "show my
    # focus brief" fell through to chat while bare "focus brief" worked, and
    # "what should i focus on" (without a trailing "now") fell through while
    # "what should i focus on now" worked.
    p = RuleBasedPlanner()
    for q in ("focus brief", "give me a focus brief", "show my focus brief", "what should i focus on", "what should i focus on now"):
        actions = p.plan(q).actions
        if [a.tool_name for a in actions] != ["focus_brief"]:
            raise SystemExit(f"focus_brief route missed: {q!r} -> {[a.tool_name for a in actions]}")


READ_ONLY_FLAGS = [
    "calls_model",
    "executes_tools",
    "reads_private_data",
    "reads_personal_data",
    "writes_files",
    "writes_database",
    "writes_memory",
    "writes_notes",
    "external_side_effect",
    "queues_approval",
    "requires_approval",
    "controls_computer",
    "speaks",
    "completes_tasks",
    "authorizes_execution",
    "authorizes_completion_claim",
    "approval_granted",
]
BACKGROUND_RHYTHM_KEYS = [
    "background_ready",
    "background_next_command",
    "background_issue",
    "background_priority",
    "state_snapshot_jobs",
    "enabled_state_snapshot_jobs",
    "disabled_state_snapshot_jobs",
    "conversation_compaction_jobs",
    "enabled_conversation_compaction_jobs",
    "disabled_conversation_compaction_jobs",
    "unreadable_scheduled_job_rows",
]


def assert_focus_metadata_bool_is_exact() -> None:
    if _metadata_bool(True) is not True:
        raise SystemExit("focus exact metadata bool rejected True")
    if _metadata_bool(False) is not False:
        raise SystemExit("focus exact metadata bool rejected False")
    for value in ("true", "false", "yes", "no", 1, 0, [True], {"ready": True}, None):
        if _metadata_bool(value) is not False:
            raise SystemExit(f"focus exact metadata bool accepted malformed truthy value: {value!r}")
    if _metadata_bool("false", default=True) is not True:
        raise SystemExit("focus exact metadata bool did not preserve explicit default")


def assert_execution_health_handoff(metadata: dict, label: str) -> None:
    for key in [
        "execution_health_review_required",
        "execution_health_next_command",
        "execution_health_next_commands",
        "execution_health_next_command_count",
        "execution_health_proof_queue",
        "execution_health_proof_queue_count",
        "execution_health_next_required_command",
        "execution_health_next_proof_command",
        "execution_health_recent_tool_run_rows",
        "execution_health_readable_recent_tool_runs",
        "execution_health_unreadable_recent_tool_run_rows",
        "failed_action_runs",
        "approval_held_action_runs",
        "execution_health_failed_action_runs",
        "execution_health_approval_held_action_runs",
        "approval_held_review_required",
        "approval_held_review_commands",
        "approval_held_review_command_count",
        "approval_held_review_next_command",
        "execution_health_approval_held_review_required",
        "execution_health_approval_held_review_commands",
        "execution_health_approval_held_review_command_count",
        "execution_health_approval_held_review_next_command",
        "execution_health_blocker_categories",
        "execution_health_blocker_count",
        "execution_health_verification_coverage",
        "execution_health_approval_proof_chains",
        "execution_health_approval_proof_chain_count",
        "execution_health_repeated_failure_tools",
        "execution_health_repeated_failure_count",
        "execution_health_failure_promotion_queue",
        "execution_health_failure_promotion_queue_count",
        "execution_health_failure_promotion_command",
        "execution_health_failure_implementation_command",
        "execution_health_failure_apply_contract_command",
        "execution_health_learning_target_run_id",
        "execution_health_learning_target_tool",
        "execution_health_verification_handoff_command",
        "execution_health_recovery_handoff_command",
        "execution_health_learning_handoff_command",
        "execution_health_target_verification_receipts",
        "execution_health_target_recovery_packets",
        "execution_health_target_after_action_learning_packets",
        "execution_health_recovery_closure_state",
        "execution_health_recovery_closure_missing",
        "execution_health_recovery_closure_missing_count",
        "execution_health_recovery_closure_required_commands",
        "execution_health_recovery_closure_next_required_command",
        "execution_health_recovery_closure_proof_queue",
        "execution_health_recovery_closure_proof_queue_count",
        "execution_health_recovery_closure_next_proof_command",
        "execution_health_recovery_closure_ready_to_retry",
        "execution_health_recovery_closure_blocks_auto_execution",
        "execution_health_recovery_closure_checklist_command",
        "execution_health_recovery_closure_should_open_checklist",
        "execution_health_operator_timeboxes_override_priority",
        "execution_health_stop_times_override_priority",
    ]:
        if key not in metadata:
            raise SystemExit(f"{label} missed execution-health metadata key {key}: {metadata}")
    if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
        raise SystemExit(f"{label} missed top-level operator-limit metadata: {metadata}")
    if metadata.get("execution_health_operator_timeboxes_override_priority") is not True or metadata.get("execution_health_stop_times_override_priority") is not True:
        raise SystemExit(f"{label} missed execution-health operator-limit metadata: {metadata}")
    if metadata["execution_health_next_command"] not in metadata["execution_health_next_commands"]:
        raise SystemExit(f"{label} missed primary execution-health command in queue: {metadata}")
    if metadata["execution_health_next_command_count"] != len(metadata["execution_health_next_commands"]):
        raise SystemExit(f"{label} execution-health queue count diverged: {metadata}")
    if metadata["execution_health_proof_queue"] != metadata["execution_health_next_commands"]:
        raise SystemExit(f"{label} execution-health proof queue should mirror next commands: {metadata}")
    if metadata["execution_health_proof_queue_count"] != len(metadata["execution_health_proof_queue"]):
        raise SystemExit(f"{label} execution-health proof queue count diverged: {metadata}")
    if metadata["execution_health_next_proof_command"] != metadata["execution_health_next_command"]:
        raise SystemExit(f"{label} execution-health next proof command diverged: {metadata}")
    if metadata["execution_health_next_required_command"] != metadata["execution_health_next_command"]:
        raise SystemExit(f"{label} execution-health next required command diverged: {metadata}")
    if metadata["execution_health_recent_tool_run_rows"] != (
        metadata["execution_health_readable_recent_tool_runs"] + metadata["execution_health_unreadable_recent_tool_run_rows"]
    ):
        raise SystemExit(f"{label} execution-health readable/unreadable row count diverged: {metadata}")
    if metadata["execution_health_blocker_count"] != len(metadata["execution_health_blocker_categories"]):
        raise SystemExit(f"{label} execution-health blocker count diverged: {metadata}")
    if metadata["approval_held_review_required"] != bool(metadata["approval_held_action_runs"]):
        raise SystemExit(f"{label} approval-held review flag diverged from approval-held count: {metadata}")
    if metadata["execution_health_approval_held_review_required"] != metadata["approval_held_review_required"]:
        raise SystemExit(f"{label} execution-health approval-held review flag diverged: {metadata}")
    if metadata["approval_held_review_commands"] != metadata["execution_health_approval_held_review_commands"]:
        raise SystemExit(f"{label} approval-held review command aliases diverged: {metadata}")
    if metadata["approval_held_review_command_count"] != len(metadata["approval_held_review_commands"]):
        raise SystemExit(f"{label} approval-held review command count diverged: {metadata}")
    if metadata["execution_health_approval_held_review_command_count"] != len(metadata["execution_health_approval_held_review_commands"]):
        raise SystemExit(f"{label} execution-health approval-held review command count diverged: {metadata}")
    if metadata["approval_held_review_commands"]:
        if metadata["approval_held_review_next_command"] != metadata["approval_held_review_commands"][0]:
            raise SystemExit(f"{label} approval-held review next command diverged: {metadata}")
        if metadata["execution_health_approval_held_review_next_command"] != metadata["approval_held_review_next_command"]:
            raise SystemExit(f"{label} execution-health approval-held next command diverged: {metadata}")
        for command in metadata["approval_held_review_commands"]:
            if command not in metadata["execution_health_next_commands"]:
                raise SystemExit(f"{label} missed approval-held review command in health queue: {metadata}")
    if metadata["execution_health_verification_coverage"] not in {"present", "missing"}:
        raise SystemExit(f"{label} missed verification coverage state: {metadata}")
    if not isinstance(metadata["execution_health_approval_proof_chains"], dict):
        raise SystemExit(f"{label} approval proof chains should be structured: {metadata}")
    if metadata["execution_health_repeated_failure_count"] != len(metadata["execution_health_repeated_failure_tools"]):
        raise SystemExit(f"{label} repeated failure count diverged: {metadata}")
    if metadata["execution_health_recovery_closure_missing_count"] != len(metadata["execution_health_recovery_closure_missing"]):
        raise SystemExit(f"{label} recovery closure missing count diverged: {metadata}")
    closure_required_commands = metadata["execution_health_recovery_closure_required_commands"]
    if closure_required_commands:
        if metadata.get("execution_health_recovery_closure_ready_to_retry") is not False:
            raise SystemExit(f"{label} should not be ready to retry while seeded closure blockers remain: {metadata}")
        if metadata.get("execution_health_recovery_closure_blocks_auto_execution") is not True:
            raise SystemExit(f"{label} should block auto-run while seeded recovery closure is incomplete: {metadata}")
        if metadata.get("execution_health_recovery_closure_checklist_command") != "recovery closure checklist":
            raise SystemExit(f"{label} missed recovery closure checklist command: {metadata}")
        if metadata.get("execution_health_recovery_closure_should_open_checklist") is not True:
            raise SystemExit(f"{label} should recommend opening recovery closure checklist: {metadata}")
    elif metadata.get("execution_health_recovery_closure_state") == "not_needed":
        if metadata.get("execution_health_recovery_closure_blocks_auto_execution") is not False:
            raise SystemExit(f"{label} should not block auto-run when no recovery closure is needed: {metadata}")
        if metadata.get("execution_health_recovery_closure_checklist_command") != "":
            raise SystemExit(f"{label} should not suggest recovery checklist when no recovery closure is needed: {metadata}")
        if metadata.get("execution_health_recovery_closure_should_open_checklist") is not False:
            raise SystemExit(f"{label} should not open recovery checklist when no recovery closure is needed: {metadata}")
    if closure_required_commands and metadata["execution_health_next_command"] != closure_required_commands[0]:
        raise SystemExit(f"{label} should route next command to the first recovery-closure proof: {metadata}")
    if closure_required_commands and metadata["execution_health_recovery_closure_next_required_command"] != closure_required_commands[0]:
        raise SystemExit(f"{label} missed explicit first recovery-closure proof command: {metadata}")
    if metadata["execution_health_recovery_closure_proof_queue"] != closure_required_commands:
        raise SystemExit(f"{label} recovery-closure proof queue should mirror required commands: {metadata}")
    if metadata["execution_health_recovery_closure_proof_queue_count"] != len(closure_required_commands):
        raise SystemExit(f"{label} recovery-closure proof queue count diverged: {metadata}")
    if closure_required_commands and metadata["execution_health_recovery_closure_next_proof_command"] != closure_required_commands[0]:
        raise SystemExit(f"{label} missed next recovery-closure proof command: {metadata}")
    if closure_required_commands and metadata["execution_health_next_commands"][: len(closure_required_commands)] != closure_required_commands:
        raise SystemExit(f"{label} should place recovery-closure proofs first in the queue: {metadata}")
    if not closure_required_commands and metadata["approval_held_review_commands"] and metadata["failed_action_runs"] == 0:
        if metadata["execution_health_next_command"] != metadata["approval_held_review_commands"][0]:
            raise SystemExit(f"{label} should route approval-held-only health to the first approval proof: {metadata}")
    for command in closure_required_commands:
        if command and command not in metadata["execution_health_next_commands"]:
            raise SystemExit(f"{label} missed recovery closure command in queue: {command} in {metadata}")
    queue = metadata["execution_health_failure_promotion_queue"]
    if metadata["execution_health_failure_promotion_queue_count"] != len(queue):
        raise SystemExit(f"{label} failure promotion queue count diverged: {metadata}")
    if metadata["execution_health_repeated_failure_count"] > 0:
        for key in [
            "execution_health_failure_promotion_command",
            "execution_health_failure_implementation_command",
            "execution_health_failure_apply_contract_command",
        ]:
            command = metadata.get(key)
            if not command or command not in queue or command not in metadata["execution_health_next_commands"]:
                raise SystemExit(f"{label} missed failure promotion command {key}: {metadata}")


def assert_approval_handoff(metadata: dict, label: str) -> None:
    expected = {
        "approval_handoff_pending_count": 1,
        "approval_handoff_first_id": 1,
        "approval_handoff_first_tool": "run_shell_command",
        "approval_handoff_readiness_command": "approval readiness 1",
        "approval_handoff_last_look_command": "approval packet 1",
        "approval_handoff_proof_command": "approval chain proof 1",
        "approval_handoff_verification_command": "verification receipt <approved run id from approval chain proof 1>",
        "approval_handoff_approve_command": "approve approval 1",
        "approval_handoff_dismiss_command": "dismiss approval 1",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise SystemExit(f"{label} missed approval handoff metadata {key}: {metadata}")
    expected_chain = [
        "approval readiness 1",
        "approval packet 1",
        "approval chain proof 1",
        "verification receipt <approved run id from approval chain proof 1>",
    ]
    if metadata.get("approval_handoff_proof_chain_commands") != expected_chain:
        raise SystemExit(f"{label} missed ordered proof chain commands: {metadata}")


def assert_agi_focus_handoff(metadata: dict, label: str) -> None:
    for key in [
        "agi_next_gate",
        "agi_next_target_title",
        "agi_next_build_command",
        "agi_next_evidence_closure_commands",
        "agi_next_evidence_closure_command_count",
        "agi_next_focused_verification_commands",
        "agi_next_focused_verification_command_count",
        "agi_next_likely_files",
        "agi_next_likely_file_count",
        "agi_next_target_file_integrity_status",
        "agi_next_target_files_checked",
        "agi_next_target_files_exist",
        "agi_next_missing_target_files",
        "agi_next_missing_target_file_count",
        "agi_next_target_integrity_blocks_start",
        "agi_next_acceptance_checks",
        "agi_next_acceptance_check_count",
        "agi_next_build_packet_ready_for_review",
        "agi_focus_review_only",
        "agi_focus_draft_only",
        "agi_focus_loads_without_execution",
        "agi_focus_authorizes_execution",
        "agi_focus_authorizes_completion_claim",
        "agi_focus_approval_granted",
        "agi_focus_handoff",
    ]:
        if key not in metadata:
            raise SystemExit(f"{label} missed AGI focus handoff key {key}: {metadata}")
    handoff = metadata.get("agi_focus_handoff") or {}
    if handoff.get("source") != "agi_focus":
        raise SystemExit(f"{label} missed nested AGI focus handoff source: {metadata}")
    parity = {
        "selected_gate": "agi_next_gate",
        "target_title": "agi_next_target_title",
        "build_command": "agi_next_build_command",
        "evidence_closure_commands": "agi_next_evidence_closure_commands",
        "evidence_closure_command_count": "agi_next_evidence_closure_command_count",
        "focused_verification_commands": "agi_next_focused_verification_commands",
        "focused_verification_command_count": "agi_next_focused_verification_command_count",
        "likely_files": "agi_next_likely_files",
        "likely_file_count": "agi_next_likely_file_count",
        "target_file_integrity_status": "agi_next_target_file_integrity_status",
        "target_files_checked": "agi_next_target_files_checked",
        "target_files_exist": "agi_next_target_files_exist",
        "missing_target_files": "agi_next_missing_target_files",
        "missing_target_file_count": "agi_next_missing_target_file_count",
        "target_integrity_blocks_start": "agi_next_target_integrity_blocks_start",
        "acceptance_checks": "agi_next_acceptance_checks",
        "acceptance_check_count": "agi_next_acceptance_check_count",
        "build_packet_ready_for_review": "agi_next_build_packet_ready_for_review",
    }
    for nested_key, flat_key in parity.items():
        if handoff.get(nested_key) != metadata.get(flat_key):
            raise SystemExit(f"{label} AGI focus handoff field {nested_key} diverged from {flat_key}: {metadata}")
    if metadata.get("agi_focus_review_only") is not True or metadata.get("agi_focus_draft_only") is not True:
        raise SystemExit(f"{label} missed AGI focus review-only flags: {metadata}")
    if metadata.get("agi_focus_loads_without_execution") is not True:
        raise SystemExit(f"{label} missed AGI focus load-only flag: {metadata}")
    if metadata.get("agi_focus_authorizes_execution") is not False or metadata.get("agi_focus_authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} AGI focus metadata should not authorize execution or completion claims: {metadata}")
    if metadata.get("agi_focus_approval_granted") is not False:
        raise SystemExit(f"{label} AGI focus metadata should not grant approval: {metadata}")
    if metadata.get("agi_focus_selection_source") != "operator_focus_handoff":
        raise SystemExit(f"{label} missed explicit AGI focus selection source: {metadata}")
    if metadata.get("agi_focus_canonical_selector_command") != "agi gates":
        raise SystemExit(f"{label} missed canonical AGI selector command: {metadata}")
    if metadata.get("agi_focus_deliberate_focus_override") is not True:
        raise SystemExit(f"{label} missed deliberate AGI focus override flag: {metadata}")
    if "operator-facing build target" not in str(metadata.get("agi_focus_selection_reason") or ""):
        raise SystemExit(f"{label} missed AGI focus selection reason: {metadata}")
    for flat_key, nested_key in [
        ("agi_focus_selection_source", "selection_source"),
        ("agi_focus_selection_reason", "selection_reason"),
        ("agi_focus_canonical_selector_command", "canonical_selector_command"),
        ("agi_focus_deliberate_focus_override", "deliberate_focus_override"),
    ]:
        if metadata.get(flat_key) != handoff.get(nested_key):
            raise SystemExit(f"{label} AGI focus selection field {nested_key} diverged from {flat_key}: {metadata}")
    for key in [
        "review_only",
        "draft_only",
        "loads_without_execution",
    ]:
        if handoff.get(key) is not True:
            raise SystemExit(f"{label} AGI focus nested handoff missed true {key}: {metadata}")
    for key in [
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "calls_model",
        "executes_tools",
        "writes_files",
        "reads_personal_data",
        "external_side_effect",
        "controls_computer",
        "queues_approval",
    ]:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} AGI focus nested handoff should keep {key}=False: {metadata}")
    if metadata.get("agi_next_gate") != "personal integrations":
        raise SystemExit(f"{label} missed selected AGI gate: {metadata}")
    if not metadata.get("agi_next_target_title"):
        raise SystemExit(f"{label} missed selected AGI target title: {metadata}")
    if metadata.get("agi_next_build_command") != "agi next build move: personal integrations":
        raise SystemExit(f"{label} missed selected AGI build command: {metadata}")
    if metadata.get("agi_next_likely_file_count") != len(metadata.get("agi_next_likely_files") or []):
        raise SystemExit(f"{label} AGI likely-file count diverged: {metadata}")
    if metadata.get("agi_next_evidence_closure_command_count") != len(metadata.get("agi_next_evidence_closure_commands") or []):
        raise SystemExit(f"{label} AGI evidence-closure count diverged: {metadata}")
    if metadata.get("agi_next_focused_verification_command_count") != len(metadata.get("agi_next_focused_verification_commands") or []):
        raise SystemExit(f"{label} AGI verification count diverged: {metadata}")
    if metadata.get("agi_next_acceptance_check_count") != len(metadata.get("agi_next_acceptance_checks") or []):
        raise SystemExit(f"{label} AGI acceptance count diverged: {metadata}")
    if bool(metadata.get("agi_next_target_integrity_blocks_start")) is metadata.get("agi_next_target_files_exist"):
        raise SystemExit(f"{label} AGI target-integrity blocker diverged: {metadata}")
    expected_ready = bool(
        metadata.get("agi_next_target_files_exist")
        and metadata.get("agi_next_focused_verification_commands")
        and metadata.get("agi_next_acceptance_checks")
    )
    if metadata.get("agi_next_build_packet_ready_for_review") is not expected_ready:
        raise SystemExit(f"{label} AGI ready-for-review flag diverged: {metadata}")


def assert_focus_brief_handoff(metadata: dict, label: str, *, objective: str | None = None) -> None:
    handoff = metadata.get("focus_brief_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed focus_brief_handoff: {metadata}")
    if metadata.get("focus_brief_handoff_ready") is not True or handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} should report handoff ready: {metadata}")
    if metadata.get("focus_brief_ready_for_operator") is not True or handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} should report ready_for_operator: {metadata}")
    if metadata.get("focus_brief_state_changed") is not False or handoff.get("state_changed") is not False:
        raise SystemExit(f"{label} should report unchanged state: {metadata}")
    if metadata.get("focus_brief_changed") != [] or handoff.get("changed") != []:
        raise SystemExit(f"{label} should report no changed resources: {metadata}")
    if metadata.get("focus_brief_content_in_handoff") is not False or handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} should keep content out of handoff: {metadata}")
    if objective and handoff.get("objective") != objective:
        raise SystemExit(f"{label} objective diverged: {metadata}")
    for key in [
        "pending_approvals",
        "open_tasks",
        "active_goals",
        "active_decisions",
        "active_preferences",
        "enabled_jobs",
        "scheduled_jobs",
        "limit",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} {key} diverged: {metadata}")
    for key in BACKGROUND_RHYTHM_KEYS:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} background rhythm key {key} diverged: {metadata}")
    for flat_key, nested_key in [
        ("focus_brief_start_kind", "start_kind"),
        ("focus_brief_start_command", "start_command"),
        ("focus_brief_start_label", "start_label"),
        ("focus_brief_session_plan", "session_plan"),
        ("focus_brief_next_commands", "next_commands"),
        ("focus_brief_next_safe_commands", "next_safe_commands"),
    ]:
        if metadata.get(flat_key) != handoff.get(nested_key):
            raise SystemExit(f"{label} {flat_key} diverged from nested handoff: {metadata}")
    if metadata.get("focus_brief_next_command_count") != len(handoff.get("next_commands") or []):
        raise SystemExit(f"{label} next-command count diverged: {metadata}")
    if metadata.get("focus_brief_next_safe_command_count") != len(handoff.get("next_safe_commands") or []):
        raise SystemExit(f"{label} next-safe-command count diverged: {metadata}")
    if handoff.get("next_safe_commands") != handoff.get("next_commands"):
        raise SystemExit(f"{label} next-safe commands should mirror next commands: {metadata}")
    if handoff.get("approval_handoff_pending_count") != metadata.get("approval_handoff_pending_count"):
        raise SystemExit(f"{label} approval handoff count diverged: {metadata}")
    if handoff.get("approval_handoff_proof_chain_commands") != metadata.get("approval_handoff_proof_chain_commands"):
        raise SystemExit(f"{label} approval proof chain diverged: {metadata}")
    if handoff.get("execution_health_next_commands") != metadata.get("execution_health_next_commands"):
        raise SystemExit(f"{label} execution health queue diverged: {metadata}")
    for key in [
        "execution_health_approval_held_review_required",
        "execution_health_approval_held_review_commands",
        "execution_health_approval_held_review_command_count",
        "execution_health_approval_held_review_next_command",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} focus handoff {key} diverged: {metadata}")
    if handoff.get("execution_health_blocker_categories") != metadata.get("execution_health_blocker_categories"):
        raise SystemExit(f"{label} execution health blockers diverged: {metadata}")
    if handoff.get("agi_next_build_command") != metadata.get("agi_next_build_command"):
        raise SystemExit(f"{label} AGI build command diverged: {metadata}")
    if metadata.get("approval_handoff_pending_count", 0) > 0 and not any(
        str(command).startswith("approval readiness") for command in handoff.get("next_commands") or []
    ):
        raise SystemExit(f"{label} missed approval readiness next command: {metadata}")
    if "safe next actions" not in handoff.get("next_commands", []):
        raise SystemExit(f"{label} missed safe next actions next command: {metadata}")
    if handoff.get("start_kind") not in {"approval_review", "task", "goal", "capture_task", "background_repair"}:
        raise SystemExit(f"{label} start kind is not a known resume category: {metadata}")
    boundaries = handoff.get("boundaries") or {}
    if metadata.get("focus_brief_boundaries") != boundaries:
        raise SystemExit(f"{label} boundary packet diverged: {metadata}")
    if boundaries.get("read_only") is not True:
        raise SystemExit(f"{label} boundary should report read_only=True: {metadata}")
    for flat_key, boundary_key in [
        ("focus_brief_authorizes_execution", "authorizes_execution"),
        ("focus_brief_authorizes_completion_claim", "authorizes_completion_claim"),
        ("focus_brief_approval_granted", "approval_granted"),
    ]:
        if metadata.get(flat_key) is not False or boundaries.get(boundary_key) is not False:
            raise SystemExit(f"{label} missed flat no-authority parity for {flat_key}: {metadata}")
    for flag in READ_ONLY_FLAGS:
        if boundaries.get(flag) is not False:
            raise SystemExit(f"{label} boundary should report {flag}=False: {metadata}")
    handoff_text = json.dumps(handoff, sort_keys=True)
    for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
        if forbidden in handoff_text:
            raise SystemExit(f"{label} handoff leaked local path fragment {forbidden!r}: {metadata}")


def assert_work_session_packet_handoff(metadata: dict, label: str, *, objective: str | None = None) -> None:
    handoff = metadata.get("work_session_packet_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed work_session_packet_handoff: {metadata}")
    if metadata.get("work_session_packet_handoff_ready") is not True or handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} should report handoff ready: {metadata}")
    if metadata.get("work_session_packet_ready_for_operator") is not True or handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} should report ready_for_operator: {metadata}")
    if metadata.get("work_session_packet_state_changed") is not False or handoff.get("state_changed") is not False:
        raise SystemExit(f"{label} should report unchanged state: {metadata}")
    if metadata.get("work_session_packet_changed") != [] or handoff.get("changed") != []:
        raise SystemExit(f"{label} should report no changed resources: {metadata}")
    if metadata.get("work_session_packet_content_in_handoff") is not False or handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} should keep content out of handoff: {metadata}")
    if objective and handoff.get("objective") != objective:
        raise SystemExit(f"{label} objective diverged: {metadata}")
    for key in [
        "pending_approvals",
        "open_tasks",
        "active_goals",
        "active_decisions",
        "active_preferences",
        "enabled_jobs",
        "scheduled_jobs",
        "limit",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} {key} diverged: {metadata}")
    for key in BACKGROUND_RHYTHM_KEYS:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} background rhythm key {key} diverged: {metadata}")
    if metadata.get("work_session_packet_start_move") != handoff.get("start_move"):
        raise SystemExit(f"{label} start move diverged: {metadata}")
    if metadata.get("work_session_packet_stop_condition") != handoff.get("stop_condition"):
        raise SystemExit(f"{label} stop condition diverged: {metadata}")
    if metadata.get("work_session_packet_next_commands") != handoff.get("next_commands"):
        raise SystemExit(f"{label} next commands diverged: {metadata}")
    if metadata.get("work_session_packet_next_safe_commands") != handoff.get("next_safe_commands"):
        raise SystemExit(f"{label} next-safe commands diverged: {metadata}")
    if handoff.get("next_safe_commands") != handoff.get("next_commands"):
        raise SystemExit(f"{label} next-safe commands should mirror next commands: {metadata}")
    if metadata.get("work_session_packet_next_command_count") != len(handoff.get("next_commands") or []):
        raise SystemExit(f"{label} next-command count diverged: {metadata}")
    if metadata.get("work_session_packet_next_safe_command_count") != len(handoff.get("next_safe_commands") or []):
        raise SystemExit(f"{label} next-safe-command count diverged: {metadata}")
    if handoff.get("approval_handoff_pending_count") != metadata.get("approval_handoff_pending_count"):
        raise SystemExit(f"{label} approval handoff count diverged: {metadata}")
    if handoff.get("approval_handoff_proof_chain_commands") != metadata.get("approval_handoff_proof_chain_commands"):
        raise SystemExit(f"{label} approval proof chain diverged: {metadata}")
    if handoff.get("execution_health_next_commands") != metadata.get("execution_health_next_commands"):
        raise SystemExit(f"{label} execution health queue diverged: {metadata}")
    for key in [
        "execution_health_approval_held_review_required",
        "execution_health_approval_held_review_commands",
        "execution_health_approval_held_review_command_count",
        "execution_health_approval_held_review_next_command",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} work-session handoff {key} diverged: {metadata}")
    if handoff.get("execution_health_blocker_categories") != metadata.get("execution_health_blocker_categories"):
        raise SystemExit(f"{label} execution health blockers diverged: {metadata}")
    if handoff.get("agi_next_build_command") != metadata.get("agi_next_build_command"):
        raise SystemExit(f"{label} AGI build command diverged: {metadata}")
    if metadata.get("approval_handoff_pending_count", 0) > 0 and not any(
        str(command).startswith("approval readiness") for command in handoff.get("next_commands") or []
    ):
        raise SystemExit(f"{label} missed approval readiness next command: {metadata}")
    if "build delta" not in handoff.get("next_commands", []):
        raise SystemExit(f"{label} missed build delta next command: {metadata}")
    boundaries = handoff.get("boundaries") or {}
    if metadata.get("work_session_packet_boundaries") != boundaries:
        raise SystemExit(f"{label} boundary packet diverged: {metadata}")
    if boundaries.get("read_only") is not True:
        raise SystemExit(f"{label} boundary should report read_only=True: {metadata}")
    for flat_key, boundary_key in [
        ("work_session_packet_authorizes_execution", "authorizes_execution"),
        ("work_session_packet_authorizes_completion_claim", "authorizes_completion_claim"),
        ("work_session_packet_approval_granted", "approval_granted"),
    ]:
        if metadata.get(flat_key) is not False or boundaries.get(boundary_key) is not False:
            raise SystemExit(f"{label} missed flat no-authority parity for {flat_key}: {metadata}")
    for flag in READ_ONLY_FLAGS:
        if boundaries.get(flag) is not False:
            raise SystemExit(f"{label} boundary should report {flag}=False: {metadata}")
    handoff_text = json.dumps(handoff, sort_keys=True)
    for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
        if forbidden in handoff_text:
            raise SystemExit(f"{label} handoff leaked local path fragment {forbidden!r}: {metadata}")


def assert_next_session_plan_handoff(metadata: dict, label: str, *, objective: str | None = None) -> None:
    handoff = metadata.get("next_session_plan_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed next_session_plan_handoff: {metadata}")
    if metadata.get("next_session_plan_handoff_ready") is not True or handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} should report handoff ready: {metadata}")
    if metadata.get("next_session_plan_ready_for_operator") is not True or handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} should report ready_for_operator: {metadata}")
    if metadata.get("next_session_plan_state_changed") is not False or handoff.get("state_changed") is not False:
        raise SystemExit(f"{label} should report unchanged state: {metadata}")
    if metadata.get("next_session_plan_changed") != [] or handoff.get("changed") != []:
        raise SystemExit(f"{label} should report no changed resources: {metadata}")
    if metadata.get("next_session_plan_content_in_handoff") is not False or handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} should keep content out of handoff: {metadata}")
    if objective and handoff.get("objective") != objective:
        raise SystemExit(f"{label} objective diverged: {metadata}")
    for key in [
        "pending_approvals",
        "open_tasks",
        "active_goals",
        "active_decisions",
        "active_preferences",
        "enabled_jobs",
        "scheduled_jobs",
        "limit",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} {key} diverged: {metadata}")
    for key in BACKGROUND_RHYTHM_KEYS:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} background rhythm key {key} diverged: {metadata}")
    if metadata.get("next_session_plan_first_safe_move") != handoff.get("first_safe_move"):
        raise SystemExit(f"{label} first safe move diverged: {metadata}")
    if metadata.get("next_session_plan_resume_order") != handoff.get("resume_order"):
        raise SystemExit(f"{label} resume order diverged: {metadata}")
    if metadata.get("next_session_plan_next_commands") != handoff.get("next_commands"):
        raise SystemExit(f"{label} next commands diverged: {metadata}")
    if metadata.get("next_session_plan_next_safe_commands") != handoff.get("next_safe_commands"):
        raise SystemExit(f"{label} next-safe commands diverged: {metadata}")
    if handoff.get("next_safe_commands") != handoff.get("next_commands"):
        raise SystemExit(f"{label} next-safe commands should mirror next commands: {metadata}")
    if metadata.get("next_session_plan_next_command_count") != len(handoff.get("next_commands") or []):
        raise SystemExit(f"{label} next-command count diverged: {metadata}")
    if metadata.get("next_session_plan_next_safe_command_count") != len(handoff.get("next_safe_commands") or []):
        raise SystemExit(f"{label} next-safe-command count diverged: {metadata}")
    if handoff.get("approval_handoff_pending_count") != metadata.get("approval_handoff_pending_count"):
        raise SystemExit(f"{label} approval handoff count diverged: {metadata}")
    if handoff.get("approval_handoff_proof_chain_commands") != metadata.get("approval_handoff_proof_chain_commands"):
        raise SystemExit(f"{label} approval proof chain diverged: {metadata}")
    if handoff.get("execution_health_next_commands") != metadata.get("execution_health_next_commands"):
        raise SystemExit(f"{label} execution health queue diverged: {metadata}")
    for key in [
        "execution_health_approval_held_review_required",
        "execution_health_approval_held_review_commands",
        "execution_health_approval_held_review_command_count",
        "execution_health_approval_held_review_next_command",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} next-session handoff {key} diverged: {metadata}")
    if handoff.get("execution_health_blocker_categories") != metadata.get("execution_health_blocker_categories"):
        raise SystemExit(f"{label} execution health blockers diverged: {metadata}")
    if handoff.get("agi_next_build_command") != metadata.get("agi_next_build_command"):
        raise SystemExit(f"{label} AGI build command diverged: {metadata}")
    if metadata.get("approval_handoff_pending_count", 0) > 0 and not any(
        str(command).startswith("approval readiness") for command in handoff.get("next_commands") or []
    ):
        raise SystemExit(f"{label} missed approval readiness next command: {metadata}")
    if "safe next actions" not in handoff.get("next_commands", []):
        raise SystemExit(f"{label} missed safe next actions next command: {metadata}")
    boundaries = handoff.get("boundaries") or {}
    if metadata.get("next_session_plan_boundaries") != boundaries:
        raise SystemExit(f"{label} boundary packet diverged: {metadata}")
    if boundaries.get("read_only") is not True:
        raise SystemExit(f"{label} boundary should report read_only=True: {metadata}")
    for flat_key, boundary_key in [
        ("next_session_plan_authorizes_execution", "authorizes_execution"),
        ("next_session_plan_authorizes_completion_claim", "authorizes_completion_claim"),
        ("next_session_plan_approval_granted", "approval_granted"),
    ]:
        if metadata.get(flat_key) is not False or boundaries.get(boundary_key) is not False:
            raise SystemExit(f"{label} missed flat no-authority parity for {flat_key}: {metadata}")
    for flag in READ_ONLY_FLAGS:
        if boundaries.get(flag) is not False:
            raise SystemExit(f"{label} boundary should report {flag}=False: {metadata}")
    handoff_text = json.dumps(handoff, sort_keys=True)
    for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
        if forbidden in handoff_text:
            raise SystemExit(f"{label} handoff leaked local path fragment {forbidden!r}: {metadata}")


def assert_background_rhythm_result(
    label: str,
    result,
    handoff_key: str,
    *,
    expected_ready: bool,
    expected_issue: str,
    expected_command: str,
    expected_priority: str,
    expected_counts: dict[str, int],
    expected_fragments: list[str],
) -> None:
    if not result.ok:
        raise SystemExit(f"{label} should be ok: {result.output}")
    metadata = result.metadata
    handoff = metadata.get(handoff_key)
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed nested handoff {handoff_key}: {metadata}")
    for fragment in expected_fragments:
        if fragment not in result.output:
            raise SystemExit(f"{label} missed background rhythm text {fragment!r}: {result.output}")
    expected_values = {
        "background_ready": expected_ready,
        "background_issue": expected_issue,
        "background_next_command": expected_command,
        "background_priority": expected_priority,
        **expected_counts,
    }
    for key, expected in expected_values.items():
        if metadata.get(key) != expected:
            raise SystemExit(f"{label} missed metadata {key}={expected!r}: {metadata}")
        if handoff.get(key) != expected:
            raise SystemExit(f"{label} missed handoff {key}={expected!r}: {metadata}")
    if not expected_ready and expected_command and expected_command not in (handoff.get("next_commands") or []):
        raise SystemExit(f"{label} missed background command in next commands: {metadata}")


def assert_focus_tools_report_background_rhythm() -> None:
    objective = "continue Jarvis V2 safely"

    with TemporaryDirectory(prefix="jarvis-focus-bg-missing-compaction-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.upsert_job("State Snapshot", 360, "state_snapshot", "2099-01-01T00:00:00")
        focus_brief, next_session_plan, work_session_packet = make_focus_tools(runtime.store)
        expected_counts = {
            "state_snapshot_jobs": 1,
            "enabled_state_snapshot_jobs": 1,
            "disabled_state_snapshot_jobs": 0,
            "conversation_compaction_jobs": 0,
            "enabled_conversation_compaction_jobs": 0,
            "disabled_conversation_compaction_jobs": 0,
            "unreadable_scheduled_job_rows": 0,
        }
        cases = [
            ("focus brief missing compaction", focus_brief({"objective": objective}), "focus_brief_handoff"),
            ("next session missing compaction", next_session_plan({"objective": objective}), "next_session_plan_handoff"),
            ("work session missing compaction", work_session_packet({"objective": objective}), "work_session_packet_handoff"),
        ]
        for label, result, handoff_key in cases:
            assert_background_rhythm_result(
                label,
                result,
                handoff_key,
                expected_ready=False,
                expected_issue=f"{COMPACTION_JOB_NAME} scheduled job is not configured.",
                expected_command="schedule assistant basics",
                expected_priority="schedule assistant basics for durable memory compaction.",
                expected_counts=expected_counts,
                expected_fragments=[
                    f"{COMPACTION_JOB_NAME} scheduled job is not configured.",
                    "schedule assistant basics",
                ],
            )
            if handoff_key == "focus_brief_handoff":
                assert_focus_brief_handoff(result.metadata, label, objective=objective)
                if result.metadata.get("focus_brief_start_kind") != "background_repair":
                    raise SystemExit(f"{label} should start with background repair: {result.metadata}")
            elif handoff_key == "next_session_plan_handoff":
                assert_next_session_plan_handoff(result.metadata, label, objective=objective)
            else:
                assert_work_session_packet_handoff(result.metadata, label, objective=objective)

    with TemporaryDirectory(prefix="jarvis-focus-bg-paused-compaction-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.upsert_job("State Snapshot", 360, "state_snapshot", "2099-01-01T00:00:00")
        runtime.store.upsert_job(COMPACTION_JOB_NAME, 1440, COMPACTION_JOB_TYPE, "2099-01-01T00:00:00")
        if not runtime.store.set_job_enabled(COMPACTION_JOB_NAME, False):
            raise SystemExit("Paused compaction fixture could not disable the compaction job.")
        focus_brief, next_session_plan, work_session_packet = make_focus_tools(runtime.store)
        expected_counts = {
            "state_snapshot_jobs": 1,
            "enabled_state_snapshot_jobs": 1,
            "disabled_state_snapshot_jobs": 0,
            "conversation_compaction_jobs": 1,
            "enabled_conversation_compaction_jobs": 0,
            "disabled_conversation_compaction_jobs": 1,
            "unreadable_scheduled_job_rows": 0,
        }
        cases = [
            ("focus brief paused compaction", focus_brief({"objective": objective}), "focus_brief_handoff"),
            ("next session paused compaction", next_session_plan({"objective": objective}), "next_session_plan_handoff"),
            ("work session paused compaction", work_session_packet({"objective": objective}), "work_session_packet_handoff"),
        ]
        for label, result, handoff_key in cases:
            assert_background_rhythm_result(
                label,
                result,
                handoff_key,
                expected_ready=False,
                expected_issue=f"{COMPACTION_JOB_NAME} scheduled job is paused.",
                expected_command=f"resume job {COMPACTION_JOB_NAME}",
                expected_priority=f"resume job {COMPACTION_JOB_NAME} for durable memory compaction.",
                expected_counts=expected_counts,
                expected_fragments=[
                    f"{COMPACTION_JOB_NAME} scheduled job is paused.",
                    f"resume job {COMPACTION_JOB_NAME}",
                ],
            )
            if handoff_key == "focus_brief_handoff":
                assert_focus_brief_handoff(result.metadata, label, objective=objective)
            elif handoff_key == "next_session_plan_handoff":
                assert_next_session_plan_handoff(result.metadata, label, objective=objective)
            else:
                assert_work_session_packet_handoff(result.metadata, label, objective=objective)

    with TemporaryDirectory(prefix="jarvis-focus-bg-healthy-compaction-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.upsert_job("State Snapshot", 360, "state_snapshot", "2099-01-01T00:00:00")
        runtime.store.upsert_job(COMPACTION_JOB_NAME, 1440, COMPACTION_JOB_TYPE, "2099-01-01T00:00:00")
        focus_brief, next_session_plan, work_session_packet = make_focus_tools(runtime.store)
        expected_counts = {
            "state_snapshot_jobs": 1,
            "enabled_state_snapshot_jobs": 1,
            "disabled_state_snapshot_jobs": 0,
            "conversation_compaction_jobs": 1,
            "enabled_conversation_compaction_jobs": 1,
            "disabled_conversation_compaction_jobs": 0,
            "unreadable_scheduled_job_rows": 0,
        }
        cases = [
            ("focus brief healthy compaction", focus_brief({"objective": objective}), "focus_brief_handoff"),
            ("next session healthy compaction", next_session_plan({"objective": objective}), "next_session_plan_handoff"),
            ("work session healthy compaction", work_session_packet({"objective": objective}), "work_session_packet_handoff"),
        ]
        for label, result, handoff_key in cases:
            assert_background_rhythm_result(
                label,
                result,
                handoff_key,
                expected_ready=True,
                expected_issue="",
                expected_command="list scheduled jobs",
                expected_priority="",
                expected_counts=expected_counts,
                expected_fragments=[
                    f"State Snapshot and {COMPACTION_JOB_NAME} are enabled.",
                ],
            )
            if handoff_key == "focus_brief_handoff":
                assert_focus_brief_handoff(result.metadata, label, objective=objective)
            elif handoff_key == "next_session_plan_handoff":
                assert_next_session_plan_handoff(result.metadata, label, objective=objective)
            else:
                assert_work_session_packet_handoff(result.metadata, label, objective=objective)


def assert_focus_tools_fail_background_rhythm_closed_on_unreadable_job_rows() -> None:
    objective = "verify unreadable scheduled job rows"
    leak_marker = "SCHEDULED_JOB_ROW_SECRET /\x55sers/example/private/scheduler.sqlite"
    with TemporaryDirectory(prefix="jarvis-focus-bg-unreadable-jobs-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.upsert_job("State Snapshot", 360, "state_snapshot", "2099-01-01T00:00:00")
        runtime.store.upsert_job(COMPACTION_JOB_NAME, 1440, COMPACTION_JOB_TYPE, "2099-01-01T00:00:00")
        original_list_jobs = runtime.store.list_jobs

        def hostile_jobs(limit: int | None = None):
            rows = [HostileRow(leak_marker), *original_list_jobs()]
            return rows[:limit] if limit else rows

        runtime.store.list_jobs = hostile_jobs
        focus_brief, next_session_plan, work_session_packet = make_focus_tools(runtime.store)
        expected_counts = {
            "state_snapshot_jobs": 1,
            "enabled_state_snapshot_jobs": 1,
            "disabled_state_snapshot_jobs": 0,
            "conversation_compaction_jobs": 1,
            "enabled_conversation_compaction_jobs": 1,
            "disabled_conversation_compaction_jobs": 0,
            "unreadable_scheduled_job_rows": 1,
        }
        cases = [
            ("unreadable scheduled job focus_brief", focus_brief({"objective": objective}), "focus_brief_handoff", assert_focus_brief_handoff),
            (
                "unreadable scheduled job next_session_plan",
                next_session_plan({"objective": objective}),
                "next_session_plan_handoff",
                assert_next_session_plan_handoff,
            ),
            (
                "unreadable scheduled job work_session_packet",
                work_session_packet({"objective": objective}),
                "work_session_packet_handoff",
                assert_work_session_packet_handoff,
            ),
        ]
        for label, result, handoff_key, handoff_assertion in cases:
            assert_background_rhythm_result(
                label,
                result,
                handoff_key,
                expected_ready=False,
                expected_issue="Scheduled job table has 1 unreadable row(s).",
                expected_command="list scheduled jobs",
                expected_priority="review scheduled jobs before trusting background rhythm.",
                expected_counts=expected_counts,
                expected_fragments=[
                    "Scheduled job table has 1 unreadable row(s).",
                    "list scheduled jobs",
                ],
            )
            combined_text = result.output + json.dumps(result.metadata, sort_keys=True)
            for marker in [leak_marker, "scheduler.sqlite", "/\x55sers/example/private"]:
                if marker in combined_text:
                    raise SystemExit(f"{label} leaked unreadable scheduled-job row marker {marker!r}: {combined_text}")
            if result.metadata.get("enabled_jobs") != 2 or result.metadata.get("scheduled_jobs") != 3:
                raise SystemExit(f"{label} should count readable enabled jobs separately from total visible rows: {result.metadata}")
            handoff_assertion(result.metadata, label, objective=objective)


class HostileRow:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def keys(self):
        raise RuntimeError(self.marker)

    def __getitem__(self, key: str):
        raise RuntimeError(f"{self.marker}:{key}")

    def __str__(self) -> str:
        raise RuntimeError(self.marker)

    def __repr__(self) -> str:
        return f"<hostile-row {self.marker}>"


class HostileMetadataValue:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __bool__(self) -> bool:
        raise RuntimeError(self.marker)

    def __str__(self) -> str:
        raise RuntimeError(self.marker)

    def __repr__(self) -> str:
        return f"<hostile-metadata {self.marker}>"


class HostileTruthinessText:
    def __init__(self, text: str) -> None:
        self.text = text

    def __bool__(self) -> bool:
        raise RuntimeError(f"truthiness forbidden for {self.text}")

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"<hostile-truthiness-text {self.text!r}>"


def assert_focus_tools_redact_hostile_truthiness_objectives() -> None:
    leak_marker = "FOCUS_OBJECTIVE_SHOULD_NOT_LEAK"
    hostile_objective = HostileTruthinessText(f"/\x55sers/example/private/{leak_marker}/focus-objective.md")
    with TemporaryDirectory(prefix="jarvis-focus-hostile-objective-") as temp:
        runtime = make_temp_runtime(Path(temp))
        focus_brief, next_session_plan, work_session_packet = make_focus_tools(runtime.store)
        cases = [
            ("hostile objective focus_brief", focus_brief({"objective": hostile_objective}), assert_focus_brief_handoff),
            (
                "hostile objective next_session_plan",
                next_session_plan({"objective": hostile_objective}),
                assert_next_session_plan_handoff,
            ),
            (
                "hostile objective work_session_packet",
                work_session_packet({"objective": hostile_objective}),
                assert_work_session_packet_handoff,
            ),
        ]
        for label, result, handoff_assertion in cases:
            if not result.ok:
                raise SystemExit(f"{label} should tolerate hostile objective truthiness: {result.output}")
            combined_text = result.output + json.dumps(result.metadata, sort_keys=True)
            for marker in [leak_marker, "focus-objective.md", "/\x55sers/example/private"]:
                if marker in combined_text:
                    raise SystemExit(f"{label} leaked hostile objective marker {marker!r}: {combined_text}")
            if "<local-path>" not in combined_text:
                raise SystemExit(f"{label} should keep a redacted objective marker: {combined_text}")
            assert_execution_health_handoff(result.metadata, label)
            assert_agi_focus_handoff(result.metadata, label)
            handoff_assertion(result.metadata, label, objective="<local-path>")


def assert_focus_tools_tolerate_malformed_local_rows() -> None:
    leak_markers = ["AUDIT_ROW_SECRET", "APPROVAL_ROW_SECRET"]
    with TemporaryDirectory(prefix="jarvis-focus-malformed-rows-") as temp:
        runtime = make_temp_runtime(Path(temp))
        recent_rows = [
            HostileRow(leak_markers[0]),
            {
                "id": 21,
                "tool_name": "run_shell_command",
                "risk": "HIGH_RISK",
                "ok": False,
                "approved": False,
                "approval_id": 77,
                "metadata": json.dumps({"approval_id": 77}),
            },
            {
                "id": 22,
                "tool_name": "verification_receipt",
                "risk": "LOCAL_SAFE",
                "ok": True,
                "approved": False,
                "approval_id": None,
                "metadata": json.dumps({"run_id": 21}),
            },
        ]
        approval_rows = [
            HostileRow(leak_markers[1]),
            {"id": 77, "tool_name": "send_telegram", "user_input": "send a message"},
        ]
        runtime.store.recent_tool_runs = lambda limit=40: recent_rows[:limit]
        runtime.store.list_pending_approvals = lambda limit=6: approval_rows[:limit]

        focus_brief, next_session_plan, work_session_packet = make_focus_tools(runtime.store)
        cases = [
            ("malformed row focus_brief", focus_brief({"objective": "malformed row handoff"}), assert_focus_brief_handoff),
            (
                "malformed row next_session_plan",
                next_session_plan({"objective": "malformed row handoff"}),
                assert_next_session_plan_handoff,
            ),
            (
                "malformed row work_session_packet",
                work_session_packet({"objective": "malformed row handoff"}),
                assert_work_session_packet_handoff,
            ),
        ]
        for label, result, handoff_assertion in cases:
            if not result.ok:
                raise SystemExit(f"{label} should tolerate malformed rows: {result.output}")
            metadata_text = json.dumps(result.metadata, sort_keys=True)
            combined_text = result.output + metadata_text
            for marker in leak_markers:
                if marker in combined_text:
                    raise SystemExit(f"{label} leaked hostile row marker {marker}: {combined_text}")
            metadata = result.metadata
            if metadata.get("approval_handoff_pending_count") != 2 or metadata.get("approval_handoff_first_id") != 77:
                raise SystemExit(f"{label} should skip malformed approval rows and use the first readable approval: {metadata}")
            if metadata.get("execution_health_unreadable_recent_tool_run_rows") != 1:
                raise SystemExit(f"{label} missed unreadable audit row count: {metadata}")
            if "unreadable_audit_rows" not in metadata.get("execution_health_blocker_categories", []):
                raise SystemExit(f"{label} missed unreadable audit-row blocker: {metadata}")
            for command in ["execution recovery packet 21", "approval readiness 77", "execution health report"]:
                if command not in metadata.get("execution_health_next_commands", []):
                    raise SystemExit(f"{label} missed guarded next command {command!r}: {metadata}")
            assert_execution_health_handoff(metadata, label)
            assert_agi_focus_handoff(metadata, label)
            handoff_assertion(metadata, label, objective="malformed row handoff")


def assert_focus_tools_tolerate_malformed_tool_run_metadata() -> None:
    leak_marker = "FOCUS_METADATA_SHOULD_NOT_LEAK /\x55sers/example/private/focus.sqlite"
    with TemporaryDirectory(prefix="jarvis-focus-malformed-metadata-") as temp:
        runtime = make_temp_runtime(Path(temp))
        recent_rows = [
            {
                "id": 31,
                "tool_name": "send_kakao",
                "risk": "HIGH_RISK",
                "ok": False,
                "approved": True,
                "approval_id": None,
                "metadata": {
                    "failure_kind": "transport_error",
                    "failure_stage": HostileMetadataValue(leak_marker),
                },
            },
            {
                "id": 32,
                "tool_name": "send_telegram",
                "risk": "HIGH_RISK",
                "ok": False,
                "approved": False,
                "approval_id": 12,
                "metadata": {
                    "failure_kind": HostileMetadataValue(leak_marker),
                    "failure_stage": "explicit approval required",
                },
            },
            {
                "id": 33,
                "tool_name": "verification_receipt",
                "risk": "LOCAL_SAFE",
                "ok": True,
                "approved": False,
                "approval_id": None,
                "metadata": json.dumps({"run_id": 31}),
            },
        ]
        runtime.store.recent_tool_runs = lambda limit=40: recent_rows[:limit]

        focus_brief, next_session_plan, work_session_packet = make_focus_tools(runtime.store)
        cases = [
            (
                "malformed metadata focus_brief",
                focus_brief({"objective": "malformed metadata handoff"}),
                assert_focus_brief_handoff,
            ),
            (
                "malformed metadata next_session_plan",
                next_session_plan({"objective": "malformed metadata handoff"}),
                assert_next_session_plan_handoff,
            ),
            (
                "malformed metadata work_session_packet",
                work_session_packet({"objective": "malformed metadata handoff"}),
                assert_work_session_packet_handoff,
            ),
        ]
        for label, result, handoff_assertion in cases:
            if not result.ok:
                raise SystemExit(f"{label} should tolerate malformed tool-run metadata: {result.output}")
            metadata = result.metadata
            if metadata.get("failed_action_runs") != 1 or metadata.get("execution_health_failed_action_runs") != 1:
                raise SystemExit(f"{label} should count the hostile transport row as a true failure: {metadata}")
            if metadata.get("approval_held_action_runs") != 1 or metadata.get("execution_health_approval_held_action_runs") != 1:
                raise SystemExit(f"{label} should still classify the approval-held row after hostile metadata: {metadata}")
            for command in [
                "execution recovery packet 31",
                "approval readiness 12",
                "approval packet 12",
                "approval chain proof 12",
                "verification receipt <approved run id from approval chain proof 12>",
            ]:
                if command not in metadata.get("execution_health_next_commands", []):
                    raise SystemExit(f"{label} missed guarded next command {command!r}: {metadata}")
            combined_text = result.output + json.dumps(metadata, sort_keys=True)
            for marker in [leak_marker, "focus.sqlite", "/\x55sers/example/private"]:
                if marker in combined_text:
                    raise SystemExit(f"{label} leaked hostile metadata marker {marker}: {combined_text}")
            if "Action run attention: 1 failed/blocked, 1 approval-held." not in result.output:
                raise SystemExit(f"{label} missed split action attention text: {result.output}")
            assert_execution_health_handoff(metadata, label)
            assert_agi_focus_handoff(metadata, label)
            handoff_assertion(metadata, label, objective="malformed metadata handoff")


def assert_focus_tools_separate_approval_held_tool_runs() -> None:
    with TemporaryDirectory(prefix="jarvis-focus-approval-held-") as temp:
        runtime = make_temp_runtime(Path(temp))
        failed_run_id = runtime.store.log_tool_run(
            session_id="focus-approval-held",
            tool_name="send_kakao",
            risk="HIGH_RISK",
            ok=False,
            approved=True,
            output="transport timed out before delivery",
            metadata={"failure_kind": "transport_error", "failure_stage": "transport_timeout"},
        )
        approval_held_run_id = runtime.store.log_tool_run(
            session_id="focus-approval-held",
            tool_name="send_telegram",
            risk="HIGH_RISK",
            ok=False,
            approved=False,
            output="raw focus approval held marker should stay private",
            approval_id=12,
            metadata={"failure_kind": "approval-gate", "requires_confirmation": True},
        )

        focus_brief, next_session_plan, work_session_packet = make_focus_tools(runtime.store)
        cases = [
            (
                "approval-held focus_brief",
                focus_brief({"objective": "approval-held focus handoff"}),
                assert_focus_brief_handoff,
            ),
            (
                "approval-held next_session_plan",
                next_session_plan({"objective": "approval-held focus handoff"}),
                assert_next_session_plan_handoff,
            ),
            (
                "approval-held work_session_packet",
                work_session_packet({"objective": "approval-held focus handoff"}),
                assert_work_session_packet_handoff,
            ),
        ]
        for label, result, handoff_assertion in cases:
            if not result.ok:
                raise SystemExit(f"{label} should render a read-only focus handoff: {result.output}")
            metadata = result.metadata
            if metadata.get("failed_action_runs") != 1 or metadata.get("execution_health_failed_action_runs") != 1:
                raise SystemExit(f"{label} should count only true failures as failed action runs: {metadata}")
            if metadata.get("approval_held_action_runs") != 1 or metadata.get("execution_health_approval_held_action_runs") != 1:
                raise SystemExit(f"{label} should count approval-held action runs separately: {metadata}")
            if metadata.get("execution_health_learning_target_run_id") != failed_run_id:
                raise SystemExit(f"{label} should target recovery learning at the true failure: {metadata}")
            if metadata.get("execution_health_recovery_closure_target_run_id") != failed_run_id:
                raise SystemExit(f"{label} should keep recovery closure pointed at the true failure: {metadata}")
            if "approval_held" not in metadata.get("execution_health_blocker_categories", []):
                raise SystemExit(f"{label} missed approval-held blocker category: {metadata}")
            next_commands = metadata.get("execution_health_next_commands", [])
            if f"execution recovery packet {approval_held_run_id}" in next_commands:
                raise SystemExit(f"{label} should not create a recovery packet for an approval-held run: {metadata}")
            if f"execution recovery packet {failed_run_id}" not in next_commands:
                raise SystemExit(f"{label} missed recovery packet for the true failed run: {metadata}")
            for command in [
                "approval readiness 12",
                "approval packet 12",
                "approval chain proof 12",
                "verification receipt <approved run id from approval chain proof 12>",
            ]:
                if command not in next_commands:
                    raise SystemExit(f"{label} missed approval-held proof command {command!r}: {metadata}")
            if "Action run attention: 1 failed/blocked, 1 approval-held." not in result.output:
                raise SystemExit(f"{label} missed approval-held action attention text: {result.output}")
            if "raw focus approval held marker should stay private" in result.output:
                raise SystemExit(f"{label} leaked raw approval-held audit output: {result.output}")
            assert_execution_health_handoff(metadata, label)
            assert_agi_focus_handoff(metadata, label)
            handoff_assertion(metadata, label, objective="approval-held focus handoff")

    with TemporaryDirectory(prefix="jarvis-focus-approval-held-only-") as temp:
        runtime = make_temp_runtime(Path(temp))
        approval_held_run_id = runtime.store.log_tool_run(
            session_id="focus-approval-held-only",
            tool_name="send_telegram",
            risk="HIGH_RISK",
            ok=False,
            approved=False,
            output="raw approval only marker should stay private",
            approval_id=13,
            metadata={"failure_stage": "explicit approval required"},
        )

        focus_brief, next_session_plan, work_session_packet = make_focus_tools(runtime.store)
        cases = [
            (
                "approval-held-only focus_brief",
                focus_brief({"objective": "approval-held-only focus handoff"}),
                assert_focus_brief_handoff,
            ),
            (
                "approval-held-only next_session_plan",
                next_session_plan({"objective": "approval-held-only focus handoff"}),
                assert_next_session_plan_handoff,
            ),
            (
                "approval-held-only work_session_packet",
                work_session_packet({"objective": "approval-held-only focus handoff"}),
                assert_work_session_packet_handoff,
            ),
        ]
        for label, result, handoff_assertion in cases:
            metadata = result.metadata
            if metadata.get("failed_action_runs") != 0 or metadata.get("execution_health_failed_action_runs") != 0:
                raise SystemExit(f"{label} should not count approval-held-only rows as failures: {metadata}")
            if metadata.get("approval_held_action_runs") != 1 or metadata.get("execution_health_approval_held_action_runs") != 1:
                raise SystemExit(f"{label} should expose approval-held-only count: {metadata}")
            if metadata.get("execution_health_recovery_closure_state") != "not_needed":
                raise SystemExit(f"{label} should not create recovery closure debt for approval-held-only state: {metadata}")
            if metadata.get("execution_health_recovery_closure_target_run_id") is not None:
                raise SystemExit(f"{label} should not target approval-held-only row for recovery closure: {metadata}")
            next_commands = metadata.get("execution_health_next_commands", [])
            if f"execution recovery packet {approval_held_run_id}" in next_commands:
                raise SystemExit(f"{label} should not create a recovery packet for approval-held-only row: {metadata}")
            if "approval readiness 13" not in next_commands or "approval packet 13" not in next_commands:
                raise SystemExit(f"{label} missed approval proof commands for approval-held-only row: {metadata}")
            if "Action run attention: 0 failed/blocked, 1 approval-held." not in result.output:
                raise SystemExit(f"{label} missed approval-held-only action attention text: {result.output}")
            if "raw approval only marker should stay private" in result.output:
                raise SystemExit(f"{label} leaked raw approval-held-only audit output: {result.output}")
            assert_execution_health_handoff(metadata, label)
            assert_agi_focus_handoff(metadata, label)
            handoff_assertion(metadata, label, objective="approval-held-only focus handoff")


def assert_agi_focus_handoff_redacts_path_shaped_targets() -> None:
    original_targets = dict(focus_module.AGI_GATE_BUILD_TARGETS)
    original_selected_gate = focus_module.AGI_FOCUS_SELECTED_GATE
    secret_marker = "AGI_PATH_SECRET"
    raw_gate = f"/\x55sers/example/{secret_marker}/personal-integrations"
    raw_file = f"/\x55sers/example/{secret_marker}/jarvis_v2/tools/live_connector.py"
    raw_private_note = f"/private/tmp/{secret_marker}/proof.txt"
    try:
        focus_module.AGI_GATE_BUILD_TARGETS.clear()
        focus_module.AGI_GATE_BUILD_TARGETS.update(
            {
                raw_gate: {
                    "title": HostileTruthinessText(f"Keep {raw_gate} hidden from focus packets."),
                    "files": [
                        HostileTruthinessText(raw_file),
                        "jarvis_v2/tools/focus.py",
                    ],
                    "tests": [
                        HostileTruthinessText(f"python3 -m py_compile {raw_file}"),
                        "python3 -m jarvis_v2.scripts.smoke_test_focus",
                    ],
                    "acceptance": [
                        HostileTruthinessText(
                            f"missing target file {raw_private_note} is redacted in handoff metadata"
                        ),
                        "relative target files remain visible for the next maintainer",
                    ],
                }
            }
        )
        focus_module.AGI_FOCUS_SELECTED_GATE = raw_gate

        with TemporaryDirectory(prefix="jarvis-focus-path-redaction-") as temp:
            runtime = make_temp_runtime(Path(temp))
            focus_brief, next_session_plan, work_session_packet = make_focus_tools(runtime.store)
            cases = [
                ("path target focus_brief", focus_brief({"objective": "path target redaction"}), assert_focus_brief_handoff),
                (
                    "path target next_session_plan",
                    next_session_plan({"objective": "path target redaction"}),
                    assert_next_session_plan_handoff,
                ),
                (
                    "path target work_session_packet",
                    work_session_packet({"objective": "path target redaction"}),
                    assert_work_session_packet_handoff,
                ),
            ]
            for label, result, handoff_assertion in cases:
                if not result.ok:
                    raise SystemExit(f"{label} should produce a read-only handoff: {result.output}")
                metadata = result.metadata
                combined = result.output + json.dumps(metadata, sort_keys=True)
                for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/", secret_marker]:
                    if forbidden in combined:
                        raise SystemExit(f"{label} leaked path-shaped AGI target data {forbidden!r}: {combined}")
                if "<local-path>" not in combined:
                    raise SystemExit(f"{label} should preserve a redacted path marker for debugging: {combined}")
                if metadata.get("agi_next_gate") != "<local-path>":
                    raise SystemExit(f"{label} should redact the selected AGI gate: {metadata}")
                if metadata.get("agi_next_likely_files") != ["<local-path>", "jarvis_v2/tools/focus.py"]:
                    raise SystemExit(f"{label} should redact only path-shaped target files: {metadata}")
                if metadata.get("agi_next_missing_target_files") != ["<local-path>"]:
                    raise SystemExit(f"{label} should report path-shaped target files as stale without leaking them: {metadata}")
                if metadata.get("agi_next_target_files_checked") != 2 or metadata.get("agi_next_target_files_exist") is not False:
                    raise SystemExit(f"{label} should keep relative integrity checks while blocking absolute target paths: {metadata}")
                if metadata.get("agi_next_target_integrity_blocks_start") is not True:
                    raise SystemExit(f"{label} should block start when a target path is stale/redacted: {metadata}")
                handoff = metadata.get("agi_focus_handoff") or {}
                if handoff.get("selected_gate") != metadata.get("agi_next_gate"):
                    raise SystemExit(f"{label} nested handoff should mirror redacted selected gate: {metadata}")
                if handoff.get("missing_target_files") != metadata.get("agi_next_missing_target_files"):
                    raise SystemExit(f"{label} nested handoff should mirror redacted missing files: {metadata}")
                for flag in READ_ONLY_FLAGS:
                    if metadata.get(flag) or handoff.get(flag):
                        raise SystemExit(f"{label} should remain read-only/non-authorizing for {flag}: {metadata}")
                handoff_assertion(metadata, label, objective="path target redaction")
    finally:
        focus_module.AGI_GATE_BUILD_TARGETS.clear()
        focus_module.AGI_GATE_BUILD_TARGETS.update(original_targets)
        focus_module.AGI_FOCUS_SELECTED_GATE = original_selected_gate


def main() -> None:
    test_planner_routes_focus_brief_aliases()
    assert_focus_metadata_bool_is_exact()
    assert_focus_tools_report_background_rhythm()
    assert_focus_tools_fail_background_rhythm_closed_on_unreadable_job_rows()
    assert_focus_tools_redact_hostile_truthiness_objectives()
    assert_focus_tools_tolerate_malformed_local_rows()
    assert_focus_tools_tolerate_malformed_tool_run_metadata()
    assert_focus_tools_separate_approval_held_tool_runs()
    assert_agi_focus_handoff_redacts_path_shaped_targets()
    with TemporaryDirectory(prefix="jarvis-focus-") as temp:
        runtime = make_temp_runtime(Path(temp))
        focus_brief, next_session_plan, work_session_packet = make_focus_tools(runtime.store)
        planner_routes = {
            "start work please": ("focus_brief", {"objective": ""}),
            "focus me please": ("focus_brief", {"objective": ""}),
            "what should I focus on now": ("focus_brief", {"objective": ""}),
            "plan next session please": ("next_session_plan", {"objective": ""}),
            "next session please": ("next_session_plan", {"objective": ""}),
            "what should I do next when I return": ("next_session_plan", {"objective": ""}),
            "what should we do next session": ("next_session_plan", {"objective": ""}),
            "plan the next Jarvis session": ("next_session_plan", {"objective": ""}),
            "how should I restart Jarvis work": ("next_session_plan", {"objective": ""}),
            "how should I resume Jarvis": ("next_session_plan", {"objective": ""}),
        }
        for command, (expected_tool, expected_args) in planner_routes.items():
            plan = runtime.planner.plan(command)
            actual = [(action.tool_name, action.args) for action in plan.actions]
            if actual != [(expected_tool, expected_args)]:
                raise SystemExit(f"Focus planner route mismatch for {command!r}: {actual}")
        cases = [
            "add task review focus brief output priority high",
            "create goal Build Jarvis focus mode because work sessions need orientation",
            "add step to goal 1: keep risky actions approval gated",
            "record decision Focus briefs stay read only because session planning should be safe impact no computer control runs automatically",
            "set preference work session style to concise and action oriented category work",
            "run command python3 --version",
            "schedule assistant basics",
            "focus brief: continue Jarvis V2 safely",
            "work session packet: continue Jarvis V2 safely",
            "next session plan: continue Jarvis V2 safely",
            "resume plan",
            "start work session",
        ]
        for case in cases:
            result = runtime.handle(case)
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:1800])
            print()
            if case.startswith("focus brief"):
                for expected in [
                    "Jarvis focus brief",
                    "Objective: continue Jarvis V2 safely",
                    "Session plan",
                    "Execution health handoff",
                    "next required command:",
                    "Action run attention:",
                    "Operator limits",
                    "explicit stop times",
                    "review focus brief output",
                    "approval readiness",
                    "approval packet",
                    "approval chain proof",
                    "first safe approval handoff",
                    "approve approval",
                    "dismiss approval",
                    "AGI build-readiness handoff",
                    "Selected target:",
                    "agi next build move: personal integrations",
                    "Selection source: operator_focus_handoff",
                    "Canonical gate selector: `agi gates`",
                    "Ready for review:",
                    "Approval blockers",
                    "Do not auto-run shell",
                ]:
                    if expected not in result.response:
                        raise SystemExit(f"Focus brief missing expected context: {expected}")
                if "Execution proof queue:\n  - next proof command:" in result.response:
                    raise SystemExit("Focus brief should render execution queue head as next required command.")
                metadata = result.tool_results[0].metadata
                assert_execution_health_handoff(metadata, "focus_brief")
                assert_approval_handoff(metadata, "focus_brief")
                assert_agi_focus_handoff(metadata, "focus_brief")
                assert_focus_brief_handoff(metadata, "focus_brief", objective="continue Jarvis V2 safely")
            if case.startswith("work session packet"):
                for expected in [
                    "Jarvis work session packet",
                    "read-only",
                    "Objective: continue Jarvis V2 safely",
                    "Start move",
                    "approval readiness",
                    "approval packet",
                    "approval chain proof",
                    "approval detail",
                    "Approval handoff",
                    "first safe approval handoff",
                    "Preflight checklist",
                    "next action packet",
                    "action rehearsal",
                    "Visible context",
                    "Stop condition",
                    "End-of-session verification",
                    "build delta",
                    "save handoff brief",
                    "Execution health handoff",
                    "next required command:",
                    "Action run attention:",
                    "Next audit command",
                    "AGI build-readiness handoff",
                    "agi next build move: personal integrations",
                    "Selection source: operator_focus_handoff",
                    "Canonical gate selector: `agi gates`",
                    "Ready for review:",
                    "Operator limits",
                    "explicit stop times",
                ]:
                    if expected not in result.response:
                        raise SystemExit(f"Work-session packet missing expected context: {expected}")
                if "Execution proof queue:\n  - next proof command:" in result.response:
                    raise SystemExit("Work-session packet should render execution queue head as next required command.")
                metadata = result.tool_results[0].metadata
                assert_execution_health_handoff(metadata, "work_session_packet")
                assert_approval_handoff(metadata, "work_session_packet")
                assert_agi_focus_handoff(metadata, "work_session_packet")
                assert_work_session_packet_handoff(metadata, "work_session_packet", objective="continue Jarvis V2 safely")
                for flag in READ_ONLY_FLAGS:
                    if metadata.get(flag):
                        raise SystemExit(f"Work-session packet should not set {flag}: {metadata}")
            if case.startswith("next session plan"):
                for expected in [
                    "Jarvis next session plan",
                    "Objective: continue Jarvis V2 safely",
                    "Resume order",
                    "First safe move",
                    "review focus brief output",
                    "approval readiness",
                    "approval packet",
                    "approval chain proof",
                    "first safe approval handoff",
                    "approve approval",
                    "dismiss approval",
                    "Approval blockers",
                    "Do not auto-run shell",
                    "Execution health handoff",
                    "next required command:",
                    "Action run attention:",
                    "Next audit command",
                    "AGI build-readiness handoff",
                    "agi next build move: personal integrations",
                    "Selection source: operator_focus_handoff",
                    "Canonical gate selector: `agi gates`",
                    "Ready for review:",
                    "Operator limits",
                    "explicit stop times",
                ]:
                    if expected not in result.response:
                        raise SystemExit(f"Next-session plan missing expected context: {expected}")
                if "Execution proof queue:\n  - next proof command:" in result.response:
                    raise SystemExit("Next-session plan should render execution queue head as next required command.")
                metadata = result.tool_results[0].metadata
                assert_execution_health_handoff(metadata, "next_session_plan")
                assert_approval_handoff(metadata, "next_session_plan")
                assert_agi_focus_handoff(metadata, "next_session_plan")
                assert_next_session_plan_handoff(metadata, "next_session_plan", objective="continue Jarvis V2 safely")
                if "writes_notes" not in str(metadata) or metadata.get("queues_approval"):
                    raise SystemExit("Next-session plan metadata missed read-only flags.")
            if case == "resume plan":
                for expected in ["Jarvis next session plan", "Session guardrails", "Visible context"]:
                    if expected not in result.response:
                        raise SystemExit(f"Resume-plan route missing expected context: {expected}")
            if case == "start work session":
                for expected in ["Jarvis focus brief", "Start here", "Safety boundary"]:
                    if expected not in result.response:
                        raise SystemExit(f"Work-session route missing expected context: {expected}")

        focus_result = focus_brief({"limit": "not-a-number", "objective": "test focus metadata"})
        if focus_result.metadata.get("limit") != 6 or focus_result.metadata.get("writes_files"):
            raise SystemExit("focus_brief should sanitize bad limits and remain read-only.")
        assert_agi_focus_handoff(focus_result.metadata, "bad-limit focus_brief")
        assert_focus_brief_handoff(focus_result.metadata, "bad-limit focus_brief", objective="test focus metadata")
        for flag in READ_ONLY_FLAGS:
            if focus_result.metadata.get(flag):
                raise SystemExit(f"focus_brief should not set {flag}: {focus_result.metadata}")

        bool_focus = focus_brief({"limit": False, "objective": "test focus boolean metadata"})
        if bool_focus.metadata.get("limit") != 6 or bool_focus.metadata.get("writes_files"):
            raise SystemExit("focus_brief should treat boolean limits as malformed and remain read-only.")
        assert_agi_focus_handoff(bool_focus.metadata, "boolean focus_brief")
        assert_focus_brief_handoff(bool_focus.metadata, "boolean focus_brief", objective="test focus boolean metadata")
        for flag in READ_ONLY_FLAGS:
            if bool_focus.metadata.get(flag):
                raise SystemExit(f"focus_brief boolean limit should not set {flag}: {bool_focus.metadata}")

        long_objective = "x" * (MAX_OBJECTIVE_CHARS + 100)
        clipped_focus = focus_brief({"objective": long_objective})
        if clipped_focus.metadata.get("objective_length") != MAX_OBJECTIVE_CHARS:
            raise SystemExit("focus_brief should bound oversized objectives.")

        runtime.store.log_tool_run(
            session_id="malformed-focus-approval",
            tool_name="run_shell_command",
            risk="HIGH_RISK",
            ok=False,
            approved=False,
            output="approval metadata malformed fixture",
            metadata={"approval_id": float("inf"), "approved_approval_id": True},
        )
        malformed_approval_focus = focus_brief({"objective": "malformed approval metadata"})
        if not malformed_approval_focus.ok or malformed_approval_focus.metadata.get("execution_health_approval_proof_chain_count", 0) < 1:
            raise SystemExit(f"focus_brief should ignore malformed approval metadata while preserving valid chains: {malformed_approval_focus.metadata}")
        if "inf" in str(malformed_approval_focus.metadata.get("execution_health_approval_proof_chains", {})):
            raise SystemExit(f"focus_brief should not create proof chains for malformed approval ids: {malformed_approval_focus.metadata}")
        assert_agi_focus_handoff(malformed_approval_focus.metadata, "malformed approval focus_brief")
        assert_focus_brief_handoff(malformed_approval_focus.metadata, "malformed approval focus_brief", objective="malformed approval metadata")

        bool_plan = next_session_plan({"limit": True})
        if bool_plan.metadata.get("limit") != 6 or bool_plan.metadata.get("writes_database"):
            raise SystemExit("next_session_plan should treat boolean limits as malformed and stay read-only.")
        assert_execution_health_handoff(bool_plan.metadata, "boolean next_session_plan")
        assert_agi_focus_handoff(bool_plan.metadata, "boolean next_session_plan")
        assert_next_session_plan_handoff(bool_plan.metadata, "boolean next_session_plan")
        for flag in READ_ONLY_FLAGS:
            if bool_plan.metadata.get(flag):
                raise SystemExit(f"next_session_plan boolean limit should not set {flag}: {bool_plan.metadata}")

        clipped_plan = next_session_plan({"limit": 999999})
        if clipped_plan.metadata.get("limit") != 50 or clipped_plan.metadata.get("writes_database"):
            raise SystemExit("next_session_plan should clamp huge limits and stay read-only.")
        assert_execution_health_handoff(clipped_plan.metadata, "clipped next_session_plan")
        assert_agi_focus_handoff(clipped_plan.metadata, "clipped next_session_plan")
        assert_next_session_plan_handoff(clipped_plan.metadata, "clipped next_session_plan")
        for flag in READ_ONLY_FLAGS:
            if clipped_plan.metadata.get(flag):
                raise SystemExit(f"next_session_plan should not set {flag}: {clipped_plan.metadata}")

        low_packet = work_session_packet({"limit": -10})
        if low_packet.metadata.get("limit") != 1 or low_packet.metadata.get("writes_files"):
            raise SystemExit("work_session_packet should clamp low limits and stay read-only.")
        assert_execution_health_handoff(low_packet.metadata, "low work_session_packet")
        assert_agi_focus_handoff(low_packet.metadata, "low work_session_packet")
        assert_work_session_packet_handoff(low_packet.metadata, "low work_session_packet")
        for flag in READ_ONLY_FLAGS:
            if low_packet.metadata.get(flag):
                raise SystemExit(f"work_session_packet should not set {flag}.")

        bool_packet = work_session_packet({"limit": False})
        if bool_packet.metadata.get("limit") != 6 or bool_packet.metadata.get("writes_files"):
            raise SystemExit("work_session_packet should treat boolean limits as malformed and stay read-only.")
        assert_execution_health_handoff(bool_packet.metadata, "boolean work_session_packet")
        assert_agi_focus_handoff(bool_packet.metadata, "boolean work_session_packet")
        assert_work_session_packet_handoff(bool_packet.metadata, "boolean work_session_packet")
        for flag in READ_ONLY_FLAGS:
            if bool_packet.metadata.get(flag):
                raise SystemExit(f"work_session_packet boolean limit should not set {flag}.")

    with TemporaryDirectory(prefix="jarvis-focus-repeated-failure-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.log_tool_run(
            session_id="focus-repeated-failure",
            tool_name="send_kakao",
            risk="HIGH_RISK",
            ok=False,
            approved=True,
            output="transport timed out before delivery",
            metadata={"failure_kind": "transport_error", "failure_stage": "transport_timeout"},
        )
        runtime.store.log_tool_run(
            session_id="focus-repeated-failure",
            tool_name="send_kakao",
            risk="HIGH_RISK",
            ok=False,
            approved=True,
            output="transport timed out before delivery",
            metadata={"failure_kind": "transport_error", "failure_stage": "transport_timeout"},
        )
        focus_brief, next_session_plan, work_session_packet = make_focus_tools(runtime.store)
        for label, result in [
            ("focus_brief repeated failures", focus_brief({"objective": "repeated failure handoff"})),
            ("next_session_plan repeated failures", next_session_plan({"objective": "repeated failure handoff"})),
            ("work_session_packet repeated failures", work_session_packet({"objective": "repeated failure handoff"})),
        ]:
            assert_execution_health_handoff(result.metadata, label)
            assert_agi_focus_handoff(result.metadata, label)
            if label.startswith("focus_brief"):
                assert_focus_brief_handoff(result.metadata, label, objective="repeated failure handoff")
            if label.startswith("next_session_plan"):
                assert_next_session_plan_handoff(result.metadata, label, objective="repeated failure handoff")
            if label.startswith("work_session_packet"):
                assert_work_session_packet_handoff(result.metadata, label, objective="repeated failure handoff")
            if result.metadata["execution_health_repeated_failure_count"] < 1:
                raise SystemExit(f"{label} missed repeated failure count: {result.metadata}")
            if "Failure promotion queue" not in result.output:
                raise SystemExit(f"{label} missed failure promotion queue text.")
            if "Operator limits" not in result.output or "explicit stop times" not in result.output:
                raise SystemExit(f"{label} missed operator-limit handoff text.")


if __name__ == "__main__":
    main()
