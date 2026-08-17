from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Callable

from jarvis_v2.agent.types import ToolResult
from jarvis_v2.automations.compaction import COMPACTION_JOB_NAME, COMPACTION_JOB_TYPE
from jarvis_v2.config import JarvisConfig
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import MemoryStore
from jarvis_v2.tools.audit_meta import EXECUTION_META_TOOLS
from jarvis_v2.tools.harness import (
    AGI_GATE_BUILD_TARGETS,
    _agi_gate_summary,
    _is_execution_action_row,
    _selected_agi_target_readiness,
    _target_file_integrity,
)
from jarvis_v2.tools.storage import (
    BOOTSTRAP_CHECK_COMMAND,
    BOOTSTRAP_WRITE_COMMAND,
    STORAGE_RECOVERY_CHECK_API,
    STORAGE_RECOVERY_CHECK_COMMAND,
    STORAGE_RECOVERY_PLAN_COMMAND,
    _configured_storage_diagnostics,
)


MAX_NEXT_STEP_LIMIT = 20
MAX_NEXT_STEP_TEXT_CHARS = 260
MAX_OBJECTIVE_CHARS = 300
CHECKPOINT_REVIEW_MINUTES = 120
CHECKPOINT_STALE_MINUTES = 360
AUDIT_META_TOOLS = {*EXECUTION_META_TOOLS, "execution_learning_closure_packet"}
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")


def _bounded_limit(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return max(1, min(MAX_NEXT_STEP_LIMIT, int(value)))
    except (TypeError, ValueError):
        return default


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _short(value: Any, *, limit: int = MAX_NEXT_STEP_TEXT_CHARS) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _safe_text(value: Any, *, limit: int = MAX_NEXT_STEP_TEXT_CHARS) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", _short(value, limit=limit))


def _safe_vault_path_display(path: Path | str | None, vault: ObsidianVault) -> str:
    if path is None:
        return ""
    candidate = Path(path)
    try:
        return str(candidate.relative_to(vault.root_path))
    except ValueError:
        return _short(candidate, limit=160)


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "calls_model": False,
        "executes_tools": False,
        "queues_approval": False,
        "requires_approval": False,
        "approves_request": False,
        "dismisses_request": False,
        "reads_personal_data": False,
        "reads_private_data": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "edits_files": False,
        "controls_computer": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "speaks": False,
        "completes_tasks": False,
    }
    metadata.update(extra)
    return metadata


NEXT_STEP_BOUNDARY_FALSE_FLAGS = {
    "calls_model": False,
    "executes_tools": False,
    "queues_approval": False,
    "requires_approval": False,
    "approves_request": False,
    "dismisses_request": False,
    "reads_personal_data": False,
    "reads_private_data": False,
    "executes_side_effect": False,
    "external_side_effect": False,
    "writes_files": False,
    "writes_database": False,
    "writes_memory": False,
    "writes_notes": False,
    "edits_files": False,
    "controls_computer": False,
    "authorizes_execution": False,
    "authorizes_completion_claim": False,
    "approval_granted": False,
    "speaks": False,
    "completes_tasks": False,
}


def _scheduler_background_state(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    enabled_jobs = [row for row in jobs if row["enabled"]]
    state_snapshot_jobs = [row for row in jobs if row["job_type"] == "state_snapshot"]
    enabled_state_snapshot_jobs = [row for row in state_snapshot_jobs if row["enabled"]]
    conversation_compaction_jobs = [
        row
        for row in jobs
        if row["job_type"] == COMPACTION_JOB_TYPE
        or str(row["name"]).casefold() == COMPACTION_JOB_NAME.casefold()
    ]
    enabled_conversation_compaction_jobs = [row for row in conversation_compaction_jobs if row["enabled"]]
    if enabled_state_snapshot_jobs and enabled_conversation_compaction_jobs:
        command = "list scheduled jobs"
        message = f"Background rhythm is on: {len(enabled_jobs)} scheduled job(s) enabled."
        action_kind = ""
        action_title = ""
        rationale = ""
        verification = ""
        risk = ""
    elif not state_snapshot_jobs:
        command = "schedule assistant basics"
        message = "Turn on basic safe background reviews with `schedule assistant basics`."
        action_kind = "schedule_basics"
        action_title = "Enable safe scheduled context refreshes"
        rationale = "No background rhythm is visible, so safe context refreshes would improve continuity."
        verification = "Run `list scheduled jobs` and confirm the safe jobs are enabled."
        risk = "LOCAL_SAFE scheduler writes only."
    elif not enabled_state_snapshot_jobs:
        command = "resume job State Snapshot"
        message = "Background rhythm is paused: State Snapshot exists but is disabled; use `resume job State Snapshot`."
        action_kind = "resume_state_snapshot"
        action_title = "Resume safe scheduled context refreshes"
        rationale = "A State Snapshot job already exists but is paused, so resuming it avoids creating duplicate default jobs."
        verification = "Run `list scheduled jobs` and confirm State Snapshot is enabled."
        risk = "LOCAL_SAFE scheduler write only."
    elif not conversation_compaction_jobs:
        command = "schedule assistant basics"
        message = "Memory Trees compaction is not scheduled; use `schedule assistant basics`."
        action_kind = "schedule_basics"
        action_title = "Enable Memory Trees conversation compaction"
        rationale = "State Snapshot is present, but Conversation Compaction is missing, so old conversations will not become durable memory digests."
        verification = f"Run `list scheduled jobs` and confirm {COMPACTION_JOB_NAME} is enabled."
        risk = "LOCAL_SAFE scheduler writes only."
    elif not enabled_conversation_compaction_jobs:
        command = f"resume job {COMPACTION_JOB_NAME}"
        message = f"Memory Trees compaction is paused; use `resume job {COMPACTION_JOB_NAME}`."
        action_kind = "resume_conversation_compaction"
        action_title = "Resume Memory Trees conversation compaction"
        rationale = "A Conversation Compaction job exists but is paused, so resuming it keeps old conversations flowing into durable memory digests."
        verification = f"Run `list scheduled jobs` and confirm {COMPACTION_JOB_NAME} is enabled."
        risk = "LOCAL_SAFE scheduler write only."
    else:
        command = "schedule assistant basics"
        message = "Turn on basic safe background reviews with `schedule assistant basics`."
        action_kind = "schedule_basics"
        action_title = "Enable safe scheduled context refreshes"
        rationale = "The scheduled background rhythm is incomplete, so safe context refreshes would improve continuity."
        verification = "Run `list scheduled jobs` and confirm the safe jobs are enabled."
        risk = "LOCAL_SAFE scheduler writes only."
    return {
        "enabled_jobs": enabled_jobs,
        "state_snapshot_jobs": state_snapshot_jobs,
        "enabled_state_snapshot_jobs": enabled_state_snapshot_jobs,
        "conversation_compaction_jobs": conversation_compaction_jobs,
        "enabled_conversation_compaction_jobs": enabled_conversation_compaction_jobs,
        "command": command,
        "message": message,
        "action_kind": action_kind,
        "action_title": action_title,
        "rationale": rationale,
        "verification": verification,
        "risk": risk,
    }


def _ids_from_text(value: Any, markers: tuple[str, ...]) -> list[int]:
    text = str(value or "")
    ids: list[int] = []
    seen: set[int] = set()
    for marker in markers:
        marker_pattern = re.escape(marker).replace(r"\ ", r"\s+")
        pattern = re.compile(rf"\b{marker_pattern}\s*#?\s*(\d+)\b", re.IGNORECASE)
        for match in pattern.finditer(text):
            candidate = int(match.group(1))
            if candidate not in seen:
                ids.append(candidate)
                seen.add(candidate)
    return ids


def _row_metadata(row: Any) -> dict[str, Any]:
    try:
        if "metadata" not in row.keys():
            return {}
        parsed = json.loads(row["metadata"] or "{}")
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


APPROVAL_HOLD_METADATA_KEYS = (
    "requires_confirmation",
    "requires_approval",
    "approval_required",
)
APPROVAL_HOLD_VALUES = {
    "approval_required",
    "approval_gate",
    "approval_gated",
    "approval_held",
    "approval_hold",
    "explicit_approval_required",
    "confirmation_required",
    "requires_confirmation",
    "requires_approval",
}


def _metadata_truthy_loose(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on"}
    return False


def _normalized_metadata_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().casefold()).strip("_")


def _is_approval_hold_metadata(metadata: dict[str, Any]) -> bool:
    if any(_metadata_truthy_loose(metadata.get(key)) for key in APPROVAL_HOLD_METADATA_KEYS):
        return True
    for key in (
        "failure_kind",
        "failure_stage",
        "stage",
        "guard_reason",
        "reason",
        "status",
        "send_status",
        "call_status",
    ):
        token = _normalized_metadata_token(metadata.get(key))
        if token in APPROVAL_HOLD_VALUES:
            return True
        if "approval" in token and any(
            marker in token for marker in ("required", "requires", "gate", "gated", "hold", "held")
        ):
            return True
    return False


def _is_approval_held_tool_run(row: Any) -> bool:
    try:
        if bool(row["ok"]):
            return False
    except Exception:
        return False
    return _is_approval_hold_metadata(_row_metadata(row))


def _recent_tool_run_attention_buckets(rows: list[Any]) -> tuple[list[Any], list[Any]]:
    failed: list[Any] = []
    approval_held: list[Any] = []
    for row in rows:
        try:
            ok = bool(row["ok"])
        except Exception:
            continue
        if ok:
            continue
        if _is_approval_held_tool_run(row):
            approval_held.append(row)
        else:
                failed.append(row)
    return failed, approval_held


def _readable_tool_run_rows(rows: list[Any]) -> tuple[list[Any], int]:
    readable: list[Any] = []
    unreadable = 0
    for row in rows:
        try:
            if str(row["tool_name"] or "").strip():
                readable.append(row)
            else:
                unreadable += 1
        except Exception:
            unreadable += 1
    return readable, unreadable


def _approval_proof_chain_commands(approval_id: int) -> list[str]:
    return [
        f"approval readiness {approval_id}",
        f"approval packet {approval_id}",
        f"approval chain proof {approval_id}",
        f"verification receipt <approved run id from approval chain proof {approval_id}>",
    ]


def _append_unique(commands: list[str], command: str) -> None:
    if command not in commands:
        commands.append(command)


def _recovery_closure_checklist_command(health: dict[str, Any]) -> str:
    if health.get("recovery_closure_blocks_auto_execution"):
        return "recovery closure checklist"
    return ""


def _doctor_default_agi_handoff(list_tools: Any | None = None) -> dict[str, Any]:
    selected_gate = "personal integrations"
    selected_readiness: dict[str, Any] = {}
    gate_summary = _agi_gate_summary(set())
    selection_source = "next_step_doctor_default_gate"
    selection_reason = "Next-step doctor metadata is using its isolated fallback AGI handoff track."
    if list_tools is not None:
        try:
            tool_names = {str(tool.name) for tool in list_tools()}
            gate_summary = _agi_gate_summary(tool_names)
            selected_readiness = _selected_agi_target_readiness(gate_summary)
            selected_gate = str(selected_readiness.get("gate_name") or selected_gate)
            selection_source = "harness_dynamic_registry"
            selection_reason = "Next-step doctor metadata mirrors the current harness AGI selector from registry evidence."
        except Exception:
            selected_readiness = {}
            selection_source = "next_step_doctor_default_gate_after_selector_error"
            selection_reason = "Next-step doctor metadata fell back after registry-evidence selection failed."
    if selected_gate not in AGI_GATE_BUILD_TARGETS and AGI_GATE_BUILD_TARGETS:
        selected_gate = next(iter(AGI_GATE_BUILD_TARGETS))
    target = AGI_GATE_BUILD_TARGETS.get(selected_gate, {})
    likely_files = list(selected_readiness.get("likely_files") or target.get("files") or [])
    file_integrity = selected_readiness.get("file_integrity") or _target_file_integrity(likely_files)
    evidence_closure_commands = list(selected_readiness.get("closure_commands") or [])
    if not evidence_closure_commands:
        evidence_closure_commands = [
            f"agi next build move: {selected_gate}",
            f"completion audit: improve AGI gate {selected_gate}",
            "evidence ledger",
            f"completion claim gate: improve AGI gate {selected_gate}",
        ]
    focused_verification_commands = list(selected_readiness.get("verification_commands") or target.get("tests") or [])
    acceptance_checks = list(selected_readiness.get("acceptance_checks") or target.get("acceptance") or [])
    return {
        "gate": selected_gate,
        "selection_source": selection_source,
        "selection_reason": selection_reason,
        "target_title": str(selected_readiness.get("target_title") or target.get("title") or ""),
        "build_command": evidence_closure_commands[0],
        "evidence_closure_commands": evidence_closure_commands,
        "evidence_closure_command_count": len(evidence_closure_commands),
        "focused_verification_commands": focused_verification_commands,
        "focused_verification_command_count": len(focused_verification_commands),
        "likely_files": likely_files,
        "likely_file_count": len(likely_files),
        "target_file_integrity_status": file_integrity["status"],
        "target_files_checked": file_integrity["checked"],
        "target_files_exist": file_integrity["all_exist"],
        "missing_target_files": file_integrity["missing"],
        "missing_target_file_count": file_integrity["missing_count"],
        "target_integrity_blocks_start": not file_integrity["all_exist"],
        "acceptance_checks": acceptance_checks,
        "acceptance_check_count": len(acceptance_checks),
        "build_packet_ready_for_review": bool(file_integrity["all_exist"] and focused_verification_commands and acceptance_checks),
        "real_execution_gap_count": int(gate_summary.get("real_execution_gap_count") or 0),
        "real_execution_gaps_by_gate": dict(gate_summary.get("real_execution_gaps_by_gate") or {}),
        "selected_real_execution_gap": str((gate_summary.get("real_execution_gaps_by_gate") or {}).get(selected_gate) or ""),
    }


def _doctor_agi_handoff_metadata(snapshot: dict[str, Any]) -> dict[str, Any]:
    agi = snapshot.get("agi_next") or {}
    return {
        "doctor_agi_next_gate": agi.get("gate", ""),
        "doctor_agi_next_selection_source": agi.get("selection_source", ""),
        "doctor_agi_next_selection_reason": agi.get("selection_reason", ""),
        "doctor_agi_next_canonical_selector_command": agi.get("canonical_selector_command") or agi.get("build_command") or agi.get("next_build_command") or "",
        "doctor_agi_next_deliberate_focus_override": bool(agi.get("deliberate_focus_override")),
        "doctor_agi_next_target_title": agi.get("target_title", ""),
        "doctor_agi_next_build_command": agi.get("build_command", ""),
        "doctor_agi_next_evidence_closure_commands": agi.get("evidence_closure_commands", []),
        "doctor_agi_next_evidence_closure_command_count": agi.get("evidence_closure_command_count", 0),
        "doctor_agi_next_focused_verification_commands": agi.get("focused_verification_commands", []),
        "doctor_agi_next_focused_verification_command_count": agi.get("focused_verification_command_count", 0),
        "doctor_agi_next_likely_files": agi.get("likely_files", []),
        "doctor_agi_next_likely_file_count": agi.get("likely_file_count", 0),
        "doctor_agi_next_target_file_integrity_status": agi.get("target_file_integrity_status", ""),
        "doctor_agi_next_target_files_checked": agi.get("target_files_checked", 0),
        "doctor_agi_next_target_files_exist": agi.get("target_files_exist", False),
        "doctor_agi_next_missing_target_files": agi.get("missing_target_files", []),
        "doctor_agi_next_missing_target_file_count": agi.get("missing_target_file_count", 0),
        "doctor_agi_next_target_integrity_blocks_start": agi.get("target_integrity_blocks_start", True),
        "doctor_agi_next_acceptance_checks": agi.get("acceptance_checks", []),
        "doctor_agi_next_acceptance_check_count": agi.get("acceptance_check_count", 0),
        "doctor_agi_next_build_packet_ready_for_review": agi.get("build_packet_ready_for_review", False),
        "doctor_agi_next_real_execution_gap_count": agi.get("real_execution_gap_count", 0),
        "doctor_agi_next_real_execution_gaps_by_gate": agi.get("real_execution_gaps_by_gate", {}),
        "doctor_agi_next_selected_real_execution_gap": _safe_text(agi.get("selected_real_execution_gap") or ""),
    }


def _doctor_execution_learning_handoff_metadata(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "doctor_execution_learning_state": snapshot.get("execution_learning_state", ""),
        "doctor_execution_learning_missing": snapshot.get("execution_learning_missing", []),
        "doctor_execution_learning_missing_count": snapshot.get("execution_learning_missing_count", 0),
        "doctor_execution_learning_required_commands": snapshot.get("execution_learning_required_commands", []),
        "doctor_execution_learning_next_required_command": snapshot.get("execution_learning_next_required_command", ""),
        "doctor_execution_learning_proof_queue": snapshot.get("execution_learning_proof_queue", []),
        "doctor_execution_learning_proof_queue_count": snapshot.get("execution_learning_proof_queue_count", 0),
        "doctor_execution_learning_next_proof_command": snapshot.get("execution_learning_next_proof_command", ""),
        "doctor_execution_learning_actionable_required_commands": snapshot.get(
            "execution_learning_actionable_required_commands", []
        ),
        "doctor_execution_learning_actionable_required_command_count": snapshot.get(
            "execution_learning_actionable_required_command_count", 0
        ),
        "doctor_execution_learning_actionable_next_required_command": snapshot.get(
            "execution_learning_actionable_next_required_command", ""
        ),
        "doctor_execution_learning_actionable_proof_queue": snapshot.get(
            "execution_learning_actionable_proof_queue", []
        ),
        "doctor_execution_learning_actionable_proof_queue_count": snapshot.get(
            "execution_learning_actionable_proof_queue_count", 0
        ),
        "doctor_execution_learning_actionable_next_proof_command": snapshot.get(
            "execution_learning_actionable_next_proof_command", ""
        ),
        "doctor_execution_learning_next_evidence_command": snapshot.get("execution_learning_next_evidence_command", ""),
        "doctor_execution_learning_blocks_completion_claim": snapshot.get(
            "execution_learning_blocks_completion_claim", False
        ),
    }


def _doctor_audit_readability_handoff_metadata(snapshot: dict[str, Any]) -> dict[str, Any]:
    commands = [_safe_text(command) for command in snapshot.get("audit_readability_review_commands", [])]
    return {
        "doctor_audit_readability_review_required": _metadata_bool(
            snapshot.get("audit_readability_review_required")
        ),
        "doctor_audit_readability_review_commands": commands,
        "doctor_audit_readability_review_command_count": snapshot.get(
            "audit_readability_review_command_count", len(commands)
        ),
        "doctor_audit_readability_review_next_command": _safe_text(
            snapshot.get("audit_readability_review_next_command") or ""
        ),
        "doctor_unreadable_recent_tool_run_rows": int(snapshot.get("unreadable_recent_tool_run_rows") or 0),
    }


def _storage_completion_handoff(
    storage_fallback: Any | None = None,
    config: JarvisConfig | None = None,
) -> dict[str, Any]:
    fallback: dict[str, Any] | None = None
    if storage_fallback is not None:
        try:
            fallback = storage_fallback()
        except Exception:
            fallback = None
    fallback_active = bool(fallback)
    diagnostics, _diagnostics_source = _configured_storage_diagnostics(config, fallback)
    configured_ready = _metadata_bool(diagnostics.get("available"))
    if fallback_active and configured_ready:
        recovery_mode = "restart_runtime_to_configured_storage"
        recovery_next_operator_action = (
            "restart or reload Jarvis with the configured durable storage envs, then run `storage status`"
        )
        recovery_restart_required = True
        recovery_commands = [
            "storage status",
            STORAGE_RECOVERY_CHECK_COMMAND,
            "storage status",
        ]
        blocker = "runtime is using workspace-local fallback storage"
    elif fallback_active:
        recovery_mode = "repair_configured_storage_then_restart_runtime"
        recovery_next_operator_action = (
            "review the storage recovery plan, point Jarvis at writable durable storage, run the no-write storage check, then restart or reload Jarvis"
        )
        recovery_restart_required = True
        recovery_commands = [
            "storage status",
            STORAGE_RECOVERY_PLAN_COMMAND,
            STORAGE_RECOVERY_CHECK_COMMAND,
            BOOTSTRAP_CHECK_COMMAND,
            BOOTSTRAP_WRITE_COMMAND,
        ]
        blocker = "primary storage is not writable; runtime is using workspace-local fallback memory"
    elif not configured_ready:
        recovery_mode = "repair_configured_storage"
        recovery_next_operator_action = (
            "review the storage recovery plan, point Jarvis at writable durable storage, and run the no-write storage check"
        )
        recovery_restart_required = False
        recovery_commands = [
            "storage status",
            STORAGE_RECOVERY_PLAN_COMMAND,
            STORAGE_RECOVERY_CHECK_COMMAND,
            BOOTSTRAP_CHECK_COMMAND,
            BOOTSTRAP_WRITE_COMMAND,
        ]
        issues = list(diagnostics.get("issues") or ["configured storage needs attention"])
        blocker = "; ".join(str(issue) for issue in issues)
    else:
        recovery_mode = "none"
        recovery_next_operator_action = ""
        recovery_restart_required = False
        recovery_commands = []
        blocker = ""
    recovery_required = fallback_active or not configured_ready
    issues = [str(issue) for issue in diagnostics.get("issues", [])]
    next_proof_command = recovery_commands[0] if recovery_commands else ""
    handoff = {
        "source": "next_step_storage",
        "handoff_ready": True,
        "storage_handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "active": fallback_active,
        "blocks_completion_claim": recovery_required,
        "recovery_required": recovery_required,
        "reason": str((fallback or {}).get("reason") or ""),
        "exception_type": str((fallback or {}).get("exception_type") or ""),
        "db_path_display": str((fallback or {}).get("db_path_display") or ""),
        "vault_path_display": str((fallback or {}).get("vault_path_display") or ""),
        "blocker": blocker,
        "recovery_reason": blocker,
        "recovery_mode": recovery_mode,
        "recovery_next_operator_action": recovery_next_operator_action,
        "recovery_restart_required": recovery_restart_required,
        "recovery_check_tool_command": STORAGE_RECOVERY_CHECK_COMMAND if recovery_required else "",
        "recovery_check_command": BOOTSTRAP_CHECK_COMMAND if recovery_required else "",
        "recovery_check_api": STORAGE_RECOVERY_CHECK_API if recovery_required else "",
        "recovery_command": BOOTSTRAP_WRITE_COMMAND if recovery_required else "",
        "next_commands": recovery_commands,
        "next_command_count": len(recovery_commands),
        "next_proof_command": next_proof_command,
        "storage_status": str(diagnostics.get("status") or "unknown"),
        "storage_available": configured_ready and not fallback_active,
        "storage_runtime_fallback_active": fallback_active,
        "storage_runtime_fallback_reason": str((fallback or {}).get("reason") or ""),
        "storage_runtime_fallback_exception_type": str((fallback or {}).get("exception_type") or ""),
        "storage_runtime_fallback_db_path_display": str((fallback or {}).get("db_path_display") or ""),
        "storage_runtime_fallback_vault_path_display": str((fallback or {}).get("vault_path_display") or ""),
        "storage_readiness_blocks_completion_claim": recovery_required,
        "storage_readiness_blocker": blocker,
        "storage_readiness_next_commands": recovery_commands,
        "storage_readiness_next_command_count": len(recovery_commands),
        "storage_readiness_next_required_command": next_proof_command,
        "storage_readiness_next_proof_command": next_proof_command,
        "storage_readiness_proof_queue": recovery_commands,
        "storage_readiness_proof_queue_count": len(recovery_commands),
        "storage_readiness_first_proof_command": next_proof_command,
        "storage_recovery_required": recovery_required,
        "storage_recovery_reason": blocker,
        "storage_recovery_mode": recovery_mode,
        "storage_recovery_next_operator_action": recovery_next_operator_action,
        "storage_recovery_restart_required": recovery_restart_required,
        "storage_recovery_check_tool_command": STORAGE_RECOVERY_CHECK_COMMAND if recovery_required else "",
        "storage_recovery_check_command": BOOTSTRAP_CHECK_COMMAND if recovery_required else "",
        "storage_recovery_check_api": STORAGE_RECOVERY_CHECK_API if recovery_required else "",
        "storage_recovery_command": BOOTSTRAP_WRITE_COMMAND if recovery_required else "",
        "storage_issues": issues,
        "storage_issue_count": len(issues),
        "selection_source": (
            "runtime_storage_fallback"
            if fallback_active
            else "configured_storage_needs_attention"
            if not configured_ready
            else "configured_storage"
        ),
        "calls_model": False,
        "executes_tools": False,
        "queues_approval": False,
        "requires_approval": False,
        "approval_granted": False,
        "approves_request": False,
        "dismisses_request": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "writes_files": False,
        "writes_database": False,
        "writes_notes": False,
        "writes_memory": False,
        "controls_computer": False,
        "calls_external_service": False,
        "external_side_effect": False,
        "speaks": False,
        "completes_tasks": False,
        "boundaries": {
            "metadata_only": True,
            "reads_database_file": False,
            "reads_db_file_contents": False,
            "reads_vault_files": False,
            "scans_obsidian_vault": False,
            "writes_files": False,
            "writes_database": False,
            "writes_notes": False,
            "writes_memory": False,
            "queues_approval": False,
            "approves_requests": False,
            "dismisses_approvals": False,
            "controls_computer": False,
            "calls_external_service": False,
            "executes_tools": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
        },
    }
    return {
        "active": fallback_active,
        "blocks_completion_claim": recovery_required,
        "recovery_required": recovery_required,
        "reason": str((fallback or {}).get("reason") or ""),
        "exception_type": str((fallback or {}).get("exception_type") or ""),
        "db_path_display": str((fallback or {}).get("db_path_display") or ""),
        "vault_path_display": str((fallback or {}).get("vault_path_display") or ""),
        "blocker": blocker,
        "recovery_reason": blocker,
        "recovery_mode": recovery_mode,
        "recovery_next_operator_action": recovery_next_operator_action,
        "recovery_restart_required": recovery_restart_required,
        "recovery_check_tool_command": STORAGE_RECOVERY_CHECK_COMMAND if recovery_required else "",
        "recovery_check_command": BOOTSTRAP_CHECK_COMMAND if recovery_required else "",
        "recovery_check_api": STORAGE_RECOVERY_CHECK_API if recovery_required else "",
        "recovery_command": BOOTSTRAP_WRITE_COMMAND if recovery_required else "",
        "next_commands": recovery_commands,
        "next_command_count": len(recovery_commands),
        "next_proof_command": next_proof_command,
        "storage_readiness_next_required_command": next_proof_command,
        "storage_readiness_next_proof_command": next_proof_command,
        "storage_readiness_proof_queue": recovery_commands,
        "storage_readiness_proof_queue_count": len(recovery_commands),
        "storage_readiness_first_proof_command": next_proof_command,
        "storage_issues": issues,
        "storage_issue_count": len(issues),
        "selection_source": handoff["selection_source"],
        "storage_handoff": handoff,
    }


def _doctor_storage_handoff_metadata(doctor: dict[str, Any]) -> dict[str, Any]:
    commands = [_safe_text(command) for command in doctor.get("storage_readiness_next_commands", [])]
    proof_queue = [_safe_text(command) for command in doctor.get("storage_readiness_proof_queue", commands)]
    next_proof = _safe_text(
        doctor.get("storage_readiness_next_proof_command")
        or doctor.get("storage_next_proof_command")
        or (proof_queue[0] if proof_queue else "")
    )
    next_required = _safe_text(doctor.get("storage_readiness_next_required_command") or next_proof)
    return {
        "doctor_storage_readiness_next_required_command": next_required,
        "doctor_storage_readiness_next_proof_command": next_proof,
        "doctor_storage_readiness_proof_queue": proof_queue,
        "doctor_storage_readiness_proof_queue_count": len(proof_queue),
        "doctor_storage_readiness_first_proof_command": _safe_text(
            doctor.get("storage_readiness_first_proof_command") or (proof_queue[0] if proof_queue else "")
        ),
        "doctor_storage_issues": [str(issue) for issue in doctor.get("storage_issues", [])],
        "doctor_storage_issue_count": int(doctor.get("storage_issue_count", len(doctor.get("storage_issues", [])))),
        "doctor_storage_handoff": doctor.get("storage_handoff", {}),
    }


def _continuation_packet_handoff(
    *,
    objective: str,
    next_focus: str,
    approvals: list[Any],
    tasks: list[Any],
    goals: list[Any],
    enabled_jobs: list[dict[str, Any]],
    state_snapshot_jobs: list[dict[str, Any]],
    enabled_state_snapshot_jobs: list[dict[str, Any]],
    disabled_state_snapshot_jobs: int,
    scheduler_next_command: str,
    health: dict[str, Any],
    checkpoint_contract: dict[str, Any],
) -> dict[str, Any]:
    execution_health_commands = [str(command) for command in health.get("next_commands", [])]
    checkpoint_commands = [str(command) for command in checkpoint_contract.get("checkpoint_recovery_queue", [])]
    next_commands = []
    if health.get("next_command"):
        next_commands.append(str(health["next_command"]))
    if checkpoint_contract.get("checkpoint_recovery_next_command"):
        next_commands.append(str(checkpoint_contract["checkpoint_recovery_next_command"]))
    next_commands.extend(
        [
            f"build target packet: {_safe_text(objective, limit=MAX_OBJECTIVE_CHARS)}",
            "acceptance gate: <changed behavior>; evidence <receipt>; tests <verification>; recovery <rollback or stop condition>",
        ]
    )
    deduped_next_commands: list[str] = []
    for command in next_commands:
        command = _safe_text(command, limit=MAX_NEXT_STEP_TEXT_CHARS)
        if command and command not in deduped_next_commands:
            deduped_next_commands.append(command)
    return {
        "source": "continuation_packet",
        "status": "ready",
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "objective": _safe_text(objective, limit=MAX_OBJECTIVE_CHARS),
        "recommended_focus": _safe_text(next_focus),
        "pending_approvals": len(approvals),
        "open_tasks": len(tasks),
        "active_goals": len(goals),
        "enabled_jobs": len(enabled_jobs),
        "state_snapshot_jobs": len(state_snapshot_jobs),
        "enabled_state_snapshot_jobs": len(enabled_state_snapshot_jobs),
        "disabled_state_snapshot_jobs": int(disabled_state_snapshot_jobs),
        "scheduler_next_command": _safe_text(scheduler_next_command),
        "execution_health_review_required": _metadata_bool(health.get("review_required")),
        "execution_health_next_command": _safe_text(health.get("next_command") or ""),
        "execution_health_next_commands": [_safe_text(command) for command in execution_health_commands],
        "execution_health_next_command_count": len(execution_health_commands),
        "execution_health_blocker_categories": [str(item) for item in health.get("blocker_categories", [])],
        "execution_health_blocker_count": int(health.get("blocker_count", 0)),
        "execution_health_verification_coverage": str(health.get("verification_coverage_state") or ""),
        "checkpoint_found": _metadata_bool(checkpoint_contract.get("checkpoint_found")),
        "checkpoint_path_display": _safe_text(checkpoint_contract.get("checkpoint_path_display") or ""),
        "checkpoint_freshness": str(checkpoint_contract.get("checkpoint_freshness") or ""),
        "checkpoint_recovery_required": _metadata_bool(checkpoint_contract.get("checkpoint_recovery_required")),
        "checkpoint_recovery_next_command": _safe_text(checkpoint_contract.get("checkpoint_recovery_next_command") or ""),
        "checkpoint_recovery_queue": [_safe_text(command) for command in checkpoint_commands],
        "checkpoint_recovery_queue_count": len(checkpoint_commands),
        "checkpoint_recovery_verification_target": _safe_text(checkpoint_contract.get("checkpoint_recovery_verification_target") or ""),
        "checkpoint_recovery_stop_condition": _safe_text(checkpoint_contract.get("checkpoint_recovery_stop_condition") or ""),
        "next_commands": deduped_next_commands,
        "next_safe_commands": deduped_next_commands,
        "next_command_count": len(deduped_next_commands),
        "next_safe_command_count": len(deduped_next_commands),
        "boundaries": {
            "read_only": True,
            "calls_model": False,
            "executes_tools": False,
            "queues_approval": False,
            "writes_files": False,
            "writes_database": False,
            "writes_memory": False,
            "writes_notes": False,
            "edits_files": False,
            "controls_computer": False,
            "reads_personal_data": False,
            "reads_private_data": False,
            "executes_side_effect": False,
            "external_side_effect": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
        },
    }


def _build_target_packet_handoff(
    *,
    objective: str,
    target_kind: str,
    target_title: str,
    target_reason: str,
    first_boundary: str,
    likely_files: list[str],
    file_integrity: dict[str, Any],
    focused_verification_commands: list[str],
    aggregate_verification_command: str,
    acceptance_gate_command: str,
    approvals: list[Any],
    tasks: list[Any],
    goals: list[Any],
    enabled_jobs: list[dict[str, Any]],
    state_snapshot_jobs: list[dict[str, Any]],
    enabled_state_snapshot_jobs: list[dict[str, Any]],
    disabled_state_snapshot_jobs: int,
    scheduler_next_command: str,
    health: dict[str, Any],
) -> dict[str, Any]:
    execution_health_commands = [str(command) for command in health.get("next_commands", [])]
    next_commands = [
        focused_verification_commands[0] if focused_verification_commands else "",
        aggregate_verification_command,
        acceptance_gate_command,
        f"save build target packet: {_safe_text(objective, limit=MAX_OBJECTIVE_CHARS)}",
    ]
    deduped_next_commands: list[str] = []
    for command in next_commands:
        command = _safe_text(command, limit=MAX_NEXT_STEP_TEXT_CHARS)
        if command and command not in deduped_next_commands:
            deduped_next_commands.append(command)
    return {
        "source": "build_target_packet",
        "status": "ready",
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "objective": _safe_text(objective, limit=MAX_OBJECTIVE_CHARS),
        "target_kind": _safe_text(target_kind, limit=120),
        "target_title": _safe_text(target_title),
        "target_reason": _safe_text(target_reason),
        "first_boundary": _safe_text(first_boundary),
        "likely_files": [_safe_text(path, limit=160) for path in likely_files],
        "likely_file_count": len(likely_files),
        "target_file_integrity_status": str(file_integrity.get("status") or ""),
        "target_files_checked": int(file_integrity.get("checked", 0)),
        "target_files_exist": bool(file_integrity.get("all_exist")),
        "missing_target_files": [_safe_text(path, limit=160) for path in file_integrity.get("missing", [])],
        "missing_target_file_count": int(file_integrity.get("missing_count", 0)),
        "target_file_rows": file_integrity.get("rows", []),
        "focused_verification_commands": [_safe_text(command) for command in focused_verification_commands],
        "focused_verification_command_count": len(focused_verification_commands),
        "aggregate_verification_command": _safe_text(aggregate_verification_command),
        "acceptance_gate_command": _safe_text(acceptance_gate_command),
        "pending_approvals": len(approvals),
        "open_tasks": len(tasks),
        "active_goals": len(goals),
        "enabled_jobs": len(enabled_jobs),
        "state_snapshot_jobs": len(state_snapshot_jobs),
        "enabled_state_snapshot_jobs": len(enabled_state_snapshot_jobs),
        "disabled_state_snapshot_jobs": int(disabled_state_snapshot_jobs),
        "scheduler_next_command": _safe_text(scheduler_next_command),
        "execution_health_review_required": _metadata_bool(health.get("review_required")),
        "execution_health_next_command": _safe_text(health.get("next_command") or ""),
        "execution_health_next_commands": [_safe_text(command) for command in execution_health_commands],
        "execution_health_next_command_count": len(execution_health_commands),
        "execution_health_blocker_categories": [str(item) for item in health.get("blocker_categories", [])],
        "execution_health_blocker_count": int(health.get("blocker_count", 0)),
        "execution_health_verification_coverage": str(health.get("verification_coverage_state") or ""),
        "next_commands": deduped_next_commands,
        "next_safe_commands": deduped_next_commands,
        "next_command_count": len(deduped_next_commands),
        "next_safe_command_count": len(deduped_next_commands),
        "boundaries": {
            "read_only": True,
            "calls_model": False,
            "executes_tools": False,
            "queues_approval": False,
            "writes_files": False,
            "writes_database": False,
            "writes_memory": False,
            "writes_notes": False,
            "edits_files": False,
            "controls_computer": False,
            "reads_personal_data": False,
            "reads_private_data": False,
            "executes_side_effect": False,
            "external_side_effect": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
        },
    }


def _work_queue_handoff(
    *,
    limit: int,
    approvals: list[Any],
    tasks: list[Any],
    goals: list[Any],
    decisions: list[Any],
    background_state: dict[str, Any],
    health: dict[str, Any],
    doctor: dict[str, Any],
    approval_handoff: dict[str, Any],
) -> dict[str, Any]:
    start_kind = "capture_task"
    start_command = "add task <next concrete Jarvis task>"
    start_label = "Capture one concrete task before reactive work."
    if approvals:
        approval_id = int(approvals[0]["id"])
        start_kind = "approval_review"
        start_command = str(approval_handoff.get("next_command") or f"approval readiness {approval_id}")
        start_label = f"Review approval #{approval_id} before risky work."
    elif health.get("review_required"):
        start_kind = "execution_health"
        start_command = _safe_text(health.get("next_command") or "execution health report")
        start_label = "Review failed or weakly evidenced execution before new work."
    elif tasks:
        start_kind = "task"
        start_command = f"complete task {tasks[0]['id']}"
        start_label = f"Work one visible step on task #{tasks[0]['id']}."
    elif goals:
        start_kind = "goal"
        start_command = f"show goal {goals[0]['id']}"
        start_label = f"Advance one visible step on goal #{goals[0]['id']}."
    elif background_state.get("command"):
        start_kind = str(background_state.get("action_kind") or "background_upkeep")
        start_command = _safe_text(background_state["command"])
        start_label = _safe_text(background_state.get("action_title") or background_state.get("message") or "Review background upkeep.")
    if doctor.get("storage_readiness_blocks_completion_claim"):
        start_kind = "storage_recovery"
        start_command = _safe_text(
            doctor.get("storage_readiness_next_required_command")
            or doctor.get("storage_next_proof_command")
            or "storage status"
        )
        start_label = "Check durable storage recovery before completion or autonomy claims."

    next_commands = ["handoff brief", "catch me up", "safety status"]
    next_commands.extend(str(command) for command in doctor.get("storage_readiness_next_commands", []))
    next_commands.append(start_command)
    if approval_handoff.get("readiness_command"):
        next_commands.extend(
            [
                str(approval_handoff["readiness_command"]),
                str(approval_handoff["last_look_command"]),
                str(approval_handoff["proof_command"]),
            ]
        )
    if health.get("next_command"):
        next_commands.append(str(health["next_command"]))
    next_commands.extend(str(command) for command in health.get("next_commands", []))
    if doctor.get("recovery_closure_checklist_command"):
        next_commands.append(str(doctor["recovery_closure_checklist_command"]))
    learning_next = doctor.get("execution_learning_actionable_next_proof_command") or doctor.get("execution_learning_next_proof_command")
    if learning_next:
        next_commands.append(str(learning_next))
    next_commands.extend(["focus brief", "safe next actions", "save handoff brief"])
    deduped_next_commands: list[str] = []
    for command in next_commands:
        command = _safe_text(command)
        if command and command not in deduped_next_commands:
            deduped_next_commands.append(command)

    return {
        "source": "work_queue",
        "status": "ready",
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "start_kind": start_kind,
        "start_command": start_command,
        "start_label": _safe_text(start_label),
        "queue_order": [
            "start safely",
            "approval blockers",
            "execution health",
            "open tasks",
            "goal steps",
            "background upkeep",
        ],
        "next_commands": deduped_next_commands,
        "next_command_count": len(deduped_next_commands),
        "next_safe_commands": deduped_next_commands,
        "next_safe_command_count": len(deduped_next_commands),
        "pending_approvals": len(approvals),
        "open_tasks": len(tasks),
        "active_goals": len(goals),
        "active_decisions": len(decisions),
        "enabled_jobs": len(background_state["enabled_jobs"]),
        "state_snapshot_jobs": len(background_state["state_snapshot_jobs"]),
        "enabled_state_snapshot_jobs": len(background_state["enabled_state_snapshot_jobs"]),
        "disabled_state_snapshot_jobs": len(background_state["state_snapshot_jobs"]) - len(background_state["enabled_state_snapshot_jobs"]),
        "scheduler_next_command": _safe_text(background_state["command"]),
        "failed_action_runs": int(health.get("failed_action_runs") or 0),
        "approval_held_action_runs": int(health.get("approval_held_action_runs") or 0),
        "execution_health_approval_held_action_runs": int(health.get("approval_held_action_runs") or 0),
        "execution_health_review_required": _metadata_bool(health.get("review_required")),
        "execution_health_next_command": _safe_text(health.get("next_command") or ""),
        "execution_health_next_commands": [_safe_text(command) for command in health.get("next_commands", [])],
        "execution_health_next_command_count": len(health.get("next_commands", [])),
        "execution_health_blocker_categories": [str(item) for item in health.get("blocker_categories", [])],
        "execution_health_blocker_count": int(health.get("blocker_count") or 0),
        "execution_health_verification_coverage": str(health.get("verification_coverage_state") or ""),
        "doctor_next_audit_command": _safe_text(doctor.get("next_audit_command") or ""),
        "doctor_completion_claim_state": str(doctor.get("completion_claim_state") or ""),
        "doctor_completion_claim_ready": _metadata_bool(doctor.get("completion_claim_ready")),
        "doctor_completion_blockers": [str(item) for item in doctor.get("completion_blockers", [])],
        "doctor_completion_blocker_count": int(doctor.get("completion_blocker_count") or 0),
        "doctor_recent_failed_runs": int(doctor.get("recent_failed_runs") or 0),
        "doctor_recent_approval_held_runs": int(doctor.get("recent_approval_held_runs") or 0),
        **_doctor_audit_readability_handoff_metadata(doctor),
        "doctor_storage_runtime_fallback_active": _metadata_bool(doctor.get("storage_runtime_fallback_active")),
        "doctor_storage_readiness_blocks_completion_claim": _metadata_bool(
            doctor.get("storage_readiness_blocks_completion_claim")
        ),
        "doctor_storage_recovery_required": _metadata_bool(doctor.get("storage_recovery_required")),
        "doctor_storage_recovery_reason": _safe_text(doctor.get("storage_recovery_reason") or ""),
        "doctor_storage_recovery_mode": _safe_text(doctor.get("storage_recovery_mode") or ""),
        "doctor_storage_recovery_next_operator_action": _safe_text(doctor.get("storage_recovery_next_operator_action") or ""),
        "doctor_storage_recovery_restart_required": _metadata_bool(doctor.get("storage_recovery_restart_required")),
        "doctor_storage_recovery_check_tool_command": _safe_text(doctor.get("storage_recovery_check_tool_command") or ""),
        "doctor_storage_recovery_check_command": _safe_text(doctor.get("storage_recovery_check_command") or ""),
        "doctor_storage_recovery_check_api": _safe_text(doctor.get("storage_recovery_check_api") or STORAGE_RECOVERY_CHECK_API),
        "doctor_storage_recovery_command": _safe_text(doctor.get("storage_recovery_command") or ""),
        "doctor_storage_readiness_blocker": _safe_text(doctor.get("storage_readiness_blocker") or ""),
        "doctor_storage_readiness_next_commands": [_safe_text(command) for command in doctor.get("storage_readiness_next_commands", [])],
        "doctor_storage_readiness_next_command_count": len(doctor.get("storage_readiness_next_commands", [])),
        **_doctor_storage_handoff_metadata(doctor),
        "doctor_recovery_closure_state": str(doctor.get("recovery_closure_state") or ""),
        "doctor_recovery_closure_next_required_command": _safe_text(doctor.get("recovery_closure_next_required_command") or ""),
        "doctor_recovery_closure_next_proof_command": _safe_text(doctor.get("recovery_closure_next_proof_command") or ""),
        "doctor_execution_learning_state": str(doctor.get("execution_learning_state") or ""),
        "doctor_execution_learning_next_required_command": _safe_text(doctor.get("execution_learning_next_required_command") or ""),
        "doctor_execution_learning_next_proof_command": _safe_text(doctor.get("execution_learning_next_proof_command") or ""),
        "doctor_execution_learning_actionable_next_required_command": _safe_text(
            doctor.get("execution_learning_actionable_next_required_command") or ""
        ),
        "doctor_execution_learning_actionable_next_proof_command": _safe_text(
            doctor.get("execution_learning_actionable_next_proof_command") or ""
        ),
        "doctor_execution_learning_next_evidence_command": _safe_text(
            doctor.get("execution_learning_next_evidence_command") or ""
        ),
        "doctor_agi_next_gate": str(doctor.get("agi_next", {}).get("gate") or ""),
        "doctor_agi_next_selection_source": str(doctor.get("agi_next", {}).get("selection_source") or ""),
        "doctor_agi_next_selection_reason": _safe_text(doctor.get("agi_next", {}).get("selection_reason") or ""),
        "doctor_agi_next_canonical_selector_command": _safe_text(doctor.get("agi_next", {}).get("canonical_selector_command") or doctor.get("agi_next", {}).get("build_command") or doctor.get("agi_next", {}).get("next_build_command") or ""),
        "doctor_agi_next_deliberate_focus_override": _metadata_bool(
            doctor.get("agi_next", {}).get("deliberate_focus_override")
        ),
        "doctor_agi_next_target_title": _safe_text(doctor.get("agi_next", {}).get("target_title") or ""),
        "doctor_agi_next_build_command": _safe_text(doctor.get("agi_next", {}).get("build_command") or ""),
        "doctor_agi_next_real_execution_gap_count": int(doctor.get("agi_next", {}).get("real_execution_gap_count") or 0),
        "doctor_agi_next_real_execution_gaps_by_gate": dict(doctor.get("agi_next", {}).get("real_execution_gaps_by_gate") or {}),
        "doctor_agi_next_selected_real_execution_gap": _safe_text(doctor.get("agi_next", {}).get("selected_real_execution_gap") or ""),
        "approval_handoff": approval_handoff,
        "approval_handoff_pending_count": approval_handoff.get("pending_count", 0),
        "approval_handoff_first_id": approval_handoff.get("first_id"),
        "approval_handoff_next_command": _safe_text(approval_handoff.get("next_command") or ""),
        "approval_handoff_readiness_command": _safe_text(approval_handoff.get("readiness_command") or ""),
        "approval_handoff_last_look_command": _safe_text(approval_handoff.get("last_look_command") or ""),
        "approval_handoff_proof_command": _safe_text(approval_handoff.get("proof_command") or ""),
        "approval_handoff_proof_chain_commands": [_safe_text(command) for command in approval_handoff.get("proof_chain_commands", [])],
        "limit": limit,
        "boundaries": {
            "read_only": True,
            **NEXT_STEP_BOUNDARY_FALSE_FLAGS,
        },
    }


def _safe_next_actions_handoff(
    *,
    limit: int,
    approvals: list[Any],
    tasks: list[Any],
    goals: list[Any],
    decisions: list[Any],
    preferences: list[Any],
    background_state: dict[str, Any],
    health: dict[str, Any],
    doctor: dict[str, Any],
    approval_handoff: dict[str, Any],
) -> dict[str, Any]:
    start_kind = "review"
    start_command = "catch me up"
    start_label = "Review context before choosing the next action."
    if approvals:
        approval_id = int(approvals[0]["id"])
        start_kind = "approval_review"
        start_command = str(approval_handoff.get("next_command") or f"approval readiness {approval_id}")
        start_label = f"Review approval #{approval_id} before any risky continuation."
    elif health.get("review_required"):
        start_kind = "execution_health"
        start_command = _safe_text(health.get("next_command") or "execution health report")
        start_label = "Close execution-health proof debt before fresh work."
    elif tasks:
        start_kind = "task"
        start_command = f"complete task {tasks[0]['id']}"
        start_label = f"Advance the top open task #{tasks[0]['id']}."
    elif goals:
        start_kind = "goal"
        start_command = f"show goal {goals[0]['id']}"
        start_label = f"Advance the first active goal #{goals[0]['id']}."
    elif background_state.get("command"):
        start_kind = str(background_state.get("action_kind") or "background_upkeep")
        start_command = _safe_text(background_state["command"])
        start_label = _safe_text(background_state.get("action_title") or background_state.get("message") or "Review background upkeep.")
    if doctor.get("storage_readiness_blocks_completion_claim"):
        start_kind = "storage_recovery"
        start_command = _safe_text(
            doctor.get("storage_readiness_next_required_command")
            or doctor.get("storage_next_proof_command")
            or "storage status"
        )
        start_label = "Check durable storage recovery before completion or autonomy claims."

    next_commands = [
        "catch me up",
        "safety status",
    ]
    next_commands.extend(str(command) for command in doctor.get("storage_readiness_next_commands", []))
    next_commands.append(start_command)
    if approval_handoff.get("readiness_command"):
        next_commands.extend(
            [
                str(approval_handoff["readiness_command"]),
                str(approval_handoff["last_look_command"]),
                str(approval_handoff["proof_command"]),
            ]
        )
    if health.get("next_command"):
        next_commands.append(str(health["next_command"]))
    next_commands.extend(str(command) for command in health.get("next_commands", []))
    if doctor.get("recovery_closure_checklist_command"):
        next_commands.append(str(doctor["recovery_closure_checklist_command"]))
    learning_next = doctor.get("execution_learning_actionable_next_proof_command") or doctor.get("execution_learning_next_proof_command")
    if learning_next:
        next_commands.append(str(learning_next))
    if doctor.get("agi_next", {}).get("build_command"):
        next_commands.append(str(doctor["agi_next"]["build_command"]))
    next_commands.extend(["work queue", "focus brief", "build target packet", "save handoff brief"])

    deduped_next_commands: list[str] = []
    for command in next_commands:
        command = _safe_text(command)
        if command and command not in deduped_next_commands:
            deduped_next_commands.append(command)

    return {
        "source": "safe_next_actions",
        "status": "ready",
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "start_kind": start_kind,
        "start_command": _safe_text(start_command),
        "start_label": _safe_text(start_label),
        "review_order": [
            "review context",
            "check safety",
            "clear approval blockers",
            "close execution health debt",
            "advance task or goal",
            "refresh AGI build target",
        ],
        "next_commands": deduped_next_commands,
        "next_command_count": len(deduped_next_commands),
        "next_safe_commands": deduped_next_commands,
        "next_safe_command_count": len(deduped_next_commands),
        "pending_approvals": len(approvals),
        "open_tasks": len(tasks),
        "active_goals": len(goals),
        "active_decisions": len(decisions),
        "active_preferences": len(preferences),
        "enabled_jobs": len(background_state["enabled_jobs"]),
        "state_snapshot_jobs": len(background_state["state_snapshot_jobs"]),
        "enabled_state_snapshot_jobs": len(background_state["enabled_state_snapshot_jobs"]),
        "disabled_state_snapshot_jobs": len(background_state["state_snapshot_jobs"]) - len(background_state["enabled_state_snapshot_jobs"]),
        "scheduler_next_command": _safe_text(background_state["command"]),
        "failed_action_runs": int(health.get("failed_action_runs") or 0),
        "approval_held_action_runs": int(health.get("approval_held_action_runs") or 0),
        "execution_health_approval_held_action_runs": int(health.get("approval_held_action_runs") or 0),
        "execution_health_review_required": _metadata_bool(health.get("review_required")),
        "execution_health_next_command": _safe_text(health.get("next_command") or ""),
        "execution_health_next_commands": [_safe_text(command) for command in health.get("next_commands", [])],
        "execution_health_next_command_count": len(health.get("next_commands", [])),
        "execution_health_blocker_categories": [str(item) for item in health.get("blocker_categories", [])],
        "execution_health_blocker_count": int(health.get("blocker_count") or 0),
        "execution_health_verification_coverage": str(health.get("verification_coverage_state") or ""),
        "doctor_next_audit_command": _safe_text(doctor.get("next_audit_command") or ""),
        "doctor_completion_claim_state": str(doctor.get("completion_claim_state") or ""),
        "doctor_completion_claim_ready": _metadata_bool(doctor.get("completion_claim_ready")),
        "doctor_completion_blockers": [str(item) for item in doctor.get("completion_blockers", [])],
        "doctor_completion_blocker_count": int(doctor.get("completion_blocker_count") or 0),
        "doctor_recent_failed_runs": int(doctor.get("recent_failed_runs") or 0),
        "doctor_recent_approval_held_runs": int(doctor.get("recent_approval_held_runs") or 0),
        **_doctor_audit_readability_handoff_metadata(doctor),
        "doctor_storage_runtime_fallback_active": _metadata_bool(doctor.get("storage_runtime_fallback_active")),
        "doctor_storage_readiness_blocks_completion_claim": _metadata_bool(
            doctor.get("storage_readiness_blocks_completion_claim")
        ),
        "doctor_storage_recovery_required": _metadata_bool(doctor.get("storage_recovery_required")),
        "doctor_storage_recovery_reason": _safe_text(doctor.get("storage_recovery_reason") or ""),
        "doctor_storage_recovery_mode": _safe_text(doctor.get("storage_recovery_mode") or ""),
        "doctor_storage_recovery_next_operator_action": _safe_text(doctor.get("storage_recovery_next_operator_action") or ""),
        "doctor_storage_recovery_restart_required": _metadata_bool(doctor.get("storage_recovery_restart_required")),
        "doctor_storage_recovery_check_tool_command": _safe_text(doctor.get("storage_recovery_check_tool_command") or ""),
        "doctor_storage_recovery_check_command": _safe_text(doctor.get("storage_recovery_check_command") or ""),
        "doctor_storage_recovery_check_api": _safe_text(doctor.get("storage_recovery_check_api") or STORAGE_RECOVERY_CHECK_API),
        "doctor_storage_recovery_command": _safe_text(doctor.get("storage_recovery_command") or ""),
        "doctor_storage_readiness_blocker": _safe_text(doctor.get("storage_readiness_blocker") or ""),
        "doctor_storage_readiness_next_commands": [_safe_text(command) for command in doctor.get("storage_readiness_next_commands", [])],
        "doctor_storage_readiness_next_command_count": len(doctor.get("storage_readiness_next_commands", [])),
        **_doctor_storage_handoff_metadata(doctor),
        "doctor_recovery_closure_state": str(doctor.get("recovery_closure_state") or ""),
        "doctor_recovery_closure_next_required_command": _safe_text(doctor.get("recovery_closure_next_required_command") or ""),
        "doctor_recovery_closure_next_proof_command": _safe_text(doctor.get("recovery_closure_next_proof_command") or ""),
        "doctor_execution_learning_state": str(doctor.get("execution_learning_state") or ""),
        "doctor_execution_learning_next_required_command": _safe_text(doctor.get("execution_learning_next_required_command") or ""),
        "doctor_execution_learning_next_proof_command": _safe_text(doctor.get("execution_learning_next_proof_command") or ""),
        "doctor_execution_learning_actionable_next_required_command": _safe_text(
            doctor.get("execution_learning_actionable_next_required_command") or ""
        ),
        "doctor_execution_learning_actionable_next_proof_command": _safe_text(
            doctor.get("execution_learning_actionable_next_proof_command") or ""
        ),
        "doctor_execution_learning_next_evidence_command": _safe_text(
            doctor.get("execution_learning_next_evidence_command") or ""
        ),
        "doctor_agi_next_gate": str(doctor.get("agi_next", {}).get("gate") or ""),
        "doctor_agi_next_selection_source": str(doctor.get("agi_next", {}).get("selection_source") or ""),
        "doctor_agi_next_selection_reason": _safe_text(doctor.get("agi_next", {}).get("selection_reason") or ""),
        "doctor_agi_next_canonical_selector_command": _safe_text(doctor.get("agi_next", {}).get("canonical_selector_command") or doctor.get("agi_next", {}).get("build_command") or doctor.get("agi_next", {}).get("next_build_command") or ""),
        "doctor_agi_next_deliberate_focus_override": _metadata_bool(
            doctor.get("agi_next", {}).get("deliberate_focus_override")
        ),
        "doctor_agi_next_target_title": _safe_text(doctor.get("agi_next", {}).get("target_title") or ""),
        "doctor_agi_next_build_command": _safe_text(doctor.get("agi_next", {}).get("build_command") or ""),
        "doctor_agi_next_real_execution_gap_count": int(doctor.get("agi_next", {}).get("real_execution_gap_count") or 0),
        "doctor_agi_next_real_execution_gaps_by_gate": dict(doctor.get("agi_next", {}).get("real_execution_gaps_by_gate") or {}),
        "doctor_agi_next_selected_real_execution_gap": _safe_text(doctor.get("agi_next", {}).get("selected_real_execution_gap") or ""),
        "approval_handoff": approval_handoff,
        "approval_handoff_pending_count": approval_handoff.get("pending_count", 0),
        "approval_handoff_first_id": approval_handoff.get("first_id"),
        "approval_handoff_next_command": _safe_text(approval_handoff.get("next_command") or ""),
        "approval_handoff_readiness_command": _safe_text(approval_handoff.get("readiness_command") or ""),
        "approval_handoff_last_look_command": _safe_text(approval_handoff.get("last_look_command") or ""),
        "approval_handoff_proof_command": _safe_text(approval_handoff.get("proof_command") or ""),
        "approval_handoff_proof_chain_commands": [_safe_text(command) for command in approval_handoff.get("proof_chain_commands", [])],
        "limit": limit,
        "boundaries": {
            "read_only": True,
            **NEXT_STEP_BOUNDARY_FALSE_FLAGS,
        },
    }


def _next_action_packet_handoff(
    *,
    action_kind: str,
    action_title: str,
    command: str,
    rationale: str,
    verification: str,
    risk: str,
    approvals: list[Any],
    tasks: list[Any],
    goals: list[Any],
    decisions: list[Any],
    preferences: list[Any],
    background_state: dict[str, Any],
    health: dict[str, Any],
    doctor: dict[str, Any],
    approval_handoff: dict[str, Any],
) -> dict[str, Any]:
    next_commands = [
        "catch me up",
        "safety status",
        command,
    ]
    if approval_handoff.get("readiness_command"):
        next_commands.extend(
            [
                str(approval_handoff["readiness_command"]),
                str(approval_handoff["last_look_command"]),
                str(approval_handoff["proof_command"]),
            ]
        )
    if health.get("next_command"):
        next_commands.append(str(health["next_command"]))
    next_commands.extend(str(item) for item in health.get("next_commands", []))
    if doctor.get("recovery_closure_checklist_command"):
        next_commands.append(str(doctor["recovery_closure_checklist_command"]))
    if doctor.get("execution_learning_next_proof_command"):
        next_commands.append(str(doctor["execution_learning_next_proof_command"]))
    if doctor.get("agi_next", {}).get("build_command"):
        next_commands.append(str(doctor["agi_next"]["build_command"]))
    next_commands.extend(["safe next actions", "work queue", "priority stack", "focus brief", "build target packet"])

    deduped_next_commands: list[str] = []
    for item in next_commands:
        item = _safe_text(item)
        if item and item not in deduped_next_commands:
            deduped_next_commands.append(item)

    return {
        "source": "next_action_packet",
        "status": "ready",
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "action_kind": _safe_text(action_kind),
        "action_title": _safe_text(action_title),
        "command": _safe_text(command),
        "rationale": _safe_text(rationale),
        "verification": _safe_text(verification),
        "risk": _safe_text(risk),
        "review_order": [
            "review recommended move",
            "check approval readiness",
            "check execution health",
            "check doctor blockers",
            "run or rehearse only through the normal risk gate",
        ],
        "next_commands": deduped_next_commands,
        "next_command_count": len(deduped_next_commands),
        "next_safe_commands": deduped_next_commands,
        "next_safe_command_count": len(deduped_next_commands),
        "pending_approvals": len(approvals),
        "open_tasks": len(tasks),
        "active_goals": len(goals),
        "active_decisions": len(decisions),
        "active_preferences": len(preferences),
        "enabled_jobs": len(background_state["enabled_jobs"]),
        "state_snapshot_jobs": len(background_state["state_snapshot_jobs"]),
        "enabled_state_snapshot_jobs": len(background_state["enabled_state_snapshot_jobs"]),
        "disabled_state_snapshot_jobs": len(background_state["state_snapshot_jobs"]) - len(background_state["enabled_state_snapshot_jobs"]),
        "scheduler_next_command": _safe_text(background_state["command"]),
        "failed_action_runs": int(health.get("failed_action_runs") or 0),
        "approval_held_action_runs": int(health.get("approval_held_action_runs") or 0),
        "execution_health_approval_held_action_runs": int(health.get("approval_held_action_runs") or 0),
        "execution_health_review_required": _metadata_bool(health.get("review_required")),
        "execution_health_next_command": _safe_text(health.get("next_command") or ""),
        "execution_health_next_commands": [_safe_text(item) for item in health.get("next_commands", [])],
        "execution_health_next_command_count": len(health.get("next_commands", [])),
        "execution_health_blocker_categories": [str(item) for item in health.get("blocker_categories", [])],
        "execution_health_blocker_count": int(health.get("blocker_count") or 0),
        "execution_health_verification_coverage": str(health.get("verification_coverage_state") or ""),
        "doctor_next_audit_command": _safe_text(doctor.get("next_audit_command") or ""),
        "doctor_completion_claim_state": str(doctor.get("completion_claim_state") or ""),
        "doctor_completion_claim_ready": _metadata_bool(doctor.get("completion_claim_ready")),
        "doctor_completion_blockers": [str(item) for item in doctor.get("completion_blockers", [])],
        "doctor_completion_blocker_count": int(doctor.get("completion_blocker_count") or 0),
        "doctor_recent_failed_runs": int(doctor.get("recent_failed_runs") or 0),
        "doctor_recent_approval_held_runs": int(doctor.get("recent_approval_held_runs") or 0),
        **_doctor_audit_readability_handoff_metadata(doctor),
        "doctor_recovery_closure_state": str(doctor.get("recovery_closure_state") or ""),
        "doctor_recovery_closure_next_required_command": _safe_text(doctor.get("recovery_closure_next_required_command") or ""),
        "doctor_recovery_closure_next_proof_command": _safe_text(doctor.get("recovery_closure_next_proof_command") or ""),
        "doctor_execution_learning_state": str(doctor.get("execution_learning_state") or ""),
        "doctor_execution_learning_next_required_command": _safe_text(doctor.get("execution_learning_next_required_command") or ""),
        "doctor_execution_learning_next_proof_command": _safe_text(doctor.get("execution_learning_next_proof_command") or ""),
        "doctor_agi_next_gate": str(doctor.get("agi_next", {}).get("gate") or ""),
        "doctor_agi_next_selection_source": str(doctor.get("agi_next", {}).get("selection_source") or ""),
        "doctor_agi_next_selection_reason": _safe_text(doctor.get("agi_next", {}).get("selection_reason") or ""),
        "doctor_agi_next_canonical_selector_command": _safe_text(doctor.get("agi_next", {}).get("canonical_selector_command") or doctor.get("agi_next", {}).get("build_command") or doctor.get("agi_next", {}).get("next_build_command") or ""),
        "doctor_agi_next_deliberate_focus_override": _metadata_bool(
            doctor.get("agi_next", {}).get("deliberate_focus_override")
        ),
        "doctor_agi_next_target_title": _safe_text(doctor.get("agi_next", {}).get("target_title") or ""),
        "doctor_agi_next_build_command": _safe_text(doctor.get("agi_next", {}).get("build_command") or ""),
        "doctor_agi_next_real_execution_gap_count": int(doctor.get("agi_next", {}).get("real_execution_gap_count") or 0),
        "doctor_agi_next_real_execution_gaps_by_gate": dict(doctor.get("agi_next", {}).get("real_execution_gaps_by_gate") or {}),
        "doctor_agi_next_selected_real_execution_gap": _safe_text(doctor.get("agi_next", {}).get("selected_real_execution_gap") or ""),
        "approval_handoff": approval_handoff,
        "approval_handoff_pending_count": approval_handoff.get("pending_count", 0),
        "approval_handoff_first_id": approval_handoff.get("first_id"),
        "approval_handoff_next_command": _safe_text(approval_handoff.get("next_command") or ""),
        "approval_handoff_readiness_command": _safe_text(approval_handoff.get("readiness_command") or ""),
        "approval_handoff_last_look_command": _safe_text(approval_handoff.get("last_look_command") or ""),
        "approval_handoff_proof_command": _safe_text(approval_handoff.get("proof_command") or ""),
        "approval_handoff_proof_chain_commands": [_safe_text(item) for item in approval_handoff.get("proof_chain_commands", [])],
        "boundaries": {
            "read_only": True,
            **NEXT_STEP_BOUNDARY_FALSE_FLAGS,
        },
    }


def _priority_stack_handoff(
    *,
    limit: int,
    stack: list[dict[str, Any]],
    approvals: list[Any],
    tasks: list[Any],
    goals: list[Any],
    decisions: list[Any],
    preferences: list[Any],
    background_state: dict[str, Any],
    health: dict[str, Any],
    doctor: dict[str, Any],
    approval_handoff: dict[str, Any],
) -> dict[str, Any]:
    visible_stack = stack[:limit]
    top = visible_stack[0] if visible_stack else {}
    stack_rows = [
        {
            "rank": int(item.get("rank") or index + 1),
            "kind": _safe_text(item.get("kind") or "unknown"),
            "command": _safe_text(item.get("command") or ""),
            "risk": _safe_text(item.get("risk") or ""),
        }
        for index, item in enumerate(visible_stack)
    ]
    next_commands = ["catch me up", "safety status"]
    next_commands.extend(str(row["command"]) for row in stack_rows if row.get("command"))
    if approval_handoff.get("readiness_command"):
        next_commands.extend(
            [
                str(approval_handoff["readiness_command"]),
                str(approval_handoff["last_look_command"]),
                str(approval_handoff["proof_command"]),
            ]
        )
    if health.get("next_command"):
        next_commands.append(str(health["next_command"]))
    next_commands.extend(str(command) for command in health.get("next_commands", []))
    if doctor.get("recovery_closure_checklist_command"):
        next_commands.append(str(doctor["recovery_closure_checklist_command"]))
    if doctor.get("execution_learning_next_proof_command"):
        next_commands.append(str(doctor["execution_learning_next_proof_command"]))
    if doctor.get("agi_next", {}).get("build_command"):
        next_commands.append(str(doctor["agi_next"]["build_command"]))
    next_commands.extend(["safe next actions", "work queue", "focus brief", "build target packet"])

    deduped_next_commands: list[str] = []
    for command in next_commands:
        command = _safe_text(command)
        if command and command not in deduped_next_commands:
            deduped_next_commands.append(command)

    return {
        "source": "priority_stack",
        "status": "ready",
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "top_kind": _safe_text(top.get("kind") or ""),
        "top_command": _safe_text(top.get("command") or ""),
        "stack_rows": stack_rows,
        "stack_count": len(stack_rows),
        "ranking_order": [
            "approval review",
            "execution health",
            "open task",
            "goal step",
            "background rhythm",
            "task capture",
        ],
        "next_commands": deduped_next_commands,
        "next_command_count": len(deduped_next_commands),
        "next_safe_commands": deduped_next_commands,
        "next_safe_command_count": len(deduped_next_commands),
        "pending_approvals": len(approvals),
        "open_tasks": len(tasks),
        "active_goals": len(goals),
        "active_decisions": len(decisions),
        "active_preferences": len(preferences),
        "enabled_jobs": len(background_state["enabled_jobs"]),
        "state_snapshot_jobs": len(background_state["state_snapshot_jobs"]),
        "enabled_state_snapshot_jobs": len(background_state["enabled_state_snapshot_jobs"]),
        "disabled_state_snapshot_jobs": len(background_state["state_snapshot_jobs"]) - len(background_state["enabled_state_snapshot_jobs"]),
        "scheduler_next_command": _safe_text(background_state["command"]),
        "failed_action_runs": int(health.get("failed_action_runs") or 0),
        "approval_held_action_runs": int(health.get("approval_held_action_runs") or 0),
        "execution_health_approval_held_action_runs": int(health.get("approval_held_action_runs") or 0),
        "execution_health_review_required": _metadata_bool(health.get("review_required")),
        "execution_health_next_command": _safe_text(health.get("next_command") or ""),
        "execution_health_next_commands": [_safe_text(command) for command in health.get("next_commands", [])],
        "execution_health_next_command_count": len(health.get("next_commands", [])),
        "execution_health_blocker_categories": [str(item) for item in health.get("blocker_categories", [])],
        "execution_health_blocker_count": int(health.get("blocker_count") or 0),
        "execution_health_verification_coverage": str(health.get("verification_coverage_state") or ""),
        "doctor_next_audit_command": _safe_text(doctor.get("next_audit_command") or ""),
        "doctor_completion_claim_state": str(doctor.get("completion_claim_state") or ""),
        "doctor_completion_claim_ready": _metadata_bool(doctor.get("completion_claim_ready")),
        "doctor_completion_blockers": [str(item) for item in doctor.get("completion_blockers", [])],
        "doctor_completion_blocker_count": int(doctor.get("completion_blocker_count") or 0),
        "doctor_recent_failed_runs": int(doctor.get("recent_failed_runs") or 0),
        "doctor_recent_approval_held_runs": int(doctor.get("recent_approval_held_runs") or 0),
        **_doctor_audit_readability_handoff_metadata(doctor),
        "doctor_recovery_closure_state": str(doctor.get("recovery_closure_state") or ""),
        "doctor_recovery_closure_next_required_command": _safe_text(doctor.get("recovery_closure_next_required_command") or ""),
        "doctor_recovery_closure_next_proof_command": _safe_text(doctor.get("recovery_closure_next_proof_command") or ""),
        "doctor_execution_learning_state": str(doctor.get("execution_learning_state") or ""),
        "doctor_execution_learning_next_required_command": _safe_text(doctor.get("execution_learning_next_required_command") or ""),
        "doctor_execution_learning_next_proof_command": _safe_text(doctor.get("execution_learning_next_proof_command") or ""),
        "doctor_agi_next_gate": str(doctor.get("agi_next", {}).get("gate") or ""),
        "doctor_agi_next_selection_source": str(doctor.get("agi_next", {}).get("selection_source") or ""),
        "doctor_agi_next_selection_reason": _safe_text(doctor.get("agi_next", {}).get("selection_reason") or ""),
        "doctor_agi_next_canonical_selector_command": _safe_text(doctor.get("agi_next", {}).get("canonical_selector_command") or doctor.get("agi_next", {}).get("build_command") or doctor.get("agi_next", {}).get("next_build_command") or ""),
        "doctor_agi_next_deliberate_focus_override": _metadata_bool(
            doctor.get("agi_next", {}).get("deliberate_focus_override")
        ),
        "doctor_agi_next_target_title": _safe_text(doctor.get("agi_next", {}).get("target_title") or ""),
        "doctor_agi_next_build_command": _safe_text(doctor.get("agi_next", {}).get("build_command") or ""),
        "doctor_agi_next_real_execution_gap_count": int(doctor.get("agi_next", {}).get("real_execution_gap_count") or 0),
        "doctor_agi_next_real_execution_gaps_by_gate": dict(doctor.get("agi_next", {}).get("real_execution_gaps_by_gate") or {}),
        "doctor_agi_next_selected_real_execution_gap": _safe_text(doctor.get("agi_next", {}).get("selected_real_execution_gap") or ""),
        "approval_handoff": approval_handoff,
        "approval_handoff_pending_count": approval_handoff.get("pending_count", 0),
        "approval_handoff_first_id": approval_handoff.get("first_id"),
        "approval_handoff_next_command": _safe_text(approval_handoff.get("next_command") or ""),
        "approval_handoff_readiness_command": _safe_text(approval_handoff.get("readiness_command") or ""),
        "approval_handoff_last_look_command": _safe_text(approval_handoff.get("last_look_command") or ""),
        "approval_handoff_proof_command": _safe_text(approval_handoff.get("proof_command") or ""),
        "approval_handoff_proof_chain_commands": [_safe_text(command) for command in approval_handoff.get("proof_chain_commands", [])],
        "limit": limit,
        "boundaries": {
            "read_only": True,
            **NEXT_STEP_BOUNDARY_FALSE_FLAGS,
        },
    }


def _approval_proof_chain_handoff_lines(health: dict[str, Any], *, indent: str = "") -> list[str]:
    chains = health.get("approval_proof_chains")
    if not isinstance(chains, dict) or not chains:
        return []
    lines = [f"{indent}- Approval proof chain handoff:"]
    for approval_id, commands in sorted(chains.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 999999):
        lines.append(f"{indent}  - approval #{approval_id}:")
        if isinstance(commands, list):
            lines.extend(f"{indent}    - `{command}`" for command in commands)
    return lines


def _planned_args(row: Any) -> dict[str, Any]:
    raw = row["planned_args"] if "planned_args" in row.keys() else "{}"
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _approval_age_minutes(row: Any) -> int | None:
    raw = str(row["created_at"] if "created_at" in row.keys() else "").strip()
    if not raw:
        return None
    try:
        created = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - created.astimezone(timezone.utc)
    return max(0, int(delta.total_seconds() // 60))


def _approval_staleness_label(age_minutes: int | None) -> str:
    if age_minutes is None:
        return "unknown"
    if age_minutes < 15:
        return "fresh"
    if age_minutes < 120:
        return "review_again"
    return "stale"


def _checkpoint_freshness_label(age_minutes: int | None) -> str:
    if age_minutes is None:
        return "missing"
    if age_minutes < CHECKPOINT_REVIEW_MINUTES:
        return "fresh"
    if age_minutes < CHECKPOINT_STALE_MINUTES:
        return "review_again"
    return "stale"


def make_next_step_tools(
    store: MemoryStore,
    vault: ObsidianVault,
    list_tools: Any | None = None,
    storage_fallback: Any | None = None,
    config: JarvisConfig | None = None,
    *,
    effect_authority: Callable[[], None] | None = None,
):
    def _priority_rank(priority: str) -> int:
        return {"high": 0, "normal": 1, "low": 2}.get(priority.lower(), 1)

    def _priority_goal_summary() -> list[str]:
        return [
            "Priority goal:",
            "- Finish Jarvis V2 as an AI agent harness toward a fully functional AGI-like personal assistant.",
            "- Prefer work that strengthens command-first UX, routing, memory/state, tools, approvals, audit, recovery, verification, or learning.",
            "- Side quests come after the harness is dependable.",
        ]

    def _checkpoint_recovery_contract(objective: str, health: dict[str, Any]) -> dict[str, Any]:
        reflections = vault.root_path / "Reflections"
        checkpoint_files = sorted(
            reflections.glob("* Work Block Checkpoint.md") if reflections.exists() else [],
            key=lambda path: path.stat().st_mtime if path.exists() else 0,
            reverse=True,
        )
        latest_checkpoint = checkpoint_files[0] if checkpoint_files else None
        checkpoint_age_minutes: int | None = None
        if latest_checkpoint:
            try:
                modified = datetime.fromtimestamp(latest_checkpoint.stat().st_mtime, timezone.utc)
                checkpoint_age_minutes = max(0, int((datetime.now(timezone.utc) - modified).total_seconds() // 60))
            except OSError:
                checkpoint_age_minutes = None
        checkpoint_freshness = _checkpoint_freshness_label(checkpoint_age_minutes)
        checkpoint_recovery_required = checkpoint_freshness in {"missing", "review_again", "stale", "unknown"}
        proof_queue: list[str] = []
        for command in [
            f"checkpoint recovery: {objective}",
            f"checkpoint recovery apply: {objective}",
            "checkpoint recovery execute reviewed=true step=<reviewed local-safe step> verification=<evidence>",
            "checkpoint recovery follow-through: step=<reviewed local-safe step> verification=<evidence> receipt=<receipt path> receipt_sha256=<hash> checkpoint=<checkpoint path> checkpoint_sha256=<hash> stop=<stop condition>",
            "work block checkpoint",
        ]:
            _append_unique(proof_queue, command)
        for command in health.get("next_commands", [])[:4]:
            _append_unique(proof_queue, str(command))
        verification_target = str(
            health.get("verification_handoff_command")
            or health.get("next_command")
            or "focused smoke test for the recovered local-safe step"
        )
        stop_condition = (
            "stop if the checkpoint is missing, review-again, stale, or unknown; "
            "if the reviewed step is risky without approval proof; if verification evidence is absent; "
            "or if recovery changes scope"
        )
        return {
            "checkpoint_found": bool(latest_checkpoint),
            "checkpoint_path": str(latest_checkpoint or ""),
            "checkpoint_path_display": _safe_vault_path_display(latest_checkpoint, vault) if latest_checkpoint else "",
            "checkpoint_age_minutes": checkpoint_age_minutes,
            "checkpoint_freshness": checkpoint_freshness,
            "checkpoint_recovery_required": checkpoint_recovery_required,
            "checkpoint_recovery_queue": proof_queue,
            "checkpoint_recovery_queue_count": len(proof_queue),
            "checkpoint_recovery_next_command": proof_queue[0] if proof_queue else "",
            "checkpoint_recovery_verification_target": verification_target,
            "checkpoint_recovery_stop_condition": stop_condition,
            "checkpoint_recovery_followthrough_gate": "checkpoint recovery execute must emit Recovery closure gate with receipt, fresh checkpoint, verification evidence, stop condition, and approval boundary before normal follow-through resumes",
            "checkpoint_recovery_followthrough_packet_command": "checkpoint recovery follow-through: step=<reviewed local-safe step> verification=<evidence> receipt=<receipt path> receipt_sha256=<hash> checkpoint=<checkpoint path> checkpoint_sha256=<hash> stop=<stop condition>",
        }

    def _looks_like_priority_goal(goal: Any) -> bool:
        text = f"{goal['title']} {goal['purpose']}".lower()
        return any(marker in text for marker in ("jarvis", "harness", "agi", "assistant"))

    def _execution_health_snapshot(limit: int = 40) -> dict[str, Any]:
        rows = store.recent_tool_runs(limit=limit)
        action_runs = [row for row in rows if _is_execution_action_row(row)]
        failed_action_runs, approval_held_action_runs = _recent_tool_run_attention_buckets(action_runs)
        approval_held_action_run_ids = {int(row["id"]) for row in approval_held_action_runs}
        learning_context_action_runs = [row for row in action_runs if int(row["id"]) not in approval_held_action_run_ids]
        risky_levels = {"HIGH_RISK", "PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT"}
        risky_approval_problems = [
            row
            for row in action_runs
            if str(row["risk"]) in risky_levels
            and bool(row["ok"])
            and (not bool(row["approved"]) or ("approval_id" not in row.keys() or row["approval_id"] is None))
        ]
        verification_runs = [
            row
            for row in rows
            if str(row["tool_name"]) in {
                "verification_receipt",
                "runtime_trace_receipt",
                "execution_audit_gate",
                "execution_recovery_packet",
                "after_action_learning_packet",
                "execution_health_report",
            }
        ]
        recovery_runs = [row for row in rows if str(row["tool_name"]) == "execution_recovery_packet"]
        learning_runs = [row for row in rows if str(row["tool_name"]) == "after_action_learning_packet"]
        failure_counts = {}
        for row in failed_action_runs:
            tool_name = str(row["tool_name"])
            failure_counts[tool_name] = failure_counts.get(tool_name, 0) + 1
        approval_proof_chains: dict[str, list[str]] = {}
        for row in failed_action_runs + approval_held_action_runs + risky_approval_problems:
            approval_candidates: list[int] = []
            approval_id = row["approval_id"] if "approval_id" in row.keys() else None
            if approval_id is not None:
                approval_candidates.append(int(approval_id))
            for candidate in _ids_from_text(row["output"], ("approval #", "approval id", "approval packet", "queued as approval")):
                if candidate not in approval_candidates:
                    approval_candidates.append(candidate)
            audit_metadata = _row_metadata(row)
            metadata_approval_id = audit_metadata.get("approval_id") or audit_metadata.get("approved_approval_id")
            try:
                if metadata_approval_id is not None and int(metadata_approval_id) not in approval_candidates:
                    approval_candidates.append(int(metadata_approval_id))
            except (TypeError, ValueError):
                pass
            for candidate in approval_candidates:
                approval_proof_chains.setdefault(str(candidate), _approval_proof_chain_commands(candidate))
        repeated_failure_tools = [
            {"tool": tool_name, "count": count}
            for tool_name, count in sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))
            if count > 1
        ]
        failure_promotion_queue: list[str] = []
        if repeated_failure_tools:
            first_repeated_tool = repeated_failure_tools[0]["tool"]
            failure_promotion_queue = [
                f"failure to test preview: repeated {first_repeated_tool} failures",
                f"failure promotion packet: {first_repeated_tool}",
                f"failure implementation packet: {first_repeated_tool}",
                f"failure apply contract: {first_repeated_tool}",
                "learning review",
            ]
        blocker_categories: list[str] = []
        if risky_approval_problems:
            blocker_categories.append("approval_chain")
        if failed_action_runs:
            blocker_categories.append("failed_or_blocked")
        if approval_held_action_runs:
            blocker_categories.append("approval_held")
        if repeated_failure_tools:
            blocker_categories.append("repeated_failure")
        if not verification_runs:
            blocker_categories.append("verification_gap")
        first_problem = failed_action_runs[0] if failed_action_runs else (risky_approval_problems[0] if risky_approval_problems else None)
        if first_problem is not None:
            next_command = f"execution recovery packet {first_problem['id']}"
        elif rows:
            next_command = "execution health report"
        else:
            next_command = "readiness report"
        learning_target = first_problem if first_problem is not None else (
            learning_context_action_runs[0] if learning_context_action_runs else None
        )
        learning_target_run_id = int(learning_target["id"]) if learning_target is not None else None
        learning_target_tool = str(learning_target["tool_name"]) if learning_target is not None else ""
        verification_handoff_command = (
            f"verification receipt {learning_target_run_id}" if learning_target_run_id is not None else "verification receipt latest"
        )
        recovery_handoff_command = (
            f"execution recovery packet {learning_target_run_id}" if learning_target_run_id is not None else ""
        )
        learning_handoff_command = (
            f"after-action learning packet {learning_target_run_id}" if learning_target_run_id is not None else "after-action learning packet"
        )
        learning_closure_command = (
            f"execution learning closure {learning_target_run_id}" if learning_target_run_id is not None else "execution learning closure"
        )
        target_verification_runs = [
            row
            for row in verification_runs
            if str(row["tool_name"]) == "verification_receipt" and _row_metadata(row).get("run_id") == learning_target_run_id
        ]
        target_recovery_runs = [
            row
            for row in recovery_runs
            if _row_metadata(row).get("run_id") == learning_target_run_id
        ]
        target_learning_runs = [
            row
            for row in learning_runs
            if _row_metadata(row).get("run_id") == learning_target_run_id
        ]
        execution_learning_missing: list[str] = []
        execution_learning_commands: list[str] = []
        if learning_target_run_id is not None and not target_learning_runs:
            execution_learning_missing.append("target_after_action_learning_packet")
            execution_learning_commands.append(learning_handoff_command)
            execution_learning_commands.append(learning_closure_command)
        if failed_action_runs:
            execution_learning_missing.append("failure_review")
            if recovery_handoff_command:
                execution_learning_commands.append(recovery_handoff_command)
            execution_learning_commands.append(
                f"failure to test preview: run #{learning_target_run_id} {learning_target_tool} failed or blocked"
            )
        if failed_action_runs and not learning_runs:
            execution_learning_state = "LEARNING_DEBT_AFTER_FAILURE"
        elif execution_learning_missing:
            execution_learning_state = "LEARNING_REVIEW_REQUIRED"
        elif action_runs:
            execution_learning_state = "LEARNING_LOOP_HAS_RECENT_ACTION_CONTEXT"
        else:
            execution_learning_state = "NO_RECENT_ACTION_RUNS"
        execution_learning_blocks_completion = execution_learning_state in {"LEARNING_DEBT_AFTER_FAILURE", "LEARNING_REVIEW_REQUIRED"}
        deduped_execution_learning_commands: list[str] = []
        for command in execution_learning_commands:
            if command:
                _append_unique(deduped_execution_learning_commands, command)
        execution_learning_commands = deduped_execution_learning_commands
        actionable_execution_learning_commands: list[str] = []
        for command in execution_learning_commands:
            if command == learning_closure_command and "target_after_action_learning_packet" in execution_learning_missing:
                _append_unique(actionable_execution_learning_commands, learning_handoff_command)
            elif command != learning_closure_command:
                _append_unique(actionable_execution_learning_commands, command)
        if execution_learning_commands:
            _append_unique(actionable_execution_learning_commands, learning_closure_command)
        recovery_closure_missing: list[str] = []
        recovery_closure_required_commands: list[str] = []
        if learning_target_run_id is not None and first_problem is not None:
            if not target_verification_runs:
                recovery_closure_missing.append("target_verification_receipt")
                recovery_closure_required_commands.append(verification_handoff_command)
            if not target_recovery_runs:
                recovery_closure_missing.append("target_recovery_packet")
                recovery_closure_required_commands.append(recovery_handoff_command)
            if not target_learning_runs:
                recovery_closure_missing.append("target_after_action_learning_packet")
                recovery_closure_required_commands.append(learning_handoff_command)
                recovery_closure_required_commands.append(learning_closure_command)
            if approval_proof_chains:
                recovery_closure_missing.append("approval_chain_proof")
                recovery_closure_required_commands.append("approval history")
        if repeated_failure_tools:
            for command in failure_promotion_queue:
                if command not in recovery_closure_required_commands:
                    recovery_closure_required_commands.append(command)
        if first_problem is None and not repeated_failure_tools:
            recovery_closure_state = "not_needed"
        elif recovery_closure_missing:
            recovery_closure_state = "blocked_missing_" + "_and_".join(recovery_closure_missing)
        elif repeated_failure_tools:
            recovery_closure_state = "blocked_repeated_failure_promotion_required"
        else:
            recovery_closure_state = "ready_for_operator_retry_review"
        recovery_closure_ready_to_retry = recovery_closure_state == "ready_for_operator_retry_review"
        recovery_closure_blocks_auto_execution = recovery_closure_state not in {"not_needed", "ready_for_operator_retry_review"}
        recovery_closure_blocks_completion_claim = recovery_closure_state != "not_needed"
        next_commands: list[str] = []
        for row in failed_action_runs[:3]:
            command = f"execution recovery packet {row['id']}"
            _append_unique(next_commands, command)
            approval_id = row["approval_id"] if "approval_id" in row.keys() else None
            if approval_id is not None:
                for proof_command in _approval_proof_chain_commands(int(approval_id)):
                    _append_unique(next_commands, proof_command)
        for row in approval_held_action_runs[:3]:
            approval_id = row["approval_id"] if "approval_id" in row.keys() else None
            if approval_id is not None:
                for proof_command in _approval_proof_chain_commands(int(approval_id)):
                    _append_unique(next_commands, proof_command)
        for row in risky_approval_problems[:3]:
            command = f"execution recovery packet {row['id']}"
            _append_unique(next_commands, command)
            approval_id = row["approval_id"] if "approval_id" in row.keys() else None
            if approval_id is not None:
                for proof_command in _approval_proof_chain_commands(int(approval_id)):
                    _append_unique(next_commands, proof_command)
        for item in repeated_failure_tools[:3]:
            command = f"failure to test preview: repeated {item['tool']} failures"
            _append_unique(next_commands, command)
        for command in failure_promotion_queue:
            _append_unique(next_commands, command)
        handoff_commands = [verification_handoff_command, recovery_handoff_command]
        if learning_target_run_id is not None:
            handoff_commands.extend(actionable_execution_learning_commands or [learning_closure_command, learning_handoff_command])
        for command in handoff_commands:
            if command:
                _append_unique(next_commands, command)
        for command in recovery_closure_required_commands:
            if command:
                _append_unique(next_commands, command)
        for command in execution_learning_commands:
            if command:
                _append_unique(next_commands, command)
        if not verification_runs and rows and "verification receipt latest" not in next_commands:
            next_commands.append("verification receipt latest")
        if "execution health report" not in next_commands:
            next_commands.append("execution health report")
        if recovery_closure_blocks_auto_execution and recovery_closure_required_commands:
            ordered_next_commands: list[str] = []
            for command in recovery_closure_required_commands:
                if command:
                    _append_unique(ordered_next_commands, command)
            for command in next_commands:
                _append_unique(ordered_next_commands, command)
            next_commands = ordered_next_commands
            next_command = recovery_closure_required_commands[0]
        return {
            "tool_runs": len(rows),
            "action_runs": len(action_runs),
            "failed_action_runs": len(failed_action_runs),
            "approval_held_action_runs": len(approval_held_action_runs),
            "risky_approval_problems": len(risky_approval_problems),
            "verification_runs": len(verification_runs),
            "repeated_failure_tools": repeated_failure_tools,
            "repeated_failure_count": sum(item["count"] for item in repeated_failure_tools),
            "failure_promotion_queue": failure_promotion_queue,
            "failure_promotion_queue_count": len(failure_promotion_queue),
            "failure_promotion_command": failure_promotion_queue[1] if len(failure_promotion_queue) > 1 else "",
            "failure_implementation_command": failure_promotion_queue[2] if len(failure_promotion_queue) > 2 else "",
            "failure_apply_contract_command": failure_promotion_queue[3] if len(failure_promotion_queue) > 3 else "",
            "blocker_categories": blocker_categories,
            "blocker_count": len(blocker_categories),
            "verification_coverage_state": "present" if verification_runs else "missing",
            "first_problem_run_id": int(first_problem["id"]) if first_problem is not None else None,
            "first_problem_tool": str(first_problem["tool_name"]) if first_problem is not None else "",
            "learning_target_run_id": learning_target_run_id,
            "learning_target_tool": learning_target_tool,
            "verification_handoff_command": verification_handoff_command,
            "recovery_handoff_command": recovery_handoff_command,
            "learning_handoff_command": learning_handoff_command,
            "learning_closure_command": learning_closure_command,
            "execution_learning_closure_command": learning_closure_command,
            "target_verification_receipts": len(target_verification_runs),
            "target_recovery_packets": len(target_recovery_runs),
            "target_after_action_learning_packets": len(target_learning_runs),
            "execution_learning_state": execution_learning_state,
            "execution_learning_blocks_completion_claim": execution_learning_blocks_completion,
            "execution_learning_missing": execution_learning_missing,
            "execution_learning_missing_count": len(execution_learning_missing),
            "execution_learning_required_commands": execution_learning_commands,
            "execution_learning_next_required_command": execution_learning_commands[0] if execution_learning_commands else "",
            "execution_learning_proof_queue": execution_learning_commands,
            "execution_learning_proof_queue_count": len(execution_learning_commands),
            "execution_learning_next_proof_command": execution_learning_commands[0] if execution_learning_commands else "",
            "execution_learning_actionable_required_commands": actionable_execution_learning_commands,
            "execution_learning_actionable_required_command_count": len(actionable_execution_learning_commands),
            "execution_learning_actionable_next_required_command": actionable_execution_learning_commands[0]
            if actionable_execution_learning_commands
            else "",
            "execution_learning_actionable_proof_queue": actionable_execution_learning_commands,
            "execution_learning_actionable_proof_queue_count": len(actionable_execution_learning_commands),
            "execution_learning_actionable_next_proof_command": actionable_execution_learning_commands[0]
            if actionable_execution_learning_commands
            else "",
            "execution_learning_next_evidence_command": actionable_execution_learning_commands[0]
            if actionable_execution_learning_commands
            else "",
            "execution_learning_recent_action_runs": len(action_runs),
            "execution_learning_failed_or_blocked_action_runs": len(failed_action_runs),
            "execution_learning_approval_held_action_runs": len(approval_held_action_runs),
            "execution_learning_recent_after_action_learning_runs": len(learning_runs),
            "execution_learning_target_run_id": learning_target_run_id,
            "execution_learning_target_tool_name": learning_target_tool,
            "execution_learning_target_after_action_learning_packets": len(target_learning_runs),
            "recovery_closure_state": recovery_closure_state,
            "recovery_closure_missing": recovery_closure_missing,
            "recovery_closure_missing_count": len(recovery_closure_missing),
            "recovery_closure_required_commands": recovery_closure_required_commands,
            "recovery_closure_next_required_command": recovery_closure_required_commands[0] if recovery_closure_required_commands else "",
            "recovery_closure_proof_queue": recovery_closure_required_commands,
            "recovery_closure_proof_queue_count": len(recovery_closure_required_commands),
            "recovery_closure_next_proof_command": recovery_closure_required_commands[0] if recovery_closure_required_commands else "",
            "recovery_closure_ready_to_retry": recovery_closure_ready_to_retry,
            "recovery_closure_blocks_auto_execution": recovery_closure_blocks_auto_execution,
            "recovery_closure_blocks_completion_claim": recovery_closure_blocks_completion_claim,
            "recovery_closure_target_run_id": learning_target_run_id,
            "recovery_closure_target_tool_name": learning_target_tool,
            "recovery_closure_target_verification_receipts": len(target_verification_runs),
            "recovery_closure_target_recovery_packets": len(target_recovery_runs),
            "recovery_closure_target_after_action_learning_packets": len(target_learning_runs),
            "next_command": next_command,
            "next_commands": next_commands,
            "approval_proof_chains": approval_proof_chains,
            "approval_proof_chain_count": len(approval_proof_chains),
            "review_required": first_problem is not None,
        }

    def _execution_health_learning_handoff_lines(health: dict[str, Any], *, indent: str = "") -> list[str]:
        target = health.get("learning_target_run_id")
        target_present = target is not None
        tool_name = str(health.get("learning_target_tool") or "")
        verification_command = str(health.get("verification_handoff_command") or "verification receipt latest")
        recovery_command = str(health.get("recovery_handoff_command") or "")
        learning_closure_command = str(health.get("learning_closure_command") or "execution learning closure")
        learning_command = str(health.get("learning_handoff_command") or "after-action learning packet")
        target_line = f"run #{target} `{tool_name}`" if target_present else "none in reviewed window"
        verification_label = "verify" if target_present else "coverage check"
        lines = [
            f"{indent}Verification-to-learning handoff:",
            f"{indent}- target: {target_line}",
            f"{indent}- {verification_label}: `{verification_command}`",
        ]
        if recovery_command:
            lines.append(f"{indent}- recover: `{recovery_command}`")
        if target_present:
            lines.append(f"{indent}- learn: `{learning_command}`")
            lines.append(f"{indent}- close learning gate: `{learning_closure_command}`")
        lines.extend(
            [
                f"{indent}Execution learning debt:",
                f"{indent}- state: {health.get('execution_learning_state')}",
                f"{indent}- blocks completion: {'yes' if health.get('execution_learning_blocks_completion_claim') else 'no'}",
                f"{indent}- missing: {', '.join(health.get('execution_learning_missing') or []) if health.get('execution_learning_missing') else 'none'}",
            ]
        )
        learning_queue = health.get("execution_learning_required_commands")
        actionable_learning_queue = health.get("execution_learning_actionable_required_commands")
        if isinstance(actionable_learning_queue, list) and actionable_learning_queue:
            lines.append(f"{indent}- next actionable learning required: `{actionable_learning_queue[0]}`")
            lines.append(
                f"{indent}- actionable learning proof queue: "
                + ", ".join(f"`{command}`" for command in actionable_learning_queue[:4])
            )
        if isinstance(learning_queue, list) and learning_queue:
            lines.append(f"{indent}- ordered learning gate required: `{learning_queue[0]}`")
            lines.append(f"{indent}- learning proof queue: {', '.join(f'`{command}`' for command in learning_queue[:4])}")
        next_commands = health.get("next_commands")
        if isinstance(next_commands, list) and next_commands:
            lines.extend(
                [
                    f"{indent}Execution proof queue:",
                    f"{indent}- next required command: `{health.get('next_command')}`",
                    f"{indent}- proof queue count: {len(next_commands)}",
                    f"{indent}- proof queue: {', '.join(f'`{command}`' for command in next_commands[:8])}",
                ]
            )
        lines.extend(
            [
                f"{indent}Recovery closure gate:",
                f"{indent}- state: {health.get('recovery_closure_state')}",
                f"{indent}- ready for operator retry review: {'yes' if health.get('recovery_closure_ready_to_retry') else 'no'}",
                f"{indent}- blocks auto-run: {'yes' if health.get('recovery_closure_blocks_auto_execution') else 'no'}",
                f"{indent}- missing: {', '.join(health.get('recovery_closure_missing') or []) if health.get('recovery_closure_missing') else 'none'}",
            ]
        )
        checklist_command = _recovery_closure_checklist_command(health)
        if checklist_command:
            lines.append(f"{indent}- checklist overview: `{checklist_command}`")
        queue = health.get("failure_promotion_queue")
        if isinstance(queue, list) and queue:
            lines.append(f"{indent}Failure promotion queue:")
            lines.extend(f"{indent}- `{command}`" for command in queue)
        return lines

    def _execution_health_learning_handoff_metadata(health: dict[str, Any]) -> dict[str, Any]:
        target_present = health.get("learning_target_run_id") is not None
        learning_handoff_command = str(health.get("learning_handoff_command") or "") if target_present else ""
        learning_closure_command = str(health.get("learning_closure_command") or "") if target_present else ""
        return {
            "execution_health_learning_target_present": target_present,
            "execution_health_approval_held_action_runs": health.get("approval_held_action_runs", 0),
            "execution_health_learning_target_run_id": health.get("learning_target_run_id"),
            "execution_health_learning_target_tool": str(health.get("learning_target_tool") or ""),
            "execution_health_verification_handoff_command": str(health.get("verification_handoff_command") or ""),
            "execution_health_recovery_handoff_command": str(health.get("recovery_handoff_command") or ""),
            "execution_health_learning_handoff_command": learning_handoff_command,
            "execution_health_learning_closure_command": learning_closure_command,
            "execution_health_execution_learning_closure_command": learning_closure_command,
            "execution_health_target_verification_receipts": health.get("target_verification_receipts", 0),
            "execution_health_target_recovery_packets": health.get("target_recovery_packets", 0),
            "execution_health_target_after_action_learning_packets": health.get("target_after_action_learning_packets", 0),
            "execution_learning_state": str(health.get("execution_learning_state") or ""),
            "execution_learning_blocks_completion_claim": _metadata_bool(
                health.get("execution_learning_blocks_completion_claim")
            ),
            "execution_learning_missing": health.get("execution_learning_missing", []),
            "execution_learning_missing_count": health.get("execution_learning_missing_count", 0),
            "execution_learning_required_commands": health.get("execution_learning_required_commands", []),
            "execution_learning_next_required_command": str(health.get("execution_learning_next_required_command") or ""),
            "execution_learning_proof_queue": health.get("execution_learning_proof_queue"),
            "execution_learning_proof_queue_count": health.get("execution_learning_proof_queue_count"),
            "execution_learning_next_proof_command": str(health.get("execution_learning_next_proof_command") or ""),
            "execution_learning_actionable_required_commands": health.get("execution_learning_actionable_required_commands", []),
            "execution_learning_actionable_required_command_count": health.get(
                "execution_learning_actionable_required_command_count", 0
            ),
            "execution_learning_actionable_next_required_command": str(
                health.get("execution_learning_actionable_next_required_command") or ""
            ),
            "execution_learning_actionable_proof_queue": health.get("execution_learning_actionable_proof_queue", []),
            "execution_learning_actionable_proof_queue_count": health.get(
                "execution_learning_actionable_proof_queue_count", 0
            ),
            "execution_learning_actionable_next_proof_command": str(
                health.get("execution_learning_actionable_next_proof_command") or ""
            ),
            "execution_learning_next_evidence_command": str(health.get("execution_learning_next_evidence_command") or ""),
            "execution_learning_closure_command": str(health.get("execution_learning_closure_command") or ""),
            "execution_learning_recent_action_runs": health.get("execution_learning_recent_action_runs", 0),
            "execution_learning_failed_or_blocked_action_runs": health.get("execution_learning_failed_or_blocked_action_runs", 0),
            "execution_learning_approval_held_action_runs": health.get("execution_learning_approval_held_action_runs", 0),
            "execution_learning_recent_after_action_learning_runs": health.get("execution_learning_recent_after_action_learning_runs", 0),
            "execution_learning_target_run_id": health.get("execution_learning_target_run_id"),
            "execution_learning_target_tool_name": str(health.get("execution_learning_target_tool_name") or ""),
            "execution_learning_target_after_action_learning_packets": health.get("execution_learning_target_after_action_learning_packets", 0),
            "execution_health_recovery_closure_state": str(health.get("recovery_closure_state") or ""),
            "execution_health_recovery_closure_missing": health.get("recovery_closure_missing", []),
            "execution_health_recovery_closure_missing_count": health.get("recovery_closure_missing_count", 0),
            "execution_health_recovery_closure_required_commands": health.get("recovery_closure_required_commands", []),
            "execution_health_recovery_closure_next_required_command": str(health.get("recovery_closure_next_required_command") or ""),
            "execution_health_recovery_closure_proof_queue": health.get("recovery_closure_proof_queue"),
            "execution_health_recovery_closure_proof_queue_count": health.get("recovery_closure_proof_queue_count"),
            "execution_health_recovery_closure_next_proof_command": str(health.get("recovery_closure_next_proof_command") or ""),
            "execution_health_recovery_closure_ready_to_retry": _metadata_bool(
                health.get("recovery_closure_ready_to_retry")
            ),
            "execution_health_recovery_closure_blocks_auto_execution": _metadata_bool(
                health.get("recovery_closure_blocks_auto_execution")
            ),
            "execution_health_recovery_closure_blocks_completion_claim": _metadata_bool(
                health.get("recovery_closure_blocks_completion_claim")
            ),
            "execution_health_recovery_closure_checklist_command": _recovery_closure_checklist_command(health),
            "execution_health_recovery_closure_should_open_checklist": bool(_recovery_closure_checklist_command(health)),
            "execution_health_recovery_closure_target_run_id": health.get("recovery_closure_target_run_id"),
            "execution_health_recovery_closure_target_tool_name": str(health.get("recovery_closure_target_tool_name") or ""),
            "execution_health_recovery_closure_target_verification_receipts": health.get("recovery_closure_target_verification_receipts", 0),
            "execution_health_recovery_closure_target_recovery_packets": health.get("recovery_closure_target_recovery_packets", 0),
            "execution_health_recovery_closure_target_after_action_learning_packets": health.get("recovery_closure_target_after_action_learning_packets", 0),
            "execution_health_failure_promotion_queue": health.get("failure_promotion_queue", []),
            "execution_health_failure_promotion_queue_count": health.get("failure_promotion_queue_count", 0),
            "execution_health_failure_promotion_command": str(health.get("failure_promotion_command") or ""),
            "execution_health_failure_implementation_command": str(health.get("failure_implementation_command") or ""),
            "execution_health_failure_apply_contract_command": str(health.get("failure_apply_contract_command") or ""),
            "execution_health_proof_queue": health.get("next_commands", []),
            "execution_health_proof_queue_count": len(health.get("next_commands", [])),
            "execution_health_next_proof_command": str(health.get("next_command") or ""),
        }

    def _doctor_handoff_snapshot(limit: int = 40) -> dict[str, Any]:
        approvals = store.list_pending_approvals(limit=20)
        tasks = store.list_tasks(status="open", limit=20)
        goals = store.list_goals(status="active", limit=20)
        rows = store.recent_tool_runs(limit=limit)
        readable_rows, unreadable_recent_run_rows = _readable_tool_run_rows(rows)
        failures, approval_held_runs = _recent_tool_run_attention_buckets(readable_rows)
        verification_runs = [
            row
            for row in readable_rows
            if str(row["tool_name"]) in {
                "verification_receipt",
                "runtime_trace_receipt",
                "execution_audit_gate",
                "execution_recovery_packet",
                "after_action_learning_packet",
                "execution_health_report",
                "execution_acceptance_gate",
                "completion_audit_packet",
                "task_completion_packet",
                "evidence_ledger",
            }
        ]
        learning_runs = [row for row in readable_rows if str(row["tool_name"]) == "after_action_learning_packet"]
        health = _execution_health_snapshot(limit=limit)
        audit_readability_review_commands = (
            ["storage status", "recent tool runs", "execution health report"]
            if unreadable_recent_run_rows
            else []
        )
        blockers: list[str] = []
        if unreadable_recent_run_rows:
            blockers.append(f"{unreadable_recent_run_rows} unreadable recent tool run row(s)")
        if approvals:
            blockers.append(f"{len(approvals)} pending approval(s)")
        if tasks:
            blockers.append(f"{len(tasks)} open task(s)")
        if goals:
            blockers.append(f"{len(goals)} active goal(s)")
        if failures:
            blockers.append(f"{len(failures)} recent failed/blocked run(s)")
        if approval_held_runs:
            blockers.append(f"{len(approval_held_runs)} recent approval-held run(s) need approval review")
        if not verification_runs:
            blockers.append("no recent verification/audit packet")
        if not learning_runs:
            blockers.append("no recent after-action learning packet")
        if health["recovery_closure_blocks_auto_execution"]:
            blockers.append(
                "execution health recovery closure incomplete: "
                + ", ".join(health["recovery_closure_missing"] or [str(health["recovery_closure_state"])])
            )
        if health["execution_learning_blocks_completion_claim"]:
            blockers.append(
                "execution learning debt incomplete: "
                + ", ".join(health["execution_learning_missing"] or [str(health["execution_learning_state"])])
            )
        storage_handoff = _storage_completion_handoff(storage_fallback, config)
        if storage_handoff["blocks_completion_claim"]:
            blockers.append(str(storage_handoff["blocker"]))
        completion_proof_queue: list[str] = []
        for command in storage_handoff["next_commands"]:
            if command:
                completion_proof_queue.append(command)
        for command in audit_readability_review_commands:
            if command not in completion_proof_queue:
                completion_proof_queue.append(command)
        if approvals:
            first = approvals[0]
            for command in [
                f"approval readiness {first['id']}",
                f"approval packet {first['id']}",
                f"approval chain proof {first['id']}",
                f"verification receipt <approved run id from approval chain proof {first['id']}>",
            ]:
                if command not in completion_proof_queue:
                    completion_proof_queue.append(command)
        if approval_held_runs:
            first_held = approval_held_runs[0]
            approval_id = None
            try:
                if first_held["approval_id"] is not None:
                    approval_id = int(first_held["approval_id"])
            except Exception:
                approval_id = None
            if approval_id and approval_id > 0:
                for command in _approval_proof_chain_commands(approval_id):
                    if command not in completion_proof_queue:
                        completion_proof_queue.append(command)
            elif "recent tool runs" not in completion_proof_queue:
                completion_proof_queue.append("recent tool runs")
        if tasks:
            command = f"task completion packet {tasks[0]['id']}"
            if command not in completion_proof_queue:
                completion_proof_queue.append(command)
        for command in health["recovery_closure_proof_queue"]:
            if command not in completion_proof_queue:
                completion_proof_queue.append(command)
        learning_proof_queue = health.get("execution_learning_actionable_proof_queue") or health["execution_learning_proof_queue"]
        for command in learning_proof_queue:
            if command not in completion_proof_queue:
                completion_proof_queue.append(command)
        for command in ["harness completion", "completion audit", "evidence ledger", "completion claim gate"]:
            if command not in completion_proof_queue:
                completion_proof_queue.append(command)
        agi_next = _doctor_default_agi_handoff(list_tools)
        return {
            "agi_next": agi_next,
            "next_audit_command": health["next_command"],
            "recovery_closure_checklist_command": _recovery_closure_checklist_command(health),
            "completion_claim_state": "BLOCKED" if blockers else "READY_FOR_HUMAN_REVIEW",
            "completion_claim_ready": not blockers,
            "completion_blockers": blockers,
            "completion_blocker_count": len(blockers),
            "completion_proof_queue": completion_proof_queue,
            "completion_proof_queue_count": len(completion_proof_queue),
            "completion_next_proof_command": completion_proof_queue[0] if completion_proof_queue else "",
            "storage_runtime_fallback_active": storage_handoff["active"],
            "storage_runtime_fallback_reason": storage_handoff["reason"],
            "storage_runtime_fallback_exception_type": storage_handoff["exception_type"],
            "storage_runtime_fallback_db_path_display": storage_handoff["db_path_display"],
            "storage_runtime_fallback_vault_path_display": storage_handoff["vault_path_display"],
            "storage_readiness_blocks_completion_claim": storage_handoff["blocks_completion_claim"],
            "storage_recovery_required": storage_handoff["recovery_required"],
            "storage_recovery_reason": storage_handoff["recovery_reason"],
            "storage_recovery_mode": storage_handoff["recovery_mode"],
            "storage_recovery_next_operator_action": storage_handoff["recovery_next_operator_action"],
            "storage_recovery_restart_required": storage_handoff["recovery_restart_required"],
            "storage_recovery_check_tool_command": storage_handoff["recovery_check_tool_command"],
            "storage_recovery_check_command": storage_handoff["recovery_check_command"],
            "storage_recovery_check_api": storage_handoff["recovery_check_api"],
            "storage_recovery_command": storage_handoff["recovery_command"],
            "storage_readiness_blocker": storage_handoff["blocker"],
            "storage_readiness_next_commands": storage_handoff["next_commands"],
            "storage_readiness_next_command_count": storage_handoff["next_command_count"],
            "storage_readiness_next_required_command": storage_handoff["storage_readiness_next_required_command"],
            "storage_readiness_next_proof_command": storage_handoff["storage_readiness_next_proof_command"],
            "storage_readiness_proof_queue": storage_handoff["storage_readiness_proof_queue"],
            "storage_readiness_proof_queue_count": storage_handoff["storage_readiness_proof_queue_count"],
            "storage_readiness_first_proof_command": storage_handoff["storage_readiness_first_proof_command"],
            "storage_next_proof_command": storage_handoff["next_proof_command"],
            "storage_issues": storage_handoff["storage_issues"],
            "storage_issue_count": storage_handoff["storage_issue_count"],
            "storage_handoff": storage_handoff["storage_handoff"],
            "storage_selection_source": storage_handoff["selection_source"],
            "pending_approvals": len(approvals),
            "open_tasks": len(tasks),
            "active_goals": len(goals),
            "recent_tool_runs": len(rows),
            "unreadable_recent_tool_run_rows": unreadable_recent_run_rows,
            "audit_readability_review_required": bool(audit_readability_review_commands),
            "audit_readability_review_commands": audit_readability_review_commands,
            "audit_readability_review_command_count": len(audit_readability_review_commands),
            "audit_readability_review_next_command": (
                audit_readability_review_commands[0] if audit_readability_review_commands else ""
            ),
            "recent_failed_runs": len(failures),
            "recent_approval_held_runs": len(approval_held_runs),
            "recent_verification_runs": len(verification_runs),
            "recent_after_action_learning_runs": len(learning_runs),
            "recovery_closure_state": health["recovery_closure_state"],
            "recovery_closure_missing": health["recovery_closure_missing"],
            "recovery_closure_missing_count": health["recovery_closure_missing_count"],
            "recovery_closure_required_commands": health["recovery_closure_required_commands"],
            "recovery_closure_next_required_command": health["recovery_closure_next_required_command"],
            "recovery_closure_proof_queue": health["recovery_closure_proof_queue"],
            "recovery_closure_proof_queue_count": health["recovery_closure_proof_queue_count"],
            "recovery_closure_next_proof_command": health["recovery_closure_next_proof_command"],
            "recovery_closure_ready_to_retry": health["recovery_closure_ready_to_retry"],
            "recovery_closure_blocks_completion_claim": health["recovery_closure_blocks_completion_claim"],
            "recovery_closure_should_open_checklist": bool(_recovery_closure_checklist_command(health)),
            "recovery_closure_target_run_id": health["recovery_closure_target_run_id"],
            "recovery_closure_target_tool_name": health["recovery_closure_target_tool_name"],
            "recovery_closure_target_verification_receipts": health["recovery_closure_target_verification_receipts"],
            "recovery_closure_target_recovery_packets": health["recovery_closure_target_recovery_packets"],
            "recovery_closure_target_after_action_learning_packets": health["recovery_closure_target_after_action_learning_packets"],
            "execution_learning_state": health["execution_learning_state"],
            "execution_learning_missing": health["execution_learning_missing"],
            "execution_learning_missing_count": health["execution_learning_missing_count"],
            "execution_learning_required_commands": health["execution_learning_required_commands"],
            "execution_learning_next_required_command": health["execution_learning_next_required_command"],
            "execution_learning_proof_queue": health["execution_learning_proof_queue"],
            "execution_learning_proof_queue_count": health["execution_learning_proof_queue_count"],
            "execution_learning_next_proof_command": health["execution_learning_next_proof_command"],
            "execution_learning_actionable_required_commands": health["execution_learning_actionable_required_commands"],
            "execution_learning_actionable_required_command_count": health["execution_learning_actionable_required_command_count"],
            "execution_learning_actionable_next_required_command": health["execution_learning_actionable_next_required_command"],
            "execution_learning_actionable_proof_queue": health["execution_learning_actionable_proof_queue"],
            "execution_learning_actionable_proof_queue_count": health["execution_learning_actionable_proof_queue_count"],
            "execution_learning_actionable_next_proof_command": health["execution_learning_actionable_next_proof_command"],
            "execution_learning_next_evidence_command": health["execution_learning_next_evidence_command"],
            "execution_learning_blocks_completion_claim": health["execution_learning_blocks_completion_claim"],
        }

    def _doctor_handoff_lines(snapshot: dict[str, Any], *, indent: str = "") -> list[str]:
        lines = [
            f"{indent}- Doctor handoff:",
            f"{indent}  - completion claim state: {snapshot['completion_claim_state']}",
            f"{indent}  - next audit command: `{snapshot['next_audit_command']}`",
            f"{indent}  - recovery closure state: {snapshot['recovery_closure_state']}",
            f"{indent}  - recovery closure next required: `{snapshot['recovery_closure_next_required_command']}`" if snapshot.get("recovery_closure_next_required_command") else f"{indent}  - recovery closure next required: none",
            f"{indent}  - recovery closure checklist: `{snapshot['recovery_closure_checklist_command']}`" if snapshot.get("recovery_closure_checklist_command") else f"{indent}  - recovery closure checklist: none",
            f"{indent}  - execution learning debt state: {snapshot.get('execution_learning_state')}",
            f"{indent}  - execution learning actionable next required: `{snapshot['execution_learning_actionable_next_required_command']}`" if snapshot.get("execution_learning_actionable_next_required_command") else f"{indent}  - execution learning actionable next required: none",
            f"{indent}  - execution learning ordered gate required: `{snapshot['execution_learning_next_required_command']}`" if snapshot.get("execution_learning_next_required_command") else f"{indent}  - execution learning ordered gate required: none",
            f"{indent}  - completion blockers: {snapshot['completion_blocker_count']}",
            f"{indent}  - recent failed/blocked runs: {snapshot.get('recent_failed_runs', 0)}",
            f"{indent}  - recent approval-held runs: {snapshot.get('recent_approval_held_runs', 0)}",
            f"{indent}  - unreadable recent tool-run rows: {snapshot.get('unreadable_recent_tool_run_rows', 0)}",
        ]
        audit_readability_commands = snapshot.get("audit_readability_review_commands") or []
        if audit_readability_commands:
            lines.append(
                f"{indent}  - audit readability review: "
                + ", ".join(f"`{command}`" for command in audit_readability_commands)
            )
        storage_commands = snapshot.get("storage_readiness_next_commands") or []
        if storage_commands:
            lines.extend(
                [
                    f"{indent}  - storage recovery required: yes",
                    f"{indent}  - storage recovery mode: {snapshot.get('storage_recovery_mode') or 'none'}",
                    f"{indent}  - storage next operator action: {snapshot.get('storage_recovery_next_operator_action') or 'none'}",
                    f"{indent}  - storage restart required: {'yes' if snapshot.get('storage_recovery_restart_required') else 'no'}",
                    f"{indent}  - storage next required: `{snapshot.get('storage_readiness_next_required_command') or snapshot.get('storage_next_proof_command')}`",
                    f"{indent}  - storage proof alias: `{snapshot.get('storage_readiness_next_proof_command') or snapshot.get('storage_next_proof_command')}`",
                    f"{indent}  - storage recovery queue: " + ", ".join(f"`{command}`" for command in storage_commands),
                ]
            )
        agi = snapshot.get("agi_next") or {}
        if agi:
            lines.extend(
                [
                    f"{indent}  - selected AGI target: {agi.get('target_title') or agi.get('gate')}",
                    f"{indent}  - selected AGI next build: `{agi.get('build_command')}`",
                    f"{indent}  - selected AGI target files: {agi.get('target_file_integrity_status')}",
                    f"{indent}  - selected AGI ready for review: {'yes' if agi.get('build_packet_ready_for_review') else 'no'}",
                    f"{indent}  - AGI real-execution gaps still tracked: {agi.get('real_execution_gap_count', 0)}",
                    f"{indent}  - selected AGI real-execution gap: {agi.get('selected_real_execution_gap') or 'none'}",
                ]
            )
        recovery_commands = snapshot.get("recovery_closure_proof_queue") or []
        if recovery_commands:
            lines.append(
                f"{indent}  - recovery closure proof queue: "
                + ", ".join(f"`{command}`" for command in recovery_commands[:5])
            )
        learning_commands = snapshot.get("execution_learning_proof_queue") or []
        learning_actionable_commands = snapshot.get("execution_learning_actionable_proof_queue") or []
        if learning_actionable_commands:
            lines.append(
                f"{indent}  - execution learning actionable proof queue: "
                + ", ".join(f"`{command}`" for command in learning_actionable_commands[:5])
            )
        if learning_commands:
            lines.append(
                f"{indent}  - execution learning proof queue: "
                + ", ".join(f"`{command}`" for command in learning_commands[:5])
            )
        completion_commands = snapshot.get("completion_proof_queue") or []
        if completion_commands:
            lines.append(
                f"{indent}  - completion proof queue: "
                + ", ".join(f"`{command}`" for command in completion_commands[:6])
            )
        blockers = snapshot.get("completion_blockers") or []
        if blockers:
            lines.append(f"{indent}  - blocker details: {'; '.join(str(item) for item in blockers[:4])}")
        return lines

    def _approval_handoff_snapshot(limit: int = 20) -> dict[str, Any]:
        approvals = store.list_pending_approvals(limit=limit)
        if not approvals:
            return {
                "pending_count": 0,
                "first_id": None,
                "first_tool_name": "",
                "first_request": "",
                "first_age_minutes": None,
                "first_staleness": "none",
                "first_planned_arg_keys": [],
                "readiness_command": "",
                "last_look_command": "",
                "proof_command": "",
                "verification_command": "",
                "decision_matrix": [],
                "decision_commands": [],
                "proof_chain_commands": [],
                "next_command": "pending approvals",
            }
        first = approvals[0]
        approval_id = int(first["id"])
        age_minutes = _approval_age_minutes(first)
        proof_chain = _approval_proof_chain_commands(approval_id)
        return {
            "pending_count": len(approvals),
            "first_id": approval_id,
            "first_tool_name": str(first["tool_name"]),
            "first_request": _short(first["user_input"], limit=140),
            "first_age_minutes": age_minutes,
            "first_staleness": _approval_staleness_label(age_minutes),
            "first_planned_arg_keys": sorted(_planned_args(first)),
            "readiness_command": f"approval readiness {approval_id}",
            "last_look_command": f"approval packet {approval_id}",
            "proof_command": f"approval chain proof {approval_id}",
            "verification_command": f"verification receipt <approved run id from approval chain proof {approval_id}>",
            "decision_matrix": [
                "approve only if the operator still wants the exact stored request now",
                "approve only if planned arguments, target, and expected result are clear and verifiable",
                "dismiss if stale, broad, surprising, private, destructive, wrong target, or no longer needed",
            ],
            "decision_commands": [f"approve approval {approval_id}", f"dismiss approval {approval_id}"],
            "proof_chain_commands": proof_chain,
            "next_command": f"approval readiness {approval_id}",
        }

    def _approval_handoff_lines(snapshot: dict[str, Any], *, indent: str = "") -> list[str]:
        if not snapshot.get("first_id"):
            return [f"{indent}- Approval readiness handoff: no pending approvals."]
        planned_arg_keys = snapshot.get("first_planned_arg_keys") or []
        return [
            f"{indent}- Approval readiness handoff:",
            f"{indent}  - first pending approval: #{snapshot['first_id']} `{snapshot['first_tool_name']}`",
            f"{indent}  - staleness: {snapshot['first_staleness']}",
            f"{indent}  - planned arg keys: {', '.join(planned_arg_keys) if planned_arg_keys else 'none'}",
            f"{indent}  - next: `{snapshot['readiness_command']}`",
            f"{indent}  - last look: `{snapshot['last_look_command']}`",
            f"{indent}  - proof: `{snapshot['proof_command']}`",
            f"{indent}  - verify after approval: `{snapshot['verification_command']}`",
            f"{indent}  - decision: approve only if exact and verifiable; dismiss stale, broad, private, surprising, destructive, wrong-target, or no-longer-needed requests",
        ]

    def _approval_handoff_metadata(snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "approval_handoff_pending_count": snapshot["pending_count"],
            "approval_handoff_first_id": snapshot["first_id"],
            "approval_handoff_first_tool_name": snapshot["first_tool_name"],
            "approval_handoff_first_request": snapshot["first_request"],
            "approval_handoff_first_age_minutes": snapshot["first_age_minutes"],
            "approval_handoff_first_staleness": snapshot["first_staleness"],
            "approval_handoff_first_planned_arg_keys": snapshot["first_planned_arg_keys"],
            "approval_handoff_next_command": snapshot["next_command"],
            "approval_handoff_readiness_command": snapshot["readiness_command"],
            "approval_handoff_last_look_command": snapshot["last_look_command"],
            "approval_handoff_proof_command": snapshot["proof_command"],
            "approval_handoff_verification_command": snapshot["verification_command"],
            "approval_handoff_decision_matrix": snapshot["decision_matrix"],
            "approval_handoff_decision_commands": snapshot["decision_commands"],
            "approval_handoff_proof_chain_commands": snapshot["proof_chain_commands"],
        }

    def build_safe_next_actions(limit: int) -> tuple[str, dict[str, int]]:
        approvals = store.list_pending_approvals(limit=limit)
        tasks = store.list_tasks(status="open", limit=limit)
        goals = store.list_goals(status="active", limit=limit)
        decisions = store.list_decisions(status="active", limit=3)
        preferences = store.list_preferences(status="active", limit=3)
        jobs = store.list_jobs()
        background_state = _scheduler_background_state(jobs)

        lines = [
            "Safe next actions:",
            *_priority_goal_summary(),
            "",
            "- Start with read-only review: `catch me up`, `safety status`, or `daily plan`.",
        ]

        if approvals:
            lines.append("- Review blocked risky requests with `pending approvals`, then use `approval readiness #ID`, `approval packet #ID`, and `approval chain proof #ID` before deciding.")
            for row in approvals[:3]:
                lines.append(f"  - Approval #{row['id']} {_short(row['tool_name'], limit=80)}: {_short(row['user_input'])}")
                lines.append(f"    Readiness: `approval readiness {row['id']}`")
                lines.append(f"    Last look: `approval packet {row['id']}`")
                lines.append(f"    Proof: `approval chain proof {row['id']}`")
        else:
            lines.append("- No pending approvals are blocking risky work right now.")

        health = _execution_health_snapshot()
        doctor = _doctor_handoff_snapshot()
        approval_handoff = _approval_handoff_snapshot()
        safe_next_handoff = _safe_next_actions_handoff(
            limit=limit,
            approvals=approvals,
            tasks=tasks,
            goals=goals,
            decisions=decisions,
            preferences=preferences,
            background_state=background_state,
            health=health,
            doctor=doctor,
            approval_handoff=approval_handoff,
        )
        lines.append("- Execution health:")
        if health["review_required"]:
            lines.append(f"  - Recovery review is needed for run #{health['first_problem_run_id']} `{health['first_problem_tool']}`; start with `{health['next_command']}`.")
        else:
            lines.append(f"  - No failed action run is blocking in the reviewed window; spot-check with `{health['next_command']}`.")
        lines.append(f"  - Blocker categories: {', '.join(health['blocker_categories']) if health['blocker_categories'] else 'none'}")
        lines.append(f"  - Verification coverage: {health['verification_coverage_state']}")
        if health["next_commands"]:
            lines.append("  - Recovery queue:")
            lines.extend(f"    - `{command}`" for command in health["next_commands"][:4])
        lines.extend(_execution_health_learning_handoff_lines(health, indent="  "))
        lines.extend(_approval_handoff_lines(approval_handoff, indent="  "))
        lines.extend(_approval_proof_chain_handoff_lines(health, indent="  "))
        lines.extend(_doctor_handoff_lines(doctor, indent="  "))

        if tasks:
            lines.append("- Work the top open task:")
            task = tasks[0]
            priority = f" [{_short(task['priority'], limit=40)}]" if task["priority"] != "normal" else ""
            due = f" due {_short(task['due'], limit=80)}" if task["due"] else ""
            lines.append(f"  - `complete task {task['id']}` when done: {_short(task['body'])}{due}{priority}")
        else:
            lines.append("- Capture one concrete task if the next move is still vague: `add task ...`.")

        if goals:
            lines.append("- Advance the first active goal step:")
            ordered_goals = sorted(goals[:3], key=lambda row: (0 if _looks_like_priority_goal(row) else 1, row["updated_at"]))
            for goal in ordered_goals:
                steps = store.list_goal_steps(goal["id"])
                open_steps = [step for step in steps if step["status"] != "done"]
                if open_steps:
                    step = open_steps[0]
                    lines.append(f"  - Goal #{goal['id']} {_short(goal['title'])}: {_short(step['body'])} (`complete goal step {step['id']}`)")
                else:
                    lines.append(f"  - Goal #{goal['id']} {_short(goal['title'])}: add a next step with `add step to goal {goal['id']}: ...`")
        else:
            lines.append("- Create a durable goal if this is larger than one task: `create goal ... because ...`.")

        lines.append(f"- {background_state['message']}")

        if decisions:
            lines.append(f"- Keep current decision context in mind: #{decisions[0]['id']} {_short(decisions[0]['title'])}.")
        if preferences:
            pref = preferences[0]
            lines.append(f"- Apply active preference: {_short(pref['key'], limit=80)} = {_short(pref['value'])}.")

        lines.append("- Do not auto-run shell, file-write, clipboard-read, or computer-control actions without explicit approval.")

        metadata = _safe_metadata(
            pending_approvals=len(approvals),
            open_tasks=len(tasks),
            active_goals=len(goals),
            active_decisions=len(decisions),
            active_preferences=len(preferences),
            enabled_jobs=len(background_state["enabled_jobs"]),
            state_snapshot_jobs=len(background_state["state_snapshot_jobs"]),
            enabled_state_snapshot_jobs=len(background_state["enabled_state_snapshot_jobs"]),
            disabled_state_snapshot_jobs=(
                len(background_state["state_snapshot_jobs"]) - len(background_state["enabled_state_snapshot_jobs"])
            ),
            scheduler_next_command=background_state["command"],
            failed_action_runs=health["failed_action_runs"],
            approval_held_action_runs=health["approval_held_action_runs"],
            execution_health_review_required=health["review_required"],
            execution_health_next_command=health["next_command"],
            execution_health_next_commands=health["next_commands"],
            execution_health_next_command_count=len(health["next_commands"]),
            execution_health_blocker_categories=health["blocker_categories"],
            execution_health_blocker_count=health["blocker_count"],
            execution_health_verification_coverage=health["verification_coverage_state"],
            execution_health_repeated_failure_count=health["repeated_failure_count"],
            execution_health_approval_proof_chains=health["approval_proof_chains"],
            execution_health_approval_proof_chain_count=health["approval_proof_chain_count"],
            **_execution_health_learning_handoff_metadata(health),
            doctor_next_audit_command=doctor["next_audit_command"],
            doctor_completion_claim_state=doctor["completion_claim_state"],
            doctor_completion_claim_ready=doctor["completion_claim_ready"],
            doctor_completion_blockers=doctor["completion_blockers"],
            doctor_completion_blocker_count=doctor["completion_blocker_count"],
            doctor_recent_failed_runs=doctor["recent_failed_runs"],
            doctor_recent_approval_held_runs=doctor["recent_approval_held_runs"],
            doctor_completion_proof_queue=doctor["completion_proof_queue"],
            doctor_completion_proof_queue_count=doctor["completion_proof_queue_count"],
            doctor_completion_next_proof_command=doctor["completion_next_proof_command"],
            doctor_storage_runtime_fallback_active=doctor["storage_runtime_fallback_active"],
            doctor_storage_runtime_fallback_reason=doctor["storage_runtime_fallback_reason"],
            doctor_storage_runtime_fallback_exception_type=doctor["storage_runtime_fallback_exception_type"],
            doctor_storage_runtime_fallback_db_path_display=doctor["storage_runtime_fallback_db_path_display"],
            doctor_storage_runtime_fallback_vault_path_display=doctor["storage_runtime_fallback_vault_path_display"],
            doctor_storage_readiness_blocks_completion_claim=doctor["storage_readiness_blocks_completion_claim"],
            doctor_storage_recovery_required=doctor["storage_recovery_required"],
            doctor_storage_recovery_reason=doctor["storage_recovery_reason"],
            doctor_storage_recovery_mode=doctor["storage_recovery_mode"],
            doctor_storage_recovery_next_operator_action=doctor["storage_recovery_next_operator_action"],
            doctor_storage_recovery_restart_required=doctor["storage_recovery_restart_required"],
            doctor_storage_recovery_check_tool_command=doctor["storage_recovery_check_tool_command"],
            doctor_storage_recovery_check_command=doctor["storage_recovery_check_command"],
            doctor_storage_recovery_check_api=doctor["storage_recovery_check_api"],
            doctor_storage_recovery_command=doctor["storage_recovery_command"],
            doctor_storage_readiness_blocker=doctor["storage_readiness_blocker"],
            doctor_storage_readiness_next_commands=doctor["storage_readiness_next_commands"],
            doctor_storage_readiness_next_command_count=doctor["storage_readiness_next_command_count"],
            **_doctor_storage_handoff_metadata(doctor),
            doctor_storage_next_proof_command=doctor["storage_next_proof_command"],
            doctor_storage_selection_source=doctor["storage_selection_source"],
            doctor_recent_verification_runs=doctor["recent_verification_runs"],
            doctor_recent_after_action_learning_runs=doctor["recent_after_action_learning_runs"],
            **_doctor_audit_readability_handoff_metadata(doctor),
            doctor_recovery_closure_state=doctor["recovery_closure_state"],
            doctor_recovery_closure_missing=doctor["recovery_closure_missing"],
            doctor_recovery_closure_missing_count=doctor["recovery_closure_missing_count"],
            doctor_recovery_closure_required_commands=doctor["recovery_closure_required_commands"],
            doctor_recovery_closure_next_required_command=doctor["recovery_closure_next_required_command"],
            doctor_recovery_closure_proof_queue=doctor["recovery_closure_proof_queue"],
            doctor_recovery_closure_proof_queue_count=doctor["recovery_closure_proof_queue_count"],
            doctor_recovery_closure_next_proof_command=doctor["recovery_closure_next_proof_command"],
            doctor_recovery_closure_ready_to_retry=doctor["recovery_closure_ready_to_retry"],
            doctor_recovery_closure_blocks_completion_claim=doctor["recovery_closure_blocks_completion_claim"],
            doctor_recovery_closure_checklist_command=doctor["recovery_closure_checklist_command"],
            doctor_recovery_closure_should_open_checklist=doctor["recovery_closure_should_open_checklist"],
            doctor_execution_learning_state=doctor["execution_learning_state"],
            doctor_execution_learning_missing=doctor["execution_learning_missing"],
            doctor_execution_learning_missing_count=doctor["execution_learning_missing_count"],
            doctor_execution_learning_required_commands=doctor["execution_learning_required_commands"],
            doctor_execution_learning_next_required_command=doctor["execution_learning_next_required_command"],
            doctor_execution_learning_proof_queue=doctor["execution_learning_proof_queue"],
            doctor_execution_learning_proof_queue_count=doctor["execution_learning_proof_queue_count"],
            doctor_execution_learning_next_proof_command=doctor["execution_learning_next_proof_command"],
            doctor_execution_learning_actionable_required_commands=doctor["execution_learning_actionable_required_commands"],
            doctor_execution_learning_actionable_required_command_count=doctor[
                "execution_learning_actionable_required_command_count"
            ],
            doctor_execution_learning_actionable_next_required_command=doctor[
                "execution_learning_actionable_next_required_command"
            ],
            doctor_execution_learning_actionable_proof_queue=doctor["execution_learning_actionable_proof_queue"],
            doctor_execution_learning_actionable_proof_queue_count=doctor["execution_learning_actionable_proof_queue_count"],
            doctor_execution_learning_actionable_next_proof_command=doctor[
                "execution_learning_actionable_next_proof_command"
            ],
            doctor_execution_learning_next_evidence_command=doctor["execution_learning_next_evidence_command"],
            doctor_execution_learning_blocks_completion_claim=doctor["execution_learning_blocks_completion_claim"],
            **_doctor_agi_handoff_metadata(doctor),
            **_approval_handoff_metadata(approval_handoff),
            safe_next_actions_handoff_ready=safe_next_handoff["handoff_ready"],
            safe_next_actions_ready_for_operator=safe_next_handoff["ready_for_operator"],
            safe_next_actions_state_changed=safe_next_handoff["state_changed"],
            safe_next_actions_changed=safe_next_handoff["changed"],
            safe_next_actions_content_in_handoff=safe_next_handoff["content_in_handoff"],
            safe_next_actions_start_kind=safe_next_handoff["start_kind"],
            safe_next_actions_start_command=safe_next_handoff["start_command"],
            safe_next_actions_start_label=safe_next_handoff["start_label"],
            safe_next_actions_review_order=safe_next_handoff["review_order"],
            safe_next_actions_next_commands=safe_next_handoff["next_commands"],
            safe_next_actions_next_safe_commands=safe_next_handoff["next_safe_commands"],
            safe_next_actions_next_command_count=safe_next_handoff["next_command_count"],
            safe_next_actions_next_safe_command_count=safe_next_handoff["next_safe_command_count"],
            safe_next_actions_authorizes_execution=safe_next_handoff["boundaries"]["authorizes_execution"],
            safe_next_actions_authorizes_completion_claim=safe_next_handoff["boundaries"]["authorizes_completion_claim"],
            safe_next_actions_approval_granted=safe_next_handoff["boundaries"]["approval_granted"],
            safe_next_actions_boundaries=safe_next_handoff["boundaries"],
            safe_next_actions_handoff=safe_next_handoff,
            limit=limit,
        )
        return "\n".join(lines), metadata

    def build_work_queue(limit: int) -> tuple[str, dict[str, int]]:
        approvals = store.list_pending_approvals(limit=limit)
        tasks = sorted(store.list_tasks(status="open", limit=limit), key=lambda row: (_priority_rank(row["priority"]), row["created_at"]))
        goals = store.list_goals(status="active", limit=limit)
        decisions = store.list_decisions(status="active", limit=3)
        jobs = store.list_jobs()
        background_state = _scheduler_background_state(jobs)
        health = _execution_health_snapshot()
        doctor = _doctor_handoff_snapshot()
        approval_handoff = _approval_handoff_snapshot()
        work_queue_handoff = _work_queue_handoff(
            limit=limit,
            approvals=approvals,
            tasks=tasks,
            goals=goals,
            decisions=decisions,
            background_state=background_state,
            health=health,
            doctor=doctor,
            approval_handoff=approval_handoff,
        )

        lines = [
            "Jarvis work queue:",
            "Rule: do safe review and local memory work first; approval-gated work stays blocked until the operator approves it.",
            "",
            *_priority_goal_summary(),
            "",
            "1. Start safely",
            "- Run `handoff brief` or `catch me up` before resuming after time away.",
            "- Run `safety status` if any action touches files, shell, clipboard, private data, or computer control.",
        ]

        lines.extend(["", "2. Approval blockers"])
        if approvals:
            lines.append("- Review these before trusting blocked risky work:")
            for row in approvals[:limit]:
                lines.append(f"  - Approval #{row['id']} {_short(row['tool_name'], limit=80)}: {_short(row['user_input'])}")
                lines.append(f"    Readiness: `approval readiness {row['id']}`")
                lines.append(f"    Last look: `approval packet {row['id']}`")
                lines.append(f"    Proof: `approval chain proof {row['id']}`")
                lines.append(f"    Decide: `approve approval {row['id']}` or `dismiss approval {row['id']}`")
        else:
            lines.append("- No pending approvals are blocking risky work right now.")

        lines.extend(["", "3. Execution health"])
        if health["review_required"]:
            lines.append(f"- Recovery review required for run #{health['first_problem_run_id']} `{health['first_problem_tool']}`.")
            lines.append(f"  Command: `{health['next_command']}`")
        else:
            lines.append(f"- No failed action run is blocking in the reviewed window. Check: `{health['next_command']}`")
        lines.append(f"- Blocker categories: {', '.join(health['blocker_categories']) if health['blocker_categories'] else 'none'}")
        lines.append(f"- Verification coverage: {health['verification_coverage_state']}")
        if health["next_commands"]:
            lines.append("- Recovery queue:")
            lines.extend(f"  - `{command}`" for command in health["next_commands"][:5])
        lines.extend(_execution_health_learning_handoff_lines(health))
        lines.extend(_approval_handoff_lines(approval_handoff))
        lines.extend(_approval_proof_chain_handoff_lines(health))
        lines.extend(_doctor_handoff_lines(doctor))

        lines.extend(["", "4. Open tasks"])
        if tasks:
            for row in tasks[:limit]:
                priority = f" [{_short(row['priority'], limit=40)}]" if row["priority"] != "normal" else ""
                due = f" due {_short(row['due'], limit=80)}" if row["due"] else ""
                lines.append(f"- Task #{row['id']}{priority}{due}: {_short(row['body'])}")
                lines.append(f"  Command when done: `complete task {row['id']}`")
        else:
            lines.append("- No open tasks. Capture one with `add task ...` if the next move is vague.")

        lines.extend(["", "5. Goal steps"])
        if goals:
            for goal in goals[:limit]:
                open_steps = [step for step in store.list_goal_steps(goal["id"]) if step["status"] != "done"]
                if open_steps:
                    step = open_steps[0]
                    lines.append(f"- Goal #{goal['id']} {_short(goal['title'])}: {_short(step['body'])}")
                    lines.append(f"  Command when done: `complete goal step {step['id']}`")
                else:
                    lines.append(f"- Goal #{goal['id']} {_short(goal['title'])}: define the next step with `add step to goal {goal['id']}: ...`")
        else:
            lines.append("- No active goals. Create one with `create goal ... because ...` for multi-step work.")

        lines.extend(["", "6. Background upkeep"])
        lines.append(f"- {background_state['message']}")
        if decisions:
            lines.append(f"- Current decision anchor: #{decisions[0]['id']} {_short(decisions[0]['title'])}.")
        lines.append("- End longer sessions by running `save handoff brief` so the operator has a resume point.")

        lines.extend(
            [
                "",
                "7. Agent landscape research guidance",
                "- Zoey/OpenClaw/Hermes research points Jarvis toward a visible control plane, not companion personas.",
                "- OpenAI Agents SDK/LangGraph/CrewAI/n8n converge on traces, evals, guardrails, human-in-the-loop checks, and health/control planes.",
                "- OpenHands/Devin-style long-running agents reinforce sandboxed execution, reviewable progress, and receipt-backed changes.",
                "- OpenClaw/Hermes always-on agents warn against broad privileges, skill supply-chain risk, and persistent prompt-injection across memory, skills, and schedules.",
                "- Borrow UX from Zoey/Lindy/Zapier: voice/text, phone commands, capability discovery, and health visibility without companion identities or broad ungated integrations.",
                "- Other Jarvis-style systems reinforce the same lesson: make capabilities, approvals, health, and failures visible before adding broad autonomy.",
                "- Keep one trusted coordinator; use internal subagents only when parallel work has a real task benefit.",
                "- Prioritize phone reliability, voice input, capability health, morning brief delivery, contact resolution, and operator-real evals.",
                "- Defer broad integration bridges until the live channel proofs and core trust loop are solid.",
            ]
        )

        metadata = _safe_metadata(
            pending_approvals=len(approvals),
            open_tasks=len(tasks),
            active_goals=len(goals),
            agent_landscape_research_note_available=True,
            agent_landscape_research_includes_other_jarvis_models=True,
            agent_landscape_research_next_command="work queue",
            agent_landscape_research_recommends_companion_personas=False,
            agent_landscape_research_recommends_visible_control_plane=True,
            agent_landscape_research_requires_control_plane=True,
            agent_landscape_research_requires_human_in_loop=True,
            agent_landscape_research_warns_persistent_agent_risk=True,
            agent_landscape_research_defers_broad_integrations=True,
            agent_landscape_research_next_build_focus=[
                "capability_cockpit",
                "phone_control_center",
                "operator_workflow_evals",
            ],
            enabled_jobs=len(background_state["enabled_jobs"]),
            state_snapshot_jobs=len(background_state["state_snapshot_jobs"]),
            enabled_state_snapshot_jobs=len(background_state["enabled_state_snapshot_jobs"]),
            disabled_state_snapshot_jobs=(
                len(background_state["state_snapshot_jobs"]) - len(background_state["enabled_state_snapshot_jobs"])
            ),
            scheduler_next_command=background_state["command"],
            failed_action_runs=health["failed_action_runs"],
            approval_held_action_runs=health["approval_held_action_runs"],
            execution_health_review_required=health["review_required"],
            execution_health_next_command=health["next_command"],
            execution_health_next_commands=health["next_commands"],
            execution_health_next_command_count=len(health["next_commands"]),
            execution_health_blocker_categories=health["blocker_categories"],
            execution_health_blocker_count=health["blocker_count"],
            execution_health_verification_coverage=health["verification_coverage_state"],
            execution_health_repeated_failure_count=health["repeated_failure_count"],
            execution_health_approval_proof_chains=health["approval_proof_chains"],
            execution_health_approval_proof_chain_count=health["approval_proof_chain_count"],
            **_execution_health_learning_handoff_metadata(health),
            doctor_next_audit_command=doctor["next_audit_command"],
            doctor_completion_claim_state=doctor["completion_claim_state"],
            doctor_completion_claim_ready=doctor["completion_claim_ready"],
            doctor_completion_blockers=doctor["completion_blockers"],
            doctor_completion_blocker_count=doctor["completion_blocker_count"],
            doctor_recent_failed_runs=doctor["recent_failed_runs"],
            doctor_recent_approval_held_runs=doctor["recent_approval_held_runs"],
            doctor_completion_proof_queue=doctor["completion_proof_queue"],
            doctor_completion_proof_queue_count=doctor["completion_proof_queue_count"],
            doctor_completion_next_proof_command=doctor["completion_next_proof_command"],
            doctor_storage_runtime_fallback_active=doctor["storage_runtime_fallback_active"],
            doctor_storage_runtime_fallback_reason=doctor["storage_runtime_fallback_reason"],
            doctor_storage_runtime_fallback_exception_type=doctor["storage_runtime_fallback_exception_type"],
            doctor_storage_runtime_fallback_db_path_display=doctor["storage_runtime_fallback_db_path_display"],
            doctor_storage_runtime_fallback_vault_path_display=doctor["storage_runtime_fallback_vault_path_display"],
            doctor_storage_readiness_blocks_completion_claim=doctor["storage_readiness_blocks_completion_claim"],
            doctor_storage_recovery_required=doctor["storage_recovery_required"],
            doctor_storage_recovery_reason=doctor["storage_recovery_reason"],
            doctor_storage_recovery_mode=doctor["storage_recovery_mode"],
            doctor_storage_recovery_next_operator_action=doctor["storage_recovery_next_operator_action"],
            doctor_storage_recovery_restart_required=doctor["storage_recovery_restart_required"],
            doctor_storage_recovery_check_tool_command=doctor["storage_recovery_check_tool_command"],
            doctor_storage_recovery_check_command=doctor["storage_recovery_check_command"],
            doctor_storage_recovery_check_api=doctor["storage_recovery_check_api"],
            doctor_storage_recovery_command=doctor["storage_recovery_command"],
            doctor_storage_readiness_blocker=doctor["storage_readiness_blocker"],
            doctor_storage_readiness_next_commands=doctor["storage_readiness_next_commands"],
            doctor_storage_readiness_next_command_count=doctor["storage_readiness_next_command_count"],
            **_doctor_storage_handoff_metadata(doctor),
            doctor_storage_next_proof_command=doctor["storage_next_proof_command"],
            doctor_storage_selection_source=doctor["storage_selection_source"],
            doctor_recent_verification_runs=doctor["recent_verification_runs"],
            doctor_recent_after_action_learning_runs=doctor["recent_after_action_learning_runs"],
            **_doctor_audit_readability_handoff_metadata(doctor),
            doctor_recovery_closure_state=doctor["recovery_closure_state"],
            doctor_recovery_closure_missing=doctor["recovery_closure_missing"],
            doctor_recovery_closure_missing_count=doctor["recovery_closure_missing_count"],
            doctor_recovery_closure_required_commands=doctor["recovery_closure_required_commands"],
            doctor_recovery_closure_next_required_command=doctor["recovery_closure_next_required_command"],
            doctor_recovery_closure_proof_queue=doctor["recovery_closure_proof_queue"],
            doctor_recovery_closure_proof_queue_count=doctor["recovery_closure_proof_queue_count"],
            doctor_recovery_closure_next_proof_command=doctor["recovery_closure_next_proof_command"],
            doctor_recovery_closure_ready_to_retry=doctor["recovery_closure_ready_to_retry"],
            doctor_recovery_closure_blocks_completion_claim=doctor["recovery_closure_blocks_completion_claim"],
            doctor_recovery_closure_checklist_command=doctor["recovery_closure_checklist_command"],
            doctor_recovery_closure_should_open_checklist=doctor["recovery_closure_should_open_checklist"],
            doctor_execution_learning_state=doctor["execution_learning_state"],
            doctor_execution_learning_missing=doctor["execution_learning_missing"],
            doctor_execution_learning_missing_count=doctor["execution_learning_missing_count"],
            doctor_execution_learning_required_commands=doctor["execution_learning_required_commands"],
            doctor_execution_learning_next_required_command=doctor["execution_learning_next_required_command"],
            doctor_execution_learning_proof_queue=doctor["execution_learning_proof_queue"],
            doctor_execution_learning_proof_queue_count=doctor["execution_learning_proof_queue_count"],
            doctor_execution_learning_next_proof_command=doctor["execution_learning_next_proof_command"],
            doctor_execution_learning_actionable_required_commands=doctor["execution_learning_actionable_required_commands"],
            doctor_execution_learning_actionable_required_command_count=doctor[
                "execution_learning_actionable_required_command_count"
            ],
            doctor_execution_learning_actionable_next_required_command=doctor[
                "execution_learning_actionable_next_required_command"
            ],
            doctor_execution_learning_actionable_proof_queue=doctor["execution_learning_actionable_proof_queue"],
            doctor_execution_learning_actionable_proof_queue_count=doctor["execution_learning_actionable_proof_queue_count"],
            doctor_execution_learning_actionable_next_proof_command=doctor[
                "execution_learning_actionable_next_proof_command"
            ],
            doctor_execution_learning_next_evidence_command=doctor["execution_learning_next_evidence_command"],
            doctor_execution_learning_blocks_completion_claim=doctor["execution_learning_blocks_completion_claim"],
            **_doctor_agi_handoff_metadata(doctor),
            **_approval_handoff_metadata(approval_handoff),
            limit=limit,
            active_decisions=len(decisions),
            work_queue_handoff_ready=work_queue_handoff["handoff_ready"],
            work_queue_ready_for_operator=work_queue_handoff["ready_for_operator"],
            work_queue_state_changed=work_queue_handoff["state_changed"],
            work_queue_changed=work_queue_handoff["changed"],
            work_queue_content_in_handoff=work_queue_handoff["content_in_handoff"],
            work_queue_start_kind=work_queue_handoff["start_kind"],
            work_queue_start_command=work_queue_handoff["start_command"],
            work_queue_start_label=work_queue_handoff["start_label"],
            work_queue_order=work_queue_handoff["queue_order"],
            work_queue_next_commands=work_queue_handoff["next_commands"],
            work_queue_next_safe_commands=work_queue_handoff["next_safe_commands"],
            work_queue_next_command_count=work_queue_handoff["next_command_count"],
            work_queue_next_safe_command_count=work_queue_handoff["next_safe_command_count"],
            work_queue_authorizes_execution=work_queue_handoff["boundaries"]["authorizes_execution"],
            work_queue_authorizes_completion_claim=work_queue_handoff["boundaries"]["authorizes_completion_claim"],
            work_queue_approval_granted=work_queue_handoff["boundaries"]["approval_granted"],
            work_queue_boundaries=work_queue_handoff["boundaries"],
            work_queue_handoff=work_queue_handoff,
        )
        return "\n".join(lines), metadata

    def safe_next_actions(args: dict[str, Any]) -> ToolResult:
        body, metadata = build_safe_next_actions(_bounded_limit(args.get("limit", 5), 5))
        return ToolResult(
            "safe_next_actions",
            True,
            body,
            metadata,
        )

    def next_action_packet(args: dict[str, Any]) -> ToolResult:
        approvals = store.list_pending_approvals(limit=5)
        tasks = sorted(store.list_tasks(status="open", limit=8), key=lambda row: (_priority_rank(row["priority"]), row["created_at"]))
        goals = store.list_goals(status="active", limit=8)
        decisions = store.list_decisions(status="active", limit=3)
        preferences = store.list_preferences(status="active", limit=3)
        jobs = store.list_jobs()
        background_state = _scheduler_background_state(jobs)

        action_kind = "capture_task"
        action_title = "Capture one concrete next task"
        command = "add task <next concrete Jarvis task> priority normal"
        rationale = "No open task or active goal is visible, so the safest next move is to make the work explicit before acting."
        verification = "Run `work queue` and confirm the new task appears."
        risk = "READ_ONLY planning; suggested command is LOCAL_SAFE if used."

        if approvals:
            approval = approvals[0]
            action_kind = "review_approval"
            action_title = f"Run readiness for approval #{approval['id']} before risky work"
            command = f"approval readiness {approval['id']}"
            rationale = "A blocked risky request is pending. Check queue position, staleness, exact arguments, the last-look approval packet, and approval chain proof before deciding."
            verification = f"Then run `approval packet {approval['id']}` only if readiness points to it; do not approve unless the operator explicitly trusts the exact action."
            risk = "READ_ONLY review; approval or rerun remains gated."
        health = _execution_health_snapshot()
        doctor = _doctor_handoff_snapshot()
        approval_handoff = _approval_handoff_snapshot()
        if not approvals and health["review_required"]:
            action_kind = "recover_execution"
            action_title = f"Review recovery for run #{health['first_problem_run_id']}: {health['first_problem_tool']}"
            command = health["next_command"]
            rationale = "A failed or weakly evidenced action run is visible; recovery review comes before normal task follow-through."
            verification = "Run `execution health report` after the recovery packet and confirm the review-required signal is understood or resolved."
            risk = "READ_ONLY recovery review; retries and risky execution remain approval-gated."
        elif not approvals and tasks:
            task = tasks[0]
            action_kind = "work_task"
            action_title = f"Work task #{task['id']}: {_short(task['body'])}"
            command = f"complete task {task['id']}"
            rationale = "The top open task is the clearest low-risk continuation target."
            verification = "Run `work queue` and confirm the task is gone or the remaining task order still makes sense."
            risk = "LOCAL_SAFE only when marking complete; do the actual work through appropriate gated tools first."
        elif not approvals and goals:
            goal = goals[0]
            open_steps = [step for step in store.list_goal_steps(goal["id"]) if step["status"] != "done"]
            if open_steps:
                step = open_steps[0]
                action_kind = "advance_goal_step"
                action_title = f"Advance goal #{goal['id']}: {_short(goal['title'])}"
                command = f"complete goal step {step['id']}"
                rationale = f"The next open goal step is visible: {_short(step['body'])}"
                verification = f"Run `show goal {goal['id']}` and confirm the next step changed."
                risk = "LOCAL_SAFE only when marking complete; any real-world action still follows its tool risk."
            else:
                action_kind = "define_goal_step"
                action_title = f"Define the next step for goal #{goal['id']}: {_short(goal['title'])}"
                command = f"add step to goal {goal['id']}: <next concrete step>"
                rationale = "The goal is active but has no open step, so planning should be made explicit first."
                verification = f"Run `show goal {goal['id']}` and confirm the step appears."
                risk = "LOCAL_SAFE planning write."
        elif background_state["action_kind"]:
            action_kind = background_state["action_kind"]
            action_title = background_state["action_title"]
            command = background_state["command"]
            rationale = background_state["rationale"]
            verification = background_state["verification"]
            risk = background_state["risk"]

        next_action_handoff = _next_action_packet_handoff(
            action_kind=action_kind,
            action_title=action_title,
            command=command,
            rationale=rationale,
            verification=verification,
            risk=risk,
            approvals=approvals,
            tasks=tasks,
            goals=goals,
            decisions=decisions,
            preferences=preferences,
            background_state=background_state,
            health=health,
            doctor=doctor,
            approval_handoff=approval_handoff,
        )

        lines = [
            "Jarvis next action packet:",
            "This is read-only. It chooses one proposed next move without executing it, writing notes, approving requests, controlling the computer, or queuing approvals.",
            "",
            *_priority_goal_summary(),
            "",
            "Recommended move:",
            f"- kind: {action_kind}",
            f"- action: {action_title}",
            f"- command to review/run later: `{command}`",
            f"- risk: {risk}",
            "",
            "Why this move:",
            f"- {rationale}",
            "",
            "Execution health:",
            f"- failed action runs: {health['failed_action_runs']}",
            f"- approval-held action runs: {health['approval_held_action_runs']}",
            f"- next health command: `{health['next_command']}`",
            f"- blocker categories: {', '.join(health['blocker_categories']) if health['blocker_categories'] else 'none'}",
            f"- verification coverage: {health['verification_coverage_state']}",
            f"- recovery queue: {', '.join(f'`{command}`' for command in health['next_commands'][:3]) if health['next_commands'] else 'none'}",
            *_execution_health_learning_handoff_lines(health),
            *_approval_handoff_lines(approval_handoff),
            *_approval_proof_chain_handoff_lines(health),
            *_doctor_handoff_lines(doctor),
            "",
            "Verification:",
            f"- {verification}",
            "",
            "Guardrails:",
            "- If the move needs shell/code, personal data, external side effects, destructive file changes, or computer control, stop and use `action rehearsal` or `approval review` first.",
            "- This packet is planning only. It does not complete tasks or approve blocked work.",
        ]
        return ToolResult(
            "next_action_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                action_kind=action_kind,
                action_title=_safe_text(action_title),
                action_command=_safe_text(command),
                action_rationale=_safe_text(rationale),
                action_verification=_safe_text(verification),
                action_risk=_safe_text(risk),
                pending_approvals=len(approvals),
                open_tasks=len(tasks),
                active_goals=len(goals),
                active_decisions=len(decisions),
                active_preferences=len(preferences),
                enabled_jobs=len(background_state["enabled_jobs"]),
                state_snapshot_jobs=len(background_state["state_snapshot_jobs"]),
                enabled_state_snapshot_jobs=len(background_state["enabled_state_snapshot_jobs"]),
                disabled_state_snapshot_jobs=(
                    len(background_state["state_snapshot_jobs"]) - len(background_state["enabled_state_snapshot_jobs"])
                ),
                scheduler_next_command=background_state["command"],
                failed_action_runs=health["failed_action_runs"],
                approval_held_action_runs=health["approval_held_action_runs"],
                execution_health_review_required=health["review_required"],
                execution_health_next_command=health["next_command"],
                execution_health_next_commands=health["next_commands"],
                execution_health_next_command_count=len(health["next_commands"]),
                execution_health_blocker_categories=health["blocker_categories"],
                execution_health_blocker_count=health["blocker_count"],
                execution_health_verification_coverage=health["verification_coverage_state"],
                execution_health_repeated_failure_count=health["repeated_failure_count"],
                execution_health_approval_proof_chains=health["approval_proof_chains"],
                execution_health_approval_proof_chain_count=health["approval_proof_chain_count"],
                **_execution_health_learning_handoff_metadata(health),
                doctor_next_audit_command=doctor["next_audit_command"],
                doctor_completion_claim_state=doctor["completion_claim_state"],
                doctor_completion_claim_ready=doctor["completion_claim_ready"],
                doctor_completion_blockers=doctor["completion_blockers"],
                doctor_completion_blocker_count=doctor["completion_blocker_count"],
                doctor_recent_failed_runs=doctor["recent_failed_runs"],
                doctor_recent_approval_held_runs=doctor["recent_approval_held_runs"],
                doctor_completion_proof_queue=doctor["completion_proof_queue"],
                doctor_completion_proof_queue_count=doctor["completion_proof_queue_count"],
                doctor_completion_next_proof_command=doctor["completion_next_proof_command"],
                doctor_recent_verification_runs=doctor["recent_verification_runs"],
                doctor_recent_after_action_learning_runs=doctor["recent_after_action_learning_runs"],
                **_doctor_audit_readability_handoff_metadata(doctor),
                doctor_recovery_closure_state=doctor["recovery_closure_state"],
                doctor_recovery_closure_missing=doctor["recovery_closure_missing"],
                doctor_recovery_closure_missing_count=doctor["recovery_closure_missing_count"],
                doctor_recovery_closure_required_commands=doctor["recovery_closure_required_commands"],
                doctor_recovery_closure_next_required_command=doctor["recovery_closure_next_required_command"],
                doctor_recovery_closure_proof_queue=doctor["recovery_closure_proof_queue"],
                doctor_recovery_closure_proof_queue_count=doctor["recovery_closure_proof_queue_count"],
                doctor_recovery_closure_next_proof_command=doctor["recovery_closure_next_proof_command"],
                doctor_recovery_closure_ready_to_retry=doctor["recovery_closure_ready_to_retry"],
                doctor_recovery_closure_blocks_completion_claim=doctor["recovery_closure_blocks_completion_claim"],
                doctor_recovery_closure_checklist_command=doctor["recovery_closure_checklist_command"],
                doctor_recovery_closure_should_open_checklist=doctor["recovery_closure_should_open_checklist"],
                doctor_execution_learning_state=doctor["execution_learning_state"],
                doctor_execution_learning_missing=doctor["execution_learning_missing"],
                doctor_execution_learning_missing_count=doctor["execution_learning_missing_count"],
                doctor_execution_learning_required_commands=doctor["execution_learning_required_commands"],
                doctor_execution_learning_next_required_command=doctor["execution_learning_next_required_command"],
                doctor_execution_learning_proof_queue=doctor["execution_learning_proof_queue"],
                doctor_execution_learning_proof_queue_count=doctor["execution_learning_proof_queue_count"],
                doctor_execution_learning_next_proof_command=doctor["execution_learning_next_proof_command"],
                doctor_execution_learning_actionable_required_commands=doctor["execution_learning_actionable_required_commands"],
                doctor_execution_learning_actionable_required_command_count=doctor[
                    "execution_learning_actionable_required_command_count"
                ],
                doctor_execution_learning_actionable_next_required_command=doctor[
                    "execution_learning_actionable_next_required_command"
                ],
                doctor_execution_learning_actionable_proof_queue=doctor["execution_learning_actionable_proof_queue"],
                doctor_execution_learning_actionable_proof_queue_count=doctor[
                    "execution_learning_actionable_proof_queue_count"
                ],
                doctor_execution_learning_actionable_next_proof_command=doctor[
                    "execution_learning_actionable_next_proof_command"
                ],
                doctor_execution_learning_next_evidence_command=doctor["execution_learning_next_evidence_command"],
                doctor_execution_learning_blocks_completion_claim=doctor["execution_learning_blocks_completion_claim"],
                **_doctor_agi_handoff_metadata(doctor),
                **_approval_handoff_metadata(approval_handoff),
                next_action_packet_handoff_ready=next_action_handoff["handoff_ready"],
                next_action_packet_ready_for_operator=next_action_handoff["ready_for_operator"],
                next_action_packet_state_changed=next_action_handoff["state_changed"],
                next_action_packet_changed=next_action_handoff["changed"],
                next_action_packet_content_in_handoff=next_action_handoff["content_in_handoff"],
                next_action_packet_action_kind=next_action_handoff["action_kind"],
                next_action_packet_command=next_action_handoff["command"],
                next_action_packet_review_order=next_action_handoff["review_order"],
                next_action_packet_next_commands=next_action_handoff["next_commands"],
                next_action_packet_next_safe_commands=next_action_handoff["next_safe_commands"],
                next_action_packet_next_command_count=next_action_handoff["next_command_count"],
                next_action_packet_next_safe_command_count=next_action_handoff["next_safe_command_count"],
                next_action_packet_authorizes_execution=next_action_handoff["boundaries"]["authorizes_execution"],
                next_action_packet_authorizes_completion_claim=next_action_handoff["boundaries"]["authorizes_completion_claim"],
                next_action_packet_approval_granted=next_action_handoff["boundaries"]["approval_granted"],
                next_action_packet_boundaries=next_action_handoff["boundaries"],
                next_action_packet_handoff=next_action_handoff,
            ),
        )

    def priority_stack(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_limit(args.get("limit", 8), 8)
        approvals = store.list_pending_approvals(limit=limit)
        tasks = sorted(store.list_tasks(status="open", limit=limit), key=lambda row: (_priority_rank(row["priority"]), row["created_at"]))
        goals = store.list_goals(status="active", limit=limit)
        decisions = store.list_decisions(status="active", limit=3)
        preferences = store.list_preferences(status="active", limit=3)
        jobs = store.list_jobs()
        background_state = _scheduler_background_state(jobs)
        enabled_jobs = background_state["enabled_jobs"]
        state_snapshot_jobs = background_state["state_snapshot_jobs"]
        enabled_state_snapshot_jobs = background_state["enabled_state_snapshot_jobs"]
        disabled_state_snapshot_jobs = len(state_snapshot_jobs) - len(enabled_state_snapshot_jobs)
        health = _execution_health_snapshot()
        doctor = _doctor_handoff_snapshot()
        approval_handoff = _approval_handoff_snapshot()

        stack: list[dict[str, Any]] = []
        if approvals:
            first = approvals[0]
            stack.append(
                {
                    "rank": 1,
                    "kind": "approval_review",
                    "title": f"Review approval #{first['id']} before any risky continuation",
                    "why": "Pending risky work can hide stale, private, destructive, broad, or out-of-order requests; readiness review comes before the last-look approval packet and approval chain proof.",
                    "command": f"approval readiness {first['id']}",
                    "risk": "READ_ONLY review; approving remains explicit.",
                }
            )

        if health["review_required"]:
            stack.append(
                {
                    "rank": len(stack) + 1,
                    "kind": "execution_health",
                    "title": f"Review recovery for run #{health['first_problem_run_id']}: {health['first_problem_tool']}",
                    "why": "A failed or weakly evidenced action run is visible; recovery review keeps Jarvis from treating broken work as progress.",
                    "command": health["next_command"],
                    "risk": "READ_ONLY recovery review; retries and risky execution remain approval-gated.",
                }
            )

        if tasks:
            task = tasks[0]
            priority = f" [{task['priority']}]" if task["priority"] != "normal" else ""
            due = f" due {task['due']}" if task["due"] else ""
            stack.append(
                {
                    "rank": len(stack) + 1,
                    "kind": "task",
                    "title": f"Work task #{task['id']}: {task['body']}{due}{priority}",
                    "why": "A concrete open task is the clearest safe unit of progress.",
                    "command": f"complete task {task['id']}",
                    "risk": "LOCAL_SAFE only when marking complete; real work still follows tool risk.",
                }
            )

        for goal in goals[:2]:
            open_steps = [step for step in store.list_goal_steps(goal["id"]) if step["status"] != "done"]
            if open_steps:
                step = open_steps[0]
                title = f"Advance goal #{goal['id']}: {goal['title']} -> {step['body']}"
                command = f"complete goal step {step['id']}"
                why = "Active goals keep multi-step Jarvis work from becoming scattered."
            else:
                title = f"Define next step for goal #{goal['id']}: {goal['title']}"
                command = f"add step to goal {goal['id']}: <next concrete step>"
                why = "An active goal without an open step needs a concrete next move before action."
            stack.append(
                {
                    "rank": len(stack) + 1,
                    "kind": "goal",
                    "title": title,
                    "why": why,
                    "command": command,
                    "risk": "LOCAL_SAFE planning/update only; external work remains gated.",
                }
            )

        if background_state["action_kind"] in {"resume_state_snapshot", "resume_conversation_compaction", "schedule_basics"}:
            title = background_state["action_title"]
            stack.append(
                {
                    "rank": len(stack) + 1,
                    "kind": "background_rhythm",
                    "title": title,
                    "why": background_state["rationale"],
                    "command": background_state["command"],
                    "risk": background_state["risk"],
                }
            )

        if not stack:
            stack.append(
                {
                    "rank": 1,
                    "kind": "capture_task",
                    "title": "Capture one concrete next task",
                    "why": "No approval, task, goal, or missing background rhythm is visible.",
                    "command": "add task <next concrete Jarvis task> priority normal",
                    "risk": "LOCAL_SAFE task capture.",
                }
            )
        priority_handoff = _priority_stack_handoff(
            limit=limit,
            stack=stack,
            approvals=approvals,
            tasks=tasks,
            goals=goals,
            decisions=decisions,
            preferences=preferences,
            background_state=background_state,
            health=health,
            doctor=doctor,
            approval_handoff=approval_handoff,
        )

        lines = [
            "Jarvis priority stack:",
            "This is read-only. It ranks what Jarvis should pay attention to before acting, without executing tools, writing notes, approving requests, controlling the computer, or queuing approvals.",
            "",
            *_priority_goal_summary(),
            "",
            "Ranking rule:",
            "- 1. Pending risky approvals get reviewed first, but not approved automatically.",
            "- 2. Execution-health recovery comes before normal task follow-through.",
            "- 3. High-priority concrete tasks beat vague goals when they move Jarvis toward the harness goal.",
            "- 4. Active goal steps keep longer Jarvis/harness work moving.",
            "- 5. Missing background context refreshes are setup work, not urgent action.",
            "",
            "Stack:",
        ]
        for item in stack[:limit]:
            lines.append(f"- #{item['rank']} [{item['kind']}] {item['title']}")
            lines.append(f"  why: {item['why']}")
            lines.append(f"  command to review/run later: `{item['command']}`")
            lines.append(f"  risk: {item['risk']}")

        lines.extend(["", "Context signals:"])
        lines.append(f"- pending approvals: {len(approvals)}")
        lines.append(f"- open tasks considered: {len(tasks)}")
        lines.append(f"- active goals considered: {len(goals)}")
        lines.append(f"- enabled scheduled jobs: {len(enabled_jobs)}")
        lines.append(f"- State Snapshot jobs: {len(state_snapshot_jobs)} total, {len(enabled_state_snapshot_jobs)} enabled")
        lines.append(f"- disabled State Snapshot jobs: {disabled_state_snapshot_jobs}")
        lines.append(f"- scheduler next command: `{background_state['command']}`")
        lines.append(f"- failed action runs: {health['failed_action_runs']}")
        lines.append(f"- approval-held action runs: {health['approval_held_action_runs']}")
        lines.append(f"- execution health next command: `{health['next_command']}`")
        lines.append(f"- execution health blockers: {', '.join(health['blocker_categories']) if health['blocker_categories'] else 'none'}")
        if health["next_commands"]:
            lines.append(f"- execution health recovery queue: {', '.join(f'`{command}`' for command in health['next_commands'][:4])}")
        lines.extend(_execution_health_learning_handoff_lines(health))
        lines.extend(_approval_handoff_lines(approval_handoff))
        lines.extend(_approval_proof_chain_handoff_lines(health))
        lines.extend(_doctor_handoff_lines(doctor))
        if decisions:
            lines.append(f"- decision anchor: #{decisions[0]['id']} {_short(decisions[0]['title'])}")
        if preferences:
            preference = preferences[0]
            lines.append(f"- active preference: {_short(preference['key'], limit=80)} = {_short(preference['value'])}")
        lines.extend(
            [
                "",
                "Guardrails:",
                "- Use `approval readiness #ID`, `approval packet #ID`, and `approval chain proof #ID` before approving any risky blocked request.",
                "- Use `action rehearsal: <request>` before shell/code, personal data, external effects, destructive changes, or computer control.",
                "- This stack does not complete tasks or goal steps; it only explains the ordering.",
            ]
        )

        return ToolResult(
            "priority_stack",
            True,
            "\n".join(lines),
            _safe_metadata(
                items=len(stack[:limit]),
                top_kind=stack[0]["kind"],
                pending_approvals=len(approvals),
                open_tasks=len(tasks),
                active_goals=len(goals),
                active_decisions=len(decisions),
                active_preferences=len(preferences),
                enabled_jobs=len(enabled_jobs),
                state_snapshot_jobs=len(state_snapshot_jobs),
                enabled_state_snapshot_jobs=len(enabled_state_snapshot_jobs),
                disabled_state_snapshot_jobs=disabled_state_snapshot_jobs,
                scheduler_next_command=background_state["command"],
                failed_action_runs=health["failed_action_runs"],
                approval_held_action_runs=health["approval_held_action_runs"],
                execution_health_review_required=health["review_required"],
                execution_health_next_command=health["next_command"],
                execution_health_next_commands=health["next_commands"],
                execution_health_next_command_count=len(health["next_commands"]),
                execution_health_blocker_categories=health["blocker_categories"],
                execution_health_blocker_count=health["blocker_count"],
                execution_health_verification_coverage=health["verification_coverage_state"],
                execution_health_repeated_failure_count=health["repeated_failure_count"],
                execution_health_approval_proof_chains=health["approval_proof_chains"],
                execution_health_approval_proof_chain_count=health["approval_proof_chain_count"],
                **_execution_health_learning_handoff_metadata(health),
                doctor_next_audit_command=doctor["next_audit_command"],
                doctor_completion_claim_state=doctor["completion_claim_state"],
                doctor_completion_claim_ready=doctor["completion_claim_ready"],
                doctor_completion_blockers=doctor["completion_blockers"],
                doctor_completion_blocker_count=doctor["completion_blocker_count"],
                doctor_recent_failed_runs=doctor["recent_failed_runs"],
                doctor_recent_approval_held_runs=doctor["recent_approval_held_runs"],
                doctor_completion_proof_queue=doctor["completion_proof_queue"],
                doctor_completion_proof_queue_count=doctor["completion_proof_queue_count"],
                doctor_completion_next_proof_command=doctor["completion_next_proof_command"],
                doctor_recent_verification_runs=doctor["recent_verification_runs"],
                doctor_recent_after_action_learning_runs=doctor["recent_after_action_learning_runs"],
                **_doctor_audit_readability_handoff_metadata(doctor),
                doctor_recovery_closure_state=doctor["recovery_closure_state"],
                doctor_recovery_closure_missing=doctor["recovery_closure_missing"],
                doctor_recovery_closure_missing_count=doctor["recovery_closure_missing_count"],
                doctor_recovery_closure_required_commands=doctor["recovery_closure_required_commands"],
                doctor_recovery_closure_next_required_command=doctor["recovery_closure_next_required_command"],
                doctor_recovery_closure_proof_queue=doctor["recovery_closure_proof_queue"],
                doctor_recovery_closure_proof_queue_count=doctor["recovery_closure_proof_queue_count"],
                doctor_recovery_closure_next_proof_command=doctor["recovery_closure_next_proof_command"],
                doctor_recovery_closure_ready_to_retry=doctor["recovery_closure_ready_to_retry"],
                doctor_recovery_closure_blocks_completion_claim=doctor["recovery_closure_blocks_completion_claim"],
                doctor_recovery_closure_checklist_command=doctor["recovery_closure_checklist_command"],
                doctor_recovery_closure_should_open_checklist=doctor["recovery_closure_should_open_checklist"],
                doctor_execution_learning_state=doctor["execution_learning_state"],
                doctor_execution_learning_missing=doctor["execution_learning_missing"],
                doctor_execution_learning_missing_count=doctor["execution_learning_missing_count"],
                doctor_execution_learning_required_commands=doctor["execution_learning_required_commands"],
                doctor_execution_learning_next_required_command=doctor["execution_learning_next_required_command"],
                doctor_execution_learning_proof_queue=doctor["execution_learning_proof_queue"],
                doctor_execution_learning_proof_queue_count=doctor["execution_learning_proof_queue_count"],
                doctor_execution_learning_next_proof_command=doctor["execution_learning_next_proof_command"],
                doctor_execution_learning_actionable_required_commands=doctor["execution_learning_actionable_required_commands"],
                doctor_execution_learning_actionable_required_command_count=doctor[
                    "execution_learning_actionable_required_command_count"
                ],
                doctor_execution_learning_actionable_next_required_command=doctor[
                    "execution_learning_actionable_next_required_command"
                ],
                doctor_execution_learning_actionable_proof_queue=doctor["execution_learning_actionable_proof_queue"],
                doctor_execution_learning_actionable_proof_queue_count=doctor[
                    "execution_learning_actionable_proof_queue_count"
                ],
                doctor_execution_learning_actionable_next_proof_command=doctor[
                    "execution_learning_actionable_next_proof_command"
                ],
                doctor_execution_learning_next_evidence_command=doctor["execution_learning_next_evidence_command"],
                doctor_execution_learning_blocks_completion_claim=doctor["execution_learning_blocks_completion_claim"],
                **_doctor_agi_handoff_metadata(doctor),
                **_approval_handoff_metadata(approval_handoff),
                priority_stack_handoff_ready=priority_handoff["handoff_ready"],
                priority_stack_ready_for_operator=priority_handoff["ready_for_operator"],
                priority_stack_state_changed=priority_handoff["state_changed"],
                priority_stack_changed=priority_handoff["changed"],
                priority_stack_content_in_handoff=priority_handoff["content_in_handoff"],
                priority_stack_top_kind=priority_handoff["top_kind"],
                priority_stack_top_command=priority_handoff["top_command"],
                priority_stack_rows=priority_handoff["stack_rows"],
                priority_stack_count=priority_handoff["stack_count"],
                priority_stack_ranking_order=priority_handoff["ranking_order"],
                priority_stack_next_commands=priority_handoff["next_commands"],
                priority_stack_next_safe_commands=priority_handoff["next_safe_commands"],
                priority_stack_next_command_count=priority_handoff["next_command_count"],
                priority_stack_next_safe_command_count=priority_handoff["next_safe_command_count"],
                priority_stack_authorizes_execution=priority_handoff["boundaries"]["authorizes_execution"],
                priority_stack_authorizes_completion_claim=priority_handoff["boundaries"]["authorizes_completion_claim"],
                priority_stack_approval_granted=priority_handoff["boundaries"]["approval_granted"],
                priority_stack_boundaries=priority_handoff["boundaries"],
                priority_stack_handoff=priority_handoff,
                limit=limit,
            ),
        )

    def continuation_packet(args: dict[str, Any]) -> ToolResult:
        objective = _short(args.get("objective") or "continue building Jarvis V2 safely", limit=MAX_OBJECTIVE_CHARS)
        approvals = store.list_pending_approvals(limit=5)
        tasks = sorted(store.list_tasks(status="open", limit=5), key=lambda row: (_priority_rank(row["priority"]), row["created_at"]))
        goals = store.list_goals(status="active", limit=5)
        jobs = store.list_jobs()
        background_state = _scheduler_background_state(jobs)
        enabled_jobs = background_state["enabled_jobs"]
        state_snapshot_jobs = background_state["state_snapshot_jobs"]
        enabled_state_snapshot_jobs = background_state["enabled_state_snapshot_jobs"]
        disabled_state_snapshot_jobs = len(state_snapshot_jobs) - len(enabled_state_snapshot_jobs)
        health = _execution_health_snapshot()
        checkpoint_contract = _checkpoint_recovery_contract(objective, health)

        next_focus = "capture one concrete task before editing"
        if approvals:
            next_focus = f"review approval #{approvals[0]['id']} with `approval readiness {approvals[0]['id']}` before any risky continuation"
        elif health["review_required"]:
            next_focus = f"review execution recovery for run #{health['first_problem_run_id']} with `{health['next_command']}` before normal task follow-through"
        elif tasks:
            next_focus = f"work the top visible task #{tasks[0]['id']}: {_short(tasks[0]['body'])}"
        elif goals:
            open_steps = [step for step in store.list_goal_steps(goals[0]["id"]) if step["status"] != "done"]
            if open_steps:
                next_focus = f"advance goal #{goals[0]['id']}: {_short(open_steps[0]['body'])}"
            else:
                next_focus = f"define the next step for goal #{goals[0]['id']}: {_short(goals[0]['title'])}"
        elif background_state["action_kind"] == "resume_state_snapshot":
            next_focus = "resume the paused State Snapshot context refresh with `resume job State Snapshot`"
        elif background_state["action_kind"] == "resume_conversation_compaction":
            next_focus = f"resume Memory Trees conversation compaction with `resume job {COMPACTION_JOB_NAME}`"
        elif background_state["action_kind"] == "schedule_basics":
            next_focus = "enable safe scheduled context refreshes with `schedule assistant basics`"

        lines = [
            "Jarvis continuation packet:",
            "This is read-only. It prepares a safe build-continuation loop without editing files, executing tools, calling models, writing notes, approving requests, controlling the computer, or queuing approvals.",
            "",
            f"Objective: {objective}",
            *_priority_goal_summary(),
            "",
            f"Recommended focus: {next_focus}",
            "",
            "Loop:",
            "1. Orient with `handoff brief`, `build delta`, `priority stack`, and `work queue`.",
            "2. Check `execution health report`; resolve failed-run recovery review before normal task follow-through.",
            "3. Choose exactly one scoped assistant function or safety/UI contract to improve.",
            "4. Read the owning files before editing and keep changes inside the existing pattern.",
            "5. Add or update the closest smoke test before broad verification.",
            "6. Run focused tests, then the full smoke suite when the change touches shared routing, dashboard, approvals, or continuity.",
            "7. Run `acceptance gate: <changed behavior>; evidence <receipt>; tests <verification>; recovery <rollback or stop condition>` before treating the slice as done.",
            "8. Summarize what changed, what passed, and any remaining blocker.",
            "",
            "Approval boundaries:",
            "- Do not approve pending requests during an autonomous build loop.",
            "- Do not run shell/code through Jarvis tools without the operator approving the queued request.",
            "- Do not read personal data, use clipboard contents, control the computer, send messages, create reminders, delete data, or make external side effects without explicit approval.",
            "",
            "Stop conditions:",
            "- Stop and summarize if a test fails in a way that needs the operator's decision.",
            "- Stop before destructive commands, credential/account access, private-data reads, or broad refactors.",
            "- Stop after a coherent scoped improvement is implemented and verified.",
            "",
            "Verification ladder:",
            "- `python3 -m compileall jarvis_v2`",
            "- focused smoke test for the touched area",
            "- `python3 -m jarvis_v2.scripts.smoke_test_all`",
            "",
            "Context signals:",
            f"- pending approvals: {len(approvals)}",
            f"- open tasks considered: {len(tasks)}",
            f"- active goals considered: {len(goals)}",
            f"- enabled scheduled jobs: {len(enabled_jobs)}",
            f"- State Snapshot jobs: {len(state_snapshot_jobs)} total, {len(enabled_state_snapshot_jobs)} enabled",
            f"- disabled State Snapshot jobs: {disabled_state_snapshot_jobs}",
            f"- scheduler next command: `{background_state['command']}`",
            f"- failed action runs: {health['failed_action_runs']}",
            f"- approval-held action runs: {health['approval_held_action_runs']}",
            f"- execution health next command: `{health['next_command']}`",
            f"- execution health blockers: {', '.join(health['blocker_categories']) if health['blocker_categories'] else 'none'}",
            f"- execution health recovery queue: {', '.join(f'`{command}`' for command in health['next_commands'][:4]) if health['next_commands'] else 'none'}",
            "",
            "Checkpoint recovery contract:",
            f"- checkpoint found: {'yes' if checkpoint_contract['checkpoint_found'] else 'no'}",
            f"- checkpoint: {checkpoint_contract['checkpoint_path_display'] or 'none'}",
            f"- checkpoint freshness: {checkpoint_contract['checkpoint_freshness']}",
            f"- checkpoint age minutes: {checkpoint_contract['checkpoint_age_minutes'] if checkpoint_contract['checkpoint_age_minutes'] is not None else 'unknown'}",
            f"- recovery required before normal follow-through: {'yes' if checkpoint_contract['checkpoint_recovery_required'] else 'no'}",
            f"- next checkpoint proof: `{checkpoint_contract['checkpoint_recovery_next_command']}`",
            f"- recovery queue: {', '.join(f'`{command}`' for command in checkpoint_contract['checkpoint_recovery_queue'][:5])}",
            f"- verification target: `{checkpoint_contract['checkpoint_recovery_verification_target']}`",
            f"- stop condition: {checkpoint_contract['checkpoint_recovery_stop_condition']}",
            f"- follow-through gate: {checkpoint_contract['checkpoint_recovery_followthrough_gate']}",
            f"- checkpoint recovery follow-through packet: `{checkpoint_contract['checkpoint_recovery_followthrough_packet_command']}`",
        ]
        lines.extend(_execution_health_learning_handoff_lines(health))
        lines.extend(_approval_proof_chain_handoff_lines(health))
        if approvals:
            lines.append(f"- first approval blocker: #{approvals[0]['id']} {_short(approvals[0]['tool_name'], limit=80)}")
            lines.append(f"- approval review path: `approval readiness {approvals[0]['id']}` -> `approval packet {approvals[0]['id']}` -> `approval chain proof {approvals[0]['id']}`")
        if tasks:
            lines.append(f"- first task: #{tasks[0]['id']} {_short(tasks[0]['body'])}")
        if goals:
            lines.append(f"- first goal: #{goals[0]['id']} {_short(goals[0]['title'])}")
        continuation_handoff = _continuation_packet_handoff(
            objective=objective,
            next_focus=next_focus,
            approvals=approvals,
            tasks=tasks,
            goals=goals,
            enabled_jobs=enabled_jobs,
            state_snapshot_jobs=state_snapshot_jobs,
            enabled_state_snapshot_jobs=enabled_state_snapshot_jobs,
            disabled_state_snapshot_jobs=disabled_state_snapshot_jobs,
            scheduler_next_command=background_state["command"],
            health=health,
            checkpoint_contract=checkpoint_contract,
        )

        return ToolResult(
            "continuation_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                pending_approvals=len(approvals),
                open_tasks=len(tasks),
                active_goals=len(goals),
                enabled_jobs=len(enabled_jobs),
                state_snapshot_jobs=len(state_snapshot_jobs),
                enabled_state_snapshot_jobs=len(enabled_state_snapshot_jobs),
                disabled_state_snapshot_jobs=disabled_state_snapshot_jobs,
                scheduler_next_command=background_state["command"],
                failed_action_runs=health["failed_action_runs"],
                approval_held_action_runs=health["approval_held_action_runs"],
                execution_health_review_required=health["review_required"],
                execution_health_next_command=health["next_command"],
                execution_health_next_commands=health["next_commands"],
                execution_health_next_command_count=len(health["next_commands"]),
                execution_health_blocker_categories=health["blocker_categories"],
                execution_health_blocker_count=health["blocker_count"],
                execution_health_verification_coverage=health["verification_coverage_state"],
                execution_health_repeated_failure_count=health["repeated_failure_count"],
                execution_health_approval_proof_chains=health["approval_proof_chains"],
                execution_health_approval_proof_chain_count=health["approval_proof_chain_count"],
                checkpoint_found=checkpoint_contract["checkpoint_found"],
                checkpoint_path=checkpoint_contract["checkpoint_path"],
                checkpoint_path_display=checkpoint_contract["checkpoint_path_display"],
                checkpoint_age_minutes=checkpoint_contract["checkpoint_age_minutes"],
                checkpoint_freshness=checkpoint_contract["checkpoint_freshness"],
                checkpoint_recovery_required=checkpoint_contract["checkpoint_recovery_required"],
                checkpoint_recovery_queue=checkpoint_contract["checkpoint_recovery_queue"],
                checkpoint_recovery_queue_count=checkpoint_contract["checkpoint_recovery_queue_count"],
                checkpoint_recovery_next_command=checkpoint_contract["checkpoint_recovery_next_command"],
                checkpoint_recovery_verification_target=checkpoint_contract["checkpoint_recovery_verification_target"],
                checkpoint_recovery_stop_condition=checkpoint_contract["checkpoint_recovery_stop_condition"],
                checkpoint_recovery_followthrough_gate=checkpoint_contract["checkpoint_recovery_followthrough_gate"],
                checkpoint_recovery_followthrough_packet_command=checkpoint_contract["checkpoint_recovery_followthrough_packet_command"],
                continuation_packet_handoff_ready=continuation_handoff["handoff_ready"],
                continuation_packet_ready_for_operator=continuation_handoff["ready_for_operator"],
                continuation_packet_state_changed=continuation_handoff["state_changed"],
                continuation_packet_changed=continuation_handoff["changed"],
                continuation_packet_content_in_handoff=continuation_handoff["content_in_handoff"],
                continuation_packet_recommended_focus=continuation_handoff["recommended_focus"],
                continuation_packet_next_commands=continuation_handoff["next_commands"],
                continuation_packet_next_safe_commands=continuation_handoff["next_safe_commands"],
                continuation_packet_next_command_count=continuation_handoff["next_command_count"],
                continuation_packet_next_safe_command_count=continuation_handoff["next_safe_command_count"],
                continuation_packet_authorizes_execution=continuation_handoff["boundaries"]["authorizes_execution"],
                continuation_packet_authorizes_completion_claim=continuation_handoff["boundaries"]["authorizes_completion_claim"],
                continuation_packet_approval_granted=continuation_handoff["boundaries"]["approval_granted"],
                continuation_packet_boundaries=continuation_handoff["boundaries"],
                continuation_packet_handoff=continuation_handoff,
                **_execution_health_learning_handoff_metadata(health),
            ),
        )

    def build_target_packet(args: dict[str, Any]) -> ToolResult:
        objective = _short(args.get("objective") or "continue Jarvis V2 safely", limit=MAX_OBJECTIVE_CHARS)
        approvals = store.list_pending_approvals(limit=5)
        tasks = sorted(store.list_tasks(status="open", limit=8), key=lambda row: (_priority_rank(row["priority"]), row["created_at"]))
        goals = store.list_goals(status="active", limit=5)
        jobs = store.list_jobs()
        background_state = _scheduler_background_state(jobs)
        enabled_jobs = background_state["enabled_jobs"]
        state_snapshot_jobs = background_state["state_snapshot_jobs"]
        enabled_state_snapshot_jobs = background_state["enabled_state_snapshot_jobs"]
        disabled_state_snapshot_jobs = len(state_snapshot_jobs) - len(enabled_state_snapshot_jobs)
        health = _execution_health_snapshot()

        target_kind = "continuity_contract"
        target_title = "Add one inspectable read-only assistant function or status contract"
        target_reason = "Jarvis is safest when each autonomy layer is previewable before it can act."
        first_boundary = "Keep the change read-only until a smoke test proves it does not queue approvals or perform side effects."

        if approvals:
            approval = approvals[0]
            target_kind = "approval_visibility"
            target_title = f"Improve readiness visibility around approval #{approval['id']} before risky continuation"
            target_reason = "Pending approvals are the first safety bottleneck; make queue position, staleness, exact args, and last-look review clearer before considering approval."
            first_boundary = f"Do not approve approval #{approval['id']}; only improve preview, wording, API, or tests."
        elif health["review_required"]:
            target_kind = "execution_recovery_visibility"
            target_title = f"Improve recovery visibility for failed run #{health['first_problem_run_id']}"
            target_reason = "A failed action run should be reviewed through recovery and health packets before normal task follow-through."
            first_boundary = f"Do not retry run #{health['first_problem_run_id']}; only improve recovery review, wording, API, or tests."
        elif tasks:
            task = tasks[0]
            target_kind = "task_followthrough"
            target_title = f"Turn task #{task['id']} into one verifiable assistant behavior"
            target_reason = f"The top open task is concrete enough to guide a scoped build target: {_short(task['body'])}."
            first_boundary = "Do the implementation through code/tests here, but do not mark the task complete until verification passes."
        elif goals:
            goal = goals[0]
            open_steps = [step for step in store.list_goal_steps(goal["id"]) if step["status"] != "done"]
            if open_steps:
                target_kind = "goal_step"
                target_title = f"Implement one slice for goal #{goal['id']}: {_short(goal['title'])}"
                target_reason = f"The next visible goal step is: {_short(open_steps[0]['body'])}."
                first_boundary = "Keep the slice narrow and leave unrelated goal steps untouched."
            else:
                target_kind = "goal_planning"
                target_title = f"Make goal #{goal['id']} actionable by defining the next step"
                target_reason = "An active goal without an open step needs a concrete plan before code changes."
                first_boundary = "Suggest the step only; a real goal update is a separate local-safe action."
        elif background_state["action_kind"] in {"resume_state_snapshot", "resume_conversation_compaction", "schedule_basics"}:
            target_kind = "background_continuity"
            target_title = background_state["action_title"] or "Improve safe scheduled context refresh readiness"
            if background_state["action_kind"] == "resume_state_snapshot":
                target_reason = "A State Snapshot continuity job already exists but is paused, so Jarvis should point to resuming it instead of creating duplicate default jobs."
                first_boundary = "Do not create duplicate default jobs from this packet; use `resume job State Snapshot` separately if the operator wants the paused continuity job re-enabled."
            elif background_state["action_kind"] == "resume_conversation_compaction":
                target_reason = "A Conversation Compaction job already exists but is paused, so Jarvis should point to resuming it instead of losing old conversation memory."
                first_boundary = f"Do not create duplicate default jobs from this packet; use `resume job {COMPACTION_JOB_NAME}` separately if the operator wants Memory Trees compaction re-enabled."
            else:
                target_reason = background_state["rationale"] or "No background rhythm is visible, so continuity will decay between sessions."
                first_boundary = "Do not create jobs from this packet; use `schedule assistant basics` separately if the operator wants missing local-safe background jobs created."

        likely_files = [
            "jarvis_v2/tools/next_step.py",
            "jarvis_v2/agent/planner.py",
            "jarvis_v2/tools/registry.py",
            "jarvis_v2/tools/help.py",
            "jarvis_v2/tools/capabilities.py",
            "jarvis_v2/scripts/smoke_test_next_step.py",
        ]
        if target_kind == "approval_visibility":
            likely_files.extend(["jarvis_v2/tools/approvals.py", "jarvis_v2/scripts/smoke_test_approval_review.py"])
        if target_kind == "background_continuity":
            likely_files.extend(["jarvis_v2/tools/scheduler.py", "jarvis_v2/scripts/smoke_test_scheduler_basics.py"])
        file_integrity = _target_file_integrity(likely_files)
        focused_verification_commands = [
            "python3 -m jarvis_v2.scripts.smoke_test_next_step",
            "python3 -m compileall jarvis_v2",
        ]
        aggregate_verification_command = "python3 -m jarvis_v2.scripts.smoke_test_all"
        acceptance_gate_command = "acceptance gate: <changed behavior>; evidence <receipt>; tests <verification>; recovery <rollback or stop condition>"

        lines = [
            "Jarvis build target packet:",
            "This is read-only. It selects one scoped build target without editing files, executing tools, calling models, writing notes, approving requests, controlling the computer, or queuing approvals.",
            "",
            f"Objective: {objective}",
            *_priority_goal_summary(),
            "",
            "Selected target:",
            f"- kind: {target_kind}",
            f"- title: {target_title}",
            f"- why: {target_reason}",
            f"- scheduler next command: `{background_state['command']}`",
            "",
            "Likely owning files:",
        ]
        for path in likely_files:
            lines.append(f"- `{path}`")

        lines.extend(
            [
                "",
                "Target integrity:",
                f"- file check: {file_integrity['status']}",
                f"- files checked: {file_integrity['checked']}",
                f"- missing files: {', '.join(file_integrity['missing']) if file_integrity['missing'] else 'none'}",
                "",
                "Smoke-test target:",
                f"- Start with `{focused_verification_commands[0]}` for continuity routing and read-only metadata.",
                f"- Run `{focused_verification_commands[1]}` after edits.",
                f"- Run `{aggregate_verification_command}` when registry, planner, dashboard, approvals, or shared continuity contracts change.",
                "",
                "Acceptance checks:",
                "- The new or changed surface states its safety boundary plainly.",
                "- Metadata reports `calls_model=False`, `executes_tools=False`, `queues_approval=False`, `writes_notes=False`, `edits_files=False`, and `controls_computer=False` when the packet itself is read-only.",
                "- Pending approvals count does not change while previewing the packet.",
            "- The help/capability map includes the new command if it is user-facing.",
            "- `execution health report` points to the next safe audit or recovery command when recent action runs are unhealthy.",
            "- Verification-to-learning handoff names the verification receipt, recovery packet, and learning packet for the selected action run.",
            f"- `{acceptance_gate_command}` can be run before calling the slice done.",
                "",
                "First boundary:",
                f"- {first_boundary}",
                "",
                "Stop conditions:",
                "- Stop before broad refactors, destructive commands, personal-data reads, external side effects, or real computer control.",
                "- Stop and summarize if focused smoke tests fail for reasons that need the operator's decision.",
                "- Stop after one coherent target is implemented and verified.",
            ]
        )
        build_target_handoff = _build_target_packet_handoff(
            objective=objective,
            target_kind=target_kind,
            target_title=target_title,
            target_reason=target_reason,
            first_boundary=first_boundary,
            likely_files=likely_files,
            file_integrity=file_integrity,
            focused_verification_commands=focused_verification_commands,
            aggregate_verification_command=aggregate_verification_command,
            acceptance_gate_command=acceptance_gate_command,
            approvals=approvals,
            tasks=tasks,
            goals=goals,
            enabled_jobs=enabled_jobs,
            state_snapshot_jobs=state_snapshot_jobs,
            enabled_state_snapshot_jobs=enabled_state_snapshot_jobs,
            disabled_state_snapshot_jobs=disabled_state_snapshot_jobs,
            scheduler_next_command=background_state["command"],
            health=health,
        )

        return ToolResult(
            "build_target_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                target_kind=target_kind,
                likely_files=len(likely_files),
                likely_file_paths=likely_files,
                target_file_integrity_status=file_integrity["status"],
                target_files_checked=file_integrity["checked"],
                target_files_exist=file_integrity["all_exist"],
                missing_target_files=file_integrity["missing"],
                missing_target_file_count=file_integrity["missing_count"],
                target_file_rows=file_integrity["rows"],
                focused_verification_commands=focused_verification_commands,
                focused_verification_command_count=len(focused_verification_commands),
                aggregate_verification_command=aggregate_verification_command,
                acceptance_gate_command=acceptance_gate_command,
                pending_approvals=len(approvals),
                open_tasks=len(tasks),
                active_goals=len(goals),
                enabled_jobs=len(enabled_jobs),
                state_snapshot_jobs=len(state_snapshot_jobs),
                enabled_state_snapshot_jobs=len(enabled_state_snapshot_jobs),
                disabled_state_snapshot_jobs=disabled_state_snapshot_jobs,
                scheduler_next_command=background_state["command"],
                failed_action_runs=health["failed_action_runs"],
                approval_held_action_runs=health["approval_held_action_runs"],
                execution_health_review_required=health["review_required"],
                execution_health_next_command=health["next_command"],
                execution_health_next_commands=health["next_commands"],
                execution_health_next_command_count=len(health["next_commands"]),
                execution_health_blocker_categories=health["blocker_categories"],
                execution_health_blocker_count=health["blocker_count"],
                execution_health_verification_coverage=health["verification_coverage_state"],
                execution_health_repeated_failure_count=health["repeated_failure_count"],
                execution_health_approval_proof_chains=health["approval_proof_chains"],
                execution_health_approval_proof_chain_count=health["approval_proof_chain_count"],
                build_target_packet_handoff_ready=build_target_handoff["handoff_ready"],
                build_target_packet_ready_for_operator=build_target_handoff["ready_for_operator"],
                build_target_packet_state_changed=build_target_handoff["state_changed"],
                build_target_packet_changed=build_target_handoff["changed"],
                build_target_packet_content_in_handoff=build_target_handoff["content_in_handoff"],
                build_target_packet_target_title=build_target_handoff["target_title"],
                build_target_packet_first_boundary=build_target_handoff["first_boundary"],
                build_target_packet_next_commands=build_target_handoff["next_commands"],
                build_target_packet_next_safe_commands=build_target_handoff["next_safe_commands"],
                build_target_packet_next_command_count=build_target_handoff["next_command_count"],
                build_target_packet_next_safe_command_count=build_target_handoff["next_safe_command_count"],
                build_target_packet_authorizes_execution=build_target_handoff["boundaries"]["authorizes_execution"],
                build_target_packet_authorizes_completion_claim=build_target_handoff["boundaries"]["authorizes_completion_claim"],
                build_target_packet_approval_granted=build_target_handoff["boundaries"]["approval_granted"],
                build_target_packet_boundaries=build_target_handoff["boundaries"],
                build_target_packet_handoff=build_target_handoff,
                **_execution_health_learning_handoff_metadata(health),
            ),
        )

    def harness_build_slice(args: dict[str, Any]) -> ToolResult:
        objective = _short(args.get("objective") or "continue building Jarvis V2 as an agent harness", limit=MAX_OBJECTIVE_CHARS)
        focus = _short(args.get("focus") or args.get("area") or "", limit=120).lower()
        approvals = store.list_pending_approvals(limit=5)
        tasks = sorted(store.list_tasks(status="open", limit=8), key=lambda row: (_priority_rank(row["priority"]), row["created_at"]))
        goals = store.list_goals(status="active", limit=5)
        health = _execution_health_snapshot()
        recovery_debt_visible = bool(health["review_required"])
        execution_learning_debt_visible = bool(
            health.get("execution_learning_blocks_completion_claim")
            or health.get("execution_learning_required_commands")
            or health.get("execution_learning_missing")
        )
        recovery_or_learning_debt_visible = recovery_debt_visible or execution_learning_debt_visible
        read_only_slice_allowed_with_recovery_debt = True
        recovery_debt_blocks_current_slice = False

        target_kind = "execution_readiness"
        target_title = "Strengthen one read-only route, proof packet, or status API for safer execution."
        target_reason = "Execution-readiness is the harness steering layer: Jarvis should know the route, gates, proof, recovery, and learning hook before acting."
        likely_files = [
            "jarvis_v2/tools/autonomy.py",
            "jarvis_v2/tools/harness.py",
            "jarvis_v2/tools/registry.py",
            "jarvis_v2/agent/planner.py",
            "jarvis_v2/scripts/smoke_test_harness.py",
            "jarvis_v2/scripts/smoke_test_status_server.py",
        ]
        focused_tests = [
            "python3 -m py_compile <touched files>",
            "python3 -m jarvis_v2.scripts.smoke_test_harness",
            "python3 -m compileall -q jarvis_v2",
        ]
        starter_commands = [
            "harness operations",
            "execution health report",
            "completion claim gate",
        ]
        acceptance_checks = [
            "changed behavior is covered by a focused smoke test",
            "read-only packets keep unsafe metadata flags false",
            "execution health report is clear or names the next recovery command",
            "acceptance gate packet includes evidence, tests, and recovery/stop condition",
        ]
        selected_slice_command = "harness build slice"
        completion_audit_command = "completion audit: improve selected harness build slice"
        evidence_ledger_command = "evidence ledger"
        completion_claim_gate_command = "completion claim gate: improve selected harness build slice"

        if any(marker in f"{objective} {focus}".lower() for marker in ("connector", "personal", "email", "calendar", "messages", "drive")):
            target_kind = "personal_integration_harness"
            target_title = "Move personal connectors toward implementation without enabling accounts or personal-data reads."
            target_reason = "Future AGI-like behavior needs personal connectors, but the harness must prove scope, approval, audit, verification, and rollback first."
            likely_files = [
                "jarvis_v2/tools/personal.py",
                "jarvis_v2/tools/registry.py",
                "jarvis_v2/agent/planner.py",
                "jarvis_v2/tools/help.py",
                "jarvis_v2/tools/capabilities.py",
                "jarvis_v2/tools/harness.py",
                "jarvis_v2/ui/status_server.py",
                "jarvis_v2/scripts/smoke_test_personal.py",
                "jarvis_v2/scripts/smoke_test_harness.py",
                "jarvis_v2/scripts/smoke_test_status_server.py",
            ]
            focused_tests = [
                "python3 -m py_compile jarvis_v2/tools/personal.py jarvis_v2/tools/registry.py jarvis_v2/agent/planner.py jarvis_v2/tools/help.py jarvis_v2/tools/capabilities.py jarvis_v2/tools/harness.py jarvis_v2/ui/status_server.py",
                "python3 -m jarvis_v2.scripts.smoke_test_personal",
                "python3 -m jarvis_v2.scripts.smoke_test_harness",
                "python3 -m jarvis_v2.scripts.smoke_test_status_server",
                "python3 -m compileall -q jarvis_v2",
            ]
            connector_focus = focus if focus in {"calendar", "email", "messages", "contacts", "reminders", "browser"} else "email"
            metadata_action_by_connector = {
                "calendar": "list calendars",
                "email": "search mailbox metadata",
                "messages": "summarize selected conversations",
                "contacts": "search contact names",
                "reminders": "list selected reminder lists",
                "browser": "fetch public pages",
            }
            metadata_action = metadata_action_by_connector.get(connector_focus, "search mailbox metadata")
            starter_commands = [
                f"integration execution matrix: {connector_focus}",
                f"integration adapter manifest: {connector_focus}",
                f"integration adapter acceptance: {connector_focus}",
                f"integration dry run contract: {connector_focus} -> {metadata_action}; target selected source; time selected window; data metadata-only; verification fake rows only",
                f"integration metadata preview: {connector_focus} -> {metadata_action}; target selected source; time selected window; data metadata-only; verification fake rows only; tests blocked full-content smoke; audit metadata preview receipt; acceptance gate passed",
                f"integration promotion gate: {connector_focus} -> {metadata_action}; target selected source; time selected window; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt",
                f"integration preflight contract: {connector_focus} -> {metadata_action}; target selected source; time selected window; data metadata-only; verification fake rows only; tests blocked full-content smoke; audit tool run receipt",
                f"integration enablement gate: {connector_focus} -> {metadata_action}; target selected source; time selected window; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed",
                f"integration proof bundle: {connector_focus} -> {metadata_action}; target selected source; time selected window; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed",
                f"integration implementation review: {connector_focus} -> {metadata_action}; target selected source; time selected window; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed; status status api smoke passed",
                "execution health report",
            ]
            acceptance_checks = [
                f"`integration dry run contract: {connector_focus}` proves the metadata row contract before metadata preview or promotion.",
                f"`integration proof bundle: {connector_focus}` passes metadata preview, disabled adapter acceptance, metadata row contract, enablement gate, and rehearsal receipt before implementation review.",
                f"`integration promotion gate: {connector_focus}` carries metadata row contract proof, row limit, blocked payload fields, tests, audit, rollback, and approval boundaries before implementation spec.",
                f"`integration implementation review: {connector_focus}` requires proof bundle, metadata row contract, preflight row-contract proof, implementation spec, status/API smoke evidence, audit, verification, and rollback before code review.",
                f"`integration adapter acceptance: {connector_focus}` passes metadata happy path, row limit, full-content blocked, side-effect blocked, and missing-scope blocked cases.",
                "`integration enablement gate` reports disabled adapter acceptance proof before review is allowed.",
                "Metadata preview returns bounded fake rows only: id, timestamp, label, source.",
                "Natural-language connector auto-routing remains disabled until focused personal and status smoke tests pass.",
                "Any personal-data read or side effect still stops at one-shot approval with approval chain proof, audit, verification, and rollback evidence.",
            ]
            selected_slice_command = f"harness build slice: personal connector readiness; focus {connector_focus}"
            completion_audit_command = "completion audit: improve AGI gate personal integrations"
            completion_claim_gate_command = "completion claim gate: improve AGI gate personal integrations"
        elif any(marker in f"{objective} {focus}".lower() for marker in ("ui", "gui", "dashboard", "interface", "status")):
            target_kind = "command_first_dashboard"
            target_title = "Improve command-first cockpit visibility without adding modes or overlap."
            target_reason = "The dashboard is the harness cockpit: it should show state, routes, diagnostics, approvals, and composer controls without stealing focus from Jarvis."
            likely_files = [
                "jarvis_v2/ui/status_server.py",
                "jarvis_v2/scripts/smoke_test_status_server.py",
            ]
            focused_tests = [
                "python3 -m py_compile jarvis_v2/ui/status_server.py jarvis_v2/scripts/smoke_test_status_server.py",
                "python3 -m jarvis_v2.scripts.smoke_test_status_server",
                "python3 -m compileall -q jarvis_v2",
            ]
            selected_slice_command = "harness build slice: command-first dashboard"
        elif any(marker in f"{objective} {focus}".lower() for marker in ("voice", "speech", "mic", "talk", "spoken")):
            target_kind = "voice_command_harness"
            target_title = "Improve speech/text parity while keeping risky spoken orders approval-gated."
            target_reason = "Voice is a control surface, not a permission bypass; spoken and typed orders need the same route, approval, and stop behavior."
            likely_files = [
                "jarvis_v2/tools/voice.py",
                "jarvis_v2/agent/planner.py",
                "jarvis_v2/ui/status_server.py",
                "jarvis_v2/scripts/smoke_test_voice.py",
                "jarvis_v2/scripts/smoke_test_status_server.py",
            ]
            focused_tests = [
                "python3 -m py_compile jarvis_v2/tools/voice.py jarvis_v2/agent/planner.py jarvis_v2/ui/status_server.py",
                "python3 -m jarvis_v2.scripts.smoke_test_voice",
                "python3 -m jarvis_v2.scripts.smoke_test_status_server",
                "python3 -m compileall -q jarvis_v2",
            ]
            selected_slice_command = "harness build slice: voice command harness"
        elif tasks:
            task_text = _short(tasks[0]["body"], limit=180)
            target_title = f"Turn top task #{tasks[0]['id']} into one verified harness behavior: {task_text}"
            target_reason = "Concrete tasks should become tested harness behavior before they are marked complete."
            selected_slice_command = f"harness build slice: task {tasks[0]['id']}"

        file_integrity = _target_file_integrity(likely_files)
        evidence_closure_commands = [
            selected_slice_command,
            completion_audit_command,
            evidence_ledger_command,
            completion_claim_gate_command,
        ]

        lines = [
            "Jarvis harness build slice:",
            "This is read-only. It selects one implementation slice for autonomous Jarvis build work without approving requests, executing Jarvis tools, editing files, reading private data, controlling the computer, or queuing approvals.",
            "",
            f"Objective: {objective}",
            *_priority_goal_summary(),
            "",
            "Selected slice:",
            f"- kind: {target_kind}",
            f"- title: {target_title}",
            f"- why: {target_reason}",
            "",
            "Owning files to inspect first:",
        ]
        lines.extend(f"- `{path}`" for path in likely_files)
        lines.extend(
            [
                "",
                "Target integrity:",
                f"- file check: {file_integrity['status']}",
                f"- files checked: {file_integrity['checked']}",
                f"- missing files: {', '.join(file_integrity['missing']) if file_integrity['missing'] else 'none'}",
                "",
                "Build loop:",
                "1. Read the owning files and existing smoke tests.",
                "2. Implement one narrowly scoped harness behavior.",
                "3. Add or update the focused smoke test before broader verification.",
                "4. Run the focused tests below, then broaden only if shared routing/status code changed.",
                "5. Run `acceptance gate: <changed behavior>; evidence <receipt>; tests <verification>; recovery <rollback or stop condition>` before treating the behavior as done.",
                "6. Summarize what changed and what still blocks real execution.",
                "",
                "Focused verification:",
            ]
        )
        lines.extend(f"- `{command}`" for command in focused_tests)
        lines.extend(
            [
                "",
                "Starter harness commands:",
            ]
        )
        lines.extend(f"- `{command}`" for command in starter_commands)
        lines.extend(
            [
                "",
                "Acceptance checks:",
            ]
        )
        lines.extend(f"- {check}" for check in acceptance_checks)
        lines.extend(
            [
                "",
                "Completion proof handoff:",
                f"- selected slice packet: `{selected_slice_command}`",
                f"- audit: `{completion_audit_command}`",
                f"- evidence: `{evidence_ledger_command}`",
                f"- claim gate: `{completion_claim_gate_command}`",
                f"- focused verification: {', '.join(f'`{command}`' for command in focused_tests) if focused_tests else 'none configured'}",
            ]
        )
        lines.extend(
            [
                "",
                "Approval boundary:",
                "- Pending approval packets may be reviewed, but this build slice does not approve, dismiss, or rerun them.",
                "- Real personal connectors, shell/code execution, computer control, destructive file changes, private data, and external side effects remain approval-gated.",
                "",
                "Context signals:",
                f"- pending approvals visible: {len(approvals)}",
                f"- open tasks visible: {len(tasks)}",
                f"- active goals visible: {len(goals)}",
                f"- recovery debt visible: {'yes' if recovery_debt_visible else 'no'}",
                f"- execution learning debt visible: {'yes' if execution_learning_debt_visible else 'no'}",
                f"- recovery or learning debt visible: {'yes' if recovery_or_learning_debt_visible else 'no'}",
                "- recovery debt blocks this selector: no, this packet is read-only and does not authorize risky execution",
            ]
        )
        if approvals:
            lines.append(f"- first approval remains blocked: #{approvals[0]['id']} {_short(approvals[0]['tool_name'], limit=80)}")
        if tasks:
            lines.append(f"- top task: #{tasks[0]['id']} {_short(tasks[0]['body'])}")
        if goals:
            lines.append(f"- top goal: #{goals[0]['id']} {_short(goals[0]['title'])}")
        lines.extend(_execution_health_learning_handoff_lines(health))

        return ToolResult(
            "harness_build_slice",
            True,
            "\n".join(lines),
            _safe_metadata(
                target_kind=target_kind,
                likely_files=len(likely_files),
                likely_file_paths=likely_files,
                target_file_integrity_status=file_integrity["status"],
                target_files_checked=file_integrity["checked"],
                target_files_exist=file_integrity["all_exist"],
                missing_target_files=file_integrity["missing"],
                missing_target_file_count=file_integrity["missing_count"],
                target_file_rows=file_integrity["rows"],
                focused_tests=len(focused_tests),
                starter_commands=len(starter_commands),
                starter_command_names=starter_commands,
                acceptance_checks=len(acceptance_checks),
                acceptance_check_names=acceptance_checks,
                selected_slice_command=selected_slice_command,
                completion_audit_command=completion_audit_command,
                evidence_ledger_command=evidence_ledger_command,
                completion_claim_gate_command=completion_claim_gate_command,
                evidence_closure_commands=evidence_closure_commands,
                evidence_closure_command_count=len(evidence_closure_commands),
                focused_verification_commands=focused_tests,
                focused_verification_command_count=len(focused_tests),
                recovery_debt_visible=recovery_debt_visible,
                execution_learning_debt_visible=execution_learning_debt_visible,
                recovery_or_learning_debt_visible=recovery_or_learning_debt_visible,
                read_only_slice_allowed_with_recovery_debt=read_only_slice_allowed_with_recovery_debt,
                recovery_debt_blocks_current_slice=recovery_debt_blocks_current_slice,
                failed_action_runs=health["failed_action_runs"],
                approval_held_action_runs=health["approval_held_action_runs"],
                execution_health_review_required=health["review_required"],
                execution_health_next_command=health["next_command"],
                execution_health_next_commands=health["next_commands"],
                execution_health_next_command_count=len(health["next_commands"]),
                execution_health_blocker_categories=health["blocker_categories"],
                execution_health_blocker_count=health["blocker_count"],
                execution_health_verification_coverage=health["verification_coverage_state"],
                execution_health_repeated_failure_count=health["repeated_failure_count"],
                execution_health_approval_proof_chains=health["approval_proof_chains"],
                execution_health_approval_proof_chain_count=health["approval_proof_chain_count"],
                **_execution_health_learning_handoff_metadata(health),
                pending_approvals=len(approvals),
                open_tasks=len(tasks),
                active_goals=len(goals),
            ),
        )

    def save_build_target_packet(args: dict[str, Any]) -> ToolResult:
        packet = build_target_packet(args)
        objective = _short(args.get("objective") or "continue Jarvis V2 safely", limit=MAX_OBJECTIVE_CHARS)
        path = vault.root_path / "Automations" / "Build Target Packet.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        receipt_lines = [
            "# Build Target Packet",
            "",
            "## Receipt",
            "",
            f"- Generated: {generated_at}",
            f"- Objective: {objective}",
            f"- Target kind: {packet.metadata.get('target_kind', 'unknown')}",
            f"- Pending approvals seen: {packet.metadata.get('pending_approvals', 0)}",
            f"- Open tasks seen: {packet.metadata.get('open_tasks', 0)}",
            f"- Active goals seen: {packet.metadata.get('active_goals', 0)}",
            f"- Enabled scheduled jobs seen: {packet.metadata.get('enabled_jobs', 0)}",
            "- Save action: local Obsidian note write only.",
            "- Safety: this save does not call a model, execute tools, queue approvals, approve requests, edit code, or control the computer.",
            "",
            "## Resume Commands",
            "",
            "- Review: `build target packet: continue Jarvis V2 safely`",
            "- Save again: `save build target packet: continue Jarvis V2 safely`",
            "- Start safely: `continuation packet: continue Jarvis V2 safely`",
            "- Check blockers: `pending approvals`, `approval readiness #ID`, `approval packet #ID`, then `approval chain proof #ID` before any risky work.",
            "",
            "## Packet",
            "",
            packet.output,
            "",
        ]
        path.write_text("\n".join(receipt_lines), encoding="utf-8")
        metadata = dict(packet.metadata)
        path_display = _safe_vault_path_display(path, vault)
        metadata.update(
            {
                "path": str(path),
                "path_display": path_display,
                "writes_notes": True,
                "writes_files": True,
                "edits_files": False,
                "executes_tools": False,
                "queues_approval": False,
                "controls_computer": False,
            }
        )
        return ToolResult(
            "save_build_target_packet",
            True,
            f"Build target packet saved: {path_display}\n\n{packet.output}",
            metadata,
        )

    def export_mission_control(args: dict[str, Any]) -> ToolResult:
        body, metadata = build_safe_next_actions(_bounded_limit(args.get("limit", 5), 5))
        if effect_authority is not None:
            effect_authority()
        path = vault.write_mission_control(body)
        path_display = _safe_vault_path_display(path, vault)
        metadata["path"] = str(path)
        metadata["path_display"] = path_display
        metadata["writes_files"] = True
        metadata["writes_notes"] = True
        return ToolResult(
            "export_mission_control",
            True,
            f"Mission Control updated: {path_display}\n\n{body}",
            metadata,
        )

    def work_queue(args: dict[str, Any]) -> ToolResult:
        body, metadata = build_work_queue(_bounded_limit(args.get("limit", 8), 8))
        return ToolResult("work_queue", True, body, metadata)

    def save_work_queue(args: dict[str, Any]) -> ToolResult:
        body, metadata = build_work_queue(_bounded_limit(args.get("limit", 8), 8))
        path = vault.root_path / "Automations" / "Work Queue.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Work Queue\n\n{body}\n", encoding="utf-8")
        path_display = _safe_vault_path_display(path, vault)
        metadata["path"] = str(path)
        metadata["path_display"] = path_display
        metadata["writes_files"] = True
        metadata["writes_notes"] = True
        return ToolResult(
            "save_work_queue",
            True,
            f"Work queue saved: {path_display}\n\n{body}",
            metadata,
        )

    return safe_next_actions, next_action_packet, priority_stack, continuation_packet, build_target_packet, harness_build_slice, save_build_target_packet, export_mission_control, work_queue, save_work_queue
