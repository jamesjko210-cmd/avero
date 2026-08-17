from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jarvis_v2.agent.types import ToolResult
from jarvis_v2.automations.compaction import COMPACTION_JOB_NAME, COMPACTION_JOB_TYPE
from jarvis_v2.memory.store import MemoryStore
from jarvis_v2.tools.audit_meta import EXECUTION_META_TOOLS
from jarvis_v2.tools.harness import AGI_GATE_BUILD_TARGETS, _target_file_integrity


MAX_FOCUS_LIMIT = 50
MAX_OBJECTIVE_CHARS = 600
AUDIT_META_TOOLS = {*EXECUTION_META_TOOLS, "execution_learning_closure_packet"}
WORK_SESSION_BOUNDARY_FALSE_FLAGS = {
    "calls_model": False,
    "executes_tools": False,
    "queues_approval": False,
    "requires_approval": False,
    "writes_files": False,
    "writes_database": False,
    "writes_memory": False,
    "writes_notes": False,
    "reads_private_data": False,
    "reads_personal_data": False,
    "external_side_effect": False,
    "controls_computer": False,
    "speaks": False,
    "completes_tasks": False,
    "authorizes_execution": False,
    "authorizes_completion_claim": False,
    "approval_granted": False,
}
AGI_FOCUS_SELECTED_GATE = "personal integrations"
AGI_FOCUS_SELECTION_SOURCE = "operator_focus_handoff"
AGI_FOCUS_SELECTION_REASON = (
    "Return, handoff, focus, and session packets keep the current operator-facing build target on the "
    "personal integrations proof lane until that handoff is deliberately superseded; `agi gates` remains "
    "the canonical registry-gap selector for the global next AGI gate."
)
AGI_FOCUS_CANONICAL_SELECTOR_COMMAND = "agi gates"
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)


def _bounded_int(value: Any, default: int, low: int = 1, high: int = MAX_FOCUS_LIMIT) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        number = default
    return max(low, min(high, number))


def _metadata_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value).strip()
    except Exception:
        return ""


def _bounded_text(value: Any, limit: int = MAX_OBJECTIVE_CHARS) -> str:
    text = LOCAL_PATH_RE.sub("<local-path>", _coerce_text(value))
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _safe_focus_text(value: Any, limit: int = 500) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", _bounded_text(value, limit=limit))


def _coerce_list(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        return [values]
    try:
        return list(values)
    except Exception:
        return []


def _safe_focus_list(values: Any, limit: int = 500) -> list[str]:
    safe_values: list[str] = []
    for value in _coerce_list(values):
        safe_value = _safe_focus_text(value, limit=limit)
        if safe_value:
            safe_values.append(safe_value)
    return safe_values


def _metadata_count(values: Any) -> int:
    return len(_coerce_list(values))


def _metadata_list(values: Any, limit: int = 500) -> list[str]:
    return _safe_focus_list(values, limit=limit)


def _metadata_text(value: Any, limit: int = 500) -> str:
    return _safe_focus_text(value, limit=limit)


def _agi_focus_target_file_integrity(raw_paths: list[Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for raw_path in raw_paths:
        path = _coerce_text(raw_path)
        if not path:
            continue
        safe_path = _safe_focus_text(path, limit=260)
        path_obj = Path(path)
        if path_obj.is_absolute() or path.startswith("~"):
            exists = False
        else:
            exists = bool(_target_file_integrity([path]).get("all_exist"))
        rows.append({"path": safe_path, "exists": exists})
        if not exists:
            missing.append(safe_path)
    return {
        "rows": rows,
        "checked": len(rows),
        "missing": missing,
        "missing_count": len(missing),
        "all_exist": not missing,
        "status": "TARGETS_EXIST" if not missing else "TARGETS_STALE",
    }


def _read_only_metadata(**extra: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "calls_model": False,
        "executes_tools": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "external_side_effect": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": False,
        "speaks": False,
        "completes_tasks": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
    }
    metadata.update(extra)
    return metadata


def _background_rhythm_state(jobs: list[Any]) -> dict[str, Any]:
    readable_jobs, unreadable_jobs = _readable_scheduled_job_rows(jobs)
    enabled_jobs = [job for job in readable_jobs if _row_bool(job, "enabled")]
    state_snapshot_jobs = [job for job in readable_jobs if _row_text(job, "job_type") == "state_snapshot"]
    enabled_state_snapshot_jobs = [job for job in state_snapshot_jobs if _row_bool(job, "enabled")]
    disabled_state_snapshot_jobs = [job for job in state_snapshot_jobs if not _row_bool(job, "enabled")]
    conversation_compaction_jobs = [
        job
        for job in readable_jobs
        if _row_text(job, "job_type") == COMPACTION_JOB_TYPE
        or _row_text(job, "name").casefold() == COMPACTION_JOB_NAME.casefold()
    ]
    enabled_conversation_compaction_jobs = [job for job in conversation_compaction_jobs if _row_bool(job, "enabled")]
    disabled_conversation_compaction_jobs = [job for job in conversation_compaction_jobs if not _row_bool(job, "enabled")]
    if unreadable_jobs:
        ready = False
        issue = f"Scheduled job table has {unreadable_jobs} unreadable row(s)."
        next_command = "list scheduled jobs"
        priority = "review scheduled jobs before trusting background rhythm."
    elif not enabled_state_snapshot_jobs and state_snapshot_jobs:
        ready = False
        issue = "State Snapshot scheduled job is paused."
        next_command = "resume job State Snapshot"
        priority = "resume job State Snapshot for context refresh."
    elif not enabled_state_snapshot_jobs:
        ready = False
        issue = "State Snapshot scheduled job is not configured."
        next_command = "schedule assistant basics"
        priority = "schedule assistant basics for context refresh."
    elif not enabled_conversation_compaction_jobs and conversation_compaction_jobs:
        ready = False
        issue = f"{COMPACTION_JOB_NAME} scheduled job is paused."
        next_command = f"resume job {COMPACTION_JOB_NAME}"
        priority = f"resume job {COMPACTION_JOB_NAME} for durable memory compaction."
    elif not enabled_conversation_compaction_jobs:
        ready = False
        issue = f"{COMPACTION_JOB_NAME} scheduled job is not configured."
        next_command = "schedule assistant basics"
        priority = "schedule assistant basics for durable memory compaction."
    else:
        ready = True
        issue = ""
        next_command = "list scheduled jobs"
        priority = ""
    return {
        "ready": ready,
        "issue": issue,
        "next_command": next_command,
        "priority": priority,
        "enabled_jobs": enabled_jobs,
        "state_snapshot_jobs": state_snapshot_jobs,
        "enabled_state_snapshot_jobs": enabled_state_snapshot_jobs,
        "disabled_state_snapshot_jobs": disabled_state_snapshot_jobs,
        "conversation_compaction_jobs": conversation_compaction_jobs,
        "enabled_conversation_compaction_jobs": enabled_conversation_compaction_jobs,
        "disabled_conversation_compaction_jobs": disabled_conversation_compaction_jobs,
        "unreadable_scheduled_job_rows": unreadable_jobs,
    }


def _background_rhythm_line(background: dict[str, Any]) -> str:
    if background["ready"]:
        return f"Background rhythm: State Snapshot and {COMPACTION_JOB_NAME} are enabled."
    return f"Background rhythm: {background['issue']} Next safe command: `{background['next_command']}`."


def _background_rhythm_metadata(background: dict[str, Any]) -> dict[str, Any]:
    return {
        "background_ready": bool(background["ready"]),
        "background_next_command": str(background["next_command"]),
        "background_issue": str(background["issue"]),
        "background_priority": str(background["priority"]),
        "state_snapshot_jobs": len(background["state_snapshot_jobs"]),
        "enabled_state_snapshot_jobs": len(background["enabled_state_snapshot_jobs"]),
        "disabled_state_snapshot_jobs": len(background["disabled_state_snapshot_jobs"]),
        "conversation_compaction_jobs": len(background["conversation_compaction_jobs"]),
        "enabled_conversation_compaction_jobs": len(background["enabled_conversation_compaction_jobs"]),
        "disabled_conversation_compaction_jobs": len(background["disabled_conversation_compaction_jobs"]),
        "unreadable_scheduled_job_rows": int(background["unreadable_scheduled_job_rows"]),
    }


def _work_session_packet_handoff(
    *,
    objective: str,
    start_move: str,
    stop_condition: str,
    limit: int,
    approvals: list[Any],
    tasks: list[Any],
    goals: list[Any],
    decisions: list[Any],
    preferences: list[Any],
    enabled_jobs: list[Any],
    jobs: list[Any],
    approval_handoff: dict[str, Any],
    health: dict[str, Any],
    agi_focus: dict[str, Any],
    background: dict[str, Any],
) -> dict[str, Any]:
    next_commands: list[str] = [
        "chat continuity brief",
        "next action packet",
        "action rehearsal: <request>",
    ]
    if approval_handoff.get("approval_handoff_readiness_command"):
        next_commands.extend(
            [
                str(approval_handoff["approval_handoff_readiness_command"]),
                str(approval_handoff["approval_handoff_last_look_command"]),
                str(approval_handoff["approval_handoff_proof_command"]),
            ]
        )
    if health.get("next_command"):
        next_commands.append(str(health["next_command"]))
    if not background["ready"]:
        next_commands.append(str(background["next_command"]))
    next_commands.extend(["build delta", "session closeout", "save handoff brief"])
    deduped_next_commands = list(dict.fromkeys(command for command in next_commands if command))
    return {
        "source": "work_session_packet",
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "objective": objective or "start the smallest useful safe work chunk",
        "start_move": _bounded_text(start_move),
        "stop_condition": _bounded_text(stop_condition),
        "preflight_commands": [
            "chat continuity brief",
            "return brief",
            "next action packet",
            "action rehearsal: <request>",
            "approval readiness #ID",
            "approval packet #ID",
            "approval chain proof #ID",
        ],
        "end_verification_commands": [
            "build delta",
            _metadata_text(health.get("next_command")) or "execution health report",
            "session closeout",
            "save handoff brief",
        ],
        "next_commands": deduped_next_commands,
        "next_safe_commands": deduped_next_commands,
        "next_command_count": len(deduped_next_commands),
        "next_safe_command_count": len(deduped_next_commands),
        "pending_approvals": len(approvals),
        "open_tasks": len(tasks),
        "active_goals": len(goals),
        "active_decisions": len(decisions),
        "active_preferences": len(preferences),
        "enabled_jobs": len(enabled_jobs),
        "scheduled_jobs": len(jobs),
        **_background_rhythm_metadata(background),
        "limit": limit,
        "approval_handoff": approval_handoff,
        "approval_handoff_pending_count": approval_handoff.get("approval_handoff_pending_count", 0),
        "approval_handoff_proof_chain_commands": approval_handoff.get("approval_handoff_proof_chain_commands", []),
        "execution_health_review_required": _metadata_bool(health.get("review_required")),
        "execution_health_next_command": _metadata_text(health.get("next_command")),
        "execution_health_next_commands": _metadata_list(health.get("next_commands")),
        "execution_health_next_command_count": _metadata_count(health.get("next_commands")),
        "execution_health_approval_held_review_required": _metadata_bool(health.get("approval_held_review_required")),
        "execution_health_approval_held_review_commands": _metadata_list(health.get("approval_held_review_commands")),
        "execution_health_approval_held_review_command_count": _metadata_count(health.get("approval_held_review_commands")),
        "execution_health_approval_held_review_next_command": _metadata_text(health.get("approval_held_review_next_command")),
        "execution_health_blocker_categories": _metadata_list(health.get("blocker_categories"), limit=160),
        "execution_health_blocker_count": _metadata_int(health.get("blocker_count")) or 0,
        "execution_health_verification_coverage": _metadata_text(health.get("verification_coverage_state"), limit=80),
        "agi_next_build_command": _metadata_text(agi_focus.get("build_command")),
        "agi_next_gate": _metadata_text(agi_focus.get("selected_gate"), limit=160),
        "agi_next_target_title": _metadata_text(agi_focus.get("target_title")),
        "agi_next_build_packet_ready_for_review": _metadata_bool(agi_focus.get("build_packet_ready_for_review")),
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
        "boundaries": {
            "read_only": True,
            "operator_timeboxes_override_priority": True,
            "stop_times_override_priority": True,
            **WORK_SESSION_BOUNDARY_FALSE_FLAGS,
        },
    }


def _next_session_plan_handoff(
    *,
    objective: str,
    first_safe_move: str,
    limit: int,
    approvals: list[Any],
    tasks: list[Any],
    goals: list[Any],
    decisions: list[Any],
    preferences: list[Any],
    enabled_jobs: list[Any],
    jobs: list[Any],
    approval_handoff: dict[str, Any],
    health: dict[str, Any],
    agi_focus: dict[str, Any],
    background: dict[str, Any],
) -> dict[str, Any]:
    next_commands: list[str] = [
        "return brief",
        "handoff brief",
        "pending approvals",
    ]
    if approval_handoff.get("approval_handoff_readiness_command"):
        next_commands.extend(
            [
                str(approval_handoff["approval_handoff_readiness_command"]),
                str(approval_handoff["approval_handoff_last_look_command"]),
                str(approval_handoff["approval_handoff_proof_command"]),
            ]
        )
    if health.get("next_command"):
        next_commands.append(str(health["next_command"]))
    if not background["ready"]:
        next_commands.append(str(background["next_command"]))
    if agi_focus.get("build_command"):
        next_commands.append(str(agi_focus["build_command"]))
    next_commands.extend(["session closeout", "safe next actions"])
    deduped_next_commands = list(dict.fromkeys(command for command in next_commands if command))
    return {
        "source": "next_session_plan",
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "objective": objective or "resume safely with the smallest useful next move",
        "first_safe_move": _bounded_text(first_safe_move),
        "resume_order": [
            "return brief or handoff brief",
            "pending approvals",
            "approval readiness #ID",
            "approval packet #ID",
            "approval chain proof #ID",
            "one visible task or goal step",
            "session closeout or safe next actions",
        ],
        "next_commands": deduped_next_commands,
        "next_safe_commands": deduped_next_commands,
        "next_command_count": len(deduped_next_commands),
        "next_safe_command_count": len(deduped_next_commands),
        "pending_approvals": len(approvals),
        "open_tasks": len(tasks),
        "active_goals": len(goals),
        "active_decisions": len(decisions),
        "active_preferences": len(preferences),
        "enabled_jobs": len(enabled_jobs),
        "scheduled_jobs": len(jobs),
        **_background_rhythm_metadata(background),
        "limit": limit,
        "approval_handoff": approval_handoff,
        "approval_handoff_pending_count": approval_handoff.get("approval_handoff_pending_count", 0),
        "approval_handoff_proof_chain_commands": approval_handoff.get("approval_handoff_proof_chain_commands", []),
        "execution_health_review_required": _metadata_bool(health.get("review_required")),
        "execution_health_next_command": _metadata_text(health.get("next_command")),
        "execution_health_next_commands": _metadata_list(health.get("next_commands")),
        "execution_health_next_command_count": _metadata_count(health.get("next_commands")),
        "execution_health_approval_held_review_required": _metadata_bool(health.get("approval_held_review_required")),
        "execution_health_approval_held_review_commands": _metadata_list(health.get("approval_held_review_commands")),
        "execution_health_approval_held_review_command_count": _metadata_count(health.get("approval_held_review_commands")),
        "execution_health_approval_held_review_next_command": _metadata_text(health.get("approval_held_review_next_command")),
        "execution_health_blocker_categories": _metadata_list(health.get("blocker_categories"), limit=160),
        "execution_health_blocker_count": _metadata_int(health.get("blocker_count")) or 0,
        "execution_health_verification_coverage": _metadata_text(health.get("verification_coverage_state"), limit=80),
        "agi_next_build_command": _metadata_text(agi_focus.get("build_command")),
        "agi_next_gate": _metadata_text(agi_focus.get("selected_gate"), limit=160),
        "agi_next_target_title": _metadata_text(agi_focus.get("target_title")),
        "agi_next_build_packet_ready_for_review": _metadata_bool(agi_focus.get("build_packet_ready_for_review")),
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
        "boundaries": {
            "read_only": True,
            "operator_timeboxes_override_priority": True,
            "stop_times_override_priority": True,
            **WORK_SESSION_BOUNDARY_FALSE_FLAGS,
        },
    }


def _row_metadata(row: Any) -> dict[str, Any]:
    try:
        metadata = _row_value(row, "metadata")
        if metadata is _MISSING_ROW_VALUE:
            return {}
        if isinstance(metadata, dict):
            return dict(metadata)
        if metadata is None:
            return {}
        parsed = json.loads(metadata)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


_MISSING_ROW_VALUE = object()


def _row_value(row: Any, key: str, default: Any = _MISSING_ROW_VALUE) -> Any:
    try:
        if isinstance(row, dict):
            return row.get(key, default)
        keys = row.keys()
        if key not in keys:
            return default
        return row[key]
    except Exception:
        return default


def _row_text(row: Any, key: str, default: str = "") -> str:
    value = _row_value(row, key)
    if value is _MISSING_ROW_VALUE or value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


def _row_bool(row: Any, key: str, default: bool = False) -> bool:
    value = _row_value(row, key)
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    return default


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
    try:
        text = "" if value is None else str(value)
    except Exception:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", text.strip().casefold()).strip("_")


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
    if _row_bool(row, "ok", default=True):
        return False
    return _is_approval_hold_metadata(_row_metadata(row))


def _recent_tool_run_attention_buckets(rows: list[Any]) -> tuple[list[Any], list[Any]]:
    failed: list[Any] = []
    approval_held: list[Any] = []
    for row in rows:
        if _row_bool(row, "ok", default=True):
            continue
        if _is_approval_held_tool_run(row):
            approval_held.append(row)
        else:
            failed.append(row)
    return failed, approval_held


def _row_int(row: Any, key: str, default: int | None = None) -> int | None:
    value = _row_value(row, key)
    if isinstance(value, bool) or value is _MISSING_ROW_VALUE or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _row_positive_int_text(row: Any, key: str = "id") -> str:
    value = _row_int(row, key)
    return str(value) if value is not None and value > 0 else ""


def _first_row_with_positive_id(rows: list[Any], key: str = "id") -> tuple[str, Any | None]:
    for row in rows:
        row_id = _row_positive_int_text(row, key)
        if row_id:
            return row_id, row
    return "", None


def _row_text_by_keys(row: Any, keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        text = _row_text(row, key)
        if text:
            return text
    return default


def _readable_tool_run_rows(rows: list[Any]) -> tuple[list[Any], int]:
    readable: list[Any] = []
    unreadable = 0
    for row in rows:
        if _row_text(row, "tool_name"):
            readable.append(row)
        else:
            unreadable += 1
    return readable, unreadable


def _readable_scheduled_job_rows(rows: list[Any]) -> tuple[list[Any], int]:
    readable: list[Any] = []
    unreadable = 0
    for row in rows:
        if _row_text(row, "name") or _row_text(row, "job_type"):
            readable.append(row)
        else:
            unreadable += 1
    return readable, unreadable


def _approval_proof_chain_commands(approval_id: int) -> list[str]:
    return [
        f"approval readiness {approval_id}",
        f"approval packet {approval_id}",
        f"approval chain proof {approval_id}",
        f"verification receipt <approved run id from approval chain proof {approval_id}>",
    ]


def _approval_handoff(approvals: list[Any]) -> dict[str, Any]:
    approval_id_text, approval = _first_row_with_positive_id(approvals)
    if approval is None:
        return {
            "approval_handoff_pending_count": len(approvals),
            "approval_handoff_first_id": None,
            "approval_handoff_first_tool": "",
            "approval_handoff_readiness_command": "",
            "approval_handoff_last_look_command": "",
            "approval_handoff_proof_command": "",
            "approval_handoff_verification_command": "",
            "approval_handoff_approve_command": "",
            "approval_handoff_dismiss_command": "",
            "approval_handoff_proof_chain_commands": [],
        }
    approval_id = int(approval_id_text)
    chain = _approval_proof_chain_commands(approval_id)
    return {
        "approval_handoff_pending_count": len(approvals),
        "approval_handoff_first_id": approval_id,
        "approval_handoff_first_tool": _row_text(approval, "tool_name", "unknown"),
        "approval_handoff_readiness_command": chain[0],
        "approval_handoff_last_look_command": chain[1],
        "approval_handoff_proof_command": chain[2],
        "approval_handoff_verification_command": chain[3],
        "approval_handoff_approve_command": f"approve approval {approval_id}",
        "approval_handoff_dismiss_command": f"dismiss approval {approval_id}",
        "approval_handoff_proof_chain_commands": chain,
    }


def _approval_handoff_lines(handoff: dict[str, Any], *, prefix: str = "- ") -> list[str]:
    if not handoff.get("approval_handoff_first_id"):
        return []
    return [
        f"{prefix}first safe approval handoff: `{handoff['approval_handoff_readiness_command']}` -> `{handoff['approval_handoff_last_look_command']}` -> `{handoff['approval_handoff_proof_command']}`",
        f"{prefix}decision commands: `{handoff['approval_handoff_approve_command']}` or `{handoff['approval_handoff_dismiss_command']}`",
    ]


def _append_unique(commands: list[str], command: str) -> None:
    if command not in commands:
        commands.append(command)


def _focus_brief_handoff(
    *,
    objective: str,
    limit: int,
    approvals: list[Any],
    tasks: list[Any],
    goals: list[Any],
    decisions: list[Any],
    preferences: list[Any],
    enabled_jobs: list[Any],
    jobs: list[Any],
    approval_handoff: dict[str, Any],
    health: dict[str, Any],
    agi_focus: dict[str, Any],
    background: dict[str, Any],
) -> dict[str, Any]:
    start_kind = "capture_task"
    start_label = "Capture one task or create one goal before reactive work."
    start_command = "safe next actions"
    approval_id_text, approval = _first_row_with_positive_id(approvals)
    task_id_text, task = _first_row_with_positive_id(tasks)
    goal_id_text, goal = _first_row_with_positive_id(goals)
    if approval is not None:
        start_kind = "approval_review"
        start_label = f"Review approval #{approval_id_text} for {_row_text(approval, 'tool_name', 'unknown')}."
        start_command = _metadata_text(
            approval_handoff.get("approval_handoff_readiness_command")
        ) or f"approval readiness {approval_id_text}"
    elif task is not None:
        start_kind = "task"
        start_label = f"Work one visible step on task #{task_id_text}."
        start_command = f"task detail {task_id_text}"
    elif goal is not None:
        start_kind = "goal"
        start_label = f"Work one visible step on goal #{goal_id_text}."
        start_command = f"goal detail {goal_id_text}"
    elif not background["ready"]:
        start_kind = "background_repair"
        start_label = str(background["priority"])
        start_command = str(background["next_command"])

    next_commands: list[str] = [
        "return brief",
        "mission control",
        start_command,
    ]
    if approval_handoff.get("approval_handoff_readiness_command"):
        next_commands.extend(
            [
                str(approval_handoff["approval_handoff_readiness_command"]),
                str(approval_handoff["approval_handoff_last_look_command"]),
                str(approval_handoff["approval_handoff_proof_command"]),
            ]
        )
    if health.get("next_command"):
        next_commands.append(str(health["next_command"]))
    if not background["ready"]:
        next_commands.append(str(background["next_command"]))
    if agi_focus.get("build_command"):
        next_commands.append(str(agi_focus["build_command"]))
    next_commands.append("safe next actions")
    deduped_next_commands = list(dict.fromkeys(command for command in next_commands if command))

    return {
        "source": "focus_brief",
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "objective": objective or "choose the smallest useful next step",
        "start_kind": start_kind,
        "start_label": _bounded_text(start_label),
        "start_command": start_command,
        "session_plan": [
            "return brief or mission control",
            "approval readiness #ID",
            "approval packet #ID",
            "approval chain proof #ID",
            "one visible task or goal step",
            "safe next actions",
        ],
        "next_commands": deduped_next_commands,
        "next_safe_commands": deduped_next_commands,
        "next_command_count": len(deduped_next_commands),
        "next_safe_command_count": len(deduped_next_commands),
        "pending_approvals": len(approvals),
        "open_tasks": len(tasks),
        "active_goals": len(goals),
        "active_decisions": len(decisions),
        "active_preferences": len(preferences),
        "enabled_jobs": len(enabled_jobs),
        "scheduled_jobs": len(jobs),
        **_background_rhythm_metadata(background),
        "limit": limit,
        "approval_handoff": approval_handoff,
        "approval_handoff_pending_count": approval_handoff.get("approval_handoff_pending_count", 0),
        "approval_handoff_proof_chain_commands": approval_handoff.get("approval_handoff_proof_chain_commands", []),
        "execution_health_review_required": _metadata_bool(health.get("review_required")),
        "execution_health_next_command": _metadata_text(health.get("next_command")),
        "execution_health_next_commands": _metadata_list(health.get("next_commands")),
        "execution_health_next_command_count": _metadata_count(health.get("next_commands")),
        "execution_health_approval_held_review_required": _metadata_bool(health.get("approval_held_review_required")),
        "execution_health_approval_held_review_commands": _metadata_list(health.get("approval_held_review_commands")),
        "execution_health_approval_held_review_command_count": _metadata_count(health.get("approval_held_review_commands")),
        "execution_health_approval_held_review_next_command": _metadata_text(health.get("approval_held_review_next_command")),
        "execution_health_blocker_categories": _metadata_list(health.get("blocker_categories"), limit=160),
        "execution_health_blocker_count": _metadata_int(health.get("blocker_count")) or 0,
        "execution_health_verification_coverage": _metadata_text(health.get("verification_coverage_state"), limit=80),
        "agi_next_build_command": _metadata_text(agi_focus.get("build_command")),
        "agi_next_gate": _metadata_text(agi_focus.get("selected_gate"), limit=160),
        "agi_next_target_title": _metadata_text(agi_focus.get("target_title")),
        "agi_next_build_packet_ready_for_review": _metadata_bool(agi_focus.get("build_packet_ready_for_review")),
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
        "boundaries": {
            "read_only": True,
            "operator_timeboxes_override_priority": True,
            "stop_times_override_priority": True,
            **WORK_SESSION_BOUNDARY_FALSE_FLAGS,
        },
    }


def _recovery_closure_checklist_command(health: dict[str, Any]) -> str:
    if health.get("recovery_closure_blocks_auto_execution"):
        return "recovery closure checklist"
    return ""


def _execution_health_snapshot(store: MemoryStore, limit: int = 40) -> dict[str, Any]:
    rows = store.recent_tool_runs(limit=limit)
    readable_rows, unreadable_rows = _readable_tool_run_rows(rows)
    action_runs = [row for row in readable_rows if _row_text(row, "tool_name") not in AUDIT_META_TOOLS]
    failed_action_runs, approval_held_action_runs = _recent_tool_run_attention_buckets(action_runs)
    approval_held_action_run_ids = {
        row_id for row_id in (_row_int(row, "id") for row in approval_held_action_runs) if row_id is not None
    }
    learning_context_action_runs = [
        row for row in action_runs if _row_int(row, "id") not in approval_held_action_run_ids
    ]
    risky_levels = {"HIGH_RISK", "PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT"}
    risky_approval_problems = [
        row
        for row in action_runs
        if _row_text(row, "risk") in risky_levels
        and _row_bool(row, "ok")
        and (not _row_bool(row, "approved") or _row_value(row, "approval_id") is _MISSING_ROW_VALUE or _row_value(row, "approval_id") is None)
    ]
    verification_runs = [
        row
        for row in readable_rows
        if _row_text(row, "tool_name") in {
            "verification_receipt",
            "runtime_trace_receipt",
            "execution_audit_gate",
            "execution_recovery_packet",
            "after_action_learning_packet",
            "execution_health_report",
        }
    ]
    recovery_runs = [row for row in readable_rows if _row_text(row, "tool_name") == "execution_recovery_packet"]
    learning_runs = [row for row in readable_rows if _row_text(row, "tool_name") == "after_action_learning_packet"]
    failure_counts: dict[str, int] = {}
    for row in failed_action_runs:
        tool_name = _row_text(row, "tool_name", "unknown")
        failure_counts[tool_name] = failure_counts.get(tool_name, 0) + 1
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
    if unreadable_rows:
        blocker_categories.append("unreadable_audit_rows")
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
        first_problem_id = _row_positive_int_text(first_problem)
        next_command = f"execution recovery packet {first_problem_id}" if first_problem_id else "execution health report"
    elif rows:
        next_command = "execution health report"
    else:
        next_command = "readiness report"

    approval_proof_chains: dict[str, list[str]] = {}
    next_commands: list[str] = []
    for row in failed_action_runs[:3] + approval_held_action_runs[:3] + risky_approval_problems[:3]:
        row_id = _row_positive_int_text(row)
        if row in approval_held_action_runs:
            pass
        elif row_id:
            _append_unique(next_commands, f"execution recovery packet {row_id}")
        else:
            _append_unique(next_commands, "execution health report")
        approval_candidates: list[int] = []
        approval_id = _metadata_int(_row_value(row, "approval_id", None))
        if approval_id is not None:
            approval_candidates.append(approval_id)
        metadata = _row_metadata(row)
        for key in ("approval_id", "approved_approval_id"):
            metadata_id = _metadata_int(metadata.get(key))
            if metadata_id is not None and metadata_id not in approval_candidates:
                approval_candidates.append(metadata_id)
        for candidate in approval_candidates:
            chain = _approval_proof_chain_commands(candidate)
            approval_proof_chains.setdefault(str(candidate), chain)
            for proof_command in chain:
                _append_unique(next_commands, proof_command)
    approval_held_review_commands: list[str] = []
    for row in approval_held_action_runs[:3]:
        approval_candidates: list[int] = []
        approval_id = _metadata_int(_row_value(row, "approval_id", None))
        if approval_id is not None:
            approval_candidates.append(approval_id)
        metadata = _row_metadata(row)
        for key in ("approval_id", "approved_approval_id"):
            metadata_id = _metadata_int(metadata.get(key))
            if metadata_id is not None and metadata_id not in approval_candidates:
                approval_candidates.append(metadata_id)
        for candidate in approval_candidates:
            for proof_command in approval_proof_chains.get(str(candidate), _approval_proof_chain_commands(candidate)):
                _append_unique(approval_held_review_commands, proof_command)

    learning_target = first_problem if first_problem is not None else (
        learning_context_action_runs[0] if learning_context_action_runs else None
    )
    learning_target_run_id = _row_int(learning_target, "id") if learning_target is not None else None
    learning_target_tool = _row_text(learning_target, "tool_name") if learning_target is not None else ""
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
        if _row_text(row, "tool_name") == "verification_receipt" and _row_metadata(row).get("run_id") == learning_target_run_id
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

    for item in repeated_failure_tools[:3]:
        _append_unique(next_commands, f"failure to test preview: repeated {item['tool']} failures")
    for command in failure_promotion_queue:
        _append_unique(next_commands, command)
    for command in [verification_handoff_command, recovery_handoff_command, learning_closure_command, learning_handoff_command]:
        if command:
            _append_unique(next_commands, command)
    for command in recovery_closure_required_commands:
        if command:
            _append_unique(next_commands, command)
    if unreadable_rows:
        _append_unique(next_commands, "execution health report")
    if not verification_runs and readable_rows:
        _append_unique(next_commands, "verification receipt latest")
    _append_unique(next_commands, "execution health report")
    if recovery_closure_blocks_auto_execution and recovery_closure_required_commands:
        ordered_next_commands: list[str] = []
        for command in recovery_closure_required_commands:
            if command:
                _append_unique(ordered_next_commands, command)
        for command in next_commands:
            _append_unique(ordered_next_commands, command)
        next_commands = ordered_next_commands
        next_command = recovery_closure_required_commands[0]
    elif first_problem is None and not repeated_failure_tools and unreadable_rows == 0 and approval_held_review_commands:
        next_command = approval_held_review_commands[0]
    if next_command:
        ordered_next_commands = [next_command]
        for command in next_commands:
            _append_unique(ordered_next_commands, command)
        next_commands = ordered_next_commands

    return {
        "review_required": first_problem is not None or unreadable_rows > 0 or bool(approval_held_action_runs),
        "next_command": next_command,
        "next_commands": next_commands,
        "next_command_count": len(next_commands),
        "recent_tool_run_rows": len(rows),
        "readable_recent_tool_runs": len(readable_rows),
        "unreadable_recent_tool_run_rows": unreadable_rows,
        "failed_action_runs": len(failed_action_runs),
        "approval_held_action_runs": len(approval_held_action_runs),
        "approval_held_review_required": bool(approval_held_action_runs),
        "approval_held_review_commands": approval_held_review_commands,
        "approval_held_review_command_count": len(approval_held_review_commands),
        "approval_held_review_next_command": approval_held_review_commands[0] if approval_held_review_commands else "",
        "blocker_categories": blocker_categories,
        "blocker_count": len(blocker_categories),
        "verification_coverage_state": "present" if verification_runs else "missing",
        "approval_proof_chains": approval_proof_chains,
        "approval_proof_chain_count": len(approval_proof_chains),
        "repeated_failure_tools": repeated_failure_tools,
        "repeated_failure_count": len(repeated_failure_tools),
        "failure_promotion_queue": failure_promotion_queue,
        "failure_promotion_queue_count": len(failure_promotion_queue),
        "failure_promotion_command": failure_promotion_queue[1] if len(failure_promotion_queue) > 1 else None,
        "failure_implementation_command": failure_promotion_queue[2] if len(failure_promotion_queue) > 2 else None,
        "failure_apply_contract_command": failure_promotion_queue[3] if len(failure_promotion_queue) > 3 else None,
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
        "recovery_closure_state": recovery_closure_state,
        "recovery_closure_missing": recovery_closure_missing,
        "recovery_closure_missing_count": len(recovery_closure_missing),
        "recovery_closure_required_commands": recovery_closure_required_commands,
        "recovery_closure_next_required_command": recovery_closure_required_commands[0] if recovery_closure_required_commands else "",
        "recovery_closure_ready_to_retry": recovery_closure_ready_to_retry,
        "recovery_closure_blocks_auto_execution": recovery_closure_blocks_auto_execution,
        "recovery_closure_blocks_completion_claim": recovery_closure_blocks_completion_claim,
    }


def _execution_health_lines(health: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "Execution health handoff:",
        f"- Review required: {'yes' if health['review_required'] else 'no'}.",
        f"- Next audit command: `{health['next_command']}`.",
        f"- Verification coverage: {health['verification_coverage_state']}.",
        "- Operator limits: the operator's explicit stop times, work windows, pause commands, and newer instructions override priority goals.",
    ]
    if health["blocker_categories"]:
        lines.append(f"- Blocker categories: {', '.join(health['blocker_categories'])}.")
    if health.get("unreadable_recent_tool_run_rows"):
        lines.append(
            f"- Unreadable recent tool-run rows: {health['unreadable_recent_tool_run_rows']}; review `execution health report` before auto-run."
        )
    if health.get("failed_action_runs") or health.get("approval_held_action_runs"):
        lines.append(
            "- Action run attention: "
            f"{health['failed_action_runs']} failed/blocked, "
            f"{health['approval_held_action_runs']} approval-held."
        )
    if health["approval_proof_chains"]:
        lines.append("- Approval proof chains:")
        for approval_id, commands in sorted(health["approval_proof_chains"].items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 999999):
            lines.append(f"  - approval #{approval_id}:")
            lines.extend(f"    - `{command}`" for command in commands)
    if health["failure_promotion_queue"]:
        lines.append("- Failure promotion queue:")
        lines.extend(f"  - `{command}`" for command in health["failure_promotion_queue"])
    lines.extend(
        [
            "- Execution proof queue:",
            f"  - next required command: `{health['next_command']}`",
            f"  - proof queue count: {health['next_command_count']}",
            "  - proof queue: " + ", ".join(f"`{command}`" for command in health["next_commands"][:8]),
        ]
    )
    lines.extend(
        [
            "- Recovery closure gate:",
            f"  - state: {health['recovery_closure_state']}",
            f"  - ready for operator retry review: {'yes' if health['recovery_closure_ready_to_retry'] else 'no'}",
            f"  - blocks auto-run: {'yes' if health['recovery_closure_blocks_auto_execution'] else 'no'}",
            f"  - missing: {', '.join(health['recovery_closure_missing']) if health['recovery_closure_missing'] else 'none'}",
        ]
    )
    checklist_command = _recovery_closure_checklist_command(health)
    if checklist_command:
        lines.append(f"  - checklist overview: `{checklist_command}`")
    return lines


def _execution_health_metadata(health: dict[str, Any]) -> dict[str, Any]:
    return {
        "execution_health_review_required": health["review_required"],
        "execution_health_next_command": health["next_command"],
        "execution_health_next_commands": health["next_commands"],
        "execution_health_next_command_count": health["next_command_count"],
        "execution_health_proof_queue": health["next_commands"],
        "execution_health_proof_queue_count": health["next_command_count"],
        "execution_health_next_required_command": health["next_command"],
        "execution_health_next_proof_command": health["next_command"],
        "execution_health_recent_tool_run_rows": health["recent_tool_run_rows"],
        "execution_health_readable_recent_tool_runs": health["readable_recent_tool_runs"],
        "execution_health_unreadable_recent_tool_run_rows": health["unreadable_recent_tool_run_rows"],
        "failed_action_runs": health["failed_action_runs"],
        "approval_held_action_runs": health["approval_held_action_runs"],
        "approval_held_review_required": health["approval_held_review_required"],
        "approval_held_review_commands": health["approval_held_review_commands"],
        "approval_held_review_command_count": health["approval_held_review_command_count"],
        "approval_held_review_next_command": health["approval_held_review_next_command"],
        "execution_health_failed_action_runs": health["failed_action_runs"],
        "execution_health_approval_held_action_runs": health["approval_held_action_runs"],
        "execution_health_approval_held_review_required": health["approval_held_review_required"],
        "execution_health_approval_held_review_commands": health["approval_held_review_commands"],
        "execution_health_approval_held_review_command_count": health["approval_held_review_command_count"],
        "execution_health_approval_held_review_next_command": health["approval_held_review_next_command"],
        "execution_health_blocker_categories": health["blocker_categories"],
        "execution_health_blocker_count": health["blocker_count"],
        "execution_health_verification_coverage": health["verification_coverage_state"],
        "execution_health_approval_proof_chains": health["approval_proof_chains"],
        "execution_health_approval_proof_chain_count": health["approval_proof_chain_count"],
        "execution_health_repeated_failure_tools": health["repeated_failure_tools"],
        "execution_health_repeated_failure_count": health["repeated_failure_count"],
        "execution_health_failure_promotion_queue": health["failure_promotion_queue"],
        "execution_health_failure_promotion_queue_count": health["failure_promotion_queue_count"],
        "execution_health_failure_promotion_command": health["failure_promotion_command"],
        "execution_health_failure_implementation_command": health["failure_implementation_command"],
        "execution_health_failure_apply_contract_command": health["failure_apply_contract_command"],
        "execution_health_learning_target_run_id": health["learning_target_run_id"],
        "execution_health_learning_target_tool": health["learning_target_tool"],
        "execution_health_verification_handoff_command": health["verification_handoff_command"],
        "execution_health_recovery_handoff_command": health["recovery_handoff_command"],
        "execution_health_learning_handoff_command": health["learning_handoff_command"],
        "execution_health_learning_closure_command": health["learning_closure_command"],
        "execution_health_execution_learning_closure_command": health["execution_learning_closure_command"],
        "execution_learning_closure_command": health["execution_learning_closure_command"],
        "execution_health_target_verification_receipts": health["target_verification_receipts"],
        "execution_health_target_recovery_packets": health["target_recovery_packets"],
        "execution_health_target_after_action_learning_packets": health["target_after_action_learning_packets"],
        "execution_health_recovery_closure_state": health["recovery_closure_state"],
        "execution_health_recovery_closure_missing": health["recovery_closure_missing"],
        "execution_health_recovery_closure_missing_count": health["recovery_closure_missing_count"],
        "execution_health_recovery_closure_required_commands": health["recovery_closure_required_commands"],
        "execution_health_recovery_closure_next_required_command": health["recovery_closure_next_required_command"],
        "execution_health_recovery_closure_proof_queue": health["recovery_closure_required_commands"],
        "execution_health_recovery_closure_proof_queue_count": len(health["recovery_closure_required_commands"]),
        "execution_health_recovery_closure_next_proof_command": health["recovery_closure_next_required_command"],
        "execution_health_recovery_closure_ready_to_retry": health["recovery_closure_ready_to_retry"],
        "execution_health_recovery_closure_blocks_auto_execution": health["recovery_closure_blocks_auto_execution"],
        "execution_health_recovery_closure_blocks_completion_claim": health["recovery_closure_blocks_completion_claim"],
        "execution_health_recovery_closure_checklist_command": _recovery_closure_checklist_command(health),
        "execution_health_recovery_closure_should_open_checklist": bool(_recovery_closure_checklist_command(health)),
        "execution_health_recovery_closure_target_run_id": health["learning_target_run_id"],
        "execution_health_recovery_closure_target_tool_name": health["learning_target_tool"],
        "execution_health_recovery_closure_target_verification_receipts": health["target_verification_receipts"],
        "execution_health_recovery_closure_target_recovery_packets": health["target_recovery_packets"],
        "execution_health_recovery_closure_target_after_action_learning_packets": health["target_after_action_learning_packets"],
        "execution_health_operator_timeboxes_override_priority": True,
        "execution_health_stop_times_override_priority": True,
    }


def _agi_focus_snapshot() -> dict[str, Any]:
    raw_selected_gate = AGI_FOCUS_SELECTED_GATE
    if raw_selected_gate not in AGI_GATE_BUILD_TARGETS and AGI_GATE_BUILD_TARGETS:
        raw_selected_gate = next(iter(AGI_GATE_BUILD_TARGETS))
    target = AGI_GATE_BUILD_TARGETS.get(raw_selected_gate, {})
    selected_gate = _safe_focus_text(raw_selected_gate, limit=160)
    raw_likely_files = _coerce_list(target.get("files"))
    likely_files = _safe_focus_list(raw_likely_files, limit=260)
    file_integrity = _agi_focus_target_file_integrity(raw_likely_files)
    evidence_closure_commands = [
        f"agi next build move: {selected_gate}",
        f"completion audit: improve AGI gate {selected_gate}",
        "evidence ledger",
        f"completion claim gate: improve AGI gate {selected_gate}",
    ]
    focused_verification_commands = _safe_focus_list(target.get("tests"), limit=500)
    acceptance_checks = _safe_focus_list(target.get("acceptance"), limit=500)
    acceptance_gap_preview = acceptance_checks[:3]
    first_acceptance_gap = acceptance_gap_preview[0] if acceptance_gap_preview else ""
    return {
        "selected_gate": selected_gate,
        "target_title": _safe_focus_text(target.get("title"), limit=500),
        "build_command": f"agi next build move: {selected_gate}",
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
        "acceptance_gap_preview": acceptance_gap_preview,
        "acceptance_gap_preview_count": len(acceptance_gap_preview),
        "first_acceptance_gap": first_acceptance_gap,
        "build_packet_ready_for_review": bool(file_integrity["all_exist"] and focused_verification_commands and acceptance_checks),
        "selection_source": AGI_FOCUS_SELECTION_SOURCE,
        "selection_reason": AGI_FOCUS_SELECTION_REASON,
        "canonical_selector_command": AGI_FOCUS_CANONICAL_SELECTOR_COMMAND,
        "deliberate_focus_override": True,
    }


def _agi_focus_lines(agi: dict[str, Any]) -> list[str]:
    return [
        "",
        "AGI build-readiness handoff:",
        f"- Selected target: {agi['target_title'] or agi['selected_gate']}.",
        f"- Next build review: `{agi['build_command']}`.",
        f"- Selection source: {agi['selection_source']}.",
        f"- Canonical gate selector: `{agi['canonical_selector_command']}`.",
        f"- Target file integrity: {agi['target_file_integrity_status']}.",
        f"- Focused verification count: {agi['focused_verification_command_count']}.",
        f"- Acceptance check count: {agi['acceptance_check_count']}.",
        f"- Ready for review: {'yes' if agi['build_packet_ready_for_review'] else 'no'}.",
    ]


def _agi_focus_metadata(agi: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "agi_next_gate": agi["selected_gate"],
        "agi_next_target_title": agi["target_title"],
        "agi_next_build_command": agi["build_command"],
        "agi_next_evidence_closure_commands": agi["evidence_closure_commands"],
        "agi_next_evidence_closure_command_count": agi["evidence_closure_command_count"],
        "agi_next_focused_verification_commands": agi["focused_verification_commands"],
        "agi_next_focused_verification_command_count": agi["focused_verification_command_count"],
        "agi_next_likely_files": agi["likely_files"],
        "agi_next_likely_file_count": agi["likely_file_count"],
        "agi_next_target_file_integrity_status": agi["target_file_integrity_status"],
        "agi_next_target_files_checked": agi["target_files_checked"],
        "agi_next_target_files_exist": agi["target_files_exist"],
        "agi_next_missing_target_files": agi["missing_target_files"],
        "agi_next_missing_target_file_count": agi["missing_target_file_count"],
        "agi_next_target_integrity_blocks_start": agi["target_integrity_blocks_start"],
        "agi_next_acceptance_checks": agi["acceptance_checks"],
        "agi_next_acceptance_check_count": agi["acceptance_check_count"],
        "agi_next_acceptance_gap_preview": agi["acceptance_gap_preview"],
        "agi_next_acceptance_gap_preview_count": agi["acceptance_gap_preview_count"],
        "agi_next_first_acceptance_gap": agi["first_acceptance_gap"],
        "agi_next_build_packet_ready_for_review": agi["build_packet_ready_for_review"],
        "agi_focus_selection_source": agi["selection_source"],
        "agi_focus_selection_reason": agi["selection_reason"],
        "agi_focus_canonical_selector_command": agi["canonical_selector_command"],
        "agi_focus_deliberate_focus_override": agi["deliberate_focus_override"],
        "agi_focus_review_only": True,
        "agi_focus_draft_only": True,
        "agi_focus_loads_without_execution": True,
        "agi_focus_authorizes_execution": False,
        "agi_focus_authorizes_completion_claim": False,
        "agi_focus_approval_granted": False,
    }
    metadata["agi_focus_handoff"] = {
        "source": "agi_focus",
        "selected_gate": agi["selected_gate"],
        "target_title": agi["target_title"],
        "build_command": agi["build_command"],
        "evidence_closure_commands": agi["evidence_closure_commands"],
        "evidence_closure_command_count": agi["evidence_closure_command_count"],
        "focused_verification_commands": agi["focused_verification_commands"],
        "focused_verification_command_count": agi["focused_verification_command_count"],
        "likely_files": agi["likely_files"],
        "likely_file_count": agi["likely_file_count"],
        "target_file_integrity_status": agi["target_file_integrity_status"],
        "target_files_checked": agi["target_files_checked"],
        "target_files_exist": agi["target_files_exist"],
        "missing_target_files": agi["missing_target_files"],
        "missing_target_file_count": agi["missing_target_file_count"],
        "target_integrity_blocks_start": agi["target_integrity_blocks_start"],
        "acceptance_checks": agi["acceptance_checks"],
        "acceptance_check_count": agi["acceptance_check_count"],
        "acceptance_gap_preview": agi["acceptance_gap_preview"],
        "acceptance_gap_preview_count": agi["acceptance_gap_preview_count"],
        "first_acceptance_gap": agi["first_acceptance_gap"],
        "build_packet_ready_for_review": agi["build_packet_ready_for_review"],
        "selection_source": agi["selection_source"],
        "selection_reason": agi["selection_reason"],
        "canonical_selector_command": agi["canonical_selector_command"],
        "deliberate_focus_override": agi["deliberate_focus_override"],
        "review_only": True,
        "draft_only": True,
        "loads_without_execution": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "calls_model": False,
        "executes_tools": False,
        "writes_files": False,
        "reads_personal_data": False,
        "external_side_effect": False,
        "controls_computer": False,
        "queues_approval": False,
    }
    return metadata


def make_focus_tools(store: MemoryStore):
    def _first_goal_step(goals):
        for goal in goals:
            goal_id = _row_int(goal, "id")
            if goal_id is None:
                continue
            open_steps = [step for step in store.list_goal_steps(goal_id) if _row_text(step, "status") != "done"]
            if open_steps:
                return goal, open_steps[0]
        _, goal = _first_row_with_positive_id(goals)
        return (goal, None) if goal is not None else (None, None)

    def focus_brief(args: dict[str, Any]) -> ToolResult:
        objective = _bounded_text(args.get("objective"))
        limit = _bounded_int(args.get("limit"), 6)
        tasks = store.list_tasks(status="open", limit=limit)
        goals = store.list_goals(status="active", limit=limit)
        approvals = store.list_pending_approvals(limit=limit)
        approval_handoff = _approval_handoff(approvals)
        decisions = store.list_decisions(status="active", limit=3)
        preferences = store.list_preferences(status="active", limit=3)
        jobs = store.list_jobs()
        background = _background_rhythm_state(jobs)
        health = _execution_health_snapshot(store)
        agi_focus = _agi_focus_snapshot()

        lines = ["Jarvis focus brief:"]
        if objective:
            lines.append(f"Objective: {objective}")
        else:
            lines.append("Objective: choose the smallest useful next step.")

        lines.extend(["", "Start here:"])
        if approvals:
            lines.append("- Review pending approvals, then run `approval readiness #ID` before the last-look `approval packet #ID` and `approval chain proof #ID`.")
            lines.extend(_approval_handoff_lines(approval_handoff))
        task_id, task = _first_row_with_positive_id(tasks)
        goal_id, goal = _first_row_with_positive_id(goals)
        if task is not None:
            due = f" due {_row_text(task, 'due')}" if _row_text(task, "due") else ""
            priority_value = _row_text(task, "priority", "normal")
            priority = f" [{priority_value}]" if priority_value != "normal" else ""
            lines.append(f"- Top task: #{task_id} {_row_text(task, 'body', 'untitled task')}{due}{priority}.")
        elif tasks:
            lines.append("- Open tasks include unreadable rows; run `safe next actions` before choosing a task.")
        elif goal is not None:
            steps = store.list_goal_steps(int(goal_id))
            open_steps = [step for step in steps if _row_text(step, "status") != "done"]
            next_step = _row_text(open_steps[0], "body", "define one next step") if open_steps else "define one next step"
            lines.append(f"- Top goal: #{goal_id} {_row_text(goal, 'title', 'untitled goal')} -> {next_step}.")
        elif goals:
            lines.append("- Active goals include unreadable rows; run `safe next actions` before choosing a goal.")
        elif not background["ready"]:
            lines.append(f"- Restore the background rhythm: `{background['next_command']}` ({background['priority']})")
        else:
            lines.append("- Capture one task or create one goal before doing reactive work.")

        lines.extend(["", "Session plan:"])
        lines.append("- 1. Review context: `return brief`, `mission control`, or this focus brief.")
        if approvals:
            lines.append("- 2. Run `approval readiness #ID`, then `approval packet #ID`, then `approval chain proof #ID`, before deciding to approve, dismiss, or leave it blocked.")
        else:
            lines.append("- 2. Pick one low-risk task or goal step.")
        lines.append("- 3. Work in one visible chunk, then record the result as a task completion, decision, memory, or goal step.")
        lines.append("- 4. End by running `safe next actions` so Jarvis knows where to resume.")

        lines.extend(["", "Visible context:"])
        if tasks:
            lines.append("Open tasks:")
            for task in tasks[:4]:
                task_id = _row_positive_int_text(task)
                if not task_id:
                    continue
                due = f" due {_row_text(task, 'due')}" if _row_text(task, "due") else ""
                priority_value = _row_text(task, "priority", "normal")
                priority = f" [{priority_value}]" if priority_value != "normal" else ""
                lines.append(f"- #{task_id} {_row_text(task, 'body', 'untitled task')}{due}{priority}")
        else:
            lines.append("- No open tasks.")

        if goals:
            lines.append("Active goals:")
            for goal in goals[:4]:
                goal_id = _row_positive_int_text(goal)
                if not goal_id:
                    continue
                steps = store.list_goal_steps(int(goal_id))
                open_steps = [step for step in steps if _row_text(step, "status") != "done"]
                next_step = _row_text(open_steps[0], "body", "define one next step") if open_steps else "define one next step"
                lines.append(f"- #{goal_id} {_row_text(goal, 'title', 'untitled goal')}: {next_step}")
        else:
            lines.append("- No active goals.")

        if approvals:
            lines.append("Approval blockers:")
            for approval in approvals[:4]:
                approval_id = _row_positive_int_text(approval)
                if not approval_id:
                    continue
                lines.append(
                    f"- #{approval_id} {_row_text(approval, 'tool_name', 'unknown')}: "
                    f"{_row_text_by_keys(approval, ('user_input', 'request'), 'unreadable request')}"
                )
                lines.append(f"  readiness: `approval readiness {approval_id}`")
                lines.append(f"  last look: `approval packet {approval_id}`")
                lines.append(f"  proof: `approval chain proof {approval_id}`")
        else:
            lines.append("- No pending approvals.")

        if decisions:
            lines.append("Decision context:")
            for decision in decisions:
                decision_id = _row_positive_int_text(decision)
                if decision_id:
                    lines.append(f"- #{decision_id} {_row_text(decision, 'title', 'untitled decision')}")
        if preferences:
            lines.append("Active preferences:")
            for preference in preferences:
                key = _row_text(preference, "key")
                if key:
                    lines.append(f"- {key}: {_row_text(preference, 'value')}")

        enabled_jobs = background["enabled_jobs"]
        focus_handoff = _focus_brief_handoff(
            objective=objective,
            limit=limit,
            approvals=approvals,
            tasks=tasks,
            goals=goals,
            decisions=decisions,
            preferences=preferences,
            enabled_jobs=enabled_jobs,
            jobs=jobs,
            approval_handoff=approval_handoff,
            health=health,
            agi_focus=agi_focus,
            background=background,
        )
        lines.extend(
            [
                "",
                "Safety boundary:",
                "- Do not auto-run shell, file writes/deletes, clipboard reads, computer-control, reminders, or outside-world actions without explicit approval.",
                f"- Background jobs enabled: {len(enabled_jobs)} / {len(jobs)}.",
                f"- {_background_rhythm_line(background)}",
            ]
        )
        lines.extend(_execution_health_lines(health))
        lines.extend(_agi_focus_lines(agi_focus))

        return ToolResult(
            "focus_brief",
            True,
            "\n".join(lines),
            _read_only_metadata(
                open_tasks=len(tasks),
                active_goals=len(goals),
                pending_approvals=len(approvals),
                **approval_handoff,
                active_decisions=len(decisions),
                active_preferences=len(preferences),
                enabled_jobs=len(enabled_jobs),
                scheduled_jobs=len(jobs),
                **_background_rhythm_metadata(background),
                limit=limit,
                objective_length=len(objective),
                **_execution_health_metadata(health),
                **_agi_focus_metadata(agi_focus),
                focus_brief_handoff_ready=focus_handoff["handoff_ready"],
                focus_brief_ready_for_operator=focus_handoff["ready_for_operator"],
                focus_brief_state_changed=focus_handoff["state_changed"],
                focus_brief_changed=focus_handoff["changed"],
                focus_brief_content_in_handoff=focus_handoff["content_in_handoff"],
                focus_brief_start_kind=focus_handoff["start_kind"],
                focus_brief_start_command=focus_handoff["start_command"],
                focus_brief_start_label=focus_handoff["start_label"],
                focus_brief_session_plan=focus_handoff["session_plan"],
                focus_brief_next_commands=focus_handoff["next_commands"],
                focus_brief_next_safe_commands=focus_handoff["next_safe_commands"],
                focus_brief_next_command_count=focus_handoff["next_command_count"],
                focus_brief_next_safe_command_count=focus_handoff["next_safe_command_count"],
                focus_brief_authorizes_execution=focus_handoff["boundaries"]["authorizes_execution"],
                focus_brief_authorizes_completion_claim=focus_handoff["boundaries"]["authorizes_completion_claim"],
                focus_brief_approval_granted=focus_handoff["boundaries"]["approval_granted"],
                focus_brief_boundaries=focus_handoff["boundaries"],
                focus_brief_handoff=focus_handoff,
            ),
        )

    def next_session_plan(args: dict[str, Any]) -> ToolResult:
        objective = _bounded_text(args.get("objective"))
        limit = _bounded_int(args.get("limit"), 6)
        tasks = store.list_tasks(status="open", limit=limit)
        goals = store.list_goals(status="active", limit=limit)
        approvals = store.list_pending_approvals(limit=limit)
        approval_handoff = _approval_handoff(approvals)
        decisions = store.list_decisions(status="active", limit=3)
        preferences = store.list_preferences(status="active", limit=3)
        jobs = store.list_jobs()
        background = _background_rhythm_state(jobs)
        enabled_jobs = background["enabled_jobs"]
        health = _execution_health_snapshot(store)
        agi_focus = _agi_focus_snapshot()

        lines = ["Jarvis next session plan:"]
        if objective:
            lines.append(f"Objective: {objective}")
        else:
            lines.append("Objective: resume safely with the smallest useful next move.")

        lines.extend(
            [
                "",
                "Resume order:",
                "- 1. Read `return brief` or `handoff brief` before assuming old context is complete.",
                "- 2. Review `pending approvals`, then run `approval readiness #ID`, the last-look `approval packet #ID`, and `approval chain proof #ID`.",
                "- 3. Pick one visible task or goal step and keep the session to one clear chunk.",
                "- 4. End with `session closeout` or `safe next actions` so the next resume point is obvious.",
            ]
        )

        lines.extend(["", "First safe move:"])
        approval_id, approval = _first_row_with_positive_id(approvals)
        task_id, task = _first_row_with_positive_id(tasks)
        goal_id, goal = _first_row_with_positive_id(goals)
        if approval is not None:
            first_safe_move = f"Run `approval readiness {approval_id}` for `{_row_text(approval, 'tool_name', 'unknown')}`; use `approval packet {approval_id}` only after readiness points to last-look review, then run `approval chain proof {approval_id}` before any approval decision."
            lines.append(f"- {first_safe_move}")
            lines.extend(_approval_handoff_lines(approval_handoff))
        elif approvals:
            first_safe_move = "Review pending approvals with `pending approvals`; at least one row needs handoff repair before action."
            lines.append(f"- {first_safe_move}")
        elif task is not None:
            priority_value = _row_text(task, "priority", "normal")
            priority = f" [{priority_value}]" if priority_value != "normal" else ""
            due = f" due {_row_text(task, 'due')}" if _row_text(task, "due") else ""
            first_safe_move = f"Work task #{task_id} {_row_text(task, 'body', 'untitled task')}{due}{priority}."
            lines.append(f"- {first_safe_move}")
        elif tasks:
            first_safe_move = "Open tasks include unreadable rows; run `safe next actions` before choosing a task."
            lines.append(f"- {first_safe_move}")
        elif goal is not None:
            steps = store.list_goal_steps(int(goal_id))
            open_steps = [step for step in steps if _row_text(step, "status") != "done"]
            next_step = _row_text(open_steps[0], "body", "define one next step") if open_steps else "define one next step"
            first_safe_move = f"Advance goal #{goal_id} {_row_text(goal, 'title', 'untitled goal')}: {next_step}."
            lines.append(f"- {first_safe_move}")
        elif goals:
            first_safe_move = "Active goals include unreadable rows; run `safe next actions` before choosing a goal."
            lines.append(f"- {first_safe_move}")
        elif not background["ready"]:
            first_safe_move = f"Restore the background rhythm with `{background['next_command']}` ({background['priority']})."
            lines.append(f"- {first_safe_move}")
        else:
            first_safe_move = "Capture one task before starting open-ended assistant work."
            lines.append(f"- {first_safe_move}")

        lines.extend(["", "Session guardrails:"])
        lines.append("- Do not auto-run shell/code, file writes/deletes, clipboard reads, reminders, personal data, external side effects, or computer control.")
        lines.append("- Prefer read-only previews first: `action rehearsal`, `autonomy plan`, `computer task plan`, or `integration action preview`.")
        lines.append("- Do not mark tasks complete or save new notes from this plan alone.")

        lines.extend(["", "Visible context:"])
        if tasks:
            lines.append("Open tasks:")
            for task in tasks[:4]:
                task_id = _row_positive_int_text(task)
                if not task_id:
                    continue
                priority_value = _row_text(task, "priority", "normal")
                priority = f" [{priority_value}]" if priority_value != "normal" else ""
                lines.append(f"- #{task_id} {_row_text(task, 'body', 'untitled task')}{priority}")
        else:
            lines.append("- No open tasks.")

        if goals:
            lines.append("Active goals:")
            for goal in goals[:4]:
                goal_id = _row_positive_int_text(goal)
                if not goal_id:
                    continue
                steps = store.list_goal_steps(int(goal_id))
                open_steps = [step for step in steps if _row_text(step, "status") != "done"]
                next_step = _row_text(open_steps[0], "body", "define one next step") if open_steps else "define one next step"
                lines.append(f"- #{goal_id} {_row_text(goal, 'title', 'untitled goal')}: {next_step}")
        else:
            lines.append("- No active goals.")

        if approvals:
            lines.append("Approval blockers:")
            for approval in approvals[:4]:
                approval_id = _row_positive_int_text(approval)
                if not approval_id:
                    continue
                lines.append(
                    f"- #{approval_id} {_row_text(approval, 'tool_name', 'unknown')}: "
                    f"{_row_text_by_keys(approval, ('user_input', 'request'), 'unreadable request')}"
                )
                lines.append(f"  readiness: `approval readiness {approval_id}`")
                lines.append(f"  last look: `approval packet {approval_id}`")
                lines.append(f"  proof: `approval chain proof {approval_id}`")
        else:
            lines.append("- No pending approvals.")

        if decisions:
            lines.append("Decision anchors:")
            for decision in decisions:
                decision_id = _row_positive_int_text(decision)
                if decision_id:
                    lines.append(f"- #{decision_id} {_row_text(decision, 'title', 'untitled decision')}")
        if preferences:
            lines.append("Style preferences:")
            for preference in preferences:
                key = _row_text(preference, "key")
                if key:
                    lines.append(f"- {key}: {_row_text(preference, 'value')}")
        lines.append(f"Scheduled jobs visible: {len(enabled_jobs)} enabled / {len(jobs)} total.")
        lines.append(_background_rhythm_line(background))
        lines.extend(_execution_health_lines(health))
        lines.extend(_agi_focus_lines(agi_focus))
        next_session_handoff = _next_session_plan_handoff(
            objective=objective,
            first_safe_move=first_safe_move,
            limit=limit,
            approvals=approvals,
            tasks=tasks,
            goals=goals,
            decisions=decisions,
            preferences=preferences,
            enabled_jobs=enabled_jobs,
            jobs=jobs,
            approval_handoff=approval_handoff,
            health=health,
            agi_focus=agi_focus,
            background=background,
        )

        return ToolResult(
            "next_session_plan",
            True,
            "\n".join(lines),
            _read_only_metadata(
                open_tasks=len(tasks),
                active_goals=len(goals),
                pending_approvals=len(approvals),
                **approval_handoff,
                active_decisions=len(decisions),
                active_preferences=len(preferences),
                enabled_jobs=len(enabled_jobs),
                scheduled_jobs=len(jobs),
                **_background_rhythm_metadata(background),
                limit=limit,
                objective_length=len(objective),
                next_session_plan_handoff_ready=next_session_handoff["handoff_ready"],
                next_session_plan_ready_for_operator=next_session_handoff["ready_for_operator"],
                next_session_plan_state_changed=next_session_handoff["state_changed"],
                next_session_plan_changed=next_session_handoff["changed"],
                next_session_plan_content_in_handoff=next_session_handoff["content_in_handoff"],
                next_session_plan_first_safe_move=next_session_handoff["first_safe_move"],
                next_session_plan_resume_order=next_session_handoff["resume_order"],
                next_session_plan_next_commands=next_session_handoff["next_commands"],
                next_session_plan_next_safe_commands=next_session_handoff["next_safe_commands"],
                next_session_plan_next_command_count=next_session_handoff["next_command_count"],
                next_session_plan_next_safe_command_count=next_session_handoff["next_safe_command_count"],
                next_session_plan_authorizes_execution=next_session_handoff["boundaries"]["authorizes_execution"],
                next_session_plan_authorizes_completion_claim=next_session_handoff["boundaries"]["authorizes_completion_claim"],
                next_session_plan_approval_granted=next_session_handoff["boundaries"]["approval_granted"],
                next_session_plan_boundaries=next_session_handoff["boundaries"],
                next_session_plan_handoff=next_session_handoff,
                **_execution_health_metadata(health),
                **_agi_focus_metadata(agi_focus),
            ),
        )

    def work_session_packet(args: dict[str, Any]) -> ToolResult:
        objective = _bounded_text(args.get("objective"))
        limit = _bounded_int(args.get("limit"), 6)
        tasks = store.list_tasks(status="open", limit=limit)
        goals = store.list_goals(status="active", limit=limit)
        approvals = store.list_pending_approvals(limit=limit)
        approval_handoff = _approval_handoff(approvals)
        decisions = store.list_decisions(status="active", limit=3)
        preferences = store.list_preferences(status="active", limit=3)
        jobs = store.list_jobs()
        background = _background_rhythm_state(jobs)
        enabled_jobs = background["enabled_jobs"]
        goal, step = _first_goal_step(goals)
        health = _execution_health_snapshot(store)
        agi_focus = _agi_focus_snapshot()

        approval_id, approval = _first_row_with_positive_id(approvals)
        task_id, task = _first_row_with_positive_id(tasks)
        goal_id = _row_positive_int_text(goal) if goal is not None else ""
        if approval is not None:
            start_move = f"Run `approval readiness {approval_id}` before risky work; use `approval packet {approval_id}` only after readiness points to last-look review, then run `approval chain proof {approval_id}` before deciding. Use `approval detail {approval_id}` if anything still looks unclear."
            stop_condition = "Stop if readiness, the approval packet, or the approval chain proof is stale, unclear, broad, destructive, mismatched, or not explicitly trusted by the operator."
        elif approvals:
            start_move = "Review pending approvals with `pending approvals`; at least one row needs handoff repair before action."
            stop_condition = "Stop before approving or dismissing anything until the pending approval row renders with a clear ID and tool."
        elif task is not None:
            start_move = f"Work one visible chunk toward task #{task_id}: {_row_text(task, 'body', 'untitled task')}."
            stop_condition = "Stop before marking the task complete unless the result is verified."
        elif tasks:
            start_move = "Open tasks include unreadable rows; run `safe next actions` before choosing a task."
            stop_condition = "Stop after producing a readable task handoff; do not complete unreadable task rows."
        elif goal is not None and step is not None:
            step_id = _row_positive_int_text(step)
            step_label = f" step #{step_id}" if step_id else ""
            start_move = f"Work one visible chunk toward goal #{goal_id}{step_label}: {_row_text(step, 'body', 'define one next step')}."
            stop_condition = "Stop before completing the goal step unless the result is verified."
        elif goal is not None:
            start_move = f"Define the next concrete step for goal #{goal_id}: {_row_text(goal, 'title', 'untitled goal')}."
            stop_condition = "Stop after adding a clear next step; do not start open-ended work from a vague goal."
        elif goals:
            start_move = "Active goals include unreadable rows; run `safe next actions` before choosing a goal."
            stop_condition = "Stop after producing a readable goal handoff; do not complete unreadable goal rows."
        elif not background["ready"]:
            start_move = f"Restore the background rhythm with `{background['next_command']}` ({background['priority']})."
            stop_condition = "Stop after confirming the background rhythm command is visible; do not run or create jobs from this packet alone."
        else:
            start_move = "Capture one concrete task before doing open-ended assistant work."
            stop_condition = "Stop after capturing the task and rerun `work session packet`."

        lines = [
            "Jarvis work session packet:",
            "This is read-only. It prepares a start packet without executing tools, writing notes, completing tasks, approving requests, controlling the computer, or queuing approvals.",
            "",
            f"Objective: {objective or 'start the smallest useful safe work chunk'}",
            "",
            "Start move:",
            f"- {start_move}",
            "",
            "Preflight checklist:",
            "- Read `chat continuity brief` or `return brief` if context may be stale.",
            "- Run `next action packet` if the next move is still ambiguous.",
            "- Run `action rehearsal: <request>` before shell/code, file writes/deletes, clipboard reads, personal data, external side effects, or computer control.",
            "- Run `approval readiness #ID`, then `approval packet #ID`, then `approval chain proof #ID`, before approving any queued risky action.",
            "- Keep pending approvals blocked unless the operator explicitly approves the exact request.",
            "",
            "Visible context:",
            f"- pending approvals: {len(approvals)}",
            f"- open tasks: {len(tasks)}",
            f"- active goals: {len(goals)}",
            f"- active decisions: {len(decisions)}",
            f"- active preferences: {len(preferences)}",
            f"- enabled scheduled jobs: {len(enabled_jobs)} / {len(jobs)}",
            f"- {_background_rhythm_line(background)}",
            "",
            "Stop condition:",
            f"- {stop_condition}",
            "",
            "End-of-session verification:",
            "- Run `build delta` if code or tool behavior changed.",
            f"- Run `{health['next_command']}` before advancing if execution health review is required.",
            "- Run `session closeout` or `save handoff brief` after a longer work chunk.",
            "- Leave risky actions in the approval queue unless the operator approves them.",
        ]
        if approvals:
            lines.extend(["", "Approval handoff:"])
            lines.extend(_approval_handoff_lines(approval_handoff))
        lines.extend(_execution_health_lines(health))
        lines.extend(_agi_focus_lines(agi_focus))
        if decisions:
            decision_id, decision = _first_row_with_positive_id(decisions)
            if decision is not None:
                lines.append(f"- Decision anchor: #{decision_id} {_row_text(decision, 'title', 'untitled decision')}")
        if preferences:
            preference = preferences[0]
            key = _row_text(preference, "key")
            if key:
                lines.append(f"- Preference anchor: {key} = {_row_text(preference, 'value')}")
        work_session_handoff = _work_session_packet_handoff(
            objective=objective,
            start_move=start_move,
            stop_condition=stop_condition,
            limit=limit,
            approvals=approvals,
            tasks=tasks,
            goals=goals,
            decisions=decisions,
            preferences=preferences,
            enabled_jobs=enabled_jobs,
            jobs=jobs,
            approval_handoff=approval_handoff,
            health=health,
            agi_focus=agi_focus,
            background=background,
        )

        return ToolResult(
            "work_session_packet",
            True,
            "\n".join(lines),
            _read_only_metadata(
                pending_approvals=len(approvals),
                **approval_handoff,
                open_tasks=len(tasks),
                active_goals=len(goals),
                active_decisions=len(decisions),
                active_preferences=len(preferences),
                enabled_jobs=len(enabled_jobs),
                scheduled_jobs=len(jobs),
                **_background_rhythm_metadata(background),
                limit=limit,
                objective_length=len(objective),
                work_session_packet_handoff_ready=work_session_handoff["handoff_ready"],
                work_session_packet_ready_for_operator=work_session_handoff["ready_for_operator"],
                work_session_packet_state_changed=work_session_handoff["state_changed"],
                work_session_packet_changed=work_session_handoff["changed"],
                work_session_packet_content_in_handoff=work_session_handoff["content_in_handoff"],
                work_session_packet_start_move=work_session_handoff["start_move"],
                work_session_packet_stop_condition=work_session_handoff["stop_condition"],
                work_session_packet_next_commands=work_session_handoff["next_commands"],
                work_session_packet_next_safe_commands=work_session_handoff["next_safe_commands"],
                work_session_packet_next_command_count=work_session_handoff["next_command_count"],
                work_session_packet_next_safe_command_count=work_session_handoff["next_safe_command_count"],
                work_session_packet_authorizes_execution=work_session_handoff["boundaries"]["authorizes_execution"],
                work_session_packet_authorizes_completion_claim=work_session_handoff["boundaries"]["authorizes_completion_claim"],
                work_session_packet_approval_granted=work_session_handoff["boundaries"]["approval_granted"],
                work_session_packet_boundaries=work_session_handoff["boundaries"],
                work_session_packet_handoff=work_session_handoff,
                **_execution_health_metadata(health),
                **_agi_focus_metadata(agi_focus),
            ),
        )

    return focus_brief, next_session_plan, work_session_packet
