from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Callable

from jarvis_v2.agent.types import ToolResult
from jarvis_v2.automations.compaction import COMPACTION_JOB_NAME, COMPACTION_JOB_TYPE
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import MemoryStore


MAX_STATE_LIMIT = 25
MAX_STATE_FIELD_CHARS = 240
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
READ_ONLY_BOUNDARY_FALSE_FLAGS = {
    "calls_model": False,
    "executes_tools": False,
    "queues_approval": False,
    "requires_approval": False,
    "authorizes_execution": False,
    "authorizes_completion_claim": False,
    "approval_granted": False,
    "approves_request": False,
    "dismisses_request": False,
    "reads_personal_data": False,
    "reads_private_data": False,
    "executes_side_effect": False,
    "external_side_effect": False,
    "controls_computer": False,
    "writes_files": False,
    "writes_memory": False,
    "writes_notes": False,
    "speaks": False,
    "completes_tasks": False,
}


def _bounded_limit(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return max(1, min(MAX_STATE_LIMIT, int(value)))
    except (TypeError, ValueError):
        return default


def _raw_limit_metadata(value: Any, *, limit: int) -> dict[str, Any]:
    if value is None:
        return {"limit": limit}
    if isinstance(value, bool):
        return {"limit": limit, "raw_limit": str(value)}
    try:
        int(value)
    except (TypeError, ValueError):
        return {"limit": limit, "raw_limit": _short_metadata(value, limit=80)}
    return {"limit": limit}


def _short(value: Any, *, limit: int = MAX_STATE_FIELD_CHARS) -> str:
    try:
        text = "" if value is None else str(value)
    except Exception:
        return "<unreadable>"
    text = text.strip()
    text = " ".join(text.split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _short_metadata(value: Any, *, limit: int = 80) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", _short(value, limit=limit))


def _safe_vault_path_display(path: Path | str | None, vault: ObsidianVault) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return str(candidate.relative_to(vault.root_path))
    except ValueError:
        return _short_metadata(candidate, limit=MAX_STATE_FIELD_CHARS)


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "calls_model": False,
        "executes_tools": False,
        "queues_approval": False,
        "requires_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "approves_request": False,
        "dismisses_request": False,
        "reads_personal_data": False,
        "reads_private_data": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "controls_computer": False,
        "writes_files": False,
        "writes_memory": False,
        "writes_notes": False,
        "speaks": False,
        "completes_tasks": False,
    }
    metadata.update(extra)
    return metadata


_STATE_INT_FIELDS = {"id", "messages"}
_STATE_BOOL_FIELDS = {"enabled", "ok"}
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


def _state_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    return False


def _job_enabled(job: Any) -> bool:
    try:
        return _state_bool(job["enabled"])
    except Exception:
        return False


def _safe_row_payload(row: Any, fields: tuple[str, ...]) -> dict[str, Any] | None:
    payload: dict[str, Any] = {}
    for field in fields:
        try:
            value = row[field]
        except Exception:
            if field == "metadata":
                value = "{}"
            else:
                return None
        if field in _STATE_INT_FIELDS:
            try:
                payload[field] = int(value)
            except (TypeError, ValueError, OverflowError):
                return None
        elif field in _STATE_BOOL_FIELDS:
            try:
                payload[field] = _state_bool(value)
            except Exception:
                return None
        elif field == "metadata":
            if isinstance(value, dict):
                payload[field] = dict(value)
            elif value is None:
                payload[field] = {}
            else:
                try:
                    payload[field] = str(value)
                except Exception:
                    return None
        else:
            if value is None:
                payload[field] = ""
            else:
                try:
                    payload[field] = str(value)
                except Exception:
                    return None
    return payload


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


def _payload_metadata(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata", {})
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw or "{}")
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


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


def _tool_run_status(row: dict[str, Any]) -> str:
    if row.get("ok"):
        return "ok"
    if _is_approval_hold_metadata(_payload_metadata(row)):
        return "approval_held"
    return "failed"


def _tool_run_status_counts(tool_runs: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"ok": 0, "failed": 0, "approval_held": 0}
    for row in tool_runs:
        counts[_tool_run_status(row)] += 1
    return counts


def _tool_run_status_label(row: dict[str, Any]) -> str:
    status = _tool_run_status(row)
    return "approval held" if status == "approval_held" else status


def _safe_row_payloads(rows: list[Any], fields: tuple[str, ...]) -> tuple[list[dict[str, Any]], int]:
    readable: list[dict[str, Any]] = []
    unreadable = 0
    for row in rows:
        payload = _safe_row_payload(row, fields)
        if payload is None:
            unreadable += 1
        else:
            readable.append(payload)
    return readable, unreadable


def _safe_goal_steps(store: MemoryStore, goal_id: int) -> tuple[list[dict[str, Any]], int]:
    try:
        rows = store.list_goal_steps(goal_id)
    except Exception:
        return [], 1
    return _safe_row_payloads(rows, ("body", "status"))


def _first_approval_handoff(approvals: list[Any]) -> dict[str, Any]:
    if not approvals:
        return {
            "first_approval_id": None,
            "first_approval_tool": "",
            "first_readiness_command": "",
            "first_last_look_command": "",
            "first_proof_command": "",
            "first_approve_command": "",
            "first_dismiss_command": "",
        }
    approval = approvals[0]
    approval_id = int(approval["id"])
    return {
        "first_approval_id": approval_id,
        "first_approval_tool": str(approval["tool_name"]),
        "first_readiness_command": f"approval readiness {approval_id}",
        "first_last_look_command": f"approval packet {approval_id}",
        "first_proof_command": f"approval chain proof {approval_id}",
        "first_approve_command": f"approve approval {approval_id}",
        "first_dismiss_command": f"dismiss approval {approval_id}",
    }


def _background_rhythm_state(jobs: list[Any], *, unreadable_job_rows: int = 0) -> dict[str, Any]:
    enabled_jobs = [job for job in jobs if _job_enabled(job)]
    state_snapshot_jobs = [job for job in jobs if str(job["job_type"]) == "state_snapshot"]
    enabled_state_snapshot_jobs = [job for job in state_snapshot_jobs if _job_enabled(job)]
    disabled_state_snapshot_jobs = [job for job in state_snapshot_jobs if not _job_enabled(job)]
    conversation_compaction_jobs = [
        job
        for job in jobs
        if str(job["job_type"]) == COMPACTION_JOB_TYPE
        or str(job["name"]).casefold() == COMPACTION_JOB_NAME.casefold()
    ]
    enabled_conversation_compaction_jobs = [job for job in conversation_compaction_jobs if _job_enabled(job)]
    disabled_conversation_compaction_jobs = [job for job in conversation_compaction_jobs if not _job_enabled(job)]
    if unreadable_job_rows:
        ready = False
        issue = "Scheduled job state has unreadable rows."
        next_command = "list scheduled jobs"
        priority = "inspect scheduled jobs before trusting the background rhythm."
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
        "unreadable_scheduled_job_rows": unreadable_job_rows,
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


def _brain_loop_handoff(
    *,
    limit: int,
    memories: list[Any],
    skills: list[Any],
    goals: list[Any],
    tasks: list[Any],
    decisions: list[Any],
    preferences: list[Any],
    sessions: list[Any],
    tool_runs: list[Any],
    approvals: list[Any],
    jobs: list[Any],
    enabled_jobs: list[Any],
    approval_handoff: dict[str, Any],
    background: dict[str, Any],
    has_profile: bool,
) -> dict[str, Any]:
    tool_run_status_counts = _tool_run_status_counts(tool_runs)
    start_kind = "capture_task"
    start_command = "safe next actions"
    start_label = "Capture one task or create one goal before reactive work."
    if approvals:
        start_kind = "approval_review"
        approval_id = int(approvals[0]["id"])
        start_command = str(approval_handoff.get("first_readiness_command") or f"approval readiness {approval_id}")
        start_label = f"Review approval #{approval_id} for {approvals[0]['tool_name']}."
    elif tasks:
        start_kind = "task"
        start_command = f"task detail {tasks[0]['id']}"
        start_label = f"Work one visible step on task #{tasks[0]['id']}."
    elif goals:
        start_kind = "goal"
        start_command = f"goal detail {goals[0]['id']}"
        start_label = f"Work one visible step on goal #{goals[0]['id']}."
    elif not background["ready"]:
        start_kind = "background_repair"
        start_command = str(background["next_command"])
        start_label = str(background["priority"])

    next_commands = [
        start_command,
        "skill match preview: <request>",
        "assistant turn rehearsal: <message>",
        "rehearse: <request>",
        "focus brief",
        "work queue",
    ]
    if approval_handoff.get("first_readiness_command"):
        next_commands.extend(
            [
                str(approval_handoff["first_readiness_command"]),
                str(approval_handoff["first_last_look_command"]),
                str(approval_handoff["first_proof_command"]),
            ]
        )
    if not background["ready"]:
        next_commands.append(str(background["next_command"]))
    next_commands.append("safe next actions")
    deduped_next_commands = list(dict.fromkeys(command for command in next_commands if command))

    return {
        "source": "brain_loop_report",
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "start_kind": start_kind,
        "start_command": start_command,
        "start_label": _short_metadata(start_label, limit=120),
        "loop_order": [
            "perception and conversation",
            "memory and skills",
            "planning and current work",
            "execution boundary",
            "verification and audit",
        ],
        "next_commands": deduped_next_commands,
        "next_command_count": len(deduped_next_commands),
        "next_safe_commands": deduped_next_commands,
        "next_safe_command_count": len(deduped_next_commands),
        "profile_context": has_profile,
        "recent_memories": len(memories),
        "saved_skills": len(skills),
        "active_goals": len(goals),
        "open_tasks": len(tasks),
        "active_decisions": len(decisions),
        "active_preferences": len(preferences),
        "recent_sessions": len(sessions),
        "recent_tool_runs": len(tool_runs),
        "recent_ok_tool_runs": tool_run_status_counts["ok"],
        "recent_failed_tool_runs": tool_run_status_counts["failed"],
        "recent_approval_held_tool_runs": tool_run_status_counts["approval_held"],
        "pending_approvals": len(approvals),
        "enabled_jobs": len(enabled_jobs),
        "scheduled_jobs": len(jobs),
        **_background_rhythm_metadata(background),
        "limit": limit,
        "approval_handoff": approval_handoff,
        "first_approval_id": approval_handoff.get("first_approval_id"),
        "first_approval_tool": approval_handoff.get("first_approval_tool", ""),
        "first_readiness_command": approval_handoff.get("first_readiness_command", ""),
        "first_last_look_command": approval_handoff.get("first_last_look_command", ""),
        "first_proof_command": approval_handoff.get("first_proof_command", ""),
        "first_approve_command": approval_handoff.get("first_approve_command", ""),
        "first_dismiss_command": approval_handoff.get("first_dismiss_command", ""),
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
        "boundaries": {
            "read_only": True,
            "operator_timeboxes_override_priority": True,
            "stop_times_override_priority": True,
            **READ_ONLY_BOUNDARY_FALSE_FLAGS,
        },
    }


def make_state_tools(
    store: MemoryStore,
    vault: ObsidianVault,
    *,
    effect_authority: Callable[[], None] | None = None,
):
    def brain_loop_report(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_limit(args.get("limit", 6), 6)
        memories, unreadable_memory_rows = _safe_row_payloads(store.recent_memories(limit=limit), ("category", "title"))
        skills, unreadable_skill_rows = _safe_row_payloads(store.list_active_skills(limit=limit), ("name", "trigger", "body"))
        goals, unreadable_goal_rows = _safe_row_payloads(store.list_goals(status="active", limit=limit), ("id", "title", "status"))
        tasks, unreadable_task_rows = _safe_row_payloads(store.list_tasks(status="open", limit=limit), ("id", "body"))
        decisions, unreadable_decision_rows = _safe_row_payloads(store.list_decisions(status="active", limit=3), ("id", "title"))
        preferences, unreadable_preference_rows = _safe_row_payloads(store.list_preferences(status="active", limit=5), ("key", "value"))
        sessions, unreadable_session_rows = _safe_row_payloads(store.list_sessions(limit=3), ("session_id", "messages", "last_at"))
        tool_runs, unreadable_tool_run_rows = _safe_row_payloads(
            store.recent_tool_runs(limit=limit),
            ("tool_name", "risk", "ok", "metadata"),
        )
        approvals, unreadable_approval_rows = _safe_row_payloads(store.list_pending_approvals(limit=limit), ("id", "tool_name"))
        approval_handoff = _first_approval_handoff(approvals)
        jobs, unreadable_job_rows = _safe_row_payloads(store.list_jobs(), ("name", "job_type", "enabled"))
        background = _background_rhythm_state(jobs, unreadable_job_rows=unreadable_job_rows)
        profile = vault.read_profile(max_chars=1200).strip()
        has_profile = bool(profile and profile != "# Profile")
        enabled_jobs = background["enabled_jobs"]
        goal_steps: list[dict[str, Any]] = []
        unreadable_goal_step_rows = 0
        if goals:
            goal_steps, unreadable_goal_step_rows = _safe_goal_steps(store, goals[0]["id"])
        unreadable_state_rows = (
            unreadable_memory_rows
            + unreadable_skill_rows
            + unreadable_goal_rows
            + unreadable_goal_step_rows
            + unreadable_task_rows
            + unreadable_decision_rows
            + unreadable_preference_rows
            + unreadable_session_rows
            + unreadable_tool_run_rows
            + unreadable_approval_rows
            + unreadable_job_rows
        )
        readable_state_rows = (
            len(memories)
            + len(skills)
            + len(goals)
            + len(goal_steps)
            + len(tasks)
            + len(decisions)
            + len(preferences)
            + len(sessions)
            + len(tool_runs)
            + len(approvals)
            + len(jobs)
        )
        brain_loop_handoff = _brain_loop_handoff(
            limit=limit,
            memories=memories,
            skills=skills,
            goals=goals,
            tasks=tasks,
            decisions=decisions,
            preferences=preferences,
            sessions=sessions,
            tool_runs=tool_runs,
            approvals=approvals,
            jobs=jobs,
            enabled_jobs=enabled_jobs,
            approval_handoff=approval_handoff,
            background=background,
            has_profile=has_profile,
        )

        lines = [
            "Jarvis brain loop report:",
            "This is read-only. It orients Jarvis before action without executing tools, writing memory, changing skills, controlling the computer, or queuing approvals.",
            "",
            "1. Perception and conversation",
            f"- profile context: {'present' if has_profile else 'empty'}",
            f"- recent sessions visible: {len(sessions)}",
            "- text chat is the active input surface; voice and desktop actions remain separately gated.",
            "",
        ]
        if unreadable_state_rows:
            lines.extend(
                [
                    "Local row safety",
                    f"- hidden malformed local state rows: {unreadable_state_rows}",
                    "",
                ]
            )
        lines.extend(
            [
                "2. Memory and skills",
                f"- recent memories loaded: {len(memories)}",
                f"- saved skills visible: {len(skills)}",
            ]
        )
        if memories:
            lines.append(f"- newest memory: [{_short(memories[0]['category'], limit=40)}] {_short(memories[0]['title'])}")
        if skills:
            lines.append(f"- first skill candidate: {_short(skills[0]['name'], limit=80)} -> {_short(skills[0]['trigger'] or skills[0]['body'])}")

        lines.extend(
            [
                "",
                "3. Planning and current work",
                f"- open tasks: {len(tasks)}",
                f"- active goals: {len(goals)}",
            ]
        )
        if tasks:
            lines.append(f"- next task: #{tasks[0]['id']} {_short(tasks[0]['body'])}")
        if goals:
            goal = goals[0]
            open_steps = [step for step in goal_steps if step["status"] != "done"]
            next_step = _short(open_steps[0]["body"]) if open_steps else "define the next step"
            lines.append(f"- next goal step: #{goal['id']} {_short(goal['title'])} -> {next_step}")
        if decisions:
            lines.append(f"- decision anchor: #{decisions[0]['id']} {_short(decisions[0]['title'])}")
        if preferences:
            preference = preferences[0]
            lines.append(f"- active preference: {_short(preference['key'], limit=80)} = {_short(preference['value'])}")

        lines.extend(
            [
                "",
                "4. Execution boundary",
                f"- pending approvals: {len(approvals)}",
                f"- scheduled jobs enabled: {len(enabled_jobs)} / {len(jobs)}",
                f"- {_background_rhythm_line(background)}",
                "- safe default: read-only and local-safe actions may run; personal data, shell/code, computer control, external effects, and destructive actions ask first.",
            ]
        )
        if approvals:
            lines.append(f"- first blocker: approval #{approvals[0]['id']} {_short(approvals[0]['tool_name'], limit=80)}")
            lines.append(f"- first safe approval handoff: `{approval_handoff['first_readiness_command']}` -> `{approval_handoff['first_last_look_command']}` -> `{approval_handoff['first_proof_command']}`")
            lines.append(f"- decision commands: `{approval_handoff['first_approve_command']}` or `{approval_handoff['first_dismiss_command']}`")

        lines.extend(["", "5. Verification and audit"])
        if tool_runs:
            tool_run_status_counts = _tool_run_status_counts(tool_runs)
            lines.append(
                f"- recent tool runs: {len(tool_runs)} "
                f"({tool_run_status_counts['ok']} ok, {tool_run_status_counts['failed']} failed/blocked, "
                f"{tool_run_status_counts['approval_held']} approval-held)"
            )
            latest_status = _tool_run_status_label(tool_runs[0])
            lines.append(
                f"- latest run: {_short(tool_runs[0]['tool_name'], limit=80)} "
                f"[{_short(tool_runs[0]['risk'], limit=40)}, {latest_status}]"
            )
        else:
            lines.append("- no recent tool runs logged")

        lines.extend(
            [
                "",
                "Good next checks:",
                "- `skill match preview: <request>` before using saved procedures.",
                "- `assistant turn rehearsal: <message>` before ambiguous chat-vs-action turns.",
                "- `rehearse: <request>` before risky tools.",
                "- `work queue` or `focus brief` before longer build sessions.",
            ]
        )
        return ToolResult(
            "brain_loop_report",
            True,
            "\n".join(lines),
            _safe_metadata(
                profile_context=has_profile,
                **_raw_limit_metadata(args.get("limit"), limit=limit),
                recent_memories=len(memories),
                saved_skills=len(skills),
                open_tasks=len(tasks),
                active_goals=len(goals),
                active_decisions=len(decisions),
                active_preferences=len(preferences),
                pending_approvals=len(approvals),
                **approval_handoff,
                enabled_jobs=len(enabled_jobs),
                scheduled_jobs=len(jobs),
                **_background_rhythm_metadata(background),
                recent_sessions=len(sessions),
                recent_tool_runs=len(tool_runs),
                recent_ok_tool_runs=brain_loop_handoff["recent_ok_tool_runs"],
                recent_failed_tool_runs=brain_loop_handoff["recent_failed_tool_runs"],
                recent_approval_held_tool_runs=brain_loop_handoff["recent_approval_held_tool_runs"],
                readable_state_rows=readable_state_rows,
                unreadable_state_rows=unreadable_state_rows,
                unreadable_memory_rows=unreadable_memory_rows,
                unreadable_skill_rows=unreadable_skill_rows,
                unreadable_goal_rows=unreadable_goal_rows,
                unreadable_goal_step_rows=unreadable_goal_step_rows,
                unreadable_task_rows=unreadable_task_rows,
                unreadable_decision_rows=unreadable_decision_rows,
                unreadable_preference_rows=unreadable_preference_rows,
                unreadable_session_rows=unreadable_session_rows,
                unreadable_tool_run_rows=unreadable_tool_run_rows,
                unreadable_approval_rows=unreadable_approval_rows,
                unreadable_job_rows=unreadable_job_rows,
                brain_loop_handoff_ready=brain_loop_handoff["handoff_ready"],
                brain_loop_ready_for_operator=brain_loop_handoff["ready_for_operator"],
                brain_loop_state_changed=brain_loop_handoff["state_changed"],
                brain_loop_changed=brain_loop_handoff["changed"],
                brain_loop_content_in_handoff=brain_loop_handoff["content_in_handoff"],
                brain_loop_start_kind=brain_loop_handoff["start_kind"],
                brain_loop_start_command=brain_loop_handoff["start_command"],
                brain_loop_start_label=brain_loop_handoff["start_label"],
                brain_loop_order=brain_loop_handoff["loop_order"],
                brain_loop_next_commands=brain_loop_handoff["next_commands"],
                brain_loop_next_command_count=brain_loop_handoff["next_command_count"],
                brain_loop_next_safe_commands=brain_loop_handoff["next_safe_commands"],
                brain_loop_next_safe_command_count=brain_loop_handoff["next_safe_command_count"],
                brain_loop_authorizes_execution=brain_loop_handoff["boundaries"]["authorizes_execution"],
                brain_loop_authorizes_completion_claim=brain_loop_handoff["boundaries"]["authorizes_completion_claim"],
                brain_loop_approval_granted=brain_loop_handoff["boundaries"]["approval_granted"],
                brain_loop_boundaries=brain_loop_handoff["boundaries"],
                brain_loop_handoff=brain_loop_handoff,
            ),
        )

    def _export_state_snapshot(_: dict[str, Any]) -> ToolResult:
        memories, unreadable_memory_rows = _safe_row_payloads(store.recent_memories(limit=12), ("category", "title"))
        goals, unreadable_goal_rows = _safe_row_payloads(store.list_goals(limit=20), ("id", "title", "status"))
        tasks, unreadable_task_rows = _safe_row_payloads(store.list_tasks(status="open", limit=12), ("id", "body", "due"))
        decisions, unreadable_decision_rows = _safe_row_payloads(store.list_decisions(status="active", limit=8), ("id", "title", "rationale"))
        people, unreadable_people_rows = _safe_row_payloads(store.list_people(limit=8), ("id", "name", "relation", "last_contact_at"))
        preferences, unreadable_preference_rows = _safe_row_payloads(store.list_preferences(status="active", limit=12), ("category", "key", "value"))
        jobs, unreadable_job_rows = _safe_row_payloads(store.list_jobs(), ("name", "job_type", "enabled", "next_run_at"))
        background = _background_rhythm_state(jobs, unreadable_job_rows=unreadable_job_rows)
        sessions, unreadable_session_rows = _safe_row_payloads(store.list_sessions(limit=8), ("session_id", "messages", "last_at"))
        tool_runs, unreadable_tool_run_rows = _safe_row_payloads(
            store.recent_tool_runs(limit=8),
            ("tool_name", "risk", "ok", "created_at", "metadata"),
        )
        pending_approvals, unreadable_approval_rows = _safe_row_payloads(store.list_pending_approvals(limit=8), ("id", "tool_name", "user_input"))
        approval_handoff = _first_approval_handoff(pending_approvals)
        readable_goal_step_rows = 0
        unreadable_goal_step_rows = 0
        goal_steps_for_source: list[dict[str, Any]] = []

        lines = [
            "# Current Context",
            "",
            f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Active Goals",
        ]
        active_goals = [goal for goal in goals if goal["status"] == "active"]
        if active_goals:
            for goal in active_goals:
                steps, unreadable_steps = _safe_goal_steps(store, goal["id"])
                goal_steps_for_source.extend(steps)
                readable_goal_step_rows += len(steps)
                unreadable_goal_step_rows += unreadable_steps
                open_steps = [step for step in steps if step["status"] != "done"]
                next_step = _short(open_steps[0]["body"]) if open_steps else "define the next step"
                lines.append(f"- #{goal['id']} {_short(goal['title'])}: {next_step}")
        else:
            lines.append("- No active goals.")
        unreadable_state_rows = (
            unreadable_memory_rows
            + unreadable_goal_rows
            + unreadable_goal_step_rows
            + unreadable_task_rows
            + unreadable_decision_rows
            + unreadable_people_rows
            + unreadable_preference_rows
            + unreadable_session_rows
            + unreadable_tool_run_rows
            + unreadable_approval_rows
            + unreadable_job_rows
        )
        if unreadable_state_rows:
            lines.extend(
                [
                    "",
                    "## Local Row Safety",
                    f"- Hidden malformed local state rows: {unreadable_state_rows}",
                ]
            )

        lines.extend(["", "## Recent Memories"])
        if memories:
            for row in memories:
                lines.append(f"- [{_short(row['category'], limit=40)}] {_short(row['title'])}")
        else:
            lines.append("- No memories yet.")

        lines.extend(["", "## Open Tasks"])
        if tasks:
            for row in tasks:
                due = f" due {_short(row['due'], limit=60)}" if row["due"] else ""
                lines.append(f"- #{row['id']} {_short(row['body'])}{due}")
        else:
            lines.append("- No open tasks.")

        lines.extend(["", "## Active Decisions"])
        if decisions:
            for row in decisions:
                rationale = f": {_short(row['rationale'])}" if row["rationale"] else ""
                lines.append(f"- #{row['id']} {_short(row['title'])}{rationale}")
        else:
            lines.append("- No active decisions.")

        lines.extend(["", "## People Context"])
        if people:
            for row in people:
                relation = f" ({_short(row['relation'], limit=80)})" if row["relation"] else ""
                last = f", last contact {_short(row['last_contact_at'], limit=80)}" if row["last_contact_at"] else ""
                lines.append(f"- #{row['id']} {_short(row['name'], limit=120)}{relation}{last}")
        else:
            lines.append("- No people saved.")

        lines.extend(["", "## Active Preferences"])
        if preferences:
            for row in preferences:
                lines.append(f"- [{_short(row['category'], limit=40)}] {_short(row['key'], limit=80)}: {_short(row['value'])}")
        else:
            lines.append("- No active preferences.")

        lines.extend(["", "## Scheduled Jobs"])
        if jobs:
            for row in jobs:
                status = "on" if row["enabled"] else "off"
                lines.append(f"- {_short(row['name'], limit=120)} [{_short(row['job_type'], limit=40)}, {status}] next {_short(row['next_run_at'], limit=80)}")
        else:
            lines.append("- No scheduled jobs.")
        lines.extend(
            [
                "",
                "## Background Rhythm",
                f"- {_background_rhythm_line(background)}",
                f"- State Snapshot jobs: {len(background['state_snapshot_jobs'])} total, {len(background['enabled_state_snapshot_jobs'])} enabled",
                f"- Conversation Compaction jobs: {len(background['conversation_compaction_jobs'])} total, {len(background['enabled_conversation_compaction_jobs'])} enabled",
            ]
        )

        lines.extend(["", "## Recent Sessions"])
        if sessions:
            for row in sessions:
                lines.append(f"- {_short(row['session_id'], limit=80)}: {row['messages']} messages, last {_short(row['last_at'], limit=80)}")
        else:
            lines.append("- No sessions yet.")

        lines.extend(["", "## Recent Tool Runs"])
        if tool_runs:
            for row in tool_runs:
                status = _tool_run_status_label(row)
                lines.append(f"- {_short(row['tool_name'], limit=80)} [{_short(row['risk'], limit=40)}, {status}] {_short(row['created_at'], limit=80)}")
        else:
            lines.append("- No tool runs yet.")
        tool_run_status_counts = _tool_run_status_counts(tool_runs)

        lines.extend(["", "## Pending Approvals"])
        if pending_approvals:
            for row in pending_approvals:
                lines.append(f"- #{row['id']} {_short(row['tool_name'], limit=80)}: {_short(row['user_input'])}")
            lines.append("")
            lines.append("First safe approval handoff:")
            lines.append(f"- readiness: `{approval_handoff['first_readiness_command']}`")
            lines.append(f"- last look: `{approval_handoff['first_last_look_command']}`")
            lines.append(f"- proof: `{approval_handoff['first_proof_command']}`")
            lines.append(f"- decide: `{approval_handoff['first_approve_command']}` or `{approval_handoff['first_dismiss_command']}`")
        else:
            lines.append("- No pending approvals.")

        snapshot_body = "\n".join(lines) + "\n"
        store_identity = store.get_store_identity()
        source_payload = {
            "memories": memories,
            "goals": goals,
            "goal_steps": goal_steps_for_source,
            "tasks": tasks,
            "decisions": decisions,
            "people": people,
            "preferences": preferences,
            "jobs": jobs,
            "sessions": sessions,
            "tool_runs": tool_runs,
            "pending_approvals": pending_approvals,
            "unreadable_state_rows": unreadable_state_rows,
        }
        if effect_authority is not None:
            effect_authority()
        path, content_sha256, source_revision = vault.write_current_context_with_evidence(
            snapshot_body,
            store_identity=store_identity,
            source_payload=source_payload,
        )
        path_display = _safe_vault_path_display(path, vault)
        return ToolResult(
            "export_state_snapshot",
            True,
            f"State snapshot written: {path_display}",
            _safe_metadata(
                path=str(path),
                path_display=path_display,
                writes_files=True,
                writes_database=False,
                writes_memory=False,
                writes_notes=True,
                reads_personal_data=True,
                reads_private_data=True,
                write_atomic=True,
                snapshot_consistency="transaction_fenced",
                content_sha256=content_sha256,
                source_revision=source_revision,
                readable_state_rows=(
                    len(memories)
                    + len(goals)
                    + readable_goal_step_rows
                    + len(tasks)
                    + len(decisions)
                    + len(people)
                    + len(preferences)
                    + len(sessions)
                    + len(tool_runs)
                    + len(pending_approvals)
                    + len(jobs)
                ),
                unreadable_state_rows=unreadable_state_rows,
                unreadable_memory_rows=unreadable_memory_rows,
                unreadable_goal_rows=unreadable_goal_rows,
                unreadable_goal_step_rows=unreadable_goal_step_rows,
                unreadable_task_rows=unreadable_task_rows,
                unreadable_decision_rows=unreadable_decision_rows,
                unreadable_people_rows=unreadable_people_rows,
                unreadable_preference_rows=unreadable_preference_rows,
                unreadable_session_rows=unreadable_session_rows,
                unreadable_tool_run_rows=unreadable_tool_run_rows,
                unreadable_approval_rows=unreadable_approval_rows,
                unreadable_job_rows=unreadable_job_rows,
                scheduled_jobs=len(jobs),
                enabled_jobs=len(background["enabled_jobs"]),
                **_background_rhythm_metadata(background),
                pending_approvals=len(pending_approvals),
                recent_tool_runs=len(tool_runs),
                recent_ok_tool_runs=tool_run_status_counts["ok"],
                recent_failed_tool_runs=tool_run_status_counts["failed"],
                recent_approval_held_tool_runs=tool_run_status_counts["approval_held"],
                **approval_handoff,
            ),
        )

    def export_state_snapshot(args: dict[str, Any]) -> ToolResult:
        if type(args) is not dict or args:
            return ToolResult(
                "export_state_snapshot",
                False,
                "State snapshot does not accept arguments.",
                _safe_metadata(
                    failure_kind="invalid_arguments",
                    argument_keys=sorted(
                        key[:80] if type(key) is str else "<non-string>" for key in args
                    )[:20]
                    if type(args) is dict
                    else [],
                ),
            )
        with store.current_context_publication_fence():
            return _export_state_snapshot(args)

    return brain_loop_report, export_state_snapshot
