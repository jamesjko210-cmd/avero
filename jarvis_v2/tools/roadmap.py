from __future__ import annotations

import json
import re
from typing import Any, Callable

from jarvis_v2.agent.types import ToolResult
from jarvis_v2.automations.compaction import COMPACTION_JOB_NAME, COMPACTION_JOB_TYPE
from jarvis_v2.memory.store import MemoryStore


ROADMAP_PHASES = [
    {
        "name": "1. Keep the brain inspectable",
        "goal": "Make Jarvis explain its architecture, capabilities, readiness, and current work state before taking action.",
        "commands": "`architecture map`, `capability map`, `readiness report`, `focus brief`, `return brief`",
        "safety": "read-only",
    },
    {
        "name": "2. Strengthen continuity",
        "goal": "Improve Current Context, Mission Control, return briefs, focus briefs, and scheduled reviews so Jarvis resumes cleanly after breaks.",
        "commands": "`schedule assistant basics`, `mission control`, `safe next actions`, `weekly review`",
        "safety": "local-safe Obsidian/SQLite writes",
    },
    {
        "name": "3. Upgrade planning without bypassing approvals",
        "goal": "Use stronger model routing for planning, chat, summarization, and future vision while keeping ToolRegistry and PermissionPolicy as the execution boundary.",
        "commands": "`autonomy plan: ...`, `safety status`, `recent tool runs`",
        "safety": "read-only planning; risky actions remain approval-gated",
    },
    {
        "name": "4. Migrate personal integrations",
        "goal": "Bring calendar, email, messages, reminders, and browser workflows from old Jarvis into V2 as explicit tools with personal-data and side-effect gates.",
        "commands": "`integration status`, future approval-gated personal tools",
        "safety": "personal data and external side effects require approval",
    },
    {
        "name": "5. Improve computer-control loops",
        "goal": "Make observe-act-verify more reliable with richer screen understanding, step-by-step verification, and stop conditions.",
        "commands": "`computer control status`, `autonomy plan: ...`, future vision verification",
        "safety": "screenshots, clicks, typing, and observe-act-verify stay approval-gated",
    },
    {
        "name": "6. Learn safely from use",
        "goal": "Turn repeated workflows into reviewable Markdown skills and memory updates, not autonomous self-modifying code.",
        "commands": "`summarize this session`, `draft skill from this session called ...`, `memory tree summary`",
        "safety": "reviewable local memory writes; code changes stay human-reviewed",
    },
]

RECENT_TOOL_RUN_LIMIT = 8
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")
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


def _safe_metadata(**extra: Any) -> dict[str, Any]:
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
    }
    metadata.update(extra)
    return metadata


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


def _row_int_text(row: Any, key: str = "id") -> str:
    value = _row_value(row, key)
    if isinstance(value, bool) or value is _MISSING_ROW_VALUE or value is None:
        return ""
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return ""
    return str(parsed) if parsed > 0 else ""


def _short(value: Any, limit: int = 100) -> str:
    if value is None:
        return ""
    try:
        text = str(value).strip()
    except Exception:
        return ""
    text = " ".join(text.split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def _metadata_truthy_loose(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on"}
    return False


def _normalized_metadata_token(value: Any) -> str:
    if value is None:
        return ""
    try:
        text = str(value).strip().casefold()
    except Exception:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _display_row_text(row: Any, key: str, default: str, limit: int = 180) -> str:
    return _short(_row_text(row, key, default), limit=limit) or default


def _row_metadata(row: Any) -> dict[str, Any]:
    raw = _row_value(row, "metadata", {})
    if isinstance(raw, dict):
        return raw
    if raw is _MISSING_ROW_VALUE or raw is None or not isinstance(raw, (str, bytes, bytearray)):
        return {}
    try:
        parsed = json.loads(raw or "{}")
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


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


def _recent_run_failure_kind(metadata: dict[str, Any]) -> str:
    if _is_approval_hold_metadata(metadata):
        return "approval_required"
    return _short(metadata.get("failure_kind"), limit=80)


def _readable_tool_run_rows(rows: list[Any]) -> tuple[list[dict[str, Any]], int]:
    readable: list[dict[str, Any]] = []
    unreadable = 0
    for row in rows:
        row_id = _row_int_text(row)
        tool_name = _short(_row_text(row, "tool_name"), limit=80)
        ok_value = _row_value(row, "ok")
        if not row_id or not tool_name or ok_value is _MISSING_ROW_VALUE:
            unreadable += 1
            continue
        approval_id_text = _row_int_text(row, "approval_id")
        metadata = _row_metadata(row)
        readable.append(
            {
                "id": int(row_id),
                "tool_name": tool_name,
                "risk": _short(_row_text(row, "risk", "UNKNOWN"), limit=40) or "UNKNOWN",
                "ok": _row_bool(row, "ok"),
                "approved": _row_bool(row, "approved"),
                "approval_id": int(approval_id_text) if approval_id_text else None,
                "failure_kind": _recent_run_failure_kind(metadata),
                "failure_stage": _short(
                    metadata.get("failure_stage")
                    or metadata.get("stage")
                    or metadata.get("guard_reason")
                    or metadata.get("reason"),
                    limit=80,
                ),
                "metadata": metadata,
                "created_at": _short(_row_text(row, "created_at"), limit=40),
            }
        )
    return readable, unreadable


def _approval_proof_chain_commands(approval_id: int | None) -> list[str]:
    if approval_id is None:
        return ["pending approvals", "recent tool runs"]
    return [
        f"approval readiness {approval_id}",
        f"approval packet {approval_id}",
        f"approval chain proof {approval_id}",
    ]


def _recent_tool_run_state(rows: list[Any]) -> dict[str, Any]:
    readable_runs, unreadable_runs = _readable_tool_run_rows(rows)
    ok_runs: list[dict[str, Any]] = []
    failed_runs: list[dict[str, Any]] = []
    approval_held_runs: list[dict[str, Any]] = []
    for row in readable_runs:
        if row["ok"]:
            ok_runs.append(row)
        elif _is_approval_hold_metadata(row["metadata"]):
            approval_held_runs.append(row)
        else:
            failed_runs.append(row)

    first_approval_held = approval_held_runs[0] if approval_held_runs else None
    first_failed = failed_runs[0] if failed_runs else None
    approval_commands = _approval_proof_chain_commands(
        first_approval_held["approval_id"] if first_approval_held else None
    )
    recovery_command = f"execution recovery packet {first_failed['id']}" if first_failed else ""
    return {
        "tool_runs": rows,
        "readable_runs": readable_runs,
        "unreadable_runs": unreadable_runs,
        "ok_runs": ok_runs,
        "failed_runs": failed_runs,
        "approval_held_runs": approval_held_runs,
        "first_failed": first_failed,
        "first_approval_held": first_approval_held,
        "approval_commands": approval_commands if first_approval_held else [],
        "recovery_command": recovery_command,
    }


def _recent_tool_run_line(recent: dict[str, Any]) -> str:
    total = len(recent["tool_runs"])
    readable = len(recent["readable_runs"])
    return (
        f"{readable} readable / {total} total "
        f"({len(recent['ok_runs'])} ok, {len(recent['failed_runs'])} failed/blocked, "
        f"{len(recent['approval_held_runs'])} approval-held, {int(recent['unreadable_runs'])} unreadable hidden)"
    )


def _recent_tool_run_metadata(recent: dict[str, Any]) -> dict[str, Any]:
    first_failed = recent["first_failed"]
    first_approval_held = recent["first_approval_held"]
    return {
        "recent_tool_runs": len(recent["tool_runs"]),
        "readable_recent_tool_runs": len(recent["readable_runs"]),
        "unreadable_recent_tool_run_rows": int(recent["unreadable_runs"]),
        "recent_ok_tool_runs": len(recent["ok_runs"]),
        "recent_failed_tool_runs": len(recent["failed_runs"]),
        "recent_approval_held_tool_runs": len(recent["approval_held_runs"]),
        "roadmap_recent_failed_run_id": first_failed["id"] if first_failed else None,
        "roadmap_recent_failed_tool_name": first_failed["tool_name"] if first_failed else "",
        "roadmap_recent_recovery_command": recent["recovery_command"],
        "roadmap_recent_approval_held_run_id": first_approval_held["id"] if first_approval_held else None,
        "roadmap_recent_approval_held_tool_name": first_approval_held["tool_name"] if first_approval_held else "",
        "roadmap_recent_approval_held_approval_id": first_approval_held["approval_id"] if first_approval_held else None,
        "roadmap_recent_approval_review_commands": recent["approval_commands"],
    }


def _first_readable_row(rows: list[Any]) -> tuple[str, Any | None]:
    for row in rows:
        row_id = _row_int_text(row)
        if row_id:
            return row_id, row
    return "", None


def _readable_job_rows(jobs: list[Any]) -> tuple[list[Any], int]:
    readable: list[Any] = []
    unreadable = 0
    for job in jobs:
        if _row_text(job, "name") or _row_text(job, "job_type"):
            readable.append(job)
        else:
            unreadable += 1
    return readable, unreadable


def _background_rhythm_state(jobs: list[Any]) -> dict[str, Any]:
    readable_jobs, unreadable_jobs = _readable_job_rows(jobs)
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
        issue = "Scheduled job rows include unreadable data."
        next_command = "list scheduled jobs"
    elif not enabled_state_snapshot_jobs and state_snapshot_jobs:
        ready = False
        issue = "State Snapshot scheduled job is paused."
        next_command = "resume job State Snapshot"
    elif not enabled_state_snapshot_jobs:
        ready = False
        issue = "State Snapshot scheduled job is not configured."
        next_command = "schedule assistant basics"
    elif not enabled_conversation_compaction_jobs and conversation_compaction_jobs:
        ready = False
        issue = f"{COMPACTION_JOB_NAME} scheduled job is paused."
        next_command = f"resume job {COMPACTION_JOB_NAME}"
    elif not enabled_conversation_compaction_jobs:
        ready = False
        issue = f"{COMPACTION_JOB_NAME} scheduled job is not configured."
        next_command = "schedule assistant basics"
    else:
        ready = True
        issue = ""
        next_command = "list scheduled jobs"

    return {
        "ready": ready,
        "issue": issue,
        "next_command": next_command,
        "scheduled_jobs": jobs,
        "readable_jobs": readable_jobs,
        "unreadable_jobs": unreadable_jobs,
        "enabled_jobs": enabled_jobs,
        "state_snapshot_jobs": state_snapshot_jobs,
        "enabled_state_snapshot_jobs": enabled_state_snapshot_jobs,
        "disabled_state_snapshot_jobs": disabled_state_snapshot_jobs,
        "conversation_compaction_jobs": conversation_compaction_jobs,
        "enabled_conversation_compaction_jobs": enabled_conversation_compaction_jobs,
        "disabled_conversation_compaction_jobs": disabled_conversation_compaction_jobs,
    }


def _background_rhythm_line(background: dict[str, Any]) -> str:
    if background["ready"]:
        return f"State Snapshot and {COMPACTION_JOB_NAME} are enabled."
    return f"{background['issue']} Next safe command: `{background['next_command']}`."


def _background_rhythm_metadata(background: dict[str, Any]) -> dict[str, Any]:
    return {
        "background_ready": bool(background["ready"]),
        "background_next_command": str(background["next_command"]),
        "background_issue": str(background["issue"]),
        "readable_scheduled_jobs": len(background["readable_jobs"]),
        "unreadable_scheduled_job_rows": int(background["unreadable_jobs"]),
        "state_snapshot_jobs": len(background["state_snapshot_jobs"]),
        "enabled_state_snapshot_jobs": len(background["enabled_state_snapshot_jobs"]),
        "disabled_state_snapshot_jobs": len(background["disabled_state_snapshot_jobs"]),
        "conversation_compaction_jobs": len(background["conversation_compaction_jobs"]),
        "enabled_conversation_compaction_jobs": len(background["enabled_conversation_compaction_jobs"]),
        "disabled_conversation_compaction_jobs": len(background["disabled_conversation_compaction_jobs"]),
    }


def make_roadmap_tools(store: MemoryStore, list_tools: Callable[[], list[Any]]):
    def roadmap_report(_: dict[str, Any]) -> ToolResult:
        tools = list_tools()
        tool_names = {tool.name for tool in tools}
        pending = store.list_pending_approvals(limit=5)
        jobs = store.list_jobs()
        background = _background_rhythm_state(jobs)
        enabled_jobs = background["enabled_jobs"]
        tasks = store.list_tasks(status="open", limit=5)
        goals = store.list_goals(status="active", limit=5)
        recent_runs = store.recent_tool_runs(limit=RECENT_TOOL_RUN_LIMIT)
        recent = _recent_tool_run_state(recent_runs)

        lines = [
            "Jarvis roadmap report:",
            "",
            "Principle: finish Jarvis as an agent harness, not just an agent. The model is the reasoning core; the harness is the surrounding runtime that makes perception, memory, tools, approvals, state, recovery, and verification reliable.",
            "Build autonomy by making each layer inspectable first, then approval-gated, then more capable.",
            "",
            "Current signals:",
            f"- registered tools: {len(tools)}",
            f"- pending approvals: {len(pending)}",
            f"- enabled scheduled jobs: {len(enabled_jobs)} / {len(jobs)}",
            f"- background rhythm: {_background_rhythm_line(background)}",
            f"- State Snapshot jobs: {len(background['state_snapshot_jobs'])} total, {len(background['enabled_state_snapshot_jobs'])} enabled",
            f"- Conversation Compaction jobs: {len(background['conversation_compaction_jobs'])} total, {len(background['enabled_conversation_compaction_jobs'])} enabled",
            f"- open tasks: {len(tasks)}",
            f"- active goals: {len(goals)}",
            f"- recent tool runs: {_recent_tool_run_line(recent)}",
        ]
        if recent["first_approval_held"]:
            held = recent["first_approval_held"]
            commands = ", ".join(f"`{command}`" for command in recent["approval_commands"])
            lines.append(
                f"- approval review: run #{held['id']} {held['tool_name']} is approval-held. Next safe commands: {commands}."
            )
        if recent["first_failed"]:
            failed = recent["first_failed"]
            detail = failed["failure_stage"] or failed["failure_kind"] or "failure logged"
            lines.append(
                f"- execution recovery: run #{failed['id']} {failed['tool_name']} needs recovery review ({detail}). Next safe command: `{recent['recovery_command']}`."
            )
        lines.extend(["", "Build phases:"])
        for phase in ROADMAP_PHASES:
            lines.extend(
                [
                    f"- {phase['name']}",
                    f"  Goal: {phase['goal']}",
                    f"  Commands: {phase['commands']}",
                    f"  Safety: {phase['safety']}",
                ]
            )

        missing_command_tools = [
            name
            for name in (
                "architecture_map",
                "capability_map",
                "readiness_report",
                "focus_brief",
                "return_brief",
                "autonomy_plan",
                "safety_status",
            )
            if name not in tool_names
        ]

        lines.extend(["", "Recommended next move:"])
        if missing_command_tools:
            lines.append("- Restore missing foundation tools before expanding autonomy: " + ", ".join(missing_command_tools))
        elif recent["first_approval_held"]:
            lines.append(
                "- Review the held approval proof chain before treating that action as a failed execution: "
                + ", ".join(f"`{command}`" for command in recent["approval_commands"])
                + "."
            )
        elif pending:
            lines.append("- Review `pending approvals` before continuing risky work.")
        elif recent["first_failed"]:
            lines.append(f"- Inspect `{recent['recovery_command']}` before retrying or claiming completion.")
        elif not background["ready"]:
            lines.append(
                f"- Restore the background rhythm: {background['issue']} Run `{background['next_command']}`."
            )
        else:
            task_id, task = _first_readable_row(tasks)
            goal_id, goal = _first_readable_row(goals)
            if task is not None:
                lines.append(
                    f"- Use `focus brief` and work task #{task_id}: {_display_row_text(task, 'body', 'untitled task')}"
                )
            elif tasks:
                lines.append("- Use `focus brief`; open tasks include unreadable rows, so choose from the safe next-action handoff.")
            elif goal is not None:
                lines.append(
                    f"- Use `focus brief` and advance goal #{goal_id}: {_display_row_text(goal, 'title', 'untitled goal')}"
                )
            elif goals:
                lines.append("- Use `focus brief`; active goals include unreadable rows, so choose from the safe next-action handoff.")
            else:
                lines.append("- Add one concrete Jarvis task or goal, then run `focus brief`.")

        lines.extend(
            [
                "",
                "Do not skip:",
                "- Treat harness pieces as completion-critical: stable chat loop, tool registry, permission policy, approvals, audit trail, memory/state, scheduler, diagnostics, and recovery.",
                "- Keep shell/code, file writes/deletes, clipboard reads, computer control, reminders, and external actions behind approvals.",
                "- Prefer Markdown skills and Obsidian notes for learning loops instead of autonomous code generation.",
            ]
        )

        return ToolResult(
            "roadmap_report",
            True,
            "\n".join(lines),
            _safe_metadata(
                phases=len(ROADMAP_PHASES),
                tools=len(tools),
                pending_approvals=len(pending),
                enabled_jobs=len(enabled_jobs),
                scheduled_jobs=len(jobs),
                open_tasks=len(tasks),
                active_goals=len(goals),
                **_background_rhythm_metadata(background),
                **_recent_tool_run_metadata(recent),
            ),
        )

    return roadmap_report
