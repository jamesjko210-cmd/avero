from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from jarvis_v2.agent.failure_guidance import declare_failure_guidance
from jarvis_v2.agent.types import ToolResult
from jarvis_v2.automations.compaction import COMPACTION_INTERVAL_MINUTES, COMPACTION_JOB_NAME, COMPACTION_JOB_TYPE
from jarvis_v2.automations.scheduler import (
    MORNING_BRIEF_DEFAULT_HHMM,
    MORNING_BRIEF_JOB_NAME,
    MORNING_BRIEF_JOB_TYPE,
    Scheduler,
    job_enabled,
    parse_hhmm,
)
from jarvis_v2.config import JarvisConfig
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import MemoryStore


MIN_JOB_INTERVAL_MINUTES = 1
MAX_JOB_INTERVAL_MINUTES = 43200
MAX_JOB_NAME_CHARS = 120
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
SECRET_VALUE_RE = re.compile(
    r"(?:\bsk_(?:live|test)_[A-Za-z0-9_-]+|\bgh[pousr]_[A-Za-z0-9_-]+|"
    r"\bxox[baprs]-[A-Za-z0-9-]+|\bAIza[A-Za-z0-9_-]{16,}|"
    r"\b\d{6,}:[A-Za-z0-9_-]{20,})",
    re.IGNORECASE,
)
SCHEDULER_INPUT_RECOVERY_ACTION = (
    "Correct the scheduler request, then submit it again through the normal policy."
)
SCHEDULER_NOT_FOUND_RECOVERY_ACTION = (
    "Run `list scheduled jobs`, correct the job name, then submit a new scheduler request."
)
SCHEDULER_READ_RECOVERY_ACTION = (
    "Run `setup check`, correct the local storage issue, then retry the read."
)
SCHEDULER_MUTATION_UNKNOWN_ACTION = (
    "The scheduler mutation outcome is unknown. Run `list scheduled jobs` to verify state; "
    "do not replay the request automatically."
)
SCHEDULER_RUN_NOT_STARTED_ACTION = (
    "Run `list scheduled jobs` to confirm no run is active, then submit a new request "
    "through the normal policy."
)
SCHEDULER_RUN_INSPECTION_ACTION = (
    "Run `list scheduled jobs` and inspect `recent tool runs`; do not replay the job automatically."
)
OPERATOR_LIMIT_RULE = (
    "the operator's explicit stop times, work windows, pause commands, and newer instructions "
    "override scheduled jobs, due runs, and priority goals."
)
SCHEDULER_BOUNDARY = (
    "\n\nScheduler boundary:\n"
    f"- {OPERATOR_LIMIT_RULE}\n"
    "- Scheduled jobs may write local Jarvis notes/state. Configured owner Telegram jobs may send only "
    "to the allowlisted owner chat; all other personal data, external side effects, destructive changes, "
    "computer control, and shell/code execution stay approval-gated."
)
_MISSING_ROW_VALUE = object()


def _safe_text(value: Any, default: str = "") -> str:
    if value is None or value is _MISSING_ROW_VALUE:
        return default
    try:
        return str(value)
    except Exception:
        return default


def _bounded_interval(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        interval = int(value)
    except (TypeError, ValueError):
        interval = default
    return max(MIN_JOB_INTERVAL_MINUTES, min(MAX_JOB_INTERVAL_MINUTES, interval))


def _has_local_path(value: Any) -> bool:
    return bool(LOCAL_PATH_RE.search(_safe_text(value)))


def _display_text(value: Any, default: str = "Daily Brief") -> str:
    text = _safe_text(value, default).strip() or default
    text = " ".join(text.split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    text = SECRET_VALUE_RE.sub("<private-value>", text)
    if len(text) <= MAX_JOB_NAME_CHARS:
        return text
    return text[: max(0, MAX_JOB_NAME_CHARS - 3)].rstrip() + "..."


def _row_value(row: Any, key: str, default: Any = _MISSING_ROW_VALUE) -> Any:
    try:
        return row[key]
    except Exception:
        return default


def _job_row_snapshot(row: Any) -> dict[str, Any] | None:
    values = {
        key: _row_value(row, key)
        for key in ("id", "name", "job_type", "enabled", "next_run_at")
    }
    if any(value is _MISSING_ROW_VALUE for value in values.values()):
        return None
    return {
        "id": _display_text(values["id"], default="?"),
        "name": _display_text(values["name"], default="Unnamed job"),
        "job_type": _display_text(values["job_type"], default="unknown"),
        "enabled": job_enabled(row),
        "next_run_at": _display_text(values["next_run_at"], default="unknown"),
    }


def _job_row_snapshots(rows: list[Any]) -> tuple[list[dict[str, Any]], int]:
    readable: list[dict[str, Any]] = []
    unreadable = 0
    for row in rows:
        snapshot = _job_row_snapshot(row)
        if snapshot is None:
            unreadable += 1
        else:
            readable.append(snapshot)
    return readable, unreadable


def _existing_job_name_sets(rows: list[Any]) -> tuple[set[str], list[str], int]:
    match_names: set[str] = set()
    display_names: set[str] = set()
    unreadable = 0
    for row in rows:
        raw_name = _row_value(row, "name")
        if raw_name is _MISSING_ROW_VALUE:
            unreadable += 1
            continue
        text = _safe_text(raw_name).strip()
        if not text:
            continue
        match_names.add(text.casefold())
        display_names.add(_display_text(text, default="Unnamed job"))
    return match_names, sorted(display_names), unreadable


def _preserved_job_snapshot(row: Any) -> dict[str, Any] | None:
    values = {
        key: _row_value(row, key)
        for key in (
            "id",
            "name",
            "interval_minutes",
            "job_type",
            "enabled",
            "next_run_at",
            "schedule_revision",
            "metadata",
        )
    }
    if any(value is _MISSING_ROW_VALUE for value in values.values()):
        return None
    if type(values["id"]) is not int or values["id"] < 1:
        return None
    if type(values["name"]) is not str or not values["name"]:
        return None
    if (
        type(values["interval_minutes"]) is not int
        or not MIN_JOB_INTERVAL_MINUTES
        <= values["interval_minutes"]
        <= MAX_JOB_INTERVAL_MINUTES
    ):
        return None
    if type(values["job_type"]) is not str or not values["job_type"].strip():
        return None
    if type(values["enabled"]) is not int or values["enabled"] not in {0, 1}:
        return None
    if type(values["schedule_revision"]) is not int or values["schedule_revision"] < 0:
        return None
    if type(values["next_run_at"]) is not str or not values["next_run_at"]:
        return None
    try:
        datetime.fromisoformat(values["next_run_at"])
        metadata = json.loads(values["metadata"] or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if type(metadata) is not dict:
        return None
    return {
        "id": values["id"],
        "interval_minutes": values["interval_minutes"],
        "enabled": values["enabled"] == 1,
    }


def _format_job_snapshot(row: dict[str, Any]) -> str:
    status = "on" if row["enabled"] else "off"
    return f"- #{row['id']} {row['name']} [{row['job_type']}, {status}] next: {row['next_run_at']}"


def _format_scheduled_job_list(rows: list[dict[str, Any]], unreadable_rows: int) -> str:
    lines: list[str] = []
    if rows:
        lines.extend(_format_job_snapshot(row) for row in rows)
    elif unreadable_rows:
        lines.append("No readable scheduled jobs.")
    else:
        lines.append("No scheduled jobs.")
    if unreadable_rows:
        lines.append(f"- {unreadable_rows} unreadable scheduled job row(s) hidden for safety.")
    return "\n".join(lines)


def _job_name(value: Any, default: str = "Daily Brief") -> str:
    return _display_text(value, default)


def _scheduler_refusal_boundaries() -> dict[str, bool]:
    return {
        "read_only": True,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "external_side_effect": False,
        "queues_approval": False,
        "requires_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "controls_computer": False,
        "calls_model": False,
        "executes_tools": False,
        "speaks": False,
        "completes_tasks": False,
        "runs_scheduled_job": False,
        "mutates_scheduled_job": False,
    }


def _scheduler_refusal_handoff(*, source: str, reason: str, job_name: str = "") -> dict[str, Any]:
    safe_job_name = _display_text(job_name, default="") if job_name else ""
    retry_command = f"{source.replace('_', ' ')} {safe_job_name or '<job name>'}"
    return {
        "source": source,
        "ready_for_operator": True,
        "mutation": "scheduler_job_control",
        "reason": reason,
        "refused": True,
        "job_name": safe_job_name,
        "changed": [],
        "retry_command": retry_command,
        "next_commands": [
            "list scheduled jobs",
            retry_command,
            "scheduler context refresh",
        ],
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
        "boundaries": _scheduler_refusal_boundaries(),
    }


def _scheduler_refusal_metadata(*, source: str, reason: str, job_name: str = "") -> dict[str, Any]:
    safe_job_name = _display_text(job_name, default="") if job_name else ""
    metadata = _safe_metadata(
        reason=reason,
        scheduler_refusal_handoff=_scheduler_refusal_handoff(source=source, reason=reason, job_name=job_name),
        scheduler_refusal_handoff_ready=True,
    )
    if safe_job_name:
        metadata["job_name"] = safe_job_name
    return metadata


def _failure_output(message: Any, action: str) -> str:
    public_message = _safe_text(message, "Scheduler request failed.")
    public_message = LOCAL_PATH_RE.sub("<local-path>", public_message)
    public_message = SECRET_VALUE_RE.sub("<private-value>", public_message)
    return _with_scheduler_boundary(f"{public_message}\n\nRecovery: {action}")


def _known_no_change_failure_metadata(
    metadata: dict[str, Any],
    *,
    output: str,
    action: str,
    commands: tuple[str, ...] = (),
) -> dict[str, Any]:
    truth = dict(metadata)
    truth.update(
        {
            "state_changed": False,
            "outcome_known": True,
            "outcome_unknown": False,
            "execution_outcome_unknown": False,
            "side_effect_possible": False,
            "retry_safe": True,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
            "executes_side_effect": False,
            "mutates_scheduled_job": False,
            "runs_scheduled_job": False,
        }
    )
    return declare_failure_guidance(
        truth,
        output=output,
        action=action,
        commands=commands,
    )


def _unknown_mutation_failure_metadata(
    metadata: dict[str, Any],
    *,
    output: str,
    mutation: str,
    commands: tuple[str, ...] = ("list scheduled jobs",),
) -> dict[str, Any]:
    truth = dict(metadata)
    truth.update(
        {
            "state_changed": None,
            "mutation": mutation,
            "mutation_outcome": "unknown",
            "outcome_known": False,
            "outcome_unknown": True,
            "execution_outcome_unknown": True,
            "side_effect_possible": True,
            "retry_safe": False,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
            "mutates_scheduled_job": True,
        }
    )
    return declare_failure_guidance(
        truth,
        output=output,
        action=SCHEDULER_MUTATION_UNKNOWN_ACTION,
        commands=commands,
    )


def _run_failure_metadata(
    metadata: dict[str, Any],
    *,
    output: str,
    outcome_unknown: bool,
    state_changed: bool | None = None,
) -> dict[str, Any]:
    truth = dict(metadata)
    truth.update(
        {
            "state_changed": state_changed,
            "outcome_known": not outcome_unknown,
            "outcome_unknown": outcome_unknown,
            "execution_outcome_unknown": outcome_unknown,
            "side_effect_possible": True,
            "retry_safe": False,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
        }
    )
    return declare_failure_guidance(
        truth,
        output=output,
        action=SCHEDULER_RUN_INSPECTION_ACTION,
        commands=("list scheduled jobs", "recent tool runs"),
    )


def _run_not_started_failure_metadata(
    metadata: dict[str, Any],
    *,
    output: str,
    state_changed: bool,
) -> dict[str, Any]:
    truth = dict(metadata)
    truth.update(
        {
            "state_changed": state_changed,
            "outcome_known": True,
            "outcome_unknown": False,
            "execution_outcome_unknown": False,
            "side_effect_possible": False,
            "retry_safe": True,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
            "executes_side_effect": False,
            "runs_scheduled_job": False,
        }
    )
    return declare_failure_guidance(
        truth,
        output=output,
        action=SCHEDULER_RUN_NOT_STARTED_ACTION,
        commands=("list scheduled jobs",),
    )


def _read_failure_result(source: str) -> ToolResult:
    output = _failure_output(
        "Scheduler storage could not be read safely.",
        SCHEDULER_READ_RECOVERY_ACTION,
    )
    metadata = _known_no_change_failure_metadata(
        _safe_metadata(reason="scheduler_storage_read_failed"),
        output=output,
        action=SCHEDULER_READ_RECOVERY_ACTION,
        commands=("setup check",),
    )
    return ToolResult(source, False, output, metadata)


def _mutation_unknown_result(source: str, mutation: str) -> ToolResult:
    output = _failure_output(
        "Scheduler storage failed while applying the request.",
        SCHEDULER_MUTATION_UNKNOWN_ACTION,
    )
    metadata = _unknown_mutation_failure_metadata(
        _safe_metadata(
            reason="scheduler_mutation_outcome_unknown",
            writes_files=True,
            writes_database=True,
        ),
        output=output,
        mutation=mutation,
    )
    return ToolResult(source, False, output, metadata)


def _missing_job_result(source: str, job_name: str, message: str) -> ToolResult:
    output = _failure_output(message, SCHEDULER_NOT_FOUND_RECOVERY_ACTION)
    metadata = _known_no_change_failure_metadata(
        _safe_metadata(reason="job_not_found", job_name=_display_text(job_name)),
        output=output,
        action=SCHEDULER_NOT_FOUND_RECOVERY_ACTION,
        commands=("list scheduled jobs",),
    )
    return ToolResult(source, False, output, metadata)


def _required_job_name(value: Any, *, source: str) -> tuple[str | None, ToolResult | None]:
    raw = _safe_text(value).strip()
    if not raw:
        output = _failure_output("Job name is required.", SCHEDULER_INPUT_RECOVERY_ACTION)
        metadata = _known_no_change_failure_metadata(
            _scheduler_refusal_metadata(source=source, reason="missing_job_name"),
            output=output,
            action=SCHEDULER_INPUT_RECOVERY_ACTION,
        )
        return None, ToolResult(source, False, output, metadata)
    if _has_local_path(raw):
        output = _failure_output(
            "Job name cannot be a local file path.",
            SCHEDULER_INPUT_RECOVERY_ACTION,
        )
        metadata = _known_no_change_failure_metadata(
            _scheduler_refusal_metadata(source=source, reason="invalid_job_name", job_name=raw),
            output=output,
            action=SCHEDULER_INPUT_RECOVERY_ACTION,
        )
        return None, ToolResult(
            source,
            False,
            output,
            metadata,
        )
    return _job_name(raw), None


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
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "speaks": False,
        "completes_tasks": False,
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
    }
    metadata.update(extra)
    return metadata


def _with_scheduler_boundary(body: str) -> str:
    return f"{body}{SCHEDULER_BOUNDARY}"


def _scheduler_run_ok(output: str) -> bool:
    text = str(output or "").casefold()
    return not any(
        marker in text
        for marker in (
            "job failed:",
            "delivery failed:",
            "not sent:",
            "outcome is unknown",
            "outcome unknown",
            "run was not finalized",
            "skipped stale claim",
            "retry limit",
            "automatic retries stopped",
            "receipt not recorded",
        )
    )


def _scheduler_control_changed(output: str) -> bool:
    return not str(output or "").casefold().startswith("no job found named ")


def _scheduler_run_now_ok(output: str) -> bool:
    text = str(output or "").casefold()
    return _scheduler_run_ok(output) and not any(
        marker in text
        for marker in (
            "no job found named ",
            "is already running; run skipped.",
            "unknown job type:",
        )
    )


def _scheduler_run_started(output: str) -> bool:
    text = str(output or "").casefold()
    return text.startswith("ran ") and "unknown job type:" not in text


def _scheduler_run_changed_state(output: str) -> bool:
    text = str(output or "").casefold()
    return text.startswith("ran ")


def _scheduler_attempted_owner_send(output: str) -> bool:
    text = str(output or "").casefold()
    return any(
        marker in text
        for marker in (
            "accepted by telegram api",
            "delivery failed:",
            "not sent:",
            "outcome is unknown",
            "outcome unknown",
            "local receipt is uncertain",
        )
    ) and "already accepted by telegram api today" not in text


def make_scheduler_tools(store: MemoryStore, vault: ObsidianVault, config: JarvisConfig | None = None):
    scheduler = Scheduler(store, vault, config)

    def scheduler_context_refresh_packet(_: dict[str, Any]) -> ToolResult:
        try:
            raw_jobs = store.list_jobs()
        except Exception:
            return _read_failure_result("scheduler_context_refresh_packet")
        jobs, unreadable_job_rows = _job_row_snapshots(raw_jobs)
        enabled_jobs = [row for row in jobs if row["enabled"]]
        state_jobs = [row for row in jobs if row["job_type"] == "state_snapshot"]
        enabled_state_jobs = [row for row in state_jobs if row["enabled"]]
        disabled_state_jobs = [row for row in state_jobs if not row["enabled"]]
        compaction_jobs = [
            row
            for row in jobs
            if row["job_type"] == COMPACTION_JOB_TYPE or row["name"].casefold() == COMPACTION_JOB_NAME.casefold()
        ]
        enabled_compaction_jobs = [row for row in compaction_jobs if row["enabled"]]
        disabled_compaction_jobs = [row for row in compaction_jobs if not row["enabled"]]
        mission_control = vault.root_path / "Automations" / "Mission Control.md"
        current_context = vault.root_path / "Memory Tree" / "Current Context.md"
        try:
            mission_exists = mission_control.exists()
            context_exists = current_context.exists()
        except Exception:
            return _read_failure_result("scheduler_context_refresh_packet")
        missing: list[str] = []
        if not jobs:
            missing.append("no scheduled jobs")
        if not enabled_state_jobs:
            missing.append("enabled State Snapshot job")
        if not enabled_compaction_jobs:
            missing.append("enabled Conversation Compaction job")
        if not context_exists:
            missing.append("Current Context note")
        if not mission_exists:
            missing.append("Mission Control note")
        if unreadable_job_rows:
            missing.append("readable scheduled job rows")

        if unreadable_job_rows:
            refresh_state = "REFRESH_SCHEDULED_JOBS_UNREADABLE"
            next_command = "list scheduled jobs"
        elif not jobs:
            refresh_state = "REFRESH_NOT_SCHEDULED"
            next_command = "schedule assistant basics"
        elif not enabled_state_jobs:
            refresh_state = "REFRESH_STATE_SNAPSHOT_DISABLED"
            next_command = "resume job State Snapshot" if state_jobs else "schedule assistant basics"
        elif not enabled_compaction_jobs:
            refresh_state = (
                "REFRESH_CONVERSATION_COMPACTION_DISABLED"
                if compaction_jobs
                else "REFRESH_CONVERSATION_COMPACTION_NOT_SCHEDULED"
            )
            next_command = f"resume job {COMPACTION_JOB_NAME}" if compaction_jobs else "schedule assistant basics"
        elif not context_exists or not mission_exists:
            refresh_state = "REFRESH_READY_TO_RUN_LOCAL"
            next_command = "run job State Snapshot now"
        else:
            refresh_state = "REFRESH_READY"
            next_command = "list scheduled jobs"

        lines = [
            "Jarvis scheduler context refresh packet:",
            "This is read-only. It checks whether background continuity can refresh Current Context and Mission Control without creating jobs, running jobs, writing notes, reading personal data, controlling the computer, calling external services, or queuing approvals.",
            "",
            "Continuity state:",
            f"- refresh state: {refresh_state}",
            f"- next safe command: `{next_command}`",
            f"- scheduled jobs: {len(raw_jobs)}",
            f"- readable scheduled jobs: {len(jobs)}",
            f"- unreadable scheduled job rows: {unreadable_job_rows}",
            f"- enabled jobs: {len(enabled_jobs)}",
            f"- state snapshot jobs: {len(state_jobs)}",
            f"- enabled state snapshot jobs: {len(enabled_state_jobs)}",
            f"- disabled state snapshot jobs: {len(disabled_state_jobs)}",
            f"- conversation compaction jobs: {len(compaction_jobs)}",
            f"- enabled conversation compaction jobs: {len(enabled_compaction_jobs)}",
            f"- disabled conversation compaction jobs: {len(disabled_compaction_jobs)}",
            f"- Current Context note: {'present' if context_exists else 'missing'}",
            f"- Mission Control note: {'present' if mission_exists else 'missing'}",
            f"- missing readiness: {', '.join(missing) if missing else 'none'}",
            "",
            "Scheduled job view:",
        ]
        if jobs:
            for row in jobs[:12]:
                lines.append(_format_job_snapshot(row))
            if unreadable_job_rows:
                lines.append(f"- {unreadable_job_rows} unreadable scheduled job row(s) hidden for safety.")
        elif unreadable_job_rows:
            lines.append(f"- {unreadable_job_rows} unreadable scheduled job row(s) hidden for safety.")
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "Safe refresh path:",
                "- Use `schedule assistant basics` only when the operator wants the default local-safe background jobs created.",
                "- Use `resume job State Snapshot` when the continuity job already exists but is paused.",
                f"- Use `resume job {COMPACTION_JOB_NAME}` when the Memory Trees compaction job exists but is paused.",
                "- Use `run job State Snapshot now` only when a local note refresh is desired now.",
                "- Use `list scheduled jobs` to inspect existing jobs without changing state.",
                "- Do not auto-run shell/code, computer control, personal integrations, destructive actions, or external side effects from scheduled context refresh.",
            ]
        )
        return ToolResult(
            "scheduler_context_refresh_packet",
            True,
            _with_scheduler_boundary("\n".join(lines)),
            _safe_metadata(
                refresh_state=refresh_state,
                next_command=next_command,
                scheduled_jobs=len(raw_jobs),
                readable_scheduled_jobs=len(jobs),
                unreadable_scheduled_job_rows=unreadable_job_rows,
                enabled_jobs=len(enabled_jobs),
                state_snapshot_jobs=len(state_jobs),
                enabled_state_snapshot_jobs=len(enabled_state_jobs),
                disabled_state_snapshot_jobs=len(disabled_state_jobs),
                conversation_compaction_jobs=len(compaction_jobs),
                enabled_conversation_compaction_jobs=len(enabled_compaction_jobs),
                disabled_conversation_compaction_jobs=len(disabled_compaction_jobs),
                current_context_exists=context_exists,
                mission_control_exists=mission_exists,
                missing_readiness=missing,
                missing_readiness_count=len(missing),
            ),
        )

    def schedule_daily_brief(args: dict[str, Any]) -> ToolResult:
        interval = _bounded_interval(args.get("interval_minutes"), 1440)
        try:
            job_id = scheduler.schedule_job("Daily Brief", interval, "daily_brief")
        except Exception:
            return _mutation_unknown_result("schedule_daily_brief", "schedule_daily_brief")
        return ToolResult(
            "schedule_daily_brief",
            True,
            _with_scheduler_boundary(f"Scheduled Daily Brief job #{job_id} every {interval} minutes."),
            _safe_metadata(job_id=job_id, interval_minutes=interval, writes_files=True, writes_database=True),
        )

    def schedule_morning_brief(args: dict[str, Any]) -> ToolResult:
        raw_time = args.get("time") or args.get("hhmm") or args.get("at") or MORNING_BRIEF_DEFAULT_HHMM
        try:
            parsed_time = parse_hhmm(raw_time)
        except Exception:
            parsed_time = None
        if parsed_time is None:
            safe_time = _display_text(raw_time, default="")
            output = _failure_output(
                "Morning brief time must look like 07:30, 7:30am, or 19:45.",
                SCHEDULER_INPUT_RECOVERY_ACTION,
            )
            metadata = _known_no_change_failure_metadata(
                _safe_metadata(
                    reason="invalid_time",
                    requested_time=safe_time,
                    writes_files=False,
                    writes_database=False,
                ),
                output=output,
                action=SCHEDULER_INPUT_RECOVERY_ACTION,
            )
            return ToolResult(
                "schedule_morning_brief",
                False,
                output,
                metadata,
            )
        try:
            scheduled = scheduler.schedule_morning_brief(str(raw_time))
        except Exception:
            return _mutation_unknown_result("schedule_morning_brief", "schedule_morning_brief")
        if not scheduled:
            return _mutation_unknown_result("schedule_morning_brief", "schedule_morning_brief")
        job_id, hhmm = scheduled
        return ToolResult(
            "schedule_morning_brief",
            True,
            _with_scheduler_boundary(
                f"Scheduled {MORNING_BRIEF_JOB_NAME} job #{job_id} at {hhmm} daily via Telegram."
            ),
            _safe_metadata(
                job_id=job_id,
                job_name=MORNING_BRIEF_JOB_NAME,
                job_type=MORNING_BRIEF_JOB_TYPE,
                time_hhmm=hhmm,
                writes_files=True,
                writes_database=True,
            ),
        )

    def schedule_goal_nudge(args: dict[str, Any]) -> ToolResult:
        interval = _bounded_interval(args.get("interval_minutes"), 1440)
        try:
            job_id = scheduler.schedule_job("Goal Nudge", interval, "goal_nudge")
        except Exception:
            return _mutation_unknown_result("schedule_goal_nudge", "schedule_goal_nudge")
        return ToolResult(
            "schedule_goal_nudge",
            True,
            _with_scheduler_boundary(f"Scheduled Goal Nudge job #{job_id} every {interval} minutes."),
            _safe_metadata(job_id=job_id, interval_minutes=interval, writes_files=True, writes_database=True),
        )

    def schedule_weekly_review(args: dict[str, Any]) -> ToolResult:
        interval = _bounded_interval(args.get("interval_minutes"), 10080)
        try:
            job_id = scheduler.schedule_job("Weekly Review", interval, "weekly_review")
        except Exception:
            return _mutation_unknown_result("schedule_weekly_review", "schedule_weekly_review")
        return ToolResult(
            "schedule_weekly_review",
            True,
            _with_scheduler_boundary(f"Scheduled Weekly Review job #{job_id} every {interval} minutes."),
            _safe_metadata(job_id=job_id, interval_minutes=interval, writes_files=True, writes_database=True),
        )

    def schedule_inbox_ingest(args: dict[str, Any]) -> ToolResult:
        interval = _bounded_interval(args.get("interval_minutes"), 240)
        try:
            job_id = scheduler.schedule_job("Inbox Ingest", interval, "inbox_ingest")
        except Exception:
            return _mutation_unknown_result("schedule_inbox_ingest", "schedule_inbox_ingest")
        return ToolResult(
            "schedule_inbox_ingest",
            True,
            _with_scheduler_boundary(f"Scheduled Inbox Ingest job #{job_id} every {interval} minutes."),
            _safe_metadata(job_id=job_id, interval_minutes=interval, writes_files=True, writes_database=True),
        )

    def schedule_file_digest(args: dict[str, Any]) -> ToolResult:
        interval = _bounded_interval(args.get("interval_minutes"), 720)
        try:
            job_id = scheduler.schedule_job("Recent File Digest", interval, "recent_file_digest")
        except Exception:
            return _mutation_unknown_result("schedule_file_digest", "schedule_file_digest")
        return ToolResult(
            "schedule_file_digest",
            True,
            _with_scheduler_boundary(f"Scheduled Recent File Digest job #{job_id} every {interval} minutes."),
            _safe_metadata(job_id=job_id, interval_minutes=interval, writes_files=True, writes_database=True),
        )

    def schedule_assistant_basics(_: dict[str, Any]) -> ToolResult:
        jobs = [
            ("Daily Brief", 1440, "daily_brief"),
            ("Inbox Ingest", 240, "inbox_ingest"),
            ("Recent File Digest", 720, "recent_file_digest"),
            ("Goal Nudge", 1440, "goal_nudge"),
            ("Weekly Review", 10080, "weekly_review"),
            ("State Snapshot", 360, "state_snapshot"),
            (COMPACTION_JOB_NAME, COMPACTION_INTERVAL_MINUTES, COMPACTION_JOB_TYPE),
        ]
        try:
            existing_rows = store.list_jobs()
        except Exception:
            return _read_failure_result("schedule_assistant_basics")
        _, existing_job_names_display, unreadable_job_rows = _existing_job_name_sets(existing_rows)
        lines = []
        created = 0
        preserved = 0
        paused_preserved = 0
        unreadable_preserved = 0
        for name, interval, job_type in jobs:
            try:
                row, was_created = scheduler.schedule_job_if_missing(name, interval, job_type)
            except Exception:
                return _mutation_unknown_result("schedule_assistant_basics", "schedule_assistant_basics")
            try:
                job_id = int(row["id"])
            except (KeyError, IndexError, TypeError, ValueError):
                return _mutation_unknown_result("schedule_assistant_basics", "schedule_assistant_basics")
            if was_created:
                created += 1
                status = "created"
                display_interval = interval
            else:
                preserved += 1
                snapshot = _preserved_job_snapshot(row)
                if snapshot is None:
                    unreadable_preserved += 1
                    status = "existing state unreadable; preserved; inspect list scheduled jobs"
                    lines.append(f"- #{job_id} {name} ({status})")
                    continue
                display_interval = int(snapshot["interval_minutes"])
                if snapshot["enabled"]:
                    status = "existing; preserved"
                else:
                    paused_preserved += 1
                    status = f"paused; preserved; use resume job {name}"
            lines.append(f"- #{job_id} {name} every {display_interval} minutes ({status})")
        if unreadable_job_rows:
            lines.append(
                f"- {unreadable_job_rows} unreadable scheduled job row(s) ignored while "
                "creating missing defaults."
            )
        return ToolResult(
            "schedule_assistant_basics",
            True,
            _with_scheduler_boundary("Scheduled assistant basics:\n" + "\n".join(lines)),
            _safe_metadata(
                count=len(jobs),
                created_jobs=created,
                refreshed_jobs=0,
                preserved_jobs=preserved,
                paused_preserved_jobs=paused_preserved,
                unreadable_preserved_jobs=unreadable_preserved,
                existing_job_names=existing_job_names_display,
                unreadable_scheduled_job_rows=unreadable_job_rows,
                idempotent_by_job_name=True,
                writes_files=False,
                writes_database=created > 0,
            ),
        )

    def list_scheduled_jobs(_: dict[str, Any]) -> ToolResult:
        try:
            raw_jobs = store.list_jobs()
        except Exception:
            return _read_failure_result("list_scheduled_jobs")
        jobs, unreadable_job_rows = _job_row_snapshots(raw_jobs)
        try:
            delivery_snapshot = store.scheduled_delivery_operational_snapshot()
            delivery_counts = delivery_snapshot["state_counts"]
            delivery_total = delivery_snapshot["chunk_receipts"]
            delivery_summary = (
                "\n\nScheduled Telegram delivery ledger:\n"
                f"- occurrences: {delivery_snapshot['occurrences']}\n"
                f"- active occurrences: {delivery_snapshot['active_occurrences']}\n"
                f"- total chunk receipts: {delivery_total}\n"
                f"- prepared: {delivery_counts.get('prepared', 0)}\n"
                f"- claimed before network I/O: {delivery_counts.get('claimed', 0)}\n"
                f"- sending with ambiguous crash risk: {delivery_counts.get('sending', 0)}\n"
                f"- retryable rejection: {delivery_counts.get('rejected', 0)}\n"
                f"- accepted by API: {delivery_counts.get('accepted', 0)}\n"
                f"- outcome-unknown occurrences (never auto-resent): {delivery_snapshot['uncertain_occurrences']}\n"
                f"- terminal-failure occurrences: {delivery_snapshot['failed_occurrences']}\n"
                f"- content-bearing retry chunks: {delivery_snapshot['content_bearing_chunks']}\n"
                f"- orphaned historical chunks: {delivery_snapshot['orphaned_chunks']}\n"
                f"- oldest active receipt: {delivery_snapshot['oldest_active_at'] or 'none'}\n"
                f"- counts saturated: {'yes' if delivery_snapshot['counts_saturated'] else 'no'}"
            )
        except Exception:
            return _read_failure_result("list_scheduled_jobs")
        return ToolResult(
            "list_scheduled_jobs",
            True,
            _with_scheduler_boundary(_format_scheduled_job_list(jobs, unreadable_job_rows) + delivery_summary),
            _safe_metadata(
                scheduled_jobs=len(raw_jobs),
                readable_scheduled_jobs=len(jobs),
                unreadable_scheduled_job_rows=unreadable_job_rows,
                scheduled_delivery_receipts=delivery_total,
                scheduled_delivery_state_counts=delivery_counts,
                scheduled_delivery_occurrences=delivery_snapshot["occurrences"],
                scheduled_delivery_active_occurrences=delivery_snapshot["active_occurrences"],
                scheduled_delivery_uncertain_occurrences=delivery_snapshot["uncertain_occurrences"],
                scheduled_delivery_failed_occurrences=delivery_snapshot["failed_occurrences"],
                scheduled_delivery_content_bearing_chunks=delivery_snapshot["content_bearing_chunks"],
                scheduled_delivery_orphaned_chunks=delivery_snapshot["orphaned_chunks"],
                scheduled_delivery_oldest_active_at=delivery_snapshot["oldest_active_at"],
                scheduled_delivery_counts_saturated=delivery_snapshot["counts_saturated"],
            ),
        )

    def run_due_jobs(_: dict[str, Any]) -> ToolResult:
        try:
            output = scheduler.run_due_jobs()
        except Exception:
            output = _failure_output(
                "The scheduled-run batch ended without a trustworthy final outcome.",
                SCHEDULER_RUN_INSPECTION_ACTION,
            )
            metadata = _run_failure_metadata(
                _safe_metadata(
                    reason="scheduled_run_outcome_unknown",
                    writes_files=True,
                    writes_database=True,
                    writes_notes=True,
                    reads_private_data=True,
                    reads_personal_data=True,
                    external_side_effect=True,
                    may_send_owner_telegram=True,
                    owner_telegram_only=True,
                    runs_scheduled_job=True,
                ),
                output=output,
                outcome_unknown=True,
            )
            return ToolResult("run_due_jobs", False, output, metadata)
        attempted_owner_send = _scheduler_attempted_owner_send(output)
        ok = _scheduler_run_ok(output)
        public_output = _with_scheduler_boundary(
            LOCAL_PATH_RE.sub("<local-path>", SECRET_VALUE_RE.sub("<private-value>", str(output)))
        )
        metadata = _safe_metadata(
            writes_files=True,
            writes_database=True,
            writes_notes=True,
            reads_private_data=True,
            reads_personal_data=True,
            external_side_effect=attempted_owner_send,
            may_send_owner_telegram=True,
            owner_telegram_only=True,
            runs_scheduled_job=True,
        )
        if not ok:
            public_output = _failure_output(output, SCHEDULER_RUN_INSPECTION_ACTION)
            lowered = str(output or "").casefold()
            outcome_unknown = attempted_owner_send or any(
                marker in lowered
                for marker in (
                    "outcome is unknown",
                    "outcome unknown",
                    "receipt is uncertain",
                    "receipt not recorded",
                    "local receipt is uncertain",
                )
            )
            metadata = _run_failure_metadata(
                metadata,
                output=public_output,
                outcome_unknown=outcome_unknown,
                state_changed=None,
            )
        return ToolResult("run_due_jobs", ok, public_output, metadata)

    def pause_job(args: dict[str, Any]) -> ToolResult:
        name, error = _required_job_name(args.get("name"), source="pause_job")
        if error:
            return ToolResult("pause_job", False, error.output, error.metadata)
        try:
            output = scheduler.pause_job(name)
        except Exception:
            return _mutation_unknown_result("pause_job", "pause_job")
        changed = _scheduler_control_changed(output)
        if not changed:
            return _missing_job_result("pause_job", name, output)
        return ToolResult(
            "pause_job",
            changed,
            _with_scheduler_boundary(output),
            _safe_metadata(job_name=name, writes_files=changed, writes_database=changed),
        )

    def resume_job(args: dict[str, Any]) -> ToolResult:
        name, error = _required_job_name(args.get("name"), source="resume_job")
        if error:
            return ToolResult("resume_job", False, error.output, error.metadata)
        try:
            output = scheduler.resume_job(name)
        except Exception:
            return _mutation_unknown_result("resume_job", "resume_job")
        changed = _scheduler_control_changed(output)
        if not changed:
            return _missing_job_result("resume_job", name, output)
        return ToolResult(
            "resume_job",
            changed,
            _with_scheduler_boundary(output),
            _safe_metadata(job_name=name, writes_files=changed, writes_database=changed),
        )

    def delete_job(args: dict[str, Any]) -> ToolResult:
        name, error = _required_job_name(args.get("name"), source="delete_job")
        if error:
            return ToolResult("delete_job", False, error.output, error.metadata)
        try:
            output = scheduler.delete_job(name)
        except Exception:
            return _mutation_unknown_result("delete_job", "delete_job")
        changed = _scheduler_control_changed(output)
        if not changed:
            return _missing_job_result("delete_job", name, output)
        return ToolResult(
            "delete_job",
            changed,
            _with_scheduler_boundary(output),
            _safe_metadata(job_name=name, writes_files=changed, writes_database=changed),
        )

    def run_job_now(args: dict[str, Any]) -> ToolResult:
        name, error = _required_job_name(args.get("name"), source="run_job_now")
        if error:
            return ToolResult("run_job_now", False, error.output, error.metadata)
        try:
            matching_jobs = [
                row
                for row in scheduler.store.list_jobs()
                if str(row["name"]).casefold() == name.casefold()
            ]
        except Exception:
            return _read_failure_result("run_job_now")
        state_snapshot = any(str(row["job_type"]) == "state_snapshot" for row in matching_jobs)
        try:
            output = scheduler.run_job_now(name)
        except Exception:
            output = _failure_output(
                "The requested scheduled run ended without a trustworthy final outcome.",
                SCHEDULER_RUN_INSPECTION_ACTION,
            )
            metadata = _run_failure_metadata(
                _safe_metadata(
                    job_name=name,
                    reason="scheduled_run_outcome_unknown",
                    writes_files=True,
                    writes_database=True,
                    writes_notes=True,
                    reads_personal_data=True,
                    reads_private_data=True,
                    external_side_effect=True,
                    may_send_owner_telegram=True,
                    owner_telegram_only=True,
                    runs_scheduled_job=True,
                ),
                output=output,
                outcome_unknown=True,
            )
            return ToolResult("run_job_now", False, output, metadata)
        morning_brief = name.casefold() == MORNING_BRIEF_JOB_NAME.casefold()
        run_started = _scheduler_run_started(output)
        changed_state = _scheduler_run_changed_state(output)
        attempted_owner_send = run_started and morning_brief and _scheduler_attempted_owner_send(output)
        ok = _scheduler_run_now_ok(output)
        metadata = _safe_metadata(
            job_name=name,
            writes_files=run_started,
            writes_database=changed_state,
            writes_notes=run_started,
            reads_personal_data=run_started and (morning_brief or state_snapshot),
            reads_private_data=run_started and (morning_brief or state_snapshot),
            external_side_effect=attempted_owner_send,
            may_send_owner_telegram=morning_brief,
            owner_telegram_only=morning_brief,
            runs_scheduled_job=run_started,
        )
        public_output = _with_scheduler_boundary(
            LOCAL_PATH_RE.sub("<local-path>", SECRET_VALUE_RE.sub("<private-value>", str(output)))
        )
        if not ok:
            lowered = str(output or "").casefold()
            if lowered.startswith("no job found named "):
                return _missing_job_result("run_job_now", name, output)
            if "is already running; run skipped." in lowered:
                public_output = _failure_output(output, SCHEDULER_RUN_NOT_STARTED_ACTION)
                metadata = _run_not_started_failure_metadata(
                    metadata,
                    output=public_output,
                    state_changed=False,
                )
            elif "skipped stale claim" in lowered and "not started" in lowered:
                public_output = _failure_output(output, SCHEDULER_RUN_NOT_STARTED_ACTION)
                metadata = _run_not_started_failure_metadata(
                    metadata,
                    output=public_output,
                    state_changed=True,
                )
            else:
                public_output = _failure_output(output, SCHEDULER_RUN_INSPECTION_ACTION)
                outcome_unknown = attempted_owner_send or any(
                    marker in lowered
                    for marker in (
                        "outcome is unknown",
                        "outcome unknown",
                        "receipt is uncertain",
                        "receipt not recorded",
                        "local receipt is uncertain",
                    )
                )
                metadata = _run_failure_metadata(
                    metadata,
                    output=public_output,
                    outcome_unknown=outcome_unknown,
                    state_changed=None if outcome_unknown else changed_state,
                )
        return ToolResult(
            "run_job_now",
            ok,
            public_output,
            metadata,
        )

    return (
        scheduler_context_refresh_packet,
        schedule_daily_brief,
        schedule_morning_brief,
        schedule_goal_nudge,
        schedule_weekly_review,
        schedule_inbox_ingest,
        schedule_file_digest,
        schedule_assistant_basics,
        list_scheduled_jobs,
        run_due_jobs,
        pause_job,
        resume_job,
        delete_job,
        run_job_now,
    )
