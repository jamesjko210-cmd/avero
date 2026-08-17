from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from jarvis_v2.agent.types import ToolResult
from jarvis_v2.automations.compaction import COMPACTION_JOB_NAME, COMPACTION_JOB_TYPE
from jarvis_v2.automations.jobs import (
    build_daily_brief,
    build_daily_plan,
    build_goal_nudge,
    build_weekly_review,
    build_weekly_review_model_context,
    normalize_daily_plan_target_date,
)
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import MemoryStore


MAX_PROACTIVE_LIMIT = 50
MAX_STALE_DAYS = 365
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)


def _safe_text(value: Any) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", str(value or ""))


def _safe_vault_path_display(path: Path | str | None, vault: ObsidianVault) -> str:
    if path is None:
        return ""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(vault.root_path.resolve()))
    except ValueError:
        return LOCAL_PATH_RE.sub("<local-path>", str(candidate))


def _daily_note_path(vault: ObsidianVault, target_date: str | None = None) -> Path:
    day = target_date or datetime.now().strftime("%Y-%m-%d")
    return vault.root_path / "Daily" / f"{day}.md"


def _dated_reflection_path(vault: ObsidianVault, title: str) -> Path:
    safe_title = re.sub(r"[^A-Za-z0-9._ -]+", "-", title).strip(" .-_") or "Untitled"
    stamp = datetime.now().strftime("%Y-%m-%d")
    return vault.root_path / "Reflections" / f"{stamp} {safe_title}.md"


def _write_note_metadata(vault: ObsidianVault, path: Path, **extra: Any) -> dict[str, Any]:
    path_display = _safe_vault_path_display(path, vault)
    return _safety_metadata(
        path=str(path),
        path_display=path_display,
        writes_files=True,
        writes_notes=True,
        **extra,
    )


def _bounded_int(value: Any, default: int, low: int = 1, high: int = MAX_PROACTIVE_LIMIT) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _job_enabled(job: Any) -> bool:
    try:
        value = job["enabled"]
    except Exception:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    return False


def _safety_metadata(**extra: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "calls_model": False,
        "executes_tools": False,
        "approves_request": False,
        "dismisses_request": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "writes_files": False,
        "edits_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "speaks": False,
        "completes_tasks": False,
    }
    metadata.update(extra)
    return metadata


def _weekly_review_boundaries() -> dict[str, bool]:
    return {
        "calls_model": False,
        "executes_tools": False,
        "approves_request": False,
        "dismisses_request": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "writes_files": False,
        "edits_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "speaks": False,
        "completes_tasks": False,
    }


def make_proactive_tools(store: MemoryStore, vault: ObsidianVault):
    def build_morning_startup(limit: int = 6) -> tuple[str, dict[str, int]]:
        limit = _bounded_int(limit, 6)
        tasks = store.list_tasks(status="open", limit=limit)
        goals = store.list_goals(status="active", limit=limit)
        approvals = store.list_pending_approvals(limit=limit)
        decisions = store.list_decisions(status="active", limit=3)
        preferences = store.list_preferences(status="active", limit=3)
        jobs = store.list_jobs()
        enabled_jobs = [job for job in jobs if _job_enabled(job)]
        state_snapshot_jobs = [job for job in jobs if job["job_type"] == "state_snapshot"]
        enabled_state_snapshot_jobs = [job for job in state_snapshot_jobs if _job_enabled(job)]
        conversation_compaction_jobs = [
            job
            for job in jobs
            if str(job["job_type"]) == COMPACTION_JOB_TYPE
            or str(job["name"]).casefold() == COMPACTION_JOB_NAME.casefold()
        ]
        enabled_conversation_compaction_jobs = [
            job for job in conversation_compaction_jobs if _job_enabled(job)
        ]
        if not state_snapshot_jobs:
            background_next_command = "schedule assistant basics"
            background_ready = False
            background_line = "- Background rhythm: State Snapshot is not scheduled; use `schedule assistant basics`."
        elif not enabled_state_snapshot_jobs:
            background_next_command = "resume job State Snapshot"
            background_ready = False
            background_line = "- Background rhythm: State Snapshot exists but is paused; use `resume job State Snapshot`."
        elif not conversation_compaction_jobs:
            background_next_command = "schedule assistant basics"
            background_ready = False
            background_line = "- Background rhythm: Conversation Compaction is not scheduled; use `schedule assistant basics`."
        elif not enabled_conversation_compaction_jobs:
            background_next_command = f"resume job {COMPACTION_JOB_NAME}"
            background_ready = False
            background_line = f"- Background rhythm: {COMPACTION_JOB_NAME} exists but is paused; use `resume job {COMPACTION_JOB_NAME}`."
        elif enabled_jobs:
            background_next_command = "list scheduled jobs"
            background_ready = True
            background_line = (
                f"- Background rhythm: {len(enabled_jobs)} scheduled job(s) enabled, "
                "including State Snapshot and Conversation Compaction."
            )
        else:
            background_next_command = "schedule assistant basics"
            background_ready = False
            background_line = "- Background rhythm: no scheduled jobs enabled; use `schedule assistant basics`."

        lines = [
            "Jarvis morning startup:",
            f"Generated: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}",
            "",
            "Safety first:",
            "- Start read-only: `handoff brief`, `work queue`, `safety status`, or this startup.",
            "- Do not auto-run shell, file-write, clipboard-read, personal-data, external-side-effect, or computer-control actions without explicit approval.",
        ]

        lines.extend(["", "Blocked or waiting:"])
        if approvals:
            lines.append(f"- {len(approvals)} pending approval(s) need review before blocked risky work can continue.")
            for row in approvals[:3]:
                lines.append(f"  - Approval #{row['id']} {_safe_text(row['tool_name'])}: {_safe_text(row['user_input'])}")
            lines.append("- Use `approval review` or `save approval review` before approving anything.")
        else:
            lines.append("- No pending approvals are blocking risky work right now.")

        lines.extend(["", "Today queue:"])
        if tasks:
            first = tasks[0]
            priority = f" [{_safe_text(first['priority'])}]" if first["priority"] != "normal" else ""
            due = f" due {_safe_text(first['due'])}" if first["due"] else ""
            lines.append(f"- First task: #{first['id']} {_safe_text(first['body'])}{due}{priority}")
            for row in tasks[1:4]:
                priority = f" [{_safe_text(row['priority'])}]" if row["priority"] != "normal" else ""
                due = f" due {_safe_text(row['due'])}" if row["due"] else ""
                lines.append(f"- Next task: #{row['id']} {_safe_text(row['body'])}{due}{priority}")
        else:
            lines.append("- No open tasks. Capture one clear task before doing reactive work.")

        if goals:
            lines.append("")
            lines.append("Goal momentum:")
            for goal in goals[:3]:
                steps = store.list_goal_steps(goal["id"])
                open_steps = [step for step in steps if step["status"] != "done"]
                next_step = _safe_text(open_steps[0]["body"]) if open_steps else "define the next step"
                lines.append(f"- Goal #{goal['id']} {_safe_text(goal['title'])}: {next_step}")

        lines.extend(["", "Operating context:"])
        if decisions:
            lines.append(f"- Decision anchor: #{decisions[0]['id']} {_safe_text(decisions[0]['title'])}")
        if preferences:
            pref = preferences[0]
            lines.append(f"- Apply preference: {_safe_text(pref['key'])} = {_safe_text(pref['value'])}")
        lines.append(background_line)

        lines.extend(["", "Suggested first safe move:"])
        if approvals:
            lines.append("- Run `approval review`, then dismiss stale approvals or approve only the exact trusted request.")
        elif tasks:
            lines.append(f"- Work task #{tasks[0]['id']}: {_safe_text(tasks[0]['body'])}")
        elif goals:
            first_goal = goals[0]
            steps = store.list_goal_steps(first_goal["id"])
            open_steps = [step for step in steps if step["status"] != "done"]
            next_step = _safe_text(open_steps[0]["body"]) if open_steps else "define the next step"
            lines.append(f"- Advance {_safe_text(first_goal['title'])}: {next_step}")
        elif not background_ready:
            lines.append(f"- Restore the background rhythm: `{background_next_command}`")
        else:
            lines.append("- Run `work queue` and capture the next concrete task.")

        metadata = _safety_metadata(
            pending_approvals=len(approvals),
            open_tasks=len(tasks),
            active_goals=len(goals),
            enabled_jobs=len(enabled_jobs),
            state_snapshot_jobs=len(state_snapshot_jobs),
            enabled_state_snapshot_jobs=len(enabled_state_snapshot_jobs),
            disabled_state_snapshot_jobs=(len(state_snapshot_jobs) - len(enabled_state_snapshot_jobs)),
            conversation_compaction_jobs=len(conversation_compaction_jobs),
            enabled_conversation_compaction_jobs=len(enabled_conversation_compaction_jobs),
            disabled_conversation_compaction_jobs=(
                len(conversation_compaction_jobs) - len(enabled_conversation_compaction_jobs)
            ),
            background_ready=background_ready,
            background_next_command=background_next_command,
            limit=limit,
        )
        return "\n".join(lines), metadata

    def daily_brief(_: dict[str, Any]) -> ToolResult:
        target_date = datetime.now().strftime("%Y-%m-%d")
        brief = build_daily_brief(
            store,
            vault,
            target_date=target_date,
        )
        path = _daily_note_path(vault, target_date)
        path_display = _safe_vault_path_display(path, vault)
        return ToolResult(
            "daily_brief",
            True,
            f"Daily brief saved: {path_display}\n\n{brief}",
            _write_note_metadata(vault, path, target_date=target_date),
        )

    def daily_plan(args: dict[str, Any]) -> ToolResult:
        try:
            target_date = normalize_daily_plan_target_date(
                args.get("target_date"),
                default_today=True,
            )
        except (TypeError, ValueError):
            return ToolResult(
                "daily_plan",
                False,
                "Daily plan target date must be a real date in YYYY-MM-DD form.",
                _safety_metadata(
                    failure_kind="daily_plan_invalid_target_date",
                    reason="invalid_target_date",
                    handler_invoked=True,
                ),
            )
        plan = build_daily_plan(store, vault, target_date=target_date)
        path = _daily_note_path(vault, target_date)
        path_display = _safe_vault_path_display(path, vault)
        return ToolResult(
            "daily_plan",
            True,
            f"Daily plan saved: {path_display}\n\n{plan}",
            _write_note_metadata(vault, path, target_date=target_date),
        )

    def goal_nudge(args: dict[str, Any]) -> ToolResult:
        stale_days = _bounded_int(args.get("stale_days"), 7, 1, MAX_STALE_DAYS)
        target_date = datetime.now().strftime("%Y-%m-%d")
        nudge = build_goal_nudge(
            store,
            vault,
            stale_days,
            target_date=target_date,
        )
        path = _daily_note_path(vault, target_date)
        path_display = _safe_vault_path_display(path, vault)
        return ToolResult(
            "goal_nudge",
            True,
            f"Goal nudge saved: {path_display}\n\n{nudge}",
            _write_note_metadata(
                vault,
                path,
                stale_days=stale_days,
                target_date=target_date,
            ),
        )

    def weekly_review(_: dict[str, Any]) -> ToolResult:
        review = build_weekly_review(store, vault)
        path = _dated_reflection_path(vault, "Weekly Review")
        path_display = _safe_vault_path_display(path, vault)
        return ToolResult(
            "weekly_review",
            True,
            f"Weekly review saved: {path_display}\n\n{review}",
            _write_note_metadata(vault, path),
        )

    def weekly_review_context(_: dict[str, Any]) -> ToolResult:
        open_tasks = len(store.list_tasks(status="open", limit=1000))
        active_goals = len(store.list_goals(status="active", limit=1000))
        pending_approvals = len(store.list_pending_approvals(limit=1000))
        recent_sessions = len(store.list_sessions(limit=1000))
        active_decisions = len(store.list_decisions(status="active", limit=1000))
        active_preferences = len(store.list_preferences(status="active", limit=1000))
        recent_tool_runs = len(store.recent_tool_runs(limit=1000))
        handoff = {
            "source": "weekly_review_context",
            "ready_for_operator": True,
            "model_assist_packet_ready": True,
            "model_use_allowed": True,
            "model_use_scope": "reflection_and_planning_only",
            "requires_manual_review": True,
            "content_in_handoff": True,
            "state_changed": False,
            "changed": [],
            "open_tasks": open_tasks,
            "active_goals": active_goals,
            "pending_approvals": pending_approvals,
            "recent_sessions": recent_sessions,
            "active_decisions": active_decisions,
            "active_preferences": active_preferences,
            "recent_tool_runs": recent_tool_runs,
            "review_questions": [
                "What changed that the operator should remember?",
                "Which blockers are real versus stale?",
                "Which next action is safe, specific, and small?",
                "Which repeated workflow should become a saved skill?",
            ],
            "next_commands": {
                "write_review": "weekly review",
                "prompt_preview": "weekly review prompt preview",
                "inspect_safety": "safety status",
                "inspect_approvals": "pending approvals",
                "plan_next": "safe next actions",
            },
            "boundaries": _weekly_review_boundaries(),
        }
        context = build_weekly_review_model_context(store)
        return ToolResult(
            "weekly_review_context",
            True,
            context,
            _safety_metadata(
                weekly_review_context_handoff_ready=True,
                weekly_review_context_handoff=handoff,
                ready_for_operator=True,
                model_assist_packet_ready=True,
                model_use_allowed=True,
                model_use_scope="reflection_and_planning_only",
                requires_manual_review=True,
                content_in_handoff=True,
                state_changed=False,
                changed=[],
                open_tasks=open_tasks,
                active_goals=active_goals,
                pending_approvals=pending_approvals,
                recent_sessions=recent_sessions,
                active_decisions=active_decisions,
                active_preferences=active_preferences,
                recent_tool_runs=recent_tool_runs,
            ),
        )

    def weekly_review_prompt_preview(_: dict[str, Any]) -> ToolResult:
        context = build_weekly_review_model_context(store)
        open_tasks = len(store.list_tasks(status="open", limit=1000))
        active_goals = len(store.list_goals(status="active", limit=1000))
        pending_approvals = len(store.list_pending_approvals(limit=1000))
        prompt_sections = {
            "system": (
                "You are helping the operator review the week from a Jarvis read-only context packet. "
                "Summarize themes, progress, blockers, risks, and three safe next actions. "
                "Use only the supplied context. Do not claim actions are complete unless the context proves it. "
                "Do not approve, dismiss, rerun, execute tools, write files, read private data, send messages, or control the computer."
            ),
            "user": context,
            "response_contract": (
                "Return: 1) themes, 2) progress, 3) blockers/risks, 4) three safe next actions, "
                "5) questions for the operator. Label uncertainty when evidence is thin."
            ),
        }
        handoff = {
            "source": "weekly_review_prompt_preview",
            "ready_for_operator": True,
            "model_assist_prompt_ready": True,
            "model_use_allowed": True,
            "model_use_scope": "reflection_and_planning_only",
            "requires_manual_review": True,
            "content_in_handoff": True,
            "state_changed": False,
            "changed": [],
            "prompt_sections": prompt_sections,
            "context_chars": len(context),
            "open_tasks": open_tasks,
            "active_goals": active_goals,
            "pending_approvals": pending_approvals,
            "next_commands": {
                "inspect_context": "weekly review context",
                "write_review_after_manual_review": "weekly review",
                "inspect_safety": "safety status",
                "inspect_approvals": "pending approvals",
            },
            "boundaries": _weekly_review_boundaries(),
        }
        output = "\n".join(
            [
                "Weekly review prompt preview:",
                "This packet is ready to paste into a model for reflection only. Jarvis does not call the model or grant execution authority.",
                "",
                "System prompt:",
                prompt_sections["system"],
                "",
                "User context:",
                prompt_sections["user"],
                "",
                "Response contract:",
                prompt_sections["response_contract"],
                "",
                "Boundary:",
                "- Read-only prompt preview. No model call, tool execution, approval, note write, personal-data read, external side effect, speech, task completion, or computer control.",
                "- Manual review is required before writing a weekly review or taking any next action.",
            ]
        )
        return ToolResult(
            "weekly_review_prompt_preview",
            True,
            output,
            _safety_metadata(
                weekly_review_prompt_preview_handoff_ready=True,
                weekly_review_prompt_preview_handoff=handoff,
                ready_for_operator=True,
                model_assist_prompt_ready=True,
                model_use_allowed=True,
                model_use_scope="reflection_and_planning_only",
                requires_manual_review=True,
                content_in_handoff=True,
                state_changed=False,
                changed=[],
                context_chars=len(context),
                open_tasks=open_tasks,
                active_goals=active_goals,
                pending_approvals=pending_approvals,
            ),
        )

    def morning_startup(args: dict[str, Any]) -> ToolResult:
        body, metadata = build_morning_startup(_bounded_int(args.get("limit"), 6))
        return ToolResult("morning_startup", True, body, metadata)

    def save_morning_startup(args: dict[str, Any]) -> ToolResult:
        body, metadata = build_morning_startup(_bounded_int(args.get("limit"), 6))
        path = vault.append_daily("Jarvis Morning Startup", body)
        path_display = _safe_vault_path_display(path, vault)
        metadata.update(_write_note_metadata(vault, path))
        return ToolResult("save_morning_startup", True, f"Morning startup saved: {path_display}\n\n{body}", metadata)

    return daily_brief, daily_plan, goal_nudge, weekly_review, weekly_review_context, weekly_review_prompt_preview, morning_startup, save_morning_startup
