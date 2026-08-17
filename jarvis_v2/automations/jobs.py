from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from jarvis_v2.memory.obsidian import ObsidianVault, safe_text as _safe_note_text
from jarvis_v2.memory.store import (
    DailyBriefSnapshot,
    DailyPlanSnapshot,
    GoalNudgeSnapshot,
    MemoryStore,
)


DAILY_PLAN_DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


@dataclass(frozen=True)
class GoalNudgeBuild:
    output: str
    source_manifest_kind: str
    source_manifest_digest: str


def _goal_nudge_title_key(value: object) -> str:
    """Coalesce only exact user-visible title duplicates in the nudge view."""
    return " ".join(unicodedata.normalize("NFKC", _safe_text(value)).split()).casefold()


def _goal_nudge_title_display(value: object) -> str:
    return " ".join(_safe_text(value).split())


def _render_goal_nudge(snapshot: GoalNudgeSnapshot, stale_days: int) -> GoalNudgeBuild:
    steps_by_goal = {int(step["goal_id"]): step for step in snapshot.open_steps}
    stale_goal_ids = set(snapshot.stale_goal_ids)
    stale = [goal for goal in snapshot.goals if int(goal["id"]) in stale_goal_ids]
    lines = [
        f"Generated: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}",
        "",
        f"## Goals Quiet For {stale_days}+ Days",
    ]
    if not stale:
        lines.append("- No stale active goals.")
    grouped: dict[str, list[object]] = {}
    for goal in stale:
        title_key = _goal_nudge_title_key(goal["title"])
        # Never merge malformed/blank titles: only explicit normalized text is a duplicate.
        if not title_key:
            title_key = f"\x00{int(goal['id'])}"
        grouped.setdefault(title_key, []).append(goal)
    for goals in grouped.values():
        goal = goals[0]
        next_step = "choose a new next step"
        for candidate in goals:
            open_step = steps_by_goal.get(int(candidate["id"]))
            if open_step is not None:
                candidate_step = _safe_text(open_step["body"])
                if candidate_step:
                    next_step = candidate_step
                    break
        duplicate_suffix = (
            f" (+{len(goals) - 1} duplicate-title goal"
            f"{'s' if len(goals) > 2 else ''})"
            if len(goals) > 1
            else ""
        )
        lines.append(
            f"- #{goal['id']}{duplicate_suffix} "
            f"{_goal_nudge_title_display(goal['title'])} | next: {next_step}"
        )
    if snapshot.truncated:
        lines.append("- Additional active goals were omitted from this bounded nudge.")
    return GoalNudgeBuild(
        output="\n".join(lines),
        source_manifest_kind=snapshot.source_manifest_kind,
        source_manifest_digest=snapshot.source_manifest_digest,
    )


def _safe_text(value: object, limit: int | None = None) -> str:
    text = str(value or "")
    if limit is not None:
        text = text[:limit]
    return _safe_note_text(text)


def _job_enabled(row: object) -> bool:
    try:
        value = row["enabled"]  # type: ignore[index]
    except Exception:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    return False


def normalize_daily_plan_target_date(
    value: object,
    *,
    default_today: bool = False,
) -> str:
    if value is None and default_today:
        return datetime.now().strftime("%Y-%m-%d")
    if type(value) is not str or DAILY_PLAN_DATE_RE.fullmatch(value) is None:
        raise ValueError("Daily plan target date must use YYYY-MM-DD.")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("Daily plan target date must be a real calendar date.") from exc
    if parsed.strftime("%Y-%m-%d") != value:
        raise ValueError("Daily plan target date must be canonical.")
    return value


def daily_plan_auto_mutation_operation_key(args: dict[str, object]) -> dict[str, object]:
    return {"target_date": args.get("target_date")}


def daily_plan_auto_mutation_preflight(args: dict[str, object]) -> str | None:
    try:
        normalize_daily_plan_target_date(args.get("target_date"))
    except (TypeError, ValueError):
        return "invalid_target_date"
    return None


def goal_nudge_auto_mutation_operation_key(
    _args: dict[str, object],
) -> dict[str, str]:
    # The direct tool always appends to a daily note. An unresolved outcome must
    # fence every direct retry, including one with a different stale-day window.
    return {"projection": "goal_nudge"}


def daily_brief_auto_mutation_operation_key(
    _args: dict[str, object],
) -> dict[str, str]:
    return {"projection": "daily_brief"}


def _render_daily_brief(snapshot: DailyBriefSnapshot) -> str:
    memories = snapshot.memories
    skills = snapshot.skills
    goals = snapshot.goals
    tasks = snapshot.tasks
    approvals = snapshot.approvals
    decisions = snapshot.decisions
    people = snapshot.people
    preferences = snapshot.preferences
    steps_by_goal = {int(step["goal_id"]): step for step in snapshot.open_steps}

    lines = [
        f"Generated: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}",
        "",
        "## Recent Memory",
    ]
    if memories:
        for row in memories:
            lines.append(f"- [{_safe_text(row['category'])}] {_safe_text(row['title'])}: {_safe_text(row['body'], 120)}")
    else:
        lines.append("- No recent memories.")

    lines.append("")
    lines.append("## Available Skills")
    if skills:
        for row in skills:
            lines.append(f"- {_safe_text(row['name'])}: {_safe_text(row['trigger'])}")
    else:
        lines.append("- No saved skills yet.")

    lines.append("")
    lines.append("## Active Goals")
    if goals:
        for row in goals:
            open_step = steps_by_goal.get(int(row["id"]))
            next_step = _safe_text(open_step["body"]) if open_step is not None else "no open steps"
            lines.append(f"- #{row['id']} {_safe_text(row['title'])}: {next_step}")
    else:
        lines.append("- No active goals.")

    lines.append("")
    lines.append("## Open Tasks")
    if tasks:
        for row in tasks:
            due = f" due {_safe_text(row['due'])}" if row["due"] else ""
            priority = f" [{_safe_text(row['priority'])}]" if row["priority"] != "normal" else ""
            lines.append(f"- #{row['id']} {_safe_text(row['body'])}{due}{priority}")
    else:
        lines.append("- No open tasks.")

    lines.append("")
    lines.append("## Approval Queue")
    if approvals:
        for row in approvals:
            lines.append(f"- #{row['id']} {_safe_text(row['tool_name'])}: {_safe_text(row['user_input'])}")
    else:
        lines.append("- No pending approvals.")

    lines.append("")
    lines.append("## Active Decisions")
    if decisions:
        for row in decisions:
            rationale = f": {_safe_text(row['rationale'], 100)}" if row["rationale"] else ""
            lines.append(f"- #{row['id']} {_safe_text(row['title'])}{rationale}")
    else:
        lines.append("- No active decisions.")

    lines.append("")
    lines.append("## People Context")
    if people:
        for row in people:
            relation = f" | {_safe_text(row['relation'])}" if row["relation"] else ""
            last_contact = f" | last contact {_safe_text(row['last_contact_at'])}" if row["last_contact_at"] else ""
            lines.append(f"- #{row['id']} {_safe_text(row['name'])}{relation}{last_contact}")
    else:
        lines.append("- No people profiles yet.")

    lines.append("")
    lines.append("## Active Preferences")
    if preferences:
        for row in preferences:
            lines.append(f"- [{_safe_text(row['category'])}] {_safe_text(row['key'])}: {_safe_text(row['value'])}")
    else:
        lines.append("- No structured preferences yet.")

    lines.append("")
    lines.append("## Suggested Next Work")
    if approvals:
        lines.append("- Review pending approvals before relying on blocked high-risk actions.")
    if tasks:
        lines.append(f"- Start with task #{tasks[0]['id']}: {_safe_text(tasks[0]['body'])}")
    if goals:
        for row in goals[:3]:
            open_step = steps_by_goal.get(int(row["id"]))
            next_step = _safe_text(open_step["body"]) if open_step is not None else "define the next step"
            lines.append(f"- {_safe_text(row['title'])}: {next_step}")
    if not approvals and not tasks and not goals:
        lines.append("- Curate weak memories.")
        lines.append("- Move more old Jarvis tools into V2 with permission levels.")
        lines.append("- Teach Jarvis one reusable skill after each successful workflow.")

    if snapshot.truncated_sources or snapshot.clipped_sources:
        lines.extend(
            [
                "",
                "## Source Limits",
                "- Additional or oversized source context was omitted to keep this brief bounded.",
            ]
        )

    return "\n".join(lines)


def build_daily_brief(
    store: MemoryStore,
    vault: ObsidianVault,
    *,
    target_date: str | None = None,
    effect_authority: Callable[[], None] | None = None,
    scheduled_note_key: str = "",
    scheduled_note_date: str = "",
) -> str:
    require_effect_authority = effect_authority or (lambda: None)
    if scheduled_note_key:
        return _render_daily_brief(store.read_daily_brief_snapshot())
    resolved_target_date = normalize_daily_plan_target_date(
        target_date,
        default_today=True,
    )
    with vault.daily_append_fence(resolved_target_date) as append:
        with store.daily_brief_publication_snapshot() as snapshot:
            brief = _render_daily_brief(snapshot)
            require_effect_authority()
            append("Jarvis Proactive Brief", brief)
    return brief


def _render_daily_plan(snapshot: DailyPlanSnapshot, target_date: str) -> str:
    tasks = snapshot.tasks
    goals = snapshot.goals
    approvals = snapshot.approvals
    jobs = snapshot.jobs
    steps_by_goal = {int(step["goal_id"]): step for step in snapshot.open_steps}

    lines = [
        f"Plan date: {target_date}",
        f"Generated: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}",
        "",
        "## Focus Queue",
    ]
    focus_count = 0
    for task in tasks[:5]:
        due = f" due {_safe_text(task['due'])}" if task["due"] else ""
        priority = f" [{_safe_text(task['priority'])}]" if task["priority"] != "normal" else ""
        lines.append(f"- Task #{task['id']}: {_safe_text(task['body'])}{due}{priority}")
        focus_count += 1
    for goal in goals:
        if focus_count >= 8:
            break
        open_step = steps_by_goal.get(int(goal["id"]))
        next_step = _safe_text(open_step["body"]) if open_step is not None else "define the next step"
        lines.append(f"- Goal #{goal['id']} {_safe_text(goal['title'])}: {next_step}")
        focus_count += 1
    if focus_count == 0:
        lines.append("- No open tasks or active goal steps. Pick one clear target for today.")

    lines.extend(["", "## Pending Approvals"])
    if approvals:
        for approval in approvals:
            lines.append(f"- Approval #{approval['id']} {_safe_text(approval['tool_name'])}: {_safe_text(approval['user_input'])}")
    else:
        lines.append("- No pending approvals.")

    lines.extend(["", "## Today/Next Automations"])
    if jobs:
        for job in jobs[:8]:
            status = "on" if _job_enabled(job) else "off"
            lines.append(f"- {_safe_text(job['name'])} [{_safe_text(job['job_type'])}, {status}] next {_safe_text(job['next_run_at'])}")
    else:
        lines.append("- No scheduled jobs.")

    lines.extend(["", "## Suggested Order"])
    if approvals:
        lines.append("- Review pending approvals before expecting blocked high-risk actions to run.")
    if tasks:
        lines.append(f"- Start with task #{tasks[0]['id']}: {_safe_text(tasks[0]['body'])}")
    elif goals:
        first_goal = goals[0]
        open_step = steps_by_goal.get(int(first_goal["id"]))
        next_step = _safe_text(open_step["body"]) if open_step is not None else "define the next step"
        lines.append(f"- Start with {_safe_text(first_goal['title'])}: {next_step}")
    else:
        lines.append("- Capture a task or create a goal before doing reactive work.")

    if snapshot.clipped_sources:
        lines.extend(
            [
                "",
                "## Source Limits",
                "- Oversized source text was clipped to keep this plan bounded.",
            ]
        )

    return "\n".join(lines)


def build_daily_plan(
    store: MemoryStore,
    vault: ObsidianVault,
    *,
    target_date: str | None = None,
) -> str:
    resolved_target_date = normalize_daily_plan_target_date(
        target_date,
        default_today=True,
    )
    with vault.daily_append_fence(resolved_target_date) as append:
        with store.daily_plan_publication_snapshot() as snapshot:
            plan = _render_daily_plan(snapshot, resolved_target_date)
            append("Jarvis Daily Plan", plan)
    return plan


def build_goal_nudge(
    store: MemoryStore,
    vault: ObsidianVault,
    stale_days: int = 7,
    *,
    target_date: str | None = None,
    effect_authority: Callable[[], None] | None = None,
    scheduled_note_key: str = "",
    scheduled_note_date: str = "",
    include_source_manifest: bool = False,
) -> str | GoalNudgeBuild:
    require_effect_authority = effect_authority or (lambda: None)
    if scheduled_note_key:
        build = _render_goal_nudge(
            store.read_goal_nudge_snapshot(stale_days=stale_days),
            stale_days,
        )
        return build if include_source_manifest else build.output
    resolved_target_date = normalize_daily_plan_target_date(
        target_date,
        default_today=True,
    )
    with store.goal_nudge_publication_snapshot(stale_days=stale_days) as snapshot:
        build = _render_goal_nudge(snapshot, stale_days)
        require_effect_authority()
        vault.append_daily_for_date(
            resolved_target_date,
            "Jarvis Goal Nudge",
            build.output,
        )
    return build.output


def build_weekly_review(
    store: MemoryStore,
    vault: ObsidianVault,
    *,
    effect_authority: Callable[[], None] | None = None,
    scheduled_note_key: str = "",
    scheduled_note_date: str = "",
) -> str:
    require_effect_authority = effect_authority or (lambda: None)
    model_context = build_weekly_review_model_context(store)
    memories = store.recent_memories(limit=20)
    goals = store.list_goals(limit=100)
    active_goals = [goal for goal in goals if goal["status"] == "active"]
    done_goals = [goal for goal in goals if goal["status"] == "done"]
    sessions = store.list_sessions(limit=10)
    jobs = store.list_jobs()

    lines = [
        f"Generated: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}",
        "",
        "## Goals",
        f"- Active: {len(active_goals)}",
        f"- Done: {len(done_goals)}",
    ]
    for goal in active_goals[:10]:
        steps = store.list_goal_steps(goal["id"])
        open_steps = [step for step in steps if step["status"] != "done"]
        lines.append(f"- #{goal['id']} {_safe_text(goal['title'])}: {len(open_steps)} open steps")

    lines.extend(["", "## Recent Memory Themes"])
    if memories:
        for row in memories[:10]:
            lines.append(f"- [{_safe_text(row['category'])}] {_safe_text(row['title'])}")
    else:
        lines.append("- No recent memories.")

    lines.extend(["", "## Conversation Activity"])
    if sessions:
        for row in sessions[:5]:
            lines.append(f"- {_safe_text(row['session_id'])}: {row['messages']} messages, last {_safe_text(row['last_at'])}")
    else:
        lines.append("- No sessions indexed.")

    lines.extend(["", "## Automations"])
    if jobs:
        for row in jobs:
            status = "on" if _job_enabled(row) else "off"
            lines.append(f"- {_safe_text(row['name'])} [{_safe_text(row['job_type'])}, {status}] next {_safe_text(row['next_run_at'])}")
    else:
        lines.append("- No scheduled jobs.")

    lines.extend(["", "## Suggested Next Week"])
    if active_goals:
        for goal in active_goals[:5]:
            steps = store.list_goal_steps(goal["id"])
            open_steps = [step for step in steps if step["status"] != "done"]
            next_step = _safe_text(open_steps[0]["body"]) if open_steps else "define the next step"
            lines.append(f"- {_safe_text(goal['title'])}: {next_step}")
    else:
        lines.append("- Pick one active goal and define its next physical action.")
    lines.append("- Convert one repeated workflow into a saved skill.")
    lines.append("- Summarize important sessions into Reflections.")
    lines.extend(
        [
            "",
            "## Model Assist Packet",
            "Use `weekly review context` to inspect the full read-only packet before asking any model to summarize it.",
            "Rule: a model may suggest themes and next questions, but tools, approvals, shell, personal data, reminders, and computer control stay behind Jarvis runtime gates.",
            "",
            model_context,
        ]
    )

    review = "\n".join(lines)
    if scheduled_note_key:
        return review
    require_effect_authority()
    vault.write_reflection("Weekly Review", "# Weekly Review\n\n" + review + "\n")
    return review


def build_weekly_review_model_context(store: MemoryStore) -> str:
    memories = store.recent_memories(limit=12)
    goals = store.list_goals(limit=20)
    tasks = store.list_tasks(status="open", limit=12)
    approvals = store.list_pending_approvals(limit=8)
    decisions = store.list_decisions(status="active", limit=8)
    preferences = store.list_preferences(status="active", limit=8)
    sessions = store.list_sessions(limit=8)
    tool_runs = store.recent_tool_runs(limit=12)

    lines = [
        "Weekly review model context:",
        "Purpose: provide a compact, read-only packet a model can use to help the operator reflect and plan.",
        "Safety boundary: this packet must not approve, dismiss, rerun, write files, send messages, read private data, or control the computer.",
        "",
        "Model prompt:",
        "Summarize the week into themes, progress, blockers, risks, and 3 safe next actions. Use only this context. If evidence is thin, say so. Do not claim actions were completed unless listed here.",
        "",
        "Open tasks:",
    ]
    if tasks:
        for row in tasks:
            priority = f" [{_safe_text(row['priority'])}]" if row["priority"] != "normal" else ""
            due = f" due {_safe_text(row['due'])}" if row["due"] else ""
            lines.append(f"- Task #{row['id']}: {_safe_text(row['body'])}{due}{priority}")
    else:
        lines.append("- None.")

    lines.extend(["", "Goals:"])
    if goals:
        for goal in goals[:12]:
            steps = store.list_goal_steps(goal["id"])
            open_steps = [step for step in steps if step["status"] != "done"]
            next_step = _safe_text(open_steps[0]["body"]) if open_steps else "no open step"
            lines.append(f"- Goal #{goal['id']} [{_safe_text(goal['status'])}] {_safe_text(goal['title'])}: {next_step}")
    else:
        lines.append("- None.")

    lines.extend(["", "Pending approvals:"])
    if approvals:
        for row in approvals:
            lines.append(f"- Approval #{row['id']} {_safe_text(row['tool_name'])}: {_safe_text(row['user_input'])}")
    else:
        lines.append("- None.")

    lines.extend(["", "Active decisions:"])
    if decisions:
        for row in decisions:
            rationale = f" | rationale: {_safe_text(row['rationale'], 120)}" if row["rationale"] else ""
            lines.append(f"- Decision #{row['id']}: {_safe_text(row['title'])}{rationale}")
    else:
        lines.append("- None.")

    lines.extend(["", "Active preferences:"])
    if preferences:
        for row in preferences:
            lines.append(f"- [{_safe_text(row['category'])}] {_safe_text(row['key'])}: {_safe_text(row['value'])}")
    else:
        lines.append("- None.")

    lines.extend(["", "Recent memories:"])
    if memories:
        for row in memories:
            lines.append(f"- [{_safe_text(row['category'])}] {_safe_text(row['title'])}: {_safe_text(row['body'], 140)}")
    else:
        lines.append("- None.")

    lines.extend(["", "Conversation sessions:"])
    if sessions:
        for row in sessions:
            lines.append(f"- {_safe_text(row['session_id'])}: {row['messages']} messages, last {_safe_text(row['last_at'])}")
    else:
        lines.append("- None.")

    lines.extend(["", "Recent tool runs:"])
    if tool_runs:
        for row in tool_runs:
            status = "ok" if row["ok"] else "blocked/failed"
            approval = "approved" if row["approved"] else "not approved"
            lines.append(f"- {_safe_text(row['tool_name'])} [{_safe_text(row['risk'])}, {status}, {approval}] at {_safe_text(row['created_at'])}")
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "Review questions:",
            "- What changed that the operator should remember?",
            "- Which blockers are real versus stale?",
            "- Which next action is safe, specific, and small?",
            "- Which repeated workflow should become a saved skill?",
        ]
    )
    return "\n".join(lines)
