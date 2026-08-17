from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import math
import re
from typing import Callable, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from jarvis_v2.agent.failure_guidance import (
    declare_failure_guidance,
    declare_retryable_local_read_failure,
)
from jarvis_v2.agent.types import ApprovalArgumentResolution, RiskLevel, ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.memory.decision_projection import reconcile_pending_decision_projections
from jarvis_v2.memory.goal_projection import reconcile_pending_goal_projections
from jarvis_v2.memory.memory_projection import (
    reconcile_memory_projection,
    reconcile_pending_memory_projections,
)
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.person_projection import reconcile_pending_person_projections
from jarvis_v2.memory.preference_projection import reconcile_pending_preference_projections
from jarvis_v2.memory.profile_projection import reconcile_pending_profile_projections
from jarvis_v2.memory.store import MemoryRecord, MemoryStore
from jarvis_v2.tools import approvals, architecture, audit, autonomy, brain, browser, capabilities, channel_health, cockpit, computer, continuity, decisions, doctor, feedback, files, focus, goals, harness, help, ingest, knowledge_promotion, model_status, next_step, notes, organize, people, preferences, privacy, profile, readiness, rehearsal, roadmap, safety, shell, startup_recovery, state, storage, subagents, system, tasks, utilities, voice
from jarvis_v2.tools.conversation import make_conversation_tools
from jarvis_v2.tools.memory_curator import (
    make_memory_approval_resolvers,
    make_memory_curator_tools,
    queue_learning_tasks_auto_mutation_operation_key,
)
from jarvis_v2.tools.personal import make_personal_tools
from jarvis_v2.tools.proactive import make_proactive_tools
from jarvis_v2.automations.jobs import (
    daily_brief_auto_mutation_operation_key,
    daily_plan_auto_mutation_operation_key,
    daily_plan_auto_mutation_preflight,
    goal_nudge_auto_mutation_operation_key,
)
from jarvis_v2.tools.scheduler import make_scheduler_tools
from jarvis_v2.tools.skills import make_skill_tools
from jarvis_v2.v3_commands import (
    V3_DASHBOARD_COMMAND,
    V3_DASHBOARD_INFO_COMMAND,
    V3_DASHBOARD_MODULE_COMMAND,
)
from jarvis_v2.tools.calendar_connector import make_calendar_tools
from jarvis_v2.tools.email_connector import make_email_tools
from jarvis_v2.tools.imessage_connector import make_imessage_tools
from jarvis_v2.tools.weather_connector import make_weather_tools
from jarvis_v2.tools.reminder_tools import make_reminder_tools
from jarvis_v2.tools.news_connector import make_news_tools
from jarvis_v2.tools.brief_tools import make_brief_tools
from jarvis_v2.tools.currency_connector import make_currency_tools
from jarvis_v2.tools.translate_connector import make_translate_tools
from jarvis_v2.tools.markets_connector import make_markets_tools
from jarvis_v2.tools.research_connector import make_research_tools
from jarvis_v2.tools.dictionary_connector import make_dictionary_tools
from jarvis_v2.tools.wikipedia_connector import make_wikipedia_tools
from jarvis_v2.tools.sun_connector import make_sun_tools
from jarvis_v2.tools.history_connector import make_history_tools
from jarvis_v2.tools.air_connector import make_air_tools
from jarvis_v2.tools.holidays_connector import make_holidays_tools
from jarvis_v2.tools.fun_connector import make_fun_tools
from jarvis_v2.tools.countdown_connector import make_countdown_tools
from jarvis_v2.tools.writer_connector import make_writer_tools
from jarvis_v2.tools.compose_connector import make_compose_tools
from jarvis_v2.tools.instagram_connector import make_instagram_tools
from jarvis_v2.tools.kakao_connector import make_kakao_tools
from jarvis_v2.tools.contacts_connector import make_contacts_tools
from jarvis_v2.tools.call_connector import make_call_tools
from jarvis_v2.tools.ocr import make_ocr_tools
from jarvis_v2.tools.apple_reminders import make_apple_reminders_tools
from jarvis_v2.tools.auto_mutation_reconciliation import make_auto_mutation_reconciliation_tools


ToolHandler = Callable[[dict[str, Any]], ToolResult]
MAX_CORE_TEXT_CHARS = 1200
MAX_CORE_FIELD_CHARS = 160
MAX_CORE_LIMIT = 100
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")

TIMEZONE_ALIASES = {
    "seoul": ("Asia/Seoul", "Seoul"),
    "korea": ("Asia/Seoul", "Korea"),
    "south korea": ("Asia/Seoul", "South Korea"),
    "tokyo": ("Asia/Tokyo", "Tokyo"),
    "japan": ("Asia/Tokyo", "Japan"),
    "new york": ("America/New_York", "New York"),
    "nyc": ("America/New_York", "New York"),
    "los angeles": ("America/Los_Angeles", "Los Angeles"),
    "la": ("America/Los_Angeles", "Los Angeles"),
    "san francisco": ("America/Los_Angeles", "San Francisco"),
    "london": ("Europe/London", "London"),
    "paris": ("Europe/Paris", "Paris"),
    "berlin": ("Europe/Berlin", "Berlin"),
    "singapore": ("Asia/Singapore", "Singapore"),
    "hong kong": ("Asia/Hong_Kong", "Hong Kong"),
    "sydney": ("Australia/Sydney", "Sydney"),
    "melbourne": ("Australia/Melbourne", "Melbourne"),
    "utc": ("UTC", "UTC"),
}
KOREAN_WEEKDAY_LABELS = ("월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일")
KOREAN_LOCATION_LABELS = {
    "Seoul": "서울",
    "Korea": "한국",
    "South Korea": "대한민국",
    "Tokyo": "도쿄",
    "Japan": "일본",
    "New York": "뉴욕",
    "Los Angeles": "로스앤젤레스",
    "San Francisco": "샌프란시스코",
    "London": "런던",
    "Paris": "파리",
    "Berlin": "베를린",
    "Singapore": "싱가포르",
    "Hong Kong": "홍콩",
    "Sydney": "시드니",
    "Melbourne": "멜버른",
    "UTC": "UTC",
}

CORE_INPUT_RECOVERY_ACTION = (
    "Correct the reported input, then retry through the normal policy."
)
MEMORY_PROJECTION_RECOVERY_ACTION = (
    "Run `repair memory projections`; do not save the memory again."
)


def _timezone_database_recovery_guidance() -> str:
    return (
        "Run `setup check`, then install or update the `tzdata` package in Jarvis's "
        "Python environment and retry. Run `time in UTC` as a temporary fallback."
    )


def _timezone_database_recovery_metadata() -> dict[str, Any]:
    return {
        "next_command": "setup check",
        "recovery_commands": ["setup check", "time in UTC"],
        "retry_requires_timezone_data_repair": True,
        "authorizes_retry": False,
    }

WEEKDAY_ALIASES = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
RELATIVE_DATE_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
}


def _bounded_int(value: Any, default: int, low: int = 1, high: int = MAX_CORE_LIMIT) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _resolve_timezone_location(value: Any) -> tuple[str, str] | None:
    text = str(value or "").strip().lower()
    text = re.sub(r"^(?:in|for|at)\s+", "", text)
    text = re.sub(r"[\?\.!]+$", "", text).strip()
    if not text or LOCAL_PATH_RE.search(text):
        return None
    return TIMEZONE_ALIASES.get(text)


def _format_korean_datetime(value: datetime) -> str:
    period = "오전" if value.hour < 12 else "오후"
    hour = value.hour % 12 or 12
    minute = f" {value.minute}분" if value.minute else ""
    weekday = KOREAN_WEEKDAY_LABELS[value.weekday()]
    return f"{value.year}년 {value.month}월 {value.day}일 {weekday} {period} {hour}시{minute}"


def _parse_relative_date(value: Any, today: datetime | None = None) -> tuple[str, datetime, int] | None:
    now = today or datetime.now()
    text = str(value or "").strip().lower()
    text = re.sub(r"[\?\.!]+$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^the\s+", "", text)
    if not text or LOCAL_PATH_RE.search(text):
        return None
    offsets = {
        "tomorrow": 1,
        "yesterday": -1,
        "day after tomorrow": 2,
        "day before yesterday": -2,
    }
    if text in offsets:
        offset = offsets[text]
        label = {
            1: "Tomorrow",
            -1: "Yesterday",
            2: "The day after tomorrow",
            -2: "The day before yesterday",
        }[offset]
        return label, now + timedelta(days=offset), offset

    if text in {"next week", "a week from now", "one week from now"}:
        return "One week from now", now + timedelta(days=7), 7
    if text in {"last week", "a week ago", "one week ago"}:
        return "One week ago", now - timedelta(days=7), -7

    quantity_match = re.fullmatch(
        r"(?:(?:in\s+)?(?P<count_future>\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty)\s+(?P<unit_future>days?|weeks?)(?:\s+from\s+now)?|(?P<count_past>\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty)\s+(?P<unit_past>days?|weeks?)\s+ago)",
        text,
    )
    if quantity_match:
        raw_count = quantity_match.group("count_future") or quantity_match.group("count_past") or ""
        unit = quantity_match.group("unit_future") or quantity_match.group("unit_past") or ""
        count = int(raw_count) if raw_count.isdigit() else RELATIVE_DATE_NUMBER_WORDS.get(raw_count, 0)
        if count < 1:
            return None
        days = count * (7 if unit.startswith("week") else 1)
        if days > 370:
            return None
        is_past = bool(quantity_match.group("count_past"))
        offset = -days if is_past else days
        unit_label = "week" if unit.startswith("week") else "day"
        plural = "" if count == 1 else "s"
        label = f"{count} {unit_label}{plural} ago" if is_past else f"In {count} {unit_label}{plural}"
        return label, now + timedelta(days=offset), offset

    match = re.fullmatch(r"(next|this|last)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", text)
    if not match:
        return None
    direction, weekday_name = match.groups()
    target_weekday = WEEKDAY_ALIASES[weekday_name]
    current_weekday = now.weekday()
    if direction == "next":
        offset = (target_weekday - current_weekday) % 7
        if offset == 0:
            offset = 7
    elif direction == "this":
        offset = (target_weekday - current_weekday) % 7
    else:
        offset = -((current_weekday - target_weekday) % 7 or 7)
    label = f"{direction.title()} {weekday_name.title()}"
    return label, now + timedelta(days=offset), offset


def _short(value: Any, *, limit: int = MAX_CORE_FIELD_CHARS) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def remember_auto_mutation_operation_key(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "body": _short(args.get("body"), limit=MAX_CORE_TEXT_CHARS),
        "category": _short(args.get("category") or "facts", limit=80),
        "title": _short(args.get("title") or "Untitled", limit=MAX_CORE_FIELD_CHARS),
    }


def _short_metadata(value: Any, *, limit: int = MAX_CORE_FIELD_CHARS) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", _short(value, limit=limit))


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _empty_planner_diagnostic_metadata() -> dict[str, Any]:
    return {
        "planner_type": None,
        "model_planner_attempted": False,
        "model_planner_state": None,
        "model_planner_used": False,
        "model_planner_fell_back": False,
        "model_planner_fallback_reason": None,
        "model_planner_fallback_detail": None,
        "model_planner_recovery_hint": None,
        "model_planner_exception_type": None,
        "model_planner_model": None,
        "model_planner_timeout_seconds": None,
        "model_planner_action_count": None,
        "model_planner_ignored_unknown_tools": [],
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _safe_vault_path_display(path: Path | str | None, vault: ObsidianVault) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return str(candidate.relative_to(vault.root_path))
    except ValueError:
        return _short_metadata(candidate, limit=MAX_CORE_FIELD_CHARS)


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "calls_model": False,
        "executes_tools": False,
        "queues_approval": False,
        "requires_approval": False,
        "approves_request": False,
        "dismisses_request": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "reads_note_contents": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "controls_computer": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "speaks": False,
        "completes_tasks": False,
    }
    metadata.update(extra)
    return metadata


def _core_input_failure_metadata(output: str, **extra: Any) -> dict[str, Any]:
    return declare_retryable_local_read_failure(
        _safe_metadata(**extra),
        output=output,
        action=CORE_INPUT_RECOVERY_ACTION,
    )


def _timezone_database_failure_metadata(output: str, **extra: Any) -> dict[str, Any]:
    return declare_retryable_local_read_failure(
        _safe_metadata(
            **extra,
            **_timezone_database_recovery_metadata(),
        ),
        output=output,
        action=_timezone_database_recovery_guidance(),
        commands=("setup check", "time in UTC"),
    )


def _memory_projection_failure_metadata(output: str, **extra: Any) -> dict[str, Any]:
    return declare_failure_guidance(
        _safe_metadata(
            outcome_known=True,
            outcome_unknown=False,
            execution_outcome_unknown=False,
            side_effect_possible=True,
            retry_safe=False,
            automatic_retry_allowed=False,
            authorizes_retry=False,
            **extra,
        ),
        output=output,
        action=MEMORY_PROJECTION_RECOVERY_ACTION,
        commands=("repair memory projections",),
    )


def _memory_rows_handoff(
    rows: list[Any],
    *,
    source: str,
    limit: int,
    query: str | None = None,
) -> dict[str, Any]:
    compact_rows: list[dict[str, Any]] = []
    for row in rows[:12]:
        memory_id = int(row["id"])
        compact_rows.append(
            {
                "id": memory_id,
                "category": _short_metadata(row["category"], limit=80),
                "title": _short_metadata(row["title"], limit=160),
                "body_preview": _short_metadata(row["body"], limit=220),
                "source": _short_metadata(row["source"], limit=80),
            }
        )
    memory_ids = [row["id"] for row in compact_rows]
    next_commands = ["what do you remember"]
    if query:
        next_commands.insert(0, f"search memory for {query}")
    return {
        "source": source,
        "ready_for_operator": True,
        "query": query,
        "limit": limit,
        "count": len(rows),
        "memory_ids": memory_ids,
        "rows": compact_rows,
        "first_memory_id": memory_ids[0] if memory_ids else None,
        "next_commands": next_commands,
        "boundaries": {
            "read_only": True,
            "writes_files": False,
            "writes_memory": False,
            "writes_notes": False,
            "queues_approval": False,
            "controls_computer": False,
            "external_side_effect": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
        },
    }


def _memory_write_handoff(record: MemoryRecord, *, memory_id: int, note_path: Path, vault: ObsidianVault) -> dict[str, Any]:
    path_display = _safe_vault_path_display(note_path, vault)
    return {
        "source": "remember",
        "ready_for_operator": True,
        "memory_id": memory_id,
        "category": _short_metadata(record.category, limit=80),
        "title": _short_metadata(record.title, limit=160),
        "body_preview": _short_metadata(record.body, limit=220),
        "path_display": path_display,
        "next_commands": [
            f"search memory for {_short_metadata(record.title, limit=80)}",
            "what do you remember",
        ],
        "boundaries": {
            "read_only": False,
            "writes_files": True,
            "writes_memory": True,
            "writes_notes": True,
            "queues_approval": False,
            "controls_computer": False,
            "external_side_effect": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
        },
    }


def _daily_note_handoff(*, heading: str, body: str, path: Path, vault: ObsidianVault) -> dict[str, Any]:
    path_display = _safe_vault_path_display(path, vault)
    return {
        "source": "write_daily_note",
        "ready_for_operator": True,
        "heading": _short_metadata(heading, limit=MAX_CORE_FIELD_CHARS),
        "body_preview": _short_metadata(body, limit=220),
        "path_display": path_display,
        "next_commands": [
            "export state",
            "what do you remember",
        ],
        "boundaries": {
            "read_only": False,
            "writes_files": True,
            "writes_memory": True,
            "writes_notes": True,
            "queues_approval": False,
            "controls_computer": False,
            "external_side_effect": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
        },
    }


def _command_diagnosis_handoff(
    *,
    display_request: str,
    route: str,
    recommendation: str,
    next_commands: list[str],
    plan_goal: str,
    planned_actions: list[dict[str, Any]],
    approval_required: bool,
    pending_approvals: int,
    approval_queue_forecast: list[dict[str, Any]],
    forecast_new_approvals: int,
    forecast_reused_approval_ids: list[int],
    recovery_closure: dict[str, Any],
    execution_learning_debt: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source": "command_diagnosis",
        "request_length": len(display_request),
        "route": route,
        "recommendation": recommendation,
        "next_command": next_commands[0] if next_commands else "",
        "recommended_next_commands": list(next_commands),
        "recommended_next_command_count": len(next_commands),
        "goal": plan_goal,
        "planned_action_count": len(planned_actions),
        "planned_tools": [str(item.get("tool") or "") for item in planned_actions],
        "approval_required": bool(approval_required),
        "pending_approvals": int(pending_approvals),
        "approval_forecast_count": len(approval_queue_forecast),
        "forecast_new_approvals": int(forecast_new_approvals),
        "forecast_reused_approval_ids": list(forecast_reused_approval_ids),
        "forecast_queue_before": int(pending_approvals),
        "forecast_queue_after_if_sent": int(pending_approvals) + int(forecast_new_approvals),
        "safe_to_execute_now": route in {"chat", "auto_tool"},
        "recovery_closure_state": recovery_closure.get("state") or "not_available",
        "recovery_closure_blocks_auto_execution": _metadata_bool(recovery_closure.get("blocks_auto_execution")),
        "recovery_closure_recent_action_runs": int(recovery_closure.get("recent_action_runs") or 0),
        "recovery_closure_failed_or_blocked_action_runs": int(recovery_closure.get("failed_or_blocked_action_runs") or 0),
        "recovery_closure_approval_held_action_runs": int(recovery_closure.get("approval_held_action_runs") or 0),
        "recovery_closure_approval_held_target_run_id": recovery_closure.get("approval_held_target_run_id"),
        "recovery_closure_approval_held_target_tool_name": recovery_closure.get("approval_held_target_tool_name") or "",
        "recovery_closure_approval_review_commands": list(recovery_closure.get("approval_review_commands") or []),
        "recovery_closure_approval_review_command_count": int(recovery_closure.get("approval_review_command_count") or 0),
        "recovery_closure_proof_queue": list(recovery_closure.get("required_commands") or []),
        "recovery_closure_proof_queue_count": len(recovery_closure.get("required_commands") or []),
        "recovery_closure_next_proof_command": recovery_closure.get("next_required_command") or "",
        "execution_learning_state": execution_learning_debt.get("state") or "NO_RECENT_ACTION_RUNS",
        "execution_learning_blocks_completion_claim": _metadata_bool(execution_learning_debt.get("blocks_completion_claim")),
        "execution_learning_proof_queue": list(execution_learning_debt.get("required_commands") or []),
        "execution_learning_proof_queue_count": len(execution_learning_debt.get("required_commands") or []),
        "execution_learning_next_proof_command": execution_learning_debt.get("next_required_command") or "",
        "calls_model": False,
        "executes_tools": False,
        "queues_approval": False,
        "approves_request": False,
        "dismisses_request": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "executes_side_effect": False,
        "writes_files": False,
        "writes_memory": False,
        "writes_notes": False,
        "controls_computer": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _continuity_brief_handoff(
    *,
    source: str,
    limit: int,
    readiness_metadata: dict[str, Any] | None = None,
    activity_metadata: dict[str, Any] | None = None,
    build_metadata: dict[str, Any] | None = None,
    next_metadata: dict[str, Any] | None = None,
    work_queue_metadata: dict[str, Any] | None = None,
    approval_metadata: dict[str, Any] | None = None,
    agi_metadata: dict[str, Any],
) -> dict[str, Any]:
    readiness_metadata = readiness_metadata or {}
    activity_metadata = activity_metadata or {}
    build_metadata = build_metadata or {}
    next_metadata = next_metadata or {}
    work_queue_metadata = work_queue_metadata or {}
    approval_metadata = approval_metadata or {}
    safe_next_handoff = next_metadata.get("safe_next_actions_handoff") or {}
    work_queue_handoff = work_queue_metadata.get("work_queue_handoff") or {}
    next_commands = [
        "catch me up",
        "safety status",
        "return brief",
        "handoff brief",
        "session closeout",
        "safe next actions",
        "work queue",
        "next action packet",
        "priority stack",
        "focus brief",
        "build target packet",
    ]
    next_commands.extend(str(command) for command in safe_next_handoff.get("next_commands", []) if command)
    next_commands.extend(str(command) for command in work_queue_handoff.get("next_commands", []) if command)
    if source in {"handoff_brief", "session_closeout"}:
        next_commands.extend(["save handoff brief", "save session closeout"])
    if source == "session_closeout":
        next_commands.extend(["approval review", "save approval review"])
    agi_command = str(agi_metadata.get("agi_next_build_command") or "")
    if agi_command:
        next_commands.append(agi_command)

    deduped_next_commands: list[str] = []
    for command in next_commands:
        command = _short_metadata(command, limit=MAX_CORE_FIELD_CHARS)
        if command and command not in deduped_next_commands:
            deduped_next_commands.append(command)

    section_states: list[dict[str, Any]] = []
    for name, metadata, count_keys in [
        ("readiness", readiness_metadata, ("required_count", "count")),
        ("activity", activity_metadata, ("count",)),
        ("build_progress", build_metadata, ("count", "item_count")),
        ("safe_next_actions", next_metadata, ("items", "count")),
        ("work_queue", work_queue_metadata, ("work_queue_count", "item_count", "count")),
        ("approval_review", approval_metadata, ("pending_count", "count")),
        ("agi_build_readiness", agi_metadata, ("agi_next_evidence_closure_command_count",)),
    ]:
        if not metadata:
            continue
        item_count = 0
        for key in count_keys:
            if metadata.get(key) is not None:
                try:
                    item_count = int(metadata.get(key) or 0)
                except (TypeError, ValueError):
                    item_count = 0
                break
        section_states.append({"section": name, "ok": bool(metadata), "item_count": item_count})
    pending_approvals = (
        next_metadata.get("pending_approvals")
        or work_queue_metadata.get("pending_approvals")
        or approval_metadata.get("pending_count")
        or approval_metadata.get("pending_approvals")
        or 0
    )
    open_tasks = next_metadata.get("open_tasks") or work_queue_metadata.get("open_tasks") or 0
    active_goals = next_metadata.get("active_goals") or work_queue_metadata.get("active_goals") or 0
    return {
        "source": source,
        "status": "ready",
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "limit": limit,
        "sections": section_states,
        "section_count": len(section_states),
        "next_commands": deduped_next_commands,
        "next_command_count": len(deduped_next_commands),
        "pending_approvals": int(pending_approvals or 0),
        "open_tasks": int(open_tasks or 0),
        "active_goals": int(active_goals or 0),
        "scheduler_next_command": _short_metadata(next_metadata.get("scheduler_next_command") or ""),
        "execution_health_review_required": _metadata_bool(next_metadata.get("execution_health_review_required")),
        "execution_health_next_command": _short_metadata(next_metadata.get("execution_health_next_command") or ""),
        "doctor_completion_claim_state": _short_metadata(next_metadata.get("doctor_completion_claim_state") or ""),
        "doctor_completion_claim_ready": _metadata_bool(next_metadata.get("doctor_completion_claim_ready")),
        "agi_next_gate": _short_metadata(agi_metadata.get("agi_next_gate") or ""),
        "agi_next_target_title": _short_metadata(agi_metadata.get("agi_next_target_title") or ""),
        "agi_next_build_command": _short_metadata(agi_metadata.get("agi_next_build_command") or ""),
        "boundaries": {
            "read_only": True,
            "calls_model": False,
            "executes_tools": False,
            "queues_approval": False,
            "requires_approval": False,
            "approves_request": False,
            "dismisses_request": False,
            "reads_private_data": False,
            "reads_personal_data": False,
            "reads_note_contents": False,
            "executes_side_effect": False,
            "external_side_effect": False,
            "writes_files": False,
            "writes_database": False,
            "writes_memory": False,
            "writes_notes": False,
            "controls_computer": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "speaks": False,
            "completes_tasks": False,
        },
    }


def _brief_handoff_metadata(prefix: str, handoff: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_handoff_ready": handoff["handoff_ready"],
        f"{prefix}_ready_for_operator": handoff["ready_for_operator"],
        f"{prefix}_state_changed": handoff["state_changed"],
        f"{prefix}_changed": handoff["changed"],
        f"{prefix}_content_in_handoff": handoff["content_in_handoff"],
        f"{prefix}_sections": handoff["sections"],
        f"{prefix}_section_count": handoff["section_count"],
        f"{prefix}_next_commands": handoff["next_commands"],
        f"{prefix}_next_command_count": handoff["next_command_count"],
        f"{prefix}_boundaries": handoff["boundaries"],
        f"{prefix}_handoff": handoff,
    }


class AutoMutationEffect(str, Enum):
    LOCAL_DATABASE = "local_database"
    OBSIDIAN_VAULT = "obsidian_vault"


class AutoMutationReplayPolicy(str, Enum):
    COALESCE_BY_OPERATION_KEY = "coalesce_by_operation_key"


class AutoMutationCrashPolicy(str, Enum):
    STOP_AS_OUTCOME_UNKNOWN = "stop_as_outcome_unknown"


class ToolArgumentType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"
    NULL = "null"


@dataclass(frozen=True)
class ToolArgumentSpec:
    name: str
    types: frozenset[ToolArgumentType]
    required: bool = False
    minimum: int | float | None = None
    maximum: int | float | None = None


@dataclass(frozen=True)
class ToolArgumentContract:
    version: int
    fields: tuple[ToolArgumentSpec, ...]
    allow_unknown: bool = False


@dataclass(frozen=True)
class ToolArgumentValidation:
    valid: bool
    status: str
    missing_keys: tuple[str, ...] = ()
    unknown_keys: tuple[str, ...] = ()
    type_mismatch_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class AutoMutationContract:
    version: int
    effects: frozenset[AutoMutationEffect]
    replay_policy: AutoMutationReplayPolicy
    crash_policy: AutoMutationCrashPolicy
    operation_key_builder: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    execution_args_builder: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None
    execution_argument_contract: ToolArgumentContract | None = None
    semantic_preflight: Callable[[dict[str, Any]], str | None] | None = None
    semantic_preflight_result_builder: (
        Callable[[dict[str, Any], str], ToolResult] | None
    ) = None
    definite_no_effect_failure_reasons: frozenset[str] = frozenset()
    operation_scope: str | None = None
    legacy_operation_aliases: frozenset[str] = frozenset()


AUTO_MUTATION_CONTRACT_VERSION = 1
AUTO_MUTATION_META_TOOLSETS = frozenset(
    {
        "approvals",
        "audit",
        "continuity",
        "learning",
        "safety",
        "scheduler",
    }
)
AUTO_MUTATION_META_TOOL_EXCEPTIONS = frozenset(
    {("queue_learning_tasks", "learning")}
)
AUTO_MUTATION_EXTERNAL_OR_COMPUTER_TOOLSETS = frozenset(
    {"browser", "computer", "personal", "voice"}
)


ApprovalArgumentResolver = Callable[
    [dict[str, Any]], ApprovalArgumentResolution | ToolResult
]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    risk: RiskLevel
    handler: ToolHandler
    toolset: str = "core"
    auto_mutation_contract: AutoMutationContract | None = None
    argument_contract: ToolArgumentContract | None = None
    approval_argument_resolver: ApprovalArgumentResolver | None = None
    approval_argument_contract: ToolArgumentContract | None = None


TOOL_ARGUMENT_CONTRACT_VERSION = 1
MAX_TOOL_ARGUMENT_FIELDS = 64
MAX_TOOL_ARGUMENT_NAME_CHARS = 80
TOOL_ARGUMENT_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tool_argument_contract(
    *,
    required_strings: tuple[str, ...] = (),
    optional_strings: tuple[str, ...] = (),
    required_integers: tuple[str, ...] = (),
    optional_integers: tuple[str, ...] = (),
    required_numbers: tuple[str, ...] = (),
    optional_numbers: tuple[str, ...] = (),
    required_booleans: tuple[str, ...] = (),
    optional_booleans: tuple[str, ...] = (),
    required_integer_inputs: tuple[str, ...] = (),
    optional_integer_inputs: tuple[str, ...] = (),
    required_numeric_inputs: tuple[str, ...] = (),
    optional_numeric_inputs: tuple[str, ...] = (),
    required_integer_ranges: tuple[tuple[str, int, int], ...] = (),
    optional_integer_ranges: tuple[tuple[str, int, int], ...] = (),
) -> ToolArgumentContract:
    string_type = frozenset({ToolArgumentType.STRING})
    integer_type = frozenset({ToolArgumentType.INTEGER})
    number_type = frozenset({ToolArgumentType.NUMBER})
    boolean_type = frozenset({ToolArgumentType.BOOLEAN})
    integer_input_type = frozenset({ToolArgumentType.INTEGER, ToolArgumentType.STRING})
    numeric_input_type = frozenset({ToolArgumentType.NUMBER, ToolArgumentType.STRING})
    fields = tuple(
        [ToolArgumentSpec(name, string_type, True) for name in required_strings]
        + [ToolArgumentSpec(name, string_type, False) for name in optional_strings]
        + [ToolArgumentSpec(name, integer_type, True) for name in required_integers]
        + [ToolArgumentSpec(name, integer_type, False) for name in optional_integers]
        + [ToolArgumentSpec(name, number_type, True) for name in required_numbers]
        + [ToolArgumentSpec(name, number_type, False) for name in optional_numbers]
        + [ToolArgumentSpec(name, boolean_type, True) for name in required_booleans]
        + [ToolArgumentSpec(name, boolean_type, False) for name in optional_booleans]
        + [ToolArgumentSpec(name, integer_input_type, True) for name in required_integer_inputs]
        + [ToolArgumentSpec(name, integer_input_type, False) for name in optional_integer_inputs]
        + [ToolArgumentSpec(name, numeric_input_type, True) for name in required_numeric_inputs]
        + [ToolArgumentSpec(name, numeric_input_type, False) for name in optional_numeric_inputs]
        + [
            ToolArgumentSpec(name, integer_type, True, minimum, maximum)
            for name, minimum, maximum in required_integer_ranges
        ]
        + [
            ToolArgumentSpec(name, integer_type, False, minimum, maximum)
            for name, minimum, maximum in optional_integer_ranges
        ]
    )
    return ToolArgumentContract(TOOL_ARGUMENT_CONTRACT_VERSION, fields, False)


def _validate_argument_contract(tool_name: str, contract: ToolArgumentContract | None) -> None:
    if contract is None:
        return
    if not isinstance(contract, ToolArgumentContract):
        raise ValueError(f"Invalid tool-argument contract for {tool_name}: wrong contract type")
    if type(contract.version) is not int or contract.version != TOOL_ARGUMENT_CONTRACT_VERSION:
        raise ValueError(f"Invalid tool-argument contract for {tool_name}: unsupported version")
    if type(contract.allow_unknown) is not bool:
        raise ValueError(f"Invalid tool-argument contract for {tool_name}: allow_unknown must be boolean")
    if not isinstance(contract.fields, tuple) or len(contract.fields) > MAX_TOOL_ARGUMENT_FIELDS:
        raise ValueError(f"Invalid tool-argument contract for {tool_name}: fields must be a bounded tuple")
    names: set[str] = set()
    for field in contract.fields:
        if not isinstance(field, ToolArgumentSpec):
            raise ValueError(f"Invalid tool-argument contract for {tool_name}: wrong field type")
        if (
            type(field.name) is not str
            or not field.name
            or len(field.name) > MAX_TOOL_ARGUMENT_NAME_CHARS
            or TOOL_ARGUMENT_NAME_RE.fullmatch(field.name) is None
            or field.name in names
        ):
            raise ValueError(f"Invalid tool-argument contract for {tool_name}: invalid or duplicate field name")
        names.add(field.name)
        if not isinstance(field.types, frozenset) or not field.types:
            raise ValueError(f"Invalid tool-argument contract for {tool_name}: field types must be non-empty")
        if any(not isinstance(argument_type, ToolArgumentType) for argument_type in field.types):
            raise ValueError(f"Invalid tool-argument contract for {tool_name}: unknown field type")
        if type(field.required) is not bool:
            raise ValueError(f"Invalid tool-argument contract for {tool_name}: required must be boolean")
        if field.minimum is not None or field.maximum is not None:
            numeric_types = {ToolArgumentType.INTEGER, ToolArgumentType.NUMBER}
            if not field.types or not field.types.issubset(numeric_types):
                raise ValueError(
                    f"Invalid tool-argument contract for {tool_name}: bounds require numeric-only types"
                )
            for label, bound in (("minimum", field.minimum), ("maximum", field.maximum)):
                if bound is None:
                    continue
                if isinstance(bound, bool) or not isinstance(bound, (int, float)):
                    raise ValueError(
                        f"Invalid tool-argument contract for {tool_name}: {label} must be numeric"
                    )
                if isinstance(bound, float) and not math.isfinite(bound):
                    raise ValueError(
                        f"Invalid tool-argument contract for {tool_name}: {label} must be finite"
                    )
            if (
                field.minimum is not None
                and field.maximum is not None
                and field.minimum > field.maximum
            ):
                raise ValueError(
                    f"Invalid tool-argument contract for {tool_name}: minimum exceeds maximum"
                )


def _validate_tool_argument_contract(tool: Tool) -> None:
    _validate_argument_contract(tool.name, tool.argument_contract)
    _validate_argument_contract(tool.name, tool.approval_argument_contract)


def _argument_value_matches(value: Any, expected: frozenset[ToolArgumentType]) -> bool:
    if ToolArgumentType.STRING in expected and type(value) is str:
        return True
    if ToolArgumentType.INTEGER in expected and type(value) is int:
        return True
    if ToolArgumentType.NUMBER in expected and (
        type(value) is int or (type(value) is float and math.isfinite(value))
    ):
        return True
    if ToolArgumentType.BOOLEAN in expected and type(value) is bool:
        return True
    if ToolArgumentType.OBJECT in expected and type(value) is dict:
        return True
    if ToolArgumentType.ARRAY in expected and type(value) is list:
        return True
    if ToolArgumentType.NULL in expected and value is None:
        return True
    return False


def validate_tool_arguments(
    tool: Tool,
    args: Any,
    *,
    approved: bool = False,
) -> ToolArgumentValidation:
    contract = (
        tool.approval_argument_contract
        if approved and tool.approval_argument_contract is not None
        else tool.argument_contract
    )
    if contract is None:
        return ToolArgumentValidation(True, "untyped")
    if type(args) is not dict:
        return ToolArgumentValidation(False, "arguments_not_object")

    fields = {field.name: field for field in contract.fields}
    missing = tuple(sorted(field.name for field in contract.fields if field.required and field.name not in args))
    unknown: list[str] = []
    mismatched: list[str] = []
    for key, value in args.items():
        if type(key) is not str:
            unknown.append("<non-string>")
            continue
        field = fields.get(key)
        if field is None:
            if not contract.allow_unknown:
                unknown.append("<unknown>")
            continue
        if not _argument_value_matches(value, field.types):
            mismatched.append(key)
            continue
        if (
            field.minimum is not None
            and type(value) in {int, float}
            and value < field.minimum
        ) or (
            field.maximum is not None
            and type(value) in {int, float}
            and value > field.maximum
        ):
            mismatched.append(key)
    unknown_keys = tuple(sorted(set(unknown))[:MAX_TOOL_ARGUMENT_FIELDS])
    mismatch_keys = tuple(sorted(set(mismatched))[:MAX_TOOL_ARGUMENT_FIELDS])
    if not missing and not unknown_keys and not mismatch_keys:
        return ToolArgumentValidation(True, "valid")
    failures = sum(bool(items) for items in (missing, unknown_keys, mismatch_keys))
    if failures > 1:
        status = "multiple_errors"
    elif missing:
        status = "missing_required"
    elif unknown_keys:
        status = "unknown_arguments"
    else:
        status = "type_mismatch"
    return ToolArgumentValidation(False, status, missing, unknown_keys, mismatch_keys)


def tool_argument_contract_summary(tool: Tool) -> str:
    contract = tool.argument_contract
    if contract is None:
        return ""
    fields = []
    for field in contract.fields:
        types = "|".join(sorted(argument_type.value for argument_type in field.types))
        bounds = ""
        if field.minimum is not None or field.maximum is not None:
            low = "" if field.minimum is None else str(field.minimum)
            high = "" if field.maximum is None else str(field.maximum)
            bounds = f"[{low}..{high}]"
        fields.append(f"{field.name}:{types}{bounds}{'' if field.required else '?'}")
    unknown = "unknown rejected" if not contract.allow_unknown else "unknown allowed"
    return f"args {{{', '.join(fields)}}}; {unknown}"


def _validate_auto_mutation_contract(tool: Tool) -> None:
    contract = tool.auto_mutation_contract
    if contract is None:
        return
    if not isinstance(contract, AutoMutationContract):
        raise ValueError(f"Invalid auto-mutation contract for {tool.name}: wrong contract type")
    if contract.version != AUTO_MUTATION_CONTRACT_VERSION or isinstance(contract.version, bool):
        raise ValueError(f"Invalid auto-mutation contract for {tool.name}: unsupported version")
    if tool.risk is not RiskLevel.LOCAL_SAFE:
        raise ValueError(f"Invalid auto-mutation contract for {tool.name}: risk must be LOCAL_SAFE")
    if tool.toolset in AUTO_MUTATION_EXTERNAL_OR_COMPUTER_TOOLSETS:
        raise ValueError(f"Invalid auto-mutation contract for {tool.name}: external/computer toolset")
    if (
        tool.toolset in AUTO_MUTATION_META_TOOLSETS
        and (tool.name, tool.toolset) not in AUTO_MUTATION_META_TOOL_EXCEPTIONS
    ):
        raise ValueError(f"Invalid auto-mutation contract for {tool.name}: meta toolset")
    if not isinstance(contract.effects, frozenset) or not contract.effects:
        raise ValueError(f"Invalid auto-mutation contract for {tool.name}: effects must be non-empty")
    if any(not isinstance(effect, AutoMutationEffect) for effect in contract.effects):
        raise ValueError(f"Invalid auto-mutation contract for {tool.name}: unknown effect")
    if contract.replay_policy is not AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY:
        raise ValueError(f"Invalid auto-mutation contract for {tool.name}: unsupported replay policy")
    if contract.crash_policy is not AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN:
        raise ValueError(f"Invalid auto-mutation contract for {tool.name}: unsupported crash policy")
    if contract.operation_key_builder is not None and not callable(contract.operation_key_builder):
        raise ValueError(f"Invalid auto-mutation contract for {tool.name}: operation key builder must be callable")
    if contract.execution_args_builder is not None and not callable(contract.execution_args_builder):
        raise ValueError(f"Invalid auto-mutation contract for {tool.name}: execution args builder must be callable")
    _validate_argument_contract(tool.name, contract.execution_argument_contract)
    if (contract.execution_args_builder is None) != (contract.execution_argument_contract is None):
        raise ValueError(
            f"Invalid auto-mutation contract for {tool.name}: execution builder and contract must pair"
        )
    if contract.execution_argument_contract is not None:
        if contract.execution_argument_contract.allow_unknown or not contract.execution_argument_contract.fields:
            raise ValueError(
                f"Invalid auto-mutation contract for {tool.name}: execution arguments must be exact"
            )
        if any(
            not field.name.startswith("_auto_mutation_")
            for field in contract.execution_argument_contract.fields
        ):
            raise ValueError(
                f"Invalid auto-mutation contract for {tool.name}: execution argument names must be private"
            )
    if contract.operation_scope is not None and (
        type(contract.operation_scope) is not str
        or re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", contract.operation_scope) is None
    ):
        raise ValueError(
            f"Invalid auto-mutation contract for {tool.name}: invalid operation scope"
        )
    if (
        not isinstance(contract.legacy_operation_aliases, frozenset)
        or len(contract.legacy_operation_aliases) > 32
        or any(
            type(alias) is not str
            or re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", alias) is None
            for alias in contract.legacy_operation_aliases
        )
    ):
        raise ValueError(
            f"Invalid auto-mutation contract for {tool.name}: invalid legacy operation aliases"
        )
    if contract.legacy_operation_aliases and (
        contract.operation_scope is None
        or tool.name not in contract.legacy_operation_aliases
    ):
        raise ValueError(
            f"Invalid auto-mutation contract for {tool.name}: legacy aliases require a shared scope "
            "and must include the current tool"
        )
    if contract.semantic_preflight is not None and not callable(contract.semantic_preflight):
        raise ValueError(f"Invalid auto-mutation contract for {tool.name}: semantic preflight must be callable")
    if contract.semantic_preflight_result_builder is not None and not callable(
        contract.semantic_preflight_result_builder
    ):
        raise ValueError(
            f"Invalid auto-mutation contract for {tool.name}: semantic preflight result builder must be callable"
        )
    if contract.semantic_preflight_result_builder is not None and contract.semantic_preflight is None:
        raise ValueError(
            f"Invalid auto-mutation contract for {tool.name}: semantic result builder requires preflight"
        )
    if not isinstance(contract.definite_no_effect_failure_reasons, frozenset) or any(
        not isinstance(reason, str)
        or re.fullmatch(r"[A-Za-z0-9_:-]{1,80}", reason) is None
        for reason in contract.definite_no_effect_failure_reasons
    ):
        raise ValueError(
            f"Invalid auto-mutation contract for {tool.name}: bad definite-no-effect reasons"
        )


def _validate_approval_argument_resolver(tool: Tool) -> None:
    resolver = tool.approval_argument_resolver
    approval_contract = tool.approval_argument_contract
    if resolver is None and approval_contract is None:
        return
    if resolver is None or approval_contract is None:
        raise ValueError(
            f"Invalid approval argument resolver for {tool.name}: resolver and approval contract must appear together"
        )
    if not callable(resolver):
        raise ValueError(f"Invalid approval argument resolver for {tool.name}: resolver must be callable")
    if tool.risk <= RiskLevel.LOCAL_SAFE:
        raise ValueError(
            f"Invalid approval argument resolver for {tool.name}: risk must require approval"
        )


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        if type(tool.risk) is not RiskLevel:
            raise ValueError(f"Invalid risk level for {tool.name}: exact RiskLevel required")
        _validate_auto_mutation_contract(tool)
        _validate_tool_argument_contract(tool)
        _validate_approval_argument_resolver(tool)
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def list(self, toolset: str | None = None) -> list[Tool]:
        readable: list[tuple[str, Tool]] = []
        for tool in self._tools.values():
            try:
                name = str(tool.name)
            except Exception:
                continue
            if toolset:
                try:
                    if tool.toolset != toolset:
                        continue
                except Exception:
                    continue
            readable.append((name, tool))
        return [tool for _, tool in sorted(readable, key=lambda item: item[0])]


def _registry_list_views(registry: ToolRegistry, toolset: str | None = None) -> tuple[list[dict[str, str]], int]:
    """Best-effort display rows; malformed tools must not disappear silently."""
    unreadable = 0
    raw_tools = getattr(registry, "_tools", None)
    if isinstance(raw_tools, dict):
        try:
            tools = list(raw_tools.values())
        except Exception:
            tools = []
            unreadable += 1
    else:
        try:
            tools = registry.list(toolset)
        except Exception:
            return [], 1

    views: list[dict[str, str]] = []
    for tool in tools:
        try:
            name = _short_metadata(tool.name, limit=MAX_CORE_FIELD_CHARS)
            tool_toolset = _short_metadata(tool.toolset, limit=80)
            risk = _short_metadata(tool.risk.name, limit=80)
            description = _short_metadata(tool.description, limit=MAX_CORE_TEXT_CHARS)
        except Exception:
            unreadable += 1
            continue
        if not name or not tool_toolset or not risk:
            unreadable += 1
            continue
        if toolset and tool_toolset != toolset:
            continue
        views.append(
            {
                "name": name,
                "toolset": tool_toolset,
                "risk": risk,
                "description": description,
            }
        )
    return sorted(views, key=lambda item: item["name"]), unreadable


def build_core_registry(
    store: MemoryStore,
    vault: ObsidianVault,
    config: JarvisConfig | None = None,
    session_id: str = "",
    storage_fallback: Callable[[], dict[str, Any] | None] | None = None,
    subagent_fleet: Callable[[], Any] | None = None,
    chat_brain: Callable[[], Any] | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()

    def current_time(args: dict[str, Any]) -> ToolResult:
        location = args.get("location") or args.get("city") or args.get("timezone") or ""
        show_timezone = bool(args.get("show_timezone"))
        locale = str(args.get("locale") or "").strip().lower()
        korean = locale == "ko"
        resolved = _resolve_timezone_location(location)
        if location and not resolved:
            output = (
                "I don't know that timezone yet. Try a common city like Seoul, Tokyo, "
                f"New York, London, or UTC. {CORE_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "current_time",
                False,
                output,
                _core_input_failure_metadata(
                    output,
                    time_status="unknown_timezone",
                    timezone_lookup_supported=False,
                ),
            )
        if resolved:
            timezone, label = resolved
            try:
                now = datetime.now(ZoneInfo(timezone))
            except ZoneInfoNotFoundError:
                output = (
                    "I couldn't load that timezone locally. "
                    f"{_timezone_database_recovery_guidance()}"
                )
                return ToolResult(
                    "current_time",
                    False,
                    output,
                    _timezone_database_failure_metadata(
                        output,
                        time_status="timezone_unavailable",
                        timezone=timezone,
                        location=label,
                    ),
                )
            if korean:
                localized_label = KOREAN_LOCATION_LABELS.get(label, label)
                output = f"{localized_label}의 현재 시간은 {_format_korean_datetime(now)}입니다. ({now.tzname()}, {timezone})"
            else:
                output = f"{label}: {now.strftime('%A, %B %d, %Y at %I:%M %p')} ({now.tzname()}, {timezone})"
            return ToolResult(
                "current_time",
                True,
                output,
                _safe_metadata(time_status="ok", timezone=timezone, location=label, timezone_requested=show_timezone, locale="ko" if korean else "en"),
            )
        if show_timezone:
            now = datetime.now().astimezone()
            tzinfo = now.tzinfo
            abbreviation = now.tzname() or "local"
            timezone_name = getattr(tzinfo, "key", "") or abbreviation
            if korean:
                output = f"현재 시간대: {abbreviation} ({timezone_name}). 현재 현지 시간은 {_format_korean_datetime(now)}입니다."
            else:
                output = f"Local timezone: {abbreviation} ({timezone_name}). Current local time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"
            return ToolResult(
                "current_time",
                True,
                output,
                _safe_metadata(time_status="ok", timezone=timezone_name, timezone_abbreviation=abbreviation, timezone_requested=True, locale="ko" if korean else "en"),
            )
        now = datetime.now()
        output = f"현재 시간은 {_format_korean_datetime(now)}입니다." if korean else now.strftime("%A, %B %d, %Y at %I:%M %p")
        return ToolResult("current_time", True, output, _safe_metadata(time_status="ok", locale="ko" if korean else "en"))

    def time_difference(args: dict[str, Any]) -> ToolResult:
        source_raw = args.get("source") or args.get("from") or args.get("left") or ""
        target_raw = args.get("target") or args.get("to") or args.get("right") or ""
        source = _resolve_timezone_location(source_raw)
        target = _resolve_timezone_location(target_raw)
        if not source or not target:
            output = (
                "I don't know one of those timezones yet. Try common cities like Seoul, "
                f"Tokyo, New York, London, or UTC. {CORE_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "time_difference",
                False,
                output,
                _core_input_failure_metadata(
                    output,
                    time_difference_status="unknown_timezone",
                    source_lookup_supported=bool(source),
                    target_lookup_supported=bool(target),
                    source=_short_metadata(source_raw),
                    target=_short_metadata(target_raw),
                ),
            )
        source_timezone, source_label = source
        target_timezone, target_label = target
        try:
            source_now = datetime.now(ZoneInfo(source_timezone))
            target_now = datetime.now(ZoneInfo(target_timezone))
        except ZoneInfoNotFoundError:
            output = (
                "I couldn't load one of those timezones locally. "
                f"{_timezone_database_recovery_guidance()}"
            )
            return ToolResult(
                "time_difference",
                False,
                output,
                _timezone_database_failure_metadata(
                    output,
                    time_difference_status="timezone_unavailable",
                    source_timezone=source_timezone,
                    target_timezone=target_timezone,
                    source=source_label,
                    target=target_label,
                ),
            )
        source_offset = source_now.utcoffset()
        target_offset = target_now.utcoffset()
        if source_offset is None or target_offset is None:
            output = (
                "I couldn't calculate one of those timezone offsets locally. "
                f"{_timezone_database_recovery_guidance()}"
            )
            return ToolResult(
                "time_difference",
                False,
                output,
                _timezone_database_failure_metadata(
                    output,
                    time_difference_status="offset_unavailable",
                    source_timezone=source_timezone,
                    target_timezone=target_timezone,
                    source=source_label,
                    target=target_label,
                ),
            )
        difference_seconds = int((source_offset - target_offset).total_seconds())
        absolute_seconds = abs(difference_seconds)
        hours, remainder = divmod(absolute_seconds, 3600)
        minutes = remainder // 60
        if minutes:
            amount = f"{hours} hour{'s' if hours != 1 else ''} {minutes} minute{'s' if minutes != 1 else ''}"
        else:
            amount = f"{hours} hour{'s' if hours != 1 else ''}"
        if difference_seconds > 0:
            relation = f"{source_label} is {amount} ahead of {target_label} right now."
        elif difference_seconds < 0:
            relation = f"{source_label} is {amount} behind {target_label} right now."
        else:
            relation = f"{source_label} and {target_label} have the same UTC offset right now."
        output = (
            f"{relation}\n"
            f"{source_label}: {source_now.strftime('%A, %B %d, %Y at %I:%M %p')} ({source_now.tzname()}, {source_timezone})\n"
            f"{target_label}: {target_now.strftime('%A, %B %d, %Y at %I:%M %p')} ({target_now.tzname()}, {target_timezone})"
        )
        return ToolResult(
            "time_difference",
            True,
            output,
            _safe_metadata(
                time_difference_status="ok",
                source=source_label,
                target=target_label,
                source_timezone=source_timezone,
                target_timezone=target_timezone,
                source_offset_seconds=int(source_offset.total_seconds()),
                target_offset_seconds=int(target_offset.total_seconds()),
                difference_seconds=difference_seconds,
                difference_hours=difference_seconds / 3600,
            ),
        )

    def relative_date(args: dict[str, Any]) -> ToolResult:
        target_raw = args.get("target") or args.get("date") or args.get("day") or ""
        parsed = _parse_relative_date(target_raw)
        if not parsed:
            output = (
                "I can answer local date math for tomorrow, yesterday, the day after "
                "tomorrow, in two days, or next/this/last weekdays. "
                f"{CORE_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "relative_date",
                False,
                output,
                _core_input_failure_metadata(
                    output,
                    relative_date_status="unsupported_target",
                    target=_short_metadata(target_raw),
                    local_path_target=bool(LOCAL_PATH_RE.search(str(target_raw or ""))),
                ),
            )
        label, target, offset = parsed
        output = f"{label} is {target.strftime('%A, %B %d, %Y')}."
        return ToolResult(
            "relative_date",
            True,
            output,
            _safe_metadata(
                relative_date_status="ok",
                target=_short_metadata(target_raw),
                label=label,
                target_date=target.date().isoformat(),
                weekday=target.strftime("%A"),
                day_offset=offset,
                local_path_target=False,
            ),
        )

    def remember(args: dict[str, Any]) -> ToolResult:
        operation_key = remember_auto_mutation_operation_key(args)
        category = operation_key["category"]
        title = operation_key["title"]
        body = operation_key["body"]
        if not body.strip():
            output = f"Nothing to remember. {CORE_INPUT_RECOVERY_ACTION}"
            return ToolResult(
                "remember",
                False,
                output,
                _core_input_failure_metadata(
                    output,
                    writes_files=False,
                    writes_memory=False,
                    writes_notes=False,
                ),
            )
        if LOCAL_PATH_RE.search(category):
            output = (
                "Memory category should describe a topic, not a local file path. "
                f"{CORE_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "remember",
                False,
                output,
                _core_input_failure_metadata(
                    output,
                    reason="invalid_category",
                    raw_category=_short_metadata(args.get("category") or "facts", limit=80),
                ),
            )
        if LOCAL_PATH_RE.search(title):
            output = (
                "Memory title should describe the memory, not a local file path. "
                f"{CORE_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "remember",
                False,
                output,
                _core_input_failure_metadata(
                    output,
                    reason="invalid_title",
                    raw_title=_short_metadata(args.get("title") or "Untitled"),
                ),
            )
        if LOCAL_PATH_RE.search(body):
            output = (
                "Memory text should not contain local file paths. "
                f"{CORE_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "remember",
                False,
                output,
                _core_input_failure_metadata(
                    output,
                    reason="invalid_body",
                    raw_body=_short_metadata(args.get("body"), limit=80),
                ),
            )
        record = MemoryRecord(category=category, title=title, body=body, source="jarvis-v2")
        projection_target = store.add_memory_with_projection(record)
        memory_id = projection_target.memory_id
        projection = reconcile_memory_projection(
            store,
            vault,
            memory_id,
            expected_operation=projection_target.operation,
            expected_revision=projection_target.revision,
            expected_source_digest=projection_target.source_digest,
        )
        if projection.status != "completed":
            output = (
                f"Memory #{memory_id} was saved, but its note projection is "
                f"{projection.status}. {MEMORY_PROJECTION_RECOVERY_ACTION}"
            )
            return ToolResult(
                "remember",
                False,
                output,
                _memory_projection_failure_metadata(
                    output,
                    memory_id=memory_id,
                    projection_status=projection.status,
                    projection_pending=True,
                    writes_files=True,
                    writes_memory=True,
                    writes_notes=True,
                ),
            )
        note_path = vault.root_path / projection.path_display
        write_handoff = _memory_write_handoff(record, memory_id=memory_id, note_path=note_path, vault=vault)
        return ToolResult(
            "remember",
            True,
            f"Remembered '{title}' in {category}.",
            _safe_metadata(
                memory_id=memory_id,
                note_path=str(note_path),
                path_display=write_handoff["path_display"],
                memory_write_handoff=write_handoff,
                writes_files=True,
                writes_memory=True,
                writes_notes=True,
            ),
        )

    def repair_memory_projections(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 20, high=100)
        profile_summary = reconcile_pending_profile_projections(
            store,
            vault,
            limit=limit,
        )
        goal_summary = reconcile_pending_goal_projections(
            store,
            vault,
            limit=limit,
        )
        summary = reconcile_pending_memory_projections(store, vault, limit=limit)
        attempted = profile_summary.attempted + goal_summary.attempted + summary.attempted
        completed = profile_summary.completed + goal_summary.completed + summary.completed
        pending = profile_summary.pending + goal_summary.pending + summary.pending
        outcome_statuses = [
            *(f"profile:{outcome.status}" for outcome in profile_summary.outcomes),
            *(f"goal:{outcome.status}" for outcome in goal_summary.outcomes),
            *(outcome.status for outcome in summary.outcomes),
        ]
        complete = pending == 0
        output = (
            f"Memory projection repair attempted {attempted} job(s); "
            f"completed {completed}; pending {pending}."
        )
        if not complete:
            output += " Pending jobs remain; no source memory mutation was retried."
        return ToolResult(
            "repair_memory_projections",
            complete,
            output,
            _safe_metadata(
                attempted=attempted,
                completed=completed,
                pending=pending,
                outcome_statuses=outcome_statuses,
                writes_files=attempted > 0,
                writes_database=attempted > 0,
                writes_memory=False,
                writes_notes=attempted > 0,
            ),
        )

    def repair_person_projections(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 20, high=100)
        summary = reconcile_pending_person_projections(store, vault, limit=limit)
        outcome_statuses = [outcome.status for outcome in summary.outcomes]
        complete = summary.pending == 0
        output = (
            f"Person projection repair attempted {summary.attempted} job(s); "
            f"completed {summary.completed}; pending {summary.pending}."
        )
        if not complete:
            output += " Pending jobs remain; no person or interaction mutation was retried."
        return ToolResult(
            "repair_person_projections",
            complete,
            output,
            _safe_metadata(
                attempted=summary.attempted,
                completed=summary.completed,
                pending=summary.pending,
                outcome_statuses=outcome_statuses,
                writes_files=summary.attempted > 0,
                writes_memory=False,
                writes_notes=summary.attempted > 0,
            ),
        )

    def repair_decision_projections(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 20, high=100)
        summary = reconcile_pending_decision_projections(store, vault, limit=limit)
        outcome_statuses = [outcome.status for outcome in summary.outcomes]
        complete = summary.pending == 0
        output = (
            f"Decision projection repair attempted {summary.attempted} job(s); "
            f"completed {summary.completed}; pending {summary.pending}; "
            f"legacy custody backfilled {summary.custody_backfilled}."
        )
        if not complete:
            output += " Pending work remains; no decision source mutation was retried."
        return ToolResult(
            "repair_decision_projections",
            complete,
            output,
            _safe_metadata(
                attempted=summary.attempted,
                completed=summary.completed,
                pending=summary.pending,
                custody_backfilled=summary.custody_backfilled,
                outcome_statuses=outcome_statuses,
                writes_files=summary.attempted > 0,
                writes_database=summary.attempted > 0 or summary.custody_backfilled > 0,
                writes_memory=summary.custody_backfilled > 0,
                writes_notes=summary.attempted > 0,
            ),
        )

    def repair_preference_projections(args: dict[str, Any]) -> ToolResult:
        summary = reconcile_pending_preference_projections(store, vault)
        outcome_statuses = [outcome.status for outcome in summary.outcomes]
        complete = summary.pending == 0
        output = (
            f"Preference projection repair attempted {summary.attempted} job(s); "
            f"completed {summary.completed}; pending {summary.pending}; "
            f"legacy custody backfilled {summary.custody_backfilled}; "
            f"identity conflicts {summary.custody_conflicts}."
        )
        if not complete:
            output += " Pending work remains; no preference source mutation was retried."
        return ToolResult(
            "repair_preference_projections",
            complete,
            output,
            _safe_metadata(
                attempted=summary.attempted,
                completed=summary.completed,
                pending=summary.pending,
                custody_backfilled=summary.custody_backfilled,
                custody_conflicts=summary.custody_conflicts,
                outcome_statuses=outcome_statuses,
                writes_files=summary.attempted > 0,
                writes_database=summary.attempted > 0 or summary.custody_backfilled > 0,
                writes_memory=summary.custody_backfilled > 0,
                writes_notes=summary.attempted > 0,
            ),
        )

    def search_memory(args: dict[str, Any]) -> ToolResult:
        query = _short(args.get("query"), limit=MAX_CORE_TEXT_CHARS)
        if not query:
            output = f"Search query is empty. {CORE_INPUT_RECOVERY_ACTION}"
            return ToolResult(
                "search_memory",
                False,
                output,
                _core_input_failure_metadata(output, query=""),
            )
        limit = _bounded_int(args.get("limit", 5), 5)
        rows = store.search_memories(query, limit)
        handoff = _memory_rows_handoff(rows, source="search_memory", query=query, limit=limit)
        if not rows:
            return ToolResult(
                "search_memory",
                True,
                f"No memories found for '{query}'.",
                _safe_metadata(count=0, query=query, limit=limit, memory_search_handoff=handoff),
            )
        lines = [f"- [{_short(row['category'], limit=80)}] {_short(row['title'])}: {_short(row['body'], limit=180)}" for row in rows]
        return ToolResult(
            "search_memory",
            True,
            "\n".join(lines),
            _safe_metadata(count=len(rows), query=query, limit=limit, memory_search_handoff=handoff),
        )

    def recent_memories(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit", 10), 10)
        rows = store.recent_memories(limit)
        handoff = _memory_rows_handoff(rows, source="recent_memories", limit=limit)
        if not rows:
            return ToolResult(
                "recent_memories",
                True,
                "No memories yet.",
                _safe_metadata(count=0, limit=limit, recent_memories_handoff=handoff),
            )
        lines = [f"- [{_short(row['category'], limit=80)}] {_short(row['title'])}: {_short(row['body'], limit=140)}" for row in rows]
        return ToolResult(
            "recent_memories",
            True,
            "\n".join(lines),
            _safe_metadata(count=len(rows), limit=limit, recent_memories_handoff=handoff),
        )

    def write_daily_note(args: dict[str, Any]) -> ToolResult:
        heading = _short(args.get("heading") or "Jarvis", limit=MAX_CORE_FIELD_CHARS)
        body = _short(args.get("body"), limit=MAX_CORE_TEXT_CHARS)
        if not body.strip():
            output = f"Daily note body is empty. {CORE_INPUT_RECOVERY_ACTION}"
            return ToolResult(
                "write_daily_note",
                False,
                output,
                _core_input_failure_metadata(output),
            )
        path = vault.append_daily(heading, body)
        path_display = _safe_vault_path_display(path, vault)
        daily_handoff = _daily_note_handoff(heading=heading, body=body, path=path, vault=vault)
        return ToolResult(
            "write_daily_note",
            True,
            f"Daily note updated: {path_display}",
            _safe_metadata(
                path=str(path),
                path_display=path_display,
                daily_note_handoff=daily_handoff,
                writes_files=True,
                writes_memory=True,
                writes_notes=True,
            ),
        )

    def respond(args: dict[str, Any]) -> ToolResult:
        text = _short(args.get("text"), limit=MAX_CORE_TEXT_CHARS)
        return ToolResult("respond", bool(text.strip()), text or "No response text provided.", _safe_metadata())

    def list_tools(args: dict[str, Any]) -> ToolResult:
        toolset = _short(args.get("toolset"), limit=80) or None
        tools, unreadable_registry_tools = _registry_list_views(registry, toolset)
        if not tools:
            note = ""
            if unreadable_registry_tools:
                note = f" {unreadable_registry_tools} unreadable registry tool(s) hidden for safety."
            return ToolResult(
                "list_tools",
                True,
                f"No tools found for toolset: {toolset}.{note}",
                _safe_metadata(
                    count=0,
                    toolset=toolset or "all",
                    unreadable_registry_tools=unreadable_registry_tools,
                ),
            )
        lines = [
            f"- {tool['name']} [{tool['toolset']}, {tool['risk']}]: {tool['description']}"
            for tool in tools
        ]
        if unreadable_registry_tools:
            lines.append(f"Note: {unreadable_registry_tools} unreadable registry tool(s) hidden for safety.")
        label = f" in {toolset}" if toolset else ""
        return ToolResult(
            "list_tools",
            True,
            f"Available tools{label}:\n" + "\n".join(lines),
            _safe_metadata(
                count=len(tools),
                toolset=toolset or "all",
                unreadable_registry_tools=unreadable_registry_tools,
            ),
        )

    def jarvis_status(_: dict[str, Any]) -> ToolResult:
        memories = store.recent_memories(1)
        skills_rows = store.list_skills(1000)
        goals_rows = store.list_goals(limit=1000)
        sessions = store.list_sessions(5)
        jobs = store.list_jobs()
        subagent_status = subagents.subagent_fleet_status_snapshot(subagent_fleet)
        active_goals = [row for row in goals_rows if row["status"] == "active"]
        enabled_jobs = [row for row in jobs if row["enabled"]]
        project_root = Path(__file__).resolve().parents[2]
        dashboard_launcher = project_root / "launch_jarvis_v3_dashboard.py"
        project_root_display = _short_metadata(project_root)
        dashboard_launcher_display = _short_metadata(dashboard_launcher)
        obsidian_root_display = _short_metadata(vault.root_path)
        subagent_line = (
            f"- internal worker fleet: {int(subagent_status.get('ready_agents') or 0)} / "
            f"{int(subagent_status.get('total_agents') or 0)} ready"
            if subagent_status.get("ok")
            else (
                f"- internal worker fleet: unavailable ({subagent_status.get('diagnostic') or 'unknown'}); "
                "run `setup check`, then `jarvis status`; if it remains unavailable, restart Jarvis with "
                "its normal launcher and retry `subagent fleet status`"
            )
        )
        lines = [
            "Jarvis status:",
            "- active project: Jarvis V3",
            f"- project root: {project_root_display}",
            f"- dashboard launcher: {dashboard_launcher_display}",
            f"- dashboard command: `{V3_DASHBOARD_COMMAND}`",
            f"- ask-Jarvis dashboard command: `{V3_DASHBOARD_INFO_COMMAND}`",
            f"- memories: {'online' if memories else 'empty'}",
            f"- skills: {len(skills_rows)} saved",
            f"- goals: {len(active_goals)} active / {len(goals_rows)} total",
            f"- sessions indexed: {len(sessions)} recent sessions visible",
            f"- scheduled jobs: {len(enabled_jobs)} enabled / {len(jobs)} total",
            subagent_line,
            f"- Obsidian root: {obsidian_root_display}",
        ]
        return ToolResult(
            "jarvis_status",
            True,
            "\n".join(lines),
            _safe_metadata(
                project_name="Jarvis V3",
                project_root=project_root_display,
                dashboard_launcher=dashboard_launcher_display,
                dashboard_launcher_exists=dashboard_launcher.exists(),
                dashboard_launch_command=V3_DASHBOARD_COMMAND,
                dashboard_ask_command=V3_DASHBOARD_INFO_COMMAND,
                obsidian_root=obsidian_root_display,
                path_metadata_redacted=True,
                memories=bool(memories),
                skills=len(skills_rows),
                active_goals=len(active_goals),
                total_goals=len(goals_rows),
                sessions=len(sessions),
                enabled_jobs=len(enabled_jobs),
                jobs=len(jobs),
                subagent_fleet=subagent_status,
                ready_agents=int(subagent_status.get("ready_agents") or 0),
                total_agents=int(subagent_status.get("total_agents") or 0),
            ),
        )

    def status_dashboard(_: dict[str, Any]) -> ToolResult:
        from jarvis_v2.ui.status_config import status_base_url

        base_url = status_base_url()
        output = (
            "Jarvis has a read-only local status dashboard.\n"
            "Run it with:\n"
            f"{V3_DASHBOARD_COMMAND}\n\n"
            "Equivalent module command:\n"
            f"{V3_DASHBOARD_MODULE_COMMAND}\n\n"
            "Required .env authentication: set JARVIS_STATUS_AUTH_TOKEN to a unique "
            "printable 32-256 character secret. Sign in as username jarvis with that token "
            "as the password.\n"
            "Optional .env overrides: JARVIS_STATUS_HOST and JARVIS_STATUS_PORT.\n\n"
            "Then open:\n"
            f"{base_url}\n\n"
            "JSON endpoint:\n"
            f"{base_url}/api/status"
        )
        return ToolResult("status_dashboard", True, output, _safe_metadata())

    registry.register(
        Tool(
            "current_time",
            "Get the current local date and time.",
            RiskLevel.READ_ONLY,
            current_time,
            argument_contract=_tool_argument_contract(
                optional_strings=("location", "city", "timezone", "locale"),
                optional_booleans=("show_timezone",),
            ),
        )
    )
    registry.register(
        Tool(
            "time_difference",
            "Compare the current UTC offset between two supported locations.",
            RiskLevel.READ_ONLY,
            time_difference,
            argument_contract=_tool_argument_contract(
                optional_strings=("source", "from", "left", "target", "to", "right"),
            ),
        )
    )
    registry.register(
        Tool(
            "relative_date",
            "Answer simple local relative-date questions such as tomorrow, in two days, or next Friday.",
            RiskLevel.READ_ONLY,
            relative_date,
            argument_contract=_tool_argument_contract(optional_strings=("target", "date", "day")),
        )
    )
    registry.register(
        Tool(
            "remember",
            "Store a memory in SQLite and Obsidian.",
            RiskLevel.LOCAL_SAFE,
            remember,
            auto_mutation_contract=AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=remember_auto_mutation_operation_key,
            ),
            argument_contract=_tool_argument_contract(
                required_strings=("body",),
                optional_strings=("category", "title"),
            ),
        )
    )
    registry.register(
        Tool(
            "search_memory",
            "Search Jarvis memories.",
            RiskLevel.READ_ONLY,
            search_memory,
            argument_contract=_tool_argument_contract(
                required_strings=("query",),
                optional_integers=("limit",),
            ),
        )
    )
    registry.register(
        Tool(
            "repair_memory_projections",
            "Repair pending owned memory-note projections without repeating source mutations.",
            RiskLevel.LOCAL_SAFE,
            repair_memory_projections,
            argument_contract=_tool_argument_contract(
                optional_integers=("limit",),
            ),
        )
    )
    registry.register(
        Tool(
            "repair_person_projections",
            "Repair pending owned person-note projections without repeating interactions.",
            RiskLevel.LOCAL_SAFE,
            repair_person_projections,
            argument_contract=_tool_argument_contract(optional_integers=("limit",)),
        )
    )
    registry.register(
        Tool(
            "repair_decision_projections",
            "Repair pending decision custody and owned notes without changing source decisions.",
            RiskLevel.LOCAL_SAFE,
            repair_decision_projections,
            argument_contract=_tool_argument_contract(optional_integers=("limit",)),
        )
    )
    registry.register(
        Tool(
            "repair_preference_projections",
            "Repair pending preference custody and the owned note without changing source preferences.",
            RiskLevel.LOCAL_SAFE,
            repair_preference_projections,
            argument_contract=_tool_argument_contract(),
        )
    )
    registry.register(
        Tool(
            "recent_memories",
            "List recent Jarvis memories.",
            RiskLevel.READ_ONLY,
            recent_memories,
            argument_contract=_tool_argument_contract(optional_integers=("limit",)),
        )
    )
    brain_search, brain_think, brain_graph, brain_neighbors = brain.make_brain_tools(store)
    registry.register(
        Tool(
            "brain_search",
            "Search Jarvis memory with GBrain-style citations and raw local evidence.",
            RiskLevel.READ_ONLY,
            brain_search,
            "brain",
            argument_contract=_tool_argument_contract(
                required_strings=("query",),
                optional_integers=("limit",),
            ),
        )
    )
    registry.register(Tool("brain_think", "Synthesize a GBrain-style answer from local Jarvis memory with citations and gap analysis.", RiskLevel.READ_ONLY, brain_think, "brain"))
    registry.register(Tool("brain_graph", "Preview a local GBrain-style knowledge graph from memories, categories, and people.", RiskLevel.READ_ONLY, brain_graph, "brain"))
    registry.register(Tool("brain_neighbors", "Find read-only related local memories for one cited memory.", RiskLevel.READ_ONLY, brain_neighbors, "brain"))
    (
        record_feedback,
        feedback_report,
        save_feedback_report,
        feedback_actions,
        save_feedback_actions,
        failure_to_test_preview,
        repeated_failure_clusters,
        failure_promotion_packet,
        failure_implementation_packet,
        failure_apply_contract,
        failure_learning_cockpit,
        failure_patch_receipt_packet,
        failure_patch_application_bridge,
        failure_patch_completion_gate,
        failure_patch_handoff_packet,
        failure_patch_closeout_packet,
        failure_learning_record_packet,
        failure_learning_closure_ledger,
    ) = feedback.make_feedback_tools(store, vault)
    registry.register(
        Tool(
            "record_feedback",
            "Record reviewable user feedback about Jarvis behavior.",
            RiskLevel.LOCAL_SAFE,
            record_feedback,
            "feedback",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
            ),
            _tool_argument_contract(
                required_strings=("body",),
                optional_strings=("theme", "title"),
            ),
        )
    )
    registry.register(
        Tool(
            "feedback_report",
            "Summarize recent Jarvis feedback and safe improvement paths.",
            RiskLevel.READ_ONLY,
            feedback_report,
            "feedback",
            argument_contract=_tool_argument_contract(
                optional_integer_ranges=(("limit", 1, 200),),
            ),
        )
    )
    registry.register(
        Tool(
            "save_feedback_report",
            "Write the Jarvis feedback report to Obsidian Automations.",
            RiskLevel.LOCAL_SAFE,
            save_feedback_report,
            "feedback",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=feedback.save_feedback_report_auto_mutation_operation_key,
            ),
            _tool_argument_contract(optional_integers=("limit",)),
        )
    )
    registry.register(
        Tool(
            "feedback_actions",
            "Suggest reviewable preference, skill, test, or code follow-ups from Jarvis feedback.",
            RiskLevel.READ_ONLY,
            feedback_actions,
            "feedback",
            argument_contract=_tool_argument_contract(
                optional_integer_ranges=(("limit", 1, 200),),
            ),
        )
    )
    registry.register(
        Tool(
            "save_feedback_actions",
            "Write feedback-driven improvement suggestions to Obsidian Automations.",
            RiskLevel.LOCAL_SAFE,
            save_feedback_actions,
            "feedback",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=feedback.save_feedback_actions_auto_mutation_operation_key,
            ),
            _tool_argument_contract(optional_integers=("limit",)),
        )
    )
    registry.register(Tool("failure_to_test_preview", "Turn a Jarvis miss into a read-only regression test preview and learning follow-up.", RiskLevel.READ_ONLY, failure_to_test_preview, "learning"))
    registry.register(Tool("repeated_failure_clusters", "Group recent Jarvis misses into read-only regression-test and learning candidates.", RiskLevel.READ_ONLY, repeated_failure_clusters, "learning"))
    registry.register(Tool("failure_promotion_packet", "Draft a read-only promotion plan from a repeated failure cluster to a real test or learning artifact.", RiskLevel.READ_ONLY, failure_promotion_packet, "learning"))
    registry.register(Tool("failure_implementation_packet", "Draft an implementation-ready read-only work packet from a repeated failure cluster.", RiskLevel.READ_ONLY, failure_implementation_packet, "learning"))
    registry.register(Tool("failure_apply_contract", "Draft a read-only last-look contract before applying a repeated-failure patch.", RiskLevel.READ_ONLY, failure_apply_contract, "learning"))
    registry.register(Tool("failure_learning_cockpit", "Consolidate repeated failure clusters, promotion, implementation, apply contract, blockers, and proof queue before patch review.", RiskLevel.READ_ONLY, failure_learning_cockpit, "learning"))
    registry.register(Tool("failure_patch_receipt_packet", "Check post-patch repeated-failure evidence before completion review.", RiskLevel.READ_ONLY, failure_patch_receipt_packet, "learning"))
    registry.register(Tool("failure_patch_application_bridge", "Bind an applied repeated-failure patch back to its reviewed apply contract, receipt, verification, compile pass, and rollback before completion review.", RiskLevel.READ_ONLY, failure_patch_application_bridge, "learning"))
    registry.register(Tool("failure_patch_completion_gate", "Gate repeated-failure patch completion claims on cockpit, apply-contract, receipt, verification, compile, and rollback evidence.", RiskLevel.READ_ONLY, failure_patch_completion_gate, "learning"))
    registry.register(Tool("failure_patch_handoff_packet", "Package repeated-failure patch proof for completion review after the patch completion gate.", RiskLevel.READ_ONLY, failure_patch_handoff_packet, "learning"))
    registry.register(Tool("failure_patch_closeout_packet", "Close repeated-failure patch review after handoff, audit, evidence ledger, claim gate, and post-claim review evidence.", RiskLevel.READ_ONLY, failure_patch_closeout_packet, "learning"))
    registry.register(Tool("failure_learning_record_packet", "Verify repeated-failure patch closeout has durable learning-record evidence before treating learning as recorded.", RiskLevel.READ_ONLY, failure_learning_record_packet, "learning"))
    registry.register(Tool("failure_learning_closure_ledger", "Bind repeated-failure cockpit, receipt, completion, handoff, closeout, and learning-record proof into one read-only closure ledger.", RiskLevel.READ_ONLY, failure_learning_closure_ledger, "learning"))
    registry.register(
        Tool(
            "write_daily_note",
            "Append an entry to today's Obsidian daily note.",
            RiskLevel.LOCAL_SAFE,
            write_daily_note,
            auto_mutation_contract=AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
            ),
            argument_contract=_tool_argument_contract(
                required_strings=("body",),
                optional_strings=("heading",),
            ),
        )
    )
    registry.register(Tool("respond", "Return a direct response to the user.", RiskLevel.READ_ONLY, respond))
    registry.register(Tool("jarvis_help", "Show concise command examples by topic.", RiskLevel.READ_ONLY, help.jarvis_help))
    registry.register(Tool("list_tools", "List registered Jarvis tools and risk levels.", RiskLevel.READ_ONLY, list_tools))
    registry.register(Tool("architecture_map", "Map Jarvis V2 onto the six core assistant architecture layers.", RiskLevel.READ_ONLY, architecture.architecture_map))
    priority_goal, harness_status, harness_cycle_preview, harness_lifecycle_state, harness_control_surface, harness_operations_brief, agi_gate_report, agi_next_build_move, harness_completion_assessment, harness_doctrine, coding_discipline_packet, completion_audit_packet, evidence_ledger, completion_claim_gate, completion_next_proof_packet, completion_proof_refresh_packet, operator_handoff_packet, harness_readiness_digest, execution_proof_bundle, execution_mission_control, execution_case_handoff_packet, save_execution_case, inspect_execution_case, execution_case_evidence_packet, append_execution_case_evidence, execution_case_gate, execution_case_review_packet, execution_case_closure_packet, execution_case_timeline, execution_runbook = harness.make_harness_tools(store, vault, registry.list, storage_fallback, config)
    registry.register(Tool("priority_goal", "Show the top Jarvis build goal and how it should steer future work.", RiskLevel.READ_ONLY, priority_goal))
    registry.register(Tool("harness_status", "Show Jarvis as an AI agent harness progressing toward AGI-like assistant behavior.", RiskLevel.READ_ONLY, harness_status))
    registry.register(Tool("harness_cycle_preview", "Preview how the Jarvis agent harness would route, gate, act, verify, and learn from an order without executing it.", RiskLevel.READ_ONLY, harness_cycle_preview))
    registry.register(Tool("harness_lifecycle_state", "Map an order onto Jarvis perceive-ground-route-plan-gate-act-verify-learn lifecycle without executing anything.", RiskLevel.READ_ONLY, harness_lifecycle_state))
    registry.register(Tool("harness_control_surface", "Inspect the engine, steering, pedals, brakes, dashboard, and proof controls for a natural order without executing anything.", RiskLevel.READ_ONLY, harness_control_surface))
    registry.register(Tool("harness_operations_brief", "Choose the next safe Jarvis harness build move from approvals, tasks, goals, and recent failures without executing anything.", RiskLevel.READ_ONLY, harness_operations_brief))
    registry.register(Tool("agi_gate_report", "Report AGI-direction harness gates, current evidence, missing pieces, and next safe build moves.", RiskLevel.READ_ONLY, agi_gate_report))
    registry.register(Tool("agi_next_build_move", "Select the next AGI-direction real-execution gate gap with owning files, tests, acceptance checks, and safety stops.", RiskLevel.READ_ONLY, agi_next_build_move))
    registry.register(Tool("harness_completion_assessment", "Estimate Jarvis V2 agent-harness prototype completion, current strengths, and real-execution gaps.", RiskLevel.READ_ONLY, harness_completion_assessment))
    registry.register(Tool("harness_doctrine", "Show the agent-harness doctrine that should steer Jarvis completion work.", RiskLevel.READ_ONLY, harness_doctrine))
    registry.register(Tool("coding_discipline_packet", "Build a read-only Karpathy-style pre-code discipline packet with assumptions, surgical scope, success criteria, and verification.", RiskLevel.READ_ONLY, coding_discipline_packet))
    registry.register(Tool("completion_audit_packet", "Audit a Jarvis objective against requirement-level evidence before claiming completion.", RiskLevel.READ_ONLY, completion_audit_packet))
    registry.register(Tool("evidence_ledger", "Show a read-only proof ledger across harness lanes, tool runs, tasks, approvals, and verification evidence.", RiskLevel.READ_ONLY, evidence_ledger))
    registry.register(Tool("completion_claim_gate", "Block or allow a completion claim based on approvals, open work, evidence lanes, verification packets, and failed runs.", RiskLevel.READ_ONLY, completion_claim_gate))
    registry.register(Tool("completion_next_proof_packet", "Choose the next required command from the completion gate proof queue without running it.", RiskLevel.READ_ONLY, completion_next_proof_packet))
    registry.register(Tool("completion_proof_refresh_packet", "Refresh all completion proof lanes, stale-lane commands, AGI closure commands, and placeholder boundaries without running them.", RiskLevel.READ_ONLY, completion_proof_refresh_packet))
    registry.register(Tool("operator_handoff_packet", "Show the next operator-reviewable required command, reason, blocker, and queue position without running it.", RiskLevel.READ_ONLY, operator_handoff_packet))
    registry.register(Tool("harness_readiness_digest", "Summarize Jarvis completion readiness, top blockers, next required command, recovery/learning debt, and AGI gate focus without executing anything.", RiskLevel.READ_ONLY, harness_readiness_digest))
    registry.register(Tool("execution_proof_bundle", "Bundle route, argument, risk, verification, acceptance, audit, recovery, and learning proof before trusting a real action.", RiskLevel.READ_ONLY, execution_proof_bundle))
    registry.register(Tool("execution_mission_control", "Rehearse one real order across route, gate, act, verify, recover, and learn stages without executing it.", RiskLevel.READ_ONLY, execution_mission_control))
    registry.register(Tool("execution_case_handoff_packet", "Bridge a command cockpit and mission-control queue into a read-only durable execution-case handoff without creating the case.", RiskLevel.READ_ONLY, execution_case_handoff_packet))
    registry.register(Tool("save_execution_case", "Save a Jarvis-owned execution case file from mission control without executing the order.", RiskLevel.LOCAL_SAFE, save_execution_case))
    registry.register(Tool("inspect_execution_case", "Inspect a saved Jarvis execution case file and its next required command plus proof aliases.", RiskLevel.READ_ONLY, inspect_execution_case))
    registry.register(Tool("execution_case_evidence_packet", "Preview and validate execution-case evidence, inferred receipt metadata, and receipt target existence before appending it.", RiskLevel.READ_ONLY, execution_case_evidence_packet))
    registry.register(Tool("append_execution_case_evidence", "Append a local evidence event to a saved Jarvis execution case without executing the order.", RiskLevel.LOCAL_SAFE, append_execution_case_evidence))
    registry.register(Tool("execution_case_gate", "Read-only gate for deciding whether a saved execution case has enough evidence for review.", RiskLevel.READ_ONLY, execution_case_gate))
    registry.register(Tool("execution_case_review_packet", "Build a read-only human-review packet for a saved execution case before trusting completion.", RiskLevel.READ_ONLY, execution_case_review_packet))
    registry.register(Tool("execution_case_closure_packet", "Decide whether a saved execution case has closed proof debt across gate, review, receipts, approval chain, recovery, and learning.", RiskLevel.READ_ONLY, execution_case_closure_packet))
    registry.register(Tool("execution_case_timeline", "Show a read-only chronological timeline for a saved execution case and its evidence events.", RiskLevel.READ_ONLY, execution_case_timeline))
    registry.register(Tool("execution_runbook", "Build a read-only before/during/after runbook for a real order, including gates, proof, recovery, and learning hooks.", RiskLevel.READ_ONLY, execution_runbook))
    capability_map = capabilities.make_capability_tools(registry.list)
    registry.register(Tool("capability_map", "Show Jarvis capabilities grouped by area, risk level, and starter command.", RiskLevel.READ_ONLY, capability_map))
    tool_search = capabilities.make_tool_search_tool(registry.list)
    registry.register(
        Tool(
            "tool_search",
            "Search Jarvis tools by name, area, risk, or description without running them.",
            RiskLevel.READ_ONLY,
            tool_search,
            argument_contract=_tool_argument_contract(
                required_strings=("query",),
                optional_integers=("limit",),
            ),
        )
    )
    tool_detail = capabilities.make_tool_detail_tool(registry.list)
    registry.register(Tool("tool_detail", "Inspect one Jarvis tool's area, risk level, and approval requirement without running it.", RiskLevel.READ_ONLY, tool_detail))
    risk_matrix = capabilities.make_risk_matrix_tool(registry.list)
    registry.register(Tool("risk_matrix", "Show Jarvis tool risk totals and approval-gated areas without running tools.", RiskLevel.READ_ONLY, risk_matrix))
    registry.register(
        Tool(
            "subagent_fleet_status",
            "Show bounded read-only internal subagent fleet readiness without exposing raw results, errors, instructions, or shared memory.",
            RiskLevel.READ_ONLY,
            subagents.make_subagent_fleet_status_tool(subagent_fleet),
        )
    )
    capability_cockpit_tool = cockpit.make_capability_cockpit_tool(store, registry.list)
    registry.register(
        Tool(
            "capability_cockpit",
            "Show each Jarvis capability lane with health, risk, approval need, last success/failure, and an example command.",
            RiskLevel.READ_ONLY,
            capability_cockpit_tool,
            argument_contract=_tool_argument_contract(),
        )
    )
    (channel_health_tool,) = channel_health.make_channel_health_tools(store)
    registry.register(
        Tool(
            "channel_health",
            "Report recent messaging and calling channel health from audit metadata only.",
            RiskLevel.READ_ONLY,
            channel_health_tool,
            "personal",
            argument_contract=_tool_argument_contract(),
        )
    )
    roadmap_report = roadmap.make_roadmap_tools(store, registry.list)
    registry.register(Tool("roadmap_report", "Show a safe phased roadmap for building Jarvis V2 toward fuller assistant autonomy.", RiskLevel.READ_ONLY, roadmap_report))
    if config is not None:
        registry.register(
            Tool(
                "model_routing_status",
                "Show chat/planner model routing readiness without running a model request.",
                RiskLevel.READ_ONLY,
                model_status.make_model_status_tool(config),
                argument_contract=_tool_argument_contract(),
            )
        )
        registry.register(Tool("model_planner_prompt_preview", "Preview the model-planner prompt, tool list, and safety boundaries without calling a model.", RiskLevel.READ_ONLY, model_status.make_model_planner_prompt_preview_tool(config, registry.list), "core"))
        registry.register(Tool("specialist_router_contract", "Choose a read-only specialist brain route and handoff contract without calling models or tools.", RiskLevel.READ_ONLY, model_status.make_specialist_router_contract_tool(config, registry.list), "core"))
        registry.register(Tool("specialist_orchestration_packet", "Build a read-only multi-brain specialist orchestration flow with proof lanes, blockers, and execution locks.", RiskLevel.READ_ONLY, model_status.make_specialist_orchestration_packet_tool(config, registry.list), "core"))
        registry.register(Tool("specialist_route_quality", "Measure specialist route confidence, ambiguity, verifier coverage, and approval triggers before model execution.", RiskLevel.READ_ONLY, model_status.make_specialist_route_quality_tool(config, registry.list), "core"))
        registry.register(Tool("specialist_execution_readiness", "Gate whether a request is ready for a specialist model draft, route review, model setup, or approval review.", RiskLevel.READ_ONLY, model_status.make_specialist_execution_readiness_tool(config, registry.list), "core"))
        registry.register(Tool("specialist_handoff_receipt", "Package a request for the selected specialist brain with input, output, safety, and verification contracts.", RiskLevel.READ_ONLY, model_status.make_specialist_handoff_receipt_tool(config, registry.list), "core"))
        registry.register(Tool("specialist_handoff_quality_gate", "Bind specialist route quality and handoff receipt quality before proposal-gate review.", RiskLevel.READ_ONLY, model_status.make_specialist_handoff_quality_gate_tool(config, registry.list), "core"))
        registry.register(Tool("specialist_proposal_gate", "Join specialist route, readiness, handoff, verifier, model, and approval proof before any specialist draft can become a tool proposal.", RiskLevel.READ_ONLY, model_status.make_specialist_proposal_gate_tool(config, registry.list), "core"))
        registry.register(Tool("specialist_model_draft", "Run or safely fall back from a bounded specialist model draft after route, readiness, and handoff proof gates.", RiskLevel.READ_ONLY, model_status.make_specialist_model_draft_tool(config, registry.list), "core"))
        registry.register(Tool("specialist_action_proposal_contract", "Inspect specialist draft-to-tool proposal readiness without emitting executable actions or bypassing ToolRegistry and PermissionPolicy.", RiskLevel.READ_ONLY, model_status.make_specialist_action_proposal_contract_tool(config, registry.list), "core"))
        registry.register(Tool("specialist_tool_dry_run_packet", "Review a specialist tool proposal for ToolRegistry, arguments, PermissionPolicy, and verification proof without executing it.", RiskLevel.READ_ONLY, model_status.make_specialist_tool_dry_run_packet_tool(config, registry.list), "core"))
        registry.register(Tool("specialist_proposal_completion_gate", "Final-gate a specialist draft-to-tool proposal across contract, dry-run, verification, audit, recovery, and learning proof without executing it.", RiskLevel.READ_ONLY, model_status.make_specialist_proposal_completion_gate_tool(config, registry.list), "core"))
        registry.register(Tool("specialist_execution_handoff_packet", "Package a completed specialist proposal for the normal runtime review path without executing it or bypassing ToolRegistry and PermissionPolicy.", RiskLevel.READ_ONLY, model_status.make_specialist_execution_handoff_packet_tool(config, registry.list), "core"))
        registry.register(Tool("specialist_post_run_closure_packet", "Close a specialist execution handoff after runtime trace, verification, audit, recovery, learning, and completion-claim evidence.", RiskLevel.READ_ONLY, model_status.make_specialist_post_run_closure_packet_tool(config, registry.list), "core"))
        registry.register(Tool("specialist_cycle_ledger", "Bind route, readiness, handoff, proposal, dry-run, runtime handoff, and post-run closure proof before starting a fresh specialist review.", RiskLevel.READ_ONLY, model_status.make_specialist_cycle_ledger_tool(config, registry.list), "core"))
    registry.register(Tool("jarvis_status", "Show a compact Jarvis brain/system status.", RiskLevel.READ_ONLY, jarvis_status))
    registry.register(Tool("status_dashboard", "Show how to launch the read-only status dashboard.", RiskLevel.READ_ONLY, status_dashboard))
    registry.register(Tool("storage_status", "Show read-only Jarvis storage routing, fallback, and recovery diagnostics.", RiskLevel.READ_ONLY, storage.make_storage_status_tool(config, storage_fallback), "safety"))
    registry.register(Tool("storage_recovery_check", "Run Jarvis' native no-write durable-storage recovery readiness check before bootstrap writes.", RiskLevel.READ_ONLY, storage.make_storage_recovery_check_tool(config, storage_fallback), "safety"))
    registry.register(Tool("storage_recovery_plan", "Draft a read-only project-local durable-storage recovery plan with env exports and the existing proof ladder.", RiskLevel.READ_ONLY, storage.make_storage_recovery_plan_tool(config, storage_fallback), "safety"))
    registry.register(
        Tool(
            "startup_recovery_report",
            "Show content-free durable receipts for projection repair performed before Jarvis accepted commands.",
            RiskLevel.READ_ONLY,
            startup_recovery.make_startup_recovery_report_tool(store),
            "safety",
            argument_contract=_tool_argument_contract(optional_integers=("limit",)),
        )
    )
    safety_status = safety.make_safety_tools(store, registry.list)
    registry.register(Tool("safety_status", "Show active safety boundaries, risk-gated tools, and pending approvals.", RiskLevel.READ_ONLY, safety_status, "safety"))
    frozen_routing_risk_report = safety.make_frozen_routing_risk_tool()
    registry.register(Tool("frozen_routing_risk_report", "Show the open frozen send/call routing ReDoS risk and current no-edit boundary without inspecting source or running probes.", RiskLevel.READ_ONLY, frozen_routing_risk_report, "safety"))
    planner_input_guard_report = safety.make_planner_input_guard_report_tool()
    registry.register(Tool("planner_input_guard_report", "Show the open planner-wide input-length guard decision without editing routing or running probes.", RiskLevel.READ_ONLY, planner_input_guard_report, "safety"))
    privacy_report = privacy.make_privacy_tools(store, config, registry.list)
    registry.register(Tool("privacy_report", "Show what data Jarvis may use by default, what requires approval, and what is not migrated.", RiskLevel.READ_ONLY, privacy_report, "safety"))
    readiness_report, prototype_readiness_checklist = readiness.make_readiness_tools(store, vault, config, registry.list, storage_fallback)
    registry.register(Tool("readiness_report", "Show whether Jarvis is ready for safe use, including setup, memory, schedules, approvals, and risk gates.", RiskLevel.READ_ONLY, readiness_report, "safety"))
    registry.register(Tool("prototype_readiness_checklist", "Show what is safe to try in the Jarvis prototype now and what remains approval-gated or setup-blocked.", RiskLevel.READ_ONLY, prototype_readiness_checklist, "safety"))
    autonomy_plan, risk_preflight, agent_loop_preview, agent_loop_packet, risky_request_lifecycle, action_readiness_packet, execution_contract, argument_contract_packet, verification_packet, execution_acceptance_gate, execution_readiness_matrix, dispatch_decision_packet, command_intake_packet, execution_governor_packet, planner_gap_packet, command_cockpit_packet = autonomy.make_autonomy_tools(registry.list, store)
    registry.register(Tool("autonomy_plan", "Draft a safe autonomy plan with approval-gated steps and verification checkpoints.", RiskLevel.READ_ONLY, autonomy_plan, "safety"))
    registry.register(Tool("risk_preflight", "Classify a proposed request's likely risk areas and safer preview path without executing tools.", RiskLevel.READ_ONLY, risk_preflight, "safety"))
    registry.register(Tool("agent_loop_preview", "Preview Jarvis's second task/action loop without calling a model or executing tools.", RiskLevel.READ_ONLY, agent_loop_preview, "safety"))
    registry.register(Tool("agent_loop_packet", "Prepare a read-only second-loop task packet with phases, stop conditions, preview commands, and approval boundaries.", RiskLevel.READ_ONLY, agent_loop_packet, "safety"))
    registry.register(Tool("risky_request_lifecycle", "Map a risky request from preflight through approval readiness, last-look packet, rerun, audit, and stop conditions.", RiskLevel.READ_ONLY, risky_request_lifecycle, "safety"))
    registry.register(Tool("action_readiness_packet", "Decide whether a proposed action is ready, needs preflight, or should stop for approval review without executing tools.", RiskLevel.READ_ONLY, action_readiness_packet, "safety"))
    registry.register(Tool("execution_contract", "Turn an order into a read-only harness contract with route, gates, verification, recovery, and learning hooks.", RiskLevel.READ_ONLY, execution_contract, "safety"))
    registry.register(Tool("argument_contract_packet", "Prove exact planned tool arguments before execution or approval without running the order.", RiskLevel.READ_ONLY, argument_contract_packet, "safety"))
    registry.register(Tool("verification_packet", "Turn an order into a read-only verification plan with evidence requirements, failure signals, and recovery steps.", RiskLevel.READ_ONLY, verification_packet, "safety"))
    registry.register(Tool("execution_acceptance_gate", "Gate whether a behavior has enough route, approval, audit, verification, test, and recovery evidence to be treated as done.", RiskLevel.READ_ONLY, execution_acceptance_gate, "safety"))
    registry.register(Tool("execution_readiness_matrix", "Show a read-only harness matrix for route, risk gates, approvals, proof, recovery, and go/no-go state.", RiskLevel.READ_ONLY, execution_readiness_matrix, "safety"))
    registry.register(Tool("dispatch_decision_packet", "Choose read-only dispatch state for an order: chat, auto-safe tools, approval-gated work, hold, or clarify.", RiskLevel.READ_ONLY, dispatch_decision_packet, "safety"))
    registry.register(Tool("command_intake_packet", "Turn a natural text or speech order into a read-only route, risk, approval, next-command, and proof packet.", RiskLevel.READ_ONLY, command_intake_packet, "safety"))
    registry.register(Tool("execution_governor_packet", "Give a read-only command-first go/no-go verdict across intake, dispatch, proof, recovery, and approval state.", RiskLevel.READ_ONLY, execution_governor_packet, "safety"))
    registry.register(Tool("planner_gap_packet", "Detect read-only planner gaps where an order sounds actionable but lacks an exact safe route.", RiskLevel.READ_ONLY, planner_gap_packet, "safety"))
    registry.register(Tool("command_cockpit_packet", "Show a single read-only command-first cockpit across intake, governor, dispatch, readiness, verification, recovery, and learning state.", RiskLevel.READ_ONLY, command_cockpit_packet, "safety"))
    action_rehearsal = rehearsal.make_rehearsal_tool(registry.get, store)
    registry.register(Tool("action_rehearsal", "Preview planner routing, tool risks, and approval requirements without executing tools.", RiskLevel.READ_ONLY, action_rehearsal, "safety"))
    def command_diagnosis(args: dict[str, Any]) -> ToolResult:
        request = _short(args.get("request") or args.get("message"), limit=MAX_CORE_TEXT_CHARS)
        if not request:
            output = (
                "Give Jarvis a request to diagnose, for example: "
                "`command diagnosis: run command python3 --version`. "
                f"{CORE_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "command_diagnosis",
                False,
                output,
                _core_input_failure_metadata(output),
            )
        display_request = _short_metadata(request, limit=MAX_CORE_TEXT_CHARS)
        from jarvis_v2.agent.planner import RuleBasedPlanner

        plan = RuleBasedPlanner().plan(request)
        planner_metadata = _empty_planner_diagnostic_metadata()
        recent_runs = store.recent_tool_runs(limit=12)
        recovery_closure = autonomy._execution_health_recovery_closure_snapshot(recent_runs)
        execution_learning_debt = harness._execution_learning_debt_snapshot(recent_runs)
        planned_actions = []
        approval_required = False
        for action in plan.actions:
            try:
                tool = registry.get(action.tool_name)
                risk = tool.risk.name
                toolset = tool.toolset
                needs_approval = tool.risk in {RiskLevel.PERSONAL_DATA, RiskLevel.EXTERNAL_SIDE_EFFECT, RiskLevel.HIGH_RISK}
            except KeyError:
                risk = "UNKNOWN"
                toolset = "unknown"
                needs_approval = True
            approval_required = approval_required or needs_approval
            planned_actions.append(
                {
                    "tool": action.tool_name,
                    "risk": risk,
                    "toolset": toolset,
                    "requires_approval": needs_approval,
                    "args": {str(key): _short_metadata(value, limit=240) for key, value in action.args.items()},
                    "reason": _short_metadata(action.reason, limit=240),
                }
            )
        pending_approvals = len(store.list_pending_approvals(limit=100))
        approval_queue_forecast = []
        for action, item in zip(plan.actions, planned_actions):
            if not item["requires_approval"]:
                continue
            existing = store.find_matching_pending_approval(request, action.tool_name, action.args)
            existing_id = int(existing["id"]) if existing is not None else None
            approval_queue_forecast.append(
                {
                    "tool_name": action.tool_name,
                    "risk": item["risk"],
                    "existing_approval_id": existing_id,
                    "would_queue_new_approval": existing_id is None,
                    "would_reuse_pending_approval": existing_id is not None,
                    "planned_arg_keys": sorted(str(key) for key in action.args),
                }
            )
        forecast_new_approvals = sum(1 for item in approval_queue_forecast if item["would_queue_new_approval"])
        forecast_reused_approval_ids = [
            item["existing_approval_id"]
            for item in approval_queue_forecast
            if item["existing_approval_id"] is not None
        ]
        read_only_actions = bool(planned_actions) and all(item["risk"] == "READ_ONLY" for item in planned_actions)
        if plan.needs_model and not plan.actions:
            route = "chat"
            recommendation = "ANSWER_IN_CHAT"
        elif approval_required and pending_approvals:
            route = "hold"
            recommendation = "REVIEW_EXISTING_APPROVALS"
        elif recovery_closure["blocks_auto_execution"] and not read_only_actions:
            route = "recovery_closure"
            recommendation = "RECOVERY_CLOSURE_REQUIRED"
        elif rehearsal._learning_debt_blocks_rehearsal(execution_learning_debt) and not read_only_actions:
            route = "execution_learning_debt"
            recommendation = "EXECUTION_LEARNING_REQUIRED"
        elif approval_required:
            route = "approval"
            recommendation = "ENTER_EXECUTION_GOVERNOR"
        elif plan.actions:
            route = "auto_tool"
            recommendation = "AUTO_RUN_LOCAL_SAFE"
        else:
            route = "none"
            recommendation = "ASK_FOR_MORE_DETAIL"
        if route == "approval":
            next_commands = [
                f"execution governor: {display_request}",
                "send this command normally to queue an approval receipt",
                "approval readiness latest",
                "approval packet latest",
                "approval chain proof latest",
                "pending approvals",
                "approve approval latest",
            ]
        elif route == "hold":
            next_commands = [
                "pending approvals",
                "approval review",
                "approval readiness latest",
                "approval packet latest",
                "approval chain proof latest",
                "approve approval latest",
            ]
        elif route == "recovery_closure":
            next_commands = list(recovery_closure["required_commands"]) or ["execution health report"]
        elif route == "execution_learning_debt":
            next_commands = list(execution_learning_debt["required_commands"]) or ["after-action learning packet"]
        elif route == "auto_tool":
            next_commands = [f"execution governor: {display_request}", "send this command normally"]
        elif route == "chat":
            next_commands = ["send this message normally"]
        else:
            next_commands = ["add more detail, or ask for `jarvis help`"]

        lines = [
            "Jarvis command diagnosis",
            f"Request: {display_request}",
            f"Route: {route}",
            f"Recommendation: {recommendation}",
            f"Goal: {_short_metadata(plan.goal, limit=240)}",
            f"Approval required: {'yes' if approval_required else 'no'}",
            f"Pending approvals visible: {pending_approvals}",
            f"Safe to execute now: {'yes' if route in {'chat', 'auto_tool'} else 'no'}",
            "",
            "Approval queue forecast:",
            f"- current pending approvals: {pending_approvals}",
            f"- would queue new approvals if sent: {forecast_new_approvals}",
            f"- would reuse pending approval ids: {', '.join(str(item) for item in forecast_reused_approval_ids) if forecast_reused_approval_ids else 'none'}",
            f"- forecast queue after if sent: {pending_approvals + forecast_new_approvals}",
            "",
            "Planned actions:",
        ]
        if planned_actions:
            for index, item in enumerate(planned_actions, start=1):
                approval_text = "approval required" if item["requires_approval"] else "auto-safe"
                lines.append(f"- {index}. {item['tool']} [{item['toolset']}, {item['risk']}]: {approval_text}")
                lines.append(f"  Reason: {item['reason']}")
                if item["args"]:
                    arg_text = ", ".join(f"{key}={value!r}" for key, value in sorted(item["args"].items()))
                    lines.append(f"  Planned arguments: {arg_text}")
                if item["requires_approval"]:
                    category, concern, safer = approvals._risk_review_for(item["tool"], request)
                    lines.append(f"  Risk category: {category}")
                    lines.append(f"  Why gated: {concern}")
                    lines.append(f"  Safer check: {safer}")
        else:
            lines.append("- none")
        if recovery_closure["blocks_auto_execution"]:
            recovery_heading = (
                "Approval-held execution review:"
                if recovery_closure["state"] == "approval_held_review_required"
                else "Execution health recovery closure:"
            )
            lines.extend(
                [
                    "",
                    recovery_heading,
                    f"- state: {recovery_closure['state']}",
                    f"- ready to retry: {'yes' if recovery_closure['ready_to_retry'] else 'no'}",
                    f"- target run: #{recovery_closure['target_run_id']} `{recovery_closure['target_tool_name']}`" if recovery_closure["target_run_id"] is not None else "- target run: none",
                    f"- real failed/blocked action runs: {recovery_closure.get('failed_or_blocked_action_runs', 0)}",
                    f"- approval-held action runs: {recovery_closure.get('approval_held_action_runs', 0)}",
                    f"- missing: {', '.join(recovery_closure['missing']) if recovery_closure['missing'] else 'none'}",
                    f"- next required: `{recovery_closure['next_required_command']}`" if recovery_closure["next_required_command"] else "- next required: none",
                    f"- proof queue: {', '.join(f'`{command}`' for command in recovery_closure['required_commands']) if recovery_closure['required_commands'] else 'none'}",
                ]
            )
        if route != "chat" and execution_learning_debt["blocks_completion_claim"]:
            lines.extend(
                [
                    "",
                    "Execution learning debt:",
                    f"- state: {execution_learning_debt['state']}",
                    f"- blocks completion claim: {'yes' if execution_learning_debt['blocks_completion_claim'] else 'no'}",
                    f"- target run: #{execution_learning_debt['target_run_id']} `{execution_learning_debt['target_tool_name']}`" if execution_learning_debt["target_run_id"] is not None else "- target run: none",
                    f"- missing: {', '.join(execution_learning_debt['missing']) if execution_learning_debt['missing'] else 'none'}",
                    f"- next learning required: `{execution_learning_debt['next_required_command']}`" if execution_learning_debt["next_required_command"] else "- next learning required: none",
                    f"- learning proof queue: {', '.join(f'`{command}`' for command in execution_learning_debt['required_commands']) if execution_learning_debt['required_commands'] else 'none'}",
                ]
            )
        lines.extend(
            [
                "",
                "Next safe commands:",
                *[f"- {command}" for command in next_commands],
                "",
                "Safety boundary:",
                "- This diagnosis is read-only and does not call models, execute tools, approve requests, read private data, write files, control the computer, or queue approvals.",
            ]
        )
        return ToolResult(
            "command_diagnosis",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=display_request,
                display_request=display_request,
                route=route,
                recommendation=recommendation,
                next_command=next_commands[0],
                goal=_short_metadata(plan.goal, limit=240),
                needs_model=plan.needs_model,
                planner_notes=plan.notes,
                planner_metadata=planner_metadata,
                planner_model_planner_attempted=planner_metadata["model_planner_attempted"],
                planner_model_planner_state=planner_metadata["model_planner_state"],
                planner_model_planner_used=planner_metadata["model_planner_used"],
                planner_model_planner_fell_back=planner_metadata["model_planner_fell_back"],
                planner_model_planner_fallback_reason=planner_metadata["model_planner_fallback_reason"],
                planner_model_planner_fallback_detail=planner_metadata["model_planner_fallback_detail"],
                planner_model_planner_recovery_hint=planner_metadata.get("model_planner_recovery_hint"),
                planner_model_planner_exception_type=planner_metadata["model_planner_exception_type"],
                planner_model_planner_model=planner_metadata["model_planner_model"],
                planner_model_planner_timeout_seconds=planner_metadata["model_planner_timeout_seconds"],
                planner_model_planner_action_count=planner_metadata["model_planner_action_count"],
                planner_model_planner_ignored_unknown_tools=planner_metadata["model_planner_ignored_unknown_tools"],
                planned_actions=planned_actions,
                approval_required=approval_required,
                pending_approvals=pending_approvals,
                approval_queue_forecast=approval_queue_forecast,
                forecast_new_approvals=forecast_new_approvals,
                forecast_reused_approval_ids=forecast_reused_approval_ids,
                forecast_queue_before=pending_approvals,
                forecast_queue_after_if_sent=pending_approvals + forecast_new_approvals,
                forecast_queue_delta_if_sent=forecast_new_approvals,
                safe_to_execute_now=route in {"chat", "auto_tool"},
                recommended_next_commands=next_commands,
                recovery_closure_state=recovery_closure["state"],
                recovery_closure_ready_to_retry=recovery_closure["ready_to_retry"],
                recovery_closure_recent_action_runs=recovery_closure["recent_action_runs"],
                recovery_closure_failed_or_blocked_action_runs=recovery_closure["failed_or_blocked_action_runs"],
                recovery_closure_approval_held_action_runs=recovery_closure["approval_held_action_runs"],
                recovery_closure_approval_held_target_run_id=recovery_closure["approval_held_target_run_id"],
                recovery_closure_approval_held_target_tool_name=recovery_closure["approval_held_target_tool_name"],
                recovery_closure_approval_review_commands=recovery_closure["approval_review_commands"],
                recovery_closure_approval_review_command_count=recovery_closure["approval_review_command_count"],
                recovery_closure_missing=recovery_closure["missing"],
                recovery_closure_missing_count=recovery_closure["missing_count"],
                recovery_closure_required_commands=recovery_closure["required_commands"],
                recovery_closure_next_required_command=recovery_closure["next_required_command"],
                recovery_closure_blocks_auto_execution=recovery_closure["blocks_auto_execution"],
                recovery_closure_target_run_id=recovery_closure["target_run_id"],
                recovery_closure_target_tool_name=recovery_closure["target_tool_name"],
                recovery_closure_proof_queue=recovery_closure["required_commands"],
                recovery_closure_proof_queue_count=len(recovery_closure["required_commands"]),
                recovery_closure_next_proof_command=recovery_closure["next_required_command"],
                execution_learning_state=execution_learning_debt["state"],
                execution_learning_blocks_completion_claim=execution_learning_debt["blocks_completion_claim"],
                execution_learning_recent_action_runs=execution_learning_debt["recent_action_runs"],
                execution_learning_failed_or_blocked_action_runs=execution_learning_debt["failed_or_blocked_action_runs"],
                execution_learning_recent_verification_runs=execution_learning_debt["recent_verification_runs"],
                execution_learning_recent_recovery_runs=execution_learning_debt["recent_recovery_runs"],
                execution_learning_recent_after_action_learning_runs=execution_learning_debt["recent_after_action_learning_runs"],
                execution_learning_target_run_id=execution_learning_debt["target_run_id"],
                execution_learning_target_tool_name=execution_learning_debt["target_tool_name"],
                execution_learning_target_after_action_learning_packets=execution_learning_debt["target_after_action_learning_packets"],
                execution_learning_missing=execution_learning_debt["missing"],
                execution_learning_missing_count=execution_learning_debt["missing_count"],
                execution_learning_required_commands=execution_learning_debt["required_commands"],
                execution_learning_next_required_command=execution_learning_debt["next_required_command"],
                execution_learning_proof_queue=execution_learning_debt["required_commands"],
                execution_learning_proof_queue_count=len(execution_learning_debt["required_commands"]),
                execution_learning_next_proof_command=execution_learning_debt["next_required_command"],
                command_diagnosis_handoff=_command_diagnosis_handoff(
                    display_request=display_request,
                    route=route,
                    recommendation=recommendation,
                    next_commands=next_commands,
                    plan_goal=_short_metadata(plan.goal, limit=240),
                    planned_actions=planned_actions,
                    approval_required=approval_required,
                    pending_approvals=pending_approvals,
                    approval_queue_forecast=approval_queue_forecast,
                    forecast_new_approvals=forecast_new_approvals,
                    forecast_reused_approval_ids=forecast_reused_approval_ids,
                    recovery_closure=recovery_closure,
                    execution_learning_debt=execution_learning_debt,
                ),
            ),
        )

    registry.register(Tool("command_diagnosis", "Diagnose whether a message would route to chat, auto-safe tools, approval, or hold without executing it.", RiskLevel.READ_ONLY, command_diagnosis, "safety"))
    assistant_turn_rehearsal = rehearsal.make_assistant_turn_rehearsal_tool(registry.get, store, vault, session_id)
    registry.register(Tool("assistant_turn_rehearsal", "Preview whether a Jarvis message would become chat or tools, including context and approval boundaries.", RiskLevel.READ_ONLY, assistant_turn_rehearsal, "conversation"))
    (
        activity_digest,
        operator_instruction_supersession_packet,
        operator_timebox_contract,
        build_progress_report,
        build_delta_report,
        work_block_checkpoint,
        save_build_progress,
        save_build_delta,
        save_work_block_checkpoint,
        checkpoint_recovery_preview,
        checkpoint_recovery_apply_packet,
        checkpoint_recovery_receipt,
        checkpoint_recovery_execute,
        checkpoint_recovery_followthrough_packet,
        checkpoint_recovery_cockpit,
        autonomy_resume_gate,
        autonomy_continuation_execution_packet,
        autonomy_step_closure_packet,
        autonomy_cycle_ledger,
        recent_saved_notes,
    ) = continuity.make_continuity_tools(store, vault)
    registry.register(Tool("activity_digest", "Summarize recent Jarvis activity, blockers, and continuity context.", RiskLevel.READ_ONLY, activity_digest, "continuity"))
    registry.register(Tool("operator_instruction_supersession_packet", "Prove the newest operator instruction and stop window override older autonomy goals or scheduled continuations.", RiskLevel.READ_ONLY, operator_instruction_supersession_packet, "continuity"))
    registry.register(Tool("operator_timebox_contract", "Check whether Jarvis continuation is still inside the operator's explicit work window.", RiskLevel.READ_ONLY, operator_timebox_contract, "continuity"))
    registry.register(Tool("build_progress_report", "Summarize recent Jarvis build progress, safety posture, and safe next build moves.", RiskLevel.READ_ONLY, build_progress_report, "continuity"))
    registry.register(Tool("build_delta_report", "Summarize the latest checkpoint window of tool activity, conversation, and safety state.", RiskLevel.READ_ONLY, build_delta_report, "continuity"))
    registry.register(Tool("work_block_checkpoint", "Prepare a resumable work-block checkpoint with verification, blockers, and next safe command.", RiskLevel.READ_ONLY, work_block_checkpoint, "continuity"))
    registry.register(Tool("save_build_progress", "Write the current Jarvis build-progress report to Obsidian Reflections.", RiskLevel.LOCAL_SAFE, save_build_progress, "continuity"))
    registry.register(Tool("save_build_delta", "Write the latest Jarvis build-delta checkpoint to Obsidian Reflections.", RiskLevel.LOCAL_SAFE, save_build_delta, "continuity"))
    registry.register(Tool("save_work_block_checkpoint", "Write a resumable Jarvis work-block checkpoint to Obsidian Reflections.", RiskLevel.LOCAL_SAFE, save_work_block_checkpoint, "continuity"))
    registry.register(Tool("checkpoint_recovery_preview", "Preview how to resume from the latest saved work-block checkpoint and verify the next step.", RiskLevel.READ_ONLY, checkpoint_recovery_preview, "continuity"))
    registry.register(Tool("checkpoint_recovery_apply_packet", "Prepare an approval-gated packet for applying one checkpoint recovery step.", RiskLevel.READ_ONLY, checkpoint_recovery_apply_packet, "continuity"))
    registry.register(Tool("checkpoint_recovery_receipt", "Save an after-action receipt for one reviewed checkpoint recovery step and its verification evidence.", RiskLevel.LOCAL_SAFE, checkpoint_recovery_receipt, "continuity"))
    registry.register(Tool("checkpoint_recovery_execute", "Record one reviewed local-safe checkpoint recovery step and save a fresh continuity checkpoint.", RiskLevel.LOCAL_SAFE, checkpoint_recovery_execute, "continuity"))
    registry.register(Tool("checkpoint_recovery_followthrough_packet", "Prove checkpoint recovery closure evidence before normal autonomous follow-through resumes.", RiskLevel.READ_ONLY, checkpoint_recovery_followthrough_packet, "continuity"))
    registry.register(Tool("checkpoint_recovery_cockpit", "Consolidate operator timebox, checkpoint recovery preview, apply gate, blockers, and proof queue before resuming local-safe work.", RiskLevel.READ_ONLY, checkpoint_recovery_cockpit, "continuity"))
    registry.register(Tool("autonomy_resume_gate", "Bind operator timebox, checkpoint recovery cockpit, and follow-through closure before normal local-safe continuation resumes.", RiskLevel.READ_ONLY, autonomy_resume_gate, "continuity"))
    registry.register(Tool("autonomy_continuation_execution_packet", "Gate one normal local-safe continuation step after autonomy resume proof without executing it.", RiskLevel.READ_ONLY, autonomy_continuation_execution_packet, "continuity"))
    registry.register(Tool("autonomy_step_closure_packet", "Close one local-safe autonomy continuation step with post-step proof before another continuation review.", RiskLevel.READ_ONLY, autonomy_step_closure_packet, "continuity"))
    registry.register(Tool("autonomy_cycle_ledger", "Bind a full autonomy timebox, recovery, one-step continuation, and closure cycle before the next continuation review.", RiskLevel.READ_ONLY, autonomy_cycle_ledger, "continuity"))
    registry.register(Tool("recent_saved_notes", "List recent Jarvis Obsidian notes by metadata without opening note contents.", RiskLevel.READ_ONLY, recent_saved_notes, "continuity"))
    focus_brief, next_session_plan, work_session_packet = focus.make_focus_tools(store)
    registry.register(Tool("focus_brief", "Build a read-only work-session brief from tasks, goals, approvals, decisions, preferences, and jobs.", RiskLevel.READ_ONLY, focus_brief, "continuity"))
    registry.register(Tool("next_session_plan", "Build a read-only resume plan for the next Jarvis work session without taking actions.", RiskLevel.READ_ONLY, next_session_plan, "continuity"))
    registry.register(Tool("work_session_packet", "Prepare a read-only start packet with preflight checks, stop conditions, and verification.", RiskLevel.READ_ONLY, work_session_packet, "continuity"))
    safe_next_actions, next_action_packet, priority_stack, continuation_packet, build_target_packet, harness_build_slice, save_build_target_packet, export_mission_control, work_queue, save_work_queue = next_step.make_next_step_tools(store, vault, registry.list, storage_fallback, config)
    registry.register(Tool("safe_next_actions", "Suggest safe next assistant actions from current tasks, goals, approvals, and jobs.", RiskLevel.READ_ONLY, safe_next_actions, "continuity"))
    registry.register(Tool("next_action_packet", "Choose one proposed next move with rationale, risk, and verification without executing it.", RiskLevel.READ_ONLY, next_action_packet, "continuity"))
    registry.register(Tool("priority_stack", "Rank pending approvals, tasks, goals, and background setup before Jarvis acts.", RiskLevel.READ_ONLY, priority_stack, "continuity"))
    registry.register(Tool("continuation_packet", "Prepare a read-only safe continuation loop for ongoing Jarvis build work.", RiskLevel.READ_ONLY, continuation_packet, "continuity"))
    registry.register(Tool("build_target_packet", "Select one scoped Jarvis build target with likely files, tests, boundaries, and stop conditions.", RiskLevel.READ_ONLY, build_target_packet, "continuity"))
    registry.register(Tool("harness_build_slice", "Select one implementation slice for autonomous Jarvis harness work without approving or executing risky actions.", RiskLevel.READ_ONLY, harness_build_slice, "continuity"))
    registry.register(Tool("save_build_target_packet", "Write the current scoped Jarvis build target packet to Obsidian Automations.", RiskLevel.LOCAL_SAFE, save_build_target_packet, "continuity"))
    registry.register(Tool("export_mission_control", "Write safe next actions and current blockers to Obsidian Mission Control.", RiskLevel.LOCAL_SAFE, export_mission_control, "continuity"))
    registry.register(Tool("work_queue", "Show an ordered safe work queue from approvals, tasks, goals, decisions, and scheduled upkeep.", RiskLevel.READ_ONLY, work_queue, "continuity"))
    registry.register(Tool("save_work_queue", "Write the current safe work queue to Obsidian Automations.", RiskLevel.LOCAL_SAFE, save_work_queue, "continuity"))

    def build_return_brief(limit: int) -> tuple[str, bool, dict[str, Any]]:
        readiness_result = readiness_report({})
        activity_result = activity_digest({"limit": limit})
        next_result = safe_next_actions({"limit": limit})
        agi_handoff = focus._agi_focus_snapshot()
        agi_lines = "\n".join(focus._agi_focus_lines(agi_handoff))
        output = "\n\n".join(
            [
                "# Return Brief",
                readiness_result.output,
                activity_result.output,
                next_result.output,
                agi_lines,
            ]
        )
        ok = readiness_result.ok and activity_result.ok and next_result.ok
        agi_metadata = focus._agi_focus_metadata(agi_handoff)
        return_handoff = _continuity_brief_handoff(
            source="return_brief",
            limit=limit,
            readiness_metadata=readiness_result.metadata,
            activity_metadata=activity_result.metadata,
            next_metadata=next_result.metadata,
            agi_metadata=agi_metadata,
        )
        metadata = _safe_metadata(
            readiness=readiness_result.metadata,
            activity=activity_result.metadata,
            next_actions=next_result.metadata,
            **agi_metadata,
            **_brief_handoff_metadata("return_brief", return_handoff),
        )
        return output, ok, metadata

    def return_brief(args: dict[str, Any]) -> ToolResult:
        output, ok, metadata = build_return_brief(_bounded_int(args.get("limit"), 6))
        return ToolResult(
            "return_brief",
            ok,
            output,
            metadata,
        )

    def save_return_brief(args: dict[str, Any]) -> ToolResult:
        output, ok, metadata = build_return_brief(_bounded_int(args.get("limit"), 6))
        path = vault.write_return_brief(output)
        path_display = _safe_vault_path_display(path, vault)
        metadata["path"] = str(path)
        metadata["path_display"] = path_display
        metadata["writes_files"] = True
        metadata["writes_notes"] = True
        return ToolResult(
            "save_return_brief",
            ok,
            f"Return brief saved: {path_display}\n\n{output}",
            metadata,
        )

    def build_handoff_brief(limit: int) -> tuple[str, bool, dict[str, Any]]:
        readiness_result = readiness_report({})
        activity_result = activity_digest({"limit": limit})
        build_result = build_progress_report({"limit": max(limit, 12)})
        next_result = safe_next_actions({"limit": limit})
        agi_handoff = focus._agi_focus_snapshot()
        agi_lines = "\n".join(focus._agi_focus_lines(agi_handoff))
        output = "\n\n".join(
            [
                "# Jarvis Handoff Brief",
                "Purpose: give the operator one safe, inspectable resume point before continuing assistant work.",
                readiness_result.output,
                activity_result.output,
                build_result.output,
                next_result.output,
                agi_lines,
                "Safety reminder:\n- Do not auto-run shell, file-write, clipboard-read, personal-data, external-side-effect, or computer-control actions without explicit approval.\n- Use `approval review` or `save approval review` before approving queued risky work.",
            ]
        )
        ok = readiness_result.ok and activity_result.ok and build_result.ok and next_result.ok
        agi_metadata = focus._agi_focus_metadata(agi_handoff)
        handoff = _continuity_brief_handoff(
            source="handoff_brief",
            limit=limit,
            readiness_metadata=readiness_result.metadata,
            activity_metadata=activity_result.metadata,
            build_metadata=build_result.metadata,
            next_metadata=next_result.metadata,
            agi_metadata=agi_metadata,
        )
        metadata = _safe_metadata(
            readiness=readiness_result.metadata,
            activity=activity_result.metadata,
            build_progress=build_result.metadata,
            next_actions=next_result.metadata,
            **agi_metadata,
            **_brief_handoff_metadata("handoff_brief", handoff),
        )
        return output, ok, metadata

    def handoff_brief(args: dict[str, Any]) -> ToolResult:
        output, ok, metadata = build_handoff_brief(_bounded_int(args.get("limit"), 8))
        return ToolResult(
            "handoff_brief",
            ok,
            output,
            metadata,
        )

    def save_handoff_brief(args: dict[str, Any]) -> ToolResult:
        output, ok, metadata = build_handoff_brief(_bounded_int(args.get("limit"), 8))
        path = vault.write_reflection("Handoff Brief", output)
        path_display = _safe_vault_path_display(path, vault)
        metadata["path"] = str(path)
        metadata["path_display"] = path_display
        metadata["writes_files"] = True
        metadata["writes_notes"] = True
        return ToolResult(
            "save_handoff_brief",
            ok,
            f"Handoff brief saved: {path_display}\n\n{output}",
            metadata,
        )

    def build_session_closeout(limit: int) -> tuple[str, bool, dict[str, Any]]:
        activity_result = activity_digest({"limit": limit})
        build_result = build_progress_report({"limit": max(limit, 12)})
        queue_result = work_queue({"limit": limit})
        approval_result = review_pending_approvals({"limit": limit})
        agi_handoff = focus._agi_focus_snapshot()
        agi_lines = "\n".join(focus._agi_focus_lines(agi_handoff))
        output = "\n\n".join(
            [
                "# Session Closeout",
                "Use this at the end of a Jarvis work session so the next resume point is clear and safe.",
                "Closeout checklist:\n- Save this note with `save session closeout`.\n- If risky work is queued, run `save approval review` before approving anything.\n- Run `save handoff brief` after major build sessions.\n- Leave shell, file-write, clipboard, personal-data, external-side-effect, and computer-control actions blocked until the operator explicitly approves them.",
                activity_result.output,
                build_result.output,
                queue_result.output,
                approval_result.output,
                agi_lines,
            ]
        )
        ok = activity_result.ok and build_result.ok and queue_result.ok and approval_result.ok
        agi_metadata = focus._agi_focus_metadata(agi_handoff)
        handoff = _continuity_brief_handoff(
            source="session_closeout",
            limit=limit,
            activity_metadata=activity_result.metadata,
            build_metadata=build_result.metadata,
            work_queue_metadata=queue_result.metadata,
            approval_metadata=approval_result.metadata,
            agi_metadata=agi_metadata,
        )
        metadata = _safe_metadata(
            activity=activity_result.metadata,
            build_progress=build_result.metadata,
            work_queue=queue_result.metadata,
            approval_review=approval_result.metadata,
            **agi_metadata,
            **_brief_handoff_metadata("session_closeout", handoff),
        )
        return output, ok, metadata

    def session_closeout(args: dict[str, Any]) -> ToolResult:
        output, ok, metadata = build_session_closeout(_bounded_int(args.get("limit"), 8))
        return ToolResult("session_closeout", ok, output, metadata)

    def save_session_closeout(args: dict[str, Any]) -> ToolResult:
        output, ok, metadata = build_session_closeout(_bounded_int(args.get("limit"), 8))
        path = vault.write_reflection("Session Closeout", output)
        path_display = _safe_vault_path_display(path, vault)
        metadata["path"] = str(path)
        metadata["path_display"] = path_display
        metadata["writes_files"] = True
        metadata["writes_notes"] = True
        return ToolResult(
            "save_session_closeout",
            ok,
            f"Session closeout saved: {path_display}\n\n{output}",
            metadata,
        )

    registry.register(Tool("return_brief", "Combine readiness, recent activity, blockers, and safe next actions into one catch-up brief.", RiskLevel.READ_ONLY, return_brief, "continuity"))
    registry.register(Tool("save_return_brief", "Write the current return brief to Obsidian Reflections.", RiskLevel.LOCAL_SAFE, save_return_brief, "continuity"))
    registry.register(Tool("handoff_brief", "Combine readiness, activity, build progress, blockers, safety reminders, and safe next actions into one resume brief.", RiskLevel.READ_ONLY, handoff_brief, "continuity"))
    registry.register(Tool("save_handoff_brief", "Write the current handoff brief to Obsidian Reflections.", RiskLevel.LOCAL_SAFE, save_handoff_brief, "continuity"))
    registry.register(Tool("session_closeout", "Build a safe end-of-session closeout with activity, build progress, work queue, and approval review.", RiskLevel.READ_ONLY, session_closeout, "continuity"))
    registry.register(Tool("save_session_closeout", "Write the current session closeout to Obsidian Reflections.", RiskLevel.LOCAL_SAFE, save_session_closeout, "continuity"))
    organize_note = organize.make_organize_tools(store, vault)
    registry.register(
        Tool(
            "organize_note",
            "Turn prefixed brain-dump lines into tasks, goals, memories, and decisions.",
            RiskLevel.LOCAL_SAFE,
            organize_note,
            "organize",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset(
                    {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
                ),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=organize.organize_note_auto_mutation_operation_key,
                semantic_preflight=organize.organize_note_auto_mutation_preflight,
                semantic_preflight_result_builder=(
                    organize.organize_note_auto_mutation_preflight_result
                ),
            ),
            _tool_argument_contract(required_strings=("text",)),
        )
    )
    recent_tool_runs, verification_receipt, runtime_trace_receipt, execution_audit_gate, execution_recovery_packet, after_action_learning_packet, execution_health_report, recovery_closure_checklist, execution_learning_closure_packet = audit.make_audit_tools(store)
    registry.register(
        Tool(
            "recent_tool_runs",
            "Show recent tool execution audit records.",
            RiskLevel.READ_ONLY,
            recent_tool_runs,
            "audit",
            argument_contract=_tool_argument_contract(
                optional_integer_ranges=(("limit", 1, 200),),
            ),
        )
    )
    registry.register(
        Tool(
            "verification_receipt",
            "Build a read-only after-action verification receipt from one logged tool run.",
            RiskLevel.READ_ONLY,
            verification_receipt,
            "audit",
            argument_contract=_tool_argument_contract(
                optional_strings=("expectation", "expected", "target"),
                optional_integer_inputs=("run_id", "id", "tool_run_id"),
            ),
        )
    )
    registry.register(
        Tool(
            "runtime_trace_receipt",
            "Show the latest Jarvis runtime route, stages, approvals, and tool results from stored assistant metadata.",
            RiskLevel.READ_ONLY,
            runtime_trace_receipt,
            "audit",
            argument_contract=_tool_argument_contract(
                optional_strings=("session_id",),
                optional_integer_inputs=("message_id", "id", "trace_id"),
                optional_integer_ranges=(("limit", 1, 200),),
            ),
        )
    )
    registry.register(
        Tool(
            "execution_audit_gate",
            "Scan recent tool runs for failed execution, missing approval evidence, and recovery needs before trusting completion.",
            RiskLevel.READ_ONLY,
            execution_audit_gate,
            "audit",
            argument_contract=_tool_argument_contract(
                optional_integer_ranges=(("limit", 1, 200),),
            ),
        )
    )
    registry.register(
        Tool(
            "execution_recovery_packet",
            "Build a read-only recovery packet for the latest failed, blocked, or weakly evidenced tool run.",
            RiskLevel.READ_ONLY,
            execution_recovery_packet,
            "audit",
            argument_contract=_tool_argument_contract(
                optional_integer_inputs=("run_id", "id", "tool_run_id"),
                optional_integer_ranges=(("limit", 1, 200),),
            ),
        )
    )
    registry.register(
        Tool(
            "after_action_learning_packet",
            "Turn one execution audit row into read-only learning, task, skill, or regression-test candidates.",
            RiskLevel.READ_ONLY,
            after_action_learning_packet,
            "audit",
            argument_contract=_tool_argument_contract(
                optional_integer_inputs=("run_id", "id", "tool_run_id"),
                optional_integer_ranges=(("limit", 1, 200),),
            ),
        )
    )
    registry.register(
        Tool(
            "execution_health_report",
            "Summarize recent action-run health, repeated failures, approval-chain risk, verification coverage, and next safe audit command.",
            RiskLevel.READ_ONLY,
            execution_health_report,
            "audit",
            argument_contract=_tool_argument_contract(
                optional_integer_ranges=(("limit", 1, 200),),
            ),
        )
    )
    registry.register(
        Tool(
            "recovery_closure_checklist",
            "Show the exact verification, recovery, learning, approval, and failure-promotion proofs needed to close execution-health debt.",
            RiskLevel.READ_ONLY,
            recovery_closure_checklist,
            "audit",
            argument_contract=_tool_argument_contract(
                optional_integer_ranges=(("limit", 1, 200),),
            ),
        )
    )
    registry.register(
        Tool(
            "execution_learning_closure_packet",
            "Gate whether execution learning debt is closed with verification, recovery, after-action learning, and repeated-failure promotion proof.",
            RiskLevel.READ_ONLY,
            execution_learning_closure_packet,
            "audit",
            argument_contract=_tool_argument_contract(
                optional_integer_inputs=("run_id", "id", "tool_run_id"),
                optional_integer_ranges=(("limit", 1, 200),),
            ),
        )
    )
    (
        auto_mutation_reconciliation_queue,
        auto_mutation_reconciliation_status,
        resolve_auto_mutation_receipt,
    ) = make_auto_mutation_reconciliation_tools(store, session_id)
    registry.register(
        Tool(
            "auto_mutation_reconciliation_queue",
            "List content-free uncertain local mutations awaiting explicit operator reconciliation.",
            RiskLevel.READ_ONLY,
            auto_mutation_reconciliation_queue,
            "audit",
            argument_contract=_tool_argument_contract(optional_integers=("limit",)),
        )
    )
    registry.register(
        Tool(
            "auto_mutation_reconciliation_status",
            "Inspect one content-free uncertain local mutation and create a short-lived review receipt.",
            RiskLevel.READ_ONLY,
            auto_mutation_reconciliation_status,
            "audit",
            argument_contract=_tool_argument_contract(required_integers=("receipt_id",)),
        )
    )
    registry.register(
        Tool(
            "resolve_auto_mutation_receipt",
            "Record an explicit operator reconciliation decision without rerunning the mutation.",
            RiskLevel.LOCAL_SAFE,
            resolve_auto_mutation_receipt,
            "audit",
            argument_contract=_tool_argument_contract(
                required_strings=("disposition",),
                required_integers=("receipt_id",),
            ),
        )
    )
    (
        list_pending_approvals,
        dismiss_pending_approval,
        inspect_pending_approval,
        approval_execution_packet,
        approval_resume_packet,
        approval_history,
        approval_chain_proof,
        approval_queue_summary,
        approval_readiness_packet,
        approve_pending_approval,
        review_pending_approvals,
        save_approval_review,
    ) = approvals.make_approval_tools(store, vault)
    registry.register(
        Tool(
            "list_pending_approvals",
            "List high-risk requests waiting for user approval.",
            RiskLevel.READ_ONLY,
            list_pending_approvals,
            "approvals",
            argument_contract=_tool_argument_contract(
                optional_strings=("status",),
                optional_integer_ranges=(("limit", 1, 100),),
            ),
        )
    )
    registry.register(Tool("dismiss_pending_approval", "Dismiss a pending approval request without running it.", RiskLevel.LOCAL_SAFE, dismiss_pending_approval, "approvals"))
    registry.register(
        Tool(
            "inspect_pending_approval",
            "Inspect one pending approval request, risk reason, and approve/dismiss commands.",
            RiskLevel.READ_ONLY,
            inspect_pending_approval,
            "approvals",
            argument_contract=_tool_argument_contract(
                optional_strings=("status",),
                optional_integer_inputs=("approval_id", "id", "target"),
            ),
        )
    )
    registry.register(
        Tool(
            "approval_execution_packet",
            "Preview exactly what approving one request would rerun, without approving it.",
            RiskLevel.READ_ONLY,
            approval_execution_packet,
            "approvals",
            argument_contract=_tool_argument_contract(
                optional_strings=("status",),
                optional_integer_inputs=("approval_id", "id", "target"),
            ),
        )
    )
    registry.register(
        Tool(
            "approval_resume_packet",
            "Preview the one-shot resume contract for one queued approval without approving or rerunning it.",
            RiskLevel.READ_ONLY,
            approval_resume_packet,
            "approvals",
            argument_contract=_tool_argument_contract(
                optional_integer_inputs=("approval_id", "id", "target"),
            ),
        )
    )
    registry.register(
        Tool(
            "approval_history",
            "Show recent approval decisions and pending approval records without acting on them.",
            RiskLevel.READ_ONLY,
            approval_history,
            "approvals",
            argument_contract=_tool_argument_contract(
                optional_integer_ranges=(("limit", 1, 50),),
            ),
        )
    )
    registry.register(
        Tool(
            "approval_chain_proof",
            "Verify whether an approval has an approved linked rerun that can count as execution proof.",
            RiskLevel.READ_ONLY,
            approval_chain_proof,
            "approvals",
            argument_contract=_tool_argument_contract(
                optional_integer_inputs=("approval_id", "id", "target"),
            ),
        )
    )
    registry.register(
        Tool(
            "approval_queue_summary",
            "Show a compact pending approval queue summary without acting on approvals.",
            RiskLevel.READ_ONLY,
            approval_queue_summary,
            "approvals",
            argument_contract=_tool_argument_contract(
                optional_strings=("status",),
                optional_integer_ranges=(("limit", 1, 20),),
            ),
        )
    )
    registry.register(
        Tool(
            "approval_readiness_packet",
            "Check approval queue position, staleness, exact arguments, and next safe command before last-look approval.",
            RiskLevel.READ_ONLY,
            approval_readiness_packet,
            "approvals",
            argument_contract=_tool_argument_contract(
                optional_integer_inputs=("approval_id", "id", "target"),
            ),
        )
    )
    registry.register(Tool("approve_pending_approval", "Approve and rerun a pending high-risk request.", RiskLevel.LOCAL_SAFE, approve_pending_approval, "approvals"))
    registry.register(
        Tool(
            "review_pending_approvals",
            "Explain pending approval risks and safer checks before rerunning blocked requests.",
            RiskLevel.READ_ONLY,
            review_pending_approvals,
            "approvals",
            argument_contract=_tool_argument_contract(
                optional_strings=("status",),
                optional_integer_ranges=(("limit", 1, 100),),
            ),
        )
    )
    registry.register(Tool("save_approval_review", "Write a reviewed pending-approval risk summary to Obsidian.", RiskLevel.LOCAL_SAFE, save_approval_review, "approvals"))
    brain_loop_report, export_state_snapshot = state.make_state_tools(store, vault)
    registry.register(Tool("brain_loop_report", "Show Jarvis's current perception-memory-skill-planning-action-verification loop.", RiskLevel.READ_ONLY, brain_loop_report, "state"))
    registry.register(
        Tool(
            "export_state_snapshot",
            "Write a consolidated Current Context snapshot to Obsidian.",
            RiskLevel.LOCAL_SAFE,
            export_state_snapshot,
            "state",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=lambda _args: {"projection": "current_context"},
            ),
            _tool_argument_contract(),
        )
    )
    add_profile_note, read_profile = profile.make_profile_tools(store, vault)
    registry.register(
        Tool(
            "add_profile_note",
            "Add a curated note to Profile.md and memory.",
            RiskLevel.LOCAL_SAFE,
            add_profile_note,
            "profile",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset(
                    {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
                ),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=profile.add_profile_note_auto_mutation_operation_key,
                semantic_preflight=profile.add_profile_note_auto_mutation_preflight,
            ),
            _tool_argument_contract(
                required_strings=("body",),
                optional_strings=("heading", "category"),
            ),
        )
    )
    registry.register(
        Tool(
            "read_profile",
            "Read Jarvis Profile.md.",
            RiskLevel.READ_ONLY,
            read_profile,
            "profile",
            argument_contract=_tool_argument_contract(
                optional_integers=("max_chars",),
            ),
        )
    )
    set_preference, list_preferences, get_preference, set_preference_status = preferences.make_preference_tools(store, vault)
    registry.register(
        Tool(
            "set_preference",
            "Save or update a structured user preference.",
            RiskLevel.LOCAL_SAFE,
            set_preference,
            "preferences",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=preferences.preference_auto_mutation_operation_key,
                semantic_preflight=preferences.preference_auto_mutation_preflight,
            ),
            argument_contract=_tool_argument_contract(
                required_strings=("key", "value"),
                optional_strings=("category",),
            ),
        )
    )
    registry.register(
        Tool(
            "list_preferences",
            "List structured user preferences.",
            RiskLevel.READ_ONLY,
            list_preferences,
            "preferences",
            argument_contract=_tool_argument_contract(
                optional_strings=("category", "status"),
                optional_integers=("limit",),
            ),
        )
    )
    registry.register(
        Tool(
            "get_preference",
            "Inspect one structured user preference.",
            RiskLevel.READ_ONLY,
            get_preference,
            "preferences",
            argument_contract=_tool_argument_contract(
                required_strings=("key",),
                optional_strings=("category",),
            ),
        )
    )
    registry.register(
        Tool(
            "set_preference_status",
            "Mark a preference active or retired.",
            RiskLevel.LOCAL_SAFE,
            set_preference_status,
            "preferences",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
            ),
            _tool_argument_contract(
                required_strings=("status",),
                required_integers=("preference_id",),
            ),
        )
    )
    add_person, list_people, get_person, log_interaction = people.make_people_tools(store, vault)
    registry.register(
        Tool(
            "add_person",
            "Save a person profile with relation and notes.",
            RiskLevel.LOCAL_SAFE,
            add_person,
            "people",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=people.person_auto_mutation_operation_key,
                semantic_preflight=people.person_auto_mutation_preflight,
            ),
            _tool_argument_contract(
                required_strings=("name",),
                optional_strings=("relation", "notes"),
            ),
        )
    )
    registry.register(
        Tool(
            "list_people",
            "List saved people profiles.",
            RiskLevel.READ_ONLY,
            list_people,
            "people",
            argument_contract=_tool_argument_contract(
                optional_integers=("limit",),
            ),
        )
    )
    registry.register(
        Tool(
            "get_person",
            "Inspect one saved person profile.",
            RiskLevel.READ_ONLY,
            get_person,
            "people",
            argument_contract=_tool_argument_contract(
                optional_strings=("name",),
                optional_integers=("person_id", "limit"),
            ),
        )
    )
    registry.register(
        Tool(
            "log_interaction",
            "Log an interaction with a saved person.",
            RiskLevel.LOCAL_SAFE,
            log_interaction,
            "people",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset(
                    {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
                ),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=(
                    people.make_log_interaction_auto_mutation_operation_key(store)
                ),
                semantic_preflight=(
                    people.make_log_interaction_auto_mutation_preflight(store)
                ),
                semantic_preflight_result_builder=(
                    people.log_interaction_auto_mutation_preflight_result
                ),
                definite_no_effect_failure_reasons=frozenset(
                    {
                        "bad_person_id",
                        "invalid_name",
                        "invalid_unicode",
                        "missing_person",
                        "missing_person_id",
                        "missing_summary",
                        "person_target_mismatch",
                    }
                ),
            ),
            _tool_argument_contract(
                required_strings=("summary",),
                optional_strings=("name", "happened_at"),
                optional_integer_inputs=("person_id",),
            ),
        )
    )
    (
        record_decision,
        record_decision_outcome,
        list_decisions,
        get_decision,
        set_decision_status,
    ) = decisions.make_decision_tools(store, vault)
    registry.register(
        Tool(
            "record_decision",
            "Record a durable decision with rationale and impact.",
            RiskLevel.LOCAL_SAFE,
            record_decision,
            "decisions",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=decisions.decision_auto_mutation_operation_key,
                semantic_preflight=decisions.decision_auto_mutation_preflight,
            ),
            argument_contract=_tool_argument_contract(
                required_strings=("title",),
                optional_strings=("rationale", "impact"),
            ),
        )
    )
    registry.register(
        Tool(
            "record_decision_outcome",
            "Append a user-reported, not-verified outcome to one durable decision.",
            RiskLevel.LOCAL_SAFE,
            record_decision_outcome,
            "decisions",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset(
                    {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
                ),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=(
                    decisions.decision_outcome_auto_mutation_operation_key
                ),
                semantic_preflight=(
                    decisions.make_decision_outcome_auto_mutation_preflight(store)
                ),
                definite_no_effect_failure_reasons=frozenset(
                    {"missing_decision", "not_found", "outcome_limit_reached"}
                ),
            ),
            argument_contract=_tool_argument_contract(
                required_strings=("summary",),
                required_integers=("decision_id",),
            ),
        )
    )
    registry.register(
        Tool(
            "list_decisions",
            "List durable decisions.",
            RiskLevel.READ_ONLY,
            list_decisions,
            "decisions",
            argument_contract=_tool_argument_contract(
                optional_strings=("status",),
                optional_integers=("limit",),
            ),
        )
    )
    registry.register(
        Tool(
            "get_decision",
            "Inspect one durable decision.",
            RiskLevel.READ_ONLY,
            get_decision,
            "decisions",
            argument_contract=_tool_argument_contract(
                required_integers=("decision_id",),
            ),
        )
    )
    registry.register(
        Tool(
            "set_decision_status",
            "Mark a decision active, superseded, or retired.",
            RiskLevel.LOCAL_SAFE,
            set_decision_status,
            "decisions",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
            ),
            _tool_argument_contract(
                required_strings=("status",),
                required_integers=("decision_id",),
            ),
        )
    )
    list_jarvis_notes, search_jarvis_notes, read_jarvis_note, outline_jarvis_note, write_jarvis_note = notes.make_note_tools(vault)
    registry.register(
        Tool(
            "list_jarvis_notes",
            "List Markdown notes inside the Jarvis Obsidian folder without opening their contents.",
            RiskLevel.READ_ONLY,
            list_jarvis_notes,
            "notes",
            argument_contract=_tool_argument_contract(
                optional_strings=("folder",),
                optional_integers=("limit",),
            ),
        )
    )
    registry.register(
        Tool(
            "search_jarvis_notes",
            "Search Markdown notes inside the Jarvis Obsidian folder.",
            RiskLevel.READ_ONLY,
            search_jarvis_notes,
            "notes",
            argument_contract=_tool_argument_contract(
                required_strings=("query",),
                optional_integers=("limit",),
            ),
        )
    )
    registry.register(
        Tool(
            "read_jarvis_note",
            "Read a Markdown note inside the Jarvis Obsidian folder.",
            RiskLevel.READ_ONLY,
            read_jarvis_note,
            "notes",
            argument_contract=_tool_argument_contract(
                required_strings=("path",),
                optional_integers=("max_chars",),
            ),
        )
    )
    registry.register(
        Tool(
            "outline_jarvis_note",
            "Summarize headings, tasks, and links for a Markdown note inside the Jarvis Obsidian folder.",
            RiskLevel.READ_ONLY,
            outline_jarvis_note,
            "notes",
            argument_contract=_tool_argument_contract(
                required_strings=("path",),
            ),
        )
    )
    registry.register(
        Tool(
            "write_jarvis_note",
            "Create or append a Markdown note inside the Jarvis Obsidian folder.",
            RiskLevel.LOCAL_SAFE,
            write_jarvis_note,
            "notes",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=(
                    notes.make_write_jarvis_note_auto_mutation_operation_key(vault)
                ),
                semantic_preflight=(
                    notes.make_write_jarvis_note_auto_mutation_preflight(vault)
                ),
                semantic_preflight_result_builder=(
                    notes.make_write_jarvis_note_auto_mutation_preflight_result(vault)
                ),
            ),
            _tool_argument_contract(
                required_strings=("path", "body"),
                optional_strings=("mode",),
            ),
        )
    )
    if config is not None:
        registry.register(
            Tool(
                "jarvis_doctor",
                "Diagnose Jarvis setup and recommend next install/start commands.",
                RiskLevel.READ_ONLY,
                doctor.make_doctor_tool(store, vault, config, storage_fallback, registry.list),
            )
        )
    create_goal, list_goals, goal_status, add_goal_step, complete_goal_step, set_goal_status, export_goal, next_actions = goals.make_goal_tools(store, vault)
    registry.register(
        Tool(
            "create_goal",
            "Create a durable goal/project.",
            RiskLevel.LOCAL_SAFE,
            create_goal,
            "goals",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=goals.create_goal_auto_mutation_operation_key,
                semantic_preflight=goals.create_goal_auto_mutation_preflight,
                semantic_preflight_result_builder=(
                    goals.create_goal_auto_mutation_preflight_result
                ),
            ),
            _tool_argument_contract(
                required_strings=("title",),
                optional_strings=("purpose", "horizon"),
            ),
        )
    )
    registry.register(
        Tool(
            "list_goals",
            "List durable goals/projects.",
            RiskLevel.READ_ONLY,
            list_goals,
            "goals",
            argument_contract=_tool_argument_contract(
                optional_strings=("status",),
                optional_integers=("limit",),
            ),
        )
    )
    registry.register(
        Tool(
            "goal_status",
            "Show a goal and its steps.",
            RiskLevel.READ_ONLY,
            goal_status,
            "goals",
            argument_contract=_tool_argument_contract(
                required_integers=("goal_id",),
            ),
        )
    )
    registry.register(
        Tool(
            "add_goal_step",
            "Add a step to a goal.",
            RiskLevel.LOCAL_SAFE,
            add_goal_step,
            "goals",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=goals.add_goal_step_auto_mutation_operation_key,
                semantic_preflight=goals.make_add_goal_step_auto_mutation_preflight(store),
                semantic_preflight_result_builder=(
                    goals.add_goal_step_auto_mutation_preflight_result
                ),
                definite_no_effect_failure_reasons=frozenset({"missing_goal"}),
            ),
            ToolArgumentContract(
                TOOL_ARGUMENT_CONTRACT_VERSION,
                (
                    ToolArgumentSpec(
                        "goal_id",
                        frozenset({ToolArgumentType.INTEGER, ToolArgumentType.STRING}),
                        True,
                    ),
                    ToolArgumentSpec("body", frozenset({ToolArgumentType.STRING}), True),
                ),
                False,
            ),
        )
    )
    registry.register(
        Tool(
            "complete_goal_step",
            "Mark a goal step complete.",
            RiskLevel.LOCAL_SAFE,
            complete_goal_step,
            "goals",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=goals.complete_goal_step_auto_mutation_operation_key,
                semantic_preflight=goals.make_complete_goal_step_auto_mutation_preflight(store),
                semantic_preflight_result_builder=(
                    goals.complete_goal_step_auto_mutation_preflight_result
                ),
                definite_no_effect_failure_reasons=frozenset({"missing_step"}),
            ),
            ToolArgumentContract(
                TOOL_ARGUMENT_CONTRACT_VERSION,
                (
                    ToolArgumentSpec(
                        "step_id",
                        frozenset({ToolArgumentType.INTEGER, ToolArgumentType.STRING}),
                        True,
                    ),
                ),
                False,
            ),
        )
    )
    registry.register(
        Tool(
            "set_goal_status",
            "Set a goal status.",
            RiskLevel.LOCAL_SAFE,
            set_goal_status,
            "goals",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=goals.set_goal_status_auto_mutation_operation_key,
                semantic_preflight=goals.make_set_goal_status_auto_mutation_preflight(store),
                semantic_preflight_result_builder=(
                    goals.set_goal_status_auto_mutation_preflight_result
                ),
                definite_no_effect_failure_reasons=frozenset({"missing_goal"}),
            ),
            ToolArgumentContract(
                TOOL_ARGUMENT_CONTRACT_VERSION,
                (
                    ToolArgumentSpec(
                        "goal_id",
                        frozenset({ToolArgumentType.INTEGER, ToolArgumentType.STRING}),
                        True,
                    ),
                    ToolArgumentSpec("status", frozenset({ToolArgumentType.STRING}), True),
                ),
                False,
            ),
        )
    )
    registry.register(
        Tool(
            "export_goal",
            "Export a goal/project note to Obsidian.",
            RiskLevel.LOCAL_SAFE,
            export_goal,
            "goals",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset(
                    {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
                ),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=goals.export_goal_auto_mutation_operation_key,
                semantic_preflight=goals.make_export_goal_auto_mutation_preflight(store),
                semantic_preflight_result_builder=goals.export_goal_auto_mutation_preflight_result,
                definite_no_effect_failure_reasons=frozenset({"missing_goal"}),
            ),
            ToolArgumentContract(
                TOOL_ARGUMENT_CONTRACT_VERSION,
                (
                    ToolArgumentSpec(
                        "goal_id",
                        frozenset({ToolArgumentType.INTEGER, ToolArgumentType.STRING}),
                        True,
                    ),
                ),
                False,
            ),
        )
    )
    registry.register(
        Tool(
            "next_actions",
            "List next actions across active goals.",
            RiskLevel.READ_ONLY,
            next_actions,
            "goals",
            argument_contract=_tool_argument_contract(
                optional_integers=("limit",),
            ),
        )
    )
    add_task, list_tasks, inspect_task, search_tasks, overdue_tasks, task_overview, next_task, task_board, complete_task, task_completion_packet, complete_task_with_evidence, update_task_status, update_task_details, export_tasks, preview_tasks_from_note, import_tasks_from_note = tasks.make_task_tools(store, vault)
    registry.register(
        Tool(
            "add_task",
            "Capture a lightweight open task.",
            RiskLevel.LOCAL_SAFE,
            add_task,
            "tasks",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
            ),
            _tool_argument_contract(
                required_strings=("body",),
                optional_strings=("due", "priority"),
            ),
        )
    )
    registry.register(
        Tool(
            "list_tasks",
            "List lightweight tasks.",
            RiskLevel.READ_ONLY,
            list_tasks,
            "tasks",
            argument_contract=_tool_argument_contract(
                optional_strings=("status",),
                optional_integer_inputs=("limit",),
            ),
        )
    )
    registry.register(
        Tool(
            "inspect_task",
            "Inspect one lightweight task without changing it.",
            RiskLevel.READ_ONLY,
            inspect_task,
            "tasks",
            argument_contract=_tool_argument_contract(required_integer_inputs=("task_id",)),
        )
    )
    registry.register(
        Tool(
            "search_tasks",
            "Search lightweight tasks without changing them.",
            RiskLevel.READ_ONLY,
            search_tasks,
            "tasks",
            argument_contract=_tool_argument_contract(
                required_strings=("query",),
                optional_strings=("status",),
                optional_integer_inputs=("limit",),
            ),
        )
    )
    registry.register(
        Tool(
            "overdue_tasks",
            "List open tasks with parseable due labels before today without changing tasks.",
            RiskLevel.READ_ONLY,
            overdue_tasks,
            "tasks",
            argument_contract=_tool_argument_contract(optional_integer_inputs=("limit",)),
        )
    )
    registry.register(
        Tool(
            "task_overview",
            "Summarize task counts and priority work without changing tasks.",
            RiskLevel.READ_ONLY,
            task_overview,
            "tasks",
            argument_contract=_tool_argument_contract(optional_integer_inputs=("limit",)),
        )
    )
    registry.register(
        Tool(
            "next_task",
            "Pick the next open task without changing tasks.",
            RiskLevel.READ_ONLY,
            next_task,
            "tasks",
            argument_contract=_tool_argument_contract(),
        )
    )
    registry.register(
        Tool(
            "task_board",
            "Show tasks grouped by status without changing tasks.",
            RiskLevel.READ_ONLY,
            task_board,
            "tasks",
            argument_contract=_tool_argument_contract(optional_integer_inputs=("limit",)),
        )
    )
    registry.register(
        Tool(
            "complete_task",
            "Mark a lightweight task complete.",
            RiskLevel.LOCAL_SAFE,
            complete_task,
            "tasks",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=tasks.task_status_auto_mutation_operation_key,
                semantic_preflight=tasks.make_complete_task_auto_mutation_preflight(store),
                semantic_preflight_result_builder=(
                    tasks.complete_task_auto_mutation_preflight_result
                ),
                definite_no_effect_failure_reasons=frozenset({"missing_task"}),
                operation_scope="task_status",
                legacy_operation_aliases=frozenset(
                    {
                        "complete_task",
                        "complete_task_with_evidence",
                        "update_task_status",
                    }
                ),
            ),
            _tool_argument_contract(required_integers=("task_id",)),
        )
    )
    registry.register(Tool("task_completion_packet", "Check evidence before marking a task complete.", RiskLevel.READ_ONLY, task_completion_packet, "tasks"))
    registry.register(
        Tool(
            "complete_task_with_evidence",
            "Mark a task complete with durable explicit evidence.",
            RiskLevel.LOCAL_SAFE,
            complete_task_with_evidence,
            "tasks",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset(
                    {
                        AutoMutationEffect.LOCAL_DATABASE,
                        AutoMutationEffect.OBSIDIAN_VAULT,
                    }
                ),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=tasks.task_status_auto_mutation_operation_key,
                semantic_preflight=(
                    tasks.make_complete_task_with_evidence_auto_mutation_preflight(store)
                ),
                semantic_preflight_result_builder=(
                    tasks.complete_task_with_evidence_auto_mutation_preflight_result
                ),
                definite_no_effect_failure_reasons=frozenset(
                    {"missing_task", "recovery_state_changed"}
                ),
                operation_scope="task_status",
                legacy_operation_aliases=frozenset(
                    {
                        "complete_task",
                        "complete_task_with_evidence",
                        "update_task_status",
                    }
                ),
            ),
            _tool_argument_contract(
                optional_strings=("evidence", "verification_run_id"),
                required_integers=("task_id",),
            ),
        )
    )
    registry.register(
        Tool(
            "update_task_status",
            "Pause, reopen, complete, or drop a lightweight task.",
            RiskLevel.LOCAL_SAFE,
            update_task_status,
            "tasks",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=tasks.task_status_auto_mutation_operation_key,
                semantic_preflight=tasks.make_update_task_status_auto_mutation_preflight(store),
                semantic_preflight_result_builder=(
                    tasks.update_task_status_auto_mutation_preflight_result
                ),
                definite_no_effect_failure_reasons=frozenset({"missing_task"}),
                operation_scope="task_status",
                legacy_operation_aliases=frozenset(
                    {
                        "complete_task",
                        "complete_task_with_evidence",
                        "update_task_status",
                    }
                ),
            ),
            _tool_argument_contract(
                required_strings=("status",),
                required_integers=("task_id",),
            ),
        )
    )
    registry.register(
        Tool(
            "update_task_details",
            "Update a lightweight task body, due label, or priority.",
            RiskLevel.LOCAL_SAFE,
            update_task_details,
            "tasks",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset(
                    {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
                ),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=tasks.update_task_details_auto_mutation_operation_key,
                semantic_preflight=(
                    tasks.make_update_task_details_auto_mutation_preflight(store)
                ),
                semantic_preflight_result_builder=(
                    tasks.update_task_details_auto_mutation_preflight_result
                ),
                definite_no_effect_failure_reasons=frozenset({"missing_task"}),
            ),
            _tool_argument_contract(
                optional_strings=("body", "due", "priority"),
                required_integers=("task_id",),
            ),
        )
    )
    registry.register(
        Tool(
            "export_tasks",
            "Export open lightweight tasks to Obsidian.",
            RiskLevel.LOCAL_SAFE,
            export_tasks,
            "tasks",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=tasks.export_tasks_auto_mutation_operation_key,
            ),
            _tool_argument_contract(optional_integers=("limit",)),
        )
    )
    registry.register(Tool("preview_tasks_from_note", "Preview open and completed markdown checkboxes in a Jarvis-owned Obsidian note without importing them.", RiskLevel.READ_ONLY, preview_tasks_from_note, "tasks"))
    registry.register(
        Tool(
            "import_tasks_from_note",
            "Import open markdown checkboxes from a Jarvis-owned Obsidian note into Jarvis tasks.",
            RiskLevel.LOCAL_SAFE,
            import_tasks_from_note,
            "tasks",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset(
                    {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
                ),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=tasks.import_tasks_from_note_auto_mutation_operation_key,
                semantic_preflight=(
                    tasks.make_import_tasks_from_note_auto_mutation_preflight(vault)
                ),
                semantic_preflight_result_builder=(
                    tasks.make_import_tasks_from_note_auto_mutation_preflight_result(vault)
                ),
                definite_no_effect_failure_reasons=frozenset(
                    {
                        "note_not_found",
                        "note_not_regular",
                        "note_read_failed",
                        "note_too_large",
                        "too_many_tasks",
                        "unsafe_path",
                    }
                ),
            ),
            _tool_argument_contract(
                required_strings=("path",),
                optional_strings=("priority",),
            ),
        )
    )
    (
        search_conversations,
        recent_conversation,
        chat_response_health,
        chat_continuity_brief,
        list_sessions,
        export_session,
        summarize_session,
        draft_skill_from_session,
        chat_context,
        chat_prompt_preview,
        chat_loop_preview,
        chat_safety_report,
        save_chat_context,
        session_learning_preview,
    ) = make_conversation_tools(store, vault, session_id, config, chat_brain)
    registry.register(Tool("search_conversations", "Search past conversation messages.", RiskLevel.READ_ONLY, search_conversations, "conversation"))
    registry.register(Tool("recent_conversation", "Show recent conversation messages.", RiskLevel.READ_ONLY, recent_conversation, "conversation"))
    registry.register(Tool("chat_response_health", "Show recent chat response sources, model/fallback counts, and safety boundaries.", RiskLevel.READ_ONLY, chat_response_health, "conversation"))
    registry.register(Tool("chat_continuity_brief", "Summarize the current chat thread, latest response path, and safe next checks without acting.", RiskLevel.READ_ONLY, chat_continuity_brief, "conversation"))
    registry.register(Tool("list_sessions", "List conversation sessions.", RiskLevel.READ_ONLY, list_sessions, "conversation"))
    registry.register(Tool("export_session", "Export a conversation session to Obsidian.", RiskLevel.LOCAL_SAFE, export_session, "conversation"))
    registry.register(Tool("summarize_session", "Write a compact session reflection to Obsidian.", RiskLevel.LOCAL_SAFE, summarize_session, "conversation"))
    registry.register(Tool("draft_skill_from_session", "Draft a reusable skill from the current conversation.", RiskLevel.LOCAL_SAFE, draft_skill_from_session, "conversation"))
    registry.register(Tool("chat_context", "Preview bounded profile, preferences, memories, skills, recent messages, and relevant local goals, commitments, decisions, and relationships.", RiskLevel.READ_ONLY, chat_context, "conversation"))
    registry.register(Tool("chat_prompt_preview", "Preview the conversational system prompt, grounding packet, and safety boundaries without calling a model.", RiskLevel.READ_ONLY, chat_prompt_preview, "conversation"))
    registry.register(Tool("chat_loop_preview", "Preview Jarvis's perceive-ground-classify-reply conversational loop without calling a model or executing tools.", RiskLevel.READ_ONLY, chat_loop_preview, "conversation"))
    registry.register(Tool("chat_safety_report", "Show how conversational answers are grounded and kept behind execution approvals.", RiskLevel.READ_ONLY, chat_safety_report, "conversation"))
    registry.register(Tool("save_chat_context", "Write the current chat-context preview to Obsidian Reflections.", RiskLevel.LOCAL_SAFE, save_chat_context, "conversation"))
    registry.register(Tool("session_learning_preview", "Suggest possible memories, preferences, tasks, and skills from the current session without saving anything.", RiskLevel.READ_ONLY, session_learning_preview, "learning"))
    registry.register(Tool("list_files", "List files in a directory. Requires approval because filenames can reveal private data.", RiskLevel.PERSONAL_DATA, files.list_files, "files"))
    registry.register(Tool("read_text_file", "Read a known text file. Requires approval because file contents can be private.", RiskLevel.PERSONAL_DATA, files.read_text_file, "files"))
    registry.register(Tool("write_text_file", "Write a text file. Requires approval outside low-risk mode.", RiskLevel.HIGH_RISK, files.write_text_file, "files"))
    registry.register(Tool("find_files", "Find files by filename under a directory. Requires approval because filenames can reveal private data.", RiskLevel.PERSONAL_DATA, files.find_files, "files"))
    registry.register(Tool("run_shell_command", "Run a local command without shell expansion. Always requires approval.", RiskLevel.HIGH_RISK, shell.run_shell_command, "code"))
    ingest_obsidian_inbox, clear_obsidian_inbox, recent_file_digest = ingest.make_ingest_tools(store, vault, config)
    resolve_clear_inbox_approval = ingest.make_clear_inbox_approval_resolver(vault)
    registry.register(
        Tool(
            "ingest_obsidian_inbox",
            "Index Jarvis/Inbox.md notes into memory.",
            RiskLevel.LOCAL_SAFE,
            ingest_obsidian_inbox,
            "ingest",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset(
                    {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
                ),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=ingest.make_inbox_ingest_auto_mutation_operation_key(vault),
                execution_args_builder=ingest.inbox_ingest_auto_mutation_execution_args,
                execution_argument_contract=_tool_argument_contract(
                    required_strings=("_auto_mutation_inbox_snapshot_binding",),
                ),
                definite_no_effect_failure_reasons=frozenset(
                    {"inbox_source_changed", "inbox_source_too_large"}
                ),
            ),
            argument_contract=_tool_argument_contract(
                optional_integer_ranges=(
                    ("limit", 1, ingest.MAX_INBOX_INGEST_LIMIT),
                ),
            ),
        )
    )
    registry.register(
        Tool(
            "clear_obsidian_inbox",
            "Reset only the exact reviewed Jarvis/Inbox.md snapshot. Requires approval.",
            RiskLevel.HIGH_RISK,
            clear_obsidian_inbox,
            "ingest",
            argument_contract=_tool_argument_contract(),
            approval_argument_resolver=resolve_clear_inbox_approval,
            approval_argument_contract=_tool_argument_contract(
                required_strings=("target_binding",),
            ),
        )
    )
    registry.register(
        Tool(
            "recent_file_digest",
            "Write a metadata-only digest of recently changed watched files.",
            RiskLevel.LOCAL_SAFE,
            recent_file_digest,
            "ingest",
            argument_contract=_tool_argument_contract(
                optional_integer_ranges=(
                    ("hours", 1, ingest.MAX_DIGEST_HOURS),
                    ("limit", 1, ingest.MAX_DIGEST_LIMIT),
                ),
            ),
        )
    )
    registry.register(Tool("system_info", "Read basic system information.", RiskLevel.READ_ONLY, system.system_info, "system"))
    registry.register(Tool("setup_check", "Check optional Jarvis runtime dependencies.", RiskLevel.READ_ONLY, system.setup_check, "system"))
    registry.register(Tool("list_running_apps", "List visible running applications. Requires approval because app activity is private.", RiskLevel.PERSONAL_DATA, system.list_running_apps, "system"))
    registry.register(Tool("frontmost_app", "Get the currently focused app. Requires approval because app activity is private.", RiskLevel.PERSONAL_DATA, system.frontmost_app, "system"))
    registry.register(Tool("stop_jarvis", "Acknowledge a non-destructive stop/cancel request without hiding messages or changing approval state.", RiskLevel.READ_ONLY, system.stop_jarvis, "system"))
    registry.register(Tool("telegram_control_restart_guidance", "Explain the read-only Telegram control daemon status path; Jarvis never restarts local daemons from chat.", RiskLevel.READ_ONLY, system.telegram_control_restart_guidance, "system"))
    registry.register(
        Tool(
            "restart_target_clarification",
            "Clarify an ambiguous restart follow-up without restarting any service or controlling the computer.",
            RiskLevel.READ_ONLY,
            system.restart_target_clarification,
            "system",
            argument_contract=_tool_argument_contract(),
        )
    )
    registry.register(
        Tool(
            "open_application",
            "Open a macOS application.",
            RiskLevel.HIGH_RISK,
            system.open_application,
            "system",
            argument_contract=_tool_argument_contract(required_strings=("name",)),
            approval_argument_resolver=system.resolve_open_application_approval,
            approval_argument_contract=_tool_argument_contract(required_strings=("name",)),
        )
    )
    registry.register(Tool("volume", "Read or set system volume.", RiskLevel.LOCAL_SAFE, system.volume, "system"))
    registry.register(Tool("get_clipboard", "Read the current clipboard contents.", RiskLevel.PERSONAL_DATA, system.get_clipboard, "system"))
    registry.register(Tool("set_clipboard", "Copy text to the clipboard.", RiskLevel.LOCAL_SAFE, system.set_clipboard, "system"))
    registry.register(Tool("voice_input_plan", "Plan microphone, ASR, wake-word, transcript, and approval boundaries without recording audio.", RiskLevel.READ_ONLY, voice.voice_input_plan, "voice"))
    registry.register(
        Tool(
            "voice_setup_check",
            "Check voice input and ASR readiness without requesting microphone access or recording audio.",
            RiskLevel.READ_ONLY,
            voice.voice_setup_check,
            "voice",
            argument_contract=_tool_argument_contract(),
        )
    )
    registry.register(Tool("voice_capture_privacy_packet", "Preview the visible push-to-talk microphone privacy boundary before browser capture.", RiskLevel.READ_ONLY, voice.voice_capture_privacy_packet, "voice"))
    registry.register(Tool("voice_native_microphone_gate_packet", "Gate future native microphone capture behind visible permission receipts and transcript confirmation proof without recording audio.", RiskLevel.READ_ONLY, voice.voice_native_microphone_gate_packet, "voice"))
    registry.register(Tool("voice_file_transcription_plan", "Plan safe user-supplied audio-file transcription without reading audio or saving transcripts.", RiskLevel.READ_ONLY, voice.voice_file_transcription_plan, "voice"))
    registry.register(Tool("voice_audio_file_gate_packet", "Gate whether a user-supplied audio file is ready for future approved transcription without opening it.", RiskLevel.READ_ONLY, voice.voice_audio_file_gate_packet, "voice"))
    registry.register(Tool("voice_audio_file_transcription_preview", "Transcribe an explicitly supplied audio file into a temporary preview. Always requires approval.", RiskLevel.PERSONAL_DATA, voice.voice_audio_file_transcription_preview, "voice"))
    registry.register(Tool("voice_transcript_review", "Review a supplied voice transcript and preview routing without executing tools.", RiskLevel.READ_ONLY, voice.make_voice_transcript_review(registry.get), "voice"))
    registry.register(Tool("voice_confirmation_packet", "Create a read-only confirmation packet for a supplied voice transcript.", RiskLevel.READ_ONLY, voice.make_voice_confirmation_packet(registry.get), "voice"))
    registry.register(Tool("voice_confirmation_receipt", "Create an auditable read-only confirmation receipt before routing a supplied voice transcript.", RiskLevel.READ_ONLY, voice.make_voice_confirmation_receipt(registry.get), "voice"))
    registry.register(Tool("voice_confirmation_audit_ledger", "Require supplied voice privacy and confirmation receipt proof before command-intake routing can count.", RiskLevel.READ_ONLY, voice.make_voice_confirmation_audit_ledger(registry.get), "voice"))
    registry.register(Tool("voice_route_gate_packet", "Gate whether a confirmed spoken transcript may enter normal routing without executing it.", RiskLevel.READ_ONLY, voice.make_voice_route_gate_packet(registry.get), "voice"))
    registry.register(Tool("voice_route_proof_bundle", "Bundle voice privacy, transcript receipt, route gate, approval boundary, and proof commands before spoken routing.", RiskLevel.READ_ONLY, voice.make_voice_route_proof_bundle(registry.get), "voice"))
    registry.register(Tool("voice_runtime_bridge_packet", "Bridge a confirmed spoken transcript into command-intake proof only without executing or queuing approvals.", RiskLevel.READ_ONLY, voice.make_voice_runtime_bridge_packet(registry.get), "voice"))
    registry.register(Tool("voice_command_cockpit", "Consolidate voice confirmation, route gate, proof bundle, and runtime bridge before command intake.", RiskLevel.READ_ONLY, voice.make_voice_command_cockpit(registry.get), "voice"))
    registry.register(Tool("voice_action_audit_packet", "Audit a confirmed spoken transcript before command intake so speech never becomes direct execution.", RiskLevel.READ_ONLY, voice.make_voice_action_audit_packet(registry.get), "voice"))
    registry.register(Tool("voice_execution_handoff_packet", "Package a confirmed spoken transcript for command-intake handoff after voice action audit without executing.", RiskLevel.READ_ONLY, voice.make_voice_execution_handoff_packet(registry.get), "voice"))
    registry.register(Tool("voice_post_run_closure_packet", "Close post-run verification, audit, health, and learning proof for a spoken command before the next voice review.", RiskLevel.READ_ONLY, voice.make_voice_post_run_closure_packet(registry.get), "voice"))
    registry.register(Tool("voice_cycle_ledger", "Bind the full confirmed-speech lifecycle into one prior-command proof ledger before the next voice review.", RiskLevel.READ_ONLY, voice.make_voice_cycle_ledger(registry.get), "voice"))
    registry.register(Tool("voice_command_lifecycle", "Show the read-only voice-command proof chain from setup through confirmation receipt, route proof, runtime bridge, handoff, approval boundary, audit, and closure.", RiskLevel.READ_ONLY, voice.make_voice_command_lifecycle(registry.get), "voice"))
    registry.register(Tool("voice_stop_intent_packet", "Separate stop/cancel/rerecord speech from transcript routing without deleting or executing anything.", RiskLevel.READ_ONLY, voice.voice_stop_intent_packet, "voice"))
    registry.register(Tool("voice_reply_preview", "Preview a spoken response packet without speaking or recording audio.", RiskLevel.READ_ONLY, voice.voice_reply_preview, "voice"))
    registry.register(Tool("spoken_turn_rehearsal", "Preview chat-vs-tool routing plus a spoken-response packet without speaking or executing tools.", RiskLevel.READ_ONLY, voice.make_spoken_turn_rehearsal(registry.get), "voice"))
    registry.register(
        Tool(
            "speak",
            "Speak text aloud with macOS text-to-speech.",
            RiskLevel.HIGH_RISK,
            voice.speak,
            "voice",
            argument_contract=_tool_argument_contract(
                required_strings=("text",),
                optional_strings=("voice",),
                optional_integer_ranges=(("rate", 80, 360),),
                optional_booleans=("wait", "dry_run"),
            ),
            approval_argument_resolver=voice.resolve_speak_approval,
            approval_argument_contract=_tool_argument_contract(
                required_strings=("text",),
                optional_strings=("voice",),
                optional_integer_ranges=(("rate", 80, 360),),
                optional_booleans=("wait", "dry_run"),
            ),
        )
    )
    registry.register(Tool("list_voices", "List available macOS text-to-speech voices.", RiskLevel.READ_ONLY, voice.list_voices, "voice"))
    registry.register(
        Tool(
            "calculate",
            "Calculate a math expression.",
            RiskLevel.READ_ONLY,
            utilities.calculate,
            "utilities",
            argument_contract=_tool_argument_contract(required_strings=("expression",)),
        )
    )
    registry.register(
        Tool(
            "generate_password",
            "Generate a secure random password.",
            RiskLevel.READ_ONLY,
            utilities.generate_password,
            "utilities",
            argument_contract=ToolArgumentContract(
                TOOL_ARGUMENT_CONTRACT_VERSION,
                (
                    ToolArgumentSpec(
                        "length",
                        frozenset({ToolArgumentType.INTEGER, ToolArgumentType.STRING}),
                        False,
                    ),
                    ToolArgumentSpec("include_symbols", frozenset({ToolArgumentType.BOOLEAN}), False),
                ),
                False,
            ),
        )
    )
    registry.register(
        Tool(
            "generate_uuid",
            "Generate a local random UUID v4.",
            RiskLevel.READ_ONLY,
            utilities.generate_uuid,
            "utilities",
            argument_contract=_tool_argument_contract(),
        )
    )
    registry.register(
        Tool(
            "spell_word",
            "Spell a word or short phrase aloud as letters.",
            RiskLevel.READ_ONLY,
            utilities.spell_word,
            "utilities",
            argument_contract=_tool_argument_contract(optional_strings=("word", "text")),
        )
    )
    registry.register(
        Tool(
            "count_text",
            "Count words and characters in short text.",
            RiskLevel.READ_ONLY,
            utilities.count_text,
            "utilities",
            argument_contract=_tool_argument_contract(optional_strings=("text", "phrase")),
        )
    )
    registry.register(
        Tool(
            "transform_text",
            "Transform or clean up short text locally, such as uppercase, camel case, initials, slugify, repeat, or sort words.",
            RiskLevel.READ_ONLY,
            utilities.transform_text,
            "utilities",
            argument_contract=ToolArgumentContract(
                TOOL_ARGUMENT_CONTRACT_VERSION,
                (
                    ToolArgumentSpec("mode", frozenset({ToolArgumentType.STRING}), False),
                    ToolArgumentSpec("text", frozenset({ToolArgumentType.STRING}), False),
                    ToolArgumentSpec("phrase", frozenset({ToolArgumentType.STRING}), False),
                    ToolArgumentSpec(
                        "count",
                        frozenset({ToolArgumentType.INTEGER, ToolArgumentType.STRING}),
                        False,
                    ),
                ),
                False,
            ),
        )
    )
    registry.register(
        Tool(
            "calculate_bmi",
            "Calculate BMI from explicit weight and height units.",
            RiskLevel.READ_ONLY,
            utilities.calculate_bmi,
            "utilities",
            argument_contract=_tool_argument_contract(
                required_strings=("weight_unit", "height_unit"),
                required_numeric_inputs=("weight", "height"),
            ),
        )
    )
    registry.register(
        Tool(
            "convert_units",
            "Convert common units.",
            RiskLevel.READ_ONLY,
            utilities.convert_units,
            "utilities",
            argument_contract=_tool_argument_contract(
                required_strings=("from_unit", "to_unit"),
                required_numeric_inputs=("value",),
            ),
        )
    )
    registry.register(Tool("enable_computer_control", "Enable mouse/keyboard control.", RiskLevel.HIGH_RISK, computer.enable_computer_control, "computer"))
    registry.register(Tool("disable_computer_control", "Disable mouse/keyboard control.", RiskLevel.LOCAL_SAFE, computer.disable_computer_control, "computer"))
    registry.register(Tool("computer_control_status", "Check computer-control status.", RiskLevel.READ_ONLY, computer.computer_control_status, "computer"))
    registry.register(Tool("computer_control_readiness", "Preflight computer-control readiness without observing or controlling the computer.", RiskLevel.READ_ONLY, computer.computer_control_readiness, "computer"))
    registry.register(Tool("computer_task_plan", "Plan a safe observe-act-verify desktop task without observing or controlling the computer.", RiskLevel.READ_ONLY, computer.computer_task_plan, "computer"))
    registry.register(Tool("computer_action_packet", "Prepare one read-only observe-act-verify action packet before desktop control approval.", RiskLevel.READ_ONLY, computer.computer_action_packet, "computer"))
    registry.register(Tool("approved_screen_observation_receipt", "Validate supplied approved screenshot-observation receipt metadata before OAV proof review.", RiskLevel.READ_ONLY, computer.approved_screen_observation_receipt, "computer"))
    registry.register(Tool("screen_observation_freshness_packet", "Check whether a supplied approved screenshot-observation receipt is fresh enough for one OAV primitive.", RiskLevel.READ_ONLY, computer.screen_observation_freshness_packet, "computer"))
    registry.register(Tool("screen_observation_confidence_packet", "Score supplied screen evidence before any computer-control primitive can move toward approval.", RiskLevel.READ_ONLY, computer.screen_observation_confidence_packet, "computer"))
    registry.register(Tool("screen_vision_prompt_preview", "Package approved screen-observation evidence for future vision-model verification without observing, reading images, or controlling the computer.", RiskLevel.READ_ONLY, computer.screen_vision_prompt_preview, "computer"))
    registry.register(Tool("screen_vision_model_review_preview", "Analyze one approved screenshot artifact with a configured local vision reviewer. Always requires approval.", RiskLevel.PERSONAL_DATA, computer.screen_vision_model_review_preview, "computer"))
    registry.register(Tool("screen_verification_contract", "Compare an expected screen state to supplied observation text without observing or controlling the computer.", RiskLevel.READ_ONLY, computer.screen_verification_contract, "computer"))
    registry.register(Tool("observe_act_verify_proof_packet", "Combine primitive action, fresh observation confidence, and screen verification proof before computer-control approval review.", RiskLevel.READ_ONLY, computer.observe_act_verify_proof_packet, "computer"))
    registry.register(Tool("observe_act_verify_route_lock", "Keep computer-control routing locked until OAV proof, after-action verification, approval packets, approval chain proof, and approved rerun evidence are present.", RiskLevel.READ_ONLY, computer.observe_act_verify_route_lock, "computer"))
    registry.register(Tool("observe_act_verify_approval_bridge", "Bind OAV proof, route lock, approval id, approved rerun, and verification receipt before final computer-control review.", RiskLevel.READ_ONLY, computer.observe_act_verify_approval_bridge, "computer"))
    registry.register(Tool("observe_act_verify_cockpit", "Consolidate OAV proof, route lock, approval bridge, receipt binding, and blockers before final computer-control review.", RiskLevel.READ_ONLY, computer.observe_act_verify_cockpit, "computer"))
    registry.register(Tool("observe_act_verify_final_review", "Check OAV cockpit, audit evidence, execution health, and operator review evidence before a human decision on one computer-control primitive.", RiskLevel.READ_ONLY, computer.observe_act_verify_final_review, "computer"))
    registry.register(Tool("observe_act_verify_action_audit", "Audit the final OAV packet before the operator decides on one exact computer-control primitive.", RiskLevel.READ_ONLY, computer.observe_act_verify_action_audit, "computer"))
    registry.register(Tool("observe_act_verify_execution_handoff", "Package the final one-primitive OAV approval handoff and post-run proof queue without executing computer control.", RiskLevel.READ_ONLY, computer.observe_act_verify_execution_handoff, "computer"))
    registry.register(Tool("observe_act_verify_post_run_closure", "Close the proof loop after one approved OAV primitive before any next computer-control primitive review.", RiskLevel.READ_ONLY, computer.observe_act_verify_post_run_closure, "computer"))
    registry.register(Tool("observe_act_verify_cycle_ledger", "Bind the full one-primitive observe-act-verify lifecycle into a read-only proof ledger before any next primitive review.", RiskLevel.READ_ONLY, computer.observe_act_verify_cycle_ledger, "computer"))
    registry.register(Tool("screen_size", "Read screen size.", RiskLevel.READ_ONLY, computer.screen_size, "computer"))
    registry.register(Tool("mouse_position", "Read mouse position.", RiskLevel.READ_ONLY, computer.mouse_position, "computer"))
    registry.register(Tool("move_mouse", "Move the mouse. Requires computer control enabled.", RiskLevel.HIGH_RISK, computer.move_mouse, "computer"))
    registry.register(Tool("click", "Click the mouse. Requires computer control enabled.", RiskLevel.HIGH_RISK, computer.click, "computer"))
    registry.register(Tool("type_text", "Type text. Requires computer control enabled.", RiskLevel.HIGH_RISK, computer.type_text, "computer"))
    registry.register(Tool("screenshot", "Take a screenshot.", RiskLevel.HIGH_RISK, computer.screenshot, "computer"))
    registry.register(Tool("observe_screen", "Take a screenshot and report screen/mouse state.", RiskLevel.HIGH_RISK, computer.observe_screen, "computer"))
    registry.register(Tool("verify_screen", "Observe screen and attach a verification expectation.", RiskLevel.HIGH_RISK, computer.verify_screen, "computer"))
    registry.register(Tool("observe_act_verify", "Run one observe-act-verify computer-control step.", RiskLevel.HIGH_RISK, computer.observe_act_verify, "computer"))
    fetch_page, extract_links, web_search, recent_browser_pages, summarize_page, save_page_note = browser.make_browser_tools(store, vault)
    registry.register(
        Tool(
            "fetch_page",
            "Fetch readable page text and record it in local browser history when storage is available.",
            RiskLevel.LOCAL_SAFE,
            fetch_page,
            "browser",
            argument_contract=_tool_argument_contract(
                required_strings=("url",),
                optional_integers=("max_chars",),
            ),
        )
    )
    registry.register(
        Tool(
            "extract_links",
            "Extract page links and record the page in local browser history when storage is available.",
            RiskLevel.LOCAL_SAFE,
            extract_links,
            "browser",
            argument_contract=_tool_argument_contract(required_strings=("url",)),
        )
    )
    registry.register(Tool("open_url", "Open a URL in the default browser.", RiskLevel.LOCAL_SAFE, browser.open_url, "browser"))
    registry.register(
        Tool(
            "web_search",
            "Search the web and record fetched result pages in local browser history when applicable.",
            RiskLevel.LOCAL_SAFE,
            web_search,
            "browser",
            argument_contract=_tool_argument_contract(
                required_strings=("query",),
                optional_strings=("engine",),
                optional_integers=("max_chars",),
            ),
        )
    )
    registry.register(
        Tool(
            "recent_browser_pages",
            "List recently fetched browser pages.",
            RiskLevel.READ_ONLY,
            recent_browser_pages,
            "browser",
            argument_contract=_tool_argument_contract(optional_integers=("limit",)),
        )
    )
    registry.register(
        Tool(
            "summarize_page",
            "Summarize a fetched browser page.",
            RiskLevel.READ_ONLY,
            summarize_page,
            "browser",
            argument_contract=_tool_argument_contract(
                optional_integers=("page_id",),
                optional_strings=("url",),
            ),
        )
    )
    registry.register(
        Tool(
            "save_page_note",
            "Save a fetched browser page summary into Obsidian Sources.",
            RiskLevel.LOCAL_SAFE,
            save_page_note,
            "browser",
            argument_contract=_tool_argument_contract(optional_integers=("page_id",)),
        )
    )

    (
        save_skill,
        list_skills,
        search_skills,
        get_skill,
        delete_skill,
        extract_linked_skills,
        install_linked_skills,
        skill_match_preview,
        resolve_delete_skill_approval,
    ) = make_skill_tools(store, vault)
    registry.register(
        Tool(
            "save_skill",
            "Save a reusable Markdown procedure skill.",
            RiskLevel.LOCAL_SAFE,
            save_skill,
            "skills",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
            ),
            _tool_argument_contract(
                required_strings=("name", "body"),
                optional_strings=("trigger", "tags"),
            ),
        )
    )
    registry.register(
        Tool(
            "list_skills",
            "List saved skills.",
            RiskLevel.READ_ONLY,
            list_skills,
            "skills",
            argument_contract=_tool_argument_contract(optional_integers=("limit",)),
        )
    )
    registry.register(
        Tool(
            "search_skills",
            "Search saved skills.",
            RiskLevel.READ_ONLY,
            search_skills,
            "skills",
            argument_contract=_tool_argument_contract(
                required_strings=("query",),
                optional_integers=("limit",),
            ),
        )
    )
    registry.register(
        Tool(
            "get_skill",
            "Read a saved skill.",
            RiskLevel.READ_ONLY,
            get_skill,
            "skills",
            argument_contract=_tool_argument_contract(required_strings=("name",)),
        )
    )
    registry.register(
        Tool(
            "delete_skill",
            "Delete a saved skill from SQLite and Obsidian.",
            RiskLevel.HIGH_RISK,
            delete_skill,
            "skills",
            argument_contract=_tool_argument_contract(required_strings=("name",)),
            approval_argument_resolver=resolve_delete_skill_approval,
            approval_argument_contract=_tool_argument_contract(
                required_strings=("name", "target_skill_name"),
                required_integers=("target_skill_id", "target_skill_revision"),
            ),
        )
    )
    registry.register(Tool("extract_linked_skills", "Extract reusable skill candidates from linked agent documents without saving them.", RiskLevel.READ_ONLY, extract_linked_skills, "skills"))
    registry.register(Tool("install_linked_skills", "Save link-inspired reusable skills into local Jarvis skill memory.", RiskLevel.LOCAL_SAFE, install_linked_skills, "skills"))
    registry.register(Tool("skill_match_preview", "Preview which saved skills match a request without executing anything.", RiskLevel.READ_ONLY, skill_match_preview, "skills"))

    (
        list_weak_memories,
        delete_memory,
        get_memory,
        edit_memory,
        list_duplicate_memories,
        merge_memories,
        memory_tree_summary,
        memory_stats,
        learning_review,
        save_learning_review,
        queue_learning_tasks,
        personal_context_status,
    ) = make_memory_curator_tools(store, vault)
    (
        resolve_delete_memory_approval,
        resolve_edit_memory_approval,
        resolve_merge_memories_approval,
    ) = make_memory_approval_resolvers(store)
    registry.register(
        Tool(
            "list_weak_memories",
            "List low-confidence or inference-like memories.",
            RiskLevel.READ_ONLY,
            list_weak_memories,
            "memory",
            argument_contract=_tool_argument_contract(
                optional_integer_ranges=(("limit", 1, 200),),
            ),
        )
    )
    registry.register(
        Tool(
            "delete_memory",
            "Delete only the exact reviewed memory version by id.",
            RiskLevel.HIGH_RISK,
            delete_memory,
            "memory",
            argument_contract=_tool_argument_contract(required_integers=("memory_id",)),
            approval_argument_resolver=resolve_delete_memory_approval,
            approval_argument_contract=_tool_argument_contract(
                required_strings=("target_binding",),
                required_integers=("memory_id", "target_revision"),
            ),
        )
    )
    registry.register(
        Tool(
            "get_memory",
            "Inspect one memory by id.",
            RiskLevel.READ_ONLY,
            get_memory,
            "memory",
            argument_contract=_tool_argument_contract(
                required_integers=("memory_id",),
            ),
        )
    )
    registry.register(
        Tool(
            "edit_memory",
            "Edit only the exact reviewed memory version by id.",
            RiskLevel.HIGH_RISK,
            edit_memory,
            "memory",
            argument_contract=_tool_argument_contract(
                optional_strings=("category", "title", "body"),
                required_integers=("memory_id",),
                optional_numbers=("confidence",),
            ),
            approval_argument_resolver=resolve_edit_memory_approval,
            approval_argument_contract=_tool_argument_contract(
                required_strings=("target_binding",),
                optional_strings=("category", "title", "body"),
                required_integers=("memory_id", "target_revision"),
                optional_numbers=("confidence",),
            ),
        )
    )
    registry.register(
        Tool(
            "list_duplicate_memories",
            "List duplicate-looking memories.",
            RiskLevel.READ_ONLY,
            list_duplicate_memories,
            "memory",
            argument_contract=_tool_argument_contract(
                optional_integer_ranges=(("limit", 1, 200),),
            ),
        )
    )
    registry.register(
        Tool(
            "merge_memories",
            "Merge the exact reviewed versions of two memories, keeping one id and deleting the other.",
            RiskLevel.HIGH_RISK,
            merge_memories,
            "memory",
            argument_contract=_tool_argument_contract(required_integers=("keep_id", "delete_id")),
            approval_argument_resolver=resolve_merge_memories_approval,
            approval_argument_contract=_tool_argument_contract(
                required_strings=("keep_binding", "delete_binding"),
                required_integers=(
                    "keep_id",
                    "keep_revision",
                    "delete_id",
                    "delete_revision",
                ),
            ),
        )
    )
    registry.register(
        Tool(
            "memory_tree_summary",
            "Write a summarized Memory Tree snapshot to Obsidian.",
            RiskLevel.LOCAL_SAFE,
            memory_tree_summary,
            "memory",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
            ),
            _tool_argument_contract(optional_integers=("limit",)),
        )
    )
    registry.register(
        Tool(
            "memory_stats",
            "Show memory counts by category and source.",
            RiskLevel.READ_ONLY,
            memory_stats,
            "memory",
            argument_contract=_tool_argument_contract(),
        )
    )
    registry.register(
        Tool(
            "personal_context_status",
            "Show content-free counts for Jarvis's durable personal context across memory, preferences, decisions, people, goals, tasks, and skills.",
            RiskLevel.READ_ONLY,
            personal_context_status,
            "memory",
            argument_contract=_tool_argument_contract(),
        )
    )
    registry.register(
        Tool(
            "learning_review",
            "Privately inspect revision-bound, category-label-only generic-memory candidates; labels are not semantic proof, and this review does not itself promote, queue, or duplicate structured records.",
            RiskLevel.READ_ONLY,
            learning_review,
            "learning",
            argument_contract=_tool_argument_contract(optional_integers=("limit",)),
        )
    )
    (
        knowledge_promotion_packet,
        promote_memory_to_decision,
        resolve_promote_memory_to_decision_approval,
        promote_memory_to_preference,
        resolve_promote_memory_to_preference_approval,
    ) = knowledge_promotion.make_knowledge_promotion_tools(store, vault)
    (
        promote_memory_to_profile,
        resolve_promote_memory_to_profile_approval,
    ) = knowledge_promotion.make_profile_promotion_tools(store, vault)
    registry.register(
        Tool(
            "knowledge_promotion_packet",
            "Privately review one exact generic-memory candidate and show the explicit, custody-aware promotion command without changing it.",
            RiskLevel.READ_ONLY,
            knowledge_promotion_packet,
            "learning",
            argument_contract=_tool_argument_contract(required_integers=("memory_id",)),
        )
    )
    registry.register(
        Tool(
            "promote_memory_to_decision",
            "Promote one exact reviewed generic memory into a decision while preserving its memory identity and requiring approval.",
            RiskLevel.HIGH_RISK,
            promote_memory_to_decision,
            "learning",
            argument_contract=_tool_argument_contract(
                required_strings=("review_token", "title", "rationale", "impact"),
                required_integers=("memory_id", "reviewed_revision"),
            ),
            approval_argument_resolver=resolve_promote_memory_to_decision_approval,
            approval_argument_contract=_tool_argument_contract(
                required_strings=(
                    "review_binding",
                    "title",
                    "rationale",
                    "impact",
                    "target_binding",
                ),
                required_integers=("memory_id", "reviewed_revision", "target_revision"),
            ),
        )
    )
    registry.register(
        Tool(
            "promote_memory_to_preference",
            "Promote one exact reviewed generic memory into a new preference while preserving its memory identity and requiring approval.",
            RiskLevel.HIGH_RISK,
            promote_memory_to_preference,
            "learning",
            argument_contract=_tool_argument_contract(
                required_strings=("review_token", "category", "key", "value"),
                required_integers=("memory_id", "reviewed_revision"),
            ),
            approval_argument_resolver=resolve_promote_memory_to_preference_approval,
            approval_argument_contract=_tool_argument_contract(
                required_strings=(
                    "review_binding",
                    "category",
                    "key",
                    "value",
                    "target_binding",
                ),
                required_integers=("memory_id", "reviewed_revision", "target_revision"),
            ),
        )
    )
    registry.register(
        Tool(
            "promote_memory_to_profile",
            "Promote one exact reviewed generic memory into an owned profile note while preserving its memory identity and requiring approval.",
            RiskLevel.HIGH_RISK,
            promote_memory_to_profile,
            "learning",
            argument_contract=_tool_argument_contract(
                required_strings=("review_token", "heading", "category", "body"),
                required_integers=("memory_id", "reviewed_revision"),
            ),
            approval_argument_resolver=resolve_promote_memory_to_profile_approval,
            approval_argument_contract=_tool_argument_contract(
                required_strings=(
                    "review_binding",
                    "heading",
                    "category",
                    "body",
                    "target_binding",
                ),
                required_integers=("memory_id", "reviewed_revision", "target_revision"),
            ),
        )
    )
    registry.register(Tool("save_learning_review", "Publish the private learning-review report to Obsidian Automations with a local publication lock; it does not promote or queue label-only candidates.", RiskLevel.LOCAL_SAFE, save_learning_review, "learning"))
    registry.register(
        Tool(
            "queue_learning_tasks",
            "Create local tasks from established feedback and health signals; label-only structuring candidates are excluded.",
            RiskLevel.LOCAL_SAFE,
            queue_learning_tasks,
            "learning",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset(
                    {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
                ),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=queue_learning_tasks_auto_mutation_operation_key,
            ),
            _tool_argument_contract(optional_integers=("limit",)),
        )
    )

    if config is not None:
        integration_status, integration_readiness_report, legacy_connector_migration_audit, integration_migration_plan, integration_boundary_contract, integration_action_preview, integration_scope_packet, integration_dry_run_contract, integration_runbook, integration_promotion_gate, integration_implementation_spec, integration_preflight_contract, integration_enablement_gate, integration_rehearsal_receipt, integration_metadata_preview, integration_proof_bundle, integration_implementation_review, integration_route_lock, integration_execution_matrix, integration_adapter_manifest, integration_adapter_probe, integration_adapter_acceptance, open_jarvis_vault, create_reminder = make_personal_tools(config)
        registry.register(Tool("integration_status", "Show personal integration status.", RiskLevel.READ_ONLY, integration_status, "personal"))
        registry.register(Tool("integration_readiness_report", "Rank personal connector migration readiness without account access, personal-data reads, or side effects.", RiskLevel.READ_ONLY, integration_readiness_report, "personal"))
        registry.register(Tool("legacy_connector_migration_audit", "Audit old Jarvis browser/calendar/email migration readiness without account access, personal-data reads, side effects, route unlocks, or approval grants.", RiskLevel.READ_ONLY, legacy_connector_migration_audit, "personal"))
        registry.register(Tool("integration_migration_plan", "Plan a safe, approval-gated migration for a personal connector.", RiskLevel.READ_ONLY, integration_migration_plan, "personal"))
        registry.register(Tool("integration_boundary_contract", "Show the read-only, personal-data, side-effect, caching, and approval contract for a connector.", RiskLevel.READ_ONLY, integration_boundary_contract, "personal"))
        registry.register(Tool("integration_action_preview", "Classify a future personal connector action by risk and approval boundary without connecting accounts.", RiskLevel.READ_ONLY, integration_action_preview, "personal"))
        registry.register(Tool("integration_scope_packet", "Prepare a scoped future connector request packet without connecting accounts or reading personal data.", RiskLevel.READ_ONLY, integration_scope_packet, "personal"))
        registry.register(Tool("integration_dry_run_contract", "Prepare a dry-run connector harness receipt with exact scope, metadata row contract proof, approval, and verification gates.", RiskLevel.READ_ONLY, integration_dry_run_contract, "personal"))
        registry.register(Tool("integration_runbook", "Prepare a connector runbook with preflight, metadata row contract, approval, execution, verification, rollback, and audit steps.", RiskLevel.READ_ONLY, integration_runbook, "personal"))
        registry.register(Tool("integration_promotion_gate", "Decide whether a future personal connector can move from runbook to implementation spec, read-only prototype, or blocked state with metadata row contract proof.", RiskLevel.READ_ONLY, integration_promotion_gate, "personal"))
        registry.register(Tool("integration_implementation_spec", "Draft the exact ToolRegistry, planner, API, test, metadata row contract, audit, verification, and approval contract for a future personal connector.", RiskLevel.READ_ONLY, integration_implementation_spec, "personal"))
        registry.register(Tool("integration_preflight_contract", "Draft the final read-only connector preflight contract with exact future args, metadata row contract proof, enablement gates, proof, and stop conditions.", RiskLevel.READ_ONLY, integration_preflight_contract, "personal"))
        registry.register(Tool("integration_enablement_gate", "Check the final read-only connector enablement gate with tests, audit, rollback, approval, and acceptance evidence.", RiskLevel.READ_ONLY, integration_enablement_gate, "personal"))
        registry.register(Tool("integration_rehearsal_receipt", "Rehearse a future personal connector execution envelope with disabled adapter, approval checkpoint, audit, rollback, verification, and acceptance proof.", RiskLevel.READ_ONLY, integration_rehearsal_receipt, "personal"))
        registry.register(Tool("integration_metadata_preview", "Preview a future connector metadata adapter with fake bounded rows, disabled adapter boundary, audit, tests, acceptance evidence, and blocked private/full-content paths.", RiskLevel.READ_ONLY, integration_metadata_preview, "personal"))
        registry.register(Tool("integration_proof_bundle", "Bundle metadata preview, disabled adapter acceptance, metadata row contract, enablement, rehearsal, audit, verification, rollback, and stop proof before connector implementation review.", RiskLevel.READ_ONLY, integration_proof_bundle, "personal"))
        registry.register(Tool("integration_implementation_review", "Gate whether a future personal connector adapter is ready for code review across proof bundle, metadata row contract, preflight row-contract proof, spec, status, smoke, audit, verification, rollback, and stop evidence.", RiskLevel.READ_ONLY, integration_implementation_review, "personal"))
        registry.register(Tool("integration_route_lock", "Keep personal connector routing disabled until implementation review, status/API smoke evidence, metadata row proof, scope hash, and explicit route review are present.", RiskLevel.READ_ONLY, integration_route_lock, "personal"))
        registry.register(Tool("integration_execution_matrix", "Show connector execution lanes, disabled adapter states, approval boundaries, proof gates, and safest next harness commands.", RiskLevel.READ_ONLY, integration_execution_matrix, "personal"))
        registry.register(Tool("integration_adapter_manifest", "Define disabled connector adapter stubs, fake fixtures, blocked-path tests, audit fields, and enablement gates without account access.", RiskLevel.READ_ONLY, integration_adapter_manifest, "personal"))
        registry.register(Tool("integration_adapter_probe", "Exercise a disabled connector adapter boundary with fake bounded metadata rows and blocked full-content/side-effect paths.", RiskLevel.READ_ONLY, integration_adapter_probe, "personal"))
        registry.register(Tool("integration_adapter_acceptance", "Run disabled connector adapter acceptance cases for metadata-only happy path and blocked full-content, side-effect, and missing-scope paths.", RiskLevel.READ_ONLY, integration_adapter_acceptance, "personal"))
        registry.register(
            Tool(
                "open_jarvis_vault",
                "Open the Jarvis Obsidian folder.",
                RiskLevel.HIGH_RISK,
                open_jarvis_vault,
                "personal",
                argument_contract=_tool_argument_contract(),
            )
        )
        registry.register(Tool("create_reminder", "Create a macOS Reminder.", RiskLevel.HIGH_RISK, create_reminder, "personal"))

    for tool in make_calendar_tools(config):
        registry.register(tool)
    for tool in make_email_tools(config):
        registry.register(tool)
    for tool in make_imessage_tools(config):
        registry.register(tool)
    for tool in make_weather_tools(config):
        registry.register(tool)
    for tool in make_reminder_tools(config):
        registry.register(tool)
    for tool in make_news_tools(config):
        registry.register(tool)
    for tool in make_brief_tools(config):
        registry.register(tool)
    for tool in make_currency_tools(config):
        registry.register(tool)
    for tool in make_translate_tools(config):
        registry.register(tool)
    for tool in make_markets_tools(config):
        registry.register(tool)
    for tool in make_research_tools(config):
        registry.register(tool)
    for tool in make_dictionary_tools(config):
        registry.register(tool)
    for tool in make_wikipedia_tools(config):
        registry.register(tool)
    for tool in make_sun_tools(config):
        registry.register(tool)
    for tool in make_history_tools(config):
        registry.register(tool)
    for tool in make_air_tools(config):
        registry.register(tool)
    for tool in make_holidays_tools(config):
        registry.register(tool)
    for tool in make_fun_tools(config):
        registry.register(tool)
    for tool in make_countdown_tools(config):
        registry.register(tool)
    for tool in make_writer_tools(config):
        registry.register(tool)
    for tool in make_compose_tools(config):
        registry.register(tool)
    for tool in make_kakao_tools(config):
        registry.register(tool)
    for tool in make_instagram_tools(config):
        registry.register(tool)
    for tool in make_contacts_tools(config):
        registry.register(tool)
    for tool in make_call_tools(config):
        registry.register(tool)
    for tool in make_ocr_tools(config):
        registry.register(tool)
    for tool in make_apple_reminders_tools(config):
        registry.register(tool)

    daily_brief, daily_plan, goal_nudge, weekly_review, weekly_review_context, weekly_review_prompt_preview, morning_startup, save_morning_startup = make_proactive_tools(store, vault)
    registry.register(
        Tool(
            "daily_brief",
            "Generate a proactive Jarvis daily brief and write it to Obsidian.",
            RiskLevel.LOCAL_SAFE,
            daily_brief,
            "proactive",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=daily_brief_auto_mutation_operation_key,
            ),
            _tool_argument_contract(),
        )
    )
    registry.register(
        Tool(
            "daily_plan",
            "Generate a tactical daily plan from tasks, goals, approvals, and jobs.",
            RiskLevel.LOCAL_SAFE,
            daily_plan,
            "proactive",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=daily_plan_auto_mutation_operation_key,
                semantic_preflight=daily_plan_auto_mutation_preflight,
            ),
            _tool_argument_contract(required_strings=("target_date",)),
        )
    )
    registry.register(
        Tool(
            "goal_nudge",
            "Generate stale-goal nudges and write them to Obsidian.",
            RiskLevel.LOCAL_SAFE,
            goal_nudge,
            "proactive",
            AutoMutationContract(
                version=AUTO_MUTATION_CONTRACT_VERSION,
                effects=frozenset({AutoMutationEffect.OBSIDIAN_VAULT}),
                replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder=goal_nudge_auto_mutation_operation_key,
            ),
            _tool_argument_contract(
                optional_integer_ranges=(("stale_days", 1, 365),),
            ),
        )
    )
    registry.register(Tool("weekly_review", "Generate a weekly Jarvis review and write it to Obsidian.", RiskLevel.LOCAL_SAFE, weekly_review, "proactive"))
    registry.register(Tool("weekly_review_context", "Build a read-only model-assist packet for weekly review without executing actions.", RiskLevel.READ_ONLY, weekly_review_context, "proactive"))
    registry.register(Tool("weekly_review_prompt_preview", "Preview a bounded weekly-review model prompt without calling a model or granting execution authority.", RiskLevel.READ_ONLY, weekly_review_prompt_preview, "proactive"))
    registry.register(Tool("morning_startup", "Show a read-only safe startup brief for the day.", RiskLevel.READ_ONLY, morning_startup, "proactive"))
    registry.register(Tool("save_morning_startup", "Write the current morning startup brief to today's Obsidian daily note.", RiskLevel.LOCAL_SAFE, save_morning_startup, "proactive"))
    (
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
    ) = make_scheduler_tools(store, vault, config)
    registry.register(Tool("scheduler_context_refresh_packet", "Check scheduled continuity readiness for Current Context and Mission Control without creating or running jobs.", RiskLevel.READ_ONLY, scheduler_context_refresh_packet, "scheduler"))
    registry.register(Tool("schedule_daily_brief", "Schedule a recurring daily brief job.", RiskLevel.LOCAL_SAFE, schedule_daily_brief, "scheduler"))
    registry.register(
        Tool(
            "schedule_morning_brief",
            "Schedule a daily Telegram morning brief to the allowlisted owner.",
            RiskLevel.EXTERNAL_SIDE_EFFECT,
            schedule_morning_brief,
            "scheduler",
            argument_contract=_tool_argument_contract(optional_strings=("time", "hhmm", "at")),
        )
    )
    registry.register(Tool("schedule_goal_nudge", "Schedule recurring stale-goal nudges.", RiskLevel.LOCAL_SAFE, schedule_goal_nudge, "scheduler"))
    registry.register(Tool("schedule_weekly_review", "Schedule a recurring weekly review.", RiskLevel.LOCAL_SAFE, schedule_weekly_review, "scheduler"))
    registry.register(Tool("schedule_inbox_ingest", "Schedule recurring Obsidian Inbox ingestion.", RiskLevel.LOCAL_SAFE, schedule_inbox_ingest, "scheduler"))
    registry.register(Tool("schedule_file_digest", "Schedule recurring watched-file metadata digests.", RiskLevel.LOCAL_SAFE, schedule_file_digest, "scheduler"))
    registry.register(
        Tool(
            "schedule_assistant_basics",
            "Create missing default Jarvis background jobs without changing existing schedules.",
            RiskLevel.LOCAL_SAFE,
            schedule_assistant_basics,
            "scheduler",
            argument_contract=_tool_argument_contract(),
        )
    )
    registry.register(
        Tool(
            "list_scheduled_jobs",
            "List scheduled Jarvis jobs.",
            RiskLevel.READ_ONLY,
            list_scheduled_jobs,
            "scheduler",
            argument_contract=_tool_argument_contract(),
        )
    )
    registry.register(
        Tool(
            "run_due_jobs",
            "Run scheduled jobs that are currently due; some jobs may deliver externally.",
            RiskLevel.EXTERNAL_SIDE_EFFECT,
            run_due_jobs,
            "scheduler",
            argument_contract=_tool_argument_contract(),
        )
    )
    registry.register(Tool("pause_job", "Pause a scheduled job by name.", RiskLevel.LOCAL_SAFE, pause_job, "scheduler"))
    registry.register(
        Tool(
            "resume_job",
            "Resume a scheduled job by name; the resumed job may deliver externally when due.",
            RiskLevel.EXTERNAL_SIDE_EFFECT,
            resume_job,
            "scheduler",
            argument_contract=_tool_argument_contract(required_strings=("name",)),
        )
    )
    registry.register(Tool("delete_job", "Delete a scheduled job by name.", RiskLevel.HIGH_RISK, delete_job, "scheduler"))
    registry.register(
        Tool(
            "run_job_now",
            "Run one scheduled job immediately by name; the selected job may deliver externally.",
            RiskLevel.EXTERNAL_SIDE_EFFECT,
            run_job_now,
            "scheduler",
            argument_contract=_tool_argument_contract(required_strings=("name",)),
        )
    )
    return registry
