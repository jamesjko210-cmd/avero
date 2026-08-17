from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable

from jarvis_v2.agent.chat import SYSTEM_PROMPT, ChatBrain
from jarvis_v2.agent.types import ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.skill_projection import reconcile_skill_projection
from jarvis_v2.memory.store import (
    MAX_SELF_KNOWLEDGE_DECISIONS,
    MAX_SELF_KNOWLEDGE_GOALS,
    MAX_SELF_KNOWLEDGE_PEOPLE,
    MAX_SELF_KNOWLEDGE_STEPS_PER_GOAL,
    MAX_SELF_KNOWLEDGE_TASKS,
    MemoryStore,
    SkillRecord,
)


MAX_CONVERSATION_LIMIT = 200
MAX_CHAT_CONTEXT_LIMIT = 20
MAX_CHAT_CONTEXT_OUTPUT_CHARS = 12_000
MAX_CHAT_PROMPT_PREVIEW_OUTPUT_CHARS = 20_000
MAX_SELF_KNOWLEDGE_PREVIEW_CHARS = 8_000
SELF_KNOWLEDGE_SECTION_CHAR_LIMITS = {
    "goals": 2_700,
    "tasks": 1_400,
    "decisions": 1_700,
    "people": 1_700,
}
MAX_CONVERSATION_TEXT_CHARS = 600
MAX_CONVERSATION_LINE_CHARS = 260
MAX_SESSION_ID_CHARS = 80
MAX_SKILL_FIELD_CHARS = 120
MIN_WS4_CHAT_MEASUREMENT_TURNS = 20
CHAT_LATENCY_METADATA_KEYS = ("latency_ms", "duration_ms", "response_latency_ms", "elapsed_ms")
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
HOME_PATH_RE = re.compile(r"(?<![\w.])~/(?:[^\s;]+)")
_MISSING = object()


def _bounded_limit(value: Any, default: int, maximum: int = MAX_CONVERSATION_LIMIT) -> int:
    if isinstance(value, bool):
        return default
    try:
        return max(1, min(maximum, int(value)))
    except (TypeError, ValueError):
        return default


def _short(value: Any, *, limit: int = MAX_CONVERSATION_LINE_CHARS) -> str:
    if value is None:
        text = ""
    else:
        try:
            text = str(value)
        except Exception:
            text = "<unreadable>"
    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        " " if unicodedata.category(char) in {"Cc", "Cf"} else char
        for char in text
    )
    text = text.strip()
    text = " ".join(text.split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    text = HOME_PATH_RE.sub("<local-path>", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _chat_latency_ms(response: dict[str, Any]) -> float | None:
    for key in CHAT_LATENCY_METADATA_KEYS:
        value = _safe_float(response.get(key))
        if value is not None and value >= 0:
            return round(value, 1)
    return None


def _model_usage_count(response: dict[str, Any], key: str) -> int | None:
    if response.get("model_usage_available") is not True:
        return None
    value = response.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0 or value > 1_000_000_000:
        return None
    return value


def _percentile_ms(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 1)
    rank = (len(ordered) - 1) * percentile
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return round(ordered[lower], 1)
    weight = rank - lower
    return round(ordered[lower] + ((ordered[upper] - ordered[lower]) * weight), 1)


def _percent(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((count / total) * 100, 1)


def _row_value(row: Any, key: str, default: str = "") -> str:
    try:
        if isinstance(row, dict):
            value = row.get(key, default)
        else:
            value = row[key]
    except Exception:
        return default
    text = _short(value)
    return text if text else default


def _positive_row_id(row: Any, key: str = "id") -> int | None:
    try:
        value = row.get(key) if isinstance(row, dict) else row[key]
    except Exception:
        return None
    if type(value) is not int or value < 1:
        return None
    return value


def _raw_row_value(row: Any, key: str) -> Any:
    try:
        return row.get(key, _MISSING) if isinstance(row, dict) else row[key]
    except Exception:
        return _MISSING


def _has_exact_text_fields(row: Any, *keys: str) -> bool:
    return all(type(_raw_row_value(row, key)) is str for key in keys)


def _self_row_text(row: Any, key: str, default: str = "", *, limit: int = MAX_CONVERSATION_LINE_CHARS) -> str:
    try:
        value = row.get(key) if isinstance(row, dict) else row[key]
    except Exception:
        return default
    if type(value) is not str:
        return default
    normalized = unicodedata.normalize("NFKC", value)
    normalized = "".join(
        " " if unicodedata.category(char) in {"Cc", "Cf"} else char
        for char in normalized
    )
    text = _short(normalized, limit=limit)
    return text if text else default


def _bounded_snapshot_rows(value: Any, maximum: int) -> tuple[list[Any], bool]:
    if isinstance(value, (str, bytes, bytearray, dict)) or value is None:
        return [], True
    rows: list[Any] = []
    degraded = False
    try:
        iterator = iter(value)
    except Exception:
        return [], True
    try:
        for row in iterator:
            if len(rows) >= maximum:
                degraded = True
                break
            rows.append(row)
    except Exception:
        degraded = True
    return rows, degraded


def _bounded_preview_section(lines: list[str], maximum: int) -> tuple[list[str], bool]:
    if not lines:
        return [], False
    marker = "- More entries were omitted by this section's fixed display limit."
    kept: list[str] = []
    used = 0
    for line in lines:
        added = len(line) + (1 if kept else 0)
        if used + added + len(marker) + 1 > maximum:
            kept.append(marker)
            return kept, True
        kept.append(line)
        used += added
    return kept, False


def _render_with_preserved_footer(
    lines: list[str],
    footer_lines: list[str],
    maximum: int,
) -> tuple[str, bool]:
    main = "\n".join(lines).rstrip()
    footer = "\n".join(footer_lines).strip()
    separator = "\n\n" if main and footer else ""
    full = main + separator + footer
    if len(full) <= maximum:
        return full, False
    marker = "\n- Earlier preview sections were truncated at the fixed display limit."
    main_budget = max(0, maximum - len(separator) - len(footer) - len(marker))
    bounded_main = main[:main_budget].rstrip() + marker
    return bounded_main + separator + footer, True


def _self_knowledge_preview(snapshot: Any | None) -> tuple[list[str], dict[str, Any]]:
    section_names = ("goals", "goal_steps", "tasks", "decisions", "people")
    if snapshot is None:
        states = {name: "unavailable" for name in section_names}
        return (
            [
                "Local personal state:",
                "- Jarvis could not fully check structured goals, commitments, decisions, or relationships. Their contents are not assumed empty.",
            ],
            {
                "self_knowledge_state": "unavailable",
                "self_knowledge_source_states": states,
                "self_knowledge_counts": {name: 0 for name in section_names},
                "self_knowledge_unreadable_rows": 0,
                "self_knowledge_truncated_sources": [],
                "self_knowledge_clipped_sources": [],
                "self_knowledge_contract_degraded": False,
                "self_knowledge_display_truncated_sources": [],
                "self_knowledge_output_truncated": False,
            },
        )

    limits = {
        "goals": MAX_SELF_KNOWLEDGE_GOALS,
        "goal_steps": MAX_SELF_KNOWLEDGE_GOALS * MAX_SELF_KNOWLEDGE_STEPS_PER_GOAL,
        "tasks": MAX_SELF_KNOWLEDGE_TASKS,
        "decisions": MAX_SELF_KNOWLEDGE_DECISIONS,
        "people": MAX_SELF_KNOWLEDGE_PEOPLE,
    }
    raw_rows: dict[str, list[Any]] = {}
    degraded: dict[str, bool] = {}
    for name, maximum in limits.items():
        try:
            value = getattr(snapshot, name)
        except Exception:
            value = None
        raw_rows[name], degraded[name] = _bounded_snapshot_rows(value, maximum)

    def source_name_set(attribute: str) -> tuple[set[str], bool]:
        try:
            raw_names = getattr(snapshot, attribute, ())
        except Exception:
            raw_names = None
        names, contract_degraded = _bounded_snapshot_rows(raw_names, len(section_names))
        valid_names = {name for name in names if type(name) is str and name in section_names}
        return valid_names, contract_degraded or len(valid_names) != len(names)

    truncated_sources, truncated_contract_degraded = source_name_set("truncated_sources")
    clipped_sources, clipped_contract_degraded = source_name_set("clipped_sources")
    snapshot_contract_degraded = truncated_contract_degraded or clipped_contract_degraded

    valid: dict[str, list[Any]] = {name: [] for name in section_names}
    for row in raw_rows["goals"]:
        if (
            _positive_row_id(row) is None
            or not _has_exact_text_fields(row, "title", "purpose", "horizon", "status", "updated_at")
            or _self_row_text(row, "status").lower() != "active"
            or not _self_row_text(row, "title")
        ):
            degraded["goals"] = True
            continue
        valid["goals"].append(row)
    for row in raw_rows["goal_steps"]:
        if (
            _positive_row_id(row) is None
            or _positive_row_id(row, "goal_id") is None
            or not _has_exact_text_fields(row, "body", "status", "updated_at")
            or _self_row_text(row, "status").lower() != "open"
            or not _self_row_text(row, "body")
        ):
            degraded["goal_steps"] = True
            continue
        valid["goal_steps"].append(row)
    for row in raw_rows["tasks"]:
        if (
            _positive_row_id(row) is None
            or not _has_exact_text_fields(row, "body", "due", "priority", "status", "updated_at")
            or _self_row_text(row, "priority").lower() not in {"high", "normal", "low"}
            or _self_row_text(row, "status").lower() != "open"
            or not _self_row_text(row, "body")
        ):
            degraded["tasks"] = True
            continue
        valid["tasks"].append(row)
    for row in raw_rows["decisions"]:
        if (
            _positive_row_id(row) is None
            or _positive_row_id(row, "revision") is None
            or not _has_exact_text_fields(row, "title", "rationale", "impact", "status", "updated_at")
            or _self_row_text(row, "status").lower() != "active"
            or not _self_row_text(row, "title")
        ):
            degraded["decisions"] = True
            continue
        valid["decisions"].append(row)
    for row in raw_rows["people"]:
        revision = _raw_row_value(row, "revision")
        last_contact = _raw_row_value(row, "last_contact_at")
        if (
            _positive_row_id(row) is None
            or type(revision) is not int
            or revision < 0
            or (last_contact is not None and type(last_contact) is not str)
            or not _has_exact_text_fields(row, "name", "relation", "updated_at")
            or not _self_row_text(row, "name")
        ):
            degraded["people"] = True
            continue
        valid["people"].append(row)

    for name in ("goals", "tasks", "decisions", "people"):
        unique_rows: list[Any] = []
        seen_ids: set[int] = set()
        for row in valid[name]:
            row_id = _positive_row_id(row)
            if row_id is None or row_id in seen_ids:
                degraded[name] = True
                continue
            seen_ids.add(row_id)
            unique_rows.append(row)
        valid[name] = unique_rows

    valid_goal_ids = {_positive_row_id(row) for row in valid["goals"]}
    bounded_steps: list[Any] = []
    step_counts: dict[int, int] = {}
    seen_step_ids: set[int] = set()
    for row in valid["goal_steps"]:
        step_id = _positive_row_id(row)
        goal_id = _positive_row_id(row, "goal_id")
        if step_id is None or goal_id not in valid_goal_ids or step_id in seen_step_ids:
            degraded["goal_steps"] = True
            continue
        seen_step_ids.add(step_id)
        count = step_counts.get(goal_id, 0)
        if count >= MAX_SELF_KNOWLEDGE_STEPS_PER_GOAL:
            truncated_sources.add("goal_steps")
            continue
        step_counts[goal_id] = count + 1
        bounded_steps.append(row)
    valid["goal_steps"] = bounded_steps

    states = {}
    for name in section_names:
        if degraded[name]:
            states[name] = "degraded"
        elif name in truncated_sources or name in clipped_sources:
            states[name] = "partial"
        else:
            states[name] = "ok" if valid[name] else "empty"
    unreadable = sum(len(raw_rows[name]) - len(valid[name]) for name in section_names)
    if snapshot_contract_degraded or any(state == "degraded" for state in states.values()):
        overall_state = "degraded"
    elif any(state == "partial" for state in states.values()):
        overall_state = "partial"
    elif any(valid.values()):
        overall_state = "ok"
    else:
        overall_state = "empty"

    steps_by_goal: dict[int, list[Any]] = {}
    for row in valid["goal_steps"]:
        goal_id = _positive_row_id(row, "goal_id")
        if goal_id is not None:
            steps_by_goal.setdefault(goal_id, []).append(row)

    lines = [
        "Local personal state:",
        "- Read from one bounded local database snapshot. This preview does not send it to a model.",
    ]
    goal_lines = ["Current goals and next steps:"]
    if valid["goals"]:
        for row in valid["goals"]:
            goal_id = _positive_row_id(row)
            purpose = _self_row_text(row, "purpose", limit=140)
            horizon = _self_row_text(row, "horizon", limit=60)
            detail = ""
            if purpose:
                detail += f" — {_short(purpose, limit=140)}"
            if horizon:
                detail += f" (horizon: {_short(horizon, limit=60)})"
            goal_lines.append(f"- Goal #{goal_id}: {_self_row_text(row, 'title')}{detail}")
            for step in steps_by_goal.get(goal_id or 0, []):
                goal_lines.append(f"  - Next: {_self_row_text(step, 'body', limit=160)}")
    else:
        goal_lines.append("- No active goals recorded." if states["goals"] == "empty" else "- Goal rows were not fully readable.")

    task_lines = ["Open commitments:"]
    if valid["tasks"]:
        for row in valid["tasks"]:
            due = _self_row_text(row, "due", limit=60)
            priority = _self_row_text(row, "priority", "normal", limit=24)
            detail = f" [{_short(priority, limit=24)}]"
            if due:
                detail += f" due {_short(due, limit=60)}"
            task_lines.append(f"- Task #{_positive_row_id(row)}{detail}: {_self_row_text(row, 'body', limit=180)}")
    else:
        task_lines.append("- No open commitments recorded." if states["tasks"] == "empty" else "- Commitment rows were not fully readable.")

    decision_lines = ["Active decisions:"]
    if valid["decisions"]:
        for row in valid["decisions"]:
            rationale = _self_row_text(row, "rationale", limit=120)
            impact = _self_row_text(row, "impact", limit=120)
            detail_parts = []
            if rationale:
                detail_parts.append(f"why: {_short(rationale, limit=120)}")
            if impact:
                detail_parts.append(f"impact: {_short(impact, limit=120)}")
            detail = f" — {'; '.join(detail_parts)}" if detail_parts else ""
            decision_lines.append(f"- Decision #{_positive_row_id(row)}: {_self_row_text(row, 'title')}{detail}")
    else:
        decision_lines.append("- No active decisions recorded." if states["decisions"] == "empty" else "- Decision rows were not fully readable.")

    people_lines = ["Relationships on record:"]
    if valid["people"]:
        for row in valid["people"]:
            relation = _self_row_text(row, "relation", "relationship not labeled", limit=100)
            last_contact = _self_row_text(row, "last_contact_at", limit=60)
            detail = f" — {_short(relation, limit=100)}"
            if last_contact:
                detail += f"; last contact recorded {_short(last_contact, limit=60)}"
            people_lines.append(f"- {_self_row_text(row, 'name')}{detail}")
    else:
        people_lines.append("- No relationships recorded." if states["people"] == "empty" else "- Relationship rows were not fully readable.")

    display_truncated_sources: list[str] = []
    for source, section_lines in (
        ("goals", goal_lines),
        ("tasks", task_lines),
        ("decisions", decision_lines),
        ("people", people_lines),
    ):
        bounded_lines, was_truncated = _bounded_preview_section(
            section_lines,
            SELF_KNOWLEDGE_SECTION_CHAR_LIMITS[source],
        )
        lines.extend(["", *bounded_lines])
        if was_truncated:
            display_truncated_sources.append(source)

    if truncated_sources:
        lines.extend(["", "- Additional rows exist and were omitted by the fixed personal-context limits."])
    if clipped_sources:
        lines.extend(["", "- Oversized stored fields were clipped at the database read boundary."])
    if unreadable or any(degraded.values()):
        lines.extend(["", "- Some stored rows were hidden because they were malformed or exceeded the preview limits."])
    rendered = "\n".join(lines)
    global_output_truncated = len(rendered) > MAX_SELF_KNOWLEDGE_PREVIEW_CHARS
    if global_output_truncated:
        rendered = (
            rendered[: MAX_SELF_KNOWLEDGE_PREVIEW_CHARS - 70].rstrip()
            + "\n- Personal-state preview truncated at its fixed local display limit."
        )
    return rendered.splitlines(), {
        "self_knowledge_state": overall_state,
        "self_knowledge_source_states": states,
        "self_knowledge_counts": {name: len(valid[name]) for name in section_names},
        "self_knowledge_unreadable_rows": unreadable,
        "self_knowledge_truncated_sources": sorted(truncated_sources),
        "self_knowledge_clipped_sources": sorted(clipped_sources),
        "self_knowledge_contract_degraded": snapshot_contract_degraded,
        "self_knowledge_display_truncated_sources": display_truncated_sources,
        "self_knowledge_output_truncated": bool(display_truncated_sources) or global_output_truncated,
    }


def _safe_session_payloads(rows: list[Any]) -> tuple[list[dict[str, Any]], int]:
    payloads: list[dict[str, Any]] = []
    unreadable = 0
    for row in rows:
        try:
            session_id = _short(row["session_id"], limit=MAX_SESSION_ID_CHARS)
            messages = _safe_int(row["messages"])
            started_at = _short(row["started_at"], limit=80)
            last_at = _short(row["last_at"], limit=80)
        except Exception:
            unreadable += 1
            continue
        if not session_id or "<unreadable>" in {session_id, started_at, last_at}:
            unreadable += 1
            continue
        payloads.append(
            {
                "session_id": session_id,
                "messages": messages,
                "started_at": started_at,
                "last_at": last_at,
            }
        )
    return payloads, unreadable


def _safe_message_payloads(rows: list[Any]) -> tuple[list[dict[str, Any]], int]:
    payloads: list[dict[str, Any]] = []
    unreadable = 0
    for row in rows:
        try:
            message_id = int(row["id"])
            session_id = str(row["session_id"] or "")
            role = str(row["role"] or "")
            content = str(row["content"] or "")
            created_at = row["created_at"]
            metadata_raw = row["metadata"] or "{}"
        except Exception:
            unreadable += 1
            continue
        try:
            metadata = json.loads(metadata_raw)
        except Exception:
            metadata = {}
        payloads.append(
            {
                "id": message_id,
                "session_id": session_id,
                "role": role,
                "content": content,
                "created_at": created_at,
                "metadata": metadata if isinstance(metadata, dict) else {},
            }
        )
    return payloads, unreadable


def _safe_vault_path_display(path: Path | str | None, vault: ObsidianVault) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return str(candidate.relative_to(vault.root_path))
    except ValueError:
        return _short(candidate, limit=120)


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "calls_model": False,
        "calls_chat_model": False,
        "executes_tools": False,
        "queues_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "approves_request": False,
        "dismisses_request": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "executes_side_effect": False,
        "writes_files": False,
        "writes_memory": False,
        "writes_notes": False,
        "controls_computer": False,
    }
    metadata.update(extra)
    return metadata


def _read_only_metadata(**extra: Any) -> dict[str, Any]:
    return _safe_metadata(**extra)


def _write_note_metadata(**extra: Any) -> dict[str, Any]:
    metadata = _safe_metadata(**extra)
    metadata.update({"writes_files": True, "writes_memory": True, "writes_notes": True})
    return metadata


def _conversation_contract(
    *,
    state_changed: bool = False,
    changed: list[str] | None = None,
    content_in_handoff: bool = False,
) -> dict[str, Any]:
    return {
        "ready_for_operator": True,
        "state_changed": state_changed,
        "changed": changed or [],
        "content_in_handoff": content_in_handoff,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _conversation_handoff_metadata(
    handoff_key: str,
    handoff: dict[str, Any],
    *,
    writes: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    _normalize_conversation_handoff(handoff)
    prefix = handoff_key.removesuffix("_handoff")
    metadata = _write_note_metadata(**extra) if writes else _read_only_metadata(**extra)
    metadata.update(
        _conversation_contract(
            state_changed=_metadata_bool(handoff.get("state_changed")),
            changed=list(handoff.get("changed") or []),
            content_in_handoff=_metadata_bool(handoff.get("content_in_handoff")),
        )
    )
    metadata[f"{handoff_key}_ready"] = True
    metadata[f"{prefix}_handoff_ready"] = True
    metadata[f"{prefix}_ready_for_operator"] = True
    metadata[f"{prefix}_state_changed"] = _metadata_bool(handoff.get("state_changed"))
    metadata[f"{prefix}_changed"] = list(handoff.get("changed") or [])
    metadata[f"{prefix}_content_in_handoff"] = _metadata_bool(handoff.get("content_in_handoff"))
    metadata[f"{prefix}_next_safe_command"] = handoff["next_safe_command"]
    metadata[f"{prefix}_next_safe_commands"] = list(handoff["next_safe_commands"])
    metadata[f"{prefix}_next_safe_command_count"] = handoff["next_safe_command_count"]
    metadata[f"{prefix}_authorizes_execution"] = False
    metadata[f"{prefix}_authorizes_completion_claim"] = False
    metadata[f"{prefix}_approval_granted"] = False
    metadata["next_safe_command"] = handoff["next_safe_command"]
    metadata["next_safe_commands"] = list(handoff["next_safe_commands"])
    metadata["next_safe_command_count"] = handoff["next_safe_command_count"]
    metadata[handoff_key] = handoff
    return metadata


def _normalize_conversation_handoff(handoff: dict[str, Any]) -> dict[str, Any]:
    raw_next = handoff.get("next_commands") or []
    if isinstance(raw_next, dict):
        next_safe_commands = [str(value) for value in raw_next.values() if str(value or "").strip()]
    else:
        next_safe_commands = [str(value) for value in raw_next if str(value or "").strip()]
    next_safe_command = next_safe_commands[0] if next_safe_commands else ""
    handoff["handoff_ready"] = True
    handoff["next_safe_command"] = next_safe_command
    handoff["next_safe_commands"] = list(next_safe_commands)
    handoff["next_safe_command_count"] = len(next_safe_commands)
    return handoff


def _conversation_handoff(
    *,
    source: str,
    session_id: str,
    count: int = 0,
    query: str = "",
    path_display: str = "",
    state_changed: bool = False,
    changed: list[str] | None = None,
    content_in_handoff: bool = False,
    next_commands: list[str] | dict[str, str] | None = None,
    writes: bool = False,
) -> dict[str, Any]:
    return {
        "source": source,
        "session_id": _short(session_id, limit=MAX_SESSION_ID_CHARS),
        "count": count,
        "query": _short(query, limit=MAX_CONVERSATION_TEXT_CHARS) if query else "",
        "path_display": path_display,
        "next_commands": next_commands
        or {
            "recent": "recent conversation",
            "search": "search conversations for <query>",
            "sessions": "list sessions",
            "summary": "summarize this session",
        },
        "boundaries": {
            "read_only": not writes,
            "writes_files": writes,
            "writes_memory": writes,
            "writes_notes": writes,
            "calls_model": False,
            "executes_tools": False,
            "queues_approval": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "controls_computer": False,
            "executes_side_effect": False,
            "reads_private_data": False,
            "reads_personal_data": False,
        },
        **_conversation_contract(
            state_changed=state_changed,
            changed=changed,
            content_in_handoff=content_in_handoff,
        ),
    }


def _summarize_messages(rows) -> str:
    user_messages = [row["content"].strip() for row in rows if row["role"] == "user" and row["content"].strip()]
    assistant_messages = [row["content"].strip() for row in rows if row["role"] == "assistant" and row["content"].strip()]
    recent_user = user_messages[-5:]
    recent_assistant = assistant_messages[-3:]

    lines = ["## What the operator Asked For", ""]
    if recent_user:
        lines.extend(f"- {_short(message, limit=220)}" for message in recent_user)
    else:
        lines.append("- No user messages captured.")

    lines.extend(["", "## What Jarvis Did", ""])
    if recent_assistant:
        lines.extend(f"- {_short(message, limit=220)}" for message in recent_assistant)
    else:
        lines.append("- No assistant responses captured.")

    action_words = ("continue", "next", "build", "add", "fix", "work", "need", "make")
    likely_next = [message for message in user_messages if any(word in message.lower() for word in action_words)]
    lines.extend(["", "## Likely Continuity", ""])
    if likely_next:
        lines.extend(f"- {_short(message, limit=220)}" for message in likely_next[-5:])
    else:
        lines.append("- No explicit next action detected.")
    return "\n".join(lines)


def make_conversation_tools(
    store: MemoryStore,
    vault: ObsidianVault,
    session_id: str,
    config: JarvisConfig | None = None,
    chat_brain: Callable[[], ChatBrain | None] | None = None,
):
    def _preview_brain() -> ChatBrain:
        if chat_brain is not None:
            try:
                live_brain = chat_brain()
            except Exception:
                live_brain = None
            if isinstance(live_brain, ChatBrain):
                return live_brain
        if config is None:
            return ChatBrain("preview-only", store, vault)
        return ChatBrain(
            config.chat_model,
            store,
            vault,
            model_timeout_seconds=config.chat_timeout_seconds,
            max_reply_tokens=config.chat_max_reply_tokens,
            max_history_messages=config.chat_max_history_messages,
            provider=config.model_provider,
            reasoning_effort=config.chat_reasoning_effort,
            openai_max_output_tokens=config.openai_max_output_tokens,
            allow_remote_personal_context=config.allow_remote_personal_context,
        )

    def _terms(text: str) -> list[str]:
        terms = re.findall(r"[a-z0-9]{4,}", text.lower())
        return [word for word in terms if len(word) >= 4]

    def _includes_personal_state_preview(prompt: str) -> bool:
        low = prompt.lower().strip()
        if not low:
            return True
        if re.search(
            r"\b(?:what do you know|what have you learned|what do you remember) about "
            r"(?:me|the operator)(?=$|\s*[?.!,;:]|\s+(?:right now|currently|so far|today)(?=$|\s*[?.!,;:])|"
            r"\s+(?:and|including)\s+(?:my|the operator(?:'s)?))",
            low,
        ):
            return True
        if low.rstrip("?.! ") in {"what have you learned", "personal context", "personal state"}:
            return True
        if re.search(r"\bmy personal (?:context|state)\b", low):
            return True
        if re.search(r"\b(?:tell me about myself|who am i as a person)\b", low):
            return True
        korean_phrases = (
            "나에 대해",
            "저에 대해",
            "내 목표",
            "제 목표",
            "내 할 일",
            "제 할 일",
            "내 결정",
            "제 결정",
            "내 관계",
            "제 관계",
            "내 프로필",
            "제 프로필",
            "내 기억",
            "제 기억",
        )
        if any(phrase in low for phrase in korean_phrases):
            return True
        return re.search(
            r"\b(?:my|the operator(?:'s)?)\s+(?:(?:current|active|open|saved|recorded)\s+)?"
            r"(?:goals?|tasks?|commitments?|decisions?|relationships?|people|profile|preferences?|memories|memory)\b",
            low,
        ) is not None

    def build_chat_context(prompt: str, limit: int) -> tuple[str, dict[str, Any]]:
        prompt = _short(prompt, limit=MAX_CONVERSATION_TEXT_CHARS)
        query_terms = _terms(prompt)
        query = " OR ".join(query_terms[:6])
        memories: list[Any] = []
        skills: list[Any] = []
        context_degraded_sources: set[str] = set()
        if query:
            try:
                raw_memories = store.search_memories(query, limit=limit)
            except Exception:
                raw_memories = None
            memories, memory_rows_degraded = _bounded_snapshot_rows(raw_memories, limit)
            if memory_rows_degraded:
                context_degraded_sources.add("memory")
            try:
                raw_skills = store.search_active_skills(" ".join(query_terms[:6]), limit=3)
            except Exception:
                raw_skills = None
            skills, skill_rows_degraded = _bounded_snapshot_rows(raw_skills, 3)
            if skill_rows_degraded:
                context_degraded_sources.add("skills")
            if not skills:
                seen_skill_ids = set()
                for term in query_terms[:6]:
                    try:
                        raw_term_skill_rows = store.search_active_skills(term, limit=3)
                    except Exception:
                        raw_term_skill_rows = None
                    term_skill_rows, term_skills_degraded = _bounded_snapshot_rows(
                        raw_term_skill_rows,
                        3,
                    )
                    if term_skills_degraded:
                        context_degraded_sources.add("skills")
                    for row in term_skill_rows:
                        row_id = _row_value(row, "id")
                        if row_id in seen_skill_ids:
                            continue
                        seen_skill_ids.add(row_id)
                        skills.append(row)
                        if len(skills) >= 3:
                            break
                    if len(skills) >= 3:
                        break
        if not memories:
            try:
                raw_memories = store.recent_memories(limit=min(limit, 5))
            except Exception:
                raw_memories = None
            memories, recent_memories_degraded = _bounded_snapshot_rows(
                raw_memories,
                min(limit, 5),
            )
            if recent_memories_degraded:
                context_degraded_sources.add("memory")
        try:
            raw_preferences = store.list_preferences(status="active", limit=8)
        except Exception:
            raw_preferences = None
        preferences, preferences_degraded = _bounded_snapshot_rows(raw_preferences, 8)
        if preferences_degraded:
            context_degraded_sources.add("preferences")
        try:
            raw_recent = store.recent_messages(limit=8, session_id=session_id)
        except Exception:
            raw_recent = None
        recent_rows, recent_rows_degraded = _bounded_snapshot_rows(raw_recent, 16)
        if recent_rows_degraded:
            context_degraded_sources.add("recent_messages")
        recent, unreadable_recent = _safe_message_payloads(recent_rows)
        if unreadable_recent:
            context_degraded_sources.add("recent_messages")
        if len(recent) > 8:
            recent = recent[-8:]
            context_degraded_sources.add("recent_messages")
        profile = ""
        profile_degraded = False
        try:
            profile, profile_state = _preview_brain()._read_profile_context()
            profile = profile[:900].strip()
            profile_degraded = profile_state == "unavailable"
        except Exception:
            profile = ""
            profile_degraded = True
        if profile_degraded:
            context_degraded_sources.add("profile")
        if profile == "# Profile":
            profile = ""

        include_personal_state = _includes_personal_state_preview(prompt)
        personal_state_lines: list[str] = []
        personal_state_metadata: dict[str, Any] = {
            "self_knowledge_state": "not_requested",
            "self_knowledge_source_states": {},
            "self_knowledge_counts": {},
            "self_knowledge_unreadable_rows": 0,
            "self_knowledge_truncated_sources": [],
            "self_knowledge_clipped_sources": [],
            "self_knowledge_contract_degraded": False,
            "self_knowledge_display_truncated_sources": [],
            "self_knowledge_output_truncated": False,
        }
        if include_personal_state:
            try:
                personal_snapshot = store.read_self_knowledge_snapshot()
            except Exception:
                personal_snapshot = None
            personal_state_lines, personal_state_metadata = _self_knowledge_preview(personal_snapshot)

        lines = [
            "Jarvis chat context preview:",
            "This explicit read-only preview checks Jarvis's stored personal context. It does not call a model, execute tools, or change memory.",
            "",
            "Prompt:",
            f"- {prompt or '(no prompt supplied)'}",
        ]
        if personal_state_lines:
            lines.extend(["", *personal_state_lines])
        lines.extend(["", "Profile context:"])
        if profile:
            profile_bits = [line.strip("# ").strip() for line in profile.splitlines() if line.strip() and line.strip() != "# Profile"]
            for line in profile_bits[:4]:
                lines.append(f"- {_short(line, limit=180)}")
        else:
            lines.append("- No curated profile context yet.")

        lines.extend(["", "Active preferences:"])
        if preferences:
            for row in preferences[:8]:
                category = _row_value(row, "category", "uncategorized")
                key = _row_value(row, "key", "preference")
                value = _row_value(row, "value", "<unreadable>")
                lines.append(f"- [{_short(category, limit=40)}] {_short(key, limit=80)}: {value}")
        else:
            lines.append("- No active preferences.")

        lines.extend(["", "Relevant memories:"])
        if memories:
            for row in memories[:limit]:
                memory_id = _row_value(row, "id", "?")
                category = _row_value(row, "category", "memory")
                title = _row_value(row, "title", "<unreadable>")
                body = _row_value(row, "body", "<unreadable>")
                lines.append(f"- #{memory_id} [{_short(category, limit=40)}] {title}: {_short(body, limit=180)}")
        else:
            lines.append("- No memories available.")

        lines.extend(["", "Relevant skills:"])
        if skills:
            for row in skills:
                name = _row_value(row, "name", "<unreadable>")
                trigger = _row_value(row, "trigger", "<unreadable>")
                lines.append(f"- {_short(name, limit=120)} | trigger: {trigger}")
        else:
            lines.append("- No matching skills.")

        lines.extend(["", "Recent conversation window:"])
        if recent:
            for row in recent[-6:]:
                lines.append(f"- {_short(row['role'], limit=40)}: {_short(row['content'], limit=180)}")
        else:
            lines.append("- No recent messages in this session.")
        if unreadable_recent:
            lines.append(f"- {unreadable_recent} unreadable message row(s) hidden for safety.")

        output, output_truncated = _render_with_preserved_footer(
            lines,
            [
                "Safety reminder:",
                "- Chat context can inform answers, but tool execution still goes through the planner, registry, permission policy, and approval queue.",
            ],
            MAX_CHAT_CONTEXT_OUTPUT_CHARS,
        )
        return output, _safe_metadata(
            reads_private_data=True,
            reads_personal_data=True,
            memories=len(memories),
            preferences=len(preferences),
            skills=len(skills),
            recent_messages=len(recent),
            readable_message_rows=len(recent),
            unreadable_message_rows=unreadable_recent,
            has_profile=bool(profile),
            personal_state_included=include_personal_state,
            context_output_truncated=output_truncated,
            context_degraded_sources=sorted(context_degraded_sources),
            **personal_state_metadata,
        )

    def _uses_grounded_memory_reply(prompt: str) -> bool:
        low = prompt.lower()
        memory_phrases = (
            "memory",
            "remember",
            "what do you know about me",
            "what do you know about the operator",
            "what have you learned",
            "profile",
            "preference",
        )
        return any(phrase in low for phrase in memory_phrases)

    def _dedupe_short(items: list[str], limit: int) -> list[str]:
        seen: set[str] = set()
        kept: list[str] = []
        for item in items:
            compact = " ".join(item.split()).strip()
            if not compact:
                continue
            key = compact.lower()
            if key in seen:
                continue
            seen.add(key)
            kept.append(_short(compact, limit=220))
            if len(kept) >= limit:
                break
        return kept

    def search_conversations(args: dict[str, Any]) -> ToolResult:
        query = _short(args.get("query"), limit=MAX_CONVERSATION_TEXT_CHARS)
        if not query:
            return ToolResult("search_conversations", False, "Conversation search query is empty.", _safe_metadata())
        raw_rows = store.search_messages(query, _bounded_limit(args.get("limit", 12), 12))
        rows, unreadable_rows = _safe_message_payloads(raw_rows)
        handoff = _conversation_handoff(
            source="search_conversations",
            session_id=session_id,
            count=len(rows),
            query=query,
            content_in_handoff=bool(rows),
            next_commands={
                "recent": "recent conversation",
                "summarize": "summarize this session",
                "search_again": "search conversations for <query>",
                "list_sessions": "list sessions",
            },
        )
        if not rows:
            output = f"No readable conversation messages found for '{query}'."
            if unreadable_rows:
                output += f" {unreadable_rows} unreadable message row(s) hidden for safety."
            return ToolResult(
                "search_conversations",
                True,
                output,
                _conversation_handoff_metadata(
                    "conversation_search_handoff",
                    handoff,
                    count=0,
                    query=query,
                    readable_message_rows=0,
                    unreadable_message_rows=unreadable_rows,
                ),
            )
        lines = [
            f"- [{_short(row['created_at'], limit=80)}] {_short(row['session_id'], limit=80)} {_short(row['role'], limit=40)}: {_short(row['content'], limit=220)}"
            for row in rows
        ]
        if unreadable_rows:
            lines.append(f"- {unreadable_rows} unreadable message row(s) hidden for safety.")
        return ToolResult(
            "search_conversations",
            True,
            "\n".join(lines),
            _conversation_handoff_metadata(
                "conversation_search_handoff",
                handoff,
                count=len(rows),
                query=query,
                readable_message_rows=len(rows),
                unreadable_message_rows=unreadable_rows,
            ),
        )

    def recent_conversation(args: dict[str, Any]) -> ToolResult:
        target_session = _short(args.get("session_id") or session_id, limit=MAX_SESSION_ID_CHARS)
        raw_rows = store.recent_messages(_bounded_limit(args.get("limit", 20), 20), target_session)
        rows, unreadable_rows = _safe_message_payloads(raw_rows)
        handoff = _conversation_handoff(
            source="recent_conversation",
            session_id=target_session,
            count=len(rows),
            content_in_handoff=bool(rows),
            next_commands={
                "search": "search conversations for <query>",
                "summarize": "summarize this session",
                "export": "export this session",
                "list_sessions": "list sessions",
            },
        )
        if not rows:
            output = "No readable conversation history yet."
            if unreadable_rows:
                output += f" {unreadable_rows} unreadable message row(s) hidden for safety."
            return ToolResult(
                "recent_conversation",
                True,
                output,
                _conversation_handoff_metadata(
                    "conversation_recent_handoff",
                    handoff,
                    count=0,
                    session_id=target_session,
                    readable_message_rows=0,
                    unreadable_message_rows=unreadable_rows,
                ),
            )
        lines = [f"{_short(row['role'], limit=40)}: {_short(row['content'], limit=300)}" for row in rows]
        if unreadable_rows:
            lines.append(f"{unreadable_rows} unreadable message row(s) hidden for safety.")
        return ToolResult(
            "recent_conversation",
            True,
            "\n".join(lines),
            _conversation_handoff_metadata(
                "conversation_recent_handoff",
                handoff,
                count=len(rows),
                session_id=target_session,
                readable_message_rows=len(rows),
                unreadable_message_rows=unreadable_rows,
            ),
        )

    def _message_metadata(row) -> dict[str, Any]:
        try:
            raw = row["metadata"]
        except Exception:
            return {}
        if isinstance(raw, dict):
            return raw
        try:
            value = json.loads(raw or "{}")
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}

    def chat_response_health(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_limit(args.get("limit", 24), 24, 100)
        raw_rows = store.recent_messages(limit, _short(args.get("session_id") or session_id, limit=MAX_SESSION_ID_CHARS))
        rows, unreadable_rows = _safe_message_payloads(raw_rows)
        assistant_metadata = [_message_metadata(row) for row in rows if row["role"] == "assistant"]
        chat_responses = [
            metadata.get("chat_response") or {}
            for metadata in assistant_metadata
            if metadata.get("runtime_route") == "chat"
        ]
        ordinary_chat_responses = [
            response
            for response in chat_responses
            if response.get("runtime_route") != "command_suggestion"
            and not response.get("pre_planner_command_suggestion")
        ]
        tool_responses = [
            metadata
            for metadata in assistant_metadata
            if metadata.get("runtime_route") == "tools"
        ]
        source_counts: dict[str, int] = {}
        fallback_count = 0
        model_count = 0
        external_model_attempts = 0
        usage_receipt_count = 0
        recorded_input_tokens = 0
        recorded_cached_input_tokens = 0
        recorded_output_tokens = 0
        recorded_reasoning_tokens = 0
        recorded_total_tokens = 0
        latency_samples: list[float] = []
        for response in ordinary_chat_responses:
            source = str(response.get("source") or "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
            if response.get("used_fallback"):
                fallback_count += 1
            if response.get("used_model"):
                model_count += 1
            if response.get("calls_external_service"):
                external_model_attempts += 1
            input_tokens = _model_usage_count(response, "model_input_tokens")
            output_tokens = _model_usage_count(response, "model_output_tokens")
            total_tokens = _model_usage_count(response, "model_total_tokens")
            if input_tokens is not None and output_tokens is not None and total_tokens is not None:
                usage_receipt_count += 1
                recorded_input_tokens += input_tokens
                recorded_output_tokens += output_tokens
                recorded_total_tokens += total_tokens
                cached_tokens = _model_usage_count(response, "model_cached_input_tokens")
                reasoning_tokens = _model_usage_count(response, "model_reasoning_tokens")
                recorded_cached_input_tokens += cached_tokens or 0
                recorded_reasoning_tokens += reasoning_tokens or 0
            latency_ms = _chat_latency_ms(response)
            if latency_ms is not None:
                latency_samples.append(latency_ms)
        source_bits = ", ".join(f"{source}={count}" for source, count in sorted(source_counts.items())) or "none"
        ordinary_count = len(ordinary_chat_responses)
        latency_p50_ms = _percentile_ms(latency_samples, 0.50)
        latency_p95_ms = _percentile_ms(latency_samples, 0.95)
        ws4_turns_remaining = max(0, MIN_WS4_CHAT_MEASUREMENT_TURNS - ordinary_count)
        ws4_latency_samples_remaining = max(0, MIN_WS4_CHAT_MEASUREMENT_TURNS - len(latency_samples))
        ws4_measurement_ready = ordinary_count >= MIN_WS4_CHAT_MEASUREMENT_TURNS and len(latency_samples) >= MIN_WS4_CHAT_MEASUREMENT_TURNS
        if latency_p50_ms is None or latency_p95_ms is None:
            latency_line = "- latency p50/p95: unavailable (no recorded latency samples in this window)"
        else:
            latency_line = f"- latency p50/p95: {latency_p50_ms}ms / {latency_p95_ms}ms"
        lines = [
            "Jarvis chat response health:",
            "",
            f"- recent assistant messages inspected: {len(assistant_metadata)}",
            f"- readable message rows: {len(rows)}",
            f"- unreadable message rows hidden: {unreadable_rows}",
            f"- chat-route responses: {len(chat_responses)}",
            f"- chat-produced responses: {len(chat_responses)}",
            f"- ordinary chat responses: {ordinary_count}",
            f"- tool-routed responses: {len(tool_responses)}",
            f"- source counts: {source_bits}",
            f"- model-backed responses: {model_count} ({_percent(model_count, ordinary_count)}%)",
            f"- fallback responses: {fallback_count} ({_percent(fallback_count, ordinary_count)}%)",
            f"- external model attempts: {external_model_attempts}",
            f"- content-free token usage receipts: {usage_receipt_count}",
            f"- recorded input/output/total tokens: {recorded_input_tokens} / {recorded_output_tokens} / {recorded_total_tokens}",
            f"- recorded cached input/reasoning tokens: {recorded_cached_input_tokens} / {recorded_reasoning_tokens}",
            f"- latency samples: {len(latency_samples)}",
            latency_line,
            "",
            "WS4 measurement readiness:",
            f"- required real mixed chat turns: {MIN_WS4_CHAT_MEASUREMENT_TURNS}",
            f"- turns remaining: {ws4_turns_remaining}",
            f"- latency samples remaining: {ws4_latency_samples_remaining}",
            f"- ready for WS4 measurement review: {ws4_measurement_ready}",
            "",
            "Boundary:",
            "- This report reads stored response metadata only; it does not call models, execute tools, read private data, control the computer, or queue approvals.",
        ]
        if ordinary_chat_responses:
            latest = ordinary_chat_responses[-1]
            lines.extend(
                [
                    "",
                    "Latest chat response:",
                    f"- source: {latest.get('source', 'unknown')}",
                    f"- model: {latest.get('model', 'unknown')}",
                    f"- model provider: {latest.get('model_provider', 'unknown')}",
                    f"- model call attempted: {'yes' if latest.get('model_call_attempted') else 'no'}",
                    f"- external model call: {'yes' if latest.get('calls_external_service') else 'no'}",
                    f"- external request outcome: {latest.get('external_model_request_outcome', 'unknown')}",
                    f"- conversation shared with external model: {'yes' if latest.get('shares_conversation_with_external_model') else 'no'}",
                    f"- current message included in external request attempt: {'yes' if latest.get('current_user_message_in_external_model_request') else 'no'}",
                    f"- stored personal context included in external request attempt: {'yes' if latest.get('stored_personal_context_in_external_model_request') else 'no'}",
                    f"- prior chat history included in external request attempt: {'yes' if latest.get('history_in_external_model_request') else 'no'}",
                    f"- remote stored-context policy: {latest.get('remote_personal_context_policy', 'unknown')}",
                    "- personal-context content copied into health metadata: no",
                    "- prompt/response content copied into health metadata: no",
                    f"- fallback: {bool(latest.get('used_fallback'))}",
                    f"- timeout: {latest.get('model_timeout_seconds', 'unknown')}s",
                ]
            )
            if latest.get("model_error"):
                lines.append(f"- model error: {str(latest['model_error'])[:180]}")
            if latest.get("model_usage_available") is True:
                lines.extend(
                    [
                        f"- recorded input/output/total tokens: {latest.get('model_input_tokens', 0)} / {latest.get('model_output_tokens', 0)} / {latest.get('model_total_tokens', 0)}",
                        f"- cached input tokens: {latest.get('model_cached_input_tokens', 0) if latest.get('model_cached_input_tokens_available') else 'unavailable'}",
                        f"- reasoning tokens: {latest.get('model_reasoning_tokens', 0) if latest.get('model_reasoning_tokens_available') else 'unavailable'}",
                        "- token receipt contains counts only; no prompt or response content",
                    ]
                )
            elif latest.get("model_call_attempted"):
                lines.append("- token usage: unavailable for this model response")
        else:
            lines.extend(["", "Latest chat response:", "- No ordinary chat response metadata in the recent window yet."])
        return ToolResult(
            "chat_response_health",
            True,
            "\n".join(lines),
            {
                "assistant_messages": len(assistant_metadata),
                "chat_responses": len(chat_responses),
                "ordinary_chat_responses": ordinary_count,
                "tool_responses": len(tool_responses),
                "source_counts": source_counts,
                "model_responses": model_count,
                "fallback_responses": fallback_count,
                "external_model_attempts": external_model_attempts,
                "model_usage_receipts": usage_receipt_count,
                "recorded_model_input_tokens": recorded_input_tokens,
                "recorded_model_cached_input_tokens": recorded_cached_input_tokens,
                "recorded_model_output_tokens": recorded_output_tokens,
                "recorded_model_reasoning_tokens": recorded_reasoning_tokens,
                "recorded_model_total_tokens": recorded_total_tokens,
                "latest_model_provider": (
                    str(ordinary_chat_responses[-1].get("model_provider") or "unknown")
                    if ordinary_chat_responses
                    else ""
                ),
                "latest_model_call_attempted": bool(
                    ordinary_chat_responses and ordinary_chat_responses[-1].get("model_call_attempted")
                ),
                "latest_calls_external_service": bool(
                    ordinary_chat_responses and ordinary_chat_responses[-1].get("calls_external_service")
                ),
                "latest_external_model_request_outcome": (
                    str(
                        ordinary_chat_responses[-1].get(
                            "external_model_request_outcome"
                        )
                        or "unknown"
                    )
                    if ordinary_chat_responses
                    else ""
                ),
                "latest_shares_conversation_with_external_model": bool(
                    ordinary_chat_responses
                    and ordinary_chat_responses[-1].get("shares_conversation_with_external_model")
                ),
                "latest_current_user_message_shared_with_external_model": bool(
                    ordinary_chat_responses
                    and ordinary_chat_responses[-1].get(
                        "current_user_message_shared_with_external_model"
                    )
                ),
                "latest_shares_stored_personal_context_with_external_model": bool(
                    ordinary_chat_responses
                    and ordinary_chat_responses[-1].get(
                        "shares_stored_personal_context_with_external_model"
                    )
                ),
                "latest_shares_history_with_external_model": bool(
                    ordinary_chat_responses
                    and ordinary_chat_responses[-1].get(
                        "shares_history_with_external_model"
                    )
                ),
                "latest_remote_personal_context_policy": (
                    str(
                        ordinary_chat_responses[-1].get(
                            "remote_personal_context_policy"
                        )
                        or "unknown"
                    )
                    if ordinary_chat_responses
                    else ""
                ),
                "latest_remote_personal_context_sources_shared": (
                    list(
                        ordinary_chat_responses[-1].get(
                            "remote_personal_context_sources_shared"
                        )
                        or []
                    )
                    if ordinary_chat_responses
                    else []
                ),
                "latest_remote_personal_context_content_in_metadata": False,
                "latest_model_usage_available": bool(
                    ordinary_chat_responses
                    and ordinary_chat_responses[-1].get("model_usage_available") is True
                ),
                "latest_model_input_tokens": (
                    _model_usage_count(ordinary_chat_responses[-1], "model_input_tokens") or 0
                    if ordinary_chat_responses
                    else 0
                ),
                "latest_model_cached_input_tokens": (
                    _model_usage_count(ordinary_chat_responses[-1], "model_cached_input_tokens") or 0
                    if ordinary_chat_responses
                    else 0
                ),
                "latest_model_output_tokens": (
                    _model_usage_count(ordinary_chat_responses[-1], "model_output_tokens") or 0
                    if ordinary_chat_responses
                    else 0
                ),
                "latest_model_reasoning_tokens": (
                    _model_usage_count(ordinary_chat_responses[-1], "model_reasoning_tokens") or 0
                    if ordinary_chat_responses
                    else 0
                ),
                "latest_model_total_tokens": (
                    _model_usage_count(ordinary_chat_responses[-1], "model_total_tokens") or 0
                    if ordinary_chat_responses
                    else 0
                ),
                "model_usage_content_in_health_metadata": False,
                "model_request_content_in_health_metadata": False,
                "model_response_content_in_health_metadata": False,
                "model_response_rate_pct": _percent(model_count, ordinary_count),
                "fallback_response_rate_pct": _percent(fallback_count, ordinary_count),
                "latency_measurement_available": bool(latency_samples),
                "latency_sample_count": len(latency_samples),
                "latency_p50_ms": latency_p50_ms,
                "latency_p95_ms": latency_p95_ms,
                "ws4_required_chat_turns": MIN_WS4_CHAT_MEASUREMENT_TURNS,
                "ws4_turns_remaining": ws4_turns_remaining,
                "ws4_latency_samples_remaining": ws4_latency_samples_remaining,
                "ws4_measurement_ready": ws4_measurement_ready,
                "readable_message_rows": len(rows),
                "unreadable_message_rows": unreadable_rows,
                **_safe_metadata(),
            },
        )

    def chat_continuity_brief(args: dict[str, Any]) -> ToolResult:
        target_session = _short(args.get("session_id") or session_id, limit=MAX_SESSION_ID_CHARS)
        limit = max(6, _bounded_limit(args.get("limit", 32), 32, 100))
        raw_rows = store.recent_messages(limit, target_session)
        rows, unreadable_rows = _safe_message_payloads(raw_rows)
        if not rows:
            output = f"No readable conversation history found for session {target_session}."
            if unreadable_rows:
                output += f" {unreadable_rows} unreadable message row(s) hidden for safety."
            return ToolResult(
                "chat_continuity_brief",
                True,
                output,
                _read_only_metadata(
                    session_id=target_session,
                    messages=0,
                    readable_message_rows=0,
                    unreadable_message_rows=unreadable_rows,
                ),
            )

        user_messages = [row["content"].strip() for row in rows if row["role"] == "user" and row["content"].strip()]
        assistant_rows = [row for row in rows if row["role"] == "assistant"]
        assistant_messages = [row["content"].strip() for row in assistant_rows if row["content"].strip()]
        latest_metadata = _message_metadata(assistant_rows[-1]) if assistant_rows else {}
        latest_chat_response = latest_metadata.get("chat_response") or {}
        latest_route = latest_metadata.get("runtime_route") or "unknown"
        pending_approvals = len(store.list_pending_approvals(limit=100))

        recent_requests = _dedupe_short(user_messages[-8:], 5)
        recent_responses = _dedupe_short(assistant_messages[-5:], 3)
        actionable_requests = _dedupe_short(
            [
                message
                for message in user_messages
                if any(
                    word in message.lower()
                    for word in (
                        "continue",
                        "build",
                        "work",
                        "add",
                        "fix",
                        "implement",
                        "verify",
                        "test",
                        "make",
                    )
                )
            ][-8:],
            5,
        )

        lines = [
            "Jarvis chat continuity brief:",
            "This is read-only. It catches up the current conversation without calling models, executing tools, writing memory, controlling the computer, or queuing approvals.",
            "",
            f"Session: {target_session}",
            f"Messages reviewed: {len(rows)} ({len(user_messages)} user, {len(assistant_messages)} assistant)",
            f"Unreadable message rows hidden: {unreadable_rows}",
            f"Pending approvals: {pending_approvals}",
            "",
            "Recent user asks:",
        ]
        lines.extend(f"- {item}" for item in recent_requests) if recent_requests else lines.append("- None captured.")
        lines.extend(["", "Recent Jarvis responses:"])
        lines.extend(f"- {item}" for item in recent_responses) if recent_responses else lines.append("- None captured.")
        lines.extend(["", "Likely continuation thread:"])
        lines.extend(f"- {item}" for item in actionable_requests) if actionable_requests else lines.append("- No explicit build/continue thread detected.")
        lines.extend(
            [
                "",
                "Latest response path:",
                f"- runtime route: {latest_route}",
            ]
        )
        if latest_chat_response:
            lines.extend(
                [
                    f"- chat source: {latest_chat_response.get('source', 'unknown')}",
                    f"- model: {latest_chat_response.get('model', 'unknown')}",
                    f"- fallback: {bool(latest_chat_response.get('used_fallback'))}",
                    f"- timeout: {latest_chat_response.get('model_timeout_seconds', 'unknown')}s",
                ]
            )
            if latest_chat_response.get("model_error"):
                lines.append(f"- model error: {str(latest_chat_response['model_error'])[:180]}")
        else:
            lines.append("- chat source: no chat-produced response metadata on the latest assistant message")
        lines.extend(
            [
                "",
                "Safe next checks:",
                "- `chat response health` to inspect model/fallback sources.",
                "- `session learning preview` to see what might be worth saving.",
                "- `work queue` or `safe next actions` before continuing autonomous work.",
                "",
                "Boundary:",
                "- This brief is orientation only. Shell/code, personal data, external side effects, destructive changes, and computer control remain approval-gated.",
            ]
        )
        return ToolResult(
            "chat_continuity_brief",
            True,
            "\n".join(lines),
            _read_only_metadata(
                session_id=target_session,
                messages=len(rows),
                user_messages=len(user_messages),
                assistant_messages=len(assistant_messages),
                readable_message_rows=len(rows),
                unreadable_message_rows=unreadable_rows,
                pending_approvals=pending_approvals,
                latest_runtime_route=_short(latest_route, limit=80),
                latest_chat_source=_short(latest_chat_response.get("source") if latest_chat_response else "", limit=80),
            ),
        )

    def list_sessions(args: dict[str, Any]) -> ToolResult:
        try:
            raw_rows = store.list_sessions(_bounded_limit(args.get("limit", 20), 20, 100))
        except Exception:
            raw_rows = []
        rows, unreadable_rows = _safe_session_payloads(raw_rows)
        handoff = _conversation_handoff(
            source="list_sessions",
            session_id=session_id,
            count=len(rows),
            content_in_handoff=bool(rows),
            next_commands={
                "recent": "recent conversation",
                "summarize": "summarize this session",
                "export": "export this session",
                "search": "search conversations for <query>",
            },
        )
        if not rows:
            output = "No sessions yet."
            if unreadable_rows:
                output = f"No readable sessions yet. {unreadable_rows} unreadable session row(s) hidden for safety."
            return ToolResult(
                "list_sessions",
                True,
                output,
                _conversation_handoff_metadata(
                    "conversation_sessions_handoff",
                    handoff,
                    count=0,
                    readable_session_rows=0,
                    unreadable_session_rows=unreadable_rows,
                ),
            )
        lines = [
            f"- {_short(row['session_id'], limit=MAX_SESSION_ID_CHARS)} | {row['messages']} messages | {_short(row['started_at'], limit=80)} to {_short(row['last_at'], limit=80)}"
            for row in rows
        ]
        if unreadable_rows:
            lines.append(f"- {unreadable_rows} unreadable session row(s) hidden for safety.")
        return ToolResult(
            "list_sessions",
            True,
            "\n".join(lines),
            _conversation_handoff_metadata(
                "conversation_sessions_handoff",
                handoff,
                count=len(rows),
                readable_session_rows=len(rows),
                unreadable_session_rows=unreadable_rows,
            ),
        )

    def export_session(args: dict[str, Any]) -> ToolResult:
        target_session = _short(args.get("session_id") or session_id, limit=MAX_SESSION_ID_CHARS)
        try:
            raw_rows = store.recent_messages(_bounded_limit(args.get("limit", 200), 200, 300), target_session)
        except Exception:
            raw_rows = []
        rows, unreadable_rows = _safe_message_payloads(raw_rows)
        if not rows:
            output = f"No readable messages found for session {target_session}."
            if unreadable_rows:
                output += f" {unreadable_rows} unreadable message row(s) hidden for safety."
            return ToolResult(
                "export_session",
                False,
                output,
                _read_only_metadata(
                    session_id=target_session,
                    messages=0,
                    readable_message_rows=0,
                    unreadable_message_rows=unreadable_rows,
                ),
            )
        lines = [f"# Session {target_session}\n"]
        for row in rows:
            role = _short(row["role"], limit=40)
            created_at = _short(row["created_at"], limit=80)
            content = _short(row["content"], limit=1200)
            lines.append(f"## {role} | {created_at}\n\n{content}\n")
        if unreadable_rows:
            lines.append(f"## Hidden Rows\n\n- {unreadable_rows} unreadable message row(s) hidden for safety.\n")
        path = vault.write_session(target_session, "\n".join(lines))
        path_display = _safe_vault_path_display(path, vault)
        output = f"Session exported: {path_display}"
        if unreadable_rows:
            output += f"\n{unreadable_rows} unreadable message row(s) hidden for safety."
        return ToolResult(
            "export_session",
            True,
            output,
            _conversation_handoff_metadata(
                "conversation_export_handoff",
                _conversation_handoff(
                    source="export_session",
                    session_id=target_session,
                    count=len(rows),
                    path_display=path_display,
                    state_changed=True,
                    changed=["session_export"],
                    content_in_handoff=False,
                    next_commands={
                        "read_export": f"read jarvis note {path_display}",
                        "summarize": "summarize this session",
                        "recent": "recent conversation",
                    },
                    writes=True,
                ),
                writes=True,
                path=str(path),
                path_display=path_display,
                session_id=target_session,
                messages=len(rows),
                readable_message_rows=len(rows),
                unreadable_message_rows=unreadable_rows,
            ),
        )

    def summarize_session(args: dict[str, Any]) -> ToolResult:
        target_session = _short(args.get("session_id") or session_id, limit=MAX_SESSION_ID_CHARS)
        try:
            raw_rows = store.recent_messages(_bounded_limit(args.get("limit", 200), 200, 300), target_session)
        except Exception:
            raw_rows = []
        rows, unreadable_rows = _safe_message_payloads(raw_rows)
        if not rows:
            output = f"No readable messages found for session {target_session}."
            if unreadable_rows:
                output += f" {unreadable_rows} unreadable message row(s) hidden for safety."
            return ToolResult(
                "summarize_session",
                False,
                output,
                _read_only_metadata(
                    session_id=target_session,
                    messages=0,
                    readable_message_rows=0,
                    unreadable_message_rows=unreadable_rows,
                ),
            )
        body = f"# Session Reflection {target_session}\n\n" + _summarize_messages(rows) + "\n"
        if unreadable_rows:
            body += f"\n## Hidden Rows\n\n- {unreadable_rows} unreadable message row(s) hidden for safety.\n"
        path = vault.write_reflection(f"Session {target_session}", body)
        path_display = _safe_vault_path_display(path, vault)
        output = f"Session summary written: {path_display}\n\n{body}"
        return ToolResult(
            "summarize_session",
            True,
            output,
            _conversation_handoff_metadata(
                "conversation_summary_handoff",
                _conversation_handoff(
                    source="summarize_session",
                    session_id=target_session,
                    count=len(rows),
                    path_display=path_display,
                    state_changed=True,
                    changed=["session_summary"],
                    content_in_handoff=True,
                    next_commands={
                        "read_summary": f"read jarvis note {path_display}",
                        "export": "export this session",
                        "draft_skill": "draft skill from this session called <name>",
                    },
                    writes=True,
                ),
                writes=True,
                path=str(path),
                path_display=path_display,
                session_id=target_session,
                messages=len(rows),
                readable_message_rows=len(rows),
                unreadable_message_rows=unreadable_rows,
            ),
        )

    def draft_skill_from_session(args: dict[str, Any]) -> ToolResult:
        target_session = _short(args.get("session_id") or session_id, limit=MAX_SESSION_ID_CHARS)
        name = _short(args.get("name"), limit=MAX_SKILL_FIELD_CHARS)
        trigger = _short(args.get("trigger"), limit=MAX_CONVERSATION_TEXT_CHARS)
        try:
            raw_rows = store.recent_messages(_bounded_limit(args.get("limit", 80), 80, 160), target_session)
        except Exception:
            raw_rows = []
        rows, unreadable_rows = _safe_message_payloads(raw_rows)
        if not rows:
            output = f"No readable messages found for session {target_session}."
            if unreadable_rows:
                output += f" {unreadable_rows} unreadable message row(s) hidden for safety."
            return ToolResult(
                "draft_skill_from_session",
                False,
                output,
                _read_only_metadata(
                    session_id=target_session,
                    messages=0,
                    readable_message_rows=0,
                    unreadable_message_rows=unreadable_rows,
                ),
            )

        user_messages = [row["content"].strip() for row in rows if row["role"] == "user" and row["content"].strip()]
        if not name:
            name = _short("Draft Skill - " + (user_messages[-1][:48] if user_messages else target_session), limit=MAX_SKILL_FIELD_CHARS)
        if not trigger:
            trigger = "When a similar workflow appears again."

        source_messages = _dedupe_short(user_messages[-10:], 8)
        procedure_lines = [
            "> Review status: DRAFT. Human review is required before this skill is treated as reliable.",
            "",
            "This is a reviewable skill draft inferred from a Jarvis session. It may help repeat a workflow, but it must be edited by the operator before promotion.",
            "",
            "## Human Review Gate",
            "",
            "- Keep as draft until the operator confirms the trigger, scope, and verification steps.",
            "- Do not let this draft edit code, approve actions, send messages, read personal data, control the computer, or run shell commands by itself.",
            "- Promote only after a focused smoke test or a reviewed live run proves the workflow.",
            "",
            "## Recognize",
            "",
            trigger,
            "",
            "## Procedure",
            "",
        ]
        actionable = [
            message
            for message in user_messages
            if any(word in message.lower() for word in ("create", "add", "run", "fix", "continue", "test", "schedule", "export", "summarize"))
        ]
        if actionable:
            for index, message in enumerate(actionable[-8:], start=1):
                procedure_lines.append(f"{index}. {_short(message, limit=300)}")
        else:
                procedure_lines.append("1. Reconstruct the user's goal from the session.")
                procedure_lines.append("2. Identify the tool calls or notes that helped.")
                procedure_lines.append("3. Repeat the useful steps and verify the result.")

        procedure_lines.extend(
            [
                "",
                "## Verification",
                "",
                "- Confirm the intended output exists.",
                "- Run the relevant smoke test or status command when available.",
                "- Ask the operator before any high-risk side effect.",
                "",
                "## Failure And Stop Conditions",
                "",
                "- Stop if the target app, file, account, recipient, date, coordinate, or side effect is ambiguous.",
                "- Stop if any step needs personal data, external side effects, shell/code, destructive file changes, or computer control without an explicit approval receipt.",
                "- Convert repeated misses into a smoke test or a smaller reviewed skill draft instead of broadening this one.",
                "",
                "## Source Session Signals",
                "",
            ]
        )
        if source_messages:
            procedure_lines.extend(f"- {_short(message, limit=220)}" for message in source_messages)
        else:
            procedure_lines.append("- No user messages captured.")
        if unreadable_rows:
            procedure_lines.append(f"- {unreadable_rows} unreadable message row(s) hidden for safety.")
        body = "\n".join(procedure_lines)
        tags = "draft,session,human-review-required"
        try:
            target = store.save_skill_draft_with_projection_target(
                SkillRecord(name=name, trigger=trigger, body=body, tags=tags)
            )
        except ValueError:
            return ToolResult(
                "draft_skill_from_session",
                False,
                (
                    f"A skill named '{name}' already exists. It was preserved; choose a "
                    "different draft name or review the existing skill first."
                ),
                _read_only_metadata(
                    reason="skill_name_conflict",
                    session_id=target_session,
                    review_status="draft_refused",
                    existing_skill_preserved=True,
                    human_review_required=True,
                    messages=len(rows),
                    readable_message_rows=len(rows),
                    unreadable_message_rows=unreadable_rows,
                ),
            )
        skill_id = target.skill_id
        projection = reconcile_skill_projection(
            store,
            vault,
            skill_id,
            expected_operation=target.operation,
            expected_revision=target.revision,
            expected_source_digest=target.source_digest,
        )
        current = store.get_skill_by_id(skill_id)
        current_status = _row_value(current, "review_status", "missing")
        current_revision = _safe_int(_row_value(current, "revision", "0"))
        if current_status != "draft" or current_revision != target.revision:
            try:
                logical_current = store.get_skill_by_identity(name)
            except ValueError:
                logical_current = None
                current_status = "ambiguous"
                behaviorally_active = True
            else:
                if logical_current is not None:
                    current_status = _row_value(
                        logical_current, "review_status", "missing"
                    )
                behaviorally_active = current_status == "active"
            superseded_handoff = _conversation_handoff(
                source="draft_skill_from_session",
                session_id=target_session,
                count=len(rows),
                path_display="",
                state_changed=True,
                changed=["skill_draft_superseded"],
                content_in_handoff=False,
                next_commands={
                    "review": f"get skill {name}",
                    "retry": f"draft skill {name} from this session",
                },
                writes=True,
            )
            return ToolResult(
                "draft_skill_from_session",
                False,
                (
                    f"The draft for '{name}' was superseded by another skill update. "
                    "The current skill was preserved; review it before retrying."
                ),
                _conversation_handoff_metadata(
                    "conversation_skill_draft_handoff",
                    superseded_handoff,
                    writes=True,
                    skill_id=skill_id,
                    session_id=target_session,
                    messages=len(rows),
                    readable_message_rows=len(rows),
                    unreadable_message_rows=unreadable_rows,
                    reason="skill_draft_superseded",
                    review_status=current_status,
                    behaviorally_active=behaviorally_active,
                    authorizes_skill_activation=False,
                    human_review_required=True,
                    self_modifying_code=False,
                    skill_projection_operation=projection.operation,
                    skill_projection_status=projection.status,
                    skill_projection_pending=projection.status != "completed",
                ),
            )
        projection_completed = projection.status == "completed" and bool(projection.path_display)
        path_display = projection.path_display if projection_completed else ""
        path = vault.root_path / path_display if projection_completed else None
        if projection_completed:
            output = f"Drafted review-required skill '{name}' from session {target_session}: {path_display}\nHuman review is required before use."
        else:
            output = (
                f"Drafted review-required skill '{name}' from session {target_session} in durable memory.\n"
                f"The Obsidian projection is pending ({projection.status}).\n"
                "Human review is required before use."
            )
        if unreadable_rows:
            output += f"\n{unreadable_rows} unreadable message row(s) hidden for safety."
        next_commands = {
            "read_draft": f"get skill {name}",
            "summarize": "summarize this session",
            "review": f"read jarvis note {path_display}" if projection_completed else f"get skill {name}",
        }
        handoff = _conversation_handoff(
            source="draft_skill_from_session",
            session_id=target_session,
            count=len(rows),
            path_display=path_display,
            state_changed=True,
            changed=["skill_draft"],
            content_in_handoff=False,
            next_commands=next_commands,
            writes=True,
        )
        if not projection_completed:
            handoff["boundaries"]["writes_files"] = False
            handoff["boundaries"]["writes_notes"] = False
        metadata = _conversation_handoff_metadata(
            "conversation_skill_draft_handoff",
            handoff,
            writes=True,
            skill_id=skill_id,
            path=str(path) if path is not None else "",
            path_display=path_display,
            session_id=target_session,
            messages=len(rows),
            readable_message_rows=len(rows),
            unreadable_message_rows=unreadable_rows,
            review_status="draft",
            behaviorally_active=False,
            authorizes_skill_activation=False,
            human_review_required=True,
            self_modifying_code=False,
            source_session_signals=len(source_messages),
            skill_projection_operation=projection.operation,
            skill_projection_status=projection.status,
            skill_projection_pending=not projection_completed,
            skill_projection_content_digest=(
                projection.content_digest if projection_completed else ""
            ),
        )
        if not projection_completed:
            metadata["writes_files"] = False
            metadata["writes_notes"] = False
        return ToolResult(
            "draft_skill_from_session",
            True,
            output,
            metadata,
        )

    def chat_context(args: dict[str, Any]) -> ToolResult:
        prompt = _short(args.get("prompt"), limit=MAX_CONVERSATION_TEXT_CHARS)
        body, metadata = build_chat_context(prompt, _bounded_limit(args.get("limit", 5), 5, MAX_CHAT_CONTEXT_LIMIT))
        return ToolResult("chat_context", True, body, metadata)

    def chat_prompt_preview(args: dict[str, Any]) -> ToolResult:
        prompt = _short(args.get("prompt"), limit=MAX_CONVERSATION_TEXT_CHARS)
        body, context_metadata = build_chat_context(prompt, _bounded_limit(args.get("limit", 5), 5, MAX_CHAT_CONTEXT_LIMIT))
        grounded_memory_reply = _uses_grounded_memory_reply(prompt)
        boundary = _preview_brain().preview_model_boundary(
            prompt,
            has_profile=bool(context_metadata.get("has_profile")),
            preferences=int(context_metadata.get("preferences") or 0),
            memories=int(context_metadata.get("memories") or 0),
            skills=int(context_metadata.get("skills") or 0),
        )
        lines = [
            "Jarvis chat prompt preview:",
            "",
            "Purpose:",
            "- Show bounded grounding context before any model call and identify the local-only personal-state section explicitly.",
            "- This preview does not call Ollama, OpenAI, or any configured model.",
            "- This preview reads bounded Jarvis-owned personal context, but does not call the configured model, execute tools, approve requests, write memory, read outside sources, control the computer, or queue approvals.",
            "",
            "User message:",
            f"- {prompt or '(no prompt supplied)'}",
            "",
            "System prompt preview:",
            SYSTEM_PROMPT.strip(),
            "",
            "Grounding packet preview:",
            body,
            "",
            "Reply path:",
        ]
        if grounded_memory_reply:
            lines.append("- Memory/profile/preference wording would use Jarvis's deterministic grounded-memory reply before model chat.")
        else:
            lines.append("- Normal conversation would try the configured chat model, then use grounded fallback chat if the model is unavailable.")
        lines.extend(
            [
                "",
                "Model boundary:",
                f"- configured provider: {boundary['model_provider']}",
                f"- would call an external model: {'yes' if boundary['would_call_external_service'] else 'no'}",
                f"- would share this current message externally: {'yes' if boundary['would_share_current_message_with_external_model'] else 'no'}",
                f"- would share this local grounding packet externally: {'yes' if boundary['would_share_stored_personal_context_with_external_model'] else 'no'}",
                f"- would share prior chat history externally: {'yes' if boundary.get('would_share_history_with_external_model') else 'no'}",
                f"- remote stored-context opt-in: {boundary['remote_personal_context_policy']}",
            ]
        )
        footer_lines = [
                "Execution boundary:",
                "- Chat may shape words, but actions still go through the planner, ToolRegistry, PermissionPolicy, approval queue, and audit log.",
                "- Shell/code, destructive file changes, clipboard reads, personal data, reminders, external side effects, and computer control remain approval-gated.",
        ]
        metadata = {
            **context_metadata,
            "prompt": prompt,
            "system_prompt_chars": len(SYSTEM_PROMPT),
            "grounded_memory_reply": grounded_memory_reply,
            "model_provider": boundary["model_provider"],
            "would_call_external_service": boundary["would_call_external_service"],
            "would_share_current_message_with_external_model": boundary[
                "would_share_current_message_with_external_model"
            ],
            "would_share_stored_personal_context_with_external_model": boundary[
                "would_share_stored_personal_context_with_external_model"
            ],
            "would_share_history_with_external_model": boundary.get(
                "would_share_history_with_external_model", False
            ),
            "remote_personal_context_allowed": boundary["remote_personal_context_allowed"],
            "remote_personal_context_policy": boundary["remote_personal_context_policy"],
        }
        output, output_truncated = _render_with_preserved_footer(
            lines,
            footer_lines,
            MAX_CHAT_PROMPT_PREVIEW_OUTPUT_CHARS,
        )
        metadata["prompt_preview_output_truncated"] = output_truncated
        return ToolResult("chat_prompt_preview", True, output, metadata)

    def chat_loop_preview(args: dict[str, Any]) -> ToolResult:
        prompt = _short(args.get("prompt"), limit=MAX_CONVERSATION_TEXT_CHARS)
        if not prompt:
            return ToolResult("chat_loop_preview", False, "Give Jarvis a message to preview.", _read_only_metadata())
        brain = _preview_brain()
        body, metadata = brain.preview_loop_text(prompt)
        metadata = _read_only_metadata(**metadata)
        return ToolResult("chat_loop_preview", True, body, metadata)

    def chat_safety_report(args: dict[str, Any]) -> ToolResult:
        profile = ""
        try:
            profile, _profile_state = _preview_brain()._read_profile_context()
            profile = profile[:1200].strip()
        except Exception:
            profile = ""
        if profile == "# Profile":
            profile = ""
        try:
            preferences = store.list_preferences(status="active", limit=20)
        except Exception:
            preferences = []
        try:
            memories = store.recent_memories(limit=8)
        except Exception:
            memories = []
        try:
            skills = store.list_active_skills(limit=8)
        except Exception:
            skills = []
        try:
            raw_recent = store.recent_messages(limit=8, session_id=session_id)
        except Exception:
            raw_recent = []
        recent, unreadable_recent = _safe_message_payloads(raw_recent)
        lines = [
            "Jarvis chat safety report:",
            "",
            "Grounding rules:",
            "- Jarvis may use only visible profile notes, active preferences, saved memories, active reviewed skills, and recent conversation context.",
            "- Memory/profile/preference questions use a deterministic grounded reply instead of free-form model invention.",
            "- If context is thin, Jarvis should say that instead of fabricating personal facts, projects, files, emails, calendar events, or private context.",
            "",
            "Execution boundary:",
            "- Chat can inform answers, but actions still go through the planner, ToolRegistry, PermissionPolicy, approval queue, and audit log.",
            "- Shell/code, destructive changes, clipboard reads, personal data, reminders, external side effects, and computer control require explicit approval.",
            "",
            "Visible context counts:",
            f"- profile context: {'present' if profile else 'empty'}",
            f"- active preferences: {len(preferences)}",
            f"- recent memories sampled: {len(memories)}",
            f"- saved skills sampled: {len(skills)}",
            f"- recent session messages: {len(recent)}",
            f"- unreadable message row(s) hidden for safety: {unreadable_recent}",
            "",
            "Useful checks:",
            "- `chat context: <question>` previews what chat can use.",
            "- `recent conversation` shows the current stored conversation window.",
            "- `privacy report` and `safety status` show broader data and action boundaries.",
        ]
        return ToolResult(
            "chat_safety_report",
            True,
            "\n".join(lines),
            _read_only_metadata(
                has_profile=bool(profile),
                preferences=len(preferences),
                recent_memories=len(memories),
                saved_skills=len(skills),
                recent_messages=len(recent),
                readable_message_rows=len(recent),
                unreadable_message_rows=unreadable_recent,
            ),
        )

    def save_chat_context(args: dict[str, Any]) -> ToolResult:
        prompt = _short(args.get("prompt"), limit=MAX_CONVERSATION_TEXT_CHARS)
        limit = _bounded_limit(args.get("limit", 5), 5, MAX_CHAT_CONTEXT_LIMIT)
        with store.generated_report_publication_fence():
            body, metadata = build_chat_context(prompt, limit)
            path, _content_sha256, _source_revision = vault.write_chat_context_with_evidence(
                f"# Chat Context\n\n{body}",
                store_identity=store.get_store_identity(),
                source_payload={
                    "prompt": prompt,
                    "limit": limit,
                    "body": body,
                    "metadata": metadata,
                },
            )
        path_display = _safe_vault_path_display(path, vault)
        metadata = _write_note_metadata(**metadata, path=str(path), path_display=path_display)
        metadata["save_chat_context_handoff"] = {
            "prompt_length": len(prompt),
            "has_profile": bool(metadata.get("has_profile")),
            "active_preference_count": int(metadata.get("preferences") or 0),
            "relevant_memory_count": int(metadata.get("memories") or 0),
            "relevant_skill_count": int(metadata.get("skills") or 0),
            "recent_message_count": int(metadata.get("recent_messages") or 0),
            "path_display": path_display,
            "command": "save chat context",
            "writes_notes": True,
            "writes_files": True,
            "writes_memory": True,
            "calls_model": bool(metadata.get("calls_model")),
            "executes_tools": bool(metadata.get("executes_tools")),
            "queues_approval": bool(metadata.get("queues_approval")),
            "controls_computer": bool(metadata.get("controls_computer")),
        }
        return ToolResult("save_chat_context", True, f"Chat context saved: {path_display}\n\n{body}", metadata)

    def session_learning_preview(args: dict[str, Any]) -> ToolResult:
        target_session = _short(args.get("session_id") or session_id, limit=MAX_SESSION_ID_CHARS)
        limit = _bounded_limit(args.get("limit", 120), 120, 200)
        raw_rows = store.recent_messages(limit=limit, session_id=target_session)
        rows, unreadable_rows = _safe_message_payloads(raw_rows)
        if not rows:
            output = f"No readable messages found for session {target_session}."
            if unreadable_rows:
                output += f" {unreadable_rows} unreadable message row(s) hidden for safety."
            return ToolResult(
                "session_learning_preview",
                True,
                output,
                _read_only_metadata(
                    session_id=target_session,
                    messages=0,
                    readable_message_rows=0,
                    unreadable_message_rows=unreadable_rows,
                ),
            )

        user_messages = [row["content"].strip() for row in rows if row["role"] == "user" and row["content"].strip()]
        assistant_messages = [row["content"].strip() for row in rows if row["role"] == "assistant" and row["content"].strip()]
        memory_candidates: list[str] = []
        preference_candidates: list[str] = []
        task_candidates: list[str] = []
        skill_candidates: list[str] = []

        for message in user_messages:
            low = message.lower()
            if any(word in low for word in ("remember", "don't forget", "important", "i want jarvis", "i need jarvis")):
                memory_candidates.append(message)
            if any(word in low for word in ("prefer", "preference", "should be", "should feel", "tone", "style", "shorter", "concise", "safe")):
                preference_candidates.append(message)
            if any(word in low for word in ("continue", "build", "work on", "add", "fix", "test", "verify", "make jarvis")):
                task_candidates.append(message)
            if any(word in low for word in ("skill", "workflow", "again", "always", "when i", "from session")):
                skill_candidates.append(message)

        if assistant_messages:
            for message in assistant_messages[-8:]:
                low = message.lower()
                if any(word in low for word in ("added", "verified", "implemented", "smoke test", "all smoke tests passed")):
                    skill_candidates.append("Successful workflow signal: " + message)

        memory_candidates = _dedupe_short(memory_candidates, 8)
        preference_candidates = _dedupe_short(preference_candidates, 8)
        task_candidates = _dedupe_short(task_candidates, 8)
        skill_candidates = _dedupe_short(skill_candidates, 6)

        lines = [
            "Jarvis session learning preview:",
            "This is read-only. It suggests what might be worth turning into memory, preferences, tasks, or skills; it does not save memory, write notes, queue tasks, change preferences, call a model, execute tools, or queue approvals.",
            "",
            f"Session: {target_session}",
            f"Messages reviewed: {len(rows)} ({len(user_messages)} user, {len(assistant_messages)} assistant)",
            f"Unreadable message row(s) hidden for safety: {unreadable_rows}",
            "",
            "Candidate memories:",
        ]
        lines.extend(f"- {item}" for item in memory_candidates) if memory_candidates else lines.append("- None detected.")
        lines.append("")
        lines.append("Candidate preferences:")
        lines.extend(f"- {item}" for item in preference_candidates) if preference_candidates else lines.append("- None detected.")
        lines.append("")
        lines.append("Candidate tasks:")
        lines.extend(f"- {item}" for item in task_candidates) if task_candidates else lines.append("- None detected.")
        lines.append("")
        lines.append("Candidate skill/workflow signals:")
        lines.extend(f"- {item}" for item in skill_candidates) if skill_candidates else lines.append("- None detected.")
        lines.extend(
            [
                "",
                "Safe follow-up commands:",
                "- Save only stable facts with `remember that ...`.",
                "- Turn repeated style requests into `set preference ...`.",
                "- Capture real follow-up work with `add task ...`.",
                "- Draft reusable workflows with `draft skill from this session called <name>`.",
                "",
                "Boundary:",
                "- This preview is advisory. Personal data, external side effects, destructive changes, shell/code, and computer control remain approval-gated.",
            ]
        )
        return ToolResult(
            "session_learning_preview",
            True,
            "\n".join(lines),
            _read_only_metadata(
                session_id=target_session,
                messages=len(rows),
                user_messages=len(user_messages),
                assistant_messages=len(assistant_messages),
                readable_message_rows=len(rows),
                unreadable_message_rows=unreadable_rows,
                memory_candidates=len(memory_candidates),
                preference_candidates=len(preference_candidates),
                task_candidates=len(task_candidates),
                skill_candidates=len(skill_candidates),
            ),
        )

    return (
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
    )
