from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    LOCAL_READ_INPUT_RECOVERY_ACTION,
    declare_failure_guidance,
    declare_retryable_local_read_failure,
)
from jarvis_v2.agent.types import ToolResult
from jarvis_v2.memory.goal_projection import reconcile_goal_projection
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import GoalRecord, MemoryStore


MAX_GOAL_LIMIT = 200
MAX_GOAL_TITLE_CHARS = 160
MAX_GOAL_PURPOSE_CHARS = 600
MAX_GOAL_HORIZON_CHARS = 120
MAX_GOAL_STEP_CHARS = 600
MAX_GOAL_COMMAND_CHARS = 180
MAX_GOAL_ID_INPUT_CHARS = 80
MAX_SQLITE_ID = 9_223_372_036_854_775_807
LOCAL_PATH_RE = re.compile(
    r"(?:^|[\s('\"])(?:~[/\\]|\.\.?[/\\]|[A-Z]:[/\\]|\\\\[^\\/\s]+[/\\]|"
    r"/(?:Users|private|var(?:/folders)?|tmp|Volumes|home|opt|etc|usr|Library)"
    r"(?:[/\\]|$))[^\n\r;]*",
    re.IGNORECASE,
)
_MISSING_ROW_VALUE = object()
GOAL_INPUT_RECOVERY_ACTION = (
    "Correct the reported goal input, then retry through the normal policy."
)
GOAL_NOT_FOUND_RECOVERY_ACTION = (
    "Refresh the goal list, correct the missing identifier, then retry through the normal policy."
)


def _bounded_int(value: Any, default: int, low: int = 1, high: int = MAX_GOAL_LIMIT) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


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


def _short(value: Any, *, limit: int) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _short_raw(value: Any, *, limit: int) -> str:
    text = "" if value is None else str(value).strip()
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _short_metadata(value: Any, *, limit: int) -> str:
    text = _short_raw(value, limit=limit)
    utf8_safe = text.encode("utf-8", errors="replace").decode("utf-8")
    return LOCAL_PATH_RE.sub("<local-path>", utf8_safe)


def _row_value(row: Any, key: str, default: Any = _MISSING_ROW_VALUE) -> Any:
    try:
        return row[key]
    except Exception:
        return default


def _row_text(row: Any, key: str, default: str = "", *, limit: int = MAX_GOAL_PURPOSE_CHARS) -> str:
    value = _row_value(row, key)
    if value is _MISSING_ROW_VALUE:
        return default
    text = _short_metadata(value, limit=limit)
    return text if text else default


def _row_int(row: Any, key: str) -> int | None:
    value = _row_value(row, key)
    if value is _MISSING_ROW_VALUE or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "calls_model": False,
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


def _goal_write_metadata(**extra: Any) -> dict[str, Any]:
    metadata = _safe_metadata(**extra)
    metadata.update({"writes_files": True, "writes_memory": True, "writes_notes": True})
    return metadata


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _safe_goal_command(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.strip().split())
    if not text or LOCAL_PATH_RE.search(text):
        return ""
    if len(text) > MAX_GOAL_COMMAND_CHARS:
        return text[: MAX_GOAL_COMMAND_CHARS - 1].rstrip() + "…"
    return text


def _sanitize_goal_next_commands(raw_next: Any) -> tuple[dict[str, str] | list[str], list[str], int]:
    hidden = 0
    if isinstance(raw_next, dict):
        sanitized: dict[str, str] = {}
        for key, value in raw_next.items():
            command = _safe_goal_command(value)
            key_text = _short_raw(key, limit=80)
            if command:
                sanitized[key_text] = command
            elif value in (None, ""):
                sanitized[key_text] = ""
            elif value not in (None, ""):
                hidden += 1
        return sanitized, [command for command in sanitized.values() if command], hidden
    if isinstance(raw_next, str):
        command = _safe_goal_command(raw_next)
        return ([command] if command else []), ([command] if command else []), 0 if command or not raw_next.strip() else 1
    try:
        iterator = iter(raw_next or [])
    except TypeError:
        return [], [], 1 if raw_next is not None else 0
    commands: list[str] = []
    for value in iterator:
        command = _safe_goal_command(value)
        if command:
            commands.append(command)
        elif value not in (None, ""):
            hidden += 1
    return commands, commands, hidden


def _goal_contract(
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


def _goal_handoff_metadata(
    handoff_key: str,
    handoff: dict[str, Any],
    *,
    writes: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    _normalize_goal_handoff(handoff)
    prefix = handoff_key.removesuffix("_handoff")
    metadata = _goal_write_metadata(**extra) if writes else _safe_metadata(**extra)
    metadata.update(
        _goal_contract(
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


def _normalize_goal_handoff(handoff: dict[str, Any]) -> dict[str, Any]:
    raw_next = handoff.get("next_commands") or []
    sanitized_next, next_safe_commands, hidden_next_commands = _sanitize_goal_next_commands(raw_next)
    next_safe_command = next_safe_commands[0] if next_safe_commands else ""
    handoff["next_commands"] = sanitized_next
    handoff["handoff_ready"] = True
    handoff["next_safe_command"] = next_safe_command
    handoff["next_safe_commands"] = list(next_safe_commands)
    handoff["next_safe_command_count"] = len(next_safe_commands)
    handoff["hidden_next_command_count"] = hidden_next_commands
    return handoff


def _safe_vault_path_display(path: str | Path | None, vault: ObsidianVault) -> str:
    if path is None:
        return ""
    candidate = Path(path)
    try:
        return str(candidate.relative_to(vault.root_path))
    except ValueError:
        return _short_metadata(candidate, limit=160)


def _goal_projection_path_display(goal_id: int) -> str:
    return f"Projects/Goal {goal_id}.md"


def _positive_id(value: Any, label: str) -> tuple[int | None, str | None, dict[str, Any]]:
    if isinstance(value, bool):
        return None, f"{label} must be a number.", {"raw_id": str(value)}
    if isinstance(value, str) and len(value) > MAX_GOAL_ID_INPUT_CHARS:
        return None, f"{label} is too long.", {"raw_id": "<oversized-id>"}
    try:
        item_id = int(value)
    except (TypeError, ValueError):
        return None, f"{label} must be a number.", {"raw_id": _short_metadata(value, limit=80)}
    if item_id <= 0:
        return None, f"{label} must be a positive number.", {"raw_id": _short_metadata(value, limit=80)}
    if item_id > MAX_SQLITE_ID:
        return None, f"{label} is too large.", {"raw_id": _short_metadata(value, limit=80)}
    return item_id, None, {}


def _normalized_goal_create_fields(args: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _short(
            unicodedata.normalize("NFKC", str(args.get("title") or "")),
            limit=MAX_GOAL_TITLE_CHARS,
        ),
        _short(args.get("purpose"), limit=MAX_GOAL_PURPOSE_CHARS),
        _short(args.get("horizon"), limit=MAX_GOAL_HORIZON_CHARS),
    )


def _goal_create_text_is_utf8_safe(*values: str) -> bool:
    try:
        for value in values:
            value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _goal_text_has_forbidden_controls(*values: str) -> bool:
    return any(
        ord(character) < 32 or ord(character) == 127
        for value in values
        for character in value
    )


def _validated_goal_store_identity(store: MemoryStore) -> str:
    identity = store.get_store_identity()
    if type(identity) is not str or re.fullmatch(r"[0-9a-f]{32}", identity) is None:
        raise RuntimeError("Goal projection store identity is unavailable.")
    return identity


def create_goal_auto_mutation_operation_key(args: dict[str, Any]) -> dict[str, str]:
    title, _purpose, _horizon = _normalized_goal_create_fields(args)
    return {"title": title.casefold()}


def _goal_create_arg_types_valid(args: dict[str, Any]) -> bool:
    return all(
        key not in args or isinstance(args.get(key), str)
        for key in ("title", "purpose", "horizon")
    )


def create_goal_auto_mutation_preflight(args: dict[str, Any]) -> str | None:
    if not _goal_create_arg_types_valid(args):
        return "invalid_type"
    title, purpose, horizon = _normalized_goal_create_fields(args)
    if not title:
        return "missing_title"
    if LOCAL_PATH_RE.search(title):
        return "invalid_title"
    if not _goal_create_text_is_utf8_safe(title, purpose, horizon):
        return "invalid_unicode"
    if _goal_text_has_forbidden_controls(title, purpose, horizon):
        return "invalid_unicode"
    return None


def export_goal_auto_mutation_operation_key(args: dict[str, Any]) -> dict[str, int]:
    goal_id, _error, _metadata = _positive_id(args.get("goal_id"), "Goal id")
    return {"goal_id": goal_id or 0}


def add_goal_step_auto_mutation_operation_key(args: dict[str, Any]) -> dict[str, int]:
    goal_id, _error, _metadata = _positive_id(args.get("goal_id"), "Goal id")
    return {"goal_id": goal_id or 0}


def complete_goal_step_auto_mutation_operation_key(args: dict[str, Any]) -> dict[str, int]:
    step_id, _error, _metadata = _positive_id(args.get("step_id"), "Step id")
    return {"step_id": step_id or 0}


def set_goal_status_auto_mutation_operation_key(args: dict[str, Any]) -> dict[str, int]:
    goal_id, _error, _metadata = _positive_id(args.get("goal_id"), "Goal id")
    return {"goal_id": goal_id or 0}


def _goal_step_text_is_utf8_safe(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def make_export_goal_auto_mutation_preflight(store: MemoryStore):
    def preflight(args: dict[str, Any]) -> str | None:
        goal_id, _error, _metadata = _positive_id(args.get("goal_id"), "Goal id")
        if goal_id is None:
            return "bad_goal_id"
        if store.get_goal(goal_id) is None:
            return "missing_goal"
        return None

    return preflight


def make_add_goal_step_auto_mutation_preflight(store: MemoryStore):
    def preflight(args: dict[str, Any]) -> str | None:
        goal_id, _error, _metadata = _positive_id(args.get("goal_id"), "Goal id")
        if goal_id is None:
            return "bad_goal_id"
        body = _short(args.get("body"), limit=MAX_GOAL_STEP_CHARS)
        if not body:
            return "missing_body"
        if not _goal_step_text_is_utf8_safe(body):
            return "invalid_unicode"
        if store.get_goal(goal_id) is None:
            return "missing_goal"
        return None

    return preflight


def make_complete_goal_step_auto_mutation_preflight(store: MemoryStore):
    def preflight(args: dict[str, Any]) -> str | None:
        step_id, _error, _metadata = _positive_id(args.get("step_id"), "Step id")
        if step_id is None:
            return "bad_step_id"
        if store.get_goal_step(step_id) is None:
            return "missing_step"
        return None

    return preflight


def make_set_goal_status_auto_mutation_preflight(store: MemoryStore):
    def preflight(args: dict[str, Any]) -> str | None:
        goal_id, _error, _metadata = _positive_id(args.get("goal_id"), "Goal id")
        if goal_id is None:
            return "bad_goal_id"
        status = str(args.get("status") or "").strip().lower()
        if status not in {"active", "paused", "done", "dropped"}:
            return "bad_status"
        if store.get_goal(goal_id) is None:
            return "missing_goal"
        return None

    return preflight


def _goal_summary(goal, steps) -> str:
    lines = [
        f"#{goal['id']} {goal['title']} [{goal['status']}]",
        f"Purpose: {goal['purpose'] or 'not set'}",
        f"Horizon: {goal['horizon'] or 'not set'}",
        "Steps:",
    ]
    if not steps:
        lines.append("- No steps yet.")
    for step in steps:
        mark = "x" if step["status"] == "done" else " "
        lines.append(f"- [{mark}] step #{step['id']}: {step['body']}")
    return "\n".join(lines)


def _goal_record_payload(goal: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    open_steps = [step for step in steps if step["status"] != "done"]
    return {
        "id": int(goal["id"]),
        "title": str(goal["title"]),
        "status": str(goal["status"]),
        "purpose": str(goal["purpose"] or ""),
        "horizon": str(goal["horizon"] or ""),
        "step_count": len(steps),
        "open_step_count": len(open_steps),
        "done_step_count": len(steps) - len(open_steps),
        "next_step_id": int(open_steps[0]["id"]) if open_steps else None,
        "next_step": str(open_steps[0]["body"]) if open_steps else "",
    }


def _goal_step_payloads(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": int(step["id"]),
            "goal_id": int(step["goal_id"]),
            "status": str(step["status"]),
            "body": str(step["body"]),
            "body_chars": len(str(step["body"])),
        }
        for step in steps
    ]


def _safe_goal_payload(goal: Any, store: MemoryStore) -> tuple[dict[str, Any] | None, int]:
    goal_id = _row_int(goal, "id")
    if goal_id is None:
        return None, 0
    try:
        raw_steps = store.list_goal_steps(goal_id)
    except Exception:
        raw_steps = []
        unreadable_steps = 1
    else:
        unreadable_steps = 0

    steps: list[dict[str, Any]] = []
    for step in raw_steps:
        step_id = _row_int(step, "id")
        step_goal_id = _row_int(step, "goal_id")
        if step_id is None or step_goal_id is None:
            unreadable_steps += 1
            continue
        status = _row_text(step, "status", "open", limit=40)
        body = _row_text(step, "body", "unreadable step", limit=MAX_GOAL_STEP_CHARS)
        steps.append(
            {
                "id": step_id,
                "goal_id": step_goal_id,
                "status": status,
                "body": body,
            }
        )

    open_steps = [step for step in steps if step["status"] != "done"]
    title = _row_text(goal, "title", "Unreadable goal", limit=MAX_GOAL_TITLE_CHARS)
    status = _row_text(goal, "status", "unknown", limit=40)
    purpose = _row_text(goal, "purpose", "", limit=MAX_GOAL_PURPOSE_CHARS)
    horizon = _row_text(goal, "horizon", "", limit=MAX_GOAL_HORIZON_CHARS)
    return (
        {
            "id": goal_id,
            "title": title,
            "status": status,
            "purpose": purpose,
            "horizon": horizon,
            "step_count": len(steps),
            "open_step_count": len(open_steps),
            "done_step_count": len(steps) - len(open_steps),
            "next_step_id": int(open_steps[0]["id"]) if open_steps else None,
            "next_step": str(open_steps[0]["body"]) if open_steps else "",
        },
        unreadable_steps,
    )


def _safe_goal_payloads(goals: list[Any], store: MemoryStore) -> tuple[list[dict[str, Any]], int, int]:
    payloads: list[dict[str, Any]] = []
    unreadable_goals = 0
    unreadable_steps = 0
    for goal in goals:
        payload, skipped_steps = _safe_goal_payload(goal, store)
        unreadable_steps += skipped_steps
        if payload is None:
            unreadable_goals += 1
        else:
            payloads.append(payload)
    return payloads, unreadable_goals, unreadable_steps


def _goal_export_handoff_payload(
    *,
    goal: dict[str, Any],
    steps: list[dict[str, Any]],
    path_display: str,
) -> dict[str, Any]:
    goal_id = int(goal["id"])
    return {
        "source": "export_goal",
        "goal_id": goal_id,
        "goal": _goal_record_payload(goal, steps),
        "steps": _goal_step_payloads(steps),
        "path_display": path_display,
        "next_commands": {
            "show_goal": f"goal {goal_id} status",
            "list_goals": "list goals",
            "next_actions": "next actions",
            "export_again": f"export goal {goal_id} to obsidian",
            "mark_done": f"goal {goal_id} done",
        },
        "boundaries": {
            "exports_goal": True,
            "writes_files": True,
            "writes_memory": False,
            "writes_notes": True,
            "adds_goal": False,
            "adds_step": False,
            "changes_goal_status": False,
            "completes_goal": False,
            "completes_step": False,
            "calls_model": False,
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
            "controls_computer": False,
        },
        **_goal_contract(state_changed=True, changed=["goal_export"], content_in_handoff=True),
    }


def _goal_status_handoff_payload(
    *,
    goal: dict[str, Any],
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    goal_id = int(goal["id"])
    next_step_id = _goal_record_payload(goal, steps)["next_step_id"]
    return {
        "source": "goal_status",
        "goal_id": goal_id,
        "goal": _goal_record_payload(goal, steps),
        "steps": _goal_step_payloads(steps),
        "next_commands": {
            "export_goal": f"export goal {goal_id} to obsidian",
            "list_goals": "list goals",
            "next_actions": "next actions",
            "add_step": f"add step to goal {goal_id}: <next step>",
            "complete_next_step": f"complete goal step {next_step_id}" if next_step_id is not None else "",
            "mark_done": f"goal {goal_id} done",
        },
        "boundaries": {
            "reads_goal": True,
            "writes_files": False,
            "writes_memory": False,
            "writes_notes": False,
            "exports_goal": False,
            "adds_goal": False,
            "adds_step": False,
            "changes_goal_status": False,
            "completes_goal": False,
            "completes_step": False,
            "calls_model": False,
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
            "controls_computer": False,
        },
        **_goal_contract(content_in_handoff=True),
    }


def _goal_list_handoff_payload(
    *,
    source: str,
    status: str,
    limit: int,
    goals: list[dict[str, Any]],
    unreadable_goal_rows: int = 0,
    unreadable_goal_step_rows: int = 0,
) -> dict[str, Any]:
    goal_ids = [goal["id"] for goal in goals]
    first_goal_id = goal_ids[0] if goal_ids else None
    return {
        "source": source,
        "status": status,
        "limit": limit,
        "goal_count": len(goals),
        "readable_goal_rows": len(goals),
        "unreadable_goal_rows": unreadable_goal_rows,
        "unreadable_goal_step_rows": unreadable_goal_step_rows,
        "goal_ids": goal_ids,
        "first_goal_id": first_goal_id,
        "goals": goals,
        "next_commands": {
            "show_first_goal": f"goal {first_goal_id} status" if first_goal_id is not None else "",
            "export_first_goal": f"export goal {first_goal_id} to obsidian" if first_goal_id is not None else "",
            "next_actions": "next actions",
            "create_goal": "create goal <title> because <purpose>",
            "list_active": "list goals",
        },
        "boundaries": {
            "reads_goals": True,
            "writes_files": False,
            "writes_memory": False,
            "writes_notes": False,
            "exports_goal": False,
            "adds_goal": False,
            "adds_step": False,
            "changes_goal_status": False,
            "completes_goal": False,
            "completes_step": False,
            "calls_model": False,
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
            "controls_computer": False,
        },
        **_goal_contract(content_in_handoff=bool(goals)),
    }


def _goal_next_actions_handoff_payload(
    *,
    source: str,
    limit: int,
    actions: list[dict[str, Any]],
    readable_goal_rows: int = 0,
    unreadable_goal_rows: int = 0,
    unreadable_goal_step_rows: int = 0,
) -> dict[str, Any]:
    first_action = actions[0] if actions else None
    first_goal_id = first_action["goal_id"] if first_action else None
    first_step_id = first_action["step_id"] if first_action else None
    return {
        "source": source,
        "limit": limit,
        "action_count": len(actions),
        "readable_goal_rows": readable_goal_rows,
        "unreadable_goal_rows": unreadable_goal_rows,
        "unreadable_goal_step_rows": unreadable_goal_step_rows,
        "goal_ids": [action["goal_id"] for action in actions],
        "first_goal_id": first_goal_id,
        "first_step_id": first_step_id,
        "actions": actions,
        "next_commands": {
            "show_first_goal": f"goal {first_goal_id} status" if first_goal_id is not None else "",
            "complete_first_step": f"complete goal step {first_step_id}" if first_step_id is not None else "",
            "add_step_to_first_goal": f"add step to goal {first_goal_id}: <next step>" if first_goal_id is not None else "",
            "list_goals": "list goals",
            "create_goal": "create goal <title> because <purpose>",
        },
        "boundaries": {
            "reads_goals": True,
            "reads_steps": True,
            "writes_files": False,
            "writes_memory": False,
            "writes_notes": False,
            "exports_goal": False,
            "adds_goal": False,
            "adds_step": False,
            "changes_goal_status": False,
            "completes_goal": False,
            "completes_step": False,
            "calls_model": False,
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
            "controls_computer": False,
        },
        **_goal_contract(content_in_handoff=bool(actions)),
    }


def _goal_mutation_handoff_payload(
    *,
    source: str,
    goal: dict[str, Any],
    steps: list[dict[str, Any]],
    path_display: str,
    mutation: str,
    changed: list[str],
    step_id: int | None = None,
    state_changed: bool = True,
    writes_projection: bool | None = None,
) -> dict[str, Any]:
    goal_id = int(goal["id"])
    projection_written = state_changed if writes_projection is None else writes_projection
    return {
        "source": source,
        "mutation": mutation,
        "goal_id": goal_id,
        "step_id": step_id,
        "changed": list(changed),
        "goal": _goal_record_payload(goal, steps),
        "steps": _goal_step_payloads(steps),
        "path_display": path_display,
        "next_commands": {
            "show_goal": f"goal {goal_id} status",
            "export_goal": f"export goal {goal_id} to obsidian",
            "list_goals": "list goals",
            "next_actions": "next actions",
            "add_step": f"add step to goal {goal_id}: <next step>",
            "mark_done": f"goal {goal_id} done",
        },
        "boundaries": {
            "writes_files": projection_written,
            "writes_memory": projection_written,
            "writes_notes": projection_written,
            "adds_goal": state_changed and mutation == "goal_create",
            "adds_step": state_changed and mutation == "step_create",
            "changes_goal_status": state_changed and mutation == "status_update",
            "completes_goal": state_changed and mutation == "status_update" and str(goal["status"]) == "done",
            "completes_step": state_changed and mutation == "step_completion",
            "exports_goal": False,
            "calls_model": False,
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
            "controls_computer": False,
        },
        **_goal_contract(state_changed=state_changed, changed=changed, content_in_handoff=True),
    }


def _goal_refusal_handoff_payload(
    *,
    source: str,
    reason: str,
    mutation: str,
    goal_id: int | None = None,
    step_id: int | None = None,
    raw_field: str | None = None,
    raw_value: Any = None,
    next_command: str = "",
) -> dict[str, Any]:
    sanitized_raw = _short_metadata(raw_value, limit=80) if raw_field else None
    handoff = {
        "source": source,
        "reason": reason,
        "mutation": mutation,
        "goal_id": goal_id,
        "step_id": step_id,
        "changed": [],
        "refused": True,
        "next_commands": {
            "retry": next_command,
            "list_goals": "list goals",
            "next_actions": "next actions",
        },
        "boundaries": {
            "writes_files": False,
            "writes_memory": False,
            "writes_notes": False,
            "adds_goal": False,
            "adds_step": False,
            "changes_goal_status": False,
            "completes_goal": False,
            "completes_step": False,
            "exports_goal": False,
            "calls_model": False,
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
            "controls_computer": False,
        },
        **_goal_contract(content_in_handoff=raw_field is not None),
    }
    if raw_field:
        handoff["raw_field"] = raw_field
        handoff["raw_value"] = sanitized_raw
    return handoff


def _goal_refusal_metadata(
    *,
    source: str,
    reason: str,
    mutation: str,
    goal_id: int | None = None,
    step_id: int | None = None,
    raw_field: str | None = None,
    raw_value: Any = None,
    next_command: str = "",
    **extra: Any,
) -> dict[str, Any]:
    handoff = _goal_refusal_handoff_payload(
        source=source,
        reason=reason,
        mutation=mutation,
        goal_id=goal_id,
        step_id=step_id,
        raw_field=raw_field,
        raw_value=raw_value,
        next_command=next_command,
    )
    metadata = _goal_handoff_metadata(
        "goal_refusal_handoff",
        handoff,
        reason=reason,
        goal_id=goal_id,
        step_id=step_id,
        **extra,
    )
    metadata["goal_mutation_handoff_ready"] = False
    metadata["writes_database"] = False
    metadata["auto_mutation_effects_started"] = False
    metadata["external_side_effect"] = False
    if raw_field:
        metadata[f"raw_{raw_field}"] = _short_metadata(raw_value, limit=80)
    return metadata


def _goal_refusal_result(
    tool_name: str,
    output: str,
    metadata: dict[str, Any],
    *,
    action: str = GOAL_INPUT_RECOVERY_ACTION,
) -> ToolResult:
    """Return a canonical, known-no-change goal refusal.

    This helper is limited to validation or lookup failures that happen before
    a goal database/projection mutation starts. A corrected request must still
    pass through normal policy.
    """

    public_output = str(output).strip()
    next_command = _safe_goal_command(metadata.get("next_safe_command"))
    commands: tuple[str, ...] = ()
    if next_command:
        commands = (next_command,)
        if next_command not in public_output:
            public_output = f"{public_output} Run `{next_command}`."
    if action not in public_output:
        public_output = f"{public_output} {action}"
    result_metadata = dict(metadata)
    result_metadata.update(
        {
            "outcome_known": True,
            "outcome_unknown": False,
            "execution_outcome_unknown": False,
            "side_effect_possible": False,
            "retry_safe": True,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
            "state_changed": False,
            "auto_mutation_effects_started": False,
            "writes_database": False,
            "external_side_effect": False,
        }
    )
    return ToolResult(
        tool_name,
        False,
        public_output,
        declare_failure_guidance(
            result_metadata,
            output=public_output,
            action=action,
            commands=commands,
        ),
    )


def _missing_goal_result(
    *,
    tool_name: str,
    goal_id: int,
    mutation: str,
    retry_command: str,
    **extra: Any,
) -> ToolResult:
    verify_command = "goal <correct goal id> status"
    recovery_commands = ["list goals", verify_command]
    if retry_command and retry_command not in recovery_commands:
        recovery_commands.append(retry_command)
    output = (
        f"No goal found for #{goal_id}. Run `list goals` to refresh goal IDs, then "
        f"`{verify_command}` to verify the correct record."
    )
    if retry_command != verify_command:
        output += f" After that, run `{retry_command}` through the normal policy."
    else:
        output += " Use that command through the normal policy."
    metadata = _goal_refusal_metadata(
        source=tool_name,
        reason="missing_goal",
        mutation=mutation,
        goal_id=goal_id,
        next_command="list goals",
        **extra,
    )
    metadata.update(
        {
            "next_command": "list goals",
            "recovery_commands": recovery_commands,
            "retry_requires_goal_refresh": True,
            "retry_requires_corrected_id": True,
            "recovery_commands_require_normal_policy": True,
            "authorizes_retry": False,
            "authorizes_goal_mutation": False,
        }
    )
    output = f"{output} {GOAL_NOT_FOUND_RECOVERY_ACTION}"
    metadata.update(
        {
            "outcome_known": True,
            "outcome_unknown": False,
            "execution_outcome_unknown": False,
            "side_effect_possible": False,
            "retry_safe": True,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
            "state_changed": False,
            "auto_mutation_effects_started": False,
            "writes_database": False,
            "external_side_effect": False,
        }
    )
    return ToolResult(
        tool_name,
        False,
        output,
        declare_failure_guidance(
            metadata,
            output=output,
            action=GOAL_NOT_FOUND_RECOVERY_ACTION,
        ),
    )


def _missing_goal_step_result(*, step_id: int) -> ToolResult:
    retry_command = "complete goal step <correct step id>"
    output = (
        f"No goal step found for #{step_id}. Run `next actions` to refresh goal-step IDs, then "
        f"run `{retry_command}` through the normal policy."
    )
    metadata = _goal_refusal_metadata(
        source="complete_goal_step",
        reason="missing_step",
        mutation="step_completion",
        step_id=step_id,
        next_command="next actions",
    )
    metadata.update(
        {
            "next_command": "next actions",
            "recovery_commands": ["next actions", retry_command],
            "retry_requires_goal_step_refresh": True,
            "retry_requires_corrected_id": True,
            "recovery_commands_require_normal_policy": True,
            "authorizes_retry": False,
            "authorizes_goal_mutation": False,
        }
    )
    output = f"{output} {GOAL_NOT_FOUND_RECOVERY_ACTION}"
    metadata.update(
        {
            "outcome_known": True,
            "outcome_unknown": False,
            "execution_outcome_unknown": False,
            "side_effect_possible": False,
            "retry_safe": True,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
            "state_changed": False,
            "auto_mutation_effects_started": False,
            "writes_database": False,
            "external_side_effect": False,
        }
    )
    return ToolResult(
        "complete_goal_step",
        False,
        output,
        declare_failure_guidance(
            metadata,
            output=output,
            action=GOAL_NOT_FOUND_RECOVERY_ACTION,
        ),
    )


def _goal_semantic_preflight_result(result: ToolResult, args: dict[str, Any]) -> ToolResult:
    result.metadata.update(
        {
            "failure_kind": "auto_mutation_semantic_preflight_rejected",
            "requires_confirmation": False,
            "requires_approval": False,
            "executed_handler": False,
            "handler_invoked": False,
            "planned_arg_keys": sorted(str(key)[:80] for key in args),
            "authorizes_retry": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "queues_approval": False,
            "state_changed": False,
            "auto_mutation_effects_started": False,
            "writes_files": False,
            "writes_database": False,
            "writes_memory": False,
            "writes_notes": False,
            "external_side_effect": False,
            "controls_computer": False,
        }
    )
    return result


def _create_goal_refusal_result(args: dict[str, Any], reason: str) -> ToolResult:
    raw_title = args.get("title")
    if reason == "invalid_type":
        output = "Goal title, purpose, and horizon must be text."
        raw_field = None
        raw_value = None
    elif reason == "invalid_title":
        output = "Goal title should be a project title, not a local file path."
        raw_field = "title"
        raw_value = raw_title
    elif reason == "invalid_unicode":
        output = "Goal title, purpose, and horizon must contain valid Unicode text."
        raw_field = None
        raw_value = None
    else:
        output = "Goal title is required."
        raw_field = "title"
        raw_value = raw_title
        reason = "missing_title"
    return _goal_refusal_result(
        "create_goal",
        output,
        _goal_refusal_metadata(
            source="create_goal",
            reason=reason,
            mutation="goal_create",
            raw_field=raw_field,
            raw_value=raw_value,
            next_command="create goal <title> because <purpose>",
        ),
    )


def create_goal_auto_mutation_preflight_result(
    args: dict[str, Any],
    reason: str,
) -> ToolResult:
    return _goal_semantic_preflight_result(
        _create_goal_refusal_result(args, reason),
        args,
    )


def add_goal_step_auto_mutation_preflight_result(
    args: dict[str, Any],
    reason: str,
) -> ToolResult:
    goal_id, error, error_metadata = _positive_id(args.get("goal_id"), "Goal id")
    body = _short(args.get("body"), limit=MAX_GOAL_STEP_CHARS)
    if reason == "missing_goal" and goal_id is not None:
        result = _missing_goal_result(
            tool_name="add_goal_step",
            goal_id=goal_id,
            mutation="step_create",
            retry_command="add step to goal <correct goal id>: <next step>",
        )
    elif reason == "missing_body" and goal_id is not None:
        result = _goal_refusal_result(
            "add_goal_step",
            "Step body is required.",
            _goal_refusal_metadata(
                source="add_goal_step",
                reason="missing_body",
                mutation="step_create",
                goal_id=goal_id,
                raw_field="body",
                raw_value=args.get("body"),
                next_command=f"add step to goal {goal_id}: <next step>",
            ),
        )
    elif reason == "invalid_unicode" and goal_id is not None:
        result = _goal_refusal_result(
            "add_goal_step",
            "Step body must contain valid Unicode text.",
            _goal_refusal_metadata(
                source="add_goal_step",
                reason="invalid_unicode",
                mutation="step_create",
                goal_id=goal_id,
                next_command=f"add step to goal {goal_id}: <next step>",
            ),
        )
    else:
        result = _goal_refusal_result(
            "add_goal_step",
            error or "Goal id is required.",
            _goal_refusal_metadata(
                source="add_goal_step",
                reason="bad_goal_id",
                mutation="step_create",
                raw_field="id",
                raw_value=error_metadata.get("raw_id"),
                next_command="add step to goal <goal id>: <next step>",
            ),
        )
    return _goal_semantic_preflight_result(result, args)


def complete_goal_step_auto_mutation_preflight_result(
    args: dict[str, Any],
    reason: str,
) -> ToolResult:
    step_id, error, error_metadata = _positive_id(args.get("step_id"), "Step id")
    if reason == "missing_step" and step_id is not None:
        result = _missing_goal_step_result(step_id=step_id)
    else:
        result = _goal_refusal_result(
            "complete_goal_step",
            error or "Step id is required.",
            _goal_refusal_metadata(
                source="complete_goal_step",
                reason="bad_step_id",
                mutation="step_completion",
                raw_field="id",
                raw_value=error_metadata.get("raw_id"),
                next_command="next actions",
            ),
        )
    return _goal_semantic_preflight_result(result, args)


def set_goal_status_auto_mutation_preflight_result(
    args: dict[str, Any],
    reason: str,
) -> ToolResult:
    goal_id, error, error_metadata = _positive_id(args.get("goal_id"), "Goal id")
    raw_status = args.get("status")
    status = str(raw_status or "").strip().lower()
    status_metadata = _short_metadata(raw_status, limit=80)
    if reason == "missing_goal" and goal_id is not None:
        result = _missing_goal_result(
            tool_name="set_goal_status",
            goal_id=goal_id,
            mutation="status_update",
            retry_command="goal <correct goal id> <active|paused|done|dropped>",
            status=status,
        )
    elif reason == "bad_status" and goal_id is not None:
        result = _goal_refusal_result(
            "set_goal_status",
            "Status must be active, paused, done, or dropped.",
            _goal_refusal_metadata(
                source="set_goal_status",
                reason="bad_status",
                mutation="status_update",
                goal_id=goal_id,
                raw_field="status",
                raw_value=raw_status,
                next_command=f"goal {goal_id} <active|paused|done|dropped>",
                status=status_metadata,
            ),
        )
    else:
        result = _goal_refusal_result(
            "set_goal_status",
            error or "Goal id is required.",
            _goal_refusal_metadata(
                source="set_goal_status",
                reason="bad_goal_id",
                mutation="status_update",
                raw_field="id",
                raw_value=error_metadata.get("raw_id"),
                next_command="list goals",
            ),
        )
    return _goal_semantic_preflight_result(result, args)


def export_goal_auto_mutation_preflight_result(
    args: dict[str, Any],
    reason: str,
) -> ToolResult:
    goal_id, error, error_metadata = _positive_id(args.get("goal_id"), "Goal id")
    if reason == "missing_goal" and goal_id is not None:
        result = _missing_goal_result(
            tool_name="export_goal",
            goal_id=goal_id,
            mutation="goal_export",
            retry_command="export goal <correct goal id> to obsidian",
        )
    else:
        result = _goal_refusal_result(
            "export_goal",
            error or "Goal id is required.",
            _goal_refusal_metadata(
                source="export_goal",
                reason="bad_goal_id",
                mutation="goal_export",
                raw_field="id",
                raw_value=error_metadata.get("raw_id"),
                next_command="list goals",
            ),
        )
    return _goal_semantic_preflight_result(result, args)


def _publish_goal_mirror(
    store: MemoryStore,
    vault: ObsidianVault,
    goal_id: int,
    *,
    store_identity: str,
) -> tuple[Any, list[Any], Path, str, str] | None:
    """Publish the current durable goal target and return its verified evidence."""
    for _attempt in range(3):
        try:
            target = store.ensure_current_goal_projection_job(goal_id)
        except ValueError:
            return None
        if target is None:
            return None
        outcome = reconcile_goal_projection(store, vault, goal_id, target)
        if outcome.status not in {"completed", "superseded"}:
            raise RuntimeError("Goal projection could not be reconciled safely.")
        snapshot = store.get_completed_goal_projection_snapshot(target)
        if snapshot is None:
            continue
        job, goal, steps = snapshot
        if type(job["content_digest"]) is not str or type(job["source_digest"]) is not str:
            raise RuntimeError("Goal projection completion evidence is unavailable.")
        path_display = _goal_projection_path_display(goal_id)
        return (
            goal,
            list(steps),
            vault.root_path / path_display,
            str(job["content_digest"]),
            str(job["source_digest"]),
        )
    raise RuntimeError("Goal projection changed during completion evidence capture.")


def make_goal_tools(store: MemoryStore, vault: ObsidianVault):
    def create_goal(args: dict[str, Any]) -> ToolResult:
        title, purpose, horizon = _normalized_goal_create_fields(args)
        refusal_reason = create_goal_auto_mutation_preflight(args)
        if refusal_reason is not None:
            return _create_goal_refusal_result(args, refusal_reason)
        store_identity = _validated_goal_store_identity(store)
        goal_id = store.create_goal(GoalRecord(title=title, purpose=purpose, horizon=horizon))
        published = _publish_goal_mirror(
            store,
            vault,
            goal_id,
            store_identity=store_identity,
        )
        if published is None:
            raise RuntimeError("Created goal was unavailable for canonical publication.")
        goal, steps, path, _content_sha256, _source_revision = published
        path_display = _goal_projection_path_display(goal_id)
        return ToolResult(
            "create_goal",
            True,
            f"Created goal #{goal_id}: {title}",
            _goal_handoff_metadata(
                "goal_mutation_handoff",
                _goal_mutation_handoff_payload(
                    source="create_goal",
                    goal=goal,
                    steps=steps,
                    path_display=path_display,
                    mutation="goal_create",
                    changed=["goal"],
                ),
                writes=True,
                goal_id=goal_id,
                path_display=path_display,
                title_chars=len(title),
                purpose_chars=len(purpose),
                horizon=horizon,
            ),
        )

    def list_goals(args: dict[str, Any]) -> ToolResult:
        raw_status = args.get("status")
        status = _short(raw_status, limit=40) or None
        if status is not None and status not in {"active", "paused", "done", "dropped"}:
            failure_output = (
                "Goal status must be active, paused, done, or dropped. "
                f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "list_goals",
                False,
                failure_output,
                declare_retryable_local_read_failure(
                    _goal_refusal_metadata(
                        source="list_goals",
                        reason="bad_status",
                        mutation="goal_list",
                        raw_field="status",
                        raw_value=raw_status,
                        next_command="list goals",
                        status=status,
                    ),
                    output=failure_output,
                    action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                ),
            )
        limit = _bounded_int(args.get("limit"), 25)
        rows = store.list_goals(status=status, limit=limit)
        goal_payloads, unreadable_goal_rows, unreadable_step_rows = _safe_goal_payloads(rows, store)
        if not rows or (not goal_payloads and not unreadable_goal_rows):
            label = f" with status '{status}'" if status else ""
            status_label = status or "all"
            return ToolResult(
                "list_goals",
                True,
                f"No goals{label}. Count: 0.",
                _goal_handoff_metadata(
                    "goal_list_handoff",
                    _goal_list_handoff_payload(
                        source="list_goals",
                        status=status_label,
                        limit=limit,
                        goals=[],
                        unreadable_goal_rows=unreadable_goal_rows,
                        unreadable_goal_step_rows=unreadable_step_rows,
                    ),
                    count=0,
                    status=status_label,
                    readable_goal_rows=0,
                    unreadable_goal_rows=unreadable_goal_rows,
                    unreadable_goal_step_rows=unreadable_step_rows,
                    **_raw_limit_metadata(args.get("limit"), limit=limit),
                ),
            )
        if not goal_payloads:
            status_label = status or "all"
            return ToolResult(
                "list_goals",
                True,
                f"No readable goals. Count: 0 readable. {unreadable_goal_rows} unreadable goal row(s) hidden for safety.",
                _goal_handoff_metadata(
                    "goal_list_handoff",
                    _goal_list_handoff_payload(
                        source="list_goals",
                        status=status_label,
                        limit=limit,
                        goals=[],
                        unreadable_goal_rows=unreadable_goal_rows,
                        unreadable_goal_step_rows=unreadable_step_rows,
                    ),
                    count=0,
                    status=status_label,
                    readable_goal_rows=0,
                    unreadable_goal_rows=unreadable_goal_rows,
                    unreadable_goal_step_rows=unreadable_step_rows,
                    **_raw_limit_metadata(args.get("limit"), limit=limit),
                ),
            )
        lines = [
            f"- #{goal['id']} {goal['title']} [{goal['status']}]: {goal['purpose'][:120] or 'no purpose captured'}"
            for goal in goal_payloads
        ]
        if unreadable_goal_rows:
            lines.append(f"- {unreadable_goal_rows} unreadable goal row(s) hidden for safety.")
        if unreadable_step_rows:
            lines.append(f"- {unreadable_step_rows} unreadable goal step row(s) hidden for safety.")
        count_line = f"Count: {len(goal_payloads)}"
        if unreadable_goal_rows:
            count_line += f" readable ({unreadable_goal_rows} hidden)"
        status_label = status or "all"
        return ToolResult(
            "list_goals",
            True,
            "Goals:\n" + count_line + "\n" + "\n".join(lines),
            _goal_handoff_metadata(
                "goal_list_handoff",
                _goal_list_handoff_payload(
                    source="list_goals",
                    status=status_label,
                    limit=limit,
                    goals=goal_payloads,
                    unreadable_goal_rows=unreadable_goal_rows,
                    unreadable_goal_step_rows=unreadable_step_rows,
                ),
                count=len(goal_payloads),
                status=status_label,
                readable_goal_rows=len(goal_payloads),
                unreadable_goal_rows=unreadable_goal_rows,
                unreadable_goal_step_rows=unreadable_step_rows,
                **_raw_limit_metadata(args.get("limit"), limit=limit),
            ),
        )

    def goal_status(args: dict[str, Any]) -> ToolResult:
        goal_id, error, error_metadata = _positive_id(args.get("goal_id"), "Goal id")
        if goal_id is None:
            failure_output = (
                f"{error or 'Goal id is required.'} "
                f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "goal_status",
                False,
                failure_output,
                declare_retryable_local_read_failure(
                    _goal_refusal_metadata(
                        source="goal_status",
                        reason="bad_goal_id",
                        mutation="goal_read",
                        raw_field="id",
                        raw_value=error_metadata.get("raw_id"),
                        next_command="list goals",
                    ),
                    output=failure_output,
                    action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                ),
            )
        goal = store.get_goal(goal_id)
        if not goal:
            return _missing_goal_result(
                tool_name="goal_status",
                goal_id=goal_id,
                mutation="goal_read",
                retry_command="goal <correct goal id> status",
            )
        steps = store.list_goal_steps(goal_id)
        return ToolResult(
            "goal_status",
            True,
            _goal_summary(goal, steps),
            _goal_handoff_metadata(
                "goal_status_handoff",
                _goal_status_handoff_payload(goal=goal, steps=steps),
                goal_id=goal_id,
                steps=len(steps),
                open_steps=len([step for step in steps if step["status"] != "done"]),
                status=goal["status"],
            ),
        )

    def add_goal_step(args: dict[str, Any]) -> ToolResult:
        goal_id, error, error_metadata = _positive_id(args.get("goal_id"), "Goal id")
        body = _short(args.get("body"), limit=MAX_GOAL_STEP_CHARS)
        if goal_id is None:
            return _goal_refusal_result(
                "add_goal_step",
                error or "Goal id is required.",
                _goal_refusal_metadata(
                    source="add_goal_step",
                    reason="bad_goal_id",
                    mutation="step_create",
                    raw_field="id",
                    raw_value=error_metadata.get("raw_id"),
                    next_command="add step to goal <goal id>: <next step>",
                ),
            )
        if not body:
            return _goal_refusal_result(
                "add_goal_step",
                "Step body is required.",
                _goal_refusal_metadata(
                    source="add_goal_step",
                    reason="missing_body",
                    mutation="step_create",
                    goal_id=goal_id,
                    raw_field="body",
                    raw_value=args.get("body"),
                    next_command=f"add step to goal {goal_id}: <next step>",
                ),
            )
        if not _goal_step_text_is_utf8_safe(body):
            return _goal_refusal_result(
                "add_goal_step",
                "Step body must contain valid Unicode text.",
                _goal_refusal_metadata(
                    source="add_goal_step",
                    reason="invalid_unicode",
                    mutation="step_create",
                    goal_id=goal_id,
                    next_command=f"add step to goal {goal_id}: <next step>",
                ),
            )
        goal = store.get_goal(goal_id)
        if not goal:
            return _missing_goal_result(
                tool_name="add_goal_step",
                goal_id=goal_id,
                mutation="step_create",
                retry_command="add step to goal <correct goal id>: <next step>",
                auto_mutation_effects_started=False,
                writes_database=False,
            )
        store_identity = _validated_goal_store_identity(store)
        step_id = store.add_goal_step(goal_id, body)
        if step_id is None:
            return _missing_goal_result(
                tool_name="add_goal_step",
                goal_id=goal_id,
                mutation="step_create",
                retry_command="add step to goal <correct goal id>: <next step>",
                auto_mutation_effects_started=False,
                writes_database=False,
            )
        published = _publish_goal_mirror(
            store,
            vault,
            goal_id,
            store_identity=store_identity,
        )
        if published is None:
            raise RuntimeError("Updated goal was unavailable for canonical publication.")
        goal, steps, path, _content_sha256, _source_revision = published
        path_display = _goal_projection_path_display(goal_id)
        return ToolResult(
            "add_goal_step",
            True,
            f"Added step #{step_id} to goal #{goal_id}: {body}",
            _goal_handoff_metadata(
                "goal_mutation_handoff",
                _goal_mutation_handoff_payload(
                    source="add_goal_step",
                    goal=goal,
                    steps=steps,
                    path_display=path_display,
                    mutation="step_create",
                    changed=["steps"],
                    step_id=step_id,
                ),
                writes=True,
                goal_id=goal_id,
                step_id=step_id,
                path_display=path_display,
                body_chars=len(body),
            ),
        )

    def complete_goal_step(args: dict[str, Any]) -> ToolResult:
        step_id, error, error_metadata = _positive_id(args.get("step_id"), "Step id")
        if step_id is None:
            return _goal_refusal_result(
                "complete_goal_step",
                error or "Step id is required.",
                _goal_refusal_metadata(
                    source="complete_goal_step",
                    reason="bad_step_id",
                    mutation="step_completion",
                    raw_field="id",
                    raw_value=error_metadata.get("raw_id"),
                    next_command="next actions",
                ),
            )
        store_identity = _validated_goal_store_identity(store)
        mutation_result = store.complete_goal_step_result(step_id)
        step = mutation_result.row
        if not mutation_result.found or step is None:
            result = _missing_goal_step_result(step_id=step_id)
            result.metadata.update(
                {"auto_mutation_effects_started": False, "writes_database": False}
            )
            return result
        if not mutation_result.changed:
            published = _publish_goal_mirror(
                store,
                vault,
                int(step["goal_id"]),
                store_identity=store_identity,
            )
            if published is None:
                raise RuntimeError("Completed goal step lost its parent goal.")
            goal, steps, path, _content_sha256, _source_revision = published
            path_display = _goal_projection_path_display(int(step["goal_id"]))
            return ToolResult(
                "complete_goal_step",
                True,
                f"Step #{step_id} for goal #{step['goal_id']} was already complete.",
                _goal_handoff_metadata(
                    "goal_mutation_handoff",
                    _goal_mutation_handoff_payload(
                        source="complete_goal_step",
                        goal=goal,
                        steps=steps,
                        path_display=path_display,
                        mutation="step_completion",
                        changed=[],
                        step_id=step_id,
                        state_changed=False,
                        writes_projection=True,
                    ),
                    writes=True,
                    writes_database=False,
                    goal_id=step["goal_id"],
                    step_id=step_id,
                    path_display=path_display,
                ),
            )
        published = _publish_goal_mirror(
            store,
            vault,
            int(step["goal_id"]),
            store_identity=store_identity,
        )
        if published is None:
            raise RuntimeError("Updated goal was unavailable for canonical publication.")
        goal, steps, path, _content_sha256, _source_revision = published
        path_display = _goal_projection_path_display(int(step["goal_id"]))
        return ToolResult(
            "complete_goal_step",
            True,
            f"Completed step #{step_id} for goal #{step['goal_id']}.",
            _goal_handoff_metadata(
                "goal_mutation_handoff",
                _goal_mutation_handoff_payload(
                    source="complete_goal_step",
                    goal=goal,
                    steps=steps,
                    path_display=path_display,
                    mutation="step_completion",
                    changed=["steps"],
                    step_id=step_id,
                ),
                writes=True,
                goal_id=step["goal_id"],
                step_id=step_id,
                path_display=path_display,
            ),
        )

    def set_goal_status(args: dict[str, Any]) -> ToolResult:
        goal_id, error, error_metadata = _positive_id(args.get("goal_id"), "Goal id")
        raw_status = args.get("status")
        status = str(raw_status or "").strip().lower()
        status_metadata = _short_metadata(raw_status, limit=80)
        if goal_id is None:
            return _goal_refusal_result(
                "set_goal_status",
                error or "Goal id is required.",
                _goal_refusal_metadata(
                    source="set_goal_status",
                    reason="bad_goal_id",
                    mutation="status_update",
                    raw_field="id",
                    raw_value=error_metadata.get("raw_id"),
                    next_command="list goals",
                ),
            )
        if status not in {"active", "paused", "done", "dropped"}:
            return _goal_refusal_result(
                "set_goal_status",
                "Status must be active, paused, done, or dropped.",
                _goal_refusal_metadata(
                    source="set_goal_status",
                    reason="bad_status",
                    mutation="status_update",
                    goal_id=goal_id,
                    raw_field="status",
                    raw_value=raw_status,
                    next_command=f"goal {goal_id} <active|paused|done|dropped>",
                    status=status_metadata,
                ),
            )
        store_identity = _validated_goal_store_identity(store)
        mutation_result = store.set_goal_status_result(goal_id, status)
        if not mutation_result.found:
            return _missing_goal_result(
                tool_name="set_goal_status",
                goal_id=goal_id,
                mutation="status_update",
                retry_command="goal <correct goal id> <active|paused|done|dropped>",
                status=status,
                auto_mutation_effects_started=False,
                writes_database=False,
            )
        if not mutation_result.changed:
            published = _publish_goal_mirror(
                store,
                vault,
                goal_id,
                store_identity=store_identity,
            )
            if published is None:
                raise RuntimeError("Unchanged goal status lost its goal row.")
            goal, steps, path, _content_sha256, _source_revision = published
            path_display = _goal_projection_path_display(goal_id)
            return ToolResult(
                "set_goal_status",
                True,
                f"Goal #{goal_id} was already {status}.",
                _goal_handoff_metadata(
                    "goal_mutation_handoff",
                    _goal_mutation_handoff_payload(
                        source="set_goal_status",
                        goal=goal,
                        steps=steps,
                        path_display=path_display,
                        mutation="status_update",
                        changed=[],
                        state_changed=False,
                        writes_projection=True,
                    ),
                    writes=True,
                    writes_database=False,
                    goal_id=goal_id,
                    status=status,
                    path_display=path_display,
                ),
            )
        published = _publish_goal_mirror(
            store,
            vault,
            goal_id,
            store_identity=store_identity,
        )
        if published is None:
            raise RuntimeError("Updated goal was unavailable for canonical publication.")
        goal, steps, path, _content_sha256, _source_revision = published
        path_display = _goal_projection_path_display(goal_id)
        return ToolResult(
            "set_goal_status",
            True,
            f"Set goal #{goal_id} to {status}.",
            _goal_handoff_metadata(
                "goal_mutation_handoff",
                _goal_mutation_handoff_payload(
                    source="set_goal_status",
                    goal=goal,
                    steps=steps,
                    path_display=path_display,
                    mutation="status_update",
                    changed=["status"],
                ),
                writes=True,
                goal_id=goal_id,
                status=status,
                path_display=path_display,
            ),
        )

    def export_goal(args: dict[str, Any]) -> ToolResult:
        goal_id, error, error_metadata = _positive_id(args.get("goal_id"), "Goal id")
        if goal_id is None:
            return _goal_refusal_result(
                "export_goal",
                error or "Goal id is required.",
                _goal_refusal_metadata(
                    source="export_goal",
                    reason="bad_goal_id",
                    mutation="goal_export",
                    raw_field="id",
                    raw_value=error_metadata.get("raw_id"),
                    next_command="list goals",
                ),
            )
        store_identity = _validated_goal_store_identity(store)
        published = _publish_goal_mirror(
            store,
            vault,
            goal_id,
            store_identity=store_identity,
        )
        if published is None:
            return _missing_goal_result(
                tool_name="export_goal",
                goal_id=goal_id,
                mutation="goal_export",
                retry_command="export goal <correct goal id> to obsidian",
                auto_mutation_effects_started=False,
                writes_database=False,
            )
        goal, steps, path, content_sha256, source_revision = published
        path_display = _goal_projection_path_display(goal_id)
        return ToolResult(
            "export_goal",
            True,
            f"Exported goal #{goal_id} to {path_display}",
            _goal_handoff_metadata(
                "goal_export_handoff",
                _goal_export_handoff_payload(goal=goal, steps=steps, path_display=path_display),
                writes=False,
                goal_id=goal_id,
                path_display=path_display,
                content_sha256=content_sha256,
                source_revision=source_revision,
                steps=len(steps),
                open_steps=len([step for step in steps if step["status"] != "done"]),
                writes_files=True,
                writes_database=True,
                writes_memory=False,
                writes_notes=True,
            ),
        )

    def next_actions(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 10)
        goals, unreadable_goal_rows, unreadable_step_rows = _safe_goal_payloads(
            store.list_goals(status="active", limit=100),
            store,
        )
        lines = []
        action_payloads = []
        for goal in goals:
            if goal["next_step_id"] is not None:
                action_body = str(goal["next_step"])
                step_id = int(goal["next_step_id"])
                lines.append(f"- Goal #{goal['id']} {goal['title']}: {action_body} (step #{step_id})")
            else:
                action_body = "define the next step"
                step_id = None
                lines.append(f"- Goal #{goal['id']} {goal['title']}: {action_body}")
            action_payloads.append(
                {
                    "goal_id": int(goal["id"]),
                    "goal_title": str(goal["title"]),
                    "goal_status": str(goal["status"]),
                    "step_id": step_id,
                    "action": action_body,
                    "has_open_step": step_id is not None,
                    "show_goal_command": f"goal {int(goal['id'])} status",
                    "complete_step_command": f"complete goal step {step_id}" if step_id is not None else "",
                    "add_step_command": f"add step to goal {int(goal['id'])}: <next step>",
                }
            )
            if len(lines) >= limit:
                break
        if not lines:
            unreadable_note = (
                f" {unreadable_goal_rows} unreadable active goal row(s) hidden for safety."
                if unreadable_goal_rows
                else ""
            )
            return ToolResult(
                "next_actions",
                True,
                f"No readable active goals. Create a goal or pick a focus for today.{unreadable_note}",
                _goal_handoff_metadata(
                    "goal_next_actions_handoff",
                    _goal_next_actions_handoff_payload(
                        source="next_actions",
                        limit=limit,
                        actions=[],
                        readable_goal_rows=len(goals),
                        unreadable_goal_rows=unreadable_goal_rows,
                        unreadable_goal_step_rows=unreadable_step_rows,
                    ),
                    count=0,
                    readable_goal_rows=len(goals),
                    unreadable_goal_rows=unreadable_goal_rows,
                    unreadable_goal_step_rows=unreadable_step_rows,
                    **_raw_limit_metadata(args.get("limit"), limit=limit),
                ),
            )
        if unreadable_goal_rows:
            lines.append(f"- {unreadable_goal_rows} unreadable active goal row(s) hidden for safety.")
        if unreadable_step_rows:
            lines.append(f"- {unreadable_step_rows} unreadable goal step row(s) hidden for safety.")
        return ToolResult(
            "next_actions",
            True,
            "Next actions:\n" + "\n".join(lines),
            _goal_handoff_metadata(
                "goal_next_actions_handoff",
                _goal_next_actions_handoff_payload(
                        source="next_actions",
                        limit=limit,
                        actions=action_payloads,
                        readable_goal_rows=len(goals),
                        unreadable_goal_rows=unreadable_goal_rows,
                        unreadable_goal_step_rows=unreadable_step_rows,
                    ),
                count=len(action_payloads),
                readable_goal_rows=len(goals),
                unreadable_goal_rows=unreadable_goal_rows,
                unreadable_goal_step_rows=unreadable_step_rows,
                **_raw_limit_metadata(args.get("limit"), limit=limit),
            ),
        )

    return create_goal, list_goals, goal_status, add_goal_step, complete_goal_step, set_goal_status, export_goal, next_actions
