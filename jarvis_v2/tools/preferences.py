from __future__ import annotations

import re
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    LOCAL_READ_INPUT_RECOVERY_ACTION,
    RESOURCE_NOT_FOUND_RECOVERY_ACTION,
    declare_failure_guidance,
    declare_resource_not_found_failure,
    declare_retryable_local_read_failure,
)
from jarvis_v2.agent.types import ToolResult
from jarvis_v2.memory.memory_projection import reconcile_memory_projection
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.preference_projection import reconcile_preference_projection
from jarvis_v2.memory.store import (
    MemoryStore,
    PreferenceRecord,
    preference_identity_part,
)


MAX_PREFERENCE_LIMIT = 200
MAX_KEY_CHARS = 120
MAX_VALUE_CHARS = 2000
MAX_CATEGORY_CHARS = 64
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")
PREFERENCE_INPUT_RECOVERY_ACTION = (
    "Correct the reported preference field, then submit a fresh request through the normal policy."
)


def preference_auto_mutation_preflight(args: dict[str, Any]) -> str | None:
    key = _short(args.get("key"), MAX_KEY_CHARS)
    value = _short(args.get("value"), MAX_VALUE_CHARS)
    if not key:
        return "missing_key"
    if LOCAL_PATH_RE.search(key):
        return "invalid_key"
    if not value:
        return "missing_value"
    return None


def preference_auto_mutation_operation_key(args: dict[str, Any]) -> dict[str, Any]:
    key = _short(args.get("key"), MAX_KEY_CHARS)
    category = _short(args.get("category"), MAX_CATEGORY_CHARS).lower() or "general"
    return {
        "category": preference_identity_part(category),
        "key": preference_identity_part(key),
    }


def _safe_vault_path_display(path: Any, vault: ObsidianVault) -> str:
    try:
        return str(path.relative_to(vault.root_path))
    except (AttributeError, ValueError):
        return _short_metadata(path, 160)


def _bounded_int(value: Any, default: int, low: int = 1, high: int = MAX_PREFERENCE_LIMIT) -> int:
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
        return {"limit": limit, "raw_limit": _short_metadata(value, 80)}
    return {"limit": limit}


def _short(value: Any, limit: int) -> str:
    try:
        text = str(value or "").strip()
    except Exception:
        text = ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _short_raw(value: Any, limit: int) -> str:
    try:
        text = "" if value is None else str(value).strip()
    except Exception:
        text = ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _short_metadata(value: Any, limit: int) -> str:
    text = _short_raw(value, limit)
    return LOCAL_PATH_RE.sub("<local-path>", text)


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "reads_private_data": False,
        "writes_files": False,
        "writes_memory": False,
        "writes_notes": False,
        "controls_computer": False,
        "queues_approval": False,
        "external_side_effect": False,
        "requires_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    metadata.update(extra)
    return metadata


def _preference_input_failure(
    tool_name: str,
    message: str,
    metadata: dict[str, Any],
) -> ToolResult:
    output = f"{message} {PREFERENCE_INPUT_RECOVERY_ACTION}"
    declared = dict(metadata)
    declared.update(
        {
            "outcome_known": True,
            "outcome_unknown": False,
            "execution_outcome_unknown": False,
            "side_effect_possible": False,
            "retry_safe": True,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
        }
    )
    return ToolResult(
        tool_name,
        False,
        output,
        declare_failure_guidance(
            declared,
            output=output,
            action=PREFERENCE_INPUT_RECOVERY_ACTION,
        ),
    )


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
    return _short_metadata(value, 2000)


def _row_positive_int(row: Any, key: str = "id") -> int | None:
    value = _row_value(row, key)
    if isinstance(value, bool) or value is _MISSING_ROW_VALUE or value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number > 0 else None


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _preference_contract(*, state_changed: bool = False, changed: list[str] | None = None, content_in_handoff: bool = False) -> dict[str, Any]:
    return {
        "ready_for_operator": True,
        "state_changed": state_changed,
        "changed": changed or [],
        "content_in_handoff": content_in_handoff,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _preference_handoff_metadata(handoff_key: str, handoff: dict[str, Any], *, writes: bool = False, **extra: Any) -> dict[str, Any]:
    _normalize_preference_handoff(handoff)
    prefix = handoff_key.removesuffix("_handoff")
    base = _preference_write_metadata(**extra) if writes else _safe_metadata(**extra)
    base.update(
        _preference_contract(
            state_changed=_metadata_bool(handoff.get("state_changed")),
            changed=list(handoff.get("changed") or []),
            content_in_handoff=_metadata_bool(handoff.get("content_in_handoff")),
        )
    )
    base[handoff_key] = handoff
    base[f"{handoff_key}_ready"] = True
    base[f"{prefix}_handoff_ready"] = True
    base[f"{prefix}_ready_for_operator"] = True
    base[f"{prefix}_state_changed"] = _metadata_bool(handoff.get("state_changed"))
    base[f"{prefix}_changed"] = list(handoff.get("changed") or [])
    base[f"{prefix}_content_in_handoff"] = _metadata_bool(handoff.get("content_in_handoff"))
    base[f"{prefix}_next_safe_command"] = handoff["next_safe_command"]
    base[f"{prefix}_next_safe_commands"] = list(handoff["next_safe_commands"])
    base[f"{prefix}_next_safe_command_count"] = handoff["next_safe_command_count"]
    base[f"{prefix}_authorizes_execution"] = False
    base[f"{prefix}_authorizes_completion_claim"] = False
    base[f"{prefix}_approval_granted"] = False
    base["next_safe_command"] = handoff["next_safe_command"]
    base["next_safe_commands"] = list(handoff["next_safe_commands"])
    base["next_safe_command_count"] = handoff["next_safe_command_count"]
    return base


def _normalize_preference_handoff(handoff: dict[str, Any]) -> dict[str, Any]:
    raw_next = handoff.get("next_commands") or []
    if isinstance(raw_next, dict):
        next_safe_commands = [str(value) for value in raw_next.values() if str(value or "").strip()]
    else:
        next_safe_commands = [str(value) for value in raw_next if str(value or "").strip()]
    handoff["handoff_ready"] = True
    handoff["next_safe_command"] = next_safe_commands[0] if next_safe_commands else ""
    handoff["next_safe_commands"] = list(next_safe_commands)
    handoff["next_safe_command_count"] = len(next_safe_commands)
    return handoff


def _preference_write_metadata(**extra: Any) -> dict[str, Any]:
    metadata = _safe_metadata(**extra)
    metadata.update(
        {
            "writes_files": True,
            "writes_memory": True,
            "writes_notes": True,
        }
    )
    return metadata


def _bad_preference_id_metadata(value: Any) -> dict[str, Any]:
    return _safe_metadata(reason="bad_preference_id", preference_id=None, raw_preference_id=_short_metadata(value, 80))


def _preference_refusal_boundaries(*, read_only: bool) -> dict[str, bool]:
    return {
        "read_only": read_only,
        "writes_files": False,
        "writes_memory": False,
        "writes_notes": False,
        "writes_database": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": False,
        "external_side_effect": False,
        "calls_model": False,
        "executes_tools": False,
        "completes_tasks": False,
        "creates_preference": False,
        "updates_preference": False,
        "reads_private_data": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _preference_refusal_handoff(
    *,
    source: str,
    mutation: str,
    reason: str,
    read_only: bool,
    preference_id: int | None = None,
    category: str | None = None,
    status: str | None = None,
    raw_key: str | None = None,
    raw_value: str | None = None,
    raw_status: str | None = None,
    raw_preference_id: str | None = None,
) -> dict[str, Any]:
    handoff: dict[str, Any] = {
        "source": source,
        **_preference_contract(
            content_in_handoff=any(
                value is not None
                for value in (raw_key, raw_value, raw_status, raw_preference_id)
            )
        ),
        "refused": True,
        "mutation": mutation,
        "reason": reason,
        "preference_id": preference_id,
        "category": category,
        "status": status,
        "changed": [],
        "next_commands": {
            "retry": f"retry {source} with valid preference input",
            "list": "preferences",
            "all": "all preferences",
        },
        "boundaries": _preference_refusal_boundaries(read_only=read_only),
    }
    if raw_key is not None:
        handoff["raw_key"] = raw_key
    if raw_value is not None:
        handoff["raw_value"] = raw_value
    if raw_status is not None:
        handoff["raw_status"] = raw_status
    if raw_preference_id is not None:
        handoff["raw_preference_id"] = raw_preference_id
    return handoff


def _preference_refusal_metadata(
    *,
    source: str,
    mutation: str,
    reason: str,
    read_only: bool,
    preference_id: int | None = None,
    category: str | None = None,
    status: str | None = None,
    raw_key: Any = None,
    raw_value: Any = None,
    raw_status: Any = None,
    raw_preference_id: Any = None,
) -> dict[str, Any]:
    safe_category = _short_metadata(category, MAX_CATEGORY_CHARS) if category is not None else None
    safe_status = _short_metadata(status, 80) if status is not None else None
    safe_raw_key = _short_metadata(raw_key, 80) if raw_key is not None else None
    safe_raw_value = _short_metadata(raw_value, 80) if raw_value is not None else None
    safe_raw_status = _short_metadata(raw_status, 80) if raw_status is not None else None
    safe_raw_preference_id = _short_metadata(raw_preference_id, 80) if raw_preference_id is not None else None
    metadata = _safe_metadata(
        reason=reason,
        preference_id=preference_id,
        category=safe_category,
        status=safe_status,
        preference_refusal_handoff_ready=True,
        **_preference_contract(
            content_in_handoff=any(
                value is not None
                for value in (safe_raw_key, safe_raw_value, safe_raw_status, safe_raw_preference_id)
            )
        ),
    )
    if safe_raw_key is not None:
        metadata["raw_key"] = safe_raw_key
    if safe_raw_value is not None:
        metadata["raw_value"] = safe_raw_value
    if safe_raw_status is not None:
        metadata["raw_status"] = safe_raw_status
    if safe_raw_preference_id is not None:
        metadata["raw_preference_id"] = safe_raw_preference_id
    handoff = _preference_refusal_handoff(
        source=source,
        mutation=mutation,
        reason=reason,
        read_only=read_only,
        preference_id=preference_id,
        category=safe_category,
        status=safe_status,
        raw_key=safe_raw_key,
        raw_value=safe_raw_value,
        raw_status=safe_raw_status,
        raw_preference_id=safe_raw_preference_id,
    )
    _normalize_preference_handoff(handoff)
    metadata["preference_refusal_handoff"] = handoff
    metadata["preference_refusal_handoff_ready"] = True
    metadata["preference_refusal_ready_for_operator"] = True
    metadata["preference_refusal_state_changed"] = False
    metadata["preference_refusal_changed"] = []
    metadata["preference_refusal_content_in_handoff"] = _metadata_bool(handoff.get("content_in_handoff"))
    metadata["preference_refusal_next_safe_command"] = handoff["next_safe_command"]
    metadata["preference_refusal_next_safe_commands"] = list(handoff["next_safe_commands"])
    metadata["preference_refusal_next_safe_command_count"] = handoff["next_safe_command_count"]
    metadata["preference_refusal_authorizes_execution"] = False
    metadata["preference_refusal_authorizes_completion_claim"] = False
    metadata["preference_refusal_approval_granted"] = False
    metadata["next_safe_command"] = handoff["next_safe_command"]
    metadata["next_safe_commands"] = list(handoff["next_safe_commands"])
    metadata["next_safe_command_count"] = handoff["next_safe_command_count"]
    return metadata


def _missing_preference_status_result(*, preference_id: int, status: str) -> ToolResult:
    verify_command = "show preference <correct preference key>"
    retry_command = "preference <correct preference id> <active|retired>"
    metadata = _preference_refusal_metadata(
        source="set_preference_status",
        mutation="preference_status_update",
        reason="not_found",
        read_only=False,
        preference_id=preference_id,
        status=status,
    )
    metadata.update(
        {
            "next_command": "preferences",
            "recovery_commands": ["preferences", verify_command, retry_command],
            "retry_requires_preference_refresh": True,
            "retry_requires_corrected_id": True,
            "retry_requires_fresh_approval": False,
            "recovery_commands_require_normal_policy": True,
            "authorizes_retry": False,
            "authorizes_preference_mutation": False,
        }
    )
    output = (
        f"No preference found for #{preference_id}. Run `preferences` to refresh preference IDs, then "
        f"`{verify_command}` to verify the correct record. After that, run `{retry_command}` through "
        "the normal local-safe policy."
    )
    output += f" {RESOURCE_NOT_FOUND_RECOVERY_ACTION}"
    return ToolResult(
        "set_preference_status",
        False,
        output,
        declare_resource_not_found_failure(
            metadata,
            output=output,
            action=RESOURCE_NOT_FOUND_RECOVERY_ACTION,
        ),
    )


def _positive_preference_id(value: Any) -> tuple[int | None, str | None]:
    if isinstance(value, bool):
        return None, "preference_id must be a number."
    try:
        preference_id = int(value)
    except (TypeError, ValueError):
        return None, "preference_id must be a number."
    if preference_id <= 0:
        return None, "preference_id must be a positive number."
    return preference_id, None


def _preference_list_handoff(rows: list[Any], *, category: str | None, status: str, limit: int) -> dict[str, Any]:
    compact_rows: list[dict[str, Any]] = []
    unreadable_rows = 0
    for row in rows[:12]:
        preference_id = _row_positive_int(row)
        key = _short_metadata(_row_text(row, "key"), 120)
        if preference_id is None or not key:
            unreadable_rows += 1
            continue
        row_category = _short_metadata(_row_text(row, "category", "general"), MAX_CATEGORY_CHARS)
        compact_rows.append(
            {
                "id": preference_id,
                "category": row_category,
                "key": key,
                "value": _short_metadata(_row_text(row, "value"), 180),
                "status": _short_metadata(_row_text(row, "status", "unknown"), 32),
                "show_command": f"show preference {key}",
                "retire_command": f"preference {preference_id} retired",
                "reactivate_command": f"preference {preference_id} active",
            }
        )
    preference_ids = [row["id"] for row in compact_rows]
    next_commands = ["preferences", "all preferences"]
    if compact_rows:
        next_commands.insert(1, compact_rows[0]["show_command"])
    return {
        "source": "list_preferences",
        **_preference_contract(content_in_handoff=bool(compact_rows)),
        "category": category,
        "status": status,
        "limit": limit,
        "count": len(rows),
        "readable_preference_rows": len(compact_rows),
        "unreadable_preference_rows": unreadable_rows,
        "preference_ids": preference_ids,
        "rows": compact_rows,
        "first_show_command": compact_rows[0]["show_command"] if compact_rows else None,
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


def _preference_read_handoff(row: Any | None, *, key: str, category: str | None) -> dict[str, Any]:
    found = row is not None
    preference_id = _row_positive_int(row) if found else None
    readable = found and preference_id is not None
    return {
        "source": "get_preference",
        **_preference_contract(content_in_handoff=readable),
        "status": "found" if readable else ("unreadable" if found else "missing"),
        "key": _short_metadata(key, MAX_KEY_CHARS),
        "category": _short_metadata(category, MAX_CATEGORY_CHARS) if category else None,
        "preference_id": preference_id,
        "preference_status": _short_metadata(_row_text(row, "status", "unknown"), 32) if readable else None,
        "value_preview": _short_metadata(_row_text(row, "value"), 180) if readable else "",
        "next_commands": ["preferences", "all preferences"],
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


def _preference_mutation_handoff(
    *,
    source: str,
    mutation: str,
    preference_id: int,
    status: str = "",
    path_display: str = "",
    memory_id: int | None = None,
    state_changed: bool = True,
) -> dict[str, Any]:
    changed = (
        ["preference"] if mutation == "preference_create" else ["preference_status"]
    ) if state_changed else []
    return {
        "source": source,
        **_preference_contract(state_changed=state_changed, changed=changed),
        "mutation": mutation,
        "preference_id": preference_id,
        "status": status,
        "path_display": path_display,
        "memory_id": memory_id,
        "next_commands": ["preferences", "all preferences", f"show preference {preference_id}"],
        "boundaries": {
            "read_only": False,
            "writes_files": True,
            "writes_memory": True,
            "writes_notes": True,
            "queues_approval": False,
            "requires_approval": False,
            "controls_computer": False,
            "external_side_effect": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
        },
    }


def make_preference_tools(store: MemoryStore, vault: ObsidianVault):
    def set_preference(args: dict[str, Any]) -> ToolResult:
        raw_key = args.get("key")
        key = _short(raw_key, MAX_KEY_CHARS)
        value = _short(args.get("value"), MAX_VALUE_CHARS)
        category = _short(args.get("category") or "general", MAX_CATEGORY_CHARS).lower() or "general"
        if not key:
            return _preference_input_failure(
                "set_preference",
                "Preference key is required.",
                _preference_refusal_metadata(
                    source="set_preference",
                    mutation="preference_create",
                    reason="missing_key",
                    read_only=False,
                    raw_key=raw_key,
                    category=category,
                ),
            )
        if LOCAL_PATH_RE.search(key):
            return _preference_input_failure(
                "set_preference",
                "Preference key should describe a preference, not a local file path.",
                _preference_refusal_metadata(
                    source="set_preference",
                    mutation="preference_create",
                    reason="invalid_key",
                    read_only=False,
                    raw_key=raw_key,
                    category=category,
                ),
            )
        if not value:
            return _preference_input_failure(
                "set_preference",
                "Preference value is required.",
                _preference_refusal_metadata(
                    source="set_preference",
                    mutation="preference_create",
                    reason="missing_value",
                    read_only=False,
                    raw_key=raw_key,
                    raw_value=args.get("value"),
                    category=category,
                ),
            )
        mutation = store.set_preference_with_projections(
            PreferenceRecord(key=key, value=value, category=category)
        )
        preference_id = mutation.preference_id
        memory_id = mutation.memory_id
        preference_projection = reconcile_preference_projection(
            store, vault, mutation.preference_projection_target
        )
        memory_projection = reconcile_memory_projection(
            store,
            vault,
            memory_id,
            expected_operation=mutation.memory_projection_target.operation,
            expected_revision=mutation.memory_projection_target.revision,
            expected_source_digest=mutation.memory_projection_target.source_digest,
        )
        preference_complete, memory_complete = store.preference_mutation_projection_completion(
            preference_id,
            memory_id,
            mutation.preference_projection_target,
            mutation.memory_projection_target,
        )
        path_display = "Memory Tree/Preferences.md"
        handoff = _preference_mutation_handoff(
            source="set_preference",
            mutation="preference_create",
            preference_id=preference_id,
            path_display=path_display,
            memory_id=memory_id,
            state_changed=mutation.changed,
        )
        if (
            preference_projection.status != "completed"
            or memory_projection.status != "completed"
            or not preference_complete
            or not memory_complete
        ):
            current = store.get_preference_by_id(preference_id)
            if preference_projection.status == "superseded" or (
                current is not None and str(current["value"]) != value
            ):
                current_value = _short_metadata(
                    current["value"] if current is not None else "unknown",
                    MAX_VALUE_CHARS,
                )
                return ToolResult(
                    "set_preference",
                    False,
                    f"Preference #{preference_id} changed again before this receipt completed. "
                    f"Current value: {current_value}. Review `show preference {key}`; "
                    "the newer state was not overwritten.",
                    _preference_handoff_metadata(
                        "preference_mutation_handoff",
                        _preference_mutation_handoff(
                            source="set_preference",
                            mutation="preference_create",
                            preference_id=preference_id,
                            path_display=path_display,
                            memory_id=memory_id,
                            state_changed=False,
                        ),
                        writes=True,
                        preference_id=preference_id,
                        memory_id=memory_id,
                        path_display=path_display,
                        preference_projection_status=preference_projection.status,
                        projection_status=memory_projection.status,
                        projection_pending=False,
                        mutation_superseded=True,
                        key_chars=len(key),
                        value_chars=len(value),
                    ),
                )
            pending = []
            repairs = []
            if not preference_complete:
                pending.append("preferences note")
                repairs.append("`repair preference projections`")
            if not memory_complete:
                pending.append("memory note")
                repairs.append("`repair memory projections`")
            return ToolResult(
                "set_preference",
                False,
                (
                    f"Saved preference #{preference_id}, but {' and '.join(pending) or 'projection'} "
                    f"publication remains pending. Run {' and '.join(repairs)}; do not save it again."
                ),
                _preference_handoff_metadata(
                    "preference_mutation_handoff",
                    handoff,
                    writes=True,
                    preference_id=preference_id,
                    memory_id=memory_id,
                    path_display=path_display,
                    preference_projection_status=preference_projection.status,
                    projection_status=memory_projection.status,
                    projection_pending=True,
                    key_chars=len(key),
                    value_chars=len(value),
                ),
            )
        value_display = _short_metadata(value, MAX_VALUE_CHARS)
        output = (
            f"Saved preference #{preference_id}: {key} = {value_display}"
            if mutation.changed
            else f"Preference #{preference_id} already matches {key} = {value_display}"
        )
        return ToolResult(
            "set_preference",
            True,
            f"{output}\nSaved note: {path_display}",
            _preference_handoff_metadata(
                "preference_mutation_handoff",
                handoff,
                writes=True,
                preference_id=preference_id,
                memory_id=memory_id,
                path_display=path_display,
                key_chars=len(key),
                value_chars=len(value),
            ),
        )

    def list_preferences(args: dict[str, Any]) -> ToolResult:
        category = _short(args.get("category"), MAX_CATEGORY_CHARS).lower() or None
        raw_status = args.get("status") or "active"
        status = _short(raw_status, 16).lower()
        if status not in {"active", "retired", "all"}:
            failure_output = (
                "Preference status must be active, retired, or all. "
                f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "list_preferences",
                False,
                failure_output,
                declare_retryable_local_read_failure(
                    _preference_refusal_metadata(
                        source="list_preferences",
                        mutation="preference_list",
                        reason="bad_status",
                        read_only=True,
                        category=category,
                        status=status,
                        raw_status=raw_status,
                    ),
                    output=failure_output,
                    action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                ),
            )
        status_filter = None if status == "all" else status
        limit = _bounded_int(args.get("limit"), 100)
        rows = store.list_preferences(category=category, status=status_filter, limit=limit)
        handoff = _preference_list_handoff(rows, category=category, status=status, limit=limit)
        if not rows:
            label = f" in {category}" if category else ""
            return ToolResult(
                "list_preferences",
                True,
                f"No {status} preferences{label}. Count: 0.",
                _preference_handoff_metadata(
                    "preference_list_handoff",
                    handoff,
                    count=0,
                    total_preferences=0,
                    category=category,
                    status=status,
                    readable_preference_rows=handoff["readable_preference_rows"],
                    unreadable_preference_rows=handoff["unreadable_preference_rows"],
                    **_raw_limit_metadata(args.get("limit"), limit=limit),
                ),
            )
        lines = ["Preferences:", f"Count: {len(rows)}"]
        unreadable_rows = 0
        for row in rows:
            preference_id = _row_positive_int(row)
            key = _short_metadata(_row_text(row, "key"), 120)
            if preference_id is None or not key:
                unreadable_rows += 1
                continue
            category_display = _short_metadata(_row_text(row, "category", "general"), MAX_CATEGORY_CHARS)
            value_display = _short_metadata(_row_text(row, "value"), 500)
            status_display = _short_metadata(_row_text(row, "status", "unknown"), 32)
            lines.append(f"- #{preference_id} [{category_display}] {key}: {value_display} ({status_display})")
        if unreadable_rows:
            lines.append(f"- {unreadable_rows} preference row(s) could not be read safely; run `preferences` again after memory repair.")
        return ToolResult(
            "list_preferences",
            True,
            "\n".join(lines),
            _preference_handoff_metadata(
                "preference_list_handoff",
                handoff,
                count=len(rows),
                total_preferences=len(rows),
                category=category,
                status=status,
                readable_preference_rows=handoff["readable_preference_rows"],
                unreadable_preference_rows=handoff["unreadable_preference_rows"],
                **_raw_limit_metadata(args.get("limit"), limit=limit),
            ),
        )

    def get_preference(args: dict[str, Any]) -> ToolResult:
        raw_key = args.get("key")
        key = _short(raw_key, MAX_KEY_CHARS)
        category = _short(args.get("category"), MAX_CATEGORY_CHARS).lower() or None
        if not key:
            failure_output = (
                "Preference key is required. "
                f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "get_preference",
                False,
                failure_output,
                declare_retryable_local_read_failure(
                    _preference_refusal_metadata(
                        source="get_preference",
                        mutation="preference_read",
                        reason="missing_key",
                        read_only=True,
                        preference_id=None,
                        raw_key=raw_key,
                        category=category,
                    ),
                    output=failure_output,
                    action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                ),
            )
        if LOCAL_PATH_RE.search(key):
            failure_output = (
                "Preference key should describe a preference, not a local file path. "
                f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "get_preference",
                False,
                failure_output,
                declare_retryable_local_read_failure(
                    _preference_refusal_metadata(
                    source="get_preference",
                    mutation="preference_read",
                    reason="invalid_key",
                    read_only=True,
                    preference_id=None,
                    raw_key=raw_key,
                    category=category,
                    ),
                    output=failure_output,
                    action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                ),
            )
        row = store.get_preference(key, category=category)
        handoff = _preference_read_handoff(row, key=key, category=category)
        if not row:
            return ToolResult(
                "get_preference",
                True,
                (
                    f"No preference found for '{key}'. Run `preferences` to review saved keys. "
                    "To create or correct it, run `set preference <key> to <value> category <category>` "
                    "through the normal local-safe policy."
                ),
                _preference_handoff_metadata(
                    "preference_read_handoff",
                    handoff,
                    preference_id=None,
                    category=category,
                    next_command="preferences",
                    recovery_commands=[
                        "preferences",
                        "set preference <key> to <value> category <category>",
                    ],
                    retry_requires_preference_refresh=True,
                    recovery_commands_require_normal_policy=True,
                    authorizes_retry=False,
                    authorizes_preference_mutation=False,
                ),
            )
        if handoff.get("status") == "unreadable":
            return ToolResult(
                "get_preference",
                True,
                f"Preference row for '{key}' could not be read safely. Run `preferences` to review memory state.",
                _preference_handoff_metadata(
                    "preference_read_handoff",
                    handoff,
                    preference_id=None,
                    category=category,
                    preference_status="unreadable",
                ),
            )
        output = (
            f"#{handoff['preference_id']} [{_short_metadata(_row_text(row, 'category', 'general'), MAX_CATEGORY_CHARS)}] "
            f"{_short_metadata(_row_text(row, 'key', key), MAX_KEY_CHARS)}: {_short_metadata(_row_text(row, 'value'), MAX_VALUE_CHARS)}\n"
            f"Status: {_short_metadata(_row_text(row, 'status', 'unknown'), 32)}\n"
            f"Created: {_short_metadata(_row_text(row, 'created_at'), 80)} | Updated: {_short_metadata(_row_text(row, 'updated_at'), 80)}"
        )
        return ToolResult(
            "get_preference",
            True,
            output,
            _preference_handoff_metadata(
                "preference_read_handoff",
                handoff,
                preference_id=handoff["preference_id"],
                category=_short_metadata(_row_text(row, "category", "general"), MAX_CATEGORY_CHARS),
                status=_short_metadata(_row_text(row, "status", "unknown"), 32),
            ),
        )

    def set_preference_status(args: dict[str, Any]) -> ToolResult:
        preference_id, error = _positive_preference_id(args.get("preference_id"))
        if preference_id is None:
            return _preference_input_failure(
                "set_preference_status",
                error or "preference_id must be a number.",
                _preference_refusal_metadata(
                    source="set_preference_status",
                    mutation="preference_status_update",
                    reason="bad_preference_id",
                    read_only=False,
                    preference_id=None,
                    raw_preference_id=args.get("preference_id"),
                ),
            )
        raw_status = args.get("status")
        status = _short(raw_status, 16).lower()
        if status not in {"active", "retired"}:
            return _preference_input_failure(
                "set_preference_status",
                "Status must be active or retired.",
                _preference_refusal_metadata(
                    source="set_preference_status",
                    mutation="preference_status_update",
                    reason="bad_status",
                    read_only=False,
                    preference_id=preference_id,
                    status=status,
                    raw_status=raw_status,
                ),
            )
        mutation = store.set_preference_status_with_projections(preference_id, status)
        if mutation is None:
            return _missing_preference_status_result(preference_id=preference_id, status=status)
        preference_projection = reconcile_preference_projection(
            store, vault, mutation.preference_projection_target
        )
        memory_projection = reconcile_memory_projection(
            store,
            vault,
            mutation.memory_id,
            expected_operation=mutation.memory_projection_target.operation,
            expected_revision=mutation.memory_projection_target.revision,
            expected_source_digest=mutation.memory_projection_target.source_digest,
        )
        preference_complete, memory_complete = store.preference_mutation_projection_completion(
            preference_id,
            mutation.memory_id,
            mutation.preference_projection_target,
            mutation.memory_projection_target,
        )
        path_display = "Memory Tree/Preferences.md"
        current = store.get_preference_by_id(preference_id)
        current_status = (
            _short_metadata(current["status"], 32) if current is not None else "unknown"
        )
        if preference_projection.status == "superseded" or current_status != status:
            return ToolResult(
                "set_preference_status",
                False,
                f"Preference #{preference_id} changed again before this status receipt completed. "
                f"Current status: {current_status}. Review `preferences`; "
                "the newer state was not overwritten.",
                _preference_handoff_metadata(
                    "preference_mutation_handoff",
                    _preference_mutation_handoff(
                        source="set_preference_status",
                        mutation="preference_status_update",
                        preference_id=preference_id,
                        status=current_status,
                        path_display=path_display,
                        memory_id=mutation.memory_id,
                        state_changed=False,
                    ),
                    writes=True,
                    preference_id=preference_id,
                    requested_status=status,
                    status=current_status,
                    memory_id=mutation.memory_id,
                    path_display=path_display,
                    preference_projection_status=preference_projection.status,
                    projection_status=memory_projection.status,
                    projection_pending=False,
                    mutation_superseded=True,
                ),
            )
        if (
            preference_projection.status != "completed"
            or memory_projection.status != "completed"
            or not preference_complete
            or not memory_complete
        ):
            return ToolResult(
                "set_preference_status",
                False,
                f"Set preference #{preference_id} to {status}, but projection repair is pending. "
                "Run `repair preference projections` and `repair memory projections`; "
                "do not repeat the status change.",
                _preference_handoff_metadata(
                    "preference_mutation_handoff",
                    _preference_mutation_handoff(
                        source="set_preference_status",
                        mutation="preference_status_update",
                        preference_id=preference_id,
                        status=status,
                        path_display=path_display,
                        memory_id=mutation.memory_id,
                        state_changed=mutation.changed,
                    ),
                    writes=True,
                    preference_id=preference_id,
                    status=status,
                    memory_id=mutation.memory_id,
                    path_display=path_display,
                    preference_projection_status=preference_projection.status,
                    projection_status=memory_projection.status,
                    projection_pending=True,
                ),
            )
        handoff = _preference_mutation_handoff(
            source="set_preference_status",
            mutation="preference_status_update",
            preference_id=preference_id,
            status=status,
            path_display=path_display,
            memory_id=mutation.memory_id,
            state_changed=mutation.changed,
        )
        output = (
            f"Set preference #{preference_id} to {status}."
            if mutation.changed
            else f"Preference #{preference_id} is already {status}."
        )
        return ToolResult(
            "set_preference_status",
            True,
            f"{output}\nSaved note: {path_display}",
            _preference_handoff_metadata(
                "preference_mutation_handoff",
                handoff,
                writes=True,
                preference_id=preference_id,
                status=status,
                path_display=path_display,
                memory_id=mutation.memory_id,
            ),
        )

    return set_preference, list_preferences, get_preference, set_preference_status
