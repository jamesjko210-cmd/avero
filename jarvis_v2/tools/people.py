from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from jarvis_v2.agent.failure_guidance import (
    LOCAL_READ_INPUT_RECOVERY_ACTION,
    declare_failure_guidance,
    declare_retryable_local_read_failure,
    declare_resource_not_found_failure,
)
from jarvis_v2.agent.types import ToolResult
from jarvis_v2.memory.memory_projection import reconcile_memory_projection
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.person_projection import reconcile_person_projection
from jarvis_v2.memory.store import (
    MemoryStore,
    PersonIdentityUnavailable,
    PersonRecord,
    person_identity_part,
)


MAX_PEOPLE_LIMIT = 200
MAX_NAME_CHARS = 120
MAX_RELATION_CHARS = 120
MAX_NOTES_CHARS = 4000
MAX_INTERACTION_CHARS = 4000
MAX_HAPPENED_AT_CHARS = 80
MAX_PERSON_ID = 9223372036854775807
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")
AUTOMATIC_HAPPENED_AT = "<automatic>"
PEOPLE_INPUT_RECOVERY_ACTION = (
    "Correct the reported people input, then submit a new request through the normal policy."
)
PEOPLE_IDENTITY_RECOVERY_ACTION = (
    "Run `setup check`, repair person identity indexing, then submit a new request through the normal policy."
)
PEOPLE_NOT_FOUND_RECOVERY_ACTION = (
    "Run `people`, choose a saved person, then retry the read."
)
PEOPLE_UNREADABLE_RECOVERY_ACTION = (
    "Run `people`, select a readable saved person, then retry the read."
)
PEOPLE_PROJECTION_RECOVERY_ACTION = (
    "Repair the pending people projections; do not repeat the saved mutation."
)


@dataclass(frozen=True)
class _LogInteractionResolution:
    summary: str
    happened_at: str
    name: str
    person: Any | None
    person_id: int | None
    person_identity: str
    reason: str | None


@dataclass(frozen=True)
class _GetPersonResolution:
    person: Any | None
    reason: str | None


def _valid_unicode_text(*values: str) -> bool:
    try:
        for value in values:
            value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _resolve_log_interaction(
    store: MemoryStore,
    args: dict[str, Any],
) -> _LogInteractionResolution:
    summary = _short(args.get("summary"), MAX_INTERACTION_CHARS)
    happened_at = _short(args.get("happened_at"), MAX_HAPPENED_AT_CHARS)
    name = _short(args.get("name"), MAX_NAME_CHARS)
    person_id, person_id_error = _person_id(args)

    if not summary:
        return _LogInteractionResolution(
            summary, happened_at, name, None, person_id, "<missing>", "missing_summary"
        )
    if person_id_error:
        return _LogInteractionResolution(
            summary, happened_at, name, None, None, "<invalid>", "bad_person_id"
        )
    if name and LOCAL_PATH_RE.search(name):
        return _LogInteractionResolution(
            summary, happened_at, name, None, person_id, "<invalid>", "invalid_name"
        )
    if not _valid_unicode_text(summary, happened_at, name):
        return _LogInteractionResolution(
            summary, happened_at, name, None, person_id, "<invalid>", "invalid_unicode"
        )

    if person_id is not None:
        person = store.get_person(person_id=person_id)
        if person is None:
            return _LogInteractionResolution(
                summary,
                happened_at,
                name,
                None,
                person_id,
                f"id:{person_id}",
                "missing_person_id",
            )
        try:
            stored_name = _short(person["name"], MAX_NAME_CHARS)
            stored_person_id = int(person["id"])
        except (IndexError, KeyError, TypeError, ValueError):
            return _LogInteractionResolution(
                summary,
                happened_at,
                name,
                None,
                person_id,
                f"id:{person_id}",
                "invalid_name",
            )
        if not stored_name or not _valid_unicode_text(stored_name):
            return _LogInteractionResolution(
                summary,
                happened_at,
                name,
                None,
                person_id,
                f"id:{person_id}",
                "invalid_unicode",
            )
        if LOCAL_PATH_RE.search(stored_name):
            return _LogInteractionResolution(
                summary,
                happened_at,
                stored_name,
                None,
                person_id,
                f"id:{person_id}",
                "invalid_name",
            )
        stored_identity = person_identity_part(stored_name)
        if name and person_identity_part(name) != stored_identity:
            return _LogInteractionResolution(
                summary,
                happened_at,
                name,
                person,
                person_id,
                f"id:{person_id}",
                "person_target_mismatch",
            )
        try:
            canonical = store.get_person(name=stored_name)
        except PersonIdentityUnavailable:
            return _LogInteractionResolution(
                summary,
                happened_at,
                stored_name,
                person,
                stored_person_id,
                f"id:{stored_person_id}",
                "identity_index_unavailable",
            )
        try:
            canonical_id = int(canonical["id"]) if canonical is not None else None
        except (IndexError, KeyError, TypeError, ValueError):
            canonical_id = None
        if canonical_id != stored_person_id:
            return _LogInteractionResolution(
                summary,
                happened_at,
                stored_name,
                person,
                stored_person_id,
                f"id:{stored_person_id}",
                "noncanonical_person_id",
            )
        return _LogInteractionResolution(
            summary,
            happened_at,
            stored_name,
            person,
            stored_person_id,
            f"name:{stored_identity}",
            None,
        )

    if not name:
        return _LogInteractionResolution(
            summary, happened_at, name, None, None, "<missing>", "missing_person"
        )
    identity = person_identity_part(name)
    try:
        person = store.get_person(name=name)
    except PersonIdentityUnavailable:
        return _LogInteractionResolution(
            summary,
            happened_at,
            name,
            None,
            None,
            f"name:{identity}",
            "identity_index_unavailable",
        )
    try:
        resolved_person_id = int(person["id"]) if person is not None else None
    except (IndexError, KeyError, TypeError, ValueError):
        return _LogInteractionResolution(
            summary,
            happened_at,
            name,
            None,
            None,
            f"name:{identity}",
            "invalid_name",
        )
    return _LogInteractionResolution(
        summary,
        happened_at,
        name,
        person,
        resolved_person_id,
        f"name:{identity}",
        None,
    )


def make_log_interaction_auto_mutation_operation_key(
    store: MemoryStore,
) -> Callable[[dict[str, Any]], dict[str, str]]:
    def operation_key(args: dict[str, Any]) -> dict[str, str]:
        resolved = _resolve_log_interaction(store, args)
        return {
            "person": resolved.person_identity,
            "summary": resolved.summary,
            "happened_at": resolved.happened_at or AUTOMATIC_HAPPENED_AT,
        }

    return operation_key


def make_log_interaction_auto_mutation_preflight(
    store: MemoryStore,
) -> Callable[[dict[str, Any]], str | None]:
    def preflight(args: dict[str, Any]) -> str | None:
        return _resolve_log_interaction(store, args).reason

    return preflight


def person_auto_mutation_preflight(args: dict[str, Any]) -> str | None:
    name = _short(args.get("name"), MAX_NAME_CHARS)
    if not name:
        return "missing_name"
    if LOCAL_PATH_RE.search(name):
        return "invalid_name"
    return None


def person_auto_mutation_operation_key(args: dict[str, Any]) -> dict[str, str]:
    name = _short(args.get("name"), MAX_NAME_CHARS)
    return {"name": person_identity_part(name)}


def _safe_vault_path_display(path: Any, vault: ObsidianVault) -> str:
    try:
        return str(path.relative_to(vault.root_path))
    except (AttributeError, ValueError):
        return _short_metadata(path, 160)


def _opaque_person_path_display(person_id: int) -> str:
    if type(person_id) is not int or person_id < 1:
        raise ValueError("Person metadata id is invalid.")
    return f"People/person-{person_id}.md"


def _bounded_int(value: Any, default: int, low: int = 1, high: int = MAX_PEOPLE_LIMIT) -> int:
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
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _short_raw(value: Any, limit: int) -> str:
    text = "" if value is None else str(value).strip()
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
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "controls_computer": False,
        "queues_approval": False,
        "external_side_effect": False,
        "requires_approval": False,
        "authorizes_retry": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    metadata.update(extra)
    return metadata


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _people_write_metadata(**extra: Any) -> dict[str, Any]:
    metadata = _safe_metadata(**extra)
    metadata.update(
        {
            "writes_files": True,
            "writes_memory": True,
            "writes_notes": True,
        }
    )
    return metadata


def _people_contract(
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
        "authorizes_retry": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _people_handoff_metadata(
    handoff_key: str,
    handoff: dict[str, Any],
    *,
    writes: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    _normalize_people_handoff(handoff)
    prefix = handoff_key.removesuffix("_handoff")
    metadata = _people_write_metadata(**extra) if writes else _safe_metadata(**extra)
    metadata.update(
        _people_contract(
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


def _normalize_people_handoff(handoff: dict[str, Any]) -> dict[str, Any]:
    next_safe_commands = list(handoff.get("next_commands") or [])
    next_safe_command = next_safe_commands[0] if next_safe_commands else ""
    handoff["handoff_ready"] = True
    handoff["next_safe_command"] = next_safe_command
    handoff["next_safe_commands"] = list(next_safe_commands)
    handoff["next_safe_command_count"] = len(next_safe_commands)
    return handoff


def _people_refusal_boundaries() -> dict[str, bool]:
    return {
        "read_only": True,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": False,
        "external_side_effect": False,
        "calls_model": False,
        "executes_tools": False,
        "reads_private_data": False,
        "creates_person": False,
        "creates_interaction": False,
        "authorizes_retry": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _people_refusal_handoff(
    *,
    source: str,
    mutation: str,
    reason: str,
    person_id: int | None = None,
    raw_name: Any = None,
    raw_person_id: Any = None,
) -> dict[str, Any]:
    del raw_name, raw_person_id
    target = str(person_id) if type(person_id) is int and person_id > 0 else "<person-id>"
    retry_command = "people"
    if source == "add_person":
        retry_command = "add person <name>"
    elif source == "get_person":
        retry_command = f"show person {target}"
    elif source == "log_interaction":
        retry_command = f"log interaction with {target}: <summary>"
    return {
        "source": source,
        "mutation": mutation,
        "reason": reason,
        "refused": True,
        "person_id": person_id,
        "retry_command": retry_command,
        "next_commands": ["people", retry_command],
        "boundaries": _people_refusal_boundaries(),
        **_people_contract(content_in_handoff=False),
    }


def _people_refusal_metadata(
    *,
    source: str,
    mutation: str,
    reason: str,
    person_id: int | None = None,
    raw_name: Any = None,
    raw_person_id: Any = None,
    **extra: Any,
) -> dict[str, Any]:
    handoff = _people_refusal_handoff(
        source=source,
        mutation=mutation,
        reason=reason,
        person_id=person_id,
        raw_name=raw_name,
        raw_person_id=raw_person_id,
    )
    metadata = _people_handoff_metadata(
        "people_refusal_handoff",
        handoff,
        reason=reason,
        person_id=person_id,
    )
    metadata.update(extra)
    return metadata


def _bad_person_id_metadata(value: Any, *, source: str, mutation: str) -> dict[str, Any]:
    return _people_refusal_metadata(
        source=source,
        mutation=mutation,
        reason="bad_person_id",
        person_id=None,
        raw_person_id=value,
    )


def _people_known_no_change_failure(
    tool_name: str,
    output: str,
    metadata: dict[str, Any],
    *,
    action: str = PEOPLE_INPUT_RECOVERY_ACTION,
    commands: tuple[str, ...] = (),
) -> ToolResult:
    """Return a canonical refusal whose local effects never started."""

    public_output = output.strip()
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


def _people_projection_pending_failure(
    tool_name: str,
    output: str,
    metadata: dict[str, Any],
) -> ToolResult:
    """Describe a committed mutation with publication work still pending."""

    public_output = f"{output.strip()} {PEOPLE_PROJECTION_RECOVERY_ACTION}"
    result_metadata = dict(metadata)
    result_metadata.update(
        {
            "outcome_known": True,
            "outcome_unknown": False,
            "execution_outcome_unknown": False,
            "side_effect_possible": False,
            "retry_safe": False,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
        }
    )
    return ToolResult(
        tool_name,
        False,
        public_output,
        declare_failure_guidance(
            result_metadata,
            output=public_output,
            action=PEOPLE_PROJECTION_RECOVERY_ACTION,
        ),
    )


def _log_interaction_refusal_result(
    args: dict[str, Any],
    reason: str,
) -> ToolResult:
    person_id, person_id_error = _person_id(args)
    outputs = {
        "missing_summary": "Interaction summary is required.",
        "bad_person_id": person_id_error or "person_id must be a positive number.",
        "missing_person": "Person name or id is required.",
        "missing_person_id": "No person found for that person id.",
        "noncanonical_person_id": "That person id is a legacy duplicate; use the canonical saved person.",
        "identity_index_unavailable": "Person identity matching is temporarily unavailable; nothing was written.",
        "person_target_mismatch": "Person id and name refer to different saved people.",
        "invalid_name": "Person name should be a name, not a local file path.",
        "invalid_unicode": "Person name, summary, and happened_at must contain valid Unicode text.",
    }
    return _people_known_no_change_failure(
        "log_interaction",
        outputs.get(reason, "The interaction failed deterministic validation; nothing was written."),
        _people_refusal_metadata(
            source="log_interaction",
            mutation="interaction_create",
            reason=reason,
            person_id=person_id,
        ),
    )


def _identity_index_unavailable_result(tool_name: str, *, mutation: str) -> ToolResult:
    return _people_known_no_change_failure(
        tool_name,
        "Person identity matching is temporarily unavailable; nothing was written.",
        _people_refusal_metadata(
            source=tool_name,
            mutation=mutation,
            reason="identity_index_unavailable",
            person_id=None,
        ),
        action=PEOPLE_IDENTITY_RECOVERY_ACTION,
    )


def _get_person_selector_refusal_result(reason: str) -> ToolResult:
    return _people_known_no_change_failure(
        "get_person",
        "Person id and name could not be resolved to one saved person.",
        _people_refusal_metadata(
            source="get_person",
            mutation="person_read",
            reason=reason,
            person_id=None,
        ),
    )


def log_interaction_auto_mutation_preflight_result(
    args: dict[str, Any],
    reason: str,
) -> ToolResult:
    result = _log_interaction_refusal_result(args, reason)
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


def _safe_people_list_row(row: Any) -> dict[str, Any] | None:
    try:
        person_id = int(row["id"])
        name = _short_metadata(row["name"], MAX_NAME_CHARS)
        relation = _short_metadata(row["relation"], MAX_RELATION_CHARS)
        last_contact_at = _short_metadata(row["last_contact_at"], MAX_HAPPENED_AT_CHARS)
    except Exception:
        return None
    return {
        "id": person_id,
        "name": name,
        "relation": relation,
        "last_contact_at": last_contact_at,
    }


def _safe_people_list_rows(rows: list[Any]) -> tuple[list[dict[str, Any]], int]:
    readable: list[dict[str, Any]] = []
    unreadable = 0
    for row in rows:
        payload = _safe_people_list_row(row)
        if payload is None:
            unreadable += 1
        else:
            readable.append(payload)
    return readable, unreadable


def _safe_person_read_row(row: Any) -> dict[str, Any] | None:
    try:
        person_id = int(row["id"])
        name = _short_metadata(row["name"], MAX_NAME_CHARS)
        relation = _short_metadata(row["relation"], MAX_RELATION_CHARS)
        notes = _short_metadata(row["notes"], MAX_NOTES_CHARS)
        last_contact_at = _short_metadata(row["last_contact_at"], MAX_HAPPENED_AT_CHARS)
    except Exception:
        return None
    return {
        "id": person_id,
        "name": name,
        "relation": relation,
        "notes": notes,
        "last_contact_at": last_contact_at,
    }


def _safe_person_interaction_row(row: Any) -> dict[str, Any] | None:
    try:
        interaction_id = int(row["id"])
        happened_at = _short_metadata(row["happened_at"], MAX_HAPPENED_AT_CHARS)
        summary = _short_metadata(row["summary"], MAX_INTERACTION_CHARS)
    except Exception:
        return None
    return {
        "id": interaction_id,
        "happened_at": happened_at,
        "summary": summary,
    }


def _safe_person_interaction_rows(rows: list[Any]) -> tuple[list[dict[str, Any]], int]:
    readable: list[dict[str, Any]] = []
    unreadable = 0
    for row in rows:
        payload = _safe_person_interaction_row(row)
        if payload is None:
            unreadable += 1
        else:
            readable.append(payload)
    return readable, unreadable


def _people_list_handoff(rows: list[Any], *, limit: int) -> dict[str, Any]:
    compact_rows: list[dict[str, Any]] = []
    for row in rows[:12]:
        person_id = row["id"]
        compact_rows.append(
            {
                "id": person_id,
                "has_relation": bool(row["relation"]),
                "has_last_contact": bool(row["last_contact_at"]),
                "show_command": f"show person {person_id}",
                "log_interaction_command": f"log interaction with {person_id}: <summary>",
            }
        )
    person_ids = [row["id"] for row in compact_rows]
    next_commands = ["people"]
    if compact_rows:
        next_commands.extend([compact_rows[0]["show_command"], compact_rows[0]["log_interaction_command"]])
    return {
        "source": "list_people",
        "limit": limit,
        "count": len(rows),
        "person_ids": person_ids,
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
        **_people_contract(content_in_handoff=False),
    }


def _people_read_handoff(person: Any, interactions: list[Any], *, limit: int) -> dict[str, Any]:
    person_id = int(person["id"])
    interaction_ids = [int(item["id"]) for item in interactions[:12]]
    return {
        "source": "get_person",
        "person_id": person_id,
        "has_relation": bool(person["relation"]),
        "has_notes": bool(person["notes"]),
        "has_last_contact": bool(person["last_contact_at"]),
        "limit": limit,
        "interaction_count": len(interactions),
        "interaction_ids": interaction_ids,
        "next_commands": [f"log interaction with {person_id}: <summary>", "people"],
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
        **_people_contract(content_in_handoff=False),
    }


def _people_mutation_handoff(
    *,
    source: str,
    mutation: str,
    person_id: int,
    changed: list[str],
    interaction_id: int | None = None,
    memory_id: int | None = None,
    path_display: str = "",
) -> dict[str, Any]:
    return {
        "source": source,
        "mutation": mutation,
        "person_id": person_id,
        "interaction_id": interaction_id,
        "memory_id": memory_id,
        "path_display": path_display,
        "next_commands": [f"show person {person_id}", "people"],
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
        **_people_contract(state_changed=True, changed=changed),
    }


def make_people_tools(store: MemoryStore, vault: ObsidianVault):
    def add_person(args: dict[str, Any]) -> ToolResult:
        raw_name = args.get("name")
        name = _short(raw_name, MAX_NAME_CHARS)
        relation = _short(args.get("relation"), MAX_RELATION_CHARS)
        notes = _short(args.get("notes"), MAX_NOTES_CHARS)
        if not name:
            return _people_known_no_change_failure(
                "add_person",
                "Person name is required.",
                _people_refusal_metadata(
                    source="add_person",
                    mutation="person_create",
                    reason="missing_name",
                    person_id=None,
                    raw_name=raw_name,
                ),
            )
        if LOCAL_PATH_RE.search(name):
            return _people_known_no_change_failure(
                "add_person",
                "Person name should be a name, not a local file path.",
                _people_refusal_metadata(
                    source="add_person",
                    mutation="person_create",
                    reason="invalid_name",
                    person_id=None,
                    raw_name=raw_name,
                ),
            )
        try:
            mutation = store.record_person_with_projections(
                PersonRecord(name=name, relation=relation, notes=notes)
            )
        except PersonIdentityUnavailable:
            return _identity_index_unavailable_result(
                "add_person", mutation="person_create"
            )
        person_id = mutation.person_id
        memory_id = mutation.memory_id
        path_display = _opaque_person_path_display(person_id)
        person_projection = reconcile_person_projection(
            store,
            vault,
            person_id,
            mutation.person_projection_target,
        )
        memory_projection = reconcile_memory_projection(
            store,
            vault,
            memory_id,
            expected_operation=mutation.memory_projection_target.operation,
            expected_revision=mutation.memory_projection_target.revision,
            expected_source_digest=mutation.memory_projection_target.source_digest,
        )
        current_person_complete, current_memory_complete = (
            store.person_mutation_projection_completion(
                person_id, memory_id
            )
        )
        person_complete = (
            person_projection.status in {"completed", "superseded"}
            and current_person_complete
        )
        memory_complete = (
            memory_projection.status in {"completed", "superseded"}
            and current_memory_complete
        )
        if not person_complete or not memory_complete:
            pending_kinds = []
            repair_commands = []
            if not person_complete:
                pending_kinds.append("person note")
                repair_commands.append("`repair person projections`")
            if not memory_complete:
                pending_kinds.append("memory note")
                repair_commands.append("`repair memory projections`")
            projection_output = (
                f"Saved person #{person_id}, but {' and '.join(pending_kinds)} "
                f"publication remains pending. Run {' and '.join(repair_commands)}; "
                "do not add the person again."
            )
            return _people_projection_pending_failure(
                "add_person",
                projection_output,
                _people_handoff_metadata(
                    "people_mutation_handoff",
                    _people_mutation_handoff(
                        source="add_person",
                        mutation="person_create",
                        person_id=person_id,
                        changed=["person"],
                        memory_id=memory_id,
                        path_display=path_display,
                    ),
                    writes=True,
                    person_id=person_id,
                    memory_id=memory_id,
                    path_display=path_display,
                    person_projection_status=person_projection.status,
                    projection_status=memory_projection.status,
                    projection_pending=True,
                    name_chars=len(name),
                    notes_chars=len(notes),
                ),
            )
        return ToolResult(
            "add_person",
            True,
            f"Saved person #{person_id}: {name}\nPerson note updated.",
            _people_handoff_metadata(
                "people_mutation_handoff",
                _people_mutation_handoff(
                    source="add_person",
                    mutation="person_create",
                    person_id=person_id,
                    changed=["person"],
                    memory_id=memory_id,
                    path_display=path_display,
                ),
                writes=True,
                person_id=person_id,
                memory_id=memory_id,
                path_display=path_display,
                person_projection_status=person_projection.status,
                projection_status=memory_projection.status,
                name_chars=len(name),
                notes_chars=len(notes),
            ),
        )

    def list_people(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 25)
        raw_rows = store.list_people(limit=limit)
        rows, unreadable_people_rows = _safe_people_list_rows(raw_rows)
        handoff = _people_list_handoff(rows, limit=limit)
        if not raw_rows:
            return ToolResult(
                "list_people",
                True,
                "No people saved yet. Count: 0.",
                _people_handoff_metadata(
                    "people_list_handoff",
                    handoff,
                    count=0,
                    total_people=0,
                    readable_people_rows=0,
                    unreadable_people_rows=0,
                    **_raw_limit_metadata(args.get("limit"), limit=limit),
                ),
            )
        lines = ["People:", f"Count: {len(rows)}"]
        if unreadable_people_rows:
            lines.append(f"- hidden malformed people rows: {unreadable_people_rows}")
        if not rows:
            lines.append("- No readable people rows.")
        for row in rows:
            relation = f" | {row['relation']}" if row["relation"] else ""
            last = f" | last contact {row['last_contact_at']}" if row["last_contact_at"] else ""
            lines.append(f"- #{row['id']} {row['name']}{relation}{last}")
        return ToolResult(
            "list_people",
            True,
            "\n".join(lines),
            _people_handoff_metadata(
                "people_list_handoff",
                handoff,
                count=len(rows),
                total_people=len(rows),
                readable_people_rows=len(rows),
                unreadable_people_rows=unreadable_people_rows,
                **_raw_limit_metadata(args.get("limit"), limit=limit),
            ),
        )

    def get_person(args: dict[str, Any]) -> ToolResult:
        has_both_selectors = "person_id" in args and "name" in args
        if (
            not has_both_selectors
            and args.get("person_id") is None
            and not _short(args.get("name"), MAX_NAME_CHARS)
        ):
            failure_output = (
                "Person name or id is required. "
                f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "get_person",
                False,
                failure_output,
                declare_retryable_local_read_failure(
                    _people_refusal_metadata(
                        source="get_person",
                        mutation="person_read",
                        reason="missing_person",
                        person_id=None,
                    ),
                    output=failure_output,
                    action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                ),
            )
        _, error = _person_id(args)
        if error:
            failure_output = f"{error} {LOCAL_READ_INPUT_RECOVERY_ACTION}"
            return ToolResult(
                "get_person",
                False,
                failure_output,
                declare_retryable_local_read_failure(
                    _bad_person_id_metadata(
                        args.get("person_id"),
                        source="get_person",
                        mutation="person_read",
                    ),
                    output=failure_output,
                    action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                ),
            )
        try:
            resolution = _resolve_person(store, args)
        except PersonIdentityUnavailable:
            return _identity_index_unavailable_result(
                "get_person", mutation="person_read"
            )
        except ValueError:
            return _get_person_selector_refusal_result("ambiguous_person_identity")
        if resolution.reason is not None:
            return _get_person_selector_refusal_result(resolution.reason)
        person = resolution.person
        if not person:
            failure_output = f"No person found. {PEOPLE_NOT_FOUND_RECOVERY_ACTION}"
            return ToolResult(
                "get_person",
                False,
                failure_output,
                declare_resource_not_found_failure(
                    _people_refusal_metadata(
                        source="get_person",
                        mutation="person_read",
                        reason="not_found",
                        person_id=None,
                        raw_name=args.get("name"),
                        raw_person_id=args.get("person_id"),
                    ),
                    output=failure_output,
                    action=PEOPLE_NOT_FOUND_RECOVERY_ACTION,
                ),
            )
        safe_person = _safe_person_read_row(person)
        if safe_person is None:
            failure_output = (
                "Person record is unreadable. "
                f"{PEOPLE_UNREADABLE_RECOVERY_ACTION}"
            )
            return _people_known_no_change_failure(
                "get_person",
                failure_output,
                _people_refusal_metadata(
                        source="get_person",
                        mutation="person_read",
                        reason="unreadable_person",
                        person_id=None,
                        raw_name=args.get("name"),
                        raw_person_id=args.get("person_id"),
                ),
                action=PEOPLE_UNREADABLE_RECOVERY_ACTION,
                commands=("people",),
            )
        limit = _bounded_int(args.get("limit"), 10)
        raw_interactions = store.list_person_interactions(safe_person["id"], limit=limit)
        interactions, unreadable_interaction_rows = _safe_person_interaction_rows(raw_interactions)
        lines = [
            f"#{safe_person['id']} {safe_person['name']}",
            f"Relation: {safe_person['relation'] or 'not captured'}",
            f"Notes: {safe_person['notes'] or 'not captured'}",
            f"Last contact: {safe_person['last_contact_at'] or 'not captured'}",
            "Interactions:",
        ]
        if unreadable_interaction_rows:
            lines.append(f"- hidden malformed interaction rows: {unreadable_interaction_rows}")
        if interactions:
            for item in interactions:
                lines.append(f"- {item['happened_at']}: {item['summary']}")
        else:
            lines.append("- No interactions captured.")
        return ToolResult(
            "get_person",
            True,
            "\n".join(lines),
            _people_handoff_metadata(
                "people_read_handoff",
                _people_read_handoff(safe_person, interactions, limit=limit),
                person_id=safe_person["id"],
                interactions=len(interactions),
                readable_person_interaction_rows=len(interactions),
                unreadable_person_interaction_rows=unreadable_interaction_rows,
                **_raw_limit_metadata(args.get("limit"), limit=limit),
            ),
        )

    def log_interaction(args: dict[str, Any]) -> ToolResult:
        resolved = _resolve_log_interaction(store, args)
        if resolved.reason is not None:
            return _log_interaction_refusal_result(args, resolved.reason)
        try:
            mutation = store.record_person_interaction_with_projections(
                person_id=resolved.person_id,
                name=resolved.name,
                summary=resolved.summary,
                happened_at=resolved.happened_at,
            )
        except PersonIdentityUnavailable:
            return _identity_index_unavailable_result(
                "log_interaction", mutation="interaction_create"
            )
        person_projection = reconcile_person_projection(
            store,
            vault,
            mutation.person_id,
            mutation.person_projection_target,
        )
        memory_projection = reconcile_memory_projection(
            store,
            vault,
            mutation.memory_id,
            expected_operation=mutation.memory_projection_target.operation,
            expected_revision=mutation.memory_projection_target.revision,
            expected_source_digest=mutation.memory_projection_target.source_digest,
        )
        changed = ["person", "interaction"] if mutation.created_person else ["interaction"]
        person_complete = person_projection.status in {"completed", "superseded"}
        memory_complete = memory_projection.status in {"completed", "superseded"}
        path_display = _opaque_person_path_display(mutation.person_id)
        if not person_complete or not memory_complete:
            pending_kinds = []
            repair_commands = []
            if not person_complete:
                pending_kinds.append("person note")
                repair_commands.append("`repair person projections`")
            if not memory_complete:
                pending_kinds.append("memory note")
                repair_commands.append("`repair memory projections`")
            projection_output = (
                f"Logged interaction #{mutation.interaction_id}, but "
                f"{' and '.join(pending_kinds)} publication remains pending. Run "
                f"{' and '.join(repair_commands)}; do not log it again."
            )
            return _people_projection_pending_failure(
                "log_interaction",
                projection_output,
                _people_handoff_metadata(
                    "people_mutation_handoff",
                    _people_mutation_handoff(
                        source="log_interaction",
                        mutation="interaction_create",
                        person_id=mutation.person_id,
                        interaction_id=mutation.interaction_id,
                        memory_id=mutation.memory_id,
                        path_display=path_display,
                        changed=changed,
                    ),
                    writes=True,
                    person_id=mutation.person_id,
                    interaction_id=mutation.interaction_id,
                    memory_id=mutation.memory_id,
                    path_display=path_display,
                    person_projection_status=person_projection.status,
                    projection_status=memory_projection.status,
                    projection_pending=True,
                    summary_chars=len(resolved.summary),
                ),
            )
        return ToolResult(
            "log_interaction",
            True,
            (
                f"Logged interaction #{mutation.interaction_id} with {mutation.person_name}.\n"
                "Person note updated."
            ),
            _people_handoff_metadata(
                "people_mutation_handoff",
                _people_mutation_handoff(
                    source="log_interaction",
                    mutation="interaction_create",
                    person_id=mutation.person_id,
                    interaction_id=mutation.interaction_id,
                    memory_id=mutation.memory_id,
                    path_display=path_display,
                    changed=changed,
                ),
                writes=True,
                person_id=mutation.person_id,
                interaction_id=mutation.interaction_id,
                memory_id=mutation.memory_id,
                path_display=path_display,
                person_projection_status=person_projection.status,
                projection_status=memory_projection.status,
                summary_chars=len(resolved.summary),
            ),
        )

    return add_person, list_people, get_person, log_interaction


def _person_id(args: dict[str, Any]) -> tuple[int | None, str | None]:
    raw_person_id = args.get("person_id")
    if raw_person_id is None:
        return None, None
    if type(raw_person_id) is int:
        person_id = raw_person_id
    elif type(raw_person_id) is str:
        normalized = raw_person_id.strip()
        if re.fullmatch(r"[0-9]+", normalized) is None or len(normalized) > 19:
            return None, "person_id must be a valid number."
        person_id = int(normalized)
    else:
        return None, "person_id must be a number."
    if person_id <= 0:
        return None, "person_id must be a positive number."
    if person_id > MAX_PERSON_ID:
        return None, "person_id must be a valid number."
    return person_id, None


def _strict_person_name_identity(value: Any) -> str | None:
    if type(value) is not str:
        return None
    name = value.strip()
    if (
        not name
        or len(name) > MAX_NAME_CHARS
        or LOCAL_PATH_RE.search(name)
        or not _valid_unicode_text(name)
    ):
        return None
    identity = person_identity_part(name)
    return identity or None


def _person_identity_binding(row: Any) -> tuple[int, str] | None:
    try:
        person_id = row["id"]
        name = row["name"]
    except Exception:
        return None
    if type(person_id) is not int or person_id < 1 or person_id > MAX_PERSON_ID:
        return None
    identity = _strict_person_name_identity(name)
    if identity is None:
        return None
    return person_id, identity


def _resolve_person(store: MemoryStore, args: dict[str, Any]) -> _GetPersonResolution:
    person_id, _ = _person_id(args)
    if "person_id" not in args or "name" not in args:
        name = _short(args.get("name"), MAX_NAME_CHARS)
        person = (
            store.get_person(person_id=person_id)
            if person_id
            else store.get_person(name=name)
        )
        return _GetPersonResolution(person, None)

    selector_identity = _strict_person_name_identity(args.get("name"))
    if person_id is None or selector_identity is None:
        return _GetPersonResolution(None, "invalid_person_selector")

    status, person = store.get_person_if_identity_matches(
        person_id,
        selector_identity,
    )
    if status == "resolved":
        return _GetPersonResolution(person, None)
    if status == "not_found":
        return _GetPersonResolution(None, None)
    return _GetPersonResolution(None, status)
