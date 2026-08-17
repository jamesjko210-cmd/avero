from __future__ import annotations

import re
import unicodedata
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    RESOURCE_NOT_FOUND_RECOVERY_ACTION,
    declare_failure_guidance,
    declare_resource_not_found_failure,
)
from jarvis_v2.agent.types import ToolResult
from jarvis_v2.memory.decision_projection import reconcile_decision_projection
from jarvis_v2.memory.memory_projection import reconcile_memory_projection
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import (
    DecisionRecord,
    MemoryStore,
    decision_outcome_summary_safety_reason,
    normalize_decision_outcome_summary,
)


MAX_DECISION_LIMIT = 200
MAX_TITLE_CHARS = 240
MAX_RATIONALE_CHARS = 4000
MAX_IMPACT_CHARS = 4000
MAX_DECISION_OUTCOME_SUMMARY_CHARS = 4000
MAX_DECISION_OUTCOMES = 100
MAX_DECISION_OUTCOME_READ_ROWS = 20
MAX_DECISION_OUTCOME_READ_SUMMARY_CHARS = 1000
MAX_SQLITE_ID = 9223372036854775807
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")
DECISION_INPUT_RECOVERY_ACTION = (
    "Correct the reported decision field, then submit a fresh request through the normal policy."
)
DECISION_LIMIT_RECOVERY_ACTION = (
    "Review the decision history and keep the current outcome unchanged; do not retry this request."
)
def _normalize_decision_outcome_summary(value: Any) -> str:
    return normalize_decision_outcome_summary(value)


def _validated_decision_outcome_args(
    args: dict[str, Any],
) -> tuple[int | None, str, str | None]:
    decision_id = args.get("decision_id")
    if type(decision_id) is not int or not (0 < decision_id <= MAX_SQLITE_ID):
        return None, "", "bad_decision_id"
    raw_summary = args.get("summary")
    if type(raw_summary) is not str:
        return decision_id, "", "invalid_summary_type"
    summary = _normalize_decision_outcome_summary(raw_summary)
    return decision_id, summary, decision_outcome_summary_safety_reason(summary)


def decision_outcome_auto_mutation_operation_key(args: dict[str, Any]) -> dict[str, Any]:
    decision_id = args.get("decision_id")
    return {
        "decision_id": (
            decision_id
            if type(decision_id) is int and 0 < decision_id <= MAX_SQLITE_ID
            else 0
        ),
        "summary": _normalize_decision_outcome_summary(args.get("summary")),
    }


def decision_outcome_auto_mutation_preflight(args: dict[str, Any]) -> str | None:
    _decision_id, _summary, reason = _validated_decision_outcome_args(args)
    return reason


def make_decision_outcome_auto_mutation_preflight(store: MemoryStore):
    def preflight(args: dict[str, Any]) -> str | None:
        decision_id, _summary, reason = _validated_decision_outcome_args(args)
        if reason is not None:
            return reason
        assert decision_id is not None
        if store.get_decision(decision_id) is None:
            return "missing_decision"
        if len(store.list_decision_outcomes(decision_id, limit=MAX_DECISION_OUTCOMES)) >= (
            MAX_DECISION_OUTCOMES
        ):
            return "outcome_limit_reached"
        return None

    return preflight


def decision_auto_mutation_preflight(args: dict[str, Any]) -> str | None:
    """Reject deterministic decision-title errors before a mutation receipt exists."""
    title = _short(args.get("title"), MAX_TITLE_CHARS)
    if not title:
        return "missing_title"
    if LOCAL_PATH_RE.search(title):
        return "invalid_title"
    return None


def decision_auto_mutation_operation_key(args: dict[str, Any]) -> dict[str, str]:
    """Return the effective bounded decision payload used by the handler."""
    return {
        "title": _short(args.get("title"), MAX_TITLE_CHARS),
        "rationale": _short(args.get("rationale"), MAX_RATIONALE_CHARS),
        "impact": _short(args.get("impact"), MAX_IMPACT_CHARS),
    }


def _safe_vault_path_display(path: Any, vault: ObsidianVault) -> str:
    try:
        return str(path.relative_to(vault.root_path))
    except (AttributeError, ValueError):
        return _short_metadata(path, 160)


def _bounded_int(value: Any, default: int, low: int = 1, high: int = MAX_DECISION_LIMIT) -> int:
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
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _short_raw(value: Any, limit: int) -> str:
    try:
        text = "" if value is None else str(value).strip()
    except Exception:
        return ""
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
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "external_side_effect": False,
        "requires_approval": False,
    }
    metadata.update(extra)
    return metadata


def _decision_input_failure(
    tool_name: str,
    message: str,
    metadata: dict[str, Any],
    *,
    retry_safe: bool = True,
) -> ToolResult:
    action = DECISION_INPUT_RECOVERY_ACTION if retry_safe else DECISION_LIMIT_RECOVERY_ACTION
    output = f"{message} {action}"
    declared = dict(metadata)
    declared.update(
        {
            "outcome_known": True,
            "outcome_unknown": False,
            "execution_outcome_unknown": False,
            "side_effect_possible": False,
            "retry_safe": retry_safe,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
        }
    )
    return ToolResult(
        tool_name,
        False,
        output,
        declare_failure_guidance(declared, output=output, action=action),
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


def _row_short(row: Any, key: str, limit: int, default: str = "") -> str:
    value = _row_value(row, key)
    if value is _MISSING_ROW_VALUE or value is None:
        return default
    return _short(value, limit)


def _row_metadata_short(row: Any, key: str, limit: int, default: str = "") -> str:
    value = _row_value(row, key)
    if value is _MISSING_ROW_VALUE or value is None:
        return default
    return _short_metadata(value, limit)


def _row_positive_int(row: Any, key: str = "id") -> int | None:
    value = _row_value(row, key)
    if isinstance(value, bool) or value is _MISSING_ROW_VALUE or value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number > 0 else None


def _decision_outcome_row_matches(
    row: Any,
    *,
    outcome_id: int,
    decision_id: int,
    decision_revision: int,
    summary: str,
) -> bool:
    return (
        _row_positive_int(row) == outcome_id
        and _row_positive_int(row, "decision_id") == decision_id
        and _row_positive_int(row, "decision_revision") == decision_revision
        and _row_value(row, "summary") == summary
        and _row_value(row, "provenance") == "user_reported"
    )


def _decision_outcome_views(
    rows: list[Any],
    *,
    decision_id: int,
) -> tuple[list[dict[str, Any]], int, int]:
    readable: list[dict[str, Any]] = []
    unreadable = 0
    ordered_rows = sorted(
        rows[:MAX_DECISION_OUTCOMES],
        key=lambda row: (
            _row_positive_int(row, "decision_revision") or MAX_SQLITE_ID,
            _row_positive_int(row) or MAX_SQLITE_ID,
        ),
    )
    for row in ordered_rows:
        outcome_id = _row_positive_int(row)
        row_decision_id = _row_positive_int(row, "decision_id")
        decision_revision = _row_positive_int(row, "decision_revision")
        raw_summary = _row_value(row, "summary")
        provenance = _row_value(row, "provenance")
        _validated_id, summary, reason = _validated_decision_outcome_args(
            {"decision_id": decision_id, "summary": raw_summary}
        )
        if (
            outcome_id is None
            or row_decision_id != decision_id
            or decision_revision is None
            or provenance != "user_reported"
            or reason is not None
        ):
            unreadable += 1
            continue
        summary_truncated = len(summary) > MAX_DECISION_OUTCOME_READ_SUMMARY_CHARS
        readable.append(
            {
                "id": outcome_id,
                "decision_revision": decision_revision,
                "summary": _short(summary, MAX_DECISION_OUTCOME_READ_SUMMARY_CHARS),
                "summary_truncated": summary_truncated,
                "reported_at": _row_metadata_short(row, "created_at", 80),
                "provenance": "user_reported",
                "verification_status": "not_verified",
                "label": "user-reported (not verified)",
            }
        )
    hidden = max(0, len(readable) - MAX_DECISION_OUTCOME_READ_ROWS)
    return readable[-MAX_DECISION_OUTCOME_READ_ROWS:], hidden, unreadable


def _decision_write_metadata(**extra: Any) -> dict[str, Any]:
    metadata = _safe_metadata()
    metadata.update(
        {
            "writes_files": True,
            "writes_memory": True,
            "writes_notes": True,
        }
    )
    metadata.update(extra)
    return metadata


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _decision_contract(
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


def _decision_handoff_metadata(
    handoff_key: str,
    handoff: dict[str, Any],
    *,
    writes: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    _normalize_decision_handoff(handoff)
    prefix = handoff_key.removesuffix("_handoff")
    metadata = _decision_write_metadata(**extra) if writes else _safe_metadata(**extra)
    metadata.update(
        _decision_contract(
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


def _normalize_decision_handoff(handoff: dict[str, Any]) -> dict[str, Any]:
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


def _positive_id(value: Any, label: str) -> tuple[int | None, str | None]:
    if isinstance(value, bool):
        return None, f"{label} must be a number."
    try:
        item_id = int(value)
    except (TypeError, ValueError):
        return None, f"{label} must be a number."
    if item_id <= 0:
        return None, f"{label} must be a positive number."
    return item_id, None


def _bad_decision_id_metadata(value: Any) -> dict[str, Any]:
    return _safe_metadata(
        reason="bad_decision_id",
        decision_id=None,
        raw_decision_id=_short_metadata(value, 80),
    )


def _decision_refusal_boundaries(*, read_only: bool) -> dict[str, bool]:
    return {
        "read_only": read_only,
        "writes_files": False,
        "writes_memory": False,
        "writes_notes": False,
        "writes_database": False,
        "queues_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "requires_approval": False,
        "controls_computer": False,
        "external_side_effect": False,
        "calls_model": False,
        "executes_tools": False,
        "completes_tasks": False,
        "creates_decision": False,
        "updates_decision": False,
        "reads_private_data": False,
    }


def _decision_refusal_handoff(
    *,
    source: str,
    mutation: str,
    reason: str,
    read_only: bool,
    decision_id: int | None = None,
    status: str | None = None,
    raw_title: str | None = None,
    raw_status: str | None = None,
    raw_decision_id: str | None = None,
) -> dict[str, Any]:
    handoff: dict[str, Any] = {
        "source": source,
        "refused": True,
        "mutation": mutation,
        "reason": reason,
        "decision_id": decision_id,
        "status": status,
        "next_commands": {
            "retry": f"retry {source} with valid decision input",
            "list": "decisions",
            "all": "all decisions",
        },
        "boundaries": _decision_refusal_boundaries(read_only=read_only),
        **_decision_contract(
            content_in_handoff=raw_title is not None or raw_status is not None or raw_decision_id is not None
        ),
    }
    if raw_title is not None:
        handoff["raw_title"] = raw_title
    if raw_status is not None:
        handoff["raw_status"] = raw_status
    if raw_decision_id is not None:
        handoff["raw_decision_id"] = raw_decision_id
    return handoff


def _decision_refusal_metadata(
    *,
    source: str,
    mutation: str,
    reason: str,
    read_only: bool,
    decision_id: int | None = None,
    status: str | None = None,
    raw_title: Any = None,
    raw_status: Any = None,
    raw_decision_id: Any = None,
) -> dict[str, Any]:
    safe_status = _short_metadata(status, 80) if status is not None else None
    safe_raw_title = _short_metadata(raw_title, 80) if raw_title is not None else None
    safe_raw_status = _short_metadata(raw_status, 80) if raw_status is not None else None
    safe_raw_decision_id = _short_metadata(raw_decision_id, 80) if raw_decision_id is not None else None
    handoff = _decision_refusal_handoff(
        source=source,
        mutation=mutation,
        reason=reason,
        read_only=read_only,
        decision_id=decision_id,
        status=safe_status,
        raw_title=safe_raw_title,
        raw_status=safe_raw_status,
        raw_decision_id=safe_raw_decision_id,
    )
    metadata = _decision_handoff_metadata(
        "decision_refusal_handoff",
        handoff,
        reason=reason,
        decision_id=decision_id,
        status=safe_status,
    )
    if safe_raw_title is not None:
        metadata["raw_title"] = safe_raw_title
    if safe_raw_status is not None:
        metadata["raw_status"] = safe_raw_status
    if safe_raw_decision_id is not None:
        metadata["raw_decision_id"] = safe_raw_decision_id
    return metadata


def _missing_decision_result(
    *,
    tool_name: str,
    decision_id: int,
    mutation: str,
    read_only: bool,
    retry_command: str,
    status: str | None = None,
) -> ToolResult:
    verify_command = "show decision <correct decision id>"
    recovery_commands = ["decisions", verify_command]
    if retry_command and retry_command not in recovery_commands:
        recovery_commands.append(retry_command)
    policy = "normal read-only policy" if read_only else "normal local-safe policy"
    output = (
        f"No decision found for #{decision_id}. Run `decisions` to refresh decision IDs, then "
        f"`{verify_command}` to verify the correct record."
    )
    if retry_command != verify_command:
        output += f" After that, run `{retry_command}` through the {policy}."
    else:
        output += f" Use that command through the {policy}."
    output += f" {RESOURCE_NOT_FOUND_RECOVERY_ACTION}"
    metadata = _decision_refusal_metadata(
        source=tool_name,
        mutation=mutation,
        reason="not_found",
        read_only=read_only,
        decision_id=decision_id,
        status=status,
    )
    metadata.update(
        {
            "next_command": "decisions",
            "recovery_commands": recovery_commands,
            "retry_requires_decision_refresh": True,
            "retry_requires_corrected_id": True,
            "retry_requires_fresh_approval": False,
            "recovery_commands_require_normal_policy": True,
            "authorizes_retry": False,
            "authorizes_decision_mutation": False,
        }
    )
    return ToolResult(
        tool_name,
        False,
        output,
        declare_resource_not_found_failure(
            metadata,
            output=output,
            action=RESOURCE_NOT_FOUND_RECOVERY_ACTION,
        ),
    )


def _decision_list_handoff(rows: list[Any], *, status: str, limit: int) -> dict[str, Any]:
    compact_rows: list[dict[str, Any]] = []
    unreadable_rows = 0
    for row in rows[:12]:
        decision_id = _row_positive_int(row)
        title = _row_metadata_short(row, "title", 160)
        if decision_id is None or not title:
            unreadable_rows += 1
            continue
        compact_rows.append(
            {
                "id": decision_id,
                "title": title,
                "status": _row_metadata_short(row, "status", 80, "unknown"),
                "show_command": f"show decision {decision_id}",
                "supersede_command": f"decision {decision_id} superseded",
                "retire_command": f"decision {decision_id} retired",
            }
        )
    decision_ids = [row["id"] for row in compact_rows]
    next_commands = ["decisions", "all decisions"]
    if decision_ids:
        next_commands.insert(1, f"show decision {decision_ids[0]}")
    return {
        "source": "list_decisions",
        "status": status,
        "limit": limit,
        "count": len(rows),
        "readable_decision_rows": len(compact_rows),
        "unreadable_decision_rows": unreadable_rows,
        "decision_ids": decision_ids,
        "rows": compact_rows,
        "first_show_command": compact_rows[0]["show_command"] if compact_rows else None,
        "next_commands": next_commands,
        "boundaries": {
            "read_only": True,
            "writes_files": False,
            "writes_memory": False,
            "writes_notes": False,
            "queues_approval": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "controls_computer": False,
            "external_side_effect": False,
        },
        **_decision_contract(content_in_handoff=bool(compact_rows)),
    }


def _decision_read_handoff(
    decision: Any,
    *,
    requested_id: int | None = None,
    outcomes: list[dict[str, Any]] | None = None,
    hidden_outcome_rows: int = 0,
    unreadable_outcome_rows: int = 0,
) -> dict[str, Any]:
    decision_id = _row_positive_int(decision) or requested_id
    title = _row_metadata_short(decision, "title", 160)
    status = _row_metadata_short(decision, "status", 80, "unreadable")
    readable = decision_id is not None and bool(title)
    if not readable:
        return {
            "source": "get_decision",
            "decision_id": decision_id,
            "status": "unreadable",
            "readable_decision_row": False,
            "unreadable_decision_row": True,
            "next_commands": ["decisions", "all decisions"],
            "boundaries": {
                "read_only": True,
                "writes_files": False,
                "writes_memory": False,
                "writes_notes": False,
                "queues_approval": False,
                "authorizes_execution": False,
                "authorizes_completion_claim": False,
                "approval_granted": False,
                "controls_computer": False,
                "external_side_effect": False,
            },
            **_decision_contract(content_in_handoff=False),
        }
    assert decision_id is not None
    outcome_rows = list(outcomes or [])
    return {
        "source": "get_decision",
        "decision_id": decision_id,
        "status": status,
        "title": title,
        "rationale_preview": _row_metadata_short(decision, "rationale", 240),
        "impact_preview": _row_metadata_short(decision, "impact", 240),
        "created_at": _row_metadata_short(decision, "created_at", 80),
        "updated_at": _row_metadata_short(decision, "updated_at", 80),
        "outcomes": outcome_rows,
        "outcome_count": len(outcome_rows),
        "hidden_outcome_rows": hidden_outcome_rows,
        "unreadable_outcome_rows": unreadable_outcome_rows,
        "outcomes_bounded": hidden_outcome_rows > 0,
        "outcome_label": "user-reported (not verified)",
        "outcomes_provenance": "user_reported",
        "outcomes_verification_status": "not_verified",
        "readable_decision_row": True,
        "unreadable_decision_row": False,
        "next_commands": [
            f"decision outcome #{decision_id}: <summary>",
            f"decision {decision_id} superseded",
            f"decision {decision_id} retired",
            "decisions",
        ],
        "boundaries": {
            "read_only": True,
            "writes_files": False,
            "writes_memory": False,
            "writes_notes": False,
            "queues_approval": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "controls_computer": False,
            "external_side_effect": False,
        },
        **_decision_contract(content_in_handoff=True),
    }


def _decision_mutation_handoff(
    *,
    source: str,
    mutation: str,
    decision_id: int,
    status: str,
    changed: list[str],
    memory_id: int | None = None,
    path_display: str = "",
) -> dict[str, Any]:
    return {
        "source": source,
        "mutation": mutation,
        "decision_id": decision_id,
        "status": status,
        "memory_id": memory_id,
        "path_display": path_display,
        "next_commands": [f"show decision {decision_id}", "decisions", "all decisions"],
        "boundaries": {
            "read_only": False,
            "writes_files": True,
            "writes_memory": memory_id is not None,
            "writes_notes": True,
            "queues_approval": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "requires_approval": False,
            "controls_computer": False,
            "external_side_effect": False,
        },
        **_decision_contract(state_changed=True, changed=changed),
    }


def _decision_outcome_mutation_handoff(
    *,
    decision_id: int,
    status: str,
    outcome_id: int,
    decision_revision: int,
    summary: str,
    projection_pending: bool,
) -> dict[str, Any]:
    summary_preview = _short(summary, MAX_DECISION_OUTCOME_READ_SUMMARY_CHARS)
    handoff = _decision_mutation_handoff(
        source="record_decision_outcome",
        mutation="decision_outcome_append",
        decision_id=decision_id,
        status=status,
        changed=["outcome"],
        path_display="" if projection_pending else f"Decisions/decision-{decision_id}.md",
    )
    handoff["boundaries"]["writes_files"] = not projection_pending
    handoff["boundaries"]["writes_notes"] = not projection_pending
    handoff.update(
        {
            "outcome": {
                "id": outcome_id,
                "decision_revision": decision_revision,
                "summary": summary_preview,
                "summary_truncated": len(summary) > len(summary_preview),
                "provenance": "user_reported",
                "verification_status": "not_verified",
                "label": "user-reported (not verified)",
            },
            "outcome_id": outcome_id,
            "outcome_label": "user-reported (not verified)",
            "provenance": "user_reported",
            "verification_status": "not_verified",
            "verified": False,
            "projection_pending": projection_pending,
            "retryable": False,
            "authorizes_retry": False,
            "do_not_record_again": projection_pending,
            "content_in_handoff": True,
        }
    )
    handoff["next_commands"] = (
        ["repair decision projections", f"show decision {decision_id}", "decisions"]
        if projection_pending
        else [f"show decision {decision_id}", "decisions", "all decisions"]
    )
    return handoff


def _decision_outcome_refusal_result(
    *,
    decision_id: int | None,
    reason: str,
    raw_decision_id: Any = None,
) -> ToolResult:
    messages = {
        "bad_decision_id": "decision_id must be a positive bounded integer.",
        "invalid_summary_type": "Decision outcome summary must be text.",
        "missing_summary": "Decision outcome summary is required.",
        "oversized_summary": (
            f"Decision outcome summary exceeds the {MAX_DECISION_OUTCOME_SUMMARY_CHARS}-character limit."
        ),
        "unsafe_unicode": "Decision outcome summary contains unsafe Unicode characters.",
        "active_render_content": "Decision outcome summary contains active render content.",
        "reserved_projection_marker": (
            "Decision outcome summary uses a reserved Jarvis projection marker."
        ),
        "path_shaped_summary": (
            "Decision outcome summary must describe the result, not a local or file-like path."
        ),
        "outcome_limit_reached": "This decision has reached its bounded outcome history limit.",
    }
    metadata = _decision_refusal_metadata(
        source="record_decision_outcome",
        mutation="decision_outcome_append",
        reason=reason,
        read_only=False,
        decision_id=decision_id,
        raw_decision_id=raw_decision_id if decision_id is None else None,
    )
    metadata.update(
        {
            "auto_mutation_effects_started": False,
            "writes_database": False,
            "provenance": "user_reported",
            "verification_status": "not_verified",
            "verified": False,
            "outcome_label": "user-reported (not verified)",
            "retryable": reason not in {"outcome_limit_reached"},
            "authorizes_retry": False,
        }
    )
    return _decision_input_failure(
        "record_decision_outcome",
        messages.get(reason, "Decision outcome could not be recorded."),
        metadata,
        retry_safe=reason not in {"outcome_limit_reached"},
    )


def make_decision_tools(store: MemoryStore, vault: ObsidianVault):
    def record_decision(args: dict[str, Any]) -> ToolResult:
        raw_title = args.get("title")
        title = _short(raw_title, MAX_TITLE_CHARS)
        rationale = _short(args.get("rationale"), MAX_RATIONALE_CHARS)
        impact = _short(args.get("impact"), MAX_IMPACT_CHARS)
        if not title:
            return _decision_input_failure(
                "record_decision",
                "Decision title is required.",
                _decision_refusal_metadata(
                    source="record_decision",
                    mutation="decision_create",
                    reason="missing_title",
                    read_only=False,
                    decision_id=None,
                    raw_title=raw_title,
                ),
            )
        if LOCAL_PATH_RE.search(title):
            return _decision_input_failure(
                "record_decision",
                "Decision title should describe a choice, not a local file path.",
                _decision_refusal_metadata(
                    source="record_decision",
                    mutation="decision_create",
                    reason="invalid_title",
                    read_only=False,
                    decision_id=None,
                    raw_title=raw_title,
                ),
            )
        mutation = store.record_decision_with_projections(
            DecisionRecord(title=title, rationale=rationale, impact=impact)
        )
        decision_id = mutation.decision_id
        memory_id = mutation.memory_id
        decision_projection = reconcile_decision_projection(
            store, vault, mutation.decision_projection_target
        )
        memory_projection = reconcile_memory_projection(
            store,
            vault,
            memory_id,
            expected_operation=mutation.memory_projection_target.operation,
            expected_revision=mutation.memory_projection_target.revision,
            expected_source_digest=mutation.memory_projection_target.source_digest,
        )
        decision_complete, memory_complete = store.decision_mutation_projection_completion(
            decision_id,
            memory_id,
            mutation.decision_projection_target,
            mutation.memory_projection_target,
        )
        decision = store.get_decision(decision_id)
        path_display = f"Decisions/decision-{decision_id}.md"
        if (
            decision is None
            or decision_projection.status != "completed"
            or memory_projection.status != "completed"
            or not decision_complete
            or not memory_complete
        ):
            pending = []
            repairs = []
            if not decision_complete:
                pending.append("decision note")
                repairs.append("`repair decision projections`")
            if not memory_complete:
                pending.append("memory note")
                repairs.append("`repair memory projections`")
            return ToolResult(
                "record_decision",
                False,
                (
                    f"Recorded decision #{decision_id}, but {' and '.join(pending) or 'projection'} "
                    f"publication remains pending. Run {' and '.join(repairs)}; "
                    "do not record it again."
                ),
                _decision_handoff_metadata(
                    "decision_mutation_handoff",
                    _decision_mutation_handoff(
                        source="record_decision",
                        mutation="decision_create",
                        decision_id=decision_id,
                        status=decision["status"],
                        changed=["decision"],
                        memory_id=memory_id,
                        path_display=path_display,
                    ),
                    writes=True,
                    decision_id=decision_id,
                    status=decision["status"] if decision is not None else "active",
                    memory_id=memory_id,
                    path_display=path_display,
                    decision_projection_status=decision_projection.status,
                    projection_status=memory_projection.status,
                    projection_pending=True,
                ),
            )
        return ToolResult(
            "record_decision",
            True,
            f"Recorded decision #{decision_id}: {title}\nSaved note: {path_display}",
            _decision_handoff_metadata(
                "decision_mutation_handoff",
                _decision_mutation_handoff(
                    source="record_decision",
                    mutation="decision_create",
                    decision_id=decision_id,
                    status=decision["status"],
                    changed=["decision"],
                    memory_id=memory_id,
                    path_display=path_display,
                ),
                writes=True,
                decision_id=decision_id,
                status=decision["status"],
                memory_id=memory_id,
                path_display=path_display,
                title_chars=len(title),
                rationale_chars=len(rationale),
                impact_chars=len(impact),
            ),
        )

    def record_decision_outcome(args: dict[str, Any]) -> ToolResult:
        decision_id, summary, reason = _validated_decision_outcome_args(args)
        if reason is not None:
            return _decision_outcome_refusal_result(
                decision_id=decision_id,
                reason=reason,
                raw_decision_id=args.get("decision_id"),
            )
        assert decision_id is not None
        existing_outcomes = store.list_decision_outcomes(
            decision_id, limit=MAX_DECISION_OUTCOMES
        )
        if len(existing_outcomes) >= MAX_DECISION_OUTCOMES:
            return _decision_outcome_refusal_result(
                decision_id=decision_id,
                reason="outcome_limit_reached",
            )
        mutation = store.append_decision_outcome_with_projection(decision_id, summary)
        if mutation is None:
            missing = _missing_decision_result(
                tool_name="record_decision_outcome",
                decision_id=decision_id,
                mutation="decision_outcome_append",
                read_only=False,
                retry_command="decision outcome <correct decision id>: <summary>",
            )
            metadata = dict(missing.metadata)
            metadata.update(
                {
                    "auto_mutation_effects_started": False,
                    "writes_database": False,
                    "provenance": "user_reported",
                    "verification_status": "not_verified",
                    "verified": False,
                    "outcome_label": "user-reported (not verified)",
                    "retryable": False,
                    "authorizes_retry": False,
                }
            )
            return ToolResult(
                missing.tool_name,
                False,
                missing.output
                + " No user-reported (not verified) outcome was recorded.",
                metadata,
            )

        projection = reconcile_decision_projection(
            store, vault, mutation.decision_projection_target
        )
        projection_complete = store.decision_projection_completion(
            mutation.decision_projection_target
        )
        current_projection_complete = (
            projection_complete
            or store.current_decision_projection_completion(decision_id)
        )
        current_decision = store.get_decision(decision_id)
        current_outcomes = store.list_decision_outcomes(
            decision_id, limit=MAX_DECISION_OUTCOMES
        )
        current_outcome = next(
            (
                row
                for row in current_outcomes
                if _row_positive_int(row) == mutation.outcome_id
            ),
            None,
        )
        current_outcome_matches = current_outcome is not None and _decision_outcome_row_matches(
            current_outcome,
            outcome_id=mutation.outcome_id,
            decision_id=decision_id,
            decision_revision=mutation.decision_revision,
            summary=summary,
        )
        current_revision = (
            _row_positive_int(current_decision, "revision")
            if current_decision is not None
            else None
        )
        current_status = (
            _row_short(current_decision, "status", 80, "unknown")
            if current_decision is not None
            else "unknown"
        )
        current_outcome_verified = (
            current_outcome_matches
            and current_revision is not None
            and current_revision >= mutation.decision_revision
        )
        projection_pending = not current_projection_complete
        pending_or_unverified = projection_pending or not current_outcome_verified
        path_display = "" if pending_or_unverified else f"Decisions/decision-{decision_id}.md"
        handoff = _decision_outcome_mutation_handoff(
            decision_id=decision_id,
            status=current_status,
            outcome_id=mutation.outcome_id,
            decision_revision=mutation.decision_revision,
            summary=summary,
            projection_pending=pending_or_unverified,
        )
        metadata = _decision_handoff_metadata(
            "decision_mutation_handoff",
            handoff,
            writes=not pending_or_unverified,
            decision_id=decision_id,
            status=current_status,
            outcome_id=mutation.outcome_id,
            decision_revision=mutation.decision_revision,
            path_display=path_display,
            provenance="user_reported",
            verification_status="not_verified",
            verified=False,
            outcome_label="user-reported (not verified)",
            outcome_current=current_outcome_verified,
            current_decision_revision=current_revision,
            decision_projection_status=projection.status,
            projection_pending=pending_or_unverified,
            writes_database=True,
            writes_memory=False,
            retryable=False,
            authorizes_retry=False,
            do_not_record_again=pending_or_unverified,
        )
        if not current_outcome_verified:
            return ToolResult(
                "record_decision_outcome",
                False,
                f"The user-reported (not verified) outcome for decision #{decision_id} may be "
                "committed, but its current row could not be verified. Run `repair decision "
                "projections` and review the decision; do not record again.",
                metadata,
            )
        if projection_pending:
            return ToolResult(
                "record_decision_outcome",
                True,
                f"Recorded user-reported (not verified) outcome #{mutation.outcome_id} for "
                f"decision #{decision_id}, but its decision projection remains pending. Run "
                "`repair decision projections`; do not record again.",
                metadata,
            )
        return ToolResult(
            "record_decision_outcome",
            True,
            f"Recorded user-reported (not verified) outcome #{mutation.outcome_id} for "
            f"decision #{decision_id}: {summary}\nDecision note updated.",
            metadata,
        )

    def list_decisions(args: dict[str, Any]) -> ToolResult:
        raw_status = args.get("status") or "active"
        status = _short(raw_status, 16).lower()
        if status not in {"active", "superseded", "retired", "all"}:
            return _decision_input_failure(
                "list_decisions",
                "Decision status must be active, superseded, retired, or all.",
                _decision_refusal_metadata(
                    source="list_decisions",
                    mutation="decision_list",
                    reason="bad_status",
                    read_only=True,
                    status=status,
                    raw_status=raw_status,
                ),
            )
        status_filter = None if status == "all" else status
        limit = _bounded_int(args.get("limit"), 25)
        rows = store.list_decisions(status=status_filter, limit=limit)
        handoff = _decision_list_handoff(rows, status=status, limit=limit)
        if not rows:
            return ToolResult(
                "list_decisions",
                True,
                f"No {status} decisions. Count: 0.",
                _decision_handoff_metadata(
                    "decision_list_handoff",
                    handoff,
                    count=0,
                    total_decisions=0,
                    readable_decision_rows=handoff["readable_decision_rows"],
                    unreadable_decision_rows=handoff["unreadable_decision_rows"],
                    status=status,
                    **_raw_limit_metadata(args.get("limit"), limit=limit),
                ),
            )
        lines = ["Decisions:", f"Count: {len(rows)}"]
        unreadable_rows = 0
        for row in rows:
            decision_row_id = _row_positive_int(row)
            title = _row_short(row, "title", MAX_TITLE_CHARS)
            if decision_row_id is None or not title:
                unreadable_rows += 1
                continue
            status_text = _row_short(row, "status", 80, "unknown")
            rationale_text = _row_short(row, "rationale", 120)
            rationale = f": {rationale_text}" if rationale_text else ""
            lines.append(f"- #{decision_row_id} [{status_text}] {title}{rationale}")
        if unreadable_rows:
            lines.append(
                f"- {unreadable_rows} decision row(s) could not be read safely; run `decisions` again after memory repair."
            )
        return ToolResult(
            "list_decisions",
            True,
            "\n".join(lines),
            _decision_handoff_metadata(
                "decision_list_handoff",
                handoff,
                count=len(rows),
                total_decisions=len(rows),
                readable_decision_rows=handoff["readable_decision_rows"],
                unreadable_decision_rows=handoff["unreadable_decision_rows"],
                status=status,
                **_raw_limit_metadata(args.get("limit"), limit=limit),
            ),
        )

    def get_decision(args: dict[str, Any]) -> ToolResult:
        decision_id, error = _positive_id(args.get("decision_id"), "decision_id")
        if decision_id is None:
            return _decision_input_failure(
                "get_decision",
                error or "decision_id is required.",
                _decision_refusal_metadata(
                    source="get_decision",
                    mutation="decision_read",
                    reason="bad_decision_id",
                    read_only=True,
                    decision_id=None,
                    raw_decision_id=args.get("decision_id"),
                ),
            )
        decision = store.get_decision(decision_id)
        if decision is None:
            return _missing_decision_result(
                tool_name="get_decision",
                decision_id=decision_id,
                mutation="decision_read",
                read_only=True,
                retry_command="show decision <correct decision id>",
            )
        outcome_rows = store.list_decision_outcomes(
            decision_id, limit=MAX_DECISION_OUTCOMES
        )
        outcome_views, hidden_outcome_rows, unreadable_outcome_rows = (
            _decision_outcome_views(outcome_rows, decision_id=decision_id)
        )
        handoff = _decision_read_handoff(
            decision,
            requested_id=decision_id,
            outcomes=outcome_views,
            hidden_outcome_rows=hidden_outcome_rows,
            unreadable_outcome_rows=unreadable_outcome_rows,
        )
        if handoff.get("status") == "unreadable":
            return ToolResult(
                "get_decision",
                True,
                f"Decision row for #{decision_id} could not be read safely. Run `decisions` to review memory state.",
                _decision_handoff_metadata(
                    "decision_read_handoff",
                    handoff,
                    decision_id=decision_id,
                    status="unreadable",
                    readable_decision_row=False,
                    unreadable_decision_row=True,
                ),
            )
        decision_row_id = _row_positive_int(decision) or decision_id
        status_text = _row_short(decision, "status", 80, "unknown")
        title = _row_short(decision, "title", MAX_TITLE_CHARS)
        rationale = _row_short(decision, "rationale", MAX_RATIONALE_CHARS)
        impact = _row_short(decision, "impact", MAX_IMPACT_CHARS)
        lines = [
            f"#{decision_row_id} {title} [{status_text}]",
            f"Rationale: {rationale or 'not captured'}",
            f"Impact: {impact or 'not captured'}",
            f"Created: {_row_short(decision, 'created_at', 80)}",
            f"Updated: {_row_short(decision, 'updated_at', 80)}",
            "Outcomes (user-reported, not verified):",
        ]
        if outcome_views:
            for outcome in outcome_views:
                reported_at = f" at {outcome['reported_at']}" if outcome["reported_at"] else ""
                lines.append(
                    f"- Outcome #{outcome['id']}{reported_at} "
                    f"[provenance={outcome['provenance']}]: {outcome['summary']}"
                )
        else:
            lines.append("- None reported.")
        if hidden_outcome_rows:
            lines.append(
                f"- {hidden_outcome_rows} older user-reported outcome row(s) omitted by the read bound."
            )
        if unreadable_outcome_rows:
            lines.append(
                f"- {unreadable_outcome_rows} outcome row(s) could not be rendered safely."
            )
        output = "\n".join(lines)
        return ToolResult(
            "get_decision",
            True,
            output,
            _decision_handoff_metadata(
                "decision_read_handoff",
                handoff,
                decision_id=decision_id,
                status=status_text,
                readable_decision_row=True,
                unreadable_decision_row=False,
                outcome_count=len(outcome_views),
                hidden_outcome_rows=hidden_outcome_rows,
                unreadable_outcome_rows=unreadable_outcome_rows,
                outcomes_bounded=hidden_outcome_rows > 0,
                outcome_label="user-reported (not verified)",
                provenance="user_reported",
                verification_status="not_verified",
                verified=False,
            ),
        )

    def set_decision_status(args: dict[str, Any]) -> ToolResult:
        decision_id, error = _positive_id(args.get("decision_id"), "decision_id")
        if decision_id is None:
            return _decision_input_failure(
                "set_decision_status",
                error or "decision_id is required.",
                _decision_refusal_metadata(
                    source="set_decision_status",
                    mutation="decision_status_update",
                    reason="bad_decision_id",
                    read_only=False,
                    decision_id=None,
                    raw_decision_id=args.get("decision_id"),
                ),
            )
        raw_status = args.get("status")
        status = _short(raw_status, 16).lower()
        if status not in {"active", "superseded", "retired"}:
            return _decision_input_failure(
                "set_decision_status",
                "Status must be active, superseded, or retired.",
                _decision_refusal_metadata(
                    source="set_decision_status",
                    mutation="decision_status_update",
                    reason="bad_status",
                    read_only=False,
                    decision_id=decision_id,
                    status=status,
                    raw_status=raw_status,
                ),
            )
        mutation = store.set_decision_status_with_projection(decision_id, status)
        if mutation is None:
            return _missing_decision_result(
                tool_name="set_decision_status",
                decision_id=decision_id,
                mutation="decision_status_update",
                read_only=False,
                retry_command="decision <correct decision id> <active|superseded|retired>",
                status=status,
            )
        projection = reconcile_decision_projection(
            store, vault, mutation.decision_projection_target
        )
        projection_complete = store.decision_projection_completion(
            mutation.decision_projection_target
        )
        current_row = store.get_decision(decision_id)
        current_status = (
            _row_short(current_row, "status", 80, "unknown")
            if current_row is not None
            else "unknown"
        )
        published_status = str(mutation.row["status"])
        path_display = f"Decisions/decision-{decision_id}.md"
        if projection.status == "superseded" or (
            current_row is not None and current_status != published_status
        ):
            return ToolResult(
                "set_decision_status",
                False,
                f"Decision #{decision_id} changed again before this status receipt completed. "
                f"Current status: {current_status}. Review `show decision {decision_id}`; "
                "the newer state was not overwritten.",
                _decision_handoff_metadata(
                    "decision_mutation_handoff",
                    _decision_mutation_handoff(
                        source="set_decision_status",
                        mutation="decision_status_update",
                        decision_id=decision_id,
                        status=current_status,
                        changed=[],
                        path_display=path_display,
                    ),
                    writes=True,
                    decision_id=decision_id,
                    requested_status=published_status,
                    status=current_status,
                    path_display=path_display,
                    decision_projection_status=projection.status,
                    projection_pending=False,
                    mutation_superseded=True,
                    writes_memory=False,
                ),
            )
        if projection.status != "completed" or not projection_complete:
            pending_handoff = _decision_mutation_handoff(
                source="set_decision_status",
                mutation="decision_status_update",
                decision_id=decision_id,
                status=published_status,
                changed=["status"] if mutation.changed else [],
                path_display="",
            )
            pending_handoff["boundaries"]["writes_files"] = False
            pending_handoff["boundaries"]["writes_notes"] = False
            return ToolResult(
                "set_decision_status",
                True,
                f"Set decision #{decision_id} to {published_status}, but its note is pending. "
                "Run `repair decision projections`; do not repeat the status change.",
                _decision_handoff_metadata(
                    "decision_mutation_handoff",
                    pending_handoff,
                    writes=False,
                    decision_id=decision_id,
                    status=published_status,
                    path_display="",
                    decision_projection_status=projection.status,
                    projection_pending=True,
                    writes_database=True,
                    writes_memory=False,
                    retryable=False,
                    authorizes_retry=False,
                    do_not_repeat=True,
                ),
            )
        return ToolResult(
            "set_decision_status",
            True,
            f"Set decision #{decision_id} to {published_status}.\nSaved note: {path_display}",
            _decision_handoff_metadata(
                "decision_mutation_handoff",
                _decision_mutation_handoff(
                    source="set_decision_status",
                    mutation="decision_status_update",
                    decision_id=decision_id,
                    status=published_status,
                    changed=["status"] if mutation.changed else [],
                    path_display=path_display,
                ),
                writes=True,
                decision_id=decision_id,
                status=published_status,
                path_display=path_display,
                writes_memory=False,
            ),
        )

    return (
        record_decision,
        record_decision_outcome,
        list_decisions,
        get_decision,
        set_decision_status,
    )
