from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sqlite3
import unicodedata
from typing import Any

from jarvis_v2.agent.failure_guidance import declare_failure_guidance
from jarvis_v2.agent.types import ApprovalArgumentResolution, ToolResult
from jarvis_v2.memory.memory_projection import reconcile_memory_projection
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import MemoryStore, TaskRecord
from jarvis_v2.tools.audit_meta import EXECUTION_META_TOOLS


WEAK_MARKERS = [
    "basic inference",
    "suggests that",
    "likely",
    "maybe",
    "might be",
    "i infer",
    "search query sent",
]

MAX_MEMORY_CURATOR_LIMIT = 200
MAX_MEMORY_BODY_CHARS = 50000
MAX_LEARNING_REVIEW_BODY_CHARS = 16384
PERSONAL_CONTEXT_STATUS_LIMIT = 1000
KNOWLEDGE_PROMOTION_TARGET_KINDS = (
    "decision",
    "goal",
    "person",
    "preference",
    "profile",
)
OPERATOR_LIMIT_RULE = (
    "the operator's explicit stop times, work windows, pause commands, and newer instructions "
    "override memory curation, learning queues, destructive memory actions, and priority goals."
)
MEMORY_BOUNDARY = (
    "\n\nMemory curation boundary:\n"
    f"- {OPERATOR_LIMIT_RULE}\n"
    "- Memory deletion, edits, merges, code changes, shell/code, personal data, external actions, "
    "and computer control remain approval-gated."
)
EXECUTION_LEARNING_META_TOOLS = {*EXECUTION_META_TOOLS, "execution_learning_closure_packet"}
LOCAL_PATH_RE = re.compile(
    r"(?:file:///[^\n\r]*|"
    r"~[A-Z0-9._-]*[/\\][^\n\r]*|"
    r"\.\.?[/\\][^\n\r]*|"
    r"[A-Z]:[/\\][^\n\r]*|"
    r"\\\\[^\\/\s]+[/\\][^\n\r]*|"
    r"//[^/\s]+/[^\n\r]*|"
    r"(?<![A-Z0-9:/])/(?!/)[^\n\r]*)",
    re.IGNORECASE,
)
APPROVAL_HOLD_METADATA_KEYS = ("requires_confirmation", "requires_approval", "approval_required")
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
MEMORY_REFUSAL_RECOVERY_ACTION = (
    "Run `show recent memories`, correct the reported memory issue, then retry through the normal policy."
)
MEMORY_READ_RECOVERY_ACTION = (
    "Run `show recent memories`, verify the current record, then retry the read through the normal policy."
)
MEMORY_PROJECTION_RECOVERY_ACTION = (
    "Run `repair memory projections`, inspect the current record, and do not repeat the original mutation."
)


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "calls_model": False,
        "calls_external_service": False,
        "executes_tools": False,
        "reads_personal_data": False,
        "reads_private_data": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "controls_computer": False,
        "speaks": False,
        "completes_tasks": False,
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
    }
    metadata.update(extra)
    return metadata


def _memory_metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _memory_contract(
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


def _memory_handoff_metadata(
    handoff_key: str,
    handoff: dict[str, Any],
    *,
    writes: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    metadata = _safe_metadata(**extra)
    if writes:
        metadata.update({"writes_files": True, "writes_notes": True})
    metadata.update(
        _memory_contract(
            state_changed=_memory_metadata_bool(handoff.get("state_changed")),
            changed=list(handoff.get("changed") or []),
            content_in_handoff=_memory_metadata_bool(handoff.get("content_in_handoff")),
        )
    )
    metadata[f"{handoff_key}_ready"] = True
    metadata[handoff_key] = handoff
    return metadata


def _with_memory_boundary(body: str) -> str:
    return f"{body}{MEMORY_BOUNDARY}"


def _memory_failure_result(
    tool_name: str,
    output: str,
    metadata: dict[str, Any],
    *,
    action: str = MEMORY_REFUSAL_RECOVERY_ACTION,
    commands: tuple[str, ...] = (),
) -> ToolResult:
    public_output = f"{output} {action}"
    bounded_output = _with_memory_boundary(public_output)
    return ToolResult(
        tool_name,
        False,
        bounded_output,
        declare_failure_guidance(
            metadata,
            output=bounded_output,
            action=action,
            commands=commands,
        ),
    )


def _short(value: Any, *, limit: int = 240) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _short_raw(value: Any, *, limit: int = 240) -> str:
    text = "" if value is None else str(value).strip()
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _short_metadata(value: Any, *, limit: int = 240) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", _short_raw(value, limit=limit))


def _learning_review_display(value: Any, *, limit: int = 240) -> str:
    try:
        text = "" if value is None else str(value)
    except Exception:
        text = ""
    text = "".join(
        " " if char.isspace() else "" if unicodedata.category(char) in {"Cc", "Cf"} else char
        for char in text
    )
    text = " ".join(text.strip().split())
    path_marker = "\x00jarvis-local-path\x00"
    text = re.sub(LOCAL_PATH_RE.pattern, path_marker, text, flags=re.IGNORECASE)
    text = (
        text.replace("\\", r"\\")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("`", "'")
        .replace("[", r"\[")
        .replace("]", r"\]")
    )
    text = text.replace(path_marker, "&lt;local-path&gt;")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _select_rotated_knowledge_promotion_rows(
    rows: list[Any], *, limit: int, target_rotation_seed: int
) -> list[Any]:
    groups = {kind: [] for kind in KNOWLEDGE_PROMOTION_TARGET_KINDS}
    for row in rows:
        try:
            target_kind = row["target_kind"]
        except (KeyError, IndexError, TypeError):
            return rows[:limit]
        if type(target_kind) is not str or target_kind not in groups:
            return rows[:limit]
        groups[target_kind].append(row)

    start = target_rotation_seed % len(KNOWLEDGE_PROMOTION_TARGET_KINDS)
    target_order = (
        KNOWLEDGE_PROMOTION_TARGET_KINDS[start:]
        + KNOWLEDGE_PROMOTION_TARGET_KINDS[:start]
    )
    selected: list[Any] = []
    max_target_rows = max((len(group) for group in groups.values()), default=0)
    for target_rank in range(max_target_rows):
        for target_kind in target_order:
            group = groups[target_kind]
            if target_rank < len(group):
                selected.append(group[target_rank])
                if len(selected) == limit:
                    return selected
    return selected


def _safe_vault_path_display(path: str | Path | None, vault: ObsidianVault) -> str:
    if path is None:
        return ""
    candidate = Path(path)
    try:
        return str(candidate.relative_to(vault.root_path))
    except ValueError:
        return _short_metadata(candidate, limit=160)


def _learning_review_boundaries(*, writes: bool) -> dict[str, bool]:
    return {
        "read_only": not writes,
        "reads_personal_data": True,
        "reads_private_data": True,
        "writes_files": writes,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": writes,
        "queues_approval": False,
        "requires_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "controls_computer": False,
        "external_side_effect": False,
        "calls_model": False,
        "executes_tools": False,
        "completes_tasks": False,
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
    }


def _learning_review_handoff(
    *,
    metadata: dict[str, Any],
    writes: bool,
    path_display: str = "",
) -> dict[str, Any]:
    proof_queue = list(metadata.get("execution_learning_proof_queue") or [])
    approval_review_commands = list(metadata.get("execution_learning_approval_review_commands") or [])
    return {
        "source": "save_learning_review" if writes else "learning_review",
        "ready_for_operator": True,
        "limit": metadata.get("limit", 12),
        "feedback": metadata.get("feedback", 0),
        "weak_memories": metadata.get("weak_memories", 0),
        "duplicate_groups": metadata.get("duplicate_groups", 0),
        "preferences": metadata.get("preferences", 0),
        "skills": metadata.get("skills", 0),
        "knowledge_promotion_candidates": metadata.get(
            "knowledge_promotion_candidates", 0
        ),
        "knowledge_promotion_selected_rows": metadata.get(
            "knowledge_promotion_selected_rows", 0
        ),
        "knowledge_promotion_candidate_total": metadata.get(
            "knowledge_promotion_candidate_total", 0
        ),
        "knowledge_promotion_totals_reliable": bool(
            metadata.get("knowledge_promotion_totals_reliable", True)
        ),
        "knowledge_promotion_totals_conflict": bool(
            metadata.get("knowledge_promotion_totals_conflict", False)
        ),
        "knowledge_promotion_rendered_candidates": metadata.get(
            "knowledge_promotion_rendered_candidates", 0
        ),
        "knowledge_promotion_hidden_count": metadata.get(
            "knowledge_promotion_hidden_count", 0
        ),
        "knowledge_promotion_output_hidden_count": metadata.get(
            "knowledge_promotion_output_hidden_count", 0
        ),
        "knowledge_promotion_state": metadata.get(
            "knowledge_promotion_state", ""
        ),
        "knowledge_promotion_unreadable_rows": metadata.get(
            "knowledge_promotion_unreadable_rows", 0
        ),
        "knowledge_promotion_classification_basis": metadata.get(
            "knowledge_promotion_classification_basis", "category_label_only"
        ),
        "knowledge_promotion_semantic_proof": False,
        "knowledge_promotion_authorizes_promotion": False,
        "knowledge_promotion_rotation_seed": metadata.get(
            "knowledge_promotion_rotation_seed", 0
        ),
        "knowledge_promotion_candidate_ids": list(
            metadata.get("knowledge_promotion_candidate_ids") or []
        ),
        "knowledge_promotion_candidate_bindings": list(
            metadata.get("knowledge_promotion_candidate_bindings") or []
        ),
        "knowledge_promotion_target_counts": dict(
            metadata.get("knowledge_promotion_target_counts") or {}
        ),
        "knowledge_promotion_review_commands": list(
            metadata.get("knowledge_promotion_review_commands") or []
        ),
        "recent_tool_runs": metadata.get("recent_tool_runs", 0),
        "recent_action_runs": metadata.get("recent_action_runs", 0),
        "failed_or_blocked_action_runs": metadata.get("failed_or_blocked_action_runs", 0),
        "approval_held_action_runs": metadata.get("approval_held_action_runs", 0),
        "execution_learning_state": metadata.get("execution_learning_state", ""),
        "execution_learning_blocks_completion_claim": bool(metadata.get("execution_learning_blocks_completion_claim")),
        "execution_learning_target_run_id": metadata.get("execution_learning_target_run_id"),
        "execution_learning_target_tool": _short_metadata(metadata.get("execution_learning_target_tool"), limit=120),
        "execution_learning_missing": list(metadata.get("execution_learning_missing") or []),
        "execution_learning_missing_count": metadata.get("execution_learning_missing_count", 0),
        "execution_learning_approval_review_commands": approval_review_commands,
        "execution_learning_proof_queue": proof_queue,
        "execution_learning_proof_queue_count": len(proof_queue),
        "execution_learning_next_required_command": proof_queue[0] if proof_queue else "",
        "execution_learning_next_proof_command": proof_queue[0] if proof_queue else "",
        "path_display": _short_metadata(path_display, limit=160),
        "content_in_handoff": False,
        "next_commands": proof_queue
        or list(metadata.get("knowledge_promotion_review_commands") or [])
        or ["save learning review", "queue learning tasks", "weak memories", "duplicate memories"],
        "boundaries": _learning_review_boundaries(writes=writes),
        **_memory_contract(
            state_changed=writes,
            changed=["learning_review_export"] if writes else [],
            content_in_handoff=False,
        ),
    }


def _queue_learning_tasks_handoff(
    *,
    added: list[tuple[int, str]],
    skipped: list[str],
    limit: int,
    path_display: str,
) -> dict[str, Any]:
    added_rows = [
        {
            "task_id": task_id,
            "body_preview": _short_metadata(body, limit=180),
        }
        for task_id, body in added
    ]
    skipped_rows = [_short_metadata(body, limit=180) for body in skipped[:20]]
    return {
        "source": "queue_learning_tasks",
        "ready_for_operator": True,
        "added": len(added),
        "skipped": len(skipped),
        "limit": limit,
        "path_display": _short_metadata(path_display, limit=160),
        "added_tasks": added_rows,
        "added_task_ids": [row["task_id"] for row in added_rows],
        "skipped_task_previews": skipped_rows,
        "next_commands": [
            "tasks",
            "learning review",
            "save learning review",
            "weak memories",
            "duplicate memories",
        ],
        "boundaries": {
            "read_only": False,
            "writes_files": True,
            "writes_database": bool(added),
            "writes_memory": bool(added),
            "writes_notes": True,
            "queues_approval": False,
            "requires_approval": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "controls_computer": False,
            "external_side_effect": False,
            "calls_model": False,
            "executes_tools": False,
            "completes_tasks": False,
            "operator_timeboxes_override_priority": True,
            "stop_times_override_priority": True,
        },
        **_memory_contract(
            state_changed=True,
            changed=["learning_tasks"] if added else ["task_export"],
            content_in_handoff=bool(added or skipped),
        ),
    }


CONVERSATION_COMPACTION_LEGACY_REVIEW_LIMIT = 20
CONVERSATION_COMPACTION_LEGACY_REVIEW_COUNT_KEYS = (
    "total_digest_rows",
    "linked_digest_rows",
    "unlinked_legacy_rows",
    "canonical_unlinked_groups",
    "preserved_duplicate_rows",
    "conflicting_groups",
    "unparseable_rows",
)
CONVERSATION_COMPACTION_LEGACY_REVIEW_KEYS = {
    *CONVERSATION_COMPACTION_LEGACY_REVIEW_COUNT_KEYS,
    "review_required",
    "review_memory_ids",
    "review_memory_ids_truncated",
    "watermark",
}


def _conversation_compaction_legacy_review_unavailable() -> dict[str, Any]:
    return {
        "available": False,
        "limit": CONVERSATION_COMPACTION_LEGACY_REVIEW_LIMIT,
        "review_required": False,
        "review_memory_ids": [],
        "review_memory_ids_truncated": False,
        "next_command": "",
        "read_only": True,
        "ranges_inferred": False,
        "memory_edited": False,
        "memory_deleted": False,
        "repair_performed": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": False,
        "calls_model": False,
        "executes_tools": False,
        "external_side_effect": False,
        "controls_computer": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _safe_conversation_compaction_legacy_review(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != CONVERSATION_COMPACTION_LEGACY_REVIEW_KEYS:
        return None

    counts: dict[str, int] = {}
    for key in CONVERSATION_COMPACTION_LEGACY_REVIEW_COUNT_KEYS:
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            return None
        counts[key] = item

    watermark = value.get("watermark")
    review_required = value.get("review_required")
    ids_truncated = value.get("review_memory_ids_truncated")
    memory_ids = value.get("review_memory_ids")
    if isinstance(watermark, bool) or not isinstance(watermark, int) or watermark < 0:
        return None
    if not isinstance(review_required, bool) or not isinstance(ids_truncated, bool):
        return None
    if not isinstance(memory_ids, list) or len(memory_ids) > CONVERSATION_COMPACTION_LEGACY_REVIEW_LIMIT:
        return None
    if any(isinstance(memory_id, bool) or not isinstance(memory_id, int) or memory_id <= 0 for memory_id in memory_ids):
        return None
    if review_required and not memory_ids:
        return None

    next_command = f"show memory {memory_ids[0]}" if review_required else ""
    return {
        "available": True,
        "limit": CONVERSATION_COMPACTION_LEGACY_REVIEW_LIMIT,
        **counts,
        "review_required": review_required,
        "review_memory_ids": list(memory_ids),
        "review_memory_ids_truncated": ids_truncated,
        "watermark": watermark,
        "next_command": next_command,
        "read_only": True,
        "ranges_inferred": False,
        "memory_edited": False,
        "memory_deleted": False,
        "repair_performed": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": False,
        "calls_model": False,
        "executes_tools": False,
        "external_side_effect": False,
        "controls_computer": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _conversation_compaction_legacy_review_lines(review: dict[str, Any]) -> list[str]:
    if not review.get("available"):
        return [
            "Conversation compaction legacy review: unavailable.",
            "- No ranges are inferred. No memory is edited or deleted. No repair is performed.",
            "- Manual next command: none.",
        ]

    memory_ids = list(review.get("review_memory_ids") or [])
    lines = ["Conversation compaction legacy review (limit 20):"]
    for key in CONVERSATION_COMPACTION_LEGACY_REVIEW_COUNT_KEYS:
        lines.append(f"- {key}: {review[key]}")
    lines.extend(
        [
            f"- review_required: {str(review['review_required']).lower()}",
            f"- review_memory_ids: {', '.join(str(memory_id) for memory_id in memory_ids) if memory_ids else 'none'}",
            f"- review_memory_ids_truncated: {str(review['review_memory_ids_truncated']).lower()}",
            f"- watermark: {review['watermark']}",
            "- No ranges are inferred. No memory is edited or deleted. No repair is performed.",
            f"- Manual next command: `{review['next_command']}`." if review["next_command"] else "- Manual next command: none.",
        ]
    )
    return lines


def _memory_stats_handoff(
    *,
    rows: list[dict[str, Any]],
    conversation_compaction_legacy_review: dict[str, Any],
) -> dict[str, Any]:
    stat_rows = [
        {
            "category": _short_metadata(row.get("category"), limit=80),
            "source": _short_metadata(row.get("source"), limit=80),
            "count": int(row.get("count") or 0),
            "latest": _short_metadata(row.get("latest"), limit=80),
        }
        for row in rows
    ]
    review_has_content = bool(
        conversation_compaction_legacy_review.get("available")
        and (
            conversation_compaction_legacy_review.get("total_digest_rows")
            or conversation_compaction_legacy_review.get("watermark")
            or conversation_compaction_legacy_review.get("review_required")
        )
    )
    return {
        "source": "memory_stats",
        "ready_for_operator": True,
        "count": len(stat_rows),
        "stats": stat_rows,
        "total_memories": sum(row["count"] for row in stat_rows),
        "conversation_compaction_legacy_review": conversation_compaction_legacy_review,
        "next_commands": [
            "memory tree summary",
            "learning review",
            "weak memories",
            "duplicate memories",
        ],
        "boundaries": {
            "read_only": True,
            "writes_files": False,
            "writes_database": False,
            "writes_memory": False,
            "writes_notes": False,
            "queues_approval": False,
            "requires_approval": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "controls_computer": False,
            "external_side_effect": False,
            "calls_model": False,
            "executes_tools": False,
            "operator_timeboxes_override_priority": True,
            "stop_times_override_priority": True,
        },
        **_memory_contract(
            content_in_handoff=bool(stat_rows) or review_has_content
        ),
    }


def _safe_memory_stat_row(row: Any) -> dict[str, Any] | None:
    try:
        category = _short_metadata(row["category"], limit=80)
        source = _short_metadata(row["source"], limit=80)
        count = int(row["count"] or 0)
        latest = _short_metadata(row["latest"], limit=80)
    except Exception:
        return None
    return {
        "category": category,
        "source": source,
        "count": count,
        "latest": latest,
    }


def _memory_tree_summary_handoff(
    *,
    grouped: dict[str, list[Any]],
    limit: int,
    path_display: str,
    draft_skill_count: int,
    writes: bool,
) -> dict[str, Any]:
    category_rows = [
        {
            "category": _short_metadata(category, limit=80),
            "count": len(items),
            "sample_memory_ids": [int(row["id"]) for row in items[:5]],
        }
        for category, items in sorted(grouped.items())
    ]
    return {
        "source": "memory_tree_summary",
        "ready_for_operator": True,
        "count": sum(row["count"] for row in category_rows),
        "limit": limit,
        "path_display": _short_metadata(path_display, limit=160),
        "categories": category_rows,
        "category_count": len(category_rows),
        "draft_skill_count": draft_skill_count,
        "human_review_required_for_skill_promotion": True,
        "self_modifying_code": False,
        "content_in_handoff": False,
        "next_commands": [
            "read jarvis note "
            + (_short_metadata(path_display, limit=160) if path_display else "Memory Tree/Memory Tree Snapshot.md"),
            "memory stats",
            "learning review",
            "queue learning tasks",
        ],
        "boundaries": {
            "read_only": not writes,
            "writes_files": writes,
            "writes_database": False,
            "writes_memory": False,
            "writes_notes": writes,
            "queues_approval": False,
            "requires_approval": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "controls_computer": False,
            "external_side_effect": False,
            "calls_model": False,
            "executes_tools": False,
            "operator_timeboxes_override_priority": True,
            "stop_times_override_priority": True,
        },
        **_memory_contract(
            state_changed=writes,
            changed=["memory_tree_summary_export"] if writes else [],
            content_in_handoff=False,
        ),
    }


def _row_metadata(row: Any) -> dict[str, Any]:
    try:
        if "metadata" not in row.keys():
            return {}
        metadata = row["metadata"]
        if isinstance(metadata, dict):
            return dict(metadata)
        parsed = json.loads(metadata or "{}")
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _metadata_truthy_loose(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _normalized_metadata_token(value: Any) -> str:
    try:
        text = "" if value is None else str(value)
    except Exception:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def _row_int(row: Any, key: str) -> int | None:
    try:
        value = row[key]
    except Exception:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _row_metadata_positive_int(row: Any, key: str) -> int | None:
    value = _row_metadata(row).get(key)
    return value if type(value) is int and value > 0 else None


def _ids_from_text(value: Any, markers: tuple[str, ...]) -> list[int]:
    text = str(value or "")
    ids: list[int] = []
    seen: set[int] = set()
    for marker in markers:
        marker_pattern = re.escape(marker).replace(r"\ ", r"\s+")
        pattern = re.compile(rf"\b{marker_pattern}\s*#?\s*(\d+)\b", re.IGNORECASE)
        for match in pattern.finditer(text):
            candidate = int(match.group(1))
            if candidate not in seen:
                ids.append(candidate)
                seen.add(candidate)
    return ids


def _is_approval_held_tool_run(row: Any) -> bool:
    try:
        if bool(row["ok"]):
            return False
    except Exception:
        return False
    metadata = _row_metadata(row)
    if any(_metadata_truthy_loose(metadata.get(key)) for key in APPROVAL_HOLD_METADATA_KEYS):
        return True
    for key in ("failure_kind", "failure_stage", "stage", "guard_reason", "reason", "status", "send_status", "call_status"):
        if _normalized_metadata_token(metadata.get(key)) in APPROVAL_HOLD_VALUES:
            return True
    return False


def _approval_id_for_run(row: Any) -> int | None:
    candidates: list[Any] = []
    row_approval_id = _row_int(row, "approval_id")
    if row_approval_id is not None:
        candidates.append(row_approval_id)
    metadata = _row_metadata(row)
    candidates.append(metadata.get("approval_id"))
    try:
        candidates.extend(
            _ids_from_text(
                row["output"],
                ("approval #", "approval id", "approval packet", "queued as approval"),
            )
        )
    except Exception:
        pass
    for candidate in candidates:
        if isinstance(candidate, bool):
            continue
        try:
            parsed = int(candidate)
        except (TypeError, ValueError, OverflowError):
            continue
        if parsed > 0:
            return parsed
    return None


def _approval_review_commands_for_run(row: Any) -> list[str]:
    approval_id = _approval_id_for_run(row)
    if approval_id is None:
        return ["approval readiness latest", "approval packet latest", "approval chain proof latest"]
    return [f"approval readiness {approval_id}", f"approval packet {approval_id}", f"approval chain proof {approval_id}"]


def _bounded_int(value: Any, default: int, low: int = 1, high: int = MAX_MEMORY_CURATOR_LIMIT) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def queue_learning_tasks_auto_mutation_operation_key(
    _args: dict[str, Any],
) -> dict[str, str]:
    """Fence every learning-queue projection behind one logical operation target."""
    return {"target": "learning_task_queue"}


def _positive_id(value: Any, label: str) -> tuple[int | None, str | None]:
    if isinstance(value, bool):
        return None, f"{label} must be a number."
    try:
        item_id = int(value)
    except (TypeError, ValueError, OverflowError):
        return None, f"{label} must be a number."
    if not isinstance(value, str):
        try:
            is_exact_integer = value == item_id
        except Exception:
            is_exact_integer = False
        if is_exact_integer is not True:
            return None, f"{label} must be a number."
    if item_id <= 0:
        return None, f"{label} is required."
    return item_id, None


def _bad_memory_id_metadata(value: Any, *, requires_approval: bool = False) -> dict[str, Any]:
    return _safe_metadata(memory_id=None, raw_memory_id=_short_metadata(value, limit=80), requires_approval=requires_approval)


def _memory_refusal_boundaries(*, read_only: bool, requires_approval: bool) -> dict[str, bool]:
    return {
        "read_only": read_only,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": requires_approval,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "controls_computer": False,
        "external_side_effect": False,
        "calls_model": False,
        "executes_tools": False,
        "completes_tasks": False,
        "edits_memory": False,
        "deletes_memory": False,
        "merges_memory": False,
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
    }


def _memory_refusal_handoff(
    *,
    source: str,
    mutation: str,
    reason: str,
    read_only: bool,
    requires_approval: bool,
    memory_id: int | None = None,
    keep_id: int | None = None,
    delete_id: int | None = None,
    raw_memory_id: str | None = None,
    raw_keep_id: str | None = None,
    raw_delete_id: str | None = None,
    raw_confidence: str | None = None,
) -> dict[str, Any]:
    handoff: dict[str, Any] = {
        "source": source,
        "ready_for_operator": True,
        "refused": True,
        "mutation": mutation,
        "reason": reason,
        "memory_id": memory_id,
        "keep_id": keep_id,
        "delete_id": delete_id,
        "changed": [],
        "next_commands": {
            "retry": f"retry {source} with a valid memory id",
            "search": "search memory for the relevant title or detail",
            "recent": "show recent memories",
        },
        "boundaries": _memory_refusal_boundaries(read_only=read_only, requires_approval=requires_approval),
        **_memory_contract(content_in_handoff=any(value is not None for value in (raw_memory_id, raw_keep_id, raw_delete_id, raw_confidence))),
    }
    if raw_memory_id is not None:
        handoff["raw_memory_id"] = raw_memory_id
    if raw_keep_id is not None:
        handoff["raw_keep_id"] = raw_keep_id
    if raw_delete_id is not None:
        handoff["raw_delete_id"] = raw_delete_id
    if raw_confidence is not None:
        handoff["raw_confidence"] = raw_confidence
    return handoff


def _memory_refusal_metadata(
    *,
    source: str,
    mutation: str,
    reason: str,
    read_only: bool = False,
    requires_approval: bool = True,
    memory_id: int | None = None,
    keep_id: int | None = None,
    delete_id: int | None = None,
    raw_memory_id: Any = None,
    raw_keep_id: Any = None,
    raw_delete_id: Any = None,
    raw_confidence: Any = None,
    **extra: Any,
) -> dict[str, Any]:
    safe_raw_memory_id = _short_metadata(raw_memory_id, limit=80) if raw_memory_id is not None else None
    safe_raw_keep_id = _short_metadata(raw_keep_id, limit=80) if raw_keep_id is not None else None
    safe_raw_delete_id = _short_metadata(raw_delete_id, limit=80) if raw_delete_id is not None else None
    safe_raw_confidence = _short_metadata(raw_confidence, limit=80) if raw_confidence is not None else None
    metadata = _safe_metadata(
        memory_id=memory_id,
        keep_id=keep_id,
        delete_id=delete_id,
        requires_approval=requires_approval,
        memory_refusal_handoff_ready=True,
        **extra,
    )
    if safe_raw_memory_id is not None:
        metadata["raw_memory_id"] = safe_raw_memory_id
    if safe_raw_keep_id is not None:
        metadata["raw_keep_id"] = safe_raw_keep_id
    if safe_raw_delete_id is not None:
        metadata["raw_delete_id"] = safe_raw_delete_id
    if safe_raw_confidence is not None:
        metadata["raw_confidence"] = safe_raw_confidence
    metadata["memory_refusal_handoff"] = _memory_refusal_handoff(
        source=source,
        mutation=mutation,
        reason=reason,
        read_only=read_only,
        requires_approval=requires_approval,
        memory_id=memory_id,
        keep_id=keep_id,
        delete_id=delete_id,
        raw_memory_id=safe_raw_memory_id,
        raw_keep_id=safe_raw_keep_id,
        raw_delete_id=safe_raw_delete_id,
        raw_confidence=safe_raw_confidence,
    )
    handoff = metadata["memory_refusal_handoff"]
    metadata.update(
        _memory_contract(
            state_changed=False,
            changed=[],
            content_in_handoff=_memory_metadata_bool(handoff.get("content_in_handoff")),
        )
    )
    return metadata


def _missing_memory_result(
    *,
    tool_name: str,
    memory_id: int,
    mutation: str,
    requires_approval: bool,
    retry_command: str = "",
    retry_action: str = "",
) -> ToolResult:
    verify_command = "show memory <correct memory id>"
    recovery_commands = [
        "show recent memories",
        "search memory for <title or detail>",
        verify_command,
    ]
    if retry_command and retry_command not in recovery_commands:
        recovery_commands.append(retry_command)
    output = (
        f"No memory found with id #{memory_id}. Run `show recent memories` or "
        "`search memory for <title or detail>` to refresh memory IDs, then "
        f"`{verify_command}` to verify the correct record."
    )
    if retry_command and retry_command != verify_command:
        policy = "normal approval flow" if requires_approval else "normal read-only policy"
        output += f" After that, run `{retry_command}` through the {policy}."
    elif retry_action:
        output += f" After that, {retry_action} through the normal approval flow."
    else:
        output += " Use the verified command through the normal read-only policy."
    if requires_approval:
        output += " The failed request does not reuse or grant approval."
    metadata = _memory_refusal_metadata(
        source=tool_name,
        mutation=mutation,
        reason="missing_memory",
        read_only=not requires_approval,
        requires_approval=requires_approval,
        memory_id=memory_id,
    )
    metadata.update(
        {
            "reason": "missing_memory",
            "next_command": "show recent memories",
            "recovery_commands": recovery_commands,
            "retry_action": retry_action,
            "retry_requires_memory_refresh": True,
            "retry_requires_corrected_id": True,
            "retry_requires_fresh_approval": requires_approval,
            "recovery_commands_require_normal_policy": True,
            "authorizes_retry": False,
            "authorizes_memory_mutation": False,
        }
    )
    return _memory_failure_result(tool_name, output, metadata)


def _memory_approval_resolution_refusal(
    *,
    tool_name: str,
    mutation: str,
    reason: str,
    output: str,
    memory_id: int | None = None,
    keep_id: int | None = None,
    delete_id: int | None = None,
) -> ToolResult:
    return _memory_failure_result(
        tool_name,
        output,
        _memory_refusal_metadata(
            source=tool_name,
            mutation=mutation,
            reason=reason,
            memory_id=memory_id,
            keep_id=keep_id,
            delete_id=delete_id,
            requires_approval=True,
            requires_confirmation=False,
            executed_handler=False,
            handler_invoked=False,
            authorizes_execution=False,
            approval_granted=False,
            approval_argument_resolution_status=reason,
        ),
    )


def _stale_memory_binding_result(
    *,
    tool_name: str,
    mutation: str,
    memory_id: int | None = None,
    keep_id: int | None = None,
    delete_id: int | None = None,
    target_status: str = "stale",
) -> ToolResult:
    return _memory_failure_result(
        tool_name,
        "The approved memory target changed or no longer exists. Nothing was changed; "
        "inspect the current memory rows and issue a fresh request if still intended.",
        _memory_refusal_metadata(
            source=tool_name,
            mutation=mutation,
            reason="stale_memory_binding",
            memory_id=memory_id,
            keep_id=keep_id,
            delete_id=delete_id,
            requires_approval=True,
            destructive=True,
            target_binding_status=target_status,
            approval_rerun_blocked=True,
            approval_rerun_block_reason="memory_target_changed",
        ),
    )


def make_memory_approval_resolvers(store: MemoryStore):
    def resolve_delete_memory_approval(
        args: dict[str, Any],
    ) -> ApprovalArgumentResolution | ToolResult:
        memory_id, error = _positive_id(args.get("memory_id"), "memory_id")
        if memory_id is None:
            return _memory_approval_resolution_refusal(
                tool_name="delete_memory",
                mutation="memory_delete",
                reason="invalid_memory_id",
                output=error or "memory_id is required.",
            )
        target = store.resolve_memory_approval_target(memory_id)
        if target is None:
            return _memory_approval_resolution_refusal(
                tool_name="delete_memory",
                mutation="memory_delete",
                reason="not_found",
                output=f"No memory found with id #{memory_id}, so no approval was queued.",
                memory_id=memory_id,
            )
        return ApprovalArgumentResolution(
            {
                "memory_id": memory_id,
                "target_revision": target.revision,
                "target_binding": target.binding,
            },
            {
                "approval_argument_resolution_status": "resolved",
                "memory_id": memory_id,
                "target_revision": target.revision,
                "memory_target_bound": True,
            },
        )

    def resolve_edit_memory_approval(
        args: dict[str, Any],
    ) -> ApprovalArgumentResolution | ToolResult:
        memory_id, error = _positive_id(args.get("memory_id"), "memory_id")
        if memory_id is None:
            return _memory_approval_resolution_refusal(
                tool_name="edit_memory",
                mutation="memory_edit",
                reason="invalid_memory_id",
                output=error or "memory_id is required.",
            )
        if "body" in args and len(str(args["body"])) > MAX_MEMORY_BODY_CHARS:
            return _memory_approval_resolution_refusal(
                tool_name="edit_memory",
                mutation="memory_edit",
                reason="memory_body_too_large",
                output=f"Memory body exceeds the {MAX_MEMORY_BODY_CHARS}-character limit.",
                memory_id=memory_id,
            )
        target = store.resolve_memory_approval_target(memory_id)
        if target is None:
            return _memory_approval_resolution_refusal(
                tool_name="edit_memory",
                mutation="memory_edit",
                reason="not_found",
                output=f"No memory found with id #{memory_id}, so no approval was queued.",
                memory_id=memory_id,
            )
        bound: dict[str, Any] = {
            "memory_id": memory_id,
            "target_revision": target.revision,
            "target_binding": target.binding,
        }
        for field in ("category", "title", "body"):
            value = args.get(field)
            if isinstance(value, str) and value:
                bound[field] = value
        if "confidence" in args:
            confidence = float(args["confidence"])
            if not math.isfinite(confidence):
                return _memory_approval_resolution_refusal(
                    tool_name="edit_memory",
                    mutation="memory_edit",
                    reason="invalid_confidence",
                    output="confidence must be finite.",
                    memory_id=memory_id,
                )
            bound["confidence"] = max(0.0, min(1.0, confidence))
        if len(bound) == 3:
            return _memory_approval_resolution_refusal(
                tool_name="edit_memory",
                mutation="memory_edit",
                reason="no_changes",
                output="No memory fields were supplied to edit, so no approval was queued.",
                memory_id=memory_id,
            )
        return ApprovalArgumentResolution(
            bound,
            {
                "approval_argument_resolution_status": "resolved",
                "memory_id": memory_id,
                "target_revision": target.revision,
                "memory_target_bound": True,
                "memory_edit_fields": sorted(set(bound).intersection({"category", "title", "body", "confidence"})),
            },
        )

    def resolve_merge_memories_approval(
        args: dict[str, Any],
    ) -> ApprovalArgumentResolution | ToolResult:
        keep_id, keep_error = _positive_id(args.get("keep_id"), "keep_id")
        delete_id, delete_error = _positive_id(args.get("delete_id"), "delete_id")
        if keep_id is None or delete_id is None:
            return _memory_approval_resolution_refusal(
                tool_name="merge_memories",
                mutation="memory_merge",
                reason="invalid_memory_id",
                output=keep_error or delete_error or "keep_id and delete_id are required.",
                keep_id=keep_id,
                delete_id=delete_id,
            )
        if keep_id == delete_id:
            return _memory_approval_resolution_refusal(
                tool_name="merge_memories",
                mutation="memory_merge",
                reason="same_memory",
                output="keep_id and delete_id must be different memories.",
                keep_id=keep_id,
                delete_id=delete_id,
            )
        targets = store.resolve_memory_merge_approval_targets(keep_id, delete_id)
        if targets is None:
            return _memory_approval_resolution_refusal(
                tool_name="merge_memories",
                mutation="memory_merge",
                reason="not_found",
                output="One or both memory rows no longer exist, so no approval was queued.",
                keep_id=keep_id,
                delete_id=delete_id,
            )
        keep, delete = targets
        return ApprovalArgumentResolution(
            {
                "keep_id": keep_id,
                "keep_revision": keep.revision,
                "keep_binding": keep.binding,
                "delete_id": delete_id,
                "delete_revision": delete.revision,
                "delete_binding": delete.binding,
            },
            {
                "approval_argument_resolution_status": "resolved",
                "keep_id": keep_id,
                "keep_revision": keep.revision,
                "delete_id": delete_id,
                "delete_revision": delete.revision,
                "memory_targets_bound": True,
            },
        )

    return (
        resolve_delete_memory_approval,
        resolve_edit_memory_approval,
        resolve_merge_memories_approval,
    )


def _missing_memory_merge_result(*, keep_id: int, delete_id: int) -> ToolResult:
    retry_command = "merge memory <correct delete memory id> into <correct keep memory id>"
    recovery_commands = [
        "show recent memories",
        "search memory for <title or detail>",
        "show memory <correct keep memory id>",
        "show memory <correct delete memory id>",
        retry_command,
    ]
    output = (
        "One or both memory IDs were not found. Run `show recent memories` or "
        "`search memory for <title or detail>` to refresh memory IDs, verify both corrected records, "
        f"then run `{retry_command}` through the normal approval flow. "
        "The failed request does not reuse or grant approval."
    )
    metadata = _memory_refusal_metadata(
        source="merge_memories",
        mutation="memory_merge",
        reason="missing_memory",
        read_only=False,
        requires_approval=True,
        keep_id=keep_id,
        delete_id=delete_id,
    )
    metadata.update(
        {
            "reason": "missing_memory",
            "next_command": "show recent memories",
            "recovery_commands": recovery_commands,
            "retry_requires_memory_refresh": True,
            "retry_requires_corrected_id": True,
            "retry_requires_fresh_approval": True,
            "recovery_commands_require_normal_policy": True,
            "authorizes_retry": False,
            "authorizes_memory_mutation": False,
        }
    )
    return _memory_failure_result("merge_memories", output, metadata)


def _failed_memory_merge_result(*, keep_id: int, delete_id: int) -> ToolResult:
    output = (
        "Memory merge failed before an atomic change could be committed. No memory changes were claimed. "
        "Run `show recent memories` and verify both records before requesting a fresh merge through the "
        "normal approval flow. The failed request does not reuse or grant approval."
    )
    metadata = _memory_refusal_metadata(
        source="merge_memories",
        mutation="memory_merge",
        reason="merge_failed",
        read_only=False,
        requires_approval=True,
        keep_id=keep_id,
        delete_id=delete_id,
    )
    metadata.update(
        {
            "reason": "merge_failed",
            "next_command": "show recent memories",
            "recovery_commands": [
                "show recent memories",
                "show memory <correct keep memory id>",
                "show memory <correct delete memory id>",
                "merge memory <correct delete memory id> into <correct keep memory id>",
            ],
            "retry_requires_memory_refresh": True,
            "retry_requires_fresh_approval": True,
            "authorizes_retry": False,
            "authorizes_memory_mutation": False,
        }
    )
    return _memory_failure_result("merge_memories", output, metadata)


def make_memory_curator_tools(store: MemoryStore, vault: ObsidianVault):
    def reconcile_projection_target(target):
        outcome = reconcile_memory_projection(
            store,
            vault,
            target.memory_id,
            expected_operation=target.operation,
            expected_revision=target.revision,
            expected_source_digest=target.source_digest,
        )
        if outcome.status == "superseded":
            return reconcile_memory_projection(store, vault, target.memory_id)
        return outcome

    def _fingerprint(value: str) -> str:
        words = re.findall(r"[a-z0-9]+", value.lower())
        return " ".join(words[:16])

    def _weak_rows(limit: int):
        rows = store.list_memories(limit=200)
        weak = []
        for row in rows:
            blob = f"{row['title']} {row['body']}".lower()
            if any(marker in blob for marker in WEAK_MARKERS):
                weak.append(row)
            if len(weak) >= limit:
                break
        return weak

    def _duplicate_groups(limit: int):
        rows = store.list_memories(limit=limit)
        groups = defaultdict(list)
        for row in rows:
            key = _fingerprint(f"{row['category']} {row['title']} {row['body']}")
            if key:
                groups[key].append(row)
        return [items for items in groups.values() if len(items) > 1]

    def list_weak_memories(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 25)
        weak = _weak_rows(limit)
        if not weak:
            return ToolResult(
                "list_weak_memories",
                True,
                _with_memory_boundary("No weak-looking memories found."),
                _safe_metadata(count=0, limit=limit),
            )
        lines = [f"- #{row['id']} [{row['category']}] {row['title']}: {row['body'][:160]}" for row in weak]
        return ToolResult(
            "list_weak_memories",
            True,
            _with_memory_boundary("\n".join(lines)),
            _safe_metadata(count=len(weak), limit=limit),
        )

    def delete_memory(args: dict[str, Any]) -> ToolResult:
        memory_id, error = _positive_id(args.get("memory_id"), "memory_id")
        if memory_id is None:
            return _memory_failure_result(
                "delete_memory",
                error or "memory_id is required.",
                _memory_refusal_metadata(
                    source="delete_memory",
                    mutation="memory_delete",
                    reason=error or "memory_id is required.",
                    raw_memory_id=args.get("memory_id"),
                ),
            )
        target_revision = args.get("target_revision")
        target_binding = args.get("target_binding")
        if type(target_revision) is not int or target_revision < 1 or type(target_binding) is not str:
            if store.get_memory(memory_id) is None:
                return _missing_memory_result(
                    tool_name="delete_memory",
                    memory_id=memory_id,
                    mutation="memory_delete",
                    requires_approval=True,
                    retry_command="delete memory <correct memory id>",
                )
            return _stale_memory_binding_result(
                tool_name="delete_memory",
                mutation="memory_delete",
                memory_id=memory_id,
                target_status="unbound",
            )
        deleted = store.delete_memory_exact_with_projection(
            memory_id,
            target_revision,
            target_binding,
        )
        if deleted.status == "protected":
            return _memory_failure_result(
                "delete_memory",
                f"Memory #{memory_id} is owned by a curated profile note and was not deleted. "
                "Profile notes need their own removal workflow so both copies are removed together.",
                _safe_metadata(
                    memory_id=memory_id,
                    target_status="protected",
                    writes_database=False,
                    writes_memory=False,
                    requires_approval=True,
                ),
            )
        if deleted.status != "deleted":
            return _stale_memory_binding_result(
                tool_name="delete_memory",
                mutation="memory_delete",
                memory_id=memory_id,
                target_status=deleted.status,
            )
        projections = [reconcile_projection_target(target) for target in deleted.projection_targets]
        projection_statuses = [projection.status for projection in projections]
        if not projections or any(status != "completed" for status in projection_statuses):
            return _memory_failure_result(
                "delete_memory",
                f"Deleted memory #{memory_id}, but its note projection remains pending. "
                "Run `repair memory projections`; do not delete it again.",
                _safe_metadata(
                    memory_id=memory_id,
                    projection_statuses=projection_statuses or ["missing"],
                    projection_pending=True,
                    writes_database=True,
                    writes_memory=True,
                    requires_approval=True,
                ),
                action=MEMORY_PROJECTION_RECOVERY_ACTION,
                commands=("repair memory projections",),
            )
        return ToolResult(
            "delete_memory",
            True,
            _with_memory_boundary(f"Deleted memory #{memory_id}."),
            _safe_metadata(memory_id=memory_id, writes_database=True, writes_memory=True, requires_approval=True),
        )

    def get_memory(args: dict[str, Any]) -> ToolResult:
        memory_id, error = _positive_id(args.get("memory_id"), "memory_id")
        if memory_id is None:
            return _memory_failure_result(
                "get_memory",
                error or "memory_id is required.",
                _memory_refusal_metadata(
                    source="get_memory",
                    mutation="memory_read",
                    reason=error or "memory_id is required.",
                    read_only=True,
                    requires_approval=False,
                    raw_memory_id=args.get("memory_id"),
                ),
                action=MEMORY_READ_RECOVERY_ACTION,
            )
        with store.memory_read_custody(memory_id) as custody:
            row = custody.row
            if row is None:
                return _missing_memory_result(
                    tool_name="get_memory",
                    memory_id=memory_id,
                    mutation="memory_read",
                    requires_approval=False,
                    retry_command="show memory <correct memory id>",
                )
            custody_state = "not_required"
            if custody.profile_owned:
                note = custody.profile_note
                if note is None:
                    return _memory_failure_result(
                        "get_memory",
                        f"Memory #{memory_id} is profile-owned, but its current projection "
                        "custody could not be verified. No memory content was shown.",
                        _safe_metadata(
                            memory_id=memory_id,
                            profile_owned=True,
                            profile_custody_state="unavailable",
                        ),
                        action=MEMORY_READ_RECOVERY_ACTION,
                    )
                try:
                    with vault.canonical_memory_projection_evidence_lock(
                        memory_id=memory_id,
                        store_identity=custody.store_identity,
                        expected_relative_path=note.canonical_path_display,
                        expected_content_digest=note.content_digest,
                    ) as evidence_current:
                        if evidence_current is not True:
                            return _memory_failure_result(
                                "get_memory",
                                f"Memory #{memory_id} is profile-owned, but its current "
                                "projection custody could not be verified. No memory content was shown.",
                                _safe_metadata(
                                    memory_id=memory_id,
                                    profile_owned=True,
                                    profile_custody_state="unavailable",
                                ),
                                action=MEMORY_READ_RECOVERY_ACTION,
                            )
                        output = (
                            f"#{row['id']} [{row['category']}] {row['title']}\n"
                            f"Source: {row['source']} | Confidence: {row['confidence']} | "
                            f"Revision: {row['revision']}\n"
                            f"Created: {row['created_at']} | Updated: {row['updated_at']}\n\n"
                            f"{row['body']}"
                        )
                        custody_state = "verified"
                except Exception:
                    return _memory_failure_result(
                        "get_memory",
                        f"Memory #{memory_id} is profile-owned, but its current projection "
                        "custody could not be verified. No memory content was shown.",
                        _safe_metadata(
                            memory_id=memory_id,
                            profile_owned=True,
                            profile_custody_state="unavailable",
                        ),
                        action=MEMORY_READ_RECOVERY_ACTION,
                    )
            else:
                output = (
                    f"#{row['id']} [{row['category']}] {row['title']}\n"
                    f"Source: {row['source']} | Confidence: {row['confidence']} | "
                    f"Revision: {row['revision']}\n"
                    f"Created: {row['created_at']} | Updated: {row['updated_at']}\n\n"
                    f"{row['body']}"
                )
            return ToolResult(
                "get_memory",
                True,
                _with_memory_boundary(output),
                _safe_metadata(
                    memory_id=memory_id,
                    profile_owned=custody.profile_owned,
                    profile_custody_state=custody_state,
                ),
            )

    def edit_memory(args: dict[str, Any]) -> ToolResult:
        memory_id, error = _positive_id(args.get("memory_id"), "memory_id")
        if memory_id is None:
            return _memory_failure_result(
                "edit_memory",
                error or "memory_id is required.",
                _memory_refusal_metadata(
                    source="edit_memory",
                    mutation="memory_edit",
                    reason=error or "memory_id is required.",
                    raw_memory_id=args.get("memory_id"),
                ),
            )
        raw_body = args.get("body")
        if isinstance(raw_body, str) and len(raw_body) > MAX_MEMORY_BODY_CHARS:
            return _memory_failure_result(
                "edit_memory",
                f"Memory body is too large ({len(raw_body)} chars). Limit is {MAX_MEMORY_BODY_CHARS}.",
                _memory_refusal_metadata(
                    source="edit_memory",
                    mutation="memory_edit",
                    reason="memory body is too large",
                    memory_id=memory_id,
                    limit=MAX_MEMORY_BODY_CHARS,
                ),
            )
        confidence: float | None = None
        if "confidence" in args:
            try:
                confidence = float(args["confidence"])
            except (TypeError, ValueError):
                return _memory_failure_result(
                    "edit_memory",
                    "confidence must be a number.",
                    _memory_refusal_metadata(
                        source="edit_memory",
                        mutation="memory_edit",
                        reason="confidence must be a number.",
                        memory_id=memory_id,
                        raw_confidence=args.get("confidence"),
                    ),
                )
        if confidence is not None and not math.isfinite(confidence):
            return _memory_failure_result(
                "edit_memory",
                "confidence must be finite.",
                _memory_refusal_metadata(
                    source="edit_memory",
                    mutation="memory_edit",
                    reason="confidence must be finite.",
                    memory_id=memory_id,
                    raw_confidence=args.get("confidence"),
                ),
            )
        if confidence is not None:
            confidence = max(0.0, min(1.0, confidence))
        target_revision = args.get("target_revision")
        target_binding = args.get("target_binding")
        if type(target_revision) is not int or target_revision < 1 or type(target_binding) is not str:
            if store.get_memory(memory_id) is None:
                return _missing_memory_result(
                    tool_name="edit_memory",
                    memory_id=memory_id,
                    mutation="memory_edit",
                    requires_approval=True,
                    retry_action="retry the memory edit with the corrected ID",
                )
            return _stale_memory_binding_result(
                tool_name="edit_memory",
                mutation="memory_edit",
                memory_id=memory_id,
                target_status="unbound",
            )
        category = args.get("category") if isinstance(args.get("category"), str) and args.get("category") else None
        title = args.get("title") if isinstance(args.get("title"), str) and args.get("title") else None
        body = raw_body if isinstance(raw_body, str) and raw_body else None
        if category is None and title is None and body is None and confidence is None:
            return _memory_failure_result(
                "edit_memory",
                "No memory fields were supplied to edit. Nothing was changed.",
                _memory_refusal_metadata(
                    source="edit_memory",
                    mutation="memory_edit",
                    reason="no_changes",
                    memory_id=memory_id,
                    requires_approval=True,
                    approval_rerun_blocked=True,
                    approval_rerun_block_reason="memory_edit_has_no_changes",
                ),
            )
        updated = store.update_memory_exact_with_projection(
            memory_id,
            target_revision,
            target_binding,
            category=category,
            title=title,
            body=body,
            confidence=confidence,
        )
        if updated.status == "protected":
            return _memory_failure_result(
                "edit_memory",
                f"Memory #{memory_id} is owned by a curated profile note and was not edited. "
                "Profile notes need their own update workflow so both copies stay in custody.",
                _safe_metadata(
                    memory_id=memory_id,
                    target_status="protected",
                    writes_database=False,
                    writes_memory=False,
                    requires_approval=True,
                ),
            )
        if updated.status != "updated":
            return _stale_memory_binding_result(
                tool_name="edit_memory",
                mutation="memory_edit",
                memory_id=memory_id,
                target_status=updated.status,
            )
        projections = [reconcile_projection_target(target) for target in updated.projection_targets]
        projection_statuses = [projection.status for projection in projections]
        if not projections or any(status != "completed" for status in projection_statuses):
            return _memory_failure_result(
                "edit_memory",
                f"Updated memory #{memory_id}, but its note projection remains pending. "
                "Run `repair memory projections`; do not repeat the edit.",
                _safe_metadata(
                    memory_id=memory_id,
                    projection_statuses=projection_statuses or ["missing"],
                    projection_pending=True,
                    writes_database=True,
                    writes_memory=True,
                    requires_approval=True,
                ),
                action=MEMORY_PROJECTION_RECOVERY_ACTION,
                commands=("repair memory projections",),
            )
        return ToolResult(
            "edit_memory",
            True,
            _with_memory_boundary(f"Updated memory #{memory_id}."),
            _safe_metadata(memory_id=memory_id, writes_database=True, writes_memory=True, requires_approval=True),
        )

    def list_duplicate_memories(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 200)
        duplicates = _duplicate_groups(limit)
        if not duplicates:
            return ToolResult(
                "list_duplicate_memories",
                True,
                _with_memory_boundary("No duplicate-looking memories found."),
                _safe_metadata(groups=0, limit=limit),
            )
        lines = []
        for items in duplicates[:20]:
            lines.append("Duplicate group:")
            for row in items:
                lines.append(f"  - #{row['id']} [{row['category']}] {row['title']}: {row['body'][:120]}")
        return ToolResult(
            "list_duplicate_memories",
            True,
            _with_memory_boundary("\n".join(lines)),
            _safe_metadata(groups=len(duplicates), limit=limit),
        )

    def merge_memories(args: dict[str, Any]) -> ToolResult:
        keep_id, keep_error = _positive_id(args.get("keep_id"), "keep_id")
        delete_id, delete_error = _positive_id(args.get("delete_id"), "delete_id")
        if keep_id is None or delete_id is None:
            return _memory_failure_result(
                "merge_memories",
                keep_error or delete_error or "keep_id and delete_id are required.",
                _memory_refusal_metadata(
                    source="merge_memories",
                    mutation="memory_merge",
                    reason=keep_error or delete_error or "keep_id and delete_id are required.",
                    keep_id=keep_id,
                    delete_id=delete_id,
                    raw_keep_id=args.get("keep_id"),
                    raw_delete_id=args.get("delete_id"),
                ),
            )
        if keep_id == delete_id:
            return _memory_failure_result(
                "merge_memories",
                "keep_id and delete_id must be different memories.",
                _memory_refusal_metadata(
                    source="merge_memories",
                    mutation="memory_merge",
                    reason="keep_id and delete_id must be different memories.",
                    keep_id=keep_id,
                    delete_id=delete_id,
                ),
            )
        keep_revision = args.get("keep_revision")
        keep_binding = args.get("keep_binding")
        delete_revision = args.get("delete_revision")
        delete_binding = args.get("delete_binding")
        if (
            type(keep_revision) is not int
            or keep_revision < 1
            or type(keep_binding) is not str
            or type(delete_revision) is not int
            or delete_revision < 1
            or type(delete_binding) is not str
        ):
            if store.get_memory(keep_id) is None or store.get_memory(delete_id) is None:
                return _missing_memory_merge_result(keep_id=keep_id, delete_id=delete_id)
            return _stale_memory_binding_result(
                tool_name="merge_memories",
                mutation="memory_merge",
                keep_id=keep_id,
                delete_id=delete_id,
                target_status="unbound",
            )
        try:
            merged = store.merge_memories_exact_with_projection(
                keep_id,
                keep_revision,
                keep_binding,
                delete_id,
                delete_revision,
                delete_binding,
            )
        except sqlite3.Error:
            return _failed_memory_merge_result(keep_id=keep_id, delete_id=delete_id)
        if merged.status == "protected":
            protected_ids = sorted(store.profile_owned_memory_ids((keep_id, delete_id)))
            return _memory_failure_result(
                "merge_memories",
                "Profile-owned memories were not merged. Profile notes need their own "
                "workflow so the source ledger and both projections stay consistent.",
                _safe_metadata(
                    keep_id=keep_id,
                    delete_id=delete_id,
                    protected_memory_ids=protected_ids,
                    target_status="protected",
                    writes_database=False,
                    writes_memory=False,
                    requires_approval=True,
                ),
            )
        if merged.status != "merged":
            return _stale_memory_binding_result(
                tool_name="merge_memories",
                mutation="memory_merge",
                keep_id=keep_id,
                delete_id=delete_id,
                target_status=merged.status,
            )
        projections = [reconcile_projection_target(target) for target in merged.projection_targets]
        projection_statuses = [projection.status for projection in projections]
        if len(projections) != 2 or any(status != "completed" for status in projection_statuses):
            return _memory_failure_result(
                "merge_memories",
                f"Merged memory #{delete_id} into #{keep_id}, but a note projection remains pending. "
                "Run `repair memory projections`; do not merge them again.",
                _safe_metadata(
                    keep_id=keep_id,
                    delete_id=delete_id,
                    projection_statuses=projection_statuses or ["missing"],
                    projection_pending=True,
                    writes_database=True,
                    writes_memory=True,
                    requires_approval=True,
                ),
                action=MEMORY_PROJECTION_RECOVERY_ACTION,
                commands=("repair memory projections",),
            )
        return ToolResult(
            "merge_memories",
            True,
            _with_memory_boundary(f"Merged memory #{delete_id} into #{keep_id}."),
            _safe_metadata(keep_id=keep_id, delete_id=delete_id, writes_database=True, writes_memory=True, requires_approval=True),
        )

    def memory_tree_summary(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 200)
        rows = store.list_memories(limit=limit)
        if not rows:
            handoff = _memory_tree_summary_handoff(
                grouped={},
                limit=limit,
                path_display="",
                draft_skill_count=0,
                writes=False,
            )
            return ToolResult(
                "memory_tree_summary",
                True,
                _with_memory_boundary("No memories to summarize."),
                _memory_handoff_metadata(
                    "memory_tree_summary_handoff",
                    handoff,
                    count=0,
                    limit=limit,
                    draft_skill_count=0,
                    human_review_required_for_skill_promotion=True,
                    self_modifying_code=False,
                ),
            )
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["category"]].append(row)
        skills = store.list_skills(limit=20)
        draft_skills = [row for row in skills if str(row["review_status"] or "") == "draft"]

        parts = ["# Memory Tree Snapshot\n"]
        for category in sorted(grouped):
            parts.append(f"## {category.title()}")
            for row in grouped[category][:10]:
                parts.append(f"- {row['title']}: {row['body'][:160]}")
            parts.append("")
        parts.extend(
            [
                "## Reviewable Learning Surfaces",
                "- Skill drafts are review artifacts, not autonomous self-modifying code.",
                "- Promote a skill only after the operator reviews the trigger, scope, verification steps, and stop conditions.",
                "- Keep shell/code, personal data, external side effects, destructive actions, and computer control approval-gated.",
                "",
                "### Draft Skills",
            ]
        )
        if draft_skills:
            for row in draft_skills[:10]:
                parts.append(f"- {row['name']} | trigger: {row['trigger']} | tags: {row['tags']}")
        else:
            parts.append("- No draft skills found.")
        parts.append("")
        snapshot_body = "\n".join(parts)
        path, content_sha256, source_revision = vault.write_memory_tree_snapshot_with_evidence(
            snapshot_body,
            store_identity=store.get_store_identity(),
            source_payload={
                "memories": [{key: row[key] for key in row.keys()} for row in rows],
                "skills": [{key: row[key] for key in row.keys()} for row in skills],
            },
        )
        path_display = _safe_vault_path_display(path, vault)
        handoff = _memory_tree_summary_handoff(
            grouped=grouped,
            limit=limit,
            path_display=path_display,
            draft_skill_count=len(draft_skills),
            writes=True,
        )
        return ToolResult(
            "memory_tree_summary",
            True,
            _with_memory_boundary(f"Memory tree snapshot written: {path_display}"),
            _memory_handoff_metadata(
                "memory_tree_summary_handoff",
                handoff,
                writes=True,
                path=str(path),
                path_display=path_display,
                count=len(rows),
                limit=limit,
                draft_skill_count=len(draft_skills),
                content_sha256=content_sha256,
                source_revision=source_revision,
                write_atomic=True,
                human_review_required_for_skill_promotion=True,
                self_modifying_code=False,
            ),
        )

    def memory_stats(_: dict[str, Any]) -> ToolResult:
        rows = store.memory_stats()
        legacy_review = _conversation_compaction_legacy_review_unavailable()
        try:
            review_payload = store.conversation_compaction_legacy_review(
                limit=CONVERSATION_COMPACTION_LEGACY_REVIEW_LIMIT
            )
            safe_review = _safe_conversation_compaction_legacy_review(review_payload)
            if safe_review is not None:
                legacy_review = safe_review
        except Exception:
            pass
        stat_rows: list[dict[str, Any]] = []
        unreadable_rows = 0
        for row in rows:
            payload = _safe_memory_stat_row(row)
            if payload is None:
                unreadable_rows += 1
            else:
                stat_rows.append(payload)
        if not rows:
            handoff = _memory_stats_handoff(
                rows=[],
                conversation_compaction_legacy_review=legacy_review,
            )
            lines = ["No memories yet. Count: 0.", *_conversation_compaction_legacy_review_lines(legacy_review)]
            return ToolResult(
                "memory_stats",
                True,
                _with_memory_boundary("\n".join(lines)),
                _memory_handoff_metadata(
                    "memory_stats_handoff",
                    handoff,
                    read_only=True,
                    count=0,
                    total_memories=0,
                    readable_memory_stat_rows=0,
                    unreadable_memory_stat_rows=0,
                    conversation_compaction_legacy_review=legacy_review,
                ),
            )
        lines = ["Memory stats:"]
        if unreadable_rows:
            lines.append(f"- hidden malformed memory stats rows: {unreadable_rows}")
        if not stat_rows:
            lines.append("- No readable memory stats rows.")
        for row in stat_rows:
            lines.append(f"- {row['category']} / {row['source']}: {row['count']} latest {row['latest']}")
        lines.extend(_conversation_compaction_legacy_review_lines(legacy_review))
        handoff = _memory_stats_handoff(
            rows=stat_rows,
            conversation_compaction_legacy_review=legacy_review,
        )
        total_memories = handoff.get("total_memories", 0)
        lines.insert(1, f"Total memories: {total_memories}")
        return ToolResult(
            "memory_stats",
            True,
            _with_memory_boundary("\n".join(lines)),
            _memory_handoff_metadata(
                "memory_stats_handoff",
                handoff,
                read_only=True,
                count=len(stat_rows),
                total_memories=total_memories,
                readable_memory_stat_rows=len(stat_rows),
                unreadable_memory_stat_rows=unreadable_rows,
                conversation_compaction_legacy_review=legacy_review,
            ),
        )

    def personal_context_status(_: dict[str, Any]) -> ToolResult:
        """Summarize the durable personal-context surface without rendering content."""
        component_readers = {
            "active_preferences": lambda: store.list_preferences(
                status="active",
                limit=PERSONAL_CONTEXT_STATUS_LIMIT,
            ),
            "active_decisions": lambda: store.list_decisions(status="active", limit=PERSONAL_CONTEXT_STATUS_LIMIT),
            "people": lambda: store.list_people(limit=PERSONAL_CONTEXT_STATUS_LIMIT),
            "active_goals": lambda: store.list_goals(status="active", limit=PERSONAL_CONTEXT_STATUS_LIMIT),
            "open_tasks": lambda: store.list_tasks(status="open", limit=PERSONAL_CONTEXT_STATUS_LIMIT),
            "active_skills": lambda: store.list_active_skills(limit=PERSONAL_CONTEXT_STATUS_LIMIT),
        }
        counts: dict[str, int] = {}
        truncated_components: list[str] = []
        unavailable: list[str] = []
        counts["indexed_memories"] = 0
        try:
            for row in store.memory_stats():
                value = row["count"]
                if type(value) is not int or value < 0:
                    raise ValueError("invalid memory count")
                counts["indexed_memories"] = counts.get("indexed_memories", 0) + value
        except Exception:
            counts.pop("indexed_memories", None)
            unavailable.append("indexed memories")

        for key, reader in component_readers.items():
            try:
                rows = reader()
                if not isinstance(rows, list):
                    raise ValueError("invalid local context rows")
                counts[key] = len(rows)
                if len(rows) >= PERSONAL_CONTEXT_STATUS_LIMIT:
                    truncated_components.append(key)
            except Exception:
                unavailable.append(key.replace("_", " "))

        labels = (
            ("indexed_memories", "indexed memories"),
            ("active_preferences", "active preferences"),
            ("active_decisions", "active decisions"),
            ("people", "people records"),
            ("active_goals", "active goals"),
            ("open_tasks", "open tasks"),
            ("active_skills", "active skills"),
        )
        lines = [
            "Jarvis personal context status:",
            "This is a local, count-only coverage view. It does not reveal memory text, names, notes, preferences, messages, calendar, email, or connector content.",
            "",
            "Durable context coverage:",
        ]
        for key, label in labels:
            if key in counts:
                suffix = "+" if key in truncated_components else ""
                lines.append(f"- {label}: {counts[key]}{suffix}")
            else:
                lines.append(f"- {label}: unavailable")

        represented = [label for key, label in labels if counts.get(key, 0) > 0]
        if represented:
            lines.append("- represented dimensions: " + ", ".join(represented))
        else:
            lines.append("- represented dimensions: none yet")
        if unavailable:
            lines.append("- unavailable count sources: " + ", ".join(sorted(unavailable)))
        if truncated_components:
            lines.append(
                f"- `+` means at least {PERSONAL_CONTEXT_STATUS_LIMIT}; the local count display is intentionally bounded"
            )
        lines.extend(
            [
                "",
                "Boundary:",
                "- Calendar, email, messages, files, research, watchlists, and external accounts are not read by this status view.",
                "- For bounded content you explicitly want to review, ask `what do you know about me`; for memory quality, use `learning review`.",
            ]
        )
        metadata = _safe_metadata(
            **counts,
            unavailable_components=sorted(unavailable),
            truncated_components=sorted(truncated_components),
            represented_dimensions=[label for key, label in labels if counts.get(key, 0) > 0],
            content_free=True,
        )
        return ToolResult(
            "personal_context_status",
            True,
            _with_memory_boundary("\n".join(lines)),
            metadata,
        )

    def build_learning_review(limit: int) -> tuple[str, dict[str, Any]]:
        limit = _bounded_int(limit, 12)
        stats = store.memory_stats()
        feedback_rows = [row for row in store.recent_memories(limit=100) if row["category"] == "feedback"][:limit]
        weak = _weak_rows(8)
        duplicates = _duplicate_groups(300)
        preferences = store.list_preferences(status="active", limit=12)
        skills = store.list_active_skills(limit=12)
        promotion_state = "ready"
        promotion_exception_type = ""
        promotion_rotation_seed = datetime.now(timezone.utc).date().toordinal()
        promotion_within_target_rotation_seed = (
            promotion_rotation_seed // len(KNOWLEDGE_PROMOTION_TARGET_KINDS)
        )
        try:
            promotion_scan_rows = store.list_knowledge_promotion_candidates(
                MAX_MEMORY_CURATOR_LIMIT,
                rotation_seed=promotion_within_target_rotation_seed,
            )
            promotion_rows = _select_rotated_knowledge_promotion_rows(
                promotion_scan_rows,
                limit=limit,
                target_rotation_seed=promotion_rotation_seed,
            )
        except (sqlite3.Error, RuntimeError, ValueError) as exc:
            promotion_rows = []
            promotion_state = "unavailable"
            promotion_exception_type = type(exc).__name__[:80]
        recent_runs = store.recent_tool_runs(limit=80)
        action_runs = [row for row in recent_runs if str(row["tool_name"]) not in EXECUTION_LEARNING_META_TOOLS]
        approval_held_action_runs = [row for row in action_runs if _is_approval_held_tool_run(row)]
        approval_held_action_run_ids = {
            row_id for row_id in (_row_int(row, "id") for row in approval_held_action_runs) if row_id is not None
        }
        failed_or_blocked_runs = [
            row
            for row in action_runs
            if not bool(row["ok"]) and _row_int(row, "id") not in approval_held_action_run_ids
        ]
        non_held_action_runs = [
            row
            for row in action_runs
            if _row_int(row, "id") not in approval_held_action_run_ids
        ]
        after_action_runs = [row for row in recent_runs if str(row["tool_name"]) == "after_action_learning_packet"]
        recovery_runs = [row for row in recent_runs if str(row["tool_name"]) == "execution_recovery_packet"]
        verification_runs = [row for row in recent_runs if str(row["tool_name"]) in {"verification_receipt", "runtime_trace_receipt", "execution_audit_gate"}]
        newest_learning_target = failed_or_blocked_runs[0] if failed_or_blocked_runs else (non_held_action_runs[0] if non_held_action_runs else None)
        learning_target_run_id = int(newest_learning_target["id"]) if newest_learning_target is not None else None
        learning_target_tool_raw = str(newest_learning_target["tool_name"]) if newest_learning_target is not None else ""
        learning_target_tool = _learning_review_display(learning_target_tool_raw, limit=120)
        learning_target_tool_display = learning_target_tool
        target_after_action_runs = [
            row
            for row in after_action_runs
            if _row_int(row, "ok") == 1
            and (_row_int(row, "id") or 0) > (learning_target_run_id or 0)
            and _row_metadata_positive_int(row, "run_id") == learning_target_run_id
        ]
        execution_learning_missing: list[str] = []
        execution_learning_commands: list[str] = []
        if learning_target_run_id is not None:
            if not target_after_action_runs:
                execution_learning_missing.append("target_after_action_learning_packet")
                execution_learning_commands.append(f"after-action learning packet {learning_target_run_id}")
            if failed_or_blocked_runs:
                execution_learning_missing.append("failure_review")
                execution_learning_commands.append(f"execution recovery packet {learning_target_run_id}")
                failure_summary = _short(
                    f"run #{learning_target_run_id} {learning_target_tool} failed or blocked",
                    limit=120,
                )
                execution_learning_commands.append(f"failure to test preview: {failure_summary}")
        approval_review_commands = _approval_review_commands_for_run(approval_held_action_runs[0]) if approval_held_action_runs else []
        if approval_review_commands and not execution_learning_commands:
            execution_learning_commands.extend(approval_review_commands)
        if failed_or_blocked_runs and not after_action_runs:
            execution_learning_state = "LEARNING_DEBT_AFTER_FAILURE"
        elif execution_learning_missing:
            execution_learning_state = "LEARNING_REVIEW_REQUIRED"
        elif approval_held_action_runs:
            execution_learning_state = "APPROVAL_REVIEW_REQUIRED"
        elif action_runs:
            execution_learning_state = "LEARNING_LOOP_HAS_RECENT_ACTION_CONTEXT"
        else:
            execution_learning_state = "NO_RECENT_ACTION_RUNS"
        execution_learning_blocks_completion_claim = execution_learning_state in {
            "LEARNING_DEBT_AFTER_FAILURE",
            "LEARNING_REVIEW_REQUIRED",
        }

        feedback_themes: dict[str, int] = defaultdict(int)
        for row in feedback_rows:
            blob = f"{row['title']} {row['body']}".lower()
            if any(word in blob for word in ("unsafe", "harm", "risk", "approval", "permission", "danger")):
                feedback_themes["safety"] += 1
            elif any(word in blob for word in ("voice", "tone", "talk", "chat", "conversation", "shorter", "concise")):
                feedback_themes["conversation"] += 1
            elif any(word in blob for word in ("memory", "remember", "obsidian", "forget")):
                feedback_themes["memory"] += 1
            elif any(word in blob for word in ("tool", "command", "function", "workflow", "automation")):
                feedback_themes["tools"] += 1
            else:
                feedback_themes["general"] += 1

        lines = [
            "Jarvis learning review:",
            "This is a review-only loop. Jarvis can suggest memory, preference, skill, and test improvements, but it does not auto-edit behavior, permissions, code, or risky tools from this report.",
            "",
            "Feedback signals:",
        ]
        if feedback_rows:
            lines.append(f"- recent feedback captured: {len(feedback_rows)}")
            lines.append("- themes: " + ", ".join(f"{theme} x{count}" for theme, count in sorted(feedback_themes.items())))
            for row in feedback_rows[:4]:
                title = _learning_review_display(row["title"], limit=160)
                body = _learning_review_display(row["body"], limit=140)
                lines.append(f"- #{row['id']} {title}: {body}")
        else:
            lines.append("- no feedback captured yet; use `feedback: ...` after Jarvis does something worth reviewing")

        lines.extend(["", "Memory health:"])
        if stats:
            lines.append(
                "- memory buckets: "
                + ", ".join(
                    f"{_learning_review_display(row['category'], limit=80)}={row['count']}"
                    for row in stats[:8]
                )
            )
        else:
            lines.append("- no memories indexed yet")
        lines.append(f"- weak-looking memories: {len(weak)}")
        for row in weak[:3]:
            category = _learning_review_display(row["category"], limit=80)
            title = _learning_review_display(row["title"], limit=160)
            lines.append(f"  - review memory #{row['id']} [{category}] {title}")
        lines.append(f"- duplicate-looking memory groups: {len(duplicates)}")
        for group in duplicates[:3]:
            ids = ", ".join(f"#{row['id']}" for row in group)
            lines.append(f"  - inspect duplicate group: {ids}")

        lines.extend(["", "Personalization base:"])
        lines.append(f"- active preferences: {len(preferences)}")
        for row in preferences[:4]:
            category = _learning_review_display(row["category"], limit=80)
            key = _learning_review_display(row["key"], limit=120)
            value = _learning_review_display(row["value"], limit=240)
            lines.append(f"  - [{category}] {key}: {value}")
        lines.append(f"- saved skills: {len(skills)}")
        for row in skills[:4]:
            summary = row["summary"] if "summary" in row.keys() else row["body"]
            name = _learning_review_display(row["name"], limit=120)
            summary_display = _learning_review_display(summary, limit=120)
            lines.append(f"  - {name}: {summary_display}")

        promotion_target_counts: dict[str, int] = {}
        promotion_candidate_ids: list[int] = []
        promotion_candidate_bindings: list[dict[str, Any]] = []
        promotion_review_commands: list[str] = []
        promotion_render_blocks: list[tuple[str, str]] = []
        promotion_unreadable_rows = 0
        promotion_candidate_total = 0
        promotion_totals_conflict = False
        lines.extend(
            [
                "",
                "Structured knowledge classification review:",
                "- Advisory only: category labels can suggest where a generic memory might belong, but they are not semantic proof.",
                "- Decision, new-preference, and profile candidates have exact, approval-gated ownership-transfer paths that preserve the source memory ID; person and goal transfers remain review-only.",
                "- This classification step does not promote records, call a model, write the database, or change behavior.",
                "- `learning review` does not write files; `save learning review` publishes this report to Obsidian and maintains a local lock file that protects that publication.",
                f"- candidate scan state: {promotion_state}",
            ]
        )
        if promotion_state == "unavailable":
            lines.append("  - structured candidate custody could not be inspected; existing memories remain unchanged")
        elif promotion_rows:
            for row in promotion_rows:
                try:
                    memory_id_raw = row["id"]
                    revision_raw = row["revision"]
                    target_kind_raw = row["target_kind"]
                    target_total_raw = row["target_total"]
                    candidate_total_raw = row["candidate_total"]
                    all_target_totals_raw = {
                        kind: row[f"{kind}_total"]
                        for kind in ("decision", "goal", "person", "preference", "profile")
                    }
                    if (
                        type(memory_id_raw) is not int
                        or type(revision_raw) is not int
                        or type(target_kind_raw) is not str
                        or type(target_total_raw) is not int
                        or type(candidate_total_raw) is not int
                        or any(type(total) is not int for total in all_target_totals_raw.values())
                        or any(total < 0 for total in all_target_totals_raw.values())
                    ):
                        raise ValueError("invalid promotion candidate types")
                    memory_id = memory_id_raw
                    revision = revision_raw
                    target_kind = target_kind_raw
                    target_total = target_total_raw
                    candidate_total = candidate_total_raw
                    all_target_totals = {
                        kind: total
                        for kind, total in all_target_totals_raw.items()
                        if total > 0
                    }
                    category_raw = row["category"]
                    title_raw = row["title"]
                    if memory_id < 1 or revision < 1 or target_kind not in {
                        "profile", "preference", "decision", "person", "goal"
                    } or target_total < 1 or candidate_total < target_total \
                        or sum(all_target_totals_raw.values()) != candidate_total \
                        or all_target_totals.get(target_kind) != target_total \
                        or type(category_raw) is not str \
                        or type(title_raw) is not str:
                        raise ValueError("invalid promotion candidate")
                except (KeyError, IndexError, TypeError, ValueError):
                    promotion_unreadable_rows += 1
                    continue
                if promotion_candidate_total and promotion_candidate_total != candidate_total:
                    promotion_totals_conflict = True
                    promotion_unreadable_rows += 1
                    continue
                if promotion_target_counts and promotion_target_counts != all_target_totals:
                    promotion_totals_conflict = True
                    promotion_unreadable_rows += 1
                    continue
                promotion_candidate_total = candidate_total
                promotion_target_counts = all_target_totals
                promotion_candidate_ids.append(memory_id)
                promotion_candidate_bindings.append(
                    {
                        "memory_id": memory_id,
                        "memory_revision": revision,
                        "target_kind": target_kind,
                        "classification_basis": "category_label_only",
                        "semantic_proof": False,
                        "authorizes_promotion": False,
                    }
                )
                promotion_review_commands.append(f"knowledge promotion packet {memory_id}")
                category = _learning_review_display(category_raw, limit=80)
                title = _learning_review_display(title_raw, limit=160)
                candidate_line = (
                    f"  - memory #{memory_id} revision {revision} [{category}] -> possible {target_kind}: {title}"
                )
                review_line = (
                    f"    Review first: `knowledge promotion packet {memory_id}`. Decision, new-preference, and profile transfer are available only after explicit fields and approval; person and goal remain review-only."
                )
                lines.extend((candidate_line, review_line))
                promotion_render_blocks.append((candidate_line, review_line))
            if promotion_unreadable_rows:
                lines.append(
                    f"  - unreadable candidate rows hidden: {promotion_unreadable_rows}"
                )
        else:
            lines.append(
                "  - no unowned memories matched the supported category-label aliases; other memories may still contain structured personal knowledge"
            )
        promotion_selected_rows = len(promotion_rows) if promotion_state == "ready" else 0
        promotion_totals_reliable = bool(
            promotion_state == "ready"
            and not promotion_totals_conflict
            and promotion_selected_rows <= promotion_candidate_total
        )
        if not promotion_totals_reliable:
            promotion_candidate_total = promotion_selected_rows
            promotion_target_counts = {}
        promotion_hidden_count = max(
            0, promotion_candidate_total - promotion_selected_rows
        )
        if promotion_state == "ready":
            if promotion_totals_reliable:
                lines.append(
                    f"- candidate rows scanned for the review packet: {promotion_selected_rows} of {promotion_candidate_total}; not selected by the display cap: {promotion_hidden_count}"
                )
            else:
                lines.append(
                    f"- candidate rows scanned for the review packet: {promotion_selected_rows}; global totals unavailable because selected rows were malformed"
                )
            lines.append(
                f"- valid candidate records in the packet: {len(promotion_candidate_ids)}; unreadable selected rows: {promotion_unreadable_rows}"
            )
            if promotion_target_counts:
                lines.append(
                    "- totals by possible target: "
                    + ", ".join(
                        f"{kind}={count}"
                        for kind, count in sorted(promotion_target_counts.items())
                    )
                )

        lines.extend(
            [
                "",
                "Execution learning debt:",
                f"- recent tool runs inspected: {len(recent_runs)}",
                f"- recent action runs: {len(action_runs)}",
                f"- failed or blocked action runs: {len(failed_or_blocked_runs)}",
                f"- approval-held action runs: {len(approval_held_action_runs)}",
                f"- recent verification/audit packets: {len(verification_runs)}",
                f"- recent recovery packets: {len(recovery_runs)}",
                f"- recent after-action learning packets: {len(after_action_runs)}",
                f"- learning state: {execution_learning_state}",
                f"- blocks completion claim: {'yes' if execution_learning_blocks_completion_claim else 'no'}",
                f"- target run: #{learning_target_run_id} `{learning_target_tool_display}`" if learning_target_run_id is not None else "- target run: none",
                f"- target after-action learning packets: {len(target_after_action_runs)}",
                f"- missing learning proof: {', '.join(execution_learning_missing) if execution_learning_missing else 'none'}",
            ]
        )
        if execution_learning_commands:
            lines.append("- next learning required commands:")
            lines.extend(f"  - `{_learning_review_display(command, limit=200)}`" for command in execution_learning_commands)
        else:
            lines.append("- next learning required commands: none")
        if approval_review_commands:
            lines.append("- approval review commands:")
            lines.extend(f"  - `{_learning_review_display(command, limit=160)}`" for command in approval_review_commands)

        lines.extend(
            [
                "",
                "Safe learning actions:",
                "- Run `save feedback report` and `save feedback actions` after meaningful feedback accumulates.",
                "- Run `execution health report` and the next after-action learning packet when recent failed/blocked runs appear.",
                "- Review approval-held runs through `approval readiness`, `approval packet`, and `approval chain proof` before treating them as execution evidence.",
                "- Run `weak memories` before deleting or editing uncertain memories.",
                "- Run `duplicate memories` before merging repeated memories.",
                "- Convert stable feedback into explicit `set preference ...` commands.",
                "- Inspect category-labeled structuring candidates with `knowledge promotion packet <id>`; decision, new-preference, and profile transfer require explicit fields and approval, while person and goal transfer remain review-only.",
                "- Draft skills from repeated successful workflows, then review them before relying on them.",
                "",
                "Still manual or approval-gated:",
                "- Memory deletion, memory merging, code changes, shell, personal data, external actions, and computer control.",
                "- the operator's explicit stop times, work windows, pause commands, and newer instructions override this learning loop.",
            ]
        )
        metadata = _safe_metadata(
            feedback=len(feedback_rows),
            weak_memories=len(weak),
            duplicate_groups=len(duplicates),
            preferences=len(preferences),
            skills=len(skills),
            knowledge_promotion_candidates=len(promotion_candidate_ids),
            knowledge_promotion_selected_rows=promotion_selected_rows,
            knowledge_promotion_rendered_candidates=len(promotion_candidate_ids),
            knowledge_promotion_candidate_total=promotion_candidate_total,
            knowledge_promotion_totals_reliable=promotion_totals_reliable,
            knowledge_promotion_totals_conflict=promotion_totals_conflict,
            knowledge_promotion_hidden_count=promotion_hidden_count,
            knowledge_promotion_output_hidden_count=promotion_hidden_count,
            knowledge_promotion_state=promotion_state,
            knowledge_promotion_unreadable_rows=promotion_unreadable_rows,
            knowledge_promotion_exception_type=promotion_exception_type,
            knowledge_promotion_classification_basis="category_label_only",
            knowledge_promotion_semantic_proof=False,
            knowledge_promotion_authorizes_promotion=False,
            knowledge_promotion_rotation_seed=promotion_rotation_seed,
            knowledge_promotion_candidate_ids=promotion_candidate_ids,
            knowledge_promotion_candidate_bindings=promotion_candidate_bindings,
            knowledge_promotion_target_counts=dict(sorted(promotion_target_counts.items())),
            knowledge_promotion_review_commands=promotion_review_commands,
            reads_personal_data=True,
            reads_private_data=True,
            recent_tool_runs=len(recent_runs),
            recent_action_runs=len(action_runs),
            failed_or_blocked_action_runs=len(failed_or_blocked_runs),
            approval_held_action_runs=len(approval_held_action_runs),
            recent_verification_runs=len(verification_runs),
            recent_recovery_runs=len(recovery_runs),
            recent_after_action_learning_runs=len(after_action_runs),
            execution_learning_state=execution_learning_state,
            execution_learning_blocks_completion_claim=execution_learning_blocks_completion_claim,
            execution_learning_target_run_id=learning_target_run_id,
            execution_learning_target_tool=learning_target_tool,
            execution_learning_target_after_action_packets=len(target_after_action_runs),
            execution_learning_missing=execution_learning_missing,
            execution_learning_missing_count=len(execution_learning_missing),
            execution_learning_required_commands=execution_learning_commands,
            execution_learning_approval_review_commands=approval_review_commands,
            execution_learning_next_required_command=execution_learning_commands[0] if execution_learning_commands else "",
            execution_learning_proof_queue=execution_learning_commands,
            execution_learning_proof_queue_count=len(execution_learning_commands),
            execution_learning_next_proof_command=execution_learning_commands[0] if execution_learning_commands else "",
            execution_learning_next_commands=execution_learning_commands,
            execution_learning_next_command=execution_learning_commands[0] if execution_learning_commands else "",
            execution_learning_next_command_count=len(execution_learning_commands),
            limit=limit,
        )
        handoff = _learning_review_handoff(metadata=metadata, writes=False)
        metadata.update(
            _memory_contract(
                state_changed=_memory_metadata_bool(handoff.get("state_changed")),
                changed=list(handoff.get("changed") or []),
                content_in_handoff=_memory_metadata_bool(handoff.get("content_in_handoff")),
            )
        )
        metadata["learning_review_handoff"] = handoff
        metadata["learning_review_handoff_ready"] = True
        body = "\n".join(lines)
        if len(body) + len(MEMORY_BOUNDARY) > MAX_LEARNING_REVIEW_BODY_CHARS:
            suffix = "\n\n[Learning review content truncated.]"
            body_limit = MAX_LEARNING_REVIEW_BODY_CHARS - len(MEMORY_BOUNDARY) - len(suffix)
            body = body[: max(0, body_limit)].rstrip() + suffix
        rendered_lines = body.splitlines()
        rendered_line_pairs = set(zip(rendered_lines, rendered_lines[1:]))
        rendered_candidate_count = sum(
            render_block in rendered_line_pairs
            for render_block in promotion_render_blocks
        )
        output_hidden_count = max(
            0, promotion_candidate_total - rendered_candidate_count
        )
        metadata["knowledge_promotion_rendered_candidates"] = rendered_candidate_count
        metadata["knowledge_promotion_output_hidden_count"] = output_hidden_count
        metadata["learning_review_handoff"][
            "knowledge_promotion_rendered_candidates"
        ] = rendered_candidate_count
        metadata["learning_review_handoff"][
            "knowledge_promotion_output_hidden_count"
        ] = output_hidden_count
        return _with_memory_boundary(body), metadata

    def learning_review(args: dict[str, Any]) -> ToolResult:
        body, metadata = build_learning_review(_bounded_int(args.get("limit"), 12))
        return ToolResult("learning_review", True, body, metadata)

    def save_learning_review(args: dict[str, Any]) -> ToolResult:
        with store.generated_report_publication_fence():
            body, metadata = build_learning_review(_bounded_int(args.get("limit"), 12))
            path, _content_sha256, _source_revision = vault.write_learning_review_with_evidence(
                f"# Learning Review\n\n{body}",
                store_identity=store.get_store_identity(),
                source_payload={"body": body, "metadata": metadata},
            )
        path_display = _safe_vault_path_display(path, vault)
        metadata["path"] = str(path)
        metadata["path_display"] = path_display
        handoff = _learning_review_handoff(
            metadata=metadata,
            writes=True,
            path_display=path_display,
        )
        metadata.update(
            _memory_contract(
                state_changed=_memory_metadata_bool(handoff.get("state_changed")),
                changed=list(handoff.get("changed") or []),
                content_in_handoff=_memory_metadata_bool(handoff.get("content_in_handoff")),
            )
        )
        metadata["writes_files"] = True
        metadata["writes_notes"] = True
        metadata["learning_review_handoff"] = handoff
        metadata["learning_review_handoff_ready"] = True
        return ToolResult("save_learning_review", True, f"Learning review saved: {path_display}\n\n{body}", metadata)

    def queue_learning_tasks(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 12)
        feedback_rows = [row for row in store.recent_memories(limit=100) if row["category"] == "feedback"][:limit]
        weak = _weak_rows(8)
        duplicates = _duplicate_groups(300)
        preferences = store.list_preferences(status="active", limit=12)
        skills = store.list_active_skills(limit=12)
        recent_runs = store.recent_tool_runs(limit=80)
        action_runs = [row for row in recent_runs if str(row["tool_name"]) not in EXECUTION_LEARNING_META_TOOLS]
        approval_held_action_runs = [row for row in action_runs if _is_approval_held_tool_run(row)]
        approval_held_action_run_ids = {
            row_id for row_id in (_row_int(row, "id") for row in approval_held_action_runs) if row_id is not None
        }
        failed_or_blocked_runs = [
            row
            for row in action_runs
            if not bool(row["ok"]) and _row_int(row, "id") not in approval_held_action_run_ids
        ]
        after_action_runs = [row for row in recent_runs if str(row["tool_name"]) == "after_action_learning_packet"]
        candidates: list[tuple[str, str]] = []
        if feedback_rows:
            candidates.append(("Review Jarvis feedback report and decide whether to add a preference, skill, or smoke test.", "normal"))
        if weak:
            candidates.append(("Review weak-looking memories before editing or deleting any uncertain memory.", "normal"))
        if duplicates:
            candidates.append(("Inspect duplicate-looking memories before approving any memory merge.", "normal"))
        if not preferences and feedback_rows:
            candidates.append(("Consider adding one explicit preference from repeated Jarvis feedback.", "low"))
        if not skills and feedback_rows:
            candidates.append(("Consider drafting one reviewed skill from repeated successful Jarvis workflows.", "low"))
        target = failed_or_blocked_runs[0] if failed_or_blocked_runs else None
        target_run_id = _row_int(target, "id") if target is not None else None
        target_after_action_runs = [
            row
            for row in after_action_runs
            if target_run_id is not None
            and _row_int(row, "ok") == 1
            and (_row_int(row, "id") or 0) > target_run_id
            and _row_metadata_positive_int(row, "run_id") == target_run_id
        ]

        if target is not None and not target_after_action_runs:
            tool_name = _learning_review_display(target["tool_name"], limit=120)
            candidates.append(
                (
                    f"Review after-action learning for failed run #{target['id']} {tool_name} before promoting memories, skills, or tests.",
                    "high",
                )
            )
        elif approval_held_action_runs:
            target = approval_held_action_runs[0]
            review_command = _approval_review_commands_for_run(target)[0]
            tool_name = _learning_review_display(target["tool_name"], limit=120)
            candidates.append(
                (
                    f"Review approval-held run #{target['id']} {tool_name} with `{review_command}` before treating it as execution evidence.",
                    "normal",
                )
            )
        if not candidates:
            candidates.append(("Capture one concrete Jarvis feedback item after the next meaningful assistant interaction.", "low"))

        records = [
            TaskRecord(body=body, priority=priority, source="learning-review")
            for body, priority in candidates
        ]
        inserted = store.add_tasks_if_identities_absent(
            records,
            existing_status="open",
        )
        added: list[tuple[int, str]] = []
        skipped: list[str] = []
        for (body, _priority), task_id in zip(candidates, inserted):
            if task_id is None:
                skipped.append(body)
                continue
            added.append((task_id, body))

        path = vault.sync_open_tasks(store)
        path_display = _safe_vault_path_display(path, vault)
        lines = [
            "Queued learning tasks:",
            "These are local task reminders only. Jarvis did not edit memory, merge records, change preferences, write code, or run risky tools.",
        ]
        if added:
            for task_id, body in added:
                lines.append(f"- Added task #{task_id}: {body}")
        else:
            lines.append("- No new tasks added; matching open tasks already exist.")
        if skipped:
            lines.append("")
            lines.append("Already queued:")
            for body in skipped:
                lines.append(f"- {body}")
        lines.extend(
            [
                "",
                "Still approval-gated:",
                "- Memory deletion, memory merging, code changes, shell, personal data, external actions, and computer control.",
                "- the operator's explicit stop times, work windows, pause commands, and newer instructions override queued learning work.",
            ]
        )
        handoff = _queue_learning_tasks_handoff(
            added=added,
            skipped=skipped,
            limit=limit,
            path_display=path_display,
        )
        return ToolResult(
            "queue_learning_tasks",
            True,
            _with_memory_boundary("\n".join(lines)),
            _memory_handoff_metadata(
                "queue_learning_tasks_handoff",
                handoff,
                writes=True,
                added=len(added),
                skipped=len(skipped),
                limit=limit,
                path_display=path_display,
                added_task_ids=[task_id for task_id, _body in added],
                writes_database=bool(added),
                writes_memory=bool(added),
            ),
        )

    return (
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
    )
