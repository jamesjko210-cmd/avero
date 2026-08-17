from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from jarvis_v2.agent.failure_guidance import declare_failure_guidance
from jarvis_v2.agent.types import ToolResult
from jarvis_v2.memory.memory_projection import reconcile_memory_projection
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import MemoryRecord, MemoryStore


MAX_FEEDBACK_LIMIT = 200
MAX_FEEDBACK_BODY_CHARS = 50000
MAX_FEEDBACK_TITLE_CHARS = 160
MAX_FEEDBACK_THEME_CHARS = 80
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
FEEDBACK_INPUT_RECOVERY_ACTION = (
    "Correct the reported feedback field, then submit it again through the normal local-safe policy."
)


def save_feedback_report_auto_mutation_operation_key(
    _args: dict[str, Any],
) -> dict[str, Any]:
    return {"projection": "feedback_report"}


def save_feedback_actions_auto_mutation_operation_key(
    _args: dict[str, Any],
) -> dict[str, Any]:
    return {"projection": "feedback_actions"}


def _bounded_int(value: Any, default: int, low: int = 1, high: int = MAX_FEEDBACK_LIMIT) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _short(value: Any, limit: int) -> str:
    try:
        text = str(value or "").strip()
    except Exception:
        text = ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _short_metadata(value: Any, limit: int = 80) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", _short(value, limit))


def _safe_display(value: Any, limit: int) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", _short(value, limit))


def _safe_vault_path_display(path: Path | str | None, vault: ObsidianVault) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return str(candidate.relative_to(vault.root_path))
    except ValueError:
        return _short_metadata(candidate, 120)


def _text_sha256(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "reads_private_data": False,
        "writes_files": False,
        "writes_memory": False,
        "writes_notes": False,
        "writes_database": False,
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


def _feedback_input_failure(message: str, *, reason: str, **extra: Any) -> ToolResult:
    output = f"{message} {FEEDBACK_INPUT_RECOVERY_ACTION}"
    metadata = _safe_metadata(
        reason=reason,
        memory_id=None,
        outcome_known=True,
        outcome_unknown=False,
        execution_outcome_unknown=False,
        side_effect_possible=False,
        retry_safe=True,
        automatic_retry_allowed=False,
        authorizes_retry=False,
        **extra,
    )
    return ToolResult(
        "record_feedback",
        False,
        output,
        declare_failure_guidance(
            metadata,
            output=output,
            action=FEEDBACK_INPUT_RECOVERY_ACTION,
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


def _row_positive_int(row: Any, key: str = "id") -> int | None:
    value = _row_value(row, key)
    if isinstance(value, bool) or value is _MISSING_ROW_VALUE or value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number > 0 else None


def _safe_feedback_row(row: Any) -> tuple[dict[str, Any] | None, bool]:
    category_raw = _row_value(row, "category")
    if category_raw is _MISSING_ROW_VALUE:
        return None, True
    category = _safe_display(category_raw, MAX_FEEDBACK_THEME_CHARS).lower()
    if category != "feedback":
        return None, False
    memory_id = _row_positive_int(row)
    body_raw = _row_value(row, "body")
    body = _safe_display(body_raw, MAX_FEEDBACK_BODY_CHARS)
    if memory_id is None or not body:
        return None, True
    title = _safe_display(_row_value(row, "title", "Jarvis feedback"), MAX_FEEDBACK_TITLE_CHARS) or "Jarvis feedback"
    return {"id": memory_id, "category": "feedback", "title": title, "body": body}, False


def _safe_feedback_rows(rows: list[Any], *, limit: int) -> tuple[list[dict[str, Any]], int]:
    readable: list[dict[str, Any]] = []
    unreadable = 0
    for row in rows:
        payload, unreadable_row = _safe_feedback_row(row)
        if unreadable_row:
            unreadable += 1
            continue
        if payload is None:
            continue
        if len(readable) < limit:
            readable.append(payload)
    return readable, unreadable


def _feedback_metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _feedback_write_metadata(**extra: Any) -> dict[str, Any]:
    metadata = _safe_metadata()
    metadata.update({"writes_files": True, "writes_memory": True, "writes_notes": True, "writes_database": True})
    metadata.update(extra)
    return metadata


def _feedback_contract(
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


def _feedback_handoff_metadata(
    handoff_key: str,
    handoff: dict[str, Any],
    *,
    writes: bool = False,
    writes_memory: bool | None = None,
    writes_database: bool | None = None,
    **extra: Any,
) -> dict[str, Any]:
    _normalize_feedback_handoff(handoff)
    prefix = handoff_key.removesuffix("_handoff")
    metadata = _feedback_write_metadata(**extra) if writes else _safe_metadata(**extra)
    if writes and writes_memory is False:
        metadata["writes_memory"] = False
    if writes and writes_database is False:
        metadata["writes_database"] = False
    metadata.update(
        _feedback_contract(
            state_changed=_feedback_metadata_bool(handoff.get("state_changed")),
            changed=list(handoff.get("changed") or []),
            content_in_handoff=_feedback_metadata_bool(handoff.get("content_in_handoff")),
        )
    )
    metadata[f"{handoff_key}_ready"] = True
    metadata[f"{prefix}_handoff_ready"] = True
    metadata[f"{prefix}_ready_for_operator"] = True
    metadata[f"{prefix}_state_changed"] = _feedback_metadata_bool(handoff.get("state_changed"))
    metadata[f"{prefix}_changed"] = list(handoff.get("changed") or [])
    metadata[f"{prefix}_content_in_handoff"] = _feedback_metadata_bool(handoff.get("content_in_handoff"))
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


def _normalize_feedback_handoff(handoff: dict[str, Any]) -> dict[str, Any]:
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


def _feedback_capture_handoff(*, memory_id: int, theme: str, body_chars: int, title_chars: int) -> dict[str, Any]:
    return {
        "source": "record_feedback",
        "memory_id": memory_id,
        "theme": _short_metadata(theme, MAX_FEEDBACK_THEME_CHARS),
        "body_chars": body_chars,
        "title_chars": title_chars,
        "next_commands": ["feedback report", "feedback actions", "failure to test: <miss>"],
        "boundaries": {
            "read_only": False,
            "writes_files": True,
            "writes_memory": True,
            "writes_notes": True,
            "writes_database": True,
            "queues_approval": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "requires_approval": False,
            "controls_computer": False,
            "external_side_effect": False,
        },
        **_feedback_contract(state_changed=True, changed=["feedback"], content_in_handoff=False),
    }


def _feedback_report_handoff(
    *,
    source: str,
    count: int,
    themes: Any,
    limit: int,
    saved: bool = False,
    path_display: str = "",
    unreadable_feedback_rows: int = 0,
) -> dict[str, Any]:
    normalized_themes = themes if isinstance(themes, list) else sorted(dict(themes or {}).keys())
    return {
        "source": source,
        "count": count,
        "readable_feedback_rows": count,
        "unreadable_feedback_rows": unreadable_feedback_rows,
        "themes": normalized_themes,
        "limit": limit,
        "saved": saved,
        "path_display": path_display,
        "next_commands": {
            "capture_feedback": "feedback: <what should Jarvis improve?>",
            "review_report": "feedback report",
            "review_actions": "feedback actions",
            "save_report": "save feedback report",
            "save_actions": "save feedback actions",
        },
        "boundaries": {
            "read_only": not saved,
            "writes_files": saved,
            "writes_memory": False,
            "writes_notes": saved,
            "writes_database": False,
            "queues_approval": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "requires_approval": False,
            "controls_computer": False,
            "external_side_effect": False,
        },
        **_feedback_contract(
            state_changed=saved,
            changed=["feedback_actions" if source == "save_feedback_actions" else "feedback_report"] if saved else [],
            content_in_handoff=count > 0,
        ),
    }


def _feedback_theme(text: str) -> str:
    low = text.lower()
    if any(word in low for word in ("unsafe", "harm", "risk", "approval", "permission", "danger")):
        return "safety"
    if any(word in low for word in ("slow", "fast", "speed", "latency", "performance")):
        return "performance"
    if any(word in low for word in ("voice", "tone", "talk", "chat", "conversation")):
        return "conversation"
    if any(word in low for word in ("memory", "remember", "obsidian", "forget")):
        return "memory"
    if any(word in low for word in ("tool", "command", "function", "workflow", "automation")):
        return "tools"
    return "general"


def _failure_surface(theme: str, text: str) -> tuple[str, str, list[str], str]:
    low = text.lower()
    if any(word in low for word in ("overlap", "layout", "ui", "button", "panel", "scroll", "text", "screen", "dashboard", "diagnostic")):
        return (
            "frontend layout and visual regression",
            "smoke_test_status_server_layout",
            [
                "Rendered dashboard keeps diagnostics, conversation log, and composer in separate regions.",
                "No visible status, diagnostic, or composer text overlaps at the target viewport.",
                "System diagnostics remain reachable without hiding the command composer.",
            ],
            "Add a viewport layout assertion before changing the Jarvis HUD again.",
        )
    if theme == "safety" or any(word in low for word in ("approve", "approval", "unsafe", "permission", "risk")):
        return (
            "approval and safety routing",
            "smoke_test_safety_boundaries",
            [
                "Risky requests produce approval readiness, a last-look approval packet, and approval chain proof before action.",
                "Read-only preview commands do not queue approval or mutate state.",
                "The response explains the approval boundary in plain language.",
            ],
            "Capture an approval-boundary preference only after the failing case repeats.",
        )
    if theme == "memory":
        return (
            "memory retrieval and learning review",
            "smoke_test_memory_learning",
            [
                "Relevant memories are cited or summarized when the request needs context.",
                "Weak or duplicate memories remain reviewable before any merge/delete action.",
                "Learning preview stays read-only.",
            ],
            "Draft a memory-use preference if the same memory miss repeats.",
        )
    if theme == "conversation":
        return (
            "conversation response quality",
            "smoke_test_chat_response_health",
            [
                "Jarvis answers the user request directly before background detail.",
                "The response keeps requested summary length and tone.",
                "Safety caveats appear only when they affect the action.",
            ],
            "Draft a response-style preference after one more concrete example.",
        )
    if theme == "performance":
        return (
            "latency and bounded output",
            "smoke_test_performance_bounds",
            [
                "The slow path reports progress or a bounded summary.",
                "Large outputs are clipped with a useful next command.",
                "The command remains local-safe unless explicit approval is needed.",
            ],
            "Add a performance budget note to the relevant smoke test.",
        )
    return (
        "general assistant behavior",
        "smoke_test_failure_regression",
        [
            "The original failing request gets a stable, direct response.",
            "The output includes the expected next safe command.",
            "The command remains review-only unless the user approves mutation.",
        ],
        "Collect one more failure example before turning this into a saved skill.",
    )


def _cluster_key(theme: str, surface: str) -> str:
    return f"{theme} / {surface}"


def _target_test_file(test_name: str) -> str:
    return {
        "smoke_test_status_server_layout": "jarvis_v2/scripts/smoke_test_status_server.py",
        "smoke_test_safety_boundaries": "jarvis_v2/scripts/smoke_test_safety.py",
        "smoke_test_memory_learning": "jarvis_v2/scripts/smoke_test_learning_review.py",
        "smoke_test_chat_response_health": "jarvis_v2/scripts/smoke_test_chat_context.py",
        "smoke_test_performance_bounds": "jarvis_v2/scripts/smoke_test_all.py",
    }.get(test_name, "jarvis_v2/scripts/smoke_test_feedback.py")


def _focused_command_for_test(test_name: str) -> str:
    if test_name == "smoke_test_status_server_layout":
        return "python3 -m jarvis_v2.scripts.smoke_test_status_server"
    if test_name == "smoke_test_all":
        return "python3 -m jarvis_v2.scripts.smoke_test_all"
    return f"python3 -m jarvis_v2.scripts.{test_name}"


def _rollback_note(test_name: str) -> str:
    test_file = _target_test_file(test_name)
    return f"Revert the scoped assertion and any matching implementation edit in `{test_file}` if the examples do not describe one failure mode."


def _receipt_field(summary: str, key: str) -> str:
    pattern = re.compile(
        rf"(?:^|[;\n])\s*{re.escape(key)}\s*:?\s*(?P<value>.*?)(?=(?:[;\n]\s*(?:changed files?|verification|focused verification|compile|rollback|cluster|test first|pre[- ]patch test|failing test|application bridge|patch application|applied patch|apply bridge|apply contract sha256|contract sha256|apply contract|contract|completion audit|audit|evidence ledger|ledger|completion claim gate|claim gate|post claim review|review|after-action learning|after action learning|learning|record target|target|regression link|regression|durability|durable|notes?)\s*:?)|$)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(summary)
    return _short(match.group("value"), 500) if match else ""


def _ordered_commands(commands: list[str]) -> list[str]:
    ordered: list[str] = []
    for command in commands:
        if command and command not in ordered:
            ordered.append(command)
    return ordered


def _looks_like_sha256(value: str) -> bool:
    text = value.strip().lower()
    return bool(re.fullmatch(r"[0-9a-f]{64}", text))


def _contract_review_evidence(summary: str) -> str:
    for key in ("apply contract", "contract"):
        evidence = _receipt_field(summary, key)
        if evidence and not evidence.lower().lstrip().startswith("sha256"):
            return evidence
    match = re.search(
        r"(?:^|[;\n])\s*(?:apply\s+contract|contract)\s+(?!sha256\b)(?P<value>[^;\n]*(?:reviewed|passed|accepted|confirmed|ready)[^;\n]*)",
        summary,
        re.IGNORECASE,
    )
    return _short(match.group("value"), 500) if match else ""


def _apply_contract_sha256(*, surface: str, target_test: str, target_file: str, focused_command: str) -> str:
    return _text_sha256(
        "\n".join(
            [
                "failure_apply_contract_v1",
                surface,
                target_test,
                target_file,
                focused_command,
            ]
        )
    )


def _failure_regression_test_contract_sha256(
    *,
    surface: str,
    target_test: str,
    target_file: str,
    focused_command: str,
    expected_behavior: str,
    observed_failures: list[str],
    rollback_note: str,
    assertions: list[str],
) -> str:
    return _text_sha256(
        "\n".join(
            [
                "failure_regression_test_contract_v1",
                surface,
                target_test,
                target_file,
                focused_command,
                expected_behavior,
                rollback_note,
                repr([str(item).strip() for item in observed_failures if str(item).strip()]),
                repr([str(item).strip() for item in assertions if str(item).strip()]),
                "exact_test_before_patch",
            ]
        )
    )


def _failure_regression_test_contract_metadata(cluster: dict[str, Any], *, focused_command: str) -> dict[str, Any]:
    observed_failures = [str(row["body"] if "body" in row.keys() else "")[:220] for row in cluster.get("rows", [])[:4]]
    token = _failure_regression_test_contract_sha256(
        surface=str(cluster.get("surface") or ""),
        target_test=str(cluster.get("test_name") or ""),
        target_file=str(cluster.get("target_file") or _target_test_file(str(cluster.get("test_name") or ""))),
        focused_command=focused_command,
        expected_behavior=str(cluster.get("expected_behavior") or ""),
        observed_failures=observed_failures,
        rollback_note=str(cluster.get("rollback_note") or ""),
        assertions=[str(item) for item in (cluster.get("assertions") or [])],
    )
    return {
        "failure_regression_test_contract_sha256": token,
        "failure_regression_test_contract_present": _looks_like_sha256(token),
        "failure_regression_test_contract_ready": _looks_like_sha256(token),
        "exact_test_contract_ready": _looks_like_sha256(token),
        "exact_test_contract_requires_pre_patch_failure": True,
        "exact_test_contract_authorizes_file_write": False,
        "exact_test_contract_authorizes_patch_application": False,
        "exact_test_contract_authorizes_tool_execution": False,
        "exact_test_contract_authorizes_completion_claim": False,
        "exact_test_contract_reusable_for_next_patch": False,
    }


def _failure_patch_review_contract_rows(*, stage: str, target_test: str, target_file: str) -> list[dict[str, Any]]:
    target = target_test or "selected cluster"
    file_target = target_file or "selected file"
    rows = [
        {
            "item": "review_scope_binding",
            "status": "required",
            "evidence": f"{stage}:{target}:{file_target}",
            "authorizes_patch_application": False,
            "authorizes_file_write": False,
            "authorizes_tool_execution": False,
            "authorizes_completion_claim": False,
            "authorizes_learning_record": False,
            "authorizes_approval": False,
            "reusable_for_next_patch": False,
        },
        {
            "item": "applied_patch_bridge_only",
            "status": "proof_only",
            "evidence": "applied patch evidence must bind back to the reviewed apply contract",
            "authorizes_patch_application": False,
            "authorizes_file_write": False,
            "authorizes_tool_execution": False,
            "authorizes_completion_claim": False,
            "authorizes_learning_record": False,
            "authorizes_approval": False,
            "reusable_for_next_patch": False,
        },
        {
            "item": "verification_receipt_boundary",
            "status": "required",
            "evidence": "focused verification, compile pass, rollback, and patch receipt hash remain separate proof",
            "authorizes_patch_application": False,
            "authorizes_file_write": False,
            "authorizes_tool_execution": False,
            "authorizes_completion_claim": False,
            "authorizes_learning_record": False,
            "authorizes_approval": False,
            "reusable_for_next_patch": False,
        },
        {
            "item": "completion_claim_boundary",
            "status": "review_only",
            "evidence": "ready evidence can enter completion review but cannot mark the regression fixed",
            "authorizes_patch_application": False,
            "authorizes_file_write": False,
            "authorizes_tool_execution": False,
            "authorizes_completion_claim": False,
            "authorizes_learning_record": False,
            "authorizes_approval": False,
            "reusable_for_next_patch": False,
        },
        {
            "item": "learning_record_boundary",
            "status": "post_closeout_only",
            "evidence": "durable learning requires closeout, after-action learning, regression link, and learning hash",
            "authorizes_patch_application": False,
            "authorizes_file_write": False,
            "authorizes_tool_execution": False,
            "authorizes_completion_claim": False,
            "authorizes_learning_record": False,
            "authorizes_approval": False,
            "reusable_for_next_patch": False,
        },
        {
            "item": "fresh_patch_review_token",
            "status": "required",
            "evidence": "every new repeated-failure patch needs a fresh cockpit, apply contract, receipt, and bridge",
            "authorizes_patch_application": False,
            "authorizes_file_write": False,
            "authorizes_tool_execution": False,
            "authorizes_completion_claim": False,
            "authorizes_learning_record": False,
            "authorizes_approval": False,
            "reusable_for_next_patch": False,
        },
    ]
    for row in rows:
        row.update(
            {
                "authorizes_model_call": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
            }
        )
    return rows


def _failure_patch_review_token_sha256(
    *,
    target_test: str,
    target_file: str,
    apply_contract_sha256: str,
    failure_regression_test_contract_sha256: str,
    patch_receipt_sha256: str,
    review_contract_rows: list[dict[str, Any]],
    authorizes_patch_application: bool = False,
    authorizes_file_write: bool = False,
    authorizes_tool_execution: bool = False,
    authorizes_model_call: bool = False,
    authorizes_personal_data_read: bool = False,
    authorizes_external_side_effect: bool = False,
    authorizes_approval: bool = False,
    authorizes_completion_claim: bool = False,
    authorizes_learning_record: bool = False,
    reusable_for_next_patch: bool = False,
) -> str:
    return _text_sha256(
        "\n".join(
            [
                "failure_patch_review_token_v1",
                target_test,
                target_file,
                apply_contract_sha256,
                failure_regression_test_contract_sha256,
                patch_receipt_sha256,
                repr(
                    [
                        (
                            row.get("item"),
                            row.get("status"),
                            row.get("evidence"),
                            row.get("authorizes_patch_application"),
                            row.get("authorizes_file_write"),
                            row.get("authorizes_tool_execution"),
                            row.get("authorizes_model_call"),
                            row.get("authorizes_personal_data_read"),
                            row.get("authorizes_external_side_effect"),
                            row.get("authorizes_approval"),
                            row.get("authorizes_completion_claim"),
                            row.get("authorizes_learning_record"),
                            row.get("reusable_for_next_patch"),
                        )
                        for row in review_contract_rows
                    ]
                ),
                "proof_only_failure_patch_review",
                f"authorizes_patch_application={authorizes_patch_application}",
                f"authorizes_file_write={authorizes_file_write}",
                f"authorizes_tool_execution={authorizes_tool_execution}",
                f"authorizes_model_call={authorizes_model_call}",
                f"authorizes_personal_data_read={authorizes_personal_data_read}",
                f"authorizes_external_side_effect={authorizes_external_side_effect}",
                f"authorizes_approval={authorizes_approval}",
                f"authorizes_completion_claim={authorizes_completion_claim}",
                f"authorizes_learning_record={authorizes_learning_record}",
                f"reusable_for_next_patch={reusable_for_next_patch}",
            ]
        )
    )


def _failure_patch_review_token_metadata(
    *,
    target_test: str,
    target_file: str,
    apply_contract_sha256: str,
    failure_regression_test_contract_sha256: str,
    patch_receipt_sha256: str,
    review_contract_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    token = _failure_patch_review_token_sha256(
        target_test=target_test,
        target_file=target_file,
        apply_contract_sha256=apply_contract_sha256,
        failure_regression_test_contract_sha256=failure_regression_test_contract_sha256,
        patch_receipt_sha256=patch_receipt_sha256,
        review_contract_rows=review_contract_rows,
    )
    return {
        "failure_patch_review_token_sha256": token,
        "failure_patch_review_token_present": _looks_like_sha256(token),
        "review_token_authorizes_patch_application": False,
        "review_token_authorizes_file_write": False,
        "review_token_authorizes_tool_execution": False,
        "review_token_authorizes_model_call": False,
        "review_token_authorizes_personal_data_read": False,
        "review_token_authorizes_external_side_effect": False,
        "review_token_authorizes_approval": False,
        "review_token_authorizes_completion_claim": False,
        "review_token_authorizes_learning_record": False,
        "review_token_reusable_for_next_patch": False,
        "review_token_binds_exact_test_contract": True,
        "next_patch_requires_fresh_review_token": True,
    }


def _failure_patch_review_contract_ready(metadata: dict[str, Any]) -> bool:
    rows = metadata.get("patch_review_contract_rows")
    token = str(metadata.get("failure_patch_review_token_sha256") or "")
    target_test = str(metadata.get("target_test") or "")
    target_file = str(metadata.get("target_file") or "")
    apply_contract_sha256 = str(metadata.get("apply_contract_sha256") or "")
    exact_test_contract_sha256 = str(metadata.get("failure_regression_test_contract_sha256") or "")
    patch_receipt_sha256 = str(metadata.get("patch_receipt_sha256") or "")
    expected_items = {
        "review_scope_binding",
        "applied_patch_bridge_only",
        "verification_receipt_boundary",
        "completion_claim_boundary",
        "learning_record_boundary",
        "fresh_patch_review_token",
    }
    if metadata.get("patch_review_contract_ready") is not True:
        return False
    if not isinstance(rows, list) or len(rows) != 6 or metadata.get("patch_review_contract_row_count") != len(rows):
        return False
    if {row.get("item") for row in rows if isinstance(row, dict)} != expected_items:
        return False
    if not all(isinstance(row, dict) for row in rows):
        return False
    if not all(_looks_like_sha256(value) for value in [token, apply_contract_sha256, exact_test_contract_sha256, patch_receipt_sha256]):
        return False
    recomputed = _failure_patch_review_token_sha256(
        target_test=target_test,
        target_file=target_file,
        apply_contract_sha256=apply_contract_sha256,
        failure_regression_test_contract_sha256=exact_test_contract_sha256,
        patch_receipt_sha256=patch_receipt_sha256,
        review_contract_rows=rows,
    )
    if recomputed != token or metadata.get("failure_patch_review_token_present") is not True:
        return False
    if metadata.get("review_token_binds_exact_test_contract") is not True:
        return False
    if metadata.get("next_patch_requires_fresh_review_token") is not True:
        return False
    for flag in [
        "review_authorizes_patch_application",
        "review_authorizes_file_write",
        "review_authorizes_tool_execution",
        "review_authorizes_model_call",
        "review_authorizes_personal_data_read",
        "review_authorizes_external_side_effect",
        "review_authorizes_approval",
        "review_authorizes_completion_claim",
        "review_authorizes_learning_record",
        "review_reusable_for_next_patch",
        "review_token_authorizes_patch_application",
        "review_token_authorizes_file_write",
        "review_token_authorizes_tool_execution",
        "review_token_authorizes_model_call",
        "review_token_authorizes_personal_data_read",
        "review_token_authorizes_external_side_effect",
        "review_token_authorizes_approval",
        "review_token_authorizes_completion_claim",
        "review_token_authorizes_learning_record",
        "review_token_reusable_for_next_patch",
    ]:
        if metadata.get(flag) is not False:
            return False
    for row in rows:
        for flag in [
            "authorizes_patch_application",
            "authorizes_file_write",
            "authorizes_tool_execution",
            "authorizes_model_call",
            "authorizes_personal_data_read",
            "authorizes_external_side_effect",
            "authorizes_approval",
            "authorizes_completion_claim",
            "authorizes_learning_record",
            "reusable_for_next_patch",
        ]:
            if row.get(flag) is not False:
                return False
    return True


def _failure_patch_handoff_ready(metadata: dict[str, Any]) -> bool:
    if metadata.get("handoff_state") != "FAILURE_PATCH_HANDOFF_READY_FOR_COMPLETION_REVIEW":
        return False
    if metadata.get("ready_for_completion_review") is not True:
        return False
    if metadata.get("missing") not in ([], None) or int(metadata.get("missing_count") or 0) != 0:
        return False
    for flag in [
        "completion_claim_ready",
        "apply_contract_reviewed",
        "apply_contract_hash_present",
        "apply_contract_hash_matches_expected",
        "failure_regression_test_contract_present",
        "failure_regression_test_contract_ready",
        "exact_test_contract_ready",
        "application_bridge_ready",
        "applied_patch_bound_to_contract",
        "patch_receipt_ready",
        "focused_verification_present",
        "compile_pass_present",
        "rollback_note_present",
        "pre_patch_failing_test_receipt_present",
        "test_first_receipt_present",
        "patch_receipt_hash_present",
    ]:
        if metadata.get(flag) is not True:
            return False
    if not all(
        _looks_like_sha256(str(metadata.get(key) or ""))
        for key in [
            "apply_contract_sha256",
            "supplied_apply_contract_sha256",
            "failure_regression_test_contract_sha256",
            "patch_receipt_sha256",
            "failure_patch_review_token_sha256",
        ]
    ):
        return False
    if metadata.get("apply_contract_sha256") != metadata.get("supplied_apply_contract_sha256"):
        return False
    if not _failure_patch_review_contract_ready(metadata):
        return False
    for flag in [
        "exact_test_contract_authorizes_file_write",
        "exact_test_contract_authorizes_patch_application",
        "exact_test_contract_authorizes_tool_execution",
        "exact_test_contract_authorizes_completion_claim",
        "exact_test_contract_reusable_for_next_patch",
    ]:
        if metadata.get(flag) is not False:
            return False
    if not metadata.get("pre_claim_commands") or metadata.get("pre_claim_command_count") != len(metadata.get("pre_claim_commands") or []):
        return False
    if not metadata.get("post_claim_commands") or metadata.get("post_claim_command_count") != len(metadata.get("post_claim_commands") or []):
        return False
    if "evidence ledger" not in " ".join(str(command) for command in (metadata.get("post_claim_commands") or [])):
        return False
    for flag in [
        "writes_files",
        "writes_memory",
        "writes_notes",
        "writes_database",
        "controls_computer",
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "external_side_effect",
        "requires_approval",
    ]:
        if metadata.get(flag) is not False:
            return False
    return True


def _failure_learning_closure_token_sha256(
    *,
    target_test: str,
    target_file: str,
    patch_receipt_sha256: str,
    learning_record_sha256: str,
    apply_contract_sha256: str,
    failure_patch_review_token_sha256: str,
    failure_learning_record_token_sha256: str,
    after_action_learning: str,
    learning_record_target: str,
    regression_link: str,
    durability_note: str,
    review_contract_rows: list[dict[str, Any]],
    closure_stage_rows: list[tuple[str, Any, Any, Any]],
    proof_queue: list[str],
    authorizes_patch_application: bool = False,
    authorizes_file_write: bool = False,
    authorizes_tool_execution: bool = False,
    authorizes_model_call: bool = False,
    authorizes_personal_data_read: bool = False,
    authorizes_external_side_effect: bool = False,
    authorizes_approval: bool = False,
    authorizes_completion_claim: bool = False,
    authorizes_learning_record: bool = False,
    reusable_for_next_patch: bool = False,
    reusable_for_next_learning_record: bool = False,
    reusable_for_next_completion_claim: bool = False,
) -> str:
    return _text_sha256(
        "\n".join(
            [
                "failure_learning_closure_token_v1",
                target_test,
                target_file,
                patch_receipt_sha256,
                learning_record_sha256,
                apply_contract_sha256,
                failure_patch_review_token_sha256,
                failure_learning_record_token_sha256,
                after_action_learning,
                learning_record_target,
                regression_link,
                durability_note,
                repr(
                    [
                        (
                            row.get("item"),
                            row.get("status"),
                            row.get("authorizes_patch_application"),
                            row.get("authorizes_file_write"),
                            row.get("authorizes_tool_execution"),
                            row.get("authorizes_model_call"),
                            row.get("authorizes_personal_data_read"),
                            row.get("authorizes_external_side_effect"),
                            row.get("authorizes_approval"),
                            row.get("authorizes_completion_claim"),
                            row.get("authorizes_learning_record"),
                            row.get("reusable_for_next_patch"),
                        )
                        for row in review_contract_rows
                    ]
                ),
                repr(
                    [
                        (
                            row.get("stage"),
                            row.get("state"),
                            row.get("ready"),
                            tuple(str(item) for item in (row.get("missing") or []) if str(item).strip()),
                            row.get("authorizes_patch_application"),
                            row.get("authorizes_file_write"),
                            row.get("authorizes_tool_execution"),
                            row.get("authorizes_model_call"),
                            row.get("authorizes_personal_data_read"),
                            row.get("authorizes_external_side_effect"),
                            row.get("authorizes_approval"),
                            row.get("authorizes_completion_claim"),
                            row.get("authorizes_learning_record"),
                            row.get("reusable_for_next_patch"),
                            row.get("reusable_for_next_learning_record"),
                            row.get("reusable_for_next_completion_claim"),
                        )
                        if isinstance(row, dict)
                        else (
                            row[0],
                            row[1],
                            row[2],
                            tuple(str(item) for item in (row[3] or []) if str(item).strip()),
                        )
                        for row in closure_stage_rows
                    ]
                ),
                repr(list(proof_queue)),
                "proof_only_failure_learning_closure",
                f"authorizes_patch_application={authorizes_patch_application}",
                f"authorizes_file_write={authorizes_file_write}",
                f"authorizes_tool_execution={authorizes_tool_execution}",
                f"authorizes_model_call={authorizes_model_call}",
                f"authorizes_personal_data_read={authorizes_personal_data_read}",
                f"authorizes_external_side_effect={authorizes_external_side_effect}",
                f"authorizes_approval={authorizes_approval}",
                f"authorizes_completion_claim={authorizes_completion_claim}",
                f"authorizes_learning_record={authorizes_learning_record}",
                f"reusable_for_next_patch={reusable_for_next_patch}",
                f"reusable_for_next_learning_record={reusable_for_next_learning_record}",
                f"reusable_for_next_completion_claim={reusable_for_next_completion_claim}",
            ]
        )
    )


def _failure_learning_closure_token_boundary_rows(
    *,
    token_sha256: str,
    record_token_sha256: str,
    source: str,
) -> list[dict[str, Any]]:
    rows = [
        {
            "item": "failure_learning_closure_token",
            "status": "present" if _looks_like_sha256(token_sha256) else "missing",
            "evidence": token_sha256 or "missing",
        },
        {
            "item": "record_token_carry_forward",
            "status": "proof_only",
            "evidence": record_token_sha256 or "missing",
        },
        {
            "item": "fresh_patch_learning_review",
            "status": "fresh_review_required",
            "evidence": "new repeated-failure patches need fresh cockpit, apply contract, patch receipt, closeout, learning record, and closure ledger",
        },
    ]
    for row in rows:
        row.update(
            {
                "source": source,
                "token_sha256": token_sha256,
                "record_token_sha256": record_token_sha256,
                "authorizes_patch_application": False,
                "authorizes_file_write": False,
                "authorizes_tool_execution": False,
                "authorizes_model_call": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "authorizes_approval": False,
                "authorizes_completion_claim": False,
                "authorizes_learning_record": False,
                "reusable_for_next_patch": False,
                "reusable_for_next_learning_record": False,
                "reusable_for_next_completion_claim": False,
            }
        )
    return rows


def _failure_learning_closure_token_boundary_ready(
    rows: Any,
    *,
    token_sha256: Any,
    record_token_sha256: Any,
) -> bool:
    token = str(token_sha256 or "")
    record_token = str(record_token_sha256 or "")
    if not _looks_like_sha256(token) or not _looks_like_sha256(record_token):
        return False
    if not isinstance(rows, list) or len(rows) != 3:
        return False
    expected = [
        ("failure_learning_closure_token", "present"),
        ("record_token_carry_forward", "proof_only"),
        ("fresh_patch_learning_review", "fresh_review_required"),
    ]
    for row, (item, status) in zip(rows, expected):
        if not isinstance(row, dict):
            return False
        if row.get("item") != item or row.get("status") != status:
            return False
        if row.get("source") != "failure_learning_closure_ledger":
            return False
        if row.get("token_sha256") != token or row.get("record_token_sha256") != record_token:
            return False
        for key in [
            "authorizes_patch_application",
            "authorizes_file_write",
            "authorizes_tool_execution",
            "authorizes_model_call",
            "authorizes_personal_data_read",
            "authorizes_external_side_effect",
            "authorizes_approval",
            "authorizes_completion_claim",
            "authorizes_learning_record",
            "reusable_for_next_patch",
            "reusable_for_next_learning_record",
            "reusable_for_next_completion_claim",
        ]:
            if row.get(key) is not False:
                return False
    return True


def _failure_learning_record_token_sha256(
    *,
    target_test: str,
    target_file: str,
    after_action_learning: str,
    record_target: str,
    regression_link: str,
    durability_note: str,
    patch_receipt_sha256: str,
    learning_record_sha256: str,
    apply_contract_sha256: str,
    review_contract_rows: list[dict[str, Any]],
    authorizes_learning_record: bool = False,
    authorizes_memory_write: bool = False,
    authorizes_file_write: bool = False,
    authorizes_tool_execution: bool = False,
    authorizes_model_call: bool = False,
    authorizes_personal_data_read: bool = False,
    authorizes_external_side_effect: bool = False,
    authorizes_approval: bool = False,
    authorizes_completion_claim: bool = False,
    reusable_for_next_learning_record: bool = False,
    reusable_for_next_patch: bool = False,
) -> str:
    return _text_sha256(
        "\n".join(
            [
                "failure_learning_record_token_v1",
                target_test,
                target_file,
                after_action_learning,
                record_target,
                regression_link,
                durability_note,
                patch_receipt_sha256,
                learning_record_sha256,
                apply_contract_sha256,
                repr(
                    [
                        (
                            row.get("item"),
                            row.get("status"),
                            row.get("authorizes_learning_record"),
                            row.get("authorizes_file_write"),
                            row.get("authorizes_tool_execution"),
                            row.get("authorizes_completion_claim"),
                            row.get("authorizes_model_call"),
                            row.get("authorizes_personal_data_read"),
                            row.get("authorizes_external_side_effect"),
                            row.get("authorizes_approval"),
                            row.get("reusable_for_next_patch"),
                        )
                        for row in review_contract_rows
                    ]
                ),
                "proof_only_failure_learning_record",
                f"authorizes_learning_record={authorizes_learning_record}",
                f"authorizes_memory_write={authorizes_memory_write}",
                f"authorizes_file_write={authorizes_file_write}",
                f"authorizes_tool_execution={authorizes_tool_execution}",
                f"authorizes_model_call={authorizes_model_call}",
                f"authorizes_personal_data_read={authorizes_personal_data_read}",
                f"authorizes_external_side_effect={authorizes_external_side_effect}",
                f"authorizes_approval={authorizes_approval}",
                f"authorizes_completion_claim={authorizes_completion_claim}",
                f"reusable_for_next_learning_record={reusable_for_next_learning_record}",
                f"reusable_for_next_patch={reusable_for_next_patch}",
            ]
        )
    )


def _failure_learning_record_token_ready(metadata: dict[str, Any]) -> bool:
    token = str(metadata.get("failure_learning_record_token_sha256") or "")
    if not _looks_like_sha256(token) or metadata.get("failure_learning_record_token_present") is not True:
        return False
    if metadata.get("next_learning_record_requires_fresh_record_token") is not True:
        return False
    review_contract_rows = list(metadata.get("patch_review_contract_rows") or [])
    if not _failure_patch_review_contract_ready(metadata):
        return False
    recomputed = _failure_learning_record_token_sha256(
        target_test=str(metadata.get("target_test") or ""),
        target_file=str(metadata.get("target_file") or ""),
        after_action_learning=str(metadata.get("after_action_learning") or ""),
        record_target=str(metadata.get("learning_record_target") or ""),
        regression_link=str(metadata.get("regression_link") or ""),
        durability_note=str(metadata.get("durability_note") or ""),
        patch_receipt_sha256=str(metadata.get("patch_receipt_sha256") or ""),
        learning_record_sha256=str(metadata.get("learning_record_sha256") or ""),
        apply_contract_sha256=str(metadata.get("apply_contract_sha256") or ""),
        review_contract_rows=review_contract_rows,
    )
    if recomputed != token:
        return False
    for flag in [
        "record_token_authorizes_learning_record",
        "record_token_authorizes_memory_write",
        "record_token_authorizes_file_write",
        "record_token_authorizes_tool_execution",
        "record_token_authorizes_model_call",
        "record_token_authorizes_personal_data_read",
        "record_token_authorizes_external_side_effect",
        "record_token_authorizes_approval",
        "record_token_authorizes_completion_claim",
        "record_token_reusable_for_next_learning_record",
    ]:
        if metadata.get(flag) is not False:
            return False
    return True


def make_feedback_tools(store: MemoryStore, vault: ObsidianVault):
    def recent_feedback(limit: int):
        limit = _bounded_int(limit, 12)
        return _safe_feedback_rows(store.list_memories(category="feedback", limit=limit), limit=limit)

    def build_feedback_report(limit: int) -> tuple[str, dict[str, Any]]:
        limit = _bounded_int(limit, 12)
        rows, unreadable_feedback_rows = recent_feedback(limit)
        lines = [
            "Jarvis feedback report:",
            "Use this to improve behavior through reviewable memory, preferences, skills, or code changes. It never changes risky permissions by itself.",
            "",
        ]
        if not rows:
            lines.extend(
                [
                    "Recent feedback:",
                    "- No feedback captured yet.",
                    "",
                    "Good feedback commands:",
                    "- feedback: Jarvis should explain approvals more clearly",
                    "- Jarvis feedback: responses should be shorter when I ask what changed",
                ]
            )
            if unreadable_feedback_rows:
                lines.extend(["", f"- hidden malformed feedback rows: {unreadable_feedback_rows}"])
            handoff = _feedback_report_handoff(source="feedback_report", count=0, themes={}, limit=limit, unreadable_feedback_rows=unreadable_feedback_rows)
            return "\n".join(lines), _feedback_handoff_metadata(
                "feedback_report_handoff",
                handoff,
                count=0,
                readable_feedback_rows=0,
                unreadable_feedback_rows=unreadable_feedback_rows,
                themes={},
                limit=limit,
            )

        themes: dict[str, int] = {}
        lines.append("Recent feedback:")
        for row in rows:
            theme = _feedback_theme(row["body"])
            themes[theme] = themes.get(theme, 0) + 1
            lines.append(f"- #{row['id']} [{theme}] {_safe_display(row['title'], MAX_FEEDBACK_TITLE_CHARS)}: {_safe_display(row['body'], 180)}")
        if unreadable_feedback_rows:
            lines.append(f"- hidden malformed feedback rows: {unreadable_feedback_rows}")

        lines.extend(["", "Themes:"])
        for theme, count in sorted(themes.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {theme}: {count}")

        lines.extend(
            [
                "",
                "Safe improvement paths:",
                "- Convert stable behavior feedback into `set preference ...`.",
                "- Turn repeated workflow feedback into a reviewed skill draft.",
                "- Use code changes only after a smoke test can prove the new behavior.",
                "- Keep shell, personal data, reminders, external actions, and computer control approval-gated.",
            ]
        )
        handoff = _feedback_report_handoff(source="feedback_report", count=len(rows), themes=themes, limit=limit, unreadable_feedback_rows=unreadable_feedback_rows)
        return "\n".join(lines), _feedback_handoff_metadata(
            "feedback_report_handoff",
            handoff,
            count=len(rows),
            readable_feedback_rows=len(rows),
            unreadable_feedback_rows=unreadable_feedback_rows,
            themes=themes,
            limit=limit,
        )

    def build_feedback_actions(limit: int) -> tuple[str, dict[str, Any]]:
        limit = _bounded_int(limit, 12)
        rows, unreadable_feedback_rows = recent_feedback(limit)
        lines = [
            "Jarvis feedback actions:",
            "These are reviewable suggestions only. Jarvis does not auto-change preferences, skills, code, permissions, or risky tools from feedback.",
            "",
        ]
        if not rows:
            lines.extend(
                [
                    "No feedback captured yet.",
                    "Capture one with `feedback: ...`, then rerun `feedback actions`.",
                ]
            )
            if unreadable_feedback_rows:
                lines.extend(["", f"- hidden malformed feedback rows: {unreadable_feedback_rows}"])
            handoff = _feedback_report_handoff(source="feedback_actions", count=0, themes=[], limit=limit, unreadable_feedback_rows=unreadable_feedback_rows)
            return "\n".join(lines), _feedback_handoff_metadata(
                "feedback_actions_handoff",
                handoff,
                count=0,
                readable_feedback_rows=0,
                unreadable_feedback_rows=unreadable_feedback_rows,
                themes=[],
                limit=limit,
            )

        themes: dict[str, list[str]] = {}
        for row in rows:
            themes.setdefault(_feedback_theme(row["body"]), []).append(row["body"])

        lines.append("Suggested preference updates:")
        if "conversation" in themes:
            lines.append("- `set preference response length to concise when asking what changed category communication`")
        if "safety" in themes:
            lines.append("- `set preference safety explanations to clear before risky actions category safety`")
        if "memory" in themes:
            lines.append("- `set preference memory behavior to cite saved context when relevant category memory`")
        if not any(theme in themes for theme in ("conversation", "safety", "memory")):
            lines.append("- No obvious preference update yet; capture one more concrete feedback example.")

        lines.extend(["", "Skill or workflow candidates:"])
        if "tools" in themes:
            lines.append("- Draft a skill for the repeated tool/workflow friction after one more successful example.")
        if "safety" in themes:
            lines.append("- Draft a short approval-explanation skill if approval confusion repeats.")
        if "conversation" in themes:
            lines.append("- Draft a conversational-summary style skill if tone/length feedback repeats.")
        if not any(theme in themes for theme in ("tools", "safety", "conversation")):
            lines.append("- No skill candidate is strong yet; keep collecting feedback.")

        lines.extend(["", "Test/code candidates:"])
        if "performance" in themes:
            lines.append("- Add a smoke test that keeps the reported slow path bounded or visibly summarized.")
        if "safety" in themes:
            lines.append("- Add/extend a safety smoke test before changing approval wording or behavior.")
        if "conversation" in themes:
            lines.append("- Add/extend a chat smoke test for the expected response style.")
        if not any(theme in themes for theme in ("performance", "safety", "conversation")):
            lines.append("- No code change candidate yet; prefer preferences or skill drafts first.")

        lines.extend(
            [
                "",
                "Still approval-gated:",
                "- Shell, file writes outside safe tools, clipboard reads, personal data, reminders, external actions, and computer control.",
            ]
        )
        if unreadable_feedback_rows:
            lines.extend(["", f"- hidden malformed feedback rows: {unreadable_feedback_rows}"])
        sorted_themes = sorted(themes)
        handoff = _feedback_report_handoff(source="feedback_actions", count=len(rows), themes=sorted_themes, limit=limit, unreadable_feedback_rows=unreadable_feedback_rows)
        return "\n".join(lines), _feedback_handoff_metadata(
            "feedback_actions_handoff",
            handoff,
            count=len(rows),
            readable_feedback_rows=len(rows),
            unreadable_feedback_rows=unreadable_feedback_rows,
            themes=sorted_themes,
            limit=limit,
        )

    def record_feedback(args: dict[str, Any]) -> ToolResult:
        body = str(args.get("body") or "").strip()
        if not body:
            return _feedback_input_failure("Feedback body is empty.", reason="missing_body")
        if len(body) > MAX_FEEDBACK_BODY_CHARS:
            return _feedback_input_failure(
                f"Feedback body is too large ({len(body)} chars). Limit is {MAX_FEEDBACK_BODY_CHARS}.",
                reason="body_too_large",
                limit=MAX_FEEDBACK_BODY_CHARS,
                body_chars=len(body),
            )
        raw_theme = args.get("theme")
        raw_title = args.get("title")
        theme = _short(raw_theme, MAX_FEEDBACK_THEME_CHARS).lower() or _feedback_theme(body)
        title = _short(raw_title, MAX_FEEDBACK_TITLE_CHARS) or f"Jarvis feedback: {theme}"
        if LOCAL_PATH_RE.search(body):
            return _feedback_input_failure(
                "Feedback body looks like a local file path; summarize it without the path.",
                reason="invalid_body",
                raw_body=_short_metadata(body),
            )
        if LOCAL_PATH_RE.search(theme):
            return _feedback_input_failure(
                "Feedback theme looks like a local file path; use a short category like safety, memory, or conversation.",
                reason="invalid_theme",
                raw_theme=_short_metadata(raw_theme),
            )
        if LOCAL_PATH_RE.search(title):
            return _feedback_input_failure(
                "Feedback title looks like a local file path; use a short descriptive title.",
                reason="invalid_title",
                raw_title=_short_metadata(raw_title),
            )
        record = MemoryRecord("feedback", title, body, "jarvis-feedback")
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
            return ToolResult(
                "record_feedback",
                False,
                (
                    f"Feedback #{memory_id} was recorded, but its note projection is "
                    f"{projection.status}. Run `repair memory projections`; do not record it again."
                ),
                _feedback_handoff_metadata(
                    "feedback_capture_handoff",
                    _feedback_capture_handoff(
                        memory_id=memory_id,
                        theme=theme,
                        body_chars=len(body),
                        title_chars=len(title),
                    ),
                    writes=True,
                    memory_id=memory_id,
                    theme=theme,
                    projection_status=projection.status,
                    projection_pending=True,
                    body_chars=len(body),
                    title_chars=len(title),
                ),
            )
        path = vault.root_path / projection.path_display
        return ToolResult(
            "record_feedback",
            True,
            f"Feedback recorded as #{memory_id} [{theme}]: {body}",
            _feedback_handoff_metadata(
                "feedback_capture_handoff",
                _feedback_capture_handoff(memory_id=memory_id, theme=theme, body_chars=len(body), title_chars=len(title)),
                writes=True,
                memory_id=memory_id,
                theme=theme,
                path=str(path),
                body_chars=len(body),
                title_chars=len(title),
            ),
        )

    def feedback_report(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 12)
        body, metadata = build_feedback_report(limit)
        return ToolResult("feedback_report", True, body, metadata)

    def save_feedback_report(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 12)
        with store.generated_report_publication_fence():
            body, metadata = build_feedback_report(limit)
            path, _content_sha256, _source_revision = vault.write_feedback_report_with_evidence(
                f"# Feedback Report\n\n{body}",
                store_identity=store.get_store_identity(),
                source_payload={"body": body, "metadata": metadata},
            )
        path_display = _safe_vault_path_display(path, vault)
        metadata = _feedback_handoff_metadata(
            "feedback_report_handoff",
            _feedback_report_handoff(
                source="save_feedback_report",
                count=int(metadata.get("count") or 0),
                themes=metadata.get("themes") or {},
                limit=limit,
                saved=True,
                path_display=path_display,
                unreadable_feedback_rows=int(metadata.get("unreadable_feedback_rows") or 0),
            ),
            writes=True,
            writes_memory=False,
            writes_database=False,
            count=int(metadata.get("count") or 0),
            themes=metadata.get("themes") or {},
            limit=limit,
            readable_feedback_rows=int(metadata.get("readable_feedback_rows") or 0),
            unreadable_feedback_rows=int(metadata.get("unreadable_feedback_rows") or 0),
            path=str(path),
            path_display=path_display,
        )
        return ToolResult("save_feedback_report", True, f"Feedback report saved: {path_display}\n\n{body}", metadata)

    def feedback_actions(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 12)
        body, metadata = build_feedback_actions(limit)
        return ToolResult("feedback_actions", True, body, metadata)

    def failure_to_test_preview(args: dict[str, Any]) -> ToolResult:
        failure = str(args.get("failure") or args.get("body") or args.get("request") or "").strip()
        if not failure:
            return ToolResult(
                "failure_to_test_preview",
                False,
                "Failure-to-test preview needs a failure description.",
                _safe_metadata(reason="missing_failure", failure_chars=0),
            )
        if len(failure) > MAX_FEEDBACK_BODY_CHARS:
            return ToolResult(
                "failure_to_test_preview",
                False,
                f"Failure description is too large ({len(failure)} chars). Limit is {MAX_FEEDBACK_BODY_CHARS}.",
                _safe_metadata(reason="failure_too_large", limit=MAX_FEEDBACK_BODY_CHARS, failure_chars=len(failure)),
            )
        theme = _feedback_theme(failure)
        surface, test_name, assertions, preference_candidate = _failure_surface(theme, failure)
        summary = _safe_display(failure.replace("\n", " "), 220)
        lines = [
            "Jarvis failure-to-test preview:",
            "This converts a miss into a read-only, reviewable regression shape without writing tests, preferences, skills, code, memory, or notes.",
            "",
            "Failure summary:",
            f"- {summary}",
            "",
            "Likely theme:",
            f"- {theme}",
            "",
            "Likely regression surface:",
            f"- {surface}",
            "",
            "Proposed smoke test:",
            f"- `{test_name}`",
            "",
            "Proposed assertions:",
        ]
        lines.extend(f"- {assertion}" for assertion in assertions)
        lines.extend(
            [
                "",
                "Preference or skill candidate:",
                f"- {preference_candidate}",
                "",
                "Safe next commands:",
                f"- `feedback: {summary}`",
                "- `feedback actions`",
                "- `learning review`",
                "",
                "Boundary:",
                "- Read-only preview only. It does not execute tools, call models, write files, change memory, queue approvals, speak, or control the computer.",
            ]
        )
        return ToolResult(
            "failure_to_test_preview",
            True,
            "\n".join(lines),
            _safe_metadata(
                theme=theme,
                surface=surface,
                suggested_test=test_name,
                assertion_count=len(assertions),
                failure_chars=len(failure),
            ),
        )

    def build_failure_clusters(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        clusters: dict[str, dict[str, Any]] = {}
        for row in rows:
            body = str(row["body"])
            theme = _feedback_theme(body)
            surface, test_name, assertions, preference_candidate = _failure_surface(theme, body)
            key = _cluster_key(theme, surface)
            cluster = clusters.setdefault(
                key,
                {
                    "theme": theme,
                    "surface": surface,
                    "test_name": test_name,
                    "target_file": _target_test_file(test_name),
                    "assertions": assertions,
                    "expected_behavior": assertions[0],
                    "rollback_note": _rollback_note(test_name),
                    "preference_candidate": preference_candidate,
                    "rows": [],
                },
            )
            cluster["rows"].append(row)
        ranked = sorted(clusters.values(), key=lambda item: (-len(item["rows"]), item["surface"]))
        strong = [cluster for cluster in ranked if len(cluster["rows"]) >= 2]
        return ranked, strong

    def repeated_failure_clusters(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 24)
        rows, unreadable_feedback_rows = recent_feedback(limit)
        lines = [
            "Jarvis repeated-failure cluster report:",
            "This is read-only. It groups recent feedback/misses into likely regression surfaces so repeated failures can become reviewed tests, preferences, or skills.",
            "",
        ]
        if not rows:
            lines.extend(
                [
                    "No captured feedback yet.",
                    "Safe next commands:",
                    "- `feedback: Jarvis overlapped dashboard text and hid diagnostics`",
                    "- `failure to test: Jarvis overlapped dashboard text and hid diagnostics`",
                ]
            )
            if unreadable_feedback_rows:
                lines.extend(["", f"- hidden malformed feedback rows: {unreadable_feedback_rows}"])
            return ToolResult(
                "repeated_failure_clusters",
                True,
                "\n".join(lines),
                _safe_metadata(
                    count=0,
                    readable_feedback_rows=0,
                    unreadable_feedback_rows=unreadable_feedback_rows,
                    clusters=0,
                    limit=limit,
                ),
            )

        ranked, strong = build_failure_clusters(rows)
        lines.append("Clusters:")
        for index, cluster in enumerate(ranked[:8], start=1):
            examples = cluster["rows"][:3]
            strength = "strong repeated signal" if len(cluster["rows"]) >= 2 else "single signal"
            lines.extend(
                [
                    f"{index}. {cluster['surface']} ({strength})",
                    f"   - theme: {cluster['theme']}",
                    f"   - count: {len(cluster['rows'])}",
                    f"   - proposed smoke test: `{cluster['test_name']}`",
                    f"   - target file: `{cluster['target_file']}`",
                    f"   - expected behavior: {cluster['expected_behavior']}",
                    f"   - rollback: {cluster['rollback_note']}",
                    f"   - candidate: {cluster['preference_candidate']}",
                    "   - examples:",
                ]
            )
            for example in examples:
                lines.append(f"     - #{example['id']} {example['body'][:140]}")
            lines.append("   - observed failure: " + examples[0]["body"][:160])
            lines.append("   - first assertion: " + cluster["assertions"][0])

        lines.extend(["", "Ranked next safe move:"])
        if strong:
            top = strong[0]
            lines.append(f"- Promote `{top['test_name']}` to a real smoke test only after reviewing the examples above.")
        else:
            lines.append("- Keep collecting examples; no cluster is repeated enough to justify a code change yet.")

        lines.extend(
            [
                "",
                "Safe next commands:",
                "- `failure to test: <one concrete miss>`",
                "- `feedback actions`",
                "- `learning review`",
                "",
                "Boundary:",
                "- Read-only report only. It does not write files, change memory, queue approvals, execute tools, speak, or control the computer.",
            ]
        )
        if unreadable_feedback_rows:
            lines.extend(["", f"- hidden malformed feedback rows: {unreadable_feedback_rows}"])
        return ToolResult(
            "repeated_failure_clusters",
            True,
            "\n".join(lines),
            _safe_metadata(
                count=len(rows),
                readable_feedback_rows=len(rows),
                unreadable_feedback_rows=unreadable_feedback_rows,
                clusters=len(ranked),
                repeated_clusters=len(strong),
                top_surface=ranked[0]["surface"] if ranked else "",
                top_test=ranked[0]["test_name"] if ranked else "",
                top_target_file=ranked[0]["target_file"] if ranked else "",
                top_expected_behavior=ranked[0]["expected_behavior"] if ranked else "",
                top_observed_failure=ranked[0]["rows"][0]["body"][:220] if ranked and ranked[0]["rows"] else "",
                top_rollback_note=ranked[0]["rollback_note"] if ranked else "",
                cluster_rows=[
                    {
                        "surface": cluster["surface"],
                        "theme": cluster["theme"],
                        "count": len(cluster["rows"]),
                        "target_test": cluster["test_name"],
                        "target_file": cluster["target_file"],
                        "focused_command": _focused_command_for_test(cluster["test_name"]),
                        "expected_behavior": cluster["expected_behavior"],
                        "observed_failures": [row["body"][:220] for row in cluster["rows"][:4]],
                        "rollback_note": cluster["rollback_note"],
                        **_failure_regression_test_contract_metadata(
                            cluster,
                            focused_command=_focused_command_for_test(cluster["test_name"]),
                        ),
                    }
                    for cluster in ranked[:8]
                ],
                limit=limit,
            ),
        )

    def failure_promotion_packet(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 24)
        selector = str(args.get("cluster") or args.get("surface") or args.get("test") or "").strip().lower()
        rows, unreadable_feedback_rows = recent_feedback(limit)
        ranked, strong = build_failure_clusters(rows)
        candidates = strong or ranked
        if selector:
            candidates = [
                cluster
                for cluster in candidates
                if selector in cluster["surface"].lower() or selector in cluster["test_name"].lower() or selector in cluster["theme"].lower()
            ]
        if not candidates:
            return ToolResult(
                "failure_promotion_packet",
                False,
                "No repeated failure cluster is ready for promotion. Run `failure clusters` after capturing at least two similar misses.",
                _safe_metadata(
                    count=len(rows),
                    readable_feedback_rows=len(rows),
                    unreadable_feedback_rows=unreadable_feedback_rows,
                    clusters=len(ranked),
                    repeated_clusters=len(strong),
                    selector=selector,
                    promoted=False,
                    limit=limit,
                ),
            )

        cluster = candidates[0]
        examples = cluster["rows"][:4]
        test_file = _target_test_file(cluster["test_name"])
        focused_command = _focused_command_for_test(cluster["test_name"])
        exact_test_contract = _failure_regression_test_contract_metadata(cluster, focused_command=focused_command)
        lines = [
            "Jarvis failure promotion packet:",
            "This is a read-only promotion plan. It drafts how a reviewed cluster could become a real smoke test, preference, or skill without editing code.",
            "",
            "Selected cluster:",
            f"- surface: {cluster['surface']}",
            f"- theme: {cluster['theme']}",
            f"- repeated examples: {len(cluster['rows'])}",
            f"- target smoke test: `{cluster['test_name']}`",
            f"- likely file: `{test_file}`",
            f"- focused verification: `{focused_command}`",
            f"- expected behavior: {cluster['expected_behavior']}",
            f"- rollback: {cluster['rollback_note']}",
            f"- exact test contract sha256: {exact_test_contract['failure_regression_test_contract_sha256']}",
            "",
            "Evidence examples:",
        ]
        lines.extend(f"- #{example['id']} {example['body'][:160]}" for example in examples)
        lines.extend(["", "Draft smoke-test assertions:"])
        lines.extend(f"- {assertion}" for assertion in cluster["assertions"])
        lines.extend(
            [
                "",
                "Promotion checklist:",
                "- Confirm the examples describe the same failure mode.",
                "- Add the smallest test that fails on the old behavior and passes on the intended behavior.",
                "- Keep the test local and deterministic.",
                "- Run the focused smoke test and a compile pass.",
                "",
                "Optional learning artifact:",
                f"- {cluster['preference_candidate']}",
                "",
                "Safe next commands:",
                f"- `failure to test: {examples[0]['body'][:180]}`",
                "- `failure clusters`",
                "- `learning review`",
                "",
                "Boundary:",
                "- Read-only packet only. It does not write files, change memory, queue approvals, execute tools, speak, or control the computer.",
            ]
        )
        return ToolResult(
            "failure_promotion_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                count=len(rows),
                readable_feedback_rows=len(rows),
                unreadable_feedback_rows=unreadable_feedback_rows,
                clusters=len(ranked),
                repeated_clusters=len(strong),
                selected_surface=cluster["surface"],
                target_test=cluster["test_name"],
                target_file=test_file,
                focused_command=focused_command,
                expected_behavior=cluster["expected_behavior"],
                observed_failures=[example["body"][:220] for example in examples],
                rollback_note=cluster["rollback_note"],
                **exact_test_contract,
                selector=selector,
                promoted=False,
                limit=limit,
            ),
        )

    def failure_implementation_packet(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 24)
        selector = str(args.get("cluster") or args.get("surface") or args.get("test") or "").strip().lower()
        rows, unreadable_feedback_rows = recent_feedback(limit)
        ranked, strong = build_failure_clusters(rows)
        candidates = strong or ranked
        if selector:
            candidates = [
                cluster
                for cluster in candidates
                if selector in cluster["surface"].lower() or selector in cluster["test_name"].lower() or selector in cluster["theme"].lower()
            ]
        if not candidates:
            return ToolResult(
                "failure_implementation_packet",
                False,
                "No failure cluster is implementation-ready. Capture repeated feedback first, then run `failure clusters`.",
                _safe_metadata(
                    count=len(rows),
                    readable_feedback_rows=len(rows),
                    unreadable_feedback_rows=unreadable_feedback_rows,
                    clusters=len(ranked),
                    repeated_clusters=len(strong),
                    selector=selector,
                    implementation_ready=False,
                    limit=limit,
                ),
            )

        cluster = candidates[0]
        examples = cluster["rows"][:4]
        test_file = _target_test_file(cluster["test_name"])
        focused_command = _focused_command_for_test(cluster["test_name"])
        exact_test_contract = _failure_regression_test_contract_metadata(cluster, focused_command=focused_command)

        implementation_steps = [
            "Add or tighten the smallest deterministic smoke assertion for the selected failure.",
            "Change only the UI/tool/planner surface needed for that assertion.",
            "Run the focused smoke test first, then compileall.",
            "Keep risky execution, personal data, shell/code, and computer control approval-gated.",
        ]
        stop_conditions = [
            "Stop if the examples are not the same failure mode.",
            "Stop if the fix needs private data, external side effects, or broad refactors.",
            "Stop if the expected behavior cannot be verified locally.",
            "Stop if the patch would weaken approval gates or hide diagnostics.",
        ]
        lines = [
            "Jarvis failure implementation packet:",
            "This is an implementation-ready, read-only work packet. It does not edit code or mark the failure fixed.",
            "",
            "Selected failure cluster:",
            f"- surface: {cluster['surface']}",
            f"- theme: {cluster['theme']}",
            f"- examples: {len(cluster['rows'])}",
            f"- target smoke test: `{cluster['test_name']}`",
            f"- target file: `{test_file}`",
            f"- focused verification: `{focused_command}`",
            f"- exact test contract sha256: {exact_test_contract['failure_regression_test_contract_sha256']}",
            "",
            "Example evidence:",
        ]
        lines.extend(f"- #{example['id']} {example['body'][:160]}" for example in examples)
        lines.extend(["", "Required assertions:"])
        lines.extend(f"- {assertion}" for assertion in cluster["assertions"])
        lines.extend(["", "Implementation steps:"])
        lines.extend(f"- {step}" for step in implementation_steps)
        lines.extend(["", "Stop conditions:"])
        lines.extend(f"- {condition}" for condition in stop_conditions)
        lines.extend(
            [
                "",
                "Verification sequence:",
                f"- `{focused_command}`",
                "- `python3 -m compileall -q jarvis_v2`",
                "- If dashboard/API code changed, rerun the status-server smoke test.",
                "",
                "Boundary:",
                "- Read-only packet only. It does not write tests, edit code, change memory, queue approvals, execute tools, speak, control the computer, or claim the failure is fixed.",
            ]
        )
        return ToolResult(
            "failure_implementation_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                count=len(rows),
                readable_feedback_rows=len(rows),
                unreadable_feedback_rows=unreadable_feedback_rows,
                clusters=len(ranked),
                repeated_clusters=len(strong),
                selected_surface=cluster["surface"],
                target_test=cluster["test_name"],
                target_file=test_file,
                focused_command=focused_command,
                expected_behavior=cluster["expected_behavior"],
                observed_failures=[example["body"][:220] for example in examples],
                rollback_note=cluster["rollback_note"],
                **exact_test_contract,
                assertion_count=len(cluster["assertions"]),
                stop_conditions=len(stop_conditions),
                selector=selector,
                implementation_ready=bool(strong),
                limit=limit,
            ),
        )

    def failure_apply_contract(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 24)
        selector = str(args.get("cluster") or args.get("surface") or args.get("test") or "").strip().lower()
        rows, unreadable_feedback_rows = recent_feedback(limit)
        ranked, strong = build_failure_clusters(rows)
        candidates = strong or ranked
        if selector:
            candidates = [
                cluster
                for cluster in candidates
                if selector in cluster["surface"].lower() or selector in cluster["test_name"].lower() or selector in cluster["theme"].lower()
            ]
        if not candidates:
            return ToolResult(
                "failure_apply_contract",
                False,
                "No failure cluster is ready for an apply contract. Capture repeated feedback first, then run `failure implementation packet`.",
                _safe_metadata(
                    count=len(rows),
                    readable_feedback_rows=len(rows),
                    unreadable_feedback_rows=unreadable_feedback_rows,
                    clusters=len(ranked),
                    repeated_clusters=len(strong),
                    selector=selector,
                    apply_ready=False,
                    limit=limit,
                ),
            )

        cluster = candidates[0]
        examples = cluster["rows"][:4]
        test_file = _target_test_file(cluster["test_name"])
        focused_command = _focused_command_for_test(cluster["test_name"])
        exact_test_contract = _failure_regression_test_contract_metadata(cluster, focused_command=focused_command)

        contract_steps = [
            "Start from `failure implementation packet` and confirm the selected examples describe one failure mode.",
            f"Patch `{test_file}` with the smallest deterministic assertion first.",
            "Capture a pre-patch failing-test receipt showing the new assertion fails before implementation changes.",
            "Patch only the implementation file(s) required to satisfy that assertion.",
            f"Run `{focused_command}` and capture the result.",
            "Run `python3 -m compileall -q jarvis_v2` before claiming the failure is fixed.",
        ]
        approval_rules = [
            "This contract is read-only and does not apply the patch itself.",
            "If the patch requires shell/code execution beyond local verification, computer control, personal data, external side effects, or destructive changes, stop for explicit approval.",
            "Do not weaken approval gates to make the regression pass.",
            "Do not mark the failure resolved until focused verification and compileall pass.",
        ]
        rollback_plan = [
            "Keep the patch scoped to the target test and the smallest implementation surface.",
            "If verification fails, preserve the failing assertion and adjust only the behavior that caused the original miss.",
            "If the assertion was wrong, revert that assertion and return to the implementation packet for a narrower contract.",
        ]
        apply_contract_sha256 = _apply_contract_sha256(
            surface=cluster["surface"],
            target_test=cluster["test_name"],
            target_file=test_file,
            focused_command=focused_command,
        )
        lines = [
            "Jarvis failure apply contract:",
            "This is the last-look contract before turning a repeated failure cluster into a patch. It stays read-only and approval-aware.",
            "",
            "Selected cluster:",
            f"- surface: {cluster['surface']}",
            f"- theme: {cluster['theme']}",
            f"- examples: {len(cluster['rows'])}",
            f"- target smoke test: `{cluster['test_name']}`",
            f"- target file: `{test_file}`",
            f"- focused verification: `{focused_command}`",
            f"- apply contract sha256: {apply_contract_sha256}",
            f"- exact test contract sha256: {exact_test_contract['failure_regression_test_contract_sha256']}",
            "",
            "Example evidence:",
        ]
        lines.extend(f"- #{example['id']} {example['body'][:160]}" for example in examples)
        lines.extend(["", "Required assertions:"])
        lines.extend(f"- {assertion}" for assertion in cluster["assertions"])
        lines.extend(["", "Apply sequence:"])
        lines.extend(f"- {step}" for step in contract_steps)
        lines.extend(["", "Approval and safety rules:"])
        lines.extend(f"- {rule}" for rule in approval_rules)
        lines.extend(["", "Rollback plan:"])
        lines.extend(f"- {step}" for step in rollback_plan)
        lines.extend(
            [
                "",
                "Completion receipt must include:",
                "- test-first receipt: the new assertion failed before implementation changes",
                f"- focused verification: `{focused_command}`",
                "- compile pass: `python3 -m compileall -q jarvis_v2`",
                "- changed files and the exact failure cluster addressed",
                "",
                "Boundary:",
                "- Read-only contract only. It does not write tests, edit code, change memory, queue approvals, execute tools, speak, control the computer, or claim the patch is applied.",
            ]
        )
        return ToolResult(
            "failure_apply_contract",
            True,
            "\n".join(lines),
            _safe_metadata(
                count=len(rows),
                readable_feedback_rows=len(rows),
                unreadable_feedback_rows=unreadable_feedback_rows,
                clusters=len(ranked),
                repeated_clusters=len(strong),
                selected_surface=cluster["surface"],
                target_test=cluster["test_name"],
                target_file=test_file,
                focused_command=focused_command,
                apply_contract_sha256=apply_contract_sha256,
                apply_contract_hash_present=True,
                expected_behavior=cluster["expected_behavior"],
                observed_failures=[example["body"][:220] for example in examples],
                rollback_note=cluster["rollback_note"],
                **exact_test_contract,
                assertion_count=len(cluster["assertions"]),
                apply_steps=len(contract_steps),
                approval_rules=len(approval_rules),
                rollback_steps=len(rollback_plan),
                test_first_receipt_required=True,
                pre_patch_failing_test_receipt_required=True,
                selector=selector,
                apply_ready=bool(strong),
                limit=limit,
            ),
        )

    def failure_learning_cockpit(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 24)
        selector = str(args.get("cluster") or args.get("surface") or args.get("test") or "").strip().lower()
        clusters = repeated_failure_clusters({"limit": limit})
        promotion = failure_promotion_packet({"cluster": selector, "limit": limit})
        implementation = failure_implementation_packet({"cluster": selector, "limit": limit})
        apply_contract = failure_apply_contract({"cluster": selector, "limit": limit})
        blockers: list[str] = []
        if not promotion.ok:
            blockers.append("promotion packet unavailable")
        if not implementation.ok:
            blockers.append("implementation packet unavailable")
        if not apply_contract.ok:
            blockers.append("apply contract unavailable")
        if int(clusters.metadata.get("repeated_clusters") or 0) < 1:
            blockers.append("no repeated failure cluster")
        if implementation.metadata.get("implementation_ready") is not True:
            blockers.append("implementation packet not ready")
        if apply_contract.metadata.get("apply_ready") is not True:
            blockers.append("apply contract not ready")

        target_test = str(apply_contract.metadata.get("target_test") or implementation.metadata.get("target_test") or promotion.metadata.get("target_test") or "")
        target_file = str(apply_contract.metadata.get("target_file") or implementation.metadata.get("target_file") or promotion.metadata.get("target_file") or "")
        focused_command = str(apply_contract.metadata.get("focused_command") or implementation.metadata.get("focused_command") or "")
        apply_contract_sha256 = str(apply_contract.metadata.get("apply_contract_sha256") or "")
        exact_test_contract_sha256 = str(apply_contract.metadata.get("failure_regression_test_contract_sha256") or "")
        state = "FAILURE_LEARNING_READY_FOR_PATCH_REVIEW" if not blockers else "FAILURE_LEARNING_HELD"
        proof_queue = [
            "failure clusters",
            f"failure promotion packet: {selector or target_test or 'selected cluster'}",
            f"failure implementation packet: {selector or target_test or 'selected cluster'}",
            f"failure apply contract: {selector or target_test or 'selected cluster'}",
        ]
        if focused_command:
            proof_queue.append(focused_command)
        proof_queue.append("python3 -m compileall -q jarvis_v2")
        next_safe_command = "review the apply contract, patch the scoped test first, then run focused verification" if state.endswith("PATCH_REVIEW") else proof_queue[0]
        lines = [
            "Jarvis failure learning cockpit:",
            "This is read-only. It consolidates repeated failure clusters, promotion packet, implementation packet, apply contract, blockers, proof queue, target file, verification, and rollback before any patch is attempted.",
            "",
            "Selector:",
            f"- {selector or 'strongest repeated cluster'}",
            "",
            "Cockpit state:",
            f"- state: {state}",
            f"- ready for patch review: {'yes' if state.endswith('PATCH_REVIEW') else 'no'}",
            f"- blockers: {', '.join(blockers) if blockers else 'none'}",
            f"- next safe command: `{next_safe_command}`",
            "",
            "Selected target:",
            f"- target smoke test: `{target_test or 'unknown'}`",
            f"- target file: `{target_file or 'unknown'}`",
            f"- focused verification: `{focused_command or 'unknown'}`",
            f"- apply contract sha256: {apply_contract_sha256 or 'missing'}",
            f"- exact test contract sha256: {exact_test_contract_sha256 or 'missing'}",
            "",
            "Failure proof queue:",
            f"- next required command: `{proof_queue[0]}`",
            f"- proof queue count: {len(proof_queue)}",
        ]
        lines.extend(f"- `{command}`" for command in proof_queue[:8])
        lines.extend(
            [
                "",
                "Required patch receipt:",
                "- selected repeated failure cluster",
                "- target smoke test and target file",
                "- smallest deterministic assertion",
                "- focused verification result",
                "- compile pass",
                "- rollback note",
                "",
                "Boundary:",
                "- read-only learning cockpit; no tests are written, no code is edited, no memory is changed, no approvals are queued, no shell/code is executed, and no completion claim is made.",
            ]
        )
        return ToolResult(
            "failure_learning_cockpit",
            True,
            "\n".join(lines),
            _safe_metadata(
                limit=limit,
                selector=selector,
                cockpit_state=state,
                ready_for_patch_review=state.endswith("PATCH_REVIEW"),
                blockers=blockers,
                blocker_count=len(blockers),
                next_safe_command=next_safe_command,
                clusters=clusters.metadata.get("clusters", 0),
                repeated_clusters=clusters.metadata.get("repeated_clusters", 0),
                selected_surface=apply_contract.metadata.get("selected_surface") or implementation.metadata.get("selected_surface") or promotion.metadata.get("selected_surface", ""),
                target_test=target_test,
                target_file=target_file,
                focused_command=focused_command,
                apply_contract_sha256=apply_contract_sha256,
                apply_contract_hash_present=_looks_like_sha256(apply_contract_sha256),
                failure_regression_test_contract_sha256=exact_test_contract_sha256,
                failure_regression_test_contract_present=_looks_like_sha256(exact_test_contract_sha256),
                failure_regression_test_contract_ready=apply_contract.metadata.get("failure_regression_test_contract_ready", False),
                exact_test_contract_ready=apply_contract.metadata.get("exact_test_contract_ready", False),
                exact_test_contract_requires_pre_patch_failure=apply_contract.metadata.get("exact_test_contract_requires_pre_patch_failure", True),
                exact_test_contract_authorizes_file_write=False,
                exact_test_contract_authorizes_patch_application=False,
                exact_test_contract_authorizes_tool_execution=False,
                exact_test_contract_authorizes_completion_claim=False,
                exact_test_contract_reusable_for_next_patch=False,
                apply_ready=apply_contract.metadata.get("apply_ready", False),
                implementation_ready=implementation.metadata.get("implementation_ready", False),
                failure_proof_queue=proof_queue,
                failure_proof_queue_count=len(proof_queue),
                failure_next_proof_command=proof_queue[0],
                failure_next_required_command=proof_queue[0],
                next_required_command=proof_queue[0],
                promotion_ok=promotion.ok,
                implementation_ok=implementation.ok,
                apply_contract_ok=apply_contract.ok,
                promotion_output=promotion.output[:1200],
                implementation_output=implementation.output[:1200],
                apply_contract_output=apply_contract.output[:1200],
                clusters_output=clusters.output[:1200],
            ),
        )

    def failure_patch_receipt_packet(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 24)
        summary = str(args.get("summary") or args.get("receipt") or args.get("body") or "").strip()
        selector = str(args.get("cluster") or args.get("surface") or args.get("test") or "").strip().lower()
        if not selector and summary:
            selector = _receipt_field(summary, "cluster").lower()
        if not selector and summary:
            selector = summary.split(";", 1)[0].strip().lower()

        rows, unreadable_feedback_rows = recent_feedback(limit)
        ranked, strong = build_failure_clusters(rows)
        candidates = strong or ranked
        if selector:
            candidates = [
                cluster
                for cluster in candidates
                if selector in cluster["surface"].lower() or selector in cluster["test_name"].lower() or selector in cluster["theme"].lower()
            ]
        if not candidates:
            return ToolResult(
                "failure_patch_receipt_packet",
                False,
                "No failure cluster is ready for a patch receipt. Run `failure learning cockpit` after capturing repeated feedback.",
                _safe_metadata(
                    count=len(rows),
                    readable_feedback_rows=len(rows),
                    unreadable_feedback_rows=unreadable_feedback_rows,
                    clusters=len(ranked),
                    repeated_clusters=len(strong),
                    selector=selector,
                    receipt_ready=False,
                    limit=limit,
                ),
            )

        cluster = candidates[0]
        test_file = _target_test_file(cluster["test_name"])
        focused_command = _focused_command_for_test(cluster["test_name"])
        exact_test_contract = _failure_regression_test_contract_metadata(cluster, focused_command=focused_command)

        changed_files_raw = _short(args.get("changed_files") or args.get("files") or _receipt_field(summary, "changed files") or _receipt_field(summary, "changed file"), 500)
        test_first_receipt_raw = _short(
            args.get("test_first")
            or args.get("pre_patch_test")
            or args.get("failing_test")
            or _receipt_field(summary, "test first")
            or _receipt_field(summary, "pre-patch test")
            or _receipt_field(summary, "pre patch test")
            or _receipt_field(summary, "failing test"),
            500,
        )
        verification_raw = _short(args.get("verification") or args.get("focused_verification") or _receipt_field(summary, "focused verification") or _receipt_field(summary, "verification"), 500)
        compile_receipt_raw = _short(args.get("compile") or args.get("compile_receipt") or _receipt_field(summary, "compile"), 500)
        rollback_raw = _short(args.get("rollback") or _receipt_field(summary, "rollback"), 500)
        patch_receipt_sha256 = _short(
            args.get("patch_receipt_sha256")
            or args.get("receipt_sha256")
            or args.get("sha256")
            or _receipt_field(summary, "patch receipt sha256")
            or _receipt_field(summary, "receipt sha256")
            or _receipt_field(summary, "sha256"),
            80,
        ).lower()
        notes_raw = _short(args.get("notes") or _receipt_field(summary, "notes") or summary, 500)

        changed_files = _safe_display(changed_files_raw, 500)
        test_first_receipt = _safe_display(test_first_receipt_raw, 500)
        verification = _safe_display(verification_raw, 500)
        compile_receipt = _safe_display(compile_receipt_raw, 500)
        rollback = _safe_display(rollback_raw, 500)
        notes = _safe_display(notes_raw, 500)

        changed_file_items = [item.strip(" `") for item in re.split(r"[,;\n]", changed_files_raw) if item.strip()]
        has_target_file = not changed_file_items or test_file in changed_files_raw or any(item.endswith(test_file) for item in changed_file_items)
        test_first_present = bool(test_first_receipt_raw) and any(token in test_first_receipt_raw.lower() for token in ("reviewed", "failed", "failing", "red", "pre-patch", "pre patch", "assertion"))
        verification_matches = bool(verification_raw) and (focused_command in verification_raw or cluster["test_name"] in verification_raw or "passed" in verification_raw.lower())
        compile_matches = bool(compile_receipt_raw) and any(token in compile_receipt_raw.lower() for token in ("compileall", "py_compile", "compiled", "passed"))
        rollback_present = bool(rollback_raw)
        patch_receipt_hash_present = _looks_like_sha256(patch_receipt_sha256)
        changed_files_present = bool(changed_files_raw)
        receipt_ready = changed_files_present and test_first_present and verification_matches and compile_matches and rollback_present and patch_receipt_hash_present and has_target_file

        missing: list[str] = []
        if not changed_files_present:
            missing.append("changed_files")
        if not has_target_file:
            missing.append("target_test_file_in_changed_files")
        if not test_first_present:
            missing.append("pre_patch_failing_test_receipt")
        if not verification_matches:
            missing.append("focused_verification_pass")
        if not compile_matches:
            missing.append("compile_pass")
        if not rollback_present:
            missing.append("rollback_note")
        if not patch_receipt_hash_present:
            missing.append("patch_receipt_sha256")

        proof_queue = [
            f"failure learning cockpit: {selector or cluster['test_name']}",
            f"failure apply contract: {selector or cluster['test_name']}",
            focused_command,
            "python3 -m compileall -q jarvis_v2",
            f"completion audit: patch repeated failure {cluster['test_name']}",
            "completion claim gate",
        ]
        apply_contract_sha256 = _apply_contract_sha256(
            surface=str(cluster["surface"]),
            target_test=str(cluster["test_name"]),
            target_file=test_file,
            focused_command=focused_command,
        )
        review_contract_rows = _failure_patch_review_contract_rows(
            stage="patch_review",
            target_test=str(cluster["test_name"]),
            target_file=test_file,
        )
        review_contract_summary = [
            "patch-application-not-authorized",
            "file-write-not-authorized",
            "tool-execution-not-authorized",
            "completion-claim-not-authorized",
            "learning-record-not-authorized",
            "fresh-patch-review-required",
        ]
        review_token_metadata = _failure_patch_review_token_metadata(
            target_test=str(cluster["test_name"]),
            target_file=test_file,
            apply_contract_sha256=apply_contract_sha256,
            failure_regression_test_contract_sha256=str(exact_test_contract["failure_regression_test_contract_sha256"]),
            patch_receipt_sha256=patch_receipt_sha256,
            review_contract_rows=review_contract_rows,
        )
        lines = [
            "Jarvis failure patch receipt packet:",
            "This is the post-patch learning receipt. It is read-only and checks whether a repeated-failure patch has the minimum evidence before Jarvis can claim the failure was fixed.",
            "",
            f"Verdict: {'PATCH_RECEIPT_READY_FOR_COMPLETION_REVIEW' if receipt_ready else 'PATCH_RECEIPT_INCOMPLETE'}",
            "",
            "Selected cluster:",
            f"- surface: {cluster['surface']}",
            f"- theme: {cluster['theme']}",
            f"- examples: {len(cluster['rows'])}",
            f"- target smoke test: `{cluster['test_name']}`",
            f"- target file: `{test_file}`",
            f"- focused verification: `{focused_command}`",
            f"- exact test contract sha256: {exact_test_contract['failure_regression_test_contract_sha256']}",
            "",
            "Receipt evidence:",
            f"- changed files: {changed_files or 'missing'}",
            f"- test-first receipt: {test_first_receipt or 'missing'}",
            f"- focused verification: {verification or 'missing'}",
            f"- compile pass: {compile_receipt or 'missing'}",
            f"- rollback note: {rollback or 'missing'}",
            f"- patch receipt sha256: {patch_receipt_sha256 or 'missing'}",
            f"- notes: {notes or 'none'}",
            "",
            "Receipt checklist:",
            f"- changed files present: {'yes' if changed_files_present else 'no'}",
            f"- target test file included or intentionally omitted: {'yes' if has_target_file else 'no'}",
            f"- pre-patch failing test receipt present: {'yes' if test_first_present else 'no'}",
            f"- focused verification passed: {'yes' if verification_matches else 'no'}",
            f"- compile pass present: {'yes' if compile_matches else 'no'}",
            f"- rollback note present: {'yes' if rollback_present else 'no'}",
            f"- patch receipt hash present: {'yes' if patch_receipt_hash_present else 'no'}",
            f"- missing: {', '.join(missing) if missing else 'none'}",
            "",
            "Proof queue:",
        ]
        lines.extend(f"- `{command}`" for command in proof_queue)
        lines.extend(
            [
                "",
                "Non-authorizing patch review token:",
                f"- token sha256: {review_token_metadata['failure_patch_review_token_sha256']}",
                "- authorizes patch application: no",
                "- authorizes file writes: no",
                "- authorizes tool execution: no",
                "- authorizes learning records: no",
                "",
                "Boundary:",
                "- Read-only receipt packet only. It does not write tests, edit code, change memory, queue approvals, execute tools, speak, control the computer, or mark the patch complete.",
            ]
        )
        return ToolResult(
            "failure_patch_receipt_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                count=len(rows),
                readable_feedback_rows=len(rows),
                unreadable_feedback_rows=unreadable_feedback_rows,
                clusters=len(ranked),
                repeated_clusters=len(strong),
                selector=selector,
                selected_surface=cluster["surface"],
                target_test=cluster["test_name"],
                target_file=test_file,
                focused_command=focused_command,
                expected_behavior=cluster["expected_behavior"],
                observed_failures=[row["body"][:220] for row in cluster["rows"][:4]],
                rollback_note=cluster["rollback_note"],
                **exact_test_contract,
                changed_files=changed_files,
                changed_file_count=len(changed_file_items),
                test_first_receipt=test_first_receipt,
                test_first_receipt_required=True,
                pre_patch_failing_test_receipt_required=True,
                pre_patch_failing_test_receipt_present=test_first_present,
                test_first_receipt_present=test_first_present,
                verification=verification,
                compile_receipt=compile_receipt,
                rollback=rollback,
                patch_receipt_sha256=patch_receipt_sha256,
                patch_receipt_hash_present=patch_receipt_hash_present,
                changed_files_present=changed_files_present,
                target_file_in_changed_files=has_target_file,
                focused_verification_present=verification_matches,
                compile_pass_present=compile_matches,
                rollback_note_present=rollback_present,
                apply_contract_sha256=apply_contract_sha256,
                missing=missing,
                missing_count=len(missing),
                receipt_ready=receipt_ready,
                patch_receipt_ready=receipt_ready,
                completion_review_ready=receipt_ready,
                patch_review_contract_rows=review_contract_rows,
                patch_review_contract_row_count=len(review_contract_rows),
                patch_review_contract_ready=True,
                patch_review_contract_summary=review_contract_summary,
                review_authorizes_patch_application=False,
                review_authorizes_file_write=False,
                review_authorizes_tool_execution=False,
                review_authorizes_model_call=False,
                review_authorizes_personal_data_read=False,
                review_authorizes_external_side_effect=False,
                review_authorizes_approval=False,
                review_authorizes_completion_claim=False,
                review_authorizes_learning_record=False,
                review_reusable_for_next_patch=False,
                **review_token_metadata,
                proof_queue=proof_queue,
                proof_queue_count=len(proof_queue),
                next_proof_command=proof_queue[0],
                limit=limit,
            ),
        )

    def failure_patch_application_bridge(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 24)
        summary = str(args.get("summary") or args.get("receipt") or args.get("body") or "").strip()
        selector = str(args.get("cluster") or args.get("surface") or args.get("test") or "").strip().lower()
        if not selector and summary:
            selector = _receipt_field(summary, "cluster").lower()
        if not selector and summary:
            selector = summary.split(";", 1)[0].strip().lower()

        apply_contract = failure_apply_contract({"cluster": selector, "limit": limit})
        receipt = failure_patch_receipt_packet({"summary": summary, "cluster": selector, "limit": limit})
        apply_meta = apply_contract.metadata
        receipt_meta = receipt.metadata

        contract_evidence_raw = _short(
            args.get("contract")
            or args.get("apply_contract")
            or _contract_review_evidence(summary),
            500,
        )
        supplied_apply_contract_sha256 = _short(
            args.get("apply_contract_sha256")
            or args.get("contract_sha256")
            or args.get("contract_hash")
            or _receipt_field(summary, "apply contract sha256")
            or _receipt_field(summary, "contract sha256"),
            80,
        ).lower()
        application_evidence_raw = _short(
            args.get("application_bridge")
            or args.get("patch_application")
            or args.get("applied_patch")
            or args.get("apply_bridge")
            or _receipt_field(summary, "application bridge")
            or _receipt_field(summary, "patch application")
            or _receipt_field(summary, "applied patch")
            or _receipt_field(summary, "apply bridge"),
            500,
        )
        contract_evidence = _safe_display(contract_evidence_raw, 500)
        application_evidence = _safe_display(application_evidence_raw, 500)
        contract_reviewed = bool(contract_evidence_raw) and any(
            token in contract_evidence_raw.lower()
            for token in ("reviewed", "passed", "accepted", "confirmed", "ready")
        )
        application_bound = bool(application_evidence_raw) and any(
            token in application_evidence_raw.lower()
            for token in ("bound", "applied", "reviewed", "confirmed", "matches", "matched")
        )
        receipt_ready = receipt_meta.get("patch_receipt_ready") is True
        apply_ready = apply_meta.get("apply_ready") is True
        expected_apply_contract_sha256 = str(apply_meta.get("apply_contract_sha256") or "")
        exact_test_contract_sha256 = str(receipt_meta.get("failure_regression_test_contract_sha256") or apply_meta.get("failure_regression_test_contract_sha256") or "")
        exact_test_contract_present = _looks_like_sha256(exact_test_contract_sha256)
        exact_test_contract_matches_apply = bool(
            exact_test_contract_present
            and exact_test_contract_sha256 == str(apply_meta.get("failure_regression_test_contract_sha256") or "")
        )
        apply_contract_hash_present = _looks_like_sha256(supplied_apply_contract_sha256)
        apply_contract_hash_matches_expected = bool(
            apply_contract_hash_present
            and expected_apply_contract_sha256
            and supplied_apply_contract_sha256 == expected_apply_contract_sha256
        )
        target_file_bound = receipt_meta.get("target_file_in_changed_files") is True
        test_first_present = receipt_meta.get("pre_patch_failing_test_receipt_present") is True
        focused_present = receipt_meta.get("focused_verification_present") is True
        compile_present = receipt_meta.get("compile_pass_present") is True
        rollback_present = receipt_meta.get("rollback_note_present") is True
        patch_receipt_hash_present = receipt_meta.get("patch_receipt_hash_present") is True
        bridge_ready = bool(
            apply_contract.ok
            and receipt.ok
            and apply_ready
            and contract_reviewed
            and apply_contract_hash_matches_expected
            and application_bound
            and receipt_ready
            and target_file_bound
            and test_first_present
            and focused_present
            and compile_present
            and rollback_present
            and patch_receipt_hash_present
            and exact_test_contract_matches_apply
        )

        missing = list(receipt_meta.get("missing") or [])
        if not apply_ready:
            missing.append("failure_apply_contract_ready")
        if not contract_reviewed:
            missing.append("apply_contract_reviewed")
        if not apply_contract_hash_present:
            missing.append("apply_contract_sha256")
        elif not apply_contract_hash_matches_expected:
            missing.append("apply_contract_sha256_matches_reviewed_contract")
        if not application_bound:
            missing.append("applied_patch_bound_to_contract")
        if not receipt_ready:
            missing.append("patch_receipt_ready")
        if not patch_receipt_hash_present:
            missing.append("patch_receipt_sha256")
        if not exact_test_contract_present:
            missing.append("failure_regression_test_contract_sha256")
        elif not exact_test_contract_matches_apply:
            missing.append("failure_regression_test_contract_matches_apply_contract")
        missing = list(dict.fromkeys(item for item in missing if item))

        target_test = str(receipt_meta.get("target_test") or apply_meta.get("target_test") or "")
        target_file = str(receipt_meta.get("target_file") or apply_meta.get("target_file") or "")
        focused_command = str(receipt_meta.get("focused_command") or apply_meta.get("focused_command") or "")
        review_contract_rows = list(receipt_meta.get("patch_review_contract_rows") or []) or _failure_patch_review_contract_rows(
            stage="patch_review",
            target_test=target_test,
            target_file=target_file,
        )
        review_contract_summary = list(receipt_meta.get("patch_review_contract_summary") or []) or [
            "patch-application-not-authorized",
            "file-write-not-authorized",
            "tool-execution-not-authorized",
            "completion-claim-not-authorized",
            "learning-record-not-authorized",
            "fresh-patch-review-required",
        ]
        review_token_metadata = _failure_patch_review_token_metadata(
            target_test=target_test,
            target_file=target_file,
            apply_contract_sha256=expected_apply_contract_sha256,
            failure_regression_test_contract_sha256=exact_test_contract_sha256,
            patch_receipt_sha256=str(receipt_meta.get("patch_receipt_sha256") or ""),
            review_contract_rows=review_contract_rows,
        )
        proof_queue = _ordered_commands(
            [
                f"failure apply contract: {selector or target_test or 'selected cluster'}",
                f"failure patch receipt: {selector or target_test or 'selected cluster'}; changed files <files>; test first <pre-patch failing receipt>; verification <focused pass>; compile <compile pass>; rollback <rollback note>",
                f"failure patch application bridge: {selector or target_test or 'selected cluster'}; contract <apply contract reviewed>; application bridge <applied patch bound to contract>",
                focused_command,
                "python3 -m compileall -q jarvis_v2",
                f"failure patch completion gate: {selector or target_test or 'selected cluster'}",
            ]
        )
        state = "FAILURE_PATCH_APPLICATION_BRIDGE_READY" if bridge_ready else "FAILURE_PATCH_APPLICATION_BRIDGE_HELD"
        next_command = proof_queue[-1] if bridge_ready else proof_queue[0]
        lines = [
            "Jarvis failure patch application bridge:",
            "This is read-only. It binds an actually applied repeated-failure patch back to the reviewed apply contract, patch receipt, test-first proof, focused verification, compile pass, and rollback before any completion gate can count it.",
            "",
            "Bridge state:",
            f"- state: {state}",
            f"- ready for completion gate: {'yes' if bridge_ready else 'no'}",
            f"- missing: {', '.join(missing) if missing else 'none'}",
            f"- next command: `{next_command}`",
            "",
            "Patch target:",
            f"- target smoke test: `{target_test or 'unknown'}`",
            f"- target file: `{target_file or 'unknown'}`",
            f"- focused verification: `{focused_command or 'unknown'}`",
            "",
            "Bridge evidence:",
            f"- apply contract ready: {'yes' if apply_ready else 'no'}",
            f"- apply contract reviewed: {'yes' if contract_reviewed else 'no'}",
            f"- reviewed apply contract sha256: {expected_apply_contract_sha256 or 'missing'}",
            f"- exact test contract sha256: {exact_test_contract_sha256 or 'missing'}",
            f"- exact test contract matches apply contract: {'yes' if exact_test_contract_matches_apply else 'no'}",
            f"- supplied apply contract sha256: {supplied_apply_contract_sha256 or 'missing'}",
            f"- apply contract sha256 matches reviewed contract: {'yes' if apply_contract_hash_matches_expected else 'no'}",
            f"- applied patch bound to contract: {'yes' if application_bound else 'no'}",
            f"- patch receipt ready: {'yes' if receipt_ready else 'no'}",
            f"- target file bound to changed files: {'yes' if target_file_bound else 'no'}",
            f"- pre-patch failing test receipt present: {'yes' if test_first_present else 'no'}",
            f"- focused verification present: {'yes' if focused_present else 'no'}",
            f"- compile pass present: {'yes' if compile_present else 'no'}",
            f"- rollback note present: {'yes' if rollback_present else 'no'}",
            f"- patch receipt hash present: {'yes' if patch_receipt_hash_present else 'no'}",
            "",
            "Non-authorizing review contract:",
            f"- row count: {len(review_contract_rows)}",
            f"- summary: {', '.join(review_contract_summary)}",
            f"- patch review token sha256: {review_token_metadata['failure_patch_review_token_sha256']}",
        ]
        lines.extend(
            f"- {row['item']}: {row['status']}; {row['evidence']}" for row in review_contract_rows
        )
        lines.extend(
            [
                "",
            "Proof queue:",
            f"- next required command: `{next_command}`",
            f"- proof queue count: {len(proof_queue)}",
            ]
        )
        lines.extend(f"- `{command}`" for command in proof_queue)
        lines.extend(
            [
                "",
                "Boundary:",
                "- Read-only application bridge only. It does not write tests, edit code, run verification, change memory, queue approvals, execute tools, speak, control the computer, complete tasks, or claim the patch fixed.",
            ]
        )
        return ToolResult(
            "failure_patch_application_bridge",
            receipt.ok and apply_contract.ok,
            "\n".join(lines),
            _safe_metadata(
                limit=limit,
                selector=selector,
                bridge_state=state,
                application_bridge_ready=bridge_ready,
                ready_for_completion_gate=bridge_ready,
                target_test=target_test,
                target_file=target_file,
                focused_command=focused_command,
                contract_evidence=contract_evidence,
                application_evidence=application_evidence,
                apply_contract_ready=apply_ready,
                apply_contract_reviewed=contract_reviewed,
                apply_contract_sha256=expected_apply_contract_sha256,
                failure_regression_test_contract_sha256=exact_test_contract_sha256,
                failure_regression_test_contract_present=exact_test_contract_present,
                failure_regression_test_contract_ready=exact_test_contract_matches_apply,
                exact_test_contract_ready=exact_test_contract_matches_apply,
                exact_test_contract_matches_apply_contract=exact_test_contract_matches_apply,
                exact_test_contract_requires_pre_patch_failure=True,
                exact_test_contract_authorizes_file_write=False,
                exact_test_contract_authorizes_patch_application=False,
                exact_test_contract_authorizes_tool_execution=False,
                exact_test_contract_authorizes_completion_claim=False,
                exact_test_contract_reusable_for_next_patch=False,
                supplied_apply_contract_sha256=supplied_apply_contract_sha256,
                apply_contract_hash_present=apply_contract_hash_present,
                apply_contract_hash_matches_expected=apply_contract_hash_matches_expected,
                applied_patch_bound_to_contract=application_bound,
                patch_receipt_ready=receipt_ready,
                target_file_in_changed_files=target_file_bound,
                pre_patch_failing_test_receipt_present=test_first_present,
                test_first_receipt_present=test_first_present,
                focused_verification_present=focused_present,
                compile_pass_present=compile_present,
                rollback_note_present=rollback_present,
                patch_receipt_sha256=receipt_meta.get("patch_receipt_sha256", ""),
                patch_receipt_hash_present=patch_receipt_hash_present,
                patch_review_contract_rows=review_contract_rows,
                patch_review_contract_row_count=len(review_contract_rows),
                patch_review_contract_ready=True,
                patch_review_contract_summary=review_contract_summary,
                review_authorizes_patch_application=False,
                review_authorizes_file_write=False,
                review_authorizes_tool_execution=False,
                review_authorizes_model_call=False,
                review_authorizes_personal_data_read=False,
                review_authorizes_external_side_effect=False,
                review_authorizes_approval=False,
                review_authorizes_completion_claim=False,
                review_authorizes_learning_record=False,
                review_reusable_for_next_patch=False,
                **review_token_metadata,
                missing=missing,
                missing_count=len(missing),
                proof_queue=proof_queue,
                proof_queue_count=len(proof_queue),
                next_proof_command=proof_queue[0],
                next_required_command=next_command,
                next_command=next_command,
                apply_contract_metadata=apply_meta,
                patch_receipt_metadata=receipt_meta,
                apply_contract_output=apply_contract.output[:1200],
                patch_receipt_output=receipt.output[:1200],
            ),
        )

    def failure_patch_completion_gate(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 24)
        summary = str(args.get("summary") or args.get("receipt") or args.get("body") or "").strip()
        selector = str(args.get("cluster") or args.get("surface") or args.get("test") or "").strip().lower()
        if not selector and summary:
            selector = _receipt_field(summary, "cluster").lower()
        if not selector and summary:
            selector = summary.split(";", 1)[0].strip().lower()

        cockpit = failure_learning_cockpit({"cluster": selector, "limit": limit})
        receipt = failure_patch_receipt_packet({"summary": summary, "cluster": selector, "limit": limit})
        application_bridge = failure_patch_application_bridge({"summary": summary, "cluster": selector, "limit": limit})
        cockpit_meta = cockpit.metadata
        receipt_meta = receipt.metadata
        bridge_meta = application_bridge.metadata

        contract_evidence_raw = _short(
            args.get("contract")
            or args.get("apply_contract")
            or _contract_review_evidence(summary),
            500,
        )
        contract_evidence = _safe_display(contract_evidence_raw, 500)
        contract_reviewed = bool(contract_evidence_raw) and any(
            token in contract_evidence_raw.lower()
            for token in ("reviewed", "passed", "accepted", "confirmed", "ready")
        )
        cockpit_ready = cockpit_meta.get("ready_for_patch_review") is True
        receipt_ready = receipt_meta.get("patch_receipt_ready") is True
        focused_matches = _feedback_metadata_bool(receipt_meta.get("focused_verification_present"))
        compile_passed = _feedback_metadata_bool(receipt_meta.get("compile_pass_present"))
        rollback_present = _feedback_metadata_bool(receipt_meta.get("rollback_note_present"))
        target_file_bound = _feedback_metadata_bool(receipt_meta.get("target_file_in_changed_files"))
        test_first_present = _feedback_metadata_bool(receipt_meta.get("pre_patch_failing_test_receipt_present"))
        application_bridge_ready = _feedback_metadata_bool(bridge_meta.get("application_bridge_ready"))
        applied_patch_bound = _feedback_metadata_bool(bridge_meta.get("applied_patch_bound_to_contract"))
        apply_contract_hash_matches = _feedback_metadata_bool(bridge_meta.get("apply_contract_hash_matches_expected"))
        exact_test_contract_ready = _feedback_metadata_bool(bridge_meta.get("exact_test_contract_ready"))
        patch_receipt_hash_present = _feedback_metadata_bool(bridge_meta.get("patch_receipt_hash_present"))
        review_contract_rows = list(bridge_meta.get("patch_review_contract_rows") or [])
        review_contract_ready = bridge_meta.get("patch_review_contract_ready") is True
        review_contract_summary = list(bridge_meta.get("patch_review_contract_summary") or [])
        failure_patch_review_token_sha256 = str(bridge_meta.get("failure_patch_review_token_sha256") or "")
        failure_patch_review_token_present = bridge_meta.get("failure_patch_review_token_present") is True

        missing = list(receipt_meta.get("missing") or [])
        if not cockpit_ready:
            missing.append("failure_learning_cockpit_ready")
        if not contract_reviewed:
            missing.append("apply_contract_reviewed")
        if not application_bridge_ready:
            missing.append("failure_patch_application_bridge_ready")
        if not applied_patch_bound:
            missing.append("applied_patch_bound_to_contract")
        if not apply_contract_hash_matches:
            missing.append("apply_contract_sha256_matches_reviewed_contract")
        if not exact_test_contract_ready:
            missing.append("failure_regression_test_contract_ready")
        if not receipt_ready:
            missing.append("patch_receipt_ready")
        if not patch_receipt_hash_present:
            missing.append("patch_receipt_sha256")
        if not review_contract_ready:
            missing.append("non_authorizing_patch_review_contract")
        if not failure_patch_review_token_present:
            missing.append("failure_patch_review_token_sha256")
        missing = list(dict.fromkeys(missing))

        completion_ready = bool(
            cockpit_ready
            and contract_reviewed
            and receipt_ready
            and focused_matches
            and compile_passed
            and rollback_present
            and target_file_bound
            and test_first_present
            and application_bridge_ready
            and applied_patch_bound
            and apply_contract_hash_matches
            and exact_test_contract_ready
            and patch_receipt_hash_present
            and review_contract_ready
            and failure_patch_review_token_present
        )
        gate_state = "FAILURE_PATCH_COMPLETION_READY_FOR_CLAIM_REVIEW" if completion_ready else "FAILURE_PATCH_COMPLETION_HELD"
        target_test = str(receipt_meta.get("target_test") or cockpit_meta.get("target_test") or "")
        target_file = str(receipt_meta.get("target_file") or cockpit_meta.get("target_file") or "")
        focused_command = str(receipt_meta.get("focused_command") or cockpit_meta.get("focused_command") or "")
        proof_queue = [
            f"failure learning cockpit: {selector or target_test or 'selected cluster'}",
            f"failure apply contract: {selector or target_test or 'selected cluster'}",
            f"failure patch receipt: {selector or target_test or 'selected cluster'}; changed files <files>; verification <focused pass>; compile <compile pass>; rollback <rollback note>; contract <apply contract reviewed>",
            f"failure patch application bridge: {selector or target_test or 'selected cluster'}; contract <apply contract reviewed>; application bridge <applied patch bound to contract>",
        ]
        if focused_command:
            proof_queue.append(focused_command)
        proof_queue.extend(
            [
                "python3 -m compileall -q jarvis_v2",
                f"completion audit: patch repeated failure {target_test or 'selected cluster'}",
                "evidence ledger",
                f"completion claim gate: patch repeated failure {target_test or 'selected cluster'}",
            ]
        )
        next_command = proof_queue[-1] if completion_ready else proof_queue[0]

        lines = [
            "Jarvis failure patch completion gate:",
            "This is read-only. It checks the learning cockpit, apply contract evidence, patch receipt, focused verification, compile pass, rollback note, and completion-review handoff before Jarvis claims a repeated failure is fixed.",
            "",
            "Gate state:",
            f"- state: {gate_state}",
            f"- ready for completion claim review: {'yes' if completion_ready else 'no'}",
            f"- missing: {', '.join(missing) if missing else 'none'}",
            f"- next command: `{next_command}`",
            "",
            "Selected patch target:",
            f"- target smoke test: `{target_test or 'unknown'}`",
            f"- target file: `{target_file or 'unknown'}`",
            f"- focused verification: `{focused_command or 'unknown'}`",
            "",
            "Evidence checks:",
            f"- failure learning cockpit ready: {'yes' if cockpit_ready else 'no'}",
            f"- apply contract reviewed: {'yes' if contract_reviewed else 'no'}",
            f"- application bridge ready: {'yes' if application_bridge_ready else 'no'}",
            f"- applied patch bound to contract: {'yes' if applied_patch_bound else 'no'}",
            f"- apply contract sha256 matches reviewed contract: {'yes' if apply_contract_hash_matches else 'no'}",
            f"- exact test contract ready: {'yes' if exact_test_contract_ready else 'no'}",
            f"- patch receipt ready: {'yes' if receipt_ready else 'no'}",
            f"- target file bound to changed files: {'yes' if target_file_bound else 'no'}",
            f"- pre-patch failing test receipt present: {'yes' if test_first_present else 'no'}",
            f"- focused verification present: {'yes' if focused_matches else 'no'}",
            f"- compile pass present: {'yes' if compile_passed else 'no'}",
            f"- rollback note present: {'yes' if rollback_present else 'no'}",
            f"- patch receipt hash present: {'yes' if patch_receipt_hash_present else 'no'}",
            f"- non-authorizing review contract ready: {'yes' if review_contract_ready else 'no'}",
            f"- patch review token sha256: {failure_patch_review_token_sha256 or 'missing'}",
            "",
            "Completion proof queue:",
            f"- next required command: `{next_command}`",
            f"- proof queue count: {len(proof_queue)}",
        ]
        lines.extend(f"- `{command}`" for command in proof_queue)
        lines.extend(
            [
                "",
                "Claim boundary:",
                "- A ready gate only permits a completion-review claim; it does not edit files, run tests, approve requests, complete tasks, or mark the regression fixed by itself.",
                "- If changed files, focused verification, compile pass, rollback, or apply-contract evidence changes, rerun this gate before claiming completion.",
                "",
                "Boundary:",
                "- Read-only completion gate only. It does not write tests, edit code, change memory, queue approvals, execute tools, speak, control the computer, complete tasks, or claim the patch is applied.",
            ]
        )
        return ToolResult(
            "failure_patch_completion_gate",
            receipt.ok,
            "\n".join(lines),
            _safe_metadata(
                limit=limit,
                selector=selector,
                gate_state=gate_state,
                ready_for_completion_claim_review=completion_ready,
                completion_claim_ready=completion_ready,
                cockpit_ready=cockpit_ready,
                apply_contract_reviewed=contract_reviewed,
                application_bridge_ready=application_bridge_ready,
                applied_patch_bound_to_contract=applied_patch_bound,
                apply_contract_sha256=bridge_meta.get("apply_contract_sha256", ""),
                supplied_apply_contract_sha256=bridge_meta.get("supplied_apply_contract_sha256", ""),
                apply_contract_hash_present=bridge_meta.get("apply_contract_hash_present"),
                apply_contract_hash_matches_expected=apply_contract_hash_matches,
                failure_regression_test_contract_sha256=bridge_meta.get("failure_regression_test_contract_sha256", ""),
                failure_regression_test_contract_present=bridge_meta.get("failure_regression_test_contract_present", False),
                failure_regression_test_contract_ready=exact_test_contract_ready,
                exact_test_contract_ready=exact_test_contract_ready,
                exact_test_contract_matches_apply_contract=bridge_meta.get("exact_test_contract_matches_apply_contract", False),
                exact_test_contract_requires_pre_patch_failure=True,
                exact_test_contract_authorizes_file_write=False,
                exact_test_contract_authorizes_patch_application=False,
                exact_test_contract_authorizes_tool_execution=False,
                exact_test_contract_authorizes_completion_claim=False,
                exact_test_contract_reusable_for_next_patch=False,
                contract_evidence=contract_evidence,
                patch_receipt_ready=receipt_ready,
                target_test=target_test,
                target_file=target_file,
                focused_command=focused_command,
                target_file_in_changed_files=target_file_bound,
                focused_verification_present=focused_matches,
                compile_pass_present=compile_passed,
                rollback_note_present=rollback_present,
                pre_patch_failing_test_receipt_present=test_first_present,
                test_first_receipt_present=test_first_present,
                patch_receipt_sha256=bridge_meta.get("patch_receipt_sha256", ""),
                patch_receipt_hash_present=patch_receipt_hash_present,
                patch_review_contract_rows=review_contract_rows,
                patch_review_contract_row_count=len(review_contract_rows),
                patch_review_contract_ready=review_contract_ready,
                patch_review_contract_summary=review_contract_summary,
                failure_patch_review_token_sha256=failure_patch_review_token_sha256,
                failure_patch_review_token_present=failure_patch_review_token_present,
                review_authorizes_patch_application=False,
                review_authorizes_file_write=False,
                review_authorizes_tool_execution=False,
                review_authorizes_model_call=False,
                review_authorizes_personal_data_read=False,
                review_authorizes_external_side_effect=False,
                review_authorizes_approval=False,
                review_authorizes_completion_claim=False,
                review_authorizes_learning_record=False,
                review_reusable_for_next_patch=False,
                review_token_authorizes_patch_application=False,
                review_token_authorizes_file_write=False,
                review_token_authorizes_tool_execution=False,
                review_token_authorizes_model_call=False,
                review_token_authorizes_personal_data_read=False,
                review_token_authorizes_external_side_effect=False,
                review_token_authorizes_approval=False,
                review_token_authorizes_completion_claim=False,
                review_token_authorizes_learning_record=False,
                review_token_reusable_for_next_patch=False,
                review_token_binds_exact_test_contract=bridge_meta.get("review_token_binds_exact_test_contract", False),
                next_patch_requires_fresh_review_token=True,
                missing=missing,
                missing_count=len(missing),
                proof_queue=proof_queue,
                proof_queue_count=len(proof_queue),
                next_proof_command=proof_queue[0],
                next_required_command=next_command,
                next_command=next_command,
                cockpit_metadata=cockpit_meta,
                patch_receipt_metadata=receipt_meta,
                application_bridge_metadata=bridge_meta,
                cockpit_output=cockpit.output[:1200],
                patch_receipt_output=receipt.output[:1200],
                application_bridge_output=application_bridge.output[:1200],
            ),
        )

    def failure_patch_handoff_packet(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 24)
        summary = str(args.get("summary") or args.get("receipt") or args.get("body") or "").strip()
        selector = str(args.get("cluster") or args.get("surface") or args.get("test") or "").strip().lower()
        if not selector and summary:
            selector = _receipt_field(summary, "cluster").lower()
        if not selector and summary:
            selector = summary.split(";", 1)[0].strip().lower()

        completion_gate = failure_patch_completion_gate({"summary": summary, "cluster": selector, "limit": limit})
        gate_meta = completion_gate.metadata
        target_test = str(gate_meta.get("target_test") or "")
        target_file = str(gate_meta.get("target_file") or "")
        focused_command = str(gate_meta.get("focused_command") or "")
        missing = list(gate_meta.get("missing") or [])
        for flag, label in [
            ("ready_for_completion_claim_review", "completion_gate_ready"),
            ("apply_contract_reviewed", "apply_contract_reviewed"),
            ("apply_contract_hash_matches_expected", "apply_contract_sha256_matches_reviewed_contract"),
            ("exact_test_contract_ready", "failure_regression_test_contract_ready"),
            ("application_bridge_ready", "failure_patch_application_bridge_ready"),
            ("applied_patch_bound_to_contract", "applied_patch_bound_to_contract"),
            ("patch_receipt_ready", "patch_receipt_ready"),
            ("focused_verification_present", "focused_verification_present"),
            ("compile_pass_present", "compile_pass_present"),
            ("rollback_note_present", "rollback_note_present"),
            ("pre_patch_failing_test_receipt_present", "pre_patch_failing_test_receipt_present"),
            ("patch_receipt_hash_present", "patch_receipt_sha256"),
            ("patch_review_contract_ready", "non_authorizing_patch_review_contract"),
            ("failure_patch_review_token_present", "failure_patch_review_token_sha256"),
        ]:
            if gate_meta.get(flag) is not True:
                missing.append(label)
        missing = list(dict.fromkeys(item for item in missing if item))
        handoff_state = "FAILURE_PATCH_HANDOFF_HELD" if missing else "FAILURE_PATCH_HANDOFF_READY_FOR_COMPLETION_REVIEW"

        pre_claim_commands = list(gate_meta.get("proof_queue") or [])
        handoff_command = f"failure patch handoff: {selector or target_test or 'selected cluster'}; changed files <files>; verification <focused pass>; compile <compile pass>; rollback <rollback note>; contract <apply contract reviewed>"
        if handoff_command not in pre_claim_commands:
            pre_claim_commands.append(handoff_command)
        post_claim_commands = [
            f"completion audit: patch repeated failure {target_test or 'selected cluster'}",
            "evidence ledger",
            f"completion claim gate: patch repeated failure {target_test or 'selected cluster'}",
            "work block checkpoint",
        ]
        next_command = pre_claim_commands[0] if pre_claim_commands else "failure learning cockpit"
        metadata = _safe_metadata(
            limit=limit,
            selector=selector,
            handoff_state=handoff_state,
            ready_for_completion_review=False,
            target_test=target_test,
            target_file=target_file,
            focused_command=focused_command,
            missing=missing,
            missing_count=len(missing),
            next_command=next_command,
            pre_claim_commands=pre_claim_commands,
            pre_claim_command_count=len(pre_claim_commands),
            post_claim_commands=post_claim_commands,
            post_claim_command_count=len(post_claim_commands),
            completion_gate_metadata=gate_meta,
            completion_gate_output=completion_gate.output[:1600],
            completion_claim_ready=gate_meta.get("completion_claim_ready"),
            apply_contract_reviewed=gate_meta.get("apply_contract_reviewed"),
            apply_contract_sha256=gate_meta.get("apply_contract_sha256", ""),
            supplied_apply_contract_sha256=gate_meta.get("supplied_apply_contract_sha256", ""),
            apply_contract_hash_present=gate_meta.get("apply_contract_hash_present"),
            apply_contract_hash_matches_expected=gate_meta.get("apply_contract_hash_matches_expected"),
            failure_regression_test_contract_sha256=gate_meta.get("failure_regression_test_contract_sha256", ""),
            failure_regression_test_contract_present=gate_meta.get("failure_regression_test_contract_present", False),
            failure_regression_test_contract_ready=gate_meta.get("failure_regression_test_contract_ready", False),
            exact_test_contract_ready=gate_meta.get("exact_test_contract_ready", False),
            exact_test_contract_matches_apply_contract=gate_meta.get("exact_test_contract_matches_apply_contract", False),
            exact_test_contract_requires_pre_patch_failure=True,
            exact_test_contract_authorizes_file_write=False,
            exact_test_contract_authorizes_patch_application=False,
            exact_test_contract_authorizes_tool_execution=False,
            exact_test_contract_authorizes_completion_claim=False,
            exact_test_contract_reusable_for_next_patch=False,
            application_bridge_ready=gate_meta.get("application_bridge_ready"),
            applied_patch_bound_to_contract=gate_meta.get("applied_patch_bound_to_contract"),
            patch_receipt_ready=gate_meta.get("patch_receipt_ready"),
            focused_verification_present=gate_meta.get("focused_verification_present"),
            compile_pass_present=gate_meta.get("compile_pass_present"),
            rollback_note_present=gate_meta.get("rollback_note_present"),
            pre_patch_failing_test_receipt_present=gate_meta.get("pre_patch_failing_test_receipt_present"),
            test_first_receipt_present=gate_meta.get("test_first_receipt_present"),
            patch_receipt_sha256=gate_meta.get("patch_receipt_sha256", ""),
            patch_receipt_hash_present=gate_meta.get("patch_receipt_hash_present"),
            patch_review_contract_rows=gate_meta.get("patch_review_contract_rows", []),
            patch_review_contract_row_count=gate_meta.get("patch_review_contract_row_count", 0),
            patch_review_contract_ready=gate_meta.get("patch_review_contract_ready"),
            patch_review_contract_summary=gate_meta.get("patch_review_contract_summary", []),
            failure_patch_review_token_sha256=gate_meta.get("failure_patch_review_token_sha256", ""),
            failure_patch_review_token_present=gate_meta.get("failure_patch_review_token_present"),
            review_authorizes_patch_application=False,
            review_authorizes_file_write=False,
            review_authorizes_tool_execution=False,
            review_authorizes_model_call=False,
            review_authorizes_personal_data_read=False,
            review_authorizes_external_side_effect=False,
            review_authorizes_approval=False,
            review_authorizes_completion_claim=False,
            review_authorizes_learning_record=False,
            review_reusable_for_next_patch=False,
            review_token_authorizes_patch_application=False,
            review_token_authorizes_file_write=False,
            review_token_authorizes_tool_execution=False,
            review_token_authorizes_model_call=False,
            review_token_authorizes_personal_data_read=False,
            review_token_authorizes_external_side_effect=False,
            review_token_authorizes_approval=False,
            review_token_authorizes_completion_claim=False,
            review_token_authorizes_learning_record=False,
            review_token_reusable_for_next_patch=False,
            review_token_binds_exact_test_contract=gate_meta.get("review_token_binds_exact_test_contract", False),
            next_patch_requires_fresh_review_token=True,
        )
        handoff_ready = completion_gate.ok and _failure_patch_handoff_ready({**metadata, "ready_for_completion_review": True})
        metadata["ready_for_completion_review"] = handoff_ready
        if handoff_ready:
            metadata["handoff_state"] = "FAILURE_PATCH_HANDOFF_READY_FOR_COMPLETION_REVIEW"
            handoff_state = metadata["handoff_state"]
        else:
            metadata["handoff_state"] = "FAILURE_PATCH_HANDOFF_HELD"
            handoff_state = metadata["handoff_state"]
        next_command = post_claim_commands[0] if handoff_ready else (pre_claim_commands[0] if pre_claim_commands else "failure learning cockpit")
        metadata["next_command"] = next_command

        lines = [
            "Jarvis failure patch handoff packet:",
            "This is read-only. It packages a repeated-failure patch for completion review after the learning cockpit, apply contract, patch receipt, focused verification, compile pass, rollback note, and completion gate are all visible.",
            "",
            "Handoff state:",
            f"- state: {handoff_state}",
            f"- ready for completion review: {'yes' if handoff_ready else 'no'}",
            f"- missing: {', '.join(missing) if missing else 'none'}",
            f"- next command: `{next_command}`",
            "",
            "Patch target:",
            f"- target smoke test: `{target_test or 'unknown'}`",
            f"- target file: `{target_file or 'unknown'}`",
            f"- focused verification: `{focused_command or 'unknown'}`",
            "",
            "Evidence carried forward:",
            f"- completion gate state: {gate_meta.get('gate_state')}",
            f"- learning cockpit ready: {'yes' if gate_meta.get('cockpit_ready') else 'no'}",
            f"- apply contract reviewed: {'yes' if gate_meta.get('apply_contract_reviewed') else 'no'}",
            f"- apply contract sha256 matches reviewed contract: {'yes' if gate_meta.get('apply_contract_hash_matches_expected') else 'no'}",
            f"- exact test contract ready: {'yes' if gate_meta.get('exact_test_contract_ready') else 'no'}",
            f"- application bridge ready: {'yes' if gate_meta.get('application_bridge_ready') else 'no'}",
            f"- applied patch bound to contract: {'yes' if gate_meta.get('applied_patch_bound_to_contract') else 'no'}",
            f"- patch receipt ready: {'yes' if gate_meta.get('patch_receipt_ready') else 'no'}",
            f"- pre-patch failing test receipt present: {'yes' if gate_meta.get('pre_patch_failing_test_receipt_present') else 'no'}",
            f"- focused verification present: {'yes' if gate_meta.get('focused_verification_present') else 'no'}",
            f"- compile pass present: {'yes' if gate_meta.get('compile_pass_present') else 'no'}",
            f"- rollback note present: {'yes' if gate_meta.get('rollback_note_present') else 'no'}",
            f"- patch receipt hash present: {'yes' if gate_meta.get('patch_receipt_hash_present') else 'no'}",
            f"- non-authorizing review contract ready: {'yes' if gate_meta.get('patch_review_contract_ready') else 'no'}",
            f"- patch review token sha256: {gate_meta.get('failure_patch_review_token_sha256') or 'missing'}",
            "",
            "Pre-claim proof chain:",
            f"- command count: {len(pre_claim_commands)}",
            *[f"- `{command}`" for command in pre_claim_commands],
            "",
            "Post-claim review queue:",
            *[f"- `{command}`" for command in post_claim_commands],
            "",
            "Boundary:",
            "- Read-only handoff only. It does not write tests, edit code, run verification, change memory, queue approvals, execute tools, speak, control the computer, complete tasks, or mark the regression fixed.",
        ]
        return ToolResult(
            "failure_patch_handoff_packet",
            completion_gate.ok,
            "\n".join(lines),
            metadata,
        )

    def failure_patch_closeout_packet(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 24)
        summary = str(args.get("summary") or args.get("receipt") or args.get("body") or "").strip()
        selector = str(args.get("cluster") or args.get("surface") or args.get("test") or "").strip().lower()
        if not selector and summary:
            selector = _receipt_field(summary, "cluster").lower()
        if not selector and summary:
            selector = summary.split(";", 1)[0].strip().lower()

        handoff = failure_patch_handoff_packet({"summary": summary, "cluster": selector, "limit": limit})
        handoff_meta = handoff.metadata
        target_test = str(handoff_meta.get("target_test") or "")
        target_file = str(handoff_meta.get("target_file") or "")
        focused_command = str(handoff_meta.get("focused_command") or "")

        audit_evidence = _short(
            args.get("completion_audit")
            or args.get("audit")
            or _receipt_field(summary, "completion audit")
            or _receipt_field(summary, "audit"),
            500,
        )
        evidence_ledger = _short(
            args.get("evidence_ledger")
            or args.get("ledger")
            or _receipt_field(summary, "evidence ledger")
            or _receipt_field(summary, "ledger"),
            500,
        )
        claim_gate = _short(
            args.get("completion_claim_gate")
            or args.get("claim_gate")
            or _receipt_field(summary, "completion claim gate")
            or _receipt_field(summary, "claim gate"),
            500,
        )
        post_claim_review = _short(
            args.get("post_claim_review")
            or args.get("review")
            or _receipt_field(summary, "post claim review")
            or _receipt_field(summary, "review"),
            500,
        )

        handoff_ready = handoff_meta.get("ready_for_completion_review") is True
        audit_ready = bool(audit_evidence) and any(token in audit_evidence.lower() for token in ("passed", "reviewed", "complete", "ready"))
        ledger_ready = bool(evidence_ledger) and any(token in evidence_ledger.lower() for token in ("passed", "reviewed", "present", "ready"))
        claim_gate_ready = bool(claim_gate) and any(token in claim_gate.lower() for token in ("passed", "reviewed", "ready", "allowed", "no blocker"))
        post_claim_ready = bool(post_claim_review) and any(token in post_claim_review.lower() for token in ("reviewed", "complete", "ready", "approved", "accepted"))
        patch_receipt_hash_present = handoff_meta.get("patch_receipt_hash_present") is True
        exact_test_contract_ready = handoff_meta.get("exact_test_contract_ready") is True
        review_contract_ready = handoff_meta.get("patch_review_contract_ready") is True
        failure_patch_review_token_present = handoff_meta.get("failure_patch_review_token_present") is True

        missing = list(handoff_meta.get("missing") or [])
        if not handoff_ready:
            missing.append("failure_patch_handoff_ready")
        if not audit_ready:
            missing.append("completion_audit_reviewed")
        if not ledger_ready:
            missing.append("evidence_ledger_reviewed")
        if not claim_gate_ready:
            missing.append("completion_claim_gate_reviewed")
        if not post_claim_ready:
            missing.append("post_claim_reviewed")
        if not patch_receipt_hash_present:
            missing.append("patch_receipt_sha256")
        if not exact_test_contract_ready:
            missing.append("failure_regression_test_contract_ready")
        if not review_contract_ready:
            missing.append("non_authorizing_patch_review_contract")
        if not failure_patch_review_token_present:
            missing.append("failure_patch_review_token_sha256")
        missing = list(dict.fromkeys(item for item in missing if item))

        closeout_ready = not missing
        closeout_state = "FAILURE_PATCH_CLOSEOUT_READY_FOR_LEARNING_RECORD" if closeout_ready else "FAILURE_PATCH_CLOSEOUT_HELD"
        pre_closeout_commands = list(dict.fromkeys(list(handoff_meta.get("pre_claim_commands") or []) + list(handoff_meta.get("post_claim_commands") or [])))
        closeout_command = f"failure patch closeout: {selector or target_test or 'selected cluster'}; completion audit <reviewed>; evidence ledger <reviewed>; completion claim gate <reviewed>; review <post-claim review>"
        if closeout_command not in pre_closeout_commands:
            pre_closeout_commands.append(closeout_command)
        learning_commands = [
            f"after-action learning packet <patch run id for {target_test or 'selected cluster'}>",
            "learning review",
            "work block checkpoint",
        ]
        next_command = learning_commands[0] if closeout_ready else (pre_closeout_commands[0] if pre_closeout_commands else "failure patch handoff")

        lines = [
            "Jarvis failure patch closeout packet:",
            "This is read-only. It closes the repeated-failure patch review after the handoff by requiring completion audit, evidence ledger, completion claim gate, and post-claim review evidence before learning can be recorded.",
            "",
            "Closeout state:",
            f"- state: {closeout_state}",
            f"- ready for learning record: {'yes' if closeout_ready else 'no'}",
            f"- missing: {', '.join(missing) if missing else 'none'}",
            f"- next command: `{next_command}`",
            "",
            "Patch target:",
            f"- target smoke test: `{target_test or 'unknown'}`",
            f"- target file: `{target_file or 'unknown'}`",
            f"- focused verification: `{focused_command or 'unknown'}`",
            "",
            "Closeout evidence:",
            f"- handoff ready: {'yes' if handoff_ready else 'no'}",
            f"- completion audit reviewed: {'yes' if audit_ready else 'no'}",
            f"- evidence ledger reviewed: {'yes' if ledger_ready else 'no'}",
            f"- completion claim gate reviewed: {'yes' if claim_gate_ready else 'no'}",
            f"- post-claim review complete: {'yes' if post_claim_ready else 'no'}",
            f"- patch receipt hash present: {'yes' if patch_receipt_hash_present else 'no'}",
            f"- non-authorizing review contract ready: {'yes' if review_contract_ready else 'no'}",
            f"- patch review token sha256: {handoff_meta.get('failure_patch_review_token_sha256') or 'missing'}",
            "",
            "Pre-closeout proof chain:",
            f"- command count: {len(pre_closeout_commands)}",
            *[f"- `{command}`" for command in pre_closeout_commands],
            "",
            "Learning record queue:",
            *[f"- `{command}`" for command in learning_commands],
            "",
            "Boundary:",
            "- Read-only closeout only. It does not write tests, edit code, run verification, change memory, queue approvals, execute tools, speak, control the computer, complete tasks, or mark the regression fixed.",
        ]
        return ToolResult(
            "failure_patch_closeout_packet",
            handoff.ok,
            "\n".join(lines),
            _safe_metadata(
                limit=limit,
                selector=selector,
                closeout_state=closeout_state,
                ready_for_learning_record=closeout_ready,
                target_test=target_test,
                target_file=target_file,
                focused_command=focused_command,
                handoff_ready=handoff_ready,
                completion_audit_reviewed=audit_ready,
                evidence_ledger_reviewed=ledger_ready,
                completion_claim_gate_reviewed=claim_gate_ready,
                post_claim_reviewed=post_claim_ready,
                missing=missing,
                missing_count=len(missing),
                next_command=next_command,
                pre_closeout_commands=pre_closeout_commands,
                pre_closeout_command_count=len(pre_closeout_commands),
                learning_record_commands=learning_commands,
                learning_record_command_count=len(learning_commands),
                failure_patch_handoff_metadata=handoff_meta,
                failure_patch_handoff_output=handoff.output[:1600],
                patch_receipt_ready=handoff_meta.get("patch_receipt_ready"),
                application_bridge_ready=handoff_meta.get("application_bridge_ready"),
                applied_patch_bound_to_contract=handoff_meta.get("applied_patch_bound_to_contract"),
                apply_contract_sha256=handoff_meta.get("apply_contract_sha256", ""),
                supplied_apply_contract_sha256=handoff_meta.get("supplied_apply_contract_sha256", ""),
                apply_contract_hash_present=handoff_meta.get("apply_contract_hash_present"),
                apply_contract_hash_matches_expected=handoff_meta.get("apply_contract_hash_matches_expected"),
                failure_regression_test_contract_sha256=handoff_meta.get("failure_regression_test_contract_sha256", ""),
                failure_regression_test_contract_present=handoff_meta.get("failure_regression_test_contract_present", False),
                failure_regression_test_contract_ready=exact_test_contract_ready,
                exact_test_contract_ready=exact_test_contract_ready,
                exact_test_contract_matches_apply_contract=handoff_meta.get("exact_test_contract_matches_apply_contract", False),
                exact_test_contract_requires_pre_patch_failure=True,
                exact_test_contract_authorizes_file_write=False,
                exact_test_contract_authorizes_patch_application=False,
                exact_test_contract_authorizes_tool_execution=False,
                exact_test_contract_authorizes_completion_claim=False,
                exact_test_contract_reusable_for_next_patch=False,
                pre_patch_failing_test_receipt_present=handoff_meta.get("pre_patch_failing_test_receipt_present"),
                test_first_receipt_present=handoff_meta.get("test_first_receipt_present"),
                focused_verification_present=handoff_meta.get("focused_verification_present"),
                compile_pass_present=handoff_meta.get("compile_pass_present"),
                rollback_note_present=handoff_meta.get("rollback_note_present"),
                patch_receipt_sha256=handoff_meta.get("patch_receipt_sha256", ""),
                patch_receipt_hash_present=patch_receipt_hash_present,
                patch_review_contract_rows=handoff_meta.get("patch_review_contract_rows", []),
                patch_review_contract_row_count=handoff_meta.get("patch_review_contract_row_count", 0),
                patch_review_contract_ready=review_contract_ready,
                patch_review_contract_summary=handoff_meta.get("patch_review_contract_summary", []),
                failure_patch_review_token_sha256=handoff_meta.get("failure_patch_review_token_sha256", ""),
                failure_patch_review_token_present=failure_patch_review_token_present,
                review_authorizes_patch_application=False,
                review_authorizes_file_write=False,
                review_authorizes_tool_execution=False,
                review_authorizes_model_call=False,
                review_authorizes_personal_data_read=False,
                review_authorizes_external_side_effect=False,
                review_authorizes_approval=False,
                review_authorizes_completion_claim=False,
                review_authorizes_learning_record=False,
                review_reusable_for_next_patch=False,
                review_token_authorizes_patch_application=False,
                review_token_authorizes_file_write=False,
                review_token_authorizes_tool_execution=False,
                review_token_authorizes_model_call=False,
                review_token_authorizes_personal_data_read=False,
                review_token_authorizes_external_side_effect=False,
                review_token_authorizes_approval=False,
                review_token_authorizes_completion_claim=False,
                review_token_authorizes_learning_record=False,
                review_token_reusable_for_next_patch=False,
                review_token_binds_exact_test_contract=handoff_meta.get("review_token_binds_exact_test_contract", False),
                next_patch_requires_fresh_review_token=True,
            ),
        )

    def failure_learning_record_packet(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 24)
        summary = str(args.get("summary") or args.get("receipt") or args.get("body") or "").strip()
        selector = str(args.get("cluster") or args.get("surface") or args.get("test") or "").strip().lower()
        if not selector and summary:
            selector = _receipt_field(summary, "cluster").lower()
        if not selector and summary:
            selector = summary.split(";", 1)[0].strip().lower()

        closeout = failure_patch_closeout_packet({"summary": summary, "cluster": selector, "limit": limit})
        closeout_meta = closeout.metadata
        target_test = str(closeout_meta.get("target_test") or "")
        target_file = str(closeout_meta.get("target_file") or "")
        focused_command = str(closeout_meta.get("focused_command") or "")

        after_action_learning_raw = _short(
            args.get("after_action_learning")
            or args.get("learning")
            or _receipt_field(summary, "after-action learning")
            or _receipt_field(summary, "after action learning")
            or _receipt_field(summary, "learning"),
            500,
        )
        record_target_raw = _short(
            args.get("record_target")
            or args.get("target")
            or _receipt_field(summary, "record target")
            or _receipt_field(summary, "target"),
            260,
        )
        regression_link_raw = _short(
            args.get("regression_link")
            or args.get("regression")
            or _receipt_field(summary, "regression link")
            or _receipt_field(summary, "regression"),
            500,
        )
        durability_note_raw = _short(
            args.get("durability")
            or args.get("durability_note")
            or _receipt_field(summary, "durability")
            or _receipt_field(summary, "durable"),
            500,
        )
        learning_record_sha256 = _short(
            args.get("learning_record_sha256")
            or args.get("record_sha256")
            or args.get("learning_sha256")
            or _receipt_field(summary, "learning record sha256")
            or _receipt_field(summary, "record sha256")
            or _receipt_field(summary, "learning sha256"),
            80,
        ).lower()

        closeout_ready = closeout_meta.get("ready_for_learning_record") is True
        after_action_learning = _safe_display(after_action_learning_raw, 500)
        record_target = _safe_display(record_target_raw, 260)
        regression_link = _safe_display(regression_link_raw, 500)
        durability_note = _safe_display(durability_note_raw, 500)

        learning_ready = bool(after_action_learning_raw) and any(
            token in after_action_learning_raw.lower()
            for token in ("reviewed", "recorded", "saved", "complete", "ready")
        )
        record_target_ready = bool(record_target_raw) and any(
            token in record_target_raw.lower()
            for token in ("learning review", "memory", "preference", "skill", "note", "test", "after-action")
        )
        regression_link_ready = bool(regression_link_raw) and bool(target_test) and (
            target_test in regression_link_raw or target_file in regression_link_raw or "regression" in regression_link_raw.lower()
        )
        durability_ready = bool(durability_note_raw) and any(
            token in durability_note_raw.lower()
            for token in ("durable", "saved", "checkpoint", "persisted", "recorded", "reviewed")
        )
        patch_receipt_hash_present = closeout_meta.get("patch_receipt_hash_present") is True
        exact_test_contract_ready = closeout_meta.get("exact_test_contract_ready") is True
        review_contract_ready = closeout_meta.get("patch_review_contract_ready") is True
        review_contract_rows = list(closeout_meta.get("patch_review_contract_rows") or [])
        failure_patch_review_token_sha256 = str(closeout_meta.get("failure_patch_review_token_sha256") or "")
        failure_patch_review_token_present = closeout_meta.get("failure_patch_review_token_present") is True
        learning_record_hash_present = _looks_like_sha256(learning_record_sha256)
        failure_learning_record_token_sha256 = _failure_learning_record_token_sha256(
            target_test=target_test,
            target_file=target_file,
            after_action_learning=after_action_learning,
            record_target=record_target,
            regression_link=regression_link,
            durability_note=durability_note,
            patch_receipt_sha256=str(closeout_meta.get("patch_receipt_sha256") or ""),
            learning_record_sha256=learning_record_sha256,
            apply_contract_sha256=str(closeout_meta.get("apply_contract_sha256") or ""),
            review_contract_rows=review_contract_rows,
        )
        learning_record_token_present = _looks_like_sha256(failure_learning_record_token_sha256)

        missing = list(closeout_meta.get("missing") or [])
        if not closeout_ready:
            missing.append("failure_patch_closeout_ready")
        if not learning_ready:
            missing.append("after_action_learning_reviewed")
        if not record_target_ready:
            missing.append("learning_record_target")
        if not regression_link_ready:
            missing.append("regression_test_linked")
        if not durability_ready:
            missing.append("durable_learning_record")
        if not patch_receipt_hash_present:
            missing.append("patch_receipt_sha256")
        if not exact_test_contract_ready:
            missing.append("failure_regression_test_contract_ready")
        if not review_contract_ready:
            missing.append("non_authorizing_patch_review_contract")
        if not failure_patch_review_token_present:
            missing.append("failure_patch_review_token_sha256")
        if not learning_record_hash_present:
            missing.append("learning_record_sha256")
        if not learning_record_token_present:
            missing.append("failure_learning_record_token_sha256")
        missing = list(dict.fromkeys(item for item in missing if item))

        record_ready = not missing
        record_state = "FAILURE_LEARNING_RECORD_READY" if record_ready else "FAILURE_LEARNING_RECORD_HELD"
        pre_record_commands = list(closeout_meta.get("pre_closeout_commands") or [])
        learning_commands = list(closeout_meta.get("learning_record_commands") or [])
        record_command = (
            f"failure learning record: {selector or target_test or 'selected cluster'}; "
            "after-action learning <reviewed>; record target <learning review or memory>; "
            "regression <target test linked>; durability <saved checkpoint or note>"
        )
        if record_command not in learning_commands:
            learning_commands.append(record_command)
        next_command = "work block checkpoint" if record_ready else (
            pre_record_commands[0] if pre_record_commands else (learning_commands[0] if learning_commands else "failure patch closeout")
        )

        lines = [
            "Jarvis failure learning record packet:",
            "This is read-only. It checks that a repeated-failure patch closeout has become a durable learning record rather than only a ready-to-record queue.",
            "",
            "Learning record state:",
            f"- state: {record_state}",
            f"- ready as durable learning record: {'yes' if record_ready else 'no'}",
            f"- missing: {', '.join(missing) if missing else 'none'}",
            f"- next command: `{next_command}`",
            "",
            "Patch target:",
            f"- target smoke test: `{target_test or 'unknown'}`",
            f"- target file: `{target_file or 'unknown'}`",
            f"- focused verification: `{focused_command or 'unknown'}`",
            "",
            "Learning evidence:",
            f"- closeout ready: {'yes' if closeout_ready else 'no'}",
            f"- after-action learning reviewed: {'yes' if learning_ready else 'no'}",
            f"- learning record target: {record_target or 'missing'}",
            f"- regression linked: {'yes' if regression_link_ready else 'no'}",
            f"- durable record evidence: {'yes' if durability_ready else 'no'}",
            f"- patch receipt hash present: {'yes' if patch_receipt_hash_present else 'no'}",
            f"- exact test contract ready: {'yes' if exact_test_contract_ready else 'no'}",
            f"- non-authorizing review contract ready: {'yes' if review_contract_ready else 'no'}",
            f"- patch review token sha256: {failure_patch_review_token_sha256 or 'missing'}",
            f"- learning record sha256: {learning_record_sha256 or 'missing'}",
            f"- failure learning record token sha256: {failure_learning_record_token_sha256 or 'missing'}",
            f"- record token authorizes learning record writes: no",
            f"- record token authorizes model calls: no",
            f"- record token authorizes personal-data reads: no",
            f"- record token authorizes external side effects: no",
            f"- record token authorizes completion claim: no",
            f"- record token reusable for next learning record: no",
            "",
            "Pre-record proof chain:",
            f"- command count: {len(pre_record_commands)}",
            *[f"- `{command}`" for command in pre_record_commands],
            "",
            "Learning record queue:",
            *[f"- `{command}`" for command in learning_commands],
            "",
            "Boundary:",
            "- Read-only learning record packet only. It does not write memory, notes, tests, files, approvals, or claims; it only verifies that the reviewed closeout has durable learning evidence.",
        ]
        return ToolResult(
            "failure_learning_record_packet",
            closeout.ok,
            "\n".join(lines),
            _safe_metadata(
                limit=limit,
                selector=selector,
                record_state=record_state,
                ready_for_durable_learning_record=record_ready,
                ready_for_learning_record=record_ready,
                target_test=target_test,
                target_file=target_file,
                focused_command=focused_command,
                closeout_ready=closeout_ready,
                after_action_learning=after_action_learning,
                after_action_learning_reviewed=learning_ready,
                learning_record_target=record_target,
                learning_record_target_present=record_target_ready,
                regression_link=regression_link,
                regression_test_linked=regression_link_ready,
                durable_learning_record=durability_ready,
                durability_note=durability_note,
                patch_receipt_sha256=closeout_meta.get("patch_receipt_sha256", ""),
                patch_receipt_hash_present=patch_receipt_hash_present,
                learning_record_sha256=learning_record_sha256,
                learning_record_hash_present=learning_record_hash_present,
                learning_artifact_hashes_present=bool(patch_receipt_hash_present and learning_record_hash_present),
                failure_learning_record_token_sha256=failure_learning_record_token_sha256,
                failure_learning_record_token_present=learning_record_token_present,
                record_token_authorizes_learning_record=False,
                record_token_authorizes_memory_write=False,
                record_token_authorizes_file_write=False,
                record_token_authorizes_tool_execution=False,
                record_token_authorizes_model_call=False,
                record_token_authorizes_personal_data_read=False,
                record_token_authorizes_external_side_effect=False,
                record_token_authorizes_approval=False,
                record_token_authorizes_completion_claim=False,
                record_token_reusable_for_next_learning_record=False,
                next_learning_record_requires_fresh_record_token=True,
                missing=missing,
                missing_count=len(missing),
                next_command=next_command,
                pre_record_commands=pre_record_commands,
                pre_record_command_count=len(pre_record_commands),
                learning_record_commands=learning_commands,
                learning_record_command_count=len(learning_commands),
                failure_patch_closeout_metadata=closeout_meta,
                failure_patch_closeout_output=closeout.output[:1600],
                patch_receipt_ready=closeout_meta.get("patch_receipt_ready"),
                application_bridge_ready=closeout_meta.get("application_bridge_ready"),
                applied_patch_bound_to_contract=closeout_meta.get("applied_patch_bound_to_contract"),
                apply_contract_sha256=closeout_meta.get("apply_contract_sha256", ""),
                supplied_apply_contract_sha256=closeout_meta.get("supplied_apply_contract_sha256", ""),
                apply_contract_hash_present=closeout_meta.get("apply_contract_hash_present"),
                apply_contract_hash_matches_expected=closeout_meta.get("apply_contract_hash_matches_expected"),
                failure_regression_test_contract_sha256=closeout_meta.get("failure_regression_test_contract_sha256", ""),
                failure_regression_test_contract_present=closeout_meta.get("failure_regression_test_contract_present", False),
                failure_regression_test_contract_ready=exact_test_contract_ready,
                exact_test_contract_ready=exact_test_contract_ready,
                exact_test_contract_matches_apply_contract=closeout_meta.get("exact_test_contract_matches_apply_contract", False),
                exact_test_contract_requires_pre_patch_failure=True,
                exact_test_contract_authorizes_file_write=False,
                exact_test_contract_authorizes_patch_application=False,
                exact_test_contract_authorizes_tool_execution=False,
                exact_test_contract_authorizes_completion_claim=False,
                exact_test_contract_reusable_for_next_patch=False,
                pre_patch_failing_test_receipt_present=closeout_meta.get("pre_patch_failing_test_receipt_present"),
                test_first_receipt_present=closeout_meta.get("test_first_receipt_present"),
                focused_verification_present=closeout_meta.get("focused_verification_present"),
                compile_pass_present=closeout_meta.get("compile_pass_present"),
                rollback_note_present=closeout_meta.get("rollback_note_present"),
                patch_review_contract_rows=review_contract_rows,
                patch_review_contract_row_count=closeout_meta.get("patch_review_contract_row_count", 0),
                patch_review_contract_ready=review_contract_ready,
                patch_review_contract_summary=closeout_meta.get("patch_review_contract_summary", []),
                failure_patch_review_token_sha256=failure_patch_review_token_sha256,
                failure_patch_review_token_present=failure_patch_review_token_present,
                review_authorizes_patch_application=False,
                review_authorizes_file_write=False,
                review_authorizes_tool_execution=False,
                review_authorizes_model_call=False,
                review_authorizes_personal_data_read=False,
                review_authorizes_external_side_effect=False,
                review_authorizes_approval=False,
                review_authorizes_completion_claim=False,
                review_authorizes_learning_record=False,
                review_reusable_for_next_patch=False,
                review_token_authorizes_patch_application=False,
                review_token_authorizes_file_write=False,
                review_token_authorizes_tool_execution=False,
                review_token_authorizes_model_call=False,
                review_token_authorizes_personal_data_read=False,
                review_token_authorizes_external_side_effect=False,
                review_token_authorizes_approval=False,
                review_token_authorizes_completion_claim=False,
                review_token_authorizes_learning_record=False,
                review_token_reusable_for_next_patch=False,
                review_token_binds_exact_test_contract=closeout_meta.get("review_token_binds_exact_test_contract", False),
                next_patch_requires_fresh_review_token=True,
            ),
        )

    def failure_learning_closure_ledger(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 24)
        summary = str(args.get("summary") or args.get("receipt") or args.get("body") or "").strip()
        selector = str(args.get("cluster") or args.get("surface") or args.get("test") or "").strip().lower()
        if not selector and summary:
            selector = _receipt_field(summary, "cluster").lower()
        if not selector and summary:
            selector = summary.split(";", 1)[0].strip().lower()

        cockpit = failure_learning_cockpit({"cluster": selector, "limit": limit})
        receipt = failure_patch_receipt_packet({"summary": summary, "cluster": selector, "limit": limit})
        application_bridge = failure_patch_application_bridge({"summary": summary, "cluster": selector, "limit": limit})
        completion = failure_patch_completion_gate({"summary": summary, "cluster": selector, "limit": limit})
        handoff = failure_patch_handoff_packet({"summary": summary, "cluster": selector, "limit": limit})
        closeout = failure_patch_closeout_packet({"summary": summary, "cluster": selector, "limit": limit})
        record = failure_learning_record_packet({"summary": summary, "cluster": selector, "limit": limit})

        cockpit_meta = cockpit.metadata
        receipt_meta = receipt.metadata
        bridge_meta = application_bridge.metadata
        completion_meta = completion.metadata
        handoff_meta = handoff.metadata
        closeout_meta = closeout.metadata
        record_meta = record.metadata

        stage_rows = [
            ("cockpit", cockpit_meta.get("cockpit_state"), cockpit_meta.get("ready_for_patch_review"), cockpit_meta.get("missing", [])),
            ("patch_receipt", "PATCH_RECEIPT_READY_FOR_COMPLETION_REVIEW" if receipt_meta.get("patch_receipt_ready") else "PATCH_RECEIPT_INCOMPLETE", receipt_meta.get("patch_receipt_ready"), receipt_meta.get("missing", [])),
            ("application_bridge", bridge_meta.get("bridge_state"), bridge_meta.get("application_bridge_ready"), bridge_meta.get("missing", [])),
            ("completion_gate", completion_meta.get("gate_state"), completion_meta.get("ready_for_completion_claim_review"), completion_meta.get("missing", [])),
            ("handoff", handoff_meta.get("handoff_state"), handoff_meta.get("ready_for_completion_review"), handoff_meta.get("missing", [])),
            ("closeout", closeout_meta.get("closeout_state"), closeout_meta.get("ready_for_learning_record"), closeout_meta.get("missing", [])),
            ("learning_record", record_meta.get("record_state"), record_meta.get("ready_for_durable_learning_record"), record_meta.get("missing", [])),
        ]
        closure_stage_rows = [
            {
                "stage": stage,
                "state": state,
                "ready": ready is True,
                "missing": list(stage_missing or []),
                "authorizes_patch_application": False,
                "authorizes_file_write": False,
                "authorizes_tool_execution": False,
                "authorizes_model_call": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "authorizes_approval": False,
                "authorizes_completion_claim": False,
                "authorizes_learning_record": False,
                "reusable_for_next_patch": False,
                "reusable_for_next_learning_record": False,
                "reusable_for_next_completion_claim": False,
            }
            for stage, state, ready, stage_missing in stage_rows
        ]
        missing: list[str] = []
        for stage, _state, ready, stage_missing in stage_rows:
            if ready is not True:
                missing.append(stage)
            missing.extend(str(item) for item in (stage_missing or []) if str(item).strip())
        missing = list(dict.fromkeys(missing))

        if record_meta.get("ready_for_durable_learning_record") is True:
            ledger_state = "FAILURE_LEARNING_CLOSURE_READY"
        elif closeout_meta.get("ready_for_learning_record") is not True:
            ledger_state = "FAILURE_LEARNING_CLOSURE_HELD_FOR_CLOSEOUT"
        elif completion_meta.get("ready_for_completion_claim_review") is not True:
            ledger_state = "FAILURE_LEARNING_CLOSURE_HELD_FOR_COMPLETION_REVIEW"
        elif bridge_meta.get("application_bridge_ready") is not True:
            ledger_state = "FAILURE_LEARNING_CLOSURE_HELD_FOR_APPLICATION_BRIDGE"
        elif receipt_meta.get("patch_receipt_ready") is not True:
            ledger_state = "FAILURE_LEARNING_CLOSURE_HELD_FOR_PATCH_RECEIPT"
        else:
            ledger_state = "FAILURE_LEARNING_CLOSURE_HELD_FOR_DURABLE_RECORD"

        target_test = str(record_meta.get("target_test") or closeout_meta.get("target_test") or receipt_meta.get("target_test") or "")
        target_file = str(record_meta.get("target_file") or closeout_meta.get("target_file") or receipt_meta.get("target_file") or "")
        focused_command = str(record_meta.get("focused_command") or closeout_meta.get("focused_command") or receipt_meta.get("focused_command") or "")
        patch_receipt_hash_present = record_meta.get("patch_receipt_hash_present") is True
        learning_record_hash_present = record_meta.get("learning_record_hash_present") is True
        learning_artifact_hashes_present = bool(patch_receipt_hash_present and learning_record_hash_present)
        failure_learning_record_token_sha256 = str(record_meta.get("failure_learning_record_token_sha256") or "")
        learning_record_token_present = record_meta.get("failure_learning_record_token_present") is True
        apply_contract_hash_matches = record_meta.get("apply_contract_hash_matches_expected") is True
        exact_test_contract_ready = record_meta.get("exact_test_contract_ready") is True
        review_contract_ready = record_meta.get("patch_review_contract_ready") is True
        review_contract_rows = list(record_meta.get("patch_review_contract_rows") or [])
        failure_patch_review_token_sha256 = str(record_meta.get("failure_patch_review_token_sha256") or "")
        failure_patch_review_token_present = record_meta.get("failure_patch_review_token_present") is True
        proof_queue = _ordered_commands(
            [
                f"failure learning cockpit: {selector or target_test or 'selected cluster'}",
                f"failure apply contract: {selector or target_test or 'selected cluster'}",
                f"failure patch receipt: {selector or target_test or 'selected cluster'}; changed files <files>; test first <pre-patch failing receipt>; verification <focused pass>; compile <compile pass>; rollback <rollback note>",
                f"failure patch application bridge: {selector or target_test or 'selected cluster'}; contract <apply contract reviewed>; application bridge <applied patch bound to contract>",
                f"failure patch completion gate: {selector or target_test or 'selected cluster'}; contract <apply contract reviewed>",
                f"failure patch handoff: {selector or target_test or 'selected cluster'}",
                f"failure patch closeout: {selector or target_test or 'selected cluster'}; completion audit <reviewed>; evidence ledger <reviewed>; completion claim gate <reviewed>; review <post-claim review>",
                f"failure learning record: {selector or target_test or 'selected cluster'}; after-action learning <reviewed>; record target <learning review>; regression <linked>; durability <saved>",
                "learning review",
                "evidence ledger",
                f"completion claim gate: improve AGI gate evaluation and learning loop",
                "failure learning closure ledger: <next repeated failure with fresh patch receipt and learning record>",
            ]
        )
        failure_learning_closure_token_sha256 = _failure_learning_closure_token_sha256(
            target_test=target_test,
            target_file=target_file,
            patch_receipt_sha256=str(record_meta.get("patch_receipt_sha256") or ""),
            learning_record_sha256=str(record_meta.get("learning_record_sha256") or ""),
            apply_contract_sha256=str(record_meta.get("apply_contract_sha256") or ""),
            failure_patch_review_token_sha256=failure_patch_review_token_sha256,
            failure_learning_record_token_sha256=failure_learning_record_token_sha256,
            after_action_learning=str(record_meta.get("after_action_learning") or ""),
            learning_record_target=str(record_meta.get("learning_record_target") or ""),
            regression_link=str(record_meta.get("regression_link") or ""),
            durability_note=str(record_meta.get("durability_note") or ""),
            review_contract_rows=review_contract_rows,
            closure_stage_rows=closure_stage_rows,
            proof_queue=proof_queue,
        )
        closure_token_present = _looks_like_sha256(failure_learning_closure_token_sha256)
        closure_token_boundary_rows = _failure_learning_closure_token_boundary_rows(
            token_sha256=failure_learning_closure_token_sha256,
            record_token_sha256=failure_learning_record_token_sha256,
            source="failure_learning_closure_ledger",
        )
        closure_token_boundary_ready = _failure_learning_closure_token_boundary_ready(
            closure_token_boundary_rows,
            token_sha256=failure_learning_closure_token_sha256,
            record_token_sha256=failure_learning_record_token_sha256,
        )
        if not learning_artifact_hashes_present:
            missing.append("learning_artifact_hashes")
            missing = list(dict.fromkeys(missing))
            if ledger_state == "FAILURE_LEARNING_CLOSURE_READY":
                ledger_state = "FAILURE_LEARNING_CLOSURE_HELD_FOR_ARTIFACT_HASHES"
        if not apply_contract_hash_matches:
            missing.append("apply_contract_sha256_matches_reviewed_contract")
            missing = list(dict.fromkeys(missing))
            if ledger_state == "FAILURE_LEARNING_CLOSURE_READY":
                ledger_state = "FAILURE_LEARNING_CLOSURE_HELD_FOR_CONTRACT_HASH"
        if not exact_test_contract_ready:
            missing.append("failure_regression_test_contract_ready")
            missing = list(dict.fromkeys(missing))
            if ledger_state == "FAILURE_LEARNING_CLOSURE_READY":
                ledger_state = "FAILURE_LEARNING_CLOSURE_HELD_FOR_EXACT_TEST_CONTRACT"
        if not review_contract_ready:
            missing.append("non_authorizing_patch_review_contract")
            missing = list(dict.fromkeys(missing))
            if ledger_state == "FAILURE_LEARNING_CLOSURE_READY":
                ledger_state = "FAILURE_LEARNING_CLOSURE_HELD_FOR_REVIEW_CONTRACT"
        if not failure_patch_review_token_present:
            missing.append("failure_patch_review_token_sha256")
            missing = list(dict.fromkeys(missing))
            if ledger_state == "FAILURE_LEARNING_CLOSURE_READY":
                ledger_state = "FAILURE_LEARNING_CLOSURE_HELD_FOR_REVIEW_TOKEN"
        if not learning_record_token_present:
            missing.append("failure_learning_record_token_sha256")
            missing = list(dict.fromkeys(missing))
            if ledger_state == "FAILURE_LEARNING_CLOSURE_READY":
                ledger_state = "FAILURE_LEARNING_CLOSURE_HELD_FOR_RECORD_TOKEN"
        if not closure_token_present:
            missing.append("failure_learning_closure_token_sha256")
            missing = list(dict.fromkeys(missing))
            if ledger_state == "FAILURE_LEARNING_CLOSURE_READY":
                ledger_state = "FAILURE_LEARNING_CLOSURE_HELD_FOR_CLOSURE_TOKEN"
        next_command = (
            "learning review"
            if ledger_state == "FAILURE_LEARNING_CLOSURE_READY"
            else proof_queue[0]
        )

        lines = [
            "Jarvis failure learning closure ledger:",
            "This is read-only. It binds the repeated-failure learning cockpit, apply contract, patch receipt, completion gate, handoff, closeout, and durable learning record before Jarvis treats a fix as learned.",
            "",
            "Ledger state:",
            f"- state: {ledger_state}",
            f"- ready as durable learning closure: {'yes' if ledger_state == 'FAILURE_LEARNING_CLOSURE_READY' else 'no'}",
            f"- missing: {', '.join(missing) if missing else 'none'}",
            f"- next command: `{next_command}`",
            "",
            "Patch target:",
            f"- target smoke test: `{target_test or 'unknown'}`",
            f"- target file: `{target_file or 'unknown'}`",
            f"- focused verification: `{focused_command or 'unknown'}`",
            f"- apply contract sha256: {record_meta.get('apply_contract_sha256') or 'missing'}",
            f"- exact test contract sha256: {record_meta.get('failure_regression_test_contract_sha256') or 'missing'}",
            f"- exact test contract ready: {'yes' if exact_test_contract_ready else 'no'}",
            f"- supplied apply contract sha256: {record_meta.get('supplied_apply_contract_sha256') or 'missing'}",
            f"- apply contract hash matches reviewed contract: {'yes' if apply_contract_hash_matches else 'no'}",
            f"- patch receipt sha256: {record_meta.get('patch_receipt_sha256') or 'missing'}",
            f"- learning record sha256: {record_meta.get('learning_record_sha256') or 'missing'}",
            f"- learning artifact hashes present: {'yes' if learning_artifact_hashes_present else 'no'}",
            f"- failure learning record token sha256: {failure_learning_record_token_sha256 or 'missing'}",
            f"- record token authorizes learning record writes: no",
            f"- record token authorizes model calls: no",
            f"- record token authorizes personal-data reads: no",
            f"- record token authorizes external side effects: no",
            f"- record token authorizes completion claim: no",
            f"- non-authorizing review contract ready: {'yes' if review_contract_ready else 'no'}",
            f"- patch review token sha256: {failure_patch_review_token_sha256 or 'missing'}",
            f"- failure learning closure token sha256: {failure_learning_closure_token_sha256 or 'missing'}",
            f"- closure token boundary rows: {len(closure_token_boundary_rows)}",
            f"- closure token reusable for next patch: no",
            f"- closure token authorizes completion claim: no",
            "",
            "Closure stages:",
        ]
        for stage, state, ready, stage_missing in stage_rows:
            lines.append(
                f"- {stage}: {'ready' if ready is True else 'held'}; state {state or 'unknown'}; missing {', '.join(str(item) for item in (stage_missing or []) if str(item).strip()) or 'none'}"
            )
        lines.extend(
            [
                "",
            "Fresh learning review boundary:",
                "- prior learning closure token is proof-only and cannot authorize patch application, file writes, tool execution, model calls, personal-data reads, external side effects, approvals, completion claims, learning records, or reuse for the next patch.",
                "- every new repeated-failure patch needs a fresh learning cockpit, apply contract, patch receipt, application bridge, closeout, learning record, and closure ledger token.",
                "- closure token boundary rows are non-authorizing proof only and cannot be reused for the next patch, learning record, or completion claim.",
            "",
            "Proof queue:",
                f"- next required command: `{next_command}`",
                f"- proof queue count: {len(proof_queue)}",
                *[f"- `{command}`" for command in proof_queue],
                "",
                "Boundary:",
                "- Read-only closure ledger only. It does not write tests, edit code, run verification, change memory, queue approvals, execute tools, speak, control the computer, complete tasks, or claim the regression fixed.",
            ]
        )
        return ToolResult(
            "failure_learning_closure_ledger",
            True,
            "\n".join(lines),
            _safe_metadata(
                limit=limit,
                selector=selector,
                ledger_state=ledger_state,
                ready_as_durable_learning_closure=ledger_state == "FAILURE_LEARNING_CLOSURE_READY",
                patch_receipt_sha256=record_meta.get("patch_receipt_sha256", ""),
                patch_receipt_hash_present=patch_receipt_hash_present,
                learning_record_sha256=record_meta.get("learning_record_sha256", ""),
                learning_record_hash_present=learning_record_hash_present,
                learning_artifact_hashes_present=learning_artifact_hashes_present,
                after_action_learning=record_meta.get("after_action_learning", ""),
                learning_record_target=record_meta.get("learning_record_target", ""),
                regression_link=record_meta.get("regression_link", ""),
                durability_note=record_meta.get("durability_note", ""),
                failure_learning_record_token_sha256=failure_learning_record_token_sha256,
                failure_learning_record_token_present=learning_record_token_present,
                record_token_authorizes_learning_record=False,
                record_token_authorizes_memory_write=False,
                record_token_authorizes_file_write=False,
                record_token_authorizes_tool_execution=False,
                record_token_authorizes_model_call=False,
                record_token_authorizes_personal_data_read=False,
                record_token_authorizes_external_side_effect=False,
                record_token_authorizes_approval=False,
                record_token_authorizes_completion_claim=False,
                record_token_reusable_for_next_learning_record=False,
                next_learning_record_requires_fresh_record_token=True,
                failure_learning_closure_token_sha256=failure_learning_closure_token_sha256,
                failure_learning_closure_token_present=closure_token_present,
                failure_learning_closure_token_boundary_rows=closure_token_boundary_rows,
                failure_learning_closure_token_boundary_row_count=len(closure_token_boundary_rows),
                failure_learning_closure_token_boundary_ready=closure_token_boundary_ready,
                closure_stage_rows=closure_stage_rows,
                closure_stage_row_count=len(closure_stage_rows),
                closure_token_reusable_for_next_patch=False,
                closure_token_reusable_for_next_learning_record=False,
                closure_token_reusable_for_next_completion_claim=False,
                closure_token_authorizes_patch_application=False,
                closure_token_authorizes_file_write=False,
                closure_token_authorizes_tool_execution=False,
                closure_token_authorizes_model_call=False,
                closure_token_authorizes_personal_data_read=False,
                closure_token_authorizes_external_side_effect=False,
                closure_token_authorizes_approval=False,
                closure_token_authorizes_completion_claim=False,
                closure_token_authorizes_learning_record=False,
                next_patch_requires_fresh_learning_closure=True,
                next_completion_claim_requires_fresh_learning_closure=True,
                apply_contract_sha256=record_meta.get("apply_contract_sha256", ""),
                supplied_apply_contract_sha256=record_meta.get("supplied_apply_contract_sha256", ""),
                apply_contract_hash_present=record_meta.get("apply_contract_hash_present"),
                apply_contract_hash_matches_expected=apply_contract_hash_matches,
                failure_regression_test_contract_sha256=record_meta.get("failure_regression_test_contract_sha256", ""),
                failure_regression_test_contract_present=record_meta.get("failure_regression_test_contract_present", False),
                failure_regression_test_contract_ready=exact_test_contract_ready,
                exact_test_contract_ready=exact_test_contract_ready,
                exact_test_contract_matches_apply_contract=record_meta.get("exact_test_contract_matches_apply_contract", False),
                exact_test_contract_requires_pre_patch_failure=True,
                exact_test_contract_authorizes_file_write=False,
                exact_test_contract_authorizes_patch_application=False,
                exact_test_contract_authorizes_tool_execution=False,
                exact_test_contract_authorizes_completion_claim=False,
                exact_test_contract_reusable_for_next_patch=False,
                patch_review_contract_rows=review_contract_rows,
                patch_review_contract_row_count=record_meta.get("patch_review_contract_row_count", 0),
                patch_review_contract_ready=review_contract_ready,
                patch_review_contract_summary=record_meta.get("patch_review_contract_summary", []),
                failure_patch_review_token_sha256=failure_patch_review_token_sha256,
                failure_patch_review_token_present=failure_patch_review_token_present,
                review_authorizes_patch_application=False,
                review_authorizes_file_write=False,
                review_authorizes_tool_execution=False,
                review_authorizes_model_call=False,
                review_authorizes_personal_data_read=False,
                review_authorizes_external_side_effect=False,
                review_authorizes_approval=False,
                review_authorizes_completion_claim=False,
                review_authorizes_learning_record=False,
                review_reusable_for_next_patch=False,
                review_token_authorizes_patch_application=False,
                review_token_authorizes_file_write=False,
                review_token_authorizes_tool_execution=False,
                review_token_authorizes_model_call=False,
                review_token_authorizes_personal_data_read=False,
                review_token_authorizes_external_side_effect=False,
                review_token_authorizes_approval=False,
                review_token_authorizes_completion_claim=False,
                review_token_authorizes_learning_record=False,
                review_token_reusable_for_next_patch=False,
                review_token_binds_exact_test_contract=record_meta.get("review_token_binds_exact_test_contract", False),
                next_patch_requires_fresh_review_token=True,
                target_test=target_test,
                target_file=target_file,
                focused_command=focused_command,
                missing=missing,
                missing_count=len(missing),
                next_command=next_command,
                next_required_command=next_command,
                proof_queue=proof_queue,
                proof_queue_count=len(proof_queue),
                next_proof_command=proof_queue[0],
                stage_rows=closure_stage_rows,
                stage_count=len(closure_stage_rows),
                cockpit_metadata=cockpit_meta,
                patch_receipt_metadata=receipt_meta,
                application_bridge_metadata=bridge_meta,
                completion_gate_metadata=completion_meta,
                handoff_metadata=handoff_meta,
                closeout_metadata=closeout_meta,
                learning_record_metadata=record_meta,
            ),
        )

    def save_feedback_actions(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 12)
        with store.generated_report_publication_fence():
            body, metadata = build_feedback_actions(limit)
            path, _content_sha256, _source_revision = vault.write_feedback_actions_with_evidence(
                f"# Feedback Actions\n\n{body}",
                store_identity=store.get_store_identity(),
                source_payload={"body": body, "metadata": metadata},
            )
        path_display = _safe_vault_path_display(path, vault)
        metadata = _feedback_handoff_metadata(
            "feedback_actions_handoff",
            _feedback_report_handoff(
                source="save_feedback_actions",
                count=int(metadata.get("count") or 0),
                themes=metadata.get("themes") or [],
                limit=limit,
                saved=True,
                path_display=path_display,
                unreadable_feedback_rows=int(metadata.get("unreadable_feedback_rows") or 0),
            ),
            writes=True,
            writes_memory=False,
            writes_database=False,
            count=int(metadata.get("count") or 0),
            themes=metadata.get("themes") or [],
            limit=limit,
            readable_feedback_rows=int(metadata.get("readable_feedback_rows") or 0),
            unreadable_feedback_rows=int(metadata.get("unreadable_feedback_rows") or 0),
            path=str(path),
            path_display=path_display,
        )
        return ToolResult("save_feedback_actions", True, f"Feedback actions saved: {path_display}\n\n{body}", metadata)

    return (
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
    )
