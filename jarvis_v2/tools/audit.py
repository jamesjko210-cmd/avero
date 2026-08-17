from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import re
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
    LOCAL_READ_INPUT_RECOVERY_ACTION,
    RESOURCE_NOT_FOUND_RECOVERY_ACTION,
    declare_resource_not_found_failure,
    declare_retryable_local_read_failure,
)
from jarvis_v2.agent.model_provider import safe_model_usage_receipt
from jarvis_v2.agent.types import ToolResult
from jarvis_v2.memory.store import MemoryStore
from jarvis_v2.tools.audit_meta import EXECUTION_META_TOOLS
from jarvis_v2.tools.continuity import _checkpoint_recovery_execute_handoff_ready


MAX_AUDIT_LIMIT = 200
AFTER_ACTION_META_TOOLS = {*EXECUTION_META_TOOLS, "execution_learning_closure_packet"}
KNOWN_RISK_LEVELS = frozenset(
    {"READ_ONLY", "LOCAL_SAFE", "PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"}
)
RISKY_RISK_LEVELS = frozenset({"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"})
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")
SECRET_VALUE_RE = re.compile(
    r"(?:"
    r"\bsk_(?:live|test)_[A-Za-z0-9_-]+"
    r"|\bgh[pousr]_[A-Za-z0-9_-]+"
    r"|\bxox[baprs]-[A-Za-z0-9-]+"
    r"|\bAIza[A-Za-z0-9_-]{16,}"
    r"|\b\d{6,}:[A-Za-z0-9_-]{20,}"
    r")",
    re.IGNORECASE,
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
APPROVAL_HOLD_METADATA_KEYS = (
    "requires_confirmation",
    "requires_approval",
    "approval_required",
)
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

CHECKPOINT_RECOVERY_EXECUTE_RECEIPT_CONTRACT_KEYS = (
    "reviewed",
    "missing_fields",
    "recovery_followthrough_gate_state",
    "recovery_closure_missing",
    "recovery_closure_missing_count",
    "recovery_closure_required_evidence",
    "recovery_closure_required_evidence_count",
    "recovery_closure_proof_queue",
    "recovery_closure_proof_queue_count",
    "recovery_closure_next_proof_command",
    "recovery_closure_proof_queue_ready",
    "recovery_closure_approval_boundary",
    "recovery_closure_next_safe_command",
    "risky_recovery_signals",
    "risky_recovery_signal_count",
    "approval_reference_provided",
    "approval_reference",
    "recovery_step_approval_required_before_recovery",
    "recovery_step_approval_boundary_rows",
    "recovery_step_approval_boundary_row_count",
    "recovery_step_approval_boundary_ready",
    "recovery_step_approval_boundary_token_sha256",
    "recovery_step_approval_boundary_token_present",
    "recovery_step_approval_boundary_as_prior_proof",
    "recovery_step_approval_proof_queue",
    "recovery_step_approval_proof_queue_count",
    "recovery_step_next_approval_proof_command",
    "recovery_step_sha256",
    "recovery_verification_sha256",
    "contains_raw_step",
    "contains_raw_verification",
    "recovery_step_approval_boundary_authorizes_action_now",
    "recovery_step_approval_boundary_authorizes_risky_work",
    "recovery_step_approval_boundary_authorizes_unreviewed_followthrough",
    "recovery_step_approval_boundary_authorizes_approval",
    "recovery_step_approval_boundary_authorizes_model_call",
    "recovery_step_approval_boundary_authorizes_tool_execution",
    "recovery_step_approval_boundary_authorizes_personal_data_read",
    "recovery_step_approval_boundary_authorizes_external_side_effect",
    "recovery_step_approval_boundary_authorizes_timebox_reuse",
    "recovery_step_approval_boundary_reusable_for_next_review",
    "recovery_step_approval_boundary_reusable_for_recovery_review",
    "changed_state",
    "writes_notes",
    "writes_files",
    "calls_model",
    "executes_tools",
    "queues_approval",
    "controls_computer",
    "reads_private_data",
    "reads_personal_data",
    "executes_side_effect",
    "external_side_effect",
    "authorizes_execution",
    "authorizes_recovery_followthrough",
    "authorizes_local_safe_step",
    "authorizes_risky_work",
    "authorizes_approval",
    "authorizes_model_call",
    "authorizes_tool_execution",
    "authorizes_personal_data_read",
    "authorizes_external_side_effect",
    "reusable_for_recovery_review",
    "timebox_state",
    "can_continue_now",
    "should_stop_now",
    "timebox_review_contract_ready",
    "operator_timebox_precontinuation_contract_ready",
    "operator_timebox_precontinuation_contract_state",
    "operator_timebox_precontinuation_allows_continuation",
    "operator_timebox_precontinuation_blocks_continuation",
    "operator_timebox_precontinuation_requires_stop",
    "operator_timebox_precontinuation_requires_fresh_timebox_review",
    "operator_timebox_precontinuation_allowed_states",
    "supersession_state",
    "can_continue_under_latest_instruction",
    "latest_instruction_is_stop",
    "supersession_contract_ready",
    "supersession_token_boundary_ready",
    "operator_supersession_precontinuation_contract_ready",
    "operator_supersession_precontinuation_contract_state",
    "operator_supersession_precontinuation_token_boundary_ready",
    "operator_supersession_precontinuation_allows_continuation",
    "operator_supersession_precontinuation_blocks_continuation",
    "operator_supersession_precontinuation_requires_stop",
    "operator_supersession_precontinuation_requires_fresh_latest_instruction_review",
    "operator_supersession_precontinuation_allowed_states",
)


@dataclass(frozen=True)
class _RunApprovalEvidence:
    risk_needs_approval: bool
    approval_linked: bool
    successful_proof: bool
    approval_problem: bool
    outcome_unknown: bool = False
    verdict: str = ""


def _bounded_int(value: Any, default: int, low: int = 1, high: int = MAX_AUDIT_LIMIT) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _metadata_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


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


def _safe_text(value: Any) -> str:
    try:
        return "" if value is None else str(value)
    except Exception:
        return ""


def _normalized_risk(value: Any) -> str | None:
    risk = _safe_text(value).strip().upper()
    return risk if risk in KNOWN_RISK_LEVELS else None


def _risk_needs_approval(value: Any) -> bool:
    normalized = _normalized_risk(value)
    return normalized is None or normalized in RISKY_RISK_LEVELS


def _first_safe_text(*values: Any, default: str = "") -> str:
    for value in values:
        text = _safe_text(value).strip()
        if text:
            return text
    return default


def _is_approval_hold_metadata(metadata: dict[str, Any]) -> bool:
    if any(_metadata_truthy_loose(metadata.get(key)) for key in APPROVAL_HOLD_METADATA_KEYS):
        return True
    for key in (
        "failure_kind",
        "failure_stage",
        "stage",
        "guard_reason",
        "reason",
        "status",
        "send_status",
        "call_status",
    ):
        token = _normalized_metadata_token(metadata.get(key))
        if token in APPROVAL_HOLD_VALUES:
            return True
        if "approval" in token and any(marker in token for marker in ("required", "requires", "gate", "gated", "hold", "held")):
            return True
    return False


def _recent_run_failure_kind(metadata: dict[str, Any]) -> str:
    if _is_approval_hold_metadata(metadata):
        return "approval_required"
    return _short(metadata.get("failure_kind"), limit=80)


def _looks_like_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "approved_linked_runs": 0,
        "calls_model": False,
        "calls_external_service": False,
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
        "writes_database": False,
        "writes_notes": False,
        "writes_memory": False,
        "controls_computer": False,
        "external_side_effect": False,
        "requires_approval": False,
        "speaks": False,
        "completes_tasks": False,
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
    }
    metadata.update(extra)
    return metadata


def _lookup_recovery_metadata(
    refresh_command: str,
    retry_command: str,
    *,
    requires_setup_check: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    commands = ["setup check"] if requires_setup_check else []
    for command in (refresh_command, retry_command):
        if command not in commands:
            commands.append(command)
    return _safe_metadata(
        next_command=commands[0],
        recovery_commands=commands,
        retry_requires_audit_refresh=True,
        retry_requires_storage_repair=requires_setup_check,
        authorizes_retry=False,
        **extra,
    )


def _runtime_trace_missing_metadata_guidance() -> str:
    return (
        "Send a Jarvis command first, then run `runtime trace receipt` without an ID "
        "to load the latest available trace."
    )


def _short(value: Any, limit: int = 220) -> str:
    text = _safe_text(value).strip()
    text = " ".join(text.split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _short_raw(value: Any, limit: int = 80) -> str:
    text = _safe_text(value).strip()
    text = " ".join(text.split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    text = SECRET_VALUE_RE.sub("<redacted-secret>", text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _planner_trace_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    ignored = value.get("model_planner_ignored_unknown_tools")
    ignored_items = ignored if isinstance(ignored, list) else []
    ignored_tools = [_short(item, limit=80) for item in ignored_items if _short(item, limit=80)][:8]
    planner_provider = _short(value.get("model_planner_provider"), limit=32) or "ollama"
    usage_receipt = safe_model_usage_receipt(
        {
            "model_usage_available": value.get("model_planner_usage_available"),
            "model_input_tokens": value.get("model_planner_input_tokens"),
            "model_cached_input_tokens": value.get("model_planner_cached_input_tokens"),
            "model_cached_input_tokens_available": value.get(
                "model_planner_cached_input_tokens_available"
            ),
            "model_output_tokens": value.get("model_planner_output_tokens"),
            "model_reasoning_tokens": value.get("model_planner_reasoning_tokens"),
            "model_reasoning_tokens_available": value.get(
                "model_planner_reasoning_tokens_available"
            ),
            "model_total_tokens": value.get("model_planner_total_tokens"),
        },
        planner_provider,
    )
    return {
        "planner_type": _short(value.get("planner_type"), limit=80),
        "model_planner_attempted": _metadata_bool(value.get("model_planner_attempted")),
        "model_planner_state": _short(value.get("model_planner_state"), limit=80),
        "model_planner_used": _metadata_bool(value.get("model_planner_used")),
        "model_planner_fell_back": _metadata_bool(value.get("model_planner_fell_back")),
        "model_planner_fallback_reason": _short(value.get("model_planner_fallback_reason"), limit=120),
        "model_planner_fallback_detail": _short(value.get("model_planner_fallback_detail"), limit=180),
        "model_planner_recovery_hint": _short(value.get("model_planner_recovery_hint"), limit=220),
        "model_planner_exception_type": _short(value.get("model_planner_exception_type"), limit=80),
        "model_planner_model": _short(value.get("model_planner_model"), limit=120),
        "model_planner_provider": planner_provider,
        "model_planner_timeout_seconds": value.get("model_planner_timeout_seconds"),
        "model_planner_usage_available": usage_receipt["model_usage_available"],
        "model_planner_usage_consistent": usage_receipt["model_usage_consistent"],
        "model_planner_input_tokens": usage_receipt["model_input_tokens"],
        "model_planner_cached_input_tokens": usage_receipt["model_cached_input_tokens"],
        "model_planner_cached_input_tokens_available": usage_receipt[
            "model_cached_input_tokens_available"
        ],
        "model_planner_output_tokens": usage_receipt["model_output_tokens"],
        "model_planner_reasoning_tokens": usage_receipt["model_reasoning_tokens"],
        "model_planner_reasoning_tokens_available": usage_receipt[
            "model_reasoning_tokens_available"
        ],
        "model_planner_total_tokens": usage_receipt["model_total_tokens"],
        "model_planner_usage_content_recorded": False,
        "model_planner_action_count": _metadata_int(value.get("model_planner_action_count")),
        "model_planner_ignored_unknown_tools": ignored_tools,
        "authorizes_execution": _metadata_bool(value.get("authorizes_execution")),
        "authorizes_completion_claim": _metadata_bool(value.get("authorizes_completion_claim")),
        "approval_granted": _metadata_bool(value.get("approval_granted")),
    }


def _first_present(args: dict[str, Any], names: tuple[str, ...], default: Any = "") -> Any:
    for name in names:
        if name in args and args[name] is not None:
            return args[name]
    return default


def _has_explicit_value(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and value.strip() == "")


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


def _checkpoint_recovery_execute_receipt_source_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if any(str(key).startswith("checkpoint_recovery_execute_handoff") for key in metadata):
        return metadata
    tool_results = metadata.get("tool_results")
    if isinstance(tool_results, list):
        for result in tool_results:
            if not isinstance(result, dict):
                continue
            result_metadata = result.get("metadata")
            if not isinstance(result_metadata, dict):
                continue
            if any(str(key).startswith("checkpoint_recovery_execute_handoff") for key in result_metadata):
                return result_metadata
    return {}


def _checkpoint_recovery_execute_receipt_handoff(metadata: dict[str, Any]) -> dict[str, Any]:
    stored_handoff = metadata.get("checkpoint_recovery_execute_handoff")
    if not isinstance(stored_handoff, dict) or not stored_handoff:
        return {}
    if not _checkpoint_recovery_execute_handoff_ready(metadata):
        return {}
    return dict(stored_handoff)


def _checkpoint_recovery_execute_receipt_invalid_reason(metadata: dict[str, Any]) -> str:
    stored_handoff = metadata.get("checkpoint_recovery_execute_handoff")
    if isinstance(stored_handoff, dict) and stored_handoff:
        if _checkpoint_recovery_execute_handoff_ready(metadata):
            return ""
        return "invalid_or_stale_checkpoint_recovery_execute_handoff"
    if metadata.get("checkpoint_recovery_execute_handoff_ready") is True:
        handoff_token_present = metadata.get("checkpoint_recovery_execute_handoff_token_present") is True
        handoff_token = _safe_text(metadata.get("checkpoint_recovery_execute_handoff_token_sha256"))
        boundary_token_present = (
            metadata.get(
                "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_token_present"
            )
            is True
        )
        boundary_token = _safe_text(
            metadata.get(
                "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_token_sha256"
            )
        )
        if (
            not handoff_token_present
            or not _looks_like_sha256(handoff_token)
            or not boundary_token_present
            or not _looks_like_sha256(boundary_token)
        ):
            return "invalid_or_stale_checkpoint_recovery_execute_handoff"
    return ""


def _ids_from_text(value: Any, markers: tuple[str, ...]) -> list[int]:
    text = _safe_text(value)
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


def _preferred_after_action_target(rows: list[Any]) -> Any | None:
    non_meta_rows = [row for row in rows if str(row["tool_name"]) not in AFTER_ACTION_META_TOOLS]
    for row in non_meta_rows:
        ok = bool(row["ok"])
        approved = bool(row["approved"])
        approval_id = row["approval_id"] if "approval_id" in row.keys() else None
        if not ok or (_risk_needs_approval(row["risk"]) and (not approved or approval_id is None)):
            return row
    return non_meta_rows[0] if non_meta_rows else (rows[0] if rows else None)


def _approval_proof_chain_commands(approval_id: int) -> list[str]:
    return [
        f"approval readiness {approval_id}",
        f"approval packet {approval_id}",
        f"approve approval {approval_id}",
        f"approval chain proof {approval_id}",
        f"verification receipt <approved run id from approval chain proof {approval_id}>",
    ]


def _extend_unique(commands: list[str], candidates: list[str]) -> None:
    for command in candidates:
        if command not in commands:
            commands.append(command)


def _ordered_commands(commands: list[str]) -> list[str]:
    ordered: list[str] = []
    _extend_unique(ordered, [command for command in commands if str(command).strip()])
    return ordered


def _actionable_learning_commands(commands: list[str], target_run_id: int | None, missing: list[str]) -> list[str]:
    closure_command = f"execution learning closure {target_run_id}" if target_run_id is not None else "execution learning closure"
    learning_command = f"after-action learning packet {target_run_id}" if target_run_id is not None else "after-action learning packet"
    actionable: list[str] = []
    for command in commands:
        if command == closure_command and "target_after_action_learning_packet" in missing:
            _extend_unique(actionable, [learning_command])
        elif command != closure_command:
            _extend_unique(actionable, [command])
    if commands:
        _extend_unique(actionable, [closure_command])
    return actionable


def _safe_tool_run_payloads(rows: list[Any]) -> tuple[list[dict[str, Any]], int]:
    payloads: list[dict[str, Any]] = []
    unreadable = 0
    for row in rows:
        try:
            keys = set(row.keys())
            run_id = int(row["id"])
            tool_name = str(row["tool_name"])
            risk = str(row["risk"])
            ok = bool(row["ok"])
            approved = bool(row["approved"])
            approval_id = row["approval_id"] if "approval_id" in keys else None
            output = row["output"]
            created_at = row["created_at"]
        except Exception:
            unreadable += 1
            continue
        payloads.append(
            {
                "id": run_id,
                "tool_name": tool_name,
                "risk": risk,
                "ok": ok,
                "approved": approved,
                "approval_id": approval_id,
                "output": output,
                "created_at": created_at,
                "metadata": _row_metadata(row),
            }
        )
    return payloads, unreadable


def _safe_message_payloads(rows: list[Any]) -> tuple[list[dict[str, Any]], int]:
    payloads: list[dict[str, Any]] = []
    unreadable = 0
    for row in rows:
        try:
            message_id = int(row["id"])
            role = str(row["role"])
            metadata_raw = row["metadata"] or "{}"
            session_id = str(row["session_id"])
            created_at = row["created_at"]
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
                "role": role,
                "metadata": metadata if isinstance(metadata, dict) else {},
                "session_id": session_id,
                "created_at": created_at,
            }
        )
    return payloads, unreadable


def _audit_input_failure(
    tool_name: str,
    message: str,
    metadata: dict[str, Any],
    *,
    commands: tuple[str, ...] = (),
    storage_recovery: bool = False,
) -> ToolResult:
    action = (
        LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION
        if storage_recovery
        else LOCAL_READ_INPUT_RECOVERY_ACTION
    )
    output = f"{message} {action}"
    return ToolResult(
        tool_name,
        False,
        output,
        declare_retryable_local_read_failure(
            metadata,
            output=output,
            action=action,
            commands=commands,
        ),
    )


def _audit_not_found_failure(
    tool_name: str,
    message: str,
    metadata: dict[str, Any],
) -> ToolResult:
    output = f"{message} {RESOURCE_NOT_FOUND_RECOVERY_ACTION}"
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


def _recent_runs_next_diagnostic(status_counts: Counter[str], pending_approval_count: int = 0) -> tuple[str, str]:
    if status_counts.get("failed", 0) > 0:
        return "execution health report", "summarizes failed runs and the recovery path"
    if status_counts.get("approval_held", 0) > 0:
        if pending_approval_count > 0:
            return "pending approvals", "reviews approval-held actions without treating them as tool failures"
        return "execution health report", "reviews historical approval-held runs and recovery state"
    return "", ""


def make_audit_tools(store: MemoryStore):
    def _run_approval_evidence(
        row: Any,
        cache: dict[int, Any],
    ) -> _RunApprovalEvidence:
        """Classify approval evidence for this exact run, failing closed on malformed risk."""
        try:
            keys = set(row.keys())
            risk_needs_approval = _risk_needs_approval(row["risk"])
            metadata = row.get("metadata", {}) if isinstance(row, dict) else _row_metadata(row)
            pre_invocation_unknown_tool = (
                _normalized_metadata_token(metadata.get("failure_kind")) == "unknown_tool"
                and metadata.get("executed_handler") is False
            )
            invoked = not _is_approval_hold_metadata(metadata) and not pre_invocation_unknown_tool
            if not risk_needs_approval or not invoked:
                return _RunApprovalEvidence(
                    risk_needs_approval=risk_needs_approval,
                    approval_linked=True,
                    successful_proof=True,
                    approval_problem=False,
                )
            if not bool(row["approved"]) or "approval_id" not in keys:
                return _RunApprovalEvidence(True, False, False, True)
            approval_id = row["approval_id"]
            run_id = row["id"]
            if (
                isinstance(approval_id, bool)
                or not isinstance(approval_id, int)
                or approval_id <= 0
                or isinstance(run_id, bool)
                or not isinstance(run_id, int)
                or run_id <= 0
            ):
                return _RunApprovalEvidence(True, False, False, True)
        except Exception:
            return _RunApprovalEvidence(True, False, False, True)

        classified = cache.get(approval_id)
        if classified is None:
            try:
                classified = store.classify_approval_execution_evidence(approval_id)
            except Exception:
                return _RunApprovalEvidence(True, False, False, True)
            cache[approval_id] = classified

        linked_run_ids = tuple(int(item["id"]) for item in classified.linked_runs)
        matching_run_ids = tuple(int(item["id"]) for item in classified.matching_runs)
        successful_run_ids = tuple(int(item["id"]) for item in classified.successful_runs)
        failed_run_ids = tuple(int(item["id"]) for item in classified.failed_runs)
        exact_row_bound = linked_run_ids == (run_id,) and matching_run_ids == (run_id,)
        successful_proof = (
            bool(row["ok"])
            and exact_row_bound
            and classified.verdict == "APPROVAL_CHAIN_PROVEN"
            and classified.valid_execution_proof is True
            and successful_run_ids == (run_id,)
        )
        exact_failed = (
            not bool(row["ok"])
            and exact_row_bound
            and classified.verdict == "APPROVAL_EXECUTION_FAILED"
            and failed_run_ids == (run_id,)
        )
        exact_unknown = (
            exact_row_bound
            and classified.outcome_unknown is True
        )
        approval_linked = successful_proof or exact_failed or exact_unknown
        return _RunApprovalEvidence(
            risk_needs_approval=True,
            approval_linked=approval_linked,
            successful_proof=successful_proof,
            approval_problem=not approval_linked,
            outcome_unknown=exact_unknown,
            verdict=str(classified.verdict),
        )

    def recent_tool_runs(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 20)
        raw_rows = store.recent_tool_runs(limit)
        rows, unreadable_rows = _safe_tool_run_payloads(raw_rows)
        approval_proof_cache: dict[int, Any] = {}
        if not rows and unreadable_rows == 0:
            recent_tool_runs_handoff = {
                "source": "recent_tool_runs",
                "count": 0,
                "limit": limit,
                "approved_linked_runs": 0,
                "audit_toolsets": {},
                "risk_counts": {},
                "status_counts": {"ok": 0, "failed": 0, "approval_held": 0},
                "rows": [],
                "readable_tool_run_rows": 0,
                "unreadable_tool_run_rows": 0,
                "review_only": True,
                "draft_only": True,
                "loads_without_execution": True,
                "authorizes_execution": False,
                "authorizes_completion_claim": False,
                "approval_granted": False,
                "calls_model": False,
                "executes_tools": False,
                "writes_files": False,
                "reads_personal_data": False,
                "external_side_effect": False,
                "controls_computer": False,
                "queues_approval": False,
            }
            return ToolResult(
                "recent_tool_runs",
                True,
                "No tool runs logged yet.",
                _safe_metadata(
                    count=0,
                    limit=limit,
                    approved_linked_runs=0,
                    audit_toolsets={},
                    risk_counts={},
                    status_counts={"ok": 0, "failed": 0, "approval_held": 0},
                    readable_tool_run_rows=0,
                    unreadable_tool_run_rows=0,
                    recent_tool_runs_handoff=recent_tool_runs_handoff,
                    recent_tool_runs_review_only=True,
                    recent_tool_runs_draft_only=True,
                    recent_tool_runs_loads_without_execution=True,
                    recent_tool_runs_authorizes_execution=False,
                    recent_tool_runs_authorizes_completion_claim=False,
                    recent_tool_runs_approval_granted=False,
                ),
            )
        if not rows:
            next_diagnostic_command = "jarvis doctor"
            next_diagnostic_reason = "checks local storage and audit readability without exposing hidden rows"
            recent_tool_runs_handoff = {
                "source": "recent_tool_runs",
                "count": 0,
                "limit": limit,
                "approved_linked_runs": 0,
                "audit_toolsets": {},
                "risk_counts": {},
                "status_counts": {"ok": 0, "failed": 0, "approval_held": 0},
                "rows": [],
                "readable_tool_run_rows": 0,
                "unreadable_tool_run_rows": unreadable_rows,
                "review_only": True,
                "draft_only": True,
                "loads_without_execution": True,
                "authorizes_execution": False,
                "authorizes_completion_claim": False,
                "approval_granted": False,
                "calls_model": False,
                "executes_tools": False,
                "writes_files": False,
                "reads_personal_data": False,
                "external_side_effect": False,
                "controls_computer": False,
                "queues_approval": False,
                "next_diagnostic_command": next_diagnostic_command,
                "next_diagnostic_reason": next_diagnostic_reason,
            }
            return ToolResult(
                "recent_tool_runs",
                True,
                (
                    f"No readable tool runs. {unreadable_rows} unreadable tool run row(s) hidden for safety.\n"
                    f"Next diagnostic: `{next_diagnostic_command}` ({next_diagnostic_reason})."
                ),
                _safe_metadata(
                    count=0,
                    limit=limit,
                    approved_linked_runs=0,
                    audit_toolsets={},
                    risk_counts={},
                    status_counts={"ok": 0, "failed": 0, "approval_held": 0},
                    readable_tool_run_rows=0,
                    unreadable_tool_run_rows=unreadable_rows,
                    recent_tool_run_rows=[],
                    recent_tool_runs_handoff=recent_tool_runs_handoff,
                    recent_tool_runs_review_only=True,
                    recent_tool_runs_draft_only=True,
                    recent_tool_runs_loads_without_execution=True,
                    recent_tool_runs_authorizes_execution=False,
                    recent_tool_runs_authorizes_completion_claim=False,
                    recent_tool_runs_approval_granted=False,
                    next_diagnostic_command=next_diagnostic_command,
                    next_diagnostic_reason=next_diagnostic_reason,
                ),
            )
        lines = []
        audit_toolsets: Counter[str] = Counter()
        risk_counts: Counter[str] = Counter()
        status_counts: Counter[str] = Counter()
        row_summaries: list[dict[str, Any]] = []
        for row in rows:
            audit_metadata = row["metadata"]
            approval_held = (not row["ok"]) and _is_approval_hold_metadata(audit_metadata)
            approval_evidence = _run_approval_evidence(row, approval_proof_cache)
            approval_proven = approval_evidence.approval_linked
            status = (
                "ok"
                if row["ok"] and approval_evidence.successful_proof
                else ("approval_held" if approval_held else "failed")
            )
            status_label = "approval held" if status == "approval_held" else status
            status_counts[status] += 1
            effective_approved = bool(row["approved"]) and approval_proven
            approved = ", approved" if effective_approved else ""
            approval_id = row["approval_id"]
            approval_link = f", approval #{approval_id}" if approval_id else ""
            failure_kind = _recent_run_failure_kind(audit_metadata)
            failure_text = f", {failure_kind}" if failure_kind else ""
            toolset = _short(audit_metadata.get("toolset"), limit=80)
            toolset_text = f", toolset {toolset}" if toolset else ""
            if toolset:
                audit_toolsets[toolset] += 1
            risk = str(row["risk"])
            risk_counts[risk] += 1
            snippet = _short(row["output"], limit=140)
            lines.append(f"- #{row['id']} {row['tool_name']} [{row['risk']}, {status_label}{approved}{approval_link}{failure_text}{toolset_text}] {row['created_at']}: {snippet}")
            row_summaries.append(
                {
                    "run_id": int(row["id"]),
                    "tool_name": str(row["tool_name"]),
                    "risk": risk,
                    "ok": bool(row["ok"]),
                    "status": status,
                    "approved": effective_approved,
                    "approval_id": approval_id,
                    "failure_kind": failure_kind or None,
                    "toolset": toolset or None,
                }
            )
        for status_key in ("ok", "failed", "approval_held"):
            status_counts.setdefault(status_key, 0)
        if unreadable_rows:
            lines.append(f"- {unreadable_rows} unreadable tool run row(s) hidden for safety.")
        try:
            pending_approval_count = len(store.list_pending_approvals(limit=5))
        except Exception:
            pending_approval_count = 0
        next_diagnostic_command, next_diagnostic_reason = _recent_runs_next_diagnostic(
            status_counts,
            pending_approval_count,
        )
        if next_diagnostic_command:
            lines.append(
                f"- Next diagnostic: `{next_diagnostic_command}` ({next_diagnostic_reason})."
            )
        approved_linked = sum(
            1
            for row in rows
            if row["approved"]
            and row["approval_id"] is not None
            and _run_approval_evidence(row, approval_proof_cache).approval_linked
        )
        recent_tool_runs_handoff = {
            "source": "recent_tool_runs",
            "count": len(rows),
            "limit": limit,
            "approved_linked_runs": approved_linked,
            "audit_toolsets": dict(audit_toolsets),
            "risk_counts": dict(risk_counts),
            "status_counts": dict(status_counts),
            "rows": row_summaries,
            "readable_tool_run_rows": len(rows),
            "unreadable_tool_run_rows": unreadable_rows,
            "review_only": True,
            "draft_only": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "calls_model": False,
            "executes_tools": False,
            "writes_files": False,
            "reads_personal_data": False,
            "external_side_effect": False,
            "controls_computer": False,
            "queues_approval": False,
            "next_diagnostic_command": next_diagnostic_command,
            "next_diagnostic_reason": next_diagnostic_reason,
            "pending_approval_count": pending_approval_count,
        }
        return ToolResult(
            "recent_tool_runs",
            True,
            "Recent tool runs:\n" + "\n".join(lines),
            _safe_metadata(
                count=len(rows),
                limit=limit,
                approved_linked_runs=approved_linked,
                audit_toolsets=dict(audit_toolsets),
                risk_counts=dict(risk_counts),
                status_counts=dict(status_counts),
                recent_tool_run_rows=row_summaries,
                readable_tool_run_rows=len(rows),
                unreadable_tool_run_rows=unreadable_rows,
                recent_tool_runs_handoff=recent_tool_runs_handoff,
                recent_tool_runs_review_only=True,
                recent_tool_runs_draft_only=True,
                recent_tool_runs_loads_without_execution=True,
                recent_tool_runs_authorizes_execution=False,
                recent_tool_runs_authorizes_completion_claim=False,
                recent_tool_runs_approval_granted=False,
                next_diagnostic_command=next_diagnostic_command,
                next_diagnostic_reason=next_diagnostic_reason,
                pending_approval_count=pending_approval_count,
            ),
        )

    def verification_receipt(args: dict[str, Any]) -> ToolResult:
        run_id_raw = _first_present(args, ("run_id", "id", "tool_run_id"), "latest")
        expectation = _short(args.get("expectation") or args.get("expected") or args.get("target"), limit=260)
        raw_rows = store.recent_tool_runs(limit=MAX_AUDIT_LIMIT)
        rows, unreadable_rows = _safe_tool_run_payloads(raw_rows)
        approval_proof_cache: dict[int, Any] = {}
        if not rows:
            verification_receipt_handoff = {
                "source": "verification_receipt",
                "found": False,
                "verdict": "UNREADABLE_RUNS" if unreadable_rows else "UNVERIFIED",
                "expectation": expectation,
                "target": {
                    "run_id": None,
                    "tool_name": "",
                    "risk": "",
                    "ok": False,
                    "approved": False,
                    "approval_id": None,
                    "toolset": None,
                    "failure_kind": None,
                    "planned_arg_keys": [],
                },
                "approval_evidence_required": False,
                "output_chars": 0,
                "readable_tool_run_rows": 0,
                "unreadable_tool_run_rows": unreadable_rows,
                "review_only": True,
                "draft_only": True,
                "loads_without_execution": True,
                "authorizes_execution": False,
                "authorizes_completion_claim": False,
                "approval_granted": False,
                "calls_model": False,
                "executes_tools": False,
                "writes_files": False,
                "reads_personal_data": False,
                "external_side_effect": False,
                "controls_computer": False,
                "queues_approval": False,
            }
            return ToolResult(
                "verification_receipt",
                True,
                "\n".join(
                    [
                        "Jarvis verification receipt:",
                        (
                            f"No readable tool runs. {unreadable_rows} unreadable tool run row(s) hidden for safety."
                            if unreadable_rows
                            else "No tool runs are logged yet."
                        ),
                        "",
                        f"Verdict: {'UNREADABLE_RUNS' if unreadable_rows else 'UNVERIFIED'}",
                        "Next step: run a read-only or local-safe command first, then ask for `verification receipt`.",
                    ]
                ),
                _safe_metadata(
                    count=0,
                    run_id=None,
                    found=False,
                    verdict="UNREADABLE_RUNS" if unreadable_rows else "UNVERIFIED",
                    expectation=expectation,
                    readable_tool_run_rows=0,
                    unreadable_tool_run_rows=unreadable_rows,
                    verification_receipt_handoff=verification_receipt_handoff,
                    verification_receipt_review_only=True,
                    verification_receipt_draft_only=True,
                    verification_receipt_loads_without_execution=True,
                    verification_receipt_authorizes_execution=False,
                    verification_receipt_authorizes_completion_claim=False,
                    verification_receipt_approval_granted=False,
                ),
            )

        target = None
        if isinstance(run_id_raw, str) and run_id_raw.strip().lower() in {"", "latest", "last", "newest", "current"}:
            target = rows[0]
        else:
            try:
                run_id = int(run_id_raw)
            except (TypeError, ValueError):
                return _audit_input_failure(
                    "verification_receipt",
                    "Tool run id must be a number or `latest`.",
                    _safe_metadata(run_id=_short_raw(run_id_raw), raw_run_id=_short_raw(run_id_raw), found=False, verdict="INVALID_ID", expectation=expectation),
                )
            if run_id <= 0:
                return _audit_input_failure(
                    "verification_receipt",
                    "Tool run id must be a positive number or `latest`.",
                    _safe_metadata(run_id=_short_raw(run_id_raw), raw_run_id=_short_raw(run_id_raw), found=False, verdict="INVALID_ID", expectation=expectation),
                )
            raw_target = store.get_tool_run(run_id)
            if raw_target is not None:
                target_payloads, unreadable_target_rows = _safe_tool_run_payloads([raw_target])
                if target_payloads:
                    target = target_payloads[0]
                elif unreadable_target_rows:
                    return _audit_input_failure(
                        "verification_receipt",
                        f"Tool run #{run_id} could not be read from the audit log. {unreadable_target_rows} "
                        "unreadable tool run row(s) hidden for safety. Run `setup check`, then "
                        "`recent tool runs`; retry `verification receipt latest` after audit storage is readable.",
                        _lookup_recovery_metadata(
                            "recent tool runs",
                            "verification receipt latest",
                            requires_setup_check=True,
                            run_id=_short_raw(run_id_raw),
                            found=False,
                            verdict="UNREADABLE_RUN",
                            expectation=expectation,
                            readable_tool_run_rows=len(rows),
                            unreadable_tool_run_rows=unreadable_rows + unreadable_target_rows,
                        ),
                        commands=("setup check", "recent tool runs", "verification receipt latest"),
                        storage_recovery=True,
                    )

        if target is None:
            safe_run_id = _short_raw(run_id_raw)
            return _audit_not_found_failure(
                "verification_receipt",
                f"Tool run #{safe_run_id} was not found in the audit log. Run `recent tool runs` to "
                "refresh valid IDs, then retry `verification receipt latest` or use one listed run ID.",
                _lookup_recovery_metadata(
                    "recent tool runs",
                    "verification receipt latest",
                    run_id=safe_run_id,
                    found=False,
                    verdict="MISSING_RUN",
                    expectation=expectation,
                ),
            )

        ok = bool(target["ok"])
        risk = str(target["risk"])
        approval_evidence = _run_approval_evidence(target, approval_proof_cache)
        approval_proven = approval_evidence.approval_linked
        approved = bool(target["approved"]) and approval_proven
        approval_id = target["approval_id"]
        audit_metadata = target["metadata"]
        toolset = _short(audit_metadata.get("toolset"), limit=80)
        failure_kind = _short(audit_metadata.get("failure_kind"), limit=80)
        planned_arg_keys = audit_metadata.get("planned_arg_keys") if isinstance(audit_metadata.get("planned_arg_keys"), list) else []
        checkpoint_recovery_execute_handoff = _checkpoint_recovery_execute_receipt_handoff(audit_metadata)
        checkpoint_recovery_execute_handoff_invalid_reason = (
            _checkpoint_recovery_execute_receipt_invalid_reason(audit_metadata)
        )
        checkpoint_recovery_execute_handoff_mirrors = (
            {
                key: value
                for key, value in audit_metadata.items()
                if key.startswith("checkpoint_recovery_execute_handoff_")
                or key in CHECKPOINT_RECOVERY_EXECUTE_RECEIPT_CONTRACT_KEYS
            }
            if checkpoint_recovery_execute_handoff
            else {}
        )
        risk_needs_approval = approval_evidence.risk_needs_approval
        approval_problem = approval_evidence.approval_problem
        output = _short(target["output"], limit=700)

        if approval_problem:
            verdict = "APPROVAL_EVIDENCE_MISSING"
        elif not ok or approval_evidence.outcome_unknown:
            verdict = "FAILED_OR_BLOCKED"
        elif expectation:
            verdict = "OUTPUT_REVIEW_REQUIRED"
        else:
            verdict = "PASS_WITH_AUDIT_EVIDENCE"

        evidence_lines = [
            f"- tool run: #{target['id']} `{target['tool_name']}`",
            f"- created: {target['created_at']}",
            f"- risk: {risk}",
            f"- result: {'ok' if ok else 'failed/blocked'}",
            f"- approved: {'yes' if approved else 'no'}",
        ]
        if approval_id:
            evidence_lines.append(f"- approval id: #{approval_id}")
        if expectation:
            evidence_lines.append(f"- expected outcome: {expectation}")
        evidence_lines.append(f"- output evidence: {output or '(empty output)'}")

        checks = [
            "Audit row exists in `tool_runs`.",
            "Tool name, risk level, approval state, timestamp, and output are visible.",
        ]
        if risk_needs_approval:
            checks.append("Risk-gated run must be linked to an approved approval id.")
        if expectation:
            checks.append("Human or verifier should compare the output evidence against the expected outcome.")
        if not ok or approval_evidence.outcome_unknown:
            checks.append("Failed or blocked runs require recovery before Jarvis claims completion.")

        next_steps = []
        if not ok or approval_evidence.outcome_unknown:
            next_steps.extend(
                [
                    f"`recent tool runs` to inspect neighboring failures around run #{target['id']}",
                    "`work block checkpoint` or `checkpoint recovery preview` before retrying",
                ]
            )
        elif approval_problem:
            next_steps.extend(
                [
                    "`approval history` to prove the approval chain",
                    f"`approval readiness {approval_id}` before `approval packet {approval_id}`, then `approval chain proof {approval_id}` if an approval id exists but was not linked",
                ]
            )
        elif expectation:
            next_steps.append("Compare the output evidence with the expected outcome before marking the task done.")
        else:
            next_steps.append("Use this receipt as current audit evidence for a low-risk completed step.")

        verification_receipt_handoff = {
            "source": "verification_receipt",
            "found": True,
            "verdict": verdict,
            "expectation": expectation,
            "target": {
                "run_id": int(target["id"]),
                "tool_name": target["tool_name"],
                "risk": risk,
                "ok": ok,
                "approved": approved,
                "approval_id": approval_id,
                "toolset": toolset or None,
                "failure_kind": failure_kind or None,
                "planned_arg_keys": planned_arg_keys,
            },
            "approval_evidence_required": risk_needs_approval,
            "approval_problem": approval_problem,
            "output_chars": len(_safe_text(target["output"])),
            "readable_tool_run_rows": len(rows),
            "unreadable_tool_run_rows": unreadable_rows,
            "checkpoint_recovery_execute_handoff": checkpoint_recovery_execute_handoff,
            "has_checkpoint_recovery_execute_handoff": bool(checkpoint_recovery_execute_handoff),
            "checkpoint_recovery_execute_handoff_invalid_reason": checkpoint_recovery_execute_handoff_invalid_reason,
            "review_only": True,
            "draft_only": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "calls_model": False,
            "executes_tools": False,
            "writes_files": False,
            "reads_personal_data": False,
            "external_side_effect": False,
            "controls_computer": False,
            "queues_approval": False,
        }

        lines = [
            "Jarvis verification receipt:",
            "This is the after-action proof layer for the harness. It is read-only and does not rerun the tool.",
            "",
            "Evidence:",
            *evidence_lines,
            "",
            "Checks:",
            *[f"- {item}" for item in checks],
            "",
            f"Verdict: {verdict}",
            "",
            "Next step:",
            *[f"- {item}" for item in next_steps],
            "",
            "Boundary:",
            "- This receipt does not call models, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, or queue approvals.",
        ]
        if unreadable_rows:
            lines.insert(8, f"- {unreadable_rows} unreadable tool run row(s) hidden for safety.")

        return ToolResult(
            "verification_receipt",
            True,
            "\n".join(lines),
            _safe_metadata(
                count=len(rows),
                run_id=int(target["id"]),
                found=True,
                tool_name=target["tool_name"],
                risk=risk,
                ok=ok,
                approved=approved,
                approval_id=approval_id,
                expectation=expectation,
                verdict=verdict,
                output_chars=len(_safe_text(target["output"])),
                approval_evidence_required=risk_needs_approval,
                toolset=toolset or None,
                failure_kind=failure_kind or None,
                planned_arg_keys=planned_arg_keys,
                readable_tool_run_rows=len(rows),
                unreadable_tool_run_rows=unreadable_rows,
                checkpoint_recovery_execute_handoff=checkpoint_recovery_execute_handoff,
                has_checkpoint_recovery_execute_handoff=bool(checkpoint_recovery_execute_handoff),
                verification_receipt_checkpoint_recovery_execute_handoff=checkpoint_recovery_execute_handoff,
                verification_receipt_has_checkpoint_recovery_execute_handoff=bool(checkpoint_recovery_execute_handoff),
                checkpoint_recovery_execute_handoff_invalid_reason=(
                    checkpoint_recovery_execute_handoff_invalid_reason
                ),
                **checkpoint_recovery_execute_handoff_mirrors,
                verification_receipt_handoff=verification_receipt_handoff,
                verification_receipt_review_only=True,
                verification_receipt_draft_only=True,
                verification_receipt_loads_without_execution=True,
                verification_receipt_authorizes_execution=False,
                verification_receipt_authorizes_completion_claim=False,
                verification_receipt_approval_granted=False,
            ),
        )

    def runtime_trace_receipt(args: dict[str, Any]) -> ToolResult:
        session_id = _short(args.get("session_id"), limit=80)
        limit = _bounded_int(args.get("limit"), 40)
        message_id_raw = _first_present(args, ("message_id", "id", "trace_id"), "")
        raw_rows = store.recent_messages(limit=limit, session_id=session_id or None)
        rows, unreadable_rows = _safe_message_payloads(raw_rows)
        target = None
        target_metadata: dict[str, Any] = {}

        if _has_explicit_value(message_id_raw):
            try:
                message_id = int(message_id_raw)
            except (TypeError, ValueError):
                return _audit_input_failure(
                    "runtime_trace_receipt",
                    "Runtime trace message id must be a number when supplied.",
                    _safe_metadata(
                        found=False,
                        message_id=_short_raw(message_id_raw),
                        raw_message_id=_short_raw(message_id_raw),
                        session_id=session_id or None,
                        inspected_messages=len(rows),
                        readable_message_rows=len(rows),
                        unreadable_message_rows=unreadable_rows,
                        verdict="INVALID_ID",
                    ),
                )
            if message_id <= 0:
                return _audit_input_failure(
                    "runtime_trace_receipt",
                    "Runtime trace message id must be a positive number when supplied.",
                    _safe_metadata(
                        found=False,
                        message_id=_short_raw(message_id_raw),
                        raw_message_id=_short_raw(message_id_raw),
                        session_id=session_id or None,
                        inspected_messages=len(rows),
                        readable_message_rows=len(rows),
                        unreadable_message_rows=unreadable_rows,
                        verdict="INVALID_ID",
                    ),
                )
            message = store.get_message(message_id)
            if message is None:
                safe_message_id = _short_raw(message_id_raw)
                return _audit_not_found_failure(
                    "runtime_trace_receipt",
                    f"Message #{safe_message_id} was not found in the conversation log. Run "
                    "`runtime trace receipt` without an ID to load the latest available trace.",
                    _lookup_recovery_metadata(
                        "runtime trace receipt",
                        "runtime trace receipt",
                        found=False,
                        message_id=safe_message_id,
                        session_id=session_id or None,
                        inspected_messages=len(rows),
                        readable_message_rows=len(rows),
                        unreadable_message_rows=unreadable_rows,
                        verdict="MISSING_MESSAGE",
                    ),
                )
            message_payloads, unreadable_target_rows = _safe_message_payloads([message])
            if unreadable_target_rows:
                return _audit_input_failure(
                    "runtime_trace_receipt",
                    f"Message #{message_id} could not be read from the conversation log. "
                    f"{unreadable_target_rows} unreadable message row(s) hidden for safety. Run "
                    "`setup check`, then retry `runtime trace receipt` without an ID.",
                    _lookup_recovery_metadata(
                        "runtime trace receipt",
                        "runtime trace receipt",
                        requires_setup_check=True,
                        found=False,
                        message_id=_short_raw(message_id_raw),
                        session_id=session_id or None,
                        inspected_messages=len(rows),
                        readable_message_rows=len(rows),
                        unreadable_message_rows=unreadable_rows + unreadable_target_rows,
                        verdict="UNREADABLE_MESSAGE",
                    ),
                    commands=("setup check", "runtime trace receipt"),
                    storage_recovery=True,
                )
            message = message_payloads[0]
            target_metadata = message["metadata"]
            if message["role"] == "assistant" and isinstance(target_metadata.get("runtime_trace"), dict):
                target = message
        else:
            for row in reversed(rows):
                if row["role"] != "assistant":
                    continue
                metadata = row["metadata"]
                trace = metadata.get("runtime_trace")
                if isinstance(trace, dict):
                    target = row
                    target_metadata = metadata
                    break

        if target is None and _has_explicit_value(message_id_raw):
            safe_message_id = _short_raw(message_id_raw)
            return _audit_input_failure(
                "runtime_trace_receipt",
                (
                    f"Message #{safe_message_id} does not contain assistant runtime trace metadata. "
                    f"{_runtime_trace_missing_metadata_guidance()}"
                ),
                _safe_metadata(
                    found=False,
                    message_id=safe_message_id,
                    session_id=session_id or None,
                    inspected_messages=len(rows),
                    readable_message_rows=len(rows),
                    unreadable_message_rows=unreadable_rows,
                    verdict="MISSING_TRACE",
                    next_command="send a Jarvis command",
                    recovery_commands=[
                        "send a Jarvis command",
                        "runtime trace receipt",
                    ],
                    retry_requires_fresh_runtime_trace=True,
                    authorizes_retry=False,
                ),
            )

        if target is None:
            return _audit_input_failure(
                "runtime_trace_receipt",
                "\n".join(
                    [
                        "Jarvis runtime trace receipt:",
                        "No recent assistant runtime trace was found.",
                        "",
                        "Next step: send a Jarvis command first, then ask for `runtime trace receipt`.",
                    ]
                ),
                _safe_metadata(
                    found=False,
                    session_id=session_id or None,
                    inspected_messages=len(rows),
                    readable_message_rows=len(rows),
                    unreadable_message_rows=unreadable_rows,
                    verdict="MISSING_TRACE",
                ),
                commands=("send a Jarvis command", "runtime trace receipt"),
            )

        trace = target_metadata.get("runtime_trace") or {}
        checkpoint_recovery_execute_metadata = _checkpoint_recovery_execute_receipt_source_metadata(
            target_metadata
        )
        checkpoint_recovery_execute_handoff = _checkpoint_recovery_execute_receipt_handoff(
            checkpoint_recovery_execute_metadata
        )
        checkpoint_recovery_execute_handoff_invalid_reason = (
            _checkpoint_recovery_execute_receipt_invalid_reason(checkpoint_recovery_execute_metadata)
            if checkpoint_recovery_execute_metadata
            else ""
        )
        stages = trace.get("stages") if isinstance(trace.get("stages"), list) else []
        planned_actions = trace.get("planned_actions") if isinstance(trace.get("planned_actions"), list) else []
        tool_results = trace.get("tool_results") if isinstance(trace.get("tool_results"), list) else []
        queued_ids = trace.get("queued_approval_ids") if isinstance(trace.get("queued_approval_ids"), list) else []
        new_approval_ids = trace.get("new_approval_ids") if isinstance(trace.get("new_approval_ids"), list) else []
        reused_approval_ids = trace.get("reused_approval_ids") if isinstance(trace.get("reused_approval_ids"), list) else []
        referenced_ids = trace.get("referenced_approval_ids") if isinstance(trace.get("referenced_approval_ids"), list) else []
        approved_reruns = _metadata_int(trace.get("approved_reruns"))
        approved_rerun_run_ids = trace.get("approved_rerun_run_ids") if isinstance(trace.get("approved_rerun_run_ids"), list) else []
        approved_rerun_approval_ids = trace.get("approved_rerun_approval_ids") if isinstance(trace.get("approved_rerun_approval_ids"), list) else []
        risk_levels = trace.get("risk_levels") if isinstance(trace.get("risk_levels"), list) else []
        result_toolsets = Counter(
            _short(result.get("toolset"), limit=80)
            for result in tool_results
            if isinstance(result, dict) and _short(result.get("toolset"), limit=80)
        )
        approval_queue_before = _metadata_int(trace.get("approval_queue_before"))
        approval_queue_after = _metadata_int(trace.get("approval_queue_after"), approval_queue_before)
        approval_queue_delta = _metadata_int(trace.get("approval_queue_delta"))
        route = _short(trace.get("route"), limit=80) or "unknown"
        planner_notes = _short(trace.get("planner_notes"), limit=120)
        planner_metadata = _planner_trace_metadata(trace.get("planner_metadata"))
        verified = _metadata_bool(trace.get("verified"))
        approval_required = _metadata_bool(trace.get("approval_required"))
        ran_tool_handlers = _metadata_bool(trace.get("ran_tool_handlers"))
        verdict = "VERIFIED" if verified else "HELD_OR_FAILED"
        if approval_required and not ran_tool_handlers:
            verdict = "HELD_FOR_APPROVAL"
        elif approval_required and queued_ids:
            verdict = "APPROVAL_QUEUED"

        lines = [
            "Jarvis runtime trace receipt:",
            "This is the stage-by-stage receipt for the latest Jarvis turn with runtime metadata.",
            "",
            "Trace identity:",
            f"- message id: #{target['id']}",
            f"- session id: {trace.get('session_id') or target['session_id']}",
            f"- created: {target['created_at']}",
            f"- route: {route}",
            f"- goal: {_short(trace.get('goal'), limit=260)}",
            f"- verified: {'yes' if verified else 'no'}",
            f"- verdict: {verdict}",
            "",
            "Lifecycle stages:",
        ]
        if unreadable_rows:
            lines.insert(11, f"- {unreadable_rows} unreadable message row(s) hidden for safety.")
        if stages:
            for stage in stages:
                if not isinstance(stage, dict):
                    continue
                detail = _short(stage.get("detail"), limit=180)
                lines.append(f"- {stage.get('stage', 'unknown')}: {stage.get('status', 'unknown')} - {detail}")
        else:
            lines.append("- no stages found in trace metadata")

        lines.extend(
            [
                "",
                "Planner diagnostics:",
                f"- planner notes: {planner_notes or 'none'}",
            ]
        )
        if planner_metadata.get("model_planner_attempted"):
            lines.extend(
                [
                    f"- model planner state: {planner_metadata.get('model_planner_state') or 'unknown'}",
                    f"- model planner used: {'yes' if planner_metadata.get('model_planner_used') else 'no'}",
                    f"- model planner fallback: {'yes' if planner_metadata.get('model_planner_fell_back') else 'no'}",
                    f"- model planner fallback reason: {planner_metadata.get('model_planner_fallback_reason') or 'none'}",
                    f"- model planner model: {planner_metadata.get('model_planner_model') or 'unknown'}",
                    f"- model planner provider: {planner_metadata.get('model_planner_provider') or 'unknown'}",
                    f"- model planner timeout: {planner_metadata.get('model_planner_timeout_seconds') or 'unknown'}s",
                    f"- ignored model tools: {', '.join(planner_metadata.get('model_planner_ignored_unknown_tools') or []) if planner_metadata.get('model_planner_ignored_unknown_tools') else 'none'}",
                ]
            )
            if planner_metadata.get("model_planner_fallback_detail"):
                lines.append(f"- model planner detail: {planner_metadata.get('model_planner_fallback_detail')}")
            if planner_metadata.get("model_planner_recovery_hint"):
                lines.append(f"- model planner recovery: {planner_metadata.get('model_planner_recovery_hint')}")
            if planner_metadata.get("model_planner_usage_available"):
                lines.extend(
                    [
                        "- model planner token usage: "
                        f"input {planner_metadata.get('model_planner_input_tokens', 0)}, "
                        f"output {planner_metadata.get('model_planner_output_tokens', 0)}, "
                        f"total {planner_metadata.get('model_planner_total_tokens', 0)}",
                        "- model planner cached input tokens: "
                        + (
                            str(planner_metadata.get("model_planner_cached_input_tokens", 0))
                            if planner_metadata.get("model_planner_cached_input_tokens_available")
                            else "unavailable"
                        ),
                        "- model planner reasoning tokens: "
                        + (
                            str(planner_metadata.get("model_planner_reasoning_tokens", 0))
                            if planner_metadata.get("model_planner_reasoning_tokens_available")
                            else "unavailable"
                        ),
                        "- model planner usage receipt contains counts only; no request or response content",
                    ]
                )
            else:
                lines.append("- model planner token usage: unavailable")
        else:
            lines.append("- model planner: not attempted or no diagnostic metadata in trace")

        lines.extend(["", "Planned actions:"])
        if planned_actions:
            for index, action in enumerate(planned_actions, start=1):
                if not isinstance(action, dict):
                    continue
                lines.append(
                    f"- {index}. {action.get('tool_name', 'unknown')} [{action.get('toolset', 'unknown')}, {action.get('risk', 'UNKNOWN')}]"
                )
                lines.append(f"  reason: {_short(action.get('reason'), limit=180)}")
        else:
            lines.append("- none")

        lines.extend(["", "Tool result evidence:"])
        if tool_results:
            for result in tool_results:
                if not isinstance(result, dict):
                    continue
                approval_id = result.get("approval_id")
                approval_text = f", approval #{approval_id}" if approval_id else ""
                status = "ok" if result.get("ok") else "failed/held"
                toolset = _short(result.get("toolset"), limit=80)
                risk_level = _short(result.get("risk_level"), limit=80)
                context_parts = [part for part in [toolset, risk_level] if part]
                context_text = f" [{', '.join(context_parts)}]" if context_parts else ""
                lines.append(f"- {result.get('tool_name', 'unknown')}: {status}{approval_text}{context_text}")
                preview = _short(result.get("output_preview"), limit=220)
                if preview:
                    lines.append(f"  output: {preview}")
                if "execution_started" in result:
                    lines.append(
                        f"  execution started: {'yes' if result.get('execution_started') else 'no'}; "
                        f"handler executed: {'yes' if result.get('executed_handler') else 'no'}; "
                        f"direct process reaped: {'yes' if result.get('process_reaped') else 'no'}"
                    )
                if result.get("output_capture_streaming"):
                    lines.append(
                        "  output capture: bounded streaming; "
                        f"cap {result.get('capture_limit_bytes_per_stream', 'unknown')} bytes/stream; "
                        f"observed stdout {result.get('stdout_bytes_seen', 'unknown')} bytes, "
                        f"stderr {result.get('stderr_bytes_seen', 'unknown')} bytes; "
                        f"truncated {'yes' if result.get('output_truncated') else 'no'}"
                    )
                if result.get("process_group_cleanup_attempted"):
                    lines.append(
                        "  cleanup: original process group stop attempted; "
                        f"SIGKILL escalation {'yes' if result.get('termination_escalated') else 'no'}; "
                        "independently detached descendants are not verified"
                    )
                approved_rerun = result.get("approved_rerun_result")
                if isinstance(approved_rerun, dict):
                    rerun_status = "ok" if approved_rerun.get("ok") else "failed/held"
                    rerun_approval_id = approved_rerun.get("approved_approval_id") or approval_id
                    rerun_approval_text = f", approval #{rerun_approval_id}" if rerun_approval_id else ""
                    arg_keys = approved_rerun.get("rerun_arg_keys")
                    arg_text = ", ".join(str(item) for item in arg_keys) if isinstance(arg_keys, list) and arg_keys else "none"
                    lines.append(
                        f"  approved rerun: {approved_rerun.get('tool_name', 'unknown')} {rerun_status}{rerun_approval_text}"
                    )
                    lines.append(f"  approved rerun exact: {'yes' if approved_rerun.get('approval_rerun_exact') else 'no'}")
                    lines.append(f"  approved rerun arg keys: {arg_text}")
                    rerun_preview = _short(approved_rerun.get("output_preview"), limit=220)
                    if rerun_preview:
                        lines.append(f"  approved rerun output: {rerun_preview}")
                    if "execution_started" in approved_rerun:
                        lines.append(
                            "  approved rerun execution started: "
                            f"{'yes' if approved_rerun.get('execution_started') else 'no'}; "
                            f"handler executed: {'yes' if approved_rerun.get('executed_handler') else 'no'}; "
                            f"direct process reaped: {'yes' if approved_rerun.get('process_reaped') else 'no'}"
                        )
                    if approved_rerun.get("output_capture_streaming"):
                        lines.append(
                            "  approved rerun output capture: bounded streaming; "
                            f"cap {approved_rerun.get('capture_limit_bytes_per_stream', 'unknown')} bytes/stream; "
                            f"observed stdout {approved_rerun.get('stdout_bytes_seen', 'unknown')} bytes, "
                            f"stderr {approved_rerun.get('stderr_bytes_seen', 'unknown')} bytes; "
                            f"truncated {'yes' if approved_rerun.get('output_truncated') else 'no'}"
                        )
                    if approved_rerun.get("process_group_cleanup_attempted"):
                        lines.append(
                            "  approved rerun cleanup: original process group stop attempted; "
                            f"SIGKILL escalation {'yes' if approved_rerun.get('termination_escalated') else 'no'}; "
                            "independently detached descendants are not verified"
                        )
        else:
            lines.append("- none")

        lines.extend(
            [
                "",
                "Safety and approval state:",
                f"- risk levels: {', '.join(str(item) for item in risk_levels) if risk_levels else 'none'}",
                f"- approval required: {'yes' if approval_required else 'no'}",
                f"- approval queue before: {approval_queue_before}",
                f"- approval queue after: {approval_queue_after}",
                f"- approval queue delta: {approval_queue_delta}",
                f"- queued approval ids: {', '.join(str(item) for item in queued_ids) if queued_ids else 'none'}",
                f"- new approval ids: {', '.join(str(item) for item in new_approval_ids) if new_approval_ids else 'none'}",
                f"- reused approval ids: {', '.join(str(item) for item in reused_approval_ids) if reused_approval_ids else 'none'}",
                f"- referenced approval ids: {', '.join(str(item) for item in referenced_ids) if referenced_ids else 'none'}",
                f"- approved rerun: {'yes' if approved_reruns else 'no'}",
                f"- approved rerun run ids: {', '.join(str(item) for item in approved_rerun_run_ids) if approved_rerun_run_ids else 'none'}",
                f"- approved rerun approval ids: {', '.join(str(item) for item in approved_rerun_approval_ids) if approved_rerun_approval_ids else 'none'}",
                "",
                "Boundary:",
                "- This receipt reads stored runtime metadata only. It does not call models, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, or queue approvals.",
            ]
        )
        if checkpoint_recovery_execute_metadata:
            lines.extend(
                [
                    "",
                    "Checkpoint recovery handoff:",
                    f"- handoff present: {'yes' if checkpoint_recovery_execute_handoff else 'no'}",
                    f"- invalid reason: {checkpoint_recovery_execute_handoff_invalid_reason or 'none'}",
                    "- authority: non-authorizing runtime-trace evidence only",
                ]
            )

        runtime_trace_receipt_handoff = {
            "source": "runtime_trace_receipt",
            "found": True,
            "message_id": int(target["id"]),
            "session_id": target["session_id"],
            "route": route,
            "planner_notes": planner_notes,
            "planner_metadata": planner_metadata,
            "verdict": verdict,
            "verified": verified,
            "approval_required": approval_required,
            "ran_tool_handlers": ran_tool_handlers,
            "counts": {
                "stages": len(stages),
                "planned_actions": len(planned_actions),
                "tool_results": len(tool_results),
                "inspected_messages": len(rows),
                "readable_message_rows": len(rows),
                "unreadable_message_rows": unreadable_rows,
            },
            "approval": {
                "queued_approvals": len(queued_ids),
                "new_approval_count": len(new_approval_ids),
                "reused_approval_count": len(reused_approval_ids),
                "new_approval_ids": new_approval_ids,
                "reused_approval_ids": reused_approval_ids,
                "approval_queue_before": approval_queue_before,
                "approval_queue_after": approval_queue_after,
                "approval_queue_delta": approval_queue_delta,
                "referenced_approvals": len(referenced_ids),
                "referenced_approval_ids": referenced_ids,
                "approved_reruns": approved_reruns,
                "approved_rerun_run_ids": approved_rerun_run_ids,
                "approved_rerun_approval_ids": approved_rerun_approval_ids,
            },
            "risk_levels": risk_levels,
            "result_toolsets": dict(result_toolsets),
            "review_only": True,
            "draft_only": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "calls_model": False,
            "executes_tools": False,
            "writes_files": False,
            "reads_personal_data": False,
            "external_side_effect": False,
            "controls_computer": False,
            "queues_approval": False,
            "checkpoint_recovery_execute_handoff": checkpoint_recovery_execute_handoff,
            "has_checkpoint_recovery_execute_handoff": bool(checkpoint_recovery_execute_handoff),
            "checkpoint_recovery_execute_handoff_invalid_reason": (
                checkpoint_recovery_execute_handoff_invalid_reason
            ),
        }

        return ToolResult(
            "runtime_trace_receipt",
            True,
            "\n".join(lines),
            _safe_metadata(
                found=True,
                message_id=int(target["id"]),
                session_id=target["session_id"],
                route=route,
                planner_notes=planner_notes,
                planner_metadata=planner_metadata,
                planner_model_planner_attempted=planner_metadata.get("model_planner_attempted", False),
                planner_model_planner_state=planner_metadata.get("model_planner_state", ""),
                planner_model_planner_used=planner_metadata.get("model_planner_used", False),
                planner_model_planner_fell_back=planner_metadata.get("model_planner_fell_back", False),
                planner_model_planner_fallback_reason=planner_metadata.get("model_planner_fallback_reason", ""),
                planner_model_planner_fallback_detail=planner_metadata.get("model_planner_fallback_detail", ""),
                planner_model_planner_recovery_hint=planner_metadata.get("model_planner_recovery_hint", ""),
                planner_model_planner_exception_type=planner_metadata.get("model_planner_exception_type", ""),
                planner_model_planner_model=planner_metadata.get("model_planner_model", ""),
                planner_model_planner_provider=planner_metadata.get("model_planner_provider", ""),
                planner_model_planner_timeout_seconds=planner_metadata.get("model_planner_timeout_seconds"),
                planner_model_planner_usage_available=planner_metadata.get("model_planner_usage_available", False),
                planner_model_planner_usage_consistent=planner_metadata.get("model_planner_usage_consistent", False),
                planner_model_planner_input_tokens=planner_metadata.get("model_planner_input_tokens", 0),
                planner_model_planner_cached_input_tokens=planner_metadata.get("model_planner_cached_input_tokens", 0),
                planner_model_planner_output_tokens=planner_metadata.get("model_planner_output_tokens", 0),
                planner_model_planner_reasoning_tokens=planner_metadata.get("model_planner_reasoning_tokens", 0),
                planner_model_planner_total_tokens=planner_metadata.get("model_planner_total_tokens", 0),
                planner_model_planner_usage_content_recorded=False,
                planner_model_planner_action_count=planner_metadata.get("model_planner_action_count", 0),
                planner_model_planner_ignored_unknown_tools=planner_metadata.get("model_planner_ignored_unknown_tools", []),
                verdict=verdict,
                verified=verified,
                approval_required=approval_required,
                queued_approvals=len(queued_ids),
                new_approval_count=len(new_approval_ids),
                reused_approval_count=len(reused_approval_ids),
                new_approval_ids=new_approval_ids,
                reused_approval_ids=reused_approval_ids,
                approval_queue_before=approval_queue_before,
                approval_queue_after=approval_queue_after,
                approval_queue_delta=approval_queue_delta,
                referenced_approvals=len(referenced_ids),
                referenced_approval_ids=referenced_ids,
                approved_reruns=approved_reruns,
                approved_rerun_run_ids=approved_rerun_run_ids,
                approved_rerun_approval_ids=approved_rerun_approval_ids,
                stages=len(stages),
                planned_actions=len(planned_actions),
                tool_results=len(tool_results),
                risk_levels=risk_levels,
                result_toolsets=dict(result_toolsets),
                inspected_messages=len(rows),
                readable_message_rows=len(rows),
                unreadable_message_rows=unreadable_rows,
                runtime_trace_receipt_handoff=runtime_trace_receipt_handoff,
                checkpoint_recovery_execute_handoff=checkpoint_recovery_execute_handoff,
                has_checkpoint_recovery_execute_handoff=bool(checkpoint_recovery_execute_handoff),
                runtime_trace_receipt_checkpoint_recovery_execute_handoff=checkpoint_recovery_execute_handoff,
                runtime_trace_receipt_has_checkpoint_recovery_execute_handoff=bool(
                    checkpoint_recovery_execute_handoff
                ),
                checkpoint_recovery_execute_handoff_invalid_reason=(
                    checkpoint_recovery_execute_handoff_invalid_reason
                ),
                runtime_trace_receipt_checkpoint_recovery_execute_handoff_invalid_reason=(
                    checkpoint_recovery_execute_handoff_invalid_reason
                ),
                runtime_trace_receipt_review_only=True,
                runtime_trace_receipt_draft_only=True,
                runtime_trace_receipt_loads_without_execution=True,
                runtime_trace_receipt_authorizes_execution=False,
                runtime_trace_receipt_authorizes_completion_claim=False,
                runtime_trace_receipt_approval_granted=False,
            ),
        )

    def execution_audit_gate(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 30)
        rows = store.recent_tool_runs(limit)
        approval_proof_cache: dict[int, Any] = {}
        if not rows:
            execution_audit_handoff = {
                "source": "execution_audit_gate",
                "verdict": "NO_RECENT_RUNS",
                "review_required": False,
                "safe_to_trust_recent_execution": False,
                "inspected_runs": 0,
                "failed_or_blocked_runs": 0,
                "risk_gated_runs": 0,
                "risky_unapproved_successes": 0,
                "risky_runs_missing_approval_ids": 0,
                "verification_runs": 0,
                "problem_runs": [],
                "next_recovery_command": "",
                "audit_proof_queue": [],
                "audit_proof_queue_count": 0,
                "audit_next_required_command": "",
                "audit_next_proof_command": "",
                "proof_queue": [],
                "proof_queue_count": 0,
                "next_required_command": "",
                "next_proof_command": "",
                "review_only": True,
                "draft_only": True,
                "loads_without_execution": True,
                "authorizes_execution": False,
                "authorizes_completion_claim": False,
                "approval_granted": False,
                "calls_model": False,
                "executes_tools": False,
                "writes_files": False,
                "reads_personal_data": False,
                "external_side_effect": False,
                "controls_computer": False,
                "queues_approval": False,
            }
            return ToolResult(
                "execution_audit_gate",
                True,
                "\n".join(
                    [
                        "Jarvis execution audit gate:",
                        "No recent tool runs are logged yet.",
                        "",
                        "Verdict: NO_RECENT_RUNS",
                        "Next step: run a read-only or local-safe command, then inspect `execution audit gate` again.",
                    ]
                ),
                _safe_metadata(
                    verdict="NO_RECENT_RUNS",
                    inspected_runs=0,
                    failed_or_blocked_runs=0,
                    risky_unapproved_successes=0,
                    risky_runs_missing_approval_ids=0,
                    review_required=False,
                    safe_to_trust_recent_execution=False,
                    execution_audit_handoff=execution_audit_handoff,
                    execution_audit_review_only=True,
                    execution_audit_draft_only=True,
                    execution_audit_loads_without_execution=True,
                    execution_audit_authorizes_execution=False,
                    execution_audit_authorizes_completion_claim=False,
                    execution_audit_approval_granted=False,
                ),
            )

        failed_or_blocked = [
            row
            for row in rows
            if not bool(row["ok"])
            or _run_approval_evidence(row, approval_proof_cache).outcome_unknown
        ]
        risky_runs = [row for row in rows if _risk_needs_approval(row["risk"])]
        risky_unapproved_successes = [
            row
            for row in risky_runs
            if _run_approval_evidence(row, approval_proof_cache).approval_problem
            and not (
                bool(row["approved"])
                and ("approval_id" not in row.keys() or row["approval_id"] is None)
            )
        ]
        risky_missing_approval_ids = [
            row
            for row in risky_runs
            if _run_approval_evidence(row, approval_proof_cache).approval_problem
            and bool(row["approved"])
            and ("approval_id" not in row.keys() or row["approval_id"] is None)
        ]
        verification_runs = [
            row
            for row in rows
            if str(row["tool_name"]) in {"verification_receipt", "runtime_trace_receipt", "verification_packet", "completion_audit_packet", "evidence_ledger", "task_completion_packet"}
        ]
        review_required = bool(failed_or_blocked or risky_unapproved_successes or risky_missing_approval_ids)
        if risky_unapproved_successes:
            verdict = "RISKY_SUCCESS_WITHOUT_APPROVAL_EVIDENCE"
        elif risky_missing_approval_ids:
            verdict = "APPROVAL_LINK_MISSING"
        elif failed_or_blocked:
            verdict = "RECOVERY_REVIEW_REQUIRED"
        else:
            verdict = "AUDIT_GATE_CLEAR"

        lines = [
            "Jarvis execution audit gate:",
            "This is the run-integrity stop-check before Jarvis trusts recent execution. It is read-only and does not rerun tools.",
            "",
            f"Verdict: {verdict}",
            f"Inspected runs: {len(rows)}",
            f"Failed or blocked runs: {len(failed_or_blocked)}",
            f"Risk-gated runs: {len(risky_runs)}",
            f"Risky invoked runs without approval evidence: {len(risky_unapproved_successes)}",
            f"Approved risky runs missing approval id: {len(risky_missing_approval_ids)}",
            f"Recent verification/audit packets: {len(verification_runs)}",
            "",
            "Problem runs:",
        ]
        problem_rows = failed_or_blocked[:5] + risky_unapproved_successes[:5] + risky_missing_approval_ids[:5]
        seen_ids: set[int] = set()
        unique_problem_rows = []
        for row in problem_rows:
            row_id = int(row["id"])
            if row_id in seen_ids:
                continue
            seen_ids.add(row_id)
            unique_problem_rows.append(row)
        if unique_problem_rows:
            for row in unique_problem_rows:
                approved = (
                    "approved"
                    if bool(row["approved"])
                    and _run_approval_evidence(row, approval_proof_cache).approval_linked
                    else "not approved"
                )
                approval_id = row["approval_id"] if "approval_id" in row.keys() else None
                approval_text = f", approval #{approval_id}" if approval_id else ""
                status = "ok" if bool(row["ok"]) else "failed/blocked"
                lines.append(
                    f"- #{row['id']} {row['tool_name']} [{row['risk']}, {status}, {approved}{approval_text}]"
                )
                snippet = _short(row["output"], limit=180)
                if snippet:
                    lines.append(f"  output: {snippet}")
        else:
            lines.append("- none found in inspected window")

        lines.extend(["", "Recent verification evidence:"])
        if verification_runs:
            for row in verification_runs[:5]:
                status = "ok" if bool(row["ok"]) else "failed"
                lines.append(f"- #{row['id']} {row['tool_name']} [{status}] {row['created_at']}")
        else:
            lines.append("- none in inspected window")

        lines.extend(["", "Required recovery path:"])
        if risky_unapproved_successes:
            lines.append("- Stop trusting any invoked risky run without exact linked approval evidence; inspect `approval history` and `runtime trace receipt`.")
        if risky_missing_approval_ids:
            lines.append("- Link approved risky execution back to a specific approval id before counting it as completion evidence.")
        if failed_or_blocked:
            first = failed_or_blocked[0]
            lines.append(f"- Inspect `execution recovery packet {first['id']}` before retrying or claiming completion.")
            lines.append(f"- Use `verification receipt {first['id']}` inside that recovery packet for exact stored evidence.")
        if not review_required:
            lines.append("- No execution-integrity blockers found in the inspected window.")
        if not verification_runs:
            lines.append("- Run `verification receipt latest` or `runtime trace receipt` after the next meaningful action to create proof.")
        audit_proof_queue: list[str] = []
        if unique_problem_rows:
            first_problem = unique_problem_rows[0]
            audit_proof_queue.extend(
                [
                    f"verification receipt {first_problem['id']}",
                    f"execution recovery packet {first_problem['id']}",
                    f"after-action learning packet {first_problem['id']}",
                    f"execution learning closure {first_problem['id']}",
                ]
            )
            first_approval_id = first_problem["approval_id"] if "approval_id" in first_problem.keys() else None
            if first_approval_id:
                audit_proof_queue.extend(_approval_proof_chain_commands(int(first_approval_id)))
        if not verification_runs:
            audit_proof_queue.extend(["verification receipt latest", "runtime trace receipt"])
        audit_proof_queue.extend(["execution health report", "execution audit gate"])
        audit_proof_queue = _ordered_commands(audit_proof_queue)
        next_audit_proof_command = audit_proof_queue[0] if audit_proof_queue else ""
        problem_run_summaries = [
            {
                "run_id": int(row["id"]),
                "tool_name": str(row["tool_name"]),
                "risk": str(row["risk"]),
                "ok": bool(row["ok"]),
                "approved": bool(row["approved"])
                and _run_approval_evidence(row, approval_proof_cache).approval_linked,
                "approval_id": row["approval_id"] if "approval_id" in row.keys() else None,
            }
            for row in unique_problem_rows
        ]
        execution_audit_handoff = {
            "source": "execution_audit_gate",
            "verdict": verdict,
            "review_required": review_required,
            "safe_to_trust_recent_execution": not review_required,
            "inspected_runs": len(rows),
            "failed_or_blocked_runs": len(failed_or_blocked),
            "risk_gated_runs": len(risky_runs),
            "risky_unapproved_successes": len(risky_unapproved_successes),
            "risky_runs_missing_approval_ids": len(risky_missing_approval_ids),
            "verification_runs": len(verification_runs),
            "newest_run_id": int(rows[0]["id"]),
            "newest_problem_run_id": int(unique_problem_rows[0]["id"]) if unique_problem_rows else None,
            "problem_runs": problem_run_summaries,
            "problem_run_count": len(problem_run_summaries),
            "next_recovery_command": f"execution recovery packet {unique_problem_rows[0]['id']}" if unique_problem_rows else "",
            "audit_proof_queue": audit_proof_queue,
            "audit_proof_queue_count": len(audit_proof_queue),
            "audit_next_required_command": next_audit_proof_command,
            "audit_next_proof_command": next_audit_proof_command,
            "proof_queue": audit_proof_queue,
            "proof_queue_count": len(audit_proof_queue),
            "next_required_command": next_audit_proof_command,
            "next_proof_command": next_audit_proof_command,
            "review_only": True,
            "draft_only": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "calls_model": False,
            "executes_tools": False,
            "writes_files": False,
            "reads_personal_data": False,
            "external_side_effect": False,
            "controls_computer": False,
            "queues_approval": False,
        }

        lines.extend(
            [
                "",
                "Execution proof queue:",
                f"- next required command: `{next_audit_proof_command}`" if next_audit_proof_command else "- next required command: none",
                f"- proof queue count: {len(audit_proof_queue)}",
                "- proof queue: " + ", ".join(f"`{command}`" for command in audit_proof_queue) if audit_proof_queue else "- proof queue: none",
            ]
        )

        lines.extend(
            [
                "",
                "Boundary:",
                "- This gate reads recent tool-run audit rows only. It does not call models, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, or queue approvals.",
            ]
        )

        return ToolResult(
            "execution_audit_gate",
            True,
            "\n".join(lines),
            _safe_metadata(
                verdict=verdict,
                inspected_runs=len(rows),
                failed_or_blocked_runs=len(failed_or_blocked),
                risk_gated_runs=len(risky_runs),
                risky_unapproved_successes=len(risky_unapproved_successes),
                risky_runs_missing_approval_ids=len(risky_missing_approval_ids),
                verification_runs=len(verification_runs),
                review_required=review_required,
                safe_to_trust_recent_execution=not review_required,
                newest_run_id=int(rows[0]["id"]),
                newest_problem_run_id=int(unique_problem_rows[0]["id"]) if unique_problem_rows else None,
                problem_runs=problem_run_summaries,
                problem_run_count=len(problem_run_summaries),
                next_recovery_command=f"execution recovery packet {unique_problem_rows[0]['id']}" if unique_problem_rows else "",
                audit_proof_queue=audit_proof_queue,
                audit_proof_queue_count=len(audit_proof_queue),
                audit_next_required_command=next_audit_proof_command,
                audit_next_proof_command=next_audit_proof_command,
                proof_queue=audit_proof_queue,
                proof_queue_count=len(audit_proof_queue),
                next_required_command=next_audit_proof_command,
                next_proof_command=next_audit_proof_command,
                execution_audit_handoff=execution_audit_handoff,
                execution_audit_review_only=True,
                execution_audit_draft_only=True,
                execution_audit_loads_without_execution=True,
                execution_audit_authorizes_execution=False,
                execution_audit_authorizes_completion_claim=False,
                execution_audit_approval_granted=False,
            ),
        )

    def execution_recovery_packet(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 30)
        run_id_raw = _first_present(args, ("run_id", "id", "tool_run_id"), "")
        rows = store.recent_tool_runs(limit)
        approval_proof_cache: dict[int, Any] = {}
        if not rows:
            return ToolResult(
                "execution_recovery_packet",
                True,
                "\n".join(
                    [
                        "Jarvis execution recovery packet:",
                        "No recent tool runs are logged yet.",
                        "",
                        "Verdict: NO_RECENT_RUNS",
                        "Safe next command:",
                        "- `readiness report` or one read-only command, then `execution audit gate`.",
                        "",
                        "Boundary:",
                        "- This packet is read-only; it does not retry, approve, dismiss, write files, run shell/code, read private data, control the computer, or queue approvals.",
                    ]
                ),
                _safe_metadata(verdict="NO_RECENT_RUNS", inspected_runs=0, problem_found=False),
            )

        target = None
        if _has_explicit_value(run_id_raw):
            try:
                run_id = int(run_id_raw)
            except (TypeError, ValueError):
                return _audit_input_failure(
                    "execution_recovery_packet",
                    "Tool run id must be a number when supplied.",
                    _safe_metadata(verdict="INVALID_ID", run_id=_short_raw(run_id_raw), raw_run_id=_short_raw(run_id_raw), inspected_runs=len(rows), problem_found=False),
                )
            if run_id <= 0:
                return _audit_input_failure(
                    "execution_recovery_packet",
                    "Tool run id must be a positive number when supplied.",
                    _safe_metadata(verdict="INVALID_ID", run_id=_short_raw(run_id_raw), raw_run_id=_short_raw(run_id_raw), inspected_runs=len(rows), problem_found=False),
                )
            target = store.get_tool_run(run_id)
            if target is None:
                safe_run_id = _short_raw(run_id_raw)
                return _audit_not_found_failure(
                    "execution_recovery_packet",
                    f"Tool run #{safe_run_id} was not found in the audit log. Run `recent tool runs` to "
                    "refresh valid IDs, then retry `execution recovery` for the latest failed run.",
                    _lookup_recovery_metadata(
                        "recent tool runs",
                        "execution recovery",
                        verdict="MISSING_RUN",
                        run_id=safe_run_id,
                        inspected_runs=len(rows),
                        problem_found=False,
                    ),
                )
        else:
            for row in rows:
                ok = bool(row["ok"])
                row_approval_evidence = _run_approval_evidence(row, approval_proof_cache)
                if not ok or row_approval_evidence.outcome_unknown or (
                    _risk_needs_approval(row["risk"])
                    and row_approval_evidence.approval_problem
                ):
                    target = row
                    break

        if target is None:
            newest = rows[0]
            return ToolResult(
                "execution_recovery_packet",
                True,
                "\n".join(
                    [
                        "Jarvis execution recovery packet:",
                        "No recovery blocker was found in the inspected window.",
                        "",
                        "Verdict: RECOVERY_NOT_REQUIRED",
                        f"Newest run: #{newest['id']} {newest['tool_name']}",
                        "",
                        "Safe next command:",
                        "- `verification receipt latest` to preserve after-action evidence for the newest run.",
                        "- `execution audit gate` before claiming broader completion.",
                        "",
                        "Boundary:",
                        "- This packet is read-only; it does not retry, approve, dismiss, write files, run shell/code, read private data, control the computer, or queue approvals.",
                    ]
                ),
                _safe_metadata(
                    verdict="RECOVERY_NOT_REQUIRED",
                    inspected_runs=len(rows),
                    problem_found=False,
                    newest_run_id=int(newest["id"]),
                ),
            )

        run_id = int(target["id"])
        tool_name = str(target["tool_name"])
        risk = str(target["risk"])
        ok = bool(target["ok"])
        stored_approved = bool(target["approved"])
        approval_evidence = _run_approval_evidence(target, approval_proof_cache)
        approval_proven = approval_evidence.approval_linked
        approved = stored_approved and approval_proven
        approval_id = target["approval_id"] if "approval_id" in target.keys() else None
        output = _safe_text(target["output"])
        audit_metadata = _row_metadata(target)
        failure_kind = _safe_text(audit_metadata.get("failure_kind"))
        executed_handler = audit_metadata.get("executed_handler")
        planned_arg_keys = audit_metadata.get("planned_arg_keys") if isinstance(audit_metadata.get("planned_arg_keys"), list) else []
        toolset = _short(audit_metadata.get("toolset"), limit=80)
        output_preview = _short(output, limit=520)
        output_approval_ids = _ids_from_text(output, ("approval #", "approval id", "approval packet", "queued as approval"))
        inferred_approval_ids: list[int] = []
        if not output_approval_ids:
            for approval in store.recent_approvals(limit=20):
                if str(approval["tool_name"]) != tool_name:
                    continue
                approval_candidate = int(approval["id"])
                if approval_candidate not in inferred_approval_ids:
                    inferred_approval_ids.append(approval_candidate)
        risk_needs_approval = approval_evidence.risk_needs_approval
        approval_problem = approval_evidence.approval_problem

        if not ok and failure_kind == "approval_required":
            verdict = "HELD_FOR_APPROVAL"
            recovery_kind = "last_look_approval_review"
        elif approval_problem and stored_approved and approval_id is None:
            verdict = "APPROVAL_LINK_MISSING"
            recovery_kind = "link_approved_rerun_before_trusting"
        elif approval_problem:
            verdict = "APPROVAL_EVIDENCE_MISSING"
            recovery_kind = "prove_approval_chain_before_trusting"
        elif approval_evidence.outcome_unknown:
            verdict = "FAILED_OR_BLOCKED"
            recovery_kind = "verify_unknown_outcome_before_retry"
        elif not ok and failure_kind == "unknown_tool":
            verdict = "UNKNOWN_TOOL"
            recovery_kind = "register_tool_or_fix_planner_route"
        elif not ok and failure_kind == "tool_error":
            verdict = "TOOL_ERROR"
            recovery_kind = "diagnose_handler_error_before_retry"
        elif not ok and "explicit approval required" in output.lower():
            verdict = "HELD_FOR_APPROVAL"
            recovery_kind = "last_look_approval_review"
        elif not ok:
            verdict = "FAILED_OR_BLOCKED"
            recovery_kind = "diagnose_failure_before_retry"
        else:
            verdict = "RECOVERY_NOT_REQUIRED"
            recovery_kind = "preserve_verification_evidence"
        failure_summary = _short(
            f"tool run #{run_id} {tool_name} produced {verdict}: {output_preview or '(empty output)'}",
            limit=260,
        )
        failure_to_test_command = f"failure to test preview: {failure_summary}"
        next_commands = [f"verification receipt {run_id}"]
        approval_packet_ids = output_approval_ids or inferred_approval_ids
        approval_proof_chains = {
            str(candidate): _approval_proof_chain_commands(candidate)
            for candidate in approval_packet_ids
        }
        if risk_needs_approval and approval_id and str(approval_id) not in approval_proof_chains:
            approval_proof_chains[str(approval_id)] = _approval_proof_chain_commands(int(approval_id))

        lines = [
            "Jarvis execution recovery packet:",
            "This is the read-only recovery steering layer after a failed, blocked, or weakly evidenced tool run.",
            "",
            "Problem run:",
            f"- run: #{run_id} `{tool_name}`",
            f"- created: {target['created_at']}",
            f"- risk: {risk}",
            f"- toolset: {toolset or 'unknown'}",
            f"- result: {'ok' if ok else 'failed/blocked'}",
            f"- approved: {'yes' if approved else 'no'}",
            f"- linked approval id: #{approval_id}" if approval_id else "- linked approval id: none",
            f"- failure kind: {failure_kind or 'unknown'}",
            f"- handler executed: {'yes' if executed_handler is True else 'no' if executed_handler is False else 'unknown'}",
            f"- planned arg keys: {', '.join(str(item) for item in planned_arg_keys) if planned_arg_keys else 'none'}",
            f"- output: {output_preview or '(empty output)'}",
            "",
            f"Verdict: {verdict}",
            f"Recovery kind: {recovery_kind}",
            "",
            "Safe recovery sequence:",
            f"1. Inspect `verification receipt {run_id}` for the exact stored evidence.",
        ]
        if approval_packet_ids:
            _extend_unique(next_commands, _approval_proof_chain_commands(approval_packet_ids[0]))
            lines.append(f"2. Review `approval readiness {approval_packet_ids[0]}`, then `approval packet {approval_packet_ids[0]}`, before any approval.")
            lines.append(f"3. If the operator explicitly trusts it, run `approve approval {approval_packet_ids[0]}`, then `approval chain proof {approval_packet_ids[0]}`, and verify the approved rerun.")
        elif risk_needs_approval and approval_id:
            _extend_unique(next_commands, _approval_proof_chain_commands(int(approval_id)) + ["approval history"])
            lines.append(f"2. Inspect `approval readiness {approval_id}`, `approval packet {approval_id}`, `approval chain proof {approval_id}`, and `approval history` for approval-chain evidence.")
            lines.append("3. Verify the linked approved rerun before treating this as completion evidence.")
        elif not ok:
            if failure_kind == "unknown_tool":
                next_commands.extend(["tool search: " + tool_name, "list tools"])
                lines.append("2. Check whether the planner selected a nonexistent tool or a tool registration is missing.")
                lines.append("3. Fix the route/registry mismatch, then add or update a focused planner smoke test.")
            elif failure_kind == "tool_error":
                next_commands.append(failure_to_test_command)
                lines.append("2. Inspect the handler error and exact planned argument keys before retrying.")
                lines.append("3. Add a focused regression around the handler path before relying on the fix.")
            else:
                next_commands.extend(["execution health report", failure_to_test_command])
                lines.append("2. Do not silently retry; choose a narrower read-only/local-safe diagnostic or recovery step.")
                lines.append("3. If the failure is repeatable, promote it with `failure to test preview` before patching.")
        else:
            next_commands.extend(["runtime trace receipt", "execution audit gate"])
            lines.append("2. Preserve `runtime trace receipt` or `verification receipt latest` as proof.")
            lines.append("3. Continue only after `execution audit gate` is clear.")
        if approval_proof_chains:
            lines.extend(["", "Approval proof chain handoff:"])
            for chain_approval_id, commands in approval_proof_chains.items():
                lines.append(f"- approval #{chain_approval_id}:")
                lines.extend(f"  - `{command}`" for command in commands)
            lines.append("- Do not claim risky recovery complete until the chain proof is valid and the approved run has a verification receipt.")
        for command in [f"after-action learning packet {run_id}", f"execution learning closure {run_id}", "execution health report", "execution audit gate"]:
            if command not in next_commands:
                next_commands.append(command)
        closure_checks = [
            {
                "name": "verification receipt",
                "state": "required",
                "command": f"verification receipt {run_id}",
                "blocks_retry": True,
            },
            {
                "name": "recovery packet",
                "state": "current packet",
                "command": f"execution recovery packet {run_id}",
                "blocks_retry": False,
            },
            {
                "name": "after-action learning",
                "state": "required",
                "command": f"after-action learning packet {run_id}",
                "blocks_retry": True,
            },
            {
                "name": "execution learning closure",
                "state": "required",
                "command": f"execution learning closure {run_id}",
                "blocks_retry": True,
            },
            {
                "name": "execution audit gate",
                "state": "required",
                "command": "execution audit gate",
                "blocks_retry": True,
            },
        ]
        if approval_proof_chains:
            for chain_approval_id, commands in approval_proof_chains.items():
                closure_checks.append(
                    {
                        "name": f"approval proof chain {chain_approval_id}",
                        "state": "required",
                        "command": f"approval chain proof {chain_approval_id}",
                        "blocks_retry": True,
                        "proof_commands": commands,
                    }
                )
        closure_required_commands: list[str] = []
        for check in closure_checks:
            command = _safe_text(check.get("command"))
            if command and command not in closure_required_commands:
                closure_required_commands.append(command)
        execution_recovery_handoff = {
            "source": "execution_recovery_packet",
            "verdict": verdict,
            "recovery_kind": recovery_kind,
            "problem_found": verdict != "RECOVERY_NOT_REQUIRED",
            "target": {
                "run_id": run_id,
                "tool_name": tool_name,
                "risk": risk,
                "toolset": toolset,
                "ok": ok,
                "approved": approved,
                "approval_id": approval_id,
                "failure_kind": failure_kind,
                "executed_handler": executed_handler,
                "planned_arg_keys": planned_arg_keys,
            },
            "approval": {
                "risk_needs_approval": risk_needs_approval,
                "output_approval_ids": output_approval_ids,
                "inferred_approval_ids": inferred_approval_ids,
                "recovery_approval_ids": approval_packet_ids,
                "proof_chains": approval_proof_chains,
            },
            "next_command": next_commands[0],
            "next_commands": next_commands,
            "next_command_count": len(next_commands),
            "failure_to_test_command": failure_to_test_command,
            "recovery_closure": {
                "checks": closure_checks,
                "check_count": len(closure_checks),
                "required_commands": closure_required_commands,
                "required_command_count": len(closure_required_commands),
                "proof_queue": closure_required_commands,
                "proof_queue_count": len(closure_required_commands),
                "next_required_command": closure_required_commands[0] if closure_required_commands else "",
                "next_proof_command": closure_required_commands[0] if closure_required_commands else "",
                "blocks_retry": bool(closure_required_commands),
            },
            "review_only": True,
            "draft_only": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "calls_model": False,
            "executes_tools": False,
            "writes_files": False,
            "reads_personal_data": False,
            "external_side_effect": False,
            "controls_computer": False,
            "queues_approval": False,
        }
        lines.extend(
            [
                "",
                "Next command queue:",
                *[f"- `{command}`" for command in next_commands],
                "",
                "Recovery closure checklist:",
                *[
                    f"- {check['name']}: {check['state']} -> `{check['command']}`"
                    for check in closure_checks
                ],
                "- Retry readiness: blocked until required verification, learning, audit, and approval proof checks are reviewed.",
                "",
                "Learning hook:",
                "- If this failure repeats, turn the command, blocker, expected behavior, and verification target into a smoke test or reviewed skill.",
                f"- Regression preview command: `{failure_to_test_command}`",
                "",
                "Stop conditions:",
                "- stop at the operator's explicit stop times, work windows, pause commands, or newer instructions, even if a priority goal is active",
                "- stop if the target command, file, app, account, recipient, coordinate, or success condition is unclear",
                "- stop if approval evidence is missing for risky execution",
                "- stop if verification cannot prove the approved rerun",
                "",
                "Boundary:",
                "- This packet reads audit rows only. It does not retry tools, approve requests, dismiss approvals, write files, run shell/code, read private data, control the computer, call external services, complete tasks, or queue approvals.",
            ]
        )

        return ToolResult(
            "execution_recovery_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                verdict=verdict,
                recovery_kind=recovery_kind,
                inspected_runs=len(rows),
                problem_found=verdict != "RECOVERY_NOT_REQUIRED",
                run_id=run_id,
                tool_name=tool_name,
                risk=risk,
                toolset=toolset,
                ok=ok,
                approved=approved,
                approval_id=approval_id,
                failure_kind=failure_kind,
                # `executed_handler` is reserved for the current Executor
                # invocation. Keep historical target truth under an explicit
                # target-prefixed alias so the executor cannot overwrite it.
                target_executed_handler=executed_handler,
                planned_arg_keys=planned_arg_keys,
                output_approval_ids=output_approval_ids,
                inferred_approval_ids=inferred_approval_ids,
                recovery_approval_ids=approval_packet_ids,
                approval_proof_chains=approval_proof_chains,
                failure_to_test_command=failure_to_test_command,
                next_command=next_commands[0],
                next_commands=next_commands,
                next_command_count=len(next_commands),
                recovery_closure_checks=closure_checks,
                recovery_closure_check_count=len(closure_checks),
                recovery_closure_required_commands=closure_required_commands,
                recovery_closure_next_required_command=closure_required_commands[0] if closure_required_commands else "",
                recovery_closure_proof_queue=closure_required_commands,
                recovery_closure_proof_queue_count=len(closure_required_commands),
                recovery_closure_next_proof_command=closure_required_commands[0] if closure_required_commands else "",
                recovery_closure_blocks_retry=bool(closure_required_commands),
                risk_needs_approval=risk_needs_approval,
                execution_recovery_handoff=execution_recovery_handoff,
                execution_recovery_review_only=True,
                execution_recovery_draft_only=True,
                execution_recovery_loads_without_execution=True,
                execution_recovery_authorizes_execution=False,
                execution_recovery_authorizes_completion_claim=False,
                execution_recovery_approval_granted=False,
            ),
        )

    def after_action_learning_packet(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 30)
        run_id_raw = _first_present(args, ("run_id", "id", "tool_run_id"), "")
        rows = store.recent_tool_runs(limit)
        approval_proof_cache: dict[int, Any] = {}
        if not rows:
            return ToolResult(
                "after_action_learning_packet",
                True,
                "\n".join(
                    [
                        "Jarvis after-action learning packet:",
                        "No recent tool runs are logged yet.",
                        "",
                        "Verdict: NO_RECENT_RUNS",
                        "Safe next command:",
                        "- Run one read-only or local-safe command, then ask for `after-action learning packet`.",
                        "",
                        "Boundary:",
                        "- This packet is read-only. It does not save memories, create tasks, write tests, edit files, approve requests, dismiss approvals, retry tools, read private data, control the computer, or queue approvals.",
                    ]
                ),
                _safe_metadata(verdict="NO_RECENT_RUNS", inspected_runs=0, learning_candidates=0),
            )

        target = None
        if _has_explicit_value(run_id_raw):
            try:
                run_id = int(run_id_raw)
            except (TypeError, ValueError):
                return _audit_input_failure(
                    "after_action_learning_packet",
                    "Tool run id must be a number when supplied.",
                    _safe_metadata(verdict="INVALID_ID", run_id=_short_raw(run_id_raw), raw_run_id=_short_raw(run_id_raw), inspected_runs=len(rows), learning_candidates=0),
                )
            if run_id <= 0:
                return _audit_input_failure(
                    "after_action_learning_packet",
                    "Tool run id must be a positive number when supplied.",
                    _safe_metadata(verdict="INVALID_ID", run_id=_short_raw(run_id_raw), raw_run_id=_short_raw(run_id_raw), inspected_runs=len(rows), learning_candidates=0),
                )
            target = store.get_tool_run(run_id)
            if target is None:
                safe_run_id = _short_raw(run_id_raw)
                return _audit_not_found_failure(
                    "after_action_learning_packet",
                    f"Tool run #{safe_run_id} was not found in the audit log. Run `recent tool runs` to "
                    "refresh valid IDs, then retry `after action learning packet` for the latest run.",
                    _lookup_recovery_metadata(
                        "recent tool runs",
                        "after action learning packet",
                        verdict="MISSING_RUN",
                        run_id=safe_run_id,
                        inspected_runs=len(rows),
                        learning_candidates=0,
                    ),
                )
        else:
            target = next(
                (
                    row
                    for row in rows
                    if not bool(row["ok"])
                    or _run_approval_evidence(row, approval_proof_cache).outcome_unknown
                    or (
                        _risk_needs_approval(row["risk"])
                        and _run_approval_evidence(row, approval_proof_cache).approval_problem
                    )
                ),
                _preferred_after_action_target(rows),
            )

        run_id = int(target["id"])
        tool_name = str(target["tool_name"])
        risk = str(target["risk"])
        ok = bool(target["ok"])
        approval_evidence = _run_approval_evidence(target, approval_proof_cache)
        approval_proven = approval_evidence.approval_linked
        approved = bool(target["approved"]) and approval_proven
        approval_id = target["approval_id"] if "approval_id" in target.keys() else None
        output = _short(target["output"], limit=520)
        output_approval_ids = _ids_from_text(target["output"], ("approval #", "approval id", "approval packet", "queued as approval"))
        audit_metadata = _row_metadata(target)
        failure_kind = _short(audit_metadata.get("failure_kind"), limit=90)
        planned_arg_keys = audit_metadata.get("planned_arg_keys") if isinstance(audit_metadata.get("planned_arg_keys"), list) else []
        toolset = _short(audit_metadata.get("toolset"), limit=80)
        risk_needs_approval = approval_evidence.risk_needs_approval
        approval_problem = approval_evidence.approval_problem
        approval_proof_ids: list[int] = []
        for candidate in ([int(approval_id)] if approval_id is not None else []) + output_approval_ids:
            if candidate not in approval_proof_ids:
                approval_proof_ids.append(candidate)
        approval_proof_chains = {
            str(candidate): _approval_proof_chain_commands(candidate)
            for candidate in approval_proof_ids
        }

        learning_candidates: list[str] = []
        if not ok:
            learning_candidates.append("regression-test candidate from failed or blocked execution")
        if failure_kind == "unknown_tool":
            learning_candidates.append("planner or registry route correction candidate")
        if failure_kind == "tool_error":
            learning_candidates.append("handler error regression candidate")
        if approval_problem:
            learning_candidates.append("approval-chain proof candidate")
        if approval_evidence.outcome_unknown:
            learning_candidates.append("outcome-unknown execution review candidate")
        if ok and not approval_problem and not approval_evidence.outcome_unknown:
            learning_candidates.append("workflow memory or reusable command-pattern candidate")
        if planned_arg_keys:
            learning_candidates.append("argument-contract coverage candidate")

        if approval_problem:
            verdict = "PROVE_APPROVAL_CHAIN_BEFORE_LEARNING"
        elif not ok or approval_evidence.outcome_unknown:
            verdict = "PROMOTE_FAILURE_TO_REVIEW"
        else:
            verdict = "SAFE_TO_REVIEW_FOR_LEARNING"

        failure_summary = _short(
            f"tool run #{run_id} {tool_name} [{risk}] {'ok' if ok else 'failed/blocked'}: {output or '(empty output)'}",
            limit=260,
        )
        regression_command = f"failure to test preview: {failure_summary}"
        verification_command = f"verification receipt {run_id}"
        recovery_command = f"execution recovery packet {run_id}"
        review_command = "learning review"
        task_command = (
            f"add task Review after-action learning for run #{run_id}: {tool_name}; "
            f"verification {verification_command}; candidate {learning_candidates[0] if learning_candidates else 'none'}"
        )
        next_commands = [
            verification_command,
            recovery_command,
            regression_command,
            review_command,
        ]
        if approval_proof_ids and (approval_problem or failure_kind == "approval_required"):
            next_commands = [verification_command, recovery_command]
            _extend_unique(next_commands, _approval_proof_chain_commands(approval_proof_ids[0]))
            _extend_unique(next_commands, [regression_command, review_command])
        next_commands.append(task_command)
        next_required_command = next_commands[0]

        lines = [
            "Jarvis after-action learning packet:",
            "This is the read-only learning layer after execution. It converts audit evidence into reviewable memory, task, skill, or regression-test candidates without saving anything.",
            "",
            "Source run:",
            f"- run: #{run_id} `{tool_name}`",
            f"- created: {target['created_at']}",
            f"- risk: {risk}",
            f"- toolset: {toolset or 'unknown'}",
            f"- result: {'ok' if ok else 'failed/blocked'}",
            f"- approved: {'yes' if approved else 'no'}",
            f"- linked approval id: #{approval_id}" if approval_id else "- linked approval id: none",
            f"- failure kind: {failure_kind or 'none'}",
            f"- planned arg keys: {', '.join(str(item) for item in planned_arg_keys) if planned_arg_keys else 'none'}",
            f"- output: {output or '(empty output)'}",
            "",
            f"Verdict: {verdict}",
            "",
            "Learning candidates:",
        ]
        lines.extend(f"- {candidate}" for candidate in learning_candidates)
        lines.extend(
            [
                "",
                "Safe review commands:",
                f"- `{verification_command}`",
                f"- `{recovery_command}`",
                f"- `{regression_command}`",
                f"- `{review_command}`",
            ]
        )
        if approval_proof_chains:
            lines.extend(["", "Approval proof chain handoff:"])
            for chain_approval_id, commands in approval_proof_chains.items():
                lines.append(f"- approval #{chain_approval_id}:")
                lines.extend(f"  - `{command}`" for command in commands)
            lines.append("- Do not promote risky learning until the chain proof is valid and the approved run has a verification receipt.")
        lines.extend(
            [
                "",
                "Next command queue:",
                *[f"- `{command}`" for command in next_commands],
                "",
                "Learning proof queue:",
                f"- next required command: `{next_required_command}`" if next_commands else "- next required command: none",
                f"- proof queue count: {len(next_commands)}",
                "- proof queue: " + ", ".join(f"`{command}`" for command in next_commands) if next_commands else "- proof queue: none",
                "",
                "Optional local-safe follow-up after human review:",
                f"- `{task_command}`",
                "",
                "Promotion rules:",
                "- Do not promote or continue learning work past the operator's explicit stop times, work windows, pause commands, or newer instructions.",
                "- Promote to a test only when the observed failure, expected behavior, and verification check are explicit.",
                "- Promote to memory or preference only when it reflects the operator's stable instruction or a repeated workflow.",
                "- Promote to a skill only after the workflow repeats and the safety boundary is clear.",
                "- Do not learn from risky execution until approval and verification evidence are linked.",
                "",
                "Boundary:",
                "- This packet is read-only. It does not save memories, create tasks, write tests, edit files, approve requests, dismiss approvals, retry tools, read private data, control the computer, call external services, complete tasks, or queue approvals.",
            ]
        )

        after_action_learning_handoff = {
            "source": "after_action_learning_packet",
            "verdict": verdict,
            "promotion_allowed": verdict == "SAFE_TO_REVIEW_FOR_LEARNING",
            "target": {
                "run_id": run_id,
                "tool_name": tool_name,
                "risk": risk,
                "toolset": toolset,
                "ok": ok,
                "approved": approved,
                "approval_id": approval_id,
                "failure_kind": failure_kind,
                "planned_arg_keys": planned_arg_keys,
            },
            "learning_candidates": learning_candidates,
            "learning_candidate_count": len(learning_candidates),
            "approval": {
                "risk_needs_approval": risk_needs_approval,
                "approval_problem": approval_problem,
                "output_approval_ids": output_approval_ids,
                "approval_proof_ids": approval_proof_ids,
                "proof_chains": approval_proof_chains,
            },
            "verification_command": verification_command,
            "recovery_command": recovery_command,
            "regression_command": regression_command,
            "task_command": task_command,
            "next_command": next_commands[0],
            "next_required_command": next_required_command,
            "next_commands": next_commands,
            "next_command_count": len(next_commands),
            "proof_queue": next_commands,
            "proof_queue_count": len(next_commands),
            "next_proof_command": next_commands[0],
            "learning_proof_queue": next_commands,
            "learning_proof_queue_count": len(next_commands),
            "learning_next_required_command": next_required_command,
            "learning_next_proof_command": next_commands[0],
            "after_action_learning_proof_queue": next_commands,
            "after_action_learning_proof_queue_count": len(next_commands),
            "after_action_learning_next_required_command": next_required_command,
            "after_action_learning_next_proof_command": next_commands[0],
            "review_only": True,
            "draft_only": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "calls_model": False,
            "executes_tools": False,
            "writes_files": False,
            "reads_personal_data": False,
            "external_side_effect": False,
            "controls_computer": False,
            "queues_approval": False,
        }

        return ToolResult(
            "after_action_learning_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                verdict=verdict,
                inspected_runs=len(rows),
                run_id=run_id,
                tool_name=tool_name,
                risk=risk,
                toolset=toolset,
                ok=ok,
                approved=approved,
                approval_id=approval_id,
                risk_needs_approval=risk_needs_approval,
                approval_problem=approval_problem,
                output_approval_ids=output_approval_ids,
                approval_proof_ids=approval_proof_ids,
                approval_proof_chains=approval_proof_chains,
                failure_kind=failure_kind,
                planned_arg_keys=planned_arg_keys,
                learning_candidates=len(learning_candidates),
                learning_candidate_names=learning_candidates,
                verification_command=verification_command,
                recovery_command=recovery_command,
                regression_command=regression_command,
                task_command=task_command,
                next_command=next_commands[0],
                next_required_command=next_required_command,
                next_commands=next_commands,
                next_command_count=len(next_commands),
                proof_queue=next_commands,
                proof_queue_count=len(next_commands),
                next_proof_command=next_commands[0],
                learning_proof_queue=next_commands,
                learning_proof_queue_count=len(next_commands),
                learning_next_required_command=next_required_command,
                learning_next_proof_command=next_commands[0],
                after_action_learning_proof_queue=next_commands,
                after_action_learning_proof_queue_count=len(next_commands),
                after_action_learning_next_required_command=next_required_command,
                after_action_learning_next_proof_command=next_commands[0],
                after_action_learning_handoff=after_action_learning_handoff,
                after_action_learning_review_only=True,
                after_action_learning_draft_only=True,
                after_action_learning_loads_without_execution=True,
                after_action_learning_authorizes_execution=False,
                after_action_learning_authorizes_completion_claim=False,
                after_action_learning_approval_granted=False,
            ),
        )

    def execution_health_report(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 80)
        rows = store.recent_tool_runs(limit)
        approval_proof_cache: dict[int, Any] = {}
        if not rows:
            return ToolResult(
                "execution_health_report",
                True,
                "\n".join(
                    [
                        "Jarvis execution health report:",
                        "No recent tool runs are logged yet.",
                        "",
                        "Verdict: NO_RECENT_RUNS",
                        "Safe next command:",
                        "- Run one read-only or local-safe command, then inspect `execution health report` again.",
                        "",
                        "Boundary:",
                        "- This report is read-only. It does not call models, execute tools, approve requests, dismiss approvals, retry tools, write files, read private data, control the computer, call external services, complete tasks, or queue approvals.",
                    ]
                ),
                _safe_metadata(
                    verdict="NO_RECENT_RUNS",
                    inspected_runs=0,
                    action_runs=0,
                    meta_runs=0,
                    failed_or_blocked_runs=0,
                    risky_runs=0,
                    risky_approval_problems=0,
                    repeated_failure_tools=[],
                    repeated_failure_count=0,
                    blocker_categories=[],
                    blocker_count=0,
                    approval_chain_gap_count=0,
                    verification_coverage_state="missing",
                    safe_to_continue=True,
                    review_required=False,
                    next_command="readiness report",
                    next_commands=["readiness report"],
                    next_command_count=1,
                    recovery_closure_state="not_needed",
                    recovery_closure_missing=[],
                    recovery_closure_missing_count=0,
                    recovery_closure_required_commands=[],
                    recovery_closure_next_required_command="",
                    recovery_closure_proof_queue=[],
                    recovery_closure_proof_queue_count=0,
                    recovery_closure_next_proof_command="",
                    recovery_closure_ready_to_retry=False,
                    recovery_closure_blocks_auto_execution=False,
                    recovery_closure_blocks_completion_claim=False,
                    recovery_closure_target_run_id=None,
                    recovery_closure_target_tool_name="",
                    recovery_closure_target_verification_receipts=0,
                    recovery_closure_target_recovery_packets=0,
                    recovery_closure_target_after_action_learning_packets=0,
                    execution_proof_queue=["readiness report"],
                    execution_proof_queue_count=1,
                    execution_next_proof_command="readiness report",
                    proof_queue=["readiness report"],
                    proof_queue_count=1,
                    next_proof_command="readiness report",
                ),
            )

        def _int_trace(value: Any, default: int = 0) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        runtime_traces: list[dict[str, Any]] = []
        for message in store.recent_messages(limit=limit):
            if message["role"] != "assistant":
                continue
            try:
                message_metadata = json.loads(message["metadata"] or "{}")
            except json.JSONDecodeError:
                continue
            trace = message_metadata.get("runtime_trace")
            if isinstance(trace, dict):
                runtime_traces.append(trace)
        runtime_queue_delta_total = sum(_int_trace(trace.get("approval_queue_delta")) for trace in runtime_traces)
        runtime_new_approval_ids = sorted(
            {
                int(approval_id)
                for trace in runtime_traces
                for approval_id in (trace.get("new_approval_ids") if isinstance(trace.get("new_approval_ids"), list) else [])
                if approval_id is not None
            }
        )
        runtime_reused_approval_ids = sorted(
            {
                int(approval_id)
                for trace in runtime_traces
                for approval_id in (trace.get("reused_approval_ids") if isinstance(trace.get("reused_approval_ids"), list) else [])
                if approval_id is not None
            }
        )
        runtime_queued_approval_ids = sorted(
            {
                int(approval_id)
                for trace in runtime_traces
                for approval_id in (trace.get("queued_approval_ids") if isinstance(trace.get("queued_approval_ids"), list) else [])
                if approval_id is not None
            }
        )
        runtime_referenced_approval_ids = sorted(
            {
                int(approval_id)
                for trace in runtime_traces
                for approval_id in (trace.get("referenced_approval_ids") if isinstance(trace.get("referenced_approval_ids"), list) else [])
                if approval_id is not None
            }
        )
        latest_runtime_trace = runtime_traces[-1] if runtime_traces else {}
        latest_approval_queue_before = _int_trace(latest_runtime_trace.get("approval_queue_before"))
        latest_approval_queue_after = _int_trace(latest_runtime_trace.get("approval_queue_after"), latest_approval_queue_before)
        latest_approval_queue_delta = _int_trace(latest_runtime_trace.get("approval_queue_delta"))

        meta_runs = [row for row in rows if str(row["tool_name"]) in AFTER_ACTION_META_TOOLS]
        action_runs = [row for row in rows if str(row["tool_name"]) not in AFTER_ACTION_META_TOOLS]
        try:
            active_approval_ids = {
                int(row["id"])
                for row in store.list_pending_approvals(limit=MAX_AUDIT_LIMIT)
                if isinstance(row["id"], int) and not isinstance(row["id"], bool)
            }
        except Exception:
            active_approval_ids = set()

        def _is_active_approval_hold(row: Any) -> bool:
            metadata = _row_metadata(row)
            if not _is_approval_hold_metadata(metadata):
                return False
            try:
                approval_id = row["approval_id"]
            except Exception:
                return False
            return (
                isinstance(approval_id, int)
                and not isinstance(approval_id, bool)
                and approval_id in active_approval_ids
            )

        active_approval_holds = [row for row in action_runs if _is_active_approval_hold(row)]
        failed_or_blocked = [
            row
            for row in action_runs
            if (
                not _is_active_approval_hold(row)
                and (
                    not bool(row["ok"])
                    or _run_approval_evidence(row, approval_proof_cache).outcome_unknown
                )
            )
        ]
        risky_runs = [row for row in action_runs if _risk_needs_approval(row["risk"])]
        risky_approval_problems = [
            row
            for row in risky_runs
            if _run_approval_evidence(row, approval_proof_cache).approval_problem
        ]
        verification_runs = [
            row
            for row in rows
            if str(row["tool_name"]) in {
                "verification_receipt",
                "runtime_trace_receipt",
                "execution_audit_gate",
                "execution_recovery_packet",
                "after_action_learning_packet",
                "execution_health_report",
            }
        ]
        learning_runs = [row for row in rows if str(row["tool_name"]) == "after_action_learning_packet"]
        recovery_runs = [row for row in rows if str(row["tool_name"]) == "execution_recovery_packet"]
        failure_tools = Counter(str(row["tool_name"]) for row in failed_or_blocked)
        repeated_failure_tools = [
            {"tool": tool_name, "count": count}
            for tool_name, count in failure_tools.most_common()
            if count > 1
        ]
        failure_promotion_queue: list[str] = []
        if repeated_failure_tools:
            first_repeated_tool = repeated_failure_tools[0]["tool"]
            failure_promotion_queue = [
                f"failure to test preview: repeated {first_repeated_tool} failures",
                f"failure promotion packet: {first_repeated_tool}",
                f"failure implementation packet: {first_repeated_tool}",
                f"failure apply contract: {first_repeated_tool}",
                "learning review",
            ]
        blocker_categories: list[str] = []
        if risky_approval_problems:
            blocker_categories.append("approval_chain")
        if failed_or_blocked:
            blocker_categories.append("failed_or_blocked")
        if repeated_failure_tools:
            blocker_categories.append("repeated_failure")
        if not verification_runs:
            blocker_categories.append("verification_gap")
        review_required = bool(failed_or_blocked or risky_approval_problems or repeated_failure_tools)
        if risky_approval_problems:
            verdict = "APPROVAL_REVIEW_REQUIRED"
        elif repeated_failure_tools:
            verdict = "REPEATED_FAILURE_REVIEW_REQUIRED"
        elif failed_or_blocked:
            verdict = "RECOVERY_REVIEW_REQUIRED"
        elif not verification_runs:
            verdict = "NEEDS_VERIFICATION_PACKET"
        else:
            verdict = "HEALTHY_WITH_AUDIT_EVIDENCE"

        newest_problem = (risky_approval_problems or failed_or_blocked or [None])[0]
        if newest_problem is not None:
            next_command = f"execution recovery packet {newest_problem['id']}"
        elif repeated_failure_tools:
            next_command = f"failure to test preview: repeated {repeated_failure_tools[0]['tool']} failures"
        elif not verification_runs:
            next_command = "verification receipt latest"
        else:
            next_command = "after-action learning packet"
        next_commands: list[str] = []
        for row in risky_approval_problems[:3]:
            command = f"execution recovery packet {row['id']}"
            if command not in next_commands:
                next_commands.append(command)
        for row in failed_or_blocked[:3]:
            command = f"execution recovery packet {row['id']}"
            if command not in next_commands:
                next_commands.append(command)
        for item in repeated_failure_tools[:3]:
            command = f"failure to test preview: repeated {item['tool']} failures"
            if command not in next_commands:
                next_commands.append(command)
        for command in failure_promotion_queue:
            if command not in next_commands:
                next_commands.append(command)
        if not verification_runs and "verification receipt latest" not in next_commands:
            next_commands.append("verification receipt latest")
        if "execution audit gate" not in next_commands:
            next_commands.append("execution audit gate")
        if "after-action learning packet" not in next_commands:
            next_commands.append("after-action learning packet")
        verification_coverage_state = "present" if verification_runs else "missing"
        learning_target = newest_problem if newest_problem is not None else (action_runs[0] if action_runs else None)
        learning_target_run_id = int(learning_target["id"]) if learning_target is not None else None
        learning_target_tool = str(learning_target["tool_name"]) if learning_target is not None else ""
        learning_target_metadata = _row_metadata(learning_target) if learning_target is not None else {}
        learning_target_toolset = _short(learning_target_metadata.get("toolset"), limit=80)
        verification_handoff_command = (
            f"verification receipt {learning_target_run_id}" if learning_target_run_id is not None else "verification receipt latest"
        )
        learning_handoff_command = (
            f"after-action learning packet {learning_target_run_id}" if learning_target_run_id is not None else "after-action learning packet"
        )
        learning_closure_command = (
            f"execution learning closure {learning_target_run_id}" if learning_target_run_id is not None else "execution learning closure"
        )
        recovery_handoff_command = (
            f"execution recovery packet {learning_target_run_id}" if learning_target_run_id is not None else ""
        )
        target_verification_runs = [
            row
            for row in verification_runs
            if str(row["tool_name"]) == "verification_receipt"
            and bool(row["ok"])
            and _run_approval_evidence(row, approval_proof_cache).approval_linked
            and _row_metadata(row).get("run_id") == learning_target_run_id
            and (
                learning_target is None
                or not _run_approval_evidence(learning_target, approval_proof_cache).approval_problem
            )
        ]
        target_recovery_runs = [
            row
            for row in recovery_runs
            if bool(row["ok"])
            and _run_approval_evidence(row, approval_proof_cache).approval_linked
            and _row_metadata(row).get("run_id") == learning_target_run_id
            and (
                learning_target is None
                or not _run_approval_evidence(learning_target, approval_proof_cache).approval_problem
            )
        ]
        target_learning_runs = [
            row
            for row in learning_runs
            if bool(row["ok"])
            and _run_approval_evidence(row, approval_proof_cache).approval_linked
            and _row_metadata(row).get("run_id") == learning_target_run_id
            and (
                learning_target is None
                or not _run_approval_evidence(learning_target, approval_proof_cache).approval_problem
            )
        ]
        recovery_closure_missing: list[str] = []
        recovery_closure_required_commands: list[str] = []
        if learning_target_run_id is not None and newest_problem is not None:
            if not target_verification_runs:
                recovery_closure_missing.append("target_verification_receipt")
                recovery_closure_required_commands.append(verification_handoff_command)
            if not target_recovery_runs:
                recovery_closure_missing.append("target_recovery_packet")
                recovery_closure_required_commands.append(recovery_handoff_command)
            if not target_learning_runs:
                recovery_closure_missing.append("target_after_action_learning_packet")
                recovery_closure_required_commands.append(learning_handoff_command)
                recovery_closure_required_commands.append(learning_closure_command)
            if risky_approval_problems:
                recovery_closure_missing.append("approval_chain_proof")
                recovery_closure_required_commands.append("approval history")
        if repeated_failure_tools:
            for command in failure_promotion_queue:
                if command not in recovery_closure_required_commands:
                    recovery_closure_required_commands.append(command)
        if newest_problem is None and not repeated_failure_tools:
            recovery_closure_state = "not_needed"
        elif recovery_closure_missing:
            recovery_closure_state = "blocked_missing_" + "_and_".join(recovery_closure_missing)
        elif repeated_failure_tools:
            recovery_closure_state = "blocked_repeated_failure_promotion_required"
        else:
            recovery_closure_state = "ready_for_operator_retry_review"
        recovery_closure_ready_to_retry = recovery_closure_state == "ready_for_operator_retry_review"
        recovery_closure_blocks_auto_execution = recovery_closure_state not in {"not_needed", "ready_for_operator_retry_review"}
        recovery_closure_blocks_completion_claim = recovery_closure_state != "not_needed"
        for command in [verification_handoff_command, recovery_handoff_command, learning_handoff_command, learning_closure_command]:
            if command and command not in next_commands:
                next_commands.append(command)
        for command in recovery_closure_required_commands:
            if command and command not in next_commands:
                next_commands.append(command)
        if recovery_closure_blocks_auto_execution and recovery_closure_required_commands:
            ordered_next_commands: list[str] = []
            for command in recovery_closure_required_commands:
                if command and command not in ordered_next_commands:
                    ordered_next_commands.append(command)
            for command in next_commands:
                if command not in ordered_next_commands:
                    ordered_next_commands.append(command)
            next_commands = ordered_next_commands
            next_command = recovery_closure_required_commands[0]
        execution_proof_queue = _ordered_commands(next_commands)
        next_proof_command = execution_proof_queue[0] if execution_proof_queue else next_command
        learning_followup_command = learning_handoff_command
        if next_proof_command:
            next_command = next_proof_command
        execution_health_handoff = {
            "source": "execution_health_report",
            "verdict": verdict,
            "review_required": review_required,
            "safe_to_continue": not review_required,
            "inspected_runs": len(rows),
            "action_runs": len(action_runs),
            "active_approval_holds": len(active_approval_holds),
            "failed_or_blocked_runs": len(failed_or_blocked),
            "risky_approval_problems": len(risky_approval_problems),
            "verification_coverage_state": verification_coverage_state,
            "blocker_categories": blocker_categories,
            "blocker_count": len(blocker_categories),
            "next_command": next_command,
            "next_commands": next_commands,
            "next_command_count": len(next_commands),
            "proof_queue": execution_proof_queue,
            "proof_queue_count": len(execution_proof_queue),
            "next_required_command": next_proof_command,
            "next_proof_command": next_proof_command,
            "learning_followup_command": learning_followup_command,
            "learning_target": {
                "run_id": learning_target_run_id,
                "tool": learning_target_tool,
                "toolset": learning_target_toolset,
                "verification_handoff_command": verification_handoff_command,
                "recovery_handoff_command": recovery_handoff_command,
                "learning_closure_command": learning_closure_command,
                "learning_handoff_command": learning_handoff_command,
            },
            "recovery_closure": {
                "state": recovery_closure_state,
                "missing": recovery_closure_missing,
                "missing_count": len(recovery_closure_missing),
                "proof_queue": recovery_closure_required_commands,
                "proof_queue_count": len(recovery_closure_required_commands),
                "next_proof_command": recovery_closure_required_commands[0] if recovery_closure_required_commands else "",
                "ready_to_retry": recovery_closure_ready_to_retry,
                "blocks_auto_execution": recovery_closure_blocks_auto_execution,
                "blocks_completion_claim": recovery_closure_blocks_completion_claim,
                "target_run_id": learning_target_run_id,
                "target_tool_name": learning_target_tool,
                "target_toolset": learning_target_toolset,
                "target_verification_receipts": len(target_verification_runs),
                "target_recovery_packets": len(target_recovery_runs),
                "target_after_action_learning_packets": len(target_learning_runs),
            },
            "failure_promotion": {
                "repeated_failure_tools": repeated_failure_tools,
                "repeated_failure_count": sum(item["count"] for item in repeated_failure_tools),
                "queue": failure_promotion_queue,
                "queue_count": len(failure_promotion_queue),
                "promotion_command": failure_promotion_queue[1] if len(failure_promotion_queue) > 1 else "",
                "implementation_command": failure_promotion_queue[2] if len(failure_promotion_queue) > 2 else "",
                "apply_contract_command": failure_promotion_queue[3] if len(failure_promotion_queue) > 3 else "",
            },
            "review_only": True,
            "draft_only": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "calls_model": False,
            "executes_tools": False,
            "writes_files": False,
            "reads_personal_data": False,
            "external_side_effect": False,
            "controls_computer": False,
            "queues_approval": False,
        }

        lines = [
            "Jarvis execution health report:",
            "This is the aggregate runtime health layer. It summarizes recent action runs, verification coverage, repeated failures, approval-chain risk, and the next safe audit command without running anything.",
            "",
            f"Verdict: {verdict}",
            f"Inspected runs: {len(rows)}",
            f"Action runs: {len(action_runs)}",
            f"Meta audit runs: {len(meta_runs)}",
            f"Active approval holds: {len(active_approval_holds)}",
            f"Failed or blocked action runs: {len(failed_or_blocked)}",
            f"Risk-gated action runs: {len(risky_runs)}",
            f"Risky action approval problems: {len(risky_approval_problems)}",
            f"Verification/audit packets: {len(verification_runs)}",
            f"Recovery packets: {len(recovery_runs)}",
            f"After-action learning packets: {len(learning_runs)}",
            f"Blocker categories: {', '.join(blocker_categories) if blocker_categories else 'none'}",
            f"Verification coverage: {verification_coverage_state}",
            "",
            "Runtime queue ledger:",
            f"- runtime traces inspected: {len(runtime_traces)}",
            f"- latest queue before: {latest_approval_queue_before}",
            f"- latest queue after: {latest_approval_queue_after}",
            f"- latest queue delta: {latest_approval_queue_delta}",
            f"- total queue delta: {runtime_queue_delta_total}",
            f"- queued approval ids: {', '.join(str(item) for item in runtime_queued_approval_ids) if runtime_queued_approval_ids else 'none'}",
            f"- new approval ids: {', '.join(str(item) for item in runtime_new_approval_ids) if runtime_new_approval_ids else 'none'}",
            f"- reused approval ids: {', '.join(str(item) for item in runtime_reused_approval_ids) if runtime_reused_approval_ids else 'none'}",
            f"- referenced approval ids: {', '.join(str(item) for item in runtime_referenced_approval_ids) if runtime_referenced_approval_ids else 'none'}",
            "",
            "Repeated failure tools:",
        ]
        if repeated_failure_tools:
            for item in repeated_failure_tools[:5]:
                lines.append(f"- {item['tool']}: {item['count']} failed/blocked runs")
        else:
            lines.append("- none detected in inspected window")

        lines.extend(["", "Newest action signals:"])
        for row in action_runs[:5]:
            if _is_active_approval_hold(row):
                status = "approval held"
            else:
                status = "ok" if bool(row["ok"]) else "failed/blocked"
            approved = (
                "approved"
                if bool(row["approved"])
                and _run_approval_evidence(row, approval_proof_cache).approval_linked
                else "not approved"
            )
            approval_id = row["approval_id"] if "approval_id" in row.keys() else None
            approval_text = f", approval #{approval_id}" if approval_id else ""
            lines.append(f"- #{row['id']} {row['tool_name']} [{row['risk']}, {status}, {approved}{approval_text}]")
        if not action_runs:
            lines.append("- none; recent rows are audit/meta packets only")

        lines.extend(
            [
                "",
                "Health interpretation:",
                "- Failed/blocked action runs require recovery before Jarvis treats the work as done.",
                "- Risky action runs need linked approval evidence before counting as trusted execution.",
                "- Repeated failures should become a regression-test or route-fix candidate before increasing autonomy.",
                "- Verification and after-action learning packets are positive evidence, but they do not replace real outcome proof.",
                "- Operator limits still apply: the operator's explicit stop times, work windows, pause commands, and newer instructions override priority goals.",
                "",
                "Verification-to-learning handoff:",
                f"- target run: #{learning_target_run_id} `{learning_target_tool}`" if learning_target_run_id is not None else "- target run: none yet",
                f"- target toolset: {learning_target_toolset or 'unknown'}",
                f"- verify: `{verification_handoff_command}`",
                f"- recover: `{recovery_handoff_command}`" if recovery_handoff_command else "- recover: none needed yet",
                f"- learn: `{learning_handoff_command}`",
                f"- close learning gate: `{learning_closure_command}`",
                "- Order: verify stored evidence first, inspect recovery if needed, capture after-action learning evidence, then close the execution learning gate before promoting tests, memory, tasks, or skills.",
                f"- command-first rule: the top-level next command is `{next_command}` because it is the first execution proof command; learning remains `{learning_followup_command}`.",
                "",
                "Recovery closure gate:",
                f"- state: {recovery_closure_state}",
                f"- ready for operator retry review: {'yes' if recovery_closure_ready_to_retry else 'no'}",
                f"- blocks auto-run: {'yes' if recovery_closure_blocks_auto_execution else 'no'}",
                f"- target verification receipts: {len(target_verification_runs)}",
                f"- target recovery packets: {len(target_recovery_runs)}",
                f"- target learning packets: {len(target_learning_runs)}",
                f"- missing: {', '.join(recovery_closure_missing) if recovery_closure_missing else 'none'}",
                "- A retry is not trusted until target verification, recovery, learning closure, after-action learning, and any approval-chain proof are present.",
                "",
                "Failure promotion queue:",
            ]
        )
        if failure_promotion_queue:
            lines.extend(f"- `{command}`" for command in failure_promotion_queue)
        else:
            lines.append("- none; no repeated failure cluster is ready for promotion")
        lines.extend(
            [
                "",
                "Next safe command:",
                f"- `{next_command}`",
                "",
                "Execution proof queue:",
                f"- next required command: `{next_proof_command}`" if next_proof_command else "- next required command: none",
                f"- proof queue count: {len(execution_proof_queue)}",
                "- proof queue: " + ", ".join(f"`{command}`" for command in execution_proof_queue) if execution_proof_queue else "- proof queue: none",
                "",
                "Recovery queue:",
                *[f"- `{command}`" for command in next_commands],
                "",
                "Boundary:",
                "- This report is read-only. It does not call models, execute tools, approve requests, dismiss approvals, retry tools, write files, read private data, control the computer, call external services, complete tasks, or queue approvals.",
            ]
        )

        return ToolResult(
            "execution_health_report",
            True,
            "\n".join(lines),
            _safe_metadata(
                verdict=verdict,
                inspected_runs=len(rows),
                action_runs=len(action_runs),
                meta_runs=len(meta_runs),
                active_approval_holds=len(active_approval_holds),
                failed_or_blocked_runs=len(failed_or_blocked),
                risky_runs=len(risky_runs),
                risky_approval_problems=len(risky_approval_problems),
                verification_runs=len(verification_runs),
                recovery_runs=len(recovery_runs),
                after_action_learning_runs=len(learning_runs),
                runtime_trace_count=len(runtime_traces),
                runtime_queue_delta_total=runtime_queue_delta_total,
                runtime_queued_approval_ids=runtime_queued_approval_ids,
                runtime_new_approval_ids=runtime_new_approval_ids,
                runtime_reused_approval_ids=runtime_reused_approval_ids,
                runtime_referenced_approval_ids=runtime_referenced_approval_ids,
                latest_approval_queue_before=latest_approval_queue_before,
                latest_approval_queue_after=latest_approval_queue_after,
                latest_approval_queue_delta=latest_approval_queue_delta,
                repeated_failure_tools=repeated_failure_tools,
                repeated_failure_count=sum(item["count"] for item in repeated_failure_tools),
                failure_promotion_queue=failure_promotion_queue,
                failure_promotion_queue_count=len(failure_promotion_queue),
                failure_promotion_command=failure_promotion_queue[1] if len(failure_promotion_queue) > 1 else "",
                failure_implementation_command=failure_promotion_queue[2] if len(failure_promotion_queue) > 2 else "",
                failure_apply_contract_command=failure_promotion_queue[3] if len(failure_promotion_queue) > 3 else "",
                blocker_categories=blocker_categories,
                blocker_count=len(blocker_categories),
                approval_chain_gap_count=len(risky_approval_problems),
                verification_coverage_state=verification_coverage_state,
                safe_to_continue=not review_required,
                review_required=review_required,
                newest_run_id=int(rows[0]["id"]),
                newest_problem_run_id=int(newest_problem["id"]) if newest_problem is not None else None,
                learning_target_run_id=learning_target_run_id,
                learning_target_tool=learning_target_tool,
                learning_target_toolset=learning_target_toolset,
                verification_handoff_command=verification_handoff_command,
                recovery_handoff_command=recovery_handoff_command,
                learning_handoff_command=learning_handoff_command,
                learning_followup_command=learning_followup_command,
                learning_closure_command=learning_closure_command,
                execution_learning_closure_command=learning_closure_command,
                target_verification_receipts=len(target_verification_runs),
                target_recovery_packets=len(target_recovery_runs),
                target_after_action_learning_packets=len(target_learning_runs),
                recovery_closure_state=recovery_closure_state,
                recovery_closure_missing=recovery_closure_missing,
                recovery_closure_missing_count=len(recovery_closure_missing),
                recovery_closure_required_commands=recovery_closure_required_commands,
                recovery_closure_next_required_command=recovery_closure_required_commands[0] if recovery_closure_required_commands else "",
                recovery_closure_proof_queue=recovery_closure_required_commands,
                recovery_closure_proof_queue_count=len(recovery_closure_required_commands),
                recovery_closure_next_proof_command=recovery_closure_required_commands[0] if recovery_closure_required_commands else "",
                recovery_closure_ready_to_retry=recovery_closure_ready_to_retry,
                recovery_closure_blocks_auto_execution=recovery_closure_blocks_auto_execution,
                recovery_closure_blocks_completion_claim=recovery_closure_blocks_completion_claim,
                recovery_closure_target_run_id=learning_target_run_id,
                recovery_closure_target_tool_name=learning_target_tool,
                recovery_closure_target_toolset=learning_target_toolset,
                recovery_closure_target_verification_receipts=len(target_verification_runs),
                recovery_closure_target_recovery_packets=len(target_recovery_runs),
                recovery_closure_target_after_action_learning_packets=len(target_learning_runs),
                next_command=next_command,
                next_commands=next_commands,
                next_command_count=len(next_commands),
                execution_proof_queue=execution_proof_queue,
                execution_proof_queue_count=len(execution_proof_queue),
                execution_next_required_command=next_proof_command,
                execution_next_proof_command=next_proof_command,
                proof_queue=execution_proof_queue,
                proof_queue_count=len(execution_proof_queue),
                next_required_command=next_proof_command,
                next_proof_command=next_proof_command,
                execution_health_handoff=execution_health_handoff,
                execution_health_review_only=True,
                execution_health_draft_only=True,
                execution_health_loads_without_execution=True,
                execution_health_authorizes_execution=False,
                execution_health_authorizes_completion_claim=False,
                execution_health_approval_granted=False,
            ),
        )

    def execution_learning_closure_packet(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 80)
        run_id_raw = _first_present(args, ("run_id", "id", "tool_run_id"), "")
        health = execution_health_report({"limit": limit})
        health_metadata = dict(health.metadata)
        approval_proof_cache: dict[int, Any] = {}

        target = None
        if _has_explicit_value(run_id_raw):
            try:
                target_run_id = int(run_id_raw)
            except (TypeError, ValueError):
                return _audit_input_failure(
                    "execution_learning_closure_packet",
                    "Tool run id must be a number when supplied.",
                    _safe_metadata(verdict="INVALID_ID", run_id=_short_raw(run_id_raw), raw_run_id=_short_raw(run_id_raw), inspected_runs=health_metadata.get("inspected_runs", 0)),
                )
            if target_run_id <= 0:
                return _audit_input_failure(
                    "execution_learning_closure_packet",
                    "Tool run id must be a positive number when supplied.",
                    _safe_metadata(verdict="INVALID_ID", run_id=_short_raw(run_id_raw), raw_run_id=_short_raw(run_id_raw), inspected_runs=health_metadata.get("inspected_runs", 0)),
                )
            target = store.get_tool_run(target_run_id)
            if target is None:
                safe_run_id = _short_raw(run_id_raw)
                return _audit_not_found_failure(
                    "execution_learning_closure_packet",
                    f"Tool run #{safe_run_id} was not found in the audit log. Run `recent tool runs` to "
                    "refresh valid IDs, then retry `execution learning closure` for the latest run.",
                    _lookup_recovery_metadata(
                        "recent tool runs",
                        "execution learning closure",
                        verdict="MISSING_RUN",
                        run_id=safe_run_id,
                        inspected_runs=health_metadata.get("inspected_runs", 0),
                    ),
                )
        else:
            target_run_id = health_metadata.get("learning_target_run_id") or health_metadata.get("recovery_closure_target_run_id")
            if target_run_id is not None:
                target = store.get_tool_run(int(target_run_id))

        target_tool = (
            _safe_text(target["tool_name"])
            if target is not None
            else _first_safe_text(
                health_metadata.get("learning_target_tool"),
                health_metadata.get("recovery_closure_target_tool_name"),
            )
        )
        target_metadata = _row_metadata(target) if target is not None else {}
        target_toolset = _short(
            _first_safe_text(
                target_metadata.get("toolset"),
                health_metadata.get("learning_target_toolset"),
                health_metadata.get("recovery_closure_target_toolset"),
            ),
            limit=80,
        )
        target_risk = str(target["risk"]) if target is not None else ""
        target_ok = bool(target["ok"]) if target is not None else None
        target_approval_proven = (
            _run_approval_evidence(target, approval_proof_cache).approval_linked
            if target is not None
            else False
        )
        target_approved = (
            bool(target["approved"]) and target_approval_proven
            if target is not None
            else None
        )
        target_approval_id = target["approval_id"] if target is not None and "approval_id" in target.keys() else None
        target_run_id = int(target["id"]) if target is not None else (int(target_run_id) if target_run_id is not None else None)

        target_learning_packets = _metadata_int(
            health_metadata.get("target_after_action_learning_packets")
            or health_metadata.get("recovery_closure_target_after_action_learning_packets")
        )
        target_verification_receipts = _metadata_int(
            health_metadata.get("target_verification_receipts")
            or health_metadata.get("recovery_closure_target_verification_receipts")
        )
        target_recovery_packets = _metadata_int(
            health_metadata.get("target_recovery_packets")
            or health_metadata.get("recovery_closure_target_recovery_packets")
        )
        if target is not None and not target_approval_proven:
            target_learning_packets = 0
            target_verification_receipts = 0
            target_recovery_packets = 0
        repeated_failure_count = _metadata_int(health_metadata.get("repeated_failure_count"))
        failure_promotion_queue = list(health_metadata.get("failure_promotion_queue") or [])
        missing: list[str] = []
        required_commands: list[str] = []
        if target_run_id is None:
            missing.append("learning_target_run")
            required_commands.append("execution health report")
        else:
            if target_verification_receipts < 1:
                missing.append("target_verification_receipt")
                required_commands.append(f"verification receipt {target_run_id}")
            if target is not None and not bool(target["ok"]) and target_recovery_packets < 1:
                missing.append("target_recovery_packet")
                required_commands.append(f"execution recovery packet {target_run_id}")
            if target_learning_packets < 1:
                missing.append("target_after_action_learning_packet")
                required_commands.append(f"after-action learning packet {target_run_id}")
                required_commands.append(f"execution learning closure {target_run_id}")
        if repeated_failure_count > 0:
            missing.append("repeated_failure_promotion_review")
            for command in failure_promotion_queue:
                if command not in required_commands:
                    required_commands.append(command)

        proof_artifacts = {
            "verification": {
                "present": target_verification_receipts > 0,
                "count": target_verification_receipts,
                "command": f"verification receipt {target_run_id}" if target_run_id is not None else "verification receipt latest",
                "missing_key": "target_verification_receipt",
            },
            "recovery": {
                "present": target is None or bool(target["ok"]) or target_recovery_packets > 0,
                "count": target_recovery_packets,
                "command": f"execution recovery packet {target_run_id}" if target_run_id is not None else "execution recovery packet",
                "missing_key": "target_recovery_packet",
            },
            "after_action_learning": {
                "present": target_learning_packets > 0,
                "count": target_learning_packets,
                "command": f"after-action learning packet {target_run_id}" if target_run_id is not None else "after-action learning packet",
                "missing_key": "target_after_action_learning_packet",
            },
            "execution_learning_closure": {
                "present": target_learning_packets > 0,
                "count": target_learning_packets,
                "command": f"execution learning closure {target_run_id}" if target_run_id is not None else "execution learning closure",
                "missing_key": "target_after_action_learning_packet",
            },
            "repeated_failure_promotion": {
                "present": repeated_failure_count == 0,
                "count": 0 if repeated_failure_count > 0 else 1,
                "command": failure_promotion_queue[1] if len(failure_promotion_queue) > 1 else "failure promotion packet",
                "missing_key": "repeated_failure_promotion_review",
            },
        }

        if target_run_id is None:
            verdict = "NO_LEARNING_TARGET"
        elif missing:
            verdict = "LEARNING_CLOSURE_INCOMPLETE"
        else:
            verdict = "LEARNING_CLOSURE_READY"

        phase_rows = [
            {
                "phase": "target run",
                "state": "ready" if target_run_id is not None else "missing",
                "command": f"recent tool runs" if target_run_id is None else f"verification receipt {target_run_id}",
                "proof_count": 1 if target_run_id is not None else 0,
            },
            {
                "phase": "verification",
                "state": "ready" if target_verification_receipts > 0 else "missing",
                "command": f"verification receipt {target_run_id}" if target_run_id is not None else "verification receipt latest",
                "proof_count": target_verification_receipts,
            },
            {
                "phase": "recovery",
                "state": "ready" if target is None or bool(target["ok"]) or target_recovery_packets > 0 else "missing",
                "command": f"execution recovery packet {target_run_id}" if target_run_id is not None else "execution recovery packet",
                "proof_count": target_recovery_packets,
            },
            {
                "phase": "after-action learning",
                "state": "ready" if target_learning_packets > 0 else "missing",
                "command": f"after-action learning packet {target_run_id}" if target_run_id is not None else "after-action learning packet",
                "proof_count": target_learning_packets,
            },
            {
                "phase": "execution learning closure",
                "state": "ready" if target_learning_packets > 0 else "missing",
                "command": f"execution learning closure {target_run_id}" if target_run_id is not None else "execution learning closure",
                "proof_count": target_learning_packets,
            },
            {
                "phase": "repeated failure promotion",
                "state": "missing" if repeated_failure_count > 0 else "ready",
                "command": failure_promotion_queue[1] if len(failure_promotion_queue) > 1 else "failure promotion packet",
                "proof_count": 0 if repeated_failure_count > 0 else 1,
            },
        ]
        proof_queue = _ordered_commands(required_commands)
        next_command = proof_queue[0] if proof_queue else "learning review"
        next_required_command = next_command
        actionable_proof_queue = _actionable_learning_commands(proof_queue, target_run_id, missing)
        actionable_next_command = actionable_proof_queue[0] if actionable_proof_queue else next_command
        actionable_next_required_command = actionable_next_command
        execution_learning_closure_handoff = {
            "source": "execution_learning_closure_packet",
            "source_tool": "execution_health_report",
            "verdict": verdict,
            "health_verdict": health_metadata.get("verdict", ""),
            "target": {
                "run_id": target_run_id,
                "tool_name": target_tool,
                "toolset": target_toolset,
                "risk": target_risk,
                "ok": target_ok,
                "approved": target_approved,
                "approval_id": target_approval_id,
            },
            "proof_artifacts": proof_artifacts,
            "proof_artifact_count": len(proof_artifacts),
            "phase_rows": phase_rows,
            "phase_row_count": len(phase_rows),
            "missing": missing,
            "missing_count": len(missing),
            "required_commands": proof_queue,
            "required_command_count": len(proof_queue),
            "proof_queue": proof_queue,
            "proof_queue_count": len(proof_queue),
            "next_command": next_command,
            "next_required_command": next_required_command,
            "next_proof_command": next_command,
            "execution_learning_closure_next_required_command": next_required_command,
            "execution_learning_closure_next_proof_command": next_command,
            "actionable_required_commands": actionable_proof_queue,
            "actionable_required_command_count": len(actionable_proof_queue),
            "actionable_proof_queue": actionable_proof_queue,
            "actionable_proof_queue_count": len(actionable_proof_queue),
            "actionable_next_command": actionable_next_command,
            "actionable_next_required_command": actionable_next_required_command,
            "actionable_next_proof_command": actionable_next_command,
            "execution_learning_closure_actionable_next_required_command": actionable_next_required_command,
            "execution_learning_closure_actionable_next_proof_command": actionable_next_command,
            "learning_closure_ready": verdict == "LEARNING_CLOSURE_READY",
            "blocks_completion_claim": verdict != "LEARNING_CLOSURE_READY",
            "review_only": True,
            "draft_only": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "calls_model": False,
            "executes_tools": False,
            "writes_files": False,
            "reads_personal_data": False,
            "external_side_effect": False,
            "controls_computer": False,
            "queues_approval": False,
        }

        lines = [
            "Jarvis execution learning closure packet:",
            "This is the completion gate for the learning loop after real or blocked execution. It is read-only and checks whether verification, recovery, after-action learning, and repeated-failure promotion proof exist before Jarvis claims learning debt is closed.",
            "",
            f"Verdict: {verdict}",
            f"Health verdict: {health_metadata.get('verdict', 'unknown')}",
            f"Target run: #{target_run_id} `{target_tool}`" if target_run_id is not None else "Target run: none",
            f"Target risk: {target_risk or 'unknown'}",
            f"Target toolset: {target_toolset or 'unknown'}",
            f"Target result: {'ok' if target_ok else 'failed/blocked' if target_ok is False else 'unknown'}",
            f"Target approved: {'yes' if target_approved else 'no' if target_approved is False else 'unknown'}",
            f"Linked approval id: #{target_approval_id}" if target_approval_id else "Linked approval id: none",
            "",
            "Learning closure checklist:",
        ]
        for row in phase_rows:
            lines.append(f"- {row['phase']}: {row['state']}; proof count {row['proof_count']}; command `{row['command']}`")
        lines.extend(["", "Target proof artifact ledger:"])
        for name, artifact in proof_artifacts.items():
            state = "present" if artifact["present"] else "missing"
            lines.append(f"- {name}: {state}; count {artifact['count']}; command `{artifact['command']}`")
        lines.extend(
            [
                "",
                "Missing proof:",
                f"- {', '.join(missing) if missing else 'none'}",
                "",
                "Required learning proof queue:",
            ]
        )
        if proof_queue:
            lines.extend(f"- `{command}`" for command in proof_queue)
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "Actionable evidence queue:",
                f"- next evidence command: `{actionable_next_command}`" if actionable_next_command else "- next evidence command: none",
                *[f"- `{command}`" for command in actionable_proof_queue],
                "",
                "Closure rules:",
                "- Verify stored outcome evidence before learning from the run.",
                "- Inspect recovery for failed or blocked execution before retry or promotion.",
                "- Open after-action learning before turning a run into memory, a task, a test, a preference, or a skill.",
                "- Repeated failures require failure-promotion review before Jarvis increases autonomy around that route.",
                "- the operator's explicit stop times, work windows, pause commands, and newer instructions override learning-promotion momentum.",
                "",
                "Next safe command:",
                f"- `{next_command}`",
                "",
                "Boundary:",
                "- This packet is read-only. It does not call models, execute tools, save memories, create tasks, write tests, edit files, approve requests, dismiss approvals, retry tools, read private data, control the computer, call external services, complete tasks, or queue approvals.",
            ]
        )
        return ToolResult(
            "execution_learning_closure_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                verdict=verdict,
                health_verdict=health_metadata.get("verdict", ""),
                inspected_runs=health_metadata.get("inspected_runs", 0),
                run_id=target_run_id,
                target_run_id=target_run_id,
                target_tool_name=target_tool,
                target_toolset=target_toolset,
                target_risk=target_risk,
                target_ok=target_ok,
                target_approved=target_approved,
                target_approval_id=target_approval_id,
                target_verification_receipts=target_verification_receipts,
                target_recovery_packets=target_recovery_packets,
                target_after_action_learning_packets=target_learning_packets,
                target_proof_artifacts=proof_artifacts,
                target_proof_artifact_count=len(proof_artifacts),
                target_verification_present=proof_artifacts["verification"]["present"],
                target_recovery_present=proof_artifacts["recovery"]["present"],
                target_after_action_learning_present=proof_artifacts["after_action_learning"]["present"],
                repeated_failure_promotion_present=proof_artifacts["repeated_failure_promotion"]["present"],
                repeated_failure_count=repeated_failure_count,
                failure_promotion_queue=failure_promotion_queue,
                failure_promotion_queue_count=len(failure_promotion_queue),
                missing=missing,
                missing_count=len(missing),
                required_commands=proof_queue,
                required_command_count=len(proof_queue),
                next_command=next_command,
                next_required_command=next_required_command,
                proof_queue=proof_queue,
                proof_queue_count=len(proof_queue),
                next_proof_command=next_command,
                execution_learning_closure_required_commands=proof_queue,
                execution_learning_closure_required_command_count=len(proof_queue),
                execution_learning_closure_proof_queue=proof_queue,
                execution_learning_closure_proof_queue_count=len(proof_queue),
                execution_learning_closure_next_required_command=next_required_command,
                execution_learning_closure_next_proof_command=next_command,
                actionable_required_commands=actionable_proof_queue,
                actionable_required_command_count=len(actionable_proof_queue),
                actionable_proof_queue=actionable_proof_queue,
                actionable_proof_queue_count=len(actionable_proof_queue),
                actionable_next_command=actionable_next_command,
                actionable_next_required_command=actionable_next_required_command,
                actionable_next_proof_command=actionable_next_command,
                execution_learning_closure_actionable_required_commands=actionable_proof_queue,
                execution_learning_closure_actionable_required_command_count=len(actionable_proof_queue),
                execution_learning_closure_actionable_proof_queue=actionable_proof_queue,
                execution_learning_closure_actionable_proof_queue_count=len(actionable_proof_queue),
                execution_learning_closure_actionable_next_required_command=actionable_next_required_command,
                execution_learning_closure_actionable_next_proof_command=actionable_next_command,
                next_evidence_command=actionable_next_command,
                learning_closure_state=verdict,
                learning_closure_ready=verdict == "LEARNING_CLOSURE_READY",
                learning_closure_blocks_completion_claim=verdict != "LEARNING_CLOSURE_READY",
                phase_rows=phase_rows,
                phase_row_count=len(phase_rows),
                source_tool="execution_health_report",
                execution_learning_closure_handoff=execution_learning_closure_handoff,
                execution_learning_closure_review_only=True,
                execution_learning_closure_draft_only=True,
                execution_learning_closure_loads_without_execution=True,
                execution_learning_closure_authorizes_execution=False,
                execution_learning_closure_authorizes_completion_claim=False,
                execution_learning_closure_approval_granted=False,
            ),
        )

    def recovery_closure_checklist(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 80)
        health = execution_health_report({"limit": limit})
        metadata = dict(health.metadata)
        target_run_id = metadata.get("recovery_closure_target_run_id")
        target_tool_name = _safe_text(metadata.get("recovery_closure_target_tool_name"))
        target_toolset = _short(
            _first_safe_text(
                metadata.get("recovery_closure_target_toolset"),
                metadata.get("learning_target_toolset"),
            ),
            limit=80,
        )
        missing = set(metadata.get("recovery_closure_missing") or [])
        required_commands = list(metadata.get("recovery_closure_required_commands") or metadata.get("next_commands") or [])
        phase_rows = [
            {
                "phase": "target verification",
                "state": "missing" if "target_verification_receipt" in missing else "ready",
                "command": f"verification receipt {target_run_id}" if target_run_id is not None else "verification receipt latest",
                "proof_count": _metadata_int(metadata.get("recovery_closure_target_verification_receipts")),
            },
            {
                "phase": "target recovery",
                "state": "missing" if "target_recovery_packet" in missing else "ready",
                "command": f"execution recovery packet {target_run_id}" if target_run_id is not None else "execution recovery packet",
                "proof_count": _metadata_int(metadata.get("recovery_closure_target_recovery_packets")),
            },
            {
                "phase": "execution learning closure",
                "state": "missing" if "target_after_action_learning_packet" in missing else "ready",
                "command": f"execution learning closure {target_run_id}" if target_run_id is not None else "execution learning closure",
                "proof_count": _metadata_int(metadata.get("recovery_closure_target_after_action_learning_packets")),
            },
            {
                "phase": "after-action learning",
                "state": "missing" if "target_after_action_learning_packet" in missing else "ready",
                "command": f"after-action learning packet {target_run_id}" if target_run_id is not None else "after-action learning packet",
                "proof_count": _metadata_int(metadata.get("recovery_closure_target_after_action_learning_packets")),
            },
            {
                "phase": "approval chain",
                "state": "missing" if "approval_chain_proof" in missing else "ready",
                "command": "approval history",
                "proof_count": 0 if "approval_chain_proof" in missing else 1,
            },
        ]
        if metadata.get("failure_promotion_queue"):
            phase_rows.append(
                {
                    "phase": "repeated failure promotion",
                    "state": "missing",
                    "command": _first_safe_text(metadata.get("failure_promotion_command"), default="failure promotion packet"),
                    "proof_count": 0,
                }
            )

        checklist_state = _first_safe_text(metadata.get("recovery_closure_state"), default="unknown")
        ready_to_retry = _metadata_bool(metadata.get("recovery_closure_ready_to_retry", False))
        blocks_auto_execution = _metadata_bool(metadata.get("recovery_closure_blocks_auto_execution", False))
        blocks_completion_claim = _metadata_bool(metadata.get("recovery_closure_blocks_completion_claim", False))
        next_command = _first_safe_text(
            metadata.get("recovery_closure_next_required_command"),
            metadata.get("next_command"),
            default="execution health report",
        )
        recovery_closure_handoff = {
            "source": "recovery_closure_checklist",
            "source_tool": "execution_health_report",
            "closure_state": checklist_state,
            "target_run_id": target_run_id,
            "target_tool_name": target_tool_name,
            "target_toolset": target_toolset,
            "ready_to_retry": ready_to_retry,
            "blocks_auto_execution": blocks_auto_execution,
            "blocks_completion_claim": blocks_completion_claim,
            "missing": list(metadata.get("recovery_closure_missing") or []),
            "missing_count": _metadata_int(metadata.get("recovery_closure_missing_count")),
            "required_commands": required_commands,
            "required_command_count": len(required_commands),
            "next_command": next_command,
            "proof_queue": required_commands,
            "proof_queue_count": len(required_commands),
            "next_proof_command": next_command,
            "checklist_rows": phase_rows,
            "checklist_row_count": len(phase_rows),
            "target_verification_receipts": _metadata_int(metadata.get("recovery_closure_target_verification_receipts")),
            "target_recovery_packets": _metadata_int(metadata.get("recovery_closure_target_recovery_packets")),
            "target_after_action_learning_packets": _metadata_int(metadata.get("recovery_closure_target_after_action_learning_packets")),
            "failure_promotion_queue": list(metadata.get("failure_promotion_queue") or []),
            "failure_promotion_queue_count": _metadata_int(metadata.get("failure_promotion_queue_count")),
            "health_verdict": metadata.get("verdict", ""),
            "inspected_runs": metadata.get("inspected_runs", 0),
            "review_only": True,
            "draft_only": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "calls_model": False,
            "executes_tools": False,
            "writes_files": False,
            "reads_personal_data": False,
            "external_side_effect": False,
            "controls_computer": False,
            "queues_approval": False,
        }

        lines = [
            "Jarvis recovery closure checklist:",
            "This is the operator checklist for closing execution-health debt before Jarvis retries risky work or claims completion. It is read-only and reuses `execution health report` evidence.",
            "",
            f"Closure state: {checklist_state}",
            f"Target run: #{target_run_id} `{target_tool_name}`" if target_run_id is not None else "Target run: none",
            f"Target toolset: {target_toolset or 'unknown'}",
            f"Ready for retry review: {'yes' if ready_to_retry else 'no'}",
            f"Blocks auto-run: {'yes' if blocks_auto_execution else 'no'}",
            f"Blocks completion claim: {'yes' if blocks_completion_claim else 'no'}",
            f"Next closure command: `{next_command}`",
            "",
            "Checklist rows:",
        ]
        for row in phase_rows:
            lines.append(f"- {row['phase']}: {row['state']}; proof count {row['proof_count']}; command `{row['command']}`")
        lines.extend(
            [
                "",
                "Required closure queue:",
            ]
        )
        if required_commands:
            lines.extend(f"- `{command}`" for command in required_commands)
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "Rules:",
                "- Close target verification, recovery, learning, and approval-chain proof before treating a failed or blocked run as resolved.",
                "- Repeated failures should enter failure-promotion review before more autonomous retries.",
                "- This checklist does not execute, approve, dismiss, retry, write files, read private data, control the computer, call external services, or queue approvals.",
            ]
        )

        return ToolResult(
            "recovery_closure_checklist",
            True,
            "\n".join(lines),
            _safe_metadata(
                closure_state=checklist_state,
                recovery_closure_state=checklist_state,
                target_run_id=target_run_id,
                target_tool_name=target_tool_name,
                target_toolset=target_toolset,
                recovery_closure_target_run_id=target_run_id,
                recovery_closure_target_tool_name=target_tool_name,
                recovery_closure_target_toolset=target_toolset,
                ready_to_retry=ready_to_retry,
                blocks_auto_execution=blocks_auto_execution,
                blocks_completion_claim=blocks_completion_claim,
                missing=list(metadata.get("recovery_closure_missing") or []),
                missing_count=_metadata_int(metadata.get("recovery_closure_missing_count")),
                required_commands=required_commands,
                required_command_count=len(required_commands),
                next_command=next_command,
                proof_queue=required_commands,
                proof_queue_count=len(required_commands),
                next_proof_command=next_command,
                checklist_rows=phase_rows,
                checklist_row_count=len(phase_rows),
                target_verification_receipts=_metadata_int(metadata.get("recovery_closure_target_verification_receipts")),
                target_recovery_packets=_metadata_int(metadata.get("recovery_closure_target_recovery_packets")),
                target_after_action_learning_packets=_metadata_int(metadata.get("recovery_closure_target_after_action_learning_packets")),
                failure_promotion_queue=list(metadata.get("failure_promotion_queue") or []),
                failure_promotion_queue_count=_metadata_int(metadata.get("failure_promotion_queue_count")),
                health_verdict=metadata.get("verdict", ""),
                inspected_runs=metadata.get("inspected_runs", 0),
                source_tool="execution_health_report",
                recovery_closure_handoff=recovery_closure_handoff,
                recovery_closure_review_only=True,
                recovery_closure_draft_only=True,
                recovery_closure_loads_without_execution=True,
                recovery_closure_authorizes_execution=False,
                recovery_closure_authorizes_completion_claim=False,
                recovery_closure_approval_granted=False,
            ),
        )

    return recent_tool_runs, verification_receipt, runtime_trace_receipt, execution_audit_gate, execution_recovery_packet, after_action_learning_packet, execution_health_report, recovery_closure_checklist, execution_learning_closure_packet
