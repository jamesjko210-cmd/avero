from __future__ import annotations

import json
import re
from typing import Any, Callable

from jarvis_v2.agent.failure_guidance import (
    LOCAL_READ_INPUT_RECOVERY_ACTION,
    declare_retryable_local_read_failure,
)
from jarvis_v2.agent.types import ToolResult
from jarvis_v2.tools.audit_meta import RECOVERY_CLOSURE_META_TOOLS


_HINT_PATTERN_CACHE: dict[str, "re.Pattern[str]"] = {}


def _hint_matches(hint: str, text: str) -> bool:
    """Word-boundary match for a risk hint.

    Naive substring checks (`hint in text`) let short hints collide with
    unrelated words -- "app" (meant for "open the app") also matches inside
    "approve"/"approval", so every approve/dismiss command was self-blocking
    the moment one approval was already pending: the message itself contains
    the word "approval", got tagged as a "computer" risk, and the governor
    held it because a risky-looking request arrived while approvals were
    already queued. No approve/dismiss command could ever get through once
    the queue was non-empty. Word boundaries fix this without weakening
    detection of the intended whole-word hints.
    """
    pattern = _HINT_PATTERN_CACHE.get(hint)
    if pattern is None:
        pattern = re.compile(r"\b" + re.escape(hint) + r"\b")
        _HINT_PATTERN_CACHE[hint] = pattern
    return bool(pattern.search(text))


RISKY_HINTS = {
    "computer": ["computer", "screen", "click", "mouse", "keyboard", "type", "app", "desktop"],
    "shell/code": ["command", "terminal", "shell", "python", "script", "install", "run"],
    "files": ["file", "write file", "edit file", "delete", "move", "rename", "clear"],
    "personal data": ["clipboard", "email", "calendar", "message", "text", "imessage", "dm", "kakao", "kakaotalk", "카카오", "카카오톡", "contact", "browser profile"],
    "external side effect": ["send", "email me", "post", "purchase", "call", "book", "pay", "share", "message", "text", "imessage", "dm", "kakao", "kakaotalk", "카카오", "카카오톡"],
}

MAX_AUTONOMY_TEXT_CHARS = 600
OPERATOR_LIMIT_RULE = "the operator's explicit stop times, work windows, pause commands, and newer instructions override priority goals and execution momentum."
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)


def _short(value: Any, *, limit: int = MAX_AUTONOMY_TEXT_CHARS) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _safe_short(value: Any, *, limit: int = MAX_AUTONOMY_TEXT_CHARS) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", _short(value, limit=limit))


def _metadata_int(value: Any, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _metadata_any_bool(*values: Any) -> bool:
    return any(_metadata_bool(value) for value in values)


def _metadata_all_bool(*values: Any) -> bool:
    return all(_metadata_bool(value) for value in values)


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
        "writes_files": False,
        "writes_database": False,
        "writes_notes": False,
        "writes_memory": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "controls_computer": False,
        "calls_external_service": False,
        "calls_external_services": False,
        "requires_approval": False,
        "speaks": False,
        "completes_tasks": False,
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
    }
    metadata.update(extra)
    return metadata


def _autonomy_input_refusal(
    tool_name: str,
    prompt: str,
    metadata: dict[str, Any] | None = None,
) -> ToolResult:
    """Return one side-effect-free, private-safe missing-input refusal."""

    output = f"{prompt} {LOCAL_READ_INPUT_RECOVERY_ACTION}"
    truth = dict(metadata or _safe_metadata())
    truth.setdefault("state_changed", False)
    return ToolResult(
        tool_name,
        False,
        output,
        declare_retryable_local_read_failure(
            truth,
            output=output,
            action=LOCAL_READ_INPUT_RECOVERY_ACTION,
        ),
    )


def _no_authority_fields() -> dict[str, bool]:
    return {
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
        "writes_files": False,
        "writes_database": False,
        "writes_notes": False,
        "writes_memory": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "controls_computer": False,
        "calls_external_service": False,
        "calls_external_services": False,
        "requires_approval": False,
        "speaks": False,
        "completes_tasks": False,
    }


def _frontdoor_handoff(
    *,
    source: str,
    status: str,
    reason: str = "",
    display_request: str = "",
    next_command: str = "",
    recommended_next_commands: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    commands = list(recommended_next_commands or [])
    if next_command and next_command in commands:
        commands = [command for command in commands if command != next_command]
        commands.insert(0, next_command)
    elif next_command:
        commands.insert(0, next_command)
    return {
        "source": source,
        "status": status,
        "reason": reason,
        "display_request": display_request,
        "request_length": len(display_request),
        "next_command": next_command,
        "recommended_next_commands": commands,
        "recommended_next_command_count": len(commands),
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "ready_for_operator": True,
        "operator_limit_rule": OPERATOR_LIMIT_RULE,
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
        **_no_authority_fields(),
        **extra,
    }


def _risk_preflight_handoff(
    *,
    display_request: str,
    matched_risks: list[str],
    risk_gated_tool_count: int,
    likely_approval_required: bool,
) -> dict[str, Any]:
    recommended_next_commands = [
        f"action rehearsal: {display_request}",
        f"autonomy plan: {display_request}",
        f"computer task plan: {display_request}",
        "integration action preview: <connector> -> <action>",
    ]
    return {
        "source": "risk_preflight",
        "request_length": len(display_request),
        "matched_risks": list(matched_risks),
        "matched_risk_count": len(matched_risks),
        "risk_gated_tool_count": int(risk_gated_tool_count),
        "likely_approval_required": bool(likely_approval_required),
        "next_command": recommended_next_commands[0],
        "recommended_next_commands": recommended_next_commands,
        "recommended_next_command_count": len(recommended_next_commands),
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
        "writes_files": False,
        "writes_database": False,
        "writes_notes": False,
        "writes_memory": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "controls_computer": False,
        "calls_external_service": False,
        "calls_external_services": False,
        "requires_approval": False,
        "speaks": False,
        "completes_tasks": False,
    }


def _action_readiness_handoff(
    *,
    display_request: str,
    route: str,
    recommendation: str,
    reason: str,
    recommended_next_commands: list[str],
    safe_to_execute_now: bool,
    approval_required: bool,
    matched_risks: list[str],
    missing_checks: list[str],
    pending_approval_count: int,
    recovery_closure: dict[str, Any],
    recovery_debt_visible: bool,
    readiness_preview_allowed_with_recovery_debt: bool,
    recovery_closure_blocks_current_action: bool,
    risk_gated_tool_count: int,
) -> dict[str, Any]:
    return {
        "source": "action_readiness_packet",
        "request_length": len(display_request),
        "route": route,
        "recommendation": recommendation,
        "reason": reason,
        "next_command": recommended_next_commands[0] if recommended_next_commands else "",
        "recommended_next_commands": list(recommended_next_commands),
        "recommended_next_command_count": len(recommended_next_commands),
        "safe_to_execute_now": _metadata_bool(safe_to_execute_now),
        "approval_required": _metadata_bool(approval_required),
        "matched_risks": list(matched_risks),
        "matched_risk_count": len(matched_risks),
        "missing_checks": list(missing_checks),
        "missing_check_count": len(missing_checks),
        "pending_approval_count": int(pending_approval_count),
        "risk_gated_tool_count": int(risk_gated_tool_count),
        "recovery_debt_visible": _metadata_bool(recovery_debt_visible),
        "readiness_preview_allowed_with_recovery_debt": _metadata_bool(readiness_preview_allowed_with_recovery_debt),
        "recovery_closure_blocks_current_action": _metadata_bool(recovery_closure_blocks_current_action),
        "recovery_closure_state": recovery_closure.get("state") or "not_available",
        "recovery_closure_ready_to_retry": _metadata_bool(recovery_closure.get("ready_to_retry")),
        "recovery_closure_missing_count": _metadata_int(recovery_closure.get("missing_count")),
        "recovery_closure_proof_queue": list(recovery_closure.get("proof_queue") or recovery_closure.get("required_commands") or []),
        "recovery_closure_proof_queue_count": _metadata_int(recovery_closure.get("proof_queue_count")),
        "recovery_closure_next_proof_command": recovery_closure.get("next_proof_command") or "",
        "approval_held_review_required": _approval_held_review_required(recovery_closure),
        "approval_held_review_commands": list(recovery_closure.get("required_commands") or []) if _approval_held_review_required(recovery_closure) else [],
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
        "writes_files": False,
        "writes_database": False,
        "writes_notes": False,
        "writes_memory": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "controls_computer": False,
        "calls_external_service": False,
        "calls_external_services": False,
        "requires_approval": False,
        "speaks": False,
        "completes_tasks": False,
    }


def _execution_contract_handoff(
    *,
    display_request: str,
    planner_goal: str,
    route: str,
    approval_required: bool,
    approval_state: str,
    matched_risks: list[str],
    pending_approval_count: int,
    planned_actions: list[dict[str, Any]],
    unknown_tools: int,
    verification_target_count: int,
    recovery_step_count: int,
    learning_hook_count: int,
    risk_gated_tool_count: int,
) -> dict[str, Any]:
    recommended_next_commands = [
        f"risk preflight: {display_request}",
        f"action rehearsal: {display_request}",
        f"harness cycle: {display_request}",
        f"action readiness: {display_request}",
    ]
    return {
        "source": "execution_contract",
        "request_length": len(display_request),
        "planner_goal": planner_goal,
        "route": route,
        "approval_required": _metadata_bool(approval_required),
        "approval_state": approval_state,
        "matched_risks": list(matched_risks),
        "matched_risk_count": len(matched_risks),
        "pending_approval_count": int(pending_approval_count),
        "planned_actions": list(planned_actions),
        "planned_action_count": len(planned_actions),
        "unknown_tools": int(unknown_tools),
        "verification_target_count": int(verification_target_count),
        "recovery_step_count": int(recovery_step_count),
        "learning_hook_count": int(learning_hook_count),
        "risk_gated_tool_count": int(risk_gated_tool_count),
        "next_command": recommended_next_commands[0],
        "recommended_next_commands": recommended_next_commands,
        "recommended_next_command_count": len(recommended_next_commands),
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
        "writes_files": False,
        "writes_database": False,
        "writes_notes": False,
        "writes_memory": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "controls_computer": False,
        "calls_external_service": False,
        "calls_external_services": False,
        "requires_approval": False,
        "speaks": False,
        "completes_tasks": False,
    }


def _argument_contract_handoff(
    *,
    display_request: str,
    planner_goal: str,
    verdict: str,
    next_step: str,
    next_command: str,
    matched_risks: list[str],
    pending_approval_count: int,
    contracts: list[dict[str, Any]],
    missing_argument_count: int,
    warning_count: int,
    approval_required: bool,
    unknown_tools: int,
    risk_gated_tool_count: int,
    approval_forecast: dict[str, Any],
) -> dict[str, Any]:
    recommended_next_commands = [
        f"execution governor: {display_request}",
        f"dispatch decision: {display_request}",
        f"execution contract: {display_request}",
        f"verification packet: {display_request}",
    ]
    if verdict == "PLANNER_GAP":
        recommended_next_commands.append(f"planner gap: {display_request}")
    return {
        "source": "argument_contract_packet",
        "request_length": len(display_request),
        "planner_goal": planner_goal,
        "verdict": verdict,
        "next_step": next_step,
        "next_command": next_command,
        "matched_risks": list(matched_risks),
        "matched_risk_count": len(matched_risks),
        "pending_approval_count": int(pending_approval_count),
        "planned_actions": list(contracts),
        "planned_action_count": len(contracts),
        "missing_argument_count": int(missing_argument_count),
        "argument_warnings": int(warning_count),
        "approval_required": _metadata_bool(approval_required),
        "unknown_tools": int(unknown_tools),
        "risk_gated_tool_count": int(risk_gated_tool_count),
        "approval_queue_forecast": list(approval_forecast.get("approval_queue_forecast") or []),
        "forecast_new_approvals": _metadata_int(approval_forecast.get("forecast_new_approvals")),
        "forecast_reused_approval_ids": list(approval_forecast.get("forecast_reused_approval_ids") or []),
        "forecast_queue_before": _metadata_int(approval_forecast.get("forecast_queue_before")),
        "forecast_queue_after_if_sent": _metadata_int(approval_forecast.get("forecast_queue_after_if_sent")),
        "forecast_queue_delta_if_sent": _metadata_int(approval_forecast.get("forecast_queue_delta_if_sent")),
        "recommended_next_commands": recommended_next_commands,
        "recommended_next_command_count": len(recommended_next_commands),
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
        "writes_files": False,
        "writes_database": False,
        "writes_notes": False,
        "writes_memory": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "controls_computer": False,
        "calls_external_service": False,
        "calls_external_services": False,
        "requires_approval": False,
        "speaks": False,
        "completes_tasks": False,
    }


def _verification_packet_handoff(
    *,
    display_request: str,
    planner_goal: str,
    verdict: str,
    next_step: str,
    next_command: str,
    matched_risks: list[str],
    pending_approval_count: int,
    planned_actions: list[dict[str, Any]],
    approval_required: bool,
    unknown_tools: int,
    evidence_requirement_count: int,
    failure_signal_count: int,
    recovery_step_count: int,
    risk_gated_tool_count: int,
    approval_forecast: dict[str, Any],
) -> dict[str, Any]:
    suggested_commands = [
        f"execution contract: {display_request}",
        f"action readiness: {display_request}",
        f"risk preflight: {display_request}",
    ]
    return {
        "source": "verification_packet",
        "request_length": len(display_request),
        "planner_goal": planner_goal,
        "verdict": verdict,
        "next_step": next_step,
        "next_command": next_command,
        "matched_risks": list(matched_risks),
        "matched_risk_count": len(matched_risks),
        "pending_approval_count": int(pending_approval_count),
        "planned_actions": list(planned_actions),
        "planned_action_count": len(planned_actions),
        "approval_required": _metadata_bool(approval_required),
        "unknown_tools": int(unknown_tools),
        "evidence_requirement_count": int(evidence_requirement_count),
        "failure_signal_count": int(failure_signal_count),
        "recovery_step_count": int(recovery_step_count),
        "risk_gated_tool_count": int(risk_gated_tool_count),
        "approval_queue_forecast": list(approval_forecast.get("approval_queue_forecast") or []),
        "forecast_new_approvals": _metadata_int(approval_forecast.get("forecast_new_approvals")),
        "forecast_reused_approval_ids": list(approval_forecast.get("forecast_reused_approval_ids") or []),
        "forecast_queue_before": _metadata_int(approval_forecast.get("forecast_queue_before")),
        "forecast_queue_after_if_sent": _metadata_int(approval_forecast.get("forecast_queue_after_if_sent")),
        "forecast_queue_delta_if_sent": _metadata_int(approval_forecast.get("forecast_queue_delta_if_sent")),
        "suggested_commands": suggested_commands,
        "suggested_command_count": len(suggested_commands),
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
        "writes_files": False,
        "writes_database": False,
        "writes_notes": False,
        "writes_memory": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "controls_computer": False,
        "calls_external_service": False,
        "calls_external_services": False,
        "requires_approval": False,
        "speaks": False,
        "completes_tasks": False,
    }


def _execution_acceptance_handoff(
    *,
    display_request: str,
    planner_goal: str,
    verdict: str,
    next_step: str,
    next_command: str,
    matched_risks: list[str],
    planned_action_count: int,
    approval_required: bool,
    pending_approval_count: int,
    recent_run_count: int,
    has_recent_success: bool,
    has_evidence: bool,
    has_tests: bool,
    has_recovery: bool,
    blocking_reasons: list[str],
    unknown_tools: int,
    risk_gated_tool_count: int,
    approval_forecast: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source": "execution_acceptance_gate",
        "request_length": len(display_request),
        "planner_goal": planner_goal,
        "verdict": verdict,
        "next_step": next_step,
        "next_command": next_command,
        "matched_risks": list(matched_risks),
        "matched_risk_count": len(matched_risks),
        "planned_action_count": int(planned_action_count),
        "approval_required": _metadata_bool(approval_required),
        "pending_approval_count": int(pending_approval_count),
        "recent_run_count": int(recent_run_count),
        "has_recent_success": _metadata_bool(has_recent_success),
        "has_evidence": _metadata_bool(has_evidence),
        "has_tests": _metadata_bool(has_tests),
        "has_recovery": _metadata_bool(has_recovery),
        "blocking_reasons": list(blocking_reasons),
        "blocking_reason_count": len(blocking_reasons),
        "unknown_tools": int(unknown_tools),
        "risk_gated_tool_count": int(risk_gated_tool_count),
        "approval_queue_forecast": list(approval_forecast.get("approval_queue_forecast") or []),
        "forecast_new_approvals": _metadata_int(approval_forecast.get("forecast_new_approvals")),
        "forecast_reused_approval_ids": list(approval_forecast.get("forecast_reused_approval_ids") or []),
        "forecast_queue_before": _metadata_int(approval_forecast.get("forecast_queue_before")),
        "forecast_queue_after_if_sent": _metadata_int(approval_forecast.get("forecast_queue_after_if_sent")),
        "forecast_queue_delta_if_sent": _metadata_int(approval_forecast.get("forecast_queue_delta_if_sent")),
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
        "writes_files": False,
        "writes_database": False,
        "writes_notes": False,
        "writes_memory": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "controls_computer": False,
        "calls_external_service": False,
        "calls_external_services": False,
        "requires_approval": False,
        "speaks": False,
        "completes_tasks": False,
    }


def _execution_readiness_handoff(
    *,
    display_request: str,
    planner_goal: str,
    verdict: str,
    next_command: str,
    matched_risks: list[str],
    pending_approval_count: int,
    planned_actions: list[dict[str, Any]],
    approval_required: bool,
    unknown_tools: int,
    matrix_row_count: int,
    proof_requirement_count: int,
    failed_or_blocked_run_count: int,
    approval_held_run_count: int,
    verification_run_count: int,
    recovery_closure: dict[str, Any],
    recovery_debt_visible: bool,
    readiness_matrix_preview_allowed_with_recovery_debt: bool,
    recovery_closure_blocks_matrix_execution: bool,
    learning_debt: dict[str, Any],
    risk_gated_tool_count: int,
    approval_forecast: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source": "execution_readiness_matrix",
        "request_length": len(display_request),
        "planner_goal": planner_goal,
        "verdict": verdict,
        "next_command": next_command,
        "matched_risks": list(matched_risks),
        "matched_risk_count": len(matched_risks),
        "pending_approval_count": int(pending_approval_count),
        "planned_actions": list(planned_actions),
        "planned_action_count": len(planned_actions),
        "approval_required": _metadata_bool(approval_required),
        "unknown_tools": int(unknown_tools),
        "matrix_row_count": int(matrix_row_count),
        "proof_requirement_count": int(proof_requirement_count),
        "failed_or_blocked_run_count": int(failed_or_blocked_run_count),
        "recent_failed_runs": int(failed_or_blocked_run_count),
        "recent_approval_held_runs": int(approval_held_run_count),
        "verification_run_count": int(verification_run_count),
        "recovery_closure_state": recovery_closure.get("state") or "not_available",
        "recovery_closure_ready_to_retry": _metadata_bool(recovery_closure.get("ready_to_retry")),
        "recovery_closure_missing": list(recovery_closure.get("missing") or []),
        "recovery_closure_missing_count": _metadata_int(recovery_closure.get("missing_count")),
        "recovery_closure_required_commands": list(recovery_closure.get("required_commands") or []),
        "recovery_closure_next_required_command": recovery_closure.get("next_required_command") or "",
        "recovery_closure_blocks_auto_execution": _metadata_bool(recovery_closure.get("blocks_auto_execution")),
        "recovery_debt_visible": _metadata_bool(recovery_debt_visible),
        "readiness_matrix_preview_allowed_with_recovery_debt": _metadata_bool(readiness_matrix_preview_allowed_with_recovery_debt),
        "recovery_closure_blocks_matrix_execution": _metadata_bool(recovery_closure_blocks_matrix_execution),
        "recovery_closure_target_run_id": recovery_closure.get("target_run_id"),
        "recovery_closure_target_tool_name": recovery_closure.get("target_tool_name"),
        "recovery_closure_proof_queue": list(recovery_closure.get("proof_queue") or recovery_closure.get("required_commands") or []),
        "recovery_closure_proof_queue_count": _metadata_int(recovery_closure.get("proof_queue_count")),
        "recovery_closure_next_proof_command": recovery_closure.get("next_proof_command") or "",
        "approval_held_review_required": _approval_held_review_required(recovery_closure),
        "approval_held_review_commands": list(recovery_closure.get("required_commands") or []) if _approval_held_review_required(recovery_closure) else [],
        "execution_learning_state": learning_debt.get("state"),
        "execution_learning_blocks_completion_claim": _metadata_bool(learning_debt.get("blocks_completion_claim")),
        "execution_learning_missing": list(learning_debt.get("missing") or []),
        "execution_learning_missing_count": _metadata_int(learning_debt.get("missing_count")),
        "execution_learning_proof_queue": list(learning_debt.get("proof_queue") or learning_debt.get("required_commands") or []),
        "execution_learning_proof_queue_count": len(learning_debt.get("proof_queue") or learning_debt.get("required_commands") or []),
        "execution_learning_next_proof_command": learning_debt.get("next_required_command") or "",
        "risk_gated_tool_count": int(risk_gated_tool_count),
        "approval_queue_forecast": list(approval_forecast.get("approval_queue_forecast") or []),
        "forecast_new_approvals": _metadata_int(approval_forecast.get("forecast_new_approvals")),
        "forecast_reused_approval_ids": list(approval_forecast.get("forecast_reused_approval_ids") or []),
        "forecast_queue_before": _metadata_int(approval_forecast.get("forecast_queue_before")),
        "forecast_queue_after_if_sent": _metadata_int(approval_forecast.get("forecast_queue_after_if_sent")),
        "forecast_queue_delta_if_sent": _metadata_int(approval_forecast.get("forecast_queue_delta_if_sent")),
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
        "writes_files": False,
        "writes_database": False,
        "writes_notes": False,
        "writes_memory": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "controls_computer": False,
        "calls_external_service": False,
        "calls_external_services": False,
        "requires_approval": False,
        "speaks": False,
        "completes_tasks": False,
    }


def _approval_queue_forecast(
    store: Any,
    request: str,
    actions: list[Any],
    planned_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    pending_count = len(store.list_pending_approvals(limit=100)) if store is not None else 0
    forecast = []
    for action, item in zip(actions, planned_actions):
        if not item.get("requires_approval"):
            continue
        existing = store.find_matching_pending_approval(request, action.tool_name, action.args) if store is not None else None
        existing_id = int(existing["id"]) if existing is not None else None
        forecast.append(
            {
                "tool_name": action.tool_name,
                "risk": item.get("risk"),
                "existing_approval_id": existing_id,
                "would_queue_new_approval": existing_id is None,
                "would_reuse_pending_approval": existing_id is not None,
                "planned_arg_keys": sorted(str(key) for key in action.args),
            }
        )
    forecast_new = sum(1 for item in forecast if item["would_queue_new_approval"])
    forecast_reused_ids = [
        item["existing_approval_id"]
        for item in forecast
        if item["existing_approval_id"] is not None
    ]
    return {
        "approval_queue_forecast": forecast,
        "forecast_new_approvals": forecast_new,
        "forecast_reused_approval_ids": forecast_reused_ids,
        "forecast_queue_before": pending_count,
        "forecast_queue_after_if_sent": pending_count + forecast_new,
        "forecast_queue_delta_if_sent": forecast_new,
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


def _metadata_truthy_loose(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on"}
    return False


def _normalized_metadata_token(value: Any) -> str:
    try:
        text = "" if value is None else str(value)
    except Exception:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", text.strip().casefold()).strip("_")


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
        if "approval" in token and any(
            marker in token for marker in ("required", "requires", "gate", "gated", "hold", "held")
        ):
            return True
    return False


def _is_approval_held_tool_run(row: Any) -> bool:
    try:
        if bool(row["ok"]):
            return False
    except Exception:
        return False
    return _is_approval_hold_metadata(_row_metadata(row))


def _recent_tool_run_attention_buckets(rows: list[Any]) -> tuple[list[Any], list[Any]]:
    failed: list[Any] = []
    approval_held: list[Any] = []
    for row in rows:
        try:
            if bool(row["ok"]):
                continue
        except Exception:
            failed.append(row)
            continue
        if _is_approval_held_tool_run(row):
            approval_held.append(row)
        else:
            failed.append(row)
    return failed, approval_held


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


def _append_unique(items: list[str], candidates: list[str]) -> None:
    for candidate in candidates:
        if candidate and candidate not in items:
            items.append(candidate)


def _approval_proof_ids_for_row(row: Any | None) -> list[int]:
    if row is None:
        return []
    output_approval_ids = _ids_from_text(
        row["output"] if "output" in row.keys() else "",
        ("approval #", "approval id", "approval packet", "queued as approval"),
    )
    approval_id = row["approval_id"] if "approval_id" in row.keys() else None
    candidate_ids: list[int] = []
    if approval_id is not None:
        try:
            candidate_ids.append(int(approval_id))
        except (TypeError, ValueError):
            pass
    candidate_ids.extend(output_approval_ids)

    proof_ids: list[int] = []
    for candidate in candidate_ids:
        if candidate not in proof_ids:
            proof_ids.append(candidate)
    return proof_ids


def _approval_review_commands_for_row(row: Any | None) -> list[str]:
    commands: list[str] = []
    approval_proof_ids = _approval_proof_ids_for_row(row)
    if approval_proof_ids:
        first = approval_proof_ids[0]
        _append_unique(
            commands,
            [
                f"approval readiness {first}",
                f"approval packet {first}",
                f"approval chain proof {first}",
                f"verification receipt <approved run id from approval chain proof {first}>",
            ],
        )
    else:
        _append_unique(commands, ["pending approvals", "approval review", "approval history"])
    return commands


def _execution_health_recovery_closure_snapshot(recent_runs: list[Any]) -> dict[str, Any]:
    risky_levels = {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"}
    action_runs = [
        row
        for row in recent_runs
        if str(row["tool_name"]) not in RECOVERY_CLOSURE_META_TOOLS
    ]
    newest_problem = None
    failed_or_blocked_action_runs: list[Any] = []
    approval_held_action_runs: list[Any] = []
    for row in action_runs:
        risk = str(row["risk"])
        try:
            ok = bool(row["ok"])
        except Exception:
            ok = False
        try:
            approved = bool(row["approved"])
        except Exception:
            approved = False
        approval_id = row["approval_id"] if "approval_id" in row.keys() else None
        is_problem = not ok or (risk in risky_levels and (not approved or approval_id is None))
        if not is_problem:
            continue
        if _is_approval_held_tool_run(row):
            approval_held_action_runs.append(row)
            continue
        failed_or_blocked_action_runs.append(row)
        if newest_problem is None:
            newest_problem = row

    approval_held_target = approval_held_action_runs[0] if approval_held_action_runs else None
    target = newest_problem or approval_held_target or (action_runs[0] if action_runs else None)
    target_run_id = int(target["id"]) if target is not None else None
    target_tool_name = str(target["tool_name"]) if target is not None else ""
    target_risk = str(target["risk"]) if target is not None else ""
    target_approved = bool(target["approved"]) if target is not None else None
    target_approval_id = target["approval_id"] if target is not None and "approval_id" in target.keys() else None
    approval_problem = bool(
        target is not None
        and target_risk in risky_levels
        and (not target_approved or target_approval_id is None)
    )

    target_verification_runs = [
        row
        for row in recent_runs
        if str(row["tool_name"]) == "verification_receipt" and _row_metadata(row).get("run_id") == target_run_id
    ]
    target_recovery_runs = [
        row
        for row in recent_runs
        if str(row["tool_name"]) == "execution_recovery_packet" and _row_metadata(row).get("run_id") == target_run_id
    ]
    target_learning_runs = [
        row
        for row in recent_runs
        if str(row["tool_name"]) == "after_action_learning_packet" and _row_metadata(row).get("run_id") == target_run_id
    ]

    approval_proof_ids = _approval_proof_ids_for_row(target)
    approval_review_commands = _approval_review_commands_for_row(approval_held_target)

    missing: list[str] = []
    commands: list[str] = []
    if newest_problem is None and approval_held_target is not None:
        missing.append("approval_review")
        _append_unique(commands, approval_review_commands)
    elif target_run_id is not None and newest_problem is not None:
        if not target_verification_runs:
            missing.append("target_verification_receipt")
            commands.append(f"verification receipt {target_run_id}")
        if not target_recovery_runs:
            missing.append("target_recovery_packet")
            commands.append(f"execution recovery packet {target_run_id}")
        if not target_learning_runs:
            missing.append("target_after_action_learning_packet")
            commands.append(f"after-action learning packet {target_run_id}")
            commands.append(f"execution learning closure {target_run_id}")
        if approval_problem:
            missing.append("approval_chain_proof")
            if approval_proof_ids:
                first = approval_proof_ids[0]
                _append_unique(
                    commands,
                    [
                        f"approval readiness {first}",
                        f"approval packet {first}",
                        f"approval chain proof {first}",
                        f"verification receipt <approved run id from approval chain proof {first}>",
                    ],
                )
            else:
                commands.append("approval history")

    if target_run_id is None:
        state = "no_recent_execution"
    elif newest_problem is None and approval_held_target is not None:
        state = "approval_held_review_required"
    elif newest_problem is None:
        state = "not_needed"
    elif missing:
        state = "blocked_missing_" + "_and_".join(missing)
    else:
        state = "ready_for_operator_retry_review"

    return {
        "target_run_id": target_run_id,
        "target_tool_name": target_tool_name,
        "recent_action_runs": len(action_runs),
        "failed_or_blocked_action_runs": len(failed_or_blocked_action_runs),
        "approval_held_action_runs": len(approval_held_action_runs),
        "approval_held_target_run_id": int(approval_held_target["id"]) if approval_held_target is not None else None,
        "approval_held_target_tool_name": str(approval_held_target["tool_name"]) if approval_held_target is not None else "",
        "approval_review_commands": approval_review_commands,
        "approval_review_command_count": len(approval_review_commands),
        "state": state,
        "ready_to_retry": state == "ready_for_operator_retry_review",
        "missing": missing,
        "missing_count": len(missing),
        "required_commands": commands,
        "next_required_command": commands[0] if commands else "",
        "proof_queue": list(commands),
        "proof_queue_count": len(commands),
        "next_proof_command": commands[0] if commands else "",
        "blocks_auto_execution": state not in {"no_recent_execution", "not_needed", "ready_for_operator_retry_review"},
        "learning_closure_command": f"execution learning closure {target_run_id}" if target_run_id is not None else "execution learning closure",
        "execution_learning_closure_command": f"execution learning closure {target_run_id}" if target_run_id is not None else "execution learning closure",
    }


def _execution_learning_debt_snapshot(recent_runs: list[Any]) -> dict[str, Any]:
    from jarvis_v2.tools import harness

    return harness._execution_learning_debt_snapshot(recent_runs)


def _recovery_closure_proof_metadata(recovery_closure: dict[str, Any]) -> dict[str, Any]:
    queue = list(recovery_closure.get("proof_queue") or [])
    return {
        "recovery_closure_proof_queue": queue,
        "recovery_closure_proof_queue_count": recovery_closure.get("proof_queue_count"),
        "recovery_closure_next_proof_command": str(recovery_closure.get("next_proof_command") or ""),
    }


def _approval_held_review_required(recovery_closure: dict[str, Any]) -> bool:
    return str(recovery_closure.get("state") or "") == "approval_held_review_required"


def _execution_review_section_title(recovery_closure: dict[str, Any]) -> str:
    if _approval_held_review_required(recovery_closure):
        return "Approval-held execution review"
    return "Execution health recovery closure"


def _execution_review_debt_label(recovery_closure: dict[str, Any]) -> str:
    if _approval_held_review_required(recovery_closure):
        return "approval review"
    return "recovery debt"


def _execution_review_state_label(recovery_closure: dict[str, Any]) -> str:
    if _approval_held_review_required(recovery_closure):
        return "approval-held review"
    return "recovery closure"


def _execution_learning_metadata(learning_debt: dict[str, Any]) -> dict[str, Any]:
    return {
        "execution_learning_state": learning_debt["state"],
        "execution_learning_blocks_completion_claim": learning_debt["blocks_completion_claim"],
        "execution_learning_recent_action_runs": learning_debt["recent_action_runs"],
        "execution_learning_failed_or_blocked_action_runs": learning_debt["failed_or_blocked_action_runs"],
        "execution_learning_recent_verification_runs": learning_debt["recent_verification_runs"],
        "execution_learning_recent_recovery_runs": learning_debt["recent_recovery_runs"],
        "execution_learning_recent_after_action_learning_runs": learning_debt["recent_after_action_learning_runs"],
        "execution_learning_target_run_id": learning_debt["target_run_id"],
        "execution_learning_target_tool_name": learning_debt["target_tool_name"],
        "execution_learning_target_after_action_learning_packets": learning_debt["target_after_action_learning_packets"],
        "execution_learning_missing": learning_debt["missing"],
        "execution_learning_missing_count": learning_debt["missing_count"],
        "execution_learning_required_commands": learning_debt["required_commands"],
        "execution_learning_next_required_command": learning_debt["next_required_command"],
        "execution_learning_proof_queue": learning_debt["required_commands"],
        "execution_learning_proof_queue_count": len(learning_debt["required_commands"]),
        "execution_learning_next_proof_command": learning_debt["next_required_command"],
    }


def _execution_learning_lines(learning_debt: dict[str, Any]) -> list[str]:
    return [
        "",
        "Execution learning debt:",
        f"- state: {learning_debt['state']}",
        f"- blocks completion claim: {'yes' if learning_debt['blocks_completion_claim'] else 'no'}",
        f"- target run: #{learning_debt['target_run_id']} `{learning_debt['target_tool_name']}`" if learning_debt["target_run_id"] is not None else "- target run: none",
        f"- missing: {', '.join(learning_debt['missing']) if learning_debt['missing'] else 'none'}",
        f"- next learning required: `{learning_debt['next_required_command']}`" if learning_debt["next_required_command"] else "- next learning required: none",
        f"- learning proof queue: {', '.join(f'`{command}`' for command in learning_debt['required_commands']) if learning_debt['required_commands'] else 'none'}",
    ]


def make_autonomy_tools(list_tools: Callable[[], list[Any]], store: Any | None = None):
    def classify_request(request: str) -> tuple[list[str], list[Any]]:
        low = request.lower()
        tools = list_tools()
        risky_tools = [
            tool
            for tool in tools
            if tool.risk.name in {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"}
        ]
        matched_risks = [
            label
            for label, hints in RISKY_HINTS.items()
            if any(_hint_matches(hint, low) for hint in hints)
        ]
        return matched_risks, risky_tools

    def autonomy_plan(args: dict[str, Any]) -> ToolResult:
        request = _short(args.get("request") or args.get("order") or args.get("goal"))
        if not request:
            return _autonomy_input_refusal("autonomy_plan", "Tell me what you want Jarvis to plan.")
        display_request = _safe_short(request)

        matched_risks, risky_tools = classify_request(request)

        lines = [
            "Jarvis autonomy plan:",
            f"Request: {display_request}",
            "",
            "Safe first steps:",
            f"- Operator limit: {OPERATOR_LIMIT_RULE}",
            "- Clarify the desired outcome and success criteria if they are not explicit.",
            "- Check `readiness report` and `return brief` for current blockers and context.",
            "- Search memory, notes, tasks, goals, and recent activity for relevant context.",
            "- Draft a reversible plan before using tools with side effects.",
            "",
            "Approval-gated steps:",
        ]
        if matched_risks:
            for label in matched_risks:
                lines.append(f"- {label}: require explicit approval before acting.")
        else:
            lines.append("- No obvious risky step detected from wording, but any HIGH_RISK, PERSONAL_DATA, or EXTERNAL_SIDE_EFFECT tool still requires approval.")

        lines.extend(
            [
                "- Shell commands, file writes/deletes, clipboard reads, computer control, reminders, and external actions stay blocked until approved.",
                "",
                "Verification:",
                f"- Operator limit: {OPERATOR_LIMIT_RULE}",
                "- Report exactly what was done and what was blocked.",
                "- Use `recent tool runs`, `pending approvals`, and Obsidian notes to verify state.",
                "- For computer-control work, observe before acting and verify the screen after each approved action.",
                "",
                "Relevant risk-gated tool examples:",
            ]
        )
        priority_names = {
            "run_shell_command",
            "write_text_file",
            "observe_screen",
            "observe_act_verify",
            "get_clipboard",
            "enable_computer_control",
        }
        priority_tools = [tool for tool in risky_tools if tool.name in priority_names]
        remaining_tools = [tool for tool in risky_tools if tool.name not in priority_names]
        shown_tools = priority_tools + remaining_tools
        for tool in shown_tools[:12]:
            lines.append(f"- {tool.name} [{tool.risk.name}]: {tool.description}")
        if len(shown_tools) > 12:
            lines.append(f"- ...and {len(shown_tools) - 12} more gated tool(s)")

        return ToolResult(
            "autonomy_plan",
            True,
            "\n".join(lines),
            _safe_metadata(request=display_request, display_request=display_request, matched_risks=matched_risks, risk_gated_tools=len(risky_tools)),
        )

    def risk_preflight(args: dict[str, Any]) -> ToolResult:
        request = _short(args.get("request") or args.get("order") or args.get("goal"))
        if not request:
            return _autonomy_input_refusal("risk_preflight", "Tell me what Jarvis should preflight.")
        display_request = _safe_short(request)

        matched_risks, risky_tools = classify_request(request)
        auto_run = not matched_risks
        lines = [
            "Jarvis risk preflight:",
            f"Request: {display_request}",
            "",
            "Risk signal:",
        ]
        if matched_risks:
            for label in matched_risks:
                lines.append(f"- {label}: likely approval-gated before action.")
        else:
            lines.append("- No obvious risky wording detected; tool risk policy still decides execution.")

        lines.extend(
            [
                "",
                "Safe preview path:",
                "- `action rehearsal: <request>` previews exact planned tools.",
                "- `autonomy plan: <request>` drafts approval-gated steps and verification checkpoints.",
                "- `computer task plan: <objective>` is safer before any desktop-control request.",
                "- `integration action preview: <connector> -> <action>` is safer before personal connectors.",
                "",
                "Boundary:",
                f"- Operator limit: {OPERATOR_LIMIT_RULE}",
                "- This preflight is read-only and does not execute tools, approve requests, dismiss approvals, read private data, write files, call external services, control the computer, or queue approvals.",
            ]
        )
        if auto_run:
            lines.append("- Possible low-risk path: proceed only if the planner chooses read-only or local-safe tools.")
        else:
            lines.append("- Approval likely required before real execution.")

        return ToolResult(
            "risk_preflight",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=display_request,
                display_request=display_request,
                matched_risks=matched_risks,
                risk_gated_tools=len(risky_tools),
                likely_approval_required=bool(matched_risks),
                risk_preflight_handoff=_risk_preflight_handoff(
                    display_request=display_request,
                    matched_risks=matched_risks,
                    risk_gated_tool_count=len(risky_tools),
                    likely_approval_required=bool(matched_risks),
                ),
            ),
        )

    def action_readiness_packet(args: dict[str, Any]) -> ToolResult:
        request = _short(args.get("request") or args.get("order") or args.get("goal"))
        if not request:
            return _autonomy_input_refusal(
                "action_readiness_packet",
                "Tell me what action Jarvis should check for readiness.",
            )
        display_request = _safe_short(request)

        matched_risks, risky_tools = classify_request(request)
        approvals = store.list_pending_approvals(limit=5) if store is not None else []
        recent_runs = store.recent_tool_runs(limit=12) if store is not None else []
        recovery_closure = _execution_health_recovery_closure_snapshot(recent_runs)
        approval_held_review_required = _approval_held_review_required(recovery_closure)
        execution_review_title = _execution_review_section_title(recovery_closure)
        execution_review_debt_label = _execution_review_debt_label(recovery_closure)
        execution_review_state_label = _execution_review_state_label(recovery_closure)
        recovery_debt_visible = bool(recovery_closure["blocks_auto_execution"])
        readiness_preview_allowed_with_recovery_debt = True
        recovery_closure_blocks_current_action = recovery_debt_visible
        needs_scope = bool(matched_risks)
        needs_integration_scope = any(label in matched_risks for label in ("personal data", "external side effect"))
        needs_computer_scope = "computer" in matched_risks
        needs_shell_scope = "shell/code" in matched_risks
        missing_checks: list[str] = []
        if approvals:
            missing_checks.append("review existing pending approvals first")
        if recovery_closure["blocks_auto_execution"]:
            if approval_held_review_required:
                missing_checks.append("review approval-held execution before new action routing")
            else:
                missing_checks.append("close execution health recovery proof before new action routing")
        if needs_scope:
            missing_checks.append("route through execution governor before preflight/rehearsal")
        if needs_computer_scope:
            missing_checks.append("run computer readiness/task plan")
        if needs_integration_scope:
            missing_checks.append("run integration action preview or scope packet")
        if needs_shell_scope:
            missing_checks.append("confirm exact command and avoid shell control operators")

        if approvals:
            recommendation = "STOP_AND_REVIEW_APPROVALS"
            reason = "Pending approval-gated work is already queued; review or dismiss stale blockers before starting a new risky action."
            route = "approval_review"
            recommended_next_commands = [
                f"approval readiness {approvals[0]['id']}",
                f"approval packet {approvals[0]['id']}",
                f"approval chain proof {approvals[0]['id']}",
                "pending approvals",
            ]
            safe_to_execute_now = False
        elif approval_held_review_required:
            recommendation = "APPROVAL_HELD_REVIEW_REQUIRED"
            reason = "A recent approval-gated run was held before execution, so Jarvis should review the approval packet before treating more work as ready."
            route = "approval_held_review"
            recommended_next_commands = list(recovery_closure["required_commands"]) or ["pending approvals"]
            safe_to_execute_now = False
        elif recovery_closure["blocks_auto_execution"]:
            recommendation = "RECOVERY_CLOSURE_REQUIRED"
            reason = "Execution health recovery closure is incomplete, so Jarvis should close the missing proof queue before presenting more work as ready."
            route = "recovery_closure"
            recommended_next_commands = list(recovery_closure["required_commands"]) or ["execution health report"]
            safe_to_execute_now = False
        elif matched_risks:
            recommendation = "PREFLIGHT_REQUIRED"
            reason = "The request contains risky action signals, so Jarvis should enter through the execution governor before any preflight, rehearsal, or real execution."
            route = "preflight_required"
            recommended_next_commands = [
                f"execution governor: {display_request}",
                f"risk preflight: {display_request}",
                f"action rehearsal: {display_request}",
                f"autonomy plan: {display_request}",
            ]
            safe_to_execute_now = False
        else:
            recommendation = "READY_FOR_LOW_RISK_ROUTING"
            reason = "No obvious risky wording was detected; route through the execution governor so ToolRegistry and PermissionPolicy still decide whether the real request may run."
            route = "low_risk_routing"
            recommended_next_commands = [f"execution governor: {display_request}", display_request]
            safe_to_execute_now = True

        lines = [
            "Jarvis action readiness packet:",
            "This is read-only. It decides whether a proposed action is ready for low-risk routing, needs preflight, or should stop for approval review without executing tools.",
            "",
            f"Request: {display_request}",
            f"Recommendation: {recommendation}",
            f"Reason: {reason}",
            "",
            "Detected risk areas:",
        ]
        if matched_risks:
            lines.extend(f"- {label}" for label in matched_risks)
        else:
            lines.append("- none from keyword scan; registered tool risk still applies at execution time")

        lines.extend(["", "Current blockers:"])
        if approvals:
            for row in approvals:
                lines.append(f"- approval #{row['id']} {row['tool_name']}: {row['user_input']}")
                lines.append(f"  readiness: `approval readiness {row['id']}`")
                lines.append(f"  last look: `approval packet {row['id']}`")
                lines.append(f"  proof: `approval chain proof {row['id']}`")
        else:
            lines.append("- no pending approvals visible")
        if recovery_closure["blocks_auto_execution"]:
            lines.append(f"- {execution_review_state_label}: {recovery_closure['state']}")
            lines.append(f"  next: `{recommended_next_commands[0]}`")

        lines.extend(["", "Missing checks before action:"])
        if missing_checks:
            lines.extend(f"- {item}" for item in missing_checks)
        else:
            lines.append("- none detected; proceed only if the planner chooses read-only or local-safe tools")

        lines.extend(
            [
                "",
                "Recommended preview path:",
                f"- `execution governor: {display_request}`",
                f"- `risk preflight: {display_request}`",
                f"- `action rehearsal: {display_request}`",
                f"- `autonomy plan: {display_request}`",
            ]
        )
        if needs_computer_scope:
            lines.append(f"- `computer readiness: {display_request}`")
            lines.append(f"- `computer task plan: {display_request}`")
        if needs_integration_scope:
            lines.append("- `integration action preview: <connector> -> <action>`")
            lines.append("- `integration scope packet: <connector> -> <action>; target <source>; time <range>; data <level>`")
        for command in recovery_closure["required_commands"]:
            lines.append(f"- `{command}`")

        lines.extend(
            [
                "",
                f"{execution_review_title}:",
                f"- state: {recovery_closure['state']}",
                f"- ready to retry: {'yes' if recovery_closure['ready_to_retry'] else 'no'}",
                f"- {execution_review_debt_label} blocks proposed action: {'yes' if recovery_closure_blocks_current_action else 'no'}",
                f"- {execution_review_debt_label} blocks this readiness preview: no, this packet is read-only and does not authorize execution",
                f"- target run: #{recovery_closure['target_run_id']} `{recovery_closure['target_tool_name']}`" if recovery_closure["target_run_id"] is not None else "- target run: none",
                f"- missing: {', '.join(recovery_closure['missing']) if recovery_closure['missing'] else 'none'}",
                f"- next required: `{recovery_closure['next_required_command']}`" if recovery_closure["next_required_command"] else "- next required: none",
                f"- command queue: {', '.join(f'`{command}`' for command in recovery_closure['required_commands']) if recovery_closure['required_commands'] else 'none'}",
                "",
                "Go/no-go rule:",
                f"- Operator limit: {OPERATOR_LIMIT_RULE}",
                "- GO only for read-only or local-safe tool routing with clear target and verification.",
                "- NO-GO if target, account, file, command, coordinate, recipient, or expected result is unclear.",
                "- NO-GO if pending approvals have not gone through readiness review and last-look packet review.",
                "- NO-GO if the request needs shell/code, personal data, external side effects, destructive changes, or computer control without an approval receipt.",
                "",
                "Boundary:",
                "- This packet does not call a model, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, or queue approvals.",
            ]
        )

        return ToolResult(
            "action_readiness_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=display_request,
                display_request=display_request,
                route=route,
                recommendation=recommendation,
                next_command=recommended_next_commands[0],
                recommended_next_commands=recommended_next_commands,
                safe_to_execute_now=safe_to_execute_now,
                recovery_debt_visible=recovery_debt_visible,
                readiness_preview_allowed_with_recovery_debt=readiness_preview_allowed_with_recovery_debt,
                recovery_closure_blocks_current_action=recovery_closure_blocks_current_action,
                approval_required=bool(matched_risks),
                matched_risks=matched_risks,
                pending_approvals=len(approvals),
                recovery_closure_state=recovery_closure["state"],
                recovery_closure_ready_to_retry=recovery_closure["ready_to_retry"],
                recovery_closure_missing=recovery_closure["missing"],
                recovery_closure_missing_count=recovery_closure["missing_count"],
                recovery_closure_required_commands=recovery_closure["required_commands"],
                recovery_closure_next_required_command=recovery_closure["next_required_command"],
                recovery_closure_blocks_auto_execution=recovery_closure["blocks_auto_execution"],
                approval_held_review_required=approval_held_review_required,
                approval_held_review_commands=list(recovery_closure["required_commands"]) if approval_held_review_required else [],
                recovery_closure_target_run_id=recovery_closure["target_run_id"],
                recovery_closure_target_tool_name=recovery_closure["target_tool_name"],
                recovery_closure_learning_closure_command=recovery_closure["learning_closure_command"],
                recovery_closure_execution_learning_closure_command=recovery_closure["execution_learning_closure_command"],
                **_recovery_closure_proof_metadata(recovery_closure),
                missing_checks=missing_checks,
                risk_gated_tools=len(risky_tools),
                action_readiness_handoff=_action_readiness_handoff(
                    display_request=display_request,
                    route=route,
                    recommendation=recommendation,
                    reason=reason,
                    recommended_next_commands=recommended_next_commands,
                    safe_to_execute_now=safe_to_execute_now,
                    approval_required=bool(matched_risks),
                    matched_risks=matched_risks,
                    missing_checks=missing_checks,
                    pending_approval_count=len(approvals),
                    recovery_closure=recovery_closure,
                    recovery_debt_visible=recovery_debt_visible,
                    readiness_preview_allowed_with_recovery_debt=readiness_preview_allowed_with_recovery_debt,
                    recovery_closure_blocks_current_action=recovery_closure_blocks_current_action,
                    risk_gated_tool_count=len(risky_tools),
                ),
            ),
        )

    def execution_contract(args: dict[str, Any]) -> ToolResult:
        request = _short(args.get("request") or args.get("order") or args.get("goal"))
        if not request:
            return _autonomy_input_refusal(
                "execution_contract",
                "Tell Jarvis the order to contract, for example: `execution contract: organize my downloads and summarize what changed`.",
            )
        display_request = _safe_short(request)

        from jarvis_v2.agent.planner import RuleBasedPlanner

        tools_by_name = {tool.name: tool for tool in list_tools()}
        plan = RuleBasedPlanner().plan(request, allow_risky_natural_dispatch=False)
        display_goal = _safe_short(plan.goal)
        matched_risks, risky_tools = classify_request(request)
        pending_approvals = store.list_pending_approvals(limit=5) if store is not None else []
        planned_actions: list[dict[str, Any]] = []
        approval_required = False
        unknown_tools = 0
        for action in plan.actions:
            tool = tools_by_name.get(action.tool_name)
            if tool is None:
                risk = "UNKNOWN"
                toolset = "unknown"
                requires_approval = True
                description = "Tool is not registered."
                unknown_tools += 1
            else:
                risk = tool.risk.name
                toolset = tool.toolset
                requires_approval = risk in {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"}
                description = tool.description
            approval_required = approval_required or requires_approval
            planned_actions.append(
                {
                    "tool": action.tool_name,
                    "toolset": toolset,
                    "risk": risk,
                    "requires_approval": requires_approval,
                    "args": {str(key): _safe_short(value, limit=220) for key, value in action.args.items()},
                    "reason": _safe_short(action.reason, limit=220),
                    "description": _safe_short(description, limit=220),
                }
            )

        if pending_approvals and (approval_required or matched_risks):
            route = "hold_for_approval_review"
            approval_state = "review existing pending approvals before new risky execution"
        elif approval_required or matched_risks:
            route = "approval_gated"
            approval_state = "required before real execution"
        elif plan.needs_model and not planned_actions:
            route = "chat_brain"
            approval_state = "not needed unless the chat response proposes risky follow-up action"
        elif planned_actions:
            route = "auto_safe_tool_route"
            approval_state = "not required for the planned low-risk route"
        else:
            route = "needs_more_detail"
            approval_state = "not decidable yet"

        verification_targets = [
            "audit log records the selected route and any tool result",
            "user-visible response says what changed and what stayed blocked",
        ]
        if any(item["risk"] == "HIGH_RISK" for item in planned_actions) or "shell/code" in matched_risks:
            verification_targets.append("exact command output, exit status, and no hidden shell expansion")
        if "computer" in matched_risks:
            verification_targets.append("observe before acting and verify post-action screen state")
        if "files" in matched_risks:
            verification_targets.append("list exact file paths touched and summarize diffs or moved items")
        if any(label in matched_risks for label in ("personal data", "external side effect")):
            verification_targets.append("show connector scope, target account/source, recipient/action, and final confirmation receipt")

        recovery_steps = [
            "stop if target, account, file path, command, coordinate, or expected result is unclear",
            "stop after any failed risky action; do not retry silently",
            "surface the pending approval id or failure reason in the conversation",
            "use recent tool runs, approval readiness, approval packet, approval chain proof, or checkpoint recovery before resuming",
        ]

        learning_hooks = [
            "if the operator corrects the route, record feedback or a preference candidate",
            "if a failure repeats, promote it to a regression-test or skill draft",
            "if the workflow becomes common, turn it into a reviewed command pattern",
        ]

        lines = [
            "Jarvis execution contract:",
            "This is the steering-wheel contract before action. It is read-only and does not run the order.",
            "",
            f"Order: {display_request}",
            f"Planner goal: {display_goal}",
            f"Route: {route}",
            f"Approval state: {approval_state}",
            f"Pending approvals visible: {len(pending_approvals)}",
            "",
            "Harness stages:",
            "1. Perceive: preserve the exact user order.",
            "2. Ground: attach memory, preferences, current state, tasks, and pending approvals.",
            "3. Route: choose chat, read-only/local-safe tools, approval-gated action, or hold.",
            "4. Plan: produce exact tools and arguments before execution.",
            "5. Gate: auto-run only read-only/local-safe tools; stop for risky actions.",
            "6. Act: run only the selected safe or approved step through ToolRegistry.",
            "7. Verify: compare result evidence against the success condition.",
            "8. Learn: convert corrections and repeated failures into memory, tests, or skills.",
            "",
            "Planned tool route:",
        ]
        if planned_actions:
            for index, item in enumerate(planned_actions, start=1):
                approval_text = "approval required" if item["requires_approval"] else "auto-safe"
                lines.append(f"- {index}. {item['tool']} [{item['toolset']}, {item['risk']}]: {approval_text}")
                lines.append(f"  Reason: {item['reason'] or 'planner selected this tool'}")
                if item["args"]:
                    arg_text = ", ".join(f"{key}={value!r}" for key, value in sorted(item["args"].items()))
                    lines.append(f"  Arguments: {arg_text}")
        elif plan.needs_model:
            lines.append("- chat brain route; no tool action is planned yet")
        else:
            lines.append("- no tool action is planned yet")

        lines.extend(["", "Risk signals:"])
        if matched_risks:
            lines.extend(f"- {label}" for label in matched_risks)
        else:
            lines.append("- none from wording; registered tool risk still controls execution")

        lines.extend(["", "Verification targets:"])
        lines.extend(f"- {target}" for target in verification_targets)
        lines.extend(["", "Recovery and stop rules:", f"- Operator limit: {OPERATOR_LIMIT_RULE}"])
        lines.extend(f"- {step}" for step in recovery_steps)
        lines.extend(["", "Learning hooks:"])
        lines.extend(f"- {hook}" for hook in learning_hooks)
        lines.extend(
            [
                "",
                "Next preview commands:",
                f"- `risk preflight: {display_request}`",
                f"- `action rehearsal: {display_request}`",
                f"- `harness cycle: {display_request}`",
                f"- `action readiness: {display_request}`",
                "",
                "Safety boundary:",
                "- This contract does not call a model, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, or queue approvals.",
            ]
        )

        return ToolResult(
            "execution_contract",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=display_request,
                display_request=display_request,
                planner_goal=display_goal,
                route=route,
                approval_required=approval_required or bool(matched_risks),
                approval_state=approval_state,
                matched_risks=matched_risks,
                pending_approvals=len(pending_approvals),
                planned_actions=planned_actions,
                planned_action_count=len(planned_actions),
                unknown_tools=unknown_tools,
                verification_targets=len(verification_targets),
                recovery_steps=len(recovery_steps),
                learning_hooks=len(learning_hooks),
                risk_gated_tools=len(risky_tools),
                execution_contract_handoff=_execution_contract_handoff(
                    display_request=display_request,
                    planner_goal=display_goal,
                    route=route,
                    approval_required=approval_required or bool(matched_risks),
                    approval_state=approval_state,
                    matched_risks=matched_risks,
                    pending_approval_count=len(pending_approvals),
                    planned_actions=planned_actions,
                    unknown_tools=unknown_tools,
                    verification_target_count=len(verification_targets),
                    recovery_step_count=len(recovery_steps),
                    learning_hook_count=len(learning_hooks),
                    risk_gated_tool_count=len(risky_tools),
                ),
            ),
        )

    def argument_contract_packet(args: dict[str, Any]) -> ToolResult:
        request = _short(args.get("request") or args.get("order") or args.get("goal"))
        if not request:
            return _autonomy_input_refusal(
                "argument_contract_packet",
                "Tell Jarvis the order to argument-contract, for example: `argument contract: run command python3 --version`.",
            )
        display_request = _safe_short(request)

        from jarvis_v2.agent.planner import RuleBasedPlanner

        tools_by_name = {tool.name: tool for tool in list_tools()}
        plan = RuleBasedPlanner().plan(request, allow_risky_natural_dispatch=False)
        display_goal = _safe_short(plan.goal)
        matched_risks, risky_tools = classify_request(request)
        pending_approvals = store.list_pending_approvals(limit=5) if store is not None else []
        contracts: list[dict[str, Any]] = []
        missing_argument_count = 0
        warning_count = 0
        approval_required = False
        unknown_tools = 0

        for action in plan.actions:
            tool = tools_by_name.get(action.tool_name)
            if tool is None:
                risk = "UNKNOWN"
                toolset = "unknown"
                description = "Tool is not registered."
                requires_approval = True
                unknown_tools += 1
            else:
                risk = tool.risk.name
                toolset = tool.toolset
                description = tool.description
                requires_approval = risk in {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"}
            raw_provided_args = {str(key): _short(value, limit=220) for key, value in action.args.items()}
            provided_args = {key: _safe_short(value, limit=220) for key, value in raw_provided_args.items()}
            missing_args = [key for key, value in provided_args.items() if not str(value).strip()]
            warnings: list[str] = []
            command_value = str(raw_provided_args.get("command", ""))
            if action.tool_name == "run_shell_command" and any(token in command_value for token in (";", "&&", "||", "|", "$(", "`", ">", "<")):
                warnings.append("command contains shell-control syntax; keep command argv-style and approval-gated")
            if requires_approval and not provided_args:
                warnings.append("risk-gated tool needs exact arguments before approval")
            if tool is None:
                warnings.append("tool is unknown to registry")
            missing_argument_count += len(missing_args)
            warning_count += len(warnings)
            approval_required = approval_required or requires_approval
            contracts.append(
                {
                    "tool": action.tool_name,
                    "toolset": toolset,
                    "risk": risk,
                    "requires_approval": requires_approval,
                    "provided_args": provided_args,
                    "provided_arg_count": len(provided_args),
                    "missing_args": missing_args,
                    "warnings": warnings,
                    "reason": _safe_short(action.reason, limit=220),
                    "description": _safe_short(description, limit=220),
                }
            )

        approval_forecast = _approval_queue_forecast(store, request, plan.actions, contracts)
        if contracts and not missing_argument_count and not warning_count and not unknown_tools:
            verdict = "EXACT_ARGUMENTS_READY"
            next_step = "Use `execution governor` before dispatch; risky tools still require approval first."
            next_command = f"execution governor: {display_request}"
        elif contracts:
            verdict = "ARGUMENT_REVIEW_REQUIRED"
            next_step = "Fix missing/unsafe arguments before approval or execution."
            next_command = f"action rehearsal: {display_request}"
        elif plan.needs_model and matched_risks:
            verdict = "PLANNER_GAP"
            next_step = "Use `planner gap` and add a deterministic route only after the exact tool contract is clear."
            next_command = f"planner gap: {display_request}"
        elif plan.needs_model:
            verdict = "CHAT_ONLY"
            next_step = "No tool arguments are planned; verify the chat response does not imply execution."
            next_command = "send this message normally"
        else:
            verdict = "NO_TOOL_ARGUMENTS"
            next_step = "Ask for a clearer observable action before execution."
            next_command = "ask the operator for exact target, allowed scope, and success condition"

        lines = [
            "Jarvis argument contract packet:",
            "This is the steering proof that exact tool arguments exist before execution. It is read-only and does not run the order.",
            "",
            f"Order: {display_request}",
            f"Planner goal: {display_goal}",
            f"Verdict: {verdict}",
            f"Next safe step: {next_step}",
            f"Pending approvals visible: {len(pending_approvals)}",
            "",
            "Tool argument contracts:",
        ]
        if contracts:
            for index, contract in enumerate(contracts, start=1):
                approval_text = "approval required" if contract["requires_approval"] else "auto-safe"
                lines.append(f"- {index}. {contract['tool']} [{contract['toolset']}, {contract['risk']}]: {approval_text}")
                lines.append(f"  Reason: {contract['reason'] or 'planner selected this tool'}")
                lines.append(f"  Description: {contract['description']}")
                if contract["provided_args"]:
                    arg_text = ", ".join(f"{key}={value!r}" for key, value in sorted(contract["provided_args"].items()))
                    lines.append(f"  Provided args: {arg_text}")
                else:
                    lines.append("  Provided args: none")
                if contract["missing_args"]:
                    lines.append(f"  Missing args: {', '.join(contract['missing_args'])}")
                else:
                    lines.append("  Missing args: none detected from planner output")
                if contract["warnings"]:
                    lines.extend(f"  Warning: {warning}" for warning in contract["warnings"])
        elif plan.needs_model:
            lines.append("- no exact tool arguments; planner would use chat/model route")
        else:
            lines.append("- no exact tool arguments planned")

        lines.extend(
            [
                "",
                "Approval queue forecast:",
                f"- would queue new approvals if sent: {approval_forecast['forecast_new_approvals']}",
                f"- would reuse pending approval ids: {', '.join(str(item) for item in approval_forecast['forecast_reused_approval_ids']) if approval_forecast['forecast_reused_approval_ids'] else 'none'}",
                f"- forecast queue after if sent: {approval_forecast['forecast_queue_after_if_sent']}",
            ]
        )

        lines.extend(["", "Risk signals:"])
        if matched_risks:
            lines.extend(f"- {label}" for label in matched_risks)
        else:
            lines.append("- none from wording; registered tool risk still controls execution")

        lines.extend(
            [
                "",
                "Argument proof rules:",
                "- Every executable tool must show exact arguments before Jarvis acts.",
                "- Approval packets must match the stored tool name and arguments exactly.",
                "- Shell/code arguments must remain bounded and avoid hidden shell-control syntax.",
                "- Computer-control arguments must be one primitive step with observable expectation.",
                "- File and connector arguments must show exact path/source/target/scope before action.",
                "",
                "Preview commands:",
                f"- `execution governor: {display_request}`",
                f"- `dispatch decision: {display_request}`",
                f"- `execution contract: {display_request}`",
                f"- `verification packet: {display_request}`",
            ]
        )
        if verdict == "PLANNER_GAP":
            lines.append(f"- `planner gap: {display_request}`")
        lines.extend(
            [
                "",
                "Boundary:",
                "- This packet does not call a model, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, speak, complete tasks, or queue approvals.",
            ]
        )

        return ToolResult(
            "argument_contract_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=display_request,
                display_request=display_request,
                planner_goal=display_goal,
                verdict=verdict,
                next_step=next_step,
                next_command=next_command,
                matched_risks=matched_risks,
                pending_approvals=len(pending_approvals),
                planned_actions=contracts,
                planned_action_count=len(contracts),
                missing_argument_count=missing_argument_count,
                argument_warnings=warning_count,
                approval_required=approval_required or bool(matched_risks),
                unknown_tools=unknown_tools,
                risk_gated_tools=len(risky_tools),
                **approval_forecast,
                argument_contract_handoff=_argument_contract_handoff(
                    display_request=display_request,
                    planner_goal=display_goal,
                    verdict=verdict,
                    next_step=next_step,
                    next_command=next_command,
                    matched_risks=matched_risks,
                    pending_approval_count=len(pending_approvals),
                    contracts=contracts,
                    missing_argument_count=missing_argument_count,
                    warning_count=warning_count,
                    approval_required=approval_required or bool(matched_risks),
                    unknown_tools=unknown_tools,
                    risk_gated_tool_count=len(risky_tools),
                    approval_forecast=approval_forecast,
                ),
            ),
        )

    def verification_packet(args: dict[str, Any]) -> ToolResult:
        request = _short(args.get("request") or args.get("order") or args.get("goal"))
        if not request:
            return _autonomy_input_refusal(
                "verification_packet",
                "Tell Jarvis what order to verify, for example: `verification packet: organize my downloads and summarize what changed`.",
            )
        display_request = _safe_short(request)

        from jarvis_v2.agent.planner import RuleBasedPlanner

        tools_by_name = {tool.name: tool for tool in list_tools()}
        plan = RuleBasedPlanner().plan(request, allow_risky_natural_dispatch=False)
        display_goal = _safe_short(plan.goal)
        matched_risks, risky_tools = classify_request(request)
        pending_approvals = store.list_pending_approvals(limit=5) if store is not None else []
        planned_actions: list[dict[str, Any]] = []
        approval_required = False
        unknown_tools = 0
        for action in plan.actions:
            tool = tools_by_name.get(action.tool_name)
            if tool is None:
                risk = "UNKNOWN"
                toolset = "unknown"
                requires_approval = True
                unknown_tools += 1
            else:
                risk = tool.risk.name
                toolset = tool.toolset
                requires_approval = risk in {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"}
            approval_required = approval_required or requires_approval
            planned_actions.append(
                {
                    "tool": action.tool_name,
                    "toolset": toolset,
                    "risk": risk,
                    "requires_approval": requires_approval,
                    "args": {str(key): _safe_short(value, limit=220) for key, value in action.args.items()},
                    "reason": _safe_short(action.reason, limit=220),
                }
            )
        approval_forecast = _approval_queue_forecast(store, request, plan.actions, planned_actions)

        evidence_requirements = [
            "the final Jarvis response states the selected route, what ran, what did not run, and why",
            "recent tool runs or audit state show every executed tool with ok/failure status",
            "pending approvals remain visible if any risky step was blocked",
        ]
        if planned_actions:
            evidence_requirements.append("each planned tool has exact arguments, observed output, and a success/failure result")
        if "shell/code" in matched_risks or any(item["risk"] == "HIGH_RISK" for item in planned_actions):
            evidence_requirements.append("shell/code evidence includes exact command, exit status, bounded output, and no hidden shell expansion")
        if "computer" in matched_risks:
            evidence_requirements.append("computer-control evidence includes observe-before, one approved primitive action, observe-after, and matched screen expectation")
        if "files" in matched_risks:
            evidence_requirements.append("file evidence lists exact paths touched and gives diff, existence, or metadata proof")
        if "personal data" in matched_risks:
            evidence_requirements.append("personal-data evidence states source, scope, visible redactions, and why no extra private data was read")
        if "external side effect" in matched_risks:
            evidence_requirements.append("external-effect evidence includes target account/recipient/action, final confirmation, and an audit receipt")
        if plan.needs_model and not planned_actions:
            evidence_requirements.append("chat-brain evidence cites the memory/context basis and avoids claiming tool execution")

        failure_signals = [
            "target, command, path, account, recipient, coordinate, or success condition is vague",
            "planned tool risk is higher than the user-facing route claimed",
            "a risky tool queues approval but no approval id or packet command is shown",
            "tool output is missing, truncated without explanation, contradictory, or unverifiable",
            "the result claims completion without audit, screen, file, note, or tool evidence",
        ]
        recovery_steps = [
            "stop and report the missing evidence instead of retrying silently",
            "run `execution contract: <order>` to re-check route and gates",
            "run `approval readiness #ID`, then `approval packet #ID`, then `approval chain proof #ID`, before any queued risky rerun",
            "use checkpoint recovery only for reviewed local-safe resume steps",
        ]

        if pending_approvals and (approval_required or matched_risks):
            verdict = "VERIFY_AFTER_APPROVAL_REVIEW"
            next_step = f"Run `approval readiness {pending_approvals[0]['id']}`, then `approval packet {pending_approvals[0]['id']}`, then `approval chain proof {pending_approvals[0]['id']}`, or dismiss stale approvals before real execution."
            next_command = f"approval readiness {pending_approvals[0]['id']}"
        elif approval_required or matched_risks:
            verdict = "VERIFY_AFTER_APPROVAL"
            next_step = "Route through `execution governor` first; run the real order only after exact approval-gated arguments are available."
            next_command = f"execution governor: {display_request}"
        elif planned_actions:
            verdict = "VERIFY_AFTER_AUTO_SAFE_ACTION"
            next_step = "If executed, compare each local-safe tool result against the evidence checklist."
            next_command = f"execution acceptance gate: {display_request}; evidence <receipt>; tests <verification>; recovery <stop condition>"
        elif plan.needs_model:
            verdict = "VERIFY_CHAT_RESPONSE_ONLY"
            next_step = "Verify answer quality against available context; do not imply tools ran."
            next_command = f"chat context: {display_request}"
        else:
            verdict = "NEEDS_CLEARER_SUCCESS_CONDITION"
            next_step = "Ask for a clearer observable result before execution."
            next_command = "ask the operator for a clearer observable result"

        lines = [
            "Jarvis verification packet:",
            "This is the dashboard proof plan for an order. It is read-only and does not run the order.",
            "",
            f"Order: {display_request}",
            f"Planner goal: {display_goal}",
            f"Verification verdict: {verdict}",
            f"Next safe verification step: {next_step}",
            f"Pending approvals visible: {len(pending_approvals)}",
            "",
            "Planned route to verify:",
        ]
        if planned_actions:
            for index, item in enumerate(planned_actions, start=1):
                gate = "approval-gated" if item["requires_approval"] else "auto-safe"
                lines.append(f"- {index}. {item['tool']} [{item['toolset']}, {item['risk']}]: {gate}")
                if item["args"]:
                    arg_text = ", ".join(f"{key}={value!r}" for key, value in sorted(item["args"].items()))
                    lines.append(f"  Arguments to verify: {arg_text}")
        elif plan.needs_model:
            lines.append("- chat brain only; verify answer context, not tool execution")
        else:
            lines.append("- no concrete route yet; success condition is not verifiable")

        lines.extend(
            [
                "",
                "Approval queue forecast:",
                f"- would queue new approvals if sent: {approval_forecast['forecast_new_approvals']}",
                f"- would reuse pending approval ids: {', '.join(str(item) for item in approval_forecast['forecast_reused_approval_ids']) if approval_forecast['forecast_reused_approval_ids'] else 'none'}",
                f"- forecast queue after if sent: {approval_forecast['forecast_queue_after_if_sent']}",
            ]
        )
        lines.extend(["", "Evidence requirements:"])
        lines.extend(f"- {item}" for item in evidence_requirements)
        lines.extend(["", "Failure signals:"])
        lines.extend(f"- {item}" for item in failure_signals)
        lines.extend(["", "Recovery if verification fails:"])
        lines.extend(f"- {item}" for item in recovery_steps)
        lines.extend(["", "Suggested commands:"])
        lines.append(f"- `execution contract: {display_request}`")
        lines.append(f"- `action readiness: {display_request}`")
        lines.append(f"- `risk preflight: {display_request}`")
        if pending_approvals:
            lines.append(f"- `approval readiness {pending_approvals[0]['id']}`")
            lines.append(f"- `approval packet {pending_approvals[0]['id']}`")
        lines.extend(
            [
                "",
                "Boundary:",
                "- This packet does not call a model, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, speak, complete tasks, or queue approvals.",
            ]
        )

        return ToolResult(
            "verification_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=display_request,
                display_request=display_request,
                planner_goal=display_goal,
                verdict=verdict,
                next_step=next_step,
                next_command=next_command,
                matched_risks=matched_risks,
                pending_approvals=len(pending_approvals),
                planned_actions=planned_actions,
                planned_action_count=len(planned_actions),
                approval_required=approval_required or bool(matched_risks),
                unknown_tools=unknown_tools,
                evidence_requirements=len(evidence_requirements),
                failure_signals=len(failure_signals),
                recovery_steps=len(recovery_steps),
                risk_gated_tools=len(risky_tools),
                **approval_forecast,
                verification_packet_handoff=_verification_packet_handoff(
                    display_request=display_request,
                    planner_goal=display_goal,
                    verdict=verdict,
                    next_step=next_step,
                    next_command=next_command,
                    matched_risks=matched_risks,
                    pending_approval_count=len(pending_approvals),
                    planned_actions=planned_actions,
                    approval_required=approval_required or bool(matched_risks),
                    unknown_tools=unknown_tools,
                    evidence_requirement_count=len(evidence_requirements),
                    failure_signal_count=len(failure_signals),
                    recovery_step_count=len(recovery_steps),
                    risk_gated_tool_count=len(risky_tools),
                    approval_forecast=approval_forecast,
                ),
            ),
        )

    def execution_acceptance_gate(args: dict[str, Any]) -> ToolResult:
        request = _short(args.get("request") or args.get("order") or args.get("goal") or args.get("behavior"))
        evidence = _short(args.get("evidence") or args.get("proof") or args.get("receipt"), limit=900)
        tests = _short(args.get("tests") or args.get("verification") or args.get("test"), limit=700)
        rollback = _short(args.get("rollback") or args.get("recovery"), limit=500)
        if not request:
            return _autonomy_input_refusal(
                "execution_acceptance_gate",
                "Tell Jarvis which behavior or order to gate, for example: `acceptance gate: organize downloads; evidence recent tool run ok; tests smoke passed`.",
            )
        display_request = _safe_short(request)
        display_evidence = _safe_short(evidence, limit=900)
        display_tests = _safe_short(tests, limit=700)
        display_rollback = _safe_short(rollback, limit=500)

        from jarvis_v2.agent.planner import RuleBasedPlanner

        tools_by_name = {tool.name: tool for tool in list_tools()}
        plan = RuleBasedPlanner().plan(request, allow_risky_natural_dispatch=False)
        display_goal = _safe_short(plan.goal)
        matched_risks, risky_tools = classify_request(request)
        pending_approvals = store.list_pending_approvals(limit=5) if store is not None else []
        recent_runs = store.recent_tool_runs(limit=20) if store is not None else []

        planned_actions: list[dict[str, Any]] = []
        approval_required = bool(matched_risks)
        unknown_tools = 0
        for action in plan.actions:
            tool = tools_by_name.get(action.tool_name)
            if tool is None:
                risk = "UNKNOWN"
                toolset = "unknown"
                requires_approval = True
                unknown_tools += 1
            else:
                risk = tool.risk.name
                toolset = tool.toolset
                requires_approval = risk in {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"}
            approval_required = approval_required or requires_approval
            planned_actions.append(
                {
                    "tool": action.tool_name,
                    "toolset": toolset,
                    "risk": risk,
                    "requires_approval": requires_approval,
                    "args": {str(key): _safe_short(value, limit=180) for key, value in action.args.items()},
                }
            )
        approval_forecast = _approval_queue_forecast(store, request, plan.actions, planned_actions)

        has_evidence = bool(evidence)
        has_tests = bool(tests)
        has_recovery = bool(rollback)
        has_recent_success = any(row["ok"] for row in recent_runs)
        blocking_reasons: list[str] = []
        if pending_approvals and approval_required:
            blocking_reasons.append("pending approval review remains before accepting risky work")
        if approval_required and not has_evidence:
            blocking_reasons.append("approval/audit evidence is missing for risky work")
        if not has_evidence:
            blocking_reasons.append("completion proof/evidence is missing")
        if not has_tests:
            blocking_reasons.append("test or verification receipt is missing")
        if (approval_required or matched_risks) and not has_recovery:
            blocking_reasons.append("rollback or recovery path is missing for risky work")
        if unknown_tools:
            blocking_reasons.append("planner selected an unknown tool")
        if not planned_actions and not plan.needs_model:
            blocking_reasons.append("planner has no clear action route or chat route")

        if blocking_reasons:
            verdict = "NOT_ACCEPTED"
            if blocking_reasons == ["pending approval review remains before accepting risky work"]:
                next_step = "Resolve or dismiss pending approvals before treating this risky behavior as accepted."
                next_command = f"approval readiness {pending_approvals[0]['id']}" if pending_approvals else f"execution proof bundle: {display_request}"
            else:
                next_step = "Fill the missing evidence, tests, approval/audit, or recovery fields before claiming the behavior is done."
                next_command = f"execution proof bundle: {display_request}; verification <target>; tests <smoke>; evidence <receipt>; recovery <stop condition>"
        elif approval_required:
            verdict = "READY_FOR_HUMAN_ACCEPTANCE_REVIEW"
            next_step = "Have the operator review the approval/audit evidence and recovery path before treating the behavior as accepted."
            next_command = f"execution proof bundle: {display_request}; verification <target>; tests <smoke>; evidence <receipt>; recovery <stop condition>"
        else:
            verdict = "ACCEPTABLE_FOR_LOCAL_SAFE_COMPLETION_REVIEW"
            next_step = "Use recent tool runs and the supplied verification receipt before marking related work complete."
            next_command = "execution audit gate"

        required_receipts = [
            "route receipt: planned tool or chat path matches the original order",
            "argument receipt: exact tool arguments are visible before execution",
            "audit receipt: recent tool run or approval record proves what actually happened",
            "verification receipt: tests or observed evidence match the success condition",
            "recovery receipt: rollback, retry boundary, or stop condition is stated when risk is present",
        ]

        lines = [
            "Jarvis execution acceptance gate:",
            "This is the final read-only gate before Jarvis treats a behavior as done. It does not run tools, approve requests, mark tasks complete, or change state.",
            "",
            f"Behavior/order: {display_request}",
            f"Planner goal: {display_goal}",
            f"Verdict: {verdict}",
            f"Next step: {next_step}",
            "",
            "Route summary:",
        ]
        if planned_actions:
            for index, item in enumerate(planned_actions, start=1):
                approval_text = "approval required" if item["requires_approval"] else "auto-safe"
                lines.append(f"- {index}. {item['tool']} [{item['toolset']}, {item['risk']}]: {approval_text}")
                if item["args"]:
                    lines.append(f"  args: {item['args']}")
        elif plan.needs_model:
            lines.append("- chat/model response route; no tool execution should be claimed")
        else:
            lines.append("- no exact route found")

        lines.extend(["", "Acceptance receipts:"])
        for item in required_receipts:
            lines.append(f"- {item}")

        lines.extend(
            [
                "",
                "Supplied proof:",
                f"- evidence: {display_evidence or '<missing>'}",
                f"- tests: {display_tests or '<missing>'}",
                f"- recovery: {display_rollback or '<missing>'}",
                "",
                "Blocking reasons:",
            ]
        )
        if blocking_reasons:
            lines.extend(f"- {reason}" for reason in blocking_reasons)
        else:
            lines.append("- none from this gate; still do human review before any final completion claim")

        lines.extend(
            [
                "",
                "Current state:",
                f"- pending approvals visible: {len(pending_approvals)}",
                f"- would queue new approvals if sent: {approval_forecast['forecast_new_approvals']}",
                f"- would reuse pending approval ids: {', '.join(str(item) for item in approval_forecast['forecast_reused_approval_ids']) if approval_forecast['forecast_reused_approval_ids'] else 'none'}",
                f"- forecast queue after if sent: {approval_forecast['forecast_queue_after_if_sent']}",
                f"- recent tool runs inspected: {len(recent_runs)}",
                f"- recent successful tool run visible: {'yes' if has_recent_success else 'no'}",
                "",
                "Boundary:",
                "- This gate does not call a model, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, speak, complete tasks, or queue approvals.",
            ]
        )

        return ToolResult(
            "execution_acceptance_gate",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=display_request,
                display_request=display_request,
                planner_goal=display_goal,
                verdict=verdict,
                next_step=next_step,
                next_command=next_command,
                matched_risks=matched_risks,
                planned_actions=len(planned_actions),
                approval_required=approval_required,
                pending_approvals=len(pending_approvals),
                recent_runs=len(recent_runs),
                has_recent_success=has_recent_success,
                has_evidence=has_evidence,
                has_tests=has_tests,
                has_recovery=has_recovery,
                blocking_reasons=len(blocking_reasons),
                blocking_reason_names=blocking_reasons,
                unknown_tools=unknown_tools,
                risk_gated_tools=len(risky_tools),
                **approval_forecast,
                execution_acceptance_handoff=_execution_acceptance_handoff(
                    display_request=display_request,
                    planner_goal=display_goal,
                    verdict=verdict,
                    next_step=next_step,
                    next_command=next_command,
                    matched_risks=matched_risks,
                    planned_action_count=len(planned_actions),
                    approval_required=approval_required,
                    pending_approval_count=len(pending_approvals),
                    recent_run_count=len(recent_runs),
                    has_recent_success=has_recent_success,
                    has_evidence=has_evidence,
                    has_tests=has_tests,
                    has_recovery=has_recovery,
                    blocking_reasons=blocking_reasons,
                    unknown_tools=unknown_tools,
                    risk_gated_tool_count=len(risky_tools),
                    approval_forecast=approval_forecast,
                ),
            ),
        )

    def execution_readiness_matrix(args: dict[str, Any]) -> ToolResult:
        request = _short(args.get("request") or args.get("order") or args.get("goal"))
        if not request:
            return _autonomy_input_refusal(
                "execution_readiness_matrix",
                "Tell Jarvis the order to matrix, for example: `execution readiness matrix: organize my downloads and summarize what changed`.",
            )
        display_request = _safe_short(request)

        from jarvis_v2.agent.planner import RuleBasedPlanner

        tools_by_name = {tool.name: tool for tool in list_tools()}
        plan = RuleBasedPlanner().plan(request, allow_risky_natural_dispatch=False)
        display_goal = _safe_short(plan.goal)
        matched_risks, risky_tools = classify_request(request)
        pending_approvals = store.list_pending_approvals(limit=5) if store is not None else []
        recent_runs = store.recent_tool_runs(limit=20) if store is not None else []
        failed_or_blocked, approval_held_runs = _recent_tool_run_attention_buckets(recent_runs)
        recovery_closure = _execution_health_recovery_closure_snapshot(recent_runs)
        approval_held_review_required = _approval_held_review_required(recovery_closure)
        execution_review_title = _execution_review_section_title(recovery_closure)
        execution_review_debt_label = _execution_review_debt_label(recovery_closure)
        execution_review_state_label = _execution_review_state_label(recovery_closure)
        learning_debt = _execution_learning_debt_snapshot(recent_runs)
        recovery_debt_visible = bool(recovery_closure["blocks_auto_execution"])
        readiness_matrix_preview_allowed_with_recovery_debt = True
        recovery_closure_blocks_matrix_execution = recovery_debt_visible
        verification_runs = [
            row
            for row in recent_runs
            if row["tool_name"] in {"verification_receipt", "runtime_trace_receipt", "verification_packet", "completion_audit_packet", "evidence_ledger", "execution_audit_gate", "execution_health_report"}
        ]

        planned_actions: list[dict[str, Any]] = []
        approval_required = False
        unknown_tools = 0
        for action in plan.actions:
            tool = tools_by_name.get(action.tool_name)
            if tool is None:
                risk = "UNKNOWN"
                toolset = "unknown"
                requires_approval = True
                description = "Tool is not registered."
                unknown_tools += 1
            else:
                risk = tool.risk.name
                toolset = tool.toolset
                requires_approval = risk in {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"}
                description = tool.description
            approval_required = approval_required or requires_approval
            planned_actions.append(
                {
                    "tool": action.tool_name,
                    "toolset": toolset,
                    "risk": risk,
                    "requires_approval": requires_approval,
                    "args": {str(key): _safe_short(value, limit=180) for key, value in action.args.items()},
                    "reason": _safe_short(action.reason, limit=180),
                    "description": _safe_short(description, limit=180),
                }
            )

        approval_forecast = _approval_queue_forecast(store, request, plan.actions, planned_actions)
        gates: list[tuple[str, str, str]] = [
            ("Intent", "ready" if request else "blocked", "Exact order is preserved."),
            ("Grounding", "review" if pending_approvals else "ready", "Check memory/tasks/goals and pending approvals before action."),
            ("Routing", "ready" if planned_actions or plan.needs_model else "review", display_goal),
            ("Risk", "approval" if approval_required or matched_risks else "auto-safe", ", ".join(matched_risks) if matched_risks else "registered tool risk still applies"),
            ("Approval", "blocked" if pending_approvals and (approval_required or matched_risks) else ("required" if approval_required or matched_risks else "not needed"), f"{len(pending_approvals)} pending approval(s) visible"),
            ("Verification", "review" if not verification_runs else "ready", f"{len(verification_runs)} recent verification/audit packet(s)"),
            (
                "Recovery",
                "approval review" if approval_held_review_required else ("required" if recovery_closure["blocks_auto_execution"] else ("review" if failed_or_blocked or approval_held_runs else "standby")),
                f"{len(failed_or_blocked)} failed/blocked and {len(approval_held_runs)} approval-held recent run(s); {execution_review_state_label} {recovery_closure['state']}",
            ),
            ("Learning", "required" if learning_debt["blocks_completion_claim"] else "ready", f"{learning_debt['state']}; {learning_debt['missing_count']} missing proof item(s)"),
        ]

        if pending_approvals and (approval_required or matched_risks):
            verdict = "HOLD_FOR_APPROVAL_REVIEW"
            next_command = f"approval readiness {pending_approvals[0]['id']}"
        elif approval_held_review_required:
            verdict = "APPROVAL_HELD_REVIEW_REQUIRED"
            next_command = recovery_closure["required_commands"][0] if recovery_closure["required_commands"] else "pending approvals"
        elif recovery_closure["blocks_auto_execution"]:
            verdict = "RECOVERY_CLOSURE_REQUIRED"
            next_command = recovery_closure["required_commands"][0] if recovery_closure["required_commands"] else "execution health report"
        elif approval_required or matched_risks:
            verdict = "APPROVAL_REQUIRED_BEFORE_EXECUTION"
            next_command = f"execution contract: {display_request}"
        elif planned_actions:
            verdict = "READY_FOR_AUTO_SAFE_TOOL_ROUTE"
            next_command = f"verification packet: {display_request}"
        elif plan.needs_model:
            verdict = "READY_FOR_CHAT_BRAIN"
            next_command = f"verification packet: {display_request}"
        else:
            verdict = "NEEDS_CLEARER_ORDER"
            next_command = "ask the operator for a clearer observable result"

        proof_requirements = [
            "runtime trace records route, gate, planned tool actions, and final verdict",
            "tool runs show ok/failure state for every executed action",
            "final response distinguishes done work from approval-held or blocked work",
            "execution audit gate is clear before claiming completion",
        ]
        if approval_required or matched_risks:
            proof_requirements.append("approval readiness, approval packet id, approval chain proof, and approved rerun id are linked for risky execution")
        if "shell/code" in matched_risks:
            proof_requirements.append("exact command, exit status, and bounded output are visible")
        if "computer" in matched_risks:
            proof_requirements.append("observe-before and observe-after screen evidence are visible")
        if "files" in matched_risks:
            proof_requirements.append("exact changed paths and diff/existence evidence are visible")
        if any(label in matched_risks for label in ("personal data", "external side effect")):
            proof_requirements.append("connector scope, account/source, recipient/action, and final confirmation receipt are visible")

        lines = [
            "Jarvis execution readiness matrix:",
            "This is the harness dashboard before action. It is read-only and does not run the order.",
            "",
            f"Order: {display_request}",
            f"Planner goal: {display_goal}",
            f"Verdict: {verdict}",
            f"Next safe command: `{next_command}`",
            "",
            "Readiness matrix:",
        ]
        for name, state, reason in gates:
            lines.append(f"- {name}: {state} - {reason}")
        lines.append(f"- Approval forecast: {approval_forecast['forecast_new_approvals']} new, {len(approval_forecast['forecast_reused_approval_ids'])} reused, queue after {approval_forecast['forecast_queue_after_if_sent']}")

        lines.extend(["", "Planned route:"])
        if planned_actions:
            for index, item in enumerate(planned_actions, start=1):
                gate = "approval-gated" if item["requires_approval"] else "auto-safe"
                lines.append(f"- {index}. {item['tool']} [{item['toolset']}, {item['risk']}]: {gate}")
                if item["args"]:
                    arg_text = ", ".join(f"{key}={value!r}" for key, value in sorted(item["args"].items()))
                    lines.append(f"  Args: {arg_text}")
        elif plan.needs_model:
            lines.append("- chat brain route; no tool action is planned yet")
        else:
            lines.append("- no executable route is clear yet")

        lines.extend(["", "Proof requirements:"])
        lines.extend(f"- {item}" for item in proof_requirements)
        lines.extend(["", "Stop conditions:", f"- Operator limit: {OPERATOR_LIMIT_RULE}"])
        lines.extend(
            [
                "- stop if route, target, account, file, command, coordinate, recipient, or success condition is unclear",
                "- stop if a risky action is requested without approval readiness, a last-look approval packet, approval chain proof, and explicit approval",
                "- stop after failed or blocked risky execution; run `execution audit gate` before retrying",
                "- stop while approval-held execution review is missing approval review proof" if approval_held_review_required else "- stop while execution health recovery closure is missing target verification, recovery, learning, or approval-chain proof",
                "- stop if proof requirements cannot be produced",
                "",
                f"{execution_review_title}:",
                f"- recent failed/blocked runs: {len(failed_or_blocked)}",
                f"- recent approval-held runs: {len(approval_held_runs)}",
                f"- state: {recovery_closure['state']}",
                f"- ready to retry: {'yes' if recovery_closure['ready_to_retry'] else 'no'}",
                f"- target run: #{recovery_closure['target_run_id']} `{recovery_closure['target_tool_name']}`" if recovery_closure["target_run_id"] is not None else "- target run: none",
                f"- missing: {', '.join(recovery_closure['missing']) if recovery_closure['missing'] else 'none'}",
                f"- next required: `{recovery_closure['next_required_command']}`" if recovery_closure["next_required_command"] else "- next required: none",
                f"- command queue: {', '.join(f'`{command}`' for command in recovery_closure['required_commands']) if recovery_closure['required_commands'] else 'none'}",
                f"- {execution_review_debt_label} blocks matrix execution: {'yes' if recovery_closure_blocks_matrix_execution else 'no'}",
                f"- {execution_review_debt_label} blocks this readiness matrix: no, this packet is read-only and does not authorize execution",
                *_execution_learning_lines(learning_debt),
                "",
                "Boundary:",
                "- This matrix does not call a model, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, speak, complete tasks, or queue approvals.",
            ]
        )

        return ToolResult(
            "execution_readiness_matrix",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=display_request,
                display_request=display_request,
                planner_goal=display_goal,
                verdict=verdict,
                next_command=next_command,
                matched_risks=matched_risks,
                pending_approvals=len(pending_approvals),
                planned_actions=planned_actions,
                planned_action_count=len(planned_actions),
                approval_required=approval_required or bool(matched_risks),
                unknown_tools=unknown_tools,
                matrix_rows=len(gates),
                proof_requirements=len(proof_requirements),
                failed_or_blocked_runs=len(failed_or_blocked),
                recent_failed_runs=len(failed_or_blocked),
                recent_approval_held_runs=len(approval_held_runs),
                verification_runs=len(verification_runs),
                recovery_closure_state=recovery_closure["state"],
                recovery_closure_ready_to_retry=recovery_closure["ready_to_retry"],
                recovery_closure_missing=recovery_closure["missing"],
                recovery_closure_missing_count=recovery_closure["missing_count"],
                recovery_closure_required_commands=recovery_closure["required_commands"],
                recovery_closure_next_required_command=recovery_closure["next_required_command"],
                recovery_closure_blocks_auto_execution=recovery_closure["blocks_auto_execution"],
                approval_held_review_required=approval_held_review_required,
                approval_held_review_commands=list(recovery_closure["required_commands"]) if approval_held_review_required else [],
                recovery_debt_visible=recovery_debt_visible,
                readiness_matrix_preview_allowed_with_recovery_debt=readiness_matrix_preview_allowed_with_recovery_debt,
                recovery_closure_blocks_matrix_execution=recovery_closure_blocks_matrix_execution,
                recovery_closure_target_run_id=recovery_closure["target_run_id"],
                recovery_closure_target_tool_name=recovery_closure["target_tool_name"],
                **_recovery_closure_proof_metadata(recovery_closure),
                **_execution_learning_metadata(learning_debt),
                risk_gated_tools=len(risky_tools),
                **approval_forecast,
                execution_readiness_handoff=_execution_readiness_handoff(
                    display_request=display_request,
                    planner_goal=display_goal,
                    verdict=verdict,
                    next_command=next_command,
                    matched_risks=matched_risks,
                    pending_approval_count=len(pending_approvals),
                    planned_actions=planned_actions,
                    approval_required=approval_required or bool(matched_risks),
                    unknown_tools=unknown_tools,
                    matrix_row_count=len(gates),
                    proof_requirement_count=len(proof_requirements),
                    failed_or_blocked_run_count=len(failed_or_blocked),
                    approval_held_run_count=len(approval_held_runs),
                    verification_run_count=len(verification_runs),
                    recovery_closure=recovery_closure,
                    recovery_debt_visible=recovery_debt_visible,
                    readiness_matrix_preview_allowed_with_recovery_debt=readiness_matrix_preview_allowed_with_recovery_debt,
                    recovery_closure_blocks_matrix_execution=recovery_closure_blocks_matrix_execution,
                    learning_debt=learning_debt,
                    risk_gated_tool_count=len(risky_tools),
                    approval_forecast=approval_forecast,
                ),
            ),
        )

    def dispatch_decision_packet(args: dict[str, Any]) -> ToolResult:
        request = _short(args.get("request") or args.get("order") or args.get("goal"))
        if not request:
            return _autonomy_input_refusal(
                "dispatch_decision_packet",
                "Tell Jarvis the order to dispatch-check, for example: `dispatch decision: summarize my recent Jarvis work`.",
                _safe_metadata(
                    reason="missing_request",
                    dispatch_decision_handoff_ready=True,
                    dispatch_decision_handoff=_frontdoor_handoff(
                        source="dispatch_decision_packet",
                        status="refused",
                        reason="missing_request",
                        next_command="dispatch decision: summarize my recent Jarvis work",
                        recommended_next_commands=[
                            "dispatch decision: summarize my recent Jarvis work",
                            "command intake: summarize my recent Jarvis work",
                            "execution readiness matrix: summarize my recent Jarvis work",
                        ],
                    ),
                ),
            )
        display_request = _safe_short(request)

        from jarvis_v2.agent.planner import RuleBasedPlanner

        tools_by_name = {tool.name: tool for tool in list_tools()}
        plan = RuleBasedPlanner().plan(request, allow_risky_natural_dispatch=False)
        display_goal = _safe_short(plan.goal)
        matched_risks, risky_tools = classify_request(request)
        pending_approvals = store.list_pending_approvals(limit=5) if store is not None else []
        recent_runs = store.recent_tool_runs(limit=12) if store is not None else []
        recent_failed, recent_approval_held = _recent_tool_run_attention_buckets(recent_runs)
        recovery_closure = _execution_health_recovery_closure_snapshot(recent_runs)
        approval_held_review_required = _approval_held_review_required(recovery_closure)
        execution_review_title = _execution_review_section_title(recovery_closure)
        execution_review_debt_label = _execution_review_debt_label(recovery_closure)
        execution_review_state_label = _execution_review_state_label(recovery_closure)
        learning_debt = _execution_learning_debt_snapshot(recent_runs)
        recovery_debt_visible = bool(recovery_closure["blocks_auto_execution"])
        dispatch_preview_allowed_with_recovery_debt = True
        recovery_closure_blocks_current_dispatch = recovery_debt_visible

        planned_actions: list[dict[str, Any]] = []
        approval_required = False
        unknown_tools = 0
        for action in plan.actions:
            tool = tools_by_name.get(action.tool_name)
            if tool is None:
                risk = "UNKNOWN"
                toolset = "unknown"
                requires_approval = True
                description = "Tool is not registered."
                unknown_tools += 1
            else:
                risk = tool.risk.name
                toolset = tool.toolset
                requires_approval = risk in {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"}
                description = tool.description
            approval_required = approval_required or requires_approval
            planned_actions.append(
                {
                    "tool": action.tool_name,
                    "toolset": toolset,
                    "risk": risk,
                    "requires_approval": requires_approval,
                    "args": {str(key): _safe_short(value, limit=180) for key, value in action.args.items()},
                    "reason": _safe_short(action.reason, limit=180),
                    "description": _safe_short(description, limit=180),
                }
            )

        ambiguous_risky = bool(matched_risks and not planned_actions)
        has_risky_route = bool(approval_required or matched_risks or unknown_tools)
        approval_forecast = _approval_queue_forecast(store, request, plan.actions, planned_actions)
        if pending_approvals and has_risky_route:
            decision = "HOLD_REVIEW_PENDING_APPROVALS"
            route = "approval_review"
            primary_command = f"approval readiness {pending_approvals[0]['id']}"
            reason = "Risky or ambiguous work is requested while an approval-gated item is already pending."
            can_auto_run = False
        elif approval_held_review_required:
            decision = "APPROVAL_HELD_REVIEW_REQUIRED"
            route = "approval_held_review"
            primary_command = recovery_closure["required_commands"][0] if recovery_closure["required_commands"] else "pending approvals"
            reason = "A recent approval-gated run was held before execution, so dispatch stays held until approval review is complete."
            can_auto_run = False
        elif recovery_closure["blocks_auto_execution"]:
            decision = "RECOVERY_CLOSURE_REQUIRED"
            route = "recovery_closure"
            primary_command = recovery_closure["required_commands"][0] if recovery_closure["required_commands"] else "execution health report"
            reason = "Execution health recovery closure is incomplete, so dispatch stays held until the missing proof queue is closed."
            can_auto_run = False
        elif approval_required or unknown_tools:
            decision = "QUEUE_APPROVAL_IF_SENT_FOR_REAL"
            route = "approval_gated_tools"
            primary_command = display_request
            reason = "The planner selected at least one registered high-risk, personal-data, external-effect, or unknown tool."
            can_auto_run = False
        elif ambiguous_risky:
            decision = "ASK_FOR_EXACT_ACTION_OR_PREFLIGHT"
            route = "ambiguous_risky_order"
            primary_command = f"execution readiness matrix: {display_request}"
            reason = "Risky wording is present, but the deterministic planner did not produce exact tool arguments."
            can_auto_run = False
        elif planned_actions:
            decision = "AUTO_RUN_LOCAL_SAFE_IF_SENT_FOR_REAL"
            route = "auto_safe_tools"
            primary_command = display_request
            reason = "The planner selected only read-only or local-safe tools."
            can_auto_run = True
        elif plan.needs_model:
            decision = "ANSWER_IN_CHAT"
            route = "chat_brain"
            primary_command = display_request
            reason = "The order is conversational and has no tool action planned."
            can_auto_run = True
        else:
            decision = "ASK_FOR_MORE_DETAIL"
            route = "unclear"
            primary_command = "ask the operator for the target, success condition, and allowed scope"
            reason = "No executable or conversational route is clear enough."
            can_auto_run = False

        preflight_commands = [
            f"command diagnosis: {display_request}",
            f"execution readiness matrix: {display_request}",
            f"verification packet: {display_request}",
            f"acceptance gate: {display_request}; evidence <receipt>; tests <verification>; recovery <rollback or stop condition>",
        ]
        if matched_risks:
            preflight_commands.insert(1, f"risk preflight: {display_request}")
            preflight_commands.insert(2, f"action rehearsal: {display_request}")
        if "computer" in matched_risks:
            preflight_commands.append(f"computer task plan: {display_request}")
        if any(label in matched_risks for label in ("personal data", "external side effect")):
            preflight_commands.append("integration action preview: <connector> -> <action>")
        if pending_approvals:
            preflight_commands.append(f"approval readiness {pending_approvals[0]['id']}")
            preflight_commands.append(f"approval packet {pending_approvals[0]['id']}")
        if recent_failed:
            preflight_commands.append(f"execution recovery packet {recent_failed[0]['id']}")
            preflight_commands.append(f"verification receipt {recent_failed[0]['id']}")
            preflight_commands.append("execution audit gate")
        for command in recovery_closure["required_commands"]:
            if command not in preflight_commands:
                preflight_commands.append(command)

        proof_contract = [
            "runtime trace shows the same dispatch route and final verdict",
            "every executed tool has an audit row with ok/failure status",
            "risk-gated tools have approval readiness, an approval packet, approval chain proof, and approved rerun link",
            "final response separates completed work from held, blocked, or skipped work",
            "execution acceptance gate is run with evidence, tests, and recovery before the behavior is treated as done",
        ]
        if ambiguous_risky:
            proof_contract.append("exact tool arguments are produced before the real order is sent")
        if approval_held_review_required:
            proof_contract.append("approval-held execution review is complete before new work is dispatched")
        elif recovery_closure["blocks_auto_execution"]:
            proof_contract.append("execution health recovery closure is complete before new work is dispatched")
        if learning_debt["blocks_completion_claim"]:
            proof_contract.append("execution learning debt is reviewed before any completion claim")
        if recent_failed:
            proof_contract.append("recent failed or blocked run is reviewed before trusting completion")

        lines = [
            "Jarvis dispatch decision packet:",
            "This is the final read-only go/no-go packet before a command becomes chat, auto-safe tool work, approval-gated work, or a clarification request.",
            "",
            f"Order: {display_request}",
            f"Planner goal: {display_goal}",
            f"Dispatch decision: {decision}",
            f"Runtime route: {route}",
            f"Can auto-run now: {'yes' if can_auto_run else 'no'}",
            f"Reason: {reason}",
            f"Primary next command: `{primary_command}`",
            "",
            "Planned tool actions:",
        ]
        if planned_actions:
            for index, item in enumerate(planned_actions, start=1):
                gate = "approval-gated" if item["requires_approval"] else "auto-safe"
                lines.append(f"- {index}. {item['tool']} [{item['toolset']}, {item['risk']}]: {gate}")
                if item["args"]:
                    arg_text = ", ".join(f"{key}={value!r}" for key, value in sorted(item["args"].items()))
                    lines.append(f"  Args: {arg_text}")
        elif plan.needs_model:
            lines.append("- chat response only; no tool action planned")
        else:
            lines.append("- no exact tool action planned")

        lines.extend(["", "Risk and blockers:"])
        lines.append(f"- risk signals: {', '.join(matched_risks) if matched_risks else 'none from wording'}")
        lines.append(f"- pending approvals: {len(pending_approvals)}")
        lines.append(f"- would queue new approvals if sent: {approval_forecast['forecast_new_approvals']}")
        lines.append(f"- would reuse pending approval ids: {', '.join(str(item) for item in approval_forecast['forecast_reused_approval_ids']) if approval_forecast['forecast_reused_approval_ids'] else 'none'}")
        lines.append(f"- forecast queue after if sent: {approval_forecast['forecast_queue_after_if_sent']}")
        lines.append(f"- recent failed/blocked runs inspected: {len(recent_failed)}")
        lines.append(f"- recent approval-held runs inspected: {len(recent_approval_held)}")
        lines.append(f"- {execution_review_state_label} state: {recovery_closure['state']}")
        lines.append(f"- {execution_review_state_label} blocks auto-run: {'yes' if recovery_closure['blocks_auto_execution'] else 'no'}")
        lines.append(f"- {execution_review_debt_label} blocks proposed dispatch: {'yes' if recovery_closure_blocks_current_dispatch else 'no'}")
        lines.append(f"- {execution_review_debt_label} blocks this dispatch preview: no, this packet is read-only and does not authorize execution")
        lines.append(f"- execution learning debt state: {learning_debt['state']}")
        lines.append(f"- execution learning blocks completion: {'yes' if learning_debt['blocks_completion_claim'] else 'no'}")
        lines.extend(["", "Preflight commands:"])
        lines.extend(f"- `{command}`" for command in preflight_commands)
        lines.extend(["", "Proof contract after execution:"])
        lines.extend(f"- {item}" for item in proof_contract)
        lines.extend(
            [
                "",
                f"{execution_review_title}:",
                f"- target run: #{recovery_closure['target_run_id']} `{recovery_closure['target_tool_name']}`" if recovery_closure["target_run_id"] is not None else "- target run: none",
                f"- missing: {', '.join(recovery_closure['missing']) if recovery_closure['missing'] else 'none'}",
                f"- next required: `{recovery_closure['next_required_command']}`" if recovery_closure["next_required_command"] else "- next required: none",
                f"- command queue: {', '.join(f'`{command}`' for command in recovery_closure['required_commands']) if recovery_closure['required_commands'] else 'none'}",
                *_execution_learning_lines(learning_debt),
                "",
                "Boundary:",
                "- This dispatch packet does not call a model, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, speak, complete tasks, or queue approvals.",
            ]
        )

        return ToolResult(
            "dispatch_decision_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=display_request,
                display_request=display_request,
                decision=decision,
                route=route,
                can_auto_run=can_auto_run,
                safe_to_execute_now=can_auto_run and not recovery_closure["blocks_auto_execution"],
                recovery_debt_visible=recovery_debt_visible,
                dispatch_preview_allowed_with_recovery_debt=dispatch_preview_allowed_with_recovery_debt,
                recovery_closure_blocks_current_dispatch=recovery_closure_blocks_current_dispatch,
                primary_command=primary_command,
                matched_risks=matched_risks,
                pending_approvals=len(pending_approvals),
                recent_failed_runs=len(recent_failed),
                recent_approval_held_runs=len(recent_approval_held),
                recovery_closure_state=recovery_closure["state"],
                recovery_closure_ready_to_retry=recovery_closure["ready_to_retry"],
                recovery_closure_missing=recovery_closure["missing"],
                recovery_closure_missing_count=recovery_closure["missing_count"],
                recovery_closure_required_commands=recovery_closure["required_commands"],
                recovery_closure_next_required_command=recovery_closure["next_required_command"],
                recovery_closure_blocks_auto_execution=recovery_closure["blocks_auto_execution"],
                approval_held_review_required=approval_held_review_required,
                approval_held_review_commands=list(recovery_closure["required_commands"]) if approval_held_review_required else [],
                recovery_closure_target_run_id=recovery_closure["target_run_id"],
                recovery_closure_target_tool_name=recovery_closure["target_tool_name"],
                **_recovery_closure_proof_metadata(recovery_closure),
                **_execution_learning_metadata(learning_debt),
                planned_actions=planned_actions,
                planned_action_count=len(planned_actions),
                approval_required=approval_required or bool(matched_risks),
                ambiguous_risky_order=ambiguous_risky,
                unknown_tools=unknown_tools,
                preflight_commands=len(preflight_commands),
                proof_contract=len(proof_contract),
                risk_gated_tools=len(risky_tools),
                dispatch_decision_handoff_ready=True,
                dispatch_decision_handoff=_frontdoor_handoff(
                    source="dispatch_decision_packet",
                    status="ok",
                    display_request=display_request,
                    decision=decision,
                    route=route,
                    can_auto_run=can_auto_run,
                    safe_to_execute_now=can_auto_run and not recovery_closure["blocks_auto_execution"],
                    primary_command=primary_command,
                    next_command=primary_command,
                    matched_risks=matched_risks,
                    matched_risk_count=len(matched_risks),
                    pending_approval_count=len(pending_approvals),
                    recent_failed_runs=len(recent_failed),
                    recent_approval_held_runs=len(recent_approval_held),
                    planned_actions=planned_actions,
                    planned_action_count=len(planned_actions),
                    approval_required=approval_required or bool(matched_risks),
                    ambiguous_risky_order=ambiguous_risky,
                    unknown_tools=unknown_tools,
                    recovery_debt_visible=recovery_debt_visible,
                    dispatch_preview_allowed_with_recovery_debt=dispatch_preview_allowed_with_recovery_debt,
                    recovery_closure_blocks_current_dispatch=recovery_closure_blocks_current_dispatch,
                    approval_held_review_required=approval_held_review_required,
                    approval_held_review_commands=list(recovery_closure["required_commands"]) if approval_held_review_required else [],
                    recovery_closure_state=recovery_closure["state"],
                    recovery_closure_ready_to_retry=recovery_closure["ready_to_retry"],
                    recovery_closure_missing_count=recovery_closure["missing_count"],
                    recovery_closure_proof_queue=recovery_closure["proof_queue"],
                    recovery_closure_proof_queue_count=recovery_closure["proof_queue_count"],
                    recovery_closure_next_proof_command=recovery_closure["next_proof_command"],
                    execution_learning_proof_queue=learning_debt["proof_queue"],
                    execution_learning_proof_queue_count=learning_debt["proof_queue_count"],
                    execution_learning_next_proof_command=learning_debt["next_required_command"],
                    recommended_next_commands=preflight_commands,
                    proof_contract_count=len(proof_contract),
                    risk_gated_tool_count=len(risky_tools),
                ),
                **approval_forecast,
            ),
        )

    def command_intake_packet(args: dict[str, Any]) -> ToolResult:
        request = _short(args.get("request") or args.get("order") or args.get("message") or args.get("goal"))
        if not request:
            return _autonomy_input_refusal(
                "command_intake_packet",
                "Tell Jarvis the order to intake, for example: `command intake: organize my downloads and summarize what changed`.",
                _safe_metadata(
                    reason="missing_request",
                    command_intake_handoff_ready=True,
                    command_intake_handoff=_frontdoor_handoff(
                        source="command_intake_packet",
                        status="refused",
                        reason="missing_request",
                        next_command="command intake: organize my downloads and summarize what changed",
                        recommended_next_commands=[
                            "command intake: organize my downloads and summarize what changed",
                            "dispatch decision: organize my downloads and summarize what changed",
                            "execution readiness matrix: organize my downloads and summarize what changed",
                        ],
                    ),
                ),
            )
        display_request = _safe_short(request)

        from jarvis_v2.agent.planner import RuleBasedPlanner

        tools_by_name = {tool.name: tool for tool in list_tools()}
        plan = RuleBasedPlanner().plan(request, allow_risky_natural_dispatch=False)
        display_goal = _safe_short(plan.goal)
        matched_risks, risky_tools = classify_request(request)
        pending_approvals = store.list_pending_approvals(limit=5) if store is not None else []
        recent_runs = store.recent_tool_runs(limit=12) if store is not None else []
        recent_failed, recent_approval_held = _recent_tool_run_attention_buckets(recent_runs)
        recovery_closure = _execution_health_recovery_closure_snapshot(recent_runs)
        approval_held_review_required = _approval_held_review_required(recovery_closure)
        execution_review_title = _execution_review_section_title(recovery_closure)
        execution_review_debt_label = _execution_review_debt_label(recovery_closure)
        execution_review_state_label = _execution_review_state_label(recovery_closure)
        learning_debt = _execution_learning_debt_snapshot(recent_runs)
        recovery_debt_visible = bool(recovery_closure["blocks_auto_execution"])
        command_intake_preview_allowed_with_recovery_debt = True
        recovery_closure_blocks_current_order = recovery_debt_visible

        planned_actions: list[dict[str, Any]] = []
        approval_required = False
        unknown_tools = 0
        exact_arguments_ready = True
        for action in plan.actions:
            tool = tools_by_name.get(action.tool_name)
            if tool is None:
                risk = "UNKNOWN"
                toolset = "unknown"
                requires_approval = True
                description = "Tool is not registered."
                unknown_tools += 1
            else:
                risk = tool.risk.name
                toolset = tool.toolset
                requires_approval = risk in {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"}
                description = tool.description
            action_args = {str(key): _safe_short(value, limit=180) for key, value in action.args.items()}
            exact_arguments_ready = exact_arguments_ready and bool(action_args or action.tool_name in {"respond", "current_time", "readiness_report", "safety_status"})
            approval_required = approval_required or requires_approval
            planned_actions.append(
                {
                    "tool": action.tool_name,
                    "toolset": toolset,
                    "risk": risk,
                    "requires_approval": requires_approval,
                    "args": action_args,
                    "reason": _safe_short(action.reason, limit=180),
                    "description": _safe_short(description, limit=180),
                }
            )

        risky_by_wording = bool(matched_risks)
        risky_or_unknown = bool(approval_required or risky_by_wording or unknown_tools)
        ambiguous_risky = bool(risky_by_wording and not planned_actions)
        approval_forecast = _approval_queue_forecast(store, request, plan.actions, planned_actions)

        if pending_approvals and risky_or_unknown:
            route = "hold_for_approval_review"
            next_command = f"approval readiness {pending_approvals[0]['id']}"
            can_auto_run = False
            intake_state = "blocked_by_existing_approval"
        elif approval_held_review_required:
            route = "approval_held_review"
            next_command = recovery_closure["required_commands"][0] if recovery_closure["required_commands"] else "pending approvals"
            can_auto_run = False
            intake_state = "approval_held_review_required"
        elif recovery_closure["blocks_auto_execution"]:
            route = "recovery_closure"
            next_command = recovery_closure["required_commands"][0] if recovery_closure["required_commands"] else "execution health report"
            can_auto_run = False
            intake_state = "recovery_closure_required"
        elif approval_required or unknown_tools:
            route = "approval_gated_tool_route"
            next_command = f"execution governor: {display_request}"
            can_auto_run = False
            intake_state = "approval_required"
        elif ambiguous_risky:
            route = "planner_gap"
            next_command = f"execution governor: {display_request}"
            can_auto_run = False
            exact_arguments_ready = False
            intake_state = "needs_exact_route"
        elif planned_actions:
            route = "auto_safe_tool_route"
            next_command = display_request
            can_auto_run = True
            intake_state = "ready_for_policy_checked_dispatch"
        elif plan.needs_model:
            route = "chat_brain"
            next_command = display_request
            can_auto_run = True
            intake_state = "ready_for_chat"
        else:
            route = "clarify"
            next_command = "ask the operator for target, scope, and definition of done"
            can_auto_run = False
            exact_arguments_ready = False
            intake_state = "needs_clarification"

        proof_requirements = [
            "preserve the original order text",
            "record the selected route and next command",
            "show exact tool arguments before execution",
            "keep approval-gated work behind approval readiness and a last-look packet",
            "log every executed tool result for audit",
            "verify outcome before claiming completion",
        ]
        if matched_risks:
            proof_requirements.append("link approval id to approved rerun and verification receipt for risky work")
        if approval_held_review_required:
            proof_requirements.append("review the approval-held execution before presenting new work as safe to run")
        elif recovery_closure["blocks_auto_execution"]:
            proof_requirements.append("close execution health recovery proof before presenting new work as safe to run")
        if learning_debt["blocks_completion_claim"]:
            proof_requirements.append("review execution learning debt before treating this route as complete")
        if recent_failed:
            proof_requirements.append("review recent failed or blocked run before retrying or claiming success")
        if recent_approval_held:
            proof_requirements.append("separate approval-held runs from transport/tool failures before recovery review")

        lines = [
            "Jarvis command intake packet:",
            "This is the front-door harness packet for a natural text or speech order. It is read-only and does not run the order.",
            "",
            f"Order: {display_request}",
            f"Planner goal: {display_goal}",
            f"Intake state: {intake_state}",
            f"Route: {route}",
            f"Can auto-run after policy check: {'yes' if can_auto_run else 'no'}",
            f"Next command: `{next_command}`",
            "",
            "Risk and approval:",
            f"- risk signals: {', '.join(matched_risks) if matched_risks else 'none from wording'}",
            f"- approval required: {'yes' if approval_required or risky_by_wording else 'no'}",
            f"- existing pending approvals: {len(pending_approvals)}",
            f"- approval review required first: {'yes' if pending_approvals and risky_or_unknown else 'no'}",
            f"- would queue new approvals if sent: {approval_forecast['forecast_new_approvals']}",
            f"- would reuse pending approval ids: {', '.join(str(item) for item in approval_forecast['forecast_reused_approval_ids']) if approval_forecast['forecast_reused_approval_ids'] else 'none'}",
            f"- forecast queue after if sent: {approval_forecast['forecast_queue_after_if_sent']}",
            f"- recent failed/blocked runs inspected: {len(recent_failed)}",
            f"- recent approval-held runs inspected: {len(recent_approval_held)}",
            f"- {execution_review_state_label} state: {recovery_closure['state']}",
            f"- {execution_review_state_label} blocks auto-run: {'yes' if recovery_closure['blocks_auto_execution'] else 'no'}",
            f"- {execution_review_debt_label} blocks proposed order: {'yes' if recovery_closure_blocks_current_order else 'no'}",
            f"- {execution_review_debt_label} blocks this intake preview: no, this packet is read-only and does not authorize execution",
            f"- execution learning debt state: {learning_debt['state']}",
            f"- execution learning blocks completion: {'yes' if learning_debt['blocks_completion_claim'] else 'no'}",
            "",
            "Planned actions:",
        ]
        if planned_actions:
            for index, item in enumerate(planned_actions, start=1):
                gate = "approval-gated" if item["requires_approval"] else "auto-safe"
                lines.append(f"- {index}. {item['tool']} [{item['toolset']}, {item['risk']}]: {gate}")
                if item["args"]:
                    arg_text = ", ".join(f"{key}={value!r}" for key, value in sorted(item["args"].items()))
                    lines.append(f"  Args: {arg_text}")
                else:
                    lines.append("  Args: none")
        elif plan.needs_model:
            lines.append("- chat response only; no tool action planned")
        else:
            lines.append("- no exact tool action planned")

        lines.extend(["", "Proof requirements:"])
        lines.extend(f"- {item}" for item in proof_requirements)
        lines.extend(
            [
                "",
                "Recommended follow-up packets:",
                f"- `execution governor: {display_request}`",
                f"- `dispatch decision: {display_request}`",
                f"- `execution readiness matrix: {display_request}`",
                f"- `verification packet: {display_request}`",
            ]
        )
        if ambiguous_risky:
            lines.append(f"- `planner gap: {display_request}`")
        if pending_approvals and risky_or_unknown:
            lines.append(f"- `approval readiness {pending_approvals[0]['id']}`")
            lines.append(f"- `approval packet {pending_approvals[0]['id']}`")
        if recent_failed:
            lines.append(f"- `execution recovery packet {recent_failed[0]['id']}`")
        for command in recovery_closure["required_commands"]:
            lines.append(f"- `{command}`")

        lines.extend(
            [
                "",
                f"{execution_review_title}:",
                f"- target run: #{recovery_closure['target_run_id']} `{recovery_closure['target_tool_name']}`" if recovery_closure["target_run_id"] is not None else "- target run: none",
                f"- missing: {', '.join(recovery_closure['missing']) if recovery_closure['missing'] else 'none'}",
                f"- next required: `{recovery_closure['next_required_command']}`" if recovery_closure["next_required_command"] else "- next required: none",
                f"- command queue: {', '.join(f'`{command}`' for command in recovery_closure['required_commands']) if recovery_closure['required_commands'] else 'none'}",
                *_execution_learning_lines(learning_debt),
                "",
                "Boundary:",
                "- This intake packet does not call a model, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, speak, complete tasks, or queue approvals.",
            ]
        )

        return ToolResult(
            "command_intake_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=display_request,
                display_request=display_request,
                intake_state=intake_state,
                route=route,
                next_command=next_command,
                can_auto_run=can_auto_run,
                safe_to_execute_now=can_auto_run and not (pending_approvals and risky_or_unknown) and not recovery_closure["blocks_auto_execution"],
                recovery_debt_visible=recovery_debt_visible,
                command_intake_preview_allowed_with_recovery_debt=command_intake_preview_allowed_with_recovery_debt,
                recovery_closure_blocks_current_order=recovery_closure_blocks_current_order,
                matched_risks=matched_risks,
                approval_required=approval_required or risky_by_wording,
                approval_review_required=bool(pending_approvals and risky_or_unknown),
                pending_approvals=len(pending_approvals),
                **approval_forecast,
                recovery_closure_state=recovery_closure["state"],
                recovery_closure_ready_to_retry=recovery_closure["ready_to_retry"],
                recovery_closure_missing=recovery_closure["missing"],
                recovery_closure_missing_count=recovery_closure["missing_count"],
                recovery_closure_required_commands=recovery_closure["required_commands"],
                recovery_closure_next_required_command=recovery_closure["next_required_command"],
                recovery_closure_blocks_auto_execution=recovery_closure["blocks_auto_execution"],
                approval_held_review_required=approval_held_review_required,
                approval_held_review_commands=list(recovery_closure["required_commands"]) if approval_held_review_required else [],
                recovery_closure_target_run_id=recovery_closure["target_run_id"],
                recovery_closure_target_tool_name=recovery_closure["target_tool_name"],
                **_recovery_closure_proof_metadata(recovery_closure),
                **_execution_learning_metadata(learning_debt),
                planned_actions=planned_actions,
                planned_action_count=len(planned_actions),
                exact_arguments_ready=exact_arguments_ready,
                ambiguous_risky_order=ambiguous_risky,
                unknown_tools=unknown_tools,
                recent_failed_runs=len(recent_failed),
                recent_approval_held_runs=len(recent_approval_held),
                proof_requirements=len(proof_requirements),
                risk_gated_tools=len(risky_tools),
                command_intake_handoff_ready=True,
                command_intake_handoff=_frontdoor_handoff(
                    source="command_intake_packet",
                    status="ok",
                    display_request=display_request,
                    intake_state=intake_state,
                    route=route,
                    next_command=next_command,
                    can_auto_run=can_auto_run,
                    safe_to_execute_now=can_auto_run and not (pending_approvals and risky_or_unknown) and not recovery_closure["blocks_auto_execution"],
                    matched_risks=matched_risks,
                    matched_risk_count=len(matched_risks),
                    approval_required=approval_required or risky_by_wording,
                    approval_review_required=bool(pending_approvals and risky_or_unknown),
                    pending_approval_count=len(pending_approvals),
                    recent_failed_runs=len(recent_failed),
                    recent_approval_held_runs=len(recent_approval_held),
                    planned_actions=planned_actions,
                    planned_action_count=len(planned_actions),
                    exact_arguments_ready=exact_arguments_ready,
                    ambiguous_risky_order=ambiguous_risky,
                    unknown_tools=unknown_tools,
                    recovery_debt_visible=recovery_debt_visible,
                    command_intake_preview_allowed_with_recovery_debt=command_intake_preview_allowed_with_recovery_debt,
                    recovery_closure_blocks_current_order=recovery_closure_blocks_current_order,
                    approval_held_review_required=approval_held_review_required,
                    approval_held_review_commands=list(recovery_closure["required_commands"]) if approval_held_review_required else [],
                    recovery_closure_state=recovery_closure["state"],
                    recovery_closure_ready_to_retry=recovery_closure["ready_to_retry"],
                    recovery_closure_missing_count=recovery_closure["missing_count"],
                    recovery_closure_proof_queue=recovery_closure["proof_queue"],
                    recovery_closure_proof_queue_count=recovery_closure["proof_queue_count"],
                    recovery_closure_next_proof_command=recovery_closure["next_proof_command"],
                    execution_learning_proof_queue=learning_debt["proof_queue"],
                    execution_learning_proof_queue_count=learning_debt["proof_queue_count"],
                    execution_learning_next_proof_command=learning_debt["next_required_command"],
                    recommended_next_commands=[
                        f"execution governor: {display_request}",
                        f"dispatch decision: {display_request}",
                        f"execution readiness matrix: {display_request}",
                        f"verification packet: {display_request}",
                        *recovery_closure["required_commands"],
                    ],
                    proof_requirement_count=len(proof_requirements),
                    risk_gated_tool_count=len(risky_tools),
                ),
            ),
        )

    def execution_governor_packet(args: dict[str, Any]) -> ToolResult:
        request = _short(args.get("request") or args.get("order") or args.get("message") or args.get("goal"))
        if not request:
            return _autonomy_input_refusal(
                "execution_governor_packet",
                "Tell Jarvis the order to govern, for example: `execution governor: organize my downloads and summarize what changed`.",
                _safe_metadata(
                    reason="missing_request",
                    execution_governor_handoff_ready=True,
                    execution_governor_handoff=_frontdoor_handoff(
                        source="execution_governor_packet",
                        status="refused",
                        reason="missing_request",
                        next_command="execution governor: organize my downloads and summarize what changed",
                        recommended_next_commands=[
                            "execution governor: organize my downloads and summarize what changed",
                            "command intake: organize my downloads and summarize what changed",
                            "dispatch decision: organize my downloads and summarize what changed",
                        ],
                    ),
                ),
            )
        display_request = _safe_short(request)

        from jarvis_v2.agent.planner import RuleBasedPlanner

        tools_by_name = {tool.name: tool for tool in list_tools()}
        plan = RuleBasedPlanner().plan(request, allow_risky_natural_dispatch=False)
        display_goal = _safe_short(plan.goal)
        matched_risks, risky_tools = classify_request(request)
        pending_approvals = store.list_pending_approvals(limit=5) if store is not None else []
        recent_runs = store.recent_tool_runs(limit=16) if store is not None else []
        recent_failed, recent_approval_held = _recent_tool_run_attention_buckets(recent_runs)
        recovery_closure = _execution_health_recovery_closure_snapshot(recent_runs)
        approval_held_review_required = _approval_held_review_required(recovery_closure)
        execution_review_debt_label = _execution_review_debt_label(recovery_closure)
        execution_review_state_label = _execution_review_state_label(recovery_closure)
        learning_debt = _execution_learning_debt_snapshot(recent_runs)
        recovery_debt_visible = bool(recovery_closure["blocks_auto_execution"])
        governor_preview_allowed_with_recovery_debt = True
        recovery_closure_blocks_governed_execution = recovery_debt_visible

        planned_actions: list[dict[str, Any]] = []
        approval_required = bool(matched_risks)
        unknown_tools = 0
        exact_arguments_ready = True
        for action in plan.actions:
            tool = tools_by_name.get(action.tool_name)
            if tool is None:
                risk = "UNKNOWN"
                toolset = "unknown"
                requires_approval = True
                description = "Tool is not registered."
                unknown_tools += 1
            else:
                risk = tool.risk.name
                toolset = tool.toolset
                requires_approval = risk in {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"}
                description = tool.description
            action_args = {str(key): _safe_short(value, limit=180) for key, value in action.args.items()}
            exact_arguments_ready = exact_arguments_ready and bool(action_args or action.tool_name in {"respond", "current_time", "readiness_report", "safety_status"})
            approval_required = approval_required or requires_approval
            planned_actions.append(
                {
                    "tool": action.tool_name,
                    "toolset": toolset,
                    "risk": risk,
                    "requires_approval": requires_approval,
                    "args": action_args,
                    "reason": _safe_short(action.reason, limit=180),
                    "description": _safe_short(description, limit=180),
                }
            )

        ambiguous_risky = bool(matched_risks and not planned_actions)
        approval_forecast = _approval_queue_forecast(store, request, plan.actions, planned_actions)
        has_pending_risky_gate = bool(pending_approvals and (approval_required or matched_risks or unknown_tools))
        has_recent_failed_gate = bool(recent_failed and (approval_required or matched_risks or unknown_tools or not planned_actions))

        if has_pending_risky_gate:
            governor_verdict = "HOLD_FOR_APPROVAL_REVIEW"
            route = "approval_review"
            next_command = f"approval readiness {pending_approvals[0]['id']}"
            can_auto_run = False
            safe_to_execute_now = False
            reason = "A risky or ambiguous order is present while approval-gated work is already pending."
        elif approval_held_review_required:
            governor_verdict = "APPROVAL_HELD_REVIEW_REQUIRED"
            route = "approval_held_review"
            next_command = recovery_closure["required_commands"][0] if recovery_closure["required_commands"] else "pending approvals"
            can_auto_run = False
            safe_to_execute_now = False
            reason = "A recent approval-gated run was held before execution, so Jarvis must review the approval packet before trusting more execution."
        elif recovery_closure["blocks_auto_execution"]:
            governor_verdict = "RECOVERY_CLOSURE_REQUIRED"
            route = "recovery_closure"
            next_command = recovery_closure["required_commands"][0] if recovery_closure["required_commands"] else "execution health report"
            can_auto_run = False
            safe_to_execute_now = False
            reason = "Execution health recovery closure is incomplete, so Jarvis must close the proof queue before trusting more execution."
        elif has_recent_failed_gate:
            governor_verdict = "RECOVERY_REVIEW_FIRST"
            route = "recovery_review"
            next_command = f"execution recovery packet {recent_failed[0]['id']}"
            can_auto_run = False
            safe_to_execute_now = False
            reason = "Recent failed or blocked execution must be reviewed before this order is trusted."
        elif approval_required or unknown_tools:
            governor_verdict = "APPROVAL_REQUIRED"
            route = "approval_gated_dispatch"
            next_command = f"dispatch decision: {display_request}"
            can_auto_run = False
            safe_to_execute_now = False
            reason = "The route contains risky wording, approval-gated tools, or an unknown tool."
        elif ambiguous_risky or not exact_arguments_ready:
            governor_verdict = "PLANNER_GAP"
            route = "planner_gap"
            next_command = f"planner gap: {display_request}"
            can_auto_run = False
            safe_to_execute_now = False
            reason = "The order needs exact route and argument proof before it can be executed."
        elif planned_actions:
            governor_verdict = "AUTO_SAFE_READY"
            route = "auto_safe_dispatch"
            next_command = display_request
            can_auto_run = True
            safe_to_execute_now = True
            reason = "The planner selected only read-only or local-safe tools and no blocker is visible."
        elif plan.needs_model:
            governor_verdict = "CHAT_READY"
            route = "chat_brain"
            next_command = display_request
            can_auto_run = True
            safe_to_execute_now = True
            reason = "The order is conversational; Jarvis should answer without claiming tool execution."
        else:
            governor_verdict = "CLARIFY_FIRST"
            route = "clarify"
            next_command = "ask the operator for target, scope, and definition of done"
            can_auto_run = False
            safe_to_execute_now = False
            reason = "No clear chat or tool route is available."

        stop_conditions = [
            "target, file, account, command, coordinate, recipient, or success condition is vague",
            "planned risk is higher than the route label claims",
            "approval-gated work lacks approval readiness, an approval packet, approval chain proof, and exact rerun command",
            "recent failed or blocked run has not been reviewed",
            "approval-held execution review is missing approval proof" if approval_held_review_required else "execution health recovery closure is missing target verification, recovery, learning, or approval-chain proof",
            "verification evidence cannot prove what changed",
        ]
        proof_requirements = [
            "command intake preserves the original order and route",
            "dispatch decision selects chat, auto-safe tools, approval-gated work, recovery, or clarification",
            "argument contract shows exact tool arguments before execution",
            "execution readiness matrix records route, risk, approvals, proof, recovery, and learning state",
            "verification packet defines evidence requirements and failure signals",
            "execution learning debt is visible and reviewed before completion claims",
            "execution audit gate or runtime trace proves every executed tool result before completion is claimed",
        ]
        required_followups = [
            f"command intake: {display_request}",
            f"dispatch decision: {display_request}",
            f"execution readiness matrix: {display_request}",
            f"verification packet: {display_request}",
            "execution audit gate",
        ]
        if approval_required or matched_risks or unknown_tools:
            required_followups.insert(2, f"argument contract: {display_request}")
        if pending_approvals:
            required_followups.append(f"approval readiness {pending_approvals[0]['id']}")
            required_followups.append(f"approval packet {pending_approvals[0]['id']}")
        if recent_failed:
            required_followups.append(f"execution recovery packet {recent_failed[0]['id']}")
        for command in recovery_closure["required_commands"]:
            if command not in required_followups:
                required_followups.append(command)

        lines = [
            "Jarvis execution governor packet:",
            "This is the command-first harness governor: intake, dispatch, proof, recovery, and approval state in one read-only go/no-go packet.",
            "",
            f"Order: {display_request}",
            f"Planner goal: {display_goal}",
            f"Governor verdict: {governor_verdict}",
            f"Route: {route}",
            f"Can auto-run now: {'yes' if can_auto_run else 'no'}",
            f"Safe to execute now: {'yes' if safe_to_execute_now else 'no'}",
            f"Reason: {reason}",
            f"Next command: `{next_command}`",
            "",
            "Planned actions:",
        ]
        if planned_actions:
            for index, item in enumerate(planned_actions, start=1):
                gate = "approval-gated" if item["requires_approval"] else "auto-safe"
                lines.append(f"- {index}. {item['tool']} [{item['toolset']}, {item['risk']}]: {gate}")
                if item["args"]:
                    arg_text = ", ".join(f"{key}={value!r}" for key, value in sorted(item["args"].items()))
                    lines.append(f"  Args: {arg_text}")
        elif plan.needs_model:
            lines.append("- chat response only; no tool execution should be claimed")
        else:
            lines.append("- no exact tool route yet")

        lines.extend(["", "Governor gates:"])
        lines.append(f"- risk signals: {', '.join(matched_risks) if matched_risks else 'none from wording'}")
        lines.append(f"- approval required: {'yes' if approval_required else 'no'}")
        lines.append(f"- approval review required: {'yes' if has_pending_risky_gate else 'no'}")
        lines.append(f"- would queue new approvals if sent: {approval_forecast['forecast_new_approvals']}")
        lines.append(f"- would reuse pending approval ids: {', '.join(str(item) for item in approval_forecast['forecast_reused_approval_ids']) if approval_forecast['forecast_reused_approval_ids'] else 'none'}")
        lines.append(f"- forecast queue after if sent: {approval_forecast['forecast_queue_after_if_sent']}")
        lines.append(f"- exact arguments ready: {'yes' if exact_arguments_ready else 'no'}")
        lines.append(f"- pending approvals visible: {len(pending_approvals)}")
        lines.append(f"- recent failed/blocked runs visible: {len(recent_failed)}")
        lines.append(f"- recent approval-held runs visible: {len(recent_approval_held)}")
        lines.append(f"- {execution_review_state_label} state: {recovery_closure['state']}")
        lines.append(f"- {execution_review_state_label} ready to retry: {'yes' if recovery_closure['ready_to_retry'] else 'no'}")
        lines.append(f"- {execution_review_state_label} missing: {', '.join(recovery_closure['missing']) if recovery_closure['missing'] else 'none'}")
        lines.append(f"- {execution_review_state_label} next required: `{recovery_closure['next_required_command']}`" if recovery_closure["next_required_command"] else f"- {execution_review_state_label} next required: none")
        lines.append(f"- {execution_review_state_label} command queue: {', '.join(f'`{command}`' for command in recovery_closure['required_commands']) if recovery_closure['required_commands'] else 'none'}")
        lines.append(f"- {execution_review_debt_label} blocks governed execution: {'yes' if recovery_closure_blocks_governed_execution else 'no'}")
        lines.append(f"- {execution_review_debt_label} blocks this governor preview: no, this packet is read-only and does not authorize execution")
        lines.append(f"- execution learning debt state: {learning_debt['state']}")
        lines.append(f"- execution learning blocks completion: {'yes' if learning_debt['blocks_completion_claim'] else 'no'}")
        lines.append(f"- execution learning next required: `{learning_debt['next_required_command']}`" if learning_debt["next_required_command"] else "- execution learning next required: none")
        if approval_held_review_required:
            lines.extend(
                [
                    "",
                    "Approval-held execution review:",
                    f"- target run: #{recovery_closure['target_run_id']} `{recovery_closure['target_tool_name']}`" if recovery_closure["target_run_id"] is not None else "- target run: none",
                    f"- missing: {', '.join(recovery_closure['missing']) if recovery_closure['missing'] else 'none'}",
                    f"- next required: `{recovery_closure['next_required_command']}`" if recovery_closure["next_required_command"] else "- next required: none",
                    f"- command queue: {', '.join(f'`{command}`' for command in recovery_closure['required_commands']) if recovery_closure['required_commands'] else 'none'}",
                ]
            )
        lines.extend(["", "Required proof before completion:"])
        lines.extend(f"- {item}" for item in proof_requirements)
        lines.extend(["", "Stop conditions:", f"- Operator limit: {OPERATOR_LIMIT_RULE}"])
        lines.extend(f"- {item}" for item in stop_conditions)
        lines.extend(["", "Required follow-up packets:"])
        lines.extend(f"- `{command}`" for command in required_followups)
        lines.extend(_execution_learning_lines(learning_debt))
        lines.extend(
            [
                "",
                "Boundary:",
                "- This governor packet does not call a model, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, speak, complete tasks, or queue approvals.",
            ]
        )

        return ToolResult(
            "execution_governor_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=display_request,
                display_request=display_request,
                governor_verdict=governor_verdict,
                route=route,
                next_command=next_command,
                can_auto_run=can_auto_run,
                safe_to_execute_now=safe_to_execute_now,
                recovery_debt_visible=recovery_debt_visible,
                governor_preview_allowed_with_recovery_debt=governor_preview_allowed_with_recovery_debt,
                recovery_closure_blocks_governed_execution=recovery_closure_blocks_governed_execution,
                reason=reason,
                matched_risks=matched_risks,
                approval_required=approval_required,
                approval_review_required=has_pending_risky_gate,
                pending_approvals=len(pending_approvals),
                **approval_forecast,
                recent_failed_runs=len(recent_failed),
                recent_approval_held_runs=len(recent_approval_held),
                recovery_closure_state=recovery_closure["state"],
                recovery_closure_ready_to_retry=recovery_closure["ready_to_retry"],
                recovery_closure_missing=recovery_closure["missing"],
                recovery_closure_missing_count=recovery_closure["missing_count"],
                recovery_closure_required_commands=recovery_closure["required_commands"],
                recovery_closure_next_required_command=recovery_closure["next_required_command"],
                recovery_closure_blocks_auto_execution=recovery_closure["blocks_auto_execution"],
                approval_held_review_required=approval_held_review_required,
                approval_held_review_commands=list(recovery_closure["required_commands"]) if approval_held_review_required else [],
                recovery_closure_target_run_id=recovery_closure["target_run_id"],
                recovery_closure_target_tool_name=recovery_closure["target_tool_name"],
                **_recovery_closure_proof_metadata(recovery_closure),
                **_execution_learning_metadata(learning_debt),
                planned_actions=planned_actions,
                planned_action_count=len(planned_actions),
                exact_arguments_ready=exact_arguments_ready,
                ambiguous_risky_order=ambiguous_risky,
                unknown_tools=unknown_tools,
                proof_requirements=len(proof_requirements),
                stop_conditions=len(stop_conditions),
                required_followups=len(required_followups),
                risk_gated_tools=len(risky_tools),
                execution_governor_handoff_ready=True,
                execution_governor_handoff=_frontdoor_handoff(
                    source="execution_governor_packet",
                    status="ok",
                    display_request=display_request,
                    governor_verdict=governor_verdict,
                    route=route,
                    next_command=next_command,
                    can_auto_run=can_auto_run,
                    safe_to_execute_now=safe_to_execute_now,
                    reason=reason,
                    matched_risks=matched_risks,
                    matched_risk_count=len(matched_risks),
                    approval_required=approval_required,
                    approval_review_required=has_pending_risky_gate,
                    pending_approval_count=len(pending_approvals),
                    recent_failed_runs=len(recent_failed),
                    recent_approval_held_runs=len(recent_approval_held),
                    planned_actions=planned_actions,
                    planned_action_count=len(planned_actions),
                    exact_arguments_ready=exact_arguments_ready,
                    ambiguous_risky_order=ambiguous_risky,
                    unknown_tools=unknown_tools,
                    recovery_debt_visible=recovery_debt_visible,
                    governor_preview_allowed_with_recovery_debt=governor_preview_allowed_with_recovery_debt,
                    recovery_closure_blocks_governed_execution=recovery_closure_blocks_governed_execution,
                    approval_held_review_required=approval_held_review_required,
                    approval_held_review_commands=list(recovery_closure["required_commands"]) if approval_held_review_required else [],
                    recovery_closure_state=recovery_closure["state"],
                    recovery_closure_ready_to_retry=recovery_closure["ready_to_retry"],
                    recovery_closure_missing_count=recovery_closure["missing_count"],
                    recovery_closure_proof_queue=recovery_closure["proof_queue"],
                    recovery_closure_proof_queue_count=recovery_closure["proof_queue_count"],
                    recovery_closure_next_proof_command=recovery_closure["next_proof_command"],
                    execution_learning_proof_queue=learning_debt["proof_queue"],
                    execution_learning_proof_queue_count=learning_debt["proof_queue_count"],
                    execution_learning_next_proof_command=learning_debt["next_required_command"],
                    recommended_next_commands=required_followups,
                    proof_requirement_count=len(proof_requirements),
                    stop_condition_count=len(stop_conditions),
                    required_followup_count=len(required_followups),
                    risk_gated_tool_count=len(risky_tools),
                ),
            ),
        )

    def planner_gap_packet(args: dict[str, Any]) -> ToolResult:
        request = _short(args.get("request") or args.get("order") or args.get("goal"))
        if not request:
            return _autonomy_input_refusal(
                "planner_gap_packet",
                "Tell Jarvis the order to gap-check, for example: `planner gap: organize my downloads and summarize what changed`.",
                _safe_metadata(
                    reason="missing_request",
                    planner_gap_handoff_ready=True,
                    planner_gap_handoff=_frontdoor_handoff(
                        source="planner_gap_packet",
                        status="refused",
                        reason="missing_request",
                        next_command="planner gap: organize my downloads and summarize what changed",
                        recommended_next_commands=[
                            "planner gap: organize my downloads and summarize what changed",
                            "command intake: organize my downloads and summarize what changed",
                            "dispatch decision: organize my downloads and summarize what changed",
                        ],
                    ),
                ),
            )
        display_request = _safe_short(request)

        from jarvis_v2.agent.planner import RuleBasedPlanner

        tools_by_name = {tool.name: tool for tool in list_tools()}
        plan = RuleBasedPlanner().plan(request, allow_risky_natural_dispatch=False)
        display_goal = _safe_short(plan.goal)
        matched_risks, risky_tools = classify_request(request)
        pending_approvals = store.list_pending_approvals(limit=5) if store is not None else []

        planned_actions: list[dict[str, Any]] = []
        unknown_tools = 0
        approval_required = False
        for action in plan.actions:
            tool = tools_by_name.get(action.tool_name)
            if tool is None:
                risk = "UNKNOWN"
                toolset = "unknown"
                requires_approval = True
                description = "Tool is not registered."
                unknown_tools += 1
            else:
                risk = tool.risk.name
                toolset = tool.toolset
                requires_approval = risk in {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"}
                description = tool.description
            approval_required = approval_required or requires_approval
            planned_actions.append(
                {
                    "tool": action.tool_name,
                    "toolset": toolset,
                    "risk": risk,
                    "requires_approval": requires_approval,
                    "args": {str(key): _safe_short(value, limit=180) for key, value in action.args.items()},
                    "reason": _safe_short(action.reason, limit=180),
                    "description": _safe_short(description, limit=180),
                }
            )

        gap_signals: list[str] = []
        if matched_risks and not planned_actions:
            gap_signals.append("risky wording but no exact tool route")
        if plan.needs_model and matched_risks:
            gap_signals.append("model fallback would be asked to reason about risky work")
        if unknown_tools:
            gap_signals.append("planner referenced an unregistered tool")
        if not planned_actions and not plan.needs_model:
            gap_signals.append("no chat or tool route")
        if pending_approvals and (matched_risks or approval_required):
            gap_signals.append("pending approval should be reviewed before new risky route")

        if matched_risks and not planned_actions:
            classification = "PLANNER_GAP"
            next_command = f"execution governor: {display_request}"
        elif unknown_tools:
            classification = "REGISTRY_GAP"
            next_command = "list tools"
        elif planned_actions:
            classification = "ROUTED"
            next_command = f"execution governor: {display_request}"
        elif plan.needs_model:
            classification = "CHAT_ROUTE"
            next_command = f"execution governor: {display_request}"
        else:
            classification = "UNCLEAR"
            next_command = "ask the operator for exact target, allowed scope, and success condition"

        suggested_followups = [
            f"execution governor: {display_request}",
            f"dispatch decision: {display_request}",
            f"execution readiness matrix: {display_request}",
            f"verification packet: {display_request}",
        ]
        if matched_risks:
            suggested_followups.insert(1, f"risk preflight: {display_request}")
            suggested_followups.insert(2, f"action rehearsal: {display_request}")
        if matched_risks and not planned_actions:
            suggested_followups.append("add a deterministic planner pattern only after the exact tool contract is clear")
            suggested_followups.append("add a smoke test that proves the route holds for approval instead of becoming vague chat")
        if unknown_tools:
            suggested_followups.append("register the missing tool or change the planner to an existing tool")

        lines = [
            "Jarvis planner gap packet:",
            "This is a read-only harness check for whether the steering layer can turn an order into an exact route.",
            "",
            f"Order: {display_request}",
            f"Planner goal: {display_goal}",
            f"Classification: {classification}",
            f"Next safe command: `{next_command}`",
            "",
            "Gap signals:",
        ]
        if gap_signals:
            lines.extend(f"- {item}" for item in gap_signals)
        else:
            lines.append("- none")

        lines.extend(["", "Planned route:"])
        if planned_actions:
            for index, item in enumerate(planned_actions, start=1):
                gate = "approval-gated" if item["requires_approval"] else "auto-safe"
                lines.append(f"- {index}. {item['tool']} [{item['toolset']}, {item['risk']}]: {gate}")
        elif plan.needs_model:
            lines.append("- chat brain route; no exact tool action planned")
        else:
            lines.append("- no route planned")

        lines.extend(["", "Suggested follow-up commands:"])
        lines.extend(f"- `{item}`" for item in suggested_followups)
        lines.extend(
            [
                "",
                "Boundary:",
                "- This packet does not call a model, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, speak, complete tasks, or queue approvals.",
            ]
        )

        return ToolResult(
            "planner_gap_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=display_request,
                display_request=display_request,
                classification=classification,
                next_command=next_command,
                matched_risks=matched_risks,
                gap_signals=gap_signals,
                gap_signal_count=len(gap_signals),
                pending_approvals=len(pending_approvals),
                planned_actions=planned_actions,
                planned_action_count=len(planned_actions),
                approval_required=approval_required or bool(matched_risks),
                needs_model=plan.needs_model,
                unknown_tools=unknown_tools,
                suggested_followups=len(suggested_followups),
                risk_gated_tools=len(risky_tools),
                planner_gap_handoff_ready=True,
                planner_gap_handoff=_frontdoor_handoff(
                    source="planner_gap_packet",
                    status="ok",
                    display_request=display_request,
                    classification=classification,
                    next_command=next_command,
                    matched_risks=matched_risks,
                    matched_risk_count=len(matched_risks),
                    gap_signals=gap_signals,
                    gap_signal_count=len(gap_signals),
                    pending_approval_count=len(pending_approvals),
                    planned_actions=planned_actions,
                    planned_action_count=len(planned_actions),
                    approval_required=approval_required or bool(matched_risks),
                    needs_model=plan.needs_model,
                    unknown_tools=unknown_tools,
                    recommended_next_commands=suggested_followups,
                    suggested_followup_count=len(suggested_followups),
                    risk_gated_tool_count=len(risky_tools),
                ),
            ),
        )

    def agent_loop_preview(args: dict[str, Any]) -> ToolResult:
        goal = _short(args.get("goal") or args.get("request"))
        if not goal:
            return _autonomy_input_refusal(
                "agent_loop_preview",
                "Tell me the goal for Jarvis's second loop.",
            )
        display_goal = _safe_short(goal)

        matched_risks, risky_tools = classify_request(goal)
        suggested_steps = [
            "Clarify the desired outcome and stop condition.",
            "Load visible context from memory, tasks, goals, notes, recent conversation, and pending approvals.",
            "Route through execution governor before choosing tools.",
            "Run risk preflight and action rehearsal under the governor to preview exact tool calls and approval requirements.",
            "Execute only read-only/local-safe steps automatically; queue approval receipts for risky steps.",
            "Verify the result through tool output, audit logs, notes, or screen verification after approved computer-control steps.",
            "Record a closeout: what changed, what stayed blocked, and the next safe action.",
        ]
        preview_commands = [
            f"execution governor: {display_goal}",
            f"risk preflight: {display_goal}",
            f"action rehearsal: {display_goal}",
            f"autonomy plan: {display_goal}",
            f"next session plan: {display_goal}",
        ]

        lines = [
            "Jarvis second loop preview:",
            "Purpose: show the task/action loop that starts after normal chat decides the operator wants work done.",
            "",
            f"Goal: {display_goal}",
            "",
            "Loop shape:",
        ]
        for index, step in enumerate(suggested_steps, start=1):
            lines.append(f"- {index}. {step}")

        lines.extend(["", "Risk gates for this goal:"])
        if matched_risks:
            for label in matched_risks:
                lines.append(f"- {label}: likely approval-gated before execution.")
        else:
            lines.append("- No obvious risky wording detected; the registry still decides risk at tool time.")

        lines.extend(
            [
                "",
                "Useful preview commands:",
                *[f"- `{command}`" for command in preview_commands],
                "",
                "Stop conditions:",
                f"- Operator limit: {OPERATOR_LIMIT_RULE}",
                "- Stop when the goal is done, blocked by approval, ambiguous, unsafe, or no verification signal is available.",
                "- Never repeat a failed risky action automatically.",
                "- Ask the operator before broad computer-control, shell/code, destructive file, personal-data, reminder, or external-side-effect work.",
                "",
                "Boundary:",
                "- This preview is read-only and does not call a model, execute tools, approve requests, dismiss approvals, write notes, read private data, control the computer, or queue approvals.",
            ]
        )

        return ToolResult(
            "agent_loop_preview",
            True,
            "\n".join(lines),
            _safe_metadata(
                goal=display_goal,
                display_goal=display_goal,
                steps=len(suggested_steps),
                matched_risks=matched_risks,
                risk_gated_tools=len(risky_tools),
                next_command=preview_commands[0],
                recommended_next_commands=preview_commands,
            ),
        )

    def agent_loop_packet(args: dict[str, Any]) -> ToolResult:
        goal = _short(args.get("goal") or args.get("request"))
        if not goal:
            return _autonomy_input_refusal(
                "agent_loop_packet",
                "Tell me the goal for Jarvis's task loop packet.",
            )
        display_goal = _safe_short(goal)

        matched_risks, risky_tools = classify_request(goal)
        likely_approval = bool(matched_risks)
        phases = [
            ("Orient", "Load return brief, current work queue, relevant memory, and pending approvals."),
            ("Define done", "Write the observable success condition and the stop condition before any action."),
            ("Govern", "Route through execution governor before choosing tools."),
            ("Preflight", "Run risk preflight and action rehearsal under the governor to inspect planned tools and arguments."),
            ("Act", "Run only read-only/local-safe actions automatically; risky work creates approval receipts."),
            ("Verify", "Check tool output, audit records, notes, files, or screen state after approved actions."),
            ("Close out", "Report what changed, what stayed blocked, and the safest next action."),
        ]
        preview_commands = [
            f"execution governor: {display_goal}",
            f"risk preflight: {display_goal}",
            f"action rehearsal: {display_goal}",
            f"autonomy plan: {display_goal}",
            f"risky request lifecycle: {display_goal}",
        ]

        lines = [
            "Jarvis second loop packet:",
            "This is a read-only task packet for the work loop after chat decides the operator wants action.",
            "",
            f"Goal: {display_goal}",
            "",
            "Definition of done:",
            "- A concrete result is visible, verified, and summarized for the operator.",
            "- Risky steps are either approved through a last-look packet or left blocked with the approval id.",
            "",
            "Loop packet:",
        ]
        for index, (name, body) in enumerate(phases, start=1):
            lines.append(f"- {index}. {name}: {body}")

        lines.extend(["", "Risk gates:"])
        if matched_risks:
            for label in matched_risks:
                lines.append(f"- {label}: approval likely required before execution.")
        else:
            lines.append("- No obvious risky wording detected; ToolRegistry and PermissionPolicy still decide at execution time.")

        lines.extend(
            [
                "",
                "Recommended next preview commands:",
                *[f"- `{command}`" for command in preview_commands],
                "",
                "Approval rule:",
                "- If the real request queues an approval, inspect `approval readiness #ID`, then the last-look `approval packet #ID`, then `approval chain proof #ID`, before `approve approval #ID`.",
                "- An approval reruns only the exact queued request once, then the result should appear in `recent tool runs`.",
                "",
                "Hard stops:",
                "- Stop if the target, account, file path, command, coordinate, or expected result is unclear.",
                "- Stop if verification is unavailable.",
                "- Stop after any failed risky action; do not retry it automatically.",
                "",
                "Boundary:",
                "- This packet does not call a model, execute tools, approve requests, dismiss approvals, write notes, read personal data, control the computer, call external services, or queue approvals.",
            ]
        )

        return ToolResult(
            "agent_loop_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                goal=display_goal,
                display_goal=display_goal,
                phases=len(phases),
                matched_risks=matched_risks,
                risk_gated_tools=len(risky_tools),
                likely_approval_required=likely_approval,
                next_command=preview_commands[0],
                recommended_next_commands=preview_commands,
            ),
        )

    def risky_request_lifecycle(args: dict[str, Any]) -> ToolResult:
        request = _short(args.get("request") or args.get("order") or args.get("goal"))
        if not request:
            return _autonomy_input_refusal(
                "risky_request_lifecycle",
                "Tell me the risky request to map.",
            )
        display_request = _safe_short(request)

        matched_risks, risky_tools = classify_request(request)
        lifecycle_commands = [
            f"execution governor: {display_request}",
            f"risk preflight: {display_request}",
            f"action rehearsal: {display_request}",
            "pending approvals",
            "approval readiness #ID",
            "approval packet #ID",
            "approval detail #ID",
            "dismiss approval #ID",
        ]
        lines = [
            "Jarvis risky request lifecycle:",
            "This is read-only. It maps the safety path for a risky request without running it.",
            "",
            f"Request: {display_request}",
            "",
            "Detected risk areas:",
        ]
        if matched_risks:
            lines.extend(f"- {label}" for label in matched_risks)
        else:
            lines.append("- none from keyword scan; ToolRegistry still decides risk at execution time")

        lines.extend(
            [
                "",
                "Lifecycle:",
                "1. Govern: run `execution governor: <request>` to choose the route, blockers, and proof path.",
                "2. Preflight: run `risk preflight: <request>` to classify likely risk under the governor.",
                "3. Rehearse: run `action rehearsal: <request>` to preview exact planned tools and arguments.",
                "4. Request for real: if the operator still wants it, send the exact request normally.",
                "5. Safety receipt: HIGH_RISK, PERSONAL_DATA, and EXTERNAL_SIDE_EFFECT tools queue a pending approval instead of running.",
                "6. Readiness: run `approval readiness #ID` to check queue position, staleness, and exact stored arguments.",
                "7. Last-look packet: run `approval packet #ID` to preview the exact queued rerun.",
                "8. Decide: approve only the exact trusted request, or dismiss stale/broad/surprising approvals.",
                "9. Rerun and audit: approved requests rerun once and appear in `recent tool runs` and continuity reports.",
                "",
                "Stop conditions:",
                f"- Operator limit: {OPERATOR_LIMIT_RULE}",
                "- Stop if the target, account, file, command, coordinate, or expected result is unclear.",
                "- Stop if the request is broad, destructive, private, stale, or mismatched with what the operator wants now.",
                "- Stop if verification cannot prove what changed.",
                "",
                "Useful commands:",
                *[f"- `{command}`" for command in lifecycle_commands],
                "",
                "Boundary:",
                "- This lifecycle map does not call a model, execute tools, approve requests, dismiss approvals, read personal data, write files, control the computer, call external services, or queue approvals.",
            ]
        )

        return ToolResult(
            "risky_request_lifecycle",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=display_request,
                display_request=display_request,
                matched_risks=matched_risks,
                risk_gated_tools=len(risky_tools),
                next_command=lifecycle_commands[0],
                recommended_next_commands=lifecycle_commands,
            ),
        )

    def command_cockpit_packet(args: dict[str, Any]) -> ToolResult:
        request = _short(args.get("request") or args.get("order") or args.get("message") or args.get("goal"))
        if not request:
            return _autonomy_input_refusal(
                "command_cockpit_packet",
                "Tell Jarvis the order to cockpit-check, for example: `command cockpit: organize my downloads and summarize what changed`.",
                _safe_metadata(
                    reason="missing_request",
                    command_cockpit_handoff_ready=True,
                    command_cockpit_handoff=_frontdoor_handoff(
                        source="command_cockpit_packet",
                        status="refused",
                        reason="missing_request",
                        next_command="command cockpit: organize my downloads and summarize what changed",
                        recommended_next_commands=[
                            "command cockpit: organize my downloads and summarize what changed",
                            "execution governor: organize my downloads and summarize what changed",
                            "command intake: organize my downloads and summarize what changed",
                        ],
                    ),
                ),
            )
        display_request = _safe_short(request)

        cockpit_args = {"request": request}
        intake = command_intake_packet(cockpit_args)
        governor = execution_governor_packet(cockpit_args)
        dispatch = dispatch_decision_packet(cockpit_args)
        matrix = execution_readiness_matrix(cockpit_args)
        verification = verification_packet(cockpit_args)
        packets = [intake, governor, dispatch, matrix, verification]
        packet_rows = [
            {
                "tool": packet.tool_name,
                "ok": packet.ok,
                "verdict": packet.metadata.get("governor_verdict") or packet.metadata.get("decision") or packet.metadata.get("verdict") or packet.metadata.get("intake_state") or "",
                "route": packet.metadata.get("route") or packet.metadata.get("runtime_route") or packet.metadata.get("route_label") or "",
                "next_command": packet.metadata.get("next_command") or packet.metadata.get("primary_command") or "",
            }
            for packet in packets
        ]

        governor_meta = governor.metadata
        dispatch_meta = dispatch.metadata
        matrix_meta = matrix.metadata
        verification_meta = verification.metadata
        next_command = (
            governor_meta.get("next_command")
            or dispatch_meta.get("primary_command")
            or matrix_meta.get("next_command")
            or verification_meta.get("next_command")
            or f"execution governor: {display_request}"
        )
        can_auto_run = (
            _metadata_all_bool(governor_meta.get("can_auto_run"), dispatch_meta.get("can_auto_run"))
            and matrix_meta.get("verdict") in {"READY_FOR_AUTO_SAFE_TOOL_ROUTE", "READY_FOR_CHAT_BRAIN"}
        )
        recovery_blocks = _metadata_any_bool(
            governor_meta.get("recovery_closure_blocks_auto_execution"),
            dispatch_meta.get("recovery_closure_blocks_auto_execution"),
            matrix_meta.get("recovery_closure_blocks_auto_execution"),
        )
        recovery_debt_visible = _metadata_any_bool(
            governor_meta.get("recovery_debt_visible"),
            dispatch_meta.get("recovery_debt_visible"),
            matrix_meta.get("recovery_debt_visible"),
            recovery_blocks,
        )
        cockpit_preview_allowed_with_recovery_debt = True
        recovery_closure_blocks_cockpit_execution = recovery_debt_visible
        approval_required = _metadata_any_bool(
            governor_meta.get("approval_required"),
            dispatch_meta.get("approval_required"),
            matrix_meta.get("approval_required"),
            verification_meta.get("approval_required"),
        )
        pending_approvals = max(
            _metadata_int(governor_meta.get("pending_approvals")),
            _metadata_int(dispatch_meta.get("pending_approvals")),
            _metadata_int(matrix_meta.get("pending_approvals")),
        )
        learning_blocks = _metadata_any_bool(
            governor_meta.get("execution_learning_blocks_completion_claim"),
            dispatch_meta.get("execution_learning_blocks_completion_claim"),
            matrix_meta.get("execution_learning_blocks_completion_claim"),
        )
        recovery_closure_state = str(matrix_meta.get("recovery_closure_state") or "unknown")
        recovery_closure_missing = list(matrix_meta.get("recovery_closure_missing") or [])
        recovery_closure_required_commands = list(matrix_meta.get("recovery_closure_required_commands") or [])
        recovery_closure_proof_queue = list(matrix_meta.get("recovery_closure_proof_queue") or [])
        recovery_closure_target_run_id = matrix_meta.get("recovery_closure_target_run_id")
        recovery_closure_target_tool_name = str(matrix_meta.get("recovery_closure_target_tool_name") or "")
        recovery_closure_next_required_command = str(matrix_meta.get("recovery_closure_next_required_command") or "")
        recovery_closure_next_proof_command = str(matrix_meta.get("recovery_closure_next_proof_command") or "")
        recovery_closure_ready_to_retry = _metadata_bool(matrix_meta.get("recovery_closure_ready_to_retry"))
        approval_held_review_required = recovery_closure_state == "approval_held_review_required"
        execution_review_debt_label = "approval review" if approval_held_review_required else "recovery debt"
        if pending_approvals:
            cockpit_verdict = "HOLD_FOR_APPROVAL_REVIEW"
        elif approval_held_review_required:
            cockpit_verdict = "APPROVAL_HELD_REVIEW_REQUIRED"
        elif recovery_blocks:
            cockpit_verdict = "RECOVERY_CLOSURE_REQUIRED"
        elif approval_required:
            cockpit_verdict = "APPROVAL_REQUIRED_BEFORE_EXECUTION"
        elif not all(packet.ok for packet in packets):
            cockpit_verdict = "COCKPIT_PACKET_INCOMPLETE"
        elif can_auto_run:
            cockpit_verdict = "READY_FOR_SAFE_DISPATCH"
        else:
            cockpit_verdict = "PREFLIGHT_OR_CLARIFICATION_REQUIRED"

        required_commands: list[str] = []
        for command in [
            f"command intake: {display_request}",
            f"execution governor: {display_request}",
            f"dispatch decision: {display_request}",
            f"execution readiness matrix: {display_request}",
            f"verification packet: {display_request}",
        ]:
            if command not in required_commands:
                required_commands.append(command)
        for metadata in [governor_meta, dispatch_meta, matrix_meta, verification_meta]:
            for key in ("required_followups", "recommended_next_commands", "recovery_closure_required_commands", "execution_learning_required_commands"):
                raw_commands = metadata.get(key, []) or []
                if isinstance(raw_commands, str):
                    commands_to_add = [raw_commands]
                elif isinstance(raw_commands, list):
                    commands_to_add = raw_commands
                else:
                    commands_to_add = []
                for command in commands_to_add:
                    if command and command not in required_commands:
                        required_commands.append(command)

        lines = [
            "Jarvis command cockpit packet:",
            "This is the command-first dashboard for one natural order. It is read-only and composes intake, governor, dispatch, readiness, and verification packets without running the order.",
            "",
            f"Order: {display_request}",
            f"Cockpit verdict: {cockpit_verdict}",
            f"Can auto-run now: {'yes' if can_auto_run else 'no'}",
            f"Next safe command: `{next_command}`",
            "",
            "Cockpit instruments:",
            f"- intake: {intake.metadata.get('intake_state', 'unknown')} -> `{intake.metadata.get('next_command') or 'none'}`",
            f"- governor: {governor_meta.get('governor_verdict', 'unknown')} via {governor_meta.get('route', 'unknown')} -> `{governor_meta.get('next_command') or 'none'}`",
            f"- dispatch: {dispatch_meta.get('decision', 'unknown')} via {dispatch_meta.get('route', 'unknown')} -> `{dispatch_meta.get('primary_command') or 'none'}`",
            f"- readiness matrix: {matrix_meta.get('verdict', 'unknown')} -> `{matrix_meta.get('next_command') or 'none'}`",
            f"- verification: {verification_meta.get('verdict', 'unknown')} -> `{verification_meta.get('next_command') or 'none'}`",
            "",
            "Operator-facing decision:",
        ]
        if cockpit_verdict == "READY_FOR_SAFE_DISPATCH":
            lines.append("- Ready for the normal request path only if ToolRegistry still selects read-only or local-safe tools.")
        elif cockpit_verdict == "HOLD_FOR_APPROVAL_REVIEW":
            lines.append("- Hold new risky work and review the existing approval queue first.")
        elif cockpit_verdict == "APPROVAL_HELD_REVIEW_REQUIRED":
            lines.append("- Review the approval-held execution before treating more work as ready.")
        elif cockpit_verdict == "RECOVERY_CLOSURE_REQUIRED":
            lines.append("- Close the execution recovery proof queue before treating more work as ready.")
        elif cockpit_verdict == "APPROVAL_REQUIRED_BEFORE_EXECUTION":
            lines.append("- Prepare proof and approval review before any real execution.")
        else:
            lines.append("- Use preflight or clarification before execution.")

        lines.extend(
            [
                "",
                "Approval-held execution review:" if approval_held_review_required else "Recovery closure:",
                f"- state: {recovery_closure_state}",
                f"- ready for retry review: {'yes' if recovery_closure_ready_to_retry else 'no'}",
                f"- target run: #{recovery_closure_target_run_id} `{recovery_closure_target_tool_name}`" if recovery_closure_target_run_id is not None else "- target run: none",
                f"- missing: {', '.join(recovery_closure_missing) if recovery_closure_missing else 'none'}",
                f"- next closure required: `{recovery_closure_next_required_command}`" if recovery_closure_next_required_command else "- next closure required: none",
                f"- closure proof queue: {', '.join(f'`{command}`' for command in recovery_closure_proof_queue) if recovery_closure_proof_queue else 'none'}",
                f"- {execution_review_debt_label} blocks cockpit execution: {'yes' if recovery_closure_blocks_cockpit_execution else 'no'}",
                f"- {execution_review_debt_label} blocks this cockpit preview: no, this packet is read-only and does not authorize execution",
                "",
                "Execution learning debt:",
                f"- state: {matrix_meta.get('execution_learning_state', 'unknown')}",
                f"- blocks completion claim: {'yes' if learning_blocks else 'no'}",
                f"- target run: #{matrix_meta.get('execution_learning_target_run_id')} `{matrix_meta.get('execution_learning_target_tool_name')}`" if matrix_meta.get("execution_learning_target_run_id") is not None else "- target run: none",
                f"- missing: {', '.join(matrix_meta.get('execution_learning_missing') or []) if matrix_meta.get('execution_learning_missing') else 'none'}",
                f"- next learning required: `{matrix_meta.get('execution_learning_next_required_command')}`" if matrix_meta.get("execution_learning_next_required_command") else "- next learning required: none",
                "",
                "Required proof queue:",
                *[f"- `{command}`" for command in required_commands],
                "",
                "Learning and completion rule:",
                f"- execution learning blocks completion claim: {'yes' if learning_blocks else 'no'}",
                "- The cockpit can make a route ready, but it cannot by itself prove the task is done.",
                "",
                "Boundary:",
                "- This cockpit does not call a model, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, speak, complete tasks, or queue approvals.",
            ]
        )

        return ToolResult(
            "command_cockpit_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=display_request,
                display_request=display_request,
                cockpit_verdict=cockpit_verdict,
                next_command=next_command,
                can_auto_run=can_auto_run,
                approval_required=approval_required,
                pending_approvals=pending_approvals,
                recovery_closure_blocks_auto_execution=recovery_blocks,
                approval_held_review_required=approval_held_review_required,
                approval_held_review_commands=recovery_closure_required_commands if approval_held_review_required else [],
                recovery_debt_visible=recovery_debt_visible,
                cockpit_preview_allowed_with_recovery_debt=cockpit_preview_allowed_with_recovery_debt,
                recovery_closure_blocks_cockpit_execution=recovery_closure_blocks_cockpit_execution,
                recovery_closure_state=recovery_closure_state,
                recovery_closure_ready_to_retry=recovery_closure_ready_to_retry,
                recovery_closure_missing=recovery_closure_missing,
                recovery_closure_missing_count=matrix_meta.get("recovery_closure_missing_count", len(recovery_closure_missing)),
                recovery_closure_required_commands=recovery_closure_required_commands,
                recovery_closure_next_required_command=recovery_closure_next_required_command,
                recovery_closure_proof_queue=recovery_closure_proof_queue,
                recovery_closure_proof_queue_count=matrix_meta.get("recovery_closure_proof_queue_count", len(recovery_closure_proof_queue)),
                recovery_closure_next_proof_command=recovery_closure_next_proof_command,
                recovery_closure_target_run_id=recovery_closure_target_run_id,
                recovery_closure_target_tool_name=recovery_closure_target_tool_name,
                execution_learning_blocks_completion_claim=learning_blocks,
                execution_learning_state=matrix_meta.get("execution_learning_state", ""),
                execution_learning_recent_action_runs=matrix_meta.get("execution_learning_recent_action_runs", 0),
                execution_learning_failed_or_blocked_action_runs=matrix_meta.get("execution_learning_failed_or_blocked_action_runs", 0),
                execution_learning_recent_verification_runs=matrix_meta.get("execution_learning_recent_verification_runs", 0),
                execution_learning_recent_recovery_runs=matrix_meta.get("execution_learning_recent_recovery_runs", 0),
                execution_learning_recent_after_action_learning_runs=matrix_meta.get("execution_learning_recent_after_action_learning_runs", 0),
                execution_learning_target_run_id=matrix_meta.get("execution_learning_target_run_id"),
                execution_learning_target_tool_name=matrix_meta.get("execution_learning_target_tool_name", ""),
                execution_learning_target_after_action_learning_packets=matrix_meta.get("execution_learning_target_after_action_learning_packets", 0),
                execution_learning_missing=matrix_meta.get("execution_learning_missing", []),
                execution_learning_missing_count=matrix_meta.get("execution_learning_missing_count", 0),
                execution_learning_required_commands=matrix_meta.get("execution_learning_required_commands", []),
                execution_learning_next_required_command=matrix_meta.get("execution_learning_next_required_command", ""),
                execution_learning_proof_queue=matrix_meta.get("execution_learning_proof_queue", []),
                execution_learning_proof_queue_count=matrix_meta.get("execution_learning_proof_queue_count", 0),
                execution_learning_next_proof_command=matrix_meta.get("execution_learning_next_proof_command", ""),
                packet_rows=packet_rows,
                packet_row_count=len(packet_rows),
                required_commands=required_commands,
                required_command_count=len(required_commands),
                proof_queue=required_commands,
                proof_queue_count=len(required_commands),
                governor_verdict=governor_meta.get("governor_verdict", ""),
                governor_route=governor_meta.get("route", ""),
                dispatch_decision=dispatch_meta.get("decision", ""),
                dispatch_route=dispatch_meta.get("route", ""),
                readiness_verdict=matrix_meta.get("verdict", ""),
                verification_verdict=verification_meta.get("verdict", ""),
                command_cockpit_handoff_ready=True,
                command_cockpit_handoff=_frontdoor_handoff(
                    source="command_cockpit_packet",
                    status="ok",
                    display_request=display_request,
                    cockpit_verdict=cockpit_verdict,
                    next_command=next_command,
                    can_auto_run=can_auto_run,
                    approval_required=approval_required,
                    pending_approval_count=pending_approvals,
                    recovery_debt_visible=recovery_debt_visible,
                    cockpit_preview_allowed_with_recovery_debt=cockpit_preview_allowed_with_recovery_debt,
                    recovery_closure_blocks_cockpit_execution=recovery_closure_blocks_cockpit_execution,
                    approval_held_review_required=approval_held_review_required,
                    approval_held_review_commands=recovery_closure_required_commands if approval_held_review_required else [],
                    recovery_closure_state=recovery_closure_state,
                    recovery_closure_ready_to_retry=recovery_closure_ready_to_retry,
                    recovery_closure_missing_count=matrix_meta.get("recovery_closure_missing_count", len(recovery_closure_missing)),
                    recovery_closure_proof_queue=recovery_closure_proof_queue,
                    recovery_closure_proof_queue_count=matrix_meta.get("recovery_closure_proof_queue_count", len(recovery_closure_proof_queue)),
                    recovery_closure_next_proof_command=recovery_closure_next_proof_command,
                    execution_learning_blocks_completion_claim=learning_blocks,
                    execution_learning_state=matrix_meta.get("execution_learning_state", ""),
                    execution_learning_proof_queue=matrix_meta.get("execution_learning_proof_queue", []),
                    execution_learning_proof_queue_count=matrix_meta.get("execution_learning_proof_queue_count", 0),
                    execution_learning_next_proof_command=matrix_meta.get("execution_learning_next_proof_command", ""),
                    packet_rows=packet_rows,
                    packet_row_count=len(packet_rows),
                    recommended_next_commands=required_commands,
                    required_command_count=len(required_commands),
                    proof_queue_count=len(required_commands),
                    governor_verdict=governor_meta.get("governor_verdict", ""),
                    governor_route=governor_meta.get("route", ""),
                    dispatch_decision=dispatch_meta.get("decision", ""),
                    dispatch_route=dispatch_meta.get("route", ""),
                    readiness_verdict=matrix_meta.get("verdict", ""),
                    verification_verdict=verification_meta.get("verdict", ""),
                ),
            ),
        )

    return autonomy_plan, risk_preflight, agent_loop_preview, agent_loop_packet, risky_request_lifecycle, action_readiness_packet, execution_contract, argument_contract_packet, verification_packet, execution_acceptance_gate, execution_readiness_matrix, dispatch_decision_packet, command_intake_packet, execution_governor_packet, planner_gap_packet, command_cockpit_packet
