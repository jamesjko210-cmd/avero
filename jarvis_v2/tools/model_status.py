from __future__ import annotations

import importlib.util
import hashlib
import re
from collections import Counter
from typing import Any

from jarvis_v2.agent.failure_guidance import declare_failure_guidance
from jarvis_v2.agent.types import ToolResult
from jarvis_v2.agent.model_provider import (
    OLLAMA_NUM_CTX,
    OPENAI_SAFETY_IDENTIFIER_MAX_CHARS,
    generate_model_text,
    normalized_model_provider,
    openai_api_key_configured,
    ollama_local_only_policy,
    probe_ollama_models,
    provider_output_token_limit,
    resolve_ollama_destination,
)
from jarvis_v2.config import JarvisConfig


MAX_MODEL_DETAIL_CHARS = 220
MAX_PREVIEW_REQUEST_CHARS = 500
MAX_SPECIALIST_REQUEST_CHARS = 500
MAX_HANDOFF_REQUEST_CHARS = 700
MAX_READINESS_REQUEST_CHARS = 700
MAX_SPECIALIST_DRAFT_CHARS = 1200
MAX_PROPOSAL_ARGUMENT_CHARS = 900
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")
MISSING_SPECIALIST_REQUEST_RECOVERY_ACTION = (
    "Give Jarvis the missing request, then retry through the normal policy."
)


SPECIALIST_PROFILES = {
    "chat": {
        "purpose": "Natural conversation, explanation, and lightweight coaching.",
        "signals": ("talk", "explain", "chat", "discuss", "what is", "why"),
        "handoff": "Use chat context, memory, preferences, and skills; do not execute tools unless planner reroutes.",
        "verification": "Answer should cite current Jarvis state when making project claims and avoid inventing private context.",
    },
    "planner": {
        "purpose": "Turn an actionable order into exact tools, arguments, risk, and verification steps.",
        "signals": ("plan", "route", "task", "do", "execute", "organize", "build"),
        "handoff": "Produce a tool plan for ToolRegistry and PermissionPolicy review before any action.",
        "verification": "Every planned action needs expected output, approval state, and recovery path.",
    },
    "code": {
        "purpose": "Reason about local code edits, tests, failures, and implementation strategy.",
        "signals": ("code", "test", "bug", "implement", "file", "patch", "repo"),
        "handoff": "Prepare code-change intent and verification commands; shell/code execution remains approval-gated in Jarvis runtime.",
        "verification": "Focused test or compile evidence must match the edited surface.",
    },
    "vision": {
        "purpose": "Interpret approved screen observations and compare expected vs observed state.",
        "signals": ("screen", "screenshot", "see", "visual", "click", "window", "button"),
        "handoff": "Use approved observation evidence only; never capture screen contents from this contract.",
        "verification": "Use screen_verification_contract or approved observe-act-verify evidence before continuing.",
    },
    "summarizer": {
        "purpose": "Compress long context, logs, notes, pages, or work history into a faithful brief.",
        "signals": ("summarize", "summary", "brief", "catch me up", "compress", "history"),
        "handoff": "Summarize supplied or already-authorized context, preserving uncertainty and source limits.",
        "verification": "Summary should separate evidence, inference, missing context, and next actions.",
    },
    "reflection": {
        "purpose": "Turn failures, feedback, and repeated workflow into tests, preferences, skills, or safe process changes.",
        "signals": ("learn", "feedback", "failure", "mistake", "improve", "reflect"),
        "handoff": "Create reviewable learning or failure-promotion packets before saving or applying changes.",
        "verification": "Learning changes need explicit evidence and should not rewrite memory from unsupported claims.",
    },
}

SPECIALIST_VERIFIER_TOOLS = {
    "chat": ["chat_response_health", "chat_safety_report", "runtime_trace_receipt"],
    "planner": ["execution_governor_packet", "execution_contract", "action_rehearsal", "verification_packet", "runtime_trace_receipt"],
    "code": ["verification_packet", "verification_receipt", "recent_tool_runs", "runtime_trace_receipt"],
    "vision": ["screen_verification_contract", "computer_task_plan", "runtime_trace_receipt"],
    "summarizer": ["verification_packet", "chat_response_health", "runtime_trace_receipt"],
    "reflection": ["learning_review", "failure_promotion_packet", "runtime_trace_receipt"],
}


def _clean_text(value: Any, *, limit: int) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _ollama_http_api_status(reachable: bool) -> str:
    return f"- Ollama HTTP API: {'reachable' if reachable else 'needs attention'}"


def _ollama_unreachable_guidance(chat_model: str) -> str:
    safe_model = _clean_text(chat_model, limit=120) or "<configured-chat-model>"
    return (
        "- Run `setup check` first. Start Ollama, then retry `model routing status`; run `ollama pull "
        f"{safe_model}` outside Jarvis only if the chat model is still missing."
    )


def _text_sha256(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _specialist_review_token_sha256(
    *,
    request: str,
    proposed_tool: str,
    proposed_arguments: str,
    verification: str,
    primary_specialist: str,
    target_model: str,
    completion_state: str,
) -> str:
    return _text_sha256(
        "\n".join(
            [
                "specialist_review_v1",
                request,
                proposed_tool,
                proposed_arguments,
                verification,
                primary_specialist,
                target_model,
                completion_state,
            ]
        )
    )


def _specialist_action_proposal_review_token_sha256(
    *,
    request: str,
    proposed_tool: str,
    proposed_arguments: str,
    verification: str,
    primary_specialist: str,
    target_model: str,
    risk_level: str,
    review_state: str,
    next_command: str,
    scorecard_rows: list[dict[str, Any]],
    authorizes_model_call: bool = False,
    authorizes_tool_execution: bool = False,
    authorizes_approval: bool = False,
    authorizes_personal_data_read: bool = False,
    authorizes_external_side_effect: bool = False,
    authorizes_executable_action: bool = False,
) -> str:
    scorecard_entries = [
        "|".join(
            [
                str(row.get("item") or ""),
                str(int(row.get("points") or 0)),
                str(int(row.get("max_points") or 0)),
                "passed" if row.get("passed") else "held",
                "required" if row.get("required_before_execution") else "optional",
                "auth_model" if row.get("authorizes_model_call") else "no_model",
                "auth_tool" if row.get("authorizes_tool_execution") else "no_tool",
                "auth_approval" if row.get("authorizes_approval") else "no_approval",
                "auth_personal" if row.get("authorizes_personal_data_read") else "no_personal",
                "auth_external" if row.get("authorizes_external_side_effect") else "no_external",
                "auth_executable_action" if row.get("authorizes_executable_action") else "no_executable_action",
                "reusable" if row.get("reusable_for_next_review") else "not_reusable",
            ]
        )
        for row in scorecard_rows
    ]
    return _text_sha256(
        "\n".join(
            [
                "specialist_action_proposal_review_v1",
                request,
                proposed_tool,
                proposed_arguments,
                verification,
                primary_specialist,
                target_model,
                risk_level,
                review_state,
                next_command,
                *scorecard_entries,
                "auth_model" if authorizes_model_call else "no_model",
                "auth_tool" if authorizes_tool_execution else "no_tool",
                "auth_approval" if authorizes_approval else "no_approval",
                "auth_personal" if authorizes_personal_data_read else "no_personal",
                "auth_external" if authorizes_external_side_effect else "no_external",
                "auth_executable_action" if authorizes_executable_action else "no_executable_action",
                "proof_only_review",
                "requires_fresh_action_proposal_review",
                "authorizes_nothing",
            ]
        )
    )


def _specialist_fresh_review_boundary_token_sha256(
    *,
    request: str,
    cycle_state: str,
    fresh_review_preflight_queue: list[str],
    fresh_review_contract_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    specialist_review_token_sha256: str,
    runtime_review_boundary_token_sha256: str,
    route_to_runtime_contract_token_sha256: str,
    next_command: str,
) -> str:
    contract_entries = [
        "|".join(
            [
                str(row.get("item") or ""),
                str(row.get("source") or ""),
                "fresh" if row.get("fresh_required") else "not_fresh",
                "reusable" if row.get("prior_artifact_reusable") else "not_reusable",
                "auth_action" if row.get("authorizes_action_now") else "no_action",
                "auth_model" if row.get("authorizes_model_call") else "no_model",
                "auth_tool" if row.get("authorizes_tool_execution") else "no_tool",
                "auth_approval" if row.get("authorizes_approval") else "no_approval",
                "auth_personal" if row.get("authorizes_personal_data_read") else "no_personal",
                "auth_external" if row.get("authorizes_external_side_effect") else "no_external",
                "auth_fresh_review" if row.get("authorizes_fresh_review") else "no_fresh_review",
            ]
        )
        for row in fresh_review_contract_rows
    ]
    stage_entries = [
        "|".join(
            [
                str(row.get("stage") or ""),
                str(row.get("state") or ""),
                str(row.get("proof") or ""),
                "ready" if row.get("ready") else "held",
                "auth_action" if row.get("authorizes_action_now") else "no_action",
                "auth_model" if row.get("authorizes_model_call") else "no_model",
                "auth_tool" if row.get("authorizes_tool_execution") else "no_tool",
                "auth_approval" if row.get("authorizes_approval") else "no_approval",
                "auth_personal" if row.get("authorizes_personal_data_read") else "no_personal",
                "auth_external" if row.get("authorizes_external_side_effect") else "no_external",
                "auth_fresh_review" if row.get("authorizes_fresh_review") else "no_fresh_review",
                "reusable_next_cycle" if row.get("reusable_for_next_cycle") else "not_reusable_next_cycle",
            ]
        )
        for row in (stage_rows or [])
    ]
    return _text_sha256(
        "\n".join(
            [
                "specialist_fresh_review_boundary_v1",
                request,
                cycle_state,
                *fresh_review_preflight_queue,
                *contract_entries,
                *stage_entries,
                specialist_review_token_sha256,
                runtime_review_boundary_token_sha256,
                route_to_runtime_contract_token_sha256,
                next_command,
                "proof_only_boundary",
                "requires_fresh_router_contract",
                "requires_fresh_specialist_review_token",
                "authorizes_nothing",
            ]
        )
    )


def _specialist_cycle_ledger_token_sha256(
    *,
    request: str,
    raw_request: str,
    cycle_state: str,
    stage_rows: list[dict[str, Any]],
    fresh_review_preflight_queue: list[str],
    fresh_review_contract_rows: list[dict[str, Any]],
    required_commands: list[str],
    specialist_review_token_sha256: str,
    runtime_review_boundary_token_sha256: str,
    route_to_runtime_contract_token_sha256: str,
    specialist_fresh_review_boundary_token_sha256: str,
    specialist_post_run_closure_token_sha256: str,
    runtime_trace_sha256: str,
    verification_receipt_sha256: str,
    execution_audit_sha256: str,
    execution_recovery_sha256: str,
    after_action_learning_sha256: str,
    completion_claim_sha256: str,
    action_contract_scorecard_rows: list[dict[str, Any]],
    dry_run_scorecard_rows: list[dict[str, Any]],
    completion_scorecard_rows: list[dict[str, Any]],
    post_run_proof_queue: list[str],
    next_command: str,
) -> str:
    def serialize_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[str]:
        return ["|".join(str(row.get(key) or "") for key in keys) for row in rows]

    stage_entries = serialize_rows(
        stage_rows,
        (
            "stage",
            "state",
            "ready",
            "proof",
            "authorizes_action_now",
            "authorizes_model_call",
            "authorizes_tool_execution",
            "authorizes_approval",
            "authorizes_personal_data_read",
            "authorizes_external_side_effect",
            "authorizes_fresh_review",
            "reusable_for_next_cycle",
        ),
    )
    fresh_review_entries = serialize_rows(
        fresh_review_contract_rows,
        (
            "item",
            "source",
            "fresh_required",
            "prior_artifact_reusable",
            "authorizes_action_now",
            "authorizes_model_call",
            "authorizes_tool_execution",
            "authorizes_approval",
            "authorizes_personal_data_read",
            "authorizes_external_side_effect",
            "authorizes_fresh_review",
        ),
    )
    scorecard_entries = [
        "action_contract_scorecard",
        *serialize_rows(action_contract_scorecard_rows, ("item", "points", "max_points", "passed", "required_before_execution")),
        "dry_run_scorecard",
        *serialize_rows(dry_run_scorecard_rows, ("item", "points", "max_points", "passed", "required_before_execution")),
        "completion_scorecard",
        *serialize_rows(completion_scorecard_rows, ("item", "score", "max_points", "required_rows_ready", "contract_state", "dry_run_state")),
    ]
    return _text_sha256(
        "\n".join(
            [
                "specialist_cycle_ledger_v1",
                request,
                raw_request,
                cycle_state,
                "stage_rows",
                *stage_entries,
                "fresh_review_preflight_queue",
                *fresh_review_preflight_queue,
                "fresh_review_contract_rows",
                *fresh_review_entries,
                "required_commands",
                *required_commands,
                "post_run_proof_queue",
                *post_run_proof_queue,
                "specialist_review_token",
                specialist_review_token_sha256,
                "runtime_review_boundary_token",
                runtime_review_boundary_token_sha256,
                "route_to_runtime_contract_token",
                route_to_runtime_contract_token_sha256,
                "fresh_review_boundary_token",
                specialist_fresh_review_boundary_token_sha256,
                "post_run_closure_token",
                specialist_post_run_closure_token_sha256,
                "post_run_artifact_hashes",
                runtime_trace_sha256,
                verification_receipt_sha256,
                execution_audit_sha256,
                execution_recovery_sha256,
                after_action_learning_sha256,
                completion_claim_sha256,
                "scorecards",
                *scorecard_entries,
                next_command,
                "proof_only_cycle_ledger",
                "requires_fresh_specialist_review",
                "authorizes_nothing",
            ]
        )
    )


_SPECIALIST_CYCLE_STAGE_NAMES = {
    "route_quality",
    "execution_readiness",
    "handoff_receipt",
    "handoff_quality_gate",
    "proposal_gate",
    "action_proposal_contract",
    "tool_dry_run",
    "proposal_completion_gate",
    "execution_handoff",
    "post_run_closure",
}

_SPECIALIST_FRESH_REVIEW_CONTRACT_ITEMS = {
    "router_contract",
    "route_quality",
    "execution_readiness",
    "handoff_receipt",
    "proposal_gate",
    "action_proposal_contract",
    "tool_dry_run",
    "execution_handoff",
    "post_run_closure",
    "post_run_closure_token",
    "specialist_review_token",
    "route_to_runtime_contract_token",
}

_SPECIALIST_ACTION_PROPOSAL_SCORECARD_ITEMS = (
    ("tool_registry", 20),
    ("exact_arguments", 20),
    ("verification_expectation", 20),
    ("permission_boundary", 20),
    ("proof_lane", 10),
    ("execution_lock", 10),
)

_SPECIALIST_COMBINED_ACTION_PROPOSAL_SCORECARD_ITEMS = (
    "action_contract_scorecard",
    "tool_dry_run_scorecard",
)


def _specialist_action_proposal_scorecard_shape_ready(
    rows: list[dict[str, Any]],
    *,
    require_all_passed: bool = False,
) -> bool:
    if len(rows) != len(_SPECIALIST_ACTION_PROPOSAL_SCORECARD_ITEMS):
        return False
    for row, (expected_item, expected_max_points) in zip(rows, _SPECIALIST_ACTION_PROPOSAL_SCORECARD_ITEMS):
        points = row.get("points")
        if (
            row.get("item") != expected_item
            or row.get("max_points") != expected_max_points
            or not isinstance(points, int)
            or points < 0
            or points > expected_max_points
            or row.get("required_before_execution") is not True
        ):
            return False
        expected_passed = points == expected_max_points
        if expected_item == "permission_boundary":
            expected_passed = points >= 10
        if row.get("passed") is not expected_passed:
            return False
        for key in (
            "authorizes_model_call",
            "authorizes_tool_execution",
            "authorizes_approval",
            "authorizes_personal_data_read",
            "authorizes_external_side_effect",
            "authorizes_executable_action",
            "reusable_for_next_review",
        ):
            if row.get(key) is not False:
                return False
    if sum(int(row.get("max_points") or 0) for row in rows) != 100:
        return False
    if require_all_passed and not all(row.get("passed") is True for row in rows):
        return False
    return True


def _specialist_combined_action_proposal_scorecard_shape_ready(
    rows: list[dict[str, Any]],
    *,
    require_all_passed: bool = False,
) -> bool:
    if len(rows) != len(_SPECIALIST_COMBINED_ACTION_PROPOSAL_SCORECARD_ITEMS):
        return False
    if tuple(row.get("item") for row in rows) != _SPECIALIST_COMBINED_ACTION_PROPOSAL_SCORECARD_ITEMS:
        return False
    for row in rows:
        score = row.get("score")
        row_count = row.get("row_count")
        required_rows_ready = row.get("required_rows_ready")
        if (
            not isinstance(score, int)
            or score < 0
            or score > 100
            or row.get("max_points") != 100
            or row_count != len(_SPECIALIST_ACTION_PROPOSAL_SCORECARD_ITEMS)
            or not isinstance(required_rows_ready, bool)
            or row.get("required_before_execution") is not True
        ):
            return False
        if row.get("grade") not in {"ready_for_operator_review", "review_required", "held"}:
            return False
        for key in (
            "authorizes_model_call",
            "authorizes_tool_execution",
            "authorizes_approval",
            "authorizes_personal_data_read",
            "authorizes_external_side_effect",
            "authorizes_executable_action",
            "reusable_for_next_review",
        ):
            if row.get(key) is not False:
                return False
    if require_all_passed and not all(row.get("required_rows_ready") is True for row in rows):
        return False
    return True


def _specialist_cycle_ledger_ready(
    *,
    stage_rows: list[dict[str, Any]],
    fresh_review_contract_rows: list[dict[str, Any]],
    action_contract_scorecard_rows: list[dict[str, Any]],
    dry_run_scorecard_rows: list[dict[str, Any]],
    completion_scorecard_rows: list[dict[str, Any]],
    post_run_artifact_hashes_present: bool,
    specialist_review_token_sha256: str,
    runtime_review_boundary_token_sha256: str,
    route_to_runtime_contract_token_sha256: str,
    specialist_post_run_closure_token_sha256: str,
    specialist_fresh_review_boundary_token_sha256: str,
    specialist_cycle_ledger_token_sha256: str,
) -> bool:
    if {str(row.get("stage") or "") for row in stage_rows} != _SPECIALIST_CYCLE_STAGE_NAMES:
        return False
    if {str(row.get("item") or "") for row in fresh_review_contract_rows} != _SPECIALIST_FRESH_REVIEW_CONTRACT_ITEMS:
        return False
    if not all(row.get("ready") is True for row in stage_rows):
        return False
    if not post_run_artifact_hashes_present:
        return False
    if not _specialist_action_proposal_scorecard_shape_ready(action_contract_scorecard_rows):
        return False
    if not _specialist_action_proposal_scorecard_shape_ready(dry_run_scorecard_rows):
        return False
    if not _specialist_combined_action_proposal_scorecard_shape_ready(completion_scorecard_rows):
        return False
    for token in (
        specialist_review_token_sha256,
        runtime_review_boundary_token_sha256,
        route_to_runtime_contract_token_sha256,
        specialist_post_run_closure_token_sha256,
        specialist_fresh_review_boundary_token_sha256,
        specialist_cycle_ledger_token_sha256,
    ):
        if not _looks_like_sha256(token):
            return False
    stage_non_authority_fields = (
        "authorizes_action_now",
        "authorizes_model_call",
        "authorizes_tool_execution",
        "authorizes_approval",
        "authorizes_personal_data_read",
        "authorizes_external_side_effect",
        "authorizes_fresh_review",
        "reusable_for_next_cycle",
    )
    fresh_review_non_authority_fields = (
        "authorizes_action_now",
        "authorizes_model_call",
        "authorizes_tool_execution",
        "authorizes_approval",
        "authorizes_personal_data_read",
        "authorizes_external_side_effect",
        "authorizes_fresh_review",
    )
    for row in stage_rows:
        if not str(row.get("state") or "") or not str(row.get("proof") or ""):
            return False
        if any(row.get(field) is not False for field in stage_non_authority_fields):
            return False
    for row in fresh_review_contract_rows:
        if row.get("fresh_required") is not True or row.get("prior_artifact_reusable") is not False:
            return False
        if any(row.get(field) is not False for field in fresh_review_non_authority_fields):
            return False
    return True


def _specialist_cycle_ledger_token_ready_from_metadata(metadata: dict[str, Any]) -> bool:
    token = str(metadata.get("specialist_cycle_ledger_token_sha256") or "")
    if not _looks_like_sha256(token):
        return False
    expected = _specialist_cycle_ledger_token_sha256(
        request=str(metadata.get("request") or ""),
        raw_request=str(metadata.get("raw_request") or ""),
        cycle_state=str(metadata.get("cycle_state") or ""),
        stage_rows=list(metadata.get("stage_rows") or []),
        fresh_review_preflight_queue=list(metadata.get("fresh_review_preflight_queue") or []),
        fresh_review_contract_rows=list(metadata.get("fresh_review_contract_rows") or []),
        required_commands=list(metadata.get("required_commands") or []),
        specialist_review_token_sha256=str(metadata.get("specialist_review_token_sha256") or ""),
        runtime_review_boundary_token_sha256=str(metadata.get("runtime_review_boundary_token_sha256") or ""),
        route_to_runtime_contract_token_sha256=str(metadata.get("route_to_runtime_contract_token_sha256") or ""),
        specialist_fresh_review_boundary_token_sha256=str(metadata.get("specialist_fresh_review_boundary_token_sha256") or ""),
        specialist_post_run_closure_token_sha256=str(metadata.get("specialist_post_run_closure_token_sha256") or ""),
        runtime_trace_sha256=str(metadata.get("runtime_trace_sha256") or ""),
        verification_receipt_sha256=str(metadata.get("verification_receipt_sha256") or ""),
        execution_audit_sha256=str(metadata.get("execution_audit_sha256") or ""),
        execution_recovery_sha256=str(metadata.get("execution_recovery_sha256") or ""),
        after_action_learning_sha256=str(metadata.get("after_action_learning_sha256") or ""),
        completion_claim_sha256=str(metadata.get("completion_claim_sha256") or ""),
        action_contract_scorecard_rows=list(metadata.get("action_contract_scorecard_rows") or []),
        dry_run_scorecard_rows=list(metadata.get("dry_run_scorecard_rows") or []),
        completion_scorecard_rows=list(metadata.get("completion_scorecard_rows") or []),
        post_run_proof_queue=list((metadata.get("post_run_closure_metadata") or {}).get("proof_queue") or []),
        next_command=str(metadata.get("next_command") or ""),
    )
    if token != expected:
        return False
    for key in (
        "specialist_cycle_ledger_token_authorizes_model_call",
        "specialist_cycle_ledger_token_authorizes_tool_execution",
        "specialist_cycle_ledger_token_authorizes_approval",
        "specialist_cycle_ledger_token_authorizes_personal_data_read",
        "specialist_cycle_ledger_token_authorizes_external_side_effect",
        "specialist_cycle_ledger_token_authorizes_completion_claim",
        "specialist_cycle_ledger_token_reusable_for_next_specialist_review",
    ):
        if metadata.get(key) is not False:
            return False
    return metadata.get("next_specialist_review_requires_new_cycle_ledger_token") is True


def _specialist_post_run_closure_token_sha256(
    *,
    request: str,
    raw_request: str,
    closure_state: str,
    missing: list[str],
    required_commands: list[str],
    specialist_review_token_sha256: str,
    runtime_review_boundary_token_sha256: str,
    route_to_runtime_contract_token_sha256: str,
    runtime_trace_sha256: str,
    verification_receipt_sha256: str,
    execution_audit_sha256: str,
    execution_recovery_sha256: str,
    after_action_learning_sha256: str,
    completion_claim_sha256: str,
    post_run_artifact_hashes_present: bool,
    blockers_cleared: bool,
    next_command: str,
) -> str:
    return _text_sha256(
        "\n".join(
            [
                "specialist_post_run_closure_v1",
                request,
                raw_request,
                closure_state,
                "missing",
                *missing,
                "required_commands",
                *required_commands,
                "specialist_review_token",
                specialist_review_token_sha256,
                "runtime_review_boundary_token",
                runtime_review_boundary_token_sha256,
                "route_to_runtime_contract_token",
                route_to_runtime_contract_token_sha256,
                "post_run_artifact_hashes",
                runtime_trace_sha256,
                verification_receipt_sha256,
                execution_audit_sha256,
                execution_recovery_sha256,
                after_action_learning_sha256,
                completion_claim_sha256,
                f"post_run_artifact_hashes_present={post_run_artifact_hashes_present}",
                f"blockers_cleared={blockers_cleared}",
                next_command,
                "authorizes_execution=False",
                "authorizes_completion_claim=False",
                "approval_granted=False",
                "proof_only_closure",
                "requires_cycle_ledger_before_fresh_review",
                "authorizes_nothing",
            ]
        )
    )


def _specialist_runtime_review_contract_rows(
    *,
    handoff_state: str,
    specialist_review_token_sha256: str,
    proposed_tool: str,
    risk_level: str,
) -> list[dict[str, Any]]:
    status = "ready_for_runtime_review" if handoff_state == "SPECIALIST_EXECUTION_HANDOFF_READY_FOR_RUNTIME_REVIEW" else "held"
    base = {
        "status": status,
        "specialist_review_token_sha256": specialist_review_token_sha256,
        "proposed_tool": proposed_tool,
        "risk_level": risk_level,
        "authorizes_model_call": False,
        "authorizes_tool_execution": False,
        "authorizes_approval": False,
        "authorizes_personal_data_read": False,
        "authorizes_external_side_effect": False,
        "authorizes_completion_claim": False,
        "bypasses_post_run_proof": False,
        "reusable_for_next_specialist_review": False,
    }
    return [
        {
            **base,
            "item": "proposal_completion_gate",
            "required": True,
            "source": "specialist proposal completion gate",
        },
        {
            **base,
            "item": "registered_tool_registry_match",
            "required": True,
            "source": "ToolRegistry",
        },
        {
            **base,
            "item": "exact_argument_contract",
            "required": True,
            "source": "argument contract",
        },
        {
            **base,
            "item": "permission_policy_risk_boundary",
            "required": True,
            "source": "PermissionPolicy",
        },
        {
            **base,
            "item": "verification_packet_required",
            "required": True,
            "source": "verification packet",
        },
        {
            **base,
            "item": "post_run_proof_required",
            "required": True,
            "source": "runtime trace, receipt, audit, recovery, learning, completion claim",
        },
    ]


def _specialist_runtime_review_contract_ready(
    rows: list[dict[str, Any]],
    *,
    handoff_state: str,
    specialist_review_token_sha256: str,
    proposed_tool: str,
    risk_level: str,
) -> bool:
    expected_items = [
        ("proposal_completion_gate", "specialist proposal completion gate"),
        ("registered_tool_registry_match", "ToolRegistry"),
        ("exact_argument_contract", "argument contract"),
        ("permission_policy_risk_boundary", "PermissionPolicy"),
        ("verification_packet_required", "verification packet"),
        ("post_run_proof_required", "runtime trace, receipt, audit, recovery, learning, completion claim"),
    ]
    if len(rows) != len(expected_items):
        return False
    expected_status = "ready_for_runtime_review" if handoff_state == "SPECIALIST_EXECUTION_HANDOFF_READY_FOR_RUNTIME_REVIEW" else "held"
    non_authority_fields = (
        "authorizes_model_call",
        "authorizes_tool_execution",
        "authorizes_approval",
        "authorizes_personal_data_read",
        "authorizes_external_side_effect",
        "authorizes_completion_claim",
        "bypasses_post_run_proof",
        "reusable_for_next_specialist_review",
    )
    for row, (expected_item, expected_source) in zip(rows, expected_items, strict=True):
        if row.get("item") != expected_item:
            return False
        if row.get("source") != expected_source:
            return False
        if row.get("required") is not True:
            return False
        if row.get("status") != expected_status:
            return False
        if row.get("specialist_review_token_sha256") != specialist_review_token_sha256:
            return False
        if row.get("proposed_tool") != proposed_tool:
            return False
        if row.get("risk_level") != risk_level:
            return False
        if any(row.get(field) is not False for field in non_authority_fields):
            return False
    return True


def _specialist_runtime_review_boundary_token_sha256(
    *,
    request: str,
    handoff_state: str,
    next_command: str,
    specialist_review_token_sha256: str,
    runtime_review_contract_rows: list[dict[str, Any]],
) -> str:
    contract_entries = [
        "|".join(
            [
                str(row.get("item") or ""),
                str(row.get("status") or ""),
                str(row.get("source") or ""),
                "auth_model" if row.get("authorizes_model_call") else "no_model",
                "auth_tool" if row.get("authorizes_tool_execution") else "no_tool",
                "auth_approval" if row.get("authorizes_approval") else "no_approval",
                "auth_personal" if row.get("authorizes_personal_data_read") else "no_personal",
                "auth_side_effect" if row.get("authorizes_external_side_effect") else "no_side_effect",
                "auth_completion" if row.get("authorizes_completion_claim") else "no_completion",
                "bypasses_post_run" if row.get("bypasses_post_run_proof") else "requires_post_run",
                "reusable_next_review" if row.get("reusable_for_next_specialist_review") else "not_reusable",
            ]
        )
        for row in runtime_review_contract_rows
    ]
    return _text_sha256(
        "\n".join(
            [
                "specialist_runtime_review_boundary_v1",
                request,
                handoff_state,
                next_command,
                specialist_review_token_sha256,
                *contract_entries,
                "normal_runtime_review_only",
                "proof_only_boundary",
                "authorizes_nothing",
                "requires_post_run_proof",
                "requires_fresh_specialist_review",
            ]
        )
    )


def _specialist_route_to_runtime_contract_token_sha256(
    *,
    request: str,
    handoff_state: str,
    proposed_tool: str,
    proposed_arguments: str,
    verification: str,
    primary_specialist: str,
    target_model: str,
    risk_level: str,
    completion_state: str,
    specialist_review_token_sha256: str,
    action_proposal_review_token_sha256: str,
    runtime_review_boundary_token_sha256: str,
    runtime_review_contract_rows: list[dict[str, Any]],
    pre_run_proof_queue: list[str],
    post_run_proof_queue: list[str],
    next_command: str,
    authorizes_model_call: bool = False,
    authorizes_tool_execution: bool = False,
    authorizes_approval: bool = False,
    authorizes_personal_data_read: bool = False,
    authorizes_external_side_effect: bool = False,
    authorizes_completion_claim: bool = False,
    reusable_for_next_specialist_review: bool = False,
) -> str:
    runtime_entries = [
        "|".join(
            [
                str(row.get("item") or ""),
                str(row.get("status") or ""),
                str(row.get("source") or ""),
                "required" if row.get("required") else "optional",
                "auth_model" if row.get("authorizes_model_call") else "no_model",
                "auth_tool" if row.get("authorizes_tool_execution") else "no_tool",
                "auth_approval" if row.get("authorizes_approval") else "no_approval",
                "auth_personal" if row.get("authorizes_personal_data_read") else "no_personal",
                "auth_side_effect" if row.get("authorizes_external_side_effect") else "no_side_effect",
                "auth_completion" if row.get("authorizes_completion_claim") else "no_completion",
                "bypass_post_run" if row.get("bypasses_post_run_proof") else "post_run_required",
                "reusable" if row.get("reusable_for_next_specialist_review") else "not_reusable",
            ]
        )
        for row in runtime_review_contract_rows
    ]
    return _text_sha256(
        "\n".join(
            [
                "specialist_route_to_runtime_contract_v1",
                request,
                handoff_state,
                proposed_tool,
                proposed_arguments,
                verification,
                primary_specialist,
                target_model,
                risk_level,
                completion_state,
                specialist_review_token_sha256,
                action_proposal_review_token_sha256,
                runtime_review_boundary_token_sha256,
                *runtime_entries,
                "pre_run_proof_queue",
                *pre_run_proof_queue,
                "post_run_proof_queue",
                *post_run_proof_queue,
                next_command,
                "auth_model" if authorizes_model_call else "no_model",
                "auth_tool" if authorizes_tool_execution else "no_tool",
                "auth_approval" if authorizes_approval else "no_approval",
                "auth_personal" if authorizes_personal_data_read else "no_personal",
                "auth_side_effect" if authorizes_external_side_effect else "no_side_effect",
                "auth_completion" if authorizes_completion_claim else "no_completion",
                "reusable" if reusable_for_next_specialist_review else "not_reusable",
                "normal_runtime_path_only",
                "fresh_contract_required_for_next_review",
            ]
        )
    )


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "calls_model": False,
        "calls_external_service": False,
        "executes_tools": False,
        "queues_approval": False,
        "approves_request": False,
        "dismisses_request": False,
        "writes_memory": False,
        "writes_files": False,
        "writes_notes": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "controls_computer": False,
        "requires_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "speaks": False,
    }
    metadata.update(extra)
    return metadata


def _missing_specialist_request_result(
    tool_name: str,
    example: str,
) -> ToolResult:
    """Return a canonical, known-no-action specialist input refusal."""

    output = (
        f"{MISSING_SPECIALIST_REQUEST_RECOVERY_ACTION} "
        f"For example: `{example}`."
    )
    metadata = _safe_metadata(
        reason="missing_request",
        outcome_known=True,
        outcome_unknown=False,
        execution_outcome_unknown=False,
        side_effect_possible=False,
        retry_safe=True,
        automatic_retry_allowed=False,
        authorizes_retry=False,
    )
    return ToolResult(
        tool_name,
        False,
        output,
        declare_failure_guidance(
            metadata,
            output=output,
            action=MISSING_SPECIALIST_REQUEST_RECOVERY_ACTION,
        ),
    )


def _config_unavailable_result(tool_name: str, retry_command: str) -> ToolResult:
    return ToolResult(
        tool_name,
        False,
        "Jarvis configuration is unavailable. Run `setup check`, fix the reported configuration "
        f"issue, then retry `{retry_command}`.",
        _safe_metadata(
            reason="config_unavailable",
            next_command="setup check",
            recovery_commands=["setup check", retry_command],
            retry_requires_setup_repair=True,
            authorizes_retry=False,
        ),
    )


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _specialist_route(request: str) -> tuple[str, list[str], list[tuple[int, str]]]:
    low = request.lower()
    scored: list[tuple[int, str]] = []
    for name, profile in SPECIALIST_PROFILES.items():
        score = sum(1 for signal in profile["signals"] if signal in low)
        scored.append((score, name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    primary = scored[0][1] if scored and scored[0][0] > 0 else "planner"
    supporting = [name for score, name in scored if name != primary and score > 0][:3]
    if not supporting and primary != "chat":
        supporting.append("chat")
    return primary, supporting, scored


def _approval_triggers_from_request(request: str) -> list[str]:
    approval_triggers = []
    low = request.lower()
    if any(token in low for token in ("run command", "shell", "script", "execute", "terminal", "run focused tests", "run tests", "tests")):
        approval_triggers.append("shell/code execution")
    if any(token in low for token in ("screen", "screenshot", "click", "type", "window", "mouse", "keyboard")):
        approval_triggers.append("computer control or visual observation")
    if any(token in low for token in ("email", "send", "post", "publish", "calendar", "remind")):
        approval_triggers.append("external side effect")
    if any(token in low for token in ("private", "personal", "gmail", "calendar", "contacts", "clipboard")):
        approval_triggers.append("personal data")
    if any(token in low for token in ("delete", "remove", "overwrite", "destroy")):
        approval_triggers.append("destructive change")
    return approval_triggers


def _proposal_field_from_text(text: str, names: tuple[str, ...]) -> str:
    joined = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
    match = re.search(rf"(?:^|[;\n])\s*(?:{joined})(?:\s*:|\s+)(?P<value>[^;\n]+)", text, re.IGNORECASE)
    return _clean_text(match.group("value"), limit=MAX_PROPOSAL_ARGUMENT_CHARS) if match else ""


def _proposal_request_without_fields(text: str) -> str:
    parts = []
    for part in re.split(r"\s*;\s*", text):
        if re.match(
            r"^(?:tool|tool name|action tool|args|arguments|input|payload|verification|expected|expectation|success|"
            r"runtime_trace|runtime trace|runtime_trace_receipt|verification_receipt|verification receipt|audit|execution_audit|"
            r"recovery|execution_recovery|learning|after_action_learning|completion_claim|claim|"
            r"runtime_trace_sha256|runtime trace sha256|verification_receipt_sha256|verification receipt sha256|"
            r"audit_sha256|execution_audit_sha256|recovery_sha256|execution_recovery_sha256|"
            r"learning_sha256|after_action_learning_sha256|completion_claim_sha256|claim_sha256)\s*:?",
            part.strip(),
            re.IGNORECASE,
        ):
            continue
        parts.append(part.strip())
    return _clean_text("; ".join(part for part in parts if part) or text, limit=MAX_HANDOFF_REQUEST_CHARS)


def _arguments_are_exact(arguments: str) -> bool:
    if not arguments:
        return False
    lowered = arguments.lower()
    if any(marker in lowered for marker in ("<", ">", "tbd", "todo", "unknown", "whatever", "some ")):
        return False
    return ":" in arguments or "=" in arguments or (arguments.startswith("{") and arguments.endswith("}"))


def _evidence_present(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return bool(text and text not in {"none", "missing", "false", "no", "n/a", "not reviewed", "not supplied"})


def _looks_like_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(value or "").strip()))


def _specialist_target_model(config: JarvisConfig | None, primary: str) -> str:
    if config is None:
        return ""
    return config.planner_model if primary == "planner" else config.chat_model


def _specialist_proof_lanes(request: str, primary: str, config: JarvisConfig | None) -> dict[str, str]:
    target_model = _specialist_target_model(config, primary)
    return {
        "target_model": target_model,
        "fallback_lane": "bounded handoff receipt and route-quality packet; no model call or tool execution",
        "proof_target": f"specialist route quality: {request}",
        "handoff_target": f"specialist handoff receipt: {request}",
        "readiness_target": f"specialist execution readiness: {request}",
        "stop_condition": "stop if route confidence, verifier coverage, model readiness, or approval boundaries are not proven",
    }


def _specialist_handoff_scorecard(
    *,
    primary_score: int,
    score_margin: int,
    verifier_coverage: int,
    approval_triggers: list[str],
    ambiguous: bool,
    model_ready: bool | None = None,
) -> dict[str, Any]:
    route_signal_points = min(primary_score, 3) * 10
    route_margin_points = max(0, min(score_margin, 2)) * 10
    verifier_points = round(verifier_coverage * 0.3)
    safety_points = 10 if approval_triggers else 15
    model_points = 5 if model_ready is True else 0
    raw_total = route_signal_points + route_margin_points + verifier_points + safety_points + model_points
    if ambiguous:
        raw_total = min(raw_total, 49)
    score = max(0, min(100, raw_total))
    if score >= 75 and not ambiguous:
        grade = "strong"
    elif score >= 50:
        grade = "review"
    else:
        grade = "weak"
    return {
        "score": score,
        "grade": grade,
        "route_signal_points": route_signal_points,
        "route_margin_points": route_margin_points,
        "verifier_points": verifier_points,
        "safety_points": safety_points,
        "model_points": model_points,
        "max_points": 100,
        "approval_gate_required": bool(approval_triggers),
        "ambiguous_penalty_applied": ambiguous,
        "model_readiness_included": model_ready is not None,
    }


def _specialist_action_proposal_scorecard(
    *,
    registered_tool: bool,
    exact_arguments_supplied: bool,
    verification_supplied: bool,
    approval_required: bool,
    local_safe_or_read_only: bool,
    proof_lane_bound: bool,
    execution_locked: bool = True,
) -> dict[str, Any]:
    registry_points = 20 if registered_tool else 0
    argument_points = 20 if exact_arguments_supplied else 0
    verification_points = 20 if verification_supplied else 0
    permission_points = 20 if local_safe_or_read_only and not approval_required else (10 if approval_required else 0)
    proof_lane_points = 10 if proof_lane_bound else 0
    lock_points = 10 if execution_locked else 0
    score = registry_points + argument_points + verification_points + permission_points + proof_lane_points + lock_points
    if score >= 85 and local_safe_or_read_only and not approval_required:
        grade = "ready_for_operator_review"
    elif score >= 60:
        grade = "review_required"
    else:
        grade = "held"
    return {
        "score": score,
        "grade": grade,
        "max_points": 100,
        "registry_points": registry_points,
        "argument_points": argument_points,
        "verification_points": verification_points,
        "permission_points": permission_points,
        "proof_lane_points": proof_lane_points,
        "execution_lock_points": lock_points,
        "approval_boundary_visible": approval_required,
        "local_safe_or_read_only": local_safe_or_read_only,
    }


def _specialist_action_proposal_scorecard_rows(scorecard: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "item": "tool_registry",
            "points": int(scorecard.get("registry_points") or 0),
            "max_points": 20,
            "passed": int(scorecard.get("registry_points") or 0) == 20,
            "required_before_execution": True,
        },
        {
            "item": "exact_arguments",
            "points": int(scorecard.get("argument_points") or 0),
            "max_points": 20,
            "passed": int(scorecard.get("argument_points") or 0) == 20,
            "required_before_execution": True,
        },
        {
            "item": "verification_expectation",
            "points": int(scorecard.get("verification_points") or 0),
            "max_points": 20,
            "passed": int(scorecard.get("verification_points") or 0) == 20,
            "required_before_execution": True,
        },
        {
            "item": "permission_boundary",
            "points": int(scorecard.get("permission_points") or 0),
            "max_points": 20,
            "passed": int(scorecard.get("permission_points") or 0) >= 10,
            "required_before_execution": True,
        },
        {
            "item": "proof_lane",
            "points": int(scorecard.get("proof_lane_points") or 0),
            "max_points": 10,
            "passed": int(scorecard.get("proof_lane_points") or 0) == 10,
            "required_before_execution": True,
        },
        {
            "item": "execution_lock",
            "points": int(scorecard.get("execution_lock_points") or 0),
            "max_points": 10,
            "passed": int(scorecard.get("execution_lock_points") or 0) == 10,
            "required_before_execution": True,
        },
    ]
    for row in rows:
        row.update(
            {
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_approval": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "authorizes_executable_action": False,
                "reusable_for_next_review": False,
            }
        )
    return rows


def _ollama_models() -> tuple[bool, list[str], str, str]:
    return probe_ollama_models()


def _model_available(configured: str, models: list[str]) -> bool:
    """Ollama model-list names carry an explicit tag (e.g. "llama3.1:latest"), but
    config values are commonly written without one (e.g. "llama3.1"), which
    Ollama itself treats as an implicit ":latest". A bare `in` check on the
    raw list therefore reports a real, pulled, working model as unavailable.
    Real gap found live 2026-07-09: `model routing status` reported both the
    chat and planner models as "needs attention" all session despite
    `llama3.1` being pulled and in active use, because the configured name
    never carried the ":latest" suffix the Ollama model-list API reports."""
    name = (configured or "").strip()
    if not name:
        return False
    if name in models:
        return True
    if ":" not in name:
        return f"{name}:latest" in models
    return False


def _specialist_model_readiness(config: JarvisConfig | None, target_model: str) -> dict[str, Any]:
    if config is None:
        return {
            "model_provider": "",
            "model_status": "not_checked",
            "model_ready": False,
            "model_configuration_ready": False,
            "model_access_live_verified": False,
            "model_detail": "",
        }
    provider = normalized_model_provider(config.model_provider)
    if provider == "invalid":
        return {
            "model_provider": "invalid",
            "model_status": "needs_attention",
            "model_ready": False,
            "model_configuration_ready": False,
            "model_access_live_verified": False,
            "model_detail": "Configured model provider is invalid; no model probe was attempted.",
        }
    if provider == "openai":
        configured = openai_api_key_configured()
        return {
            "model_provider": "openai",
            "model_status": "ready" if configured else "needs_attention",
            "model_ready": configured,
            "model_configuration_ready": configured,
            "model_access_live_verified": False,
            "model_detail": (
                "OpenAI configuration is present; live model access remains unverified."
                if configured
                else "OPENAI_API_KEY is missing; run `model routing status`."
            ),
        }

    ollama_ok, models, detail, _exception_type = _ollama_models()
    local_only = ollama_local_only_policy(target_model)
    model_present = bool(ollama_ok and _model_available(target_model, models))
    ready = bool(model_present and not local_only.cloud_model_alias)
    if local_only.cloud_model_alias:
        detail = (
            "Configured Ollama model is classified as a cloud alias and is blocked before client construction."
        )
    return {
        "model_provider": "ollama",
        "model_status": "ready" if ready else "needs_attention",
        "model_ready": ready,
        "model_configuration_ready": ready,
        "model_access_live_verified": False,
        "model_execution_state": "unknown",
        "model_present": model_present,
        "model_context_compatible": None,
        "ollama_num_ctx_requested": OLLAMA_NUM_CTX,
        "ollama_num_ctx_request_configured": True,
        "ollama_num_ctx_effective": None,
        "ollama_num_ctx_effective_available": False,
        "ollama_prompt_truncated": None,
        "ollama_prompt_truncation_verified": False,
        "ollama_context_compatibility_verified": False,
        "ollama_larger_context_may_increase_local_resource_use": True,
        "model_detail": detail,
        **local_only.receipt(),
    }


def make_model_status_tool(config: JarvisConfig | None):
    def model_routing_status(_: dict[str, Any]) -> ToolResult:
        if config is None:
            return _config_unavailable_result("model_routing_status", "model routing status")

        provider = normalized_model_provider(config.model_provider)
        remote_personal_context_allowed = bool(config.allow_remote_personal_context)
        uses_external_model = provider == "openai"
        uses_ollama = provider == "ollama"
        ollama_destination = resolve_ollama_destination() if uses_ollama else None
        ollama_local_only = (
            ollama_local_only_policy(config.chat_model) if uses_ollama else None
        )
        ollama_personal_context_allowed = bool(
            ollama_destination is not None
            and ollama_destination.allowed
            and ollama_local_only is not None
            and ollama_local_only.personal_context_allowed
        )
        context_routing_metadata = {
            "would_share_current_message_with_external_model": uses_external_model,
            "would_share_stored_personal_context_with_external_model": bool(
                uses_external_model and remote_personal_context_allowed
            ),
            "stored_personal_context_stays_local": bool(
                (uses_external_model and not remote_personal_context_allowed)
                or (uses_ollama and not ollama_personal_context_allowed)
                or provider == "invalid"
            ),
            "stored_personal_context_eligible_for_ollama": ollama_personal_context_allowed,
            "stored_personal_context_execution_locality_unknown": bool(
                uses_ollama and ollama_personal_context_allowed
            ),
            "remote_personal_context_allowed": remote_personal_context_allowed,
            "remote_personal_context_policy": (
                (
                    "blocked_destination"
                    if ollama_destination is not None and not ollama_destination.allowed
                    else "local_provider_unverified_context_consent"
                    if ollama_personal_context_allowed
                    else "local_provider_stateless"
                )
                if uses_ollama
                else ("enabled" if remote_personal_context_allowed else "disabled")
                if uses_external_model
                else "invalid_provider"
            ),
            "remote_personal_context_sources": [
                "profile",
                "preferences",
                "memory",
                "skills",
                "prior_history",
            ],
            "remote_personal_context_env": "JARVIS_ALLOW_REMOTE_PERSONAL_CONTEXT",
            "model_routing_status_metadata_content_free": True,
            "model_request_content_in_metadata": False,
            "model_routing_status_includes_message_content": False,
            "model_routing_status_includes_personal_context_content": False,
            "personal_context_content_in_metadata": False,
        }

        if provider == "invalid":
            return ToolResult(
                "model_routing_status",
                True,
                "\n".join(
                    [
                        "Jarvis model routing status:",
                        "- provider: invalid (needs attention)",
                        f"- chat model: {config.chat_model} (needs attention)",
                        f"- planner model: {config.planner_model} (needs attention)",
                        "- model probe: not attempted because the provider configuration is invalid",
                        "- model execution: unknown",
                        "",
                        "Safe next steps:",
                        "- Set JARVIS_MODEL_PROVIDER to `ollama` or `openai`, run `setup check`, then retry `model routing status`.",
                        "- Do not let model routing bypass approval gates for shell, files, personal data, or computer control.",
                    ]
                ),
                _safe_metadata(
                    model_provider="invalid",
                    model_provider_valid=False,
                    model_probe_attempted=False,
                    model_access_live_verified=False,
                    model_execution_state="unknown",
                    chat_model=config.chat_model,
                    planner_model=config.planner_model,
                    chat_ready=False,
                    planner_ready=False,
                    planner_model_required=bool(config.use_model_planner),
                    planner_model_available=False,
                    planner_routing_ready=False,
                    model_routing_ready=False,
                    authorizes_retry=False,
                    **context_routing_metadata,
                ),
            )

        if provider == "openai":
            key_configured = openai_api_key_configured()
            planner_model_required = bool(config.use_model_planner)
            status = "configured (live access unverified)" if key_configured else "needs attention"
            lines = [
                "Jarvis model routing status:",
                "- provider: OpenAI Responses API (opt-in)",
                f"- chat model: {config.chat_model} ({status})",
                f"- planner model: {config.planner_model} ({status if planner_model_required else 'standby (model planner disabled)'})",
                f"- chat reasoning effort: {config.chat_reasoning_effort}",
                f"- planner reasoning effort: {config.planner_reasoning_effort}",
                f"- total reasoning/output ceiling per OpenAI request: {config.openai_max_output_tokens:,} tokens",
                "- cost boundary: the ceiling includes hidden reasoning, visible output, and formatting; actual usage may be lower",
                f"- planner timeout: {config.model_timeout_seconds:g}s",
                f"- chat timeout: {config.chat_timeout_seconds:g}s",
                f"- API key: {'configured' if key_configured else 'missing'} (value never displayed)",
                "- safety identifier: stable single-owner pseudonym sent; value hidden and not derived from personal data or the API key",
                "- Responses persistence request flag: disabled (`store: false`)",
                "- account retention controls: not checked by Jarvis",
                "- retention note: `store: false` is not a zero-retention guarantee; default abuse-monitoring logs may retain prompts/responses for up to 30 days unless approved OpenAI retention controls apply",
                "- live model access: not tested by this read-only status command",
                "",
                "Safety boundary:",
                "- Ordinary OpenAI chat sends the current message explicitly typed for that turn.",
                (
                    "- Stored profile, preferences, memory, skills, and prior history may also be sent because "
                    "JARVIS_ALLOW_REMOTE_PERSONAL_CONTEXT is enabled."
                    if remote_personal_context_allowed
                    else "- Stored profile, preferences, memory, skills, and prior history stay local and are omitted from OpenAI requests by default."
                ),
                "- Set JARVIS_ALLOW_REMOTE_PERSONAL_CONTEXT=1 only to opt in to sending that stored personal context.",
                "- Avoid sending sensitive personal content unless remote processing and the account's retention policy are acceptable.",
                "- ToolRegistry and PermissionPolicy remain the execution boundary regardless of model choice.",
                "- Model routing never bypasses approval gates for shell, files, personal data, or computer control.",
                "",
                "Safe next steps:",
            ]
            if not key_configured:
                lines.append(
                    "- Set OPENAI_API_KEY in Jarvis's local environment, then retry `model routing status`."
                )
            else:
                lines.append(
                    "- Configuration is ready for an operator-triggered model call; this status check makes no paid API request."
                )
            return ToolResult(
                "model_routing_status",
                True,
                "\n".join(lines),
                _safe_metadata(
                    model_provider="openai",
                    model_provider_valid=True,
                    model_probe_attempted=False,
                    chat_model=config.chat_model,
                    planner_model=config.planner_model,
                    chat_reasoning_effort=config.chat_reasoning_effort,
                    planner_reasoning_effort=config.planner_reasoning_effort,
                    openai_max_output_tokens=config.openai_max_output_tokens,
                    openai_output_ceiling_includes_reasoning=True,
                    openai_output_ceiling_is_usage_target=False,
                    model_timeout_seconds=config.model_timeout_seconds,
                    chat_timeout_seconds=config.chat_timeout_seconds,
                    model_planner=config.use_model_planner,
                    openai_api_key_configured=key_configured,
                    openai_api_key_value_exposed=False,
                    openai_safety_identifier_sent=True,
                    openai_safety_identifier_scope="single_owner",
                    openai_safety_identifier_max_chars=OPENAI_SAFETY_IDENTIFIER_MAX_CHARS,
                    openai_safety_identifier_value_exposed=False,
                    openai_safety_identifier_uses_personal_data=False,
                    openai_safety_identifier_uses_api_key=False,
                    openai_store=False,
                    openai_request_store_flag=False,
                    openai_account_retention_controls_checked=False,
                    openai_zero_data_retention_verified=False,
                    openai_default_abuse_monitoring_may_retain_content=True,
                    openai_default_abuse_monitoring_max_days=30,
                    model_access_live_verified=False,
                    model_routing_ready=key_configured,
                    planner_model_required=planner_model_required,
                    authorizes_model_call=False,
                    authorizes_retry=False,
                    **context_routing_metadata,
                ),
            )

        ollama_python = importlib.util.find_spec("ollama") is not None
        ollama_destination = resolve_ollama_destination()
        chat_local_only = ollama_local_only_policy(config.chat_model)
        planner_local_only = ollama_local_only_policy(config.planner_model)
        ollama_ok, models, ollama_detail, ollama_exception_type = _ollama_models()
        chat_model_present = bool(
            ollama_ok and _model_available(config.chat_model, models)
        )
        chat_ready = bool(chat_model_present and not chat_local_only.cloud_model_alias)
        planner_model_required = bool(config.use_model_planner)
        planner_model_present = bool(
            ollama_ok and _model_available(config.planner_model, models)
        )
        planner_model_available = bool(
            planner_model_present and not planner_local_only.cloud_model_alias
        )
        planner_ready = planner_model_available
        planner_routing_ready = (not planner_model_required) or planner_model_available
        model_routing_ready = chat_ready and planner_routing_ready
        planner_status = (
            "ready"
            if planner_model_available
            else ("standby (model planner disabled)" if not planner_model_required else "needs attention")
        )

        lines = [
            "Jarvis model routing status:",
            f"- chat model: {config.chat_model} ({'ready' if chat_ready else 'needs attention'})",
            f"- planner model: {config.planner_model} ({planner_status})",
            "- model timeout split: planner and chat use separate bounds",
            f"- planner timeout: {config.model_timeout_seconds:g}s",
            f"- chat timeout: {config.chat_timeout_seconds:g}s",
            f"- chat reply token cap: {config.chat_max_reply_tokens}",
            f"- chat history window: {config.chat_max_history_messages} message(s)",
            f"- model planner: {'enabled' if config.use_model_planner else 'disabled'}",
            f"- Ollama requested context window on each generation call: {OLLAMA_NUM_CTX:,} tokens",
            "- This read-only status makes no Ollama generation call",
            "- Ollama effective context support: unknown; daemon and model compatibility are unverified",
            "- Ollama prompt truncation status: unverified",
            "- Ollama context compatibility: unverified; installed model presence is tracked separately",
            "- Ollama larger context resource cost: may increase local memory use and compute time",
            f"- Ollama destination policy: {'validated loopback' if ollama_destination.allowed else 'blocked'}",
            f"- Ollama destination source: {ollama_destination.source}",
            f"- Ollama destination family: {ollama_destination.address_family or 'unavailable'}",
            f"- Ollama destination port: {ollama_destination.port or 'unavailable'}",
            "- Ollama redirects: disabled",
            "- Ollama HTTP proxy use: disabled",
            f"- Ollama no-cloud mode requested: {'yes' if chat_local_only.requested else 'no'}",
            f"- Ollama no-cloud setting valid: {'yes' if chat_local_only.valid else 'no'}",
            f"- Ollama unverified personal-context consent configured: {'yes' if chat_local_only.unverified_context_consent_configured else 'no'}",
            f"- Ollama unverified personal-context consent valid: {'yes' if chat_local_only.unverified_context_consent_valid else 'no'}",
            f"- Ollama unverified personal-context consent allowed: {'yes' if chat_local_only.unverified_context_consent_allowed else 'no'}",
            "- Ollama no-cloud request: configuration request only; not a daemon attestation",
            "- Ollama daemon cloud-disabled state: unknown",
            "- Ollama model execution locality: unknown",
            (
                "- stored personal context to Ollama: eligible for the loopback daemon under explicit unverified consent"
                if ollama_destination.allowed and chat_local_only.personal_context_allowed
                else "- stored personal context to Ollama: withheld; current-message-only mode"
            ),
            "- Ollama Python package: not required (stdlib loopback HTTP adapter)",
            _ollama_http_api_status(ollama_ok),
        ]

        if ollama_ok:
            lines.append("- available Ollama models: " + (", ".join(models[:8]) if models else "none listed"))
        else:
            lines.append(f"- ollama detail: {_clean_text(ollama_detail, limit=MAX_MODEL_DETAIL_CHARS) or 'not reachable'}")

        lines.extend(
            [
                "",
                "Routing plan:",
                "- ChatBrain handles natural conversation and uses memory, profile, preferences, and skills as context.",
                (
                    "- Ollama transport is pinned to a validated loopback endpoint; stored personal context is eligible only when OLLAMA_NO_CLOUD=1 and JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT=1 are both explicit and valid."
                    if ollama_destination.allowed and chat_local_only.personal_context_allowed
                    else "- Ollama transport is loopback, but stored personal context is withheld without both a valid no-cloud request and explicit unverified personal-context consent."
                    if ollama_destination.allowed
                    else "- Ollama transport is blocked before any request because OLLAMA_HOST is not a validated loopback endpoint."
                ),
                "- Loopback transport and OLLAMA_NO_CLOUD configuration do not attest daemon cloud state or prove where the daemon executed a model.",
                "- ModelBackedPlanner may choose tools for uncaught actionable requests when enabled.",
                "- Model calls are bounded by the configured timeout so Jarvis can fall back instead of hanging.",
                "- RuleBasedPlanner remains the deterministic safety-friendly fallback.",
                "- ToolRegistry and PermissionPolicy remain the execution boundary regardless of model choice.",
                "",
                "Conversation latency proof:",
                "- Mixed-conversation acceptance target: chat p95 <= 8000ms with 10+ turns across chat, research, calendar, and tasks.",
                "- Opt-in tuning knobs: JARVIS_CHAT_MAX_REPLY_TOKENS and JARVIS_CHAT_MAX_HISTORY_MESSAGES.",
                "- After any tuning, rerun the mixed-conversation proof and live_check before claiming the latency item is done.",
                "",
                "Safe next steps:",
            ]
        )
        if not ollama_destination.allowed:
            lines.append(
                "- Set OLLAMA_HOST to a loopback address such as http://127.0.0.1:11434, then retry `model routing status`."
            )
        if not chat_local_only.valid or not chat_local_only.requested:
            lines.append(
                "- Configure the Ollama daemon with OLLAMA_NO_CLOUD=1, restart Ollama, then retry `model routing status`; Jarvis will keep stored personal context out of Ollama until then."
            )
        if (
            not chat_local_only.unverified_context_consent_valid
            or not chat_local_only.unverified_context_consent_allowed
        ):
            lines.append(
                "- Set JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT=1 only to explicitly allow stored context despite unknown daemon and execution locality; otherwise Jarvis keeps it withheld."
            )
        if chat_local_only.cloud_model_alias or planner_local_only.cloud_model_alias:
            lines.append(
                "- Replace configured Ollama cloud model aliases with on-device model aliases; cloud aliases are blocked before client construction."
            )
        elif not ollama_ok:
            lines.append(_ollama_unreachable_guidance(config.chat_model))
        if ollama_ok and not chat_ready:
            lines.append(
                "- Run `ollama pull " + config.chat_model + "` outside Jarvis, then retry `model routing status`."
            )
        if not planner_model_required:
            lines.append("- Planner model is optional right now because model planner is disabled; no pull is needed unless enabling JARVIS_USE_MODEL_PLANNER.")
        if planner_model_required and ollama_ok and not planner_ready:
            lines.append(
                "- Run `ollama pull " + config.planner_model + "` outside Jarvis, then retry `model routing status`."
            )
        if model_routing_ready:
            lines.append("- Model routing looks ready; keep approvals enabled for all risky tools.")
        lines.append("- Do not let model routing bypass approval gates for shell, files, personal data, or computer control.")

        return ToolResult(
            "model_routing_status",
            True,
            "\n".join(lines),
            _safe_metadata(
                chat_model=config.chat_model,
                planner_model=config.planner_model,
                available_model_count=len(models),
                model_timeout_seconds=config.model_timeout_seconds,
                chat_timeout_seconds=config.chat_timeout_seconds,
                chat_max_reply_tokens=config.chat_max_reply_tokens,
                chat_max_history_messages=config.chat_max_history_messages,
                chat_latency_tuning_available=True,
                chat_latency_target_p95_ms=8000,
                chat_latency_tuning_knobs=["JARVIS_CHAT_MAX_REPLY_TOKENS", "JARVIS_CHAT_MAX_HISTORY_MESSAGES"],
                chat_latency_requires_fresh_mixed_conversation_proof=True,
                chat_latency_tuning_authorizes_completion_claim=False,
                model_planner=config.use_model_planner,
                model_provider="ollama",
                model_provider_valid=True,
                model_probe_attempted=True,
                ollama_python=ollama_python,
                ollama_python_required=False,
                **ollama_destination.receipt(),
                **chat_local_only.receipt(),
                ollama_request_blocked=not ollama_destination.allowed,
                ollama_transport_loopback_verified=ollama_destination.allowed,
                ollama_on_device_model_execution_verified=False,
                ollama_execution_state="unknown",
                ollama_cloud_features_disabled_requested=chat_local_only.requested,
                ollama_cloud_features_disabled_verified=False,
                ollama_daemon_cloud_state="unknown",
                ollama_reachable=ollama_ok,
                ollama_exception_type=ollama_exception_type,
                ollama_recovery_commands=[] if ollama_ok else ["setup check", "model routing status"],
                ollama_retry_requires_setup_repair=not ollama_ok,
                authorizes_retry=False,
                ollama_num_ctx_requested=OLLAMA_NUM_CTX,
                ollama_num_ctx_request_configured=True,
                ollama_num_ctx_status_probe_sent_generation_request=False,
                ollama_num_ctx_effective=None,
                ollama_num_ctx_effective_available=False,
                ollama_prompt_truncated=None,
                ollama_prompt_truncation_verified=False,
                ollama_context_compatibility_verified=False,
                ollama_larger_context_may_increase_local_resource_use=True,
                chat_model_present=chat_model_present,
                chat_model_context_compatible=None,
                chat_ready=chat_ready,
                planner_model_present=planner_model_present,
                planner_model_context_compatible=None,
                planner_ready=planner_ready,
                planner_model_required=planner_model_required,
                planner_model_available=planner_model_available,
                planner_routing_ready=planner_routing_ready,
                model_routing_ready=model_routing_ready,
                **context_routing_metadata,
            ),
        )

    return model_routing_status


def make_model_planner_prompt_preview_tool(config: JarvisConfig | None, list_tools):
    def model_planner_prompt_preview(args: dict[str, Any]) -> ToolResult:
        if config is None:
            return _config_unavailable_result(
                "model_planner_prompt_preview",
                "model planner prompt preview: <request>",
            )
        request = _clean_text(args.get("request"), limit=MAX_PREVIEW_REQUEST_CHARS)
        if not request:
            return _missing_specialist_request_result(
                "model_planner_prompt_preview",
                "model planner prompt preview: find files README in .",
            )
        from jarvis_v2.agent.model_planner import PLANNER_PROMPT

        tools = list_tools()
        risk_counts = Counter(tool.risk.name for tool in tools)
        examples = sorted(tools, key=lambda tool: (tool.toolset, tool.risk, tool.name))[:18]
        lines = [
            "Jarvis model planner prompt preview:",
            "",
            "Purpose:",
            "- Show what a future model-backed planner would see before any model call happens.",
            "- This preview does not call Ollama, OpenAI, or any configured model.",
            "- This preview is read-only and does not call the configured model, execute tools, approve requests, write memory, read personal data, control the computer, or queue approvals.",
            "",
            "Planner model:",
            f"- {config.planner_model}",
            f"- model planner enabled in config: {config.use_model_planner}",
            "- model timeout: planner path only",
            f"- planner timeout: {config.model_timeout_seconds:g}s",
            "",
            "User request:",
            f"- {request}",
            "",
            "System prompt preview:",
            PLANNER_PROMPT.strip(),
            "",
            "Available tool summary:",
            f"- total tools: {len(tools)}",
        ]
        for risk, count in sorted(risk_counts.items()):
            lines.append(f"- {risk}: {count}")
        lines.extend(["", "Tool examples:"])
        for tool in examples:
            approval = "approval-gated" if tool.risk.name in {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"} else "default allowed"
            lines.append(f"- {tool.name} [{tool.toolset}, {tool.risk.name}, {approval}]: {tool.description}")
        lines.extend(
            [
                "",
                "Execution boundary:",
                "- A model may suggest tool names and arguments only.",
                "- ToolRegistry decides whether the tool exists.",
                "- PermissionPolicy decides whether approval is required.",
                "- Shell/code, personal data, computer control, destructive actions, reminders, and external side effects remain approval-gated.",
                "- If the request is ambiguous, the planner should choose chat instead of tools.",
            ]
        )
        return ToolResult(
            "model_planner_prompt_preview",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=request,
                planner_model=config.planner_model,
                model_planner_enabled=config.use_model_planner,
                model_timeout_seconds=config.model_timeout_seconds,
                tools=len(tools),
                risk_counts=dict(risk_counts),
            ),
        )

    return model_planner_prompt_preview


def make_specialist_router_contract_tool(config: JarvisConfig | None, list_tools):
    def specialist_router_contract(args: dict[str, Any]) -> ToolResult:
        request = _clean_text(args.get("request"), limit=MAX_SPECIALIST_REQUEST_CHARS)
        if not request:
            return _missing_specialist_request_result(
                "specialist_router_contract",
                "specialist router contract: summarize this work and propose next code test",
            )

        primary, supporting, scored = _specialist_route(request)
        tools = list_tools()
        risk_counts = Counter(tool.risk.name for tool in tools)
        risk_gated = sum(1 for tool in tools if tool.risk.name in {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"})
        primary_profile = SPECIALIST_PROFILES[primary]
        proof_lanes = _specialist_proof_lanes(request, primary, config)

        lines = [
            "Jarvis specialist router contract:",
            "This is read-only. It chooses a proposed specialist brain route without calling models, executing tools, approving requests, reading personal data, controlling the computer, writing files, or queuing approvals.",
            "",
            f"Request: {request}",
            "",
            "Route decision:",
            f"- primary specialist: {primary}",
            f"- supporting specialists: {', '.join(supporting) if supporting else 'none'}",
            f"- purpose: {primary_profile['purpose']}",
            f"- handoff contract: {primary_profile['handoff']}",
            f"- verification contract: {primary_profile['verification']}",
            "",
            "Specialist proof lane:",
            f"- model target: {proof_lanes['target_model'] or '<no config>'}",
            f"- fallback lane: {proof_lanes['fallback_lane']}",
            f"- route proof target: `{proof_lanes['proof_target']}`",
            f"- handoff target: `{proof_lanes['handoff_target']}`",
            f"- readiness target: `{proof_lanes['readiness_target']}`",
            f"- stop condition: {proof_lanes['stop_condition']}",
            "",
            "Specialist candidates:",
        ]
        for score, name in scored:
            profile = SPECIALIST_PROFILES[name]
            marker = "primary" if name == primary else ("supporting" if name in supporting else "available")
            lines.append(f"- {name}: {marker}; signal score {score}; {profile['purpose']}")
        lines.extend(
            [
                "",
                "Execution boundary:",
                "- The specialist may draft a response, plan, prompt packet, or verifier packet only.",
                "- ToolRegistry still decides whether a tool exists.",
                "- PermissionPolicy still decides whether approval is required.",
                "- Shell/code, computer control, personal data, destructive actions, reminders, and outside-world effects remain approval-gated.",
                "- If the route is uncertain or a user-facing answer is enough, prefer chat/summarizer over action.",
                "",
                "Harness evidence:",
                f"- configured chat model: {config.chat_model if config else '<no config>'}",
                f"- configured planner model: {config.planner_model if config else '<no config>'}",
                f"- model planner enabled: {config.use_model_planner if config else '<no config>'}",
                f"- registered tools: {len(tools)}",
                f"- risk-gated tools: {risk_gated}",
            ]
        )
        for risk, count in sorted(risk_counts.items()):
            lines.append(f"- {risk}: {count}")

        return ToolResult(
            "specialist_router_contract",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=request,
                primary_specialist=primary,
                supporting_specialists=supporting,
                candidate_scores={name: score for score, name in scored},
                tools=len(tools),
                risk_gated_tools=risk_gated,
                risk_counts=dict(risk_counts),
                chat_model=config.chat_model if config else "",
                planner_model=config.planner_model if config else "",
                target_model=proof_lanes["target_model"],
                fallback_lane=proof_lanes["fallback_lane"],
                proof_target=proof_lanes["proof_target"],
                handoff_target=proof_lanes["handoff_target"],
                readiness_target=proof_lanes["readiness_target"],
                stop_condition=proof_lanes["stop_condition"],
                model_planner_enabled=config.use_model_planner if config else False,
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
            ),
        )

    return specialist_router_contract


def make_specialist_orchestration_packet_tool(config: JarvisConfig | None, list_tools):
    def specialist_orchestration_packet(args: dict[str, Any]) -> ToolResult:
        request = _clean_text(args.get("request"), limit=MAX_HANDOFF_REQUEST_CHARS)
        if not request:
            return _missing_specialist_request_result(
                "specialist_orchestration_packet",
                "specialist orchestration packet: summarize this work and propose next code test",
            )

        primary, supporting, scored = _specialist_route(request)
        tools = list_tools()
        tool_names = {tool.name for tool in tools}
        primary_score = next((score for score, name in scored if name == primary), 0)
        runner_up_score = next((score for score, name in scored if name != primary), 0)
        score_margin = primary_score - runner_up_score
        ambiguous = primary_score == 0 or score_margin == 0
        approval_triggers = _approval_triggers_from_request(request)
        verifier_targets = SPECIALIST_VERIFIER_TOOLS[primary]
        present_verifiers = [name for name in verifier_targets if name in tool_names]
        missing_verifiers = [name for name in verifier_targets if name not in tool_names]
        verifier_coverage = round((len(present_verifiers) / len(verifier_targets)) * 100) if verifier_targets else 0
        proof_lanes = _specialist_proof_lanes(request, primary, config)

        selected = [primary]
        for candidate in supporting:
            if candidate not in selected:
                selected.append(candidate)
        for fallback in ("planner", "summarizer", "reflection"):
            if len(selected) >= 3:
                break
            if fallback not in selected:
                selected.append(fallback)

        blockers: list[str] = []
        if ambiguous:
            blockers.append("ambiguous_specialist_route")
        if verifier_coverage < 67:
            blockers.append("thin_verifier_coverage")
        if approval_triggers:
            blockers.append("approval_boundary_required")

        if approval_triggers:
            orchestration_state = "ORCHESTRATION_HELD_FOR_APPROVAL_BOUNDARY"
        elif ambiguous or verifier_coverage < 67:
            orchestration_state = "ORCHESTRATION_HELD_FOR_ROUTE_REVIEW"
        else:
            orchestration_state = "ORCHESTRATION_READY_FOR_BOUNDED_DRAFTS"

        lanes = [
            {
                "lane": "primary_draft",
                "specialist": primary,
                "purpose": SPECIALIST_PROFILES[primary]["purpose"],
                "allowed_output": "bounded draft only; no tool execution or completion claim",
            },
            {
                "lane": "verifier_review",
                "specialist": selected[1] if len(selected) > 1 else primary,
                "purpose": "challenge assumptions, verifier coverage, missing inputs, and route fit",
                "allowed_output": "review notes and required proof commands only",
            },
            {
                "lane": "safety_gate",
                "specialist": "planner",
                "purpose": "bind ToolRegistry, PermissionPolicy, exact arguments, and approval triggers",
                "allowed_output": "go/no-go packet and proof queue only",
            },
            {
                "lane": "reconciler",
                "specialist": "summarizer",
                "purpose": "merge draft, verifier notes, safety gates, and residual uncertainty",
                "allowed_output": "human-readable handoff summary with no claim that action ran",
            },
        ]

        required_proof_commands = [
            proof_lanes["proof_target"],
            proof_lanes["readiness_target"],
            proof_lanes["handoff_target"],
            f"specialist handoff quality gate: {request}",
            f"specialist proposal gate: {request}",
            f"specialist model draft: {request}",
            f"specialist action proposal contract: {request}; tool <proposed_tool>; args <exact_args>",
            f"specialist tool dry run: {request}; tool <proposed_tool>; args <exact_args>; verification <expected proof>",
            f"specialist proposal completion gate: {request}; tool <proposed_tool>; args <exact_args>; verification <expected proof>",
            f"specialist execution handoff: {request}; tool <proposed_tool>; args <exact_args>; verification <expected proof>",
            f"specialist post-run closure: {request}; tool <proposed_tool>; args <exact_args>; verification <expected proof>; runtime_trace <receipt>; verification_receipt <receipt>; audit <receipt>; recovery <receipt>; learning <receipt>; completion_claim <gate>",
        ]
        if approval_triggers:
            required_proof_commands.extend(["approval readiness latest", "approval packet latest", "approval chain proof latest"])

        next_command = (
            "approval readiness latest"
            if approval_triggers
            else proof_lanes["proof_target"]
            if orchestration_state == "ORCHESTRATION_HELD_FOR_ROUTE_REVIEW"
            else proof_lanes["handoff_target"]
        )

        lines = [
            "Jarvis specialist orchestration packet:",
            "This is read-only. It turns a user request into a multi-brain harness flow without calling models, executing tools, approving requests, reading personal data, controlling the computer, writing files, or queuing approvals.",
            "",
            f"Request: {request}",
            "",
            "Orchestration state:",
            f"- state: {orchestration_state}",
            f"- primary specialist: {primary}",
            f"- selected specialists: {', '.join(selected)}",
            f"- next command: `{next_command}`",
            f"- blockers: {', '.join(blockers) if blockers else 'none'}",
            "- specialist drafts locked: yes",
            "- executable actions locked: yes",
            "- can bypass ToolRegistry or PermissionPolicy: no",
            "",
            "Route and verifier evidence:",
            f"- primary signal score: {primary_score}",
            f"- runner-up margin: {score_margin}",
            f"- ambiguous route: {'yes' if ambiguous else 'no'}",
            f"- verifier coverage: {verifier_coverage}%",
            f"- available verifier hooks: {', '.join(present_verifiers) if present_verifiers else 'none'}",
            f"- missing verifier hooks: {', '.join(missing_verifiers) if missing_verifiers else 'none'}",
            f"- approval triggers detected: {', '.join(approval_triggers) if approval_triggers else 'none from wording'}",
            "",
            "Specialist lanes:",
        ]
        for lane in lanes:
            lines.append(f"- {lane['lane']}: {lane['specialist']} - {lane['purpose']}; output: {lane['allowed_output']}")
        lines.extend(
            [
                "",
                "Proof queue:",
                *[f"- `{command}`" for command in required_proof_commands],
                "",
                "Safety boundary:",
                "- Multi-brain means more review lanes, not more permission.",
                "- Any proposed tool still needs ToolRegistry lookup, exact arguments, PermissionPolicy review, verification, audit, recovery, learning, and approval proof where risk requires it.",
            ]
        )

        return ToolResult(
            "specialist_orchestration_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=request,
                orchestration_state=orchestration_state,
                orchestration_ready_for_bounded_drafts=orchestration_state == "ORCHESTRATION_READY_FOR_BOUNDED_DRAFTS",
                specialist_drafts_locked=True,
                executable_actions_locked=True,
                can_bypass_tool_registry=False,
                can_bypass_permission_policy=False,
                next_command=next_command,
                blockers=blockers,
                blocker_count=len(blockers),
                primary_specialist=primary,
                supporting_specialists=supporting,
                selected_specialists=selected,
                selected_specialist_count=len(selected),
                candidate_scores={name: score for score, name in scored},
                primary_signal_score=primary_score,
                runner_up_signal_score=runner_up_score,
                score_margin=score_margin,
                ambiguous_route=ambiguous,
                verifier_tools=present_verifiers,
                missing_verifier_tools=missing_verifiers,
                verifier_coverage_percent=verifier_coverage,
                approval_triggers=approval_triggers,
                approval_required_by_wording=bool(approval_triggers),
                lanes=lanes,
                lane_count=len(lanes),
                target_model=proof_lanes["target_model"],
                fallback_lane=proof_lanes["fallback_lane"],
                proof_target=proof_lanes["proof_target"],
                handoff_target=proof_lanes["handoff_target"],
                readiness_target=proof_lanes["readiness_target"],
                stop_condition=proof_lanes["stop_condition"],
                proof_queue=required_proof_commands,
                proof_queue_count=len(required_proof_commands),
                next_proof_command=required_proof_commands[0],
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
            ),
        )

    return specialist_orchestration_packet


def make_specialist_route_quality_tool(config: JarvisConfig | None, list_tools):
    def specialist_route_quality(args: dict[str, Any]) -> ToolResult:
        request = _clean_text(args.get("request"), limit=MAX_SPECIALIST_REQUEST_CHARS)
        if not request:
            return _missing_specialist_request_result(
                "specialist_route_quality",
                "specialist route quality: fix this code bug and run focused tests",
            )

        primary, supporting, scored = _specialist_route(request)
        tools = list_tools()
        tool_names = {tool.name for tool in tools}
        profile = SPECIALIST_PROFILES[primary]
        primary_score = next((score for score, name in scored if name == primary), 0)
        runner_up_score = next((score for score, name in scored if name != primary), 0)
        score_margin = primary_score - runner_up_score
        verifier_targets = SPECIALIST_VERIFIER_TOOLS[primary]
        present_verifiers = [name for name in verifier_targets if name in tool_names]
        missing_verifiers = [name for name in verifier_targets if name not in tool_names]
        verifier_coverage = round((len(present_verifiers) / len(verifier_targets)) * 100) if verifier_targets else 0
        ambiguous = primary_score == 0 or score_margin == 0
        measured_quality = bool(primary_score and present_verifiers)

        if primary_score >= 2 and score_margin >= 1 and verifier_coverage >= 67:
            verdict = "ROUTE_STRONG_WITH_LOCAL_PROOF"
            confidence = "high"
        elif primary_score >= 1 and verifier_coverage >= 50:
            verdict = "ROUTE_USABLE_BUT_REVIEW"
            confidence = "medium"
        else:
            verdict = "ROUTE_UNMEASURED_OR_AMBIGUOUS"
            confidence = "low"

        approval_triggers = _approval_triggers_from_request(request)
        proof_lanes = _specialist_proof_lanes(request, primary, config)
        scorecard = _specialist_handoff_scorecard(
            primary_score=primary_score,
            score_margin=score_margin,
            verifier_coverage=verifier_coverage,
            approval_triggers=approval_triggers,
            ambiguous=ambiguous,
        )

        lines = [
            "Jarvis specialist route quality packet:",
            "This is read-only. It measures route confidence and verifier coverage before any specialist model call, tool execution, approval, personal data access, or computer control.",
            "",
            f"Request: {request}",
            "",
            "Route quality:",
            f"- verdict: {verdict}",
            f"- primary specialist: {primary}",
            f"- confidence: {confidence}",
            f"- measured quality: {'yes' if measured_quality else 'no'}",
            f"- ambiguous route: {'yes' if ambiguous else 'no'}",
            f"- primary signal score: {primary_score}",
            f"- runner-up margin: {score_margin}",
            f"- supporting specialists: {', '.join(supporting) if supporting else 'none'}",
            f"- purpose: {profile['purpose']}",
            "",
            "Measured handoff scorecard:",
            f"- score: {scorecard['score']}/100",
            f"- grade: {scorecard['grade']}",
            f"- route signal points: {scorecard['route_signal_points']}",
            f"- route margin points: {scorecard['route_margin_points']}",
            f"- verifier points: {scorecard['verifier_points']}",
            f"- safety points: {scorecard['safety_points']}",
            f"- ambiguous penalty applied: {'yes' if scorecard['ambiguous_penalty_applied'] else 'no'}",
            "",
            "Specialist proof lane:",
            f"- model target: {proof_lanes['target_model'] or '<no config>'}",
            f"- fallback lane: {proof_lanes['fallback_lane']}",
            f"- route proof target: `{proof_lanes['proof_target']}`",
            f"- handoff target: `{proof_lanes['handoff_target']}`",
            f"- readiness target: `{proof_lanes['readiness_target']}`",
            f"- stop condition: {proof_lanes['stop_condition']}",
            "",
            "Verifier coverage:",
            f"- coverage: {verifier_coverage}%",
            f"- available verifier hooks: {', '.join(present_verifiers) if present_verifiers else 'none'}",
            f"- missing verifier hooks: {', '.join(missing_verifiers) if missing_verifiers else 'none'}",
            "",
            "Candidate scores:",
        ]
        for score, name in scored:
            lines.append(f"- {name}: {score}")
        lines.extend(
            [
                "",
                "Safety posture:",
                f"- approval triggers detected: {', '.join(approval_triggers) if approval_triggers else 'none from wording'}",
                "- A strong route is still only a routing measurement; risky execution remains approval-gated.",
                "- If the route is ambiguous or verifier coverage is thin, use `specialist router contract` and `specialist handoff receipt` before relying on it.",
            ]
        )

        return ToolResult(
            "specialist_route_quality",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=request,
                verdict=verdict,
                primary_specialist=primary,
                supporting_specialists=supporting,
                confidence=confidence,
                measured_quality=measured_quality,
                ambiguous_route=ambiguous,
                primary_signal_score=primary_score,
                runner_up_signal_score=runner_up_score,
                score_margin=score_margin,
                candidate_scores={name: score for score, name in scored},
                verifier_tools=present_verifiers,
                missing_verifier_tools=missing_verifiers,
                verifier_coverage_percent=verifier_coverage,
                handoff_scorecard=scorecard,
                handoff_score=scorecard["score"],
                handoff_score_grade=scorecard["grade"],
                handoff_scorecard_max_points=scorecard["max_points"],
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                approval_triggers=approval_triggers,
                approval_required_by_wording=bool(approval_triggers),
                tools=len(tools),
                chat_model=config.chat_model if config else "",
                planner_model=config.planner_model if config else "",
                target_model=proof_lanes["target_model"],
                fallback_lane=proof_lanes["fallback_lane"],
                proof_target=proof_lanes["proof_target"],
                handoff_target=proof_lanes["handoff_target"],
                readiness_target=proof_lanes["readiness_target"],
                stop_condition=proof_lanes["stop_condition"],
                model_planner_enabled=config.use_model_planner if config else False,
            ),
        )

    return specialist_route_quality


def make_specialist_execution_readiness_tool(config: JarvisConfig | None, list_tools):
    def specialist_execution_readiness(args: dict[str, Any]) -> ToolResult:
        request = _clean_text(args.get("request"), limit=MAX_READINESS_REQUEST_CHARS)
        if not request:
            return _missing_specialist_request_result(
                "specialist_execution_readiness",
                "specialist execution readiness: fix this code bug and run focused tests",
            )

        primary, supporting, scored = _specialist_route(request)
        tools = list_tools()
        tool_names = {tool.name for tool in tools}
        primary_score = next((score for score, name in scored if name == primary), 0)
        runner_up_score = next((score for score, name in scored if name != primary), 0)
        score_margin = primary_score - runner_up_score
        verifier_targets = SPECIALIST_VERIFIER_TOOLS[primary]
        present_verifiers = [name for name in verifier_targets if name in tool_names]
        missing_verifiers = [name for name in verifier_targets if name not in tool_names]
        verifier_coverage = round((len(present_verifiers) / len(verifier_targets)) * 100) if verifier_targets else 0
        approval_triggers = _approval_triggers_from_request(request)
        ambiguous = primary_score == 0 or score_margin == 0
        proof_lanes = _specialist_proof_lanes(request, primary, config)
        model_readiness = _specialist_model_readiness(config, proof_lanes["target_model"])
        model_status = str(model_readiness["model_status"])
        model_ready = model_readiness["model_ready"] is True
        model_detail = str(model_readiness["model_detail"])

        blockers: list[str] = []
        if ambiguous:
            blockers.append("ambiguous_specialist_route")
        if verifier_coverage < 67:
            blockers.append("thin_verifier_coverage")
        if approval_triggers:
            blockers.append("approval_gated_request_wording")
        if config is not None and not model_ready:
            blockers.append("configured_model_not_ready")

        if approval_triggers:
            verdict = "HOLD_FOR_APPROVAL_GATE"
            next_command = f"specialist handoff receipt: {request}"
        elif ambiguous or verifier_coverage < 67:
            verdict = "HOLD_FOR_ROUTE_REVIEW"
            next_command = f"specialist router contract: {request}"
        elif config is not None and not model_ready:
            verdict = "HOLD_FOR_MODEL_SETUP"
            next_command = "model routing status"
        else:
            verdict = "READY_FOR_SPECIALIST_MODEL_DRAFT"
            next_command = f"specialist handoff receipt: {request}"
        scorecard = _specialist_handoff_scorecard(
            primary_score=primary_score,
            score_margin=score_margin,
            verifier_coverage=verifier_coverage,
            approval_triggers=approval_triggers,
            ambiguous=ambiguous,
            model_ready=model_ready if config is not None else None,
        )

        lines = [
            "Jarvis specialist execution readiness:",
            "This is read-only. It gates whether a request is ready to enter a specialist brain before any model call, tool execution, approval, personal data access, computer control, file write, or external side effect.",
            "",
            f"Request: {request}",
            "",
            "Readiness verdict:",
            f"- verdict: {verdict}",
            f"- next command: `{next_command}`",
            f"- blockers: {', '.join(blockers) if blockers else 'none'}",
            "",
            "Route evidence:",
            f"- primary specialist: {primary}",
            f"- supporting specialists: {', '.join(supporting) if supporting else 'none'}",
            f"- primary signal score: {primary_score}",
            f"- runner-up margin: {score_margin}",
            f"- ambiguous route: {'yes' if ambiguous else 'no'}",
            "",
            "Measured handoff scorecard:",
            f"- score: {scorecard['score']}/100",
            f"- grade: {scorecard['grade']}",
            f"- model readiness points: {scorecard['model_points']}",
            f"- approval gate required: {'yes' if scorecard['approval_gate_required'] else 'no'}",
            "",
            "Verifier evidence:",
            f"- verifier coverage: {verifier_coverage}%",
            f"- available verifier hooks: {', '.join(present_verifiers) if present_verifiers else 'none'}",
            f"- missing verifier hooks: {', '.join(missing_verifiers) if missing_verifiers else 'none'}",
            "",
            "Model readiness:",
            f"- status: {model_status}",
            f"- provider: {model_readiness['model_provider'] or '<no config>'}",
            f"- model target: {proof_lanes['target_model'] or '<no config>'}",
            f"- configuration ready: {'yes' if model_readiness['model_configuration_ready'] else 'no'}",
            f"- live access verified: {'yes' if model_readiness['model_access_live_verified'] else 'no'}",
            f"- chat model: {config.chat_model if config else '<no config>'}",
            f"- planner model: {config.planner_model if config else '<no config>'}",
        ]
        if config is not None and not model_ready and model_detail:
            lines.append(f"- detail: {_clean_text(model_detail, limit=MAX_MODEL_DETAIL_CHARS)}")
        lines.extend(
            [
                "",
                "Specialist proof lane:",
                f"- fallback lane: {proof_lanes['fallback_lane']}",
                f"- route proof target: `{proof_lanes['proof_target']}`",
                f"- handoff target: `{proof_lanes['handoff_target']}`",
                f"- readiness target: `{proof_lanes['readiness_target']}`",
                f"- stop condition: {proof_lanes['stop_condition']}",
                "",
                "Safety gates:",
                f"- approval triggers detected: {', '.join(approval_triggers) if approval_triggers else 'none from wording'}",
                "- Ready means ready for a bounded specialist draft, not ready for real-world action.",
                "- ToolRegistry and PermissionPolicy still gate every proposed tool after the specialist draft.",
                "- Risky execution still needs exact arguments, approval readiness, approval packet, approval chain proof, verification, audit, recovery, and learning evidence.",
            ]
        )

        return ToolResult(
            "specialist_execution_readiness",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=request,
                verdict=verdict,
                next_command=next_command,
                blockers=blockers,
                blocker_count=len(blockers),
                ready_for_specialist_model_draft=verdict == "READY_FOR_SPECIALIST_MODEL_DRAFT",
                primary_specialist=primary,
                supporting_specialists=supporting,
                candidate_scores={name: score for score, name in scored},
                primary_signal_score=primary_score,
                runner_up_signal_score=runner_up_score,
                score_margin=score_margin,
                ambiguous_route=ambiguous,
                verifier_tools=present_verifiers,
                missing_verifier_tools=missing_verifiers,
                verifier_coverage_percent=verifier_coverage,
                handoff_scorecard=scorecard,
                handoff_score=scorecard["score"],
                handoff_score_grade=scorecard["grade"],
                handoff_scorecard_max_points=scorecard["max_points"],
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                approval_triggers=approval_triggers,
                approval_required_by_wording=bool(approval_triggers),
                model_status=model_status,
                model_ready=model_ready,
                model_provider=model_readiness["model_provider"],
                model_configuration_ready=model_readiness["model_configuration_ready"],
                model_access_live_verified=model_readiness["model_access_live_verified"],
                chat_model=config.chat_model if config else "",
                planner_model=config.planner_model if config else "",
                target_model=proof_lanes["target_model"],
                fallback_lane=proof_lanes["fallback_lane"],
                proof_target=proof_lanes["proof_target"],
                handoff_target=proof_lanes["handoff_target"],
                readiness_target=proof_lanes["readiness_target"],
                stop_condition=proof_lanes["stop_condition"],
                model_planner_enabled=config.use_model_planner if config else False,
            ),
        )

    return specialist_execution_readiness


def make_specialist_handoff_receipt_tool(config: JarvisConfig | None, list_tools):
    def specialist_handoff_receipt(args: dict[str, Any]) -> ToolResult:
        request = _clean_text(args.get("request"), limit=MAX_HANDOFF_REQUEST_CHARS)
        if not request:
            return _missing_specialist_request_result(
                "specialist_handoff_receipt",
                "specialist handoff receipt: fix this code bug and run focused tests",
            )

        primary, supporting, scored = _specialist_route(request)
        profile = SPECIALIST_PROFILES[primary]
        tools = list_tools()
        tool_names = {tool.name for tool in tools}
        risk_counts = Counter(tool.risk.name for tool in tools)
        risk_gated = sum(1 for tool in tools if tool.risk.name in {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"})
        primary_score = next((score for score, name in scored if name == primary), 0)
        runner_up_score = next((score for score, name in scored if name != primary), 0)
        score_margin = primary_score - runner_up_score

        input_contracts = {
            "chat": [
                "user request text",
                "relevant memory/profile/preferences when already authorized",
                "recent conversation context",
            ],
            "planner": [
                "user request text",
                "available ToolRegistry entries",
                "PermissionPolicy risk levels",
                "current pending approval count",
            ],
            "code": [
                "user request text",
                "specific file paths or error output supplied by the user/runtime",
                "focused verification command candidates",
            ],
            "vision": [
                "user request text",
                "approved screenshot or observation evidence only",
                "expected visual state and verification target",
            ],
            "summarizer": [
                "user request text",
                "bounded source text or authorized stored context",
                "source limits and uncertainty notes",
            ],
            "reflection": [
                "user feedback, failure, or repeated workflow",
                "supporting evidence from audit/runtime traces",
                "candidate test, preference, memory, or skill change",
            ],
        }
        output_contracts = {
            "chat": "natural answer with uncertainty and no unsupported private claims",
            "planner": "exact tool plan, arguments, risk level, approval state, verification target, and recovery path",
            "code": "implementation intent, affected files, focused tests, and rollback or failure notes",
            "vision": "observed-vs-expected comparison and next safe visual verification step",
            "summarizer": "faithful brief separating evidence, inference, omissions, and next actions",
            "reflection": "reviewable learning packet that can become a test, preference, skill, or memory only with evidence",
        }
        present_verifiers = [name for name in SPECIALIST_VERIFIER_TOOLS[primary] if name in tool_names]
        missing_verifiers = [name for name in SPECIALIST_VERIFIER_TOOLS[primary] if name not in tool_names]
        verifier_targets = SPECIALIST_VERIFIER_TOOLS[primary]
        verifier_coverage = round((len(present_verifiers) / len(verifier_targets)) * 100) if verifier_targets else 0

        approval_triggers = _approval_triggers_from_request(request)
        ambiguous = primary_score == 0 or score_margin == 0
        if primary_score >= 2 and score_margin >= 1 and verifier_coverage >= 67:
            handoff_quality_state = "MEASURED_HANDOFF_READY"
            handoff_confidence = "high"
        elif primary_score >= 1 and verifier_coverage >= 50:
            handoff_quality_state = "HANDOFF_REVIEW_REQUIRED"
            handoff_confidence = "medium"
        else:
            handoff_quality_state = "HANDOFF_UNMEASURED_OR_AMBIGUOUS"
            handoff_confidence = "low"
        scorecard = _specialist_handoff_scorecard(
            primary_score=primary_score,
            score_margin=score_margin,
            verifier_coverage=verifier_coverage,
            approval_triggers=approval_triggers,
            ambiguous=ambiguous,
        )
        handoff_receipt_id = "handoff-" + hashlib.sha256(
            "|".join([request, primary, str(primary_score), str(score_margin), str(verifier_coverage)]).encode("utf-8")
        ).hexdigest()[:12]
        proof_lanes = _specialist_proof_lanes(request, primary, config)

        lines = [
            "Jarvis specialist handoff receipt:",
            "This is the steering-to-engine packet. It proves how a request would be packaged for a specialist brain before any model call or tool execution.",
            "",
            f"Handoff receipt id: {handoff_receipt_id}",
            f"Request: {request}",
            "",
            "Route receipt:",
            f"- primary specialist: {primary}",
            f"- supporting specialists: {', '.join(supporting) if supporting else 'none'}",
            f"- purpose: {profile['purpose']}",
            f"- route scores: "
            + ", ".join(f"{name}={score}" for score, name in scored),
            "",
            "Input contract:",
        ]
        lines.extend(f"- {item}" for item in input_contracts[primary])
        lines.extend(
            [
                "",
                "Output contract:",
                f"- {output_contracts[primary]}",
                "",
                "Verification hooks:",
                f"- available: {', '.join(present_verifiers) if present_verifiers else 'none'}",
                f"- missing: {', '.join(missing_verifiers) if missing_verifiers else 'none'}",
                f"- verifier coverage: {verifier_coverage}%",
                "",
                "Measured handoff quality:",
                f"- quality state: {handoff_quality_state}",
                f"- confidence: {handoff_confidence}",
                f"- scorecard score: {scorecard['score']}/100",
                f"- scorecard grade: {scorecard['grade']}",
                f"- model target: {proof_lanes['target_model'] or '<no config>'}",
                f"- fallback lane: {proof_lanes['fallback_lane']}",
                f"- primary signal score: {primary_score}",
                f"- runner-up margin: {score_margin}",
                f"- ambiguous route: {'yes' if ambiguous else 'no'}",
                f"- proof target: `{proof_lanes['proof_target']}`",
                f"- readiness target: `{proof_lanes['readiness_target']}`",
                f"- stop condition: {proof_lanes['stop_condition']}",
                "",
                "Safety gates:",
                f"- approval triggers detected: {', '.join(approval_triggers) if approval_triggers else 'none from wording'}",
                "- ToolRegistry must validate every proposed tool before use.",
                "- PermissionPolicy must stop shell/code, computer control, personal data, destructive changes, and external side effects for approval.",
                "- This receipt does not call models, execute tools, read personal data, write files, control the computer, or queue approvals.",
                "",
                "Harness telemetry:",
                f"- configured chat model: {config.chat_model if config else '<no config>'}",
                f"- configured planner model: {config.planner_model if config else '<no config>'}",
                f"- registered tools: {len(tools)}",
                f"- risk-gated tools: {risk_gated}",
            ]
        )
        for risk, count in sorted(risk_counts.items()):
            lines.append(f"- {risk}: {count}")

        return ToolResult(
            "specialist_handoff_receipt",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=request,
                handoff_receipt_id=handoff_receipt_id,
                primary_specialist=primary,
                supporting_specialists=supporting,
                candidate_scores={name: score for score, name in scored},
                handoff_quality_state=handoff_quality_state,
                handoff_confidence=handoff_confidence,
                handoff_quality_measured=handoff_quality_state != "HANDOFF_UNMEASURED_OR_AMBIGUOUS",
                primary_signal_score=primary_score,
                runner_up_signal_score=runner_up_score,
                score_margin=score_margin,
                ambiguous_route=ambiguous,
                input_contract_items=len(input_contracts[primary]),
                output_contract=output_contracts[primary],
                verifier_tools=present_verifiers,
                missing_verifier_tools=missing_verifiers,
                verifier_coverage_percent=verifier_coverage,
                handoff_scorecard=scorecard,
                handoff_score=scorecard["score"],
                handoff_score_grade=scorecard["grade"],
                handoff_scorecard_max_points=scorecard["max_points"],
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                target_model=proof_lanes["target_model"],
                fallback_lane=proof_lanes["fallback_lane"],
                proof_target=proof_lanes["proof_target"],
                handoff_target=proof_lanes["handoff_target"],
                readiness_target=proof_lanes["readiness_target"],
                stop_condition=proof_lanes["stop_condition"],
                approval_triggers=approval_triggers,
                approval_required_by_wording=bool(approval_triggers),
                tools=len(tools),
                risk_gated_tools=risk_gated,
                risk_counts=dict(risk_counts),
                chat_model=config.chat_model if config else "",
                planner_model=config.planner_model if config else "",
            ),
        )

    return specialist_handoff_receipt


def make_specialist_handoff_quality_gate_tool(config: JarvisConfig | None, list_tools):
    def specialist_handoff_quality_gate(args: dict[str, Any]) -> ToolResult:
        request = _clean_text(args.get("request"), limit=MAX_HANDOFF_REQUEST_CHARS)
        if not request:
            return _missing_specialist_request_result(
                "specialist_handoff_quality_gate",
                "specialist handoff quality gate: summarize this work history into a brief",
            )

        quality = make_specialist_route_quality_tool(config, list_tools)({"request": request})
        handoff = make_specialist_handoff_receipt_tool(config, list_tools)({"request": request})
        quality_meta = dict(quality.metadata)
        handoff_meta = dict(handoff.metadata)

        primary_matches = quality_meta.get("primary_specialist") == handoff_meta.get("primary_specialist")
        request_matches = quality_meta.get("request") == handoff_meta.get("request") == request
        stop_condition_matches = quality_meta.get("stop_condition") == handoff_meta.get("stop_condition")
        verifier_coverage = int(handoff_meta.get("verifier_coverage_percent") or 0)
        handoff_score = int(handoff_meta.get("handoff_score") or 0)
        approval_triggers = list(handoff_meta.get("approval_triggers") or [])
        blockers: list[str] = []
        if not request_matches:
            blockers.append("request_binding_mismatch")
        if not primary_matches:
            blockers.append("specialist_binding_mismatch")
        if not stop_condition_matches:
            blockers.append("stop_condition_binding_mismatch")
        if quality_meta.get("verdict") == "ROUTE_UNMEASURED_OR_AMBIGUOUS":
            blockers.append("route_quality_unmeasured_or_ambiguous")
        if not handoff_meta.get("handoff_quality_measured"):
            blockers.append("handoff_quality_unmeasured")
        if verifier_coverage < 50:
            blockers.append("thin_verifier_coverage")
        if handoff_score < 50:
            blockers.append("weak_handoff_score")
        if not handoff_meta.get("handoff_receipt_id"):
            blockers.append("missing_handoff_receipt_id")

        quality_binding_ready = not blockers
        if quality_binding_ready and approval_triggers:
            quality_gate_state = "HANDOFF_QUALITY_READY_WITH_APPROVAL_BOUNDARY"
        elif quality_binding_ready:
            quality_gate_state = "HANDOFF_QUALITY_READY_FOR_PROPOSAL_GATE"
        else:
            quality_gate_state = "HANDOFF_QUALITY_HELD_FOR_REVIEW"

        required_proof_commands = [
            str(quality_meta.get("proof_target") or f"specialist route quality: {request}"),
            str(handoff_meta.get("readiness_target") or f"specialist execution readiness: {request}"),
            str(handoff_meta.get("handoff_target") or f"specialist handoff receipt: {request}"),
            f"specialist proposal gate: {request}",
        ]
        next_command = (
            f"specialist proposal gate: {request}"
            if quality_binding_ready
            else str(quality_meta.get("proof_target") or f"specialist route quality: {request}")
        )

        lines = [
            "Jarvis specialist handoff quality gate:",
            "This is read-only. It binds route quality and the handoff receipt before any specialist draft can be treated as ready for proposal-gate review.",
            "",
            f"Request: {request}",
            "",
            "Quality gate:",
            f"- state: {quality_gate_state}",
            f"- quality binding ready: {'yes' if quality_binding_ready else 'no'}",
            "- action proposal locked: yes",
            "- can emit tool proposals: no",
            f"- next command: `{next_command}`",
            f"- blockers: {', '.join(blockers) if blockers else 'none'}",
            "",
            "Bound route and handoff evidence:",
            f"- route verdict: {quality_meta.get('verdict')}",
            f"- handoff receipt id: {handoff_meta.get('handoff_receipt_id') or 'missing'}",
            f"- primary specialist: {handoff_meta.get('primary_specialist')}",
            f"- route and handoff specialist match: {'yes' if primary_matches else 'no'}",
            f"- handoff quality state: {handoff_meta.get('handoff_quality_state')}",
            f"- handoff score: {handoff_score}/100 ({handoff_meta.get('handoff_score_grade')})",
            f"- verifier coverage: {verifier_coverage}%",
            f"- ambiguous route: {'yes' if handoff_meta.get('ambiguous_route') else 'no'}",
            f"- stop condition bound: {'yes' if stop_condition_matches else 'no'}",
            "",
            "Approval and proposal boundary:",
            f"- approval triggers detected: {', '.join(approval_triggers) if approval_triggers else 'none from wording'}",
            "- A ready quality gate permits only `specialist proposal gate` review.",
            "- Tool proposals remain locked until ToolRegistry, PermissionPolicy, exact arguments, approval readiness, approval packet, approval chain proof, verification, audit, recovery, and learning evidence are present.",
            "",
            "Required proof commands:",
            *[f"- `{command}`" for command in required_proof_commands],
            "",
            "Safety boundary:",
            "- This gate does not call models, execute tools, approve requests, dismiss approvals, read personal data, write files, control the computer, call external services, or queue approvals.",
        ]

        return ToolResult(
            "specialist_handoff_quality_gate",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=request,
                quality_gate_state=quality_gate_state,
                quality_binding_ready=quality_binding_ready,
                ready_for_proposal_gate_review=quality_binding_ready,
                action_proposal_locked=True,
                can_emit_tool_proposals=False,
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                next_command=next_command,
                blockers=blockers,
                blocker_count=len(blockers),
                route_verdict=quality_meta.get("verdict"),
                route_confidence=quality_meta.get("confidence"),
                handoff_receipt_id=handoff_meta.get("handoff_receipt_id"),
                request_binding_ready=request_matches,
                specialist_binding_ready=primary_matches,
                stop_condition_binding_ready=stop_condition_matches,
                primary_specialist=handoff_meta.get("primary_specialist"),
                supporting_specialists=handoff_meta.get("supporting_specialists") or [],
                handoff_quality_state=handoff_meta.get("handoff_quality_state"),
                handoff_quality_measured=bool(handoff_meta.get("handoff_quality_measured")),
                handoff_score=handoff_score,
                handoff_score_grade=handoff_meta.get("handoff_score_grade"),
                handoff_scorecard=handoff_meta.get("handoff_scorecard"),
                handoff_scorecard_max_points=handoff_meta.get("handoff_scorecard_max_points"),
                verifier_tools=handoff_meta.get("verifier_tools") or [],
                missing_verifier_tools=handoff_meta.get("missing_verifier_tools") or [],
                verifier_coverage_percent=verifier_coverage,
                approval_triggers=approval_triggers,
                approval_required_by_wording=bool(approval_triggers),
                target_model=handoff_meta.get("target_model"),
                fallback_lane=handoff_meta.get("fallback_lane"),
                proof_target=quality_meta.get("proof_target"),
                handoff_target=handoff_meta.get("handoff_target"),
                readiness_target=handoff_meta.get("readiness_target"),
                stop_condition=handoff_meta.get("stop_condition"),
                required_proof_commands=required_proof_commands,
                required_proof_command_count=len(required_proof_commands),
            ),
        )

    return specialist_handoff_quality_gate


def make_specialist_proposal_gate_tool(config: JarvisConfig | None, list_tools):
    def specialist_proposal_gate(args: dict[str, Any]) -> ToolResult:
        request = _clean_text(args.get("request"), limit=MAX_HANDOFF_REQUEST_CHARS)
        if not request:
            return _missing_specialist_request_result(
                "specialist_proposal_gate",
                "specialist proposal gate: summarize this work history into a brief",
            )

        primary, supporting, scored = _specialist_route(request)
        tools = list_tools()
        tool_names = {tool.name for tool in tools}
        primary_score = next((score for score, name in scored if name == primary), 0)
        runner_up_score = next((score for score, name in scored if name != primary), 0)
        score_margin = primary_score - runner_up_score
        verifier_targets = SPECIALIST_VERIFIER_TOOLS[primary]
        present_verifiers = [name for name in verifier_targets if name in tool_names]
        missing_verifiers = [name for name in verifier_targets if name not in tool_names]
        verifier_coverage = round((len(present_verifiers) / len(verifier_targets)) * 100) if verifier_targets else 0
        approval_triggers = _approval_triggers_from_request(request)
        ambiguous = primary_score == 0 or score_margin == 0
        proof_lanes = _specialist_proof_lanes(request, primary, config)

        model_readiness = _specialist_model_readiness(config, proof_lanes["target_model"])
        model_status = str(model_readiness["model_status"])
        model_ready = model_readiness["model_ready"] is True
        model_detail = str(model_readiness["model_detail"])

        scorecard = _specialist_handoff_scorecard(
            primary_score=primary_score,
            score_margin=score_margin,
            verifier_coverage=verifier_coverage,
            approval_triggers=approval_triggers,
            ambiguous=ambiguous,
            model_ready=model_ready if config is not None else None,
        )

        blockers: list[str] = []
        if ambiguous:
            blockers.append("ambiguous_specialist_route")
        if verifier_coverage < 67:
            blockers.append("thin_verifier_coverage")
        if approval_triggers:
            blockers.append("approval_gate_required_before_action_proposal")
        if config is not None and not model_ready:
            blockers.append("configured_specialist_model_not_ready")

        if approval_triggers:
            proposal_gate_state = "PROPOSAL_HELD_FOR_APPROVAL_BOUNDARY"
        elif ambiguous or verifier_coverage < 67:
            proposal_gate_state = "PROPOSAL_HELD_FOR_ROUTE_REVIEW"
        elif config is not None and not model_ready:
            proposal_gate_state = "PROPOSAL_FALLBACK_NO_MODEL"
        else:
            proposal_gate_state = "PROPOSAL_DRAFT_READY"

        can_enter_specialist_draft = proposal_gate_state == "PROPOSAL_DRAFT_READY"
        required_proof_commands = [
            proof_lanes["proof_target"],
            proof_lanes["readiness_target"],
            proof_lanes["handoff_target"],
            f"specialist model draft: {request}",
        ]
        if approval_triggers:
            required_proof_commands.extend(["approval readiness latest", "approval packet latest", "approval chain proof latest"])
        next_command = (
            "model routing status"
            if proposal_gate_state == "PROPOSAL_FALLBACK_NO_MODEL"
            else proof_lanes["readiness_target"]
            if proposal_gate_state == "PROPOSAL_HELD_FOR_ROUTE_REVIEW"
            else proof_lanes["handoff_target"]
        )

        lines = [
            "Jarvis specialist proposal gate:",
            "This is read-only. It joins route quality, execution readiness, handoff receipt, verifier coverage, model readiness, and approval boundaries before a specialist draft can become any tool proposal.",
            "",
            f"Request: {request}",
            "",
            "Proposal gate:",
            f"- state: {proposal_gate_state}",
            f"- can enter specialist draft: {'yes' if can_enter_specialist_draft else 'no'}",
            "- action proposal locked: yes",
            f"- next command: `{next_command}`",
            f"- blockers: {', '.join(blockers) if blockers else 'none'}",
            "",
            "Route and handoff proof:",
            f"- primary specialist: {primary}",
            f"- supporting specialists: {', '.join(supporting) if supporting else 'none'}",
            f"- primary signal score: {primary_score}",
            f"- runner-up margin: {score_margin}",
            f"- ambiguous route: {'yes' if ambiguous else 'no'}",
            f"- handoff score: {scorecard['score']}/100 ({scorecard['grade']})",
            "",
            "Verifier and model proof:",
            f"- verifier coverage: {verifier_coverage}%",
            f"- available verifier hooks: {', '.join(present_verifiers) if present_verifiers else 'none'}",
            f"- missing verifier hooks: {', '.join(missing_verifiers) if missing_verifiers else 'none'}",
            f"- target model: {proof_lanes['target_model'] or '<no config>'}",
            f"- model provider: {model_readiness['model_provider'] or '<no config>'}",
            f"- model status: {model_status}",
            f"- model configuration ready: {'yes' if model_readiness['model_configuration_ready'] else 'no'}",
            f"- model live access verified: {'yes' if model_readiness['model_access_live_verified'] else 'no'}",
        ]
        if model_detail and not model_ready:
            lines.append(f"- model detail: {_clean_text(model_detail, limit=MAX_MODEL_DETAIL_CHARS)}")
        lines.extend(
            [
                "",
                "Approval boundary:",
                f"- approval triggers detected: {', '.join(approval_triggers) if approval_triggers else 'none from wording'}",
                "- A ready proposal gate permits only a bounded specialist draft.",
                "- Tool proposals remain locked until ToolRegistry, PermissionPolicy, exact arguments, approval readiness, approval packet, approval chain proof, verification, audit, recovery, and learning evidence are present.",
                "",
                "Required proof commands:",
                *[f"- `{command}`" for command in required_proof_commands],
                "",
                "Safety boundary:",
                "- This gate does not call models, execute tools, approve requests, dismiss approvals, read personal data, write files, control the computer, call external services, or queue approvals.",
            ]
        )

        return ToolResult(
            "specialist_proposal_gate",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=request,
                proposal_gate_state=proposal_gate_state,
                can_enter_specialist_draft=can_enter_specialist_draft,
                action_proposal_locked=True,
                can_emit_tool_proposals=False,
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                next_command=next_command,
                blockers=blockers,
                blocker_count=len(blockers),
                primary_specialist=primary,
                supporting_specialists=supporting,
                candidate_scores={name: score for score, name in scored},
                primary_signal_score=primary_score,
                runner_up_signal_score=runner_up_score,
                score_margin=score_margin,
                ambiguous_route=ambiguous,
                verifier_tools=present_verifiers,
                missing_verifier_tools=missing_verifiers,
                verifier_coverage_percent=verifier_coverage,
                handoff_scorecard=scorecard,
                handoff_score=scorecard["score"],
                handoff_score_grade=scorecard["grade"],
                handoff_scorecard_max_points=scorecard["max_points"],
                approval_triggers=approval_triggers,
                approval_required_by_wording=bool(approval_triggers),
                model_status=model_status,
                model_ready=model_ready,
                model_provider=model_readiness["model_provider"],
                model_configuration_ready=model_readiness["model_configuration_ready"],
                model_access_live_verified=model_readiness["model_access_live_verified"],
                target_model=proof_lanes["target_model"],
                fallback_lane=proof_lanes["fallback_lane"],
                proof_target=proof_lanes["proof_target"],
                handoff_target=proof_lanes["handoff_target"],
                readiness_target=proof_lanes["readiness_target"],
                stop_condition=proof_lanes["stop_condition"],
                required_proof_commands=required_proof_commands,
                required_proof_command_count=len(required_proof_commands),
            ),
        )

    return specialist_proposal_gate


def make_specialist_model_draft_tool(config: JarvisConfig | None, list_tools):
    def specialist_model_draft(args: dict[str, Any]) -> ToolResult:
        request = _clean_text(args.get("request"), limit=MAX_HANDOFF_REQUEST_CHARS)
        if not request:
            return _missing_specialist_request_result(
                "specialist_model_draft",
                "specialist model draft: summarize this work history into a brief",
            )

        primary, supporting, scored = _specialist_route(request)
        tools = list_tools()
        tool_names = {tool.name for tool in tools}
        profile = SPECIALIST_PROFILES[primary]
        primary_score = next((score for score, name in scored if name == primary), 0)
        runner_up_score = next((score for score, name in scored if name != primary), 0)
        score_margin = primary_score - runner_up_score
        verifier_targets = SPECIALIST_VERIFIER_TOOLS[primary]
        present_verifiers = [name for name in verifier_targets if name in tool_names]
        verifier_coverage = round((len(present_verifiers) / len(verifier_targets)) * 100) if verifier_targets else 0
        approval_triggers = _approval_triggers_from_request(request)
        ambiguous = primary_score == 0 or score_margin == 0
        target_model = ""
        draft = ""
        model_error = ""
        calls_model = False
        model_provider = normalized_model_provider(config.model_provider) if config is not None else ""
        proof_lanes = _specialist_proof_lanes(request, primary, config)
        target_model = proof_lanes["target_model"]
        model_readiness = _specialist_model_readiness(config, target_model)
        model_status = str(model_readiness["model_status"])
        model_ready = model_readiness["model_ready"] is True
        model_detail = str(model_readiness["model_detail"])
        scorecard = _specialist_handoff_scorecard(
            primary_score=primary_score,
            score_margin=score_margin,
            verifier_coverage=verifier_coverage,
            approval_triggers=approval_triggers,
            ambiguous=ambiguous,
            model_ready=model_ready if config is not None else None,
        )
        if approval_triggers:
            draft_state = "DRAFT_HELD_FOR_APPROVAL_GATE"
            fallback_reason = "approval-triggering wording requires handoff and approval review before any specialist draft"
        elif ambiguous or verifier_coverage < 67:
            draft_state = "DRAFT_HELD_FOR_ROUTE_REVIEW"
            fallback_reason = "route is ambiguous or verifier coverage is thin"
        elif not model_ready:
            draft_state = "DRAFT_FALLBACK_PREVIEW"
            fallback_reason = "configured specialist model is unavailable, so Jarvis returns the bounded handoff contract instead"
        else:
            try:
                calls_model = True
                prompt = (
                    f"Specialist: {primary}\n"
                    f"Purpose: {profile['purpose']}\n"
                    f"Safety: Draft only. Do not claim tools ran, files changed, private data was read, or approvals were granted.\n"
                    f"Request: {request}\n"
                    "Return a concise draft plus verification notes."
                )
                draft = generate_model_text(
                    provider=config.model_provider if config else "ollama",
                    model=target_model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout_seconds=config.model_timeout_seconds if config else 2.5,
                    max_output_tokens=provider_output_token_limit(
                        config.model_provider if config else "ollama",
                        local_output_tokens=600,
                        openai_output_tokens=(config.openai_max_output_tokens if config else 25_000),
                    ),
                    temperature=0.2,
                    reasoning_effort=config.chat_reasoning_effort if config else "medium",
                )
                draft = _clean_text(draft, limit=MAX_SPECIALIST_DRAFT_CHARS)
                draft_state = "DRAFT_READY"
                fallback_reason = ""
            except Exception as exc:
                model_error = _clean_text(exc, limit=MAX_MODEL_DETAIL_CHARS)
                draft_state = "DRAFT_FALLBACK_PREVIEW"
                fallback_reason = "model call failed within the bounded draft path"

        external_model_call_attempted = bool(calls_model and model_provider == "openai")

        handoff_command = proof_lanes["handoff_target"]
        quality_command = proof_lanes["proof_target"]
        readiness_command = proof_lanes["readiness_target"]
        lines = [
            "Jarvis specialist model draft:",
            "This is a bounded specialist-draft lane. It never executes tools, approves requests, reads personal data, controls the computer, writes files, or changes outside-world state.",
            "",
            f"Request: {request}",
            "",
            "Draft gate:",
            f"- draft state: {draft_state}",
            f"- primary specialist: {primary}",
            f"- supporting specialists: {', '.join(supporting) if supporting else 'none'}",
            f"- target model: {target_model or '<no config>'}",
            f"- model provider: {model_provider or '<no config>'}",
            f"- model call attempted: {'yes' if calls_model else 'no'}",
            f"- external model call attempted: {'yes' if external_model_call_attempted else 'no'}",
            f"- model status: {model_status}",
            f"- model configuration ready: {'yes' if model_readiness['model_configuration_ready'] else 'no'}",
            f"- model live access verified: {'yes' if model_readiness['model_access_live_verified'] else 'no'}",
            f"- model timeout: {config.model_timeout_seconds:g}s" if config else "- model timeout: <no config>",
            f"- approval triggers detected: {', '.join(approval_triggers) if approval_triggers else 'none from wording'}",
            f"- verifier coverage: {verifier_coverage}%",
            f"- ambiguous route: {'yes' if ambiguous else 'no'}",
            f"- handoff score: {scorecard['score']}/100 ({scorecard['grade']})",
            "",
            "Specialist proof lane:",
            f"- fallback lane: {proof_lanes['fallback_lane']}",
            f"- route proof target: `{quality_command}`",
            f"- handoff target: `{handoff_command}`",
            f"- readiness target: `{readiness_command}`",
            f"- stop condition: {proof_lanes['stop_condition']}",
            "",
            "Required proof before relying on this lane:",
            f"- `{quality_command}`",
            f"- `{readiness_command}`",
            f"- `{handoff_command}`",
            "",
            "Draft output:",
        ]
        if draft:
            lines.append(draft)
        else:
            lines.append(f"- No model draft was produced: {fallback_reason}.")
            lines.append(f"- Use the handoff receipt first: `{handoff_command}`")
        if model_error:
            lines.append(f"- model error: {model_error}")
        if model_detail and not model_ready:
            lines.append(f"- model detail: {_clean_text(model_detail, limit=MAX_MODEL_DETAIL_CHARS)}")
        lines.extend(
            [
                "",
                "Stop conditions:",
                "- stop if the route is ambiguous, verifier coverage is thin, model readiness is missing, approval-triggering work appears, or the draft asks to act without a tool/approval proof chain",
                "- risky execution still needs exact arguments, approval readiness, last-look approval packet, approval chain proof, verification, audit, recovery, and learning evidence",
            ]
        )

        return ToolResult(
            "specialist_model_draft",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=request,
                calls_model=calls_model,
                calls_external_service=external_model_call_attempted,
                model_provider=model_provider,
                model_call_attempted=calls_model,
                external_model_call_attempted=external_model_call_attempted,
                shares_specialist_request_with_external_model=external_model_call_attempted,
                model_response_content_in_metadata=False,
                draft_content_in_metadata=False,
                draft_state=draft_state,
                draft_produced=bool(draft),
                fallback_reason=fallback_reason,
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                action_proposal_locked=True,
                can_emit_tool_proposals=False,
                primary_specialist=primary,
                supporting_specialists=supporting,
                candidate_scores={name: score for score, name in scored},
                primary_signal_score=primary_score,
                runner_up_signal_score=runner_up_score,
                score_margin=score_margin,
                ambiguous_route=ambiguous,
                verifier_tools=present_verifiers,
                verifier_coverage_percent=verifier_coverage,
                handoff_scorecard=scorecard,
                handoff_score=scorecard["score"],
                handoff_score_grade=scorecard["grade"],
                handoff_scorecard_max_points=scorecard["max_points"],
                approval_triggers=approval_triggers,
                approval_required_by_wording=bool(approval_triggers),
                model_status=model_status,
                model_ready=model_ready,
                model_configuration_ready=model_readiness["model_configuration_ready"],
                model_access_live_verified=model_readiness["model_access_live_verified"],
                model_error=model_error,
                target_model=target_model,
                fallback_lane=proof_lanes["fallback_lane"],
                proof_target=quality_command,
                handoff_command=handoff_command,
                quality_command=quality_command,
                readiness_command=readiness_command,
                stop_condition=proof_lanes["stop_condition"],
            ),
        )

    return specialist_model_draft


def make_specialist_action_proposal_contract_tool(config: JarvisConfig | None, list_tools):
    def specialist_action_proposal_contract(args: dict[str, Any]) -> ToolResult:
        raw_request = _clean_text(args.get("request") or args.get("proposal") or "", limit=MAX_HANDOFF_REQUEST_CHARS)
        if not raw_request:
            return _missing_specialist_request_result(
                "specialist_action_proposal_contract",
                "specialist action proposal contract: fix this bug; tool run_shell_command; args command=python3 -m py_compile ...",
            )

        proposed_tool = _clean_text(
            args.get("tool") or args.get("tool_name") or _proposal_field_from_text(raw_request, ("tool", "tool name", "action tool")),
            limit=120,
        )
        proposed_arguments = _clean_text(
            args.get("arguments") or args.get("args") or _proposal_field_from_text(raw_request, ("args", "arguments", "input", "payload")),
            limit=MAX_PROPOSAL_ARGUMENT_CHARS,
        )
        request = _proposal_request_without_fields(raw_request)
        tools = list_tools()
        tool_by_name = {tool.name: tool for tool in tools}
        registry_match = tool_by_name.get(proposed_tool) if proposed_tool else None
        registered_tool = registry_match is not None
        risk_level = registry_match.risk.name if registry_match else "UNKNOWN"
        toolset = registry_match.toolset if registry_match else ""
        exact_arguments_supplied = _arguments_are_exact(proposed_arguments)
        primary, supporting, scored = _specialist_route(request)
        approval_triggers = _approval_triggers_from_request(f"{request} {proposed_arguments}")
        risk_requires_approval = risk_level in {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"}
        approval_required = risk_requires_approval or bool(approval_triggers)
        proof_lanes = _specialist_proof_lanes(request, primary, config)
        proof_lane_bound = bool(proof_lanes.get("proof_target") and proof_lanes.get("readiness_target") and proof_lanes.get("handoff_target"))
        proposal_gate_command = f"specialist proposal gate: {request}"
        model_draft_command = f"specialist model draft: {request}"
        tool_registry_command = f"tool detail: {proposed_tool}" if proposed_tool else "tool search: <proposed tool>"
        argument_contract_command = f"argument contract: {request}"
        risk_preflight_command = f"risk preflight: {request}"
        verification_command = f"verification packet: {request}"
        local_safe_or_read_only = risk_level in {"READ_ONLY", "LOCAL_SAFE"}

        blockers: list[str] = []
        if not proposed_tool:
            blockers.append("missing_proposed_tool")
        elif not registered_tool:
            blockers.append("tool_not_registered")
        if not exact_arguments_supplied:
            blockers.append("exact_arguments_missing")
        if approval_required:
            blockers.append("approval_boundary_required")

        if not proposed_tool or not registered_tool:
            contract_state = "HELD_FOR_TOOL_REGISTRY"
            next_command = tool_registry_command
        elif not exact_arguments_supplied:
            contract_state = "HELD_FOR_EXACT_ARGUMENTS"
            next_command = argument_contract_command
        elif approval_required:
            contract_state = "HELD_FOR_APPROVAL_BOUNDARY"
            next_command = "approval readiness latest"
        else:
            contract_state = "READY_FOR_LOCAL_SAFE_DRY_RUN_REVIEW"
            next_command = verification_command

        required_proof_commands = [
            proof_lanes["proof_target"],
            proof_lanes["readiness_target"],
            proof_lanes["handoff_target"],
            f"specialist handoff quality gate: {request}",
            proposal_gate_command,
            model_draft_command,
            tool_registry_command,
            argument_contract_command,
            risk_preflight_command,
            verification_command,
        ]
        if approval_required:
            required_proof_commands.extend(["approval readiness latest", "approval packet latest", "approval chain proof latest"])
        scorecard = _specialist_action_proposal_scorecard(
            registered_tool=registered_tool,
            exact_arguments_supplied=exact_arguments_supplied,
            verification_supplied=False,
            approval_required=approval_required,
            local_safe_or_read_only=local_safe_or_read_only,
            proof_lane_bound=proof_lane_bound,
        )
        scorecard_rows = _specialist_action_proposal_scorecard_rows(scorecard)
        scorecard_shape_ready = _specialist_action_proposal_scorecard_shape_ready(scorecard_rows)
        scorecard_required_rows_ready = _specialist_action_proposal_scorecard_shape_ready(scorecard_rows, require_all_passed=True)
        action_proposal_review_token_sha256 = _specialist_action_proposal_review_token_sha256(
            request=request,
            proposed_tool=proposed_tool,
            proposed_arguments=proposed_arguments,
            verification="",
            primary_specialist=primary,
            target_model=proof_lanes["target_model"],
            risk_level=risk_level,
            review_state=contract_state,
            next_command=next_command,
            scorecard_rows=scorecard_rows,
        )

        lines = [
            "Jarvis specialist action proposal contract:",
            "This is read-only. It inspects whether specialist output may even be shaped as a tool proposal; it never executes tools, approves requests, writes files, reads personal data, controls the computer, or queues approvals.",
            "",
            f"Request: {request}",
            "",
            "Proposal contract:",
            f"- state: {contract_state}",
            "- action execution locked: yes",
            "- can emit executable tool action: no",
            f"- next command: `{next_command}`",
            f"- blockers: {', '.join(blockers) if blockers else 'none for local-safe dry-run review'}",
            "",
            "Specialist binding:",
            f"- primary specialist: {primary}",
            f"- supporting specialists: {', '.join(supporting) if supporting else 'none'}",
            f"- route proof: `{proof_lanes['proof_target']}`",
            f"- readiness proof: `{proof_lanes['readiness_target']}`",
            f"- handoff proof: `{proof_lanes['handoff_target']}`",
            f"- proposal gate: `{proposal_gate_command}`",
            "",
            "ToolRegistry and PermissionPolicy:",
            f"- proposed tool: {proposed_tool or 'missing'}",
            f"- registered in ToolRegistry: {'yes' if registered_tool else 'no'}",
            f"- toolset: {toolset or 'unknown'}",
            f"- risk level: {risk_level}",
            f"- exact arguments supplied: {'yes' if exact_arguments_supplied else 'no'}",
            f"- approval required before execution: {'yes' if approval_required else 'no'}",
            f"- approval triggers detected: {', '.join(approval_triggers) if approval_triggers else 'none from wording/arguments'}",
            "",
            "Measured action proposal scorecard:",
            f"- score: {scorecard['score']}/100",
            f"- grade: {scorecard['grade']}",
            f"- registry points: {scorecard['registry_points']}",
            f"- argument points: {scorecard['argument_points']}",
            f"- verification points: {scorecard['verification_points']} (verified in dry-run packet)",
            f"- permission points: {scorecard['permission_points']}",
            f"- proof lane points: {scorecard['proof_lane_points']}",
            f"- execution lock points: {scorecard['execution_lock_points']}",
            f"- evidence rows ready: {'yes' if scorecard_required_rows_ready else 'no'}",
            f"- action proposal review token sha256: {action_proposal_review_token_sha256}",
            "- action proposal review token authorizes executable action: no",
            "- action proposal review token reusable for next review: no",
            *[
                f"- {row['item']}: {row['points']}/{row['max_points']} points; passed {'yes' if row['passed'] else 'no'}"
                for row in scorecard_rows
            ],
            "",
            "Required proof commands:",
            *[f"- `{command}`" for command in required_proof_commands],
            "",
            "Boundary:",
            "- A ready local-safe contract still permits only dry-run review and verification planning.",
            "- Any real shell/code, computer-control, personal-data, external side-effect, destructive, or ambiguous action remains behind approval readiness, approval packet, approval chain proof, execution receipt, verification, audit, recovery, and learning evidence.",
        ]

        return ToolResult(
            "specialist_action_proposal_contract",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=request,
                proposed_tool=proposed_tool,
                proposed_arguments=proposed_arguments,
                contract_state=contract_state,
                action_execution_locked=True,
                can_emit_executable_tool_action=False,
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                proposal_ready_for_local_safe_dry_run_review=contract_state == "READY_FOR_LOCAL_SAFE_DRY_RUN_REVIEW",
                next_command=next_command,
                blockers=blockers,
                blocker_count=len(blockers),
                primary_specialist=primary,
                supporting_specialists=supporting,
                candidate_scores={name: score for score, name in scored},
                registered_tool=registered_tool,
                toolset=toolset,
                risk_level=risk_level,
                exact_arguments_supplied=exact_arguments_supplied,
                local_safe_or_read_only=local_safe_or_read_only,
                approval_required_before_execution=approval_required,
                risk_requires_approval=risk_requires_approval,
                approval_triggers=approval_triggers,
                action_proposal_scorecard=scorecard,
                action_proposal_scorecard_rows=scorecard_rows,
                action_proposal_scorecard_row_count=len(scorecard_rows),
                action_proposal_scorecard_shape_ready=scorecard_shape_ready,
                action_proposal_scorecard_required_rows_ready=scorecard_required_rows_ready,
                action_proposal_score=scorecard["score"],
                action_proposal_grade=scorecard["grade"],
                action_proposal_scorecard_max_points=scorecard["max_points"],
                action_proposal_review_token_sha256=action_proposal_review_token_sha256,
                action_proposal_review_token_present=_looks_like_sha256(action_proposal_review_token_sha256),
                action_proposal_review_token_authorizes_model_call=False,
                action_proposal_review_token_authorizes_tool_execution=False,
                action_proposal_review_token_authorizes_approval=False,
                action_proposal_review_token_authorizes_personal_data_read=False,
                action_proposal_review_token_authorizes_external_side_effect=False,
                action_proposal_review_token_authorizes_executable_action=False,
                action_proposal_review_token_reusable_for_next_review=False,
                next_action_proposal_review_requires_new_token=True,
                proof_lane_bound=proof_lane_bound,
                target_model=proof_lanes["target_model"],
                fallback_lane=proof_lanes["fallback_lane"],
                proof_target=proof_lanes["proof_target"],
                handoff_target=proof_lanes["handoff_target"],
                readiness_target=proof_lanes["readiness_target"],
                stop_condition=proof_lanes["stop_condition"],
                proposal_gate_command=proposal_gate_command,
                model_draft_command=model_draft_command,
                tool_registry_command=tool_registry_command,
                argument_contract_command=argument_contract_command,
                risk_preflight_command=risk_preflight_command,
                verification_command=verification_command,
                required_proof_commands=required_proof_commands,
                required_proof_command_count=len(required_proof_commands),
            ),
        )

    return specialist_action_proposal_contract


def make_specialist_tool_dry_run_packet_tool(config: JarvisConfig | None, list_tools):
    def specialist_tool_dry_run_packet(args: dict[str, Any]) -> ToolResult:
        raw_request = _clean_text(args.get("request") or args.get("proposal") or "", limit=MAX_HANDOFF_REQUEST_CHARS)
        if not raw_request:
            return _missing_specialist_request_result(
                "specialist_tool_dry_run_packet",
                "specialist tool dry run: summarize memory; tool brain_search; args query=Jarvis harness; verification cites local memory only",
            )

        proposed_tool = _clean_text(
            args.get("tool") or args.get("tool_name") or _proposal_field_from_text(raw_request, ("tool", "tool name", "action tool")),
            limit=120,
        )
        proposed_arguments = _clean_text(
            args.get("arguments") or args.get("args") or _proposal_field_from_text(raw_request, ("args", "arguments", "input", "payload")),
            limit=MAX_PROPOSAL_ARGUMENT_CHARS,
        )
        verification = _clean_text(
            args.get("verification") or args.get("expected") or _proposal_field_from_text(raw_request, ("verification", "expected", "expectation", "success")),
            limit=MAX_PROPOSAL_ARGUMENT_CHARS,
        )
        request = _proposal_request_without_fields(raw_request)
        tools = list_tools()
        tool_by_name = {tool.name: tool for tool in tools}
        registry_match = tool_by_name.get(proposed_tool) if proposed_tool else None
        registered_tool = registry_match is not None
        risk_level = registry_match.risk.name if registry_match else "UNKNOWN"
        toolset = registry_match.toolset if registry_match else ""
        exact_arguments_supplied = _arguments_are_exact(proposed_arguments)
        verification_supplied = bool(verification)
        primary, supporting, scored = _specialist_route(request)
        approval_triggers = _approval_triggers_from_request(f"{request} {proposed_arguments}")
        risk_requires_approval = risk_level in {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"}
        approval_required = risk_requires_approval or bool(approval_triggers)
        local_safe_or_read_only = risk_level in {"READ_ONLY", "LOCAL_SAFE"}
        proof_lanes = _specialist_proof_lanes(request, primary, config)
        proof_lane_bound = bool(proof_lanes.get("proof_target") and proof_lanes.get("readiness_target") and proof_lanes.get("handoff_target"))
        contract_command = f"specialist action proposal contract: {raw_request}"
        verification_command = f"verification packet: {request}"
        receipt_command = "verification receipt <dry-run review id>"

        missing: list[str] = []
        if not proposed_tool:
            missing.append("proposed_tool")
        elif not registered_tool:
            missing.append("registered_tool")
        if not exact_arguments_supplied:
            missing.append("exact_arguments")
        if not verification_supplied:
            missing.append("verification_expectation")
        if approval_required:
            missing.append("approval_boundary")

        if not registered_tool:
            state = "DRY_RUN_HELD_FOR_TOOL_REGISTRY"
            next_command = f"tool search: {proposed_tool or '<proposed tool>'}"
        elif not exact_arguments_supplied:
            state = "DRY_RUN_HELD_FOR_EXACT_ARGUMENTS"
            next_command = f"argument contract: {request}"
        elif not verification_supplied:
            state = "DRY_RUN_HELD_FOR_VERIFICATION_EXPECTATION"
            next_command = verification_command
        elif approval_required:
            state = "DRY_RUN_HELD_FOR_APPROVAL_BOUNDARY"
            next_command = "approval readiness latest"
        elif local_safe_or_read_only:
            state = "DRY_RUN_READY_FOR_OPERATOR_REVIEW"
            next_command = verification_command
        else:
            state = "DRY_RUN_HELD_FOR_PERMISSION_POLICY"
            next_command = "risk preflight: " + request

        proof_queue = [
            proof_lanes["proof_target"],
            proof_lanes["readiness_target"],
            proof_lanes["handoff_target"],
            f"specialist handoff quality gate: {request}",
            f"specialist proposal gate: {request}",
            contract_command,
            f"tool detail: {proposed_tool}" if proposed_tool else "tool search: <proposed tool>",
            f"argument contract: {request}",
            f"risk preflight: {request}",
            verification_command,
            receipt_command,
        ]
        if approval_required:
            proof_queue.extend(["approval readiness latest", "approval packet latest", "approval chain proof latest"])
        scorecard = _specialist_action_proposal_scorecard(
            registered_tool=registered_tool,
            exact_arguments_supplied=exact_arguments_supplied,
            verification_supplied=verification_supplied,
            approval_required=approval_required,
            local_safe_or_read_only=local_safe_or_read_only,
            proof_lane_bound=proof_lane_bound,
        )
        scorecard_rows = _specialist_action_proposal_scorecard_rows(scorecard)
        scorecard_shape_ready = _specialist_action_proposal_scorecard_shape_ready(scorecard_rows)
        scorecard_required_rows_ready = _specialist_action_proposal_scorecard_shape_ready(scorecard_rows, require_all_passed=True)
        action_proposal_review_token_sha256 = _specialist_action_proposal_review_token_sha256(
            request=request,
            proposed_tool=proposed_tool,
            proposed_arguments=proposed_arguments,
            verification=verification,
            primary_specialist=primary,
            target_model=proof_lanes["target_model"],
            risk_level=risk_level,
            review_state=state,
            next_command=next_command,
            scorecard_rows=scorecard_rows,
        )

        lines = [
            "Jarvis specialist tool dry-run packet:",
            "This is read-only. It reviews whether a specialist tool proposal has enough ToolRegistry, exact-argument, PermissionPolicy, and verification evidence for operator review without executing the tool.",
            "",
            f"Request: {request}",
            "",
            "Dry-run verdict:",
            f"- state: {state}",
            "- tool executed: no",
            "- executable action emitted: no",
            f"- next command: `{next_command}`",
            f"- missing: {', '.join(missing) if missing else 'none'}",
            "",
            "Specialist binding:",
            f"- primary specialist: {primary}",
            f"- supporting specialists: {', '.join(supporting) if supporting else 'none'}",
            f"- route proof: `{proof_lanes['proof_target']}`",
            f"- readiness proof: `{proof_lanes['readiness_target']}`",
            f"- handoff proof: `{proof_lanes['handoff_target']}`",
            "",
            "Tool proposal review:",
            f"- proposed tool: {proposed_tool or 'missing'}",
            f"- registered in ToolRegistry: {'yes' if registered_tool else 'no'}",
            f"- toolset: {toolset or 'unknown'}",
            f"- risk level: {risk_level}",
            f"- exact arguments supplied: {'yes' if exact_arguments_supplied else 'no'}",
            f"- verification expectation supplied: {'yes' if verification_supplied else 'no'}",
            f"- approval required before execution: {'yes' if approval_required else 'no'}",
            f"- approval triggers detected: {', '.join(approval_triggers) if approval_triggers else 'none from wording/arguments'}",
            "",
            "Dry-run evidence:",
            f"- arguments: {proposed_arguments or 'missing'}",
            f"- verification: {verification or 'missing'}",
            "",
            "Measured action proposal scorecard:",
            f"- score: {scorecard['score']}/100",
            f"- grade: {scorecard['grade']}",
            f"- registry points: {scorecard['registry_points']}",
            f"- argument points: {scorecard['argument_points']}",
            f"- verification points: {scorecard['verification_points']}",
            f"- permission points: {scorecard['permission_points']}",
            f"- proof lane points: {scorecard['proof_lane_points']}",
            f"- execution lock points: {scorecard['execution_lock_points']}",
            f"- evidence rows ready: {'yes' if scorecard_required_rows_ready else 'no'}",
            f"- action proposal review token sha256: {action_proposal_review_token_sha256}",
            "- action proposal review token authorizes executable action: no",
            "- action proposal review token reusable for next review: no",
            *[
                f"- {row['item']}: {row['points']}/{row['max_points']} points; passed {'yes' if row['passed'] else 'no'}"
                for row in scorecard_rows
            ],
            "",
            "Proof queue:",
            *[f"- `{command}`" for command in proof_queue],
            "",
            "Boundary:",
            "- This packet does not call models, execute tools, approve requests, dismiss approvals, write files, read personal data, control the computer, call external services, queue approvals, or claim the action completed.",
        ]
        return ToolResult(
            "specialist_tool_dry_run_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=request,
                proposed_tool=proposed_tool,
                proposed_arguments=proposed_arguments,
                verification=verification,
                dry_run_state=state,
                dry_run_ready_for_operator_review=state == "DRY_RUN_READY_FOR_OPERATOR_REVIEW",
                tool_executed=False,
                executable_action_emitted=False,
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                can_emit_executable_tool_action=False,
                next_command=next_command,
                missing=missing,
                missing_count=len(missing),
                primary_specialist=primary,
                supporting_specialists=supporting,
                candidate_scores={name: score for score, name in scored},
                registered_tool=registered_tool,
                toolset=toolset,
                risk_level=risk_level,
                exact_arguments_supplied=exact_arguments_supplied,
                verification_expectation_supplied=verification_supplied,
                local_safe_or_read_only=local_safe_or_read_only,
                approval_required_before_execution=approval_required,
                risk_requires_approval=risk_requires_approval,
                approval_triggers=approval_triggers,
                action_proposal_scorecard=scorecard,
                action_proposal_scorecard_rows=scorecard_rows,
                action_proposal_scorecard_row_count=len(scorecard_rows),
                action_proposal_scorecard_shape_ready=scorecard_shape_ready,
                action_proposal_scorecard_required_rows_ready=scorecard_required_rows_ready,
                action_proposal_score=scorecard["score"],
                action_proposal_grade=scorecard["grade"],
                action_proposal_scorecard_max_points=scorecard["max_points"],
                action_proposal_review_token_sha256=action_proposal_review_token_sha256,
                action_proposal_review_token_present=_looks_like_sha256(action_proposal_review_token_sha256),
                action_proposal_review_token_authorizes_model_call=False,
                action_proposal_review_token_authorizes_tool_execution=False,
                action_proposal_review_token_authorizes_approval=False,
                action_proposal_review_token_authorizes_personal_data_read=False,
                action_proposal_review_token_authorizes_external_side_effect=False,
                action_proposal_review_token_authorizes_executable_action=False,
                action_proposal_review_token_reusable_for_next_review=False,
                next_action_proposal_review_requires_new_token=True,
                proof_lane_bound=proof_lane_bound,
                target_model=proof_lanes["target_model"],
                fallback_lane=proof_lanes["fallback_lane"],
                proof_target=proof_lanes["proof_target"],
                handoff_target=proof_lanes["handoff_target"],
                readiness_target=proof_lanes["readiness_target"],
                stop_condition=proof_lanes["stop_condition"],
                action_proposal_contract_command=contract_command,
                verification_command=verification_command,
                receipt_command=receipt_command,
                proof_queue=proof_queue,
                proof_queue_count=len(proof_queue),
                next_proof_command=proof_queue[0],
            ),
        )

    return specialist_tool_dry_run_packet


def make_specialist_proposal_completion_gate_tool(config: JarvisConfig | None, list_tools):
    def specialist_proposal_completion_gate(args: dict[str, Any]) -> ToolResult:
        raw_request = _clean_text(args.get("request") or args.get("proposal") or "", limit=MAX_HANDOFF_REQUEST_CHARS)
        if not raw_request:
            return _missing_specialist_request_result(
                "specialist_proposal_completion_gate",
                "specialist proposal completion gate: summarize memory; tool brain_search; args query=Jarvis harness; verification cites local memory only",
            )

        proposed_tool = _clean_text(
            args.get("tool") or args.get("tool_name") or _proposal_field_from_text(raw_request, ("tool", "tool name", "action tool")),
            limit=120,
        )
        proposed_arguments = _clean_text(
            args.get("arguments") or args.get("args") or _proposal_field_from_text(raw_request, ("args", "arguments", "input", "payload")),
            limit=MAX_PROPOSAL_ARGUMENT_CHARS,
        )
        verification = _clean_text(
            args.get("verification") or args.get("expected") or _proposal_field_from_text(raw_request, ("verification", "expected", "expectation", "success")),
            limit=MAX_PROPOSAL_ARGUMENT_CHARS,
        )
        request = _proposal_request_without_fields(raw_request)

        contract = make_specialist_action_proposal_contract_tool(config, list_tools)(
            {"request": raw_request, "tool": proposed_tool, "arguments": proposed_arguments}
        )
        dry_run = make_specialist_tool_dry_run_packet_tool(config, list_tools)(
            {"request": raw_request, "tool": proposed_tool, "arguments": proposed_arguments, "verification": verification}
        )
        contract_meta = dict(contract.metadata)
        dry_run_meta = dict(dry_run.metadata)

        tool_matches = contract_meta.get("proposed_tool") == dry_run_meta.get("proposed_tool") == proposed_tool
        argument_matches = contract_meta.get("proposed_arguments") == dry_run_meta.get("proposed_arguments") == proposed_arguments
        risk_matches = contract_meta.get("risk_level") == dry_run_meta.get("risk_level")
        approval_matches = contract_meta.get("approval_required_before_execution") == dry_run_meta.get("approval_required_before_execution")
        proof_lanes = _specialist_proof_lanes(request, str(contract_meta.get("primary_specialist") or "planner"), config)

        blockers: list[str] = []
        if contract_meta.get("contract_state") != "READY_FOR_LOCAL_SAFE_DRY_RUN_REVIEW":
            blockers.append("action_contract_not_ready")
        if dry_run_meta.get("dry_run_state") != "DRY_RUN_READY_FOR_OPERATOR_REVIEW":
            blockers.append("dry_run_not_ready_for_operator_review")
        if not tool_matches:
            blockers.append("tool_binding_mismatch")
        if not argument_matches:
            blockers.append("argument_binding_mismatch")
        if not risk_matches:
            blockers.append("risk_binding_mismatch")
        if not approval_matches:
            blockers.append("approval_boundary_mismatch")
        if not dry_run_meta.get("verification_expectation_supplied"):
            blockers.append("verification_expectation_missing")
        if dry_run_meta.get("tool_executed") is not False or dry_run_meta.get("executable_action_emitted") is not False:
            blockers.append("dry_run_claims_execution")
        if contract_meta.get("action_execution_locked") is not True or dry_run_meta.get("executable_action_emitted") is not False:
            blockers.append("execution_lock_not_proven")

        completion_ready = not blockers
        contract_scorecard = contract_meta.get("action_proposal_scorecard") or {}
        dry_run_scorecard = dry_run_meta.get("action_proposal_scorecard") or {}
        contract_scorecard_rows = list(contract_meta.get("action_proposal_scorecard_rows") or [])
        dry_run_scorecard_rows = list(dry_run_meta.get("action_proposal_scorecard_rows") or [])
        score_values = [
            value
            for value in (contract_scorecard.get("score"), dry_run_scorecard.get("score"))
            if isinstance(value, int)
        ]
        combined_score = min(score_values) if score_values else 0
        combined_grade = "ready_for_operator_review" if completion_ready and combined_score >= 85 else ("review_required" if combined_score >= 60 else "held")
        combined_scorecard_rows = [
            {
                "item": "action_contract_scorecard",
                "score": int(contract_scorecard.get("score") or 0),
                "grade": str(contract_scorecard.get("grade") or ""),
                "max_points": 100,
                "row_count": len(contract_scorecard_rows),
                "required_rows_ready": _metadata_bool(contract_meta.get("action_proposal_scorecard_required_rows_ready")),
                "required_before_execution": True,
            },
            {
                "item": "tool_dry_run_scorecard",
                "score": int(dry_run_scorecard.get("score") or 0),
                "grade": str(dry_run_scorecard.get("grade") or ""),
                "max_points": 100,
                "row_count": len(dry_run_scorecard_rows),
                "required_rows_ready": _metadata_bool(dry_run_meta.get("action_proposal_scorecard_required_rows_ready")),
                "required_before_execution": True,
            },
        ]
        for row in combined_scorecard_rows:
            row.update(
                {
                    "authorizes_model_call": False,
                    "authorizes_tool_execution": False,
                    "authorizes_approval": False,
                    "authorizes_personal_data_read": False,
                    "authorizes_external_side_effect": False,
                    "authorizes_executable_action": False,
                    "reusable_for_next_review": False,
                }
            )
        combined_scorecard_shape_ready = _specialist_combined_action_proposal_scorecard_shape_ready(combined_scorecard_rows)
        combined_scorecard_rows_ready = bool(
            completion_ready
            and _specialist_combined_action_proposal_scorecard_shape_ready(combined_scorecard_rows, require_all_passed=True)
        )
        if completion_ready:
            completion_state = "PROPOSAL_COMPLETION_READY_FOR_OPERATOR_REVIEW"
            next_command = f"verification packet: {request}"
        elif contract_meta.get("contract_state") != "READY_FOR_LOCAL_SAFE_DRY_RUN_REVIEW":
            completion_state = "PROPOSAL_COMPLETION_HELD_FOR_ACTION_CONTRACT"
            next_command = f"specialist action proposal contract: {raw_request}"
        else:
            completion_state = "PROPOSAL_COMPLETION_HELD_FOR_DRY_RUN_PROOF"
            next_command = f"specialist tool dry run: {raw_request}"

        action_proposal_review_token_sha256 = _specialist_action_proposal_review_token_sha256(
            request=request,
            proposed_tool=proposed_tool,
            proposed_arguments=proposed_arguments,
            verification=verification,
            primary_specialist=str(dry_run_meta.get("primary_specialist") or contract_meta.get("primary_specialist") or ""),
            target_model=proof_lanes["target_model"],
            risk_level=str(dry_run_meta.get("risk_level") or contract_meta.get("risk_level") or ""),
            review_state=completion_state,
            next_command=next_command,
            scorecard_rows=combined_scorecard_rows,
        )

        audit_recovery_queue = [
            f"runtime trace receipt: specialist_proposal_completion_gate",
            f"execution audit gate: {request}",
            f"verification packet: {request}",
            "verification receipt <operator-reviewed dry-run id>",
            f"execution recovery packet: {request}",
            f"after action learning packet: {request}",
        ]
        proof_queue = list(dict.fromkeys(
            [
                proof_lanes["proof_target"],
                proof_lanes["readiness_target"],
                proof_lanes["handoff_target"],
                f"specialist handoff quality gate: {request}",
                f"specialist proposal gate: {request}",
                f"specialist action proposal contract: {raw_request}",
                f"specialist tool dry run: {raw_request}",
                *audit_recovery_queue,
            ]
        ))

        lines = [
            "Jarvis specialist proposal completion gate:",
            "This is read-only. It is the final specialist draft-to-tool proposal gate before operator review; it does not execute tools, emit executable actions, approve requests, queue approvals, call models, write files, read personal data, control the computer, or claim work completed.",
            "",
            f"Request: {request}",
            "",
            "Completion gate:",
            f"- state: {completion_state}",
            f"- completion ready for operator review: {'yes' if completion_ready else 'no'}",
            "- action execution locked: yes",
            "- executable action emitted: no",
            f"- next command: `{next_command}`",
            f"- blockers: {', '.join(blockers) if blockers else 'none'}",
            "",
            "Bound proposal evidence:",
            f"- proposed tool: {proposed_tool or 'missing'}",
            f"- arguments: {proposed_arguments or 'missing'}",
            f"- verification: {verification or 'missing'}",
            f"- contract state: {contract_meta.get('contract_state')}",
            f"- dry-run state: {dry_run_meta.get('dry_run_state')}",
            f"- registered in ToolRegistry: {'yes' if dry_run_meta.get('registered_tool') else 'no'}",
            f"- risk level: {dry_run_meta.get('risk_level')}",
            f"- approval required before execution: {'yes' if dry_run_meta.get('approval_required_before_execution') else 'no'}",
            f"- local safe/read-only lane: {'yes' if dry_run_meta.get('local_safe_or_read_only') else 'no'}",
            "",
            "Binding checks:",
            f"- tool binding match: {'yes' if tool_matches else 'no'}",
            f"- argument binding match: {'yes' if argument_matches else 'no'}",
            f"- risk binding match: {'yes' if risk_matches else 'no'}",
            f"- approval boundary match: {'yes' if approval_matches else 'no'}",
            f"- verification expectation supplied: {'yes' if dry_run_meta.get('verification_expectation_supplied') else 'no'}",
            f"- dry-run executed tool: {'yes' if dry_run_meta.get('tool_executed') else 'no'}",
            f"- dry-run emitted executable action: {'yes' if dry_run_meta.get('executable_action_emitted') else 'no'}",
            "",
            "Measured action proposal scorecard:",
            f"- combined score: {combined_score}/100",
            f"- combined grade: {combined_grade}",
            f"- contract score: {contract_scorecard.get('score', 0)}/100",
            f"- dry-run score: {dry_run_scorecard.get('score', 0)}/100",
            f"- measurement source: action proposal contract + specialist tool dry-run packet",
            f"- combined evidence rows ready: {'yes' if combined_scorecard_rows_ready else 'no'}",
            f"- action proposal review token sha256: {action_proposal_review_token_sha256}",
            "- action proposal review token authorizes executable action: no",
            "- action proposal review token reusable for next review: no",
            *[
                f"- {row['item']}: score {row['score']}/100; grade {row['grade'] or 'unknown'}; rows {row['row_count']}; ready {'yes' if row['required_rows_ready'] else 'no'}"
                for row in combined_scorecard_rows
            ],
            "",
            "Audit, recovery, and learning handoff:",
            *[f"- `{command}`" for command in audit_recovery_queue],
            "",
            "Required proof queue:",
            *[f"- `{command}`" for command in proof_queue],
            "",
            "Boundary:",
            "- A ready completion gate is still not execution permission. It only says the proposal is coherent enough for the operator/operator review.",
            "- Real execution still requires the normal runtime path, ToolRegistry, PermissionPolicy, approval gates when needed, execution receipt, verification receipt, audit, recovery, and learning evidence.",
        ]

        return ToolResult(
            "specialist_proposal_completion_gate",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=request,
                raw_request=raw_request,
                completion_state=completion_state,
                completion_ready_for_operator_review=completion_ready,
                action_execution_locked=True,
                executable_action_emitted=False,
                can_emit_executable_tool_action=False,
                tool_executed=False,
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                next_command=next_command,
                blockers=blockers,
                blocker_count=len(blockers),
                proposed_tool=proposed_tool,
                proposed_arguments=proposed_arguments,
                verification=verification,
                contract_state=contract_meta.get("contract_state"),
                dry_run_state=dry_run_meta.get("dry_run_state"),
                registered_tool=dry_run_meta.get("registered_tool"),
                toolset=dry_run_meta.get("toolset"),
                risk_level=dry_run_meta.get("risk_level"),
                exact_arguments_supplied=dry_run_meta.get("exact_arguments_supplied"),
                verification_expectation_supplied=dry_run_meta.get("verification_expectation_supplied"),
                local_safe_or_read_only=dry_run_meta.get("local_safe_or_read_only"),
                approval_required_before_execution=dry_run_meta.get("approval_required_before_execution"),
                approval_triggers=dry_run_meta.get("approval_triggers") or [],
                action_proposal_scorecard={
                    "score": combined_score,
                    "grade": combined_grade,
                    "max_points": 100,
                    "contract_score": contract_scorecard.get("score", 0),
                    "contract_grade": contract_scorecard.get("grade", ""),
                    "dry_run_score": dry_run_scorecard.get("score", 0),
                    "dry_run_grade": dry_run_scorecard.get("grade", ""),
                },
                action_proposal_scorecard_rows=combined_scorecard_rows,
                action_proposal_scorecard_row_count=len(combined_scorecard_rows),
                action_proposal_scorecard_shape_ready=combined_scorecard_shape_ready,
                action_proposal_scorecard_required_rows_ready=combined_scorecard_rows_ready,
                contract_action_proposal_scorecard_rows=contract_scorecard_rows,
                contract_action_proposal_scorecard_row_count=len(contract_scorecard_rows),
                dry_run_action_proposal_scorecard_rows=dry_run_scorecard_rows,
                dry_run_action_proposal_scorecard_row_count=len(dry_run_scorecard_rows),
                action_proposal_score=combined_score,
                action_proposal_grade=combined_grade,
                action_proposal_scorecard_max_points=100,
                action_proposal_review_token_sha256=action_proposal_review_token_sha256,
                action_proposal_review_token_present=_looks_like_sha256(action_proposal_review_token_sha256),
                action_proposal_review_token_authorizes_model_call=False,
                action_proposal_review_token_authorizes_tool_execution=False,
                action_proposal_review_token_authorizes_approval=False,
                action_proposal_review_token_authorizes_personal_data_read=False,
                action_proposal_review_token_authorizes_external_side_effect=False,
                action_proposal_review_token_authorizes_executable_action=False,
                action_proposal_review_token_reusable_for_next_review=False,
                next_action_proposal_review_requires_new_token=True,
                contract_action_proposal_score=contract_scorecard.get("score", 0),
                dry_run_action_proposal_score=dry_run_scorecard.get("score", 0),
                tool_binding_match=tool_matches,
                argument_binding_match=argument_matches,
                risk_binding_match=risk_matches,
                approval_boundary_match=approval_matches,
                primary_specialist=dry_run_meta.get("primary_specialist") or contract_meta.get("primary_specialist"),
                supporting_specialists=dry_run_meta.get("supporting_specialists") or [],
                target_model=proof_lanes["target_model"],
                fallback_lane=proof_lanes["fallback_lane"],
                proof_target=proof_lanes["proof_target"],
                handoff_target=proof_lanes["handoff_target"],
                readiness_target=proof_lanes["readiness_target"],
                stop_condition=proof_lanes["stop_condition"],
                action_contract_command=f"specialist action proposal contract: {raw_request}",
                dry_run_command=f"specialist tool dry run: {raw_request}",
                audit_recovery_queue=audit_recovery_queue,
                audit_recovery_queue_count=len(audit_recovery_queue),
                proof_queue=proof_queue,
                proof_queue_count=len(proof_queue),
                next_proof_command=proof_queue[0] if proof_queue else "",
            ),
        )

    return specialist_proposal_completion_gate


def make_specialist_execution_handoff_packet_tool(config: JarvisConfig | None, list_tools):
    def specialist_execution_handoff_packet(args: dict[str, Any]) -> ToolResult:
        raw_request = _clean_text(args.get("request") or args.get("proposal") or "", limit=MAX_HANDOFF_REQUEST_CHARS)
        if not raw_request:
            return _missing_specialist_request_result(
                "specialist_execution_handoff_packet",
                "specialist execution handoff: summarize memory; tool brain_search; args query=Jarvis harness; verification cites local memory only",
            )

        proposed_tool = _clean_text(
            args.get("tool") or args.get("tool_name") or _proposal_field_from_text(raw_request, ("tool", "tool name", "action tool")),
            limit=120,
        )
        proposed_arguments = _clean_text(
            args.get("arguments") or args.get("args") or _proposal_field_from_text(raw_request, ("args", "arguments", "input", "payload")),
            limit=MAX_PROPOSAL_ARGUMENT_CHARS,
        )
        verification = _clean_text(
            args.get("verification") or args.get("expected") or _proposal_field_from_text(raw_request, ("verification", "expected", "expectation", "success")),
            limit=MAX_PROPOSAL_ARGUMENT_CHARS,
        )
        request = _proposal_request_without_fields(raw_request)

        completion = make_specialist_proposal_completion_gate_tool(config, list_tools)(
            {"request": raw_request, "tool": proposed_tool, "arguments": proposed_arguments, "verification": verification}
        )
        completion_meta = dict(completion.metadata)
        tools = list_tools()
        tool_by_name = {tool.name: tool for tool in tools}
        registry_match = tool_by_name.get(proposed_tool) if proposed_tool else None
        risk_level = registry_match.risk.name if registry_match else "UNKNOWN"
        toolset = registry_match.toolset if registry_match else ""
        local_safe_or_read_only = risk_level in {"READ_ONLY", "LOCAL_SAFE"}
        approval_required = _metadata_bool(completion_meta.get("approval_required_before_execution"))
        completion_ready = _metadata_bool(completion_meta.get("completion_ready_for_operator_review"))
        exact_arguments_supplied = _metadata_bool(completion_meta.get("exact_arguments_supplied"))
        verification_supplied = _metadata_bool(completion_meta.get("verification_expectation_supplied"))

        blockers: list[str] = []
        if not completion_ready:
            blockers.append("proposal_completion_not_ready")
        if not registry_match:
            blockers.append("registered_tool_missing")
        if not exact_arguments_supplied:
            blockers.append("exact_arguments_missing")
        if not verification_supplied:
            blockers.append("verification_expectation_missing")
        if approval_required:
            blockers.append("approval_boundary_required")
        if not local_safe_or_read_only:
            blockers.append("permission_policy_not_local_safe")

        if not completion_ready:
            handoff_state = "SPECIALIST_EXECUTION_HANDOFF_HELD_FOR_COMPLETION_GATE"
            next_command = f"specialist proposal completion gate: {raw_request}"
        elif approval_required or not local_safe_or_read_only:
            handoff_state = "SPECIALIST_EXECUTION_HANDOFF_HELD_FOR_APPROVAL_BOUNDARY"
            next_command = "approval readiness latest"
        else:
            handoff_state = "SPECIALIST_EXECUTION_HANDOFF_READY_FOR_RUNTIME_REVIEW"
            next_command = f"execution contract: {request}"
        specialist_review_token_sha256 = _specialist_review_token_sha256(
            request=request,
            proposed_tool=proposed_tool,
            proposed_arguments=proposed_arguments,
            verification=verification,
            primary_specialist=str(completion_meta.get("primary_specialist") or ""),
            target_model=str(completion_meta.get("target_model") or ""),
            completion_state=str(completion_meta.get("completion_state") or ""),
        )
        runtime_review_contract_rows = _specialist_runtime_review_contract_rows(
            handoff_state=handoff_state,
            specialist_review_token_sha256=specialist_review_token_sha256,
            proposed_tool=proposed_tool,
            risk_level=risk_level,
        )
        runtime_review_contract_ready = _specialist_runtime_review_contract_ready(
            runtime_review_contract_rows,
            handoff_state=handoff_state,
            specialist_review_token_sha256=specialist_review_token_sha256,
            proposed_tool=proposed_tool,
            risk_level=risk_level,
        )
        runtime_review_contract_summary = [
            "normal-runtime-review-only",
            "model-call-not-authorized",
            "tool-execution-not-authorized",
            "approval-not-authorized",
            "post-run-proof-still-required",
            "fresh-specialist-review-token-required",
        ]
        runtime_review_boundary_token_sha256 = _specialist_runtime_review_boundary_token_sha256(
            request=request,
            handoff_state=handoff_state,
            next_command=next_command,
            specialist_review_token_sha256=specialist_review_token_sha256,
            runtime_review_contract_rows=runtime_review_contract_rows,
        )

        pre_run_proof_queue = list(dict.fromkeys(
            [
                *list(completion_meta.get("proof_queue") or []),
                f"specialist proposal completion gate: {raw_request}",
                f"execution contract: {request}",
                f"dispatch decision packet: {request}",
                f"argument contract: {request}",
                f"risk preflight: {request}",
                f"verification packet: {request}",
            ]
        ))
        post_run_proof_queue = [
            "runtime trace receipt: <specialist proposed action run>",
            "verification receipt <specialist proposed action run>",
            f"execution audit gate: {request}",
            f"execution recovery packet: {request}",
            f"after action learning packet: {request}",
            f"completion claim gate: specialist execution handoff {request}",
        ]
        route_to_runtime_contract_token_sha256 = _specialist_route_to_runtime_contract_token_sha256(
            request=request,
            handoff_state=handoff_state,
            proposed_tool=proposed_tool,
            proposed_arguments=proposed_arguments,
            verification=verification,
            primary_specialist=str(completion_meta.get("primary_specialist") or ""),
            target_model=str(completion_meta.get("target_model") or ""),
            risk_level=risk_level,
            completion_state=str(completion_meta.get("completion_state") or ""),
            specialist_review_token_sha256=specialist_review_token_sha256,
            action_proposal_review_token_sha256=str(completion_meta.get("action_proposal_review_token_sha256") or ""),
            runtime_review_boundary_token_sha256=runtime_review_boundary_token_sha256,
            runtime_review_contract_rows=runtime_review_contract_rows,
            pre_run_proof_queue=pre_run_proof_queue,
            post_run_proof_queue=post_run_proof_queue,
            next_command=next_command,
        )

        lines = [
            "Jarvis specialist execution handoff packet:",
            "This is read-only. It packages a specialist proposal for the normal runtime review path without executing the tool, approving requests, queueing approvals, calling models, writing files, reading personal data, controlling the computer, or claiming the action completed.",
            "",
            f"Request: {request}",
            "",
            "Execution handoff:",
            f"- state: {handoff_state}",
            f"- ready for runtime review: {'yes' if handoff_state == 'SPECIALIST_EXECUTION_HANDOFF_READY_FOR_RUNTIME_REVIEW' else 'no'}",
            "- tool executed: no",
            "- executable action emitted: no",
            "- approval queued: no",
            f"- next command: `{next_command}`",
            f"- blockers: {', '.join(blockers) if blockers else 'none'}",
            "",
            "Bound specialist proposal:",
            f"- completion state: {completion_meta.get('completion_state')}",
            f"- completion ready for operator review: {'yes' if completion_ready else 'no'}",
            f"- proposed tool: {proposed_tool or 'missing'}",
            f"- arguments: {proposed_arguments or 'missing'}",
            f"- verification: {verification or 'missing'}",
            f"- registered in ToolRegistry: {'yes' if registry_match else 'no'}",
            f"- toolset: {toolset or 'unknown'}",
            f"- risk level: {risk_level}",
            f"- local safe/read-only lane: {'yes' if local_safe_or_read_only else 'no'}",
            f"- approval required before execution: {'yes' if approval_required else 'no'}",
            f"- specialist review token sha256: {specialist_review_token_sha256 or 'missing'}",
            "- specialist review token reusable for next review: no",
            "",
            "Runtime boundary:",
            "- This handoff can only move into the normal runtime review path.",
            "- ToolRegistry, PermissionPolicy, dispatch, argument contract, risk preflight, verification packet, execution receipt, audit, recovery, and learning evidence still have to pass.",
            "- Shell/code execution, computer control, personal data, external side effects, destructive work, and ambiguous actions remain approval-gated.",
            "",
            "Runtime review contract:",
            f"- contract ready: {'yes' if runtime_review_contract_ready else 'no'}",
            f"- contract rows: {len(runtime_review_contract_rows)}",
            "- contract summary: " + ", ".join(runtime_review_contract_summary),
            "- authorizes model call: no",
            "- authorizes tool execution: no",
            "- authorizes approval: no",
            "- authorizes personal-data read: no",
            "- authorizes external side effect: no",
            "- authorizes completion claim: no",
            "- bypasses post-run proof: no",
            "- reusable for next specialist review: no",
            f"- runtime review boundary token sha256: {runtime_review_boundary_token_sha256}",
            "- runtime review boundary token authorizes model call: no",
            "- runtime review boundary token authorizes tool execution: no",
            "- runtime review boundary token authorizes approval: no",
            "- runtime review boundary token authorizes completion claim: no",
            "- runtime review boundary token reusable for next review: no",
            f"- route-to-runtime contract token sha256: {route_to_runtime_contract_token_sha256}",
            "- route-to-runtime contract token authorizes model call: no",
            "- route-to-runtime contract token authorizes tool execution: no",
            "- route-to-runtime contract token authorizes approval: no",
            "- route-to-runtime contract token authorizes completion claim: no",
            "- route-to-runtime contract token reusable for next review: no",
            f"- action proposal review token sha256: {completion_meta.get('action_proposal_review_token_sha256') or 'missing'}",
            "- route-to-runtime contract binds action proposal review token: yes",
            "",
            "Pre-run proof queue:",
            *[f"- `{command}`" for command in pre_run_proof_queue],
            "",
            "Post-run proof queue:",
            *[f"- `{command}`" for command in post_run_proof_queue],
        ]

        return ToolResult(
            "specialist_execution_handoff_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=request,
                raw_request=raw_request,
                handoff_state=handoff_state,
                ready_for_runtime_review=handoff_state == "SPECIALIST_EXECUTION_HANDOFF_READY_FOR_RUNTIME_REVIEW",
                tool_executed=False,
                executable_action_emitted=False,
                can_emit_executable_tool_action=False,
                queues_approval=False,
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                next_command=next_command,
                blockers=blockers,
                blocker_count=len(blockers),
                completion_state=completion_meta.get("completion_state"),
                completion_ready_for_operator_review=completion_ready,
                proposed_tool=proposed_tool,
                proposed_arguments=proposed_arguments,
                verification=verification,
                registered_tool=bool(registry_match),
                toolset=toolset,
                risk_level=risk_level,
                exact_arguments_supplied=exact_arguments_supplied,
                verification_expectation_supplied=verification_supplied,
                local_safe_or_read_only=local_safe_or_read_only,
                approval_required_before_execution=approval_required,
                approval_triggers=completion_meta.get("approval_triggers") or [],
                primary_specialist=completion_meta.get("primary_specialist"),
                supporting_specialists=completion_meta.get("supporting_specialists") or [],
                target_model=completion_meta.get("target_model"),
                specialist_review_token_sha256=specialist_review_token_sha256,
                specialist_review_token_present=_looks_like_sha256(specialist_review_token_sha256),
                specialist_review_token_reusable_for_next_review=False,
                next_specialist_review_requires_new_token=True,
                action_proposal_review_token_sha256=completion_meta.get("action_proposal_review_token_sha256", ""),
                action_proposal_review_token_present=completion_meta.get("action_proposal_review_token_present", False),
                runtime_review_contract_rows=runtime_review_contract_rows,
                runtime_review_contract_row_count=len(runtime_review_contract_rows),
                runtime_review_contract_ready=runtime_review_contract_ready,
                runtime_review_contract_summary=runtime_review_contract_summary,
                runtime_review_boundary_token_sha256=runtime_review_boundary_token_sha256,
                runtime_review_boundary_token_present=_looks_like_sha256(runtime_review_boundary_token_sha256),
                runtime_review_boundary_token_authorizes_model_call=False,
                runtime_review_boundary_token_authorizes_tool_execution=False,
                runtime_review_boundary_token_authorizes_approval=False,
                runtime_review_boundary_token_authorizes_personal_data_read=False,
                runtime_review_boundary_token_authorizes_external_side_effect=False,
                runtime_review_boundary_token_authorizes_completion_claim=False,
                runtime_review_boundary_token_bypasses_post_run_proof=False,
                runtime_review_boundary_token_reusable_for_next_specialist_review=False,
                next_specialist_review_requires_new_runtime_review_boundary_token=True,
                route_to_runtime_contract_token_sha256=route_to_runtime_contract_token_sha256,
                route_to_runtime_contract_token_present=_looks_like_sha256(route_to_runtime_contract_token_sha256),
                route_to_runtime_contract_token_authorizes_model_call=False,
                route_to_runtime_contract_token_authorizes_tool_execution=False,
                route_to_runtime_contract_token_authorizes_approval=False,
                route_to_runtime_contract_token_authorizes_personal_data_read=False,
                route_to_runtime_contract_token_authorizes_external_side_effect=False,
                route_to_runtime_contract_token_authorizes_completion_claim=False,
                route_to_runtime_contract_token_reusable_for_next_specialist_review=False,
                route_to_runtime_contract_binds_action_proposal_review_token=True,
                next_specialist_review_requires_new_route_to_runtime_contract_token=True,
                runtime_review_authorizes_model_call=False,
                runtime_review_authorizes_tool_execution=False,
                runtime_review_authorizes_approval=False,
                runtime_review_authorizes_personal_data_read=False,
                runtime_review_authorizes_external_side_effect=False,
                runtime_review_authorizes_completion_claim=False,
                runtime_review_bypasses_post_run_proof=False,
                runtime_review_reusable_for_next_specialist_review=False,
                fallback_lane=completion_meta.get("fallback_lane"),
                proof_target=completion_meta.get("proof_target"),
                handoff_target=completion_meta.get("handoff_target"),
                readiness_target=completion_meta.get("readiness_target"),
                stop_condition=completion_meta.get("stop_condition"),
                pre_run_proof_queue=pre_run_proof_queue,
                pre_run_proof_queue_count=len(pre_run_proof_queue),
                post_run_proof_queue=post_run_proof_queue,
                post_run_proof_queue_count=len(post_run_proof_queue),
                next_proof_command=pre_run_proof_queue[0] if pre_run_proof_queue else "",
                post_run_next_proof_command=post_run_proof_queue[0],
            ),
        )

    return specialist_execution_handoff_packet


def make_specialist_post_run_closure_packet_tool(config: JarvisConfig | None, list_tools):
    def specialist_post_run_closure_packet(args: dict[str, Any]) -> ToolResult:
        raw_request = _clean_text(args.get("request") or args.get("proposal") or "", limit=MAX_HANDOFF_REQUEST_CHARS)
        if not raw_request:
            return _missing_specialist_request_result(
                "specialist_post_run_closure_packet",
                "specialist post-run closure: summarize memory; tool brain_search; args query=Jarvis harness; verification cites local memory only; runtime_trace=...; verification_receipt=...; audit=...; recovery=...; learning=...",
            )

        proposed_tool = _clean_text(
            args.get("tool") or args.get("tool_name") or _proposal_field_from_text(raw_request, ("tool", "tool name", "action tool")),
            limit=120,
        )
        proposed_arguments = _clean_text(
            args.get("arguments") or args.get("args") or _proposal_field_from_text(raw_request, ("args", "arguments", "input", "payload")),
            limit=MAX_PROPOSAL_ARGUMENT_CHARS,
        )
        verification = _clean_text(
            args.get("verification") or args.get("expected") or _proposal_field_from_text(raw_request, ("verification", "expected", "expectation", "success")),
            limit=MAX_PROPOSAL_ARGUMENT_CHARS,
        )
        runtime_trace = _clean_text(args.get("runtime_trace") or args.get("runtime_trace_receipt") or args.get("trace") or "", limit=MAX_PROPOSAL_ARGUMENT_CHARS)
        verification_receipt = _clean_text(args.get("verification_receipt") or args.get("receipt") or "", limit=MAX_PROPOSAL_ARGUMENT_CHARS)
        execution_audit = _clean_text(args.get("execution_audit") or args.get("audit") or "", limit=MAX_PROPOSAL_ARGUMENT_CHARS)
        execution_recovery = _clean_text(args.get("execution_recovery") or args.get("recovery") or "", limit=MAX_PROPOSAL_ARGUMENT_CHARS)
        after_action_learning = _clean_text(args.get("after_action_learning") or args.get("learning") or "", limit=MAX_PROPOSAL_ARGUMENT_CHARS)
        completion_claim = _clean_text(args.get("completion_claim") or args.get("claim") or "", limit=MAX_PROPOSAL_ARGUMENT_CHARS)
        runtime_trace_sha256 = _clean_text(
            args.get("runtime_trace_sha256")
            or args.get("runtime_trace_hash")
            or _proposal_field_from_text(raw_request, ("runtime_trace_sha256", "runtime trace sha256", "runtime trace hash")),
            limit=80,
        )
        verification_receipt_sha256 = _clean_text(
            args.get("verification_receipt_sha256")
            or args.get("verification_receipt_hash")
            or args.get("receipt_sha256")
            or _proposal_field_from_text(raw_request, ("verification_receipt_sha256", "verification receipt sha256", "receipt sha256")),
            limit=80,
        )
        execution_audit_sha256 = _clean_text(
            args.get("execution_audit_sha256")
            or args.get("audit_sha256")
            or args.get("audit_hash")
            or _proposal_field_from_text(raw_request, ("execution_audit_sha256", "audit_sha256", "audit sha256")),
            limit=80,
        )
        execution_recovery_sha256 = _clean_text(
            args.get("execution_recovery_sha256")
            or args.get("recovery_sha256")
            or args.get("recovery_hash")
            or _proposal_field_from_text(raw_request, ("execution_recovery_sha256", "recovery_sha256", "recovery sha256")),
            limit=80,
        )
        after_action_learning_sha256 = _clean_text(
            args.get("after_action_learning_sha256")
            or args.get("learning_sha256")
            or args.get("learning_hash")
            or _proposal_field_from_text(raw_request, ("after_action_learning_sha256", "learning_sha256", "learning sha256")),
            limit=80,
        )
        completion_claim_sha256 = _clean_text(
            args.get("completion_claim_sha256")
            or args.get("claim_sha256")
            or args.get("claim_hash")
            or _proposal_field_from_text(raw_request, ("completion_claim_sha256", "claim_sha256", "claim sha256")),
            limit=80,
        )
        blockers_text = _clean_text(args.get("blockers") or "none", limit=MAX_PROPOSAL_ARGUMENT_CHARS)
        request = _proposal_request_without_fields(raw_request)

        handoff = make_specialist_execution_handoff_packet_tool(config, list_tools)(
            {"request": raw_request, "tool": proposed_tool, "arguments": proposed_arguments, "verification": verification}
        )
        handoff_meta = dict(handoff.metadata)
        runtime_trace_ready = _evidence_present(runtime_trace)
        verification_receipt_ready = _evidence_present(verification_receipt)
        execution_audit_ready = _evidence_present(execution_audit)
        execution_recovery_ready = _evidence_present(execution_recovery)
        after_action_learning_ready = _evidence_present(after_action_learning)
        completion_claim_ready = _evidence_present(completion_claim)
        blockers_cleared = blockers_text.strip().lower() in {"", "none", "none reported", "no blockers", "clear", "cleared"}
        runtime_trace_hash_present = _looks_like_sha256(runtime_trace_sha256)
        verification_receipt_hash_present = _looks_like_sha256(verification_receipt_sha256)
        execution_audit_hash_present = _looks_like_sha256(execution_audit_sha256)
        execution_recovery_hash_present = _looks_like_sha256(execution_recovery_sha256)
        after_action_learning_hash_present = _looks_like_sha256(after_action_learning_sha256)
        completion_claim_hash_present = _looks_like_sha256(completion_claim_sha256)
        post_run_artifact_hashes_present = all(
            [
                runtime_trace_hash_present,
                verification_receipt_hash_present,
                execution_audit_hash_present,
                execution_recovery_hash_present,
                after_action_learning_hash_present,
                completion_claim_hash_present,
            ]
        )

        missing: list[str] = []
        if handoff_meta.get("handoff_state") != "SPECIALIST_EXECUTION_HANDOFF_READY_FOR_RUNTIME_REVIEW":
            missing.append("ready specialist execution handoff")
        if not runtime_trace_ready:
            missing.append("runtime trace receipt")
        if not verification_receipt_ready:
            missing.append("verification receipt")
        if not execution_audit_ready:
            missing.append("execution audit")
        if not execution_recovery_ready:
            missing.append("execution recovery packet")
        if not after_action_learning_ready:
            missing.append("after-action learning packet")
        if not completion_claim_ready:
            missing.append("completion claim gate")
        if not runtime_trace_hash_present:
            missing.append("runtime_trace_sha256")
        if not verification_receipt_hash_present:
            missing.append("verification_receipt_sha256")
        if not execution_audit_hash_present:
            missing.append("execution_audit_sha256")
        if not execution_recovery_hash_present:
            missing.append("execution_recovery_sha256")
        if not after_action_learning_hash_present:
            missing.append("after_action_learning_sha256")
        if not completion_claim_hash_present:
            missing.append("completion_claim_sha256")
        if not blockers_cleared:
            missing.append("remaining blockers cleared")

        closure_ready = not missing
        closure_state = "SPECIALIST_POST_RUN_CLOSURE_READY" if closure_ready else "SPECIALIST_POST_RUN_CLOSURE_HELD"
        next_command = (
            f"completion claim gate: specialist execution handoff {request}"
            if closure_ready
            else "specialist post-run closure: <request>; runtime_trace=<receipt>; verification_receipt=<receipt>; audit=<audit>; recovery=<recovery>; learning=<learning>; completion_claim=<claim>"
        )
        required_commands = [
            "specialist execution handoff: " + raw_request,
            "runtime trace receipt: <specialist proposed action run>",
            "verification receipt <specialist proposed action run>",
            f"execution audit gate: {request}",
            f"execution recovery packet: {request}",
            f"after action learning packet: {request}",
            f"completion claim gate: specialist execution handoff {request}",
        ]
        specialist_post_run_closure_token_sha256 = _specialist_post_run_closure_token_sha256(
            request=request,
            raw_request=raw_request,
            closure_state=closure_state,
            missing=missing,
            required_commands=required_commands,
            specialist_review_token_sha256=str(handoff_meta.get("specialist_review_token_sha256") or ""),
            runtime_review_boundary_token_sha256=str(handoff_meta.get("runtime_review_boundary_token_sha256") or ""),
            route_to_runtime_contract_token_sha256=str(handoff_meta.get("route_to_runtime_contract_token_sha256") or ""),
            runtime_trace_sha256=runtime_trace_sha256,
            verification_receipt_sha256=verification_receipt_sha256,
            execution_audit_sha256=execution_audit_sha256,
            execution_recovery_sha256=execution_recovery_sha256,
            after_action_learning_sha256=after_action_learning_sha256,
            completion_claim_sha256=completion_claim_sha256,
            post_run_artifact_hashes_present=post_run_artifact_hashes_present,
            blockers_cleared=blockers_cleared,
            next_command=next_command,
        )

        lines = [
            "Jarvis specialist post-run closure packet:",
            "This is read-only. It proves a specialist proposal that entered the normal runtime path has post-run trace, verification, audit, recovery, learning, and completion-claim evidence before the specialist route can be counted as closed.",
            "",
            f"Request: {request}",
            "",
            "Closure state:",
            f"- state: {closure_state}",
            f"- ready to count specialist execution closed: {'yes' if closure_ready else 'no'}",
            "- action execution allowed now: no",
            "- executable action emitted: no",
            f"- next command: `{next_command}`",
            f"- missing: {', '.join(missing) if missing else 'none'}",
            "",
            "Bound specialist handoff:",
            f"- handoff state: {handoff_meta.get('handoff_state')}",
            f"- ready for runtime review: {'yes' if handoff_meta.get('ready_for_runtime_review') else 'no'}",
            f"- proposed tool: {proposed_tool or 'missing'}",
            f"- arguments: {proposed_arguments or 'missing'}",
            f"- verification expectation: {verification or 'missing'}",
            f"- risk level: {handoff_meta.get('risk_level')}",
            f"- specialist review token sha256: {handoff_meta.get('specialist_review_token_sha256') or 'missing'}",
            "- specialist review token reusable for next review: no",
            f"- runtime review boundary token sha256: {handoff_meta.get('runtime_review_boundary_token_sha256') or 'missing'}",
            "- runtime review boundary token reusable for next review: no",
            f"- route-to-runtime contract token sha256: {handoff_meta.get('route_to_runtime_contract_token_sha256') or 'missing'}",
            "- route-to-runtime contract token reusable for next review: no",
            "",
            "Post-run evidence:",
            f"- runtime trace receipt: {runtime_trace or 'missing'}",
            f"- runtime trace sha256: {runtime_trace_sha256 or 'missing'}",
            f"- verification receipt: {verification_receipt or 'missing'}",
            f"- verification receipt sha256: {verification_receipt_sha256 or 'missing'}",
            f"- execution audit: {execution_audit or 'missing'}",
            f"- execution audit sha256: {execution_audit_sha256 or 'missing'}",
            f"- execution recovery: {execution_recovery or 'missing'}",
            f"- execution recovery sha256: {execution_recovery_sha256 or 'missing'}",
            f"- after-action learning: {after_action_learning or 'missing'}",
            f"- after-action learning sha256: {after_action_learning_sha256 or 'missing'}",
            f"- completion claim: {completion_claim or 'missing'}",
            f"- completion claim sha256: {completion_claim_sha256 or 'missing'}",
            f"- post-run artifact hashes present: {'yes' if post_run_artifact_hashes_present else 'no'}",
            f"- blockers: {blockers_text or 'none'}",
            f"- specialist post-run closure token sha256: {specialist_post_run_closure_token_sha256}",
            "- specialist post-run closure token authorizes model call: no",
            "- specialist post-run closure token authorizes tool execution: no",
            "- specialist post-run closure token authorizes approval: no",
            "- specialist post-run closure token authorizes completion claim: no",
            "- specialist post-run closure token reusable for next specialist review: no",
            "",
            "Required closure commands:",
            *[f"- `{command}`" for command in required_commands],
            "",
            "Boundary:",
            "- This packet does not call models, execute tools, approve requests, dismiss approvals, write files, read personal data, control the computer, call external services, queue approvals, or claim AGI completion.",
            "- It only closes the proof loop after the normal runtime path has independently produced receipts.",
        ]

        return ToolResult(
            "specialist_post_run_closure_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=request,
                raw_request=raw_request,
                closure_state=closure_state,
                ready_to_count_specialist_execution_closed=closure_ready,
                action_allowed_now=False,
                executable_action_emitted=False,
                can_emit_executable_tool_action=False,
                can_auto_execute_now=False,
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                next_command=next_command,
                missing=missing,
                missing_count=len(missing),
                proposed_tool=proposed_tool,
                proposed_arguments=proposed_arguments,
                verification=verification,
                handoff_state=handoff_meta.get("handoff_state"),
                handoff_ready_for_runtime_review=handoff_meta.get("ready_for_runtime_review"),
                handoff_blockers=handoff_meta.get("blockers", []),
                registered_tool=handoff_meta.get("registered_tool"),
                toolset=handoff_meta.get("toolset"),
                risk_level=handoff_meta.get("risk_level"),
                primary_specialist=handoff_meta.get("primary_specialist"),
                supporting_specialists=handoff_meta.get("supporting_specialists") or [],
                target_model=handoff_meta.get("target_model"),
                specialist_review_token_sha256=handoff_meta.get("specialist_review_token_sha256", ""),
                specialist_review_token_present=handoff_meta.get("specialist_review_token_present", False),
                specialist_review_token_reusable_for_next_review=False,
                next_specialist_review_requires_new_token=True,
                runtime_review_boundary_token_sha256=handoff_meta.get("runtime_review_boundary_token_sha256", ""),
                runtime_review_boundary_token_present=handoff_meta.get("runtime_review_boundary_token_present", False),
                runtime_review_boundary_token_authorizes_model_call=False,
                runtime_review_boundary_token_authorizes_tool_execution=False,
                runtime_review_boundary_token_authorizes_approval=False,
                runtime_review_boundary_token_authorizes_personal_data_read=False,
                runtime_review_boundary_token_authorizes_external_side_effect=False,
                runtime_review_boundary_token_authorizes_completion_claim=False,
                runtime_review_boundary_token_bypasses_post_run_proof=False,
                runtime_review_boundary_token_reusable_for_next_specialist_review=False,
                next_specialist_review_requires_new_runtime_review_boundary_token=True,
                route_to_runtime_contract_token_sha256=handoff_meta.get("route_to_runtime_contract_token_sha256", ""),
                route_to_runtime_contract_token_present=handoff_meta.get("route_to_runtime_contract_token_present", False),
                route_to_runtime_contract_token_authorizes_model_call=False,
                route_to_runtime_contract_token_authorizes_tool_execution=False,
                route_to_runtime_contract_token_authorizes_approval=False,
                route_to_runtime_contract_token_authorizes_personal_data_read=False,
                route_to_runtime_contract_token_authorizes_external_side_effect=False,
                route_to_runtime_contract_token_authorizes_completion_claim=False,
                route_to_runtime_contract_token_reusable_for_next_specialist_review=False,
                next_specialist_review_requires_new_route_to_runtime_contract_token=True,
                fallback_lane=handoff_meta.get("fallback_lane"),
                proof_target=handoff_meta.get("proof_target"),
                handoff_target=handoff_meta.get("handoff_target"),
                readiness_target=handoff_meta.get("readiness_target"),
                stop_condition=handoff_meta.get("stop_condition"),
                runtime_trace_receipt=runtime_trace,
                runtime_trace_receipt_ready=runtime_trace_ready,
                runtime_trace_sha256=runtime_trace_sha256,
                runtime_trace_hash_present=runtime_trace_hash_present,
                verification_receipt=verification_receipt,
                verification_receipt_ready=verification_receipt_ready,
                verification_receipt_sha256=verification_receipt_sha256,
                verification_receipt_hash_present=verification_receipt_hash_present,
                execution_audit=execution_audit,
                execution_audit_ready=execution_audit_ready,
                execution_audit_sha256=execution_audit_sha256,
                execution_audit_hash_present=execution_audit_hash_present,
                execution_recovery=execution_recovery,
                execution_recovery_ready=execution_recovery_ready,
                execution_recovery_sha256=execution_recovery_sha256,
                execution_recovery_hash_present=execution_recovery_hash_present,
                after_action_learning=after_action_learning,
                after_action_learning_ready=after_action_learning_ready,
                after_action_learning_sha256=after_action_learning_sha256,
                after_action_learning_hash_present=after_action_learning_hash_present,
                completion_claim=completion_claim,
                completion_claim_ready=completion_claim_ready,
                completion_claim_sha256=completion_claim_sha256,
                completion_claim_hash_present=completion_claim_hash_present,
                post_run_artifact_hashes_present=post_run_artifact_hashes_present,
                specialist_post_run_closure_token_sha256=specialist_post_run_closure_token_sha256,
                specialist_post_run_closure_token_present=_looks_like_sha256(specialist_post_run_closure_token_sha256),
                specialist_post_run_closure_token_authorizes_model_call=False,
                specialist_post_run_closure_token_authorizes_tool_execution=False,
                specialist_post_run_closure_token_authorizes_approval=False,
                specialist_post_run_closure_token_authorizes_personal_data_read=False,
                specialist_post_run_closure_token_authorizes_external_side_effect=False,
                specialist_post_run_closure_token_authorizes_completion_claim=False,
                specialist_post_run_closure_token_reusable_for_next_specialist_review=False,
                next_specialist_review_requires_new_post_run_closure_token=True,
                blockers=blockers_text,
                blockers_cleared=blockers_cleared,
                required_commands=required_commands,
                required_command_count=len(required_commands),
                proof_queue=required_commands,
                proof_queue_count=len(required_commands),
                next_proof_command=required_commands[0],
                handoff_pre_run_proof_queue=handoff_meta.get("pre_run_proof_queue", []),
                handoff_post_run_proof_queue=handoff_meta.get("post_run_proof_queue", []),
                handoff_output=handoff.output[:1600],
            ),
        )

    return specialist_post_run_closure_packet


def make_specialist_cycle_ledger_tool(config: JarvisConfig | None, list_tools):
    def specialist_cycle_ledger(args: dict[str, Any]) -> ToolResult:
        raw_request = _clean_text(args.get("request") or args.get("proposal") or "", limit=MAX_HANDOFF_REQUEST_CHARS)
        if not raw_request:
            return _missing_specialist_request_result(
                "specialist_cycle_ledger",
                "specialist cycle ledger: summarize memory; tool brain_search; args query=Jarvis harness; verification cites local memory only; runtime_trace reviewed; verification_receipt reviewed; audit reviewed; recovery reviewed; learning reviewed; completion_claim reviewed",
            )

        proposed_tool = _clean_text(
            args.get("tool") or args.get("tool_name") or _proposal_field_from_text(raw_request, ("tool", "tool name", "action tool")),
            limit=120,
        )
        proposed_arguments = _clean_text(
            args.get("arguments") or args.get("args") or _proposal_field_from_text(raw_request, ("args", "arguments", "input", "payload")),
            limit=MAX_PROPOSAL_ARGUMENT_CHARS,
        )
        verification = _clean_text(
            args.get("verification") or args.get("expected") or _proposal_field_from_text(raw_request, ("verification", "expected", "expectation", "success")),
            limit=MAX_PROPOSAL_ARGUMENT_CHARS,
        )
        runtime_trace = _clean_text(args.get("runtime_trace") or args.get("runtime_trace_receipt") or args.get("trace") or "", limit=MAX_PROPOSAL_ARGUMENT_CHARS)
        verification_receipt = _clean_text(args.get("verification_receipt") or args.get("receipt") or "", limit=MAX_PROPOSAL_ARGUMENT_CHARS)
        execution_audit = _clean_text(args.get("execution_audit") or args.get("audit") or "", limit=MAX_PROPOSAL_ARGUMENT_CHARS)
        execution_recovery = _clean_text(args.get("execution_recovery") or args.get("recovery") or "", limit=MAX_PROPOSAL_ARGUMENT_CHARS)
        after_action_learning = _clean_text(args.get("after_action_learning") or args.get("learning") or "", limit=MAX_PROPOSAL_ARGUMENT_CHARS)
        completion_claim = _clean_text(args.get("completion_claim") or args.get("claim") or "", limit=MAX_PROPOSAL_ARGUMENT_CHARS)
        runtime_trace_sha256 = _clean_text(
            args.get("runtime_trace_sha256")
            or args.get("runtime_trace_hash")
            or _proposal_field_from_text(raw_request, ("runtime_trace_sha256", "runtime trace sha256", "runtime trace hash")),
            limit=80,
        )
        verification_receipt_sha256 = _clean_text(
            args.get("verification_receipt_sha256")
            or args.get("verification_receipt_hash")
            or args.get("receipt_sha256")
            or _proposal_field_from_text(raw_request, ("verification_receipt_sha256", "verification receipt sha256", "receipt sha256")),
            limit=80,
        )
        execution_audit_sha256 = _clean_text(
            args.get("execution_audit_sha256")
            or args.get("audit_sha256")
            or args.get("audit_hash")
            or _proposal_field_from_text(raw_request, ("execution_audit_sha256", "audit_sha256", "audit sha256")),
            limit=80,
        )
        execution_recovery_sha256 = _clean_text(
            args.get("execution_recovery_sha256")
            or args.get("recovery_sha256")
            or args.get("recovery_hash")
            or _proposal_field_from_text(raw_request, ("execution_recovery_sha256", "recovery_sha256", "recovery sha256")),
            limit=80,
        )
        after_action_learning_sha256 = _clean_text(
            args.get("after_action_learning_sha256")
            or args.get("learning_sha256")
            or args.get("learning_hash")
            or _proposal_field_from_text(raw_request, ("after_action_learning_sha256", "learning_sha256", "learning sha256")),
            limit=80,
        )
        completion_claim_sha256 = _clean_text(
            args.get("completion_claim_sha256")
            or args.get("claim_sha256")
            or args.get("claim_hash")
            or _proposal_field_from_text(raw_request, ("completion_claim_sha256", "claim_sha256", "claim sha256")),
            limit=80,
        )
        blockers_text = _clean_text(args.get("blockers") or "none", limit=MAX_PROPOSAL_ARGUMENT_CHARS)
        request = _proposal_request_without_fields(raw_request)

        route_quality = make_specialist_route_quality_tool(config, list_tools)({"request": request})
        readiness = make_specialist_execution_readiness_tool(config, list_tools)({"request": request})
        handoff_receipt = make_specialist_handoff_receipt_tool(config, list_tools)({"request": request})
        handoff_quality = make_specialist_handoff_quality_gate_tool(config, list_tools)({"request": request})
        proposal_gate = make_specialist_proposal_gate_tool(config, list_tools)({"request": request})
        action_contract = make_specialist_action_proposal_contract_tool(config, list_tools)(
            {"request": raw_request, "tool": proposed_tool, "arguments": proposed_arguments}
        )
        dry_run = make_specialist_tool_dry_run_packet_tool(config, list_tools)(
            {"request": raw_request, "tool": proposed_tool, "arguments": proposed_arguments, "verification": verification}
        )
        completion = make_specialist_proposal_completion_gate_tool(config, list_tools)(
            {"request": raw_request, "tool": proposed_tool, "arguments": proposed_arguments, "verification": verification}
        )
        execution_handoff = make_specialist_execution_handoff_packet_tool(config, list_tools)(
            {"request": raw_request, "tool": proposed_tool, "arguments": proposed_arguments, "verification": verification}
        )
        post_run = make_specialist_post_run_closure_packet_tool(config, list_tools)(
            {
                "request": raw_request,
                "tool": proposed_tool,
                "arguments": proposed_arguments,
                "verification": verification,
                "runtime_trace": runtime_trace,
                "verification_receipt": verification_receipt,
                "execution_audit": execution_audit,
                "execution_recovery": execution_recovery,
                "after_action_learning": after_action_learning,
                "completion_claim": completion_claim,
                "runtime_trace_sha256": runtime_trace_sha256,
                "verification_receipt_sha256": verification_receipt_sha256,
                "execution_audit_sha256": execution_audit_sha256,
                "execution_recovery_sha256": execution_recovery_sha256,
                "after_action_learning_sha256": after_action_learning_sha256,
                "completion_claim_sha256": completion_claim_sha256,
                "blockers": blockers_text,
            }
        )

        route_meta = dict(route_quality.metadata)
        readiness_meta = dict(readiness.metadata)
        receipt_meta = dict(handoff_receipt.metadata)
        handoff_quality_meta = dict(handoff_quality.metadata)
        proposal_gate_meta = dict(proposal_gate.metadata)
        action_contract_meta = dict(action_contract.metadata)
        dry_run_meta = dict(dry_run.metadata)
        completion_meta = dict(completion.metadata)
        execution_handoff_meta = dict(execution_handoff.metadata)
        post_run_meta = dict(post_run.metadata)

        stage_rows = [
            {
                "stage": "route_quality",
                "state": route_meta.get("verdict"),
                "ready": bool(route_meta.get("measured_quality")) and int(route_meta.get("verifier_coverage_percent") or 0) >= 67,
                "proof": route_meta.get("proof_target") or f"specialist route quality: {request}",
            },
            {
                "stage": "execution_readiness",
                "state": readiness_meta.get("verdict"),
                "ready": readiness_meta.get("verdict") in {"READY_FOR_SPECIALIST_MODEL_DRAFT", "HOLD_FOR_MODEL_SETUP"},
                "proof": readiness_meta.get("readiness_target") or f"specialist execution readiness: {request}",
            },
            {
                "stage": "handoff_receipt",
                "state": receipt_meta.get("handoff_quality_state"),
                "ready": bool(receipt_meta.get("handoff_receipt_id")) and bool(receipt_meta.get("handoff_quality_measured")),
                "proof": receipt_meta.get("handoff_target") or f"specialist handoff receipt: {request}",
            },
            {
                "stage": "handoff_quality_gate",
                "state": handoff_quality_meta.get("quality_gate_state"),
                "ready": _metadata_bool(handoff_quality_meta.get("quality_binding_ready")),
                "proof": f"specialist handoff quality gate: {request}",
            },
            {
                "stage": "proposal_gate",
                "state": proposal_gate_meta.get("proposal_gate_state"),
                "ready": proposal_gate_meta.get("proposal_gate_state") in {"PROPOSAL_DRAFT_READY", "PROPOSAL_FALLBACK_NO_MODEL"},
                "proof": f"specialist proposal gate: {request}",
            },
            {
                "stage": "action_proposal_contract",
                "state": action_contract_meta.get("contract_state"),
                "ready": _metadata_bool(action_contract_meta.get("proposal_ready_for_local_safe_dry_run_review")),
                "proof": f"specialist action proposal contract: {raw_request}",
            },
            {
                "stage": "tool_dry_run",
                "state": dry_run_meta.get("dry_run_state"),
                "ready": _metadata_bool(dry_run_meta.get("dry_run_ready_for_operator_review")),
                "proof": f"specialist tool dry run: {raw_request}",
            },
            {
                "stage": "proposal_completion_gate",
                "state": completion_meta.get("completion_state"),
                "ready": _metadata_bool(completion_meta.get("completion_ready_for_operator_review")),
                "proof": f"specialist proposal completion gate: {raw_request}",
            },
            {
                "stage": "execution_handoff",
                "state": execution_handoff_meta.get("handoff_state"),
                "ready": _metadata_bool(execution_handoff_meta.get("ready_for_runtime_review")),
                "proof": f"specialist execution handoff: {raw_request}",
            },
            {
                "stage": "post_run_closure",
                "state": post_run_meta.get("closure_state"),
                "ready": _metadata_bool(post_run_meta.get("ready_to_count_specialist_execution_closed")),
                "proof": f"specialist post-run closure: {raw_request}",
            },
        ]
        for row in stage_rows:
            row.update(
                {
                    "authorizes_action_now": False,
                    "authorizes_model_call": False,
                    "authorizes_tool_execution": False,
                    "authorizes_approval": False,
                    "authorizes_personal_data_read": False,
                    "authorizes_external_side_effect": False,
                    "authorizes_fresh_review": False,
                    "reusable_for_next_cycle": False,
                }
            )
        missing = [str(row["stage"]) for row in stage_rows if not row["ready"]]
        ledger_ready = not missing
        cycle_state = "SPECIALIST_CYCLE_LEDGER_READY_FOR_FRESH_REVIEW" if ledger_ready else "SPECIALIST_CYCLE_LEDGER_HELD"
        required_commands = list(dict.fromkeys(str(row["proof"]) for row in stage_rows))
        next_command = (
            f"specialist router contract: {request}"
            if ledger_ready
            else next((str(row["proof"]) for row in stage_rows if not row["ready"]), required_commands[0])
        )
        fresh_review_preflight_queue = [
            f"specialist router contract: {request}",
            f"specialist route quality: {request}",
            f"specialist execution readiness: {request}",
            f"specialist handoff receipt: {request}",
            f"specialist proposal gate: {request}",
            f"specialist action proposal contract: {raw_request}",
            f"specialist tool dry run: {raw_request}",
            f"specialist execution handoff: {raw_request}",
            f"specialist post-run closure: {raw_request}",
        ]
        fresh_review_preflight_queue = list(dict.fromkeys(command for command in fresh_review_preflight_queue if command))
        fresh_review_contract_rows = [
            {
                "item": "router_contract",
                "source": "specialist router contract",
                "prior_artifact_reusable": False,
                "fresh_required": True,
                "authorizes_action_now": False,
            },
            {
                "item": "route_quality",
                "source": "specialist route quality",
                "prior_artifact_reusable": False,
                "fresh_required": True,
                "authorizes_action_now": False,
            },
            {
                "item": "execution_readiness",
                "source": "specialist execution readiness",
                "prior_artifact_reusable": False,
                "fresh_required": True,
                "authorizes_action_now": False,
            },
            {
                "item": "handoff_receipt",
                "source": "specialist handoff receipt",
                "prior_artifact_reusable": False,
                "fresh_required": True,
                "authorizes_action_now": False,
            },
            {
                "item": "proposal_gate",
                "source": "specialist proposal gate",
                "prior_artifact_reusable": False,
                "fresh_required": True,
                "authorizes_action_now": False,
            },
            {
                "item": "action_proposal_contract",
                "source": "specialist action proposal contract",
                "prior_artifact_reusable": False,
                "fresh_required": True,
                "authorizes_action_now": False,
            },
            {
                "item": "tool_dry_run",
                "source": "specialist tool dry-run packet",
                "prior_artifact_reusable": False,
                "fresh_required": True,
                "authorizes_action_now": False,
            },
            {
                "item": "execution_handoff",
                "source": "specialist execution handoff packet",
                "prior_artifact_reusable": False,
                "fresh_required": True,
                "authorizes_action_now": False,
            },
            {
                "item": "post_run_closure",
                "source": "specialist post-run closure packet",
                "prior_artifact_reusable": False,
                "fresh_required": True,
                "authorizes_action_now": False,
            },
            {
                "item": "post_run_closure_token",
                "source": "specialist post-run closure token",
                "prior_artifact_reusable": False,
                "fresh_required": True,
                "authorizes_action_now": False,
            },
            {
                "item": "specialist_review_token",
                "source": "specialist review token",
                "prior_artifact_reusable": False,
                "fresh_required": True,
                "authorizes_action_now": False,
            },
            {
                "item": "route_to_runtime_contract_token",
                "source": "route-to-runtime contract token",
                "prior_artifact_reusable": False,
                "fresh_required": True,
                "authorizes_action_now": False,
            },
        ]
        for row in fresh_review_contract_rows:
            row.update(
                {
                    "authorizes_model_call": False,
                    "authorizes_tool_execution": False,
                    "authorizes_approval": False,
                    "authorizes_personal_data_read": False,
                    "authorizes_external_side_effect": False,
                    "authorizes_fresh_review": False,
                }
            )
        all_prior_artifacts_non_authorizing = all(
            row["prior_artifact_reusable"] is False
            and row["authorizes_action_now"] is False
            and row["authorizes_model_call"] is False
            and row["authorizes_tool_execution"] is False
            and row["authorizes_approval"] is False
            and row["authorizes_personal_data_read"] is False
            and row["authorizes_external_side_effect"] is False
            and row["authorizes_fresh_review"] is False
            for row in fresh_review_contract_rows
        )
        action_contract_scorecard_rows = list(action_contract_meta.get("action_proposal_scorecard_rows") or [])
        dry_run_scorecard_rows = list(dry_run_meta.get("action_proposal_scorecard_rows") or [])
        completion_scorecard_rows = list(completion_meta.get("action_proposal_scorecard_rows") or [])
        proposal_scorecard_contract_shape_ready = bool(
            _specialist_action_proposal_scorecard_shape_ready(action_contract_scorecard_rows)
            and _specialist_action_proposal_scorecard_shape_ready(dry_run_scorecard_rows)
            and _specialist_combined_action_proposal_scorecard_shape_ready(completion_scorecard_rows)
        )
        proposal_scorecard_contract_ready = (
            _metadata_bool(action_contract_meta.get("action_proposal_scorecard_required_rows_ready"))
            and _metadata_bool(dry_run_meta.get("action_proposal_scorecard_required_rows_ready"))
            and _metadata_bool(completion_meta.get("action_proposal_scorecard_required_rows_ready"))
        )
        specialist_fresh_review_boundary_token_sha256 = _specialist_fresh_review_boundary_token_sha256(
            request=request,
            cycle_state=cycle_state,
            fresh_review_preflight_queue=fresh_review_preflight_queue,
            fresh_review_contract_rows=fresh_review_contract_rows,
            stage_rows=stage_rows,
            specialist_review_token_sha256=str(post_run_meta.get("specialist_review_token_sha256") or ""),
            runtime_review_boundary_token_sha256=str(post_run_meta.get("runtime_review_boundary_token_sha256") or ""),
            route_to_runtime_contract_token_sha256=str(post_run_meta.get("route_to_runtime_contract_token_sha256") or ""),
            next_command=next_command,
        )
        specialist_cycle_ledger_token_sha256 = _specialist_cycle_ledger_token_sha256(
            request=request,
            raw_request=raw_request,
            cycle_state=cycle_state,
            stage_rows=stage_rows,
            fresh_review_preflight_queue=fresh_review_preflight_queue,
            fresh_review_contract_rows=fresh_review_contract_rows,
            required_commands=required_commands,
            specialist_review_token_sha256=str(post_run_meta.get("specialist_review_token_sha256") or ""),
            runtime_review_boundary_token_sha256=str(post_run_meta.get("runtime_review_boundary_token_sha256") or ""),
            route_to_runtime_contract_token_sha256=str(post_run_meta.get("route_to_runtime_contract_token_sha256") or ""),
            specialist_fresh_review_boundary_token_sha256=specialist_fresh_review_boundary_token_sha256,
            specialist_post_run_closure_token_sha256=str(post_run_meta.get("specialist_post_run_closure_token_sha256") or ""),
            runtime_trace_sha256=str(post_run_meta.get("runtime_trace_sha256") or ""),
            verification_receipt_sha256=str(post_run_meta.get("verification_receipt_sha256") or ""),
            execution_audit_sha256=str(post_run_meta.get("execution_audit_sha256") or ""),
            execution_recovery_sha256=str(post_run_meta.get("execution_recovery_sha256") or ""),
            after_action_learning_sha256=str(post_run_meta.get("after_action_learning_sha256") or ""),
            completion_claim_sha256=str(post_run_meta.get("completion_claim_sha256") or ""),
            action_contract_scorecard_rows=action_contract_scorecard_rows,
            dry_run_scorecard_rows=dry_run_scorecard_rows,
            completion_scorecard_rows=completion_scorecard_rows,
            post_run_proof_queue=list(post_run_meta.get("proof_queue") or []),
            next_command=next_command,
        )
        cycle_token_metadata = {
            "request": request,
            "raw_request": raw_request,
            "cycle_state": cycle_state,
            "stage_rows": stage_rows,
            "fresh_review_preflight_queue": fresh_review_preflight_queue,
            "fresh_review_contract_rows": fresh_review_contract_rows,
            "required_commands": required_commands,
            "specialist_review_token_sha256": str(post_run_meta.get("specialist_review_token_sha256") or ""),
            "runtime_review_boundary_token_sha256": str(post_run_meta.get("runtime_review_boundary_token_sha256") or ""),
            "route_to_runtime_contract_token_sha256": str(post_run_meta.get("route_to_runtime_contract_token_sha256") or ""),
            "specialist_fresh_review_boundary_token_sha256": specialist_fresh_review_boundary_token_sha256,
            "specialist_post_run_closure_token_sha256": str(post_run_meta.get("specialist_post_run_closure_token_sha256") or ""),
            "runtime_trace_sha256": str(post_run_meta.get("runtime_trace_sha256") or ""),
            "verification_receipt_sha256": str(post_run_meta.get("verification_receipt_sha256") or ""),
            "execution_audit_sha256": str(post_run_meta.get("execution_audit_sha256") or ""),
            "execution_recovery_sha256": str(post_run_meta.get("execution_recovery_sha256") or ""),
            "after_action_learning_sha256": str(post_run_meta.get("after_action_learning_sha256") or ""),
            "completion_claim_sha256": str(post_run_meta.get("completion_claim_sha256") or ""),
            "action_contract_scorecard_rows": action_contract_scorecard_rows,
            "dry_run_scorecard_rows": dry_run_scorecard_rows,
            "completion_scorecard_rows": completion_scorecard_rows,
            "post_run_closure_metadata": post_run_meta,
            "next_command": next_command,
            "specialist_cycle_ledger_token_sha256": specialist_cycle_ledger_token_sha256,
            "specialist_cycle_ledger_token_authorizes_model_call": False,
            "specialist_cycle_ledger_token_authorizes_tool_execution": False,
            "specialist_cycle_ledger_token_authorizes_approval": False,
            "specialist_cycle_ledger_token_authorizes_personal_data_read": False,
            "specialist_cycle_ledger_token_authorizes_external_side_effect": False,
            "specialist_cycle_ledger_token_authorizes_completion_claim": False,
            "specialist_cycle_ledger_token_reusable_for_next_specialist_review": False,
            "next_specialist_review_requires_new_cycle_ledger_token": True,
        }
        specialist_cycle_ledger_token_ready = _specialist_cycle_ledger_token_ready_from_metadata(cycle_token_metadata)
        specialist_cycle_ledger_ready = _specialist_cycle_ledger_ready(
            stage_rows=stage_rows,
            fresh_review_contract_rows=fresh_review_contract_rows,
            action_contract_scorecard_rows=action_contract_scorecard_rows,
            dry_run_scorecard_rows=dry_run_scorecard_rows,
            completion_scorecard_rows=completion_scorecard_rows,
            post_run_artifact_hashes_present=_metadata_bool(post_run_meta.get("post_run_artifact_hashes_present")),
            specialist_review_token_sha256=str(post_run_meta.get("specialist_review_token_sha256") or ""),
            runtime_review_boundary_token_sha256=str(post_run_meta.get("runtime_review_boundary_token_sha256") or ""),
            route_to_runtime_contract_token_sha256=str(post_run_meta.get("route_to_runtime_contract_token_sha256") or ""),
            specialist_post_run_closure_token_sha256=str(post_run_meta.get("specialist_post_run_closure_token_sha256") or ""),
            specialist_fresh_review_boundary_token_sha256=specialist_fresh_review_boundary_token_sha256,
            specialist_cycle_ledger_token_sha256=specialist_cycle_ledger_token_sha256,
        )
        specialist_cycle_ledger_ready = bool(specialist_cycle_ledger_ready and specialist_cycle_ledger_token_ready)
        cycle_ready_for_fresh_review = bool(ledger_ready and specialist_cycle_ledger_ready)
        cycle_state = (
            "SPECIALIST_CYCLE_LEDGER_READY_FOR_FRESH_REVIEW"
            if cycle_ready_for_fresh_review
            else "SPECIALIST_CYCLE_LEDGER_HELD"
        )

        lines = [
            "Jarvis specialist cycle ledger:",
            "This is read-only. It binds route quality, readiness, handoff, proposal, dry-run, runtime handoff, and post-run closure proof before a specialist cycle can count as closed or a fresh specialist review can start.",
            "",
            f"Request: {request}",
            "",
            "Cycle state:",
            f"- state: {cycle_state}",
            f"- ready for fresh specialist review: {'yes' if cycle_ready_for_fresh_review else 'no'}",
            f"- cycle ledger contract ready: {'yes' if specialist_cycle_ledger_ready else 'no'}",
            "- previous specialist execution permission reusable for next review: no",
            "- action execution allowed now: no",
            "- executable action emitted: no",
            "- model called: no",
            "- approval queued: no",
            f"- next command: `{next_command}`",
            f"- missing stages: {', '.join(missing) if missing else 'none'}",
            "",
            "Cycle stages:",
            *[
                f"- {row['stage']}: {'ready' if row['ready'] else 'held'} ({row['state'] or 'unknown'}) via `{row['proof']}`"
                for row in stage_rows
            ],
            "",
            "Closure evidence:",
            f"- post-run missing evidence: {', '.join(post_run_meta.get('missing') or []) if post_run_meta.get('missing') else 'none'}",
            f"- runtime trace receipt ready: {'yes' if post_run_meta.get('runtime_trace_receipt_ready') else 'no'}",
            f"- verification receipt ready: {'yes' if post_run_meta.get('verification_receipt_ready') else 'no'}",
            f"- completion claim ready: {'yes' if post_run_meta.get('completion_claim_ready') else 'no'}",
            f"- post-run artifact hashes present: {'yes' if post_run_meta.get('post_run_artifact_hashes_present') else 'no'}",
            "",
            "Measured proposal quality:",
            f"- action contract score: {action_contract_meta.get('action_proposal_score', 0)}/100",
            f"- dry-run score: {dry_run_meta.get('action_proposal_score', 0)}/100",
            f"- completion score: {completion_meta.get('action_proposal_score', 0)}/100",
            f"- scorecard rows ready: {'yes' if proposal_scorecard_contract_ready else 'no'}",
            f"- action contract scorecard rows: {len(action_contract_scorecard_rows)}",
            f"- dry-run scorecard rows: {len(dry_run_scorecard_rows)}",
            f"- completion scorecard rows: {len(completion_scorecard_rows)}",
            "",
            "Fresh-review boundary:",
            "- A closed cycle does not grant carry-over execution permission.",
            "- Any next specialist review must start from a new router contract, route-quality proof, readiness gate, handoff receipt, proposal gate, action contract, dry-run, execution handoff, and post-run closure evidence.",
            f"- fresh-review preflight queue: {', '.join(f'`{command}`' for command in fresh_review_preflight_queue)}",
            "- fresh-review contract:",
            *[
                f"  - {row['item']}: fresh required yes; prior reusable no; authorizes action now no; authorizes model call no; authorizes tool execution no; authorizes approval no; authorizes personal-data read no; authorizes external side effect no; source {row['source']}"
                for row in fresh_review_contract_rows
            ],
            f"- all prior artifacts non-authorizing: {'yes' if all_prior_artifacts_non_authorizing else 'no'}",
            "- next specialist review requires full preflight: yes",
            f"- prior specialist review token sha256: {post_run_meta.get('specialist_review_token_sha256') or 'missing'}",
            "- previous specialist review token reusable for next review: no",
            "- new specialist review token required before next specialist action: yes",
            f"- runtime review boundary token sha256: {post_run_meta.get('runtime_review_boundary_token_sha256') or 'missing'}",
            "- previous runtime review boundary token reusable for next review: no",
            "- new runtime review boundary token required before next specialist action: yes",
            f"- route-to-runtime contract token sha256: {post_run_meta.get('route_to_runtime_contract_token_sha256') or 'missing'}",
            "- previous route-to-runtime contract token reusable for next review: no",
            "- new route-to-runtime contract token required before next specialist action: yes",
            f"- specialist fresh-review boundary token sha256: {specialist_fresh_review_boundary_token_sha256 or 'missing'}",
            "- fresh-review boundary token authorizes model call: no",
            "- fresh-review boundary token authorizes tool execution: no",
            "- fresh-review boundary token authorizes approval: no",
            "- fresh-review boundary token authorizes personal-data read: no",
            "- fresh-review boundary token authorizes external side effect: no",
            "- fresh-review boundary token authorizes fresh review: no",
            "- fresh-review boundary token reusable for next specialist review: no",
            "- next specialist review requires new fresh-review boundary token: yes",
            f"- specialist post-run closure token sha256: {post_run_meta.get('specialist_post_run_closure_token_sha256') or 'missing'}",
            "- previous specialist post-run closure token reusable for next review: no",
            "- new specialist post-run closure token required before next specialist action: yes",
            f"- specialist cycle ledger token sha256: {specialist_cycle_ledger_token_sha256 or 'missing'}",
            "- specialist cycle ledger token authorizes model call: no",
            "- specialist cycle ledger token authorizes tool execution: no",
            "- specialist cycle ledger token authorizes approval: no",
            "- specialist cycle ledger token authorizes completion claim: no",
            "- specialist cycle ledger token reusable for next specialist review: no",
            "- Shell/code execution, computer control, personal-data access, external side effects, destructive work, and ambiguous actions remain approval-gated.",
            "",
            "Required cycle proof chain:",
            *[f"- `{command}`" for command in required_commands],
            "",
            "Boundary:",
            "- This ledger does not call models, execute tools, approve requests, dismiss approvals, write files, read personal data, control the computer, call external services, queue approvals, or claim AGI completion.",
        ]

        return ToolResult(
            "specialist_cycle_ledger",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=request,
                raw_request=raw_request,
                cycle_state=cycle_state,
                ready_for_fresh_specialist_review=cycle_ready_for_fresh_review,
                ready_to_count_specialist_cycle_closed=cycle_ready_for_fresh_review,
                specialist_cycle_ledger_ready=specialist_cycle_ledger_ready,
                previous_specialist_execution_permission_reusable_for_next_review=False,
                previous_specialist_review_token_reusable_for_next_review=False,
                next_specialist_review_requires_new_token=True,
                fresh_review_preflight_queue=fresh_review_preflight_queue,
                fresh_review_preflight_queue_count=len(fresh_review_preflight_queue),
                fresh_review_next_preflight_command=fresh_review_preflight_queue[0] if fresh_review_preflight_queue else "",
                fresh_review_contract_rows=fresh_review_contract_rows,
                fresh_review_contract_count=len(fresh_review_contract_rows),
                all_prior_artifacts_non_authorizing=all_prior_artifacts_non_authorizing,
                next_specialist_review_requires_full_preflight=True,
                action_allowed_now=False,
                executable_action_emitted=False,
                can_emit_executable_tool_action=False,
                can_auto_execute_now=False,
                tool_executed=False,
                calls_model=False,
                queues_approval=False,
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                cycle_missing_stages=missing,
                cycle_missing_stage_count=len(missing),
                stage_rows=stage_rows,
                stage_count=len(stage_rows),
                required_commands=required_commands,
                required_command_count=len(required_commands),
                proof_queue=required_commands,
                proof_queue_count=len(required_commands),
                next_command=next_command,
                proposed_tool=proposed_tool,
                proposed_arguments=proposed_arguments,
                verification=verification,
                primary_specialist=completion_meta.get("primary_specialist") or route_meta.get("primary_specialist"),
                supporting_specialists=completion_meta.get("supporting_specialists") or route_meta.get("supporting_specialists") or [],
                target_model=completion_meta.get("target_model") or route_meta.get("target_model"),
                specialist_review_token_sha256=post_run_meta.get("specialist_review_token_sha256", ""),
                specialist_review_token_present=post_run_meta.get("specialist_review_token_present", False),
                specialist_review_token_reusable_for_next_review=False,
                runtime_review_boundary_token_sha256=post_run_meta.get("runtime_review_boundary_token_sha256", ""),
                runtime_review_boundary_token_present=post_run_meta.get("runtime_review_boundary_token_present", False),
                runtime_review_boundary_token_authorizes_model_call=False,
                runtime_review_boundary_token_authorizes_tool_execution=False,
                runtime_review_boundary_token_authorizes_approval=False,
                runtime_review_boundary_token_authorizes_personal_data_read=False,
                runtime_review_boundary_token_authorizes_external_side_effect=False,
                runtime_review_boundary_token_authorizes_completion_claim=False,
                runtime_review_boundary_token_bypasses_post_run_proof=False,
                runtime_review_boundary_token_reusable_for_next_specialist_review=False,
                previous_runtime_review_boundary_token_reusable_for_next_review=False,
                next_specialist_review_requires_new_runtime_review_boundary_token=True,
                route_to_runtime_contract_token_sha256=post_run_meta.get("route_to_runtime_contract_token_sha256", ""),
                route_to_runtime_contract_token_present=post_run_meta.get("route_to_runtime_contract_token_present", False),
                route_to_runtime_contract_token_authorizes_model_call=False,
                route_to_runtime_contract_token_authorizes_tool_execution=False,
                route_to_runtime_contract_token_authorizes_approval=False,
                route_to_runtime_contract_token_authorizes_personal_data_read=False,
                route_to_runtime_contract_token_authorizes_external_side_effect=False,
                route_to_runtime_contract_token_authorizes_completion_claim=False,
                route_to_runtime_contract_token_reusable_for_next_specialist_review=False,
                previous_route_to_runtime_contract_token_reusable_for_next_review=False,
                next_specialist_review_requires_new_route_to_runtime_contract_token=True,
                specialist_post_run_closure_token_sha256=post_run_meta.get("specialist_post_run_closure_token_sha256", ""),
                specialist_post_run_closure_token_present=post_run_meta.get("specialist_post_run_closure_token_present", False),
                specialist_post_run_closure_token_authorizes_model_call=False,
                specialist_post_run_closure_token_authorizes_tool_execution=False,
                specialist_post_run_closure_token_authorizes_approval=False,
                specialist_post_run_closure_token_authorizes_personal_data_read=False,
                specialist_post_run_closure_token_authorizes_external_side_effect=False,
                specialist_post_run_closure_token_authorizes_completion_claim=False,
                specialist_post_run_closure_token_reusable_for_next_specialist_review=False,
                previous_specialist_post_run_closure_token_reusable_for_next_review=False,
                next_specialist_review_requires_new_post_run_closure_token=True,
                specialist_fresh_review_boundary_token_sha256=specialist_fresh_review_boundary_token_sha256,
                specialist_fresh_review_boundary_token_present=_looks_like_sha256(specialist_fresh_review_boundary_token_sha256),
                specialist_fresh_review_boundary_token_authorizes_action_now=False,
                specialist_fresh_review_boundary_token_authorizes_model_call=False,
                specialist_fresh_review_boundary_token_authorizes_tool_execution=False,
                specialist_fresh_review_boundary_token_authorizes_approval=False,
                specialist_fresh_review_boundary_token_authorizes_personal_data_read=False,
                specialist_fresh_review_boundary_token_authorizes_external_side_effect=False,
                specialist_fresh_review_boundary_token_authorizes_fresh_review=False,
                specialist_fresh_review_boundary_token_reusable_for_next_review=False,
                next_specialist_review_requires_new_fresh_review_boundary_token=True,
                specialist_cycle_ledger_token_sha256=specialist_cycle_ledger_token_sha256,
                specialist_cycle_ledger_token_present=_looks_like_sha256(specialist_cycle_ledger_token_sha256),
                specialist_cycle_ledger_token_ready=specialist_cycle_ledger_token_ready,
                specialist_cycle_ledger_token_authorizes_model_call=False,
                specialist_cycle_ledger_token_authorizes_tool_execution=False,
                specialist_cycle_ledger_token_authorizes_approval=False,
                specialist_cycle_ledger_token_authorizes_personal_data_read=False,
                specialist_cycle_ledger_token_authorizes_external_side_effect=False,
                specialist_cycle_ledger_token_authorizes_completion_claim=False,
                specialist_cycle_ledger_token_reusable_for_next_specialist_review=False,
                next_specialist_review_requires_new_cycle_ledger_token=True,
                fallback_lane=completion_meta.get("fallback_lane") or route_meta.get("fallback_lane"),
                proof_target=completion_meta.get("proof_target") or route_meta.get("proof_target"),
                handoff_target=completion_meta.get("handoff_target") or route_meta.get("handoff_target"),
                readiness_target=completion_meta.get("readiness_target") or route_meta.get("readiness_target"),
                stop_condition=completion_meta.get("stop_condition") or route_meta.get("stop_condition"),
                route_quality_metadata=route_meta,
                execution_readiness_metadata=readiness_meta,
                handoff_receipt_metadata=receipt_meta,
                handoff_quality_metadata=handoff_quality_meta,
                proposal_gate_metadata=proposal_gate_meta,
                action_contract_metadata=action_contract_meta,
                dry_run_metadata=dry_run_meta,
                proposal_completion_metadata=completion_meta,
                execution_handoff_metadata=execution_handoff_meta,
                post_run_closure_metadata=post_run_meta,
                action_contract_scorecard_rows=action_contract_scorecard_rows,
                action_contract_scorecard_row_count=len(action_contract_scorecard_rows),
                action_contract_scorecard_required_rows_ready=_metadata_bool(
                    action_contract_meta.get("action_proposal_scorecard_required_rows_ready")
                ),
                dry_run_scorecard_rows=dry_run_scorecard_rows,
                dry_run_scorecard_row_count=len(dry_run_scorecard_rows),
                dry_run_scorecard_required_rows_ready=_metadata_bool(dry_run_meta.get("action_proposal_scorecard_required_rows_ready")),
                completion_scorecard_rows=completion_scorecard_rows,
                completion_scorecard_row_count=len(completion_scorecard_rows),
                completion_scorecard_required_rows_ready=_metadata_bool(completion_meta.get("action_proposal_scorecard_required_rows_ready")),
                proposal_scorecard_contract_shape_ready=proposal_scorecard_contract_shape_ready,
                proposal_scorecard_contract_ready=proposal_scorecard_contract_ready,
                post_run_artifact_hashes_present=_metadata_bool(post_run_meta.get("post_run_artifact_hashes_present")),
                runtime_trace_sha256=post_run_meta.get("runtime_trace_sha256", ""),
                verification_receipt_sha256=post_run_meta.get("verification_receipt_sha256", ""),
                execution_audit_sha256=post_run_meta.get("execution_audit_sha256", ""),
                execution_recovery_sha256=post_run_meta.get("execution_recovery_sha256", ""),
                after_action_learning_sha256=post_run_meta.get("after_action_learning_sha256", ""),
                completion_claim_sha256=post_run_meta.get("completion_claim_sha256", ""),
                post_run_closure_output=post_run.output[:1600],
            ),
        )

    return specialist_cycle_ledger
