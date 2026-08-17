from __future__ import annotations

import hashlib
import re
import subprocess
from shutil import which
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    declare_failure_guidance,
    declare_outcome_unknown_failure,
)
from jarvis_v2.agent.types import ToolResult
from jarvis_v2.config import JarvisConfig


MAX_CONNECTOR_CHARS = 64
MAX_ACTION_CHARS = 220
MAX_SCOPE_FIELD_CHARS = 180
MAX_REMINDER_TITLE_CHARS = 160
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
OPERATOR_LIMIT_STOP = "- Stop at the operator's explicit stop times, work windows, pause commands, or newer instructions, even when connector readiness looks green."
METADATA_ROW_SCHEMA = ["id", "timestamp", "label", "source"]
METADATA_ROW_LIMIT = 2
BLOCKED_METADATA_PAYLOAD_FIELDS = [
    "body",
    "content",
    "notes",
    "attachments",
    "tokens",
    "cookies",
    "contact_cards",
    "authenticated_page_content",
]


CONNECTOR_PROFILES = {
    "calendar": {
        "read_only": ["list calendars", "show free/busy windows", "summarize upcoming events"],
        "personal_data": ["event titles", "attendees", "locations", "meeting notes"],
        "side_effects": ["create event", "move event", "cancel event", "invite attendees"],
    },
    "email": {
        "read_only": ["search mailbox metadata", "summarize selected threads", "draft replies"],
        "personal_data": ["message bodies", "senders", "recipients", "attachments"],
        "side_effects": ["send email", "archive/delete mail", "apply labels", "forward messages"],
    },
    "messages": {
        "read_only": ["summarize selected conversations", "draft replies"],
        "personal_data": ["message bodies", "contacts", "phone numbers"],
        "side_effects": ["send message", "mark read", "start a new conversation"],
    },
    "contacts": {
        "read_only": ["search contact names", "inspect selected contact cards"],
        "personal_data": ["phone numbers", "emails", "addresses", "relationship notes"],
        "side_effects": ["create contact", "edit contact", "delete contact"],
    },
    "reminders": {
        "read_only": ["list selected reminder lists", "summarize upcoming reminders"],
        "personal_data": ["reminder titles", "due dates", "list names"],
        "side_effects": ["create reminder", "complete reminder", "delete reminder"],
    },
    "browser": {
        "read_only": ["open URL", "fetch public pages", "summarize fetched pages"],
        "personal_data": ["logged-in pages", "browser profile data", "cookies", "history"],
        "side_effects": ["submit forms", "click authenticated controls", "make purchases"],
    },
}


def _clean_text(value: Any, *, limit: int) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _display_text(value: Any, *, limit: int) -> str:
    text = _clean_text(value, limit=limit)
    return LOCAL_PATH_RE.sub("<local-path>", text)


def _has_local_path(value: Any) -> bool:
    return bool(LOCAL_PATH_RE.search(str(value or "")))


def _safe_value(value: Any) -> Any:
    if isinstance(value, str):
        return LOCAL_PATH_RE.sub("<local-path>", value)
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_safe_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _safe_value(item) for key, item in value.items()}
    return value


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
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
        "controls_computer": False,
        "speaks": False,
        "completes_tasks": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
    }
    metadata.update({key: _safe_value(value) for key, value in extra.items()})
    return metadata


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if value is True:
        return True
    if value is False:
        return False
    return default


def _personal_input_failure(
    tool_name: str,
    message: str,
    recovery_action: str,
    **metadata: Any,
) -> ToolResult:
    """Return one canonical refusal before any personal action was attempted."""

    output = f"{message} {recovery_action}"
    failure_metadata: dict[str, Any] = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
    }
    failure_metadata.update(metadata)
    return ToolResult(
        tool_name,
        False,
        output,
        declare_failure_guidance(
            _safe_metadata(**failure_metadata),
            output=output,
            action=recovery_action,
        ),
    )


def _action_required_failure(tool_name: str) -> ToolResult:
    return _personal_input_failure(
        tool_name,
        "Action is required.",
        "Provide one exact connector action, then retry this read-only planning tool.",
        reason="missing_action",
    )


def _open_vault_outcome_unknown(
    path: Any,
    *,
    exception_type: str | None = None,
    returncode: int | None = None,
) -> ToolResult:
    output = (
        "Could not open Jarvis vault. It may already be open; check first and do not automatically "
        "retry. If absent, check JARVIS_OBSIDIAN_VAULT and Finder > Get Info > Sharing & "
        "Permissions, run `setup check`, then retry through normal policy."
    )
    metadata = declare_outcome_unknown_failure(
        _safe_metadata(
            path=str(path),
            path_display="Jarvis vault",
            executes_tools=True,
            executes_side_effect=True,
            external_side_effect=False,
            controls_computer=True,
            exception_type=exception_type,
            returncode=returncode,
        ),
        output=output,
        commands=("setup check",),
    )
    return ToolResult("open_jarvis_vault", False, output, metadata)


def _connector_profile(name: str) -> tuple[str, dict[str, list[str]]]:
    normalized = _clean_text(name, limit=MAX_CONNECTOR_CHARS).lower().replace("sms", "messages")
    if normalized in CONNECTOR_PROFILES:
        return normalized, CONNECTOR_PROFILES[normalized]
    return "generic", {
        "read_only": ["status check", "limited metadata listing", "draft-only summaries"],
        "personal_data": ["private content", "account identifiers", "local app state"],
        "side_effects": ["create/update/delete actions", "send/share actions", "state-changing automation"],
    }


def _classify_connector_action(action: str) -> tuple[str, bool, str]:
    action_low = action.lower()
    side_effect_words = (
        "send",
        "create",
        "delete",
        "cancel",
        "move",
        "edit",
        "update",
        "archive",
        "label",
        "forward",
        "invite",
        "submit",
        "purchase",
        "complete",
        "mark read",
        "reply",
    )
    personal_data_words = (
        "read",
        "show",
        "list",
        "summarize",
        "search",
        "inspect",
        "open",
        "body",
        "attachment",
        "attendee",
        "contact",
        "phone",
        "email",
        "location",
        "history",
        "cookie",
        "logged-in",
    )
    read_only_words = ("draft", "plan", "preview", "contract", "metadata", "free/busy", "public")

    if any(word in action_low for word in side_effect_words):
        return "EXTERNAL_SIDE_EFFECT", True, "The action appears to change outside-world state or another app/account."
    if any(word in action_low for word in personal_data_words):
        return "PERSONAL_DATA", True, "The action may expose private account, message, calendar, contact, browser, or reminder data."
    if any(word in action_low for word in read_only_words):
        return "READ_ONLY", False, "The action appears limited to drafting, planning, metadata, or public/read-only preview."
    return "PERSONAL_DATA", True, "The action is ambiguous, so Jarvis should treat it as private until the operator narrows the scope."


def _fake_metadata_rows(connector: str, *, target: str, time_range: str) -> list[dict[str, str]]:
    source = target or "sample source"
    timestamp = time_range or "sample time"
    return [
        {"id": "sample-1", "timestamp": timestamp, "label": f"{connector} metadata item", "source": source},
        {"id": "sample-2", "timestamp": timestamp, "label": "second bounded metadata item", "source": source},
    ]


def _metadata_row_contract(sample_rows: list[dict[str, str]]) -> dict[str, Any]:
    observed_fields = sorted({key for row in sample_rows for key in row})
    return {
        "sample_row_schema": list(METADATA_ROW_SCHEMA),
        "sample_row_schema_fields": list(METADATA_ROW_SCHEMA),
        "sample_row_limit": METADATA_ROW_LIMIT,
        "sample_rows_bounded": len(sample_rows) <= METADATA_ROW_LIMIT and observed_fields == sorted(METADATA_ROW_SCHEMA),
        "blocked_payload_fields": list(BLOCKED_METADATA_PAYLOAD_FIELDS),
    }


def _metadata_row_contract_required(data_level: str) -> bool:
    data_low = data_level.lower()
    return "metadata" in data_low and not any(
        word in data_low for word in ("full", "body", "content", "attachment", "transcript", "notes")
    )


def _metadata_preview_handoff_payload(
    *,
    connector: str,
    requested_connector: str,
    action: str,
    tool_name: str,
    preview_state: str,
    adapter_state: str,
    metadata_only: bool,
    missing_fields: list[str],
    target: str,
    time_range: str,
    data_level: str,
    verification: str,
    tests: str,
    audit: str,
    acceptance: str,
    sample_rows: list[dict[str, str]],
    row_contract: dict[str, Any],
    suggested_risk: str,
    approval_required: bool,
) -> dict[str, Any]:
    preview_ready = preview_state == "METADATA_PREVIEW_READY"
    next_safe_commands = [
        (
            f"integration proof bundle: {connector} -> {action}; "
            f"target {target or '<exact source>'}; "
            f"time {time_range or '<exact window>'}; "
            "data metadata-only; "
            f"verification {verification or '<proof target>'}; "
            "rollback no connector call; "
            f"tests {tests or '<focused smoke>'}; "
            f"audit {audit or '<audit receipt>'}; "
            f"acceptance {acceptance or '<gate evidence>'}"
        )
        if preview_ready
        else (
            f"integration metadata preview: {connector} -> {action}; "
            "fill exact target, time range, metadata-only data level, verification, tests, audit, and acceptance evidence"
        )
    ]
    return {
        "handoff_ready": preview_ready,
        "metadata_preview_handoff_ready": preview_ready,
        "ready_for_operator": preview_ready,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "connector": connector,
        "requested_connector": requested_connector,
        "action": action,
        "tool_name": tool_name,
        "preview_state": preview_state,
        "metadata_preview_ready": preview_ready,
        "metadata_only": metadata_only,
        "adapter_state": adapter_state,
        "missing_fields": list(missing_fields),
        "scope": {
            "target": target,
            "time_range": time_range,
            "data_level": data_level,
            "verification": verification,
        },
        "proof": {
            "tests": tests,
            "audit": audit,
            "acceptance": acceptance,
        },
        "sample_rows": sample_rows,
        "sample_row_count": len(sample_rows),
        "row_contract": {
            "sample_row_schema": list(row_contract["sample_row_schema"]),
            "sample_row_schema_fields": list(row_contract["sample_row_schema_fields"]),
            "sample_row_limit": row_contract["sample_row_limit"],
            "sample_rows_bounded": row_contract["sample_rows_bounded"],
            "blocked_payload_fields": list(row_contract["blocked_payload_fields"]),
        },
        "risk": {
            "suggested_risk": suggested_risk,
            "approval_required": approval_required,
        },
        "boundaries": {
            "natural_language_routing_enabled": False,
            "adapter_call_skipped": True,
            "calls_external_service": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "authorizes_natural_language_routing": False,
            "authorizes_personal_data_read": False,
            "authorizes_side_effect": False,
            "authorizes_approval": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_external_service": False,
            "reusable_for_other_scope": False,
        },
        "next_safe_commands": next_safe_commands,
        "next_step_requires_proof_bundle": True,
        "next_route_lock_requires_fresh_review": True,
    }


def _adapter_manifest_handoff_payload(
    *,
    requested_connector: str,
    rows: list[dict[str, Any]],
    sample_row_schema: list[str],
    sample_row_limit: int,
    blocked_payload_fields: list[str],
    all_manifest_row_contracts_present: bool,
) -> dict[str, Any]:
    next_safe_commands = [
        f"integration adapter probe: {requested_connector or '<connector>'}",
        f"integration adapter acceptance: {requested_connector or '<connector>'}",
        f"integration metadata preview: {requested_connector or '<connector>'} -> <metadata action>; target <exact source>; time <exact window>; data metadata-only; verification fake rows only; tests blocked full-content smoke; audit metadata preview receipt; acceptance gate passed",
    ]
    return {
        "handoff_ready": True,
        "adapter_manifest_handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "connector": requested_connector or "all",
        "connectors_reviewed": len(rows),
        "manifest_rows": rows,
        "adapter_states": [row["adapter_state"] for row in rows],
        "future_tools": [row["future_tool"] for row in rows],
        "fixture_names": [row["fixture_name"] for row in rows],
        "row_contract": {
            "sample_row_schema": list(sample_row_schema),
            "sample_row_limit": sample_row_limit,
            "blocked_payload_fields": list(blocked_payload_fields),
            "all_manifest_row_contracts_present": all_manifest_row_contracts_present,
        },
        "proof_requirements": {
            "requires_metadata_preview": True,
            "requires_blocked_full_content_test": True,
            "requires_blocked_side_effect_test": True,
            "requires_acceptance_gate": True,
            "requires_execution_health_report": True,
            "requires_fresh_route_review": True,
        },
        "implementation_order": [
            "disabled metadata adapter",
            "read-only preview tool",
            "explicit-command routing only",
            "status/api adapter-state visibility",
            "focused personal/status smokes plus aggregate smoke",
            "separate approval-gated manifest for authenticated reads or side effects",
        ],
        "boundaries": {
            "natural_language_routing_enabled": False,
            "adapter_default_state": "disabled",
            "calls_external_service": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "writes_memory": False,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "authorizes_natural_language_routing": False,
            "authorizes_personal_data_read": False,
            "authorizes_side_effect": False,
            "authorizes_approval": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_external_service": False,
            "reusable_for_other_scope": False,
        },
        "next_safe_commands": next_safe_commands,
        "next_step_requires_adapter_probe": True,
        "next_step_requires_adapter_acceptance": True,
        "next_route_lock_requires_fresh_review": True,
    }


def _execution_matrix_handoff_payload(
    *,
    requested_connector: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    safest_commands = [row["safest_command"] for row in rows]
    return {
        "connector": requested_connector or "all",
        "connectors_reviewed": len(rows),
        "handoff_ready": True,
        "execution_matrix_handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "lane_rows": rows,
        "lanes": [row["lane"] for row in rows],
        "adapter_states": [row["adapter_state"] for row in rows],
        "safest_commands": safest_commands,
        "next_safe_commands": safest_commands + ["execution health report"],
        "global_connector_rule": {
            "metadata_or_draft_rehearsals_only": True,
            "personal_data_and_side_effects_require_exact_scope": True,
            "one_shot_approval_required_for_personal_data_or_side_effects": True,
            "approval_chain_proof_required": True,
            "audit_receipt_required": True,
            "verification_required": True,
            "rollback_or_stop_evidence_required": True,
            "natural_language_auto_routing_disabled_until_smokes_pass": True,
        },
        "proof_requirements": {
            "requires_preflight_contract": True,
            "requires_enablement_gate": True,
            "requires_rehearsal_receipt": True,
            "requires_metadata_preview_for_metadata_paths": True,
            "requires_smoke_tests": True,
            "requires_acceptance_gate": True,
            "requires_execution_health_report": True,
            "requires_fresh_route_review": True,
        },
        "blocked_test_requirements": [
            "missing scope",
            "full private content",
            "side effect without approval",
            "audit receipt visibility",
            "acceptance smoke",
        ],
        "boundaries": {
            "natural_language_routing_enabled": False,
            "calls_external_service": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "authorizes_natural_language_routing": False,
            "authorizes_personal_data_read": False,
            "authorizes_side_effect": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_external_service": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "reusable_for_other_scope": False,
        },
        "next_step_requires_preflight_contract": True,
        "next_step_requires_enablement_gate": True,
        "next_route_lock_requires_fresh_review": True,
    }


def _action_preview_handoff_payload(
    *,
    connector: str,
    requested_connector: str,
    action: str,
    suggested_risk: str,
    approval_required: bool,
    reason: str,
) -> dict[str, Any]:
    future_tool = f"future_{connector}_tool"
    next_commands = {
        "scope_packet": f"integration scope packet: {connector} -> {action}",
        "dry_run_contract": f"integration dry run contract: {connector} -> {action}",
        "boundary_contract": f"integration boundary contract: {connector}",
    }
    return {
        "connector": connector,
        "requested_connector": requested_connector,
        "action": action,
        "handoff_ready": True,
        "action_preview_handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "classification": {
            "suggested_risk": suggested_risk,
            "approval_required": approval_required,
            "requires_approval": approval_required,
            "reason": reason,
        },
        "safe_preview_packet": {
            "future_tool": future_tool,
            "requested_action": action,
            "scope_limit": "<selected item, narrow date range, named thread, or explicit recipient>",
            "safer_alternative": "paste relevant text, use metadata-only summary, or draft without sending",
        },
        "next_commands": next_commands,
        "next_safe_commands": list(next_commands.values()),
        "stop_conditions": {
            "requires_explicit_per_action_approval_for_side_effect": True,
            "requires_approval_chain_proof": True,
            "requires_exact_source_scope_for_private_reads": True,
            "stop_if_account_recipient_thread_event_or_date_range_ambiguous": True,
        },
        "boundaries": {
            "natural_language_routing_enabled": False,
            "calls_external_service": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "writes_memory": False,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "authorizes_natural_language_routing": False,
            "authorizes_personal_data_read": False,
            "authorizes_side_effect": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_external_service": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "reusable_for_other_scope": False,
        },
        "next_step_requires_scope_packet": True,
        "next_step_requires_dry_run_contract": True,
        "next_route_lock_requires_fresh_review": True,
    }


def _scope_packet_handoff_payload(
    *,
    connector: str,
    requested_connector: str,
    action: str,
    target: str,
    time_range: str,
    data_level: str,
    missing_fields: list[str],
    suggested_risk: str,
    approval_required: bool,
    reason: str,
    approval_command: str,
) -> dict[str, Any]:
    scope_ready = not missing_fields
    next_commands = {
        "action_preview": approval_command,
        "dry_run_contract": (
            f"integration dry run contract: {connector} -> {action}; "
            f"target {target or '<required>'}; time {time_range or '<required>'}; "
            f"data {data_level or '<required>'}"
        ),
        "runbook": (
            f"integration runbook: {connector} -> {action}; "
            f"target {target or '<required>'}; time {time_range or '<required>'}; "
            f"data {data_level or '<required>'}"
        ),
    }
    return {
        "connector": connector,
        "requested_connector": requested_connector,
        "action": action,
        "handoff_ready": True,
        "scope_packet_handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "scope_ready": scope_ready,
        "missing_fields": list(missing_fields),
        "scope": {
            "target": target,
            "time_range": time_range,
            "data_level": data_level,
        },
        "classification": {
            "suggested_risk": suggested_risk,
            "approval_required": approval_required,
            "requires_approval": approval_required,
            "reason": reason,
        },
        "approval_packet_seed": {
            "future_tool": f"future_{connector}_tool",
            "action": action,
            "connector": connector,
            "target": target or "<required before use>",
            "time_range": time_range or "<required before use>",
            "data_level": data_level or "<metadata-only, summary-only, full content, or side effect>",
            "preview_command": approval_command,
            "one_shot": True,
            "cannot_transfer_scope": True,
        },
        "next_commands": next_commands,
        "next_safe_commands": list(next_commands.values()),
        "stop_conditions": {
            "stop_if_account_recipient_thread_event_page_list_or_date_range_ambiguous": True,
            "stop_if_mutating_action_without_fresh_approval_receipt": True,
            "stop_if_private_source_scope_not_explicitly_approved": True,
            "requires_approval_chain_proof_for_real_use": approval_required,
        },
        "boundaries": {
            "natural_language_routing_enabled": False,
            "calls_external_service": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "writes_memory": False,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "authorizes_natural_language_routing": False,
            "authorizes_personal_data_read": False,
            "authorizes_side_effect": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_external_service": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "reusable_for_other_scope": False,
        },
        "next_step_requires_dry_run_contract": scope_ready,
        "next_step_requires_runbook": scope_ready,
        "next_route_lock_requires_fresh_review": True,
    }


def _readiness_report_handoff_payload(
    *,
    requested_connector: str,
    rows: list[dict[str, Any]],
    recommendations: list[str],
) -> dict[str, Any]:
    return {
        "connector": requested_connector or "all",
        "connectors_reviewed": len(rows),
        "handoff_ready": True,
        "readiness_report_handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "readiness_rows": rows,
        "recommendations": list(recommendations),
        "migration_order": [
            "browser public-page tools and Obsidian notes",
            "approval-gated reminder creation",
            "calendar free/busy or metadata-only summaries",
            "email/message/contact draft-only or user-pasted text workflows",
        ],
        "proof_requirements": {
            "requires_boundary_contract": True,
            "requires_action_preview": True,
            "requires_audit_logging": True,
            "requires_blocked_action_smoke_tests": True,
            "requires_fresh_route_review": True,
        },
        "routing_rule": {
            "start_with_narrow_read_only_or_draft_only_tools": True,
            "personal_data_reads_approval_gated": True,
            "side_effects_approval_gated": True,
            "natural_language_routing_enabled": False,
        },
        "boundaries": {
            "natural_language_routing_enabled": False,
            "calls_external_service": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "authorizes_natural_language_routing": False,
            "authorizes_personal_data_read": False,
            "authorizes_side_effect": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_external_service": False,
            "reusable_for_other_scope": False,
        },
        "next_step_requires_boundary_contract": True,
        "next_step_requires_action_preview": True,
        "next_route_lock_requires_fresh_review": True,
    }


def _migration_plan_handoff_payload(
    *,
    connector: str,
    requested_connector: str,
    profile: dict[str, list[str]],
) -> dict[str, Any]:
    phases = [
        {
            "phase": "boundary_contract",
            "requirements": [
                "define connector owner",
                "define account/source",
                "define data scope",
                "define off switch",
                "keep first implementation read-only unless already approval-gated",
                "document what can be cached and what stays transient",
            ],
        },
        {
            "phase": "risk_mapping",
            "requirements": [
                "map read-only candidates",
                "map personal-data surfaces",
                "map external side effects or high-risk actions",
            ],
        },
        {
            "phase": "implementation_checklist",
            "requirements": [
                "add one small tool with clear risk level",
                "use a narrow argument schema",
                "route natural-language commands only after smoke tests",
                "log every run through audit trail",
                "mirror blocked risky requests into pending approvals",
                "add safe and approval-gated examples",
            ],
        },
        {
            "phase": "verification",
            "requirements": [
                "focused read-only smoke test",
                "blocked personal-data or side-effect smoke test",
                "full aggregate smoke suite",
            ],
        },
    ]
    next_commands = {
        "boundary_contract": f"integration boundary contract: {connector}",
        "action_preview": f"integration action preview: {connector} -> {profile['read_only'][0]}",
        "execution_matrix": f"integration execution matrix: {connector}",
    }
    return {
        "connector": connector,
        "requested_connector": requested_connector,
        "handoff_ready": True,
        "migration_plan_handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "phases": phases,
        "phase_count": len(phases),
        "risk_map": {
            "read_only_candidates": list(profile["read_only"]),
            "personal_data_candidates": list(profile["personal_data"]),
            "side_effect_candidates": list(profile["side_effects"]),
            "read_only_count": len(profile["read_only"]),
            "personal_data_count": len(profile["personal_data"]),
            "side_effect_count": len(profile["side_effects"]),
        },
        "next_commands": next_commands,
        "next_safe_commands": list(next_commands.values()),
        "proof_requirements": {
            "requires_boundary_contract": True,
            "requires_narrow_tool_schema": True,
            "requires_smoke_tests_before_nl_routing": True,
            "requires_audit_trail": True,
            "requires_pending_approval_for_blocked_risky_request": True,
            "requires_read_only_smoke": True,
            "requires_blocked_action_smoke": True,
            "requires_aggregate_smoke": True,
            "requires_fresh_route_review": True,
        },
        "boundaries": {
            "natural_language_routing_enabled": False,
            "calls_external_service": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "writes_memory": False,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "authorizes_natural_language_routing": False,
            "authorizes_personal_data_read": False,
            "authorizes_side_effect": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_external_service": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "reusable_for_other_scope": False,
        },
        "next_step_requires_boundary_contract": True,
        "next_step_requires_action_preview": True,
        "next_route_lock_requires_fresh_review": True,
    }


def _boundary_contract_handoff_payload(
    *,
    connector: str,
    requested_connector: str,
    profile: dict[str, list[str]],
) -> dict[str, Any]:
    next_commands = {
        "migration_plan": f"integration migration plan: {connector}",
        "action_preview": f"integration action preview: {connector} -> {profile['read_only'][0]}",
        "scope_packet": f"integration scope packet: {connector} -> {profile['read_only'][0]}",
    }
    return {
        "connector": connector,
        "requested_connector": requested_connector,
        "handoff_ready": True,
        "boundary_contract_handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "default_state": {
            "connected_by_default": False,
            "account_tokens_available": False,
            "cookies_available": False,
            "mailbox_calendar_message_or_contact_data_available": False,
            "first_implementation_should_be_read_only_and_narrow": True,
        },
        "allowed_without_approval": [
            "show boundary contract",
            "draft migration plan",
            "draft reply/event/reminder text without sending or creating it",
            "summarize user-pasted text intentionally provided in chat",
        ],
        "approval_required": {
            "personal_data": list(profile["personal_data"]),
            "side_effects": list(profile["side_effects"]),
            "personal_data_count": len(profile["personal_data"]),
            "side_effect_count": len(profile["side_effects"]),
        },
        "read_only_candidates": list(profile["read_only"]),
        "read_only_candidate_count": len(profile["read_only"]),
        "approval_prompt_template": {
            "future_tool": f"future_{connector}_tool",
            "requested_action": "<exact read or side effect>",
            "data_source": "<account, list, thread, event, or conversation>",
            "scope_limit": "<time range, selected item, or explicit recipients>",
            "why_needed": "<user-visible reason>",
            "safer_alternative": "paste relevant text, use metadata-only summary, or draft without sending",
            "one_shot": True,
            "cannot_transfer_scope": True,
        },
        "caching_rules": {
            "allowed_memory": ["short summaries", "task references", "decisions", "user-approved notes"],
            "blocked_by_default": [
                "full email bodies",
                "message transcripts",
                "calendar notes",
                "contact cards",
                "tokens",
                "cookies",
                "attachments",
            ],
            "source_and_timestamp_required_for_saved_summary": True,
        },
        "audit_and_rollback": {
            "tool_run_audit_required": True,
            "side_effect_pending_approval_receipt_required": True,
            "read_only_preview_required_before_mutation": True,
            "cancel_or_dismiss_path_required_before_mutation": True,
        },
        "next_commands": next_commands,
        "next_safe_commands": list(next_commands.values()),
        "boundaries": {
            "natural_language_routing_enabled": False,
            "calls_external_service": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "writes_memory": False,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "authorizes_natural_language_routing": False,
            "authorizes_personal_data_read": False,
            "authorizes_side_effect": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_external_service": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "reusable_for_other_scope": False,
        },
        "next_step_requires_action_preview": True,
        "next_step_requires_scope_packet": True,
        "next_route_lock_requires_fresh_review": True,
    }


def _adapter_probe_handoff_payload(
    *,
    connector: str,
    requested_connector: str,
    action: str,
    adapter_name: str,
    fixture_name: str,
    adapter_state: str,
    probe_state: str,
    missing_fields: list[str],
    target: str,
    time_range: str,
    data_level: str,
    verification: str,
    metadata_only: bool,
    sample_rows: list[dict[str, str]],
    row_contract: dict[str, Any],
    suggested_risk: str,
    approval_required: bool,
    full_content_blocked: bool,
    side_effect_blocked: bool,
) -> dict[str, Any]:
    probe_ready = probe_state == "ADAPTER_PROBE_READY"
    next_safe_commands = [
        f"integration adapter manifest: {connector}",
        (
            f"integration metadata preview: {connector} -> {action}; "
            f"target {target or '<target>'}; time {time_range or '<time range>'}; "
            f"data metadata-only; verification {verification or 'fake rows only'}; "
            "tests blocked full-content smoke; audit adapter probe receipt; acceptance gate passed"
        ),
        "execution health report",
    ]
    return {
        "connector": connector,
        "requested_connector": requested_connector,
        "action": action,
        "adapter_name": adapter_name,
        "fixture_name": fixture_name,
        "adapter_state": adapter_state,
        "probe_state": probe_state,
        "handoff_ready": True,
        "adapter_probe_handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "adapter_probe_ready": probe_ready,
        "missing_fields": list(missing_fields),
        "scope": {
            "target": target,
            "time_range": time_range,
            "data_level": data_level,
            "verification": verification,
        },
        "probe_result": {
            "metadata_only": metadata_only,
            "sample_rows": sample_rows,
            "sample_row_count": len(sample_rows),
            "full_content_blocked": full_content_blocked,
            "side_effect_blocked": side_effect_blocked,
            "adapter_call_skipped": True,
            "disabled_fixture_used": probe_ready,
        },
        "row_contract": {
            "sample_row_schema": list(row_contract["sample_row_schema"]),
            "sample_row_schema_fields": list(row_contract["sample_row_schema_fields"]),
            "sample_row_limit": row_contract["sample_row_limit"],
            "sample_rows_bounded": row_contract["sample_rows_bounded"],
            "blocked_payload_fields": list(row_contract["blocked_payload_fields"]),
        },
        "next_proof_commands": {
            "adapter_manifest": next_safe_commands[0],
            "metadata_preview": next_safe_commands[1],
            "execution_health": next_safe_commands[2],
        },
        "next_safe_commands": next_safe_commands,
        "risk": {
            "suggested_risk": suggested_risk,
            "approval_required": approval_required,
            "requires_approval": approval_required,
        },
        "boundaries": {
            "natural_language_routing_enabled": False,
            "adapter_default_state": "disabled",
            "real_adapter_call_skipped": True,
            "calls_external_service": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "writes_memory": False,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "authorizes_natural_language_routing": False,
            "authorizes_personal_data_read": False,
            "authorizes_side_effect": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_external_service": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "reusable_for_other_scope": False,
        },
        "next_step_requires_metadata_preview": probe_ready,
        "next_step_requires_adapter_acceptance": probe_ready,
        "next_route_lock_requires_fresh_review": True,
    }


def _adapter_acceptance_handoff_payload(
    *,
    connector: str,
    requested_connector: str,
    adapter_name: str,
    acceptance_state: str,
    case_results: list[dict[str, Any]],
    passed_cases: int,
    sample_row_schema: list[str],
    sample_row_limit: int,
    blocked_payload_fields: list[str],
    all_case_row_contracts_present: bool,
) -> dict[str, Any]:
    acceptance_passed = acceptance_state == "ADAPTER_ACCEPTANCE_PASSED"
    receipt_sha256 = _adapter_acceptance_receipt_sha256(
        connector=connector,
        adapter_name=adapter_name,
        acceptance_state=acceptance_state,
        case_results=case_results,
        sample_row_schema=sample_row_schema,
        sample_row_limit=sample_row_limit,
        blocked_payload_fields=blocked_payload_fields,
    )
    receipt_contract_summary = [
        "proof-only",
        "enablement-review-required",
        "account-access-not-authorized",
        "route-unlock-not-authorized",
        "fresh-scope-review-required",
    ]
    receipt_boundary_rows = _adapter_acceptance_receipt_boundary_rows(receipt_sha256)
    next_safe_commands = [
        f"integration adapter probe: {connector} -> <metadata action>; target <target>; time <time range>; data metadata-only; verification fake rows only",
        f"integration enablement gate: {connector} -> <metadata action>; target <target>; time <time range>; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed",
        "execution health report",
    ]
    return {
        "connector": connector,
        "requested_connector": requested_connector,
        "adapter_name": adapter_name,
        "acceptance_state": acceptance_state,
        "handoff_ready": True,
        "adapter_acceptance_handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "adapter_acceptance_passed": acceptance_passed,
        "adapter_acceptance_receipt_sha256": receipt_sha256,
        "adapter_acceptance_receipt_present": True,
        "adapter_acceptance_receipt_boundary_rows": receipt_boundary_rows,
        "adapter_acceptance_receipt_boundary_row_count": len(receipt_boundary_rows),
        "adapter_acceptance_receipt_boundary_ready": True,
        "adapter_acceptance_receipt_contract_summary": receipt_contract_summary,
        "passed_cases": passed_cases,
        "total_cases": len(case_results),
        "case_results": case_results,
        "row_contract": {
            "sample_row_schema": list(sample_row_schema),
            "sample_row_limit": sample_row_limit,
            "blocked_payload_fields": list(blocked_payload_fields),
            "all_case_row_contracts_present": all_case_row_contracts_present,
        },
        "proof_rules": {
            "metadata_happy_path_requires_two_fake_rows": True,
            "full_content_must_be_blocked": True,
            "side_effect_must_be_blocked": True,
            "missing_scope_must_be_blocked": True,
        },
        "boundaries": {
            "natural_language_routing_enabled": False,
            "adapter_default_state": "disabled",
            "real_adapter_call_skipped": True,
            "calls_external_service": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "writes_memory": False,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "authorizes_natural_language_routing": False,
            "authorizes_personal_data_read": False,
            "authorizes_side_effect": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_external_service": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "reusable_for_other_scope": False,
        },
        "next_safe_commands": next_safe_commands,
        "requires_enablement_gate": True,
        "requires_execution_health_report": True,
        "next_route_lock_requires_fresh_review": True,
    }


def _adapter_acceptance_receipt_sha256(
    *,
    connector: str,
    adapter_name: str,
    acceptance_state: str,
    case_results: list[dict[str, Any]],
    sample_row_schema: list[str],
    sample_row_limit: int,
    blocked_payload_fields: list[str],
) -> str:
    case_parts = []
    for row in case_results:
        case_parts.append(
            "|".join(
                [
                    str(row.get("name", "")),
                    str(row.get("passed", False)),
                    str(row.get("probe_state", "")),
                    str(row.get("sample_row_count", 0)),
                    ",".join(str(field) for field in row.get("sample_row_schema", [])),
                    str(row.get("sample_row_limit", "")),
                    str(row.get("sample_rows_bounded", False)),
                    ",".join(str(field) for field in row.get("blocked_payload_fields", [])),
                    ",".join(str(field) for field in row.get("missing_fields", [])),
                    str(row.get("full_content_blocked", False)),
                    str(row.get("side_effect_blocked", False)),
                ]
            )
        )
    payload = "\n".join(
        [
            connector,
            adapter_name,
            acceptance_state,
            ",".join(sample_row_schema),
            str(sample_row_limit),
            ",".join(blocked_payload_fields),
            *case_parts,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _adapter_acceptance_receipt_boundary_rows(receipt_sha256: str) -> list[dict[str, Any]]:
    row_names = [
        "adapter_acceptance_receipt",
        "enablement_review_required",
        "account_access_not_authorized",
        "route_unlock_not_authorized",
        "fresh_scope_review_required",
    ]
    return [
        {
            "item": row_name,
            "token_sha256": receipt_sha256,
            "present": True,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "authorizes_natural_language_routing": False,
            "authorizes_personal_data_read": False,
            "authorizes_side_effect": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_external_service": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "reusable_for_other_scope": False,
        }
        for row_name in row_names
    ]


def _enablement_gate_handoff_payload(
    *,
    connector: str,
    requested_connector: str,
    action: str,
    tool_name: str,
    enablement_verdict: str,
    missing_fields: list[str],
    target: str,
    time_range: str,
    data_level: str,
    verification: str,
    rollback: str,
    tests: str,
    audit: str,
    acceptance: str,
    adapter_acceptance_required: bool,
    adapter_acceptance_state: str,
    adapter_acceptance_passed: bool,
    adapter_acceptance_case_count: int,
    adapter_acceptance_handoff: dict[str, Any] | None,
    metadata_row_contract_required: bool,
    metadata_row_contract_ready: bool,
    suggested_risk: str,
    approval_required: bool,
    rollback_required: bool,
    implementation_review_allowed: bool,
) -> dict[str, Any]:
    next_commands = {
        "preflight_contract": (
            f"integration preflight contract: {connector} -> {action}; "
            f"target {target or '<required>'}; time {time_range or '<required>'}; "
            f"data {data_level or '<required>'}; verification {verification or '<required>'}; "
            f"rollback {rollback or '<required>'}; tests {tests or '<required>'}; audit {audit or '<required>'}"
        ),
        "rehearsal_receipt": (
            f"integration rehearsal receipt: {connector} -> {action}; "
            f"target {target or '<required>'}; time {time_range or '<required>'}; "
            f"data {data_level or '<required>'}; verification {verification or '<required>'}; "
            f"rollback {rollback or '<required>'}; tests {tests or '<required>'}; "
            f"audit {audit or '<required>'}; acceptance {acceptance or '<required>'}"
        ),
        "proof_bundle": _integration_proof_bundle_command(
            connector,
            action,
            target,
            time_range,
            data_level,
            verification,
            rollback,
            tests,
            audit,
            acceptance,
        ),
    }
    return {
        "connector": connector,
        "requested_connector": requested_connector,
        "action": action,
        "tool_name": tool_name,
        "handoff_ready": True,
        "enablement_handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "enablement_verdict": enablement_verdict,
        "enablement_ready_for_review": implementation_review_allowed,
        "implementation_review_allowed": implementation_review_allowed,
        "missing_fields": list(missing_fields),
        "scope": {
            "target": target,
            "time_range": time_range,
            "data_level": data_level,
            "verification": verification,
            "rollback": rollback,
        },
        "proof": {
            "tests": tests,
            "audit": audit,
            "acceptance": acceptance,
        },
        "adapter_acceptance": {
            "required": adapter_acceptance_required,
            "state": adapter_acceptance_state,
            "passed": adapter_acceptance_passed,
            "passed_case_count": adapter_acceptance_case_count,
            "handoff": adapter_acceptance_handoff,
        },
        "metadata_row_contract": {
            "required": metadata_row_contract_required,
            "ready": metadata_row_contract_ready,
            "sample_row_schema": list(METADATA_ROW_SCHEMA),
            "sample_row_limit": METADATA_ROW_LIMIT,
            "blocked_payload_fields": list(BLOCKED_METADATA_PAYLOAD_FIELDS),
        },
        "risk": {
            "suggested_risk": suggested_risk,
            "approval_required": approval_required,
            "requires_approval": approval_required,
            "rollback_required": rollback_required,
        },
        "next_commands": next_commands,
        "next_safe_commands": list(next_commands.values()),
        "boundaries": {
            "natural_language_routing_enabled": False,
            "adapter_default_state": "disabled",
            "calls_external_service": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "writes_memory": False,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "authorizes_natural_language_routing": False,
            "authorizes_personal_data_read": False,
            "authorizes_side_effect": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_external_service": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "reusable_for_other_scope": False,
        },
        "enablement_gates": 7,
        "next_step_requires_rehearsal_receipt": True,
        "next_step_requires_proof_bundle": True,
        "next_route_lock_requires_fresh_review": True,
    }


def _implementation_spec_handoff_payload(
    *,
    connector: str,
    requested_connector: str,
    action: str,
    tool_name: str,
    spec_state: str,
    missing_fields: list[str],
    target: str,
    time_range: str,
    data_level: str,
    verification: str,
    rollback: str,
    tests: str,
    audit: str,
    endpoint: str,
    planner_route: str,
    risk_level: str,
    suggested_risk: str,
    approval_required: bool,
    rollback_required: bool,
    metadata_row_contract_required: bool,
    metadata_row_contract_ready: bool,
) -> dict[str, Any]:
    spec_ready = spec_state == "SPEC_READY_FOR_REVIEW"
    next_commands = {
        "promotion_gate": (
            f"integration promotion gate: {connector} -> {action}; "
            f"target {target or '<required>'}; time {time_range or '<required>'}; "
            f"data {data_level or '<required>'}; verification {verification or '<required>'}; "
            f"rollback {rollback or '<required>'}; tests {tests or '<required>'}; audit {audit or '<required>'}"
        ),
        "preflight_contract": (
            f"integration preflight contract: {connector} -> {action}; "
            f"target {target or '<required>'}; time {time_range or '<required>'}; "
            f"data {data_level or '<required>'}; verification {verification or '<required>'}; "
            f"rollback {rollback or '<required>'}; tests {tests or '<required>'}; audit {audit or '<required>'}"
        ),
        "enablement_gate": (
            f"integration enablement gate: {connector} -> {action}; "
            f"target {target or '<required>'}; time {time_range or '<required>'}; "
            f"data {data_level or '<required>'}; verification {verification or '<required>'}"
        ),
    }
    return {
        "connector": connector,
        "requested_connector": requested_connector,
        "action": action,
        "tool_name": tool_name,
        "handoff_ready": True,
        "implementation_spec_handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "spec_state": spec_state,
        "spec_ready_for_review": spec_ready,
        "implementation_spec_version": 1,
        "missing_fields": list(missing_fields),
        "implementation_contract": {
            "tool_name": tool_name,
            "toolset": "personal",
            "tool_registry_risk": risk_level,
            "planner_route": planner_route,
            "optional_status_api": endpoint,
            "default_enabled_state": "disabled for natural-language auto-routing until smoke tests pass",
            "natural_language_routing_enabled": False,
            "output_shape": "concise user-facing summary plus bounded metadata, never raw private payloads by default",
        },
        "scope": {
            "target": target,
            "time_range": time_range,
            "data_level": data_level,
            "verification": verification,
            "rollback": rollback,
        },
        "proof": {
            "tests": tests,
            "audit": audit,
        },
        "metadata_row_contract": {
            "required": metadata_row_contract_required,
            "ready": metadata_row_contract_ready,
            "sample_row_schema": list(METADATA_ROW_SCHEMA),
            "sample_row_limit": METADATA_ROW_LIMIT,
            "blocked_payload_fields": list(BLOCKED_METADATA_PAYLOAD_FIELDS),
        },
        "risk": {
            "suggested_risk": suggested_risk,
            "approval_required": approval_required,
            "requires_approval": approval_required,
            "rollback_required": rollback_required,
        },
        "execution_gates": {
            "preflight_required": True,
            "read_only_path_guarded": True,
            "personal_data_requires_one_shot_approval": True,
            "side_effect_requires_one_shot_approval": True,
            "verification_required_before_completion": True,
            "rollback_requires_fresh_approval_for_side_effect": True,
        },
        "audit_contract": {
            "implementation_spec_version": 1,
            "bounded_metadata_only": True,
            "metadata_row_schema": list(METADATA_ROW_SCHEMA),
            "metadata_row_limit": METADATA_ROW_LIMIT,
            "blocked_payload_fields": list(BLOCKED_METADATA_PAYLOAD_FIELDS),
        },
        "next_commands": next_commands,
        "next_safe_commands": list(next_commands.values()),
        "boundaries": {
            "natural_language_routing_enabled": False,
            "adapter_default_state": "disabled",
            "calls_external_service": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "writes_memory": False,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "authorizes_natural_language_routing": False,
            "authorizes_personal_data_read": False,
            "authorizes_side_effect": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_external_service": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "reusable_for_other_scope": False,
        },
        "next_step_requires_preflight_contract": True,
        "next_step_requires_enablement_gate": True,
        "next_route_lock_requires_fresh_review": True,
    }


def _promotion_handoff_payload(
    *,
    connector: str,
    requested_connector: str,
    action: str,
    promotion_verdict: str,
    next_move: str,
    runbook_command: str,
    missing_fields: list[str],
    target: str,
    time_range: str,
    data_level: str,
    verification: str,
    rollback: str,
    tests: str,
    audit: str,
    suggested_risk: str,
    approval_required: bool,
    rollback_required: bool,
    metadata_row_contract_required: bool,
    metadata_row_contract_ready: bool,
    implementation_allowed: bool,
) -> dict[str, Any]:
    next_commands = {
        "runbook": runbook_command,
        "implementation_spec": (
            f"integration implementation spec: {connector} -> {action}; "
            f"target {target or '<required>'}; time {time_range or '<required>'}; "
            f"data {data_level or '<required>'}; verification {verification or '<required>'}; "
            f"rollback {rollback or '<required>'}; tests {tests or '<required>'}; audit {audit or '<required>'}"
        ),
        "preflight_contract": (
            f"integration preflight contract: {connector} -> {action}; "
            f"target {target or '<required>'}; time {time_range or '<required>'}; "
            f"data {data_level or '<required>'}; verification {verification or '<required>'}"
        ),
    }
    return {
        "connector": connector,
        "requested_connector": requested_connector,
        "action": action,
        "handoff_ready": True,
        "promotion_handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "promotion_verdict": promotion_verdict,
        "implementation_allowed": implementation_allowed,
        "next_move": next_move,
        "runbook_command": runbook_command,
        "missing_fields": list(missing_fields),
        "evidence": {
            "target": target,
            "time_range": time_range,
            "data_level": data_level,
            "verification": verification,
            "rollback": rollback,
            "tests": tests,
            "audit": audit,
        },
        "metadata_row_contract": {
            "required": metadata_row_contract_required,
            "ready": metadata_row_contract_ready,
            "sample_row_schema": list(METADATA_ROW_SCHEMA),
            "sample_row_limit": METADATA_ROW_LIMIT,
            "blocked_payload_fields": list(BLOCKED_METADATA_PAYLOAD_FIELDS),
        },
        "risk": {
            "suggested_risk": suggested_risk,
            "approval_required": approval_required,
            "requires_approval": approval_required,
            "rollback_required": rollback_required,
        },
        "implementation_invariants": {
            "narrow_tool_registry_risk_required": True,
            "natural_language_routing_enabled": False,
            "bounded_audit_summary_only": True,
            "metadata_rows_bounded_until_review": True,
            "personal_data_and_side_effects_need_approval_chain": True,
        },
        "smoke_tests_required": {
            "happy_path": tests,
            "blocked_scope_path": True,
            "approval_path_for_private_or_side_effect": approval_required,
            "audit_path": True,
        },
        "next_commands": next_commands,
        "next_safe_commands": list(next_commands.values()),
        "boundaries": {
            "natural_language_routing_enabled": False,
            "adapter_default_state": "disabled",
            "calls_external_service": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "writes_memory": False,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "authorizes_natural_language_routing": False,
            "authorizes_personal_data_read": False,
            "authorizes_side_effect": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_external_service": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "reusable_for_other_scope": False,
        },
        "next_step_requires_implementation_spec": implementation_allowed,
        "next_step_requires_preflight_contract": implementation_allowed,
        "next_route_lock_requires_fresh_review": True,
    }


def _runbook_handoff_payload(
    *,
    connector: str,
    requested_connector: str,
    action: str,
    readiness: str,
    preview_command: str,
    verifier: str,
    rollback_plan: str,
    missing_fields: list[str],
    target: str,
    time_range: str,
    data_level: str,
    verification: str,
    rollback: str,
    suggested_risk: str,
    approval_required: bool,
    rollback_required: bool,
    metadata_row_contract_required: bool,
    metadata_row_contract_ready: bool,
) -> dict[str, Any]:
    ready_for_dry_run = readiness == "READY_FOR_DRY_RUN"
    next_commands = {
        "scope_packet": f"integration scope packet: {connector} -> {action}",
        "dry_run_contract": preview_command,
        "promotion_gate": (
            f"integration promotion gate: {connector} -> {action}; "
            f"target {target or '<required>'}; time {time_range or '<required>'}; "
            f"data {data_level or '<required>'}; verification {verification or '<required>'}; "
            f"rollback {rollback or '<required>'}"
        ),
    }
    return {
        "connector": connector,
        "requested_connector": requested_connector,
        "action": action,
        "handoff_ready": True,
        "runbook_handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "readiness": readiness,
        "ready_for_dry_run": ready_for_dry_run,
        "missing_fields": list(missing_fields),
        "scope_lock": {
            "target": target,
            "time_range": time_range,
            "data_level": data_level,
            "verification": verification,
            "rollback": rollback,
            "verifier": verifier,
            "rollback_plan": rollback_plan,
        },
        "preflight": {
            "connector_risk_declared": True,
            "exact_scope_required": True,
            "fresh_runbook_required_if_request_changes": True,
            "metadata_row_contract_required": metadata_row_contract_required,
            "metadata_row_contract_ready": metadata_row_contract_ready,
            "prefer_metadata_or_draft_only": True,
        },
        "approval_gate": {
            "preview_command": preview_command,
            "required_for_real_execution": approval_required,
            "one_shot": True,
            "cannot_transfer_scope": True,
        },
        "execution_plan": {
            "future_tool": f"future_{connector}_connector",
            "dry_run_output": "planned data touched, planned side effect, expected result, and verifier",
            "real_run_output": "connector response plus audit id, timestamp, scope, and approval id if one was required",
        },
        "verification": {
            "primary_verifier": verifier,
            "recent_tool_runs_required": True,
            "runtime_trace_required": True,
            "verification_receipt_required_before_completion": True,
        },
        "rollback_and_recovery": {
            "rollback_plan": rollback_plan,
            "recovery_packet_required_on_verification_failure": True,
            "corrective_side_effect_requires_fresh_approval": True,
        },
        "metadata_row_contract": {
            "required": metadata_row_contract_required,
            "ready": metadata_row_contract_ready,
            "sample_row_schema": list(METADATA_ROW_SCHEMA),
            "sample_row_limit": METADATA_ROW_LIMIT,
            "blocked_payload_fields": list(BLOCKED_METADATA_PAYLOAD_FIELDS),
        },
        "risk": {
            "suggested_risk": suggested_risk,
            "approval_required": approval_required,
            "requires_approval": approval_required,
            "rollback_required": rollback_required,
        },
        "next_commands": next_commands,
        "next_safe_commands": list(next_commands.values()),
        "boundaries": {
            "natural_language_routing_enabled": False,
            "adapter_default_state": "disabled",
            "calls_external_service": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "writes_memory": False,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "authorizes_natural_language_routing": False,
            "authorizes_personal_data_read": False,
            "authorizes_side_effect": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_external_service": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "reusable_for_other_scope": False,
        },
        "next_step_requires_dry_run_contract": ready_for_dry_run,
        "next_step_requires_promotion_gate": ready_for_dry_run,
        "next_route_lock_requires_fresh_review": True,
    }


def _dry_run_handoff_payload(
    *,
    connector: str,
    requested_connector: str,
    action: str,
    execution_state: str,
    dry_run_ready: bool,
    one_shot_seed: str,
    missing_fields: list[str],
    target: str,
    time_range: str,
    data_level: str,
    verification: str,
    suggested_risk: str,
    approval_required: bool,
    metadata_row_contract_required: bool,
    metadata_row_contract_ready: bool,
) -> dict[str, Any]:
    next_commands = {
        "action_preview": f"integration action preview: {connector} -> {action}",
        "scope_packet": f"integration scope packet: {connector} -> {action}",
        "runbook": (
            f"integration runbook: {connector} -> {action}; "
            f"target {target or '<required>'}; time {time_range or '<required>'}; "
            f"data {data_level or '<required>'}; verification {verification or '<required>'}"
        ),
        "promotion_gate": (
            f"integration promotion gate: {connector} -> {action}; "
            f"target {target or '<required>'}; time {time_range or '<required>'}; "
            f"data {data_level or '<required>'}; verification {verification or '<required>'}"
        ),
    }
    return {
        "connector": connector,
        "requested_connector": requested_connector,
        "action": action,
        "handoff_ready": True,
        "dry_run_handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "execution_state": execution_state,
        "dry_run_ready": dry_run_ready,
        "missing_fields": list(missing_fields),
        "scope": {
            "target": target,
            "time_range": time_range,
            "data_level": data_level,
            "verification": verification,
        },
        "harness_receipt": {
            "future_tool": f"future_{connector}_connector",
            "preflight": "classify risk, confirm scope, confirm connector availability, prepare audit id",
            "dry_run_only": "return planned request, expected data touched, expected side effect, and verification target",
            "real_run_gate": "one-shot approval receipt plus approval chain proof required" if approval_required else "allowed only if implementation stays read-only and metadata-only",
            "post_run_verifier": verification or "<required before real connector use>",
        },
        "one_shot_approval_seed": {
            "command": one_shot_seed,
            "approval_required_for_real_use": approval_required,
            "one_shot": True,
            "cannot_transfer_scope": True,
        },
        "metadata_row_contract": {
            "required": metadata_row_contract_required,
            "ready": metadata_row_contract_ready,
            "sample_row_schema": list(METADATA_ROW_SCHEMA),
            "sample_row_limit": METADATA_ROW_LIMIT,
            "blocked_payload_fields": list(BLOCKED_METADATA_PAYLOAD_FIELDS),
        },
        "risk": {
            "suggested_risk": suggested_risk,
            "approval_required": approval_required,
            "requires_approval": approval_required,
        },
        "next_commands": next_commands,
        "next_safe_commands": list(next_commands.values()),
        "boundaries": {
            "natural_language_routing_enabled": False,
            "adapter_default_state": "disabled",
            "calls_external_service": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "writes_memory": False,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "authorizes_natural_language_routing": False,
            "authorizes_personal_data_read": False,
            "authorizes_side_effect": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_external_service": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "reusable_for_other_scope": False,
        },
        "next_step_requires_runbook": dry_run_ready,
        "next_step_requires_promotion_gate": dry_run_ready,
        "next_route_lock_requires_fresh_review": True,
    }


def _preflight_handoff_payload(
    *,
    connector: str,
    requested_connector: str,
    action: str,
    tool_name: str,
    preflight_state: str,
    exact_args: dict[str, str],
    missing_fields: list[str],
    target: str,
    time_range: str,
    data_level: str,
    verification: str,
    rollback: str,
    tests: str,
    audit: str,
    approval_packet: str,
    suggested_risk: str,
    approval_required: bool,
    rollback_required: bool,
    metadata_row_contract_required: bool,
    metadata_row_contract_ready: bool,
) -> dict[str, Any]:
    preflight_ready = preflight_state == "PREFLIGHT_READY_FOR_REVIEW"
    next_commands = {
        "implementation_spec": (
            f"integration implementation spec: {connector} -> {action}; "
            f"target {target or '<required>'}; time {time_range or '<required>'}; "
            f"data {data_level or '<required>'}; verification {verification or '<required>'}; "
            f"rollback {rollback or '<required>'}; tests {tests or '<required>'}; audit {audit or '<required>'}"
        ),
        "enablement_gate": (
            f"integration enablement gate: {connector} -> {action}; "
            f"target {target or '<required>'}; time {time_range or '<required>'}; "
            f"data {data_level or '<required>'}; verification {verification or '<required>'}; "
            f"rollback {rollback or '<required>'}; tests {tests or '<required>'}; audit {audit or '<required>'}"
        ),
        "rehearsal_receipt": (
            f"integration rehearsal receipt: {connector} -> {action}; "
            f"target {target or '<required>'}; time {time_range or '<required>'}; "
            f"data {data_level or '<required>'}; verification {verification or '<required>'}"
        ),
    }
    return {
        "connector": connector,
        "requested_connector": requested_connector,
        "action": action,
        "tool_name": tool_name,
        "handoff_ready": True,
        "preflight_handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "preflight_state": preflight_state,
        "preflight_ready_for_review": preflight_ready,
        "missing_fields": list(missing_fields),
        "exact_args": dict(exact_args),
        "scope": {
            "target": target,
            "time_range": time_range,
            "data_level": data_level,
            "verification": verification,
            "rollback": rollback,
        },
        "proof": {
            "tests": tests,
            "audit": audit,
        },
        "approval_packet": {
            "seed": approval_packet,
            "one_shot": True,
            "cannot_transfer_scope": True,
            "required_for_real_use": approval_required,
        },
        "metadata_row_contract": {
            "required": metadata_row_contract_required,
            "ready": metadata_row_contract_ready,
            "sample_row_schema": list(METADATA_ROW_SCHEMA),
            "sample_row_limit": METADATA_ROW_LIMIT,
            "blocked_payload_fields": list(BLOCKED_METADATA_PAYLOAD_FIELDS),
        },
        "risk": {
            "suggested_risk": suggested_risk,
            "approval_required": approval_required,
            "requires_approval": approval_required,
            "rollback_required": rollback_required,
        },
        "enablement_gates": {
            "count": 6,
            "implementation_spec_exists": True,
            "focused_smoke_required": True,
            "blocked_action_smoke_required": True,
            "approval_readiness_required_for_private_or_side_effect": approval_required,
            "audit_row_required": True,
            "verification_receipt_required": True,
        },
        "required_proof_before_enabling": {
            "smoke_tests": tests,
            "audit_plan": audit,
            "metadata_row_contract_required": metadata_row_contract_required,
            "compile_command": "python3 -m compileall -q jarvis_v2",
            "focused_personal_smoke_command": "python3 -m jarvis_v2.scripts.smoke_test_personal",
        },
        "next_commands": next_commands,
        "next_safe_commands": list(next_commands.values()),
        "boundaries": {
            "natural_language_routing_enabled": False,
            "adapter_default_state": "disabled",
            "calls_external_service": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "writes_memory": False,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "authorizes_natural_language_routing": False,
            "authorizes_personal_data_read": False,
            "authorizes_side_effect": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_external_service": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "reusable_for_other_scope": False,
        },
        "next_step_requires_enablement_gate": True,
        "next_step_requires_rehearsal_receipt": True,
        "next_step_requires_proof_bundle": True,
        "next_route_lock_requires_fresh_review": True,
    }


def _rehearsal_handoff_payload(
    *,
    connector: str,
    requested_connector: str,
    action: str,
    tool_name: str,
    rehearsal_state: str,
    adapter_state: str,
    simulated_result: str,
    missing_fields: list[str],
    target: str,
    time_range: str,
    data_level: str,
    verification: str,
    rollback: str,
    tests: str,
    audit: str,
    acceptance: str,
    metadata_row_contract_required: bool,
    metadata_row_contract_ready: bool,
    suggested_risk: str,
    approval_required: bool,
    rollback_required: bool,
) -> dict[str, Any]:
    rehearsal_ready = rehearsal_state in {"REHEARSAL_READY_APPROVAL_GATED", "REHEARSAL_READY_READ_ONLY"}
    implementation_review_command = _integration_implementation_review_command(
        connector,
        action,
        target,
        time_range,
        data_level,
        verification,
        rollback,
        tests,
        audit,
        acceptance,
        "",
    )
    next_commands = {
        "proof_bundle": _integration_proof_bundle_command(
            connector,
            action,
            target,
            time_range,
            data_level,
            verification,
            rollback,
            tests,
            audit,
            acceptance,
        ),
        "implementation_review": implementation_review_command,
        "route_lock": (
            f"integration route lock: {connector} -> {action}; target {target or '<target>'}; "
            f"time {time_range or '<time range>'}; data {data_level or '<data level>'}; "
            f"verification {verification or '<verification>'}; rollback {rollback or 'no connector call'}; "
            f"tests {tests or '<tests>'}; audit {audit or '<audit>'}; acceptance {acceptance or '<acceptance>'}; "
            "status <status API and smoke evidence>; expected_scope_hash <scope hash>"
        ),
    }
    return {
        "connector": connector,
        "requested_connector": requested_connector,
        "action": action,
        "tool_name": tool_name,
        "handoff_ready": True,
        "rehearsal_handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "rehearsal_state": rehearsal_state,
        "rehearsal_ready": rehearsal_ready,
        "adapter_state": adapter_state,
        "simulated_result": simulated_result,
        "missing_fields": list(missing_fields),
        "scope": {
            "target": target,
            "time_range": time_range,
            "data_level": data_level,
            "verification": verification,
            "rollback": rollback,
        },
        "proof": {
            "tests": tests,
            "audit": audit,
            "acceptance": acceptance,
        },
        "metadata_row_contract": {
            "required": metadata_row_contract_required,
            "ready": metadata_row_contract_ready,
            "sample_row_schema": list(METADATA_ROW_SCHEMA),
            "sample_row_limit": METADATA_ROW_LIMIT,
            "blocked_payload_fields": list(BLOCKED_METADATA_PAYLOAD_FIELDS),
        },
        "risk": {
            "suggested_risk": suggested_risk,
            "approval_required": approval_required,
            "requires_approval": approval_required,
            "rollback_required": rollback_required,
        },
        "lifecycle": {
            "preflight": "normalize connector/action, classify risk, validate exact scope",
            "gate": "stop if scope, tests, audit, acceptance, verification, or rollback proof is missing",
            "approval": "required before adapter call in any real run" if approval_required else "not required for read-only rehearsal",
            "adapter_boundary": "real connector adapter remains disabled",
            "verification": "compare simulated result with declared verification target without claiming real completion",
            "audit": "record only bounded rehearsal metadata, not raw private payloads",
        },
        "boundaries": {
            "natural_language_routing_enabled": False,
            "adapter_default_state": "disabled",
            "real_adapter_call_skipped": True,
            "calls_external_service": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "writes_memory": False,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "authorizes_natural_language_routing": False,
            "authorizes_personal_data_read": False,
            "authorizes_side_effect": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_external_service": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "reusable_for_other_scope": False,
        },
        "next_commands": next_commands,
        "next_safe_commands": list(next_commands.values()),
        "next_step_requires_proof_bundle": True,
        "next_route_lock_requires_fresh_review": True,
    }


def _ordered_commands(commands: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for command in commands:
        if command and command not in seen:
            seen.add(command)
            ordered.append(command)
    return ordered


def _integration_scope_hash(
    connector: str,
    action: str,
    target: str,
    time_range: str,
    data_level: str,
    verification: str,
    rollback: str = "",
    tests: str = "",
    audit: str = "",
    acceptance: str = "",
    status: str = "",
) -> str:
    material = "|".join(
        [
            connector.strip().lower(),
            action.strip().lower(),
            target.strip().lower(),
            time_range.strip().lower(),
            data_level.strip().lower(),
            verification.strip().lower(),
            rollback.strip().lower(),
            tests.strip().lower(),
            audit.strip().lower(),
            acceptance.strip().lower(),
            status.strip().lower(),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _integration_proof_bundle_command(
    connector: str,
    action: str,
    target: str,
    time_range: str,
    data_level: str,
    verification: str,
    rollback: str,
    tests: str,
    audit: str,
    acceptance: str,
) -> str:
    return (
        f"integration proof bundle: {connector} -> {action}; target {target or '<target>'}; "
        f"time {time_range or '<time range>'}; data {data_level or '<data level>'}; "
        f"verification {verification or '<verification>'}; rollback {rollback or 'no connector call'}; "
        f"tests {tests or '<tests>'}; audit {audit or '<audit>'}; acceptance {acceptance or '<acceptance>'}"
    )


def _integration_implementation_review_command(
    connector: str,
    action: str,
    target: str,
    time_range: str,
    data_level: str,
    verification: str,
    rollback: str,
    tests: str,
    audit: str,
    acceptance: str,
    status: str,
) -> str:
    return (
        f"integration implementation review: {connector} -> {action}; target {target or '<target>'}; "
        f"time {time_range or '<time range>'}; data {data_level or '<data level>'}; "
        f"verification {verification or '<verification>'}; rollback {rollback or 'no connector call'}; "
        f"tests {tests or '<tests>'}; audit {audit or '<audit>'}; acceptance {acceptance or '<acceptance>'}; "
        f"status {status or '<status API and smoke evidence>'}"
    )


def _valid_integration_scope_hash(value: str) -> bool:
    return len(value) == 16 and all(char in "0123456789abcdefABCDEF" for char in value)


def _implementation_review_receipt_contract_rows(
    *,
    review_ready: bool,
    scope_hash: str,
    review_receipt_id: str,
) -> list[dict[str, Any]]:
    status = "ready" if review_ready else "blocked"
    return [
        {
            "item": "exact_scope_hash",
            "required": True,
            "status": "ready" if scope_hash else "blocked",
            "scope_hash": scope_hash,
            "review_receipt_id": review_receipt_id,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "reusable_for_other_scope": False,
        },
        {
            "item": "metadata_only_output_boundary",
            "required": True,
            "status": status,
            "scope_hash": scope_hash,
            "review_receipt_id": review_receipt_id,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "reusable_for_other_scope": False,
        },
        {
            "item": "status_api_smoke_evidence",
            "required": True,
            "status": status,
            "scope_hash": scope_hash,
            "review_receipt_id": review_receipt_id,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "reusable_for_other_scope": False,
        },
        {
            "item": "blocked_private_payload_fields",
            "required": True,
            "status": status,
            "scope_hash": scope_hash,
            "review_receipt_id": review_receipt_id,
            "blocked_payload_fields": list(BLOCKED_METADATA_PAYLOAD_FIELDS),
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "reusable_for_other_scope": False,
        },
        {
            "item": "fresh_route_lock_review_required",
            "required": True,
            "status": status,
            "scope_hash": scope_hash,
            "review_receipt_id": review_receipt_id,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "reusable_for_other_scope": False,
        },
    ]


def _implementation_review_receipt_contract_ready(
    rows: Any,
    *,
    expected_ready: bool | None = None,
    scope_hash: str = "",
    review_receipt_id: str = "",
) -> bool:
    if not isinstance(rows, list) or len(rows) != 5:
        return False
    expected_items = [
        "exact_scope_hash",
        "metadata_only_output_boundary",
        "status_api_smoke_evidence",
        "blocked_private_payload_fields",
        "fresh_route_lock_review_required",
    ]
    expected_status = None if expected_ready is None else ("ready" if expected_ready else "blocked")
    for index, (row, item) in enumerate(zip(rows, expected_items)):
        if not isinstance(row, dict):
            return False
        if row.get("item") != item or row.get("required") is not True:
            return False
        if scope_hash and row.get("scope_hash") != scope_hash:
            return False
        if review_receipt_id and row.get("review_receipt_id") != review_receipt_id:
            return False
        if expected_status is not None:
            if index == 0:
                expected_first_status = "ready" if scope_hash else "blocked"
                if row.get("status") != expected_first_status:
                    return False
            elif row.get("status") != expected_status:
                return False
        elif row.get("status") not in {"ready", "blocked"}:
            return False
        if item == "blocked_private_payload_fields" and row.get("blocked_payload_fields") != list(BLOCKED_METADATA_PAYLOAD_FIELDS):
            return False
        for key in ["authorizes_account_access", "authorizes_route_unlock", "reusable_for_other_scope"]:
            if row.get(key) is not False:
                return False
    return True


def _route_review_contract_rows(
    *,
    route_unlock_candidate: bool,
    scope_hash: str,
    review_receipt_id: str,
    route_lock_state: str,
) -> list[dict[str, Any]]:
    status = "ready_for_explicit_review" if route_unlock_candidate else "held"
    base = {
        "scope_hash": scope_hash,
        "review_receipt_id": review_receipt_id,
        "route_lock_state": route_lock_state,
        "authorizes_account_access": False,
        "authorizes_route_unlock": False,
        "authorizes_natural_language_routing": False,
        "authorizes_personal_data_read": False,
        "authorizes_side_effect": False,
        "reusable_for_other_scope": False,
    }
    return [
        {
            **base,
            "item": "implementation_review_receipt",
            "required": True,
            "status": status if review_receipt_id else "held",
        },
        {
            **base,
            "item": "exact_scope_hash",
            "required": True,
            "status": status if scope_hash else "held",
        },
        {
            **base,
            "item": "status_api_smoke_evidence",
            "required": True,
            "status": status,
        },
        {
            **base,
            "item": "metadata_row_contract",
            "required": True,
            "status": status,
            "allowed_fields": list(METADATA_ROW_SCHEMA),
            "row_limit": METADATA_ROW_LIMIT,
        },
        {
            **base,
            "item": "explicit_route_review_required",
            "required": True,
            "status": status,
        },
        {
            **base,
            "item": "approval_chain_boundary",
            "required": True,
            "status": status,
        },
    ]


def _route_review_contract_ready(
    rows: Any,
    *,
    route_unlock_candidate: bool | None = None,
    scope_hash: str = "",
    review_receipt_id: str = "",
    route_lock_state: str = "",
) -> bool:
    if not isinstance(rows, list) or len(rows) != 6:
        return False
    expected_items = [
        "implementation_review_receipt",
        "exact_scope_hash",
        "status_api_smoke_evidence",
        "metadata_row_contract",
        "explicit_route_review_required",
        "approval_chain_boundary",
    ]
    expected_status = None
    if route_unlock_candidate is not None:
        expected_status = "ready_for_explicit_review" if route_unlock_candidate else "held"
    for index, (row, item) in enumerate(zip(rows, expected_items)):
        if not isinstance(row, dict):
            return False
        if row.get("item") != item or row.get("required") is not True:
            return False
        if scope_hash and row.get("scope_hash") != scope_hash:
            return False
        if review_receipt_id and row.get("review_receipt_id") != review_receipt_id:
            return False
        if route_lock_state and row.get("route_lock_state") != route_lock_state:
            return False
        if expected_status is not None:
            if index == 0:
                if row.get("status") != (expected_status if review_receipt_id else "held"):
                    return False
            elif index == 1:
                if row.get("status") != (expected_status if scope_hash else "held"):
                    return False
            elif row.get("status") != expected_status:
                return False
        elif row.get("status") not in {"ready_for_explicit_review", "held"}:
            return False
        if item == "metadata_row_contract":
            if row.get("allowed_fields") != list(METADATA_ROW_SCHEMA) or row.get("row_limit") != METADATA_ROW_LIMIT:
                return False
        for key in [
            "authorizes_account_access",
            "authorizes_route_unlock",
            "authorizes_natural_language_routing",
            "authorizes_personal_data_read",
            "authorizes_side_effect",
            "reusable_for_other_scope",
        ]:
            if row.get(key) is not False:
                return False
    return True


def _route_lock_token_sha256(
    *,
    scope_hash: str,
    review_receipt_id: str,
    route_lock_state: str,
    expected_scope_hash: str,
    implementation_review_receipt_contract_rows: list[dict[str, Any]],
    route_review_contract_rows: list[dict[str, Any]],
) -> str:
    implementation_receipt_entries = [
        "|".join(
            [
                str(row.get("item") or ""),
                str(row.get("status") or ""),
                str(row.get("scope_hash") or ""),
                str(row.get("review_receipt_id") or ""),
                str(_metadata_bool(row.get("authorizes_account_access"))),
                str(_metadata_bool(row.get("authorizes_route_unlock"))),
                str(_metadata_bool(row.get("reusable_for_other_scope"))),
            ]
        )
        for row in implementation_review_receipt_contract_rows
    ]
    contract_entries = [
        "|".join(
            [
                str(row.get("item") or ""),
                str(row.get("status") or ""),
                str(row.get("scope_hash") or ""),
                str(row.get("review_receipt_id") or ""),
                str(row.get("route_lock_state") or ""),
                str(_metadata_bool(row.get("authorizes_account_access"))),
                str(_metadata_bool(row.get("authorizes_route_unlock"))),
                str(_metadata_bool(row.get("authorizes_natural_language_routing"))),
                str(_metadata_bool(row.get("authorizes_personal_data_read"))),
                str(_metadata_bool(row.get("authorizes_side_effect"))),
                str(_metadata_bool(row.get("reusable_for_other_scope"))),
            ]
        )
        for row in route_review_contract_rows
    ]
    material = "|".join(
        [
            "integration-route-lock",
            scope_hash.strip().lower(),
            review_receipt_id.strip().lower(),
            route_lock_state.strip().lower(),
            expected_scope_hash.strip().lower(),
            "\n".join(implementation_receipt_entries),
            "\n".join(contract_entries),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _route_lock_token_boundary_rows(
    *,
    token_sha256: str,
    scope_hash: str,
    review_receipt_id: str,
    route_lock_state: str,
) -> list[dict[str, Any]]:
    base = {
        "scope_hash": scope_hash,
        "review_receipt_id": review_receipt_id,
        "route_lock_state": route_lock_state,
        "token_sha256": token_sha256,
        "authorizes_account_access": False,
        "authorizes_route_unlock": False,
        "authorizes_natural_language_routing": False,
        "authorizes_personal_data_read": False,
        "authorizes_side_effect": False,
        "authorizes_approval": False,
        "authorizes_model_call": False,
        "authorizes_tool_execution": False,
        "authorizes_external_service": False,
        "reusable_for_other_scope": False,
        "reusable_for_next_route_review": False,
    }
    return [
        {
            **base,
            "item": "route_lock_token",
            "status": "present" if len(token_sha256) == 64 else "missing",
        },
        {
            **base,
            "item": "explicit_route_review_only",
            "status": "route_unlock_candidate_requires_human_review",
        },
        {
            **base,
            "item": "natural_language_routing_disabled",
            "status": "disabled_until_fresh_route_review_and_approval_chain",
        },
        {
            **base,
            "item": "next_connector_action_approval_boundary",
            "status": "approval_readiness_packet_required_for_personal_data_or_side_effect",
        },
    ]


def _route_lock_token_boundary_ready(token_sha256: Any, rows: Any) -> bool:
    token = str(token_sha256 or "")
    if len(token) != 64 or any(char not in "0123456789abcdef" for char in token.lower()):
        return False
    if not isinstance(rows, list) or len(rows) != 4:
        return False
    expected = [
        ("route_lock_token", "present"),
        ("explicit_route_review_only", "route_unlock_candidate_requires_human_review"),
        ("natural_language_routing_disabled", "disabled_until_fresh_route_review_and_approval_chain"),
        ("next_connector_action_approval_boundary", "approval_readiness_packet_required_for_personal_data_or_side_effect"),
    ]
    for row, (item, status) in zip(rows, expected):
        if not isinstance(row, dict):
            return False
        if row.get("item") != item or row.get("status") != status or row.get("token_sha256") != token:
            return False
        for key in [
            "scope_hash",
            "review_receipt_id",
            "route_lock_state",
        ]:
            if not str(row.get(key) or "").strip():
                return False
        for key in [
            "authorizes_account_access",
            "authorizes_route_unlock",
            "authorizes_natural_language_routing",
            "authorizes_personal_data_read",
            "authorizes_side_effect",
            "authorizes_approval",
            "authorizes_model_call",
            "authorizes_tool_execution",
            "authorizes_external_service",
            "reusable_for_other_scope",
            "reusable_for_next_route_review",
        ]:
            if row.get(key) is not False:
                return False
    return True


def _route_lock_handoff_payload(
    *,
    connector: str,
    action: str,
    route_lock_state: str,
    route_unlock_candidate: bool,
    explicit_route_review_required: bool,
    natural_language_routing_enabled: bool,
    implementation_review_ready: bool,
    status_evidence_present: bool,
    metadata_row_contract_ready: bool,
    review_receipt_id: str,
    integration_scope_hash: str,
    expected_scope_hash: str,
    expected_scope_hash_required: bool,
    expected_scope_hash_valid: bool,
    scope_hash_matches_expected: bool,
    missing_fields: list[str],
    route_review_contract_rows: list[dict[str, Any]],
    route_review_contract_ready: bool,
    route_review_contract_summary: list[str],
    implementation_review_receipt_contract_rows: list[dict[str, Any]],
    route_lock_token_sha256: str,
    route_lock_token_boundary_rows: list[dict[str, Any]],
    route_lock_token_boundary_ready: bool,
    route_proof_queue: list[str],
    next_route_proof_command: str,
    implementation_review_command: str,
    route_lock_command: str,
) -> dict[str, Any]:
    next_commands = {
        "implementation_review": implementation_review_command,
        "route_lock": route_lock_command,
    }
    return {
        "connector": connector,
        "action": action,
        "handoff_ready": True,
        "route_lock_handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "route_lock_state": route_lock_state,
        "route_unlock_candidate": route_unlock_candidate,
        "explicit_route_review_required": explicit_route_review_required,
        "natural_language_routing_enabled": natural_language_routing_enabled,
        "implementation_review_ready": implementation_review_ready,
        "status_evidence_present": status_evidence_present,
        "metadata_row_contract_ready": metadata_row_contract_ready,
        "review_receipt_id": review_receipt_id,
        "integration_scope_hash": integration_scope_hash,
        "expected_scope_hash": expected_scope_hash,
        "expected_scope_hash_required": expected_scope_hash_required,
        "expected_scope_hash_valid": expected_scope_hash_valid,
        "scope_hash_matches_expected": scope_hash_matches_expected,
        "missing_fields": list(missing_fields),
        "route_review_contract": {
            "ready": route_review_contract_ready,
            "row_count": len(route_review_contract_rows),
            "summary": list(route_review_contract_summary),
            "rows": route_review_contract_rows,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "authorizes_natural_language_routing": False,
            "authorizes_personal_data_read": False,
            "authorizes_side_effect": False,
            "reusable_for_other_scope": False,
        },
        "implementation_review_receipt": {
            "row_count": len(implementation_review_receipt_contract_rows),
            "rows": implementation_review_receipt_contract_rows,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "reusable_for_other_scope": False,
        },
        "route_lock_token": {
            "sha256": route_lock_token_sha256,
            "present": len(route_lock_token_sha256) == 64,
            "boundary_ready": route_lock_token_boundary_ready,
            "boundary_row_count": len(route_lock_token_boundary_rows),
            "boundary_rows": route_lock_token_boundary_rows,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "authorizes_natural_language_routing": False,
            "authorizes_personal_data_read": False,
            "authorizes_side_effect": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_external_service": False,
            "reusable_for_other_scope": False,
            "reusable_for_next_route_review": False,
        },
        "proof_queue": {
            "items": route_proof_queue,
            "count": len(route_proof_queue),
            "next": next_route_proof_command,
            "next_required": next_route_proof_command,
            "next_proof": next_route_proof_command,
        },
        "commands": next_commands,
        "next_commands": next_commands,
        "next_safe_commands": list(next_commands.values()),
        "boundaries": {
            "natural_language_routing_enabled": False,
            "calls_external_service": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "writes_memory": False,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "authorizes_natural_language_routing": False,
            "authorizes_personal_data_read": False,
            "authorizes_side_effect": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_external_service": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "reusable_for_other_scope": False,
        },
        "authorizes": {
            "account_access": False,
            "route_unlock": False,
            "natural_language_routing": False,
            "personal_data_read": False,
            "side_effect": False,
            "approval": False,
            "model_call": False,
            "tool_execution": False,
            "external_service": False,
        },
        "fresh_review_required_for_next_route": True,
    }


def _implementation_review_handoff_payload(
    *,
    connector: str,
    action: str,
    tool_name: str,
    review_state: str,
    review_checks: list[dict[str, Any]],
    passed_review_checks: int,
    review_receipt_id: str,
    integration_scope_hash: str,
    proof_bundle_command: str,
    implementation_review_command: str,
    receipt_contract_rows: list[dict[str, Any]],
    receipt_contract_summary: list[str],
    proof_queue: list[str],
    next_proof_command: str,
    next_safe_command: str,
    missing_fields: list[str],
    proof_bundle_state: Any,
    preflight_state: Any,
    spec_state: Any,
    status_evidence_present: bool,
    metadata_row_contract_ready: bool,
    suggested_risk: str,
    approval_required: bool,
) -> dict[str, Any]:
    review_ready = review_state == "IMPLEMENTATION_REVIEW_READY"
    next_commands = {
        "proof_bundle": proof_bundle_command,
        "implementation_review": implementation_review_command,
        "next_safe": next_safe_command,
    }
    return {
        "connector": connector,
        "action": action,
        "tool_name": tool_name,
        "handoff_ready": True,
        "implementation_review_handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "review_state": review_state,
        "review_ready": review_ready,
        "implementation_review_allowed": review_ready,
        "can_open_code_review_after_receipt": review_ready,
        "route_blocked_until_implementation_review": not review_ready,
        "review_checks": review_checks,
        "review_check_count": len(review_checks),
        "passed_review_checks": passed_review_checks,
        "missing_fields": list(missing_fields),
        "review_receipt": {
            "id": review_receipt_id,
            "integration_scope_hash": integration_scope_hash,
            "contract_ready": review_ready,
            "contract_row_count": len(receipt_contract_rows),
            "contract_rows": receipt_contract_rows,
            "contract_summary": list(receipt_contract_summary),
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "reusable_for_other_scope": False,
        },
        "commands": next_commands,
        "next_commands": next_commands,
        "next_safe_commands": list(next_commands.values()),
        "proof_queue": {
            "items": proof_queue,
            "count": len(proof_queue),
            "next": next_proof_command,
            "next_required": next_proof_command,
            "next_proof": next_proof_command,
        },
        "source_states": {
            "proof_bundle_state": proof_bundle_state,
            "preflight_state": preflight_state,
            "spec_state": spec_state,
            "status_evidence_present": status_evidence_present,
            "metadata_row_contract_ready": metadata_row_contract_ready,
        },
        "risk": {
            "suggested_risk": suggested_risk,
            "approval_required": approval_required,
        },
        "boundaries": {
            "natural_language_routing_enabled": False,
            "adapter_default_state": "disabled",
            "calls_external_service": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "writes_memory": False,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "authorizes_natural_language_routing": False,
            "authorizes_personal_data_read": False,
            "authorizes_side_effect": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_external_service": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "reusable_for_other_scope": False,
        },
        "next_route_lock_requires_fresh_review": True,
    }


def _proof_bundle_handoff_payload(
    *,
    connector: str,
    action: str,
    tool_name: str,
    bundle_state: str,
    proof_checks: list[dict[str, Any]],
    passed_proof_checks: int,
    proof_queue: list[str],
    next_proof_command: str,
    next_safe_command: str,
    missing_fields: list[str],
    review_receipt_id: str,
    integration_scope_hash: str,
    proof_bundle_command: str,
    implementation_review_command: str,
    receipt_contract_rows: list[dict[str, Any]],
    receipt_contract_summary: list[str],
    metadata_preview_state: Any,
    adapter_acceptance_state: Any,
    enablement_verdict: Any,
    rehearsal_state: Any,
    metadata_row_contract_ready: bool,
    suggested_risk: str,
    approval_required: bool,
) -> dict[str, Any]:
    bundle_ready = bundle_state == "PROOF_BUNDLE_READY_FOR_IMPLEMENTATION_REVIEW"
    next_commands = {
        "proof_bundle": proof_bundle_command,
        "implementation_review": implementation_review_command,
        "next_safe": next_safe_command,
    }
    return {
        "connector": connector,
        "action": action,
        "tool_name": tool_name,
        "handoff_ready": True,
        "proof_bundle_handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "bundle_state": bundle_state,
        "bundle_ready_for_implementation_review": bundle_ready,
        "implementation_review_allowed": bundle_ready,
        "can_open_code_review_after_receipt": False,
        "route_blocked_until_implementation_review": True,
        "proof_checks": proof_checks,
        "proof_check_count": len(proof_checks),
        "passed_proof_checks": passed_proof_checks,
        "missing_fields": list(missing_fields),
        "review_receipt": {
            "id": review_receipt_id,
            "integration_scope_hash": integration_scope_hash,
            "contract_ready": False,
            "contract_row_count": len(receipt_contract_rows),
            "contract_rows": receipt_contract_rows,
            "contract_summary": list(receipt_contract_summary),
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "reusable_for_other_scope": False,
        },
        "commands": next_commands,
        "next_commands": next_commands,
        "next_safe_commands": list(next_commands.values()),
        "proof_queue": {
            "items": proof_queue,
            "count": len(proof_queue),
            "next": next_proof_command,
            "next_required": next_proof_command,
            "next_proof": next_proof_command,
        },
        "source_states": {
            "metadata_preview_state": metadata_preview_state,
            "adapter_acceptance_state": adapter_acceptance_state,
            "enablement_verdict": enablement_verdict,
            "rehearsal_state": rehearsal_state,
            "metadata_row_contract_ready": metadata_row_contract_ready,
        },
        "risk": {
            "suggested_risk": suggested_risk,
            "approval_required": approval_required,
        },
        "boundaries": {
            "natural_language_routing_enabled": False,
            "calls_external_service": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "writes_memory": False,
            "authorizes_account_access": False,
            "authorizes_route_unlock": False,
            "authorizes_natural_language_routing": False,
            "authorizes_personal_data_read": False,
            "authorizes_side_effect": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_external_service": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "reusable_for_other_scope": False,
        },
        "next_step_requires_implementation_review": True,
        "next_route_lock_requires_fresh_review": True,
    }


def make_personal_tools(config: JarvisConfig):
    def integration_readiness_report(args: dict[str, Any]) -> ToolResult:
        requested = _clean_text(args.get("connector"), limit=MAX_CONNECTOR_CHARS).lower()
        connectors = [requested] if requested else list(CONNECTOR_PROFILES.keys())
        rows: list[tuple[str, dict[str, list[str]]]] = []
        for connector in connectors:
            normalized, profile = _connector_profile(connector)
            rows.append((normalized, profile))

        lines = [
            "Jarvis personal integration readiness report:",
            "This is read-only. It ranks connector migration readiness without connecting accounts, reading personal data, calling external services, changing state, controlling the computer, or queuing approvals.",
            "",
            "Readiness rule:",
            "- Start with narrow read-only or draft-only tools.",
            "- Personal-data reads and side effects remain approval-gated.",
            "- Every connector needs a boundary contract, action preview, audit logging, and blocked-action smoke tests before real use.",
            "",
            "Connector readiness:",
        ]
        recommendations: list[str] = []
        readiness_rows: list[dict[str, Any]] = []
        for normalized, profile in rows:
            read_count = len(profile["read_only"])
            private_count = len(profile["personal_data"])
            side_effect_count = len(profile["side_effects"])
            if normalized in {"browser", "reminders"}:
                readiness = "partly migrated"
                next_step = f"integration action preview: {normalized} -> {profile['read_only'][0]}"
            elif read_count >= private_count and side_effect_count <= 3:
                readiness = "good first candidate"
                next_step = f"integration migration plan: {normalized}"
            else:
                readiness = "needs tighter scope first"
                next_step = f"integration contract: {normalized}"
            recommendations.append(next_step)
            readiness_rows.append(
                {
                    "connector": normalized,
                    "readiness": readiness,
                    "read_only_count": read_count,
                    "personal_data_count": private_count,
                    "side_effect_count": side_effect_count,
                    "safest_next_command": next_step,
                    "approval_gate": "required for personal data and side effects",
                }
            )
            lines.extend(
                [
                    f"- {normalized}: {readiness}",
                    f"  read-only candidates: {read_count}; personal-data surfaces: {private_count}; side-effect surfaces: {side_effect_count}",
                    f"  safest next command: `{next_step}`",
                ]
            )

        lines.extend(
            [
                "",
                "Suggested migration order:",
                "- 1. Browser public-page tools and Obsidian notes, because V2 already has safe read-only/local-safe pieces.",
                "- 2. Reminders only as approval-gated creation, because it is a small side effect with a clear receipt.",
                "- 3. Calendar free/busy or metadata-only summaries before event titles, attendees, locations, or invites.",
                "- 4. Email/message/contact tools last, starting with draft-only or user-pasted text workflows.",
                "",
                "Hard stops:",
                OPERATOR_LIMIT_STOP,
                "- Do not request account tokens, cookies, mailbox bodies, message transcripts, calendar notes, contact cards, or attachments from this report.",
                "- Do not send, delete, invite, forward, submit, purchase, mark read, or mutate records without an explicit per-action approval receipt and approval chain proof.",
            ]
        )
        readiness_report_handoff = _readiness_report_handoff_payload(
            requested_connector=requested,
            rows=readiness_rows,
            recommendations=recommendations,
        )

        return ToolResult(
            "integration_readiness_report",
            True,
            "\n".join(lines),
            _safe_metadata(
                connectors_reviewed=len(rows),
                recommendations=recommendations,
                readiness_rows=readiness_rows,
                readiness_report_handoff=readiness_report_handoff,
                readiness_report_handoff_ready=readiness_report_handoff["handoff_ready"],
                readiness_report_ready_for_operator=readiness_report_handoff["ready_for_operator"],
                readiness_report_state_changed=readiness_report_handoff["state_changed"],
                readiness_report_changed=readiness_report_handoff["changed"],
                readiness_report_content_in_handoff=readiness_report_handoff["content_in_handoff"],
                readiness_report_authorizes_execution=readiness_report_handoff["authorizes_execution"],
                readiness_report_authorizes_completion_claim=readiness_report_handoff["authorizes_completion_claim"],
                readiness_report_approval_granted=readiness_report_handoff["approval_granted"],
                readiness_report_boundaries=readiness_report_handoff["boundaries"],
            ),
        )

    def integration_action_preview(args: dict[str, Any]) -> ToolResult:
        connector = _clean_text(args.get("connector") or "calendar", limit=MAX_CONNECTOR_CHARS) or "calendar"
        action = _clean_text(args.get("action"), limit=MAX_ACTION_CHARS)
        if not action:
            return _action_required_failure("integration_action_preview")
        normalized, profile = _connector_profile(connector)
        risk, approval_required, reason = _classify_connector_action(action)
        action_preview_handoff = _action_preview_handoff_payload(
            connector=normalized,
            requested_connector=connector,
            action=action,
            suggested_risk=risk,
            approval_required=approval_required,
            reason=reason,
        )

        lines = [
            "Jarvis integration action preview:",
            "This is read-only. It classifies a future connector action without connecting accounts, reading personal data, calling external services, writing memory, changing state, controlling the computer, or queuing approvals.",
            "",
            f"Connector: {connector}",
            f"Normalized connector: {normalized}",
            f"Requested action: {action}",
            f"Suggested risk: {risk}",
            f"Approval required: {'yes' if approval_required else 'no'}",
            f"Reason: {reason}",
            "",
            "Known connector surfaces:",
            "- READ_ONLY candidates: " + ", ".join(profile["read_only"]) + ".",
            "- PERSONAL_DATA candidates: " + ", ".join(profile["personal_data"]) + ".",
            "- EXTERNAL_SIDE_EFFECT or HIGH_RISK candidates: " + ", ".join(profile["side_effects"]) + ".",
            "",
            "Safe preview packet:",
            f"- Tool: <future_{normalized}_tool>",
            f"- Requested action: {action}",
            "- Scope limit: <selected item, narrow date range, named thread, or explicit recipient>",
            "- Safer alternative: paste relevant text, use metadata-only summary, or draft without sending.",
            "",
            "Hard stops:",
            OPERATOR_LIMIT_STOP,
            "- Do not send, delete, purchase, invite, forward, submit, mark read, or mutate records without explicit per-action approval and approval chain proof.",
            "- Do not read full private content unless the operator approved the exact source and scope.",
            "- If account, recipient, thread, event, or date range is ambiguous, stop and ask the operator.",
        ]
        return ToolResult(
            "integration_action_preview",
            True,
            "\n".join(lines),
            _safe_metadata(
                connector=normalized,
                requested_connector=connector,
                action=action,
                suggested_risk=risk,
                approval_required=approval_required,
                requires_approval=approval_required,
                action_preview_handoff=action_preview_handoff,
                action_preview_handoff_ready=action_preview_handoff["handoff_ready"],
                action_preview_ready_for_operator=action_preview_handoff["ready_for_operator"],
                action_preview_state_changed=action_preview_handoff["state_changed"],
                action_preview_changed=action_preview_handoff["changed"],
                action_preview_content_in_handoff=action_preview_handoff["content_in_handoff"],
                action_preview_authorizes_execution=action_preview_handoff["authorizes_execution"],
                action_preview_authorizes_completion_claim=action_preview_handoff["authorizes_completion_claim"],
                action_preview_approval_granted=action_preview_handoff["approval_granted"],
                action_preview_boundaries=action_preview_handoff["boundaries"],
                action_preview_next_safe_commands=action_preview_handoff["next_safe_commands"],
            ),
        )

    def integration_scope_packet(args: dict[str, Any]) -> ToolResult:
        connector = _clean_text(args.get("connector") or "calendar", limit=MAX_CONNECTOR_CHARS) or "calendar"
        action = _clean_text(args.get("action"), limit=MAX_ACTION_CHARS)
        target = _clean_text(args.get("target"), limit=MAX_SCOPE_FIELD_CHARS)
        time_range = _clean_text(args.get("time_range"), limit=MAX_SCOPE_FIELD_CHARS)
        data_level = _clean_text(args.get("data_level"), limit=MAX_SCOPE_FIELD_CHARS)
        if not action:
            return _action_required_failure("integration_scope_packet")

        normalized, profile = _connector_profile(connector)
        risk, approval_required, reason = _classify_connector_action(action)
        missing: list[str] = []
        if not target:
            missing.append("target/source")
        if normalized in {"calendar", "email", "messages", "browser", "reminders"} and not time_range:
            missing.append("time range or selected item")
        if risk != "READ_ONLY" and not data_level:
            missing.append("data level")

        approval_command = f"integration action preview: {normalized} -> {action}"
        scope_packet_handoff = _scope_packet_handoff_payload(
            connector=normalized,
            requested_connector=connector,
            action=action,
            target=target,
            time_range=time_range,
            data_level=data_level,
            missing_fields=missing,
            suggested_risk=risk,
            approval_required=approval_required,
            reason=reason,
            approval_command=approval_command,
        )
        lines = [
            "Jarvis personal integration scope packet:",
            "This is read-only. It narrows a future connector request without connecting accounts, reading personal data, calling external services, writing memory, changing state, controlling the computer, or queuing approvals.",
            "",
            f"Connector: {connector}",
            f"Normalized connector: {normalized}",
            f"Requested action: {action}",
            f"Suggested risk: {risk}",
            f"Approval required: {'yes' if approval_required else 'no'}",
            f"Reason: {reason}",
            "",
            "Scope fields:",
            f"- target/source: {target or '<missing>'}",
            f"- time range or selected item: {time_range or '<missing>'}",
            f"- data level: {data_level or '<missing>'}",
            f"- missing fields: {', '.join(missing) if missing else 'none'}",
            "",
            "Connector boundary:",
            "- READ_ONLY candidates: " + ", ".join(profile["read_only"]) + ".",
            "- PERSONAL_DATA candidates: " + ", ".join(profile["personal_data"]) + ".",
            "- EXTERNAL_SIDE_EFFECT or HIGH_RISK candidates: " + ", ".join(profile["side_effects"]) + ".",
            "",
            "Approval packet seed:",
            f"- future tool: <future_{normalized}_tool>",
            f"- action: {action}",
            f"- connector: {normalized}",
            f"- target/source: {target or '<required before use>'}",
            f"- time range or selected item: {time_range or '<required before use>'}",
            f"- data level: {data_level or '<metadata-only, summary-only, full content, or side effect>'}",
            f"- preview command: `{approval_command}`",
            "",
            "Stop conditions:",
            OPERATOR_LIMIT_STOP,
            "- Stop if the account, recipient, thread, event, page, list, or date range is ambiguous.",
            "- Stop if the request asks to send, delete, invite, forward, submit, purchase, mark read, or mutate records without a fresh approval receipt.",
            "- Stop if the operator has not explicitly approved the exact private source and scope before any personal-data read.",
        ]
        return ToolResult(
            "integration_scope_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                connector=normalized,
                requested_connector=connector,
                action=action,
                target=target,
                time_range=time_range,
                data_level=data_level,
                missing_fields=missing,
                suggested_risk=risk,
                approval_required=approval_required,
                requires_approval=approval_required,
                scope_ready=scope_packet_handoff["scope_ready"],
                scope_packet_handoff=scope_packet_handoff,
                scope_packet_handoff_ready=scope_packet_handoff["handoff_ready"],
                scope_packet_ready_for_operator=scope_packet_handoff["ready_for_operator"],
                scope_packet_state_changed=scope_packet_handoff["state_changed"],
                scope_packet_changed=scope_packet_handoff["changed"],
                scope_packet_content_in_handoff=scope_packet_handoff["content_in_handoff"],
                scope_packet_authorizes_execution=scope_packet_handoff["authorizes_execution"],
                scope_packet_authorizes_completion_claim=scope_packet_handoff["authorizes_completion_claim"],
                scope_packet_approval_granted=scope_packet_handoff["approval_granted"],
                scope_packet_boundaries=scope_packet_handoff["boundaries"],
                scope_packet_next_safe_commands=scope_packet_handoff["next_safe_commands"],
            ),
        )

    def integration_dry_run_contract(args: dict[str, Any]) -> ToolResult:
        connector = _clean_text(args.get("connector") or "calendar", limit=MAX_CONNECTOR_CHARS) or "calendar"
        action = _clean_text(args.get("action"), limit=MAX_ACTION_CHARS)
        target = _clean_text(args.get("target"), limit=MAX_SCOPE_FIELD_CHARS)
        time_range = _clean_text(args.get("time_range"), limit=MAX_SCOPE_FIELD_CHARS)
        data_level = _clean_text(args.get("data_level"), limit=MAX_SCOPE_FIELD_CHARS)
        verification = _clean_text(args.get("verification"), limit=MAX_SCOPE_FIELD_CHARS)
        if not action:
            return _action_required_failure("integration_dry_run_contract")

        normalized, profile = _connector_profile(connector)
        risk, approval_required, reason = _classify_connector_action(action)
        metadata_row_contract_required = _metadata_row_contract_required(data_level)
        missing: list[str] = []
        if not target:
            missing.append("target/source")
        if normalized in {"calendar", "email", "messages", "browser", "reminders"} and not time_range:
            missing.append("time range or selected item")
        if not data_level:
            missing.append("data level")
        if approval_required and not verification:
            missing.append("verification target")

        dry_run_ready = not missing
        metadata_row_contract_ready = metadata_row_contract_required and not missing
        execution_state = "DRY_RUN_READY" if dry_run_ready else "HELD_FOR_SCOPE"
        approval_marker = "required before real connector use" if approval_required else "not required for dry-run only"
        one_shot_seed = (
            f"integration dry run contract: {normalized} -> {action}; "
            f"target {target or '<required>'}; time {time_range or '<required>'}; "
            f"data {data_level or '<required>'}; verification {verification or '<required for risky/private>'}"
        )
        dry_run_handoff = _dry_run_handoff_payload(
            connector=normalized,
            requested_connector=connector,
            action=action,
            execution_state=execution_state,
            dry_run_ready=dry_run_ready,
            one_shot_seed=one_shot_seed,
            missing_fields=missing,
            target=target,
            time_range=time_range,
            data_level=data_level,
            verification=verification,
            suggested_risk=risk,
            approval_required=approval_required,
            metadata_row_contract_required=metadata_row_contract_required,
            metadata_row_contract_ready=metadata_row_contract_ready,
        )

        lines = [
            "Jarvis personal integration dry-run contract:",
            "This is read-only. It prepares a connector harness receipt without connecting accounts, reading personal data, calling external services, writing memory, changing state, controlling the computer, or queuing approvals.",
            "",
            f"Connector: {connector}",
            f"Normalized connector: {normalized}",
            f"Requested action: {action}",
            f"Execution state: {execution_state}",
            f"Suggested risk: {risk}",
            f"Approval: {approval_marker}",
            f"Reason: {reason}",
            "",
            "Dry-run scope:",
            f"- target/source: {target or '<missing>'}",
            f"- time range or selected item: {time_range or '<missing>'}",
            f"- data level: {data_level or '<missing>'}",
            f"- verification target: {verification or '<missing>'}",
            f"- missing fields: {', '.join(missing) if missing else 'none'}",
            f"- metadata row contract: {'ready' if metadata_row_contract_ready else 'not required for this dry run' if not metadata_row_contract_required else 'blocked'}",
            f"- metadata row schema: {', '.join(METADATA_ROW_SCHEMA)}",
            f"- metadata row limit: {METADATA_ROW_LIMIT}",
            "- blocked payload fields: " + ", ".join(BLOCKED_METADATA_PAYLOAD_FIELDS) + ".",
            "",
            "Harness receipt:",
            f"- future tool: <future_{normalized}_connector>",
            f"- preflight: classify risk, confirm scope, confirm connector availability, prepare audit id",
            f"- dry-run only: return planned request, expected data touched, expected side effect, and verification target",
            f"- real run gate: {'one-shot approval receipt plus approval chain proof required' if approval_required else 'allowed only if implementation stays read-only and metadata-only'}",
            f"- post-run verifier: {verification or '<required before real connector use>'}",
            "",
            "Known connector boundaries:",
            "- READ_ONLY candidates: " + ", ".join(profile["read_only"]) + ".",
            "- PERSONAL_DATA candidates: " + ", ".join(profile["personal_data"]) + ".",
            "- EXTERNAL_SIDE_EFFECT or HIGH_RISK candidates: " + ", ".join(profile["side_effects"]) + ".",
            "",
            "One-shot approval seed:",
            f"- command: `{one_shot_seed}`",
            f"- approval required for real use: {'yes' if approval_required else 'no, unless implementation expands scope'}",
            "- approval cannot be reused for a different account, recipient, thread, event, date range, or side effect.",
            "",
            "Hard stops:",
            OPERATOR_LIMIT_STOP,
            "- Do not connect an account, request tokens, read private content, or execute a connector from this dry run.",
            "- Do not send, delete, purchase, invite, forward, submit, mark read, or mutate records without a fresh per-action approval receipt and approval chain proof.",
            "- If the account, recipient, thread, event, list, page, or time range is ambiguous, stop and ask the operator.",
        ]
        return ToolResult(
            "integration_dry_run_contract",
            True,
            "\n".join(lines),
            _safe_metadata(
                connector=normalized,
                requested_connector=connector,
                action=action,
                target=target,
                time_range=time_range,
                data_level=data_level,
                verification=verification,
                missing_fields=missing,
                execution_state=execution_state,
                dry_run_ready=dry_run_ready,
                suggested_risk=risk,
                approval_required=approval_required,
                requires_approval=approval_required,
                metadata_row_contract_required=metadata_row_contract_required,
                metadata_row_contract_ready=metadata_row_contract_ready,
                sample_row_schema=list(METADATA_ROW_SCHEMA),
                sample_row_limit=METADATA_ROW_LIMIT,
                blocked_payload_fields=list(BLOCKED_METADATA_PAYLOAD_FIELDS),
                dry_run_handoff=dry_run_handoff,
                dry_run_handoff_ready=dry_run_handoff["handoff_ready"],
                dry_run_ready_for_operator=dry_run_handoff["ready_for_operator"],
                dry_run_state_changed=dry_run_handoff["state_changed"],
                dry_run_changed=dry_run_handoff["changed"],
                dry_run_content_in_handoff=dry_run_handoff["content_in_handoff"],
                dry_run_authorizes_execution=dry_run_handoff["authorizes_execution"],
                dry_run_authorizes_completion_claim=dry_run_handoff["authorizes_completion_claim"],
                dry_run_approval_granted=dry_run_handoff["approval_granted"],
                dry_run_boundaries=dry_run_handoff["boundaries"],
                dry_run_next_safe_commands=dry_run_handoff["next_safe_commands"],
            ),
        )

    def integration_runbook(args: dict[str, Any]) -> ToolResult:
        connector = _clean_text(args.get("connector") or "calendar", limit=MAX_CONNECTOR_CHARS) or "calendar"
        action = _clean_text(args.get("action"), limit=MAX_ACTION_CHARS)
        target = _clean_text(args.get("target"), limit=MAX_SCOPE_FIELD_CHARS)
        time_range = _clean_text(args.get("time_range"), limit=MAX_SCOPE_FIELD_CHARS)
        data_level = _clean_text(args.get("data_level"), limit=MAX_SCOPE_FIELD_CHARS)
        verification = _clean_text(args.get("verification"), limit=MAX_SCOPE_FIELD_CHARS)
        rollback = _clean_text(args.get("rollback"), limit=MAX_SCOPE_FIELD_CHARS)
        if not action:
            return _action_required_failure("integration_runbook")

        normalized, profile = _connector_profile(connector)
        risk, approval_required, reason = _classify_connector_action(action)
        metadata_row_contract_required = _metadata_row_contract_required(data_level)
        missing: list[str] = []
        if not target:
            missing.append("target/source")
        if normalized in {"calendar", "email", "messages", "browser", "reminders"} and not time_range:
            missing.append("time range or selected item")
        if not data_level:
            missing.append("data level")
        if approval_required and not verification:
            missing.append("verification target")
        if risk == "EXTERNAL_SIDE_EFFECT" and not rollback:
            missing.append("rollback/cancel path")

        readiness = "READY_FOR_DRY_RUN" if not missing else "HELD_FOR_SCOPE"
        metadata_row_contract_ready = metadata_row_contract_required and not missing
        approval_state = "approval required before real connector use" if approval_required else "approval not required while implementation stays metadata-only/read-only"
        verifier = verification or "<required before real connector use>"
        rollback_plan = rollback or ("dismiss approval before execution; no connector call has been made" if risk != "EXTERNAL_SIDE_EFFECT" else "<required before side-effect execution>")
        preview_command = (
            f"integration dry run contract: {normalized} -> {action}; "
            f"target {target or '<required>'}; time {time_range or '<required>'}; "
            f"data {data_level or '<required>'}; verification {verification or '<required>'}"
        )
        runbook_handoff = _runbook_handoff_payload(
            connector=normalized,
            requested_connector=connector,
            action=action,
            readiness=readiness,
            preview_command=preview_command,
            verifier=verifier,
            rollback_plan=rollback_plan,
            missing_fields=missing,
            target=target,
            time_range=time_range,
            data_level=data_level,
            verification=verification,
            rollback=rollback,
            suggested_risk=risk,
            approval_required=approval_required,
            rollback_required=risk == "EXTERNAL_SIDE_EFFECT",
            metadata_row_contract_required=metadata_row_contract_required,
            metadata_row_contract_ready=metadata_row_contract_ready,
        )

        lines = [
            "Jarvis personal integration runbook:",
            "This is read-only. It turns a future connector request into preflight, approval, execution, verification, rollback, and audit steps without connecting accounts, reading personal data, calling external services, changing state, controlling the computer, or queuing approvals.",
            "",
            f"Connector: {connector}",
            f"Normalized connector: {normalized}",
            f"Requested action: {action}",
            f"Runbook readiness: {readiness}",
            f"Suggested risk: {risk}",
            f"Approval state: {approval_state}",
            f"Reason: {reason}",
            "",
            "Scope lock:",
            f"- target/source: {target or '<missing>'}",
            f"- time range or selected item: {time_range or '<missing>'}",
            f"- data level: {data_level or '<missing>'}",
            f"- verification target: {verifier}",
            f"- rollback/cancel path: {rollback_plan}",
            f"- missing fields: {', '.join(missing) if missing else 'none'}",
            "",
            "Preflight steps:",
            "- Confirm connector implementation exists and declares a ToolRegistry risk level.",
            "- Confirm the requested connector, target/source, time range, data level, verification target, and rollback path match this runbook.",
            f"- Metadata row contract: {'ready' if metadata_row_contract_ready else 'not required for this runbook' if not metadata_row_contract_required else 'blocked'}; schema {', '.join(METADATA_ROW_SCHEMA)}; row limit {METADATA_ROW_LIMIT}.",
            "- Full private payload fields stay blocked: " + ", ".join(BLOCKED_METADATA_PAYLOAD_FIELDS) + ".",
            "- Prefer metadata-only or draft-only behavior when it satisfies the operator's order.",
            "- If the request changed since this runbook was generated, build a fresh runbook.",
            "",
            "Approval gate:",
            f"- Real connector execution: {'blocked until the operator approves the exact scoped action' if approval_required else 'allowed only for implemented read-only/metadata-only tools'}",
            f"- Approval preview command: `{preview_command}`",
            "- Approval is one-shot and cannot transfer to another account, recipient, thread, event, page, time range, data level, or side effect.",
            "",
            "Execution plan:",
            f"- future tool: <future_{normalized}_connector>",
            f"- dry-run output: planned data touched, planned side effect, expected result, and verifier",
            "- real run output: connector response plus audit id, timestamp, scope, and approval id if one was required",
            "",
            "Verification steps:",
            f"- primary verifier: {verifier}",
            "- inspect `recent tool runs` and `runtime trace receipt` after execution",
            "- use `verification receipt <run id>: <expectation>` before claiming completion",
            "",
            "Rollback and recovery:",
            f"- rollback/cancel: {rollback_plan}",
            "- if verification fails, stop and produce a recovery packet instead of silently retrying",
            "- side effects need explicit corrective approval before any undo attempt that changes outside-world state",
            "",
            "Known connector boundaries:",
            "- READ_ONLY candidates: " + ", ".join(profile["read_only"]) + ".",
            "- PERSONAL_DATA candidates: " + ", ".join(profile["personal_data"]) + ".",
            "- EXTERNAL_SIDE_EFFECT or HIGH_RISK candidates: " + ", ".join(profile["side_effects"]) + ".",
            "",
            "Hard stops:",
            OPERATOR_LIMIT_STOP,
            "- Do not connect an account, request tokens, read private content, or execute a connector from this runbook.",
            "- Do not send, delete, purchase, invite, forward, submit, mark read, or mutate records without a fresh per-action approval receipt and approval chain proof.",
            "- If the account, recipient, thread, event, list, page, time range, verification target, or rollback path is ambiguous, stop and ask the operator.",
        ]
        return ToolResult(
            "integration_runbook",
            True,
            "\n".join(lines),
            _safe_metadata(
                connector=normalized,
                requested_connector=connector,
                action=action,
                target=target,
                time_range=time_range,
                data_level=data_level,
                verification=verification,
                rollback=rollback,
                missing_fields=missing,
                readiness=readiness,
                suggested_risk=risk,
                approval_required=approval_required,
                requires_approval=approval_required,
                rollback_required=risk == "EXTERNAL_SIDE_EFFECT",
                metadata_row_contract_required=metadata_row_contract_required,
                metadata_row_contract_ready=metadata_row_contract_ready,
                sample_row_schema=list(METADATA_ROW_SCHEMA),
                sample_row_limit=METADATA_ROW_LIMIT,
                blocked_payload_fields=list(BLOCKED_METADATA_PAYLOAD_FIELDS),
                runbook_handoff=runbook_handoff,
                runbook_handoff_ready=runbook_handoff["handoff_ready"],
                runbook_ready_for_operator=runbook_handoff["ready_for_operator"],
                runbook_state_changed=runbook_handoff["state_changed"],
                runbook_changed=runbook_handoff["changed"],
                runbook_content_in_handoff=runbook_handoff["content_in_handoff"],
                runbook_authorizes_execution=runbook_handoff["authorizes_execution"],
                runbook_authorizes_completion_claim=runbook_handoff["authorizes_completion_claim"],
                runbook_approval_granted=runbook_handoff["approval_granted"],
                runbook_boundaries=runbook_handoff["boundaries"],
                runbook_next_safe_commands=runbook_handoff["next_safe_commands"],
            ),
        )

    def integration_promotion_gate(args: dict[str, Any]) -> ToolResult:
        connector = _clean_text(args.get("connector") or "calendar", limit=MAX_CONNECTOR_CHARS) or "calendar"
        action = _clean_text(args.get("action"), limit=MAX_ACTION_CHARS)
        target = _clean_text(args.get("target"), limit=MAX_SCOPE_FIELD_CHARS)
        time_range = _clean_text(args.get("time_range"), limit=MAX_SCOPE_FIELD_CHARS)
        data_level = _clean_text(args.get("data_level"), limit=MAX_SCOPE_FIELD_CHARS)
        verification = _clean_text(args.get("verification"), limit=MAX_SCOPE_FIELD_CHARS)
        rollback = _clean_text(args.get("rollback"), limit=MAX_SCOPE_FIELD_CHARS)
        tests = _clean_text(args.get("tests") or args.get("test_plan"), limit=MAX_SCOPE_FIELD_CHARS)
        audit = _clean_text(args.get("audit") or args.get("audit_plan"), limit=MAX_SCOPE_FIELD_CHARS)
        if not action:
            return _action_required_failure("integration_promotion_gate")

        normalized, profile = _connector_profile(connector)
        risk, approval_required, reason = _classify_connector_action(action)
        metadata_row_contract_required = _metadata_row_contract_required(data_level)
        missing: list[str] = []
        if not target:
            missing.append("target/source")
        if normalized in {"calendar", "email", "messages", "browser", "reminders"} and not time_range:
            missing.append("time range or selected item")
        if not data_level:
            missing.append("data level")
        if approval_required and not verification:
            missing.append("verification target")
        if risk == "EXTERNAL_SIDE_EFFECT" and not rollback:
            missing.append("rollback/cancel path")
        if not tests:
            missing.append("smoke test plan")
        if not audit:
            missing.append("audit trail plan")

        if missing:
            verdict = "PROMOTION_BLOCKED"
            next_move = "Finish the missing scope, test, audit, verification, or rollback evidence before implementation."
        elif risk == "READ_ONLY":
            verdict = "READY_FOR_READ_ONLY_PROTOTYPE"
            next_move = f"Implement the narrow `<future_{normalized}_connector>` read-only path with the listed tests and audit log."
        else:
            verdict = "READY_FOR_APPROVAL_GATED_IMPLEMENTATION_SPEC"
            next_move = "Write the implementation spec and blocked-action tests first; real connector execution still needs one-shot approval and approval chain proof."

        runbook_command = (
            f"integration runbook: {normalized} -> {action}; "
            f"target {target or '<required>'}; time {time_range or '<required>'}; "
            f"data {data_level or '<required>'}; verification {verification or '<required>'}; "
            f"rollback {rollback or '<required for side effects>'}"
        )
        test_plan = tests or "focused smoke test plus blocked personal-data/side-effect regression"
        audit_plan = audit or "ToolRegistry run record with connector, scope, approval id, verification target, and result"
        metadata_row_contract_ready = metadata_row_contract_required and not missing
        implementation_allowed = verdict in {"READY_FOR_READ_ONLY_PROTOTYPE", "READY_FOR_APPROVAL_GATED_IMPLEMENTATION_SPEC"}
        promotion_handoff = _promotion_handoff_payload(
            connector=normalized,
            requested_connector=connector,
            action=action,
            promotion_verdict=verdict,
            next_move=next_move,
            runbook_command=runbook_command,
            missing_fields=missing,
            target=target,
            time_range=time_range,
            data_level=data_level,
            verification=verification,
            rollback=rollback,
            tests=tests,
            audit=audit,
            suggested_risk=risk,
            approval_required=approval_required,
            rollback_required=risk == "EXTERNAL_SIDE_EFFECT",
            metadata_row_contract_required=metadata_row_contract_required,
            metadata_row_contract_ready=metadata_row_contract_ready,
            implementation_allowed=implementation_allowed,
        )

        lines = [
            "Jarvis personal integration promotion gate:",
            "This is read-only. It decides whether a future connector can move from design/runbook into implementation without connecting accounts, reading personal data, calling external services, changing state, controlling the computer, approving requests, or queuing approvals.",
            "",
            f"Connector: {connector}",
            f"Normalized connector: {normalized}",
            f"Requested action: {action}",
            f"Promotion verdict: {verdict}",
            f"Suggested risk: {risk}",
            f"Approval required for real use: {'yes' if approval_required else 'no, while implementation stays read-only/metadata-only'}",
            f"Reason: {reason}",
            "",
            "Evidence checked:",
            f"- target/source: {target or '<missing>'}",
            f"- time range or selected item: {time_range or '<missing>'}",
            f"- data level: {data_level or '<missing>'}",
            f"- verification target: {verification or '<missing>'}",
            f"- rollback/cancel path: {rollback or '<missing>'}",
            f"- smoke test plan: {test_plan}",
            f"- audit trail plan: {audit_plan}",
            f"- metadata row contract: {'ready' if metadata_row_contract_ready else 'not required for this promotion' if not metadata_row_contract_required else 'blocked'}",
            f"- metadata row schema: {', '.join(METADATA_ROW_SCHEMA)}",
            f"- metadata row limit: {METADATA_ROW_LIMIT}",
            f"- blocked payload fields: {', '.join(BLOCKED_METADATA_PAYLOAD_FIELDS)}",
            f"- missing blockers: {', '.join(missing) if missing else 'none'}",
            "",
            "Required implementation invariants:",
            "- Register the connector with a narrow ToolRegistry risk level before planner routing can use it.",
            "- Keep natural-language routing disabled until the focused smoke tests pass.",
            "- Store only the bounded audit summary, not full private connector payloads, unless the operator explicitly asks to save a summary.",
            "- Metadata-only adapters must keep rows limited to id, timestamp, label, source and max 2 fake rows until a real connector passes separate approval-gated review.",
            "- Personal-data reads and side effects must include exact scope, one-shot approval, approval chain proof, audit id, and verification target.",
            "",
            "Smoke tests required:",
            f"- happy path: {test_plan}",
            "- blocked path: same connector action without required scope must not run.",
            "- approval path: personal-data or side-effect action must queue/require a one-shot approval and approval chain proof before real execution.",
            "- audit path: recent tool runs and runtime trace must show connector, risk, scope, approval id if any, and verification target.",
            "",
            "Allowed next build move:",
            f"- {next_move}",
            f"- Runbook to inspect first: `{runbook_command}`",
            "",
            "Known connector boundaries:",
            "- READ_ONLY candidates: " + ", ".join(profile["read_only"]) + ".",
            "- PERSONAL_DATA candidates: " + ", ".join(profile["personal_data"]) + ".",
            "- EXTERNAL_SIDE_EFFECT or HIGH_RISK candidates: " + ", ".join(profile["side_effects"]) + ".",
            "",
            "Hard stops:",
            OPERATOR_LIMIT_STOP,
            "- Do not implement broad account access, token capture, cookie capture, full mailbox/message/calendar/contact ingestion, or authenticated browser shortcuts from this gate.",
            "- Do not claim a connector is ready until its focused smoke tests and blocked-action tests prove the exact risk boundary.",
            "- Do not send, delete, purchase, invite, forward, submit, mark read, or mutate records without a fresh per-action approval receipt and approval chain proof.",
        ]
        return ToolResult(
            "integration_promotion_gate",
            True,
            "\n".join(lines),
            _safe_metadata(
                connector=normalized,
                requested_connector=connector,
                action=action,
                target=target,
                time_range=time_range,
                data_level=data_level,
                verification=verification,
                rollback=rollback,
                tests=tests,
                audit=audit,
                missing_fields=missing,
                promotion_verdict=verdict,
                suggested_risk=risk,
                approval_required=approval_required,
                requires_approval=approval_required,
                rollback_required=risk == "EXTERNAL_SIDE_EFFECT",
                metadata_row_contract_required=metadata_row_contract_required,
                metadata_row_contract_ready=metadata_row_contract_ready,
                sample_row_schema=list(METADATA_ROW_SCHEMA),
                sample_row_limit=METADATA_ROW_LIMIT,
                blocked_payload_fields=list(BLOCKED_METADATA_PAYLOAD_FIELDS),
                implementation_allowed=implementation_allowed,
                promotion_handoff=promotion_handoff,
                promotion_handoff_ready=promotion_handoff["handoff_ready"],
                promotion_ready_for_operator=promotion_handoff["ready_for_operator"],
                promotion_state_changed=promotion_handoff["state_changed"],
                promotion_changed=promotion_handoff["changed"],
                promotion_content_in_handoff=promotion_handoff["content_in_handoff"],
                promotion_authorizes_execution=promotion_handoff["authorizes_execution"],
                promotion_authorizes_completion_claim=promotion_handoff["authorizes_completion_claim"],
                promotion_approval_granted=promotion_handoff["approval_granted"],
                promotion_boundaries=promotion_handoff["boundaries"],
                promotion_next_safe_commands=promotion_handoff["next_safe_commands"],
            ),
        )

    def integration_implementation_spec(args: dict[str, Any]) -> ToolResult:
        connector = _clean_text(args.get("connector") or "calendar", limit=MAX_CONNECTOR_CHARS) or "calendar"
        action = _clean_text(args.get("action"), limit=MAX_ACTION_CHARS)
        target = _clean_text(args.get("target"), limit=MAX_SCOPE_FIELD_CHARS)
        time_range = _clean_text(args.get("time_range"), limit=MAX_SCOPE_FIELD_CHARS)
        data_level = _clean_text(args.get("data_level"), limit=MAX_SCOPE_FIELD_CHARS)
        verification = _clean_text(args.get("verification"), limit=MAX_SCOPE_FIELD_CHARS)
        rollback = _clean_text(args.get("rollback"), limit=MAX_SCOPE_FIELD_CHARS)
        tests = _clean_text(args.get("tests") or args.get("test_plan"), limit=MAX_SCOPE_FIELD_CHARS)
        audit = _clean_text(args.get("audit") or args.get("audit_plan"), limit=MAX_SCOPE_FIELD_CHARS)
        if not action:
            return _action_required_failure("integration_implementation_spec")

        normalized, profile = _connector_profile(connector)
        risk, approval_required, reason = _classify_connector_action(action)
        metadata_row_contract_required = _metadata_row_contract_required(data_level)
        missing: list[str] = []
        if not target:
            missing.append("target/source")
        if normalized in {"calendar", "email", "messages", "browser", "reminders"} and not time_range:
            missing.append("time range or selected item")
        if not data_level:
            missing.append("data level")
        if approval_required and not verification:
            missing.append("verification target")
        if risk == "EXTERNAL_SIDE_EFFECT" and not rollback:
            missing.append("rollback/cancel path")
        if not tests:
            missing.append("smoke test plan")
        if not audit:
            missing.append("audit trail plan")

        spec_state = "SPEC_BLOCKED" if missing else "SPEC_READY_FOR_REVIEW"
        implementation_name = f"future_{normalized}_{'_'.join(action.lower().split()[:4]) or 'connector'}"
        tool_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in implementation_name).strip("_")
        if not tool_name:
            tool_name = f"future_{normalized}_connector"
        risk_level = "READ_ONLY" if risk == "READ_ONLY" else risk
        approval_gate = (
            "not required while the implementation stays metadata-only/read-only"
            if not approval_required
            else "required before any real connector call that can reveal personal data or change outside-world state"
        )
        planner_route = f"{normalized} {action} -> {tool_name}"
        endpoint_slug = "-".join(part for part in tool_name.split("_") if part)
        endpoint = f"/api/{endpoint_slug}"
        test_plan = tests or "focused smoke test plus blocked personal-data/side-effect regression"
        audit_plan = audit or "ToolRegistry run record with connector, scope, risk, approval id if any, verification target, and result"
        metadata_row_contract_ready = metadata_row_contract_required and not missing
        implementation_spec_handoff = _implementation_spec_handoff_payload(
            connector=normalized,
            requested_connector=connector,
            action=action,
            tool_name=tool_name,
            spec_state=spec_state,
            missing_fields=missing,
            target=target,
            time_range=time_range,
            data_level=data_level,
            verification=verification,
            rollback=rollback,
            tests=tests,
            audit=audit,
            endpoint=endpoint,
            planner_route=planner_route,
            risk_level=risk_level,
            suggested_risk=risk,
            approval_required=approval_required,
            rollback_required=risk == "EXTERNAL_SIDE_EFFECT",
            metadata_row_contract_required=metadata_row_contract_required,
            metadata_row_contract_ready=metadata_row_contract_ready,
        )

        lines = [
            "Jarvis personal integration implementation spec:",
            "This is read-only. It converts a connector promotion idea into an implementation contract without connecting accounts, reading personal data, calling external services, changing state, controlling the computer, approving requests, or queuing approvals.",
            "",
            f"Connector: {connector}",
            f"Normalized connector: {normalized}",
            f"Requested action: {action}",
            f"Spec state: {spec_state}",
            f"Suggested risk: {risk}",
            f"Approval gate: {approval_gate}",
            f"Reason: {reason}",
            "",
            "Implementation contract:",
            f"- tool name: {tool_name}",
            f"- toolset: personal",
            f"- ToolRegistry risk: {risk_level}",
            f"- planner route: `{planner_route}`",
            f"- optional status API: `{endpoint}`",
            "- default enabled state: disabled for natural-language auto-routing until smoke tests pass",
            "- output shape: concise user-facing summary plus bounded metadata, never raw private payloads by default",
            "",
            "Required arguments:",
            f"- connector: {normalized}",
            f"- action: {action}",
            f"- target/source: {target or '<missing>'}",
            f"- time range or selected item: {time_range or '<missing>'}",
            f"- data level: {data_level or '<missing>'}",
            f"- verification target: {verification or '<missing>'}",
            f"- rollback/cancel path: {rollback or '<missing>'}",
            f"- metadata row contract: {'ready' if metadata_row_contract_ready else 'not required for this spec' if not metadata_row_contract_required else 'blocked'}",
            f"- metadata row schema: {', '.join(METADATA_ROW_SCHEMA)}",
            f"- metadata row limit: {METADATA_ROW_LIMIT}",
            f"- blocked payload fields: {', '.join(BLOCKED_METADATA_PAYLOAD_FIELDS)}",
            "",
            "Execution gates:",
            "- preflight: normalize connector, validate scope, classify risk, and allocate an audit id before any connector call.",
            "- read-only path: may only return metadata, public data, draft text, or user-pasted content unless the operator approved the exact private source.",
            "- personal-data path: must stop at a one-shot approval receipt and approval chain proof before reading private account data.",
            "- side-effect path: must stop at a one-shot approval receipt and approval chain proof before send/create/delete/update/submit/mark-read actions.",
            "- verification path: compare the connector result with the declared verification target before claiming success.",
            "- rollback path: show the rollback/cancel plan; corrective outside-world changes need a fresh approval.",
            "",
            "Smoke tests required:",
            f"- happy path: {test_plan}",
            "- missing-scope path: omit target/time/data and confirm the tool blocks before connector access.",
            "- approval path: private data or side-effect action must require approval and must not auto-run.",
            "- audit path: recent tool runs must include connector, action, scope, risk, approval id if any, verification target, and result.",
            "",
            "Audit contract:",
            f"- {audit_plan}",
            "- audit metadata must include implementation_spec_version=1 and spec_state.",
            "- metadata-only audit rows must use only id, timestamp, label, source; max 2 fake rows; full private payload fields stay blocked.",
            "- audit metadata must not store tokens, cookies, full email/message bodies, contact cards, attachments, or broad calendar notes by default.",
            "",
            "Known connector boundaries:",
            "- READ_ONLY candidates: " + ", ".join(profile["read_only"]) + ".",
            "- PERSONAL_DATA candidates: " + ", ".join(profile["personal_data"]) + ".",
            "- EXTERNAL_SIDE_EFFECT or HIGH_RISK candidates: " + ", ".join(profile["side_effects"]) + ".",
            "",
            "Missing blockers:",
            f"- {', '.join(missing) if missing else 'none'}",
            "",
            "Hard stops:",
            OPERATOR_LIMIT_STOP,
            "- Do not implement account token capture, cookie capture, broad mailbox/message/calendar/contact ingestion, or authenticated browser shortcuts from this spec.",
            "- Do not turn on natural-language auto-routing until focused smoke tests and blocked-action tests pass.",
            "- Do not send, delete, purchase, invite, forward, submit, mark read, or mutate records without a fresh per-action approval receipt and approval chain proof.",
        ]
        return ToolResult(
            "integration_implementation_spec",
            True,
            "\n".join(lines),
            _safe_metadata(
                connector=normalized,
                requested_connector=connector,
                action=action,
                target=target,
                time_range=time_range,
                data_level=data_level,
                verification=verification,
                rollback=rollback,
                tests=tests,
                audit=audit,
                missing_fields=missing,
                spec_state=spec_state,
                tool_name=tool_name,
                endpoint=endpoint,
                planner_route=planner_route,
                suggested_risk=risk,
                approval_required=approval_required,
                requires_approval=approval_required,
                rollback_required=risk == "EXTERNAL_SIDE_EFFECT",
                implementation_spec_version=1,
                metadata_row_contract_required=metadata_row_contract_required,
                metadata_row_contract_ready=metadata_row_contract_ready,
                sample_row_schema=list(METADATA_ROW_SCHEMA),
                sample_row_limit=METADATA_ROW_LIMIT,
                blocked_payload_fields=list(BLOCKED_METADATA_PAYLOAD_FIELDS),
                natural_language_routing_enabled=False,
                implementation_spec_handoff=implementation_spec_handoff,
                implementation_spec_handoff_ready=implementation_spec_handoff["handoff_ready"],
                implementation_spec_ready_for_operator=implementation_spec_handoff["ready_for_operator"],
                implementation_spec_state_changed=implementation_spec_handoff["state_changed"],
                implementation_spec_changed=implementation_spec_handoff["changed"],
                implementation_spec_content_in_handoff=implementation_spec_handoff["content_in_handoff"],
                implementation_spec_authorizes_execution=implementation_spec_handoff["authorizes_execution"],
                implementation_spec_authorizes_completion_claim=implementation_spec_handoff["authorizes_completion_claim"],
                implementation_spec_approval_granted=implementation_spec_handoff["approval_granted"],
                implementation_spec_boundaries=implementation_spec_handoff["boundaries"],
                implementation_spec_next_safe_commands=implementation_spec_handoff["next_safe_commands"],
            ),
        )

    def integration_preflight_contract(args: dict[str, Any]) -> ToolResult:
        connector = _clean_text(args.get("connector") or "calendar", limit=MAX_CONNECTOR_CHARS) or "calendar"
        action = _clean_text(args.get("action"), limit=MAX_ACTION_CHARS)
        target = _clean_text(args.get("target"), limit=MAX_SCOPE_FIELD_CHARS)
        time_range = _clean_text(args.get("time_range"), limit=MAX_SCOPE_FIELD_CHARS)
        data_level = _clean_text(args.get("data_level"), limit=MAX_SCOPE_FIELD_CHARS)
        verification = _clean_text(args.get("verification"), limit=MAX_SCOPE_FIELD_CHARS)
        rollback = _clean_text(args.get("rollback"), limit=MAX_SCOPE_FIELD_CHARS)
        tests = _clean_text(args.get("tests") or args.get("test_plan"), limit=MAX_SCOPE_FIELD_CHARS)
        audit = _clean_text(args.get("audit") or args.get("audit_plan"), limit=MAX_SCOPE_FIELD_CHARS)
        if not action:
            return _action_required_failure("integration_preflight_contract")

        normalized, profile = _connector_profile(connector)
        risk, approval_required, reason = _classify_connector_action(action)
        metadata_row_contract_required = _metadata_row_contract_required(data_level)
        missing: list[str] = []
        if not target:
            missing.append("target/source")
        if normalized in {"calendar", "email", "messages", "browser", "reminders"} and not time_range:
            missing.append("time range or selected item")
        if not data_level:
            missing.append("data level")
        if approval_required and not verification:
            missing.append("verification target")
        if risk == "EXTERNAL_SIDE_EFFECT" and not rollback:
            missing.append("rollback/cancel path")
        if not tests:
            missing.append("smoke test plan")
        if not audit:
            missing.append("audit trail plan")

        implementation_name = f"future_{normalized}_{'_'.join(action.lower().split()[:4]) or 'connector'}"
        tool_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in implementation_name).strip("_") or f"future_{normalized}_connector"
        risk_level = "READ_ONLY" if risk == "READ_ONLY" else risk
        preflight_state = "PREFLIGHT_READY_FOR_REVIEW" if not missing else "PREFLIGHT_BLOCKED"
        exact_args = {
            "connector": normalized,
            "action": action,
            "target": target,
            "time_range": time_range,
            "data_level": data_level,
            "verification": verification,
            "rollback": rollback,
        }
        test_plan = tests or "focused smoke test plus blocked personal-data/side-effect regression"
        audit_plan = audit or "ToolRegistry run record with connector, action, scope, risk, approval id if any, verification target, and result"
        metadata_row_contract_ready = metadata_row_contract_required and not missing
        approval_packet = (
            f"approval packet for {tool_name}: connector={normalized}; action={action}; "
            f"target={target or '<required>'}; time={time_range or '<required>'}; "
            f"data={data_level or '<required>'}; verification={verification or '<required>'}; "
            f"rollback={rollback or '<required for side effects>'}"
        )
        preflight_handoff = _preflight_handoff_payload(
            connector=normalized,
            requested_connector=connector,
            action=action,
            tool_name=tool_name,
            preflight_state=preflight_state,
            exact_args=exact_args,
            missing_fields=missing,
            target=target,
            time_range=time_range,
            data_level=data_level,
            verification=verification,
            rollback=rollback,
            tests=tests,
            audit=audit,
            approval_packet=approval_packet,
            suggested_risk=risk,
            approval_required=approval_required,
            rollback_required=risk == "EXTERNAL_SIDE_EFFECT",
            metadata_row_contract_required=metadata_row_contract_required,
            metadata_row_contract_ready=metadata_row_contract_ready,
        )

        lines = [
            "Jarvis personal integration preflight contract:",
            "This is read-only. It is the final harness packet before a future connector can be implemented or enabled; it does not connect accounts, read personal data, call external services, approve requests, execute tools, change state, control the computer, or queue approvals.",
            "",
            f"Connector: {connector}",
            f"Normalized connector: {normalized}",
            f"Requested action: {action}",
            f"Preflight state: {preflight_state}",
            f"Suggested risk: {risk}",
            f"ToolRegistry risk: {risk_level}",
            f"Approval required for real use: {'yes' if approval_required else 'no, while implementation stays read-only/metadata-only'}",
            f"Reason: {reason}",
            "",
            "Exact future tool contract:",
            f"- tool name: {tool_name}",
            "- toolset: personal",
            f"- arguments: connector={normalized}; action={action}; target={target or '<missing>'}; time_range={time_range or '<missing>'}; data_level={data_level or '<missing>'}; verification={verification or '<missing>'}; rollback={rollback or '<missing>'}",
            f"- missing blockers: {', '.join(missing) if missing else 'none'}",
            f"- metadata row contract: {'ready' if metadata_row_contract_ready else 'not required for this preflight' if not metadata_row_contract_required else 'blocked'}",
            f"- metadata row schema: {', '.join(METADATA_ROW_SCHEMA)}",
            f"- metadata row limit: {METADATA_ROW_LIMIT}",
            f"- blocked payload fields: {', '.join(BLOCKED_METADATA_PAYLOAD_FIELDS)}",
            "",
            "Enablement gates:",
            "- gate 1: implementation spec exists and keeps natural-language routing disabled by default.",
            "- gate 2: focused smoke test proves the happy path without broad account access.",
            "- gate 3: blocked-action smoke test proves missing scope cannot reach connector code.",
            "- gate 4: personal-data or side-effect route creates approval readiness, a one-shot approval packet, and approval chain proof before real use.",
            "- gate 5: audit row includes connector, action, scope, risk, approval id if any, verification target, and result.",
            "- gate 6: verification receipt proves the declared success condition before completion is claimed.",
            "",
            "Required proof before enabling:",
            f"- smoke tests: {test_plan}",
            f"- audit plan: {audit_plan}",
            "- metadata row contract for metadata-only adapters: id, timestamp, label, source; max 2 fake rows; full private payload fields blocked.",
            "- compile pass: `python3 -m compileall -q jarvis_v2`",
            "- focused personal smoke: `python3 -m jarvis_v2.scripts.smoke_test_personal`",
            "",
            "Approval packet seed:",
            f"- {approval_packet}",
            "- approval is one-shot and cannot transfer to another account, recipient, thread, event, page, time range, data level, or side effect.",
            "",
            "Known connector boundaries:",
            "- READ_ONLY candidates: " + ", ".join(profile["read_only"]) + ".",
            "- PERSONAL_DATA candidates: " + ", ".join(profile["personal_data"]) + ".",
            "- EXTERNAL_SIDE_EFFECT or HIGH_RISK candidates: " + ", ".join(profile["side_effects"]) + ".",
            "",
            "Stop conditions:",
            OPERATOR_LIMIT_STOP,
            "- Stop if any exact argument is missing or changed since this contract was generated.",
            "- Stop if implementation would store tokens, cookies, full email/message bodies, contact cards, attachments, or broad calendar notes by default.",
            "- Stop if a personal-data or side-effect action can run without one-shot approval and approval chain proof.",
            "- Stop if verification or rollback is ambiguous.",
        ]
        return ToolResult(
            "integration_preflight_contract",
            True,
            "\n".join(lines),
            _safe_metadata(
                connector=normalized,
                requested_connector=connector,
                action=action,
                target=target,
                time_range=time_range,
                data_level=data_level,
                verification=verification,
                rollback=rollback,
                tests=tests,
                audit=audit,
                missing_fields=missing,
                preflight_state=preflight_state,
                tool_name=tool_name,
                exact_args=exact_args,
                suggested_risk=risk,
                approval_required=approval_required,
                requires_approval=approval_required,
                rollback_required=risk == "EXTERNAL_SIDE_EFFECT",
                metadata_row_contract_required=metadata_row_contract_required,
                metadata_row_contract_ready=metadata_row_contract_ready,
                sample_row_schema=list(METADATA_ROW_SCHEMA),
                sample_row_limit=METADATA_ROW_LIMIT,
                blocked_payload_fields=list(BLOCKED_METADATA_PAYLOAD_FIELDS),
                natural_language_routing_enabled=False,
                enablement_gates=6,
                preflight_handoff=preflight_handoff,
                preflight_handoff_ready=preflight_handoff["handoff_ready"],
                preflight_ready_for_operator=preflight_handoff["ready_for_operator"],
                preflight_state_changed=preflight_handoff["state_changed"],
                preflight_changed=preflight_handoff["changed"],
                preflight_content_in_handoff=preflight_handoff["content_in_handoff"],
                preflight_authorizes_execution=preflight_handoff["authorizes_execution"],
                preflight_authorizes_completion_claim=preflight_handoff["authorizes_completion_claim"],
                preflight_approval_granted=preflight_handoff["approval_granted"],
                preflight_boundaries=preflight_handoff["boundaries"],
                preflight_next_safe_commands=preflight_handoff["next_safe_commands"],
            ),
        )

    def integration_enablement_gate(args: dict[str, Any]) -> ToolResult:
        connector = _clean_text(args.get("connector") or "calendar", limit=MAX_CONNECTOR_CHARS) or "calendar"
        action = _clean_text(args.get("action"), limit=MAX_ACTION_CHARS)
        target = _clean_text(args.get("target"), limit=MAX_SCOPE_FIELD_CHARS)
        time_range = _clean_text(args.get("time_range"), limit=MAX_SCOPE_FIELD_CHARS)
        data_level = _clean_text(args.get("data_level"), limit=MAX_SCOPE_FIELD_CHARS)
        verification = _clean_text(args.get("verification"), limit=MAX_SCOPE_FIELD_CHARS)
        rollback = _clean_text(args.get("rollback"), limit=MAX_SCOPE_FIELD_CHARS)
        tests = _clean_text(args.get("tests") or args.get("test_plan"), limit=MAX_SCOPE_FIELD_CHARS)
        audit = _clean_text(args.get("audit") or args.get("audit_plan"), limit=MAX_SCOPE_FIELD_CHARS)
        acceptance = _clean_text(args.get("acceptance") or args.get("acceptance_gate") or args.get("evidence") or args.get("proof"), limit=MAX_SCOPE_FIELD_CHARS)
        if not action:
            return _action_required_failure("integration_enablement_gate")

        normalized, profile = _connector_profile(connector)
        risk, approval_required, reason = _classify_connector_action(action)
        data_low = data_level.lower()
        metadata_only = "metadata" in data_low and not any(word in data_low for word in ("full", "body", "content", "attachment", "transcript", "notes"))
        adapter_acceptance_required = metadata_only and risk != "EXTERNAL_SIDE_EFFECT"
        adapter_acceptance_state = "NOT_REQUIRED"
        adapter_acceptance_passed = False
        adapter_acceptance_case_count = 0
        adapter_acceptance_receipt = "not required for this action"
        adapter_acceptance_handoff: dict[str, Any] | None = None
        metadata_row_contract_ready = not adapter_acceptance_required
        missing: list[str] = []
        if not target:
            missing.append("target/source")
        if normalized in {"calendar", "email", "messages", "browser", "reminders"} and not time_range:
            missing.append("time range or selected item")
        if not data_level:
            missing.append("data level")
        if approval_required and not verification:
            missing.append("verification target")
        if risk == "EXTERNAL_SIDE_EFFECT" and not rollback:
            missing.append("rollback/cancel path")
        if not tests:
            missing.append("smoke test plan")
        if not audit:
            missing.append("audit trail plan")
        if not acceptance:
            missing.append("acceptance-gate evidence")

        if adapter_acceptance_required and target and time_range:
            acceptance_result = integration_adapter_acceptance(
                {
                    "connector": normalized,
                    "target": target,
                    "time_range": time_range,
                }
            )
            acceptance_metadata = acceptance_result.metadata or {}
            adapter_acceptance_state = str(acceptance_metadata.get("acceptance_state") or "ADAPTER_ACCEPTANCE_BLOCKED")
            adapter_acceptance_passed = adapter_acceptance_state == "ADAPTER_ACCEPTANCE_PASSED"
            adapter_acceptance_case_count = int(acceptance_metadata.get("passed_cases") or 0)
            adapter_acceptance_handoff = acceptance_metadata.get("adapter_acceptance_handoff")
            metadata_row_contract_ready = (
                acceptance_metadata.get("sample_row_schema") == METADATA_ROW_SCHEMA
                and acceptance_metadata.get("sample_row_limit") == METADATA_ROW_LIMIT
                and acceptance_metadata.get("all_case_row_contracts_present") is True
            )
            adapter_acceptance_receipt = (
                f"{adapter_acceptance_state}; "
                f"{adapter_acceptance_case_count}/{acceptance_metadata.get('total_cases', 0)} disabled adapter cases passed"
            )
            if not adapter_acceptance_passed:
                missing.append("disabled adapter acceptance proof")
            if not metadata_row_contract_ready:
                missing.append("metadata row contract proof")
        elif adapter_acceptance_required:
            adapter_acceptance_state = "ADAPTER_ACCEPTANCE_SCOPE_MISSING"
            adapter_acceptance_receipt = "target/source and time range are required before disabled adapter acceptance can run"
            metadata_row_contract_ready = False
            missing.append("disabled adapter acceptance proof")

        if missing:
            verdict = "ENABLEMENT_BLOCKED"
            decision = "Do not enable or implement the connector route yet."
        elif risk == "READ_ONLY":
            verdict = "READY_FOR_READ_ONLY_ENABLEMENT_REVIEW"
            decision = "A narrow read-only connector prototype may be reviewed, with natural-language routing still disabled until tests pass."
        else:
            verdict = "READY_FOR_APPROVAL_GATED_ENABLEMENT_REVIEW"
            decision = "The connector may be reviewed only as an approval-gated implementation; real use must still stop for a one-shot approval and approval chain proof."

        implementation_name = f"future_{normalized}_{'_'.join(action.lower().split()[:4]) or 'connector'}"
        tool_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in implementation_name).strip("_") or f"future_{normalized}_connector"
        acceptance_plan = acceptance or "acceptance gate: changed behavior, evidence receipt, tests, and rollback/stop condition are all proven"
        test_plan = tests or "focused smoke test plus blocked personal-data/side-effect regression"
        audit_plan = audit or "ToolRegistry run record with connector, action, scope, risk, approval id if any, verification target, and result"
        approval_check = "not required for read-only/metadata-only review" if not approval_required else "required before any real personal-data read or side effect"
        implementation_review_allowed = verdict in {"READY_FOR_READ_ONLY_ENABLEMENT_REVIEW", "READY_FOR_APPROVAL_GATED_ENABLEMENT_REVIEW"}
        enablement_handoff = _enablement_gate_handoff_payload(
            connector=normalized,
            requested_connector=connector,
            action=action,
            tool_name=tool_name,
            enablement_verdict=verdict,
            missing_fields=missing,
            target=target,
            time_range=time_range,
            data_level=data_level,
            verification=verification,
            rollback=rollback,
            tests=tests,
            audit=audit,
            acceptance=acceptance,
            adapter_acceptance_required=adapter_acceptance_required,
            adapter_acceptance_state=adapter_acceptance_state,
            adapter_acceptance_passed=adapter_acceptance_passed,
            adapter_acceptance_case_count=adapter_acceptance_case_count,
            adapter_acceptance_handoff=adapter_acceptance_handoff if isinstance(adapter_acceptance_handoff, dict) else None,
            metadata_row_contract_required=adapter_acceptance_required,
            metadata_row_contract_ready=metadata_row_contract_ready,
            suggested_risk=risk,
            approval_required=approval_required,
            rollback_required=risk == "EXTERNAL_SIDE_EFFECT",
            implementation_review_allowed=implementation_review_allowed,
        )

        lines = [
            "Jarvis personal integration enablement gate:",
            "This is read-only. It is the final harness gate before a future connector can be enabled for review; it does not connect accounts, read personal data, call external services, execute tools, change state, control the computer, approve requests, or queue approvals.",
            "",
            f"Connector: {connector}",
            f"Normalized connector: {normalized}",
            f"Requested action: {action}",
            f"Enablement verdict: {verdict}",
            f"Suggested risk: {risk}",
            f"Approval required for real use: {approval_check}",
            f"Reason: {reason}",
            "",
            "Evidence checked:",
            f"- target/source: {target or '<missing>'}",
            f"- time range or selected item: {time_range or '<missing>'}",
            f"- data level: {data_level or '<missing>'}",
            f"- verification target: {verification or '<missing>'}",
            f"- rollback/cancel path: {rollback or '<missing>'}",
            f"- smoke tests: {test_plan}",
            f"- audit trail: {audit_plan}",
            f"- acceptance gate: {acceptance_plan}",
            f"- disabled adapter acceptance: {adapter_acceptance_receipt}",
            f"- metadata row contract: {'ready' if metadata_row_contract_ready else 'blocked'}",
            f"- metadata row schema: {', '.join(METADATA_ROW_SCHEMA)}",
            f"- metadata row limit: {METADATA_ROW_LIMIT}",
            f"- missing blockers: {', '.join(missing) if missing else 'none'}",
            "",
            "Enablement decision:",
            f"- {decision}",
            f"- future tool: {tool_name}",
            "- default state: disabled for natural-language auto-routing.",
            "- any personal-data or side-effect path must stop at approval readiness, a one-shot approval packet, and approval chain proof before real use.",
            "",
            "Required proof before enabling:",
            "- focused happy-path smoke test for the exact connector/action/scope.",
            "- blocked missing-scope smoke test that proves connector code is not reached.",
            "- disabled adapter acceptance proof for metadata-only routes: metadata happy path, metadata row contract, full-content blocked, side-effect blocked, and missing-scope blocked.",
            "- approval-path smoke test for any personal-data read or side effect.",
            "- audit receipt with connector, action, target, time range, data level, risk, approval id if any, verification target, result, and acceptance evidence.",
            "- rollback or stop-condition receipt before claiming the slice complete.",
            "",
            "Acceptance gate:",
            f"- {acceptance_plan}",
            "- Treat the slice as incomplete if evidence, verification, rollback/stop condition, or audit receipt is missing.",
            "",
            "Planner routing state:",
            "- natural-language auto-routing: disabled.",
            "- explicit planner command only until the focused smoke suite passes.",
            "- no broad account ingestion, token capture, cookie capture, or authenticated browser shortcuts.",
            "",
            "Known connector boundaries:",
            "- READ_ONLY candidates: " + ", ".join(profile["read_only"]) + ".",
            "- PERSONAL_DATA candidates: " + ", ".join(profile["personal_data"]) + ".",
            "- EXTERNAL_SIDE_EFFECT or HIGH_RISK candidates: " + ", ".join(profile["side_effects"]) + ".",
            "",
            "Stop conditions:",
            OPERATOR_LIMIT_STOP,
            "- Stop if any exact argument is missing or changed since this gate was generated.",
            "- Stop if implementation would store tokens, cookies, full email/message bodies, contact cards, attachments, or broad calendar notes by default.",
            "- Stop if a personal-data or side-effect action can run without one-shot approval and approval chain proof.",
            "- Stop if verification, audit, rollback, or acceptance evidence is ambiguous.",
        ]
        return ToolResult(
            "integration_enablement_gate",
            True,
            "\n".join(lines),
            _safe_metadata(
                connector=normalized,
                requested_connector=connector,
                action=action,
                target=target,
                time_range=time_range,
                data_level=data_level,
                verification=verification,
                rollback=rollback,
                tests=tests,
                audit=audit,
                acceptance=acceptance,
                missing_fields=missing,
                enablement_verdict=verdict,
                adapter_acceptance_required=adapter_acceptance_required,
                adapter_acceptance_state=adapter_acceptance_state,
                adapter_acceptance_passed=adapter_acceptance_passed,
                adapter_acceptance_case_count=adapter_acceptance_case_count,
                metadata_row_contract_required=adapter_acceptance_required,
                metadata_row_contract_ready=metadata_row_contract_ready,
                sample_row_schema=list(METADATA_ROW_SCHEMA),
                sample_row_limit=METADATA_ROW_LIMIT,
                blocked_payload_fields=list(BLOCKED_METADATA_PAYLOAD_FIELDS),
                tool_name=tool_name,
                suggested_risk=risk,
                approval_required=approval_required,
                requires_approval=approval_required,
                rollback_required=risk == "EXTERNAL_SIDE_EFFECT",
                implementation_review_allowed=implementation_review_allowed,
                enablement_handoff=enablement_handoff,
                enablement_handoff_ready=enablement_handoff["handoff_ready"],
                enablement_ready_for_operator=enablement_handoff["ready_for_operator"],
                enablement_state_changed=enablement_handoff["state_changed"],
                enablement_changed=enablement_handoff["changed"],
                enablement_content_in_handoff=enablement_handoff["content_in_handoff"],
                enablement_authorizes_execution=enablement_handoff["authorizes_execution"],
                enablement_authorizes_completion_claim=enablement_handoff["authorizes_completion_claim"],
                enablement_approval_granted=enablement_handoff["approval_granted"],
                enablement_boundaries=enablement_handoff["boundaries"],
                enablement_next_safe_commands=enablement_handoff["next_safe_commands"],
                natural_language_routing_enabled=False,
                enablement_gates=7,
            ),
        )

    def integration_rehearsal_receipt(args: dict[str, Any]) -> ToolResult:
        connector = _clean_text(args.get("connector") or "calendar", limit=MAX_CONNECTOR_CHARS) or "calendar"
        action = _clean_text(args.get("action"), limit=MAX_ACTION_CHARS)
        target = _clean_text(args.get("target"), limit=MAX_SCOPE_FIELD_CHARS)
        time_range = _clean_text(args.get("time_range"), limit=MAX_SCOPE_FIELD_CHARS)
        data_level = _clean_text(args.get("data_level"), limit=MAX_SCOPE_FIELD_CHARS)
        verification = _clean_text(args.get("verification"), limit=MAX_SCOPE_FIELD_CHARS)
        rollback = _clean_text(args.get("rollback"), limit=MAX_SCOPE_FIELD_CHARS)
        tests = _clean_text(args.get("tests") or args.get("test_plan"), limit=MAX_SCOPE_FIELD_CHARS)
        audit = _clean_text(args.get("audit") or args.get("audit_plan"), limit=MAX_SCOPE_FIELD_CHARS)
        acceptance = _clean_text(args.get("acceptance") or args.get("acceptance_gate") or args.get("evidence") or args.get("proof"), limit=MAX_SCOPE_FIELD_CHARS)
        if not action:
            return _action_required_failure("integration_rehearsal_receipt")

        normalized, profile = _connector_profile(connector)
        risk, approval_required, reason = _classify_connector_action(action)
        data_low = data_level.lower()
        metadata_row_contract_required = "metadata" in data_low and not any(
            word in data_low for word in ("full", "body", "content", "attachment", "transcript", "notes")
        )
        missing: list[str] = []
        if not target:
            missing.append("target/source")
        if normalized in {"calendar", "email", "messages", "browser", "reminders"} and not time_range:
            missing.append("time range or selected item")
        if not data_level:
            missing.append("data level")
        if approval_required and not verification:
            missing.append("verification target")
        if risk == "EXTERNAL_SIDE_EFFECT" and not rollback:
            missing.append("rollback/cancel path")
        if not tests:
            missing.append("smoke test plan")
        if not audit:
            missing.append("audit trail plan")
        if not acceptance:
            missing.append("acceptance-gate evidence")

        tool_name = "".join(
            ch if ch.isalnum() or ch == "_" else "_"
            for ch in f"future_{normalized}_{'_'.join(action.lower().split()[:4]) or 'connector'}"
        ).strip("_") or f"future_{normalized}_connector"
        rehearsal_state = "REHEARSAL_BLOCKED" if missing else "REHEARSAL_READY_APPROVAL_GATED" if approval_required else "REHEARSAL_READY_READ_ONLY"
        adapter_state = "disabled"
        approval_checkpoint = "not required for read-only rehearsal" if not approval_required else "required before adapter call in any real run"
        test_plan = tests or "focused smoke test plus blocked personal-data/side-effect regression"
        audit_plan = audit or "ToolRegistry run record with connector, action, scope, risk, approval id if any, verification target, and result"
        acceptance_plan = acceptance or "acceptance gate: changed behavior, evidence receipt, tests, and rollback/stop condition are all proven"
        simulated_result = "no connector call made; rehearsal stops at disabled adapter boundary"
        metadata_row_contract_ready = metadata_row_contract_required and not missing
        rehearsal_handoff = _rehearsal_handoff_payload(
            connector=normalized,
            requested_connector=connector,
            action=action,
            tool_name=tool_name,
            rehearsal_state=rehearsal_state,
            adapter_state=adapter_state,
            simulated_result=simulated_result,
            missing_fields=missing,
            target=target,
            time_range=time_range,
            data_level=data_level,
            verification=verification,
            rollback=rollback,
            tests=tests,
            audit=audit,
            acceptance=acceptance,
            metadata_row_contract_required=metadata_row_contract_required,
            metadata_row_contract_ready=metadata_row_contract_ready,
            suggested_risk=risk,
            approval_required=approval_required,
            rollback_required=risk == "EXTERNAL_SIDE_EFFECT",
        )

        lines = [
            "Jarvis personal integration rehearsal receipt:",
            "This is read-only. It rehearses the future connector execution envelope without connecting accounts, reading personal data, calling external services, executing tools, changing state, controlling the computer, approving requests, or queuing approvals.",
            "",
            f"Connector: {connector}",
            f"Normalized connector: {normalized}",
            f"Requested action: {action}",
            f"Rehearsal state: {rehearsal_state}",
            f"Suggested risk: {risk}",
            f"Approval checkpoint: {approval_checkpoint}",
            f"Reason: {reason}",
            "",
            "Exact future execution envelope:",
            f"- future tool: {tool_name}",
            f"- adapter state: {adapter_state}",
            f"- target/source: {target or '<missing>'}",
            f"- time range or selected item: {time_range or '<missing>'}",
            f"- data level: {data_level or '<missing>'}",
            f"- verification target: {verification or '<missing>'}",
            f"- rollback/cancel path: {rollback or '<missing>'}",
            f"- missing blockers: {', '.join(missing) if missing else 'none'}",
            f"- metadata row contract: {'ready' if metadata_row_contract_ready else 'not required for this rehearsal' if not metadata_row_contract_required else 'blocked'}",
            f"- metadata row schema: {', '.join(METADATA_ROW_SCHEMA)}",
            f"- metadata row limit: {METADATA_ROW_LIMIT}",
            f"- blocked payload fields: {', '.join(BLOCKED_METADATA_PAYLOAD_FIELDS)}",
            "",
            "Rehearsed lifecycle:",
            "- 1. preflight: normalize connector/action, classify risk, and validate exact scope.",
            "- 2. gate: stop if scope, tests, audit, acceptance evidence, verification, or rollback proof is missing.",
            f"- 3. approval: {approval_checkpoint}.",
            "- 4. adapter boundary: real connector adapter remains disabled in this rehearsal.",
            f"- 5. simulated result: {simulated_result}.",
            "- 6. verification: compare the simulated result with the declared verification target without claiming real completion.",
            "- 7. rollback: show the rollback/cancel path but do not mutate outside-world state.",
            "- 8. audit: record only bounded rehearsal metadata, not raw private payloads.",
            "",
            "Proof carried into implementation:",
            f"- smoke tests: {test_plan}",
            f"- audit trail: {audit_plan}",
            f"- acceptance gate: {acceptance_plan}",
            "- natural-language auto-routing remains disabled until the real adapter has focused happy-path and blocked-action tests.",
            "",
            "Known connector boundaries:",
            "- READ_ONLY candidates: " + ", ".join(profile["read_only"]) + ".",
            "- PERSONAL_DATA candidates: " + ", ".join(profile["personal_data"]) + ".",
            "- EXTERNAL_SIDE_EFFECT or HIGH_RISK candidates: " + ", ".join(profile["side_effects"]) + ".",
            "",
            "Stop conditions:",
            OPERATOR_LIMIT_STOP,
            "- Stop if this rehearsal would need tokens, cookies, full private payloads, broad account ingestion, or authenticated browser shortcuts.",
            "- Stop if personal-data reads or side effects can bypass the one-shot approval checkpoint and approval chain proof.",
            "- Stop if audit, verification, rollback, acceptance evidence, or exact scope is missing.",
            "- Stop if a future adapter implementation stores raw private content by default.",
        ]
        return ToolResult(
            "integration_rehearsal_receipt",
            True,
            "\n".join(lines),
            _safe_metadata(
                connector=normalized,
                requested_connector=connector,
                action=action,
                target=target,
                time_range=time_range,
                data_level=data_level,
                verification=verification,
                rollback=rollback,
                tests=tests,
                audit=audit,
                acceptance=acceptance,
                missing_fields=missing,
                rehearsal_state=rehearsal_state,
                tool_name=tool_name,
                adapter_state=adapter_state,
                simulated_result=simulated_result,
                metadata_row_contract_required=metadata_row_contract_required,
                metadata_row_contract_ready=metadata_row_contract_ready,
                sample_row_schema=list(METADATA_ROW_SCHEMA),
                sample_row_limit=METADATA_ROW_LIMIT,
                blocked_payload_fields=list(BLOCKED_METADATA_PAYLOAD_FIELDS),
                suggested_risk=risk,
                approval_required=approval_required,
                requires_approval=approval_required,
                rollback_required=risk == "EXTERNAL_SIDE_EFFECT",
                natural_language_routing_enabled=False,
                rehearsal_ready=not missing,
                rehearsal_handoff=rehearsal_handoff,
                rehearsal_handoff_ready=rehearsal_handoff["handoff_ready"],
                rehearsal_ready_for_operator=rehearsal_handoff["ready_for_operator"],
                rehearsal_state_changed=rehearsal_handoff["state_changed"],
                rehearsal_changed=rehearsal_handoff["changed"],
                rehearsal_content_in_handoff=rehearsal_handoff["content_in_handoff"],
                rehearsal_authorizes_execution=rehearsal_handoff["authorizes_execution"],
                rehearsal_authorizes_completion_claim=rehearsal_handoff["authorizes_completion_claim"],
                rehearsal_approval_granted=rehearsal_handoff["approval_granted"],
                rehearsal_boundaries=rehearsal_handoff["boundaries"],
                rehearsal_next_safe_commands=rehearsal_handoff["next_safe_commands"],
            ),
        )

    def integration_metadata_preview(args: dict[str, Any]) -> ToolResult:
        connector = _clean_text(args.get("connector") or "calendar", limit=MAX_CONNECTOR_CHARS) or "calendar"
        action = _clean_text(args.get("action"), limit=MAX_ACTION_CHARS)
        target = _clean_text(args.get("target"), limit=MAX_SCOPE_FIELD_CHARS)
        time_range = _clean_text(args.get("time_range"), limit=MAX_SCOPE_FIELD_CHARS)
        data_level = _clean_text(args.get("data_level"), limit=MAX_SCOPE_FIELD_CHARS)
        verification = _clean_text(args.get("verification"), limit=MAX_SCOPE_FIELD_CHARS)
        tests = _clean_text(args.get("tests") or args.get("test_plan"), limit=MAX_SCOPE_FIELD_CHARS)
        audit = _clean_text(args.get("audit") or args.get("audit_plan"), limit=MAX_SCOPE_FIELD_CHARS)
        acceptance = _clean_text(args.get("acceptance") or args.get("acceptance_gate") or args.get("evidence") or args.get("proof"), limit=MAX_SCOPE_FIELD_CHARS)
        if not action:
            return _action_required_failure("integration_metadata_preview")

        normalized, profile = _connector_profile(connector)
        risk, approval_required, reason = _classify_connector_action(action)
        data_low = data_level.lower()
        metadata_only = "metadata" in data_low and not any(word in data_low for word in ("full", "body", "content", "attachment", "transcript", "notes"))
        side_effect_requested = risk == "EXTERNAL_SIDE_EFFECT"
        missing: list[str] = []
        if not target:
            missing.append("target/source")
        if normalized in {"calendar", "email", "messages", "browser", "reminders"} and not time_range:
            missing.append("time range or selected item")
        if not data_level:
            missing.append("data level")
        elif not metadata_only:
            missing.append("metadata-only data level")
        if side_effect_requested:
            missing.append("non-side-effect action")
        if approval_required and not verification:
            missing.append("verification target")
        if not tests:
            missing.append("smoke test plan")
        if not audit:
            missing.append("audit trail plan")
        if not acceptance:
            missing.append("acceptance-gate evidence")

        preview_state = "METADATA_PREVIEW_BLOCKED" if missing else "METADATA_PREVIEW_READY"
        adapter_state = "metadata adapter disabled"
        tool_name = "".join(
            ch if ch.isalnum() or ch == "_" else "_"
            for ch in f"future_{normalized}_{'_'.join(action.lower().split()[:4]) or 'metadata_preview'}"
        ).strip("_") or f"future_{normalized}_metadata_preview"
        sample_rows = _fake_metadata_rows(normalized, target=target, time_range=time_range)
        row_contract = _metadata_row_contract(sample_rows)
        test_plan = tests or "focused metadata preview smoke plus blocked full-content and side-effect regressions"
        audit_plan = audit or "ToolRegistry run record with connector, action, metadata-only scope, risk, adapter disabled state, verification target, and result"
        acceptance_plan = acceptance or "acceptance gate: fake rows only, no account access, blocked private/full-content paths, audit receipt, and verification evidence"
        metadata_preview_handoff = _metadata_preview_handoff_payload(
            connector=normalized,
            requested_connector=connector,
            action=action,
            tool_name=tool_name,
            preview_state=preview_state,
            adapter_state=adapter_state,
            metadata_only=metadata_only,
            missing_fields=missing,
            target=target,
            time_range=time_range,
            data_level=data_level,
            verification=verification,
            tests=tests,
            audit=audit,
            acceptance=acceptance,
            sample_rows=sample_rows,
            row_contract=row_contract,
            suggested_risk=risk,
            approval_required=approval_required,
        )

        lines = [
            "Jarvis personal integration metadata preview:",
            "This is read-only. It previews a future connector metadata adapter with fake bounded rows only; it does not connect accounts, read personal data, call external services, execute tools, change state, control the computer, approve requests, or queue approvals.",
            "",
            f"Connector: {connector}",
            f"Normalized connector: {normalized}",
            f"Requested action: {action}",
            f"Preview state: {preview_state}",
            f"Suggested risk before metadata clamp: {risk}",
            f"Metadata clamp: {'active' if metadata_only else 'blocked until data level is metadata-only'}",
            f"Adapter state: {adapter_state}",
            f"Reason: {reason}",
            "",
            "Exact metadata scope:",
            f"- future tool: {tool_name}",
            f"- target/source: {target or '<missing>'}",
            f"- time range or selected item: {time_range or '<missing>'}",
            f"- data level: {data_level or '<missing>'}",
            f"- verification target: {verification or '<missing>'}",
            f"- missing blockers: {', '.join(missing) if missing else 'none'}",
            "",
            "Fake preview rows:",
            "- sample-1: id, timestamp, label, source only.",
            "- sample-2: id, timestamp, label, source only.",
            "- no email bodies, message text, calendar notes, contact cards, attachments, tokens, cookies, or authenticated page content.",
            "",
            "Row schema contract:",
            f"- allowed fields: {', '.join(row_contract['sample_row_schema'])}",
            f"- row limit: {row_contract['sample_row_limit']}",
            f"- bounded rows: {'yes' if row_contract['sample_rows_bounded'] else 'no'}",
            f"- blocked payload fields: {', '.join(row_contract['blocked_payload_fields'])}",
            "",
            "Adapter contract:",
            "- adapter boundary: disabled; this preview returns deterministic fake rows only.",
            "- allowed future payload: ids, timestamps, sender/owner labels, title/status labels, and source handles only when the operator approved or the connector is genuinely read-only.",
            "- blocked future payload: full private content, attachments, broad account ingestion, cookies, tokens, and hidden authenticated browser state.",
            "- side-effect actions cannot be represented by this metadata preview and must stay behind approval-gated implementation receipts.",
            "",
            "Proof carried into implementation:",
            f"- smoke tests: {test_plan}",
            f"- audit trail: {audit_plan}",
            f"- acceptance gate: {acceptance_plan}",
            "- natural-language auto-routing remains disabled until metadata-only, blocked full-content, blocked side-effect, and audit tests pass.",
            "",
            "Known connector boundaries:",
            "- READ_ONLY candidates: " + ", ".join(profile["read_only"]) + ".",
            "- PERSONAL_DATA candidates: " + ", ".join(profile["personal_data"]) + ".",
            "- EXTERNAL_SIDE_EFFECT or HIGH_RISK candidates: " + ", ".join(profile["side_effects"]) + ".",
            "",
            "Stop conditions:",
            OPERATOR_LIMIT_STOP,
            "- Stop if the request needs full private content, account tokens, cookies, attachments, authenticated page content, or broad ingestion.",
            "- Stop if the action is send/create/delete/update/submit/mark-read or another side effect.",
            "- Stop if exact scope, smoke tests, audit trail, acceptance evidence, or verification target is missing.",
        ]
        return ToolResult(
            "integration_metadata_preview",
            True,
            "\n".join(lines),
            _safe_metadata(
                connector=normalized,
                requested_connector=connector,
                action=action,
                target=target,
                time_range=time_range,
                data_level=data_level,
                verification=verification,
                tests=tests,
                audit=audit,
                acceptance=acceptance,
                missing_fields=missing,
                preview_state=preview_state,
                tool_name=tool_name,
                adapter_state=adapter_state,
                sample_rows=sample_rows,
                sample_row_count=len(sample_rows),
                suggested_risk=risk,
                approval_required=approval_required,
                requires_approval=approval_required,
                metadata_only=metadata_only,
                natural_language_routing_enabled=False,
                metadata_preview_ready=not missing,
                metadata_preview_handoff=metadata_preview_handoff,
                metadata_preview_handoff_ready=metadata_preview_handoff["handoff_ready"],
                metadata_preview_ready_for_operator=metadata_preview_handoff["ready_for_operator"],
                metadata_preview_state_changed=metadata_preview_handoff["state_changed"],
                metadata_preview_changed=metadata_preview_handoff["changed"],
                metadata_preview_content_in_handoff=metadata_preview_handoff["content_in_handoff"],
                metadata_preview_authorizes_execution=metadata_preview_handoff["authorizes_execution"],
                metadata_preview_authorizes_completion_claim=metadata_preview_handoff["authorizes_completion_claim"],
                metadata_preview_approval_granted=metadata_preview_handoff["approval_granted"],
                metadata_preview_boundaries=metadata_preview_handoff["boundaries"],
                metadata_preview_next_safe_commands=metadata_preview_handoff["next_safe_commands"],
                **row_contract,
            ),
        )

    def integration_proof_bundle(args: dict[str, Any]) -> ToolResult:
        connector = _clean_text(args.get("connector") or "calendar", limit=MAX_CONNECTOR_CHARS) or "calendar"
        action = _clean_text(args.get("action"), limit=MAX_ACTION_CHARS)
        target = _clean_text(args.get("target"), limit=MAX_SCOPE_FIELD_CHARS)
        time_range = _clean_text(args.get("time_range"), limit=MAX_SCOPE_FIELD_CHARS)
        data_level = _clean_text(args.get("data_level"), limit=MAX_SCOPE_FIELD_CHARS)
        verification = _clean_text(args.get("verification"), limit=MAX_SCOPE_FIELD_CHARS)
        rollback = _clean_text(args.get("rollback"), limit=MAX_SCOPE_FIELD_CHARS)
        tests = _clean_text(args.get("tests") or args.get("test_plan"), limit=MAX_SCOPE_FIELD_CHARS)
        audit = _clean_text(args.get("audit") or args.get("audit_plan"), limit=MAX_SCOPE_FIELD_CHARS)
        acceptance = _clean_text(args.get("acceptance") or args.get("acceptance_gate") or args.get("evidence") or args.get("proof"), limit=MAX_SCOPE_FIELD_CHARS)
        if not action:
            return _action_required_failure("integration_proof_bundle")

        normalized, profile = _connector_profile(connector)
        risk, approval_required, reason = _classify_connector_action(action)
        bundle_args = {
            "connector": normalized,
            "action": action,
            "target": target,
            "time_range": time_range,
            "data_level": data_level,
            "verification": verification,
            "rollback": rollback,
            "tests": tests,
            "audit": audit,
            "acceptance": acceptance,
        }
        metadata_preview = integration_metadata_preview(bundle_args)
        adapter_acceptance = integration_adapter_acceptance(
            {
                "connector": normalized,
                "target": target,
                "time_range": time_range,
            }
        )
        enablement_gate = integration_enablement_gate(bundle_args)
        rehearsal_receipt = integration_rehearsal_receipt(bundle_args)

        preview_metadata = metadata_preview.metadata or {}
        acceptance_metadata = adapter_acceptance.metadata or {}
        enablement_metadata = enablement_gate.metadata or {}
        rehearsal_metadata = rehearsal_receipt.metadata or {}
        row_schema_contract_passed = (
            preview_metadata.get("sample_row_schema") == METADATA_ROW_SCHEMA
            and preview_metadata.get("sample_row_limit") == METADATA_ROW_LIMIT
            and acceptance_metadata.get("sample_row_schema") == METADATA_ROW_SCHEMA
            and acceptance_metadata.get("sample_row_limit") == METADATA_ROW_LIMIT
            and acceptance_metadata.get("all_case_row_contracts_present") is True
        )

        proof_checks = [
            {
                "name": "metadata preview",
                "state": preview_metadata.get("preview_state"),
                "passed": _metadata_bool(preview_metadata.get("metadata_preview_ready")),
                "next": f"integration metadata preview: {normalized} -> {action}; target {target or '<target>'}; time {time_range or '<time range>'}; data metadata-only; verification {verification or 'fake rows only'}; tests {tests or 'blocked full-content smoke'}; audit {audit or 'metadata preview receipt'}; acceptance {acceptance or 'acceptance gate passed'}",
            },
            {
                "name": "disabled adapter acceptance",
                "state": acceptance_metadata.get("acceptance_state"),
                "passed": _metadata_bool(acceptance_metadata.get("adapter_acceptance_passed")),
                "next": f"integration adapter acceptance: {normalized}",
            },
            {
                "name": "metadata row contract",
                "state": "ROW_SCHEMA_CONTRACT_READY" if row_schema_contract_passed else "ROW_SCHEMA_CONTRACT_BLOCKED",
                "passed": row_schema_contract_passed,
                "next": f"integration adapter manifest: {normalized}",
            },
            {
                "name": "enablement gate",
                "state": enablement_metadata.get("enablement_verdict"),
                "passed": _metadata_bool(enablement_metadata.get("implementation_review_allowed")) and not bool(enablement_metadata.get("missing_fields")),
                "next": f"integration enablement gate: {normalized} -> {action}; target {target or '<target>'}; time {time_range or '<time range>'}; data {data_level or '<data level>'}; verification {verification or '<verification>'}; rollback {rollback or 'no connector call'}; tests {tests or '<tests>'}; audit {audit or '<audit>'}; acceptance {acceptance or '<acceptance gate>'}",
            },
            {
                "name": "rehearsal receipt",
                "state": rehearsal_metadata.get("rehearsal_state"),
                "passed": _metadata_bool(rehearsal_metadata.get("rehearsal_ready")),
                "next": f"integration rehearsal receipt: {normalized} -> {action}; target {target or '<target>'}; time {time_range or '<time range>'}; data {data_level or '<data level>'}; verification {verification or '<verification>'}; rollback {rollback or 'no connector call'}; tests {tests or '<tests>'}; audit {audit or '<audit>'}; acceptance {acceptance or '<acceptance gate>'}",
            },
        ]
        passed_checks = sum(1 for check in proof_checks if check["passed"])
        missing_blockers = sorted(
            {
                str(item)
                for source in (
                    preview_metadata.get("missing_fields") or [],
                    enablement_metadata.get("missing_fields") or [],
                    rehearsal_metadata.get("missing_fields") or [],
                )
                for item in source
            }
        )
        bundle_state = "PROOF_BUNDLE_READY_FOR_IMPLEMENTATION_REVIEW" if passed_checks == len(proof_checks) and not missing_blockers else "PROOF_BUNDLE_BLOCKED"
        scope_hash = _integration_scope_hash(
            normalized,
            action,
            target,
            time_range,
            data_level,
            verification,
            rollback,
            tests,
            audit,
            acceptance,
        )
        proof_bundle_command = _integration_proof_bundle_command(
            normalized,
            action,
            target,
            time_range,
            data_level,
            verification,
            rollback,
            tests,
            audit,
            acceptance,
        )
        implementation_review_command = _integration_implementation_review_command(
            normalized,
            action,
            target,
            time_range,
            data_level,
            verification,
            rollback,
            tests,
            audit,
            acceptance,
            "<status API and smoke evidence>",
        )
        review_receipt_id = f"integration-{scope_hash[:12]}"
        receipt_contract_rows = _implementation_review_receipt_contract_rows(
            review_ready=False,
            scope_hash=scope_hash,
            review_receipt_id=review_receipt_id,
        )
        receipt_contract_summary = [
            "proof-bundle-only",
            "implementation-review-required",
            "account-access-not-authorized",
            "route-unlock-not-authorized",
            "fresh-scope-review-required",
        ]
        proof_queue = _ordered_commands(
            [str(check["next"]) for check in proof_checks]
            + [
                "execution health report",
                implementation_review_command,
            ]
        )
        next_safe_command = "execution health report" if bundle_state.endswith("REVIEW") else next(check["next"] for check in proof_checks if not check["passed"])
        next_proof_command = "execution health report" if bundle_state.endswith("REVIEW") else proof_queue[0]
        tool_name = str(enablement_metadata.get("tool_name") or preview_metadata.get("tool_name") or f"future_{normalized}_connector")
        proof_bundle_handoff = _proof_bundle_handoff_payload(
            connector=normalized,
            action=action,
            tool_name=tool_name,
            bundle_state=bundle_state,
            proof_checks=proof_checks,
            passed_proof_checks=passed_checks,
            proof_queue=proof_queue,
            next_proof_command=next_proof_command,
            next_safe_command=next_safe_command,
            missing_fields=missing_blockers,
            review_receipt_id=review_receipt_id,
            integration_scope_hash=scope_hash,
            proof_bundle_command=proof_bundle_command,
            implementation_review_command=implementation_review_command,
            receipt_contract_rows=receipt_contract_rows,
            receipt_contract_summary=receipt_contract_summary,
            metadata_preview_state=preview_metadata.get("preview_state"),
            adapter_acceptance_state=acceptance_metadata.get("acceptance_state"),
            enablement_verdict=enablement_metadata.get("enablement_verdict"),
            rehearsal_state=rehearsal_metadata.get("rehearsal_state"),
            metadata_row_contract_ready=row_schema_contract_passed,
            suggested_risk=risk,
            approval_required=approval_required,
        )

        lines = [
            "Jarvis personal integration proof bundle:",
            "This is read-only. It bundles metadata preview, disabled adapter acceptance, metadata row contract, enablement gate, rehearsal receipt, audit, verification, rollback, and stop-condition proof before any real connector implementation; it does not connect accounts, read personal data, call external services, execute tools, change state, control the computer, approve requests, or queue approvals.",
            "",
            f"Connector: {connector}",
            f"Normalized connector: {normalized}",
            f"Requested action: {action}",
            f"Bundle state: {bundle_state}",
            f"Suggested risk: {risk}",
            f"Approval required for real use: {'yes' if approval_required else 'no, while implementation stays metadata-only/read-only'}",
            f"Reason: {reason}",
            "",
            "Exact scope:",
            f"- future tool: {tool_name}",
            f"- target/source: {target or '<missing>'}",
            f"- time range or selected item: {time_range or '<missing>'}",
            f"- data level: {data_level or '<missing>'}",
            f"- verification target: {verification or '<missing>'}",
            f"- rollback/cancel path: {rollback or '<missing>'}",
            f"- smoke tests: {tests or '<missing>'}",
            f"- audit trail: {audit or '<missing>'}",
            f"- acceptance gate: {acceptance or '<missing>'}",
            f"- missing blockers: {', '.join(missing_blockers) if missing_blockers else 'none'}",
            f"- metadata row schema: {', '.join(METADATA_ROW_SCHEMA)}",
            f"- metadata row limit: {METADATA_ROW_LIMIT}",
            f"- blocked payload fields: {', '.join(BLOCKED_METADATA_PAYLOAD_FIELDS)}",
            "",
            "Proof chain:",
        ]
        for check in proof_checks:
            lines.extend(
                [
                    f"- {check['name']}: {'pass' if check['passed'] else 'blocked'}",
                    f"  state: {check['state']}",
                    f"  next: `{check['next']}`",
                ]
            )

        lines.extend(
            [
                "",
            "Implementation review posture:",
                f"- review receipt id: `{review_receipt_id}`",
                f"- scope hash: `{scope_hash}`",
                f"- proof bundle command: `{proof_bundle_command}`",
                f"- implementation review command: `{implementation_review_command}`",
                f"- receipt contract rows: {len(receipt_contract_rows)}",
                "- receipt contract summary: " + ", ".join(receipt_contract_summary),
                f"- next safe command: `{next_safe_command}`",
                f"- next required command: `{next_proof_command}`",
                f"- proof queue count: {len(proof_queue)}",
                "- proof queue: " + ", ".join(f"`{command}`" for command in proof_queue[:7]),
                "- route blocker: connector code review remains blocked until this exact scope hash reaches implementation review with status/API smoke evidence.",
                "- natural-language auto-routing: disabled.",
                "- real adapter calls remain disabled until the focused personal smoke suite, status/API visibility, execution health report, audit receipt, verification target, and rollback/stop evidence all pass.",
                "- personal-data reads and side effects still require approval readiness, a fresh one-shot approval packet, and approval chain proof for the exact account/source/target/time/action.",
                "",
                "Known connector boundaries:",
                "- READ_ONLY candidates: " + ", ".join(profile["read_only"]) + ".",
                "- PERSONAL_DATA candidates: " + ", ".join(profile["personal_data"]) + ".",
                "- EXTERNAL_SIDE_EFFECT or HIGH_RISK candidates: " + ", ".join(profile["side_effects"]) + ".",
                "",
                "Stop conditions:",
                OPERATOR_LIMIT_STOP,
                "- Stop if any proof check is blocked, stale, or scoped to a different connector/action/target/time/data level.",
                "- Stop if implementation would store tokens, cookies, full private content, contact cards, attachments, authenticated browser state, or broad account exports by default.",
                "- Stop if natural-language routing can reach a connector before explicit command smoke tests pass.",
                "- Stop if personal-data or side-effect execution can bypass approval readiness, approval packet, approval chain proof, audit, verification, and rollback proof.",
            ]
        )
        return ToolResult(
            "integration_proof_bundle",
            True,
            "\n".join(lines),
            _safe_metadata(
                connector=normalized,
                requested_connector=connector,
                action=action,
                target=target,
                time_range=time_range,
                data_level=data_level,
                verification=verification,
                rollback=rollback,
                tests=tests,
                audit=audit,
                acceptance=acceptance,
                bundle_state=bundle_state,
                proof_checks=proof_checks,
                proof_check_count=len(proof_checks),
                passed_proof_checks=passed_checks,
                proof_queue=proof_queue,
                proof_queue_count=len(proof_queue),
                next_required_command=next_proof_command,
                next_proof_command=next_proof_command,
                integration_proof_queue=proof_queue,
                integration_proof_queue_count=len(proof_queue),
                integration_next_required_command=next_proof_command,
                integration_next_proof_command=next_proof_command,
                review_receipt_id=review_receipt_id,
                integration_scope_hash=scope_hash,
                proof_bundle_command=proof_bundle_command,
                implementation_review_command=implementation_review_command,
                implementation_review_receipt_contract_rows=receipt_contract_rows,
                implementation_review_receipt_contract_row_count=len(receipt_contract_rows),
                implementation_review_receipt_contract_ready=False,
                implementation_review_receipt_contract_summary=receipt_contract_summary,
                implementation_review_receipt_authorizes_account_access=False,
                implementation_review_receipt_authorizes_route_unlock=False,
                implementation_review_receipt_reusable_for_other_scope=False,
                route_blocked_until_implementation_review=True,
                can_open_code_review_after_receipt=False,
                next_safe_command=next_safe_command,
                missing_fields=missing_blockers,
                tool_name=tool_name,
                suggested_risk=risk,
                approval_required=approval_required,
                requires_approval=approval_required,
                metadata_preview_state=preview_metadata.get("preview_state"),
                adapter_acceptance_state=acceptance_metadata.get("acceptance_state"),
                enablement_verdict=enablement_metadata.get("enablement_verdict"),
                rehearsal_state=rehearsal_metadata.get("rehearsal_state"),
                sample_row_schema=list(METADATA_ROW_SCHEMA),
                sample_row_limit=METADATA_ROW_LIMIT,
                blocked_payload_fields=list(BLOCKED_METADATA_PAYLOAD_FIELDS),
                metadata_row_contract_ready=row_schema_contract_passed,
                implementation_review_allowed=bundle_state == "PROOF_BUNDLE_READY_FOR_IMPLEMENTATION_REVIEW",
                proof_bundle_handoff=proof_bundle_handoff,
                proof_bundle_handoff_ready=proof_bundle_handoff["handoff_ready"],
                proof_bundle_ready_for_operator=proof_bundle_handoff["ready_for_operator"],
                proof_bundle_state_changed=proof_bundle_handoff["state_changed"],
                proof_bundle_changed=proof_bundle_handoff["changed"],
                proof_bundle_content_in_handoff=proof_bundle_handoff["content_in_handoff"],
                proof_bundle_authorizes_execution=proof_bundle_handoff["authorizes_execution"],
                proof_bundle_authorizes_completion_claim=proof_bundle_handoff["authorizes_completion_claim"],
                proof_bundle_approval_granted=proof_bundle_handoff["approval_granted"],
                proof_bundle_boundaries=proof_bundle_handoff["boundaries"],
                proof_bundle_next_safe_commands=proof_bundle_handoff["next_safe_commands"],
                natural_language_routing_enabled=False,
                calls_external_service=False,
                reads_personal_data=False,
                executes_side_effect=False,
            ),
        )

    def integration_implementation_review(args: dict[str, Any]) -> ToolResult:
        connector = _clean_text(args.get("connector") or "calendar", limit=MAX_CONNECTOR_CHARS) or "calendar"
        action = _clean_text(args.get("action"), limit=MAX_ACTION_CHARS)
        target = _clean_text(args.get("target"), limit=MAX_SCOPE_FIELD_CHARS)
        time_range = _clean_text(args.get("time_range"), limit=MAX_SCOPE_FIELD_CHARS)
        data_level = _clean_text(args.get("data_level"), limit=MAX_SCOPE_FIELD_CHARS)
        verification = _clean_text(args.get("verification"), limit=MAX_SCOPE_FIELD_CHARS)
        rollback = _clean_text(args.get("rollback"), limit=MAX_SCOPE_FIELD_CHARS)
        tests = _clean_text(args.get("tests") or args.get("test_plan"), limit=MAX_SCOPE_FIELD_CHARS)
        audit = _clean_text(args.get("audit") or args.get("audit_plan"), limit=MAX_SCOPE_FIELD_CHARS)
        acceptance = _clean_text(args.get("acceptance") or args.get("acceptance_gate") or args.get("evidence") or args.get("proof"), limit=MAX_SCOPE_FIELD_CHARS)
        status = _clean_text(args.get("status") or args.get("status_api") or args.get("dashboard"), limit=MAX_SCOPE_FIELD_CHARS)
        if not action:
            return _action_required_failure("integration_implementation_review")

        normalized, profile = _connector_profile(connector)
        risk, approval_required, reason = _classify_connector_action(action)
        review_args = {
            "connector": normalized,
            "action": action,
            "target": target,
            "time_range": time_range,
            "data_level": data_level,
            "verification": verification,
            "rollback": rollback,
            "tests": tests,
            "audit": audit,
            "acceptance": acceptance,
        }
        proof_bundle = integration_proof_bundle(review_args)
        preflight_contract = integration_preflight_contract(review_args)
        implementation_spec = integration_implementation_spec(review_args)

        proof_metadata = proof_bundle.metadata or {}
        preflight_metadata = preflight_contract.metadata or {}
        spec_metadata = implementation_spec.metadata or {}
        status_evidence = status or f"/api/integration-proof-bundle plus focused personal/status smoke tests for {normalized}"
        review_checks = [
            {
                "name": "proof bundle",
                "state": proof_metadata.get("bundle_state"),
                "passed": _metadata_bool(proof_metadata.get("implementation_review_allowed")),
                "next": f"integration proof bundle: {normalized} -> {action}; target {target or '<target>'}; time {time_range or '<time range>'}; data {data_level or '<data level>'}; verification {verification or '<verification>'}; rollback {rollback or 'no connector call'}; tests {tests or '<tests>'}; audit {audit or '<audit>'}; acceptance {acceptance or '<acceptance>'}",
            },
            {
                "name": "metadata row contract",
                "state": "ROW_SCHEMA_CONTRACT_READY" if _metadata_bool(proof_metadata.get("metadata_row_contract_ready")) else "ROW_SCHEMA_CONTRACT_BLOCKED",
                "passed": _metadata_bool(proof_metadata.get("metadata_row_contract_ready")),
                "next": f"integration proof bundle: {normalized} -> {action}; target {target or '<target>'}; time {time_range or '<time range>'}; data {data_level or '<data level>'}; verification {verification or '<verification>'}; rollback {rollback or 'no connector call'}; tests {tests or '<tests>'}; audit {audit or '<audit>'}; acceptance {acceptance or '<acceptance>'}",
            },
            {
                "name": "preflight contract",
                "state": preflight_metadata.get("preflight_state"),
                "passed": preflight_metadata.get("preflight_state") == "PREFLIGHT_READY_FOR_REVIEW",
                "next": f"integration preflight contract: {normalized} -> {action}; target {target or '<target>'}; time {time_range or '<time range>'}; data {data_level or '<data level>'}; verification {verification or '<verification>'}; rollback {rollback or 'no connector call'}; tests {tests or '<tests>'}; audit {audit or '<audit>'}",
            },
            {
                "name": "implementation spec",
                "state": spec_metadata.get("spec_state"),
                "passed": spec_metadata.get("spec_state") == "SPEC_READY_FOR_REVIEW",
                "next": f"integration implementation spec: {normalized} -> {action}; target {target or '<target>'}; time {time_range or '<time range>'}; data {data_level or '<data level>'}; verification {verification or '<verification>'}; rollback {rollback or 'no connector call'}; tests {tests or '<tests>'}; audit {audit or '<audit>'}",
            },
            {
                "name": "status visibility",
                "state": "STATUS_EVIDENCE_PRESENT" if status else "STATUS_EVIDENCE_DEFAULTED",
                "passed": bool(status),
                "next": f"integration implementation review: {normalized} -> {action}; target {target or '<target>'}; time {time_range or '<time range>'}; data {data_level or '<data level>'}; verification {verification or '<verification>'}; rollback {rollback or 'no connector call'}; tests {tests or '<tests>'}; audit {audit or '<audit>'}; acceptance {acceptance or '<acceptance>'}; status <status API and smoke evidence>",
            },
        ]
        passed_checks = sum(1 for check in review_checks if check["passed"])
        missing_blockers = sorted(
            {
                str(item)
                for source in (
                    proof_metadata.get("missing_fields") or [],
                    preflight_metadata.get("missing_fields") or [],
                    spec_metadata.get("missing_fields") or [],
                    ([] if status else ["status/API smoke evidence"]),
                )
                for item in source
            }
        )
        review_state = "IMPLEMENTATION_REVIEW_READY" if passed_checks == len(review_checks) and not missing_blockers else "IMPLEMENTATION_REVIEW_BLOCKED"
        tool_name = str(spec_metadata.get("tool_name") or proof_metadata.get("tool_name") or f"future_{normalized}_connector")
        scope_hash = _integration_scope_hash(
            normalized,
            action,
            target,
            time_range,
            data_level,
            verification,
            rollback,
            tests,
            audit,
            acceptance,
            status,
        )
        proof_bundle_command = _integration_proof_bundle_command(
            normalized,
            action,
            target,
            time_range,
            data_level,
            verification,
            rollback,
            tests,
            audit,
            acceptance,
        )
        implementation_review_command = _integration_implementation_review_command(
            normalized,
            action,
            target,
            time_range,
            data_level,
            verification,
            rollback,
            tests,
            audit,
            acceptance,
            status,
        )
        review_receipt_id = f"integration-{scope_hash[:12]}"
        receipt_contract_rows = _implementation_review_receipt_contract_rows(
            review_ready=review_state == "IMPLEMENTATION_REVIEW_READY",
            scope_hash=scope_hash,
            review_receipt_id=review_receipt_id,
        )
        receipt_contract_summary = [
            "exact-scope-only",
            "code-review-only" if review_state == "IMPLEMENTATION_REVIEW_READY" else "code-review-blocked",
            "account-access-not-authorized",
            "route-unlock-not-authorized",
            "fresh-route-lock-review-required",
        ]
        proof_queue = _ordered_commands(
            [str(check["next"]) for check in review_checks]
            + [
                "execution health report",
                f"integration metadata preview: {normalized} -> {action}; target {target or '<target>'}; time {time_range or '<time range>'}; data metadata-only; verification {verification or 'fake rows only'}; tests {tests or 'blocked full-content smoke'}; audit {audit or 'metadata preview receipt'}; acceptance {acceptance or 'acceptance gate passed'}",
                f"integration adapter acceptance: {normalized}",
            ]
        )
        next_safe_command = "execution health report" if review_state == "IMPLEMENTATION_REVIEW_READY" else next(check["next"] for check in review_checks if not check["passed"])
        next_proof_command = "execution health report" if review_state == "IMPLEMENTATION_REVIEW_READY" else proof_queue[0]
        implementation_review_handoff = _implementation_review_handoff_payload(
            connector=normalized,
            action=action,
            tool_name=tool_name,
            review_state=review_state,
            review_checks=review_checks,
            passed_review_checks=passed_checks,
            review_receipt_id=review_receipt_id,
            integration_scope_hash=scope_hash,
            proof_bundle_command=proof_bundle_command,
            implementation_review_command=implementation_review_command,
            receipt_contract_rows=receipt_contract_rows,
            receipt_contract_summary=receipt_contract_summary,
            proof_queue=proof_queue,
            next_proof_command=next_proof_command,
            next_safe_command=next_safe_command,
            missing_fields=missing_blockers,
            proof_bundle_state=proof_metadata.get("bundle_state"),
            preflight_state=preflight_metadata.get("preflight_state"),
            spec_state=spec_metadata.get("spec_state"),
            status_evidence_present=bool(status),
            metadata_row_contract_ready=_metadata_bool(proof_metadata.get("metadata_row_contract_ready")),
            suggested_risk=risk,
            approval_required=approval_required,
        )

        lines = [
            "Jarvis personal integration implementation review:",
            "This is read-only. It gates whether a future connector adapter is ready for code review by checking proof bundle, metadata row contract, preflight row-contract proof, implementation spec, status visibility, smoke evidence, audit, verification, rollback, and stop conditions. It does not connect accounts, read personal data, call external services, execute tools, change state, control the computer, approve requests, or queue approvals.",
            "",
            f"Connector: {connector}",
            f"Normalized connector: {normalized}",
            f"Requested action: {action}",
            f"Review state: {review_state}",
            f"Suggested risk: {risk}",
            f"Approval required for real use: {'yes' if approval_required else 'no, while implementation stays metadata-only/read-only'}",
            f"Reason: {reason}",
            "",
            "Exact implementation review scope:",
            f"- future tool: {tool_name}",
            f"- target/source: {target or '<missing>'}",
            f"- time range or selected item: {time_range or '<missing>'}",
            f"- data level: {data_level or '<missing>'}",
            f"- verification target: {verification or '<missing>'}",
            f"- rollback/cancel path: {rollback or '<missing>'}",
            f"- smoke tests: {tests or '<missing>'}",
            f"- audit trail: {audit or '<missing>'}",
            f"- acceptance gate: {acceptance or '<missing>'}",
            f"- status/API evidence: {status_evidence}",
            f"- missing blockers: {', '.join(missing_blockers) if missing_blockers else 'none'}",
            f"- metadata row schema: {', '.join(METADATA_ROW_SCHEMA)}",
            f"- metadata row limit: {METADATA_ROW_LIMIT}",
            f"- blocked payload fields: {', '.join(BLOCKED_METADATA_PAYLOAD_FIELDS)}",
            "",
            "Review checks:",
        ]
        for check in review_checks:
            lines.extend(
                [
                    f"- {check['name']}: {'pass' if check['passed'] else 'blocked'}",
                    f"  state: {check['state']}",
                    f"  next: `{check['next']}`",
                ]
            )

        lines.extend(
            [
                "",
                "Implementation invariants before code review:",
                f"- review receipt id: `{review_receipt_id}`",
                f"- scope hash: `{scope_hash}`",
                f"- proof bundle command: `{proof_bundle_command}`",
                f"- implementation review command: `{implementation_review_command}`",
                f"- receipt contract rows: {len(receipt_contract_rows)}",
                "- receipt contract summary: " + ", ".join(receipt_contract_summary),
                f"- code-review unlock: {'ready for this exact scope' if review_state == 'IMPLEMENTATION_REVIEW_READY' else 'blocked until every review check passes'}",
                "- natural-language auto-routing remains disabled.",
                "- adapter default state remains disabled or fake-fixture-only until focused smoke tests pass.",
                "- output is bounded metadata, not raw private connector payloads.",
                "- personal-data reads and side effects still stop at approval readiness, a one-shot approval packet, and approval chain proof.",
                "- status/API visibility must expose adapter state, routing state, approval requirement, and proof state.",
                "",
                "Next safe command:",
                f"- `{next_safe_command}`",
                "",
                "Implementation proof queue:",
                f"- next required command: `{next_proof_command}`",
                f"- proof queue count: {len(proof_queue)}",
                "- proof queue: " + ", ".join(f"`{command}`" for command in proof_queue[:8]),
                "",
                "Known connector boundaries:",
                "- READ_ONLY candidates: " + ", ".join(profile["read_only"]) + ".",
                "- PERSONAL_DATA candidates: " + ", ".join(profile["personal_data"]) + ".",
                "- EXTERNAL_SIDE_EFFECT or HIGH_RISK candidates: " + ", ".join(profile["side_effects"]) + ".",
                "",
                "Stop conditions:",
                OPERATOR_LIMIT_STOP,
                "- Stop if status/API smoke evidence is missing or points at a different connector/action/scope.",
                "- Stop if implementation would store tokens, cookies, full private content, contact cards, attachments, authenticated browser state, or broad account exports by default.",
                "- Stop if natural-language routing can reach a connector before explicit command smoke tests pass.",
                "- Stop if personal-data or side-effect execution can bypass approval readiness, approval packet, approval chain proof, audit, verification, and rollback proof.",
            ]
        )
        return ToolResult(
            "integration_implementation_review",
            True,
            "\n".join(lines),
            _safe_metadata(
                connector=normalized,
                requested_connector=connector,
                action=action,
                target=target,
                time_range=time_range,
                data_level=data_level,
                verification=verification,
                rollback=rollback,
                tests=tests,
                audit=audit,
                acceptance=acceptance,
                status=status,
                review_state=review_state,
                review_checks=review_checks,
                review_check_count=len(review_checks),
                passed_review_checks=passed_checks,
                proof_queue=proof_queue,
                proof_queue_count=len(proof_queue),
                next_required_command=next_proof_command,
                next_proof_command=next_proof_command,
                integration_proof_queue=proof_queue,
                integration_proof_queue_count=len(proof_queue),
                integration_next_required_command=next_proof_command,
                integration_next_proof_command=next_proof_command,
                review_receipt_id=review_receipt_id,
                integration_scope_hash=scope_hash,
                proof_bundle_command=proof_bundle_command,
                implementation_review_command=implementation_review_command,
                implementation_review_receipt_contract_rows=receipt_contract_rows,
                implementation_review_receipt_contract_row_count=len(receipt_contract_rows),
                implementation_review_receipt_contract_ready=review_state == "IMPLEMENTATION_REVIEW_READY",
                implementation_review_receipt_contract_summary=receipt_contract_summary,
                implementation_review_receipt_authorizes_account_access=False,
                implementation_review_receipt_authorizes_route_unlock=False,
                implementation_review_receipt_reusable_for_other_scope=False,
                route_blocked_until_implementation_review=review_state != "IMPLEMENTATION_REVIEW_READY",
                can_open_code_review_after_receipt=review_state == "IMPLEMENTATION_REVIEW_READY",
                next_safe_command=next_safe_command,
                missing_fields=missing_blockers,
                tool_name=tool_name,
                suggested_risk=risk,
                approval_required=approval_required,
                requires_approval=approval_required,
                proof_bundle_state=proof_metadata.get("bundle_state"),
                preflight_state=preflight_metadata.get("preflight_state"),
                spec_state=spec_metadata.get("spec_state"),
                status_evidence_present=bool(status),
                sample_row_schema=list(METADATA_ROW_SCHEMA),
                sample_row_limit=METADATA_ROW_LIMIT,
                blocked_payload_fields=list(BLOCKED_METADATA_PAYLOAD_FIELDS),
                metadata_row_contract_ready=_metadata_bool(proof_metadata.get("metadata_row_contract_ready")),
                implementation_review_allowed=review_state == "IMPLEMENTATION_REVIEW_READY",
                implementation_review_handoff=implementation_review_handoff,
                implementation_review_handoff_ready=implementation_review_handoff["handoff_ready"],
                implementation_review_ready_for_operator=implementation_review_handoff["ready_for_operator"],
                implementation_review_state_changed=implementation_review_handoff["state_changed"],
                implementation_review_changed=implementation_review_handoff["changed"],
                implementation_review_content_in_handoff=implementation_review_handoff["content_in_handoff"],
                implementation_review_authorizes_execution=implementation_review_handoff["authorizes_execution"],
                implementation_review_authorizes_completion_claim=implementation_review_handoff["authorizes_completion_claim"],
                implementation_review_approval_granted=implementation_review_handoff["approval_granted"],
                implementation_review_boundaries=implementation_review_handoff["boundaries"],
                implementation_review_next_safe_commands=implementation_review_handoff["next_safe_commands"],
                natural_language_routing_enabled=False,
                adapter_default_state="disabled",
                calls_external_service=False,
                reads_personal_data=False,
                executes_side_effect=False,
            ),
        )

    def integration_route_lock(args: dict[str, Any]) -> ToolResult:
        connector = _clean_text(args.get("connector") or "calendar", limit=MAX_CONNECTOR_CHARS) or "calendar"
        action = _clean_text(args.get("action"), limit=MAX_ACTION_CHARS)
        target = _clean_text(args.get("target"), limit=MAX_SCOPE_FIELD_CHARS)
        time_range = _clean_text(args.get("time_range"), limit=MAX_SCOPE_FIELD_CHARS)
        data_level = _clean_text(args.get("data_level"), limit=MAX_SCOPE_FIELD_CHARS)
        verification = _clean_text(args.get("verification"), limit=MAX_SCOPE_FIELD_CHARS)
        rollback = _clean_text(args.get("rollback"), limit=MAX_SCOPE_FIELD_CHARS)
        tests = _clean_text(args.get("tests") or args.get("test_plan"), limit=MAX_SCOPE_FIELD_CHARS)
        audit = _clean_text(args.get("audit") or args.get("audit_plan"), limit=MAX_SCOPE_FIELD_CHARS)
        acceptance = _clean_text(args.get("acceptance") or args.get("acceptance_gate") or args.get("evidence") or args.get("proof"), limit=MAX_SCOPE_FIELD_CHARS)
        status = _clean_text(args.get("status") or args.get("status_api") or args.get("dashboard"), limit=MAX_SCOPE_FIELD_CHARS)
        expected_scope_hash = _clean_text(
            args.get("expected_scope_hash") or args.get("scope_hash") or args.get("integration_scope_hash"),
            limit=128,
        )
        if not action:
            return _action_required_failure("integration_route_lock")

        normalized, profile = _connector_profile(connector)
        risk, approval_required, reason = _classify_connector_action(action)
        review_args = {
            "connector": normalized,
            "action": action,
            "target": target,
            "time_range": time_range,
            "data_level": data_level,
            "verification": verification,
            "rollback": rollback,
            "tests": tests,
            "audit": audit,
            "acceptance": acceptance,
            "status": status,
        }
        review = integration_implementation_review(review_args)
        review_metadata = review.metadata or {}
        review_ready = _metadata_bool(review_metadata.get("implementation_review_allowed"))
        status_ready = _metadata_bool(review_metadata.get("status_evidence_present"))
        metadata_contract_ready = _metadata_bool(review_metadata.get("metadata_row_contract_ready"))
        exact_scope_hash = str(review_metadata.get("integration_scope_hash") or "")
        review_receipt_id = str(review_metadata.get("review_receipt_id") or "")
        implementation_review_receipt_contract_rows = list(
            review_metadata.get("implementation_review_receipt_contract_rows") or []
        )
        expected_scope_hash_required = bool(expected_scope_hash)
        expected_scope_hash_valid = (not expected_scope_hash_required) or _valid_integration_scope_hash(expected_scope_hash)
        scope_hash_matches_expected = bool(
            expected_scope_hash_valid
            and expected_scope_hash
            and exact_scope_hash
            and expected_scope_hash.lower() == exact_scope_hash.lower()
        )
        route_unlock_candidate = (
            review_ready
            and status_ready
            and metadata_contract_ready
            and expected_scope_hash_valid
            and (not expected_scope_hash_required or scope_hash_matches_expected)
        )
        natural_language_routing_enabled = False
        route_lock_state = "ROUTE_LOCK_HELD"
        if route_unlock_candidate:
            route_lock_state = "ROUTE_LOCK_READY_FOR_EXPLICIT_REVIEW"

        missing: list[str] = []
        if not review_ready:
            missing.append("implementation review ready receipt")
        if not status_ready:
            missing.append("status/API smoke evidence")
        if not metadata_contract_ready:
            missing.append("metadata row contract proof")
        if not exact_scope_hash:
            missing.append("scope hash")
        if expected_scope_hash_required and not expected_scope_hash_valid:
            missing.append("valid expected scope hash")
        if expected_scope_hash_required and not scope_hash_matches_expected:
            missing.append("expected scope hash match")
        if not review_receipt_id:
            missing.append("implementation review receipt id")

        route_review_contract_rows = _route_review_contract_rows(
            route_unlock_candidate=route_unlock_candidate,
            scope_hash=exact_scope_hash,
            review_receipt_id=review_receipt_id,
            route_lock_state=route_lock_state,
        )
        route_review_contract_ready = _route_review_contract_ready(
            route_review_contract_rows,
            route_unlock_candidate=route_unlock_candidate,
            scope_hash=exact_scope_hash,
            review_receipt_id=review_receipt_id,
            route_lock_state=route_lock_state,
        )
        route_review_contract_summary = [
            "explicit-review-only",
            "account-access-not-authorized",
            "route-unlock-not-authorized",
            "natural-language-routing-not-authorized",
            "approval-chain-still-required",
            "fresh-scope-review-required",
        ]
        route_lock_token_sha256 = _route_lock_token_sha256(
            scope_hash=exact_scope_hash,
            review_receipt_id=review_receipt_id,
            route_lock_state=route_lock_state,
            expected_scope_hash=expected_scope_hash,
            implementation_review_receipt_contract_rows=implementation_review_receipt_contract_rows,
            route_review_contract_rows=route_review_contract_rows,
        )
        route_lock_token_boundary_rows = _route_lock_token_boundary_rows(
            token_sha256=route_lock_token_sha256,
            scope_hash=exact_scope_hash,
            review_receipt_id=review_receipt_id,
            route_lock_state=route_lock_state,
        )
        route_lock_token_boundary_ready = _route_lock_token_boundary_ready(
            route_lock_token_sha256,
            route_lock_token_boundary_rows,
        )
        route_lock_command = (
            f"integration route lock: {normalized} -> {action}; target {target or '<target>'}; "
            f"time {time_range or '<time range>'}; data {data_level or '<data level>'}; "
            f"verification {verification or '<verification>'}; rollback {rollback or 'no connector call'}; "
            f"tests {tests or '<tests>'}; audit {audit or '<audit>'}; acceptance {acceptance or '<acceptance>'}; "
            f"status {status or '<status API and smoke evidence>'}; expected_scope_hash {expected_scope_hash or '<scope hash>'}"
        )
        implementation_review_command = str(review_metadata.get("implementation_review_command") or _integration_implementation_review_command(
            normalized,
            action,
            target,
            time_range,
            data_level,
            verification,
            rollback,
            tests,
            audit,
            acceptance,
            status,
        ))
        route_proof_queue = _ordered_commands(
            [
                implementation_review_command,
                str(review_metadata.get("proof_bundle_command") or ""),
                "integration adapter manifest: " + normalized,
                "integration adapter acceptance: " + normalized,
                "execution health report",
                route_lock_command,
            ]
        )
        next_route_proof_command = "execution health report" if route_unlock_candidate else route_proof_queue[0]
        route_lock_handoff = _route_lock_handoff_payload(
            connector=normalized,
            action=action,
            route_lock_state=route_lock_state,
            route_unlock_candidate=route_unlock_candidate,
            explicit_route_review_required=True,
            natural_language_routing_enabled=natural_language_routing_enabled,
            implementation_review_ready=review_ready,
            status_evidence_present=status_ready,
            metadata_row_contract_ready=metadata_contract_ready,
            review_receipt_id=review_receipt_id,
            integration_scope_hash=exact_scope_hash,
            expected_scope_hash=expected_scope_hash,
            expected_scope_hash_required=expected_scope_hash_required,
            expected_scope_hash_valid=expected_scope_hash_valid,
            scope_hash_matches_expected=scope_hash_matches_expected,
            missing_fields=missing,
            route_review_contract_rows=route_review_contract_rows,
            route_review_contract_ready=route_review_contract_ready,
            route_review_contract_summary=route_review_contract_summary,
            implementation_review_receipt_contract_rows=implementation_review_receipt_contract_rows,
            route_lock_token_sha256=route_lock_token_sha256,
            route_lock_token_boundary_rows=route_lock_token_boundary_rows,
            route_lock_token_boundary_ready=route_lock_token_boundary_ready,
            route_proof_queue=route_proof_queue,
            next_route_proof_command=next_route_proof_command,
            implementation_review_command=implementation_review_command,
            route_lock_command=route_lock_command,
        )

        lines = [
            "Jarvis personal integration route lock:",
            "This is read-only. It decides whether a future connector route may leave disabled/manual-only mode without connecting accounts, reading personal data, calling external services, executing tools, changing state, controlling the computer, approving requests, or queuing approvals.",
            "",
            f"Connector: {connector}",
            f"Normalized connector: {normalized}",
            f"Requested action: {action}",
            f"Route lock state: {route_lock_state}",
            f"Suggested risk: {risk}",
            f"Approval required for real use: {'yes' if approval_required else 'no, while implementation stays metadata-only/read-only'}",
            f"Reason: {reason}",
            "",
            "Route posture:",
            f"- implementation review ready: {'yes' if review_ready else 'no'}",
            f"- status/API smoke evidence: {'present' if status_ready else 'missing'}",
            f"- metadata row contract: {'ready' if metadata_contract_ready else 'blocked'}",
            f"- review receipt id: `{review_receipt_id or '<missing>'}`",
            f"- scope hash: `{exact_scope_hash or '<missing>'}`",
            f"- expected scope hash: `{expected_scope_hash or '<not supplied>'}`",
            f"- expected scope hash valid: {'yes' if expected_scope_hash_valid else 'not required' if not expected_scope_hash_required else 'no'}",
            f"- expected scope hash match: {'yes' if scope_hash_matches_expected else 'not required' if not expected_scope_hash_required else 'no'}",
            f"- natural-language auto-routing: {'enabled' if natural_language_routing_enabled else 'disabled'}",
            f"- route unlock candidate: {'yes, explicit review still required' if route_unlock_candidate else 'no'}",
            f"- missing blockers: {', '.join(missing) if missing else 'none'}",
            "",
            "Route lock rules:",
            "- Implementation review readiness is not the same thing as natural-language auto-routing.",
            "- Explicit command routes may be reviewed first; natural-language connector dispatch remains disabled by default.",
            "- Personal-data reads and side effects still require approval readiness, a one-shot approval packet, approval chain proof, audit receipt, verification receipt, and rollback/stop evidence.",
            "- Unlock applies only to the exact connector/action/target/time/data/scope hash shown here.",
            "- If `expected_scope_hash` is supplied, it must match the computed integration scope hash before the route can be considered ready for explicit review.",
            "- Any changed account, target, thread, event, date range, data level, action, status evidence, or test receipt needs a fresh route lock packet.",
            "",
            "Explicit route review contract:",
            f"- contract ready: {'yes' if route_review_contract_ready else 'no'}",
            f"- contract rows: {len(route_review_contract_rows)}",
            "- contract summary: " + ", ".join(route_review_contract_summary),
            "- authorizes account access: no",
            "- authorizes route unlock: no",
            "- authorizes natural-language routing: no",
            "- authorizes personal-data read: no",
            "- authorizes side effect: no",
            "- reusable for other scope: no",
            f"- upstream implementation receipt rows bound into token: {len(implementation_review_receipt_contract_rows)}",
            "- next real connector step still needs approval readiness, one-shot approval packet, approval chain proof, audit receipt, verification receipt, and rollback/stop evidence.",
            "",
            "Route lock token boundary:",
            f"- route lock token sha256: {route_lock_token_sha256}",
            f"- boundary rows: {len(route_lock_token_boundary_rows)}",
            f"- boundary ready: {'yes' if route_lock_token_boundary_ready else 'no'}",
            "- route-lock tokens are proof-only; they do not authorize account access, route unlock, natural-language routing, personal-data reads, side effects, approvals, reuse for other scopes, or reuse for the next route review.",
            *[
                f"- {row['item']}: {row['status']}; authorizes account access no; route unlock no; natural-language routing no; model/tool/external no; approval no; reusable no"
                for row in route_lock_token_boundary_rows
            ],
            "",
            "Route proof queue:",
            f"- next route required: `{next_route_proof_command}`",
            f"- route proof queue count: {len(route_proof_queue)}",
            "- route proof queue: " + ", ".join(f"`{command}`" for command in route_proof_queue),
            "",
            "Known connector boundaries:",
            "- READ_ONLY candidates: " + ", ".join(profile["read_only"]) + ".",
            "- PERSONAL_DATA candidates: " + ", ".join(profile["personal_data"]) + ".",
            "- EXTERNAL_SIDE_EFFECT or HIGH_RISK candidates: " + ", ".join(profile["side_effects"]) + ".",
            "",
            "Stop conditions:",
            OPERATOR_LIMIT_STOP,
            "- Stop if implementation review, status/API smoke evidence, metadata row contract, scope hash, or review receipt is missing.",
            "- Stop if natural-language routing could reach private data or side effects before the explicit route lock review.",
            "- Stop if approval readiness, approval packet, approval chain proof, audit, verification, or rollback proof can be bypassed.",
            "- Stop if implementation stores tokens, cookies, full private content, contact cards, attachments, authenticated browser state, or broad account exports by default.",
        ]
        return ToolResult(
            "integration_route_lock",
            True,
            "\n".join(lines),
            _safe_metadata(
                connector=normalized,
                requested_connector=connector,
                action=action,
                target=target,
                time_range=time_range,
                data_level=data_level,
                verification=verification,
                rollback=rollback,
                tests=tests,
                audit=audit,
                acceptance=acceptance,
                status=status,
                expected_scope_hash=expected_scope_hash,
                expected_scope_hash_required=expected_scope_hash_required,
                expected_scope_hash_valid=expected_scope_hash_valid,
                scope_hash_matches_expected=scope_hash_matches_expected,
                route_lock_state=route_lock_state,
                route_unlock_candidate=route_unlock_candidate,
                natural_language_routing_enabled=natural_language_routing_enabled,
                explicit_route_review_required=True,
                implementation_review_ready=review_ready,
                status_evidence_present=status_ready,
                metadata_row_contract_ready=metadata_contract_ready,
                review_receipt_id=review_receipt_id,
                integration_scope_hash=exact_scope_hash,
                missing_fields=missing,
                route_proof_queue=route_proof_queue,
                route_proof_queue_count=len(route_proof_queue),
                next_required_command=next_route_proof_command,
                next_route_required_command=next_route_proof_command,
                next_route_proof_command=next_route_proof_command,
                integration_proof_queue=route_proof_queue,
                integration_proof_queue_count=len(route_proof_queue),
                integration_next_required_command=next_route_proof_command,
                integration_next_proof_command=next_route_proof_command,
                route_review_contract_rows=route_review_contract_rows,
                route_review_contract_row_count=len(route_review_contract_rows),
                route_review_contract_ready=route_review_contract_ready,
                route_review_contract_summary=route_review_contract_summary,
                implementation_review_receipt_contract_rows=implementation_review_receipt_contract_rows,
                implementation_review_receipt_contract_row_count=len(implementation_review_receipt_contract_rows),
                route_lock_token_binds_implementation_review_receipt=True,
                route_review_authorizes_account_access=False,
                route_review_authorizes_route_unlock=False,
                route_review_authorizes_natural_language_routing=False,
                route_review_authorizes_personal_data_read=False,
                route_review_authorizes_side_effect=False,
                route_review_reusable_for_other_scope=False,
                route_lock_token_sha256=route_lock_token_sha256,
                route_lock_token_present=len(route_lock_token_sha256) == 64,
                route_lock_token_boundary_rows=route_lock_token_boundary_rows,
                route_lock_token_boundary_row_count=len(route_lock_token_boundary_rows),
                route_lock_token_boundary_ready=route_lock_token_boundary_ready,
                route_lock_token_authorizes_account_access=False,
                route_lock_token_authorizes_route_unlock=False,
                route_lock_token_authorizes_natural_language_routing=False,
                route_lock_token_authorizes_personal_data_read=False,
                route_lock_token_authorizes_side_effect=False,
                route_lock_token_authorizes_approval=False,
                route_lock_token_authorizes_model_call=False,
                route_lock_token_authorizes_tool_execution=False,
                route_lock_token_authorizes_external_service=False,
                route_lock_token_reusable_for_other_scope=False,
                route_lock_token_reusable_for_next_route_review=False,
                route_lock_handoff=route_lock_handoff,
                route_lock_handoff_ready=route_lock_handoff["handoff_ready"],
                route_lock_ready_for_operator=route_lock_handoff["ready_for_operator"],
                route_lock_state_changed=route_lock_handoff["state_changed"],
                route_lock_changed=route_lock_handoff["changed"],
                route_lock_content_in_handoff=route_lock_handoff["content_in_handoff"],
                route_lock_authorizes_execution=route_lock_handoff["authorizes_execution"],
                route_lock_authorizes_completion_claim=route_lock_handoff["authorizes_completion_claim"],
                route_lock_approval_granted=route_lock_handoff["approval_granted"],
                route_lock_boundaries=route_lock_handoff["boundaries"],
                route_lock_next_safe_commands=route_lock_handoff["next_safe_commands"],
                next_integration_route_requires_fresh_route_lock_review=True,
                implementation_review_command=implementation_review_command,
                route_lock_command=route_lock_command,
                suggested_risk=risk,
                approval_required=approval_required,
                requires_approval=approval_required,
                adapter_default_state="disabled",
                calls_external_service=False,
                reads_personal_data=False,
                executes_side_effect=False,
            ),
        )

    def integration_adapter_manifest(args: dict[str, Any]) -> ToolResult:
        requested = _clean_text(args.get("connector"), limit=MAX_CONNECTOR_CHARS).lower()
        connector_names = [requested] if requested else list(CONNECTOR_PROFILES.keys())
        rows: list[dict[str, Any]] = []
        for connector_name in connector_names:
            normalized, profile = _connector_profile(connector_name)
            safe_action = profile["read_only"][0]
            side_effect_action = profile["side_effects"][0]
            adapter_name = f"{normalized}_metadata_adapter"
            future_tool = f"future_{normalized}_metadata_preview"
            fixture_name = f"{normalized}_metadata_fixture"
            rows.append(
                {
                    "connector": normalized,
                    "adapter_name": adapter_name,
                    "future_tool": future_tool,
                    "fixture_name": fixture_name,
                    "safe_action": safe_action,
                    "blocked_action": side_effect_action,
                    "adapter_state": "disabled",
                    "risk": "READ_ONLY_METADATA_ONLY",
                    "enablement_gate": "integration_enablement_gate",
                    "sample_row_schema": list(METADATA_ROW_SCHEMA),
                    "sample_row_limit": METADATA_ROW_LIMIT,
                    "sample_rows_bounded": True,
                    "blocked_payload_fields": list(BLOCKED_METADATA_PAYLOAD_FIELDS),
                }
            )
        all_manifest_row_contracts_present = all(
            row["sample_row_schema"] == METADATA_ROW_SCHEMA
            and row["sample_row_limit"] == METADATA_ROW_LIMIT
            and row["sample_rows_bounded"] is True
            for row in rows
        )
        adapter_manifest_handoff = _adapter_manifest_handoff_payload(
            requested_connector=requested,
            rows=rows,
            sample_row_schema=list(METADATA_ROW_SCHEMA),
            sample_row_limit=METADATA_ROW_LIMIT,
            blocked_payload_fields=list(BLOCKED_METADATA_PAYLOAD_FIELDS),
            all_manifest_row_contracts_present=all_manifest_row_contracts_present,
        )

        lines = [
            "Jarvis personal integration adapter manifest:",
            "This is read-only. It defines disabled adapter stubs, fake fixtures, blocked-path tests, audit fields, and enablement gates without connecting accounts, reading personal data, calling external services, changing state, controlling the computer, approving requests, or queuing approvals.",
            "",
            "Adapter rule:",
            "- The first real connector implementation must start as a disabled metadata-only adapter with deterministic fake rows.",
            "- Natural-language routing stays off until manifest, metadata preview, blocked full-content test, blocked side-effect test, audit receipt, and acceptance gate all pass.",
            "- Personal-data reads and side effects remain behind approval readiness, exact-scope approval packets, and approval chain proof even after a connector exists.",
            "",
            "Adapter stubs:",
        ]
        for row in rows:
            lines.extend(
                [
                    f"- {row['connector']}: {row['adapter_state']}",
                    f"  adapter: `{row['adapter_name']}`",
                    f"  future tool: `{row['future_tool']}`",
                    f"  fixture: `{row['fixture_name']}`",
                    f"  first safe action: {row['safe_action']}",
                    f"  blocked side-effect path: {row['blocked_action']}",
                    f"  row schema: {', '.join(row['sample_row_schema'])}",
                    f"  row limit: {row['sample_row_limit']}",
                    f"  bounded rows: {'yes' if row['sample_rows_bounded'] else 'no'}",
                    f"  blocked payload fields: {', '.join(row['blocked_payload_fields'])}",
                    "  audit fields: connector, action, target, time_range, data_level, adapter_state, sample_row_count, approval_id, verification, result",
                    "  tests: metadata-only happy path, full-content blocked path, side-effect blocked path, missing-scope blocked path, audit metadata bounded",
                ]
            )

        lines.extend(
            [
                "",
                "Implementation order:",
                "1. Add an adapter interface that returns bounded metadata rows from a disabled fake fixture.",
                "2. Register only a read-only preview tool first.",
                "3. Add planner routing only for explicit `integration metadata preview` style commands.",
                "4. Add status/API visibility for the adapter state.",
                "5. Run focused personal/status smoke tests and the full smoke suite.",
                "6. Keep authenticated reads, full content, sends, deletes, edits, and browser control blocked until a later approval-gated adapter passes its own manifest.",
                "",
                "Hard stops:",
                OPERATOR_LIMIT_STOP,
                "- Do not request account tokens, cookies, mailbox bodies, message transcripts, calendar notes, contact cards, attachments, or authenticated page content from this manifest.",
                "- Do not allow natural language to auto-dispatch a connector action just because a manifest exists.",
                "- Do not mark a connector enabled without `integration enablement gate` evidence and an `execution health report` showing no unresolved failed runs.",
            ]
        )

        return ToolResult(
            "integration_adapter_manifest",
            True,
            "\n".join(lines),
            _safe_metadata(
                connectors_reviewed=len(rows),
                connector=requested or "all",
                manifests=rows,
                adapter_states=[row["adapter_state"] for row in rows],
                future_tools=[row["future_tool"] for row in rows],
                fixture_names=[row["fixture_name"] for row in rows],
                sample_row_schema=list(METADATA_ROW_SCHEMA),
                sample_row_limit=METADATA_ROW_LIMIT,
                blocked_payload_fields=list(BLOCKED_METADATA_PAYLOAD_FIELDS),
                all_manifest_row_contracts_present=all_manifest_row_contracts_present,
                adapter_manifest_handoff=adapter_manifest_handoff,
                adapter_manifest_handoff_ready=adapter_manifest_handoff["handoff_ready"],
                adapter_manifest_ready_for_operator=adapter_manifest_handoff["ready_for_operator"],
                adapter_manifest_state_changed=adapter_manifest_handoff["state_changed"],
                adapter_manifest_changed=adapter_manifest_handoff["changed"],
                adapter_manifest_content_in_handoff=adapter_manifest_handoff["content_in_handoff"],
                adapter_manifest_authorizes_execution=adapter_manifest_handoff["authorizes_execution"],
                adapter_manifest_authorizes_completion_claim=adapter_manifest_handoff["authorizes_completion_claim"],
                adapter_manifest_approval_granted=adapter_manifest_handoff["approval_granted"],
                adapter_manifest_boundaries=adapter_manifest_handoff["boundaries"],
                adapter_manifest_next_safe_commands=adapter_manifest_handoff["next_safe_commands"],
                natural_language_routing_enabled=False,
                requires_metadata_preview=True,
                requires_blocked_full_content_test=True,
                requires_blocked_side_effect_test=True,
                requires_acceptance_gate=True,
                requires_execution_health_report=True,
            ),
        )

    def integration_adapter_probe(args: dict[str, Any]) -> ToolResult:
        connector = _clean_text(args.get("connector") or "calendar", limit=MAX_CONNECTOR_CHARS) or "calendar"
        action = _clean_text(args.get("action"), limit=MAX_ACTION_CHARS)
        target = _clean_text(args.get("target"), limit=MAX_SCOPE_FIELD_CHARS)
        time_range = _clean_text(args.get("time_range"), limit=MAX_SCOPE_FIELD_CHARS)
        data_level = _clean_text(args.get("data_level"), limit=MAX_SCOPE_FIELD_CHARS)
        verification = _clean_text(args.get("verification"), limit=MAX_SCOPE_FIELD_CHARS)
        if not action:
            return _action_required_failure("integration_adapter_probe")

        normalized, profile = _connector_profile(connector)
        risk, approval_required, reason = _classify_connector_action(action)
        data_low = data_level.lower()
        metadata_only = "metadata" in data_low and not any(word in data_low for word in ("full", "body", "content", "attachment", "transcript", "notes"))
        side_effect_requested = risk == "EXTERNAL_SIDE_EFFECT"
        missing: list[str] = []
        if not target:
            missing.append("target/source")
        if normalized in {"calendar", "email", "messages", "browser", "reminders"} and not time_range:
            missing.append("time range or selected item")
        if not data_level:
            missing.append("data level")
        elif not metadata_only:
            missing.append("metadata-only data level")
        if side_effect_requested:
            missing.append("non-side-effect action")
        if approval_required and not verification:
            missing.append("verification target")

        probe_state = "ADAPTER_PROBE_BLOCKED" if missing else "ADAPTER_PROBE_READY"
        adapter_state = "disabled"
        adapter_name = f"{normalized}_metadata_adapter"
        fixture_name = f"{normalized}_metadata_fixture"
        sample_rows = _fake_metadata_rows(normalized, target=target, time_range=time_range) if not missing else []
        sample_row_count = len(sample_rows)
        row_contract = _metadata_row_contract(sample_rows)
        adapter_probe_handoff = _adapter_probe_handoff_payload(
            connector=normalized,
            requested_connector=connector,
            action=action,
            adapter_name=adapter_name,
            fixture_name=fixture_name,
            adapter_state=adapter_state,
            probe_state=probe_state,
            missing_fields=missing,
            target=target,
            time_range=time_range,
            data_level=data_level,
            verification=verification,
            metadata_only=metadata_only,
            sample_rows=sample_rows,
            row_contract=row_contract,
            suggested_risk=risk,
            approval_required=approval_required,
            full_content_blocked=not metadata_only,
            side_effect_blocked=side_effect_requested,
        )

        lines = [
            "Jarvis personal integration adapter probe:",
            "This is read-only. It exercises the disabled connector adapter boundary with fake bounded metadata only; it does not connect accounts, read personal data, call external services, execute real adapter code, change state, control the computer, approve requests, or queue approvals.",
            "",
            f"Connector: {connector}",
            f"Normalized connector: {normalized}",
            f"Requested action: {action}",
            f"Probe state: {probe_state}",
            f"Adapter: {adapter_name}",
            f"Fixture: {fixture_name}",
            f"Adapter state: {adapter_state}",
            f"Suggested risk before metadata clamp: {risk}",
            f"Metadata clamp: {'active' if metadata_only else 'blocked until data level is metadata-only'}",
            f"Reason: {reason}",
            "",
            "Exact probe scope:",
            f"- target/source: {target or '<missing>'}",
            f"- time range or selected item: {time_range or '<missing>'}",
            f"- data level: {data_level or '<missing>'}",
            f"- verification target: {verification or '<missing>'}",
            f"- missing blockers: {', '.join(missing) if missing else 'none'}",
            "",
            "Probe result:",
        ]
        if sample_rows:
            lines.extend(
                [
                    f"- fake rows returned: {sample_row_count}",
                    "- row shape: id, timestamp, label, source only",
                    f"- row schema: {', '.join(row_contract['sample_row_schema'])}",
                    f"- row limit: {row_contract['sample_row_limit']}",
                    f"- bounded rows: {'yes' if row_contract['sample_rows_bounded'] else 'no'}",
                    "- adapter call: skipped; disabled fixture used",
                ]
            )
        else:
            lines.extend(
                [
                    "- fake rows returned: 0",
                    "- adapter call: blocked before fixture use",
                    f"- row schema still enforced: {', '.join(row_contract['sample_row_schema'])}",
                    f"- blocked payload fields: {', '.join(row_contract['blocked_payload_fields'])}",
                    "- blocked path is intentional proof that full/private/side-effect scopes do not slip through this adapter lane",
                ]
            )

        lines.extend(
            [
                "",
                "Blocked payloads:",
                "- full email bodies, message text, calendar notes, contact cards, attachments, tokens, cookies, authenticated page content, and broad account ingestion.",
                "- send/create/delete/update/submit/mark-read or other side effects.",
                "",
                "Known connector boundaries:",
                "- READ_ONLY candidates: " + ", ".join(profile["read_only"]) + ".",
                "- PERSONAL_DATA candidates: " + ", ".join(profile["personal_data"]) + ".",
                "- EXTERNAL_SIDE_EFFECT or HIGH_RISK candidates: " + ", ".join(profile["side_effects"]) + ".",
                "",
                "Next required commands:",
                f"- `integration adapter manifest: {normalized}`",
                f"- `integration metadata preview: {normalized} -> {action}; target {target or '<target>'}; time {time_range or '<time range>'}; data metadata-only; verification {verification or 'fake rows only'}; tests blocked full-content smoke; audit adapter probe receipt; acceptance gate passed`",
                "- `execution health report` after focused smoke tests.",
                "",
                "Hard stops:",
                OPERATOR_LIMIT_STOP,
                "- Do not convert this probe into a real connector without an enablement gate, focused smoke tests, blocked-path tests, audit evidence, and explicit approval boundaries.",
                "- Do not enable natural-language auto-routing for connector actions from this probe alone.",
            ]
        )

        return ToolResult(
            "integration_adapter_probe",
            True,
            "\n".join(lines),
            _safe_metadata(
                connector=normalized,
                requested_connector=connector,
                action=action,
                target=target,
                time_range=time_range,
                data_level=data_level,
                verification=verification,
                adapter_name=adapter_name,
                fixture_name=fixture_name,
                adapter_state=adapter_state,
                probe_state=probe_state,
                missing_fields=missing,
                suggested_risk=risk,
                approval_required=approval_required,
                requires_approval=approval_required,
                metadata_only=metadata_only,
                sample_rows=sample_rows,
                sample_row_count=sample_row_count,
                full_content_blocked=not metadata_only,
                side_effect_blocked=side_effect_requested,
                natural_language_routing_enabled=False,
                adapter_probe_ready=not missing,
                adapter_probe_handoff=adapter_probe_handoff,
                adapter_probe_handoff_ready=adapter_probe_handoff["handoff_ready"],
                adapter_probe_ready_for_operator=adapter_probe_handoff["ready_for_operator"],
                adapter_probe_state_changed=adapter_probe_handoff["state_changed"],
                adapter_probe_changed=adapter_probe_handoff["changed"],
                adapter_probe_content_in_handoff=adapter_probe_handoff["content_in_handoff"],
                adapter_probe_authorizes_execution=adapter_probe_handoff["authorizes_execution"],
                adapter_probe_authorizes_completion_claim=adapter_probe_handoff["authorizes_completion_claim"],
                adapter_probe_approval_granted=adapter_probe_handoff["approval_granted"],
                adapter_probe_boundaries=adapter_probe_handoff["boundaries"],
                adapter_probe_next_safe_commands=adapter_probe_handoff["next_safe_commands"],
                **row_contract,
            ),
        )

    def integration_adapter_acceptance(args: dict[str, Any]) -> ToolResult:
        connector = _clean_text(args.get("connector") or "email", limit=MAX_CONNECTOR_CHARS) or "email"
        target = _clean_text(args.get("target") or "selected source", limit=MAX_SCOPE_FIELD_CHARS) or "selected source"
        time_range = _clean_text(args.get("time_range") or "selected window", limit=MAX_SCOPE_FIELD_CHARS) or "selected window"
        normalized, profile = _connector_profile(connector)
        safe_action = profile["read_only"][0]
        side_effect_action = profile["side_effects"][0]

        probe_cases = [
            {
                "name": "metadata happy path",
                "expect": "ready",
                "args": {
                    "connector": normalized,
                    "action": safe_action,
                    "target": target,
                    "time_range": time_range,
                    "data_level": "metadata-only",
                    "verification": "fake rows only",
                },
            },
            {
                "name": "full content blocked",
                "expect": "blocked",
                "args": {
                    "connector": normalized,
                    "action": safe_action,
                    "target": target,
                    "time_range": time_range,
                    "data_level": "full body",
                    "verification": "fake rows only",
                },
            },
            {
                "name": "side effect blocked",
                "expect": "blocked",
                "args": {
                    "connector": normalized,
                    "action": side_effect_action,
                    "target": target,
                    "time_range": time_range,
                    "data_level": "metadata-only",
                    "verification": "fake rows only",
                },
            },
            {
                "name": "missing scope blocked",
                "expect": "blocked",
                "args": {
                    "connector": normalized,
                    "action": safe_action,
                    "target": "",
                    "time_range": "",
                    "data_level": "metadata-only",
                    "verification": "fake rows only",
                },
            },
        ]

        case_results: list[dict[str, Any]] = []
        for probe_case in probe_cases:
            result = integration_adapter_probe(probe_case["args"])
            metadata = result.metadata or {}
            probe_state = str(metadata.get("probe_state") or "")
            ready = probe_state == "ADAPTER_PROBE_READY"
            expected_ready = probe_case["expect"] == "ready"
            passed = ready == expected_ready
            if probe_case["name"] == "metadata happy path":
                passed = passed and metadata.get("sample_row_count") == 2
            if probe_case["name"] == "full content blocked":
                passed = passed and metadata.get("full_content_blocked") is True and metadata.get("sample_row_count") == 0
            if probe_case["name"] == "side effect blocked":
                passed = passed and metadata.get("side_effect_blocked") is True and metadata.get("sample_row_count") == 0
            if probe_case["name"] == "missing scope blocked":
                passed = passed and bool(metadata.get("missing_fields")) and metadata.get("sample_row_count") == 0
            case_results.append(
                {
                    "name": probe_case["name"],
                    "passed": passed,
                    "probe_state": probe_state,
                    "sample_row_count": metadata.get("sample_row_count", 0),
                    "sample_row_schema": metadata.get("sample_row_schema", list(METADATA_ROW_SCHEMA)),
                    "sample_row_limit": metadata.get("sample_row_limit", METADATA_ROW_LIMIT),
                    "sample_rows_bounded": metadata.get("sample_rows_bounded", False),
                    "blocked_payload_fields": metadata.get("blocked_payload_fields", list(BLOCKED_METADATA_PAYLOAD_FIELDS)),
                    "missing_fields": metadata.get("missing_fields", []),
                    "full_content_blocked": metadata.get("full_content_blocked", False),
                    "side_effect_blocked": metadata.get("side_effect_blocked", False),
                }
            )

        passed_count = sum(1 for row in case_results if row["passed"])
        acceptance_state = "ADAPTER_ACCEPTANCE_PASSED" if passed_count == len(case_results) else "ADAPTER_ACCEPTANCE_BLOCKED"
        adapter_name = f"{normalized}_metadata_adapter"
        all_case_row_contracts_present = all(
            row["sample_row_schema"] == METADATA_ROW_SCHEMA and row["sample_row_limit"] == METADATA_ROW_LIMIT
            for row in case_results
        )
        adapter_acceptance_handoff = _adapter_acceptance_handoff_payload(
            connector=normalized,
            requested_connector=connector,
            adapter_name=adapter_name,
            acceptance_state=acceptance_state,
            case_results=case_results,
            passed_cases=passed_count,
            sample_row_schema=list(METADATA_ROW_SCHEMA),
            sample_row_limit=METADATA_ROW_LIMIT,
            blocked_payload_fields=list(BLOCKED_METADATA_PAYLOAD_FIELDS),
            all_case_row_contracts_present=all_case_row_contracts_present,
        )
        adapter_acceptance_receipt_sha256 = adapter_acceptance_handoff["adapter_acceptance_receipt_sha256"]
        adapter_acceptance_receipt_boundary_rows = adapter_acceptance_handoff["adapter_acceptance_receipt_boundary_rows"]
        adapter_acceptance_receipt_contract_summary = adapter_acceptance_handoff["adapter_acceptance_receipt_contract_summary"]

        lines = [
            "Jarvis personal integration adapter acceptance report:",
            "This is read-only. It runs the disabled connector adapter proof cases with fake bounded metadata only; it does not connect accounts, read personal data, call external services, execute real adapter code, change state, control the computer, approve requests, or queue approvals.",
            "",
            f"Connector: {connector}",
            f"Normalized connector: {normalized}",
            f"Adapter: {adapter_name}",
            f"Acceptance state: {acceptance_state}",
            f"Cases passed: {passed_count}/{len(case_results)}",
            "",
            "Acceptance receipt:",
            f"- acceptance receipt sha256: {adapter_acceptance_receipt_sha256}",
            f"- receipt boundary: {', '.join(adapter_acceptance_receipt_contract_summary)}",
            f"- receipt rows: {len(adapter_acceptance_receipt_boundary_rows)}",
            "",
            "Proof cases:",
        ]
        for row in case_results:
            lines.extend(
                [
                    f"- {row['name']}: {'pass' if row['passed'] else 'fail'}",
                    f"  probe state: {row['probe_state']}",
                    f"  fake rows: {row['sample_row_count']}",
                    f"  row schema: {', '.join(row['sample_row_schema'])}",
                    f"  row limit: {row['sample_row_limit']}",
                    f"  bounded rows: {'yes' if row['sample_rows_bounded'] else 'blocked before rows'}",
                    f"  missing fields: {', '.join(row['missing_fields']) if row['missing_fields'] else 'none'}",
                ]
            )

        lines.extend(
            [
                "",
                "Acceptance rule:",
                "- Metadata-only happy path must return exactly two fake bounded rows.",
                "- Full-content, side-effect, and missing-scope paths must return zero rows and stay blocked.",
                "- Natural-language routing remains disabled even when this acceptance report passes.",
                "",
                "Next required commands:",
                f"- `integration adapter probe: {normalized} -> {safe_action}; target {target}; time {time_range}; data metadata-only; verification fake rows only`",
                f"- `integration enablement gate: {normalized} -> {safe_action}; target {target}; time {time_range}; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed`",
                "- `execution health report` after focused smoke tests.",
                "",
                "Hard stops:",
                OPERATOR_LIMIT_STOP,
                "- Do not enable a real connector, full-content read, side effect, or natural-language auto-routing from this report alone.",
                "- Do not request tokens, cookies, mailbox bodies, message transcripts, calendar notes, contact cards, attachments, or authenticated page content.",
            ]
        )

        return ToolResult(
            "integration_adapter_acceptance",
            True,
            "\n".join(lines),
            _safe_metadata(
                connector=normalized,
                requested_connector=connector,
                adapter_name=adapter_name,
                acceptance_state=acceptance_state,
                passed_cases=passed_count,
                total_cases=len(case_results),
                case_results=case_results,
                sample_row_schema=list(METADATA_ROW_SCHEMA),
                sample_row_limit=METADATA_ROW_LIMIT,
                blocked_payload_fields=list(BLOCKED_METADATA_PAYLOAD_FIELDS),
                all_case_row_contracts_present=all_case_row_contracts_present,
                adapter_acceptance_passed=acceptance_state == "ADAPTER_ACCEPTANCE_PASSED",
                adapter_acceptance_handoff=adapter_acceptance_handoff,
                adapter_acceptance_receipt_sha256=adapter_acceptance_receipt_sha256,
                adapter_acceptance_receipt_present=adapter_acceptance_handoff["adapter_acceptance_receipt_present"],
                adapter_acceptance_receipt_boundary_rows=adapter_acceptance_receipt_boundary_rows,
                adapter_acceptance_receipt_boundary_row_count=len(adapter_acceptance_receipt_boundary_rows),
                adapter_acceptance_receipt_boundary_ready=adapter_acceptance_handoff["adapter_acceptance_receipt_boundary_ready"],
                adapter_acceptance_receipt_contract_summary=adapter_acceptance_receipt_contract_summary,
                adapter_acceptance_receipt_authorizes_account_access=False,
                adapter_acceptance_receipt_authorizes_route_unlock=False,
                adapter_acceptance_receipt_authorizes_natural_language_routing=False,
                adapter_acceptance_receipt_authorizes_personal_data_read=False,
                adapter_acceptance_receipt_authorizes_side_effect=False,
                adapter_acceptance_receipt_authorizes_approval=False,
                adapter_acceptance_receipt_authorizes_model_call=False,
                adapter_acceptance_receipt_authorizes_tool_execution=False,
                adapter_acceptance_receipt_authorizes_external_service=False,
                adapter_acceptance_receipt_authorizes_execution=False,
                adapter_acceptance_receipt_authorizes_completion_claim=False,
                adapter_acceptance_receipt_approval_granted=False,
                adapter_acceptance_receipt_reusable_for_other_scope=False,
                adapter_acceptance_handoff_ready=adapter_acceptance_handoff["handoff_ready"],
                adapter_acceptance_ready_for_operator=adapter_acceptance_handoff["ready_for_operator"],
                adapter_acceptance_state_changed=adapter_acceptance_handoff["state_changed"],
                adapter_acceptance_changed=adapter_acceptance_handoff["changed"],
                adapter_acceptance_content_in_handoff=adapter_acceptance_handoff["content_in_handoff"],
                adapter_acceptance_authorizes_execution=adapter_acceptance_handoff["authorizes_execution"],
                adapter_acceptance_authorizes_completion_claim=adapter_acceptance_handoff["authorizes_completion_claim"],
                adapter_acceptance_approval_granted=adapter_acceptance_handoff["approval_granted"],
                adapter_acceptance_boundaries=adapter_acceptance_handoff["boundaries"],
                adapter_acceptance_next_safe_commands=adapter_acceptance_handoff["next_safe_commands"],
                natural_language_routing_enabled=False,
                requires_enablement_gate=True,
                requires_execution_health_report=True,
            ),
        )

    def integration_status(_: dict[str, Any]) -> ToolResult:
        jarvis_root = config.obsidian_vault / config.obsidian_root
        osascript_available = which("osascript") is not None
        obsidian_root_status = "ready" if jarvis_root.exists() else "not created yet"
        gated_v2_connectors = [
            {
                "connector": "calendar",
                "state": "active V2 connector",
                "safe_reads": ["list_calendars", "list_events", "check_availability"],
                "approval_gated": ["create_event", "update_event", "delete_event"],
            },
            {
                "connector": "email",
                "state": "active V2 connector",
                "safe_reads": ["read_email", "search_email", "read_email_body"],
                "approval_gated": ["send_email"],
            },
            {
                "connector": "messages",
                "state": "active V2 connectors",
                "safe_reads": ["recent_imessages"],
                "approval_gated": ["send_imessage", "send_kakao", "send_instagram_dm"],
            },
            {
                "connector": "contacts",
                "state": "active V2 resolver",
                "safe_reads": ["find_contact"],
                "approval_gated": [],
            },
        ]
        still_blocked_legacy_scope = [
            "logged-in browser state, cookies, history, authenticated page actions",
            "broad mailbox export, attachments, and implicit send/archive/delete flows",
            "broad calendar notes and ambiguous event edits/invites/deletes",
            "full contact-card export or contact create/edit/delete",
        ]
        lines = [
            "Jarvis personal integration boundary report:",
            "",
            "Active integrations:",
            "- Obsidian vault: configured local vault",
            f"- Jarvis root: configured Jarvis root ({obsidian_root_status})",
            "- Jarvis may read and write only inside its configured Jarvis Obsidian root by default.",
            "",
            "Approval-gated integrations:",
            f"- Reminders: {'AppleScript bridge available' if osascript_available else 'AppleScript bridge not found'}; creating a reminder is HIGH_RISK and requires explicit approval.",
            "- Clipboard: PERSONAL_DATA and requires explicit approval before reading.",
            "- Computer control: screenshots/clicks/typing are PERSONAL_DATA or HIGH_RISK and require explicit approval.",
            "",
            "Gated V2 personal connectors:",
        ]
        for row in gated_v2_connectors:
            safe_reads = ", ".join(row["safe_reads"])
            gated = ", ".join(row["approval_gated"]) if row["approval_gated"] else "none"
            lines.append(f"- {row['connector']}: {row['state']}; safe reads: {safe_reads}; approval-gated actions: {gated}.")
        lines.extend(
            [
                "",
                "Still blocked legacy scope:",
            ]
        )
        lines.extend(f"- {item}." for item in still_blocked_legacy_scope)
        lines.extend(
            [
            "",
            "Safe migration rule:",
            "- Existing V2 connectors keep their declared risk levels, approval prompts, audit logging, smoke coverage, and explicit scope boundaries.",
            "- Any broader legacy account behavior still needs a boundary contract, route lock, focused smoke, and aggregate smoke before Jarvis can use it.",
            "",
            "Good starter commands:",
            "- privacy report",
            "- safety status",
            "- find contact 가상연락처일",
            "- list calendars",
            "- search email from Alice",
            "- remind me to review Jarvis integrations",
            "- open Jarvis vault",
            ]
        )
        return ToolResult(
            "integration_status",
            True,
            "\n".join(lines),
            _safe_metadata(
                obsidian_root_exists=jarvis_root.exists(),
                obsidian_vault_display="configured local vault",
                obsidian_root_display="configured Jarvis root",
                osascript_available=osascript_available,
                gated_v2_connectors=gated_v2_connectors,
                gated_v2_connector_names=[row["connector"] for row in gated_v2_connectors],
                gated_v2_connector_count=len(gated_v2_connectors),
                still_blocked_legacy_scope=still_blocked_legacy_scope,
                still_blocked_legacy_scope_count=len(still_blocked_legacy_scope),
                unmigrated=[],
            ),
        )

    def legacy_connector_migration_audit(args: dict[str, Any]) -> ToolResult:
        requested = _clean_text(args.get("scope") or args.get("connector") or "browser calendar email", limit=MAX_ACTION_CHARS)
        requested_low = requested.lower()
        connectors = [name for name in ("browser", "calendar", "email") if name in requested_low]
        if not connectors:
            connectors = ["browser", "calendar", "email"]

        rows: list[dict[str, Any]] = []
        for connector_name in connectors:
            normalized, profile = _connector_profile(connector_name)
            if normalized == "browser":
                current_v2_surface = "public URL fetch/open/search tools exist; logged-in browser state remains blocked"
                safest_command = "integration action preview: browser -> fetch public pages"
                blocked_legacy_surface = "cookies, history, logged-in pages, form submission, authenticated controls"
                migration_lane = "public/read-only first"
            elif normalized == "calendar":
                current_v2_surface = "Google Calendar tools exist behind explicit risk levels; legacy broad calendar access still needs boundary review"
                safest_command = "integration adapter acceptance: calendar"
                blocked_legacy_surface = "broad calendar notes, ambiguous edits, implicit invites, deletes, or moves"
                migration_lane = "metadata/read-only first, mutations approval-gated"
            elif normalized == "email":
                current_v2_surface = "Gmail SMTP/IMAP tools exist behind explicit risk levels; legacy broad mailbox access still needs boundary review"
                safest_command = "integration adapter acceptance: email"
                blocked_legacy_surface = "full mailbox export, bodies/attachments by default, send/archive/delete/forward without one-shot approval"
                migration_lane = "metadata/draft-only first, sends approval-gated"
            else:
                current_v2_surface = "connector profile exists; implementation must start from the boundary contract"
                safest_command = f"integration boundary contract: {normalized}"
                blocked_legacy_surface = "broad account access or unscoped state changes"
                migration_lane = "contract first"
            rows.append(
                {
                    "connector": normalized,
                    "migration_lane": migration_lane,
                    "current_v2_surface": current_v2_surface,
                    "blocked_legacy_surface": blocked_legacy_surface,
                    "read_only_candidates": list(profile["read_only"]),
                    "personal_data_candidates": list(profile["personal_data"]),
                    "side_effect_candidates": list(profile["side_effects"]),
                    "required_commands": [
                        f"integration boundary contract: {normalized}",
                        f"integration migration plan: {normalized}",
                        f"integration execution matrix: {normalized}",
                        safest_command,
                        f"integration route lock: {normalized} -> <scoped metadata/draft action>; target <exact source>; time <exact window>; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed; status status api smoke passed",
                    ],
                }
            )

        audit_handoff = {
            "handoff_ready": True,
            "legacy_connector_migration_audit_handoff_ready": True,
            "ready_for_operator": True,
            "state_changed": False,
            "changed": [],
            "content_in_handoff": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "requested_scope": requested,
            "connectors": [row["connector"] for row in rows],
            "legacy_surfaces_reviewed": len(rows),
            "claude_handoff_state": "TASK-F contact resolver wiring done; TASK-G morning brief Telegram scheduling done; companion-layer pivot rejected; continue scoped connector readiness first",
            "migration_rows": rows,
            "proof_requirements": {
                "requires_boundary_contract": True,
                "requires_disabled_adapter_acceptance": True,
                "requires_route_lock": True,
                "requires_focused_smoke": True,
                "requires_aggregate_smoke": True,
                "requires_fresh_approval_for_personal_data_or_side_effects": True,
            },
            "blocked_until": [
                "exact source/target/time/data scope is declared",
                "metadata/draft-only disabled adapter acceptance passes",
                "full-content and side-effect paths are smoke-blocked",
                "route lock includes status/API smoke evidence and a fresh scope hash",
                "personal-data reads or side effects have one-shot approval chain proof",
            ],
            "boundaries": {
                "natural_language_routing_enabled": False,
                "calls_external_service": False,
                "reads_personal_data": False,
                "executes_side_effect": False,
                "controls_computer": False,
                "queues_approval": False,
                "authorizes_account_access": False,
                "authorizes_route_unlock": False,
                "authorizes_natural_language_routing": False,
                "authorizes_personal_data_read": False,
                "authorizes_side_effect": False,
                "authorizes_approval": False,
                "authorizes_tool_execution": False,
                "reusable_for_other_scope": False,
            },
            "next_safe_commands": [row["required_commands"][0] for row in rows],
        }

        lines = [
            "Jarvis legacy connector migration audit:",
            "This is read-only. It reviews old Jarvis browser/calendar/email migration readiness without connecting accounts, reading personal data, calling external services, controlling the computer, changing state, approving requests, or unlocking natural-language routing.",
            "",
            "Claude handoff:",
            "- TASK-F contact resolver wiring is marked done.",
            "- TASK-G morning brief Telegram scheduling is marked done.",
            "- Companion/team-cockpit architecture was rejected for now; continue scoped connector readiness first.",
            "",
            "Legacy surfaces reviewed:",
        ]
        for row in rows:
            lines.extend(
                [
                    f"- {row['connector']}: {row['migration_lane']}",
                    f"  current V2 surface: {row['current_v2_surface']}",
                    f"  blocked legacy surface: {row['blocked_legacy_surface']}",
                    f"  safest next command: `{row['required_commands'][3]}`",
                    f"  route-lock command template: `{row['required_commands'][4]}`",
                ]
            )

        lines.extend(
            [
                "",
                "Required proof before any legacy behavior is enabled:",
                "- Boundary contract, migration plan, execution matrix, disabled adapter acceptance, route lock, focused smoke, aggregate smoke.",
                "- Full content, authenticated browser state, sends, deletes, edits, invites, and broad account exports stay blocked without one-shot approval chain proof.",
                "",
                "Hard stops:",
                OPERATOR_LIMIT_STOP,
                "- Do not migrate legacy browser cookies/history, full mailbox export, broad calendar notes, or ambiguous send/edit/delete flows as shortcuts around connector boundaries.",
            ]
        )

        return ToolResult(
            "legacy_connector_migration_audit",
            True,
            "\n".join(lines),
            _safe_metadata(
                requested_scope=requested,
                connectors_reviewed=[row["connector"] for row in rows],
                legacy_surfaces_reviewed=len(rows),
                natural_language_routing_enabled=False,
                route_locks_enabled=False,
                requires_boundary_contract=True,
                requires_adapter_acceptance=True,
                requires_route_lock=True,
                requires_focused_smoke=True,
                requires_aggregate_smoke=True,
                legacy_connector_migration_audit_handoff=audit_handoff,
                legacy_connector_migration_audit_handoff_ready=True,
                legacy_connector_migration_audit_ready_for_operator=audit_handoff["ready_for_operator"],
                legacy_connector_migration_audit_state_changed=audit_handoff["state_changed"],
                legacy_connector_migration_audit_changed=audit_handoff["changed"],
                legacy_connector_migration_audit_content_in_handoff=audit_handoff["content_in_handoff"],
                legacy_connector_migration_audit_boundaries=audit_handoff["boundaries"],
                legacy_connector_migration_audit_next_safe_commands=audit_handoff["next_safe_commands"],
            ),
        )

    def integration_execution_matrix(args: dict[str, Any]) -> ToolResult:
        requested = _clean_text(args.get("connector"), limit=MAX_CONNECTOR_CHARS).lower()
        connector_names = [requested] if requested else list(CONNECTOR_PROFILES.keys())
        rows: list[dict[str, Any]] = []
        for connector_name in connector_names:
            normalized, profile = _connector_profile(connector_name)
            first_read_only = profile["read_only"][0]
            first_side_effect = profile["side_effects"][0]
            if normalized in {"calendar", "email", "messages", "contacts"}:
                lane = "metadata-or-draft first"
                adapter_state = "disabled"
                safest_command = (
                    f"integration metadata preview: {normalized} -> {first_read_only}; "
                    "target selected source; time selected window; data metadata-only; "
                    "verification fake rows only; tests blocked full-content smoke; "
                    "audit metadata preview receipt; acceptance gate passed"
                )
            elif normalized == "browser":
                lane = "public/read-only first"
                adapter_state = "disabled for logged-in pages"
                safest_command = "integration action preview: browser -> fetch public pages"
            elif normalized == "reminders":
                lane = "approval-gated side-effect"
                adapter_state = "approval-gated"
                safest_command = (
                    "integration runbook: reminders -> create reminder; target selected list; "
                    "time explicit due date; data reminder title only; verification reminder exists; "
                    "rollback delete reminder"
                )
            else:
                lane = "contract first"
                adapter_state = "disabled"
                safest_command = f"integration boundary contract: {normalized}"
            rows.append(
                {
                    "connector": normalized,
                    "lane": lane,
                    "adapter_state": adapter_state,
                    "read_only_count": len(profile["read_only"]),
                    "personal_data_count": len(profile["personal_data"]),
                    "side_effect_count": len(profile["side_effects"]),
                    "safest_command": safest_command,
                    "approval_gate": "required for personal data and side effects",
                    "blocked_tests": "missing scope, full private content, side effect without approval",
                    "natural_language_routing": "disabled until focused smoke tests pass",
                    "example_side_effect": first_side_effect,
                }
            )
        execution_matrix_handoff = _execution_matrix_handoff_payload(
            requested_connector=requested,
            rows=rows,
        )

        lines = [
            "Jarvis personal integration execution matrix:",
            "This is read-only. It shows which personal connector lanes can be rehearsed, what remains disabled, and which proof is required before any real adapter can run. It does not connect accounts, read personal data, call external services, change state, control the computer, approve requests, or queue approvals.",
            "",
            "Global connector rule:",
            "- Metadata-only and draft-only rehearsals may be previewed with fake rows and disabled adapters.",
            "- Personal-data reads and side effects require exact scope, one-shot approval, approval chain proof, audit receipt, verification, and rollback/stop evidence.",
            "- Natural-language auto-routing stays disabled until the connector has happy-path, blocked-scope, blocked-private-content, blocked-side-effect, audit, and acceptance smoke tests.",
            "",
            "Connector lanes:",
        ]
        for row in rows:
            lines.extend(
                [
                    f"- {row['connector']}: {row['lane']}",
                    f"  adapter state: {row['adapter_state']}",
                    f"  surfaces: read-only {row['read_only_count']}; personal-data {row['personal_data_count']}; side-effect {row['side_effect_count']}",
                    f"  approval gate: {row['approval_gate']}",
                    f"  blocked tests: {row['blocked_tests']}",
                    f"  natural-language routing: {row['natural_language_routing']}",
                    f"  safest next command: `{row['safest_command']}`",
                ]
            )

        lines.extend(
            [
                "",
                "Proof required before enabling any connector:",
                "- `integration preflight contract` with exact connector, action, target/source, time range, data level, verification, rollback, tests, and audit.",
                "- `integration enablement gate` with acceptance evidence and natural-language routing still disabled.",
                "- `integration rehearsal receipt` proving the disabled adapter boundary and expected audit metadata.",
                "- `integration metadata preview` for metadata-only paths with fake bounded rows and blocked full-content tests.",
                "- `execution health report` after connector smoke tests to confirm failures, approval gaps, and recovery paths are visible.",
                "",
                "Hard stops:",
                OPERATOR_LIMIT_STOP,
                "- Do not request tokens, cookies, broad account exports, full mailbox/message bodies, contact cards, calendar notes, or authenticated browser state from this matrix.",
                "- Do not send, delete, invite, forward, submit, purchase, mark read, or mutate records without approval readiness, a fresh approval packet, and approval chain proof for the exact action.",
            ]
        )
        return ToolResult(
            "integration_execution_matrix",
            True,
            "\n".join(lines),
            _safe_metadata(
                connectors_reviewed=len(rows),
                connector=requested or "all",
                lanes=[row["lane"] for row in rows],
                adapter_states=[row["adapter_state"] for row in rows],
                natural_language_routing_enabled=False,
                requires_smoke_tests=True,
                requires_acceptance_gate=True,
                requires_execution_health_report=True,
                safest_commands=[row["safest_command"] for row in rows],
                execution_matrix_handoff=execution_matrix_handoff,
                execution_matrix_handoff_ready=execution_matrix_handoff["handoff_ready"],
                execution_matrix_ready_for_operator=execution_matrix_handoff["ready_for_operator"],
                execution_matrix_state_changed=execution_matrix_handoff["state_changed"],
                execution_matrix_changed=execution_matrix_handoff["changed"],
                execution_matrix_content_in_handoff=execution_matrix_handoff["content_in_handoff"],
                execution_matrix_authorizes_execution=execution_matrix_handoff["authorizes_execution"],
                execution_matrix_authorizes_completion_claim=execution_matrix_handoff["authorizes_completion_claim"],
                execution_matrix_approval_granted=execution_matrix_handoff["approval_granted"],
                execution_matrix_boundaries=execution_matrix_handoff["boundaries"],
                execution_matrix_next_safe_commands=execution_matrix_handoff["next_safe_commands"],
            ),
        )

    def integration_migration_plan(args: dict[str, Any]) -> ToolResult:
        connector = _clean_text(args.get("connector") or "calendar", limit=MAX_CONNECTOR_CHARS) or "calendar"
        normalized, profile = _connector_profile(connector)
        migration_plan_handoff = _migration_plan_handoff_payload(
            connector=normalized,
            requested_connector=connector,
            profile=profile,
        )
        lines = [
            "Jarvis personal integration migration plan:",
            f"Connector: {connector}",
            "",
            "Phase 1 - boundary contract:",
            "- Define the connector owner, account/source, data scope, and off switch.",
            "- Keep the first implementation read-only unless the action is already covered by an explicit approval gate.",
            "- Document what Jarvis may cache in SQLite or Obsidian and what must stay transient.",
            "",
            "Phase 2 - risk mapping:",
            "- READ_ONLY candidates: " + ", ".join(profile["read_only"]) + ".",
            "- PERSONAL_DATA candidates: " + ", ".join(profile["personal_data"]) + ".",
            "- EXTERNAL_SIDE_EFFECT or HIGH_RISK candidates: " + ", ".join(profile["side_effects"]) + ".",
            "",
            "Phase 3 - implementation checklist:",
            "- Add one small tool with a clear risk level and a narrow argument schema.",
            "- Route natural-language commands only after the tool has smoke tests.",
            "- Log every run through the existing audit trail.",
            "- Mirror any blocked risky request into the pending approval queue.",
            "- Add README examples that show the safe command and the approval-gated command.",
            "",
            "Phase 4 - verification:",
            "- Add a focused smoke test proving read-only behavior works.",
            "- Add a blocked-action smoke test for personal data or side effects.",
            "- Run `python3 -m jarvis_v2.scripts.smoke_test_all` before calling the connector ready.",
            "",
            "Hard safety rule:",
            OPERATOR_LIMIT_STOP,
            "- Do not migrate broad account access, sending, deleting, purchases, or authenticated browser actions without a per-action approval receipt and approval chain proof.",
        ]
        return ToolResult(
            "integration_migration_plan",
            True,
            "\n".join(lines),
            _safe_metadata(
                connector=normalized,
                requested_connector=connector,
                read_only_candidates=list(profile["read_only"]),
                personal_data_candidates=list(profile["personal_data"]),
                side_effect_candidates=list(profile["side_effects"]),
                migration_plan_handoff=migration_plan_handoff,
                migration_plan_handoff_ready=migration_plan_handoff["handoff_ready"],
                migration_plan_ready_for_operator=migration_plan_handoff["ready_for_operator"],
                migration_plan_state_changed=migration_plan_handoff["state_changed"],
                migration_plan_changed=migration_plan_handoff["changed"],
                migration_plan_content_in_handoff=migration_plan_handoff["content_in_handoff"],
                migration_plan_authorizes_execution=migration_plan_handoff["authorizes_execution"],
                migration_plan_authorizes_completion_claim=migration_plan_handoff["authorizes_completion_claim"],
                migration_plan_approval_granted=migration_plan_handoff["approval_granted"],
                migration_plan_boundaries=migration_plan_handoff["boundaries"],
                migration_plan_next_safe_commands=migration_plan_handoff["next_safe_commands"],
            ),
        )

    def integration_boundary_contract(args: dict[str, Any]) -> ToolResult:
        connector = _clean_text(args.get("connector") or "calendar", limit=MAX_CONNECTOR_CHARS) or "calendar"
        normalized, profile = _connector_profile(connector)
        lines = [
            "Jarvis personal integration boundary contract:",
            f"Connector: {connector}",
            f"Normalized connector: {normalized}",
            "",
            "Default state:",
            "- Not connected by default.",
            "- No account tokens, cookies, mailbox/calendar/message data, or contact cards are available to Jarvis unless a connector is explicitly implemented later.",
            "- First implementation should be read-only and narrow.",
            "",
            "Allowed without approval:",
            "- Show this contract.",
            "- Draft a migration plan.",
            "- Draft reply/event/reminder text without sending or creating it.",
            "- Summarize user-pasted text that the operator intentionally provided in chat.",
            "",
            "Requires approval before reading personal data:",
        ]
        for item in profile["personal_data"]:
            lines.append(f"- {item}")
        lines.extend(["", "Requires approval before side effects:"])
        for item in profile["side_effects"]:
            lines.append(f"- {item}")
        lines.extend(
            [
                "",
                "Read-only candidate tools:",
            ]
        )
        for item in profile["read_only"]:
            lines.append(f"- {item}")
        lines.extend(
            [
                "",
                "Approval prompt template:",
                f"- Tool: <future_{normalized}_tool>",
                "- Requested action: <exact read or side effect>",
                "- Data/source: <account, list, thread, event, or conversation>",
                "- Scope limit: <time range, selected item, or explicit recipients>",
                "- Why needed: <user-visible reason>",
                "- Safer alternative: paste relevant text, use metadata-only summary, or draft without sending.",
                "",
                "Caching rules:",
                "- Store only short summaries, task references, decisions, or user-approved notes in Jarvis memory.",
                "- Do not cache full email bodies, message transcripts, calendar notes, contact cards, tokens, cookies, or attachments by default.",
                "- Include source and timestamp when saving a user-approved summary.",
                "",
                "Audit and rollback:",
                "- Log every connector run through the existing tool-run audit trail.",
                "- Side effects must create a pending approval receipt before running.",
                "- Provide a read-only preview and a cancel/dismiss path before any send/create/delete/update action.",
                "",
                "Hard stops:",
                OPERATOR_LIMIT_STOP,
                "- Do not send, delete, purchase, invite, forward, submit, or mark items read without an explicit per-action approval and approval chain proof.",
                "- Do not use broad authenticated browser control as a shortcut around connector boundaries.",
                "- If the target account, recipient, file, or conversation is ambiguous, stop and ask the operator.",
            ]
        )
        boundary_contract_handoff = _boundary_contract_handoff_payload(
            connector=normalized,
            requested_connector=connector,
            profile=profile,
        )
        return ToolResult(
            "integration_boundary_contract",
            True,
            "\n".join(lines),
            _safe_metadata(
                connector=normalized,
                requested_connector=connector,
                read_only_candidates=len(profile["read_only"]),
                read_only_candidate_items=list(profile["read_only"]),
                personal_data_surfaces=len(profile["personal_data"]),
                personal_data_surface_items=list(profile["personal_data"]),
                side_effect_surfaces=len(profile["side_effects"]),
                side_effect_surface_items=list(profile["side_effects"]),
                boundary_contract_handoff=boundary_contract_handoff,
                boundary_contract_handoff_ready=boundary_contract_handoff["handoff_ready"],
                boundary_contract_ready_for_operator=boundary_contract_handoff["ready_for_operator"],
                boundary_contract_state_changed=boundary_contract_handoff["state_changed"],
                boundary_contract_changed=boundary_contract_handoff["changed"],
                boundary_contract_content_in_handoff=boundary_contract_handoff["content_in_handoff"],
                boundary_contract_authorizes_execution=boundary_contract_handoff["authorizes_execution"],
                boundary_contract_authorizes_completion_claim=boundary_contract_handoff["authorizes_completion_claim"],
                boundary_contract_approval_granted=boundary_contract_handoff["approval_granted"],
                boundary_contract_boundaries=boundary_contract_handoff["boundaries"],
                boundary_contract_next_safe_commands=boundary_contract_handoff["next_safe_commands"],
            ),
        )

    def open_jarvis_vault(_: dict[str, Any]) -> ToolResult:
        path = config.obsidian_vault / config.obsidian_root
        path_display = "Jarvis vault"
        try:
            result = subprocess.run(["open", str(path)], capture_output=True, text=True)
        except Exception as exc:
            return _open_vault_outcome_unknown(path, exception_type=type(exc).__name__)
        if result.returncode == 0:
            return ToolResult(
                "open_jarvis_vault",
                True,
                f"Opened {path_display}.",
                _safe_metadata(path=str(path), path_display=path_display, executes_tools=True, controls_computer=True),
            )
        return _open_vault_outcome_unknown(path, returncode=result.returncode)

    def create_reminder(args: dict[str, Any]) -> ToolResult:
        raw_title = args.get("title")
        title = _clean_text(raw_title, limit=MAX_REMINDER_TITLE_CHARS)
        if not title:
            return _personal_input_failure(
                "create_reminder",
                "Reminder title is required.",
                "Provide a short plain-text reminder title, then submit it through the normal approval policy.",
                reason="missing_title",
                retry_safe=False,
            )
        if _has_local_path(raw_title):
            return _personal_input_failure(
                "create_reminder",
                "Reminder title cannot be a local file path.",
                "Replace it with a short plain-text title, then submit it through the normal approval policy.",
                reason="invalid_title",
                title=title,
                executes_tools=False,
                executes_side_effect=False,
                external_side_effect=False,
                controls_computer=False,
                retry_safe=False,
            )

        def post_attempt_failure(
            *,
            exception_type: str | None = None,
            returncode: int | None = None,
        ) -> ToolResult:
            failure_output = (
                "Reminder creation may have completed. Check Reminders for the exact title "
                "before any new request; do not automatically retry. If absent, run "
                "`setup check`, fix Reminders Automation access, then submit a new approved request."
            )
            failure_metadata = declare_outcome_unknown_failure(
                _safe_metadata(
                    title_length=len(title),
                    executes_tools=True,
                    executes_side_effect=True,
                    external_side_effect=True,
                    controls_computer=True,
                    exception_type=exception_type,
                    returncode=returncode,
                ),
                output=failure_output,
                commands=("setup check",),
            )
            return ToolResult(
                "create_reminder",
                False,
                failure_output,
                failure_metadata,
            )

        safe_title = title.replace("\\", "\\\\").replace('"', '\\"')
        script = f'tell application "Reminders" to make new reminder with properties {{name:"{safe_title}"}}'
        try:
            result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        except Exception as exc:
            return post_attempt_failure(exception_type=type(exc).__name__)
        if result.returncode == 0:
            return ToolResult(
                "create_reminder",
                True,
                f"Reminder created: {title}",
                _safe_metadata(title_length=len(title), executes_tools=True, executes_side_effect=True, external_side_effect=True, controls_computer=True),
            )
        return post_attempt_failure(returncode=result.returncode)

    return integration_status, integration_readiness_report, legacy_connector_migration_audit, integration_migration_plan, integration_boundary_contract, integration_action_preview, integration_scope_packet, integration_dry_run_contract, integration_runbook, integration_promotion_gate, integration_implementation_spec, integration_preflight_contract, integration_enablement_gate, integration_rehearsal_receipt, integration_metadata_preview, integration_proof_bundle, integration_implementation_review, integration_route_lock, integration_execution_matrix, integration_adapter_manifest, integration_adapter_probe, integration_adapter_acceptance, open_jarvis_vault, create_reminder
