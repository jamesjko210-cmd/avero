from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from jarvis_v2.agent.model_provider import (
    normalized_model_provider,
    ollama_local_only_policy,
    probe_ollama_models,
    resolve_ollama_destination,
)
from jarvis_v2.agent.types import ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import MemoryStore
from jarvis_v2.tools.harness import (
    AGI_GATE_BUILD_TARGETS,
    AGI_GATE_EVIDENCE,
    _agi_gate_summary,
    _execution_learning_debt_snapshot,
    _execution_health_recovery_closure_snapshot,
    _selected_agi_target_readiness,
    _target_file_integrity,
)
from jarvis_v2.tools.storage import (
    BOOTSTRAP_CHECK_COMMAND,
    BOOTSTRAP_WRITE_COMMAND,
    STORAGE_RECOVERY_CHECK_API,
    STORAGE_RECOVERY_CHECK_COMMAND,
    STORAGE_RECOVERY_PLAN_COMMAND,
    _configured_storage_diagnostics,
    safe_storage_text,
)
from jarvis_v2.tools.system import (
    DEFAULT_CHAT_MAX_HISTORY_MESSAGES,
    DEFAULT_CHAT_MAX_REPLY_TOKENS,
    DEFAULT_CHAT_TIMEOUT_SECONDS,
    DEFAULT_MODEL_ALIAS,
    DEFAULT_MODEL_TIMEOUT_SECONDS,
    DEFAULT_SMOKE_MODULE_TIMEOUT_SECONDS,
    MODEL_PROVIDERS,
    REASONING_EFFORTS,
    MIN_CHAT_MAX_HISTORY_MESSAGES,
    MIN_CHAT_MAX_REPLY_TOKENS,
    MAX_CHAT_MAX_HISTORY_MESSAGES,
    MAX_CHAT_MAX_REPLY_TOKENS,
    MAX_OLLAMA_CHAT_TIMEOUT_SECONDS,
    MAX_OLLAMA_MODEL_TIMEOUT_SECONDS,
    MAX_OPENAI_CHAT_TIMEOUT_SECONDS,
    MAX_OPENAI_MODEL_TIMEOUT_SECONDS,
    MIN_CHAT_TIMEOUT_SECONDS,
    MIN_MODEL_TIMEOUT_SECONDS,
    _local_command_env_status,
    _local_path_env_status,
    _ollama_no_cloud_policy_detail,
    _enum_env_status,
    _model_alias_status,
    _model_planner_toggle_status,
    _remote_compaction_toggle_status,
    _remote_personal_context_toggle_status,
    _runtime_int_status,
    _runtime_int_validation_detail,
    _runtime_timeout_status,
    _runtime_timeout_validation_detail,
    _secret_env_status,
    _smoke_timeout_status,
    _storage_fallback_status,
)
from jarvis_v2.v3_commands import (
    V3_DASHBOARD_COMMAND,
    V3_DASHBOARD_INFO_COMMAND,
    V3_DASHBOARD_MODULE_COMMAND,
    V3_PROJECT_PYTHON,
)
from jarvis_v2.ui.status_config import DEFAULT_STATUS_HOST, DEFAULT_STATUS_PORT, status_host_config_from_env, status_port_config_from_env


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "calls_model": False,
        "calls_external_service": False,
        "executes_tools": False,
        "queues_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "requires_approval": False,
        "approves_request": False,
        "dismisses_request": False,
        "reads_personal_data": False,
        "reads_private_data": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "controls_computer": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "speaks": False,
        "completes_tasks": False,
        "reads_clipboard": False,
    }
    metadata.update(extra)
    return metadata


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


_ROW_MISSING = object()


def _row_value(row: Any, key: str, default: Any = _ROW_MISSING) -> Any:
    try:
        if isinstance(row, dict):
            return row.get(key, default)
        return row[key]
    except Exception:
        return default


def _row_text(row: Any, key: str, default: str = "") -> str:
    value = _row_value(row, key, default)
    try:
        return safe_storage_text(value).replace("\n", " ").replace("\r", " ").strip()
    except Exception:
        return default


def _row_bool(row: Any, key: str, default: bool = False) -> bool:
    value = _row_value(row, key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    return default


def _row_positive_int_text(row: Any, key: str = "id") -> str:
    value = _row_value(row, key)
    if isinstance(value, bool) or value is _ROW_MISSING:
        return ""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return ""
    if number <= 0:
        return ""
    return str(number)


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
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().casefold()).strip("_")


def _row_metadata(row: Any) -> dict[str, Any]:
    raw = _row_value(row, "metadata", {})
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


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
    if _row_bool(row, "ok", default=True):
        return False
    return _is_approval_hold_metadata(_row_metadata(row))


def _recent_tool_run_attention_buckets(rows: list[Any]) -> tuple[list[Any], list[Any]]:
    recent_failures: list[Any] = []
    approval_held_runs: list[Any] = []
    for row in rows:
        if _row_bool(row, "ok", default=True):
            continue
        if _is_approval_held_tool_run(row):
            approval_held_runs.append(row)
        else:
            recent_failures.append(row)
    return recent_failures, approval_held_runs


def _positive_int_text(value: Any) -> str:
    if isinstance(value, bool) or value is _ROW_MISSING or value is None:
        return ""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return ""
    return str(number) if number > 0 else ""


def _first_positive_int_text(rows: list[Any], key: str = "id") -> str:
    for row in rows:
        if row_id := _row_positive_int_text(row, key):
            return row_id
    return ""


def _approval_id_from_run(row: Any) -> str:
    direct = _positive_int_text(_row_value(row, "approval_id"))
    if direct:
        return direct
    metadata = _row_metadata(row)
    for key in ("approval_id", "pending_approval_id", "approved_approval_id"):
        if approval_id := _positive_int_text(metadata.get(key)):
            return approval_id
    return ""


def _approval_review_commands_for_run(row: Any) -> list[str]:
    approval_id = _approval_id_from_run(row)
    if not approval_id:
        return []
    return [
        f"approval readiness {approval_id}",
        f"approval packet {approval_id}",
        f"approval chain proof {approval_id}",
        f"verification receipt <approved run id from approval chain proof {approval_id}>",
    ]


def _first_approval_held_review_commands(rows: list[Any]) -> list[str]:
    for row in rows:
        commands = _approval_review_commands_for_run(row)
        if commands:
            return commands
    return []


_STORAGE_DIAGNOSTIC_BOOL_KEYS = (
    "available",
    "data_dir_exists",
    "db_exists",
    "db_parent_exists",
    "data_dir_writable",
    "db_parent_writable",
    "db_file_writable",
    "obsidian_vault_exists",
    "obsidian_root_exists",
    "obsidian_vault_writable",
    "obsidian_root_writable",
    "workspace_local_notes",
    "metadata_only",
)


def _dedupe_commands(commands: list[str]) -> list[str]:
    deduped: list[str] = []
    for command in commands:
        if command and command not in deduped:
            deduped.append(command)
    return deduped


def _module_ok(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _command_ok(
    args: list[str],
    timeout: int = 5,
    *,
    env: dict[str, str] | None = None,
) -> tuple[bool, str]:
    try:
        run_kwargs: dict[str, Any] = {
            "capture_output": True,
            "text": True,
            "timeout": timeout,
        }
        if env is not None:
            run_kwargs["env"] = env
        result = subprocess.run(args, **run_kwargs)
    except Exception as exc:
        return False, f"{type(exc).__name__}; check local setup logs"
    output = (result.stdout or result.stderr).strip()
    return result.returncode == 0, safe_storage_text(output)


def _command_available(name: str) -> tuple[bool, str]:
    path = shutil.which(name)
    if path:
        return True, f"available at {path}; not executed"
    return False, "not found"


def _ollama_destination_policy_detail(destination: Any) -> str:
    family = destination.address_family or "unavailable"
    port = str(destination.port) if destination.port else "unavailable"
    return (
        f"source: {destination.source}; family: {family}; port: {port}; "
        "redirects: blocked; proxy policy: stripped from accepted probes"
    )


def _safe_ollama_probe_diagnostic(value: Any, *, ok: bool) -> str:
    token = str(value or "").strip().casefold()
    if re.fullmatch(r"ollama_[a-z0-9_]{1,72}", token):
        return token
    return "ollama_probe_ok" if ok else "ollama_probe_unavailable"


def _first_execution_health_command(recent_runs: list[Any]) -> str:
    meta_tools = {
        "recent_tool_runs",
        "verification_receipt",
        "runtime_trace_receipt",
        "execution_audit_gate",
        "execution_recovery_packet",
        "after_action_learning_packet",
        "execution_health_report",
    }
    action_runs = [row for row in recent_runs if (tool_name := _row_text(row, "tool_name")) and tool_name not in meta_tools]
    failed_action_runs = [row for row in action_runs if not _row_bool(row, "ok")]
    if failed_action_runs:
        run_id = _row_positive_int_text(failed_action_runs[0])
        return f"execution recovery packet {run_id}" if run_id else "execution health report"
    if recent_runs:
        return "execution health report"
    return "readiness report"


def _next_agi_gate_handoff(list_tools: Callable[[], list[Any]] | None = None) -> dict[str, Any]:
    selected_gate = "personal integrations"
    gate_summary = _agi_gate_summary(set())
    selection_source = "doctor_default_gate"
    selection_reason = (
        "Jarvis doctor uses a setup-readiness default AGI handoff track because it "
        "does not receive live registry or tool evidence; dynamic evidence and "
        "completion ledgers may select a different next gate."
    )
    if list_tools is not None:
        try:
            tool_names = {str(tool.name) for tool in list_tools()}
            gate_summary = _agi_gate_summary(tool_names)
            selected_readiness = _selected_agi_target_readiness(gate_summary)
            selected_gate = str(selected_readiness.get("gate_name") or selected_gate)
            selection_source = "harness_dynamic_registry"
            selection_reason = (
                "Jarvis doctor is using the same current registry-evidence AGI "
                "selector as harness readiness and completion proof surfaces."
            )
        except Exception:
            selected_readiness = {}
            selection_source = "doctor_default_gate_after_selector_error"
            selection_reason = (
                "Jarvis doctor could not read live registry evidence, so it fell "
                "back to the setup-readiness default AGI handoff track."
            )
    else:
        selected_readiness = {}
    configured_gates = [str(row.get("gate") or "") for row in AGI_GATE_EVIDENCE]
    if selected_gate not in configured_gates and configured_gates:
        selected_gate = configured_gates[0]
    target = AGI_GATE_BUILD_TARGETS.get(selected_gate, {})
    likely_files = list(selected_readiness.get("likely_files") or target.get("files") or [])
    file_integrity = selected_readiness.get("file_integrity") or _target_file_integrity(likely_files)
    closure_commands = list(selected_readiness.get("closure_commands") or [])
    if not closure_commands:
        closure_commands = [
            f"agi next build move: {selected_gate}",
            f"completion audit: improve AGI gate {selected_gate}",
            "evidence ledger",
            f"completion claim gate: improve AGI gate {selected_gate}",
        ]
    next_command = closure_commands[0]
    acceptance_checks = list(selected_readiness.get("acceptance_checks") or target.get("acceptance") or [])
    acceptance_gap_preview = acceptance_checks[:3]
    first_acceptance_gap = acceptance_gap_preview[0] if acceptance_gap_preview else ""
    return {
        "gate": selected_gate,
        "selection_source": selection_source,
        "selection_reason": selection_reason,
        "canonical_selector_command": next_command,
        "deliberate_focus_override": False,
        "configured_gate_names": configured_gates,
        "configured_gate_count": len(configured_gates),
        "next_command": next_command,
        "closure_commands": closure_commands,
        "likely_files": likely_files,
        "target_file_integrity": file_integrity,
        "focused_verification_commands": list(selected_readiness.get("verification_commands") or target.get("tests") or []),
        "acceptance_checks": acceptance_checks,
        "acceptance_gap_preview": acceptance_gap_preview,
        "acceptance_gap_preview_count": len(acceptance_gap_preview),
        "first_acceptance_gap": first_acceptance_gap,
        "build_target": str(selected_readiness.get("target_title") or target.get("title") or ""),
        "real_execution_gaps_by_gate": dict(gate_summary.get("real_execution_gaps_by_gate") or {}),
        "real_execution_gap_count": int(gate_summary.get("real_execution_gap_count") or 0),
        "selected_real_execution_gap": str((gate_summary.get("real_execution_gaps_by_gate") or {}).get(selected_gate) or ""),
    }


def _doctor_handoff(
    *,
    missing: list[str],
    checks: list[tuple[str, bool, str]],
    next_commands: list[str],
    pending_approvals: list[Any],
    open_tasks: list[Any],
    active_goals: list[Any],
    recent_runs: list[Any],
    recent_failures: list[Any],
    recent_approval_held_runs: list[Any],
    verification_runs: list[Any],
    learning_runs: list[Any],
    next_audit_command: str,
    recovery_closure: dict[str, Any],
    learning_debt: dict[str, Any],
    audit_readability_review_commands: list[str],
    completion_claim_state: str,
    completion_blockers: list[str],
    completion_proof_queue: list[str],
    agi_handoff: dict[str, Any],
    agi_build_packet_ready: bool,
    storage: dict[str, Any],
    sqlite_store_exception_type: str,
) -> dict[str, Any]:
    agi_target_integrity = agi_handoff["target_file_integrity"]
    approval_held_review_commands = _first_approval_held_review_commands(recent_approval_held_runs)
    approval_held_review_next_command = approval_held_review_commands[0] if approval_held_review_commands else ""
    approval_held_review_approval_id = (
        approval_held_review_next_command.removeprefix("approval readiness ")
        if approval_held_review_next_command.startswith("approval readiness ")
        else ""
    )
    return {
        "kind": "doctor_handoff",
        "status": "needs_attention" if missing or completion_blockers else "ready_for_human_review",
        "diagnostic_only": True,
        "content_in_handoff": False,
        "checks_count": len(checks),
        "missing": missing,
        "missing_count": len(missing),
        "next_commands": next_commands,
        "next_command_count": len(next_commands),
        "harness_readiness": {
            "pending_approvals": len(pending_approvals),
            "open_tasks": len(open_tasks),
            "active_goals": len(active_goals),
            "recent_tool_runs": len(recent_runs),
            "recent_failed_runs": len(recent_failures),
            "recent_approval_held_runs": len(recent_approval_held_runs),
            "approval_held_review_required": bool(recent_approval_held_runs),
            "approval_held_review_commands": approval_held_review_commands,
            "approval_held_review_command_count": len(approval_held_review_commands),
            "approval_held_review_next_command": approval_held_review_next_command,
            "approval_held_review_approval_id": approval_held_review_approval_id,
            "recent_verification_runs": len(verification_runs),
            "recent_after_action_learning_runs": len(learning_runs),
            "next_audit_command": next_audit_command,
            "audit_readability_review_required": bool(audit_readability_review_commands),
            "audit_readability_review_commands": audit_readability_review_commands,
            "audit_readability_review_command_count": len(audit_readability_review_commands),
            "audit_readability_review_next_command": (
                audit_readability_review_commands[0] if audit_readability_review_commands else ""
            ),
        },
        "recovery_closure": {
            "state": recovery_closure["state"],
            "ready_to_retry": recovery_closure["ready_to_retry"],
            "missing": recovery_closure["missing"],
            "missing_count": recovery_closure["missing_count"],
            "proof_queue": recovery_closure["required_commands"],
            "proof_queue_count": len(recovery_closure["required_commands"]),
            "next_required_command": recovery_closure["next_required_command"],
            "next_proof_command": recovery_closure["next_required_command"],
            "blocks_completion_claim": recovery_closure["blocks_completion_claim"],
            "target_run_id": recovery_closure["target_run_id"],
            "target_tool_name": recovery_closure["target_tool_name"],
        },
        "execution_learning": {
            "state": learning_debt["state"],
            "blocks_completion_claim": learning_debt["blocks_completion_claim"],
            "missing": learning_debt["missing"],
            "missing_count": learning_debt["missing_count"],
            "proof_queue": learning_debt["required_commands"],
            "proof_queue_count": len(learning_debt["required_commands"]),
            "next_required_command": learning_debt["next_required_command"],
            "next_proof_command": learning_debt["next_required_command"],
            "actionable_required_commands": learning_debt["actionable_required_commands"],
            "actionable_required_command_count": learning_debt["actionable_required_command_count"],
            "actionable_proof_queue": learning_debt["actionable_proof_queue"],
            "actionable_proof_queue_count": learning_debt["actionable_proof_queue_count"],
            "actionable_next_required_command": learning_debt["actionable_next_required_command"],
            "actionable_next_proof_command": learning_debt["actionable_next_proof_command"],
            "next_evidence_command": learning_debt["next_evidence_command"],
            "target_run_id": learning_debt["target_run_id"],
            "target_tool_name": learning_debt["target_tool_name"],
        },
        "completion": {
            "claim_state": completion_claim_state,
            "claim_ready": completion_claim_state == "READY_FOR_HUMAN_REVIEW",
            "blockers": completion_blockers,
            "blocker_count": len(completion_blockers),
            "proof_queue": completion_proof_queue,
            "proof_queue_count": len(completion_proof_queue),
            "next_proof_command": completion_proof_queue[0] if completion_proof_queue else "",
        },
        "agi_next": {
            "gate": agi_handoff["gate"],
            "selection_source": agi_handoff["selection_source"],
            "selection_reason": agi_handoff["selection_reason"],
            "canonical_selector_command": agi_handoff["canonical_selector_command"],
            "deliberate_focus_override": agi_handoff["deliberate_focus_override"],
            "next_build_command": agi_handoff["next_command"],
            "build_command": agi_handoff["next_command"],
            "build_target": agi_handoff["build_target"],
            "target_title": agi_handoff["build_target"],
            "evidence_closure_commands": agi_handoff["closure_commands"],
            "evidence_closure_command_count": len(agi_handoff["closure_commands"]),
            "focused_verification_commands": agi_handoff["focused_verification_commands"],
            "focused_verification_command_count": len(agi_handoff["focused_verification_commands"]),
            "target_file_integrity_status": agi_target_integrity["status"],
            "target_files_checked": agi_target_integrity["checked"],
            "target_files_exist": agi_target_integrity["all_exist"],
            "missing_target_files": agi_target_integrity["missing"],
            "missing_target_file_count": agi_target_integrity["missing_count"],
            "build_packet_ready_for_review": agi_build_packet_ready,
            "real_execution_gap_count": agi_handoff["real_execution_gap_count"],
            "real_execution_gaps_by_gate": agi_handoff["real_execution_gaps_by_gate"],
            "selected_real_execution_gap": agi_handoff["selected_real_execution_gap"],
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
        },
        "storage": {
            "status": storage["status"],
            "available": storage["available"],
            "configured_status": storage.get("configured_status", storage["status"]),
            "configured_available": storage.get("configured_available", storage["available"]),
            "ready_for_completion_claim": storage.get("ready_for_completion_claim", storage["available"]),
            "runtime_fallback_active": storage.get("runtime_fallback_active", False),
            "runtime_fallback_reason": storage.get("runtime_fallback_reason", ""),
            "runtime_fallback_exception_type": storage.get("runtime_fallback_exception_type", ""),
            "metadata_only": storage["metadata_only"],
            "issues": storage.get("issues", []),
            "recovery_required": storage.get("recovery_required", bool(storage.get("issues", []))),
            "recovery_reason": storage.get("recovery_reason", ""),
            "recovery_mode": storage.get("recovery_mode", ""),
            "recovery_next_operator_action": storage.get("recovery_next_operator_action", ""),
            "recovery_restart_required": storage.get("recovery_restart_required", False),
            "readiness_blocks_completion_claim": storage.get("readiness_blocks_completion_claim", False),
            "readiness_blocker": storage.get("readiness_blocker", ""),
            "readiness_next_commands": storage.get("readiness_next_commands", []),
            "readiness_next_command_count": storage.get("readiness_next_command_count", 0),
            "readiness_next_required_command": storage.get("readiness_next_required_command", ""),
            "readiness_next_proof_command": storage.get("readiness_next_proof_command", ""),
            "recovery_check_tool_command": storage.get("recovery_check_tool_command", ""),
            "recovery_check_command": storage.get("recovery_check_command", ""),
            "recovery_command": storage.get("recovery_command", ""),
            "sqlite_store_available": not sqlite_store_exception_type,
            "sqlite_store_exception_type": sqlite_store_exception_type,
        },
        "boundaries": {
            "calls_model": False,
            "calls_external_service": False,
            "executes_tools": False,
            "queues_approval": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "requires_approval": False,
            "approves_request": False,
            "dismisses_request": False,
            "reads_private_data": False,
            "reads_personal_data": False,
            "reads_clipboard": False,
            "executes_side_effect": False,
            "external_side_effect": False,
            "controls_computer": False,
            "writes_files": False,
            "writes_database": False,
            "writes_memory": False,
            "writes_notes": False,
            "speaks": False,
            "completes_tasks": False,
        },
    }


def make_doctor_tool(
    store: MemoryStore,
    vault: ObsidianVault,
    config: JarvisConfig,
    storage_fallback: Callable[[], dict[str, Any] | None] | None = None,
    list_tools: Callable[[], list[Any]] | None = None,
):
    def jarvis_doctor(_: dict[str, Any]) -> ToolResult:
        project_root = Path(__file__).resolve().parents[2]
        dashboard_launcher = project_root / "launch_jarvis_v3_dashboard.py"
        project_root_display = safe_storage_text(project_root)
        dashboard_launcher_display = safe_storage_text(dashboard_launcher)
        status_host_config = status_host_config_from_env()
        status_port_config = status_port_config_from_env()
        status_host_configured = status_host_config.configured
        status_port_configured = status_port_config.configured
        model_provider = normalized_model_provider(config.model_provider)
        provider_valid = model_provider in MODEL_PROVIDERS
        openai_provider = model_provider == "openai"
        ollama_provider = model_provider == "ollama"
        ollama_destination = resolve_ollama_destination()
        provider_status = _enum_env_status("JARVIS_MODEL_PROVIDER", "ollama", MODEL_PROVIDERS)
        chat_reasoning = _enum_env_status(
            "JARVIS_CHAT_REASONING_EFFORT", "low" if openai_provider else "medium", REASONING_EFFORTS
        )
        planner_reasoning = _enum_env_status("JARVIS_PLANNER_REASONING_EFFORT", "low", REASONING_EFFORTS)
        planner_timeout_default = 12.0 if openai_provider else DEFAULT_MODEL_TIMEOUT_SECONDS
        chat_timeout_default = 60.0 if openai_provider else DEFAULT_CHAT_TIMEOUT_SECONDS
        planner_timeout_maximum = (
            MAX_OPENAI_MODEL_TIMEOUT_SECONDS if openai_provider else MAX_OLLAMA_MODEL_TIMEOUT_SECONDS
        )
        chat_timeout_maximum = (
            MAX_OPENAI_CHAT_TIMEOUT_SECONDS if openai_provider else MAX_OLLAMA_CHAT_TIMEOUT_SECONDS
        )
        planner_timeout = _runtime_timeout_status(
            "JARVIS_MODEL_TIMEOUT_SECONDS",
            planner_timeout_default,
            MIN_MODEL_TIMEOUT_SECONDS,
            planner_timeout_maximum,
        )
        chat_timeout = _runtime_timeout_status(
            "JARVIS_CHAT_TIMEOUT_SECONDS",
            chat_timeout_default,
            MIN_CHAT_TIMEOUT_SECONDS,
            chat_timeout_maximum,
        )
        chat_max_reply_tokens = _runtime_int_status(
            "JARVIS_CHAT_MAX_REPLY_TOKENS",
            DEFAULT_CHAT_MAX_REPLY_TOKENS,
            MIN_CHAT_MAX_REPLY_TOKENS,
            MAX_CHAT_MAX_REPLY_TOKENS,
        )
        chat_max_history_messages = _runtime_int_status(
            "JARVIS_CHAT_MAX_HISTORY_MESSAGES",
            DEFAULT_CHAT_MAX_HISTORY_MESSAGES,
            MIN_CHAT_MAX_HISTORY_MESSAGES,
            MAX_CHAT_MAX_HISTORY_MESSAGES,
        )
        smoke_timeout = _smoke_timeout_status()
        smoke_timeout_configured = smoke_timeout["configured"]
        model_planner_toggle = _model_planner_toggle_status()
        remote_compaction_toggle = _remote_compaction_toggle_status()
        remote_personal_context_toggle = _remote_personal_context_toggle_status()
        storage_fallback_env = _storage_fallback_status()
        telegram_bot_token = _secret_env_status("TELEGRAM_BOT_TOKEN", required=True)
        gmail_address = _secret_env_status("GMAIL_ADDRESS", required=True, kind="email")
        gmail_app_password = _secret_env_status("GMAIL_APP_PASSWORD", required=True)
        openai_api_key = _secret_env_status("OPENAI_API_KEY", required=openai_provider)
        oav_vision_reviewer = _local_command_env_status("JARVIS_OAV_VISION_REVIEWER_COMMAND")
        voice_whisper_model = _local_path_env_status("JARVIS_VOICE_WHISPER_MODEL_PATH", expected="file")
        voice_faster_whisper_model = _local_path_env_status("JARVIS_VOICE_FASTER_WHISPER_MODEL_PATH", expected="directory")
        fallback_model_alias = _model_alias_status("OLLAMA_MODEL", DEFAULT_MODEL_ALIAS, default_source="default")
        chat_model_fallback = config.chat_model if openai_provider else fallback_model_alias["effective_alias"]
        chat_model_default_source = "provider-default" if openai_provider else fallback_model_alias["source"]
        chat_model_alias = _model_alias_status(
            "JARVIS_CHAT_MODEL",
            chat_model_fallback,
            default_source=chat_model_default_source,
        )
        planner_model_fallback = config.planner_model if openai_provider else chat_model_alias["effective_alias"]
        planner_model_default_source = "provider-default" if openai_provider else chat_model_alias["source"]
        planner_model_alias = _model_alias_status(
            "JARVIS_PLANNER_MODEL",
            planner_model_fallback,
            default_source=planner_model_default_source,
        )
        ollama_no_cloud = ollama_local_only_policy(config.chat_model)
        data_dir = config.data_dir.expanduser()
        db_parent = config.db_path.expanduser().parent
        requirements = project_root / "requirements.txt"
        runtime_storage_fallback = storage_fallback() if storage_fallback else None
        storage, _storage_diagnostics_source = _configured_storage_diagnostics(config, runtime_storage_fallback)
        storage = dict(storage)
        for key in _STORAGE_DIAGNOSTIC_BOOL_KEYS:
            storage[key] = _metadata_bool(storage.get(key))
        storage_runtime_fallback_active = bool(runtime_storage_fallback)
        storage_issues = list(storage.get("issues") or [])
        if storage_runtime_fallback_active:
            storage_issues.append("runtime is using workspace-local fallback storage")
        storage_configured_status = str(storage.get("status", "unknown"))
        storage_configured_available = _metadata_bool(storage.get("available"))
        storage["configured_status"] = storage_configured_status
        storage["configured_available"] = storage_configured_available
        storage["status"] = "needs attention" if storage_issues else storage_configured_status
        storage["available"] = bool(storage_configured_available or storage_runtime_fallback_active)
        storage["issues"] = storage_issues
        storage["runtime_fallback_active"] = storage_runtime_fallback_active
        storage["runtime_fallback_reason"] = str((runtime_storage_fallback or {}).get("reason") or "")
        storage["runtime_fallback_exception_type"] = str((runtime_storage_fallback or {}).get("exception_type") or "")
        storage["runtime_fallback_db_path_display"] = str((runtime_storage_fallback or {}).get("db_path_display") or "")
        storage["runtime_fallback_vault_path_display"] = str((runtime_storage_fallback or {}).get("vault_path_display") or "")
        storage["ready_for_completion_claim"] = bool(storage_configured_available and not storage_runtime_fallback_active)
        storage["recovery_required"] = bool(storage_issues)
        storage["recovery_reason"] = "; ".join(storage_issues)
        storage["recovery_check_tool_command"] = STORAGE_RECOVERY_CHECK_COMMAND
        storage["recovery_check_command"] = storage.get("recovery_check_command") or BOOTSTRAP_CHECK_COMMAND
        storage["recovery_check_api"] = storage.get("recovery_check_api") or STORAGE_RECOVERY_CHECK_API
        storage["recovery_command"] = storage.get("recovery_command") or BOOTSTRAP_WRITE_COMMAND
        if storage_runtime_fallback_active and storage_configured_available:
            storage["recovery_mode"] = "restart_runtime_to_configured_storage"
            storage["recovery_next_operator_action"] = (
                "restart or reload Jarvis with the configured durable storage envs, then run `storage status`"
            )
            storage["recovery_restart_required"] = True
        elif storage_runtime_fallback_active:
            storage["recovery_mode"] = "repair_configured_storage_then_restart_runtime"
            storage["recovery_next_operator_action"] = (
                "review the storage recovery plan, point Jarvis at writable durable storage, run the no-write storage check, then restart or reload Jarvis"
            )
            storage["recovery_restart_required"] = True
        elif not storage_configured_available:
            storage["recovery_mode"] = "repair_configured_storage"
            storage["recovery_next_operator_action"] = (
                "review the storage recovery plan, point Jarvis at writable durable storage, and run the no-write storage check"
            )
            storage["recovery_restart_required"] = False
        else:
            storage["recovery_mode"] = "none"
            storage["recovery_next_operator_action"] = ""
            storage["recovery_restart_required"] = False
        storage["readiness_blocks_completion_claim"] = not storage["ready_for_completion_claim"]
        storage["readiness_blocker"] = (
            storage["recovery_reason"]
            if storage["readiness_blocks_completion_claim"]
            else ""
        )
        if storage["readiness_blocks_completion_claim"] and storage_runtime_fallback_active and storage_configured_available:
            storage["readiness_next_commands"] = [
                "storage status",
                storage["recovery_check_tool_command"],
                "storage status",
            ]
        elif storage["readiness_blocks_completion_claim"]:
            storage["readiness_next_commands"] = [
                "storage status",
                STORAGE_RECOVERY_PLAN_COMMAND,
                storage["recovery_check_tool_command"],
                storage["recovery_check_command"],
                storage["recovery_command"],
            ]
        else:
            storage["readiness_next_commands"] = []
        storage["readiness_next_command_count"] = len(storage["readiness_next_commands"])
        storage["readiness_next_proof_command"] = (
            storage["readiness_next_commands"][0]
            if storage["readiness_next_commands"]
            else ""
        )
        storage["readiness_next_required_command"] = storage["readiness_next_proof_command"]

        checks: list[tuple[str, bool, str]] = []
        checks.append(("data directory", data_dir.exists() or data_dir.parent.exists(), safe_storage_text(data_dir)))
        checks.append(("database parent", db_parent.exists() or db_parent.parent.exists(), safe_storage_text(db_parent)))
        checks.append(("database parent writable", _metadata_bool(storage["db_parent_writable"]), str(storage["db_parent"])))
        checks.append(("database file writable", _metadata_bool(storage["db_file_writable"]), str(storage["db_path"])))
        checks.append(("Obsidian root", vault.root_path.exists(), safe_storage_text(vault.root_path)))
        checks.append(("Jarvis note vault writable", _metadata_bool(storage["obsidian_vault_writable"]), str(storage["obsidian_vault"])))
        checks.append(("Jarvis note root writable", _metadata_bool(storage["obsidian_root_writable"]), str(storage["obsidian_root_path"])))
        checks.append(
            (
                "runtime storage fallback",
                not storage_runtime_fallback_active,
                "inactive"
                if not storage_runtime_fallback_active
                else "active; using workspace-local fallback storage",
            )
        )
        checks.append(("model planner", True, "on" if config.use_model_planner else "off"))
        checks.append(
            (
                "remote interactive personal context",
                bool(remote_personal_context_toggle["valid"]),
                "enabled; stored profile, preferences, memory, skills, and prior history may be sent to OpenAI"
                if openai_provider and remote_personal_context_toggle["effective_enabled"]
                else "disabled by default; stored personal context stays local while the current message may still be sent to OpenAI"
                if openai_provider
                else "Ollama route selected; stored-context use follows the explicit consent policy; execution locality unknown"
                if ollama_provider
                else "not evaluated; model provider invalid; execution locality unknown",
            )
        )
        checks.append(
            (
                "remote conversation compaction",
                bool(remote_compaction_toggle["valid"]),
                "enabled; scheduled old conversation batches may be sent to OpenAI"
                if openai_provider and remote_compaction_toggle["effective_enabled"]
                else "disabled by default; no old conversation batches are sent to OpenAI"
                if openai_provider
                else "Ollama route selected; remote-history behavior is not inferred from the provider label; execution locality unknown"
                if ollama_provider
                else "not evaluated; model provider invalid; execution locality unknown",
            )
        )
        checks.append(
            (
                "model provider",
                provider_valid,
                model_provider if provider_valid else "invalid; set JARVIS_MODEL_PROVIDER to ollama or openai",
            )
        )
        checks.append(
            (
                "model provider env",
                bool(provider_status["valid"]),
                f"using {provider_status['effective_value']}"
                if provider_status["valid"]
                else f"invalid; using {provider_status['fallback_value']} (value hidden)",
            )
        )
        checks.append(
            (
                "chat reasoning effort",
                bool(chat_reasoning["valid"]),
                f"using {chat_reasoning['effective_value']}"
                if chat_reasoning["valid"]
                else f"invalid; using {chat_reasoning['fallback_value']} (value hidden)",
            )
        )
        checks.append(
            (
                "planner reasoning effort",
                bool(planner_reasoning["valid"]),
                f"using {planner_reasoning['effective_value']}"
                if planner_reasoning["valid"]
                else f"invalid; using {planner_reasoning['fallback_value']} (value hidden)",
            )
        )
        checks.append(("chat model", True, config.chat_model))
        checks.append(("planner model", True, config.planner_model))

        module_checks = [("pyautogui", "pyautogui"), ("PIL", "pillow")]
        if ollama_provider:
            checks.append(
                (
                    "ollama python package",
                    True,
                    "not required; stdlib loopback HTTP adapter selected",
                )
            )
        elif openai_provider:
            checks.append(("ollama python package", True, "not required; OpenAI provider selected"))
        else:
            checks.append(("ollama python package", True, "not evaluated; model provider invalid"))
        for module, label in module_checks:
            checks.append((label, _module_ok(module), "python import"))
        google_dependency_specs = [
            ("google.oauth2.credentials", "google-auth"),
            ("google_auth_oauthlib.flow", "google-auth-oauthlib"),
            ("google_auth_httplib2", "google-auth-httplib2"),
            ("googleapiclient.discovery", "google-api-python-client"),
        ]
        google_dependency_status = {label: _module_ok(module) for module, label in google_dependency_specs}
        for label, ok in google_dependency_status.items():
            checks.append((f"Google connector dependency ({label})", ok, "python import only; credentials not read"))

        ollama_probe_diagnostic = "ollama_probe_not_attempted"
        ollama_probe_model_count = 0
        if openai_provider:
            ollama_cli_ok = False
            ollama_live_access_probed = False
            checks.append(("ollama server/cli", True, "not required; OpenAI provider selected"))
            checks.append(
                (
                    "OpenAI API key",
                    bool(openai_api_key["configured"] and openai_api_key["valid"]),
                    "configured; value hidden; live access not probed"
                    if openai_api_key["configured"] and openai_api_key["valid"]
                    else "set OPENAI_API_KEY locally, then run `model routing status`",
                )
            )
            checks.append(
                (
                    "OpenAI safety identifier",
                    True,
                    "stable single-owner pseudonym; value hidden; not derived from personal data or the API key",
                )
            )
            checks.append(
                (
                    "OpenAI retention boundary",
                    True,
                    "store=false request flag; account retention controls not checked; default "
                    "abuse-monitoring logs may retain prompts/responses for up to 30 days unless "
                    "approved controls apply",
                )
            )
        elif ollama_provider:
            ollama_live_access_probed = ollama_destination.allowed
            if not ollama_destination.allowed:
                ollama_cli_ok = False
                ollama_cli_output = "probe skipped; configured destination is invalid or non-loopback"
            else:
                probe_ok, model_names, raw_diagnostic, _error_type = probe_ollama_models(
                    timeout_seconds=5.0,
                )
                ollama_cli_ok = bool(probe_ok)
                ollama_probe_model_count = len(model_names) if isinstance(model_names, list) else 0
                ollama_probe_diagnostic = _safe_ollama_probe_diagnostic(raw_diagnostic, ok=ollama_cli_ok)
                ollama_cli_output = (
                    "reachable through validated loopback HTTP API; model aliases hidden"
                    if ollama_cli_ok
                    else "unavailable through validated loopback HTTP API; probe details hidden"
                )
            checks.append(("ollama loopback API", ollama_cli_ok, ollama_cli_output))
        else:
            ollama_cli_ok = False
            ollama_live_access_probed = False
            checks.append(("ollama loopback API", True, "not evaluated; model provider invalid; no probe attempted"))
            checks.append(("OpenAI API configuration", True, "not evaluated; model provider invalid; no probe attempted"))
        checks.append(
            (
                "Ollama destination policy",
                bool(not ollama_provider or ollama_destination.allowed),
                (
                    "not required; OpenAI provider selected; "
                    if openai_provider
                    else "not evaluated; model provider invalid; "
                    if not provider_valid
                    else ""
                )
                + _ollama_destination_policy_detail(ollama_destination),
            )
        )
        checks.append(
            (
                "Ollama personalized context policy",
                bool(not ollama_provider or ollama_no_cloud.personal_context_allowed),
                _ollama_no_cloud_policy_detail(
                    ollama_no_cloud,
                    required=ollama_provider,
                    provider_valid=provider_valid,
                ),
            )
        )

        for command in [["osascript", "-e", "return 1"], ["open", "--help"], ["say", "-v", "?"]]:
            ok, output = _command_ok(command)
            checks.append((command[0], ok or command[0] == "open", output[:160]))

        pbpaste_ok, pbpaste_detail = _command_available("pbpaste")
        checks.append(("pbpaste", pbpaste_ok, pbpaste_detail + "; clipboard was not read"))

        sqlite_store_exception_type = ""
        try:
            store.list_sessions(1)
            checks.append(("sqlite store", True, safe_storage_text(config.db_path)))
        except Exception as exc:
            sqlite_store_exception_type = type(exc).__name__
            checks.append(
                (
                    "sqlite store",
                    False,
                    f"unavailable; set JARVIS_DATA_DIR and JARVIS_DB_PATH to writable local paths, run `{BOOTSTRAP_CHECK_COMMAND}`, then retry",
                )
            )

        pending_approvals = store.list_pending_approvals(limit=20)
        open_tasks = store.list_tasks(status="open", limit=20)
        active_goals = store.list_goals(status="active", limit=20)
        recent_runs = store.recent_tool_runs(limit=40)
        readable_recent_runs = [row for row in recent_runs if _row_text(row, "tool_name")]
        unreadable_recent_run_rows = len(recent_runs) - len(readable_recent_runs)
        recent_failures, recent_approval_held_runs = _recent_tool_run_attention_buckets(readable_recent_runs)
        verification_runs = [
            row
            for row in readable_recent_runs
            if _row_text(row, "tool_name")
            in {
                "verification_receipt",
                "runtime_trace_receipt",
                "execution_audit_gate",
                "execution_recovery_packet",
                "after_action_learning_packet",
                "execution_health_report",
                "execution_acceptance_gate",
                "completion_audit_packet",
                "task_completion_packet",
                "evidence_ledger",
            }
        ]
        learning_runs = [
            row for row in readable_recent_runs if _row_text(row, "tool_name") == "after_action_learning_packet"
        ]
        next_audit_command = _first_execution_health_command(readable_recent_runs)
        recovery_closure = _execution_health_recovery_closure_snapshot(store, readable_recent_runs)
        learning_debt = _execution_learning_debt_snapshot(readable_recent_runs)
        approval_held_review_commands = _first_approval_held_review_commands(recent_approval_held_runs)
        approval_held_review_next_command = approval_held_review_commands[0] if approval_held_review_commands else ""
        approval_held_review_approval_id = (
            approval_held_review_next_command.removeprefix("approval readiness ")
            if approval_held_review_next_command.startswith("approval readiness ")
            else ""
        )
        audit_readability_review_commands = (
            ["storage status", "recent tool runs", "execution health report"]
            if unreadable_recent_run_rows
            else []
        )
        completion_blockers: list[str] = []
        if unreadable_recent_run_rows:
            completion_blockers.append(f"{unreadable_recent_run_rows} unreadable recent tool run row(s)")
        if pending_approvals:
            completion_blockers.append(f"{len(pending_approvals)} pending approval(s)")
        if open_tasks:
            completion_blockers.append(f"{len(open_tasks)} open task(s)")
        if active_goals:
            completion_blockers.append(f"{len(active_goals)} active goal(s)")
        if recent_failures:
            completion_blockers.append(f"{len(recent_failures)} recent failed/blocked run(s)")
        if recent_approval_held_runs:
            completion_blockers.append(f"{len(recent_approval_held_runs)} recent approval-held run(s) need approval review")
        if not verification_runs:
            completion_blockers.append("no recent verification/audit packet")
        if not learning_runs:
            completion_blockers.append("no recent after-action learning packet")
        if recovery_closure["blocks_completion_claim"]:
            completion_blockers.append(
                "execution health recovery closure incomplete: "
                + ", ".join(recovery_closure["missing"] or [str(recovery_closure["state"])])
            )
        if learning_debt["blocks_completion_claim"]:
            completion_blockers.append(
                "execution learning debt incomplete: "
                + ", ".join(learning_debt["missing"] or [str(learning_debt["state"])])
            )
        if not storage["ready_for_completion_claim"]:
            completion_blockers.append(
                "durable storage recovery required: "
                + (storage["recovery_reason"] or "configured storage is not ready for completion claim")
            )
        completion_claim_state = "BLOCKED" if completion_blockers else "READY_FOR_HUMAN_REVIEW"
        agi_handoff = _next_agi_gate_handoff(list_tools)
        agi_target_integrity = agi_handoff["target_file_integrity"]
        agi_build_packet_ready = bool(
            agi_target_integrity["all_exist"]
            and agi_handoff["focused_verification_commands"]
            and agi_handoff["acceptance_checks"]
        )
        completion_proof_queue: list[str] = []
        if storage["recovery_required"] or not storage["ready_for_completion_claim"]:
            for command in storage["readiness_next_commands"]:
                if command:
                    completion_proof_queue.append(command)
        for command in audit_readability_review_commands:
            if command not in completion_proof_queue:
                completion_proof_queue.append(command)
        if pending_approvals:
            approval_id = _first_positive_int_text(pending_approvals)
            approval_commands = (
                [
                    f"approval readiness {approval_id}",
                    f"approval packet {approval_id}",
                    f"approval chain proof {approval_id}",
                    f"verification receipt <approved run id from approval chain proof {approval_id}>",
                ]
                if approval_id
                else ["pending approvals", "approval readiness latest", "approval packet latest", "approval chain proof latest"]
            )
            for command in approval_commands:
                if command not in completion_proof_queue:
                    completion_proof_queue.append(command)
        for command in approval_held_review_commands:
            if command not in completion_proof_queue:
                completion_proof_queue.append(command)
        if open_tasks:
            task_id = _first_positive_int_text(open_tasks)
            command = f"task completion packet {task_id}" if task_id else "task board"
            if command not in completion_proof_queue:
                completion_proof_queue.append(command)
        for command in recovery_closure["required_commands"]:
            if command not in completion_proof_queue:
                completion_proof_queue.append(command)
        learning_actionable_commands = list(learning_debt["actionable_required_commands"] or learning_debt["required_commands"])
        for command in learning_actionable_commands:
            if command not in completion_proof_queue:
                completion_proof_queue.append(command)
        for command in ["harness completion", "completion audit", "evidence ledger", "completion claim gate"]:
            if command not in completion_proof_queue:
                completion_proof_queue.append(command)
        lines = ["Jarvis doctor:"]
        for label, ok, detail in checks:
            mark = "ok" if ok else "needs attention"
            suffix = f" | {detail}" if detail else ""
            lines.append(f"- {label}: {mark}{suffix}")

        missing = [label for label, ok, _ in checks if not ok]
        lines.append("")
        lines.append("Next actions:")
        if storage["readiness_blocks_completion_claim"]:
            lines.append(
                "- Restore durable storage first: run "
                f"`{storage['readiness_next_proof_command'] or 'storage status'}` "
                "and follow the storage recovery ladder before relying on autonomy or completion claims."
            )
        if _module_ok("pyautogui"):
            lines.append("- Computer-control Python dependency is installed.")
        else:
            lines.append(
                f"- Install optional computer-control deps: {V3_PROJECT_PYTHON} -m pip install -r "
                + safe_storage_text(requirements)
            )
        if not provider_valid:
            lines.append("- Set JARVIS_MODEL_PROVIDER to ollama or openai, then rerun `jarvis doctor`; no provider connectivity probe was attempted.")
        elif openai_provider and openai_api_key["configured"] and openai_api_key["valid"]:
            lines.append("- OpenAI provider configuration is present; run `model routing status` before an operator-triggered paid model call.")
        elif openai_provider:
            lines.append("- Set OPENAI_API_KEY locally, then run `model routing status`; Jarvis doctor makes no paid model call.")
        elif not ollama_destination.allowed:
            lines.append("- Set OLLAMA_HOST to an HTTP(S) loopback destination, then rerun `jarvis doctor`; the rejected value is not displayed.")
        elif not ollama_no_cloud.personal_context_allowed:
            lines.append(
                "- Configure the Ollama daemon with OLLAMA_NO_CLOUD=1 and a non-cloud model alias, "
                "restart Ollama, then rerun `jarvis doctor`; current-message-only routing may remain "
                "usable, but stored context stays withheld."
            )
        elif ollama_cli_ok:
            lines.append("- Ollama is reachable. Chat/model planning should work if the configured model is pulled.")
        else:
            lines.append("- Start Ollama, then run: ollama pull " + config.chat_model)
        if missing:
            lines.append("- Review items marked 'needs attention' before relying on full autonomy.")
        else:
            lines.append("- Core setup looks ready.")
        base_next_commands = [
            "setup check",
            "prototype readiness",
            "readiness report",
            next_audit_command,
            *list(recovery_closure["required_commands"])[:4],
            "agi gates",
            agi_handoff["next_command"],
            "completion claim gate",
            "computer control status",
            "voice setup check",
        ]
        command_prefix = completion_proof_queue if completion_claim_state == "BLOCKED" else []
        next_commands = _dedupe_commands([*command_prefix, *base_next_commands])
        doctor_handoff = _doctor_handoff(
            missing=missing,
            checks=checks,
            next_commands=next_commands,
            pending_approvals=pending_approvals,
            open_tasks=open_tasks,
            active_goals=active_goals,
            recent_runs=recent_runs,
            recent_failures=recent_failures,
            recent_approval_held_runs=recent_approval_held_runs,
            verification_runs=verification_runs,
            learning_runs=learning_runs,
            next_audit_command=next_audit_command,
            recovery_closure=recovery_closure,
            learning_debt=learning_debt,
            audit_readability_review_commands=audit_readability_review_commands,
            completion_claim_state=completion_claim_state,
            completion_blockers=completion_blockers,
            completion_proof_queue=completion_proof_queue,
            agi_handoff=agi_handoff,
            agi_build_packet_ready=agi_build_packet_ready,
            storage=storage,
            sqlite_store_exception_type=sqlite_store_exception_type,
        )
        lines.extend(
            [
                "",
                "Project discovery:",
                "- active project: Jarvis V3",
                "- this is the V3 project folder, not `jarvis-ollama`; the Python package remains `jarvis_v2` during compatibility migration",
                f"- project root: {project_root_display}",
                f"- dashboard launcher: {dashboard_launcher_display}",
                f"- dashboard command: `{V3_DASHBOARD_COMMAND}`",
                f"- ask-Jarvis command: `{V3_DASHBOARD_INFO_COMMAND}`",
                f"- equivalent module command: `{V3_DASHBOARD_MODULE_COMMAND}`",
                f"- Telegram bot token env (TELEGRAM_BOT_TOKEN): {'set' if telegram_bot_token['configured'] else 'not set'} (value hidden)",
                "- Telegram bot token validation: "
                + (
                    "ok"
                    if telegram_bot_token["valid"] and telegram_bot_token["configured"]
                    else "missing; set TELEGRAM_BOT_TOKEN"
                    if not telegram_bot_token["configured"]
                    else "invalid"
                )
                + " (value hidden)",
                f"- Gmail address env (GMAIL_ADDRESS): {'set' if gmail_address['configured'] else 'not set'} (value hidden)",
                "- Gmail address validation: "
                + (
                    "ok"
                    if gmail_address["valid"] and gmail_address["configured"]
                    else "missing; set GMAIL_ADDRESS"
                    if not gmail_address["configured"]
                    else "invalid"
                )
                + " (value hidden)",
                f"- Gmail app password env (GMAIL_APP_PASSWORD): {'set' if gmail_app_password['configured'] else 'not set'} (value hidden)",
                "- Gmail app password validation: "
                + (
                    "ok"
                    if gmail_app_password["valid"] and gmail_app_password["configured"]
                    else "missing; set GMAIL_APP_PASSWORD"
                    if not gmail_app_password["configured"]
                    else "invalid"
                )
                + " (value hidden)",
                f"- local OAV vision reviewer command env (JARVIS_OAV_VISION_REVIEWER_COMMAND): {'set' if oav_vision_reviewer['configured'] else 'not set'} (value hidden)",
                "- local OAV vision reviewer command validation: "
                + (
                    "not set; vision model review held until configured"
                    if not oav_vision_reviewer["configured"]
                    else "ok; executable resolved without running it"
                    if oav_vision_reviewer["valid"]
                    else "invalid; executable not resolvable"
                    if not oav_vision_reviewer["parse_error"]
                    else "invalid; command could not be parsed"
                )
                + " (value hidden)",
                f"- local Whisper model path env (JARVIS_VOICE_WHISPER_MODEL_PATH): {'set' if voice_whisper_model['configured'] else 'not set'} (value hidden)",
                "- local Whisper model path validation: "
                + (
                    "not set; audio-file transcription held until configured"
                    if not voice_whisper_model["configured"]
                    else "ok; model file exists (not loaded)"
                    if voice_whisper_model["valid"]
                    else "invalid; path is a directory"
                    if voice_whisper_model["is_dir"]
                    else "invalid; parent folder is missing"
                    if not voice_whisper_model["parent_exists"]
                    else "invalid; model file not found"
                )
                + " (value hidden)",
                f"- local faster-whisper model path env (JARVIS_VOICE_FASTER_WHISPER_MODEL_PATH): {'set' if voice_faster_whisper_model['configured'] else 'not set'} (value hidden)",
                "- local faster-whisper model path validation: "
                + (
                    "not set; faster-whisper audio-file transcription held until configured"
                    if not voice_faster_whisper_model["configured"]
                    else "ok; model directory exists (not loaded)"
                    if voice_faster_whisper_model["valid"]
                    else "invalid; target is a file"
                    if voice_faster_whisper_model["is_file"]
                    else "invalid; parent folder is missing"
                    if not voice_faster_whisper_model["parent_exists"]
                    else "invalid; model directory not found"
                )
                + " (value hidden)",
                f"- fallback Ollama model alias env (OLLAMA_MODEL): {'set' if fallback_model_alias['configured'] else 'not set'} (value hidden)",
                "- fallback Ollama model alias validation: "
                + (
                    "not required; OpenAI provider selected"
                    if openai_provider
                    else f"not set; using {DEFAULT_MODEL_ALIAS}"
                    if not fallback_model_alias["configured"]
                    else "ok"
                    if fallback_model_alias["valid"]
                    else f"invalid; using {DEFAULT_MODEL_ALIAS}"
                )
                + " (value hidden)",
                f"- chat model alias env (JARVIS_CHAT_MODEL): {'set' if chat_model_alias['configured'] else 'not set'} (value hidden)",
                "- chat model alias validation: "
                + (
                    "not set; using OpenAI provider default"
                    if openai_provider and not chat_model_alias["configured"]
                    else "not set; using fallback Ollama model alias"
                    if not chat_model_alias["configured"]
                    else "ok"
                    if chat_model_alias["valid"]
                    else "invalid; using fallback model alias"
                )
                + " (value hidden)",
                f"- planner model alias env (JARVIS_PLANNER_MODEL): {'set' if planner_model_alias['configured'] else 'not set'} (value hidden)",
                "- planner model alias validation: "
                + (
                    "not set; using OpenAI provider default"
                    if openai_provider and not planner_model_alias["configured"]
                    else "not set; using chat model alias"
                    if not planner_model_alias["configured"]
                    else "ok"
                    if planner_model_alias["valid"]
                    else "invalid; using chat model alias"
                )
                + " (value hidden)",
                f"- planner timeout env (JARVIS_MODEL_TIMEOUT_SECONDS): {'set' if planner_timeout['configured'] else 'not set'} (value hidden)",
                "- planner timeout validation: "
                + _runtime_timeout_validation_detail(planner_timeout, planner_timeout_default)
                + " (value hidden)",
                f"- chat timeout env (JARVIS_CHAT_TIMEOUT_SECONDS): {'set' if chat_timeout['configured'] else 'not set'} (value hidden)",
                "- chat timeout validation: "
                + _runtime_timeout_validation_detail(chat_timeout, chat_timeout_default)
                + " (value hidden)",
                f"- chat reply token cap env (JARVIS_CHAT_MAX_REPLY_TOKENS): {'set' if chat_max_reply_tokens['configured'] else 'not set'} (value hidden)",
                "- chat reply token cap validation: "
                + _runtime_int_validation_detail(
                    chat_max_reply_tokens,
                    DEFAULT_CHAT_MAX_REPLY_TOKENS,
                    MAX_CHAT_MAX_REPLY_TOKENS,
                )
                + " (value hidden)",
                f"- chat history window env (JARVIS_CHAT_MAX_HISTORY_MESSAGES): {'set' if chat_max_history_messages['configured'] else 'not set'} (value hidden)",
                "- chat history window validation: "
                + _runtime_int_validation_detail(
                    chat_max_history_messages,
                    DEFAULT_CHAT_MAX_HISTORY_MESSAGES,
                    MAX_CHAT_MAX_HISTORY_MESSAGES,
                )
                + " (value hidden)",
                f"- smoke module timeout env (JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS): {'set' if smoke_timeout_configured else 'not set'} (value hidden)",
                "- smoke module timeout validation: "
                + (
                    "ok"
                    if smoke_timeout["valid"]
                    else f"invalid; using default {DEFAULT_SMOKE_MODULE_TIMEOUT_SECONDS:g}s"
                )
                + " (value hidden)",
                f"- model planner toggle env (JARVIS_USE_MODEL_PLANNER): {'set' if model_planner_toggle['configured'] else 'not set'} (value hidden)",
                "- model planner toggle validation: "
                + (
                    "not set; enabled by default"
                    if not model_planner_toggle["configured"]
                    else "ok; enabled"
                    if model_planner_toggle["valid"] and model_planner_toggle["effective_enabled"]
                    else "ok; disabled"
                    if model_planner_toggle["valid"]
                    else "invalid; enabled by default"
                )
                + " (value hidden)",
                f"- remote conversation compaction env (JARVIS_ALLOW_REMOTE_COMPACTION): {'set' if remote_compaction_toggle['configured'] else 'not set'} (value hidden)",
                "- remote conversation compaction validation: "
                + (
                    "not set; disabled by default"
                    if not remote_compaction_toggle["configured"]
                    else "ok; enabled"
                    if remote_compaction_toggle["valid"] and remote_compaction_toggle["effective_enabled"]
                    else "ok; disabled"
                    if remote_compaction_toggle["valid"]
                    else "invalid; disabled by default"
                )
                + " (value hidden)",
                f"- remote personal context env (JARVIS_ALLOW_REMOTE_PERSONAL_CONTEXT): {'set' if remote_personal_context_toggle['configured'] else 'not set'} (value hidden)",
                "- remote personal context validation: "
                + (
                    "not set; stored personal context stays local by default"
                    if not remote_personal_context_toggle["configured"]
                    else "ok; enabled"
                    if remote_personal_context_toggle["valid"] and remote_personal_context_toggle["effective_enabled"]
                    else "ok; disabled"
                    if remote_personal_context_toggle["valid"]
                    else "invalid; stored personal context stays local by default"
                )
                + " (value hidden)",
                f"- storage fallback disable env (JARVIS_DISABLE_STORAGE_FALLBACK): {'set' if storage_fallback_env['disable_configured'] else 'not set'} (value hidden)",
                "- storage fallback disable validation: "
                + (
                    "not set; fallback enabled by default"
                    if not storage_fallback_env["disable_configured"]
                    else "ok; fallback disabled"
                    if storage_fallback_env["disable_valid"] and storage_fallback_env["disabled"]
                    else "ok; fallback enabled"
                    if storage_fallback_env["disable_valid"]
                    else "invalid; fallback enabled by default"
                )
                + " (value hidden)",
                f"- dashboard host env (JARVIS_STATUS_HOST): {'set' if status_host_configured else 'not set'} (value hidden)",
                "- dashboard host validation: "
                + ("ok" if status_host_config.valid else f"invalid; using default {DEFAULT_STATUS_HOST}")
                + " (value hidden)",
                f"- dashboard port env (JARVIS_STATUS_PORT): {'set' if status_port_configured else 'not set'} (value hidden)",
                "- dashboard port validation: "
                + ("ok" if status_port_config.valid else f"invalid; using default {DEFAULT_STATUS_PORT}")
                + " (value hidden)",
                "",
                "Storage diagnostics:",
                f"- storage readiness: {storage['status']}",
                f"- configured storage status: {storage['configured_status']}",
                f"- runtime fallback active: {'yes' if storage_runtime_fallback_active else 'no'}",
                f"- ready for completion claim: {'yes' if storage['ready_for_completion_claim'] else 'no'}",
                f"- data dir: {storage['data_dir']}",
                f"- database: {storage['db_path']}",
                f"- database parent exists: {'yes' if storage['db_parent_exists'] else 'no'}",
                f"- database parent writable: {'yes' if storage['db_parent_writable'] else 'no'}",
                f"- database file exists: {'yes' if storage['db_exists'] else 'no'}",
                f"- database file writable: {'yes' if storage['db_file_writable'] else 'no'}",
                f"- Jarvis note vault: {storage['obsidian_vault']}",
                f"- Jarvis note root: {storage['obsidian_root_path']}",
                f"- Jarvis note vault writable: {'yes' if storage['obsidian_vault_writable'] else 'no'}",
                f"- Jarvis note root writable: {'yes' if storage['obsidian_root_writable'] else 'no'}",
                f"- workspace-local notes: {'yes' if storage['workspace_local_notes'] else 'no'}",
                f"- metadata-only check: {'yes' if storage['metadata_only'] else 'no'}",
                f"- recovery required before completion claim: {'yes' if storage['recovery_required'] else 'no'}",
                f"- recovery reason: {storage['recovery_reason'] or 'none'}",
                f"- recovery mode: {storage['recovery_mode']}",
                f"- next operator action: {storage['recovery_next_operator_action'] or 'none'}",
                f"- restart required: {'yes' if storage['recovery_restart_required'] else 'no'}",
                f"- storage readiness blocks completion claim: {'yes' if storage['readiness_blocks_completion_claim'] else 'no'}",
                f"- storage readiness blocker: {storage['readiness_blocker'] or 'none'}",
                f"- storage readiness next required: `{storage['readiness_next_required_command']}`" if storage["readiness_next_required_command"] else "- storage readiness next required: none",
                f"- storage readiness queue: {', '.join(f'`{command}`' for command in storage['readiness_next_commands']) if storage['readiness_next_commands'] else 'none'}",
                f"- native recovery check: `{storage.get('recovery_check_tool_command', '')}`",
                f"- shell recovery check: `{storage.get('recovery_check_command', '')}`",
                f"- recovery write: `{storage.get('recovery_command', '')}` only after the check reports configured storage ready.",
                "",
                "Harness readiness diagnostics:",
                f"- pending approvals: {len(pending_approvals)}",
                f"- open tasks: {len(open_tasks)}",
                f"- active goals: {len(active_goals)}",
                f"- recent tool runs inspected: {len(recent_runs)}",
                f"- unreadable recent tool-run rows: {unreadable_recent_run_rows}",
                "- audit readability review: "
                + (
                    ", ".join(f"`{command}`" for command in audit_readability_review_commands)
                    if audit_readability_review_commands
                    else "none"
                ),
                f"- recent failed/blocked runs: {len(recent_failures)}",
                f"- recent approval-held runs: {len(recent_approval_held_runs)}",
                "- approval-held review: "
                + (
                    ", ".join(f"`{command}`" for command in approval_held_review_commands)
                    if approval_held_review_commands
                    else "inspect `recent tool runs` or `pending approvals`"
                ),
                f"- recent verification/audit packets: {len(verification_runs)}",
                f"- recent after-action learning packets: {len(learning_runs)}",
                f"- next audit command: `{next_audit_command}`",
                f"- recovery closure state: {recovery_closure['state']}",
                f"- recovery closure ready to retry: {'yes' if recovery_closure['ready_to_retry'] else 'no'}",
                f"- recovery closure target: #{recovery_closure['target_run_id']} `{recovery_closure['target_tool_name']}`" if recovery_closure["target_run_id"] is not None else "- recovery closure target: none",
                f"- recovery closure missing: {', '.join(recovery_closure['missing']) if recovery_closure['missing'] else 'none'}",
                f"- recovery closure next required: `{recovery_closure['next_required_command']}`" if recovery_closure["next_required_command"] else "- recovery closure next required: none",
                f"- recovery closure proof queue: {', '.join(f'`{command}`' for command in recovery_closure['required_commands']) if recovery_closure['required_commands'] else 'none'}",
                f"- execution learning debt state: {learning_debt['state']}",
                f"- execution learning missing: {', '.join(learning_debt['missing']) if learning_debt['missing'] else 'none'}",
                f"- execution learning next evidence: `{learning_debt['next_evidence_command']}`" if learning_debt["next_evidence_command"] else "- execution learning next evidence: none",
                f"- execution learning next required: `{learning_debt['next_required_command']}`" if learning_debt["next_required_command"] else "- execution learning next required: none",
                f"- execution learning proof queue: {', '.join(f'`{command}`' for command in learning_debt['required_commands']) if learning_debt['required_commands'] else 'none'}",
                f"- completion claim state: {completion_claim_state}",
                f"- completion proof queue: {', '.join(f'`{command}`' for command in completion_proof_queue) if completion_proof_queue else 'none'}",
            ]
        )
        if completion_blockers:
            lines.append("- completion blockers: " + "; ".join(completion_blockers))
        else:
            lines.append("- completion blockers: none found by this diagnostic snapshot")
        lines.append("")
        lines.extend(
            [
                "AGI harness completion handoff:",
                f"- next AGI gate: {agi_handoff['gate']}",
                f"- selection source: {agi_handoff['selection_source']}",
                f"- selection note: {agi_handoff['selection_reason']}",
                f"- build target: {agi_handoff['build_target'] or 'use the selected AGI gate gap'}",
                f"- next build command: `{agi_handoff['next_command']}`",
                f"- target file check: {agi_handoff['target_file_integrity']['status']}",
                f"- target files checked: {agi_handoff['target_file_integrity']['checked']}",
                f"- real-execution gaps still tracked: {agi_handoff['real_execution_gap_count']}",
                f"- selected gate real-execution gap: {agi_handoff['selected_real_execution_gap'] or 'none'}",
                "- missing target files: "
                + (
                    ", ".join(agi_handoff["target_file_integrity"]["missing"])
                    if agi_handoff["target_file_integrity"]["missing"]
                    else "none"
                ),
                "- evidence closure commands: "
                + ", ".join(f"`{command}`" for command in agi_handoff["closure_commands"]),
                "- focused verification: "
                + (
                    ", ".join(f"`{command}`" for command in agi_handoff["focused_verification_commands"])
                    if agi_handoff["focused_verification_commands"]
                    else "none configured"
                ),
            ]
        )
        lines.append("")
        lines.append("Safe next commands:")
        lines.extend(f"- `{command}`" for command in next_commands)
        lines.append("")
        lines.append("Boundary:")
        lines.append("- This doctor check is diagnostic only. It does not read clipboard contents, observe the screen, approve requests, run Jarvis tools, write files, call models, or change external state.")

        return ToolResult(
            "jarvis_doctor",
            True,
            "\n".join(lines),
            _safe_metadata(
                missing=missing,
                doctor_handoff=doctor_handoff,
                jarvis_doctor_handoff=doctor_handoff,
                jarvis_doctor_handoff_ready=True,
                checks=len(checks),
                next_commands=next_commands,
                next_command_count=len(next_commands),
                agi_next_gate=agi_handoff["gate"],
                agi_next_target_title=agi_handoff["build_target"],
                agi_handoff_selection_source=agi_handoff["selection_source"],
                agi_handoff_selection_reason=agi_handoff["selection_reason"],
                agi_handoff_canonical_selector_command=agi_handoff["canonical_selector_command"],
                agi_handoff_deliberate_focus_override=agi_handoff["deliberate_focus_override"],
                agi_configured_gate_names=agi_handoff["configured_gate_names"],
                agi_configured_gate_count=agi_handoff["configured_gate_count"],
                agi_next_build_command=agi_handoff["next_command"],
                agi_next_evidence_closure_commands=agi_handoff["closure_commands"],
                agi_next_evidence_closure_command_count=len(agi_handoff["closure_commands"]),
                agi_next_focused_verification_commands=agi_handoff["focused_verification_commands"],
                agi_next_focused_verification_command_count=len(agi_handoff["focused_verification_commands"]),
                agi_next_likely_files=agi_handoff["likely_files"],
                agi_next_likely_file_count=len(agi_handoff["likely_files"]),
                agi_next_target_file_integrity_status=agi_target_integrity["status"],
                agi_next_target_files_checked=agi_target_integrity["checked"],
                agi_next_target_files_exist=agi_target_integrity["all_exist"],
                agi_next_missing_target_files=agi_target_integrity["missing"],
                agi_next_missing_target_file_count=agi_target_integrity["missing_count"],
                agi_next_target_integrity_blocks_start=not agi_target_integrity["all_exist"],
                agi_next_acceptance_checks=agi_handoff["acceptance_checks"],
                agi_next_acceptance_check_count=len(agi_handoff["acceptance_checks"]),
                agi_next_acceptance_gap_preview=agi_handoff["acceptance_gap_preview"],
                agi_next_acceptance_gap_preview_count=agi_handoff["acceptance_gap_preview_count"],
                agi_next_first_acceptance_gap=agi_handoff["first_acceptance_gap"],
                agi_next_build_packet_ready_for_review=agi_build_packet_ready,
                agi_next_real_execution_gap_count=agi_handoff["real_execution_gap_count"],
                agi_next_real_execution_gaps_by_gate=agi_handoff["real_execution_gaps_by_gate"],
                agi_next_selected_real_execution_gap=agi_handoff["selected_real_execution_gap"],
                agi_next_review_only=True,
                agi_next_draft_only=True,
                agi_next_loads_without_execution=True,
                agi_next_authorizes_execution=False,
                agi_next_authorizes_completion_claim=False,
                agi_next_approval_granted=False,
                agi_likely_files=agi_handoff["likely_files"],
                agi_target_file_integrity_status=agi_target_integrity["status"],
                agi_target_files_checked=agi_target_integrity["checked"],
                agi_target_files_exist=agi_target_integrity["all_exist"],
                agi_missing_target_files=agi_target_integrity["missing"],
                agi_missing_target_file_count=agi_target_integrity["missing_count"],
                agi_target_file_rows=agi_target_integrity["rows"],
                agi_evidence_closure_commands=agi_handoff["closure_commands"],
                agi_evidence_closure_command_count=len(agi_handoff["closure_commands"]),
                agi_focused_verification_commands=agi_handoff["focused_verification_commands"],
                agi_focused_verification_command_count=len(agi_handoff["focused_verification_commands"]),
                agi_acceptance_checks=agi_handoff["acceptance_checks"],
                agi_acceptance_check_count=len(agi_handoff["acceptance_checks"]),
                agi_acceptance_gap_preview=agi_handoff["acceptance_gap_preview"],
                agi_acceptance_gap_preview_count=agi_handoff["acceptance_gap_preview_count"],
                agi_first_acceptance_gap=agi_handoff["first_acceptance_gap"],
                agi_build_target=agi_handoff["build_target"],
                agi_real_execution_gap_count=agi_handoff["real_execution_gap_count"],
                agi_real_execution_gaps_by_gate=agi_handoff["real_execution_gaps_by_gate"],
                agi_selected_real_execution_gap=agi_handoff["selected_real_execution_gap"],
                project_name="Jarvis V3",
                project_root=project_root_display,
                dashboard_launcher=dashboard_launcher_display,
                dashboard_launcher_exists=dashboard_launcher.exists(),
                dashboard_launch_command=V3_DASHBOARD_COMMAND,
                dashboard_ask_command=V3_DASHBOARD_INFO_COMMAND,
                dashboard_module_command=V3_DASHBOARD_MODULE_COMMAND,
                path_metadata_redacted=True,
                telegram_bot_token_configured=telegram_bot_token["configured"],
                telegram_bot_token_valid=telegram_bot_token["valid"],
                telegram_bot_token_source=telegram_bot_token["source"],
                telegram_bot_token_required=telegram_bot_token["required"],
                telegram_bot_token_raw_chars=telegram_bot_token["raw_chars"],
                telegram_bot_token_raw_truncated=telegram_bot_token["raw_truncated"],
                gmail_address_configured=gmail_address["configured"],
                gmail_address_valid=gmail_address["valid"],
                gmail_address_source=gmail_address["source"],
                gmail_address_required=gmail_address["required"],
                gmail_address_raw_chars=gmail_address["raw_chars"],
                gmail_address_raw_truncated=gmail_address["raw_truncated"],
                gmail_app_password_configured=gmail_app_password["configured"],
                gmail_app_password_valid=gmail_app_password["valid"],
                gmail_app_password_source=gmail_app_password["source"],
                gmail_app_password_required=gmail_app_password["required"],
                gmail_app_password_raw_chars=gmail_app_password["raw_chars"],
                gmail_app_password_raw_truncated=gmail_app_password["raw_truncated"],
                oav_vision_reviewer_command_configured=oav_vision_reviewer["configured"],
                oav_vision_reviewer_command_valid=oav_vision_reviewer["valid"],
                oav_vision_reviewer_command_source=oav_vision_reviewer["source"],
                oav_vision_reviewer_command_required=oav_vision_reviewer["required"],
                oav_vision_reviewer_command_raw_chars=oav_vision_reviewer["raw_chars"],
                oav_vision_reviewer_command_raw_truncated=oav_vision_reviewer["raw_truncated"],
                oav_vision_reviewer_command_token_count=oav_vision_reviewer["token_count"],
                oav_vision_reviewer_command_executable_resolved=oav_vision_reviewer["executable_resolved"],
                oav_vision_reviewer_command_executable_kind=oav_vision_reviewer["executable_kind"],
                oav_vision_reviewer_command_executable_exists=oav_vision_reviewer["executable_exists"],
                oav_vision_reviewer_command_executable_is_file=oav_vision_reviewer["executable_is_file"],
                oav_vision_reviewer_command_executable_is_dir=oav_vision_reviewer["executable_is_dir"],
                oav_vision_reviewer_command_executable_parent_exists=oav_vision_reviewer["executable_parent_exists"],
                oav_vision_reviewer_command_executable_on_path=oav_vision_reviewer["executable_on_path"],
                oav_vision_reviewer_command_executable_is_executable=oav_vision_reviewer["executable_is_executable"],
                oav_vision_reviewer_command_parse_error=oav_vision_reviewer["parse_error"],
                executes_oav_vision_reviewer_command=False,
                voice_whisper_model_path_configured=voice_whisper_model["configured"],
                voice_whisper_model_path_valid=voice_whisper_model["valid"],
                voice_whisper_model_path_source=voice_whisper_model["source"],
                voice_whisper_model_path_required=voice_whisper_model["required"],
                voice_whisper_model_path_expected=voice_whisper_model["expected"],
                voice_whisper_model_path_raw_chars=voice_whisper_model["raw_chars"],
                voice_whisper_model_path_raw_truncated=voice_whisper_model["raw_truncated"],
                voice_whisper_model_path_exists=voice_whisper_model["exists"],
                voice_whisper_model_path_is_file=voice_whisper_model["is_file"],
                voice_whisper_model_path_is_dir=voice_whisper_model["is_dir"],
                voice_whisper_model_path_parent_exists=voice_whisper_model["parent_exists"],
                voice_faster_whisper_model_path_configured=voice_faster_whisper_model["configured"],
                voice_faster_whisper_model_path_valid=voice_faster_whisper_model["valid"],
                voice_faster_whisper_model_path_source=voice_faster_whisper_model["source"],
                voice_faster_whisper_model_path_required=voice_faster_whisper_model["required"],
                voice_faster_whisper_model_path_expected=voice_faster_whisper_model["expected"],
                voice_faster_whisper_model_path_raw_chars=voice_faster_whisper_model["raw_chars"],
                voice_faster_whisper_model_path_raw_truncated=voice_faster_whisper_model["raw_truncated"],
                voice_faster_whisper_model_path_exists=voice_faster_whisper_model["exists"],
                voice_faster_whisper_model_path_is_file=voice_faster_whisper_model["is_file"],
                voice_faster_whisper_model_path_is_dir=voice_faster_whisper_model["is_dir"],
                voice_faster_whisper_model_path_parent_exists=voice_faster_whisper_model["parent_exists"],
                loads_voice_whisper_model=False,
                loads_voice_faster_whisper_model=False,
                reads_audio_for_voice_model_validation=False,
                model_provider=model_provider,
                model_provider_configured=provider_status["configured"],
                model_provider_valid=provider_valid,
                model_provider_source=provider_status["source"],
                model_provider_fallback=provider_status["fallback_value"],
                ollama_required=ollama_provider,
                openai_api_key_configured=openai_api_key["configured"],
                openai_api_key_valid=openai_api_key["valid"],
                openai_api_key_required=openai_api_key["required"],
                openai_api_key_value_exposed=False,
                openai_safety_identifier_sent=openai_provider,
                openai_safety_identifier_scope="single_owner" if openai_provider else "not_applicable",
                openai_safety_identifier_value_exposed=False,
                openai_safety_identifier_uses_personal_data=False,
                openai_safety_identifier_uses_api_key=False,
                openai_live_access_probed=False,
                openai_request_store_flag=False,
                openai_account_retention_controls_checked=False,
                openai_zero_data_retention_verified=False,
                openai_default_abuse_monitoring_may_retain_content=openai_provider,
                openai_default_abuse_monitoring_max_days=30 if openai_provider else 0,
                chat_reasoning_effort=chat_reasoning["effective_value"],
                chat_reasoning_effort_valid=chat_reasoning["valid"],
                planner_reasoning_effort=planner_reasoning["effective_value"],
                planner_reasoning_effort_valid=planner_reasoning["valid"],
                fallback_model_alias_configured=fallback_model_alias["configured"],
                fallback_model_alias_valid=fallback_model_alias["valid"],
                fallback_model_alias_source=fallback_model_alias["source"],
                fallback_model_alias_chars=fallback_model_alias["alias_chars"],
                fallback_model_alias_truncated=fallback_model_alias["alias_truncated"],
                fallback_model_alias_fallback=fallback_model_alias["fallback_alias"],
                chat_model_alias_configured=chat_model_alias["configured"],
                chat_model_alias_valid=chat_model_alias["valid"],
                chat_model_alias_source=chat_model_alias["source"],
                chat_model_alias_chars=chat_model_alias["alias_chars"],
                chat_model_alias_truncated=chat_model_alias["alias_truncated"],
                chat_model_alias_fallback=chat_model_alias["fallback_alias"],
                planner_model_alias_configured=planner_model_alias["configured"],
                planner_model_alias_valid=planner_model_alias["valid"],
                planner_model_alias_source=planner_model_alias["source"],
                planner_model_alias_chars=planner_model_alias["alias_chars"],
                planner_model_alias_truncated=planner_model_alias["alias_truncated"],
                planner_model_alias_fallback=planner_model_alias["fallback_alias"],
                planner_timeout_configured=planner_timeout["configured"],
                planner_timeout_valid=planner_timeout["valid"],
                planner_timeout_source=planner_timeout["source"],
                planner_timeout_default=planner_timeout_default,
                planner_timeout_minimum=MIN_MODEL_TIMEOUT_SECONDS,
                planner_timeout_maximum=planner_timeout_maximum,
                planner_timeout_effective_seconds=planner_timeout["effective_seconds"],
                chat_timeout_configured=chat_timeout["configured"],
                chat_timeout_valid=chat_timeout["valid"],
                chat_timeout_source=chat_timeout["source"],
                chat_timeout_default=chat_timeout_default,
                chat_timeout_minimum=MIN_CHAT_TIMEOUT_SECONDS,
                chat_timeout_maximum=chat_timeout_maximum,
                chat_timeout_effective_seconds=chat_timeout["effective_seconds"],
                chat_max_reply_tokens_configured=chat_max_reply_tokens["configured"],
                chat_max_reply_tokens_valid=chat_max_reply_tokens["valid"],
                chat_max_reply_tokens_source=chat_max_reply_tokens["source"],
                chat_max_reply_tokens_default=DEFAULT_CHAT_MAX_REPLY_TOKENS,
                chat_max_reply_tokens_minimum=MIN_CHAT_MAX_REPLY_TOKENS,
                chat_max_reply_tokens_maximum=MAX_CHAT_MAX_REPLY_TOKENS,
                chat_max_reply_tokens_effective=chat_max_reply_tokens["effective_value"],
                chat_max_history_messages_configured=chat_max_history_messages["configured"],
                chat_max_history_messages_valid=chat_max_history_messages["valid"],
                chat_max_history_messages_source=chat_max_history_messages["source"],
                chat_max_history_messages_default=DEFAULT_CHAT_MAX_HISTORY_MESSAGES,
                chat_max_history_messages_minimum=MIN_CHAT_MAX_HISTORY_MESSAGES,
                chat_max_history_messages_maximum=MAX_CHAT_MAX_HISTORY_MESSAGES,
                chat_max_history_messages_effective=chat_max_history_messages["effective_value"],
                smoke_module_timeout_configured=smoke_timeout_configured,
                smoke_module_timeout_valid=smoke_timeout["valid"],
                smoke_module_timeout_source=smoke_timeout["source"],
                smoke_module_timeout_default=DEFAULT_SMOKE_MODULE_TIMEOUT_SECONDS,
                smoke_module_timeout_effective_seconds=smoke_timeout["effective_seconds"],
                model_planner_toggle_configured=model_planner_toggle["configured"],
                model_planner_toggle_valid=model_planner_toggle["valid"],
                model_planner_toggle_source=model_planner_toggle["source"],
                model_planner_toggle_effective_enabled=model_planner_toggle["effective_enabled"],
                model_planner_toggle_fallback_enabled=model_planner_toggle["fallback_enabled"],
                remote_compaction_toggle_configured=remote_compaction_toggle["configured"],
                remote_compaction_toggle_valid=remote_compaction_toggle["valid"],
                remote_compaction_toggle_source=remote_compaction_toggle["source"],
                remote_conversation_compaction_enabled=remote_compaction_toggle["effective_enabled"],
                remote_compaction_sends_history_to_external_model=bool(
                    openai_provider and remote_compaction_toggle["effective_enabled"]
                ),
                remote_compaction_live_access_probed=False,
                remote_personal_context_toggle_configured=remote_personal_context_toggle["configured"],
                remote_personal_context_toggle_valid=remote_personal_context_toggle["valid"],
                remote_personal_context_toggle_source=remote_personal_context_toggle["source"],
                remote_personal_context_allowed=remote_personal_context_toggle["effective_enabled"],
                remote_personal_context_sends_stored_context_to_external_model=bool(
                    openai_provider and remote_personal_context_toggle["effective_enabled"]
                ),
                remote_personal_context_current_message_may_be_external=bool(openai_provider),
                remote_personal_context_content_in_metadata=False,
                storage_fallback_disabled=storage_fallback_env["disabled"],
                storage_fallback_disable_configured=storage_fallback_env["disable_configured"],
                storage_fallback_disable_valid=storage_fallback_env["disable_valid"],
                storage_fallback_disable_source=storage_fallback_env["disable_source"],
                storage_fallback_disable_chars=storage_fallback_env["disable_raw_chars"],
                storage_fallback_disable_truncated=storage_fallback_env["disable_raw_truncated"],
                storage_fallback_disable_fallback_disabled=storage_fallback_env["disable_fallback_disabled"],
                status_dashboard_host_configured=status_host_configured,
                status_dashboard_host_valid=status_host_config.valid,
                status_dashboard_host_source=status_host_config.source,
                status_dashboard_host_default=DEFAULT_STATUS_HOST,
                status_dashboard_port_configured=status_port_configured,
                status_dashboard_port_valid=status_port_config.valid,
                status_dashboard_port_source=status_port_config.source,
                status_dashboard_port_default=DEFAULT_STATUS_PORT,
                google_connector_dependencies=google_dependency_status,
                reads_google_credentials=False,
                pending_approvals=len(pending_approvals),
                open_tasks=len(open_tasks),
                active_goals=len(active_goals),
                recent_tool_runs=len(recent_runs),
                readable_recent_tool_runs=len(readable_recent_runs),
                unreadable_recent_tool_run_rows=unreadable_recent_run_rows,
                recent_failed_runs=len(recent_failures),
                recent_approval_held_runs=len(recent_approval_held_runs),
                approval_held_review_required=bool(recent_approval_held_runs),
                approval_held_review_commands=approval_held_review_commands,
                approval_held_review_command_count=len(approval_held_review_commands),
                approval_held_review_next_command=approval_held_review_next_command,
                approval_held_review_approval_id=approval_held_review_approval_id,
                recent_verification_runs=len(verification_runs),
                recent_after_action_learning_runs=len(learning_runs),
                next_audit_command=next_audit_command,
                execution_health_recovery_closure_state=recovery_closure["state"],
                execution_health_recovery_closure_ready_to_retry=recovery_closure["ready_to_retry"],
                execution_health_recovery_closure_missing=recovery_closure["missing"],
                execution_health_recovery_closure_missing_count=recovery_closure["missing_count"],
                execution_health_recovery_closure_required_commands=recovery_closure["required_commands"],
                execution_health_recovery_closure_next_required_command=recovery_closure["next_required_command"],
                execution_health_recovery_closure_proof_queue=recovery_closure["required_commands"],
                execution_health_recovery_closure_proof_queue_count=len(recovery_closure["required_commands"]),
                execution_health_recovery_closure_next_proof_command=recovery_closure["next_required_command"],
                execution_health_recovery_closure_blocks_completion_claim=recovery_closure["blocks_completion_claim"],
                execution_health_recovery_closure_target_run_id=recovery_closure["target_run_id"],
                execution_health_recovery_closure_target_tool_name=recovery_closure["target_tool_name"],
                execution_health_recovery_closure_target_verification_receipts=recovery_closure["target_verification_receipts"],
                execution_health_recovery_closure_target_recovery_packets=recovery_closure["target_recovery_packets"],
                execution_health_recovery_closure_target_after_action_learning_packets=recovery_closure["target_after_action_learning_packets"],
                execution_learning_state=learning_debt["state"],
                execution_learning_blocks_completion_claim=learning_debt["blocks_completion_claim"],
                execution_learning_recent_action_runs=learning_debt["recent_action_runs"],
                execution_learning_failed_or_blocked_action_runs=learning_debt["failed_or_blocked_action_runs"],
                execution_learning_recent_verification_runs=learning_debt["recent_verification_runs"],
                execution_learning_recent_recovery_runs=learning_debt["recent_recovery_runs"],
                execution_learning_recent_after_action_learning_runs=learning_debt["recent_after_action_learning_runs"],
                execution_learning_target_run_id=learning_debt["target_run_id"],
                execution_learning_target_tool_name=learning_debt["target_tool_name"],
                execution_learning_target_after_action_learning_packets=learning_debt["target_after_action_learning_packets"],
                execution_learning_missing=learning_debt["missing"],
                execution_learning_missing_count=learning_debt["missing_count"],
                execution_learning_required_commands=learning_debt["required_commands"],
                execution_learning_next_required_command=learning_debt["next_required_command"],
                execution_learning_proof_queue=learning_debt["required_commands"],
                execution_learning_proof_queue_count=len(learning_debt["required_commands"]),
                execution_learning_next_proof_command=learning_debt["next_required_command"],
                execution_learning_actionable_required_commands=learning_debt["actionable_required_commands"],
                execution_learning_actionable_required_command_count=learning_debt["actionable_required_command_count"],
                execution_learning_actionable_next_required_command=learning_debt["actionable_next_required_command"],
                execution_learning_actionable_proof_queue=learning_debt["actionable_proof_queue"],
                execution_learning_actionable_proof_queue_count=learning_debt["actionable_proof_queue_count"],
                execution_learning_actionable_next_proof_command=learning_debt["actionable_next_proof_command"],
                execution_learning_next_evidence_command=learning_debt["next_evidence_command"],
                audit_readability_review_required=bool(audit_readability_review_commands),
                audit_readability_review_commands=audit_readability_review_commands,
                audit_readability_review_command_count=len(audit_readability_review_commands),
                audit_readability_review_next_command=(
                    audit_readability_review_commands[0] if audit_readability_review_commands else ""
                ),
                completion_claim_state=completion_claim_state,
                completion_blockers=completion_blockers,
                completion_blocker_count=len(completion_blockers),
                completion_claim_ready=completion_claim_state == "READY_FOR_HUMAN_REVIEW",
                completion_proof_queue=completion_proof_queue,
                completion_proof_queue_count=len(completion_proof_queue),
                completion_next_proof_command=completion_proof_queue[0] if completion_proof_queue else "",
                next_completion_proof_command=completion_proof_queue[0] if completion_proof_queue else "",
                storage_status=storage["status"],
                storage_available=storage["available"],
                storage_configured_status=storage["configured_status"],
                storage_configured_available=storage["configured_available"],
                storage_ready_for_completion_claim=storage["ready_for_completion_claim"],
                storage_runtime_fallback_active=storage["runtime_fallback_active"],
                storage_runtime_fallback_reason=storage["runtime_fallback_reason"],
                storage_runtime_fallback_exception_type=storage["runtime_fallback_exception_type"],
                storage_runtime_fallback_db_path_display=storage["runtime_fallback_db_path_display"],
                storage_runtime_fallback_vault_path_display=storage["runtime_fallback_vault_path_display"],
                storage_data_dir=storage["data_dir"],
                storage_data_dir_exists=storage["data_dir_exists"],
                storage_data_dir_writable=storage["data_dir_writable"],
                storage_db_path=storage["db_path"],
                storage_db_parent=storage["db_parent"],
                storage_db_exists=storage["db_exists"],
                storage_db_parent_exists=storage["db_parent_exists"],
                storage_db_parent_writable=storage["db_parent_writable"],
                storage_db_file_writable=storage["db_file_writable"],
                storage_obsidian_vault=storage["obsidian_vault"],
                storage_obsidian_root=storage["obsidian_root"],
                storage_obsidian_root_path=storage["obsidian_root_path"],
                storage_obsidian_vault_exists=storage["obsidian_vault_exists"],
                storage_obsidian_root_exists=storage["obsidian_root_exists"],
                storage_obsidian_vault_writable=storage["obsidian_vault_writable"],
                storage_obsidian_root_writable=storage["obsidian_root_writable"],
                storage_workspace_local_notes=storage["workspace_local_notes"],
                storage_metadata_only=storage["metadata_only"],
                storage_issues=storage.get("issues", []),
                storage_recovery_required=storage["recovery_required"],
                storage_recovery_reason=storage["recovery_reason"],
                storage_recovery_mode=storage["recovery_mode"],
                storage_recovery_next_operator_action=storage["recovery_next_operator_action"],
                storage_recovery_restart_required=storage["recovery_restart_required"],
                storage_readiness_blocks_completion_claim=storage["readiness_blocks_completion_claim"],
                storage_readiness_blocker=storage["readiness_blocker"],
                storage_readiness_next_commands=storage["readiness_next_commands"],
                storage_readiness_next_command_count=storage["readiness_next_command_count"],
                storage_readiness_next_required_command=storage["readiness_next_required_command"],
                storage_readiness_next_proof_command=storage["readiness_next_proof_command"],
                storage_recovery_check_tool_command=storage.get("recovery_check_tool_command", ""),
                storage_recovery_check_command=storage.get("recovery_check_command", ""),
                storage_recovery_check_api=storage.get("recovery_check_api", STORAGE_RECOVERY_CHECK_API),
                storage_recovery_command=storage.get("recovery_command", ""),
                sqlite_store_available=not sqlite_store_exception_type,
                sqlite_store_exception_type=sqlite_store_exception_type,
                ollama_destination_probe_attempted=ollama_live_access_probed,
                ollama_probe_uses_explicit_loopback_env=False,
                ollama_probe_proxy_environment_stripped=False,
                ollama_probe_uses_validated_loopback_http=ollama_live_access_probed,
                ollama_probe_redirects_blocked=ollama_live_access_probed,
                ollama_probe_proxy_bypassed=ollama_live_access_probed,
                ollama_probe_uses_subprocess=False,
                ollama_probe_diagnostic=ollama_probe_diagnostic,
                ollama_probe_model_count=ollama_probe_model_count,
                ollama_daemon_cloud_disabled_verified=False,
                ollama_on_device_model_execution_verified=False,
                ollama_cloud_features_disabled_verified=False,
                **ollama_destination.receipt(),
                **ollama_no_cloud.receipt(),
            ),
        )

    return jarvis_doctor
