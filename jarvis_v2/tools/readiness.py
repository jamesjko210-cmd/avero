from __future__ import annotations

import importlib.util
import json
import re
from typing import Any, Callable

from jarvis_v2.agent.types import ToolResult
from jarvis_v2.agent.model_provider import (
    normalized_model_provider,
    ollama_local_only_policy,
    openai_api_key_configured,
    probe_ollama_models,
    resolve_ollama_destination,
)
from jarvis_v2.automations.compaction import COMPACTION_JOB_NAME, COMPACTION_JOB_TYPE
from jarvis_v2.config import JarvisConfig
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import MemoryStore
from jarvis_v2.tools.storage import (
    BOOTSTRAP_CHECK_COMMAND,
    BOOTSTRAP_WRITE_COMMAND,
    STORAGE_RECOVERY_PLAN_COMMAND,
    STORAGE_RECOVERY_CHECK_API,
    STORAGE_RECOVERY_CHECK_COMMAND,
    _configured_storage_diagnostics,
    safe_storage_text,
)


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "calls_model": False,
        "calls_external_service": False,
        "executes_tools": False,
        "queues_approval": False,
        "approves_request": False,
        "dismisses_request": False,
        "controls_computer": False,
        "reads_clipboard": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "writes_files": False,
        "edits_files": False,
        "writes_database": False,
        "writes_notes": False,
        "writes_memory": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "requires_approval": False,
        "speaks": False,
        "completes_tasks": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
    }
    metadata.update(extra)
    return metadata


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _safe_ollama_probe_diagnostic(value: Any, *, ok: bool) -> str:
    token = str(value or "").strip().casefold()
    if re.fullmatch(r"ollama_[a-z0-9_]{1,72}", token):
        return token
    return "ollama_probe_ok" if ok else "ollama_probe_unavailable"


def _ollama_no_cloud_detail(policy: Any, *, route_reachable: bool) -> str:
    if not policy.valid:
        status = (
            "invalid setting; personalized Ollama operation is not ready; "
            "current-message-only model route may remain usable; stored context withheld"
        )
    elif not policy.unverified_context_consent_valid:
        status = (
            "invalid consent setting; personalized Ollama operation is not ready; "
            "current-message-only model route may remain usable; stored context withheld"
        )
    elif policy.cloud_model_alias:
        status = "cloud model alias blocked; personalized Ollama operation is not ready; stored context withheld"
    elif policy.personal_context_allowed and route_reachable:
        status = "stored personal context allowed for the non-cloud model alias"
    elif policy.personal_context_allowed:
        status = "policy allows stored context, but the Ollama route is not reachable"
    else:
        status = "current-message-only model route may remain usable; stored context withheld"
    return (
        f"{status}; configured: {'yes' if policy.configured else 'no'}; "
        f"valid: {'yes' if policy.valid else 'no'}; "
        f"requested: {'yes' if policy.requested else 'no'}; "
        f"consent configured: {'yes' if policy.unverified_context_consent_configured else 'no'}; "
        f"consent valid: {'yes' if policy.unverified_context_consent_valid else 'no'}; "
        f"consent granted: {'yes' if policy.unverified_context_consent_allowed else 'no'}; "
        "execution locality: unknown; "
        "daemon cloud-disabled state: independently unverified; "
        "on-device execution: independently unverified; value hidden"
    )


def _model_route_status(config: JarvisConfig | None) -> dict[str, Any]:
    provider = normalized_model_provider(config.model_provider if config is not None else "ollama")
    if provider == "invalid":
        return {
            "provider": "invalid",
            "provider_valid": False,
            "label": "Model provider configuration",
            "ready": False,
            "detail": (
                "invalid; set JARVIS_MODEL_PROVIDER to ollama or openai; no Ollama or OpenAI "
                "connectivity probe attempted; execution locality unknown"
            ),
            "live_access_probed": False,
            "ollama_required": False,
            "safety_identifier_sent": False,
            "safety_identifier_scope": "not_applicable",
            "safety_identifier_value_exposed": False,
            "safety_identifier_uses_personal_data": False,
            "safety_identifier_uses_api_key": False,
            "request_store_flag": False,
            "account_retention_controls_checked": False,
            "zero_data_retention_verified": False,
            "default_abuse_monitoring_may_retain_content": False,
            "default_abuse_monitoring_max_days": 0,
            "ollama_destination_probe_attempted": False,
            "ollama_probe_uses_explicit_loopback_env": False,
            "ollama_probe_proxy_environment_stripped": False,
            "ollama_probe_uses_validated_loopback_http": False,
            "ollama_probe_redirects_blocked": False,
            "ollama_probe_proxy_bypassed": False,
            "ollama_probe_uses_subprocess": False,
            "ollama_probe_diagnostic": "ollama_probe_not_attempted",
            "ollama_probe_model_count": 0,
            "ollama_personalized_context_ready": False,
            "ollama_current_message_only_usable": False,
            "ollama_stored_context_withheld": True,
            "ollama_daemon_cloud_disabled_verified": False,
            "ollama_on_device_model_execution_verified": False,
            "ollama_cloud_features_disabled_verified": False,
            "ollama_destination_allowed": False,
            "ollama_destination_configured": False,
            "ollama_destination_source": "not_evaluated",
            "ollama_destination_scheme": "",
            "ollama_destination_address_family": "",
            "ollama_destination_port": 0,
            "ollama_destination_diagnostic": "not_evaluated",
            "ollama_destination_value_exposed": False,
            "ollama_redirects_allowed": False,
            "ollama_proxy_environment_allowed": False,
            "ollama_no_cloud_configured": False,
            "ollama_no_cloud_valid": False,
            "ollama_no_cloud_requested": False,
            "ollama_cloud_model_alias": False,
            "ollama_personal_context_allowed": False,
            "ollama_unverified_context_consent_configured": False,
            "ollama_unverified_context_consent_valid": False,
            "ollama_unverified_context_consent_allowed": False,
            "ollama_unverified_context_consent_env": "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT",
            "ollama_execution_locality_verified": False,
            "ollama_cloud_policy_diagnostic": "not_evaluated",
            "ollama_cloud_policy_value_exposed": False,
        }
    ollama_destination = resolve_ollama_destination()
    ollama_no_cloud = ollama_local_only_policy(config.chat_model if config is not None else "")
    if provider == "openai":
        configured = openai_api_key_configured()
        return {
            "provider": "openai",
            "provider_valid": True,
            "label": "OpenAI API configuration",
            "ready": configured,
            "detail": (
                "API key configured; value hidden; live access not probed; store=false is not a "
                "zero-retention guarantee; a stable non-personal single-owner safety pseudonym is sent; "
                "account controls are not checked and default "
                "abuse-monitoring logs may retain prompts/responses for up to 30 days"
                if configured
                else "set OPENAI_API_KEY locally, then run `model routing status`"
            ),
            "live_access_probed": False,
            "ollama_required": False,
            "safety_identifier_sent": True,
            "safety_identifier_scope": "single_owner",
            "safety_identifier_value_exposed": False,
            "safety_identifier_uses_personal_data": False,
            "safety_identifier_uses_api_key": False,
            "request_store_flag": False,
            "account_retention_controls_checked": False,
            "zero_data_retention_verified": False,
            "default_abuse_monitoring_may_retain_content": True,
            "default_abuse_monitoring_max_days": 30,
            "ollama_destination_probe_attempted": False,
            "ollama_probe_uses_explicit_loopback_env": False,
            "ollama_probe_proxy_environment_stripped": False,
            "ollama_probe_uses_validated_loopback_http": False,
            "ollama_probe_redirects_blocked": False,
            "ollama_probe_proxy_bypassed": False,
            "ollama_probe_uses_subprocess": False,
            "ollama_probe_diagnostic": "ollama_probe_not_attempted",
            "ollama_probe_model_count": 0,
            "ollama_personalized_context_ready": False,
            "ollama_current_message_only_usable": False,
            "ollama_stored_context_withheld": True,
            "ollama_daemon_cloud_disabled_verified": False,
            "ollama_on_device_model_execution_verified": False,
            "ollama_cloud_features_disabled_verified": False,
            **ollama_destination.receipt(),
            **ollama_no_cloud.receipt(),
        }
    live_access_probed = ollama_destination.allowed
    if live_access_probed:
        probe_ok, model_names, raw_diagnostic, _error_type = probe_ollama_models(timeout_seconds=5.0)
        reachable = bool(probe_ok)
        probe_model_count = len(model_names) if isinstance(model_names, list) else 0
        probe_diagnostic = _safe_ollama_probe_diagnostic(raw_diagnostic, ok=reachable)
    else:
        reachable = False
        probe_model_count = 0
        probe_diagnostic = "ollama_destination_not_local"
    current_message_ready = bool(reachable and not ollama_no_cloud.cloud_model_alias)
    personalized_context_ready = bool(current_message_ready and ollama_no_cloud.personal_context_allowed)
    family = ollama_destination.address_family or "unavailable"
    port = str(ollama_destination.port) if ollama_destination.port else "unavailable"
    return {
        "provider": "ollama",
        "provider_valid": True,
        "label": "Ollama current-message route",
        "ready": current_message_ready,
        "detail": (
            "needed for Ollama chat/planning; execution locality unknown; "
            + ("probe skipped; destination is invalid or non-loopback; " if not ollama_destination.allowed else "")
            + f"source: {ollama_destination.source}; family: {family}; port: {port}; "
            "redirects: blocked; proxy policy: stripped from accepted probes; "
            + _ollama_no_cloud_detail(ollama_no_cloud, route_reachable=reachable)
        ),
        "live_access_probed": live_access_probed,
        "ollama_required": True,
        "safety_identifier_sent": False,
        "safety_identifier_scope": "not_applicable",
        "safety_identifier_value_exposed": False,
        "safety_identifier_uses_personal_data": False,
        "safety_identifier_uses_api_key": False,
        "request_store_flag": False,
        "account_retention_controls_checked": False,
        "zero_data_retention_verified": False,
        "default_abuse_monitoring_may_retain_content": False,
        "default_abuse_monitoring_max_days": 0,
        "ollama_destination_probe_attempted": live_access_probed,
        "ollama_probe_uses_explicit_loopback_env": False,
        "ollama_probe_proxy_environment_stripped": False,
        "ollama_probe_uses_validated_loopback_http": live_access_probed,
        "ollama_probe_redirects_blocked": live_access_probed,
        "ollama_probe_proxy_bypassed": live_access_probed,
        "ollama_probe_uses_subprocess": False,
        "ollama_probe_diagnostic": probe_diagnostic,
        "ollama_probe_model_count": probe_model_count,
        "ollama_personalized_context_ready": personalized_context_ready,
        "ollama_current_message_only_usable": bool(current_message_ready and not personalized_context_ready),
        "ollama_stored_context_withheld": not personalized_context_ready,
        "ollama_daemon_cloud_disabled_verified": False,
        "ollama_on_device_model_execution_verified": False,
        "ollama_cloud_features_disabled_verified": False,
        **ollama_destination.receipt(),
        **ollama_no_cloud.receipt(),
    }


def _ollama_route_metadata(model_route: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "ollama_destination_probe_attempted",
        "ollama_probe_uses_explicit_loopback_env",
        "ollama_probe_proxy_environment_stripped",
        "ollama_probe_uses_validated_loopback_http",
        "ollama_probe_redirects_blocked",
        "ollama_probe_proxy_bypassed",
        "ollama_probe_uses_subprocess",
        "ollama_probe_diagnostic",
        "ollama_probe_model_count",
        "ollama_destination_allowed",
        "ollama_destination_configured",
        "ollama_destination_source",
        "ollama_destination_scheme",
        "ollama_destination_address_family",
        "ollama_destination_port",
        "ollama_destination_diagnostic",
        "ollama_destination_value_exposed",
        "ollama_redirects_allowed",
        "ollama_proxy_environment_allowed",
        "ollama_no_cloud_configured",
        "ollama_no_cloud_valid",
        "ollama_no_cloud_requested",
        "ollama_cloud_model_alias",
        "ollama_personal_context_allowed",
        "ollama_unverified_context_consent_configured",
        "ollama_unverified_context_consent_valid",
        "ollama_unverified_context_consent_allowed",
        "ollama_unverified_context_consent_env",
        "ollama_execution_locality_verified",
        "ollama_cloud_policy_diagnostic",
        "ollama_cloud_policy_value_exposed",
        "ollama_personalized_context_ready",
        "ollama_current_message_only_usable",
        "ollama_stored_context_withheld",
        "ollama_daemon_cloud_disabled_verified",
        "ollama_on_device_model_execution_verified",
        "ollama_cloud_features_disabled_verified",
    )
    return {key: model_route[key] for key in keys}


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


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
    failed: list[Any] = []
    approval_held: list[Any] = []
    for row in rows:
        if _row_bool(row, "ok", default=True):
            continue
        if _is_approval_held_tool_run(row):
            approval_held.append(row)
        else:
            failed.append(row)
    return failed, approval_held


def _positive_int_text(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return ""
    return str(number) if number > 0 else ""


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


def _approval_held_review_line(commands: list[str]) -> str:
    if not commands:
        return "- Recent approval-held tool runs exist; inspect with `recent tool runs` or `pending approvals`."
    return (
        "- Recent approval-held tool runs exist; review "
        + " -> ".join(f"`{command}`" for command in commands[:3])
        + "."
    )


def _safe_readiness_text(value: Any, default: str = "unavailable", limit: int = 240) -> str:
    try:
        text = safe_storage_text(value)
    except Exception:
        return default
    text = text.replace("\n", " ").replace("\r", " ").strip()
    if not text:
        return default
    if len(text) > limit:
        return f"{text[:limit]}... [truncated]"
    return text


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    try:
        if isinstance(row, dict):
            return row.get(key, default)
        return row[key]
    except Exception:
        return default


def _row_bool(row: Any, key: str, default: bool = False) -> bool:
    return _metadata_bool(_row_value(row, key), default)


def _row_enabled(row: Any) -> bool:
    value = _row_value(row, "enabled")
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    return False


def _row_text(row: Any, key: str, default: str = "") -> str:
    return _safe_readiness_text(_row_value(row, key, default), default=default)


def _row_has_any_key(row: Any, keys: tuple[str, ...]) -> bool:
    try:
        if isinstance(row, dict):
            return any(key in row for key in keys)
        row_keys = row.keys()
    except Exception:
        return False
    try:
        return any(key in row_keys for key in keys)
    except Exception:
        return False


def _readable_tool_run_rows(rows: list[Any]) -> tuple[list[Any], int]:
    readable: list[Any] = []
    unreadable = 0
    for row in rows:
        if _row_has_any_key(row, ("id", "tool_name", "ok", "risk", "approved", "metadata", "created_at")):
            readable.append(row)
        else:
            unreadable += 1
    return readable, unreadable


def _readable_scheduled_job_rows(rows: list[Any]) -> tuple[list[Any], int]:
    readable: list[Any] = []
    unreadable = 0
    for row in rows:
        if _row_has_any_key(row, ("id", "name", "job_type", "enabled", "next_run_at", "interval_minutes")):
            readable.append(row)
        else:
            unreadable += 1
    return readable, unreadable


_STORAGE_BOOL_KEYS = (
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


def make_readiness_tools(
    store: MemoryStore,
    vault: ObsidianVault,
    config: JarvisConfig | None,
    list_tools: Callable[[], list[Any]],
    storage_fallback: Callable[[], dict[str, Any] | None] | None = None,
):
    def _storage_readiness() -> dict[str, Any]:
        fallback: dict[str, Any] | None = None
        if storage_fallback is not None:
            try:
                fallback = storage_fallback()
            except Exception:
                fallback = None
        storage, _diagnostics_source = _configured_storage_diagnostics(config, fallback)
        storage = dict(storage)
        fallback_active = bool(fallback)
        for key in _STORAGE_BOOL_KEYS:
            storage[key] = _metadata_bool(storage.get(key))
        issues = list(storage.get("issues") or [])
        if fallback_active:
            issues.append("runtime is using workspace-local fallback storage")
        storage["configured_status"] = storage.get("status", "unknown")
        storage["configured_available"] = storage["available"]
        storage["status"] = "needs attention" if issues else str(storage.get("status", "unknown"))
        storage["available"] = storage["configured_available"] or fallback_active
        storage["ready_for_completion_claim"] = storage["configured_available"] and not fallback_active
        storage["runtime_fallback_active"] = fallback_active
        storage["runtime_fallback_reason"] = str((fallback or {}).get("reason") or "")
        storage["runtime_fallback_exception_type"] = str((fallback or {}).get("exception_type") or "")
        storage["runtime_fallback_db_path_display"] = str((fallback or {}).get("db_path_display") or "")
        storage["runtime_fallback_vault_path_display"] = str((fallback or {}).get("vault_path_display") or "")
        if fallback_active and storage["configured_available"]:
            storage["recovery_mode"] = "restart_runtime_to_configured_storage"
            storage["recovery_next_operator_action"] = (
                "restart or reload Jarvis with the configured durable storage envs, then run `storage status`"
            )
            storage["readiness_next_commands"] = ["storage status", STORAGE_RECOVERY_CHECK_COMMAND, "storage status"]
        elif fallback_active:
            storage["recovery_mode"] = "repair_configured_storage_then_restart_runtime"
            storage["recovery_next_operator_action"] = (
                "review the storage recovery plan, point Jarvis at writable durable storage, "
                "run the no-write storage check, then restart or reload Jarvis"
            )
            storage["readiness_next_commands"] = [
                "storage status",
                STORAGE_RECOVERY_PLAN_COMMAND,
                STORAGE_RECOVERY_CHECK_COMMAND,
                BOOTSTRAP_CHECK_COMMAND,
                BOOTSTRAP_WRITE_COMMAND,
            ]
        elif not storage["configured_available"]:
            storage["recovery_mode"] = "repair_configured_storage"
            storage["recovery_next_operator_action"] = (
                "review the storage recovery plan, point Jarvis at writable durable storage, "
                "and run the no-write storage check"
            )
            storage["readiness_next_commands"] = [
                "storage status",
                STORAGE_RECOVERY_PLAN_COMMAND,
                STORAGE_RECOVERY_CHECK_COMMAND,
                BOOTSTRAP_CHECK_COMMAND,
                BOOTSTRAP_WRITE_COMMAND,
            ]
        else:
            storage["recovery_mode"] = "none"
            storage["recovery_next_operator_action"] = ""
            storage["readiness_next_commands"] = []
        storage["readiness_proof_queue"] = list(storage["readiness_next_commands"])
        storage["issues"] = issues
        return storage

    def _current_readiness_context() -> dict[str, Any]:
        tools = list_tools()
        pending = store.list_pending_approvals(limit=100)
        raw_jobs = store.list_jobs()
        jobs, unreadable_scheduled_job_rows = _readable_scheduled_job_rows(raw_jobs)
        enabled_jobs = [row for row in jobs if _row_enabled(row)]
        state_snapshot_jobs = [row for row in jobs if _row_text(row, "job_type") == "state_snapshot"]
        enabled_state_snapshot_jobs = [row for row in state_snapshot_jobs if _row_enabled(row)]
        conversation_compaction_jobs = [
            row
            for row in jobs
            if _row_text(row, "job_type") == COMPACTION_JOB_TYPE
            or _row_text(row, "name").casefold() == COMPACTION_JOB_NAME.casefold()
        ]
        enabled_conversation_compaction_jobs = [
            row for row in conversation_compaction_jobs if _row_enabled(row)
        ]
        job_types = {job_type for row in enabled_jobs if (job_type := _row_text(row, "job_type"))}
        risky_tools = [
            tool
            for tool in tools
            if tool.risk.name in {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"}
        ]
        missing_required: list[str] = []
        optional_attention: list[str] = []
        storage = _storage_readiness()
        model_route = _model_route_status(config)

        if not vault.root_path.exists():
            missing_required.append("Obsidian Jarvis root is missing")
        if not storage["configured_available"] and not storage["runtime_fallback_active"]:
            missing_required.append("SQLite storage path is not writable")
        if storage["runtime_fallback_active"]:
            optional_attention.append("runtime is using workspace-local fallback storage")
        try:
            store.list_sessions(1)
        except Exception:
            missing_required.append("SQLite memory store is unavailable")
        if not risky_tools:
            missing_required.append("risk-gated tool registry is empty")
        if "state_snapshot" not in job_types:
            optional_attention.append("state snapshot schedule is not enabled")
        if COMPACTION_JOB_TYPE not in job_types:
            optional_attention.append("conversation compaction schedule is not enabled")
        if unreadable_scheduled_job_rows:
            optional_attention.append("scheduled job table has unreadable row(s)")
        if model_route["provider"] == "invalid":
            missing_required.append("Model provider must be ollama or openai")
        elif not model_route["ready"]:
            optional_attention.append(
                "OpenAI model route needs OPENAI_API_KEY"
                if model_route["provider"] == "openai"
                else "Ollama model route needs attention; execution locality unknown"
            )
        if model_route["provider"] == "ollama" and not model_route["ollama_destination_allowed"]:
            missing_required.append("Ollama destination must be a valid HTTP(S) loopback address")
        if model_route["provider"] == "ollama" and not model_route["ollama_personalized_context_ready"]:
            optional_attention.append(
                "Ollama personalized operation is not ready; stored context is withheld while "
                "current-message-only routing may remain usable"
            )
        if not _module_available("pyautogui"):
            optional_attention.append("pyautogui is unavailable for future desktop control")

        if missing_required:
            status = "not ready"
        elif pending or optional_attention:
            status = "prototype ready with attention"
        else:
            status = "prototype ready"

        return {
            "status": status,
            "tools": tools,
            "pending": pending,
            "scheduled_jobs": raw_jobs,
            "readable_scheduled_jobs": jobs,
            "unreadable_scheduled_job_rows": unreadable_scheduled_job_rows,
            "enabled_jobs": enabled_jobs,
            "state_snapshot_jobs": state_snapshot_jobs,
            "enabled_state_snapshot_jobs": enabled_state_snapshot_jobs,
            "conversation_compaction_jobs": conversation_compaction_jobs,
            "enabled_conversation_compaction_jobs": enabled_conversation_compaction_jobs,
            "job_types": job_types,
            "risky_tools": risky_tools,
            "missing_required": missing_required,
            "optional_attention": optional_attention,
            "storage": storage,
            "model_route": model_route,
        }

    def readiness_report(_: dict[str, Any]) -> ToolResult:
        context = _current_readiness_context()
        risky_tools = context["risky_tools"]
        pending = context["pending"][:10]
        enabled_jobs = context["enabled_jobs"]
        state_snapshot_jobs = context["state_snapshot_jobs"]
        enabled_state_snapshot_jobs = context["enabled_state_snapshot_jobs"]
        conversation_compaction_jobs = context["conversation_compaction_jobs"]
        enabled_conversation_compaction_jobs = context["enabled_conversation_compaction_jobs"]
        scheduled_jobs = context["scheduled_jobs"]
        readable_scheduled_jobs = context["readable_scheduled_jobs"]
        unreadable_scheduled_job_rows = context["unreadable_scheduled_job_rows"]
        job_types = context["job_types"]
        storage = context["storage"]
        model_route = context["model_route"]
        recent_runs = store.recent_tool_runs(limit=10)
        readable_recent_runs, unreadable_recent_run_rows = _readable_tool_run_rows(recent_runs)

        checks: list[tuple[str, bool, str]] = []
        sqlite_store_exception_type = ""
        checks.append(("Obsidian Jarvis root", vault.root_path.exists(), safe_storage_text(vault.root_path)))
        try:
            store.list_sessions(1)
            checks.append(("SQLite memory store", True, safe_storage_text(config.db_path) if config else "configured"))
        except Exception as exc:
            sqlite_store_exception_type = type(exc).__name__
            checks.append(
                (
                    "SQLite memory store",
                    False,
                    f"unavailable; set JARVIS_DATA_DIR and JARVIS_DB_PATH to writable local paths, run `{BOOTSTRAP_CHECK_COMMAND}`, then retry",
                )
            )
        checks.append(("SQLite storage parent writable", _metadata_bool(storage.get("db_parent_writable")), str(storage["db_parent"])))
        checks.append(("SQLite database file writable", _metadata_bool(storage.get("db_file_writable")), str(storage["db_path"])))
        checks.append(("Jarvis note vault writable", _metadata_bool(storage.get("obsidian_vault_writable")), str(storage["obsidian_vault"])))
        checks.append(("Jarvis note root writable", _metadata_bool(storage.get("obsidian_root_writable")), str(storage["obsidian_root_path"])))
        runtime_fallback_active = _metadata_bool(storage.get("runtime_fallback_active"))
        checks.append(("Runtime storage fallback", not runtime_fallback_active, "inactive" if not runtime_fallback_active else "active; using workspace-local fallback storage"))

        mission_control = vault.root_path / "Automations" / "Mission Control.md"
        current_context = vault.root_path / "Memory Tree" / "Current Context.md"
        checks.append(("Current Context note", current_context.exists(), safe_storage_text(current_context)))
        checks.append(("Mission Control note", mission_control.exists(), safe_storage_text(mission_control)))
        checks.append(("Approval queue visible", True, f"{len(pending)} pending"))
        checks.append(("Risk gates registered", bool(risky_tools), f"{len(risky_tools)} risky tool(s) gated"))
        state_snapshot_detail = (
            "list scheduled jobs"
            if enabled_state_snapshot_jobs
            else "resume job State Snapshot"
            if state_snapshot_jobs
            else "schedule assistant basics"
        )
        checks.append(("State Snapshot enabled", bool(enabled_state_snapshot_jobs), state_snapshot_detail))
        conversation_compaction_detail = (
            "list scheduled jobs"
            if enabled_conversation_compaction_jobs
            else f"resume job {COMPACTION_JOB_NAME}"
            if conversation_compaction_jobs
            else "schedule assistant basics"
        )
        checks.append(
            (
                "Conversation Compaction enabled",
                bool(enabled_conversation_compaction_jobs),
                conversation_compaction_detail,
            )
        )
        checks.append(("pyautogui optional dependency", _module_available("pyautogui"), "needed only for computer control"))
        checks.append((model_route["label"], model_route["ready"], model_route["detail"]))
        if model_route["provider"] == "ollama":
            checks.append(
                (
                    "Ollama personalized context",
                    model_route["ollama_personalized_context_ready"],
                    _ollama_no_cloud_detail(
                        ollama_local_only_policy(config.chat_model if config is not None else ""),
                        route_reachable=model_route["ready"],
                    ),
                )
            )

        failed_recent, approval_held_recent = _recent_tool_run_attention_buckets(readable_recent_runs)
        approval_held_review_commands = _first_approval_held_review_commands(approval_held_recent)
        approval_held_review_next_command = approval_held_review_commands[0] if approval_held_review_commands else ""
        approval_held_review_approval_id = (
            approval_held_review_next_command.removeprefix("approval readiness ")
            if approval_held_review_next_command.startswith("approval readiness ")
            else ""
        )
        blocking_labels = {
            "Obsidian Jarvis root",
            "SQLite memory store",
            "Approval queue visible",
            "Risk gates registered",
        }
        if model_route["provider"] == "invalid":
            blocking_labels.add(model_route["label"])
        if model_route["provider"] == "ollama" and not model_route["ollama_destination_allowed"]:
            blocking_labels.add(model_route["label"])
        blocking_issues = [label for label, ok, _ in checks if not ok and label in blocking_labels]
        optional_issues = [label for label, ok, _ in checks if not ok and label not in blocking_labels]
        if unreadable_recent_run_rows:
            optional_issues.append("Recent tool-run audit")
        if unreadable_scheduled_job_rows:
            optional_issues.append("Scheduled job table")

        if blocking_issues:
            readiness = "not ready"
        elif pending or optional_issues:
            readiness = "ready with attention"
        else:
            readiness = "ready"
        readiness_gate_state = "BLOCKED_REQUIRED_FIXES" if blocking_issues else "REVIEW_PENDING_APPROVALS" if pending else "CLEAR_FOR_READ_ONLY_AND_LOCAL_SAFE"
        auto_run_ceiling = "READ_ONLY_LOCAL_SAFE"
        approval_required_risk_levels = ["PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"]
        blocked_action_classes = [
            "computer control",
            "shell/code execution",
            "personal data reads",
            "external side effects",
            "destructive or risky actions",
        ]
        gate_commands = ["pending approvals", "approval readiness <id>", "approval packet <id>", "approval chain proof <id>"]
        conversation_latency_target_ms = 8000
        conversation_latency_required_turns = 10
        conversation_latency_required_chat_samples = 3
        conversation_latency_next_commands = [
            "mixed conversation proof matrix",
            "model routing status",
            "live_check",
        ]
        conversation_latency_tuning_knobs = [
            "JARVIS_CHAT_MAX_REPLY_TOKENS",
            "JARVIS_CHAT_MAX_HISTORY_MESSAGES",
        ]
        chat_max_reply_tokens = config.chat_max_reply_tokens if config else 300
        chat_max_history_messages = config.chat_max_history_messages if config else 16

        lines = [
            "Jarvis readiness report:",
            f"- status: {readiness}",
            "- safe-use rule: read-only and local-safe actions may run; personal-data, external-side-effect, and high-risk actions require explicit approval",
            "- stop-window rule: priority goals do not override the operator's explicit stop times, work windows, pause commands, or newer instructions",
            "",
            "Checks:",
        ]
        for label, ok, detail in checks:
            state = "ok" if ok else "needs attention"
            lines.append(f"- {label}: {state} | {detail}")

        lines.extend(["", "Current blockers:"])
        if pending:
            for row in pending[:5]:
                approval_id_value = _row_value(row, "id")
                if approval_id_value is None:
                    lines.append("- Approval details unavailable: one pending row could not be read safely.")
                    continue
                approval_id = _safe_readiness_text(approval_id_value, default="unknown", limit=80)
                tool_name = _row_text(row, "tool_name", default="unknown tool")
                user_input = _row_text(row, "user_input", default="input unavailable")
                lines.append(f"- Approval #{approval_id} {tool_name}: {user_input}")
                lines.append(f"  Readiness: `approval readiness {approval_id}`")
                lines.append(f"  Last look: `approval packet {approval_id}`")
                lines.append(f"  Proof chain: `approval chain proof {approval_id}`")
        else:
            lines.append("- No pending approvals.")
        if unreadable_recent_run_rows:
            lines.append(
                f"- Recent tool-run audit has {unreadable_recent_run_rows} unreadable row(s); inspect with `recent tool runs`."
            )
        if unreadable_scheduled_job_rows:
            lines.append(
                f"- Scheduled job table has {unreadable_scheduled_job_rows} unreadable row(s); inspect with `list scheduled jobs`."
            )
        if failed_recent:
            lines.append("- Recent failed tool runs exist; inspect with `recent tool runs`.")
        if approval_held_recent:
            lines.append(_approval_held_review_line(approval_held_review_commands))

        lines.extend(["", "Recommended safe next steps:"])
        if not storage["available"]:
            lines.append(
                "- Point `JARVIS_DATA_DIR` and `JARVIS_DB_PATH` at a writable local folder, "
                f"then run `{BOOTSTRAP_CHECK_COMMAND}`."
            )
            lines.append(
                f"- Run `{BOOTSTRAP_WRITE_COMMAND}` only after the check reports configured storage ready."
            )
        elif storage["runtime_fallback_active"]:
            if storage["recovery_mode"] == "restart_runtime_to_configured_storage":
                lines.append("- restart or reload Jarvis with the configured durable storage envs, then run `storage status`.")
                lines.append(f"- Before the restart, run `{STORAGE_RECOVERY_CHECK_COMMAND}` to confirm configured storage is ready.")
            else:
                lines.append(
                    f"- Review `{STORAGE_RECOVERY_PLAN_COMMAND}` after `storage status`, then run "
                    f"`{STORAGE_RECOVERY_CHECK_COMMAND}` after changing storage envs."
                )
                lines.append(f"- Run `{BOOTSTRAP_WRITE_COMMAND}` only after the no-write check reports configured storage ready.")
        if not mission_control.exists():
            lines.append("- Run `mission control` to write the safe next-action note.")
        if not enabled_state_snapshot_jobs and state_snapshot_jobs:
            lines.append("- Run `resume job State Snapshot` so Current Context and Mission Control stay fresh.")
        elif not enabled_state_snapshot_jobs:
            lines.append("- Run `schedule assistant basics` so Current Context and Mission Control stay fresh.")
        if not enabled_conversation_compaction_jobs and conversation_compaction_jobs:
            lines.append(f"- Run `resume job {COMPACTION_JOB_NAME}` so old conversations become durable Memory Trees digests.")
        elif not enabled_conversation_compaction_jobs:
            lines.append("- Run `schedule assistant basics` so old conversations become durable Memory Trees digests.")
        if pending:
            lines.append("- Run `pending approvals`, then `approval readiness #ID`, then `approval packet #ID`, then `approval chain proof #ID`; approve only requests you trust.")
        if optional_issues:
            lines.append("- Run `jarvis doctor` for setup details on optional items.")
        if not pending and not optional_issues and "state_snapshot" in job_types:
            lines.append("- Start with `safe next actions` or `daily plan`.")
        storage_readiness_next_commands = list(storage["readiness_next_commands"])
        storage_readiness_proof_queue = list(storage["readiness_proof_queue"])
        storage_readiness_next_required_command = storage_readiness_next_commands[0] if storage_readiness_next_commands else ""
        storage_readiness_next_proof_command = storage_readiness_next_required_command
        storage_readiness_first_proof_command = storage_readiness_proof_queue[0] if storage_readiness_proof_queue else ""
        lines.extend(
            [
                "",
                "Storage readiness proof:",
                f"- next required: `{storage_readiness_next_required_command}`" if storage_readiness_next_required_command else "- next required: none",
                "- proof queue: "
                + (
                    " -> ".join(f"`{command}`" for command in storage_readiness_proof_queue)
                    if storage_readiness_proof_queue
                    else "none"
                ),
            ]
        )
        lines.extend(
            [
                "",
                "Conversation latency proof:",
                (
                    "- status: proof required until a fresh mixed conversation stays coherent and under "
                    f"~{conversation_latency_target_ms // 1000}s/turn"
                ),
                (
                    f"- target: {conversation_latency_required_turns}+ turns across chat/research/calendar/tasks "
                    f"with at least {conversation_latency_required_chat_samples} chat latency sample(s)"
                ),
                (
                    f"- current runtime knobs: reply token cap {chat_max_reply_tokens}; "
                    f"history window {chat_max_history_messages} message(s)"
                ),
                "- next proof path: "
                + " -> ".join(f"`{command}`" for command in conversation_latency_next_commands),
                (
                    "- tuning boundary: adjust JARVIS_CHAT_MAX_REPLY_TOKENS or "
                    "JARVIS_CHAT_MAX_HISTORY_MESSAGES only with the operator, then rerun the proof and `live_check`"
                ),
            ]
        )
        lines.extend(
            [
                "",
                "Structured readiness gate:",
                f"- gate state: {readiness_gate_state}",
                f"- auto-run ceiling: {auto_run_ceiling}",
                "- operator limits: explicit stop times, work windows, pause commands, and newer instructions override priority goals",
                "- approval-required risks: " + ", ".join(approval_required_risk_levels),
                "- blocked action classes: " + ", ".join(blocked_action_classes),
                "- review command chain: " + " -> ".join(f"`{command}`" for command in gate_commands),
            ]
        )

        return ToolResult(
            "readiness_report",
            True,
            "\n".join(lines),
            _safe_metadata(
                status=readiness,
                blocking_issues=len(blocking_issues),
                optional_issues=len(optional_issues),
                pending_approvals=len(pending),
                risky_tools=len(risky_tools),
                scheduled_jobs=len(scheduled_jobs),
                readable_scheduled_jobs=len(readable_scheduled_jobs),
                unreadable_scheduled_job_rows=unreadable_scheduled_job_rows,
                enabled_jobs=len(enabled_jobs),
                state_snapshot_jobs=len(state_snapshot_jobs),
                enabled_state_snapshot_jobs=len(enabled_state_snapshot_jobs),
                disabled_state_snapshot_jobs=len(state_snapshot_jobs) - len(enabled_state_snapshot_jobs),
                state_snapshot_next_command=state_snapshot_detail,
                conversation_compaction_jobs=len(conversation_compaction_jobs),
                enabled_conversation_compaction_jobs=len(enabled_conversation_compaction_jobs),
                disabled_conversation_compaction_jobs=(
                    len(conversation_compaction_jobs) - len(enabled_conversation_compaction_jobs)
                ),
                conversation_compaction_next_command=conversation_compaction_detail,
                readiness_gate_state=readiness_gate_state,
                auto_run_ceiling=auto_run_ceiling,
                approval_required_risk_levels=approval_required_risk_levels,
                blocked_action_classes=blocked_action_classes,
                gate_commands=gate_commands,
                conversation_latency_status="proof_required",
                conversation_latency_target_ms=conversation_latency_target_ms,
                conversation_latency_required_turns=conversation_latency_required_turns,
                conversation_latency_required_chat_samples=conversation_latency_required_chat_samples,
                conversation_latency_next_commands=conversation_latency_next_commands,
                conversation_latency_next_command_count=len(conversation_latency_next_commands),
                conversation_latency_next_required_command=conversation_latency_next_commands[0],
                conversation_latency_tuning_knobs=conversation_latency_tuning_knobs,
                chat_max_reply_tokens=chat_max_reply_tokens,
                chat_max_history_messages=chat_max_history_messages,
                model_provider=model_route["provider"],
                model_provider_valid=model_route["provider_valid"],
                model_route_ready=model_route["ready"],
                model_route_live_access_probed=model_route["live_access_probed"],
                ollama_required=model_route["ollama_required"],
                openai_api_key_configured=(
                    model_route["ready"] if model_route["provider"] == "openai" else False
                ),
                openai_api_key_value_exposed=False,
                openai_safety_identifier_sent=model_route["safety_identifier_sent"],
                openai_safety_identifier_scope=model_route["safety_identifier_scope"],
                openai_safety_identifier_value_exposed=model_route["safety_identifier_value_exposed"],
                openai_safety_identifier_uses_personal_data=model_route["safety_identifier_uses_personal_data"],
                openai_safety_identifier_uses_api_key=model_route["safety_identifier_uses_api_key"],
                openai_request_store_flag=model_route["request_store_flag"],
                openai_account_retention_controls_checked=model_route["account_retention_controls_checked"],
                openai_zero_data_retention_verified=model_route["zero_data_retention_verified"],
                openai_default_abuse_monitoring_may_retain_content=model_route["default_abuse_monitoring_may_retain_content"],
                openai_default_abuse_monitoring_max_days=model_route["default_abuse_monitoring_max_days"],
                storage_status=storage["status"],
                storage_configured_status=storage["configured_status"],
                storage_available=storage["available"],
                storage_configured_available=storage["configured_available"],
                storage_ready_for_completion_claim=storage["ready_for_completion_claim"],
                storage_runtime_fallback_active=storage["runtime_fallback_active"],
                storage_runtime_fallback_reason=storage["runtime_fallback_reason"],
                storage_runtime_fallback_exception_type=storage["runtime_fallback_exception_type"],
                storage_runtime_fallback_db_path_display=storage["runtime_fallback_db_path_display"],
                storage_runtime_fallback_vault_path_display=storage["runtime_fallback_vault_path_display"],
                storage_recovery_mode=storage["recovery_mode"],
                storage_recovery_next_operator_action=storage["recovery_next_operator_action"],
                storage_readiness_next_commands=storage_readiness_next_commands,
                storage_readiness_next_command_count=len(storage_readiness_next_commands),
                storage_readiness_next_required_command=storage_readiness_next_required_command,
                storage_readiness_next_proof_command=storage_readiness_next_proof_command,
                storage_readiness_proof_queue=storage_readiness_proof_queue,
                storage_readiness_proof_queue_count=len(storage_readiness_proof_queue),
                storage_readiness_first_proof_command=storage_readiness_first_proof_command,
                storage_data_dir=storage["data_dir"],
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
                storage_recovery_check_command=storage.get("recovery_check_command", ""),
                storage_recovery_check_api=storage.get("recovery_check_api", STORAGE_RECOVERY_CHECK_API),
                storage_recovery_command=storage.get("recovery_command", ""),
                sqlite_memory_store_available=not sqlite_store_exception_type,
                sqlite_memory_store_exception_type=sqlite_store_exception_type,
                recent_tool_runs=len(recent_runs),
                readable_recent_tool_runs=len(readable_recent_runs),
                unreadable_recent_tool_run_rows=unreadable_recent_run_rows,
                recent_failed_runs=len(failed_recent),
                recent_approval_held_runs=len(approval_held_recent),
                approval_held_review_required=bool(approval_held_recent),
                approval_held_review_commands=approval_held_review_commands,
                approval_held_review_command_count=len(approval_held_review_commands),
                approval_held_review_next_command=approval_held_review_next_command,
                approval_held_review_approval_id=approval_held_review_approval_id,
                **_ollama_route_metadata(model_route),
            ),
        )

    def prototype_readiness_checklist(_: dict[str, Any]) -> ToolResult:
        context = _current_readiness_context()
        pending = context["pending"]
        missing_required = context["missing_required"]
        optional_attention = context["optional_attention"]
        model_route = context["model_route"]
        scheduled_jobs = context["scheduled_jobs"]
        readable_scheduled_jobs = context["readable_scheduled_jobs"]
        unreadable_scheduled_job_rows = context["unreadable_scheduled_job_rows"]
        recent_runs = store.recent_tool_runs(limit=10)
        readable_recent_runs, unreadable_recent_run_rows = _readable_tool_run_rows(recent_runs)
        failed_recent, approval_held_recent = _recent_tool_run_attention_buckets(readable_recent_runs)
        approval_held_review_commands = _first_approval_held_review_commands(approval_held_recent)
        approval_held_review_next_command = approval_held_review_commands[0] if approval_held_review_commands else ""
        approval_held_review_approval_id = (
            approval_held_review_next_command.removeprefix("approval readiness ")
            if approval_held_review_next_command.startswith("approval readiness ")
            else ""
        )
        optional_attention_display = []
        for item in optional_attention:
            if item == "scheduled job table has unreadable row(s)" and unreadable_scheduled_job_rows:
                optional_attention_display.append(
                    f"scheduled job table has {unreadable_scheduled_job_rows} unreadable row(s); inspect with `list scheduled jobs`"
                )
            else:
                optional_attention_display.append(item)

        safe_to_try = [
            "Talk naturally in the Jarvis GUI with typed messages.",
            "Use speech recognition from the browser microphone control when permission is granted.",
            "Ask for memory search, recent memories, chat context, and saved preferences.",
            "Ask for readiness, safety status, risk matrix, tool search, and tool detail.",
            "Capture local-safe Jarvis-owned notes, tasks, goals, preferences, people, and decisions.",
            "Preview risky actions with command diagnosis, risk preflight, action readiness, approval readiness, last-look approval packets, and approval chain proof.",
        ]
        approval_gated = [
            "Shell/code execution and scripts.",
            "Desktop control: observe screen, click, type, move mouse, or screenshots.",
            "Clipboard reads, focused/running app activity, personal account data, email/calendar style integrations, and private files.",
            "External side effects such as sending messages, opening sites for action, or modifying outside systems.",
            "Destructive or ambiguous file operations.",
        ]
        prototype_commands = [
            "prototype readiness",
            "readiness report",
            "setup check",
            "safety status",
            "risk matrix",
            "tool search: computer",
            "tool detail: run_shell_command",
            "command diagnosis: run command python3 --version",
            "voice setup check",
            "computer control status",
            "computer task plan: open settings and verify the target panel is visible",
            "chat response health",
            "task board",
            "approval summary",
            "recent tool runs",
        ]

        lines = [
            "Jarvis prototype readiness checklist",
            f"Status: {context['status']}",
            "",
            "Safe to try now:",
        ]
        lines.extend(f"- {item}" for item in safe_to_try)
        lines.extend(["", "Approval-gated before real execution:"])
        lines.extend(f"- {item}" for item in approval_gated)
        lines.extend(
            [
                "",
                "Operator limits:",
                "- Priority goals do not override the operator's explicit stop times, work windows, pause commands, or newer instructions.",
            ]
        )
        lines.extend(["", "Needs attention before Jarvis feels fully alive:"])
        if missing_required:
            lines.extend(f"- Required: {item}" for item in missing_required)
        else:
            lines.append("- Required basics are present: memory store, Jarvis vault path, and risk gates.")
        if optional_attention_display:
            lines.extend(f"- Optional/setup: {item}" for item in optional_attention_display)
        else:
            lines.append("- Optional setup checks are clear.")
        if pending:
            lines.append(f"- Pending approval queue has {len(pending)} item(s); review before approving real execution.")
        else:
            lines.append("- No pending approvals.")
        if unreadable_recent_run_rows:
            lines.append(
                f"- Recent tool-run audit has {unreadable_recent_run_rows} unreadable row(s); inspect with `recent tool runs`."
            )
        if failed_recent:
            lines.append("- Recent failed tool runs exist; inspect with `recent tool runs`.")
        if approval_held_recent:
            lines.append(_approval_held_review_line(approval_held_review_commands))
        lines.extend(["", "Prototype commands to try:"])
        lines.extend(f"- {command}" for command in prototype_commands)
        lines.extend(
            [
                "",
                "Safety boundary:",
                "- This checklist is read-only. It does not call models, execute tools, queue approvals, approve requests, dismiss approvals, read private data, write files, control the computer, or speak.",
            ]
        )

        return ToolResult(
            "prototype_readiness_checklist",
            True,
            "\n".join(lines),
            _safe_metadata(
                status=context["status"],
                safe_to_try_count=len(safe_to_try),
                approval_gated_count=len(approval_gated),
                prototype_commands=prototype_commands,
                pending_approvals=len(pending),
                missing_required=missing_required,
                optional_attention=optional_attention_display,
                model_provider=model_route["provider"],
                model_provider_valid=model_route["provider_valid"],
                model_route_ready=model_route["ready"],
                model_route_live_access_probed=model_route["live_access_probed"],
                ollama_required=model_route["ollama_required"],
                openai_api_key_configured=(
                    model_route["ready"] if model_route["provider"] == "openai" else False
                ),
                openai_api_key_value_exposed=False,
                openai_safety_identifier_sent=model_route["safety_identifier_sent"],
                openai_safety_identifier_scope=model_route["safety_identifier_scope"],
                openai_safety_identifier_value_exposed=model_route["safety_identifier_value_exposed"],
                openai_safety_identifier_uses_personal_data=model_route["safety_identifier_uses_personal_data"],
                openai_safety_identifier_uses_api_key=model_route["safety_identifier_uses_api_key"],
                openai_request_store_flag=model_route["request_store_flag"],
                openai_account_retention_controls_checked=model_route["account_retention_controls_checked"],
                openai_zero_data_retention_verified=model_route["zero_data_retention_verified"],
                openai_default_abuse_monitoring_may_retain_content=model_route["default_abuse_monitoring_may_retain_content"],
                openai_default_abuse_monitoring_max_days=model_route["default_abuse_monitoring_max_days"],
                scheduled_jobs=len(scheduled_jobs),
                readable_scheduled_jobs=len(readable_scheduled_jobs),
                unreadable_scheduled_job_rows=unreadable_scheduled_job_rows,
                recent_tool_runs=len(recent_runs),
                readable_recent_tool_runs=len(readable_recent_runs),
                unreadable_recent_tool_run_rows=unreadable_recent_run_rows,
                recent_failed_runs=len(failed_recent),
                recent_approval_held_runs=len(approval_held_recent),
                approval_held_review_required=bool(approval_held_recent),
                approval_held_review_commands=approval_held_review_commands,
                approval_held_review_command_count=len(approval_held_review_commands),
                approval_held_review_next_command=approval_held_review_next_command,
                approval_held_review_approval_id=approval_held_review_approval_id,
                **_ollama_route_metadata(model_route),
            ),
        )

    return readiness_report, prototype_readiness_checklist
