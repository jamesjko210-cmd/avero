from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable

from jarvis_v2.agent.types import ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.v3_commands import V3_BOOTSTRAP_CHECK_COMMAND, V3_BOOTSTRAP_WRITE_COMMAND


LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
BOOTSTRAP_CHECK_COMMAND = V3_BOOTSTRAP_CHECK_COMMAND
BOOTSTRAP_WRITE_COMMAND = V3_BOOTSTRAP_WRITE_COMMAND
STORAGE_RECOVERY_CHECK_COMMAND = "storage recovery check"
STORAGE_RECOVERY_PLAN_COMMAND = "storage recovery plan"
STORAGE_RECOVERY_CHECK_API = "/api/storage-recovery-check"
WORKSPACE_DURABLE_DIR = ".jarvis_v3_durable"


def safe_storage_text(value: object) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", str(value or ""))


def storage_diagnostics(config: JarvisConfig | None) -> dict[str, Any]:
    if config is None:
        return {
            "available": False,
            "status": "unknown",
            "data_dir": "",
            "db_path": "",
            "db_parent": "",
            "db_exists": False,
            "data_dir_exists": False,
            "db_parent_exists": False,
            "data_dir_writable": False,
            "db_parent_writable": False,
            "db_file_writable": False,
            "metadata_only": True,
            "issue": "configuration unavailable",
            "recovery_check_command": BOOTSTRAP_CHECK_COMMAND,
            "recovery_check_api": STORAGE_RECOVERY_CHECK_API,
            "recovery_command": BOOTSTRAP_WRITE_COMMAND,
        }

    data_dir = config.data_dir.expanduser()
    db_path = config.db_path.expanduser()
    obsidian_vault = config.obsidian_vault.expanduser()
    obsidian_root_path = obsidian_vault / config.obsidian_root
    db_parent = db_path.parent
    data_dir_exists = data_dir.exists()
    db_parent_exists = db_parent.exists()
    db_exists = db_path.exists()
    obsidian_vault_exists = obsidian_vault.exists()
    obsidian_root_exists = obsidian_root_path.exists()
    data_dir_writable = _path_writable(data_dir) if data_dir_exists else _path_writable(data_dir.parent)
    db_parent_writable = _path_writable(db_parent) if db_parent_exists else _path_writable(db_parent.parent)
    db_file_writable = _path_writable(db_path) if db_exists else db_parent_writable
    obsidian_vault_writable = _path_writable(obsidian_vault) if obsidian_vault_exists else _path_writable(obsidian_vault.parent)
    obsidian_root_writable = _path_writable(obsidian_root_path) if obsidian_root_exists else obsidian_vault_writable

    issues: list[str] = []
    if not db_parent_exists:
        issues.append("database parent does not exist")
    if not db_parent_writable:
        issues.append("database parent is not writable")
    if db_exists and not db_file_writable:
        issues.append("database file is not writable")
    if not obsidian_vault_exists:
        issues.append("Obsidian vault does not exist")
    if not obsidian_vault_writable:
        issues.append("Obsidian vault is not writable")

    return {
        "available": not issues,
        "status": "ready" if not issues else "needs attention",
        "data_dir": safe_storage_text(data_dir),
        "db_path": safe_storage_text(db_path),
        "db_parent": safe_storage_text(db_parent),
        "db_exists": db_exists,
        "data_dir_exists": data_dir_exists,
        "db_parent_exists": db_parent_exists,
        "data_dir_writable": data_dir_writable,
        "db_parent_writable": db_parent_writable,
        "db_file_writable": db_file_writable,
        "obsidian_vault": safe_storage_text(obsidian_vault),
        "obsidian_root": config.obsidian_root,
        "obsidian_root_path": safe_storage_text(obsidian_root_path),
        "obsidian_vault_exists": obsidian_vault_exists,
        "obsidian_root_exists": obsidian_root_exists,
        "obsidian_vault_writable": obsidian_vault_writable,
        "obsidian_root_writable": obsidian_root_writable,
        "workspace_local_notes": obsidian_vault == data_dir / "Vault",
        "metadata_only": True,
        "issues": issues,
        "recovery_check_command": BOOTSTRAP_CHECK_COMMAND,
        "recovery_check_api": STORAGE_RECOVERY_CHECK_API,
        "recovery_command": BOOTSTRAP_WRITE_COMMAND,
    }


def _configured_storage_diagnostics(
    config: JarvisConfig | None,
    fallback: dict[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    primary_diagnostics = (fallback or {}).get("primary_storage_diagnostics")
    if isinstance(primary_diagnostics, dict):
        return primary_diagnostics, "primary_storage_before_fallback"
    return storage_diagnostics(config), "runtime_config"


def _path_writable(path: Path) -> bool:
    try:
        return path.exists() and os.access(path, os.W_OK)
    except OSError:
        return False


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "calls_model": False,
        "calls_external_service": False,
        "executes_tools": False,
        "queues_approval": False,
        "approves_request": False,
        "dismisses_request": False,
        "controls_computer": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "reads_secret_values": False,
        "reads_env_file_contents": False,
        "reads_db_file_contents": False,
        "reads_clipboard": False,
        "scans_data_dir": False,
        "scans_obsidian_vault": False,
        "writes_files": False,
        "edits_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "requires_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "speaks": False,
        "completes_tasks": False,
    }
    metadata.update(extra)
    return metadata


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _storage_metadata_only_boundaries() -> dict[str, bool]:
    return {
        "metadata_only": True,
        "reads_database_file": False,
        "reads_db_file_contents": False,
        "reads_vault_files": False,
        "scans_obsidian_vault": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "approves_request": False,
        "approves_requests": False,
        "dismisses_request": False,
        "dismisses_approvals": False,
        "calls_model": False,
        "calls_external_service": False,
        "controls_computer": False,
        "executes_tools": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def make_storage_status_tool(
    config: JarvisConfig | None,
    storage_fallback: Callable[[], dict[str, Any] | None] | None = None,
):
    def storage_status(_: dict[str, Any]) -> ToolResult:
        fallback = storage_fallback() if storage_fallback else None
        diagnostics, diagnostics_source = _configured_storage_diagnostics(config, fallback)
        fallback_active = bool(fallback)
        configured_ready = _metadata_bool(diagnostics.get("available"))
        active_route = (
            "workspace-local fallback"
            if fallback_active
            else "configured storage"
            if configured_ready
            else "configured storage needs attention"
        )
        issues = list(diagnostics.get("issues") or [])
        if fallback_active:
            issues.append("runtime is using workspace-local fallback storage")
        storage_status_value = "needs attention" if issues else str(diagnostics.get("status", "unknown"))
        storage_available = bool(configured_ready or fallback_active)
        ready_for_completion_claim = configured_ready and not fallback_active
        recovery_required = bool(issues)
        recovery_reason = "; ".join(issues) if recovery_required else ""
        if fallback_active and configured_ready:
            storage_recovery_mode = "restart_runtime_to_configured_storage"
            storage_recovery_next_operator_action = (
                "restart or reload Jarvis with the configured durable storage envs, then run `storage status`"
            )
            storage_recovery_restart_required = True
        elif fallback_active:
            storage_recovery_mode = "repair_configured_storage_then_restart_runtime"
            storage_recovery_next_operator_action = (
                "review the storage recovery plan, point Jarvis at writable durable storage, "
                "run the no-write storage check, then restart or reload Jarvis"
            )
            storage_recovery_restart_required = True
        elif not configured_ready:
            storage_recovery_mode = "repair_configured_storage"
            storage_recovery_next_operator_action = (
                "review the storage recovery plan, point Jarvis at writable durable storage, "
                "and run the no-write storage check"
            )
            storage_recovery_restart_required = False
        else:
            storage_recovery_mode = "none"
            storage_recovery_next_operator_action = ""
            storage_recovery_restart_required = False
        storage_readiness_blocks_completion_claim = not ready_for_completion_claim
        storage_readiness_blocker = recovery_reason if storage_readiness_blocks_completion_claim else ""
        if storage_readiness_blocks_completion_claim and storage_recovery_mode == "restart_runtime_to_configured_storage":
            storage_readiness_next_commands = ["storage status", STORAGE_RECOVERY_CHECK_COMMAND, "storage status"]
        elif storage_readiness_blocks_completion_claim:
            storage_readiness_next_commands = [
                "storage status",
                STORAGE_RECOVERY_PLAN_COMMAND,
                STORAGE_RECOVERY_CHECK_COMMAND,
                BOOTSTRAP_CHECK_COMMAND,
                BOOTSTRAP_WRITE_COMMAND,
            ]
        else:
            storage_readiness_next_commands = []
        storage_readiness_proof_queue = list(storage_readiness_next_commands)
        storage_readiness_proof_queue_preview = storage_readiness_proof_queue[:5]
        storage_readiness_proof_queue_remaining_count = max(
            len(storage_readiness_proof_queue) - len(storage_readiness_proof_queue_preview),
            0,
        )
        storage_readiness_next_required_command = (
            storage_readiness_next_commands[0] if storage_readiness_next_commands else ""
        )
        recovery_envs = ["JARVIS_DATA_DIR", "JARVIS_DB_PATH", "JARVIS_OBSIDIAN_VAULT"]
        next_commands = list(storage_readiness_next_commands)
        for helper_command in ["setup check", "jarvis doctor", "readiness report"]:
            if helper_command not in next_commands:
                next_commands.append(helper_command)
        if recovery_required and storage_recovery_mode == "restart_runtime_to_configured_storage":
            pass
        elif recovery_required:
            pass
        storage_status_handoff = {
            "source": "storage_status",
            "handoff_ready": True,
            "storage_status_handoff_ready": True,
            "ready_for_operator": True,
            "state_changed": False,
            "changed": [],
            "content_in_handoff": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "storage_status": storage_status_value,
            "storage_available": storage_available,
            "storage_ready_for_completion_claim": ready_for_completion_claim,
            "storage_active_route": active_route,
            "storage_runtime_fallback_active": fallback_active,
            "storage_recovery_required": recovery_required,
            "storage_recovery_reason": recovery_reason,
            "storage_recovery_mode": storage_recovery_mode,
            "storage_recovery_next_operator_action": storage_recovery_next_operator_action,
            "storage_recovery_restart_required": storage_recovery_restart_required,
            "storage_readiness_blocks_completion_claim": storage_readiness_blocks_completion_claim,
            "storage_readiness_blocker": storage_readiness_blocker,
            "storage_readiness_next_commands": list(storage_readiness_next_commands),
            "storage_readiness_next_command_count": len(storage_readiness_next_commands),
            "storage_readiness_next_required_command": storage_readiness_next_required_command,
            "storage_readiness_next_proof_command": storage_readiness_next_required_command,
            "storage_readiness_proof_queue": list(storage_readiness_proof_queue),
            "storage_readiness_proof_queue_count": len(storage_readiness_proof_queue),
            "storage_readiness_proof_queue_preview": list(storage_readiness_proof_queue_preview),
            "storage_readiness_proof_queue_preview_count": len(storage_readiness_proof_queue_preview),
            "storage_readiness_proof_queue_remaining_count": storage_readiness_proof_queue_remaining_count,
            "storage_readiness_first_proof_command": storage_readiness_proof_queue[0] if storage_readiness_proof_queue else "",
            "storage_recovery_envs": list(recovery_envs),
            "next_commands": list(next_commands),
            "issue_count": len(issues),
            "issues": list(issues),
            "boundaries": _storage_metadata_only_boundaries(),
        }

        lines = [
            "Jarvis storage status:",
            f"- route: {active_route}",
            f"- storage readiness: {storage_status_value}",
            f"- configured storage status: {diagnostics.get('status', 'unknown')}",
            f"- runtime fallback active: {'yes' if fallback_active else 'no'}",
            f"- ready for completion claim: {'yes' if ready_for_completion_claim else 'no'}",
            f"- database display: `{diagnostics.get('db_path', '')}`",
            f"- database parent writable: {'yes' if _metadata_bool(diagnostics.get('db_parent_writable')) else 'no'}",
            f"- database file writable: {'yes' if _metadata_bool(diagnostics.get('db_file_writable')) else 'no'}",
            f"- note vault display: `{diagnostics.get('obsidian_vault', '')}`",
            f"- note root display: `{diagnostics.get('obsidian_root_path', '')}`",
            f"- note vault writable: {'yes' if _metadata_bool(diagnostics.get('obsidian_vault_writable')) else 'no'}",
            f"- workspace-local notes: {'yes' if _metadata_bool(diagnostics.get('workspace_local_notes')) else 'no'}",
            f"- metadata-only check: {'yes' if _metadata_bool(diagnostics.get('metadata_only')) else 'no'}",
            "",
            "Issues:",
        ]
        if issues:
            lines.extend(f"- {issue}" for issue in issues)
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "Recovery:",
                f"- Recovery required before completion claim: {'yes' if recovery_required else 'no'}.",
                f"- Recovery reason: {recovery_reason or 'none'}.",
                f"- Recovery mode: {storage_recovery_mode}.",
                f"- Next operator action: {storage_recovery_next_operator_action or 'none'}.",
                "- Set `JARVIS_DATA_DIR`, `JARVIS_DB_PATH`, or `JARVIS_OBSIDIAN_VAULT` to writable durable locations when the fallback is active or storage needs attention.",
                f"- Check first with `{STORAGE_RECOVERY_CHECK_COMMAND}` after changing storage envs.",
                f"- Shell equivalent: `{diagnostics.get('recovery_check_command', BOOTSTRAP_CHECK_COMMAND)}`.",
                f"- Run `{diagnostics.get('recovery_command', BOOTSTRAP_WRITE_COMMAND)}` only after the check reports configured storage ready.",
                f"- Proof ladder: {', '.join(f'`{command}`' for command in storage_readiness_proof_queue) if storage_readiness_proof_queue else 'none'}.",
                "",
                "Boundary:",
                "- This is read-only and metadata-only. It does not read database contents, scan note contents, create folders, write files, queue approvals, approve requests, call models, call external services, or control the computer.",
            ]
        )
        return ToolResult(
            "storage_status",
            True,
            "\n".join(lines),
            _safe_metadata(
                storage_status=storage_status_value,
                storage_configured_status=diagnostics.get("status", "unknown"),
                storage_available=storage_available,
                storage_configured_available=_metadata_bool(diagnostics.get("available")),
                storage_configured_diagnostics_source=diagnostics_source,
                storage_ready_for_completion_claim=ready_for_completion_claim,
                storage_active_route=active_route,
                storage_runtime_fallback_active=fallback_active,
                storage_runtime_fallback_reason=str((fallback or {}).get("reason") or ""),
                storage_runtime_fallback_exception_type=str((fallback or {}).get("exception_type") or ""),
                storage_runtime_fallback_db_path_display=str((fallback or {}).get("db_path_display") or ""),
                storage_runtime_fallback_vault_path_display=str((fallback or {}).get("vault_path_display") or ""),
                storage_data_dir=safe_storage_text(diagnostics.get("data_dir", "")),
                storage_db_path=safe_storage_text(diagnostics.get("db_path", "")),
                storage_db_parent=safe_storage_text(diagnostics.get("db_parent", "")),
                storage_db_exists=_metadata_bool(diagnostics.get("db_exists")),
                storage_data_dir_exists=_metadata_bool(diagnostics.get("data_dir_exists")),
                storage_db_parent_exists=_metadata_bool(diagnostics.get("db_parent_exists")),
                storage_data_dir_writable=_metadata_bool(diagnostics.get("data_dir_writable")),
                storage_db_parent_writable=_metadata_bool(diagnostics.get("db_parent_writable")),
                storage_db_file_writable=_metadata_bool(diagnostics.get("db_file_writable")),
                storage_obsidian_vault=safe_storage_text(diagnostics.get("obsidian_vault", "")),
                storage_obsidian_root=diagnostics.get("obsidian_root", ""),
                storage_obsidian_root_path=safe_storage_text(diagnostics.get("obsidian_root_path", "")),
                storage_obsidian_vault_exists=_metadata_bool(diagnostics.get("obsidian_vault_exists")),
                storage_obsidian_root_exists=_metadata_bool(diagnostics.get("obsidian_root_exists")),
                storage_obsidian_vault_writable=_metadata_bool(diagnostics.get("obsidian_vault_writable")),
                storage_obsidian_root_writable=_metadata_bool(diagnostics.get("obsidian_root_writable")),
                storage_workspace_local_notes=_metadata_bool(diagnostics.get("workspace_local_notes")),
                storage_metadata_only=_metadata_bool(diagnostics.get("metadata_only")),
                storage_issues=issues,
                storage_issue_count=len(issues),
                storage_recovery_command=diagnostics.get("recovery_command", ""),
                storage_recovery_check_command=diagnostics.get("recovery_check_command", ""),
                storage_recovery_check_api=diagnostics.get("recovery_check_api", STORAGE_RECOVERY_CHECK_API),
                storage_recovery_check_tool_command=STORAGE_RECOVERY_CHECK_COMMAND,
                storage_recovery_required=recovery_required,
                storage_recovery_reason=recovery_reason,
                storage_recovery_mode=storage_recovery_mode,
                storage_recovery_next_operator_action=storage_recovery_next_operator_action,
                storage_recovery_restart_required=storage_recovery_restart_required,
                storage_readiness_blocks_completion_claim=storage_readiness_blocks_completion_claim,
                storage_readiness_blocker=storage_readiness_blocker,
                storage_readiness_next_commands=storage_readiness_next_commands,
                storage_readiness_next_command_count=len(storage_readiness_next_commands),
                storage_readiness_next_required_command=storage_readiness_next_required_command,
                storage_readiness_next_proof_command=storage_readiness_next_required_command,
                storage_readiness_proof_queue=storage_readiness_proof_queue,
                storage_readiness_proof_queue_count=len(storage_readiness_proof_queue),
                storage_readiness_proof_queue_preview=storage_readiness_proof_queue_preview,
                storage_readiness_proof_queue_preview_count=len(storage_readiness_proof_queue_preview),
                storage_readiness_proof_queue_remaining_count=storage_readiness_proof_queue_remaining_count,
                storage_readiness_first_proof_command=storage_readiness_proof_queue[0] if storage_readiness_proof_queue else "",
                storage_recovery_envs=recovery_envs,
                storage_status_handoff=storage_status_handoff,
                storage_status_handoff_ready=storage_status_handoff["handoff_ready"],
                storage_status_ready_for_operator=storage_status_handoff["ready_for_operator"],
                storage_status_state_changed=storage_status_handoff["state_changed"],
                storage_status_changed=storage_status_handoff["changed"],
                storage_status_content_in_handoff=storage_status_handoff["content_in_handoff"],
                storage_status_authorizes_execution=storage_status_handoff["authorizes_execution"],
                storage_status_authorizes_completion_claim=storage_status_handoff["authorizes_completion_claim"],
                storage_status_approval_granted=storage_status_handoff["approval_granted"],
                storage_status_boundaries=storage_status_handoff["boundaries"],
                next_commands=next_commands,
            ),
        )

    return storage_status


def make_storage_recovery_check_tool(
    config: JarvisConfig | None,
    storage_fallback: Callable[[], dict[str, Any] | None] | None = None,
):
    def storage_recovery_check(_: dict[str, Any]) -> ToolResult:
        fallback = storage_fallback() if storage_fallback else None
        diagnostics, diagnostics_source = _configured_storage_diagnostics(config, fallback)
        fallback_active = bool(fallback)
        configured_ready = _metadata_bool(diagnostics.get("available"))
        check_passed = configured_ready and not fallback_active
        configured_issues = list(diagnostics.get("issues") or [])
        blocker_parts: list[str] = []
        fallback_blocker = "runtime is using workspace-local fallback storage"
        if not configured_ready:
            blocker_parts.extend(configured_issues or ["configured storage needs attention"])
        if fallback_active:
            blocker_parts.append(fallback_blocker)
        storage_issues = list(blocker_parts)
        blocker = "; ".join(blocker_parts)
        storage_status_value = "needs attention" if storage_issues else str(diagnostics.get("status", "unknown"))
        storage_available = bool(configured_ready or fallback_active)
        ready_for_completion_claim = check_passed
        active_route = (
            "workspace-local fallback"
            if fallback_active
            else "configured storage"
            if configured_ready
            else "configured storage needs attention"
        )
        if fallback_active and configured_ready:
            storage_recovery_mode = "restart_runtime_to_configured_storage"
            storage_recovery_next_operator_action = (
                "restart or reload Jarvis with the configured durable storage envs, then run `storage status`"
            )
            storage_recovery_restart_required = True
        elif fallback_active:
            storage_recovery_mode = "repair_configured_storage_then_restart_runtime"
            storage_recovery_next_operator_action = (
                "review the storage recovery plan, point Jarvis at writable durable storage, "
                "rerun this no-write check, then restart or reload Jarvis"
            )
            storage_recovery_restart_required = True
        elif not configured_ready:
            storage_recovery_mode = "repair_configured_storage"
            storage_recovery_next_operator_action = (
                "review the storage recovery plan, point Jarvis at writable durable storage, "
                "and rerun this no-write check"
            )
            storage_recovery_restart_required = False
        else:
            storage_recovery_mode = "none"
            storage_recovery_next_operator_action = ""
            storage_recovery_restart_required = False
        if check_passed:
            next_commands = [BOOTSTRAP_WRITE_COMMAND, "storage status"]
        elif storage_recovery_mode == "restart_runtime_to_configured_storage":
            next_commands = ["storage status", "setup check"]
        else:
            next_commands = [
                "storage status",
                STORAGE_RECOVERY_PLAN_COMMAND,
                STORAGE_RECOVERY_CHECK_COMMAND,
                BOOTSTRAP_CHECK_COMMAND,
                BOOTSTRAP_WRITE_COMMAND,
                "setup check",
            ]
        storage_readiness_next_commands = next_commands[:4] if not check_passed else []
        storage_readiness_proof_queue = list(storage_readiness_next_commands)
        storage_readiness_proof_queue_preview = storage_readiness_proof_queue[:5]
        storage_readiness_proof_queue_remaining_count = max(
            len(storage_readiness_proof_queue) - len(storage_readiness_proof_queue_preview),
            0,
        )
        storage_readiness_next_required_command = (
            storage_readiness_next_commands[0] if storage_readiness_next_commands else ""
        )
        storage_recovery_check_boundaries = _storage_metadata_only_boundaries()
        storage_recovery_check_handoff = {
            "source": "storage_recovery_check",
            "handoff_ready": True,
            "storage_recovery_check_handoff_ready": True,
            "ready_for_operator": True,
            "state_changed": False,
            "changed": [],
            "content_in_handoff": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "storage_recovery_check_passed": check_passed,
            "storage_status": storage_status_value,
            "storage_available": storage_available,
            "storage_ready_for_completion_claim": ready_for_completion_claim,
            "storage_active_route": active_route,
            "storage_configured_status": diagnostics.get("status", "unknown"),
            "storage_configured_available": configured_ready,
            "storage_configured_diagnostics_source": diagnostics_source,
            "storage_runtime_fallback_active": fallback_active,
            "storage_recovery_required": not check_passed,
            "storage_recovery_reason": blocker,
            "storage_recovery_blocker": blocker,
            "storage_recovery_mode": storage_recovery_mode,
            "storage_recovery_next_operator_action": storage_recovery_next_operator_action,
            "storage_recovery_restart_required": storage_recovery_restart_required,
            "storage_readiness_blocks_completion_claim": not check_passed,
            "storage_readiness_blocker": blocker,
            "storage_readiness_next_commands": storage_readiness_next_commands,
            "storage_readiness_next_command_count": len(storage_readiness_next_commands),
            "storage_readiness_next_required_command": storage_readiness_next_required_command,
            "storage_readiness_next_proof_command": storage_readiness_next_required_command,
            "storage_readiness_proof_queue": storage_readiness_proof_queue,
            "storage_readiness_proof_queue_count": len(storage_readiness_proof_queue),
            "storage_readiness_proof_queue_preview": list(storage_readiness_proof_queue_preview),
            "storage_readiness_proof_queue_preview_count": len(storage_readiness_proof_queue_preview),
            "storage_readiness_proof_queue_remaining_count": storage_readiness_proof_queue_remaining_count,
            "storage_readiness_first_proof_command": storage_readiness_proof_queue[0] if storage_readiness_proof_queue else "",
            "storage_recovery_check_tool_command": STORAGE_RECOVERY_CHECK_COMMAND,
            "storage_recovery_check_command": BOOTSTRAP_CHECK_COMMAND,
            "storage_recovery_check_api": STORAGE_RECOVERY_CHECK_API,
            "storage_recovery_check_shell_command": BOOTSTRAP_CHECK_COMMAND,
            "storage_recovery_command": BOOTSTRAP_WRITE_COMMAND,
            "next_commands": next_commands,
            "issues": storage_issues,
            "issue_count": len(storage_issues),
            "boundaries": storage_recovery_check_boundaries,
        }
        lines = [
            "Jarvis storage recovery check:",
            "- This is the Jarvis-native no-write check before durable storage bootstrap.",
            f"- route: {active_route}",
            f"- storage readiness: {storage_status_value}",
            f"- configured storage ready: {'yes' if configured_ready else 'no'}",
            f"- runtime fallback active: {'yes' if fallback_active else 'no'}",
            f"- ready for completion claim: {'yes' if ready_for_completion_claim else 'no'}",
            f"- recovery check passed: {'yes' if check_passed else 'no'}",
            f"- blocker: {blocker or 'none'}",
            f"- recovery mode: {storage_recovery_mode}",
            f"- next operator action: {storage_recovery_next_operator_action or 'none'}",
            f"- shell equivalent: `{BOOTSTRAP_CHECK_COMMAND}`",
            "",
            "Next commands:",
        ]
        lines.extend(f"- `{command}`" for command in next_commands)
        lines.extend(
            [
                f"- Proof ladder: {', '.join(f'`{command}`' for command in storage_readiness_proof_queue) if storage_readiness_proof_queue else 'none'}.",
                "",
                "Boundary:",
                "- This is read-only and metadata-only. It does not create folders, initialize SQLite, read database contents, scan note contents, import memory, queue approvals, call models, call external services, or control the computer.",
            ]
        )
        return ToolResult(
            "storage_recovery_check",
            True,
            "\n".join(lines),
            _safe_metadata(
                storage_status=storage_status_value,
                storage_available=storage_available,
                storage_ready_for_completion_claim=ready_for_completion_claim,
                storage_active_route=active_route,
                storage_recovery_check_passed=check_passed,
                storage_recovery_check_tool_command=STORAGE_RECOVERY_CHECK_COMMAND,
                storage_recovery_check_command=BOOTSTRAP_CHECK_COMMAND,
                storage_recovery_check_api=STORAGE_RECOVERY_CHECK_API,
                storage_recovery_check_shell_command=BOOTSTRAP_CHECK_COMMAND,
                storage_recovery_command=BOOTSTRAP_WRITE_COMMAND,
                storage_recovery_required=not check_passed,
                storage_recovery_reason=blocker,
                storage_recovery_blocker=blocker,
                storage_recovery_mode=storage_recovery_mode,
                storage_recovery_next_operator_action=storage_recovery_next_operator_action,
                storage_recovery_restart_required=storage_recovery_restart_required,
                storage_readiness_blocks_completion_claim=not check_passed,
                storage_readiness_blocker=blocker,
                storage_readiness_next_commands=storage_readiness_next_commands,
                storage_readiness_next_command_count=len(storage_readiness_next_commands),
                storage_readiness_next_required_command=storage_readiness_next_required_command,
                storage_readiness_next_proof_command=storage_readiness_next_required_command,
                storage_readiness_proof_queue=storage_readiness_proof_queue,
                storage_readiness_proof_queue_count=len(storage_readiness_proof_queue),
                storage_readiness_proof_queue_preview=storage_readiness_proof_queue_preview,
                storage_readiness_proof_queue_preview_count=len(storage_readiness_proof_queue_preview),
                storage_readiness_proof_queue_remaining_count=storage_readiness_proof_queue_remaining_count,
                storage_readiness_first_proof_command=storage_readiness_proof_queue[0] if storage_readiness_proof_queue else "",
                storage_runtime_fallback_active=fallback_active,
                storage_runtime_fallback_reason=str((fallback or {}).get("reason") or ""),
                storage_runtime_fallback_exception_type=str((fallback or {}).get("exception_type") or ""),
                storage_runtime_fallback_db_path_display=str((fallback or {}).get("db_path_display") or ""),
                storage_runtime_fallback_vault_path_display=str((fallback or {}).get("vault_path_display") or ""),
                storage_configured_status=diagnostics.get("status", "unknown"),
                storage_configured_available=configured_ready,
                storage_configured_diagnostics_source=diagnostics_source,
                storage_data_dir=safe_storage_text(diagnostics.get("data_dir", "")),
                storage_db_path=safe_storage_text(diagnostics.get("db_path", "")),
                storage_db_parent=safe_storage_text(diagnostics.get("db_parent", "")),
                storage_db_exists=_metadata_bool(diagnostics.get("db_exists")),
                storage_data_dir_exists=_metadata_bool(diagnostics.get("data_dir_exists")),
                storage_db_parent_exists=_metadata_bool(diagnostics.get("db_parent_exists")),
                storage_data_dir_writable=_metadata_bool(diagnostics.get("data_dir_writable")),
                storage_db_parent_writable=_metadata_bool(diagnostics.get("db_parent_writable")),
                storage_db_file_writable=_metadata_bool(diagnostics.get("db_file_writable")),
                storage_obsidian_vault=safe_storage_text(diagnostics.get("obsidian_vault", "")),
                storage_obsidian_root=diagnostics.get("obsidian_root", ""),
                storage_obsidian_root_path=safe_storage_text(diagnostics.get("obsidian_root_path", "")),
                storage_obsidian_vault_exists=_metadata_bool(diagnostics.get("obsidian_vault_exists")),
                storage_obsidian_root_exists=_metadata_bool(diagnostics.get("obsidian_root_exists")),
                storage_obsidian_vault_writable=_metadata_bool(diagnostics.get("obsidian_vault_writable")),
                storage_obsidian_root_writable=_metadata_bool(diagnostics.get("obsidian_root_writable")),
                storage_workspace_local_notes=_metadata_bool(diagnostics.get("workspace_local_notes")),
                storage_metadata_only=_metadata_bool(diagnostics.get("metadata_only")),
                storage_issues=storage_issues,
                storage_issue_count=len(storage_issues),
                storage_recovery_check_handoff=storage_recovery_check_handoff,
                storage_recovery_check_handoff_ready=storage_recovery_check_handoff["handoff_ready"],
                storage_recovery_check_ready_for_operator=storage_recovery_check_handoff["ready_for_operator"],
                storage_recovery_check_state_changed=storage_recovery_check_handoff["state_changed"],
                storage_recovery_check_changed=storage_recovery_check_handoff["changed"],
                storage_recovery_check_content_in_handoff=storage_recovery_check_handoff["content_in_handoff"],
                storage_recovery_check_authorizes_execution=storage_recovery_check_handoff["authorizes_execution"],
                storage_recovery_check_authorizes_completion_claim=storage_recovery_check_handoff["authorizes_completion_claim"],
                storage_recovery_check_approval_granted=storage_recovery_check_handoff["approval_granted"],
                storage_recovery_check_boundaries=storage_recovery_check_handoff["boundaries"],
                next_commands=next_commands,
            ),
        )

    return storage_recovery_check


def make_storage_recovery_plan_tool(
    config: JarvisConfig | None,
    storage_fallback: Callable[[], dict[str, Any] | None] | None = None,
    cwd_provider: Callable[[], Path] | None = None,
):
    def storage_recovery_plan(_: dict[str, Any]) -> ToolResult:
        fallback = storage_fallback() if storage_fallback else None
        diagnostics, diagnostics_source = _configured_storage_diagnostics(config, fallback)
        fallback_active = bool(fallback)
        configured_ready = _metadata_bool(diagnostics.get("available"))
        parent = (cwd_provider or Path.cwd)()
        parent_exists = parent.exists()
        parent_writable = _path_writable(parent)
        proposed_data_dir = WORKSPACE_DURABLE_DIR
        proposed_db_path = f"{WORKSPACE_DURABLE_DIR}/jarvis.sqlite"
        proposed_vault_path = f"{WORKSPACE_DURABLE_DIR}/Vault"
        can_bootstrap = parent_exists and parent_writable
        recovery_reason_parts = list(diagnostics.get("issues") or [])
        if fallback_active:
            recovery_reason_parts.append("runtime is using workspace-local fallback storage")
        recovery_reason = "; ".join(recovery_reason_parts)
        next_commands = [
            STORAGE_RECOVERY_CHECK_COMMAND,
            BOOTSTRAP_CHECK_COMMAND,
            BOOTSTRAP_WRITE_COMMAND,
            "storage status",
        ]
        export_lines = [
            f'export JARVIS_DATA_DIR="$PWD/{proposed_data_dir}"',
            f'export JARVIS_DB_PATH="$PWD/{proposed_db_path}"',
            f'export JARVIS_OBSIDIAN_VAULT="$PWD/{proposed_vault_path}"',
        ]
        boundaries = _storage_metadata_only_boundaries()
        handoff = {
            "source": "storage_recovery_plan",
            "handoff_ready": True,
            "storage_recovery_plan_handoff_ready": True,
            "ready_for_operator": True,
            "state_changed": False,
            "changed": [],
            "content_in_handoff": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "storage_recovery_plan_ready": can_bootstrap,
            "storage_recovery_plan_parent_exists": parent_exists,
            "storage_recovery_plan_parent_writable": parent_writable,
            "storage_recovery_plan_data_dir": proposed_data_dir,
            "storage_recovery_plan_db_path": proposed_db_path,
            "storage_recovery_plan_obsidian_vault": proposed_vault_path,
            "storage_recovery_plan_exports": list(export_lines),
            "storage_recovery_plan_next_commands": list(next_commands),
            "storage_recovery_plan_next_command_count": len(next_commands),
            "storage_recovery_plan_next_required_command": STORAGE_RECOVERY_CHECK_COMMAND,
            "storage_recovery_plan_reason": recovery_reason,
            "storage_runtime_fallback_active": fallback_active,
            "storage_configured_available": configured_ready,
            "storage_configured_diagnostics_source": diagnostics_source,
            "boundaries": boundaries,
        }
        lines = [
            "Jarvis storage recovery plan:",
            "- This is read-only. It drafts a concrete local durable-storage env plan without creating folders or changing the runtime.",
            f"- configured storage ready now: {'yes' if configured_ready else 'no'}",
            f"- runtime fallback active now: {'yes' if fallback_active else 'no'}",
            f"- current project directory writable: {'yes' if parent_writable else 'no'}",
            f"- plan ready to bootstrap: {'yes' if can_bootstrap else 'no'}",
            f"- current blocker: {recovery_reason or 'none'}",
            "",
            "Suggested env for a project-local durable store:",
        ]
        lines.extend(f"- `{line}`" for line in export_lines)
        lines.extend(
            [
                "",
                "Then run:",
            ]
        )
        lines.extend(f"- `{command}`" for command in next_commands)
        lines.extend(
            [
                "",
                "Boundary:",
                "- This does not create the directory, initialize SQLite, import memory, write env files, restart Jarvis, approve requests, read database contents, scan note contents, call models, call external services, or control the computer.",
            ]
        )
        return ToolResult(
            "storage_recovery_plan",
            True,
            "\n".join(lines),
            _safe_metadata(
                storage_recovery_plan_ready=can_bootstrap,
                storage_recovery_plan_parent_exists=parent_exists,
                storage_recovery_plan_parent_writable=parent_writable,
                storage_recovery_plan_data_dir=proposed_data_dir,
                storage_recovery_plan_db_path=proposed_db_path,
                storage_recovery_plan_obsidian_vault=proposed_vault_path,
                storage_recovery_plan_exports=export_lines,
                storage_recovery_plan_next_commands=next_commands,
                storage_recovery_plan_next_command_count=len(next_commands),
                storage_recovery_plan_next_required_command=STORAGE_RECOVERY_CHECK_COMMAND,
                storage_recovery_plan_next_proof_command=STORAGE_RECOVERY_CHECK_COMMAND,
                storage_recovery_plan_reason=recovery_reason,
                storage_recovery_plan_handoff=handoff,
                storage_recovery_plan_handoff_ready=handoff["handoff_ready"],
                storage_recovery_plan_ready_for_operator=handoff["ready_for_operator"],
                storage_recovery_plan_state_changed=handoff["state_changed"],
                storage_recovery_plan_changed=handoff["changed"],
                storage_recovery_plan_content_in_handoff=handoff["content_in_handoff"],
                storage_recovery_plan_authorizes_execution=handoff["authorizes_execution"],
                storage_recovery_plan_authorizes_completion_claim=handoff["authorizes_completion_claim"],
                storage_recovery_plan_approval_granted=handoff["approval_granted"],
                storage_recovery_plan_boundaries=boundaries,
                storage_runtime_fallback_active=fallback_active,
                storage_configured_available=configured_ready,
                storage_configured_diagnostics_source=diagnostics_source,
                storage_recovery_required=bool(recovery_reason_parts),
                storage_recovery_check_tool_command=STORAGE_RECOVERY_CHECK_COMMAND,
                storage_recovery_check_command=BOOTSTRAP_CHECK_COMMAND,
                storage_recovery_command=BOOTSTRAP_WRITE_COMMAND,
                next_commands=next_commands,
            ),
        )

    return storage_recovery_plan
