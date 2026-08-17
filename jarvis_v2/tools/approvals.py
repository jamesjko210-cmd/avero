from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    RESOURCE_NOT_FOUND_RECOVERY_ACTION,
    declare_resource_not_found_failure,
    declare_retryable_local_read_failure,
)
from jarvis_v2.agent.types import ToolResult
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import MemoryStore, approval_action_digest


APPROVAL_READINESS_RECEIPT_TTL_SECONDS = 10 * 60
APPROVAL_ID_RECOVERY_ACTION = (
    "Run `pending approvals`, choose a valid approval number, then retry through the normal policy."
)
APPROVAL_REVIEW_RECOVERY_ACTION = (
    "Refresh approval readiness and review the exact packet before making any new decision."
)


@dataclass(frozen=True)
class _ApprovalReadinessReceipt:
    approval_id: int
    queue_revision: int
    target_updated_at: str
    action_digest: str
    expires_at_monotonic: float


RISK_CLUES = {
    "run_shell_command": (
        "shell/code execution",
        "It can inspect or modify the machine depending on the command.",
        "Read the command, prefer a narrow command, and avoid shell control operators.",
    ),
    "write_text_file": (
        "file write",
        "It can create or overwrite local files.",
        "Confirm the exact path and content before approving.",
    ),
    "list_files": (
        "personal file metadata read",
        "Filenames and folder names can reveal private projects, accounts, documents, and habits.",
        "Use Jarvis-owned note listing when possible, or narrow the folder before approving.",
    ),
    "find_files": (
        "personal file metadata read",
        "Search results can reveal private filenames and folder structure.",
        "Use the narrowest folder and filename pattern possible before approving.",
    ),
    "read_text_file": (
        "personal file content read",
        "File contents may contain private notes, tokens, messages, credentials, or confidential work.",
        "Prefer Jarvis-owned note readers or paste only the needed excerpt into chat when possible.",
    ),
    "clear_obsidian_inbox": (
        "Obsidian content reset",
        "It clears reviewed inbox text from Jarvis-owned notes.",
        "Make sure the inbox was ingested or manually reviewed first.",
    ),
    "get_clipboard": (
        "personal data read",
        "Clipboard contents may contain passwords, tokens, private messages, or copied documents.",
        "Paste only the needed text into chat instead when possible.",
    ),
    "enable_computer_control": (
        "computer control",
        "It allows later mouse and keyboard actions after explicit tool requests.",
        "Use `computer control status` and an `autonomy plan: ...` first.",
    ),
    "observe_screen": (
        "screen observation",
        "Screenshots may expose private apps, files, accounts, or messages.",
        "Close private windows or describe the target manually if possible.",
    ),
    "verify_screen": (
        "screen observation",
        "Screenshots may expose private apps, files, accounts, or messages.",
        "Close private windows, confirm the expectation, and approve only the needed observation.",
    ),
    "observe_act_verify": (
        "computer control",
        "It observes the screen, performs one primitive action, then observes again for verification.",
        "Use `computer readiness: ...` and inspect exact coordinates/text, expectation, and stop conditions first.",
    ),
    "click": (
        "computer control",
        "It can click the wrong app, button, account, or destructive control.",
        "Approve only one exact coordinate/button after a fresh screen observation and expected result are clear.",
    ),
    "type_text": (
        "computer control",
        "It can type private or incorrect text into the wrong field or app.",
        "Approve only exact text and target field after a fresh screen observation confirms focus.",
    ),
    "move_mouse": (
        "computer control",
        "It moves the pointer and can set up a later click in the wrong place.",
        "Approve only exact coordinates after the target screen layout is known.",
    ),
    "screenshot": (
        "screen capture",
        "Screenshots may expose private apps, files, accounts, or messages.",
        "Crop or describe only the relevant area when possible.",
    ),
    "create_reminder": (
        "external/local side effect",
        "It creates a real macOS Reminder outside Jarvis memory.",
        "Use a Jarvis task first if a system reminder is not necessary.",
    ),
    "apple_reminders": (
        "personal Apple Reminders read",
        "It reads private reminder content from the shared macOS account.",
        "Open Reminders yourself and keep it open before approving; Jarvis will not launch the app. "
        "Narrow the read to one exact list and bounded limit when possible. If an approved attempt "
        "fails, that one-shot approval is consumed; fix the app or permission state and submit a "
        "fresh bounded request instead of replaying it.",
    ),
}

OPERATOR_LIMIT_RULE = "the operator's explicit stop times, work windows, pause commands, and newer instructions override pending approvals, approved-looking packets, and priority goals."
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")
_MISSING_ROW_VALUE = object()


def _bounded_int(value: Any, default: int, low: int, high: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _short(value: Any, limit: int = 64) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _short_metadata(value: Any, limit: int = 64) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", _short(value, limit))


def _display_text(value: Any, limit: int | None = None) -> str:
    text = str(value or "")
    if limit is not None and len(text) > limit:
        text = text[: max(0, limit - 3)].rstrip() + "..."
    return LOCAL_PATH_RE.sub("<local-path>", text)


def _display_value(value: Any) -> Any:
    if isinstance(value, str):
        return _display_text(value)
    if isinstance(value, dict):
        return {str(key): _display_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_display_value(item) for item in value]
    return value


def _row_value(row: Any, key: str, default: Any = _MISSING_ROW_VALUE) -> Any:
    try:
        return row[key]
    except Exception:
        return default


def _row_text(row: Any, key: str, default: str = "", *, limit: int = 160) -> str:
    value = _row_value(row, key)
    if value is _MISSING_ROW_VALUE:
        return default
    text = _display_text(value, limit)
    return text if text else default


def _row_raw_text(row: Any, key: str, default: str = "") -> str:
    value = _row_value(row, key)
    if value is _MISSING_ROW_VALUE:
        return default
    return str(value or default)


def _row_int(row: Any, key: str) -> int | None:
    value = _row_value(row, key)
    if value is _MISSING_ROW_VALUE or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_approval_payload(row: Any) -> dict[str, Any] | None:
    approval_id = _row_int(row, "id")
    if approval_id is None:
        return None
    return {
        "id": approval_id,
        "status": _row_text(row, "status", "unknown", limit=64),
        "tool_name": _row_text(row, "tool_name", "unknown_tool", limit=96),
        "user_input": _row_text(row, "user_input", "Unreadable request", limit=500),
        "reason": _row_text(row, "reason", "", limit=500),
        "created_at": _row_text(row, "created_at", "", limit=64),
        "updated_at": _row_text(row, "updated_at", "", limit=64),
        "planned_args": _row_raw_text(row, "planned_args", "{}"),
    }


def _safe_approval_payloads(rows: list[Any]) -> tuple[list[dict[str, Any]], int]:
    approvals: list[dict[str, Any]] = []
    unreadable = 0
    for row in rows:
        approval = _safe_approval_payload(row)
        if approval is None:
            unreadable += 1
        else:
            approvals.append(approval)
    return approvals, unreadable


def _safe_vault_path_display(path: Path | str | None, vault: ObsidianVault) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return str(candidate.relative_to(vault.root_path))
    except ValueError:
        return _short_metadata(candidate, 96)


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
        "controls_computer": False,
        "reads_private_data": False,
        "writes_files": False,
        "writes_notes": False,
        "writes_memory": False,
        "external_side_effect": False,
        "requires_approval": False,
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
    }
    metadata.update(extra)
    return metadata


def _bad_approval_id_metadata(args: dict[str, Any]) -> dict[str, Any]:
    raw = args.get("approval_id")
    if raw is None:
        raw = args.get("id")
    if raw is None:
        raw = args.get("target")
    return _safe_metadata(reason="bad_approval_id", approval_id=None, raw_approval_id=_short_metadata(raw, 80))


def _bad_approval_id_result(
    tool_name: str,
    args: dict[str, Any],
    error: str | None,
) -> ToolResult:
    output = (
        f"{error or 'approval_id must be a number or `latest`.'} "
        f"{APPROVAL_ID_RECOVERY_ACTION}"
    )
    return ToolResult(
        tool_name,
        False,
        output,
        declare_retryable_local_read_failure(
            _bad_approval_id_metadata(args),
            output=output,
            action=APPROVAL_ID_RECOVERY_ACTION,
            commands=("pending approvals",),
        ),
    )


def _approval_write_metadata(**extra: Any) -> dict[str, Any]:
    metadata = _safe_metadata()
    metadata.update({"writes_files": True, "writes_notes": True})
    metadata.update(extra)
    return metadata


def _approval_review_refusal_result(
    tool_name: str,
    output: str,
    metadata: dict[str, Any],
    *,
    commands: tuple[str, ...],
) -> ToolResult:
    public_output = f"{output}\n{APPROVAL_REVIEW_RECOVERY_ACTION}"
    return ToolResult(
        tool_name,
        False,
        public_output,
        declare_retryable_local_read_failure(
            metadata,
            output=public_output,
            action=APPROVAL_REVIEW_RECOVERY_ACTION,
            commands=commands,
        ),
    )


def _approval_not_found_result(
    tool_name: str,
    approval_id: int,
    retry_command: str,
    *,
    pending_only: bool = True,
    **extra_metadata: Any,
) -> ToolResult:
    missing_label = "pending approval" if pending_only else "approval"
    detail_command = f"approval detail {approval_id}"
    recovery_commands = ["pending approvals", detail_command]
    if retry_command != detail_command:
        recovery_commands.append(retry_command)
        recovery_text = f"then use `{detail_command}` to confirm its status before retrying `{retry_command}`."
    else:
        recovery_text = f"then retry `{detail_command}` with the refreshed queue."
    output = (
        f"No {missing_label} found for #{approval_id}. Run `pending approvals` to refresh the queue, "
        + recovery_text
        + f" {RESOURCE_NOT_FOUND_RECOVERY_ACTION}"
    )
    metadata = declare_resource_not_found_failure(
        _safe_metadata(
            reason="not_found",
            approval_id=approval_id,
            next_command="pending approvals",
            recovery_commands=recovery_commands,
            retry_requires_queue_refresh=True,
            authorizes_retry=False,
            **extra_metadata,
        ),
        output=output,
        action=RESOURCE_NOT_FOUND_RECOVERY_ACTION,
    )
    return ToolResult(
        tool_name,
        False,
        output,
        metadata,
    )


def _approval_update_failure_result(
    tool_name: str,
    approval_id: int,
    action: str,
    *,
    exception_type: str = "",
    actual_status: str = "pending",
) -> ToolResult:
    recovery_commands = ["pending approvals", f"approval detail {approval_id}", "setup check"]
    review_guidance = ""
    if action == "approve":
        recovery_commands.extend([f"approval readiness {approval_id}", f"approval packet {approval_id}"])
        review_guidance = (
            f" If it remains pending, review `approval readiness {approval_id}` and "
            f"`approval packet {approval_id}` again before deciding whether to retry."
        )
    return ToolResult(
        tool_name,
        False,
        f"Jarvis could not {action} approval #{approval_id} because the approval queue update did not "
        f"complete. Run `pending approvals`, then `approval detail {approval_id}`; if the queue cannot "
        f"be read, run `setup check`.{review_guidance} No approval decision was recorded.",
        _safe_metadata(
            reason=f"{action}_failed",
            approval_id=approval_id,
            failure_stage="approval_status_update",
            exception_type=exception_type,
            actual_status=actual_status,
            next_command="pending approvals",
            recovery_commands=recovery_commands,
            approval_decision_recorded=False,
            retry_requires_queue_refresh=True,
            retry_requires_fresh_review=action == "approve",
            authorizes_retry=False,
        ),
    )


def _approval_cas_miss_result(
    tool_name: str,
    approval_id: int,
    action: str,
    actual_status: str,
) -> ToolResult:
    status = actual_status.strip().lower() or "unknown"
    if status == "pending":
        return _approval_update_failure_result(
            tool_name,
            approval_id,
            action,
            actual_status=status,
        )
    recovery_commands = ["pending approvals", f"approval detail {approval_id}"]
    if status == "approved":
        recovery_commands.extend(
            [f"approval resume packet {approval_id}", f"approval packet {approval_id}"]
        )
        next_command = f"approval resume packet {approval_id}"
        recovery_text = (
            f"Run `approval resume packet {approval_id}` to inspect the approved request and its "
            "one-shot execution state."
        )
    else:
        next_command = f"approval detail {approval_id}"
        recovery_text = f"Run `approval detail {approval_id}` to inspect the recorded decision."
    return ToolResult(
        tool_name,
        False,
        f"Approval #{approval_id} is already {status}; this {action} attempt did not overwrite the "
        f"winning decision. {recovery_text}",
        _safe_metadata(
            reason=f"{action}_cas_miss",
            approval_id=approval_id,
            failure_stage="approval_status_compare_and_set",
            actual_status=status,
            winning_status=status,
            approval_decision_recorded=status in {"approved", "dismissed"},
            next_command=next_command,
            recovery_commands=recovery_commands,
            retry_requires_queue_refresh=True,
            authorizes_retry=False,
        ),
    )


def _already_dismissed_result(tool_name: str, approval_id: int) -> ToolResult:
    return ToolResult(
        tool_name,
        True,
        f"Approval #{approval_id} was already dismissed; no change was made.",
        _safe_metadata(
            reason="already_dismissed",
            approval_id=approval_id,
            actual_status="dismissed",
            winning_status="dismissed",
            approval_decision_recorded=True,
            approval_state_changed=False,
            idempotent_noop=True,
            dismisses_request=True,
            authorizes_retry=False,
        ),
    )


def _risk_review_for(tool_name: str, user_input: str) -> tuple[str, str, str]:
    if tool_name in RISK_CLUES:
        return RISK_CLUES[tool_name]
    low = (tool_name + " " + user_input).lower()
    if any(word in low for word in ("delete", "clear", "remove", "overwrite")):
        return (
            "destructive change",
            "It may remove or replace local state.",
            "Confirm there is a backup or use a read-only inspection command first.",
        )
    if any(word in low for word in ("click", "type", "mouse", "keyboard")):
        return (
            "computer control",
            "It can change the visible app or type into the wrong place.",
            "Observe the screen and approve one small action at a time.",
        )
    return (
        "approval-gated action",
        "It exceeds Jarvis' auto-run safety ceiling.",
        "Approve only if the request, target, and expected result are clear.",
    )


COMPUTER_APPROVAL_TOOLS = {
    "enable_computer_control",
    "observe_screen",
    "verify_screen",
    "observe_act_verify",
    "screenshot",
    "click",
    "type_text",
    "move_mouse",
}


def _computer_approval_checklist(tool_name: str, user_input: str) -> list[str]:
    if tool_name not in COMPUTER_APPROVAL_TOOLS:
        return []
    objective = user_input.strip() or "complete the computer-control request"
    return [
        "Computer-control approval checklist:",
        f"- Run first: `computer readiness: {objective}`",
        f"- Then run: `computer task plan: {objective}`",
        "- Confirm the visible target app/account/file is the one the operator intended.",
        "- Confirm the request is one primitive step, not a batch of clicks/typing.",
        "- Confirm exact coordinates, button/text, and the expected after-state for any action.",
        "- Stop if private messages, passwords, tokens, payment pages, destructive controls, or the wrong account are visible.",
    ]


def _planned_args_lines(row) -> list[str]:
    planned_args = _planned_args(row)
    if not planned_args:
        return []
    lines = ["Planned arguments:"]
    for key in sorted(planned_args):
        lines.append(f"- {_display_text(key)}: {_approval_arg_display(key, planned_args[key])}")
    return lines


def _approval_arg_display(key: str, value: Any) -> Any:
    if key == "_kakao_send_binding":
        return "bound to exact reviewed Kakao recipient, message, and target mode"
    if key == "_instagram_send_binding":
        return "bound to exact reviewed Instagram @username, message, and recipient mode"
    if key.endswith("_binding") or key == "target_binding":
        return "bound to reviewed memory version"
    if key == "owner_fingerprint":
        return "bound to configured Telegram owner"
    if key == "due_epoch":
        try:
            return datetime.fromtimestamp(float(value)).astimezone().isoformat(timespec="seconds")
        except (TypeError, ValueError, OverflowError, OSError):
            return "invalid due time"
    return _display_value(value)


def _approval_args_display(args: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _approval_arg_display(str(key), value) for key, value in args.items()}


def _approval_args_public(args: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): (
            "bound to reviewed memory version"
            if str(key).endswith("_binding") or str(key) == "target_binding"
            else value
        )
        for key, value in args.items()
    }


def _planned_args(row) -> dict[str, Any]:
    raw = row["planned_args"] if "planned_args" in row.keys() else "{}"
    try:
        planned_args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        planned_args = {}
    if not isinstance(planned_args, dict):
        return {}
    return planned_args


def _approval_age_minutes(row) -> int | None:
    raw = str(row["created_at"] if "created_at" in row.keys() else "").strip()
    if not raw:
        return None
    try:
        created = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - created.astimezone(timezone.utc)
    return max(0, int(delta.total_seconds() // 60))


def _approval_staleness_label(age_minutes: int | None) -> str:
    if age_minutes is None:
        return "unknown"
    if age_minutes < 15:
        return "fresh"
    if age_minutes < 120:
        return "review again"
    return "stale"


def _resolve_approval_id(store: MemoryStore, args: dict[str, Any]) -> tuple[int | None, str | None]:
    raw = args.get("approval_id")
    if raw is None:
        raw = args.get("id")
    if raw is None:
        raw = args.get("target")
    if isinstance(raw, str) and raw.strip().lower() in {"latest", "last", "newest", "current", ""}:
        rows = store.list_pending_approvals(limit=1)
        if not rows:
            return None, "No pending approval is available."
        return int(rows[0]["id"]), None
    try:
        approval_id = int(raw)
    except (TypeError, ValueError):
        return None, "approval_id must be a number or `latest`."
    if approval_id <= 0:
        return None, "approval_id must be a positive number or `latest`."
    return approval_id, None


def _get_approval_any_status(store: MemoryStore, approval_id: int):
    return store.get_approval(approval_id)


def _rerun_fingerprint_lines(row) -> list[str]:
    planned_args = _planned_args(row)
    arg_keys = ", ".join(sorted(planned_args)) if planned_args else "none"
    return [
        "Rerun fingerprint:",
        f"- approval id: {row['id']}",
        f"- tool: {_display_text(row['tool_name'])}",
        f"- status: {_display_text(row['status'])}",
        f"- session: {_display_text(row['session_id'])}",
        f"- created: {_display_text(row['created_at'])}",
        f"- updated: {_display_text(row['updated_at'])}",
        f"- planned arg keys: {arg_keys}",
        "- rule: approving reruns this exact stored request once; it does not grant ongoing permission.",
    ]


def _last_look_checklist_lines(row) -> list[str]:
    category, _concern, safer = _risk_review_for(row["tool_name"], row["user_input"])
    lines = [
        "Last-look checklist:",
        f"- Operator limit: {OPERATOR_LIMIT_RULE}",
        "- Confirm the operator still wants the original request now.",
        f"- Confirm the risk category is acceptable: {category}.",
        f"- Confirm the safer check was considered: {safer}",
        "- Confirm planned arguments match the exact target, command, path, text, or coordinates.",
        "- Dismiss instead of approving if the request is stale, broad, surprising, private, destructive, or aimed at the wrong account/app/file.",
    ]
    return lines


def _approval_decision_matrix_lines(row) -> list[str]:
    category, _concern, safer = _risk_review_for(row["tool_name"], row["user_input"])
    return [
        "Approval decision matrix:",
        f"- Approve only if: the operator still wants approval #{row['id']} now, the stored request is exact, and the expected result is verifiable.",
        f"- Stop first if: {OPERATOR_LIMIT_RULE}",
        "- Dismiss if: the request is stale, broad, surprising, private, destructive, aimed at the wrong app/account/file, or no longer needed.",
        f"- Narrow first if: the target, command, path, text, coordinates, recipient, or account is unclear; safer check: {safer}",
        f"- Ask the operator if: the risk category `{category}` is acceptable but the scope or success condition is still ambiguous.",
    ]


def _approval_proof_chain_commands(row) -> list[str]:
    approval_id = int(row["id"])
    return [
        f"approval readiness {approval_id}",
        f"approval packet {approval_id}",
        f"approve approval {approval_id}",
        f"approval chain proof {approval_id}",
        f"verification receipt <approved run id from approval chain proof {approval_id}>",
    ]


def _approval_proof_chain_lines(row) -> list[str]:
    return [
        "Approval proof chain:",
        *[f"- {command}" for command in _approval_proof_chain_commands(row)],
        f"- Operator limit: {OPERATOR_LIMIT_RULE}",
        "- Completion cannot count this risky action until the chain proof is valid and the approved run has a verification receipt.",
    ]


def _approval_resume_contract(row, *, approval_packet_viewed: bool) -> dict[str, Any]:
    approval_id = int(row["id"])
    planned_args = _planned_args(row)
    category, concern, safer = _risk_review_for(row["tool_name"], row["user_input"])
    return {
        "source": "approval_resume_contract",
        "approval_id": approval_id,
        "status": row["status"],
        "tool_name": row["tool_name"],
        "category": category,
        "concern": concern,
        "safer_check": safer,
        "rerun_user_input_display": _display_text(row["user_input"]),
        "exact_rerun_tool_name": row["tool_name"],
        "exact_rerun_args_display": _approval_args_display(planned_args),
        "exact_rerun_arg_keys": sorted(planned_args),
        "approval_packet_required": True,
        "approval_packet_viewed": approval_packet_viewed,
        "approve_command": f"approve approval {approval_id}",
        "dismiss_command": f"dismiss approval {approval_id}",
        "readiness_command": f"approval readiness {approval_id}",
        "last_look_command": f"approval packet {approval_id}",
        "proof_command": f"approval chain proof {approval_id}",
        "next_required_command": f"approval packet {approval_id}",
        "next_proof_command": f"approval chain proof {approval_id}",
        "proof_chain_commands": _approval_proof_chain_commands(row),
        "specific_request_only": True,
        "one_shot": True,
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
        "review_only": True,
        "draft_only": True,
        "loads_without_execution": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "approves_request": False,
        "dismisses_request": False,
        "calls_model": False,
        "executes_tools": False,
        "writes_files": False,
        "writes_notes": False,
        "writes_memory": False,
        "reads_private_data": False,
        "external_side_effect": False,
        "controls_computer": False,
        "queues_approval": False,
    }


def make_approval_tools(store: MemoryStore, vault: ObsidianVault):
    viewed_approval_packets: set[int] = set()
    reviewed_approval_recoveries: set[int] = set()
    readiness_receipts: dict[int, _ApprovalReadinessReceipt] = {}

    def invalidate_pending_review(approval_id: int | None = None) -> None:
        if approval_id is None:
            readiness_receipts.clear()
            viewed_approval_packets.clear()
            return
        readiness_receipts.pop(approval_id, None)
        viewed_approval_packets.discard(approval_id)

    def readiness_receipt_status(row: Any) -> tuple[bool, str]:
        approval_id = int(row["id"])
        receipt = readiness_receipts.get(approval_id)
        if receipt is None:
            return False, "readiness_required"
        if time.monotonic() >= receipt.expires_at_monotonic:
            invalidate_pending_review(approval_id)
            return False, "readiness_expired"
        snapshot = store.approval_readiness_snapshot(approval_id)
        current = snapshot.get("target")
        current_action_digest = (
            approval_action_digest(
                str(current["tool_name"] or "").strip(),
                _planned_args(current),
            )
            if current is not None
            else ""
        )
        if (
            current is None
            or str(current["status"]).lower() != "pending"
            or str(current["updated_at"]) != receipt.target_updated_at
            or int(snapshot["queue_revision"]) != receipt.queue_revision
            or snapshot["newest_pending_id"] != approval_id
            or current_action_digest != receipt.action_digest
            or _approval_staleness_label(_approval_age_minutes(current)) in {"stale", "unknown"}
        ):
            invalidate_pending_review(approval_id)
            return False, "readiness_changed"
        return True, "ready"

    def sync_pending_approvals(rows: list[Any] | None = None) -> str:
        pending_rows = rows if rows is not None else store.list_pending_approvals(limit=100)
        safe_rows, _unreadable_rows = _safe_approval_payloads(pending_rows)
        path = vault.write_pending_approvals(safe_rows)
        return str(path)

    def pending_approvals_path_metadata(path: str) -> dict[str, str]:
        return {
            "path": path,
            "path_display": _safe_vault_path_display(path, vault),
        }

    def build_approval_review(rows) -> str:
        approvals, unreadable_rows = _safe_approval_payloads(rows)
        if not rows or (not approvals and not unreadable_rows):
            return (
                "Approval review:\n"
                "- No pending approvals.\n"
                "- Keep using `autonomy plan: ...`, `privacy report`, and `safety status` before risky work."
            )
        if not approvals:
            return (
                "Approval review:\n"
                f"- No readable approvals. {unreadable_rows} unreadable approval row(s) hidden for safety.\n"
                "- Keep using `autonomy plan: ...`, `privacy report`, and `safety status` before risky work."
            )

        lines = [
            "Approval review:",
            "Use this before approving blocked actions. Run approval readiness before the last-look packet because approval means Jarvis may rerun the exact queued request once.",
            f"Operator limit: {OPERATOR_LIMIT_RULE}",
            "",
        ]
        if unreadable_rows:
            lines.append(f"- {unreadable_rows} unreadable approval row(s) hidden for safety.")
            lines.append("")
        for row in approvals:
            category, concern, safer = _risk_review_for(row["tool_name"], row["user_input"])
            lines.extend(
                [
                    f"Approval #{row['id']}: {_display_text(row['tool_name'])}",
                    f"- Request: {_display_text(row['user_input'])}",
                    f"- Category: {category}",
                    f"- Why gated: {concern}",
                    f"- Safer check: {safer}",
                    f"- Readiness: approval readiness {row['id']}",
                    f"- Last look: approval packet {row['id']}",
                    f"- Run if trusted: approve approval {row['id']}",
                    f"- Skip: dismiss approval {row['id']}",
                    "",
                ]
            )
            lines.extend(_approval_proof_chain_lines(row))
            lines.append("")
            planned_lines = _planned_args_lines(row)
            if planned_lines:
                lines.extend(planned_lines)
                lines.append("")
            checklist = _computer_approval_checklist(row["tool_name"], row["user_input"])
            if checklist:
                lines.extend(checklist)
                lines.append("")
            lines.extend(_approval_decision_matrix_lines(row))
            lines.append("")
        lines.extend(
            [
                "Before approving:",
                f"- Operator limit: {OPERATOR_LIMIT_RULE}",
                "- Confirm the target app/file/account is correct.",
                "- Prefer read-only inspection first when uncertain.",
                "- Dismiss stale approvals instead of leaving them ambiguous.",
            ]
        )
        return "\n".join(lines).rstrip()

    def approval_review_metadata(rows) -> dict[str, Any]:
        approvals, unreadable_rows = _safe_approval_payloads(rows)
        approval_items: list[dict[str, Any]] = []
        for row in approvals:
            approval_id = int(row["id"])
            category, concern, safer = _risk_review_for(row["tool_name"], row["user_input"])
            proof_chain = _approval_proof_chain_commands(row)
            approval_items.append(
                {
                    "approval_id": approval_id,
                    "tool_name": row["tool_name"],
                    "status": row["status"],
                    "category": category,
                    "concern": concern,
                    "safer_check": safer,
                    "readiness_command": f"approval readiness {approval_id}",
                    "last_look_command": f"approval packet {approval_id}",
                    "proof_command": f"approval chain proof {approval_id}",
                    "verification_command": f"verification receipt <approved run id from approval chain proof {approval_id}>",
                    "approve_command": f"approve approval {approval_id}",
                    "dismiss_command": f"dismiss approval {approval_id}",
                    "proof_chain_commands": proof_chain,
                }
            )

        first = approval_items[0] if approval_items else {}
        next_commands: list[str] = []
        for item in approval_items:
            next_commands.extend(
                [
                    item["readiness_command"],
                    item["last_look_command"],
                    item["proof_command"],
                    item["dismiss_command"],
                ]
            )
        return {
            "approval_review_handoff_ready": True,
            "approval_review_pending_count": len(approval_items),
            "approval_review_readable_approval_rows": len(approval_items),
            "approval_review_unreadable_approval_rows": unreadable_rows,
            "readable_approval_rows": len(approval_items),
            "unreadable_approval_rows": unreadable_rows,
            "approval_review_items": approval_items,
            "approval_review_next_commands": next_commands,
            "approval_review_next_command_count": len(next_commands),
            "approval_review_first_approval_id": first.get("approval_id"),
            "approval_review_first_tool_name": first.get("tool_name"),
            "approval_review_first_readiness_command": first.get("readiness_command"),
            "approval_review_first_last_look_command": first.get("last_look_command"),
            "approval_review_first_proof_command": first.get("proof_command"),
            "approval_review_first_verification_command": first.get("verification_command"),
            "approval_review_first_approve_command": first.get("approve_command"),
            "approval_review_first_dismiss_command": first.get("dismiss_command"),
            "next_command": first.get("readiness_command") or "safety status",
            "next_required_command": first.get("readiness_command") or "safety status",
            "next_proof_command": first.get("proof_command") or "safety status",
            "approval_review_handoff": {
                "handoff_ready": True,
                "approval_review_handoff_ready": True,
                "pending_count": len(approval_items),
                "readable_approval_rows": len(approval_items),
                "unreadable_approval_rows": unreadable_rows,
                "items": approval_items,
                "next_commands": next_commands,
                "next_command_count": len(next_commands),
                "next_required_command": first.get("readiness_command") or "safety status",
                "next_proof_command": first.get("proof_command") or "safety status",
                "first_approval_id": first.get("approval_id"),
                "first_tool_name": first.get("tool_name"),
                "first_readiness_command": first.get("readiness_command"),
                "first_last_look_command": first.get("last_look_command"),
                "first_proof_command": first.get("proof_command"),
                "first_verification_command": first.get("verification_command"),
                "first_approve_command": first.get("approve_command"),
                "first_dismiss_command": first.get("dismiss_command"),
                "approves_request": False,
                "dismisses_request": False,
                "authorizes_execution": False,
                "authorizes_completion_claim": False,
                "requires_manual_send": True,
                "operator_timeboxes_override_priority": True,
                "stop_times_override_priority": True,
            },
        }

    def build_approval_detail(row) -> str:
        category, concern, safer = _risk_review_for(row["tool_name"], row["user_input"])
        lines = [
            f"Approval detail #{row['id']}: {_display_text(row['tool_name'])}",
            f"- Status: {_display_text(row['status'])}",
            f"- Created: {_display_text(row['created_at'])}",
            f"- Updated: {_display_text(row['updated_at'])}",
            f"- Session: {_display_text(row['session_id'])}",
            f"- Original request: {_display_text(row['user_input'])}",
            f"- Category: {category}",
            f"- Why gated: {concern}",
            f"- Safer check: {safer}",
            "",
            *_rerun_fingerprint_lines(row),
            "",
            *_approval_proof_chain_lines(row),
            "",
        ]
        planned_lines = _planned_args_lines(row)
        if planned_lines:
            lines.extend(planned_lines)
            lines.append("")
        checklist = _computer_approval_checklist(row["tool_name"], row["user_input"])
        if checklist:
            lines.extend(checklist)
            lines.append("")
        lines.extend(
            [
                *_approval_decision_matrix_lines(row),
                "",
                *_last_look_checklist_lines(row),
                "",
                "Blocked output:",
                _display_text(row["reason"]),
                "",
                "Decision commands:",
                f"- Readiness: approval readiness {row['id']}",
                f"- Last look: approval packet {row['id']}",
                f"- Proof after approval: approval chain proof {row['id']}",
                f"- Run if trusted: approve approval {row['id']}",
                f"- Skip: dismiss approval {row['id']}",
            "",
            "Rule: approval reruns the exact original request once. Inspect the target, scope, and expected result before approving.",
            f"Operator limit: {OPERATOR_LIMIT_RULE}",
        ]
        )
        return "\n".join(lines)

    def build_approval_execution_packet(
        row,
        *,
        readiness_rechecked: bool = False,
    ) -> str:
        category, concern, safer = _risk_review_for(row["tool_name"], row["user_input"])
        lines = [
            f"Approval execution packet #{row['id']}: {_display_text(row['tool_name'])}",
            "",
            "What approval would do:",
            f"- Rerun exactly: {_display_text(row['user_input'])}",
            f"- Tool: {_display_text(row['tool_name'])}",
            f"- Category: {category}",
            f"- Current status: {_display_text(row['status'])}",
            "",
            *_rerun_fingerprint_lines(row),
            "",
            *_approval_proof_chain_lines(row),
            "",
            "Why this needs a last look:",
            f"- {concern}",
            f"- Safer check: {safer}",
            "",
        ]
        planned_lines = _planned_args_lines(row)
        if planned_lines:
            lines.extend(planned_lines)
            lines.append("")
        checklist = _computer_approval_checklist(row["tool_name"], row["user_input"])
        if checklist:
            lines.extend(checklist)
            lines.append("")
        lines.extend(
            [
                *_approval_decision_matrix_lines(row),
                "",
                *_last_look_checklist_lines(row),
                "",
                "Approval readiness:",
                f"- Operator limit: {OPERATOR_LIMIT_RULE}",
                *(
                    ["- Current queue readiness was freshly rechecked for this last-look packet."]
                    if readiness_rechecked
                    else []
                ),
                "- Confirm the original request still matches what the operator wants now.",
                "- Confirm the target, account, file, command, coordinates, text, or side effect is exact.",
                "- Prefer a read-only preview first if anything is ambiguous.",
                "- Dismiss the approval if it is stale, broad, surprising, or no longer needed.",
                "",
                "Decision commands:",
                f"- Run if trusted: approve approval {row['id']}",
                f"- Skip: dismiss approval {row['id']}",
                f"- Prove after approval: approval chain proof {row['id']}",
                "",
                "Rule: this packet is read-only and does not approve, dismiss, rerun, execute tools, control the computer, read private data, or queue approvals.",
                f"Operator limit: {OPERATOR_LIMIT_RULE}",
            ]
        )
        return "\n".join(lines)

    def list_pending_approvals(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 20, 1, 100)
        status = _short(args.get("status") or "pending", 32).lower()
        rows = store.list_pending_approvals(status=status, limit=limit)
        approvals, unreadable_rows = _safe_approval_payloads(rows)
        if not rows or (not approvals and not unreadable_rows):
            path = sync_pending_approvals(approvals)
            return ToolResult(
                "list_pending_approvals",
                True,
                "No pending approvals.",
                _safe_metadata(
                    **pending_approvals_path_metadata(path),
                    count=0,
                    readable_approval_rows=0,
                    unreadable_approval_rows=0,
                    limit=limit,
                    status=status,
                ),
            )
        if not approvals:
            path = sync_pending_approvals(approvals)
            return ToolResult(
                "list_pending_approvals",
                True,
                f"No readable approvals. {unreadable_rows} unreadable approval row(s) hidden for safety.",
                _safe_metadata(
                    **pending_approvals_path_metadata(path),
                    count=0,
                    readable_approval_rows=0,
                    unreadable_approval_rows=unreadable_rows,
                    limit=limit,
                    status=status,
                ),
            )
        first = approvals[0]
        first_id = int(first["id"])
        lines = [
            "Pending approvals:",
            "",
            "First safe handoff:",
            f"- approval: #{first_id} `{first['tool_name']}`",
            f"- readiness: `approval readiness {first_id}`",
            f"- last look: `approval packet {first_id}`",
            f"- proof: `approval chain proof {first_id}`",
            f"- approve only if trusted: `approve approval {first_id}`",
            f"- dismiss if stale/broad/private/surprising: `dismiss approval {first_id}`",
            "",
        ]
        if unreadable_rows:
            lines.append(f"- {unreadable_rows} unreadable approval row(s) hidden for safety.")
            lines.append("")
        for row in approvals:
            category, concern, safer = _risk_review_for(row["tool_name"], row["user_input"])
            planned_arg_keys = ", ".join(sorted(_planned_args(row))) or "none"
            lines.append(f"- #{row['id']} {_display_text(row['tool_name'])} | {category} | {_display_text(row['created_at'])}")
            lines.append(f"  Request: {_display_text(row['user_input'], 160)}")
            lines.append(f"  Why gated: {concern}")
            lines.append(f"  Safer check: {safer}")
            lines.append("  Decision matrix: approve only if exact and verifiable; dismiss stale/broad/private/surprising requests.")
            lines.append(f"  Planned arg keys: {planned_arg_keys}")
            lines.append(f"  Readiness: `approval readiness {row['id']}`")
            lines.append(f"  Last look: `approval packet {row['id']}`")
            lines.append(f"  Proof after approval: `approval chain proof {row['id']}`")
            lines.append(f"  Run if trusted: `approve approval {row['id']}`")
            lines.append(f"  Skip: `dismiss approval {row['id']}`")
        lines.extend(
            [
                "",
                f"Operator limit: {OPERATOR_LIMIT_RULE}",
                "Rule: approving reruns only the exact stored request once; it does not grant ongoing permission.",
            ]
        )
        path = sync_pending_approvals(approvals)
        return ToolResult(
            "list_pending_approvals",
            True,
            "\n".join(lines),
            _safe_metadata(
                count=len(approvals),
                readable_approval_rows=len(approvals),
                unreadable_approval_rows=unreadable_rows,
                limit=limit,
                status=status,
                **pending_approvals_path_metadata(path),
                first_approval_id=first_id,
                first_approval_tool=first["tool_name"],
                first_readiness_command=f"approval readiness {first_id}",
                first_last_look_command=f"approval packet {first_id}",
                first_proof_command=f"approval chain proof {first_id}",
                first_approve_command=f"approve approval {first_id}",
                first_dismiss_command=f"dismiss approval {first_id}",
                proof_chain_commands_by_approval={
                    str(row["id"]): _approval_proof_chain_commands(row) for row in approvals
                },
            ),
        )

    def dismiss_pending_approval(args: dict[str, Any]) -> ToolResult:
        approval_id, error = _resolve_approval_id(store, args)
        if approval_id is None:
            return _bad_approval_id_result("dismiss_pending_approval", args, error)
        row = _get_approval_any_status(store, approval_id)
        if row is None:
            return _approval_not_found_result(
                "dismiss_pending_approval",
                approval_id,
                f"dismiss approval {approval_id}",
            )
        current_status = str(row["status"] or "").lower()
        if current_status == "dismissed":
            return _already_dismissed_result("dismiss_pending_approval", approval_id)
        if current_status != "pending":
            return _approval_cas_miss_result(
                "dismiss_pending_approval",
                approval_id,
                "dismiss",
                current_status,
            )
        try:
            ok = store.set_pending_approval_status(approval_id, "dismissed")
        except Exception as exc:
            actual = _get_approval_any_status(store, approval_id)
            actual_status = str(actual["status"] or "").lower() if actual is not None else "pending"
            if actual_status == "dismissed":
                return _already_dismissed_result("dismiss_pending_approval", approval_id)
            if actual_status != "pending":
                return _approval_cas_miss_result(
                    "dismiss_pending_approval",
                    approval_id,
                    "dismiss",
                    actual_status,
                )
            return _approval_update_failure_result(
                "dismiss_pending_approval",
                approval_id,
                "dismiss",
                exception_type=type(exc).__name__,
                actual_status=actual_status,
            )
        if not ok:
            actual = _get_approval_any_status(store, approval_id)
            if actual is None:
                return _approval_not_found_result(
                    "dismiss_pending_approval",
                    approval_id,
                    f"dismiss approval {approval_id}",
                )
            actual_status = str(actual["status"] or "").lower()
            if actual_status == "dismissed":
                return _already_dismissed_result("dismiss_pending_approval", approval_id)
            return _approval_cas_miss_result(
                "dismiss_pending_approval",
                approval_id,
                "dismiss",
                actual_status,
            )
        invalidate_pending_review()
        try:
            path = sync_pending_approvals()
        except Exception as exc:
            return ToolResult(
                "dismiss_pending_approval",
                True,
                f"Dismissed approval #{approval_id}. The durable decision was recorded, but the "
                "pending-approvals note could not be synchronized; run `pending approvals` to refresh it.",
                _safe_metadata(
                    approval_id=approval_id,
                    actual_status="dismissed",
                    approval_decision_recorded=True,
                    pending_approvals_sync_failed=True,
                    sync_exception_type=type(exc).__name__,
                    next_command="pending approvals",
                    dismisses_request=True,
                ),
            )
        return ToolResult(
            "dismiss_pending_approval",
            True,
            f"Dismissed approval #{approval_id}.",
            _approval_write_metadata(
                approval_id=approval_id,
                actual_status="dismissed",
                approval_decision_recorded=True,
                pending_approvals_sync_failed=False,
                **pending_approvals_path_metadata(path),
                dismisses_request=True,
            ),
        )

    def inspect_pending_approval(args: dict[str, Any]) -> ToolResult:
        approval_id, error = _resolve_approval_id(store, args)
        if approval_id is None:
            return _bad_approval_id_result("inspect_pending_approval", args, error)
        status = _short(args.get("status") or "pending", 32).lower()
        row = store.get_pending_approval(approval_id, status=status)
        if row is None and status.lower() == "pending":
            for fallback_status in ("approved", "dismissed"):
                row = store.get_pending_approval(approval_id, status=fallback_status)
                if row is not None:
                    break
        if row is None:
            return _approval_not_found_result(
                "inspect_pending_approval",
                approval_id,
                f"approval detail {approval_id}",
                pending_only=False,
            )
        return ToolResult(
            "inspect_pending_approval",
            True,
            build_approval_detail(row),
            _safe_metadata(
                approval_id=approval_id,
                status=row["status"],
                tool_name=row["tool_name"],
                planned_arg_keys=sorted(_planned_args(row)),
                proof_chain_commands=_approval_proof_chain_commands(row),
                next_required_command=f"approval readiness {approval_id}",
                next_proof_command=f"approval readiness {approval_id}",
            ),
        )

    def approval_execution_packet(args: dict[str, Any]) -> ToolResult:
        approval_id, error = _resolve_approval_id(store, args)
        if approval_id is None:
            return _bad_approval_id_result("approval_execution_packet", args, error)
        status = _short(args.get("status") or "pending", 32).lower()
        row = store.get_pending_approval(approval_id, status=status)
        if row is None and status.lower() == "pending":
            for fallback_status in ("approved", "dismissed"):
                row = store.get_pending_approval(approval_id, status=fallback_status)
                if row is not None:
                    break
        if row is None:
            return _approval_not_found_result(
                "approval_execution_packet",
                approval_id,
                f"approval packet {approval_id}",
                pending_only=False,
                status=status,
            )
        row_status = str(row["status"]).lower()
        execution_used = store.approval_execution_used(approval_id) if row_status == "approved" else False
        recovery_available = row_status == "approved" and not execution_used
        readiness_rechecked = False
        if row_status == "pending":
            readiness_valid, readiness_reason = readiness_receipt_status(row)
            if not readiness_valid:
                # A Telegram bridge or daemon restart must not turn a valid human
                # review sequence into an impossible loop. The last-look command
                # is itself read-only, so it can repeat the same strict freshness
                # check before presenting the exact packet; approval remains gated.
                refreshed = approval_readiness_packet({"approval_id": approval_id})
                readiness_valid, readiness_reason = readiness_receipt_status(row)
                if not refreshed.ok or not readiness_valid:
                    return _approval_review_refusal_result(
                        "approval_execution_packet",
                        (
                            f"Approval #{approval_id} needs a current readiness receipt before its last-look packet.\n"
                            f"Run: approval readiness {approval_id}\n"
                            f"Then: approval packet {approval_id}"
                        ),
                        _safe_metadata(
                            reason=readiness_reason,
                            approval_id=approval_id,
                            status=row["status"],
                            approval_readiness_required=True,
                            approval_readiness_valid=False,
                            approval_packet_required=True,
                            approval_packet_viewed=False,
                            next_command=f"approval readiness {approval_id}",
                            next_required_command=f"approval readiness {approval_id}",
                        ),
                        commands=(
                            f"approval readiness {approval_id}",
                            f"approval packet {approval_id}",
                        ),
                    )
                readiness_rechecked = True
            viewed_approval_packets.add(approval_id)
        elif recovery_available:
            viewed_approval_packets.add(approval_id)
        planned_args = {}
        try:
            planned_args = json.loads(row["planned_args"] or "{}") if "planned_args" in row.keys() else {}
        except json.JSONDecodeError:
            planned_args = {}
        if not isinstance(planned_args, dict):
            planned_args = {}
        proof_chain_commands = _approval_proof_chain_commands(row)
        category, concern, safer = _risk_review_for(row["tool_name"], row["user_input"])
        resume_contract = _approval_resume_contract(row, approval_packet_viewed=approval_id in viewed_approval_packets)
        approval_execution_handoff = {
            "source": "approval_execution_packet",
            "approval_id": approval_id,
            "status": row["status"],
            "tool_name": row["tool_name"],
            "category": category,
            "concern": concern,
            "safer_check": safer,
            "planned_args": _approval_args_display(planned_args),
            "planned_arg_keys": sorted(planned_args),
            "approval_packet_viewed": approval_id in viewed_approval_packets,
            "approval_readiness_rechecked": readiness_rechecked,
            "approval_recovery_available": recovery_available,
            "approval_execution_used": execution_used,
            "proof_chain_commands": proof_chain_commands,
            "next_required_command": f"approve approval {approval_id}",
            "next_proof_command": f"approval chain proof {approval_id}",
            "decision_commands": {
                "approve": f"approve approval {approval_id}",
                "dismiss": f"dismiss approval {approval_id}",
                "proof_after_approval": f"approval chain proof {approval_id}",
            },
            "approval_resume_contract": resume_contract,
            "review_only": True,
            "draft_only": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "approves_request": False,
            "dismisses_request": False,
            "calls_model": False,
            "executes_tools": False,
            "writes_files": False,
            "writes_notes": False,
            "writes_memory": False,
            "reads_private_data": False,
            "external_side_effect": False,
            "controls_computer": False,
            "queues_approval": False,
        }
        return ToolResult(
            "approval_execution_packet",
            True,
            build_approval_execution_packet(row, readiness_rechecked=readiness_rechecked),
            _safe_metadata(
                approval_id=approval_id,
                status=row["status"],
                tool_name=row["tool_name"],
                planned_args=_approval_args_display(planned_args),
                planned_arg_keys=sorted(planned_args),
                approval_packet_viewed=approval_id in viewed_approval_packets,
                approval_readiness_required=row_status == "pending",
                approval_readiness_valid=row_status == "pending",
                approval_readiness_rechecked=readiness_rechecked,
                approval_recovery_available=recovery_available,
                approval_execution_used=execution_used,
                proof_chain_commands=proof_chain_commands,
                next_required_command=f"approve approval {approval_id}",
                next_proof_command=f"approval chain proof {approval_id}",
                approval_execution_handoff=approval_execution_handoff,
                approval_resume_contract=resume_contract,
                approval_execution_review_only=True,
                approval_execution_draft_only=True,
                approval_execution_loads_without_execution=True,
                approval_execution_authorizes_execution=False,
                approval_execution_authorizes_completion_claim=False,
                approval_execution_approval_granted=False,
            ),
        )

    def approval_resume_packet(args: dict[str, Any]) -> ToolResult:
        approval_id, error = _resolve_approval_id(store, args)
        if approval_id is None:
            return _bad_approval_id_result("approval_resume_packet", args, error)
        row = _get_approval_any_status(store, approval_id)
        if row is None:
            return _approval_not_found_result(
                "approval_resume_packet",
                approval_id,
                f"approval resume packet {approval_id}",
                pending_only=False,
                verdict="APPROVAL_NOT_FOUND",
            )

        pending_rows = store.list_pending_approvals(limit=100)
        pending_approvals, unreadable_pending_rows = _safe_approval_payloads(pending_rows)
        pending_ids = [int(item["id"]) for item in pending_approvals]
        pending_count = len(pending_approvals)
        status = str(row["status"]).lower()
        execution_used = store.approval_execution_used(approval_id) if status == "approved" else False
        recovery_available = status == "approved" and not execution_used
        age_minutes = _approval_age_minutes(row)
        staleness = _approval_staleness_label(age_minutes)
        if recovery_available and approval_id not in viewed_approval_packets:
            verdict = "APPROVED_UNCLAIMED_LAST_LOOK_REQUIRED"
            next_command = f"approval packet {approval_id}"
        elif recovery_available:
            reviewed_approval_recoveries.add(approval_id)
            verdict = "APPROVED_UNCLAIMED_READY_TO_RESUME"
            next_command = f"approve approval {approval_id}"
        elif status != "pending":
            verdict = "NOT_PENDING_REVIEW_ONLY"
            next_command = f"approval chain proof {approval_id}"
        elif approval_id not in viewed_approval_packets:
            verdict = "LAST_LOOK_REQUIRED"
            next_command = f"approval packet {approval_id}"
        elif staleness == "stale":
            verdict = "RECONFIRM_BEFORE_APPROVAL"
            next_command = f"approval packet {approval_id}"
        elif approval_id != (pending_ids[0] if pending_ids else approval_id):
            verdict = "REVIEW_NEWER_APPROVALS_FIRST"
            next_command = f"approval readiness {pending_ids[0]}"
        else:
            verdict = "READY_FOR_ONE_SHOT_APPROVAL_RERUN"
            next_command = f"approve approval {approval_id}"

        contract = _approval_resume_contract(row, approval_packet_viewed=approval_id in viewed_approval_packets)
        lines = [
            f"Approval resume packet #{approval_id}:",
            "This is a read-only contract for resuming one specific queued request after review.",
            "",
            "Resume target:",
            f"- approval id: {approval_id}",
            f"- status: {_display_text(row['status'])}",
            f"- tool: {_display_text(row['tool_name'])}",
            f"- request: {_display_text(row['user_input'])}",
            f"- planned arg keys: {', '.join(contract['exact_rerun_arg_keys']) if contract['exact_rerun_arg_keys'] else 'none'}",
            f"- approval packet viewed: {'yes' if approval_id in viewed_approval_packets else 'no'}",
            f"- approved execution already used: {'yes' if execution_used else 'no'}",
            f"- staleness: {staleness}",
            "",
            "Resume verdict:",
            f"- verdict: {verdict}",
            f"- next safe command: `{next_command}`",
            "- scope: one specific stored request, one approved rerun, no ongoing permission.",
            "",
            *_approval_proof_chain_lines(row),
            "",
            "Boundary:",
            f"- Operator limit: {OPERATOR_LIMIT_RULE}",
            "- This packet is read-only and does not approve, dismiss, rerun, execute tools, control the computer, read private data, write files, write notes, write memory, call external services, or queue approvals.",
        ]
        if unreadable_pending_rows:
            lines.insert(11, f"- {unreadable_pending_rows} unreadable approval row(s) hidden for safety.")
        return ToolResult(
            "approval_resume_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                approval_id=approval_id,
                status=row["status"],
                tool_name=row["tool_name"],
                age_minutes=age_minutes,
                staleness=staleness,
                pending_approvals=pending_count,
                readable_approval_rows=pending_count,
                unreadable_approval_rows=unreadable_pending_rows,
                newest_pending_approval_id=pending_ids[0] if pending_ids else None,
                approval_packet_required=True,
                approval_packet_viewed=approval_id in viewed_approval_packets,
                approval_recovery_available=recovery_available,
                approval_recovery_reviewed=approval_id in reviewed_approval_recoveries,
                approval_execution_used=execution_used,
                verdict=verdict,
                next_command=next_command,
                next_required_command=next_command,
                proof_chain_commands=contract["proof_chain_commands"],
                approval_resume_contract=contract,
                approval_resume_review_only=True,
                approval_resume_draft_only=True,
                approval_resume_loads_without_execution=True,
                approval_resume_authorizes_execution=False,
                approval_resume_authorizes_completion_claim=False,
                approval_resume_approval_granted=False,
            ),
        )

    def approval_history(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 15, 1, 50)
        rows = store.recent_approvals(limit=limit)
        if not rows:
            return ToolResult(
                "approval_history",
                True,
                "Approval history:\n- No approval records yet.",
                _safe_metadata(count=0, limit=limit, approved_tool_runs=0),
            )

        counts: dict[str, int] = {}
        approved_run_count = 0
        lines = [
            "Approval history:",
            "Recent approval decisions, pending requests, and linked approved reruns. This is audit-only; use `approval readiness #ID`, then `approval packet #ID`, then `approval chain proof #ID`, before approving anything still pending.",
            f"Operator limit: {OPERATOR_LIMIT_RULE}",
            "",
        ]
        for row in rows:
            status = row["status"]
            counts[status] = counts.get(status, 0) + 1
            category, concern, safer = _risk_review_for(row["tool_name"], row["user_input"])
            lines.extend(
                [
                    f"Approval #{row['id']} [{_display_text(status)}]: {_display_text(row['tool_name'])}",
                    f"- Request: {_display_text(row['user_input'])}",
                    f"- Category: {category}",
                    f"- Created: {_display_text(row['created_at'])}",
                    f"- Updated: {_display_text(row['updated_at'])}",
                    f"- Audit note: {concern}",
                    f"- Safer check: {safer}",
                ]
            )
            if status.lower() == "pending":
                lines.extend(
                    [
                        f"- Readiness: approval readiness {row['id']}",
                        f"- Last look: approval packet {row['id']}",
                        f"- Proof after approval: approval chain proof {row['id']}",
                        f"- Decide: approve approval {row['id']} or dismiss approval {row['id']}",
                    ]
                )
            try:
                evidence = store.classify_approval_execution_evidence(int(row["id"]), limit=10)
            except Exception:
                evidence = None
            linked_runs = list(evidence.linked_runs) if evidence is not None else []
            matching_runs = list(evidence.matching_runs) if evidence is not None else []
            exact_run = (
                linked_runs[0]
                if len(linked_runs) == 1
                and len(matching_runs) == 1
                and int(linked_runs[0]["id"]) == int(matching_runs[0]["id"])
                else None
            )
            exact_run_id = int(exact_run["id"]) if exact_run is not None else None
            valid_success = bool(
                evidence is not None
                and evidence.verdict == "APPROVAL_CHAIN_PROVEN"
                and evidence.valid_execution_proof is True
                and exact_run_id is not None
                and [int(run["id"]) for run in evidence.successful_runs] == [exact_run_id]
            )
            exact_failure = bool(
                evidence is not None
                and evidence.verdict == "APPROVAL_EXECUTION_FAILED"
                and exact_run_id is not None
                and [int(run["id"]) for run in evidence.failed_runs] == [exact_run_id]
            )
            exact_unknown = bool(
                evidence is not None
                and evidence.outcome_unknown is True
                and exact_run_id is not None
            )
            if valid_success or exact_failure or exact_unknown:
                approved_run_count += 1
                lines.append("- Approved rerun audit:")
                if valid_success:
                    result = "ok"
                elif exact_failure:
                    result = "failed, exact approval linkage"
                else:
                    result = "outcome unknown, exact approval linkage"
                lines.append(
                    f"  - Tool run #{exact_run['id']} {_display_text(exact_run['tool_name'])} "
                    f"[{result}] at {_display_text(exact_run['created_at'])}"
                )
            elif linked_runs or str(status).lower() == "approved":
                verdict = evidence.verdict if evidence is not None else "APPROVAL_EVIDENCE_UNREADABLE"
                lines.append(f"- Approved rerun audit: invalid linkage ({_display_text(verdict)}).")
                lines.append("  - Linked rows are not trusted unless classifier evidence is exact and row-bound.")
            lines.append("")
        lines.extend(
            [
                "Boundary:",
                f"- Operator limit: {OPERATOR_LIMIT_RULE}",
                "- This history is read-only and does not approve, dismiss, rerun, execute tools, control the computer, read private data, write files, or queue approvals.",
            ]
        )
        return ToolResult(
            "approval_history",
            True,
            "\n".join(lines).rstrip(),
            _safe_metadata(count=len(rows), limit=limit, status_counts=counts, approved_tool_runs=approved_run_count),
        )

    def approval_chain_proof(args: dict[str, Any]) -> ToolResult:
        approval_id, error = _resolve_approval_id(store, args)
        if approval_id is None:
            return _bad_approval_id_result("approval_chain_proof", args, error)
        evidence = store.classify_approval_execution_evidence(approval_id, limit=10)
        row = evidence.approval
        if row is None:
            return _approval_not_found_result(
                "approval_chain_proof",
                approval_id,
                f"approval chain proof {approval_id}",
                pending_only=False,
                verdict="APPROVAL_NOT_FOUND",
            )

        linked_runs = list(evidence.linked_runs)
        matching_runs = list(evidence.matching_runs)
        successful_runs = list(evidence.successful_runs)
        failed_runs = list(evidence.failed_runs)
        successful_run_ids = {int(run["id"]) for run in successful_runs}
        failed_run_ids = {int(run["id"]) for run in failed_runs}
        status = str(row["status"]).lower()
        planned_args = _planned_args(row)
        verdict = evidence.verdict
        valid_execution_proof = evidence.valid_execution_proof
        claim = evidence.claim
        claim_outcome = str(claim["outcome"] or "") if claim is not None else ""
        claim_completed = bool(str(claim["completed_at"] or "").strip()) if claim is not None else False

        category, concern, safer = _risk_review_for(row["tool_name"], row["user_input"])
        proof_chain_commands = _approval_proof_chain_commands(row)
        linked_run_summaries = [
            {
                "id": int(run["id"]),
                "tool_name": run["tool_name"],
                "ok": int(run["id"]) in successful_run_ids,
                "created_at": run["created_at"],
            }
            for run in linked_runs
        ]
        approved_run_ids = [int(run["id"]) for run in successful_runs]
        if successful_runs:
            next_proof_command = f"verification receipt {int(successful_runs[0]['id'])}"
        elif failed_runs:
            next_proof_command = f"verification receipt {int(failed_runs[0]['id'])}"
        else:
            next_proof_command = f"verification receipt <approved run id from approval chain proof {approval_id}>"
        approval_chain_proof_handoff = {
            "source": "approval_chain_proof",
            "approval_id": approval_id,
            "status": row["status"],
            "tool_name": row["tool_name"],
            "category": category,
            "concern": concern,
            "safer_check": safer,
            "planned_arg_keys": sorted(planned_args),
            "linked_runs": len(linked_runs),
            "matching_tool_runs": len(matching_runs),
            "approved_run_ids": approved_run_ids,
            "approved_success_count": len(successful_runs),
            "failed_run_count": len(failed_runs),
            "approved_reruns": linked_run_summaries,
            "verdict": verdict,
            "valid_execution_proof": valid_execution_proof,
            "approval_execution_claim_present": claim is not None,
            "approval_execution_claim_outcome": claim_outcome,
            "approval_execution_claim_completed": claim_completed,
            "approval_execution_outcome_unknown": evidence.outcome_unknown,
            "approval_packet_required": True,
            "approval_packet_viewed": approval_id in viewed_approval_packets,
            "proof_chain_commands": proof_chain_commands,
            "next_required_command": next_proof_command,
            "next_proof_command": next_proof_command,
            "verification_command": next_proof_command,
            "review_only": True,
            "draft_only": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "approves_request": False,
            "dismisses_request": False,
            "calls_model": False,
            "executes_tools": False,
            "writes_files": False,
            "writes_notes": False,
            "writes_memory": False,
            "reads_private_data": False,
            "external_side_effect": False,
            "controls_computer": False,
            "queues_approval": False,
        }
        lines = [
            f"Approval chain proof #{approval_id}:",
            "This is a read-only audit packet for proving whether a risky execution has a valid approval chain.",
            "",
            "Approval record:",
            f"- status: {_display_text(row['status'])}",
            f"- tool: {_display_text(row['tool_name'])}",
            f"- category: {category}",
            f"- request: {_display_text(row['user_input'])}",
            f"- created: {_display_text(row['created_at'])}",
            f"- updated: {_display_text(row['updated_at'])}",
            f"- planned arg keys: {', '.join(sorted(planned_args)) if planned_args else 'none'}",
            "",
            "Risk review:",
            f"- operator limit: {OPERATOR_LIMIT_RULE}",
            f"- why gated: {concern}",
            f"- safer check: {safer}",
            "",
            "Linked approved reruns:",
        ]
        if linked_runs:
            for run in linked_runs:
                run_id = int(run["id"])
                result = "ok" if run_id in successful_run_ids else "failed" if run_id in failed_run_ids else "unknown"
                lines.append(f"- Tool run #{run['id']} {_display_text(run['tool_name'])} [{result}] {_display_text(run['created_at'])}")
                output = _display_text(str(run["output"] or "").replace("\n", " "), 180)
                if output:
                    lines.append(f"  output: {output}")
        else:
            lines.append("- none linked to this approval id")

        lines.extend(
            [
                "",
                "Proof checks:",
                f"- approval exists: yes",
                f"- approval status is approved: {'yes' if status == 'approved' else 'no'}",
                f"- finalized execution claim exists: {'yes' if claim_completed else 'no'}",
                f"- execution claim outcome is succeeded: {'yes' if claim_outcome == 'succeeded' else 'no'}",
                f"- approved rerun is linked by approval id: {'yes' if bool(linked_runs) else 'no'}",
                f"- linked rerun matches stored tool: {'yes' if bool(matching_runs) else 'no'}",
                f"- matching linked approved rerun succeeded: {'yes' if bool(successful_runs) else 'no'}",
                "",
                f"Verdict: {verdict}",
                f"Valid execution proof: {'yes' if valid_execution_proof else 'no'}",
                "",
                "Next step:",
            ]
        )
        if status == "pending":
            lines.append(f"- Run `approval readiness {approval_id}`, then `approval packet {approval_id}`, then `approval chain proof {approval_id}`, before deciding; do not count this as execution proof yet.")
        elif status == "dismissed":
            lines.append("- Dismissed approvals are useful audit evidence, but they cannot prove execution.")
        elif verdict == "APPROVAL_STATUS_MALFORMED":
            lines.append("- The stored approval status is not recognized. Treat the record as invalid and do not trust or replay it.")
        elif evidence.outcome_unknown:
            lines.append("- The one-shot execution outcome is unknown. Do not replay it; verify the target state before issuing a fresh request.")
        elif verdict == "APPROVAL_EXECUTION_CLAIM_MISSING":
            lines.append("- No durable one-shot execution claim exists, so linked audit rows cannot prove this approval executed.")
        elif verdict == "APPROVAL_LINKED_RUN_TOOL_MISMATCH":
            lines.append("- Linked approved audit rows do not match the stored tool. Treat them as invalid proof and inspect the audit trail.")
        elif verdict == "APPROVAL_LINKED_RUN_RESULT_MALFORMED":
            lines.append("- A linked approved audit row has a malformed success flag. Treat it as invalid proof and inspect the audit trail.")
        elif verdict.startswith("APPROVAL_EXECUTION_"):
            lines.append(f"- The durable execution claim ended as `{_display_text(claim_outcome or 'unknown')}`; verify recovery before any fresh request.")
        elif not linked_runs:
            lines.append("- The approval was marked approved, but no approved tool run is linked; inspect `approval history` and `recent tool runs`.")
        elif failed_runs and not successful_runs:
            lines.append(f"- Linked execution failed; run `verification receipt {failed_runs[0]['id']}` and choose recovery before claiming success.")
        else:
            lines.append(f"- Use `verification receipt {successful_runs[0]['id']}` to attach output proof before completion.")

        lines.extend(
            [
                "",
                "Boundary:",
                f"- Operator limit: {OPERATOR_LIMIT_RULE}",
                "- This proof is read-only. It does not approve, dismiss, rerun, execute tools, control the computer, read private data, write files, write notes, write memory, call external services, or queue approvals.",
            ]
        )

        return ToolResult(
            "approval_chain_proof",
            True,
            "\n".join(lines),
            _safe_metadata(
                approval_id=approval_id,
                status=row["status"],
                tool_name=row["tool_name"],
                category=category,
                planned_arg_keys=sorted(planned_args),
                linked_runs=len(linked_runs),
                matching_tool_runs=len(matching_runs),
                successful_runs=len(successful_runs),
                failed_runs=len(failed_runs),
                verdict=verdict,
                valid_execution_proof=valid_execution_proof,
                approval_execution_claim_present=claim is not None,
                approval_execution_claim_outcome=claim_outcome,
                approval_execution_claim_completed=claim_completed,
                approval_execution_outcome_unknown=evidence.outcome_unknown,
                approved_run_ids=approved_run_ids,
                approved_success_count=len(successful_runs),
                approval_packet_required=True,
                approval_packet_viewed=approval_id in viewed_approval_packets,
                proof_chain_commands=proof_chain_commands,
                next_required_command=next_proof_command,
                next_proof_command=next_proof_command,
                verification_command=next_proof_command,
                approval_chain_proof_handoff=approval_chain_proof_handoff,
                approval_chain_proof_review_only=True,
                approval_chain_proof_draft_only=True,
                approval_chain_proof_loads_without_execution=True,
                approval_chain_proof_authorizes_execution=False,
                approval_chain_proof_authorizes_completion_claim=False,
                approval_chain_proof_approval_granted=False,
            ),
        )

    def approval_queue_summary(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 5, 1, 20)
        rows = store.list_pending_approvals(status=str(args.get("status") or "pending"), limit=limit)
        approvals, unreadable_rows = _safe_approval_payloads(rows)
        if not rows or (not approvals and not unreadable_rows):
            return ToolResult(
                "approval_queue_summary",
                True,
                "Approval queue summary:\n- clear: no pending approvals.",
                _safe_metadata(count=0, readable_approval_rows=0, unreadable_approval_rows=0, limit=limit, categories={}),
            )
        if not approvals:
            return ToolResult(
                "approval_queue_summary",
                True,
                f"Approval queue summary:\n- no readable pending approvals.\n- {unreadable_rows} unreadable approval row(s) hidden for safety.",
                _safe_metadata(count=0, readable_approval_rows=0, unreadable_approval_rows=unreadable_rows, limit=limit, categories={}),
            )

        categories: dict[str, int] = {}
        first = approvals[0]
        first_id = int(first["id"])
        lines = [
            "Approval queue summary:",
            f"- pending: {len(approvals)} shown",
            f"- operator limit: {OPERATOR_LIMIT_RULE}",
            "",
            "First safe handoff:",
            f"- approval: #{first_id} `{first['tool_name']}`",
            f"- readiness: `approval readiness {first_id}`",
            f"- last look: `approval packet {first_id}`",
            f"- proof: `approval chain proof {first_id}`",
            f"- approve only if trusted: `approve approval {first_id}`",
            f"- dismiss if stale/broad/private/surprising: `dismiss approval {first_id}`",
        ]
        if unreadable_rows:
            lines.append(f"- {unreadable_rows} unreadable approval row(s) hidden for safety.")
        for row in approvals:
            category, _concern, safer = _risk_review_for(row["tool_name"], row["user_input"])
            categories[category] = categories.get(category, 0) + 1
            request = _display_text(str(row["user_input"]).replace("\n", " "), 120)
            lines.append(f"- #{row['id']} {_display_text(row['tool_name'])} | {category} | {request}")
            lines.append(f"  Next: approval readiness {row['id']} | approval packet {row['id']} | approval chain proof {row['id']} | approve approval {row['id']} | dismiss approval {row['id']}")
            lines.append(f"  Safer check: {safer}")
            lines.append("  Decision matrix: approve only if exact and verifiable; dismiss stale/broad/private/surprising requests.")
        lines.extend(
            [
                "",
                "Category totals:",
                *[f"- {category}: {count}" for category, count in sorted(categories.items())],
                "",
                "Rule: summary only. It does not approve, dismiss, rerun, execute tools, control the computer, read private data, write files, or queue approvals.",
            ]
        )
        return ToolResult(
            "approval_queue_summary",
            True,
            "\n".join(lines),
            _safe_metadata(
                count=len(approvals),
                readable_approval_rows=len(approvals),
                unreadable_approval_rows=unreadable_rows,
                limit=limit,
                categories=categories,
                approval_ids=[int(row["id"]) for row in approvals],
                first_approval_id=first_id,
                first_approval_tool=first["tool_name"],
                first_readiness_command=f"approval readiness {first_id}",
                first_last_look_command=f"approval packet {first_id}",
                first_proof_command=f"approval chain proof {first_id}",
                first_approve_command=f"approve approval {first_id}",
                first_dismiss_command=f"dismiss approval {first_id}",
                proof_chain_commands_by_approval={
                    str(row["id"]): _approval_proof_chain_commands(row) for row in approvals
                },
            ),
        )

    def approval_readiness_packet(args: dict[str, Any]) -> ToolResult:
        approval_id, error = _resolve_approval_id(store, args)
        if approval_id is None:
            return _bad_approval_id_result("approval_readiness_packet", args, error)
        snapshot = store.approval_readiness_snapshot(approval_id)
        row = snapshot.get("target")
        if row is None:
            return _approval_not_found_result(
                "approval_readiness_packet",
                approval_id,
                f"approval readiness {approval_id}",
                pending_only=False,
            )

        pending_rows = snapshot.get("pending_rows") or []
        pending_approvals, unreadable_pending_rows = _safe_approval_payloads(pending_rows)
        pending_ids = [int(item["id"]) for item in pending_approvals]
        pending_count = int(snapshot.get("pending_count") or 0)
        newest_pending_id = snapshot.get("newest_pending_id")
        queue_revision = int(snapshot.get("queue_revision") or 0)
        planned_args = _planned_args(row)
        age_minutes = _approval_age_minutes(row)
        staleness = _approval_staleness_label(age_minutes)
        category, concern, safer = _risk_review_for(row["tool_name"], row["user_input"])
        status = str(row["status"]).lower()

        if status != "pending":
            verdict = "NOT_PENDING_REVIEW_ONLY"
            next_command = f"approval chain proof {approval_id}"
        elif unreadable_pending_rows:
            verdict = "UNREADABLE_QUEUE_REVIEW_REQUIRED"
            next_command = "pending approvals"
        elif staleness in {"stale", "unknown"}:
            verdict = "DISMISS_OR_RECONFIRM"
            next_command = f"approval packet {approval_id}"
        elif approval_id != newest_pending_id:
            verdict = "REVIEW_NEWER_APPROVALS_FIRST"
            next_command = f"approval readiness {newest_pending_id}"
        else:
            verdict = "LAST_LOOK_REQUIRED"
            next_command = f"approval packet {approval_id}"

        receipt_issued = verdict == "LAST_LOOK_REQUIRED"
        if receipt_issued:
            readiness_receipts[approval_id] = _ApprovalReadinessReceipt(
                approval_id=approval_id,
                queue_revision=queue_revision,
                target_updated_at=str(row["updated_at"]),
                action_digest=approval_action_digest(
                    str(row["tool_name"] or "").strip(),
                    _planned_args(row),
                ),
                expires_at_monotonic=time.monotonic() + APPROVAL_READINESS_RECEIPT_TTL_SECONDS,
            )
            viewed_approval_packets.discard(approval_id)
        else:
            invalidate_pending_review(approval_id)

        proof_chain_commands = _approval_proof_chain_commands(row)
        approval_readiness_handoff = {
            "source": "approval_readiness_packet",
            "approval_id": approval_id,
            "status": row["status"],
            "tool_name": row["tool_name"],
            "category": category,
            "queue": {
                "pending_approvals": pending_count,
                "readable_approval_rows": len(pending_approvals),
                "unreadable_approval_rows": unreadable_pending_rows,
                "pending_ids": pending_ids,
                "newest_pending_approval_id": newest_pending_id,
                "age_minutes": age_minutes,
                "staleness": staleness,
            },
            "planned_arg_keys": sorted(planned_args),
            "verdict": verdict,
            "next_command": next_command,
            "next_required_command": next_command,
            "next_proof_command": f"approval chain proof {approval_id}",
            "proof_chain_commands": proof_chain_commands,
            "approval_packet_required": True,
            "approval_packet_viewed": approval_id in viewed_approval_packets,
            "approval_readiness_receipt_issued": receipt_issued,
            "approval_readiness_receipt_ttl_seconds": APPROVAL_READINESS_RECEIPT_TTL_SECONDS,
            "review_only": True,
            "draft_only": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "approves_request": False,
            "dismisses_request": False,
            "calls_model": False,
            "executes_tools": False,
            "writes_files": False,
            "writes_notes": False,
            "writes_memory": False,
            "reads_private_data": False,
            "external_side_effect": False,
            "controls_computer": False,
            "queues_approval": False,
        }

        lines = [
            f"Approval readiness packet #{approval_id}:",
            "This is a read-only go/no-go packet before any approval can rerun a risky request.",
            "",
            "Queue position:",
            f"- pending approvals visible: {len(pending_approvals)}",
            f"- newest pending approval id: {newest_pending_id if newest_pending_id is not None else 'none'}",
            f"- this approval status: {row['status']}",
            f"- created age minutes: {age_minutes if age_minutes is not None else 'unknown'}",
            f"- staleness: {staleness}",
            "",
            "Stored request:",
            f"- tool: {_display_text(row['tool_name'])}",
            f"- category: {category}",
            f"- request: {_display_text(row['user_input'])}",
            f"- planned arg keys: {', '.join(sorted(planned_args)) if planned_args else 'none'}",
            "",
            "Risk and safer check:",
            f"- why gated: {concern}",
            f"- safer check: {safer}",
            "",
        ]
        if unreadable_pending_rows:
            lines.insert(8, f"- {unreadable_pending_rows} unreadable approval row(s) hidden for safety.")
        planned_lines = _planned_args_lines(row)
        if planned_lines:
            lines.extend(planned_lines)
            lines.append("")
        checklist = _computer_approval_checklist(row["tool_name"], row["user_input"])
        if checklist:
            lines.extend(checklist)
            lines.append("")
        lines.extend(
            [
                *_approval_decision_matrix_lines(row),
                "",
                "Approval readiness verdict:",
                f"- verdict: {verdict}",
                f"- next safe command: `{next_command}`",
                f"- next required command after approval: `approval chain proof {approval_id}`",
                "- approve command stays disabled by process until the last-look packet has been viewed in this runtime.",
                "",
                *_approval_proof_chain_lines(row),
                "",
                "Do not approve if:",
                f"- {OPERATOR_LIMIT_RULE}",
                "- the operator has not reconfirmed a stale request.",
                "- a newer approval is still ahead of this one in the queue.",
                "- the target, command, file, account, coordinates, recipient, or expected result is unclear.",
                "",
                "Boundary:",
                f"- Operator limit: {OPERATOR_LIMIT_RULE}",
                "- This packet is read-only and does not approve, dismiss, rerun, execute tools, control the computer, read private data, write files, write notes, write memory, call external services, or queue approvals.",
            ]
        )
        return ToolResult(
            "approval_readiness_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                approval_id=approval_id,
                status=row["status"],
                tool_name=row["tool_name"],
                category=category,
                age_minutes=age_minutes,
                staleness=staleness,
                pending_approvals=pending_count,
                readable_approval_rows=len(pending_approvals),
                unreadable_approval_rows=unreadable_pending_rows,
                newest_pending_approval_id=newest_pending_id,
                approval_queue_revision=queue_revision,
                planned_arg_keys=sorted(planned_args),
                verdict=verdict,
                next_command=next_command,
                next_required_command=next_command,
                next_proof_command=f"approval chain proof {approval_id}",
                proof_chain_commands=proof_chain_commands,
                approval_packet_required=True,
                approval_packet_viewed=approval_id in viewed_approval_packets,
                approval_readiness_receipt_issued=receipt_issued,
                approval_readiness_receipt_ttl_seconds=APPROVAL_READINESS_RECEIPT_TTL_SECONDS,
                approval_readiness_handoff=approval_readiness_handoff,
                approval_readiness_review_only=True,
                approval_readiness_draft_only=True,
                approval_readiness_loads_without_execution=True,
                approval_readiness_authorizes_execution=False,
                approval_readiness_authorizes_completion_claim=False,
                approval_readiness_approval_granted=False,
            ),
        )

    def approve_pending_approval(args: dict[str, Any]) -> ToolResult:
        approval_id, error = _resolve_approval_id(store, args)
        if approval_id is None:
            return _bad_approval_id_result("approve_pending_approval", args, error)
        row = _get_approval_any_status(store, approval_id)
        if row is None:
            return _approval_not_found_result(
                "approve_pending_approval",
                approval_id,
                f"approve approval {approval_id}",
            )

        def execution_handoff(approved_row, *, resumed: bool) -> ToolResult:
            user_input = approved_row["user_input"]
            planned_args = _planned_args(approved_row)
            sync_path = ""
            sync_exception_type = ""
            try:
                sync_path = sync_pending_approvals()
            except Exception as exc:
                sync_exception_type = type(exc).__name__
            sync_failed = bool(sync_exception_type)
            verb = "Resuming" if resumed else "Approved"
            output = f"{verb} approval #{approval_id}; rerunning: {_display_text(user_input)}"
            if sync_failed:
                output += (
                    "\n\nThe durable approval decision was recorded, but the pending-approvals note "
                    "could not be synchronized. The exact action will still continue through the "
                    "one-shot execution claim; run `pending approvals` afterward to refresh the note."
                )
            resume_contract = {
                **_approval_resume_contract(approved_row, approval_packet_viewed=True),
                "source": "approve_pending_approval_recovery" if resumed else "approve_pending_approval",
                "review_only": False,
                "approves_request": True,
                "writes_files": not sync_failed,
                "writes_notes": not sync_failed,
            }
            handoff_metadata = {
                "rerun_user_input": user_input,
                "rerun_user_input_display": _display_text(user_input),
                "exact_rerun_tool_name": approved_row["tool_name"],
                "exact_rerun_args": _approval_args_public(planned_args),
                "exact_rerun_args_display": _approval_args_display(planned_args),
                "approved_approval_id": approval_id,
                "actual_status": "approved",
                "approval_decision_recorded": True,
                "approval_decision_source": "approved_recovery" if resumed else "pending_cas",
                "approval_recovery_resumed": resumed,
                "approval_recovery_reviewed": resumed,
                "pending_approvals_sync_failed": sync_failed,
                "sync_exception_type": sync_exception_type,
                "approval_packet_required": True,
                "approval_packet_viewed": True,
                "approval_resume_contract": resume_contract,
                "specific_request_only": True,
                "one_shot_approval_rerun": True,
                "approves_request": True,
            }
            if sync_failed:
                handoff_metadata.update(
                    {
                        "next_command": "pending approvals",
                        "recovery_commands": ["pending approvals", f"approval chain proof {approval_id}"],
                    }
                )
                metadata = _safe_metadata(**handoff_metadata)
            else:
                metadata = _approval_write_metadata(
                    **handoff_metadata,
                    **pending_approvals_path_metadata(sync_path),
                )
            return ToolResult("approve_pending_approval", True, output, metadata)

        current_status = str(row["status"] or "").lower()
        if current_status == "approved":
            if store.approval_execution_used(approval_id):
                return ToolResult(
                    "approve_pending_approval",
                    False,
                    f"Approval #{approval_id} is approved, but its one-shot execution has already "
                    f"been claimed. Run `approval chain proof {approval_id}` to inspect the result.",
                    _safe_metadata(
                        reason="approval_already_used",
                        approval_id=approval_id,
                        actual_status="approved",
                        approval_decision_recorded=True,
                        approval_execution_used=True,
                        next_command=f"approval chain proof {approval_id}",
                    ),
                )
            if approval_id not in viewed_approval_packets:
                return ToolResult(
                    "approve_pending_approval",
                    False,
                    (
                        f"Approval #{approval_id} is approved but has no execution claim. Review its "
                        f"last-look packet before recovery.\nRun: approval packet {approval_id}\n"
                        f"Then, if the exact request is still trusted, run: approve approval {approval_id}"
                    ),
                    _safe_metadata(
                        reason="approved_recovery_packet_required",
                        approval_id=approval_id,
                        actual_status="approved",
                        approval_decision_recorded=True,
                        approval_execution_used=False,
                        approval_packet_required=True,
                        approval_packet_viewed=False,
                        next_command=f"approval packet {approval_id}",
                    ),
                )
            if approval_id not in reviewed_approval_recoveries:
                return ToolResult(
                    "approve_pending_approval",
                    False,
                    (
                        f"Approval #{approval_id} has a reviewed last-look packet, but approved-row "
                        f"recovery still requires its resume packet.\nRun: approval resume packet {approval_id}\n"
                        f"Then, if the exact request is still trusted, run: approve approval {approval_id}"
                    ),
                    _safe_metadata(
                        reason="approved_recovery_resume_packet_required",
                        approval_id=approval_id,
                        actual_status="approved",
                        approval_decision_recorded=True,
                        approval_execution_used=False,
                        approval_packet_required=True,
                        approval_packet_viewed=True,
                        approval_recovery_reviewed=False,
                        next_command=f"approval resume packet {approval_id}",
                    ),
                )
            return execution_handoff(row, resumed=True)
        if current_status != "pending":
            return _approval_cas_miss_result(
                "approve_pending_approval",
                approval_id,
                "approve",
                current_status,
            )
        readiness_valid, readiness_reason = readiness_receipt_status(row)
        if not readiness_valid:
            # Another concurrent approver may have persisted a decision while
            # clearing this runtime's freshness receipt. Report the durable
            # winner instead of asking the caller to review a stale pending row.
            actual = _get_approval_any_status(store, approval_id)
            actual_status = str(actual["status"] or "").lower() if actual is not None else ""
            if actual_status and actual_status != "pending":
                return _approval_cas_miss_result(
                    "approve_pending_approval",
                    approval_id,
                    "approve",
                    actual_status,
                )
            return _approval_review_refusal_result(
                "approve_pending_approval",
                (
                    f"Approval #{approval_id} needs readiness review and a last-look packet before it can run.\n"
                    f"Run: approval readiness {approval_id}\n"
                    f"Then: approval packet {approval_id}\n"
                    f"Then, if the request is still trusted, run: approve approval {approval_id}"
                ),
                _safe_metadata(
                    reason=readiness_reason,
                    approval_id=approval_id,
                    approval_readiness_required=True,
                    approval_readiness_valid=False,
                    approval_packet_required=True,
                    approval_packet_viewed=False,
                    next_command=f"approval readiness {approval_id}",
                ),
                commands=(
                    f"approval readiness {approval_id}",
                    f"approval packet {approval_id}",
                    f"approve approval {approval_id}",
                ),
            )
        if approval_id not in viewed_approval_packets:
            # A concurrent approver may have already won the durable CAS and
            # cleared this runtime's last-look state. Do not describe that
            # approved row as still pending; reread it before issuing guidance.
            actual = _get_approval_any_status(store, approval_id)
            actual_status = str(actual["status"] or "").lower() if actual is not None else ""
            if actual_status and actual_status != "pending":
                return _approval_cas_miss_result(
                    "approve_pending_approval",
                    approval_id,
                    "approve",
                    actual_status,
                )
            return _approval_review_refusal_result(
                "approve_pending_approval",
                (
                    f"Approval #{approval_id} has a current readiness receipt but still needs its last-look packet.\n"
                    f"Run: approval packet {approval_id}\n"
                    f"Then, if the request is still trusted, run: approve approval {approval_id}"
                ),
                _safe_metadata(
                    reason="approval_packet_required",
                    approval_id=approval_id,
                    approval_readiness_required=True,
                    approval_readiness_valid=True,
                    approval_packet_required=True,
                    approval_packet_viewed=False,
                    next_command=f"approval packet {approval_id}",
                ),
                commands=(
                    f"approval packet {approval_id}",
                    f"approve approval {approval_id}",
                ),
            )
        receipt = readiness_receipts.get(approval_id)
        if receipt is None:
            invalidate_pending_review(approval_id)
            return _approval_review_refusal_result(
                "approve_pending_approval",
                f"Approval #{approval_id} lost its readiness receipt. Run: approval readiness {approval_id}",
                _safe_metadata(
                    reason="readiness_required",
                    approval_id=approval_id,
                    approval_readiness_required=True,
                    approval_readiness_valid=False,
                    next_command=f"approval readiness {approval_id}",
                ),
                commands=(f"approval readiness {approval_id}",),
            )
        try:
            ok = store.approve_pending_approval_if_ready(
                approval_id,
                expected_queue_revision=receipt.queue_revision,
                expected_updated_at=receipt.target_updated_at,
                expected_action_digest=receipt.action_digest,
            )
        except Exception as exc:
            invalidate_pending_review(approval_id)
            actual = _get_approval_any_status(store, approval_id)
            actual_status = str(actual["status"] or "").lower() if actual is not None else "pending"
            if actual_status != "pending":
                return _approval_cas_miss_result(
                    "approve_pending_approval",
                    approval_id,
                    "approve",
                    actual_status,
                )
            return _approval_update_failure_result(
                "approve_pending_approval",
                approval_id,
                "approve",
                exception_type=type(exc).__name__,
                actual_status=actual_status,
            )
        if not ok:
            invalidate_pending_review(approval_id)
            actual = _get_approval_any_status(store, approval_id)
            if actual is None:
                return _approval_not_found_result(
                    "approve_pending_approval",
                    approval_id,
                    f"approve approval {approval_id}",
                )
            actual_status = str(actual["status"] or "").lower()
            if actual_status != "pending":
                return _approval_cas_miss_result(
                    "approve_pending_approval",
                    approval_id,
                    "approve",
                    actual_status,
                )
            return _approval_review_refusal_result(
                "approve_pending_approval",
                (
                    f"Approval #{approval_id} changed after readiness review and was not approved.\n"
                    f"Run: approval readiness {approval_id}"
                ),
                _safe_metadata(
                    reason="readiness_changed",
                    approval_id=approval_id,
                    actual_status=str(actual["status"] or ""),
                    approval_readiness_required=True,
                    approval_readiness_valid=False,
                    approval_packet_required=True,
                    approval_packet_viewed=False,
                    next_command=f"approval readiness {approval_id}",
                ),
                commands=(f"approval readiness {approval_id}",),
            )
        invalidate_pending_review()
        approved_row = _get_approval_any_status(store, approval_id) or row
        return execution_handoff(approved_row, resumed=False)

    def review_pending_approvals(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 10, 1, 100)
        status = _short(args.get("status") or "pending", 32).lower()
        rows = store.list_pending_approvals(status=status, limit=limit)
        review_metadata = approval_review_metadata(rows)
        path = sync_pending_approvals()
        return ToolResult(
            "review_pending_approvals",
            True,
            build_approval_review(rows),
            _safe_metadata(
                **pending_approvals_path_metadata(path),
                **review_metadata,
                count=review_metadata["approval_review_pending_count"],
                limit=limit,
                status=status,
            ),
        )

    def save_approval_review(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 20, 1, 100)
        status = _short(args.get("status") or "pending", 32).lower()
        with store.approval_review_publication_fence():
            rows = store.list_pending_approvals(status=status, limit=limit)
            body = build_approval_review(rows)
            review_metadata = approval_review_metadata(rows)
            lines = [
                "# Approval Review",
                "",
                f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "",
                body,
                "",
            ]
            path, _content_sha256, _source_revision = vault.write_approval_review_with_evidence(
                "\n".join(lines),
                store_identity=store.get_store_identity(),
                source_payload={
                    "body": body,
                    "limit": limit,
                    "status": status,
                    "review_metadata": review_metadata,
                },
            )
            pending_path = sync_pending_approvals()
        path_display = _safe_vault_path_display(path, vault)
        pending_path_display = _safe_vault_path_display(pending_path, vault)
        return ToolResult(
            "save_approval_review",
            True,
            f"Approval review saved: {path_display}\n\n{body}",
            _approval_write_metadata(
                path=str(path),
                path_display=path_display,
                pending_approvals_path=pending_path,
                pending_approvals_path_display=pending_path_display,
                **review_metadata,
                count=review_metadata["approval_review_pending_count"],
                limit=limit,
                status=status,
            ),
        )

    return (
        list_pending_approvals,
        dismiss_pending_approval,
        inspect_pending_approval,
        approval_execution_packet,
        approval_resume_packet,
        approval_history,
        approval_chain_proof,
        approval_queue_summary,
        approval_readiness_packet,
        approve_pending_approval,
        review_pending_approvals,
        save_approval_review,
    )
