from __future__ import annotations

import json
from functools import wraps
from pathlib import Path
import re
import unicodedata
from datetime import date, datetime, timedelta
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
    LOCAL_READ_INPUT_RECOVERY_ACTION,
    RESOURCE_NOT_FOUND_RECOVERY_ACTION,
    declare_failure_guidance,
    declare_retryable_local_read_failure,
    declare_resource_not_found_failure,
)

from jarvis_v2.agent.types import ToolResult
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import (
    MAX_TASK_COMPLETION_EVIDENCE_CHARS,
    MAX_TASK_COMPLETION_RUN_ID_CHARS,
    MemoryStore,
    TaskRecord,
    normalized_task_identity,
)
from jarvis_v2.tools.audit_meta import RECOVERY_CLOSURE_META_TOOLS
from jarvis_v2.tools.notes import is_protected_profile_path_or_content


MAX_TASK_LIMIT = 200
MAX_TASK_BODY_CHARS = 600
MAX_TASK_DUE_CHARS = 120
MAX_TASK_QUERY_CHARS = 240
MAX_NOTE_PATH_CHARS = 500
MAX_TASK_IMPORT_NOTE_CHARS = 100_000
MAX_TASK_IMPORT_CANDIDATES = 200
MAX_TASK_COMMAND_CHARS = 180
MAX_SQLITE_TASK_ID = 9223372036854775807
LOCAL_PATH_RE = re.compile(
    r"/(?:Users|private|var/folders|tmp)/[^\n\r]*",
    re.IGNORECASE,
)
MONTH_DUE_FORMATS = ("%B %d %Y", "%b %d %Y", "%B %d, %Y", "%b %d, %Y")
_MISSING_ROW_VALUE = object()
TASK_STATUS_LABELS = {
    "open": "Reopened",
    "done": "Completed",
    "paused": "Paused",
    "dropped": "Dropped",
}
TASK_INPUT_RECOVERY_ACTION = (
    "Correct the reported task input, then submit a new task request through the normal policy."
)
TASK_STATE_RECOVERY_ACTION = (
    "Run the named task inspection command, resolve the reported blocker, then submit a new task request through the normal policy."
)


def _bounded_int(value: Any, default: int, low: int = 1, high: int = MAX_TASK_LIMIT) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _raw_limit_metadata(value: Any, *, limit: int) -> dict[str, Any]:
    if value is None:
        return {"limit": limit}
    if isinstance(value, bool):
        return {"limit": limit, "raw_limit": str(value)}
    try:
        int(value)
    except (TypeError, ValueError):
        return {"limit": limit, "raw_limit": _short_metadata(value, limit=80)}
    return {"limit": limit}


def _short(value: Any, *, limit: int = MAX_TASK_BODY_CHARS) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _short_raw(value: Any, *, limit: int = 80) -> str:
    text = "" if value is None else str(value).strip()
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _short_metadata(value: Any, *, limit: int = 80) -> str:
    text = _short_raw(value, limit=limit)
    return LOCAL_PATH_RE.sub("<local-path>", text)


def _row_value(row: Any, key: str, default: Any = _MISSING_ROW_VALUE) -> Any:
    try:
        return row[key]
    except Exception:
        return default


def _row_text(row: Any, key: str, default: str = "", *, limit: int = MAX_TASK_BODY_CHARS) -> str:
    value = _row_value(row, key)
    if value is _MISSING_ROW_VALUE:
        return default
    text = _short_metadata(value, limit=limit)
    return text if text else default


def _row_int(row: Any, key: str) -> int | None:
    value = _row_value(row, key)
    if value is _MISSING_ROW_VALUE or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _note_path_metadata(value: Any) -> str:
    text = "" if value is None else str(value)
    display_safe = "".join(
        " " if unicodedata.category(char) in {"Cc", "Cf", "Cs", "Zl", "Zp"} else char
        for char in text
    )
    return _short_metadata(display_safe, limit=MAX_NOTE_PATH_CHARS)


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "calls_model": False,
        "executes_tools": False,
        "queues_approval": False,
        "approves_request": False,
        "dismisses_request": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "executes_side_effect": False,
        "writes_files": False,
        "writes_memory": False,
        "writes_notes": False,
        "controls_computer": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    metadata.update(extra)
    return metadata


def _task_write_metadata(**extra: Any) -> dict[str, Any]:
    metadata = _safe_metadata(**extra)
    metadata.update({"writes_files": True, "writes_memory": True, "writes_notes": True})
    return metadata


def _task_known_no_change_failure(
    tool_name: str,
    output: str,
    metadata: dict[str, Any],
    *,
    action: str = TASK_INPUT_RECOVERY_ACTION,
) -> ToolResult:
    """Attach canonical recovery guidance to a rejected task request.

    This helper is only for failures whose task/file/database effects have not
    started. It deliberately does not authorize a retry; a corrected request
    still goes through the normal policy.
    """

    public_output = str(output).strip()
    command = _safe_task_command(metadata.get("next_safe_command"))
    commands: tuple[str, ...] = ()
    if command:
        commands = (command,)
        if command not in public_output:
            public_output = f"{public_output} Run `{command}`."
    if action not in public_output:
        public_output = f"{public_output} {action}"
    result_metadata = dict(metadata)
    result_metadata.update(
        {
            "outcome_known": True,
            "outcome_unknown": False,
            "execution_outcome_unknown": False,
            "side_effect_possible": False,
            "retry_safe": True,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
            "authorizes_task_mutation": False,
            "state_changed": False,
            "writes_database": False,
            "auto_mutation_effects_started": False,
        }
    )
    result_metadata = declare_failure_guidance(
        result_metadata,
        output=public_output,
        action=action,
        commands=commands,
    )
    return ToolResult(tool_name, False, public_output, result_metadata)


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _safe_task_command(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = re.sub(r"[\r\n\t]+", " ", value.strip())
    if not text or LOCAL_PATH_RE.search(text):
        return ""
    if len(text) > MAX_TASK_COMMAND_CHARS:
        return text[: MAX_TASK_COMMAND_CHARS - 1].rstrip() + "…"
    return text


def _sanitize_task_next_commands(raw_next: Any) -> tuple[dict[str, Any] | list[str], list[str], int]:
    hidden = 0
    if isinstance(raw_next, dict):
        sanitized: dict[str, Any] = {}
        next_safe_commands: list[str] = []
        for key, value in raw_next.items():
            key_text = _short_raw(key, limit=80)
            if isinstance(value, (list, tuple)):
                commands: list[str] = []
                for item in value:
                    command = _safe_task_command(item)
                    if command:
                        commands.append(command)
                    elif item not in (None, ""):
                        hidden += 1
                sanitized[key_text] = commands
                next_safe_commands.extend(commands)
                continue
            command = _safe_task_command(value)
            if command:
                sanitized[key_text] = command
                next_safe_commands.append(command)
            elif value in (None, ""):
                sanitized[key_text] = ""
            else:
                hidden += 1
        return sanitized, next_safe_commands, hidden
    if isinstance(raw_next, str):
        command = _safe_task_command(raw_next)
        return ([command] if command else []), ([command] if command else []), 0 if command or not raw_next.strip() else 1
    try:
        iterator = iter(raw_next or [])
    except TypeError:
        return [], [], 1 if raw_next is not None else 0
    commands: list[str] = []
    for value in iterator:
        command = _safe_task_command(value)
        if command:
            commands.append(command)
        elif value not in (None, ""):
            hidden += 1
    return commands, commands, hidden


def _task_contract(
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


def _task_handoff_metadata(
    handoff_key: str,
    handoff: dict[str, Any],
    *,
    writes: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    _normalize_task_handoff(handoff)
    prefix = handoff_key.removesuffix("_handoff")
    metadata = _task_write_metadata(**extra) if writes else _safe_metadata(**extra)
    metadata.update(
        _task_contract(
            state_changed=_metadata_bool(handoff.get("state_changed")),
            changed=list(handoff.get("changed") or []),
            content_in_handoff=_metadata_bool(handoff.get("content_in_handoff")),
        )
    )
    metadata[f"{handoff_key}_ready"] = True
    metadata[f"{prefix}_handoff_ready"] = True
    metadata[f"{prefix}_ready_for_operator"] = True
    metadata[f"{prefix}_state_changed"] = _metadata_bool(handoff.get("state_changed"))
    metadata[f"{prefix}_changed"] = list(handoff.get("changed") or [])
    metadata[f"{prefix}_content_in_handoff"] = _metadata_bool(handoff.get("content_in_handoff"))
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


def _normalize_task_handoff(handoff: dict[str, Any]) -> dict[str, Any]:
    raw_next = handoff.get("next_commands") or []
    sanitized_next, next_safe_commands, hidden_next_commands = _sanitize_task_next_commands(raw_next)
    next_safe_command = next_safe_commands[0] if next_safe_commands else ""
    handoff["next_commands"] = sanitized_next
    handoff["handoff_ready"] = True
    handoff["next_safe_command"] = next_safe_command
    handoff["next_safe_commands"] = list(next_safe_commands)
    handoff["next_safe_command_count"] = len(next_safe_commands)
    handoff["hidden_next_command_count"] = hidden_next_commands
    return handoff


def _safe_vault_path_display(path: str | Path | None, vault: ObsidianVault) -> str:
    if path is None:
        return ""
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    for root in (vault.root_path, vault.root_path.resolve()):
        try:
            return candidate.relative_to(root).as_posix()
        except ValueError:
            continue
    return _short_metadata(candidate, limit=160)


def _bad_task_id_metadata(value: Any) -> dict[str, Any]:
    return _safe_metadata(task_id=None, raw_task_id=_short_metadata(value, limit=80))


def _parse_task_id(value: Any) -> tuple[int | None, str | None]:
    if isinstance(value, bool):
        return None, "task_id must be a number."
    try:
        task_id = int(value)
    except (TypeError, ValueError):
        return None, "task_id must be a number."
    if task_id <= 0:
        return None, "task_id must be a positive number."
    if task_id > MAX_SQLITE_TASK_ID:
        return None, "task_id is too large."
    return task_id, None


def _parse_due_date_label(value: Any, *, today: date | None = None) -> date | None:
    label = str(value or "").strip()
    if not label:
        return None
    low = " ".join(label.lower().replace("’", "'").split())
    base = today or datetime.now().astimezone().date()
    if low in {"yesterday"}:
        return base - timedelta(days=1)
    if low in {"today"}:
        return base
    if low in {"tomorrow"}:
        return base + timedelta(days=1)
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", low):
        try:
            return datetime.strptime(low, "%Y-%m-%d").date()
        except ValueError:
            return None
    if re.fullmatch(r"\d{4}/\d{1,2}/\d{1,2}", low):
        try:
            return datetime.strptime(low, "%Y/%m/%d").date()
        except ValueError:
            return None
    month_label = re.sub(r"\s+", " ", label.replace(",", ", ")).strip()
    month_label = re.sub(r"\s+,", ",", month_label)
    for fmt in MONTH_DUE_FORMATS:
        try:
            return datetime.strptime(month_label, fmt).date()
        except ValueError:
            continue
    return None


def _row_metadata(row: Any) -> dict[str, Any]:
    try:
        if "metadata" not in row.keys():
            return {}
        parsed = json.loads(row["metadata"] or "{}")
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


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


def _truthy_metadata(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _is_approval_held_tool_run(row: Any) -> bool:
    try:
        if bool(row["ok"]):
            return False
    except Exception:
        return False
    metadata = _row_metadata(row)
    raw_values = (
        metadata.get("failure_kind"),
        metadata.get("failure_stage"),
        metadata.get("status"),
        metadata.get("reason"),
    )
    normalized = " ".join(str(value or "").lower().replace("-", "_") for value in raw_values)
    if "approval" in normalized and any(token in normalized for token in ("required", "requires", "gate", "gated", "held")):
        return True
    return any(_truthy_metadata(metadata.get(key)) for key in ("requires_confirmation", "requires_approval", "approval_required"))


def _task_recent_run_status(row: Any) -> str:
    try:
        if bool(row["ok"]):
            return "ok"
    except Exception:
        return "unreadable"
    return "approval held" if _is_approval_held_tool_run(row) else "failed/blocked"


def _approval_id_for_run(row: Any) -> int | None:
    candidates: list[Any] = []
    try:
        if "approval_id" in row.keys():
            candidates.append(row["approval_id"])
    except Exception:
        pass
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
    return [
        f"approval readiness {approval_id}",
        f"approval packet {approval_id}",
        f"approval chain proof {approval_id}",
    ]


def _recent_audit_context_counts(rows: list[Any]) -> dict[str, int]:
    counts = {"ok": 0, "failed": 0, "approval_held": 0, "unreadable": 0}
    for row in rows:
        status = _task_recent_run_status(row)
        if status == "ok":
            counts["ok"] += 1
        elif status == "approval held":
            counts["approval_held"] += 1
        elif status == "failed/blocked":
            counts["failed"] += 1
        else:
            counts["unreadable"] += 1
    return counts


def _append_unique(items: list[str], candidates: list[str]) -> None:
    for candidate in candidates:
        if candidate and candidate not in items:
            items.append(candidate)


def _execution_health_recovery_closure_snapshot(recent_runs: list[Any]) -> dict[str, Any]:
    risky_levels = {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"}
    action_runs = [
        row
        for row in recent_runs
        if str(row["tool_name"]) not in RECOVERY_CLOSURE_META_TOOLS
        and not (
            _row_metadata(row).get("auto_mutation_definite_no_effect") is True
            or (
                _row_metadata(row).get("handler_invoked") is False
                and str(_row_metadata(row).get("failure_kind") or "").startswith(
                    "auto_mutation_"
                )
            )
        )
    ]
    newest_problem = None
    for row in action_runs:
        risk = str(row["risk"])
        ok = bool(row["ok"])
        approved = bool(row["approved"])
        approval_id = row["approval_id"] if "approval_id" in row.keys() else None
        if not ok or (risk in risky_levels and (not approved or approval_id is None)):
            newest_problem = row
            break

    target = newest_problem or (action_runs[0] if action_runs else None)
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
    output_approval_ids = _ids_from_text(
        target["output"] if target is not None else "",
        ("approval #", "approval id", "approval packet", "queued as approval"),
    )
    approval_proof_ids: list[int] = []
    target_metadata_approval_id = (
        _row_metadata(target).get("approval_id") if target is not None else None
    )
    for candidate in (
        ([int(target_approval_id)] if target_approval_id is not None else [])
        + ([int(target_metadata_approval_id)] if type(target_metadata_approval_id) is int else [])
        + output_approval_ids
    ):
        if candidate not in approval_proof_ids:
            approval_proof_ids.append(candidate)
    target_verification_runs = [
        row
        for row in recent_runs
        if str(row["tool_name"]) == "verification_receipt"
        and bool(row["ok"])
        and _row_metadata(row).get("run_id") == target_run_id
    ]
    target_recovery_runs = [
        row
        for row in recent_runs
        if str(row["tool_name"]) == "execution_recovery_packet"
        and bool(row["ok"])
        and _row_metadata(row).get("run_id") == target_run_id
    ]
    target_learning_runs = [
        row
        for row in recent_runs
        if str(row["tool_name"]) == "after_action_learning_packet"
        and bool(row["ok"])
        and _row_metadata(row).get("run_id") == target_run_id
    ]
    target_approval_proof_runs = [
        row
        for row in recent_runs
        if str(row["tool_name"]) == "approval_chain_proof"
        and bool(row["ok"])
        and _row_metadata(row).get("valid_execution_proof") is True
        and _row_metadata(row).get("verdict") == "APPROVAL_CHAIN_PROVEN"
        and _row_metadata(row).get("approval_id") in approval_proof_ids
    ]

    missing: list[str] = []
    commands: list[str] = []
    if target_run_id is not None and newest_problem is not None:
        if not target_verification_runs:
            missing.append("target_verification_receipt")
            commands.append(f"verification receipt {target_run_id}")
        if not target_recovery_runs:
            missing.append("target_recovery_packet")
            commands.append(f"execution recovery packet {target_run_id}")
        if not target_learning_runs:
            missing.append("target_after_action_learning_packet")
            commands.append(f"execution learning closure {target_run_id}")
            commands.append(f"after-action learning packet {target_run_id}")
        if approval_problem and not target_approval_proof_runs:
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
    elif newest_problem is None:
        state = "not_needed"
    elif missing:
        state = "blocked_missing_" + "_and_".join(missing)
    else:
        state = "ready_for_operator_retry_review"

    return {
        "target_run_id": target_run_id,
        "target_tool_name": target_tool_name,
        "state": state,
        "ready_to_retry": state == "ready_for_operator_retry_review",
        "missing": missing,
        "missing_count": len(missing),
        "required_commands": commands,
        "next_required_command": commands[0] if commands else "",
        "blocks_task_completion": state not in {"no_recent_execution", "not_needed", "ready_for_operator_retry_review"},
    }


def _recovery_closure_checklist_command(recovery_closure: dict[str, Any]) -> str:
    return "recovery closure checklist" if recovery_closure.get("blocks_task_completion") else ""


def _task_read_boundary() -> dict[str, bool]:
    return {
        "changes_task_status": False,
        "completes_task": False,
        "pauses_task": False,
        "reopens_task": False,
        "drops_task": False,
        "imports_tasks": False,
        "exports_tasks": False,
        "writes_files": False,
        "writes_memory": False,
        "writes_notes": False,
        "calls_model": False,
        "executes_tools": False,
        "queues_approval": False,
        "approves_request": False,
        "dismisses_request": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "executes_side_effect": False,
        "controls_computer": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _task_summary_payload(task: dict[str, Any]) -> dict[str, Any]:
    task_id = int(task["id"])
    return {
        "id": task_id,
        "status": _short_metadata(task["status"], limit=40),
        "priority": _short_metadata(task["priority"], limit=40),
        "due": _short_metadata(task["due"], limit=MAX_TASK_DUE_CHARS),
        "source": _short_metadata(task["source"], limit=120),
        "body_chars": len(str(task["body"] or "")),
        "already_done": str(task["status"]).lower() == "done",
    }


def _safe_task_summary_payload(task: Any) -> dict[str, Any] | None:
    task_id = _row_int(task, "id")
    if task_id is None:
        return None
    status = _row_text(task, "status", "unknown", limit=40)
    priority = _row_text(task, "priority", "normal", limit=40)
    due = _row_text(task, "due", "", limit=MAX_TASK_DUE_CHARS)
    source = _row_text(task, "source", "", limit=120)
    body = _row_text(task, "body", "Unreadable task", limit=MAX_TASK_BODY_CHARS)
    return {
        "id": task_id,
        "status": status,
        "priority": priority,
        "due": due,
        "source": source,
        "body": body,
        "body_chars": len(body),
        "already_done": status.lower() == "done",
    }


def _safe_task_summary_payloads(tasks: list[Any]) -> tuple[list[dict[str, Any]], int]:
    summaries: list[dict[str, Any]] = []
    unreadable = 0
    for task in tasks:
        summary = _safe_task_summary_payload(task)
        if summary is None:
            unreadable += 1
        else:
            summaries.append(summary)
    return summaries, unreadable


def _format_task_summary(task: dict[str, Any]) -> str:
    due = f" | due {task['due']}" if task["due"] else ""
    priority = f" | {task['priority']}" if task["priority"] != "normal" else ""
    return f"- #{task['id']} [{task['status']}] {task['body']}{due}{priority}"


def _task_list_handoff_payload(
    *,
    source: str,
    status: str,
    limit: int,
    tasks: list[dict[str, Any]],
    unreadable_task_rows: int = 0,
) -> dict[str, Any]:
    first_task_id = int(tasks[0]["id"]) if tasks else None
    return {
        "source": source,
        "status": status,
        "limit": limit,
        "task_count": len(tasks),
        "readable_task_rows": len(tasks),
        "unreadable_task_rows": unreadable_task_rows,
        "task_ids": [row["id"] for row in tasks],
        "first_task_id": first_task_id,
        "tasks": tasks,
        "next_commands": {
            "show_first_task": f"show task {first_task_id}" if first_task_id is not None else "",
            "completion_packet_first_task": f"task completion packet {first_task_id}" if first_task_id is not None else "",
            "list_open": "list tasks",
            "task_overview": "task overview",
            "task_board": "task board",
            "next_task": "next task",
        },
        "boundaries": _task_read_boundary(),
        **_task_contract(content_in_handoff=bool(tasks)),
    }


def _task_inspection_handoff_payload(task: dict[str, Any]) -> dict[str, Any]:
    task_id = int(task["id"])
    return {
        "source": "inspect_task",
        "task_id": task_id,
        "task": _task_summary_payload(task),
        "next_commands": {
            "show": f"show task {task_id}",
            "completion_packet": f"task completion packet {task_id}",
            "complete_with_evidence": f"complete task {task_id} with evidence: <what proves this task is done>",
            "pause": f"pause task {task_id}",
            "reopen": f"reopen task {task_id}",
        },
        "boundaries": _task_read_boundary(),
        **_task_contract(content_in_handoff=True),
    }


def _task_overview_handoff_payload(
    *,
    limit: int,
    rows: list[dict[str, Any]],
    status_counts: dict[str, int],
    priority_counts: dict[str, int],
    high_open: list[dict[str, Any]],
    due_open: list[dict[str, Any]],
    recent_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    first_open = next((row for row in rows if str(row["status"]).lower() == "open"), None)
    first_open_id = int(first_open["id"]) if first_open is not None else None
    return {
        "source": "task_overview",
        "limit": limit,
        "total": len(rows),
        "status_counts": dict(status_counts),
        "priority_counts": dict(priority_counts),
        "open_count": status_counts.get("open", 0),
        "paused_count": status_counts.get("paused", 0),
        "done_count": status_counts.get("done", 0),
        "dropped_count": status_counts.get("dropped", 0),
        "high_priority_count": priority_counts.get("high", 0),
        "high_open_task_ids": [int(row["id"]) for row in high_open],
        "due_open_task_ids": [int(row["id"]) for row in due_open],
        "recent_task_ids": [int(row["id"]) for row in recent_rows],
        "first_open_task_id": first_open_id,
        "next_commands": {
            "show_first_open": f"show task {first_open_id}" if first_open_id is not None else "",
            "completion_packet_first_open": f"task completion packet {first_open_id}" if first_open_id is not None else "",
            "list_open": "list tasks",
            "task_board": "task board",
            "next_task": "next task",
        },
        "boundaries": _task_read_boundary(),
        **_task_contract(content_in_handoff=bool(rows)),
    }


def _next_task_handoff_payload(task: dict[str, Any] | None, reason_bits: list[str]) -> dict[str, Any]:
    if task is None:
        return {
            "source": "next_task",
            "task_id": None,
            "task": None,
            "reason": [],
            "task_selected": False,
            "next_commands": {
                "add_task": "add task <task body>",
                "task_overview": "task overview",
                "task_board": "task board",
            },
            "boundaries": _task_read_boundary(),
            **_task_contract(content_in_handoff=False),
        }
    task_id = int(task["id"])
    return {
        "source": "next_task",
        "task_id": task_id,
        "task": _task_summary_payload(task),
        "reason": list(reason_bits),
        "task_selected": True,
        "next_commands": {
            "show": f"show task {task_id}",
            "completion_packet": f"task completion packet {task_id}",
            "complete_with_evidence": f"complete task {task_id} with evidence: <what proves this task is done>",
            "pause": f"pause task {task_id}",
            "task_overview": "task overview",
            "task_board": "task board",
        },
        "boundaries": _task_read_boundary(),
        **_task_contract(content_in_handoff=True),
    }


def _task_board_handoff_payload(
    *,
    limit: int,
    board: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    status_counts = {status: len(rows) for status, rows in board.items()}
    first_open = board.get("open", [])[0] if board.get("open") else None
    first_open_id = int(first_open["id"]) if first_open is not None else None
    return {
        "source": "task_board",
        "limit": limit,
        "statuses": list(board.keys()),
        "status_counts": status_counts,
        "open_count": status_counts.get("open", 0),
        "paused_count": status_counts.get("paused", 0),
        "done_count": status_counts.get("done", 0),
        "dropped_count": status_counts.get("dropped", 0),
        "board": {
            status: [_task_summary_payload(row) for row in rows]
            for status, rows in board.items()
        },
        "first_open_task_id": first_open_id,
        "next_commands": {
            "show_first_open": f"show task {first_open_id}" if first_open_id is not None else "",
            "completion_packet_first_open": f"task completion packet {first_open_id}" if first_open_id is not None else "",
            "list_open": "list tasks",
            "task_overview": "task overview",
            "next_task": "next task",
        },
        "boundaries": _task_read_boundary(),
        **_task_contract(content_in_handoff=any(board.values())),
    }


def _task_mutation_handoff_payload(
    *,
    source: str,
    task: dict[str, Any],
    mutation: str,
    path: str | Path | None,
    changed: list[str] | None = None,
    evidence: str = "",
    verification_run_id: str = "",
    recovery_closure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_id = int(task["id"])
    recovery_commands = list((recovery_closure or {}).get("required_commands") or [])
    return {
        "source": source,
        "mutation": mutation,
        "task_id": task_id,
        "task": _task_summary_payload(task),
        "changed": list(changed or []),
        "path_display": _note_path_metadata(path) if path else "",
        "evidence": {
            "supplied": bool(evidence or verification_run_id),
            "evidence_chars": len(evidence),
            "verification_run_id_chars": len(verification_run_id),
            "verification_run_supplied": bool(verification_run_id),
        },
        "recovery_closure": {
            "state": (recovery_closure or {}).get("state", ""),
            "blocks_task_completion": bool((recovery_closure or {}).get("blocks_task_completion", False)),
            "proof_queue": recovery_commands,
            "proof_queue_count": len(recovery_commands),
            "next_proof_command": (recovery_closure or {}).get("next_required_command", ""),
            "checklist_command": _recovery_closure_checklist_command(recovery_closure or {}),
        },
        "next_commands": {
            "show": f"show task {task_id}",
            "completion_packet": f"task completion packet {task_id}",
            "task_overview": "task overview",
            "task_board": "task board",
            "export_tasks": "export tasks",
        },
        "boundaries": {
            "changes_task_status": mutation in {"status_update", "evidence_completion"},
            "completes_task": str(task["status"]).lower() == "done" and mutation in {"status_update", "evidence_completion"},
            "writes_files": True,
            "writes_memory": True,
            "writes_notes": True,
            "calls_model": False,
            "executes_tools": False,
            "queues_approval": False,
            "approves_request": False,
            "dismisses_request": False,
            "reads_private_data": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "controls_computer": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
        },
        **_task_contract(state_changed=True, changed=list(changed or []), content_in_handoff=True),
    }


def _task_refusal_handoff_payload(
    *,
    source: str,
    reason: str,
    mutation: str,
    task_id: int | None = None,
    raw_field: str | None = None,
    raw_value: Any = None,
    next_command: str = "",
) -> dict[str, Any]:
    handoff = {
        "source": source,
        "reason": reason,
        "mutation": mutation,
        "task_id": task_id,
        "changed": [],
        "refused": True,
        "next_commands": {
            "retry": next_command,
            "list_tasks": "list tasks",
            "task_board": "task board",
            "next_task": "next task",
        },
        "boundaries": {
            "changes_task_status": False,
            "completes_task": False,
            "writes_files": False,
            "writes_memory": False,
            "writes_notes": False,
            "calls_model": False,
            "executes_tools": False,
            "queues_approval": False,
            "approves_request": False,
            "dismisses_request": False,
            "reads_private_data": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "controls_computer": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
        },
        **_task_contract(content_in_handoff=raw_field is not None),
    }
    if raw_field:
        handoff["raw_field"] = raw_field
        handoff["raw_value"] = _short_metadata(raw_value, limit=80)
    return handoff


def _task_refusal_metadata(
    *,
    source: str,
    reason: str,
    mutation: str,
    task_id: int | None = None,
    raw_field: str | None = None,
    raw_value: Any = None,
    next_command: str = "",
    **extra: Any,
) -> dict[str, Any]:
    handoff = _normalize_task_handoff(
        _task_refusal_handoff_payload(
            source=source,
            reason=reason,
            mutation=mutation,
            task_id=task_id,
            raw_field=raw_field,
            raw_value=raw_value,
            next_command=next_command,
        )
    )
    metadata = _safe_metadata(
        reason=reason,
        task_id=task_id,
        task_refusal_handoff=handoff,
        task_refusal_handoff_ready=True,
        task_mutation_handoff_ready=False,
        task_refusal_ready_for_operator=True,
        task_refusal_state_changed=False,
        task_refusal_changed=[],
        task_refusal_content_in_handoff=_metadata_bool(handoff.get("content_in_handoff")),
        task_refusal_next_safe_command=handoff["next_safe_command"],
        task_refusal_next_safe_commands=list(handoff["next_safe_commands"]),
        task_refusal_next_safe_command_count=handoff["next_safe_command_count"],
        task_refusal_authorizes_execution=False,
        task_refusal_authorizes_completion_claim=False,
        task_refusal_approval_granted=False,
        next_safe_command=handoff["next_safe_command"],
        next_safe_commands=list(handoff["next_safe_commands"]),
        next_safe_command_count=handoff["next_safe_command_count"],
        **extra,
    )
    if raw_field:
        metadata[f"raw_{raw_field}"] = _short_metadata(raw_value, limit=80)
    handoff = metadata.get("task_refusal_handoff")
    if isinstance(handoff, dict):
        metadata.update(
            _task_contract(
                state_changed=False,
                changed=[],
                content_in_handoff=_metadata_bool(handoff.get("content_in_handoff")),
            )
        )
    return metadata


def _missing_task_result(
    *,
    tool_name: str,
    task_id: int,
    mutation: str,
    retry_command: str,
    **extra: Any,
) -> ToolResult:
    verify_command = "show task <correct task id>"
    recovery_commands = ["list tasks", verify_command]
    if retry_command and retry_command not in recovery_commands:
        recovery_commands.append(retry_command)
    output = (
        f"No task found for #{task_id}. Run `list tasks` to refresh task IDs, then "
        f"`{verify_command}` to verify the correct record."
    )
    if retry_command != verify_command:
        output += f" After that, run `{retry_command}` through the normal policy."
    else:
        output += " Use that command through the normal policy."
    output = f"{output} {RESOURCE_NOT_FOUND_RECOVERY_ACTION}"
    metadata = _task_refusal_metadata(
        source=tool_name,
        reason="missing_task",
        mutation=mutation,
        task_id=task_id,
        next_command="list tasks",
        **extra,
    )
    metadata.update(
        {
            "next_command": "list tasks",
            "recovery_commands": recovery_commands,
            "retry_requires_task_refresh": True,
            "retry_requires_corrected_id": True,
            "recovery_commands_require_normal_policy": True,
            "authorizes_retry": False,
            "authorizes_task_mutation": False,
        }
    )
    metadata = declare_resource_not_found_failure(
        metadata,
        output=output,
        action=RESOURCE_NOT_FOUND_RECOVERY_ACTION,
    )
    return ToolResult(tool_name, False, output, metadata)


def _note_task_not_found_result(
    *,
    tool_name: str,
    raw_path: str,
    metadata: dict[str, Any] | None = None,
) -> ToolResult:
    safe_raw_path = _note_path_metadata(raw_path)
    command_prefix = "preview" if tool_name == "preview_tasks_from_note" else "import"
    retry_command = f"{command_prefix} tasks from note {safe_raw_path}"
    recovery_commands = ["list jarvis notes", retry_command]
    result_metadata = dict(metadata or _safe_metadata(reason="note_not_found"))
    result_metadata.update(
        {
            "reason": "note_not_found",
            "raw_path": safe_raw_path,
            "next_command": recovery_commands[0],
            "recovery_commands": recovery_commands,
            "retry_requires_note_refresh": True,
            "recovery_commands_require_normal_policy": True,
            "authorizes_retry": False,
            "authorizes_task_mutation": False,
        }
    )
    policy_suffix = (
        " through the normal local-safe policy"
        if tool_name == "import_tasks_from_note"
        else ""
    )
    output = (
        f"Jarvis note not found: {safe_raw_path}. Run `list jarvis notes` to confirm the note "
        f"path, then retry `{retry_command}`{policy_suffix}. "
        f"{RESOURCE_NOT_FOUND_RECOVERY_ACTION}"
    )
    result_metadata = declare_resource_not_found_failure(
        result_metadata,
        output=output,
        action=RESOURCE_NOT_FOUND_RECOVERY_ACTION,
    )
    return ToolResult(
        tool_name,
        False,
        output,
        result_metadata,
    )


def _note_task_preview_handoff_payload(
    *,
    note_path: Path,
    vault: ObsidianVault,
    open_tasks: list[str],
    completed_tasks: list[str],
    new_tasks: list[str],
    duplicates: list[str],
) -> dict[str, Any]:
    path_display = _safe_vault_path_display(note_path, vault)
    return {
        "source": "preview_tasks_from_note",
        "path_display": path_display,
        "open_checkbox_count": len(open_tasks),
        "completed_checkbox_count": len(completed_tasks),
        "new_importable_count": len(new_tasks),
        "duplicate_count": len(duplicates),
        "new_task_previews": [_short_metadata(body, limit=160) for body in new_tasks[:12]],
        "duplicate_previews": [_short_metadata(body, limit=160) for body in duplicates[:12]],
        "completed_task_previews": [_short_metadata(body, limit=160) for body in completed_tasks[:12]],
        "next_commands": {
            "import": f"import tasks from note {path_display}",
            "preview": f"preview tasks from note {path_display}",
            "task_overview": "task overview",
            "task_board": "task board",
        },
        "boundaries": _task_read_boundary(),
        **_task_contract(
            content_in_handoff=bool(open_tasks or completed_tasks or new_tasks or duplicates),
        ),
    }


def _note_task_import_handoff_payload(
    *,
    note_path: Path,
    tasks_path: Path,
    vault: ObsidianVault,
    priority: str,
    imported: list[str],
    skipped: list[str],
) -> dict[str, Any]:
    note_display = _safe_vault_path_display(note_path, vault)
    tasks_display = _safe_vault_path_display(tasks_path, vault)
    return {
        "source": "import_tasks_from_note",
        "note_path_display": note_display,
        "tasks_path_display": tasks_display,
        "priority": priority,
        "imported_count": len(imported),
        "skipped_count": len(skipped),
        "imported_previews": [_short_metadata(body, limit=160) for body in imported[:12]],
        "skipped_previews": [_short_metadata(body, limit=160) for body in skipped[:12]],
        "next_commands": {
            "task_overview": "task overview",
            "task_board": "task board",
            "list_open": "list tasks",
            "export_tasks": "export tasks",
            "preview_again": f"preview tasks from note {note_display}",
        },
        "boundaries": {
            "imports_tasks": True,
            "writes_files": True,
            "writes_memory": True,
            "writes_notes": True,
            "changes_task_status": False,
            "completes_task": False,
            "calls_model": False,
            "executes_tools": False,
            "queues_approval": False,
            "approves_request": False,
            "dismisses_request": False,
            "reads_private_data": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "controls_computer": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
        },
        **_task_contract(
            state_changed=bool(imported),
            changed=["tasks"] if imported else [],
            content_in_handoff=bool(imported or skipped),
        ),
    }


def _task_export_handoff_payload(
    *,
    path_display: str,
    limit: int,
    exported_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    tasks = [_task_summary_payload(row) for row in exported_tasks]
    return {
        "source": "export_tasks",
        "path_display": path_display,
        "limit": limit,
        "exported_count": len(tasks),
        "task_ids": [row["id"] for row in tasks],
        "tasks": tasks,
        "next_commands": {
            "list_open": "list tasks",
            "task_overview": "task overview",
            "task_board": "task board",
            "next_task": "next task",
            "export_again": "export tasks",
        },
        "boundaries": {
            "exports_tasks": True,
            "writes_files": True,
            "writes_memory": False,
            "writes_notes": True,
            "imports_tasks": False,
            "changes_task_status": False,
            "completes_task": False,
            "calls_model": False,
            "executes_tools": False,
            "queues_approval": False,
            "approves_request": False,
            "dismisses_request": False,
            "reads_private_data": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "controls_computer": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
        },
        **_task_contract(
            state_changed=True,
            changed=["task_export"],
            content_in_handoff=bool(tasks),
        ),
    }


def export_tasks_auto_mutation_operation_key(_args: dict[str, Any]) -> dict[str, str]:
    return {"projection": "open_tasks"}


def task_status_auto_mutation_operation_key(
    args: dict[str, Any],
) -> dict[str, int]:
    task_id, _error = _parse_task_id(args.get("task_id"))
    return {"task_id": task_id or 0}


def _normalized_task_status(
    args: dict[str, Any],
    *,
    fixed_status: str | None = None,
) -> str:
    value = fixed_status if fixed_status is not None else args.get("status")
    return str(value or "").strip().lower()


def _task_status_retry_command(
    *,
    tool_name: str,
    task_id: int | str,
    status: str,
) -> str:
    if tool_name == "complete_task":
        return f"complete task {task_id}"
    commands = {
        "open": f"reopen task {task_id}",
        "done": f"complete task {task_id}",
        "paused": f"pause task {task_id}",
        "dropped": f"drop task {task_id}",
    }
    return commands.get(status, f"task {task_id} <open|done|paused|dropped>")


def _make_task_status_auto_mutation_preflight(
    store: MemoryStore,
    *,
    fixed_status: str | None = None,
):
    def preflight(args: dict[str, Any]) -> str | None:
        task_id, _error = _parse_task_id(args.get("task_id"))
        if task_id is None:
            return "bad_task_id"
        if _normalized_task_status(args, fixed_status=fixed_status) not in TASK_STATUS_LABELS:
            return "bad_status"
        if store.get_task(task_id) is None:
            return "missing_task"
        return None

    return preflight


def make_complete_task_auto_mutation_preflight(store: MemoryStore):
    return _make_task_status_auto_mutation_preflight(store, fixed_status="done")


def _completion_evidence_values(args: dict[str, Any]) -> tuple[str, str]:
    return (
        args.get("evidence") if type(args.get("evidence")) is str else "",
        (
            args.get("verification_run_id")
            if type(args.get("verification_run_id")) is str
            else ""
        ),
    )


def _completion_evidence_refusal_reason(
    evidence: str,
    verification_run_id: str,
) -> str | None:
    if len(evidence) > MAX_TASK_COMPLETION_EVIDENCE_CHARS:
        return "evidence_too_long"
    if len(verification_run_id) > MAX_TASK_COMPLETION_RUN_ID_CHARS:
        return "verification_run_id_too_long"
    if evidence and not _task_completion_text_is_visible(evidence):
        return "invalid_evidence"
    if verification_run_id and not _task_completion_text_is_visible(verification_run_id):
        return "invalid_evidence"
    if not evidence and not verification_run_id:
        return "missing_evidence"
    return None


def _task_completion_text_is_visible(value: str) -> bool:
    try:
        value.encode("utf-8", errors="strict")
        normalized = unicodedata.normalize("NFKC", value)
    except UnicodeError:
        return False
    allowed_controls = {"\t", "\r", "\n"}
    for text in (value, normalized):
        for char in text:
            category = unicodedata.category(char)
            if category in {"Cf", "Cs", "Zl", "Zp"} or (
                category == "Cc" and char not in allowed_controls
            ):
                return False
    return any(
        not char.isspace()
        and unicodedata.category(char) not in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        for char in normalized
    )


def make_complete_task_with_evidence_auto_mutation_preflight(store: MemoryStore):
    def preflight(args: dict[str, Any]) -> str | None:
        task_id, _error = _parse_task_id(args.get("task_id"))
        if task_id is None:
            return "bad_task_id"
        if store.get_task(task_id) is None:
            return "missing_task"
        evidence, verification_run_id = _completion_evidence_values(args)
        refusal_reason = _completion_evidence_refusal_reason(
            evidence,
            verification_run_id,
        )
        if refusal_reason is not None:
            return refusal_reason
        recovery_runs, _problem_run_id, _proof_fence = store.task_completion_recovery_snapshot(
            RECOVERY_CLOSURE_META_TOOLS
        )
        recovery_closure = _execution_health_recovery_closure_snapshot(recovery_runs)
        if recovery_closure["blocks_task_completion"]:
            return "recovery_closure_required"
        return None

    return preflight


def make_update_task_status_auto_mutation_preflight(store: MemoryStore):
    return _make_task_status_auto_mutation_preflight(store)


def _task_status_semantic_preflight_result(
    result: ToolResult,
    args: dict[str, Any],
) -> ToolResult:
    result.metadata.update(
        {
            "failure_kind": "auto_mutation_semantic_preflight_rejected",
            "requires_confirmation": False,
            "requires_approval": False,
            "executed_handler": False,
            "handler_invoked": False,
            "planned_arg_keys": sorted(str(key)[:80] for key in args),
            "authorizes_retry": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "queues_approval": False,
            "state_changed": False,
            "auto_mutation_effects_started": False,
            "writes_files": False,
            "writes_database": False,
            "writes_memory": False,
            "writes_notes": False,
            "external_side_effect": False,
            "controls_computer": False,
        }
    )
    if not result.ok and "recovery_guidance" not in result.metadata:
        action = (
            TASK_STATE_RECOVERY_ACTION
            if result.metadata.get("reason") == "recovery_closure_required"
            else TASK_INPUT_RECOVERY_ACTION
        )
        return _task_known_no_change_failure(
            result.tool_name,
            result.output,
            result.metadata,
            action=action,
        )
    return result


def _task_status_auto_mutation_preflight_result(
    args: dict[str, Any],
    reason: str,
    *,
    tool_name: str,
    fixed_status: str | None = None,
) -> ToolResult:
    task_id, error = _parse_task_id(args.get("task_id"))
    raw_status = fixed_status if fixed_status is not None else args.get("status")
    status = _normalized_task_status(args, fixed_status=fixed_status)
    if reason == "missing_task" and task_id is not None:
        result = _missing_task_result(
            tool_name=tool_name,
            task_id=task_id,
            mutation="status_update",
            retry_command=_task_status_retry_command(
                tool_name=tool_name,
                task_id="<correct task id>",
                status=status,
            ),
            status=status,
            auto_mutation_effects_started=False,
            writes_database=False,
        )
    elif reason == "bad_status" and task_id is not None:
        result = _task_known_no_change_failure(
            tool_name,
            "Task status must be open, done, paused, or dropped.",
            _task_refusal_metadata(
                source=tool_name,
                reason="bad_status",
                mutation="status_update",
                task_id=task_id,
                raw_field="status",
                raw_value=raw_status,
                next_command=f"task {task_id} <open|done|paused|dropped>",
                status=_short_metadata(raw_status, limit=80),
            ),
        )
    else:
        result = _task_known_no_change_failure(
            tool_name,
            error or "task_id must be a positive number.",
            _task_refusal_metadata(
                source=tool_name,
                reason="bad_task_id",
                mutation="status_update",
                raw_field="task_id",
                raw_value=args.get("task_id"),
                next_command=_task_status_retry_command(
                    tool_name=tool_name,
                    task_id="<task id>",
                    status=status,
                ),
            ),
        )
    return _task_status_semantic_preflight_result(result, args)


def complete_task_auto_mutation_preflight_result(
    args: dict[str, Any],
    reason: str,
) -> ToolResult:
    return _task_status_auto_mutation_preflight_result(
        args,
        reason,
        tool_name="complete_task",
        fixed_status="done",
    )


def complete_task_with_evidence_auto_mutation_preflight_result(
    args: dict[str, Any],
    reason: str,
) -> ToolResult:
    task_id, error = _parse_task_id(args.get("task_id"))
    if reason == "missing_task" and task_id is not None:
        result = _missing_task_result(
            tool_name="complete_task_with_evidence",
            task_id=task_id,
            mutation="evidence_completion",
            retry_command="complete task <correct task id> with evidence: <proof>",
            has_evidence=True,
            auto_mutation_effects_started=False,
            writes_database=False,
        )
    elif reason in {
        "missing_evidence",
        "evidence_too_long",
        "verification_run_id_too_long",
        "invalid_evidence",
    } and task_id is not None:
        refusal_messages = {
            "missing_evidence": "Completion evidence is required",
            "evidence_too_long": "Completion evidence is too long",
            "verification_run_id_too_long": "The verification run id is too long",
            "invalid_evidence": "Completion evidence must contain visible text",
        }
        result = _task_known_no_change_failure(
            "complete_task_with_evidence",
            (
                f"{refusal_messages[reason]} before closing task #{task_id}. "
                f"Run `task completion packet {task_id}` first."
            ),
            _task_refusal_metadata(
                source="complete_task_with_evidence",
                reason=reason,
                mutation="evidence_completion",
                task_id=task_id,
                next_command=f"task completion packet {task_id}",
                has_evidence=reason != "missing_evidence",
            ),
        )
    elif reason == "recovery_closure_required" and task_id is not None:
        result = _task_known_no_change_failure(
            "complete_task_with_evidence",
            (
                f"Task #{task_id} is not ready to close yet. Execution-health recovery "
                f"closure is still required; run `task completion packet {task_id}`."
            ),
            _task_refusal_metadata(
                source="complete_task_with_evidence",
                reason="recovery_closure_required",
                mutation="evidence_completion",
                task_id=task_id,
                next_command=f"task completion packet {task_id}",
                has_evidence=True,
                recovery_closure_blocks_task_completion=True,
            ),
            action=TASK_STATE_RECOVERY_ACTION,
        )
    else:
        result = _task_known_no_change_failure(
            "complete_task_with_evidence",
            error or "task_id must be a positive number.",
            _task_refusal_metadata(
                source="complete_task_with_evidence",
                reason="bad_task_id",
                mutation="evidence_completion",
                raw_field="task_id",
                raw_value=args.get("task_id"),
                next_command="complete task <task id> with evidence: <proof>",
            ),
        )
    return _task_status_semantic_preflight_result(result, args)


def update_task_status_auto_mutation_preflight_result(
    args: dict[str, Any],
    reason: str,
) -> ToolResult:
    return _task_status_auto_mutation_preflight_result(
        args,
        reason,
        tool_name="update_task_status",
    )


def _normalized_task_detail_updates(
    args: dict[str, Any],
) -> tuple[str | None, str | None, str | None]:
    body = _short(args["body"]) if "body" in args else None
    due = _short(args["due"], limit=MAX_TASK_DUE_CHARS) if "due" in args else None
    priority = str(args["priority"]).strip().lower() if "priority" in args else None
    return body, due, priority


def _task_detail_text_is_utf8_safe(*values: str | None) -> bool:
    try:
        for value in values:
            if value is not None:
                value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _task_detail_retry_command(
    args: dict[str, Any],
    task_id: int | str,
    *,
    reason: str = "",
) -> str:
    if reason == "bad_priority":
        return f"set task {task_id} priority <low|normal|high>"
    if reason in {"missing_body", "missing_update"}:
        return f"rename task {task_id}: <body>"
    if reason == "invalid_unicode":
        body, due, _priority = _normalized_task_detail_updates(args)
        if body is not None and not _task_detail_text_is_utf8_safe(body):
            return f"rename task {task_id}: <body>"
        if due is not None and not _task_detail_text_is_utf8_safe(due):
            return f"set task {task_id} due <due>"
    if "body" in args:
        return f"rename task {task_id}: <body>"
    if "due" in args:
        return f"set task {task_id} due <due>"
    if "priority" in args:
        return f"set task {task_id} priority <low|normal|high>"
    return f"rename task {task_id}: <body>"


def _task_body_display(value: Any) -> str:
    text = _short_metadata(value, limit=MAX_TASK_BODY_CHARS)
    display = "".join(
        " " if unicodedata.category(char) in {"Cc", "Cf", "Cs", "Zl", "Zp"} else char
        for char in text
    )
    return LOCAL_PATH_RE.sub("<local-path>", display)


def update_task_details_auto_mutation_operation_key(
    args: dict[str, Any],
) -> dict[str, int]:
    return {"task_id": args["task_id"]}


def make_update_task_details_auto_mutation_preflight(store: MemoryStore):
    def preflight(args: dict[str, Any]) -> str | None:
        task_id, _error = _parse_task_id(args.get("task_id"))
        if task_id is None:
            return "bad_task_id"
        body, due, priority = _normalized_task_detail_updates(args)
        if body is None and due is None and priority is None:
            return "missing_update"
        if body is not None and not body:
            return "missing_body"
        if priority is not None and priority not in {"low", "normal", "high"}:
            return "bad_priority"
        if not _task_detail_text_is_utf8_safe(body, due):
            return "invalid_unicode"
        if store.get_task(task_id) is None:
            return "missing_task"
        return None

    return preflight


def update_task_details_auto_mutation_preflight_result(
    args: dict[str, Any],
    reason: str,
) -> ToolResult:
    task_id, error = _parse_task_id(args.get("task_id"))
    _body, _due, priority = _normalized_task_detail_updates(args)
    if reason == "missing_task" and task_id is not None:
        result = _missing_task_result(
            tool_name="update_task_details",
            task_id=task_id,
            mutation="field_update",
            retry_command=_task_detail_retry_command(
                args,
                "<correct task id>",
                reason=reason,
            ),
        )
    elif reason == "missing_body" and task_id is not None:
        result = _task_known_no_change_failure(
            "update_task_details",
            "Task body cannot be empty.",
            _task_refusal_metadata(
                source="update_task_details",
                reason=reason,
                mutation="field_update",
                task_id=task_id,
                raw_field="body",
                raw_value=args.get("body"),
                next_command=_task_detail_retry_command(args, task_id, reason=reason),
            ),
        )
    elif reason == "bad_priority" and task_id is not None:
        result = _task_known_no_change_failure(
            "update_task_details",
            "Priority must be low, normal, or high.",
            _task_refusal_metadata(
                source="update_task_details",
                reason=reason,
                mutation="field_update",
                task_id=task_id,
                raw_field="priority",
                raw_value=args.get("priority"),
                next_command=_task_detail_retry_command(args, task_id, reason=reason),
                priority=priority,
            ),
        )
    elif reason == "missing_update" and task_id is not None:
        result = _task_known_no_change_failure(
            "update_task_details",
            "Give Jarvis a task body, due label, or priority to update.",
            _task_refusal_metadata(
                source="update_task_details",
                reason=reason,
                mutation="field_update",
                task_id=task_id,
                next_command=_task_detail_retry_command(args, task_id, reason=reason),
            ),
        )
    elif reason == "invalid_unicode" and task_id is not None:
        result = _task_known_no_change_failure(
            "update_task_details",
            "Task body and due label must contain valid Unicode text.",
            _task_refusal_metadata(
                source="update_task_details",
                reason=reason,
                mutation="field_update",
                task_id=task_id,
                next_command=_task_detail_retry_command(args, task_id, reason=reason),
            ),
        )
    else:
        result = _task_known_no_change_failure(
            "update_task_details",
            error or "task_id must be a positive number.",
            _task_refusal_metadata(
                source="update_task_details",
                reason="bad_task_id",
                mutation="field_update",
                raw_field="task_id",
                raw_value=args.get("task_id"),
                next_command=_task_detail_retry_command(
                    args,
                    "<task id>",
                    reason="bad_task_id",
                ),
            ),
        )
    result.metadata.update(
        {
            "failure_kind": "auto_mutation_semantic_preflight_rejected",
            "requires_confirmation": False,
            "requires_approval": False,
            "executed_handler": False,
            "handler_invoked": False,
            "planned_arg_keys": sorted(str(key)[:80] for key in args),
            "authorizes_retry": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "queues_approval": False,
            "state_changed": False,
            "auto_mutation_effects_started": False,
            "writes_files": False,
            "writes_database": False,
            "writes_memory": False,
            "writes_notes": False,
            "external_side_effect": False,
            "controls_computer": False,
        }
    )
    if not result.ok and "recovery_guidance" not in result.metadata:
        return _task_known_no_change_failure(
            result.tool_name,
            result.output,
            result.metadata,
        )
    return result


def _raw_task_import_path(args: dict[str, Any]) -> str:
    return str(args.get("path") or "").strip()


def _task_import_priority(args: dict[str, Any]) -> str:
    return str(args.get("priority") or "normal").strip().lower()


def _task_import_prewrite_metadata(**extra: Any) -> dict[str, Any]:
    return _safe_metadata(
        auto_mutation_effects_started=False,
        state_changed=False,
        writes_database=False,
        **extra,
    )


def _task_import_candidate_limit_metadata(candidate_count: int) -> dict[str, int]:
    return {
        "open_task_candidates": candidate_count,
        "max_open_task_candidates": MAX_TASK_IMPORT_CANDIDATES,
    }


def _task_import_prewrite_failure(
    output: str,
    metadata: dict[str, Any],
    *,
    action: str | None = None,
    commands: tuple[str, ...] = (),
) -> ToolResult:
    """Declare a task-import refusal before receipt claim or mutation.

    Task import is an auto-mutation, so even deterministic validation failures
    must not look like permission to replay an old execution. A corrected
    request remains subject to the normal policy and a fresh receipt chain.
    """

    public_action = action or TASK_INPUT_RECOVERY_ACTION
    public_output = str(output).strip()
    if public_action not in public_output:
        public_output = f"{public_output} {public_action}"
    result_metadata = dict(metadata)
    if "priority" in result_metadata:
        result_metadata["priority"] = _short_metadata(
            result_metadata["priority"],
            limit=80,
        )
    result_metadata.update(
        {
            "outcome_known": True,
            "outcome_unknown": False,
            "execution_outcome_unknown": False,
            "side_effect_possible": False,
            "retry_safe": True,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
            "authorizes_task_mutation": False,
        }
    )
    result_metadata = declare_failure_guidance(
        result_metadata,
        output=public_output,
        action=public_action,
        commands=commands,
    )
    return ToolResult("import_tasks_from_note", False, public_output, result_metadata)


def _protected_profile_task_result(tool_name: str, *, priority: str = "") -> ToolResult:
    metadata = (
        _task_import_prewrite_metadata(
            reason="protected_profile",
            priority=priority,
        )
        if tool_name == "import_tasks_from_note"
        else _safe_metadata(reason="protected_profile")
    )
    metadata.update(
        {
            "next_command": "read profile",
            "recovery_commands": ["read profile"],
            "authorizes_retry": False,
            "authorizes_task_mutation": False,
        }
    )
    output = (
        "That content is reserved for profile access. Use `read profile` instead. "
        f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
    )
    metadata = declare_retryable_local_read_failure(
        metadata,
        output=output,
        action=LOCAL_READ_INPUT_RECOVERY_ACTION,
        commands=("read profile",),
    )
    return ToolResult(
        tool_name,
        False,
        output,
        metadata,
    )


def _note_task_preview_failure(
    message: str,
    *,
    reason: str,
    raw_path: Any = None,
    exception_type: str = "",
    **extra: Any,
) -> ToolResult:
    action = (
        LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION
        if reason == "note_read_failed"
        else LOCAL_READ_INPUT_RECOVERY_ACTION
    )
    output = f"{message} {action}"
    details: dict[str, Any] = {"reason": reason, **extra}
    if raw_path is not None:
        details["raw_path"] = _note_path_metadata(raw_path)
    if exception_type:
        details["exception_type"] = exception_type
    metadata = declare_retryable_local_read_failure(
        _safe_metadata(**details),
        output=output,
        action=action,
    )
    return ToolResult("preview_tasks_from_note", False, output, metadata)


def import_tasks_from_note_auto_mutation_operation_key(
    args: dict[str, Any],
) -> dict[str, str]:
    raw_path = _raw_task_import_path(args).strip("\"'")
    candidate = Path(raw_path)
    if candidate.suffix.lower() != ".md":
        candidate = candidate.with_suffix(".md")
    target = unicodedata.normalize("NFKC", candidate.as_posix()).casefold()
    return {"target": target}


def make_import_tasks_from_note_auto_mutation_preflight(
    vault: ObsidianVault,
):
    def preflight(args: dict[str, Any]) -> str | None:
        raw_path = _raw_task_import_path(args)
        if not raw_path:
            return "missing_path"
        if len(raw_path) > MAX_NOTE_PATH_CHARS:
            return "path_too_large"
        if _task_import_priority(args) not in {"low", "normal", "high"}:
            return "bad_priority"
        try:
            note_path = _lexical_note_path(vault.root_path, raw_path)
            if is_protected_profile_path_or_content(
                note_path,
                vault_root=vault.root_path,
            ):
                return "protected_profile"
            target_exists = vault.note_matches_exact_bytes(note_path, b"")
        except IsADirectoryError:
            return "note_not_regular"
        except ValueError:
            return "unsafe_path"
        except OSError:
            return "note_read_failed"
        if target_exists is None:
            return "note_not_found"
        try:
            text = vault.read_note_bounded(
                note_path,
                max_chars=MAX_TASK_IMPORT_NOTE_CHARS,
            )
        except OverflowError:
            # Preserve the handler's existing bounded-note refusal and receipt release path.
            return None
        except IsADirectoryError:
            return "note_not_regular"
        except ValueError:
            return "unsafe_path"
        except OSError:
            return "note_read_failed"
        if text is None:
            return "note_not_found"
        if is_protected_profile_path_or_content(
            note_path,
            text,
            vault_root=vault.root_path,
        ):
            return "protected_profile"
        if len(_open_markdown_tasks(text)) > MAX_TASK_IMPORT_CANDIDATES:
            return "too_many_tasks"
        return None

    return preflight


def make_import_tasks_from_note_auto_mutation_preflight_result(
    vault: ObsidianVault,
):
    def refusal(args: dict[str, Any], reason: str) -> ToolResult:
        raw_path_value = _raw_task_import_path(args)
        raw_path = _short(raw_path_value, limit=MAX_NOTE_PATH_CHARS)
        priority = _task_import_priority(args)
        safe_raw_path = _note_path_metadata(raw_path)
        retry_command = f"import tasks from note {safe_raw_path}"
        recovery_commands = ["list jarvis notes", retry_command]
        metadata = _task_import_prewrite_metadata(
            reason=reason,
            priority=priority,
        )
        if reason != "protected_profile":
            metadata["raw_path"] = safe_raw_path
        if reason == "note_not_found":
            metadata.update(
                {
                    "next_command": recovery_commands[0],
                    "recovery_commands": recovery_commands,
                    "retry_requires_note_refresh": True,
                    "recovery_commands_require_normal_policy": True,
                }
            )
        if reason == "path_too_large":
            metadata.update(
                {
                    "path_chars": len(raw_path_value),
                    "max_path_chars": MAX_NOTE_PATH_CHARS,
                }
            )
        if reason == "note_too_large":
            metadata["max_chars"] = MAX_TASK_IMPORT_NOTE_CHARS
        if reason == "too_many_tasks":
            metadata["max_open_task_candidates"] = MAX_TASK_IMPORT_CANDIDATES
            try:
                note_path = _lexical_note_path(vault.root_path, raw_path_value)
                text = vault.read_note_bounded(
                    note_path,
                    max_chars=MAX_TASK_IMPORT_NOTE_CHARS,
                )
            except (IsADirectoryError, OSError, OverflowError, ValueError):
                text = None
            if text is not None:
                metadata.update(
                    _task_import_candidate_limit_metadata(
                        len(_open_markdown_tasks(text))
                    )
                )
        metadata.update(
            {
                "failure_kind": "auto_mutation_semantic_preflight_rejected",
                "requires_confirmation": False,
                "requires_approval": False,
                "executed_handler": False,
                "handler_invoked": False,
                "planned_arg_keys": sorted(str(key)[:80] for key in args),
                "authorizes_retry": False,
                "authorizes_execution": False,
                "authorizes_completion_claim": False,
                "approval_granted": False,
                "state_changed": False,
                "writes_database": False,
                "external_side_effect": False,
            }
        )
        if reason == "note_not_found":
            return _note_task_not_found_result(
                tool_name="import_tasks_from_note",
                raw_path=raw_path_value,
                metadata=metadata,
            )
        outputs = {
            "missing_path": "Note path is required.",
            "path_too_large": (
                f"Note path is too long; limit is {MAX_NOTE_PATH_CHARS} characters."
            ),
            "bad_priority": "Priority must be low, normal, or high.",
            "unsafe_path": "Note path must stay inside the Jarvis Obsidian folder.",
            "note_not_regular": "Jarvis note must be a regular Markdown file.",
            "note_too_large": (
                "Jarvis note is too large to import safely; "
                f"limit is {MAX_TASK_IMPORT_NOTE_CHARS} characters."
            ),
            "too_many_tasks": (
                "Jarvis note has too many open task candidates to import safely; "
                f"limit is {MAX_TASK_IMPORT_CANDIDATES}. Nothing was written."
            ),
            "note_read_failed": (
                "Could not read Jarvis note for task import. Check JARVIS_OBSIDIAN_VAULT and "
                "vault permissions, run `setup check`, then retry."
            ),
            "protected_profile": (
                "That content is reserved for profile access. Use `read profile` instead."
            ),
        }
        if reason == "protected_profile":
            metadata.update(
                {
                    "next_command": "read profile",
                    "recovery_commands": ["read profile"],
                }
            )
        output = outputs.get(
            reason,
            "Task import failed deterministic validation; nothing was written.",
        )
        if reason == "note_read_failed":
            output = f"{output} {LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION}"
            return _task_import_prewrite_failure(
                output,
                metadata,
                action=LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
                commands=("setup check",),
            )
        if reason == "protected_profile":
            output = f"{output} {LOCAL_READ_INPUT_RECOVERY_ACTION}"
            return _task_import_prewrite_failure(
                output,
                metadata,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                commands=("read profile",),
            )
        return _task_import_prewrite_failure(output, metadata)

    return refusal


def _task_search_handoff_payload(
    *,
    query: str,
    status: str,
    limit: int,
    matches: list[dict[str, Any]],
    unreadable_task_rows: int = 0,
) -> dict[str, Any]:
    first_task_id = int(matches[0]["id"]) if matches else None
    return {
        "source": "search_tasks",
        "query": query,
        "status": status,
        "limit": limit,
        "match_count": len(matches),
        "readable_task_rows": len(matches),
        "unreadable_task_rows": unreadable_task_rows,
        "matched_task_ids": [row["id"] for row in matches],
        "first_match_task_id": first_task_id,
        "matches": matches,
        "next_commands": {
            "show_first_match": f"show task {first_task_id}" if first_task_id is not None else "",
            "completion_packet_first_match": f"task completion packet {first_task_id}" if first_task_id is not None else "",
            "refine_search": f"search tasks {query}",
            "list_open": "list tasks",
        },
        "boundaries": _task_read_boundary(),
        **_task_contract(content_in_handoff=bool(matches)),
    }


def _task_overdue_handoff_payload(
    *,
    as_of: date,
    limit: int,
    overdue: list[dict[str, Any]],
    unparsed_due: list[dict[str, Any]],
    parsed_count: int,
) -> dict[str, Any]:
    overdue_summaries = [_task_summary_payload(row) for row in overdue]
    unparsed_summaries = [_task_summary_payload(row) for row in unparsed_due]
    first_task_id = int(overdue[0]["id"]) if overdue else None
    return {
        "source": "overdue_tasks",
        "as_of": as_of.isoformat(),
        "limit": limit,
        "parsed_due_count": parsed_count,
        "overdue_count": len(overdue),
        "unparsed_due_count": len(unparsed_due),
        "overdue_task_ids": [row["id"] for row in overdue_summaries],
        "unparsed_due_task_ids": [row["id"] for row in unparsed_summaries],
        "first_overdue_task_id": first_task_id,
        "overdue_tasks": overdue_summaries,
        "unparsed_due_tasks": unparsed_summaries,
        "next_commands": {
            "show_first_overdue": f"show task {first_task_id}" if first_task_id is not None else "",
            "completion_packet_first_overdue": f"task completion packet {first_task_id}" if first_task_id is not None else "",
            "list_open": "list tasks",
            "task_overview": "task overview",
        },
        "boundaries": _task_read_boundary(),
        **_task_contract(content_in_handoff=bool(overdue or unparsed_due)),
    }


def _task_completion_handoff_payload(
    *,
    task: dict[str, Any],
    has_evidence: bool,
    evidence: str,
    verification_run_id: str,
    recent_tool_runs_count: int,
    recent_audit_counts: dict[str, int],
    recovery_closure: dict[str, Any],
    verdict: str,
) -> dict[str, Any]:
    recovery_commands = list(recovery_closure.get("required_commands") or [])
    checklist_command = _recovery_closure_checklist_command(recovery_closure)
    task_id = int(task["id"])
    return {
        "source": "task_completion_packet",
        "task_id": task_id,
        "task": {
            "id": task_id,
            "status": task["status"],
            "priority": task["priority"],
            "due": task["due"],
            "body_chars": len(str(task["body"] or "")),
            "already_done": str(task["status"]).lower() == "done",
        },
        "verdict": verdict,
        "has_evidence": has_evidence,
        "evidence": {
            "supplied": bool(evidence),
            "evidence_chars": len(evidence),
            "verification_run_id_chars": len(verification_run_id),
            "verification_run_supplied": bool(verification_run_id),
        },
        "recent_audit_context": {
            "recent_tool_runs": recent_tool_runs_count,
            "recent_tool_runs_inspected": recent_tool_runs_count,
            "recent_ok_tool_runs": int(recent_audit_counts.get("ok") or 0),
            "recent_failed_tool_runs": int(recent_audit_counts.get("failed") or 0),
            "recent_approval_held_tool_runs": int(recent_audit_counts.get("approval_held") or 0),
            "recent_unreadable_tool_run_rows": int(recent_audit_counts.get("unreadable") or 0),
        },
        "recovery_closure": {
            "state": recovery_closure["state"],
            "ready_to_retry": recovery_closure["ready_to_retry"],
            "missing": list(recovery_closure["missing"]),
            "missing_count": recovery_closure["missing_count"],
            "required_commands": recovery_commands,
            "proof_queue": recovery_commands,
            "proof_queue_count": len(recovery_commands),
            "next_required_command": recovery_closure["next_required_command"],
            "next_proof_command": recovery_closure["next_required_command"],
            "checklist_command": checklist_command,
            "should_open_checklist": bool(checklist_command),
            "blocks_task_completion": recovery_closure["blocks_task_completion"],
            "target_run_id": recovery_closure["target_run_id"],
            "target_tool_name": recovery_closure["target_tool_name"],
        },
        "next_commands": {
            "complete_with_evidence": f"complete task {task_id} with evidence: <what proves this task is done>",
            "complete_with_verification_run": f"complete task {task_id} with verification run <id>: <expected outcome>",
            "recovery_closure_checklist": checklist_command,
            "next_recovery_proof": recovery_closure["next_required_command"],
            "recovery_proof_queue": recovery_commands,
        },
        "completion_allowed": verdict == "READY_TO_COMPLETE",
        "already_done": verdict == "ALREADY_DONE",
        "evidence_required": verdict == "EVIDENCE_REQUIRED",
        "recovery_closure_required": verdict == "RECOVERY_CLOSURE_REQUIRED",
        "boundaries": {
            "completes_task": False,
            "writes_files": False,
            "writes_memory": False,
            "writes_notes": False,
            "calls_model": False,
            "executes_tools": False,
            "queues_approval": False,
            "approves_request": False,
            "dismisses_request": False,
            "reads_private_data": False,
            "reads_personal_data": False,
            "executes_side_effect": False,
            "controls_computer": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
        },
        **_task_contract(content_in_handoff=True),
    }


def make_task_tools(store: MemoryStore, vault: ObsidianVault):
    def add_task(args: dict[str, Any]) -> ToolResult:
        raw_body = args.get("body")
        body = _short(raw_body)
        due = _short(args.get("due"), limit=MAX_TASK_DUE_CHARS)
        raw_priority = args.get("priority") or "normal"
        priority = str(raw_priority).strip().lower()
        if not body:
            return _task_known_no_change_failure(
                "add_task",
                "Task body is required.",
                _task_refusal_metadata(
                    source="add_task",
                    reason="missing_body",
                    mutation="task_create",
                    raw_field="body",
                    raw_value=raw_body,
                    next_command="add task <body>",
                ),
            )
        if priority not in {"low", "normal", "high"}:
            return _task_known_no_change_failure(
                "add_task",
                "Priority must be low, normal, or high.",
                _task_refusal_metadata(
                    source="add_task",
                    reason="bad_priority",
                    mutation="task_create",
                    raw_field="priority",
                    raw_value=raw_priority,
                    next_command="add task <body> priority <low|normal|high>",
                    priority=priority,
                ),
            )
        task_id = store.add_task(TaskRecord(body=body, due=due, priority=priority, source="jarvis-v2"))
        path = vault.sync_open_tasks(store)
        return ToolResult(
            "add_task",
            True,
            f"Added task #{task_id}: {body}",
            _task_handoff_metadata(
                "task_mutation_handoff",
                _task_mutation_handoff_payload(
                    source="add_task",
                    task=store.get_task(task_id),
                    mutation="task_create",
                    path=path,
                    changed=["task"],
                ),
                writes=True,
                task_id=task_id,
                path=str(path),
                body_chars=len(body),
                due=due,
                priority=priority,
            ),
        )

    def list_tasks(args: dict[str, Any]) -> ToolResult:
        raw_status = args.get("status") or "open"
        status = str(raw_status).strip().lower()
        if status == "all":
            status_filter = None
        elif status in {"open", "done", "paused", "dropped"}:
            status_filter = status
        else:
            failure_output = (
                "Task status must be open, done, paused, dropped, or all. "
                f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "list_tasks",
                False,
                failure_output,
                declare_retryable_local_read_failure(
                    _safe_metadata(
                        reason="bad_status",
                        status=status,
                        raw_status=_short_metadata(raw_status, limit=80),
                    ),
                    output=failure_output,
                    action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                ),
            )
        limit = _bounded_int(args.get("limit"), 25)
        rows = store.list_tasks(status=status_filter, limit=limit)
        task_summaries, unreadable_task_rows = _safe_task_summary_payloads(rows)
        if not rows:
            return ToolResult(
                "list_tasks",
                True,
                f"No {status} tasks.",
                _task_handoff_metadata(
                    "task_list_handoff",
                    _task_list_handoff_payload(source="list_tasks", status=status, limit=limit, tasks=[]),
                    count=0,
                    status=status,
                    readable_task_rows=0,
                    unreadable_task_rows=0,
                    **_raw_limit_metadata(args.get("limit"), limit=limit),
                ),
            )
        if not task_summaries:
            return ToolResult(
                "list_tasks",
                True,
                f"No readable {status} tasks. {unreadable_task_rows} unreadable task row(s) hidden for safety.",
                _task_handoff_metadata(
                    "task_list_handoff",
                    _task_list_handoff_payload(
                        source="list_tasks",
                        status=status,
                        limit=limit,
                        tasks=[],
                        unreadable_task_rows=unreadable_task_rows,
                    ),
                    count=0,
                    status=status,
                    readable_task_rows=0,
                    unreadable_task_rows=unreadable_task_rows,
                    **_raw_limit_metadata(args.get("limit"), limit=limit),
                ),
            )
        count_line = f"Count: {len(task_summaries)}"
        if unreadable_task_rows:
            count_line += f" readable ({unreadable_task_rows} hidden)"
        lines = ["Tasks:", count_line]
        lines.extend(_format_task_summary(row) for row in task_summaries)
        if unreadable_task_rows:
            lines.append(f"- {unreadable_task_rows} unreadable task row(s) hidden for safety.")
        return ToolResult(
            "list_tasks",
            True,
            "\n".join(lines),
            _task_handoff_metadata(
                "task_list_handoff",
                _task_list_handoff_payload(
                    source="list_tasks",
                    status=status,
                    limit=limit,
                    tasks=task_summaries,
                    unreadable_task_rows=unreadable_task_rows,
                ),
                count=len(task_summaries),
                status=status,
                readable_task_rows=len(task_summaries),
                unreadable_task_rows=unreadable_task_rows,
                **_raw_limit_metadata(args.get("limit"), limit=limit),
            ),
        )

    def inspect_task(args: dict[str, Any]) -> ToolResult:
        task_id, error = _parse_task_id(args.get("task_id"))
        if task_id is None:
            failure_output = (
                f"{error or 'task_id must be a number.'} "
                f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "inspect_task",
                False,
                failure_output,
                declare_retryable_local_read_failure(
                    _bad_task_id_metadata(args.get("task_id")),
                    output=failure_output,
                    action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                ),
            )
        task = store.get_task(task_id)
        if not task:
            return _missing_task_result(
                tool_name="inspect_task",
                task_id=task_id,
                mutation="task_read",
                retry_command="show task <correct task id>",
            )
        lines = [
            f"Task #{task['id']}",
            f"- status: {task['status']}",
            f"- priority: {task['priority']}",
            f"- due: {task['due'] or 'none'}",
            f"- source: {task['source']}",
            f"- created: {task['created_at']}",
            f"- updated: {task['updated_at']}",
            f"- completed: {task['completed_at'] or 'not completed'}",
            "",
            task["body"],
        ]
        return ToolResult(
            "inspect_task",
            True,
            "\n".join(lines),
            _task_handoff_metadata(
                "task_inspection_handoff",
                _task_inspection_handoff_payload(task),
                task_id=task_id,
                status=task["status"],
                priority=task["priority"],
                due=task["due"],
                source=task["source"],
            ),
        )

    def search_tasks(args: dict[str, Any]) -> ToolResult:
        raw_query = args.get("query")
        query = _short(raw_query, limit=MAX_TASK_QUERY_CHARS)
        if not query:
            failure_output = (
                "Search query is required. "
                f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "search_tasks",
                False,
                failure_output,
                declare_retryable_local_read_failure(
                    _safe_metadata(
                        reason="missing_query",
                        raw_query=_short_metadata(raw_query, limit=80),
                    ),
                    output=failure_output,
                    action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                ),
            )
        raw_status = args.get("status") or "all"
        status = str(raw_status).strip().lower()
        if status == "all":
            status_filter = None
        elif status in {"open", "done", "paused", "dropped"}:
            status_filter = status
        else:
            failure_output = (
                "Task status must be open, done, paused, dropped, or all. "
                f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "search_tasks",
                False,
                failure_output,
                declare_retryable_local_read_failure(
                    _safe_metadata(
                        query=query,
                        status=status,
                        raw_status=_short_metadata(raw_status, limit=80),
                    ),
                    output=failure_output,
                    action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                ),
            )
        limit = _bounded_int(args.get("limit"), 25)
        rows = store.list_tasks(status=status_filter, limit=1000)
        task_summaries, unreadable_task_rows = _safe_task_summary_payloads(rows)
        terms = [term.lower() for term in re.findall(r"\w+", query)]
        matches = [
            row
            for row in task_summaries
            if all(
                term
                in " ".join([row["body"], row["source"], row["due"], row["priority"], row["status"]]).lower()
                for term in terms
            )
        ][:limit]
        if not matches:
            suffix = (
                f" {unreadable_task_rows} unreadable task row(s) hidden for safety."
                if unreadable_task_rows
                else ""
            )
            return ToolResult(
                "search_tasks",
                True,
                f"No tasks found for '{query}'.{suffix}",
                _task_handoff_metadata(
                    "task_search_handoff",
                    _task_search_handoff_payload(
                        query=query,
                        status=status,
                        limit=limit,
                        matches=[],
                        unreadable_task_rows=unreadable_task_rows,
                    ),
                    query=query,
                    status=status,
                    count=0,
                    readable_task_rows=0,
                    unreadable_task_rows=unreadable_task_rows,
                    **_raw_limit_metadata(args.get("limit"), limit=limit),
                ),
            )
        lines = [f"Task search: {query}"]
        lines.extend(_format_task_summary(row) for row in matches)
        if unreadable_task_rows:
            lines.append(f"- {unreadable_task_rows} unreadable task row(s) hidden for safety.")
        return ToolResult(
            "search_tasks",
            True,
            "\n".join(lines),
            _task_handoff_metadata(
                "task_search_handoff",
                _task_search_handoff_payload(
                    query=query,
                    status=status,
                    limit=limit,
                    matches=matches,
                    unreadable_task_rows=unreadable_task_rows,
                ),
                query=query,
                status=status,
                count=len(matches),
                readable_task_rows=len(matches),
                unreadable_task_rows=unreadable_task_rows,
                **_raw_limit_metadata(args.get("limit"), limit=limit),
            ),
        )

    def overdue_tasks(args: dict[str, Any]) -> ToolResult:
        today = datetime.now().astimezone().date()
        limit = _bounded_int(args.get("limit"), 25)
        rows = store.list_tasks(status="open", limit=1000)
        parsed_count = 0
        overdue: list[dict[str, Any]] = []
        unparsed_due: list[dict[str, Any]] = []
        for row in rows:
            due_label = str(row["due"] or "").strip()
            if not due_label:
                continue
            parsed_due = _parse_due_date_label(due_label, today=today)
            if parsed_due is None:
                unparsed_due.append(row)
                continue
            parsed_count += 1
            if parsed_due < today:
                overdue.append(row)

        limited_overdue = overdue[:limit]
        limited_unparsed = unparsed_due[:limit]
        lines = [f"Overdue task report as of {today.isoformat()}:"]
        if limited_overdue:
            lines.append("Overdue open tasks:")
            for row in limited_overdue:
                priority = f" | {row['priority']}" if row["priority"] != "normal" else ""
                lines.append(f"- #{row['id']} due {row['due']}: {row['body']}{priority}")
        else:
            lines.append("No open tasks with parseable due dates before today.")
        if limited_unparsed:
            lines.extend(["", "Open tasks with due labels I did not parse as dates:"])
            for row in limited_unparsed:
                priority = f" | {row['priority']}" if row["priority"] != "normal" else ""
                lines.append(f"- #{row['id']} due {row['due']}: {row['body']}{priority}")
        return ToolResult(
            "overdue_tasks",
            True,
            "\n".join(lines),
            _task_handoff_metadata(
                "task_overdue_handoff",
                _task_overdue_handoff_payload(
                    as_of=today,
                    limit=limit,
                    overdue=limited_overdue,
                    unparsed_due=limited_unparsed,
                    parsed_count=parsed_count,
                ),
                count=len(limited_overdue),
                overdue_count=len(limited_overdue),
                unparsed_due_count=len(limited_unparsed),
                parsed_due_count=parsed_count,
                as_of=today.isoformat(),
                **_raw_limit_metadata(args.get("limit"), limit=limit),
            ),
        )

    def task_overview(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 8)
        rows = store.list_tasks(status=None, limit=1000)
        status_counts = {"open": 0, "paused": 0, "done": 0, "dropped": 0}
        priority_counts = {"high": 0, "normal": 0, "low": 0}
        for row in rows:
            status = str(row["status"]).lower()
            priority = str(row["priority"]).lower()
            status_counts[status] = status_counts.get(status, 0) + 1
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        open_rows = [row for row in rows if str(row["status"]).lower() == "open"]
        high_open = [row for row in open_rows if str(row["priority"]).lower() == "high"][:limit]
        due_open = [row for row in open_rows if str(row["due"]).strip()][:limit]
        recent_rows = rows[:limit]
        lines = [
            "Task overview:",
            f"- open: {status_counts.get('open', 0)}",
            f"- paused: {status_counts.get('paused', 0)}",
            f"- done: {status_counts.get('done', 0)}",
            f"- dropped: {status_counts.get('dropped', 0)}",
            f"- high priority: {priority_counts.get('high', 0)}",
        ]
        if high_open:
            lines.extend(["", "High-priority open tasks:", *[f"- #{row['id']} {row['body']}" for row in high_open]])
        if due_open:
            lines.extend(["", "Open tasks with due labels:", *[f"- #{row['id']} due {row['due']}: {row['body']}" for row in due_open]])
        if recent_rows:
            lines.extend(["", "Recent task records:", *[f"- #{row['id']} [{row['status']}] {row['body']}" for row in recent_rows]])
        return ToolResult(
            "task_overview",
            True,
            "\n".join(lines),
            _task_handoff_metadata(
                "task_overview_handoff",
                _task_overview_handoff_payload(
                    limit=limit,
                    rows=rows,
                    status_counts=status_counts,
                    priority_counts=priority_counts,
                    high_open=high_open,
                    due_open=due_open,
                    recent_rows=recent_rows,
                ),
                total=len(rows),
                status_counts=status_counts,
                priority_counts=priority_counts,
                open=status_counts.get("open", 0),
                paused=status_counts.get("paused", 0),
                done=status_counts.get("done", 0),
                dropped=status_counts.get("dropped", 0),
                **_raw_limit_metadata(args.get("limit"), limit=limit),
            ),
        )

    def next_task(args: dict[str, Any]) -> ToolResult:
        rows = store.list_tasks(status="open", limit=50)
        if not rows:
            return ToolResult(
                "next_task",
                True,
                "No open tasks. Jarvis has no tracked task to pick next.",
                _task_handoff_metadata(
                    "next_task_handoff",
                    _next_task_handoff_payload(None, []),
                    task_id=None,
                ),
            )
        task = rows[0]
        reason_bits = []
        if task["priority"] == "high":
            reason_bits.append("high priority")
        if task["due"]:
            reason_bits.append(f"due label: {task['due']}")
        if not reason_bits:
            reason_bits.append("first open task in the current ordering")
        lines = [
            f"Next task: #{task['id']} {task['body']}",
            f"- status: {task['status']}",
            f"- priority: {task['priority']}",
            f"- due: {task['due'] or 'none'}",
            f"- reason: {', '.join(reason_bits)}",
            "",
            f"To act: show task {task['id']}",
            f"To pause: pause task {task['id']}",
            f"To finish: complete task {task['id']}",
        ]
        return ToolResult(
            "next_task",
            True,
            "\n".join(lines),
            _task_handoff_metadata(
                "next_task_handoff",
                _next_task_handoff_payload(task, reason_bits),
                task_id=task["id"],
                status=task["status"],
                priority=task["priority"],
                due=task["due"],
                reason=reason_bits,
            ),
        )

    def task_board(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 6)
        statuses = ["open", "paused", "done", "dropped"]
        board: dict[str, list[dict[str, Any]]] = {}
        lines = ["Task board:"]
        for status in statuses:
            rows = store.list_tasks(status=status, limit=limit)
            board[status] = rows
            lines.extend(["", f"{status.title()} ({len(rows)} shown):"])
            if not rows:
                lines.append("- none")
                continue
            for row in rows:
                due = f" | due {row['due']}" if row["due"] else ""
                priority = f" | {row['priority']}" if row["priority"] != "normal" else ""
                lines.append(f"- #{row['id']} {row['body']}{due}{priority}")
        return ToolResult(
            "task_board",
            True,
            "\n".join(lines),
            _task_handoff_metadata(
                "task_board_handoff",
                _task_board_handoff_payload(limit=limit, board=board),
                board={
                    status: [
                        {
                            "id": row["id"],
                            "body": row["body"],
                            "priority": row["priority"],
                            "due": row["due"],
                        }
                        for row in rows
                    ]
                    for status, rows in board.items()
                },
                **_raw_limit_metadata(args.get("limit"), limit=limit),
            ),
        )

    def complete_task(args: dict[str, Any]) -> ToolResult:
        task_id, error = _parse_task_id(args.get("task_id"))
        if task_id is None:
            return _task_known_no_change_failure(
                "complete_task",
                error or "task_id must be a number.",
                _task_refusal_metadata(
                    source="complete_task",
                    reason="bad_task_id",
                    mutation="status_update",
                    raw_field="task_id",
                    raw_value=args.get("task_id"),
                    next_command="complete task <task id>",
                ),
            )
        return _set_task_status(task_id, "done", "Completed")

    def task_completion_packet(args: dict[str, Any]) -> ToolResult:
        task_id, error = _parse_task_id(args.get("task_id"))
        if task_id is None:
            return _task_known_no_change_failure(
                "task_completion_packet",
                error or "task_id must be a number.",
                _bad_task_id_metadata(args.get("task_id")),
            )
        task = store.get_task(task_id)
        if not task:
            return _missing_task_result(
                tool_name="task_completion_packet",
                task_id=task_id,
                mutation="completion_read",
                retry_command="task completion packet <correct task id>",
            )
        evidence, verification_run_id = _completion_evidence_values(args)
        recent_runs = store.recent_tool_runs(limit=30)
        recent_audit_counts = _recent_audit_context_counts(recent_runs)
        recovery_runs, _problem_run_id, _proof_fence = store.task_completion_recovery_snapshot(
            RECOVERY_CLOSURE_META_TOOLS
        )
        recovery_closure = _execution_health_recovery_closure_snapshot(recovery_runs)
        task_done = str(task["status"]).lower() == "done"
        has_evidence = bool(evidence or verification_run_id)
        if has_evidence and not task_done and not recovery_closure["blocks_task_completion"]:
            verdict = "READY_TO_COMPLETE"
        elif has_evidence and not task_done:
            verdict = "RECOVERY_CLOSURE_REQUIRED"
        else:
            verdict = "EVIDENCE_REQUIRED"
        if task_done:
            verdict = "ALREADY_DONE"
        task_completion_handoff = _task_completion_handoff_payload(
            task=task,
            has_evidence=has_evidence,
            evidence=evidence,
            verification_run_id=verification_run_id,
            recent_tool_runs_count=len(recent_runs),
            recent_audit_counts=recent_audit_counts,
            recovery_closure=recovery_closure,
            verdict=verdict,
        )

        lines = [
            "Jarvis task completion packet:",
            "This is read-only. It checks whether a tracked task has enough evidence before Jarvis marks it done.",
            "",
            f"Task #{task['id']}: {_task_body_display(task['body'])}",
            f"- status: {task['status']}",
            f"- priority: {task['priority']}",
            f"- due: {task['due'] or 'none'}",
            "",
            "Completion evidence:",
        ]
        if evidence:
            lines.append(f"- supplied evidence: recorded ({len(evidence)} characters)")
        if verification_run_id:
            lines.append(
                f"- verification run id: recorded ({len(verification_run_id)} characters)"
            )
        if not has_evidence:
            lines.append("- missing; provide evidence text or a verification run id")

        lines.extend(["", "Recent audit context:"])
        if recent_runs:
            for row in recent_runs[:5]:
                status = _task_recent_run_status(row)
                approved = ", approved" if row["approved"] else ""
                lines.append(f"- #{row['id']} {row['tool_name']} [{row['risk']}, {status}{approved}]")
                if status == "approval held":
                    commands = _approval_review_commands_for_run(row)
                    lines.append(f"  approval review: {', '.join(f'`{command}`' for command in commands)}")
        else:
            lines.append("- no recent tool runs")

        lines.extend(
            [
                "",
                "Execution health recovery closure:",
                f"- state: {recovery_closure['state']}",
                f"- ready to retry: {'yes' if recovery_closure['ready_to_retry'] else 'no'}",
                f"- target run: #{recovery_closure['target_run_id']} `{recovery_closure['target_tool_name']}`" if recovery_closure["target_run_id"] is not None else "- target run: none",
                f"- missing: {', '.join(recovery_closure['missing']) if recovery_closure['missing'] else 'none'}",
                f"- checklist overview: `{_recovery_closure_checklist_command(recovery_closure)}`" if _recovery_closure_checklist_command(recovery_closure) else "- checklist overview: none",
                f"- next required: `{recovery_closure['next_required_command']}`" if recovery_closure["next_required_command"] else "- next required: none",
                f"- command queue: {', '.join(f'`{command}`' for command in recovery_closure['required_commands']) if recovery_closure['required_commands'] else 'none'}",
            ]
        )

        lines.extend(
            [
                "",
                f"Verdict: {verdict}",
                "",
                "To finish with evidence:",
                f"- `complete task {task_id} with evidence: <what proves this task is done>`",
                f"- or `complete task {task_id} with verification run <id>: <expected outcome>`",
                "",
                "Boundary:",
                "- This packet does not complete tasks, write notes, approve requests, execute tools, control the computer, or queue approvals.",
            ]
        )
        return ToolResult(
            "task_completion_packet",
            True,
            "\n".join(lines),
            _task_handoff_metadata(
                "task_completion_handoff",
                task_completion_handoff,
                task_id=task_id,
                status=task["status"],
                has_evidence=has_evidence,
                verification_run_id_chars=len(verification_run_id),
                recent_tool_runs=len(recent_runs),
                recent_ok_tool_runs=recent_audit_counts["ok"],
                recent_failed_tool_runs=recent_audit_counts["failed"],
                recent_approval_held_tool_runs=recent_audit_counts["approval_held"],
                recent_unreadable_tool_run_rows=recent_audit_counts["unreadable"],
                recovery_closure_state=recovery_closure["state"],
                recovery_closure_ready_to_retry=recovery_closure["ready_to_retry"],
                recovery_closure_missing=recovery_closure["missing"],
                recovery_closure_missing_count=recovery_closure["missing_count"],
                recovery_closure_required_commands=recovery_closure["required_commands"],
                recovery_closure_next_required_command=recovery_closure["next_required_command"],
                recovery_closure_checklist_command=_recovery_closure_checklist_command(recovery_closure),
                recovery_closure_should_open_checklist=bool(_recovery_closure_checklist_command(recovery_closure)),
                recovery_closure_proof_queue=recovery_closure["required_commands"],
                recovery_closure_proof_queue_count=len(recovery_closure["required_commands"]),
                recovery_closure_next_proof_command=recovery_closure["next_required_command"],
                recovery_closure_blocks_task_completion=recovery_closure["blocks_task_completion"],
                recovery_closure_target_run_id=recovery_closure["target_run_id"],
                recovery_closure_target_tool_name=recovery_closure["target_tool_name"],
                verdict=verdict,
            ),
        )

    def complete_task_with_evidence(args: dict[str, Any]) -> ToolResult:
        task_id, error = _parse_task_id(args.get("task_id"))
        if task_id is None:
            return _task_known_no_change_failure(
                "complete_task_with_evidence",
                error or "task_id must be a number.",
                _task_refusal_metadata(
                    source="complete_task_with_evidence",
                    reason="bad_task_id",
                    mutation="evidence_completion",
                    raw_field="task_id",
                    raw_value=args.get("task_id"),
                    next_command="complete task <task id> with evidence: <proof>",
                ),
            )
        if not store.get_task(task_id):
            return _missing_task_result(
                tool_name="complete_task_with_evidence",
                task_id=task_id,
                mutation="evidence_completion",
                retry_command="complete task <correct task id> with evidence: <proof>",
                has_evidence=bool(args.get("evidence") or args.get("proof") or args.get("receipt") or args.get("verification_run_id") or args.get("run_id")),
            )
        evidence, verification_run_id = _completion_evidence_values(args)
        refusal_reason = _completion_evidence_refusal_reason(
            evidence,
            verification_run_id,
        )
        if refusal_reason is not None:
            return _task_known_no_change_failure(
                "complete_task_with_evidence",
                (
                    f"Completion evidence is not valid for closing task #{task_id}. "
                    f"Run `task completion packet {task_id}` first."
                ),
                _task_refusal_metadata(
                    source="complete_task_with_evidence",
                    reason=refusal_reason,
                    mutation="evidence_completion",
                    task_id=task_id,
                    next_command=f"task completion packet {task_id}",
                    has_evidence=refusal_reason != "missing_evidence",
                ),
            )
        (
            recovery_runs,
            recovery_problem_run_id,
            recovery_proof_fence,
        ) = store.task_completion_recovery_snapshot(RECOVERY_CLOSURE_META_TOOLS)
        recovery_closure = _execution_health_recovery_closure_snapshot(recovery_runs)
        if recovery_closure["blocks_task_completion"]:
            lines = [
                f"Task #{task_id} is not ready to close yet.",
                "",
                "Evidence was supplied, but execution health recovery closure is still blocking task completion.",
                f"- recovery closure state: {recovery_closure['state']}",
                f"- missing: {', '.join(recovery_closure['missing']) if recovery_closure['missing'] else 'none'}",
                f"- checklist overview: `{_recovery_closure_checklist_command(recovery_closure)}`",
                f"- next required: `{recovery_closure['next_required_command']}`" if recovery_closure["next_required_command"] else "- next required: none",
                f"- command queue: {', '.join(f'`{command}`' for command in recovery_closure['required_commands']) if recovery_closure['required_commands'] else 'none'}",
                "",
                f"Run `task completion packet {task_id}` after closing the recovery proof queue.",
            ]
            return _task_known_no_change_failure(
                "complete_task_with_evidence",
                "\n".join(lines),
                _safe_metadata(
                    task_id=task_id,
                    has_evidence=True,
                    recovery_closure_state=recovery_closure["state"],
                    recovery_closure_ready_to_retry=recovery_closure["ready_to_retry"],
                    recovery_closure_missing=recovery_closure["missing"],
                    recovery_closure_missing_count=recovery_closure["missing_count"],
                    recovery_closure_required_commands=recovery_closure["required_commands"],
                    recovery_closure_next_required_command=recovery_closure["next_required_command"],
                    recovery_closure_checklist_command=_recovery_closure_checklist_command(recovery_closure),
                    recovery_closure_should_open_checklist=bool(_recovery_closure_checklist_command(recovery_closure)),
                    recovery_closure_proof_queue=recovery_closure["required_commands"],
                    recovery_closure_proof_queue_count=len(recovery_closure["required_commands"]),
                    recovery_closure_next_proof_command=recovery_closure["next_required_command"],
                    recovery_closure_blocks_task_completion=True,
                    recovery_closure_target_run_id=recovery_closure["target_run_id"],
                    recovery_closure_target_tool_name=recovery_closure["target_tool_name"],
                ),
                action=TASK_STATE_RECOVERY_ACTION,
            )
        try:
            task = store.complete_task_with_evidence(
                task_id,
                evidence=evidence,
                verification_run_id=verification_run_id,
                expected_recovery_problem_run_id=recovery_problem_run_id,
                expected_recovery_proof_fence=recovery_proof_fence,
                recovery_meta_tool_names=RECOVERY_CLOSURE_META_TOOLS,
            )
        except RuntimeError as exc:
            if str(exc) != "task completion recovery state changed":
                raise
            return _task_known_no_change_failure(
                "complete_task_with_evidence",
                (
                    "Execution health changed while Jarvis was closing this task. "
                    "No task state or evidence changed; review the completion packet and retry."
                ),
                _task_refusal_metadata(
                    source="complete_task_with_evidence",
                    reason="recovery_state_changed",
                    mutation="evidence_completion",
                    task_id=task_id,
                    next_command=f"task completion packet {task_id}",
                    has_evidence=True,
                    auto_mutation_effects_started=False,
                ),
                action=TASK_STATE_RECOVERY_ACTION,
            )
        if not task:
            return _missing_task_result(
                tool_name="complete_task_with_evidence",
                task_id=task_id,
                mutation="evidence_completion",
                retry_command="complete task <correct task id> with evidence: <proof>",
                has_evidence=True,
            )
        path = vault.sync_open_tasks(store)
        lines = [
            f"Completed task #{task_id} with evidence: {_task_body_display(task['body'])}",
            "",
            "Evidence receipt:",
        ]
        if evidence:
            lines.append(f"- evidence: recorded ({len(evidence)} characters)")
        if verification_run_id:
            lines.append(
                f"- verification run id: recorded ({len(verification_run_id)} characters)"
            )
        lines.extend(
            [
                "",
                "Audit note:",
                "- The completion evidence is preserved in the immutable task evidence ledger.",
                "- Use `recent tool runs` or `verification receipt <run id>` to inspect proof before relying on this completion.",
            ]
        )
        return ToolResult(
            "complete_task_with_evidence",
            True,
            "\n".join(lines),
            _task_handoff_metadata(
                "task_mutation_handoff",
                _task_mutation_handoff_payload(
                    source="complete_task_with_evidence",
                    task=task,
                    mutation="evidence_completion",
                    path="Tasks/Open Tasks.md",
                    changed=["status"],
                    evidence=evidence,
                    verification_run_id=verification_run_id,
                    recovery_closure=recovery_closure,
                ),
                writes=True,
                task_id=task_id,
                status="done",
                path_display="Tasks/Open Tasks.md",
                has_evidence=True,
                evidence_chars=len(evidence),
                verification_run_id_chars=len(verification_run_id),
                recovery_closure_state=recovery_closure["state"],
                recovery_closure_ready_to_retry=recovery_closure["ready_to_retry"],
                recovery_closure_missing=recovery_closure["missing"],
                recovery_closure_missing_count=recovery_closure["missing_count"],
                recovery_closure_next_required_command=recovery_closure["next_required_command"],
                recovery_closure_checklist_command=_recovery_closure_checklist_command(recovery_closure),
                recovery_closure_should_open_checklist=bool(_recovery_closure_checklist_command(recovery_closure)),
                recovery_closure_blocks_task_completion=False,
            ),
        )

    def update_task_status(args: dict[str, Any]) -> ToolResult:
        task_id, error = _parse_task_id(args.get("task_id"))
        if task_id is None:
            return _task_known_no_change_failure(
                "update_task_status",
                error or "task_id must be a number.",
                _task_refusal_metadata(
                    source="update_task_status",
                    reason="bad_task_id",
                    mutation="status_update",
                    raw_field="task_id",
                    raw_value=args.get("task_id"),
                    next_command="task <task id> <open|done|paused|dropped>",
                ),
            )
        raw_status = args.get("status")
        status = str(raw_status or "").strip().lower()
        if status not in TASK_STATUS_LABELS:
            return _task_known_no_change_failure(
                "update_task_status",
                "Task status must be open, done, paused, or dropped.",
                _task_refusal_metadata(
                    source="update_task_status",
                    reason="bad_status",
                    mutation="status_update",
                    task_id=task_id,
                    raw_field="status",
                    raw_value=raw_status,
                    next_command=f"task {task_id} <open|done|paused|dropped>",
                    status=status,
                ),
            )
        return _set_task_status(
            task_id,
            status,
            TASK_STATUS_LABELS[status],
            tool_name="update_task_status",
        )

    def update_task_details(args: dict[str, Any]) -> ToolResult:
        task_id, error = _parse_task_id(args.get("task_id"))
        if task_id is None:
            return _task_known_no_change_failure(
                "update_task_details",
                error or "task_id must be a number.",
                _task_refusal_metadata(
                    source="update_task_details",
                    reason="bad_task_id",
                    mutation="field_update",
                    raw_field="task_id",
                    raw_value=args.get("task_id"),
                    next_command=_task_detail_retry_command(
                        args,
                        "<task id>",
                        reason="bad_task_id",
                    ),
                ),
            )
        body, due, priority = _normalized_task_detail_updates(args)
        if body is not None and not body:
            return _task_known_no_change_failure(
                "update_task_details",
                "Task body cannot be empty.",
                _task_refusal_metadata(
                    source="update_task_details",
                    reason="missing_body",
                    mutation="field_update",
                    task_id=task_id,
                    raw_field="body",
                    raw_value=args.get("body"),
                    next_command=_task_detail_retry_command(
                        args,
                        task_id,
                        reason="missing_body",
                    ),
                ),
            )
        if priority is not None and priority not in {"low", "normal", "high"}:
            return _task_known_no_change_failure(
                "update_task_details",
                "Priority must be low, normal, or high.",
                _task_refusal_metadata(
                    source="update_task_details",
                    reason="bad_priority",
                    mutation="field_update",
                    task_id=task_id,
                    raw_field="priority",
                    raw_value=args.get("priority"),
                    next_command=_task_detail_retry_command(
                        args,
                        task_id,
                        reason="bad_priority",
                    ),
                    priority=priority,
                ),
            )
        if not _task_detail_text_is_utf8_safe(body, due):
            return _task_known_no_change_failure(
                "update_task_details",
                "Task body and due label must contain valid Unicode text.",
                _task_refusal_metadata(
                    source="update_task_details",
                    reason="invalid_unicode",
                    mutation="field_update",
                    task_id=task_id,
                    next_command=_task_detail_retry_command(
                        args,
                        task_id,
                        reason="invalid_unicode",
                    ),
                ),
            )
        if body is None and due is None and priority is None:
            return _task_known_no_change_failure(
                "update_task_details",
                "Give Jarvis a task body, due label, or priority to update.",
                _task_refusal_metadata(
                    source="update_task_details",
                    reason="missing_update",
                    mutation="field_update",
                    task_id=task_id,
                    next_command=_task_detail_retry_command(
                        args,
                        task_id,
                        reason="missing_update",
                    ),
                ),
            )
        task = store.update_task_fields(task_id, body=body, due=due, priority=priority)
        if not task:
            return _missing_task_result(
                tool_name="update_task_details",
                task_id=task_id,
                mutation="field_update",
                retry_command=_task_detail_retry_command(
                    args,
                    "<correct task id>",
                    reason="missing_task",
                ),
                auto_mutation_effects_started=False,
                writes_database=False,
            )
        path = vault.sync_open_tasks(store)
        path_display = _safe_vault_path_display(path, vault)
        changed = []
        if body is not None:
            changed.append("body")
        if due is not None:
            changed.append("due")
        if priority is not None:
            changed.append("priority")
        return ToolResult(
            "update_task_details",
            True,
            f"Updated task #{task_id} {', '.join(changed)}: {_task_body_display(task['body'])}",
            _task_handoff_metadata(
                "task_mutation_handoff",
                _task_mutation_handoff_payload(
                    source="update_task_details",
                    task=task,
                    mutation="field_update",
                    path=path_display,
                    changed=changed,
                ),
                writes=True,
                task_id=task_id,
                changed=changed,
                status=task["status"],
                priority=task["priority"],
                due=task["due"],
                path_display=path_display,
            ),
        )

    def _set_task_status(task_id: int, status: str, label: str, tool_name: str = "complete_task") -> ToolResult:
        task = store.set_task_status(task_id, status)
        if not task:
            retry_commands = {
                "open": "reopen task <correct task id>",
                "done": "complete task <correct task id>",
                "paused": "pause task <correct task id>",
                "dropped": "drop task <correct task id>",
            }
            return _missing_task_result(
                tool_name=tool_name,
                task_id=task_id,
                mutation="status_update",
                retry_command=retry_commands.get(status, "task <correct task id> <status>"),
                status=status,
                auto_mutation_effects_started=False,
                writes_database=False,
            )
        path = vault.sync_open_tasks(store)
        return ToolResult(
            tool_name,
            True,
            f"{label} task #{task_id}: {task['body']}",
            _task_handoff_metadata(
                "task_mutation_handoff",
                _task_mutation_handoff_payload(
                    source=tool_name,
                    task=task,
                    mutation="status_update",
                    path=path,
                    changed=["status"],
                ),
                writes=True,
                task_id=task_id,
                status=status,
                path=str(path),
            ),
        )

    def export_tasks(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 100)
        path, canonical_tasks, content_sha256, source_revision = vault.sync_open_tasks_with_evidence(store)
        tasks = canonical_tasks[:limit]
        path_display = _safe_vault_path_display(path, vault)
        return ToolResult(
            "export_tasks",
            True,
            f"Exported open tasks to {path_display}",
            _task_handoff_metadata(
                "task_export_handoff",
                _task_export_handoff_payload(
                    path_display=path_display,
                    limit=limit,
                    exported_tasks=tasks,
                ),
                writes=False,
                path=str(path),
                path_display=path_display,
                exported_count=len(tasks),
                mirror_count=len(canonical_tasks),
                content_sha256=content_sha256,
                source_revision=source_revision,
                writes_files=True,
                writes_memory=False,
                writes_notes=True,
                **_raw_limit_metadata(args.get("limit"), limit=limit),
            ),
        )

    def preview_tasks_from_note(args: dict[str, Any]) -> ToolResult:
        raw_path = _raw_task_import_path(args)
        if not raw_path:
            return _note_task_preview_failure(
                "Note path is required.",
                reason="missing_path",
            )
        if len(raw_path) > MAX_NOTE_PATH_CHARS:
            return _note_task_preview_failure(
                f"Note path is too long; limit is {MAX_NOTE_PATH_CHARS} characters.",
                reason="path_too_large",
                raw_path=raw_path,
                path_chars=len(raw_path),
                max_path_chars=MAX_NOTE_PATH_CHARS,
            )
        try:
            note_path = _lexical_note_path(vault.root_path, raw_path)
        except ValueError as exc:
            return _note_task_preview_failure(
                str(exc),
                reason="unsafe_path",
                raw_path=raw_path,
            )
        if is_protected_profile_path_or_content(
            note_path,
            vault_root=vault.root_path,
        ):
            return _protected_profile_task_result("preview_tasks_from_note")
        note_path_display = _safe_vault_path_display(note_path, vault)
        try:
            text = vault.read_note_bounded(
                note_path,
                max_chars=MAX_TASK_IMPORT_NOTE_CHARS,
            )
        except OverflowError:
            return _note_task_preview_failure(
                "Jarvis note is too large to preview for task import; "
                f"limit is {MAX_TASK_IMPORT_NOTE_CHARS} characters.",
                reason="note_too_large",
                raw_path=raw_path,
                max_chars=MAX_TASK_IMPORT_NOTE_CHARS,
            )
        except IsADirectoryError:
            return _note_task_preview_failure(
                "Jarvis note must be a regular Markdown file.",
                reason="note_not_regular",
                raw_path=raw_path,
            )
        except ValueError as exc:
            return _note_task_preview_failure(
                str(exc),
                reason="unsafe_path",
                raw_path=raw_path,
            )
        except OSError as exc:
            return _note_task_preview_failure(
                "Could not read Jarvis note for task preview. Check JARVIS_OBSIDIAN_VAULT "
                "and vault permissions.",
                reason="note_read_failed",
                raw_path=raw_path,
                exception_type=type(exc).__name__,
            )
        if text is None:
            return _note_task_not_found_result(
                tool_name="preview_tasks_from_note",
                raw_path=raw_path,
            )
        if is_protected_profile_path_or_content(
            note_path,
            text,
            vault_root=vault.root_path,
        ):
            return _protected_profile_task_result("preview_tasks_from_note")
        open_tasks, completed_tasks = _markdown_tasks(text)
        existing = store.normalized_task_identities()
        new_tasks = [body for body in open_tasks if normalized_task_identity(body) not in existing]
        duplicates = [body for body in open_tasks if normalized_task_identity(body) in existing]
        lines = [
            f"Jarvis note task preview: {note_path_display}",
            f"- open checkboxes: {len(open_tasks)}",
            f"- new importable tasks: {len(new_tasks)}",
            f"- duplicates already tracked: {len(duplicates)}",
            f"- completed checkboxes ignored: {len(completed_tasks)}",
        ]
        if new_tasks:
            lines.extend(["", "New tasks:", *[f"- {body}" for body in new_tasks[:12]]])
        if duplicates:
            lines.extend(["", "Already tracked:", *[f"- {body}" for body in duplicates[:12]]])
        if completed_tasks:
            lines.extend(["", "Completed in note:", *[f"- {body}" for body in completed_tasks[:12]]])
        return ToolResult(
            "preview_tasks_from_note",
            True,
            "\n".join(lines),
            _task_handoff_metadata(
                "note_task_preview_handoff",
                _note_task_preview_handoff_payload(
                    note_path=note_path,
                    vault=vault,
                    open_tasks=open_tasks,
                    completed_tasks=completed_tasks,
                    new_tasks=new_tasks,
                    duplicates=duplicates,
                ),
                path=str(note_path),
                path_display=_safe_vault_path_display(note_path, vault),
                open_checkboxes=len(open_tasks),
                new_tasks=len(new_tasks),
                duplicates=len(duplicates),
                completed_checkboxes=len(completed_tasks),
            ),
        )

    def import_tasks_from_note(args: dict[str, Any]) -> ToolResult:
        raw_path_value = _raw_task_import_path(args)
        raw_path = raw_path_value
        priority = _task_import_priority(args)
        if not raw_path:
            return _task_import_prewrite_failure(
                "Note path is required.",
                _task_import_prewrite_metadata(reason="missing_path", priority=priority),
            )
        if len(raw_path_value) > MAX_NOTE_PATH_CHARS:
            return _task_import_prewrite_failure(
                f"Note path is too long; limit is {MAX_NOTE_PATH_CHARS} characters.",
                _task_import_prewrite_metadata(
                    reason="path_too_large",
                    raw_path=_note_path_metadata(raw_path),
                    priority=priority,
                    path_chars=len(raw_path_value),
                    max_path_chars=MAX_NOTE_PATH_CHARS,
                ),
            )
        if priority not in {"low", "normal", "high"}:
            return _task_import_prewrite_failure(
                "Priority must be low, normal, or high.",
                _task_import_prewrite_metadata(
                    reason="bad_priority",
                    raw_path=_note_path_metadata(raw_path),
                    priority=priority,
                ),
            )
        try:
            note_path = _lexical_note_path(vault.root_path, raw_path)
        except ValueError as exc:
            return _task_import_prewrite_failure(
                str(exc),
                _task_import_prewrite_metadata(
                    reason="unsafe_path",
                    raw_path=_note_path_metadata(raw_path),
                    priority=priority,
                ),
            )
        if is_protected_profile_path_or_content(
            note_path,
            vault_root=vault.root_path,
        ):
            return _protected_profile_task_result(
                "import_tasks_from_note",
                priority=priority,
            )
        note_path_display = _safe_vault_path_display(note_path, vault)
        try:
            text = vault.read_note_bounded(
                note_path,
                max_chars=MAX_TASK_IMPORT_NOTE_CHARS,
            )
        except OverflowError:
            return _task_import_prewrite_failure(
                "Jarvis note is too large to import safely; "
                f"limit is {MAX_TASK_IMPORT_NOTE_CHARS} characters.",
                _task_import_prewrite_metadata(
                    reason="note_too_large",
                    raw_path=_note_path_metadata(raw_path),
                    priority=priority,
                    max_chars=MAX_TASK_IMPORT_NOTE_CHARS,
                ),
            )
        except IsADirectoryError:
            return _task_import_prewrite_failure(
                "Jarvis note must be a regular Markdown file.",
                _task_import_prewrite_metadata(
                    reason="note_not_regular",
                    raw_path=_note_path_metadata(raw_path),
                    priority=priority,
                ),
            )
        except ValueError as exc:
            return _task_import_prewrite_failure(
                str(exc),
                _task_import_prewrite_metadata(
                    reason="unsafe_path",
                    raw_path=_note_path_metadata(raw_path),
                    priority=priority,
                ),
            )
        except OSError as exc:
            output = (
                "Could not read Jarvis note for task import. Check JARVIS_OBSIDIAN_VAULT and vault "
                f"permissions. {LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION}"
            )
            return _task_import_prewrite_failure(
                output,
                _task_import_prewrite_metadata(
                    reason="note_read_failed",
                    raw_path=_note_path_metadata(raw_path),
                    priority=priority,
                    exception_type=type(exc).__name__,
                ),
                action=LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
                commands=("setup check",),
            )
        if text is None:
            return _note_task_not_found_result(
                tool_name="import_tasks_from_note",
                raw_path=raw_path,
                metadata=_task_import_prewrite_metadata(
                    reason="note_not_found",
                    raw_path=_note_path_metadata(raw_path),
                    priority=priority,
                ),
            )
        if is_protected_profile_path_or_content(
            note_path,
            text,
            vault_root=vault.root_path,
        ):
            return _protected_profile_task_result(
                "import_tasks_from_note",
                priority=priority,
            )
        task_bodies = _open_markdown_tasks(text)
        if len(task_bodies) > MAX_TASK_IMPORT_CANDIDATES:
            return _task_import_prewrite_failure(
                "Jarvis note has too many open task candidates to import safely; "
                f"limit is {MAX_TASK_IMPORT_CANDIDATES}. Nothing was written.",
                _task_import_prewrite_metadata(
                    reason="too_many_tasks",
                    raw_path=_note_path_metadata(raw_path),
                    priority=priority,
                    **_task_import_candidate_limit_metadata(len(task_bodies)),
                ),
            )
        if not task_bodies:
            return ToolResult(
                "import_tasks_from_note",
                True,
                f"No open markdown tasks found in {note_path_display}.",
                _task_handoff_metadata(
                    "note_task_import_handoff",
                    _note_task_import_handoff_payload(
                        note_path=note_path,
                        tasks_path=vault.root_path / "Tasks" / "Open Tasks.md",
                        vault=vault,
                        priority=priority,
                        imported=[],
                        skipped=[],
                    ),
                    imported=0,
                    skipped=0,
                    path_display=_safe_vault_path_display(note_path, vault),
                    priority=priority,
                ),
            )

        records = [
            TaskRecord(
                body=body,
                priority=priority,
                source=f"jarvis-note:{note_path.name}",
            )
            for body in task_bodies
        ]
        inserted = store.add_tasks_if_identities_absent(records)
        imported: list[str] = []
        skipped: list[str] = []
        for body, task_id in zip(task_bodies, inserted):
            if task_id is None:
                skipped.append(body)
                continue
            imported.append(body)

        path = vault.sync_open_tasks(store)
        lines = [
            f"Imported {len(imported)} task(s) from {note_path_display}.",
        ]
        if imported:
            lines.extend(["", "Imported:", *[f"- {body}" for body in imported[:12]]])
        if skipped:
            lines.extend(["", "Skipped duplicates:", *[f"- {body}" for body in skipped[:12]]])
        return ToolResult(
            "import_tasks_from_note",
            True,
            "\n".join(lines),
            _task_handoff_metadata(
                "note_task_import_handoff",
                _note_task_import_handoff_payload(
                    note_path=note_path,
                    tasks_path=path,
                    vault=vault,
                    priority=priority,
                    imported=imported,
                    skipped=skipped,
                ),
                writes=True,
                imported=len(imported),
                skipped=len(skipped),
                path_display=_safe_vault_path_display(note_path, vault),
                tasks_path_display=_safe_vault_path_display(path, vault),
                priority=priority,
            ),
        )

    def guide_remaining_failure(handler):
        @wraps(handler)
        def guided(args: dict[str, Any]) -> ToolResult:
            result = handler(args)
            if result.ok or "recovery_guidance" in result.metadata:
                return result
            action = (
                TASK_STATE_RECOVERY_ACTION
                if result.metadata.get("recovery_closure_blocks_task_completion")
                or result.metadata.get("reason") == "recovery_state_changed"
                else TASK_INPUT_RECOVERY_ACTION
            )
            return _task_known_no_change_failure(
                result.tool_name,
                result.output,
                result.metadata,
                action=action,
            )

        return guided

    return (
        guide_remaining_failure(add_task),
        guide_remaining_failure(list_tasks),
        guide_remaining_failure(inspect_task),
        guide_remaining_failure(search_tasks),
        guide_remaining_failure(overdue_tasks),
        guide_remaining_failure(task_overview),
        guide_remaining_failure(next_task),
        guide_remaining_failure(task_board),
        guide_remaining_failure(complete_task),
        guide_remaining_failure(task_completion_packet),
        guide_remaining_failure(complete_task_with_evidence),
        guide_remaining_failure(update_task_status),
        guide_remaining_failure(update_task_details),
        guide_remaining_failure(export_tasks),
        preview_tasks_from_note,
        import_tasks_from_note,
    )


def _safe_note_path(root: Path, raw_path: str) -> Path:
    lexical = _lexical_note_path(root, raw_path)
    resolved_root = root.resolve()
    resolved = lexical.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("Note path must stay inside the Jarvis Obsidian folder.") from exc
    return resolved


def _lexical_note_path(root: Path, raw_path: str) -> Path:
    cleaned = raw_path.strip().strip("\"'")
    if not cleaned:
        raise ValueError("Note path is empty.")
    if any(
        unicodedata.category(char) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        for char in cleaned
    ):
        raise ValueError("Note path contains unsupported control characters.")
    candidate = Path(cleaned)
    if candidate.is_absolute():
        raise ValueError("Note path must be relative to the Jarvis Obsidian folder.")
    if ".." in candidate.parts:
        raise ValueError("Note path must stay inside the Jarvis Obsidian folder.")
    if candidate.suffix.lower() != ".md":
        candidate = candidate.with_suffix(".md")
    resolved_root = root.resolve()
    return resolved_root / candidate


def _open_markdown_tasks(text: str) -> list[str]:
    return _markdown_tasks(text)[0]


def _markdown_tasks(text: str) -> tuple[list[str], list[str]]:
    open_tasks: list[str] = []
    completed_tasks: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        match = re.match(r"^\s*[-*]\s+\[(?P<status>[ xX])\]\s+(?P<body>.+?)\s*$", line)
        if not match:
            continue
        body = _short(match.group("body"), limit=MAX_TASK_BODY_CHARS)
        if not body:
            continue
        status = match.group("status").lower()
        key = f"{status}:{normalized_task_identity(body)}"
        if key in seen:
            continue
        seen.add(key)
        if status == " ":
            open_tasks.append(body)
        else:
            completed_tasks.append(body)
    return open_tasks, completed_tasks
