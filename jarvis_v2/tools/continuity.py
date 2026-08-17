from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable

from jarvis_v2.agent.types import ToolResult
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import MemoryStore


MAX_CONTINUITY_LIMIT = 200
CHECKPOINT_REVIEW_MINUTES = 120
CHECKPOINT_STALE_MINUTES = 360
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
RISKY_RECOVERY_SIGNALS = {
    "shell/code": ("run command", "terminal", "shell", "python", "script", "execute", "install", "npm", "pip"),
    "computer control": ("click", "type", "mouse", "keyboard", "screen", "screenshot", "window", "app"),
    "personal data": ("email", "calendar", "contact", "message", "gmail", "inbox", "clipboard", "private"),
    "external side effect": ("send", "post", "publish", "schedule", "remind", "call", "text", "book", "buy"),
    "destructive": ("delete", "remove", "erase", "wipe", "reset", "clear"),
}
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


def _bounded_int(value: Any, default: int, low: int = 1, high: int = MAX_CONTINUITY_LIMIT) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


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
        return value.strip().casefold() in {"1", "true", "yes", "on"}
    return False


def _normalized_metadata_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().casefold()).strip("_")


def _tool_run_metadata(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata", {})
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw or "{}")
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


def _tool_run_status(row: dict[str, Any]) -> str:
    if row.get("ok"):
        return "ok"
    if _is_approval_hold_metadata(_tool_run_metadata(row)):
        return "approval_held"
    return "failed"


def _tool_run_status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"ok": 0, "failed": 0, "approval_held": 0}
    for row in rows:
        counts[_tool_run_status(row)] += 1
    return counts


def _safe_preview(value: Any, *, limit: int = 240) -> str:
    text = " ".join(str(value or "").strip().split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) > limit:
        return text[: max(0, limit - 1)].rstrip() + "…"
    return text


def _safe_vault_path_display(path: Path | str | None, vault: ObsidianVault) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return str(candidate.relative_to(vault.root_path))
    except ValueError:
        return _safe_preview(candidate)


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "calls_model": False,
        "executes_tools": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "reads_note_contents": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "queues_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "requires_approval": False,
        "approves_request": False,
        "dismisses_request": False,
        "controls_computer": False,
        "speaks": False,
        "completes_tasks": False,
        "edits_files": False,
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
    }
    metadata.update(extra)
    return metadata


def _safe_tool_run_payloads(rows: list[Any]) -> tuple[list[dict[str, Any]], int]:
    payloads: list[dict[str, Any]] = []
    unreadable = 0
    for row in rows:
        try:
            keys = set(row.keys())
            payloads.append(
                {
                    "id": int(row["id"]),
                    "tool_name": str(row["tool_name"]),
                    "risk": str(row["risk"]),
                    "ok": bool(row["ok"]),
                    "approved": bool(row["approved"]),
                    "approval_id": row["approval_id"] if "approval_id" in keys else None,
                    "output": str(row["output"] or ""),
                    "metadata": str(row["metadata"] or "{}") if "metadata" in keys else "{}",
                    "created_at": str(row["created_at"] or ""),
                }
            )
        except Exception:
            unreadable += 1
    return payloads, unreadable


def _safe_message_payloads(rows: list[Any]) -> tuple[list[dict[str, Any]], int]:
    payloads: list[dict[str, Any]] = []
    unreadable = 0
    for row in rows:
        try:
            payloads.append(
                {
                    "id": int(row["id"]),
                    "session_id": str(row["session_id"] or ""),
                    "role": str(row["role"] or ""),
                    "content": str(row["content"] or ""),
                    "created_at": str(row["created_at"] or ""),
                }
            )
        except Exception:
            unreadable += 1
    return payloads, unreadable


def _safe_session_payloads(rows: list[Any]) -> tuple[list[dict[str, Any]], int]:
    payloads: list[dict[str, Any]] = []
    unreadable = 0
    for row in rows:
        try:
            keys = set(row.keys())
            payloads.append(
                {
                    "session_id": _safe_preview(row["session_id"], limit=120),
                    "messages": int(row["messages"] or 0),
                    "started_at": str(row["started_at"] if "started_at" in keys else row["first_at"] if "first_at" in keys else ""),
                    "last_at": str(row["last_at"] or ""),
                }
            )
        except Exception:
            unreadable += 1
    return payloads, unreadable


def _recovery_risk_signals(text: str) -> list[str]:
    lowered = text.lower()
    return [label for label, words in RISKY_RECOVERY_SIGNALS.items() if any(word in lowered for word in words)]


def _approval_reference_provided(value: str) -> bool:
    normalized = value.strip().lower()
    return bool(normalized and normalized not in {"not required for local-safe receipt", "not required or not supplied", "none", "n/a"})


def _approval_chain_reference_provided(value: str) -> bool:
    normalized = value.strip().lower()
    if not _approval_reference_provided(normalized):
        return False
    return all(
        required in normalized
        for required in (
            "approval readiness",
            "approval packet",
            "approval chain proof",
        )
    )


def _approval_proof_queue_for_risky_work(*, followup_command: str) -> list[str]:
    return [
        "approval readiness <id>",
        "approval packet <id>",
        "approval chain proof <id>",
        "verification receipt <approved run id from approval chain proof <id>>",
        followup_command,
    ]


def _approval_boundary_rows_for_risky_work(
    *,
    risk_signals: list[str],
    step_sha256: str,
    verification_sha256: str,
    proof_queue: list[str],
    required_field: str,
) -> list[dict[str, Any]]:
    risk_present = bool(risk_signals)
    rows = [
        {
            "item": "risk_classification",
            required_field: True,
            "status": "held" if risk_present else "not_required",
            "ready": not risk_present,
            "risk_signals": risk_signals,
            "authorizes_action_now": False,
            "authorizes_risky_work": False,
            "authorizes_unreviewed_followthrough": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_personal_data_read": False,
            "authorizes_external_side_effect": False,
        },
        {
            "item": "exact_arguments_or_step",
            required_field: risk_present,
            "status": "held" if risk_present else "not_required",
            "ready": not risk_present,
            "step_sha256": step_sha256,
            "verification_sha256": verification_sha256,
            "authorizes_action_now": False,
            "authorizes_risky_work": False,
            "authorizes_unreviewed_followthrough": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_personal_data_read": False,
            "authorizes_external_side_effect": False,
        },
        {
            "item": "approval_readiness_packet",
            required_field: risk_present,
            "status": "held" if risk_present else "not_required",
            "ready": not risk_present,
            "next_proof_command": proof_queue[0] if proof_queue else "",
            "authorizes_action_now": False,
            "authorizes_risky_work": False,
            "authorizes_unreviewed_followthrough": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_personal_data_read": False,
            "authorizes_external_side_effect": False,
        },
        {
            "item": "last_look_approval_packet",
            required_field: risk_present,
            "status": "held" if risk_present else "not_required",
            "ready": not risk_present,
            "next_proof_command": proof_queue[1] if len(proof_queue) > 1 else "",
            "authorizes_action_now": False,
            "authorizes_risky_work": False,
            "authorizes_unreviewed_followthrough": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_personal_data_read": False,
            "authorizes_external_side_effect": False,
        },
        {
            "item": "approval_chain_proof",
            required_field: risk_present,
            "status": "held" if risk_present else "not_required",
            "ready": not risk_present,
            "next_proof_command": proof_queue[2] if len(proof_queue) > 2 else "",
            "authorizes_action_now": False,
            "authorizes_risky_work": False,
            "authorizes_unreviewed_followthrough": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_personal_data_read": False,
            "authorizes_external_side_effect": False,
        },
        {
            "item": "post_approval_verification_receipt",
            required_field: risk_present,
            "status": "held" if risk_present else "not_required",
            "ready": not risk_present,
            "next_proof_command": proof_queue[3] if len(proof_queue) > 3 else "",
            "authorizes_action_now": False,
            "authorizes_risky_work": False,
            "authorizes_unreviewed_followthrough": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_personal_data_read": False,
            "authorizes_external_side_effect": False,
        },
        {
            "item": "fresh_local_safe_review_after_approval",
            required_field: risk_present,
            "status": "held" if risk_present else "not_required",
            "ready": not risk_present,
            "next_proof_command": proof_queue[4] if len(proof_queue) > 4 else "",
            "authorizes_action_now": False,
            "authorizes_risky_work": False,
            "authorizes_unreviewed_followthrough": False,
            "authorizes_approval": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_personal_data_read": False,
            "authorizes_external_side_effect": False,
        },
    ]
    for row in rows:
        row.update(
            {
                "authorizes_timebox_reuse": False,
                "reusable_for_next_review": False,
                "reusable_for_recovery_review": False,
            }
        )
    return rows


def _risky_next_step_approval_boundary_token_sha256(
    *,
    proposed_step_sha256: str,
    proposed_verification_sha256: str,
    risk_signals: list[str],
    approval_proof_queue: list[str],
    approval_boundary_rows: list[dict[str, Any]],
    approval_reference: str = "",
    authorizes_action_now: bool = False,
    authorizes_risky_work: bool = False,
    authorizes_unreviewed_followthrough: bool = False,
    reusable_for_next_review: bool = False,
) -> str:
    row_fingerprint = [
        "|".join(
            [
                str(row.get("item") or ""),
                str(row.get("status") or ""),
                str(_metadata_bool(row.get("ready"))),
                str(_metadata_bool(row.get("required_before_risky_next_step"))),
                str(row.get("next_proof_command") or ""),
                str(_metadata_bool(row.get("authorizes_action_now"))),
                str(_metadata_bool(row.get("authorizes_risky_work"))),
                str(_metadata_bool(row.get("authorizes_unreviewed_followthrough"))),
                str(_metadata_bool(row.get("authorizes_approval"))),
                str(_metadata_bool(row.get("authorizes_model_call"))),
                str(_metadata_bool(row.get("authorizes_tool_execution"))),
                str(_metadata_bool(row.get("authorizes_personal_data_read"))),
                str(_metadata_bool(row.get("authorizes_external_side_effect"))),
                str(_metadata_bool(row.get("authorizes_timebox_reuse"))),
                str(_metadata_bool(row.get("reusable_for_next_review"))),
                str(_metadata_bool(row.get("reusable_for_recovery_review"))),
            ]
        )
        for row in approval_boundary_rows
    ]
    return _text_sha256(
        "\n".join(
            [
                "risky_next_step_approval_boundary_v1",
                proposed_step_sha256,
                proposed_verification_sha256,
                ",".join(sorted(str(signal) for signal in risk_signals)),
                "\n".join(approval_proof_queue),
                "\n".join(row_fingerprint),
                f"approval_reference={approval_reference}",
                f"authorizes_action_now={authorizes_action_now}",
                f"authorizes_risky_work={authorizes_risky_work}",
                f"authorizes_unreviewed_followthrough={authorizes_unreviewed_followthrough}",
                f"reusable_for_next_review={reusable_for_next_review}",
            ]
        )
    )


def _next_step_approval_boundary_as_prior_proof(
    *,
    token_sha256: str,
    proposed_step_sha256: str,
    proposed_verification_sha256: str,
    approval_proof_queue: list[str],
    approval_boundary_rows: list[dict[str, Any]],
    required_field: str = "required_before_risky_next_step",
) -> bool:
    if not _looks_like_sha256(token_sha256):
        return False
    expected_items = {
        "risk_classification",
        "exact_arguments_or_step",
        "approval_readiness_packet",
        "last_look_approval_packet",
        "approval_chain_proof",
        "post_approval_verification_receipt",
        "fresh_local_safe_review_after_approval",
    }
    if len(approval_boundary_rows) != len(expected_items):
        return False
    if {str(row.get("item") or "") for row in approval_boundary_rows} != expected_items:
        return False

    proof_queue = [str(command) for command in approval_proof_queue]
    risk_present = bool(proof_queue)
    expected_status = "held" if risk_present else "not_required"
    expected_ready = not risk_present
    next_proof_commands = {
        "approval_readiness_packet": proof_queue[0] if len(proof_queue) > 0 else "",
        "last_look_approval_packet": proof_queue[1] if len(proof_queue) > 1 else "",
        "approval_chain_proof": proof_queue[2] if len(proof_queue) > 2 else "",
        "post_approval_verification_receipt": proof_queue[3] if len(proof_queue) > 3 else "",
        "fresh_local_safe_review_after_approval": proof_queue[4] if len(proof_queue) > 4 else "",
    }
    non_authority_fields = (
        "authorizes_action_now",
        "authorizes_risky_work",
        "authorizes_unreviewed_followthrough",
        "authorizes_approval",
        "authorizes_model_call",
        "authorizes_tool_execution",
        "authorizes_personal_data_read",
        "authorizes_external_side_effect",
        "authorizes_timebox_reuse",
        "reusable_for_next_review",
        "reusable_for_recovery_review",
    )

    risk_signals: list[str] = []
    for row in approval_boundary_rows:
        item = str(row.get("item") or "")
        if row.get("status") != expected_status:
            return False
        if row.get("ready") is not expected_ready:
            return False
        expected_required = True if item == "risk_classification" else risk_present
        if row.get(required_field) is not expected_required:
            return False
        if item in next_proof_commands and str(row.get("next_proof_command") or "") != next_proof_commands[item]:
            return False
        if item == "risk_classification":
            risk_signals = [str(signal) for signal in (row.get("risk_signals") or [])]
        if any(row.get(field) is not False for field in non_authority_fields):
            return False

    if bool(risk_signals) != risk_present:
        return False
    recomputed = _risky_next_step_approval_boundary_token_sha256(
        proposed_step_sha256=proposed_step_sha256,
        proposed_verification_sha256=proposed_verification_sha256,
        risk_signals=risk_signals,
        approval_proof_queue=proof_queue,
        approval_boundary_rows=approval_boundary_rows,
    )
    return recomputed == token_sha256


def _next_step_approval_boundary_ready(
    *,
    token_sha256: str,
    proposed_step_sha256: str,
    proposed_verification_sha256: str,
    approval_proof_queue: list[str],
    approval_boundary_rows: list[dict[str, Any]],
) -> bool:
    proof_queue = [str(command) for command in approval_proof_queue]
    risk_signals = []
    for row in approval_boundary_rows:
        if row.get("item") == "risk_classification":
            risk_signals = [str(signal) for signal in (row.get("risk_signals") or [])]
            break
    if proof_queue or risk_signals:
        return False
    return _next_step_approval_boundary_as_prior_proof(
        token_sha256=token_sha256,
        proposed_step_sha256=proposed_step_sha256,
        proposed_verification_sha256=proposed_verification_sha256,
        approval_proof_queue=proof_queue,
        approval_boundary_rows=approval_boundary_rows,
    )


def _next_step_approval_boundary_metadata_ready(
    metadata: dict[str, Any],
    *,
    prefix: str,
    followup_authority_key: str,
    require_boundary_ready: bool | None = None,
) -> bool:
    proof_queue = [str(command) for command in (metadata.get(f"{prefix}_approval_proof_queue") or [])]
    boundary_rows = list(metadata.get(f"{prefix}_approval_boundary_rows") or [])
    token_sha256 = str(metadata.get(f"{prefix}_approval_boundary_token_sha256") or "")
    boundary_ready = _next_step_approval_boundary_ready(
        token_sha256=token_sha256,
        proposed_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
        proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
        approval_proof_queue=proof_queue,
        approval_boundary_rows=boundary_rows,
    )
    prior_proof_ready = _next_step_approval_boundary_as_prior_proof(
        token_sha256=token_sha256,
        proposed_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
        proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
        approval_proof_queue=proof_queue,
        approval_boundary_rows=boundary_rows,
    )
    if require_boundary_ready is not None and boundary_ready is not require_boundary_ready:
        return False
    if metadata.get(f"{prefix}_approval_proof_queue_count") != len(proof_queue):
        return False
    if metadata.get(f"{prefix}_approval_boundary_row_count") != len(boundary_rows):
        return False
    if metadata.get(f"{prefix}_approval_boundary_token_present") is not _looks_like_sha256(token_sha256):
        return False
    if metadata.get(f"{prefix}_approval_boundary_ready") is not boundary_ready:
        return False
    if metadata.get(f"{prefix}_approval_boundary_as_prior_proof") is not prior_proof_ready:
        return False
    if metadata.get(f"{prefix}_approval_required_before_review") is not bool(proof_queue):
        return False
    expected_next_command = proof_queue[0] if proof_queue else ""
    if metadata.get(f"{prefix}_next_approval_proof_command") != expected_next_command:
        return False
    for key in [
        f"{prefix}_approval_boundary_authorizes_action_now",
        f"{prefix}_approval_boundary_authorizes_risky_work",
        f"{prefix}_approval_boundary_authorizes_approval",
        f"{prefix}_approval_boundary_authorizes_model_call",
        f"{prefix}_approval_boundary_authorizes_tool_execution",
        f"{prefix}_approval_boundary_authorizes_personal_data_read",
        f"{prefix}_approval_boundary_authorizes_external_side_effect",
        f"{prefix}_approval_boundary_authorizes_timebox_reuse",
        followup_authority_key,
        f"{prefix}_approval_boundary_reusable_for_next_review",
        f"{prefix}_approval_boundary_reusable_for_recovery_review",
    ]:
        if metadata.get(key) is not False:
            return False
    return prior_proof_ready


def _carried_next_step_approval_boundary_ready_from_metadata(metadata: dict[str, Any]) -> bool:
    return _next_step_approval_boundary_metadata_ready(
        metadata,
        prefix="carried_next_step",
        followup_authority_key="carried_next_step_approval_boundary_authorizes_unreviewed_followthrough",
        require_boundary_ready=True,
    )


def _risky_recovery_step_approval_boundary_token_sha256(
    *,
    recovery_step_sha256: str,
    recovery_verification_sha256: str,
    risk_signals: list[str],
    approval_proof_queue: list[str],
    approval_boundary_rows: list[dict[str, Any]],
    approval_reference: str = "",
    authorizes_action_now: bool = False,
    authorizes_risky_work: bool = False,
    authorizes_unreviewed_followthrough: bool = False,
    reusable_for_recovery_review: bool = False,
) -> str:
    row_fingerprint = [
        "|".join(
            [
                str(row.get("item") or ""),
                str(row.get("status") or ""),
                str(_metadata_bool(row.get("ready"))),
                str(_metadata_bool(row.get("required_before_risky_recovery_step"))),
                str(row.get("next_proof_command") or ""),
                str(_metadata_bool(row.get("authorizes_action_now"))),
                str(_metadata_bool(row.get("authorizes_risky_work"))),
                str(_metadata_bool(row.get("authorizes_unreviewed_followthrough"))),
                str(_metadata_bool(row.get("authorizes_approval"))),
                str(_metadata_bool(row.get("authorizes_model_call"))),
                str(_metadata_bool(row.get("authorizes_tool_execution"))),
                str(_metadata_bool(row.get("authorizes_personal_data_read"))),
                str(_metadata_bool(row.get("authorizes_external_side_effect"))),
                str(_metadata_bool(row.get("authorizes_timebox_reuse"))),
                str(_metadata_bool(row.get("reusable_for_next_review"))),
                str(_metadata_bool(row.get("reusable_for_recovery_review"))),
            ]
        )
        for row in approval_boundary_rows
    ]
    return _text_sha256(
        "\n".join(
            [
                "risky_recovery_step_approval_boundary_v1",
                recovery_step_sha256,
                recovery_verification_sha256,
                ",".join(sorted(str(signal) for signal in risk_signals)),
                "\n".join(approval_proof_queue),
                "\n".join(row_fingerprint),
                f"approval_reference={approval_reference}",
                f"authorizes_action_now={authorizes_action_now}",
                f"authorizes_risky_work={authorizes_risky_work}",
                f"authorizes_unreviewed_followthrough={authorizes_unreviewed_followthrough}",
                f"reusable_for_recovery_review={reusable_for_recovery_review}",
            ]
        )
    )


def _recovery_step_approval_boundary_ready(
    *,
    token_sha256: str,
    recovery_step_sha256: str,
    recovery_verification_sha256: str,
    approval_proof_queue: list[str],
    approval_boundary_rows: list[dict[str, Any]],
    approval_reference: str = "",
    require_prior_proof: bool = False,
) -> bool:
    if not _looks_like_sha256(token_sha256):
        return False
    expected_items = {
        "risk_classification",
        "exact_arguments_or_step",
        "approval_readiness_packet",
        "last_look_approval_packet",
        "approval_chain_proof",
        "post_approval_verification_receipt",
        "fresh_local_safe_review_after_approval",
    }
    if len(approval_boundary_rows) != len(expected_items):
        return False
    if {str(row.get("item") or "") for row in approval_boundary_rows} != expected_items:
        return False

    proof_queue = [str(command) for command in approval_proof_queue]
    next_proof_commands = {
        "approval_readiness_packet": proof_queue[0] if len(proof_queue) > 0 else "",
        "last_look_approval_packet": proof_queue[1] if len(proof_queue) > 1 else "",
        "approval_chain_proof": proof_queue[2] if len(proof_queue) > 2 else "",
        "post_approval_verification_receipt": proof_queue[3] if len(proof_queue) > 3 else "",
        "fresh_local_safe_review_after_approval": proof_queue[4] if len(proof_queue) > 4 else "",
    }
    non_authority_fields = (
        "authorizes_action_now",
        "authorizes_risky_work",
        "authorizes_unreviewed_followthrough",
        "authorizes_approval",
        "authorizes_model_call",
        "authorizes_tool_execution",
        "authorizes_personal_data_read",
        "authorizes_external_side_effect",
        "authorizes_timebox_reuse",
        "reusable_for_next_review",
        "reusable_for_recovery_review",
    )

    risk_signals: list[str] = []
    for row in approval_boundary_rows:
        item = str(row.get("item") or "")
        if item == "risk_classification":
            risk_signals = [str(signal) for signal in (row.get("risk_signals") or [])]
            break
    risk_present = bool(risk_signals)
    expected_status = "held" if risk_present else "not_required"
    expected_ready = not risk_present
    for row in approval_boundary_rows:
        item = str(row.get("item") or "")
        if row.get("status") != expected_status:
            return False
        if row.get("ready") is not expected_ready:
            return False
        expected_required = True if item == "risk_classification" else risk_present
        if row.get("required_before_risky_recovery_step") is not expected_required:
            return False
        if item == "exact_arguments_or_step":
            if str(row.get("step_sha256") or "") != recovery_step_sha256:
                return False
            if str(row.get("verification_sha256") or "") != recovery_verification_sha256:
                return False
        if item in next_proof_commands and str(row.get("next_proof_command") or "") != next_proof_commands[item]:
            return False
        if any(row.get(field) is not False for field in non_authority_fields):
            return False

    if risk_present and not (proof_queue or approval_reference):
        return False
    if require_prior_proof and (not risk_present or not approval_reference):
        return False
    recomputed = _risky_recovery_step_approval_boundary_token_sha256(
        recovery_step_sha256=recovery_step_sha256,
        recovery_verification_sha256=recovery_verification_sha256,
        risk_signals=risk_signals,
        approval_proof_queue=proof_queue,
        approval_boundary_rows=approval_boundary_rows,
        approval_reference=approval_reference,
    )
    if recomputed != token_sha256:
        return False
    return True if require_prior_proof else not risk_present


def _looks_like_sha256(value: Any) -> bool:
    text = str(value or "").strip()
    return len(text) == 64 and all(char in "0123456789abcdefABCDEF" for char in text)


def _text_sha256(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _checkpoint_recovery_execute_handoff_token_sha256(handoff: dict[str, Any]) -> str:
    boundary_rows = handoff.get("recovery_step_approval_boundary_rows") or []
    boundary_row_parts = [
        "|".join(
            str(row.get(key) or "")
            for key in (
                "item",
                "status",
                "ready",
                "required_before_risky_recovery_step",
                "authorizes_action_now",
                "authorizes_risky_work",
                "authorizes_unreviewed_followthrough",
            )
        )
        for row in boundary_rows
        if isinstance(row, dict)
    ]
    return _text_sha256(
        "\n".join(
            [
                "checkpoint_recovery_execute_handoff_v1",
                str(handoff.get("state") or ""),
                str(handoff.get("reviewed")),
                "|".join(str(item) for item in handoff.get("missing_fields") or []),
                str(handoff.get("missing_field_count")),
                str(handoff.get("normal_followthrough_allowed")),
                str(handoff.get("recovery_followthrough_gate_state") or ""),
                "|".join(str(item) for item in handoff.get("recovery_closure_missing") or []),
                str(handoff.get("recovery_closure_missing_count")),
                "|".join(str(item) for item in handoff.get("recovery_closure_required_evidence") or []),
                str(handoff.get("recovery_closure_required_evidence_count")),
                "|".join(str(item) for item in handoff.get("recovery_closure_proof_queue") or []),
                str(handoff.get("recovery_closure_proof_queue_count")),
                str(handoff.get("recovery_closure_next_proof_command") or ""),
                str(handoff.get("recovery_closure_proof_queue_ready")),
                str(handoff.get("recovery_closure_approval_boundary") or ""),
                str(handoff.get("next_safe_command") or ""),
                "|".join(str(item) for item in handoff.get("risky_recovery_signals") or []),
                str(handoff.get("risky_recovery_signal_count")),
                str(handoff.get("approval_reference_provided")),
                str(handoff.get("approval_reference") or ""),
                str(handoff.get("recovery_step_approval_required_before_recovery")),
                "|".join(str(item) for item in handoff.get("recovery_step_approval_proof_queue") or []),
                str(handoff.get("recovery_step_approval_proof_queue_count")),
                str(handoff.get("recovery_step_next_approval_proof_command") or ""),
                "|".join(boundary_row_parts),
                str(handoff.get("recovery_step_approval_boundary_row_count")),
                str(handoff.get("recovery_step_approval_boundary_ready")),
                str(handoff.get("recovery_step_approval_boundary_as_prior_proof")),
                str(handoff.get("recovery_step_approval_boundary_token_sha256") or ""),
                str(handoff.get("recovery_step_sha256") or ""),
                str(handoff.get("recovery_verification_sha256") or ""),
                "|".join(
                    f"{flag}={handoff.get(flag)}"
                    for flag in _CHECKPOINT_RECOVERY_EXECUTE_HANDOFF_FALSE_FLAGS
                ),
                "|".join(
                    f"{flag}={handoff.get(flag)}"
                    for flag in _RECOVERY_STEP_APPROVAL_BOUNDARY_FALSE_FLAGS
                ),
            ]
        )
    )


def _autonomy_review_token_sha256(
    *,
    objective: str,
    stop_at: str,
    current_time: str,
    timezone_label: str,
    checkpoint_path: str,
    checkpoint_sha256: str,
    proposed_step_sha256: str,
    proposed_verification_sha256: str,
) -> str:
    return _text_sha256(
        "\n".join(
            [
                "autonomy_continuation_review_v1",
                objective,
                stop_at,
                current_time,
                timezone_label,
                checkpoint_path,
                checkpoint_sha256,
                proposed_step_sha256,
                proposed_verification_sha256,
            ]
        )
    )


def _operator_timebox_receipt_sha256(
    *,
    objective: str,
    stop_at: str,
    current_time: str,
    timezone_label: str,
    timebox_state: str,
    authorizes_execution: bool = False,
    authorizes_local_safe_step: bool = False,
    authorizes_risky_work: bool = False,
    authorizes_approval: bool = False,
    authorizes_timebox_reuse: bool = False,
    authorizes_model_call: bool = False,
    authorizes_tool_execution: bool = False,
    authorizes_personal_data_read: bool = False,
    authorizes_external_side_effect: bool = False,
    reusable_for_next_step: bool = False,
) -> str:
    return _text_sha256(
        "\n".join(
            [
                "operator_timebox_receipt_v1",
                objective,
                stop_at,
                current_time,
                timezone_label,
                timebox_state,
                f"authorizes_execution={authorizes_execution}",
                f"authorizes_local_safe_step={authorizes_local_safe_step}",
                f"authorizes_risky_work={authorizes_risky_work}",
                f"authorizes_approval={authorizes_approval}",
                f"authorizes_timebox_reuse={authorizes_timebox_reuse}",
                f"authorizes_model_call={authorizes_model_call}",
                f"authorizes_tool_execution={authorizes_tool_execution}",
                f"authorizes_personal_data_read={authorizes_personal_data_read}",
                f"authorizes_external_side_effect={authorizes_external_side_effect}",
                f"reusable_for_next_step={reusable_for_next_step}",
            ]
        )
    )


def _operator_supersession_token_sha256(
    *,
    objective: str,
    previous_instruction: str,
    latest_instruction: str,
    stop_at: str,
    current_time: str,
    timezone_label: str,
    supersession_state: str,
    timebox_receipt_sha256: str,
    authorizes_execution: bool = False,
    authorizes_local_safe_step: bool = False,
    authorizes_risky_work: bool = False,
    authorizes_approval: bool = False,
    authorizes_recovery_followthrough: bool = False,
    authorizes_timebox_override: bool = False,
    authorizes_goal_override: bool = False,
    authorizes_model_call: bool = False,
    authorizes_tool_execution: bool = False,
    authorizes_personal_data_read: bool = False,
    authorizes_external_side_effect: bool = False,
    reusable_for_next_review: bool = False,
    reusable_for_next_timebox: bool = False,
) -> str:
    return _text_sha256(
        "\n".join(
            [
                "operator_instruction_supersession_v1",
                objective,
                previous_instruction,
                latest_instruction,
                stop_at,
                current_time,
                timezone_label,
                supersession_state,
                timebox_receipt_sha256,
                "proof_only_newest_instruction_boundary",
                f"authorizes_execution={authorizes_execution}",
                f"authorizes_local_safe_step={authorizes_local_safe_step}",
                f"authorizes_risky_work={authorizes_risky_work}",
                f"authorizes_approval={authorizes_approval}",
                f"authorizes_recovery_followthrough={authorizes_recovery_followthrough}",
                f"authorizes_timebox_override={authorizes_timebox_override}",
                f"authorizes_goal_override={authorizes_goal_override}",
                f"authorizes_model_call={authorizes_model_call}",
                f"authorizes_tool_execution={authorizes_tool_execution}",
                f"authorizes_personal_data_read={authorizes_personal_data_read}",
                f"authorizes_external_side_effect={authorizes_external_side_effect}",
                f"reusable_for_next_review={reusable_for_next_review}",
                f"reusable_for_next_timebox={reusable_for_next_timebox}",
            ]
        )
    )


def _operator_supersession_token_boundary_rows(*, token_sha256: str, source: str) -> list[dict[str, Any]]:
    rows = [
        {
            "item": "operator_supersession_token",
            "source": source,
            "status": "present" if _looks_like_sha256(token_sha256) else "missing",
            "token_sha256": token_sha256,
        },
        {
            "item": "newest_instruction_scope",
            "source": source,
            "status": "proof_only_for_current_operator_instruction_review",
            "token_sha256": token_sha256,
        },
        {
            "item": "next_supersession_review",
            "source": source,
            "status": "fresh_latest_instruction_review_required",
            "token_sha256": token_sha256,
        },
    ]
    for row in rows:
        row.update(
            {
                "authorizes_execution": False,
                "authorizes_local_safe_step": False,
                "authorizes_risky_work": False,
                "authorizes_approval": False,
                "authorizes_recovery_followthrough": False,
                "authorizes_timebox_override": False,
                "authorizes_goal_override": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "reusable_for_next_review": False,
                "reusable_for_next_timebox": False,
            }
        )
    return rows


def _operator_supersession_token_boundary_ready(
    token_sha256: str,
    rows: list[dict[str, Any]],
    *,
    expected_source: str | None = None,
) -> bool:
    if expected_source is None:
        expected_source = str(rows[0].get("source") or "") if rows else ""
    expected_rows = _operator_supersession_token_boundary_rows(
        token_sha256=token_sha256,
        source=expected_source,
    )
    expected_boundary = [
        {
            "item": row.get("item"),
            "source": row.get("source"),
            "status": row.get("status"),
            "token_sha256": row.get("token_sha256"),
        }
        for row in expected_rows
    ]
    actual_boundary = [
        {
            "item": row.get("item"),
            "source": row.get("source"),
            "status": row.get("status"),
            "token_sha256": row.get("token_sha256"),
        }
        for row in rows
    ]
    return bool(
        _looks_like_sha256(token_sha256)
        and expected_source
        and len(rows) == len(expected_rows)
        and actual_boundary == expected_boundary
        and all(row.get("authorizes_execution") is False for row in rows)
        and all(row.get("authorizes_local_safe_step") is False for row in rows)
        and all(row.get("authorizes_risky_work") is False for row in rows)
        and all(row.get("authorizes_approval") is False for row in rows)
        and all(row.get("authorizes_recovery_followthrough") is False for row in rows)
        and all(row.get("authorizes_timebox_override") is False for row in rows)
        and all(row.get("authorizes_goal_override") is False for row in rows)
        and all(row.get("authorizes_model_call") is False for row in rows)
        and all(row.get("authorizes_tool_execution") is False for row in rows)
        and all(row.get("authorizes_personal_data_read") is False for row in rows)
        and all(row.get("authorizes_external_side_effect") is False for row in rows)
        and all(row.get("reusable_for_next_review") is False for row in rows)
        and all(row.get("reusable_for_next_timebox") is False for row in rows)
    )


def _carried_operator_supersession_token_boundary_ready(
    metadata: dict[str, Any],
    *,
    expected_source: str | None = None,
) -> bool:
    rows = list(metadata.get("supersession_token_boundary_rows") or [])
    if metadata.get("supersession_token_present") is not True:
        return False
    if metadata.get("supersession_token_boundary_row_count") != len(rows):
        return False
    if metadata.get("supersession_token_boundary_ready") is not True:
        return False
    if metadata.get("next_supersession_requires_fresh_latest_instruction_review") is not True:
        return False
    for flag in [
        "supersession_token_authorizes_execution",
        "supersession_token_authorizes_local_safe_step",
        "supersession_token_authorizes_risky_work",
        "supersession_token_authorizes_approval",
        "supersession_token_authorizes_recovery_followthrough",
        "supersession_token_authorizes_timebox_override",
        "supersession_token_authorizes_goal_override",
        "supersession_token_authorizes_model_call",
        "supersession_token_authorizes_tool_execution",
        "supersession_token_authorizes_personal_data_read",
        "supersession_token_authorizes_external_side_effect",
        "supersession_token_reusable_for_next_review",
        "supersession_token_reusable_for_next_timebox",
    ]:
        if metadata.get(flag) is not False:
            return False
    token_sha256 = str(metadata.get("supersession_token_sha256") or "")
    timebox_rows = list(metadata.get("timebox_review_contract_rows") or [])
    explicit_timebox_row = next(
        (row for row in timebox_rows if isinstance(row, dict) and row.get("item") == "explicit_stop_window"),
        {},
    )
    expected_token_sha256 = _operator_supersession_token_sha256(
        objective=str(metadata.get("objective") or ""),
        previous_instruction=str(metadata.get("previous_instruction") or ""),
        latest_instruction=str(metadata.get("latest_instruction") or ""),
        stop_at=str(metadata.get("stop_at") or explicit_timebox_row.get("stop_at") or ""),
        current_time=str(metadata.get("current_time") or explicit_timebox_row.get("current_time") or ""),
        timezone_label=str(metadata.get("timezone") or explicit_timebox_row.get("timezone") or ""),
        supersession_state=str(metadata.get("supersession_state") or ""),
        timebox_receipt_sha256=str(metadata.get("timebox_receipt_sha256") or ""),
    )
    if token_sha256 != expected_token_sha256:
        return False
    return _operator_supersession_token_boundary_ready(
        token_sha256,
        rows,
        expected_source=expected_source,
    )


def _operator_supersession_contract_ready(rows: list[dict[str, Any]], *, latest_is_stop: bool) -> bool:
    expected_rows = [
        {
            "item": "newest_instruction",
            "required": True,
            "source": "latest operator instruction",
            "authorizes_goal_override": True,
        },
        {
            "item": "active_timebox",
            "required": True,
            "source": "operator timebox contract",
            "authorizes_goal_override": False,
        },
        {
            "item": "resume_gate_review",
            "required": not latest_is_stop,
            "source": "autonomy resume gate",
            "authorizes_goal_override": False,
        },
        {
            "item": "one_step_local_safe_review",
            "required": not latest_is_stop,
            "source": "autonomy continuation execution packet",
            "authorizes_goal_override": False,
        },
        {
            "item": "stop_or_pause_brake",
            "required": latest_is_stop,
            "source": "latest operator instruction",
            "authorizes_goal_override": True,
        },
    ]
    expected_contract = [
        {
            "item": row["item"],
            "required": row["required"],
            "source": row["source"],
            "authorizes_goal_override": row["authorizes_goal_override"],
        }
        for row in expected_rows
    ]
    actual_contract = [
        {
            "item": row.get("item"),
            "required": row.get("required"),
            "source": row.get("source"),
            "authorizes_goal_override": row.get("authorizes_goal_override"),
        }
        for row in rows
    ]
    return bool(
        len(rows) == len(expected_rows)
        and actual_contract == expected_contract
        and all(row.get("authorizes_execution") is False for row in rows)
        and all(row.get("authorizes_risky_work") is False for row in rows)
        and all(row.get("authorizes_approval") is False for row in rows)
        and all(row.get("authorizes_recovery_followthrough") is False for row in rows)
        and all(row.get("authorizes_timebox_override") is False for row in rows)
    )


def _fresh_continuation_review_contract_ready(rows: list[dict[str, Any]]) -> bool:
    expected_rows = [
        ("fresh_operator_timebox", "operator timebox contract", True),
        ("fresh_checkpoint", "work block checkpoint", True),
        ("fresh_recovery_cockpit", "checkpoint recovery cockpit", True),
        ("fresh_local_safe_step", "autonomy continuation execution packet", True),
        ("fresh_continuation_review_token", "continuation review token", True),
        ("prior_step_closure", "autonomy step closure packet", False),
    ]
    actual_contract = [
        {
            "item": row.get("item"),
            "source": row.get("source"),
            "fresh_required": row.get("fresh_required"),
        }
        for row in rows
    ]
    expected_contract = [
        {"item": item, "source": source, "fresh_required": fresh_required}
        for item, source, fresh_required in expected_rows
    ]
    return bool(
        len(rows) == len(expected_rows)
        and actual_contract == expected_contract
        and all(row.get("prior_artifact_reusable") is False for row in rows)
        and all(row.get("authorizes_action_now") is False for row in rows)
        and all(row.get("authorizes_risky_work") is False for row in rows)
        and all(row.get("authorizes_model_call") is False for row in rows)
        and all(row.get("authorizes_tool_execution") is False for row in rows)
        and all(row.get("authorizes_personal_data_read") is False for row in rows)
        and all(row.get("authorizes_external_side_effect") is False for row in rows)
        and all(row.get("authorizes_approval") is False for row in rows)
    )


def _checkpoint_route_token_sha256(
    *,
    objective: str,
    latest_checkpoint_path: str,
    latest_checkpoint_sha256: str,
    checkpoint_freshness: str,
    checkpoint_needs_review: bool,
    authorizes_continuation: bool = False,
    authorizes_local_safe_step: bool = False,
    authorizes_risky_work: bool = False,
    authorizes_approval: bool = False,
    authorizes_recovery_followthrough: bool = False,
    authorizes_checkpoint_reuse: bool = False,
    authorizes_model_call: bool = False,
    authorizes_tool_execution: bool = False,
    authorizes_personal_data_read: bool = False,
    authorizes_external_side_effect: bool = False,
    reusable_for_next_review: bool = False,
    reusable_for_next_checkpoint: bool = False,
) -> str:
    return hashlib.sha256(
        "\n".join(
            [
                "checkpoint_recovery_route_boundary_v1",
                objective,
                latest_checkpoint_path,
                latest_checkpoint_sha256,
                checkpoint_freshness,
                str(bool(checkpoint_needs_review)),
                "proof_only_checkpoint_route_boundary",
                "stale_or_missing_routes_to_recovery_review",
                f"authorizes_continuation={authorizes_continuation}",
                f"authorizes_local_safe_step={authorizes_local_safe_step}",
                f"authorizes_risky_work={authorizes_risky_work}",
                f"authorizes_approval={authorizes_approval}",
                f"authorizes_recovery_followthrough={authorizes_recovery_followthrough}",
                f"authorizes_checkpoint_reuse={authorizes_checkpoint_reuse}",
                f"authorizes_model_call={authorizes_model_call}",
                f"authorizes_tool_execution={authorizes_tool_execution}",
                f"authorizes_personal_data_read={authorizes_personal_data_read}",
                f"authorizes_external_side_effect={authorizes_external_side_effect}",
                f"reusable_for_next_review={reusable_for_next_review}",
                f"reusable_for_next_checkpoint={reusable_for_next_checkpoint}",
            ]
        ).encode("utf-8")
    ).hexdigest()


def _checkpoint_route_boundary_rows(
    *,
    token_sha256: str,
    source: str,
    checkpoint_freshness: str,
    checkpoint_needs_review: bool,
) -> list[dict[str, Any]]:
    route_status = "route_to_recovery_review" if checkpoint_needs_review else "fresh_checkpoint_bound_for_review"
    rows = [
        {
            "item": "checkpoint_route_token",
            "source": source,
            "status": "present" if _looks_like_sha256(token_sha256) else "missing",
            "token_sha256": token_sha256,
            "checkpoint_freshness": checkpoint_freshness,
            "checkpoint_needs_review": checkpoint_needs_review,
        },
        {
            "item": "stale_or_missing_checkpoint_route",
            "source": source,
            "status": route_status,
            "token_sha256": token_sha256,
            "checkpoint_freshness": checkpoint_freshness,
            "checkpoint_needs_review": checkpoint_needs_review,
        },
        {
            "item": "next_checkpoint_recovery_review",
            "source": source,
            "status": "fresh_checkpoint_recovery_review_required",
            "token_sha256": token_sha256,
            "checkpoint_freshness": checkpoint_freshness,
            "checkpoint_needs_review": checkpoint_needs_review,
        },
    ]
    for row in rows:
        row.update(
            {
                "authorizes_continuation": False,
                "authorizes_local_safe_step": False,
                "authorizes_risky_work": False,
                "authorizes_approval": False,
                "authorizes_recovery_followthrough": False,
                "authorizes_checkpoint_reuse": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "reusable_for_next_review": False,
                "reusable_for_next_checkpoint": False,
            }
        )
    return rows


def _checkpoint_route_boundary_ready(
    token_sha256: str,
    rows: list[dict[str, Any]],
    *,
    expected_source: str | None = None,
    expected_checkpoint_freshness: str | None = None,
    expected_checkpoint_needs_review: bool | None = None,
) -> bool:
    if expected_source is None:
        expected_source = str(rows[0].get("source") or "") if rows else ""
    if expected_checkpoint_freshness is None:
        expected_checkpoint_freshness = str(rows[0].get("checkpoint_freshness") or "") if rows else ""
    if expected_checkpoint_needs_review is None:
        if not rows or rows[0].get("checkpoint_needs_review") not in {True, False}:
            return False
        expected_checkpoint_needs_review = rows[0].get("checkpoint_needs_review")
    if expected_checkpoint_needs_review not in {True, False}:
        return False
    expected_rows = _checkpoint_route_boundary_rows(
        token_sha256=token_sha256,
        source=expected_source,
        checkpoint_freshness=expected_checkpoint_freshness,
        checkpoint_needs_review=expected_checkpoint_needs_review,
    )
    expected_boundary = [
        {
            "item": row.get("item"),
            "source": row.get("source"),
            "status": row.get("status"),
            "token_sha256": row.get("token_sha256"),
            "checkpoint_freshness": row.get("checkpoint_freshness"),
            "checkpoint_needs_review": row.get("checkpoint_needs_review"),
        }
        for row in expected_rows
    ]
    actual_boundary = [
        {
            "item": row.get("item"),
            "source": row.get("source"),
            "status": row.get("status"),
            "token_sha256": row.get("token_sha256"),
            "checkpoint_freshness": row.get("checkpoint_freshness"),
            "checkpoint_needs_review": row.get("checkpoint_needs_review"),
        }
        for row in rows
    ]
    return bool(
        _looks_like_sha256(token_sha256)
        and expected_source
        and expected_checkpoint_freshness
        and len(rows) == len(expected_rows)
        and actual_boundary == expected_boundary
        and all(row.get("authorizes_continuation") is False for row in rows)
        and all(row.get("authorizes_local_safe_step") is False for row in rows)
        and all(row.get("authorizes_risky_work") is False for row in rows)
        and all(row.get("authorizes_approval") is False for row in rows)
        and all(row.get("authorizes_recovery_followthrough") is False for row in rows)
        and all(row.get("authorizes_checkpoint_reuse") is False for row in rows)
        and all(row.get("authorizes_model_call") is False for row in rows)
        and all(row.get("authorizes_tool_execution") is False for row in rows)
        and all(row.get("authorizes_personal_data_read") is False for row in rows)
        and all(row.get("authorizes_external_side_effect") is False for row in rows)
        and all(row.get("reusable_for_next_review") is False for row in rows)
        and all(row.get("reusable_for_next_checkpoint") is False for row in rows)
    )


def _carried_checkpoint_route_boundary_ready(metadata: dict[str, Any]) -> bool:
    rows = list(metadata.get("checkpoint_route_boundary_rows") or [])
    if metadata.get("checkpoint_route_token_present") is not True:
        return False
    if not _looks_like_sha256(str(metadata.get("checkpoint_route_token_sha256") or "")):
        return False
    if metadata.get("checkpoint_route_boundary_row_count") != len(rows):
        return False
    if metadata.get("checkpoint_route_boundary_ready") is not True:
        return False
    if metadata.get("next_checkpoint_route_requires_fresh_recovery_review") is not True:
        return False
    for flag in [
        "checkpoint_route_authorizes_continuation",
        "checkpoint_route_authorizes_local_safe_step",
        "checkpoint_route_authorizes_risky_work",
        "checkpoint_route_authorizes_approval",
        "checkpoint_route_authorizes_recovery_followthrough",
        "checkpoint_route_authorizes_checkpoint_reuse",
        "checkpoint_route_authorizes_model_call",
        "checkpoint_route_authorizes_tool_execution",
        "checkpoint_route_authorizes_personal_data_read",
        "checkpoint_route_authorizes_external_side_effect",
        "checkpoint_route_reusable_for_next_review",
        "checkpoint_route_reusable_for_next_checkpoint",
    ]:
        if metadata.get(flag) is not False:
            return False
    if metadata.get("checkpoint_needs_review") not in {True, False}:
        return False
    return _checkpoint_route_boundary_ready(
        str(metadata.get("checkpoint_route_token_sha256") or ""),
        rows,
        expected_checkpoint_freshness=str(metadata.get("checkpoint_freshness") or ""),
        expected_checkpoint_needs_review=metadata.get("checkpoint_needs_review"),
    )


def _operator_timebox_review_contract_rows(
    *,
    timebox_state: str,
    stop_at: str,
    current_time: str,
    timezone_label: str,
) -> list[dict[str, Any]]:
    rows = [
        {
            "item": "explicit_stop_window",
            "source": "operator timebox contract",
            "state": timebox_state,
            "stop_at": stop_at,
            "current_time": current_time,
            "timezone": timezone_label,
            "fresh_required": True,
            "reusable_for_next_step": False,
            "authorizes_execution": False,
            "authorizes_local_safe_step": False,
            "authorizes_risky_work": False,
            "authorizes_approval": False,
            "authorizes_timebox_reuse": False,
        },
        {
            "item": "newer_instruction_check",
            "source": "operator instruction supersession packet",
            "state": "required_before_continuation",
            "fresh_required": True,
            "reusable_for_next_step": False,
            "authorizes_execution": False,
            "authorizes_local_safe_step": False,
            "authorizes_risky_work": False,
            "authorizes_approval": False,
            "authorizes_timebox_reuse": False,
        },
        {
            "item": "resume_gate_review",
            "source": "autonomy resume gate",
            "state": "required_after_timebox",
            "fresh_required": True,
            "reusable_for_next_step": False,
            "authorizes_execution": False,
            "authorizes_local_safe_step": False,
            "authorizes_risky_work": False,
            "authorizes_approval": False,
            "authorizes_timebox_reuse": False,
        },
        {
            "item": "one_step_continuation_limit",
            "source": "autonomy continuation execution packet",
            "state": "single_local_safe_step_only",
            "fresh_required": True,
            "reusable_for_next_step": False,
            "authorizes_execution": False,
            "authorizes_local_safe_step": False,
            "authorizes_risky_work": False,
            "authorizes_approval": False,
            "authorizes_timebox_reuse": False,
        },
        {
            "item": "post_step_closure_required",
            "source": "autonomy step closure packet",
            "state": "required_before_next_review",
            "fresh_required": True,
            "reusable_for_next_step": False,
            "authorizes_execution": False,
            "authorizes_local_safe_step": False,
            "authorizes_risky_work": False,
            "authorizes_approval": False,
            "authorizes_timebox_reuse": False,
        },
        {
            "item": "next_timebox_freshness",
            "source": "autonomy cycle ledger",
            "state": "fresh_timebox_required",
            "fresh_required": True,
            "reusable_for_next_step": False,
            "authorizes_execution": False,
            "authorizes_local_safe_step": False,
            "authorizes_risky_work": False,
            "authorizes_approval": False,
            "authorizes_timebox_reuse": False,
        },
    ]
    for row in rows:
        row.update(
            {
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
            }
        )
    return rows


def _operator_timebox_review_contract_ready(
    rows: list[dict[str, Any]],
    *,
    timebox_state: str,
    stop_at: str,
    current_time: str,
    timezone_label: str,
) -> bool:
    explicit_row = next(
        (row for row in rows if isinstance(row, dict) and row.get("item") == "explicit_stop_window"),
        {},
    )
    timebox_state = timebox_state or str(explicit_row.get("state") or "")
    stop_at = stop_at or str(explicit_row.get("stop_at") or "")
    current_time = current_time or str(explicit_row.get("current_time") or "")
    timezone_label = timezone_label or str(explicit_row.get("timezone") or "")
    expected_rows = _operator_timebox_review_contract_rows(
        timebox_state=timebox_state,
        stop_at=stop_at,
        current_time=current_time,
        timezone_label=timezone_label,
    )
    expected_shape = [
        {
            "item": row.get("item"),
            "source": row.get("source"),
            "state": row.get("state"),
            "stop_at": row.get("stop_at"),
            "current_time": row.get("current_time"),
            "timezone": row.get("timezone"),
            "fresh_required": row.get("fresh_required"),
        }
        for row in expected_rows
    ]
    actual_shape = [
        {
            "item": row.get("item"),
            "source": row.get("source"),
            "state": row.get("state"),
            "stop_at": row.get("stop_at"),
            "current_time": row.get("current_time"),
            "timezone": row.get("timezone"),
            "fresh_required": row.get("fresh_required"),
        }
        for row in rows
    ]
    no_authority_fields = [
        "reusable_for_next_step",
        "authorizes_execution",
        "authorizes_local_safe_step",
        "authorizes_risky_work",
        "authorizes_approval",
        "authorizes_timebox_reuse",
        "authorizes_model_call",
        "authorizes_tool_execution",
        "authorizes_personal_data_read",
        "authorizes_external_side_effect",
    ]
    return bool(
        timebox_state
        and current_time
        and timezone_label
        and len(rows) == len(expected_rows)
        and actual_shape == expected_shape
        and all(row.get(field) is False for row in rows for field in no_authority_fields)
    )


def _carried_operator_timebox_review_contract_ready(metadata: dict[str, Any]) -> bool:
    rows = list(metadata.get("timebox_review_contract_rows") or [])
    timebox_state = str(metadata.get("timebox_state") or "")
    if timebox_state not in {"STOP_WINDOW_ACTIVE", "STOP_TIME_REACHED", "HELD_FOR_PARSEABLE_TIMEBOX"}:
        return False
    expected_can_continue = timebox_state == "STOP_WINDOW_ACTIVE"
    if metadata.get("can_continue_now") is not expected_can_continue:
        return False
    if metadata.get("should_stop_now") is not (not expected_can_continue):
        return False
    missing_timebox_proof = list(metadata.get("missing_timebox_proof") or [])
    if metadata.get("missing_timebox_proof_count") != len(missing_timebox_proof):
        return False
    if timebox_state == "HELD_FOR_PARSEABLE_TIMEBOX" and not missing_timebox_proof:
        return False
    if timebox_state in {"STOP_WINDOW_ACTIVE", "STOP_TIME_REACHED"} and missing_timebox_proof:
        return False
    if metadata.get("timebox_receipt_present") is not True:
        return False
    timebox_receipt_sha256 = str(metadata.get("timebox_receipt_sha256") or "")
    if not _looks_like_sha256(timebox_receipt_sha256):
        return False
    explicit_row = next(
        (row for row in rows if isinstance(row, dict) and row.get("item") == "explicit_stop_window"),
        {},
    )
    stop_at = str(metadata.get("stop_at") or explicit_row.get("stop_at") or "")
    current_time = str(metadata.get("current_time") or explicit_row.get("current_time") or "")
    timezone_label = str(metadata.get("timezone") or explicit_row.get("timezone") or "")
    expected_receipt_sha256 = _operator_timebox_receipt_sha256(
        objective=str(metadata.get("objective") or ""),
        stop_at=stop_at,
        current_time=current_time,
        timezone_label=timezone_label,
        timebox_state=timebox_state,
    )
    if timebox_receipt_sha256 != expected_receipt_sha256:
        return False
    if metadata.get("timebox_review_contract_row_count") != len(rows):
        return False
    if metadata.get("timebox_review_contract_ready") is not True:
        return False
    if metadata.get("next_step_requires_fresh_timebox") is not True:
        return False
    for flag in [
        "timebox_authorizes_execution",
        "timebox_authorizes_local_safe_step",
        "timebox_authorizes_risky_work",
        "timebox_authorizes_approval",
        "timebox_authorizes_timebox_reuse",
        "timebox_authorizes_model_call",
        "timebox_authorizes_tool_execution",
        "timebox_authorizes_personal_data_read",
        "timebox_authorizes_external_side_effect",
        "timebox_reusable_for_next_step",
    ]:
        if metadata.get(flag) is not False:
            return False
    return _operator_timebox_review_contract_ready(
        rows,
        timebox_state=timebox_state,
        stop_at=stop_at,
        current_time=current_time,
        timezone_label=timezone_label,
    )


def _awake_guard_requested(text: str) -> bool:
    normalized = text.lower()
    return any(
        phrase in normalized
        for phrase in [
            "keep awake",
            "stay awake",
            "computer awake",
            "mac awake",
            "caffeinate",
            "prevent sleep",
            "disable sleep",
            "no sleep",
        ]
    )


def _awake_guard_token_sha256(
    *,
    objective: str,
    stop_at: str,
    current_time: str,
    timezone_label: str,
    requested: bool,
    authorizes_os_wake_lock: bool = False,
    authorizes_shell_execution: bool = False,
    authorizes_computer_control: bool = False,
    authorizes_approval: bool = False,
    authorizes_model_call: bool = False,
    authorizes_tool_execution: bool = False,
    authorizes_personal_data_read: bool = False,
    authorizes_external_side_effect: bool = False,
    reusable_for_next_timebox: bool = False,
) -> str:
    return _text_sha256(
        "\n".join(
            [
                "awake_guard_boundary_v1",
                objective,
                stop_at,
                current_time,
                timezone_label,
                "requested" if requested else "not_requested",
                f"authorizes_os_wake_lock={authorizes_os_wake_lock}",
                f"authorizes_shell_execution={authorizes_shell_execution}",
                f"authorizes_computer_control={authorizes_computer_control}",
                f"authorizes_approval={authorizes_approval}",
                f"authorizes_model_call={authorizes_model_call}",
                f"authorizes_tool_execution={authorizes_tool_execution}",
                f"authorizes_personal_data_read={authorizes_personal_data_read}",
                f"authorizes_external_side_effect={authorizes_external_side_effect}",
                f"reusable_for_next_timebox={reusable_for_next_timebox}",
            ]
        )
    )


def _awake_guard_boundary_rows(*, requested: bool, stop_at: str, current_time: str) -> list[dict[str, Any]]:
    rows = [
        {
            "item": "awake_request_detected",
            "status": "requested" if requested else "not_requested",
            "stop_at": stop_at,
            "current_time": current_time,
            "authorizes_os_wake_lock": False,
            "authorizes_shell_execution": False,
            "authorizes_computer_control": False,
            "authorizes_approval": False,
            "reusable_for_next_timebox": False,
        },
        {
            "item": "os_power_management_boundary",
            "status": "approval_gated_if_shell_or_system_change",
            "authorizes_os_wake_lock": False,
            "authorizes_shell_execution": False,
            "authorizes_computer_control": False,
            "authorizes_approval": False,
            "reusable_for_next_timebox": False,
        },
        {
            "item": "next_awake_guard_review",
            "status": "fresh_review_required",
            "authorizes_os_wake_lock": False,
            "authorizes_shell_execution": False,
            "authorizes_computer_control": False,
            "authorizes_approval": False,
            "reusable_for_next_timebox": False,
        },
    ]
    for row in rows:
        row.update(
            {
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
            }
        )
    return rows


def _awake_guard_boundary_ready(
    *,
    token_sha256: str,
    rows: list[dict[str, Any]],
    requested: bool,
    stop_at: str,
    current_time: str,
    objective: str,
    timezone_label: str,
) -> bool:
    expected_rows = _awake_guard_boundary_rows(
        requested=requested,
        stop_at=stop_at,
        current_time=current_time,
    )
    expected_token = _awake_guard_token_sha256(
        objective=objective,
        stop_at=stop_at,
        current_time=current_time,
        timezone_label=timezone_label,
        requested=requested,
    )
    no_authority_fields = [
        "authorizes_os_wake_lock",
        "authorizes_shell_execution",
        "authorizes_computer_control",
        "authorizes_approval",
        "authorizes_model_call",
        "authorizes_tool_execution",
        "authorizes_personal_data_read",
        "authorizes_external_side_effect",
        "reusable_for_next_timebox",
    ]
    return bool(
        _looks_like_sha256(token_sha256)
        and token_sha256 == expected_token
        and len(rows) == len(expected_rows)
        and [
            {
                "item": row.get("item"),
                "status": row.get("status"),
                "stop_at": row.get("stop_at"),
                "current_time": row.get("current_time"),
            }
            for row in rows
        ]
        == [
            {
                "item": row.get("item"),
                "status": row.get("status"),
                "stop_at": row.get("stop_at"),
                "current_time": row.get("current_time"),
            }
            for row in expected_rows
        ]
        and all(row.get(field) is False for row in rows for field in no_authority_fields)
    )


def _awake_guard_boundary_ready_from_metadata(
    metadata: dict[str, Any],
    *,
    default_objective: str = "",
    default_timezone: str = "Asia/Seoul",
) -> bool:
    rows = metadata.get("awake_guard_boundary_rows")
    if not isinstance(rows, list):
        rows = []
    if metadata.get("awake_guard_token_present") is not True:
        return False
    if metadata.get("awake_guard_boundary_row_count") != len(rows):
        return False
    if metadata.get("awake_guard_os_wake_lock_boundary_ready") is not True:
        return False
    if metadata.get("next_awake_guard_requires_fresh_review") is not True:
        return False
    for flag in [
        "awake_guard_authorizes_os_wake_lock",
        "awake_guard_authorizes_shell_execution",
        "awake_guard_authorizes_computer_control",
        "awake_guard_authorizes_approval",
        "awake_guard_authorizes_model_call",
        "awake_guard_authorizes_tool_execution",
        "awake_guard_authorizes_personal_data_read",
        "awake_guard_authorizes_external_side_effect",
        "awake_guard_reusable_for_next_timebox",
        "awake_guard_caffeinate_command_authorized",
        "awake_guard_keep_awake_command_authorized",
        "awake_guard_authorizes_unattended_execution",
        "awake_guard_authorizes_continuation_window",
        "awake_guard_reusable_as_execution_permission",
    ]:
        if metadata.get(flag) is not False:
            return False
    for flag in [
        "awake_guard_requires_separate_operator_request",
        "awake_guard_requires_separate_shell_approval",
    ]:
        if metadata.get(flag) is not True:
            return False
    requested = metadata.get("awake_guard_requested")
    if requested is not True and requested is not False:
        return False
    request_row = next((row for row in rows if isinstance(row, dict) and row.get("item") == "awake_request_detected"), {})
    stop_at = str(metadata.get("stop_at") or request_row.get("stop_at") or "")
    current_time = str(metadata.get("current_time") or request_row.get("current_time") or "")
    return _awake_guard_boundary_ready(
        token_sha256=str(metadata.get("awake_guard_token_sha256") or ""),
        rows=rows,
        requested=requested,
        stop_at=stop_at,
        current_time=current_time,
        objective=str(metadata.get("objective") or default_objective),
        timezone_label=str(metadata.get("timezone") or default_timezone),
    )


def _local_safe_recovery_execution_token_boundary_rows(*, token_sha256: str, source: str) -> list[dict[str, Any]]:
    rows = [
        {
            "item": "local_safe_recovery_execution_token",
            "source": source,
            "status": "present" if _looks_like_sha256(token_sha256) else "missing",
            "token_sha256": token_sha256,
        },
        {
            "item": "recovery_execution_scope",
            "source": source,
            "status": "proof_only_for_reviewed_local_safe_recovery",
            "token_sha256": token_sha256,
        },
        {
            "item": "next_recovery_execution_review",
            "source": source,
            "status": "fresh_local_safe_token_required",
            "token_sha256": token_sha256,
        },
    ]
    for row in rows:
        row.update(
            {
                "authorizes_resume_gate": False,
                "authorizes_next_step": False,
                "authorizes_risky_work": False,
                "authorizes_approval": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "reusable_for_future_recovery": False,
            }
        )
    return rows


def _local_safe_recovery_execution_token_boundary_ready(
    token_sha256: str,
    rows: list[dict[str, Any]],
    *,
    expected_source: str | None = None,
) -> bool:
    if expected_source is None:
        expected_source = str(rows[0].get("source") or "") if rows else ""
    expected_rows = _local_safe_recovery_execution_token_boundary_rows(
        token_sha256=token_sha256,
        source=expected_source,
    )
    no_authority_fields = [
        "authorizes_resume_gate",
        "authorizes_next_step",
        "authorizes_risky_work",
        "authorizes_approval",
        "authorizes_model_call",
        "authorizes_tool_execution",
        "authorizes_personal_data_read",
        "authorizes_external_side_effect",
        "reusable_for_future_recovery",
    ]
    return bool(
        _looks_like_sha256(token_sha256)
        and expected_source
        and len(rows) == len(expected_rows)
        and [
            {
                "item": row.get("item"),
                "source": row.get("source"),
                "status": row.get("status"),
                "token_sha256": row.get("token_sha256"),
            }
            for row in rows
        ]
        == [
            {
                "item": row.get("item"),
                "source": row.get("source"),
                "status": row.get("status"),
                "token_sha256": row.get("token_sha256"),
            }
            for row in expected_rows
        ]
        and all(row.get(field) is False for row in rows for field in no_authority_fields)
    )


def _recovery_execution_readiness_token_boundary_rows(*, token_sha256: str, source: str) -> list[dict[str, Any]]:
    rows = [
        {
            "item": "recovery_execution_readiness_token",
            "source": source,
            "status": "present" if _looks_like_sha256(token_sha256) else "missing",
            "token_sha256": token_sha256,
        },
        {
            "item": "recovery_execution_readiness_scope",
            "source": source,
            "status": "proof_only_for_scorecard_and_contract",
            "token_sha256": token_sha256,
        },
        {
            "item": "next_recovery_execution_readiness_review",
            "source": source,
            "status": "fresh_readiness_token_required",
            "token_sha256": token_sha256,
        },
    ]
    for row in rows:
        row.update(
            {
                "authorizes_resume_gate": False,
                "authorizes_next_step": False,
                "authorizes_risky_work": False,
                "authorizes_approval": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "authorizes_unreviewed_followthrough": False,
                "reusable_for_future_recovery": False,
            }
        )
    return rows


def _recovery_execution_readiness_token_boundary_ready(
    token_sha256: str,
    rows: list[dict[str, Any]],
    *,
    expected_source: str | None = None,
) -> bool:
    if expected_source is None:
        expected_source = str(rows[0].get("source") or "") if rows else ""
    expected_rows = _recovery_execution_readiness_token_boundary_rows(
        token_sha256=token_sha256,
        source=expected_source,
    )
    expected_boundary = [
        {
            "item": row.get("item"),
            "source": row.get("source"),
            "status": row.get("status"),
            "token_sha256": row.get("token_sha256"),
        }
        for row in expected_rows
    ]
    actual_boundary = [
        {
            "item": row.get("item"),
            "source": row.get("source"),
            "status": row.get("status"),
            "token_sha256": row.get("token_sha256"),
        }
        for row in rows
    ]
    return bool(
        _looks_like_sha256(token_sha256)
        and expected_source
        and len(rows) == len(expected_rows)
        and actual_boundary == expected_boundary
        and all(row.get("authorizes_resume_gate") is False for row in rows)
        and all(row.get("authorizes_next_step") is False for row in rows)
        and all(row.get("authorizes_risky_work") is False for row in rows)
        and all(row.get("authorizes_approval") is False for row in rows)
        and all(row.get("authorizes_model_call") is False for row in rows)
        and all(row.get("authorizes_tool_execution") is False for row in rows)
        and all(row.get("authorizes_personal_data_read") is False for row in rows)
        and all(row.get("authorizes_external_side_effect") is False for row in rows)
        and all(row.get("authorizes_unreviewed_followthrough") is False for row in rows)
        and all(row.get("reusable_for_future_recovery") is False for row in rows)
    )


def _recovery_followthrough_token_boundary_rows(*, token_sha256: str, source: str) -> list[dict[str, Any]]:
    rows = [
        {
            "item": "recovery_followthrough_token",
            "source": source,
            "status": "present" if _looks_like_sha256(token_sha256) else "missing",
            "token_sha256": token_sha256,
        },
        {
            "item": "reviewed_recovery_scope",
            "source": source,
            "status": "proof_only_for_reviewed_recovery_followthrough",
            "token_sha256": token_sha256,
        },
        {
            "item": "next_recovery_followthrough_review",
            "source": source,
            "status": "fresh_followthrough_token_required",
            "token_sha256": token_sha256,
        },
    ]
    for row in rows:
        row.update(
            {
                "authorizes_resume_gate": False,
                "authorizes_next_step": False,
                "authorizes_risky_work": False,
                "authorizes_approval": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "authorizes_unreviewed_followthrough": False,
                "reusable_for_future_recovery": False,
                "reusable_for_next_review": False,
            }
        )
    return rows


def _recovery_followthrough_token_boundary_ready(
    token_sha256: str,
    rows: list[dict[str, Any]],
    *,
    expected_source: str | None = None,
) -> bool:
    if expected_source is None:
        expected_source = str(rows[0].get("source") or "") if rows else ""
    expected_rows = _recovery_followthrough_token_boundary_rows(
        token_sha256=token_sha256,
        source=expected_source,
    )
    expected_boundary = [
        {
            "item": row.get("item"),
            "source": row.get("source"),
            "status": row.get("status"),
            "token_sha256": row.get("token_sha256"),
        }
        for row in expected_rows
    ]
    actual_boundary = [
        {
            "item": row.get("item"),
            "source": row.get("source"),
            "status": row.get("status"),
            "token_sha256": row.get("token_sha256"),
        }
        for row in rows
    ]
    return bool(
        _looks_like_sha256(token_sha256)
        and expected_source
        and len(rows) == len(expected_rows)
        and actual_boundary == expected_boundary
        and all(row.get("authorizes_resume_gate") is False for row in rows)
        and all(row.get("authorizes_next_step") is False for row in rows)
        and all(row.get("authorizes_risky_work") is False for row in rows)
        and all(row.get("authorizes_approval") is False for row in rows)
        and all(row.get("authorizes_model_call") is False for row in rows)
        and all(row.get("authorizes_tool_execution") is False for row in rows)
        and all(row.get("authorizes_personal_data_read") is False for row in rows)
        and all(row.get("authorizes_external_side_effect") is False for row in rows)
        and all(row.get("authorizes_unreviewed_followthrough") is False for row in rows)
        and all(row.get("reusable_for_future_recovery") is False for row in rows)
        and all(row.get("reusable_for_next_review") is False for row in rows)
    )


def _recovery_token_metadata_ready(
    metadata: dict[str, Any],
    *,
    prefix: str,
    boundary_ready: Callable[[str, list[dict[str, Any]]], bool],
    fresh_token_key: str,
    non_authority_keys: list[str],
) -> bool:
    rows = list(metadata.get(f"{prefix}_boundary_rows") or [])
    token_sha256 = str(metadata.get(f"{prefix}_sha256") or "")
    if metadata.get(f"{prefix}_present") is not _looks_like_sha256(token_sha256):
        return False
    if metadata.get(f"{prefix}_boundary_row_count") != len(rows):
        return False
    recomputed_ready = boundary_ready(token_sha256, rows)
    if metadata.get(f"{prefix}_boundary_ready") is not recomputed_ready:
        return False
    if metadata.get(fresh_token_key) is not True:
        return False
    for key in non_authority_keys:
        if metadata.get(key) is not False:
            return False
    return recomputed_ready


def _recovery_followthrough_token_metadata_ready(
    metadata: dict[str, Any],
    *,
    expected_source: str | None = None,
) -> bool:
    token_metadata_ready = _recovery_token_metadata_ready(
        metadata,
        prefix="recovery_followthrough_token",
        boundary_ready=lambda token_sha256, rows: _recovery_followthrough_token_boundary_ready(
            token_sha256,
            rows,
            expected_source=expected_source,
        ),
        fresh_token_key="next_recovery_followthrough_requires_new_token",
        non_authority_keys=[
            "recovery_followthrough_token_reusable_for_future_recovery",
            "recovery_followthrough_token_authorizes_resume_gate",
            "recovery_followthrough_token_authorizes_next_step",
            "recovery_followthrough_token_authorizes_risky_work",
            "recovery_followthrough_token_authorizes_approval",
            "recovery_followthrough_token_authorizes_model_call",
            "recovery_followthrough_token_authorizes_tool_execution",
            "recovery_followthrough_token_authorizes_personal_data_read",
            "recovery_followthrough_token_authorizes_external_side_effect",
        ],
    )
    if not token_metadata_ready:
        return False
    if (
        "previous_recovery_followthrough_token_reusable_for_next_review" in metadata
        and metadata.get("previous_recovery_followthrough_token_reusable_for_next_review") is not False
    ):
        return False
    token_fields = [
        "objective",
        "reviewed_step",
        "verification",
        "receipt_path",
        "receipt_sha256",
        "receipt_file_sha256",
        "receipt_hash_matches_file",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_file_sha256",
        "checkpoint_hash_matches_file",
        "stop_condition",
    ]
    if all(field in metadata for field in token_fields):
        expected_token = _recovery_followthrough_token_sha256(
            objective=str(metadata.get("objective") or ""),
            reviewed_step=str(metadata.get("reviewed_step") or ""),
            verification=str(metadata.get("verification") or ""),
            receipt_path=str(metadata.get("receipt_path") or ""),
            receipt_sha256=str(metadata.get("receipt_sha256") or ""),
            receipt_file_sha256=str(metadata.get("receipt_file_sha256") or ""),
            receipt_hash_matches_file=bool(metadata.get("receipt_hash_matches_file")),
            checkpoint_path=str(metadata.get("checkpoint_path") or ""),
            checkpoint_sha256=str(metadata.get("checkpoint_sha256") or ""),
            checkpoint_file_sha256=str(metadata.get("checkpoint_file_sha256") or ""),
            checkpoint_hash_matches_file=bool(metadata.get("checkpoint_hash_matches_file")),
            stop_condition=str(metadata.get("stop_condition") or ""),
        )
        if metadata.get("recovery_followthrough_token_sha256") != expected_token:
            return False
    post_step_queue_present = any(
        key in metadata
        for key in [
            "post_step_proof_queue",
            "post_step_proof_queue_count",
            "post_step_next_proof_command",
            "continuation_post_step_proof_queue",
            "continuation_post_step_proof_queue_count",
            "continuation_post_step_next_proof_command",
        ]
    )
    if post_step_queue_present and not _autonomy_post_step_proof_queue_ready(metadata):
        return False
    return True


def _local_safe_recovery_execution_token_metadata_ready(
    metadata: dict[str, Any],
    *,
    expected_source: str | None = None,
) -> bool:
    token_metadata_ready = _recovery_token_metadata_ready(
        metadata,
        prefix="local_safe_recovery_execution_token",
        boundary_ready=lambda token_sha256, rows: _local_safe_recovery_execution_token_boundary_ready(
            token_sha256,
            rows,
            expected_source=expected_source,
        ),
        fresh_token_key="next_recovery_execution_requires_new_local_safe_token",
        non_authority_keys=[
            "local_safe_recovery_execution_token_authorizes_resume_gate",
            "local_safe_recovery_execution_token_authorizes_next_step",
            "local_safe_recovery_execution_token_authorizes_risky_work",
            "local_safe_recovery_execution_token_authorizes_approval",
            "local_safe_recovery_execution_token_authorizes_model_call",
            "local_safe_recovery_execution_token_authorizes_tool_execution",
            "local_safe_recovery_execution_token_authorizes_personal_data_read",
            "local_safe_recovery_execution_token_authorizes_external_side_effect",
            "local_safe_recovery_execution_token_reusable_for_future_recovery",
        ],
    )
    if not token_metadata_ready:
        return False
    if (
        "previous_local_safe_recovery_execution_token_reusable_for_next_review" in metadata
        and metadata.get("previous_local_safe_recovery_execution_token_reusable_for_next_review") is not False
    ):
        return False
    token_fields = [
        "objective",
        "reviewed_step",
        "verification",
        "receipt_sha256",
        "checkpoint_sha256",
        "recovery_followthrough_token_sha256",
        "stop_condition",
    ]
    if all(field in metadata for field in token_fields):
        expected_token = _local_safe_recovery_execution_token_sha256(
            objective=str(metadata.get("objective") or ""),
            reviewed_step=str(metadata.get("reviewed_step") or ""),
            verification=str(metadata.get("verification") or ""),
            receipt_sha256=str(metadata.get("receipt_sha256") or ""),
            checkpoint_sha256=str(metadata.get("checkpoint_sha256") or ""),
            recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
            stop_condition=str(metadata.get("stop_condition") or ""),
        )
        if metadata.get("local_safe_recovery_execution_token_sha256") != expected_token:
            return False
    return True


def _recovery_execution_readiness_token_metadata_ready(
    metadata: dict[str, Any],
    *,
    expected_source: str | None = None,
) -> bool:
    token_metadata_ready = _recovery_token_metadata_ready(
        metadata,
        prefix="recovery_execution_readiness_token",
        boundary_ready=lambda token_sha256, rows: _recovery_execution_readiness_token_boundary_ready(
            token_sha256,
            rows,
            expected_source=expected_source,
        ),
        fresh_token_key="next_recovery_execution_requires_new_readiness_token",
        non_authority_keys=[
            "recovery_execution_readiness_token_authorizes_resume_gate",
            "recovery_execution_readiness_token_authorizes_next_step",
            "recovery_execution_readiness_token_authorizes_risky_work",
            "recovery_execution_readiness_token_authorizes_approval",
            "recovery_execution_readiness_token_authorizes_model_call",
            "recovery_execution_readiness_token_authorizes_tool_execution",
            "recovery_execution_readiness_token_authorizes_personal_data_read",
            "recovery_execution_readiness_token_authorizes_external_side_effect",
            "recovery_execution_readiness_token_authorizes_unreviewed_followthrough",
            "recovery_execution_readiness_token_reusable_for_future_recovery",
        ],
    )
    if not token_metadata_ready:
        return False
    token_fields = [
        "objective",
        "reviewed_step",
        "receipt_sha256",
        "receipt_file_sha256",
        "receipt_hash_matches_file",
        "checkpoint_sha256",
        "checkpoint_file_sha256",
        "checkpoint_hash_matches_file",
        "recovery_step_approval_boundary_token_sha256",
        "recovery_followthrough_token_sha256",
        "local_safe_recovery_execution_token_sha256",
        "recovery_execution_scorecard_rows",
        "recovery_execution_contract_fields",
        "stop_condition",
        "risky_recovery_signals",
        "approval_reference_provided",
    ]
    if all(field in metadata for field in token_fields):
        expected_token = _recovery_execution_readiness_token_sha256(
            objective=str(metadata.get("objective") or ""),
            reviewed_step=str(metadata.get("reviewed_step") or ""),
            verification=str(metadata.get("verification_target") or metadata.get("verification") or ""),
            receipt_sha256=str(metadata.get("receipt_sha256") or ""),
            receipt_file_sha256=str(metadata.get("receipt_file_sha256") or ""),
            receipt_hash_matches_file=bool(metadata.get("receipt_hash_matches_file")),
            checkpoint_sha256=str(metadata.get("checkpoint_sha256") or ""),
            checkpoint_file_sha256=str(metadata.get("checkpoint_file_sha256") or ""),
            checkpoint_hash_matches_file=bool(metadata.get("checkpoint_hash_matches_file")),
            recovery_step_approval_boundary_token_sha256=str(
                metadata.get("recovery_step_approval_boundary_token_sha256") or ""
            ),
            recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
            local_safe_recovery_execution_token_sha256=str(
                metadata.get("local_safe_recovery_execution_token_sha256") or ""
            ),
            recovery_execution_scorecard_rows=list(metadata.get("recovery_execution_scorecard_rows") or []),
            recovery_execution_contract_fields=list(metadata.get("recovery_execution_contract_fields") or []),
            stop_condition=str(metadata.get("stop_condition") or ""),
            risk_signals=list(metadata.get("risky_recovery_signals") or []),
            approval_reference_present=bool(metadata.get("approval_reference_provided")),
        )
        if metadata.get("recovery_execution_readiness_token_sha256") != expected_token:
            return False
    return True


def _carried_recovery_followthrough_token_boundary_ready(
    metadata: dict[str, Any],
    *,
    expected_source: str | None = None,
) -> bool:
    return _recovery_followthrough_token_metadata_ready(
        metadata,
        expected_source=expected_source,
    )


def _carried_local_safe_recovery_execution_token_boundary_ready(
    metadata: dict[str, Any],
    *,
    expected_source: str | None = None,
) -> bool:
    return _local_safe_recovery_execution_token_metadata_ready(
        metadata,
        expected_source=expected_source,
    )


def _carried_recovery_execution_readiness_token_boundary_ready(
    metadata: dict[str, Any],
    *,
    expected_source: str | None = None,
) -> bool:
    return _recovery_execution_readiness_token_metadata_ready(
        metadata,
        expected_source=expected_source,
    )


def _recovery_followthrough_token_sha256(
    *,
    objective: str,
    reviewed_step: str,
    verification: str,
    receipt_path: str,
    receipt_sha256: str,
    receipt_file_sha256: str,
    receipt_hash_matches_file: bool,
    checkpoint_path: str,
    checkpoint_sha256: str,
    checkpoint_file_sha256: str,
    checkpoint_hash_matches_file: bool,
    stop_condition: str,
    authorizes_resume_gate: bool = False,
    authorizes_next_step: bool = False,
    authorizes_risky_work: bool = False,
    authorizes_approval: bool = False,
    authorizes_model_call: bool = False,
    authorizes_tool_execution: bool = False,
    authorizes_personal_data_read: bool = False,
    authorizes_external_side_effect: bool = False,
    authorizes_unreviewed_followthrough: bool = False,
    reusable_for_future_recovery: bool = False,
    reusable_for_next_review: bool = False,
) -> str:
    return _text_sha256(
        "\n".join(
            [
                "checkpoint_recovery_followthrough_v1",
                objective,
                reviewed_step,
                verification,
                receipt_path,
                receipt_sha256,
                receipt_file_sha256,
                "receipt_hash_matches_file" if receipt_hash_matches_file else "receipt_hash_mismatch_file",
                checkpoint_path,
                checkpoint_sha256,
                checkpoint_file_sha256,
                "checkpoint_hash_matches_file" if checkpoint_hash_matches_file else "checkpoint_hash_mismatch_file",
                stop_condition,
                "proof_only_reviewed_recovery_followthrough",
                f"authorizes_resume_gate={authorizes_resume_gate}",
                f"authorizes_next_step={authorizes_next_step}",
                f"authorizes_risky_work={authorizes_risky_work}",
                f"authorizes_approval={authorizes_approval}",
                f"authorizes_model_call={authorizes_model_call}",
                f"authorizes_tool_execution={authorizes_tool_execution}",
                f"authorizes_personal_data_read={authorizes_personal_data_read}",
                f"authorizes_external_side_effect={authorizes_external_side_effect}",
                f"authorizes_unreviewed_followthrough={authorizes_unreviewed_followthrough}",
                f"reusable_for_future_recovery={reusable_for_future_recovery}",
                f"reusable_for_next_review={reusable_for_next_review}",
            ]
        )
    )


def _local_safe_recovery_execution_token_sha256(
    *,
    objective: str,
    reviewed_step: str,
    verification: str,
    receipt_sha256: str,
    checkpoint_sha256: str,
    recovery_followthrough_token_sha256: str,
    stop_condition: str,
    authorizes_resume_gate: bool = False,
    authorizes_next_step: bool = False,
    authorizes_risky_work: bool = False,
    authorizes_approval: bool = False,
    authorizes_model_call: bool = False,
    authorizes_tool_execution: bool = False,
    authorizes_personal_data_read: bool = False,
    authorizes_external_side_effect: bool = False,
    reusable_for_future_recovery: bool = False,
) -> str:
    return _text_sha256(
        "\n".join(
            [
                "local_safe_recovery_execution_v1",
                objective,
                reviewed_step,
                verification,
                receipt_sha256,
                checkpoint_sha256,
                recovery_followthrough_token_sha256,
                stop_condition,
                "proof_only_receipt",
                "not_reusable_for_resume",
                "not_reusable_for_next_step",
                "authorizes_nothing_after_recording",
                f"authorizes_resume_gate={authorizes_resume_gate}",
                f"authorizes_next_step={authorizes_next_step}",
                f"authorizes_risky_work={authorizes_risky_work}",
                f"authorizes_approval={authorizes_approval}",
                f"authorizes_model_call={authorizes_model_call}",
                f"authorizes_tool_execution={authorizes_tool_execution}",
                f"authorizes_personal_data_read={authorizes_personal_data_read}",
                f"authorizes_external_side_effect={authorizes_external_side_effect}",
                f"reusable_for_future_recovery={reusable_for_future_recovery}",
            ]
        )
    )


def _recovery_execution_readiness_token_sha256(
    *,
    objective: str,
    reviewed_step: str,
    verification: str,
    receipt_sha256: str,
    receipt_file_sha256: str,
    receipt_hash_matches_file: bool,
    checkpoint_sha256: str,
    checkpoint_file_sha256: str,
    checkpoint_hash_matches_file: bool,
    recovery_step_approval_boundary_token_sha256: str,
    recovery_followthrough_token_sha256: str,
    local_safe_recovery_execution_token_sha256: str,
    recovery_execution_scorecard_rows: list[dict[str, Any]],
    recovery_execution_contract_fields: list[str],
    stop_condition: str,
    risk_signals: list[str],
    approval_reference_present: bool,
    authorizes_resume_gate: bool = False,
    authorizes_next_step: bool = False,
    authorizes_risky_work: bool = False,
    authorizes_approval: bool = False,
    authorizes_model_call: bool = False,
    authorizes_tool_execution: bool = False,
    authorizes_personal_data_read: bool = False,
    authorizes_external_side_effect: bool = False,
    authorizes_unreviewed_followthrough: bool = False,
    reusable_for_future_recovery: bool = False,
) -> str:
    row_fingerprint = [
        "|".join(
            [
                str(row.get("item") or ""),
                str(row.get("points") or ""),
                str(row.get("max_points") or ""),
                str(_metadata_bool(row.get("ready"))),
                str(_metadata_bool(row.get("required_before_normal_followthrough"))),
                str(_metadata_bool(row.get("authorizes_action_now"))),
                str(_metadata_bool(row.get("authorizes_risky_work"))),
                str(_metadata_bool(row.get("authorizes_unreviewed_followthrough"))),
                str(_metadata_bool(row.get("authorizes_model_call"))),
                str(_metadata_bool(row.get("authorizes_tool_execution"))),
                str(_metadata_bool(row.get("authorizes_personal_data_read"))),
                str(_metadata_bool(row.get("authorizes_external_side_effect"))),
            ]
        )
        for row in recovery_execution_scorecard_rows
    ]
    return _text_sha256(
        "\n".join(
            [
                "recovery_execution_readiness_v1",
                objective,
                reviewed_step,
                verification,
                receipt_sha256,
                receipt_file_sha256,
                "receipt_hash_matches_file" if receipt_hash_matches_file else "receipt_hash_mismatch_file",
                checkpoint_sha256,
                checkpoint_file_sha256,
                "checkpoint_hash_matches_file" if checkpoint_hash_matches_file else "checkpoint_hash_mismatch_file",
                recovery_step_approval_boundary_token_sha256,
                recovery_followthrough_token_sha256,
                local_safe_recovery_execution_token_sha256,
                "\n".join(row_fingerprint),
                ",".join(sorted(str(field) for field in recovery_execution_contract_fields)),
                stop_condition,
                ",".join(sorted(str(signal) for signal in risk_signals)),
                f"approval_reference_present={approval_reference_present}",
                f"authorizes_resume_gate={authorizes_resume_gate}",
                f"authorizes_next_step={authorizes_next_step}",
                f"authorizes_risky_work={authorizes_risky_work}",
                f"authorizes_approval={authorizes_approval}",
                f"authorizes_model_call={authorizes_model_call}",
                f"authorizes_tool_execution={authorizes_tool_execution}",
                f"authorizes_personal_data_read={authorizes_personal_data_read}",
                f"authorizes_external_side_effect={authorizes_external_side_effect}",
                f"authorizes_unreviewed_followthrough={authorizes_unreviewed_followthrough}",
                f"reusable_for_future_recovery={reusable_for_future_recovery}",
            ]
        )
    )


def _autonomy_cycle_ledger_token_sha256(
    *,
    objective: str,
    ledger_state: str,
    stage_rows: list[dict[str, Any]],
    timebox_receipt_sha256: str,
    awake_guard_token_sha256: str,
    supersession_token_sha256: str,
    checkpoint_route_token_sha256: str,
    latest_checkpoint_sha256: str,
    recovery_followthrough_token_sha256: str,
    local_safe_recovery_execution_token_sha256: str,
    one_step_execution_contract_token_sha256: str,
    continuation_review_token_sha256: str,
    prior_cycle_ledger_token_sha256: str,
    proposed_next_step_sha256: str,
    completed_step_sha256: str,
    proposed_verification_sha256: str,
    post_step_verification_sha256: str,
    receipt_file_sha256: str,
    recovery_checkpoint_file_sha256: str,
    post_step_receipt_file_sha256: str,
    post_step_checkpoint_file_sha256: str,
    next_review_start_command: str,
    autonomy_preflight_scorecard_rows: list[dict[str, Any]] | None = None,
    carried_step_closure_readiness_scorecard_rows: list[dict[str, Any]] | None = None,
    carried_recovery_execution_scorecard_rows: list[dict[str, Any]] | None = None,
    carried_next_step_approval_boundary_rows: list[dict[str, Any]] | None = None,
    authorizes_action_now: bool = False,
    authorizes_local_safe_step: bool = False,
    authorizes_risky_work: bool = False,
    authorizes_new_cycle: bool = False,
    authorizes_unreviewed_followthrough: bool = False,
    authorizes_timebox_reuse: bool = False,
    authorizes_checkpoint_reuse: bool = False,
    authorizes_token_reuse: bool = False,
    authorizes_model_call: bool = False,
    authorizes_tool_execution: bool = False,
    authorizes_personal_data_read: bool = False,
    authorizes_external_side_effect: bool = False,
    reusable_for_next_review: bool = False,
    reusable_for_next_cycle: bool = False,
) -> str:
    stage_entries = [
        "|".join(
            [
                str(row.get("stage") or ""),
                str(row.get("state") or ""),
                "ready" if row.get("ready") else "held",
                str(row.get("proof") or ""),
                "auth_action" if row.get("authorizes_action_now") else "no_action",
                "auth_local_safe" if row.get("authorizes_local_safe_step") else "no_local_safe",
                "auth_risky" if row.get("authorizes_risky_work") else "no_risky",
                "auth_followthrough" if row.get("authorizes_unreviewed_followthrough") else "no_followthrough",
                "auth_timebox_reuse" if row.get("authorizes_timebox_reuse") else "no_timebox_reuse",
                "auth_checkpoint_reuse" if row.get("authorizes_checkpoint_reuse") else "no_checkpoint_reuse",
                "auth_token_reuse" if row.get("authorizes_token_reuse") else "no_token_reuse",
                "auth_approval" if row.get("authorizes_approval") else "no_approval",
                "auth_model" if row.get("authorizes_model_call") else "no_model",
                "auth_tool" if row.get("authorizes_tool_execution") else "no_tool",
                "auth_personal" if row.get("authorizes_personal_data_read") else "no_personal",
                "auth_external" if row.get("authorizes_external_side_effect") else "no_external",
                "reusable" if row.get("reusable_for_next_review") else "not_reusable",
                "cycle_reusable" if row.get("reusable_for_next_cycle") else "cycle_not_reusable",
            ]
        )
        for row in stage_rows
    ]
    scorecard_entries = [
        "|".join(
            [
                str(group),
                str(row.get("item") or ""),
                str(row.get("points") or ""),
                str(row.get("max_points") or ""),
                "ready" if row.get("ready") else "held",
                str(row.get("proof") or row.get("evidence") or ""),
                "auth_action" if row.get("authorizes_action") or row.get("authorizes_action_now") else "no_action",
                "auth_risky" if row.get("authorizes_risky_work") else "no_risky",
                "auth_followup" if row.get("authorizes_followup_without_fresh_review") or row.get("authorizes_unreviewed_followthrough") else "no_followup",
                "auth_model" if row.get("authorizes_model_call") else "no_model",
                "auth_tool" if row.get("authorizes_tool_execution") else "no_tool",
                "auth_personal" if row.get("authorizes_personal_data_read") else "no_personal",
                "auth_external" if row.get("authorizes_external_side_effect") else "no_external",
            ]
        )
        for group, rows in [
            ("autonomy_preflight", autonomy_preflight_scorecard_rows or []),
            ("carried_step_closure", carried_step_closure_readiness_scorecard_rows or []),
            ("carried_recovery_execution", carried_recovery_execution_scorecard_rows or []),
        ]
        for row in rows
    ]
    carried_boundary_entries = [
        "|".join(
            [
                str(row.get("item") or ""),
                str(row.get("status") or ""),
                "auth_action" if row.get("authorizes_action_now") else "no_action",
                "auth_risky" if row.get("authorizes_risky_work") else "no_risky",
                "auth_followthrough" if row.get("authorizes_unreviewed_followthrough") else "no_followthrough",
                "auth_approval" if row.get("authorizes_approval") else "no_approval",
                "auth_model" if row.get("authorizes_model_call") else "no_model",
                "auth_tool" if row.get("authorizes_tool_execution") else "no_tool",
                "auth_personal" if row.get("authorizes_personal_data_read") else "no_personal",
                "auth_external" if row.get("authorizes_external_side_effect") else "no_external",
                "reusable" if row.get("reusable_for_next_review") else "not_reusable",
            ]
        )
        for row in (carried_next_step_approval_boundary_rows or [])
    ]
    return _text_sha256(
        "\n".join(
            [
                "autonomy_cycle_ledger_v1",
                objective,
                ledger_state,
                *stage_entries,
                *scorecard_entries,
                *carried_boundary_entries,
                timebox_receipt_sha256,
                awake_guard_token_sha256,
                supersession_token_sha256,
                checkpoint_route_token_sha256,
                latest_checkpoint_sha256,
                recovery_followthrough_token_sha256,
                local_safe_recovery_execution_token_sha256,
                one_step_execution_contract_token_sha256,
                continuation_review_token_sha256,
                prior_cycle_ledger_token_sha256,
                proposed_next_step_sha256,
                completed_step_sha256,
                proposed_verification_sha256,
                post_step_verification_sha256,
                receipt_file_sha256,
                recovery_checkpoint_file_sha256,
                post_step_receipt_file_sha256,
                post_step_checkpoint_file_sha256,
                next_review_start_command,
                "fresh_operator_timebox_required",
                "fresh_checkpoint_required",
                "prior_step_proof_only",
                f"authorizes_action_now={authorizes_action_now}",
                f"authorizes_local_safe_step={authorizes_local_safe_step}",
                f"authorizes_risky_work={authorizes_risky_work}",
                f"authorizes_new_cycle={authorizes_new_cycle}",
                f"authorizes_unreviewed_followthrough={authorizes_unreviewed_followthrough}",
                f"authorizes_timebox_reuse={authorizes_timebox_reuse}",
                f"authorizes_checkpoint_reuse={authorizes_checkpoint_reuse}",
                f"authorizes_token_reuse={authorizes_token_reuse}",
                f"authorizes_model_call={authorizes_model_call}",
                f"authorizes_tool_execution={authorizes_tool_execution}",
                f"authorizes_personal_data_read={authorizes_personal_data_read}",
                f"authorizes_external_side_effect={authorizes_external_side_effect}",
                f"reusable_for_next_review={reusable_for_next_review}",
                f"reusable_for_next_cycle={reusable_for_next_cycle}",
            ]
        )
    )


def _fresh_review_boundary_token_sha256(
    *,
    objective: str,
    ledger_state: str,
    fresh_review_preflight_queue: list[str],
    fresh_review_contract_rows: list[dict[str, Any]],
    autonomy_cycle_ledger_token_sha256: str,
    next_review_start_command: str,
) -> str:
    contract_entries = [
        "|".join(
            [
                str(row.get("item") or ""),
                str(row.get("source") or ""),
                "fresh" if row.get("fresh_required") else "not_fresh",
                "reusable" if row.get("prior_artifact_reusable") else "not_reusable",
                "auth_action" if row.get("authorizes_action_now") else "no_action",
                "auth_local_safe" if row.get("authorizes_local_safe_step") else "no_local_safe",
                "auth_risky" if row.get("authorizes_risky_work") else "no_risky",
                "auth_followthrough" if row.get("authorizes_unreviewed_followthrough") else "no_followthrough",
                "auth_timebox_reuse" if row.get("authorizes_timebox_reuse") else "no_timebox_reuse",
                "auth_checkpoint_reuse" if row.get("authorizes_checkpoint_reuse") else "no_checkpoint_reuse",
                "auth_token_reuse" if row.get("authorizes_token_reuse") else "no_token_reuse",
                "auth_model" if row.get("authorizes_model_call") else "no_model",
                "auth_tool" if row.get("authorizes_tool_execution") else "no_tool",
                "auth_personal" if row.get("authorizes_personal_data_read") else "no_personal",
                "auth_external" if row.get("authorizes_external_side_effect") else "no_external",
            ]
        )
        for row in fresh_review_contract_rows
    ]
    return _text_sha256(
        "\n".join(
            [
                "fresh_review_boundary_v1",
                objective,
                ledger_state,
                *fresh_review_preflight_queue,
                *contract_entries,
                autonomy_cycle_ledger_token_sha256,
                next_review_start_command,
                "proof_only_boundary",
                "requires_fresh_operator_timebox",
                "requires_fresh_checkpoint",
                "requires_fresh_continuation_review",
                "authorizes_nothing",
            ]
        )
    )


def _fresh_review_boundary_token_rows(*, token_sha256: str, source: str) -> list[dict[str, Any]]:
    rows = [
        {
            "item": "fresh_review_boundary_token",
            "source": source,
            "status": "present" if _looks_like_sha256(token_sha256) else "missing",
            "token_sha256": token_sha256,
        },
        {
            "item": "next_continuation_preflight",
            "source": source,
            "status": "fresh_operator_timebox_checkpoint_and_review_required",
            "token_sha256": token_sha256,
        },
        {
            "item": "prior_cycle_reuse_boundary",
            "source": source,
            "status": "prior_cycle_proof_only_not_reusable",
            "token_sha256": token_sha256,
        },
    ]
    for row in rows:
        row.update(
            {
                "fresh_required": True,
                "authorizes_action_now": False,
                "authorizes_local_safe_step": False,
                "authorizes_risky_work": False,
                "authorizes_unreviewed_followthrough": False,
                "authorizes_timebox_reuse": False,
                "authorizes_checkpoint_reuse": False,
                "authorizes_token_reuse": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "reusable_for_next_review": False,
                "reusable_for_next_cycle": False,
            }
        )
    return rows


def _fresh_review_boundary_token_ready(token_sha256: str, rows: list[dict[str, Any]]) -> bool:
    expected_rows = {
        "fresh_review_boundary_token": "present",
        "next_continuation_preflight": "fresh_operator_timebox_checkpoint_and_review_required",
        "prior_cycle_reuse_boundary": "prior_cycle_proof_only_not_reusable",
    }
    if not _looks_like_sha256(token_sha256):
        return False
    if len(rows) != len(expected_rows):
        return False
    if {str(row.get("item") or "") for row in rows} != set(expected_rows):
        return False
    non_authorizing_fields = [
        "authorizes_action_now",
        "authorizes_local_safe_step",
        "authorizes_risky_work",
        "authorizes_unreviewed_followthrough",
        "authorizes_timebox_reuse",
        "authorizes_checkpoint_reuse",
        "authorizes_token_reuse",
        "authorizes_model_call",
        "authorizes_tool_execution",
        "authorizes_personal_data_read",
        "authorizes_external_side_effect",
        "reusable_for_next_review",
        "reusable_for_next_cycle",
    ]
    for row in rows:
        item = str(row.get("item") or "")
        if row.get("status") != expected_rows[item]:
            return False
        if row.get("token_sha256") != token_sha256:
            return False
        if row.get("fresh_required") is not True:
            return False
        if any(row.get(field) is not False for field in non_authorizing_fields):
            return False
    return True


def _fresh_review_boundary_metadata_ready(
    metadata: dict[str, Any],
    *,
    token_prefix: str,
    authority_prefix: str,
    next_fresh_token_key: str,
    expected_source: str = "",
) -> bool:
    token_sha256 = str(metadata.get(f"{token_prefix}_sha256") or "")
    rows = list(metadata.get(f"{token_prefix}_rows") or [])
    if metadata.get(f"{token_prefix}_present") is not True:
        return False
    if metadata.get(f"{token_prefix}_ready") is not True:
        return False
    if metadata.get(f"{token_prefix}_row_count") != len(rows):
        return False
    if metadata.get(next_fresh_token_key) is not True:
        return False
    for field in [
        "authorizes_action_now",
        "authorizes_local_safe_step",
        "authorizes_risky_work",
        "authorizes_unreviewed_followthrough",
        "authorizes_timebox_reuse",
        "authorizes_checkpoint_reuse",
        "authorizes_token_reuse",
        "authorizes_model_call",
        "authorizes_tool_execution",
        "authorizes_personal_data_read",
        "authorizes_external_side_effect",
        "reusable_for_next_review",
        "reusable_for_next_cycle",
    ]:
        if metadata.get(f"{authority_prefix}_{field}") is not False:
            return False
    if expected_source and any(str(row.get("source") or "") != expected_source for row in rows):
        return False
    return _fresh_review_boundary_token_ready(token_sha256, rows)


def _continuation_review_token_metadata_ready(
    metadata: dict[str, Any],
    *,
    reusable_key: str,
) -> bool:
    token_sha256 = str(metadata.get("continuation_review_token_sha256") or "")
    if not _looks_like_sha256(token_sha256):
        return False
    if metadata.get("continuation_review_token_present") is not True:
        return False
    if metadata.get(reusable_key) is not False:
        return False
    for key in [
        "continuation_review_token_reusable_for_next_review",
        "previous_continuation_review_token_reusable_for_next_review",
    ]:
        if metadata.get(key) is not False:
            return False
    if metadata.get("next_review_requires_new_continuation_review_token") is not True:
        return False
    return True


def _next_review_start_command_token_sha256(
    *,
    objective: str,
    ledger_state: str,
    next_review_start_command: str,
    fresh_review_preflight_queue: list[str],
    autonomy_cycle_ledger_token_sha256: str,
    authorizes_action_now: bool = False,
    authorizes_local_safe_step: bool = False,
    authorizes_risky_work: bool = False,
    authorizes_unreviewed_followthrough: bool = False,
    authorizes_timebox_reuse: bool = False,
    authorizes_checkpoint_reuse: bool = False,
    authorizes_token_reuse: bool = False,
    authorizes_model_call: bool = False,
    authorizes_tool_execution: bool = False,
    authorizes_personal_data_read: bool = False,
    authorizes_external_side_effect: bool = False,
    authorizes_approval: bool = False,
    reusable_for_next_review: bool = False,
    reusable_for_next_cycle: bool = False,
) -> str:
    return _text_sha256(
        "\n".join(
            [
                "next_review_start_command_boundary_v1",
                objective,
                ledger_state,
                next_review_start_command,
                *fresh_review_preflight_queue,
                autonomy_cycle_ledger_token_sha256,
                "proof_only_next_review_pointer",
                "fresh_operator_timebox_required_before_execution",
                "fresh_checkpoint_required_before_execution",
                "fresh_continuation_review_required_before_execution",
                f"authorizes_action_now={authorizes_action_now}",
                f"authorizes_local_safe_step={authorizes_local_safe_step}",
                f"authorizes_risky_work={authorizes_risky_work}",
                f"authorizes_unreviewed_followthrough={authorizes_unreviewed_followthrough}",
                f"authorizes_timebox_reuse={authorizes_timebox_reuse}",
                f"authorizes_checkpoint_reuse={authorizes_checkpoint_reuse}",
                f"authorizes_token_reuse={authorizes_token_reuse}",
                f"authorizes_model_call={authorizes_model_call}",
                f"authorizes_tool_execution={authorizes_tool_execution}",
                f"authorizes_personal_data_read={authorizes_personal_data_read}",
                f"authorizes_external_side_effect={authorizes_external_side_effect}",
                f"authorizes_approval={authorizes_approval}",
                f"reusable_for_next_review={reusable_for_next_review}",
                f"reusable_for_next_cycle={reusable_for_next_cycle}",
            ]
        )
    )


def _next_review_start_command_boundary_rows(*, token_sha256: str, source: str) -> list[dict[str, Any]]:
    rows = [
        {
            "item": "next_review_start_command_token",
            "source": source,
            "status": "present" if _looks_like_sha256(token_sha256) else "missing",
            "token_sha256": token_sha256,
        },
        {
            "item": "next_review_pointer_scope",
            "source": source,
            "status": "proof_only_pointer_not_permission",
            "token_sha256": token_sha256,
        },
        {
            "item": "next_review_preflight_required",
            "source": source,
            "status": "fresh_timebox_checkpoint_and_continuation_review_required",
            "token_sha256": token_sha256,
        },
    ]
    for row in rows:
        row.update(
            {
                "authorizes_action_now": False,
                "authorizes_local_safe_step": False,
                "authorizes_risky_work": False,
                "authorizes_unreviewed_followthrough": False,
                "authorizes_timebox_reuse": False,
                "authorizes_checkpoint_reuse": False,
                "authorizes_token_reuse": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "authorizes_approval": False,
                "reusable_for_next_review": False,
                "reusable_for_next_cycle": False,
            }
        )
    return rows


def _next_review_start_command_boundary_ready(token_sha256: str, rows: list[dict[str, Any]]) -> bool:
    if not _looks_like_sha256(token_sha256):
        return False
    expected = {
        "next_review_start_command_token": "present",
        "next_review_pointer_scope": "proof_only_pointer_not_permission",
        "next_review_preflight_required": "fresh_timebox_checkpoint_and_continuation_review_required",
    }
    if len(rows) != len(expected):
        return False
    if {str(row.get("item") or "") for row in rows} != set(expected):
        return False
    for row in rows:
        item = str(row.get("item") or "")
        if row.get("status") != expected.get(item) or str(row.get("token_sha256") or "") != token_sha256:
            return False
        for key in (
            "authorizes_action_now",
            "authorizes_local_safe_step",
            "authorizes_risky_work",
            "authorizes_unreviewed_followthrough",
            "authorizes_timebox_reuse",
            "authorizes_checkpoint_reuse",
            "authorizes_token_reuse",
            "authorizes_model_call",
            "authorizes_tool_execution",
            "authorizes_personal_data_read",
            "authorizes_external_side_effect",
            "authorizes_approval",
            "reusable_for_next_review",
            "reusable_for_next_cycle",
        ):
            if row.get(key) is not False:
                return False
    return True


def _next_review_start_command_metadata_ready(
    metadata: dict[str, Any],
    *,
    expected_source: str = "",
) -> bool:
    token_sha256 = str(metadata.get("next_review_start_command_token_sha256") or "")
    rows = list(metadata.get("next_review_start_command_boundary_rows") or [])
    if metadata.get("next_review_start_command_token_present") is not True:
        return False
    if metadata.get("next_review_start_command_boundary_ready") is not True:
        return False
    if metadata.get("next_review_start_command_boundary_row_count") != len(rows):
        return False
    if metadata.get("next_review_start_command_requires_fresh_preflight") is not True:
        return False
    for field in [
        "authorizes_action_now",
        "authorizes_local_safe_step",
        "authorizes_risky_work",
        "authorizes_unreviewed_followthrough",
        "authorizes_timebox_reuse",
        "authorizes_checkpoint_reuse",
        "authorizes_token_reuse",
        "authorizes_model_call",
        "authorizes_tool_execution",
        "authorizes_personal_data_read",
        "authorizes_external_side_effect",
        "authorizes_approval",
        "reusable_for_next_review",
        "reusable_for_next_cycle",
    ]:
        if metadata.get(f"next_review_start_command_{field}") is not False:
            return False
    if expected_source and any(str(row.get("source") or "") != expected_source for row in rows):
        return False
    return _next_review_start_command_boundary_ready(token_sha256, rows)


def _fresh_continuation_review_boundary_token_sha256(
    *,
    objective: str,
    closure_state: str,
    next_safe_command: str,
    continuation_review_token_sha256: str,
    fresh_continuation_review_contract_rows: list[dict[str, Any]],
) -> str:
    contract_entries = [
        "|".join(
            [
                str(row.get("item") or ""),
                str(row.get("source") or ""),
                "fresh" if row.get("fresh_required") else "not_fresh",
                "reusable" if row.get("prior_artifact_reusable") else "not_reusable",
                "auth_action" if row.get("authorizes_action_now") else "no_action",
                "auth_risky" if row.get("authorizes_risky_work") else "no_risky",
                "auth_model" if row.get("authorizes_model_call") else "no_model",
                "auth_tool" if row.get("authorizes_tool_execution") else "no_tool",
                "auth_personal" if row.get("authorizes_personal_data_read") else "no_personal",
                "auth_external" if row.get("authorizes_external_side_effect") else "no_external",
                "auth_approval" if row.get("authorizes_approval") else "no_approval",
            ]
        )
        for row in fresh_continuation_review_contract_rows
    ]
    return _text_sha256(
        "\n".join(
            [
                "fresh_continuation_review_boundary_v1",
                objective,
                closure_state,
                next_safe_command,
                continuation_review_token_sha256,
                *contract_entries,
                "proof_only_boundary",
                "requires_fresh_operator_timebox",
                "requires_fresh_checkpoint",
                "requires_fresh_recovery_cockpit",
                "requires_fresh_continuation_review_token",
                "authorizes_nothing",
            ]
        )
    )


def _autonomy_step_closure_receipt_token_sha256(
    *,
    objective: str,
    closure_state: str,
    completed_step_sha256: str,
    post_step_verification_sha256: str,
    post_step_receipt_sha256: str,
    post_step_receipt_file_sha256: str,
    post_step_checkpoint_sha256: str,
    post_step_checkpoint_file_sha256: str,
    execution_health: str,
    execution_audit: str,
    after_action_learning: str,
    fresh_continuation_review_boundary_token_sha256: str,
    step_closure_scorecard_rows: list[dict[str, Any]],
) -> str:
    scorecard_entries = [
        "|".join(
            [
                str(row.get("item") or ""),
                str(row.get("points") or ""),
                str(row.get("max_points") or ""),
                "ready" if row.get("ready") else "held",
                "auth_action" if row.get("authorizes_action") else "no_action",
                "auth_risky" if row.get("authorizes_risky_work") else "no_risky",
                "auth_followup" if row.get("authorizes_followup_without_fresh_review") else "no_followup",
            ]
        )
        for row in step_closure_scorecard_rows
    ]
    return _text_sha256(
        "\n".join(
            [
                "autonomy_step_closure_receipt_v1",
                objective,
                closure_state,
                completed_step_sha256,
                post_step_verification_sha256,
                post_step_receipt_sha256,
                post_step_receipt_file_sha256,
                post_step_checkpoint_sha256,
                post_step_checkpoint_file_sha256,
                _text_sha256(execution_health),
                _text_sha256(execution_audit),
                _text_sha256(after_action_learning),
                fresh_continuation_review_boundary_token_sha256,
                *scorecard_entries,
                "proof_only_boundary",
                "authorizes_action_now=False",
                "authorizes_local_safe_step=False",
                "authorizes_risky_work=False",
                "authorizes_new_cycle=False",
                "authorizes_unreviewed_followthrough=False",
                "authorizes_model_call=False",
                "authorizes_tool_execution=False",
                "authorizes_personal_data_read=False",
                "authorizes_external_side_effect=False",
                "reusable_for_next_review=False",
            ]
        )
    )


def _autonomy_step_closure_receipt_boundary_rows(*, token_sha256: str, source: str) -> list[dict[str, Any]]:
    rows = [
        {
            "item": "step_closure_receipt_token",
            "source": source,
            "status": "present" if _looks_like_sha256(token_sha256) else "missing",
            "token_sha256": token_sha256,
        },
        {
            "item": "post_step_artifact_binding",
            "source": source,
            "status": "receipt_checkpoint_and_verification_bound",
            "token_sha256": token_sha256,
        },
        {
            "item": "next_continuation_review_boundary",
            "source": source,
            "status": "fresh_review_required_before_followup",
            "token_sha256": token_sha256,
        },
    ]
    for row in rows:
        row.update(
            {
                "authorizes_action_now": False,
                "authorizes_local_safe_step": False,
                "authorizes_risky_work": False,
                "authorizes_new_cycle": False,
                "authorizes_unreviewed_followthrough": False,
                "authorizes_timebox_reuse": False,
                "authorizes_checkpoint_reuse": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "reusable_for_next_review": False,
                "reusable_for_next_cycle": False,
            }
        )
    return rows


def _autonomy_step_closure_receipt_boundary_ready(token_sha256: str, rows: list[dict[str, Any]]) -> bool:
    if not _looks_like_sha256(token_sha256):
        return False
    expected = {
        "step_closure_receipt_token": "present",
        "post_step_artifact_binding": "receipt_checkpoint_and_verification_bound",
        "next_continuation_review_boundary": "fresh_review_required_before_followup",
    }
    if len(rows) != len(expected):
        return False
    if {str(row.get("item") or "") for row in rows} != set(expected):
        return False
    for row in rows:
        item = str(row.get("item") or "")
        if row.get("status") != expected.get(item) or str(row.get("token_sha256") or "") != token_sha256:
            return False
        for key in (
            "authorizes_action_now",
            "authorizes_local_safe_step",
            "authorizes_risky_work",
            "authorizes_new_cycle",
            "authorizes_unreviewed_followthrough",
            "authorizes_timebox_reuse",
            "authorizes_checkpoint_reuse",
            "authorizes_model_call",
            "authorizes_tool_execution",
            "authorizes_personal_data_read",
            "authorizes_external_side_effect",
            "reusable_for_next_review",
            "reusable_for_next_cycle",
        ):
            if row.get(key) is not False:
                return False
    return True


def _step_closure_receipt_metadata_ready(
    metadata: dict[str, Any],
    *,
    token_prefix: str,
    boundary_prefix: str,
    authority_prefix: str,
    expected_source: str = "",
) -> bool:
    token_sha256 = str(metadata.get(f"{token_prefix}_sha256") or "")
    rows = list(metadata.get(f"{boundary_prefix}_rows") or [])
    if metadata.get(f"{token_prefix}_present") is not True:
        return False
    if metadata.get(f"{boundary_prefix}_row_count") != len(rows):
        return False
    if metadata.get(f"{boundary_prefix}_ready") is not True:
        return False
    if metadata.get("next_review_requires_new_step_closure_receipt_token") is not True:
        return False
    for field in [
        "authorizes_action_now",
        "authorizes_local_safe_step",
        "authorizes_risky_work",
        "authorizes_new_cycle",
        "authorizes_unreviewed_followthrough",
        "authorizes_timebox_reuse",
        "authorizes_checkpoint_reuse",
        "authorizes_model_call",
        "authorizes_tool_execution",
        "authorizes_personal_data_read",
        "authorizes_external_side_effect",
        "reusable_for_next_review",
        "reusable_for_next_cycle",
    ]:
        if metadata.get(f"{authority_prefix}_{field}") is not False:
            return False
    if expected_source and any(str(row.get("source") or "") != expected_source for row in rows):
        return False
    return _autonomy_step_closure_receipt_boundary_ready(token_sha256, rows)


def _carried_step_closure_receipt_boundary_ready(metadata: dict[str, Any]) -> bool:
    if (
        "carried_step_closure_receipt_token_sha256" in metadata
        or "carried_step_closure_receipt_boundary_rows" in metadata
    ):
        return _step_closure_receipt_metadata_ready(
            metadata,
            token_prefix="carried_step_closure_receipt_token",
            boundary_prefix="carried_step_closure_receipt_boundary",
            authority_prefix="carried_step_closure_receipt",
        )
    return _step_closure_receipt_metadata_ready(
        metadata,
        token_prefix="step_closure_receipt_token",
        boundary_prefix="step_closure_receipt_boundary",
        authority_prefix="step_closure_receipt",
    )


def _one_step_execution_contract_token_sha256(
    *,
    objective: str,
    continuation_state: str,
    resume_gate_state: str,
    timebox_receipt_sha256: str,
    awake_guard_token_sha256: str,
    supersession_token_sha256: str,
    checkpoint_route_token_sha256: str,
    recovery_followthrough_token_sha256: str,
    local_safe_recovery_execution_token_sha256: str,
    prior_cycle_ledger_token_sha256: str,
    continuation_review_token_sha256: str,
    proposed_next_step_sha256: str,
    proposed_verification_sha256: str,
    risk_signals: list[str],
    one_step_execution_contract_rows: list[dict[str, Any]],
    timebox_review_contract_rows: list[dict[str, Any]] | None = None,
    post_step_proof_queue: list[str],
    authorizes_action_now: bool = False,
    authorizes_risky_work: bool = False,
    authorizes_unreviewed_followthrough: bool = False,
    authorizes_batching: bool = False,
    reusable_for_next_step: bool = False,
) -> str:
    row_entries = [
        "|".join(
            [
                str(row.get("item") or ""),
                str(row.get("status") or ""),
                "required" if row.get("required") else "optional",
                "one_local_safe" if row.get("authorizes_only_one_local_safe_step") else "no_local_safe",
                "auth_risky" if row.get("authorizes_risky_work") else "no_risky",
                "reusable" if row.get("reusable_for_next_step") else "not_reusable",
            ]
        )
        for row in one_step_execution_contract_rows
    ]
    timebox_row_entries = [
        "|".join(
            [
                str(row.get("item") or ""),
                str(row.get("source") or ""),
                str(row.get("state") or ""),
                "fresh" if row.get("fresh_required") else "not_fresh",
                "auth_execution" if row.get("authorizes_execution") else "no_execution",
                "auth_local_safe" if row.get("authorizes_local_safe_step") else "no_local_safe",
                "auth_risky" if row.get("authorizes_risky_work") else "no_risky",
                "auth_approval" if row.get("authorizes_approval") else "no_approval",
                "auth_timebox_reuse" if row.get("authorizes_timebox_reuse") else "no_timebox_reuse",
                "reusable" if row.get("reusable_for_next_step") else "not_reusable",
            ]
        )
        for row in (timebox_review_contract_rows or [])
    ]
    return _text_sha256(
        "\n".join(
            [
                "one_step_execution_contract_boundary_v1",
                objective,
                continuation_state,
                resume_gate_state,
                timebox_receipt_sha256,
                awake_guard_token_sha256,
                supersession_token_sha256,
                checkpoint_route_token_sha256,
                recovery_followthrough_token_sha256,
                local_safe_recovery_execution_token_sha256,
                prior_cycle_ledger_token_sha256,
                continuation_review_token_sha256,
                proposed_next_step_sha256,
                proposed_verification_sha256,
                ",".join(sorted(str(signal) for signal in risk_signals)),
                *row_entries,
                *timebox_row_entries,
                *post_step_proof_queue,
                "proof_only_pre_step_contract",
                "binds_awake_guard_supersession_and_timebox_review_contract",
                f"authorizes_action_now={authorizes_action_now}",
                f"authorizes_risky_work={authorizes_risky_work}",
                f"authorizes_unreviewed_followthrough={authorizes_unreviewed_followthrough}",
                f"authorizes_batching={authorizes_batching}",
                f"reusable_for_next_step={reusable_for_next_step}",
            ]
        )
    )


_ONE_STEP_EXECUTION_CONTRACT_EXPECTED_STATUSES = {
    "ready_resume_gate": "ready",
    "exact_next_step": "ready",
    "post_step_verification_target": "ready",
    "stop_condition": "ready",
    "risky_work_approval_boundary": "ready",
    "fresh_post_step_closure": "fresh_required_after_step",
}


def _one_step_execution_contract_rows_ready(rows: list[dict[str, Any]]) -> bool:
    if len(rows) != len(_ONE_STEP_EXECUTION_CONTRACT_EXPECTED_STATUSES):
        return False
    by_item = {str(row.get("item") or ""): row for row in rows}
    if set(by_item) != set(_ONE_STEP_EXECUTION_CONTRACT_EXPECTED_STATUSES):
        return False
    for item, expected_status in _ONE_STEP_EXECUTION_CONTRACT_EXPECTED_STATUSES.items():
        row = by_item[item]
        if row.get("status") != expected_status:
            return False
        if row.get("required") is not True:
            return False
        if row.get("authorizes_only_one_local_safe_step") is not True:
            return False
        if row.get("authorizes_risky_work") is not False:
            return False
        if row.get("reusable_for_next_step") is not False:
            return False
    return True


def _one_step_execution_contract_ready(
    *,
    objective: str,
    continuation_state: str,
    resume_gate_state: str,
    continuation_ready: bool,
    one_step_execution_contract_rows: list[dict[str, Any]],
    one_step_execution_contract_token_sha256: str,
    timebox_receipt_sha256: str,
    awake_guard_token_sha256: str,
    supersession_token_sha256: str,
    checkpoint_route_token_sha256: str,
    recovery_followthrough_token_sha256: str,
    local_safe_recovery_execution_token_sha256: str,
    continuation_review_token_sha256: str,
    proposed_next_step_sha256: str,
    proposed_verification_sha256: str,
    prior_cycle_ledger_token_sha256: str = "",
    risk_signals: list[str] | None = None,
    timebox_review_contract_ready: bool = False,
    timebox_review_contract_rows: list[dict[str, Any]] | None = None,
    post_step_proof_queue: list[str] | None = None,
) -> bool:
    if continuation_ready is not True or timebox_review_contract_ready is not True:
        return False
    required_hashes = [
        one_step_execution_contract_token_sha256,
        timebox_receipt_sha256,
        awake_guard_token_sha256,
        supersession_token_sha256,
        checkpoint_route_token_sha256,
        recovery_followthrough_token_sha256,
        local_safe_recovery_execution_token_sha256,
        continuation_review_token_sha256,
        proposed_next_step_sha256,
        proposed_verification_sha256,
    ]
    expected_token = _one_step_execution_contract_token_sha256(
        objective=objective,
        continuation_state=continuation_state,
        resume_gate_state=resume_gate_state,
        timebox_receipt_sha256=timebox_receipt_sha256,
        awake_guard_token_sha256=awake_guard_token_sha256,
        supersession_token_sha256=supersession_token_sha256,
        checkpoint_route_token_sha256=checkpoint_route_token_sha256,
        recovery_followthrough_token_sha256=recovery_followthrough_token_sha256,
        local_safe_recovery_execution_token_sha256=local_safe_recovery_execution_token_sha256,
        prior_cycle_ledger_token_sha256=prior_cycle_ledger_token_sha256,
        continuation_review_token_sha256=continuation_review_token_sha256,
        proposed_next_step_sha256=proposed_next_step_sha256,
        proposed_verification_sha256=proposed_verification_sha256,
        risk_signals=risk_signals or [],
        one_step_execution_contract_rows=one_step_execution_contract_rows,
        timebox_review_contract_rows=timebox_review_contract_rows or [],
        post_step_proof_queue=post_step_proof_queue or [],
    )
    return bool(
        _one_step_execution_contract_rows_ready(one_step_execution_contract_rows)
        and all(_looks_like_sha256(value) for value in required_hashes)
        and (not prior_cycle_ledger_token_sha256 or _looks_like_sha256(prior_cycle_ledger_token_sha256))
        and one_step_execution_contract_token_sha256 == expected_token
        and not (risk_signals or [])
        and bool(post_step_proof_queue)
    )


def _carried_one_step_execution_contract_ready(metadata: dict[str, Any]) -> bool:
    rows = list(metadata.get("one_step_execution_contract_rows") or [])
    if metadata.get("one_step_execution_contract_token_present") is not True:
        return False
    if metadata.get("one_step_execution_contract_token_as_prior_proof") is not True:
        return False
    if metadata.get("one_step_execution_contract_ready") is not True:
        return False
    for flag in [
        "one_step_execution_contract_binds_awake_guard",
        "one_step_execution_contract_binds_operator_supersession",
        "one_step_execution_contract_binds_timebox_review_contract",
        "one_step_execution_contract_all_local_safe_step_limited",
        "one_step_execution_contract_all_non_reusable",
        "one_step_execution_contract_all_risky_work_gated",
        "one_step_execution_contract_requires_fresh_closure",
    ]:
        if metadata.get(flag) is not True:
            return False
    if metadata.get("one_step_execution_contract_row_count") != len(rows):
        return False
    if metadata.get("next_step_requires_new_one_step_execution_contract_token") is not True:
        return False
    for flag in [
        "one_step_execution_contract_token_authorizes_action_now",
        "one_step_execution_contract_token_authorizes_risky_work",
        "one_step_execution_contract_token_authorizes_unreviewed_followthrough",
        "one_step_execution_contract_token_authorizes_batching",
        "one_step_execution_contract_token_reusable_for_next_step",
    ]:
        if metadata.get(flag) is not False:
            return False
    return _one_step_execution_contract_ready(
        objective=str(metadata.get("objective") or ""),
        continuation_state=str(metadata.get("continuation_state") or ""),
        resume_gate_state=str(metadata.get("resume_gate_state") or ""),
        continuation_ready=metadata.get("one_step_execution_contract_ready"),
        one_step_execution_contract_rows=rows,
        one_step_execution_contract_token_sha256=str(metadata.get("one_step_execution_contract_token_sha256") or ""),
        timebox_receipt_sha256=str(metadata.get("timebox_receipt_sha256") or ""),
        awake_guard_token_sha256=str(metadata.get("awake_guard_token_sha256") or ""),
        supersession_token_sha256=str(metadata.get("supersession_token_sha256") or ""),
        checkpoint_route_token_sha256=str(metadata.get("checkpoint_route_token_sha256") or ""),
        recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
        local_safe_recovery_execution_token_sha256=str(metadata.get("local_safe_recovery_execution_token_sha256") or ""),
        prior_cycle_ledger_token_sha256=str(metadata.get("prior_cycle_ledger_token_sha256") or ""),
        continuation_review_token_sha256=str(metadata.get("continuation_review_token_sha256") or ""),
        proposed_next_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
        proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
        risk_signals=[str(signal) for signal in (metadata.get("proposed_next_step_risk_signals") or [])],
        timebox_review_contract_ready=metadata.get("timebox_review_contract_ready"),
        timebox_review_contract_rows=list(metadata.get("timebox_review_contract_rows") or []),
        post_step_proof_queue=[
            str(command)
            for command in (metadata.get("post_step_proof_queue") or metadata.get("continuation_post_step_proof_queue") or [])
        ],
    ) and _autonomy_post_step_proof_queue_ready(metadata)


def _prior_cycle_ledger_token_boundary_rows(*, token_sha256: str, source: str) -> list[dict[str, Any]]:
    rows = [
        {
            "item": "prior_cycle_ledger_token",
            "source": source,
            "status": "present" if _looks_like_sha256(token_sha256) else "not_supplied",
            "token_sha256": token_sha256,
        },
        {
            "item": "current_review_scope",
            "source": source,
            "status": "proof_only_for_current_review",
            "token_sha256": token_sha256,
        },
        {
            "item": "next_cycle_ledger_review",
            "source": source,
            "status": "fresh_cycle_ledger_token_required",
            "token_sha256": token_sha256,
        },
    ]
    for row in rows:
        row.update(
            {
                "authorizes_action_now": False,
                "authorizes_local_safe_step": False,
                "authorizes_risky_work": False,
                "authorizes_post_step_closure": False,
                "authorizes_new_action": False,
                "authorizes_unreviewed_followthrough": False,
                "authorizes_timebox_reuse": False,
                "authorizes_checkpoint_reuse": False,
                "authorizes_token_reuse": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "reusable_for_this_review": False,
                "reusable_for_this_closure": False,
                "reusable_for_this_cycle": False,
                "reusable_for_next_review": False,
            }
        )
    return rows


def _prior_cycle_ledger_token_boundary_ready(token_sha256: str, rows: list[dict[str, Any]]) -> bool:
    expected_token_status = "present" if _looks_like_sha256(token_sha256) else "not_supplied"
    if token_sha256 and not _looks_like_sha256(token_sha256):
        return False
    expected = {
        "prior_cycle_ledger_token": expected_token_status,
        "current_review_scope": "proof_only_for_current_review",
        "next_cycle_ledger_review": "fresh_cycle_ledger_token_required",
    }
    if len(rows) != len(expected):
        return False
    if {str(row.get("item") or "") for row in rows} != set(expected):
        return False
    for row in rows:
        item = str(row.get("item") or "")
        if row.get("status") != expected.get(item) or str(row.get("token_sha256") or "") != token_sha256:
            return False
        for key in (
            "authorizes_action_now",
            "authorizes_local_safe_step",
            "authorizes_risky_work",
            "authorizes_post_step_closure",
            "authorizes_new_action",
            "authorizes_unreviewed_followthrough",
            "authorizes_timebox_reuse",
            "authorizes_checkpoint_reuse",
            "authorizes_token_reuse",
            "authorizes_model_call",
            "authorizes_tool_execution",
            "authorizes_personal_data_read",
            "authorizes_external_side_effect",
            "reusable_for_this_review",
            "reusable_for_this_closure",
            "reusable_for_this_cycle",
            "reusable_for_next_review",
        ):
            if row.get(key) is not False:
                return False
    return True


def _prior_cycle_ledger_token_metadata_ready(
    metadata: dict[str, Any],
    *,
    reusable_key: str,
    authority_action_key: str,
    expected_source: str = "",
) -> bool:
    token_sha256 = str(metadata.get("prior_cycle_ledger_token_sha256") or "")
    rows = list(metadata.get("prior_cycle_ledger_token_boundary_rows") or [])
    if metadata.get("prior_cycle_ledger_token_present") is not _looks_like_sha256(token_sha256):
        return False
    if metadata.get("prior_cycle_ledger_token_boundary_row_count") != len(rows):
        return False
    if metadata.get("prior_cycle_ledger_token_boundary_ready") is not True:
        return False
    for key in [
        reusable_key,
        authority_action_key,
        "prior_cycle_ledger_token_reusable_for_this_review",
        "prior_cycle_ledger_token_reusable_for_this_closure",
        "prior_cycle_ledger_token_reusable_for_this_cycle",
        "prior_cycle_ledger_proof_authorizes_action_now",
        "prior_cycle_ledger_proof_authorizes_post_step_closure",
        "prior_cycle_ledger_proof_authorizes_new_action",
        "prior_cycle_ledger_proof_authorizes_model_call",
        "prior_cycle_ledger_proof_authorizes_tool_execution",
        "prior_cycle_ledger_proof_authorizes_personal_data_read",
        "prior_cycle_ledger_proof_authorizes_external_side_effect",
    ]:
        if metadata.get(key) is not False:
            return False
    if expected_source and any(str(row.get("source") or "") != expected_source for row in rows):
        return False
    return _prior_cycle_ledger_token_boundary_ready(token_sha256, rows)


def _autonomy_cycle_ledger_token_boundary_rows(*, token_sha256: str, source: str) -> list[dict[str, Any]]:
    rows = [
        {
            "item": "autonomy_cycle_ledger_token",
            "source": source,
            "status": "present" if _looks_like_sha256(token_sha256) else "missing",
            "token_sha256": token_sha256,
        },
        {
            "item": "current_cycle_closure_scope",
            "source": source,
            "status": "proof_only_for_completed_cycle",
            "token_sha256": token_sha256,
        },
        {
            "item": "next_cycle_review_boundary",
            "source": source,
            "status": "fresh_cycle_ledger_token_required",
            "token_sha256": token_sha256,
        },
    ]
    for row in rows:
        row.update(
            {
                "authorizes_action_now": False,
                "authorizes_local_safe_step": False,
                "authorizes_risky_work": False,
                "authorizes_new_cycle": False,
                "authorizes_unreviewed_followthrough": False,
                "authorizes_timebox_reuse": False,
                "authorizes_checkpoint_reuse": False,
                "authorizes_token_reuse": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "reusable_for_next_review": False,
                "reusable_for_next_cycle": False,
            }
        )
    return rows


def _autonomy_cycle_ledger_token_boundary_ready(token_sha256: str, rows: list[dict[str, Any]]) -> bool:
    expected = {
        "autonomy_cycle_ledger_token": "present",
        "current_cycle_closure_scope": "proof_only_for_completed_cycle",
        "next_cycle_review_boundary": "fresh_cycle_ledger_token_required",
    }
    if not _looks_like_sha256(token_sha256) or len(rows) != len(expected):
        return False
    if {str(row.get("item") or "") for row in rows} != set(expected):
        return False
    for row in rows:
        item = str(row.get("item") or "")
        if row.get("status") != expected.get(item) or row.get("token_sha256") != token_sha256:
            return False
        for key in (
            "authorizes_action_now",
            "authorizes_local_safe_step",
            "authorizes_risky_work",
            "authorizes_new_cycle",
            "authorizes_unreviewed_followthrough",
            "authorizes_timebox_reuse",
            "authorizes_checkpoint_reuse",
            "authorizes_token_reuse",
            "authorizes_model_call",
            "authorizes_tool_execution",
            "authorizes_personal_data_read",
            "authorizes_external_side_effect",
            "reusable_for_next_review",
            "reusable_for_next_cycle",
        ):
            if row.get(key) is not False:
                return False
    return True


def _autonomy_cycle_ledger_token_metadata_ready(
    metadata: dict[str, Any],
    *,
    expected_source: str = "",
) -> bool:
    token_sha256 = str(metadata.get("autonomy_cycle_ledger_token_sha256") or "")
    rows = list(metadata.get("autonomy_cycle_ledger_token_boundary_rows") or [])
    if metadata.get("autonomy_cycle_ledger_token_present") is not True:
        return False
    if metadata.get("autonomy_cycle_ledger_token_boundary_ready") is not True:
        return False
    if metadata.get("autonomy_cycle_ledger_token_boundary_row_count") != len(rows):
        return False
    if metadata.get("next_review_requires_new_cycle_ledger_token") is not True:
        return False
    if metadata.get("previous_cycle_ledger_token_reusable_for_next_review") is not False:
        return False
    for field in [
        "authorizes_action_now",
        "authorizes_local_safe_step",
        "authorizes_risky_work",
        "authorizes_new_cycle",
        "authorizes_unreviewed_followthrough",
        "authorizes_timebox_reuse",
        "authorizes_checkpoint_reuse",
        "authorizes_token_reuse",
        "authorizes_model_call",
        "authorizes_tool_execution",
        "authorizes_personal_data_read",
        "authorizes_external_side_effect",
    ]:
        if metadata.get(f"autonomy_cycle_ledger_token_{field}") is not False:
            return False
    if expected_source and any(str(row.get("source") or "") != expected_source for row in rows):
        return False
    return _autonomy_cycle_ledger_token_boundary_ready(token_sha256, rows)


def _autonomy_cycle_ledger_ready_from_metadata(metadata: dict[str, Any]) -> bool:
    if metadata.get("cycle_state") != "AUTONOMY_CYCLE_LEDGER_READY_FOR_FRESH_REVIEW":
        return False
    if metadata.get("ready_for_fresh_next_continuation_review") is not True:
        return False
    if metadata.get("ready_for_next_continuation_review") is not True:
        return False
    if metadata.get("missing_blockers") != [] or metadata.get("missing_blocker_count") != 0:
        return False

    stage_rows = list(metadata.get("stage_rows") or [])
    if metadata.get("stage_count") != len(stage_rows) or not _autonomy_cycle_stage_rows_ready(stage_rows):
        return False
    if metadata.get("autonomy_cycle_stage_rows_ready") is not True:
        return False

    preflight_rows = list(metadata.get("autonomy_preflight_scorecard_rows") or [])
    if metadata.get("autonomy_preflight_scorecard_row_count") != len(preflight_rows):
        return False
    if metadata.get("autonomy_preflight_score") != 100 or metadata.get("autonomy_preflight_max_score") != 100:
        return False
    if metadata.get("autonomy_preflight_required_rows_ready") is not True:
        return False
    if metadata.get("autonomy_preflight_scorecard_ready") is not True:
        return False
    if not _autonomy_cycle_preflight_scorecard_ready(preflight_rows):
        return False

    closure_metadata = metadata.get("step_closure_metadata")
    if not isinstance(closure_metadata, dict) or not _autonomy_step_closure_ready(closure_metadata):
        return False
    if closure_metadata.get("step_closure_ready_contract_ready") is not True:
        return False
    if metadata.get("carried_step_closure_readiness_as_prior_proof") is not True:
        return False
    closure_rows = list(metadata.get("carried_step_closure_readiness_scorecard_rows") or [])
    if metadata.get("carried_step_closure_readiness_scorecard_row_count") != len(closure_rows):
        return False
    if metadata.get("carried_step_closure_readiness_score") != 100 or metadata.get("carried_step_closure_readiness_max_score") != 100:
        return False
    if metadata.get("carried_step_closure_readiness_required_rows_ready") is not True:
        return False
    if metadata.get("carried_step_closure_readiness_scorecard_ready") is not True:
        return False
    if not _autonomy_step_closure_readiness_scorecard_ready(closure_rows):
        return False
    if not _step_closure_receipt_metadata_ready(
        metadata,
        token_prefix="carried_step_closure_receipt_token",
        boundary_prefix="carried_step_closure_receipt_boundary",
        authority_prefix="carried_step_closure_receipt",
        expected_source="autonomy_step_closure",
    ):
        return False

    recovery_rows = list(metadata.get("carried_recovery_execution_scorecard_rows") or [])
    if metadata.get("carried_recovery_execution_scorecard_row_count") != len(recovery_rows):
        return False
    if metadata.get("carried_recovery_execution_score") != 100 or metadata.get("carried_recovery_execution_max_score") != 100:
        return False
    if metadata.get("carried_recovery_execution_required_rows_ready") is not True:
        return False
    if metadata.get("carried_recovery_execution_scorecard_ready") is not True:
        return False
    if metadata.get("carried_recovery_execution_as_prior_proof") is not True:
        return False
    if not _recovery_execution_readiness_scorecard_ready(recovery_rows):
        return False

    if not _autonomy_cycle_ledger_token_metadata_ready(
        metadata,
        expected_source="autonomy_cycle_ledger",
    ):
        return False
    if not _fresh_review_boundary_metadata_ready(
        metadata,
        token_prefix="fresh_review_boundary_token",
        authority_prefix="fresh_review_boundary_token",
        next_fresh_token_key="next_review_requires_new_fresh_review_boundary_token",
        expected_source="autonomy_cycle_ledger",
    ):
        return False
    if not _continuation_review_token_metadata_ready(
        metadata,
        reusable_key="previous_continuation_review_token_reusable_for_next_review",
    ):
        return False
    if not _next_review_start_command_metadata_ready(
        metadata,
        expected_source="autonomy_cycle_ledger",
    ):
        return False
    if not _prior_cycle_ledger_token_metadata_ready(
        metadata,
        reusable_key="prior_cycle_ledger_token_reusable_for_this_cycle",
        authority_action_key="prior_cycle_ledger_proof_authorizes_new_action",
        expected_source="autonomy_cycle_ledger",
    ):
        return False
    if not _carried_next_step_approval_boundary_ready_from_metadata(metadata):
        return False

    required_commands = list(metadata.get("required_commands") or [])
    if metadata.get("required_command_count") != len(required_commands):
        return False
    proof_queue = list(metadata.get("proof_queue") or [])
    if metadata.get("proof_queue_count") != len(proof_queue):
        return False
    if not required_commands or proof_queue != required_commands:
        return False
    if metadata.get("next_required_command") != required_commands[0]:
        return False
    if metadata.get("next_proof_command") != required_commands[0]:
        return False
    if not any(str(command).startswith("autonomy cycle ledger:") for command in required_commands):
        return False
    next_review_start_command = str(metadata.get("next_review_start_command") or "")
    next_safe_command = str(metadata.get("next_safe_command") or "")
    expected_next_review_start_command = "autonomy continuation execution: <next reviewed local-safe step>"
    if next_review_start_command != expected_next_review_start_command:
        return False
    if next_safe_command != next_review_start_command:
        return False
    fresh_review_preflight_queue = list(metadata.get("fresh_review_preflight_queue") or [])
    if metadata.get("fresh_review_preflight_queue_count") != len(fresh_review_preflight_queue):
        return False
    if metadata.get("fresh_review_next_preflight_command") != (
        fresh_review_preflight_queue[0] if fresh_review_preflight_queue else ""
    ):
        return False
    if next_review_start_command not in fresh_review_preflight_queue:
        return False

    true_flags = [
        "fresh_review_contract_ready",
        "all_prior_artifacts_non_authorizing",
        "closure_fresh_continuation_review_contract_enforced",
        "timebox_review_contract_ready",
        "awake_guard_token_present",
        "awake_guard_os_wake_lock_boundary_ready",
        "supersession_token_present",
        "supersession_token_boundary_ready",
        "checkpoint_route_token_present",
        "checkpoint_route_boundary_ready",
        "recovery_followthrough_token_present",
        "recovery_followthrough_token_boundary_ready",
        "local_safe_recovery_execution_token_present",
        "local_safe_recovery_execution_token_boundary_ready",
        "recovery_execution_readiness_token_present",
        "recovery_execution_readiness_token_boundary_ready",
        "one_step_execution_contract_ready",
        "one_step_execution_contract_token_as_prior_proof",
        "one_step_execution_contract_binds_awake_guard",
        "one_step_execution_contract_binds_operator_supersession",
        "one_step_execution_contract_binds_timebox_review_contract",
        "one_step_execution_contract_all_local_safe_step_limited",
        "one_step_execution_contract_all_non_reusable",
        "one_step_execution_contract_all_risky_work_gated",
        "one_step_execution_contract_requires_fresh_closure",
        "next_review_requires_full_preflight",
        "next_review_requires_fresh_operator_timebox",
        "next_review_requires_fresh_checkpoint",
        "next_review_requires_new_cycle_ledger_token",
        "next_review_requires_new_fresh_review_boundary_token",
        "next_review_start_command_requires_fresh_preflight",
        "prior_continuation_step_proof_only",
        "recovery_artifact_hashes_present",
        "recovery_artifact_hashes_match_files",
        "completed_step_matches_proposed",
        "post_step_verification_matches_proposed",
        "post_step_artifact_hashes_present",
        "post_step_artifact_hashes_match_files",
        "fresh_continuation_review_boundary_token_present",
        "fresh_continuation_review_boundary_token_ready",
        "continuation_review_token_present",
        "closure_next_continuation_requires_fresh_operator_timebox",
        "closure_next_continuation_requires_fresh_checkpoint",
        "closure_next_continuation_requires_fresh_recovery_cockpit",
        "closure_next_continuation_requires_fresh_local_safe_step",
        "closure_next_continuation_requires_fresh_review_token",
    ]
    for flag in true_flags:
        if metadata.get(flag) is not True:
            return False
    if not _carried_operator_timebox_review_contract_ready(metadata):
        return False
    if not _carried_operator_supersession_token_boundary_ready(metadata):
        return False
    if not _carried_checkpoint_route_boundary_ready(metadata):
        return False
    if not _carried_one_step_execution_contract_ready(metadata):
        return False
    if not _carried_local_safe_recovery_execution_token_boundary_ready(
        metadata,
        expected_source="autonomy_cycle_ledger",
    ):
        return False

    false_flags = [
        "action_allowed_now",
        "executable_tool_action_emitted",
        "can_emit_executable_tool_action",
        "can_auto_execute_now",
        "prior_artifacts_authorize_local_safe_step",
        "prior_artifacts_authorize_risky_work",
        "prior_artifacts_authorize_unreviewed_followthrough",
        "prior_artifacts_authorize_timebox_reuse",
        "prior_artifacts_authorize_checkpoint_reuse",
        "prior_artifacts_authorize_token_reuse",
        "prior_artifacts_authorize_model_call",
        "prior_artifacts_authorize_tool_execution",
        "prior_artifacts_authorize_personal_data_read",
        "prior_artifacts_authorize_external_side_effect",
        "carried_step_closure_readiness_authorizes_action_now",
        "carried_step_closure_readiness_authorizes_risky_work",
        "carried_step_closure_readiness_authorizes_followup_without_fresh_review",
        "carried_step_closure_receipt_authorizes_action_now",
        "carried_step_closure_receipt_authorizes_local_safe_step",
        "carried_step_closure_receipt_authorizes_risky_work",
        "carried_step_closure_receipt_authorizes_new_cycle",
        "carried_step_closure_receipt_authorizes_unreviewed_followthrough",
        "carried_step_closure_receipt_reusable_for_next_review",
        "carried_step_closure_receipt_reusable_for_next_cycle",
        "carried_recovery_execution_authorizes_action_now",
        "carried_recovery_execution_authorizes_risky_work",
        "carried_recovery_execution_authorizes_unreviewed_followthrough",
        "carried_recovery_execution_authorizes_model_call",
        "carried_recovery_execution_authorizes_tool_execution",
        "carried_recovery_execution_authorizes_personal_data_read",
        "carried_recovery_execution_authorizes_external_side_effect",
        "previous_recovery_followthrough_token_reusable_for_next_review",
        "previous_local_safe_recovery_execution_token_reusable_for_next_review",
        "carried_next_step_approval_boundary_authorizes_action_now",
        "carried_next_step_approval_boundary_authorizes_risky_work",
        "carried_next_step_approval_boundary_authorizes_unreviewed_followthrough",
        "carried_next_step_approval_boundary_authorizes_approval",
        "carried_next_step_approval_boundary_authorizes_model_call",
        "carried_next_step_approval_boundary_authorizes_tool_execution",
        "carried_next_step_approval_boundary_authorizes_personal_data_read",
        "carried_next_step_approval_boundary_authorizes_external_side_effect",
        "carried_next_step_approval_boundary_authorizes_timebox_reuse",
        "carried_next_step_approval_boundary_reusable_for_next_review",
        "carried_next_step_approval_boundary_reusable_for_recovery_review",
        "closure_prior_step_authorizes_followup",
        "closure_prior_step_reusable_for_next_step",
        "autonomy_cycle_ledger_token_authorizes_action_now",
        "autonomy_cycle_ledger_token_authorizes_local_safe_step",
        "autonomy_cycle_ledger_token_authorizes_risky_work",
        "autonomy_cycle_ledger_token_authorizes_new_cycle",
        "autonomy_cycle_ledger_token_authorizes_unreviewed_followthrough",
        "autonomy_cycle_ledger_token_authorizes_timebox_reuse",
        "autonomy_cycle_ledger_token_authorizes_checkpoint_reuse",
        "autonomy_cycle_ledger_token_authorizes_token_reuse",
        "autonomy_cycle_ledger_token_authorizes_model_call",
        "autonomy_cycle_ledger_token_authorizes_tool_execution",
        "autonomy_cycle_ledger_token_authorizes_personal_data_read",
        "autonomy_cycle_ledger_token_authorizes_external_side_effect",
        "previous_cycle_ledger_token_reusable_for_next_review",
        "fresh_review_boundary_token_authorizes_action_now",
        "fresh_review_boundary_token_authorizes_local_safe_step",
        "fresh_review_boundary_token_authorizes_risky_work",
        "fresh_review_boundary_token_authorizes_unreviewed_followthrough",
        "fresh_review_boundary_token_authorizes_timebox_reuse",
        "fresh_review_boundary_token_authorizes_checkpoint_reuse",
        "fresh_review_boundary_token_authorizes_token_reuse",
        "fresh_review_boundary_token_authorizes_model_call",
        "fresh_review_boundary_token_authorizes_tool_execution",
        "fresh_review_boundary_token_authorizes_personal_data_read",
        "fresh_review_boundary_token_authorizes_external_side_effect",
        "fresh_review_boundary_token_reusable_for_next_review",
        "fresh_review_boundary_token_reusable_for_next_cycle",
        "next_review_start_command_authorizes_action_now",
        "next_review_start_command_authorizes_local_safe_step",
        "next_review_start_command_authorizes_risky_work",
        "next_review_start_command_authorizes_unreviewed_followthrough",
        "next_review_start_command_authorizes_timebox_reuse",
        "next_review_start_command_authorizes_checkpoint_reuse",
        "next_review_start_command_authorizes_token_reuse",
        "next_review_start_command_authorizes_model_call",
        "next_review_start_command_authorizes_tool_execution",
        "next_review_start_command_authorizes_personal_data_read",
        "next_review_start_command_authorizes_external_side_effect",
        "next_review_start_command_authorizes_approval",
        "next_review_start_command_reusable_for_next_review",
        "next_review_start_command_reusable_for_next_cycle",
        "prior_cycle_ledger_proof_authorizes_new_action",
        "prior_cycle_ledger_proof_authorizes_model_call",
        "prior_cycle_ledger_proof_authorizes_tool_execution",
        "prior_cycle_ledger_proof_authorizes_personal_data_read",
        "prior_cycle_ledger_proof_authorizes_external_side_effect",
        "previous_continuation_permission_reusable_for_next_review",
        "previous_post_step_receipt_reusable_for_next_review",
        "previous_post_step_receipt_hash_reusable_for_next_review",
        "previous_post_step_checkpoint_hash_reusable_for_next_review",
    ]
    return all(metadata.get(flag) is False for flag in false_flags)


def _scorecard_required_rows_ready(rows: list[dict[str, Any]]) -> bool:
    return all(row.get("ready") is True for row in rows)


def _metadata_flag_ready(metadata: dict[str, Any], key: str) -> bool:
    return metadata.get(key) is True


def _metadata_flag_disabled(metadata: dict[str, Any], key: str) -> bool:
    return metadata.get(key) is False


def _autonomy_cycle_preflight_scorecard_rows(
    *,
    continuation_ready: bool,
    closure_ready: bool,
    stage_rows: list[dict[str, Any]],
    recovery_artifact_hashes_match_files: bool,
    completed_step_matches_proposed: bool,
    post_step_verification_matches_proposed: bool,
    post_step_artifact_hashes_present: bool,
    post_step_artifact_hashes_match_files: bool,
    closure_fresh_contract_enforced: bool,
    all_prior_artifacts_non_authorizing: bool,
    ledger_ready: bool,
) -> list[dict[str, Any]]:
    all_stages_ready = _scorecard_required_rows_ready(stage_rows)
    checks = [
        ("pre_step_continuation_permission", 15, continuation_ready),
        ("cycle_stage_readiness", 15, all_stages_ready),
        ("recovery_artifact_file_binding", 15, recovery_artifact_hashes_match_files),
        ("completed_step_identity", 10, completed_step_matches_proposed),
        ("post_step_verification_identity", 10, post_step_verification_matches_proposed),
        ("post_step_artifact_file_binding", 10, post_step_artifact_hashes_present and post_step_artifact_hashes_match_files),
        ("fresh_continuation_contract", 10, closure_fresh_contract_enforced),
        ("prior_artifacts_non_authorizing", 10, all_prior_artifacts_non_authorizing),
        ("fresh_review_cycle_boundary", 5, ledger_ready and closure_ready),
    ]
    return [
        {
            "item": item,
            "points": max_points if ready else 0,
            "max_points": max_points,
            "ready": ready,
            "required_before_next_continuation_review": True,
            "authorizes_action": False,
            "authorizes_risky_work": False,
        }
        for item, max_points, ready in checks
    ]


_AUTONOMY_RESUME_SCORECARD_ITEMS = {
    "operator_timebox_active",
    "recovery_cockpit_ready",
    "latest_checkpoint_path_binding",
    "latest_checkpoint_hash_binding",
    "followthrough_closure_ready",
    "artifact_hashes_match_files",
    "risky_recovery_approval_boundary",
    "fresh_followthrough_token",
    "one_step_boundary_intact",
}
_AUTONOMY_CONTINUATION_SCORECARD_ITEMS = {
    "ready_resume_gate",
    "exact_next_step",
    "post_step_verification_target",
    "stop_condition",
    "risky_work_approval_boundary",
    "fresh_continuation_review_token",
    "prior_cycle_proof_non_authorizing",
    "one_step_contract_non_reusable",
    "post_step_proof_queue_ready",
}
_AUTONOMY_STEP_CLOSURE_SCORECARD_ITEMS = {
    "pre_step_permission_ready",
    "completed_step_matches_allowed",
    "post_step_verification_matches_target",
    "post_step_receipt_hash_binding",
    "post_step_checkpoint_hash_binding",
    "execution_health_supplied",
    "execution_audit_supplied",
    "after_action_learning_supplied",
    "fresh_review_contract_non_reusable",
}
_AUTONOMY_CYCLE_PREFLIGHT_SCORECARD_ITEMS = {
    "pre_step_continuation_permission",
    "cycle_stage_readiness",
    "recovery_artifact_file_binding",
    "completed_step_identity",
    "post_step_verification_identity",
    "post_step_artifact_file_binding",
    "fresh_continuation_contract",
    "prior_artifacts_non_authorizing",
    "fresh_review_cycle_boundary",
}
_AUTONOMY_CYCLE_STAGE_NAMES = {
    "operator_timebox",
    "operator_supersession",
    "checkpoint_route",
    "recovery_cockpit",
    "resume_gate",
    "one_step_continuation",
    "recovery_followthrough",
    "post_step_closure",
}
_RECOVERY_EXECUTION_READINESS_SCORECARD_ITEMS = {
    "reviewed_step_permission",
    "verification_target",
    "receipt_hash_binding",
    "checkpoint_hash_binding",
    "stop_condition",
    "risky_recovery_approval_boundary",
    "followthrough_packet_ready",
    "fresh_followthrough_token",
}
_RECOVERY_COCKPIT_READINESS_SCORECARD_ITEMS = {
    "operator_timebox_active",
    "checkpoint_available",
    "checkpoint_fresh_for_review",
    "no_pending_approval_blockers",
    "no_failed_run_blockers",
    "proof_queue_ready",
    "apply_boundary_intact",
    "resume_state_consistent",
}
_AUTONOMY_SCORECARD_NO_AUTHORITY_FIELDS = [
    "authorizes_action",
    "authorizes_action_now",
    "authorizes_risky_work",
    "authorizes_execution",
    "authorizes_unreviewed_continuation",
    "authorizes_unreviewed_followthrough",
    "authorizes_batching",
    "authorizes_followup_without_closure",
    "authorizes_followup_without_fresh_review",
    "prior_cycle_ledger_token_authorizes_action",
]
_RECOVERY_EXECUTION_SCORECARD_NO_AUTHORITY_FIELDS = [
    "authorizes_action_now",
    "authorizes_risky_work",
    "authorizes_unreviewed_followthrough",
    "authorizes_model_call",
    "authorizes_tool_execution",
    "authorizes_personal_data_read",
    "authorizes_external_side_effect",
]
_AUTONOMY_CYCLE_STAGE_NO_AUTHORITY_FIELDS = [
    "authorizes_action_now",
    "authorizes_local_safe_step",
    "authorizes_risky_work",
    "authorizes_unreviewed_followthrough",
    "authorizes_timebox_reuse",
    "authorizes_checkpoint_reuse",
    "authorizes_token_reuse",
    "authorizes_approval",
    "authorizes_model_call",
    "authorizes_tool_execution",
    "authorizes_personal_data_read",
    "authorizes_external_side_effect",
    "reusable_for_next_review",
    "reusable_for_next_cycle",
]


def _autonomy_expected_post_step_proof_queue(*, proposed_next_step: str, proposed_verification: str) -> list[str]:
    return [
        f"checkpoint recovery execute reviewed=true step={proposed_next_step or '<reviewed local-safe step>'} verification={proposed_verification or '<evidence>'}",
        "autonomy step closure: step=<completed local-safe step> verification=<post-step evidence> receipt=<post-step receipt> checkpoint=<fresh checkpoint> health=<execution health> audit=<execution audit> learning=<after-action learning>",
        "work block checkpoint",
        "build delta",
        "execution health report",
        "completion claim gate",
    ]


def _autonomy_post_step_proof_queue_ready(metadata: dict[str, Any]) -> bool:
    expected = _autonomy_expected_post_step_proof_queue(
        proposed_next_step=str(metadata.get("proposed_next_step") or ""),
        proposed_verification=str(metadata.get("proposed_verification") or ""),
    )
    queue = [
        str(command)
        for command in (
            metadata.get("post_step_proof_queue")
            or metadata.get("continuation_post_step_proof_queue")
            or []
        )
    ]
    if queue != expected:
        return False

    top_level_present = "post_step_proof_queue" in metadata or "post_step_proof_queue_count" in metadata
    if top_level_present:
        if metadata.get("post_step_proof_queue") != expected:
            return False
        if metadata.get("post_step_proof_queue_count") != len(expected):
            return False
        if metadata.get("post_step_next_proof_command") != expected[0]:
            return False

    carried_present = (
        "continuation_post_step_proof_queue" in metadata
        or "continuation_post_step_proof_queue_count" in metadata
    )
    if carried_present:
        if metadata.get("continuation_post_step_proof_queue") != expected:
            return False
        if metadata.get("continuation_post_step_proof_queue_count") != len(expected):
            return False
        if metadata.get("continuation_post_step_next_proof_command") != expected[0]:
            return False

    return True


def _autonomy_scorecard_ready(
    rows: list[dict[str, Any]],
    *,
    expected_items: set[str],
    required_field: str,
) -> bool:
    if len(rows) != len(expected_items):
        return False
    if {str(row.get("item") or "") for row in rows} != expected_items:
        return False
    total_max = 0
    for row in rows:
        if row.get(required_field) is not True:
            return False
        if row.get("ready") is not True:
            return False
        try:
            points = int(row.get("points"))
            max_points = int(row.get("max_points"))
        except (TypeError, ValueError):
            return False
        if max_points <= 0 or points != max_points:
            return False
        total_max += max_points
        for field in _AUTONOMY_SCORECARD_NO_AUTHORITY_FIELDS:
            if field in row and row.get(field) is not False:
                return False
    return total_max == 100


def _autonomy_cycle_stage_rows_ready(rows: list[dict[str, Any]]) -> bool:
    if len(rows) != len(_AUTONOMY_CYCLE_STAGE_NAMES):
        return False
    if {str(row.get("stage") or "") for row in rows} != _AUTONOMY_CYCLE_STAGE_NAMES:
        return False
    for row in rows:
        if row.get("ready") is not True:
            return False
        if not str(row.get("state") or "").strip():
            return False
        if not str(row.get("proof") or "").strip():
            return False
        for field in _AUTONOMY_CYCLE_STAGE_NO_AUTHORITY_FIELDS:
            if row.get(field) is not False:
                return False
    return True


def _recovery_execution_readiness_scorecard_ready(rows: list[dict[str, Any]]) -> bool:
    if not _autonomy_scorecard_ready(
        rows,
        expected_items=_RECOVERY_EXECUTION_READINESS_SCORECARD_ITEMS,
        required_field="required_before_normal_followthrough",
    ):
        return False
    return all(
        row.get(field) is False
        for row in rows
        for field in _RECOVERY_EXECUTION_SCORECARD_NO_AUTHORITY_FIELDS
    )


def _recovery_execution_scorecard_metadata_ready(
    metadata: dict[str, Any],
    *,
    prefix: str = "recovery_execution",
    prior_proof_key: str | None = None,
) -> bool:
    rows = list(metadata.get(f"{prefix}_scorecard_rows") or [])
    scorecard_ready = _recovery_execution_readiness_scorecard_ready(rows)
    if metadata.get(f"{prefix}_scorecard_row_count") != len(rows):
        return False
    try:
        score = sum(int(row.get("points")) for row in rows)
        max_score = sum(int(row.get("max_points")) for row in rows)
    except (TypeError, ValueError):
        return False
    if metadata.get(f"{prefix}_score") != score:
        return False
    if metadata.get(f"{prefix}_max_score") != max_score:
        return False
    if max_score != 100:
        return False
    if metadata.get(f"{prefix}_required_rows_ready") is not scorecard_ready:
        return False
    if metadata.get(f"{prefix}_scorecard_ready") is not scorecard_ready:
        return False
    if prior_proof_key and metadata.get(prior_proof_key) is True and not scorecard_ready:
        return False
    return scorecard_ready


def _recovery_cockpit_readiness_scorecard_ready(
    rows: list[dict[str, Any]],
    *,
    require_all_ready: bool = False,
) -> bool:
    if len(rows) != len(_RECOVERY_COCKPIT_READINESS_SCORECARD_ITEMS):
        return False
    if {str(row.get("item") or "") for row in rows} != _RECOVERY_COCKPIT_READINESS_SCORECARD_ITEMS:
        return False
    total_max = 0
    total_points = 0
    for row in rows:
        if row.get("required_before_local_safe_review") is not True:
            return False
        if row.get("authorizes_risky_work") is not False or row.get("authorizes_execution") is not False:
            return False
        for field in [
            "authorizes_model_call",
            "authorizes_tool_execution",
            "authorizes_personal_data_read",
            "authorizes_external_side_effect",
        ]:
            if field in row and row.get(field) is not False:
                return False
        if row.get("item") == "apply_boundary_intact":
            recovery_apply_queue = [str(command) for command in (row.get("recovery_apply_proof_queue") or [])]
            if row.get("apply_requires_approval") is not True:
                return False
            if not recovery_apply_queue:
                return False
            if row.get("recovery_apply_proof_queue_count") != len(recovery_apply_queue):
                return False
            if row.get("recovery_apply_next_proof_command") != recovery_apply_queue[0]:
                return False
            if row.get("recovery_apply_queue_ready") is not True:
                return False
        if row.get("ready") not in {True, False}:
            return False
        try:
            points = int(row.get("points"))
            max_points = int(row.get("max_points"))
        except (TypeError, ValueError):
            return False
        if max_points <= 0:
            return False
        if row.get("ready") is True and points != max_points:
            return False
        if row.get("ready") is False and points != 0:
            return False
        total_points += points
        total_max += max_points
    if total_max != 100:
        return False
    if require_all_ready and total_points != total_max:
        return False
    return True


def _autonomy_resume_readiness_scorecard_ready(rows: list[dict[str, Any]]) -> bool:
    if not _autonomy_scorecard_ready(
        rows,
        expected_items=_AUTONOMY_RESUME_SCORECARD_ITEMS,
        required_field="required_before_autonomy_resume",
    ):
        return False
    one_step_rows = [row for row in rows if row.get("item") == "one_step_boundary_intact"]
    if len(one_step_rows) != 1:
        return False
    one_step_row = one_step_rows[0]
    resume_proof_queue = [str(command) for command in (one_step_row.get("resume_proof_queue") or [])]
    if not resume_proof_queue:
        return False
    if one_step_row.get("resume_proof_queue_count") != len(resume_proof_queue):
        return False
    if one_step_row.get("resume_next_proof_command") != resume_proof_queue[0]:
        return False
    if one_step_row.get("resume_one_step_boundary_ready") is not True:
        return False
    if one_step_row.get("authorizes_local_safe_step") is not False:
        return False
    if one_step_row.get("authorizes_approval") is not False:
        return False
    if one_step_row.get("authorizes_model_call") is not False:
        return False
    if one_step_row.get("authorizes_tool_execution") is not False:
        return False
    if one_step_row.get("authorizes_personal_data_read") is not False:
        return False
    if one_step_row.get("authorizes_external_side_effect") is not False:
        return False
    return True


def _autonomy_resume_gate_ready_from_metadata(metadata: dict[str, Any]) -> bool:
    if metadata.get("resume_gate_state") != "AUTONOMY_RESUME_READY_FOR_LOCAL_SAFE_CONTINUATION":
        return False
    if metadata.get("normal_autonomous_followthrough_allowed") is not True:
        return False
    if metadata.get("blockers") != [] or metadata.get("blocker_count") != 0:
        return False
    if metadata.get("next_safe_command") != "continue one scoped local-safe Jarvis build step from the fresh checkpoint":
        return False

    proof_queue = list(metadata.get("proof_queue") or [])
    if metadata.get("proof_queue_count") != len(proof_queue):
        return False
    if not proof_queue:
        return False
    if metadata.get("next_proof_command") != proof_queue[0]:
        return False

    rows = list(metadata.get("resume_readiness_scorecard_rows") or [])
    if metadata.get("resume_readiness_scorecard_row_count") != len(rows):
        return False
    if metadata.get("resume_readiness_score") != 100 or metadata.get("resume_readiness_max_score") != 100:
        return False
    if metadata.get("resume_readiness_required_rows_ready") is not True:
        return False
    if metadata.get("resume_readiness_scorecard_ready") is not True:
        return False
    if not _autonomy_resume_readiness_scorecard_ready(rows):
        return False

    if metadata.get("timebox_state") != "STOP_WINDOW_ACTIVE":
        return False
    if metadata.get("can_continue_now") is not True:
        return False
    if not _carried_operator_timebox_review_contract_ready(metadata):
        return False
    if not _awake_guard_boundary_ready_from_metadata(metadata):
        return False
    if metadata.get("can_continue_under_latest_instruction") is not True:
        return False
    if not _carried_operator_supersession_token_boundary_ready(metadata):
        return False
    if metadata.get("can_resume_local_safe_review") is not True:
        return False
    if not _carried_checkpoint_route_boundary_ready(metadata):
        return False
    if metadata.get("checkpoint_path_matches_latest") is not True:
        return False
    if metadata.get("checkpoint_hash_matches_latest") is not True:
        return False
    if metadata.get("followthrough_closure_ready") is not True:
        return False
    if metadata.get("recovery_artifact_hashes_match_files") is not True:
        return False
    if not _carried_recovery_followthrough_token_boundary_ready(
        metadata,
        expected_source="checkpoint_recovery_followthrough",
    ):
        return False
    if not _carried_local_safe_recovery_execution_token_boundary_ready(
        metadata,
        expected_source="autonomy_resume_gate",
    ):
        return False
    if not _carried_recovery_execution_readiness_token_boundary_ready(
        metadata,
        expected_source="checkpoint_recovery_followthrough",
    ):
        return False

    recovery_rows = list(metadata.get("carried_recovery_execution_scorecard_rows") or [])
    if metadata.get("carried_recovery_execution_scorecard_row_count") != len(recovery_rows):
        return False
    if metadata.get("carried_recovery_execution_score") != 100 or metadata.get("carried_recovery_execution_max_score") != 100:
        return False
    if metadata.get("carried_recovery_execution_required_rows_ready") is not True:
        return False
    if metadata.get("carried_recovery_execution_scorecard_ready") is not True:
        return False
    if metadata.get("carried_recovery_execution_as_prior_proof") is not True:
        return False
    if not _recovery_execution_readiness_scorecard_ready(recovery_rows):
        return False

    false_flags = [
        "tools_executed",
        "approvals_queued",
        "timebox_authorizes_execution",
        "timebox_authorizes_local_safe_step",
        "timebox_authorizes_risky_work",
        "timebox_authorizes_approval",
        "timebox_authorizes_timebox_reuse",
        "supersession_token_authorizes_execution",
        "supersession_token_authorizes_local_safe_step",
        "supersession_token_authorizes_risky_work",
        "supersession_token_authorizes_approval",
        "supersession_token_authorizes_recovery_followthrough",
        "checkpoint_route_authorizes_continuation",
        "checkpoint_route_authorizes_local_safe_step",
        "checkpoint_route_authorizes_risky_work",
        "checkpoint_route_authorizes_approval",
        "recovery_followthrough_token_reusable_for_future_recovery",
        "recovery_followthrough_token_authorizes_resume_gate",
        "recovery_followthrough_token_authorizes_next_step",
        "recovery_followthrough_token_authorizes_risky_work",
        "recovery_followthrough_token_authorizes_approval",
        "local_safe_recovery_execution_token_authorizes_resume_gate",
        "local_safe_recovery_execution_token_authorizes_next_step",
        "local_safe_recovery_execution_token_authorizes_risky_work",
        "local_safe_recovery_execution_token_authorizes_approval",
        "recovery_execution_readiness_token_authorizes_resume_gate",
        "recovery_execution_readiness_token_authorizes_next_step",
        "recovery_execution_readiness_token_authorizes_risky_work",
        "recovery_execution_readiness_token_authorizes_approval",
        "recovery_execution_readiness_token_authorizes_unreviewed_followthrough",
        "carried_recovery_execution_authorizes_action_now",
        "carried_recovery_execution_authorizes_risky_work",
        "carried_recovery_execution_authorizes_unreviewed_followthrough",
        "risky_recovery_without_approval",
        "calls_model",
        "executes_tools",
        "queues_approval",
        "requires_approval",
        "approves_request",
        "dismisses_request",
        "reads_personal_data",
        "reads_private_data",
        "executes_side_effect",
        "external_side_effect",
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "controls_computer",
        "speaks",
    ]
    return all(metadata.get(flag) is False for flag in false_flags)


def _autonomy_continuation_execution_ready_from_metadata(metadata: dict[str, Any]) -> bool:
    if metadata.get("continuation_state") != "AUTONOMY_CONTINUATION_READY_FOR_ONE_LOCAL_SAFE_STEP":
        return False
    if metadata.get("one_local_safe_step_allowed") is not True:
        return False
    if metadata.get("normal_autonomous_followthrough_allowed") is not True:
        return False
    if metadata.get("missing_blockers") != [] or metadata.get("missing_blocker_count") != 0:
        return False

    post_step_queue = [str(command) for command in (metadata.get("post_step_proof_queue") or [])]
    if not post_step_queue:
        return False
    if metadata.get("post_step_proof_queue_count") != len(post_step_queue):
        return False
    if metadata.get("post_step_next_proof_command") != post_step_queue[0]:
        return False
    if metadata.get("next_safe_command") != post_step_queue[0]:
        return False
    if not _autonomy_post_step_proof_queue_ready(metadata):
        return False

    rows = list(metadata.get("continuation_readiness_scorecard_rows") or [])
    if metadata.get("continuation_readiness_scorecard_row_count") != len(rows):
        return False
    if metadata.get("continuation_readiness_score") != 100 or metadata.get("continuation_readiness_max_score") != 100:
        return False
    if metadata.get("continuation_readiness_required_rows_ready") is not True:
        return False
    if metadata.get("continuation_readiness_scorecard_ready") is not True:
        return False
    if not _autonomy_continuation_readiness_scorecard_ready(rows):
        return False

    if metadata.get("resume_gate_state") != "AUTONOMY_RESUME_READY_FOR_LOCAL_SAFE_CONTINUATION":
        return False
    if not _carried_operator_timebox_review_contract_ready(metadata):
        return False
    if not _awake_guard_boundary_ready_from_metadata(metadata):
        return False
    if metadata.get("can_continue_under_latest_instruction") is not True:
        return False
    if not _carried_operator_supersession_token_boundary_ready(metadata):
        return False
    if not _carried_checkpoint_route_boundary_ready(metadata):
        return False
    if metadata.get("checkpoint_path_matches_latest") is not True:
        return False
    if metadata.get("checkpoint_hash_matches_latest") is not True:
        return False
    if metadata.get("recovery_artifact_hashes_match_files") is not True:
        return False
    if not _carried_recovery_followthrough_token_boundary_ready(
        metadata,
        expected_source="checkpoint_recovery_followthrough",
    ):
        return False
    if not _local_safe_recovery_execution_token_metadata_ready(
        metadata,
        expected_source="autonomy_continuation_execution",
    ):
        return False
    if not _carried_recovery_execution_readiness_token_boundary_ready(
        metadata,
        expected_source="checkpoint_recovery_followthrough",
    ):
        return False

    recovery_rows = list(metadata.get("carried_recovery_execution_scorecard_rows") or [])
    if metadata.get("carried_recovery_execution_scorecard_row_count") != len(recovery_rows):
        return False
    if metadata.get("carried_recovery_execution_score") != 100 or metadata.get("carried_recovery_execution_max_score") != 100:
        return False
    if metadata.get("carried_recovery_execution_required_rows_ready") is not True:
        return False
    if metadata.get("carried_recovery_execution_scorecard_ready") is not True:
        return False
    if metadata.get("carried_recovery_execution_as_prior_proof") is not True:
        return False
    if not _recovery_execution_readiness_scorecard_ready(recovery_rows):
        return False

    if not _continuation_review_token_metadata_ready(
        metadata,
        reusable_key="continuation_review_token_reusable_for_next_review",
    ):
        return False
    if not _prior_cycle_ledger_token_metadata_ready(
        metadata,
        reusable_key="prior_cycle_ledger_token_reusable_for_this_review",
        authority_action_key="prior_cycle_ledger_proof_authorizes_action_now",
    ):
        return False
    if metadata.get("next_step_requires_fresh_cycle_ledger_token") is not True:
        return False
    contract_rows = list(metadata.get("one_step_execution_contract_rows") or [])
    if metadata.get("one_step_execution_contract_row_count") != len(contract_rows):
        return False
    if metadata.get("one_step_execution_contract_ready") is not True:
        return False
    for flag in [
        "one_step_execution_contract_binds_awake_guard",
        "one_step_execution_contract_binds_operator_supersession",
        "one_step_execution_contract_binds_timebox_review_contract",
        "one_step_execution_contract_all_local_safe_step_limited",
        "one_step_execution_contract_all_non_reusable",
        "one_step_execution_contract_all_risky_work_gated",
        "one_step_execution_contract_requires_fresh_closure",
    ]:
        if metadata.get(flag) is not True:
            return False
    if not _one_step_execution_contract_ready(
        objective=str(metadata.get("objective") or ""),
        continuation_state=str(metadata.get("continuation_state") or ""),
        resume_gate_state=str(metadata.get("resume_gate_state") or ""),
        continuation_ready=metadata.get("one_local_safe_step_allowed"),
        one_step_execution_contract_rows=contract_rows,
        one_step_execution_contract_token_sha256=str(metadata.get("one_step_execution_contract_token_sha256") or ""),
        timebox_receipt_sha256=str(metadata.get("timebox_receipt_sha256") or ""),
        awake_guard_token_sha256=str(metadata.get("awake_guard_token_sha256") or ""),
        supersession_token_sha256=str(metadata.get("supersession_token_sha256") or ""),
        checkpoint_route_token_sha256=str(metadata.get("checkpoint_route_token_sha256") or ""),
        recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
        local_safe_recovery_execution_token_sha256=str(metadata.get("local_safe_recovery_execution_token_sha256") or ""),
        prior_cycle_ledger_token_sha256=str(metadata.get("prior_cycle_ledger_token_sha256") or ""),
        continuation_review_token_sha256=str(metadata.get("continuation_review_token_sha256") or ""),
        proposed_next_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
        proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
        risk_signals=[str(signal) for signal in (metadata.get("proposed_next_step_risk_signals") or [])],
        timebox_review_contract_ready=metadata.get("timebox_review_contract_ready"),
        timebox_review_contract_rows=list(metadata.get("timebox_review_contract_rows") or []),
        post_step_proof_queue=post_step_queue,
    ):
        return False
    if metadata.get("proposed_next_step_requires_approval") is not False:
        return False
    if metadata.get("proposed_next_step_risk_signals") != [] or metadata.get("proposed_next_step_risk_signal_count") != 0:
        return False
    if metadata.get("proposed_next_step_approval_proof_queue") != []:
        return False
    if metadata.get("proposed_next_step_approval_proof_queue_count") != 0:
        return False
    if metadata.get("proposed_next_step_next_approval_proof_command") != "":
        return False
    if metadata.get("proposed_next_step_approval_required_before_review") is not False:
        return False
    if metadata.get("proposed_next_step_approval_boundary_ready") is not True:
        return False
    if not _next_step_approval_boundary_metadata_ready(
        metadata,
        prefix="proposed_next_step",
        followup_authority_key="proposed_next_step_approval_boundary_authorizes_followup_without_closure",
        require_boundary_ready=True,
    ):
        return False

    false_flags = [
        "timebox_authorizes_execution",
        "timebox_authorizes_local_safe_step",
        "timebox_authorizes_risky_work",
        "timebox_authorizes_approval",
        "timebox_authorizes_timebox_reuse",
        "awake_guard_authorizes_os_wake_lock",
        "awake_guard_authorizes_shell_execution",
        "awake_guard_authorizes_computer_control",
        "awake_guard_authorizes_approval",
        "supersession_token_authorizes_execution",
        "supersession_token_authorizes_local_safe_step",
        "supersession_token_authorizes_risky_work",
        "supersession_token_authorizes_approval",
        "checkpoint_route_authorizes_continuation",
        "checkpoint_route_authorizes_local_safe_step",
        "checkpoint_route_authorizes_risky_work",
        "checkpoint_route_authorizes_approval",
        "recovery_followthrough_token_reusable_for_future_recovery",
        "recovery_followthrough_token_authorizes_resume_gate",
        "recovery_followthrough_token_authorizes_next_step",
        "recovery_followthrough_token_authorizes_risky_work",
        "recovery_followthrough_token_authorizes_approval",
        "local_safe_recovery_execution_token_authorizes_resume_gate",
        "local_safe_recovery_execution_token_authorizes_next_step",
        "local_safe_recovery_execution_token_authorizes_risky_work",
        "local_safe_recovery_execution_token_authorizes_approval",
        "recovery_execution_readiness_token_authorizes_resume_gate",
        "recovery_execution_readiness_token_authorizes_next_step",
        "recovery_execution_readiness_token_authorizes_risky_work",
        "recovery_execution_readiness_token_authorizes_approval",
        "recovery_execution_readiness_token_authorizes_unreviewed_followthrough",
        "carried_recovery_execution_authorizes_action_now",
        "carried_recovery_execution_authorizes_risky_work",
        "carried_recovery_execution_authorizes_unreviewed_followthrough",
        "prior_cycle_ledger_proof_authorizes_action_now",
        "one_step_execution_contract_token_as_prior_proof",
        "one_step_execution_contract_token_authorizes_action_now",
        "one_step_execution_contract_token_authorizes_risky_work",
        "one_step_execution_contract_token_authorizes_unreviewed_followthrough",
        "one_step_execution_contract_token_authorizes_batching",
        "one_step_execution_contract_token_reusable_for_next_step",
        "one_step_execution_contract_authorizes_batching",
        "one_step_execution_contract_authorizes_followup_without_closure",
        "proposed_next_step_approval_boundary_authorizes_action_now",
        "proposed_next_step_approval_boundary_authorizes_risky_work",
        "proposed_next_step_approval_boundary_authorizes_followup_without_closure",
        "proposed_next_step_approval_boundary_authorizes_approval",
        "proposed_next_step_approval_boundary_authorizes_model_call",
        "proposed_next_step_approval_boundary_authorizes_tool_execution",
        "proposed_next_step_approval_boundary_authorizes_personal_data_read",
        "proposed_next_step_approval_boundary_authorizes_external_side_effect",
        "proposed_next_step_approval_boundary_authorizes_timebox_reuse",
        "proposed_next_step_approval_boundary_reusable_for_next_review",
        "proposed_next_step_approval_boundary_reusable_for_recovery_review",
        "calls_model",
        "executes_tools",
        "queues_approval",
        "requires_approval",
        "approves_request",
        "dismisses_request",
        "reads_personal_data",
        "reads_private_data",
        "executes_side_effect",
        "external_side_effect",
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "controls_computer",
        "speaks",
    ]
    return all(metadata.get(flag) is False for flag in false_flags)


def _autonomy_continuation_readiness_scorecard_ready(rows: list[dict[str, Any]]) -> bool:
    return _autonomy_scorecard_ready(
        rows,
        expected_items=_AUTONOMY_CONTINUATION_SCORECARD_ITEMS,
        required_field="required_before_one_local_safe_step",
    )


def _autonomy_step_closure_readiness_scorecard_ready(rows: list[dict[str, Any]]) -> bool:
    return _autonomy_scorecard_ready(
        rows,
        expected_items=_AUTONOMY_STEP_CLOSURE_SCORECARD_ITEMS,
        required_field="required_before_next_continuation_review",
    )


def _autonomy_cycle_preflight_scorecard_ready(rows: list[dict[str, Any]]) -> bool:
    return _autonomy_scorecard_ready(
        rows,
        expected_items=_AUTONOMY_CYCLE_PREFLIGHT_SCORECARD_ITEMS,
        required_field="required_before_next_continuation_review",
    )


def _recovery_execution_readiness_scorecard_rows(
    *,
    reviewed: bool,
    approved_step: str,
    verification: str,
    receipt_hash_matches_file: bool,
    checkpoint_hash_matches_file: bool,
    stop_condition: str,
    risk_signals: list[str],
    approval_reference_present: bool,
    followthrough_ready: bool,
    recovery_followthrough_token_sha256: str,
) -> list[dict[str, Any]]:
    risk_boundary_ready = not risk_signals or approval_reference_present
    checks = [
        ("reviewed_step_permission", 15, reviewed and bool(approved_step)),
        ("verification_target", 15, bool(verification)),
        ("receipt_hash_binding", 15, receipt_hash_matches_file),
        ("checkpoint_hash_binding", 15, checkpoint_hash_matches_file),
        ("stop_condition", 10, bool(stop_condition)),
        ("risky_recovery_approval_boundary", 10, risk_boundary_ready),
        ("followthrough_packet_ready", 10, followthrough_ready),
        ("fresh_followthrough_token", 10, _looks_like_sha256(recovery_followthrough_token_sha256)),
    ]
    return [
        {
            "item": item,
            "points": max_points if ready else 0,
            "max_points": max_points,
            "ready": ready,
            "required_before_normal_followthrough": True,
            "authorizes_action_now": False,
            "authorizes_risky_work": False,
            "authorizes_unreviewed_followthrough": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_personal_data_read": False,
            "authorizes_external_side_effect": False,
        }
        for item, max_points, ready in checks
    ]


def _recovery_cockpit_readiness_scorecard_rows(
    *,
    can_continue_now: bool,
    checkpoint_found: bool,
    checkpoint_read: bool,
    checkpoint_needs_review: bool,
    pending_approvals: int,
    failed_runs: int,
    proof_queue: list[str],
    apply_requires_approval: bool,
    recovery_apply_proof_queue: list[str],
    recovery_apply_proof_queue_count: int,
    recovery_apply_next_proof_command: str,
    can_resume_local_safe: bool,
    recovery_ready_for_review: bool,
) -> list[dict[str, Any]]:
    expected_resume_state = bool(can_continue_now and recovery_ready_for_review)
    recovery_apply_queue_ready = bool(recovery_apply_proof_queue) and (
        recovery_apply_proof_queue_count == len(recovery_apply_proof_queue)
        and recovery_apply_next_proof_command == recovery_apply_proof_queue[0]
    )
    checks = [
        ("operator_timebox_active", 20, can_continue_now),
        ("checkpoint_available", 15, checkpoint_found and checkpoint_read),
        ("checkpoint_fresh_for_review", 15, not checkpoint_needs_review),
        ("no_pending_approval_blockers", 15, pending_approvals == 0),
        ("no_failed_run_blockers", 15, failed_runs == 0),
        ("proof_queue_ready", 10, bool(proof_queue)),
        ("apply_boundary_intact", 5, apply_requires_approval and recovery_apply_queue_ready),
        ("resume_state_consistent", 5, can_resume_local_safe == expected_resume_state),
    ]
    rows = [
        {
            "item": item,
            "points": max_points if ready else 0,
            "max_points": max_points,
            "ready": ready,
            "required_before_local_safe_review": True,
            "authorizes_risky_work": False,
            "authorizes_execution": False,
        }
        for item, max_points, ready in checks
    ]
    for row in rows:
        if row["item"] == "apply_boundary_intact":
            row.update(
                {
                    "apply_requires_approval": apply_requires_approval,
                    "recovery_apply_proof_queue": recovery_apply_proof_queue,
                    "recovery_apply_proof_queue_count": recovery_apply_proof_queue_count,
                    "recovery_apply_next_proof_command": recovery_apply_next_proof_command,
                    "recovery_apply_queue_ready": recovery_apply_queue_ready,
                    "authorizes_model_call": False,
                    "authorizes_tool_execution": False,
                    "authorizes_personal_data_read": False,
                    "authorizes_external_side_effect": False,
                }
            )
    return rows


def _autonomy_resume_readiness_scorecard_rows(
    *,
    can_continue_now: bool,
    can_resume_local_safe_review: bool,
    checkpoint_path_matches_latest: bool,
    checkpoint_hash_matches_latest: bool,
    followthrough_closure_ready: bool,
    recovery_artifact_hashes_match_files: bool,
    risky_recovery_without_approval: bool,
    recovery_followthrough_token_sha256: str,
    resume_proof_queue: list[str],
    resume_proof_queue_count: int,
    resume_next_proof_command: str,
) -> list[dict[str, Any]]:
    resume_one_step_boundary_ready = bool(resume_proof_queue) and (
        resume_proof_queue_count == len(resume_proof_queue)
        and resume_next_proof_command == resume_proof_queue[0]
    )
    checks = [
        ("operator_timebox_active", 15, can_continue_now),
        ("recovery_cockpit_ready", 15, can_resume_local_safe_review),
        ("latest_checkpoint_path_binding", 10, checkpoint_path_matches_latest),
        ("latest_checkpoint_hash_binding", 10, checkpoint_hash_matches_latest),
        ("followthrough_closure_ready", 15, followthrough_closure_ready),
        ("artifact_hashes_match_files", 15, recovery_artifact_hashes_match_files),
        ("risky_recovery_approval_boundary", 10, not risky_recovery_without_approval),
        ("fresh_followthrough_token", 5, _looks_like_sha256(recovery_followthrough_token_sha256)),
        ("one_step_boundary_intact", 5, resume_one_step_boundary_ready),
    ]
    rows = [
        {
            "item": item,
            "points": max_points if ready else 0,
            "max_points": max_points,
            "ready": ready,
            "required_before_autonomy_resume": True,
            "authorizes_risky_work": False,
            "authorizes_unreviewed_continuation": False,
        }
        for item, max_points, ready in checks
    ]
    for row in rows:
        if row["item"] == "one_step_boundary_intact":
            row.update(
                {
                    "resume_proof_queue": resume_proof_queue,
                    "resume_proof_queue_count": resume_proof_queue_count,
                    "resume_next_proof_command": resume_next_proof_command,
                    "resume_one_step_boundary_ready": resume_one_step_boundary_ready,
                    "authorizes_local_safe_step": False,
                    "authorizes_approval": False,
                    "authorizes_model_call": False,
                    "authorizes_tool_execution": False,
                    "authorizes_personal_data_read": False,
                    "authorizes_external_side_effect": False,
                }
            )
    return rows


def _autonomy_continuation_readiness_scorecard_rows(
    *,
    resume_ready: bool,
    proposed_next_step: str,
    proposed_verification: str,
    stop_condition: str,
    next_risk_signals: list[str],
    continuation_review_token_sha256: str,
    prior_cycle_ledger_token_sha256: str,
    one_step_contract_ready: bool,
    post_step_queue: list[str],
) -> list[dict[str, Any]]:
    checks = [
        ("ready_resume_gate", 20, resume_ready),
        ("exact_next_step", 15, bool(proposed_next_step)),
        ("post_step_verification_target", 15, bool(proposed_verification)),
        ("stop_condition", 10, bool(stop_condition)),
        ("risky_work_approval_boundary", 15, not next_risk_signals),
        ("fresh_continuation_review_token", 10, _looks_like_sha256(continuation_review_token_sha256)),
        ("prior_cycle_proof_non_authorizing", 5, True),
        ("one_step_contract_non_reusable", 5, one_step_contract_ready),
        ("post_step_proof_queue_ready", 5, bool(post_step_queue)),
    ]
    return [
        {
            "item": item,
            "points": max_points if ready else 0,
            "max_points": max_points,
            "ready": ready,
            "required_before_one_local_safe_step": True,
            "authorizes_risky_work": False,
            "authorizes_batching": False,
            "authorizes_followup_without_closure": False,
            "prior_cycle_ledger_token_authorizes_action": False,
            "prior_cycle_ledger_token_supplied": bool(prior_cycle_ledger_token_sha256),
        }
        for item, max_points, ready in checks
    ]


def _autonomy_step_closure_readiness_scorecard_rows(
    *,
    pre_step_permission_ready: bool,
    completed_step_matches_proposed: bool,
    post_step_verification_matches_proposed: bool,
    post_step_receipt_hash_matches_file: bool,
    post_step_checkpoint_hash_matches_file: bool,
    execution_health: str,
    execution_audit: str,
    after_action_learning: str,
    fresh_contract_enforced: bool,
    closure_ready: bool,
) -> list[dict[str, Any]]:
    checks = [
        ("pre_step_permission_ready", 12, pre_step_permission_ready),
        ("completed_step_matches_allowed", 12, completed_step_matches_proposed),
        ("post_step_verification_matches_target", 12, post_step_verification_matches_proposed),
        ("post_step_receipt_hash_binding", 12, post_step_receipt_hash_matches_file),
        ("post_step_checkpoint_hash_binding", 12, post_step_checkpoint_hash_matches_file),
        ("execution_health_supplied", 10, bool(execution_health)),
        ("execution_audit_supplied", 10, bool(execution_audit)),
        ("after_action_learning_supplied", 10, bool(after_action_learning)),
        ("fresh_review_contract_non_reusable", 10, fresh_contract_enforced and closure_ready),
    ]
    return [
        {
            "item": item,
            "points": max_points if ready else 0,
            "max_points": max_points,
            "ready": ready,
            "required_before_next_continuation_review": True,
            "authorizes_action": False,
            "authorizes_risky_work": False,
            "authorizes_followup_without_fresh_review": False,
        }
        for item, max_points, ready in checks
    ]


_RECOVERY_CLOSURE_REQUIRED_EVIDENCE = (
    "reviewed local-safe step",
    "verification evidence",
    "checkpoint recovery receipt",
    "readable checkpoint recovery receipt",
    "checkpoint recovery receipt sha256",
    "checkpoint recovery receipt sha256 matches file",
    "fresh work-block checkpoint",
    "readable fresh work-block checkpoint",
    "fresh work-block checkpoint sha256",
    "fresh work-block checkpoint sha256 matches file",
    "stop condition",
    "approval boundary",
)


def _recovery_closure_gate(
    *,
    reviewed: bool,
    approved_step: str,
    verification: str,
    receipt_path: str = "",
    receipt_sha256: str = "",
    receipt_file_sha256: str = "",
    receipt_hash_matches_file: bool = False,
    checkpoint_path: str = "",
    checkpoint_sha256: str = "",
    checkpoint_file_sha256: str = "",
    checkpoint_hash_matches_file: bool = False,
    stop_condition: str = "",
    risk_signals: list[str] | None = None,
    approval_reference_present: bool = False,
    blockers: str = "",
) -> dict[str, Any]:
    signals = risk_signals or []
    missing: list[str] = []
    if not reviewed:
        missing.append("reviewed=true")
    if not approved_step:
        missing.append("reviewed local-safe step")
    if not verification:
        missing.append("verification evidence")
    if not receipt_path:
        missing.append("checkpoint recovery receipt")
    elif not receipt_file_sha256:
        missing.append("readable checkpoint recovery receipt")
    if not _looks_like_sha256(receipt_sha256):
        missing.append("valid checkpoint recovery receipt sha256")
    elif not receipt_hash_matches_file:
        missing.append("checkpoint recovery receipt sha256 matches file")
    if not checkpoint_path:
        missing.append("fresh work-block checkpoint")
    elif not checkpoint_file_sha256:
        missing.append("readable fresh work-block checkpoint")
    if not _looks_like_sha256(checkpoint_sha256):
        missing.append("valid fresh work-block checkpoint sha256")
    elif not checkpoint_hash_matches_file:
        missing.append("fresh work-block checkpoint sha256 matches file")
    if not stop_condition:
        missing.append("stop condition")
    if signals and not approval_reference_present:
        missing.append("approval reference for risky recovery step")
    if blockers.strip().lower() not in {"", "none", "none reported", "no blockers"}:
        missing.append("remaining blockers cleared")

    normal_followthrough_allowed = not missing
    state = "READY_FOR_NORMAL_FOLLOWTHROUGH_REVIEW" if normal_followthrough_allowed else "HELD_BEFORE_NORMAL_FOLLOWTHROUGH"
    return {
        "state": state,
        "normal_followthrough_allowed": normal_followthrough_allowed,
        "missing": missing,
        "missing_count": len(missing),
        "required_evidence": list(_RECOVERY_CLOSURE_REQUIRED_EVIDENCE),
        "approval_boundary": "future shell/code, computer control, personal-data, external-side-effect, destructive, or otherwise risky recovery steps still require approval readiness, a last-look approval packet, approval chain proof, and explicit approval",
        "next_safe_command": "resume normal follow-through from the fresh checkpoint" if normal_followthrough_allowed else "checkpoint recovery execute reviewed=true step=<reviewed local-safe step> verification=<evidence>",
    }


_CHECKPOINT_RECOVERY_EXECUTE_HANDOFF_FALSE_FLAGS = (
    "contains_raw_step",
    "contains_raw_verification",
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
)

_RECOVERY_STEP_APPROVAL_BOUNDARY_FALSE_FLAGS = (
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
)


def _checkpoint_recovery_execute_handoff_ready(metadata: dict[str, Any]) -> bool:
    handoff = metadata.get("checkpoint_recovery_execute_handoff") or {}
    if not isinstance(handoff, dict):
        return False
    if handoff.get("handoff_ready") is not True:
        return False
    if handoff.get("state") != "HELD_BEFORE_REVIEW":
        return False
    if metadata.get("checkpoint_recovery_execute_handoff_state") != handoff.get("state"):
        return False
    if handoff.get("normal_followthrough_allowed") is not False:
        return False
    if metadata.get("checkpoint_recovery_execute_handoff_normal_followthrough_allowed") is not False:
        return False
    if "reviewed" not in metadata:
        return False
    reviewed = bool(metadata.get("reviewed"))
    if handoff.get("reviewed") != reviewed:
        return False
    if metadata.get("checkpoint_recovery_execute_handoff_reviewed") != reviewed:
        return False
    if "recovery_followthrough_gate_state" not in metadata:
        return False
    followthrough_gate_state = str(metadata.get("recovery_followthrough_gate_state") or "")
    if not followthrough_gate_state:
        return False
    if handoff.get("recovery_followthrough_gate_state") != followthrough_gate_state:
        return False
    if (
        metadata.get("checkpoint_recovery_execute_handoff_recovery_followthrough_gate_state")
        != followthrough_gate_state
    ):
        return False

    missing = list(metadata.get("missing_fields") or [])
    if handoff.get("missing_fields") != missing:
        return False
    if metadata.get("checkpoint_recovery_execute_handoff_missing_fields") != missing:
        return False
    if handoff.get("missing_field_count") != len(missing):
        return False
    if metadata.get("checkpoint_recovery_execute_handoff_missing_field_count") != len(missing):
        return False

    closure_missing = list(metadata.get("recovery_closure_missing") or [])
    if handoff.get("recovery_closure_missing") != closure_missing:
        return False
    if metadata.get("checkpoint_recovery_execute_handoff_recovery_closure_missing") != closure_missing:
        return False
    if handoff.get("recovery_closure_missing_count") != len(closure_missing):
        return False
    if metadata.get("recovery_closure_missing_count") != len(closure_missing):
        return False
    if (
        metadata.get("checkpoint_recovery_execute_handoff_recovery_closure_missing_count")
        != len(closure_missing)
    ):
        return False

    required_evidence = list(
        metadata.get("recovery_closure_required_evidence") or []
    )
    if required_evidence != list(_RECOVERY_CLOSURE_REQUIRED_EVIDENCE):
        return False
    if handoff.get("recovery_closure_required_evidence") != required_evidence:
        return False
    if metadata.get("checkpoint_recovery_execute_handoff_recovery_closure_required_evidence") != required_evidence:
        return False
    if handoff.get("recovery_closure_required_evidence_count") != len(required_evidence):
        return False
    if metadata.get("recovery_closure_required_evidence_count") != len(required_evidence):
        return False
    if (
        metadata.get("checkpoint_recovery_execute_handoff_recovery_closure_required_evidence_count")
        != len(required_evidence)
    ):
        return False

    recovery_closure_proof_queue = list(
        metadata.get("recovery_closure_proof_queue") or []
    )
    if not recovery_closure_proof_queue:
        return False
    if handoff.get("recovery_closure_proof_queue") != recovery_closure_proof_queue:
        return False
    if metadata.get("checkpoint_recovery_execute_handoff_recovery_closure_proof_queue") != recovery_closure_proof_queue:
        return False
    if handoff.get("recovery_closure_proof_queue_count") != len(recovery_closure_proof_queue):
        return False
    if metadata.get("recovery_closure_proof_queue_count") != len(recovery_closure_proof_queue):
        return False
    if (
        metadata.get("checkpoint_recovery_execute_handoff_recovery_closure_proof_queue_count")
        != len(recovery_closure_proof_queue)
    ):
        return False
    next_recovery_closure_command = str(metadata.get("recovery_closure_next_proof_command") or "")
    if not next_recovery_closure_command or next_recovery_closure_command != recovery_closure_proof_queue[0]:
        return False
    if handoff.get("recovery_closure_next_proof_command") != next_recovery_closure_command:
        return False
    if (
        metadata.get("checkpoint_recovery_execute_handoff_recovery_closure_next_proof_command")
        != next_recovery_closure_command
    ):
        return False
    if handoff.get("recovery_closure_proof_queue_ready") is not True:
        return False
    if metadata.get("recovery_closure_proof_queue_ready") is not True:
        return False
    if metadata.get("checkpoint_recovery_execute_handoff_recovery_closure_proof_queue_ready") is not True:
        return False

    approval_boundary = str(metadata.get("recovery_closure_approval_boundary") or "")
    if "explicit approval" not in approval_boundary:
        return False
    if handoff.get("recovery_closure_approval_boundary") != approval_boundary:
        return False
    if metadata.get("checkpoint_recovery_execute_handoff_recovery_closure_approval_boundary") != approval_boundary:
        return False

    next_safe_command = str(metadata.get("recovery_closure_next_safe_command") or "")
    if not next_safe_command or handoff.get("next_safe_command") != next_safe_command:
        return False
    if metadata.get("checkpoint_recovery_execute_handoff_next_safe_command") != next_safe_command:
        return False

    if "risky_recovery_signals" not in metadata:
        return False
    risk_signals = list(metadata.get("risky_recovery_signals") or [])
    if handoff.get("risky_recovery_signals") != risk_signals:
        return False
    if metadata.get("checkpoint_recovery_execute_handoff_risky_recovery_signals") != risk_signals:
        return False
    if handoff.get("risky_recovery_signal_count") != len(risk_signals):
        return False
    if "risky_recovery_signal_count" not in metadata:
        return False
    if metadata.get("risky_recovery_signal_count") != len(risk_signals):
        return False
    if metadata.get("checkpoint_recovery_execute_handoff_risky_recovery_signal_count") != len(risk_signals):
        return False
    if "approval_reference_provided" not in metadata:
        return False
    approval_reference_provided = bool(metadata.get("approval_reference_provided"))
    if handoff.get("approval_reference_provided") != approval_reference_provided:
        return False
    if (
        metadata.get("checkpoint_recovery_execute_handoff_approval_reference_provided")
        != approval_reference_provided
    ):
        return False
    if "recovery_step_approval_required_before_recovery" not in metadata:
        return False
    approval_required = bool(metadata.get("recovery_step_approval_required_before_recovery"))
    if approval_required != bool(risk_signals and not approval_reference_provided):
        return False
    if handoff.get("recovery_step_approval_required_before_recovery") != approval_required:
        return False
    if (
        metadata.get("checkpoint_recovery_execute_handoff_recovery_step_approval_required_before_recovery")
        != approval_required
    ):
        return False
    expected_missing = []
    if not reviewed:
        expected_missing.append("reviewed=true")
    if str(metadata.get("recovery_step_sha256") or "") == _text_sha256(""):
        expected_missing.append("approved_step")
    if str(metadata.get("recovery_verification_sha256") or "") == _text_sha256(""):
        expected_missing.append("verification")
    if approval_required:
        expected_missing.append("approval_reference_for_risky_recovery_step")
    if missing != expected_missing:
        return False

    boundary_rows = list(metadata.get("recovery_step_approval_boundary_rows") or [])
    if len(boundary_rows) != 7:
        return False
    if handoff.get("recovery_step_approval_boundary_rows") != boundary_rows:
        return False
    if (
        metadata.get("checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_rows")
        != boundary_rows
    ):
        return False
    if handoff.get("recovery_step_approval_boundary_row_count") != len(boundary_rows):
        return False
    if metadata.get("recovery_step_approval_boundary_row_count") != len(boundary_rows):
        return False
    if (
        metadata.get("checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_row_count")
        != len(boundary_rows)
    ):
        return False
    approval_reference = ""
    if approval_reference_provided:
        approval_reference = str(metadata.get("approval_reference") or "")
        if not approval_reference:
            return False
    if "approval_reference" not in handoff:
        return False
    if handoff.get("approval_reference") != approval_reference:
        return False

    if "recovery_step_approval_proof_queue" not in metadata:
        return False
    if "checkpoint_recovery_execute_handoff_recovery_step_approval_proof_queue" not in metadata:
        return False
    if "recovery_step_approval_proof_queue" not in handoff:
        return False
    if "recovery_step_approval_proof_queue_count" not in metadata:
        return False
    if "checkpoint_recovery_execute_handoff_recovery_step_approval_proof_queue_count" not in metadata:
        return False
    if "recovery_step_approval_proof_queue_count" not in handoff:
        return False
    proof_queue = list(
        metadata.get("recovery_step_approval_proof_queue") or []
    )
    expected_proof_queue = (
        _approval_proof_queue_for_risky_work(
            followup_command="checkpoint recovery execute: <reviewed local-safe recovery step after approval proof>"
        )
        if approval_required
        else []
    )
    if proof_queue != expected_proof_queue:
        return False
    if handoff.get("recovery_step_approval_proof_queue") != proof_queue:
        return False
    if metadata.get("checkpoint_recovery_execute_handoff_recovery_step_approval_proof_queue") != proof_queue:
        return False
    if handoff.get("recovery_step_approval_proof_queue_count") != len(proof_queue):
        return False
    if metadata.get("recovery_step_approval_proof_queue_count") != len(proof_queue):
        return False
    if metadata.get("checkpoint_recovery_execute_handoff_recovery_step_approval_proof_queue_count") != len(proof_queue):
        return False
    next_approval = proof_queue[0] if proof_queue else ""
    if "recovery_step_next_approval_proof_command" not in handoff:
        return False
    if "recovery_step_next_approval_proof_command" not in metadata:
        return False
    if "checkpoint_recovery_execute_handoff_recovery_step_next_approval_proof_command" not in metadata:
        return False
    if handoff.get("recovery_step_next_approval_proof_command", "") != next_approval:
        return False
    if metadata.get("recovery_step_next_approval_proof_command") != next_approval:
        return False
    if metadata.get("checkpoint_recovery_execute_handoff_recovery_step_next_approval_proof_command") != next_approval:
        return False

    boundary_token = str(metadata.get("recovery_step_approval_boundary_token_sha256") or "")
    if not _looks_like_sha256(boundary_token):
        return False
    if handoff.get("recovery_step_approval_boundary_token_sha256") != boundary_token:
        return False
    if metadata.get("checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_token_sha256") != boundary_token:
        return False
    if metadata.get("recovery_step_approval_boundary_token_present") is not True:
        return False
    if metadata.get("checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_token_present") is not True:
        return False
    if handoff.get("recovery_step_approval_boundary_token_present") is not True:
        return False

    handoff_token = str(metadata.get("checkpoint_recovery_execute_handoff_token_sha256") or "")
    if not _looks_like_sha256(handoff_token):
        return False
    if handoff.get("handoff_token_sha256") != handoff_token:
        return False
    if handoff.get("handoff_token_present") is not True:
        return False
    if metadata.get("checkpoint_recovery_execute_handoff_token_present") is not True:
        return False
    recomputed_handoff_token = _checkpoint_recovery_execute_handoff_token_sha256(handoff)
    if recomputed_handoff_token != handoff_token:
        return False

    for key in (
        "recovery_step_sha256",
        "recovery_verification_sha256",
        "checkpoint_recovery_execute_handoff_recovery_step_sha256",
        "checkpoint_recovery_execute_handoff_recovery_verification_sha256",
    ):
        if key not in metadata:
            return False
    step_sha256 = str(metadata.get("recovery_step_sha256") or "")
    verification_sha256 = str(metadata.get("recovery_verification_sha256") or "")
    if step_sha256 and not _looks_like_sha256(step_sha256):
        return False
    if verification_sha256 and not _looks_like_sha256(verification_sha256):
        return False
    if handoff.get("recovery_step_sha256") != step_sha256:
        return False
    if handoff.get("recovery_verification_sha256") != verification_sha256:
        return False
    if metadata.get("checkpoint_recovery_execute_handoff_recovery_step_sha256") != step_sha256:
        return False
    if metadata.get("checkpoint_recovery_execute_handoff_recovery_verification_sha256") != verification_sha256:
        return False

    boundary_ready = _recovery_step_approval_boundary_ready(
        token_sha256=boundary_token,
        recovery_step_sha256=step_sha256,
        recovery_verification_sha256=verification_sha256,
        approval_proof_queue=proof_queue,
        approval_boundary_rows=boundary_rows,
        approval_reference=approval_reference,
    )
    recomputed_boundary_token = _risky_recovery_step_approval_boundary_token_sha256(
        recovery_step_sha256=step_sha256,
        recovery_verification_sha256=verification_sha256,
        risk_signals=risk_signals,
        approval_proof_queue=proof_queue,
        approval_boundary_rows=boundary_rows,
        approval_reference=approval_reference,
    )
    if recomputed_boundary_token != boundary_token:
        return False
    if handoff.get("recovery_step_approval_boundary_ready") is not boundary_ready:
        return False
    if metadata.get("recovery_step_approval_boundary_ready") is not boundary_ready:
        return False
    if (
        metadata.get("checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_ready")
        is not boundary_ready
    ):
        return False
    boundary_as_prior_proof = _recovery_step_approval_boundary_ready(
        token_sha256=boundary_token,
        recovery_step_sha256=step_sha256,
        recovery_verification_sha256=verification_sha256,
        approval_proof_queue=proof_queue,
        approval_boundary_rows=boundary_rows,
        approval_reference=approval_reference,
        require_prior_proof=True,
    )
    if metadata.get("recovery_step_approval_boundary_as_prior_proof") is not boundary_as_prior_proof:
        return False
    if handoff.get("recovery_step_approval_boundary_as_prior_proof") is not boundary_as_prior_proof:
        return False
    if (
        metadata.get(
            "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_as_prior_proof"
        )
        is not boundary_as_prior_proof
    ):
        return False
    for key in _RECOVERY_STEP_APPROVAL_BOUNDARY_FALSE_FLAGS:
        if key not in handoff:
            return False
        if handoff.get(key) is not False:
            return False
        if metadata.get(key) is not False:
            return False
        flat_key = f"checkpoint_recovery_execute_handoff_{key}"
        if metadata.get(flat_key) is not False:
            return False

    for flag in _CHECKPOINT_RECOVERY_EXECUTE_HANDOFF_FALSE_FLAGS:
        if flag not in metadata:
            return False
        if metadata.get(flag) is not False:
            return False
        if flag not in handoff:
            return False
        if handoff.get(flag) is not False:
            return False
        flat_key = f"checkpoint_recovery_execute_handoff_{flag}"
        if flat_key not in metadata:
            return False
        if metadata.get(flat_key) is not False:
            return False

    timebox_precontinuation_contract = {
        "operator_timebox_precontinuation_contract_ready": _metadata_bool(metadata.get("timebox_review_contract_ready")),
        "operator_timebox_precontinuation_contract_state": metadata.get("timebox_state"),
        "operator_timebox_precontinuation_allows_continuation": metadata.get("can_continue_now"),
        "operator_timebox_precontinuation_blocks_continuation": not _metadata_bool(metadata.get("can_continue_now")),
        "operator_timebox_precontinuation_requires_stop": metadata.get("should_stop_now"),
        "operator_timebox_precontinuation_requires_fresh_timebox_review": True,
    }
    for key, expected in timebox_precontinuation_contract.items():
        if key not in metadata:
            return False
        actual = metadata.get(key)
        if isinstance(expected, bool):
            if actual is not expected:
                return False
        elif actual != expected:
            return False
    if set(metadata.get("operator_timebox_precontinuation_allowed_states") or []) != {
        "STOP_WINDOW_ACTIVE",
        "STOP_TIME_REACHED",
        "HELD_FOR_PARSEABLE_TIMEBOX",
    }:
        return False

    supersession_precontinuation_contract = {
        "operator_supersession_precontinuation_contract_ready": _metadata_bool(
            metadata.get("supersession_contract_ready")
        )
        and _metadata_bool(metadata.get("supersession_token_boundary_ready")),
        "operator_supersession_precontinuation_contract_state": metadata.get("supersession_state"),
        "operator_supersession_precontinuation_token_boundary_ready": metadata.get("supersession_token_boundary_ready"),
        "operator_supersession_precontinuation_allows_continuation": metadata.get(
            "can_continue_under_latest_instruction"
        ),
        "operator_supersession_precontinuation_blocks_continuation": not _metadata_bool(
            metadata.get("can_continue_under_latest_instruction")
        ),
        "operator_supersession_precontinuation_requires_stop": metadata.get("latest_instruction_is_stop"),
        "operator_supersession_precontinuation_requires_fresh_latest_instruction_review": True,
    }
    for key, expected in supersession_precontinuation_contract.items():
        if key not in metadata:
            return False
        actual = metadata.get(key)
        if isinstance(expected, bool):
            if actual is not expected:
                return False
        elif actual != expected:
            return False
    if set(metadata.get("operator_supersession_precontinuation_allowed_states") or []) != {
        "LATEST_INSTRUCTION_READY_TO_GOVERN_CONTINUATION",
        "NEWER_STOP_OR_PAUSE_OVERRIDES_AUTONOMY",
        "HELD_FOR_LATEST_INSTRUCTION_REVIEW",
    }:
        return False
    return True


def _checkpoint_recovery_followthrough_ready(metadata: dict[str, Any]) -> bool:
    if metadata.get("followthrough_state") != "RECOVERY_FOLLOWTHROUGH_READY":
        return False
    if metadata.get("normal_followthrough_allowed") is not True:
        return False
    if metadata.get("recovery_followthrough_gate_state") != "READY_FOR_NORMAL_FOLLOWTHROUGH_REVIEW":
        return False
    if metadata.get("recovery_closure_missing") != [] or metadata.get("recovery_closure_missing_count") != 0:
        return False
    required_evidence = list(metadata.get("recovery_closure_required_evidence") or [])
    if "approval boundary" not in required_evidence:
        return False
    if metadata.get("recovery_closure_required_evidence_count") != len(required_evidence):
        return False
    approval_boundary = str(metadata.get("approval_boundary") or "")
    if metadata.get("approval_boundary_provided") is not True:
        return False
    if "explicit approval" not in approval_boundary:
        return False
    for flag in [
        "reviewed_step_provided",
        "verification_provided",
        "receipt_path_provided",
        "receipt_hash_provided",
        "receipt_file_hash_present",
        "receipt_hash_matches_file",
        "checkpoint_path_provided",
        "checkpoint_hash_provided",
        "checkpoint_file_hash_present",
        "checkpoint_hash_matches_file",
        "recovery_artifact_hashes_present",
        "recovery_artifact_hashes_match_files",
        "recovery_followthrough_token_present",
        "recovery_followthrough_token_boundary_ready",
        "local_safe_recovery_execution_token_present",
        "local_safe_recovery_execution_token_boundary_ready",
        "recovery_execution_required_rows_ready",
        "recovery_execution_scorecard_ready",
        "recovery_execution_readiness_token_present",
        "recovery_execution_readiness_token_boundary_ready",
        "recovery_execution_readiness_as_prior_proof",
    ]:
        if metadata.get(flag) is not True:
            return False
    for key in [
        "receipt_sha256",
        "receipt_file_sha256",
        "checkpoint_sha256",
        "checkpoint_file_sha256",
        "recovery_followthrough_token_sha256",
        "local_safe_recovery_execution_token_sha256",
        "recovery_execution_readiness_token_sha256",
    ]:
        if not _looks_like_sha256(str(metadata.get(key) or "")):
            return False
    if metadata.get("receipt_sha256") != metadata.get("receipt_file_sha256"):
        return False
    if metadata.get("checkpoint_sha256") != metadata.get("checkpoint_file_sha256"):
        return False
    scorecard_rows = list(metadata.get("recovery_execution_scorecard_rows") or [])
    if not _recovery_execution_scorecard_metadata_ready(
        metadata,
        prefix="recovery_execution",
        prior_proof_key="recovery_execution_readiness_as_prior_proof",
    ):
        return False
    recovery_followthrough_rows = list(metadata.get("recovery_followthrough_token_boundary_rows") or [])
    recovery_readiness_rows = list(metadata.get("recovery_execution_readiness_token_boundary_rows") or [])
    if metadata.get("recovery_followthrough_token_boundary_row_count") != len(recovery_followthrough_rows):
        return False
    if metadata.get("recovery_execution_readiness_token_boundary_row_count") != len(recovery_readiness_rows):
        return False
    for flag in [
        "next_recovery_followthrough_requires_new_token",
        "next_recovery_execution_requires_new_local_safe_token",
        "next_recovery_execution_requires_new_readiness_token",
    ]:
        if metadata.get(flag) is not True:
            return False
    if not _recovery_followthrough_token_boundary_ready(
        str(metadata.get("recovery_followthrough_token_sha256") or ""),
        recovery_followthrough_rows,
        expected_source="checkpoint_recovery_followthrough",
    ):
        return False
    if not _carried_local_safe_recovery_execution_token_boundary_ready(
        metadata,
        expected_source="checkpoint_recovery_followthrough",
    ):
        return False
    if not _recovery_execution_readiness_token_boundary_ready(
        str(metadata.get("recovery_execution_readiness_token_sha256") or ""),
        recovery_readiness_rows,
        expected_source="checkpoint_recovery_followthrough",
    ):
        return False
    for flag in [
        "recovery_followthrough_token_reusable_for_future_recovery",
        "recovery_followthrough_token_authorizes_resume_gate",
        "recovery_followthrough_token_authorizes_next_step",
        "recovery_followthrough_token_authorizes_risky_work",
        "recovery_followthrough_token_authorizes_approval",
        "recovery_followthrough_token_authorizes_model_call",
        "recovery_followthrough_token_authorizes_tool_execution",
        "recovery_followthrough_token_authorizes_personal_data_read",
        "recovery_followthrough_token_authorizes_external_side_effect",
        "local_safe_recovery_execution_token_authorizes_resume_gate",
        "local_safe_recovery_execution_token_authorizes_next_step",
        "local_safe_recovery_execution_token_authorizes_risky_work",
        "local_safe_recovery_execution_token_authorizes_approval",
        "local_safe_recovery_execution_token_authorizes_model_call",
        "local_safe_recovery_execution_token_authorizes_tool_execution",
        "local_safe_recovery_execution_token_authorizes_personal_data_read",
        "local_safe_recovery_execution_token_authorizes_external_side_effect",
        "local_safe_recovery_execution_token_reusable_for_future_recovery",
        "recovery_execution_readiness_token_authorizes_resume_gate",
        "recovery_execution_readiness_token_authorizes_next_step",
        "recovery_execution_readiness_token_authorizes_risky_work",
        "recovery_execution_readiness_token_authorizes_approval",
        "recovery_execution_readiness_token_authorizes_model_call",
        "recovery_execution_readiness_token_authorizes_tool_execution",
        "recovery_execution_readiness_token_authorizes_personal_data_read",
        "recovery_execution_readiness_token_authorizes_external_side_effect",
        "recovery_execution_readiness_token_authorizes_unreviewed_followthrough",
        "recovery_execution_readiness_token_reusable_for_future_recovery",
        "recovery_execution_readiness_authorizes_action_now",
        "recovery_execution_readiness_authorizes_risky_work",
        "recovery_execution_readiness_authorizes_unreviewed_followthrough",
        "calls_model",
        "executes_tools",
        "queues_approval",
        "controls_computer",
        "reads_private_data",
        "reads_personal_data",
        "executes_side_effect",
        "external_side_effect",
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "edits_files",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if metadata.get(flag) is not False:
            return False
    return True


def _autonomy_step_closure_gate(
    *,
    continuation_ready: bool,
    continued_step: str,
    completed_step_matches_proposed: bool = True,
    verification_matches_proposed: bool = True,
    verification_evidence: str,
    post_step_receipt_path: str = "",
    post_step_receipt_sha256: str = "",
    post_step_receipt_file_sha256: str = "",
    post_step_receipt_hash_matches_file: bool = False,
    post_step_checkpoint_path: str = "",
    post_step_checkpoint_sha256: str = "",
    post_step_checkpoint_file_sha256: str = "",
    post_step_checkpoint_hash_matches_file: bool = False,
    execution_health: str = "",
    execution_audit: str = "",
    after_action_learning: str = "",
    blockers: str = "",
) -> dict[str, Any]:
    missing: list[str] = []
    if continuation_ready is not True:
        missing.append("ready autonomy continuation execution packet")
    if not continued_step:
        missing.append("completed local-safe continuation step")
    if not completed_step_matches_proposed:
        missing.append("completed step matches allowed continuation step")
    if not verification_evidence:
        missing.append("post-step verification evidence")
    if not verification_matches_proposed:
        missing.append("post-step verification matches allowed verification target")
    if not post_step_receipt_path:
        missing.append("post-step verification receipt")
    elif not post_step_receipt_file_sha256:
        missing.append("readable post-step verification receipt")
    if not _looks_like_sha256(post_step_receipt_sha256):
        missing.append("valid post-step verification receipt sha256")
    elif not post_step_receipt_hash_matches_file:
        missing.append("post-step verification receipt sha256 matches file")
    if not post_step_checkpoint_path:
        missing.append("fresh post-step work-block checkpoint")
    elif not post_step_checkpoint_file_sha256:
        missing.append("readable fresh post-step work-block checkpoint")
    if not _looks_like_sha256(post_step_checkpoint_sha256):
        missing.append("valid fresh post-step checkpoint sha256")
    elif not post_step_checkpoint_hash_matches_file:
        missing.append("fresh post-step checkpoint sha256 matches file")
    if not execution_health:
        missing.append("post-step execution health")
    if not execution_audit:
        missing.append("post-step execution audit")
    if not after_action_learning:
        missing.append("post-step after-action learning")
    if blockers.strip().lower() not in {"", "none", "none reported", "no blockers"}:
        missing.append("remaining blockers cleared")

    ready = not missing
    required_commands = [
        "verification receipt <post-step run id or evidence id>",
        "execution health report",
        "execution audit",
        "after-action learning packet <post-step run id>",
        "work block checkpoint",
        "autonomy continuation execution: <next reviewed local-safe step>",
    ]
    return {
        "state": "AUTONOMY_STEP_CLOSURE_READY_FOR_NEXT_CONTINUATION_REVIEW" if ready else "AUTONOMY_STEP_CLOSURE_HELD",
        "ready_for_next_continuation_review": ready,
        "missing": missing,
        "missing_count": len(missing),
        "required_evidence": [
            "ready autonomy continuation execution packet",
            "completed local-safe continuation step",
            "completed step matches allowed continuation step",
            "post-step verification evidence",
            "post-step verification matches allowed verification target",
            "post-step verification receipt",
            "readable post-step verification receipt",
            "post-step verification receipt hash",
            "post-step verification receipt hash matches file",
            "fresh post-step work-block checkpoint",
            "readable fresh post-step work-block checkpoint",
            "fresh post-step checkpoint hash",
            "fresh post-step checkpoint hash matches file",
            "post-step execution health",
            "post-step execution audit",
            "post-step after-action learning",
            "approval boundary",
        ],
        "required_commands": required_commands,
        "next_required_command": required_commands[0] if missing else "autonomy continuation execution: <next reviewed local-safe step>",
        "approval_boundary": "future shell/code, computer control, personal-data, external-side-effect, destructive, or otherwise risky continuation steps still require approval readiness, a last-look approval packet, approval chain proof, and explicit approval",
    }


def _autonomy_step_closure_ready(metadata: dict[str, Any]) -> bool:
    if metadata.get("closure_state") != "AUTONOMY_STEP_CLOSURE_READY_FOR_NEXT_CONTINUATION_REVIEW":
        return False
    if metadata.get("ready_for_next_continuation_review") is not True:
        return False
    if metadata.get("missing_blockers") != [] or metadata.get("missing_blocker_count") != 0:
        return False
    expected_next_safe_command = "autonomy continuation execution: <next reviewed local-safe step>"
    if metadata.get("next_safe_command") != expected_next_safe_command:
        return False
    required_commands = list(metadata.get("required_commands") or [])
    proof_queue = list(metadata.get("proof_queue") or [])
    if metadata.get("required_command_count") != len(required_commands):
        return False
    if metadata.get("proof_queue_count") != len(proof_queue):
        return False
    if not required_commands or proof_queue != required_commands:
        return False
    if expected_next_safe_command not in required_commands:
        return False
    if metadata.get("next_required_command") != expected_next_safe_command:
        return False
    if metadata.get("next_proof_command") != expected_next_safe_command:
        return False

    true_flags = [
        "timebox_review_contract_ready",
        "supersession_token_boundary_ready",
        "checkpoint_route_boundary_ready",
        "pre_step_one_local_safe_step_allowed",
        "one_step_execution_contract_token_as_prior_proof",
        "recovery_artifact_hashes_present",
        "recovery_artifact_hashes_match_files",
        "recovery_followthrough_token_boundary_ready",
        "local_safe_recovery_execution_token_boundary_ready",
        "recovery_execution_readiness_token_boundary_ready",
        "carried_recovery_execution_required_rows_ready",
        "carried_recovery_execution_scorecard_ready",
        "carried_recovery_execution_as_prior_proof",
        "completed_step_provided",
        "completed_step_matches_proposed",
        "post_step_verification_provided",
        "post_step_verification_matches_proposed",
        "post_step_receipt_path_provided",
        "post_step_receipt_hash_provided",
        "post_step_receipt_file_hash_present",
        "post_step_receipt_hash_matches_file",
        "post_step_checkpoint_path_provided",
        "post_step_checkpoint_hash_provided",
        "post_step_checkpoint_file_hash_present",
        "post_step_checkpoint_hash_matches_file",
        "post_step_artifact_hashes_present",
        "post_step_artifact_hashes_match_files",
        "execution_health_reviewed",
        "execution_audit_reviewed",
        "after_action_learning_reviewed",
        "blockers_cleared",
        "fresh_continuation_review_contract_enforced",
        "fresh_continuation_review_boundary_token_present",
        "fresh_continuation_review_boundary_token_ready",
        "step_closure_receipt_token_present",
        "step_closure_receipt_boundary_ready",
        "step_closure_readiness_required_rows_ready",
        "step_closure_readiness_scorecard_ready",
        "next_continuation_requires_fresh_operator_timebox",
        "next_continuation_requires_fresh_checkpoint",
        "next_continuation_requires_fresh_recovery_cockpit",
        "next_continuation_requires_fresh_local_safe_step",
        "next_continuation_requires_fresh_review_token",
    ]
    for flag in true_flags:
        if metadata.get(flag) is not True:
            return False
    if not _carried_operator_timebox_review_contract_ready(metadata):
        return False
    if not _carried_operator_supersession_token_boundary_ready(metadata):
        return False
    if not _carried_checkpoint_route_boundary_ready(metadata):
        return False
    if not _carried_one_step_execution_contract_ready(metadata):
        return False

    sha_fields = [
        "timebox_receipt_sha256",
        "awake_guard_token_sha256",
        "supersession_token_sha256",
        "checkpoint_route_token_sha256",
        "receipt_sha256",
        "receipt_file_sha256",
        "checkpoint_sha256",
        "recovery_checkpoint_file_sha256",
        "recovery_followthrough_token_sha256",
        "local_safe_recovery_execution_token_sha256",
        "recovery_execution_readiness_token_sha256",
        "one_step_execution_contract_token_sha256",
        "proposed_next_step_sha256",
        "completed_step_sha256",
        "proposed_verification_sha256",
        "post_step_verification_sha256",
        "continuation_review_token_sha256",
        "fresh_continuation_review_boundary_token_sha256",
        "step_closure_receipt_token_sha256",
        "post_step_receipt_sha256",
        "post_step_receipt_file_sha256",
        "post_step_checkpoint_sha256",
        "post_step_checkpoint_file_sha256",
    ]
    for field in sha_fields:
        if not _looks_like_sha256(str(metadata.get(field) or "")):
            return False
    if metadata.get("receipt_sha256") != metadata.get("receipt_file_sha256"):
        return False
    if metadata.get("checkpoint_sha256") != metadata.get("recovery_checkpoint_file_sha256"):
        return False
    if metadata.get("proposed_next_step_sha256") != metadata.get("completed_step_sha256"):
        return False
    if metadata.get("proposed_verification_sha256") != metadata.get("post_step_verification_sha256"):
        return False
    if metadata.get("post_step_receipt_sha256") != metadata.get("post_step_receipt_file_sha256"):
        return False
    if metadata.get("post_step_checkpoint_sha256") != metadata.get("post_step_checkpoint_file_sha256"):
        return False

    step_rows = metadata.get("step_closure_readiness_scorecard_rows")
    if not isinstance(step_rows, list) or len(step_rows) != 9:
        return False
    if metadata.get("step_closure_readiness_scorecard_row_count") != len(step_rows):
        return False
    if metadata.get("step_closure_readiness_score") != 100 or metadata.get("step_closure_readiness_max_score") != 100:
        return False
    if not _autonomy_step_closure_readiness_scorecard_ready(step_rows):
        return False

    recovery_rows = metadata.get("carried_recovery_execution_scorecard_rows")
    if not isinstance(recovery_rows, list) or len(recovery_rows) != 8:
        return False
    if metadata.get("carried_recovery_execution_scorecard_row_count") != len(recovery_rows):
        return False
    if metadata.get("carried_recovery_execution_score") != 100 or metadata.get("carried_recovery_execution_max_score") != 100:
        return False
    if not _recovery_execution_readiness_scorecard_ready(recovery_rows):
        return False

    fresh_rows = metadata.get("fresh_continuation_review_contract_rows")
    if not isinstance(fresh_rows, list) or len(fresh_rows) != 6:
        return False
    if metadata.get("fresh_continuation_review_contract_row_count") != len(fresh_rows):
        return False
    if not _fresh_continuation_review_contract_ready(fresh_rows):
        return False
    if not _fresh_review_boundary_metadata_ready(
        metadata,
        token_prefix="fresh_continuation_review_boundary_token",
        authority_prefix="fresh_continuation_review_boundary",
        next_fresh_token_key="next_review_requires_new_fresh_continuation_review_boundary_token",
        expected_source="autonomy_step_closure",
    ):
        return False
    if not _continuation_review_token_metadata_ready(
        metadata,
        reusable_key="continuation_review_token_reusable_for_next_review",
    ):
        return False
    if not _step_closure_receipt_metadata_ready(
        metadata,
        token_prefix="step_closure_receipt_token",
        boundary_prefix="step_closure_receipt_boundary",
        authority_prefix="step_closure_receipt",
        expected_source="autonomy_step_closure",
    ):
        return False
    if not _carried_local_safe_recovery_execution_token_boundary_ready(
        metadata,
        expected_source="autonomy_step_closure",
    ):
        return False
    if not _carried_recovery_followthrough_token_boundary_ready(
        metadata,
        expected_source="checkpoint_recovery_followthrough",
    ):
        return False
    if not _carried_recovery_execution_readiness_token_boundary_ready(
        metadata,
        expected_source="checkpoint_recovery_followthrough",
    ):
        return False
    if not _carried_next_step_approval_boundary_ready_from_metadata(metadata):
        return False
    if not _prior_cycle_ledger_token_metadata_ready(
        metadata,
        reusable_key="prior_cycle_ledger_token_reusable_for_this_closure",
        authority_action_key="prior_cycle_ledger_proof_authorizes_post_step_closure",
        expected_source="autonomy_step_closure",
    ):
        return False
    if not _one_step_execution_contract_ready(
        objective=str(metadata.get("objective") or ""),
        continuation_state=str(metadata.get("continuation_state") or ""),
        resume_gate_state=str(metadata.get("resume_gate_state") or ""),
        continuation_ready=metadata.get("one_step_execution_contract_ready"),
        one_step_execution_contract_rows=list(metadata.get("one_step_execution_contract_rows") or []),
        one_step_execution_contract_token_sha256=str(metadata.get("one_step_execution_contract_token_sha256") or ""),
        timebox_receipt_sha256=str(metadata.get("timebox_receipt_sha256") or ""),
        awake_guard_token_sha256=str(metadata.get("awake_guard_token_sha256") or ""),
        supersession_token_sha256=str(metadata.get("supersession_token_sha256") or ""),
        checkpoint_route_token_sha256=str(metadata.get("checkpoint_route_token_sha256") or ""),
        recovery_followthrough_token_sha256=str(metadata.get("recovery_followthrough_token_sha256") or ""),
        local_safe_recovery_execution_token_sha256=str(metadata.get("local_safe_recovery_execution_token_sha256") or ""),
        prior_cycle_ledger_token_sha256=str(metadata.get("prior_cycle_ledger_token_sha256") or ""),
        continuation_review_token_sha256=str(metadata.get("continuation_review_token_sha256") or ""),
        proposed_next_step_sha256=str(metadata.get("proposed_next_step_sha256") or ""),
        proposed_verification_sha256=str(metadata.get("proposed_verification_sha256") or ""),
        risk_signals=[str(signal) for signal in (metadata.get("proposed_next_step_risk_signals") or [])],
        timebox_review_contract_ready=metadata.get("timebox_review_contract_ready"),
        timebox_review_contract_rows=list(metadata.get("timebox_review_contract_rows") or []),
        post_step_proof_queue=[str(command) for command in (metadata.get("continuation_post_step_proof_queue") or [])],
    ):
        return False

    false_flags = [
        "action_allowed_now",
        "executable_tool_action_emitted",
        "can_emit_executable_tool_action",
        "can_auto_execute_now",
        "timebox_authorizes_execution",
        "timebox_authorizes_local_safe_step",
        "timebox_authorizes_risky_work",
        "timebox_authorizes_approval",
        "timebox_authorizes_timebox_reuse",
        "timebox_authorizes_model_call",
        "timebox_authorizes_tool_execution",
        "timebox_authorizes_personal_data_read",
        "timebox_authorizes_external_side_effect",
        "timebox_reusable_for_next_step",
        "awake_guard_authorizes_os_wake_lock",
        "awake_guard_authorizes_shell_execution",
        "awake_guard_authorizes_computer_control",
        "awake_guard_authorizes_approval",
        "awake_guard_authorizes_model_call",
        "awake_guard_authorizes_tool_execution",
        "awake_guard_authorizes_personal_data_read",
        "awake_guard_authorizes_external_side_effect",
        "awake_guard_reusable_for_next_timebox",
        "awake_guard_caffeinate_command_authorized",
        "awake_guard_keep_awake_command_authorized",
        "awake_guard_authorizes_unattended_execution",
        "awake_guard_authorizes_continuation_window",
        "awake_guard_reusable_as_execution_permission",
        "supersession_token_authorizes_execution",
        "supersession_token_authorizes_local_safe_step",
        "supersession_token_authorizes_risky_work",
        "supersession_token_authorizes_approval",
        "supersession_token_authorizes_recovery_followthrough",
        "supersession_token_authorizes_timebox_override",
        "supersession_token_authorizes_goal_override",
        "supersession_token_authorizes_model_call",
        "supersession_token_authorizes_tool_execution",
        "supersession_token_authorizes_personal_data_read",
        "supersession_token_authorizes_external_side_effect",
        "supersession_token_reusable_for_next_review",
        "supersession_token_reusable_for_next_timebox",
        "one_step_execution_contract_token_authorizes_action_now",
        "one_step_execution_contract_token_authorizes_risky_work",
        "one_step_execution_contract_token_authorizes_unreviewed_followthrough",
        "one_step_execution_contract_token_authorizes_batching",
        "one_step_execution_contract_token_reusable_for_next_step",
        "checkpoint_route_authorizes_continuation",
        "checkpoint_route_authorizes_local_safe_step",
        "checkpoint_route_authorizes_risky_work",
        "checkpoint_route_authorizes_approval",
        "checkpoint_route_authorizes_recovery_followthrough",
        "checkpoint_route_authorizes_checkpoint_reuse",
        "checkpoint_route_authorizes_model_call",
        "checkpoint_route_authorizes_tool_execution",
        "checkpoint_route_authorizes_personal_data_read",
        "checkpoint_route_authorizes_external_side_effect",
        "checkpoint_route_reusable_for_next_review",
        "checkpoint_route_reusable_for_next_checkpoint",
        "recovery_followthrough_token_reusable_for_future_recovery",
        "recovery_followthrough_token_authorizes_resume_gate",
        "recovery_followthrough_token_authorizes_next_step",
        "recovery_followthrough_token_authorizes_risky_work",
        "recovery_followthrough_token_authorizes_approval",
        "recovery_followthrough_token_authorizes_model_call",
        "recovery_followthrough_token_authorizes_tool_execution",
        "recovery_followthrough_token_authorizes_personal_data_read",
        "recovery_followthrough_token_authorizes_external_side_effect",
        "local_safe_recovery_execution_token_authorizes_resume_gate",
        "local_safe_recovery_execution_token_authorizes_next_step",
        "local_safe_recovery_execution_token_authorizes_risky_work",
        "local_safe_recovery_execution_token_authorizes_approval",
        "local_safe_recovery_execution_token_authorizes_model_call",
        "local_safe_recovery_execution_token_authorizes_tool_execution",
        "local_safe_recovery_execution_token_authorizes_personal_data_read",
        "local_safe_recovery_execution_token_authorizes_external_side_effect",
        "local_safe_recovery_execution_token_reusable_for_future_recovery",
        "recovery_execution_readiness_token_authorizes_resume_gate",
        "recovery_execution_readiness_token_authorizes_next_step",
        "recovery_execution_readiness_token_authorizes_risky_work",
        "recovery_execution_readiness_token_authorizes_approval",
        "recovery_execution_readiness_token_authorizes_model_call",
        "recovery_execution_readiness_token_authorizes_tool_execution",
        "recovery_execution_readiness_token_authorizes_personal_data_read",
        "recovery_execution_readiness_token_authorizes_external_side_effect",
        "recovery_execution_readiness_token_authorizes_unreviewed_followthrough",
        "recovery_execution_readiness_token_reusable_for_future_recovery",
        "carried_recovery_execution_authorizes_action_now",
        "carried_recovery_execution_authorizes_risky_work",
        "carried_recovery_execution_authorizes_unreviewed_followthrough",
        "carried_recovery_execution_authorizes_model_call",
        "carried_recovery_execution_authorizes_tool_execution",
        "carried_recovery_execution_authorizes_personal_data_read",
        "carried_recovery_execution_authorizes_external_side_effect",
        "carried_next_step_approval_boundary_authorizes_action_now",
        "carried_next_step_approval_boundary_authorizes_risky_work",
        "carried_next_step_approval_boundary_authorizes_unreviewed_followthrough",
        "carried_next_step_approval_boundary_authorizes_approval",
        "carried_next_step_approval_boundary_authorizes_model_call",
        "carried_next_step_approval_boundary_authorizes_tool_execution",
        "carried_next_step_approval_boundary_authorizes_personal_data_read",
        "carried_next_step_approval_boundary_authorizes_external_side_effect",
        "carried_next_step_approval_boundary_authorizes_timebox_reuse",
        "carried_next_step_approval_boundary_reusable_for_next_review",
        "carried_next_step_approval_boundary_reusable_for_recovery_review",
        "continuation_review_token_reusable_for_next_review",
        "fresh_continuation_review_boundary_authorizes_action_now",
        "fresh_continuation_review_boundary_authorizes_local_safe_step",
        "fresh_continuation_review_boundary_authorizes_risky_work",
        "fresh_continuation_review_boundary_authorizes_unreviewed_followthrough",
        "fresh_continuation_review_boundary_authorizes_timebox_reuse",
        "fresh_continuation_review_boundary_authorizes_checkpoint_reuse",
        "fresh_continuation_review_boundary_authorizes_token_reuse",
        "fresh_continuation_review_boundary_authorizes_model_call",
        "fresh_continuation_review_boundary_authorizes_tool_execution",
        "fresh_continuation_review_boundary_authorizes_personal_data_read",
        "fresh_continuation_review_boundary_authorizes_external_side_effect",
        "fresh_continuation_review_boundary_reusable_for_next_review",
        "fresh_continuation_review_boundary_reusable_for_next_cycle",
        "step_closure_receipt_authorizes_action_now",
        "step_closure_receipt_authorizes_local_safe_step",
        "step_closure_receipt_authorizes_risky_work",
        "step_closure_receipt_authorizes_new_cycle",
        "step_closure_receipt_authorizes_unreviewed_followthrough",
        "step_closure_receipt_authorizes_timebox_reuse",
        "step_closure_receipt_authorizes_checkpoint_reuse",
        "step_closure_receipt_authorizes_model_call",
        "step_closure_receipt_authorizes_tool_execution",
        "step_closure_receipt_authorizes_personal_data_read",
        "step_closure_receipt_authorizes_external_side_effect",
        "step_closure_receipt_reusable_for_next_review",
        "step_closure_receipt_reusable_for_next_cycle",
        "prior_step_closure_authorizes_followup",
        "prior_step_closure_reusable_for_next_step",
        "prior_cycle_ledger_token_reusable_for_this_closure",
        "prior_cycle_ledger_proof_authorizes_post_step_closure",
        "prior_cycle_ledger_proof_authorizes_model_call",
        "prior_cycle_ledger_proof_authorizes_tool_execution",
        "prior_cycle_ledger_proof_authorizes_personal_data_read",
        "prior_cycle_ledger_proof_authorizes_external_side_effect",
        "calls_model",
        "executes_tools",
        "queues_approval",
        "controls_computer",
        "reads_private_data",
        "reads_personal_data",
        "executes_side_effect",
        "external_side_effect",
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "edits_files",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]
    for flag in false_flags:
        if metadata.get(flag) is not False:
            return False
    return True


def _checkpoint_freshness_label(age_minutes: int | None) -> str:
    if age_minutes is None:
        return "missing"
    if age_minutes < CHECKPOINT_REVIEW_MINUTES:
        return "fresh"
    if age_minutes < CHECKPOINT_STALE_MINUTES:
        return "review_again"
    return "stale"


def _parse_iso_datetime(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _file_sha256(path: str) -> str:
    if not path:
        return ""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def _first_approval_handoff(approvals: list[Any]) -> dict[str, Any]:
    if not approvals:
        return {
            "approval_handoff_pending_count": 0,
            "approval_handoff_first_id": None,
            "approval_handoff_first_tool": "",
            "approval_handoff_readiness_command": "",
            "approval_handoff_last_look_command": "",
            "approval_handoff_proof_command": "",
            "approval_handoff_approve_command": "",
            "approval_handoff_dismiss_command": "",
        }
    first = approvals[0]
    approval_id = int(first["id"])
    return {
        "approval_handoff_pending_count": len(approvals),
        "approval_handoff_first_id": approval_id,
        "approval_handoff_first_tool": str(first["tool_name"]),
        "approval_handoff_readiness_command": f"approval readiness {approval_id}",
        "approval_handoff_last_look_command": f"approval packet {approval_id}",
        "approval_handoff_proof_command": f"approval chain proof {approval_id}",
        "approval_handoff_approve_command": f"approve approval {approval_id}",
        "approval_handoff_dismiss_command": f"dismiss approval {approval_id}",
    }


def _tool_run_approval_id(row: dict[str, Any]) -> int | None:
    try:
        approval_id = int(row.get("approval_id") or 0)
    except (TypeError, ValueError):
        return None
    return approval_id if approval_id > 0 else None


def _approval_review_commands_for_tool_run(row: dict[str, Any]) -> list[str]:
    approval_id = _tool_run_approval_id(row)
    if not approval_id:
        return ["approval review"]
    return [
        f"approval readiness {approval_id}",
        f"approval packet {approval_id}",
        f"approval chain proof {approval_id}",
        f"verification receipt <approved run id from approval chain proof {approval_id}>",
    ]


def make_continuity_tools(store: MemoryStore, vault: ObsidianVault):
    def _priority_goal_lines() -> list[str]:
        return [
            "Priority goal:",
            "- Finish Jarvis V2 as an AI agent harness toward a fully functional AGI-like personal assistant.",
            "- Keep future work focused on command-first UX, routing, memory/state, tools, approvals, audit, recovery, verification, and learning.",
            "- This priority does not override approval gates for risky work.",
            "- This priority does not override the operator's explicit stop times, work windows, pause commands, or newer instructions.",
        ]

    def _build_proof_queue(
        *,
        approvals: list[Any],
        failed_runs: list[Any],
        approval_held_runs: list[Any] | None = None,
        tasks: list[Any] | None = None,
        goals: list[Any] | None = None,
    ) -> list[str]:
        queue: list[str] = []

        def add(command: str) -> None:
            if command and command not in queue:
                queue.append(command)

        if failed_runs:
            add("recovery closure checklist")
        for row in failed_runs[:3]:
            run_id = row["id"]
            add(f"verification receipt {run_id}")
            add(f"execution recovery packet {run_id}")
            add(f"execution learning closure {run_id}")
            add(f"after-action learning packet {run_id}")
        for row in (approval_held_runs or [])[:3]:
            for command in _approval_review_commands_for_tool_run(row):
                add(command)
        for row in approvals[:4]:
            approval_id = row["id"]
            add(f"approval readiness {approval_id}")
            add(f"approval packet {approval_id}")
            add(f"approval chain proof {approval_id}")
            add(f"verification receipt <approved run id from approval chain proof {approval_id}>")
        for row in (tasks or [])[:2]:
            add(f"task completion packet {row['id']}")
        for goal in (goals or [])[:2]:
            steps = store.list_goal_steps(goal["id"])
            open_steps = [step for step in steps if step["status"] != "done"]
            if open_steps:
                add(f"complete goal step {goal['id']} {open_steps[0]['id']}")
        add("build delta")
        add("work block checkpoint")
        add("harness completion")
        add("completion claim gate")
        return queue

    def _format_note_mtime(timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp).replace(microsecond=0).isoformat()

    def recent_saved_notes_body(limit: int) -> tuple[str, dict[str, int | bool]]:
        limit = _bounded_int(limit, 12, high=30)
        note_dirs = [
            ("Reflections", vault.root_path / "Reflections"),
            ("Automations", vault.root_path / "Automations"),
            ("Daily", vault.root_path / "Daily"),
            ("Memory Tree", vault.root_path / "Memory Tree"),
        ]
        rows: list[dict[str, Any]] = []
        for folder_name, folder in note_dirs:
            if not folder.exists():
                continue
            for path in folder.glob("*.md"):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                rows.append(
                    {
                        "folder": folder_name,
                        "name": path.name,
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                        "path": path,
                    }
                )
        rows.sort(key=lambda row: row["modified"], reverse=True)
        shown = rows[:limit]
        folder_counts = Counter(row["folder"] for row in rows)

        lines = [
            "Jarvis recent saved notes:",
            "This is metadata-only. It lists Jarvis note files without opening note contents, approving requests, reading private data, controlling the computer, or queuing approvals.",
            "",
            "Latest saved notes:",
        ]
        if shown:
            for row in shown:
                rel_path = row["path"].relative_to(vault.root_path)
                lines.append(
                    f"- {rel_path} | {row['size']} bytes | modified {_format_note_mtime(row['modified'])}"
                )
        else:
            lines.append("- No saved Jarvis notes found yet.")

        lines.extend(["", "Folders scanned:"])
        for folder_name, _ in note_dirs:
            lines.append(f"- {folder_name}: {folder_counts.get(folder_name, 0)} note(s)")

        lines.extend(
            [
                "",
                "Useful follow-ups:",
                "- `save build delta` to save the latest checkpoint window.",
                "- `save handoff brief` before stopping a long build session.",
                "- `mission control` to refresh the main Obsidian command note.",
            ]
        )
        metadata = _safe_metadata(
            notes=len(rows),
            shown=len(shown),
            limit=limit,
            reflections=folder_counts.get("Reflections", 0),
            automations=folder_counts.get("Automations", 0),
            daily=folder_counts.get("Daily", 0),
            memory_tree=folder_counts.get("Memory Tree", 0),
        )
        return "\n".join(lines), metadata

    def operator_timebox_contract(args: dict[str, Any]) -> ToolResult:
        objective = str(args.get("objective") or args.get("request") or "continue Jarvis work").strip()
        stop_at = str(args.get("stop_at") or args.get("stop") or args.get("until") or "").strip()
        current_time = str(args.get("current_time") or args.get("now") or "").strip()
        timezone_label = str(args.get("timezone") or "local").strip() or "local"
        stop_dt = _parse_iso_datetime(stop_at)
        current_dt = _parse_iso_datetime(current_time) if current_time else datetime.now(stop_dt.tzinfo) if stop_dt and stop_dt.tzinfo else datetime.now()
        state = "HELD_FOR_EXPLICIT_STOP_TIME"
        minutes_remaining: int | None = None
        missing: list[str] = []
        if not stop_at:
            missing.append("stop_at")
        if stop_at and not stop_dt:
            missing.append("parseable stop_at ISO timestamp")
        if current_time and not _parse_iso_datetime(current_time):
            missing.append("parseable current_time ISO timestamp")
        if stop_dt and current_dt:
            try:
                delta_seconds = int((stop_dt - current_dt).total_seconds())
                minutes_remaining = delta_seconds // 60
                state = "STOP_WINDOW_ACTIVE" if delta_seconds > 0 else "STOP_TIME_REACHED"
            except TypeError:
                missing.append("matching timezone-aware stop/current timestamps")
                state = "HELD_FOR_PARSEABLE_TIMEBOX"
        elif missing:
            state = "HELD_FOR_PARSEABLE_TIMEBOX"

        can_continue = state == "STOP_WINDOW_ACTIVE"
        next_safe_command = "continue scoped local-safe Jarvis work" if can_continue else "stop and summarize work completed"
        current_time_value = current_time or (current_dt.isoformat() if current_dt else "")
        timebox_receipt_sha256 = _operator_timebox_receipt_sha256(
            objective=objective,
            stop_at=stop_at,
            current_time=current_time_value,
            timezone_label=timezone_label,
            timebox_state=state,
        )
        timebox_review_contract_rows = _operator_timebox_review_contract_rows(
            timebox_state=state,
            stop_at=stop_at,
            current_time=current_time_value,
            timezone_label=timezone_label,
        )
        timebox_review_contract_summary = [
            "explicit-stop-window-only",
            "newer-instruction-check-required",
            "resume-gate-required",
            "one-step-only",
            "post-step-closure-required",
            "fresh-timebox-required-next",
        ]
        awake_guard_requested = _awake_guard_requested(objective)
        awake_guard_token_sha256 = _awake_guard_token_sha256(
            objective=objective,
            stop_at=stop_at,
            current_time=current_time_value,
            timezone_label=timezone_label,
            requested=awake_guard_requested,
        )
        awake_guard_boundary_rows = _awake_guard_boundary_rows(
            requested=awake_guard_requested,
            stop_at=stop_at,
            current_time=current_time_value,
        )
        lines = [
            "Jarvis operator timebox contract:",
            "This is read-only. It decides whether a continuation is still inside the operator's stated work window without editing files, running tools, approving requests, writing notes, controlling the computer, or queuing approvals.",
            "",
            "Objective:",
            f"- {objective[:300]}",
            "",
            "Timebox:",
            f"- timezone label: {timezone_label}",
            f"- current time: {current_time or (current_dt.isoformat() if current_dt else 'unknown')}",
            f"- stop at: {stop_at or 'not supplied'}",
            f"- state: {state}",
            f"- minutes remaining: {minutes_remaining if minutes_remaining is not None else 'unknown'}",
            f"- can continue now: {'yes' if can_continue else 'no'}",
            f"- missing proof: {', '.join(missing) if missing else 'none'}",
            f"- timebox receipt sha256: {timebox_receipt_sha256 or 'missing'}",
            "",
            "Operator rule:",
            "- the operator's explicit stop times, work windows, pause commands, and newer instructions override priority goals, scheduled jobs, recovery queues, and autonomous continuation momentum.",
            "- If the stop time is reached or unclear, Jarvis should stop, save or present a concise checkpoint, and wait for a newer instruction.",
            "",
            "Awake guard boundary:",
            f"- awake requested: {'yes' if awake_guard_requested else 'no'}",
            f"- awake guard token sha256: {awake_guard_token_sha256 or 'missing'}",
            f"- row count: {len(awake_guard_boundary_rows)}",
            "- a work timebox does not authorize OS wake locks, shell commands such as caffeinate, computer control, approvals, or reusable power-management state.",
            "- if an OS wake-lock command is needed, it remains a separate approval-gated shell/system action outside this read-only contract.",
            "- an awake guard token is proof-only; it cannot be reused as permission to run caffeinate, keep-awake commands, or another continuation window.",
            *[
                f"- {row['item']}: status {row['status']}; authorizes OS wake lock no; authorizes shell execution no; authorizes computer control no; authorizes approval no; reusable for next timebox no"
                for row in awake_guard_boundary_rows
            ],
            "",
            "Non-authorizing timebox review contract:",
            f"- row count: {len(timebox_review_contract_rows)}",
            f"- summary: {', '.join(timebox_review_contract_summary)}",
            *[
                f"- {row['item']}: fresh required yes; reusable for next step no; authorizes execution no; authorizes local-safe step no; authorizes risky work no; authorizes approval no; authorizes timebox reuse no"
                for row in timebox_review_contract_rows
            ],
            "",
            "Next safe command:",
            f"- {next_safe_command}",
            "",
            "Boundary:",
            "- This contract does not run shell/code, read personal data, control the computer, call external services, speak, write notes, or queue approvals.",
        ]
        return ToolResult(
            "operator_timebox_contract",
            True,
            "\n".join(lines),
            _safe_metadata(
                objective=objective[:300],
                stop_at=stop_at,
                current_time=current_time_value,
                timezone=timezone_label,
                timebox_state=state,
                minutes_remaining=minutes_remaining,
                can_continue_now=can_continue,
                should_stop_now=not can_continue,
                missing_timebox_proof=missing,
                missing_timebox_proof_count=len(missing),
                timebox_receipt_sha256=timebox_receipt_sha256,
                timebox_receipt_present=_looks_like_sha256(timebox_receipt_sha256),
                timebox_review_contract_rows=timebox_review_contract_rows,
                timebox_review_contract_row_count=len(timebox_review_contract_rows),
                timebox_review_contract_ready=_operator_timebox_review_contract_ready(
                    timebox_review_contract_rows,
                    timebox_state=state,
                    stop_at=stop_at,
                    current_time=current_time_value,
                    timezone_label=timezone_label,
                ),
                timebox_review_contract_summary=timebox_review_contract_summary,
                awake_guard_requested=awake_guard_requested,
                awake_guard_token_sha256=awake_guard_token_sha256,
                awake_guard_token_present=_looks_like_sha256(awake_guard_token_sha256),
                awake_guard_boundary_rows=awake_guard_boundary_rows,
                awake_guard_boundary_row_count=len(awake_guard_boundary_rows),
                awake_guard_authorizes_os_wake_lock=False,
                awake_guard_authorizes_shell_execution=False,
                awake_guard_authorizes_computer_control=False,
                awake_guard_authorizes_approval=False,
                awake_guard_authorizes_model_call=False,
                awake_guard_authorizes_tool_execution=False,
                awake_guard_authorizes_personal_data_read=False,
                awake_guard_authorizes_external_side_effect=False,
                awake_guard_reusable_for_next_timebox=False,
                awake_guard_requires_separate_operator_request=True,
                awake_guard_requires_separate_shell_approval=True,
                awake_guard_os_wake_lock_boundary_ready=_awake_guard_boundary_ready(
                    token_sha256=awake_guard_token_sha256,
                    rows=awake_guard_boundary_rows,
                    requested=awake_guard_requested,
                    stop_at=stop_at,
                    current_time=current_time_value,
                    objective=objective[:300],
                    timezone_label=timezone_label,
                ),
                awake_guard_caffeinate_command_authorized=False,
                awake_guard_keep_awake_command_authorized=False,
                awake_guard_authorizes_unattended_execution=False,
                awake_guard_authorizes_continuation_window=False,
                awake_guard_reusable_as_execution_permission=False,
                next_awake_guard_requires_fresh_review=True,
                timebox_authorizes_execution=False,
                timebox_authorizes_local_safe_step=False,
                timebox_authorizes_risky_work=False,
                timebox_authorizes_approval=False,
                timebox_authorizes_timebox_reuse=False,
                timebox_authorizes_model_call=False,
                timebox_authorizes_tool_execution=False,
                timebox_authorizes_personal_data_read=False,
                timebox_authorizes_external_side_effect=False,
                timebox_reusable_for_next_step=False,
                next_step_requires_fresh_timebox=True,
                next_safe_command=next_safe_command,
                operator_timeboxes_override_priority=True,
                stop_times_override_priority=True,
            ),
        )

    def operator_instruction_supersession_packet(args: dict[str, Any]) -> ToolResult:
        objective = str(args.get("objective") or args.get("request") or "continue Jarvis work").strip()
        previous_instruction = str(args.get("previous_instruction") or args.get("previous") or args.get("old_instruction") or "").strip()
        latest_instruction = str(args.get("latest_instruction") or args.get("latest") or args.get("new_instruction") or "").strip()
        stop_at = str(args.get("stop_at") or args.get("stop") or args.get("until") or "").strip()
        current_time = str(args.get("current_time") or args.get("now") or "").strip()
        timezone_label = str(args.get("timezone") or "local").strip() or "local"
        timebox = operator_timebox_contract(
            {"objective": objective, "stop_at": stop_at, "current_time": current_time, "timezone": timezone_label}
        )
        timebox_metadata = timebox.metadata
        latest_lower = latest_instruction.lower()
        previous_present = bool(previous_instruction)
        latest_present = bool(latest_instruction)
        latest_is_stop = any(term in latest_lower for term in ("stop", "pause", "halt", "cancel", "done for now", "past"))
        latest_is_continue = any(term in latest_lower for term in ("continue", "work until", "resume", "keep working"))
        supersedes_previous = latest_present and previous_present
        blockers: list[str] = []
        if not latest_present:
            blockers.append("latest instruction missing")
        if not timebox_metadata.get("can_continue_now"):
            blockers.append(f"timebox {timebox_metadata.get('timebox_state') or 'held'}")
        if latest_is_stop:
            blockers.append("latest instruction is a stop or pause")
        state = "NEWER_STOP_OR_PAUSE_OVERRIDES_AUTONOMY" if latest_is_stop else (
            "LATEST_INSTRUCTION_READY_TO_GOVERN_CONTINUATION" if latest_present and not blockers else "SUPERVISION_HELD_FOR_NEWEST_INSTRUCTION_REVIEW"
        )
        can_continue = latest_present and not blockers
        next_safe_command = (
            "stop and summarize work completed"
            if latest_is_stop or not timebox_metadata.get("can_continue_now")
            else "autonomy resume gate: stop_at=<ISO> current_time=<ISO> step=<reviewed local-safe step> verification=<evidence> receipt=<path> receipt_sha256=<hash> checkpoint=<path> checkpoint_sha256=<hash> stop_condition=<condition>"
        )
        supersession_contract_rows = [
            {
                "item": "newest_instruction",
                "required": True,
                "source": "latest operator instruction",
                "authorizes_execution": False,
                "authorizes_risky_work": False,
                "authorizes_approval": False,
                "authorizes_recovery_followthrough": False,
                "authorizes_timebox_override": False,
                "authorizes_goal_override": True,
            },
            {
                "item": "active_timebox",
                "required": True,
                "source": "operator timebox contract",
                "authorizes_execution": False,
                "authorizes_risky_work": False,
                "authorizes_approval": False,
                "authorizes_recovery_followthrough": False,
                "authorizes_timebox_override": False,
                "authorizes_goal_override": False,
            },
            {
                "item": "resume_gate_review",
                "required": not latest_is_stop,
                "source": "autonomy resume gate",
                "authorizes_execution": False,
                "authorizes_risky_work": False,
                "authorizes_approval": False,
                "authorizes_recovery_followthrough": False,
                "authorizes_timebox_override": False,
                "authorizes_goal_override": False,
            },
            {
                "item": "one_step_local_safe_review",
                "required": not latest_is_stop,
                "source": "autonomy continuation execution packet",
                "authorizes_execution": False,
                "authorizes_risky_work": False,
                "authorizes_approval": False,
                "authorizes_recovery_followthrough": False,
                "authorizes_timebox_override": False,
                "authorizes_goal_override": False,
            },
            {
                "item": "stop_or_pause_brake",
                "required": latest_is_stop,
                "source": "latest operator instruction",
                "authorizes_execution": False,
                "authorizes_risky_work": False,
                "authorizes_approval": False,
                "authorizes_recovery_followthrough": False,
                "authorizes_timebox_override": False,
                "authorizes_goal_override": True,
            },
        ]
        supersession_contract_ready = _operator_supersession_contract_ready(
            supersession_contract_rows,
            latest_is_stop=latest_is_stop,
        )
        supersession_token_sha256 = _operator_supersession_token_sha256(
            objective=objective,
            previous_instruction=previous_instruction,
            latest_instruction=latest_instruction,
            stop_at=stop_at,
            current_time=str(timebox_metadata.get("current_time") or current_time),
            timezone_label=timezone_label,
            supersession_state=state,
            timebox_receipt_sha256=str(timebox_metadata.get("timebox_receipt_sha256") or ""),
        )
        supersession_token_boundary_rows = _operator_supersession_token_boundary_rows(
            token_sha256=supersession_token_sha256,
            source="operator_instruction_supersession",
        )
        supersession_token_boundary_ready = _operator_supersession_token_boundary_ready(
            supersession_token_sha256,
            supersession_token_boundary_rows,
            expected_source="operator_instruction_supersession",
        )
        lines = [
            "Jarvis operator instruction supersession packet:",
            "This is read-only. It proves which operator instruction governs long-running autonomy before Jarvis continues, without executing tools, approving requests, writing notes, reading private data, controlling the computer, or queuing approvals.",
            "",
            "Objective:",
            f"- {objective[:300]}",
            "",
            "Instruction precedence:",
            f"- previous instruction supplied: {'yes' if previous_present else 'no'}",
            f"- latest instruction supplied: {'yes' if latest_present else 'no'}",
            f"- latest instruction supersedes previous: {'yes' if supersedes_previous else 'no'}",
            f"- latest instruction is stop/pause: {'yes' if latest_is_stop else 'no'}",
            f"- latest instruction is continue/resume: {'yes' if latest_is_continue else 'no'}",
            "",
            "Supersession gate:",
            f"- state: {state}",
            f"- can continue under latest instruction now: {'yes' if can_continue else 'no'}",
            f"- blockers: {', '.join(blockers) if blockers else 'none'}",
            f"- next safe command: `{next_safe_command}`",
            "",
            "Bound timebox:",
            f"- timebox state: {timebox_metadata.get('timebox_state')}",
            f"- can continue now: {'yes' if timebox_metadata.get('can_continue_now') else 'no'}",
            f"- should stop now: {'yes' if timebox_metadata.get('should_stop_now') else 'no'}",
            f"- minutes remaining: {timebox_metadata.get('minutes_remaining') if timebox_metadata.get('minutes_remaining') is not None else 'unknown'}",
            "",
            "Operator rule:",
            "- The newest explicit user instruction overrides older goals, automation heartbeats, scheduled continuations, recovery queues, and prior work windows.",
            "- If the newest instruction says stop/pause, or the stated stop time has passed, Jarvis must stop and summarize instead of continuing from the persistent goal.",
            "- If the newest instruction says continue and the stop window is active, Jarvis may proceed only through the autonomy resume gate and one-step local-safe continuation proof.",
            "",
            "Supersession contract:",
            f"- contract ready: {'yes' if supersession_contract_ready else 'no'}",
            f"- row count: {len(supersession_contract_rows)}",
            "- authorizes execution now: no",
            "- authorizes risky work: no",
            "- authorizes approval: no",
            "- authorizes recovery follow-through: no",
            "- authorizes timebox override: no",
            *[
                f"- {row['item']}: required {'yes' if row['required'] else 'no'}; source {row['source']}; authorizes execution no; authorizes risky work no; authorizes approval no; authorizes recovery follow-through no; authorizes timebox override no"
                for row in supersession_contract_rows
            ],
            "",
            "Supersession token boundary:",
            f"- token sha256: {supersession_token_sha256 or 'missing'}",
            f"- row count: {len(supersession_token_boundary_rows)}",
            "- token authorizes execution now: no",
            "- token authorizes local-safe step: no",
            "- token authorizes risky work: no",
            "- token authorizes approval: no",
            "- token authorizes recovery follow-through: no",
            "- token authorizes timebox override: no",
            "- token reusable for next review: no",
            *[
                f"- {row['item']}: {row['status']}; authorizes execution no; local-safe step no; risky work no; approval no; recovery follow-through no; timebox override no; reusable no"
                for row in supersession_token_boundary_rows
            ],
            "",
            "Boundary:",
            "- This packet does not run shell/code, read personal data, control the computer, call external services, speak, write notes, or queue approvals.",
        ]
        return ToolResult(
            "operator_instruction_supersession_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                objective=objective[:300],
                previous_instruction=previous_instruction[:300],
                latest_instruction=latest_instruction[:300],
                stop_at=stop_at,
                current_time=str(timebox_metadata.get("current_time") or current_time),
                timezone=timezone_label,
                previous_instruction_present=previous_present,
                latest_instruction_present=latest_present,
                latest_instruction_supersedes_previous=supersedes_previous,
                latest_instruction_is_stop=latest_is_stop,
                latest_instruction_is_continue=latest_is_continue,
                supersession_state=state,
                can_continue_under_latest_instruction=can_continue,
                blockers=blockers,
                blocker_count=len(blockers),
                next_safe_command=next_safe_command,
                timebox_state=timebox_metadata.get("timebox_state"),
                can_continue_now=timebox_metadata.get("can_continue_now"),
                should_stop_now=timebox_metadata.get("should_stop_now"),
                minutes_remaining=timebox_metadata.get("minutes_remaining"),
                missing_timebox_proof=timebox_metadata.get("missing_timebox_proof", []),
                missing_timebox_proof_count=timebox_metadata.get("missing_timebox_proof_count", 0),
                timebox_receipt_sha256=timebox_metadata.get("timebox_receipt_sha256", ""),
                timebox_receipt_present=timebox_metadata.get("timebox_receipt_present", False),
                newest_instruction_overrides_automation=True,
                newest_instruction_overrides_goal=True,
                newest_instruction_overrides_recovery_queue=True,
                stop_or_pause_blocks_autonomy=latest_is_stop,
                supersession_contract_rows=supersession_contract_rows,
                supersession_contract_row_count=len(supersession_contract_rows),
                supersession_contract_ready=supersession_contract_ready,
                supersession_authorizes_execution=False,
                supersession_authorizes_risky_work=False,
                supersession_authorizes_approval=False,
                supersession_authorizes_recovery_followthrough=False,
                supersession_authorizes_timebox_override=False,
                supersession_token_sha256=supersession_token_sha256,
                supersession_token_present=_looks_like_sha256(supersession_token_sha256),
                supersession_token_boundary_rows=supersession_token_boundary_rows,
                supersession_token_boundary_row_count=len(supersession_token_boundary_rows),
                supersession_token_boundary_ready=supersession_token_boundary_ready,
                supersession_token_authorizes_execution=False,
                supersession_token_authorizes_local_safe_step=False,
                supersession_token_authorizes_risky_work=False,
                supersession_token_authorizes_approval=False,
                supersession_token_authorizes_recovery_followthrough=False,
                supersession_token_authorizes_timebox_override=False,
                supersession_token_authorizes_goal_override=False,
                supersession_token_authorizes_model_call=False,
                supersession_token_authorizes_tool_execution=False,
                supersession_token_authorizes_personal_data_read=False,
                supersession_token_authorizes_external_side_effect=False,
                supersession_token_reusable_for_next_review=False,
                supersession_token_reusable_for_next_timebox=False,
                next_supersession_requires_fresh_latest_instruction_review=True,
                timebox_output=timebox.output[:1200],
            ),
        )

    def build_progress_body(limit: int) -> tuple[str, dict[str, int]]:
        limit = _bounded_int(limit, 18)
        raw_tool_runs = store.recent_tool_runs(limit=limit)
        tool_runs, unreadable_tool_run_rows = _safe_tool_run_payloads(raw_tool_runs)
        approvals = store.list_pending_approvals(limit=8)
        approval_handoff = _first_approval_handoff(approvals)
        tasks = store.list_tasks(status="open", limit=6)
        goals = store.list_goals(status="active", limit=6)
        sessions, unreadable_session_rows = _safe_session_payloads(store.list_sessions(limit=4))
        memories = store.recent_memories(limit=6)
        jobs = store.list_jobs()

        tool_counts = Counter(row["tool_name"] for row in tool_runs)
        tool_run_status_counts = _tool_run_status_counts(tool_runs)
        failed_runs = [row for row in tool_runs if _tool_run_status(row) == "failed"]
        approval_held_runs = [row for row in tool_runs if _tool_run_status(row) == "approval_held"]
        approved_runs = [row for row in tool_runs if row["approved"]]
        read_only_runs = [row for row in tool_runs if row["risk"] == "READ_ONLY"]
        local_safe_runs = [row for row in tool_runs if row["risk"] == "LOCAL_SAFE"]
        risky_blocked_runs = [
            row
            for row in failed_runs
            if row["risk"] in {"HIGH_RISK", "PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT"}
        ]
        build_proof_queue = _build_proof_queue(
            approvals=approvals,
            failed_runs=failed_runs,
            approval_held_runs=approval_held_runs,
            tasks=tasks,
            goals=goals,
        )

        lines = [
            "Jarvis build progress report:",
            "",
            *_priority_goal_lines(),
            "",
            "Recent execution:",
        ]
        if tool_runs:
            lines.append(f"- tool runs reviewed: {len(tool_runs)}")
            lines.append(f"- top tools: {', '.join(f'{name} x{count}' for name, count in tool_counts.most_common(6))}")
            lines.append(f"- read-only/local-safe runs: {len(read_only_runs)} read-only, {len(local_safe_runs)} local-safe")
            lines.append(f"- approved high-risk reruns: {len(approved_runs)}")
            lines.append(f"- failed or blocked runs: {len(failed_runs)}")
            lines.append(f"- approval-held runs: {len(approval_held_runs)}")
        else:
            lines.append("- no tool runs logged yet")
        if unreadable_tool_run_rows:
            lines.append(f"- unreadable tool run row(s) hidden for safety: {unreadable_tool_run_rows}")

        lines.extend(["", "Safety posture:"])
        if approvals:
            lines.append(f"- pending approvals: {len(approvals)}")
            for row in approvals[:4]:
                lines.append(f"  - #{row['id']} {_safe_preview(row['tool_name'], limit=80)}: {_safe_preview(row['user_input'])}")
                lines.append(f"    readiness: `approval readiness {row['id']}`")
                lines.append(f"    last look: `approval packet {row['id']}`")
                lines.append(f"    proof: `approval chain proof {row['id']}`")
            lines.append(f"- first safe handoff: `{approval_handoff['approval_handoff_readiness_command']}` -> `{approval_handoff['approval_handoff_last_look_command']}` -> `{approval_handoff['approval_handoff_proof_command']}`")
            lines.append(f"- decision commands: `{approval_handoff['approval_handoff_approve_command']}` or `{approval_handoff['approval_handoff_dismiss_command']}`")
        else:
            lines.append("- pending approvals: none")
        if risky_blocked_runs:
            lines.append(f"- recent blocked risky runs: {len(risky_blocked_runs)}")
        else:
            lines.append("- recent blocked risky runs: none in reviewed window")
        if approval_held_runs:
            lines.append(f"- recent approval-held runs: {len(approval_held_runs)}")
            for row in approval_held_runs[:3]:
                lines.append(
                    f"  - run #{row['id']} {_safe_preview(row['tool_name'], limit=80)}: "
                    "safety gate held before execution"
                )
                for command in _approval_review_commands_for_tool_run(row)[:3]:
                    lines.append(f"    review: `{command}`")
        else:
            lines.append("- recent approval-held runs: none in reviewed window")
        lines.append("- approval-gated actions remain paused until the operator explicitly approves them after a last-look packet.")

        lines.extend(["", "Current work state:"])
        if tasks:
            lines.append(f"- open tasks: {len(tasks)}; first #{tasks[0]['id']} {_safe_preview(tasks[0]['body'])}")
        else:
            lines.append("- open tasks: none")
        if goals:
            goal_bits = []
            for goal in goals[:3]:
                steps = store.list_goal_steps(goal["id"])
                open_steps = [step for step in steps if step["status"] != "done"]
                next_step = _safe_preview(open_steps[0]["body"]) if open_steps else "define next step"
                goal_bits.append(f"#{goal['id']} {_safe_preview(goal['title'])} -> {next_step}")
            lines.append("- active goals: " + "; ".join(goal_bits))
        else:
            lines.append("- active goals: none")
        enabled_jobs = [row for row in jobs if row["enabled"]]
        lines.append(f"- scheduled jobs: {len(enabled_jobs)} enabled / {len(jobs)} total")
        if sessions:
            latest_session = sessions[0]
            lines.append(f"- recent sessions: {len(sessions)}; latest {_safe_preview(latest_session['session_id'], limit=80)} at {_safe_preview(latest_session['last_at'], limit=80)}")
        else:
            lines.append("- recent sessions: none")
        if unreadable_session_rows:
            lines.append(f"- unreadable session row(s) hidden for safety: {unreadable_session_rows}")

        lines.extend(["", "Recent memory anchors:"])
        if memories:
            for row in memories[:5]:
                lines.append(f"- [{_safe_preview(row['category'], limit=80)}] {_safe_preview(row['title'])}")
        else:
            lines.append("- none")

        lines.extend(
            [
                "",
                "Safe next build moves:",
                "- Run `roadmap` to choose the next assistant layer.",
                "- Run `approval review`, then `approval readiness #ID`, `approval packet #ID`, and `approval chain proof #ID`, before approving blocked risky work.",
                "- Run `focus brief` before a hands-on coding or desktop-control session.",
                "- Keep new connectors read-only first, with personal-data and side-effect tools approval-gated.",
            ]
        )
        lines.extend(
            [
                "",
                "Build proof queue:",
                f"- next required command: `{build_proof_queue[0]}`" if build_proof_queue else "- next required command: none",
                f"- proof queue count: {len(build_proof_queue)}",
            ]
        )
        if build_proof_queue:
            lines.extend(f"- `{command}`" for command in build_proof_queue[:8])
        else:
            lines.append("- none")

        metadata = _safe_metadata(
            tool_runs=len(tool_runs),
            recent_ok_tool_runs=tool_run_status_counts["ok"],
            recent_failed_tool_runs=len(failed_runs),
            recent_approval_held_tool_runs=len(approval_held_runs),
            readable_tool_run_rows=len(tool_runs),
            unreadable_tool_run_rows=unreadable_tool_run_rows,
            pending_approvals=len(approvals),
            **approval_handoff,
            open_tasks=len(tasks),
            active_goals=len(goals),
            sessions=len(sessions),
            readable_session_rows=len(sessions),
            unreadable_session_rows=unreadable_session_rows,
            enabled_jobs=len(enabled_jobs),
            build_proof_queue=build_proof_queue,
            build_proof_queue_count=len(build_proof_queue),
            build_next_required_command=build_proof_queue[0] if build_proof_queue else "",
            build_next_proof_command=build_proof_queue[0] if build_proof_queue else "",
            limit=limit,
        )
        return "\n".join(lines), metadata

    def build_progress_report(args: dict[str, Any]) -> ToolResult:
        body, metadata = build_progress_body(_bounded_int(args.get("limit"), 18))
        return ToolResult("build_progress_report", True, body, metadata)

    def build_delta_body(limit: int) -> tuple[str, dict[str, int | bool]]:
        limit = _bounded_int(limit, 12, high=40)
        raw_tool_runs = store.recent_tool_runs(limit=limit)
        tool_runs, unreadable_tool_run_rows = _safe_tool_run_payloads(raw_tool_runs)
        raw_messages = store.recent_messages(limit=limit)
        messages, unreadable_message_rows = _safe_message_payloads(raw_messages)
        approvals = store.list_pending_approvals(limit=8)
        approval_handoff = _first_approval_handoff(approvals)
        tool_run_status_counts = _tool_run_status_counts(tool_runs)
        failed_runs = [row for row in tool_runs if _tool_run_status(row) == "failed"]
        approval_held_runs = [row for row in tool_runs if _tool_run_status(row) == "approval_held"]
        blocked_risky_runs = [
            row
            for row in failed_runs
            if row["risk"] in {"HIGH_RISK", "PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT"}
        ]
        tasks = store.list_tasks(status="open", limit=4)
        goals = store.list_goals(status="active", limit=4)
        build_proof_queue = _build_proof_queue(
            approvals=approvals,
            failed_runs=failed_runs,
            approval_held_runs=approval_held_runs,
            tasks=tasks,
            goals=goals,
        )
        changed_tools = Counter(row["tool_name"] for row in tool_runs)
        assistant_messages = [row for row in messages if row["role"] == "assistant"]
        user_messages = [row for row in messages if row["role"] == "user"]

        lines = [
            "Jarvis build delta:",
            "This is read-only. It summarizes the latest checkpoint window without running tools, writing notes, approving requests, controlling the computer, or queuing approvals.",
            "",
            *_priority_goal_lines(),
            "",
            "Checkpoint window:",
            f"- recent tool runs reviewed: {len(tool_runs)}",
            f"- unreadable tool run row(s) hidden for safety: {unreadable_tool_run_rows}",
            f"- recent messages reviewed: {len(messages)}",
            f"- unreadable message row(s) hidden for safety: {unreadable_message_rows}",
            f"- pending approvals now: {len(approvals)}",
        ]
        if tool_runs:
            newest = tool_runs[0]["created_at"]
            oldest = tool_runs[-1]["created_at"]
            lines.append(f"- tool-run time span: {oldest} -> {newest}")
            lines.append("- tools touched: " + ", ".join(f"{name} x{count}" for name, count in changed_tools.most_common(6)))
        else:
            lines.append("- tool-run time span: no tool runs logged yet")

        lines.extend(["", "Latest tool activity:"])
        if tool_runs:
            for row in reversed(tool_runs[:6]):
                row_status = _tool_run_status(row)
                status = "ok" if row_status == "ok" else "approval held" if row_status == "approval_held" else "blocked/failed"
                approved = ", approved" if row["approved"] else ""
                approval_id = row.get("approval_id")
                approval_link = f", approval #{approval_id}" if approval_id else ""
                snippet = "safety gate held before execution" if row_status == "approval_held" else _safe_preview(row["output"], limit=120)
                lines.append(f"- {row['created_at']} {row['tool_name']} [{row['risk']}, {status}{approved}{approval_link}]: {snippet}")
        else:
            lines.append("- No tool activity in this window.")

        lines.extend(["", "Conversation delta:"])
        if user_messages:
            lines.append(f"- latest user ask: {_safe_preview(user_messages[-1]['content'], limit=180)}")
        else:
            lines.append("- latest user ask: none in this window")
        if assistant_messages:
            lines.append(f"- latest Jarvis response: {_safe_preview(assistant_messages[-1]['content'], limit=180)}")
        else:
            lines.append("- latest Jarvis response: none in this window")

        lines.extend(["", "Safety delta:"])
        lines.append(f"- blocked or failed runs: {len(failed_runs)}")
        lines.append(f"- approval-held runs: {len(approval_held_runs)}")
        lines.append(f"- blocked risky runs: {len(blocked_risky_runs)}")
        if approval_held_runs:
            for row in approval_held_runs[:3]:
                lines.append(
                    f"- approval-held run #{row['id']} {_safe_preview(row['tool_name'], limit=80)}: "
                    "safety gate held before execution"
                )
                for command in _approval_review_commands_for_tool_run(row)[:3]:
                    lines.append(f"  review: `{command}`")
        if approvals:
            for row in approvals[:4]:
                lines.append(f"- approval #{row['id']} still pending: {row['tool_name']} for `{row['user_input'][:120]}`")
                lines.append(f"  readiness: `approval readiness {row['id']}`")
                lines.append(f"  last look: `approval packet {row['id']}`")
                lines.append(f"  proof: `approval chain proof {row['id']}`")
            lines.append(f"- first safe handoff: `{approval_handoff['approval_handoff_readiness_command']}` -> `{approval_handoff['approval_handoff_last_look_command']}` -> `{approval_handoff['approval_handoff_proof_command']}`")
            lines.append(f"- decision commands: `{approval_handoff['approval_handoff_approve_command']}` or `{approval_handoff['approval_handoff_dismiss_command']}`")
        else:
            lines.append("- no pending approvals")

        lines.extend(
            [
                "",
                "Safe next checks:",
                "- `build progress` for the broader build state.",
                "- `chat continuity brief` for the current conversation thread.",
                "- `approval review`, then `approval readiness #ID`, `approval packet #ID`, and `approval chain proof #ID`, before rerunning any blocked risky request.",
                "- Respect the operator's explicit stop times, work windows, pause commands, and newer instructions before continuing the build loop.",
            ]
        )
        lines.extend(
            [
                "",
                "Build proof queue:",
                f"- next required command: `{build_proof_queue[0]}`" if build_proof_queue else "- next required command: none",
                f"- proof queue count: {len(build_proof_queue)}",
            ]
        )
        if build_proof_queue:
            lines.extend(f"- `{command}`" for command in build_proof_queue[:8])
        else:
            lines.append("- none")
        metadata = _safe_metadata(
            tool_runs=len(tool_runs),
            recent_ok_tool_runs=tool_run_status_counts["ok"],
            recent_failed_tool_runs=len(failed_runs),
            recent_approval_held_tool_runs=len(approval_held_runs),
            readable_tool_run_rows=len(tool_runs),
            unreadable_tool_run_rows=unreadable_tool_run_rows,
            messages=len(messages),
            readable_message_rows=len(messages),
            unreadable_message_rows=unreadable_message_rows,
            pending_approvals=len(approvals),
            **approval_handoff,
            failed_runs=len(failed_runs),
            approval_held_runs=len(approval_held_runs),
            blocked_risky_runs=len(blocked_risky_runs),
            open_tasks=len(tasks),
            active_goals=len(goals),
            build_proof_queue=build_proof_queue,
            build_proof_queue_count=len(build_proof_queue),
            build_next_required_command=build_proof_queue[0] if build_proof_queue else "",
            build_next_proof_command=build_proof_queue[0] if build_proof_queue else "",
            limit=limit,
        )
        return "\n".join(lines), metadata

    def build_delta_report(args: dict[str, Any]) -> ToolResult:
        body, metadata = build_delta_body(_bounded_int(args.get("limit"), 12))
        return ToolResult("build_delta_report", True, body, metadata)

    def work_block_checkpoint_body(args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        limit = _bounded_int(args.get("limit"), 12, high=40)
        objective = str(args.get("objective") or args.get("request") or "Continue Jarvis V2 safely.").strip()
        tests = str(args.get("tests") or args.get("verification") or "").strip()
        blockers = str(args.get("blockers") or "").strip()
        next_safe_command = str(args.get("next") or args.get("next_safe_command") or "").strip()
        raw_tool_runs = store.recent_tool_runs(limit=limit)
        tool_runs, unreadable_tool_run_rows = _safe_tool_run_payloads(raw_tool_runs)
        approvals = store.list_pending_approvals(limit=8)
        approval_handoff = _first_approval_handoff(approvals)
        tasks = store.list_tasks(status="open", limit=6)
        goals = store.list_goals(status="active", limit=6)
        tool_run_status_counts = _tool_run_status_counts(tool_runs)
        failed_runs = [row for row in tool_runs if _tool_run_status(row) == "failed"]
        approval_held_runs = [row for row in tool_runs if _tool_run_status(row) == "approval_held"]
        verification_runs = [
            row
            for row in tool_runs
            if any(token in row["tool_name"].lower() or token in row["output"].lower() for token in ("smoke", "test", "compile", "verify"))
        ]
        tool_counts = Counter(row["tool_name"] for row in tool_runs)
        build_proof_queue = _build_proof_queue(
            approvals=approvals,
            failed_runs=failed_runs,
            approval_held_runs=approval_held_runs,
            tasks=tasks,
            goals=goals,
        )

        if not next_safe_command:
            if failed_runs:
                next_safe_command = "Inspect the latest failed run, fix one scoped issue, then rerun the matching smoke test."
            elif approval_held_runs:
                next_safe_command = f"Run `{_approval_review_commands_for_tool_run(approval_held_runs[0])[0]}` before treating the held run as execution evidence."
            elif approvals:
                next_safe_command = f"Run `approval readiness {approvals[0]['id']}` before any approval-gated rerun."
            else:
                next_safe_command = "Choose one scoped harness target, implement it, and run the relevant smoke test plus py_compile."

        lines = [
            "Jarvis work-block checkpoint:",
            "This is a resumable packet. It does not run tools, approve requests, write notes, control the computer, or touch private data.",
            "",
            *_priority_goal_lines(),
            "",
            "Objective:",
            f"- {objective[:240]}",
            "",
            "Progress evidence:",
            f"- recent tool runs reviewed: {len(tool_runs)}",
            f"- unreadable tool run row(s) hidden for safety: {unreadable_tool_run_rows}",
            f"- pending approvals: {len(approvals)}",
            f"- open tasks: {len(tasks)}",
            f"- active goals: {len(goals)}",
        ]
        if tool_runs:
            newest = tool_runs[0]["created_at"]
            oldest = tool_runs[-1]["created_at"]
            lines.append(f"- audited window: {oldest} -> {newest}")
            lines.append("- tools touched: " + ", ".join(f"{name} x{count}" for name, count in tool_counts.most_common(6)))
        else:
            lines.append("- audited window: no tool runs logged yet")

        lines.extend(["", "Verification evidence:"])
        if tests:
            lines.append(f"- stated verification: {tests[:240]}")
        if verification_runs:
            for row in verification_runs[:5]:
                status = "passed" if row["ok"] else "failed"
                lines.append(f"- {row['created_at']} {row['tool_name']}: {status}")
        if not tests and not verification_runs:
            lines.append("- no verification captured in this checkpoint; run the smallest relevant smoke test before closing the block")

        lines.extend(["", "Open risks and blockers:"])
        if blockers:
            lines.append(f"- stated blocker: {blockers[:240]}")
        if approvals:
            for row in approvals[:4]:
                lines.append(f"- pending approval #{row['id']}: {row['tool_name']} for `{row['user_input'][:120]}`")
                lines.append(f"  readiness: `approval readiness {row['id']}`")
                lines.append(f"  last look: `approval packet {row['id']}`")
                lines.append(f"  proof: `approval chain proof {row['id']}`")
            lines.append(f"- first safe handoff: `{approval_handoff['approval_handoff_readiness_command']}` -> `{approval_handoff['approval_handoff_last_look_command']}` -> `{approval_handoff['approval_handoff_proof_command']}`")
            lines.append(f"- decision commands: `{approval_handoff['approval_handoff_approve_command']}` or `{approval_handoff['approval_handoff_dismiss_command']}`")
        if approval_held_runs:
            for row in approval_held_runs[:3]:
                lines.append(
                    f"- approval-held run #{row['id']} {_safe_preview(row['tool_name'], limit=80)}: "
                    "safety gate held before execution"
                )
                for command in _approval_review_commands_for_tool_run(row)[:3]:
                    lines.append(f"  review: `{command}`")
        if failed_runs:
            for row in failed_runs[:3]:
                snippet = _safe_preview(row["output"], limit=120)
                lines.append(f"- failed/blocked {row['tool_name']}: {snippet}")
        if not blockers and not approvals and not approval_held_runs and not failed_runs:
            lines.append("- none detected in the reviewed window")

        lines.extend(
            [
                "",
                "Resume packet:",
                f"- next safe command: {next_safe_command}",
                "- verify: run py_compile for touched Python files and the nearest smoke test.",
                "- audit: save this checkpoint or build delta before handing off a long work block.",
                "- stop: do not execute shell/code, computer control, personal-data, external-effect, destructive, or risky actions without approval.",
                "- stop: do not continue past the operator's explicit stop time or newer pause/stop instruction.",
                "",
                "Build proof queue:",
                f"- next required command: `{build_proof_queue[0]}`" if build_proof_queue else "- next required command: none",
                f"- proof queue count: {len(build_proof_queue)}",
                *([f"- `{command}`" for command in build_proof_queue[:8]] if build_proof_queue else ["- none"]),
                "",
                "Boundary:",
                "- read-only checkpoint; no note write, no memory write, no queued approval, no model call, no external side effect.",
            ]
        )
        metadata = _safe_metadata(
            objective_length=len(objective),
            tests_provided=bool(tests),
            blockers_provided=bool(blockers),
            tool_runs=len(tool_runs),
            recent_ok_tool_runs=tool_run_status_counts["ok"],
            recent_failed_tool_runs=len(failed_runs),
            recent_approval_held_tool_runs=len(approval_held_runs),
            readable_tool_run_rows=len(tool_runs),
            unreadable_tool_run_rows=unreadable_tool_run_rows,
            verification_runs=len(verification_runs),
            pending_approvals=len(approvals),
            **approval_handoff,
            failed_runs=len(failed_runs),
            approval_held_runs=len(approval_held_runs),
            open_tasks=len(tasks),
            active_goals=len(goals),
            build_proof_queue=build_proof_queue,
            build_proof_queue_count=len(build_proof_queue),
            build_next_required_command=build_proof_queue[0] if build_proof_queue else "",
            build_next_proof_command=build_proof_queue[0] if build_proof_queue else "",
            limit=limit,
        )
        return "\n".join(lines), metadata

    def work_block_checkpoint(args: dict[str, Any]) -> ToolResult:
        body, metadata = work_block_checkpoint_body(args)
        return ToolResult("work_block_checkpoint", True, body, metadata)

    def save_build_progress(args: dict[str, Any]) -> ToolResult:
        body, metadata = build_progress_body(_bounded_int(args.get("limit"), 18))
        path = vault.write_reflection("Build Progress", body)
        path_display = _safe_vault_path_display(path, vault)
        metadata["path"] = str(path)
        metadata["path_display"] = path_display
        metadata["writes_files"] = True
        metadata["writes_notes"] = True
        return ToolResult(
            "save_build_progress",
            True,
            f"Build progress saved: {path_display}\n\n{body}",
            metadata,
        )

    def save_build_delta(args: dict[str, Any]) -> ToolResult:
        body, metadata = build_delta_body(_bounded_int(args.get("limit"), 12))
        path = vault.write_reflection("Build Delta", body)
        path_display = _safe_vault_path_display(path, vault)
        metadata = dict(metadata)
        metadata["path"] = str(path)
        metadata["path_display"] = path_display
        metadata["writes_files"] = True
        metadata["writes_notes"] = True
        return ToolResult(
            "save_build_delta",
            True,
            f"Build delta saved: {path_display}\n\n{body}",
            metadata,
        )

    def save_work_block_checkpoint(args: dict[str, Any]) -> ToolResult:
        body, metadata = work_block_checkpoint_body(args)
        path = vault.write_reflection("Work Block Checkpoint", body)
        path_display = _safe_vault_path_display(path, vault)
        metadata = dict(metadata)
        metadata["path"] = str(path)
        metadata["path_display"] = path_display
        metadata["writes_files"] = True
        metadata["writes_notes"] = True
        return ToolResult(
            "save_work_block_checkpoint",
            True,
            f"Work-block checkpoint saved: {path_display}\n\n{body}",
            metadata,
        )

    def checkpoint_recovery_preview(args: dict[str, Any]) -> ToolResult:
        objective = str(args.get("objective") or args.get("request") or "Resume Jarvis from the latest checkpoint.").strip()
        reflections = vault.root_path / "Reflections"
        checkpoint_files = sorted(
            reflections.glob("* Work Block Checkpoint.md"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0,
            reverse=True,
        )
        latest_path = checkpoint_files[0] if checkpoint_files else None
        latest_path_display = _safe_vault_path_display(latest_path, vault)
        checkpoint_text = ""
        checkpoint_age_minutes: int | None = None
        checkpoint_freshness = "missing"
        latest_checkpoint_sha256 = _file_sha256(str(latest_path)) if latest_path else ""
        if latest_path:
            try:
                checkpoint_age_minutes = max(
                    0,
                    int((datetime.now().timestamp() - latest_path.stat().st_mtime) // 60),
                )
                checkpoint_freshness = _checkpoint_freshness_label(checkpoint_age_minutes)
                checkpoint_text = latest_path.read_text(encoding="utf-8")
            except OSError:
                checkpoint_age_minutes = None
                checkpoint_freshness = "unknown"
                checkpoint_text = ""
        lines = [
            "Jarvis checkpoint recovery preview:",
            "This is read-only. It inspects Jarvis-generated checkpoint notes only; it does not run tools, approve requests, write notes, control the computer, or queue approvals.",
            "",
            "Recovery objective:",
            f"- {objective[:240]}",
            "",
            "Latest saved checkpoint:",
        ]
        if latest_path and checkpoint_text:
            lines.append(f"- checkpoint: {latest_path_display}")
            lines.append(f"- freshness: {checkpoint_freshness}")
            lines.append(f"- age minutes: {checkpoint_age_minutes if checkpoint_age_minutes is not None else 'unknown'}")
            lines.append(f"- checkpoint sha256: {latest_checkpoint_sha256 or 'missing'}")
            for label in ("Objective:", "Verification evidence:", "Open risks and blockers:", "Resume packet:", "Boundary:"):
                index = checkpoint_text.find(label)
                if index >= 0:
                    snippet = checkpoint_text[index : index + 360].strip().replace("\n", " ")
                    lines.append(f"- {label} {_safe_preview(snippet, limit=320)}")
        elif latest_path:
            lines.append(f"- checkpoint: {latest_path_display}")
            lines.append(f"- freshness: {checkpoint_freshness}")
            lines.append(f"- age minutes: {checkpoint_age_minutes if checkpoint_age_minutes is not None else 'unknown'}")
            lines.append(f"- checkpoint sha256: {latest_checkpoint_sha256 or 'missing'}")
            lines.append("- note exists but could not be read; use `recent saved notes` and `save work block checkpoint` to refresh it.")
        else:
            lines.append("- no saved work-block checkpoint found yet")

        approvals = store.list_pending_approvals(limit=6)
        recent_tool_runs, unreadable_tool_run_rows = _safe_tool_run_payloads(store.recent_tool_runs(limit=12))
        tool_run_status_counts = _tool_run_status_counts(recent_tool_runs)
        failed_runs = [row for row in recent_tool_runs if _tool_run_status(row) == "failed"]
        approval_held_runs = [row for row in recent_tool_runs if _tool_run_status(row) == "approval_held"]
        recovery_proof_queue: list[str] = []

        def add_recovery_proof(command: str) -> None:
            if command and command not in recovery_proof_queue:
                recovery_proof_queue.append(command)

        if latest_path:
            add_recovery_proof("work block checkpoint: continue Jarvis V2 safely")
        if failed_runs:
            add_recovery_proof("recovery closure checklist")
        if unreadable_tool_run_rows:
            add_recovery_proof("recent tool runs")
        for row in failed_runs[:3]:
            add_recovery_proof(f"verification receipt {row['id']}")
            add_recovery_proof(f"execution recovery packet {row['id']}")
            add_recovery_proof(f"after-action learning packet {row['id']}")
        for row in approval_held_runs[:3]:
            for command in _approval_review_commands_for_tool_run(row):
                add_recovery_proof(command)
        for row in approvals[:4]:
            add_recovery_proof(f"approval readiness {row['id']}")
            add_recovery_proof(f"approval packet {row['id']}")
            add_recovery_proof(f"approval chain proof {row['id']}")
        add_recovery_proof("build delta")
        add_recovery_proof("work block checkpoint")
        lines.extend(
            [
                "",
                "Resume-and-verify sequence:",
                "1. Reconfirm the objective and boundary from the saved checkpoint.",
                "2. Inspect pending approvals before any risky continuation.",
                "3. Pick one local-safe or read-only harness task from the checkpoint.",
                "4. Implement the smallest scoped patch.",
                "5. Run py_compile for touched Python files and the nearest smoke test.",
                "6. Save a fresh work-block checkpoint with tests, blockers, and next safe command.",
                "",
                "Current blockers:",
            ]
        )
        if approvals:
            for row in approvals[:4]:
                lines.append(f"- pending approval #{row['id']}: {_safe_preview(row['tool_name'], limit=80)} for `{_safe_preview(row['user_input'], limit=120)}`")
        if failed_runs:
            for row in failed_runs[:3]:
                snippet = _safe_preview(row["output"], limit=120)
                lines.append(f"- failed/blocked {_safe_preview(row['tool_name'], limit=80)}: {snippet}")
        if approval_held_runs:
            for row in approval_held_runs[:3]:
                lines.append(
                    f"- approval-held {_safe_preview(row['tool_name'], limit=80)}: "
                    "safety gate held before execution"
                )
                for command in _approval_review_commands_for_tool_run(row)[:3]:
                    lines.append(f"  review: `{command}`")
        if unreadable_tool_run_rows:
            lines.append(f"- unreadable tool run row(s) hidden for safety: {unreadable_tool_run_rows}")
        if not approvals and not failed_runs and not approval_held_runs and not unreadable_tool_run_rows:
            lines.append("- none detected in the reviewed window")

        lines.extend(
            [
                "",
                "Safe next command:",
                "- `work block checkpoint: continue Jarvis V2 safely` after the next verified patch, or `approval review` if a risky task is required.",
                "",
                "Checkpoint recovery proof queue:",
                f"- next required command: `{recovery_proof_queue[0]}`" if recovery_proof_queue else "- next required command: none",
                f"- proof queue count: {len(recovery_proof_queue)}",
            ]
        )
        lines.extend(f"- `{command}`" for command in recovery_proof_queue[:8])
        return ToolResult(
            "checkpoint_recovery_preview",
            True,
            "\n".join(lines),
            _safe_metadata(
                reads_note_contents=bool(latest_path and checkpoint_text),
                checkpoint_found=bool(latest_path),
                checkpoint_read=bool(checkpoint_text),
                latest_checkpoint_path=str(latest_path) if latest_path else "",
                latest_checkpoint_display=latest_path_display,
                latest_checkpoint_sha256=latest_checkpoint_sha256,
                latest_checkpoint_hash_present=_looks_like_sha256(latest_checkpoint_sha256),
                checkpoint_age_minutes=checkpoint_age_minutes,
                checkpoint_freshness=checkpoint_freshness,
                checkpoint_stale=checkpoint_freshness == "stale",
                checkpoint_needs_review=checkpoint_freshness in {"review_again", "stale", "missing", "unknown"},
                checkpoint_notes=len(checkpoint_files),
                pending_approvals=len(approvals),
                failed_runs=len(failed_runs),
                approval_held_runs=len(approval_held_runs),
                recent_ok_tool_runs=tool_run_status_counts["ok"],
                recent_failed_tool_runs=len(failed_runs),
                recent_approval_held_tool_runs=len(approval_held_runs),
                readable_tool_run_rows=len(recent_tool_runs),
                unreadable_tool_run_rows=unreadable_tool_run_rows,
                objective_length=len(objective),
                checkpoint_recovery_proof_queue=recovery_proof_queue,
                checkpoint_recovery_proof_queue_count=len(recovery_proof_queue),
                checkpoint_recovery_next_proof_command=recovery_proof_queue[0] if recovery_proof_queue else "",
                checkpoint_recovery_next_required_command=recovery_proof_queue[0] if recovery_proof_queue else "",
                next_required_command=recovery_proof_queue[0] if recovery_proof_queue else "",
            ),
        )

    def checkpoint_recovery_apply_packet(args: dict[str, Any]) -> ToolResult:
        objective = str(args.get("objective") or args.get("request") or "Resume one checkpointed Jarvis task.").strip()
        packet = checkpoint_recovery_preview({"objective": objective})
        approvals = store.list_pending_approvals(limit=6)
        recovery_queue = list(packet.metadata.get("checkpoint_recovery_proof_queue") or [])
        recovery_next_required = str(packet.metadata.get("checkpoint_recovery_next_required_command") or "")
        lines = [
            "Jarvis checkpoint recovery apply packet:",
            "This is approval-gated planning only. It does not apply patches, run shell/code, approve requests, write notes, control the computer, or touch private data.",
            "",
            "Target:",
            f"- {objective[:240]}",
            "",
            "Preflight:",
            "- latest checkpoint recovery preview is available",
            f"- pending approvals visible: {len(approvals)}",
            "- exact file edits and commands must be reviewed before any risky execution",
            "",
            "Approval-gated apply sequence:",
            "1. Select one checkpointed task that is read-only or local-safe.",
            "2. Produce an exact patch plan with files, expected behavior, and rollback note.",
            "3. Stop before shell/code execution unless the operator approves the exact command.",
            "4. Run the approved or local-safe implementation step.",
            "5. Verify with py_compile and the nearest smoke test.",
            "6. Save a fresh work-block checkpoint that includes verification evidence.",
            "",
            "Recovery proof queue:",
            f"- next required command: `{recovery_next_required}`" if recovery_next_required else "- next required command: none",
            f"- proof queue count: {len(recovery_queue)}",
            "",
            "Recovery preview excerpt:",
            packet.output[:1600],
            "",
            "Stop conditions:",
            "- missing checkpoint",
            "- unclear target file",
            "- any personal-data, external-side-effect, destructive, computer-control, or shell/code step without approval",
            "- verification failure that needs a broader refactor",
            "",
            "Boundary:",
            "- read-only apply packet; this packet is a steering wheel and brake, not the action itself.",
        ]
        return ToolResult(
            "checkpoint_recovery_apply_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                checkpoint_found=bool(packet.metadata.get("checkpoint_found")),
                checkpoint_read=bool(packet.metadata.get("checkpoint_read")),
                checkpoint_age_minutes=packet.metadata.get("checkpoint_age_minutes"),
                checkpoint_freshness=packet.metadata.get("checkpoint_freshness"),
                checkpoint_stale=packet.metadata.get("checkpoint_stale", False),
                checkpoint_needs_review=packet.metadata.get("checkpoint_needs_review", True),
                pending_approvals=len(approvals),
                objective_length=len(objective),
                apply_requires_approval=True,
                checkpoint_recovery_proof_queue=packet.metadata.get("checkpoint_recovery_proof_queue", []),
                checkpoint_recovery_proof_queue_count=packet.metadata.get("checkpoint_recovery_proof_queue_count", 0),
                checkpoint_recovery_next_proof_command=packet.metadata.get("checkpoint_recovery_next_proof_command", ""),
                checkpoint_recovery_next_required_command=packet.metadata.get("checkpoint_recovery_next_required_command", ""),
                next_required_command=packet.metadata.get("checkpoint_recovery_next_required_command", ""),
                recovery_apply_proof_queue=packet.metadata.get("checkpoint_recovery_proof_queue", []),
                recovery_apply_proof_queue_count=packet.metadata.get("checkpoint_recovery_proof_queue_count", 0),
                recovery_apply_next_proof_command=packet.metadata.get("checkpoint_recovery_next_proof_command", ""),
                recovery_apply_next_required_command=packet.metadata.get("checkpoint_recovery_next_required_command", ""),
            ),
        )

    def checkpoint_recovery_receipt(args: dict[str, Any]) -> ToolResult:
        objective = str(args.get("objective") or args.get("request") or "Resume one checkpointed Jarvis task.").strip()
        approved_step = str(args.get("approved_step") or args.get("step") or "No approved step recorded.").strip()
        verification = str(args.get("verification") or args.get("tests") or "No verification recorded.").strip()
        outcome = str(args.get("outcome") or "pending review").strip()
        files = str(args.get("files") or args.get("file") or "not specified").strip()
        approval_reference = str(args.get("approval") or args.get("approval_reference") or "not required or not supplied").strip()
        blockers = str(args.get("blockers") or "none reported").strip()
        lines = [
            "Jarvis checkpoint recovery receipt:",
            "This records a reviewed recovery step after the fact. It does not apply patches, run shell/code, approve requests, control the computer, touch private data, call external services, or queue approvals.",
            "",
            "Objective:",
            f"- {objective[:300]}",
            "",
            "Reviewed step:",
            f"- {approved_step[:500]}",
            "",
            "Verification evidence:",
            f"- {verification[:700]}",
            "",
            "Outcome:",
            f"- {outcome[:300]}",
            "",
            "Touched files:",
            f"- {files[:500]}",
            "",
            "Approval reference:",
            f"- {approval_reference[:300]}",
            "",
            "Remaining blockers:",
            f"- {blockers[:500]}",
            "",
            "Boundary:",
            "- local-safe receipt write only; it records evidence for continuity and does not grant future approval.",
        ]
        body = "\n".join(lines)
        path = vault.write_reflection("Checkpoint Recovery Receipt", body)
        path_display = _safe_vault_path_display(path, vault)
        receipt_sha256 = _file_sha256(str(path))
        return ToolResult(
            "checkpoint_recovery_receipt",
            True,
            f"Checkpoint recovery receipt saved: {path_display}\n\n{body}",
            _safe_metadata(
                objective_length=len(objective),
                approved_step_provided=approved_step != "No approved step recorded.",
                verification_provided=verification != "No verification recorded.",
                files_provided=files != "not specified",
                approval_reference_provided=approval_reference != "not required or not supplied",
                blockers_provided=blockers != "none reported",
                path=str(path),
                path_display=path_display,
                receipt_sha256=receipt_sha256,
                receipt_hash_present=bool(receipt_sha256),
                writes_files=True,
                writes_notes=True,
            ),
        )

    def checkpoint_recovery_execute(args: dict[str, Any]) -> ToolResult:
        objective = str(args.get("objective") or args.get("request") or "Resume one checkpointed Jarvis task.").strip()
        reviewed_raw = str(args.get("reviewed") or args.get("approved") or args.get("confirmed") or "").strip().lower()
        reviewed = reviewed_raw in {"1", "true", "yes", "y", "reviewed", "approved", "confirmed"}
        approved_step = str(args.get("approved_step") or args.get("step") or "").strip()
        verification = str(args.get("verification") or args.get("tests") or "").strip()
        files = str(args.get("files") or args.get("file") or "not specified").strip()
        outcome = str(args.get("outcome") or "recovery step recorded").strip()
        approval_reference = str(args.get("approval") or args.get("approval_reference") or "not required for local-safe receipt").strip()
        blockers = str(args.get("blockers") or "none reported").strip()
        receipt_sha256 = str(args.get("receipt_sha256") or args.get("receipt_hash") or "").strip()
        checkpoint_sha256 = str(args.get("checkpoint_sha256") or args.get("checkpoint_hash") or "").strip()
        stop_at = str(args.get("stop_at") or args.get("stop") or args.get("until") or "").strip()
        current_time = str(args.get("current_time") or args.get("now") or "").strip()
        timezone_label = str(args.get("timezone") or "local").strip() or "local"
        previous_instruction = str(args.get("previous_instruction") or args.get("previous") or "").strip()
        latest_instruction = str(args.get("latest_instruction") or args.get("latest") or args.get("instruction") or objective).strip()
        timebox = operator_timebox_contract(
            {
                "objective": objective,
                "stop_at": stop_at,
                "current_time": current_time,
                "timezone": timezone_label,
            }
        )
        timebox_metadata = timebox.metadata
        timebox_review_contract_rows = list(timebox_metadata.get("timebox_review_contract_rows") or [])
        timebox_review_contract_summary = list(timebox_metadata.get("timebox_review_contract_summary") or [])
        supersession = operator_instruction_supersession_packet(
            {
                "objective": objective,
                "previous_instruction": previous_instruction,
                "latest_instruction": latest_instruction,
                "stop_at": stop_at,
                "current_time": current_time,
                "timezone": timezone_label,
            }
        )
        supersession_metadata = supersession.metadata
        supersession_token_boundary_rows = list(supersession_metadata.get("supersession_token_boundary_rows") or [])
        supersession_token_boundary_ready = _operator_supersession_token_boundary_ready(
            str(supersession_metadata.get("supersession_token_sha256") or ""),
            supersession_token_boundary_rows,
            expected_source="operator_instruction_supersession",
        )
        packet = checkpoint_recovery_apply_packet({"objective": objective})
        risk_signals = _recovery_risk_signals(" ".join([approved_step, files, outcome]))
        approval_reference_present = (
            _approval_chain_reference_provided(approval_reference)
            if risk_signals
            else _approval_reference_provided(approval_reference)
        )
        recovery_step_sha256 = _text_sha256(approved_step)
        recovery_verification_sha256 = _text_sha256(verification)
        recovery_step_approval_proof_queue = (
            _approval_proof_queue_for_risky_work(
                followup_command="checkpoint recovery execute: <reviewed local-safe recovery step after approval proof>"
            )
            if risk_signals and not approval_reference_present
            else []
        )
        recovery_step_approval_boundary_rows = _approval_boundary_rows_for_risky_work(
            risk_signals=risk_signals,
            step_sha256=recovery_step_sha256,
            verification_sha256=recovery_verification_sha256,
            proof_queue=recovery_step_approval_proof_queue,
            required_field="required_before_risky_recovery_step",
        )
        recovery_step_approval_boundary_token_sha256 = _risky_recovery_step_approval_boundary_token_sha256(
            recovery_step_sha256=recovery_step_sha256,
            recovery_verification_sha256=recovery_verification_sha256,
            risk_signals=risk_signals,
            approval_proof_queue=recovery_step_approval_proof_queue,
            approval_boundary_rows=recovery_step_approval_boundary_rows,
            approval_reference=approval_reference if approval_reference_present else "",
        )
        recovery_step_approval_boundary_ready = _recovery_step_approval_boundary_ready(
            token_sha256=recovery_step_approval_boundary_token_sha256,
            recovery_step_sha256=recovery_step_sha256,
            recovery_verification_sha256=recovery_verification_sha256,
            approval_proof_queue=recovery_step_approval_proof_queue,
            approval_boundary_rows=recovery_step_approval_boundary_rows,
            approval_reference=approval_reference if approval_reference_present else "",
        )
        recovery_step_approval_boundary_as_prior_proof = _recovery_step_approval_boundary_ready(
            token_sha256=recovery_step_approval_boundary_token_sha256,
            recovery_step_sha256=recovery_step_sha256,
            recovery_verification_sha256=recovery_verification_sha256,
            approval_proof_queue=recovery_step_approval_proof_queue,
            approval_boundary_rows=recovery_step_approval_boundary_rows,
            approval_reference=approval_reference if approval_reference_present else "",
            require_prior_proof=True,
        )

        missing = []
        if not reviewed:
            missing.append("reviewed=true")
        if not approved_step:
            missing.append("approved_step")
        if not verification:
            missing.append("verification")
        risky_without_approval = bool(risk_signals and not approval_reference_present)
        if risky_without_approval:
            missing.append("approval_reference_for_risky_recovery_step")
        if missing:
            closure_gate = _recovery_closure_gate(
                reviewed=reviewed,
                approved_step=approved_step,
                verification=verification,
                receipt_sha256=receipt_sha256,
                checkpoint_sha256=checkpoint_sha256,
                risk_signals=risk_signals,
                approval_reference_present=approval_reference_present,
                blockers=blockers,
            )
            recovery_closure_proof_queue = [closure_gate["next_safe_command"]]
            recovery_closure_next_proof_command = recovery_closure_proof_queue[0]
            recovery_closure_proof_queue_ready = (
                bool(recovery_closure_proof_queue)
                and recovery_closure_next_proof_command == recovery_closure_proof_queue[0]
                and closure_gate["next_safe_command"] == recovery_closure_proof_queue[0]
                and len(recovery_closure_proof_queue) == 1
            )
            recovery_execute_handoff = {
                "handoff_ready": True,
                "state": "HELD_BEFORE_REVIEW",
                "reviewed": reviewed,
                "missing_fields": list(missing),
                "missing_field_count": len(missing),
                "normal_followthrough_allowed": False,
                "recovery_followthrough_gate_state": closure_gate["state"],
                "recovery_closure_missing": list(closure_gate["missing"]),
                "recovery_closure_missing_count": closure_gate["missing_count"],
                "recovery_closure_required_evidence": list(closure_gate["required_evidence"]),
                "recovery_closure_required_evidence_count": len(closure_gate["required_evidence"]),
                "recovery_closure_proof_queue": list(recovery_closure_proof_queue),
                "recovery_closure_proof_queue_count": len(recovery_closure_proof_queue),
                "recovery_closure_next_proof_command": recovery_closure_next_proof_command,
                "recovery_closure_proof_queue_ready": recovery_closure_proof_queue_ready,
                "recovery_closure_approval_boundary": closure_gate["approval_boundary"],
                "next_safe_command": closure_gate["next_safe_command"],
                "risky_recovery_signals": list(risk_signals),
                "risky_recovery_signal_count": len(risk_signals),
                "approval_reference_provided": approval_reference_present,
                "approval_reference": approval_reference[:300] if approval_reference_present else "",
                "recovery_step_approval_required_before_recovery": risky_without_approval,
                "recovery_step_approval_proof_queue": list(recovery_step_approval_proof_queue),
                "recovery_step_approval_proof_queue_count": len(recovery_step_approval_proof_queue),
                "recovery_step_next_approval_proof_command": (
                    recovery_step_approval_proof_queue[0] if recovery_step_approval_proof_queue else ""
                ),
                "recovery_step_approval_boundary_ready": recovery_step_approval_boundary_ready,
                "recovery_step_approval_boundary_as_prior_proof": recovery_step_approval_boundary_as_prior_proof,
                **{flag: False for flag in _RECOVERY_STEP_APPROVAL_BOUNDARY_FALSE_FLAGS},
                "recovery_step_approval_boundary_rows": recovery_step_approval_boundary_rows,
                "recovery_step_approval_boundary_row_count": len(recovery_step_approval_boundary_rows),
                "recovery_step_approval_boundary_token_sha256": recovery_step_approval_boundary_token_sha256,
                "recovery_step_approval_boundary_token_present": _looks_like_sha256(recovery_step_approval_boundary_token_sha256),
                "recovery_step_sha256": recovery_step_sha256,
                "recovery_verification_sha256": recovery_verification_sha256,
                "contains_raw_step": False,
                "contains_raw_verification": False,
                "changed_state": False,
                "writes_notes": False,
                "writes_files": False,
                "calls_model": False,
                "executes_tools": False,
                "queues_approval": False,
                "controls_computer": False,
                "reads_private_data": False,
                "reads_personal_data": False,
                "executes_side_effect": False,
                "external_side_effect": False,
                "authorizes_execution": False,
                "authorizes_recovery_followthrough": False,
                "authorizes_local_safe_step": False,
                "authorizes_risky_work": False,
                "authorizes_approval": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "reusable_for_recovery_review": False,
            }
            recovery_execute_handoff_token_sha256 = _checkpoint_recovery_execute_handoff_token_sha256(
                recovery_execute_handoff
            )
            recovery_execute_handoff["handoff_token_sha256"] = recovery_execute_handoff_token_sha256
            recovery_execute_handoff["handoff_token_present"] = _looks_like_sha256(
                recovery_execute_handoff_token_sha256
            )
            precontinuation_contract_metadata = {
                "timebox_state": timebox_metadata.get("timebox_state", ""),
                "can_continue_now": timebox_metadata.get("can_continue_now", False),
                "should_stop_now": timebox_metadata.get("should_stop_now", True),
                "timebox_review_contract_ready": _metadata_bool(timebox_metadata.get("timebox_review_contract_ready", False)),
                "operator_timebox_precontinuation_contract_ready": _metadata_bool(timebox_metadata.get("timebox_review_contract_ready", False)),
                "operator_timebox_precontinuation_contract_state": timebox_metadata.get("timebox_state", ""),
                "operator_timebox_precontinuation_allows_continuation": _metadata_bool(timebox_metadata.get("can_continue_now", False)),
                "operator_timebox_precontinuation_blocks_continuation": not _metadata_bool(
                    timebox_metadata.get("can_continue_now", False)
                ),
                "operator_timebox_precontinuation_requires_stop": _metadata_bool(timebox_metadata.get("should_stop_now", True), default=True),
                "operator_timebox_precontinuation_requires_fresh_timebox_review": True,
                "operator_timebox_precontinuation_allowed_states": [
                    "STOP_WINDOW_ACTIVE",
                    "STOP_TIME_REACHED",
                    "HELD_FOR_PARSEABLE_TIMEBOX",
                ],
                "supersession_state": supersession_metadata.get("supersession_state", ""),
                "can_continue_under_latest_instruction": supersession_metadata.get(
                    "can_continue_under_latest_instruction", False
                ),
                "latest_instruction_is_stop": supersession_metadata.get("latest_instruction_is_stop", False),
                "supersession_contract_ready": supersession_metadata.get("supersession_contract_ready", False),
                "supersession_token_boundary_ready": supersession_token_boundary_ready,
                "operator_supersession_precontinuation_contract_ready": supersession_metadata.get(
                    "supersession_contract_ready", False
                )
                and supersession_token_boundary_ready,
                "operator_supersession_precontinuation_contract_state": supersession_metadata.get(
                    "supersession_state", ""
                ),
                "operator_supersession_precontinuation_token_boundary_ready": supersession_token_boundary_ready,
                "operator_supersession_precontinuation_allows_continuation": supersession_metadata.get(
                    "can_continue_under_latest_instruction", False
                ),
                "operator_supersession_precontinuation_blocks_continuation": not bool(
                    supersession_metadata.get("can_continue_under_latest_instruction", False)
                ),
                "operator_supersession_precontinuation_requires_stop": supersession_metadata.get(
                    "latest_instruction_is_stop", False
                ),
                "operator_supersession_precontinuation_requires_fresh_latest_instruction_review": True,
                "operator_supersession_precontinuation_allowed_states": [
                    "LATEST_INSTRUCTION_READY_TO_GOVERN_CONTINUATION",
                    "NEWER_STOP_OR_PAUSE_OVERRIDES_AUTONOMY",
                    "HELD_FOR_LATEST_INSTRUCTION_REVIEW",
                ],
            }
            recovery_execute_handoff_ready = _checkpoint_recovery_execute_handoff_ready(
                {
                    "checkpoint_recovery_execute_handoff": recovery_execute_handoff,
                    "checkpoint_recovery_execute_handoff_token_sha256": recovery_execute_handoff_token_sha256,
                    "checkpoint_recovery_execute_handoff_token_present": _looks_like_sha256(
                        recovery_execute_handoff_token_sha256
                    ),
                    "checkpoint_recovery_execute_handoff_state": recovery_execute_handoff["state"],
                    "checkpoint_recovery_execute_handoff_normal_followthrough_allowed": False,
                    "reviewed": reviewed,
                    "checkpoint_recovery_execute_handoff_reviewed": reviewed,
                    "recovery_followthrough_gate_state": closure_gate["state"],
                    "checkpoint_recovery_execute_handoff_recovery_followthrough_gate_state": closure_gate["state"],
                    "missing_fields": missing,
                    "checkpoint_recovery_execute_handoff_missing_fields": list(missing),
                    "checkpoint_recovery_execute_handoff_missing_field_count": len(missing),
                    "recovery_closure_missing": list(closure_gate["missing"]),
                    "checkpoint_recovery_execute_handoff_recovery_closure_missing": list(closure_gate["missing"]),
                    "recovery_closure_missing_count": closure_gate["missing_count"],
                    "checkpoint_recovery_execute_handoff_recovery_closure_missing_count": closure_gate["missing_count"],
                    "recovery_closure_required_evidence": list(closure_gate["required_evidence"]),
                    "recovery_closure_required_evidence_count": len(closure_gate["required_evidence"]),
                    "checkpoint_recovery_execute_handoff_recovery_closure_required_evidence": list(closure_gate["required_evidence"]),
                    "checkpoint_recovery_execute_handoff_recovery_closure_required_evidence_count": len(closure_gate["required_evidence"]),
                    "recovery_closure_proof_queue": list(recovery_closure_proof_queue),
                    "recovery_closure_proof_queue_count": len(recovery_closure_proof_queue),
                    "checkpoint_recovery_execute_handoff_recovery_closure_proof_queue": list(recovery_closure_proof_queue),
                    "checkpoint_recovery_execute_handoff_recovery_closure_proof_queue_count": len(recovery_closure_proof_queue),
                    "recovery_closure_next_proof_command": recovery_closure_next_proof_command,
                    "checkpoint_recovery_execute_handoff_recovery_closure_next_proof_command": recovery_closure_next_proof_command,
                    "recovery_closure_proof_queue_ready": recovery_closure_proof_queue_ready,
                    "checkpoint_recovery_execute_handoff_recovery_closure_proof_queue_ready": recovery_closure_proof_queue_ready,
                    "recovery_closure_approval_boundary": closure_gate["approval_boundary"],
                    "checkpoint_recovery_execute_handoff_recovery_closure_approval_boundary": closure_gate["approval_boundary"],
                    "recovery_closure_next_safe_command": closure_gate["next_safe_command"],
                    "checkpoint_recovery_execute_handoff_next_safe_command": closure_gate["next_safe_command"],
                    "risky_recovery_signals": list(risk_signals),
                    "checkpoint_recovery_execute_handoff_risky_recovery_signals": list(risk_signals),
                    "risky_recovery_signal_count": len(risk_signals),
                    "checkpoint_recovery_execute_handoff_risky_recovery_signal_count": len(risk_signals),
                    "approval_reference_provided": approval_reference_present,
                    "approval_reference": approval_reference if approval_reference_present else "",
                    "checkpoint_recovery_execute_handoff_approval_reference_provided": approval_reference_present,
                    "recovery_step_approval_required_before_recovery": risky_without_approval,
                    "checkpoint_recovery_execute_handoff_recovery_step_approval_required_before_recovery": risky_without_approval,
                    "recovery_step_approval_boundary_rows": recovery_step_approval_boundary_rows,
                    "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_rows": recovery_step_approval_boundary_rows,
                    "recovery_step_approval_boundary_row_count": len(recovery_step_approval_boundary_rows),
                    "recovery_step_approval_boundary_ready": recovery_step_approval_boundary_ready,
                    "recovery_step_approval_boundary_token_present": _looks_like_sha256(
                        recovery_step_approval_boundary_token_sha256
                    ),
                    "recovery_step_approval_boundary_as_prior_proof": recovery_step_approval_boundary_as_prior_proof,
                    "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_as_prior_proof": recovery_step_approval_boundary_as_prior_proof,
                    **{flag: False for flag in _RECOVERY_STEP_APPROVAL_BOUNDARY_FALSE_FLAGS},
                    **{
                        f"checkpoint_recovery_execute_handoff_{flag}": False
                        for flag in _RECOVERY_STEP_APPROVAL_BOUNDARY_FALSE_FLAGS
                    },
                    "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_row_count": len(
                        recovery_step_approval_boundary_rows
                    ),
                    "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_ready": recovery_step_approval_boundary_ready,
                    "recovery_step_approval_proof_queue": list(recovery_step_approval_proof_queue),
                    "checkpoint_recovery_execute_handoff_recovery_step_approval_proof_queue": list(recovery_step_approval_proof_queue),
                    "recovery_step_approval_proof_queue_count": len(recovery_step_approval_proof_queue),
                    "recovery_step_next_approval_proof_command": (
                        recovery_step_approval_proof_queue[0] if recovery_step_approval_proof_queue else ""
                    ),
                    "checkpoint_recovery_execute_handoff_recovery_step_approval_proof_queue_count": len(recovery_step_approval_proof_queue),
                    "checkpoint_recovery_execute_handoff_recovery_step_next_approval_proof_command": (
                        recovery_step_approval_proof_queue[0] if recovery_step_approval_proof_queue else ""
                    ),
                    "recovery_step_approval_boundary_token_sha256": recovery_step_approval_boundary_token_sha256,
                    "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_token_sha256": recovery_step_approval_boundary_token_sha256,
                    "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_token_present": _looks_like_sha256(
                        recovery_step_approval_boundary_token_sha256
                    ),
                    "recovery_step_sha256": recovery_step_sha256,
                    "checkpoint_recovery_execute_handoff_recovery_step_sha256": recovery_step_sha256,
                    "recovery_verification_sha256": recovery_verification_sha256,
                    "checkpoint_recovery_execute_handoff_recovery_verification_sha256": recovery_verification_sha256,
                    **{
                        flag: False
                        for flag in _CHECKPOINT_RECOVERY_EXECUTE_HANDOFF_FALSE_FLAGS
                    },
                    **{
                        f"checkpoint_recovery_execute_handoff_{flag}": False
                        for flag in _CHECKPOINT_RECOVERY_EXECUTE_HANDOFF_FALSE_FLAGS
                    },
                    **precontinuation_contract_metadata,
                }
            )
            lines = [
                "Jarvis checkpoint recovery executor:",
                "Execution held. A recovery step must be reviewed before Jarvis records it as applied.",
                "",
                "Command-first held handoff:",
                f"- handoff ready: {'yes' if recovery_execute_handoff_ready else 'no'}",
                f"- state: {recovery_execute_handoff['state']}",
                f"- next safe command: `{recovery_execute_handoff['next_safe_command']}`",
                "- contains raw step/verification: no",
                "- authorizes recovery follow-through: no",
                "",
                "Missing required fields:",
                *[f"- {item}" for item in missing],
                "",
                "Required local-safe command shape:",
                "- `checkpoint recovery execute reviewed=true step=<reviewed local-safe step> verification=<test or evidence>`",
                "",
                "Risk and approval check:",
                f"- risky recovery signals: {', '.join(risk_signals) if risk_signals else 'none'}",
                f"- approval reference supplied: {'yes' if approval_reference_present else 'no'}",
                "- risky recovery steps need an approval reference before Jarvis records them as applied",
                "",
                "Risky recovery approval proof queue:",
                f"- approval required before recovery: {'yes' if risky_without_approval else 'no'}",
                f"- proof queue count: {len(recovery_step_approval_proof_queue)}",
                f"- boundary token sha256: {recovery_step_approval_boundary_token_sha256}",
                f"- boundary rows: {len(recovery_step_approval_boundary_rows)}",
                f"- boundary ready: {'yes' if recovery_step_approval_boundary_ready else 'no'}",
                f"- next approval proof command: `{recovery_step_approval_proof_queue[0] if recovery_step_approval_proof_queue else 'none'}`",
                "- boundary authorizes action now: no",
                "- boundary authorizes risky work: no",
                *[f"- `{command}`" for command in recovery_step_approval_proof_queue],
                *[
                    f"- boundary {row['item']}: {row['status']} ({'ready' if row['ready'] else 'held'}; non-authorizing)"
                    for row in recovery_step_approval_boundary_rows
                ],
                "",
                "Recovery closure gate:",
                f"- state: {closure_gate['state']}",
                "- normal follow-through allowed: no",
                f"- missing closure proof: {', '.join(closure_gate['missing']) if closure_gate['missing'] else 'none'}",
                f"- proof queue ready: {'yes' if recovery_closure_proof_queue_ready else 'no'}",
                f"- proof queue count: {len(recovery_closure_proof_queue)}",
                f"- next required command: `{recovery_closure_next_proof_command}`",
                f"- next safe command: `{closure_gate['next_safe_command']}`",
                "",
                "Operator timebox contract before continuation:",
                f"- state: {timebox_metadata.get('timebox_state')}",
                f"- can continue now: {'yes' if timebox_metadata.get('can_continue_now') else 'no'}",
                f"- should stop now: {'yes' if timebox_metadata.get('should_stop_now') else 'no'}",
                f"- missing proof: {', '.join(timebox_metadata.get('missing_timebox_proof') or []) if timebox_metadata.get('missing_timebox_proof') else 'none'}",
                "- timebox authorizes execution: no",
                "- next continuation still requires a fresh timebox: yes",
                "",
                "Operator instruction supersession before continuation:",
                f"- state: {supersession_metadata.get('supersession_state')}",
                f"- latest instruction can govern continuation: {'yes' if supersession_metadata.get('can_continue_under_latest_instruction') else 'no'}",
                f"- latest instruction is stop/pause: {'yes' if supersession_metadata.get('latest_instruction_is_stop') else 'no'}",
                f"- supersession token sha256: {supersession_metadata.get('supersession_token_sha256') or 'missing'}",
                "- supersession authorizes execution: no",
                "- supersession authorizes recovery follow-through: no",
                "- next continuation still requires a fresh latest-instruction review: yes",
                "",
                "Apply packet excerpt:",
                packet.output[:1200],
                "",
                "Boundary:",
                "- This held executor does not apply patches, run shell/code, approve requests, read private data, control the computer, call external services, write notes, or queue approvals.",
            ]
            return ToolResult(
                "checkpoint_recovery_execute",
                False,
                "\n".join(lines),
                _safe_metadata(
                    objective_length=len(objective),
                    reviewed=reviewed,
                    missing_fields=missing,
                    risky_recovery_signals=risk_signals,
                    risky_recovery_signal_count=len(risk_signals),
                    risky_recovery_without_approval=risky_without_approval,
                    recovery_step_sha256=recovery_step_sha256,
                    recovery_verification_sha256=recovery_verification_sha256,
                    recovery_step_approval_proof_queue=recovery_step_approval_proof_queue,
                    recovery_step_approval_proof_queue_count=len(recovery_step_approval_proof_queue),
                    recovery_step_next_approval_proof_command=(
                        recovery_step_approval_proof_queue[0] if recovery_step_approval_proof_queue else ""
                    ),
                    recovery_step_approval_required_before_recovery=risky_without_approval,
                    recovery_step_approval_boundary_rows=recovery_step_approval_boundary_rows,
                    checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_rows=recovery_step_approval_boundary_rows,
                    recovery_step_approval_boundary_row_count=len(recovery_step_approval_boundary_rows),
                    recovery_step_approval_boundary_ready=recovery_step_approval_boundary_ready,
                    recovery_step_approval_boundary_token_sha256=recovery_step_approval_boundary_token_sha256,
                    recovery_step_approval_boundary_token_present=_looks_like_sha256(recovery_step_approval_boundary_token_sha256),
                    recovery_step_approval_boundary_as_prior_proof=recovery_step_approval_boundary_as_prior_proof,
                    checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_as_prior_proof=recovery_step_approval_boundary_as_prior_proof,
                    **{flag: False for flag in _RECOVERY_STEP_APPROVAL_BOUNDARY_FALSE_FLAGS},
                    **{
                        f"checkpoint_recovery_execute_handoff_{flag}": False
                        for flag in _RECOVERY_STEP_APPROVAL_BOUNDARY_FALSE_FLAGS
                    },
                    contains_raw_step=False,
                    contains_raw_verification=False,
                    changed_state=False,
                    authorizes_recovery_followthrough=False,
                    authorizes_local_safe_step=False,
                    authorizes_risky_work=False,
                    authorizes_approval=False,
                    authorizes_model_call=False,
                    authorizes_tool_execution=False,
                    authorizes_personal_data_read=False,
                    authorizes_external_side_effect=False,
                    reusable_for_recovery_review=False,
                    checkpoint_recovery_execute_handoff=recovery_execute_handoff,
                    checkpoint_recovery_execute_handoff_ready=recovery_execute_handoff_ready,
                    checkpoint_recovery_execute_handoff_token_sha256=recovery_execute_handoff_token_sha256,
                    checkpoint_recovery_execute_handoff_token_present=_looks_like_sha256(
                        recovery_execute_handoff_token_sha256
                    ),
                    checkpoint_recovery_execute_handoff_state=recovery_execute_handoff["state"],
                    checkpoint_recovery_execute_handoff_reviewed=reviewed,
                    checkpoint_recovery_execute_handoff_missing_fields=list(missing),
                    checkpoint_recovery_execute_handoff_missing_field_count=len(missing),
                    checkpoint_recovery_execute_handoff_normal_followthrough_allowed=False,
                    checkpoint_recovery_execute_handoff_recovery_followthrough_gate_state=closure_gate["state"],
                    checkpoint_recovery_execute_handoff_recovery_closure_missing=list(closure_gate["missing"]),
                    checkpoint_recovery_execute_handoff_recovery_closure_missing_count=closure_gate["missing_count"],
                    checkpoint_recovery_execute_handoff_recovery_closure_required_evidence=list(closure_gate["required_evidence"]),
                    checkpoint_recovery_execute_handoff_recovery_closure_required_evidence_count=len(closure_gate["required_evidence"]),
                    checkpoint_recovery_execute_handoff_recovery_closure_proof_queue=list(recovery_closure_proof_queue),
                    checkpoint_recovery_execute_handoff_recovery_closure_proof_queue_count=len(recovery_closure_proof_queue),
                    checkpoint_recovery_execute_handoff_recovery_closure_next_proof_command=recovery_closure_next_proof_command,
                    checkpoint_recovery_execute_handoff_recovery_closure_proof_queue_ready=recovery_closure_proof_queue_ready,
                    checkpoint_recovery_execute_handoff_recovery_closure_approval_boundary=closure_gate["approval_boundary"],
                    checkpoint_recovery_execute_handoff_next_safe_command=closure_gate["next_safe_command"],
                    checkpoint_recovery_execute_handoff_risky_recovery_signals=list(risk_signals),
                    checkpoint_recovery_execute_handoff_risky_recovery_signal_count=len(risk_signals),
                    checkpoint_recovery_execute_handoff_approval_reference_provided=approval_reference_present,
                    checkpoint_recovery_execute_handoff_recovery_step_approval_required_before_recovery=risky_without_approval,
                    checkpoint_recovery_execute_handoff_recovery_step_approval_proof_queue=list(recovery_step_approval_proof_queue),
                    checkpoint_recovery_execute_handoff_recovery_step_approval_proof_queue_count=len(recovery_step_approval_proof_queue),
                    checkpoint_recovery_execute_handoff_recovery_step_next_approval_proof_command=(
                        recovery_step_approval_proof_queue[0] if recovery_step_approval_proof_queue else ""
                    ),
                    checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_row_count=len(recovery_step_approval_boundary_rows),
                    checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_ready=recovery_step_approval_boundary_ready,
                    checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_token_sha256=recovery_step_approval_boundary_token_sha256,
                    checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_token_present=_looks_like_sha256(recovery_step_approval_boundary_token_sha256),
                    checkpoint_recovery_execute_handoff_recovery_step_sha256=recovery_step_sha256,
                    checkpoint_recovery_execute_handoff_recovery_verification_sha256=recovery_verification_sha256,
                    stop_at=stop_at,
                    current_time=timebox_metadata.get("current_time", ""),
                    timezone=timezone_label,
                    timebox_state=timebox_metadata.get("timebox_state", ""),
                    minutes_remaining=timebox_metadata.get("minutes_remaining"),
                    can_continue_now=timebox_metadata.get("can_continue_now", False),
                    should_stop_now=timebox_metadata.get("should_stop_now", True),
                    missing_timebox_proof=timebox_metadata.get("missing_timebox_proof", []),
                    missing_timebox_proof_count=timebox_metadata.get("missing_timebox_proof_count", 0),
                    timebox_receipt_sha256=timebox_metadata.get("timebox_receipt_sha256", ""),
                    timebox_receipt_present=timebox_metadata.get("timebox_receipt_present", False),
                    timebox_review_contract_rows=timebox_review_contract_rows,
                    timebox_review_contract_row_count=len(timebox_review_contract_rows),
                    timebox_review_contract_ready=timebox_metadata.get("timebox_review_contract_ready", False),
                    timebox_review_contract_summary=timebox_review_contract_summary,
                    timebox_authorizes_execution=False,
                    timebox_authorizes_local_safe_step=False,
                    timebox_authorizes_risky_work=False,
                    timebox_authorizes_approval=False,
                    timebox_authorizes_timebox_reuse=False,
                    timebox_authorizes_model_call=False,
                    timebox_authorizes_tool_execution=False,
                    timebox_authorizes_personal_data_read=False,
                    timebox_authorizes_external_side_effect=False,
                    timebox_reusable_for_next_step=False,
                    next_step_requires_fresh_timebox=True,
                    operator_timebox_precontinuation_contract_ready=_metadata_bool(timebox_metadata.get("timebox_review_contract_ready", False)),
                    operator_timebox_precontinuation_contract_state=timebox_metadata.get("timebox_state", ""),
                    operator_timebox_precontinuation_allows_continuation=_metadata_bool(timebox_metadata.get("can_continue_now", False)),
                    operator_timebox_precontinuation_blocks_continuation=not _metadata_bool(
                        timebox_metadata.get("can_continue_now", False)
                    ),
                    operator_timebox_precontinuation_requires_stop=_metadata_bool(timebox_metadata.get("should_stop_now", True), default=True),
                    operator_timebox_precontinuation_requires_fresh_timebox_review=True,
                    operator_timebox_precontinuation_allowed_states=[
                        "STOP_WINDOW_ACTIVE",
                        "STOP_TIME_REACHED",
                        "HELD_FOR_PARSEABLE_TIMEBOX",
                    ],
                    previous_instruction=previous_instruction[:300],
                    latest_instruction=latest_instruction[:300],
                    latest_instruction_present=supersession_metadata.get("latest_instruction_present", False),
                    latest_instruction_supersedes_previous=supersession_metadata.get(
                        "latest_instruction_supersedes_previous", False
                    ),
                    latest_instruction_is_stop=supersession_metadata.get("latest_instruction_is_stop", False),
                    latest_instruction_is_continue=supersession_metadata.get("latest_instruction_is_continue", False),
                    supersession_state=supersession_metadata.get("supersession_state", ""),
                    can_continue_under_latest_instruction=supersession_metadata.get(
                        "can_continue_under_latest_instruction", False
                    ),
                    newest_instruction_overrides_automation=supersession_metadata.get(
                        "newest_instruction_overrides_automation", True
                    ),
                    newest_instruction_overrides_goal=supersession_metadata.get(
                        "newest_instruction_overrides_goal", True
                    ),
                    newest_instruction_overrides_recovery_queue=supersession_metadata.get(
                        "newest_instruction_overrides_recovery_queue", True
                    ),
                    stop_or_pause_blocks_autonomy=supersession_metadata.get("stop_or_pause_blocks_autonomy", False),
                    supersession_contract_rows=supersession_metadata.get("supersession_contract_rows", []),
                    supersession_contract_row_count=supersession_metadata.get("supersession_contract_row_count", 0),
                    supersession_contract_ready=supersession_metadata.get("supersession_contract_ready", False),
                    supersession_authorizes_execution=False,
                    supersession_authorizes_risky_work=False,
                    supersession_authorizes_approval=False,
                    supersession_authorizes_recovery_followthrough=False,
                    supersession_authorizes_timebox_override=False,
                    supersession_token_sha256=supersession_metadata.get("supersession_token_sha256", ""),
                    supersession_token_present=supersession_metadata.get("supersession_token_present", False),
                    supersession_token_boundary_rows=supersession_token_boundary_rows,
                    supersession_token_boundary_row_count=len(supersession_token_boundary_rows),
                    supersession_token_boundary_ready=supersession_token_boundary_ready,
                    supersession_token_authorizes_execution=False,
                    supersession_token_authorizes_local_safe_step=False,
                    supersession_token_authorizes_risky_work=False,
                    supersession_token_authorizes_approval=False,
                    supersession_token_authorizes_recovery_followthrough=False,
                    supersession_token_authorizes_timebox_override=False,
                    supersession_token_authorizes_goal_override=False,
                    supersession_token_authorizes_model_call=False,
                    supersession_token_authorizes_tool_execution=False,
                    supersession_token_authorizes_personal_data_read=False,
                    supersession_token_authorizes_external_side_effect=False,
                    supersession_token_reusable_for_next_review=False,
                    supersession_token_reusable_for_next_timebox=False,
                    next_supersession_requires_fresh_latest_instruction_review=True,
                    operator_supersession_precontinuation_contract_ready=supersession_metadata.get(
                        "supersession_contract_ready", False
                    )
                    and supersession_token_boundary_ready,
                    operator_supersession_precontinuation_contract_state=supersession_metadata.get(
                        "supersession_state", ""
                    ),
                    operator_supersession_precontinuation_token_boundary_ready=supersession_token_boundary_ready,
                    operator_supersession_precontinuation_allows_continuation=supersession_metadata.get(
                        "can_continue_under_latest_instruction", False
                    ),
                    operator_supersession_precontinuation_blocks_continuation=not bool(
                        supersession_metadata.get("can_continue_under_latest_instruction", False)
                    ),
                    operator_supersession_precontinuation_requires_stop=supersession_metadata.get(
                        "latest_instruction_is_stop", False
                    ),
                    operator_supersession_precontinuation_requires_fresh_latest_instruction_review=True,
                    operator_supersession_precontinuation_allowed_states=[
                        "LATEST_INSTRUCTION_READY_TO_GOVERN_CONTINUATION",
                        "NEWER_STOP_OR_PAUSE_OVERRIDES_AUTONOMY",
                        "HELD_FOR_LATEST_INSTRUCTION_REVIEW",
                    ],
                    checkpoint_recovery_execute_handoff_contains_raw_step=False,
                    checkpoint_recovery_execute_handoff_contains_raw_verification=False,
                    checkpoint_recovery_execute_handoff_changed_state=False,
                    checkpoint_recovery_execute_handoff_writes_notes=False,
                    checkpoint_recovery_execute_handoff_writes_files=False,
                    checkpoint_recovery_execute_handoff_calls_model=False,
                    checkpoint_recovery_execute_handoff_executes_tools=False,
                    checkpoint_recovery_execute_handoff_queues_approval=False,
                    checkpoint_recovery_execute_handoff_controls_computer=False,
                    checkpoint_recovery_execute_handoff_reads_private_data=False,
                    checkpoint_recovery_execute_handoff_reads_personal_data=False,
                    checkpoint_recovery_execute_handoff_executes_side_effect=False,
                    checkpoint_recovery_execute_handoff_external_side_effect=False,
                    checkpoint_recovery_execute_handoff_authorizes_execution=False,
                    checkpoint_recovery_execute_handoff_authorizes_recovery_followthrough=False,
                    checkpoint_recovery_execute_handoff_authorizes_local_safe_step=False,
                    checkpoint_recovery_execute_handoff_authorizes_risky_work=False,
                    checkpoint_recovery_execute_handoff_authorizes_approval=False,
                    checkpoint_recovery_execute_handoff_authorizes_model_call=False,
                    checkpoint_recovery_execute_handoff_authorizes_tool_execution=False,
                    checkpoint_recovery_execute_handoff_authorizes_personal_data_read=False,
                    checkpoint_recovery_execute_handoff_authorizes_external_side_effect=False,
                    checkpoint_recovery_execute_handoff_reusable_for_recovery_review=False,
                    approval_reference_provided=approval_reference_present,
                    approval_reference=approval_reference[:300] if approval_reference_present else "",
                    recovery_step_requires_approval=bool(risk_signals),
                    recovery_followthrough_gate=closure_gate,
                    recovery_followthrough_gate_state=closure_gate["state"],
                    normal_followthrough_allowed=False,
                    recovery_closure_missing=closure_gate["missing"],
                    recovery_closure_missing_count=closure_gate["missing_count"],
                    recovery_closure_required_evidence=list(closure_gate["required_evidence"]),
                    recovery_closure_required_evidence_count=len(closure_gate["required_evidence"]),
                    recovery_closure_proof_queue=list(recovery_closure_proof_queue),
                    recovery_closure_proof_queue_count=len(recovery_closure_proof_queue),
                    recovery_closure_next_proof_command=recovery_closure_next_proof_command,
                    recovery_closure_proof_queue_ready=recovery_closure_proof_queue_ready,
                    recovery_closure_approval_boundary=closure_gate["approval_boundary"],
                    recovery_closure_next_safe_command=closure_gate["next_safe_command"],
                    checkpoint_found=bool(packet.metadata.get("checkpoint_found")),
                    checkpoint_read=bool(packet.metadata.get("checkpoint_read")),
                ),
            )

        receipt = checkpoint_recovery_receipt(
            {
                "objective": objective,
                "approved_step": approved_step,
                "verification": verification,
                "files": files,
                "outcome": outcome,
                "approval_reference": approval_reference,
                "blockers": blockers,
            }
        )
        receipt_path = str(receipt.metadata.get("path") or "")
        receipt_path_display = _safe_vault_path_display(receipt_path, vault)
        receipt_sha256 = str(receipt.metadata.get("receipt_sha256") or "")
        receipt_file_sha256 = _file_sha256(receipt_path)
        receipt_hash_matches_file = bool(receipt_sha256 and receipt_file_sha256 and receipt_sha256 == receipt_file_sha256)
        checkpoint = save_work_block_checkpoint(
            {
                "objective": f"{objective} | recovery applied: {approved_step[:180]} | verification: {verification[:180]}",
                "limit": args.get("limit") or 12,
            }
        )
        checkpoint_path = str(checkpoint.metadata.get("path") or "")
        checkpoint_path_display = _safe_vault_path_display(checkpoint_path, vault)
        checkpoint_sha256 = _file_sha256(checkpoint_path)
        checkpoint_file_sha256 = _file_sha256(checkpoint_path)
        checkpoint_hash_matches_file = bool(checkpoint_sha256 and checkpoint_file_sha256 and checkpoint_sha256 == checkpoint_file_sha256)
        stop_condition = "stop if verification evidence does not match the reviewed step, blockers appear, or a future step needs approval"
        recovery_execution_contract = {
            "objective": objective[:240],
            "reviewed": reviewed,
            "approved_step": approved_step[:500],
            "verification_target": verification[:240],
            "files": files[:500],
            "outcome": outcome[:300],
            "blockers": blockers[:500],
            "risky_recovery_signals": risk_signals,
            "recovery_step_requires_approval": bool(risk_signals),
            "approval_reference_supplied": approval_reference_present,
            "approval_reference": approval_reference[:300] if approval_reference_present else "",
            "receipt_path": receipt_path,
            "receipt_sha256": receipt_sha256,
            "receipt_file_sha256": receipt_file_sha256,
            "receipt_hash_matches_file": receipt_hash_matches_file,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_file_sha256": checkpoint_file_sha256,
            "checkpoint_hash_matches_file": checkpoint_hash_matches_file,
            "stop_condition": stop_condition,
        }
        recovery_followthrough_token_sha256 = _recovery_followthrough_token_sha256(
            objective=objective,
            reviewed_step=approved_step,
            verification=verification,
            receipt_path=receipt_path,
            receipt_sha256=receipt_sha256,
            receipt_file_sha256=receipt_file_sha256,
            receipt_hash_matches_file=receipt_hash_matches_file,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_file_sha256=checkpoint_file_sha256,
            checkpoint_hash_matches_file=checkpoint_hash_matches_file,
            stop_condition=stop_condition,
        )
        recovery_followthrough_token_boundary_rows = _recovery_followthrough_token_boundary_rows(
            token_sha256=recovery_followthrough_token_sha256,
            source="checkpoint_recovery_execute",
        )
        recovery_followthrough_token_boundary_ready = _recovery_followthrough_token_boundary_ready(
            recovery_followthrough_token_sha256,
            recovery_followthrough_token_boundary_rows,
            expected_source="checkpoint_recovery_execute",
        )
        local_safe_recovery_execution_token_sha256 = _local_safe_recovery_execution_token_sha256(
            objective=objective,
            reviewed_step=approved_step,
            verification=verification,
            receipt_sha256=receipt_sha256,
            checkpoint_sha256=checkpoint_sha256,
            recovery_followthrough_token_sha256=recovery_followthrough_token_sha256,
            stop_condition=stop_condition,
        )
        local_safe_recovery_execution_token_boundary_rows = _local_safe_recovery_execution_token_boundary_rows(
            token_sha256=local_safe_recovery_execution_token_sha256,
            source="checkpoint_recovery_execute",
        )
        local_safe_recovery_execution_token_boundary_ready = _local_safe_recovery_execution_token_boundary_ready(
            local_safe_recovery_execution_token_sha256,
            local_safe_recovery_execution_token_boundary_rows,
            expected_source="checkpoint_recovery_execute",
        )
        recovery_execution_contract["recovery_followthrough_token_sha256"] = recovery_followthrough_token_sha256
        recovery_execution_contract["local_safe_recovery_execution_token_sha256"] = local_safe_recovery_execution_token_sha256
        closure_gate = _recovery_closure_gate(
            reviewed=reviewed,
            approved_step=approved_step,
            verification=verification,
            receipt_path=receipt_path,
            receipt_sha256=receipt_sha256,
            receipt_file_sha256=receipt_file_sha256,
            receipt_hash_matches_file=receipt_hash_matches_file,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_file_sha256=checkpoint_file_sha256,
            checkpoint_hash_matches_file=checkpoint_hash_matches_file,
            stop_condition=stop_condition,
            risk_signals=risk_signals,
            approval_reference_present=approval_reference_present,
            blockers=blockers,
        )
        followthrough = checkpoint_recovery_followthrough_packet(
            {
                "objective": objective,
                "reviewed_step": approved_step,
                "verification": verification,
                "receipt_path": receipt_path,
                "receipt_sha256": receipt_sha256,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": checkpoint_sha256,
                "stop_condition": stop_condition,
                "approval_reference": approval_reference,
                "blockers": blockers,
            }
        )
        recovery_execution_scorecard_rows = _recovery_execution_readiness_scorecard_rows(
            reviewed=reviewed,
            approved_step=approved_step,
            verification=verification,
            receipt_hash_matches_file=receipt_hash_matches_file,
            checkpoint_hash_matches_file=checkpoint_hash_matches_file,
            stop_condition=stop_condition,
            risk_signals=risk_signals,
            approval_reference_present=approval_reference_present,
            followthrough_ready=followthrough.metadata.get("normal_followthrough_allowed") is True,
            recovery_followthrough_token_sha256=recovery_followthrough_token_sha256,
        )
        recovery_execution_score = sum(int(row["points"]) for row in recovery_execution_scorecard_rows)
        recovery_execution_max_score = sum(int(row["max_points"]) for row in recovery_execution_scorecard_rows)
        recovery_execution_required_rows_ready = _recovery_execution_readiness_scorecard_ready(
            recovery_execution_scorecard_rows
        )
        recovery_execution_scorecard_ready = recovery_execution_required_rows_ready
        recovery_execution_contract_fields = sorted(recovery_execution_contract.keys())
        recovery_execution_readiness_token_sha256 = _recovery_execution_readiness_token_sha256(
            objective=objective,
            reviewed_step=approved_step,
            verification=verification,
            receipt_sha256=receipt_sha256,
            receipt_file_sha256=receipt_file_sha256,
            receipt_hash_matches_file=receipt_hash_matches_file,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_file_sha256=checkpoint_file_sha256,
            checkpoint_hash_matches_file=checkpoint_hash_matches_file,
            recovery_step_approval_boundary_token_sha256=recovery_step_approval_boundary_token_sha256,
            recovery_followthrough_token_sha256=recovery_followthrough_token_sha256,
            local_safe_recovery_execution_token_sha256=local_safe_recovery_execution_token_sha256,
            recovery_execution_scorecard_rows=recovery_execution_scorecard_rows,
            recovery_execution_contract_fields=recovery_execution_contract_fields,
            stop_condition=stop_condition,
            risk_signals=risk_signals,
            approval_reference_present=approval_reference_present,
        )
        recovery_execution_readiness_token_boundary_rows = _recovery_execution_readiness_token_boundary_rows(
            token_sha256=recovery_execution_readiness_token_sha256,
            source="checkpoint_recovery_execute",
        )
        recovery_execution_readiness_token_boundary_ready = _recovery_execution_readiness_token_boundary_ready(
            recovery_execution_readiness_token_sha256,
            recovery_execution_readiness_token_boundary_rows,
            expected_source="checkpoint_recovery_execute",
        )
        lines = [
            "Jarvis checkpoint recovery executor:",
            "Reviewed local-safe recovery step recorded.",
            "",
            "Applied recovery step:",
            f"- {approved_step[:500]}",
            "",
            "Verification evidence:",
            f"- {verification[:700]}",
            "",
            "Recovery execution contract:",
            f"- verification target: {verification[:240]}",
            f"- stop condition: {stop_condition}",
            f"- risky recovery signals: {', '.join(risk_signals) if risk_signals else 'none'}",
            f"- approval reference supplied: {'yes' if approval_reference_present else 'no'}",
            f"- risky recovery approval boundary token sha256: {recovery_step_approval_boundary_token_sha256 or 'missing'}",
            f"- risky recovery approval boundary as prior proof: {'yes' if recovery_step_approval_boundary_as_prior_proof else 'no'}",
            f"- risky recovery approval boundary reusable for recovery review: no",
            f"- receipt path: {receipt_path_display or 'missing'}",
            f"- receipt sha256: {receipt_sha256 or 'missing'}",
            f"- receipt file sha256: {receipt_file_sha256 or 'missing'}",
            f"- receipt hash matches file: {'yes' if receipt_hash_matches_file else 'no'}",
            f"- checkpoint path: {checkpoint_path_display or 'missing'}",
            f"- checkpoint sha256: {checkpoint_sha256 or 'missing'}",
            f"- checkpoint file sha256: {checkpoint_file_sha256 or 'missing'}",
            f"- checkpoint hash matches file: {'yes' if checkpoint_hash_matches_file else 'no'}",
            f"- recovery follow-through token sha256: {recovery_followthrough_token_sha256 or 'missing'}",
            "- recovery follow-through token reusable for future recovery: no",
            f"- recovery follow-through token boundary rows: {len(recovery_followthrough_token_boundary_rows)}",
            f"- recovery follow-through token boundary ready: {'yes' if recovery_followthrough_token_boundary_ready else 'no'}",
            *[
                f"- {row['item']}: {row['status']}; authorizes resume gate no; authorizes next step no; authorizes risky work no; authorizes model/tool/private/external no; reusable no"
                for row in recovery_followthrough_token_boundary_rows
            ],
            f"- local-safe recovery execution token sha256: {local_safe_recovery_execution_token_sha256 or 'missing'}",
            "- local-safe recovery execution token authorizes resume gate: no",
            "- local-safe recovery execution token authorizes next step: no",
            "- local-safe recovery execution token authorizes risky work: no",
            "- local-safe recovery execution token authorizes approval: no",
            "- local-safe recovery execution token authorizes model call: no",
            "- local-safe recovery execution token authorizes tool execution: no",
            "- local-safe recovery execution token authorizes personal-data read: no",
            "- local-safe recovery execution token authorizes external side effect: no",
            "- local-safe recovery execution token reusable for future recovery: no",
            f"- local-safe recovery execution token boundary rows: {len(local_safe_recovery_execution_token_boundary_rows)}",
            f"- local-safe recovery execution token boundary ready: {'yes' if local_safe_recovery_execution_token_boundary_ready else 'no'}",
            f"- recovery execution readiness token sha256: {recovery_execution_readiness_token_sha256 or 'missing'}",
            "- recovery execution readiness token authorizes resume gate: no",
            "- recovery execution readiness token authorizes next step: no",
            "- recovery execution readiness token authorizes risky work: no",
            "- recovery execution readiness token authorizes approval: no",
            "- recovery execution readiness token authorizes model call: no",
            "- recovery execution readiness token authorizes tool execution: no",
            "- recovery execution readiness token authorizes personal-data read: no",
            "- recovery execution readiness token authorizes external side effect: no",
            "- recovery execution readiness token authorizes unreviewed follow-through: no",
            "- recovery execution readiness token reusable for future recovery: no",
            f"- recovery execution readiness token boundary rows: {len(recovery_execution_readiness_token_boundary_rows)}",
            f"- recovery execution readiness token boundary ready: {'yes' if recovery_execution_readiness_token_boundary_ready else 'no'}",
            "",
            "Operator timebox contract before continuation:",
            f"- state: {timebox_metadata.get('timebox_state')}",
            f"- can continue now: {'yes' if timebox_metadata.get('can_continue_now') else 'no'}",
            f"- should stop now: {'yes' if timebox_metadata.get('should_stop_now') else 'no'}",
            f"- missing proof: {', '.join(timebox_metadata.get('missing_timebox_proof') or []) if timebox_metadata.get('missing_timebox_proof') else 'none'}",
            "- timebox authorizes execution: no",
            "- timebox authorizes local-safe step: no",
            "- next continuation still requires a fresh timebox: yes",
            "",
            "Operator instruction supersession before continuation:",
            f"- state: {supersession_metadata.get('supersession_state')}",
            f"- latest instruction can govern continuation: {'yes' if supersession_metadata.get('can_continue_under_latest_instruction') else 'no'}",
            f"- latest instruction is stop/pause: {'yes' if supersession_metadata.get('latest_instruction_is_stop') else 'no'}",
            f"- supersession token sha256: {supersession_metadata.get('supersession_token_sha256') or 'missing'}",
            "- supersession authorizes execution: no",
            "- supersession authorizes recovery follow-through: no",
            "- next continuation still requires a fresh latest-instruction review: yes",
            "",
            "Measured recovery execution readiness scorecard:",
            f"- score: {recovery_execution_score}/{recovery_execution_max_score}",
            f"- required rows ready: {'yes' if recovery_execution_required_rows_ready else 'no'}",
            f"- recovery execution scorecard ready: {'yes' if recovery_execution_scorecard_ready else 'no'}",
            f"- readiness token sha256: {recovery_execution_readiness_token_sha256 or 'missing'}",
            "- readiness token authorizes resume gate: no",
            "- readiness token authorizes next step: no",
            "- readiness token authorizes risky work: no",
            "- readiness token authorizes approval/model/tool/private/external/unreviewed follow-through: no",
            "- readiness token reusable for future recovery: no",
            *[
                f"- {row['item']}: {row['points']}/{row['max_points']} ({'ready' if row['ready'] else 'held'}; does not authorize risky work or unreviewed follow-through)"
                for row in recovery_execution_scorecard_rows
            ],
            *[
                f"- readiness token boundary {row['item']}: {row['status']}; authorizes resume gate no; authorizes next step no; authorizes risky work no; authorizes model/tool/private/external no; reusable no"
                for row in recovery_execution_readiness_token_boundary_rows
            ],
            "",
            "Recovery closure gate:",
            f"- state: {closure_gate['state']}",
            f"- normal follow-through allowed: {'yes' if closure_gate['normal_followthrough_allowed'] else 'no'}",
            f"- missing closure proof: {', '.join(closure_gate['missing']) if closure_gate['missing'] else 'none'}",
            f"- next safe command: `{closure_gate['next_safe_command']}`",
            "",
            "Recovery follow-through packet:",
            f"- state: {followthrough.metadata.get('followthrough_state')}",
            f"- next safe command: `{followthrough.metadata.get('next_safe_command')}`",
            "",
            "Continuity writes:",
            f"- receipt: {receipt_path_display or 'unknown'}",
            f"- fresh checkpoint: {checkpoint_path_display or 'unknown'}",
            f"- artifact hashes present: {'yes' if receipt_hash_matches_file and checkpoint_hash_matches_file else 'no'}",
            f"- artifact hashes match files: {'yes' if receipt_hash_matches_file and checkpoint_hash_matches_file else 'no'}",
            "",
            "Outcome:",
            f"- {outcome[:300]}",
            "",
            "Boundary:",
            "- local-safe continuity executor only; it records a reviewed recovery step and fresh checkpoint, but does not run shell/code, apply patches, approve requests, read private data, control the computer, call external services, or queue approvals.",
        ]
        return ToolResult(
            "checkpoint_recovery_execute",
            True,
            "\n".join(lines),
            _safe_metadata(
                objective_length=len(objective),
                objective=objective[:240],
                reviewed=reviewed,
                checkpoint_found=bool(packet.metadata.get("checkpoint_found")),
                checkpoint_read=bool(packet.metadata.get("checkpoint_read")),
                receipt_path=receipt_path,
                receipt_path_display=receipt_path_display,
                receipt_sha256=receipt_sha256,
                receipt_file_sha256=receipt_file_sha256,
                receipt_file_hash_present=_looks_like_sha256(receipt_file_sha256),
                receipt_hash_matches_file=receipt_hash_matches_file,
                checkpoint_path=checkpoint_path,
                checkpoint_path_display=checkpoint_path_display,
                checkpoint_sha256=checkpoint_sha256,
                checkpoint_file_sha256=checkpoint_file_sha256,
                checkpoint_file_hash_present=_looks_like_sha256(checkpoint_file_sha256),
                checkpoint_hash_matches_file=checkpoint_hash_matches_file,
                receipt_hash_provided=_looks_like_sha256(receipt_sha256),
                checkpoint_hash_provided=_looks_like_sha256(checkpoint_sha256),
                recovery_artifact_hashes_present=bool(receipt_hash_matches_file and checkpoint_hash_matches_file),
                recovery_artifact_hashes_match_files=bool(receipt_hash_matches_file and checkpoint_hash_matches_file),
                recovery_step_sha256=recovery_step_sha256,
                recovery_verification_sha256=recovery_verification_sha256,
                approval_reference_provided=approval_reference_present,
                approval_reference=approval_reference[:300] if approval_reference_present else "",
                recovery_step_requires_approval=bool(risk_signals),
                recovery_step_approval_proof_queue=recovery_step_approval_proof_queue,
                recovery_step_approval_proof_queue_count=len(recovery_step_approval_proof_queue),
                recovery_step_next_approval_proof_command=(
                    recovery_step_approval_proof_queue[0] if recovery_step_approval_proof_queue else ""
                ),
                recovery_step_approval_required_before_recovery=bool(risk_signals and not approval_reference_present),
                recovery_step_approval_boundary_rows=recovery_step_approval_boundary_rows,
                checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_rows=recovery_step_approval_boundary_rows,
                recovery_step_approval_boundary_row_count=len(recovery_step_approval_boundary_rows),
                recovery_step_approval_boundary_ready=recovery_step_approval_boundary_ready,
                recovery_step_approval_boundary_token_sha256=recovery_step_approval_boundary_token_sha256,
                recovery_step_approval_boundary_token_present=_looks_like_sha256(recovery_step_approval_boundary_token_sha256),
                recovery_step_approval_boundary_as_prior_proof=recovery_step_approval_boundary_as_prior_proof,
                **{flag: False for flag in _RECOVERY_STEP_APPROVAL_BOUNDARY_FALSE_FLAGS},
                recovery_followthrough_token_sha256=recovery_followthrough_token_sha256,
                recovery_followthrough_token_present=_looks_like_sha256(recovery_followthrough_token_sha256),
                recovery_followthrough_token_boundary_rows=recovery_followthrough_token_boundary_rows,
                recovery_followthrough_token_boundary_row_count=len(recovery_followthrough_token_boundary_rows),
                recovery_followthrough_token_boundary_ready=recovery_followthrough_token_boundary_ready,
                recovery_followthrough_token_reusable_for_future_recovery=False,
                recovery_followthrough_token_authorizes_resume_gate=False,
                recovery_followthrough_token_authorizes_next_step=False,
                recovery_followthrough_token_authorizes_risky_work=False,
                recovery_followthrough_token_authorizes_approval=False,
                recovery_followthrough_token_authorizes_model_call=False,
                recovery_followthrough_token_authorizes_tool_execution=False,
                recovery_followthrough_token_authorizes_personal_data_read=False,
                recovery_followthrough_token_authorizes_external_side_effect=False,
                next_recovery_followthrough_requires_new_token=True,
                local_safe_recovery_execution_token_sha256=local_safe_recovery_execution_token_sha256,
                local_safe_recovery_execution_token_present=_looks_like_sha256(local_safe_recovery_execution_token_sha256),
                local_safe_recovery_execution_token_authorizes_resume_gate=False,
                local_safe_recovery_execution_token_authorizes_next_step=False,
                local_safe_recovery_execution_token_authorizes_risky_work=False,
                local_safe_recovery_execution_token_authorizes_approval=False,
                local_safe_recovery_execution_token_authorizes_model_call=False,
                local_safe_recovery_execution_token_authorizes_tool_execution=False,
                local_safe_recovery_execution_token_authorizes_personal_data_read=False,
                local_safe_recovery_execution_token_authorizes_external_side_effect=False,
                local_safe_recovery_execution_token_reusable_for_future_recovery=False,
                next_recovery_execution_requires_new_local_safe_token=True,
                local_safe_recovery_execution_token_boundary_rows=local_safe_recovery_execution_token_boundary_rows,
                local_safe_recovery_execution_token_boundary_row_count=len(local_safe_recovery_execution_token_boundary_rows),
                local_safe_recovery_execution_token_boundary_ready=local_safe_recovery_execution_token_boundary_ready,
                recovery_execution_contract=recovery_execution_contract,
                recovery_execution_contract_fields=recovery_execution_contract_fields,
                recovery_execution_contract_field_count=len(recovery_execution_contract),
                recovery_execution_score=recovery_execution_score,
                recovery_execution_max_score=recovery_execution_max_score,
                recovery_execution_scorecard_rows=recovery_execution_scorecard_rows,
                recovery_execution_scorecard_row_count=len(recovery_execution_scorecard_rows),
                recovery_execution_required_rows_ready=recovery_execution_required_rows_ready,
                recovery_execution_scorecard_ready=recovery_execution_scorecard_ready,
                recovery_execution_readiness_token_sha256=recovery_execution_readiness_token_sha256,
                recovery_execution_readiness_token_present=_looks_like_sha256(recovery_execution_readiness_token_sha256),
                recovery_execution_readiness_token_boundary_rows=recovery_execution_readiness_token_boundary_rows,
                recovery_execution_readiness_token_boundary_row_count=len(recovery_execution_readiness_token_boundary_rows),
                recovery_execution_readiness_token_boundary_ready=recovery_execution_readiness_token_boundary_ready,
                recovery_execution_readiness_token_authorizes_resume_gate=False,
                recovery_execution_readiness_token_authorizes_next_step=False,
                recovery_execution_readiness_token_authorizes_risky_work=False,
                recovery_execution_readiness_token_authorizes_approval=False,
                recovery_execution_readiness_token_authorizes_model_call=False,
                recovery_execution_readiness_token_authorizes_tool_execution=False,
                recovery_execution_readiness_token_authorizes_personal_data_read=False,
                recovery_execution_readiness_token_authorizes_external_side_effect=False,
                recovery_execution_readiness_token_authorizes_unreviewed_followthrough=False,
                recovery_execution_readiness_token_reusable_for_future_recovery=False,
                next_recovery_execution_requires_new_readiness_token=True,
                recovery_execution_readiness_authorizes_model_call=False,
                recovery_execution_readiness_authorizes_tool_execution=False,
                recovery_execution_readiness_authorizes_personal_data_read=False,
                recovery_execution_readiness_authorizes_external_side_effect=False,
                stop_at=stop_at,
                current_time=timebox_metadata.get("current_time", ""),
                timezone=timezone_label,
                timebox_state=timebox_metadata.get("timebox_state", ""),
                minutes_remaining=timebox_metadata.get("minutes_remaining"),
                can_continue_now=timebox_metadata.get("can_continue_now", False),
                should_stop_now=timebox_metadata.get("should_stop_now", True),
                missing_timebox_proof=timebox_metadata.get("missing_timebox_proof", []),
                missing_timebox_proof_count=timebox_metadata.get("missing_timebox_proof_count", 0),
                timebox_receipt_sha256=timebox_metadata.get("timebox_receipt_sha256", ""),
                timebox_receipt_present=timebox_metadata.get("timebox_receipt_present", False),
                timebox_review_contract_rows=timebox_review_contract_rows,
                timebox_review_contract_row_count=len(timebox_review_contract_rows),
                timebox_review_contract_ready=timebox_metadata.get("timebox_review_contract_ready", False),
                timebox_review_contract_summary=timebox_review_contract_summary,
                timebox_authorizes_execution=False,
                timebox_authorizes_local_safe_step=False,
                timebox_authorizes_risky_work=False,
                timebox_authorizes_approval=False,
                timebox_authorizes_timebox_reuse=False,
                timebox_authorizes_model_call=False,
                timebox_authorizes_tool_execution=False,
                timebox_authorizes_personal_data_read=False,
                timebox_authorizes_external_side_effect=False,
                timebox_reusable_for_next_step=False,
                next_step_requires_fresh_timebox=True,
                operator_timebox_precontinuation_contract_ready=_metadata_bool(timebox_metadata.get("timebox_review_contract_ready", False)),
                operator_timebox_precontinuation_contract_state=timebox_metadata.get("timebox_state", ""),
                operator_timebox_precontinuation_allows_continuation=_metadata_bool(timebox_metadata.get("can_continue_now", False)),
                operator_timebox_precontinuation_blocks_continuation=not _metadata_bool(
                    timebox_metadata.get("can_continue_now", False)
                ),
                operator_timebox_precontinuation_requires_stop=_metadata_bool(timebox_metadata.get("should_stop_now", True), default=True),
                operator_timebox_precontinuation_requires_fresh_timebox_review=True,
                operator_timebox_precontinuation_allowed_states=[
                    "STOP_WINDOW_ACTIVE",
                    "STOP_TIME_REACHED",
                    "HELD_FOR_PARSEABLE_TIMEBOX",
                ],
                previous_instruction=previous_instruction[:300],
                latest_instruction=latest_instruction[:300],
                latest_instruction_present=supersession_metadata.get("latest_instruction_present", False),
                latest_instruction_supersedes_previous=supersession_metadata.get(
                    "latest_instruction_supersedes_previous", False
                ),
                latest_instruction_is_stop=supersession_metadata.get("latest_instruction_is_stop", False),
                latest_instruction_is_continue=supersession_metadata.get("latest_instruction_is_continue", False),
                supersession_state=supersession_metadata.get("supersession_state", ""),
                can_continue_under_latest_instruction=supersession_metadata.get(
                    "can_continue_under_latest_instruction", False
                ),
                newest_instruction_overrides_automation=supersession_metadata.get(
                    "newest_instruction_overrides_automation", True
                ),
                newest_instruction_overrides_goal=supersession_metadata.get("newest_instruction_overrides_goal", True),
                newest_instruction_overrides_recovery_queue=supersession_metadata.get(
                    "newest_instruction_overrides_recovery_queue", True
                ),
                stop_or_pause_blocks_autonomy=supersession_metadata.get("stop_or_pause_blocks_autonomy", False),
                supersession_contract_rows=supersession_metadata.get("supersession_contract_rows", []),
                supersession_contract_row_count=supersession_metadata.get("supersession_contract_row_count", 0),
                supersession_contract_ready=supersession_metadata.get("supersession_contract_ready", False),
                supersession_authorizes_execution=False,
                supersession_authorizes_risky_work=False,
                supersession_authorizes_approval=False,
                supersession_authorizes_recovery_followthrough=False,
                supersession_authorizes_timebox_override=False,
                supersession_token_sha256=supersession_metadata.get("supersession_token_sha256", ""),
                supersession_token_present=supersession_metadata.get("supersession_token_present", False),
                supersession_token_boundary_rows=supersession_token_boundary_rows,
                supersession_token_boundary_row_count=len(supersession_token_boundary_rows),
                supersession_token_boundary_ready=supersession_token_boundary_ready,
                supersession_token_authorizes_execution=False,
                supersession_token_authorizes_local_safe_step=False,
                supersession_token_authorizes_risky_work=False,
                supersession_token_authorizes_approval=False,
                supersession_token_authorizes_recovery_followthrough=False,
                supersession_token_authorizes_timebox_override=False,
                supersession_token_authorizes_goal_override=False,
                supersession_token_authorizes_model_call=False,
                supersession_token_authorizes_tool_execution=False,
                supersession_token_authorizes_personal_data_read=False,
                supersession_token_authorizes_external_side_effect=False,
                supersession_token_reusable_for_next_review=False,
                supersession_token_reusable_for_next_timebox=False,
                next_supersession_requires_fresh_latest_instruction_review=True,
                operator_supersession_precontinuation_contract_ready=supersession_metadata.get(
                    "supersession_contract_ready", False
                )
                and supersession_token_boundary_ready,
                operator_supersession_precontinuation_contract_state=supersession_metadata.get("supersession_state", ""),
                operator_supersession_precontinuation_token_boundary_ready=supersession_token_boundary_ready,
                operator_supersession_precontinuation_allows_continuation=supersession_metadata.get(
                    "can_continue_under_latest_instruction", False
                ),
                operator_supersession_precontinuation_blocks_continuation=not bool(
                    supersession_metadata.get("can_continue_under_latest_instruction", False)
                ),
                operator_supersession_precontinuation_requires_stop=supersession_metadata.get(
                    "latest_instruction_is_stop", False
                ),
                operator_supersession_precontinuation_requires_fresh_latest_instruction_review=True,
                operator_supersession_precontinuation_allowed_states=[
                    "LATEST_INSTRUCTION_READY_TO_GOVERN_CONTINUATION",
                    "NEWER_STOP_OR_PAUSE_OVERRIDES_AUTONOMY",
                    "HELD_FOR_LATEST_INSTRUCTION_REVIEW",
                ],
                recovery_followthrough_gate=closure_gate,
                recovery_followthrough_gate_state=closure_gate["state"],
                recovery_followthrough_packet_state=followthrough.metadata.get("followthrough_state"),
                recovery_followthrough_packet_ready=followthrough.metadata.get("normal_followthrough_allowed"),
                recovery_followthrough_packet_next_safe_command=followthrough.metadata.get("next_safe_command"),
                recovery_followthrough_packet_receipt_sha256=followthrough.metadata.get("receipt_sha256"),
                recovery_followthrough_packet_receipt_file_sha256=followthrough.metadata.get("receipt_file_sha256"),
                recovery_followthrough_packet_receipt_hash_matches_file=followthrough.metadata.get("receipt_hash_matches_file"),
                recovery_followthrough_packet_checkpoint_sha256=followthrough.metadata.get("checkpoint_sha256"),
                recovery_followthrough_packet_checkpoint_file_sha256=followthrough.metadata.get("checkpoint_file_sha256"),
                recovery_followthrough_packet_checkpoint_hash_matches_file=followthrough.metadata.get("checkpoint_hash_matches_file"),
                recovery_followthrough_packet_token_sha256=followthrough.metadata.get("recovery_followthrough_token_sha256"),
                recovery_followthrough_packet_local_safe_execution_token_sha256=followthrough.metadata.get("local_safe_recovery_execution_token_sha256"),
                recovery_followthrough_packet_artifact_hashes_present=followthrough.metadata.get("recovery_artifact_hashes_present"),
                recovery_followthrough_packet_artifact_hashes_match_files=followthrough.metadata.get("recovery_artifact_hashes_match_files"),
                recovery_followthrough_packet_output=followthrough.output[:1600],
                normal_followthrough_allowed=closure_gate["normal_followthrough_allowed"],
                recovery_closure_required_evidence=closure_gate["required_evidence"],
                recovery_closure_required_evidence_count=len(closure_gate["required_evidence"]),
                recovery_closure_missing=closure_gate["missing"],
                recovery_closure_missing_count=closure_gate["missing_count"],
                recovery_closure_next_safe_command=closure_gate["next_safe_command"],
                recovery_executor_recorded=True,
                reviewed_step=approved_step[:500],
                outcome=outcome[:300],
                recovery_files=files[:500],
                blockers=blockers[:500],
                verification_provided=True,
                files_provided=files != "not specified",
                blockers_provided=blockers != "none reported",
                risky_recovery_signals=risk_signals,
                risky_recovery_signal_count=len(risk_signals),
                risky_recovery_without_approval=False,
                verification_target=verification[:240],
                stop_condition=stop_condition,
                writes_files=True,
                writes_notes=True,
            ),
        )

    def checkpoint_recovery_followthrough_packet(args: dict[str, Any]) -> ToolResult:
        objective = str(args.get("objective") or args.get("request") or "Resume normal Jarvis follow-through after checkpoint recovery.").strip()
        recovery_execution_objective = objective[:240]
        reviewed_step = str(args.get("reviewed_step") or args.get("step") or args.get("approved_step") or "").strip()
        verification = str(args.get("verification") or args.get("tests") or args.get("verification_target") or "").strip()
        receipt_path = str(args.get("receipt_path") or args.get("receipt") or "").strip()
        receipt_path_display = _safe_vault_path_display(receipt_path, vault)
        receipt_sha256 = str(args.get("receipt_sha256") or args.get("receipt_hash") or "").strip()
        checkpoint_path = str(args.get("checkpoint_path") or args.get("checkpoint") or "").strip()
        checkpoint_path_display = _safe_vault_path_display(checkpoint_path, vault)
        checkpoint_sha256 = str(args.get("checkpoint_sha256") or args.get("checkpoint_hash") or "").strip()
        receipt_file_sha256 = _file_sha256(receipt_path)
        receipt_hash_matches_file = bool(receipt_sha256 and receipt_file_sha256 and receipt_sha256 == receipt_file_sha256)
        checkpoint_file_sha256 = _file_sha256(checkpoint_path)
        checkpoint_hash_matches_file = bool(checkpoint_sha256 and checkpoint_file_sha256 and checkpoint_sha256 == checkpoint_file_sha256)
        stop_condition = str(args.get("stop_condition") or args.get("stop") or "").strip()
        approval_boundary = str(args.get("approval_boundary") or "").strip() or (
            "future shell/code, computer control, personal-data, external-side-effect, destructive, or otherwise risky recovery steps still require approval readiness, a last-look approval packet, approval chain proof, and explicit approval"
        )
        blockers = str(args.get("blockers") or "none reported").strip()
        risk_signals = _recovery_risk_signals(" ".join([reviewed_step, verification]))
        approval_reference = str(args.get("approval") or args.get("approval_reference") or "").strip()
        approval_reference_present = (
            _approval_chain_reference_provided(approval_reference)
            if risk_signals
            else _approval_reference_provided(approval_reference)
        )
        closure_gate = _recovery_closure_gate(
            reviewed=bool(reviewed_step),
            approved_step=reviewed_step,
            verification=verification,
            receipt_path=receipt_path,
            receipt_sha256=receipt_sha256,
            receipt_file_sha256=receipt_file_sha256,
            receipt_hash_matches_file=receipt_hash_matches_file,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_file_sha256=checkpoint_file_sha256,
            checkpoint_hash_matches_file=checkpoint_hash_matches_file,
            stop_condition=stop_condition,
            risk_signals=risk_signals,
            approval_reference_present=approval_reference_present,
            blockers=blockers,
        )
        recovery_followthrough_token_sha256 = _recovery_followthrough_token_sha256(
            objective=objective,
            reviewed_step=reviewed_step,
            verification=verification,
            receipt_path=receipt_path,
            receipt_sha256=receipt_sha256,
            receipt_file_sha256=receipt_file_sha256,
            receipt_hash_matches_file=receipt_hash_matches_file,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_file_sha256=checkpoint_file_sha256,
            checkpoint_hash_matches_file=checkpoint_hash_matches_file,
            stop_condition=stop_condition,
        )
        recovery_followthrough_token_boundary_rows = _recovery_followthrough_token_boundary_rows(
            token_sha256=recovery_followthrough_token_sha256,
            source="checkpoint_recovery_followthrough",
        )
        recovery_followthrough_token_boundary_ready = _recovery_followthrough_token_boundary_ready(
            recovery_followthrough_token_sha256,
            recovery_followthrough_token_boundary_rows,
            expected_source="checkpoint_recovery_followthrough",
        )
        local_safe_recovery_execution_token_sha256 = _local_safe_recovery_execution_token_sha256(
            objective=objective,
            reviewed_step=reviewed_step,
            verification=verification,
            receipt_sha256=receipt_sha256,
            checkpoint_sha256=checkpoint_sha256,
            recovery_followthrough_token_sha256=recovery_followthrough_token_sha256,
            stop_condition=stop_condition,
        )
        local_safe_recovery_execution_token_boundary_rows = _local_safe_recovery_execution_token_boundary_rows(
            token_sha256=local_safe_recovery_execution_token_sha256,
            source="checkpoint_recovery_followthrough",
        )
        local_safe_recovery_execution_token_boundary_ready = _local_safe_recovery_execution_token_boundary_ready(
            local_safe_recovery_execution_token_sha256,
            local_safe_recovery_execution_token_boundary_rows,
            expected_source="checkpoint_recovery_followthrough",
        )
        preliminary_ready = bool(closure_gate["normal_followthrough_allowed"])
        recovery_execution_scorecard_rows = _recovery_execution_readiness_scorecard_rows(
            reviewed=bool(reviewed_step),
            approved_step=reviewed_step,
            verification=verification,
            receipt_hash_matches_file=receipt_hash_matches_file,
            checkpoint_hash_matches_file=checkpoint_hash_matches_file,
            stop_condition=stop_condition,
            risk_signals=risk_signals,
            approval_reference_present=approval_reference_present,
            followthrough_ready=preliminary_ready,
            recovery_followthrough_token_sha256=recovery_followthrough_token_sha256,
        )
        recovery_execution_score = sum(int(row["points"]) for row in recovery_execution_scorecard_rows)
        recovery_execution_max_score = sum(int(row["max_points"]) for row in recovery_execution_scorecard_rows)
        recovery_execution_required_rows_ready = _recovery_execution_readiness_scorecard_ready(
            recovery_execution_scorecard_rows
        )
        recovery_execution_scorecard_ready = recovery_execution_required_rows_ready
        recovery_execution_readiness_token_sha256 = _recovery_execution_readiness_token_sha256(
            objective=recovery_execution_objective,
            reviewed_step=reviewed_step,
            verification=verification,
            receipt_sha256=receipt_sha256,
            receipt_file_sha256=receipt_file_sha256,
            receipt_hash_matches_file=receipt_hash_matches_file,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_file_sha256=checkpoint_file_sha256,
            checkpoint_hash_matches_file=checkpoint_hash_matches_file,
            recovery_step_approval_boundary_token_sha256="",
            recovery_followthrough_token_sha256=recovery_followthrough_token_sha256,
            local_safe_recovery_execution_token_sha256=local_safe_recovery_execution_token_sha256,
            recovery_execution_scorecard_rows=recovery_execution_scorecard_rows,
            recovery_execution_contract_fields=[],
            stop_condition=stop_condition,
            risk_signals=risk_signals,
            approval_reference_present=approval_reference_present,
        )
        recovery_execution_readiness_token_boundary_rows = _recovery_execution_readiness_token_boundary_rows(
            token_sha256=recovery_execution_readiness_token_sha256,
            source="checkpoint_recovery_followthrough",
        )
        recovery_execution_readiness_token_boundary_ready = _recovery_execution_readiness_token_boundary_ready(
            recovery_execution_readiness_token_sha256,
            recovery_execution_readiness_token_boundary_rows,
            expected_source="checkpoint_recovery_followthrough",
        )
        readiness_candidate = _safe_metadata(
            objective=recovery_execution_objective,
            followthrough_state="RECOVERY_FOLLOWTHROUGH_READY" if preliminary_ready else "RECOVERY_FOLLOWTHROUGH_HELD",
            normal_followthrough_allowed=preliminary_ready,
            recovery_followthrough_gate_state=closure_gate["state"],
            recovery_closure_missing=closure_gate["missing"],
            recovery_closure_missing_count=closure_gate["missing_count"],
            recovery_closure_required_evidence=closure_gate["required_evidence"],
            recovery_closure_required_evidence_count=len(closure_gate["required_evidence"]),
            approval_boundary_provided=bool(approval_boundary),
            approval_boundary=approval_boundary,
            reviewed_step_provided=bool(reviewed_step),
            verification_provided=bool(verification),
            receipt_path_provided=bool(receipt_path),
            receipt_sha256=receipt_sha256,
            receipt_hash_provided=_looks_like_sha256(receipt_sha256),
            receipt_file_sha256=receipt_file_sha256,
            receipt_file_hash_present=_looks_like_sha256(receipt_file_sha256),
            receipt_hash_matches_file=receipt_hash_matches_file,
            checkpoint_path_provided=bool(checkpoint_path),
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_hash_provided=_looks_like_sha256(checkpoint_sha256),
            checkpoint_file_sha256=checkpoint_file_sha256,
            checkpoint_file_hash_present=_looks_like_sha256(checkpoint_file_sha256),
            checkpoint_hash_matches_file=checkpoint_hash_matches_file,
            recovery_artifact_hashes_present=bool(receipt_hash_matches_file and checkpoint_hash_matches_file),
            recovery_artifact_hashes_match_files=bool(receipt_hash_matches_file and checkpoint_hash_matches_file),
            recovery_followthrough_token_sha256=recovery_followthrough_token_sha256,
            recovery_followthrough_token_present=_looks_like_sha256(recovery_followthrough_token_sha256),
            recovery_followthrough_token_boundary_rows=recovery_followthrough_token_boundary_rows,
            recovery_followthrough_token_boundary_row_count=len(recovery_followthrough_token_boundary_rows),
            recovery_followthrough_token_boundary_ready=recovery_followthrough_token_boundary_ready,
            recovery_followthrough_token_reusable_for_future_recovery=False,
            recovery_followthrough_token_authorizes_resume_gate=False,
            recovery_followthrough_token_authorizes_next_step=False,
            recovery_followthrough_token_authorizes_risky_work=False,
            recovery_followthrough_token_authorizes_approval=False,
            recovery_followthrough_token_authorizes_model_call=False,
            recovery_followthrough_token_authorizes_tool_execution=False,
            recovery_followthrough_token_authorizes_personal_data_read=False,
            recovery_followthrough_token_authorizes_external_side_effect=False,
            next_recovery_followthrough_requires_new_token=True,
            local_safe_recovery_execution_token_sha256=local_safe_recovery_execution_token_sha256,
            local_safe_recovery_execution_token_present=_looks_like_sha256(local_safe_recovery_execution_token_sha256),
            local_safe_recovery_execution_token_authorizes_resume_gate=False,
            local_safe_recovery_execution_token_authorizes_next_step=False,
            local_safe_recovery_execution_token_authorizes_risky_work=False,
            local_safe_recovery_execution_token_authorizes_approval=False,
            local_safe_recovery_execution_token_authorizes_model_call=False,
            local_safe_recovery_execution_token_authorizes_tool_execution=False,
            local_safe_recovery_execution_token_authorizes_personal_data_read=False,
            local_safe_recovery_execution_token_authorizes_external_side_effect=False,
            local_safe_recovery_execution_token_reusable_for_future_recovery=False,
            next_recovery_execution_requires_new_local_safe_token=True,
            local_safe_recovery_execution_token_boundary_rows=local_safe_recovery_execution_token_boundary_rows,
            local_safe_recovery_execution_token_boundary_row_count=len(local_safe_recovery_execution_token_boundary_rows),
            local_safe_recovery_execution_token_boundary_ready=local_safe_recovery_execution_token_boundary_ready,
            recovery_execution_score=recovery_execution_score,
            recovery_execution_max_score=recovery_execution_max_score,
            recovery_execution_scorecard_rows=recovery_execution_scorecard_rows,
            recovery_execution_scorecard_row_count=len(recovery_execution_scorecard_rows),
            recovery_execution_required_rows_ready=recovery_execution_required_rows_ready,
            recovery_execution_scorecard_ready=recovery_execution_scorecard_ready,
            recovery_execution_contract_fields=[],
            recovery_execution_contract_field_count=0,
            recovery_execution_readiness_token_sha256=recovery_execution_readiness_token_sha256,
            recovery_execution_readiness_token_present=_looks_like_sha256(recovery_execution_readiness_token_sha256),
            recovery_execution_readiness_token_boundary_rows=recovery_execution_readiness_token_boundary_rows,
            recovery_execution_readiness_token_boundary_row_count=len(recovery_execution_readiness_token_boundary_rows),
            recovery_execution_readiness_token_boundary_ready=recovery_execution_readiness_token_boundary_ready,
            recovery_execution_readiness_token_authorizes_resume_gate=False,
            recovery_execution_readiness_token_authorizes_next_step=False,
            recovery_execution_readiness_token_authorizes_risky_work=False,
            recovery_execution_readiness_token_authorizes_approval=False,
            recovery_execution_readiness_token_authorizes_model_call=False,
            recovery_execution_readiness_token_authorizes_tool_execution=False,
            recovery_execution_readiness_token_authorizes_personal_data_read=False,
            recovery_execution_readiness_token_authorizes_external_side_effect=False,
            recovery_execution_readiness_token_authorizes_unreviewed_followthrough=False,
            recovery_execution_readiness_token_reusable_for_future_recovery=False,
            next_recovery_execution_requires_new_readiness_token=True,
            recovery_execution_readiness_as_prior_proof=bool(
                preliminary_ready
                and recovery_execution_required_rows_ready
                and recovery_execution_score == recovery_execution_max_score == 100
                and all(
                    row.get("authorizes_risky_work") is False
                    and row.get("authorizes_unreviewed_followthrough") is False
                    and row.get("authorizes_model_call") is False
                    and row.get("authorizes_tool_execution") is False
                    and row.get("authorizes_personal_data_read") is False
                    and row.get("authorizes_external_side_effect") is False
                    for row in recovery_execution_scorecard_rows
                )
            ),
            recovery_execution_readiness_authorizes_action_now=False,
            recovery_execution_readiness_authorizes_risky_work=False,
            recovery_execution_readiness_authorizes_unreviewed_followthrough=False,
        )
        ready = _checkpoint_recovery_followthrough_ready(readiness_candidate)
        state = "RECOVERY_FOLLOWTHROUGH_READY" if ready else "RECOVERY_FOLLOWTHROUGH_HELD"
        next_command = "continue scoped local-safe Jarvis work from the fresh checkpoint" if ready else closure_gate["next_safe_command"]
        lines = [
            "Jarvis checkpoint recovery follow-through packet:",
            "This is read-only. It proves whether a reviewed checkpoint recovery step has enough closure evidence before normal autonomous follow-through resumes.",
            "",
            "Objective:",
            f"- {objective[:300]}",
            "",
            "Follow-through state:",
            f"- state: {state}",
            f"- normal follow-through allowed: {'yes' if ready else 'no'}",
            f"- missing closure proof: {', '.join(closure_gate['missing']) if closure_gate['missing'] else 'none'}",
            f"- next safe command: `{next_command}`",
            "",
            "Closure evidence:",
            f"- reviewed step: {reviewed_step[:500] or 'missing'}",
            f"- verification: {verification[:500] or 'missing'}",
            f"- receipt path: {receipt_path_display or 'missing'}",
            f"- receipt sha256: {receipt_sha256 or 'not supplied'}",
            f"- receipt file sha256: {receipt_file_sha256 or 'missing'}",
            f"- receipt hash matches file: {'yes' if receipt_hash_matches_file else 'no'}",
            f"- fresh checkpoint: {checkpoint_path_display or 'missing'}",
            f"- checkpoint sha256: {checkpoint_sha256 or 'not supplied'}",
            f"- checkpoint file sha256: {checkpoint_file_sha256 or 'missing'}",
            f"- checkpoint hash matches file: {'yes' if checkpoint_hash_matches_file else 'no'}",
            f"- artifact hashes bound to files: {'yes' if receipt_hash_matches_file and checkpoint_hash_matches_file else 'no'}",
            f"- recovery follow-through token sha256: {recovery_followthrough_token_sha256 or 'missing'}",
            "- recovery follow-through token reusable for future recovery: no",
            "- new recovery follow-through token required before future recovery follow-through: yes",
            f"- recovery follow-through token boundary rows: {len(recovery_followthrough_token_boundary_rows)}",
            f"- recovery follow-through token boundary ready: {'yes' if recovery_followthrough_token_boundary_ready else 'no'}",
            *[
                f"- {row['item']}: {row['status']}; authorizes resume gate no; authorizes next step no; authorizes risky work no; authorizes model/tool/private/external no; reusable no"
                for row in recovery_followthrough_token_boundary_rows
            ],
            f"- local-safe recovery execution token sha256: {local_safe_recovery_execution_token_sha256 or 'missing'}",
            "- local-safe recovery execution token authorizes resume gate: no",
            "- local-safe recovery execution token authorizes next step: no",
            "- local-safe recovery execution token authorizes risky work: no",
            "- local-safe recovery execution token authorizes approval: no",
            "- local-safe recovery execution token authorizes model call: no",
            "- local-safe recovery execution token authorizes tool execution: no",
            "- local-safe recovery execution token authorizes personal-data read: no",
            "- local-safe recovery execution token authorizes external side effect: no",
            "- local-safe recovery execution token reusable for future recovery: no",
            f"- local-safe recovery execution token boundary rows: {len(local_safe_recovery_execution_token_boundary_rows)}",
            f"- local-safe recovery execution token boundary ready: {'yes' if local_safe_recovery_execution_token_boundary_ready else 'no'}",
            f"- recovery execution readiness token sha256: {recovery_execution_readiness_token_sha256 or 'missing'}",
            "- recovery execution readiness token authorizes resume gate: no",
            "- recovery execution readiness token authorizes next step: no",
            "- recovery execution readiness token authorizes risky work: no",
            "- recovery execution readiness token authorizes approval: no",
            "- recovery execution readiness token authorizes model call: no",
            "- recovery execution readiness token authorizes tool execution: no",
            "- recovery execution readiness token authorizes personal-data read: no",
            "- recovery execution readiness token authorizes external side effect: no",
            "- recovery execution readiness token authorizes unreviewed follow-through: no",
            "- recovery execution readiness token reusable for future recovery: no",
            f"- recovery execution readiness token boundary rows: {len(recovery_execution_readiness_token_boundary_rows)}",
            f"- recovery execution readiness token boundary ready: {'yes' if recovery_execution_readiness_token_boundary_ready else 'no'}",
            f"- stop condition: {stop_condition or 'missing'}",
            f"- approval boundary: {approval_boundary}",
            "",
            "Measured recovery execution readiness proof:",
            f"- score: {recovery_execution_score}/{recovery_execution_max_score}",
            f"- required rows ready: {'yes' if recovery_execution_required_rows_ready else 'no'}",
            f"- recovery execution scorecard ready: {'yes' if recovery_execution_scorecard_ready else 'no'}",
            *[
                f"- {row['item']}: {row['points']}/{row['max_points']} ({'ready' if row['ready'] else 'held'}; prior proof only; does not authorize risky work or unreviewed follow-through)"
                for row in recovery_execution_scorecard_rows
            ],
            "",
            "Boundary:",
            "- read-only follow-through packet; no shell/code execution, computer control, private-data reads, external side effects, note writes, approvals, reruns, or queued approvals.",
        ]
        return ToolResult(
            "checkpoint_recovery_followthrough_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                objective_length=len(objective),
                objective=recovery_execution_objective,
                followthrough_state=state,
                normal_followthrough_allowed=ready,
                next_safe_command=next_command,
                recovery_followthrough_gate=closure_gate,
                recovery_followthrough_gate_state=closure_gate["state"],
                recovery_closure_missing=closure_gate["missing"],
                recovery_closure_missing_count=closure_gate["missing_count"],
                recovery_closure_required_evidence=closure_gate["required_evidence"],
                recovery_closure_required_evidence_count=len(closure_gate["required_evidence"]),
                reviewed_step_provided=bool(reviewed_step),
                verification_provided=bool(verification),
                receipt_path_provided=bool(receipt_path),
                receipt_path_display=receipt_path_display,
                receipt_sha256=receipt_sha256,
                receipt_hash_provided=_looks_like_sha256(receipt_sha256),
                receipt_file_sha256=receipt_file_sha256,
                receipt_file_hash_present=_looks_like_sha256(receipt_file_sha256),
                receipt_hash_matches_file=receipt_hash_matches_file,
                checkpoint_path_provided=bool(checkpoint_path),
                checkpoint_path_display=checkpoint_path_display,
                checkpoint_sha256=checkpoint_sha256,
                checkpoint_hash_provided=_looks_like_sha256(checkpoint_sha256),
                checkpoint_file_sha256=checkpoint_file_sha256,
                checkpoint_file_hash_present=_looks_like_sha256(checkpoint_file_sha256),
                checkpoint_hash_matches_file=checkpoint_hash_matches_file,
                recovery_artifact_hashes_present=bool(receipt_hash_matches_file and checkpoint_hash_matches_file),
                recovery_artifact_hashes_match_files=bool(receipt_hash_matches_file and checkpoint_hash_matches_file),
                recovery_followthrough_token_sha256=recovery_followthrough_token_sha256,
                recovery_followthrough_token_present=_looks_like_sha256(recovery_followthrough_token_sha256),
                recovery_followthrough_token_boundary_rows=recovery_followthrough_token_boundary_rows,
                recovery_followthrough_token_boundary_row_count=len(recovery_followthrough_token_boundary_rows),
                recovery_followthrough_token_boundary_ready=recovery_followthrough_token_boundary_ready,
                recovery_followthrough_token_reusable_for_future_recovery=False,
                recovery_followthrough_token_authorizes_resume_gate=False,
                recovery_followthrough_token_authorizes_next_step=False,
                recovery_followthrough_token_authorizes_risky_work=False,
                recovery_followthrough_token_authorizes_approval=False,
                recovery_followthrough_token_authorizes_model_call=False,
                recovery_followthrough_token_authorizes_tool_execution=False,
                recovery_followthrough_token_authorizes_personal_data_read=False,
                recovery_followthrough_token_authorizes_external_side_effect=False,
                next_recovery_followthrough_requires_new_token=True,
                local_safe_recovery_execution_token_sha256=local_safe_recovery_execution_token_sha256,
                local_safe_recovery_execution_token_present=_looks_like_sha256(local_safe_recovery_execution_token_sha256),
                local_safe_recovery_execution_token_authorizes_resume_gate=False,
                local_safe_recovery_execution_token_authorizes_next_step=False,
                local_safe_recovery_execution_token_authorizes_risky_work=False,
                local_safe_recovery_execution_token_authorizes_approval=False,
                local_safe_recovery_execution_token_authorizes_model_call=False,
                local_safe_recovery_execution_token_authorizes_tool_execution=False,
                local_safe_recovery_execution_token_authorizes_personal_data_read=False,
                local_safe_recovery_execution_token_authorizes_external_side_effect=False,
                local_safe_recovery_execution_token_reusable_for_future_recovery=False,
                next_recovery_execution_requires_new_local_safe_token=True,
                local_safe_recovery_execution_token_boundary_rows=local_safe_recovery_execution_token_boundary_rows,
                local_safe_recovery_execution_token_boundary_row_count=len(local_safe_recovery_execution_token_boundary_rows),
                local_safe_recovery_execution_token_boundary_ready=local_safe_recovery_execution_token_boundary_ready,
                recovery_execution_score=recovery_execution_score,
                recovery_execution_max_score=recovery_execution_max_score,
                recovery_execution_scorecard_rows=recovery_execution_scorecard_rows,
                recovery_execution_scorecard_row_count=len(recovery_execution_scorecard_rows),
                recovery_execution_required_rows_ready=recovery_execution_required_rows_ready,
                recovery_execution_scorecard_ready=recovery_execution_scorecard_ready,
                recovery_execution_contract_fields=[],
                recovery_execution_contract_field_count=0,
                recovery_execution_readiness_token_sha256=recovery_execution_readiness_token_sha256,
                recovery_execution_readiness_token_present=_looks_like_sha256(recovery_execution_readiness_token_sha256),
                recovery_execution_readiness_token_boundary_rows=recovery_execution_readiness_token_boundary_rows,
                recovery_execution_readiness_token_boundary_row_count=len(recovery_execution_readiness_token_boundary_rows),
                recovery_execution_readiness_token_boundary_ready=recovery_execution_readiness_token_boundary_ready,
                recovery_execution_readiness_token_authorizes_resume_gate=False,
                recovery_execution_readiness_token_authorizes_next_step=False,
                recovery_execution_readiness_token_authorizes_risky_work=False,
                recovery_execution_readiness_token_authorizes_approval=False,
                recovery_execution_readiness_token_authorizes_model_call=False,
                recovery_execution_readiness_token_authorizes_tool_execution=False,
                recovery_execution_readiness_token_authorizes_personal_data_read=False,
                recovery_execution_readiness_token_authorizes_external_side_effect=False,
                recovery_execution_readiness_token_authorizes_unreviewed_followthrough=False,
                recovery_execution_readiness_token_reusable_for_future_recovery=False,
                next_recovery_execution_requires_new_readiness_token=True,
                recovery_execution_readiness_as_prior_proof=bool(
                    ready
                    and recovery_execution_required_rows_ready
                    and recovery_execution_score == recovery_execution_max_score == 100
                    and all(
                        row.get("authorizes_risky_work") is False
                        and row.get("authorizes_unreviewed_followthrough") is False
                        and row.get("authorizes_model_call") is False
                        and row.get("authorizes_tool_execution") is False
                        and row.get("authorizes_personal_data_read") is False
                        and row.get("authorizes_external_side_effect") is False
                        for row in recovery_execution_scorecard_rows
                    )
                ),
                recovery_execution_readiness_authorizes_action_now=False,
                recovery_execution_readiness_authorizes_risky_work=False,
                recovery_execution_readiness_authorizes_unreviewed_followthrough=False,
                recovery_execution_readiness_authorizes_model_call=False,
                recovery_execution_readiness_authorizes_tool_execution=False,
                recovery_execution_readiness_authorizes_personal_data_read=False,
                recovery_execution_readiness_authorizes_external_side_effect=False,
                stop_condition_provided=bool(stop_condition),
                approval_boundary_provided=bool(approval_boundary),
                approval_reference_provided=approval_reference_present,
                risky_recovery_signals=risk_signals,
                risky_recovery_signal_count=len(risk_signals),
                risky_recovery_without_approval=bool(risk_signals and not approval_reference_present),
                recovery_step_requires_approval=bool(risk_signals),
                reviewed_step=reviewed_step[:500],
                verification_target=verification[:240],
                receipt_path=receipt_path,
                checkpoint_path=checkpoint_path,
                stop_condition=stop_condition,
                approval_boundary=approval_boundary,
                blockers=blockers[:500],
            ),
        )

    def checkpoint_recovery_cockpit(args: dict[str, Any]) -> ToolResult:
        objective = str(args.get("objective") or args.get("request") or "Continue Jarvis V2 safely.").strip()
        stop_at = str(args.get("stop_at") or args.get("stop") or args.get("until") or "").strip()
        current_time = str(args.get("current_time") or args.get("now") or "").strip()
        timezone_label = str(args.get("timezone") or "local").strip() or "local"
        timebox = operator_timebox_contract(
            {
                "objective": objective,
                "stop_at": stop_at,
                "current_time": current_time,
                "timezone": timezone_label,
            }
        )
        preview = checkpoint_recovery_preview({"objective": objective})
        apply_packet = checkpoint_recovery_apply_packet({"objective": objective})
        timebox_metadata = timebox.metadata
        preview_metadata = preview.metadata
        apply_metadata = apply_packet.metadata
        proof_queue = list(preview_metadata.get("checkpoint_recovery_proof_queue") or [])
        recovery_apply_proof_queue = [str(command) for command in (apply_metadata.get("recovery_apply_proof_queue") or [])]
        recovery_apply_proof_queue_count = int(apply_metadata.get("recovery_apply_proof_queue_count") or 0)
        recovery_apply_next_proof_command = str(apply_metadata.get("recovery_apply_next_proof_command") or "")
        recovery_apply_boundary_ready = bool(
            apply_metadata.get("apply_requires_approval")
            and recovery_apply_proof_queue
            and recovery_apply_proof_queue_count == len(recovery_apply_proof_queue)
            and recovery_apply_next_proof_command == recovery_apply_proof_queue[0]
        )
        pending_approvals = int(preview_metadata.get("pending_approvals") or 0)
        failed_runs = int(preview_metadata.get("failed_runs") or 0)
        blockers: list[str] = []
        if not timebox_metadata.get("can_continue_now"):
            blockers.append(f"timebox {timebox_metadata.get('timebox_state') or 'held'}")
        if preview_metadata.get("checkpoint_needs_review"):
            blockers.append(f"checkpoint {preview_metadata.get('checkpoint_freshness') or 'needs_review'}")
        if pending_approvals:
            blockers.append("pending approvals require approval proof chain")
        if failed_runs:
            blockers.append("failed or blocked runs require recovery closure")
        if apply_metadata.get("apply_requires_approval"):
            blockers.append("recovery apply remains approval-gated planning")
        recovery_ready_for_review = bool(
            _metadata_bool(preview_metadata.get("checkpoint_found"))
            and _metadata_bool(preview_metadata.get("checkpoint_read"))
            and not preview_metadata.get("checkpoint_needs_review")
            and not pending_approvals
            and not failed_runs
        )
        can_resume_local_safe = bool(_metadata_bool(timebox_metadata.get("can_continue_now")) and recovery_ready_for_review)
        state = "RECOVERY_COCKPIT_READY_FOR_LOCAL_SAFE_REVIEW" if can_resume_local_safe else "RECOVERY_COCKPIT_HELD"
        if not proof_queue:
            proof_queue = ["work block checkpoint", "build delta", "completion claim gate"]
        next_safe_command = (
            "choose one reviewed local-safe checkpoint step, verify it, then save a fresh checkpoint"
            if can_resume_local_safe
            else proof_queue[0]
        )
        checkpoint_needs_review = preview_metadata.get("checkpoint_needs_review")
        if checkpoint_needs_review not in {True, False}:
            checkpoint_needs_review = True
        cockpit_scorecard_rows = _recovery_cockpit_readiness_scorecard_rows(
            can_continue_now=_metadata_bool(timebox_metadata.get("can_continue_now")),
            checkpoint_found=_metadata_bool(preview_metadata.get("checkpoint_found")),
            checkpoint_read=_metadata_bool(preview_metadata.get("checkpoint_read")),
            checkpoint_needs_review=checkpoint_needs_review,
            pending_approvals=pending_approvals,
            failed_runs=failed_runs,
            proof_queue=proof_queue,
            apply_requires_approval=_metadata_bool(apply_metadata.get("apply_requires_approval")),
            recovery_apply_proof_queue=recovery_apply_proof_queue,
            recovery_apply_proof_queue_count=recovery_apply_proof_queue_count,
            recovery_apply_next_proof_command=recovery_apply_next_proof_command,
            can_resume_local_safe=can_resume_local_safe,
            recovery_ready_for_review=recovery_ready_for_review,
        )
        cockpit_score = sum(int(row["points"]) for row in cockpit_scorecard_rows)
        cockpit_max_score = sum(int(row["max_points"]) for row in cockpit_scorecard_rows)
        cockpit_required_rows_ready = _scorecard_required_rows_ready(cockpit_scorecard_rows)
        cockpit_scorecard_ready = _recovery_cockpit_readiness_scorecard_ready(cockpit_scorecard_rows)
        cockpit_scorecard_strict_ready = _recovery_cockpit_readiness_scorecard_ready(
            cockpit_scorecard_rows,
            require_all_ready=True,
        )
        checkpoint_route_token_sha256 = _checkpoint_route_token_sha256(
            objective=objective,
            latest_checkpoint_path=str(preview_metadata.get("latest_checkpoint_path") or ""),
            latest_checkpoint_sha256=str(preview_metadata.get("latest_checkpoint_sha256") or ""),
            checkpoint_freshness=str(preview_metadata.get("checkpoint_freshness") or "missing"),
            checkpoint_needs_review=checkpoint_needs_review,
        )
        checkpoint_route_boundary_rows = _checkpoint_route_boundary_rows(
            token_sha256=checkpoint_route_token_sha256,
            source="checkpoint_recovery_cockpit",
            checkpoint_freshness=str(preview_metadata.get("checkpoint_freshness") or "missing"),
            checkpoint_needs_review=checkpoint_needs_review,
        )
        checkpoint_route_boundary_ready = _checkpoint_route_boundary_ready(
            checkpoint_route_token_sha256,
            checkpoint_route_boundary_rows,
            expected_source="checkpoint_recovery_cockpit",
            expected_checkpoint_freshness=str(preview_metadata.get("checkpoint_freshness") or "missing"),
            expected_checkpoint_needs_review=checkpoint_needs_review,
        )
        lines = [
            "Jarvis checkpoint recovery cockpit:",
            "This is read-only. It consolidates the operator timebox, latest checkpoint recovery preview, recovery apply packet, proof queue, blockers, and next safe command without executing recovery.",
            "",
            "Objective:",
            f"- {objective[:300]}",
            "",
            "Cockpit state:",
            f"- state: {state}",
            f"- can resume local-safe review now: {'yes' if can_resume_local_safe else 'no'}",
            f"- blockers: {', '.join(blockers) if blockers else 'none'}",
            f"- next safe command: `{next_safe_command}`",
            "",
            "Measured cockpit readiness scorecard:",
            f"- score: {cockpit_score}/{cockpit_max_score}",
            f"- required rows ready: {'yes' if cockpit_required_rows_ready else 'no'}",
            f"- scorecard shape ready: {'yes' if cockpit_scorecard_ready else 'no'}",
            *[
                f"- {row['item']}: {row['points']}/{row['max_points']} ({'ready' if row['ready'] else 'held'}; does not authorize execution or risky work)"
                for row in cockpit_scorecard_rows
            ],
            "",
            "Operator timebox:",
            f"- state: {timebox_metadata.get('timebox_state')}",
            f"- can continue now: {'yes' if timebox_metadata.get('can_continue_now') else 'no'}",
            f"- minutes remaining: {timebox_metadata.get('minutes_remaining') if timebox_metadata.get('minutes_remaining') is not None else 'unknown'}",
            f"- missing proof: {', '.join(timebox_metadata.get('missing_timebox_proof') or []) if timebox_metadata.get('missing_timebox_proof') else 'none'}",
            "",
            "Checkpoint recovery:",
            f"- checkpoint found: {'yes' if preview_metadata.get('checkpoint_found') else 'no'}",
            f"- checkpoint read: {'yes' if preview_metadata.get('checkpoint_read') else 'no'}",
            f"- latest checkpoint: {preview_metadata.get('latest_checkpoint_display') or 'missing'}",
            f"- latest checkpoint sha256: {preview_metadata.get('latest_checkpoint_sha256') or 'missing'}",
            f"- freshness: {preview_metadata.get('checkpoint_freshness')}",
            f"- age minutes: {preview_metadata.get('checkpoint_age_minutes') if preview_metadata.get('checkpoint_age_minutes') is not None else 'unknown'}",
            f"- needs review: {'yes' if preview_metadata.get('checkpoint_needs_review') else 'no'}",
            f"- pending approvals: {pending_approvals}",
            f"- failed or blocked runs: {failed_runs}",
            "",
            "Checkpoint route boundary:",
            f"- route token sha256: {checkpoint_route_token_sha256}",
            f"- boundary rows: {len(checkpoint_route_boundary_rows)}",
            f"- boundary ready: {'yes' if checkpoint_route_boundary_ready else 'no'}",
            "- stale, missing, unknown, or review-again checkpoints route to recovery review; this token does not authorize continuation, local-safe steps, risky work, approvals, recovery follow-through, checkpoint reuse, model/tool calls, personal-data reads, external side effects, or reuse.",
            *[
                f"- {row['item']}: status {row['status']}; freshness {row['checkpoint_freshness']}; needs review {'yes' if row['checkpoint_needs_review'] else 'no'}; authorizes continuation no; reusable no"
                for row in checkpoint_route_boundary_rows
            ],
            "",
            "Recovery proof queue:",
            f"- route next recovery command: `{proof_queue[0]}`" if proof_queue else "- route next recovery command: none",
            f"- next required command: `{proof_queue[0]}`" if proof_queue else "- next required command: none",
            f"- proof queue count: {len(proof_queue)}",
        ]
        lines.extend(f"- `{command}`" for command in proof_queue[:8])
        lines.extend(
            [
                "",
                "Required closure before normal follow-through:",
                "- reviewed local-safe step",
                "- verification evidence",
                "- checkpoint recovery receipt",
                "- fresh work-block checkpoint",
                "- explicit stop condition",
                "- approval boundary for future risky steps",
                "- checkpoint recovery follow-through packet",
                "",
                "Boundary:",
                "- read-only recovery cockpit; no shell/code execution, computer control, private-data reads, external side effects, note writes, approvals, reruns, or queued approvals.",
            ]
        )
        return ToolResult(
            "checkpoint_recovery_cockpit",
            True,
            "\n".join(lines),
            _safe_metadata(
                objective_length=len(objective),
                cockpit_state=state,
                can_resume_local_safe_review=can_resume_local_safe,
                recovery_ready_for_review=recovery_ready_for_review,
                blockers=blockers,
                blocker_count=len(blockers),
                next_safe_command=next_safe_command,
                recovery_cockpit_score=cockpit_score,
                recovery_cockpit_max_score=cockpit_max_score,
                recovery_cockpit_scorecard_rows=cockpit_scorecard_rows,
                recovery_cockpit_scorecard_row_count=len(cockpit_scorecard_rows),
                recovery_cockpit_scorecard_ready=cockpit_scorecard_ready,
                recovery_cockpit_scorecard_strict_ready=cockpit_scorecard_strict_ready,
                recovery_cockpit_required_rows_ready=cockpit_required_rows_ready,
                timebox_state=timebox_metadata.get("timebox_state"),
                can_continue_now=timebox_metadata.get("can_continue_now"),
                should_stop_now=timebox_metadata.get("should_stop_now"),
                minutes_remaining=timebox_metadata.get("minutes_remaining"),
                missing_timebox_proof=timebox_metadata.get("missing_timebox_proof", []),
                missing_timebox_proof_count=timebox_metadata.get("missing_timebox_proof_count", 0),
                checkpoint_found=preview_metadata.get("checkpoint_found", False),
                checkpoint_read=preview_metadata.get("checkpoint_read", False),
                latest_checkpoint_path=preview_metadata.get("latest_checkpoint_path", ""),
                latest_checkpoint_display=preview_metadata.get("latest_checkpoint_display", ""),
                latest_checkpoint_sha256=preview_metadata.get("latest_checkpoint_sha256", ""),
                latest_checkpoint_hash_present=preview_metadata.get("latest_checkpoint_hash_present", False),
                checkpoint_age_minutes=preview_metadata.get("checkpoint_age_minutes"),
                checkpoint_freshness=preview_metadata.get("checkpoint_freshness"),
                checkpoint_needs_review=preview_metadata.get("checkpoint_needs_review", True),
                checkpoint_route_token_sha256=checkpoint_route_token_sha256,
                checkpoint_route_token_present=_looks_like_sha256(checkpoint_route_token_sha256),
                checkpoint_route_boundary_rows=checkpoint_route_boundary_rows,
                checkpoint_route_boundary_row_count=len(checkpoint_route_boundary_rows),
                checkpoint_route_boundary_ready=checkpoint_route_boundary_ready,
                checkpoint_route_authorizes_continuation=False,
                checkpoint_route_authorizes_local_safe_step=False,
                checkpoint_route_authorizes_risky_work=False,
                checkpoint_route_authorizes_approval=False,
                checkpoint_route_authorizes_recovery_followthrough=False,
                checkpoint_route_authorizes_checkpoint_reuse=False,
                checkpoint_route_authorizes_model_call=False,
                checkpoint_route_authorizes_tool_execution=False,
                checkpoint_route_authorizes_personal_data_read=False,
                checkpoint_route_authorizes_external_side_effect=False,
                checkpoint_route_reusable_for_next_review=False,
                checkpoint_route_reusable_for_next_checkpoint=False,
                next_checkpoint_route_requires_fresh_recovery_review=True,
                checkpoint_route_recovery_proof_queue=proof_queue,
                checkpoint_route_recovery_proof_queue_count=len(proof_queue),
                checkpoint_route_next_recovery_command=proof_queue[0] if proof_queue else "",
                checkpoint_route_next_required_command=proof_queue[0] if proof_queue else "",
                checkpoint_recovery_proof_queue=proof_queue,
                checkpoint_recovery_proof_queue_count=len(proof_queue),
                checkpoint_recovery_next_proof_command=proof_queue[0] if proof_queue else "",
                checkpoint_recovery_next_required_command=proof_queue[0] if proof_queue else "",
                next_required_command=proof_queue[0] if proof_queue else "",
                apply_requires_approval=apply_metadata.get("apply_requires_approval", True),
                recovery_apply_boundary_ready=recovery_apply_boundary_ready,
                recovery_apply_proof_queue=recovery_apply_proof_queue,
                recovery_apply_proof_queue_count=recovery_apply_proof_queue_count,
                recovery_apply_next_proof_command=recovery_apply_next_proof_command,
                pending_approvals=pending_approvals,
                failed_runs=failed_runs,
                required_closure_evidence=[
                    "reviewed local-safe step",
                    "verification evidence",
                    "checkpoint recovery receipt",
                    "fresh work-block checkpoint",
                    "stop condition",
                    "approval boundary",
                    "checkpoint recovery follow-through packet",
                ],
                timebox_output=timebox.output[:1200],
                recovery_preview_output=preview.output[:1600],
                recovery_apply_output=apply_packet.output[:1600],
            ),
        )

    def autonomy_resume_gate(args: dict[str, Any]) -> ToolResult:
        objective = str(args.get("objective") or args.get("request") or "Continue Jarvis V2 safely.").strip()
        previous_instruction = str(args.get("previous_instruction") or args.get("previous") or "").strip()
        latest_instruction = str(args.get("latest_instruction") or args.get("latest") or args.get("instruction") or objective).strip()
        stop_at = str(args.get("stop_at") or args.get("stop") or args.get("until") or "").strip()
        current_time = str(args.get("current_time") or args.get("now") or "").strip()
        timezone_label = str(args.get("timezone") or "local").strip() or "local"
        reviewed_step = str(args.get("reviewed_step") or args.get("step") or args.get("approved_step") or "").strip()
        verification = str(args.get("verification") or args.get("tests") or args.get("verification_target") or "").strip()
        receipt_path = str(args.get("receipt_path") or args.get("receipt") or "").strip()
        receipt_sha256 = str(args.get("receipt_sha256") or args.get("receipt_hash") or "").strip()
        checkpoint_path = str(args.get("checkpoint_path") or args.get("checkpoint") or "").strip()
        checkpoint_sha256 = str(args.get("checkpoint_sha256") or args.get("checkpoint_hash") or "").strip()
        stop_condition = str(args.get("stop_condition") or args.get("stop_rule") or "").strip()
        approval_reference = str(args.get("approval") or args.get("approval_reference") or "").strip()
        blockers_text = str(args.get("blockers") or "none reported").strip()

        timebox = operator_timebox_contract(
            {"objective": objective, "stop_at": stop_at, "current_time": current_time, "timezone": timezone_label}
        )
        supersession = operator_instruction_supersession_packet(
            {
                "objective": objective,
                "previous_instruction": previous_instruction,
                "latest_instruction": latest_instruction,
                "stop_at": stop_at,
                "current_time": current_time,
                "timezone": timezone_label,
            }
        )
        cockpit = checkpoint_recovery_cockpit(
            {"objective": objective, "stop_at": stop_at, "current_time": current_time, "timezone": timezone_label}
        )
        followthrough = checkpoint_recovery_followthrough_packet(
            {
                "objective": objective,
                "reviewed_step": reviewed_step,
                "verification": verification,
                "receipt_path": receipt_path,
                "receipt_sha256": receipt_sha256,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": checkpoint_sha256,
                "stop_condition": stop_condition,
                "approval_reference": approval_reference,
                "blockers": blockers_text,
            }
        )
        timebox_metadata = timebox.metadata
        supersession_metadata = supersession.metadata
        cockpit_metadata = cockpit.metadata
        followthrough_metadata = followthrough.metadata
        supersession_token_boundary_rows = list(supersession_metadata.get("supersession_token_boundary_rows") or [])
        supersession_token_boundary_ready = _operator_supersession_token_boundary_ready(
            str(supersession_metadata.get("supersession_token_sha256") or ""),
            supersession_token_boundary_rows,
        )
        checkpoint_route_boundary_rows = list(cockpit_metadata.get("checkpoint_route_boundary_rows") or [])
        checkpoint_route_boundary_ready = _checkpoint_route_boundary_ready(
            str(cockpit_metadata.get("checkpoint_route_token_sha256") or ""),
            checkpoint_route_boundary_rows,
        )
        timebox_review_contract_rows = list(timebox_metadata.get("timebox_review_contract_rows") or [])
        timebox_review_contract_ready = bool(
            timebox_metadata.get("timebox_review_contract_ready")
            and _operator_timebox_review_contract_ready(
                timebox_review_contract_rows,
                timebox_state=str(timebox_metadata.get("timebox_state") or ""),
                stop_at=str(timebox_metadata.get("stop_at") or ""),
                current_time=str(timebox_metadata.get("current_time") or ""),
                timezone_label=str(timebox_metadata.get("timezone") or ""),
            )
        )

        blockers: list[str] = []
        if not timebox_metadata.get("can_continue_now"):
            blockers.append(f"timebox_{timebox_metadata.get('timebox_state') or 'held'}")
        if not timebox_review_contract_ready:
            blockers.append("timebox_review_contract_not_ready")
        if not supersession_metadata.get("can_continue_under_latest_instruction"):
            blockers.append("latest_instruction_supersession_not_ready")
        if not supersession_token_boundary_ready:
            blockers.append("operator_supersession_token_boundary_not_ready")
        if not cockpit_metadata.get("can_resume_local_safe_review"):
            blockers.append("checkpoint_recovery_cockpit_not_ready")
        if not checkpoint_route_boundary_ready:
            blockers.append("checkpoint_route_boundary_not_ready")
        latest_checkpoint_path = str(cockpit_metadata.get("latest_checkpoint_path") or "").strip()
        latest_checkpoint_display = str(cockpit_metadata.get("latest_checkpoint_display") or "").strip()
        latest_checkpoint_sha256 = str(cockpit_metadata.get("latest_checkpoint_sha256") or "").strip()
        checkpoint_path_matches_latest = bool(checkpoint_path and latest_checkpoint_path and checkpoint_path == latest_checkpoint_path)
        checkpoint_hash_matches_latest = bool(
            checkpoint_sha256
            and latest_checkpoint_sha256
            and checkpoint_sha256 == latest_checkpoint_sha256
        )
        if cockpit_metadata.get("can_resume_local_safe_review") and not checkpoint_path_matches_latest:
            blockers.append("checkpoint_path_not_latest_recovery_checkpoint")
        if cockpit_metadata.get("can_resume_local_safe_review") and not checkpoint_hash_matches_latest:
            blockers.append("checkpoint_hash_not_latest_recovery_checkpoint")
        if not followthrough_metadata.get("normal_followthrough_allowed"):
            blockers.append("followthrough_closure_not_ready")
        if followthrough_metadata.get("risky_recovery_without_approval"):
            blockers.append("risky_recovery_step_without_approval_reference")

        if blockers and blockers[0].startswith("timebox_"):
            state = "AUTONOMY_RESUME_HELD_FOR_TIMEBOX"
            next_safe_command = "operator timebox contract: stop_at=<ISO> current_time=<ISO>"
        elif "latest_instruction_supersession_not_ready" in blockers or "operator_supersession_token_boundary_not_ready" in blockers:
            state = "AUTONOMY_RESUME_HELD_FOR_OPERATOR_SUPERSESSION"
            next_safe_command = "operator instruction supersession: latest=<newest instruction> stop_at=<ISO> current_time=<ISO>"
        elif "checkpoint_recovery_cockpit_not_ready" in blockers:
            state = "AUTONOMY_RESUME_HELD_FOR_RECOVERY_COCKPIT"
            next_safe_command = "checkpoint recovery cockpit: " + objective[:180]
        elif "checkpoint_path_not_latest_recovery_checkpoint" in blockers:
            state = "AUTONOMY_RESUME_HELD_FOR_CHECKPOINT_BINDING"
            next_safe_command = "checkpoint recovery preview: " + objective[:180]
        elif "checkpoint_hash_not_latest_recovery_checkpoint" in blockers:
            state = "AUTONOMY_RESUME_HELD_FOR_CHECKPOINT_HASH_BINDING"
            next_safe_command = "checkpoint recovery preview: " + objective[:180]
        elif blockers:
            state = "AUTONOMY_RESUME_HELD_FOR_FOLLOWTHROUGH_CLOSURE"
            next_safe_command = "checkpoint recovery follow-through: step=<reviewed local-safe step> verification=<evidence> receipt=<receipt path> receipt_sha256=<hash> checkpoint=<checkpoint path> checkpoint_sha256=<hash> stop=<stop condition>"
        else:
            state = "AUTONOMY_RESUME_READY_FOR_LOCAL_SAFE_CONTINUATION"
            next_safe_command = "continue one scoped local-safe Jarvis build step from the fresh checkpoint"

        recovery_artifact_hashes_match_files = bool(
            followthrough_metadata.get("receipt_hash_matches_file")
            and followthrough_metadata.get("checkpoint_hash_matches_file")
        )
        carried_recovery_execution_scorecard_rows = list(
            followthrough_metadata.get("recovery_execution_scorecard_rows") or []
        )
        carried_recovery_execution_score = int(followthrough_metadata.get("recovery_execution_score") or 0)
        carried_recovery_execution_max_score = int(followthrough_metadata.get("recovery_execution_max_score") or 0)
        carried_recovery_execution_required_rows_ready = _recovery_execution_readiness_scorecard_ready(
            carried_recovery_execution_scorecard_rows
        )
        carried_recovery_execution_scorecard_ready = carried_recovery_execution_required_rows_ready
        carried_recovery_execution_as_prior_proof = bool(
            followthrough_metadata.get("recovery_execution_readiness_as_prior_proof")
            and _recovery_execution_readiness_scorecard_ready(carried_recovery_execution_scorecard_rows)
        )
        proof_queue = [
            f"operator timebox contract: {objective[:180]}",
            f"checkpoint recovery cockpit: {objective[:180]}",
            "checkpoint recovery follow-through: step=<reviewed local-safe step> verification=<evidence> receipt=<receipt path> receipt_sha256=<hash> checkpoint=<checkpoint path> checkpoint_sha256=<hash> stop=<stop condition>",
            "continuation packet: " + objective[:180],
            "work block checkpoint",
            "execution health report",
            "completion claim gate",
        ]
        resume_scorecard_rows = _autonomy_resume_readiness_scorecard_rows(
            can_continue_now=_metadata_bool(timebox_metadata.get("can_continue_now")),
            can_resume_local_safe_review=_metadata_bool(cockpit_metadata.get("can_resume_local_safe_review")),
            checkpoint_path_matches_latest=checkpoint_path_matches_latest,
            checkpoint_hash_matches_latest=checkpoint_hash_matches_latest,
            followthrough_closure_ready=_metadata_bool(followthrough_metadata.get("normal_followthrough_allowed")),
            recovery_artifact_hashes_match_files=recovery_artifact_hashes_match_files,
            risky_recovery_without_approval=_metadata_bool(followthrough_metadata.get("risky_recovery_without_approval")),
            recovery_followthrough_token_sha256=str(
                followthrough_metadata.get("recovery_followthrough_token_sha256") or ""
            ),
            resume_proof_queue=proof_queue,
            resume_proof_queue_count=len(proof_queue),
            resume_next_proof_command=proof_queue[0],
        )
        resume_score = sum(int(row["points"]) for row in resume_scorecard_rows)
        resume_max_score = sum(int(row["max_points"]) for row in resume_scorecard_rows)
        resume_required_rows_ready = _scorecard_required_rows_ready(resume_scorecard_rows)
        resume_scorecard_ready = _autonomy_resume_readiness_scorecard_ready(resume_scorecard_rows)
        local_safe_recovery_execution_token_boundary_rows = _local_safe_recovery_execution_token_boundary_rows(
            token_sha256=str(followthrough_metadata.get("local_safe_recovery_execution_token_sha256") or ""),
            source="autonomy_resume_gate",
        )
        local_safe_recovery_execution_token_boundary_ready = _local_safe_recovery_execution_token_boundary_ready(
            str(followthrough_metadata.get("local_safe_recovery_execution_token_sha256") or ""),
            local_safe_recovery_execution_token_boundary_rows,
            expected_source="autonomy_resume_gate",
        )
        recovery_followthrough_token_boundary_rows = list(
            followthrough_metadata.get("recovery_followthrough_token_boundary_rows") or []
        )
        recovery_followthrough_token_boundary_ready = _carried_recovery_followthrough_token_boundary_ready(
            followthrough_metadata,
            expected_source="checkpoint_recovery_followthrough",
        )
        recovery_execution_readiness_token_boundary_rows = list(
            followthrough_metadata.get("recovery_execution_readiness_token_boundary_rows") or []
        )
        recovery_execution_readiness_token_boundary_ready = _carried_recovery_execution_readiness_token_boundary_ready(
            followthrough_metadata,
            expected_source="checkpoint_recovery_followthrough",
        )

        lines = [
            "Jarvis autonomy resume gate:",
            "This is read-only. It binds the operator timebox, checkpoint recovery cockpit, and recovery follow-through closure before normal autonomous local-safe continuation can resume.",
            "",
            "Objective:",
            f"- {objective[:300]}",
            "",
            "Resume gate:",
            f"- state: {state}",
            f"- normal autonomous follow-through allowed: {'yes' if not blockers else 'no'}",
            "- tools executed: no",
            "- approvals queued: no",
            f"- next safe command: `{next_safe_command}`",
            f"- blockers: {', '.join(blockers) if blockers else 'none'}",
            "",
            "Measured resume readiness scorecard:",
            f"- score: {resume_score}/{resume_max_score}",
            f"- required rows ready: {'yes' if resume_required_rows_ready else 'no'}",
            f"- scorecard ready: {'yes' if resume_scorecard_ready else 'no'}",
            *[
                f"- {row['item']}: {row['points']}/{row['max_points']} ({'ready' if row['ready'] else 'held'}; does not authorize risky work or unreviewed continuation)"
                for row in resume_scorecard_rows
            ],
            "",
            "Bound proof:",
            f"- timebox state: {timebox_metadata.get('timebox_state')}",
            f"- can continue now: {'yes' if timebox_metadata.get('can_continue_now') else 'no'}",
            f"- timebox receipt sha256: {timebox_metadata.get('timebox_receipt_sha256') or 'missing'}",
            f"- timebox review contract ready: {'yes' if timebox_review_contract_ready else 'no'}",
            f"- timebox review rows: {len(timebox_review_contract_rows)}",
            f"- cockpit state: {cockpit_metadata.get('cockpit_state')}",
            f"- can resume local-safe review: {'yes' if cockpit_metadata.get('can_resume_local_safe_review') else 'no'}",
            f"- operator supersession state: {supersession_metadata.get('supersession_state')}",
            f"- latest instruction can govern continuation: {'yes' if supersession_metadata.get('can_continue_under_latest_instruction') else 'no'}",
            f"- supersession token sha256: {supersession_metadata.get('supersession_token_sha256') or 'missing'}",
            f"- supersession token boundary rows: {len(supersession_token_boundary_rows)}",
            "- supersession token authorizes execution now: no",
            "- supersession token authorizes local-safe step: no",
            "- supersession token authorizes risky work: no",
            "- supersession token authorizes approval: no",
            "- supersession token authorizes recovery follow-through: no",
            "- supersession token authorizes timebox override: no",
            "- supersession token reusable for next review: no",
            f"- latest checkpoint: {latest_checkpoint_display or 'missing'}",
            f"- latest checkpoint sha256: {latest_checkpoint_sha256 or 'missing'}",
            f"- supplied checkpoint matches latest: {'yes' if checkpoint_path_matches_latest else 'no'}",
            f"- supplied checkpoint hash matches latest: {'yes' if checkpoint_hash_matches_latest else 'no'}",
            f"- receipt hash supplied: {'yes' if receipt_sha256 else 'no'}",
            f"- receipt hash matches file: {'yes' if followthrough_metadata.get('receipt_hash_matches_file') else 'no'}",
            f"- checkpoint hash supplied: {'yes' if checkpoint_sha256 else 'no'}",
            f"- checkpoint hash matches file: {'yes' if followthrough_metadata.get('checkpoint_hash_matches_file') else 'no'}",
            f"- recovery follow-through token sha256: {followthrough_metadata.get('recovery_followthrough_token_sha256') or 'missing'}",
            "- recovery follow-through token reusable for future recovery: no",
            f"- local-safe recovery execution token sha256: {followthrough_metadata.get('local_safe_recovery_execution_token_sha256') or 'missing'}",
            "- local-safe recovery execution token authorizes resume gate: no",
            "- local-safe recovery execution token authorizes next step: no",
            "- local-safe recovery execution token authorizes risky work: no",
            "- local-safe recovery execution token authorizes approval: no",
            "- local-safe recovery execution token authorizes model call: no",
            "- local-safe recovery execution token authorizes tool execution: no",
            "- local-safe recovery execution token authorizes personal-data read: no",
            "- local-safe recovery execution token authorizes external side effect: no",
            "- local-safe recovery execution token reusable for future recovery: no",
            f"- local-safe recovery execution token boundary rows: {len(local_safe_recovery_execution_token_boundary_rows)}",
            f"- follow-through state: {followthrough_metadata.get('followthrough_state')}",
            f"- follow-through closure ready: {'yes' if followthrough_metadata.get('normal_followthrough_allowed') else 'no'}",
            f"- recovery closure missing: {', '.join(followthrough_metadata.get('recovery_closure_missing') or []) if followthrough_metadata.get('recovery_closure_missing') else 'none'}",
            "",
            "Carried timebox review contract:",
            "- authorizes execution now: no",
            "- authorizes local-safe step: no",
            "- authorizes risky work: no",
            "- authorizes approval: no",
            "- reusable for next step: no",
            *[
                f"- {row['item']}: fresh required yes; prior reusable no; authorizes timebox reuse no"
                for row in timebox_review_contract_rows
            ],
            "",
            "Carried awake guard boundary:",
            f"- awake requested: {'yes' if timebox_metadata.get('awake_guard_requested') else 'no'}",
            f"- awake guard token sha256: {timebox_metadata.get('awake_guard_token_sha256') or 'missing'}",
            f"- boundary rows: {timebox_metadata.get('awake_guard_boundary_row_count') or 0}",
            "- authorizes OS wake lock: no",
            "- authorizes shell execution: no",
            "- authorizes computer control: no",
            "- authorizes approval: no",
            "- reusable for next timebox: no",
            "- fresh awake-guard review required next timebox: yes",
            "",
            "Carried operator supersession boundary:",
            f"- latest instruction supplied: {'yes' if supersession_metadata.get('latest_instruction_present') else 'no'}",
            f"- newest instruction overrides automation: {'yes' if supersession_metadata.get('newest_instruction_overrides_automation') else 'no'}",
            f"- boundary ready: {'yes' if supersession_token_boundary_ready else 'no'}",
            f"- boundary rows: {len(supersession_token_boundary_rows)}",
            *[
                f"- {row['item']}: {row['status']}; authorizes execution no; local-safe step no; risky work no; approval no; recovery follow-through no; reusable no"
                for row in supersession_token_boundary_rows
            ],
            "- next supersession requires fresh latest-instruction review: yes",
            "",
            "Carried checkpoint route boundary:",
            f"- checkpoint freshness: {cockpit_metadata.get('checkpoint_freshness') or 'missing'}",
            f"- checkpoint needs recovery review: {'yes' if cockpit_metadata.get('checkpoint_needs_review') else 'no'}",
            f"- route token sha256: {cockpit_metadata.get('checkpoint_route_token_sha256') or 'missing'}",
            f"- boundary ready: {'yes' if checkpoint_route_boundary_ready else 'no'}",
            f"- boundary rows: {len(checkpoint_route_boundary_rows)}",
            f"- route next recovery command: `{cockpit_metadata.get('checkpoint_route_next_recovery_command') or 'none'}`",
            *[
                f"- {row['item']}: {row['status']}; freshness {row['checkpoint_freshness']}; needs review {'yes' if row['checkpoint_needs_review'] else 'no'}; authorizes continuation no; local-safe step no; risky work no; approval no; checkpoint reuse no; reusable no"
                for row in checkpoint_route_boundary_rows
            ],
            "- next checkpoint route requires fresh recovery review: yes",
            "",
            "Carried recovery execution readiness proof:",
            f"- score: {carried_recovery_execution_score}/{carried_recovery_execution_max_score}",
            f"- required rows ready: {'yes' if carried_recovery_execution_required_rows_ready else 'no'}",
            f"- carried as prior proof: {'yes' if carried_recovery_execution_as_prior_proof else 'no'}",
            "- authorizes action now: no",
            "- authorizes risky work: no",
            "- authorizes unreviewed follow-through: no",
            *[
                f"- {row['item']}: {row['points']}/{row['max_points']} ({'ready' if row['ready'] else 'held'}; prior recovery proof only)"
                for row in carried_recovery_execution_scorecard_rows
            ],
            "",
            "Closure evidence supplied:",
            f"- reviewed step: {reviewed_step[:500] or 'missing'}",
            f"- verification: {verification[:500] or 'missing'}",
            f"- receipt path: {receipt_path or 'missing'}",
            f"- fresh checkpoint: {checkpoint_path or 'missing'}",
            f"- stop condition: {stop_condition or 'missing'}",
            f"- approval reference: {approval_reference or 'not supplied'}",
            "",
            "Proof queue:",
            *[f"- `{command}`" for command in proof_queue],
            "",
            "Boundary:",
            "- This gate does not run shell/code, execute tools, read private data, control the computer, approve requests, queue approvals, call external services, write notes, or resume risky work.",
            "- A ready state permits only one scoped local-safe continuation step; future risky steps still stop at approval readiness, approval packet, approval chain proof, and explicit approval.",
        ]
        return ToolResult(
            "autonomy_resume_gate",
            True,
            "\n".join(lines),
            _safe_metadata(
                objective=objective,
                objective_length=len(objective),
                stop_at=stop_at,
                current_time=current_time,
                timezone=timezone_label,
                resume_gate_state=state,
                normal_autonomous_followthrough_allowed=not blockers,
                tools_executed=False,
                approvals_queued=False,
                next_safe_command=next_safe_command,
                resume_readiness_score=resume_score,
                resume_readiness_max_score=resume_max_score,
                resume_readiness_scorecard_rows=resume_scorecard_rows,
                resume_readiness_scorecard_row_count=len(resume_scorecard_rows),
                resume_readiness_required_rows_ready=resume_required_rows_ready,
                resume_readiness_scorecard_ready=resume_scorecard_ready,
                autonomy_resume_gate_ready=not blockers and resume_scorecard_ready,
                blockers=blockers,
                blocker_count=len(blockers),
                timebox_state=timebox_metadata.get("timebox_state"),
                can_continue_now=timebox_metadata.get("can_continue_now"),
                should_stop_now=timebox_metadata.get("should_stop_now"),
                missing_timebox_proof=timebox_metadata.get("missing_timebox_proof", []),
                missing_timebox_proof_count=timebox_metadata.get("missing_timebox_proof_count", 0),
                timebox_receipt_sha256=timebox_metadata.get("timebox_receipt_sha256", ""),
                timebox_receipt_present=timebox_metadata.get("timebox_receipt_present", False),
                timebox_review_contract_rows=timebox_review_contract_rows,
                timebox_review_contract_row_count=len(timebox_review_contract_rows),
                timebox_review_contract_ready=timebox_review_contract_ready,
                timebox_review_contract_summary=timebox_metadata.get("timebox_review_contract_summary", []),
                timebox_authorizes_execution=False,
                timebox_authorizes_local_safe_step=False,
                timebox_authorizes_risky_work=False,
                timebox_authorizes_approval=False,
                timebox_authorizes_timebox_reuse=False,
                timebox_authorizes_model_call=False,
                timebox_authorizes_tool_execution=False,
                timebox_authorizes_personal_data_read=False,
                timebox_authorizes_external_side_effect=False,
                timebox_reusable_for_next_step=False,
                next_step_requires_fresh_timebox=True,
                awake_guard_requested=timebox_metadata.get("awake_guard_requested", False),
                awake_guard_token_sha256=timebox_metadata.get("awake_guard_token_sha256", ""),
                awake_guard_token_present=timebox_metadata.get("awake_guard_token_present", False),
                awake_guard_boundary_rows=timebox_metadata.get("awake_guard_boundary_rows", []),
                awake_guard_boundary_row_count=timebox_metadata.get("awake_guard_boundary_row_count", 0),
                awake_guard_authorizes_os_wake_lock=False,
                awake_guard_authorizes_shell_execution=False,
                awake_guard_authorizes_computer_control=False,
                awake_guard_authorizes_approval=False,
                awake_guard_authorizes_model_call=False,
                awake_guard_authorizes_tool_execution=False,
                awake_guard_authorizes_personal_data_read=False,
                awake_guard_authorizes_external_side_effect=False,
                awake_guard_reusable_for_next_timebox=False,
                awake_guard_requires_separate_operator_request=True,
                awake_guard_requires_separate_shell_approval=True,
                awake_guard_os_wake_lock_boundary_ready=_awake_guard_boundary_ready_from_metadata(
                    timebox_metadata,
                    default_objective=objective,
                    default_timezone=timezone_label,
                ),
                awake_guard_caffeinate_command_authorized=False,
                awake_guard_keep_awake_command_authorized=False,
                awake_guard_authorizes_unattended_execution=False,
                awake_guard_authorizes_continuation_window=False,
                awake_guard_reusable_as_execution_permission=False,
                next_awake_guard_requires_fresh_review=True,
                previous_instruction=previous_instruction[:300],
                latest_instruction=latest_instruction[:300],
                latest_instruction_present=supersession_metadata.get("latest_instruction_present", False),
                latest_instruction_supersedes_previous=supersession_metadata.get("latest_instruction_supersedes_previous", False),
                latest_instruction_is_stop=supersession_metadata.get("latest_instruction_is_stop", False),
                latest_instruction_is_continue=supersession_metadata.get("latest_instruction_is_continue", False),
                supersession_state=supersession_metadata.get("supersession_state", ""),
                can_continue_under_latest_instruction=supersession_metadata.get("can_continue_under_latest_instruction", False),
                newest_instruction_overrides_automation=supersession_metadata.get("newest_instruction_overrides_automation", True),
                newest_instruction_overrides_goal=supersession_metadata.get("newest_instruction_overrides_goal", True),
                newest_instruction_overrides_recovery_queue=supersession_metadata.get("newest_instruction_overrides_recovery_queue", True),
                stop_or_pause_blocks_autonomy=supersession_metadata.get("stop_or_pause_blocks_autonomy", False),
                supersession_token_sha256=supersession_metadata.get("supersession_token_sha256", ""),
                supersession_token_present=supersession_metadata.get("supersession_token_present", False),
                supersession_token_boundary_rows=supersession_token_boundary_rows,
                supersession_token_boundary_row_count=len(supersession_token_boundary_rows),
                supersession_token_boundary_ready=supersession_token_boundary_ready,
                supersession_token_authorizes_execution=False,
                supersession_token_authorizes_local_safe_step=False,
                supersession_token_authorizes_risky_work=False,
                supersession_token_authorizes_approval=False,
                supersession_token_authorizes_recovery_followthrough=False,
                supersession_token_authorizes_timebox_override=False,
                supersession_token_authorizes_goal_override=False,
                supersession_token_authorizes_model_call=False,
                supersession_token_authorizes_tool_execution=False,
                supersession_token_authorizes_personal_data_read=False,
                supersession_token_authorizes_external_side_effect=False,
                supersession_token_reusable_for_next_review=False,
                supersession_token_reusable_for_next_timebox=False,
                next_supersession_requires_fresh_latest_instruction_review=True,
                cockpit_state=cockpit_metadata.get("cockpit_state"),
                can_resume_local_safe_review=cockpit_metadata.get("can_resume_local_safe_review"),
                recovery_ready_for_review=cockpit_metadata.get("recovery_ready_for_review"),
                checkpoint_found=cockpit_metadata.get("checkpoint_found"),
                checkpoint_read=cockpit_metadata.get("checkpoint_read"),
                latest_checkpoint_path=latest_checkpoint_path,
                latest_checkpoint_display=latest_checkpoint_display,
                latest_checkpoint_sha256=latest_checkpoint_sha256,
                latest_checkpoint_hash_present=_looks_like_sha256(latest_checkpoint_sha256),
                supplied_checkpoint_path=checkpoint_path,
                checkpoint_path_matches_latest=checkpoint_path_matches_latest,
                checkpoint_hash_matches_latest=checkpoint_hash_matches_latest,
                receipt_sha256=receipt_sha256,
                receipt_hash_provided=_looks_like_sha256(receipt_sha256),
                receipt_file_sha256=followthrough_metadata.get("receipt_file_sha256", ""),
                receipt_file_hash_present=followthrough_metadata.get("receipt_file_hash_present", False),
                receipt_hash_matches_file=followthrough_metadata.get("receipt_hash_matches_file", False),
                checkpoint_sha256=checkpoint_sha256,
                checkpoint_hash_provided=_looks_like_sha256(checkpoint_sha256),
                checkpoint_file_sha256=followthrough_metadata.get("checkpoint_file_sha256", ""),
                checkpoint_file_hash_present=followthrough_metadata.get("checkpoint_file_hash_present", False),
                recovery_checkpoint_hash_matches_file=followthrough_metadata.get("checkpoint_hash_matches_file", False),
                recovery_artifact_hashes_present=recovery_artifact_hashes_match_files,
                recovery_artifact_hashes_match_files=recovery_artifact_hashes_match_files,
                recovery_followthrough_token_sha256=followthrough_metadata.get("recovery_followthrough_token_sha256", ""),
                recovery_followthrough_token_present=followthrough_metadata.get("recovery_followthrough_token_present", False),
                recovery_followthrough_token_boundary_rows=recovery_followthrough_token_boundary_rows,
                recovery_followthrough_token_boundary_row_count=len(recovery_followthrough_token_boundary_rows),
                recovery_followthrough_token_boundary_ready=recovery_followthrough_token_boundary_ready,
                recovery_followthrough_token_reusable_for_future_recovery=False,
                recovery_followthrough_token_authorizes_resume_gate=False,
                recovery_followthrough_token_authorizes_next_step=False,
                recovery_followthrough_token_authorizes_risky_work=False,
                recovery_followthrough_token_authorizes_approval=False,
                recovery_followthrough_token_authorizes_model_call=False,
                recovery_followthrough_token_authorizes_tool_execution=False,
                recovery_followthrough_token_authorizes_personal_data_read=False,
                recovery_followthrough_token_authorizes_external_side_effect=False,
                next_recovery_followthrough_requires_new_token=True,
                local_safe_recovery_execution_token_sha256=followthrough_metadata.get("local_safe_recovery_execution_token_sha256", ""),
                local_safe_recovery_execution_token_present=followthrough_metadata.get("local_safe_recovery_execution_token_present", False),
                local_safe_recovery_execution_token_authorizes_resume_gate=False,
                local_safe_recovery_execution_token_authorizes_next_step=False,
                local_safe_recovery_execution_token_authorizes_risky_work=False,
                local_safe_recovery_execution_token_authorizes_approval=False,
                local_safe_recovery_execution_token_authorizes_model_call=False,
                local_safe_recovery_execution_token_authorizes_tool_execution=False,
                local_safe_recovery_execution_token_authorizes_personal_data_read=False,
                local_safe_recovery_execution_token_authorizes_external_side_effect=False,
                local_safe_recovery_execution_token_reusable_for_future_recovery=False,
                next_recovery_execution_requires_new_local_safe_token=True,
                local_safe_recovery_execution_token_boundary_rows=local_safe_recovery_execution_token_boundary_rows,
                local_safe_recovery_execution_token_boundary_row_count=len(local_safe_recovery_execution_token_boundary_rows),
                local_safe_recovery_execution_token_boundary_ready=local_safe_recovery_execution_token_boundary_ready,
                recovery_execution_readiness_token_sha256=followthrough_metadata.get("recovery_execution_readiness_token_sha256", ""),
                recovery_execution_readiness_token_present=followthrough_metadata.get("recovery_execution_readiness_token_present", False),
                recovery_execution_readiness_token_boundary_rows=recovery_execution_readiness_token_boundary_rows,
                recovery_execution_readiness_token_boundary_row_count=len(recovery_execution_readiness_token_boundary_rows),
                recovery_execution_readiness_token_boundary_ready=recovery_execution_readiness_token_boundary_ready,
                recovery_execution_readiness_token_authorizes_resume_gate=False,
                recovery_execution_readiness_token_authorizes_next_step=False,
                recovery_execution_readiness_token_authorizes_risky_work=False,
                recovery_execution_readiness_token_authorizes_approval=False,
                recovery_execution_readiness_token_authorizes_model_call=False,
                recovery_execution_readiness_token_authorizes_tool_execution=False,
                recovery_execution_readiness_token_authorizes_personal_data_read=False,
                recovery_execution_readiness_token_authorizes_external_side_effect=False,
                recovery_execution_readiness_token_authorizes_unreviewed_followthrough=False,
                recovery_execution_readiness_token_reusable_for_future_recovery=False,
                next_recovery_execution_requires_new_readiness_token=True,
                carried_recovery_execution_score=carried_recovery_execution_score,
                carried_recovery_execution_max_score=carried_recovery_execution_max_score,
                carried_recovery_execution_scorecard_rows=carried_recovery_execution_scorecard_rows,
                carried_recovery_execution_scorecard_row_count=len(carried_recovery_execution_scorecard_rows),
                carried_recovery_execution_required_rows_ready=carried_recovery_execution_required_rows_ready,
                carried_recovery_execution_scorecard_ready=carried_recovery_execution_scorecard_ready,
                carried_recovery_execution_as_prior_proof=carried_recovery_execution_as_prior_proof,
                carried_recovery_execution_authorizes_action_now=False,
                carried_recovery_execution_authorizes_risky_work=False,
                carried_recovery_execution_authorizes_unreviewed_followthrough=False,
                carried_recovery_execution_authorizes_model_call=False,
                carried_recovery_execution_authorizes_tool_execution=False,
                carried_recovery_execution_authorizes_personal_data_read=False,
                carried_recovery_execution_authorizes_external_side_effect=False,
                checkpoint_freshness=cockpit_metadata.get("checkpoint_freshness"),
                checkpoint_needs_review=cockpit_metadata.get("checkpoint_needs_review", True),
                checkpoint_route_token_sha256=cockpit_metadata.get("checkpoint_route_token_sha256", ""),
                checkpoint_route_token_present=cockpit_metadata.get("checkpoint_route_token_present", False),
                checkpoint_route_boundary_rows=checkpoint_route_boundary_rows,
                checkpoint_route_boundary_row_count=len(checkpoint_route_boundary_rows),
                checkpoint_route_boundary_ready=checkpoint_route_boundary_ready,
                checkpoint_route_authorizes_continuation=False,
                checkpoint_route_authorizes_local_safe_step=False,
                checkpoint_route_authorizes_risky_work=False,
                checkpoint_route_authorizes_approval=False,
                checkpoint_route_authorizes_recovery_followthrough=False,
                checkpoint_route_authorizes_checkpoint_reuse=False,
                checkpoint_route_authorizes_model_call=False,
                checkpoint_route_authorizes_tool_execution=False,
                checkpoint_route_authorizes_personal_data_read=False,
                checkpoint_route_authorizes_external_side_effect=False,
                checkpoint_route_reusable_for_next_review=False,
                checkpoint_route_reusable_for_next_checkpoint=False,
                next_checkpoint_route_requires_fresh_recovery_review=True,
                checkpoint_route_recovery_proof_queue=cockpit_metadata.get("checkpoint_route_recovery_proof_queue", []),
                checkpoint_route_recovery_proof_queue_count=cockpit_metadata.get("checkpoint_route_recovery_proof_queue_count", 0),
                checkpoint_route_next_recovery_command=cockpit_metadata.get("checkpoint_route_next_recovery_command", ""),
                checkpoint_recovery_proof_queue=cockpit_metadata.get("checkpoint_recovery_proof_queue", []),
                checkpoint_recovery_proof_queue_count=cockpit_metadata.get("checkpoint_recovery_proof_queue_count", 0),
                checkpoint_recovery_next_proof_command=cockpit_metadata.get("checkpoint_recovery_next_proof_command", ""),
                followthrough_state=followthrough_metadata.get("followthrough_state"),
                followthrough_closure_ready=followthrough_metadata.get("normal_followthrough_allowed"),
                recovery_closure_missing=followthrough_metadata.get("recovery_closure_missing", []),
                recovery_closure_missing_count=followthrough_metadata.get("recovery_closure_missing_count", 0),
                reviewed_step_provided=bool(reviewed_step),
                verification_provided=bool(verification),
                receipt_path_provided=bool(receipt_path),
                checkpoint_path_provided=bool(checkpoint_path),
                stop_condition_provided=bool(stop_condition),
                approval_reference_provided=bool(approval_reference),
                risky_recovery_signals=followthrough_metadata.get("risky_recovery_signals", []),
                risky_recovery_without_approval=followthrough_metadata.get("risky_recovery_without_approval", False),
                proof_queue=proof_queue,
                proof_queue_count=len(proof_queue),
                next_proof_command=proof_queue[0],
                timebox_output=timebox.output[:1200],
                cockpit_output=cockpit.output[:1600],
                followthrough_output=followthrough.output[:1600],
            ),
        )

    def autonomy_continuation_execution_packet(args: dict[str, Any]) -> ToolResult:
        objective = str(args.get("objective") or args.get("request") or "Continue Jarvis V2 safely.").strip()
        previous_instruction = str(args.get("previous_instruction") or args.get("previous") or "").strip()
        latest_instruction = str(args.get("latest_instruction") or args.get("latest") or args.get("instruction") or objective).strip()
        stop_at = str(args.get("stop_at") or args.get("stop") or args.get("until") or "").strip()
        current_time = str(args.get("current_time") or args.get("now") or "").strip()
        timezone_label = str(args.get("timezone") or "local").strip() or "local"
        reviewed_step = str(args.get("reviewed_step") or args.get("step") or args.get("approved_step") or "").strip()
        verification = str(args.get("verification") or args.get("tests") or args.get("verification_target") or "").strip()
        receipt_path = str(args.get("receipt_path") or args.get("receipt") or "").strip()
        receipt_sha256 = str(args.get("receipt_sha256") or args.get("receipt_hash") or "").strip()
        checkpoint_path = str(args.get("checkpoint_path") or args.get("checkpoint") or "").strip()
        checkpoint_sha256 = str(args.get("checkpoint_sha256") or args.get("checkpoint_hash") or "").strip()
        stop_condition = str(args.get("stop_condition") or args.get("stop_rule") or "").strip()
        blockers_text = str(args.get("blockers") or "none reported").strip()
        approval_reference = str(args.get("approval") or args.get("approval_reference") or "").strip()
        proposed_next_step = str(args.get("next_step") or args.get("next") or "").strip()
        proposed_verification = str(args.get("next_verification") or args.get("post_verification") or "").strip()
        prior_cycle_ledger_token_sha256 = str(
            args.get("prior_cycle_ledger_token_sha256")
            or args.get("previous_cycle_ledger_token_sha256")
            or args.get("cycle_ledger_token_sha256")
            or ""
        ).strip()

        resume_gate = autonomy_resume_gate(
            {
                "objective": objective,
                "previous_instruction": previous_instruction,
                "latest_instruction": latest_instruction,
                "stop_at": stop_at,
                "current_time": current_time,
                "timezone": timezone_label,
                "reviewed_step": reviewed_step,
                "verification": verification,
                "receipt_path": receipt_path,
                "receipt_sha256": receipt_sha256,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": checkpoint_sha256,
                "stop_condition": stop_condition,
                "blockers": blockers_text,
                "approval_reference": approval_reference,
            }
        )
        resume_metadata = resume_gate.metadata
        supersession_token_boundary_rows = list(resume_metadata.get("supersession_token_boundary_rows") or [])
        supersession_token_boundary_ready = _operator_supersession_token_boundary_ready(
            str(resume_metadata.get("supersession_token_sha256") or ""),
            supersession_token_boundary_rows,
        )
        checkpoint_route_boundary_rows = list(resume_metadata.get("checkpoint_route_boundary_rows") or [])
        checkpoint_route_boundary_ready = _checkpoint_route_boundary_ready(
            str(resume_metadata.get("checkpoint_route_token_sha256") or ""),
            checkpoint_route_boundary_rows,
        )
        timebox_review_contract_rows = list(resume_metadata.get("timebox_review_contract_rows") or [])
        timebox_review_contract_ready = bool(
            resume_metadata.get("timebox_review_contract_ready")
            and _operator_timebox_review_contract_ready(
                timebox_review_contract_rows,
                timebox_state=str(resume_metadata.get("timebox_state") or ""),
                stop_at=str(resume_metadata.get("stop_at") or ""),
                current_time=str(resume_metadata.get("current_time") or ""),
                timezone_label=str(resume_metadata.get("timezone") or ""),
            )
        )
        carried_recovery_execution_scorecard_rows = list(
            resume_metadata.get("carried_recovery_execution_scorecard_rows") or []
        )
        carried_recovery_execution_score = int(resume_metadata.get("carried_recovery_execution_score") or 0)
        carried_recovery_execution_max_score = int(resume_metadata.get("carried_recovery_execution_max_score") or 0)
        carried_recovery_execution_required_rows_ready = _recovery_execution_readiness_scorecard_ready(
            carried_recovery_execution_scorecard_rows
        )
        carried_recovery_execution_scorecard_ready = carried_recovery_execution_required_rows_ready
        carried_recovery_execution_as_prior_proof = bool(
            resume_metadata.get("carried_recovery_execution_as_prior_proof")
            and _recovery_execution_readiness_scorecard_ready(carried_recovery_execution_scorecard_rows)
        )
        next_risk_signals = _recovery_risk_signals(" ".join([proposed_next_step, proposed_verification]))
        proposed_next_step_sha256 = _text_sha256(proposed_next_step)
        proposed_verification_sha256 = _text_sha256(proposed_verification)
        continuation_review_token_sha256 = _autonomy_review_token_sha256(
            objective=objective,
            stop_at=stop_at,
            current_time=current_time,
            timezone_label=timezone_label,
            checkpoint_path=str(resume_metadata.get("latest_checkpoint_path") or checkpoint_path),
            checkpoint_sha256=str(resume_metadata.get("checkpoint_sha256") or checkpoint_sha256),
            proposed_step_sha256=proposed_next_step_sha256,
            proposed_verification_sha256=proposed_verification_sha256,
        )
        missing: list[str] = []
        if not resume_metadata.get("normal_autonomous_followthrough_allowed"):
            missing.append("ready autonomy resume gate")
        if not resume_metadata.get("can_continue_under_latest_instruction"):
            missing.append("latest instruction supersession review")
        if not supersession_token_boundary_ready:
            missing.append("operator supersession token boundary")
        if not checkpoint_route_boundary_ready:
            missing.append("checkpoint route boundary")
        if not timebox_review_contract_ready:
            missing.append("non_authorizing_timebox_review_contract")
        if not proposed_next_step:
            missing.append("next local-safe step")
        if not proposed_verification:
            missing.append("post-step verification target")
        if not stop_condition:
            missing.append("stop condition")
        if next_risk_signals:
            missing.append("risky next step needs approval packet")

        ready = not missing
        proposed_next_step_approval_proof_queue = (
            _approval_proof_queue_for_risky_work(
                followup_command="autonomy continuation execution: <reviewed local-safe step after approval proof>"
            )
            if next_risk_signals
            else []
        )
        proposed_next_step_approval_boundary_rows = _approval_boundary_rows_for_risky_work(
            risk_signals=next_risk_signals,
            step_sha256=proposed_next_step_sha256,
            verification_sha256=proposed_verification_sha256,
            proof_queue=proposed_next_step_approval_proof_queue,
            required_field="required_before_risky_next_step",
        )
        proposed_next_step_approval_boundary_token_sha256 = _risky_next_step_approval_boundary_token_sha256(
            proposed_step_sha256=proposed_next_step_sha256,
            proposed_verification_sha256=proposed_verification_sha256,
            risk_signals=next_risk_signals,
            approval_proof_queue=proposed_next_step_approval_proof_queue,
            approval_boundary_rows=proposed_next_step_approval_boundary_rows,
        )
        proposed_next_step_approval_boundary_ready = _next_step_approval_boundary_ready(
            token_sha256=proposed_next_step_approval_boundary_token_sha256,
            proposed_step_sha256=proposed_next_step_sha256,
            proposed_verification_sha256=proposed_verification_sha256,
            approval_proof_queue=proposed_next_step_approval_proof_queue,
            approval_boundary_rows=proposed_next_step_approval_boundary_rows,
        )
        proposed_next_step_approval_boundary_as_prior_proof = _next_step_approval_boundary_as_prior_proof(
            token_sha256=proposed_next_step_approval_boundary_token_sha256,
            proposed_step_sha256=proposed_next_step_sha256,
            proposed_verification_sha256=proposed_verification_sha256,
            approval_proof_queue=proposed_next_step_approval_proof_queue,
            approval_boundary_rows=proposed_next_step_approval_boundary_rows,
        )
        state = "AUTONOMY_CONTINUATION_READY_FOR_ONE_LOCAL_SAFE_STEP" if ready else "AUTONOMY_CONTINUATION_HELD"
        one_step_execution_contract_rows = [
            {
                "item": "ready_resume_gate",
                "required": True,
                "status": "ready" if resume_metadata.get("normal_autonomous_followthrough_allowed") else "held",
                "authorizes_only_one_local_safe_step": bool(resume_metadata.get("normal_autonomous_followthrough_allowed")),
                "authorizes_risky_work": False,
                "reusable_for_next_step": False,
            },
            {
                "item": "exact_next_step",
                "required": True,
                "status": "ready" if proposed_next_step else "held",
                "authorizes_only_one_local_safe_step": bool(proposed_next_step),
                "authorizes_risky_work": False,
                "reusable_for_next_step": False,
            },
            {
                "item": "post_step_verification_target",
                "required": True,
                "status": "ready" if proposed_verification else "held",
                "authorizes_only_one_local_safe_step": bool(proposed_verification),
                "authorizes_risky_work": False,
                "reusable_for_next_step": False,
            },
            {
                "item": "stop_condition",
                "required": True,
                "status": "ready" if stop_condition else "held",
                "authorizes_only_one_local_safe_step": bool(stop_condition),
                "authorizes_risky_work": False,
                "reusable_for_next_step": False,
            },
            {
                "item": "risky_work_approval_boundary",
                "required": True,
                "status": "held" if next_risk_signals else "ready",
                "authorizes_only_one_local_safe_step": not bool(next_risk_signals),
                "authorizes_risky_work": False,
                "reusable_for_next_step": False,
            },
            {
                "item": "fresh_post_step_closure",
                "required": True,
                "status": "fresh_required_after_step",
                "authorizes_only_one_local_safe_step": ready,
                "authorizes_risky_work": False,
                "reusable_for_next_step": False,
            },
        ]
        one_step_execution_contract_summary = [
            "one_step_only",
            "local_safe_only",
            "risky_work_approval_gated",
            "fresh_post_step_closure_required",
            "not_reusable_for_next_step",
        ]
        post_step_queue = _autonomy_expected_post_step_proof_queue(
            proposed_next_step=proposed_next_step,
            proposed_verification=proposed_verification,
        )
        one_step_execution_contract_token_sha256 = _one_step_execution_contract_token_sha256(
            objective=objective,
            continuation_state=state,
            resume_gate_state=str(resume_metadata.get("resume_gate_state") or ""),
            timebox_receipt_sha256=str(resume_metadata.get("timebox_receipt_sha256") or ""),
            awake_guard_token_sha256=str(resume_metadata.get("awake_guard_token_sha256") or ""),
            supersession_token_sha256=str(resume_metadata.get("supersession_token_sha256") or ""),
            checkpoint_route_token_sha256=str(resume_metadata.get("checkpoint_route_token_sha256") or ""),
            recovery_followthrough_token_sha256=str(resume_metadata.get("recovery_followthrough_token_sha256") or ""),
            local_safe_recovery_execution_token_sha256=str(resume_metadata.get("local_safe_recovery_execution_token_sha256") or ""),
            prior_cycle_ledger_token_sha256=prior_cycle_ledger_token_sha256,
            continuation_review_token_sha256=continuation_review_token_sha256,
            proposed_next_step_sha256=proposed_next_step_sha256,
            proposed_verification_sha256=proposed_verification_sha256,
            risk_signals=next_risk_signals,
            one_step_execution_contract_rows=one_step_execution_contract_rows,
            timebox_review_contract_rows=timebox_review_contract_rows,
            post_step_proof_queue=post_step_queue,
        )
        one_step_execution_contract_ready = _one_step_execution_contract_ready(
            objective=objective,
            continuation_state=state,
            resume_gate_state=str(resume_metadata.get("resume_gate_state") or ""),
            continuation_ready=ready,
            one_step_execution_contract_rows=one_step_execution_contract_rows,
            one_step_execution_contract_token_sha256=one_step_execution_contract_token_sha256,
            timebox_receipt_sha256=str(resume_metadata.get("timebox_receipt_sha256") or ""),
            awake_guard_token_sha256=str(resume_metadata.get("awake_guard_token_sha256") or ""),
            supersession_token_sha256=str(resume_metadata.get("supersession_token_sha256") or ""),
            checkpoint_route_token_sha256=str(resume_metadata.get("checkpoint_route_token_sha256") or ""),
            recovery_followthrough_token_sha256=str(resume_metadata.get("recovery_followthrough_token_sha256") or ""),
            local_safe_recovery_execution_token_sha256=str(
                resume_metadata.get("local_safe_recovery_execution_token_sha256") or ""
            ),
            prior_cycle_ledger_token_sha256=prior_cycle_ledger_token_sha256,
            continuation_review_token_sha256=continuation_review_token_sha256,
            proposed_next_step_sha256=proposed_next_step_sha256,
            proposed_verification_sha256=proposed_verification_sha256,
            risk_signals=next_risk_signals,
            timebox_review_contract_ready=timebox_review_contract_ready,
            timebox_review_contract_rows=timebox_review_contract_rows,
            post_step_proof_queue=post_step_queue,
        )
        next_safe_command = post_step_queue[0] if ready else (
            "approval readiness <id>" if next_risk_signals else "autonomy resume gate: stop_at=<ISO> current_time=<ISO> step=<reviewed local-safe step> verification=<evidence> receipt=<path> receipt_sha256=<hash> checkpoint=<path> checkpoint_sha256=<hash> stop_condition=<condition>"
        )
        continuation_scorecard_rows = _autonomy_continuation_readiness_scorecard_rows(
            resume_ready=bool(resume_metadata.get("normal_autonomous_followthrough_allowed")),
            proposed_next_step=proposed_next_step,
            proposed_verification=proposed_verification,
            stop_condition=stop_condition,
            next_risk_signals=next_risk_signals,
            continuation_review_token_sha256=continuation_review_token_sha256,
            prior_cycle_ledger_token_sha256=prior_cycle_ledger_token_sha256,
            one_step_contract_ready=one_step_execution_contract_ready,
            post_step_queue=post_step_queue,
        )
        continuation_score = sum(int(row["points"]) for row in continuation_scorecard_rows)
        continuation_max_score = sum(int(row["max_points"]) for row in continuation_scorecard_rows)
        continuation_required_rows_ready = _scorecard_required_rows_ready(continuation_scorecard_rows)
        continuation_scorecard_ready = _autonomy_continuation_readiness_scorecard_ready(
            continuation_scorecard_rows
        )
        local_safe_recovery_execution_token_boundary_rows = _local_safe_recovery_execution_token_boundary_rows(
            token_sha256=str(resume_metadata.get("local_safe_recovery_execution_token_sha256") or ""),
            source="autonomy_continuation_execution",
        )
        local_safe_recovery_execution_token_boundary_ready = _local_safe_recovery_execution_token_boundary_ready(
            str(resume_metadata.get("local_safe_recovery_execution_token_sha256") or ""),
            local_safe_recovery_execution_token_boundary_rows,
            expected_source="autonomy_continuation_execution",
        )
        recovery_followthrough_token_boundary_rows = list(
            resume_metadata.get("recovery_followthrough_token_boundary_rows") or []
        )
        recovery_followthrough_token_boundary_ready = _carried_recovery_followthrough_token_boundary_ready(
            resume_metadata,
            expected_source="checkpoint_recovery_followthrough",
        )
        recovery_execution_readiness_token_boundary_rows = list(
            resume_metadata.get("recovery_execution_readiness_token_boundary_rows") or []
        )
        recovery_execution_readiness_token_boundary_ready = _carried_recovery_execution_readiness_token_boundary_ready(
            resume_metadata,
            expected_source="checkpoint_recovery_followthrough",
        )
        prior_cycle_ledger_token_boundary_rows = _prior_cycle_ledger_token_boundary_rows(
            token_sha256=prior_cycle_ledger_token_sha256,
            source="autonomy_continuation_execution",
        )
        prior_cycle_ledger_token_boundary_ready = _prior_cycle_ledger_token_boundary_ready(
            prior_cycle_ledger_token_sha256,
            prior_cycle_ledger_token_boundary_rows,
        )
        lines = [
            "Jarvis autonomy continuation execution packet:",
            "This is read-only. It is the last steering packet before one normal local-safe continuation step can run after checkpoint recovery.",
            "",
            "Objective:",
            f"- {objective[:300]}",
            "",
            "Continuation state:",
            f"- state: {state}",
            f"- one local-safe step allowed: {'yes' if ready else 'no'}",
            f"- missing blockers: {', '.join(missing) if missing else 'none'}",
            f"- next safe command: `{next_safe_command}`",
            "",
            "Measured continuation readiness scorecard:",
            f"- score: {continuation_score}/{continuation_max_score}",
            f"- required rows ready: {'yes' if continuation_required_rows_ready else 'no'}",
            f"- scorecard ready: {'yes' if continuation_scorecard_ready else 'no'}",
            *[
                f"- {row['item']}: {row['points']}/{row['max_points']} ({'ready' if row['ready'] else 'held'}; does not authorize risky work, batching, or follow-up without closure)"
                for row in continuation_scorecard_rows
            ],
            "",
            "Bound resume proof:",
            f"- resume gate state: {resume_metadata.get('resume_gate_state')}",
            f"- timebox state: {resume_metadata.get('timebox_state')}",
            f"- timebox receipt sha256: {resume_metadata.get('timebox_receipt_sha256') or 'missing'}",
            f"- timebox review contract ready: {'yes' if timebox_review_contract_ready else 'no'}",
            f"- timebox review rows: {len(timebox_review_contract_rows)}",
            f"- cockpit state: {resume_metadata.get('cockpit_state')}",
            f"- follow-through state: {resume_metadata.get('followthrough_state')}",
            f"- normal autonomous follow-through allowed: {'yes' if resume_metadata.get('normal_autonomous_followthrough_allowed') else 'no'}",
            f"- operator supersession state: {resume_metadata.get('supersession_state')}",
            f"- latest instruction can govern continuation: {'yes' if resume_metadata.get('can_continue_under_latest_instruction') else 'no'}",
            f"- supersession token sha256: {resume_metadata.get('supersession_token_sha256') or 'missing'}",
            f"- supersession token boundary rows: {len(supersession_token_boundary_rows)}",
            "- supersession token authorizes local-safe step: no",
            "- supersession token authorizes risky work: no",
            "- supersession token authorizes approval: no",
            "- supersession token reusable for next review: no",
            f"- latest checkpoint: {resume_metadata.get('latest_checkpoint_display') or 'missing'}",
            f"- latest checkpoint sha256: {resume_metadata.get('latest_checkpoint_sha256') or 'missing'}",
            f"- supplied checkpoint matches latest: {'yes' if resume_metadata.get('checkpoint_path_matches_latest') else 'no'}",
            f"- supplied checkpoint hash matches latest: {'yes' if resume_metadata.get('checkpoint_hash_matches_latest') else 'no'}",
            f"- recovery receipt hash matches file: {'yes' if resume_metadata.get('receipt_hash_matches_file') else 'no'}",
            f"- recovery checkpoint hash matches file: {'yes' if resume_metadata.get('recovery_checkpoint_hash_matches_file') else 'no'}",
            f"- artifact hashes match files: {'yes' if resume_metadata.get('recovery_artifact_hashes_match_files') else 'no'}",
            f"- recovery follow-through token sha256: {resume_metadata.get('recovery_followthrough_token_sha256') or 'missing'}",
            "- recovery follow-through token reusable for future recovery: no",
            f"- local-safe recovery execution token sha256: {resume_metadata.get('local_safe_recovery_execution_token_sha256') or 'missing'}",
            "- local-safe recovery execution token authorizes resume gate: no",
            "- local-safe recovery execution token authorizes next step: no",
            "- local-safe recovery execution token authorizes risky work: no",
            "- local-safe recovery execution token authorizes approval: no",
            "- local-safe recovery execution token authorizes model call: no",
            "- local-safe recovery execution token authorizes tool execution: no",
            "- local-safe recovery execution token authorizes personal-data read: no",
            "- local-safe recovery execution token authorizes external side effect: no",
            "- local-safe recovery execution token reusable for future recovery: no",
            f"- local-safe recovery execution token boundary rows: {len(local_safe_recovery_execution_token_boundary_rows)}",
            f"- recovery execution readiness token sha256: {resume_metadata.get('recovery_execution_readiness_token_sha256') or 'missing'}",
            "- recovery execution readiness token authorizes resume gate: no",
            "- recovery execution readiness token authorizes next step: no",
            "- recovery execution readiness token authorizes risky work: no",
            "- recovery execution readiness token authorizes approval: no",
            "- recovery execution readiness token authorizes model call: no",
            "- recovery execution readiness token authorizes tool execution: no",
            "- recovery execution readiness token authorizes personal-data read: no",
            "- recovery execution readiness token authorizes external side effect: no",
            "- recovery execution readiness token authorizes unreviewed follow-through: no",
            "- recovery execution readiness token reusable for future recovery: no",
            f"- recovery execution readiness token boundary rows: {resume_metadata.get('recovery_execution_readiness_token_boundary_row_count') or 0}",
            f"- resume blockers: {', '.join(resume_metadata.get('blockers') or []) if resume_metadata.get('blockers') else 'none'}",
            "",
            "Carried timebox review contract:",
            "- authorizes execution now: no",
            "- authorizes local-safe step: no",
            "- authorizes risky work: no",
            "- authorizes approval: no",
            "- reusable for next step: no",
            *[
                f"- {row['item']}: fresh required yes; prior reusable no; authorizes timebox reuse no"
                for row in timebox_review_contract_rows
            ],
            "",
            "Carried awake guard boundary:",
            f"- awake requested: {'yes' if resume_metadata.get('awake_guard_requested') else 'no'}",
            f"- awake guard token sha256: {resume_metadata.get('awake_guard_token_sha256') or 'missing'}",
            f"- boundary rows: {resume_metadata.get('awake_guard_boundary_row_count') or 0}",
            "- authorizes OS wake lock: no",
            "- authorizes shell execution: no",
            "- authorizes computer control: no",
            "- authorizes approval: no",
            "- reusable for next timebox: no",
            "- fresh awake-guard review required next timebox: yes",
            "",
            "Carried operator supersession boundary:",
            f"- boundary ready: {'yes' if supersession_token_boundary_ready else 'no'}",
            f"- boundary rows: {len(supersession_token_boundary_rows)}",
            *[
                f"- {row['item']}: {row['status']}; authorizes execution no; local-safe step no; risky work no; approval no; recovery follow-through no; reusable no"
                for row in supersession_token_boundary_rows
            ],
            "- next supersession requires fresh latest-instruction review: yes",
            "",
            "Carried recovery execution readiness proof:",
            f"- score: {carried_recovery_execution_score}/{carried_recovery_execution_max_score}",
            f"- required rows ready: {'yes' if carried_recovery_execution_required_rows_ready else 'no'}",
            f"- carried as prior proof: {'yes' if carried_recovery_execution_as_prior_proof else 'no'}",
            "- authorizes action now: no",
            "- authorizes risky work: no",
            "- authorizes unreviewed follow-through: no",
            *[
                f"- {row['item']}: {row['points']}/{row['max_points']} ({'ready' if row['ready'] else 'held'}; prior recovery proof only)"
                for row in carried_recovery_execution_scorecard_rows
            ],
            "",
            "Prior cycle boundary:",
            f"- prior cycle ledger token sha256: {prior_cycle_ledger_token_sha256 or 'not supplied'}",
            f"- prior cycle ledger token present: {'yes' if _looks_like_sha256(prior_cycle_ledger_token_sha256) else 'no'}",
            f"- prior cycle ledger boundary rows: {len(prior_cycle_ledger_token_boundary_rows)}",
            f"- prior cycle ledger boundary ready: {'yes' if prior_cycle_ledger_token_boundary_ready else 'no'}",
            *[
                f"- {row['item']}: {row['status']}; authorizes action now no; authorizes local-safe step no; authorizes risky work no; authorizes model/tool/private/external no; reusable no"
                for row in prior_cycle_ledger_token_boundary_rows
            ],
            "- prior cycle ledger token reusable for this review: no",
            "- fresh cycle ledger token required after this step: yes",
            "- prior cycle ledger proof authorizes action now: no",
            "",
            "Proposed one-step continuation:",
            f"- next step: {proposed_next_step[:500] or 'missing'}",
            f"- next step sha256: {proposed_next_step_sha256 or 'missing'}",
            f"- verification target: {proposed_verification[:500] or 'missing'}",
            f"- verification target sha256: {proposed_verification_sha256 or 'missing'}",
            f"- continuation review token sha256: {continuation_review_token_sha256 or 'missing'}",
            f"- risk signals: {', '.join(next_risk_signals) if next_risk_signals else 'none'}",
            f"- stop condition: {stop_condition or 'missing'}",
            "",
            "Risky next-step approval proof queue:",
            f"- approval required before review: {'yes' if next_risk_signals else 'no'}",
            f"- boundary token sha256: {proposed_next_step_approval_boundary_token_sha256}",
            f"- boundary ready: {'yes' if proposed_next_step_approval_boundary_ready else 'no'}",
            f"- next approval proof command: `{proposed_next_step_approval_proof_queue[0] if proposed_next_step_approval_proof_queue else 'none'}`",
            *[f"- `{command}`" for command in proposed_next_step_approval_proof_queue],
            *[
                f"- {row['item']}: {row['status']}; authorizes action now no; risky work authorized no; unreviewed follow-through no"
                for row in proposed_next_step_approval_boundary_rows
            ],
            "",
            "One-step execution contract:",
            f"- contract ready: {'yes' if one_step_execution_contract_ready else 'no'}",
            f"- row count: {len(one_step_execution_contract_rows)}",
            f"- contract token sha256: {one_step_execution_contract_token_sha256 or 'missing'}",
            "- contract token authorizes action now: no",
            "- contract token authorizes risky work: no",
            "- contract token reusable for next step: no",
            "- summary: " + ", ".join(one_step_execution_contract_summary),
            *[
                f"- {row['item']}: {row['status']}; one local-safe step only; risky work authorized no; reusable for next step no"
                for row in one_step_execution_contract_rows
            ],
            "",
            "Post-step proof queue:",
            *[f"- `{command}`" for command in post_step_queue],
            "",
            "Approval boundary:",
            "- If the next step touches shell/code, computer control, personal data, external side effects, destructive actions, or other risky work, stop for approval readiness, a last-look approval packet, approval chain proof, and explicit approval.",
            "",
            "Boundary:",
            "- read-only execution packet; it does not execute the next step, run shell/code, control the computer, read private data, write notes, approve requests, rerun actions, or queue approvals.",
        ]
        metadata = _safe_metadata(
                objective=objective,
                stop_at=stop_at,
                current_time=current_time,
                timezone=timezone_label,
                objective_length=len(objective),
                continuation_state=state,
                one_local_safe_step_allowed=ready,
                normal_autonomous_followthrough_allowed=resume_metadata.get("normal_autonomous_followthrough_allowed"),
                missing_blockers=missing,
                missing_blocker_count=len(missing),
                next_safe_command=next_safe_command,
                continuation_readiness_score=continuation_score,
                continuation_readiness_max_score=continuation_max_score,
                continuation_readiness_scorecard_rows=continuation_scorecard_rows,
                continuation_readiness_scorecard_row_count=len(continuation_scorecard_rows),
                continuation_readiness_required_rows_ready=continuation_required_rows_ready,
                continuation_readiness_scorecard_ready=continuation_scorecard_ready,
                resume_gate_state=resume_metadata.get("resume_gate_state"),
                timebox_state=resume_metadata.get("timebox_state"),
                can_continue_now=resume_metadata.get("can_continue_now"),
                should_stop_now=resume_metadata.get("should_stop_now"),
                missing_timebox_proof=resume_metadata.get("missing_timebox_proof", []),
                missing_timebox_proof_count=resume_metadata.get("missing_timebox_proof_count", 0),
                timebox_receipt_sha256=resume_metadata.get("timebox_receipt_sha256", ""),
                timebox_receipt_present=resume_metadata.get("timebox_receipt_present", False),
                timebox_review_contract_rows=timebox_review_contract_rows,
                timebox_review_contract_row_count=len(timebox_review_contract_rows),
                timebox_review_contract_ready=timebox_review_contract_ready,
                timebox_review_contract_summary=resume_metadata.get("timebox_review_contract_summary", []),
                timebox_authorizes_execution=False,
                timebox_authorizes_local_safe_step=False,
                timebox_authorizes_risky_work=False,
                timebox_authorizes_approval=False,
                timebox_authorizes_timebox_reuse=False,
                timebox_authorizes_model_call=False,
                timebox_authorizes_tool_execution=False,
                timebox_authorizes_personal_data_read=False,
                timebox_authorizes_external_side_effect=False,
                timebox_reusable_for_next_step=False,
                next_step_requires_fresh_timebox=True,
                awake_guard_requested=resume_metadata.get("awake_guard_requested", False),
                awake_guard_token_sha256=resume_metadata.get("awake_guard_token_sha256", ""),
                awake_guard_token_present=resume_metadata.get("awake_guard_token_present", False),
                awake_guard_boundary_rows=resume_metadata.get("awake_guard_boundary_rows", []),
                awake_guard_boundary_row_count=resume_metadata.get("awake_guard_boundary_row_count", 0),
                awake_guard_authorizes_os_wake_lock=False,
                awake_guard_authorizes_shell_execution=False,
                awake_guard_authorizes_computer_control=False,
                awake_guard_authorizes_approval=False,
                awake_guard_authorizes_model_call=False,
                awake_guard_authorizes_tool_execution=False,
                awake_guard_authorizes_personal_data_read=False,
                awake_guard_authorizes_external_side_effect=False,
                awake_guard_reusable_for_next_timebox=False,
                awake_guard_requires_separate_operator_request=True,
                awake_guard_requires_separate_shell_approval=True,
                awake_guard_os_wake_lock_boundary_ready=_awake_guard_boundary_ready_from_metadata(
                    resume_metadata,
                    default_objective=objective,
                    default_timezone=timezone_label,
                ),
                awake_guard_caffeinate_command_authorized=False,
                awake_guard_keep_awake_command_authorized=False,
                awake_guard_authorizes_unattended_execution=False,
                awake_guard_authorizes_continuation_window=False,
                awake_guard_reusable_as_execution_permission=False,
                next_awake_guard_requires_fresh_review=True,
                previous_instruction=resume_metadata.get("previous_instruction", ""),
                latest_instruction=resume_metadata.get("latest_instruction", ""),
                latest_instruction_present=resume_metadata.get("latest_instruction_present", False),
                latest_instruction_supersedes_previous=resume_metadata.get("latest_instruction_supersedes_previous", False),
                latest_instruction_is_stop=resume_metadata.get("latest_instruction_is_stop", False),
                latest_instruction_is_continue=resume_metadata.get("latest_instruction_is_continue", False),
                supersession_state=resume_metadata.get("supersession_state", ""),
                can_continue_under_latest_instruction=resume_metadata.get("can_continue_under_latest_instruction", False),
                newest_instruction_overrides_automation=resume_metadata.get("newest_instruction_overrides_automation", True),
                newest_instruction_overrides_goal=resume_metadata.get("newest_instruction_overrides_goal", True),
                newest_instruction_overrides_recovery_queue=resume_metadata.get("newest_instruction_overrides_recovery_queue", True),
                stop_or_pause_blocks_autonomy=resume_metadata.get("stop_or_pause_blocks_autonomy", False),
                supersession_token_sha256=resume_metadata.get("supersession_token_sha256", ""),
                supersession_token_present=resume_metadata.get("supersession_token_present", False),
                supersession_token_boundary_rows=supersession_token_boundary_rows,
                supersession_token_boundary_row_count=len(supersession_token_boundary_rows),
                supersession_token_boundary_ready=supersession_token_boundary_ready,
                supersession_token_authorizes_execution=False,
                supersession_token_authorizes_local_safe_step=False,
                supersession_token_authorizes_risky_work=False,
                supersession_token_authorizes_approval=False,
                supersession_token_authorizes_recovery_followthrough=False,
                supersession_token_authorizes_timebox_override=False,
                supersession_token_authorizes_goal_override=False,
                supersession_token_authorizes_model_call=False,
                supersession_token_authorizes_tool_execution=False,
                supersession_token_authorizes_personal_data_read=False,
                supersession_token_authorizes_external_side_effect=False,
                supersession_token_reusable_for_next_review=False,
                supersession_token_reusable_for_next_timebox=False,
                next_supersession_requires_fresh_latest_instruction_review=True,
                cockpit_state=resume_metadata.get("cockpit_state"),
                followthrough_state=resume_metadata.get("followthrough_state"),
                latest_checkpoint_path=resume_metadata.get("latest_checkpoint_path", ""),
                latest_checkpoint_display=resume_metadata.get("latest_checkpoint_display", ""),
                latest_checkpoint_sha256=resume_metadata.get("latest_checkpoint_sha256", ""),
                latest_checkpoint_hash_present=resume_metadata.get("latest_checkpoint_hash_present", False),
                supplied_checkpoint_path=resume_metadata.get("supplied_checkpoint_path", ""),
                checkpoint_path_matches_latest=resume_metadata.get("checkpoint_path_matches_latest"),
                checkpoint_hash_matches_latest=resume_metadata.get("checkpoint_hash_matches_latest"),
                checkpoint_freshness=resume_metadata.get("checkpoint_freshness"),
                checkpoint_needs_review=resume_metadata.get("checkpoint_needs_review", True),
                checkpoint_route_token_sha256=resume_metadata.get("checkpoint_route_token_sha256", ""),
                checkpoint_route_token_present=resume_metadata.get("checkpoint_route_token_present", False),
                checkpoint_route_boundary_rows=checkpoint_route_boundary_rows,
                checkpoint_route_boundary_row_count=len(checkpoint_route_boundary_rows),
                checkpoint_route_boundary_ready=checkpoint_route_boundary_ready,
                checkpoint_route_authorizes_continuation=False,
                checkpoint_route_authorizes_local_safe_step=False,
                checkpoint_route_authorizes_risky_work=False,
                checkpoint_route_authorizes_approval=False,
                checkpoint_route_authorizes_recovery_followthrough=False,
                checkpoint_route_authorizes_checkpoint_reuse=False,
                checkpoint_route_authorizes_model_call=False,
                checkpoint_route_authorizes_tool_execution=False,
                checkpoint_route_authorizes_personal_data_read=False,
                checkpoint_route_authorizes_external_side_effect=False,
                checkpoint_route_reusable_for_next_review=False,
                checkpoint_route_reusable_for_next_checkpoint=False,
                next_checkpoint_route_requires_fresh_recovery_review=True,
                checkpoint_route_recovery_proof_queue=resume_metadata.get("checkpoint_route_recovery_proof_queue", []),
                checkpoint_route_recovery_proof_queue_count=resume_metadata.get("checkpoint_route_recovery_proof_queue_count", 0),
                checkpoint_route_next_recovery_command=resume_metadata.get("checkpoint_route_next_recovery_command", ""),
                checkpoint_recovery_proof_queue=resume_metadata.get("checkpoint_recovery_proof_queue", []),
                checkpoint_recovery_proof_queue_count=resume_metadata.get("checkpoint_recovery_proof_queue_count", 0),
                checkpoint_recovery_next_proof_command=resume_metadata.get("checkpoint_recovery_next_proof_command", ""),
                receipt_sha256=resume_metadata.get("receipt_sha256", ""),
                receipt_hash_provided=resume_metadata.get("receipt_hash_provided", False),
                receipt_file_sha256=resume_metadata.get("receipt_file_sha256", ""),
                receipt_file_hash_present=resume_metadata.get("receipt_file_hash_present", False),
                receipt_hash_matches_file=resume_metadata.get("receipt_hash_matches_file", False),
                checkpoint_sha256=resume_metadata.get("checkpoint_sha256", ""),
                checkpoint_hash_provided=resume_metadata.get("checkpoint_hash_provided", False),
                checkpoint_file_sha256=resume_metadata.get("checkpoint_file_sha256", ""),
                checkpoint_file_hash_present=resume_metadata.get("checkpoint_file_hash_present", False),
                recovery_checkpoint_hash_matches_file=resume_metadata.get("recovery_checkpoint_hash_matches_file", False),
                recovery_artifact_hashes_present=resume_metadata.get("recovery_artifact_hashes_present", False),
                recovery_artifact_hashes_match_files=resume_metadata.get("recovery_artifact_hashes_match_files", False),
                recovery_followthrough_token_sha256=resume_metadata.get("recovery_followthrough_token_sha256", ""),
                recovery_followthrough_token_present=resume_metadata.get("recovery_followthrough_token_present", False),
                recovery_followthrough_token_boundary_rows=recovery_followthrough_token_boundary_rows,
                recovery_followthrough_token_boundary_row_count=len(recovery_followthrough_token_boundary_rows),
                recovery_followthrough_token_boundary_ready=recovery_followthrough_token_boundary_ready,
                recovery_followthrough_token_reusable_for_future_recovery=False,
                recovery_followthrough_token_authorizes_resume_gate=False,
                recovery_followthrough_token_authorizes_next_step=False,
                recovery_followthrough_token_authorizes_risky_work=False,
                recovery_followthrough_token_authorizes_approval=False,
                recovery_followthrough_token_authorizes_model_call=False,
                recovery_followthrough_token_authorizes_tool_execution=False,
                recovery_followthrough_token_authorizes_personal_data_read=False,
                recovery_followthrough_token_authorizes_external_side_effect=False,
                next_recovery_followthrough_requires_new_token=True,
                local_safe_recovery_execution_token_sha256=resume_metadata.get("local_safe_recovery_execution_token_sha256", ""),
                local_safe_recovery_execution_token_present=resume_metadata.get("local_safe_recovery_execution_token_present", False),
                local_safe_recovery_execution_token_authorizes_resume_gate=False,
                local_safe_recovery_execution_token_authorizes_next_step=False,
                local_safe_recovery_execution_token_authorizes_risky_work=False,
                local_safe_recovery_execution_token_authorizes_approval=False,
                local_safe_recovery_execution_token_authorizes_model_call=False,
                local_safe_recovery_execution_token_authorizes_tool_execution=False,
                local_safe_recovery_execution_token_authorizes_personal_data_read=False,
                local_safe_recovery_execution_token_authorizes_external_side_effect=False,
                local_safe_recovery_execution_token_reusable_for_future_recovery=False,
                next_recovery_execution_requires_new_local_safe_token=True,
                local_safe_recovery_execution_token_boundary_rows=local_safe_recovery_execution_token_boundary_rows,
                local_safe_recovery_execution_token_boundary_row_count=len(local_safe_recovery_execution_token_boundary_rows),
                local_safe_recovery_execution_token_boundary_ready=local_safe_recovery_execution_token_boundary_ready,
                recovery_execution_readiness_token_sha256=resume_metadata.get("recovery_execution_readiness_token_sha256", ""),
                recovery_execution_readiness_token_present=resume_metadata.get("recovery_execution_readiness_token_present", False),
                recovery_execution_readiness_token_boundary_rows=recovery_execution_readiness_token_boundary_rows,
                recovery_execution_readiness_token_boundary_row_count=len(recovery_execution_readiness_token_boundary_rows),
                recovery_execution_readiness_token_boundary_ready=recovery_execution_readiness_token_boundary_ready,
                recovery_execution_readiness_token_authorizes_resume_gate=False,
                recovery_execution_readiness_token_authorizes_next_step=False,
                recovery_execution_readiness_token_authorizes_risky_work=False,
                recovery_execution_readiness_token_authorizes_approval=False,
                recovery_execution_readiness_token_authorizes_model_call=False,
                recovery_execution_readiness_token_authorizes_tool_execution=False,
                recovery_execution_readiness_token_authorizes_personal_data_read=False,
                recovery_execution_readiness_token_authorizes_external_side_effect=False,
                recovery_execution_readiness_token_authorizes_unreviewed_followthrough=False,
                recovery_execution_readiness_token_reusable_for_future_recovery=False,
                next_recovery_execution_requires_new_readiness_token=True,
                carried_recovery_execution_score=carried_recovery_execution_score,
                carried_recovery_execution_max_score=carried_recovery_execution_max_score,
                carried_recovery_execution_scorecard_rows=carried_recovery_execution_scorecard_rows,
                carried_recovery_execution_scorecard_row_count=len(carried_recovery_execution_scorecard_rows),
                carried_recovery_execution_required_rows_ready=carried_recovery_execution_required_rows_ready,
                carried_recovery_execution_scorecard_ready=carried_recovery_execution_scorecard_ready,
                carried_recovery_execution_as_prior_proof=carried_recovery_execution_as_prior_proof,
                carried_recovery_execution_authorizes_action_now=False,
                carried_recovery_execution_authorizes_risky_work=False,
                carried_recovery_execution_authorizes_unreviewed_followthrough=False,
                carried_recovery_execution_authorizes_model_call=False,
                carried_recovery_execution_authorizes_tool_execution=False,
                carried_recovery_execution_authorizes_personal_data_read=False,
                carried_recovery_execution_authorizes_external_side_effect=False,
                prior_cycle_ledger_token_sha256=prior_cycle_ledger_token_sha256,
                prior_cycle_ledger_token_present=_looks_like_sha256(prior_cycle_ledger_token_sha256),
                prior_cycle_ledger_token_boundary_rows=prior_cycle_ledger_token_boundary_rows,
                prior_cycle_ledger_token_boundary_row_count=len(prior_cycle_ledger_token_boundary_rows),
                prior_cycle_ledger_token_boundary_ready=prior_cycle_ledger_token_boundary_ready,
                prior_cycle_ledger_token_reusable_for_this_review=False,
                prior_cycle_ledger_token_reusable_for_this_closure=False,
                prior_cycle_ledger_token_reusable_for_this_cycle=False,
                prior_cycle_ledger_proof_authorizes_action_now=False,
                prior_cycle_ledger_proof_authorizes_post_step_closure=False,
                prior_cycle_ledger_proof_authorizes_new_action=False,
                prior_cycle_ledger_proof_authorizes_model_call=False,
                prior_cycle_ledger_proof_authorizes_tool_execution=False,
                prior_cycle_ledger_proof_authorizes_personal_data_read=False,
                prior_cycle_ledger_proof_authorizes_external_side_effect=False,
                next_step_requires_fresh_cycle_ledger_token=True,
                proposed_next_step=proposed_next_step[:500],
                proposed_next_step_provided=bool(proposed_next_step),
                proposed_next_step_sha256=proposed_next_step_sha256,
                proposed_verification=proposed_verification[:500],
                proposed_verification_provided=bool(proposed_verification),
                proposed_verification_sha256=proposed_verification_sha256,
                continuation_review_token_sha256=continuation_review_token_sha256,
                continuation_review_token_present=_looks_like_sha256(continuation_review_token_sha256),
                continuation_review_token_reusable_for_next_review=False,
                previous_continuation_review_token_reusable_for_next_review=False,
                next_review_requires_new_continuation_review_token=True,
                one_step_execution_contract_rows=one_step_execution_contract_rows,
                one_step_execution_contract_row_count=len(one_step_execution_contract_rows),
                one_step_execution_contract_ready=one_step_execution_contract_ready,
                one_step_execution_contract_summary=one_step_execution_contract_summary,
                one_step_execution_contract_token_sha256=one_step_execution_contract_token_sha256,
                one_step_execution_contract_token_present=_looks_like_sha256(one_step_execution_contract_token_sha256),
                one_step_execution_contract_token_as_prior_proof=False,
                one_step_execution_contract_binds_awake_guard=True,
                one_step_execution_contract_binds_operator_supersession=True,
                one_step_execution_contract_binds_timebox_review_contract=True,
                one_step_execution_contract_token_authorizes_action_now=False,
                one_step_execution_contract_token_authorizes_risky_work=False,
                one_step_execution_contract_token_authorizes_unreviewed_followthrough=False,
                one_step_execution_contract_token_authorizes_batching=False,
                one_step_execution_contract_token_reusable_for_next_step=False,
                next_step_requires_new_one_step_execution_contract_token=True,
                one_step_execution_contract_all_non_reusable=all(
                    row["reusable_for_next_step"] is False for row in one_step_execution_contract_rows
                ),
                one_step_execution_contract_all_risky_work_gated=all(
                    row["authorizes_risky_work"] is False for row in one_step_execution_contract_rows
                ),
                one_step_execution_contract_all_local_safe_step_limited=all(
                    row["authorizes_only_one_local_safe_step"] is True
                    and row["authorizes_risky_work"] is False
                    and row["reusable_for_next_step"] is False
                    for row in one_step_execution_contract_rows
                ),
                one_step_execution_contract_requires_fresh_closure=True,
                one_step_execution_contract_authorizes_batching=False,
                one_step_execution_contract_authorizes_followup_without_closure=False,
                proposed_next_step_risk_signals=next_risk_signals,
                proposed_next_step_risk_signal_count=len(next_risk_signals),
                proposed_next_step_requires_approval=bool(next_risk_signals),
                proposed_next_step_approval_proof_queue=proposed_next_step_approval_proof_queue,
                proposed_next_step_approval_proof_queue_count=len(proposed_next_step_approval_proof_queue),
                proposed_next_step_next_approval_proof_command=(
                    proposed_next_step_approval_proof_queue[0] if proposed_next_step_approval_proof_queue else ""
                ),
                proposed_next_step_approval_required_before_review=bool(next_risk_signals),
                proposed_next_step_approval_boundary_rows=proposed_next_step_approval_boundary_rows,
                proposed_next_step_approval_boundary_row_count=len(proposed_next_step_approval_boundary_rows),
                proposed_next_step_approval_boundary_ready=proposed_next_step_approval_boundary_ready,
                proposed_next_step_approval_boundary_token_sha256=proposed_next_step_approval_boundary_token_sha256,
                proposed_next_step_approval_boundary_token_present=_looks_like_sha256(proposed_next_step_approval_boundary_token_sha256),
                proposed_next_step_approval_boundary_as_prior_proof=proposed_next_step_approval_boundary_as_prior_proof,
                proposed_next_step_approval_boundary_authorizes_action_now=False,
                proposed_next_step_approval_boundary_authorizes_risky_work=False,
                proposed_next_step_approval_boundary_authorizes_followup_without_closure=False,
                proposed_next_step_approval_boundary_authorizes_approval=False,
                proposed_next_step_approval_boundary_authorizes_model_call=False,
                proposed_next_step_approval_boundary_authorizes_tool_execution=False,
                proposed_next_step_approval_boundary_authorizes_personal_data_read=False,
                proposed_next_step_approval_boundary_authorizes_external_side_effect=False,
                proposed_next_step_approval_boundary_authorizes_timebox_reuse=False,
                proposed_next_step_approval_boundary_reusable_for_next_review=False,
                proposed_next_step_approval_boundary_reusable_for_recovery_review=False,
                stop_condition=stop_condition,
                stop_condition_provided=bool(stop_condition),
                post_step_proof_queue=post_step_queue,
                post_step_proof_queue_count=len(post_step_queue),
                post_step_next_proof_command=post_step_queue[0],
                resume_gate_output=resume_gate.output[:1600],
            )
        metadata["autonomy_continuation_execution_ready"] = _autonomy_continuation_execution_ready_from_metadata(metadata)
        return ToolResult(
            "autonomy_continuation_execution_packet",
            True,
            "\n".join(lines),
            metadata,
        )

    def autonomy_step_closure_packet(args: dict[str, Any]) -> ToolResult:
        objective = str(args.get("objective") or args.get("request") or "Close one Jarvis local-safe autonomy continuation step.").strip()
        previous_instruction = str(args.get("previous_instruction") or args.get("previous") or "").strip()
        latest_instruction = str(args.get("latest_instruction") or args.get("latest") or args.get("instruction") or objective).strip()
        stop_at = str(args.get("stop_at") or args.get("stop") or args.get("until") or "").strip()
        current_time = str(args.get("current_time") or args.get("now") or "").strip()
        timezone_label = str(args.get("timezone") or "local").strip() or "local"
        reviewed_step = str(args.get("reviewed_step") or args.get("recovery_step") or args.get("approved_step") or "").strip()
        recovery_verification = str(args.get("recovery_verification") or args.get("verification") or args.get("tests") or "").strip()
        recovery_receipt_path = str(args.get("recovery_receipt_path") or args.get("receipt_path") or args.get("receipt") or "").strip()
        recovery_receipt_sha256 = str(args.get("recovery_receipt_sha256") or args.get("receipt_sha256") or args.get("receipt_hash") or "").strip()
        recovery_checkpoint_path = str(args.get("recovery_checkpoint_path") or args.get("checkpoint_path") or args.get("checkpoint") or "").strip()
        recovery_checkpoint_sha256 = str(args.get("recovery_checkpoint_sha256") or args.get("checkpoint_sha256") or args.get("checkpoint_hash") or "").strip()
        stop_condition = str(args.get("stop_condition") or args.get("stop_rule") or "").strip()
        approval_reference = str(args.get("approval") or args.get("approval_reference") or "").strip()
        blockers_text = str(args.get("blockers") or "none reported").strip()
        proposed_next_step = str(args.get("next_step") or args.get("next") or "").strip()
        proposed_verification = str(args.get("next_verification") or args.get("post_verification_target") or "").strip()
        prior_cycle_ledger_token_sha256 = str(
            args.get("prior_cycle_ledger_token_sha256")
            or args.get("previous_cycle_ledger_token_sha256")
            or args.get("cycle_ledger_token_sha256")
            or ""
        ).strip()
        completed_step = str(args.get("completed_step") or args.get("actual_step") or args.get("step") or proposed_next_step).strip()
        post_step_verification = str(
            args.get("post_step_verification")
            or args.get("verification_evidence")
            or args.get("post_verification")
            or args.get("closure_verification")
            or ""
        ).strip()
        post_step_receipt_path = str(args.get("post_step_receipt_path") or args.get("post_receipt") or args.get("verification_receipt") or "").strip()
        post_step_receipt_sha256 = str(
            args.get("post_step_receipt_sha256")
            or args.get("post_receipt_sha256")
            or args.get("verification_receipt_sha256")
            or args.get("post_step_receipt_hash")
            or ""
        ).strip()
        post_step_receipt_file_sha256 = _file_sha256(post_step_receipt_path)
        post_step_receipt_hash_matches_file = bool(
            post_step_receipt_sha256
            and post_step_receipt_file_sha256
            and post_step_receipt_sha256 == post_step_receipt_file_sha256
        )
        post_step_checkpoint_path = str(args.get("post_step_checkpoint_path") or args.get("post_checkpoint") or args.get("fresh_checkpoint") or "").strip()
        post_step_checkpoint_sha256 = str(
            args.get("post_step_checkpoint_sha256")
            or args.get("post_checkpoint_sha256")
            or args.get("fresh_checkpoint_sha256")
            or args.get("post_step_checkpoint_hash")
            or ""
        ).strip()
        post_step_checkpoint_file_sha256 = _file_sha256(post_step_checkpoint_path)
        post_step_checkpoint_hash_matches_file = bool(
            post_step_checkpoint_sha256
            and post_step_checkpoint_file_sha256
            and post_step_checkpoint_sha256 == post_step_checkpoint_file_sha256
        )
        execution_health = str(args.get("execution_health") or args.get("health") or "").strip()
        execution_audit = str(args.get("execution_audit") or args.get("audit") or "").strip()
        after_action_learning = str(args.get("after_action_learning") or args.get("learning") or "").strip()

        continuation = autonomy_continuation_execution_packet(
            {
                "objective": objective,
                "previous_instruction": previous_instruction,
                "latest_instruction": latest_instruction,
                "stop_at": stop_at,
                "current_time": current_time,
                "timezone": timezone_label,
                "reviewed_step": reviewed_step,
                "verification": recovery_verification,
                "receipt_path": recovery_receipt_path,
                "receipt_sha256": recovery_receipt_sha256,
                "checkpoint_path": recovery_checkpoint_path,
                "checkpoint_sha256": recovery_checkpoint_sha256,
                "stop_condition": stop_condition,
                "blockers": blockers_text,
                "approval_reference": approval_reference,
                "next_step": proposed_next_step or completed_step,
                "next_verification": proposed_verification or post_step_verification,
                "prior_cycle_ledger_token_sha256": prior_cycle_ledger_token_sha256,
            }
        )
        continuation_metadata = continuation.metadata
        one_step_execution_contract_rows = list(continuation_metadata.get("one_step_execution_contract_rows") or [])
        one_step_execution_contract_token_sha256 = str(
            continuation_metadata.get("one_step_execution_contract_token_sha256") or ""
        )
        one_step_execution_contract_token_as_prior_proof = bool(
            continuation_metadata.get("one_step_execution_contract_token_present")
            and _looks_like_sha256(one_step_execution_contract_token_sha256)
            and _one_step_execution_contract_ready(
                objective=str(continuation_metadata.get("objective") or ""),
                continuation_state=str(continuation_metadata.get("continuation_state") or ""),
                resume_gate_state=str(continuation_metadata.get("resume_gate_state") or ""),
                continuation_ready=continuation_metadata.get("one_step_execution_contract_ready"),
                one_step_execution_contract_rows=one_step_execution_contract_rows,
                one_step_execution_contract_token_sha256=one_step_execution_contract_token_sha256,
                timebox_receipt_sha256=str(continuation_metadata.get("timebox_receipt_sha256") or ""),
                awake_guard_token_sha256=str(continuation_metadata.get("awake_guard_token_sha256") or ""),
                supersession_token_sha256=str(continuation_metadata.get("supersession_token_sha256") or ""),
                checkpoint_route_token_sha256=str(continuation_metadata.get("checkpoint_route_token_sha256") or ""),
                recovery_followthrough_token_sha256=str(
                    continuation_metadata.get("recovery_followthrough_token_sha256") or ""
                ),
                local_safe_recovery_execution_token_sha256=str(
                    continuation_metadata.get("local_safe_recovery_execution_token_sha256") or ""
                ),
                prior_cycle_ledger_token_sha256=str(continuation_metadata.get("prior_cycle_ledger_token_sha256") or ""),
                continuation_review_token_sha256=str(continuation_metadata.get("continuation_review_token_sha256") or ""),
                proposed_next_step_sha256=str(continuation_metadata.get("proposed_next_step_sha256") or ""),
                proposed_verification_sha256=str(continuation_metadata.get("proposed_verification_sha256") or ""),
                risk_signals=[str(signal) for signal in (continuation_metadata.get("proposed_next_step_risk_signals") or [])],
                timebox_review_contract_ready=continuation_metadata.get("timebox_review_contract_ready"),
                timebox_review_contract_rows=list(continuation_metadata.get("timebox_review_contract_rows") or []),
                post_step_proof_queue=[
                    str(command)
                    for command in (
                        continuation_metadata.get("post_step_proof_queue")
                        or continuation_metadata.get("continuation_post_step_proof_queue")
                        or []
                    )
                ],
            )
            and continuation_metadata.get("one_step_execution_contract_token_authorizes_action_now") is False
            and continuation_metadata.get("one_step_execution_contract_token_authorizes_risky_work") is False
            and continuation_metadata.get("one_step_execution_contract_token_authorizes_unreviewed_followthrough") is False
            and continuation_metadata.get("one_step_execution_contract_token_reusable_for_next_step") is False
            and continuation_metadata.get("one_step_execution_contract_binds_awake_guard") is True
            and continuation_metadata.get("one_step_execution_contract_binds_operator_supersession") is True
            and continuation_metadata.get("one_step_execution_contract_binds_timebox_review_contract") is True
        )
        supersession_token_boundary_rows = list(continuation_metadata.get("supersession_token_boundary_rows") or [])
        supersession_token_boundary_ready = _operator_supersession_token_boundary_ready(
            str(continuation_metadata.get("supersession_token_sha256") or ""),
            supersession_token_boundary_rows,
        )
        checkpoint_route_boundary_rows = list(continuation_metadata.get("checkpoint_route_boundary_rows") or [])
        checkpoint_route_boundary_ready = _checkpoint_route_boundary_ready(
            str(continuation_metadata.get("checkpoint_route_token_sha256") or ""),
            checkpoint_route_boundary_rows,
        )
        timebox_review_contract_rows = list(continuation_metadata.get("timebox_review_contract_rows") or [])
        timebox_review_contract_ready = bool(
            continuation_metadata.get("timebox_review_contract_ready") is True
            and _operator_timebox_review_contract_ready(
                timebox_review_contract_rows,
                timebox_state=str(continuation_metadata.get("timebox_state") or ""),
                stop_at=str(continuation_metadata.get("stop_at") or ""),
                current_time=str(continuation_metadata.get("current_time") or ""),
                timezone_label=str(continuation_metadata.get("timezone") or ""),
            )
        )
        carried_recovery_execution_scorecard_rows = list(
            continuation_metadata.get("carried_recovery_execution_scorecard_rows") or []
        )
        carried_recovery_execution_score = int(continuation_metadata.get("carried_recovery_execution_score") or 0)
        carried_recovery_execution_max_score = int(continuation_metadata.get("carried_recovery_execution_max_score") or 0)
        carried_recovery_execution_required_rows_ready = _recovery_execution_readiness_scorecard_ready(
            carried_recovery_execution_scorecard_rows
        )
        carried_recovery_execution_scorecard_ready = carried_recovery_execution_required_rows_ready
        carried_recovery_execution_as_prior_proof = bool(
            continuation_metadata.get("carried_recovery_execution_as_prior_proof")
            and _recovery_execution_readiness_scorecard_ready(carried_recovery_execution_scorecard_rows)
        )
        carried_next_step_approval_proof_queue = list(
            continuation_metadata.get("proposed_next_step_approval_proof_queue") or []
        )
        carried_next_step_approval_boundary_rows = list(
            continuation_metadata.get("proposed_next_step_approval_boundary_rows") or []
        )
        carried_next_step_approval_required_before_review = bool(
            continuation_metadata.get("proposed_next_step_approval_required_before_review")
        )
        carried_next_step_approval_boundary_token_sha256 = str(
            continuation_metadata.get("proposed_next_step_approval_boundary_token_sha256") or ""
        )
        carried_next_step_approval_boundary_ready = _next_step_approval_boundary_ready(
            token_sha256=carried_next_step_approval_boundary_token_sha256,
            proposed_step_sha256=str(continuation_metadata.get("proposed_next_step_sha256") or ""),
            proposed_verification_sha256=str(continuation_metadata.get("proposed_verification_sha256") or ""),
            approval_proof_queue=carried_next_step_approval_proof_queue,
            approval_boundary_rows=carried_next_step_approval_boundary_rows,
        )
        carried_next_step_approval_boundary_as_prior_proof = _next_step_approval_boundary_as_prior_proof(
            token_sha256=carried_next_step_approval_boundary_token_sha256,
            proposed_step_sha256=str(continuation_metadata.get("proposed_next_step_sha256") or ""),
            proposed_verification_sha256=str(continuation_metadata.get("proposed_verification_sha256") or ""),
            approval_proof_queue=carried_next_step_approval_proof_queue,
            approval_boundary_rows=carried_next_step_approval_boundary_rows,
        )
        proposed_step_from_continuation = str(continuation_metadata.get("proposed_next_step") or "").strip()
        proposed_step_sha256 = str(continuation_metadata.get("proposed_next_step_sha256") or "")
        proposed_verification_from_continuation = str(continuation_metadata.get("proposed_verification") or "").strip()
        proposed_verification_sha256 = str(continuation_metadata.get("proposed_verification_sha256") or "")
        continuation_review_token_sha256 = str(continuation_metadata.get("continuation_review_token_sha256") or "")
        completed_step_sha256 = _text_sha256(completed_step)
        post_step_verification_sha256 = _text_sha256(post_step_verification)
        completed_step_matches_proposed = bool(
            proposed_step_sha256 and completed_step_sha256 and proposed_step_sha256 == completed_step_sha256
        )
        post_step_verification_matches_proposed = bool(
            proposed_verification_sha256
            and post_step_verification_sha256
            and proposed_verification_sha256 == post_step_verification_sha256
        )
        closure_gate = _autonomy_step_closure_gate(
            continuation_ready=continuation_metadata.get("one_local_safe_step_allowed"),
            continued_step=completed_step,
            completed_step_matches_proposed=completed_step_matches_proposed,
            verification_matches_proposed=post_step_verification_matches_proposed,
            verification_evidence=post_step_verification,
            post_step_receipt_path=post_step_receipt_path,
            post_step_receipt_sha256=post_step_receipt_sha256,
            post_step_receipt_file_sha256=post_step_receipt_file_sha256,
            post_step_receipt_hash_matches_file=post_step_receipt_hash_matches_file,
            post_step_checkpoint_path=post_step_checkpoint_path,
            post_step_checkpoint_sha256=post_step_checkpoint_sha256,
            post_step_checkpoint_file_sha256=post_step_checkpoint_file_sha256,
            post_step_checkpoint_hash_matches_file=post_step_checkpoint_hash_matches_file,
            execution_health=execution_health,
            execution_audit=execution_audit,
            after_action_learning=after_action_learning,
            blockers=blockers_text,
        )
        if not timebox_review_contract_ready:
            closure_gate["missing"].append("non_authorizing_timebox_review_contract")
            closure_gate["missing"] = list(dict.fromkeys(closure_gate["missing"]))
            closure_gate["missing_count"] = len(closure_gate["missing"])
            closure_gate["ready_for_next_continuation_review"] = False
            if "operator timebox contract: stop_at=<ISO> current_time=<ISO>" not in closure_gate["required_commands"]:
                closure_gate["required_commands"].insert(0, "operator timebox contract: stop_at=<ISO> current_time=<ISO>")
            closure_gate["next_required_command"] = closure_gate["required_commands"][0]
        if not supersession_token_boundary_ready:
            closure_gate["missing"].append("operator_supersession_token_boundary")
        if not checkpoint_route_boundary_ready:
            closure_gate["missing"].append("checkpoint_route_boundary")
            closure_gate["missing"] = list(dict.fromkeys(closure_gate["missing"]))
            closure_gate["missing_count"] = len(closure_gate["missing"])
            closure_gate["ready_for_next_continuation_review"] = False
            supersession_command = "operator instruction supersession: latest=<newest instruction> stop_at=<ISO> current_time=<ISO>"
            if supersession_command not in closure_gate["required_commands"]:
                closure_gate["required_commands"].insert(0, supersession_command)
            closure_gate["next_required_command"] = closure_gate["required_commands"][0]
        ready = bool(closure_gate["ready_for_next_continuation_review"])
        fresh_continuation_review_contract_rows = [
            {
                "item": "fresh_operator_timebox",
                "source": "operator timebox contract",
                "fresh_required": True,
                "prior_artifact_reusable": False,
                "authorizes_action_now": False,
                "authorizes_risky_work": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "authorizes_approval": False,
            },
            {
                "item": "fresh_checkpoint",
                "source": "work block checkpoint",
                "fresh_required": True,
                "prior_artifact_reusable": False,
                "authorizes_action_now": False,
                "authorizes_risky_work": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "authorizes_approval": False,
            },
            {
                "item": "fresh_recovery_cockpit",
                "source": "checkpoint recovery cockpit",
                "fresh_required": True,
                "prior_artifact_reusable": False,
                "authorizes_action_now": False,
                "authorizes_risky_work": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "authorizes_approval": False,
            },
            {
                "item": "fresh_local_safe_step",
                "source": "autonomy continuation execution packet",
                "fresh_required": True,
                "prior_artifact_reusable": False,
                "authorizes_action_now": False,
                "authorizes_risky_work": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "authorizes_approval": False,
            },
            {
                "item": "fresh_continuation_review_token",
                "source": "continuation review token",
                "fresh_required": True,
                "prior_artifact_reusable": False,
                "authorizes_action_now": False,
                "authorizes_risky_work": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "authorizes_approval": False,
            },
            {
                "item": "prior_step_closure",
                "source": "autonomy step closure packet",
                "fresh_required": False,
                "prior_artifact_reusable": False,
                "authorizes_action_now": False,
                "authorizes_risky_work": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
                "authorizes_approval": False,
            },
        ]
        fresh_continuation_review_contract_summary = [
            "fresh_operator_timebox_required",
            "fresh_checkpoint_required",
            "fresh_recovery_cockpit_required",
            "fresh_local_safe_step_required",
            "fresh_continuation_review_token_required",
            "prior_step_closure_proof_only",
            "no_followup_without_new_review",
        ]
        fresh_continuation_review_contract_enforced = _fresh_continuation_review_contract_ready(
            fresh_continuation_review_contract_rows
        )
        step_closure_scorecard_rows = _autonomy_step_closure_readiness_scorecard_rows(
            pre_step_permission_ready=bool(continuation_metadata.get("one_local_safe_step_allowed")),
            completed_step_matches_proposed=completed_step_matches_proposed,
            post_step_verification_matches_proposed=post_step_verification_matches_proposed,
            post_step_receipt_hash_matches_file=post_step_receipt_hash_matches_file,
            post_step_checkpoint_hash_matches_file=post_step_checkpoint_hash_matches_file,
            execution_health=execution_health,
            execution_audit=execution_audit,
            after_action_learning=after_action_learning,
            fresh_contract_enforced=fresh_continuation_review_contract_enforced,
            closure_ready=ready,
        )
        step_closure_score = sum(int(row["points"]) for row in step_closure_scorecard_rows)
        step_closure_max_score = sum(int(row["max_points"]) for row in step_closure_scorecard_rows)
        step_closure_required_rows_ready = _scorecard_required_rows_ready(step_closure_scorecard_rows)
        step_closure_scorecard_ready = _autonomy_step_closure_readiness_scorecard_ready(
            step_closure_scorecard_rows
        )
        local_safe_recovery_execution_token_boundary_rows = _local_safe_recovery_execution_token_boundary_rows(
            token_sha256=str(continuation_metadata.get("local_safe_recovery_execution_token_sha256") or ""),
            source="autonomy_step_closure",
        )
        local_safe_recovery_execution_token_boundary_ready = _local_safe_recovery_execution_token_boundary_ready(
            str(continuation_metadata.get("local_safe_recovery_execution_token_sha256") or ""),
            local_safe_recovery_execution_token_boundary_rows,
            expected_source="autonomy_step_closure",
        )
        recovery_followthrough_token_boundary_rows = list(
            continuation_metadata.get("recovery_followthrough_token_boundary_rows") or []
        )
        recovery_followthrough_token_boundary_ready = _carried_recovery_followthrough_token_boundary_ready(
            continuation_metadata,
            expected_source="checkpoint_recovery_followthrough",
        )
        recovery_execution_readiness_token_boundary_rows = list(
            continuation_metadata.get("recovery_execution_readiness_token_boundary_rows") or []
        )
        recovery_execution_readiness_token_boundary_ready = _carried_recovery_execution_readiness_token_boundary_ready(
            continuation_metadata,
            expected_source="checkpoint_recovery_followthrough",
        )
        prior_cycle_ledger_token_boundary_rows = _prior_cycle_ledger_token_boundary_rows(
            token_sha256=str(continuation_metadata.get("prior_cycle_ledger_token_sha256") or ""),
            source="autonomy_step_closure",
        )
        prior_cycle_ledger_token_boundary_ready = _prior_cycle_ledger_token_boundary_ready(
            str(continuation_metadata.get("prior_cycle_ledger_token_sha256") or ""),
            prior_cycle_ledger_token_boundary_rows,
        )
        next_safe_command = (
            "autonomy continuation execution: <next reviewed local-safe step>"
            if ready
            else "autonomy step closure: step=<completed local-safe step> verification=<post-step evidence> receipt=<post-step receipt> checkpoint=<fresh checkpoint> health=<execution health> audit=<execution audit> learning=<after-action learning>"
        )
        fresh_continuation_review_boundary_token_sha256 = _fresh_continuation_review_boundary_token_sha256(
            objective=objective,
            closure_state=str(closure_gate["state"]),
            next_safe_command=next_safe_command,
            continuation_review_token_sha256=continuation_review_token_sha256,
            fresh_continuation_review_contract_rows=fresh_continuation_review_contract_rows,
        )
        fresh_continuation_review_boundary_token_rows = _fresh_review_boundary_token_rows(
            token_sha256=fresh_continuation_review_boundary_token_sha256,
            source="autonomy_step_closure",
        )
        fresh_continuation_review_boundary_token_ready = _fresh_review_boundary_token_ready(
            fresh_continuation_review_boundary_token_sha256,
            fresh_continuation_review_boundary_token_rows,
        )
        step_closure_receipt_token_sha256 = _autonomy_step_closure_receipt_token_sha256(
            objective=objective,
            closure_state=str(closure_gate["state"]),
            completed_step_sha256=completed_step_sha256,
            post_step_verification_sha256=post_step_verification_sha256,
            post_step_receipt_sha256=post_step_receipt_sha256,
            post_step_receipt_file_sha256=post_step_receipt_file_sha256,
            post_step_checkpoint_sha256=post_step_checkpoint_sha256,
            post_step_checkpoint_file_sha256=post_step_checkpoint_file_sha256,
            execution_health=execution_health,
            execution_audit=execution_audit,
            after_action_learning=after_action_learning,
            fresh_continuation_review_boundary_token_sha256=fresh_continuation_review_boundary_token_sha256,
            step_closure_scorecard_rows=step_closure_scorecard_rows,
        )
        step_closure_receipt_boundary_rows = _autonomy_step_closure_receipt_boundary_rows(
            token_sha256=step_closure_receipt_token_sha256,
            source="autonomy_step_closure",
        )
        step_closure_receipt_boundary_ready = _autonomy_step_closure_receipt_boundary_ready(
            step_closure_receipt_token_sha256,
            step_closure_receipt_boundary_rows,
        )
        candidate_next_safe_command = (
            "autonomy continuation execution: <next reviewed local-safe step>"
            if bool(closure_gate["ready_for_next_continuation_review"])
            else "autonomy step closure: step=<completed local-safe step> verification=<post-step evidence> receipt=<post-step receipt> checkpoint=<fresh checkpoint> health=<execution health> audit=<execution audit> learning=<after-action learning>"
        )
        readiness_candidate = _safe_metadata(
            objective=objective,
            closure_state=closure_gate["state"],
            ready_for_next_continuation_review=bool(closure_gate["ready_for_next_continuation_review"]),
            missing_blockers=closure_gate["missing"],
            missing_blocker_count=closure_gate["missing_count"],
            next_safe_command=candidate_next_safe_command,
            required_commands=closure_gate["required_commands"],
            required_command_count=len(closure_gate["required_commands"]),
            next_required_command=closure_gate["next_required_command"],
            proof_queue=closure_gate["required_commands"],
            proof_queue_count=len(closure_gate["required_commands"]),
            next_proof_command=closure_gate["next_required_command"],
            resume_gate_state=continuation_metadata.get("resume_gate_state"),
            continuation_state=continuation_metadata.get("continuation_state"),
            timebox_state=continuation_metadata.get("timebox_state"),
            can_continue_now=continuation_metadata.get("can_continue_now"),
            should_stop_now=continuation_metadata.get("should_stop_now"),
            missing_timebox_proof=continuation_metadata.get("missing_timebox_proof", []),
            missing_timebox_proof_count=continuation_metadata.get("missing_timebox_proof_count", 0),
            timebox_receipt_sha256=continuation_metadata.get("timebox_receipt_sha256", ""),
            awake_guard_token_sha256=continuation_metadata.get("awake_guard_token_sha256", ""),
            supersession_token_sha256=continuation_metadata.get("supersession_token_sha256", ""),
            checkpoint_route_token_sha256=continuation_metadata.get("checkpoint_route_token_sha256", ""),
            receipt_sha256=continuation_metadata.get("receipt_sha256", ""),
            receipt_file_sha256=continuation_metadata.get("receipt_file_sha256", ""),
            checkpoint_sha256=continuation_metadata.get("checkpoint_sha256", ""),
            recovery_checkpoint_file_sha256=continuation_metadata.get("checkpoint_file_sha256", ""),
            recovery_followthrough_token_sha256=continuation_metadata.get("recovery_followthrough_token_sha256", ""),
            local_safe_recovery_execution_token_sha256=continuation_metadata.get("local_safe_recovery_execution_token_sha256", ""),
            recovery_execution_readiness_token_sha256=continuation_metadata.get("recovery_execution_readiness_token_sha256", ""),
            one_step_execution_contract_token_sha256=one_step_execution_contract_token_sha256,
            prior_cycle_ledger_token_sha256=prior_cycle_ledger_token_sha256,
            proposed_next_step_sha256=proposed_step_sha256,
            completed_step_sha256=completed_step_sha256,
            proposed_verification_sha256=proposed_verification_sha256,
            post_step_verification_sha256=post_step_verification_sha256,
            continuation_review_token_sha256=continuation_review_token_sha256,
            fresh_continuation_review_boundary_token_sha256=fresh_continuation_review_boundary_token_sha256,
            step_closure_receipt_token_sha256=step_closure_receipt_token_sha256,
            post_step_receipt_sha256=post_step_receipt_sha256,
            post_step_receipt_file_sha256=post_step_receipt_file_sha256,
            post_step_checkpoint_sha256=post_step_checkpoint_sha256,
            post_step_checkpoint_file_sha256=post_step_checkpoint_file_sha256,
            timebox_review_contract_rows=timebox_review_contract_rows,
            timebox_review_contract_row_count=len(timebox_review_contract_rows),
            timebox_review_contract_ready=timebox_review_contract_ready,
            timebox_receipt_present=continuation_metadata.get("timebox_receipt_present", False),
            timebox_authorizes_execution=continuation_metadata.get("timebox_authorizes_execution", False),
            timebox_authorizes_local_safe_step=continuation_metadata.get("timebox_authorizes_local_safe_step", False),
            timebox_authorizes_risky_work=continuation_metadata.get("timebox_authorizes_risky_work", False),
            timebox_authorizes_approval=continuation_metadata.get("timebox_authorizes_approval", False),
            timebox_authorizes_timebox_reuse=continuation_metadata.get("timebox_authorizes_timebox_reuse", False),
            timebox_authorizes_model_call=continuation_metadata.get("timebox_authorizes_model_call", False),
            timebox_authorizes_tool_execution=continuation_metadata.get("timebox_authorizes_tool_execution", False),
            timebox_authorizes_personal_data_read=continuation_metadata.get("timebox_authorizes_personal_data_read", False),
            timebox_authorizes_external_side_effect=continuation_metadata.get("timebox_authorizes_external_side_effect", False),
            timebox_reusable_for_next_step=continuation_metadata.get("timebox_reusable_for_next_step", False),
            next_step_requires_fresh_timebox=continuation_metadata.get("next_step_requires_fresh_timebox", False),
            previous_instruction=continuation_metadata.get("previous_instruction", ""),
            latest_instruction=continuation_metadata.get("latest_instruction", ""),
            latest_instruction_present=continuation_metadata.get("latest_instruction_present", False),
            latest_instruction_supersedes_previous=continuation_metadata.get("latest_instruction_supersedes_previous", False),
            latest_instruction_is_stop=continuation_metadata.get("latest_instruction_is_stop", False),
            latest_instruction_is_continue=continuation_metadata.get("latest_instruction_is_continue", False),
            supersession_state=continuation_metadata.get("supersession_state", ""),
            can_continue_under_latest_instruction=continuation_metadata.get("can_continue_under_latest_instruction", False),
            newest_instruction_overrides_automation=continuation_metadata.get("newest_instruction_overrides_automation", True),
            newest_instruction_overrides_goal=continuation_metadata.get("newest_instruction_overrides_goal", True),
            newest_instruction_overrides_recovery_queue=continuation_metadata.get("newest_instruction_overrides_recovery_queue", True),
            stop_or_pause_blocks_autonomy=continuation_metadata.get("stop_or_pause_blocks_autonomy", False),
            supersession_token_boundary_rows=supersession_token_boundary_rows,
            supersession_token_boundary_row_count=len(supersession_token_boundary_rows),
            supersession_token_present=continuation_metadata.get("supersession_token_present", False),
            supersession_token_boundary_ready=supersession_token_boundary_ready,
            supersession_token_authorizes_execution=continuation_metadata.get(
                "supersession_token_authorizes_execution", False
            ),
            supersession_token_authorizes_local_safe_step=continuation_metadata.get(
                "supersession_token_authorizes_local_safe_step", False
            ),
            supersession_token_authorizes_risky_work=continuation_metadata.get(
                "supersession_token_authorizes_risky_work", False
            ),
            supersession_token_authorizes_approval=continuation_metadata.get(
                "supersession_token_authorizes_approval", False
            ),
            supersession_token_authorizes_recovery_followthrough=continuation_metadata.get(
                "supersession_token_authorizes_recovery_followthrough", False
            ),
            supersession_token_authorizes_timebox_override=continuation_metadata.get(
                "supersession_token_authorizes_timebox_override", False
            ),
            supersession_token_authorizes_goal_override=continuation_metadata.get(
                "supersession_token_authorizes_goal_override", False
            ),
            supersession_token_authorizes_model_call=continuation_metadata.get(
                "supersession_token_authorizes_model_call", False
            ),
            supersession_token_authorizes_tool_execution=continuation_metadata.get(
                "supersession_token_authorizes_tool_execution", False
            ),
            supersession_token_authorizes_personal_data_read=continuation_metadata.get(
                "supersession_token_authorizes_personal_data_read", False
            ),
            supersession_token_authorizes_external_side_effect=continuation_metadata.get(
                "supersession_token_authorizes_external_side_effect", False
            ),
            supersession_token_reusable_for_next_review=continuation_metadata.get(
                "supersession_token_reusable_for_next_review", False
            ),
            supersession_token_reusable_for_next_timebox=continuation_metadata.get(
                "supersession_token_reusable_for_next_timebox", False
            ),
            next_supersession_requires_fresh_latest_instruction_review=continuation_metadata.get(
                "next_supersession_requires_fresh_latest_instruction_review", False
            ),
            checkpoint_route_boundary_rows=checkpoint_route_boundary_rows,
            checkpoint_route_boundary_row_count=len(checkpoint_route_boundary_rows),
            checkpoint_route_token_present=continuation_metadata.get("checkpoint_route_token_present", False),
            checkpoint_route_boundary_ready=checkpoint_route_boundary_ready,
            checkpoint_freshness=continuation_metadata.get("checkpoint_freshness", ""),
            checkpoint_needs_review=continuation_metadata.get("checkpoint_needs_review", True),
            checkpoint_route_authorizes_continuation=continuation_metadata.get(
                "checkpoint_route_authorizes_continuation", False
            ),
            checkpoint_route_authorizes_local_safe_step=continuation_metadata.get(
                "checkpoint_route_authorizes_local_safe_step", False
            ),
            checkpoint_route_authorizes_risky_work=continuation_metadata.get(
                "checkpoint_route_authorizes_risky_work", False
            ),
            checkpoint_route_authorizes_approval=continuation_metadata.get(
                "checkpoint_route_authorizes_approval", False
            ),
            checkpoint_route_authorizes_recovery_followthrough=continuation_metadata.get(
                "checkpoint_route_authorizes_recovery_followthrough", False
            ),
            checkpoint_route_authorizes_checkpoint_reuse=continuation_metadata.get(
                "checkpoint_route_authorizes_checkpoint_reuse", False
            ),
            checkpoint_route_authorizes_model_call=continuation_metadata.get(
                "checkpoint_route_authorizes_model_call", False
            ),
            checkpoint_route_authorizes_tool_execution=continuation_metadata.get(
                "checkpoint_route_authorizes_tool_execution", False
            ),
            checkpoint_route_authorizes_personal_data_read=continuation_metadata.get(
                "checkpoint_route_authorizes_personal_data_read", False
            ),
            checkpoint_route_authorizes_external_side_effect=continuation_metadata.get(
                "checkpoint_route_authorizes_external_side_effect", False
            ),
            checkpoint_route_reusable_for_next_review=continuation_metadata.get(
                "checkpoint_route_reusable_for_next_review", False
            ),
            checkpoint_route_reusable_for_next_checkpoint=continuation_metadata.get(
                "checkpoint_route_reusable_for_next_checkpoint", False
            ),
            next_checkpoint_route_requires_fresh_recovery_review=continuation_metadata.get(
                "next_checkpoint_route_requires_fresh_recovery_review", False
            ),
            pre_step_one_local_safe_step_allowed=continuation_metadata.get("one_local_safe_step_allowed"),
            one_step_execution_contract_rows=one_step_execution_contract_rows,
            one_step_execution_contract_row_count=len(one_step_execution_contract_rows),
            one_step_execution_contract_ready=_metadata_flag_ready(
                continuation_metadata, "one_step_execution_contract_ready"
            ),
            one_step_execution_contract_token_present=_looks_like_sha256(one_step_execution_contract_token_sha256),
            one_step_execution_contract_token_as_prior_proof=one_step_execution_contract_token_as_prior_proof,
            one_step_execution_contract_binds_awake_guard=_metadata_flag_ready(
                continuation_metadata, "one_step_execution_contract_binds_awake_guard"
            ),
            one_step_execution_contract_binds_operator_supersession=_metadata_flag_ready(
                continuation_metadata, "one_step_execution_contract_binds_operator_supersession"
            ),
            one_step_execution_contract_binds_timebox_review_contract=_metadata_flag_ready(
                continuation_metadata, "one_step_execution_contract_binds_timebox_review_contract"
            ),
            one_step_execution_contract_token_authorizes_action_now=False,
            one_step_execution_contract_token_authorizes_risky_work=False,
            one_step_execution_contract_token_authorizes_unreviewed_followthrough=False,
            one_step_execution_contract_token_authorizes_batching=False,
            one_step_execution_contract_token_reusable_for_next_step=False,
            next_step_requires_new_one_step_execution_contract_token=True,
            one_step_execution_contract_all_local_safe_step_limited=_metadata_flag_ready(
                continuation_metadata, "one_step_execution_contract_all_local_safe_step_limited"
            ),
            one_step_execution_contract_all_non_reusable=_metadata_flag_ready(
                continuation_metadata, "one_step_execution_contract_all_non_reusable"
            ),
            one_step_execution_contract_all_risky_work_gated=_metadata_flag_ready(
                continuation_metadata, "one_step_execution_contract_all_risky_work_gated"
            ),
            one_step_execution_contract_requires_fresh_closure=_metadata_flag_ready(
                continuation_metadata, "one_step_execution_contract_requires_fresh_closure"
            ),
            one_step_execution_contract_authorizes_batching=False,
            one_step_execution_contract_authorizes_followup_without_closure=False,
            recovery_artifact_hashes_present=continuation_metadata.get("recovery_artifact_hashes_present", False),
            recovery_artifact_hashes_match_files=continuation_metadata.get("recovery_artifact_hashes_match_files", False),
            recovery_followthrough_token_boundary_rows=recovery_followthrough_token_boundary_rows,
            recovery_followthrough_token_boundary_row_count=len(recovery_followthrough_token_boundary_rows),
            recovery_followthrough_token_present=continuation_metadata.get("recovery_followthrough_token_present", False),
            recovery_followthrough_token_boundary_ready=recovery_followthrough_token_boundary_ready,
            recovery_followthrough_token_reusable_for_future_recovery=continuation_metadata.get(
                "recovery_followthrough_token_reusable_for_future_recovery", False
            ),
            recovery_followthrough_token_authorizes_resume_gate=continuation_metadata.get(
                "recovery_followthrough_token_authorizes_resume_gate", False
            ),
            recovery_followthrough_token_authorizes_next_step=continuation_metadata.get(
                "recovery_followthrough_token_authorizes_next_step", False
            ),
            recovery_followthrough_token_authorizes_risky_work=continuation_metadata.get(
                "recovery_followthrough_token_authorizes_risky_work", False
            ),
            recovery_followthrough_token_authorizes_approval=continuation_metadata.get(
                "recovery_followthrough_token_authorizes_approval", False
            ),
            recovery_followthrough_token_authorizes_model_call=continuation_metadata.get(
                "recovery_followthrough_token_authorizes_model_call", False
            ),
            recovery_followthrough_token_authorizes_tool_execution=continuation_metadata.get(
                "recovery_followthrough_token_authorizes_tool_execution", False
            ),
            recovery_followthrough_token_authorizes_personal_data_read=continuation_metadata.get(
                "recovery_followthrough_token_authorizes_personal_data_read", False
            ),
            recovery_followthrough_token_authorizes_external_side_effect=continuation_metadata.get(
                "recovery_followthrough_token_authorizes_external_side_effect", False
            ),
            next_recovery_followthrough_requires_new_token=continuation_metadata.get(
                "next_recovery_followthrough_requires_new_token", False
            ),
            local_safe_recovery_execution_token_boundary_rows=local_safe_recovery_execution_token_boundary_rows,
            local_safe_recovery_execution_token_boundary_row_count=len(local_safe_recovery_execution_token_boundary_rows),
            local_safe_recovery_execution_token_present=continuation_metadata.get(
                "local_safe_recovery_execution_token_present", False
            ),
            local_safe_recovery_execution_token_boundary_ready=local_safe_recovery_execution_token_boundary_ready,
            local_safe_recovery_execution_token_authorizes_resume_gate=continuation_metadata.get(
                "local_safe_recovery_execution_token_authorizes_resume_gate", False
            ),
            local_safe_recovery_execution_token_authorizes_next_step=continuation_metadata.get(
                "local_safe_recovery_execution_token_authorizes_next_step", False
            ),
            local_safe_recovery_execution_token_authorizes_risky_work=continuation_metadata.get(
                "local_safe_recovery_execution_token_authorizes_risky_work", False
            ),
            local_safe_recovery_execution_token_authorizes_approval=continuation_metadata.get(
                "local_safe_recovery_execution_token_authorizes_approval", False
            ),
            local_safe_recovery_execution_token_authorizes_model_call=continuation_metadata.get(
                "local_safe_recovery_execution_token_authorizes_model_call", False
            ),
            local_safe_recovery_execution_token_authorizes_tool_execution=continuation_metadata.get(
                "local_safe_recovery_execution_token_authorizes_tool_execution", False
            ),
            local_safe_recovery_execution_token_authorizes_personal_data_read=continuation_metadata.get(
                "local_safe_recovery_execution_token_authorizes_personal_data_read", False
            ),
            local_safe_recovery_execution_token_authorizes_external_side_effect=continuation_metadata.get(
                "local_safe_recovery_execution_token_authorizes_external_side_effect", False
            ),
            local_safe_recovery_execution_token_reusable_for_future_recovery=continuation_metadata.get(
                "local_safe_recovery_execution_token_reusable_for_future_recovery", False
            ),
            next_recovery_execution_requires_new_local_safe_token=continuation_metadata.get(
                "next_recovery_execution_requires_new_local_safe_token", False
            ),
            recovery_execution_readiness_token_boundary_rows=recovery_execution_readiness_token_boundary_rows,
            recovery_execution_readiness_token_boundary_row_count=len(recovery_execution_readiness_token_boundary_rows),
            recovery_execution_readiness_token_present=continuation_metadata.get(
                "recovery_execution_readiness_token_present", False
            ),
            recovery_execution_readiness_token_boundary_ready=recovery_execution_readiness_token_boundary_ready,
            recovery_execution_readiness_token_authorizes_resume_gate=continuation_metadata.get(
                "recovery_execution_readiness_token_authorizes_resume_gate", False
            ),
            recovery_execution_readiness_token_authorizes_next_step=continuation_metadata.get(
                "recovery_execution_readiness_token_authorizes_next_step", False
            ),
            recovery_execution_readiness_token_authorizes_risky_work=continuation_metadata.get(
                "recovery_execution_readiness_token_authorizes_risky_work", False
            ),
            recovery_execution_readiness_token_authorizes_approval=continuation_metadata.get(
                "recovery_execution_readiness_token_authorizes_approval", False
            ),
            recovery_execution_readiness_token_authorizes_model_call=continuation_metadata.get(
                "recovery_execution_readiness_token_authorizes_model_call", False
            ),
            recovery_execution_readiness_token_authorizes_tool_execution=continuation_metadata.get(
                "recovery_execution_readiness_token_authorizes_tool_execution", False
            ),
            recovery_execution_readiness_token_authorizes_personal_data_read=continuation_metadata.get(
                "recovery_execution_readiness_token_authorizes_personal_data_read", False
            ),
            recovery_execution_readiness_token_authorizes_external_side_effect=continuation_metadata.get(
                "recovery_execution_readiness_token_authorizes_external_side_effect", False
            ),
            recovery_execution_readiness_token_authorizes_unreviewed_followthrough=continuation_metadata.get(
                "recovery_execution_readiness_token_authorizes_unreviewed_followthrough", False
            ),
            recovery_execution_readiness_token_reusable_for_future_recovery=continuation_metadata.get(
                "recovery_execution_readiness_token_reusable_for_future_recovery", False
            ),
            next_recovery_execution_requires_new_readiness_token=continuation_metadata.get(
                "next_recovery_execution_requires_new_readiness_token", False
            ),
            carried_recovery_execution_score=carried_recovery_execution_score,
            carried_recovery_execution_max_score=carried_recovery_execution_max_score,
            carried_recovery_execution_scorecard_rows=carried_recovery_execution_scorecard_rows,
            carried_recovery_execution_scorecard_row_count=len(carried_recovery_execution_scorecard_rows),
            carried_recovery_execution_required_rows_ready=carried_recovery_execution_required_rows_ready,
            carried_recovery_execution_scorecard_ready=carried_recovery_execution_scorecard_ready,
            carried_recovery_execution_as_prior_proof=carried_recovery_execution_as_prior_proof,
            carried_next_step_approval_proof_queue=carried_next_step_approval_proof_queue,
            carried_next_step_approval_proof_queue_count=len(carried_next_step_approval_proof_queue),
            carried_next_step_next_approval_proof_command=(
                carried_next_step_approval_proof_queue[0] if carried_next_step_approval_proof_queue else ""
            ),
            carried_next_step_approval_required_before_review=carried_next_step_approval_required_before_review,
            carried_next_step_approval_boundary_rows=carried_next_step_approval_boundary_rows,
            carried_next_step_approval_boundary_row_count=len(carried_next_step_approval_boundary_rows),
            carried_next_step_approval_boundary_ready=carried_next_step_approval_boundary_ready,
            carried_next_step_approval_boundary_token_sha256=carried_next_step_approval_boundary_token_sha256,
            carried_next_step_approval_boundary_token_present=_looks_like_sha256(carried_next_step_approval_boundary_token_sha256),
            carried_next_step_approval_boundary_as_prior_proof=carried_next_step_approval_boundary_as_prior_proof,
            carried_next_step_approval_boundary_authorizes_action_now=False,
            carried_next_step_approval_boundary_authorizes_risky_work=False,
            carried_next_step_approval_boundary_authorizes_unreviewed_followthrough=False,
            carried_next_step_approval_boundary_authorizes_approval=False,
            carried_next_step_approval_boundary_authorizes_model_call=False,
            carried_next_step_approval_boundary_authorizes_tool_execution=False,
            carried_next_step_approval_boundary_authorizes_personal_data_read=False,
            carried_next_step_approval_boundary_authorizes_external_side_effect=False,
            carried_next_step_approval_boundary_authorizes_timebox_reuse=False,
            carried_next_step_approval_boundary_reusable_for_next_review=False,
            carried_next_step_approval_boundary_reusable_for_recovery_review=False,
            proposed_next_step=proposed_step_from_continuation[:500],
            proposed_next_step_risk_signals=continuation_metadata.get("proposed_next_step_risk_signals", []),
            proposed_next_step_risk_signal_count=continuation_metadata.get("proposed_next_step_risk_signal_count", 0),
            proposed_verification=proposed_verification_from_continuation[:500],
            continuation_post_step_proof_queue=continuation_metadata.get("post_step_proof_queue", []),
            continuation_post_step_proof_queue_count=continuation_metadata.get("post_step_proof_queue_count", 0),
            continuation_post_step_next_proof_command=continuation_metadata.get("post_step_next_proof_command", ""),
            continuation_review_token_present=_looks_like_sha256(continuation_review_token_sha256),
            continuation_review_token_reusable_for_next_review=False,
            previous_continuation_review_token_reusable_for_next_review=False,
            next_review_requires_new_continuation_review_token=True,
            prior_cycle_ledger_token_present=continuation_metadata.get("prior_cycle_ledger_token_present", False),
            prior_cycle_ledger_token_boundary_rows=prior_cycle_ledger_token_boundary_rows,
            prior_cycle_ledger_token_boundary_row_count=len(prior_cycle_ledger_token_boundary_rows),
            prior_cycle_ledger_token_boundary_ready=prior_cycle_ledger_token_boundary_ready,
            prior_cycle_ledger_token_reusable_for_this_review=False,
            prior_cycle_ledger_token_reusable_for_this_closure=False,
            prior_cycle_ledger_token_reusable_for_this_cycle=False,
            prior_cycle_ledger_proof_authorizes_action_now=False,
            prior_cycle_ledger_proof_authorizes_post_step_closure=False,
            prior_cycle_ledger_proof_authorizes_new_action=False,
            prior_cycle_ledger_proof_authorizes_model_call=False,
            prior_cycle_ledger_proof_authorizes_tool_execution=False,
            prior_cycle_ledger_proof_authorizes_personal_data_read=False,
            prior_cycle_ledger_proof_authorizes_external_side_effect=False,
            completed_step_provided=bool(completed_step),
            completed_step_matches_proposed=completed_step_matches_proposed,
            post_step_verification_provided=bool(post_step_verification),
            post_step_verification_matches_proposed=post_step_verification_matches_proposed,
            post_step_receipt_path_provided=bool(post_step_receipt_path),
            post_step_receipt_hash_provided=_looks_like_sha256(post_step_receipt_sha256),
            post_step_receipt_file_hash_present=_looks_like_sha256(post_step_receipt_file_sha256),
            post_step_receipt_hash_matches_file=post_step_receipt_hash_matches_file,
            post_step_checkpoint_path_provided=bool(post_step_checkpoint_path),
            post_step_checkpoint_hash_provided=_looks_like_sha256(post_step_checkpoint_sha256),
            post_step_checkpoint_file_hash_present=_looks_like_sha256(post_step_checkpoint_file_sha256),
            post_step_checkpoint_hash_matches_file=post_step_checkpoint_hash_matches_file,
            post_step_artifact_hashes_present=bool(
                _looks_like_sha256(post_step_receipt_sha256) and _looks_like_sha256(post_step_checkpoint_sha256)
            ),
            post_step_artifact_hashes_match_files=bool(post_step_receipt_hash_matches_file and post_step_checkpoint_hash_matches_file),
            execution_health_reviewed=bool(execution_health),
            execution_audit_reviewed=bool(execution_audit),
            after_action_learning_reviewed=bool(after_action_learning),
            blockers_cleared=blockers_text.strip().lower() in {"", "none", "none reported", "no blockers"},
            fresh_continuation_review_contract_rows=fresh_continuation_review_contract_rows,
            fresh_continuation_review_contract_row_count=len(fresh_continuation_review_contract_rows),
            fresh_continuation_review_contract_enforced=fresh_continuation_review_contract_enforced,
            fresh_continuation_review_boundary_token_present=_looks_like_sha256(fresh_continuation_review_boundary_token_sha256),
            fresh_continuation_review_boundary_token_rows=fresh_continuation_review_boundary_token_rows,
            fresh_continuation_review_boundary_token_row_count=len(fresh_continuation_review_boundary_token_rows),
            fresh_continuation_review_boundary_token_ready=fresh_continuation_review_boundary_token_ready,
            fresh_continuation_review_boundary_authorizes_action_now=False,
            fresh_continuation_review_boundary_authorizes_local_safe_step=False,
            fresh_continuation_review_boundary_authorizes_risky_work=False,
            fresh_continuation_review_boundary_authorizes_unreviewed_followthrough=False,
            fresh_continuation_review_boundary_authorizes_timebox_reuse=False,
            fresh_continuation_review_boundary_authorizes_checkpoint_reuse=False,
            fresh_continuation_review_boundary_authorizes_token_reuse=False,
            fresh_continuation_review_boundary_authorizes_model_call=False,
            fresh_continuation_review_boundary_authorizes_tool_execution=False,
            fresh_continuation_review_boundary_authorizes_personal_data_read=False,
            fresh_continuation_review_boundary_authorizes_external_side_effect=False,
            fresh_continuation_review_boundary_reusable_for_next_review=False,
            fresh_continuation_review_boundary_reusable_for_next_cycle=False,
            next_review_requires_new_fresh_continuation_review_boundary_token=True,
            step_closure_receipt_token_present=_looks_like_sha256(step_closure_receipt_token_sha256),
            step_closure_receipt_boundary_rows=step_closure_receipt_boundary_rows,
            step_closure_receipt_boundary_row_count=len(step_closure_receipt_boundary_rows),
            step_closure_receipt_boundary_ready=step_closure_receipt_boundary_ready,
            step_closure_readiness_score=step_closure_score,
            step_closure_readiness_max_score=step_closure_max_score,
            step_closure_readiness_scorecard_rows=step_closure_scorecard_rows,
            step_closure_readiness_scorecard_row_count=len(step_closure_scorecard_rows),
            step_closure_readiness_required_rows_ready=step_closure_required_rows_ready,
            step_closure_readiness_scorecard_ready=step_closure_scorecard_ready,
            next_review_requires_new_step_closure_receipt_token=True,
            next_continuation_requires_fresh_operator_timebox=True,
            next_continuation_requires_fresh_checkpoint=True,
            next_continuation_requires_fresh_recovery_cockpit=True,
            next_continuation_requires_fresh_local_safe_step=True,
            next_continuation_requires_fresh_review_token=True,
        )
        readiness_candidate.update(
            {
                key: False
                for key in [
                    "action_allowed_now",
                    "executable_tool_action_emitted",
                    "can_emit_executable_tool_action",
                    "can_auto_execute_now",
                    "timebox_authorizes_execution",
                    "timebox_authorizes_local_safe_step",
                    "timebox_authorizes_risky_work",
                    "timebox_authorizes_approval",
                    "timebox_authorizes_timebox_reuse",
                    "timebox_authorizes_model_call",
                    "timebox_authorizes_tool_execution",
                    "timebox_authorizes_personal_data_read",
                    "timebox_authorizes_external_side_effect",
                    "timebox_reusable_for_next_step",
                    "awake_guard_authorizes_os_wake_lock",
                    "awake_guard_authorizes_shell_execution",
                    "awake_guard_authorizes_computer_control",
                    "awake_guard_authorizes_approval",
                    "awake_guard_authorizes_model_call",
                    "awake_guard_authorizes_tool_execution",
                    "awake_guard_authorizes_personal_data_read",
                    "awake_guard_authorizes_external_side_effect",
                    "awake_guard_reusable_for_next_timebox",
                    "awake_guard_caffeinate_command_authorized",
                    "awake_guard_keep_awake_command_authorized",
                    "awake_guard_authorizes_unattended_execution",
                    "awake_guard_authorizes_continuation_window",
                    "awake_guard_reusable_as_execution_permission",
                    "supersession_token_authorizes_execution",
                    "supersession_token_authorizes_local_safe_step",
                    "supersession_token_authorizes_risky_work",
                    "supersession_token_authorizes_approval",
                    "supersession_token_authorizes_recovery_followthrough",
                    "supersession_token_authorizes_timebox_override",
                    "supersession_token_authorizes_goal_override",
                    "supersession_token_authorizes_model_call",
                    "supersession_token_authorizes_tool_execution",
                    "supersession_token_authorizes_personal_data_read",
                    "supersession_token_authorizes_external_side_effect",
                    "supersession_token_reusable_for_next_review",
                    "supersession_token_reusable_for_next_timebox",
                    "one_step_execution_contract_token_authorizes_action_now",
                    "one_step_execution_contract_token_authorizes_risky_work",
                    "one_step_execution_contract_token_authorizes_unreviewed_followthrough",
                    "one_step_execution_contract_token_authorizes_batching",
                    "one_step_execution_contract_token_reusable_for_next_step",
                    "checkpoint_route_authorizes_continuation",
                    "checkpoint_route_authorizes_local_safe_step",
                    "checkpoint_route_authorizes_risky_work",
                    "checkpoint_route_authorizes_approval",
                    "checkpoint_route_authorizes_recovery_followthrough",
                    "checkpoint_route_authorizes_checkpoint_reuse",
                    "checkpoint_route_authorizes_model_call",
                    "checkpoint_route_authorizes_tool_execution",
                    "checkpoint_route_authorizes_personal_data_read",
                    "checkpoint_route_authorizes_external_side_effect",
                    "checkpoint_route_reusable_for_next_review",
                    "checkpoint_route_reusable_for_next_checkpoint",
                    "recovery_followthrough_token_reusable_for_future_recovery",
                    "recovery_followthrough_token_authorizes_resume_gate",
                    "recovery_followthrough_token_authorizes_next_step",
                    "recovery_followthrough_token_authorizes_risky_work",
                    "recovery_followthrough_token_authorizes_approval",
                    "recovery_followthrough_token_authorizes_model_call",
                    "recovery_followthrough_token_authorizes_tool_execution",
                    "recovery_followthrough_token_authorizes_personal_data_read",
                    "recovery_followthrough_token_authorizes_external_side_effect",
                    "local_safe_recovery_execution_token_authorizes_resume_gate",
                    "local_safe_recovery_execution_token_authorizes_next_step",
                    "local_safe_recovery_execution_token_authorizes_risky_work",
                    "local_safe_recovery_execution_token_authorizes_approval",
                    "local_safe_recovery_execution_token_authorizes_model_call",
                    "local_safe_recovery_execution_token_authorizes_tool_execution",
                    "local_safe_recovery_execution_token_authorizes_personal_data_read",
                    "local_safe_recovery_execution_token_authorizes_external_side_effect",
                    "local_safe_recovery_execution_token_reusable_for_future_recovery",
                    "recovery_execution_readiness_token_authorizes_resume_gate",
                    "recovery_execution_readiness_token_authorizes_next_step",
                    "recovery_execution_readiness_token_authorizes_risky_work",
                    "recovery_execution_readiness_token_authorizes_approval",
                    "recovery_execution_readiness_token_authorizes_model_call",
                    "recovery_execution_readiness_token_authorizes_tool_execution",
                    "recovery_execution_readiness_token_authorizes_personal_data_read",
                    "recovery_execution_readiness_token_authorizes_external_side_effect",
                    "recovery_execution_readiness_token_authorizes_unreviewed_followthrough",
                    "recovery_execution_readiness_token_reusable_for_future_recovery",
                    "carried_recovery_execution_authorizes_action_now",
                    "carried_recovery_execution_authorizes_risky_work",
                    "carried_recovery_execution_authorizes_unreviewed_followthrough",
                    "carried_recovery_execution_authorizes_model_call",
                    "carried_recovery_execution_authorizes_tool_execution",
                    "carried_recovery_execution_authorizes_personal_data_read",
                    "carried_recovery_execution_authorizes_external_side_effect",
                    "carried_next_step_approval_boundary_authorizes_action_now",
                    "carried_next_step_approval_boundary_authorizes_risky_work",
                    "carried_next_step_approval_boundary_authorizes_unreviewed_followthrough",
                    "carried_next_step_approval_boundary_reusable_for_next_review",
                    "continuation_review_token_reusable_for_next_review",
                    "previous_continuation_review_token_reusable_for_next_review",
                    "fresh_continuation_review_boundary_authorizes_action_now",
                    "fresh_continuation_review_boundary_authorizes_local_safe_step",
                    "fresh_continuation_review_boundary_authorizes_risky_work",
                    "fresh_continuation_review_boundary_authorizes_unreviewed_followthrough",
                    "fresh_continuation_review_boundary_authorizes_timebox_reuse",
                    "fresh_continuation_review_boundary_authorizes_checkpoint_reuse",
                    "fresh_continuation_review_boundary_authorizes_token_reuse",
                    "fresh_continuation_review_boundary_authorizes_model_call",
                    "fresh_continuation_review_boundary_authorizes_tool_execution",
                    "fresh_continuation_review_boundary_authorizes_personal_data_read",
                    "fresh_continuation_review_boundary_authorizes_external_side_effect",
                    "fresh_continuation_review_boundary_reusable_for_next_review",
                    "step_closure_receipt_authorizes_action_now",
                    "step_closure_receipt_authorizes_local_safe_step",
                    "step_closure_receipt_authorizes_risky_work",
                    "step_closure_receipt_authorizes_new_cycle",
                    "step_closure_receipt_authorizes_unreviewed_followthrough",
                    "step_closure_receipt_authorizes_timebox_reuse",
                    "step_closure_receipt_authorizes_checkpoint_reuse",
                    "step_closure_receipt_authorizes_model_call",
                    "step_closure_receipt_authorizes_tool_execution",
                    "step_closure_receipt_authorizes_personal_data_read",
                    "step_closure_receipt_authorizes_external_side_effect",
                    "step_closure_receipt_reusable_for_next_review",
                    "step_closure_receipt_reusable_for_next_cycle",
                    "prior_step_closure_authorizes_followup",
                    "prior_step_closure_reusable_for_next_step",
                    "prior_cycle_ledger_token_reusable_for_this_closure",
                    "prior_cycle_ledger_proof_authorizes_post_step_closure",
                    "prior_cycle_ledger_proof_authorizes_model_call",
                    "prior_cycle_ledger_proof_authorizes_tool_execution",
                    "prior_cycle_ledger_proof_authorizes_personal_data_read",
                    "prior_cycle_ledger_proof_authorizes_external_side_effect",
                ]
            }
        )
        ready = _autonomy_step_closure_ready(readiness_candidate)
        if not ready and closure_gate["ready_for_next_continuation_review"]:
            closure_gate["state"] = "AUTONOMY_STEP_CLOSURE_HELD"
            closure_gate["ready_for_next_continuation_review"] = False
            closure_gate["missing"].append("step_closure_ready_contract")
            closure_gate["missing"] = list(dict.fromkeys(closure_gate["missing"]))
            closure_gate["missing_count"] = len(closure_gate["missing"])
            closure_gate["next_required_command"] = closure_gate["required_commands"][0]
            next_safe_command = "autonomy step closure: step=<completed local-safe step> verification=<post-step evidence> receipt=<post-step receipt> checkpoint=<fresh checkpoint> health=<execution health> audit=<execution audit> learning=<after-action learning>"
        lines = [
            "Jarvis autonomy step closure packet:",
            "This is read-only. It proves whether one previously allowed local-safe continuation step has enough post-step evidence before another autonomous continuation step is reviewed.",
            "",
            "Objective:",
            f"- {objective[:300]}",
            "",
            "Closure state:",
            f"- state: {closure_gate['state']}",
            f"- ready for next continuation review: {'yes' if ready else 'no'}",
            f"- missing blockers: {', '.join(closure_gate['missing']) if closure_gate['missing'] else 'none'}",
            f"- next safe command: `{next_safe_command}`",
            "",
            "Measured step closure readiness scorecard:",
            f"- score: {step_closure_score}/{step_closure_max_score}",
            f"- required rows ready: {'yes' if step_closure_required_rows_ready else 'no'}",
            f"- scorecard ready: {'yes' if step_closure_scorecard_ready else 'no'}",
            *[
                f"- {row['item']}: {row['points']}/{row['max_points']} ({'ready' if row['ready'] else 'held'}; does not authorize action, risky work, or follow-up without fresh review)"
                for row in step_closure_scorecard_rows
            ],
            "",
            "Bound pre-step permission:",
            f"- timebox state: {continuation_metadata.get('timebox_state')}",
            f"- timebox receipt sha256: {continuation_metadata.get('timebox_receipt_sha256') or 'missing'}",
            f"- timebox review contract ready: {'yes' if timebox_review_contract_ready else 'no'}",
            f"- timebox review rows: {len(timebox_review_contract_rows)}",
            f"- cockpit state: {continuation_metadata.get('cockpit_state')}",
            f"- resume gate state: {continuation_metadata.get('resume_gate_state')}",
            f"- follow-through state: {continuation_metadata.get('followthrough_state')}",
            f"- continuation state: {continuation_metadata.get('continuation_state')}",
            f"- one local-safe step allowed: {'yes' if continuation_metadata.get('one_local_safe_step_allowed') else 'no'}",
            f"- one-step execution contract token sha256: {one_step_execution_contract_token_sha256 or 'missing'}",
            f"- one-step execution contract token carried as prior proof: {'yes' if one_step_execution_contract_token_as_prior_proof else 'no'}",
            "- one-step execution contract token authorizes action now: no",
            "- one-step execution contract token authorizes risky work: no",
            "- one-step execution contract token reusable for next step: no",
            f"- operator supersession state: {continuation_metadata.get('supersession_state')}",
            f"- latest instruction can govern closure: {'yes' if continuation_metadata.get('can_continue_under_latest_instruction') else 'no'}",
            f"- supersession token sha256: {continuation_metadata.get('supersession_token_sha256') or 'missing'}",
            f"- supersession token boundary rows: {len(supersession_token_boundary_rows)}",
            "- supersession token authorizes post-step closure: no",
            "- supersession token authorizes follow-up: no",
            "- supersession token authorizes risky work: no",
            "- supersession token reusable for next review: no",
            f"- recovery artifact hashes supplied: {'yes' if continuation_metadata.get('recovery_artifact_hashes_present') else 'no'}",
            f"- recovery follow-through token sha256: {continuation_metadata.get('recovery_followthrough_token_sha256') or 'missing'}",
            "- recovery follow-through token reusable for future recovery: no",
            f"- local-safe recovery execution token sha256: {continuation_metadata.get('local_safe_recovery_execution_token_sha256') or 'missing'}",
            "- local-safe recovery execution token authorizes resume gate: no",
            "- local-safe recovery execution token authorizes next step: no",
            "- local-safe recovery execution token authorizes risky work: no",
            "- local-safe recovery execution token authorizes approval: no",
            "- local-safe recovery execution token authorizes model call: no",
            "- local-safe recovery execution token authorizes tool execution: no",
            "- local-safe recovery execution token authorizes personal-data read: no",
            "- local-safe recovery execution token authorizes external side effect: no",
            "- local-safe recovery execution token reusable for future recovery: no",
            f"- local-safe recovery execution token boundary rows: {len(local_safe_recovery_execution_token_boundary_rows)}",
            "",
            "Carried timebox review contract:",
            "- authorizes execution now: no",
            "- authorizes local-safe step: no",
            "- authorizes risky work: no",
            "- authorizes approval: no",
            "- reusable for next step: no",
            *[
                f"- {row['item']}: fresh required yes; prior reusable no; authorizes timebox reuse no"
                for row in timebox_review_contract_rows
            ],
            "",
            "Carried awake guard boundary:",
            f"- awake requested: {'yes' if continuation_metadata.get('awake_guard_requested') else 'no'}",
            f"- awake guard token sha256: {continuation_metadata.get('awake_guard_token_sha256') or 'missing'}",
            f"- boundary rows: {continuation_metadata.get('awake_guard_boundary_row_count') or 0}",
            "- authorizes OS wake lock: no",
            "- authorizes shell execution: no",
            "- authorizes computer control: no",
            "- authorizes approval: no",
            "- reusable for next timebox: no",
            "- fresh awake-guard review required next timebox: yes",
            "",
            "Carried operator supersession boundary:",
            f"- boundary ready: {'yes' if supersession_token_boundary_ready else 'no'}",
            f"- boundary rows: {len(supersession_token_boundary_rows)}",
            *[
                f"- {row['item']}: {row['status']}; authorizes execution no; local-safe step no; risky work no; approval no; recovery follow-through no; reusable no"
                for row in supersession_token_boundary_rows
            ],
            "- next supersession requires fresh latest-instruction review: yes",
            "",
            "Carried recovery execution readiness proof:",
            f"- score: {carried_recovery_execution_score}/{carried_recovery_execution_max_score}",
            f"- required rows ready: {'yes' if carried_recovery_execution_required_rows_ready else 'no'}",
            f"- carried as prior proof: {'yes' if carried_recovery_execution_as_prior_proof else 'no'}",
            "- authorizes action now: no",
            "- authorizes risky work: no",
            "- authorizes unreviewed follow-through: no",
            *[
                f"- {row['item']}: {row['points']}/{row['max_points']} ({'ready' if row['ready'] else 'held'}; prior recovery proof only)"
                for row in carried_recovery_execution_scorecard_rows
            ],
            "",
            "Carried risky next-step approval proof:",
            f"- approval required before review: {'yes' if carried_next_step_approval_required_before_review else 'no'}",
            f"- proof queue count: {len(carried_next_step_approval_proof_queue)}",
            f"- boundary token sha256: {carried_next_step_approval_boundary_token_sha256 or 'missing'}",
            f"- boundary rows: {len(carried_next_step_approval_boundary_rows)}",
            f"- boundary ready: {'yes' if carried_next_step_approval_boundary_ready else 'no'}",
            f"- carried as prior proof: {'yes' if carried_next_step_approval_boundary_as_prior_proof else 'no'}",
            "- authorizes action now: no",
            "- authorizes risky work: no",
            "- authorizes unreviewed follow-through: no",
            *[f"- `{command}`" for command in carried_next_step_approval_proof_queue],
            *[
                f"- {row['item']}: {row.get('status', 'held')}; prior approval-boundary proof only"
                for row in carried_next_step_approval_boundary_rows
            ],
            "",
            f"- proposed next step: {(proposed_next_step or completed_step)[:500] or 'missing'}",
            f"- proposed next step sha256: {proposed_step_sha256 or 'missing'}",
            f"- proposed verification target: {(proposed_verification or post_step_verification)[:500] or 'missing'}",
            f"- proposed verification sha256: {proposed_verification_sha256 or 'missing'}",
            f"- continuation review token sha256: {continuation_review_token_sha256 or 'missing'}",
            "- continuation review token reusable for next review: no",
            f"- prior cycle ledger token sha256: {continuation_metadata.get('prior_cycle_ledger_token_sha256') or 'not supplied'}",
            f"- prior cycle ledger token present: {'yes' if continuation_metadata.get('prior_cycle_ledger_token_present') else 'no'}",
            f"- prior cycle ledger boundary rows: {len(prior_cycle_ledger_token_boundary_rows)}",
            f"- prior cycle ledger boundary ready: {'yes' if prior_cycle_ledger_token_boundary_ready else 'no'}",
            *[
                f"- {row['item']}: {row['status']}; authorizes action now no; authorizes post-step closure no; authorizes local-safe step no; authorizes risky work no; authorizes model/tool/private/external no; reusable no"
                for row in prior_cycle_ledger_token_boundary_rows
            ],
            "- prior cycle ledger token reusable for this closure: no",
            "- prior cycle ledger proof authorizes post-step closure: no",
            "- fresh cycle ledger token required after this closure: yes",
            "",
            "Post-step evidence:",
            f"- completed step: {completed_step[:500] or 'missing'}",
            f"- completed step sha256: {completed_step_sha256 or 'missing'}",
            f"- completed step matches allowed step: {'yes' if completed_step_matches_proposed else 'no'}",
            f"- verification evidence: {post_step_verification[:500] or 'missing'}",
            f"- verification evidence sha256: {post_step_verification_sha256 or 'missing'}",
            f"- verification evidence matches allowed target: {'yes' if post_step_verification_matches_proposed else 'no'}",
            f"- post-step receipt: {post_step_receipt_path or 'missing'}",
            f"- post-step receipt sha256: {post_step_receipt_sha256 or 'missing'}",
            f"- post-step receipt file sha256: {post_step_receipt_file_sha256 or 'missing'}",
            f"- post-step receipt hash matches file: {'yes' if post_step_receipt_hash_matches_file else 'no'}",
            f"- fresh checkpoint: {post_step_checkpoint_path or 'missing'}",
            f"- fresh checkpoint sha256: {post_step_checkpoint_sha256 or 'missing'}",
            f"- fresh checkpoint file sha256: {post_step_checkpoint_file_sha256 or 'missing'}",
            f"- fresh checkpoint hash matches file: {'yes' if post_step_checkpoint_hash_matches_file else 'no'}",
            f"- post-step artifact hashes supplied: {'yes' if _looks_like_sha256(post_step_receipt_sha256) and _looks_like_sha256(post_step_checkpoint_sha256) else 'no'}",
            f"- step closure receipt token sha256: {step_closure_receipt_token_sha256 or 'missing'}",
            f"- step closure receipt boundary rows: {len(step_closure_receipt_boundary_rows)}",
            f"- step closure receipt boundary ready: {'yes' if step_closure_receipt_boundary_ready else 'no'}",
            "- step closure receipt authorizes action now: no",
            "- step closure receipt authorizes local-safe step: no",
            "- step closure receipt authorizes risky work: no",
            "- step closure receipt authorizes unreviewed follow-through: no",
            "- step closure receipt reusable for next review: no",
            *[
                f"- {row['item']}: {row['status']}; authorizes action now no; local-safe step no; risky work no; reusable no"
                for row in step_closure_receipt_boundary_rows
            ],
            f"- execution health: {execution_health[:300] or 'missing'}",
            f"- execution audit: {execution_audit[:300] or 'missing'}",
            f"- after-action learning: {after_action_learning[:300] or 'missing'}",
            f"- blockers: {blockers_text or 'none reported'}",
            "",
            "Required commands:",
            *[f"- `{command}`" for command in closure_gate["required_commands"]],
            "",
            "Fresh continuation review contract:",
            f"- contract enforced: {'yes' if fresh_continuation_review_contract_enforced else 'no'}",
            f"- row count: {len(fresh_continuation_review_contract_rows)}",
            "- summary: " + ", ".join(fresh_continuation_review_contract_summary),
            f"- fresh continuation review boundary token sha256: {fresh_continuation_review_boundary_token_sha256 or 'missing'}",
            f"- fresh continuation review boundary rows: {len(fresh_continuation_review_boundary_token_rows)}",
            f"- fresh continuation review boundary ready: {'yes' if fresh_continuation_review_boundary_token_ready else 'no'}",
            "- fresh continuation review boundary authorizes action now: no",
            "- fresh continuation review boundary authorizes local-safe step: no",
            "- fresh continuation review boundary authorizes risky work: no",
            "- fresh continuation review boundary authorizes unreviewed follow-through: no",
            "- fresh continuation review boundary reusable for next review: no",
            *[
                f"- {row['item']}: fresh required {'yes' if row['fresh_required'] else 'no'}; prior reusable no; authorizes action now no; authorizes risky work no; source {row['source']}"
                for row in fresh_continuation_review_contract_rows
            ],
            *[
                f"- {row['item']}: {row['status']}; fresh required yes; reusable next review no; authorizes action now no; local-safe step no; risky work no"
                for row in fresh_continuation_review_boundary_token_rows
            ],
            "",
            "Approval boundary:",
            f"- {closure_gate['approval_boundary']}",
            "",
            "Boundary:",
            "- read-only closure packet; it does not execute another step, run shell/code, control the computer, read private data, write notes, approve requests, rerun actions, or queue approvals.",
        ]
        return ToolResult(
            "autonomy_step_closure_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                objective=objective,
                objective_length=len(objective),
                closure_state=closure_gate["state"],
                ready_for_next_continuation_review=ready,
                next_safe_command=next_safe_command,
                missing_blockers=closure_gate["missing"],
                missing_blocker_count=closure_gate["missing_count"],
                required_evidence=closure_gate["required_evidence"],
                required_evidence_count=len(closure_gate["required_evidence"]),
                required_commands=closure_gate["required_commands"],
                required_command_count=len(closure_gate["required_commands"]),
                next_required_command=closure_gate["next_required_command"],
                proof_queue=closure_gate["required_commands"],
                proof_queue_count=len(closure_gate["required_commands"]),
                next_proof_command=closure_gate["next_required_command"],
                step_closure_readiness_score=step_closure_score,
                step_closure_readiness_max_score=step_closure_max_score,
                step_closure_readiness_scorecard_rows=step_closure_scorecard_rows,
                step_closure_readiness_scorecard_row_count=len(step_closure_scorecard_rows),
                step_closure_readiness_required_rows_ready=step_closure_required_rows_ready,
                step_closure_readiness_scorecard_ready=step_closure_scorecard_ready,
                step_closure_ready_contract_ready=ready,
                resume_gate_state=continuation_metadata.get("resume_gate_state"),
                timebox_state=continuation_metadata.get("timebox_state"),
                can_continue_now=continuation_metadata.get("can_continue_now"),
                should_stop_now=continuation_metadata.get("should_stop_now"),
                missing_timebox_proof=continuation_metadata.get("missing_timebox_proof", []),
                missing_timebox_proof_count=continuation_metadata.get("missing_timebox_proof_count", 0),
                timebox_receipt_sha256=continuation_metadata.get("timebox_receipt_sha256", ""),
                timebox_receipt_present=continuation_metadata.get("timebox_receipt_present", False),
                timebox_review_contract_rows=timebox_review_contract_rows,
                timebox_review_contract_row_count=len(timebox_review_contract_rows),
                timebox_review_contract_ready=timebox_review_contract_ready,
                timebox_review_contract_summary=continuation_metadata.get("timebox_review_contract_summary", []),
                timebox_authorizes_execution=False,
                timebox_authorizes_local_safe_step=False,
                timebox_authorizes_risky_work=False,
                timebox_authorizes_approval=False,
                timebox_authorizes_timebox_reuse=False,
                timebox_authorizes_model_call=False,
                timebox_authorizes_tool_execution=False,
                timebox_authorizes_personal_data_read=False,
                timebox_authorizes_external_side_effect=False,
                timebox_reusable_for_next_step=False,
                next_step_requires_fresh_timebox=True,
                awake_guard_requested=continuation_metadata.get("awake_guard_requested", False),
                awake_guard_token_sha256=continuation_metadata.get("awake_guard_token_sha256", ""),
                awake_guard_token_present=continuation_metadata.get("awake_guard_token_present", False),
                awake_guard_boundary_rows=continuation_metadata.get("awake_guard_boundary_rows", []),
                awake_guard_boundary_row_count=continuation_metadata.get("awake_guard_boundary_row_count", 0),
                awake_guard_authorizes_os_wake_lock=False,
                awake_guard_authorizes_shell_execution=False,
                awake_guard_authorizes_computer_control=False,
                awake_guard_authorizes_approval=False,
                awake_guard_authorizes_model_call=False,
                awake_guard_authorizes_tool_execution=False,
                awake_guard_authorizes_personal_data_read=False,
                awake_guard_authorizes_external_side_effect=False,
                awake_guard_reusable_for_next_timebox=False,
                awake_guard_requires_separate_operator_request=True,
                awake_guard_requires_separate_shell_approval=True,
                awake_guard_os_wake_lock_boundary_ready=_awake_guard_boundary_ready_from_metadata(
                    continuation_metadata,
                    default_objective=objective,
                    default_timezone=timezone_label,
                ),
                awake_guard_caffeinate_command_authorized=False,
                awake_guard_keep_awake_command_authorized=False,
                awake_guard_authorizes_unattended_execution=False,
                awake_guard_authorizes_continuation_window=False,
                awake_guard_reusable_as_execution_permission=False,
                next_awake_guard_requires_fresh_review=True,
                previous_instruction=continuation_metadata.get("previous_instruction", ""),
                latest_instruction=continuation_metadata.get("latest_instruction", ""),
                latest_instruction_present=continuation_metadata.get("latest_instruction_present", False),
                latest_instruction_supersedes_previous=continuation_metadata.get("latest_instruction_supersedes_previous", False),
                latest_instruction_is_stop=continuation_metadata.get("latest_instruction_is_stop", False),
                latest_instruction_is_continue=continuation_metadata.get("latest_instruction_is_continue", False),
                supersession_state=continuation_metadata.get("supersession_state", ""),
                can_continue_under_latest_instruction=continuation_metadata.get("can_continue_under_latest_instruction", False),
                newest_instruction_overrides_automation=continuation_metadata.get("newest_instruction_overrides_automation", True),
                newest_instruction_overrides_goal=continuation_metadata.get("newest_instruction_overrides_goal", True),
                newest_instruction_overrides_recovery_queue=continuation_metadata.get("newest_instruction_overrides_recovery_queue", True),
                stop_or_pause_blocks_autonomy=continuation_metadata.get("stop_or_pause_blocks_autonomy", False),
                supersession_token_sha256=continuation_metadata.get("supersession_token_sha256", ""),
                supersession_token_present=continuation_metadata.get("supersession_token_present", False),
                supersession_token_boundary_rows=supersession_token_boundary_rows,
                supersession_token_boundary_row_count=len(supersession_token_boundary_rows),
                supersession_token_boundary_ready=supersession_token_boundary_ready,
                supersession_token_authorizes_execution=False,
                supersession_token_authorizes_local_safe_step=False,
                supersession_token_authorizes_risky_work=False,
                supersession_token_authorizes_approval=False,
                supersession_token_authorizes_recovery_followthrough=False,
                supersession_token_authorizes_timebox_override=False,
                supersession_token_authorizes_goal_override=False,
                supersession_token_authorizes_model_call=False,
                supersession_token_authorizes_tool_execution=False,
                supersession_token_authorizes_personal_data_read=False,
                supersession_token_authorizes_external_side_effect=False,
                supersession_token_reusable_for_next_review=False,
                supersession_token_reusable_for_next_timebox=False,
                next_supersession_requires_fresh_latest_instruction_review=True,
                cockpit_state=continuation_metadata.get("cockpit_state"),
                followthrough_state=continuation_metadata.get("followthrough_state"),
                continuation_state=continuation_metadata.get("continuation_state"),
                pre_step_one_local_safe_step_allowed=continuation_metadata.get("one_local_safe_step_allowed"),
                continuation_missing_blockers=continuation_metadata.get("missing_blockers", []),
                one_step_execution_contract_rows=one_step_execution_contract_rows,
                one_step_execution_contract_row_count=len(one_step_execution_contract_rows),
                one_step_execution_contract_ready=_metadata_flag_ready(
                    continuation_metadata, "one_step_execution_contract_ready"
                ),
                one_step_execution_contract_token_sha256=one_step_execution_contract_token_sha256,
                one_step_execution_contract_token_present=_looks_like_sha256(one_step_execution_contract_token_sha256),
                one_step_execution_contract_token_as_prior_proof=one_step_execution_contract_token_as_prior_proof,
                one_step_execution_contract_binds_awake_guard=_metadata_flag_ready(
                    continuation_metadata, "one_step_execution_contract_binds_awake_guard"
                ),
                one_step_execution_contract_binds_operator_supersession=_metadata_flag_ready(
                    continuation_metadata, "one_step_execution_contract_binds_operator_supersession"
                ),
                one_step_execution_contract_binds_timebox_review_contract=_metadata_flag_ready(
                    continuation_metadata, "one_step_execution_contract_binds_timebox_review_contract"
                ),
                one_step_execution_contract_token_authorizes_action_now=False,
                one_step_execution_contract_token_authorizes_risky_work=False,
                one_step_execution_contract_token_authorizes_unreviewed_followthrough=False,
                one_step_execution_contract_token_authorizes_batching=False,
                one_step_execution_contract_token_reusable_for_next_step=False,
                next_step_requires_new_one_step_execution_contract_token=True,
                one_step_execution_contract_all_local_safe_step_limited=_metadata_flag_ready(
                    continuation_metadata, "one_step_execution_contract_all_local_safe_step_limited"
                ),
                one_step_execution_contract_all_non_reusable=_metadata_flag_ready(
                    continuation_metadata, "one_step_execution_contract_all_non_reusable"
                ),
                one_step_execution_contract_all_risky_work_gated=_metadata_flag_ready(
                    continuation_metadata, "one_step_execution_contract_all_risky_work_gated"
                ),
                one_step_execution_contract_requires_fresh_closure=_metadata_flag_ready(
                    continuation_metadata, "one_step_execution_contract_requires_fresh_closure"
                ),
                one_step_execution_contract_authorizes_batching=False,
                one_step_execution_contract_authorizes_followup_without_closure=False,
                latest_checkpoint_path=continuation_metadata.get("latest_checkpoint_path", ""),
                latest_checkpoint_sha256=continuation_metadata.get("latest_checkpoint_sha256", ""),
                latest_checkpoint_hash_present=continuation_metadata.get("latest_checkpoint_hash_present", False),
                supplied_checkpoint_path=continuation_metadata.get("supplied_checkpoint_path", ""),
                checkpoint_path_matches_latest=continuation_metadata.get("checkpoint_path_matches_latest"),
                checkpoint_hash_matches_latest=continuation_metadata.get("checkpoint_hash_matches_latest"),
                checkpoint_freshness=continuation_metadata.get("checkpoint_freshness"),
                checkpoint_needs_review=continuation_metadata.get("checkpoint_needs_review", True),
                checkpoint_route_token_sha256=continuation_metadata.get("checkpoint_route_token_sha256", ""),
                checkpoint_route_token_present=continuation_metadata.get("checkpoint_route_token_present", False),
                checkpoint_route_boundary_rows=checkpoint_route_boundary_rows,
                checkpoint_route_boundary_row_count=len(checkpoint_route_boundary_rows),
                checkpoint_route_boundary_ready=checkpoint_route_boundary_ready,
                checkpoint_route_authorizes_continuation=False,
                checkpoint_route_authorizes_local_safe_step=False,
                checkpoint_route_authorizes_risky_work=False,
                checkpoint_route_authorizes_approval=False,
                checkpoint_route_authorizes_recovery_followthrough=False,
                checkpoint_route_authorizes_checkpoint_reuse=False,
                checkpoint_route_authorizes_model_call=False,
                checkpoint_route_authorizes_tool_execution=False,
                checkpoint_route_authorizes_personal_data_read=False,
                checkpoint_route_authorizes_external_side_effect=False,
                checkpoint_route_reusable_for_next_review=False,
                checkpoint_route_reusable_for_next_checkpoint=False,
                next_checkpoint_route_requires_fresh_recovery_review=True,
                checkpoint_route_recovery_proof_queue=continuation_metadata.get("checkpoint_route_recovery_proof_queue", []),
                checkpoint_route_recovery_proof_queue_count=continuation_metadata.get("checkpoint_route_recovery_proof_queue_count", 0),
                checkpoint_route_next_recovery_command=continuation_metadata.get("checkpoint_route_next_recovery_command", ""),
                checkpoint_recovery_proof_queue=continuation_metadata.get("checkpoint_recovery_proof_queue", []),
                checkpoint_recovery_proof_queue_count=continuation_metadata.get("checkpoint_recovery_proof_queue_count", 0),
                checkpoint_recovery_next_proof_command=continuation_metadata.get("checkpoint_recovery_next_proof_command", ""),
                receipt_sha256=continuation_metadata.get("receipt_sha256", ""),
                receipt_hash_provided=continuation_metadata.get("receipt_hash_provided", False),
                receipt_file_sha256=continuation_metadata.get("receipt_file_sha256", ""),
                receipt_file_hash_present=continuation_metadata.get("receipt_file_hash_present", False),
                receipt_hash_matches_file=continuation_metadata.get("receipt_hash_matches_file", False),
                checkpoint_sha256=continuation_metadata.get("checkpoint_sha256", ""),
                checkpoint_hash_provided=continuation_metadata.get("checkpoint_hash_provided", False),
                recovery_checkpoint_file_sha256=continuation_metadata.get("checkpoint_file_sha256", ""),
                recovery_checkpoint_file_hash_present=continuation_metadata.get("checkpoint_file_hash_present", False),
                recovery_checkpoint_hash_matches_file=continuation_metadata.get("recovery_checkpoint_hash_matches_file", False),
                recovery_artifact_hashes_present=continuation_metadata.get("recovery_artifact_hashes_present", False),
                recovery_artifact_hashes_match_files=continuation_metadata.get("recovery_artifact_hashes_match_files", False),
                recovery_followthrough_token_sha256=continuation_metadata.get("recovery_followthrough_token_sha256", ""),
                recovery_followthrough_token_present=continuation_metadata.get("recovery_followthrough_token_present", False),
                recovery_followthrough_token_boundary_rows=recovery_followthrough_token_boundary_rows,
                recovery_followthrough_token_boundary_row_count=len(recovery_followthrough_token_boundary_rows),
                recovery_followthrough_token_boundary_ready=recovery_followthrough_token_boundary_ready,
                recovery_followthrough_token_reusable_for_future_recovery=False,
                recovery_followthrough_token_authorizes_resume_gate=False,
                recovery_followthrough_token_authorizes_next_step=False,
                recovery_followthrough_token_authorizes_risky_work=False,
                recovery_followthrough_token_authorizes_approval=False,
                recovery_followthrough_token_authorizes_model_call=False,
                recovery_followthrough_token_authorizes_tool_execution=False,
                recovery_followthrough_token_authorizes_personal_data_read=False,
                recovery_followthrough_token_authorizes_external_side_effect=False,
                next_recovery_followthrough_requires_new_token=True,
                local_safe_recovery_execution_token_sha256=continuation_metadata.get("local_safe_recovery_execution_token_sha256", ""),
                local_safe_recovery_execution_token_present=continuation_metadata.get("local_safe_recovery_execution_token_present", False),
                local_safe_recovery_execution_token_authorizes_resume_gate=False,
                local_safe_recovery_execution_token_authorizes_next_step=False,
                local_safe_recovery_execution_token_authorizes_risky_work=False,
                local_safe_recovery_execution_token_authorizes_approval=False,
                local_safe_recovery_execution_token_authorizes_model_call=False,
                local_safe_recovery_execution_token_authorizes_tool_execution=False,
                local_safe_recovery_execution_token_authorizes_personal_data_read=False,
                local_safe_recovery_execution_token_authorizes_external_side_effect=False,
                local_safe_recovery_execution_token_reusable_for_future_recovery=False,
                next_recovery_execution_requires_new_local_safe_token=True,
                local_safe_recovery_execution_token_boundary_rows=local_safe_recovery_execution_token_boundary_rows,
                local_safe_recovery_execution_token_boundary_row_count=len(local_safe_recovery_execution_token_boundary_rows),
                local_safe_recovery_execution_token_boundary_ready=local_safe_recovery_execution_token_boundary_ready,
                recovery_execution_readiness_token_sha256=continuation_metadata.get("recovery_execution_readiness_token_sha256", ""),
                recovery_execution_readiness_token_present=continuation_metadata.get("recovery_execution_readiness_token_present", False),
                recovery_execution_readiness_token_boundary_rows=recovery_execution_readiness_token_boundary_rows,
                recovery_execution_readiness_token_boundary_row_count=len(recovery_execution_readiness_token_boundary_rows),
                recovery_execution_readiness_token_boundary_ready=recovery_execution_readiness_token_boundary_ready,
                recovery_execution_readiness_token_authorizes_resume_gate=False,
                recovery_execution_readiness_token_authorizes_next_step=False,
                recovery_execution_readiness_token_authorizes_risky_work=False,
                recovery_execution_readiness_token_authorizes_approval=False,
                recovery_execution_readiness_token_authorizes_model_call=False,
                recovery_execution_readiness_token_authorizes_tool_execution=False,
                recovery_execution_readiness_token_authorizes_personal_data_read=False,
                recovery_execution_readiness_token_authorizes_external_side_effect=False,
                recovery_execution_readiness_token_authorizes_unreviewed_followthrough=False,
                recovery_execution_readiness_token_reusable_for_future_recovery=False,
                next_recovery_execution_requires_new_readiness_token=True,
                carried_recovery_execution_score=carried_recovery_execution_score,
                carried_recovery_execution_max_score=carried_recovery_execution_max_score,
                carried_recovery_execution_scorecard_rows=carried_recovery_execution_scorecard_rows,
                carried_recovery_execution_scorecard_row_count=len(carried_recovery_execution_scorecard_rows),
                carried_recovery_execution_required_rows_ready=carried_recovery_execution_required_rows_ready,
                carried_recovery_execution_scorecard_ready=carried_recovery_execution_scorecard_ready,
                carried_recovery_execution_as_prior_proof=carried_recovery_execution_as_prior_proof,
                carried_recovery_execution_authorizes_action_now=False,
                carried_recovery_execution_authorizes_risky_work=False,
                carried_recovery_execution_authorizes_unreviewed_followthrough=False,
                carried_recovery_execution_authorizes_model_call=False,
                carried_recovery_execution_authorizes_tool_execution=False,
                carried_recovery_execution_authorizes_personal_data_read=False,
                carried_recovery_execution_authorizes_external_side_effect=False,
                carried_next_step_approval_proof_queue=carried_next_step_approval_proof_queue,
                carried_next_step_approval_proof_queue_count=len(carried_next_step_approval_proof_queue),
                carried_next_step_next_approval_proof_command=(
                    carried_next_step_approval_proof_queue[0] if carried_next_step_approval_proof_queue else ""
                ),
                carried_next_step_approval_required_before_review=carried_next_step_approval_required_before_review,
                carried_next_step_approval_boundary_rows=carried_next_step_approval_boundary_rows,
                carried_next_step_approval_boundary_row_count=len(carried_next_step_approval_boundary_rows),
                carried_next_step_approval_boundary_ready=carried_next_step_approval_boundary_ready,
                carried_next_step_approval_boundary_token_sha256=carried_next_step_approval_boundary_token_sha256,
                carried_next_step_approval_boundary_token_present=_looks_like_sha256(carried_next_step_approval_boundary_token_sha256),
                carried_next_step_approval_boundary_as_prior_proof=carried_next_step_approval_boundary_as_prior_proof,
                carried_next_step_approval_boundary_authorizes_action_now=False,
                carried_next_step_approval_boundary_authorizes_risky_work=False,
                carried_next_step_approval_boundary_authorizes_unreviewed_followthrough=False,
                carried_next_step_approval_boundary_authorizes_approval=False,
                carried_next_step_approval_boundary_authorizes_model_call=False,
                carried_next_step_approval_boundary_authorizes_tool_execution=False,
                carried_next_step_approval_boundary_authorizes_personal_data_read=False,
                carried_next_step_approval_boundary_authorizes_external_side_effect=False,
                carried_next_step_approval_boundary_authorizes_timebox_reuse=False,
                carried_next_step_approval_boundary_reusable_for_next_review=False,
                carried_next_step_approval_boundary_reusable_for_recovery_review=False,
                continuation_post_step_proof_queue=continuation_metadata.get("post_step_proof_queue", []),
                continuation_post_step_proof_queue_count=continuation_metadata.get("post_step_proof_queue_count", 0),
                continuation_post_step_next_proof_command=continuation_metadata.get("post_step_next_proof_command", ""),
                proposed_next_step=proposed_step_from_continuation[:500],
                proposed_next_step_sha256=proposed_step_sha256,
                proposed_next_step_risk_signals=continuation_metadata.get("proposed_next_step_risk_signals", []),
                proposed_next_step_risk_signal_count=continuation_metadata.get("proposed_next_step_risk_signal_count", 0),
                proposed_verification=proposed_verification_from_continuation[:500],
                proposed_verification_sha256=proposed_verification_sha256,
                continuation_review_token_sha256=continuation_review_token_sha256,
                continuation_review_token_present=_looks_like_sha256(continuation_review_token_sha256),
                continuation_review_token_reusable_for_next_review=False,
                previous_continuation_review_token_reusable_for_next_review=False,
                next_review_requires_new_continuation_review_token=True,
                fresh_continuation_review_contract_rows=fresh_continuation_review_contract_rows,
                fresh_continuation_review_contract_row_count=len(fresh_continuation_review_contract_rows),
                fresh_continuation_review_contract_summary=fresh_continuation_review_contract_summary,
                fresh_continuation_review_contract_enforced=fresh_continuation_review_contract_enforced,
                fresh_continuation_review_boundary_token_sha256=fresh_continuation_review_boundary_token_sha256,
                fresh_continuation_review_boundary_token_present=_looks_like_sha256(fresh_continuation_review_boundary_token_sha256),
                fresh_continuation_review_boundary_token_rows=fresh_continuation_review_boundary_token_rows,
                fresh_continuation_review_boundary_token_row_count=len(fresh_continuation_review_boundary_token_rows),
                fresh_continuation_review_boundary_token_ready=fresh_continuation_review_boundary_token_ready,
                fresh_continuation_review_boundary_authorizes_action_now=False,
                fresh_continuation_review_boundary_authorizes_local_safe_step=False,
                fresh_continuation_review_boundary_authorizes_risky_work=False,
                fresh_continuation_review_boundary_authorizes_unreviewed_followthrough=False,
                fresh_continuation_review_boundary_authorizes_timebox_reuse=False,
                fresh_continuation_review_boundary_authorizes_checkpoint_reuse=False,
                fresh_continuation_review_boundary_authorizes_token_reuse=False,
                fresh_continuation_review_boundary_authorizes_model_call=False,
                fresh_continuation_review_boundary_authorizes_tool_execution=False,
                fresh_continuation_review_boundary_authorizes_personal_data_read=False,
                fresh_continuation_review_boundary_authorizes_external_side_effect=False,
                fresh_continuation_review_boundary_reusable_for_next_review=False,
                fresh_continuation_review_boundary_reusable_for_next_cycle=False,
                next_review_requires_new_fresh_continuation_review_boundary_token=True,
                step_closure_receipt_token_sha256=step_closure_receipt_token_sha256,
                step_closure_receipt_token_present=_looks_like_sha256(step_closure_receipt_token_sha256),
                step_closure_receipt_boundary_rows=step_closure_receipt_boundary_rows,
                step_closure_receipt_boundary_row_count=len(step_closure_receipt_boundary_rows),
                step_closure_receipt_boundary_ready=step_closure_receipt_boundary_ready,
                step_closure_receipt_authorizes_action_now=False,
                step_closure_receipt_authorizes_local_safe_step=False,
                step_closure_receipt_authorizes_risky_work=False,
                step_closure_receipt_authorizes_new_cycle=False,
                step_closure_receipt_authorizes_unreviewed_followthrough=False,
                step_closure_receipt_authorizes_timebox_reuse=False,
                step_closure_receipt_authorizes_checkpoint_reuse=False,
                step_closure_receipt_authorizes_model_call=False,
                step_closure_receipt_authorizes_tool_execution=False,
                step_closure_receipt_authorizes_personal_data_read=False,
                step_closure_receipt_authorizes_external_side_effect=False,
                step_closure_receipt_reusable_for_next_review=False,
                step_closure_receipt_reusable_for_next_cycle=False,
                next_review_requires_new_step_closure_receipt_token=True,
                next_continuation_requires_fresh_operator_timebox=True,
                next_continuation_requires_fresh_checkpoint=True,
                next_continuation_requires_fresh_recovery_cockpit=True,
                next_continuation_requires_fresh_local_safe_step=True,
                next_continuation_requires_fresh_review_token=True,
                prior_step_closure_authorizes_followup=False,
                prior_step_closure_reusable_for_next_step=False,
                prior_cycle_ledger_token_sha256=continuation_metadata.get("prior_cycle_ledger_token_sha256", ""),
                prior_cycle_ledger_token_present=continuation_metadata.get("prior_cycle_ledger_token_present", False),
                prior_cycle_ledger_token_boundary_rows=prior_cycle_ledger_token_boundary_rows,
                prior_cycle_ledger_token_boundary_row_count=len(prior_cycle_ledger_token_boundary_rows),
                prior_cycle_ledger_token_boundary_ready=prior_cycle_ledger_token_boundary_ready,
                prior_cycle_ledger_token_reusable_for_this_review=False,
                prior_cycle_ledger_token_reusable_for_this_closure=False,
                prior_cycle_ledger_token_reusable_for_this_cycle=False,
                prior_cycle_ledger_proof_authorizes_action_now=False,
                prior_cycle_ledger_proof_authorizes_post_step_closure=False,
                prior_cycle_ledger_proof_authorizes_new_action=False,
                prior_cycle_ledger_proof_authorizes_model_call=False,
                prior_cycle_ledger_proof_authorizes_tool_execution=False,
                prior_cycle_ledger_proof_authorizes_personal_data_read=False,
                prior_cycle_ledger_proof_authorizes_external_side_effect=False,
                next_step_requires_fresh_cycle_ledger_token=True,
                completed_step=completed_step[:500],
                completed_step_provided=bool(completed_step),
                completed_step_sha256=completed_step_sha256,
                completed_step_matches_proposed=completed_step_matches_proposed,
                post_step_verification=post_step_verification[:500],
                post_step_verification_provided=bool(post_step_verification),
                post_step_verification_sha256=post_step_verification_sha256,
                post_step_verification_matches_proposed=post_step_verification_matches_proposed,
                post_step_receipt_path=post_step_receipt_path,
                post_step_receipt_path_provided=bool(post_step_receipt_path),
                post_step_receipt_sha256=post_step_receipt_sha256,
                post_step_receipt_hash_provided=_looks_like_sha256(post_step_receipt_sha256),
                post_step_receipt_file_sha256=post_step_receipt_file_sha256,
                post_step_receipt_file_hash_present=_looks_like_sha256(post_step_receipt_file_sha256),
                post_step_receipt_hash_matches_file=post_step_receipt_hash_matches_file,
                post_step_checkpoint_path=post_step_checkpoint_path,
                post_step_checkpoint_path_provided=bool(post_step_checkpoint_path),
                post_step_checkpoint_sha256=post_step_checkpoint_sha256,
                post_step_checkpoint_hash_provided=_looks_like_sha256(post_step_checkpoint_sha256),
                post_step_checkpoint_file_sha256=post_step_checkpoint_file_sha256,
                post_step_checkpoint_file_hash_present=_looks_like_sha256(post_step_checkpoint_file_sha256),
                post_step_checkpoint_hash_matches_file=post_step_checkpoint_hash_matches_file,
                post_step_artifact_hashes_present=bool(_looks_like_sha256(post_step_receipt_sha256) and _looks_like_sha256(post_step_checkpoint_sha256)),
                post_step_artifact_hashes_match_files=bool(post_step_receipt_hash_matches_file and post_step_checkpoint_hash_matches_file),
                execution_health=execution_health[:500],
                execution_health_reviewed=bool(execution_health),
                execution_audit=execution_audit[:500],
                execution_audit_reviewed=bool(execution_audit),
                after_action_learning=after_action_learning[:500],
                after_action_learning_reviewed=bool(after_action_learning),
                blockers=blockers_text[:500],
                blockers_cleared=blockers_text.strip().lower() in {"", "none", "none reported", "no blockers"},
                action_allowed_now=False,
                executable_tool_action_emitted=False,
                can_emit_executable_tool_action=False,
                can_auto_execute_now=False,
                continuation_output=continuation.output[:1600],
            ),
        )

    def autonomy_cycle_ledger(args: dict[str, Any]) -> ToolResult:
        objective = str(args.get("objective") or args.get("request") or "Continue Jarvis V2 safely.").strip()
        closure = autonomy_step_closure_packet(args)
        if not closure.ok:
            return ToolResult("autonomy_cycle_ledger", False, closure.output, closure.metadata)

        closure_metadata = closure.metadata
        one_step_execution_contract_rows = list(closure_metadata.get("one_step_execution_contract_rows") or [])
        one_step_execution_contract_token_sha256 = str(
            closure_metadata.get("one_step_execution_contract_token_sha256") or ""
        )
        one_step_execution_contract_token_as_prior_proof = bool(
            closure_metadata.get("one_step_execution_contract_token_as_prior_proof")
            and _looks_like_sha256(one_step_execution_contract_token_sha256)
            and _one_step_execution_contract_ready(
                objective=str(closure_metadata.get("objective") or ""),
                continuation_state=str(closure_metadata.get("continuation_state") or ""),
                resume_gate_state=str(closure_metadata.get("resume_gate_state") or ""),
                continuation_ready=closure_metadata.get("one_step_execution_contract_ready"),
                one_step_execution_contract_rows=one_step_execution_contract_rows,
                one_step_execution_contract_token_sha256=one_step_execution_contract_token_sha256,
                timebox_receipt_sha256=str(closure_metadata.get("timebox_receipt_sha256") or ""),
                awake_guard_token_sha256=str(closure_metadata.get("awake_guard_token_sha256") or ""),
                supersession_token_sha256=str(closure_metadata.get("supersession_token_sha256") or ""),
                checkpoint_route_token_sha256=str(closure_metadata.get("checkpoint_route_token_sha256") or ""),
                recovery_followthrough_token_sha256=str(closure_metadata.get("recovery_followthrough_token_sha256") or ""),
                local_safe_recovery_execution_token_sha256=str(
                    closure_metadata.get("local_safe_recovery_execution_token_sha256") or ""
                ),
                prior_cycle_ledger_token_sha256=str(closure_metadata.get("prior_cycle_ledger_token_sha256") or ""),
                continuation_review_token_sha256=str(closure_metadata.get("continuation_review_token_sha256") or ""),
                proposed_next_step_sha256=str(closure_metadata.get("proposed_next_step_sha256") or ""),
                proposed_verification_sha256=str(closure_metadata.get("proposed_verification_sha256") or ""),
                risk_signals=[str(signal) for signal in (closure_metadata.get("proposed_next_step_risk_signals") or [])],
                timebox_review_contract_ready=closure_metadata.get("timebox_review_contract_ready"),
                timebox_review_contract_rows=list(closure_metadata.get("timebox_review_contract_rows") or []),
                post_step_proof_queue=[
                    str(command)
                    for command in (
                        closure_metadata.get("post_step_proof_queue")
                        or closure_metadata.get("continuation_post_step_proof_queue")
                        or []
                    )
                ],
            )
            and closure_metadata.get("one_step_execution_contract_token_authorizes_action_now") is False
            and closure_metadata.get("one_step_execution_contract_token_authorizes_risky_work") is False
            and closure_metadata.get("one_step_execution_contract_token_authorizes_unreviewed_followthrough") is False
            and closure_metadata.get("one_step_execution_contract_token_reusable_for_next_step") is False
            and closure_metadata.get("one_step_execution_contract_binds_awake_guard") is True
            and closure_metadata.get("one_step_execution_contract_binds_operator_supersession") is True
            and closure_metadata.get("one_step_execution_contract_binds_timebox_review_contract") is True
        )
        supersession_token_boundary_rows = list(closure_metadata.get("supersession_token_boundary_rows") or [])
        supersession_token_boundary_ready = _operator_supersession_token_boundary_ready(
            str(closure_metadata.get("supersession_token_sha256") or ""),
            supersession_token_boundary_rows,
        )
        checkpoint_route_boundary_rows = list(closure_metadata.get("checkpoint_route_boundary_rows") or [])
        checkpoint_route_boundary_ready = _checkpoint_route_boundary_ready(
            str(closure_metadata.get("checkpoint_route_token_sha256") or ""),
            checkpoint_route_boundary_rows,
        )
        continuation_ready = closure_metadata.get("pre_step_one_local_safe_step_allowed") is True
        closure_ready = closure_metadata.get("ready_for_next_continuation_review") is True
        stage_rows = [
            {"stage": "operator_timebox", "state": closure_metadata.get("timebox_state") or "TIMEBOX_BOUND_THROUGH_CONTINUATION", "ready": continuation_ready, "proof": "operator timebox contract"},
            {"stage": "operator_supersession", "state": closure_metadata.get("supersession_state") or "OPERATOR_SUPERSESSION_BOUND_THROUGH_CONTINUATION", "ready": supersession_token_boundary_ready, "proof": closure_metadata.get("supersession_token_sha256") or ""},
            {"stage": "checkpoint_route", "state": closure_metadata.get("checkpoint_freshness") or "CHECKPOINT_ROUTE_BOUND_THROUGH_CONTINUATION", "ready": checkpoint_route_boundary_ready, "proof": closure_metadata.get("checkpoint_route_token_sha256") or ""},
            {"stage": "recovery_cockpit", "state": closure_metadata.get("cockpit_state") or "RECOVERY_COCKPIT_BOUND_THROUGH_CONTINUATION", "ready": continuation_ready, "proof": "checkpoint recovery cockpit"},
            {"stage": "resume_gate", "state": closure_metadata.get("resume_gate_state") or "AUTONOMY_RESUME_BOUND_THROUGH_CONTINUATION", "ready": continuation_ready, "proof": "autonomy resume gate"},
            {"stage": "one_step_continuation", "state": closure_metadata.get("continuation_state") or "AUTONOMY_CONTINUATION_UNKNOWN", "ready": continuation_ready, "proof": closure_metadata.get("completed_step") or ""},
            {"stage": "recovery_followthrough", "state": closure_metadata.get("followthrough_state") or "RECOVERY_FOLLOWTHROUGH_BOUND_THROUGH_CONTINUATION", "ready": continuation_ready, "proof": "checkpoint recovery follow-through packet"},
            {"stage": "post_step_closure", "state": closure_metadata.get("closure_state") or "AUTONOMY_STEP_CLOSURE_UNKNOWN", "ready": closure_ready, "proof": closure_metadata.get("next_safe_command") or ""},
        ]
        for row in stage_rows:
            row.update(
                {
                    "authorizes_action_now": False,
                    "authorizes_local_safe_step": False,
                    "authorizes_risky_work": False,
                    "authorizes_unreviewed_followthrough": False,
                    "authorizes_timebox_reuse": False,
                    "authorizes_checkpoint_reuse": False,
                    "authorizes_token_reuse": False,
                    "authorizes_approval": False,
                    "authorizes_model_call": False,
                    "authorizes_tool_execution": False,
                    "authorizes_personal_data_read": False,
                    "authorizes_external_side_effect": False,
                    "reusable_for_next_review": False,
                    "reusable_for_next_cycle": False,
                }
            )
        timebox_review_contract_rows = list(closure_metadata.get("timebox_review_contract_rows") or [])
        timebox_review_contract_ready = bool(
            closure_metadata.get("timebox_review_contract_ready")
            and _operator_timebox_review_contract_ready(
                timebox_review_contract_rows,
                timebox_state=str(closure_metadata.get("timebox_state") or ""),
                stop_at=str(closure_metadata.get("stop_at") or ""),
                current_time=str(closure_metadata.get("current_time") or ""),
                timezone_label=str(closure_metadata.get("timezone") or ""),
            )
        )
        missing = list(closure_metadata.get("missing_blockers") or [])
        if not timebox_review_contract_ready:
            missing.append("non_authorizing_timebox_review_contract")
        if not supersession_token_boundary_ready:
            missing.append("operator_supersession_token_boundary")
        if not checkpoint_route_boundary_ready:
            missing.append("checkpoint_route_boundary")
        for row in stage_rows:
            if not row["ready"]:
                missing.append(f"{row['stage']} not ready")
        missing = list(dict.fromkeys(str(item) for item in missing if item))
        autonomy_cycle_stage_rows_ready = _autonomy_cycle_stage_rows_ready(stage_rows)
        if not autonomy_cycle_stage_rows_ready:
            missing.append("autonomy_cycle_stage_rows")
        missing = list(dict.fromkeys(str(item) for item in missing if item))
        fresh_review_contract_rows = [
            {
                "item": "operator_timebox",
                "source": "operator timebox contract",
                "prior_artifact_reusable": False,
                "fresh_required": True,
                "authorizes_action_now": False,
                "authorizes_local_safe_step": False,
                "authorizes_risky_work": False,
                "authorizes_unreviewed_followthrough": False,
                "authorizes_timebox_reuse": False,
                "authorizes_checkpoint_reuse": False,
                "authorizes_token_reuse": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
            },
            {
                "item": "work_block_checkpoint",
                "source": "work block checkpoint",
                "prior_artifact_reusable": False,
                "fresh_required": True,
                "authorizes_action_now": False,
                "authorizes_local_safe_step": False,
                "authorizes_risky_work": False,
                "authorizes_unreviewed_followthrough": False,
                "authorizes_timebox_reuse": False,
                "authorizes_checkpoint_reuse": False,
                "authorizes_token_reuse": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
            },
            {
                "item": "checkpoint_recovery_cockpit",
                "source": "checkpoint recovery cockpit",
                "prior_artifact_reusable": False,
                "fresh_required": True,
                "authorizes_action_now": False,
                "authorizes_local_safe_step": False,
                "authorizes_risky_work": False,
                "authorizes_unreviewed_followthrough": False,
                "authorizes_timebox_reuse": False,
                "authorizes_checkpoint_reuse": False,
                "authorizes_token_reuse": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
            },
            {
                "item": "one_step_continuation_review",
                "source": "autonomy continuation execution packet",
                "prior_artifact_reusable": False,
                "fresh_required": True,
                "authorizes_action_now": False,
                "authorizes_local_safe_step": False,
                "authorizes_risky_work": False,
                "authorizes_unreviewed_followthrough": False,
                "authorizes_timebox_reuse": False,
                "authorizes_checkpoint_reuse": False,
                "authorizes_token_reuse": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
            },
            {
                "item": "prior_cycle_ledger_token",
                "source": "autonomy cycle ledger token",
                "prior_artifact_reusable": False,
                "fresh_required": True,
                "authorizes_action_now": False,
                "authorizes_local_safe_step": False,
                "authorizes_risky_work": False,
                "authorizes_unreviewed_followthrough": False,
                "authorizes_timebox_reuse": False,
                "authorizes_checkpoint_reuse": False,
                "authorizes_token_reuse": False,
                "authorizes_model_call": False,
                "authorizes_tool_execution": False,
                "authorizes_personal_data_read": False,
                "authorizes_external_side_effect": False,
            },
        ]
        all_prior_artifacts_non_authorizing = all(
            row["prior_artifact_reusable"] is False
            and row["authorizes_action_now"] is False
            and row["authorizes_local_safe_step"] is False
            and row["authorizes_risky_work"] is False
            and row["authorizes_unreviewed_followthrough"] is False
            and row["authorizes_timebox_reuse"] is False
            and row["authorizes_checkpoint_reuse"] is False
            and row["authorizes_token_reuse"] is False
            and row["authorizes_model_call"] is False
            and row["authorizes_tool_execution"] is False
            and row["authorizes_personal_data_read"] is False
            and row["authorizes_external_side_effect"] is False
            for row in fresh_review_contract_rows
        )
        closure_fresh_contract_rows = list(closure_metadata.get("fresh_continuation_review_contract_rows") or [])
        closure_fresh_contract_enforced = bool(
            closure_metadata.get("fresh_continuation_review_contract_enforced")
            and _fresh_continuation_review_contract_ready(closure_fresh_contract_rows)
            and timebox_review_contract_ready
        )
        ledger_ready = bool(
            closure_ready
            and autonomy_cycle_stage_rows_ready
            and timebox_review_contract_ready
            and supersession_token_boundary_ready
            and checkpoint_route_boundary_ready
            and closure_fresh_contract_enforced
            and all_prior_artifacts_non_authorizing
        )
        ledger_state = "AUTONOMY_CYCLE_LEDGER_READY_FOR_FRESH_REVIEW" if ledger_ready else "AUTONOMY_CYCLE_LEDGER_HELD"
        next_review_start_command = "autonomy continuation execution: <next reviewed local-safe step>" if ledger_ready else ""
        fresh_review_preflight_queue = [
            "operator timebox contract: stop_at=<ISO> current_time=<ISO>",
            "work block checkpoint",
            "checkpoint recovery cockpit",
            next_review_start_command or "autonomy continuation execution: <next reviewed local-safe step>",
        ]
        fresh_review_preflight_queue = list(dict.fromkeys(command for command in fresh_review_preflight_queue if command))
        autonomy_preflight_scorecard_rows = _autonomy_cycle_preflight_scorecard_rows(
            continuation_ready=continuation_ready,
            closure_ready=closure_ready,
            stage_rows=stage_rows,
            recovery_artifact_hashes_match_files=_metadata_flag_ready(
                closure_metadata,
                "recovery_artifact_hashes_match_files",
            ),
            completed_step_matches_proposed=_metadata_flag_ready(
                closure_metadata,
                "completed_step_matches_proposed",
            ),
            post_step_verification_matches_proposed=_metadata_flag_ready(
                closure_metadata,
                "post_step_verification_matches_proposed",
            ),
            post_step_artifact_hashes_present=_metadata_flag_ready(
                closure_metadata,
                "post_step_artifact_hashes_present",
            ),
            post_step_artifact_hashes_match_files=_metadata_flag_ready(
                closure_metadata,
                "post_step_artifact_hashes_match_files",
            ),
            closure_fresh_contract_enforced=closure_fresh_contract_enforced,
            all_prior_artifacts_non_authorizing=all_prior_artifacts_non_authorizing,
            ledger_ready=ledger_ready,
        )
        autonomy_preflight_score = sum(int(row["points"]) for row in autonomy_preflight_scorecard_rows)
        autonomy_preflight_max_score = sum(int(row["max_points"]) for row in autonomy_preflight_scorecard_rows)
        autonomy_preflight_required_rows_ready = _scorecard_required_rows_ready(autonomy_preflight_scorecard_rows)
        autonomy_preflight_scorecard_ready = _autonomy_cycle_preflight_scorecard_ready(
            autonomy_preflight_scorecard_rows
        )
        closure_readiness_scorecard_rows = list(closure_metadata.get("step_closure_readiness_scorecard_rows") or [])
        closure_readiness_score = int(closure_metadata.get("step_closure_readiness_score") or 0)
        closure_readiness_max_score = int(closure_metadata.get("step_closure_readiness_max_score") or 0)
        closure_readiness_required_rows_ready = _metadata_flag_ready(
            closure_metadata,
            "step_closure_readiness_required_rows_ready",
        )
        closure_readiness_scorecard_ready = _autonomy_step_closure_readiness_scorecard_ready(
            closure_readiness_scorecard_rows
        )
        recovery_execution_scorecard_rows = list(closure_metadata.get("carried_recovery_execution_scorecard_rows") or [])
        recovery_execution_score = int(closure_metadata.get("carried_recovery_execution_score") or 0)
        recovery_execution_max_score = int(closure_metadata.get("carried_recovery_execution_max_score") or 0)
        recovery_execution_required_rows_ready = _recovery_execution_readiness_scorecard_ready(
            recovery_execution_scorecard_rows
        )
        recovery_execution_scorecard_ready = recovery_execution_required_rows_ready
        recovery_execution_carried_as_prior_proof = bool(
            closure_metadata.get("carried_recovery_execution_as_prior_proof")
            and _recovery_execution_readiness_scorecard_ready(recovery_execution_scorecard_rows)
        )
        carried_next_step_approval_proof_queue = list(
            closure_metadata.get("carried_next_step_approval_proof_queue") or []
        )
        carried_next_step_approval_boundary_rows = list(
            closure_metadata.get("carried_next_step_approval_boundary_rows") or []
        )
        carried_next_step_approval_required_before_review = bool(
            closure_metadata.get("carried_next_step_approval_required_before_review")
        )
        carried_next_step_approval_boundary_token_sha256 = str(
            closure_metadata.get("carried_next_step_approval_boundary_token_sha256") or ""
        )
        carried_next_step_approval_boundary_ready = _next_step_approval_boundary_ready(
            token_sha256=carried_next_step_approval_boundary_token_sha256,
            proposed_step_sha256=str(closure_metadata.get("proposed_next_step_sha256") or ""),
            proposed_verification_sha256=str(closure_metadata.get("proposed_verification_sha256") or ""),
            approval_proof_queue=carried_next_step_approval_proof_queue,
            approval_boundary_rows=carried_next_step_approval_boundary_rows,
        )
        carried_next_step_approval_boundary_as_prior_proof = bool(
            closure_metadata.get("carried_next_step_approval_boundary_as_prior_proof")
            and _next_step_approval_boundary_as_prior_proof(
                token_sha256=carried_next_step_approval_boundary_token_sha256,
                proposed_step_sha256=str(closure_metadata.get("proposed_next_step_sha256") or ""),
                proposed_verification_sha256=str(closure_metadata.get("proposed_verification_sha256") or ""),
                approval_proof_queue=carried_next_step_approval_proof_queue,
                approval_boundary_rows=carried_next_step_approval_boundary_rows,
            )
        )
        closure_readiness_carried_as_prior_proof = bool(
            closure_ready
            and closure_readiness_required_rows_ready
            and closure_readiness_scorecard_ready
        )
        required_commands = list(dict.fromkeys(
            list(closure_metadata.get("required_commands") or [])
            + ["autonomy cycle ledger: step=<completed local-safe step> post_step_verification=<evidence> post_step_receipt_path=<receipt> post_step_checkpoint_path=<checkpoint> execution_health=<health> execution_audit=<audit> after_action_learning=<learning>"]
        ))
        next_safe_command = next_review_start_command or "autonomy cycle ledger: step=<completed local-safe step> post_step_verification=<evidence> post_step_receipt_path=<receipt> post_step_checkpoint_path=<checkpoint> execution_health=<health> execution_audit=<audit> after_action_learning=<learning>"
        autonomy_cycle_ledger_token_sha256 = _autonomy_cycle_ledger_token_sha256(
            objective=objective,
            ledger_state=ledger_state,
            stage_rows=stage_rows,
            timebox_receipt_sha256=str(closure_metadata.get("timebox_receipt_sha256") or ""),
            awake_guard_token_sha256=str(closure_metadata.get("awake_guard_token_sha256") or ""),
            supersession_token_sha256=str(closure_metadata.get("supersession_token_sha256") or ""),
            checkpoint_route_token_sha256=str(closure_metadata.get("checkpoint_route_token_sha256") or ""),
            latest_checkpoint_sha256=str(closure_metadata.get("latest_checkpoint_sha256") or ""),
            recovery_followthrough_token_sha256=str(closure_metadata.get("recovery_followthrough_token_sha256") or ""),
            local_safe_recovery_execution_token_sha256=str(closure_metadata.get("local_safe_recovery_execution_token_sha256") or ""),
            one_step_execution_contract_token_sha256=one_step_execution_contract_token_sha256,
            continuation_review_token_sha256=str(closure_metadata.get("continuation_review_token_sha256") or ""),
            prior_cycle_ledger_token_sha256=str(closure_metadata.get("prior_cycle_ledger_token_sha256") or ""),
            proposed_next_step_sha256=str(closure_metadata.get("proposed_next_step_sha256") or ""),
            completed_step_sha256=str(closure_metadata.get("completed_step_sha256") or ""),
            proposed_verification_sha256=str(closure_metadata.get("proposed_verification_sha256") or ""),
            post_step_verification_sha256=str(closure_metadata.get("post_step_verification_sha256") or ""),
            receipt_file_sha256=str(closure_metadata.get("receipt_file_sha256") or ""),
            recovery_checkpoint_file_sha256=str(closure_metadata.get("recovery_checkpoint_file_sha256") or ""),
            post_step_receipt_file_sha256=str(closure_metadata.get("post_step_receipt_file_sha256") or ""),
            post_step_checkpoint_file_sha256=str(closure_metadata.get("post_step_checkpoint_file_sha256") or ""),
            next_review_start_command=next_review_start_command,
            autonomy_preflight_scorecard_rows=autonomy_preflight_scorecard_rows,
            carried_step_closure_readiness_scorecard_rows=closure_readiness_scorecard_rows,
            carried_recovery_execution_scorecard_rows=recovery_execution_scorecard_rows,
            carried_next_step_approval_boundary_rows=carried_next_step_approval_boundary_rows,
        )
        fresh_review_boundary_token_sha256 = _fresh_review_boundary_token_sha256(
            objective=objective,
            ledger_state=ledger_state,
            fresh_review_preflight_queue=fresh_review_preflight_queue,
            fresh_review_contract_rows=fresh_review_contract_rows,
            autonomy_cycle_ledger_token_sha256=autonomy_cycle_ledger_token_sha256,
            next_review_start_command=next_review_start_command,
        )
        fresh_review_boundary_token_rows = _fresh_review_boundary_token_rows(
            token_sha256=fresh_review_boundary_token_sha256,
            source="autonomy_cycle_ledger",
        )
        fresh_review_boundary_token_ready = _fresh_review_boundary_token_ready(
            fresh_review_boundary_token_sha256,
            fresh_review_boundary_token_rows,
        )
        next_review_start_command_token_sha256 = _next_review_start_command_token_sha256(
            objective=objective,
            ledger_state=ledger_state,
            next_review_start_command=next_review_start_command,
            fresh_review_preflight_queue=fresh_review_preflight_queue,
            autonomy_cycle_ledger_token_sha256=autonomy_cycle_ledger_token_sha256,
        )
        next_review_start_command_boundary_rows = _next_review_start_command_boundary_rows(
            token_sha256=next_review_start_command_token_sha256,
            source="autonomy_cycle_ledger",
        )
        next_review_start_command_boundary_ready = _next_review_start_command_boundary_ready(
            next_review_start_command_token_sha256,
            next_review_start_command_boundary_rows,
        )
        autonomy_cycle_ledger_token_boundary_rows = _autonomy_cycle_ledger_token_boundary_rows(
            token_sha256=autonomy_cycle_ledger_token_sha256,
            source="autonomy_cycle_ledger",
        )
        autonomy_cycle_ledger_token_boundary_ready = _autonomy_cycle_ledger_token_boundary_ready(
            autonomy_cycle_ledger_token_sha256,
            autonomy_cycle_ledger_token_boundary_rows,
        )
        local_safe_recovery_execution_token_boundary_rows = _local_safe_recovery_execution_token_boundary_rows(
            token_sha256=str(closure_metadata.get("local_safe_recovery_execution_token_sha256") or ""),
            source="autonomy_cycle_ledger",
        )
        local_safe_recovery_execution_token_boundary_ready = _local_safe_recovery_execution_token_boundary_ready(
            str(closure_metadata.get("local_safe_recovery_execution_token_sha256") or ""),
            local_safe_recovery_execution_token_boundary_rows,
            expected_source="autonomy_cycle_ledger",
        )
        recovery_followthrough_token_boundary_rows = list(
            closure_metadata.get("recovery_followthrough_token_boundary_rows") or []
        )
        recovery_followthrough_token_boundary_ready = _carried_recovery_followthrough_token_boundary_ready(
            closure_metadata,
            expected_source="checkpoint_recovery_followthrough",
        )
        recovery_execution_readiness_token_boundary_rows = list(
            closure_metadata.get("recovery_execution_readiness_token_boundary_rows") or []
        )
        recovery_execution_readiness_token_boundary_ready = _carried_recovery_execution_readiness_token_boundary_ready(
            closure_metadata,
            expected_source="checkpoint_recovery_followthrough",
        )
        prior_cycle_ledger_token_boundary_rows = _prior_cycle_ledger_token_boundary_rows(
            token_sha256=str(closure_metadata.get("prior_cycle_ledger_token_sha256") or ""),
            source="autonomy_cycle_ledger",
        )
        prior_cycle_ledger_token_boundary_ready = _prior_cycle_ledger_token_boundary_ready(
            str(closure_metadata.get("prior_cycle_ledger_token_sha256") or ""),
            prior_cycle_ledger_token_boundary_rows,
        )
        carried_step_closure_receipt_boundary_rows = list(
            closure_metadata.get("step_closure_receipt_boundary_rows") or []
        )
        carried_step_closure_receipt_boundary_ready = _carried_step_closure_receipt_boundary_ready(closure_metadata)
        lines = [
            "Jarvis autonomy cycle ledger:",
            "This is read-only. It binds the operator timebox, checkpoint recovery cockpit, autonomy resume gate, one-step continuation packet, and post-step closure into one prior-step proof ledger before any next autonomous continuation review starts.",
            "",
            "Cycle state:",
            f"- state: {ledger_state}",
            f"- objective: {objective[:300]}",
            f"- ready for fresh next continuation review: {'yes' if ledger_ready else 'no'}",
            f"- preflight score: {autonomy_preflight_score}/{autonomy_preflight_max_score}",
            f"- preflight rows ready: {'yes' if autonomy_preflight_required_rows_ready else 'no'}",
            "- action allowed now: no",
            "- executable tool action emitted: no",
            f"- missing blockers: {', '.join(missing) if missing else 'none'}",
            "",
            "Cycle stages:",
            *[f"- {row['stage']}: {row['state']} ({'ready' if row['ready'] else 'held'}; proof: {row['proof'] or 'none'})" for row in stage_rows],
            "",
            "Measured autonomy preflight scorecard:",
            f"- scorecard ready: {'yes' if autonomy_preflight_scorecard_ready else 'no'}",
            *[
                f"- {row['item']}: {row['points']}/{row['max_points']} ({'ready' if row['ready'] else 'held'}; does not authorize action or risky work)"
                for row in autonomy_preflight_scorecard_rows
            ],
            "",
            "Carried step-closure readiness proof:",
            f"- score: {closure_readiness_score}/{closure_readiness_max_score}",
            f"- required rows ready: {'yes' if closure_readiness_required_rows_ready else 'no'}",
            f"- scorecard ready: {'yes' if closure_readiness_scorecard_ready else 'no'}",
            f"- carried as prior proof: {'yes' if closure_readiness_carried_as_prior_proof else 'no'}",
            f"- receipt token sha256: {closure_metadata.get('step_closure_receipt_token_sha256') or 'missing'}",
            f"- receipt boundary rows: {len(carried_step_closure_receipt_boundary_rows)}",
            f"- receipt boundary ready: {'yes' if carried_step_closure_receipt_boundary_ready else 'no'}",
            "- authorizes action now: no",
            "- authorizes risky work: no",
            "- authorizes follow-up without fresh review: no",
            "- receipt token authorizes local-safe step: no",
            "- receipt token authorizes new cycle: no",
            *[
                f"- {row['item']}: {row['points']}/{row['max_points']} ({'ready' if row['ready'] else 'held'}; prior-step proof only)"
                for row in closure_readiness_scorecard_rows
            ],
            *[
                f"- {row['item']}: {row.get('status', 'held')}; carried receipt proof only; authorizes action now no; local-safe step no; risky work no; reusable no"
                for row in carried_step_closure_receipt_boundary_rows
            ],
            "",
            "Carried recovery execution readiness proof:",
            f"- score: {recovery_execution_score}/{recovery_execution_max_score}",
            f"- required rows ready: {'yes' if recovery_execution_required_rows_ready else 'no'}",
            f"- carried as prior proof: {'yes' if recovery_execution_carried_as_prior_proof else 'no'}",
            "- authorizes action now: no",
            "- authorizes risky work: no",
            "- authorizes unreviewed follow-through: no",
            *[
                f"- {row['item']}: {row['points']}/{row['max_points']} ({'ready' if row['ready'] else 'held'}; prior recovery proof only)"
                for row in recovery_execution_scorecard_rows
            ],
            "",
            "Carried risky next-step approval proof:",
            f"- approval required before review: {'yes' if carried_next_step_approval_required_before_review else 'no'}",
            f"- proof queue count: {len(carried_next_step_approval_proof_queue)}",
            f"- boundary token sha256: {carried_next_step_approval_boundary_token_sha256 or 'missing'}",
            f"- boundary rows: {len(carried_next_step_approval_boundary_rows)}",
            f"- boundary ready: {'yes' if carried_next_step_approval_boundary_ready else 'no'}",
            f"- carried as prior proof: {'yes' if carried_next_step_approval_boundary_as_prior_proof else 'no'}",
            "- authorizes action now: no",
            "- authorizes risky work: no",
            "- authorizes unreviewed follow-through: no",
            *[f"- `{command}`" for command in carried_next_step_approval_proof_queue],
            *[
                f"- {row['item']}: {row.get('status', 'held')}; prior approval-boundary proof only"
                for row in carried_next_step_approval_boundary_rows
            ],
            "",
            "Fresh-review boundary:",
            f"- next review start command: `{next_safe_command}`",
            f"- fresh-review preflight queue: {', '.join(f'`{command}`' for command in fresh_review_preflight_queue)}",
            "- fresh-review contract:",
            *[
                f"  - {row['item']}: fresh required yes; prior reusable no; authorizes action now no; authorizes local-safe step no; authorizes risky work no; authorizes unreviewed follow-through no; authorizes timebox/checkpoint/token reuse no; source {row['source']}"
                for row in fresh_review_contract_rows
            ],
            f"- all prior artifacts non-authorizing: {'yes' if all_prior_artifacts_non_authorizing else 'no'}",
            f"- closure fresh-continuation contract carried forward: {'yes' if closure_fresh_contract_enforced else 'no'}",
            f"- closure fresh-continuation rows: {len(closure_fresh_contract_rows)}",
            f"- carried timebox receipt sha256: {closure_metadata.get('timebox_receipt_sha256') or 'missing'}",
            f"- carried timebox review contract ready: {'yes' if timebox_review_contract_ready else 'no'}",
            f"- carried timebox review rows: {len(timebox_review_contract_rows)}",
            "- carried timebox authorizes execution: no",
            "- carried timebox authorizes local-safe step: no",
            "- carried timebox authorizes risky work: no",
            "- carried timebox authorizes approval: no",
            "- carried timebox authorizes timebox reuse: no",
            "- carried timebox reusable for next step: no",
            *[
                f"  - {row.get('item', 'unknown')}: fresh required yes; reusable next step no; authorizes execution no; authorizes local-safe step no; authorizes risky work no; authorizes approval no; authorizes timebox reuse no"
                for row in timebox_review_contract_rows
            ],
            "- carried awake guard requested: " + ("yes" if closure_metadata.get("awake_guard_requested") else "no"),
            f"- carried awake guard token sha256: {closure_metadata.get('awake_guard_token_sha256') or 'missing'}",
            f"- carried awake guard boundary rows: {closure_metadata.get('awake_guard_boundary_row_count') or 0}",
            "- carried awake guard authorizes OS wake lock: no",
            "- carried awake guard authorizes shell execution: no",
            "- carried awake guard authorizes computer control: no",
            "- carried awake guard authorizes approval: no",
            "- carried awake guard reusable for next timebox: no",
            "- next awake guard requires fresh review: yes",
            f"- carried operator supersession state: {closure_metadata.get('supersession_state') or 'unknown'}",
            f"- carried supersession token sha256: {closure_metadata.get('supersession_token_sha256') or 'missing'}",
            f"- carried supersession boundary rows: {len(supersession_token_boundary_rows)}",
            f"- carried supersession boundary ready: {'yes' if supersession_token_boundary_ready else 'no'}",
            "- carried supersession token authorizes execution: no",
            "- carried supersession token authorizes local-safe step: no",
            "- carried supersession token authorizes risky work: no",
            "- carried supersession token authorizes approval: no",
            "- carried supersession token authorizes recovery follow-through: no",
            "- carried supersession token authorizes timebox override: no",
            "- carried supersession token reusable for next review: no",
            *[
                f"- {row['item']}: {row['status']}; authorizes execution no; local-safe step no; risky work no; approval no; recovery follow-through no; reusable no"
                for row in supersession_token_boundary_rows
            ],
            "- next supersession requires fresh latest-instruction review: yes",
            f"- carried checkpoint route token sha256: {closure_metadata.get('checkpoint_route_token_sha256') or 'missing'}",
            f"- carried checkpoint route boundary rows: {len(checkpoint_route_boundary_rows)}",
            f"- carried checkpoint route boundary ready: {'yes' if checkpoint_route_boundary_ready else 'no'}",
            f"- carried checkpoint freshness: {closure_metadata.get('checkpoint_freshness') or 'missing'}",
            f"- carried checkpoint needs recovery review: {'yes' if closure_metadata.get('checkpoint_needs_review') else 'no'}",
            "- carried checkpoint route authorizes continuation: no",
            "- carried checkpoint route authorizes local-safe step: no",
            "- carried checkpoint route authorizes risky work: no",
            "- carried checkpoint route authorizes approval: no",
            "- carried checkpoint route authorizes recovery follow-through: no",
            "- carried checkpoint route authorizes checkpoint reuse: no",
            "- carried checkpoint route reusable for next review: no",
            *[
                f"- {row['item']}: {row['status']}; freshness {row['checkpoint_freshness']}; needs review {'yes' if row['checkpoint_needs_review'] else 'no'}; authorizes continuation no; local-safe step no; risky work no; approval no; checkpoint reuse no; reusable no"
                for row in checkpoint_route_boundary_rows
            ],
            "- next checkpoint route requires fresh recovery review: yes",
            "- next review requires full preflight: yes",
            "- fresh operator timebox required before next continuation: yes",
            "- fresh checkpoint required before next continuation: yes",
            f"- prior checkpoint matched latest checkpoint: {'yes' if closure_metadata.get('checkpoint_path_matches_latest') else 'no'}",
            f"- prior checkpoint hash matched latest checkpoint: {'yes' if closure_metadata.get('checkpoint_hash_matches_latest') else 'no'}",
            f"- prior recovery receipt hash matched file: {'yes' if closure_metadata.get('receipt_hash_matches_file') else 'no'}",
            f"- prior recovery checkpoint hash matched file: {'yes' if closure_metadata.get('recovery_checkpoint_hash_matches_file') else 'no'}",
            "- previous one-step continuation permission reusable for next review: no",
            f"- previous one-step execution contract token sha256: {one_step_execution_contract_token_sha256 or 'missing'}",
            f"- previous one-step execution contract token carried as prior proof: {'yes' if one_step_execution_contract_token_as_prior_proof else 'no'}",
            "- previous one-step execution contract token authorizes action now: no",
            "- previous one-step execution contract token authorizes risky work: no",
            "- previous one-step execution contract token reusable for next step: no",
            "- previous post-step receipt reusable for next review: no",
            f"- previous post-step receipt hash reusable for next review: {'no' if closure_metadata.get('post_step_receipt_hash_provided') else 'not supplied'}",
            f"- previous post-step receipt hash matched file: {'yes' if closure_metadata.get('post_step_receipt_hash_matches_file') else 'no'}",
            f"- previous post-step checkpoint hash reusable for next review: {'no' if closure_metadata.get('post_step_checkpoint_hash_provided') else 'not supplied'}",
            f"- previous post-step checkpoint hash matched file: {'yes' if closure_metadata.get('post_step_checkpoint_hash_matches_file') else 'no'}",
            f"- post-step artifact hashes supplied: {'yes' if closure_metadata.get('post_step_artifact_hashes_present') else 'no'}",
            f"- prior recovery follow-through token sha256: {closure_metadata.get('recovery_followthrough_token_sha256') or 'missing'}",
            "- previous recovery follow-through token reusable for next review: no",
            "- new recovery follow-through token required before future recovery follow-through: yes",
            f"- local-safe recovery execution token sha256: {closure_metadata.get('local_safe_recovery_execution_token_sha256') or 'missing'}",
            "- previous local-safe recovery execution token reusable for next review: no",
            "- local-safe recovery execution token authorizes resume gate: no",
            "- local-safe recovery execution token authorizes next step: no",
            "- local-safe recovery execution token authorizes local-safe step: no",
            "- local-safe recovery execution token authorizes risky work: no",
            "- local-safe recovery execution token authorizes approval: no",
            "- local-safe recovery execution token authorizes model call: no",
            "- local-safe recovery execution token authorizes tool execution: no",
            "- local-safe recovery execution token authorizes personal-data read: no",
            "- local-safe recovery execution token authorizes external side effect: no",
            f"- local-safe recovery execution token boundary rows: {len(local_safe_recovery_execution_token_boundary_rows)}",
            "- new local-safe recovery execution token required before future recovery: yes",
            f"- recovery execution readiness token sha256: {closure_metadata.get('recovery_execution_readiness_token_sha256') or 'missing'}",
            "- recovery execution readiness token authorizes resume gate: no",
            "- recovery execution readiness token authorizes next step: no",
            "- recovery execution readiness token authorizes risky work: no",
            "- recovery execution readiness token authorizes approval: no",
            "- recovery execution readiness token authorizes model call: no",
            "- recovery execution readiness token authorizes tool execution: no",
            "- recovery execution readiness token authorizes personal-data read: no",
            "- recovery execution readiness token authorizes external side effect: no",
            "- recovery execution readiness token authorizes unreviewed follow-through: no",
            "- recovery execution readiness token reusable for future recovery: no",
            f"- recovery execution readiness token boundary rows: {closure_metadata.get('recovery_execution_readiness_token_boundary_row_count') or 0}",
            "- new recovery execution readiness token required before future recovery: yes",
            f"- proposed step sha256: {closure_metadata.get('proposed_next_step_sha256') or 'missing'}",
            f"- completed step sha256: {closure_metadata.get('completed_step_sha256') or 'missing'}",
            f"- completed step matched allowed continuation step: {'yes' if closure_metadata.get('completed_step_matches_proposed') else 'no'}",
            f"- proposed verification sha256: {closure_metadata.get('proposed_verification_sha256') or 'missing'}",
            f"- post-step verification sha256: {closure_metadata.get('post_step_verification_sha256') or 'missing'}",
            f"- post-step verification matched allowed target: {'yes' if closure_metadata.get('post_step_verification_matches_proposed') else 'no'}",
            f"- prior continuation review token sha256: {closure_metadata.get('continuation_review_token_sha256') or 'missing'}",
            "- previous continuation review token reusable for next review: no",
            "- new continuation review token required before next step: yes",
            f"- prior cycle ledger token sha256 supplied to this cycle: {closure_metadata.get('prior_cycle_ledger_token_sha256') or 'not supplied'}",
            f"- prior cycle ledger token was proof-only for this cycle: {'yes' if closure_metadata.get('prior_cycle_ledger_token_present') else 'not supplied'}",
            f"- prior cycle ledger boundary rows: {len(prior_cycle_ledger_token_boundary_rows)}",
            f"- prior cycle ledger boundary ready: {'yes' if prior_cycle_ledger_token_boundary_ready else 'no'}",
            *[
                f"- {row['item']}: {row['status']}; authorizes action now no; authorizes new action no; authorizes local-safe step no; authorizes risky work no; authorizes model/tool/private/external no; reusable no"
                for row in prior_cycle_ledger_token_boundary_rows
            ],
            "- prior cycle ledger token reusable for this cycle closure: no",
            "- prior cycle ledger proof authorizes new action in this ledger: no",
            f"- autonomy cycle ledger token sha256: {autonomy_cycle_ledger_token_sha256 or 'missing'}",
            f"- autonomy cycle ledger token boundary rows: {len(autonomy_cycle_ledger_token_boundary_rows)}",
            f"- autonomy cycle ledger token boundary ready: {'yes' if autonomy_cycle_ledger_token_boundary_ready else 'no'}",
            *[
                f"- {row['item']}: {row['status']}; authorizes action now no; authorizes local-safe step no; authorizes risky work no; authorizes model/tool/private/external no; reusable next review no"
                for row in autonomy_cycle_ledger_token_boundary_rows
            ],
            "- previous cycle ledger token reusable for next review: no",
            "- new cycle ledger token required before next step: yes",
            f"- fresh-review boundary token sha256: {fresh_review_boundary_token_sha256 or 'missing'}",
            f"- fresh-review boundary rows: {len(fresh_review_boundary_token_rows)}",
            f"- fresh-review boundary ready: {'yes' if fresh_review_boundary_token_ready else 'no'}",
            *[
                f"  - {row['item']}: {row['status']}; fresh required yes; authorizes action now no; authorizes local-safe step no; authorizes risky work no; authorizes model/tool/private/external no; reusable next review no"
                for row in fresh_review_boundary_token_rows
            ],
            "- fresh-review boundary token authorizes local-safe step: no",
            "- fresh-review boundary token authorizes risky work: no",
            "- fresh-review boundary token reusable for next review: no",
            "- next review requires fresh boundary token: yes",
            f"- next review start command token sha256: {next_review_start_command_token_sha256 or 'missing'}",
            f"- next review start command boundary rows: {len(next_review_start_command_boundary_rows)}",
            f"- next review start command boundary ready: {'yes' if next_review_start_command_boundary_ready else 'no'}",
            *[
                f"- {row['item']}: {row['status']}; authorizes action now no; authorizes local-safe step no; authorizes risky work no; authorizes approval no; reusable next review no"
                for row in next_review_start_command_boundary_rows
            ],
            "- next review start command authorizes action now: no",
            "- next review start command authorizes local-safe step: no",
            "- next review start command authorizes risky work: no",
            "- next review start command authorizes approval: no",
            "- next review start command reusable for next review: no",
            "- next review start command requires fresh preflight: yes",
            "- prior continuation step proof only: yes",
            "",
            "Required cycle proof chain:",
            f"- command count: {len(required_commands)}",
            *[f"- `{command}`" for command in required_commands],
            "",
            "Boundary:",
            "- This ledger does not execute another step, run shell/code, control the computer, read private data, write notes, approve requests, rerun actions, or queue approvals.",
            "- It proves bookkeeping only; a new continuation must start from a fresh timebox and fresh reviewed local-safe step.",
        ]
        metadata = _safe_metadata(
                objective=objective,
                objective_length=len(objective),
                cycle_state=ledger_state,
                ready_for_fresh_next_continuation_review=ledger_ready,
                ready_for_next_continuation_review=closure_ready,
                next_review_start_command=next_review_start_command,
                fresh_review_preflight_queue=fresh_review_preflight_queue,
                fresh_review_preflight_queue_count=len(fresh_review_preflight_queue),
                fresh_review_next_preflight_command=fresh_review_preflight_queue[0] if fresh_review_preflight_queue else "",
                fresh_review_contract_rows=fresh_review_contract_rows,
                fresh_review_contract_count=len(fresh_review_contract_rows),
                fresh_review_contract_ready=all_prior_artifacts_non_authorizing,
                all_prior_artifacts_non_authorizing=all_prior_artifacts_non_authorizing,
                prior_artifacts_authorize_local_safe_step=False,
                prior_artifacts_authorize_risky_work=False,
                prior_artifacts_authorize_unreviewed_followthrough=False,
                prior_artifacts_authorize_timebox_reuse=False,
                prior_artifacts_authorize_checkpoint_reuse=False,
                prior_artifacts_authorize_token_reuse=False,
                prior_artifacts_authorize_model_call=False,
                prior_artifacts_authorize_tool_execution=False,
                prior_artifacts_authorize_personal_data_read=False,
                prior_artifacts_authorize_external_side_effect=False,
                autonomy_preflight_score=autonomy_preflight_score,
                autonomy_preflight_max_score=autonomy_preflight_max_score,
                autonomy_preflight_scorecard_rows=autonomy_preflight_scorecard_rows,
                autonomy_preflight_scorecard_row_count=len(autonomy_preflight_scorecard_rows),
                autonomy_preflight_required_rows_ready=autonomy_preflight_required_rows_ready,
                autonomy_preflight_scorecard_ready=autonomy_preflight_scorecard_ready,
                carried_step_closure_readiness_score=closure_readiness_score,
                carried_step_closure_readiness_max_score=closure_readiness_max_score,
                carried_step_closure_readiness_scorecard_rows=closure_readiness_scorecard_rows,
                carried_step_closure_readiness_scorecard_row_count=len(closure_readiness_scorecard_rows),
                carried_step_closure_readiness_required_rows_ready=closure_readiness_required_rows_ready,
                carried_step_closure_readiness_scorecard_ready=closure_readiness_scorecard_ready,
                carried_step_closure_readiness_as_prior_proof=closure_readiness_carried_as_prior_proof,
                carried_step_closure_readiness_authorizes_action_now=False,
                carried_step_closure_readiness_authorizes_risky_work=False,
                carried_step_closure_readiness_authorizes_followup_without_fresh_review=False,
                carried_step_closure_receipt_token_sha256=closure_metadata.get("step_closure_receipt_token_sha256", ""),
                carried_step_closure_receipt_token_present=_metadata_flag_ready(
                    closure_metadata,
                    "step_closure_receipt_token_present",
                ),
                carried_step_closure_receipt_boundary_rows=carried_step_closure_receipt_boundary_rows,
                carried_step_closure_receipt_boundary_row_count=len(carried_step_closure_receipt_boundary_rows),
                carried_step_closure_receipt_boundary_ready=carried_step_closure_receipt_boundary_ready,
                carried_step_closure_receipt_authorizes_action_now=False,
                carried_step_closure_receipt_authorizes_local_safe_step=False,
                carried_step_closure_receipt_authorizes_risky_work=False,
                carried_step_closure_receipt_authorizes_new_cycle=False,
                carried_step_closure_receipt_authorizes_unreviewed_followthrough=False,
                carried_step_closure_receipt_authorizes_timebox_reuse=False,
                carried_step_closure_receipt_authorizes_checkpoint_reuse=False,
                carried_step_closure_receipt_authorizes_model_call=False,
                carried_step_closure_receipt_authorizes_tool_execution=False,
                carried_step_closure_receipt_authorizes_personal_data_read=False,
                carried_step_closure_receipt_authorizes_external_side_effect=False,
                carried_step_closure_receipt_reusable_for_next_review=False,
                carried_step_closure_receipt_reusable_for_next_cycle=False,
                next_review_requires_new_step_closure_receipt_token=True,
                carried_recovery_execution_score=recovery_execution_score,
                carried_recovery_execution_max_score=recovery_execution_max_score,
                carried_recovery_execution_scorecard_rows=recovery_execution_scorecard_rows,
                carried_recovery_execution_scorecard_row_count=len(recovery_execution_scorecard_rows),
                carried_recovery_execution_required_rows_ready=recovery_execution_required_rows_ready,
                carried_recovery_execution_scorecard_ready=recovery_execution_scorecard_ready,
                carried_recovery_execution_as_prior_proof=recovery_execution_carried_as_prior_proof,
                carried_recovery_execution_authorizes_action_now=False,
                carried_recovery_execution_authorizes_risky_work=False,
                carried_recovery_execution_authorizes_unreviewed_followthrough=False,
                carried_recovery_execution_authorizes_model_call=False,
                carried_recovery_execution_authorizes_tool_execution=False,
                carried_recovery_execution_authorizes_personal_data_read=False,
                carried_recovery_execution_authorizes_external_side_effect=False,
                recovery_execution_readiness_token_sha256=closure_metadata.get("recovery_execution_readiness_token_sha256", ""),
                recovery_execution_readiness_token_present=_metadata_flag_ready(
                    closure_metadata,
                    "recovery_execution_readiness_token_present",
                ),
                recovery_execution_readiness_token_boundary_rows=recovery_execution_readiness_token_boundary_rows,
                recovery_execution_readiness_token_boundary_row_count=len(recovery_execution_readiness_token_boundary_rows),
                recovery_execution_readiness_token_boundary_ready=recovery_execution_readiness_token_boundary_ready,
                recovery_execution_readiness_token_authorizes_resume_gate=False,
                recovery_execution_readiness_token_authorizes_next_step=False,
                recovery_execution_readiness_token_authorizes_risky_work=False,
                recovery_execution_readiness_token_authorizes_approval=False,
                recovery_execution_readiness_token_authorizes_model_call=False,
                recovery_execution_readiness_token_authorizes_tool_execution=False,
                recovery_execution_readiness_token_authorizes_personal_data_read=False,
                recovery_execution_readiness_token_authorizes_external_side_effect=False,
                recovery_execution_readiness_token_authorizes_unreviewed_followthrough=False,
                recovery_execution_readiness_token_reusable_for_future_recovery=False,
                next_recovery_execution_requires_new_readiness_token=True,
                carried_next_step_approval_proof_queue=carried_next_step_approval_proof_queue,
                carried_next_step_approval_proof_queue_count=len(carried_next_step_approval_proof_queue),
                carried_next_step_next_approval_proof_command=(
                    carried_next_step_approval_proof_queue[0] if carried_next_step_approval_proof_queue else ""
                ),
                carried_next_step_approval_required_before_review=carried_next_step_approval_required_before_review,
                carried_next_step_approval_boundary_rows=carried_next_step_approval_boundary_rows,
                carried_next_step_approval_boundary_row_count=len(carried_next_step_approval_boundary_rows),
                carried_next_step_approval_boundary_ready=carried_next_step_approval_boundary_ready,
                carried_next_step_approval_boundary_token_sha256=carried_next_step_approval_boundary_token_sha256,
                carried_next_step_approval_boundary_token_present=_looks_like_sha256(carried_next_step_approval_boundary_token_sha256),
                carried_next_step_approval_boundary_as_prior_proof=carried_next_step_approval_boundary_as_prior_proof,
                carried_next_step_approval_boundary_authorizes_action_now=False,
                carried_next_step_approval_boundary_authorizes_risky_work=False,
                carried_next_step_approval_boundary_authorizes_unreviewed_followthrough=False,
                carried_next_step_approval_boundary_authorizes_approval=False,
                carried_next_step_approval_boundary_authorizes_model_call=False,
                carried_next_step_approval_boundary_authorizes_tool_execution=False,
                carried_next_step_approval_boundary_authorizes_personal_data_read=False,
                carried_next_step_approval_boundary_authorizes_external_side_effect=False,
                carried_next_step_approval_boundary_authorizes_timebox_reuse=False,
                carried_next_step_approval_boundary_reusable_for_next_review=False,
                carried_next_step_approval_boundary_reusable_for_recovery_review=False,
                closure_fresh_continuation_review_contract_rows=closure_fresh_contract_rows,
                closure_fresh_continuation_review_contract_row_count=len(closure_fresh_contract_rows),
                closure_fresh_continuation_review_contract_summary=closure_metadata.get("fresh_continuation_review_contract_summary", []),
                closure_fresh_continuation_review_contract_enforced=closure_fresh_contract_enforced,
                fresh_continuation_review_boundary_token_sha256=closure_metadata.get("fresh_continuation_review_boundary_token_sha256", ""),
                fresh_continuation_review_boundary_token_present=_metadata_flag_ready(
                    closure_metadata,
                    "fresh_continuation_review_boundary_token_present",
                ),
                fresh_continuation_review_boundary_token_rows=closure_metadata.get("fresh_continuation_review_boundary_token_rows", []),
                fresh_continuation_review_boundary_token_row_count=closure_metadata.get("fresh_continuation_review_boundary_token_row_count", 0),
                fresh_continuation_review_boundary_token_ready=_metadata_flag_ready(
                    closure_metadata,
                    "fresh_continuation_review_boundary_token_ready",
                ),
                fresh_continuation_review_boundary_authorizes_action_now=False,
                fresh_continuation_review_boundary_authorizes_local_safe_step=False,
                fresh_continuation_review_boundary_authorizes_risky_work=False,
                fresh_continuation_review_boundary_authorizes_unreviewed_followthrough=False,
                fresh_continuation_review_boundary_authorizes_timebox_reuse=False,
                fresh_continuation_review_boundary_authorizes_checkpoint_reuse=False,
                fresh_continuation_review_boundary_authorizes_token_reuse=False,
                fresh_continuation_review_boundary_authorizes_model_call=False,
                fresh_continuation_review_boundary_authorizes_tool_execution=False,
                fresh_continuation_review_boundary_authorizes_personal_data_read=False,
                fresh_continuation_review_boundary_authorizes_external_side_effect=False,
                fresh_continuation_review_boundary_reusable_for_next_review=False,
                fresh_continuation_review_boundary_reusable_for_next_cycle=False,
                timebox_receipt_sha256=closure_metadata.get("timebox_receipt_sha256", ""),
                timebox_receipt_present=_metadata_flag_ready(
                    closure_metadata,
                    "timebox_receipt_present",
                ),
                timebox_review_contract_rows=timebox_review_contract_rows,
                timebox_review_contract_row_count=len(timebox_review_contract_rows),
                timebox_review_contract_ready=timebox_review_contract_ready,
                timebox_review_contract_summary=closure_metadata.get("timebox_review_contract_summary", []),
                timebox_authorizes_execution=False,
                timebox_authorizes_local_safe_step=False,
                timebox_authorizes_risky_work=False,
                timebox_authorizes_approval=False,
                timebox_authorizes_timebox_reuse=False,
                timebox_authorizes_model_call=False,
                timebox_authorizes_tool_execution=False,
                timebox_authorizes_personal_data_read=False,
                timebox_authorizes_external_side_effect=False,
                timebox_reusable_for_next_step=False,
                next_step_requires_fresh_timebox=True,
                awake_guard_requested=closure_metadata.get("awake_guard_requested", False),
                awake_guard_token_sha256=closure_metadata.get("awake_guard_token_sha256", ""),
                awake_guard_token_present=_metadata_flag_ready(
                    closure_metadata,
                    "awake_guard_token_present",
                ),
                awake_guard_boundary_rows=closure_metadata.get("awake_guard_boundary_rows", []),
                awake_guard_boundary_row_count=closure_metadata.get("awake_guard_boundary_row_count", 0),
                awake_guard_authorizes_os_wake_lock=False,
                awake_guard_authorizes_shell_execution=False,
                awake_guard_authorizes_computer_control=False,
                awake_guard_authorizes_approval=False,
                awake_guard_authorizes_model_call=False,
                awake_guard_authorizes_tool_execution=False,
                awake_guard_authorizes_personal_data_read=False,
                awake_guard_authorizes_external_side_effect=False,
                awake_guard_reusable_for_next_timebox=False,
                awake_guard_requires_separate_operator_request=True,
                awake_guard_requires_separate_shell_approval=True,
                awake_guard_os_wake_lock_boundary_ready=_awake_guard_boundary_ready_from_metadata(
                    closure_metadata,
                    default_objective=objective,
                ),
                awake_guard_caffeinate_command_authorized=False,
                awake_guard_keep_awake_command_authorized=False,
                awake_guard_authorizes_unattended_execution=False,
                awake_guard_authorizes_continuation_window=False,
                awake_guard_reusable_as_execution_permission=False,
                next_awake_guard_requires_fresh_review=True,
                previous_instruction=closure_metadata.get("previous_instruction", ""),
                latest_instruction=closure_metadata.get("latest_instruction", ""),
                latest_instruction_present=closure_metadata.get("latest_instruction_present", False),
                latest_instruction_supersedes_previous=closure_metadata.get("latest_instruction_supersedes_previous", False),
                latest_instruction_is_stop=closure_metadata.get("latest_instruction_is_stop", False),
                latest_instruction_is_continue=closure_metadata.get("latest_instruction_is_continue", False),
                supersession_state=closure_metadata.get("supersession_state", ""),
                can_continue_under_latest_instruction=closure_metadata.get("can_continue_under_latest_instruction", False),
                newest_instruction_overrides_automation=closure_metadata.get("newest_instruction_overrides_automation", True),
                newest_instruction_overrides_goal=closure_metadata.get("newest_instruction_overrides_goal", True),
                newest_instruction_overrides_recovery_queue=closure_metadata.get("newest_instruction_overrides_recovery_queue", True),
                stop_or_pause_blocks_autonomy=closure_metadata.get("stop_or_pause_blocks_autonomy", False),
                supersession_token_sha256=closure_metadata.get("supersession_token_sha256", ""),
                supersession_token_present=_metadata_flag_ready(
                    closure_metadata,
                    "supersession_token_present",
                ),
                supersession_token_boundary_rows=supersession_token_boundary_rows,
                supersession_token_boundary_row_count=len(supersession_token_boundary_rows),
                supersession_token_boundary_ready=supersession_token_boundary_ready,
                supersession_token_authorizes_execution=False,
                supersession_token_authorizes_local_safe_step=False,
                supersession_token_authorizes_risky_work=False,
                supersession_token_authorizes_approval=False,
                supersession_token_authorizes_recovery_followthrough=False,
                supersession_token_authorizes_timebox_override=False,
                supersession_token_authorizes_goal_override=False,
                supersession_token_authorizes_model_call=False,
                supersession_token_authorizes_tool_execution=False,
                supersession_token_authorizes_personal_data_read=False,
                supersession_token_authorizes_external_side_effect=False,
                supersession_token_reusable_for_next_review=False,
                supersession_token_reusable_for_next_timebox=False,
                next_supersession_requires_fresh_latest_instruction_review=True,
                checkpoint_freshness=closure_metadata.get("checkpoint_freshness"),
                checkpoint_needs_review=closure_metadata.get("checkpoint_needs_review", True),
                checkpoint_route_token_sha256=closure_metadata.get("checkpoint_route_token_sha256", ""),
                checkpoint_route_token_present=_metadata_flag_ready(
                    closure_metadata,
                    "checkpoint_route_token_present",
                ),
                checkpoint_route_boundary_rows=checkpoint_route_boundary_rows,
                checkpoint_route_boundary_row_count=len(checkpoint_route_boundary_rows),
                checkpoint_route_boundary_ready=checkpoint_route_boundary_ready,
                checkpoint_route_authorizes_continuation=False,
                checkpoint_route_authorizes_local_safe_step=False,
                checkpoint_route_authorizes_risky_work=False,
                checkpoint_route_authorizes_approval=False,
                checkpoint_route_authorizes_recovery_followthrough=False,
                checkpoint_route_authorizes_checkpoint_reuse=False,
                checkpoint_route_authorizes_model_call=False,
                checkpoint_route_authorizes_tool_execution=False,
                checkpoint_route_authorizes_personal_data_read=False,
                checkpoint_route_authorizes_external_side_effect=False,
                checkpoint_route_reusable_for_next_review=False,
                checkpoint_route_reusable_for_next_checkpoint=False,
                next_checkpoint_route_requires_fresh_recovery_review=True,
                checkpoint_route_recovery_proof_queue=closure_metadata.get("checkpoint_route_recovery_proof_queue", []),
                checkpoint_route_recovery_proof_queue_count=closure_metadata.get("checkpoint_route_recovery_proof_queue_count", 0),
                checkpoint_route_next_recovery_command=closure_metadata.get("checkpoint_route_next_recovery_command", ""),
                checkpoint_recovery_proof_queue=closure_metadata.get("checkpoint_recovery_proof_queue", []),
                checkpoint_recovery_proof_queue_count=closure_metadata.get("checkpoint_recovery_proof_queue_count", 0),
                checkpoint_recovery_next_proof_command=closure_metadata.get("checkpoint_recovery_next_proof_command", ""),
                closure_next_continuation_requires_fresh_operator_timebox=_metadata_flag_ready(
                    closure_metadata,
                    "next_continuation_requires_fresh_operator_timebox",
                ),
                closure_next_continuation_requires_fresh_checkpoint=_metadata_flag_ready(
                    closure_metadata,
                    "next_continuation_requires_fresh_checkpoint",
                ),
                closure_next_continuation_requires_fresh_recovery_cockpit=_metadata_flag_ready(
                    closure_metadata,
                    "next_continuation_requires_fresh_recovery_cockpit",
                ),
                closure_next_continuation_requires_fresh_local_safe_step=_metadata_flag_ready(
                    closure_metadata,
                    "next_continuation_requires_fresh_local_safe_step",
                ),
                closure_next_continuation_requires_fresh_review_token=_metadata_flag_ready(
                    closure_metadata,
                    "next_continuation_requires_fresh_review_token",
                ),
                closure_prior_step_authorizes_followup=_metadata_flag_disabled(
                    closure_metadata,
                    "prior_step_closure_authorizes_followup",
                ) is False,
                closure_prior_step_reusable_for_next_step=_metadata_flag_disabled(
                    closure_metadata,
                    "prior_step_closure_reusable_for_next_step",
                ) is False,
                next_review_requires_full_preflight=True,
                next_safe_command=next_safe_command,
                latest_checkpoint_path=closure_metadata.get("latest_checkpoint_path", ""),
                latest_checkpoint_sha256=closure_metadata.get("latest_checkpoint_sha256", ""),
                latest_checkpoint_hash_present=closure_metadata.get("latest_checkpoint_hash_present", False),
                supplied_checkpoint_path=closure_metadata.get("supplied_checkpoint_path", ""),
                checkpoint_path_matches_latest=closure_metadata.get("checkpoint_path_matches_latest"),
                checkpoint_hash_matches_latest=closure_metadata.get("checkpoint_hash_matches_latest"),
                receipt_sha256=closure_metadata.get("receipt_sha256", ""),
                receipt_hash_provided=closure_metadata.get("receipt_hash_provided", False),
                receipt_file_sha256=closure_metadata.get("receipt_file_sha256", ""),
                receipt_file_hash_present=closure_metadata.get("receipt_file_hash_present", False),
                receipt_hash_matches_file=closure_metadata.get("receipt_hash_matches_file", False),
                checkpoint_sha256=closure_metadata.get("checkpoint_sha256", ""),
                checkpoint_hash_provided=closure_metadata.get("checkpoint_hash_provided", False),
                recovery_checkpoint_file_sha256=closure_metadata.get("recovery_checkpoint_file_sha256", ""),
                recovery_checkpoint_file_hash_present=closure_metadata.get("recovery_checkpoint_file_hash_present", False),
                recovery_checkpoint_hash_matches_file=closure_metadata.get("recovery_checkpoint_hash_matches_file", False),
                recovery_artifact_hashes_present=_metadata_flag_ready(
                    closure_metadata,
                    "recovery_artifact_hashes_present",
                ),
                recovery_artifact_hashes_match_files=_metadata_flag_ready(
                    closure_metadata,
                    "recovery_artifact_hashes_match_files",
                ),
                recovery_followthrough_token_sha256=closure_metadata.get("recovery_followthrough_token_sha256", ""),
                recovery_followthrough_token_present=_metadata_flag_ready(
                    closure_metadata,
                    "recovery_followthrough_token_present",
                ),
                recovery_followthrough_token_boundary_rows=recovery_followthrough_token_boundary_rows,
                recovery_followthrough_token_boundary_row_count=len(recovery_followthrough_token_boundary_rows),
                recovery_followthrough_token_boundary_ready=recovery_followthrough_token_boundary_ready,
                recovery_followthrough_token_reusable_for_future_recovery=False,
                previous_recovery_followthrough_token_reusable_for_next_review=False,
                recovery_followthrough_token_authorizes_resume_gate=False,
                recovery_followthrough_token_authorizes_next_step=False,
                recovery_followthrough_token_authorizes_risky_work=False,
                recovery_followthrough_token_authorizes_approval=False,
                recovery_followthrough_token_authorizes_model_call=False,
                recovery_followthrough_token_authorizes_tool_execution=False,
                recovery_followthrough_token_authorizes_personal_data_read=False,
                recovery_followthrough_token_authorizes_external_side_effect=False,
                next_recovery_followthrough_requires_new_token=True,
                local_safe_recovery_execution_token_sha256=closure_metadata.get("local_safe_recovery_execution_token_sha256", ""),
                local_safe_recovery_execution_token_present=_metadata_flag_ready(
                    closure_metadata,
                    "local_safe_recovery_execution_token_present",
                ),
                local_safe_recovery_execution_token_authorizes_resume_gate=False,
                local_safe_recovery_execution_token_authorizes_next_step=False,
                local_safe_recovery_execution_token_authorizes_risky_work=False,
                local_safe_recovery_execution_token_authorizes_approval=False,
                local_safe_recovery_execution_token_authorizes_model_call=False,
                local_safe_recovery_execution_token_authorizes_tool_execution=False,
                local_safe_recovery_execution_token_authorizes_personal_data_read=False,
                local_safe_recovery_execution_token_authorizes_external_side_effect=False,
                local_safe_recovery_execution_token_reusable_for_future_recovery=False,
                previous_local_safe_recovery_execution_token_reusable_for_next_review=False,
                next_recovery_execution_requires_new_local_safe_token=True,
                local_safe_recovery_execution_token_boundary_rows=local_safe_recovery_execution_token_boundary_rows,
                local_safe_recovery_execution_token_boundary_row_count=len(local_safe_recovery_execution_token_boundary_rows),
                local_safe_recovery_execution_token_boundary_ready=local_safe_recovery_execution_token_boundary_ready,
                post_step_receipt_sha256=closure_metadata.get("post_step_receipt_sha256", ""),
                post_step_receipt_hash_provided=closure_metadata.get("post_step_receipt_hash_provided", False),
                post_step_receipt_file_sha256=closure_metadata.get("post_step_receipt_file_sha256", ""),
                post_step_receipt_file_hash_present=closure_metadata.get("post_step_receipt_file_hash_present", False),
                post_step_receipt_hash_matches_file=closure_metadata.get("post_step_receipt_hash_matches_file", False),
                post_step_checkpoint_sha256=closure_metadata.get("post_step_checkpoint_sha256", ""),
                post_step_checkpoint_hash_provided=closure_metadata.get("post_step_checkpoint_hash_provided", False),
                post_step_checkpoint_file_sha256=closure_metadata.get("post_step_checkpoint_file_sha256", ""),
                post_step_checkpoint_file_hash_present=closure_metadata.get("post_step_checkpoint_file_hash_present", False),
                post_step_checkpoint_hash_matches_file=closure_metadata.get("post_step_checkpoint_hash_matches_file", False),
                post_step_artifact_hashes_present=_metadata_flag_ready(
                    closure_metadata,
                    "post_step_artifact_hashes_present",
                ),
                post_step_artifact_hashes_match_files=_metadata_flag_ready(
                    closure_metadata,
                    "post_step_artifact_hashes_match_files",
                ),
                execution_health=closure_metadata.get("execution_health", ""),
                execution_audit=closure_metadata.get("execution_audit", ""),
                after_action_learning=closure_metadata.get("after_action_learning", ""),
                next_review_requires_fresh_operator_timebox=True,
                next_review_requires_fresh_checkpoint=True,
                previous_continuation_permission_reusable_for_next_review=False,
                one_step_execution_contract_rows=one_step_execution_contract_rows,
                one_step_execution_contract_row_count=len(one_step_execution_contract_rows),
                one_step_execution_contract_ready=_metadata_flag_ready(
                    closure_metadata,
                    "one_step_execution_contract_ready",
                ),
                one_step_execution_contract_token_sha256=one_step_execution_contract_token_sha256,
                one_step_execution_contract_token_present=_looks_like_sha256(one_step_execution_contract_token_sha256),
                one_step_execution_contract_token_as_prior_proof=one_step_execution_contract_token_as_prior_proof,
                one_step_execution_contract_binds_awake_guard=_metadata_flag_ready(
                    closure_metadata,
                    "one_step_execution_contract_binds_awake_guard",
                ),
                one_step_execution_contract_binds_operator_supersession=_metadata_flag_ready(
                    closure_metadata,
                    "one_step_execution_contract_binds_operator_supersession",
                ),
                one_step_execution_contract_binds_timebox_review_contract=_metadata_flag_ready(
                    closure_metadata,
                    "one_step_execution_contract_binds_timebox_review_contract",
                ),
                one_step_execution_contract_token_authorizes_action_now=False,
                one_step_execution_contract_token_authorizes_risky_work=False,
                one_step_execution_contract_token_authorizes_unreviewed_followthrough=False,
                one_step_execution_contract_token_authorizes_batching=False,
                one_step_execution_contract_token_reusable_for_next_step=False,
                next_step_requires_new_one_step_execution_contract_token=True,
                one_step_execution_contract_all_local_safe_step_limited=_metadata_flag_ready(
                    closure_metadata,
                    "one_step_execution_contract_all_local_safe_step_limited",
                ),
                one_step_execution_contract_all_non_reusable=_metadata_flag_ready(
                    closure_metadata,
                    "one_step_execution_contract_all_non_reusable",
                ),
                one_step_execution_contract_all_risky_work_gated=_metadata_flag_ready(
                    closure_metadata,
                    "one_step_execution_contract_all_risky_work_gated",
                ),
                one_step_execution_contract_requires_fresh_closure=_metadata_flag_ready(
                    closure_metadata,
                    "one_step_execution_contract_requires_fresh_closure",
                ),
                one_step_execution_contract_authorizes_batching=False,
                one_step_execution_contract_authorizes_followup_without_closure=False,
                previous_post_step_receipt_reusable_for_next_review=False,
                previous_post_step_receipt_hash_reusable_for_next_review=False,
                previous_post_step_checkpoint_hash_reusable_for_next_review=False,
                previous_continuation_review_token_reusable_for_next_review=False,
                next_review_requires_new_continuation_review_token=True,
                prior_continuation_step_proof_only=True,
                post_step_proof_queue=closure_metadata.get("continuation_post_step_proof_queue", []),
                post_step_proof_queue_count=closure_metadata.get("continuation_post_step_proof_queue_count", 0),
                post_step_next_proof_command=closure_metadata.get("continuation_post_step_next_proof_command", ""),
                action_allowed_now=False,
                executable_tool_action_emitted=False,
                can_emit_executable_tool_action=False,
                can_auto_execute_now=False,
                stage_rows=stage_rows,
                stage_count=len(stage_rows),
                autonomy_cycle_stage_rows_ready=autonomy_cycle_stage_rows_ready,
                missing_blockers=missing,
                missing_blocker_count=len(missing),
                required_commands=required_commands,
                required_command_count=len(required_commands),
                next_required_command=required_commands[0] if required_commands else "",
                proof_queue=required_commands,
                proof_queue_count=len(required_commands),
                next_proof_command=required_commands[0] if required_commands else "",
                closure_state=closure_metadata.get("closure_state"),
                resume_gate_state=closure_metadata.get("resume_gate_state"),
                timebox_state=closure_metadata.get("timebox_state"),
                can_continue_now=closure_metadata.get("can_continue_now"),
                should_stop_now=closure_metadata.get("should_stop_now"),
                missing_timebox_proof=closure_metadata.get("missing_timebox_proof", []),
                missing_timebox_proof_count=closure_metadata.get("missing_timebox_proof_count", 0),
                cockpit_state=closure_metadata.get("cockpit_state"),
                followthrough_state=closure_metadata.get("followthrough_state"),
                continuation_state=closure_metadata.get("continuation_state"),
                pre_step_one_local_safe_step_allowed=continuation_ready,
                proposed_next_step=closure_metadata.get("proposed_next_step", ""),
                proposed_next_step_sha256=closure_metadata.get("proposed_next_step_sha256", ""),
                proposed_next_step_risk_signals=closure_metadata.get("proposed_next_step_risk_signals", []),
                proposed_next_step_risk_signal_count=closure_metadata.get("proposed_next_step_risk_signal_count", 0),
                proposed_verification=closure_metadata.get("proposed_verification", ""),
                proposed_verification_sha256=closure_metadata.get("proposed_verification_sha256", ""),
                continuation_review_token_sha256=closure_metadata.get("continuation_review_token_sha256", ""),
                continuation_review_token_present=_metadata_flag_ready(
                    closure_metadata,
                    "continuation_review_token_present",
                ),
                continuation_review_token_reusable_for_next_review=False,
                prior_cycle_ledger_token_sha256=closure_metadata.get("prior_cycle_ledger_token_sha256", ""),
                prior_cycle_ledger_token_present=closure_metadata.get("prior_cycle_ledger_token_present", False),
                prior_cycle_ledger_token_boundary_rows=prior_cycle_ledger_token_boundary_rows,
                prior_cycle_ledger_token_boundary_row_count=len(prior_cycle_ledger_token_boundary_rows),
                prior_cycle_ledger_token_boundary_ready=prior_cycle_ledger_token_boundary_ready,
                prior_cycle_ledger_token_reusable_for_this_review=False,
                prior_cycle_ledger_token_reusable_for_this_closure=False,
                prior_cycle_ledger_token_reusable_for_this_cycle=False,
                prior_cycle_ledger_proof_authorizes_action_now=False,
                prior_cycle_ledger_proof_authorizes_post_step_closure=False,
                prior_cycle_ledger_proof_authorizes_new_action=False,
                prior_cycle_ledger_proof_authorizes_model_call=False,
                prior_cycle_ledger_proof_authorizes_tool_execution=False,
                prior_cycle_ledger_proof_authorizes_personal_data_read=False,
                prior_cycle_ledger_proof_authorizes_external_side_effect=False,
                autonomy_cycle_ledger_token_sha256=autonomy_cycle_ledger_token_sha256,
                autonomy_cycle_ledger_token_present=_looks_like_sha256(autonomy_cycle_ledger_token_sha256),
                autonomy_cycle_ledger_token_boundary_rows=autonomy_cycle_ledger_token_boundary_rows,
                autonomy_cycle_ledger_token_boundary_row_count=len(autonomy_cycle_ledger_token_boundary_rows),
                autonomy_cycle_ledger_token_boundary_ready=autonomy_cycle_ledger_token_boundary_ready,
                autonomy_cycle_ledger_token_authorizes_action_now=False,
                autonomy_cycle_ledger_token_authorizes_local_safe_step=False,
                autonomy_cycle_ledger_token_authorizes_risky_work=False,
                autonomy_cycle_ledger_token_authorizes_new_cycle=False,
                autonomy_cycle_ledger_token_authorizes_unreviewed_followthrough=False,
                autonomy_cycle_ledger_token_authorizes_timebox_reuse=False,
                autonomy_cycle_ledger_token_authorizes_checkpoint_reuse=False,
                autonomy_cycle_ledger_token_authorizes_token_reuse=False,
                autonomy_cycle_ledger_token_authorizes_model_call=False,
                autonomy_cycle_ledger_token_authorizes_tool_execution=False,
                autonomy_cycle_ledger_token_authorizes_personal_data_read=False,
                autonomy_cycle_ledger_token_authorizes_external_side_effect=False,
                previous_cycle_ledger_token_reusable_for_next_review=False,
                next_review_requires_new_cycle_ledger_token=True,
                fresh_review_boundary_token_sha256=fresh_review_boundary_token_sha256,
                fresh_review_boundary_token_present=_looks_like_sha256(fresh_review_boundary_token_sha256),
                fresh_review_boundary_token_rows=fresh_review_boundary_token_rows,
                fresh_review_boundary_token_row_count=len(fresh_review_boundary_token_rows),
                fresh_review_boundary_token_ready=fresh_review_boundary_token_ready,
                fresh_review_boundary_token_authorizes_action_now=False,
                fresh_review_boundary_token_authorizes_local_safe_step=False,
                fresh_review_boundary_token_authorizes_risky_work=False,
                fresh_review_boundary_token_authorizes_unreviewed_followthrough=False,
                fresh_review_boundary_token_authorizes_timebox_reuse=False,
                fresh_review_boundary_token_authorizes_checkpoint_reuse=False,
                fresh_review_boundary_token_authorizes_token_reuse=False,
                fresh_review_boundary_token_authorizes_model_call=False,
                fresh_review_boundary_token_authorizes_tool_execution=False,
                fresh_review_boundary_token_authorizes_personal_data_read=False,
                fresh_review_boundary_token_authorizes_external_side_effect=False,
                fresh_review_boundary_token_reusable_for_next_review=False,
                fresh_review_boundary_token_reusable_for_next_cycle=False,
                next_review_requires_new_fresh_review_boundary_token=True,
                next_review_start_command_token_sha256=next_review_start_command_token_sha256,
                next_review_start_command_token_present=_looks_like_sha256(next_review_start_command_token_sha256),
                next_review_start_command_boundary_rows=next_review_start_command_boundary_rows,
                next_review_start_command_boundary_row_count=len(next_review_start_command_boundary_rows),
                next_review_start_command_boundary_ready=next_review_start_command_boundary_ready,
                next_review_start_command_authorizes_action_now=False,
                next_review_start_command_authorizes_local_safe_step=False,
                next_review_start_command_authorizes_risky_work=False,
                next_review_start_command_authorizes_unreviewed_followthrough=False,
                next_review_start_command_authorizes_timebox_reuse=False,
                next_review_start_command_authorizes_checkpoint_reuse=False,
                next_review_start_command_authorizes_token_reuse=False,
                next_review_start_command_authorizes_model_call=False,
                next_review_start_command_authorizes_tool_execution=False,
                next_review_start_command_authorizes_personal_data_read=False,
                next_review_start_command_authorizes_external_side_effect=False,
                next_review_start_command_authorizes_approval=False,
                next_review_start_command_reusable_for_next_review=False,
                next_review_start_command_reusable_for_next_cycle=False,
                next_review_start_command_requires_fresh_preflight=True,
                completed_step=closure_metadata.get("completed_step", ""),
                completed_step_sha256=closure_metadata.get("completed_step_sha256", ""),
                completed_step_matches_proposed=_metadata_flag_ready(
                    closure_metadata,
                    "completed_step_matches_proposed",
                ),
                post_step_verification_sha256=closure_metadata.get("post_step_verification_sha256", ""),
                post_step_verification_matches_proposed=_metadata_flag_ready(
                    closure_metadata,
                    "post_step_verification_matches_proposed",
                ),
                post_step_verification_provided=closure_metadata.get("post_step_verification_provided"),
                post_step_receipt_path_provided=closure_metadata.get("post_step_receipt_path_provided"),
                post_step_checkpoint_path_provided=closure_metadata.get("post_step_checkpoint_path_provided"),
                execution_health_reviewed=closure_metadata.get("execution_health_reviewed"),
                execution_audit_reviewed=closure_metadata.get("execution_audit_reviewed"),
                after_action_learning_reviewed=closure_metadata.get("after_action_learning_reviewed"),
                step_closure_metadata=closure_metadata,
                step_closure_output=closure.output[:2000],
            )
        metadata["autonomy_cycle_ledger_ready"] = _autonomy_cycle_ledger_ready_from_metadata(metadata)
        return ToolResult(
            "autonomy_cycle_ledger",
            True,
            "\n".join(lines),
            metadata,
        )

    def activity_digest(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 8)
        memories = store.recent_memories(limit=limit)
        sessions, unreadable_session_rows = _safe_session_payloads(store.list_sessions(limit=5))
        raw_tool_runs = store.recent_tool_runs(limit=limit)
        tool_runs, unreadable_tool_run_rows = _safe_tool_run_payloads(raw_tool_runs)
        approvals = store.list_pending_approvals(limit=limit)
        tasks = store.list_tasks(status="open", limit=limit)
        goals = store.list_goals(status="active", limit=limit)
        decisions = store.list_decisions(status="active", limit=limit)
        people = store.list_people(limit=5)
        preferences = store.list_preferences(status="active", limit=5)
        jobs = store.list_jobs()

        tool_counts = Counter(row["tool_name"] for row in tool_runs)
        tool_run_status_counts = _tool_run_status_counts(tool_runs)
        failed_runs = [row for row in tool_runs if _tool_run_status(row) == "failed"]
        approval_held_runs = [row for row in tool_runs if _tool_run_status(row) == "approval_held"]

        lines = ["Jarvis activity digest:"]
        if sessions:
            latest = sessions[0]
            lines.append(
                f"- conversation: {len(sessions)} recent session(s); latest {_safe_preview(latest['session_id'], limit=80)} "
                f"has {latest['messages']} messages, last {_safe_preview(latest['last_at'], limit=80)}"
            )
        else:
            lines.append("- conversation: no sessions indexed yet")
        if unreadable_session_rows:
            lines.append(f"- unreadable session row(s) hidden for safety: {unreadable_session_rows}")

        if tool_runs:
            top_tools = ", ".join(f"{name} x{count}" for name, count in tool_counts.most_common(5))
            lines.append(
                f"- tools: {len(tool_runs)} recent run(s); {top_tools}; "
                f"{tool_run_status_counts['ok']} ok, {tool_run_status_counts['failed']} failed, "
                f"{tool_run_status_counts['approval_held']} approval-held"
            )
        else:
            lines.append("- tools: no tool runs logged yet")
        if unreadable_tool_run_rows:
            lines.append(f"- unreadable tool run row(s) hidden for safety: {unreadable_tool_run_rows}")

        if approvals:
            lines.append(f"- blockers: {len(approvals)} pending approval(s)")
        else:
            lines.append("- blockers: no pending approvals")

        if tasks:
            lines.append(f"- tasks: {len(tasks)} open; next #{tasks[0]['id']} {_safe_preview(tasks[0]['body'])}")
        else:
            lines.append("- tasks: no open tasks")

        if goals:
            goal_bits = []
            for goal in goals[:3]:
                steps = store.list_goal_steps(goal["id"])
                open_steps = [step for step in steps if step["status"] != "done"]
                next_step = _safe_preview(open_steps[0]["body"]) if open_steps else "define next step"
                goal_bits.append(f"#{goal['id']} {_safe_preview(goal['title'])} -> {next_step}")
            lines.append("- goals: " + "; ".join(goal_bits))
        else:
            lines.append("- goals: no active goals")

        if decisions:
            lines.append(f"- decisions: {len(decisions)} active; latest #{decisions[0]['id']} {_safe_preview(decisions[0]['title'])}")
        else:
            lines.append("- decisions: no active decisions")

        if memories:
            lines.append("Recent memory:")
            for row in memories[:5]:
                lines.append(f"- [{_safe_preview(row['category'], limit=80)}] {_safe_preview(row['title'])}")
        else:
            lines.append("Recent memory: none")

        if failed_runs or approval_held_runs or approvals or unreadable_tool_run_rows or unreadable_session_rows:
            lines.append("Needs attention:")
            for row in approvals[:5]:
                lines.append(f"- Approval #{row['id']} {_safe_preview(row['tool_name'], limit=80)}: {_safe_preview(row['user_input'])}")
                lines.append(f"  Readiness: `approval readiness {row['id']}`")
                lines.append(f"  Last look: `approval packet {row['id']}`")
            for row in approval_held_runs[:3]:
                approval_id = row.get("approval_id")
                lines.append(
                    f"- Approval-held {_safe_preview(row['tool_name'], limit=80)} at {row['created_at']}: "
                    "safety gate held before execution"
                )
                if approval_id:
                    lines.append(f"  Readiness: `approval readiness {approval_id}`")
                    lines.append(f"  Last look: `approval packet {approval_id}`")
            for row in failed_runs[:3]:
                snippet = _safe_preview(row["output"], limit=120)
                lines.append(f"- Failed {_safe_preview(row['tool_name'], limit=80)} at {row['created_at']}: {snippet}")
            if unreadable_tool_run_rows:
                lines.append(f"- Review recent tool runs: {unreadable_tool_run_rows} unreadable row(s) hidden for safety")
            if unreadable_session_rows:
                lines.append(f"- Review recent sessions: {unreadable_session_rows} unreadable row(s) hidden for safety")

        lines.append("Context now:")
        lines.append(f"- people profiles: {len(people)} visible")
        lines.append(f"- active preferences: {len(preferences)} visible")
        enabled_jobs = [row for row in jobs if row["enabled"]]
        lines.append(f"- scheduled jobs: {len(enabled_jobs)} enabled / {len(jobs)} total")

        return ToolResult(
            "activity_digest",
            True,
            "\n".join(lines),
            _safe_metadata(
                memories=len(memories),
                sessions=len(sessions),
                readable_session_rows=len(sessions),
                unreadable_session_rows=unreadable_session_rows,
                tool_runs=len(tool_runs),
                recent_ok_tool_runs=tool_run_status_counts["ok"],
                recent_failed_tool_runs=tool_run_status_counts["failed"],
                recent_approval_held_tool_runs=tool_run_status_counts["approval_held"],
                readable_tool_run_rows=len(tool_runs),
                unreadable_tool_run_rows=unreadable_tool_run_rows,
                pending_approvals=len(approvals),
                open_tasks=len(tasks),
                active_goals=len(goals),
                active_decisions=len(decisions),
                limit=limit,
            ),
        )

    def recent_saved_notes(args: dict[str, Any]) -> ToolResult:
        body, metadata = recent_saved_notes_body(_bounded_int(args.get("limit"), 12, high=30))
        return ToolResult("recent_saved_notes", True, body, metadata)

    return (
        activity_digest,
        operator_instruction_supersession_packet,
        operator_timebox_contract,
        build_progress_report,
        build_delta_report,
        work_block_checkpoint,
        save_build_progress,
        save_build_delta,
        save_work_block_checkpoint,
        checkpoint_recovery_preview,
        checkpoint_recovery_apply_packet,
        checkpoint_recovery_receipt,
        checkpoint_recovery_execute,
        checkpoint_recovery_followthrough_packet,
        checkpoint_recovery_cockpit,
        autonomy_resume_gate,
        autonomy_continuation_execution_packet,
        autonomy_step_closure_packet,
        autonomy_cycle_ledger,
        recent_saved_notes,
    )
