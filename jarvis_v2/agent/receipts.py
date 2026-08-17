from __future__ import annotations

import re
from typing import Any


MAX_RECEIPT_PREVIEW_CHARS = 240
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")
EXACT_RUNTIME_TRACE_KEYS = {"storage_fallback_db_path", "storage_fallback_vault_path"}


def _receipt_text(value: object) -> str:
    try:
        return "" if value is None else str(value)
    except Exception:
        return ""


def _receipt_key(value: object) -> str:
    text = _receipt_preview(value, limit=120)
    return text or "<unreadable>"


def _receipt_scrub_text(value: object) -> str:
    text = _receipt_text(value)
    return LOCAL_PATH_RE.sub("<local-path>", text)


def _receipt_preview(value: object, *, limit: int = MAX_RECEIPT_PREVIEW_CHARS) -> str:
    text = " ".join(_receipt_scrub_text(value).split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _receipt_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return _receipt_preview(value)
    if isinstance(value, dict):
        return {_receipt_key(key): _receipt_safe_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_receipt_safe_value(item) for item in value]
    return _receipt_preview(value)


def _receipt_metadata_bool(value: Any, default: bool = False) -> bool:
    if value is True:
        return True
    if value is False:
        return False
    return default


def _receipt_safe_runtime_trace(value: Any) -> Any:
    if not isinstance(value, dict):
        return _receipt_safe_value(value)
    safe: dict[str, Any] = {}
    for key, item in value.items():
        text_key = _receipt_key(key)
        if text_key in EXACT_RUNTIME_TRACE_KEYS:
            safe[text_key] = item
        elif isinstance(item, dict):
            safe[text_key] = _receipt_safe_runtime_trace(item)
        elif isinstance(item, list | tuple):
            safe[text_key] = [_receipt_safe_runtime_trace(entry) for entry in item]
        else:
            safe[text_key] = _receipt_safe_value(item)
    return safe


def _receipt_args_preview(args: dict[str, Any]) -> dict[str, str]:
    return {_receipt_key(key): _receipt_preview(value) for key, value in sorted((args or {}).items(), key=lambda item: _receipt_key(item[0]))}


def _empty_planner_receipt_metadata() -> dict[str, Any]:
    return {
        "planner_type": None,
        "model_planner_attempted": False,
        "model_planner_state": None,
        "model_planner_used": False,
        "model_planner_fell_back": False,
        "model_planner_fallback_reason": None,
        "model_planner_fallback_detail": None,
        "model_planner_recovery_hint": None,
        "model_planner_exception_type": None,
        "model_planner_model": None,
        "model_planner_timeout_seconds": None,
        "model_planner_action_count": None,
        "model_planner_ignored_unknown_tools": [],
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _planner_receipt_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return _empty_planner_receipt_metadata()

    safe = _receipt_safe_value(value)
    if not isinstance(safe, dict):
        return _empty_planner_receipt_metadata()
    ignored = safe.get("model_planner_ignored_unknown_tools")
    ignored_items = ignored if isinstance(ignored, list) else []
    ignored_tools = [_receipt_preview(item, limit=80) for item in ignored_items if _receipt_preview(item, limit=80)][:8]
    return {
        "planner_type": _receipt_preview(safe.get("planner_type"), limit=80) if safe.get("planner_type") else None,
        "model_planner_attempted": _receipt_metadata_bool(safe.get("model_planner_attempted")),
        "model_planner_state": _receipt_preview(safe.get("model_planner_state"), limit=80) if safe.get("model_planner_state") else None,
        "model_planner_used": _receipt_metadata_bool(safe.get("model_planner_used")),
        "model_planner_fell_back": _receipt_metadata_bool(safe.get("model_planner_fell_back")),
        "model_planner_fallback_reason": _receipt_preview(safe.get("model_planner_fallback_reason"), limit=120) if safe.get("model_planner_fallback_reason") else None,
        "model_planner_fallback_detail": _receipt_preview(safe.get("model_planner_fallback_detail"), limit=180) if safe.get("model_planner_fallback_detail") else None,
        "model_planner_recovery_hint": _receipt_preview(safe.get("model_planner_recovery_hint"), limit=220) if safe.get("model_planner_recovery_hint") else None,
        "model_planner_exception_type": _receipt_preview(safe.get("model_planner_exception_type"), limit=80) if safe.get("model_planner_exception_type") else None,
        "model_planner_model": _receipt_preview(safe.get("model_planner_model"), limit=120) if safe.get("model_planner_model") else None,
        "model_planner_timeout_seconds": safe.get("model_planner_timeout_seconds") if isinstance(safe.get("model_planner_timeout_seconds"), int | float) and not isinstance(safe.get("model_planner_timeout_seconds"), bool) else None,
        "model_planner_action_count": safe.get("model_planner_action_count") if isinstance(safe.get("model_planner_action_count"), int) and not isinstance(safe.get("model_planner_action_count"), bool) else None,
        "model_planner_ignored_unknown_tools": ignored_tools,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def runtime_result_receipt(result: Any, pending_approvals: int) -> dict[str, Any]:
    raw_runtime_trace = result.metadata.get("runtime_trace", {}) if isinstance(result.metadata, dict) else {}
    runtime_trace = _receipt_safe_runtime_trace(raw_runtime_trace)
    planner_notes = _receipt_preview(raw_runtime_trace.get("planner_notes"), limit=120) if isinstance(raw_runtime_trace, dict) else ""
    planner_metadata = _planner_receipt_metadata(raw_runtime_trace.get("planner_metadata") if isinstance(raw_runtime_trace, dict) else {})
    def _int_trace(name: str, default: int = 0) -> int:
        value = raw_runtime_trace.get(name)
        if value is None or isinstance(value, bool):
            return default
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return default

    risk_levels = [
        _receipt_key(risk)
        for risk in raw_runtime_trace.get("risk_levels", [])
        if _receipt_key(risk) != "<unreadable>"
    ]
    queued_approval_ids = [
        approval_id
        for approval_id in raw_runtime_trace.get("queued_approval_ids", [])
        if approval_id is not None
    ]
    new_approval_ids = [
        approval_id
        for approval_id in raw_runtime_trace.get("new_approval_ids", [])
        if approval_id is not None
    ]
    reused_approval_ids = [
        approval_id
        for approval_id in raw_runtime_trace.get("reused_approval_ids", [])
        if approval_id is not None
    ]
    return {
        "ok": result.verified,
        "verified": result.verified,
        "message": _receipt_scrub_text(result.user_input),
        "response": _receipt_scrub_text(result.response),
        "route": result.metadata.get("runtime_route"),
        "runtime_route": result.metadata.get("runtime_route"),
        "runtime_trace": runtime_trace,
        "planner_notes": planner_notes,
        "planner_metadata": planner_metadata,
        "planner_model_planner_attempted": planner_metadata.get("model_planner_attempted"),
        "planner_model_planner_state": planner_metadata.get("model_planner_state"),
        "planner_model_planner_used": planner_metadata.get("model_planner_used"),
        "planner_model_planner_fell_back": planner_metadata.get("model_planner_fell_back"),
        "planner_model_planner_fallback_reason": planner_metadata.get("model_planner_fallback_reason"),
        "planner_model_planner_fallback_detail": planner_metadata.get("model_planner_fallback_detail"),
        "planner_model_planner_recovery_hint": planner_metadata.get("model_planner_recovery_hint"),
        "planner_model_planner_exception_type": planner_metadata.get("model_planner_exception_type"),
        "planner_model_planner_model": planner_metadata.get("model_planner_model"),
        "planner_model_planner_timeout_seconds": planner_metadata.get("model_planner_timeout_seconds"),
        "planner_model_planner_action_count": planner_metadata.get("model_planner_action_count"),
        "planner_model_planner_ignored_unknown_tools": planner_metadata.get("model_planner_ignored_unknown_tools"),
        "chat_response": _receipt_safe_value(result.metadata.get("chat_response", {})),
        "verification": _receipt_scrub_text(runtime_trace.get("verification")),
        "approved": _receipt_metadata_bool(raw_runtime_trace.get("approved")),
        "approved_approval_id": raw_runtime_trace.get("approved_approval_id"),
        "referenced_approval_ids": list(raw_runtime_trace.get("referenced_approval_ids", [])),
        "approval_required": _receipt_metadata_bool(raw_runtime_trace.get("approval_required")),
        "queued_approval_ids": queued_approval_ids,
        "queued_approval_count": len(queued_approval_ids),
        "new_approval_ids": new_approval_ids,
        "new_approval_count": len(new_approval_ids),
        "reused_approval_ids": reused_approval_ids,
        "reused_approval_count": len(reused_approval_ids),
        "approval_queue_before": _int_trace("approval_queue_before"),
        "approval_queue_after": _int_trace("approval_queue_after", pending_approvals),
        "approval_queue_delta": _int_trace("approval_queue_delta"),
        "risk_levels": risk_levels,
        "ran_tool_handlers": _receipt_metadata_bool(raw_runtime_trace.get("ran_tool_handlers")),
        "approved_reruns": _int_trace("approved_reruns"),
        "approved_rerun_run_ids": list(raw_runtime_trace.get("approved_rerun_run_ids", [])),
        "approved_rerun_approval_ids": list(raw_runtime_trace.get("approved_rerun_approval_ids", [])),
        "pending_approvals": pending_approvals,
        "plan": {
            "goal": _receipt_scrub_text(result.plan.goal),
            "needs_model": result.plan.needs_model,
            "actions": [
                {
                    "tool_name": action.tool_name,
                    "args": _receipt_args_preview(action.args),
                    "arg_keys": sorted(_receipt_key(key) for key in action.args),
                    "reason": _receipt_scrub_text(action.reason),
                }
                for action in result.plan.actions
            ],
        },
        "tool_results": [
            {
                "tool": item.tool_name,
                "tool_name": item.tool_name,
                "ok": item.ok,
                "output": _receipt_preview(item.output),
                "output_preview": _receipt_preview(item.output),
                "metadata": _receipt_safe_value(item.metadata),
            }
            for item in result.tool_results
        ],
    }
