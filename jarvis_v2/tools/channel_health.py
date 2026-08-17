from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from jarvis_v2.agent.types import ToolResult
from jarvis_v2.memory.store import MemoryStore


WINDOW_DAYS = 7
MAX_ROWS = 500
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
SAFE_STAGE_VALUE_RE = re.compile(r"[A-Za-z0-9_.:-]{1,80}")

CHANNELS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("telegram_message", "Telegram messages", ("send_telegram",)),
    ("kakao_message", "KakaoTalk messages", ("send_kakao",)),
    ("instagram_message", "Instagram DMs", ("send_instagram_dm",)),
    ("imessage", "iMessage", ("send_imessage",)),
    ("phone_facetime", "Phone / FaceTime", ("call_contact",)),
    ("kakao_call", "KakaoTalk calls", ("call_kakao",)),
    ("instagram_call", "Instagram calls", ("call_instagram",)),
    ("telegram_call", "Telegram calls", ("call_telegram",)),
)

STAGE_KEYS = (
    "telegram_send_stage",
    "telegram_call_stage",
    "instagram_send_stage",
    "instagram_call_stage",
    "kakao_send_stage",
    "kakao_call_stage",
    "imessage_send_stage",
    "call_stage",
    "failure_stage",
    "stage",
)
REASON_KEYS = (
    "guard_reason",
    "reason",
    "failure_kind",
    "error_type",
    "exception_type",
    "status",
    "send_status",
    "call_status",
)
APPROVAL_HOLD_KEYS = (
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


def _short(value: object, *, limit: int = 120) -> str:
    try:
        text = "" if value is None else str(value)
    except Exception:
        text = "<unreadable>"
    text = text.strip()
    text = " ".join(text.split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except Exception:
        return default


def _has_metadata_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (bool, int, float)):
        return bool(value)
    if isinstance(value, (dict, list, tuple, set)):
        return len(value) > 0
    return True


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _metadata(row: Any) -> dict[str, Any]:
    raw = _row_value(row, "metadata", "{}")
    if not isinstance(raw, str):
        return {}
    raw = raw.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _created_at(row: Any) -> str:
    parsed = _parse_created_at(_row_value(row, "created_at"))
    if parsed is None:
        return "invalid_timestamp"
    return parsed.replace(microsecond=0).isoformat()


def _parse_created_at(value: object) -> datetime | None:
    text = _short(value, limit=160).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.removesuffix("Z"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        return parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def _failure_stage_value(key: str, value: object) -> str:
    text = _short(value, limit=80)
    if SAFE_STAGE_VALUE_RE.fullmatch(text):
        return text
    return f"{key}_detail_suppressed"


def _truthy_metadata(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _normalized_metadata_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _short(value, limit=160).strip().lower()).strip("_")


def _is_approval_hold(row: Any) -> bool:
    metadata = _metadata(row)
    if any(_truthy_metadata(metadata.get(key)) for key in APPROVAL_HOLD_KEYS):
        return True
    for key in (*STAGE_KEYS, *REASON_KEYS):
        token = _normalized_metadata_token(metadata.get(key))
        if token in APPROVAL_HOLD_VALUES:
            return True
        if "approval" in token and any(marker in token for marker in ("required", "requires", "gate", "gated", "hold", "held")):
            return True
    return False


def _is_preexecution_stop(row: Any) -> bool:
    """Return true only for a trusted executor receipt proving no handler ran."""

    metadata = _metadata(row)
    return type(metadata.get("executed_handler")) is bool and metadata.get("executed_handler") is False


def _failure_stage(row: Any) -> str:
    metadata = _metadata(row)
    for key in STAGE_KEYS:
        value = metadata.get(key)
        if _has_metadata_value(value):
            return _failure_stage_value(key, value)
    for key in REASON_KEYS:
        value = metadata.get(key)
        if _has_metadata_value(value):
            return _failure_stage_value(key, value)
    return "unknown_failure"


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "calls_model": False,
        "executes_tools": False,
        "queues_approval": False,
        "requires_approval": False,
        "approves_request": False,
        "dismisses_request": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "reads_message_content": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "controls_computer": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    metadata.update(extra)
    return metadata


def make_channel_health_tools(store: MemoryStore):
    def channel_health(_: dict[str, Any]) -> ToolResult:
        channel_tools = {tool for _, _, tools in CHANNELS for tool in tools}
        cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=WINDOW_DAYS)
        with store.connect() as conn:
            rows = list(
                conn.execute(
                    """
                    SELECT id, tool_name, ok, metadata, created_at
                    FROM tool_runs
                    WHERE tool_name IN ({placeholders})
                       OR tool_name LIKE 'call_%'
                    ORDER BY id DESC
                    LIMIT ?
                    """.format(placeholders=",".join("?" for _ in channel_tools)),
                    (*sorted(channel_tools), MAX_ROWS),
                )
            )

        rows_by_tool: dict[str, list[Any]] = {}
        for row in rows:
            tool_name = _short(_row_value(row, "tool_name"), limit=120)
            rows_by_tool.setdefault(tool_name, []).append(row)

        lines = [
            "Jarvis channel health report:",
            "Read-only audit metadata only. No recipient names, message text, approval text, or tool output snippets are shown.",
            f"Window: last {WINDOW_DAYS} days.",
            f"Rows reviewed: {len(rows)} / {MAX_ROWS} max"
            + (" (bounded sample; older matching rows may exist)." if len(rows) >= MAX_ROWS else "."),
            "",
        ]
        row_sample_truncated = len(rows) >= MAX_ROWS
        channel_summaries: list[dict[str, Any]] = []
        for channel_id, label, tools in CHANNELS:
            channel_rows = [row for tool in tools for row in rows_by_tool.get(tool, [])]
            channel_rows.sort(key=lambda row: _safe_int(_row_value(row, "id")), reverse=True)
            successes = [row for row in channel_rows if _safe_int(_row_value(row, "ok")) == 1]
            held_approvals = [
                row for row in channel_rows if _safe_int(_row_value(row, "ok")) == 0 and _is_approval_hold(row)
            ]
            preexecution_stops = [
                row
                for row in channel_rows
                if _safe_int(_row_value(row, "ok")) == 0
                and not _is_approval_hold(row)
                and _is_preexecution_stop(row)
            ]
            failures = [
                row
                for row in channel_rows
                if _safe_int(_row_value(row, "ok")) == 0
                and not _is_approval_hold(row)
                and not _is_preexecution_stop(row)
            ]
            recent_success_count = 0
            for row in successes:
                created = _parse_created_at(_row_value(row, "created_at"))
                if created is not None and created >= cutoff:
                    recent_success_count += 1
            last_success = _created_at(successes[0]) if successes else "never"
            last_failure = _created_at(failures[0]) if failures else "never"
            last_failure_stage = _failure_stage(failures[0]) if failures else "none"
            last_approval_hold = _created_at(held_approvals[0]) if held_approvals else "never"
            last_preexecution_stop = (
                _created_at(preexecution_stops[0]) if preexecution_stops else "never"
            )
            last_preexecution_stage = (
                _failure_stage(preexecution_stops[0]) if preexecution_stops else "none"
            )
            tool_list = ", ".join(tools)
            lines.append(
                f"- {label} ({tool_list}): last success {last_success}; "
                f"last failure {last_failure}; last failure stage {last_failure_stage}; "
                f"last approval hold {last_approval_hold}; "
                f"last pre-execution stop {last_preexecution_stop}; "
                f"last pre-execution stage {last_preexecution_stage}; "
                f"7d successes {recent_success_count}"
            )
            channel_summaries.append(
                {
                    "channel": channel_id,
                    "tools": list(tools),
                    "last_success_at": last_success,
                    "last_failure_at": last_failure,
                    "last_failure_stage": last_failure_stage,
                    "last_approval_hold_at": last_approval_hold,
                    "last_approval_hold_stage": "approval_required" if held_approvals else "none",
                    "last_preexecution_stop_at": last_preexecution_stop,
                    "last_preexecution_stop_stage": last_preexecution_stage,
                    "success_count_7d": recent_success_count,
                }
            )

        return ToolResult(
            "channel_health",
            True,
            "\n".join(lines),
            _safe_metadata(
                channel_count=len(CHANNELS),
                window_days=WINDOW_DAYS,
                rows_reviewed=len(rows),
                bounded_rows_limit=MAX_ROWS,
                content_suppressed=True,
                channels=channel_summaries,
                row_sample_truncated=row_sample_truncated,
            ),
        )

    return (channel_health,)
