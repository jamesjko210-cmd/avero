from __future__ import annotations

import json
from collections import Counter
from typing import Any, Callable

from jarvis_v2.agent.types import ToolResult
from jarvis_v2.memory.store import MemoryStore, STARTUP_RECOVERY_EXCEPTION_TYPES


TOOL_NAME = "startup_recovery_report"
DEFAULT_REPORT_LIMIT = 10
MAX_REPORT_LIMIT = 20
MAX_RECOVERY_COUNT = 9_223_372_036_854_775_807
MAX_DETAILS_CHARS = 4_096

RUN_STATES = frozenset({"running", "completed", "partial", "failed", "interrupted"})
COMPONENT_NAMES = frozenset({"person", "decision", "preference", "memory", "skill"})
COMPONENT_STATES = frozenset({"completed", "partial", "failed"})
COMPONENT_DETAIL_FIELDS = frozenset(
    {
        "status",
        "attempted",
        "completed",
        "pending",
        "custody_backfilled",
        "custody_conflicts",
        "exception_type",
    }
)
COUNT_FIELDS = (
    "attempted",
    "completed",
    "pending",
    "custody_backfilled",
    "custody_conflicts",
)


def _startup_recovery_audit_storage_guidance() -> str:
    return (
        "Run `storage status`, then `setup check`; repair the reported storage access "
        "and retry `startup recovery report`."
    )


def _bounded_limit(value: Any) -> int:
    if type(value) is not int:
        return DEFAULT_REPORT_LIMIT
    return max(1, min(value, MAX_REPORT_LIMIT))


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "metadata_only": True,
        "content_free": True,
        "content_exposed": False,
        "content_in_metadata": False,
        "reads_database_metadata": True,
        "reads_database_file": False,
        "reads_db_file_contents": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "reads_secret_values": False,
        "calls_model": False,
        "calls_external_service": False,
        "executes_tools": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "writes_files": False,
        "edits_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "approves_request": False,
        "dismisses_request": False,
        "requires_approval": False,
        "controls_computer": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "changed_state": False,
        "review_only": True,
    }
    metadata.update(extra)
    return metadata


def _row_keys(row: Any) -> set[str] | None:
    try:
        keys = row.keys()
    except Exception:
        return None
    try:
        return {key for key in keys if type(key) is str}
    except Exception:
        return None


def _row_value(row: Any, key: str) -> Any:
    try:
        return row[key]
    except Exception:
        raise ValueError("unreadable startup recovery row") from None


def _parse_nonnegative_count(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= MAX_RECOVERY_COUNT:
        raise ValueError("invalid startup recovery count")
    return value


def _parse_component_summary(component: Any, value: Any) -> dict[str, Any]:
    if type(component) is not str or component not in COMPONENT_NAMES:
        raise ValueError("invalid startup recovery component")
    if not isinstance(value, dict):
        raise ValueError("invalid startup recovery component detail")
    if any(type(key) is not str for key in value) or set(value) - COMPONENT_DETAIL_FIELDS:
        raise ValueError("invalid startup recovery component detail key")

    status = value.get("status")
    if type(status) is not str or status not in COMPONENT_STATES:
        raise ValueError("invalid startup recovery component status")

    counts: dict[str, int] = {}
    for field in COUNT_FIELDS:
        if field in value:
            counts[field] = _parse_nonnegative_count(value[field])
    exception_type = value.get("exception_type")
    if status == "failed":
        if exception_type not in STARTUP_RECOVERY_EXCEPTION_TYPES:
            raise ValueError("failed startup recovery component lacks a safe exception type")
        if counts:
            raise ValueError("failed startup recovery component carries recovery counts")
    elif exception_type is not None:
        raise ValueError("non-failed startup recovery component carries an exception type")
    if status != "failed" and not {"attempted", "completed", "pending"} <= set(counts):
        raise ValueError("non-failed startup recovery component lacks complete counts")
    if counts.get("completed", 0) > counts.get("attempted", 0):
        raise ValueError("startup recovery completed count exceeds attempted count")
    if status == "completed" and (
        counts.get("pending", 0) != 0
        or counts.get("custody_conflicts", 0) != 0
    ):
        raise ValueError("completed startup recovery component has unresolved work")
    if status == "partial" and (
        counts.get("pending", 0) == 0
        and counts.get("custody_conflicts", 0) == 0
    ):
        raise ValueError("partial startup recovery component lacks partial evidence")
    return {"component": component, "status": status, "counts": counts}


def _parse_component_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > len(COMPONENT_NAMES):
        raise ValueError("invalid startup recovery component list")
    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or any(type(key) is not str for key in item):
            raise ValueError("invalid startup recovery component row")
        component = item.get("component")
        if type(component) is not str or component in seen:
            raise ValueError("duplicate or invalid startup recovery component")
        summary = _parse_component_summary(
            component,
            {key: item_value for key, item_value in item.items() if key != "component"},
        )
        parsed.append(summary)
        seen.add(component)
    return parsed


def _parse_details(raw: Any) -> list[dict[str, Any]]:
    if type(raw) is str:
        if not raw or len(raw) > MAX_DETAILS_CHARS:
            raise ValueError("invalid startup recovery details")
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            raise ValueError("invalid startup recovery details") from None
        try:
            canonical = json.dumps(
                parsed,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        except (TypeError, ValueError):
            raise ValueError("invalid startup recovery details") from None
        if canonical != raw:
            raise ValueError("noncanonical startup recovery details")
        raw = parsed

    if isinstance(raw, list):
        return _parse_component_list(raw)
    if not isinstance(raw, dict) or any(type(key) is not str for key in raw):
        raise ValueError("invalid startup recovery details")

    if set(raw) == {"components"}:
        components = raw["components"]
        if isinstance(components, list):
            return _parse_component_list(components)
        if not isinstance(components, dict):
            raise ValueError("invalid startup recovery components")
        raw = components

    if len(raw) > len(COMPONENT_NAMES):
        raise ValueError("too many startup recovery components")
    parsed = [_parse_component_summary(component, value) for component, value in raw.items()]
    if len({item["component"] for item in parsed}) != len(parsed):
        raise ValueError("duplicate startup recovery component")
    return parsed


def _parse_run_row(row: Any) -> dict[str, Any]:
    keys = _row_keys(row)
    if keys is None:
        raise ValueError("unreadable startup recovery row")

    id_key = "id" if "id" in keys else "run_id" if "run_id" in keys else ""
    state_key = "state" if "state" in keys else "status" if "status" in keys else ""
    details_key = (
        "component_summaries"
        if "component_summaries" in keys
        else "details"
        if "details" in keys
        else "component_details"
        if "component_details" in keys
        else ""
    )
    if not id_key or not state_key or not details_key:
        raise ValueError("incomplete startup recovery row")

    run_id = _row_value(row, id_key)
    state = _row_value(row, state_key)
    if type(run_id) is not int or run_id < 1:
        raise ValueError("invalid startup recovery run id")
    if type(state) is not str or state not in RUN_STATES:
        raise ValueError("invalid startup recovery run state")

    components = _parse_details(_row_value(row, details_key))
    component_names = {item["component"] for item in components}
    if state in {"completed", "partial"} and component_names != COMPONENT_NAMES:
        raise ValueError("finalized startup recovery row is incomplete")
    if state == "completed" and any(item["status"] != "completed" for item in components):
        raise ValueError("completed startup recovery row has incomplete components")
    if state == "partial" and all(item["status"] == "completed" for item in components):
        raise ValueError("partial startup recovery row has no partial component")
    component_status_counts = Counter(item["status"] for item in components)
    recovery_counts: Counter[str] = Counter()
    for item in components:
        recovery_counts.update(item["counts"])

    return {
        "status": state,
        "component_count": len(components),
        "component_status_counts": dict(sorted(component_status_counts.items())),
        "recovery_counts": {
            field: recovery_counts[field]
            for field in COUNT_FIELDS
            if recovery_counts[field]
        },
    }


def _summary_line(label: str, summary: dict[str, Any]) -> str:
    status_counts = summary["component_status_counts"]
    status_text = ", ".join(f"{key}={value}" for key, value in status_counts.items()) or "none"
    recovery_counts = summary["recovery_counts"]
    count_text = ", ".join(f"{key}={value}" for key, value in recovery_counts.items()) or "none"
    return (
        f"- {label}: status {summary['status']}; components {summary['component_count']}; "
        f"component statuses {status_text}; recovery counts {count_text}."
    )


def _handoff(
    *,
    limit: int,
    latest: dict[str, Any] | None,
    history: list[dict[str, Any]],
    readable_rows: int,
    unreadable_rows: int,
    latest_unreadable: bool,
) -> dict[str, Any]:
    status_counts = Counter(item["status"] for item in ([latest] if latest else []) + history)
    return {
        "source": TOOL_NAME,
        "handoff_ready": True,
        "startup_recovery_handoff_ready": True,
        "limit": limit,
        "latest": latest,
        "latest_unreadable": latest_unreadable,
        "history": history,
        "history_count": len(history),
        "readable_run_rows": readable_rows,
        "unreadable_run_rows": unreadable_rows,
        "status_counts": dict(sorted(status_counts.items())),
        "review_only": True,
        "metadata_only": True,
        "content_free": True,
        "content_in_handoff": False,
        "loads_without_execution": True,
        "calls_model": False,
        "executes_tools": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "external_side_effect": False,
        "controls_computer": False,
        "queues_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "refresh_command": "startup recovery report",
    }


def make_startup_recovery_report_tool(
    store: MemoryStore,
) -> Callable[[dict[str, Any]], ToolResult]:
    def startup_recovery_report(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_limit(args.get("limit") if type(args) is dict else None)
        try:
            raw_rows = store.list_startup_recovery_runs(limit)
        except Exception:
            handoff = _handoff(
                limit=limit,
                latest=None,
                history=[],
                readable_rows=0,
                unreadable_rows=0,
                latest_unreadable=False,
            )
            return ToolResult(
                TOOL_NAME,
                False,
                (
                    "Startup recovery audit metadata is unavailable. No recovery action was taken. "
                    f"{_startup_recovery_audit_storage_guidance()}"
                ),
                _safe_metadata(
                    limit=limit,
                    report_status="unavailable",
                    latest=None,
                    history=[],
                    readable_run_rows=0,
                    unreadable_run_rows=0,
                    startup_recovery_handoff=handoff,
                    startup_recovery_handoff_ready=True,
                    failure_kind="startup_recovery_audit_unavailable",
                    next_command="storage status",
                    recovery_commands=[
                        "storage status",
                        "setup check",
                        "startup recovery report",
                    ],
                    retry_requires_storage_repair=True,
                    authorizes_retry=False,
                ),
            )

        if type(raw_rows) not in {list, tuple}:
            raw_rows = [raw_rows]
        try:
            raw_rows = list(raw_rows[:limit])
        except Exception:
            raw_rows = [None]
        parsed_by_position: list[dict[str, Any] | None] = []
        unreadable_rows = 0
        for row in raw_rows:
            try:
                parsed_by_position.append(_parse_run_row(row))
            except Exception:
                parsed_by_position.append(None)
                unreadable_rows += 1

        latest = parsed_by_position[0] if parsed_by_position else None
        latest_unreadable = bool(parsed_by_position and latest is None)
        history = [item for item in parsed_by_position[1:] if item is not None]
        readable_rows = sum(item is not None for item in parsed_by_position)
        handoff = _handoff(
            limit=limit,
            latest=latest,
            history=history,
            readable_rows=readable_rows,
            unreadable_rows=unreadable_rows,
            latest_unreadable=latest_unreadable,
        )

        lines = [
            "Jarvis startup recovery audit report:",
            "Read-only, content-free recovery statuses and counts only.",
            f"Bounded rows reviewed: {len(raw_rows)} / {limit} max.",
        ]
        if not raw_rows:
            lines.append("Latest: no startup recovery run recorded.")
        elif latest_unreadable:
            lines.append("Latest: unreadable row hidden for safety.")
        elif latest is not None:
            lines.extend(["Latest:", _summary_line("latest", latest)])

        lines.append(f"History: {len(history)} readable prior run(s).")
        for index, summary in enumerate(history, start=1):
            lines.append(_summary_line(f"prior {index}", summary))
        if unreadable_rows:
            lines.append(f"Unreadable rows hidden for safety: {unreadable_rows}.")
        lines.append("Handoff is review-only and does not authorize or execute recovery work.")

        return ToolResult(
            TOOL_NAME,
            True,
            "\n".join(lines),
            _safe_metadata(
                limit=limit,
                rows_reviewed=len(raw_rows),
                report_status="available",
                latest=latest,
                latest_unreadable=latest_unreadable,
                history=history,
                history_count=len(history),
                readable_run_rows=readable_rows,
                unreadable_run_rows=unreadable_rows,
                status_counts=handoff["status_counts"],
                startup_recovery_handoff=handoff,
                startup_recovery_handoff_ready=True,
            ),
        )

    return startup_recovery_report


def make_startup_recovery_tools(
    store: MemoryStore,
) -> tuple[Callable[[dict[str, Any]], ToolResult], ...]:
    return (make_startup_recovery_report_tool(store),)
