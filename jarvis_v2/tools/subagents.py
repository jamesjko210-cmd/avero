from __future__ import annotations

from collections import Counter
import re
from typing import Any, Callable

from jarvis_v2.agent.types import ToolResult


LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")


def _scrub_text(value: Any, *, limit: int = 160) -> str:
    try:
        raw = "" if value is None else str(value)
    except Exception:
        raw = f"<unreadable:{type(value).__name__}>"
    text = LOCAL_PATH_RE.sub("<local-path>", raw)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "..."
    return text


def _safe_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return 0
    try:
        count = int(value)
    except Exception:
        return 0
    if count < 0:
        return 0
    return count


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _safe_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return False


def _state_count_pairs(value: Any) -> list[tuple[str, int]]:
    if not isinstance(value, dict):
        return []

    pairs: list[tuple[str, int]] = []
    for raw_state, raw_count in value.items():
        state = _scrub_text(raw_state, limit=48) or "unknown"
        count = _safe_int(raw_count)
        if count <= 0:
            continue
        pairs.append((state, count))
    return sorted(pairs, key=lambda pair: pair[0])


def _unavailable_fleet_snapshot(diagnostic: Any) -> dict[str, Any]:
    return {
        "ok": False,
        "ready_agents": 0,
        "total_agents": 0,
        "agent_count": 0,
        "memory_count": 0,
        "state_counts": {},
        "agents": [],
        "unreadable_agent_rows": 0,
        "results_suppressed": True,
        "memory_context_suppressed": True,
        "diagnostic": _scrub_text(diagnostic, limit=80) or "subagent_fleet_unavailable",
        "next_command": "setup check",
        "recovery_commands": ["setup check", "jarvis status", "subagent fleet status"],
        "restart_required_if_still_unavailable": True,
        "authorizes_restart": False,
        "authorizes_worker_dispatch": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
    }


def _durable_fleet_status_snapshot(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return _unavailable_fleet_snapshot("durable_subagent_control_malformed")
    if type(raw.get("ok")) is not bool or type(raw.get("readiness_verified")) is not bool:
        return _unavailable_fleet_snapshot("durable_subagent_readiness_malformed")
    if raw.get("ok") is not True or raw.get("readiness_verified") is not True:
        return _unavailable_fleet_snapshot("durable_subagent_readiness_unverified")
    integer_keys = (
        "configured_workers",
        "ready_workers",
        "stale_online_workers",
        "unreadable_worker_rows",
        "unreadable_assignment_rows",
    )
    if any(type(raw.get(key)) is not int or int(raw.get(key)) < 0 for key in integer_keys):
        return _unavailable_fleet_snapshot("durable_subagent_counts_malformed")
    configured = int(raw["configured_workers"])
    ready = int(raw["ready_workers"])
    if ready > configured:
        return _unavailable_fleet_snapshot("durable_subagent_counts_inconsistent")
    raw_runner_attached = raw.get("runner_attached")
    queued = int((raw.get("task_counts") or {}).get("queued", 0)) if isinstance(raw.get("task_counts"), dict) else -1
    expected_attached = any(
        isinstance(row, dict) and row.get("live") is True
        for row in (raw.get("workers") if isinstance(raw.get("workers"), list) else [])
    )
    expected_dispatch = expected_attached and ready > 0 and queued > 0
    if type(raw_runner_attached) is not bool or raw_runner_attached != expected_attached:
        return _unavailable_fleet_snapshot("durable_subagent_runner_truth_mismatch")
    if type(raw.get("dispatch_ready")) is not bool or raw.get("dispatch_ready") != expected_dispatch:
        return _unavailable_fleet_snapshot("durable_subagent_dispatch_truth_mismatch")
    raw_workers = raw.get("workers")
    if not isinstance(raw_workers, list) or len(raw_workers) > 50:
        return _unavailable_fleet_snapshot("durable_subagent_workers_malformed")
    agents: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    for row in raw_workers:
        if not isinstance(row, dict):
            return _unavailable_fleet_snapshot("durable_subagent_worker_row_malformed")
        worker_id = _scrub_text(row.get("worker_id"), limit=80)
        state = _scrub_text(row.get("state"), limit=48)
        if re.fullmatch(r"subagent-[0-9]{2}", worker_id) is None or state not in {
            "configured",
            "online",
            "offline",
        }:
            return _unavailable_fleet_snapshot("durable_subagent_worker_row_invalid")
        if type(row.get("live")) is not bool or type(row.get("claimable")) is not bool:
            return _unavailable_fleet_snapshot("durable_subagent_worker_truth_malformed")
        if row["claimable"] and (not row["live"] or state != "online" or row.get("has_assignment") is True):
            return _unavailable_fleet_snapshot("durable_subagent_worker_truth_inconsistent")
        generation = row.get("generation")
        if type(generation) is not int or generation < 0 or (state == "online" and generation < 1):
            return _unavailable_fleet_snapshot("durable_subagent_worker_generation_malformed")
        state_counts[state] += 1
        agents.append(
            {
                "id": worker_id,
                "state": state,
                "task_id": "",
                "has_result": False,
                "has_error": False,
                "live": row["live"],
                "claimable": row["claimable"],
                "has_assignment": row.get("has_assignment") is True,
            }
        )
    if len(agents) != configured and raw.get("workers_truncated") is not True:
        return _unavailable_fleet_snapshot("durable_subagent_worker_inventory_inconsistent")
    if sum(1 for row in agents if row["claimable"]) != ready:
        return _unavailable_fleet_snapshot("durable_subagent_ready_count_inconsistent")
    task_counts = raw.get("task_counts")
    assignment_counts = raw.get("assignment_counts")
    if not isinstance(task_counts, dict) or not isinstance(assignment_counts, dict):
        return _unavailable_fleet_snapshot("durable_subagent_lifecycle_counts_malformed")
    allowed_task_states = ("queued", "running", "succeeded", "failed", "uncertain")
    allowed_assignment_states = ("claimed", "running", "succeeded", "failed", "abandoned", "uncertain")
    if any(
        state not in allowed_task_states or type(count) is not int or count < 0
        for state, count in task_counts.items()
    ) or any(
        state not in allowed_assignment_states or type(count) is not int or count < 0
        for state, count in assignment_counts.items()
    ):
        return _unavailable_fleet_snapshot("durable_subagent_lifecycle_counts_invalid")
    bounded_task_counts = {state: int(task_counts.get(state, 0)) for state in allowed_task_states}
    bounded_assignment_counts = {
        state: int(assignment_counts.get(state, 0)) for state in allowed_assignment_states
    }
    active_count = bounded_assignment_counts["claimed"] + bounded_assignment_counts["running"]
    raw_active = raw.get("active_assignments")
    if not isinstance(raw_active, list) or len(raw_active) > 50:
        return _unavailable_fleet_snapshot("durable_subagent_active_assignments_malformed")
    if len(raw_active) != active_count and raw.get("active_assignments_truncated") is not True:
        return _unavailable_fleet_snapshot("durable_subagent_active_inventory_inconsistent")
    for row in raw_active:
        if not isinstance(row, dict):
            return _unavailable_fleet_snapshot("durable_subagent_active_assignment_row_malformed")
        if (
            re.fullmatch(r"assignment-[0-9a-f]{24}", _scrub_text(row.get("assignment_id"), limit=80)) is None
            or re.fullmatch(r"task-[0-9a-f]{24}", _scrub_text(row.get("task_id"), limit=80)) is None
            or re.fullmatch(r"subagent-[0-9]{2}", _scrub_text(row.get("worker_id"), limit=80)) is None
            or row.get("state") not in {"claimed", "running"}
            or type(row.get("lease_expired")) is not bool
        ):
            return _unavailable_fleet_snapshot("durable_subagent_active_assignment_row_invalid")
    return {
        "ok": True,
        "readiness_verified": True,
        "durable_control": True,
        "dispatch_ready": raw["dispatch_ready"],
        "runner_attached": raw_runner_attached,
        "ready_agents": ready,
        "total_agents": configured,
        "agent_count": configured,
        "memory_count": 0,
        "state_counts": dict(state_counts),
        "agents": agents,
        "unreadable_agent_rows": 0,
        "unreadable_assignment_rows": int(raw["unreadable_assignment_rows"]),
        "stale_online_workers": int(raw["stale_online_workers"]),
        "task_counts": bounded_task_counts,
        "assignment_counts": bounded_assignment_counts,
        "active_assignment_count": active_count,
        "results_suppressed": True,
        "memory_context_suppressed": True,
        "payloads_persisted": False,
        "results_persisted": False,
        "errors_persisted": False,
        "lease_tokens_exposed": False,
        "diagnostic": "",
        "next_command": "",
        "recovery_commands": [],
        "restart_required_if_still_unavailable": False,
        "authorizes_restart": False,
        "authorizes_worker_dispatch": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
    }


def subagent_fleet_status_snapshot(get_fleet: Callable[[], Any] | None) -> dict[str, Any]:
    if get_fleet is None:
        return _unavailable_fleet_snapshot("subagent_fleet_unavailable")

    try:
        fleet = get_fleet()
        raw = fleet.to_dict()
    except Exception as exc:
        return _unavailable_fleet_snapshot(type(exc).__name__)

    if isinstance(raw, dict) and "durable_control" in raw:
        return _durable_fleet_status_snapshot(raw.get("durable_control"))

    raw_agents = raw.get("agents") if isinstance(raw, dict) else {}
    unreadable_agent_rows = 0
    if not isinstance(raw_agents, dict):
        unreadable_agent_rows = 1 if raw_agents else 0
        raw_agents = {}

    state_counts: Counter[str] = Counter()
    agents: list[dict[str, Any]] = []
    for agent_id in sorted(raw_agents, key=lambda value: _scrub_text(value, limit=80)):
        try:
            row = raw_agents.get(agent_id)
            if not isinstance(row, dict):
                unreadable_agent_rows += 1
                continue
            state = _scrub_text(row.get("state"), limit=48) or "unknown"
            state_counts[state] += 1
            agents.append(
                {
                    "id": _scrub_text(row.get("id") if row.get("id") is not None else agent_id, limit=80),
                    "state": state,
                    "task_id": _scrub_text(row.get("task_id"), limit=120),
                    "has_result": row.get("result") is not None,
                    "has_error": _has_value(row.get("error")),
                }
            )
        except Exception:
            unreadable_agent_rows += 1

    raw_memories = raw.get("memories") if isinstance(raw, dict) else {}
    memory_count = len(raw_memories) if isinstance(raw_memories, dict) else 0
    if not isinstance(raw, dict):
        return _unavailable_fleet_snapshot("subagent_fleet_snapshot_malformed")
    raw_ready = raw.get("ready_agents")
    raw_total = raw.get("total_agents")
    if (
        type(raw_ready) is not int
        or type(raw_total) is not int
        or raw_ready < 0
        or raw_total < 0
        or raw_ready > raw_total
        or unreadable_agent_rows != 0
        or len(agents) != raw_total
        or sum(state_counts.get(state, 0) for state in ("ready", "idle")) != raw_ready
    ):
        return _unavailable_fleet_snapshot("subagent_fleet_readiness_inconsistent")
    return {
        "ok": True,
        "ready_agents": raw_ready,
        "total_agents": raw_total,
        "agent_count": len(agents),
        "memory_count": memory_count,
        "state_counts": dict(state_counts),
        "agents": agents,
        "unreadable_agent_rows": unreadable_agent_rows,
        "results_suppressed": True,
        "memory_context_suppressed": True,
        "diagnostic": "",
        "next_command": "",
        "recovery_commands": [],
        "restart_required_if_still_unavailable": False,
        "authorizes_restart": False,
        "authorizes_worker_dispatch": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
    }


def format_subagent_fleet_status(snapshot: dict[str, Any]) -> str:
    if not isinstance(snapshot, dict):
        snapshot = {"ok": False, "diagnostic": type(snapshot).__name__}

    ready = _safe_int(snapshot.get("ready_agents"))
    total = _safe_int(snapshot.get("total_agents"))
    if not snapshot.get("ok"):
        diagnostic = _scrub_text(snapshot.get("diagnostic"), limit=80) or "unavailable"
        return (
            f"Internal worker fleet status: unavailable ({diagnostic}).\n"
            "- Run `setup check`, then `jarvis status`.\n"
            "- If the fleet is still unavailable, restart Jarvis with its normal launcher and retry "
            "`subagent fleet status`.\n"
            "- This status does not authorize restart, worker dispatch, tool execution, or a completion claim."
        )

    lines = [
        "Internal worker fleet status:",
        f"- ready workers: {ready} / {total}",
        "- raw results: suppressed",
        "- shared memory context: suppressed",
    ]
    if snapshot.get("durable_control") is True:
        lines.extend(
            [
                "- durable control plane: verified",
                f"- runner attached: {'yes' if snapshot.get('runner_attached') is True else 'no'}",
                f"- queue ready for claims: {'yes' if snapshot.get('dispatch_ready') is True else 'no'}",
                "- configured slots are not counted as ready without a live runner heartbeat",
            ]
        )
        task_counts = _state_count_pairs(snapshot.get("task_counts"))
        if task_counts:
            lines.append("- durable tasks: " + ", ".join(f"{state}={count}" for state, count in task_counts))
        assignment_counts = _state_count_pairs(snapshot.get("assignment_counts"))
        if assignment_counts:
            lines.append("- durable attempts: " + ", ".join(f"{state}={count}" for state, count in assignment_counts))
    state_counts = _state_count_pairs(snapshot.get("state_counts"))
    if state_counts:
        counts = ", ".join(f"{state}={count}" for state, count in state_counts)
        lines.append(f"- states: {counts}")
    hidden_rows = _safe_int(snapshot.get("unreadable_agent_rows"))
    if hidden_rows:
        lines.append(f"- hidden agent rows: {hidden_rows}")
    for row in list(snapshot.get("agents") or [])[:12]:
        if not isinstance(row, dict):
            continue
        task_id = _scrub_text(row.get("task_id"), limit=120)
        suffix = f" | task: {task_id}" if task_id else ""
        if _safe_flag(row.get("has_error")):
            suffix += " | error hidden"
        elif _safe_flag(row.get("has_result")):
            suffix += " | result hidden"
        agent_id = _scrub_text(row.get("id"), limit=80) or "unknown"
        state = _scrub_text(row.get("state"), limit=48) or "unknown"
        lines.append(f"- {agent_id}: {state}{suffix}")
    return "\n".join(lines)


def make_subagent_fleet_status_tool(get_fleet: Callable[[], Any] | None):
    def subagent_fleet_status(_: dict[str, Any]) -> ToolResult:
        snapshot = subagent_fleet_status_snapshot(get_fleet)
        return ToolResult(
            "subagent_fleet_status",
            bool(snapshot.get("ok")),
            format_subagent_fleet_status(snapshot),
            {
                **snapshot,
                "calls_model": False,
                "executes_tools": False,
                "queues_approval": False,
                "requires_approval": False,
                "external_side_effect": False,
                "reads_personal_data": False,
                "authorizes_restart": False,
                "authorizes_worker_dispatch": False,
                "authorizes_execution": False,
                "authorizes_completion_claim": False,
                "approval_granted": False,
            },
        )

    return subagent_fleet_status
