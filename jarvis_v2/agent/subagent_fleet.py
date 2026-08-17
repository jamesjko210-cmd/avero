"""Parallel subagent execution with unified memory sharing.

Enables multiple subagents to:
- Execute in parallel (not sequentially)
- Share memory context with Jarvis and each other
- Auto-sync results back to Jarvis memory
- Track readiness state (0-N agents ready)
"""

from __future__ import annotations

import asyncio
import copy
import math
import re
import sqlite3
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis_v2.memory.store import MemoryStore


class AgentState(Enum):
    """Agent readiness states."""
    IDLE = "idle"
    WARMING = "warming"
    READY = "ready"
    EXECUTING = "executing"
    FAILED = "failed"


MAX_TASK_TIMEOUT_SECONDS = 600.0
MAX_TASK_ID_LENGTH = 80
TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$")
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")
MAX_ERROR_LENGTH = 160
SUPPRESSED_PUBLIC_VALUE = "<suppressed>"


def _snapshot_value(value: Any) -> Any:
    """Return a detached snapshot suitable for shared-memory/status surfaces."""
    try:
        return copy.deepcopy(value)
    except Exception:
        return {"unserializable": type(value).__name__}


def _safe_error_text(value: Any) -> str:
    try:
        raw = "" if value is None else str(value)
    except Exception:
        raw = f"<unreadable:{type(value).__name__}>"
    text = LOCAL_PATH_RE.sub("<local-path>", raw)
    text = " ".join(text.split())
    if len(text) > MAX_ERROR_LENGTH:
        return text[: MAX_ERROR_LENGTH - 1].rstrip() + "..."
    return text


def _public_snapshot_key(value: Any) -> str:
    return _safe_error_text(value) or "<blank>"


def _public_snapshot_value(value: Any) -> Any:
    """Return non-sensitive metadata for exported fleet snapshots."""
    if value is None:
        return None
    return {
        "value": SUPPRESSED_PUBLIC_VALUE,
        "type": type(value).__name__,
    }


def _public_context_snapshot(items: list[tuple[str, Any]]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for index, (key, value) in enumerate(items, start=1):
        safe_key = _public_snapshot_key(key)
        if safe_key in context:
            safe_key = f"{safe_key}#{index}"
        context[safe_key] = _public_snapshot_value(value)
    return context


def _public_memory_snapshot(items: list[tuple[str, SharedMemory]]) -> dict[str, Any]:
    memories: dict[str, Any] = {}
    for index, (agent_id, memory) in enumerate(items, start=1):
        safe_key = _public_snapshot_key(agent_id)
        if safe_key in memories:
            safe_key = f"{safe_key}#{index}"
        memories[safe_key] = memory.to_dict()
    return memories


@dataclass
class SharedMemory:
    """Unified memory context shared across all agents + Jarvis."""
    agent_id: str
    last_update: str = field(default_factory=lambda: datetime.now().isoformat())
    context: dict[str, Any] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def update(self, key: str, value: Any) -> None:
        """Write to shared memory (thread-safe)."""
        snapshot = _snapshot_value(value)
        self._store_snapshot(key, snapshot)

    def _store_snapshot(self, key: str, snapshot: Any) -> None:
        """Store a value already detached from its caller."""
        self._store_snapshots([(key, snapshot)])

    def _store_snapshots(self, updates: list[tuple[str, Any]]) -> None:
        """Store detached values as one visible memory update."""
        if not updates:
            return
        updated_at = datetime.now().isoformat()
        with self.lock:
            for key, snapshot in updates:
                self.context[key] = snapshot
            self.last_update = updated_at

    def get(self, key: str, default: Any = None) -> Any:
        """Read from shared memory (thread-safe)."""
        with self.lock:
            value = self.context.get(key, default)
        return _snapshot_value(value)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API/dashboard."""
        with self.lock:
            agent_id = self.agent_id
            last_update = self.last_update
            context = list(self.context.items())
        return {
            "agent_id": _public_snapshot_key(agent_id),
            "last_update": last_update,
            "context": _public_context_snapshot(context),
        }


class _StagedSharedMemory:
    """Execution-scoped memory view whose writes require a matching commit token."""

    def __init__(self, memory: SharedMemory, token: object) -> None:
        self.agent_id = memory.agent_id
        self._memory = memory
        self._token = token
        self._updates: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._accepting_writes = True

    def update(self, key: str, value: Any) -> None:
        """Stage a detached write while this execution remains active."""
        snapshot = _snapshot_value(value)
        with self._lock:
            if self._accepting_writes:
                self._updates[key] = snapshot

    def get(self, key: str, default: Any = None) -> Any:
        """Read this execution's staged value, then fall back to shared memory."""
        with self._lock:
            if key in self._updates:
                return _snapshot_value(self._updates[key])
        return self._memory.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Serialize a public snapshot including this execution's staged view."""
        with self._memory.lock:
            last_update = self._memory.last_update
            context = dict(self._memory.context)
        with self._lock:
            context.update(self._updates)
        return {
            "agent_id": _public_snapshot_key(self.agent_id),
            "last_update": last_update,
            "context": _public_context_snapshot(list(context.items())),
        }

    def _finish(self, token: object) -> list[tuple[str, Any]]:
        """Close the facade and yield staged writes only to its execution token."""
        with self._lock:
            self._accepting_writes = False
            if self._token is not token:
                self._updates.clear()
                return []
            updates = list(self._updates.items())
            self._updates.clear()
            return updates

    def _discard(self, token: object) -> None:
        """Close the facade and discard writes owned by this execution token."""
        with self._lock:
            if self._token is token:
                self._accepting_writes = False
                self._updates.clear()


@dataclass
class SubagentTask:
    """A unit of work for a subagent."""
    task_id: str
    description: str
    instructions: str
    priority: int = 0
    timeout_seconds: int = 60
    result: Any = None
    error: str | None = None
    state: AgentState = AgentState.IDLE


@dataclass
class SubagentFleet:
    """Manages a fleet of parallel subagents with shared memory."""
    agents: dict[str, SubagentTask] = field(default_factory=dict)
    shared_memories: dict[str, SharedMemory] = field(default_factory=dict)
    ready_count: int = 0
    total_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    _execution_slots: dict[str, object] = field(default_factory=dict, repr=False, compare=False)
    _durable_store: MemoryStore | None = field(default=None, repr=False, compare=False)
    _runtime_id: str = field(default="", repr=False, compare=False)

    def _recompute_ready_count_locked(self) -> None:
        """Refresh the visible readiness count after any state transition."""
        self.ready_count = sum(
            1 for agent in self.agents.values()
            if agent.state in (AgentState.READY, AgentState.IDLE)
        )

    def register_agent(self, agent_id: str, total: int) -> None:
        """Register a new agent in the fleet."""
        with self.lock:
            self._execution_slots.pop(agent_id, None)
            self.shared_memories[agent_id] = SharedMemory(agent_id=agent_id)
            self.agents[agent_id] = SubagentTask(
                task_id=agent_id,
                description=f"Subagent {agent_id}",
                instructions="",
                state=AgentState.READY
            )
            self.total_count = max(total, len(self.agents))
            self._recompute_ready_count_locked()

    def mark_ready(self, agent_id: str) -> None:
        """Mark an agent as ready."""
        with self.lock:
            if agent_id in self.agents:
                self._execution_slots.pop(agent_id, None)
                agent = self.agents[agent_id]
                agent.state = AgentState.READY
                agent.task_id = agent_id
                agent.instructions = ""
                agent.result = None
                agent.error = None
                self._recompute_ready_count_locked()

    def mark_executing(self, agent_id: str, task_id: str, instructions: str) -> None:
        """Mark an agent as executing a task."""
        with self.lock:
            if agent_id in self.agents:
                self._execution_slots.pop(agent_id, None)
                self.agents[agent_id].state = AgentState.EXECUTING
                self.agents[agent_id].task_id = task_id
                self.agents[agent_id].instructions = instructions
                self.agents[agent_id].result = None
                self.agents[agent_id].error = None
                self._recompute_ready_count_locked()

    def mark_failed(self, agent_id: str, error: str) -> None:
        """Mark an agent as failed."""
        safe_error = _safe_error_text(error)
        with self.lock:
            if agent_id in self.agents:
                self._execution_slots.pop(agent_id, None)
                self.agents[agent_id].state = AgentState.FAILED
                self.agents[agent_id].result = None
                self.agents[agent_id].error = safe_error
                self._recompute_ready_count_locked()

    def store_result(self, agent_id: str, result: Any) -> None:
        """Store task result and sync to shared memory."""
        stored_result = _snapshot_value(result)
        memory: SharedMemory | None = None
        with self.lock:
            if agent_id in self.agents:
                self._execution_slots.pop(agent_id, None)
                self.agents[agent_id].result = stored_result
                self.agents[agent_id].state = AgentState.READY
                self.agents[agent_id].error = None
            if agent_id in self.shared_memories:
                memory = self.shared_memories[agent_id]
            self._recompute_ready_count_locked()
        if memory is not None:
            memory.update("last_result", stored_result)

    def _claim_execution(
        self,
        agent_id: str,
        task_id: str,
        instructions: str,
        total: int,
    ) -> tuple[object, SharedMemory] | None:
        """Atomically claim an idle execution slot without disturbing a busy agent."""
        with self.lock:
            agent = self.agents.get(agent_id)
            if agent_id in self._execution_slots or (
                agent is not None and agent.state is AgentState.EXECUTING
            ):
                return None

            memory = self.shared_memories.get(agent_id)
            if memory is None:
                memory = SharedMemory(agent_id=agent_id)
                self.shared_memories[agent_id] = memory
            if agent is None:
                agent = SubagentTask(
                    task_id=agent_id,
                    description=f"Subagent {agent_id}",
                    instructions="",
                    state=AgentState.READY,
                )
                self.agents[agent_id] = agent

            token = object()
            self._execution_slots[agent_id] = token
            agent.state = AgentState.EXECUTING
            agent.task_id = task_id
            agent.instructions = instructions
            agent.result = None
            agent.error = None
            self.total_count = max(total, len(self.agents), self.total_count)
            self._recompute_ready_count_locked()
            return token, memory

    def _fail_execution(self, agent_id: str, token: object, error: str) -> bool:
        """Fail only the execution that still owns the agent's slot."""
        safe_error = _safe_error_text(error)
        with self.lock:
            if self._execution_slots.get(agent_id) is not token:
                return False
            agent = self.agents[agent_id]
            self._execution_slots.pop(agent_id, None)
            agent.state = AgentState.FAILED
            agent.result = None
            agent.error = safe_error
            self._recompute_ready_count_locked()
            return True

    def _quarantine_execution(self, agent_id: str, token: object, error: str) -> bool:
        """Fail an execution while retaining its slot until detached work exits."""
        safe_error = _safe_error_text(error)
        with self.lock:
            if self._execution_slots.get(agent_id) is not token:
                return False
            agent = self.agents[agent_id]
            agent.state = AgentState.FAILED
            agent.result = None
            agent.error = safe_error
            self._recompute_ready_count_locked()
            return True

    def _release_execution(self, agent_id: str, token: object) -> bool:
        """Release a slot only if the same execution still owns it."""
        with self.lock:
            if self._execution_slots.get(agent_id) is not token:
                return False
            self._execution_slots.pop(agent_id, None)
            return True

    def _store_execution_result(
        self,
        agent_id: str,
        token: object,
        result: Any,
        staged_memory: _StagedSharedMemory,
    ) -> bool:
        """Publish a result only while its execution token remains current."""
        staged_updates = staged_memory._finish(token)
        stored_result = _snapshot_value(result)
        memory_result = _snapshot_value(stored_result)
        with self.lock:
            if self._execution_slots.get(agent_id) is not token:
                return False
            agent = self.agents[agent_id]
            memory = self.shared_memories.get(agent_id)
            self._execution_slots.pop(agent_id, None)
            agent.result = stored_result
            agent.state = AgentState.READY
            agent.error = None
            if memory is not None:
                memory._store_snapshots(
                    [*staged_updates, ("last_result", memory_result)]
                )
            self._recompute_ready_count_locked()
            return True

    def get_shared_memory(self, agent_id: str) -> SharedMemory | None:
        """Get an agent's shared memory."""
        with self.lock:
            return self.shared_memories.get(agent_id)

    def broadcast_to_all(self, key: str, value: Any) -> None:
        """Broadcast a context value to all agents' shared memory."""
        with self.lock:
            memories = list(self.shared_memories.values())
        for memory in memories:
            memory.update(key, value)

    def get_ready_count(self) -> tuple[int, int]:
        """Return (ready_agents, total_agents)."""
        with self.lock:
            return (self.ready_count, self.total_count)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API/dashboard."""
        with self.lock:
            ready_agents = self.ready_count
            total_agents = self.total_count
            agent_rows = list(self.agents.items())
            memories = list(self.shared_memories.items())

        agents: dict[str, Any] = {}
        for index, (agent_id, agent) in enumerate(agent_rows, start=1):
            safe_agent_id = _public_snapshot_key(agent_id)
            safe_agent_key = safe_agent_id
            if safe_agent_key in agents:
                safe_agent_key = f"{safe_agent_key}#{index}"
            agents[safe_agent_key] = {
                "id": safe_agent_id,
                "state": agent.state.value,
                "task_id": _public_snapshot_key(agent.task_id),
                "result": _public_snapshot_value(agent.result),
                "error": _public_snapshot_value(agent.error),
            }

        snapshot = {
            "ready_agents": ready_agents,
            "total_agents": total_agents,
            "agents": agents,
            "memories": _public_memory_snapshot(memories),
        }
        if self._durable_store is not None:
            try:
                snapshot["durable_control"] = self._durable_store.subagent_control_snapshot()
            except Exception as exc:
                snapshot["durable_control"] = {
                    "ok": False,
                    "readiness_verified": False,
                    "dispatch_ready": False,
                    "diagnostic": type(exc).__name__,
                }
        return snapshot


def _duplicate_task_ids(tasks: list[SubagentTask]) -> list[str]:
    counts = Counter(task.task_id for task in tasks)
    return sorted(task_id for task_id, count in counts.items() if count > 1)


def _invalid_task_id_errors(tasks: list[SubagentTask]) -> list[tuple[int, str]]:
    errors: list[tuple[int, str]] = []
    for index, task in enumerate(tasks):
        task_id = task.task_id
        if not isinstance(task_id, str):
            errors.append((index, f"invalid task id: {type(task_id).__name__}"))
            continue
        if not task_id:
            errors.append((index, "invalid task id: blank"))
            continue
        if task_id != task_id.strip():
            errors.append((index, "invalid task id: surrounding whitespace"))
            continue
        if len(task_id) > MAX_TASK_ID_LENGTH:
            errors.append((index, "invalid task id: too long"))
            continue
        if not TASK_ID_RE.fullmatch(task_id):
            errors.append((index, "invalid task id: unsafe characters"))
    return errors


def _task_timeout_seconds(value: Any) -> tuple[float | None, str | None]:
    if isinstance(value, bool):
        return None, "invalid timeout seconds: bool"
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return None, f"invalid timeout seconds: {type(value).__name__}"
    if not math.isfinite(timeout):
        return None, "invalid timeout seconds: non-finite"
    if timeout <= 0:
        return None, "invalid timeout seconds: must be positive"
    return min(timeout, MAX_TASK_TIMEOUT_SECONDS), None


def _consume_background_task(task: asyncio.Task[Any]) -> None:
    """Retrieve a detached task's outcome so it cannot emit loop warnings."""
    try:
        task.result()
    except BaseException:
        pass


def _cancel_background_task(task: asyncio.Task[Any], cleanup: Callable[[], None]) -> None:
    def finish(completed: asyncio.Task[Any]) -> None:
        _consume_background_task(completed)
        cleanup()

    task.add_done_callback(finish)
    task.cancel()


async def run_parallel_subagents(
    fleet: SubagentFleet,
    tasks: list[SubagentTask],
    handler: Callable[[str, SubagentTask, _StagedSharedMemory], Coroutine[Any, Any, Any]],
) -> list[Any]:
    """Execute multiple tasks in parallel, sharing memory across agents.

    Args:
        fleet: The SubagentFleet managing all agents
        tasks: List of SubagentTasks to execute
        handler: Async function(agent_id, task, shared_memory) -> result

    Returns:
        List of results in the same order as tasks
    """
    invalid_task_ids = _invalid_task_id_errors(tasks)
    if invalid_task_ids:
        invalid_agent_count = max(1, len(invalid_task_ids))
        for index, error in invalid_task_ids:
            agent_id = f"invalid-task-{index + 1:02d}"
            claim = fleet._claim_execution(
                agent_id,
                agent_id,
                "invalid task id refused before dispatch",
                invalid_agent_count,
            )
            if claim is not None:
                token, _ = claim
                fleet._fail_execution(agent_id, token, error)
        return [None for _ in tasks]

    duplicate_ids = _duplicate_task_ids(tasks)
    if duplicate_ids:
        unique_agent_count = len({task.task_id for task in tasks})
        for task_id in duplicate_ids:
            claim = fleet._claim_execution(
                task_id,
                task_id,
                "duplicate task id refused before dispatch",
                unique_agent_count,
            )
            if claim is not None:
                token, _ = claim
                fleet._fail_execution(
                    task_id,
                    token,
                    "duplicate task id: concurrent assignment refused",
                )
        return [None for _ in tasks]

    async def execute_task(task: SubagentTask) -> tuple[str, Any]:
        agent_id = task.task_id
        claim = fleet._claim_execution(
            agent_id,
            task.task_id,
            task.instructions,
            len(tasks),
        )
        if claim is None:
            return agent_id, None
        token, memory = claim

        timeout_seconds, timeout_error = _task_timeout_seconds(task.timeout_seconds)
        if timeout_error:
            fleet._fail_execution(agent_id, token, timeout_error)
            return agent_id, None

        staged_memory = _StagedSharedMemory(memory, token)
        try:
            handler_task = asyncio.create_task(handler(agent_id, task, staged_memory))
        except Exception as exc:
            staged_memory._discard(token)
            fleet._fail_execution(agent_id, token, f"error: {exc}")
            return agent_id, None

        try:
            done, _ = await asyncio.wait({handler_task}, timeout=timeout_seconds)
        except asyncio.CancelledError:
            staged_memory._discard(token)
            fleet._quarantine_execution(agent_id, token, "cancelled")
            _cancel_background_task(
                handler_task,
                lambda: fleet._release_execution(agent_id, token),
            )
            raise

        if handler_task not in done:
            staged_memory._discard(token)
            fleet._quarantine_execution(
                agent_id,
                token,
                f"timeout after {timeout_seconds:g}s",
            )
            _cancel_background_task(
                handler_task,
                lambda: fleet._release_execution(agent_id, token),
            )
            return agent_id, None

        try:
            result = handler_task.result()
        except asyncio.CancelledError:
            staged_memory._discard(token)
            fleet._fail_execution(agent_id, token, "handler cancelled")
            return agent_id, None
        except Exception as exc:
            staged_memory._discard(token)
            fleet._fail_execution(agent_id, token, f"error: {exc}")
            return agent_id, None

        if fleet._store_execution_result(agent_id, token, result, staged_memory):
            return agent_id, result
        return agent_id, None

    dispatch_tasks = [asyncio.create_task(execute_task(task)) for task in tasks]
    try:
        results = await asyncio.gather(*dispatch_tasks, return_exceptions=False)
    except asyncio.CancelledError:
        for dispatch_task in dispatch_tasks:
            dispatch_task.cancel()
        await asyncio.gather(*dispatch_tasks, return_exceptions=True)
        raise
    return [result for _, result in results]


def spawn_subagent_fleet(
    count: int,
    *,
    store: MemoryStore | None = None,
    runtime_id: str = "",
) -> SubagentFleet:
    """Create a new subagent fleet with N agents pre-registered."""
    fleet = SubagentFleet(_durable_store=store, _runtime_id=runtime_id)
    for i in range(count):
        agent_id = f"subagent-{i+1:02d}"
        fleet.register_agent(agent_id, count)
        fleet.mark_ready(agent_id)
    if store is not None:
        try:
            store.configure_subagent_workers(count)
        except (sqlite3.Error, OSError):
            # Worker orchestration is optional infrastructure. A read-only or
            # degraded store must not prevent the core runtime from starting;
            # the durable status snapshot will fail closed instead.
            pass
    return fleet
