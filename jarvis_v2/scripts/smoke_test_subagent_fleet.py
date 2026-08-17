from __future__ import annotations

import asyncio
import threading
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.subagent_fleet import (
    AgentState,
    SubagentFleet,
    SubagentTask,
    run_parallel_subagents,
    spawn_subagent_fleet,
)
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.subagents import (
    format_subagent_fleet_status,
    make_subagent_fleet_status_tool,
    subagent_fleet_status_snapshot,
)
from jarvis_v2.ui.status_server import build_status_snapshot


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def _assert_suppressed(value, expected_type: str, message: str) -> None:
    _assert(isinstance(value, dict), f"{message}: expected suppressed metadata dict, got {value!r}")
    _assert(value.get("value") == "<suppressed>", f"{message}: missing suppressed marker: {value}")
    _assert(value.get("type") == expected_type, f"{message}: expected type {expected_type!r}: {value}")


class _PathLikeValue:
    def __str__(self) -> str:
        return "/\x55sers/example/private-subagent-status.txt"


class _HostileTruthiness:
    def __bool__(self) -> bool:
        raise RuntimeError("/\x55sers/example/hostile-subagent-bool")

    def __str__(self) -> str:
        return "/\x55sers/example/hostile-subagent-bool"


class _ExplodingValue:
    def __str__(self) -> str:
        raise RuntimeError("/\x55sers/example/hidden-subagent-value")


class _HostileFleet:
    def to_dict(self):
        return {
            "ready_agents": 1,
            "total_agents": 1,
            "agents": {
                _ExplodingValue(): {
                    "id": _ExplodingValue(),
                    "state": _ExplodingValue(),
                    "task_id": _ExplodingValue(),
                    "result": None,
                    "error": _ExplodingValue(),
                }
            },
            "memories": {},
        }


class _BlockingSnapshotValue:
    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        self.entered = entered
        self.release = release

    def __deepcopy__(self, memo):
        self.entered.set()
        self.release.wait(timeout=2)
        return {"released": self.release.is_set()}


class _MalformedFleet:
    def to_dict(self):
        return {
            "ready_agents": "not-an-int",
            "total_agents": _PathLikeValue(),
            "agents": {
                2: {
                    "id": "/\x55sers/example/agent-two",
                    "state": "/private/tmp/executing-state",
                    "task_id": "/var/folders/hidden-task",
                    "result": {"raw": "suppressed"},
                    "error": None,
                },
                "alpha": {
                    "id": "alpha",
                    "state": "failed",
                    "task_id": "/\x55sers/example/secret-task",
                    "result": None,
                    "error": "/\x55sers/example/secret-error",
                },
                "bad-row": "not a row dict",
            },
            "memories": {"subagent-01": {"context": {"hidden": "/\x55sers/example/memory"}}},
        }


class _SpoofedCountFleet:
    def to_dict(self):
        return {
            "ready_agents": True,
            "total_agents": -2,
            "agents": {
                "subagent-01": {
                    "id": "subagent-01",
                    "state": "ready",
                    "task_id": "subagent-01",
                    "result": None,
                    "error": None,
                }
            },
            "memories": {},
        }


async def _exercise_parallel_success() -> None:
    fleet = spawn_subagent_fleet(2)
    _assert(fleet.get_ready_count() == (2, 2), f"spawned fleet should start ready: {fleet.to_dict()}")

    fleet.broadcast_to_all("mission", "visible control plane")
    entered: set[str] = set()
    checked: set[str] = set()
    all_entered = asyncio.Event()
    all_checked = asyncio.Event()

    async def handler(agent_id: str, task: SubagentTask, memory):
        _assert(memory.get("mission") == "visible control plane", f"{agent_id} missed broadcast memory")
        entered.add(agent_id)
        if len(entered) == 2:
            all_entered.set()
        await asyncio.wait_for(all_entered.wait(), timeout=1)
        ready, total = fleet.get_ready_count()
        _assert(total == 2, f"total count changed during execution: {(ready, total)}")
        _assert(ready == 0, f"executing agents must not be counted as ready: {(ready, total)}")
        checked.add(agent_id)
        if len(checked) == 2:
            all_checked.set()
        await asyncio.wait_for(all_checked.wait(), timeout=1)
        return {"agent_id": agent_id, "description": task.description}

    tasks = [
        SubagentTask(task_id="subagent-01", description="research", instructions="inspect safely"),
        SubagentTask(task_id="subagent-02", description="verify", instructions="test safely"),
    ]
    results = await run_parallel_subagents(fleet, tasks, handler)
    _assert(
        [result["agent_id"] for result in results] == ["subagent-01", "subagent-02"],
        f"parallel results must preserve task order: {results}",
    )
    _assert(fleet.get_ready_count() == (2, 2), f"completed agents should return ready: {fleet.to_dict()}")

    snapshot = fleet.to_dict()
    for agent_id, result in zip(["subagent-01", "subagent-02"], results, strict=True):
        agent = snapshot["agents"][agent_id]
        public_memory = snapshot["memories"][agent_id]["context"]
        actual_memory = fleet.get_shared_memory(agent_id)
        _assert(actual_memory is not None, f"{agent_id} memory missing after success")
        _assert(agent["state"] == AgentState.READY.value, f"{agent_id} should be ready after success: {agent}")
        _assert(agent["error"] is None, f"{agent_id} should clear stale errors after success: {agent}")
        _assert_suppressed(agent["result"], "dict", f"{agent_id} result should be suppressed in snapshots")
        _assert(actual_memory.get("last_result") == result, f"{agent_id} did not sync result to memory")
        _assert(actual_memory.get("mission") == "visible control plane", f"{agent_id} lost broadcast memory")
        _assert_suppressed(
            public_memory.get("last_result"),
            "dict",
            f"{agent_id} public last_result should be suppressed",
        )
        _assert_suppressed(
            public_memory.get("mission"),
            "str",
            f"{agent_id} public mission should be suppressed",
        )


async def _exercise_successful_staged_memory_commit() -> None:
    fleet = spawn_subagent_fleet(1)
    memory = fleet.get_shared_memory("subagent-01")
    _assert(memory is not None, "staged commit smoke lost shared memory")
    memory.update("existing", {"visible": True})
    write_staged = asyncio.Event()
    release_handler = asyncio.Event()

    async def handler(agent_id: str, task: SubagentTask, staged_memory):
        _assert(
            staged_memory.get("existing") == {"visible": True},
            "staged facade could not read existing shared memory",
        )
        payload = {"committed": [1]}
        staged_memory.update("handler_update", payload)
        payload["committed"].append(2)
        _assert(
            staged_memory.get("handler_update") == {"committed": [1]},
            "handler could not read its detached staged update",
        )
        write_staged.set()
        await release_handler.wait()
        return {"status": "complete"}

    task = SubagentTask(
        task_id="subagent-01",
        description="commit staged memory",
        instructions="publish only with success",
    )
    running = asyncio.create_task(run_parallel_subagents(fleet, [task], handler))
    await asyncio.wait_for(write_staged.wait(), timeout=1)
    _assert(
        memory.get("handler_update") is None,
        "handler update became visible before successful completion",
    )
    release_handler.set()
    results = await asyncio.wait_for(running, timeout=1)
    _assert(results == [{"status": "complete"}], f"staged success result drifted: {results}")
    _assert(
        memory.get("handler_update") == {"committed": [1]},
        "successful handler did not commit its staged update",
    )
    _assert(
        memory.get("last_result") == {"status": "complete"},
        "successful staged commit did not publish its result",
    )


async def _exercise_failure_accounting() -> None:
    fleet = spawn_subagent_fleet(2)

    async def handler(agent_id: str, task: SubagentTask, memory):
        if agent_id == "subagent-01":
            memory.update("failed_handler_write", "must be discarded")
            raise RuntimeError("planned failure")
        return "ok"

    tasks = [
        SubagentTask(task_id="subagent-01", description="fail", instructions="fail safely"),
        SubagentTask(task_id="subagent-02", description="recover", instructions="stay ready"),
    ]
    results = await run_parallel_subagents(fleet, tasks, handler)
    _assert(results == [None, "ok"], f"failure path should preserve result order: {results}")
    _assert(fleet.get_ready_count() == (1, 2), f"failed agent must not count as ready: {fleet.to_dict()}")

    snapshot = fleet.to_dict()
    failed = snapshot["agents"]["subagent-01"]
    recovered = snapshot["agents"]["subagent-02"]
    _assert(failed["state"] == AgentState.FAILED.value, f"failed agent state wrong: {failed}")
    _assert(
        "planned failure" in (fleet.agents["subagent-01"].error or ""),
        f"failed agent should retain bounded internal error: {fleet.agents['subagent-01'].error}",
    )
    _assert_suppressed(failed["error"], "str", f"failed agent public error should be suppressed: {failed}")
    _assert(recovered["state"] == AgentState.READY.value, f"successful peer should be ready: {recovered}")
    recovered_memory = fleet.get_shared_memory("subagent-02")
    _assert(recovered_memory is not None, "successful peer memory missing")
    _assert(recovered_memory.get("last_result") == "ok", "success result not synced")
    _assert_suppressed(
        snapshot["memories"]["subagent-02"]["context"].get("last_result"),
        "str",
        "success result should be suppressed in public memory snapshot",
    )
    _assert("last_result" not in snapshot["memories"]["subagent-01"]["context"], "failed task should not fake a result")
    failed_memory = fleet.get_shared_memory("subagent-01")
    _assert(failed_memory is not None, "failed agent memory missing")
    _assert(
        failed_memory.get("failed_handler_write") is None,
        "failed handler committed a staged memory write",
    )


async def _exercise_handler_error_diagnostics_are_bounded() -> None:
    fleet = spawn_subagent_fleet(1)

    async def handler(agent_id: str, task: SubagentTask, memory):
        raise RuntimeError("/\x55sers/example/private/가상연락처이-token.txt raw-handle leaked")

    tasks = [
        SubagentTask(task_id="subagent-01", description="private failure", instructions="fail safely"),
    ]
    results = await run_parallel_subagents(fleet, tasks, handler)
    _assert(results == [None], f"private handler failure should return None in task order: {results}")

    snapshot = fleet.to_dict()
    failed = snapshot["agents"]["subagent-01"]
    _assert(failed["state"] == AgentState.FAILED.value, f"handler failure should fail the worker: {failed}")
    _assert(failed["result"] is None, f"handler failure leaked a result: {failed}")
    _assert_suppressed(failed["error"], "str", f"handler failure public error should be suppressed: {failed}")
    _assert(
        "<local-path>" in (fleet.agents["subagent-01"].error or ""),
        f"handler failure should retain bounded internal diagnostic: {fleet.agents['subagent-01'].error}",
    )

    status = subagent_fleet_status_snapshot(lambda: fleet)
    formatted = format_subagent_fleet_status(status)
    blob = str(snapshot) + str(status) + formatted
    for forbidden in ["/\x55sers/", "operator", "가상연락처이-token", "raw-handle leaked"]:
        _assert(forbidden not in blob, f"subagent handler failure leaked {forbidden!r}: {blob}")
    _assert("error hidden" in formatted, f"formatted status should hide handler error body: {formatted}")


async def _exercise_handler_error_without_path_is_suppressed_publicly() -> None:
    fleet = spawn_subagent_fleet(1)

    async def handler(agent_id: str, task: SubagentTask, memory):
        raise RuntimeError("가상연락처이 raw-handle leaked without a path")

    tasks = [
        SubagentTask(task_id="subagent-01", description="private failure", instructions="fail safely"),
    ]
    results = await run_parallel_subagents(fleet, tasks, handler)
    _assert(results == [None], f"private non-path handler failure should return None in task order: {results}")

    internal_error = fleet.agents["subagent-01"].error or ""
    _assert("가상연락처이 raw-handle leaked" in internal_error, f"internal recovery diagnostic missing: {internal_error}")
    snapshot = fleet.to_dict()
    _assert_suppressed(
        snapshot["agents"]["subagent-01"]["error"],
        "str",
        f"non-path handler failure public error should be suppressed: {snapshot}",
    )
    status = subagent_fleet_status_snapshot(lambda: fleet)
    formatted = format_subagent_fleet_status(status)
    _assert(status["agents"][0]["has_error"] is True, f"status should preserve error presence: {status}")
    _assert("error hidden" in formatted, f"formatted status should hide non-path error body: {formatted}")
    blob = str(snapshot) + str(status) + formatted
    for forbidden in ["가상연락처이", "raw-handle leaked"]:
        _assert(forbidden not in blob, f"public non-path handler error leaked {forbidden!r}: {blob}")


async def _exercise_duplicate_parallel_task_ids_fail_closed() -> None:
    fleet = spawn_subagent_fleet(2)
    fleet.store_result("subagent-01", {"old": "result"})
    called = 0

    async def handler(agent_id: str, task: SubagentTask, memory):
        nonlocal called
        called += 1
        raise AssertionError("duplicate task ids must fail before handler execution")

    duplicate_tasks = [
        SubagentTask(task_id="subagent-01", description="first", instructions="one"),
        SubagentTask(task_id="subagent-01", description="second", instructions="two"),
    ]
    results = await run_parallel_subagents(fleet, duplicate_tasks, handler)
    _assert(results == [None, None], f"duplicate ids should return empty results in order: {results}")
    _assert(called == 0, f"duplicate ids should not call handler: {called}")
    _assert(fleet.get_ready_count() == (1, 2), f"duplicate failed worker should not count ready: {fleet.to_dict()}")

    snapshot = fleet.to_dict()
    failed = snapshot["agents"]["subagent-01"]
    ready = snapshot["agents"]["subagent-02"]
    _assert(failed["state"] == AgentState.FAILED.value, f"duplicate worker should fail closed: {failed}")
    _assert(failed["task_id"] == "subagent-01", f"duplicate worker task id should stay bounded: {failed}")
    _assert(failed["result"] is None, f"duplicate failure leaked stale visible result: {failed}")
    _assert_suppressed(failed["error"], "str", f"duplicate failure public error should be suppressed: {failed}")
    _assert(
        "duplicate task id" in (fleet.agents["subagent-01"].error or ""),
        f"duplicate failure lost internal diagnostic: {fleet.agents['subagent-01'].error}",
    )
    _assert(ready["state"] == AgentState.READY.value, f"unassigned peer should stay ready: {ready}")
    status = subagent_fleet_status_snapshot(lambda: fleet)
    _assert(status["state_counts"] == {"failed": 1, "ready": 1}, f"duplicate status counts wrong: {status}")
    failed_row = next(row for row in status["agents"] if row["id"] == "subagent-01")
    _assert(failed_row["has_result"] is False, f"public status exposed duplicate stale result: {status}")
    _assert(failed_row["has_error"] is True, f"public status lost duplicate failure marker: {status}")

    fresh_fleet = spawn_subagent_fleet(0)
    fresh_tasks = [
        SubagentTask(task_id="fresh-worker", description="first", instructions="one"),
        SubagentTask(task_id="fresh-worker", description="second", instructions="two"),
    ]
    fresh_results = await run_parallel_subagents(fresh_fleet, fresh_tasks, handler)
    _assert(fresh_results == [None, None], f"fresh duplicate ids should fail in order: {fresh_results}")
    _assert(called == 0, f"fresh duplicate ids should not call handler: {called}")
    _assert(fresh_fleet.get_ready_count() == (0, 1), f"fresh duplicate ready count wrong: {fresh_fleet.to_dict()}")
    fresh_failed = fresh_fleet.to_dict()["agents"]["fresh-worker"]
    _assert(fresh_failed["state"] == AgentState.FAILED.value, f"fresh duplicate worker should fail: {fresh_failed}")
    _assert_suppressed(fresh_failed["error"], "str", f"fresh duplicate public error should be suppressed: {fresh_failed}")
    _assert(
        "duplicate task id" in (fresh_fleet.agents["fresh-worker"].error or ""),
        f"fresh duplicate internal diagnostic missing: {fresh_fleet.agents['fresh-worker'].error}",
    )


async def _exercise_outer_cancellation_cleans_up_and_reraises() -> None:
    fleet = spawn_subagent_fleet(1)
    entered = asyncio.Event()
    exited = asyncio.Event()

    async def blocking_handler(agent_id: str, task: SubagentTask, memory):
        memory.update("cancelled_handler_write", "must be discarded")
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            exited.set()

    task = SubagentTask(
        task_id="subagent-01",
        description="cancelled outer run",
        instructions="wait for cancellation",
    )
    outer_run = asyncio.create_task(run_parallel_subagents(fleet, [task], blocking_handler))
    await asyncio.wait_for(entered.wait(), timeout=1)
    _assert(
        fleet.agents["subagent-01"].state is AgentState.EXECUTING,
        f"cancellation precondition should be executing: {fleet.to_dict()}",
    )

    outer_run.cancel()
    try:
        await outer_run
    except asyncio.CancelledError:
        pass
    else:
        raise SystemExit("outer run cancellation must propagate to its caller")

    await asyncio.wait_for(exited.wait(), timeout=1)
    cancelled = fleet.agents["subagent-01"]
    _assert(cancelled.state is AgentState.FAILED, f"cancelled agent remained active: {fleet.to_dict()}")
    _assert(cancelled.result is None, f"cancelled agent retained a result: {cancelled.result!r}")
    _assert(cancelled.error == "cancelled", f"cancelled agent diagnostic drifted: {cancelled.error!r}")
    cancelled_memory = fleet.get_shared_memory("subagent-01")
    _assert(cancelled_memory is not None, "cancelled agent memory missing")
    _assert(
        cancelled_memory.get("cancelled_handler_write") is None,
        "outer cancellation committed a staged memory write",
    )

    async def retry_handler(agent_id: str, task: SubagentTask, memory):
        return "reused"

    retry_results = await run_parallel_subagents(fleet, [task], retry_handler)
    _assert(retry_results == ["reused"], f"cancelled execution slot was not reusable: {retry_results}")
    _assert(fleet.get_ready_count() == (1, 1), f"reused agent should return ready: {fleet.to_dict()}")


async def _exercise_concurrent_same_agent_claim_is_atomic() -> None:
    fleet = spawn_subagent_fleet(1)
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_handler_calls = 0

    async def first_handler(agent_id: str, task: SubagentTask, memory):
        first_entered.set()
        await release_first.wait()
        return {"winner": "first"}

    async def second_handler(agent_id: str, task: SubagentTask, memory):
        nonlocal second_handler_calls
        second_handler_calls += 1
        return {"winner": "second"}

    first_task = SubagentTask(
        task_id="subagent-01",
        description="first dispatch",
        instructions="retain active ownership",
    )
    second_task = SubagentTask(
        task_id="subagent-01",
        description="second dispatch",
        instructions="must be refused",
    )
    first_run = asyncio.create_task(run_parallel_subagents(fleet, [first_task], first_handler))
    await asyncio.wait_for(first_entered.wait(), timeout=1)

    second_results = await asyncio.wait_for(
        run_parallel_subagents(fleet, [second_task], second_handler),
        timeout=0.25,
    )
    _assert(second_results == [None], f"busy same-agent dispatch should be refused: {second_results}")
    _assert(second_handler_calls == 0, f"busy same-agent handler was entered: {second_handler_calls}")
    active = fleet.agents["subagent-01"]
    _assert(active.state is AgentState.EXECUTING, f"refusal corrupted active state: {fleet.to_dict()}")
    _assert(active.instructions == "retain active ownership", f"refusal overwrote active task: {active}")
    _assert(active.result is None and active.error is None, f"refusal wrote terminal data: {active}")

    release_first.set()
    first_results = await asyncio.wait_for(first_run, timeout=1)
    _assert(first_results == [{"winner": "first"}], f"active dispatch lost its result: {first_results}")
    _assert(
        fleet.agents["subagent-01"].result == {"winner": "first"},
        f"refused dispatch overwrote the active result: {fleet.agents['subagent-01'].result}",
    )
    memory = fleet.get_shared_memory("subagent-01")
    _assert(memory is not None, "same-agent concurrency smoke lost shared memory")
    _assert(memory.get("last_result") == {"winner": "first"}, "shared memory retained a refused result")


async def _exercise_timeout_is_bounded_when_handler_suppresses_cancellation() -> None:
    fleet = spawn_subagent_fleet(1)
    entered = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release_late_handler = asyncio.Event()
    late_handler_finished = asyncio.Event()
    late_poison_attempted = asyncio.Event()
    replacement_calls = 0
    memory = fleet.get_shared_memory("subagent-01")
    _assert(memory is not None, "bounded timeout smoke lost shared memory")
    memory.update("stable", "original")

    async def cancellation_suppressing_handler(agent_id: str, task: SubagentTask, staged_memory):
        _assert(staged_memory.get("stable") == "original", "timed handler lost existing memory reads")
        staged_memory.update("stable", "early poison")
        staged_memory.update("early_poison", "must be discarded")
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            staged_memory.update("stable", "late poison")
            staged_memory.update("late_poison", "must be discarded")
            late_poison_attempted.set()
            await release_late_handler.wait()
        late_handler_finished.set()
        return "late success"

    timed_out = SubagentTask(
        task_id="subagent-01",
        description="suppress cancellation",
        instructions="finish after timeout",
        timeout_seconds=0.01,
    )
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    results = await run_parallel_subagents(fleet, [timed_out], cancellation_suppressing_handler)
    elapsed = loop.time() - started_at

    _assert(results == [None], f"suppressed cancellation accepted a late result: {results}")
    _assert(entered.is_set(), "bounded timeout did not enter the handler")
    _assert(elapsed < 0.25, f"timeout waited for cancellation-suppressing handler: {elapsed:.3f}s")
    await asyncio.wait_for(cancellation_seen.wait(), timeout=1)
    await asyncio.wait_for(late_poison_attempted.wait(), timeout=1)
    timed_out_agent = fleet.agents["subagent-01"]
    _assert(timed_out_agent.state is AgentState.FAILED, f"bounded timeout state wrong: {fleet.to_dict()}")
    _assert(timed_out_agent.result is None, f"bounded timeout stored a result: {timed_out_agent.result!r}")
    _assert(timed_out_agent.error == "timeout after 0.01s", timed_out_agent.error or "missing timeout")
    _assert(memory.get("stable") == "original", "timed-out handler poisoned existing shared memory")
    _assert(memory.get("early_poison") is None, "pre-timeout staged write escaped quarantine")
    _assert(memory.get("late_poison") is None, "late cancellation-suppressed write escaped quarantine")

    async def replacement_handler(agent_id: str, task: SubagentTask, memory):
        nonlocal replacement_calls
        replacement_calls += 1
        return "replacement success"

    replacement = SubagentTask(
        task_id="subagent-01",
        description="same-id replacement",
        instructions="must wait for quarantined handler",
    )
    refused = await asyncio.wait_for(
        run_parallel_subagents(fleet, [replacement], replacement_handler),
        timeout=0.25,
    )
    _assert(refused == [None], f"quarantined agent accepted overlapping work: {refused}")
    _assert(replacement_calls == 0, f"replacement handler overlapped old work: {replacement_calls}")
    _assert(not late_handler_finished.is_set(), "old handler ended before overlap refusal was verified")
    _assert(
        timed_out_agent.error == "timeout after 0.01s",
        f"same-id refusal overwrote timeout state: {timed_out_agent.error!r}",
    )

    release_late_handler.set()
    await asyncio.wait_for(late_handler_finished.wait(), timeout=1)
    await asyncio.sleep(0)
    _assert(timed_out_agent.state is AgentState.FAILED, f"late success changed timeout state: {fleet.to_dict()}")
    _assert(timed_out_agent.result is None, f"late success overwrote timeout result: {timed_out_agent.result!r}")
    _assert(memory.get("last_result") is None, "late success leaked into shared memory")
    _assert(memory.get("stable") == "original", "late handler completion poisoned shared memory")
    _assert(memory.get("late_poison") is None, "late handler completion committed discarded memory")

    replacement_results = await asyncio.wait_for(
        run_parallel_subagents(fleet, [replacement], replacement_handler),
        timeout=1,
    )
    _assert(replacement_results == ["replacement success"], replacement_results)
    _assert(replacement_calls == 1, f"replacement handler call count drifted: {replacement_calls}")
    _assert(
        memory.get("last_result") == "replacement success",
        "completed replacement did not publish its own result",
    )


async def _exercise_timeout_diagnostics_fail_closed() -> None:
    fleet = spawn_subagent_fleet(1)
    entered = asyncio.Event()

    async def slow_handler(agent_id: str, task: SubagentTask, memory):
        entered.set()
        await asyncio.sleep(0.05)
        return "late"

    timed_out = [
        SubagentTask(
            task_id="subagent-01",
            description="timeout",
            instructions="sleep too long",
            timeout_seconds=0.01,
        )
    ]
    results = await run_parallel_subagents(fleet, timed_out, slow_handler)
    _assert(results == [None], f"timeout should return None in task order: {results}")
    _assert(entered.is_set(), "valid timeout should call the handler before timing out")
    failed = fleet.to_dict()["agents"]["subagent-01"]
    _assert(failed["state"] == AgentState.FAILED.value, f"timed-out worker should fail: {failed}")
    _assert(failed["result"] is None, f"timeout leaked a result: {failed}")
    _assert_suppressed(failed["error"], "str", f"timeout public error should be suppressed: {failed}")
    _assert(
        fleet.agents["subagent-01"].error == "timeout after 0.01s",
        f"timeout internal diagnostic should name bound: {fleet.agents['subagent-01'].error}",
    )
    status = subagent_fleet_status_snapshot(lambda: fleet)
    failed_row = status["agents"][0]
    _assert(failed_row["has_result"] is False, f"timeout status exposed stale result: {status}")
    _assert(failed_row["has_error"] is True, f"timeout status lost failure marker: {status}")

    invalid_fleet = spawn_subagent_fleet(1)
    called = 0

    async def should_not_run(agent_id: str, task: SubagentTask, memory):
        nonlocal called
        called += 1
        return "should not run"

    invalid = [
        SubagentTask(
            task_id="subagent-01",
            description="invalid timeout",
            instructions="bad timeout",
            timeout_seconds=0,
        )
    ]
    invalid_results = await run_parallel_subagents(invalid_fleet, invalid, should_not_run)
    _assert(invalid_results == [None], f"invalid timeout should fail in task order: {invalid_results}")
    _assert(called == 0, f"invalid timeout should not call handler: {called}")
    invalid_failed = invalid_fleet.to_dict()["agents"]["subagent-01"]
    _assert(invalid_failed["state"] == AgentState.FAILED.value, f"invalid timeout worker should fail: {invalid_failed}")
    _assert(invalid_failed["result"] is None, f"invalid timeout leaked a result: {invalid_failed}")
    _assert_suppressed(
        invalid_failed["error"],
        "str",
        f"invalid timeout public error should be suppressed: {invalid_failed}",
    )
    _assert(
        invalid_fleet.agents["subagent-01"].error == "invalid timeout seconds: must be positive",
        f"invalid timeout internal diagnostic should be bounded: {invalid_fleet.agents['subagent-01'].error}",
    )


async def _exercise_invalid_task_ids_fail_closed() -> None:
    fleet = spawn_subagent_fleet(1)
    called = 0

    async def handler(agent_id: str, task: SubagentTask, memory):
        nonlocal called
        called += 1
        return "should not run"

    invalid_tasks = [
        SubagentTask(task_id="subagent-01", description="valid peer", instructions="stay untouched"),
        SubagentTask(task_id=["not", "hashable"], description="bad id", instructions="reject"),
    ]
    results = await run_parallel_subagents(fleet, invalid_tasks, handler)
    _assert(results == [None, None], f"invalid task-id batch should fail in order: {results}")
    _assert(called == 0, f"invalid task ids should not call handler: {called}")
    _assert(fleet.get_ready_count() == (1, 2), f"valid peer should stay ready, invalid row failed: {fleet.to_dict()}")
    snapshot = fleet.to_dict()
    valid_peer = snapshot["agents"]["subagent-01"]
    failed = snapshot["agents"]["invalid-task-02"]
    _assert(valid_peer["state"] == AgentState.READY.value, f"valid peer should not execute: {valid_peer}")
    _assert(failed["state"] == AgentState.FAILED.value, f"invalid task id should fail closed: {failed}")
    _assert(failed["task_id"] == "invalid-task-02", f"invalid row should use synthetic id: {failed}")
    _assert(failed["result"] is None, f"invalid task id leaked a result: {failed}")
    _assert_suppressed(failed["error"], "str", f"invalid task id public error should be suppressed: {failed}")
    _assert(
        fleet.agents["invalid-task-02"].error == "invalid task id: list",
        f"invalid task id internal diagnostic wrong: {fleet.agents['invalid-task-02'].error}",
    )
    status = subagent_fleet_status_snapshot(lambda: fleet)
    _assert(status["state_counts"] == {"failed": 1, "ready": 1}, f"invalid id status counts wrong: {status}")
    failed_row = next(row for row in status["agents"] if row["id"] == "invalid-task-02")
    _assert(failed_row["has_result"] is False, f"invalid id status exposed result: {status}")
    _assert(failed_row["has_error"] is True, f"invalid id status lost error marker: {status}")

    path_fleet = spawn_subagent_fleet(0)
    path_tasks = [
        SubagentTask(task_id="/\x55sers/example/raw-worker", description="path id", instructions="reject"),
    ]
    path_results = await run_parallel_subagents(path_fleet, path_tasks, handler)
    _assert(path_results == [None], f"path-shaped task id should fail in order: {path_results}")
    _assert(called == 0, f"path-shaped task id should not call handler: {called}")
    path_snapshot = path_fleet.to_dict()
    path_failed = path_snapshot["agents"]["invalid-task-01"]
    _assert(path_failed["state"] == AgentState.FAILED.value, f"path-shaped task id should fail: {path_failed}")
    _assert_suppressed(path_failed["error"], "str", f"path-shaped task id public error should be suppressed: {path_failed}")
    _assert(
        path_fleet.agents["invalid-task-01"].error == "invalid task id: unsafe characters",
        f"path diagnostic wrong: {path_fleet.agents['invalid-task-01'].error}",
    )
    blob = str(path_snapshot) + str(subagent_fleet_status_snapshot(lambda: path_fleet))
    for forbidden in ["/\x55sers/", "operator", "raw-worker"]:
        _assert(forbidden not in blob, f"invalid task id leaked {forbidden!r}: {blob}")


def _exercise_task_transitions_clear_stale_result_state() -> None:
    fleet = spawn_subagent_fleet(1)
    fleet.store_result("subagent-01", {"old": "success"})
    completed = fleet.to_dict()["agents"]["subagent-01"]
    _assert(fleet.agents["subagent-01"].result == {"old": "success"}, "initial internal success result missing")
    _assert_suppressed(completed["result"], "dict", f"initial success result should be suppressed: {completed}")

    fleet.mark_executing("subagent-01", "fresh-task", "retry safely")
    executing = fleet.to_dict()["agents"]["subagent-01"]
    _assert(executing["state"] == AgentState.EXECUTING.value, f"agent should be executing: {executing}")
    _assert(executing["result"] is None, f"executing task leaked stale result: {executing}")
    _assert(executing["error"] is None, f"executing task leaked stale error: {executing}")
    executing_status = subagent_fleet_status_snapshot(lambda: fleet)
    executing_row = executing_status["agents"][0]
    _assert(executing_row["has_result"] is False, f"public status exposed stale executing result: {executing_status}")
    _assert(executing_row["has_error"] is False, f"public status exposed stale executing error: {executing_status}")

    fleet.mark_failed("subagent-01", "planned retry failure")
    failed = fleet.to_dict()["agents"]["subagent-01"]
    _assert(failed["state"] == AgentState.FAILED.value, f"agent should be failed: {failed}")
    _assert(failed["result"] is None, f"failed task leaked stale result: {failed}")
    _assert_suppressed(failed["error"], "str", f"failed task public error should be suppressed: {failed}")
    _assert(
        "planned retry failure" in (fleet.agents["subagent-01"].error or ""),
        f"failed task lost internal error: {fleet.agents['subagent-01'].error}",
    )
    failed_status = subagent_fleet_status_snapshot(lambda: fleet)
    failed_row = failed_status["agents"][0]
    _assert(failed_row["has_result"] is False, f"public status exposed stale failed result: {failed_status}")
    _assert(failed_row["has_error"] is True, f"public status lost failed error presence: {failed_status}")


def _exercise_mark_ready_clears_stale_task_state() -> None:
    fleet = spawn_subagent_fleet(1)
    fleet.mark_executing("subagent-01", "completed-task", "old successful instructions")
    fleet.store_result("subagent-01", {"old": "success"})
    completed = fleet.to_dict()["agents"]["subagent-01"]
    _assert(completed["state"] == AgentState.READY.value, f"completed agent should be ready: {completed}")
    _assert(completed["task_id"] == "completed-task", f"success should retain its task id: {completed}")
    _assert(fleet.agents["subagent-01"].result == {"old": "success"}, "success result missing before reset")
    _assert_suppressed(completed["result"], "dict", f"success result should be suppressed before reset: {completed}")

    fleet.mark_executing("subagent-01", "failed-task", "old failed instructions")
    fleet.mark_failed("subagent-01", "old failure")
    failed_status = subagent_fleet_status_snapshot(lambda: fleet)
    failed_row = failed_status["agents"][0]
    _assert(failed_row["state"] == AgentState.FAILED.value, f"precondition failed state missing: {failed_status}")
    _assert(failed_row["task_id"] == "failed-task", f"precondition failed task missing: {failed_status}")
    _assert(failed_row["has_error"] is True, f"precondition failed error missing: {failed_status}")

    fleet.mark_ready("subagent-01")
    ready = fleet.to_dict()["agents"]["subagent-01"]
    _assert(ready["state"] == AgentState.READY.value, f"mark_ready should set ready state: {ready}")
    _assert(ready["task_id"] == "subagent-01", f"mark_ready leaked stale task id: {ready}")
    _assert(ready["result"] is None, f"mark_ready leaked stale result: {ready}")
    _assert(ready["error"] is None, f"mark_ready leaked stale error: {ready}")
    _assert(fleet.agents["subagent-01"].instructions == "", "mark_ready should clear stale instructions")
    _assert(fleet.get_ready_count() == (1, 1), f"ready count wrong after mark_ready: {fleet.to_dict()}")
    ready_status = subagent_fleet_status_snapshot(lambda: fleet)
    ready_row = ready_status["agents"][0]
    _assert(ready_status["state_counts"] == {"ready": 1}, f"ready state count wrong: {ready_status}")
    _assert(ready_row["state"] == AgentState.READY.value, f"public status should show ready: {ready_status}")
    _assert(ready_row["task_id"] == "subagent-01", f"public status leaked stale task: {ready_status}")
    _assert(ready_row["has_result"] is False, f"public status exposed stale ready result: {ready_status}")
    _assert(ready_row["has_error"] is False, f"public status exposed stale ready error: {ready_status}")


def _exercise_status_snapshot_exposure() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-status-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        fleet = spawn_subagent_fleet(2)
        fleet.mark_executing("subagent-01", "/\x55sers/example/private-task.txt", "raw instructions hidden")
        fleet.mark_failed("subagent-02", "error: /\x55sers/example/secret-token.txt should not leak")
        runtime.subagent_fleet = fleet

        snapshot = asdict(build_status_snapshot(runtime))
        fleet_status = snapshot.get("subagent_fleet") or {}
        canonical = subagent_fleet_status_snapshot(lambda: fleet)
        _assert(fleet_status == canonical, f"dashboard fleet status drifted from tool snapshot: {fleet_status} != {canonical}")
        _assert(snapshot.get("ready_agents") == fleet_status.get("ready_agents"), "top-level ready count drifted")
        _assert(snapshot.get("total_agents") == fleet_status.get("total_agents"), "top-level total count drifted")
        _assert(fleet_status.get("ready_agents") == 0, f"executing/failed agents should not be ready: {fleet_status}")
        _assert(fleet_status.get("total_agents") == 2, f"status snapshot lost total agents: {fleet_status}")
        _assert(fleet_status.get("agent_count") == 2, f"status snapshot lost agent rows: {fleet_status}")
        _assert(fleet_status.get("state_counts") == {"executing": 1, "failed": 1}, f"state counts wrong: {fleet_status}")
        _assert(fleet_status.get("results_suppressed") is True, f"raw results must stay suppressed: {fleet_status}")
        _assert(
            fleet_status.get("memory_context_suppressed") is True,
            f"shared memory context must stay suppressed: {fleet_status}",
        )
        status_text = str(fleet_status)
        for forbidden in ["/\x55sers/", "operator", "secret-token", "raw instructions"]:
            _assert(forbidden not in status_text, f"status snapshot leaked {forbidden!r}: {fleet_status}")
        _assert("<local-path>" in status_text, f"status snapshot should keep redacted path marker: {fleet_status}")


def _exercise_runtime_tool_and_status_command() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-tool-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        fleet = spawn_subagent_fleet(2)
        fleet.mark_executing("subagent-01", "/\x55sers/example/private-task.txt", "raw instructions hidden")
        fleet.mark_failed("subagent-02", "error: /\x55sers/example/secret-token.txt should not leak")
        runtime.subagent_fleet = fleet

        tool = runtime.registry.get("subagent_fleet_status")
        _assert(tool.risk.name == "READ_ONLY", "subagent fleet status must stay read-only")
        result = tool.handler({})
        _assert(result.ok is True, f"subagent fleet status should succeed with a runtime fleet: {result}")
        metadata = result.metadata
        _assert(metadata.get("ready_agents") == 0, f"runtime tool ready count wrong: {metadata}")
        _assert(metadata.get("total_agents") == 2, f"runtime tool total count wrong: {metadata}")
        _assert(metadata.get("state_counts") == {"executing": 1, "failed": 1}, f"runtime state counts wrong: {metadata}")
        for flag in (
            "calls_model",
            "executes_tools",
            "queues_approval",
            "requires_approval",
            "external_side_effect",
            "reads_personal_data",
            "authorizes_execution",
            "authorizes_completion_claim",
            "approval_granted",
        ):
            _assert(metadata.get(flag) is False, f"subagent fleet status boundary flag {flag} must be false")
        blob = result.output + str(metadata)
        for forbidden in ["/\x55sers/", "operator", "secret-token", "raw instructions"]:
            _assert(forbidden not in blob, f"runtime tool leaked {forbidden!r}: {blob}")
        _assert("<local-path>" in blob, f"runtime tool should retain redacted path marker: {blob}")

        routed = runtime.handle("jarvis status")
        _assert(routed.verified is True, f"jarvis status command should verify: {routed}")
        _assert("- internal worker fleet: 0 / 2 ready" in routed.response, routed.response)
        _assert("secret-token" not in routed.response and "raw instructions" not in routed.response, routed.response)


def _exercise_malformed_fleet_status_fails_closed() -> None:
    snapshot = subagent_fleet_status_snapshot(lambda: _MalformedFleet())
    _assert(snapshot.get("ok") is False, f"malformed fleet should fail closed: {snapshot}")
    _assert(snapshot.get("ready_agents") == 0, f"malformed ready count should coerce to zero: {snapshot}")
    _assert(snapshot.get("total_agents") == 0, f"malformed total count should coerce to zero: {snapshot}")
    formatted = format_subagent_fleet_status(snapshot)
    blob = formatted + str(snapshot)
    _assert("Internal worker fleet status: unavailable" in formatted, formatted)
    for forbidden in ["/\x55sers/", "/private/", "/var/folders", "operator", "secret-task", "secret-error"]:
        _assert(forbidden not in blob, f"tool snapshot leaked {forbidden!r}: {blob}")
    _assert(
        snapshot.get("diagnostic") == "subagent_fleet_readiness_inconsistent",
        f"malformed fleet should expose only a bounded diagnostic: {snapshot}",
    )

    with TemporaryDirectory(prefix="jarvis-subagent-malformed-status-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        runtime.subagent_fleet = _MalformedFleet()
        status = asdict(build_status_snapshot(runtime))
        fleet_status = status.get("subagent_fleet") or {}
        _assert(fleet_status == snapshot, f"dashboard malformed status drifted from tool snapshot: {fleet_status} != {snapshot}")
        _assert(status.get("ready_agents") == 0, f"top-level ready count should fail closed: {status}")
        _assert(status.get("total_agents") == 0, f"top-level total count should fail closed: {status}")
        _assert(fleet_status.get("ok") is False, f"status snapshot should fail closed: {fleet_status}")
        status_blob = str(fleet_status)
        for forbidden in ["/\x55sers/", "/private/", "/var/folders", "operator", "secret-task", "secret-error"]:
            _assert(forbidden not in status_blob, f"status snapshot leaked {forbidden!r}: {fleet_status}")
        _assert(
            fleet_status.get("diagnostic") == "subagent_fleet_readiness_inconsistent",
            f"dashboard should retain the bounded failure reason: {fleet_status}",
        )


def _exercise_spoofed_count_metadata_fails_closed() -> None:
    snapshot = subagent_fleet_status_snapshot(lambda: _SpoofedCountFleet())
    _assert(snapshot.get("ok") is False, f"spoofed count fleet should fail closed: {snapshot}")
    _assert(snapshot.get("ready_agents") == 0, f"boolean ready count should fail closed: {snapshot}")
    _assert(snapshot.get("total_agents") == 0, f"negative total count should fail closed: {snapshot}")
    formatted = format_subagent_fleet_status(snapshot)
    _assert("Internal worker fleet status: unavailable" in formatted, formatted)

    direct = format_subagent_fleet_status(
        {
            "ok": True,
            "ready_agents": True,
            "total_agents": -5,
            "state_counts": {"ready": True, "failed": -1},
            "agents": [],
        }
    )
    _assert("ready workers: 0 / 0" in direct, direct)
    _assert("ready=1" not in direct, direct)
    _assert("failed=-1" not in direct, direct)


def _exercise_hostile_agent_values_do_not_break_status() -> None:
    snapshot = subagent_fleet_status_snapshot(lambda: _HostileFleet())
    _assert(snapshot.get("ok") is False, f"hostile contradictory readiness should fail closed: {snapshot}")
    _assert(snapshot.get("ready_agents") == 0, f"hostile ready count must not survive: {snapshot}")
    _assert(snapshot.get("total_agents") == 0, f"hostile total count must not survive: {snapshot}")
    formatted = format_subagent_fleet_status(snapshot)
    blob = formatted + str(snapshot)
    _assert("Internal worker fleet status: unavailable" in formatted, formatted)
    for forbidden in ["/\x55sers/", "operator", "hidden-subagent-value"]:
        _assert(forbidden not in blob, f"hostile status leaked {forbidden!r}: {blob}")


def _exercise_direct_formatter_hostile_snapshot_fails_closed() -> None:
    formatted = format_subagent_fleet_status(
        {
            "ok": True,
            "ready_agents": "bad-ready",
            "total_agents": object(),
            "state_counts": {
                _PathLikeValue(): "3",
                "ready": "2",
                "zero": "0",
                "negative": "-5",
                "bad": "not-a-count",
            },
            "agents": [
                "skip malformed row",
                {
                    "id": "/\x55sers/example/subagent-one",
                    "state": "/private/tmp/running",
                    "task_id": "/var/folders/raw-task",
                    "has_result": True,
                    "has_error": False,
                },
                {
                    "id": "subagent-two",
                    "state": "failed",
                    "task_id": "/\x55sers/example/error-task",
                    "has_result": True,
                    "has_error": True,
                },
            ],
        }
    )
    _assert("ready workers: 0 / 0" in formatted, formatted)
    _assert("<local-path>=3" in formatted, formatted)
    _assert("ready=2" in formatted, formatted)
    _assert("zero=0" not in formatted, formatted)
    _assert("negative=-5" not in formatted, formatted)
    _assert("bad=0" not in formatted, formatted)
    _assert("result hidden" in formatted, formatted)
    _assert("error hidden" in formatted, formatted)
    for forbidden in ["/\x55sers/", "/private/", "/var/folders", "operator", "raw-task", "error-task"]:
        _assert(forbidden not in formatted, f"direct formatter leaked {forbidden!r}: {formatted}")

    unavailable = format_subagent_fleet_status(["not", "a", "snapshot"])
    _assert("unavailable (list)" in unavailable, unavailable)
    for expected in ["setup check", "jarvis status", "subagent fleet status", "does not authorize restart"]:
        _assert(expected in unavailable, f"direct unavailable formatter missed {expected!r}: {unavailable}")


def _exercise_unavailable_fleet_recovery_contract() -> None:
    missing = subagent_fleet_status_snapshot(None)

    def raising_fleet():
        raise RuntimeError("fleet snapshot failed near /\x55sers/example/private/subagent-state")

    crashed = subagent_fleet_status_snapshot(raising_fleet)
    for snapshot, diagnostic, label in [
        (missing, "subagent_fleet_unavailable", "missing fleet"),
        (crashed, "RuntimeError", "crashed fleet"),
    ]:
        _assert(snapshot.get("ok") is False, f"{label} should fail closed: {snapshot}")
        _assert(snapshot.get("diagnostic") == diagnostic, f"{label} diagnostic drifted: {snapshot}")
        _assert(snapshot.get("next_command") == "setup check", f"{label} missed first command: {snapshot}")
        _assert(
            snapshot.get("recovery_commands") == ["setup check", "jarvis status", "subagent fleet status"],
            f"{label} missed ordered recovery commands: {snapshot}",
        )
        _assert(
            snapshot.get("restart_required_if_still_unavailable") is True,
            f"{label} should name conditional restart: {snapshot}",
        )
        for key in [
            "authorizes_restart",
            "authorizes_worker_dispatch",
            "authorizes_execution",
            "authorizes_completion_claim",
        ]:
            _assert(snapshot.get(key) is False, f"{label} must keep {key}=False: {snapshot}")
        formatted = format_subagent_fleet_status(snapshot)
        for expected in ["setup check", "jarvis status", "subagent fleet status", "normal launcher"]:
            _assert(expected in formatted, f"{label} formatter missed {expected!r}: {formatted}")
        for forbidden in ["/\x55sers/", "operator", "subagent-state"]:
            _assert(forbidden not in str(snapshot) + formatted, f"{label} leaked {forbidden!r}")

    tool_result = make_subagent_fleet_status_tool(None)({})
    _assert(tool_result.ok is False, f"missing fleet tool should fail closed: {tool_result}")
    _assert(tool_result.metadata.get("recovery_commands") == missing["recovery_commands"], tool_result.metadata)
    for key in [
        "calls_model",
        "executes_tools",
        "queues_approval",
        "requires_approval",
        "external_side_effect",
        "reads_personal_data",
        "authorizes_restart",
        "authorizes_worker_dispatch",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        _assert(tool_result.metadata.get(key) is False, f"missing fleet tool should keep {key}=False")

    with TemporaryDirectory(prefix="jarvis-subagent-unavailable-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        runtime.subagent_fleet = None
        dashboard = asdict(build_status_snapshot(runtime))
        dashboard_fleet = dashboard.get("subagent_fleet") or {}
        _assert(dashboard_fleet.get("next_command") == "setup check", dashboard_fleet)
        _assert(dashboard_fleet.get("recovery_commands") == missing["recovery_commands"], dashboard_fleet)
        status_result = runtime.handle("jarvis status")
        _assert(status_result.verified is True, f"jarvis status should remain available: {status_result}")
        for expected in ["internal worker fleet: unavailable", "setup check", "jarvis status", "subagent fleet status"]:
            _assert(expected in status_result.response, f"jarvis status missed fleet recovery {expected!r}: {status_result.response}")


def _exercise_direct_formatter_hostile_truthiness_fails_closed() -> None:
    formatted = format_subagent_fleet_status(
        {
            "ok": True,
            "ready_agents": 1,
            "total_agents": 1,
            "state_counts": {"ready": 1},
            "agents": [
                {
                    "id": "subagent-hostile-bool",
                    "state": "ready",
                    "task_id": _HostileTruthiness(),
                    "has_result": _HostileTruthiness(),
                    "has_error": _HostileTruthiness(),
                }
            ],
        }
    )
    _assert("Internal worker fleet status:" in formatted, formatted)
    _assert("subagent-hostile-bool: ready" in formatted, formatted)
    _assert("task: <local-path>" in formatted, formatted)
    _assert("result hidden" not in formatted, formatted)
    _assert("error hidden" not in formatted, formatted)
    for forbidden in ["/\x55sers/", "operator", "hostile-subagent-bool"]:
        _assert(forbidden not in formatted, f"hostile truthiness formatter leaked {forbidden!r}: {formatted}")


def _exercise_shared_memory_snapshots_are_read_only() -> None:
    fleet = spawn_subagent_fleet(1)
    memory = fleet.get_shared_memory("subagent-01")
    _assert(memory is not None, "spawned fleet should expose subagent memory")
    memory.update("mission", "trustworthy control plane")

    direct_snapshot = memory.to_dict()
    direct_snapshot["context"]["mission"] = "mutated through direct snapshot"
    direct_snapshot["context"]["new_key"] = "unexpected write"
    _assert(
        memory.get("mission") == "trustworthy control plane",
        f"direct memory snapshot mutated internal state: {memory.to_dict()}",
    )
    _assert(memory.get("new_key") is None, f"direct memory snapshot injected a key: {memory.to_dict()}")

    fleet_snapshot = fleet.to_dict()
    fleet_snapshot["memories"]["subagent-01"]["context"]["mission"] = "mutated through fleet snapshot"
    fleet_snapshot["memories"]["subagent-01"]["context"]["fleet_key"] = "unexpected write"
    _assert(
        memory.get("mission") == "trustworthy control plane",
        f"fleet snapshot mutated internal state: {memory.to_dict()}",
    )
    _assert(memory.get("fleet_key") is None, f"fleet snapshot injected a key: {memory.to_dict()}")


def _exercise_shared_memory_nested_values_are_detached() -> None:
    fleet = spawn_subagent_fleet(1)
    memory = fleet.get_shared_memory("subagent-01")
    _assert(memory is not None, "spawned fleet should expose subagent memory")

    payload = {"steps": ["inspect"], "nested": {"ok": True}}
    memory.update("payload", payload)
    payload["steps"].append("mutated caller payload")
    payload["nested"]["ok"] = False
    _assert(
        memory.get("payload") == {"steps": ["inspect"], "nested": {"ok": True}},
        f"caller-owned update payload mutated shared memory: {memory.to_dict()}",
    )

    fetched = memory.get("payload")
    fetched["steps"].append("mutated fetched payload")
    fetched["nested"]["ok"] = False
    _assert(
        memory.get("payload") == {"steps": ["inspect"], "nested": {"ok": True}},
        f"mutable get() result mutated shared memory: {memory.to_dict()}",
    )

    result = {"items": ["done"], "nested": {"score": 1}}
    fleet.store_result("subagent-01", result)
    result["items"].append("mutated caller result")
    result["nested"]["score"] = 0
    snapshot = fleet.to_dict()
    _assert(
        fleet.agents["subagent-01"].result == {"items": ["done"], "nested": {"score": 1}},
        f"caller-owned result mutated agent state: {fleet.agents['subagent-01'].result}",
    )
    _assert(
        memory.get("last_result") == {"items": ["done"], "nested": {"score": 1}},
        f"caller-owned result mutated shared memory: {memory.get('last_result')}",
    )
    _assert_suppressed(
        snapshot["agents"]["subagent-01"]["result"],
        "dict",
        f"agent result should be suppressed in snapshots: {snapshot}",
    )
    _assert_suppressed(
        snapshot["memories"]["subagent-01"]["context"]["last_result"],
        "dict",
        f"memory result should be suppressed in snapshots: {snapshot}",
    )

    snapshot["agents"]["subagent-01"]["result"]["type"] = "mutated agent snapshot"
    snapshot["memories"]["subagent-01"]["context"]["last_result"]["type"] = "mutated memory snapshot"
    fresh_snapshot = fleet.to_dict()
    _assert(
        fleet.agents["subagent-01"].result == {"items": ["done"], "nested": {"score": 1}},
        f"agent result snapshot mutated internal state: {fresh_snapshot}",
    )
    _assert(
        memory.get("last_result") == {"items": ["done"], "nested": {"score": 1}},
        f"memory result snapshot mutated internal state: {fresh_snapshot}",
    )
    _assert_suppressed(
        fresh_snapshot["agents"]["subagent-01"]["result"],
        "dict",
        f"fresh agent result should stay suppressed: {fresh_snapshot}",
    )
    _assert_suppressed(
        fresh_snapshot["memories"]["subagent-01"]["context"]["last_result"],
        "dict",
        f"fresh memory result should stay suppressed: {fresh_snapshot}",
    )


def _exercise_public_snapshots_suppress_results_and_memory_context() -> None:
    fleet = spawn_subagent_fleet(1)
    memory = fleet.get_shared_memory("subagent-01")
    _assert(memory is not None, "spawned fleet should expose subagent memory")

    sensitive_result = {
        "path": "/\x55sers/example/private/가상연락처이-token.txt",
        "handle": "raw-handle leaked",
    }
    fleet.store_result("subagent-01", sensitive_result)
    memory.update(
        "/\x55sers/example/private-memory-key",
        "memory payload mentions /\x55sers/example/private/가상연락처이-token.txt and raw-handle leaked",
    )

    snapshot = fleet.to_dict()
    _assert(
        memory.get("last_result") == sensitive_result,
        f"internal coordination memory should keep full result: {memory.get('last_result')}",
    )
    _assert(
        fleet.agents["subagent-01"].result == sensitive_result,
        f"internal agent result should keep full result: {fleet.agents['subagent-01'].result}",
    )
    _assert_suppressed(snapshot["agents"]["subagent-01"]["result"], "dict", f"agent result leaked: {snapshot}")
    _assert_suppressed(
        snapshot["memories"]["subagent-01"]["context"]["last_result"],
        "dict",
        f"last_result leaked: {snapshot}",
    )
    _assert(
        "<local-path>" in snapshot["memories"]["subagent-01"]["context"],
        f"path-shaped memory key should be redacted, not raw: {snapshot}",
    )
    _assert_suppressed(
        snapshot["memories"]["subagent-01"]["context"]["<local-path>"],
        "str",
        f"path-shaped memory value should be suppressed: {snapshot}",
    )
    status = subagent_fleet_status_snapshot(lambda: fleet)
    status_row = status["agents"][0]
    _assert(status_row["has_result"] is True, f"status should preserve result presence: {status}")
    blob = str(snapshot) + str(status) + format_subagent_fleet_status(status)
    for forbidden in ["/\x55sers/", "operator", "가상연락처이-token", "raw-handle leaked", "private-memory-key"]:
        _assert(forbidden not in blob, f"public subagent snapshot leaked {forbidden!r}: {blob}")


def _exercise_public_snapshots_sanitize_agent_and_task_labels() -> None:
    fleet = SubagentFleet()
    raw_agent_id = "/\x55sers/example/private/subagent-가상연락처이"
    raw_task_id = "/private/tmp/가상연락처이-task.txt"
    fleet.register_agent(raw_agent_id, 1)
    fleet.mark_executing(raw_agent_id, raw_task_id, "raw instructions hidden")
    memory = fleet.get_shared_memory(raw_agent_id)
    _assert(memory is not None, "path-shaped internal agent id should still address memory internally")
    memory.update("/\x55sers/example/private-context-key", "private context payload")

    snapshot = fleet.to_dict()
    _assert("<local-path>" in snapshot["agents"], f"path-shaped agent key should be redacted: {snapshot}")
    _assert("<local-path>" in snapshot["memories"], f"path-shaped memory key should be redacted: {snapshot}")
    agent = snapshot["agents"]["<local-path>"]
    public_memory = snapshot["memories"]["<local-path>"]
    _assert(agent["id"] == "<local-path>", f"path-shaped agent id leaked: {agent}")
    _assert(agent["task_id"] == "<local-path>", f"path-shaped task id leaked: {agent}")
    _assert(public_memory["agent_id"] == "<local-path>", f"path-shaped memory agent id leaked: {public_memory}")
    _assert(
        "<local-path>" in public_memory["context"],
        f"path-shaped memory context key should be redacted: {public_memory}",
    )
    _assert(memory.get("/\x55sers/example/private-context-key") == "private context payload", "internal memory key changed")

    status = subagent_fleet_status_snapshot(lambda: fleet)
    formatted = format_subagent_fleet_status(status)
    status_row = status["agents"][0]
    _assert(status_row["id"] == "<local-path>", f"status agent id should be redacted: {status}")
    _assert(status_row["task_id"] == "<local-path>", f"status task id should be redacted: {status}")
    blob = str(snapshot) + str(status) + formatted
    for forbidden in [
        "/\x55sers/",
        "/private/",
        "operator",
        "subagent-가상연락처이",
        "가상연락처이-task",
        "private-context-key",
        "raw instructions",
    ]:
        _assert(forbidden not in blob, f"public subagent labels leaked {forbidden!r}: {blob}")
    _assert("<local-path>" in blob, f"redacted labels should keep a bounded marker: {blob}")


def _exercise_shared_memory_snapshotting_does_not_hold_memory_lock() -> None:
    fleet = spawn_subagent_fleet(1)
    memory = fleet.get_shared_memory("subagent-01")
    _assert(memory is not None, "spawned fleet should expose subagent memory")
    memory.update("ready", "yes")

    entered = threading.Event()
    release = threading.Event()
    blocked_update = threading.Thread(
        target=memory.update,
        args=("blocking_update", _BlockingSnapshotValue(entered, release)),
        daemon=True,
    )
    blocked_update.start()
    _assert(entered.wait(timeout=1), "memory update did not enter blocking snapshot")

    read_done = threading.Event()
    read_values: list[str] = []

    def read_ready_value() -> None:
        read_values.append(memory.get("ready"))
        read_done.set()

    reader = threading.Thread(target=read_ready_value, daemon=True)
    reader.start()
    if not read_done.wait(timeout=0.25):
        release.set()
        raise SystemExit("memory.update held the memory lock while snapshotting input")
    _assert(read_values == ["yes"], f"memory read changed during blocked update: {read_values}")
    release.set()
    blocked_update.join(timeout=1)
    reader.join(timeout=1)
    _assert(not blocked_update.is_alive(), "blocked memory update did not finish after release")

    entered = threading.Event()
    release = threading.Event()
    with memory.lock:
        memory.context["blocking_get"] = _BlockingSnapshotValue(entered, release)

    fetched: list[dict] = []
    blocked_get = threading.Thread(target=lambda: fetched.append(memory.get("blocking_get")), daemon=True)
    blocked_get.start()
    _assert(entered.wait(timeout=1), "memory get did not enter blocking snapshot")

    write_done = threading.Event()

    def write_probe() -> None:
        memory.update("probe_after_get", "ok")
        write_done.set()

    writer = threading.Thread(target=write_probe, daemon=True)
    writer.start()
    if not write_done.wait(timeout=0.25):
        release.set()
        raise SystemExit("memory.get held the memory lock while snapshotting output")
    release.set()
    blocked_get.join(timeout=1)
    writer.join(timeout=1)
    _assert(not blocked_get.is_alive(), "blocked memory get did not finish after release")
    _assert(fetched[0]["released"] is True, fetched)
    _assert(memory.get("probe_after_get") == "ok", f"memory write during blocked get was lost: {memory.to_dict()}")

    entered = threading.Event()
    release = threading.Event()
    with memory.lock:
        memory.context["blocking_dict"] = _BlockingSnapshotValue(entered, release)

    snapshots: list[dict] = []
    dict_done = threading.Event()

    def capture_public_snapshot() -> None:
        snapshots.append(memory.to_dict())
        dict_done.set()

    snapshot_thread = threading.Thread(target=capture_public_snapshot, daemon=True)
    snapshot_thread.start()
    if not dict_done.wait(timeout=0.25):
        release.set()
        raise SystemExit("memory.to_dict blocked while exporting suppressed context")
    _assert(not entered.is_set(), "memory.to_dict should not deepcopy public-suppressed values")
    release.set()
    snapshot_thread.join(timeout=1)
    _assert(not snapshot_thread.is_alive(), "memory to_dict snapshot thread did not finish")
    _assert_suppressed(snapshots[0]["context"]["blocking_dict"], "_BlockingSnapshotValue", snapshots)

    memory.update("probe_after_dict", "ok")
    _assert(memory.get("probe_after_dict") == "ok", f"memory write during blocked to_dict was lost: {memory.to_dict()}")


def _exercise_memory_snapshotting_does_not_hold_fleet_lock() -> None:
    fleet = spawn_subagent_fleet(2)
    entered = threading.Event()
    release = threading.Event()
    broadcast = threading.Thread(
        target=fleet.broadcast_to_all,
        args=("blocking", _BlockingSnapshotValue(entered, release)),
        daemon=True,
    )
    broadcast.start()
    _assert(entered.wait(timeout=1), "broadcast did not enter blocking memory snapshot")

    ready_done = threading.Event()
    ready_values: list[tuple[int, int]] = []

    def read_ready_count() -> None:
        ready_values.append(fleet.get_ready_count())
        ready_done.set()

    reader = threading.Thread(target=read_ready_count, daemon=True)
    reader.start()
    if not ready_done.wait(timeout=0.25):
        release.set()
        raise SystemExit("broadcast held the fleet lock while snapshotting shared memory")
    _assert(ready_values == [(2, 2)], f"ready count changed during blocked broadcast: {ready_values}")
    release.set()
    broadcast.join(timeout=1)
    reader.join(timeout=1)
    _assert(not broadcast.is_alive(), "blocked broadcast thread did not finish after release")

    memory = fleet.get_shared_memory("subagent-01")
    _assert(memory is not None, "spawned fleet should expose subagent memory")
    entered = threading.Event()
    release = threading.Event()
    with memory.lock:
        memory.context["blocking"] = _BlockingSnapshotValue(entered, release)

    snapshots: list[dict] = []
    snapshot_done = threading.Event()

    def capture_fleet_snapshot() -> None:
        snapshots.append(fleet.to_dict())
        snapshot_done.set()

    snapshot_thread = threading.Thread(target=capture_fleet_snapshot, daemon=True)
    snapshot_thread.start()
    if not snapshot_done.wait(timeout=0.25):
        release.set()
        raise SystemExit("fleet.to_dict blocked while exporting suppressed shared memory")
    _assert(not entered.is_set(), "fleet.to_dict should not deepcopy public-suppressed memory values")
    release.set()
    snapshot_thread.join(timeout=1)
    _assert(not snapshot_thread.is_alive(), "fleet snapshot thread did not finish")
    _assert_suppressed(
        snapshots[0]["memories"]["subagent-01"]["context"]["blocking"],
        "_BlockingSnapshotValue",
        snapshots,
    )

    ready_values = []
    ready_done = threading.Event()
    reader = threading.Thread(target=read_ready_count, daemon=True)
    reader.start()
    _assert(ready_done.wait(timeout=0.25), "ready count read did not complete after suppressed fleet snapshot")
    reader.join(timeout=1)
    _assert(ready_values == [(2, 2)], f"ready count changed after fleet snapshot: {ready_values}")


def main() -> None:
    asyncio.run(_exercise_parallel_success())
    asyncio.run(_exercise_successful_staged_memory_commit())
    asyncio.run(_exercise_failure_accounting())
    asyncio.run(_exercise_handler_error_diagnostics_are_bounded())
    asyncio.run(_exercise_handler_error_without_path_is_suppressed_publicly())
    asyncio.run(_exercise_duplicate_parallel_task_ids_fail_closed())
    asyncio.run(_exercise_outer_cancellation_cleans_up_and_reraises())
    asyncio.run(_exercise_concurrent_same_agent_claim_is_atomic())
    asyncio.run(_exercise_timeout_is_bounded_when_handler_suppresses_cancellation())
    asyncio.run(_exercise_timeout_diagnostics_fail_closed())
    asyncio.run(_exercise_invalid_task_ids_fail_closed())
    _exercise_task_transitions_clear_stale_result_state()
    _exercise_mark_ready_clears_stale_task_state()
    _exercise_status_snapshot_exposure()
    _exercise_runtime_tool_and_status_command()
    _exercise_malformed_fleet_status_fails_closed()
    _exercise_spoofed_count_metadata_fails_closed()
    _exercise_hostile_agent_values_do_not_break_status()
    _exercise_direct_formatter_hostile_snapshot_fails_closed()
    _exercise_unavailable_fleet_recovery_contract()
    _exercise_direct_formatter_hostile_truthiness_fails_closed()
    _exercise_shared_memory_snapshots_are_read_only()
    _exercise_shared_memory_nested_values_are_detached()
    _exercise_public_snapshots_suppress_results_and_memory_context()
    _exercise_public_snapshots_sanitize_agent_and_task_labels()
    _exercise_shared_memory_snapshotting_does_not_hold_memory_lock()
    _exercise_memory_snapshotting_does_not_hold_fleet_lock()
    print("Subagent fleet smoke passed")


if __name__ == "__main__":
    main()
