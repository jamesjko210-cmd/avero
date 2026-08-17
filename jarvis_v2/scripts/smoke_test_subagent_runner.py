from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import socket
import stat
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from typing import Any

from jarvis_v2.agent.subagent_runner import (
    MAX_ENVELOPE_BYTES,
    DurableSubagentRunner,
    HandlerRegistry,
    HandlerSpec,
    InMemoryPayloadCustodian,
    InMemoryResultSink,
    ResultRecord,
    SinkLookup,
    request_envelope,
    result_envelope,
)
import jarvis_v2.agent.subagent_runner as runner_module
from jarvis_v2.memory.store import MemoryStore, SubagentAssignmentTransition
import jarvis_v2.scripts.run_subagent_runner as runner_script
from jarvis_v2.tools.subagents import _durable_fleet_status_snapshot


def _store(root: str) -> MemoryStore:
    store = MemoryStore(Path(root) / "runner.sqlite")
    store.init()
    return store


def _task_row(store: MemoryStore, task_id: str) -> dict[str, Any]:
    with store.connect() as conn:
        row = conn.execute("SELECT * FROM subagent_tasks WHERE task_id = ?", (task_id,)).fetchone()
    return dict(row) if row else {}


def _assignment_rows(store: MemoryStore) -> list[dict[str, Any]]:
    with store.connect() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM subagent_assignments ORDER BY claimed_ms")]


async def _echo(payload: Any) -> Any:
    return {"echo": payload}


async def _ignore_cancellation_forever(_payload: Any) -> Any:
    while True:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            continue


async def _cpu_spin_forever(_payload: Any) -> Any:
    while True:
        pass


async def _must_not_run(_payload: Any) -> Any:
    raise RuntimeError("adopted work replayed unexpectedly")


async def _return_after_short_sleep(payload: Any) -> Any:
    await asyncio.sleep(0.14)
    return payload


async def _raise_private_handler_error(_payload: Any) -> Any:
    raise RuntimeError("RAW HANDLER EXCEPTION MUST NOT PERSIST")


async def _typed_payload(payload: Any) -> Any:
    return payload


def _registry(handler=_echo, *, timeout: float = 5.0) -> HandlerRegistry:
    return HandlerRegistry(
        [HandlerSpec("echo", "jarvis.echo", 1, "json", timeout, handler)]
    )


async def test_happy_path_and_privacy() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-happy-") as temp:
        store = _store(temp)
        custody = InMemoryPayloadCustodian()
        sink = InMemoryResultSink()
        runner = DurableSubagentRunner(
            store=store,
            custodian=custody,
            result_sink=sink,
            handlers=_registry(),
            worker_count=2,
            runtime_id="runner-happy",
        )
        secret = "RAW-SUBAGENT-PAYLOAD-MUST-NOT-ENTER-SQLITE"
        queued = await runner.submit(kind="echo", payload={"text": secret})
        if queued.status != "ENQUEUED":
            raise SystemExit(f"runner enqueue failed: {queued}")
        outcomes = await runner.run_once()
        if not any(item.status == "SUCCEEDED" and item.task_id == queued.task_id for item in outcomes):
            raise SystemExit(f"runner happy path did not succeed: {outcomes}")
        task = _task_row(store, queued.task_id)
        lookup = await sink.lookup(queued.task_id)
        if task.get("state") != "succeeded" or lookup.status != "found":
            raise SystemExit(f"runner did not bind durable success to sink result: {task} / {lookup}")
        database_bytes = store.db_path.read_bytes()
        if secret.encode("utf-8") in database_bytes:
            raise SystemExit("raw runner payload leaked into SQLite")
        snapshot = store.subagent_control_snapshot()
        if snapshot.get("payloads_persisted") or snapshot.get("results_persisted"):
            raise SystemExit(f"runner status claimed raw content persistence: {snapshot}")
        await runner.stop()


async def test_missing_unknown_and_tampered_payloads_reject_before_start() -> None:
    cases = ("missing", "unknown", "tampered")
    for case in cases:
        with TemporaryDirectory(prefix=f"jarvis-subagent-runner-{case}-") as temp:
            store = _store(temp)
            custody = InMemoryPayloadCustodian()
            runner = DurableSubagentRunner(
                store=store,
                custodian=custody,
                result_sink=InMemoryResultSink(),
                handlers=_registry(),
                worker_count=1,
                runtime_id=f"runner-{case}",
            )
            if case == "unknown":
                digest, envelope = request_envelope(
                    kind="unknown", handler_id="unknown.handler", handler_version=1, payload={"x": 1}
                )
                await custody.put_if_absent(digest, envelope)
                queued = store.enqueue_subagent_task(
                    request_digest=digest, kind="unknown", replay_policy="safe", max_attempts=2
                )
                expected = "handler_unknown"
            else:
                digest, envelope = request_envelope(
                    kind="echo", handler_id="jarvis.echo", handler_version=1, payload={"x": case}
                )
                if case == "tampered":
                    await custody.put_if_absent(digest, envelope)
                    custody._items[digest] = envelope + b" "  # deliberate private-custodian corruption
                    expected = "payload_digest_mismatch"
                else:
                    expected = "payload_missing"
                queued = store.enqueue_subagent_task(
                    request_digest=digest, kind="echo", replay_policy="safe", max_attempts=2
                )
            outcomes = await runner.run_once()
            if not any(item.error_code == expected and item.status == "REJECTED" for item in outcomes):
                raise SystemExit(f"{case} custody case did not reject before start: {outcomes}")
            rows = _assignment_rows(store)
            if len(rows) != 1 or rows[0]["started_ms"] is not None or rows[0]["state"] != "failed":
                raise SystemExit(f"{case} custody case crossed the start boundary: {rows}")
            await runner.stop()


async def test_existing_sink_result_is_adopted_without_handler_replay() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-adopt-") as temp:
        store = _store(temp)
        custody = InMemoryPayloadCustodian("authoritative_rehydrating")
        sink = InMemoryResultSink()

        runner = DurableSubagentRunner(
            store=store,
            custodian=custody,
            result_sink=sink,
            handlers=_registry(_must_not_run),
            worker_count=1,
            runtime_id="runner-adopt",
        )
        queued = await runner.submit(kind="echo", payload={"replay": "must not run"})
        task = _task_row(store, queued.task_id)
        result_digest, envelope = result_envelope(
            task_id=queued.task_id,
            request_digest=task["request_digest"],
            kind="echo",
            handler_id="jarvis.echo",
            handler_version=1,
            result_type="json",
            result={"already": "committed"},
        )
        commit = await sink.commit(
            ResultRecord(
                queued.task_id,
                task["request_digest"],
                "echo",
                "jarvis.echo",
                1,
                result_digest,
                "json",
                envelope,
            )
        )
        if commit != "committed":
            raise SystemExit(f"adoption fixture sink commit failed: {commit}")
        outcomes = await runner.run_once()
        rows = _assignment_rows(store)
        if not any(item.status == "ADOPTED" for item in outcomes):
            raise SystemExit(f"existing result was not adopted without replay: {outcomes}")
        if len(rows) != 1 or rows[0]["started_ms"] is not None or rows[0]["state"] != "succeeded":
            raise SystemExit(f"adopted result incorrectly claimed handler start: {rows}")
        await runner.stop()


async def test_lease_renewal_and_handler_failure() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-renew-") as temp:
        store = _store(temp)

        runner = DurableSubagentRunner(
            store=store,
            custodian=InMemoryPayloadCustodian(),
            result_sink=InMemoryResultSink(),
            handlers=_registry(_return_after_short_sleep),
            worker_count=1,
            lease_ms=100,
            renew_interval_ms=20,
            runtime_id="runner-renew",
        )
        queued = await runner.submit(kind="echo", payload={"slow": True})
        outcomes = await runner.run_once()
        if not any(item.status == "SUCCEEDED" for item in outcomes) or _task_row(store, queued.task_id)["state"] != "succeeded":
            raise SystemExit(f"runner did not renew a live pure handler: {outcomes}")
        await runner.stop()

    with TemporaryDirectory(prefix="jarvis-subagent-runner-error-") as temp:
        store = _store(temp)

        runner = DurableSubagentRunner(
            store=store,
            custodian=InMemoryPayloadCustodian(),
            result_sink=InMemoryResultSink(),
            handlers=_registry(_raise_private_handler_error),
            worker_count=1,
            runtime_id="runner-error",
        )
        queued = await runner.submit(kind="echo", payload={"fails": True})
        outcomes = await runner.run_once()
        task = _task_row(store, queued.task_id)
        if task.get("state") != "failed" or task.get("error_code") != "handler_error":
            raise SystemExit(f"handler failure was not bounded: {outcomes} / {task}")
        if b"RAW HANDLER EXCEPTION" in store.db_path.read_bytes():
            raise SystemExit("raw handler exception leaked into SQLite")
        await runner.stop()


class CollisionAfterLookupSink(InMemoryResultSink):
    async def lookup(self, task_id: str) -> SinkLookup:
        return SinkLookup("missing")

    async def stage(self, record: ResultRecord) -> str:
        return "collision"


class SlowStageSink(InMemoryResultSink):
    async def stage(self, record: ResultRecord) -> str:
        await asyncio.sleep(0.16)
        return await super().stage(record)


class SlowPublishSink(InMemoryResultSink):
    async def publish(self, task_id: str, result_digest: str) -> str:
        await asyncio.sleep(0.16)
        return await super().publish(task_id, result_digest)


async def test_sink_collision_quarantines_running_assignment() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-collision-") as temp:
        store = _store(temp)
        runner = DurableSubagentRunner(
            store=store,
            custodian=InMemoryPayloadCustodian(),
            result_sink=CollisionAfterLookupSink(),
            handlers=_registry(),
            worker_count=1,
            runtime_id="runner-collision",
        )
        queued = await runner.submit(kind="echo", payload={"collision": True})
        outcomes = await runner.run_once()
        task = _task_row(store, queued.task_id)
        if task.get("state") != "uncertain" or task.get("error_code") != "sink_collision":
            raise SystemExit(f"sink collision was not quarantined: {outcomes} / {task}")
        rows = _assignment_rows(store)
        if len(rows) != 1 or rows[0]["state"] != "uncertain" or rows[0]["started_ms"] is None:
            raise SystemExit(f"sink collision lost running uncertainty evidence: {rows}")
        await runner.stop()


async def test_ephemeral_restart_rejects_lost_payload() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-restart-") as temp:
        store = _store(temp)
        first = DurableSubagentRunner(
            store=store,
            custodian=InMemoryPayloadCustodian(),
            result_sink=InMemoryResultSink(),
            handlers=_registry(),
            worker_count=1,
            runtime_id="runner-before-restart",
        )
        queued = await first.submit(kind="echo", payload={"ephemeral": "lost"})
        replacement = DurableSubagentRunner(
            store=store,
            custodian=InMemoryPayloadCustodian(),
            result_sink=InMemoryResultSink(),
            handlers=_registry(),
            worker_count=1,
            runtime_id="runner-after-restart",
        )
        outcomes = await replacement.run_once()
        task = _task_row(store, queued.task_id)
        if task.get("state") != "failed" or task.get("error_code") != "payload_missing":
            raise SystemExit(f"ephemeral restart invented payload recovery: {outcomes} / {task}")
        await replacement.stop()


async def test_transient_custody_outage_requeues_then_succeeds() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-custody-outage-") as temp:
        store = _store(temp)
        custody = InMemoryPayloadCustodian("authoritative_rehydrating")
        runner = DurableSubagentRunner(
            store=store,
            custodian=custody,
            result_sink=InMemoryResultSink(),
            handlers=_registry(),
            worker_count=1,
            runtime_id="runner-custody-outage",
        )
        queued = await runner.submit(kind="echo", payload={"retry": True})
        custody.available = False
        first = await runner.run_once()
        if not any(item.status == "REQUEUED" and item.error_code == "custodian_unavailable" for item in first):
            raise SystemExit(f"transient custody outage did not release the unstarted claim: {first}")
        if _task_row(store, queued.task_id).get("state") != "queued":
            raise SystemExit("transient custody outage terminally failed queued work")
        custody.available = True
        second = await runner.run_once()
        if not any(item.status == "SUCCEEDED" for item in second):
            raise SystemExit(f"rehydrated work did not succeed after custody recovery: {second}")
        await runner.stop()


async def test_lost_finalize_response_uses_exact_terminal_readback() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-finalize-readback-") as temp:
        store = _store(temp)
        runner = DurableSubagentRunner(
            store=store,
            custodian=InMemoryPayloadCustodian(),
            result_sink=InMemoryResultSink(),
            handlers=_registry(),
            worker_count=1,
            runtime_id="runner-finalize-readback",
        )
        queued = await runner.submit(kind="echo", payload={"lost": "response"})
        original_finalize = store.finalize_subagent_assignment

        def commit_then_lose_response(*args: Any, **kwargs: Any) -> SubagentAssignmentTransition:
            committed = original_finalize(*args, **kwargs)
            if committed.status != "SUCCEEDED":
                return committed
            raise OSError("synthetic lost finalization response")

        store.finalize_subagent_assignment = commit_then_lose_response  # type: ignore[method-assign]
        outcomes = await runner.run_once()
        store.finalize_subagent_assignment = original_finalize  # type: ignore[method-assign]
        if not any(item.status == "ALREADY_SUCCEEDED" for item in outcomes):
            raise SystemExit(f"lost finalization response was not recovered by exact readback: {outcomes}")
        if _task_row(store, queued.task_id).get("state") != "succeeded":
            raise SystemExit("lost finalization response changed committed success truth")
        await runner.stop()


async def test_lost_failed_finalize_response_uses_exact_terminal_readback() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-failed-readback-") as temp:
        store = _store(temp)
        runner = DurableSubagentRunner(
            store=store,
            custodian=InMemoryPayloadCustodian(),
            result_sink=InMemoryResultSink(),
            handlers=_registry(_raise_private_handler_error),
            worker_count=1,
            runtime_id="runner-failed-readback",
        )
        queued = await runner.submit(kind="echo", payload={"lost": "failed response"})
        original_finalize = store.finalize_subagent_assignment

        def commit_failed_then_lose(*args: Any, **kwargs: Any):
            committed = original_finalize(*args, **kwargs)
            if committed.status == "FAILED":
                raise OSError("synthetic lost failed-finalization response")
            return committed

        store.finalize_subagent_assignment = commit_failed_then_lose  # type: ignore[method-assign]
        try:
            outcomes = await runner.run_once()
        finally:
            store.finalize_subagent_assignment = original_finalize  # type: ignore[method-assign]
        if not any(item.status == "ALREADY_FAILED" for item in outcomes):
            raise SystemExit(f"lost failed finalization response crashed or lied: {outcomes}")
        row = _task_row(store, queued.task_id)
        if row.get("state") != "failed" or row.get("error_code") != "handler_error":
            raise SystemExit(f"lost failed response changed durable truth: {row}")
        await runner.stop()


async def test_enqueue_commit_response_loss_preserves_custody() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-enqueue-readback-") as temp:
        store = _store(temp)
        custody = InMemoryPayloadCustodian()
        runner = DurableSubagentRunner(
            store=store,
            custodian=custody,
            result_sink=InMemoryResultSink(),
            handlers=_registry(),
            worker_count=1,
            runtime_id="runner-enqueue-readback",
        )
        original_enqueue = store.enqueue_subagent_task
        calls = 0

        def commit_then_lose(*args: Any, **kwargs: Any):
            nonlocal calls
            calls += 1
            queued = original_enqueue(*args, **kwargs)
            if calls <= 2:
                raise OSError("synthetic lost enqueue response")
            return queued

        store.enqueue_subagent_task = commit_then_lose  # type: ignore[method-assign]
        try:
            queued = await runner.submit(kind="echo", payload={"enqueue": "lost"})
        finally:
            store.enqueue_subagent_task = original_enqueue  # type: ignore[method-assign]
        if queued.status != "EXISTING" or queued.task_state != "queued":
            raise SystemExit(f"lost enqueue response did not recover exact intake: {queued}")
        task = store.get_subagent_task_ephemeral_status(queued.task_id)
        if (await custody.fetch(task.request_digest)).status != "found":
            raise SystemExit("ambiguous enqueue discarded custody for a committed task")
        outcomes = await runner.run_once()
        if not any(item.status == "SUCCEEDED" for item in outcomes):
            raise SystemExit(f"recovered enqueue could not execute: {outcomes}")
        await runner.stop()


async def test_publish_after_retention_deadline_expires_without_resurrection() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-publish-expiry-") as temp:
        store = _store(temp)
        sink = SlowPublishSink()
        runner = DurableSubagentRunner(
            store=store,
            custodian=InMemoryPayloadCustodian(),
            result_sink=sink,
            handlers=_registry(),
            worker_count=1,
            result_retention_ms=100,
            runtime_id="runner-publish-expiry",
        )
        queued = await runner.submit(kind="echo", payload={"publish": "late"})
        outcomes = await runner.run_once()
        if not any(item.status == "RESULT_EXPIRED" for item in outcomes):
            raise SystemExit(f"late publication resurrected an expired result: {outcomes}")
        status = store.get_subagent_task_ephemeral_status(queued.task_id)
        if status.result_state != "expired" or status.result_released_ms is None:
            raise SystemExit(f"late publication did not durably expire/release: {status}")
        if (await sink.lookup(queued.task_id)).status != "missing":
            raise SystemExit("late publication retained bytes after exact expiry")
        await runner.stop()


async def test_lease_loss_cancels_and_suppresses_late_result() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-lease-loss-") as temp:
        store = _store(temp)

        sink = InMemoryResultSink()
        runner = DurableSubagentRunner(
            store=store,
            custodian=InMemoryPayloadCustodian(),
            result_sink=sink,
            handlers=_registry(_return_after_short_sleep),
            worker_count=1,
            lease_ms=100,
            renew_interval_ms=10,
            runtime_id="runner-lease-loss",
        )
        queued = await runner.submit(kind="echo", payload={"late": "must not publish"})
        original_renew = store.renew_subagent_assignment
        store.renew_subagent_assignment = lambda *args, **kwargs: False  # type: ignore[method-assign]
        outcomes = await runner.run_once()
        store.renew_subagent_assignment = original_renew  # type: ignore[method-assign]
        if not any(item.status == "LEASE_LOST" for item in outcomes):
            raise SystemExit(f"runner did not stop after lease loss: {outcomes}")
        await asyncio.sleep(0.25)
        if (await sink.lookup(queued.task_id)).status != "missing":
            raise SystemExit("lease-lost handler published a late result")
        recovered = store.recover_expired_subagent_assignments(
            checked_at_ms=int(time.time() * 1000) + 1_000,
            allow_running_requeue=False,
        )
        if recovered["uncertain"] != 1 or _task_row(store, queued.task_id).get("state") != "uncertain":
            raise SystemExit(f"lease-lost execution did not recover as uncertain: {recovered}")
        await runner.stop()


async def test_parent_delay_cannot_accept_post_deadline_result() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-deadline-order-") as temp:
        store = _store(temp)
        sink = InMemoryResultSink()
        runner = DurableSubagentRunner(
            store=store,
            custodian=InMemoryPayloadCustodian(),
            result_sink=sink,
            handlers=_registry(_return_after_short_sleep, timeout=0.1),
            worker_count=1,
            lease_ms=500,
            renew_interval_ms=50,
            runtime_id="runner-deadline-order",
        )
        queued = await runner.submit(kind="echo", payload={"late": True})
        cycle = asyncio.create_task(runner.run_once())
        loop = asyncio.get_running_loop()
        start_deadline = loop.time() + 1.0
        while loop.time() < start_deadline and _task_row(store, queued.task_id).get("state") != "running":
            await asyncio.sleep(0.005)
        if _task_row(store, queued.task_id).get("state") != "running":
            raise SystemExit("deadline-order fixture never started")
        time.sleep(0.3)
        outcomes = await asyncio.wait_for(cycle, timeout=1.0)
        row = _task_row(store, queued.task_id)
        if row.get("state") != "failed" or row.get("error_code") != "handler_timeout":
            raise SystemExit(f"post-deadline child result was accepted: {outcomes} / {row}")
        if (await sink.lookup(queued.task_id)).status != "missing":
            raise SystemExit("post-deadline child result reached the published sink")
        await runner.stop()


async def test_lease_expiry_during_result_stage_never_publishes() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-stage-lease-") as temp:
        store = _store(temp)
        sink = SlowStageSink()
        runner = DurableSubagentRunner(
            store=store,
            custodian=InMemoryPayloadCustodian(),
            result_sink=sink,
            handlers=_registry(_echo, timeout=1.0),
            worker_count=1,
            lease_ms=100,
            renew_interval_ms=20,
            runtime_id="runner-stage-lease",
        )
        queued = await runner.submit(kind="echo", payload={"lease": "expires"})
        outcomes = await runner.run_once()
        if not any(item.status == "LEASE_LOST" for item in outcomes):
            raise SystemExit(f"expired result-stage lease was not rejected: {outcomes}")
        if (await sink.lookup(queued.task_id)).status != "missing":
            raise SystemExit("lease-lost staged result became externally visible")
        recovered = store.recover_expired_subagent_assignments(
            checked_at_ms=int(time.time() * 1000) + 1_000,
            allow_running_requeue=False,
        )
        if recovered["uncertain"] != 1 or _task_row(store, queued.task_id).get("state") != "uncertain":
            raise SystemExit(f"stage lease loss did not recover as uncertain: {recovered}")
        await runner.stop()


async def test_hard_timeout_reaps_hostile_handlers_and_preserves_fleet_progress() -> None:
    for label, hostile in (
        ("cancel-resistant", _ignore_cancellation_forever),
        ("cpu-bound", _cpu_spin_forever),
    ):
        with TemporaryDirectory(prefix=f"jarvis-subagent-runner-hard-timeout-{label}-") as temp:
            store = _store(temp)
            sink = InMemoryResultSink()
            before_pids = {child.pid for child in multiprocessing.active_children()}
            runner = DurableSubagentRunner(
                store=store,
                custodian=InMemoryPayloadCustodian(),
                result_sink=sink,
                handlers=HandlerRegistry(
                    [
                        HandlerSpec("hostile", f"jarvis.hostile.{label}", 1, "json", 1.0, hostile),
                        HandlerSpec("echo", "jarvis.echo", 1, "json", 2.0, _echo),
                    ]
                ),
                worker_count=2,
                lease_ms=5_000,
                renew_interval_ms=100,
                runtime_id=f"runner-hard-timeout-{label}",
            )
            hostile_task = await runner.submit(kind="hostile", payload={"case": label})
            fast_task = await runner.submit(kind="echo", payload={"fast": label})
            loop = asyncio.get_running_loop()
            started_at = loop.time()
            cycle = asyncio.create_task(runner.run_once())
            progress_deadline = loop.time() + 5.0
            fast_found = False
            hostile_state = ""
            while loop.time() < progress_deadline:
                fast_found = (await sink.lookup(fast_task.task_id)).status == "found"
                hostile_state = _task_row(store, hostile_task.task_id).get("state", "")
                if fast_found or hostile_state in {"succeeded", "failed", "uncertain"} or cycle.done():
                    break
                await asyncio.sleep(0.01)
            if not fast_found or hostile_state != "running" or cycle.done():
                raise SystemExit(
                    f"{label} handler blocked the independent worker slot: "
                    f"hostile={hostile_state} "
                    f"fast={_task_row(store, fast_task.task_id).get('state')} "
                    f"elapsed={loop.time() - started_at:.3f} cycle_done={cycle.done()}"
                )
            outcomes = await asyncio.wait_for(cycle, timeout=5.0)
            elapsed = loop.time() - started_at
            hostile_row = _task_row(store, hostile_task.task_id)
            if hostile_row.get("state") != "failed" or hostile_row.get("error_code") != "handler_timeout":
                raise SystemExit(f"{label} handler did not reach honest timeout truth: {outcomes} / {hostile_row}")
            if (await sink.lookup(hostile_task.task_id)).status != "missing":
                raise SystemExit(f"{label} handler published a result after timeout")
            if _task_row(store, fast_task.task_id).get("state") != "succeeded":
                raise SystemExit(f"{label} timeout did not preserve bounded fleet progress: {elapsed} / {outcomes}")
            await runner.stop()
            leaked = {
                child.pid
                for child in multiprocessing.active_children()
                if child.pid not in before_pids and child.is_alive()
            }
            if leaked:
                raise SystemExit(f"{label} handler child was not terminated and reaped: {sorted(leaked)}")


async def test_delayed_reap_retains_terminal_custody() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-delayed-reap-") as temp:
        store = _store(temp)
        sink = InMemoryResultSink()
        runner = DurableSubagentRunner(
            store=store,
            custodian=InMemoryPayloadCustodian(),
            result_sink=sink,
            handlers=_registry(_cpu_spin_forever, timeout=0.1),
            worker_count=1,
            lease_ms=500,
            renew_interval_ms=50,
            runtime_id="runner-delayed-reap",
        )
        queued = await runner.submit(kind="echo", payload={"reap": "delayed"})
        original_stop = runner_module._stop_handler_process

        async def delayed_stop(process, *, allow_graceful_exit=False):
            reaped = await original_stop(
                process, allow_graceful_exit=allow_graceful_exit
            )
            await asyncio.sleep(0.6)
            return reaped

        runner_module._stop_handler_process = delayed_stop  # type: ignore[assignment]
        try:
            outcomes = await asyncio.wait_for(runner.run_once(), timeout=1.5)
        finally:
            runner_module._stop_handler_process = original_stop  # type: ignore[assignment]
        row = _task_row(store, queued.task_id)
        if row.get("state") != "failed" or row.get("error_code") != "handler_timeout":
            raise SystemExit(f"delayed reap lost terminal custody: {outcomes} / {row}")
        if (await sink.lookup(queued.task_id)).status != "missing":
            raise SystemExit("delayed timeout published a result")
        await runner.stop()


async def test_cleanup_renewal_refusal_reaps_without_false_terminalization() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-reap-lease-loss-") as temp:
        store = _store(temp)
        sink = InMemoryResultSink()
        before_pids = {child.pid for child in multiprocessing.active_children()}
        runner = DurableSubagentRunner(
            store=store,
            custodian=InMemoryPayloadCustodian(),
            result_sink=sink,
            handlers=_registry(_cpu_spin_forever, timeout=0.1),
            worker_count=1,
            lease_ms=500,
            renew_interval_ms=50,
            runtime_id="runner-reap-lease-loss",
        )
        queued = await runner.submit(kind="echo", payload={"reap": "lease-lost"})
        original_renew = store.renew_subagent_assignment
        cleanup_renewals = 0

        def refuse_cleanup_renewal(*args: Any, **kwargs: Any) -> bool:
            nonlocal cleanup_renewals
            if kwargs.get("lease_ms") == runner_module.HANDLER_PROCESS_CLEANUP_LEASE_MS:
                cleanup_renewals += 1
                return False
            return original_renew(*args, **kwargs)

        store.renew_subagent_assignment = refuse_cleanup_renewal  # type: ignore[method-assign]
        try:
            outcomes = await runner.run_once()
        finally:
            store.renew_subagent_assignment = original_renew  # type: ignore[method-assign]
        row = _task_row(store, queued.task_id)
        if cleanup_renewals != 1 or not any(item.status == "LEASE_LOST" for item in outcomes):
            raise SystemExit(f"cleanup renewal refusal was not preserved: {outcomes}")
        if row.get("state") != "running" or row.get("error_code") is not None:
            raise SystemExit(f"lease-lost cleanup recorded false terminal truth: {row}")
        if (await sink.lookup(queued.task_id)).status != "missing":
            raise SystemExit("lease-lost cleanup published a result")
        leaked = {
            child.pid
            for child in multiprocessing.active_children()
            if child.pid not in before_pids and child.is_alive()
        }
        if leaked:
            raise SystemExit(f"lease-lost cleanup left handler children alive: {sorted(leaked)}")
        recovered = store.recover_expired_subagent_assignments(
            checked_at_ms=int(time.time() * 1000) + 2_000,
            allow_running_requeue=False,
        )
        if recovered["uncertain"] != 1 or _task_row(store, queued.task_id).get("state") != "uncertain":
            raise SystemExit(f"lease-lost cleanup did not recover as uncertain: {recovered}")
        await runner.stop()


async def test_unproven_reap_quarantines_before_result_publication() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-unreaped-") as temp:
        store = _store(temp)
        sink = InMemoryResultSink()
        runner = DurableSubagentRunner(
            store=store,
            custodian=InMemoryPayloadCustodian(),
            result_sink=sink,
            handlers=_registry(_echo, timeout=1.0),
            worker_count=1,
            runtime_id="runner-unreaped",
        )
        queued = await runner.submit(kind="echo", payload={"reap": "unproven"})
        original_stop = runner_module._stop_handler_process
        calls = 0

        async def fail_first_reap(process, *, allow_graceful_exit=False):
            nonlocal calls
            calls += 1
            if calls == 1:
                return False
            return await original_stop(
                process, allow_graceful_exit=allow_graceful_exit
            )

        runner_module._stop_handler_process = fail_first_reap  # type: ignore[assignment]
        try:
            outcomes = await runner.run_once()
        finally:
            runner_module._stop_handler_process = original_stop  # type: ignore[assignment]
        row = _task_row(store, queued.task_id)
        if (
            row.get("state") != "uncertain"
            or row.get("error_code") != "handler_process_unreaped"
        ):
            raise SystemExit(f"unproven reap finalized execution: {outcomes} / {row}")
        if (await sink.lookup(queued.task_id)).status != "missing":
            raise SystemExit("unproven reap published a result")
        await runner.stop()


async def test_direct_stop_reap_failure_quarantines_assignment() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-stop-unreaped-") as temp:
        store = _store(temp)
        sink = InMemoryResultSink()
        runner = DurableSubagentRunner(
            store=store,
            custodian=InMemoryPayloadCustodian(),
            result_sink=sink,
            handlers=_registry(_cpu_spin_forever, timeout=30.0),
            worker_count=1,
            lease_ms=500,
            renew_interval_ms=50,
            runtime_id="runner-stop-unreaped",
        )
        queued = await runner.submit(kind="echo", payload={"stop": "unreaped"})
        cycle = asyncio.create_task(runner.run_once())
        loop = asyncio.get_running_loop()
        start_deadline = loop.time() + 1.0
        while loop.time() < start_deadline and not (
            _task_row(store, queued.task_id).get("state") == "running"
            and runner._active_processes
        ):
            await asyncio.sleep(0.01)
        if (
            _task_row(store, queued.task_id).get("state") != "running"
            or not runner._active_processes
        ):
            raise SystemExit("direct-stop unreaped fixture process never started")
        original_stop = runner_module._stop_handler_process
        calls = 0

        async def fail_first_stop(process, *, allow_graceful_exit=False):
            nonlocal calls
            calls += 1
            if calls == 1:
                return False
            return await original_stop(
                process, allow_graceful_exit=allow_graceful_exit
            )

        runner_module._stop_handler_process = fail_first_stop  # type: ignore[assignment]
        try:
            await runner.stop()
        finally:
            runner_module._stop_handler_process = original_stop  # type: ignore[assignment]
        row = _task_row(store, queued.task_id)
        if (
            row.get("state") != "uncertain"
            or row.get("error_code") != "handler_process_unreaped"
        ):
            raise SystemExit(f"direct stop ignored an unproven reap: {row}")
        cycle.cancel()
        await asyncio.gather(cycle, return_exceptions=True)
        if (await sink.lookup(queued.task_id)).status != "missing":
            raise SystemExit("direct-stop unreaped assignment published a result")


async def test_stop_awaits_owned_assignment_terminal_cycle() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-stop-cycle-") as temp:
        store = _store(temp)
        runner = DurableSubagentRunner(
            store=store,
            custodian=InMemoryPayloadCustodian(),
            result_sink=InMemoryResultSink(),
            handlers=_registry(),
            worker_count=1,
            runtime_id="runner-stop-cycle",
        )
        queued = await runner.submit(kind="echo", payload={"stop": "durable"})
        started = asyncio.Event()
        cancelled = asyncio.Event()
        allow_finalize = asyncio.Event()
        cycle_completed = asyncio.Event()

        async def controlled_run_claim(claim):
            transition = store.start_subagent_assignment(
                claim.assignment_id,
                claim.worker_id,
                runner.runtime_id,
                claim.worker_generation,
                claim.lease_token,
            )
            if transition.status != "STARTED":
                raise SystemExit(f"mock stop-cycle assignment did not start: {transition}")
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await allow_finalize.wait()
                return await runner._finalize_failed(claim, "service_shutdown")
            finally:
                cycle_completed.set()

        runner._run_claim = controlled_run_claim  # type: ignore[method-assign]
        cycle = asyncio.create_task(runner.run_once())
        await asyncio.wait_for(started.wait(), timeout=1.0)
        stop = asyncio.create_task(runner.stop())
        await asyncio.wait_for(cancelled.wait(), timeout=1.0)
        await asyncio.sleep(0)
        if stop.done() or cycle_completed.is_set():
            raise SystemExit("stop returned before the owned assignment cycle could finalize")
        if _task_row(store, queued.task_id).get("state") != "running":
            raise SystemExit("stop-cycle fixture lost its pre-finalization running truth")

        allow_finalize.set()
        await asyncio.wait_for(stop, timeout=1.0)
        await asyncio.wait_for(cycle, timeout=1.0)
        row = _task_row(store, queued.task_id)
        assignments = _assignment_rows(store)
        if not cycle_completed.is_set() or not cycle.done():
            raise SystemExit("stop returned before the owned assignment cycle completed")
        if (
            row.get("state") != "failed"
            or row.get("error_code") != "service_shutdown"
            or len(assignments) != 1
            or assignments[0].get("state") != "failed"
            or assignments[0].get("error_code") != "service_shutdown"
        ):
            raise SystemExit(
                f"stop returned without durable failed/service_shutdown truth: {row} / {assignments}"
            )


async def test_stop_finalizes_claim_cancelled_during_prestart_io() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-stop-prestart-") as temp:
        store = _store(temp)
        custodian = InMemoryPayloadCustodian()
        runner = DurableSubagentRunner(
            store=store,
            custodian=custodian,
            result_sink=InMemoryResultSink(),
            handlers=_registry(),
            worker_count=1,
            runtime_id="runner-stop-prestart",
        )
        queued = await runner.submit(kind="echo", payload={"stop": "prestart"})
        fetch_entered = asyncio.Event()

        async def blocked_fetch(_request_digest):
            fetch_entered.set()
            await asyncio.Event().wait()

        custodian.fetch = blocked_fetch  # type: ignore[method-assign]
        cycle = asyncio.create_task(runner.run_once())
        await asyncio.wait_for(fetch_entered.wait(), timeout=1.0)
        await asyncio.wait_for(runner.stop(), timeout=1.0)
        await asyncio.wait_for(cycle, timeout=1.0)
        row = _task_row(store, queued.task_id)
        assignments = _assignment_rows(store)
        if (
            row.get("state") != "failed"
            or row.get("error_code") != "service_shutdown"
            or len(assignments) != 1
            or assignments[0].get("state") != "failed"
            or assignments[0].get("error_code") != "service_shutdown"
        ):
            raise SystemExit(
                "prestart shutdown returned without durable service_shutdown truth: "
                f"{row} / {assignments}"
            )


async def test_shutdown_reaps_cpu_bound_handler_without_waiting_for_handler_timeout() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-hard-shutdown-") as temp:
        store = _store(temp)
        before_pids = {child.pid for child in multiprocessing.active_children()}
        runner = DurableSubagentRunner(
            store=store,
            custodian=InMemoryPayloadCustodian(),
            result_sink=InMemoryResultSink(),
            handlers=_registry(_cpu_spin_forever, timeout=30.0),
            worker_count=1,
            lease_ms=500,
            renew_interval_ms=50,
            runtime_id="runner-hard-shutdown",
        )
        queued = await runner.submit(kind="echo", payload={"spin": True})
        stop_event = asyncio.Event()
        service = asyncio.create_task(runner.serve(stop_event, poll_seconds=0.01))
        loop = asyncio.get_running_loop()
        start_deadline = loop.time() + 1.0
        while loop.time() < start_deadline and _task_row(store, queued.task_id).get("state") != "running":
            await asyncio.sleep(0.01)
        if _task_row(store, queued.task_id).get("state") != "running":
            raise SystemExit("CPU-bound shutdown fixture never crossed the started boundary")
        stopped_at = loop.time()
        stop_event.set()
        await asyncio.wait_for(service, timeout=2.0)
        if loop.time() - stopped_at >= 1.5:
            raise SystemExit("runner shutdown waited for the hostile handler timeout")
        stopped_row = _task_row(store, queued.task_id)
        if stopped_row.get("state") != "failed" or stopped_row.get("error_code") != "service_shutdown":
            raise SystemExit(f"bounded shutdown did not record exact terminal truth: {stopped_row}")
        leaked = {
            child.pid
            for child in multiprocessing.active_children()
            if child.pid not in before_pids and child.is_alive()
        }
        if leaked:
            raise SystemExit(f"shutdown left a hostile handler child alive: {sorted(leaked)}")


async def test_external_service_cancellation_reaps_active_cycle() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-cancel-service-") as temp:
        store = _store(temp)
        before_pids = {child.pid for child in multiprocessing.active_children()}
        baseline_tasks = set(asyncio.all_tasks())
        runner = DurableSubagentRunner(
            store=store,
            custodian=InMemoryPayloadCustodian(),
            result_sink=InMemoryResultSink(),
            handlers=_registry(_cpu_spin_forever, timeout=30.0),
            worker_count=1,
            lease_ms=500,
            renew_interval_ms=50,
            runtime_id="runner-cancel-service",
        )
        queued = await runner.submit(kind="echo", payload={"cancel": True})
        service = asyncio.create_task(runner.serve(asyncio.Event(), poll_seconds=0.01))
        loop = asyncio.get_running_loop()
        start_deadline = loop.time() + 1.0
        while loop.time() < start_deadline and _task_row(store, queued.task_id).get("state") != "running":
            await asyncio.sleep(0.01)
        if _task_row(store, queued.task_id).get("state") != "running":
            raise SystemExit("service-cancellation fixture never crossed the start boundary")
        service.cancel()
        try:
            await asyncio.wait_for(service, timeout=2.0)
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0)
        leaked_tasks = [
            task
            for task in asyncio.all_tasks()
            if task not in baseline_tasks and not task.done()
        ]
        if leaked_tasks:
            raise SystemExit(
                f"external service cancellation left background tasks: {leaked_tasks!r}"
            )
        row = _task_row(store, queued.task_id)
        if row.get("state") != "failed" or row.get("error_code") != "service_shutdown":
            raise SystemExit(f"external service cancellation orphaned durable work: {row}")
        leaked = {
            child.pid
            for child in multiprocessing.active_children()
            if child.pid not in before_pids and child.is_alive()
        }
        if leaked:
            raise SystemExit(f"external service cancellation left child processes: {sorted(leaked)}")


async def test_result_waiter_requires_durable_success_and_reports_restart_loss() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-result-truth-") as temp:
        store = _store(temp)
        sink = InMemoryResultSink()
        runner = DurableSubagentRunner(
            store=store,
            custodian=InMemoryPayloadCustodian(),
            result_sink=sink,
            handlers=_registry(),
            worker_count=1,
            runtime_id="runner-result-truth",
        )
        succeeded = await runner.submit(kind="echo", payload={"restart": "loss"})
        outcomes = await runner.run_once()
        if not any(item.status == "SUCCEEDED" for item in outcomes):
            raise SystemExit(f"result-truth setup did not succeed: {outcomes}")
        store.reconcile_subagent_results_after_ephemeral_restart()
        after_restart = await runner_script._wait_for_result(
            store,
            InMemoryResultSink(),
            succeeded.task_id,
            timeout_seconds=0.1,
        )
        if after_restart != {
            "version": 1,
            "ok": False,
            "task_id": succeeded.task_id,
            "state": "result_unavailable_after_restart",
        }:
            raise SystemExit(f"missing post-restart result timed out or invented success: {after_restart}")

        queued = await runner.submit(kind="echo", payload={"sink": "must not lead"})
        queued_row = _task_row(store, queued.task_id)
        result_digest, envelope = result_envelope(
            task_id=queued.task_id,
            request_digest=queued_row["request_digest"],
            kind="echo",
            handler_id="jarvis.echo",
            handler_version=1,
            result_type="json",
            result={"sink_only": True},
        )
        committed = await sink.commit(
            ResultRecord(
                queued.task_id,
                queued_row["request_digest"],
                "echo",
                "jarvis.echo",
                1,
                result_digest,
                "json",
                envelope,
            )
        )
        if committed != "committed":
            raise SystemExit(f"sink-only fixture did not commit: {committed}")
        sink_only = await runner_script._wait_for_result(
            store, sink, queued.task_id, timeout_seconds=0.05
        )
        if sink_only.get("state") != "timeout" or sink_only.get("ok") is not False:
            raise SystemExit(f"sink-only result bypassed durable task truth: {sink_only}")
        await runner.stop()


async def test_ephemeral_capacity_and_sink_convergence() -> None:
    custody = InMemoryPayloadCustodian(max_items=1)
    first_digest, first_envelope = request_envelope(
        kind="echo",
        handler_id="jarvis.echo",
        handler_version=1,
        payload={"slot": 1},
    )
    second_digest, second_envelope = request_envelope(
        kind="echo",
        handler_id="jarvis.echo",
        handler_version=1,
        payload={"slot": 2},
    )
    if await custody.put_if_absent(first_digest, first_envelope) != "stored":
        raise SystemExit("bounded custody rejected its first item")
    if await custody.put_if_absent(second_digest, second_envelope) != "capacity":
        raise SystemExit("bounded custody silently exceeded its item ceiling")
    if not await custody.discard(first_digest):
        raise SystemExit("bounded custody could not discard an exact item")
    if await custody.put_if_absent(second_digest, second_envelope) != "stored":
        raise SystemExit("bounded custody did not admit after exact release")

    byte_custody = InMemoryPayloadCustodian(
        max_items=10, max_bytes=MAX_ENVELOPE_BYTES
    )
    large_a_digest, large_a = request_envelope(
        kind="echo",
        handler_id="jarvis.echo",
        handler_version=1,
        payload={"text": "a" * 600_000},
    )
    large_b_digest, large_b = request_envelope(
        kind="echo",
        handler_id="jarvis.echo",
        handler_version=1,
        payload={"text": "b" * 600_000},
    )
    if await byte_custody.put_if_absent(large_a_digest, large_a) != "stored":
        raise SystemExit("byte-bounded custody rejected its first valid envelope")
    if await byte_custody.put_if_absent(large_b_digest, large_b) != "capacity":
        raise SystemExit("byte-bounded custody silently exceeded its byte ceiling")

    sink = InMemoryResultSink(max_items=1)
    first_result_digest, first_result_envelope = result_envelope(
        task_id="task-capacity-1",
        request_digest=first_digest,
        kind="echo",
        handler_id="jarvis.echo",
        handler_version=1,
        result_type="json",
        result={"slot": 1},
    )
    first_record = ResultRecord(
        "task-capacity-1",
        first_digest,
        "echo",
        "jarvis.echo",
        1,
        first_result_digest,
        "json",
        first_result_envelope,
    )
    if await sink.stage(first_record) != "staged":
        raise SystemExit("bounded sink did not stage its first result")
    if await sink.commit(first_record) != "committed":
        raise SystemExit("commit did not converge with an exact staged result")
    if sink._staged or len(sink._items) != 1:
        raise SystemExit("stage/commit convergence stranded duplicate result bytes")

    second_result_digest, second_result_envelope = result_envelope(
        task_id="task-capacity-2",
        request_digest=second_digest,
        kind="echo",
        handler_id="jarvis.echo",
        handler_version=1,
        result_type="json",
        result={"slot": 2},
    )
    second_record = ResultRecord(
        "task-capacity-2",
        second_digest,
        "echo",
        "jarvis.echo",
        1,
        second_result_digest,
        "json",
        second_result_envelope,
    )
    if await sink.stage(second_record) != "capacity":
        raise SystemExit("bounded sink silently exceeded its item ceiling")
    if not await sink.discard(first_record.task_id, first_record.result_digest):
        raise SystemExit("bounded sink could not release a published result")
    if await sink.commit(second_record) != "committed":
        raise SystemExit("bounded sink did not admit after exact release")


async def test_terminal_payload_cleanup_and_result_expiry() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-retention-") as temp:
        store = _store(temp)
        custody = InMemoryPayloadCustodian(max_items=1)
        sink = InMemoryResultSink()
        runner = DurableSubagentRunner(
            store=store,
            custodian=custody,
            result_sink=sink,
            handlers=_registry(),
            worker_count=1,
            result_retention_ms=100,
            runtime_id="runner-retention",
        )
        secret = "RAW-TERMINAL-PAYLOAD-MUST-BE-RELEASED"
        queued = await runner.submit(kind="echo", payload={"text": secret})
        outcomes = await runner.run_once()
        if not any(item.status == "SUCCEEDED" for item in outcomes):
            raise SystemExit(f"retention fixture did not succeed: {outcomes}")
        task = store.get_subagent_task_ephemeral_status(queued.task_id)
        if (await custody.fetch(task.request_digest)).status != "missing":
            raise SystemExit("terminal success retained its request payload")
        available = await runner_script._wait_for_result(
            store, sink, queued.task_id, timeout_seconds=0.1
        )
        if available.get("state") != "result_available" or available.get("ok") is not True:
            raise SystemExit(f"available result was not delivered: {available}")

        duplicate = await runner.submit(kind="echo", payload={"text": secret})
        if duplicate.status != "EXISTING" or duplicate.task_id != queued.task_id:
            raise SystemExit(f"pre-expiry duplicate changed durable identity: {duplicate}")
        if (await custody.fetch(task.request_digest)).status != "missing":
            raise SystemExit("terminal duplicate resurrected request custody")

        expired = store.expire_due_subagent_results(
            checked_at_ms=task.result_expires_ms,
            limit=10,
        )
        if len(expired) != 1 or expired[0].task_id != queued.task_id:
            raise SystemExit(f"exact expiry boundary did not retire result: {expired}")
        for transition in expired:
            await runner_script._mark_result_bytes_released(store, sink, transition)
        expired_response = await runner_script._wait_for_result(
            store, sink, queued.task_id, timeout_seconds=0.05
        )
        if expired_response.get("state") != "result_expired":
            raise SystemExit(f"expired duplicate did not return explicit truth: {expired_response}")
        blocker_digest, blocker_envelope = request_envelope(
            kind="echo",
            handler_id="jarvis.echo",
            handler_version=1,
            payload={"capacity": "occupied"},
        )
        if await custody.put_if_absent(blocker_digest, blocker_envelope) != "stored":
            raise SystemExit("terminal duplicate capacity fixture did not fill custody")
        after_expiry = await runner.submit(kind="echo", payload={"text": secret})
        if after_expiry.status != "EXISTING" or (await custody.fetch(task.request_digest)).status != "missing":
            raise SystemExit("post-expiry duplicate replayed or retained its payload")
        if (await custody.fetch(blocker_digest)).status != "found":
            raise SystemExit("terminal duplicate disturbed unrelated active custody")
        if (await sink.lookup(queued.task_id)).status != "missing":
            raise SystemExit("expired result bytes remained in the sink")
        if secret.encode("utf-8") in store.db_path.read_bytes():
            raise SystemExit("retention metadata persisted raw payload content")
        await runner.stop()


async def test_result_acknowledgement_is_explicit_and_irreversible() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-ack-") as temp:
        store = _store(temp)
        custody = InMemoryPayloadCustodian()
        sink = InMemoryResultSink()
        runner = DurableSubagentRunner(
            store=store,
            custodian=custody,
            result_sink=sink,
            handlers=_registry(),
            worker_count=1,
            runtime_id="runner-ack",
        )
        queued = await runner.submit(kind="echo", payload={"ack": True})
        await runner.run_once()
        delivered = await runner_script._wait_for_result(
            store, sink, queued.task_id, timeout_seconds=0.1
        )
        if delivered.get("state") != "result_available":
            raise SystemExit(f"result was not available before explicit ack: {delivered}")
        if (await sink.lookup(queued.task_id)).status != "found":
            raise SystemExit("socket/result read was incorrectly treated as acknowledgement")
        acknowledged = await runner_script._ack_result(store, sink, queued.task_id)
        if acknowledged.get("state") != "result_acknowledged" or acknowledged.get("ok") is not True:
            raise SystemExit(f"explicit acknowledgement failed: {acknowledged}")
        if (await sink.lookup(queued.task_id)).status != "missing":
            raise SystemExit("acknowledged result bytes remained published")
        repeated = await runner_script._ack_result(store, sink, queued.task_id)
        if repeated.get("state") != "result_acknowledged":
            raise SystemExit(f"acknowledgement was not idempotent: {repeated}")
        duplicate = await runner.submit(kind="echo", payload={"ack": True})
        duplicate_result = await runner_script._wait_for_result(
            store, sink, duplicate.task_id, timeout_seconds=0.05
        )
        if duplicate_result.get("state") != "result_acknowledged":
            raise SystemExit(f"acknowledged duplicate did not return tombstone truth: {duplicate_result}")
        await runner.stop()


async def test_worker_activation_retries_after_fresh_restart_owner() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-reactivate-") as temp:
        store = _store(temp)
        first = DurableSubagentRunner(
            store=store,
            custodian=InMemoryPayloadCustodian(),
            result_sink=InMemoryResultSink(),
            handlers=_registry(),
            worker_count=1,
            runtime_id="runner-old-owner",
        )
        replacement = DurableSubagentRunner(
            store=store,
            custodian=InMemoryPayloadCustodian(),
            result_sink=InMemoryResultSink(),
            handlers=_registry(),
            worker_count=1,
            runtime_id="runner-new-owner",
        )
        await first.start()
        await replacement.start()
        if replacement._generations:
            raise SystemExit("fresh worker ownership fixture did not initially block replacement")
        original_activate = store.activate_subagent_worker
        future_ms = int(time.time() * 1000) + 60_000

        def activate_after_stale(*args: Any, **kwargs: Any):
            kwargs["checked_at_ms"] = future_ms
            return original_activate(*args, **kwargs)

        store.activate_subagent_worker = activate_after_stale  # type: ignore[method-assign]
        try:
            await replacement.run_once()
        finally:
            store.activate_subagent_worker = original_activate  # type: ignore[method-assign]
        if "subagent-01" not in replacement._generations:
            raise SystemExit("replacement runner stayed permanently workerless after lease staleness")
        await replacement.stop()
        await first.stop()


async def test_private_socket_service_round_trip_and_cleanup() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-socket-") as temp:
        root = Path(temp)
        socket_path = root / "runner.sock"
        config = SimpleNamespace(data_dir=root, db_path=root / "service.sqlite")
        original_load_config = runner_script.load_config
        runner_script.load_config = lambda: config  # type: ignore[assignment]
        stop_event = asyncio.Event()
        service = asyncio.create_task(
            runner_script.run_service(workers=2, socket_path=socket_path, stop_event=stop_event)
        )

        async def socket_request(payload: dict[str, Any]) -> dict[str, Any]:
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            writer.write(json.dumps(payload).encode("utf-8") + b"\n")
            await writer.drain()
            response = json.loads(
                (await asyncio.wait_for(reader.readline(), timeout=5)).decode("ascii")
            )
            writer.close()
            await writer.wait_closed()
            return response

        dangling_reader: asyncio.StreamReader | None = None
        dangling_writer: asyncio.StreamWriter | None = None
        try:
            for _ in range(200):
                if socket_path.exists():
                    break
                await asyncio.sleep(0.01)
            if not socket_path.exists() or socket_path.stat().st_mode & 0o077:
                raise SystemExit("private runner socket was not created owner-only")
            for malformed_version in (None, True, 1.0, 2):
                malformed = {
                    "operation": "status",
                    "task_id": "task-version-check",
                }
                if malformed_version is not None:
                    malformed["version"] = malformed_version
                rejected = await socket_request(malformed)
                if rejected.get("state") != "unsupported_version":
                    raise SystemExit(
                        f"wire accepted malformed version {malformed_version!r}: {rejected}"
                    )
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            secret = "RAW-SOCKET-PAYLOAD-MUST-NOT-PERSIST"
            request = {
                "version": 1,
                "operation": "submit",
                "kind": "text_metrics",
                "payload": {"text": secret},
            }
            writer.write(json.dumps(request).encode("utf-8") + b"\n")
            await writer.drain()
            response = json.loads((await asyncio.wait_for(reader.readline(), timeout=5)).decode("ascii"))
            writer.close()
            await writer.wait_closed()
            if response.get("ok") is not True or response.get("result") != {
                "characters": len(secret),
                "lines": 1,
                "words": 1,
            }:
                raise SystemExit(f"private runner socket returned the wrong bounded result: {response}")
            if response.get("version") != 1 or response.get("state") != "result_available":
                raise SystemExit(f"private runner response was not versioned: {response}")
            task_id = response.get("task_id")
            status = await socket_request(
                {"version": 1, "operation": "status", "task_id": task_id}
            )
            if status.get("state") != "result_available" or status.get("result") != response.get("result"):
                raise SystemExit(f"content-free status lookup lost the available result: {status}")
            acknowledged = await socket_request(
                {"version": 1, "operation": "ack_result", "task_id": task_id}
            )
            if acknowledged.get("state") != "result_acknowledged" or acknowledged.get("ok") is not True:
                raise SystemExit(f"socket acknowledgement did not retire the result: {acknowledged}")
            after_ack = await socket_request(
                {"version": 1, "operation": "status", "task_id": task_id}
            )
            if after_ack.get("state") != "result_acknowledged":
                raise SystemExit(f"status lost acknowledged tombstone truth: {after_ack}")
            duplicate = await socket_request(request)
            if duplicate.get("task_id") != task_id or duplicate.get("state") != "result_acknowledged":
                raise SystemExit(f"terminal duplicate replayed or waited after acknowledgement: {duplicate}")
            if secret.encode("utf-8") in config.db_path.read_bytes():
                raise SystemExit("private socket payload leaked into durable SQLite")
            dangling_reader, dangling_writer = await asyncio.open_unix_connection(str(socket_path))
            dangling_writer.write(b'{"operation":"submit"')
            await dangling_writer.drain()
        finally:
            stop_event.set()
            await asyncio.wait_for(service, timeout=5)
            runner_script.load_config = original_load_config
            if dangling_reader is not None:
                if await asyncio.wait_for(dangling_reader.read(), timeout=1) != b"":
                    raise SystemExit("shutdown did not close a partial private-socket client")
            if dangling_writer is not None:
                dangling_writer.close()
                await dangling_writer.wait_closed()
        if socket_path.exists():
            raise SystemExit("private runner socket was not removed on shutdown")


async def test_runner_failure_still_cleans_private_socket() -> None:
    with TemporaryDirectory(prefix="jsr-fail-") as temp:
        root = Path(temp)
        socket_path = root / "s"
        config = SimpleNamespace(data_dir=root, db_path=root / "service.sqlite")
        original_load_config = runner_script.load_config
        original_serve = DurableSubagentRunner.serve
        runner_script.load_config = lambda: config  # type: ignore[assignment]

        async def fail_runner(self, stop_event, *, poll_seconds=0.5):
            await asyncio.sleep(0.05)
            raise RuntimeError("injected runner failure")

        DurableSubagentRunner.serve = fail_runner  # type: ignore[assignment]
        try:
            try:
                await asyncio.wait_for(
                    runner_script.run_service(
                        workers=1,
                        socket_path=socket_path,
                        stop_event=asyncio.Event(),
                    ),
                    timeout=3.0,
                )
            except RuntimeError as exc:
                if str(exc) != "injected runner failure":
                    raise
            else:
                raise SystemExit("runner failure fixture did not propagate its failure")
        finally:
            DurableSubagentRunner.serve = original_serve  # type: ignore[assignment]
            runner_script.load_config = original_load_config
        if socket_path.exists():
            raise SystemExit("runner failure left its private socket path behind")


async def test_proven_stale_socket_is_reclaimed_after_crash() -> None:
    with TemporaryDirectory(prefix="jsr-stale-") as temp:
        root = Path(temp)
        socket_path = root / "runner.sock"
        config = SimpleNamespace(data_dir=root, db_path=root / "service.sqlite")
        lock_fd = runner_script._acquire_socket_ownership(socket_path)
        stale_server = await asyncio.start_unix_server(
            lambda _reader, _writer: None,
            path=str(socket_path),
            **({"cleanup_socket": False} if sys.version_info >= (3, 13) else {}),
        )
        stale_stat = socket_path.lstat()
        runner_script._record_socket_identity(
            lock_fd, (stale_stat.st_dev, stale_stat.st_ino)
        )
        runner_script._release_socket_ownership(lock_fd, clear=False)
        stale_server.close()
        await stale_server.wait_closed()
        if not socket_path.exists():
            raise SystemExit("stale-socket fixture did not preserve the crash pathname")

        original_load_config = runner_script.load_config
        runner_script.load_config = lambda: config  # type: ignore[assignment]
        stop_event = asyncio.Event()
        stop_event.set()
        try:
            await runner_script.run_service(
                workers=1,
                socket_path=socket_path,
                stop_event=stop_event,
            )
        finally:
            runner_script.load_config = original_load_config
        if socket_path.exists():
            raise SystemExit("proven stale socket blocked or survived automatic restart")


async def test_socket_lock_never_modifies_unproven_files() -> None:
    with TemporaryDirectory(prefix="jsr-lock-foreign-") as temp:
        root = Path(temp)
        socket_path = root / "runner.sock"
        lock_path = runner_script._socket_lock_path(socket_path)
        foreign = b"FOREIGN LOCK CONTENT MUST SURVIVE"
        lock_path.write_bytes(foreign)
        lock_path.chmod(0o600)
        try:
            await runner_script.run_service(
                workers=1,
                socket_path=socket_path,
                stop_event=asyncio.Event(),
            )
        except RuntimeError as exc:
            if "ownership lock is invalid" not in str(exc):
                raise
        else:
            raise SystemExit("runner adopted an unmarked foreign lock file")
        if lock_path.read_bytes() != foreign or stat.S_IMODE(lock_path.stat().st_mode) != 0o600:
            raise SystemExit("runner modified an unmarked foreign lock file")

    with TemporaryDirectory(prefix="jsr-lock-hardlink-") as temp:
        root = Path(temp)
        socket_path = root / "runner.sock"
        lock_path = runner_script._socket_lock_path(socket_path)
        source = root / "foreign-source"
        foreign = b"FOREIGN HARDLINK TARGET MUST SURVIVE"
        source.write_bytes(foreign)
        source.chmod(0o600)
        os.link(source, lock_path)
        before = source.stat()
        try:
            await runner_script.run_service(
                workers=1,
                socket_path=socket_path,
                stop_event=asyncio.Event(),
            )
        except RuntimeError as exc:
            if "ownership lock is invalid" not in str(exc):
                raise
        else:
            raise SystemExit("runner adopted a foreign hardlinked lock file")
        after = source.stat()
        if source.read_bytes() != foreign or after.st_nlink != 2 or (
            before.st_dev,
            before.st_ino,
        ) != (after.st_dev, after.st_ino):
            raise SystemExit("runner modified a foreign hardlinked lock target")

    with TemporaryDirectory(prefix="jsr-lock-replace-") as temp:
        root = Path(temp)
        socket_path = root / "runner.sock"
        lock_path = runner_script._socket_lock_path(socket_path)
        replacement = b"FOREIGN REPLACEMENT LOCK MUST SURVIVE"
        original_flock = runner_script.fcntl.flock

        def replace_after_lock(fd: int, operation: int) -> None:
            original_flock(fd, operation)
            if operation & runner_script.fcntl.LOCK_EX:
                lock_path.unlink()
                lock_path.write_bytes(replacement)
                lock_path.chmod(0o600)

        runner_script.fcntl.flock = replace_after_lock  # type: ignore[assignment]
        try:
            try:
                await runner_script.run_service(
                    workers=1,
                    socket_path=socket_path,
                    stop_event=asyncio.Event(),
                )
            except RuntimeError as exc:
                if "ownership lock is invalid" not in str(exc):
                    raise
            else:
                raise SystemExit("runner accepted a replaced lock pathname")
        finally:
            runner_script.fcntl.flock = original_flock  # type: ignore[assignment]
        if lock_path.read_bytes() != replacement:
            raise SystemExit("runner modified a foreign lock-path replacement")


async def test_socket_lock_journal_survives_torn_writes_and_prepublication_crash() -> None:
    with TemporaryDirectory(prefix="jsr-lock-create-crash-") as temp:
        root = Path(temp)
        socket_path = root / "runner.sock"
        lock_path = runner_script._socket_lock_path(socket_path)
        creation_token = "1" * 24
        creation_path = root / f".jvl-{creation_token}"
        creation_fd = os.open(
            creation_path,
            os.O_CREAT | os.O_EXCL | os.O_RDWR,
            0o600,
        )
        try:
            runner_script._write_socket_lock(
                creation_fd,
                None,
                creation_token=creation_token,
            )
            os.link(creation_path, lock_path)
        finally:
            os.close(creation_fd)
        if lock_path.stat().st_nlink != 2:
            raise SystemExit("creation-crash fixture did not preserve both lock links")
        recovered_fd = runner_script._acquire_socket_ownership(socket_path)
        if creation_path.exists() or lock_path.stat().st_nlink != 1:
            raise SystemExit("runner did not recover an exact interrupted lock publication")
        runner_script._release_socket_ownership(recovered_fd, clear=True)

    with TemporaryDirectory(prefix="jsr-lock-torn-") as temp:
        root = Path(temp)
        socket_path = root / "runner.sock"
        lock_fd = runner_script._acquire_socket_ownership(socket_path)
        original_pwrite = runner_script.os.pwrite
        injected = False

        def tear_next_slot(fd: int, payload: bytes, offset: int) -> int:
            nonlocal injected
            if fd == lock_fd and not injected:
                injected = True
                original_pwrite(fd, payload[: len(payload) // 2], offset)
                raise OSError("synthetic torn lock journal write")
            return original_pwrite(fd, payload, offset)

        runner_script.os.pwrite = tear_next_slot  # type: ignore[assignment]
        try:
            try:
                runner_script._record_socket_identity(lock_fd, (123, 456))
            except OSError as exc:
                if "synthetic torn" not in str(exc):
                    raise
            else:
                raise SystemExit("torn journal fixture did not interrupt the write")
        finally:
            runner_script.os.pwrite = original_pwrite  # type: ignore[assignment]
            runner_script.fcntl.flock(lock_fd, runner_script.fcntl.LOCK_UN)
            os.close(lock_fd)
        recovered_fd = runner_script._acquire_socket_ownership(socket_path)
        runner_script._release_socket_ownership(recovered_fd, clear=True)

    with TemporaryDirectory(prefix="jsr-lock-prepublish-") as temp:
        root = Path(temp)
        socket_path = root / "runner.sock"
        private_path = root / ".jv-prepublish"
        lock_fd = runner_script._acquire_socket_ownership(socket_path)
        private_server = await asyncio.start_unix_server(
            lambda _reader, _writer: None,
            path=str(private_path),
        )
        private_stat = private_path.lstat()
        runner_script._record_socket_identity(
            lock_fd, (private_stat.st_dev, private_stat.st_ino)
        )
        runner_script._release_socket_ownership(lock_fd, clear=False)
        private_server.close()
        await private_server.wait_closed()
        if private_path.exists():
            private_path.unlink()
        recovered_fd = runner_script._acquire_socket_ownership(socket_path)
        runner_script._release_socket_ownership(recovered_fd, clear=True)


async def test_socket_absence_is_synced_before_idle_journal() -> None:
    with TemporaryDirectory(prefix="jsr-lock-sync-order-") as temp:
        root = Path(temp)
        socket_path = root / "runner.sock"
        config = SimpleNamespace(data_dir=root, db_path=root / "service.sqlite")
        original_load_config = runner_script.load_config
        original_fsync_directory = runner_script._fsync_directory
        original_release = runner_script._release_socket_ownership
        events: list[str] = []

        def tracked_fsync(path: Path) -> None:
            original_fsync_directory(path)
            events.append("directory_synced")

        def tracked_release(lock_fd: int, *, clear: bool) -> None:
            if clear:
                events.append("idle_journal")
            original_release(lock_fd, clear=clear)

        runner_script.load_config = lambda: config  # type: ignore[assignment]
        runner_script._fsync_directory = tracked_fsync  # type: ignore[assignment]
        runner_script._release_socket_ownership = tracked_release  # type: ignore[assignment]
        stop_event = asyncio.Event()
        stop_event.set()
        try:
            await runner_script.run_service(
                workers=1,
                socket_path=socket_path,
                stop_event=stop_event,
            )
        finally:
            runner_script.load_config = original_load_config
            runner_script._fsync_directory = original_fsync_directory  # type: ignore[assignment]
            runner_script._release_socket_ownership = original_release  # type: ignore[assignment]
        if events[-2:] != ["directory_synced", "idle_journal"]:
            raise SystemExit(f"socket absence was not durable before idle journal: {events}")

async def test_socket_publication_never_replaces_or_unlinks_foreign_paths() -> None:
    with TemporaryDirectory(prefix="jsr-own-") as temp:
        root = Path(temp)
        socket_path = root / "s"
        config = SimpleNamespace(data_dir=root, db_path=root / "service.sqlite")
        original_load_config = runner_script.load_config
        runner_script.load_config = lambda: config  # type: ignore[assignment]
        incumbent_store = MemoryStore(config.db_path)
        incumbent_store.init()
        now_ms = int(time.time() * 1000)
        with incumbent_store.connect() as conn:
            conn.execute(
                """
                INSERT INTO subagent_tasks(
                    task_id, request_digest, kind, priority, replay_policy, state,
                    attempt_count, max_attempts, result_digest, result_type,
                    result_state, result_expires_ms, created_ms, updated_ms, finished_ms
                ) VALUES (?, ?, 'echo', 0, 'safe', 'succeeded', 1, 1, ?, 'json',
                          'available', ?, ?, ?, ?)
                """,
                (
                    "task-live-incumbent",
                    "a" * 64,
                    "b" * 64,
                    now_ms + 60_000,
                    now_ms,
                    now_ms,
                    now_ms,
                ),
            )
        incumbent = await asyncio.start_unix_server(
            lambda _reader, _writer: None,
            path=str(socket_path),
        )
        incumbent_stat = socket_path.lstat()
        try:
            try:
                await runner_script.run_service(
                    workers=1,
                    socket_path=socket_path,
                    stop_event=asyncio.Event(),
                )
            except RuntimeError as exc:
                if "already occupied" not in str(exc):
                    raise
            else:
                raise SystemExit("runner replaced a live incumbent socket")
            preserved = socket_path.lstat()
            if (preserved.st_dev, preserved.st_ino) != (
                incumbent_stat.st_dev,
                incumbent_stat.st_ino,
            ):
                raise SystemExit("runner changed the live incumbent socket identity")
            if (
                incumbent_store.get_subagent_task_ephemeral_status(
                    "task-live-incumbent"
                ).result_state
                != "available"
            ):
                raise SystemExit("failed duplicate launch reconciled a live service result")
        finally:
            incumbent.close()
            await incumbent.wait_closed()
            if socket_path.exists():
                socket_path.unlink()

        original_link = runner_script.os.link
        replacement = b"foreign replacement"

        def replace_after_publication(source, target, *, follow_symlinks=True):
            original_link(source, target, follow_symlinks=follow_symlinks)
            Path(target).unlink()
            Path(target).write_bytes(replacement)

        runner_script.os.link = replace_after_publication  # type: ignore[assignment]
        try:
            try:
                await runner_script.run_service(
                    workers=1,
                    socket_path=socket_path,
                    stop_event=asyncio.Event(),
                )
            except RuntimeError as exc:
                if "publication changed" not in str(exc):
                    raise
            else:
                raise SystemExit("runner accepted a replaced socket publication")
        finally:
            runner_script.os.link = original_link  # type: ignore[assignment]
        if socket_path.read_bytes() != replacement:
            raise SystemExit("runner cleanup unlinked or changed a foreign socket replacement")

        socket_path.unlink()
        original_unlink = Path.unlink
        late_replacement: socket.socket | None = None
        late_replacement_identity: tuple[int, int] | None = None

        def replace_after_verification(path, *args, **kwargs):
            nonlocal late_replacement, late_replacement_identity
            result = original_unlink(path, *args, **kwargs)
            if path.name.startswith(".jv-"):
                original_unlink(socket_path)
                late_replacement = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                late_replacement.bind(str(socket_path))
                late_replacement.listen(1)
                replacement_stat = socket_path.lstat()
                late_replacement_identity = (replacement_stat.st_dev, replacement_stat.st_ino)
            return result

        Path.unlink = replace_after_verification  # type: ignore[assignment]
        stop_event = asyncio.Event()
        stop_event.set()
        try:
            await runner_script.run_service(
                workers=1,
                socket_path=socket_path,
                stop_event=stop_event,
            )
        finally:
            Path.unlink = original_unlink  # type: ignore[assignment]
            runner_script.load_config = original_load_config
        try:
            if late_replacement is None or late_replacement_identity is None:
                raise SystemExit("post-verification socket replacement was not installed")
            preserved_stat = socket_path.lstat()
            if (preserved_stat.st_dev, preserved_stat.st_ino) != late_replacement_identity:
                raise SystemExit("cleanup adopted a socket replacement after ownership verification")
        finally:
            if late_replacement is not None:
                late_replacement.close()
            if socket_path.exists():
                socket_path.unlink()


def test_registry_and_envelope_boundaries() -> None:
    source = inspect.getsource(runner_module)
    for forbidden in ("build_core_registry", "JarvisRuntime", "Executor(", "run_shell_command"):
        if forbidden in source:
            raise SystemExit(f"production runner imported a forbidden general execution surface: {forbidden}")
    try:
        HandlerRegistry([HandlerSpec("echo", "jarvis.echo", 1, "json", 1.0, _echo, pure=False)])
    except ValueError:
        pass
    else:
        raise SystemExit("runner accepted an impure handler")

    async def nested_handler(payload: Any) -> Any:
        return payload

    try:
        HandlerSpec("nested", "jarvis.nested", 1, "json", 1.0, nested_handler)
    except ValueError:
        pass
    else:
        raise SystemExit("runner accepted a non-importable closure for spawned execution")
    digest, envelope = request_envelope(
        kind="echo", handler_id="jarvis.echo", handler_version=1, payload={"b": 2, "a": 1}
    )
    repeated, repeated_envelope = request_envelope(
        kind="echo", handler_id="jarvis.echo", handler_version=1, payload={"a": 1, "b": 2}
    )
    if digest != repeated or envelope != repeated_envelope or digest != hashlib.sha256(envelope).hexdigest():
        raise SystemExit("request envelope is not canonical and digest-bound")


async def test_submit_validates_before_custody_and_enqueue() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-submit-validation-") as temp:
        store = _store(temp)
        custody = InMemoryPayloadCustodian()

        runner = DurableSubagentRunner(
            store=store,
            custodian=custody,
            result_sink=InMemoryResultSink(),
            handlers=HandlerRegistry(
                [
                    HandlerSpec(
                        "typed",
                        "jarvis.typed",
                        1,
                        "json",
                        1.0,
                        _typed_payload,
                        lambda payload: type(payload) is dict and set(payload) == {"value"},
                    )
                ]
            ),
            worker_count=1,
            runtime_id="runner-submit-validation",
        )
        try:
            await runner.submit(kind="typed", payload={"wrong": "shape"})
        except ValueError:
            pass
        else:
            raise SystemExit("runner enqueued a payload before registered schema validation")
        with store.connect() as conn:
            if conn.execute("SELECT COUNT(*) FROM subagent_tasks").fetchone()[0] != 0:
                raise SystemExit("invalid payload created a durable task")
        if custody._items:
            raise SystemExit("invalid payload entered custody")


async def test_public_status_separates_attachment_from_dispatch() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-runner-status-") as temp:
        store = _store(temp)
        runner = DurableSubagentRunner(
            store=store,
            custodian=InMemoryPayloadCustodian(),
            result_sink=InMemoryResultSink(),
            handlers=_registry(),
            worker_count=1,
            runtime_id="runner-status",
        )
        await runner.start()
        idle = _durable_fleet_status_snapshot(store.subagent_control_snapshot())
        if idle.get("ok") is not True or idle.get("runner_attached") is not True or idle.get("dispatch_ready") is not False:
            raise SystemExit(f"idle public runner status conflated attachment and dispatch: {idle}")
        await runner.submit(kind="echo", payload={"queued": True})
        queued = _durable_fleet_status_snapshot(store.subagent_control_snapshot())
        if queued.get("runner_attached") is not True or queued.get("dispatch_ready") is not True:
            raise SystemExit(f"queued public runner status missed claim readiness: {queued}")
        await runner.stop()


async def main_async() -> None:
    test_registry_and_envelope_boundaries()
    await test_ephemeral_capacity_and_sink_convergence()
    await test_submit_validates_before_custody_and_enqueue()
    await test_public_status_separates_attachment_from_dispatch()
    await test_happy_path_and_privacy()
    await test_missing_unknown_and_tampered_payloads_reject_before_start()
    await test_existing_sink_result_is_adopted_without_handler_replay()
    await test_lease_renewal_and_handler_failure()
    await test_sink_collision_quarantines_running_assignment()
    await test_ephemeral_restart_rejects_lost_payload()
    await test_transient_custody_outage_requeues_then_succeeds()
    await test_enqueue_commit_response_loss_preserves_custody()
    await test_lost_finalize_response_uses_exact_terminal_readback()
    await test_lost_failed_finalize_response_uses_exact_terminal_readback()
    await test_publish_after_retention_deadline_expires_without_resurrection()
    await test_lease_loss_cancels_and_suppresses_late_result()
    await test_parent_delay_cannot_accept_post_deadline_result()
    await test_lease_expiry_during_result_stage_never_publishes()
    await test_hard_timeout_reaps_hostile_handlers_and_preserves_fleet_progress()
    await test_delayed_reap_retains_terminal_custody()
    await test_cleanup_renewal_refusal_reaps_without_false_terminalization()
    await test_unproven_reap_quarantines_before_result_publication()
    await test_direct_stop_reap_failure_quarantines_assignment()
    await test_stop_awaits_owned_assignment_terminal_cycle()
    await test_stop_finalizes_claim_cancelled_during_prestart_io()
    await test_shutdown_reaps_cpu_bound_handler_without_waiting_for_handler_timeout()
    await test_external_service_cancellation_reaps_active_cycle()
    await test_result_waiter_requires_durable_success_and_reports_restart_loss()
    await test_terminal_payload_cleanup_and_result_expiry()
    await test_result_acknowledgement_is_explicit_and_irreversible()
    await test_worker_activation_retries_after_fresh_restart_owner()
    await test_private_socket_service_round_trip_and_cleanup()
    await test_runner_failure_still_cleans_private_socket()
    await test_proven_stale_socket_is_reclaimed_after_crash()
    await test_socket_lock_never_modifies_unproven_files()
    await test_socket_lock_journal_survives_torn_writes_and_prepublication_crash()
    await test_socket_absence_is_synced_before_idle_journal()
    await test_socket_publication_never_replaces_or_unlinks_foreign_paths()


def main() -> None:
    asyncio.run(main_async())
    print("Subagent production runner smoke passed")


if __name__ == "__main__":
    main()
