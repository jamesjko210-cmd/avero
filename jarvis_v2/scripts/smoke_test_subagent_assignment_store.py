from __future__ import annotations

import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
from typing import Callable
from unittest.mock import patch

import jarvis_v2.memory.store as store_module
from jarvis_v2.memory.store import MemoryStore


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _store(root: str, name: str = "subagents.sqlite") -> MemoryStore:
    store = MemoryStore(Path(root) / name)
    store.init()
    return store


def _expect_value_error(label: str, operation: Callable[[], object]) -> None:
    try:
        operation()
    except ValueError:
        return
    raise SystemExit(f"{label} did not reject invalid input")


def _prepare_task(
    store: MemoryStore,
    generation: int,
    label: str,
    checked_at_ms: int,
    *,
    start: bool = True,
) -> tuple[object, object, str]:
    request_digest = _digest(f"request:{label}")
    task = store.enqueue_subagent_task(
        request_digest=request_digest,
        kind="mock",
        checked_at_ms=checked_at_ms,
    )
    claim = store.claim_subagent_assignment(
        "subagent-01",
        "runtime-a",
        generation,
        checked_at_ms=checked_at_ms,
        lease_ms=10_000,
    )
    if claim.status != "CLAIMED" or claim.task_id != task.task_id:
        raise SystemExit(f"retention fixture was not claimed: {task}, {claim}")
    if start:
        started = store.start_subagent_assignment(
            claim.assignment_id,
            "subagent-01",
            "runtime-a",
            generation,
            claim.lease_token,
            checked_at_ms=checked_at_ms,
        )
        if started.status != "STARTED":
            raise SystemExit(f"retention fixture was not started: {started}")
    return task, claim, request_digest


def _finalize_retained_task(
    store: MemoryStore,
    generation: int,
    label: str,
    checked_at_ms: int,
    *,
    retention_ms: int = 100,
) -> tuple[object, str, str]:
    task, claim, request_digest = _prepare_task(store, generation, label, checked_at_ms)
    result_digest = _digest(f"result:{label}")
    transition = store.finalize_subagent_assignment(
        claim.assignment_id,
        "subagent-01",
        "runtime-a",
        generation,
        claim.lease_token,
        outcome="succeeded",
        result_digest=result_digest,
        result_type="mock",
        checked_at_ms=checked_at_ms,
        result_retention_ms=retention_ms,
    )
    if transition.status != "SUCCEEDED":
        raise SystemExit(f"retained fixture did not finalize: {transition}")
    return task, request_digest, result_digest


def test_worker_lifecycle_identity_and_capacity() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-workers-") as temp:
        store = _store(temp)
        if store.configure_subagent_workers(2, checked_at_ms=100) != 2:
            raise SystemExit("worker configuration did not create the requested capacity")
        if store.configure_subagent_workers(1, checked_at_ms=101) != 1:
            raise SystemExit("worker downsizing did not change enabled capacity")
        if store.activate_subagent_worker("subagent-99", "runtime-a", checked_at_ms=110) is not None:
            raise SystemExit("an unknown worker was activated")
        generation_a = store.activate_subagent_worker("subagent-01", "runtime-a", checked_at_ms=110)
        if generation_a != 1:
            raise SystemExit("configured worker did not activate")
        if store.heartbeat_subagent_worker("subagent-01", "runtime-b", generation_a, checked_at_ms=115):
            raise SystemExit("wrong runtime heartbeat crossed the worker identity fence")
        if not store.heartbeat_subagent_worker("subagent-01", "runtime-a", generation_a, checked_at_ms=120):
            raise SystemExit("worker heartbeat was not accepted")
        if store.activate_subagent_worker(
            "subagent-01", "runtime-b", checked_at_ms=149, stale_after_ms=30
        ) is not None:
            raise SystemExit("live worker identity was stolen before the stale boundary")
        if store.activate_subagent_worker(
            "subagent-01", "runtime-b", checked_at_ms=150, stale_after_ms=30
        ) is not None:
            raise SystemExit("worker identity was stolen at the exact live boundary")
        generation_b = store.activate_subagent_worker(
            "subagent-01", "runtime-b", checked_at_ms=151, stale_after_ms=30
        )
        if generation_b != 2:
            raise SystemExit("stale worker identity was not replaced with a new generation")
        if store.heartbeat_subagent_worker("subagent-01", "runtime-b", generation_a, checked_at_ms=151):
            raise SystemExit("old generation heartbeat crossed the ownership fence")
        if store.deactivate_subagent_worker("subagent-01", "runtime-a", generation_a, checked_at_ms=152):
            raise SystemExit("stale runtime deactivated a replacement worker")
        if not store.deactivate_subagent_worker("subagent-01", "runtime-b", generation_b, checked_at_ms=152):
            raise SystemExit("current runtime could not deactivate its worker")
        if store.heartbeat_subagent_worker("subagent-01", "runtime-b", generation_b, checked_at_ms=153):
            raise SystemExit("offline worker accepted a heartbeat")

        store.configure_subagent_workers(2, checked_at_ms=159)
        generation_c = store.activate_subagent_worker("subagent-01", "runtime-c", checked_at_ms=160)
        generation_d = store.activate_subagent_worker("subagent-02", "runtime-d", checked_at_ms=160)
        if generation_c is None or generation_d is None:
            raise SystemExit("enabled workers did not reactivate")
        first = store.enqueue_subagent_task(
            request_digest=_digest("capacity-one"), kind="mock", priority=2, checked_at_ms=160
        )
        second = store.enqueue_subagent_task(
            request_digest=_digest("capacity-two"), kind="mock", priority=1, checked_at_ms=160
        )
        claim_one = store.claim_subagent_assignment(
            "subagent-01", "runtime-c", generation_c, checked_at_ms=161, lease_ms=100
        )
        if claim_one.status != "CLAIMED" or claim_one.task_id != first.task_id:
            raise SystemExit(f"first worker did not claim the highest-priority task: {claim_one}")
        duplicate = store.claim_subagent_assignment(
            "subagent-01", "runtime-c", generation_c, checked_at_ms=161, lease_ms=100
        )
        if duplicate.status != "WORKER_NOT_READY":
            raise SystemExit(f"one worker accepted more than one active assignment: {duplicate}")
        if store.deactivate_subagent_worker("subagent-01", "runtime-c", generation_c, checked_at_ms=162):
            raise SystemExit("worker with an active assignment was deactivated")
        if store.activate_subagent_worker(
            "subagent-01", "runtime-e", checked_at_ms=1_000, stale_after_ms=1
        ) is not None:
            raise SystemExit("active worker capacity was stolen despite assignment ownership")
        claim_two = store.claim_subagent_assignment(
            "subagent-02", "runtime-d", generation_d, checked_at_ms=161, lease_ms=100
        )
        if claim_two.status != "CLAIMED" or claim_two.task_id != second.task_id:
            raise SystemExit(f"second worker did not provide independent capacity: {claim_two}")


def test_enqueue_idempotency_and_collision() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-enqueue-") as temp:
        store = _store(temp)
        request_digest = _digest("same-private-request")
        created = store.enqueue_subagent_task(
            request_digest=request_digest,
            kind="research",
            priority=7,
            replay_policy="safe",
            max_attempts=3,
            checked_at_ms=200,
        )
        repeated = store.enqueue_subagent_task(
            request_digest=request_digest,
            kind="research",
            priority=7,
            replay_policy="safe",
            max_attempts=3,
            checked_at_ms=201,
        )
        collision = store.enqueue_subagent_task(
            request_digest=request_digest,
            kind="research",
            priority=8,
            replay_policy="safe",
            max_attempts=3,
            checked_at_ms=202,
        )
        if created.status != "ENQUEUED" or repeated.status != "EXISTING":
            raise SystemExit(f"enqueue idempotency diverged: {created}, {repeated}")
        if collision.status != "COLLISION" or len({created.task_id, repeated.task_id, collision.task_id}) != 1:
            raise SystemExit(f"enqueue collision did not preserve the original identity: {collision}")
        with store.connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM subagent_tasks").fetchone()[0]
        if count != 1:
            raise SystemExit(f"idempotent enqueue wrote {count} task rows")


def test_concurrent_claims_across_store_instances() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-concurrent-") as temp:
        first_store = _store(temp)
        second_store = MemoryStore(first_store.db_path)
        second_store.init()
        first_store.configure_subagent_workers(2, checked_at_ms=300)
        generation_one = first_store.activate_subagent_worker("subagent-01", "runtime-one", checked_at_ms=300)
        generation_two = first_store.activate_subagent_worker("subagent-02", "runtime-two", checked_at_ms=300)
        if generation_one is None or generation_two is None:
            raise SystemExit("concurrent claim workers did not activate")
        task = first_store.enqueue_subagent_task(
            request_digest=_digest("concurrent-one-shot"), kind="mock", checked_at_ms=300
        )
        barrier = Barrier(2)

        def claim(store: MemoryStore, worker_id: str, runtime_id: str, generation: int) -> object:
            barrier.wait(timeout=10)
            return store.claim_subagent_assignment(
                worker_id, runtime_id, generation, checked_at_ms=301, lease_ms=100
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(claim, first_store, "subagent-01", "runtime-one", generation_one),
                pool.submit(claim, second_store, "subagent-02", "runtime-two", generation_two),
            ]
            results = [future.result(timeout=10) for future in futures]
        winners = [result for result in results if result.status == "CLAIMED"]
        losers = [result for result in results if result.status != "CLAIMED"]
        if len(winners) != 1 or winners[0].task_id != task.task_id:
            raise SystemExit(f"concurrent stores did not produce one claim winner: {results}")
        if len(losers) != 1 or losers[0].status != "QUEUE_EMPTY" or losers[0].lease_token:
            raise SystemExit(f"concurrent claim loser received authority or wrong status: {results}")
        with second_store.connect() as conn:
            assignments = conn.execute("SELECT COUNT(*) FROM subagent_assignments").fetchone()[0]
            attempts = conn.execute(
                "SELECT attempt_count FROM subagent_tasks WHERE task_id = ?", (task.task_id,)
            ).fetchone()[0]
        if assignments != 1 or attempts != 1:
            raise SystemExit("concurrent claims duplicated an assignment or attempt")


def test_lease_start_renew_finalize_fencing_and_expiry_boundary() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-leases-") as temp:
        store = _store(temp)
        store.configure_subagent_workers(1, checked_at_ms=400)
        generation_a = store.activate_subagent_worker("subagent-01", "runtime-a", checked_at_ms=400)
        if generation_a is None:
            raise SystemExit("lease worker did not activate")
        store.enqueue_subagent_task(
            request_digest=_digest("boundary-task"), kind="mock", replay_policy="safe", checked_at_ms=400
        )
        boundary = store.claim_subagent_assignment(
            "subagent-01", "runtime-a", generation_a, checked_at_ms=400, lease_ms=10
        )
        wrong_start = store.start_subagent_assignment(
            boundary.assignment_id, "subagent-01", "runtime-a", generation_a, "wrong-token", checked_at_ms=409
        )
        if wrong_start.status != "LEASE_LOST":
            raise SystemExit(f"wrong lease token crossed the start fence: {wrong_start}")
        started = store.start_subagent_assignment(
            boundary.assignment_id,
            "subagent-01",
            "runtime-a",
            generation_a,
            boundary.lease_token,
            checked_at_ms=409,
        )
        repeated_start = store.start_subagent_assignment(
            boundary.assignment_id,
            "subagent-01",
            "runtime-a",
            generation_a,
            boundary.lease_token,
            checked_at_ms=409,
        )
        if started.status != "STARTED" or repeated_start.status != "ALREADY_RUNNING":
            raise SystemExit(f"assignment start transition diverged: {started}, {repeated_start}")
        if store.renew_subagent_assignment(
            boundary.assignment_id,
            "subagent-01",
            "runtime-a",
            generation_a,
            boundary.lease_token,
            checked_at_ms=410,
            lease_ms=10,
        ):
            raise SystemExit("lease renewed at its exact expiry boundary")
        expired_finish = store.finalize_subagent_assignment(
            boundary.assignment_id,
            "subagent-01",
            "runtime-a",
            generation_a,
            boundary.lease_token,
            outcome="failed",
            error_code="late",
            checked_at_ms=410,
        )
        if expired_finish.status != "LEASE_LOST":
            raise SystemExit(f"completion succeeded at the exact expiry boundary: {expired_finish}")
        recovered = store.recover_expired_subagent_assignments(checked_at_ms=410)
        if recovered["uncertain"] != 1:
            raise SystemExit(f"running lease was not uncertain at exact expiry: {recovered}")
        stale_finish = store.finalize_subagent_assignment(
            boundary.assignment_id,
            "subagent-01",
            "runtime-a",
            generation_a,
            boundary.lease_token,
            outcome="failed",
            error_code="stale",
            checked_at_ms=411,
        )
        if stale_finish.status != "LEASE_LOST":
            raise SystemExit(f"stale completion changed a recovered assignment: {stale_finish}")

        generation_b = store.activate_subagent_worker("subagent-01", "runtime-b", checked_at_ms=420)
        if generation_b is None:
            raise SystemExit("recovered worker did not reactivate")
        store.enqueue_subagent_task(
            request_digest=_digest("renewed-task"), kind="mock", checked_at_ms=420
        )
        renewed = store.claim_subagent_assignment(
            "subagent-01", "runtime-b", generation_b, checked_at_ms=420, lease_ms=10
        )
        if renewed.replay_policy != "manual":
            raise SystemExit(f"claim omitted replay policy: {renewed}")
        if store.start_subagent_assignment(
            renewed.assignment_id,
            "subagent-01",
            "runtime-b",
            generation_b,
            renewed.lease_token,
            checked_at_ms=421,
        ).status != "STARTED":
            raise SystemExit("renewal fixture did not start")
        if store.renew_subagent_assignment(
            renewed.assignment_id,
            "subagent-01",
            "runtime-a",
            generation_a,
            renewed.lease_token,
            checked_at_ms=429,
            lease_ms=20,
        ):
            raise SystemExit("stale runtime renewed another runtime's lease")
        if not store.renew_subagent_assignment(
            renewed.assignment_id,
            "subagent-01",
            "runtime-b",
            generation_b,
            renewed.lease_token,
            checked_at_ms=429,
            lease_ms=20,
        ):
            raise SystemExit("valid lease renewal failed")
        result_digest = _digest("bounded-result")
        finished = store.finalize_subagent_assignment(
            renewed.assignment_id,
            "subagent-01",
            "runtime-b",
            generation_b,
            renewed.lease_token,
            outcome="succeeded",
            result_digest=result_digest,
            result_type="mock",
            checked_at_ms=448,
        )
        if finished.status != "SUCCEEDED":
            raise SystemExit(f"valid renewed completion failed: {finished}")
        replay = store.finalize_subagent_assignment(
            renewed.assignment_id,
            "subagent-01",
            "runtime-b",
            generation_b,
            renewed.lease_token,
            outcome="succeeded",
            result_digest=result_digest,
            result_type="mock",
            checked_at_ms=448,
        )
        if replay.status != "LEASE_LOST":
            raise SystemExit(f"finalization was not one-shot fenced: {replay}")

        readback = store.get_subagent_assignment_status(renewed.assignment_id)
        if (
            readback.status != "FOUND"
            or readback.state != "succeeded"
            or not readback.started
            or not readback.terminal
            or readback.result_digest != result_digest
            or readback.result_type != "mock"
            or readback.error_code
            or hasattr(readback, "lease_token")
        ):
            raise SystemExit(f"terminal success readback was not deterministic and bounded: {readback}")
        legacy_retention = store.get_subagent_task_ephemeral_status(renewed.task_id)
        if (
            legacy_retention.result_state != "none"
            or legacy_retention.result_expires_ms is not None
            or legacy_retention.result_released_ms is not None
        ):
            raise SystemExit(f"low-level success unexpectedly enabled retention: {legacy_retention}")


def test_prestart_rejection_is_fenced_terminal_and_inspectable() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-reject-") as temp:
        store = _store(temp)
        store.configure_subagent_workers(1, checked_at_ms=450)
        generation = store.activate_subagent_worker("subagent-01", "runtime-a", checked_at_ms=450)
        if generation is None:
            raise SystemExit("pre-start rejection worker did not activate")
        store.enqueue_subagent_task(
            request_digest=_digest("reject-before-start"),
            kind="mock",
            replay_policy="safe",
            max_attempts=3,
            checked_at_ms=450,
        )
        claim = store.claim_subagent_assignment(
            "subagent-01", "runtime-a", generation, checked_at_ms=450, lease_ms=20
        )
        if claim.status != "CLAIMED" or claim.replay_policy != "safe":
            raise SystemExit(f"pre-start rejection fixture was not claimed correctly: {claim}")
        stale = store.reject_subagent_assignment_before_start(
            claim.assignment_id,
            "subagent-01",
            "runtime-a",
            generation + 1,
            claim.lease_token,
            error_code="provider_unavailable",
            checked_at_ms=451,
        )
        wrong_token = store.reject_subagent_assignment_before_start(
            claim.assignment_id,
            "subagent-01",
            "runtime-a",
            generation,
            "wrong-token",
            error_code="provider_unavailable",
            checked_at_ms=451,
        )
        if stale.status != "LEASE_LOST" or wrong_token.status != "LEASE_LOST":
            raise SystemExit(f"pre-start rejection crossed a generation/lease fence: {stale}, {wrong_token}")
        rejected = store.reject_subagent_assignment_before_start(
            claim.assignment_id,
            "subagent-01",
            "runtime-a",
            generation,
            claim.lease_token,
            error_code="provider_unavailable",
            checked_at_ms=451,
        )
        if rejected.status != "REJECTED":
            raise SystemExit(f"valid pre-start rejection failed: {rejected}")
        status = store.get_subagent_assignment_status(claim.assignment_id)
        if (
            status.status != "FOUND"
            or status.state != "failed"
            or status.started
            or not status.terminal
            or status.error_code != "provider_unavailable"
            or status.result_digest
            or status.result_type
        ):
            raise SystemExit(f"pre-start rejection was not durably inspectable: {status}")
        repeated = store.reject_subagent_assignment_before_start(
            claim.assignment_id,
            "subagent-01",
            "runtime-a",
            generation,
            claim.lease_token,
            error_code="provider_unavailable",
            checked_at_ms=452,
        )
        if repeated.status != "LEASE_LOST" or store.get_subagent_assignment_status(claim.assignment_id) != status:
            raise SystemExit(f"replayed rejection was not deterministic: {repeated}")
        with store.connect() as conn:
            task = conn.execute(
                "SELECT state, attempt_count, error_code FROM subagent_tasks WHERE task_id = ?", (claim.task_id,)
            ).fetchone()
            worker = conn.execute(
                "SELECT state, current_assignment_id FROM subagent_workers WHERE worker_id = 'subagent-01'"
            ).fetchone()
        if tuple(task) != ("failed", 1, "provider_unavailable"):
            raise SystemExit(f"pre-start rejection did not terminally fail its task: {tuple(task)}")
        if tuple(worker) != ("online", None):
            raise SystemExit(f"pre-start rejection did not release its live worker: {tuple(worker)}")


def test_claimed_vs_running_recovery() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-recovery-") as temp:
        store = _store(temp)
        store.configure_subagent_workers(3, checked_at_ms=500)
        generations: dict[int, int] = {}
        for index in range(1, 4):
            generation = store.activate_subagent_worker(
                f"subagent-{index:02d}", f"runtime-{index}", checked_at_ms=500
            )
            if generation is None:
                raise SystemExit("recovery worker did not activate")
            generations[index] = generation
        store.enqueue_subagent_task(
            request_digest=_digest("claimed-requeue"),
            kind="mock",
            priority=30,
            replay_policy="safe",
            max_attempts=2,
            checked_at_ms=500,
        )
        store.enqueue_subagent_task(
            request_digest=_digest("claimed-final"),
            kind="mock",
            priority=20,
            replay_policy="manual",
            max_attempts=1,
            checked_at_ms=500,
        )
        store.enqueue_subagent_task(
            request_digest=_digest("running-unknown"),
            kind="mock",
            priority=10,
            replay_policy="safe",
            max_attempts=3,
            checked_at_ms=500,
        )
        claims = [
            store.claim_subagent_assignment(
                f"subagent-{index:02d}",
                f"runtime-{index}",
                generations[index],
                checked_at_ms=500,
                lease_ms=10,
            )
            for index in range(1, 4)
        ]
        if any(claim.status != "CLAIMED" for claim in claims):
            raise SystemExit(f"recovery fixtures were not claimed: {claims}")
        running = claims[2]
        if store.start_subagent_assignment(
            running.assignment_id,
            running.worker_id,
            "runtime-3",
            generations[3],
            running.lease_token,
            checked_at_ms=501,
        ).status != "STARTED":
            raise SystemExit("running recovery fixture did not start")
        before = store.recover_expired_subagent_assignments(checked_at_ms=509)
        if before != {"abandoned": 0, "requeued": 0, "failed": 0, "uncertain": 0}:
            raise SystemExit(f"recovery ran before expiry: {before}")
        counts = store.recover_expired_subagent_assignments(checked_at_ms=510)
        expected = {"abandoned": 2, "requeued": 2, "failed": 1, "uncertain": 1}
        if counts != expected:
            raise SystemExit(f"claimed/running recovery semantics diverged: {counts}")
        with store.connect() as conn:
            task_states = dict(
                conn.execute("SELECT error_code, state FROM subagent_tasks ORDER BY priority DESC")
            )
            assignment_states = [
                row[0]
                for row in conn.execute("SELECT state FROM subagent_assignments ORDER BY claimed_ms, task_id")
            ]
            offline = conn.execute(
                "SELECT COUNT(*) FROM subagent_workers WHERE state = 'offline' AND current_assignment_id IS NULL"
            ).fetchone()[0]
        if set(task_states.values()) != {"queued", "failed"}:
            raise SystemExit(f"recovered task states were wrong: {task_states}")
        if sorted(assignment_states) != ["abandoned", "abandoned", "uncertain"] or offline != 3:
            raise SystemExit(
                f"recovery did not fence assignments/workers: states={assignment_states}, offline={offline}"
            )


def test_running_recovery_can_disable_requeue_for_ephemeral_custody() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-no-running-requeue-") as temp:
        store = _store(temp)
        store.configure_subagent_workers(1, checked_at_ms=550)
        generation = store.activate_subagent_worker("subagent-01", "runtime-a", checked_at_ms=550)
        if generation is None:
            raise SystemExit("ephemeral-custody worker did not activate")
        store.enqueue_subagent_task(
            request_digest=_digest("ephemeral-custody-running"),
            kind="mock",
            replay_policy="safe",
            max_attempts=3,
            checked_at_ms=550,
        )
        claim = store.claim_subagent_assignment(
            "subagent-01", "runtime-a", generation, checked_at_ms=550, lease_ms=10
        )
        started = store.start_subagent_assignment(
            claim.assignment_id,
            claim.worker_id,
            "runtime-a",
            generation,
            claim.lease_token,
            checked_at_ms=551,
        )
        if claim.status != "CLAIMED" or started.status != "STARTED":
            raise SystemExit(f"ephemeral-custody fixture did not start: {claim}, {started}")
        counts = store.recover_expired_subagent_assignments(
            checked_at_ms=560, allow_running_requeue=False
        )
        if counts != {"abandoned": 0, "requeued": 0, "failed": 0, "uncertain": 1}:
            raise SystemExit(f"ephemeral-custody recovery requeued running work: {counts}")
        status = store.get_subagent_assignment_status(claim.assignment_id)
        with store.connect() as conn:
            task = conn.execute(
                "SELECT state, attempt_count, max_attempts, replay_policy, error_code FROM subagent_tasks"
            ).fetchone()
        if status.state != "uncertain" or not status.started or not status.terminal:
            raise SystemExit(f"expired running assignment was not marked uncertain: {status}")
        if tuple(task) != ("uncertain", 1, 3, "safe", "running_lease_expired"):
            raise SystemExit(f"safe running task escaped ephemeral custody fencing: {tuple(task)}")


def test_restart_persistence_and_snapshot_privacy_bounds() -> None:
    private_payload = "/\x55sers/private/assignment-payload-never-store"
    with TemporaryDirectory(prefix="jarvis-subagent-snapshot-") as temp:
        store = _store(temp)
        store.configure_subagent_workers(3, checked_at_ms=600)
        claims = []
        for index in range(1, 4):
            worker_id = f"subagent-{index:02d}"
            runtime_id = f"runtime-{index}"
            generation = store.activate_subagent_worker(worker_id, runtime_id, checked_at_ms=600)
            if generation is None:
                raise SystemExit("snapshot worker did not activate")
            store.enqueue_subagent_task(
                request_digest=_digest(f"{private_payload}-{index}"),
                kind="private_mock",
                priority=4 - index,
                checked_at_ms=600,
            )
            claims.append(
                store.claim_subagent_assignment(
                    worker_id, runtime_id, generation, checked_at_ms=600, lease_ms=100
                )
            )
        snapshot = store.subagent_control_snapshot(checked_at_ms=601, limit=1)
        if (
            snapshot["configured_workers"] != 3
            or len(snapshot["workers"]) != 1
            or not snapshot["workers_truncated"]
            or len(snapshot["active_assignments"]) != 1
            or not snapshot["active_assignments_truncated"]
        ):
            raise SystemExit(f"snapshot bounds were not enforced: {snapshot}")
        if not snapshot["ok"] or snapshot["ready_workers"] != 0 or snapshot["dispatch_ready"]:
            raise SystemExit(f"snapshot capacity accounting diverged: {snapshot}")
        if not snapshot["runner_attached"]:
            raise SystemExit(f"busy healthy runners were reported detached: {snapshot}")
        for flag in ("payloads_persisted", "results_persisted", "errors_persisted", "lease_tokens_exposed"):
            if snapshot[flag]:
                raise SystemExit(f"snapshot privacy flag became unsafe: {flag}")
        exposed = repr(snapshot)
        for claim in claims:
            if not claim.lease_token or claim.lease_token in exposed:
                raise SystemExit("snapshot exposed or omitted private lease authority")
        if private_payload in exposed:
            raise SystemExit("snapshot exposed a raw task payload")
        with store.connect() as conn:
            stored_leases = [row[0] for row in conn.execute("SELECT lease_digest FROM subagent_assignments")]
        if any(claim.lease_token in stored_leases for claim in claims):
            raise SystemExit("SQLite stored a raw lease token instead of its digest")
        database_bytes = store.db_path.read_bytes()
        if private_payload.encode("utf-8") in database_bytes:
            raise SystemExit("SQLite persisted a raw task payload")
        if any(claim.lease_token.encode("ascii") in database_bytes for claim in claims):
            raise SystemExit("SQLite persisted raw lease authority")

        restarted = MemoryStore(store.db_path)
        restarted.init()
        persisted = restarted.subagent_control_snapshot(checked_at_ms=601, limit=50)
        if (
            persisted["configured_workers"] != 3
            or persisted["task_counts"] != {"running": 3}
            or persisted["assignment_counts"] != {"claimed": 3}
            or len(persisted["active_assignments"]) != 3
        ):
            raise SystemExit(f"restart lost durable subagent state: {persisted}")


def test_control_snapshot_runner_attachment_and_dispatch_readiness() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-readiness-") as temp:
        store = _store(temp)
        store.configure_subagent_workers(1, checked_at_ms=700)
        generation = store.activate_subagent_worker("subagent-01", "runtime-a", checked_at_ms=700)
        if generation is None:
            raise SystemExit("readiness worker did not activate")
        idle = store.subagent_control_snapshot(checked_at_ms=701)
        if not idle["runner_attached"] or idle["ready_workers"] != 1 or idle["dispatch_ready"]:
            raise SystemExit(f"idle runner without queued work was reported dispatch-ready: {idle}")
        store.enqueue_subagent_task(
            request_digest=_digest("readiness-task"), kind="mock", checked_at_ms=701
        )
        ready = store.subagent_control_snapshot(checked_at_ms=702)
        if not ready["runner_attached"] or ready["ready_workers"] != 1 or not ready["dispatch_ready"]:
            raise SystemExit(f"live ready runner with queued work was not dispatch-ready: {ready}")
        claim = store.claim_subagent_assignment(
            "subagent-01", "runtime-a", generation, checked_at_ms=702, lease_ms=100
        )
        if claim.status != "CLAIMED":
            raise SystemExit(f"readiness task was not claimed: {claim}")
        busy = store.subagent_control_snapshot(checked_at_ms=703)
        if not busy["runner_attached"] or busy["ready_workers"] != 0 or busy["dispatch_ready"]:
            raise SystemExit(f"busy healthy runner attachment was misreported: {busy}")


def test_finalization_contracts_are_checked_before_sqlite() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-finalize-validation-") as temp:
        store = _store(temp)
        connection_attempts = 0

        def reject_connection() -> object:
            nonlocal connection_attempts
            connection_attempts += 1
            raise AssertionError("invalid finalization reached SQLite")

        store.connect = reject_connection  # type: ignore[method-assign]
        invalid: list[tuple[str, Callable[[], object]]] = [
            (
                "success missing result type",
                lambda: store.finalize_subagent_assignment(
                    "assignment-1", "subagent-01", "runtime", 1, "token",
                    outcome="succeeded", result_digest=_digest("result"),
                ),
            ),
            (
                "failure missing error code",
                lambda: store.finalize_subagent_assignment(
                    "assignment-1", "subagent-01", "runtime", 1, "token", outcome="failed"
                ),
            ),
        ]
        for label, operation in invalid:
            _expect_value_error(label, operation)
        if connection_attempts != 0:
            raise SystemExit("invalid finalization contracts were delegated to SQLite")


def test_ephemeral_result_schema_migration_and_validation() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-retention-migration-") as temp:
        db_path = Path(temp) / "legacy.sqlite"
        with sqlite3.connect(db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE subagent_tasks (
                    task_id TEXT PRIMARY KEY,
                    request_digest TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 0,
                    replay_policy TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 1,
                    result_digest TEXT,
                    result_type TEXT,
                    error_code TEXT,
                    created_ms INTEGER NOT NULL,
                    updated_ms INTEGER NOT NULL,
                    finished_ms INTEGER
                );
                """
            )
            conn.execute(
                "INSERT INTO subagent_tasks VALUES (?, ?, 'mock', 0, 'manual', 'succeeded', "
                "1, 1, ?, 'mock', NULL, 10, 20, 20)",
                ("task-legacy-success", _digest("legacy-success"), _digest("legacy-result")),
            )
            conn.execute(
                "INSERT INTO subagent_tasks VALUES (?, ?, 'mock', 0, 'manual', 'failed', "
                "1, 1, NULL, NULL, 'failed', 10, 20, 20)",
                ("task-legacy-failed", _digest("legacy-failed")),
            )
        store = MemoryStore(db_path)
        with patch.object(store_module.time, "time_ns", return_value=7_000_000_000):
            store.init()
        with store.connect() as conn:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(subagent_tasks)")}
            success = conn.execute(
                "SELECT result_state, result_expires_ms, result_released_ms "
                "FROM subagent_tasks WHERE task_id = 'task-legacy-success'"
            ).fetchone()
            failed = conn.execute(
                "SELECT result_state, result_expires_ms, result_released_ms "
                "FROM subagent_tasks WHERE task_id = 'task-legacy-failed'"
            ).fetchone()
            index_sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'index' "
                "AND name = 'subagent_result_available_expiry_idx'"
            ).fetchone()
            assignment_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(subagent_assignments)")
            }
        if columns < {"result_state", "result_expires_ms", "result_released_ms"}:
            raise SystemExit(f"retention migration omitted columns: {columns}")
        if tuple(success) != ("restart_unavailable", None, 7_000):
            raise SystemExit(f"historical success was not tombstoned at migration: {tuple(success)}")
        if tuple(failed) != ("none", None, None):
            raise SystemExit(f"historical non-success changed retention state: {tuple(failed)}")
        if index_sql is None or "result_state = 'available'" not in str(index_sql[0]):
            raise SystemExit(f"available-expiry index was not declared: {index_sql}")
        if assignment_columns & {"result_state", "result_expires_ms", "result_released_ms"}:
            raise SystemExit("retention migration changed assignment rows")

        store.init()
        with store.connect() as conn:
            conn.execute(
                "UPDATE subagent_tasks SET result_state = 'available', result_expires_ms = 100 "
                "WHERE task_id = 'task-legacy-failed'"
            )
        try:
            store.init()
        except RuntimeError as exc:
            if "subagent result retention state is invalid" not in str(exc):
                raise
        else:
            raise SystemExit("init accepted an invalid migrated cross-state combination")


def test_ephemeral_result_retention_state_machine() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-retention-") as temp:
        store = _store(temp)
        store.configure_subagent_workers(1, checked_at_ms=1_000)
        generation = store.activate_subagent_worker(
            "subagent-01", "runtime-a", checked_at_ms=1_000
        )
        if generation is None:
            raise SystemExit("retention worker did not activate")

        task, request_digest, result_digest = _finalize_retained_task(
            store, generation, "expiry-sweep", 1_000
        )
        if task.task_state != "queued":
            raise SystemExit(f"enqueue omitted exact current task state: {task}")
        repeated_enqueue = store.enqueue_subagent_task(
            request_digest=request_digest,
            kind="mock",
            checked_at_ms=1_001,
        )
        if repeated_enqueue.status != "EXISTING" or repeated_enqueue.task_state != "succeeded":
            raise SystemExit(f"enqueue did not return the durable current state: {repeated_enqueue}")
        publishing = store.get_subagent_task_ephemeral_status(task.task_id)
        by_digest = store.get_subagent_task_ephemeral_status_by_request_digest(request_digest)
        missing_by_digest = store.get_subagent_task_ephemeral_status_by_request_digest(
            _digest("missing-request")
        )
        if (
            publishing.status != "FOUND"
            or publishing.state != "succeeded"
            or publishing.result_state != "publishing"
            or publishing.result_expires_ms != 1_100
            or publishing.result_released_ms is not None
            or by_digest != publishing
            or missing_by_digest.status != "NOT_FOUND"
        ):
            raise SystemExit(
                f"ephemeral status lookup was not exact: {publishing}, {by_digest}, {missing_by_digest}"
            )
        if store.acknowledge_subagent_result(
            task.task_id, request_digest, result_digest, checked_at_ms=1_001
        ).status != "REFUSED":
            raise SystemExit("publishing result was acknowledged before availability")
        available = store.mark_subagent_result_available(
            task.task_id, request_digest, result_digest, checked_at_ms=1_001
        )
        repeated_available = store.mark_subagent_result_available(
            task.task_id, request_digest, result_digest, checked_at_ms=1_002
        )
        if available.status != "AVAILABLE" or repeated_available.status != "ALREADY_AVAILABLE":
            raise SystemExit(f"availability transition was not idempotent: {available}, {repeated_available}")
        if store.expire_due_subagent_results(checked_at_ms=1_099):
            raise SystemExit("result expired one millisecond before its boundary")
        expired = store.expire_due_subagent_results(checked_at_ms=1_100)
        if len(expired) != 1 or expired[0].status != "EXPIRED" or expired[0].task_id != task.task_id:
            raise SystemExit(f"exact-boundary expiry diverged: {expired}")
        if store.expire_due_subagent_results(checked_at_ms=1_101):
            raise SystemExit("expired tombstone was transitioned twice")
        if store.acknowledge_subagent_result(
            task.task_id, request_digest, result_digest, checked_at_ms=1_101
        ).status != "EXPIRED":
            raise SystemExit("expired tombstone did not remain irreversible")
        releases = store.list_subagent_result_releases()
        if [row.task_id for row in releases] != [task.task_id]:
            raise SystemExit(f"expiry release queue was not exact: {releases}")
        released = store.mark_subagent_result_released(
            task.task_id,
            request_digest,
            result_digest,
            "expired",
            checked_at_ms=1_102,
        )
        repeated_release = store.mark_subagent_result_released(
            task.task_id,
            request_digest,
            result_digest,
            "expired",
            checked_at_ms=1_103,
        )
        if released.status != "RELEASED" or repeated_release.status != "ALREADY_RELEASED":
            raise SystemExit(f"release transition was not idempotent: {released}, {repeated_release}")
        if store.list_subagent_result_releases():
            raise SystemExit("released tombstone remained in the cleanup queue")

        late_publish, late_request, late_result = _finalize_retained_task(
            store, generation, "late-publish", 1_200
        )
        late_transition = store.mark_subagent_result_available(
            late_publish.task_id,
            late_request,
            late_result,
            checked_at_ms=1_300,
        )
        if late_transition.status != "EXPIRED" or late_transition.result_state != "expired":
            raise SystemExit(
                f"publication resurrected a result at its expiry boundary: {late_transition}"
            )
        swept_publish, _swept_request, _swept_result = _finalize_retained_task(
            store, generation, "publishing-sweep", 1_400
        )
        swept = store.expire_due_subagent_results(checked_at_ms=1_500)
        if not any(
            row.task_id == swept_publish.task_id and row.result_state == "expired"
            for row in swept
        ):
            raise SystemExit(f"overdue publishing result escaped expiry sweep: {swept}")

        boundary_tasks = [
            _finalize_retained_task(store, generation, f"ack-boundary-{offset}", 2_000)
            for offset in (-1, 0, 1)
        ]
        for boundary_task, boundary_request, boundary_result in boundary_tasks:
            store.mark_subagent_result_available(
                boundary_task.task_id, boundary_request, boundary_result, checked_at_ms=2_001
            )
        boundary_statuses = [
            store.acknowledge_subagent_result(
                boundary_task.task_id,
                boundary_request,
                boundary_result,
                checked_at_ms=2_100 + offset,
            ).status
            for offset, (boundary_task, boundary_request, boundary_result) in zip(
                (-1, 0, 1), boundary_tasks
            )
        ]
        if boundary_statuses != ["ACKNOWLEDGED", "EXPIRED", "EXPIRED"]:
            raise SystemExit(f"acknowledgment expiry boundaries diverged: {boundary_statuses}")
        first_boundary = boundary_tasks[0]
        if store.acknowledge_subagent_result(
            first_boundary[0].task_id,
            first_boundary[1],
            first_boundary[2],
            checked_at_ms=2_101,
        ).status != "ALREADY_ACKNOWLEDGED":
            raise SystemExit("acknowledged tombstone was not idempotent")
        acknowledged_release = store.mark_subagent_result_released(
            first_boundary[0].task_id,
            first_boundary[1],
            first_boundary[2],
            "acknowledged",
            checked_at_ms=2_102,
        )
        if acknowledged_release.status != "RELEASED":
            raise SystemExit(f"acknowledged result was not releasable: {acknowledged_release}")

        unavailable_task, unavailable_request, unavailable_result = _finalize_retained_task(
            store, generation, "unavailable", 3_000
        )
        wrong_identity = store.mark_subagent_result_available(
            unavailable_task.task_id,
            _digest("wrong-request"),
            unavailable_result,
            checked_at_ms=3_001,
        )
        unavailable = store.mark_subagent_result_unavailable(
            unavailable_task.task_id,
            unavailable_request,
            unavailable_result,
            checked_at_ms=3_001,
        )
        repeated_unavailable = store.mark_subagent_result_unavailable(
            unavailable_task.task_id,
            unavailable_request,
            unavailable_result,
            checked_at_ms=3_002,
        )
        if (
            wrong_identity.status != "REFUSED"
            or unavailable.status != "UNAVAILABLE"
            or repeated_unavailable.status != "ALREADY_UNAVAILABLE"
            or store.mark_subagent_result_available(
                unavailable_task.task_id,
                unavailable_request,
                unavailable_result,
                checked_at_ms=3_003,
            ).status != "REFUSED"
        ):
            raise SystemExit(
                f"unavailable tombstone or identity refusal diverged: "
                f"{wrong_identity}, {unavailable}, {repeated_unavailable}"
            )
        wrong_release = store.mark_subagent_result_released(
            unavailable_task.task_id,
            unavailable_request,
            _digest("wrong-result"),
            "unavailable",
            checked_at_ms=3_004,
        )
        unavailable_release = store.mark_subagent_result_released(
            unavailable_task.task_id,
            unavailable_request,
            unavailable_result,
            "unavailable",
            checked_at_ms=3_004,
        )
        if wrong_release.status != "REFUSED" or unavailable_release.status != "RELEASED":
            raise SystemExit(
                f"unavailable release identity fence diverged: {wrong_release}, {unavailable_release}"
            )

        adopted_task, adopted_claim, adopted_request = _prepare_task(
            store, generation, "adopted", 3_100, start=False
        )
        adopted_result = _digest("result:adopted")
        adopted = store.adopt_subagent_result(
            adopted_claim.assignment_id,
            "subagent-01",
            "runtime-a",
            generation,
            adopted_claim.lease_token,
            result_digest=adopted_result,
            result_type="mock",
            checked_at_ms=3_100,
            result_retention_ms=250,
        )
        adopted_status = store.get_subagent_task_ephemeral_status(adopted_task.task_id)
        if (
            adopted.status != "ADOPTED"
            or adopted_status.result_state != "publishing"
            or adopted_status.result_expires_ms != 3_350
        ):
            raise SystemExit(f"adoption did not atomically start retention: {adopted}, {adopted_status}")

        failed_task, failed_claim, _ = _prepare_task(store, generation, "failed", 3_200)
        failed = store.finalize_subagent_assignment(
            failed_claim.assignment_id,
            "subagent-01",
            "runtime-a",
            generation,
            failed_claim.lease_token,
            outcome="failed",
            error_code="failed",
            checked_at_ms=3_200,
            result_retention_ms=100,
        )
        failed_status = store.get_subagent_task_ephemeral_status(failed_task.task_id)
        if (
            failed.status != "FAILED"
            or failed_status.result_state != "none"
            or failed_status.error_code != "failed"
        ):
            raise SystemExit(f"failed path entered result retention: {failed}, {failed_status}")

        publishing_restart, publishing_request, publishing_result = _finalize_retained_task(
            store, generation, "restart-publishing", 4_000
        )
        available_restart, available_request, available_result = _finalize_retained_task(
            store, generation, "restart-available", 4_000
        )
        store.mark_subagent_result_available(
            available_restart.task_id, available_request, available_result, checked_at_ms=4_001
        )
        counts = store.reconcile_subagent_results_after_ephemeral_restart(checked_at_ms=4_010)
        if counts != {"publishing": 2, "available": 1}:
            raise SystemExit(f"restart reconciliation counts diverged: {counts}")
        for restart_task, restart_request, restart_result in (
            (publishing_restart, publishing_request, publishing_result),
            (available_restart, available_request, available_result),
            (adopted_task, adopted_request, adopted_result),
        ):
            status = store.get_subagent_task_ephemeral_status(restart_task.task_id)
            if status.result_state != "restart_unavailable" or status.result_released_ms != 4_010:
                raise SystemExit(f"restart did not create a released tombstone: {status}")
            if store.mark_subagent_result_available(
                restart_task.task_id,
                restart_request,
                restart_result,
                checked_at_ms=4_011,
            ).status != "REFUSED":
                raise SystemExit("restart tombstone returned to availability")

        queued = store.enqueue_subagent_task(
            request_digest=_digest("request:queued-refusal"), kind="mock", checked_at_ms=4_020
        )
        queued_refusal = store.mark_subagent_result_available(
            queued.task_id,
            _digest("request:queued-refusal"),
            _digest("result:queued-refusal"),
            checked_at_ms=4_020,
        )
        if queued_refusal.status != "REFUSED":
            raise SystemExit(f"nonterminal task entered result retention: {queued_refusal}")

        private_payload = "raw-result-content-must-never-persist"
        if private_payload.encode("ascii") in store.db_path.read_bytes():
            raise SystemExit("retention state persisted raw result content")


def test_concurrent_acknowledgment_vs_expiry() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-retention-race-") as temp:
        first = _store(temp)
        second = MemoryStore(first.db_path)
        second.init()
        first.configure_subagent_workers(1, checked_at_ms=5_000)
        generation = first.activate_subagent_worker(
            "subagent-01", "runtime-a", checked_at_ms=5_000
        )
        if generation is None:
            raise SystemExit("retention race worker did not activate")
        task, request_digest, result_digest = _finalize_retained_task(
            first, generation, "ack-expiry-race", 5_000
        )
        first.mark_subagent_result_available(
            task.task_id, request_digest, result_digest, checked_at_ms=5_001
        )
        barrier = Barrier(2)

        def acknowledge() -> object:
            barrier.wait(timeout=10)
            return first.acknowledge_subagent_result(
                task.task_id, request_digest, result_digest, checked_at_ms=5_100
            )

        def expire() -> object:
            barrier.wait(timeout=10)
            return second.expire_due_subagent_results(checked_at_ms=5_100)

        with ThreadPoolExecutor(max_workers=2) as pool:
            acknowledgment = pool.submit(acknowledge)
            expiry = pool.submit(expire)
            ack_result = acknowledgment.result(timeout=10)
            expiry_result = expiry.result(timeout=10)
        final = first.get_subagent_task_ephemeral_status(task.task_id)
        if (
            ack_result.status != "EXPIRED"
            or len(expiry_result) not in {0, 1}
            or final.result_state != "expired"
        ):
            raise SystemExit(
                f"acknowledgment/expiry race was not terminally fenced: "
                f"{ack_result}, {expiry_result}, {final}"
            )


def test_invalid_inputs_fail_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-subagent-invalid-") as temp:
        store = _store(temp)
        digest = _digest("valid")
        invalid: list[tuple[str, Callable[[], object]]] = [
            ("boolean worker count", lambda: store.configure_subagent_workers(True)),
            ("oversized worker count", lambda: store.configure_subagent_workers(65)),
            ("negative configure timestamp", lambda: store.configure_subagent_workers(1, checked_at_ms=-1)),
            ("blank worker id", lambda: store.activate_subagent_worker(" ", "runtime")),
            ("invalid runtime id", lambda: store.activate_subagent_worker("subagent-01", "bad runtime")),
            (
                "invalid worker stale window",
                lambda: store.activate_subagent_worker("subagent-01", "runtime", stale_after_ms=0),
            ),
            ("invalid heartbeat generation", lambda: store.heartbeat_subagent_worker("subagent-01", "runtime", True)),
            ("invalid heartbeat timestamp", lambda: store.heartbeat_subagent_worker("subagent-01", "runtime", 1, checked_at_ms=True)),
            ("invalid deactivate timestamp", lambda: store.deactivate_subagent_worker("subagent-01", "runtime", 1, checked_at_ms=-1)),
            ("invalid request digest", lambda: store.enqueue_subagent_task(request_digest="raw", kind="mock")),
            ("blank task kind", lambda: store.enqueue_subagent_task(request_digest=digest, kind="")),
            ("boolean priority", lambda: store.enqueue_subagent_task(request_digest=digest, kind="mock", priority=True)),
            ("oversized priority", lambda: store.enqueue_subagent_task(request_digest=digest, kind="mock", priority=1001)),
            ("invalid replay policy", lambda: store.enqueue_subagent_task(request_digest=digest, kind="mock", replay_policy="always")),
            ("boolean max attempts", lambda: store.enqueue_subagent_task(request_digest=digest, kind="mock", max_attempts=True)),
            ("oversized max attempts", lambda: store.enqueue_subagent_task(request_digest=digest, kind="mock", max_attempts=11)),
            ("invalid enqueue timestamp", lambda: store.enqueue_subagent_task(request_digest=digest, kind="mock", checked_at_ms=True)),
            ("zero claim lease", lambda: store.claim_subagent_assignment("subagent-01", "runtime", 1, lease_ms=0)),
            ("invalid claim generation", lambda: store.claim_subagent_assignment("subagent-01", "runtime", True)),
            ("invalid claim stale window", lambda: store.claim_subagent_assignment("subagent-01", "runtime", 1, worker_stale_after_ms=0)),
            ("invalid claim timestamp", lambda: store.claim_subagent_assignment("subagent-01", "runtime", 1, checked_at_ms=True)),
            ("empty start token", lambda: store.start_subagent_assignment("assignment-1", "subagent-01", "runtime", 1, "")),
            ("boolean renewal lease", lambda: store.renew_subagent_assignment("assignment-1", "subagent-01", "runtime", 1, "token", lease_ms=True)),
            ("invalid outcome", lambda: store.finalize_subagent_assignment("assignment-1", "subagent-01", "runtime", 1, "token", outcome="uncertain")),
            ("success without digest", lambda: store.finalize_subagent_assignment("assignment-1", "subagent-01", "runtime", 1, "token", outcome="succeeded")),
            ("success without result type", lambda: store.finalize_subagent_assignment("assignment-1", "subagent-01", "runtime", 1, "token", outcome="succeeded", result_digest=digest)),
            ("failure without error code", lambda: store.finalize_subagent_assignment("assignment-1", "subagent-01", "runtime", 1, "token", outcome="failed")),
            ("invalid result digest", lambda: store.finalize_subagent_assignment("assignment-1", "subagent-01", "runtime", 1, "token", outcome="failed", result_digest="raw")),
            ("short result retention", lambda: store.finalize_subagent_assignment("assignment-1", "subagent-01", "runtime", 1, "token", outcome="failed", error_code="failed", result_retention_ms=99)),
            ("boolean result retention", lambda: store.finalize_subagent_assignment("assignment-1", "subagent-01", "runtime", 1, "token", outcome="failed", error_code="failed", result_retention_ms=True)),
            ("blank pre-start error", lambda: store.reject_subagent_assignment_before_start("assignment-1", "subagent-01", "runtime", 1, "token", error_code="")),
            ("oversized pre-start error", lambda: store.reject_subagent_assignment_before_start("assignment-1", "subagent-01", "runtime", 1, "token", error_code="x" * 81)),
            ("invalid status id", lambda: store.get_subagent_assignment_status("bad assignment")),
            ("invalid ephemeral task id", lambda: store.get_subagent_task_ephemeral_status("bad task")),
            ("invalid ephemeral request digest", lambda: store.get_subagent_task_ephemeral_status_by_request_digest("raw")),
            ("invalid expiry limit", lambda: store.expire_due_subagent_results(limit=0)),
            ("boolean release limit", lambda: store.list_subagent_result_releases(limit=True)),
            ("non-boolean running requeue", lambda: store.recover_expired_subagent_assignments(allow_running_requeue=1)),
            ("invalid snapshot timestamp", lambda: store.subagent_control_snapshot(checked_at_ms=-1)),
            ("invalid snapshot stale window", lambda: store.subagent_control_snapshot(worker_stale_after_ms=0)),
            ("boolean snapshot limit", lambda: store.subagent_control_snapshot(limit=True)),
        ]
        for label, operation in invalid:
            _expect_value_error(label, operation)
        with store.connect() as conn:
            rows = sum(
                conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("subagent_workers", "subagent_tasks", "subagent_assignments")
            )
        if rows != 0:
            raise SystemExit(f"invalid inputs left {rows} subagent rows")


def main() -> None:
    test_worker_lifecycle_identity_and_capacity()
    test_enqueue_idempotency_and_collision()
    test_concurrent_claims_across_store_instances()
    test_lease_start_renew_finalize_fencing_and_expiry_boundary()
    test_prestart_rejection_is_fenced_terminal_and_inspectable()
    test_claimed_vs_running_recovery()
    test_running_recovery_can_disable_requeue_for_ephemeral_custody()
    test_restart_persistence_and_snapshot_privacy_bounds()
    test_control_snapshot_runner_attachment_and_dispatch_readiness()
    test_finalization_contracts_are_checked_before_sqlite()
    test_ephemeral_result_schema_migration_and_validation()
    test_ephemeral_result_retention_state_machine()
    test_concurrent_acknowledgment_vs_expiry()
    test_invalid_inputs_fail_closed()
    print("Subagent assignment store smoke passed")


if __name__ == "__main__":
    main()
