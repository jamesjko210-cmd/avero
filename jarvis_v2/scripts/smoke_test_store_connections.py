"""Pin the MemoryStore connection lifecycle.

sqlite3's context manager only ends the transaction — it never closes the
connection. That leaked one fd per store call and, in the long-lived daemons,
exhausted launchd's 256-fd default (252 leaked handles observed live) so every
new open failed with "unable to open database file". That single bug caused
the missed Morning Brief and the recurring scheduler tick failures of
2026-07-04. These tests make sure it can never come back quietly.
"""

from __future__ import annotations

import gc
import json
import multiprocessing
import os
import queue
import sqlite3
import stat
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import BrokenBarrierError
from typing import Any

from jarvis_v2.memory.store import (
    MAX_FTS_QUERY_CHARS,
    GoalRecord,
    MemoryRecord,
    MemoryStore,
    PersonRecord,
    approval_action_digest,
)


def _open_connection_objects() -> int:
    gc.collect()
    return sum(
        1
        for obj in gc.get_objects()
        if isinstance(obj, sqlite3.Connection) and _connection_is_open(obj)
    )


def _connection_is_open(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("SELECT 1")
        return True
    except sqlite3.ProgrammingError:
        return False
    except Exception:
        return False


def _create_legacy_database(db_path: Path) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        with conn:
            conn.executescript(
                """
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE pending_approvals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE tool_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    risk TEXT NOT NULL,
                    ok INTEGER NOT NULL,
                    approved INTEGER NOT NULL,
                    output TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE execution_cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request TEXT NOT NULL,
                    mission_state TEXT NOT NULL,
                    go_no_go TEXT NOT NULL,
                    next_command TEXT NOT NULL,
                    risk_signals TEXT NOT NULL DEFAULT '[]',
                    note_path TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE scheduled_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    interval_minutes INTEGER NOT NULL,
                    job_type TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    last_run_at TEXT,
                    next_run_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE ingested_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_key TEXT NOT NULL UNIQUE,
                    source_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                INSERT INTO ingested_sources(source_key, source_type, title, created_at)
                VALUES ('historical-source', 'obsidian-inbox', 'Historical source', '2026-01-01T00:00:00Z');
                INSERT INTO tool_runs(session_id, tool_name, risk, ok, approved, output, created_at)
                VALUES ('legacy', 'legacy_tool', 'READ_ONLY', 1, 0, 'legacy output', '2026-01-01T00:00:00Z');
                """
            )


def _init_after_barrier(db_path: str, start_barrier: Any) -> None:
    try:
        start_barrier.wait(timeout=15.0)
    except BrokenBarrierError as exc:
        raise TimeoutError("concurrent init callers did not reach the start barrier") from exc
    MemoryStore(Path(db_path)).init()


def _process_init_worker(
    worker_id: int,
    db_path: str,
    start_barrier: Any,
    result_queue: Any,
) -> None:
    try:
        _init_after_barrier(db_path, start_barrier)
    except BaseException as exc:
        result_queue.put((worker_id, f"{type(exc).__name__}: {exc}"))
    else:
        result_queue.put((worker_id, ""))


def test_concurrent_legacy_init_is_serialized() -> None:
    thread_callers = 12
    process_callers = 4
    with TemporaryDirectory(prefix="jarvis-store-migration-race-") as temp:
        db_path = Path(temp) / "legacy.sqlite"
        _create_legacy_database(db_path)

        process_context = multiprocessing.get_context("spawn")
        start_barrier = process_context.Barrier(thread_callers + process_callers)
        result_queue = process_context.Queue()
        processes = [
            process_context.Process(
                target=_process_init_worker,
                args=(worker_id, str(db_path), start_barrier, result_queue),
            )
            for worker_id in range(process_callers)
        ]
        for process in processes:
            process.start()

        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=thread_callers) as pool:
            futures = [
                pool.submit(_init_after_barrier, str(db_path), start_barrier)
                for _ in range(thread_callers)
            ]
            for future in futures:
                try:
                    future.result(timeout=30.0)
                except BaseException as exc:
                    errors.append(f"thread caller failed: {type(exc).__name__}: {exc}")

        for process in processes:
            process.join(timeout=30.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
                errors.append(f"process caller {process.pid} did not finish")
            elif process.exitcode != 0:
                errors.append(f"process caller {process.pid} exited with {process.exitcode}")

        process_results: dict[int, str] = {}
        for _ in range(process_callers):
            try:
                worker_id, error = result_queue.get(timeout=2.0)
            except queue.Empty:
                break
            process_results[worker_id] = error
        result_queue.close()
        result_queue.join_thread()
        for worker_id in range(process_callers):
            if worker_id not in process_results:
                errors.append(f"process caller {worker_id} returned no result")
            elif process_results[worker_id]:
                errors.append(f"process caller {worker_id} failed: {process_results[worker_id]}")
        if errors:
            raise SystemExit("concurrent legacy init failed:\n" + "\n".join(errors))

        expected_columns = {
            "messages": {"metadata"},
            "pending_approvals": {"planned_args"},
            "approval_queue_state": {"id", "revision"},
            "tool_runs": {"approval_id", "metadata", "approval_action_digest"},
            "execution_cases": {"metadata"},
            "scheduled_jobs": {"metadata", "lease_token", "lease_expires_at", "schedule_revision"},
            "ingested_sources": {"memory_id", "mirror_state", "mirror_completed_at"},
            "conversation_compaction_state": {"singleton_id", "last_message_id", "updated_at"},
            "conversation_compaction_batches": {
                "batch_key",
                "first_message_id",
                "last_message_id",
                "outcome",
                "memory_id",
                "mirror_state",
            },
        }
        store = MemoryStore(db_path)
        with store.connect() as conn:
            for table, expected in expected_columns.items():
                actual = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
                missing = expected - actual
                if missing:
                    raise SystemExit(
                        f"concurrent init left {table} missing columns: {sorted(missing)}"
                    )
            historical_source = conn.execute(
                "SELECT mirror_state FROM ingested_sources WHERE source_key = 'historical-source'"
            ).fetchone()
            if historical_source is None or historical_source["mirror_state"] != "completed":
                raise SystemExit(
                    f"historical ingest sources must migrate as completed: {historical_source}"
                )
            queue_state = conn.execute(
                "SELECT id, revision FROM approval_queue_state"
            ).fetchall()
            if len(queue_state) != 1 or dict(queue_state[0]) != {"id": 1, "revision": 0}:
                raise SystemExit(
                    f"legacy init must create the singleton approval queue state: {queue_state}"
                )
            legacy_run = conn.execute(
                "SELECT approval_action_digest FROM tool_runs WHERE tool_name = 'legacy_tool'"
            ).fetchone()
            if legacy_run is None or legacy_run["approval_action_digest"] is not None:
                raise SystemExit(
                    f"legacy tool runs must remain unproven instead of being backfilled: {legacy_run}"
                )
            compaction_state = conn.execute(
                "SELECT singleton_id, last_message_id FROM conversation_compaction_state"
            ).fetchall()
            if len(compaction_state) != 1 or dict(compaction_state[0]) != {
                "singleton_id": 1,
                "last_message_id": 0,
            }:
                raise SystemExit(
                    f"legacy init must create one conservative compaction state row: {compaction_state}"
                )
            if conn.execute("SELECT COUNT(*) FROM conversation_compaction_batches").fetchone()[0] != 0:
                raise SystemExit("legacy init must not synthesize compaction range proof")


def test_migration_lock_sidecar_is_private_under_permissive_umask() -> None:
    with TemporaryDirectory(prefix="jarvis-store-migration-mode-") as temp:
        db_path = Path(temp) / "mode.sqlite"
        previous_umask = os.umask(0)
        try:
            MemoryStore(db_path).init()
        finally:
            os.umask(previous_umask)
        lock_path = Path(f"{db_path.resolve()}.migration.lock")
        mode = stat.S_IMODE(os.stat(lock_path, follow_symlinks=False).st_mode)
        if mode != 0o600:
            raise SystemExit(f"migration lock sidecar mode drifted under permissive umask: {mode:o}")


def test_migration_lock_sidecar_refuses_symlink_without_touching_target() -> None:
    with TemporaryDirectory(prefix="jarvis-store-migration-symlink-") as temp:
        root = Path(temp)
        db_path = root / "symlink.sqlite"
        lock_path = Path(f"{db_path.resolve()}.migration.lock")
        external_target = root / "external-target"
        original = b"external target must remain untouched\n"
        external_target.write_bytes(original)
        external_target.chmod(0o640)
        lock_path.symlink_to(external_target)
        try:
            MemoryStore(db_path).init()
        except RuntimeError:
            pass
        else:
            raise SystemExit("migration lock followed a symbolic-link sidecar")
        if external_target.read_bytes() != original:
            raise SystemExit("migration lock symlink handling modified the external target")
        mode = stat.S_IMODE(external_target.stat().st_mode)
        if mode != 0o640:
            raise SystemExit(
                f"migration lock symlink handling changed external target permissions: {mode:o}"
            )


def test_read_only_compatibility_requires_critical_schema_shapes() -> None:
    with TemporaryDirectory(prefix="jarvis-store-readonly-shape-") as temp:
        db_path = Path(temp) / "shape.sqlite"
        store = MemoryStore(db_path)
        store.init()

        def compatible() -> bool:
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute("PRAGMA query_only = ON")
                return MemoryStore._critical_read_only_schema_is_compatible(conn)

        if not compatible():
            raise SystemExit("fully initialized schema was not read-only compatible")
        with sqlite3.connect(db_path) as conn:
            conn.execute("DROP INDEX history_disclosure_receipts_prepared_idx")
        if compatible():
            raise SystemExit("read-only compatibility accepted a missing disclosure index")

        store.init()
        with sqlite3.connect(db_path) as conn:
            conn.execute("DROP TRIGGER history_disclosure_receipts_two_phase_update")
        if compatible():
            raise SystemExit("read-only compatibility accepted a missing disclosure fence trigger")

        store.init()
        with sqlite3.connect(db_path) as conn:
            conn.execute("ALTER TABLE skills RENAME TO hardened_skills")
            conn.execute(
                """
                CREATE TABLE skills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    trigger TEXT NOT NULL DEFAULT '',
                    body TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '',
                    success_count INTEGER NOT NULL DEFAULT 0,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    revision INTEGER NOT NULL DEFAULT 1,
                    review_status TEXT NOT NULL DEFAULT 'active',
                    origin TEXT NOT NULL DEFAULT 'legacy_unknown',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute("DROP TABLE hardened_skills")
        if compatible():
            raise SystemExit("read-only compatibility accepted missing skill custody constraints")


def test_approval_queue_revision_and_atomic_readiness() -> None:
    with TemporaryDirectory(prefix="jarvis-store-approval-queue-") as temp:
        store = MemoryStore(Path(temp) / "approval-queue.sqlite")
        store.init()

        def queue_revision() -> int:
            with store.connect() as conn:
                row = conn.execute(
                    "SELECT revision FROM approval_queue_state WHERE id = 1"
                ).fetchone()
            if row is None:
                raise SystemExit("approval queue singleton state is missing")
            return int(row["revision"])

        def reviewed_action_digest(row: sqlite3.Row) -> str:
            planned_args = json.loads(str(row["planned_args"] or "{}"))
            if not isinstance(planned_args, dict):
                raise SystemExit("approval fixture planned args are malformed")
            return approval_action_digest(str(row["tool_name"] or "").strip(), planned_args)

        if queue_revision() != 0:
            raise SystemExit("a new approval queue must start at revision zero")

        dismissed_id = store.add_pending_approval(
            "approval-smoke", "dismiss me", "shell", "dismissal revision check"
        )
        if queue_revision() != 1:
            raise SystemExit("enqueue must increment the approval queue revision")
        reviewed_id = store.add_pending_approval(
            "approval-smoke", "review me", "shell", "atomic readiness check"
        )
        if queue_revision() != 2:
            raise SystemExit("each enqueue must increment the approval queue revision")

        if not store.set_pending_approval_status(dismissed_id, "dismissed"):
            raise SystemExit("pending approval dismissal unexpectedly failed")
        if queue_revision() != 3:
            raise SystemExit("successful pending dismissal must increment queue revision")
        if store.set_pending_approval_status(dismissed_id, "dismissed"):
            raise SystemExit("repeated dismissal must not report a pending transition")
        if queue_revision() != 3:
            raise SystemExit("failed dismissal must not increment queue revision")

        reviewed = store.approval_readiness_snapshot(reviewed_id)
        pending_ids = [int(row["id"]) for row in reviewed["pending_rows"]]
        target = reviewed["target"]
        if (
            target is None
            or int(target["id"]) != reviewed_id
            or str(target["status"]).lower() != "pending"
            or pending_ids != [reviewed_id]
            or reviewed["pending_count"] != len(pending_ids)
            or reviewed["newest_pending_id"] != max(pending_ids)
            or reviewed["queue_revision"] != queue_revision()
        ):
            raise SystemExit(f"approval readiness snapshot is internally inconsistent: {reviewed}")

        reviewed_revision = int(reviewed["queue_revision"])
        reviewed_updated_at = str(target["updated_at"])
        reviewed_digest = reviewed_action_digest(target)
        if store.approve_pending_approval_if_ready(
            reviewed_id,
            expected_queue_revision=reviewed_revision + 1,
            expected_updated_at=reviewed_updated_at,
            expected_action_digest=reviewed_digest,
        ):
            raise SystemExit("atomic approval accepted the wrong reviewed queue revision")
        if store.approve_pending_approval_if_ready(
            reviewed_id,
            expected_queue_revision=reviewed_revision,
            expected_updated_at=reviewed_updated_at + "-stale",
            expected_action_digest=reviewed_digest,
        ):
            raise SystemExit("atomic approval accepted the wrong reviewed updated_at")
        if store.approve_pending_approval_if_ready(
            reviewed_id,
            expected_queue_revision=reviewed_revision,
            expected_updated_at=reviewed_updated_at,
            expected_action_digest="0" * 64,
        ):
            raise SystemExit("atomic approval accepted the wrong reviewed action digest")
        if queue_revision() != reviewed_revision:
            raise SystemExit("rejected atomic approvals must not increment queue revision")

        newest_id = store.add_pending_approval(
            "approval-smoke", "newest", "shell", "invalidate the prior review"
        )
        if queue_revision() != reviewed_revision + 1:
            raise SystemExit("newer enqueue must increment queue revision")
        if store.approve_pending_approval_if_ready(
            reviewed_id,
            expected_queue_revision=reviewed_revision,
            expected_updated_at=reviewed_updated_at,
            expected_action_digest=reviewed_digest,
        ):
            raise SystemExit("newer enqueue did not invalidate the old atomic approval review")
        if store.approve_pending_approval_if_ready(
            reviewed_id,
            expected_queue_revision=queue_revision(),
            expected_updated_at=reviewed_updated_at,
            expected_action_digest=reviewed_digest,
        ):
            raise SystemExit("atomic approval accepted a target that was not newest")

        newest = store.approval_readiness_snapshot(newest_id)
        newest_target = newest["target"]
        if newest_target is None:
            raise SystemExit("newest approval disappeared from readiness snapshot")
        revision_before_approval = int(newest["queue_revision"])
        newest_updated_at = str(newest_target["updated_at"])
        newest_digest = reviewed_action_digest(newest_target)
        original_planned_args = str(newest_target["planned_args"])
        with store.connect() as conn:
            conn.execute(
                "UPDATE pending_approvals SET planned_args = ? WHERE id = ?",
                (json.dumps({"command": "tampered after review"}, sort_keys=True), newest_id),
            )
        if queue_revision() != revision_before_approval:
            raise SystemExit("out-of-band planned-args tamper unexpectedly changed queue revision")
        tampered = store.get_approval(newest_id)
        if tampered is None or str(tampered["updated_at"]) != newest_updated_at:
            raise SystemExit("planned-args tamper fixture changed the reviewed timestamp")
        if store.approve_pending_approval_if_ready(
            newest_id,
            expected_queue_revision=revision_before_approval,
            expected_updated_at=newest_updated_at,
            expected_action_digest=newest_digest,
        ):
            raise SystemExit("atomic approval accepted unchanged-timestamp planned-args tampering")
        if queue_revision() != revision_before_approval:
            raise SystemExit("rejected action-digest mismatch changed queue revision")
        with store.connect() as conn:
            conn.execute(
                "UPDATE pending_approvals SET planned_args = ? WHERE id = ?",
                (original_planned_args, newest_id),
            )
        if not store.approve_pending_approval_if_ready(
            newest_id,
            expected_queue_revision=revision_before_approval,
            expected_updated_at=newest_updated_at,
            expected_action_digest=newest_digest,
        ):
            raise SystemExit("exact reviewed atomic approval unexpectedly failed")
        if queue_revision() != revision_before_approval + 1:
            raise SystemExit("successful atomic approval must increment queue revision")
        approved = store.get_approval(newest_id)
        if approved is None or str(approved["status"]).lower() != "approved":
            raise SystemExit("successful atomic approval did not persist approved status")


def test_with_block_closes_the_connection() -> None:
    with TemporaryDirectory(prefix="jarvis-store-conn-") as temp:
        store = MemoryStore(Path(temp) / "conn.sqlite")
        store.init()
        with store.connect() as conn:
            conn.execute("SELECT 1")
        try:
            conn.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            return
        raise SystemExit("connection must be CLOSED after the with block, not just committed")


def test_repeated_store_calls_do_not_accumulate_connections() -> None:
    with TemporaryDirectory(prefix="jarvis-store-leak-") as temp:
        store = MemoryStore(Path(temp) / "leak.sqlite")
        store.init()
        baseline = _open_connection_objects()
        for index in range(60):
            store.list_jobs()
            store.recent_memories(limit=2)
            if index % 20 == 0:
                store.add_memory(MemoryRecord(category="smoke", title=f"t{index}", body="b"))
        open_now = _open_connection_objects()
        if open_now > baseline + 1:
            raise SystemExit(
                f"store calls are leaking connections again: baseline {baseline}, now {open_now}"
            )


def test_commit_on_success_and_rollback_on_error_survive() -> None:
    with TemporaryDirectory(prefix="jarvis-store-txn-") as temp:
        store = MemoryStore(Path(temp) / "txn.sqlite")
        store.init()
        row_id = store.add_memory(MemoryRecord(category="smoke", title="commit check", body="b"))
        if store.get_memory(row_id) is None:
            raise SystemExit("auto-commit on with-exit must survive the close-on-exit change")
        try:
            with store.connect() as conn:
                conn.execute(
                    "INSERT INTO memories(category, title, body, source, confidence, created_at, updated_at)"
                    " VALUES ('smoke', 'rollback check', 'b', 'manual', 1.0, 'x', 'x')"
                )
                raise RuntimeError("force rollback")
        except RuntimeError:
            pass
        with store.connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE title = 'rollback check'"
            ).fetchone()[0]
        if count != 0:
            raise SystemExit("rollback-on-error must survive the close-on-exit change")


def test_foreign_keys_are_enforced_and_cascades_work() -> None:
    with TemporaryDirectory(prefix="jarvis-store-fk-") as temp:
        store = MemoryStore(Path(temp) / "foreign-keys.sqlite")
        store.init()
        with store.connect() as conn:
            enabled = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        if enabled != 1:
            raise SystemExit("every MemoryStore connection must enable SQLite foreign keys")

        if store.add_goal_step(999_001, "orphan goal step") is not None:
            raise SystemExit("MemoryStore accepted a goal step whose parent does not exist")
        with store.connect() as conn:
            orphan_goal_steps = conn.execute(
                "SELECT COUNT(*) FROM goal_steps WHERE goal_id = ?",
                (999_001,),
            ).fetchone()[0]
        if orphan_goal_steps != 0:
            raise SystemExit("missing-parent goal-step refusal wrote a child row")

        invalid_children = [
            lambda: store.log_person_interaction(999_002, "orphan interaction"),
            lambda: store.add_execution_case_event(999_003, "invalid", "orphan event"),
        ]
        for create_invalid_child in invalid_children:
            try:
                create_invalid_child()
            except sqlite3.IntegrityError:
                continue
            raise SystemExit("MemoryStore accepted a child row whose parent does not exist")

        goal_id = store.create_goal(GoalRecord("cascade goal"))
        store.add_goal_step(goal_id, "cascade step")
        person_id = store.upsert_person(PersonRecord("Cascade Person"))
        store.log_person_interaction(person_id, "cascade interaction")
        with store.connect() as conn:
            conn.execute("DELETE FROM goals WHERE id = ?", (goal_id,))
            conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
        if store.list_goal_steps(goal_id):
            raise SystemExit("deleting a goal must cascade to its goal steps")
        if store.list_person_interactions(person_id):
            raise SystemExit("deleting a person must cascade to their interactions")


def test_malformed_fts_queries_fail_closed_to_literal_search() -> None:
    with TemporaryDirectory(prefix="jarvis-store-fts-") as temp:
        store = MemoryStore(Path(temp) / "fts.sqlite")
        store.init()
        store.add_memory(MemoryRecord(category="smoke", title="alpha beta", body="literal search"))
        store.log_message("fts-smoke", "user", "alpha beta literal message")
        if not store.search_memories("alpha beta") or not store.search_messages("alpha beta"):
            raise SystemExit("valid natural-language FTS search regressed")
        for malformed in ('"', "alpha OR", "-", "title:"):
            memory_rows = store.search_memories(malformed)
            message_rows = store.search_messages(malformed)
            if not isinstance(memory_rows, list) or not isinstance(message_rows, list):
                raise SystemExit(f"malformed FTS query did not return bounded lists: {malformed!r}")


def test_long_valid_fts_queries_are_bounded_before_match() -> None:
    with TemporaryDirectory(prefix="jarvis-store-fts-long-") as temp:
        store = MemoryStore(Path(temp) / "fts-long.sqlite")
        store.init()
        store.add_memory(MemoryRecord(category="smoke", title="a", body="long valid query"))
        store.log_message("fts-long", "user", "a long valid query")
        long_valid_query = "a " * (MAX_FTS_QUERY_CHARS + 1)
        if not store.search_memories(long_valid_query):
            raise SystemExit("long valid memory FTS query was not bounded before MATCH")
        if not store.search_messages(long_valid_query):
            raise SystemExit("long valid message FTS query was not bounded before MATCH")


def test_equal_rank_fts_results_have_stable_newest_first_ties() -> None:
    with TemporaryDirectory(prefix="jarvis-store-fts-ties-") as temp:
        db_path = Path(temp) / "fts-ties.sqlite"
        store = MemoryStore(db_path)
        store.init()
        older_memory_id = store.add_memory(
            MemoryRecord(category="smoke", title="deterministic tie", body="equal rank token")
        )
        newer_memory_id = store.add_memory(
            MemoryRecord(category="smoke", title="deterministic tie", body="equal rank token")
        )
        older_message_id = store.log_message(
            "fts-tie", "user", "deterministic equal rank message token"
        )
        newer_message_id = store.log_message(
            "fts-tie", "user", "deterministic equal rank message token"
        )

        expected_memory_ids = [newer_memory_id, older_memory_id]
        expected_message_ids = [newer_message_id, older_message_id]
        memory_ids = [
            int(row["id"])
            for row in store.search_memories("equal rank token", limit=2)
        ]
        message_ids = [
            int(row["id"])
            for row in store.search_messages("equal rank message token", limit=2)
        ]
        if memory_ids != expected_memory_ids:
            raise SystemExit("equal-rank memory search results are not newest-first")
        if message_ids != expected_message_ids:
            raise SystemExit("equal-rank message search results are not newest-first")
        if int(store.search_memories("equal rank token", limit=1)[0]["id"]) != newer_memory_id:
            raise SystemExit("bounded memory search did not select the newest equal-rank row")
        if int(store.search_messages("equal rank message token", limit=1)[0]["id"]) != newer_message_id:
            raise SystemExit("bounded message search did not select the newest equal-rank row")

        reopened = MemoryStore(db_path)
        reopened.init()
        reopened_memory_ids = [
            int(row["id"])
            for row in reopened.search_memories("equal rank token", limit=2)
        ]
        reopened_message_ids = [
            int(row["id"])
            for row in reopened.search_messages("equal rank message token", limit=2)
        ]
        if reopened_memory_ids != expected_memory_ids:
            raise SystemExit("equal-rank memory ordering changed after reopening the store")
        if reopened_message_ids != expected_message_ids:
            raise SystemExit("equal-rank message ordering changed after reopening the store")


def test_compaction_state_migrates_valid_legacy_job_watermark_once() -> None:
    with TemporaryDirectory(prefix="jarvis-store-compaction-migration-") as temp:
        store = MemoryStore(Path(temp) / "compaction-migration.sqlite")
        store.init()
        job_id = store.upsert_job(
            "Legacy Conversation Compaction",
            1440,
            "conversation_compaction",
            "2026-01-01T00:00:00",
        )
        if not store.merge_job_metadata(job_id, {"last_compacted_id": 77}):
            raise SystemExit("could not seed legacy compaction watermark")
        with store.connect() as conn:
            conn.execute("DROP TABLE conversation_compaction_batches")
            conn.execute("DROP TABLE conversation_compaction_state")
        store.init()
        if store.conversation_compaction_watermark() != 77:
            raise SystemExit("valid legacy compaction watermark was not adopted conservatively")
        with store.connect() as conn:
            if conn.execute("SELECT COUNT(*) FROM conversation_compaction_batches").fetchone()[0] != 0:
                raise SystemExit("legacy watermark migration synthesized range evidence")
        if not store.merge_job_metadata(job_id, {"last_compacted_id": 999}):
            raise SystemExit("could not mutate compatibility watermark after state migration")
        store.init()
        if store.conversation_compaction_watermark() != 77:
            raise SystemExit("compatibility metadata overwrote initialized authoritative compaction state")


def main() -> None:
    test_concurrent_legacy_init_is_serialized()
    test_migration_lock_sidecar_is_private_under_permissive_umask()
    test_migration_lock_sidecar_refuses_symlink_without_touching_target()
    test_read_only_compatibility_requires_critical_schema_shapes()
    test_approval_queue_revision_and_atomic_readiness()
    test_with_block_closes_the_connection()
    test_repeated_store_calls_do_not_accumulate_connections()
    test_commit_on_success_and_rollback_on_error_survive()
    test_foreign_keys_are_enforced_and_cascades_work()
    test_malformed_fts_queries_fail_closed_to_literal_search()
    test_long_valid_fts_queries_are_bounded_before_match()
    test_equal_rank_fts_results_have_stable_newest_first_ties()
    test_compaction_state_migrates_valid_legacy_job_watermark_once()
    print("Store connection smoke passed")


if __name__ == "__main__":
    main()
