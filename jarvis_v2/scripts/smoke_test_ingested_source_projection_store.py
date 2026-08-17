from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.memory.store import (
    MemoryRecord,
    MemoryStore,
    ScheduledJobLeaseAuthorityLost,
)


CONTENT_DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64


def _store(root: Path, name: str) -> MemoryStore:
    store = MemoryStore(root / f"{name}.sqlite")
    store.init()
    return store


def _record(title: str = "Inbox custody") -> MemoryRecord:
    return MemoryRecord(
        "inbox",
        title,
        "exact inbox projection body",
        "obsidian-inbox",
        0.9,
    )


def _source_row(store: MemoryStore, source_key: str) -> sqlite3.Row | None:
    with store.connect() as conn:
        return conn.execute(
            "SELECT * FROM ingested_sources WHERE source_key = ?", (source_key,)
        ).fetchone()


def _expect_runtime_error(call, label: str) -> None:
    try:
        call()
    except RuntimeError:
        return
    raise SystemExit(f"{label} did not fail closed")


def _obsidian_source_key(body: str) -> str:
    return "obsidian-inbox:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def test_real_legacy_upgrade_fk_and_exact_memory_adoption() -> None:
    with TemporaryDirectory(prefix="jarvis-ingest-projection-legacy-upgrade-") as temp:
        db_path = Path(temp) / "legacy.sqlite"
        record = _record("Legacy upgrade target")
        source_key = _obsidian_source_key(record.body)
        created_at = "2026-07-10T01:02:03Z"
        with sqlite3.connect(db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE memories_fts
                USING fts5(title, body, category, content='memories', content_rowid='id');
                CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, title, body, category)
                    VALUES (new.id, new.title, new.body, new.category);
                END;
                CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, title, body, category)
                    VALUES ('delete', old.id, old.title, old.body, old.category);
                END;
                CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, title, body, category)
                    VALUES ('delete', old.id, old.title, old.body, old.category);
                    INSERT INTO memories_fts(rowid, title, body, category)
                    VALUES (new.id, new.title, new.body, new.category);
                END;
                CREATE TABLE ingested_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_key TEXT NOT NULL UNIQUE,
                    source_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                INSERT INTO memories(
                    id, category, title, body, source, confidence, created_at, updated_at
                ) VALUES (41, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.category,
                    record.title,
                    record.body,
                    record.source,
                    record.confidence,
                    created_at,
                    created_at,
                ),
            )
            conn.execute(
                """
                INSERT INTO ingested_sources(
                    id, source_key, source_type, title, created_at
                ) VALUES (73, ?, 'obsidian-inbox', ?, ?)
                """,
                (source_key, record.title, "2026-07-13T00:00:00.000009+00:00"),
            )

        store = MemoryStore(db_path)
        store.init()
        fresh = _store(Path(temp), "fresh-fk-shape")
        with store.connect() as conn, fresh.connect() as fresh_conn:
            legacy_fk = [
                (row["from"], row["table"], row["to"], row["on_delete"])
                for row in conn.execute("PRAGMA foreign_key_list(ingested_sources)")
            ]
            fresh_fk = [
                (row["from"], row["table"], row["to"], row["on_delete"])
                for row in fresh_conn.execute("PRAGMA foreign_key_list(ingested_sources)")
            ]
            source = conn.execute(
                "SELECT * FROM ingested_sources WHERE source_key = ?", (source_key,)
            ).fetchone()
        expected_fk = [("memory_id", "memories", "id", "SET NULL")]
        if legacy_fk != expected_fk or legacy_fk != fresh_fk:
            raise SystemExit(
                f"legacy ingested_sources FK does not match fresh schema: {legacy_fk}"
            )
        if source is None or source["id"] != 73 or source["memory_id"] is not None:
            raise SystemExit("legacy source rebuild did not preserve its exact row identity")

        inserted, target = store.add_memory_if_source_new_with_projection(
            record, source_key, "obsidian-inbox"
        )
        with store.connect() as conn:
            counts = (
                conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
                conn.execute("SELECT COUNT(*) FROM ingested_sources").fetchone()[0],
                conn.execute("SELECT COUNT(*) FROM memory_projection_jobs").fetchone()[0],
            )
            adopted_source = conn.execute(
                "SELECT * FROM ingested_sources WHERE source_key = ?", (source_key,)
            ).fetchone()
        if (
            inserted
            or target.memory_id != 41
            or adopted_source is None
            or adopted_source["id"] != 73
            or adopted_source["memory_id"] != 41
            or counts != (1, 1, 1)
        ):
            raise SystemExit("real legacy marker did not adopt its one exact memory")

        with store.connect() as conn:
            conn.execute("DELETE FROM memories WHERE id = 41")
            deleted_link = conn.execute(
                "SELECT id, memory_id FROM ingested_sources WHERE source_key = ?",
                (source_key,),
            ).fetchone()
        if (
            deleted_link is None
            or deleted_link["id"] != 73
            or deleted_link["memory_id"] is not None
        ):
            raise SystemExit("migrated ingested source FK did not apply ON DELETE SET NULL")


def test_legacy_memory_adoption_ambiguity_fails_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-ingest-projection-legacy-ambiguity-") as temp:
        store = _store(Path(temp), "ambiguity")
        record = _record("Ambiguous legacy target")
        source_key = _obsidian_source_key(record.body)
        created_at = "2026-07-10T04:05:06Z"
        with store.connect() as conn:
            for memory_id in (101, 102):
                conn.execute(
                    """
                    INSERT INTO memories(
                        id, category, title, body, source, confidence, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        record.category,
                        record.title,
                        record.body,
                        record.source,
                        record.confidence,
                        created_at,
                        created_at,
                    ),
                )
            conn.execute(
                """
                INSERT INTO ingested_sources(
                    source_key, source_type, title, created_at
                ) VALUES (?, 'obsidian-inbox', ?, ?)
                """,
                (source_key, record.title, created_at),
            )
        _expect_runtime_error(
            lambda: store.add_memory_if_source_new_with_projection(
                record, source_key, "obsidian-inbox"
            ),
            "ambiguous legacy memory adoption",
        )
        with store.connect() as conn:
            source = conn.execute(
                "SELECT memory_id, memory_link_required, mirror_state "
                "FROM ingested_sources WHERE source_key = ?",
                (source_key,),
            ).fetchone()
            counts = (
                conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
                conn.execute("SELECT COUNT(*) FROM memory_projection_jobs").fetchone()[0],
            )
        if (
            source is None
            or source["memory_id"] is not None
            or source["memory_link_required"] != 0
            or source["mirror_state"] != "completed"
            or counts != (2, 0)
        ):
            raise SystemExit("ambiguous legacy adoption changed source or projection custody")


def test_atomic_reservation_rollback_and_source_repair() -> None:
    with TemporaryDirectory(prefix="jarvis-ingest-projection-atomic-") as temp:
        root = Path(temp)
        rollback_store = _store(root, "new-rollback")
        with rollback_store.connect() as conn:
            conn.execute(
                """
                CREATE TRIGGER reject_ingest_projection_job
                BEFORE INSERT ON memory_projection_jobs
                BEGIN
                    SELECT RAISE(ABORT, 'injected source projection failure');
                END
                """
            )
        try:
            rollback_store.add_memory_if_source_new_with_projection(
                _record(), "atomic-new", "obsidian-inbox"
            )
        except sqlite3.DatabaseError:
            pass
        else:
            raise SystemExit("projection failure committed a new source acquisition")
        with rollback_store.connect() as conn:
            counts = tuple(
                conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("memories", "ingested_sources", "memory_projection_jobs")
            )
        if counts != (0, 0, 0):
            raise SystemExit(f"projection rollback leaked source custody rows: {counts}")

        repair_store = _store(root, "repair-rollback")
        inserted, old_memory_id = repair_store.add_memory_if_source_new(
            _record("Repair target"), "repair-source", "obsidian-inbox"
        )
        if not inserted or old_memory_id is None:
            raise SystemExit("could not seed source-linked repair fixture")
        with repair_store.connect() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (old_memory_id,))
            conn.execute(
                """
                CREATE TRIGGER reject_repair_projection_job
                BEFORE INSERT ON memory_projection_jobs
                BEGIN
                    SELECT RAISE(ABORT, 'injected repaired projection failure');
                END
                """
            )
        try:
            repair_store.add_memory_if_source_new_with_projection(
                _record("Repair target"), "repair-source", "obsidian-inbox"
            )
        except sqlite3.DatabaseError:
            pass
        else:
            raise SystemExit("projection failure committed a replacement memory")
        row = _source_row(repair_store, "repair-source")
        with repair_store.connect() as conn:
            memory_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            job_count = conn.execute(
                "SELECT COUNT(*) FROM memory_projection_jobs"
            ).fetchone()[0]
            conn.execute("DROP TRIGGER reject_repair_projection_job")
        if row is None or row["memory_id"] is not None or memory_count or job_count:
            raise SystemExit("failed repair changed the source-to-memory identity")

        repaired, target = repair_store.add_memory_if_source_new_with_projection(
            _record("Repair target"), "repair-source", "obsidian-inbox"
        )
        repaired_row = _source_row(repair_store, "repair-source")
        with repair_store.connect() as conn:
            source_count = conn.execute(
                "SELECT COUNT(*) FROM ingested_sources WHERE source_key = 'repair-source'"
            ).fetchone()[0]
            memory_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if (
            not repaired
            or repaired_row is None
            or repaired_row["memory_id"] != target.memory_id
            or source_count != 1
            or memory_count != 1
        ):
            raise SystemExit("missing-memory repair did not converge to one source and memory")

        unlinked_store = _store(root, "unlinked-repair")
        unlinked_store.mark_ingested_source(
            "historical-unlinked", "obsidian-inbox", "Historical marker"
        )
        repaired, unlinked_target = unlinked_store.add_memory_if_source_new_with_projection(
            _record("Historical marker"), "historical-unlinked", "obsidian-inbox"
        )
        unlinked_row = _source_row(unlinked_store, "historical-unlinked")
        with unlinked_store.connect() as conn:
            counts = (
                conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
                conn.execute("SELECT COUNT(*) FROM ingested_sources").fetchone()[0],
                conn.execute("SELECT COUNT(*) FROM memory_projection_jobs").fetchone()[0],
            )
        if (
            not repaired
            or unlinked_row is None
            or unlinked_row["memory_id"] != unlinked_target.memory_id
            or unlinked_row["memory_link_required"] != 1
            or unlinked_row["mirror_state"] != "pending"
            or counts != (1, 1, 1)
        ):
            raise SystemExit("historical unlinked source did not gain singular projection custody")


def test_duplicate_legacy_adoption_and_job_identity() -> None:
    with TemporaryDirectory(prefix="jarvis-ingest-projection-adopt-") as temp:
        store = _store(Path(temp), "adoption")
        record = _record("Legacy inbox title")
        inserted, memory_id = store.add_memory_if_source_new(
            record, "legacy-source", "obsidian-inbox"
        )
        if not inserted or memory_id is None or store.get_memory_projection_job(memory_id):
            raise SystemExit("legacy adoption fixture unexpectedly had projection custody")

        adopted, target = store.add_memory_if_source_new_with_projection(
            record, "legacy-source", "obsidian-inbox"
        )
        job = store.get_memory_projection_job(memory_id)
        if (
            adopted
            or target.memory_id != memory_id
            or job is None
            or job["legacy_category"] != record.category
            or job["legacy_title"] != record.title
        ):
            raise SystemExit("legacy source adoption lost identity or cleanup locator")

        duplicate, duplicate_target = store.add_memory_if_source_new_with_projection(
            MemoryRecord("changed", "changed", "changed", "changed", 0.1),
            "legacy-source",
            "obsidian-inbox",
        )
        preserved = store.get_memory(memory_id)
        if (
            duplicate
            or duplicate_target != target
            or preserved is None
            or preserved["category"] != record.category
            or preserved["title"] != record.title
            or preserved["body"] != record.body
        ):
            raise SystemExit("duplicate source rewrote its established memory identity")

        _expect_runtime_error(
            lambda: store.add_memory_if_source_new_with_projection(
                record, "legacy-source", "different-source-type"
            ),
            "source type mismatch",
        )
        if not store.complete_memory_projection_publish(
            target.memory_id,
            target.revision,
            target.source_digest,
            "Memory Tree/Records/adopted.md",
            CONTENT_DIGEST,
        ):
            raise SystemExit("could not complete adopted source projection")
        completed_duplicate, completed_target = store.add_memory_if_source_new_with_projection(
            record, "legacy-source", "obsidian-inbox"
        )
        completed_job = store.get_memory_projection_job(memory_id)
        with store.connect() as conn:
            counts = (
                conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
                conn.execute("SELECT COUNT(*) FROM ingested_sources").fetchone()[0],
                conn.execute("SELECT COUNT(*) FROM memory_projection_jobs").fetchone()[0],
            )
        if (
            completed_duplicate
            or completed_target != target
            or completed_job is None
            or completed_job["state"] != "completed"
            or completed_job["content_digest"] != CONTENT_DIGEST
            or counts != (1, 1, 1)
        ):
            raise SystemExit("exact duplicate reopened or duplicated completed projection custody")


def test_lease_fence_and_atomic_projection_claim() -> None:
    with TemporaryDirectory(prefix="jarvis-ingest-projection-lease-") as temp:
        store = _store(Path(temp), "lease")
        job_id = store.upsert_job(
            "Inbox projection lease", 5, "obsidian_inbox", datetime.now().isoformat()
        )
        lease_token = "exact-live-inbox-lease"
        if store.claim_job(job_id, lease_token, 60.0) is None:
            raise SystemExit("could not seed live scheduled job claim")

        inserted, target = store.add_memory_if_source_new_for_job_claim_with_projection(
            _record("Leased source"),
            "leased-source",
            "obsidian-inbox",
            job_id=job_id,
            lease_token=lease_token,
        )
        if not inserted or store.get_memory_projection_job(target.memory_id) is None:
            raise SystemExit("live lease did not atomically acquire projection custody")

        try:
            store.add_memory_if_source_new_for_job_claim_with_projection(
                _record("Lost lease"),
                "lost-lease-source",
                "obsidian-inbox",
                job_id=job_id,
                lease_token="wrong-token",
            )
        except ScheduledJobLeaseAuthorityLost:
            pass
        else:
            raise SystemExit("wrong lease token acquired source projection custody")
        if _source_row(store, "lost-lease-source") is not None:
            raise SystemExit("lease loss left a source ledger row")

        with store.connect() as conn:
            conn.execute(
                """
                CREATE TRIGGER reject_leased_projection_job
                BEFORE INSERT ON memory_projection_jobs
                BEGIN
                    SELECT RAISE(ABORT, 'injected leased projection failure');
                END
                """
            )
        try:
            store.add_memory_if_source_new_for_job_claim_with_projection(
                _record("Leased rollback"),
                "leased-rollback-source",
                "obsidian-inbox",
                job_id=job_id,
                lease_token=lease_token,
            )
        except sqlite3.DatabaseError:
            pass
        else:
            raise SystemExit("leased projection failure did not roll back source acquisition")
        if _source_row(store, "leased-rollback-source") is not None:
            raise SystemExit("leased projection rollback left source custody behind")
        with store.connect() as conn:
            conn.execute("DROP TRIGGER reject_leased_projection_job")

        if not store.complete_memory_projection_publish(
            target.memory_id,
            target.revision,
            target.source_digest,
            "Memory Tree/Records/leased-completion.md",
            CONTENT_DIGEST,
        ):
            raise SystemExit("could not seed live-lease completion evidence")
        evidence = store.complete_ingested_source_projection_for_job_claim(
            "leased-source",
            target.memory_id,
            target.revision,
            target.source_digest,
            job_id=job_id,
            lease_token=lease_token,
        )
        if (
            not evidence["completed_now"]
            or _source_row(store, "leased-source")["mirror_state"] != "completed"
        ):
            raise SystemExit("live finalization lease did not complete exact source custody")

        _inserted, expired_target = (
            store.add_memory_if_source_new_for_job_claim_with_projection(
                _record("Expired finalization lease"),
                "expired-finalization-source",
                "obsidian-inbox",
                job_id=job_id,
                lease_token=lease_token,
            )
        )
        if not store.complete_memory_projection_publish(
            expired_target.memory_id,
            expired_target.revision,
            expired_target.source_digest,
            "Memory Tree/Records/expired-completion.md",
            OTHER_DIGEST,
        ):
            raise SystemExit("could not seed expired-lease completion evidence")
        expired_at = (datetime.now() - timedelta(seconds=1)).isoformat(
            timespec="microseconds"
        )
        with store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_jobs SET lease_expires_at = ? WHERE id = ?",
                (expired_at, job_id),
            )
        try:
            store.complete_ingested_source_projection_for_job_claim(
                "expired-finalization-source",
                expired_target.memory_id,
                expired_target.revision,
                expired_target.source_digest,
                job_id=job_id,
                lease_token=lease_token,
            )
        except ScheduledJobLeaseAuthorityLost:
            pass
        else:
            raise SystemExit("expired finalization lease completed source custody")
        if _source_row(store, "expired-finalization-source")["mirror_state"] != "pending":
            raise SystemExit("expired finalization lease changed the source ledger")


def test_exact_completion_reopen_and_pending_fences() -> None:
    with TemporaryDirectory(prefix="jarvis-ingest-projection-complete-") as temp:
        store = _store(Path(temp), "completion")
        source_key = "completion-source"
        inserted, target = store.add_memory_if_source_new_with_projection(
            _record("Completion target"), source_key, "obsidian-inbox"
        )
        if not inserted:
            raise SystemExit("could not seed source completion fixture")

        _expect_runtime_error(
            lambda: store.complete_ingested_source_projection(
                source_key, target.memory_id, target.revision, target.source_digest
            ),
            "pending projection completion",
        )
        if _source_row(store, source_key)["mirror_state"] != "pending":
            raise SystemExit("mismatched projection evidence completed the source ledger")

        path_display = "Memory Tree/Records/completed.md"
        if not store.complete_memory_projection_publish(
            target.memory_id,
            target.revision,
            target.source_digest,
            path_display,
            CONTENT_DIGEST,
        ):
            raise SystemExit("could not seed exact completed publish evidence")
        for call, label in (
            (
                lambda: store.complete_ingested_source_projection(
                    source_key, target.memory_id, target.revision + 1, target.source_digest
                ),
                "revision mismatch completion",
            ),
            (
                lambda: store.complete_ingested_source_projection(
                    source_key, target.memory_id, target.revision, OTHER_DIGEST
                ),
                "source digest mismatch completion",
            ),
            (
                lambda: store.complete_ingested_source_projection(
                    source_key, target.memory_id + 1, target.revision, target.source_digest
                ),
                "memory identity mismatch completion",
            ),
        ):
            _expect_runtime_error(call, label)
        if _source_row(store, source_key)["mirror_state"] != "pending":
            raise SystemExit("completed-job identity mismatch changed the source ledger")
        evidence = store.complete_ingested_source_projection(
            source_key, target.memory_id, target.revision, target.source_digest
        )
        expected_evidence = {
            "completed_now": True,
            "memory_id": target.memory_id,
            "memory_revision": target.revision,
            "source_digest": target.source_digest,
            "canonical_path_display": path_display,
            "content_digest": CONTENT_DIGEST,
        }
        if evidence != expected_evidence:
            raise SystemExit(f"source completion returned unbounded or wrong evidence: {evidence}")
        replay = store.complete_ingested_source_projection(
            source_key, target.memory_id, target.revision, target.source_digest
        )
        if replay != {**expected_evidence, "completed_now": False}:
            raise SystemExit(f"exact source completion replay was not idempotent: {replay}")

        for reopened, label in (
            (
                store.reopen_completed_memory_projection(
                    target.memory_id,
                    target.revision,
                    target.source_digest,
                    OTHER_DIGEST,
                ),
                "content digest",
            ),
            (
                store.reopen_completed_memory_projection(
                    target.memory_id,
                    target.revision + 1,
                    target.source_digest,
                    CONTENT_DIGEST,
                ),
                "revision",
            ),
            (
                store.reopen_completed_memory_projection(
                    target.memory_id,
                    target.revision,
                    OTHER_DIGEST,
                    CONTENT_DIGEST,
                ),
                "source digest",
            ),
        ):
            if reopened:
                raise SystemExit(f"wrong {label} reopened completed projection custody")
        before = dict(store.get_memory(target.memory_id))
        if not store.reopen_completed_memory_projection(
            target.memory_id, target.revision, target.source_digest, CONTENT_DIGEST
        ):
            raise SystemExit("exact completed projection evidence did not reopen")
        job = store.get_memory_projection_job(target.memory_id)
        if (
            job is None
            or job["state"] != "pending"
            or job["content_digest"] is not None
            or job["prior_content_digest"] != CONTENT_DIGEST
            or job["last_error_code"] != "completed_projection_invalid"
        ):
            raise SystemExit("reopened projection did not retain exact prior digest")
        if not store.mark_ingested_source_projection_pending(
            source_key,
            target.memory_id,
            target.revision,
            target.source_digest,
        ):
            raise SystemExit("exact source ledger row did not reopen to pending")
        after = dict(store.get_memory(target.memory_id))
        source = _source_row(store, source_key)
        if (
            before != after
            or source is None
            or source["mirror_state"] != "pending"
            or source["mirror_completed_at"] is not None
        ):
            raise SystemExit("source pending transition changed memory or lost ledger custody")
        if store.mark_ingested_source_projection_pending(
            source_key,
            target.memory_id + 1,
            target.revision,
            target.source_digest,
        ):
            raise SystemExit("wrong memory identity reopened the source ledger")
        _expect_runtime_error(
            lambda: store.complete_ingested_source_projection(
                source_key, target.memory_id, target.revision, target.source_digest
            ),
            "reopened pending job completion",
        )

        stale_key = "stale-current-source"
        _inserted, stale_target = store.add_memory_if_source_new_with_projection(
            _record("Stale current row"), stale_key, "obsidian-inbox"
        )
        if not store.complete_memory_projection_publish(
            stale_target.memory_id,
            stale_target.revision,
            stale_target.source_digest,
            "Memory Tree/Records/stale.md",
            CONTENT_DIGEST,
        ):
            raise SystemExit("could not seed stale-current completed projection")
        with store.connect() as conn:
            conn.execute(
                "UPDATE memories SET body = body || ' changed', revision = revision + 1 "
                "WHERE id = ?",
                (stale_target.memory_id,),
            )
        if store.reopen_completed_memory_projection(
            stale_target.memory_id,
            stale_target.revision,
            stale_target.source_digest,
            CONTENT_DIGEST,
        ):
            raise SystemExit("stale completed job reopened after current memory revision moved")
        _expect_runtime_error(
            lambda: store.complete_ingested_source_projection(
                stale_key,
                stale_target.memory_id,
                stale_target.revision,
                stale_target.source_digest,
            ),
            "stale current-row source completion",
        )


def main() -> None:
    test_real_legacy_upgrade_fk_and_exact_memory_adoption()
    test_legacy_memory_adoption_ambiguity_fails_closed()
    test_atomic_reservation_rollback_and_source_repair()
    test_duplicate_legacy_adoption_and_job_identity()
    test_lease_fence_and_atomic_projection_claim()
    test_exact_completion_reopen_and_pending_fences()
    print("Ingested source projection store smoke passed")


if __name__ == "__main__":
    main()
