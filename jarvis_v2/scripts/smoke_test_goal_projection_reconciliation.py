from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from jarvis_v2.memory import obsidian as obsidian_module
from jarvis_v2.memory.goal_projection import (
    reconcile_goal_projection,
    reconcile_pending_goal_projections,
)
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import (
    GoalProjectionTarget,
    GoalRecord,
    MemoryStore,
    OrganizedNoteEntryRecord,
)
from jarvis_v2.tools.goals import make_goal_tools


STORE_METHODS = (
    "get_goal_projection_job",
    "ensure_current_goal_projection_job",
    "get_current_goal_projection_snapshot",
    "get_completed_goal_projection_snapshot",
    "prepare_goal_projection",
    "goal_projection_prior_content_digests",
    "narrow_goal_projection_prepared_history",
    "prepare_goal_projection_after_observed_content",
    "goal_projection_publication",
    "mark_goal_projection_error",
    "complete_goal_projection",
    "reopen_completed_goal_projection",
    "mark_goal_projection_audited",
    "ensure_missing_goal_projection_jobs",
    "list_pending_goal_projection_jobs",
    "list_completed_goal_projection_jobs_for_audit",
    "count_pending_goal_projection_jobs",
)
VAULT_METHODS = (
    "goal_projection_candidate_evidence",
    "inspect_goal_projection_prior_evidence",
    "write_goal_with_evidence",
    "verify_goal_projection_evidence",
)
GRAPH_TABLES = (
    "goals",
    "goal_steps",
    "goal_projection_jobs",
    "organized_note_batches",
    "organized_note_entries",
)
WRITE_AUTHORIZER_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_TRIGGER,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_DROP_INDEX,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_DROP_TRIGGER,
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
    }
)


class SimulatedProcessDeath(BaseException):
    pass


def _setup(root: Path) -> tuple[MemoryStore, ObsidianVault]:
    store = MemoryStore(root / "store.sqlite")
    store.init()
    vault = ObsidianVault(root / "vault")
    vault.init()
    return store, vault


def _assert_runtime_contract() -> None:
    missing = [f"MemoryStore.{name}" for name in STORE_METHODS if not hasattr(MemoryStore, name)]
    missing.extend(
        f"ObsidianVault.{name}" for name in VAULT_METHODS if not hasattr(ObsidianVault, name)
    )
    if missing:
        raise SystemExit("missing durable goal projection APIs: " + ", ".join(missing))


def _rows(store: MemoryStore, table: str) -> tuple[dict[str, Any], ...]:
    if table not in GRAPH_TABLES:
        raise ValueError(f"unsupported goal projection smoke table: {table}")
    with store.connect() as conn:
        return tuple(dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid"))


def _graph(store: MemoryStore) -> dict[str, tuple[dict[str, Any], ...]]:
    return {table: _rows(store, table) for table in GRAPH_TABLES}


def _source_snapshot(store: MemoryStore, goal_id: int) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    goal = store.get_goal(goal_id)
    if goal is None:
        raise SystemExit(f"goal #{goal_id} disappeared from its smoke fixture")
    return dict(goal), tuple(dict(step) for step in store.list_goal_steps(goal_id))


def _target(store: MemoryStore, goal_id: int) -> GoalProjectionTarget:
    target = store.ensure_current_goal_projection_job(goal_id)
    if not isinstance(target, GoalProjectionTarget):
        raise SystemExit(
            "ensure_current_goal_projection_job returned the wrong target: "
            f"{type(target).__name__}"
        )
    return target


def _job(store: MemoryStore, goal_id: int) -> sqlite3.Row:
    row = store.get_goal_projection_job(goal_id)
    if row is None:
        raise SystemExit(f"goal #{goal_id} has no projection custody row")
    return row


def _assert_pending_target(store: MemoryStore, target: GoalProjectionTarget, label: str) -> None:
    row = _job(store, target.goal_id)
    forbidden_columns = {"title", "purpose", "horizon", "body", "steps"}
    if forbidden_columns & set(row.keys()):
        raise SystemExit(f"{label} ledger stores raw goal content columns")
    if (
        row["state"] != "pending"
        or row["goal_revision"] != target.revision
        or row["source_digest"] != target.source_digest
        or row["path_display"] is not None
        or row["content_digest"] is not None
    ):
        raise SystemExit(f"{label} did not reserve the exact pending target: {dict(row)}")


def _assert_private(values: tuple[object, ...], forbidden: tuple[str, ...], label: str) -> None:
    rendered = "\n".join(repr(value) for value in values)
    leaked = [value for value in forbidden if value and value in rendered]
    if leaked:
        raise SystemExit(f"{label} leaked private goal content or a local path: {leaked}")


def _goal_note(vault: ObsidianVault, goal_id: int, label: str) -> Path:
    matches = []
    for path in (vault.root_path / "Projects").glob("*.md"):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if f"goal_id: {goal_id}\n" in text and "jarvis_projection: goal\n" in text:
            matches.append(path)
    if len(matches) != 1:
        raise SystemExit(f"{label} expected one ID-bound goal note: {matches}")
    return matches[0]


def _reconcile(
    store: MemoryStore,
    vault: ObsidianVault,
    target: GoalProjectionTarget,
):
    return reconcile_goal_projection(store, vault, target.goal_id, target)


def _install_job_failure(store: MemoryStore, label: str) -> None:
    safe_label = "".join(character for character in label if character.isalnum() or character == "_")
    with store.connect() as conn:
        conn.executescript(
            f"""
            CREATE TRIGGER reject_goal_projection_insert_{safe_label}
            BEFORE INSERT ON goal_projection_jobs
            BEGIN
                SELECT RAISE(ABORT, 'injected goal projection custody failure');
            END;
            CREATE TRIGGER reject_goal_projection_update_{safe_label}
            BEFORE UPDATE ON goal_projection_jobs
            BEGIN
                SELECT RAISE(ABORT, 'injected goal projection custody failure');
            END;
            """
        )


def _expect_database_failure(call, label: str) -> None:
    try:
        call()
    except sqlite3.DatabaseError:
        return
    raise SystemExit(f"{label} did not surface the injected late SQLite failure")


def _database_dump(db_path: Path) -> tuple[str, ...]:
    with sqlite3.connect(db_path) as conn:
        return tuple(conn.iterdump())


def test_additive_legacy_migration_and_read_only_compatibility() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-projection-migration-") as temp:
        root = Path(temp)
        store, _vault = _setup(root)
        goal_id = store.create_goal(
            GoalRecord(
                "PRIVATE-LEGACY-GOAL-TITLE",
                "PRIVATE-LEGACY-GOAL-PURPOSE",
                "this week",
            )
        )
        step_id = store.add_goal_step(goal_id, "PRIVATE-LEGACY-GOAL-STEP")
        if step_id is None:
            raise SystemExit("legacy migration fixture did not create its source step")
        source_before = _source_snapshot(store, goal_id)

        with store.connect() as conn:
            trigger_names = tuple(
                str(row["name"])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                    "AND (tbl_name IN ('goals', 'goal_steps') "
                    "OR lower(COALESCE(sql, '')) LIKE '%goal_projection%')"
                )
            )
            for trigger_name in trigger_names:
                escaped = trigger_name.replace('"', '""')
                conn.execute(f'DROP TRIGGER "{escaped}"')
            conn.execute("DROP TABLE goal_projection_jobs")
            conn.execute("ALTER TABLE goals DROP COLUMN revision")

        compatibility_error = sqlite3.OperationalError(
            "attempt to write a readonly database"
        )
        if store._read_only_init_is_compatible(compatibility_error):
            raise SystemExit("pre-migration goal schema was accepted for read-only startup")
        store.init()
        migrated_goal, migrated_steps = _source_snapshot(store, goal_id)
        expected_goal = dict(source_before[0])
        expected_goal.pop("revision")
        actual_goal = dict(migrated_goal)
        migrated_revision = actual_goal.pop("revision", None)
        if (
            actual_goal != expected_goal
            or migrated_steps != source_before[1]
            or migrated_revision != 1
        ):
            raise SystemExit("additive goal migration changed legacy source content or identity")
        with store.connect() as conn:
            columns = {
                str(row["name"]) for row in conn.execute("PRAGMA table_info(goal_projection_jobs)")
            }
        if not {
            "goal_id",
            "goal_revision",
            "source_digest",
            "state",
            "path_display",
            "content_digest",
            "prepared_content_digest",
            "prepared_content_digests",
            "prior_content_digest",
            "last_error_code",
        }.issubset(columns):
            raise SystemExit(f"additive migration left an incomplete goal ledger: {columns}")
        if {"title", "purpose", "horizon", "body"} & columns:
            raise SystemExit("goal projection migration added private source columns to the ledger")

        store.ensure_missing_goal_projection_jobs(limit=1)
        target = _target(store, goal_id)
        _assert_pending_target(store, target, "migrated legacy goal")
        job_before = dict(_job(store, goal_id))
        store.init()
        if dict(_job(store, goal_id)) != job_before or _source_snapshot(store, goal_id) != (
            migrated_goal,
            migrated_steps,
        ):
            raise SystemExit("repeated goal schema initialization changed source or custody")

        if not store._read_only_init_is_compatible(compatibility_error):
            raise SystemExit("migrated goal schema was rejected for read-only startup")
        dump_before = _database_dump(store.db_path)
        write_attempts: list[tuple[int, str | None]] = []
        real_connect = store.connect

        def audited_connect() -> sqlite3.Connection:
            conn = real_connect()

            def authorize(action, arg1, _arg2, _db_name, _trigger_name):
                if action in WRITE_AUTHORIZER_ACTIONS:
                    write_attempts.append((action, arg1))
                return sqlite3.SQLITE_OK

            conn.set_authorizer(authorize)
            return conn

        with (
            patch.object(store, "connect", side_effect=audited_connect),
            patch.object(store, "_init_schema", side_effect=compatibility_error),
        ):
            store.init()
        if (
            not store.read_only_startup
            or write_attempts
            or _database_dump(store.db_path) != dump_before
        ):
            raise SystemExit(
                "compatible read-only startup attempted a write or changed the legacy database: "
                f"{write_attempts}"
            )

    with TemporaryDirectory(prefix="jarvis-goal-projection-column-migration-") as temp:
        store, _vault = _setup(Path(temp))
        goal_id = store.create_goal(GoalRecord("PRIVATE-OLD-LEDGER-GOAL"))
        job_before = dict(_job(store, goal_id))
        with store.connect() as conn:
            conn.execute("ALTER TABLE goal_projection_jobs DROP COLUMN prior_content_digest")
        store.init()
        migrated = dict(_job(store, goal_id))
        if (
            migrated.get("prior_content_digest") is not None
            or {key: value for key, value in migrated.items() if key != "prior_content_digest"}
            != {key: value for key, value in job_before.items() if key != "prior_content_digest"}
        ):
            raise SystemExit("in-place goal projection ledger migration changed custody state")

    with TemporaryDirectory(prefix="jarvis-goal-projection-ledger-superset-") as temp:
        store, _vault = _setup(Path(temp))
        with store.connect() as conn:
            conn.execute("ALTER TABLE goal_projection_jobs ADD COLUMN title TEXT")
            conn.execute(
                "UPDATE goal_projection_jobs SET title = 'PRIVATE-RAW-GOAL-CONTENT'"
            )
        compatibility_error = sqlite3.OperationalError(
            "attempt to write a readonly database"
        )
        if store._read_only_init_is_compatible(compatibility_error):
            raise SystemExit("raw-content goal ledger superset was accepted read-only")
        try:
            store.init()
        except RuntimeError:
            pass
        else:
            raise SystemExit("raw-content goal ledger superset survived writable init")

    with TemporaryDirectory(prefix="jarvis-goal-projection-lifecycle-migration-") as temp:
        store, vault = _setup(Path(temp))
        goal_id = store.create_goal(GoalRecord("PRIVATE-LIFECYCLE-MIGRATION-GOAL"))
        legacy_job = dict(_job(store, goal_id))
        prepared_digest = "a" * 64
        with store.connect() as conn:
            conn.execute("DROP TABLE goal_projection_jobs")
            conn.executescript(
                """
                CREATE TABLE goal_projection_jobs (
                    goal_id INTEGER PRIMARY KEY REFERENCES goals(id) ON DELETE CASCADE,
                    store_identity TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('pending', 'completed')),
                    goal_revision INTEGER NOT NULL,
                    source_digest TEXT NOT NULL,
                    path_display TEXT,
                    content_digest TEXT,
                    prepared_content_digest TEXT,
                    prior_content_digest TEXT,
                    last_error_code TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    CHECK(
                        (state = 'pending' AND path_display IS NULL
                         AND content_digest IS NULL AND completed_at IS NULL)
                        OR
                        (state = 'completed' AND path_display IS NOT NULL
                         AND content_digest IS NOT NULL AND last_error_code IS NULL
                         AND completed_at IS NOT NULL)
                    )
                );
                CREATE INDEX goal_projection_jobs_pending_updated_idx
                ON goal_projection_jobs(updated_at, goal_id) WHERE state = 'pending';
                CREATE INDEX goal_projection_jobs_pending_freshness_idx
                ON goal_projection_jobs(goal_revision, goal_id) WHERE state = 'pending';
                CREATE INDEX goal_projection_jobs_completed_audit_idx
                ON goal_projection_jobs(updated_at, goal_id) WHERE state = 'completed';
                """
            )
            conn.execute(
                """
                INSERT INTO goal_projection_jobs(
                    goal_id, store_identity, state, goal_revision, source_digest,
                    path_display, content_digest, prepared_content_digest,
                    prior_content_digest, last_error_code,
                    created_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    legacy_job["goal_id"],
                    legacy_job["store_identity"],
                    legacy_job["state"],
                    legacy_job["goal_revision"],
                    legacy_job["source_digest"],
                    legacy_job["path_display"],
                    legacy_job["content_digest"],
                    prepared_digest,
                    legacy_job["prior_content_digest"],
                    legacy_job["last_error_code"],
                    legacy_job["created_at"],
                    legacy_job["updated_at"],
                    legacy_job["completed_at"],
                ),
            )
        store.init()
        migrated_job = _job(store, goal_id)
        if (
            migrated_job["prepared_content_digest"] != prepared_digest
            or json.loads(migrated_job["prepared_content_digests"])
            != [prepared_digest]
        ):
            raise SystemExit("lifecycle migration lost pending prepared custody evidence")
        outcome = _reconcile(store, vault, _target(store, goal_id))
        if outcome.status not in {"completed", "superseded"}:
            raise SystemExit("lifecycle migration did not remain publishable")
        forbidden_digest = "b" * 64
        with store.connect() as conn:
            try:
                conn.execute(
                    "UPDATE goal_projection_jobs SET prepared_content_digest = ?, "
                    "prepared_content_digests = ? WHERE goal_id = ?",
                    (
                        forbidden_digest,
                        json.dumps([forbidden_digest], separators=(",", ":")),
                        goal_id,
                    ),
                )
            except sqlite3.IntegrityError:
                pass
            else:
                raise SystemExit("migrated lifecycle admitted prepared completed custody")
        compatibility_error = sqlite3.OperationalError(
            "attempt to write a readonly database"
        )
        if not store._read_only_init_is_compatible(compatibility_error):
            raise SystemExit("valid migrated lifecycle was rejected read-only")


def test_step_bound_and_timestamp_guards() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-projection-guards-") as temp:
        store, _vault = _setup(Path(temp))
        goal_id = store.create_goal(GoalRecord("PRIVATE-GOAL-BOUND-FIXTURE"))
        timestamp = "2026-07-14T00:00:00Z"
        with store.connect() as conn:
            conn.executemany(
                "INSERT INTO goal_steps(goal_id, body, status, created_at, updated_at) "
                "VALUES (?, ?, 'open', ?, ?)",
                (
                    (goal_id, f"bounded-step-{index}", timestamp, timestamp)
                    for index in range(1_000)
                ),
            )
        source_before = _source_snapshot(store, goal_id)
        job_before = dict(_job(store, goal_id))
        try:
            store.add_goal_step(goal_id, "over-limit step")
        except RuntimeError:
            pass
        else:
            raise SystemExit("goal store admitted a source the reconciler cannot project")
        if _source_snapshot(store, goal_id) != source_before or dict(_job(store, goal_id)) != job_before:
            raise SystemExit("goal step limit refusal changed source or custody state")
        with store.connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO goal_steps(goal_id, body, status, created_at, updated_at) "
                    "VALUES (?, 'raw-over-limit', 'open', ?, ?)",
                    (goal_id, timestamp, timestamp),
                )
            except sqlite3.IntegrityError:
                pass
            else:
                raise SystemExit("goal step limit trigger accepted a raw over-limit insert")

        with store.connect() as conn:
            try:
                conn.execute(
                    "UPDATE goal_projection_jobs SET updated_at = ? WHERE goal_id = ?",
                    ("z" * 20, goal_id),
                )
            except sqlite3.IntegrityError:
                pass
            else:
                raise SystemExit("goal projection timestamp guard accepted malformed ordering data")

        target = _target(store, goal_id)
        if not store.prepare_goal_projection(target, "a" * 64):
            raise SystemExit("extreme timestamp fixture could not prepare its projection")
        with store.connect() as conn:
            conn.execute(
                "UPDATE goal_projection_jobs SET state = 'completed', "
                "path_display = 'Projects/Goal 1.md', content_digest = ?, "
                "prepared_content_digest = NULL, prepared_content_digests = '[]', "
                "last_error_code = NULL, updated_at = ?, completed_at = ? "
                "WHERE goal_id = ?",
                ("a" * 64, "9999-12-31T23:59:59Z", timestamp, goal_id),
            )
        try:
            audited_at = store.mark_goal_projection_audited(target, "a" * 64)
        except OverflowError as exc:
            raise SystemExit("canonical extreme timestamp wedged goal audit") from exc
        if not audited_at:
            raise SystemExit("canonical extreme timestamp could not be normalized")


def test_atomic_mutations_and_late_job_rollbacks() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-projection-atomic-") as temp:
        root = Path(temp)

        create_store, _ = _setup(root / "create")
        _install_job_failure(create_store, "create")
        _expect_database_failure(
            lambda: create_store.create_goal(GoalRecord("ROLLBACK-CREATE-GOAL")),
            "goal create",
        )
        if any(_rows(create_store, table) for table in ("goals", "goal_projection_jobs")):
            raise SystemExit("goal create custody failure left a source or job row")

        step_store, _ = _setup(root / "step")
        step_goal_id = step_store.create_goal(GoalRecord("ROLLBACK-STEP-GOAL"))
        step_before = _graph(step_store)
        _install_job_failure(step_store, "step")
        _expect_database_failure(
            lambda: step_store.add_goal_step(step_goal_id, "ROLLBACK-STEP-BODY"),
            "goal step create",
        )
        if _graph(step_store) != step_before:
            raise SystemExit("goal step custody failure changed source or pending target")

        status_store, _ = _setup(root / "status")
        status_goal_id = status_store.create_goal(GoalRecord("ROLLBACK-STATUS-GOAL"))
        status_before = _graph(status_store)
        _install_job_failure(status_store, "status")
        _expect_database_failure(
            lambda: status_store.set_goal_status_result(status_goal_id, "paused"),
            "goal status change",
        )
        if _graph(status_store) != status_before:
            raise SystemExit("goal status custody failure changed source or pending target")

        complete_store, _ = _setup(root / "complete")
        complete_goal_id = complete_store.create_goal(GoalRecord("ROLLBACK-COMPLETE-GOAL"))
        complete_step_id = complete_store.add_goal_step(
            complete_goal_id, "ROLLBACK-COMPLETE-STEP"
        )
        if complete_step_id is None:
            raise SystemExit("step completion rollback fixture did not create a step")
        complete_before = _graph(complete_store)
        _install_job_failure(complete_store, "complete")
        _expect_database_failure(
            lambda: complete_store.complete_goal_step_result(complete_step_id),
            "goal step completion",
        )
        if _graph(complete_store) != complete_before:
            raise SystemExit("goal completion custody failure changed source or pending target")

        organizer_store, _ = _setup(root / "organizer")
        organizer_before = _graph(organizer_store)
        _install_job_failure(organizer_store, "organizer")
        source_key = "organize-note:v1:" + hashlib.sha256(b"goal rollback fixture").hexdigest()
        entry = OrganizedNoteEntryRecord(
            "goal",
            goal=GoalRecord(
                "ROLLBACK-ORGANIZER-GOAL",
                "ROLLBACK-ORGANIZER-PURPOSE",
                "soon",
            ),
        )
        _expect_database_failure(
            lambda: organizer_store.reserve_organized_note_batch(source_key, (entry,)),
            "organizer goal reservation",
        )
        if _graph(organizer_store) != organizer_before:
            raise SystemExit("organizer custody failure left a batch, goal, entry, or job")


def test_same_second_revisions_and_noop_mutations() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-projection-revision-") as temp:
        root = Path(temp)
        store, vault = _setup(root)
        fixed_now = "2026-07-14T00:00:00Z"
        revisions: list[int] = []
        with patch("jarvis_v2.memory.store.utc_now", return_value=fixed_now):
            goal_id = store.create_goal(
                GoalRecord(
                    "PRIVATE-MONOTONIC-GOAL",
                    "PRIVATE-MONOTONIC-PURPOSE",
                    "today",
                )
            )
            revisions.append(int(store.get_goal(goal_id)["revision"]))
            step_id = store.add_goal_step(goal_id, "PRIVATE-MONOTONIC-STEP")
            if step_id is None:
                raise SystemExit("same-second fixture did not create a goal step")
            revisions.append(int(store.get_goal(goal_id)["revision"]))
            status_result = store.set_goal_status_result(goal_id, "paused")
            revisions.append(int(store.get_goal(goal_id)["revision"]))
            completion_result = store.complete_goal_step_result(step_id)
            revisions.append(int(store.get_goal(goal_id)["revision"]))
            if not status_result.changed or not completion_result.changed:
                raise SystemExit("same-second fixture missed a real source mutation")

            no_status_change = store.set_goal_status_result(goal_id, "paused")
            no_completion_change = store.complete_goal_step_result(step_id)
            revision_before_export = int(store.get_goal(goal_id)["revision"])
            export_goal = make_goal_tools(store, vault)[6]
            first_export = export_goal({"goal_id": goal_id})
            second_export = export_goal({"goal_id": goal_id})
            revision_after_export = int(store.get_goal(goal_id)["revision"])

        if revisions != [1, 2, 3, 4]:
            raise SystemExit(f"same-second goal mutations did not advance monotonically: {revisions}")
        timestamps = {
            store.get_goal(goal_id)["updated_at"],
            *(step["updated_at"] for step in store.list_goal_steps(goal_id)),
        }
        if timestamps != {fixed_now}:
            raise SystemExit(f"same-second fixture did not actually share one timestamp: {timestamps}")
        if (
            no_status_change.changed
            or no_completion_change.changed
            or revision_before_export != 4
            or revision_after_export != 4
            or not first_export.ok
            or not second_export.ok
        ):
            raise SystemExit("no-op status, completion, or export advanced goal revision")
        completed_job = _job(store, goal_id)
        if completed_job["state"] != "completed" or completed_job["goal_revision"] != 4:
            raise SystemExit("no-op exports did not retain exact revision-four evidence")


def test_pre_replace_failure_and_later_repair() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-projection-retry-") as temp:
        root = Path(temp)
        store, vault = _setup(root)
        title = "PRIVATE-PRE-REPLACE-GOAL"
        purpose = "PRIVATE-PRE-REPLACE-PURPOSE"
        goal_id = store.create_goal(GoalRecord(title, purpose, "this week"))
        target = _target(store, goal_id)
        source_before = _source_snapshot(store, goal_id)
        real_replace = obsidian_module._replace_text

        def fail_before_replace(*_args, **_kwargs):
            raise OSError(f"injected private failure {title} {purpose} {root}")

        with patch.object(obsidian_module, "_replace_text", side_effect=fail_before_replace):
            failed = _reconcile(store, vault, target)
        failed_job = _job(store, goal_id)
        if (
            failed.status != "pending_error"
            or failed_job["state"] != "pending"
            or failed_job["last_error_code"] != "vault_publish_failed"
            or list((vault.root_path / "Projects").glob("*.md"))
            or _source_snapshot(store, goal_id) != source_before
        ):
            raise SystemExit(f"pre-replace failure lost durable goal custody: {failed}")
        _assert_private(tuple(vars(failed).values()), (title, purpose, str(root)), "failed reconcile")

        with patch.object(obsidian_module, "_replace_text", wraps=real_replace) as replace_spy:
            repaired = _reconcile(store, vault, target)
        if (
            repaired.status not in {"completed", "superseded"}
            or _job(store, goal_id)["state"] != "completed"
            or replace_spy.call_count != 1
            or _source_snapshot(store, goal_id) != source_before
        ):
            raise SystemExit(f"pending goal reconciliation did not repair the projection: {repaired}")
        note = _goal_note(vault, goal_id, "pre-replace repair")
        if title not in note.read_text(encoding="utf-8") or purpose not in note.read_text(
            encoding="utf-8"
        ):
            raise SystemExit("repaired goal note omitted exact source content")


def test_post_file_process_death_adopts_exact_evidence() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-projection-adopt-") as temp:
        root = Path(temp)
        store, vault = _setup(root)
        goal_id = store.create_goal(
            GoalRecord("PRIVATE-ADOPTION-GOAL", "PRIVATE-ADOPTION-PURPOSE", "today")
        )
        target = _target(store, goal_id)
        source_before = _source_snapshot(store, goal_id)
        real_complete = store._complete_goal_projection_in_connection

        def die_before_completion(*_args, **_kwargs):
            raise SimulatedProcessDeath("injected death after goal file publication")

        try:
            with patch.object(
                store,
                "_complete_goal_projection_in_connection",
                side_effect=die_before_completion,
            ):
                _reconcile(store, vault, target)
        except SimulatedProcessDeath:
            pass
        else:
            raise SystemExit("post-file process-death injection did not interrupt reconciliation")

        note = _goal_note(vault, goal_id, "post-file process death")
        bytes_before = note.read_bytes()
        if _job(store, goal_id)["state"] != "pending":
            raise SystemExit("process death after file write falsely completed the ledger")

        def refuse_duplicate_replace(*_args, **_kwargs):
            raise AssertionError("exact retained goal evidence reached atomic replace again")

        with (
            patch.object(
                store,
                "_complete_goal_projection_in_connection",
                wraps=real_complete,
            ),
            patch.object(obsidian_module, "_replace_text", side_effect=refuse_duplicate_replace),
        ):
            adopted = _reconcile(store, vault, target)
        if (
            adopted.status not in {"completed", "superseded"}
            or _job(store, goal_id)["state"] != "completed"
            or note.read_bytes() != bytes_before
            or _source_snapshot(store, goal_id) != source_before
            or len(list((vault.root_path / "Projects").glob("*.md"))) != 1
        ):
            raise SystemExit(f"exact post-file evidence was not adopted on retry: {adopted}")


def test_source_advance_waits_for_publication_completion() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-projection-fenced-completion-") as temp:
        root = Path(temp)
        store, vault = _setup(root)
        goal_id = store.create_goal(
            GoalRecord("PRIVATE-FENCED-GOAL", "PRIVATE-FENCED-PURPOSE", "soon")
        )
        target = _target(store, goal_id)
        real_write = vault.write_goal_with_evidence
        mutation_started = threading.Event()
        mutation_finished = threading.Event()
        mutation_errors: list[BaseException] = []
        worker: list[threading.Thread] = []

        def advance_source() -> None:
            mutation_started.set()
            try:
                store.add_goal_step(goal_id, "PRIVATE-FENCED-STEP")
            except BaseException as exc:
                mutation_errors.append(exc)
            finally:
                mutation_finished.set()

        def write_then_race(*args, **kwargs):
            evidence = real_write(*args, **kwargs)
            thread = threading.Thread(target=advance_source, daemon=True)
            worker.append(thread)
            thread.start()
            if not mutation_started.wait(timeout=2):
                raise AssertionError("source-advance worker did not start")
            if mutation_finished.wait(timeout=0.05):
                raise AssertionError("source advanced before projection ledger completion")
            return evidence

        with patch.object(vault, "write_goal_with_evidence", side_effect=write_then_race):
            published = _reconcile(store, vault, target)
        if not worker:
            raise SystemExit("fenced publication did not exercise its source race")
        worker[0].join(timeout=5)
        if worker[0].is_alive() or mutation_errors:
            raise SystemExit(f"source advance failed after publication commit: {mutation_errors}")
        completed_digest = published.content_digest
        newer = _job(store, goal_id)
        if (
            published.status not in {"completed", "superseded"}
            or newer["state"] != "pending"
            or newer["goal_revision"] != target.revision + 1
            or newer["prior_content_digest"] != completed_digest
        ):
            raise SystemExit("source advance lost the just-published ownership digest")
        repaired = _reconcile(store, vault, _target(store, goal_id))
        note = _goal_note(vault, goal_id, "fenced source advance")
        if (
            repaired.status not in {"completed", "superseded"}
            or "PRIVATE-FENCED-STEP" not in note.read_text(encoding="utf-8")
        ):
            raise SystemExit("new source revision did not replace its owned projection")


def test_post_publication_crash_then_source_advance_recovers() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-projection-crash-advance-") as temp:
        root = Path(temp)
        store, vault = _setup(root)
        goal_id = store.create_goal(
            GoalRecord("PRIVATE-CRASH-ADVANCE-GOAL", "PRIVATE-CRASH-ADVANCE-PURPOSE")
        )
        old_target = _target(store, goal_id)

        def die_before_completion(*_args, **_kwargs):
            raise SimulatedProcessDeath("injected death before durable completion")

        try:
            with patch.object(
                store,
                "_complete_goal_projection_in_connection",
                side_effect=die_before_completion,
            ):
                _reconcile(store, vault, old_target)
        except SimulatedProcessDeath:
            pass
        else:
            raise SystemExit("post-publication crash injection did not interrupt")

        crashed_job = _job(store, goal_id)
        prepared_digest = crashed_job["prepared_content_digest"]
        note = _goal_note(vault, goal_id, "crash-before-source-advance")
        if (
            crashed_job["state"] != "pending"
            or type(prepared_digest) is not str
            or hashlib.sha256(note.read_bytes()).hexdigest() != prepared_digest
        ):
            raise SystemExit("crashed publication lost its durable prepared evidence")

        store.add_goal_step(goal_id, "PRIVATE-POST-CRASH-STEP")
        advanced_job = _job(store, goal_id)
        if (
            advanced_job["state"] != "pending"
            or advanced_job["prepared_content_digest"] is not None
            or prepared_digest
            not in tuple(json.loads(advanced_job["prepared_content_digests"]))
        ):
            raise SystemExit("source advance did not inherit prepared ownership evidence")
        repaired = _reconcile(store, vault, _target(store, goal_id))
        if (
            repaired.status not in {"completed", "superseded"}
            or _job(store, goal_id)["state"] != "completed"
            or "PRIVATE-POST-CRASH-STEP" not in note.read_text(encoding="utf-8")
        ):
            raise SystemExit("crash-plus-source-advance projection did not converge")


def test_prepared_without_write_preserves_completed_prior() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-projection-prepare-crash-") as temp:
        root = Path(temp)
        store, vault = _setup(root)
        goal_id = store.create_goal(
            GoalRecord("PRIVATE-PREPARE-CRASH-GOAL", "PRIVATE-PREPARE-CRASH-PURPOSE")
        )
        initial = _reconcile(store, vault, _target(store, goal_id))
        note = _goal_note(vault, goal_id, "prepared-before-write baseline")
        completed_digest = hashlib.sha256(note.read_bytes()).hexdigest()
        if initial.status not in {"completed", "superseded"}:
            raise SystemExit("prepared-before-write fixture did not publish its baseline")

        store.add_goal_step(goal_id, "PRIVATE-UNPUBLISHED-REVISION-TWO")
        revision_two = _target(store, goal_id)

        def die_before_file(*_args, **_kwargs):
            raise SimulatedProcessDeath("injected death after preparation before file")

        try:
            with patch.object(
                store,
                "goal_projection_publication",
                side_effect=die_before_file,
            ):
                _reconcile(store, vault, revision_two)
        except SimulatedProcessDeath:
            pass
        else:
            raise SystemExit("prepare-before-file crash injection did not interrupt")
        after_prepare = _job(store, goal_id)
        unpublished_digest = after_prepare["prepared_content_digest"]
        if (
            type(unpublished_digest) is not str
            or after_prepare["prior_content_digest"] != completed_digest
            or hashlib.sha256(note.read_bytes()).hexdigest() != completed_digest
        ):
            raise SystemExit("pre-file crash changed or forgot the completed projection")

        store.add_goal_step(goal_id, "PRIVATE-REVISION-THREE")
        revision_three_job = _job(store, goal_id)
        retained_candidates = tuple(
            json.loads(revision_three_job["prepared_content_digests"])
        )
        if (
            revision_three_job["prior_content_digest"] != completed_digest
            or unpublished_digest not in retained_candidates
            or revision_three_job["prepared_content_digest"] is not None
        ):
            raise SystemExit("source advance replaced proven prior with an unpublished candidate")
        repaired = _reconcile(store, vault, _target(store, goal_id))
        if (
            repaired.status not in {"completed", "superseded"}
            or "PRIVATE-REVISION-THREE" not in note.read_text(encoding="utf-8")
        ):
            raise SystemExit("prepare-crash-plus-source-advance did not converge")


def test_full_prepared_history_is_narrowed_from_vault_evidence() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-projection-history-narrow-") as temp:
        store, vault = _setup(Path(temp))
        goal_id = store.create_goal(
            GoalRecord(
                "PRIVATE-HISTORY-NARROW-GOAL",
                "PRIVATE-HISTORY-NARROW-PURPOSE",
            )
        )
        target = _target(store, goal_id)
        history = [
            hashlib.sha256(
                f"prepared-candidate-{index}".encode("utf-8")
            ).hexdigest()
            for index in range(256)
        ]
        with store.connect() as conn:
            conn.execute(
                "UPDATE goal_projection_jobs SET prepared_content_digest = ?, "
                "prepared_content_digests = ? WHERE goal_id = ?",
                (
                    history[-1],
                    json.dumps(history, separators=(",", ":")),
                    goal_id,
                ),
            )
        outcome = _reconcile(store, vault, target)
        job = _job(store, goal_id)
        if (
            outcome.status not in {"completed", "superseded"}
            or job["state"] != "completed"
            or job["prepared_content_digest"] is not None
            or job["prepared_content_digests"] != "[]"
        ):
            raise SystemExit(
                "vault evidence did not narrow a full prepared history before repair"
            )
        _goal_note(vault, goal_id, "history narrowing repair")


def test_unprepared_completion_is_rejected() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-projection-unprepared-complete-") as temp:
        store, _vault = _setup(Path(temp))
        goal_id = store.create_goal(GoalRecord("PRIVATE-UNPREPARED-COMPLETION"))
        target = _target(store, goal_id)
        digest = "a" * 64
        if store.complete_goal_projection(
            target,
            "Projects/Goal 1.md",
            digest,
        ):
            raise SystemExit("unprepared goal projection completion was accepted")
        with store.connect() as conn:
            try:
                conn.execute(
                    "UPDATE goal_projection_jobs SET state = 'completed', "
                    "path_display = 'Projects/Goal 1.md', content_digest = ?, "
                    "prepared_content_digest = NULL, prepared_content_digests = '[]', "
                    "last_error_code = NULL, completed_at = updated_at WHERE goal_id = ?",
                    (digest, goal_id),
                )
            except sqlite3.IntegrityError:
                pass
            else:
                raise SystemExit("goal lifecycle trigger accepted unprepared completion")
        if _job(store, goal_id)["state"] != "pending":
            raise SystemExit("unprepared completion changed goal projection custody")
        pending = dict(_job(store, goal_id))
        with store.connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO goal_projection_jobs(
                        goal_id, store_identity, state, goal_revision, source_digest,
                        path_display, content_digest, prepared_content_digest,
                        prepared_content_digests, prior_content_digest, last_error_code,
                        created_at, updated_at, completed_at
                    ) VALUES (?, ?, 'completed', ?, ?, 'Projects/Goal 1.md', ?,
                              NULL, '[]', ?, NULL, ?, ?, ?)
                    """,
                    (
                        goal_id,
                        pending["store_identity"],
                        pending["goal_revision"],
                        pending["source_digest"],
                        digest,
                        pending["prior_content_digest"],
                        pending["created_at"],
                        pending["updated_at"],
                        pending["updated_at"],
                    ),
                )
            except sqlite3.IntegrityError:
                pass
            else:
                raise SystemExit("replace-style SQL forged completed goal custody")
        if not store.prepare_goal_projection(target, digest) or not store.complete_goal_projection(
            target,
            "Projects/Goal 1.md",
            digest,
        ):
            raise SystemExit("prepared goal completion fixture did not complete")
        with store.connect() as conn:
            try:
                conn.execute(
                    "UPDATE goal_projection_jobs SET content_digest = ? WHERE goal_id = ?",
                    ("b" * 64, goal_id),
                )
            except sqlite3.IntegrityError:
                pass
            else:
                raise SystemExit("completed goal evidence was mutable without reopening")


def test_goal_evidence_uses_strict_exact_bytes() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-projection-exact-bytes-") as temp:
        store, vault = _setup(Path(temp))
        title = "PRIVATE-REPLACEMENT-CHAR-\ufffd-GOAL"
        goal_id = store.create_goal(GoalRecord(title, "PRIVATE-EXACT-BYTE-PURPOSE"))
        target = _target(store, goal_id)
        initial = _reconcile(store, vault, target)
        note = _goal_note(vault, goal_id, "strict byte fixture")
        job = _job(store, goal_id)
        valid_bytes = note.read_bytes()
        corrupted = valid_bytes.replace(b"\xef\xbf\xbd", b"\xff", 1)
        if corrupted == valid_bytes:
            raise SystemExit("strict byte fixture did not contain replacement bytes")
        note.write_bytes(corrupted)
        try:
            verified = vault.verify_goal_projection_evidence(
                goal_id=goal_id,
                title=title,
                store_identity=store.get_store_identity(),
                expected_content_digest=job["content_digest"],
                expected_revision=target.revision,
                expected_source_digest=target.source_digest,
            )
        except ValueError:
            verified = False
        repaired = _reconcile(store, vault, target)
        current = _job(store, goal_id)
        if (
            initial.status not in {"completed", "superseded"}
            or verified
            or repaired.status != "pending_error"
            or current["state"] != "pending"
            or current["last_error_code"] != "vault_publish_failed"
            or note.read_bytes() != corrupted
        ):
            raise SystemExit("goal evidence treated invalid raw bytes as ledger content")


def test_goal_publication_recovers_exchange_journal_before_adoption() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-projection-exchange-recovery-") as temp:
        store, vault = _setup(Path(temp))
        goal_id = store.create_goal(
            GoalRecord("PRIVATE-EXCHANGE-RECOVERY-GOAL", "PRIVATE-EXCHANGE-PURPOSE")
        )
        initial_target = _target(store, goal_id)
        initial = _reconcile(store, vault, initial_target)
        note = _goal_note(vault, goal_id, "exchange recovery baseline")
        initial_bytes = note.read_bytes()
        store.add_goal_step(goal_id, "PRIVATE-EXCHANGE-NEW-STEP")
        current_target = _target(store, goal_id)
        snapshot = store.get_current_goal_projection_snapshot(current_target)
        if snapshot is None:
            raise SystemExit("exchange recovery source snapshot is unavailable")
        goal, steps = snapshot
        candidate = vault._goal_projection_candidate(
            goal,
            steps,
            store_identity=store.get_store_identity(),
        )
        candidate_path = candidate[0]
        candidate_bytes = candidate[-1].encode("utf-8")
        if initial.status not in {"completed", "superseded"} or candidate_path != note:
            raise SystemExit("exchange recovery fixture did not bind the goal path")
        external_bytes = initial_bytes + b"EXTERNAL-EDIT-DURING-EXCHANGE\n"
        token = "a" * 16
        temp_path = note.with_name(f".{note.name}.{token}.tmp")
        journal_path = note.with_name(
            f".{note.name}.{token}.exchange.json"
        )
        note.write_bytes(candidate_bytes)
        temp_path.write_bytes(external_bytes)
        journal_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "temp_name": temp_path.name,
                    "expected_sha256": hashlib.sha256(initial_bytes).hexdigest(),
                    "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        outcome = _reconcile(store, vault, current_target)
        current = _job(store, goal_id)
        if (
            outcome.status != "pending_error"
            or current["state"] != "pending"
            or current["last_error_code"] != "vault_publish_failed"
            or note.read_bytes() != external_bytes
            or temp_path.exists()
            or journal_path.exists()
        ):
            raise SystemExit(
                "goal reconciliation adopted bytes before exchange recovery"
            )


def test_exchange_recovery_hashes_crlf_prior_as_exact_bytes() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-projection-crlf-exchange-") as temp:
        store, vault = _setup(Path(temp))
        goal_id = store.create_goal(
            GoalRecord("PRIVATE-CRLF-EXCHANGE-GOAL", "PRIVATE-CRLF-PURPOSE")
        )
        initial_target = _target(store, goal_id)
        initial = _reconcile(store, vault, initial_target)
        note = _goal_note(vault, goal_id, "CRLF exchange recovery baseline")
        initial_bytes = note.read_bytes()
        crlf_prior = initial_bytes.replace(b"\n", b"\r\n")
        if crlf_prior == initial_bytes:
            raise SystemExit("CRLF exchange fixture did not change prior bytes")

        store.add_goal_step(goal_id, "PRIVATE-CRLF-EXCHANGE-NEW-STEP")
        current_target = _target(store, goal_id)
        snapshot = store.get_current_goal_projection_snapshot(current_target)
        if snapshot is None:
            raise SystemExit("CRLF exchange source snapshot is unavailable")
        goal, steps = snapshot
        candidate = vault._goal_projection_candidate(
            goal,
            steps,
            store_identity=store.get_store_identity(),
        )
        candidate_path = candidate[0]
        candidate_bytes = candidate[-1].encode("utf-8")
        candidate_digest = hashlib.sha256(candidate_bytes).hexdigest()
        if (
            initial.status not in {"completed", "superseded"}
            or candidate_path != note
            or not store.prepare_goal_projection(current_target, candidate_digest)
        ):
            raise SystemExit("CRLF exchange fixture did not prepare current evidence")

        token = "b" * 16
        temp_path = note.with_name(f".{note.name}.{token}.tmp")
        journal_path = note.with_name(f".{note.name}.{token}.exchange.json")
        note.write_bytes(candidate_bytes)
        temp_path.write_bytes(crlf_prior)
        journal_path.write_bytes(
            json.dumps(
                {
                    "version": 1,
                    "temp_name": temp_path.name,
                    "expected_sha256": hashlib.sha256(crlf_prior).hexdigest(),
                    "candidate_sha256": candidate_digest,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )

        outcome = _reconcile(store, vault, current_target)
        current = _job(store, goal_id)
        if (
            outcome.status not in {"completed", "superseded"}
            or current["state"] != "completed"
            or current["content_digest"] != candidate_digest
            or note.read_bytes() != candidate_bytes
            or temp_path.exists()
            or journal_path.exists()
        ):
            raise SystemExit("goal exchange recovery normalized CRLF evidence")


def test_exchange_recovery_rejects_oversized_artifacts_boundedly() -> None:
    cases = ("journal", "destination", "temp")
    for case in cases:
        with TemporaryDirectory(
            prefix=f"jarvis-goal-projection-oversized-{case}-"
        ) as temp:
            store, vault = _setup(Path(temp))
            goal_id = store.create_goal(
                GoalRecord(f"PRIVATE-OVERSIZED-{case}-GOAL", "PRIVATE-BOUND-PURPOSE")
            )
            initial_target = _target(store, goal_id)
            initial = _reconcile(store, vault, initial_target)
            note = _goal_note(vault, goal_id, f"oversized {case} baseline")
            initial_bytes = note.read_bytes()
            store.add_goal_step(goal_id, f"PRIVATE-OVERSIZED-{case}-STEP")
            current_target = _target(store, goal_id)
            snapshot = store.get_current_goal_projection_snapshot(current_target)
            if snapshot is None:
                raise SystemExit(f"oversized {case} source snapshot is unavailable")
            goal, steps = snapshot
            candidate = vault._goal_projection_candidate(
                goal,
                steps,
                store_identity=store.get_store_identity(),
            )
            candidate_bytes = candidate[-1].encode("utf-8")
            candidate_digest = hashlib.sha256(candidate_bytes).hexdigest()
            if (
                initial.status not in {"completed", "superseded"}
                or not store.prepare_goal_projection(current_target, candidate_digest)
            ):
                raise SystemExit(f"oversized {case} fixture did not prepare")

            token = {"journal": "c", "destination": "d", "temp": "e"}[case] * 16
            temp_path = note.with_name(f".{note.name}.{token}.tmp")
            journal_path = note.with_name(f".{note.name}.{token}.exchange.json")
            oversized = b"X" * (obsidian_module.MAX_GOAL_PROJECTION_CHARS * 4 + 1)
            destination_bytes = oversized if case == "destination" else candidate_bytes
            displaced_bytes = oversized if case == "temp" else initial_bytes
            journal_bytes = json.dumps(
                {
                    "version": 1,
                    "temp_name": temp_path.name,
                    "expected_sha256": hashlib.sha256(displaced_bytes).hexdigest(),
                    "candidate_sha256": candidate_digest,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            if case == "journal":
                journal_bytes += b" " * (
                    obsidian_module.MAX_ATOMIC_EXCHANGE_JOURNAL_BYTES
                    - len(journal_bytes)
                    + 1
                )
            note.write_bytes(destination_bytes)
            temp_path.write_bytes(displaced_bytes)
            journal_path.write_bytes(journal_bytes)

            outcome = _reconcile(store, vault, current_target)
            current = _job(store, goal_id)
            if (
                outcome.status != "pending_error"
                or current["state"] != "pending"
                or current["last_error_code"] != "vault_publish_failed"
                or note.read_bytes() != destination_bytes
                or temp_path.read_bytes() != displaced_bytes
                or journal_path.read_bytes() != journal_bytes
            ):
                raise SystemExit(
                    f"goal exchange recovery did not fail closed for oversized {case}"
                )


def test_completed_missing_repairs_and_external_edit_fails_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-projection-audit-") as temp:
        root = Path(temp)
        store, vault = _setup(root)
        private_values = (
            "PRIVATE-MISSING-GOAL",
            "PRIVATE-MISSING-PURPOSE",
            "PRIVATE-CORRUPT-GOAL",
            "PRIVATE-CORRUPT-PURPOSE",
        )
        targets = (
            _target(
                store,
                store.create_goal(GoalRecord(private_values[0], private_values[1], "soon")),
            ),
            _target(
                store,
                store.create_goal(GoalRecord(private_values[2], private_values[3], "later")),
            ),
        )
        initial = tuple(_reconcile(store, vault, target) for target in targets)
        if any(outcome.status not in {"completed", "superseded"} for outcome in initial):
            raise SystemExit(f"completed-audit fixtures did not publish: {initial}")
        source_before = tuple(_source_snapshot(store, target.goal_id) for target in targets)
        missing_path = _goal_note(vault, targets[0].goal_id, "missing audit fixture")
        corrupt_path = _goal_note(vault, targets[1].goal_id, "corrupt audit fixture")
        missing_path.unlink()
        corrupt_path.write_text(
            corrupt_path.read_text(encoding="utf-8") + "CORRUPTED-BY-GOAL-SMOKE\n",
            encoding="utf-8",
        )

        corrupt_bytes = corrupt_path.read_bytes()
        repaired = tuple(_reconcile(store, vault, target) for target in targets)
        missing_note = _goal_note(vault, targets[0].goal_id, "completed evidence repair")
        missing_job = _job(store, targets[0].goal_id)
        if (
            repaired[0].status not in {"completed", "superseded"}
            or missing_job["state"] != "completed"
            or missing_job["content_digest"]
            != hashlib.sha256(missing_note.read_bytes()).hexdigest()
            or missing_job["last_error_code"] is not None
            or _source_snapshot(store, targets[0].goal_id) != source_before[0]
        ):
            raise SystemExit("missing completed goal evidence did not reopen and repair")
        corrupt_job = _job(store, targets[1].goal_id)
        if (
            repaired[1].status != "pending_error"
            or corrupt_job["state"] != "pending"
            or corrupt_job["last_error_code"] != "vault_publish_failed"
            or corrupt_path.read_bytes() != corrupt_bytes
            or _source_snapshot(store, targets[1].goal_id) != source_before[1]
        ):
            raise SystemExit("externally edited completed goal projection did not fail closed")
        _assert_private(
            tuple(value for outcome in repaired for value in vars(outcome).values()),
            private_values + (str(root), str(missing_path), str(corrupt_path)),
            "completed evidence repair outcomes",
        )


def test_stale_and_conflicting_targets_cannot_replace_newer_bytes() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-projection-freshness-") as temp:
        root = Path(temp)
        store, vault = _setup(root)
        goal_id = store.create_goal(
            GoalRecord("PRIVATE-FRESHNESS-GOAL", "PRIVATE-REVISION-ONE", "today")
        )
        stale_target = _target(store, goal_id)
        step_id = store.add_goal_step(goal_id, "PRIVATE-REVISION-TWO-STEP")
        if step_id is None:
            raise SystemExit("freshness fixture did not create revision two")
        current_target = _target(store, goal_id)
        if current_target.revision != stale_target.revision + 1:
            raise SystemExit("freshness fixture did not retain adjacent exact targets")
        current = _reconcile(store, vault, current_target)
        if current.status not in {"completed", "superseded"}:
            raise SystemExit(f"newest goal projection did not publish: {current}")
        note = _goal_note(vault, goal_id, "freshness current target")
        newest_bytes = note.read_bytes()

        stale = _reconcile(store, vault, stale_target)
        if note.read_bytes() != newest_bytes or len(list((vault.root_path / "Projects").glob("*.md"))) != 1:
            raise SystemExit(f"stale lower target replaced or duplicated newer bytes: {stale}")

        conflict_digest = "f" * 64 if current_target.source_digest != "f" * 64 else "e" * 64
        conflict_target = GoalProjectionTarget(
            current_target.goal_id,
            current_target.revision,
            conflict_digest,
        )
        try:
            conflict = _reconcile(store, vault, conflict_target)
        except ValueError:
            conflict = None
        if note.read_bytes() != newest_bytes:
            raise SystemExit(f"equal-revision conflicting target replaced newer bytes: {conflict}")

        goal, steps = _source_snapshot(store, goal_id)
        lower_goal = dict(goal)
        lower_goal["revision"] = int(goal["revision"]) - 1
        try:
            vault.write_goal_with_evidence(
                lower_goal,
                steps,
                store_identity=store.get_store_identity(),
            )
        except RuntimeError:
            pass
        else:
            raise SystemExit("vault accepted a lower goal revision over newer owned bytes")
        equal_conflict_goal = dict(goal)
        equal_conflict_goal["purpose"] = "PRIVATE-EQUAL-REVISION-CONFLICT"
        try:
            vault.write_goal_with_evidence(
                equal_conflict_goal,
                steps,
                store_identity=store.get_store_identity(),
            )
        except RuntimeError:
            pass
        else:
            raise SystemExit("vault accepted a conflicting digest at the same goal revision")
        if note.read_bytes() != newest_bytes:
            raise SystemExit("vault freshness refusals changed the current goal bytes")


def test_source_mutation_preserves_external_projection_edits() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-projection-edited-prior-") as temp:
        store, vault = _setup(Path(temp))
        goal_id = store.create_goal(
            GoalRecord("PRIVATE-EDITED-PRIOR-GOAL", "PRIVATE-EDITED-PRIOR-PURPOSE")
        )
        initial_target = _target(store, goal_id)
        initial = _reconcile(store, vault, initial_target)
        if initial.status not in {"completed", "superseded"}:
            raise SystemExit("external-edit fixture did not publish its initial projection")
        note = _goal_note(vault, goal_id, "external edit before source mutation")
        edited = note.read_bytes() + b"EXTERNAL-EDIT-MUST-SURVIVE\n"
        note.write_bytes(edited)

        step_id = store.add_goal_step(goal_id, "PRIVATE-NEW-REVISION-STEP")
        if step_id is None:
            raise SystemExit("external-edit fixture did not advance the source revision")
        current_target = _target(store, goal_id)
        outcome = _reconcile(store, vault, current_target)
        job = _job(store, goal_id)
        if (
            outcome.status != "pending_error"
            or job["state"] != "pending"
            or job["last_error_code"] != "vault_publish_failed"
            or note.read_bytes() != edited
        ):
            raise SystemExit("newer goal revision overwrote externally edited prior evidence")


class _MalformedOnceStore:
    def __init__(self, store: MemoryStore, goal_id: int, private_path: str) -> None:
        self._store = store
        self._goal_id = goal_id
        self._private_path = private_path
        self._armed = False
        self._served = False

    def __getattr__(self, name: str):
        return getattr(self._store, name)

    def list_pending_goal_projection_jobs(self, limit: int):
        rows = self._store.list_pending_goal_projection_jobs(limit=limit)
        if self._served:
            return rows
        for row in rows:
            if row["goal_id"] == self._goal_id:
                self._armed = True
                malformed = dict(row)
                malformed["path_display"] = self._private_path
                return [malformed]
        return rows

    def get_goal_projection_job(self, goal_id: int):
        row = self._store.get_goal_projection_job(goal_id)
        if self._armed and not self._served and goal_id == self._goal_id and row is not None:
            self._armed = False
            self._served = True
            malformed = dict(row)
            malformed["path_display"] = self._private_path
            return malformed
        return row


def test_bounded_recovery_and_malformed_fairness_privacy() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-projection-bounded-") as temp:
        root = Path(temp)
        store, vault = _setup(root / "bounded")
        targets = [
            _target(
                store,
                store.create_goal(
                    GoalRecord(
                        f"PRIVATE-BOUNDED-GOAL-{index}",
                        f"PRIVATE-BOUNDED-PURPOSE-{index}",
                        "soon",
                    )
                ),
            )
            for index in range(7)
        ]
        previous_pending = store.count_pending_goal_projection_jobs()
        for _ in range(10):
            if previous_pending == 0:
                break
            summary = reconcile_pending_goal_projections(store, vault, limit=2)
            current_pending = store.count_pending_goal_projection_jobs()
            if (
                len(summary.outcomes) > 3
                or summary.attempted != len(summary.outcomes)
                or current_pending >= previous_pending
            ):
                raise SystemExit(f"bounded goal recovery made no bounded progress: {summary}")
            previous_pending = current_pending
        if previous_pending != 0:
            raise SystemExit("bounded goal recovery starved work beyond the first limit window")
        for target in targets:
            if _job(store, target.goal_id)["state"] != "completed":
                raise SystemExit(f"bounded recovery did not complete goal #{target.goal_id}")
            _goal_note(vault, target.goal_id, "bounded recovery")

        fairness_store, fairness_vault = _setup(root / "malformed")
        bad_title = "PRIVATE-MALFORMED-GOAL-TEXT"
        bad_id = fairness_store.create_goal(GoalRecord(bad_title, "PRIVATE-BAD-PURPOSE"))
        good_id = fairness_store.create_goal(
            GoalRecord("PRIVATE-HEALTHY-GOAL-TEXT", "PRIVATE-HEALTHY-PURPOSE")
        )
        private_path = str(root / bad_title / "goal.md")
        malformed_store = _MalformedOnceStore(fairness_store, bad_id, private_path)

        malformed = reconcile_pending_goal_projections(
            malformed_store,
            fairness_vault,
            limit=1,
        )
        if (
            not malformed.outcomes
            or malformed.outcomes[0].status != "malformed"
            or _job(fairness_store, bad_id)["last_error_code"] != "malformed_job"
            or list((fairness_vault.root_path / "Projects").glob("*.md"))
        ):
            raise SystemExit(f"malformed goal job did not fail closed: {malformed}")
        _assert_private(
            tuple(vars(malformed).values())
            + tuple(value for outcome in malformed.outcomes for value in vars(outcome).values()),
            (bad_title, "PRIVATE-BAD-PURPOSE", private_path, str(root)),
            "malformed goal summary",
        )

        healthy = reconcile_pending_goal_projections(
            malformed_store,
            fairness_vault,
            limit=1,
        )
        if (
            _job(fairness_store, good_id)["state"] != "completed"
            or healthy.completed < 1
            or fairness_store.count_pending_goal_projection_jobs() != 1
        ):
            raise SystemExit(f"malformed goal job starved healthy bounded work: {healthy}")
        final = reconcile_pending_goal_projections(
            malformed_store,
            fairness_vault,
            limit=2,
        )
        if final.pending != 0 or fairness_store.count_pending_goal_projection_jobs() != 0:
            raise SystemExit(f"malformed goal job did not recover after yielding fairly: {final}")


def main() -> None:
    _assert_runtime_contract()
    test_additive_legacy_migration_and_read_only_compatibility()
    test_step_bound_and_timestamp_guards()
    test_atomic_mutations_and_late_job_rollbacks()
    test_same_second_revisions_and_noop_mutations()
    test_pre_replace_failure_and_later_repair()
    test_post_file_process_death_adopts_exact_evidence()
    test_source_advance_waits_for_publication_completion()
    test_post_publication_crash_then_source_advance_recovers()
    test_prepared_without_write_preserves_completed_prior()
    test_full_prepared_history_is_narrowed_from_vault_evidence()
    test_unprepared_completion_is_rejected()
    test_goal_evidence_uses_strict_exact_bytes()
    test_goal_publication_recovers_exchange_journal_before_adoption()
    test_exchange_recovery_hashes_crlf_prior_as_exact_bytes()
    test_exchange_recovery_rejects_oversized_artifacts_boundedly()
    test_completed_missing_repairs_and_external_edit_fails_closed()
    test_stale_and_conflicting_targets_cannot_replace_newer_bytes()
    test_source_mutation_preserves_external_projection_edits()
    test_bounded_recovery_and_malformed_fairness_privacy()
    print("Goal projection reconciliation smoke test passed.")


if __name__ == "__main__":
    main()
