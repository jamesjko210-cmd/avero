from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
from typing import Any, Iterator

from jarvis_v2.memory.store import (
    MAX_PERSON_IDENTITY_DIRTY_FOREGROUND,
    MAX_PERSON_IDENTITY_ERROR_CODE_CHARS,
    MAX_PERSON_NAME_CHARS,
    PERSON_IDENTITY_DIRTY_BACKLOG_ERROR,
    PERSON_IDENTITY_NORMALIZER_VERSION,
    MemoryStore,
    PersonRecord,
)


NOW = "2026-07-12T00:00:00+00:00"
PRIVATE_MARKERS = (
    "Alice Smith",
    "alice   smith",
    "ＡＬＩＣＥ Smith",
    "PRIVATE-BLOB-NAME",
    "PRIVATE-CONTACT-PAYLOAD",
    "PRIVATE-DIRTY-NAME",
    "/\x55sers/example/private",
)

def _create_legacy_store(path: Path) -> tuple[list[int], tuple[Any, ...]]:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name NOT NULL UNIQUE,
            relation TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT '',
            last_contact_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE contacts (
            id INTEGER PRIMARY KEY,
            display_name TEXT NOT NULL,
            payload BLOB NOT NULL
        );
        INSERT INTO contacts(id, display_name, payload)
        VALUES (9, 'PRIVATE-CONTACT-PAYLOAD', X'000102ff');
        """
    )
    names: tuple[Any, ...] = (
        " Alice Smith ",
        "alice   smith",
        "ＡＬＩＣＥ Smith",
        sqlite3.Binary(b"PRIVATE-BLOB-NAME\x00\xff"),
        9223372036854775000,
        "   ",
        "X" * (MAX_PERSON_NAME_CHARS + 1),
        "Unrelated Person",
    )
    ids: list[int] = []
    for index, name in enumerate(names, start=1):
        cursor = conn.execute(
            """
            INSERT INTO people(name, relation, notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, f"legacy-{index}", f"note-{index}", NOW, NOW),
        )
        ids.append(int(cursor.lastrowid))
    contacts = tuple(conn.execute("SELECT * FROM contacts ORDER BY id").fetchone())
    conn.commit()
    conn.close()
    return ids, contacts


def _people_snapshot(store: MemoryStore, ids: list[int]) -> list[tuple[Any, ...]]:
    placeholders = ",".join("?" for _ in ids)
    with store.connect() as conn:
        return [
            tuple(row)
            for row in conn.execute(
                f"SELECT id, typeof(name), name, relation, notes FROM people "
                f"WHERE id IN ({placeholders}) ORDER BY id",
                ids,
            )
        ]


def _contact_snapshot(store: MemoryStore) -> tuple[Any, ...]:
    with store.connect() as conn:
        row = conn.execute("SELECT * FROM contacts ORDER BY id").fetchone()
    if row is None:
        raise SystemExit("person identity migration removed the legacy contacts fixture")
    return tuple(row)


def _insert_person_sql(conn: sqlite3.Connection, name: str, *, relation: str = "") -> int:
    cursor = conn.execute(
        """
        INSERT INTO people(name, relation, notes, created_at, updated_at)
        VALUES (?, ?, '', ?, ?)
        """,
        (name, relation, NOW, NOW),
    )
    return int(cursor.lastrowid)


def _identity_write_state(store: MemoryStore, person_ids: tuple[int, ...]) -> tuple[Any, ...]:
    placeholders = ",".join("?" for _ in person_ids)
    with store.connect() as conn:
        people = tuple(
            tuple(row)
            for row in conn.execute(
                f"SELECT id, revision, last_contact_at, updated_at FROM people "
                f"WHERE id IN ({placeholders}) ORDER BY id",
                person_ids,
            )
        )
        interactions = int(
            conn.execute(
                f"SELECT COUNT(*) FROM person_interactions "
                f"WHERE person_id IN ({placeholders})",
                person_ids,
            ).fetchone()[0]
        )
        projection_jobs = tuple(
            tuple(row)
            for row in conn.execute(
                f"SELECT person_id, state, person_revision FROM person_projection_jobs "
                f"WHERE person_id IN ({placeholders}) ORDER BY person_id",
                person_ids,
            )
        )
    return people, interactions, projection_jobs


def _review_rows(store: MemoryStore, *, scan_limit: int) -> list[dict[str, Any]]:
    after = 0
    rows: list[dict[str, Any]] = []
    while True:
        page = store.review_person_identity_memberships(
            after_person_id=after,
            scan_limit=scan_limit,
        )
        if set(page) != {"rows", "scanned", "next_after_person_id", "complete"}:
            raise SystemExit(f"person identity review returned an unsafe envelope: {page}")
        batch = list(page["rows"])
        if len(batch) > scan_limit:
            raise SystemExit("person identity review exceeded its requested scan bound")
        if page["scanned"] != len(batch) or type(page["complete"]) is not bool:
            raise SystemExit(f"person identity review page metadata is inconsistent: {page}")
        rendered = json.dumps(page, default=str, sort_keys=True, ensure_ascii=False)
        if any(marker in rendered for marker in PRIVATE_MARKERS):
            raise SystemExit("person identity review leaked private source content")
        normalized: list[dict[str, Any]] = []
        for item in batch:
            if isinstance(item, sqlite3.Row):
                item = dict(item)
            elif not isinstance(item, dict):
                try:
                    item = dict(vars(item))
                except (TypeError, ValueError) as exc:
                    raise SystemExit(
                        "person identity review rows must be structured records"
                    ) from exc
            if set(item) != {"person_id", "count", "reason"}:
                raise SystemExit(
                    f"person identity review exposed non-ID custody fields: {sorted(item)}"
                )
            person_id = item.get("person_id")
            if type(person_id) is not int or person_id <= after:
                raise SystemExit(f"person identity review cursor is not ID-only/monotonic: {item}")
            if type(item.get("count")) is not int or item["count"] < 0:
                raise SystemExit(f"person identity review count is malformed: {item}")
            if item.get("reason") not in {
                "owner",
                "shadow",
                "name_not_text",
                "name_too_long",
                "name_invalid_unicode",
                "name_blank",
            }:
                raise SystemExit(f"person identity review reason is not a bounded code: {item}")
            normalized.append(item)
        if not normalized:
            if page["next_after_person_id"] != after or page["complete"] is not True:
                raise SystemExit(f"empty person identity review page advanced its cursor: {page}")
            break
        ids = [int(item["person_id"]) for item in normalized]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise SystemExit(f"person identity review order is not deterministic: {ids}")
        rows.extend(normalized)
        if page["next_after_person_id"] != ids[-1]:
            raise SystemExit(f"person identity review cursor missed its page tail: {page}")
        after = ids[-1]
        if page["complete"]:
            break
    return rows


def _assert_identity_schema_and_repair(store: MemoryStore) -> None:
    with store.connect() as conn:
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'person_identity%'"
            )
        }
        if "person_identity_memberships" not in tables:
            raise SystemExit(f"person identity membership table is missing: {sorted(tables)}")
        columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(person_identity_memberships)")
        }
        version_columns = sorted(column for column in columns if "version" in column.casefold())
        if not version_columns:
            raise SystemExit("person identity memberships do not record a normalizer version")
        version_column = version_columns[0]
        conn.execute(
            f"UPDATE person_identity_memberships SET {version_column} = 'stale-test-version' "
            "WHERE person_id = 8"
        )
    store.init()
    with store.connect() as conn:
        repaired = conn.execute(
            f"SELECT {version_column} FROM person_identity_memberships WHERE person_id = 8"
        ).fetchone()
    if (
        repaired is None
        or type(repaired[0]) is not str
        or not repaired[0]
        or repaired[0] != PERSON_IDENTITY_NORMALIZER_VERSION
    ):
        raise SystemExit(f"person identity normalizer-version repair failed: {repaired}")


def test_legacy_backfill_review_and_lifecycle() -> None:
    with TemporaryDirectory(prefix="jarvis-person-identity-legacy-") as temp:
        db_path = Path(temp) / "legacy.sqlite"
        ids, contact_before = _create_legacy_store(db_path)
        store = MemoryStore(db_path)
        before = _people_snapshot(store, ids)
        store.init()

        owner = store.get_person(name="  ALICE  SMITH ")
        if owner is None or int(owner["id"]) != ids[0]:
            raise SystemExit(f"legacy collision did not select the lowest-ID owner: {owner}")
        if _people_snapshot(store, ids) != before:
            raise SystemExit("person identity backfill modified a legacy person source row")
        if _contact_snapshot(store) != contact_before:
            raise SystemExit("person identity initialization modified contacts")

        conflicts = store.count_person_identity_conflicts()
        if type(conflicts) is not int or conflicts != 2:
            raise SystemExit(f"person identity conflict count missed valid shadows: {conflicts}")
        default_page = store.review_person_identity_memberships()
        if default_page["scanned"] != len(ids) or default_page["complete"] is not True:
            raise SystemExit(f"default person identity review page was not bounded/complete: {default_page}")
        clamped_page = store.review_person_identity_memberships(scan_limit=0)
        if clamped_page["scanned"] != 1 or clamped_page["complete"] is not False:
            raise SystemExit(f"person identity review did not clamp a zero scan limit: {clamped_page}")
        for bad_cursor in (-1, True, "0"):
            try:
                store.review_person_identity_memberships(after_person_id=bad_cursor)  # type: ignore[arg-type]
            except ValueError:
                pass
            else:
                raise SystemExit(f"person identity review accepted a bad cursor: {bad_cursor!r}")
        for bad_limit in (True, "2"):
            try:
                store.review_person_identity_memberships(scan_limit=bad_limit)  # type: ignore[arg-type]
            except TypeError:
                pass
            else:
                raise SystemExit(f"person identity review accepted a bad scan limit: {bad_limit!r}")
        review = _review_rows(store, scan_limit=2)
        reviewed_ids = {int(row["person_id"]) for row in review}
        if not set(ids[1:7]).issubset(reviewed_ids):
            raise SystemExit(f"person identity review missed collision/quarantine IDs: {reviewed_ids}")
        reasons = {int(row["person_id"]): row["reason"] for row in review}
        expected_reasons = {
            ids[0]: "owner",
            ids[1]: "shadow",
            ids[2]: "shadow",
            ids[3]: "name_not_text",
            ids[4]: "name_not_text",
            ids[5]: "name_blank",
            ids[6]: "name_too_long",
            ids[7]: "owner",
        }
        if reasons != expected_reasons:
            raise SystemExit(f"person identity review misclassified legacy custody: {reasons}")

        store.init()
        if store.count_person_identity_conflicts() != conflicts:
            raise SystemExit("repeated person identity initialization changed conflict custody")
        if _people_snapshot(store, ids) != before:
            raise SystemExit("idempotent person identity initialization changed legacy rows")
        _assert_identity_schema_and_repair(store)

        try:
            store.record_person_interaction_with_projections(
                person_id=ids[1],
                name="Alice Smith",
                summary="must refuse the shadow ID",
                happened_at=NOW,
            )
        except ValueError:
            pass
        else:
            raise SystemExit("explicit shadow person ID was accepted for interaction")

        with store.connect() as conn:
            conn.execute("DELETE FROM people WHERE id = ?", (ids[0],))
        store.init()
        promoted = store.get_person(name="alice smith")
        if promoted is None or int(promoted["id"]) != ids[1]:
            raise SystemExit(f"owner deletion did not promote the next lowest shadow: {promoted}")

        for person_id in ids[1:3]:
            with store.connect() as conn:
                conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
            store.init()
        if store.get_person(name="alice smith") is not None:
            raise SystemExit("deleted identity retained a stale owner lookup")
        recreated = store.upsert_person(PersonRecord(name="Alice Smith", relation="fresh"))
        if recreated in set(ids) or recreated <= max(ids):
            raise SystemExit(f"fresh identity recreation reused legacy custody: {recreated}")
        if _contact_snapshot(store) != contact_before:
            raise SystemExit("person owner promotion/recreation modified contacts")
        with store.connect() as conn:
            violations = list(conn.execute("PRAGMA foreign_key_check"))
        if violations:
            raise SystemExit(f"person identity lifecycle left FK violations: {violations}")


@contextmanager
def _traced_connections(
    original_connect: Any,
    statements: list[str],
) -> Iterator[sqlite3.Connection]:
    with original_connect() as conn:
        conn.set_trace_callback(statements.append)
        yield conn


def test_indexed_hot_paths_do_not_scan_people() -> None:
    with TemporaryDirectory(prefix="jarvis-person-identity-index-") as temp:
        store = MemoryStore(Path(temp) / "store.sqlite")
        store.init()
        owner_id = store.upsert_person(PersonRecord(name="Indexed Person", notes="seed"))
        statements: list[str] = []
        original_connect = store.connect
        store.connect = lambda: _traced_connections(original_connect, statements)  # type: ignore[method-assign]

        found = store.get_person(name="  INDEXED   PERSON ")
        updated = store.upsert_person(PersonRecord(name="Indexed Person", relation="friend"))
        profile = store.record_person_with_projections(
            PersonRecord(name="Ｉｎｄｅｘｅｄ Person", notes="profile update")
        )
        interaction = store.record_person_interaction_with_projections(
            person_id=owner_id,
            name="indexed person",
            summary="indexed interaction",
            happened_at=NOW,
        )
        if (
            found is None
            or int(found["id"]) != owner_id
            or updated != owner_id
            or profile.person_id != owner_id
            or interaction.person_id != owner_id
        ):
            raise SystemExit("indexed person hot paths diverged from their identity owner")

        normalized = [" ".join(statement.upper().split()) for statement in statements]
        scans = [
            statement
            for statement in normalized
            if "FROM PEOPLE ORDER BY ID" in statement
            or "FROM PEOPLE AS" in statement and "ORDER BY PEOPLE.ID" in statement
        ]
        if scans:
            raise SystemExit(f"person hot path performed a full people scan: {scans}")
        if not any("PERSON_IDENTITY" in statement for statement in normalized):
            raise SystemExit("person hot paths did not use the identity membership index")


def test_concurrent_create_update_interaction_converges() -> None:
    with TemporaryDirectory(prefix="jarvis-person-identity-concurrent-") as temp:
        store = MemoryStore(Path(temp) / "store.sqlite")
        store.init()
        workers = 12
        barrier = Barrier(workers)

        def mutate(index: int) -> int:
            barrier.wait()
            variant = ("Concurrent Person", " concurrent   person ", "ＣＯＮＣＵＲＲＥＮＴ Person")[
                index % 3
            ]
            if index % 3 == 0:
                return store.upsert_person(PersonRecord(name=variant, relation=f"relation-{index}"))
            if index % 3 == 1:
                return store.record_person_with_projections(
                    PersonRecord(name=variant, notes=f"profile-{index}")
                ).person_id
            return store.record_person_interaction_with_projections(
                person_id=None,
                name=variant,
                summary=f"interaction-{index}",
                happened_at=NOW,
            ).person_id

        with ThreadPoolExecutor(max_workers=workers) as pool:
            person_ids = list(pool.map(mutate, range(workers)))
        if len(set(person_ids)) != 1:
            raise SystemExit(f"concurrent person mutations split identity ownership: {person_ids}")
        owner_id = person_ids[0]
        owner = store.get_person(name="concurrent person")
        interactions = store.list_person_interactions(owner_id, limit=workers)
        if owner is None or int(owner["id"]) != owner_id or len(interactions) != workers // 3:
            raise SystemExit("concurrent person mutations did not converge on one complete owner")
        if store.count_person_identity_conflicts() != 0:
            raise SystemExit("concurrent person mutations left identity conflicts")
        review = _review_rows(store, scan_limit=3)
        if review != [{"person_id": owner_id, "count": 1, "reason": "owner"}]:
            raise SystemExit(f"clean concurrent ownership review was not singular: {review}")
        with store.connect() as conn:
            people = int(conn.execute("SELECT COUNT(*) FROM people").fetchone()[0])
            violations = list(conn.execute("PRAGMA foreign_key_check"))
        if people != 1 or violations:
            raise SystemExit(
                f"concurrent person convergence was not singular/FK-clean: people={people}, fk={violations}"
            )


def test_post_init_direct_sql_is_reconciled_before_owner_resolution() -> None:
    with TemporaryDirectory(prefix="jarvis-person-identity-direct-sql-") as temp:
        db_path = Path(temp) / "store.sqlite"
        store = MemoryStore(db_path)
        store.init()

        with store.connect() as conn:
            exact_id = _insert_person_sql(conn, "Mixed Version Person", relation="direct")
        if store.upsert_person(PersonRecord(name="Mixed Version Person", relation="api")) != exact_id:
            raise SystemExit("an exact direct-SQL insert was not reconciled before write resolution")

        with store.connect() as conn:
            variant_id = _insert_person_sql(conn, " mixed   version person ", relation="variant")
        resolved = store.get_person(name="ＭＩＸＥＤ VERSION PERSON")
        if resolved is None or int(resolved["id"]) != exact_id:
            raise SystemExit("a normalized direct-SQL insert changed the lowest-ID owner")
        if store.upsert_person(PersonRecord(name="mixed version person")) != exact_id:
            raise SystemExit("a normalized direct-SQL insert caused write ownership to split")

        with store.connect() as conn:
            conn.execute(
                "UPDATE people SET name = ?, updated_at = ? WHERE id = ?",
                ("Renamed Direct Person", NOW, variant_id),
            )
        old_owner = store.get_person(name="mixed version person")
        renamed_owner = store.get_person(name=" renamed   direct person ")
        if old_owner is None or int(old_owner["id"]) != exact_id:
            raise SystemExit("a direct rename disturbed the prior identity owner")
        if renamed_owner is None or int(renamed_owner["id"]) != variant_id:
            raise SystemExit("a direct rename was not reconciled before read resolution")
        if store.upsert_person(PersonRecord(name="Ｒｅｎａｍｅｄ Direct Person")) != variant_id:
            raise SystemExit("a direct rename was not reconciled before write resolution")

        with store.connect() as conn:
            people_before = int(conn.execute("SELECT COUNT(*) FROM people").fetchone()[0])
        restarted = MemoryStore(db_path)
        restarted.init()
        with restarted.connect() as conn:
            people_after = int(conn.execute("SELECT COUNT(*) FROM people").fetchone()[0])
        owners_after = (
            restarted.get_person(name="mixed version person"),
            restarted.get_person(name="renamed direct person"),
        )
        if people_after != people_before or any(row is None for row in owners_after):
            raise SystemExit("restart duplicated or lost a direct-SQL identity")
        if tuple(int(row["id"]) for row in owners_after if row is not None) != (
            exact_id,
            variant_id,
        ):
            raise SystemExit("direct-SQL identity ownership flipped after restart")


def test_direct_sql_count_and_review_reconcile_without_restart() -> None:
    with TemporaryDirectory(prefix="jarvis-person-identity-direct-review-") as temp:
        store = MemoryStore(Path(temp) / "store.sqlite")
        store.init()
        owner_id = store.upsert_person(PersonRecord(name="Immediate Conflict Person"))

        with store.connect() as conn:
            shadow_id = _insert_person_sql(conn, " immediate   conflict person ")
        if store.count_person_identity_conflicts() != 1:
            raise SystemExit("direct insert was not reconciled before conflict counting")
        review = {
            int(row["person_id"]): (int(row["count"]), str(row["reason"]))
            for row in _review_rows(store, scan_limit=2)
        }
        expected = {owner_id: (2, "owner"), shadow_id: (2, "shadow")}
        if review != expected:
            raise SystemExit(f"direct insert review missed exact owner/shadow custody: {review}")

        with store.connect() as conn:
            conn.execute(
                "UPDATE people SET name = ?, updated_at = ? WHERE id = ?",
                ("Immediate Rename Person", NOW, shadow_id),
            )
        if store.count_person_identity_conflicts() != 0:
            raise SystemExit("direct rename was not reconciled before conflict counting")
        review = {
            int(row["person_id"]): (int(row["count"]), str(row["reason"]))
            for row in _review_rows(store, scan_limit=1)
        }
        expected = {owner_id: (1, "owner"), shadow_id: (1, "owner")}
        if review != expected:
            raise SystemExit(f"direct rename review retained stale shadow custody: {review}")

        with store.connect() as conn:
            conn.execute(
                "UPDATE people SET name = ?, updated_at = ? WHERE id = ?",
                ("IMMEDIATE CONFLICT PERSON", NOW, shadow_id),
            )
        if store.count_person_identity_conflicts() != 1:
            raise SystemExit("direct rename into collision was not reconciled before counting")
        review = {
            int(row["person_id"]): (int(row["count"]), str(row["reason"]))
            for row in _review_rows(store, scan_limit=2)
        }
        expected = {owner_id: (2, "owner"), shadow_id: (2, "shadow")}
        if review != expected:
            raise SystemExit(f"direct rename review missed exact owner/shadow custody: {review}")


def test_trigger_ownership_preserves_unrelated_extras_during_repair() -> None:
    with TemporaryDirectory(prefix="jarvis-person-identity-trigger-ownership-") as temp:
        store = MemoryStore(Path(temp) / "store.sqlite")
        store.init()
        extra_names = (
            "personXidentityYunrelated",
            "person_identity_unowned_extra",
        )
        with store.connect() as conn:
            for trigger_name in extra_names:
                conn.execute(
                    f'CREATE TRIGGER "{trigger_name}" AFTER UPDATE OF relation ON people '
                    "BEGIN SELECT 1; END"
                )
            before = {
                str(row["name"]): str(row["sql"])
                for row in conn.execute(
                    "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' "
                    "AND name IN (?, ?)",
                    extra_names,
                )
            }
            conn.execute("DROP TRIGGER person_identity_people_insert_dirty")

        store.init()
        expected_owned = set(store._person_identity_trigger_sql())
        with store.connect() as conn:
            after = {
                str(row["name"]): str(row["sql"])
                for row in conn.execute(
                    "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' "
                    "AND name IN (?, ?)",
                    extra_names,
                )
            }
            owned_after = {
                str(row["name"])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger'"
                )
                if str(row["name"]) in expected_owned
            }
        if after != before:
            raise SystemExit(
                f"person identity schema repair dropped or rewrote unowned triggers: {sorted(after)}"
            )
        if owned_after != expected_owned:
            raise SystemExit(f"person identity schema repair missed owned triggers: {owned_after}")


def test_oversized_dirty_backlog_fails_closed_then_init_drains_in_batches() -> None:
    with TemporaryDirectory(prefix="jarvis-person-identity-dirty-bound-") as temp:
        store = MemoryStore(Path(temp) / "store.sqlite")
        store.init()
        owner_id = store.upsert_person(
            PersonRecord(name="Backlog Stable Owner", relation="unchanged", notes="stable")
        )
        dirty_count = MAX_PERSON_IDENTITY_DIRTY_FOREGROUND * 2 + 88
        with store.connect() as conn:
            for index in range(dirty_count):
                _insert_person_sql(conn, f"PRIVATE-DIRTY-NAME {index:04d}")
            dirty_before = int(
                conn.execute("SELECT COUNT(*) FROM person_identity_dirty").fetchone()[0]
            )
            owner_before = tuple(
                conn.execute(
                    "SELECT name, relation, notes, revision, updated_at FROM people WHERE id = ?",
                    (owner_id,),
                ).fetchone()
            )
            people_before = int(conn.execute("SELECT COUNT(*) FROM people").fetchone()[0])
            interactions_before = int(
                conn.execute("SELECT COUNT(*) FROM person_interactions").fetchone()[0]
            )
            projection_before = tuple(
                tuple(row)
                for row in conn.execute(
                    "SELECT person_id, state, person_revision FROM person_projection_jobs "
                    "ORDER BY person_id"
                )
            )
        if dirty_before != dirty_count:
            raise SystemExit(f"oversized dirty fixture was incomplete: {dirty_before}")

        failures: list[tuple[str, str]] = []
        operations = (
            ("get", lambda: store.get_person(name="PRIVATE-DIRTY-NAME 0599")),
            (
                "upsert",
                lambda: store.upsert_person(
                    PersonRecord(name="Backlog Stable Owner", relation="must-not-write")
                ),
            ),
            ("count", store.count_person_identity_conflicts),
            ("review", store.review_person_identity_memberships),
        )
        for label, operation in operations:
            try:
                operation()
            except RuntimeError as exc:
                failures.append((label, str(exc)))
            else:
                raise SystemExit(f"{label} did not fail closed on an oversized dirty backlog")

        messages = {message for _, message in failures}
        if messages != {PERSON_IDENTITY_DIRTY_BACKLOG_ERROR}:
            raise SystemExit(f"dirty backlog operations returned non-fixed errors: {failures}")
        error = messages.pop()
        if not error or len(error) > MAX_PERSON_IDENTITY_ERROR_CODE_CHARS:
            raise SystemExit("dirty backlog error was empty or unbounded")
        if any(marker in error for marker in PRIVATE_MARKERS) or any(char.isdigit() for char in error):
            raise SystemExit("dirty backlog error exposed source content or workload cardinality")

        with store.connect() as conn:
            dirty_after_failures = int(
                conn.execute("SELECT COUNT(*) FROM person_identity_dirty").fetchone()[0]
            )
            owner_after_failures = tuple(
                conn.execute(
                    "SELECT name, relation, notes, revision, updated_at FROM people WHERE id = ?",
                    (owner_id,),
                ).fetchone()
            )
            people_after_failures = int(conn.execute("SELECT COUNT(*) FROM people").fetchone()[0])
            interactions_after_failures = int(
                conn.execute("SELECT COUNT(*) FROM person_interactions").fetchone()[0]
            )
            projection_after_failures = tuple(
                tuple(row)
                for row in conn.execute(
                    "SELECT person_id, state, person_revision FROM person_projection_jobs "
                    "ORDER BY person_id"
                )
            )
        if (
            dirty_after_failures != dirty_before
            or owner_after_failures != owner_before
            or people_after_failures != people_before
            or interactions_after_failures != interactions_before
            or projection_after_failures != projection_before
        ):
            raise SystemExit("oversized dirty backlog refusal partially mutated authoritative state")

        statements: list[str] = []
        original_connect = store.connect
        store.connect = lambda: _traced_connections(  # type: ignore[method-assign]
            original_connect, statements
        )
        store.init()
        store.connect = original_connect  # type: ignore[method-assign]
        normalized = [" ".join(statement.upper().split()) for statement in statements]
        dirty_reads = [
            statement
            for statement in normalized
            if statement.startswith("SELECT DIRTY.PERSON_ID")
            and "FROM PERSON_IDENTITY_DIRTY AS DIRTY" in statement
        ]
        if len(dirty_reads) < 3 or any(" LIMIT " not in statement for statement in dirty_reads):
            raise SystemExit(
                f"person identity init did not read the dirty backlog in bounded batches: {dirty_reads}"
            )
        with store.connect() as conn:
            dirty_after_init = int(
                conn.execute("SELECT COUNT(*) FROM person_identity_dirty").fetchone()[0]
            )
        restored = store.get_person(name="private-dirty-name 0599")
        if dirty_after_init != 0 or restored is None:
            raise SystemExit("bounded init did not fully restore dirty identity lookup")


def test_malformed_names_survive_and_are_quarantined_content_free() -> None:
    with TemporaryDirectory(prefix="jarvis-person-identity-malformed-") as temp:
        db_path = Path(temp) / "legacy.sqlite"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name NOT NULL UNIQUE,
                relation TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                last_contact_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        malformed_ids = [
            int(
                conn.execute(
                    "INSERT INTO people(name, created_at, updated_at) "
                    "VALUES (CAST(X'80' AS TEXT), ?, ?)",
                    (NOW, NOW),
                ).lastrowid
            ),
            int(
                conn.execute(
                    "INSERT INTO people(name, created_at, updated_at) VALUES (?, ?, ?)",
                    (sqlite3.Binary(b"PRIVATE-BLOB-NAME\x00\xff"), NOW, NOW),
                ).lastrowid
            ),
            int(
                conn.execute(
                    "INSERT INTO people(name, created_at, updated_at) VALUES (?, ?, ?)",
                    (9223372036854775000, NOW, NOW),
                ).lastrowid
            ),
            int(
                conn.execute(
                    "INSERT INTO people(name, created_at, updated_at) VALUES ('   ', ?, ?)",
                    (NOW, NOW),
                ).lastrowid
            ),
            int(
                conn.execute(
                    "INSERT INTO people(name, created_at, updated_at) VALUES (?, ?, ?)",
                    ("X" * (MAX_PERSON_NAME_CHARS + 1), NOW, NOW),
                ).lastrowid
            ),
        ]
        conn.commit()
        conn.close()

        store = MemoryStore(db_path)
        store.init()
        review = {int(row["person_id"]): row for row in _review_rows(store, scan_limit=2)}
        expected = (
            "name_invalid_unicode",
            "name_not_text",
            "name_not_text",
            "name_blank",
            "name_too_long",
        )
        actual = tuple(review[person_id]["reason"] for person_id in malformed_ids)
        if actual != expected or any(review[person_id]["count"] != 0 for person_id in malformed_ids):
            raise SystemExit(f"malformed person quarantine codes were incorrect: {actual}")

        raw = sqlite3.connect(db_path)
        raw.text_factory = bytes
        surviving = {
            int(row[0])
            for row in raw.execute(
                "SELECT id FROM people WHERE id IN (?,?,?,?,?)",
                malformed_ids,
            )
        }
        raw.close()
        if surviving != set(malformed_ids):
            raise SystemExit("person identity initialization removed malformed source rows")


def test_malformed_membership_schema_is_rebuilt_and_indexed() -> None:
    with TemporaryDirectory(prefix="jarvis-person-identity-schema-") as temp:
        db_path = Path(temp) / "store.sqlite"
        store = MemoryStore(db_path)
        store.init()
        owner_id = store.upsert_person(PersonRecord(name="Schema Owner"))

        with sqlite3.connect(db_path) as conn:
            conn.executescript(
                """
                PRAGMA foreign_keys = OFF;
                DROP TABLE person_identity_memberships;
                CREATE TABLE person_identity_memberships (
                    person_id INTEGER PRIMARY KEY,
                    identity_key TEXT,
                    normalizer_version TEXT NOT NULL,
                    state TEXT NOT NULL,
                    error_code TEXT
                );
                CREATE INDEX person_identity_memberships_owner_idx
                ON person_identity_memberships(person_id);
                """
            )

        store.init()
        with store.connect() as conn:
            foreign_keys = list(conn.execute("PRAGMA foreign_key_list(person_identity_memberships)"))
            index_sql_row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'index' "
                "AND name = 'person_identity_memberships_owner_idx'"
            ).fetchone()
            plan = list(
                conn.execute(
                    """
                    EXPLAIN QUERY PLAN
                    SELECT people.*
                    FROM person_identity_memberships AS membership
                         INDEXED BY person_identity_memberships_owner_idx
                    JOIN people ON people.id = membership.person_id
                    WHERE membership.identity_key = ? AND membership.state = 'valid'
                    ORDER BY membership.person_id
                    LIMIT 1
                    """,
                    ("schema owner",),
                )
            )
        if not any(
            str(row[2]) == "people"
            and str(row[3]) == "person_id"
            and str(row[6]).upper() == "CASCADE"
            for row in foreign_keys
        ):
            raise SystemExit("malformed membership table was not rebuilt with its cascade FK")
        index_sql = " ".join(str(index_sql_row[0] if index_sql_row else "").upper().split())
        if "(IDENTITY_KEY, PERSON_ID)" not in index_sql or "WHERE STATE = 'VALID'" not in index_sql:
            raise SystemExit("wrong same-named membership owner index was not rebuilt")
        plan_details = tuple(str(row["detail"]).upper() for row in plan)
        if (
            not any(
                "SEARCH MEMBERSHIP USING" in item
                and "PERSON_IDENTITY_MEMBERSHIPS_OWNER_IDX" in item
                and "IDENTITY_KEY=?" in item.replace(" ", "")
                for item in plan_details
            )
            or any("SCAN " in item or "TEMP B-TREE" in item for item in plan_details)
        ):
            raise SystemExit(f"person owner query plan is not bounded SEARCH: {plan_details}")

        with store.connect() as conn:
            conn.execute("DELETE FROM people WHERE id = ?", (owner_id,))
            orphan = conn.execute(
                "SELECT 1 FROM person_identity_memberships WHERE person_id = ?", (owner_id,)
            ).fetchone()
        if orphan is not None:
            raise SystemExit("person deletion left an orphaned identity membership")


def test_clean_init_is_bounded_and_version_repair_is_exact() -> None:
    with TemporaryDirectory(prefix="jarvis-person-identity-init-bound-") as temp:
        store = MemoryStore(Path(temp) / "store.sqlite")
        store.init()
        ids = tuple(
            store.upsert_person(PersonRecord(name=f"Bounded Init Person {index}"))
            for index in range(4)
        )

        clean_statements: list[str] = []
        original_connect = store.connect
        store.connect = lambda: _traced_connections(  # type: ignore[method-assign]
            original_connect, clean_statements
        )
        store.init()
        store.connect = original_connect  # type: ignore[method-assign]
        normalized = [" ".join(statement.upper().split()) for statement in clean_statements]
        if any("SELECT ID, NAME FROM PEOPLE ORDER BY ID" in statement for statement in normalized):
            raise SystemExit("unchanged initialization performed an unbounded people identity scan")
        membership_writes = [
            statement
            for statement in normalized
            if statement.startswith(("INSERT ", "UPDATE ", "DELETE ", "REPLACE "))
            and "PERSON_IDENTITY_MEMBERSHIPS" in statement
        ]
        if membership_writes:
            raise SystemExit("unchanged initialization rewrote person identity memberships")

        db_path = store.db_path
        db_path.chmod(0o400)
        try:
            store.init()
        except Exception as exc:
            raise SystemExit(
                "clean person identity initialization should remain read-only compatible"
            ) from exc
        finally:
            db_path.chmod(0o600)
            for sidecar_suffix in ("-shm", "-wal"):
                sidecar_path = Path(f"{db_path}{sidecar_suffix}")
                if sidecar_path.exists():
                    sidecar_path.chmod(0o600)

        corrupt_id = ids[2]
        with store.connect() as conn:
            before = {
                int(row["person_id"]): tuple(row)
                for row in conn.execute(
                    "SELECT person_id, identity_key, normalizer_version, state, error_code "
                    "FROM person_identity_memberships ORDER BY person_id"
                )
            }
            conn.execute(
                "UPDATE person_identity_memberships SET normalizer_version = ? WHERE person_id = ?",
                ("corrupt-test-version", corrupt_id),
            )
        repair_statements: list[str] = []
        store.connect = lambda: _traced_connections(  # type: ignore[method-assign]
            original_connect, repair_statements
        )
        store.init()
        store.connect = original_connect  # type: ignore[method-assign]
        with store.connect() as conn:
            after = {
                int(row["person_id"]): tuple(row)
                for row in conn.execute(
                    "SELECT person_id, identity_key, normalizer_version, state, error_code "
                    "FROM person_identity_memberships ORDER BY person_id"
                )
            }
        if after.get(corrupt_id, (None, None, None))[2] != PERSON_IDENTITY_NORMALIZER_VERSION:
            raise SystemExit("normalizer-version corruption was not repaired")
        if after.get(corrupt_id) != before.get(corrupt_id):
            raise SystemExit("normalizer-version corruption was not repaired to its exact membership")
        if any(after[person_id] != before[person_id] for person_id in ids if person_id != corrupt_id):
            raise SystemExit("normalizer-version repair rewrote an unrelated membership")
        repair_normalized = [" ".join(statement.upper().split()) for statement in repair_statements]
        if any("SELECT ID, NAME FROM PEOPLE ORDER BY ID" in statement for statement in repair_normalized):
            raise SystemExit("one-row normalizer repair performed an unbounded people scan")


def test_legacy_interaction_refuses_shadow_without_writes() -> None:
    with TemporaryDirectory(prefix="jarvis-person-identity-legacy-log-") as temp:
        store = MemoryStore(Path(temp) / "store.sqlite")
        store.init()
        owner_id = store.upsert_person(PersonRecord(name="Legacy Log Person"))
        with store.connect() as conn:
            shadow_id = _insert_person_sql(conn, " legacy   log person ")
        resolved = store.get_person(name="legacy log person")
        if resolved is None or int(resolved["id"]) != owner_id:
            raise SystemExit("shadow fixture did not retain its canonical owner")
        exact_shadow = store.get_person(person_id=shadow_id)
        if exact_shadow is None or int(exact_shadow["id"]) != shadow_id:
            raise SystemExit("exact-ID person reads lost legacy compatibility")

        before = _identity_write_state(store, (owner_id, shadow_id))
        try:
            store.log_person_interaction(
                shadow_id,
                "PRIVATE-CONTACT-PAYLOAD",
                happened_at=NOW,
            )
        except ValueError:
            pass
        else:
            raise SystemExit("legacy interaction logging accepted a shadow person ID")
        after = _identity_write_state(store, (owner_id, shadow_id))
        if after != before:
            raise SystemExit("refused legacy shadow interaction performed writes")


def test_membership_tampering_repairs_and_fk_off_delete_cleans_up() -> None:
    with TemporaryDirectory(prefix="jarvis-person-identity-tamper-") as temp:
        db_path = Path(temp) / "store.sqlite"
        store = MemoryStore(db_path)
        store.init()
        deleted_membership_id = store.upsert_person(PersonRecord(name="Deleted Membership Person"))
        updated_membership_id = store.upsert_person(PersonRecord(name="Updated Membership Person"))

        with store.connect() as conn:
            conn.execute(
                "DELETE FROM person_identity_memberships WHERE person_id = ?",
                (deleted_membership_id,),
            )
        repaired_delete = store.get_person(name="deleted membership person")
        if repaired_delete is None or int(repaired_delete["id"]) != deleted_membership_id:
            raise SystemExit("manual membership deletion was not marked dirty and repaired")

        with store.connect() as conn:
            conn.execute(
                "UPDATE person_identity_memberships "
                "SET identity_key = ?, normalizer_version = ? WHERE person_id = ?",
                ("wrong-manual-identity", PERSON_IDENTITY_NORMALIZER_VERSION, updated_membership_id),
            )
        repaired_update = store.get_person(name="updated membership person")
        wrong_lookup = store.get_person(name="wrong manual identity")
        if repaired_update is None or int(repaired_update["id"]) != updated_membership_id:
            raise SystemExit("manual membership update was not marked dirty and repaired")
        if wrong_lookup is not None:
            raise SystemExit("manual membership corruption remained owner-addressable")

        raw = sqlite3.connect(db_path)
        raw.execute("PRAGMA foreign_keys = OFF")
        raw.execute("DELETE FROM people WHERE id = ?", (updated_membership_id,))
        raw.commit()
        identity_tables = tuple(
            str(row[0])
            for row in raw.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name LIKE 'person_identity%' ORDER BY name"
            )
        )
        orphan_tables = []
        for table in identity_tables:
            columns = {str(row[1]) for row in raw.execute(f'PRAGMA table_info("{table}")')}
            if "person_id" not in columns:
                continue
            orphan = raw.execute(
                f'SELECT 1 FROM "{table}" WHERE person_id = ? LIMIT 1',
                (updated_membership_id,),
            ).fetchone()
            if orphan is not None:
                orphan_tables.append(table)
        raw.close()
        if orphan_tables:
            raise SystemExit(
                "foreign-keys-off person deletion retained identity custody rows: "
                f"{orphan_tables}"
            )


def main() -> None:
    test_legacy_backfill_review_and_lifecycle()
    test_indexed_hot_paths_do_not_scan_people()
    test_concurrent_create_update_interaction_converges()
    test_post_init_direct_sql_is_reconciled_before_owner_resolution()
    test_direct_sql_count_and_review_reconcile_without_restart()
    test_trigger_ownership_preserves_unrelated_extras_during_repair()
    test_oversized_dirty_backlog_fails_closed_then_init_drains_in_batches()
    test_malformed_names_survive_and_are_quarantined_content_free()
    test_malformed_membership_schema_is_rebuilt_and_indexed()
    test_clean_init_is_bounded_and_version_repair_is_exact()
    test_legacy_interaction_refuses_shadow_without_writes()
    test_membership_tampering_repairs_and_fk_off_delete_cleans_up()
    print("Person identity ownership smoke passed")


if __name__ == "__main__":
    main()
