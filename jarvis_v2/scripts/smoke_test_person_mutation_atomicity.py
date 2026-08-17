from __future__ import annotations

import sqlite3
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Event

from jarvis_v2.memory.memory_projection import (
    reconcile_memory_projection,
    reconcile_pending_memory_projections,
)
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.person_projection import (
    reconcile_pending_person_projections,
    reconcile_person_projection,
)
from jarvis_v2.memory.store import (
    MemoryRecord,
    MemoryStore,
    PersonMutationTarget,
    PersonRecord,
)
from jarvis_v2.tools import people as people_module


GRAPH_TABLES = (
    "people",
    "memories",
    "person_projection_jobs",
    "memory_projection_jobs",
    "person_memory_links",
)


def _setup(root: Path) -> tuple[MemoryStore, ObsidianVault]:
    store = MemoryStore(root / "store.sqlite")
    store.init()
    vault = ObsidianVault(root / "vault")
    vault.init()
    return store, vault


def _record(store: MemoryStore, record: PersonRecord) -> PersonMutationTarget:
    target = store.record_person_with_projections(record)
    if not isinstance(target, PersonMutationTarget):
        raise SystemExit(
            "record_person_with_projections returned the wrong target type: "
            f"{type(target).__name__}"
        )
    return target


def _counts(store: MemoryStore) -> dict[str, int]:
    with store.connect() as conn:
        return {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in GRAPH_TABLES
        }


def _rows(store: MemoryStore, target: PersonMutationTarget):
    with store.connect() as conn:
        person = conn.execute(
            "SELECT * FROM people WHERE id = ?", (target.person_id,)
        ).fetchone()
        memory = conn.execute(
            "SELECT * FROM memories WHERE id = ?", (target.memory_id,)
        ).fetchone()
        person_job = conn.execute(
            "SELECT * FROM person_projection_jobs WHERE person_id = ?",
            (target.person_id,),
        ).fetchone()
        memory_job = conn.execute(
            "SELECT * FROM memory_projection_jobs WHERE memory_id = ?",
            (target.memory_id,),
        ).fetchone()
        link = conn.execute(
            "SELECT * FROM person_memory_links WHERE person_id = ?",
            (target.person_id,),
        ).fetchone()
    return person, memory, person_job, memory_job, link


def _assert_private_custody(
    target: PersonMutationTarget,
    person_job: sqlite3.Row,
    memory_job: sqlite3.Row,
    link: sqlite3.Row,
    *,
    forbidden: tuple[str, ...],
) -> None:
    rendered = "\n".join(
        (
            repr(target.person_projection_target),
            repr(target.memory_projection_target),
            repr(tuple(person_job)),
            repr(tuple(memory_job)),
            repr(tuple(link)),
        )
    )
    leaked = [value for value in forbidden if value and value in rendered]
    if leaked:
        raise SystemExit(f"person mutation custody leaked private source material: {leaked}")


def _assert_graph(
    store: MemoryStore,
    target: PersonMutationTarget,
    *,
    expected_body: str,
    expected_memory_revision: int,
) -> None:
    person, memory, person_job, memory_job, link = _rows(store, target)
    if any(row is None for row in (person, memory, person_job, memory_job, link)):
        raise SystemExit("person mutation omitted a source, projection, or custody row")
    person_target = target.person_projection_target
    memory_target = target.memory_projection_target
    if (
        target.person_id != person_target.person_id
        or target.memory_id != memory_target.memory_id
        or memory["body"] != expected_body
        or memory["revision"] != expected_memory_revision
        or link["person_id"] != target.person_id
        or link["memory_id"] != target.memory_id
        or person_job["state"] != "pending"
        or person_job["person_revision"] != person_target.revision
        or person_job["interaction_high_watermark"] != 0
        or person_job["source_digest"] != person_target.source_digest
        or memory_job["state"] != "pending"
        or memory_job["operation"] != "publish"
        or memory_job["memory_revision"] != expected_memory_revision
        or memory_job["memory_revision"] != memory_target.revision
        or memory_job["source_digest"] != memory_target.source_digest
    ):
        raise SystemExit("person mutation graph diverged from its exact returned targets")


def test_atomic_success_privacy_and_latest_insert_rollback() -> None:
    with TemporaryDirectory(prefix="jarvis-person-mutation-atomic-") as temp:
        root = Path(temp)
        store, vault = _setup(root / "success")
        name = "PRIVATE-PERSON-ATOMIC"
        notes = "PRIVATE-PERSON-NOTES-ATOMIC"
        files_before = {
            path.relative_to(vault.root_path) for path in vault.root_path.rglob("*.md")
        }
        target = _record(store, PersonRecord(name, "colleague", notes))
        if not target.created_person or target.person_name != name:
            raise SystemExit(f"fresh person mutation returned inaccurate identity: {target}")
        if _counts(store) != {table: 1 for table in GRAPH_TABLES}:
            raise SystemExit(f"fresh person mutation did not commit one exact graph: {_counts(store)}")
        _assert_graph(
            store,
            target,
            expected_body=f"Relation: colleague\n\nNotes: {notes}",
            expected_memory_revision=1,
        )
        _, _, person_job, memory_job, link = _rows(store, target)
        _assert_private_custody(
            target,
            person_job,
            memory_job,
            link,
            forbidden=(name, notes, str(root)),
        )
        files_after = {
            path.relative_to(vault.root_path) for path in vault.root_path.rglob("*.md")
        }
        if files_after != files_before:
            raise SystemExit("atomic database primitive wrote projection files")

        rollback_store, _ = _setup(root / "rollback")
        with rollback_store.connect() as conn:
            conn.execute(
                """
                CREATE TRIGGER reject_latest_person_mutation_insert
                BEFORE INSERT ON person_memory_links
                BEGIN
                    SELECT RAISE(ABORT, 'injected latest insert failure');
                END
                """
            )
        try:
            _record(
                rollback_store,
                PersonRecord("ROLLBACK-PERSON", "friend", "ROLLBACK-NOTES"),
            )
        except sqlite3.DatabaseError:
            pass
        else:
            raise SystemExit("latest person mutation insert failure did not abort")
        rollback_counts = _counts(rollback_store)
        if any(rollback_counts.values()):
            raise SystemExit(
                "latest insert failure left a person, memory, projection job, or link: "
                f"{rollback_counts}"
            )


def test_existing_person_reuses_canonical_memory_with_latest_revision() -> None:
    with TemporaryDirectory(prefix="jarvis-person-mutation-repeat-") as temp:
        root = Path(temp)
        store, _ = _setup(root)
        first = _record(
            store,
            PersonRecord("Casey Kim", "friend", "first private note"),
        )
        second_record = PersonRecord("  CASEY KIM  ", "teammate", "latest private note")
        second = _record(store, second_record)
        if (
            second.created_person
            or second.person_id != first.person_id
            or second.memory_id != first.memory_id
            or second.person_projection_target.revision
            != first.person_projection_target.revision + 1
            or second.memory_projection_target.revision
            != first.memory_projection_target.revision + 1
            or _counts(store) != {table: 1 for table in GRAPH_TABLES}
        ):
            raise SystemExit("repeat person mutation did not reuse one canonical person/memory graph")
        expected_body = (
            "Relation: teammate\n\nNotes: first private note\nlatest private note"
        )
        _assert_graph(
            store,
            second,
            expected_body=expected_body,
            expected_memory_revision=2,
        )
        person, memory, person_job, memory_job, link = _rows(store, second)
        if (
            person["name"] != "Casey Kim"
            or person["relation"] != "teammate"
            or person["revision"] != 1
            or memory["title"] != "Casey Kim"
            or memory_job["source_digest"] != second.memory_projection_target.source_digest
        ):
            raise SystemExit("repeat person mutation lost canonical identity or latest memory custody")
        _assert_private_custody(
            second,
            person_job,
            memory_job,
            link,
            forbidden=(second_record.name, second_record.notes, str(root)),
        )


def test_legacy_profile_memory_is_adopted_without_duplication() -> None:
    with TemporaryDirectory(prefix="jarvis-person-mutation-legacy-") as temp:
        store, _ = _setup(Path(temp))
        person_id = store.upsert_person(
            PersonRecord("Legacy Person", "friend", "legacy private note")
        )
        legacy_memory_id = store.add_memory(
            MemoryRecord(
                "people",
                "Legacy Person",
                "Relation: friend\n\nNotes: legacy private note",
                "people-log",
            )
        )
        target = _record(
            store,
            PersonRecord(" legacy person ", "colleague", "current private note"),
        )
        with store.connect() as conn:
            memories = list(conn.execute("SELECT * FROM memories ORDER BY id"))
            link = conn.execute(
                "SELECT * FROM person_memory_links WHERE person_id = ?", (person_id,)
            ).fetchone()
        if (
            target.person_id != person_id
            or target.memory_id != legacy_memory_id
            or len(memories) != 1
            or link is None
            or link["memory_id"] != legacy_memory_id
            or memories[0]["title"] != "Legacy Person"
            or memories[0]["revision"] != 2
            or memories[0]["body"]
            != "Relation: colleague\n\nNotes: legacy private note\ncurrent private note"
        ):
            raise SystemExit("pre-rollout profile memory was not adopted into canonical custody")


def test_direct_handler_refuses_success_when_concurrent_projection_is_pending() -> None:
    with TemporaryDirectory(prefix="jarvis-person-mutation-current-projection-") as temp:
        store, vault = _setup(Path(temp))
        add_person, _, _, _ = people_module.make_people_tools(store, vault)
        real_reconcile_memory = people_module.reconcile_memory_projection

        def supersede_before_memory(*_args, **_kwargs):
            store.record_person_with_projections(
                PersonRecord("Concurrent Person", "teammate", "newer private note")
            )
            return SimpleNamespace(status="superseded")

        people_module.reconcile_memory_projection = supersede_before_memory
        try:
            result = add_person(
                {
                    "name": "Concurrent Person",
                    "relation": "friend",
                    "notes": "initial private note",
                }
            )
        finally:
            people_module.reconcile_memory_projection = real_reconcile_memory
        person_job = store.get_person_projection_job(1)
        with store.connect() as conn:
            memory_job = conn.execute(
                "SELECT * FROM memory_projection_jobs ORDER BY memory_id LIMIT 1"
            ).fetchone()
        if (
            result.ok
            or not result.metadata.get("projection_pending")
            or person_job is None
            or person_job["state"] != "pending"
            or memory_job is None
            or memory_job["state"] != "pending"
            or "do not add the person again" not in result.output
        ):
            raise SystemExit("direct add_person overstated concurrent projection durability")


def test_concurrent_normalized_name_reuses_one_latest_memory() -> None:
    with TemporaryDirectory(prefix="jarvis-person-mutation-concurrent-") as temp:
        root = Path(temp)
        store, _ = _setup(root)
        records = (
            PersonRecord("Alex Park", "friend", "concurrent note zero"),
            PersonRecord(" alex park ", "colleague", "concurrent note one"),
            PersonRecord("ALEX PARK", "neighbor", "concurrent note two"),
            PersonRecord("Ａｌｅｘ Ｐａｒｋ", "teammate", "concurrent note three"),
        )
        barrier = Barrier(len(records))

        def mutate(record: PersonRecord) -> tuple[PersonRecord, PersonMutationTarget]:
            barrier.wait(timeout=10)
            return record, _record(store, record)

        with ThreadPoolExecutor(max_workers=len(records)) as pool:
            results = list(pool.map(mutate, records))
        targets = [target for _, target in results]
        if (
            len({target.person_id for target in targets}) != 1
            or len({target.memory_id for target in targets}) != 1
            or sum(target.created_person for target in targets) != 1
            or _counts(store) != {table: 1 for table in GRAPH_TABLES}
        ):
            raise SystemExit("normalized concurrent callers created duplicate people or profile memories")
        _, latest_target = max(
            results, key=lambda item: item[1].memory_projection_target.revision
        )
        with store.connect() as conn:
            committed_person = conn.execute(
                "SELECT relation, notes FROM people WHERE id = ?",
                (latest_target.person_id,),
            ).fetchone()
        expected_body = (
            f"Relation: {committed_person['relation'] or 'not captured'}"
            f"\n\nNotes: {committed_person['notes']}"
        )
        _assert_graph(
            store,
            latest_target,
            expected_body=expected_body,
            expected_memory_revision=len(records),
        )
        person, memory, person_job, memory_job, link = _rows(store, latest_target)
        if (
            person["revision"] != len(records) - 1
            or memory["title"] != person["name"]
            or sorted(target.memory_projection_target.revision for target in targets)
            != list(range(1, len(records) + 1))
        ):
            raise SystemExit("concurrent canonical memory revisions were not complete and monotonic")
        _assert_private_custody(
            latest_target,
            person_job,
            memory_job,
            link,
            forbidden=tuple(record.name for record in records)
            + tuple(record.notes for record in records)
            + (str(root),),
        )


def test_person_delete_preserves_unrelated_memory_until_tombstone_repair() -> None:
    with TemporaryDirectory(prefix="jarvis-person-mutation-delete-") as temp:
        root = Path(temp)
        store, vault = _setup(root)
        target = _record(
            store,
            PersonRecord("Delete Profile", "friend", "dedicated profile note"),
        )
        interaction = store.record_person_interaction_with_projections(
            person_id=target.person_id,
            name=target.person_name,
            summary="dedicated interaction note",
            happened_at="2026-07-12T10:00:00Z",
        )
        person_outcome = reconcile_person_projection(
            store, vault, interaction.person_projection_target
        )
        unrelated = store.add_memory_with_projection(
            MemoryRecord("fact", "Unrelated memory", "must remain", "smoke", 0.9)
        )
        dedicated_outcome = reconcile_memory_projection(
            store,
            vault,
            target.memory_id,
            expected_operation=target.memory_projection_target.operation,
            expected_revision=target.memory_projection_target.revision,
            expected_source_digest=target.memory_projection_target.source_digest,
        )
        interaction_outcome = reconcile_memory_projection(
            store,
            vault,
            interaction.memory_id,
            expected_operation=interaction.memory_projection_target.operation,
            expected_revision=interaction.memory_projection_target.revision,
            expected_source_digest=interaction.memory_projection_target.source_digest,
        )
        unrelated_outcome = reconcile_memory_projection(
            store,
            vault,
            unrelated.memory_id,
            expected_operation=unrelated.operation,
            expected_revision=unrelated.revision,
            expected_source_digest=unrelated.source_digest,
        )
        dedicated_note = vault.root_path / dedicated_outcome.path_display
        interaction_note = vault.root_path / interaction_outcome.path_display
        unrelated_note = vault.root_path / unrelated_outcome.path_display
        person_notes = list((vault.root_path / "People").glob("*.md"))
        if (
            person_outcome.status not in {"completed", "superseded"}
            or len(person_notes) != 1
            or not dedicated_note.exists()
            or not interaction_note.exists()
            or not unrelated_note.exists()
        ):
            raise SystemExit("delete fixture did not publish all memory notes")

        with store.connect() as conn:
            conn.execute("DELETE FROM people WHERE id = ?", (target.person_id,))
        with store.connect() as conn:
            delete_jobs = list(
                conn.execute(
                    "SELECT * FROM memory_projection_jobs WHERE memory_id IN (?, ?) ORDER BY memory_id",
                    (target.memory_id, interaction.memory_id),
                )
            )
            unrelated_row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (unrelated.memory_id,)
            ).fetchone()
        if (
            store.get_person(person_id=target.person_id) is not None
            or store.get_memory(target.memory_id) is not None
            or store.get_memory(interaction.memory_id) is not None
            or unrelated_row is None
            or _counts(store)["person_memory_links"] != 0
            or len(delete_jobs) != 2
            or any(job["operation"] != "delete" for job in delete_jobs)
            or any(job["state"] != "pending" for job in delete_jobs)
            or not dedicated_note.exists()
            or not interaction_note.exists()
            or not unrelated_note.exists()
        ):
            raise SystemExit("person deletion did not retain both exact pending delete tombstones")

        summary = reconcile_pending_memory_projections(store, vault, limit=10)
        person_summary = reconcile_pending_person_projections(store, vault, limit=10)
        repaired_jobs = [
            store.get_memory_projection_job(memory_id)
            for memory_id in (target.memory_id, interaction.memory_id)
        ]
        if (
            summary.completed != 2
            or summary.pending != 0
            or person_summary.completed != 1
            or person_summary.pending != 0
            or list((vault.root_path / "People").glob("*.md"))
            or dedicated_note.exists()
            or interaction_note.exists()
            or not unrelated_note.exists()
            or store.get_memory(unrelated.memory_id) is None
            or any(job is None for job in repaired_jobs)
            or any(job["operation"] != "delete" for job in repaired_jobs)
            or any(job["state"] != "completed" for job in repaired_jobs)
        ):
            raise SystemExit("generic memory reconciliation did not erase both person-owned notes")


def test_delete_reopens_after_stale_publisher_race() -> None:
    with TemporaryDirectory(prefix="jarvis-person-mutation-delete-race-") as temp:
        store, vault = _setup(Path(temp))
        target = _record(
            store,
            PersonRecord("Race Person", "friend", "race private note"),
        )
        publisher_ready = Event()
        allow_publish = Event()
        real_write = vault.write_person_with_evidence

        def paused_write(*args, **kwargs):
            publisher_ready.set()
            if not allow_publish.wait(timeout=10):
                raise RuntimeError("stale publisher smoke timed out")
            return real_write(*args, **kwargs)

        vault.write_person_with_evidence = paused_write
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    reconcile_person_projection,
                    store,
                    vault,
                    target.person_projection_target,
                )
                if not publisher_ready.wait(timeout=10):
                    raise SystemExit("stale publisher did not reach the controlled race")
                with store.connect() as conn:
                    conn.execute("DELETE FROM people WHERE id = ?", (target.person_id,))
                first_delete = reconcile_pending_person_projections(store, vault, limit=10)
                if first_delete.completed != 1 or first_delete.pending != 0:
                    raise SystemExit("initial absent person deletion did not complete")
                allow_publish.set()
                outcome = future.result(timeout=10)
        finally:
            allow_publish.set()
            vault.write_person_with_evidence = real_write

        delete_job = store.get_person_projection_delete_job(target.person_id)
        if (
            outcome.status != "completed"
            or list((vault.root_path / "People").glob("*.md"))
            or store.count_pending_person_projection_delete_jobs() != 0
            or delete_job is None
            or delete_job["state"] != "completed"
            or delete_job["completion_status"] != "deleted"
        ):
            raise SystemExit("stale person publisher recreated an orphan after deletion")


def test_schema_init_idempotency_unique_link_and_fk_rejection() -> None:
    with TemporaryDirectory(prefix="jarvis-person-mutation-schema-") as temp:
        root = Path(temp)
        store, _ = _setup(root)
        target = _record(store, PersonRecord("Schema Person", "friend", "schema note"))
        unlinked_memory_id = store.add_memory(
            MemoryRecord("fact", "Unlinked schema memory", "unlinked", "smoke", 0.8)
        )
        before = _counts(store)
        store.init()
        store.init()
        if _counts(store) != before:
            raise SystemExit("repeated schema initialization changed person mutation rows")
        with store.connect() as conn:
            unique_person_indexes = []
            for index in conn.execute("PRAGMA index_list(person_memory_links)"):
                if int(index["unique"]) != 1:
                    continue
                columns = tuple(
                    row["name"]
                    for row in conn.execute(f"PRAGMA index_info('{index['name']}')")
                )
                if columns == ("person_id",):
                    unique_person_indexes.append(str(index["name"]))
            trigger_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger' "
                    "AND name = 'people_delete_dedicated_memories'"
                ).fetchone()[0]
            )
            fk_violations = list(conn.execute("PRAGMA foreign_key_check"))
            for memory_id, person_id in (
                (9223372036854775805, target.person_id),
                (unlinked_memory_id, 9223372036854775806),
            ):
                try:
                    conn.execute(
                        "INSERT INTO person_memory_links("
                        "memory_id, person_id, created_at, updated_at"
                        ") VALUES (?, ?, 'now', 'now')",
                        (memory_id, person_id),
                    )
                except sqlite3.IntegrityError:
                    pass
                else:
                    raise SystemExit("person-memory custody foreign key accepted an orphan")
            try:
                conn.execute(
                    "INSERT INTO person_memory_links("
                    "memory_id, person_id, created_at, updated_at"
                    ") VALUES (?, ?, 'now', 'now')",
                    (unlinked_memory_id, target.person_id),
                )
            except sqlite3.IntegrityError:
                pass
            else:
                raise SystemExit("person-memory custody accepted a second memory for one person")
        if len(unique_person_indexes) != 1 or trigger_count != 1 or fk_violations:
            raise SystemExit(
                "person-memory schema was not idempotent, one-to-one, and FK-clean: "
                f"indexes={unique_person_indexes}, trigger_count={trigger_count}, "
                f"violations={fk_violations}"
            )


def main() -> None:
    test_atomic_success_privacy_and_latest_insert_rollback()
    test_existing_person_reuses_canonical_memory_with_latest_revision()
    test_legacy_profile_memory_is_adopted_without_duplication()
    test_direct_handler_refuses_success_when_concurrent_projection_is_pending()
    test_concurrent_normalized_name_reuses_one_latest_memory()
    test_person_delete_preserves_unrelated_memory_until_tombstone_repair()
    test_delete_reopens_after_stale_publisher_race()
    test_schema_init_idempotency_unique_link_and_fk_rejection()
    print("Person mutation atomicity smoke passed")


if __name__ == "__main__":
    main()
