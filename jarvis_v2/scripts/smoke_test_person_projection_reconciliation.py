from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.memory.memory_projection import reconcile_pending_memory_projections
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.person_projection import (
    reconcile_pending_person_projections,
    reconcile_person_projection,
)
from jarvis_v2.memory.store import (
    InteractionMutationTarget,
    MemoryStore,
)


PERSON_PROJECTION_METHODS = (
    "record_person_interaction_with_projections",
    "get_person_projection_job",
    "ensure_current_person_projection_job",
    "mark_person_projection_error",
    "complete_person_projection",
    "list_pending_person_projection_jobs",
    "count_pending_person_projection_jobs",
    "get_current_person_projection_snapshot",
)


class SimulatedProcessDeath(BaseException):
    pass


def _setup(root: Path) -> tuple[MemoryStore, ObsidianVault]:
    store = MemoryStore(root / "store.sqlite")
    store.init()
    vault = ObsidianVault(root / "vault")
    vault.init()
    return store, vault


def _assert_store_contract() -> None:
    missing = [name for name in PERSON_PROJECTION_METHODS if not hasattr(MemoryStore, name)]
    if missing:
        raise SystemExit("missing MemoryStore person projection APIs: " + ", ".join(missing))


def _record(
    store: MemoryStore,
    *,
    name: str,
    summary: str,
    happened_at: str = "2026-07-12T09:30:00Z",
    person_id: int | None = None,
) -> InteractionMutationTarget:
    result = store.record_person_interaction_with_projections(
        person_id=person_id,
        name=name,
        summary=summary,
        happened_at=happened_at,
    )
    if not isinstance(result, InteractionMutationTarget):
        raise SystemExit(
            "record_person_interaction_with_projections did not return "
            f"InteractionMutationTarget: {type(result).__name__}"
        )
    return result


def _table_count(store: MemoryStore, table: str) -> int:
    allowed = {
        "people",
        "person_interactions",
        "memories",
        "memory_projection_jobs",
        "interaction_memory_links",
        "person_projection_jobs",
    }
    if table not in allowed:
        raise ValueError(f"unsupported smoke table: {table}")
    with store.connect() as conn:
        row = conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
    return int(row["count"])


def _graph_counts(store: MemoryStore) -> dict[str, int]:
    return {table: _table_count(store, table) for table in (
        "people",
        "person_interactions",
        "memories",
        "memory_projection_jobs",
        "interaction_memory_links",
        "person_projection_jobs",
    )}


def _assert_atomic_graph(store: MemoryStore, target: InteractionMutationTarget) -> None:
    person_target = target.person_projection_target
    memory_target = target.memory_projection_target
    if (
        target.person_id != person_target.person_id
        or target.memory_id != memory_target.memory_id
        or person_target.interaction_high_watermark != target.interaction_id
        or memory_target.operation != "publish"
    ):
        raise SystemExit(f"atomic interaction returned incoherent targets: {target}")
    with store.connect() as conn:
        person = conn.execute("SELECT * FROM people WHERE id = ?", (target.person_id,)).fetchone()
        interaction = conn.execute(
            "SELECT * FROM person_interactions WHERE id = ?", (target.interaction_id,)
        ).fetchone()
        memory = conn.execute("SELECT * FROM memories WHERE id = ?", (target.memory_id,)).fetchone()
        link = conn.execute(
            "SELECT * FROM interaction_memory_links WHERE interaction_id = ?",
            (target.interaction_id,),
        ).fetchone()
        person_job = conn.execute(
            "SELECT * FROM person_projection_jobs WHERE person_id = ?", (target.person_id,)
        ).fetchone()
        memory_job = conn.execute(
            "SELECT * FROM memory_projection_jobs WHERE memory_id = ?", (target.memory_id,)
        ).fetchone()
    if person is None or interaction is None or memory is None or link is None:
        raise SystemExit("atomic interaction omitted a source row or custody link")
    if person_job is None or memory_job is None:
        raise SystemExit("atomic interaction omitted a projection job")
    if (
        interaction["person_id"] != target.person_id
        or link["memory_id"] != target.memory_id
        or person_job["state"] != "pending"
        or memory_job["state"] != "pending"
        or person_job["person_revision"] != person_target.revision
        or person_job["interaction_high_watermark"] != target.interaction_id
        or person_job["source_digest"] != person_target.source_digest
        or memory_job["memory_revision"] != memory_target.revision
        or memory_job["source_digest"] != memory_target.source_digest
    ):
        raise SystemExit("atomic interaction source rows and projection targets diverged")


def _assert_private(values: Sequence[object], forbidden: Sequence[str], label: str) -> None:
    rendered = "\n".join(str(value or "") for value in values)
    leaked = [secret for secret in forbidden if secret and secret in rendered]
    if leaked:
        raise SystemExit(f"{label} leaked raw source/private path material: {leaked}")


def _opaque_person_path_display(person_id: int) -> str:
    return f"People/person-{person_id}.md"


def _assert_opaque_person_path_display(value: object, person_id: int, label: str) -> None:
    expected = _opaque_person_path_display(person_id)
    if value != expected:
        raise SystemExit(f"{label} was not opaque and ID-bound: {value!r} / {expected!r}")


def _frontmatter_person_id(path: Path) -> int | None:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end < 0:
        return None
    projection = None
    person_id = None
    for line in text[4:end].splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        if key.strip() == "jarvis_projection":
            projection = value.strip()
        elif key.strip() == "id" and value.strip().isdigit():
            person_id = int(value.strip())
    return person_id if projection == "person" else None


def _frontmatter_value(path: Path, key: str) -> str | None:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end < 0:
        return None
    for line in text[4:end].splitlines():
        candidate, separator, value = line.partition(":")
        if separator and candidate.strip() == key:
            return value.strip()
    return None


def _find_person_note(vault: ObsidianVault, person_id: int, label: str) -> Path:
    candidates = list((vault.root_path / "People").glob("*.md"))
    frontmatter_matches = [
        path for path in candidates if _frontmatter_person_id(path) == person_id
    ]
    suffix_matches = [
        path for path in candidates if path.name.endswith(f" [{person_id}].md")
    ]
    matches = frontmatter_matches or suffix_matches
    if len(matches) != 1:
        raise SystemExit(f"{label} expected one ID-bound person note: {matches}")
    return matches[0]


def _assert_job_privacy(
    store: MemoryStore,
    target: InteractionMutationTarget,
    *,
    forbidden: Sequence[str],
) -> None:
    with store.connect() as conn:
        person_job = conn.execute(
            "SELECT * FROM person_projection_jobs WHERE person_id = ?", (target.person_id,)
        ).fetchone()
        memory_job = conn.execute(
            "SELECT * FROM memory_projection_jobs WHERE memory_id = ?", (target.memory_id,)
        ).fetchone()
        person_columns = {
            str(row["name"]) for row in conn.execute("PRAGMA table_info(person_projection_jobs)")
        }
    if person_job is None or memory_job is None:
        raise SystemExit("privacy fixture lost a projection job")
    if {"name", "summary", "notes", "relation"} & person_columns:
        raise SystemExit("person projection ledger added raw source-content columns")
    if person_job["path_display"] is not None:
        _assert_opaque_person_path_display(
            person_job["path_display"], target.person_id, "person projection job path_display"
        )
    _assert_private(tuple(person_job), forbidden, "person projection job")
    _assert_private(tuple(memory_job), forbidden, "memory projection job")


def _snapshot_parts(snapshot) -> tuple[object, tuple[object, ...]]:
    if isinstance(snapshot, tuple) and len(snapshot) == 2:
        person, interactions = snapshot
    elif isinstance(snapshot, Mapping):
        person, interactions = snapshot["person"], snapshot["interactions"]
    else:
        person, interactions = snapshot.person, snapshot.interactions
    if isinstance(interactions, (str, bytes)) or not isinstance(interactions, Sequence):
        raise SystemExit("person projection snapshot has an unsupported shape")
    return person, tuple(interactions)


def _current_snapshot(store: MemoryStore, target: InteractionMutationTarget):
    projection = target.person_projection_target
    snapshot = store.get_current_person_projection_snapshot(
        projection.person_id,
        projection.revision,
        projection.interaction_high_watermark,
        projection.source_digest,
    )
    if snapshot is None:
        raise SystemExit("exact person projection snapshot was unavailable")
    return _snapshot_parts(snapshot)


def _complete_person(
    store: MemoryStore,
    vault: ObsidianVault,
    target: InteractionMutationTarget,
):
    outcome = reconcile_person_projection(
        store,
        vault,
        target.person_id,
        target=target.person_projection_target,
    )
    if outcome.status != "completed":
        raise SystemExit(f"person projection did not complete: {outcome}")
    return outcome


def test_atomic_graph_privacy_and_late_rollback() -> None:
    with TemporaryDirectory(prefix="jarvis-person-projection-atomic-") as temp:
        root = Path(temp)
        store, vault = _setup(root / "success")
        name = "RAW-PERSON-NAME-ATOMIC"
        summary = "RAW-INTERACTION-SUMMARY-ATOMIC"
        files_before = {path.relative_to(vault.root_path) for path in vault.root_path.rglob("*.md")}
        target = _record(store, name=name, summary=summary)
        if not target.created_person or target.person_name != name:
            raise SystemExit(f"new-person atomic result was inaccurate: {target}")
        _assert_atomic_graph(store, target)
        files_after = {path.relative_to(vault.root_path) for path in vault.root_path.rglob("*.md")}
        if files_after != files_before:
            raise SystemExit("atomic store primitive wrote projection files")
        _assert_job_privacy(store, target, forbidden=(name, summary, str(root)))

        rollback_store, _ = _setup(root / "rollback")
        with rollback_store.connect() as conn:
            conn.execute(
                """
                CREATE TRIGGER reject_interaction_custody_link
                BEFORE INSERT ON interaction_memory_links
                BEGIN
                    SELECT RAISE(ABORT, 'injected late custody failure');
                END
                """
            )
        try:
            _record(
                rollback_store,
                name="ROLLBACK-PERSON-MUST-NOT-EXIST",
                summary="ROLLBACK-SUMMARY-MUST-NOT-EXIST",
            )
        except sqlite3.DatabaseError:
            pass
        else:
            raise SystemExit("late interaction custody failure did not abort the mutation")
        counts = _graph_counts(rollback_store)
        if any(counts.values()):
            raise SystemExit(f"late projection failure left a partial DB graph: {counts}")


def test_exact_evidence_write_failure_and_crash_adoption() -> None:
    with TemporaryDirectory(prefix="jarvis-person-projection-evidence-") as temp:
        root = Path(temp)
        store, vault = _setup(root / "exact")
        name = "PRIVATE-PERSON-EXACT"
        summary = "PRIVATE-SUMMARY-EXACT"
        target = _record(store, name=name, summary=summary)
        outcome = _complete_person(store, vault, target)
        job = store.get_person_projection_job(target.person_id)
        if job is None or job["state"] != "completed":
            raise SystemExit("person projection completion was not durable")
        _assert_opaque_person_path_display(
            outcome.path_display, target.person_id, "person projection outcome path_display"
        )
        note = _find_person_note(vault, target.person_id, "exact completion")
        text = note.read_text(encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        person_target = target.person_projection_target
        if (
            job["path_display"] != outcome.path_display
            or job["content_digest"] != outcome.content_digest
            or outcome.content_digest != digest
            or f"id: {target.person_id}" not in text
            or f"source_revision: {person_target.revision}" not in text
            or f"interaction_high_watermark: {target.interaction_id}" not in text
            or name not in text
            or summary not in text
        ):
            raise SystemExit("completed person projection evidence diverged from exact source")
        _assert_private(
            tuple(job) + tuple(vars(outcome).values()),
            (name, summary, str(root)),
            "completed person job/outcome",
        )

        failing_store, failing_vault = _setup(root / "write-failure")
        failing_target = _record(
            failing_store,
            name="PRIVATE-PERSON-RETRY",
            summary="PRIVATE-SUMMARY-RETRY",
        )
        before = _graph_counts(failing_store)
        original_write = failing_vault.write_person_with_evidence

        def fail_write(*_args, **_kwargs):
            raise OSError("injected person projection write failure")

        failing_vault.write_person_with_evidence = fail_write  # type: ignore[method-assign]
        failed = reconcile_person_projection(
            failing_store,
            failing_vault,
            failing_target.person_id,
            target=failing_target.person_projection_target,
        )
        failing_vault.write_person_with_evidence = original_write  # type: ignore[method-assign]
        failed_job = failing_store.get_person_projection_job(failing_target.person_id)
        if (
            failed.status != "pending_error"
            or failed_job is None
            or failed_job["state"] != "pending"
            or failed_job["last_error_code"] != "vault_publish_failed"
            or _graph_counts(failing_store) != before
        ):
            raise SystemExit(f"write failure did not retain exact repair custody: {failed}")
        repaired = _complete_person(failing_store, failing_vault, failing_target)
        if repaired.person_id != failing_target.person_id or _graph_counts(failing_store) != before:
            raise SystemExit("person projection retry recreated source rows or changed IDs")
        _assert_opaque_person_path_display(
            repaired.path_display, repaired.person_id, "repaired person outcome path_display"
        )
        _assert_job_privacy(
            failing_store,
            failing_target,
            forbidden=("PRIVATE-PERSON-RETRY", "PRIVATE-SUMMARY-RETRY", str(root)),
        )
        _assert_private(
            tuple(vars(repaired).values()),
            ("PRIVATE-PERSON-RETRY", "PRIVATE-SUMMARY-RETRY", str(root)),
            "repaired person projection outcome",
        )

        crash_store, crash_vault = _setup(root / "cas-crash")
        crash_target = _record(
            crash_store,
            name="PRIVATE-PERSON-CAS",
            summary="PRIVATE-SUMMARY-CAS",
        )
        person, interactions = _current_snapshot(crash_store, crash_target)
        crash_job = crash_store.get_person_projection_job(crash_target.person_id)
        crash_path, crash_digest = crash_vault.write_person_with_evidence(
            person,
            interactions,
            store_identity=str(crash_job["store_identity"]),
        )
        before_bytes = crash_path.read_bytes()
        if crash_job["state"] != "pending":
            raise SystemExit("file-before-CAS fixture completed the DB job early")
        adopted = _complete_person(crash_store, crash_vault, crash_target)
        adopted_job = crash_store.get_person_projection_job(crash_target.person_id)
        _assert_opaque_person_path_display(
            adopted.path_display, adopted.person_id, "adopted person outcome path_display"
        )
        if (
            crash_path.read_bytes() != before_bytes
            or adopted.content_digest != crash_digest
            or adopted_job["content_digest"] != crash_digest
            or _find_person_note(crash_vault, crash_target.person_id, "crash adoption")
            != crash_path
            or _graph_counts(crash_store) != {
                "people": 1,
                "person_interactions": 1,
                "memories": 1,
                "memory_projection_jobs": 1,
                "interaction_memory_links": 1,
                "person_projection_jobs": 1,
            }
        ):
            raise SystemExit("file-before-CAS crash was not adopted idempotently")
        _assert_job_privacy(
            crash_store,
            crash_target,
            forbidden=("PRIVATE-PERSON-CAS", "PRIVATE-SUMMARY-CAS", str(root)),
        )
        _assert_private(
            tuple(vars(adopted).values()),
            ("PRIVATE-PERSON-CAS", "PRIVATE-SUMMARY-CAS", str(root)),
            "adopted person projection outcome",
        )


def test_supersession_bounded_repair_and_malformed_fail_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-person-projection-repair-") as temp:
        root = Path(temp)
        store, vault = _setup(root / "supersession")
        name = "PRIVATE-PERSON-SUPERSEDE"
        first = _record(store, name=name, summary="OLDER-SUMMARY-MUST-STAY-STALE")
        old_person, old_interactions = _current_snapshot(store, first)
        second = _record(
            store,
            name=name,
            summary="NEWEST-SUMMARY-MUST-WIN",
            happened_at="2026-07-12T10:30:00Z",
        )
        if second.created_person or second.person_id != first.person_id:
            raise SystemExit("second interaction did not resolve the existing person")
        superseded = reconcile_person_projection(
            store,
            vault,
            first.person_id,
            target=first.person_projection_target,
        )
        latest_job = store.get_person_projection_job(first.person_id)
        if (
            superseded.status != "superseded"
            or latest_job["state"] != "completed"
            or latest_job["person_revision"] != second.person_projection_target.revision
            or latest_job["interaction_high_watermark"] != second.interaction_id
            or latest_job["source_digest"] != second.person_projection_target.source_digest
        ):
            raise SystemExit(f"stale target did not converge to the newest snapshot: {superseded}")
        _assert_opaque_person_path_display(
            superseded.path_display,
            superseded.person_id,
            "superseded person outcome path_display",
        )
        latest_path = _find_person_note(vault, superseded.person_id, "superseded repair")
        latest_bytes = latest_path.read_bytes()
        latest_text = latest_bytes.decode("utf-8")
        if "NEWEST-SUMMARY-MUST-WIN" not in latest_text:
            raise SystemExit("newest interaction was absent from superseded repair")
        try:
            vault.write_person_with_evidence(
                old_person,
                old_interactions,
                store_identity=str(latest_job["store_identity"]),
            )
        except RuntimeError:
            pass
        else:
            raise SystemExit("stale person snapshot overwrote a newer interaction snapshot")
        if latest_path.read_bytes() != latest_bytes:
            raise SystemExit("stale person write changed the newer projection bytes")
        supersession_forbidden = (
            name,
            "OLDER-SUMMARY-MUST-STAY-STALE",
            "NEWEST-SUMMARY-MUST-WIN",
            str(root),
        )
        _assert_job_privacy(store, second, forbidden=supersession_forbidden)
        _assert_private(
            tuple(vars(superseded).values()),
            supersession_forbidden,
            "superseded person projection outcome",
        )

        batch_store, batch_vault = _setup(root / "bounded")
        targets = [
            _record(
                batch_store,
                name=f"PRIVATE-BATCH-PERSON-{index}",
                summary=f"PRIVATE-BATCH-SUMMARY-{index}",
            )
            for index in range(3)
        ]
        summary = reconcile_pending_person_projections(batch_store, batch_vault, limit=2)
        if (
            summary.attempted != 2
            or summary.completed != 2
            or summary.pending != 1
            or len(summary.outcomes) != 2
            or batch_store.count_pending_person_projection_jobs() != 1
        ):
            raise SystemExit(f"bounded person repair summary was inaccurate: {summary}")
        final = reconcile_pending_person_projections(batch_store, batch_vault, limit=1000)
        if final.attempted != 1 or final.completed != 1 or final.pending != 0:
            raise SystemExit(f"bounded person repair did not converge on the next pass: {final}")
        for index, target in enumerate(targets):
            _assert_job_privacy(
                batch_store,
                target,
                forbidden=(
                    target.person_name,
                    f"PRIVATE-BATCH-SUMMARY-{index}",
                    str(root),
                ),
            )
        for outcome in summary.outcomes + final.outcomes:
            _assert_opaque_person_path_display(
                outcome.path_display,
                outcome.person_id,
                "bounded person outcome path_display",
            )
        _assert_private(
            tuple(value for outcome in summary.outcomes + final.outcomes for value in vars(outcome).values()),
            tuple(target.person_name for target in targets)
            + tuple(f"PRIVATE-BATCH-SUMMARY-{index}" for index in range(3))
            + (str(root),),
            "bounded person projection outcomes",
        )

        malformed_store, malformed_vault = _setup(root / "malformed")
        malformed_target = _record(
            malformed_store,
            name="PRIVATE-MALFORMED-PERSON",
            summary="PRIVATE-MALFORMED-SUMMARY",
        )
        write_calls = 0

        def count_write(*_args, **_kwargs):
            nonlocal write_calls
            write_calls += 1
            raise AssertionError("malformed job reached the vault")

        malformed_vault.write_person_with_evidence = count_write  # type: ignore[method-assign]
        malformed_store.get_person_projection_job = lambda _person_id: {  # type: ignore[method-assign]
            "person_id": malformed_target.person_id,
            "person_revision": "not-an-integer",
            "interaction_high_watermark": malformed_target.interaction_id,
            "store_identity": "f" * 32,
            "source_digest": "0" * 64,
            "state": "pending",
        }
        malformed = reconcile_person_projection(
            malformed_store,
            malformed_vault,
            malformed_target.person_id,
        )
        if malformed.status != "malformed" or write_calls != 0:
            raise SystemExit(f"malformed person projection job did not fail closed: {malformed}")

        mixed_store, mixed_vault = _setup(root / "malformed-mixed")
        malformed_front = _record(
            mixed_store,
            name="PRIVATE-MALFORMED-FRONT",
            summary="PRIVATE-MALFORMED-FRONT-SUMMARY",
        )
        healthy_later = _record(
            mixed_store,
            name="PRIVATE-HEALTHY-LATER",
            summary="PRIVATE-HEALTHY-LATER-SUMMARY",
        )
        real_get = mixed_store.get_person_projection_job
        real_list = mixed_store.list_pending_person_projection_jobs
        malformed_row = dict(real_get(malformed_front.person_id))
        malformed_row["person_revision"] = "not-an-integer"
        healthy_row = real_get(healthy_later.person_id)
        mixed_store.list_pending_person_projection_jobs = (  # type: ignore[method-assign]
            lambda limit=50: [malformed_row, healthy_row][:limit]
        )
        mixed_store.get_person_projection_job = (  # type: ignore[method-assign]
            lambda person_id: malformed_row
            if person_id == malformed_front.person_id
            else real_get(person_id)
        )
        mixed = reconcile_pending_person_projections(mixed_store, mixed_vault, limit=2)
        mixed_store.get_person_projection_job = real_get  # type: ignore[method-assign]
        mixed_store.list_pending_person_projection_jobs = real_list  # type: ignore[method-assign]
        malformed_after = real_get(malformed_front.person_id)
        healthy_after = real_get(healthy_later.person_id)
        if (
            tuple(outcome.status for outcome in mixed.outcomes) != ("malformed", "completed")
            or malformed_after["state"] != "pending"
            or malformed_after["last_error_code"] != "malformed_job"
            or healthy_after["state"] != "completed"
        ):
            raise SystemExit(f"malformed front row starved bounded healthy work: {mixed}")


def test_global_interaction_watermark_with_happened_at_render_cap() -> None:
    with TemporaryDirectory(prefix="jarvis-person-projection-watermark-") as temp:
        root = Path(temp)
        store, vault = _setup(root)
        name = "PRIVATE-WATERMARK-PERSON"
        targets = [
            _record(
                store,
                name=name,
                summary=f"CHRONOLOGICAL-SUMMARY-{index:02d}",
                happened_at=f"2026-07-{index + 1:02d}T09:30:00Z",
            )
            for index in range(11)
        ]
        high_id_old_event = _record(
            store,
            name=name,
            summary="HIGH-ID-OLD-HAPPENED-AT-MUST-NOT-RENDER",
            happened_at="2025-01-01T09:30:00Z",
        )
        expected_rendered_ids = tuple(target.interaction_id for target in reversed(targets[1:]))
        person, interactions = _current_snapshot(store, high_id_old_event)
        rendered_ids = tuple(int(interaction["id"]) for interaction in interactions)
        job = store.get_person_projection_job(high_id_old_event.person_id)
        if (
            person["id"] != high_id_old_event.person_id
            or len(interactions) != 10
            or rendered_ids != expected_rendered_ids
            or high_id_old_event.interaction_id != max(
                target.interaction_id for target in targets + [high_id_old_event]
            )
            or high_id_old_event.interaction_id in rendered_ids
            or high_id_old_event.person_projection_target.interaction_high_watermark
            != high_id_old_event.interaction_id
            or job is None
            or job["interaction_high_watermark"] != high_id_old_event.interaction_id
        ):
            raise SystemExit(
                "global interaction watermark diverged from happened_at-capped snapshot: "
                f"target={high_id_old_event.person_projection_target}, "
                f"rendered_ids={rendered_ids}, job={dict(job) if job is not None else None}"
            )

        _complete_person(store, vault, high_id_old_event)
        note = _find_person_note(vault, high_id_old_event.person_id, "watermark projection")
        note_text = note.read_text(encoding="utf-8")
        if (
            _frontmatter_value(note, "interaction_high_watermark")
            != str(high_id_old_event.interaction_id)
            or "HIGH-ID-OLD-HAPPENED-AT-MUST-NOT-RENDER" in note_text
            or "CHRONOLOGICAL-SUMMARY-00" in note_text
            or any(
                f"CHRONOLOGICAL-SUMMARY-{index:02d}" not in note_text
                for index in range(1, 11)
            )
        ):
            raise SystemExit(
                "person note did not preserve the global watermark with the latest-10 render cap"
            )


def test_legacy_people_bounded_job_seeding_and_convergence() -> None:
    with TemporaryDirectory(prefix="jarvis-person-projection-legacy-") as temp:
        root = Path(temp)
        store, vault = _setup(root)
        legacy_count = 5
        with store.connect() as conn:
            conn.executemany(
                """
                INSERT INTO people(
                    name, relation, notes, last_contact_at, revision, created_at, updated_at
                ) VALUES (?, '', '', NULL, 0, ?, ?)
                """,
                [
                    (
                        f"PRIVATE-LEGACY-PERSON-{index}",
                        f"2026-07-12T09:3{index}:00Z",
                        f"2026-07-12T09:3{index}:00Z",
                    )
                    for index in range(legacy_count)
                ],
            )
        if _table_count(store, "person_projection_jobs") != 0:
            raise SystemExit("legacy fixture unexpectedly created projection jobs")

        fresh_store = MemoryStore(root / "store.sqlite")
        fresh_store.init()
        if (
            _table_count(fresh_store, "person_projection_jobs") != 0
            or fresh_store.count_pending_person_projection_jobs() != legacy_count
        ):
            raise SystemExit("MemoryStore initialization eagerly seeded legacy people")

        limit = 2
        seeded = fresh_store.ensure_missing_person_projection_jobs(limit=limit)
        if (
            seeded != limit
            or _table_count(fresh_store, "person_projection_jobs") != limit
            or len(fresh_store.list_pending_person_projection_jobs(limit=100)) != limit
            or fresh_store.count_pending_person_projection_jobs() != legacy_count
        ):
            raise SystemExit(
                "bounded legacy seeding or mixed materialized/missing pending count was inaccurate"
            )

        first = reconcile_pending_person_projections(fresh_store, vault, limit=limit)
        if (
            first.attempted > limit
            or first.completed != first.attempted
            or _table_count(fresh_store, "person_projection_jobs") > limit * 2
            or first.pending != legacy_count - first.completed
        ):
            raise SystemExit(f"first bounded legacy repair exceeded its limit: {first}")

        passes = [first]
        while passes[-1].pending:
            if len(passes) > legacy_count:
                raise SystemExit(f"bounded legacy repair failed to converge: {passes}")
            passes.append(
                reconcile_pending_person_projections(fresh_store, vault, limit=limit)
            )
        if (
            sum(summary.attempted for summary in passes) != legacy_count
            or any(summary.attempted > limit for summary in passes)
            or any(summary.completed != summary.attempted for summary in passes)
            or _table_count(fresh_store, "person_projection_jobs") != legacy_count
            or fresh_store.count_pending_person_projection_jobs() != 0
            or len(list((vault.root_path / "People").glob("*.md"))) != legacy_count
        ):
            raise SystemExit(f"later bounded legacy repair passes did not converge: {passes}")


def test_process_death_fresh_repair_and_schema_fk_idempotency() -> None:
    with TemporaryDirectory(prefix="jarvis-person-projection-process-death-") as temp:
        root = Path(temp)
        store, vault = _setup(root / "death")
        files_before = {path.relative_to(vault.root_path) for path in vault.root_path.rglob("*.md")}
        target: InteractionMutationTarget | None = None
        try:
            target = _record(
                store,
                name="PRIVATE-PROCESS-DEATH-PERSON",
                summary="PRIVATE-PROCESS-DEATH-SUMMARY",
            )
            raise SimulatedProcessDeath("injected death after atomic DB commit")
        except SimulatedProcessDeath:
            pass
        if target is None:
            raise SystemExit("process-death fixture died before the atomic store result")
        _assert_atomic_graph(store, target)
        counts_before = _graph_counts(store)
        files_after = {path.relative_to(vault.root_path) for path in vault.root_path.rglob("*.md")}
        if files_after != files_before:
            raise SystemExit("process-death fixture wrote files before simulated death")

        fresh_store = MemoryStore(root / "death" / "store.sqlite")
        fresh_store.init()
        fresh_vault = ObsidianVault(root / "death" / "vault")
        fresh_vault.init()
        person_summary = reconcile_pending_person_projections(fresh_store, fresh_vault, limit=20)
        memory_summary = reconcile_pending_memory_projections(fresh_store, fresh_vault, limit=20)
        if (
            person_summary.attempted != 1
            or person_summary.completed != 1
            or person_summary.pending != 0
            or memory_summary.attempted != 1
            or memory_summary.completed != 1
            or memory_summary.pending != 0
            or _graph_counts(fresh_store) != counts_before
        ):
            raise SystemExit(
                "fresh reconciliation did not repair both queues without source duplication: "
                f"{person_summary} / {memory_summary}"
            )
        _assert_job_privacy(
            fresh_store,
            target,
            forbidden=(target.person_name, "PRIVATE-PROCESS-DEATH-SUMMARY", str(root)),
        )
        if len(person_summary.outcomes) != 1:
            raise SystemExit(f"process-death repair returned unexpected outcomes: {person_summary}")
        repaired_person = person_summary.outcomes[0]
        _assert_opaque_person_path_display(
            repaired_person.path_display,
            repaired_person.person_id,
            "process-death person outcome path_display",
        )
        _assert_private(
            tuple(vars(repaired_person).values()),
            (target.person_name, "PRIVATE-PROCESS-DEATH-SUMMARY", str(root)),
            "process-death person projection outcome",
        )

        migration_store, _ = _setup(root / "migration")
        migration_target = _record(
            migration_store,
            name="PRIVATE-MIGRATION-PERSON",
            summary="PRIVATE-MIGRATION-SUMMARY",
        )
        published_memory = reconcile_pending_memory_projections(
            migration_store, ObsidianVault(root / "migration" / "vault"), limit=1
        )
        if published_memory.completed != 1:
            raise SystemExit("dedicated memory projection fixture did not publish")
        with migration_store.connect() as conn:
            unrelated = conn.execute(
                """
                INSERT INTO memories(category, title, body, source, confidence, created_at, updated_at)
                VALUES ('general', 'unrelated', 'unrelated', 'smoke', 1.0, 'now', 'now')
                """
            )
            unrelated_memory_id = int(unrelated.lastrowid)
        before_job = migration_store.get_person_projection_job(migration_target.person_id)
        migration_store.init()
        migration_store.init()
        after_job = migration_store.get_person_projection_job(migration_target.person_id)
        if (
            before_job is None
            or after_job is None
            or _graph_counts(migration_store)
            != {
                "people": 1,
                "person_interactions": 1,
                "memories": 2,
                "memory_projection_jobs": 1,
                "interaction_memory_links": 1,
                "person_projection_jobs": 1,
            }
            or tuple(before_job) != tuple(after_job)
        ):
            raise SystemExit("repeated schema creation changed person projection custody")
        with migration_store.connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO interaction_memory_links("
                    "interaction_id, memory_id, created_at, updated_at"
                    ") VALUES (?, ?, ?, ?)",
                    (9223372036854775806, 9223372036854775805, "now", "now"),
                )
            except sqlite3.IntegrityError:
                pass
            else:
                raise SystemExit("interaction-memory custody foreign keys were not enforced")
            conn.execute("DELETE FROM people WHERE id = ?", (migration_target.person_id,))
        if (
            _table_count(migration_store, "people") != 0
            or _table_count(migration_store, "person_interactions") != 0
            or _table_count(migration_store, "person_projection_jobs") != 0
            or _table_count(migration_store, "interaction_memory_links") != 0
            or _table_count(migration_store, "memories") != 1
            or _table_count(migration_store, "memory_projection_jobs") != 1
        ):
            raise SystemExit("person deletion did not apply exact FK custody behavior")
        with migration_store.connect() as conn:
            retained = conn.execute(
                "SELECT id FROM memories WHERE id = ?", (unrelated_memory_id,)
            ).fetchone()
            trigger_count = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE type = 'trigger' AND name = 'person_interactions_delete_dedicated_memory'"
            ).fetchone()[0]
            delete_job = conn.execute(
                "SELECT * FROM memory_projection_jobs WHERE memory_id = ?",
                (migration_target.memory_id,),
            ).fetchone()
        if (
            retained is None
            or trigger_count != 1
            or delete_job is None
            or delete_job["operation"] != "delete"
            or delete_job["state"] != "pending"
        ):
            raise SystemExit("person deletion removed unrelated memory or migration was not idempotent")
        deleted_projection = reconcile_pending_memory_projections(
            migration_store, ObsidianVault(root / "migration" / "vault"), limit=1
        )
        if (
            deleted_projection.completed != 1
            or list((root / "migration" / "vault" / "Memory Tree" / "Records").glob("*.md"))
        ):
            raise SystemExit("dedicated memory projection tombstone did not converge")


def test_completed_note_restart_audit_and_repair() -> None:
    with TemporaryDirectory(prefix="jarvis-person-projection-completed-audit-") as temp:
        root = Path(temp)
        store, vault = _setup(root)
        missing = _record(
            store,
            name="PRIVATE-COMPLETED-MISSING",
            summary="PRIVATE-COMPLETED-MISSING-SUMMARY",
        )
        corrupt = _record(
            store,
            name="PRIVATE-COMPLETED-CORRUPT",
            summary="PRIVATE-COMPLETED-CORRUPT-SUMMARY",
        )
        _complete_person(store, vault, missing)
        _complete_person(store, vault, corrupt)
        missing_path = _find_person_note(vault, missing.person_id, "missing audit fixture")
        corrupt_path = _find_person_note(vault, corrupt.person_id, "corrupt audit fixture")
        missing_path.unlink()
        corrupt_path.write_text(
            corrupt_path.read_text(encoding="utf-8") + "\nCORRUPTED-BY-SMOKE\n",
            encoding="utf-8",
        )

        fresh_store = MemoryStore(root / "store.sqlite")
        fresh_store.init()
        fresh_vault = ObsidianVault(root / "vault")
        fresh_vault.init()
        repaired = reconcile_pending_person_projections(fresh_store, fresh_vault, limit=2)
        if repaired.attempted != 2 or repaired.completed != 2 or repaired.pending != 0:
            raise SystemExit(f"completed person audit did not repair a bounded batch: {repaired}")
        for target in (missing, corrupt):
            job = fresh_store.get_person_projection_job(target.person_id)
            note = _find_person_note(fresh_vault, target.person_id, "completed audit repair")
            digest = hashlib.sha256(note.read_bytes()).hexdigest()
            if job["state"] != "completed" or job["content_digest"] != digest:
                raise SystemExit("completed person audit did not retain repaired evidence")
            _assert_job_privacy(
                fresh_store,
                target,
                forbidden=(target.person_name, str(root)),
            )
        _assert_private(
            tuple(value for outcome in repaired.outcomes for value in vars(outcome).values()),
            (missing.person_name, corrupt.person_name, str(root)),
            "completed person audit outcomes",
        )


def main() -> None:
    _assert_store_contract()
    test_atomic_graph_privacy_and_late_rollback()
    test_exact_evidence_write_failure_and_crash_adoption()
    test_supersession_bounded_repair_and_malformed_fail_closed()
    test_global_interaction_watermark_with_happened_at_render_cap()
    test_legacy_people_bounded_job_seeding_and_convergence()
    test_process_death_fresh_repair_and_schema_fk_idempotency()
    test_completed_note_restart_audit_and_repair()
    print("Person projection reconciliation smoke passed")


if __name__ == "__main__":
    main()
