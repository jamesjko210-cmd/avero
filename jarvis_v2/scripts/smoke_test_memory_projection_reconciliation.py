from __future__ import annotations

import hashlib
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.memory.memory_projection import (
    reconcile_memory_projection,
    reconcile_pending_memory_projections,
)
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import MemoryRecord, MemoryStore


def _setup(
    root: Path,
    *,
    store_name: str = "store",
    vault_root: Path | None = None,
) -> tuple[MemoryStore, ObsidianVault]:
    store = MemoryStore(root / f"{store_name}.sqlite")
    store.init()
    vault = ObsidianVault(vault_root or (root / "vault"))
    vault.init()
    return store, vault


def _canonical_path(vault: ObsidianVault, job: sqlite3.Row) -> Path:
    return (
        vault.root_path
        / "Memory Tree"
        / "Records"
        / f"{int(job['memory_id']):06d} [{job['store_identity']}].md"
    )


def _complete(store: MemoryStore, vault: ObsidianVault, memory_id: int):
    outcome = reconcile_memory_projection(store, vault, memory_id)
    if outcome.status != "completed" or not outcome.path_display:
        raise SystemExit(f"memory projection did not complete: {outcome}")
    return outcome


def _target_args(target) -> dict[str, object]:
    return {
        "expected_operation": target.operation,
        "expected_revision": target.revision,
        "expected_source_digest": target.source_digest,
    }


def test_atomic_add_pending_privacy_and_rollback() -> None:
    with TemporaryDirectory(prefix="jarvis-memory-projection-add-") as temp:
        root = Path(temp)
        store, _vault = _setup(root)
        secret = "RAW-MEMORY-BODY-MUST-NOT-ENTER-PROJECTION-LEDGER"
        target = store.add_memory_with_projection(
            MemoryRecord("fact", "Atomic projection", secret, "smoke", 0.8)
        )
        with store.connect() as conn:
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(memory_projection_jobs)")
            }
            persisted = conn.execute(
                "SELECT * FROM memory_projection_jobs WHERE memory_id = ?",
                (target.memory_id,),
            ).fetchone()
        if persisted is None or persisted["state"] != "pending":
            raise SystemExit("memory add did not atomically reserve a pending projection")
        if {"body", "title", "source"} & columns:
            raise SystemExit("memory projection ledger persisted source-content columns")
        if any(secret in str(value or "") for value in persisted):
            raise SystemExit("memory projection ledger persisted the raw memory body")

        rollback_store, _ = _setup(root / "rollback")
        with rollback_store.connect() as conn:
            conn.execute(
                """
                CREATE TRIGGER reject_memory_projection_job
                BEFORE INSERT ON memory_projection_jobs
                BEGIN
                    SELECT RAISE(ABORT, 'injected projection reservation failure');
                END
                """
            )
        try:
            rollback_store.add_memory_with_projection(
                MemoryRecord("fact", "Must roll back", "not committed", "smoke")
            )
        except sqlite3.DatabaseError:
            pass
        else:
            raise SystemExit("projection reservation failure did not abort the memory add")
        if rollback_store.list_memories(limit=20):
            raise SystemExit("failed projection reservation left a committed memory row")


def test_publish_evidence_failure_repair_and_crash_adoption() -> None:
    with TemporaryDirectory(prefix="jarvis-memory-projection-publish-") as temp:
        root = Path(temp)
        store, vault = _setup(root)
        body = "publish evidence body"
        target = store.add_memory_with_projection(
            MemoryRecord("fact", "Publish evidence", body, "smoke", 0.9)
        )
        published = _complete(store, vault, target.memory_id)
        note = vault.root_path / published.path_display
        text = note.read_text(encoding="utf-8")
        job = store.get_memory_projection_job(target.memory_id)
        if (
            job is None
            or job["state"] != "completed"
            or published.completion_status != "published"
            or job["canonical_path_display"] != published.path_display
            or job["content_digest"] != hashlib.sha256(text.encode("utf-8")).hexdigest()
            or f"memory_revision: {target.revision}" not in text
            or f"source_digest: {target.source_digest}" not in text
            or body not in text
        ):
            raise SystemExit("completed memory projection evidence diverged from the note")
        if vault.verify_memory_projection_evidence(
            memory_id=target.memory_id,
            store_identity=str(job["store_identity"]),
            expected_relative_path="Memory Tree/not-canonical.md",
            expected_content_digest=str(job["content_digest"]),
        ):
            raise SystemExit("memory evidence verifier accepted a non-canonical path")

        failing_store, failing_vault = _setup(root / "write-failure")
        failing_target = failing_store.add_memory_with_projection(
            MemoryRecord("fact", "Repair after failure", "repair exactly once", "smoke")
        )
        original_write = failing_vault.write_memory_projection_with_evidence

        def fail_write(*_args, **_kwargs):
            raise OSError("injected memory projection write failure")

        failing_vault.write_memory_projection_with_evidence = fail_write  # type: ignore[method-assign]
        failed = reconcile_memory_projection(
            failing_store,
            failing_vault,
            failing_target.memory_id,
            **_target_args(failing_target),
        )
        failing_vault.write_memory_projection_with_evidence = original_write  # type: ignore[method-assign]
        failed_job = failing_store.get_memory_projection_job(failing_target.memory_id)
        if (
            failed.status != "pending_error"
            or failed_job is None
            or failed_job["state"] != "pending"
            or failed_job["last_error_code"] != "vault_publish_failed"
            or len(failing_store.list_memories(limit=20)) != 1
        ):
            raise SystemExit(f"write failure did not retain repair custody: {failed} / {failed_job}")
        repaired = _complete(failing_store, failing_vault, failing_target.memory_id)
        if len(failing_store.list_memories(limit=20)) != 1:
            raise SystemExit("projection repair duplicated the committed source memory")
        if "repair exactly once" not in (failing_vault.root_path / repaired.path_display).read_text(
            encoding="utf-8"
        ):
            raise SystemExit("projection repair did not publish the committed memory")

        crash_store, crash_vault = _setup(root / "crash-adoption")
        crash_target = crash_store.add_memory_with_projection(
            MemoryRecord("fact", "Crash adoption", "already durable bytes", "smoke")
        )
        crash_job = crash_store.get_memory_projection_job(crash_target.memory_id)
        if crash_job is None:
            raise SystemExit("crash-adoption fixture has no pending job")
        record = MemoryRecord(
            str(crash_job["category"]),
            str(crash_job["title"]),
            str(crash_job["body"]),
            str(crash_job["source"]),
            float(crash_job["confidence"]),
        )
        crash_path, crash_digest = crash_vault.write_memory_projection_with_evidence(
            record,
            memory_id=crash_target.memory_id,
            store_identity=str(crash_job["store_identity"]),
            memory_revision=crash_target.revision,
            source_digest=crash_target.source_digest,
            created_at=str(crash_job["memory_created_at"]),
        )
        if crash_store.get_memory_projection_job(crash_target.memory_id)["state"] != "pending":
            raise SystemExit("crash fixture completed its job before reconciliation")
        adopted = _complete(crash_store, crash_vault, crash_target.memory_id)
        adopted_job = crash_store.get_memory_projection_job(crash_target.memory_id)
        if (
            adopted_job["content_digest"] != crash_digest
            or crash_vault.root_path / adopted.path_display != crash_path
            or len(crash_store.list_memories(limit=20)) != 1
        ):
            raise SystemExit("exact post-crash bytes were not adopted without source duplication")


def test_completed_publish_audit_repair_and_tamper_refusal() -> None:
    with TemporaryDirectory(prefix="jarvis-memory-projection-completed-audit-") as temp:
        root = Path(temp)

        healthy_store, healthy_vault = _setup(root / "healthy")
        healthy_target = healthy_store.add_memory_with_projection(
            MemoryRecord("fact", "Healthy audit", "retained bytes", "smoke")
        )
        healthy = _complete(healthy_store, healthy_vault, healthy_target.memory_id)
        original_healthy_write = healthy_vault.write_memory_projection_with_evidence
        healthy_write_calls = 0
        healthy_authority_calls = 0

        def count_healthy_write(*args, **kwargs):
            nonlocal healthy_write_calls
            healthy_write_calls += 1
            return original_healthy_write(*args, **kwargs)

        def count_healthy_authority() -> None:
            nonlocal healthy_authority_calls
            healthy_authority_calls += 1

        healthy_vault.write_memory_projection_with_evidence = count_healthy_write  # type: ignore[method-assign]
        audited = reconcile_memory_projection(
            healthy_store,
            healthy_vault,
            healthy_target.memory_id,
            **_target_args(healthy_target),
            effect_authority=count_healthy_authority,
        )
        healthy_vault.write_memory_projection_with_evidence = original_healthy_write  # type: ignore[method-assign]
        if (
            audited.status != "completed"
            or audited.completion_status != "verified"
            or audited.path_display != healthy.path_display
            or audited.content_digest != healthy.content_digest
            or healthy_write_calls != 0
            or healthy_authority_calls != 1
        ):
            raise SystemExit(f"healthy completed audit rewrote retained evidence: {audited}")

        missing_store, missing_vault = _setup(root / "missing")
        missing_target = missing_store.add_memory_with_projection(
            MemoryRecord("fact", "Missing repair", "repair without duplicate", "smoke")
        )
        missing_first = _complete(missing_store, missing_vault, missing_target.memory_id)
        missing_path = missing_vault.root_path / missing_first.path_display
        missing_path.unlink()
        repaired = reconcile_memory_projection(
            missing_store,
            missing_vault,
            missing_target.memory_id,
            **_target_args(missing_target),
        )
        if (
            repaired.status != "completed"
            or repaired.completion_status != "repaired"
            or not missing_path.exists()
            or len(missing_store.list_memories(limit=20)) != 1
            or missing_store.get_memory_projection_job(missing_target.memory_id)["state"]
            != "completed"
        ):
            raise SystemExit(f"missing completed note did not repair exactly once: {repaired}")

        tamper_store, tamper_vault = _setup(root / "tamper")
        tamper_target = tamper_store.add_memory_with_projection(
            MemoryRecord("fact", "Tamper refusal", "original retained body", "smoke")
        )
        tamper_first = _complete(tamper_store, tamper_vault, tamper_target.memory_id)
        tamper_path = tamper_vault.root_path / tamper_first.path_display
        tampered_text = tamper_path.read_text(encoding="utf-8") + "\nexternal retained edit\n"
        tamper_path.write_text(tampered_text, encoding="utf-8")
        refused = reconcile_memory_projection(
            tamper_store,
            tamper_vault,
            tamper_target.memory_id,
            **_target_args(tamper_target),
        )
        tamper_job = tamper_store.get_memory_projection_job(tamper_target.memory_id)
        if (
            refused.status != "pending_error"
            or tamper_job is None
            or tamper_job["state"] != "pending"
            or tamper_job["last_error_code"] != "vault_publish_failed"
            or tamper_job["prior_content_digest"] != tamper_first.content_digest
            or tamper_path.read_text(encoding="utf-8") != tampered_text
        ):
            raise SystemExit("tampered completed evidence was overwritten or lost custody")


def test_completed_audit_exact_supersession() -> None:
    with TemporaryDirectory(prefix="jarvis-memory-projection-audit-supersession-") as temp:
        root = Path(temp)
        store, vault = _setup(root)
        first_target = store.add_memory_with_projection(
            MemoryRecord("fact", "Audit supersession", "revision one", "smoke")
        )
        first = _complete(store, vault, first_target.memory_id)
        path = vault.root_path / first.path_display
        path.unlink()
        original_verify = vault.verify_memory_projection_evidence
        second_target = None

        def supersede_before_reopen(**kwargs):
            nonlocal second_target
            approval = store.resolve_memory_approval_target(first_target.memory_id)
            second_target = store.update_memory_exact_with_projection(
                first_target.memory_id,
                approval.revision,
                approval.binding,
                body="revision two",
            ).projection_targets[0]
            return original_verify(**kwargs)

        vault.verify_memory_projection_evidence = supersede_before_reopen  # type: ignore[method-assign]
        outcome = reconcile_memory_projection(
            store,
            vault,
            first_target.memory_id,
            **_target_args(first_target),
        )
        vault.verify_memory_projection_evidence = original_verify  # type: ignore[method-assign]
        latest = store.get_memory_projection_job(first_target.memory_id)
        text = path.read_text(encoding="utf-8")
        if (
            second_target is None
            or outcome.status != "superseded"
            or latest is None
            or latest["state"] != "completed"
            or latest["memory_revision"] != second_target.revision
            or latest["source_digest"] != second_target.source_digest
            or "revision two" not in text
            or "revision one" in text
        ):
            raise SystemExit("completed audit reopened or published a superseded identity")


def test_effect_authority_publication_fences() -> None:
    class AuthorityLost(RuntimeError):
        pass

    with TemporaryDirectory(prefix="jarvis-memory-projection-authority-") as temp:
        root = Path(temp)
        before_store, before_vault = _setup(root / "before")
        before_target = before_store.add_memory_with_projection(
            MemoryRecord("fact", "Authority before", "must not publish", "smoke")
        )
        before_job = before_store.get_memory_projection_job(before_target.memory_id)
        before_path = _canonical_path(before_vault, before_job)

        def lose_before_write() -> None:
            raise AuthorityLost("authority lost before memory publication")

        try:
            reconcile_memory_projection(
                before_store,
                before_vault,
                before_target.memory_id,
                effect_authority=lose_before_write,
            )
        except AuthorityLost:
            pass
        else:
            raise SystemExit("pre-publication authority loss was swallowed")
        if before_path.exists() or before_store.get_memory_projection_job(
            before_target.memory_id
        )["state"] != "pending":
            raise SystemExit("pre-publication authority loss produced an effect")

        after_store, after_vault = _setup(root / "after")
        after_target = after_store.add_memory_with_projection(
            MemoryRecord("fact", "Authority after", "durable uncompleted bytes", "smoke")
        )
        calls = 0

        def lose_after_write() -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise AuthorityLost("authority lost after memory publication")

        try:
            reconcile_memory_projection(
                after_store,
                after_vault,
                after_target.memory_id,
                effect_authority=lose_after_write,
            )
        except AuthorityLost:
            pass
        else:
            raise SystemExit("post-publication authority loss was swallowed")
        after_job = after_store.get_memory_projection_job(after_target.memory_id)
        after_path = _canonical_path(after_vault, after_job)
        if calls != 2 or not after_path.exists() or after_job["state"] != "pending":
            raise SystemExit("post-publication authority fence did not retain pending custody")
        adopted = _complete(after_store, after_vault, after_target.memory_id)
        if adopted.completion_status != "published" or len(
            after_store.list_memories(limit=20)
        ) != 1:
            raise SystemExit("post-authority retry did not adopt exact durable bytes")


def test_concurrent_completed_audit_and_missing_repair() -> None:
    with TemporaryDirectory(prefix="jarvis-memory-projection-concurrent-audit-") as temp:
        root = Path(temp)
        store, vault = _setup(root)
        target = store.add_memory_with_projection(
            MemoryRecord("fact", "Concurrent repair", "one source memory", "smoke")
        )
        first = _complete(store, vault, target.memory_id)
        path = vault.root_path / first.path_display
        path.unlink()
        original_verify = vault.verify_memory_projection_evidence
        audit_barrier = threading.Barrier(2)
        audit_call_lock = threading.Lock()
        audit_calls = 0

        def synchronized_verify(**kwargs):
            nonlocal audit_calls
            with audit_call_lock:
                audit_calls += 1
                should_wait = audit_calls <= 2
            if should_wait:
                audit_barrier.wait(timeout=5)
            return original_verify(**kwargs)

        vault.verify_memory_projection_evidence = synchronized_verify  # type: ignore[method-assign]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(reconcile_memory_projection, store, vault, target.memory_id)
                for _ in range(2)
            ]
            outcomes = [future.result(timeout=10) for future in futures]
        vault.verify_memory_projection_evidence = original_verify  # type: ignore[method-assign]
        final = reconcile_memory_projection(store, vault, target.memory_id)
        if (
            any(
                outcome.status not in {"completed", "superseded", "superseded_or_busy"}
                for outcome in outcomes
            )
            or final.status != "completed"
            or final.completion_status != "verified"
            or not path.exists()
            or len(store.list_memories(limit=20)) != 1
            or store.get_memory_projection_job(target.memory_id)["state"] != "completed"
        ):
            raise SystemExit(f"concurrent completed audit/repair did not converge: {outcomes}")


def test_edit_delete_and_merge_converge() -> None:
    with TemporaryDirectory(prefix="jarvis-memory-projection-mutations-") as temp:
        root = Path(temp)
        store, vault = _setup(root)
        original = MemoryRecord("old category", "Old title", "old body", "smoke")
        added = store.add_memory_with_projection(original)
        first = _complete(store, vault, added.memory_id)
        canonical = vault.root_path / first.path_display
        edit_target = store.resolve_memory_approval_target(added.memory_id)
        edited = store.update_memory_exact_with_projection(
            added.memory_id,
            edit_target.revision,
            edit_target.binding,
            category="new category",
            title="New title",
            body="new body",
        )
        if edited.status != "updated" or len(edited.projection_targets) != 1:
            raise SystemExit(f"exact edit did not reserve one projection: {edited}")
        edit_outcome = _complete(store, vault, added.memory_id)
        updated_note = vault.root_path / edit_outcome.path_display
        updated_text = updated_note.read_text(encoding="utf-8")
        if (
            updated_note != canonical
            or "category: new category" not in updated_text
            or "title: New title" not in updated_text
            or "new body" not in updated_text
            or "old body" in updated_text
        ):
            raise SystemExit("path-changing memory edit did not converge on the canonical note")

        delete_target = store.resolve_memory_approval_target(added.memory_id)
        deleted = store.delete_memory_exact_with_projection(
            added.memory_id,
            delete_target.revision,
            delete_target.binding,
        )
        if deleted.status != "deleted" or len(deleted.projection_targets) != 1:
            raise SystemExit(f"exact delete did not retain a tombstone: {deleted}")
        deletion = _complete(store, vault, added.memory_id)
        replay = reconcile_memory_projection(store, vault, added.memory_id)
        delete_job = store.get_memory_projection_job(added.memory_id)
        if (
            canonical.exists()
            or deletion.completion_status not in {"deleted", "absent"}
            or replay.status != "completed"
            or delete_job["state"] != "completed"
            or store.get_memory(added.memory_id) is not None
        ):
            raise SystemExit("delete tombstone repair was not durable and idempotent")

        merge_store, merge_vault = _setup(root / "merge")
        keep = merge_store.add_memory_with_projection(
            MemoryRecord("fact", "Merge keep", "keep body", "smoke", 0.7)
        )
        remove = merge_store.add_memory_with_projection(
            MemoryRecord("fact", "Merge remove", "remove body", "smoke", 0.9)
        )
        keep_first = _complete(merge_store, merge_vault, keep.memory_id)
        remove_first = _complete(merge_store, merge_vault, remove.memory_id)
        targets = merge_store.resolve_memory_merge_approval_targets(
            keep.memory_id, remove.memory_id
        )
        if targets is None:
            raise SystemExit("merge fixture did not resolve exact targets")
        keep_approval, remove_approval = targets
        merged = merge_store.merge_memories_exact_with_projection(
            keep.memory_id,
            keep_approval.revision,
            keep_approval.binding,
            remove.memory_id,
            remove_approval.revision,
            remove_approval.binding,
        )
        if (
            merged.status != "merged"
            or len(merged.projection_targets) != 2
            or {target.operation for target in merged.projection_targets}
            != {"publish", "delete"}
        ):
            raise SystemExit(f"merge did not reserve publish and delete effects: {merged}")
        for projection_target in merged.projection_targets:
            _complete(merge_store, merge_vault, projection_target.memory_id)
        keep_text = (merge_vault.root_path / keep_first.path_display).read_text(encoding="utf-8")
        if (
            "keep body" not in keep_text
            or "remove body" not in keep_text
            or (merge_vault.root_path / remove_first.path_display).exists()
            or merge_store.get_memory(remove.memory_id) is not None
            or any(
                merge_store.get_memory_projection_job(target.memory_id)["state"] != "completed"
                for target in merged.projection_targets
            )
        ):
            raise SystemExit("merge projection effects did not converge exactly once")


def test_legacy_cleanup_exact_bytes_and_hostile_retention() -> None:
    class AuthorityLost(RuntimeError):
        pass

    def reserve_same_record_update(store: MemoryStore, memory_id: int) -> None:
        approval = store.resolve_memory_approval_target(memory_id)
        result = store.update_memory_exact_with_projection(
            memory_id,
            approval.revision,
            approval.binding,
        )
        if result.status != "updated":
            raise SystemExit(f"legacy cleanup fixture update failed: {result}")

    with TemporaryDirectory(prefix="jarvis-memory-projection-legacy-cleanup-") as temp:
        root = Path(temp)

        exact_store, exact_vault = _setup(root / "exact")
        exact_record = MemoryRecord("fact", "Exact legacy", "exact legacy body", "smoke", 0.8)
        exact_target = exact_store.add_memory_with_projection(exact_record)
        _complete(exact_store, exact_vault, exact_target.memory_id)
        exact_job = exact_store.get_memory_projection_job(exact_target.memory_id)
        exact_legacy = exact_vault.write_memory(
            exact_record,
            exact_target.memory_id,
            store_identity=str(exact_job["store_identity"]),
        )
        exact_text = exact_legacy.read_text(encoding="utf-8")
        retained_date = exact_text.split("created: ", 1)[1][:10]
        exact_legacy.write_text(
            exact_text.replace(retained_date, "2020-01-02", 1),
            encoding="utf-8",
        )
        reserve_same_record_update(exact_store, exact_target.memory_id)
        exact_outcome = reconcile_memory_projection(
            exact_store, exact_vault, exact_target.memory_id
        )
        if exact_outcome.status != "completed" or exact_legacy.exists():
            raise SystemExit("exact generator-owned legacy bytes were not cleaned up")

        edited_store, edited_vault = _setup(root / "edited")
        edited_record = MemoryRecord("fact", "Edited legacy", "owned legacy body", "smoke", 0.7)
        edited_target = edited_store.add_memory_with_projection(edited_record)
        _complete(edited_store, edited_vault, edited_target.memory_id)
        edited_job = edited_store.get_memory_projection_job(edited_target.memory_id)
        edited_legacy = edited_vault.write_memory(
            edited_record,
            edited_target.memory_id,
            store_identity=str(edited_job["store_identity"]),
        )
        manual_text = edited_legacy.read_text(encoding="utf-8") + "\nManual note retained.\n"
        edited_legacy.write_text(manual_text, encoding="utf-8")
        reserve_same_record_update(edited_store, edited_target.memory_id)
        edited_outcome = reconcile_memory_projection(
            edited_store, edited_vault, edited_target.memory_id
        )
        edited_pending = edited_store.get_memory_projection_job(edited_target.memory_id)
        if (
            edited_outcome.status != "pending_error"
            or edited_pending["state"] != "pending"
            or edited_pending["last_error_code"]
            != "legacy_cleanup_ownership_mismatch"
            or edited_legacy.read_text(encoding="utf-8") != manual_text
        ):
            raise SystemExit("edited owned legacy bytes were deleted or falsely completed")

        authority_store, authority_vault = _setup(root / "authority")
        authority_record = MemoryRecord(
            "fact", "Authority legacy", "authority legacy body", "smoke", 0.6
        )
        authority_target = authority_store.add_memory_with_projection(authority_record)
        _complete(authority_store, authority_vault, authority_target.memory_id)
        authority_job = authority_store.get_memory_projection_job(authority_target.memory_id)
        authority_legacy = authority_vault.write_memory(
            authority_record,
            authority_target.memory_id,
            store_identity=str(authority_job["store_identity"]),
        )
        authority_bytes = authority_legacy.read_bytes()
        reserve_same_record_update(authority_store, authority_target.memory_id)
        calls = 0

        def lose_immediately_before_cleanup() -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise AuthorityLost("authority lost immediately before legacy cleanup")

        try:
            reconcile_memory_projection(
                authority_store,
                authority_vault,
                authority_target.memory_id,
                effect_authority=lose_immediately_before_cleanup,
            )
        except AuthorityLost:
            pass
        else:
            raise SystemExit("pre-cleanup authority loss was swallowed")
        authority_pending = authority_store.get_memory_projection_job(
            authority_target.memory_id
        )
        if (
            calls != 2
            or authority_pending["state"] != "pending"
            or not authority_legacy.exists()
            or authority_legacy.read_bytes() != authority_bytes
        ):
            raise SystemExit("pre-cleanup authority loss deleted legacy bytes")

        error_store, error_vault = _setup(root / "cleanup-error")
        error_record = MemoryRecord(
            "fact", "Cleanup error", "cleanup exception body", "smoke", 0.5
        )
        error_target = error_store.add_memory_with_projection(error_record)
        _complete(error_store, error_vault, error_target.memory_id)
        error_job = error_store.get_memory_projection_job(error_target.memory_id)
        error_legacy = error_vault.write_memory(
            error_record,
            error_target.memory_id,
            store_identity=str(error_job["store_identity"]),
        )
        reserve_same_record_update(error_store, error_target.memory_id)

        def fail_cleanup(**_kwargs):
            raise OSError("mock cleanup failure")

        error_vault.delete_legacy_memory_projection = fail_cleanup
        error_outcome = reconcile_memory_projection(
            error_store, error_vault, error_target.memory_id
        )
        error_pending = error_store.get_memory_projection_job(error_target.memory_id)
        if (
            error_outcome.status != "pending_error"
            or error_outcome.file_written is not True
            or not error_outcome.path_display
            or error_pending["state"] != "pending"
            or error_pending["last_error_code"] != "legacy_cleanup_failed"
            or not error_legacy.exists()
        ):
            raise SystemExit("legacy cleanup exception escaped or hid canonical publication")


def test_collisions_store_isolation_and_source_mismatch() -> None:
    with TemporaryDirectory(prefix="jarvis-memory-projection-boundaries-") as temp:
        root = Path(temp)
        store, vault = _setup(root / "collisions")
        collision_targets = [
            store.add_memory_with_projection(
                MemoryRecord("fact", "Unowned collision", "must remain pending", "smoke")
            ),
            store.add_memory_with_projection(
                MemoryRecord("fact", "Foreign collision", "must also remain pending", "smoke")
            ),
        ]
        collision_jobs = [store.get_memory_projection_job(item.memory_id) for item in collision_targets]
        unowned_path = _canonical_path(vault, collision_jobs[0])
        unowned_path.parent.mkdir(parents=True, exist_ok=True)
        unowned_path.write_text("foreign unowned bytes\n", encoding="utf-8")
        foreign_path = _canonical_path(vault, collision_jobs[1])
        foreign_path.write_text(
            "---\n"
            "jarvis_projection: memory\n"
            f"store_identity: {'f' * 32}\n"
            f"memory_id: {collision_targets[1].memory_id}\n"
            "---\n\nforeign owned bytes\n",
            encoding="utf-8",
        )
        for target, path, expected in (
            (collision_targets[0], unowned_path, "foreign unowned bytes\n"),
            (collision_targets[1], foreign_path, "foreign owned bytes\n"),
        ):
            outcome = reconcile_memory_projection(store, vault, target.memory_id)
            job = store.get_memory_projection_job(target.memory_id)
            if (
                outcome.status != "pending_error"
                or job["state"] != "pending"
                or job["last_error_code"] != "vault_publish_failed"
                or expected not in path.read_text(encoding="utf-8")
            ):
                raise SystemExit("canonical collision was overwritten or falsely completed")

        shared_vault_root = root / "shared-vault"
        first_store, shared_vault = _setup(
            root / "multi-store", store_name="first", vault_root=shared_vault_root
        )
        second_store, _ = _setup(
            root / "multi-store", store_name="second", vault_root=shared_vault_root
        )
        first_target = first_store.add_memory_with_projection(
            MemoryRecord("fact", "Same numeric id", "first store", "smoke")
        )
        second_target = second_store.add_memory_with_projection(
            MemoryRecord("fact", "Same numeric id", "second store", "smoke")
        )
        first = _complete(first_store, shared_vault, first_target.memory_id)
        second = _complete(second_store, shared_vault, second_target.memory_id)
        if (
            first_target.memory_id != second_target.memory_id
            or first.path_display == second.path_display
            or "first store" not in (shared_vault.root_path / first.path_display).read_text(encoding="utf-8")
            or "second store" not in (shared_vault.root_path / second.path_display).read_text(encoding="utf-8")
        ):
            raise SystemExit("store-local memory IDs did not receive store-unique projections")

        mismatch_store, mismatch_vault = _setup(root / "source-mismatch")
        mismatch_target = mismatch_store.add_memory_with_projection(
            MemoryRecord("fact", "Source mismatch", "original source", "smoke")
        )
        wrong_expected = reconcile_memory_projection(
            mismatch_store,
            mismatch_vault,
            mismatch_target.memory_id,
            expected_operation="publish",
            expected_revision=mismatch_target.revision + 1,
            expected_source_digest="0" * 64,
        )
        mismatch_job = mismatch_store.get_memory_projection_job(mismatch_target.memory_id)
        mismatch_path = _canonical_path(mismatch_vault, mismatch_job)
        if wrong_expected.status != "superseded" or mismatch_path.exists():
            raise SystemExit("wrong expected revision/digest did not fail closed")
        with mismatch_store.connect() as conn:
            conn.execute(
                "UPDATE memories SET body = ? WHERE id = ?",
                ("tampered without revision", mismatch_target.memory_id),
            )
        drifted = reconcile_memory_projection(
            mismatch_store, mismatch_vault, mismatch_target.memory_id
        )
        mismatch_job = mismatch_store.get_memory_projection_job(mismatch_target.memory_id)
        if (
            drifted.status != "pending_error"
            or mismatch_job["state"] != "pending"
            or mismatch_job["last_error_code"] != "source_snapshot_unavailable"
            or mismatch_path.exists()
        ):
            raise SystemExit("source-row digest drift did not remain safely pending")


def test_external_edits_block_update_delete_and_unknown_supersession() -> None:
    with TemporaryDirectory(prefix="jarvis-memory-projection-external-edit-") as temp:
        root = Path(temp)

        update_store, update_vault = _setup(root / "update")
        update_target = update_store.add_memory_with_projection(
            MemoryRecord("fact", "Edited update", "original", "smoke")
        )
        update_outcome = _complete(update_store, update_vault, update_target.memory_id)
        update_path = update_vault.root_path / update_outcome.path_display
        update_path.write_text(
            update_path.read_text(encoding="utf-8") + "\nmanual update evidence\n",
            encoding="utf-8",
        )
        update_binding = update_store.resolve_memory_approval_target(update_target.memory_id)
        updated = update_store.update_memory_exact_with_projection(
            update_target.memory_id,
            update_binding.revision,
            update_binding.binding,
            body="database update",
        )
        update_repair = reconcile_memory_projection(
            update_store, update_vault, update_target.memory_id
        )
        update_job = update_store.get_memory_projection_job(update_target.memory_id)
        if (
            updated.status != "updated"
            or update_repair.status != "pending_error"
            or update_job["last_error_code"] != "vault_publish_failed"
            or "manual update evidence" not in update_path.read_text(encoding="utf-8")
        ):
            raise SystemExit("external edit was overwritten by a later memory update")

        delete_store, delete_vault = _setup(root / "delete")
        delete_target = delete_store.add_memory_with_projection(
            MemoryRecord("fact", "Edited delete", "original", "smoke")
        )
        delete_outcome = _complete(delete_store, delete_vault, delete_target.memory_id)
        delete_path = delete_vault.root_path / delete_outcome.path_display
        delete_path.write_text(
            delete_path.read_text(encoding="utf-8") + "\nmanual delete evidence\n",
            encoding="utf-8",
        )
        delete_binding = delete_store.resolve_memory_approval_target(delete_target.memory_id)
        deleted = delete_store.delete_memory_exact_with_projection(
            delete_target.memory_id,
            delete_binding.revision,
            delete_binding.binding,
        )
        delete_repair = reconcile_memory_projection(
            delete_store, delete_vault, delete_target.memory_id
        )
        delete_job = delete_store.get_memory_projection_job(delete_target.memory_id)
        if (
            deleted.status != "deleted"
            or delete_repair.status != "pending_error"
            or delete_job["state"] != "pending"
            or delete_job["last_error_code"] != "delete_ownership_mismatch"
            or "manual delete evidence" not in delete_path.read_text(encoding="utf-8")
        ):
            raise SystemExit("external edit was deleted or falsely marked complete")

        pending_store, pending_vault = _setup(root / "pending")
        first = pending_store.add_memory_with_projection(
            MemoryRecord("fact", "Pending predecessor", "revision one", "smoke")
        )
        first_job = pending_store.get_memory_projection_job(first.memory_id)
        first_path, _ = pending_vault.write_memory_projection_with_evidence(
            MemoryRecord(
                str(first_job["category"]),
                str(first_job["title"]),
                str(first_job["body"]),
                str(first_job["source"]),
                float(first_job["confidence"]),
            ),
            memory_id=first.memory_id,
            store_identity=str(first_job["store_identity"]),
            memory_revision=first.revision,
            source_digest=first.source_digest,
            created_at=str(first_job["memory_created_at"]),
        )
        first_path.write_text(
            first_path.read_text(encoding="utf-8") + "\nmanual pending evidence\n",
            encoding="utf-8",
        )
        first_binding = pending_store.resolve_memory_approval_target(first.memory_id)
        second = pending_store.update_memory_exact_with_projection(
            first.memory_id,
            first_binding.revision,
            first_binding.binding,
            body="revision two",
        )
        pending_repair = reconcile_memory_projection(
            pending_store, pending_vault, first.memory_id
        )
        if (
            second.status != "updated"
            or pending_repair.status != "pending_error"
            or "manual pending evidence" not in first_path.read_text(encoding="utf-8")
        ):
            raise SystemExit("unbound pending predecessor bytes were overwritten")


def test_pending_supersession_migration_and_bounded_summary() -> None:
    with TemporaryDirectory(prefix="jarvis-memory-projection-recovery-") as temp:
        root = Path(temp)
        store, vault = _setup(root / "supersession")
        first = store.add_memory_with_projection(
            MemoryRecord("first category", "First title", "revision one", "smoke")
        )
        first_approval = store.resolve_memory_approval_target(first.memory_id)
        second = store.update_memory_exact_with_projection(
            first.memory_id,
            first_approval.revision,
            first_approval.binding,
            category="second category",
            title="Second title",
            body="revision two",
        ).projection_targets[0]
        second_approval = store.resolve_memory_approval_target(first.memory_id)
        third = store.update_memory_exact_with_projection(
            first.memory_id,
            second_approval.revision,
            second_approval.binding,
            category="third category",
            title="Third title",
            body="revision three",
        ).projection_targets[0]
        pending = store.get_memory_projection_job(first.memory_id)
        if (
            second.revision != 2
            or third.revision != 3
            or pending["memory_revision"] != 3
            or pending["source_digest"] != third.source_digest
        ):
            raise SystemExit("pending supersession did not retain the newest exact target")
        latest = reconcile_memory_projection(
            store, vault, third.memory_id, **_target_args(third)
        )
        latest_text = (vault.root_path / latest.path_display).read_text(encoding="utf-8")
        if (
            latest.status != "completed"
            or "revision three" not in latest_text
            or "revision one" in latest_text
            or "revision two" in latest_text
            or "memory_revision: 3" not in latest_text
        ):
            raise SystemExit("repeated pending supersession did not converge to the newest revision")

        migration_path = root / "migration" / "legacy.sqlite"
        migration_store = MemoryStore(migration_path)
        migration_store.init()
        legacy_id = migration_store.add_memory(
            MemoryRecord("legacy", "Existing row", "must not be inferred complete", "legacy")
        )
        with migration_store.connect() as conn:
            conn.execute("DROP TABLE memory_projection_jobs")
        migration_store.init()
        with migration_store.connect() as conn:
            table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'memory_projection_jobs'"
            ).fetchone()
            job_count = conn.execute(
                "SELECT COUNT(*) AS count FROM memory_projection_jobs"
            ).fetchone()["count"]
        if table is None or job_count != 0 or migration_store.get_memory(legacy_id) is None:
            raise SystemExit("additive migration did not create an empty ledger beside legacy rows")

        batch_store, batch_vault = _setup(root / "bounded")
        batch_targets = [
            batch_store.add_memory_with_projection(
                MemoryRecord("fact", f"Pending {index}", f"body {index}", "smoke")
            )
            for index in range(3)
        ]
        summary = reconcile_pending_memory_projections(batch_store, batch_vault, limit=2)
        if (
            summary.attempted != 2
            or summary.completed != 2
            or summary.pending != 1
            or len(summary.outcomes) != 2
            or batch_store.count_pending_memory_projection_jobs() != 1
        ):
            raise SystemExit(f"bounded pending repair summary was inaccurate: {summary}")
        final_summary = reconcile_pending_memory_projections(batch_store, batch_vault, limit=1000)
        if (
            final_summary.attempted != 1
            or final_summary.completed != 1
            or final_summary.pending != 0
            or any(batch_store.get_memory(target.memory_id) is None for target in batch_targets)
        ):
            raise SystemExit(f"bounded pending repair did not converge on the next pass: {final_summary}")


def main() -> None:
    test_atomic_add_pending_privacy_and_rollback()
    test_publish_evidence_failure_repair_and_crash_adoption()
    test_completed_publish_audit_repair_and_tamper_refusal()
    test_completed_audit_exact_supersession()
    test_effect_authority_publication_fences()
    test_concurrent_completed_audit_and_missing_repair()
    test_edit_delete_and_merge_converge()
    test_legacy_cleanup_exact_bytes_and_hostile_retention()
    test_collisions_store_isolation_and_source_mismatch()
    test_external_edits_block_update_delete_and_unknown_supersession()
    test_pending_supersession_migration_and_bounded_summary()
    print("Memory projection reconciliation smoke passed")


if __name__ == "__main__":
    main()
