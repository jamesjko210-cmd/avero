from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from pathlib import Path

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.config import JarvisConfig
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.skill_projection import reconcile_skill_projection
from jarvis_v2.memory.store import MemoryStore, SkillRecord
from jarvis_v2.tools.skills import make_skill_tools


def _setup(root: Path, name: str = "store") -> tuple[MemoryStore, ObsidianVault]:
    store = MemoryStore(root / f"{name}.sqlite")
    store.init()
    vault = ObsidianVault(root / "vault")
    vault.init()
    return store, vault


def _path(vault: ObsidianVault, display: str) -> Path:
    return vault.root_path / display


def _row(store: MemoryStore, skill_id: int):
    with store.connect() as conn:
        return conn.execute("SELECT * FROM skills WHERE id = ?", (skill_id,)).fetchone()


def _require_completed(store: MemoryStore, vault: ObsidianVault, skill_id: int):
    outcome = reconcile_skill_projection(store, vault, skill_id)
    if outcome.status != "completed" or not outcome.path_display:
        raise SystemExit(f"skill projection did not complete: {outcome}")
    return outcome


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        store, vault = _setup(root)

        body_secret = "procedure-body-not-for-outbox"
        skill_id = store.save_skill(
            SkillRecord("Projection Smoke", "when testing", body_secret, "projection")
        )
        pending = store.get_skill_projection_job(skill_id)
        if pending is None or pending["state"] != "pending" or pending["body"] != body_secret:
            raise SystemExit("save did not atomically create a usable pending projection")
        with store.connect() as conn:
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(skill_projection_jobs)")
            }
            persisted = conn.execute(
                "SELECT * FROM skill_projection_jobs WHERE skill_id = ?", (skill_id,)
            ).fetchone()
        if {"body", "trigger", "tags"} & columns:
            raise SystemExit("projection outbox persisted skill content columns")
        if any(body_secret in str(value or "") for value in persisted):
            raise SystemExit("projection outbox persisted the skill body")

        published = _require_completed(store, vault, skill_id)
        note = _path(vault, published.path_display)
        text = note.read_text(encoding="utf-8")
        completed = store.get_skill_projection_job(skill_id)
        if (
            body_secret not in text
            or f"skill_revision: {completed['skill_revision']}" not in text
            or f"source_digest: {completed['source_digest']}" not in text
            or completed["content_digest"]
            != hashlib.sha256(text.encode("utf-8")).hexdigest()
        ):
            raise SystemExit("published skill evidence does not match the completed job")

        store.save_skill(
            SkillRecord("Projection Smoke", "when retrying", "retry body", "projection")
        )
        original_write = vault.write_skill_with_evidence

        def fail_write(*args, **kwargs):
            raise OSError("injected write failure")

        vault.write_skill_with_evidence = fail_write  # type: ignore[method-assign]
        failed = reconcile_skill_projection(store, vault, skill_id)
        if failed.status != "pending_error":
            raise SystemExit(f"write failure did not stay repairable: {failed}")
        failed_job = store.get_skill_projection_job(skill_id)
        if failed_job["state"] != "pending" or failed_job["last_error_code"] != "vault_publish_failed":
            raise SystemExit("write failure did not retain a bounded pending error")
        vault.write_skill_with_evidence = original_write  # type: ignore[method-assign]
        _require_completed(store, vault, skill_id)

        store.save_skill(
            SkillRecord("Projection Smoke", "old trigger", "old revision body", "projection")
        )
        injected = False

        def superseding_write(*args, **kwargs):
            nonlocal injected
            if not injected:
                injected = True
                store.save_skill(
                    SkillRecord(
                        "Projection Smoke", "latest trigger", "latest revision body", "projection"
                    )
                )
            return original_write(*args, **kwargs)

        vault.write_skill_with_evidence = superseding_write  # type: ignore[method-assign]
        superseded = reconcile_skill_projection(store, vault, skill_id)
        vault.write_skill_with_evidence = original_write  # type: ignore[method-assign]
        if superseded.status != "completed":
            raise SystemExit(f"stale publisher did not converge: {superseded}")
        final_text = _path(vault, superseded.path_display).read_text(encoding="utf-8")
        if "latest revision body" not in final_text or "old revision body" in final_text:
            raise SystemExit("an older projection overwrote the latest SQLite revision")

        race_store, race_vault_a = _setup(root / "stale-race")
        race_vault_b = ObsidianVault(root / "stale-race" / "vault")
        race_vault_b.init()
        old_target = race_store.save_skill_with_projection_target(
            SkillRecord("Stale Race", "old", "old body", "")
        )
        old_write_started = threading.Event()
        release_old_write = threading.Event()
        race_original_write = race_vault_a.write_skill_with_evidence

        def delayed_old_write(*args, **kwargs):
            old_write_started.set()
            if not release_old_write.wait(5):
                raise OSError("stale writer release timed out")
            return race_original_write(*args, **kwargs)

        race_vault_a.write_skill_with_evidence = delayed_old_write  # type: ignore[method-assign]
        stale_outcomes = []
        stale_thread = threading.Thread(
            target=lambda: stale_outcomes.append(
                reconcile_skill_projection(
                    race_store,
                    race_vault_a,
                    old_target.skill_id,
                    expected_operation=old_target.operation,
                    expected_revision=old_target.revision,
                    expected_source_digest=old_target.source_digest,
                )
            ),
            daemon=True,
        )
        stale_thread.start()
        if not old_write_started.wait(5):
            raise SystemExit("stale publisher did not reach the delayed write")
        new_target = race_store.save_skill_with_projection_target(
            SkillRecord("Stale Race", "new", "new body", "")
        )
        newest = reconcile_skill_projection(
            race_store,
            race_vault_b,
            new_target.skill_id,
            expected_operation=new_target.operation,
            expected_revision=new_target.revision,
            expected_source_digest=new_target.source_digest,
        )
        if newest.status != "completed":
            raise SystemExit(f"new projection did not win the stale race: {newest}")
        release_old_write.set()
        stale_thread.join(5)
        if stale_thread.is_alive() or not stale_outcomes or stale_outcomes[0].status != "superseded":
            raise SystemExit(f"stale publisher did not finish as superseded: {stale_outcomes}")
        race_job = race_store.get_skill_projection_job(new_target.skill_id)
        race_text = _path(race_vault_b, race_job["path_display"]).read_text(encoding="utf-8")
        if "new body" not in race_text or "old body" in race_text:
            raise SystemExit("stale publisher overwrote a newer completed projection")

        delete_race_store, delete_race_vault_a = _setup(root / "delete-race")
        delete_race_vault_b = ObsidianVault(root / "delete-race" / "vault")
        delete_race_vault_b.init()
        resurrect_target = delete_race_store.save_skill_with_projection_target(
            SkillRecord("Delete Race", "old", "must not resurrect", "")
        )
        resurrect_started = threading.Event()
        release_resurrect = threading.Event()
        resurrect_original_write = delete_race_vault_a.write_skill_with_evidence

        def delayed_resurrect(*args, **kwargs):
            resurrect_started.set()
            if not release_resurrect.wait(5):
                raise OSError("delete race release timed out")
            return resurrect_original_write(*args, **kwargs)

        delete_race_vault_a.write_skill_with_evidence = delayed_resurrect  # type: ignore[method-assign]
        resurrect_outcomes = []
        resurrect_thread = threading.Thread(
            target=lambda: resurrect_outcomes.append(
                reconcile_skill_projection(
                    delete_race_store,
                    delete_race_vault_a,
                    resurrect_target.skill_id,
                    expected_operation=resurrect_target.operation,
                    expected_revision=resurrect_target.revision,
                    expected_source_digest=resurrect_target.source_digest,
                )
            ),
            daemon=True,
        )
        resurrect_thread.start()
        if not resurrect_started.wait(5):
            raise SystemExit("delete-race publisher did not reach the delayed write")
        deleted_row = _row(delete_race_store, resurrect_target.skill_id)
        delete_race_store.delete_skill_exact(
            resurrect_target.skill_id,
            int(deleted_row["revision"]),
            str(deleted_row["name"]),
        )
        delete_job = delete_race_store.get_skill_projection_job(resurrect_target.skill_id)
        delete_done = reconcile_skill_projection(
            delete_race_store,
            delete_race_vault_b,
            resurrect_target.skill_id,
            expected_operation="delete",
            expected_revision=int(delete_job["skill_revision"]),
            expected_source_digest=str(delete_job["source_digest"]),
        )
        if delete_done.status != "completed":
            raise SystemExit(f"new delete did not complete before stale publish: {delete_done}")
        release_resurrect.set()
        resurrect_thread.join(5)
        if (
            resurrect_thread.is_alive()
            or not resurrect_outcomes
            or resurrect_outcomes[0].status != "superseded"
            or list((delete_race_vault_b.root_path / "Skills").glob("Delete Race*.md"))
        ):
            raise SystemExit("stale publisher resurrected a completed skill deletion")

        blocking_started = threading.Event()
        release_write = threading.Event()

        def blocked_write(*args, **kwargs):
            blocking_started.set()
            if not release_write.wait(5):
                raise OSError("test write release timed out")
            return original_write(*args, **kwargs)

        store.save_skill(
            SkillRecord("Projection Smoke", "blocked trigger", "blocked body", "projection")
        )
        vault.write_skill_with_evidence = blocked_write  # type: ignore[method-assign]
        thread = threading.Thread(
            target=reconcile_skill_projection,
            args=(store, vault, skill_id),
            daemon=True,
        )
        thread.start()
        if not blocking_started.wait(5):
            raise SystemExit("reconciler did not reach the injected vault wait")
        unrelated_id = store.save_skill(
            SkillRecord("Concurrent DB Save", "during vault wait", "still commits", "projection")
        )
        if unrelated_id < 1:
            raise SystemExit("SQLite mutation was blocked by projection I/O")
        release_write.set()
        thread.join(5)
        vault.write_skill_with_evidence = original_write  # type: ignore[method-assign]
        if thread.is_alive():
            raise SystemExit("blocked projection worker did not finish")
        _require_completed(store, vault, unrelated_id)

        delete_row = _row(store, skill_id)
        delete_result = store.delete_skill_exact(
            skill_id, int(delete_row["revision"]), str(delete_row["name"])
        )
        if delete_result.status != "deleted":
            raise SystemExit(f"exact delete did not create a tombstone: {delete_result}")
        original_delete = vault.delete_skill_projection

        def fail_delete(*args, **kwargs):
            raise OSError("injected delete failure")

        vault.delete_skill_projection = fail_delete  # type: ignore[method-assign]
        failed_delete = reconcile_skill_projection(store, vault, skill_id)
        if failed_delete.status != "pending_error":
            raise SystemExit("delete failure did not remain pending")
        vault.delete_skill_projection = original_delete  # type: ignore[method-assign]
        deleted = reconcile_skill_projection(store, vault, skill_id)
        if deleted.status != "completed" or note.exists():
            raise SystemExit(f"pending delete did not repair: {deleted}")

        crash_id = store.save_skill(
            SkillRecord("Quarantine Resume", "after crash", "delete me", "projection")
        )
        crash_publish = _require_completed(store, vault, crash_id)
        crash_note = _path(vault, crash_publish.path_display)
        crash_row = _row(store, crash_id)
        store.delete_skill_exact(crash_id, int(crash_row["revision"]), str(crash_row["name"]))
        quarantine = crash_note.with_name(f".{crash_note.name}.delete-{crash_id}.pending")
        os.rename(crash_note, quarantine)
        resumed = reconcile_skill_projection(store, vault, crash_id)
        if resumed.status != "completed" or crash_note.exists() or quarantine.exists():
            raise SystemExit("delete reconciliation did not resume a quarantined projection")

        duplicate_id = store.save_skill(
            SkillRecord("Quarantine Duplicate", "after double crash", "remove both", "projection")
        )
        duplicate_publish = _require_completed(store, vault, duplicate_id)
        duplicate_note = _path(vault, duplicate_publish.path_display)
        duplicate_row = _row(store, duplicate_id)
        store.delete_skill_exact(
            duplicate_id, int(duplicate_row["revision"]), str(duplicate_row["name"])
        )
        duplicate_quarantine = duplicate_note.with_name(
            f".{duplicate_note.name}.delete-{duplicate_id}.pending"
        )
        duplicate_quarantine.write_bytes(duplicate_note.read_bytes())
        duplicate_delete = reconcile_skill_projection(store, vault, duplicate_id)
        if (
            duplicate_delete.status != "completed"
            or duplicate_note.exists()
            or duplicate_quarantine.exists()
        ):
            raise SystemExit("delete recovery completed while an owned source duplicate survived")

        mismatch_id = store.save_skill(
            SkillRecord("Ownership Mismatch", "when foreign", "owned first", "projection")
        )
        mismatch_publish = _require_completed(store, vault, mismatch_id)
        mismatch_note = _path(vault, mismatch_publish.path_display)
        mismatch_note.write_text("foreign content\n", encoding="utf-8")
        mismatch_row = _row(store, mismatch_id)
        store.delete_skill_exact(
            mismatch_id, int(mismatch_row["revision"]), str(mismatch_row["name"])
        )
        mismatch = reconcile_skill_projection(store, vault, mismatch_id)
        mismatch_job = store.get_skill_projection_job(mismatch_id)
        if (
            mismatch.status != "completed"
            or mismatch_job["completion_status"] != "ownership_mismatch"
            or mismatch_note.read_text(encoding="utf-8") != "foreign content\n"
        ):
            raise SystemExit("foreign projection was not preserved with truthful status")

        body_key_id = store.save_skill(
            SkillRecord(
                "Ownership Text In Body",
                "when body contains metadata-looking text",
                "Step one.\njarvis_projection: skill\nStep two.",
                "",
            )
        )
        _require_completed(store, vault, body_key_id)
        body_key_row = _row(store, body_key_id)
        store.delete_skill_exact(
            body_key_id, int(body_key_row["revision"]), str(body_key_row["name"])
        )
        body_key_delete = reconcile_skill_projection(store, vault, body_key_id)
        if body_key_delete.completion_status != "deleted":
            raise SystemExit("Markdown body text invalidated frontmatter-only ownership")

        first_store, shared_vault = _setup(root / "multi", "first")
        second_store, _ = _setup(root / "multi", "second")
        first_id = first_store.save_skill(SkillRecord("Same Name", "one", "first", ""))
        second_id = second_store.save_skill(SkillRecord("Same Name", "two", "second", ""))
        first = _require_completed(first_store, shared_vault, first_id)
        second = _require_completed(second_store, shared_vault, second_id)
        if first_id != second_id or first.path_display == second.path_display:
            raise SystemExit("store-local skill IDs did not receive store-unique paths")
        if not _path(shared_vault, first.path_display).exists() or not _path(shared_vault, second.path_display).exists():
            raise SystemExit("one store displaced another store's projection")

        symlink_store, symlink_vault = _setup(root / "symlink")
        symlink_id = symlink_store.save_skill(
            SkillRecord("Symlink Target", "when unsafe", "must stay pending", "")
        )
        symlink_job = symlink_store.get_skill_projection_job(symlink_id)
        external = root / "external.txt"
        external.write_text("foreign\n", encoding="utf-8")
        target = (
            symlink_vault.root_path
            / "Skills"
            / f"Symlink Target [{symlink_id}-{symlink_job['store_identity']}].md"
        )
        target.symlink_to(external)
        symlink_outcome = reconcile_skill_projection(symlink_store, symlink_vault, symlink_id)
        if symlink_outcome.status != "pending_error" or external.read_text(encoding="utf-8") != "foreign\n":
            raise SystemExit("symlinked projection destination was not rejected safely")

        legacy_store, legacy_vault = _setup(root / "legacy")
        legacy_id = legacy_store.save_skill(
            SkillRecord("Legacy Backfill", "after migration", "publish an old row", "")
        )
        with legacy_store.connect() as conn:
            conn.execute("DELETE FROM skill_projection_jobs WHERE skill_id = ?", (legacy_id,))
        edited_legacy = legacy_vault.root_path / "Skills" / "Legacy Backfill.md"
        edited_legacy.write_text(
            "---\nname: Legacy Backfill\ntrigger: after migration\ntags: \n---\n\n"
            "# Legacy Backfill\n\n## Trigger\n\nafter migration\n\n"
            "## Procedure\n\nuser edited this note\n",
            encoding="utf-8",
        )
        legacy_store.init()
        legacy_job = legacy_store.get_skill_projection_job(legacy_id)
        if legacy_job is None or legacy_job["state"] != "pending":
            raise SystemExit("schema initialization did not backfill a legacy skill projection")
        legacy_projection = _require_completed(legacy_store, legacy_vault, legacy_id)
        if edited_legacy.read_text(encoding="utf-8").endswith("user edited this note\n") is not True:
            raise SystemExit("legacy backfill overwrote a user-edited unowned note")
        if _path(legacy_vault, legacy_projection.path_display) == edited_legacy:
            raise SystemExit("user-edited legacy note was adopted as an owned projection")

        receipt_store, receipt_vault = _setup(root / "receipt-race")
        save_handler = make_skill_tools(receipt_store, receipt_vault)[0]
        original_save_target = receipt_store.save_skill_with_projection_target

        def delete_before_receipt(record):
            target = original_save_target(record)
            row = _row(receipt_store, target.skill_id)
            receipt_store.delete_skill_exact(
                target.skill_id, int(row["revision"]), str(row["name"])
            )
            return target

        receipt_store.save_skill_with_projection_target = delete_before_receipt  # type: ignore[method-assign]
        raced_receipt = save_handler(
            {"name": "Receipt Race", "trigger": "now", "body": "do not claim success"}
        )
        if raced_receipt.ok or raced_receipt.metadata.get("skill_projection_status") != "superseded":
            raise SystemExit("save producer borrowed a concurrently completed delete receipt")

        startup_root = root / "startup"
        startup_store = MemoryStore(startup_root / "jarvis.sqlite")
        startup_store.init()
        startup_ids = [
            startup_store.save_skill(
                SkillRecord(
                    f"Startup Repair {index}",
                    "after restart",
                    f"repair projection {index}",
                    "",
                )
            )
            for index in range(21)
        ]
        runtime = JarvisRuntime(
            JarvisConfig(
                data_dir=startup_root,
                db_path=startup_root / "jarvis.sqlite",
                obsidian_vault=startup_root / "Vault",
                obsidian_root="Jarvis",
                use_model_planner=False,
                watched_dirs=(startup_root / "Watched",),
            )
        )
        startup_job = runtime.store.get_skill_projection_job(startup_ids[0])
        if (
            startup_job is None
            or startup_job["state"] != "completed"
            or runtime.skill_projection_recovery_status.get("status") != "pending"
            or runtime.skill_projection_recovery_status.get("completed") != 20
            or runtime.skill_projection_recovery_status.get("pending") != 1
            or not (runtime.vault.root_path / startup_job["path_display"]).exists()
        ):
            raise SystemExit("runtime startup did not repair a pending skill projection")
        second_runtime = JarvisRuntime(runtime.config)
        if (
            second_runtime.skill_projection_recovery_status.get("status") != "completed"
            or second_runtime.skill_projection_recovery_status.get("pending") != 0
        ):
            raise SystemExit("bounded startup repair did not converge on the next startup")

    print("Skill projection reconciliation smoke passed")


if __name__ == "__main__":
    main()
