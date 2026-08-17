from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jarvis_v2.memory.store import MemoryRecord
from jarvis_v2.scripts.test_runtime import make_temp_runtime


HEADING = "Custody invariant"
BODY = "Profile-owned memories stay aligned with their canonical projections."
CATEGORY = "identity"
ATTACK = "UNVERIFIED_PROFILE_MUTATION_PAYLOAD"


def _runtime(root: Path):
    runtime = make_temp_runtime(root)
    profile_path = runtime.vault.root_path / "Profile.md"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text("# Profile\n\nManual profile context.\n", encoding="utf-8")
    result = runtime.registry.get("add_profile_note").handler(
        {"heading": HEADING, "body": BODY, "category": CATEGORY}
    )
    if not result.ok:
        raise AssertionError(f"profile fixture failed: {result}")
    return runtime, int(result.metadata["memory_id"])


def _row(runtime: Any, table: str, where: str, params: tuple[object, ...]) -> dict[str, Any]:
    with runtime.store.connect() as conn:
        row = conn.execute(f"SELECT * FROM {table} WHERE {where}", params).fetchone()
    if row is None:
        raise AssertionError(f"missing {table} fixture row")
    return dict(row)


def _source(runtime: Any) -> dict[str, Any]:
    return _row(runtime, "ingested_sources", "source_type = ?", ("profile_note",))


def _memory(runtime: Any, memory_id: int) -> dict[str, Any]:
    return _row(runtime, "memories", "id = ?", (memory_id,))


def test_all_generic_updates_and_both_merge_sides_are_protected() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-mutation-guards-") as temp:
        runtime, profile_id = _runtime(Path(temp))
        manual_id = runtime.store.add_memory(
            MemoryRecord("manual", "Merge peer", "manual peer body", "manual", 0.8)
        )
        profile_target = runtime.store.resolve_memory_approval_target(profile_id)
        manual_target = runtime.store.resolve_memory_approval_target(manual_id)
        if profile_target is None or manual_target is None:
            raise AssertionError("mutation approval targets are unavailable")
        before_profile = _memory(runtime, profile_id)
        before_manual = _memory(runtime, manual_id)

        if runtime.store.update_memory(profile_id, "changed", "changed", ATTACK, 0.1):
            raise AssertionError("generic update changed a profile-owned memory")
        if runtime.store.update_memory_exact(
            profile_id,
            profile_target.revision,
            profile_target.binding,
            body=ATTACK,
        ).status != "protected":
            raise AssertionError("exact update bypassed profile ownership")
        if runtime.store.update_memory_exact_with_projection(
            profile_id,
            profile_target.revision,
            profile_target.binding,
            body=ATTACK,
        ).status != "protected":
            raise AssertionError("projected exact update bypassed profile ownership")

        if runtime.store.merge_memories(profile_id, manual_id):
            raise AssertionError("generic merge changed a profile-owned survivor")
        if runtime.store.merge_memories(manual_id, profile_id):
            raise AssertionError("generic merge deleted a profile-owned source")
        for keep, delete in (
            (profile_target, manual_target),
            (manual_target, profile_target),
        ):
            exact = runtime.store.merge_memories_exact(
                keep.memory_id,
                keep.revision,
                keep.binding,
                delete.memory_id,
                delete.revision,
                delete.binding,
            )
            projected = runtime.store.merge_memories_exact_with_projection(
                keep.memory_id,
                keep.revision,
                keep.binding,
                delete.memory_id,
                delete.revision,
                delete.binding,
            )
            if exact.status != "protected" or projected.status != "protected":
                raise AssertionError(
                    f"exact merge did not protect both sides: {exact} / {projected}"
                )

        edit = runtime.registry.get("edit_memory").handler(
            {
                "memory_id": profile_id,
                "target_revision": profile_target.revision,
                "target_binding": profile_target.binding,
                "body": ATTACK,
            }
        )
        merge = runtime.registry.get("merge_memories").handler(
            {
                "keep_id": manual_id,
                "keep_revision": manual_target.revision,
                "keep_binding": manual_target.binding,
                "delete_id": profile_id,
                "delete_revision": profile_target.revision,
                "delete_binding": profile_target.binding,
            }
        )
        if edit.ok or edit.metadata.get("target_status") != "protected":
            raise AssertionError(f"edit tool did not expose profile protection: {edit}")
        if merge.ok or merge.metadata.get("target_status") != "protected":
            raise AssertionError(f"merge tool did not expose profile protection: {merge}")
        if _memory(runtime, profile_id) != before_profile or _memory(runtime, manual_id) != before_manual:
            raise AssertionError("a refused profile mutation changed a memory row")


def test_get_memory_requires_current_profile_projection_custody() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-get-custody-") as temp:
        runtime, profile_id = _runtime(Path(temp))
        get_memory = runtime.registry.get("get_memory").handler
        verified = get_memory({"memory_id": profile_id})
        if (
            not verified.ok
            or BODY not in verified.output
            or verified.metadata.get("profile_custody_state") != "verified"
        ):
            raise AssertionError(f"healthy profile memory was not readable: {verified}")

        job = _row(runtime, "memory_projection_jobs", "memory_id = ?", (profile_id,))
        note_path = runtime.vault.root_path / str(job["canonical_path_display"])
        note_path.write_text(ATTACK, encoding="utf-8")
        refused = get_memory({"memory_id": profile_id})
        if (
            refused.ok
            or BODY in refused.output
            or ATTACK in refused.output
            or refused.metadata.get("profile_custody_state") != "unavailable"
        ):
            raise AssertionError(f"tampered profile projection was disclosed: {refused}")

    with TemporaryDirectory(prefix="jarvis-profile-get-pending-") as temp:
        runtime, profile_id = _runtime(Path(temp))
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE ingested_sources SET mirror_state = 'pending', "
                "mirror_completed_at = NULL WHERE source_type = 'profile_note'"
            )
        refused = runtime.registry.get("get_memory").handler({"memory_id": profile_id})
        if refused.ok or BODY in refused.output:
            raise AssertionError(f"pending profile custody was disclosed: {refused}")

        orphan_id = runtime.store.add_memory(
            MemoryRecord("identity", "Orphan profile row", ATTACK, "profile", 1.0)
        )
        orphan = runtime.registry.get("get_memory").handler({"memory_id": orphan_id})
        if orphan.ok or ATTACK in orphan.output:
            raise AssertionError(f"uncustodied source=profile row was disclosed: {orphan}")


def test_read_profile_revalidates_after_the_final_file_read() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-read-fence-") as temp:
        runtime, _profile_id = _runtime(Path(temp))
        original = runtime.vault.read_profile_grounding

        def read_then_tamper(*args, **kwargs):
            view = original(*args, **kwargs)
            path = runtime.vault.root_path / "Profile.md"
            path.write_text(
                path.read_text(encoding="utf-8").replace(BODY, ATTACK),
                encoding="utf-8",
            )
            return view

        runtime.vault.read_profile_grounding = read_then_tamper
        try:
            result = runtime.registry.get("read_profile").handler({})
        finally:
            runtime.vault.read_profile_grounding = original
        if (
            not result.ok
            or BODY in result.output
            or ATTACK in result.output
            or result.metadata.get("profile_custody_state") != "unavailable"
        ):
            raise AssertionError(f"read-time profile tamper escaped the custody fence: {result}")


def test_same_identity_stale_generation_cannot_reopen_newer_completion() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-generation-fence-") as temp:
        runtime, profile_id = _runtime(Path(temp))
        source = _source(runtime)
        record = MemoryRecord(CATEGORY, HEADING, BODY, "profile", 1.0)
        source_key = str(source["source_key"])
        _created, _repaired, older = runtime.store.ensure_profile_note_memory_with_projection(
            record, source_key
        )
        if not runtime.store.mark_ingested_source_projection_pending(
            source_key,
            profile_id,
            older.revision,
            older.source_digest,
            expected_generation=older.attempt_generation,
        ):
            raise AssertionError("current generation could not reserve pending custody")
        _created, _repaired, newer = runtime.store.ensure_profile_note_memory_with_projection(
            record, source_key
        )
        if newer.attempt_generation <= older.attempt_generation:
            raise AssertionError("profile mirror generation did not advance monotonically")
        evidence = runtime.store.complete_ingested_source_projection(
            source_key,
            profile_id,
            newer.revision,
            newer.source_digest,
            expected_generation=newer.attempt_generation,
        )
        if evidence["completed_now"] is not True:
            raise AssertionError("newer profile attempt did not complete pending custody")
        if runtime.store.mark_ingested_source_projection_pending(
            source_key,
            profile_id,
            older.revision,
            older.source_digest,
            expected_generation=older.attempt_generation,
        ):
            raise AssertionError("stale same-identity attempt reopened newer custody")
        try:
            runtime.store.complete_ingested_source_projection(
                source_key,
                profile_id,
                older.revision,
                older.source_digest,
                expected_generation=older.attempt_generation,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("stale same-identity attempt replayed completion")
        final_source = _source(runtime)
        if (
            final_source["mirror_state"] != "completed"
            or final_source["mirror_generation"] != newer.attempt_generation
        ):
            raise AssertionError(f"newer completed custody was damaged: {final_source}")


def test_required_local_path_forms_are_refused() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-local-paths-") as temp:
        runtime = make_temp_runtime(Path(temp))
        add_note = runtime.registry.get("add_profile_note").handler
        for local_path in (
            "C:/\x55sers/the operator/private.txt",
            "/System/Volumes/Data/\x55sers/the operator/private.txt",
            "/root/.ssh/id_ed25519",
        ):
            result = add_note(
                {
                    "heading": "Path refusal",
                    "body": f"Do not retain {local_path}",
                    "category": "identity",
                }
            )
            if result.ok or result.metadata.get("reason") != "invalid_body":
                raise AssertionError(f"local path was accepted: {local_path} / {result}")
        with runtime.store.connect() as conn:
            counts = tuple(
                int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in ("memories", "ingested_sources")
            )
        if counts != (0, 0):
            raise AssertionError(f"refused local paths left durable rows: {counts}")


def test_future_profile_marker_is_refused_before_any_mutation() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-future-marker-") as temp:
        runtime = make_temp_runtime(Path(temp))
        result = runtime.registry.get("add_profile_note").handler(
            {
                "heading": "Future marker refusal",
                "body": "<!-- jarvis-profile-note-start:v2:bad -->\n- [ ] protected",
                "category": "identity",
            }
        )
        if result.ok or result.metadata.get("reason") != "reserved_marker_namespace":
            raise AssertionError(f"future profile marker was not refused early: {result}")
        with runtime.store.connect() as conn:
            counts = tuple(
                int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in ("memories", "ingested_sources", "memory_projection_jobs")
            )
        if counts != (0, 0, 0):
            raise AssertionError(f"future marker refusal left durable rows: {counts}")


def main() -> None:
    test_all_generic_updates_and_both_merge_sides_are_protected()
    test_get_memory_requires_current_profile_projection_custody()
    test_read_profile_revalidates_after_the_final_file_read()
    test_same_identity_stale_generation_cannot_reopen_newer_completion()
    test_required_local_path_forms_are_refused()
    test_future_profile_marker_is_refused_before_any_mutation()
    print("profile mutation guards smoke passed")


if __name__ == "__main__":
    main()
