from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jarvis_v2.memory.store import MemoryRecord
from jarvis_v2.scripts.test_runtime import make_temp_runtime


CATEGORY = "profile-isolation"
QUERY = "durable isolation sentinel"
ORDINARY_BODY = f"Ordinary memory carries the {QUERY}."
PROFILE_CASES = {
    "completed": f"Completed profile note carries the {QUERY}.",
    "pending": f"Pending profile note carries the {QUERY}.",
    "source-mutated": f"Source-mutated profile note carries the {QUERY}.",
}


def _add_profile_note(runtime: Any, label: str, body: str) -> int:
    result = runtime.registry.get("add_profile_note").handler(
        {
            "heading": f"Profile isolation {label}",
            "body": body,
            "category": CATEGORY,
        }
    )
    if not result.ok or type(result.metadata.get("memory_id")) is not int:
        raise AssertionError(f"{label} profile fixture failed: {result}")
    return int(result.metadata["memory_id"])


def _ids(rows: list[Any]) -> set[int]:
    return {int(row["id"]) for row in rows}


def test_generic_memory_readers_exclude_durable_profile_ownership() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-generic-isolation-") as temp:
        runtime = make_temp_runtime(Path(temp))
        profile_path = runtime.vault.root_path / "Profile.md"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text("# Profile\n\nManual profile context.\n", encoding="utf-8")

        profile_ids = {
            label: _add_profile_note(runtime, label, body)
            for label, body in PROFILE_CASES.items()
        }
        ordinary_id = runtime.store.add_memory(
            MemoryRecord(CATEGORY, "Ordinary isolation control", ORDINARY_BODY, "manual", 0.8)
        )
        source_only_id = runtime.store.add_memory(
            MemoryRecord(
                CATEGORY,
                "Mutable source label control",
                f"A source-only row also carries the {QUERY}.",
                "profile",
                0.7,
            )
        )

        with runtime.store.connect() as conn:
            conn.execute(
                """
                UPDATE ingested_sources
                SET mirror_state = 'pending', mirror_completed_at = NULL
                WHERE source_type = 'profile_note' AND memory_id = ?
                """,
                (profile_ids["pending"],),
            )
            conn.execute(
                "UPDATE memories SET source = 'manual' WHERE id = ?",
                (profile_ids["source-mutated"],),
            )

        expected = {ordinary_id, source_only_id}
        generic_results = {
            "recent": _ids(runtime.store.recent_memories(limit=100)),
            "search": _ids(runtime.store.search_memories(QUERY, limit=100)),
            "list": _ids(runtime.store.list_memories(limit=100)),
            "category-list": _ids(runtime.store.list_memories(category=CATEGORY, limit=100)),
        }
        for label, visible_ids in generic_results.items():
            if visible_ids != expected:
                raise AssertionError(
                    f"generic {label} did not isolate durable profile ownership: {visible_ids}"
                )
        bounded_results = {
            "recent": runtime.store.recent_memories(limit=1),
            "search": runtime.store.search_memories(QUERY, limit=1),
            "list": runtime.store.list_memories(limit=1),
            "category-list": runtime.store.list_memories(category=CATEGORY, limit=1),
        }
        for label, rows in bounded_results.items():
            if len(rows) != 1 or _ids(rows) - expected:
                raise AssertionError(f"bounded generic {label} leaked or underfilled: {rows}")
        for memory_id in expected:
            if runtime.store.get_memory(memory_id) is None:
                raise AssertionError(f"generic direct get hid ordinary row #{memory_id}")
        for label, memory_id in profile_ids.items():
            if runtime.store.get_memory(memory_id) is not None:
                raise AssertionError(f"generic direct get exposed the {label} profile row")

        stats = {str(row["source"]): int(row["count"]) for row in runtime.store.memory_stats()}
        if stats != {"manual": 1, "profile": 1}:
            raise AssertionError(f"generic memory stats included profile-owned rows: {stats}")

        completed_id = profile_ids["completed"]
        with runtime.store.memory_read_custody(completed_id) as custody:
            if (
                custody.row is None
                or not custody.profile_owned
                or custody.profile_note is None
                or custody.profile_note.body != PROFILE_CASES["completed"]
            ):
                raise AssertionError(f"verified explicit memory custody was unavailable: {custody}")
        with runtime.store.memory_read_custody(profile_ids["pending"]) as custody:
            if custody.row is None or not custody.profile_owned or custody.profile_note is not None:
                raise AssertionError(f"pending explicit custody did not fail closed: {custody}")

        snapshot = runtime.store.read_profile_knowledge_snapshot()
        if (
            snapshot.source_count != 3
            or snapshot.invalid_count != 2
            or [note.memory_id for note in snapshot.notes] != [completed_id]
        ):
            raise AssertionError(f"explicit profile snapshot custody drifted: {snapshot}")
        explicit = runtime.registry.get("get_memory").handler({"memory_id": completed_id})
        if (
            not explicit.ok
            or PROFILE_CASES["completed"] not in explicit.output
            or explicit.metadata.get("profile_custody_state") != "verified"
        ):
            raise AssertionError(f"verified dedicated profile read was unavailable: {explicit}")

        source_only_target = runtime.store.resolve_memory_approval_target(source_only_id)
        if source_only_target is None:
            raise AssertionError("source-only profile deletion target is unavailable")
        if runtime.store.delete_memory(source_only_id):
            raise AssertionError("generic deletion removed an orphaned profile-source row")
        exact_delete = runtime.store.delete_memory_exact(
            source_only_id,
            source_only_target.revision,
            source_only_target.binding,
        )
        projected_delete = runtime.store.delete_memory_exact_with_projection(
            source_only_id,
            source_only_target.revision,
            source_only_target.binding,
        )
        if exact_delete.status != "protected" or projected_delete.status != "protected":
            raise AssertionError(
                "exact deletion did not protect an orphaned profile-source row: "
                f"{exact_delete} / {projected_delete}"
            )
        with runtime.store.connect() as conn:
            source_only_row = conn.execute(
                "SELECT source FROM memories WHERE id = ?", (source_only_id,)
            ).fetchone()
            delete_job = conn.execute(
                "SELECT 1 FROM memory_projection_jobs WHERE memory_id = ? AND operation = 'delete'",
                (source_only_id,),
            ).fetchone()
        if source_only_row is None or source_only_row["source"] != "profile" or delete_job is not None:
            raise AssertionError("protected orphaned profile-source deletion mutated durable state")


def main() -> None:
    test_generic_memory_readers_exclude_durable_profile_ownership()
    print("profile generic memory isolation smoke: PASS")


if __name__ == "__main__":
    main()
