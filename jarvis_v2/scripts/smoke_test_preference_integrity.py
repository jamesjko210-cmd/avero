from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier

from jarvis_v2.memory.store import (
    MemoryStore,
    PreferenceRecord,
)
from jarvis_v2.scripts.test_runtime import make_temp_runtime


def _store(root: Path, name: str = "preferences.sqlite") -> MemoryStore:
    store = MemoryStore(root / name)
    store.init()
    return store


def test_case_insensitive_upsert_preserves_one_identity() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-case-") as temp:
        store = _store(Path(temp))
        first = store.set_preference(PreferenceRecord("Voice Tone", "warm", "Communication"))
        second = store.set_preference(PreferenceRecord("voice tone", "direct", "communication"))
        with store.connect() as conn:
            rows = list(conn.execute("SELECT * FROM preferences"))
        if first != second or len(rows) != 1:
            raise SystemExit(f"case variant created a second preference identity: {first} / {second} / {rows}")
        row = rows[0]
        if row["category"] != "Communication" or row["key"] != "Voice Tone" or row["value"] != "direct":
            raise SystemExit(f"case-insensitive update did not preserve stable identity casing: {dict(row)}")


def test_concurrent_case_variants_cannot_split_identity() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-concurrent-") as temp:
        root = Path(temp)
        first_store = _store(root)
        second_store = MemoryStore(first_store.db_path)
        second_store.init()
        barrier = Barrier(2)

        def write(store: MemoryStore, key: str, value: str, category: str) -> int:
            barrier.wait(timeout=10)
            return store.set_preference(PreferenceRecord(key, value, category))

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(write, first_store, "Response Style", "warm", "Communication"),
                pool.submit(write, second_store, "response style", "direct", "communication"),
            ]
            ids = [future.result(timeout=10) for future in futures]
        with first_store.connect() as conn:
            rows = list(conn.execute("SELECT * FROM preferences"))
        if len(set(ids)) != 1 or len(rows) != 1 or rows[0]["value"] not in {"warm", "direct"}:
            raise SystemExit(f"concurrent case variants split preference identity: {ids} / {rows}")


def test_legacy_case_collision_uses_stable_owner_without_merging() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-legacy-conflict-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        with runtime.store.connect() as conn:
            conn.execute(
                """
                INSERT INTO preferences(category, key, value, status, created_at, updated_at)
                VALUES ('Communication', 'Tone', 'warm', 'active', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
                """
            )
            conn.execute(
                """
                INSERT INTO preferences(category, key, value, status, created_at, updated_at)
                VALUES ('communication', 'tone', 'direct', 'active', '2026-01-02T00:00:00Z', '2026-01-02T00:00:00Z')
                """
            )
        runtime.store.init()
        owner_id = runtime.store.set_preference(PreferenceRecord("TONE", "quiet", "COMMUNICATION"))
        result = runtime.registry.get("set_preference").handler(
            {"key": "tone", "value": "calm", "category": "communication"}
        )
        if owner_id != 1 or not result.ok or result.metadata.get("preference_id") != 1:
            raise SystemExit(f"legacy preference identity did not use the lowest stable owner: {owner_id} / {result}")
        with runtime.store.connect() as conn:
            rows = list(conn.execute("SELECT key, value FROM preferences ORDER BY id"))
            owners = list(conn.execute("SELECT * FROM preference_identity_owners"))
        if [(row["key"], row["value"]) for row in rows] != [("Tone", "calm"), ("tone", "direct")]:
            raise SystemExit(f"canonical update modified a legacy shadow row: {rows}")
        if len(owners) != 1 or owners[0]["preference_id"] != 1:
            raise SystemExit(f"legacy identity owner mapping was not stable: {owners}")
        read = runtime.store.get_preference("TONE", "COMMUNICATION")
        if read is None or read["id"] != 1 or read["value"] != "calm":
            raise SystemExit(f"categorized preference read did not use the canonical owner: {dict(read) if read else None}")
        note = runtime.vault.root_path / "Memory Tree" / "Preferences.md"
        text = note.read_text(encoding="utf-8")
        if text.count("Tone: calm") != 1 or text.count("tone: direct") != 1:
            raise SystemExit("full mirror did not preserve both canonical and shadow legacy rows")


def test_full_mirror_includes_rows_beyond_old_limit() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-full-mirror-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        for index in range(205):
            runtime.store.set_preference(
                PreferenceRecord(f"preference {index:03d}", f"value {index:03d}", "bulk")
            )
        result = runtime.registry.get("set_preference").handler(
            {"key": "preference 204", "value": "updated final value", "category": "bulk"}
        )
        if not result.ok:
            raise SystemExit(f"full preference mirror update failed: {result}")
        note = runtime.vault.root_path / "Memory Tree" / "Preferences.md"
        text = note.read_text(encoding="utf-8")
        if "preference 000: value 000" not in text or "preference 204: updated final value" not in text:
            raise SystemExit("canonical preference mirror silently truncated rows beyond the old 200 limit")
        if text.count("- #") != 205:
            raise SystemExit(f"canonical preference mirror did not publish all 205 rows: {text.count('- #')}")


def main() -> None:
    test_case_insensitive_upsert_preserves_one_identity()
    test_concurrent_case_variants_cannot_split_identity()
    test_legacy_case_collision_uses_stable_owner_without_merging()
    test_full_mirror_includes_rows_beyond_old_limit()
    print("Preference identity and projection smoke passed")


if __name__ == "__main__":
    main()
