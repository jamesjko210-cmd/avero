from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
from unittest.mock import patch

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.config import JarvisConfig
from jarvis_v2.memory.store import MemoryRecord
from jarvis_v2.scripts.test_runtime import make_temp_runtime


@dataclass(frozen=True)
class ProjectionCase:
    title: str
    projection: str
    save_tool: str
    read_tool: str
    vault_method: str
    metadata_count_key: str

    @property
    def relative_path(self) -> str:
        return f"Automations/{self.title}.md"


CASES = (
    ProjectionCase(
        "Feedback Report",
        "feedback_report",
        "save_feedback_report",
        "feedback_report",
        "write_feedback_report_with_evidence",
        "count",
    ),
    ProjectionCase(
        "Feedback Actions",
        "feedback_actions",
        "save_feedback_actions",
        "feedback_actions",
        "write_feedback_actions_with_evidence",
        "count",
    ),
    ProjectionCase(
        "Learning Review",
        "learning_review",
        "save_learning_review",
        "learning_review",
        "write_learning_review_with_evidence",
        "feedback",
    ),
)

PROTECTED_METADATA = (
    "content_sha256",
    "source_revision",
    "write_atomic",
    "snapshot_consistency",
    "store_identity",
)


def _other_runtime(root: Path) -> JarvisRuntime:
    return JarvisRuntime(
        JarvisConfig(
            data_dir=root / "other-data",
            db_path=root / "other.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            use_model_planner=False,
            watched_dirs=(root / "Other Watched",),
        )
    )


def _path(runtime: JarvisRuntime, case: ProjectionCase) -> Path:
    return runtime.vault.root_path / case.relative_path


def _save(runtime: JarvisRuntime, case: ProjectionCase, **args):
    return runtime.registry.get(case.save_tool).handler(args)


def _legacy_text(runtime: JarvisRuntime, case: ProjectionCase) -> str:
    result = runtime.registry.get(case.read_tool).handler({"limit": 12})
    if not result.ok:
        raise SystemExit(f"{case.title} read-only report failed while preparing legacy fixture: {result}")
    return f"# {case.title}\n\n{result.output.rstrip()}\n"


def _expect_refusal(call, message: str) -> None:
    try:
        call()
    except (FileExistsError, OSError, ValueError):
        return
    raise SystemExit(message)


def test_fresh_ownership_metadata_and_generic_note_guards() -> None:
    with TemporaryDirectory(prefix="jarvis-generated-reports-owned-") as temp:
        runtime = make_temp_runtime(Path(temp))
        note_tool = runtime.registry.get("write_jarvis_note").handler
        store_identity = runtime.store.get_store_identity()

        for case in CASES:
            result = _save(runtime, case, limit=12)
            if not result.ok:
                raise SystemExit(f"fresh {case.title} publication failed: {result}")
            path = _path(runtime, case)
            expected_header = (
                "---\n"
                f"jarvis_projection: {case.projection}\n"
                f"store_identity: {store_identity}\n"
                "---\n\n"
                f"# {case.title}\n"
            )
            text = path.read_text(encoding="utf-8")
            if not text.startswith(expected_header):
                raise SystemExit(f"fresh {case.title} missed exact owned frontmatter")
            if Path(result.metadata.get("path", "")) != path:
                raise SystemExit(f"{case.title} receipt did not identify its exact destination")
            for key in PROTECTED_METADATA:
                if key in result.metadata:
                    raise SystemExit(f"{case.title} exposed protected metadata {key}: {result.metadata}")

            before = path.read_bytes()
            note_paths = (case.relative_path.removesuffix(".md"), case.relative_path.lower())
            for note_path in note_paths:
                for mode in ("create", "append"):
                    refused = note_tool({"path": note_path, "body": "must not alter projection", "mode": mode})
                    if refused.ok or refused.metadata.get("reason") != "managed_projection":
                        raise SystemExit(
                            f"generic note writer accepted {case.title} path {note_path!r} in {mode}: {refused}"
                        )
            if path.read_bytes() != before:
                raise SystemExit(f"generic note guard changed {case.title}")


def test_exact_legacy_migration_and_near_match_refusal() -> None:
    for case in CASES:
        with TemporaryDirectory(prefix=f"jarvis-{case.projection}-legacy-") as temp:
            runtime = make_temp_runtime(Path(temp))
            path = _path(runtime, case)
            legacy = _legacy_text(runtime, case)
            path.write_text(legacy, encoding="utf-8")
            migrated = _save(runtime, case)
            if not migrated.ok or not path.read_text(encoding="utf-8").startswith(
                f"---\njarvis_projection: {case.projection}\n"
            ):
                raise SystemExit(f"exact legacy {case.title} did not migrate")

        with TemporaryDirectory(prefix=f"jarvis-{case.projection}-near-match-") as temp:
            runtime = make_temp_runtime(Path(temp))
            path = _path(runtime, case)
            legacy = _legacy_text(runtime, case)
            near_match = legacy.replace(
                legacy.splitlines()[3],
                legacy.splitlines()[3] + " This sentence belongs to a user note.",
                1,
            )
            path.write_text(near_match, encoding="utf-8")
            before = path.read_bytes()
            _expect_refusal(
                lambda: _save(runtime, case),
                f"near-match {case.title} was accepted as a legacy projection",
            )
            if path.read_bytes() != before:
                raise SystemExit(f"near-match refusal changed {case.title}")


def test_foreign_store_identity_refusal() -> None:
    with TemporaryDirectory(prefix="jarvis-generated-reports-foreign-") as temp:
        root = Path(temp)
        owner = make_temp_runtime(root)
        for case in CASES:
            _save(owner, case)

        foreign = _other_runtime(root)
        if foreign.store.get_store_identity() == owner.store.get_store_identity():
            raise SystemExit("foreign-store fixture unexpectedly reused the owner identity")
        for case in CASES:
            path = _path(owner, case)
            before = path.read_bytes()
            _expect_refusal(
                lambda case=case: _save(foreign, case),
                f"foreign database overwrote owned {case.title}",
            )
            if path.read_bytes() != before:
                raise SystemExit(f"foreign-store refusal changed {case.title}")


def test_destination_and_parent_symlink_refusal() -> None:
    for case in CASES:
        with TemporaryDirectory(prefix=f"jarvis-{case.projection}-destination-link-") as temp:
            root = Path(temp)
            runtime = make_temp_runtime(root)
            outside = root / f"outside-{case.projection}.md"
            sentinel = f"outside {case.projection} sentinel\n"
            outside.write_text(sentinel, encoding="utf-8")
            _path(runtime, case).symlink_to(outside)
            _expect_refusal(
                lambda: _save(runtime, case),
                f"{case.title} destination symlink was accepted",
            )
            if outside.read_text(encoding="utf-8") != sentinel:
                raise SystemExit(f"{case.title} destination symlink changed its external target")

        with TemporaryDirectory(prefix=f"jarvis-{case.projection}-parent-link-") as temp:
            root = Path(temp)
            runtime = make_temp_runtime(root)
            automations = runtime.vault.root_path / "Automations"
            for child in automations.iterdir():
                child.unlink()
            automations.rmdir()
            outside_dir = root / "outside-automations"
            outside_dir.mkdir()
            automations.symlink_to(outside_dir, target_is_directory=True)
            _expect_refusal(
                lambda: _save(runtime, case),
                f"{case.title} parent-directory symlink was accepted",
            )
            if list(outside_dir.iterdir()):
                raise SystemExit(f"{case.title} parent-directory symlink wrote outside the vault")


def test_replace_failure_preserves_bytes_and_concurrent_writers_are_complete() -> None:
    from jarvis_v2.memory import obsidian as obsidian_module

    for case in CASES:
        with TemporaryDirectory(prefix=f"jarvis-{case.projection}-atomic-") as temp:
            runtime = make_temp_runtime(Path(temp))
            runtime.store.add_memory(
                MemoryRecord(
                    "feedback",
                    "Conversation feedback",
                    "Conversation responses should be shorter",
                    "smoke",
                )
            )
            runtime.store.add_memory(
                MemoryRecord("feedback", "Safety feedback", "Explain approval risk clearly", "smoke")
            )
            path = _path(runtime, case)
            _save(runtime, case, limit=1)
            prior = path.read_bytes()
            original_replace = obsidian_module._replace_text

            def fail_target(root, target, content, **kwargs):
                if target.name == path.name:
                    raise OSError(f"simulated {case.projection} replacement failure")
                return original_replace(root, target, content, **kwargs)

            with patch("jarvis_v2.memory.obsidian._replace_text", side_effect=fail_target):
                _expect_refusal(
                    lambda: _save(runtime, case, limit=2),
                    f"{case.title} hid its injected replacement failure",
                )
            if path.read_bytes() != prior:
                raise SystemExit(f"failed {case.title} replacement changed prior bytes")

            _save(runtime, case, limit=1)
            expected_one = path.read_bytes()
            _save(runtime, case, limit=2)
            expected_two = path.read_bytes()
            if expected_one == expected_two:
                raise SystemExit(f"{case.title} concurrency fixtures were not distinct")
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = (
                    pool.submit(_save, runtime, case, limit=1),
                    pool.submit(_save, runtime, case, limit=2),
                )
                results = [future.result(timeout=10) for future in futures]
            if any(not result.ok for result in results):
                raise SystemExit(f"concurrent {case.title} publication failed: {results}")
            final = path.read_bytes()
            if final not in (expected_one, expected_two):
                raise SystemExit(f"concurrent writers left a mixed or incomplete {case.title}")


def test_same_timestamp_memory_order_is_newest_first() -> None:
    with TemporaryDirectory(prefix="jarvis-generated-reports-tied-time-") as temp:
        runtime = make_temp_runtime(Path(temp))
        older_id = runtime.store.add_memory(
            MemoryRecord("feedback", "Older feedback", "Conversation should be concise", "smoke")
        )
        newer_id = runtime.store.add_memory(
            MemoryRecord("feedback", "Newer feedback", "Explain approval risk clearly", "smoke")
        )
        tied_time = "2026-08-14T00:00:00Z"
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE memories SET created_at = ?, updated_at = ? WHERE id IN (?, ?)",
                (tied_time, tied_time, older_id, newer_id),
            )

        expected = [newer_id, older_id]
        recent_ids = [int(row["id"]) for row in runtime.store.recent_memories(limit=2)]
        category_ids = [
            int(row["id"])
            for row in runtime.store.list_memories(category="feedback", limit=2)
        ]
        all_ids = [int(row["id"]) for row in runtime.store.list_memories(limit=2)]
        if recent_ids != expected or category_ids != expected or all_ids != expected:
            raise SystemExit(
                "same-timestamp memory ordering was not deterministically newest-first"
            )


def test_database_mutation_waits_for_publication() -> None:
    for case in CASES:
        with TemporaryDirectory(prefix=f"jarvis-{case.projection}-fence-") as temp:
            runtime = make_temp_runtime(Path(temp))
            entered = Event()
            release = Event()
            original_write = getattr(runtime.vault, case.vault_method)

            def blocking_write(*args, **kwargs):
                entered.set()
                if not release.wait(timeout=5):
                    raise TimeoutError(f"{case.title} publication release timed out")
                return original_write(*args, **kwargs)

            setattr(runtime.vault, case.vault_method, blocking_write)
            try:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    publishing = pool.submit(_save, runtime, case, limit=12)
                    if not entered.wait(timeout=5):
                        raise SystemExit(f"{case.title} did not reach its publication write")
                    mutating = pool.submit(
                        runtime.store.add_memory,
                        MemoryRecord(
                            "feedback",
                            "Fenced feedback",
                            "Explain approval risk after the durable report is published",
                            "smoke",
                        ),
                    )
                    time.sleep(0.1)
                    if mutating.done():
                        raise SystemExit(f"{case.title} publication fence allowed a database commit early")
                    release.set()
                    published = publishing.result(timeout=5)
                    mutating.result(timeout=5)
            finally:
                setattr(runtime.vault, case.vault_method, original_write)
                release.set()

            if published.metadata.get(case.metadata_count_key) != 0:
                raise SystemExit(f"{case.title} included a mutation committed after snapshot capture")
            first = _path(runtime, case).read_bytes()
            refreshed = _save(runtime, case, limit=12)
            if refreshed.metadata.get(case.metadata_count_key) != 1:
                raise SystemExit(f"refreshed {case.title} missed the later committed database mutation")
            if _path(runtime, case).read_bytes() == first:
                raise SystemExit(f"refreshed {case.title} did not publish the newer snapshot")


def main() -> None:
    test_fresh_ownership_metadata_and_generic_note_guards()
    test_exact_legacy_migration_and_near_match_refusal()
    test_foreign_store_identity_refusal()
    test_destination_and_parent_symlink_refusal()
    test_replace_failure_preserves_bytes_and_concurrent_writers_are_complete()
    test_same_timestamp_memory_order_is_newest_first()
    test_database_mutation_waits_for_publication()
    print("generated report projection smoke passed")


if __name__ == "__main__":
    main()
