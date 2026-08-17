from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
from unittest.mock import patch

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.config import JarvisConfig
from jarvis_v2.memory.store import MemoryRecord, TaskRecord
from jarvis_v2.scripts.test_runtime import make_temp_runtime


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


def test_owned_atomic_state_and_memory_projections() -> None:
    with TemporaryDirectory(prefix="jarvis-state-projection-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        runtime.store.add_memory(MemoryRecord("project", "Projection memory", "bounded projection body", "smoke"))
        state_tool = runtime.registry.get("export_state_snapshot").handler
        memory_tool = runtime.registry.get("memory_tree_summary").handler

        for invalid_args in ({"unexpected": "private value"}, [], None):
            invalid = state_tool(invalid_args)  # type: ignore[arg-type]
            if invalid.ok or invalid.metadata.get("failure_kind") != "invalid_arguments":
                raise SystemExit(f"state export accepted arguments: {invalid}")

        exported = state_tool({})
        if not exported.ok:
            raise SystemExit(f"state export failed: {exported}")
        state_path = Path(exported.metadata["path"])
        state_text = state_path.read_text(encoding="utf-8")
        if (
            "jarvis_projection: current_context" not in state_text
            or "# Current Context" not in state_text
            or exported.metadata.get("content_sha256") != hashlib.sha256(state_text.encode("utf-8")).hexdigest()
        ):
            raise SystemExit("state projection missed ownership or content evidence")
        for key, expected in {
            "writes_files": True,
            "writes_notes": True,
            "writes_database": False,
            "writes_memory": False,
            "reads_personal_data": True,
            "reads_private_data": True,
            "write_atomic": True,
            "snapshot_consistency": "transaction_fenced",
        }.items():
            if exported.metadata.get(key) != expected:
                raise SystemExit(f"state projection metadata drifted for {key}: {exported.metadata}")

        before_state = state_path.read_bytes()
        summarized = memory_tool({"limit": 200})
        memory_path = Path(summarized.metadata["path"])
        if memory_path.name != "Memory Tree Snapshot.md" or memory_path == state_path:
            raise SystemExit(f"memory tree summary retained the Current Context destination: {memory_path}")
        if state_path.read_bytes() != before_state:
            raise SystemExit("memory tree summary overwrote Current Context")
        memory_text = memory_path.read_text(encoding="utf-8")
        if "jarvis_projection: memory_tree_snapshot" not in memory_text or "# Memory Tree Snapshot" not in memory_text:
            raise SystemExit("memory tree projection missed ownership or content")
        state_tool({})
        if memory_path.read_text(encoding="utf-8") != memory_text:
            raise SystemExit("state export overwrote the memory tree snapshot")

        note_tool = runtime.registry.get("write_jarvis_note").handler
        for path in (
            "Memory Tree/Current Context",
            "memory tree/current context.md",
            "Memory Tree/Memory Tree Snapshot",
        ):
            for mode in ("append", "create"):
                refused = note_tool({"path": path, "body": "must not append", "mode": mode})
                if refused.ok or refused.metadata.get("reason") != "managed_projection":
                    raise SystemExit(
                        f"generic note writer accepted managed projection {path!r} in {mode}: {refused}"
                    )
        if "must not append" in state_path.read_text(encoding="utf-8"):
            raise SystemExit("generic note refusal still changed Current Context")

        foreign = _other_runtime(root)
        try:
            foreign.registry.get("export_state_snapshot").handler({})
        except FileExistsError:
            pass
        else:
            raise SystemExit("a second database overwrote the owned Current Context projection")

        sentinel = state_path.read_bytes()
        with patch("jarvis_v2.memory.obsidian._replace_text", side_effect=OSError("simulated durable write failure")):
            try:
                state_tool({})
            except OSError:
                pass
            else:
                raise SystemExit("state projection hid its durable write failure")
        if state_path.read_bytes() != sentinel:
            raise SystemExit("failed atomic state publication changed the prior projection")


def test_state_capture_is_fenced_through_publication() -> None:
    with TemporaryDirectory(prefix="jarvis-state-fence-") as temp:
        runtime = make_temp_runtime(Path(temp))
        state_tool = runtime.registry.get("export_state_snapshot").handler
        entered = Event()
        release = Event()
        original_write = runtime.vault.write_current_context_with_evidence

        def blocking_write(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("state publication smoke release timed out")
            return original_write(*args, **kwargs)

        runtime.vault.write_current_context_with_evidence = blocking_write  # type: ignore[method-assign]
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                exporting = pool.submit(state_tool, {})
                if not entered.wait(timeout=5):
                    raise SystemExit("state export did not reach publication")
                mutating = pool.submit(runtime.store.add_task, TaskRecord("newer fenced task"))
                time.sleep(0.1)
                if mutating.done():
                    raise SystemExit("state publication fence allowed a newer mutation to commit early")
                release.set()
                result = exporting.result(timeout=5)
                mutating.result(timeout=5)
        finally:
            runtime.vault.write_current_context_with_evidence = original_write  # type: ignore[method-assign]
            release.set()

        first_text = Path(result.metadata["path"]).read_text(encoding="utf-8")
        if "newer fenced task" in first_text:
            raise SystemExit("fenced snapshot included state committed after its capture")
        refreshed = state_tool({})
        if "newer fenced task" not in Path(refreshed.metadata["path"]).read_text(encoding="utf-8"):
            raise SystemExit("fresh state export missed the later committed mutation")


def main() -> None:
    test_owned_atomic_state_and_memory_projections()
    test_state_capture_is_fenced_through_publication()
    print("state projection integrity smoke passed")


if __name__ == "__main__":
    main()
