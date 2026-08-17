from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import jarvis_v2.memory.obsidian as obsidian_module
from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.agent.types import Plan, PlannedAction, RiskLevel
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import MemoryStore, TaskRecord
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.registry import AutoMutationEffect


class StaticPlanner:
    def __init__(self, args: dict[str, Any]):
        self.args = args

    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Export the canonical open-task mirror.",
            [PlannedAction("export_tasks", dict(self.args), "task export receipt smoke")],
            needs_model=False,
        )


def _runtime(root: Path, args: dict[str, Any]) -> JarvisRuntime:
    runtime = make_temp_runtime(root)
    runtime.planner = StaticPlanner(args)
    return runtime


def _receipt_rows(runtime: JarvisRuntime) -> list[dict[str, Any]]:
    with runtime.store.connect() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM auto_mutation_receipts ORDER BY id")]


def _tool_run_rows(runtime: JarvisRuntime) -> list[dict[str, Any]]:
    with runtime.store.connect() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM tool_runs ORDER BY id")]


def test_contract_canonical_projection_and_replay() -> None:
    with TemporaryDirectory(prefix="jarvis-export-tasks-replay-") as temp:
        runtime = _runtime(Path(temp), {"limit": 200})
        runtime.store.add_task(
            TaskRecord(
                "PRIVATE-TASK-PATH-FIXTURE",
                source="/\x55sers/example/private/task-source",
                due="/private/tmp/task-due",
                priority="/tmp/task-priority",
            )
        )
        for index in range(104):
            runtime.store.add_task(TaskRecord(f"PRIVATE-TASK-{index:03d}"))
        tool = runtime.registry.get("export_tasks")
        contract = tool.auto_mutation_contract
        if (
            tool.risk is not RiskLevel.LOCAL_SAFE
            or contract is None
            or contract.effects != frozenset({AutoMutationEffect.OBSIDIAN_VAULT})
            or tool.argument_contract is None
        ):
            raise SystemExit("export_tasks registry contract drifted")

        calls = 0
        real_sync = runtime.vault.sync_open_tasks_with_evidence

        def counted_sync(store: Any):
            nonlocal calls
            calls += 1
            return real_sync(store)

        runtime.vault.sync_open_tasks_with_evidence = counted_sync  # type: ignore[method-assign]
        first = runtime.handle("export tasks first", request_token="export-tasks-one")
        replay = runtime.handle("export tasks replay", request_token="export-tasks-one")
        runtime.planner = StaticPlanner({"limit": 1})
        repeated = runtime.handle("export tasks fresh", request_token="export-tasks-two")
        if not first.tool_results[0].ok or not repeated.tool_results[0].ok:
            raise SystemExit("export_tasks valid execution failed")
        if replay.tool_results[0].metadata.get("failure_kind") != "auto_mutation_completed_replay":
            raise SystemExit("export_tasks same-token replay did not coalesce")
        if calls != 2:
            raise SystemExit(f"export_tasks replay/fresh publication count drifted: {calls}")
        first_metadata = first.tool_results[0].metadata
        repeated_metadata = repeated.tool_results[0].metadata
        if first_metadata.get("exported_count") != 105 or repeated_metadata.get("exported_count") != 1:
            raise SystemExit("export_tasks requested handoff subset drifted")
        for metadata in (first_metadata, repeated_metadata):
            if (
                metadata.get("mirror_count") != 105
                or metadata.get("writes_memory") is not False
                or len(str(metadata.get("content_sha256") or "")) != 64
                or len(str(metadata.get("source_revision") or "")) != 64
            ):
                raise SystemExit(f"export_tasks publication evidence/metadata drifted: {metadata}")
        note = runtime.vault.root_path / "Tasks" / "Open Tasks.md"
        text = note.read_text(encoding="utf-8")
        if "jarvis_projection: open_tasks" not in text or text.count("- [ ] #") != 105:
            raise SystemExit("export_tasks did not publish the complete canonical projection")
        receipts = _receipt_rows(runtime)
        runs = _tool_run_rows(runtime)
        successful = [row for row in runs if row["ok"] == 1]
        if len(receipts) != 2 or any(row["state"] != "completed" for row in receipts):
            raise SystemExit(f"export_tasks completed receipts drifted: {receipts}")
        if len(successful) != 2 or {row["tool_run_id"] for row in receipts} != {row["id"] for row in successful}:
            raise SystemExit("export_tasks receipts did not link to successful ordinary audits")
        ledger = json.dumps(receipts, ensure_ascii=False, default=str)
        if "PRIVATE-TASK" in ledger:
            raise SystemExit("export_tasks receipt ledger retained task content")
        exposed = json.dumps(first_metadata, ensure_ascii=False, default=str)
        for private in ("PRIVATE-TASK", "/\x55sers/example/private", "/private/tmp", "/tmp/task-priority"):
            if private in exposed:
                raise SystemExit(f"export_tasks metadata exposed private task data: {private}")


def test_no_argument_compatibility_and_typed_rejection() -> None:
    with TemporaryDirectory(prefix="jarvis-export-tasks-empty-") as temp:
        runtime = _runtime(Path(temp), {})
        result = runtime.handle("export tasks", request_token="export-tasks-empty")
        if not result.tool_results[0].ok or result.tool_results[0].metadata.get("mirror_count") != 0:
            raise SystemExit("export_tasks no-argument planner shape stopped working")

    cases = [
        ({"limit": True}, "tool_arguments_invalid"),
        ({"limit": None}, "tool_arguments_invalid"),
        ({"limit": "2"}, "tool_arguments_invalid"),
        ({"limit": 1.5}, "tool_arguments_invalid"),
        ({"limit": []}, "tool_arguments_invalid"),
        ({"limit": {}}, "tool_arguments_invalid"),
        ({"limit": 1, "extra": "rejected"}, "tool_arguments_invalid"),
    ]
    for index, (args, expected_failure) in enumerate(cases):
        with TemporaryDirectory(prefix="jarvis-export-tasks-invalid-") as temp:
            runtime = _runtime(Path(temp), args)
            result = runtime.handle("invalid task export", request_token=f"export-tasks-invalid-{index}")
            if result.tool_results[0].metadata.get("failure_kind") != expected_failure:
                raise SystemExit(f"export_tasks typed rejection drifted for case {index}")
            if _receipt_rows(runtime):
                raise SystemExit(f"export_tasks invalid case {index} created a receipt")
            if (runtime.vault.root_path / "Tasks" / "Open Tasks.md").exists():
                raise SystemExit(f"export_tasks invalid case {index} wrote a mirror")


def test_parent_fsync_failure_fences_different_limit_retry() -> None:
    with TemporaryDirectory(prefix="jarvis-export-tasks-uncertain-") as temp:
        runtime = _runtime(Path(temp), {"limit": 1})
        runtime.store.add_task(TaskRecord("PRIVATE-UNCERTAIN-TASK"))
        real_fsync = obsidian_module.os.fsync
        fsync_calls = 0

        def fail_parent_fsync(fd: int) -> None:
            nonlocal fsync_calls
            fsync_calls += 1
            if fsync_calls == 2:
                raise OSError("representative task mirror directory fsync failure")
            real_fsync(fd)

        with patch("jarvis_v2.memory.obsidian.os.fsync", side_effect=fail_parent_fsync):
            failed = runtime.handle("uncertain task export", request_token="export-tasks-uncertain-one")
        if failed.tool_results[0].metadata.get("failure_kind") != "tool_error":
            raise SystemExit("export_tasks parent fsync failure did not become uncertain")
        note = runtime.vault.root_path / "Tasks" / "Open Tasks.md"
        if fsync_calls != 2 or not note.exists() or "PRIVATE-UNCERTAIN-TASK" not in note.read_text(encoding="utf-8"):
            raise SystemExit("export_tasks fsync fixture did not prove the visible partial effect")
        runtime.planner = StaticPlanner({"limit": 200})
        blocked = runtime.handle("different limit retry", request_token="export-tasks-uncertain-two")
        if blocked.tool_results[0].metadata.get("failure_kind") != "auto_mutation_unresolved_action":
            raise SystemExit("export_tasks changed-limit retry bypassed uncertain projection identity")
        receipts = _receipt_rows(runtime)
        if len(receipts) != 1 or receipts[0]["state"] != "uncertain":
            raise SystemExit("export_tasks uncertain receipt state drifted")
        if any(row["ok"] == 1 for row in _tool_run_rows(runtime)):
            raise SystemExit("export_tasks uncertain failure created an ordinary success audit")


def test_unowned_destination_fails_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-export-tasks-ownership-") as temp:
        runtime = _runtime(Path(temp), {})
        path = runtime.vault.root_path / "Tasks" / "Open Tasks.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Open Tasks\n\n- [ ] personal checklist item\n", encoding="utf-8")
        result = runtime.handle("export tasks collision", request_token="export-tasks-collision")
        if result.tool_results[0].metadata.get("failure_kind") != "tool_error":
            raise SystemExit("export_tasks unowned destination did not fail closed")
        if path.read_text(encoding="utf-8") != "# Open Tasks\n\n- [ ] personal checklist item\n":
            raise SystemExit("export_tasks changed an unowned destination")
        receipts = _receipt_rows(runtime)
        if len(receipts) != 1 or receipts[0]["state"] != "uncertain":
            raise SystemExit("export_tasks ownership failure missed uncertain custody")


def test_store_identity_binding_and_legacy_migration() -> None:
    with TemporaryDirectory(prefix="jarvis-export-tasks-store-owner-") as temp:
        root = Path(temp)
        first = make_temp_runtime(root / "first")
        first.store.add_task(TaskRecord("first store task"))
        path = first.vault.sync_open_tasks(first.store)
        original = path.read_text(encoding="utf-8")
        second_store = MemoryStore(root / "second.sqlite")
        second_store.init()
        second_store.add_task(TaskRecord("second store task"))
        second_vault = ObsidianVault(first.vault.vault_path, first.vault.root)
        try:
            second_vault.sync_open_tasks(second_store)
        except FileExistsError:
            pass
        else:
            raise SystemExit("a second Jarvis database replaced another store's task projection")
        if path.read_text(encoding="utf-8") != original:
            raise SystemExit("store-identity refusal changed the original task projection")

    with TemporaryDirectory(prefix="jarvis-export-tasks-legacy-") as temp:
        runtime = make_temp_runtime(Path(temp))
        path = runtime.vault.root_path / "Tasks" / "Open Tasks.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# Open Tasks\n\nUpdated: 2026-01-01 09:00\n\n- [ ] No open tasks.\n",
            encoding="utf-8",
        )
        runtime.vault.sync_open_tasks(runtime.store)
        migrated = path.read_text(encoding="utf-8")
        if "jarvis_projection: open_tasks" not in migrated or "store_identity:" not in migrated:
            raise SystemExit("exact legacy Jarvis task projection was not bound to its store")


def main() -> None:
    test_contract_canonical_projection_and_replay()
    test_no_argument_compatibility_and_typed_rejection()
    test_parent_fsync_failure_fences_different_limit_retry()
    test_unowned_destination_fails_closed()
    test_store_identity_binding_and_legacy_migration()
    print("Auto mutation export-tasks smoke passed")


if __name__ == "__main__":
    main()
