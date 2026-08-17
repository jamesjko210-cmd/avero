from __future__ import annotations

import json
import multiprocessing as mp
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock
from typing import Any
from unittest.mock import patch

import jarvis_v2.memory.obsidian as obsidian_module
from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.agent.types import Plan, PlannedAction, RiskLevel
from jarvis_v2.memory.store import GoalRecord, MemoryStore
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.registry import AutoMutationEffect


class StaticPlanner:
    def __init__(self, args: dict[str, Any], tool_name: str = "export_goal"):
        self.args = args
        self.tool_name = tool_name

    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Export one goal mirror.",
            [PlannedAction(self.tool_name, dict(self.args), "goal projection receipt smoke")],
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


def _goal_status_process(
    root: str,
    goal_id: int,
    status: str,
    ready: Any,
    start: Any,
    attempted: Any,
    finished: Any,
    lock_observed: Any,
    write_entered: Any,
    release_write: Any,
    result_queue: Any,
    block_publication: bool,
) -> None:
    try:
        runtime = make_temp_runtime(Path(root))
        if block_publication:
            real_write = runtime.vault.write_goal_with_evidence

            def controlled_write(
                goal: Any,
                steps: Any,
                *,
                store_identity: str,
                expected_prior_content_digest: str | None = None,
            ) -> tuple[Path, str, str]:
                write_entered.set()
                if not release_write.wait(timeout=8):
                    raise RuntimeError("cross-process goal publication timed out")
                return real_write(
                    goal,
                    steps,
                    store_identity=store_identity,
                    expected_prior_content_digest=expected_prior_content_digest,
                )

            runtime.vault.write_goal_with_evidence = controlled_write  # type: ignore[method-assign]
        handler = runtime.registry.get("set_goal_status").handler
        ready.set()
        if not start.wait(timeout=8):
            raise RuntimeError("cross-process goal mutation did not start")
        attempted.set()
        if not block_publication:
            probe = sqlite3.connect(Path(root) / "jarvis.sqlite", timeout=0.1)
            try:
                try:
                    probe.execute("BEGIN IMMEDIATE")
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower():
                        raise
                    lock_observed.set()
                else:
                    probe.rollback()
                    raise RuntimeError("newer process did not observe the goal publication fence")
            finally:
                probe.close()
        result = handler({"goal_id": goal_id, "status": status})
        result_queue.put((status, bool(result.ok), ""))
    except BaseException as exc:
        result_queue.put((status, False, type(exc).__name__))
    finally:
        finished.set()


def test_contract_replay_and_numeric_string_compatibility() -> None:
    with TemporaryDirectory(prefix="jarvis-export-goal-replay-") as temp:
        runtime = _runtime(Path(temp), {"goal_id": 1})
        goal_id = runtime.store.create_goal(
            GoalRecord("Private export fixture", "PRIVATE-GOAL-PURPOSE", "this week")
        )
        runtime.store.add_goal_step(goal_id, "PRIVATE-GOAL-STEP")
        runtime.planner = StaticPlanner({"goal_id": goal_id})
        tool = runtime.registry.get("export_goal")
        contract = tool.auto_mutation_contract
        if (
            tool.risk is not RiskLevel.LOCAL_SAFE
            or contract is None
            or contract.effects
            != frozenset(
                {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
            )
            or tool.argument_contract is None
        ):
            raise SystemExit("export_goal registry contract drifted")

        writes = 0
        real_write = runtime.vault.write_goal_with_evidence

        def counted_write(
            goal: Any,
            steps: Any,
            *,
            store_identity: str,
            expected_prior_content_digest: str | None = None,
        ) -> tuple[Path, str, str]:
            nonlocal writes
            writes += 1
            return real_write(
                goal,
                steps,
                store_identity=store_identity,
                expected_prior_content_digest=expected_prior_content_digest,
            )

        runtime.vault.write_goal_with_evidence = counted_write  # type: ignore[method-assign]
        first = runtime.handle("export goal first", request_token="export-goal-one")
        replay = runtime.handle("export goal replay", request_token="export-goal-one")
        runtime.planner = StaticPlanner({"goal_id": str(goal_id)})
        repeated = runtime.handle("export goal fresh rewrite", request_token="export-goal-two")
        if not first.tool_results[0].ok or not repeated.tool_results[0].ok:
            raise SystemExit("export_goal valid execution failed")
        for item in (first.tool_results[0], repeated.tool_results[0]):
            if (
                len(str(item.metadata.get("content_sha256") or "")) != 64
                or len(str(item.metadata.get("source_revision") or "")) != 64
                or item.metadata.get("writes_database") is not True
                or item.metadata.get("writes_memory") is not False
            ):
                raise SystemExit(f"export_goal publication evidence/metadata drifted: {item.metadata}")
        if replay.tool_results[0].metadata.get("failure_kind") != "auto_mutation_completed_replay":
            raise SystemExit("export_goal same-token replay did not coalesce")
        if writes != 1:
            raise SystemExit(f"export_goal idempotent publication count drifted: {writes}")
        receipts = _receipt_rows(runtime)
        runs = _tool_run_rows(runtime)
        successful_runs = [row for row in runs if row["ok"] == 1]
        if len(receipts) != 2 or any(row["state"] != "completed" for row in receipts):
            raise SystemExit(f"export_goal completed receipts drifted: {receipts}")
        if len(successful_runs) != 2 or any(row["approved"] != 0 for row in successful_runs):
            raise SystemExit(f"export_goal ordinary audit linkage drifted: {runs}")
        successful_ids = {row["id"] for row in successful_runs}
        if {row["tool_run_id"] for row in receipts} != successful_ids:
            raise SystemExit("export_goal receipts did not link to the successful ordinary audits")
        note = runtime.vault.write_goal(
            runtime.store.get_goal(goal_id),
            runtime.store.list_goal_steps(goal_id),
            store_identity=runtime.store.get_store_identity(),
        )
        text = note.read_text(encoding="utf-8")
        if "PRIVATE-GOAL-PURPOSE" not in text or "PRIVATE-GOAL-STEP" not in text:
            raise SystemExit("export_goal mirror missed the captured goal snapshot")
        private_ledger = json.dumps(receipts, ensure_ascii=False, default=str)
        if "PRIVATE-GOAL" in private_ledger or "Private export fixture" in private_ledger:
            raise SystemExit("export_goal receipt ledger retained goal content")


def test_typed_and_semantic_preflight() -> None:
    cases = [
        ({}, "tool_arguments_invalid"),
        ({"goal_id": True}, "tool_arguments_invalid"),
        ({"goal_id": 1.5}, "tool_arguments_invalid"),
        ({"goal_id": []}, "tool_arguments_invalid"),
        ({"goal_id": "1", "extra": "rejected"}, "tool_arguments_invalid"),
        ({"goal_id": "0"}, "auto_mutation_semantic_preflight_rejected"),
        ({"goal_id": "-1"}, "auto_mutation_semantic_preflight_rejected"),
        ({"goal_id": "999"}, "auto_mutation_semantic_preflight_rejected"),
    ]
    for index, (args, expected_failure) in enumerate(cases):
        with TemporaryDirectory(prefix="jarvis-export-goal-preflight-") as temp:
            runtime = _runtime(Path(temp), args)
            result = runtime.handle("invalid export", request_token=f"export-goal-invalid-{index}")
            item = result.tool_results[0]
            if item.metadata.get("failure_kind") != expected_failure:
                raise SystemExit(f"export_goal preflight drifted for case {index}: {item}")
            if _receipt_rows(runtime):
                raise SystemExit(f"export_goal invalid case {index} crossed the receipt boundary")
            for run in _tool_run_rows(runtime):
                metadata = json.loads(run["metadata"] or "{}")
                if run["ok"] != 0 or metadata.get("executed_handler") is not False:
                    raise SystemExit(f"export_goal invalid case {index} looked executed: {run}")
            if list((runtime.vault.root_path / "Projects").glob("*.md")):
                raise SystemExit(f"export_goal invalid case {index} wrote a note")


def test_post_publication_failure_fences_numeric_string_retry() -> None:
    with TemporaryDirectory(prefix="jarvis-export-goal-uncertain-") as temp:
        runtime = _runtime(Path(temp), {"goal_id": 1})
        goal_id = runtime.store.create_goal(GoalRecord("Uncertain export", "PRIVATE-UNCERTAIN-PURPOSE"))
        runtime.planner = StaticPlanner({"goal_id": goal_id})
        real_fsync = obsidian_module.os.fsync
        fsync_calls = 0

        def fail_parent_fsync(fd: int) -> None:
            nonlocal fsync_calls
            fsync_calls += 1
            if fsync_calls == 2:
                raise OSError("representative parent directory fsync failure")
            real_fsync(fd)

        with patch("jarvis_v2.memory.obsidian.os.fsync", side_effect=fail_parent_fsync):
            failed = runtime.handle("uncertain goal export", request_token="export-goal-uncertain-one")
        if failed.tool_results[0].metadata.get("failure_kind") != "tool_error":
            raise SystemExit("export_goal post-publication failure did not become uncertain")
        note_paths = list((runtime.vault.root_path / "Projects").glob("*.md"))
        if fsync_calls != 2 or len(note_paths) != 1 or "PRIVATE-UNCERTAIN-PURPOSE" not in note_paths[0].read_text(encoding="utf-8"):
            raise SystemExit("export_goal post-publication fixture did not prove the partial effect")
        runtime.planner = StaticPlanner({"goal_id": str(goal_id)})
        blocked = runtime.handle("numeric string retry", request_token="export-goal-uncertain-two")
        if blocked.tool_results[0].metadata.get("failure_kind") != "auto_mutation_unresolved_action":
            raise SystemExit("export_goal numeric-string retry bypassed uncertain goal identity")
        if len(_receipt_rows(runtime)) != 1 or _receipt_rows(runtime)[0]["state"] != "uncertain":
            raise SystemExit("export_goal uncertain receipt state drifted")
        if any(row["ok"] == 1 for row in _tool_run_rows(runtime)):
            raise SystemExit("export_goal uncertain handler failure created an ordinary success audit")


def test_goal_snapshot_orders_concurrent_status_publication() -> None:
    with TemporaryDirectory(prefix="jarvis-export-goal-ordering-") as temp:
        runtime = make_temp_runtime(Path(temp))
        goal_id = runtime.store.create_goal(GoalRecord("Ordered export", "ordering proof"))
        export_handler = runtime.registry.get("export_goal").handler
        status_handler = runtime.registry.get("set_goal_status").handler
        real_write = runtime.vault.write_goal_with_evidence
        active_write_entered = Event()
        release_active_write = Event()
        first_lock = Lock()
        first_active = True

        def controlled_write(
            goal: Any,
            steps: Any,
            *,
            store_identity: str,
            expected_prior_content_digest: str | None = None,
        ) -> tuple[Path, str, str]:
            nonlocal first_active
            should_block = False
            with first_lock:
                if first_active and str(goal["status"]) == "active":
                    first_active = False
                    should_block = True
            if should_block:
                active_write_entered.set()
                if not release_active_write.wait(timeout=3):
                    raise RuntimeError("goal export ordering fixture timed out")
            return real_write(
                goal,
                steps,
                store_identity=store_identity,
                expected_prior_content_digest=expected_prior_content_digest,
            )

        runtime.vault.write_goal_with_evidence = controlled_write  # type: ignore[method-assign]
        with ThreadPoolExecutor(max_workers=2) as pool:
            export_future = pool.submit(export_handler, {"goal_id": goal_id})
            if not active_write_entered.wait(timeout=2):
                raise SystemExit("export_goal did not enter its fenced publication")
            status_future = pool.submit(status_handler, {"goal_id": goal_id, "status": "done"})
            time.sleep(0.05)
            if status_future.done():
                raise SystemExit("goal status mutation crossed the export snapshot write fence")
            release_active_write.set()
            exported = export_future.result(timeout=3)
            updated = status_future.result(timeout=3)
        if not exported.ok or not updated.ok:
            raise SystemExit("concurrent goal export/status fixture failed")
        note_paths = list((runtime.vault.root_path / "Projects").glob("*.md"))
        if len(note_paths) != 1 or 'status: "done"' not in note_paths[0].read_text(encoding="utf-8"):
            raise SystemExit("newer goal status publication did not win after fenced export")


def test_goal_mutation_snapshot_orders_two_status_commits() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-status-ordering-") as temp:
        runtime = make_temp_runtime(Path(temp))
        newer_runtime = make_temp_runtime(Path(temp))
        goal_id = runtime.store.create_goal(GoalRecord("Ordered status mutations", "ordering proof"))
        status_handler = runtime.registry.get("set_goal_status").handler
        newer_status_handler = newer_runtime.registry.get("set_goal_status").handler
        real_write = runtime.vault.write_goal_with_evidence
        paused_write_entered = Event()
        release_paused_write = Event()
        first_lock = Lock()
        first_paused = True

        def controlled_write(
            goal: Any,
            steps: Any,
            *,
            store_identity: str,
            expected_prior_content_digest: str | None = None,
        ) -> tuple[Path, str, str]:
            nonlocal first_paused
            should_block = False
            with first_lock:
                if first_paused and str(goal["status"]) == "paused":
                    first_paused = False
                    should_block = True
            if should_block:
                paused_write_entered.set()
                if not release_paused_write.wait(timeout=3):
                    raise RuntimeError("goal status ordering fixture timed out")
            return real_write(
                goal,
                steps,
                store_identity=store_identity,
                expected_prior_content_digest=expected_prior_content_digest,
            )

        runtime.vault.write_goal_with_evidence = controlled_write  # type: ignore[method-assign]
        with ThreadPoolExecutor(max_workers=2) as pool:
            paused_future = pool.submit(status_handler, {"goal_id": goal_id, "status": "paused"})
            if not paused_write_entered.wait(timeout=2):
                raise SystemExit("older goal status did not enter publication")
            done_future = pool.submit(newer_status_handler, {"goal_id": goal_id, "status": "done"})
            time.sleep(0.05)
            if done_future.done():
                raise SystemExit("newer status commit crossed the older publication fence")
            release_paused_write.set()
            paused = paused_future.result(timeout=3)
            done = done_future.result(timeout=3)

        goal = runtime.store.get_goal(goal_id)
        note_paths = list((runtime.vault.root_path / "Projects").glob("*.md"))
        if not paused.ok or not done.ok or goal is None or goal["status"] != "done":
            raise SystemExit("concurrent goal status mutations did not complete in commit order")
        if len(note_paths) != 1 or 'status: "done"' not in note_paths[0].read_text(encoding="utf-8"):
            raise SystemExit("older status snapshot overwrote the newest committed goal state")


def test_goal_mutation_snapshot_keeps_every_committed_step() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-step-ordering-") as temp:
        runtime = make_temp_runtime(Path(temp))
        newer_runtime = make_temp_runtime(Path(temp))
        goal_id = runtime.store.create_goal(GoalRecord("Ordered step mutations", "ordering proof"))
        add_step_handler = runtime.registry.get("add_goal_step").handler
        newer_add_step_handler = newer_runtime.registry.get("add_goal_step").handler
        real_write = runtime.vault.write_goal_with_evidence
        first_write_entered = Event()
        release_first_write = Event()
        first_lock = Lock()
        first_single_step = True

        def controlled_write(
            goal: Any,
            steps: Any,
            *,
            store_identity: str,
            expected_prior_content_digest: str | None = None,
        ) -> tuple[Path, str, str]:
            nonlocal first_single_step
            should_block = False
            with first_lock:
                if first_single_step and len(steps) == 1:
                    first_single_step = False
                    should_block = True
            if should_block:
                first_write_entered.set()
                if not release_first_write.wait(timeout=3):
                    raise RuntimeError("goal step ordering fixture timed out")
            return real_write(
                goal,
                steps,
                store_identity=store_identity,
                expected_prior_content_digest=expected_prior_content_digest,
            )

        runtime.vault.write_goal_with_evidence = controlled_write  # type: ignore[method-assign]
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_future = pool.submit(add_step_handler, {"goal_id": goal_id, "body": "first ordered step"})
            if not first_write_entered.wait(timeout=2):
                raise SystemExit("first goal step did not enter publication")
            second_future = pool.submit(
                newer_add_step_handler,
                {"goal_id": goal_id, "body": "second ordered step"},
            )
            time.sleep(0.05)
            if second_future.done():
                raise SystemExit("second goal step commit crossed the first publication fence")
            release_first_write.set()
            first = first_future.result(timeout=3)
            second = second_future.result(timeout=3)

        steps = runtime.store.list_goal_steps(goal_id)
        note_paths = list((runtime.vault.root_path / "Projects").glob("*.md"))
        if not first.ok or not second.ok or len(steps) != 2 or len(note_paths) != 1:
            raise SystemExit("concurrent goal step mutations did not complete cleanly")
        note = note_paths[0].read_text(encoding="utf-8")
        expected = [f"#{step['id']} {step['body']}" for step in steps]
        if any(fragment not in note or note.count(fragment) != 1 for fragment in expected):
            raise SystemExit("older step snapshot dropped a newer committed goal step")


def test_create_and_complete_publishers_share_the_goal_fence() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-create-ordering-") as temp:
        creator_runtime = make_temp_runtime(Path(temp))
        updater_runtime = make_temp_runtime(Path(temp))
        create_handler = creator_runtime.registry.get("create_goal").handler
        status_handler = updater_runtime.registry.get("set_goal_status").handler
        real_write = creator_runtime.vault.write_goal_with_evidence
        create_write_entered = Event()
        release_create_write = Event()

        def controlled_create_write(
            goal: Any,
            steps: Any,
            *,
            store_identity: str,
            expected_prior_content_digest: str | None = None,
        ) -> tuple[Path, str, str]:
            create_write_entered.set()
            if not release_create_write.wait(timeout=3):
                raise RuntimeError("goal create ordering fixture timed out")
            return real_write(
                goal,
                steps,
                store_identity=store_identity,
                expected_prior_content_digest=expected_prior_content_digest,
            )

        creator_runtime.vault.write_goal_with_evidence = controlled_create_write  # type: ignore[method-assign]
        with ThreadPoolExecutor(max_workers=2) as pool:
            create_future = pool.submit(
                create_handler,
                {"title": "Ordered create publisher", "purpose": "ordering proof"},
            )
            if not create_write_entered.wait(timeout=2):
                raise SystemExit("goal create did not enter publication")
            rows = updater_runtime.store.list_goals(limit=5)
            if len(rows) != 1:
                raise SystemExit("committed goal was not visible during its publication fence")
            goal_id = int(rows[0]["id"])
            status_future = pool.submit(status_handler, {"goal_id": goal_id, "status": "done"})
            time.sleep(0.05)
            if status_future.done():
                raise SystemExit("goal status crossed the create publication fence")
            release_create_write.set()
            created = create_future.result(timeout=3)
            updated = status_future.result(timeout=3)

        note_paths = list((creator_runtime.vault.root_path / "Projects").glob("*.md"))
        if not created.ok or not updated.ok or len(note_paths) != 1:
            raise SystemExit("create/status ordering fixture did not finish cleanly")
        if 'status: "done"' not in note_paths[0].read_text(encoding="utf-8"):
            raise SystemExit("create publisher overwrote a newer committed goal status")

    with TemporaryDirectory(prefix="jarvis-goal-complete-ordering-") as temp:
        older_runtime = make_temp_runtime(Path(temp))
        completer_runtime = make_temp_runtime(Path(temp))
        goal_id = older_runtime.store.create_goal(GoalRecord("Ordered completion publisher", "ordering proof"))
        step_id = older_runtime.store.add_goal_step(goal_id, "complete after older snapshot")
        status_handler = older_runtime.registry.get("set_goal_status").handler
        complete_handler = completer_runtime.registry.get("complete_goal_step").handler
        real_write = older_runtime.vault.write_goal_with_evidence
        open_write_entered = Event()
        release_open_write = Event()

        def controlled_open_write(
            goal: Any,
            steps: Any,
            *,
            store_identity: str,
            expected_prior_content_digest: str | None = None,
        ) -> tuple[Path, str, str]:
            if steps and str(steps[0]["status"]) == "open":
                open_write_entered.set()
                if not release_open_write.wait(timeout=3):
                    raise RuntimeError("goal completion ordering fixture timed out")
            return real_write(
                goal,
                steps,
                store_identity=store_identity,
                expected_prior_content_digest=expected_prior_content_digest,
            )

        older_runtime.vault.write_goal_with_evidence = controlled_open_write  # type: ignore[method-assign]
        with ThreadPoolExecutor(max_workers=2) as pool:
            older_future = pool.submit(status_handler, {"goal_id": goal_id, "status": "paused"})
            if not open_write_entered.wait(timeout=2):
                raise SystemExit("open-step snapshot did not enter publication")
            complete_future = pool.submit(complete_handler, {"step_id": step_id})
            time.sleep(0.05)
            if complete_future.done():
                raise SystemExit("goal completion crossed the older publication fence")
            release_open_write.set()
            older = older_future.result(timeout=3)
            completed = complete_future.result(timeout=3)

        note_paths = list((older_runtime.vault.root_path / "Projects").glob("*.md"))
        if not older.ok or not completed.ok or len(note_paths) != 1:
            raise SystemExit("status/completion ordering fixture did not finish cleanly")
        note = note_paths[0].read_text(encoding="utf-8")
        if f"- [x] #{step_id} complete after older snapshot" not in note:
            raise SystemExit("older goal snapshot overwrote a newer step completion")


def test_cross_process_goal_status_publication_has_one_lock_order() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-process-ordering-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        goal_id = runtime.store.create_goal(GoalRecord("Cross process ordering", "ordering proof"))
        ctx = mp.get_context("spawn")
        older_ready = ctx.Event()
        newer_ready = ctx.Event()
        older_start = ctx.Event()
        newer_start = ctx.Event()
        older_attempted = ctx.Event()
        newer_attempted = ctx.Event()
        older_finished = ctx.Event()
        newer_finished = ctx.Event()
        older_lock_observed = ctx.Event()
        newer_lock_observed = ctx.Event()
        write_entered = ctx.Event()
        release_write = ctx.Event()
        newer_write_entered = ctx.Event()
        newer_release_write = ctx.Event()
        result_queue = ctx.Queue()
        older = ctx.Process(
            target=_goal_status_process,
            args=(
                str(root), goal_id, "paused", older_ready, older_start, older_attempted,
                older_finished, older_lock_observed, write_entered, release_write, result_queue, True,
            ),
        )
        newer = ctx.Process(
            target=_goal_status_process,
            args=(
                str(root), goal_id, "done", newer_ready, newer_start, newer_attempted,
                newer_finished, newer_lock_observed, newer_write_entered, newer_release_write,
                result_queue, False,
            ),
        )
        older.start()
        newer.start()
        try:
            if not older_ready.wait(timeout=8) or not newer_ready.wait(timeout=8):
                raise SystemExit("cross-process goal runtimes did not initialize")
            older_start.set()
            if not older_attempted.wait(timeout=2) or not write_entered.wait(timeout=5):
                raise SystemExit("older process did not reach fenced goal publication")
            newer_start.set()
            if not newer_attempted.wait(timeout=2):
                raise SystemExit("newer process did not attempt its goal mutation")
            if not newer_lock_observed.wait(timeout=2):
                raise SystemExit("newer process did not observe the SQLite publication fence")
            release_write.set()
            older.join(10)
            newer.join(10)
        finally:
            release_write.set()
            if older.is_alive():
                older.terminate()
            if newer.is_alive():
                newer.terminate()
            older.join(3)
            newer.join(3)

        results = [result_queue.get(timeout=2), result_queue.get(timeout=2)]
        result_queue.close()
        result_queue.join_thread()
        if older.exitcode != 0 or newer.exitcode != 0 or any(not item[1] for item in results):
            raise SystemExit(f"cross-process goal mutation failed: {[(item[0], item[1], item[2]) for item in results]}")
        goal = runtime.store.get_goal(goal_id)
        note_paths = list((runtime.vault.root_path / "Projects").glob("*.md"))
        if goal is None or goal["status"] != "done" or len(note_paths) != 1:
            raise SystemExit("cross-process goal state did not converge")
        if 'status: "done"' not in note_paths[0].read_text(encoding="utf-8"):
            raise SystemExit("cross-process stale publisher overwrote the newest goal state")


def test_goal_snapshot_binds_goal_and_steps_to_one_database_revision() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-revision-snapshot-") as temp:
        runtime = make_temp_runtime(Path(temp))
        goal_id = runtime.store.create_goal(GoalRecord("One revision snapshot", "ordering proof"))
        step_id = runtime.store.add_goal_step(goal_id, "revision-bound step")
        snapshot_store = MemoryStore(runtime.store.db_path)
        mutator_store = MemoryStore(runtime.store.db_path)
        goal_read = Event()
        release_goal_read = Event()
        lock_observed = Event()
        captured: list[tuple[str, str]] = []
        real_connect = snapshot_store.connect

        class PausingConnection:
            def __init__(self, connection: sqlite3.Connection):
                self.connection = connection

            def __enter__(self):
                self.connection.__enter__()
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return self.connection.__exit__(exc_type, exc_value, traceback)

            def execute(self, sql: str, parameters: Any = ()):
                result = self.connection.execute(sql, parameters)
                if "SELECT * FROM goals WHERE id" in sql:
                    goal_read.set()
                    if not release_goal_read.wait(timeout=3):
                        raise RuntimeError("goal/step query barrier timed out")
                return result

        snapshot_store.connect = lambda: PausingConnection(real_connect())  # type: ignore[method-assign]

        def hold_snapshot() -> None:
            with snapshot_store.goal_mirror_snapshot(goal_id) as (goal, steps):
                if goal is None or len(steps) != 1:
                    raise AssertionError("goal snapshot fixture was incomplete")
                captured.append((str(goal["status"]), str(steps[0]["status"])))

        def prove_write_fence() -> None:
            probe = sqlite3.connect(mutator_store.db_path, timeout=0.1)
            try:
                try:
                    probe.execute("BEGIN IMMEDIATE")
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower():
                        raise
                    lock_observed.set()
                else:
                    probe.rollback()
                    raise AssertionError("goal/step snapshot did not hold a write fence")
            finally:
                probe.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            snapshot_future = pool.submit(hold_snapshot)
            if not goal_read.wait(timeout=2):
                raise SystemExit("goal revision snapshot did not reach the inter-query barrier")
            probe_future = pool.submit(prove_write_fence)
            probe_future.result(timeout=2)
            if not lock_observed.is_set():
                raise SystemExit("goal revision snapshot did not prove its SQLite write fence")
            release_goal_read.set()
            snapshot_future.result(timeout=3)

        if captured != [("active", "open")]:
            raise SystemExit(f"goal mirror snapshot mixed database revisions: {captured}")
        mutator_store.set_goal_status(goal_id, "done")
        mutator_store.complete_goal_step(step_id)
        final_goal = runtime.store.get_goal(goal_id)
        final_steps = runtime.store.list_goal_steps(goal_id)
        if final_goal is None or final_goal["status"] != "done" or final_steps[0]["status"] != "done":
            raise SystemExit("post-snapshot goal mutation did not commit")


def test_export_evidence_retries_after_concurrent_revision_advance() -> None:
    with TemporaryDirectory(prefix="jarvis-export-goal-evidence-race-") as temp:
        runtime = make_temp_runtime(Path(temp))
        goal_id = runtime.store.create_goal(
            GoalRecord("Evidence snapshot race", "bind evidence to one revision")
        )
        real_snapshot = runtime.store.get_completed_goal_projection_snapshot
        advanced = [False]

        def advance_before_snapshot(target: Any):
            if not advanced[0]:
                advanced[0] = True
                if not runtime.store.set_goal_status(goal_id, "done"):
                    raise AssertionError("concurrent goal revision did not advance")
            return real_snapshot(target)

        runtime.store.get_completed_goal_projection_snapshot = advance_before_snapshot  # type: ignore[method-assign]
        result = runtime.registry.get("export_goal").handler({"goal_id": goal_id})
        job = runtime.store.get_goal_projection_job(goal_id)
        notes = list((runtime.vault.root_path / "Projects").glob("*.md"))
        if (
            not result.ok
            or not advanced[0]
            or job is None
            or job["state"] != "completed"
            or result.metadata.get("source_revision") != job["source_digest"]
            or len(notes) != 1
            or 'status: "done"' not in notes[0].read_text(encoding="utf-8")
        ):
            raise SystemExit("export returned evidence from a different goal revision")


def test_goal_mutation_publication_failure_preserves_uncertain_custody() -> None:
    cases = ("create_goal", "add_goal_step", "complete_goal_step", "set_goal_status")
    for tool_name in cases:
        with TemporaryDirectory(prefix=f"jarvis-goal-publication-failure-{tool_name}-") as temp:
            runtime = make_temp_runtime(Path(temp))
            private_marker = f"PRIVATE-{tool_name}-GOAL-CONTENT"
            if tool_name == "create_goal":
                args = {"title": private_marker, "purpose": "publication failure proof"}
            else:
                goal_id = runtime.store.create_goal(GoalRecord("Failure fixture", private_marker))
                if tool_name == "add_goal_step":
                    args = {"goal_id": goal_id, "body": private_marker}
                elif tool_name == "complete_goal_step":
                    step_id = runtime.store.add_goal_step(goal_id, private_marker)
                    args = {"step_id": step_id}
                else:
                    args = {"goal_id": goal_id, "status": "done"}
            runtime.planner = StaticPlanner(args, tool_name)
            publication_calls = 0

            def fail_publication(
                _goal: Any,
                _steps: Any,
                *,
                store_identity: str,
                expected_prior_content_digest: str | None = None,
            ) -> tuple[Path, str, str]:
                nonlocal publication_calls
                publication_calls += 1
                raise OSError("synthetic goal publication failure")

            runtime.vault.write_goal_with_evidence = fail_publication  # type: ignore[method-assign]
            token = f"goal-publication-failure-{tool_name}"
            failed = runtime.handle(f"exercise {tool_name} publication failure", request_token=token)
            replay = runtime.handle(f"replay {tool_name} publication failure", request_token=token)
            cross_token = token + "-fresh-request"
            cross_replay = runtime.handle(
                f"cross-request replay {tool_name} publication failure",
                request_token=cross_token,
            )
            receipts = _receipt_rows(runtime)
            runs = _tool_run_rows(runtime)
            if publication_calls != 1:
                raise SystemExit(f"{tool_name} publication failure reran unexpectedly")
            if failed.tool_results[0].metadata.get("failure_kind") != "tool_error":
                raise SystemExit(f"{tool_name} publication failure did not preserve tool-error custody")
            if replay.tool_results[0].metadata.get("failure_kind") != "auto_mutation_outcome_uncertain":
                raise SystemExit(f"{tool_name} uncertain replay was not blocked")
            if cross_replay.tool_results[0].metadata.get("failure_kind") != "auto_mutation_unresolved_action":
                raise SystemExit(f"{tool_name} cross-request uncertain replay was not blocked")
            if len(receipts) != 1 or receipts[0]["state"] != "uncertain" or receipts[0]["result"] != "unknown":
                raise SystemExit(f"{tool_name} publication failure receipt drifted")
            if any(row["ok"] == 1 for row in runs):
                raise SystemExit(f"{tool_name} publication failure produced a success audit")
            public_failure_surfaces = json.dumps(
                {
                    "failed": {
                        "output": failed.tool_results[0].output,
                        "metadata": failed.tool_results[0].metadata,
                    },
                    "replay": {
                        "output": replay.tool_results[0].output,
                        "metadata": replay.tool_results[0].metadata,
                    },
                    "cross_replay": {
                        "output": cross_replay.tool_results[0].output,
                        "metadata": cross_replay.tool_results[0].metadata,
                    },
                    "receipts": receipts,
                    "runs": [
                        {"output": row.get("output"), "metadata": row.get("metadata")}
                        for row in runs
                    ],
                },
                default=str,
            )
            if private_marker in public_failure_surfaces:
                raise SystemExit(f"{tool_name} failure surfaces leaked goal content")

            if tool_name == "create_goal":
                rows = runtime.store.list_goals(limit=10)
                committed = len(rows) == 1 and rows[0]["title"] == private_marker
            elif tool_name == "add_goal_step":
                committed = len(runtime.store.list_goal_steps(goal_id)) == 1
            elif tool_name == "complete_goal_step":
                committed = runtime.store.list_goal_steps(goal_id)[0]["status"] == "done"
            else:
                goal = runtime.store.get_goal(goal_id)
                committed = goal is not None and goal["status"] == "done"
            if not committed:
                raise SystemExit(f"{tool_name} failure did not occur after its database commit")


def main() -> None:
    test_contract_replay_and_numeric_string_compatibility()
    test_typed_and_semantic_preflight()
    test_post_publication_failure_fences_numeric_string_retry()
    test_goal_snapshot_orders_concurrent_status_publication()
    test_goal_mutation_snapshot_orders_two_status_commits()
    test_goal_mutation_snapshot_keeps_every_committed_step()
    test_create_and_complete_publishers_share_the_goal_fence()
    test_cross_process_goal_status_publication_has_one_lock_order()
    test_goal_snapshot_binds_goal_and_steps_to_one_database_revision()
    test_export_evidence_retries_after_concurrent_revision_advance()
    test_goal_mutation_publication_failure_preserves_uncertain_custody()
    print("Auto mutation export-goal smoke passed")


if __name__ == "__main__":
    main()
