from __future__ import annotations

import os
import signal
import time
from multiprocessing import get_all_start_methods, get_context
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Barrier, Event, Thread, current_thread
from typing import Any
from unittest import mock

from jarvis_v2.automations.jobs import build_daily_plan
from jarvis_v2.memory import obsidian as obsidian_module
from jarvis_v2.memory.store import GoalRecord, MemoryStore, TaskRecord
from jarvis_v2.scripts.test_runtime import make_temp_runtime


TARGET_DATE = "2042-06-15"
OLD_STEP = "DAILY_PLAN_SNAPSHOT_OLD_STEP"
NEW_STEP = "DAILY_PLAN_SNAPSHOT_NEW_STEP"
SOURCE_FAMILIES = (
    "task",
    "goal",
    "goal_step",
    "pending_approval",
    "scheduled_job",
)


def _process_add_task(db_path: str, pipe) -> None:
    store = MemoryStore(Path(db_path))
    original_connect = store.connect
    signaled = False

    def instrumented_connect():
        nonlocal signaled
        conn = original_connect()

        def trace(statement: str) -> None:
            nonlocal signaled
            if not signaled and "INSERT INTO tasks" in statement:
                signaled = True
                pipe.send("attempting")

        conn.set_trace_callback(trace)
        return conn

    store.connect = instrumented_connect  # type: ignore[method-assign]
    try:
        pipe.send(("done", store.add_task(TaskRecord("PROCESS_DAILY_PLAN_WRITER"))))
    except BaseException as exc:
        pipe.send(("error", type(exc).__name__))
    finally:
        pipe.close()


def _fork_try_inherited_appender(append, pipe) -> None:
    pipe.send("ready")
    pipe.recv()
    try:
        append("FORKED_HEADING", "FORKED_BODY")
    except BaseException as exc:
        pipe.send(("refused", type(exc).__name__, str(exc)))
    else:
        pipe.send(("published",))
    finally:
        pipe.close()


def _process_crash_with_forked_inode_lock(
    vault_root: str,
    target_date: str,
    pipe,
) -> None:
    root = Path(vault_root)
    path = root / "Daily" / f"{target_date}.md"

    def fork_while_inode_locked(_fd: int, _content: str) -> None:
        pid = os.fork()
        if pid == 0:
            pipe.send(("grandchild_ready", os.getpid()))
            if pipe.poll(3):
                pipe.recv()
            pipe.close()
            os._exit(0)
        pipe.close()
        os._exit(0)

    obsidian_module._write_all = fork_while_inode_locked
    obsidian_module._append_text_under_inode_lock(
        root,
        path,
        "holder must exit before this content is written",
        initial_content=f"# {target_date}\n\n",
    )


def _join_threads(threads: tuple[Thread, ...], errors: Queue, label: str) -> None:
    for thread in threads:
        thread.join(timeout=10)
    alive = [thread.name for thread in threads if thread.is_alive()]
    if alive:
        raise SystemExit(f"{label} threads did not finish: {alive}")
    if not errors.empty():
        raise SystemExit(f"{label} worker failed: {errors.get()!r}")


def _daily_note_text(runtime, target_date: str = TARGET_DATE) -> str:
    path = runtime.vault.root_path / "Daily" / f"{target_date}.md"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def test_read_snapshot_stays_coherent_during_goal_step_mutation() -> None:
    with TemporaryDirectory(prefix="jarvis-daily-plan-read-snapshot-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.add_task(TaskRecord("SNAPSHOT_TRANSACTION_ANCHOR"))
        goal_id = runtime.store.create_goal(GoalRecord("SNAPSHOT_RACE_GOAL"))
        old_step_id = runtime.store.add_goal_step(goal_id, OLD_STEP)
        if old_step_id is None:
            raise SystemExit("daily-plan snapshot fixture could not create its old step")

        step_query_reached = Event()
        mutation_finished = Event()
        synchronization_timed_out = Event()
        snapshots: Queue = Queue()
        errors: Queue = Queue()
        reader_name = "daily-plan-snapshot-reader"
        original_connect = runtime.store.connect
        step_query_count = [0]

        def instrumented_connect():
            conn = original_connect()
            if current_thread().name == reader_name:
                def trace(statement: str) -> None:
                    if "WITH ranked_steps AS" not in statement:
                        return
                    step_query_count[0] += 1
                    step_query_reached.set()
                    if not mutation_finished.wait(timeout=5):
                        synchronization_timed_out.set()

                conn.set_trace_callback(trace)
            return conn

        def read_snapshot() -> None:
            try:
                snapshots.put(runtime.store.read_daily_plan_snapshot())
            except BaseException as exc:
                errors.put(exc)

        def replace_first_open_step() -> None:
            try:
                if not step_query_reached.wait(timeout=5):
                    raise RuntimeError("reader never reached the goal-step query")
                completed = runtime.store.complete_goal_step(old_step_id)
                new_step_id = runtime.store.add_goal_step(goal_id, NEW_STEP)
                if completed is None or completed["status"] != "done" or new_step_id is None:
                    raise RuntimeError("concurrent goal-step mutation did not commit")
            except BaseException as exc:
                errors.put(exc)
            finally:
                mutation_finished.set()

        with mock.patch.object(
            runtime.store,
            "connect",
            side_effect=instrumented_connect,
        ):
            reader = Thread(target=read_snapshot, name=reader_name)
            mutator = Thread(
                target=replace_first_open_step,
                name="daily-plan-snapshot-mutator",
            )
            reader.start()
            mutator.start()
            _join_threads((reader, mutator), errors, "daily-plan read snapshot race")

        if snapshots.empty() or synchronization_timed_out.is_set():
            raise SystemExit("daily-plan read snapshot race missed its forced boundary")
        snapshot = snapshots.get_nowait()
        after = runtime.store.read_daily_plan_snapshot()
        snapshot_steps = [
            str(row["body"])
            for row in snapshot.open_steps
            if int(row["goal_id"]) == goal_id
        ]
        after_steps = [
            str(row["body"])
            for row in after.open_steps
            if int(row["goal_id"]) == goal_id
        ]
        if (
            step_query_count[0] != 1
            or snapshot_steps != [OLD_STEP]
            or after_steps != [NEW_STEP]
        ):
            raise SystemExit(
                "daily-plan read mixed goal and step generations: "
                f"query_count={step_query_count[0]} before={snapshot_steps!r} "
                f"after={after_steps!r}"
            )


def test_bounded_all_family_render_uses_one_snapshot_and_redacts_paths() -> None:
    with TemporaryDirectory(prefix="jarvis-daily-plan-render-snapshot-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        private_path = str(root / "private" / "daily-plan-source.txt")

        for index in range(13):
            body = (
                f"TASK_FAMILY_CANARY at {private_path}; retained "
                + ("x" * 2300)
                + " TASK_CLIPPED_TAIL"
                if index == 0
                else f"BOUNDED_TASK_{index:02d}"
            )
            if index == 1:
                body = (
                    "PATH_VARIANTS /home/alice/plan.txt /root/private.txt "
                    "/Volumes/Work/plan.txt ~/plan.txt C:\\Users\\alice\\plan.txt"
                )
            if index == 12:
                body = "TASK_OVERFLOW_CANARY"
            runtime.store.add_task(TaskRecord(body))

        goal_ids: list[int] = []
        for index in range(9):
            title = (
                f"GOAL_FAMILY_CANARY at {private_path}; retained"
                if index == 0
                else f"BOUNDED_GOAL_{index:02d}"
            )
            if index == 8:
                title = "GOAL_OVERFLOW_CANARY"
            goal_id = runtime.store.create_goal(GoalRecord(title))
            first_step = (
                f"FIRST_OPEN_STEP_CANARY at {private_path}; retained"
                if index == 0
                else f"BOUNDED_FIRST_STEP_{index:02d}"
            )
            if runtime.store.add_goal_step(goal_id, first_step) is None:
                raise SystemExit("bounded daily-plan fixture lost a first goal step")
            goal_ids.append(goal_id)
        if runtime.store.add_goal_step(
            goal_ids[0],
            "SECOND_OPEN_STEP_MUST_NOT_RENDER",
        ) is None:
            raise SystemExit("bounded daily-plan fixture lost its second goal step")
        with runtime.store.connect() as conn:
            for index, goal_id in enumerate(goal_ids):
                conn.execute(
                    "UPDATE goals SET updated_at = ? WHERE id = ?",
                    (f"2042-01-{9 - index:02d}T00:00:00Z", goal_id),
                )

        runtime.store.add_pending_approval(
            "overflow-session",
            "APPROVAL_OVERFLOW_CANARY",
            "overflow_tool",
            "bounded fixture",
        )
        for index in range(8):
            user_input = (
                f"APPROVAL_FAMILY_CANARY at {private_path}; retained"
                if index == 7
                else f"BOUNDED_APPROVAL_{index:02d}"
            )
            runtime.store.add_pending_approval(
                f"session-{index}",
                user_input,
                "APPROVAL_TOOL_CANARY" if index == 7 else f"tool_{index}",
                "bounded fixture",
            )

        for index in range(9):
            name = (
                f"JOB_FAMILY_CANARY at {private_path}; retained"
                if index == 0
                else f"BOUNDED_JOB_{index:02d}"
            )
            if index == 8:
                name = "JOB_OVERFLOW_CANARY"
            runtime.store.upsert_job(
                name,
                60,
                "JOB_TYPE_CANARY" if index == 0 else "daily_brief",
                f"2042-02-{index + 1:02d}T08:00:00Z",
            )

        snapshot = runtime.store.read_daily_plan_snapshot()
        if (
            len(snapshot.tasks) != 5
            or len(snapshot.goals) != 3
            or len(snapshot.open_steps) != 3
            or len(snapshot.approvals) != 8
            or len(snapshot.jobs) != 8
            or snapshot.clipped_sources != ("tasks",)
            or "TASK_CLIPPED_TAIL" in str(snapshot.tasks[0]["body"])
            or str(snapshot.open_steps[0]["body"]) != "FIRST_OPEN_STEP_CANARY at "
            + private_path
            + "; retained"
        ):
            raise SystemExit(
                "daily-plan snapshot bounds or first-open-step selection drifted: "
                f"tasks={len(snapshot.tasks)} goals={len(snapshot.goals)} "
                f"steps={len(snapshot.open_steps)} approvals={len(snapshot.approvals)} "
                f"jobs={len(snapshot.jobs)}"
            )
        expected_keys = (
            (snapshot.tasks[0], {"id", "body", "due", "priority", "_clipped"}),
            (snapshot.goals[0], {"id", "title", "_clipped"}),
            (snapshot.open_steps[0], {"id", "goal_id", "body", "_clipped"}),
            (snapshot.approvals[0], {"id", "tool_name", "user_input", "_clipped"}),
            (
                snapshot.jobs[0],
                {"id", "name", "job_type", "enabled", "next_run_at", "_clipped"},
            ),
        )
        for row, keys in expected_keys:
            if set(row.keys()) != keys:
                raise SystemExit(
                    "daily-plan snapshot exposed non-rendered private columns: "
                    f"{set(row.keys())!r}"
                )

        original_connect = runtime.store.connect
        traced_statements: list[str] = []
        publication_connections = [0]

        def traced_connect():
            publication_connections[0] += 1
            conn = original_connect()
            conn.set_trace_callback(traced_statements.append)
            return conn

        with mock.patch.object(
            runtime.store,
            "connect",
            side_effect=traced_connect,
        ), mock.patch.object(
            runtime.store,
            "list_goal_steps",
            side_effect=AssertionError("daily plan performed a duplicate goal-step read"),
        ):
            plan = build_daily_plan(
                runtime.store,
                runtime.vault,
                target_date=TARGET_DATE,
            )

        note_text = _daily_note_text(runtime)
        required = (
            "TASK_FAMILY_CANARY",
            "GOAL_FAMILY_CANARY",
            "FIRST_OPEN_STEP_CANARY",
            "APPROVAL_TOOL_CANARY",
            "APPROVAL_FAMILY_CANARY",
            "JOB_FAMILY_CANARY",
            "JOB_TYPE_CANARY",
            "PATH_VARIANTS",
            "<local-path>",
            "Source Limits",
            "Oversized source text was clipped",
        )
        forbidden = (
            private_path,
            str(root),
            "SECOND_OPEN_STEP_MUST_NOT_RENDER",
            "TASK_OVERFLOW_CANARY",
            "GOAL_OVERFLOW_CANARY",
            "APPROVAL_OVERFLOW_CANARY",
            "JOB_OVERFLOW_CANARY",
            "TASK_CLIPPED_TAIL",
            "/home/alice/plan.txt",
            "/root/private.txt",
            "/Volumes/Work/plan.txt",
            "~/plan.txt",
            "C:\\Users\\alice\\plan.txt",
        )
        trace_counts = {
            "begin": sum("BEGIN IMMEDIATE" in item for item in traced_statements),
            "tasks": sum("FROM tasks" in item for item in traced_statements),
            "goals": sum("FROM goals" in item for item in traced_statements),
            "steps": sum("WITH ranked_steps AS" in item for item in traced_statements),
            "approvals": sum("FROM pending_approvals" in item for item in traced_statements),
            "jobs": sum("FROM scheduled_jobs" in item for item in traced_statements),
        }
        if (
            publication_connections[0] != 1
            or any(count != 1 for count in trace_counts.values())
            or any(value not in plan or value not in note_text for value in required)
            or any(value in plan or value in note_text for value in forbidden)
            or note_text.count("\n## Jarvis Daily Plan\n") != 1
        ):
            raise SystemExit(
                "daily plan did not render one bounded, private, all-family snapshot: "
                f"connections={publication_connections[0]} traces={trace_counts!r}"
            )


def _prepare_mutation_family(runtime, family: str) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if family == "goal_step":
        goal_id = runtime.store.create_goal(GoalRecord("FENCE_EXISTING_GOAL"))
        if runtime.store.add_goal_step(goal_id, "FENCE_EXISTING_FIRST_STEP") is None:
            raise SystemExit("publication-fence fixture could not create a goal step")
        context["goal_id"] = goal_id
    return context


def _apply_mutation_family(
    runtime,
    family: str,
    context: dict[str, Any],
    canary: str,
) -> int:
    if family == "task":
        return runtime.store.add_task(TaskRecord(canary))
    if family == "goal":
        return runtime.store.create_goal(GoalRecord(canary))
    if family == "goal_step":
        step_id = runtime.store.add_goal_step(int(context["goal_id"]), canary)
        if step_id is None:
            raise RuntimeError("goal-step publication-fence mutation returned no id")
        return step_id
    if family == "pending_approval":
        return runtime.store.add_pending_approval(
            "publication-fence-session",
            canary,
            "publication_fence_tool",
            "publication fence smoke",
        )
    if family == "scheduled_job":
        return runtime.store.upsert_job(
            canary,
            60,
            "daily_brief",
            "2042-07-01T08:00:00Z",
        )
    raise AssertionError(f"unknown daily-plan source family: {family}")


def _mutation_is_visible(
    runtime,
    family: str,
    row_id: int,
    canary: str,
) -> bool:
    if family == "task":
        row = runtime.store.get_task(row_id)
        return row is not None and row["body"] == canary
    if family == "goal":
        row = runtime.store.get_goal(row_id)
        return row is not None and row["title"] == canary
    if family == "goal_step":
        row = runtime.store.get_goal_step(row_id)
        return row is not None and row["body"] == canary
    if family == "pending_approval":
        row = runtime.store.get_approval(row_id)
        return row is not None and row["user_input"] == canary
    if family == "scheduled_job":
        return any(
            int(row["id"]) == row_id and row["name"] == canary
            for row in runtime.store.list_jobs()
        )
    raise AssertionError(f"unknown daily-plan source family: {family}")


def test_publication_snapshot_blocks_every_source_writer_through_append() -> None:
    for index, family in enumerate(SOURCE_FAMILIES):
        with TemporaryDirectory(
            prefix=f"jarvis-daily-plan-{family}-fence-"
        ) as temp:
            runtime = make_temp_runtime(Path(temp))
            context = _prepare_mutation_family(runtime, family)
            target_date = f"2042-07-{index + 1:02d}"
            canary = f"PUBLICATION_FENCE_{family.upper()}_MUTATION"

            append_entered = Event()
            allow_append = Event()
            append_effect_completed = Event()
            writer_attempted = Event()
            writer_finished = Event()
            writer_saw_completed_effect = Event()
            errors: Queue = Queue()
            outputs: Queue = Queue()
            mutation_results: Queue = Queue()
            original_append_effect = obsidian_module._append_text_under_inode_lock

            def held_append_effect(*args, **kwargs):
                append_entered.set()
                if not allow_append.wait(timeout=5):
                    raise RuntimeError("daily append effect was never released")
                result = original_append_effect(*args, **kwargs)
                append_effect_completed.set()
                return result

            def build_plan() -> None:
                try:
                    outputs.put(
                        build_daily_plan(
                            runtime.store,
                            runtime.vault,
                            target_date=target_date,
                        )
                    )
                except BaseException as exc:
                    errors.put(exc)

            def mutate_source() -> None:
                try:
                    if not append_entered.is_set():
                        raise RuntimeError("writer started before the append callback entered")
                    writer_attempted.set()
                    row_id = _apply_mutation_family(
                        runtime,
                        family,
                        context,
                        canary,
                    )
                    mutation_results.put(row_id)
                    if append_effect_completed.is_set():
                        writer_saw_completed_effect.set()
                except BaseException as exc:
                    errors.put(exc)
                finally:
                    writer_finished.set()

            with mock.patch.object(
                obsidian_module,
                "_append_text_under_inode_lock",
                side_effect=held_append_effect,
            ):
                builder = Thread(
                    target=build_plan,
                    name=f"daily-plan-{family}-builder",
                )
                builder.start()
                if not append_entered.wait(timeout=5):
                    allow_append.set()
                    _join_threads((builder,), errors, f"{family} publication fence")
                    raise SystemExit(f"{family} plan never entered its daily append callback")
                writer = Thread(
                    target=mutate_source,
                    name=f"daily-plan-{family}-writer",
                )
                writer.start()
                attempted = writer_attempted.wait(timeout=5)
                committed_before_release = writer_finished.wait(timeout=0.2)
                allow_append.set()
                _join_threads(
                    (builder, writer),
                    errors,
                    f"{family} publication fence",
                )

            if outputs.empty() or mutation_results.empty():
                raise SystemExit(f"{family} publication fence lost its result")
            plan = outputs.get_nowait()
            row_id = int(mutation_results.get_nowait())
            note_text = _daily_note_text(runtime, target_date)
            if (
                not attempted
                or committed_before_release
                or not append_effect_completed.is_set()
                or not writer_saw_completed_effect.is_set()
                or not _mutation_is_visible(runtime, family, row_id, canary)
                or canary in plan
                or canary in note_text
                or note_text.count("\n## Jarvis Daily Plan\n") != 1
            ):
                raise SystemExit(
                    f"{family} writer escaped the daily-plan publication fence: "
                    f"attempted={attempted} early={committed_before_release} "
                    f"effect_completed={append_effect_completed.is_set()}"
                )


def test_daily_appender_authority_is_scoped_and_single_use() -> None:
    with TemporaryDirectory(prefix="jarvis-daily-appender-authority-") as temp:
        runtime = make_temp_runtime(Path(temp))
        escaped = None
        with runtime.vault.daily_append_fence("2042-08-01") as append:
            escaped = append
        try:
            escaped("ESCAPED_HEADING", "ESCAPED_BODY")
        except RuntimeError as exc:
            if "authority is not active" not in str(exc):
                raise SystemExit(f"escaped appender returned the wrong refusal: {exc}")
        else:
            raise SystemExit("daily appender retained authority after its fence exited")

        escaped_path = runtime.vault.root_path / "Daily" / "2042-08-01.md"
        if escaped_path.exists():
            raise SystemExit("escaped daily appender wrote outside its authority scope")

        with runtime.vault.daily_append_fence("2042-08-02") as append:
            append("SINGLE_USE_HEADING", "SINGLE_USE_BODY")
            try:
                append("DUPLICATE_HEADING", "DUPLICATE_BODY")
            except RuntimeError as exc:
                if "already attempted" not in str(exc):
                    raise SystemExit(f"second appender use returned the wrong refusal: {exc}")
            else:
                raise SystemExit("daily appender allowed a second publication")

        note_text = _daily_note_text(runtime, "2042-08-02")
        if (
            note_text.count("## SINGLE_USE_HEADING") != 1
            or note_text.count("SINGLE_USE_BODY") != 1
            or "DUPLICATE_HEADING" in note_text
            or "DUPLICATE_BODY" in note_text
        ):
            raise SystemExit("single-use appender refusal did not preserve one exact effect")

        start = Barrier(3)
        outcomes: Queue = Queue()

        def publish_once(label: str) -> None:
            start.wait(timeout=5)
            try:
                append(f"CONCURRENT_{label}", f"CONCURRENT_BODY_{label}")
                outcomes.put(("published", label))
            except RuntimeError as exc:
                outcomes.put(("refused", str(exc)))

        with runtime.vault.daily_append_fence("2042-08-04") as append:
            first = Thread(target=publish_once, args=("A",))
            second = Thread(target=publish_once, args=("B",))
            first.start()
            second.start()
            start.wait(timeout=5)
            first.join(timeout=5)
            second.join(timeout=5)
        if first.is_alive() or second.is_alive():
            raise SystemExit("concurrent daily appenders did not finish")
        concurrent = tuple(outcomes.get_nowait() for _ in range(2))
        concurrent_text = _daily_note_text(runtime, "2042-08-04")
        if (
            sum(item[0] == "published" for item in concurrent) != 1
            or sum(item[0] == "refused" for item in concurrent) != 1
            or concurrent_text.count("## CONCURRENT_") != 1
            or concurrent_text.count("CONCURRENT_BODY_") != 1
        ):
            raise SystemExit(
                "concurrent daily append authority was not exactly once: "
                f"{concurrent!r}"
            )


def test_publication_snapshot_blocks_spawned_process_writer_and_releases() -> None:
    with TemporaryDirectory(prefix="jarvis-daily-plan-process-fence-") as temp:
        runtime = make_temp_runtime(Path(temp))
        context = get_context("spawn")
        parent_pipe, child_pipe = context.Pipe()
        process = context.Process(
            target=_process_add_task,
            args=(str(runtime.store.db_path), child_pipe),
        )
        process_started = False
        outcome: object = None
        try:
            with runtime.vault.daily_append_fence("2042-08-03") as append:
                with runtime.store.daily_plan_publication_snapshot() as snapshot:
                    process.start()
                    process_started = True
                    child_pipe.close()
                    if not parent_pipe.poll(5) or parent_pipe.recv() != "attempting":
                        raise SystemExit("spawned daily-plan writer did not start")
                    if parent_pipe.poll(0.2):
                        raise SystemExit(
                            "spawned writer committed before daily-plan publication"
                        )
                    append(
                        "Jarvis Daily Plan",
                        "Plan date: 2042-08-03\n\n## Focus Queue\n- Process fence proof",
                    )
                    if any(
                        str(row["body"]) == "PROCESS_DAILY_PLAN_WRITER"
                        for row in snapshot.tasks
                    ):
                        raise SystemExit("publication snapshot included the blocked process write")
            if not parent_pipe.poll(5):
                raise SystemExit("spawned writer stayed blocked after publication")
            outcome = parent_pipe.recv()
        finally:
            if not process_started:
                child_pipe.close()
            parent_pipe.close()
            if process_started:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
        if process.exitcode != 0 or not isinstance(outcome, tuple) or outcome[0] != "done":
            raise SystemExit(f"spawned daily-plan writer failed after release: {outcome!r}")
        row = runtime.store.get_task(int(outcome[1]))
        if row is None or row["body"] != "PROCESS_DAILY_PLAN_WRITER":
            raise SystemExit("spawned daily-plan writer did not commit after release")


def test_uncertain_append_consumes_authority_before_effect() -> None:
    with TemporaryDirectory(prefix="jarvis-daily-plan-uncertain-authority-") as temp:
        runtime = make_temp_runtime(Path(temp))
        original_append = obsidian_module._append_text_under_inode_lock

        def append_then_fail(*args, **kwargs):
            original_append(*args, **kwargs)
            raise OSError("representative post-write durability uncertainty")

        with mock.patch.object(
            obsidian_module,
            "_append_text_under_inode_lock",
            side_effect=append_then_fail,
        ):
            with runtime.vault.daily_append_fence("2042-08-05") as append:
                try:
                    append("UNCERTAIN_HEADING", "UNCERTAIN_BODY")
                except OSError:
                    pass
                else:
                    raise SystemExit("uncertain daily append fixture did not fail")
                try:
                    append("RETRY_HEADING", "RETRY_BODY")
                except RuntimeError as exc:
                    if "already attempted" not in str(exc):
                        raise SystemExit(
                            f"uncertain append retry returned the wrong refusal: {exc}"
                        )
                else:
                    raise SystemExit("uncertain daily append authority was reusable")

        note_text = _daily_note_text(runtime, "2042-08-05")
        if (
            note_text.count("## UNCERTAIN_HEADING") != 1
            or note_text.count("UNCERTAIN_BODY") != 1
            or "RETRY_HEADING" in note_text
            or "RETRY_BODY" in note_text
        ):
            raise SystemExit("uncertain daily append produced more than one effect")


def test_inode_open_registration_is_fork_atomic_for_both_append_paths() -> None:
    with TemporaryDirectory(prefix="jarvis-daily-plan-inode-registration-") as temp:
        runtime = make_temp_runtime(Path(temp))
        target_date = "2042-08-09"
        target_name = f"{target_date}.md"
        original_open = obsidian_module.os.open
        guard_states: list[bool] = []

        def observed_open(path, *args, **kwargs):
            if path == target_name:
                guard_states.append(
                    obsidian_module._ADVISORY_LOCK_FDS_GUARD.locked()
                )
            return original_open(path, *args, **kwargs)

        with mock.patch.object(
            obsidian_module.os,
            "open",
            side_effect=observed_open,
        ):
            runtime.vault.append_daily_for_date(
                target_date,
                "ORDINARY_HEADING",
                "ORDINARY_BODY",
            )
            _, appended = runtime.vault.append_daily_once_for_date(
                target_date,
                "daily-plan-inode-registration-marker",
                "IDEMPOTENT_HEADING",
                "IDEMPOTENT_BODY",
            )
        if not appended or len(guard_states) < 2 or not all(guard_states):
            raise SystemExit(
                "inode descriptor was open before fork-safe registration: "
                f"{guard_states!r}"
            )

        runtime.vault.append_daily_for_date(
            target_date,
            "AFTER_IDEMPOTENT_HEADING",
            "AFTER_IDEMPOTENT_BODY",
        )
        note_text = _daily_note_text(runtime, target_date)
        if (
            note_text.count("\n## IDEMPOTENT_HEADING\n") != 1
            or note_text.count("\n## AFTER_IDEMPOTENT_HEADING\n") != 1
        ):
            raise SystemExit("idempotent inode append did not release its lock")


def test_forked_child_cannot_retain_daily_append_authority() -> None:
    if "fork" not in get_all_start_methods():
        return
    with TemporaryDirectory(prefix="jarvis-daily-plan-fork-authority-") as temp:
        runtime = make_temp_runtime(Path(temp))
        context = get_context("fork")
        parent_pipe, child_pipe = context.Pipe()
        process = None
        try:
            with runtime.vault.daily_append_fence("2042-08-06") as append:
                process = context.Process(
                    target=_fork_try_inherited_appender,
                    args=(append, child_pipe),
                )
                process.start()
                child_pipe.close()
                if not parent_pipe.poll(5) or parent_pipe.recv() != "ready":
                    raise SystemExit("forked appender child did not initialize")
            parent_pipe.send("try-after-parent-exit")
            if not parent_pipe.poll(5):
                raise SystemExit("forked appender child did not return a result")
            outcome = parent_pipe.recv()
        finally:
            parent_pipe.close()
            if process is None:
                child_pipe.close()
            else:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
        if (
            process is None
            or process.exitcode != 0
            or not isinstance(outcome, tuple)
            or outcome[:2] != ("refused", "RuntimeError")
            or "authority is not active" not in str(outcome[2])
            or _daily_note_text(runtime, "2042-08-06")
        ):
            raise SystemExit(f"forked child retained daily append authority: {outcome!r}")


def test_forked_child_cannot_retain_inode_lock_after_holder_crash() -> None:
    if "fork" not in get_all_start_methods():
        return
    with TemporaryDirectory(prefix="jarvis-daily-plan-fork-inode-") as temp:
        runtime = make_temp_runtime(Path(temp))
        target_date = "2042-08-07"
        context = get_context("spawn")
        parent_pipe, child_pipe = context.Pipe()
        holder = context.Process(
            target=_process_crash_with_forked_inode_lock,
            args=(str(runtime.vault.root_path), target_date, child_pipe),
        )
        holder.start()
        child_pipe.close()
        try:
            if not parent_pipe.poll(5):
                raise SystemExit("forked inode-lock grandchild did not initialize")
            ready = parent_pipe.recv()
            if not isinstance(ready, tuple) or ready[0] != "grandchild_ready":
                raise SystemExit(f"forked inode-lock fixture drifted: {ready!r}")
            holder.join(timeout=5)
            if holder.is_alive() or holder.exitcode != 0:
                raise SystemExit("inode-lock holder did not crash cleanly")
            started = time.monotonic()
            runtime.vault.append_daily_for_date(
                target_date,
                "POST_CRASH_HEADING",
                "POST_CRASH_BODY",
            )
            elapsed = time.monotonic() - started
            try:
                parent_pipe.send("release-grandchild")
            except (BrokenPipeError, EOFError, OSError):
                pass
        finally:
            parent_pipe.close()
            if holder.is_alive():
                holder.terminate()
                holder.join(timeout=5)
        if elapsed >= 1.0:
            raise SystemExit(
                "forked child stranded the inherited inode lock after holder crash: "
                f"{elapsed:.3f}s"
            )
        note_text = _daily_note_text(runtime, target_date)
        if note_text.count("POST_CRASH_HEADING") != 1:
            raise SystemExit("post-crash append did not publish exactly once")


def test_child_context_cleanup_skips_inherited_publication_mutex() -> None:
    if "fork" not in get_all_start_methods():
        return
    with TemporaryDirectory(prefix="jarvis-daily-plan-fork-cleanup-") as temp:
        runtime = make_temp_runtime(Path(temp))
        entered = Event()
        release = Event()
        errors: Queue = Queue()
        original_append = obsidian_module._append_text_under_inode_lock

        def held_append(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("fork cleanup fixture was not released")
            return original_append(*args, **kwargs)

        authority = runtime.vault.daily_append_fence("2042-08-08")
        append = authority.__enter__()

        def publish() -> None:
            try:
                append("PARENT_HEADING", "PARENT_BODY")
            except BaseException as exc:
                errors.put(exc)

        with mock.patch.object(
            obsidian_module,
            "_append_text_under_inode_lock",
            side_effect=held_append,
        ):
            worker = Thread(target=publish, name="daily-plan-fork-cleanup-writer")
            worker.start()
            if not entered.wait(timeout=5):
                release.set()
                worker.join(timeout=5)
                authority.__exit__(None, None, None)
                raise SystemExit("fork cleanup fixture never held its publication mutex")
            pid = os.fork()
            if pid == 0:
                signal.signal(signal.SIGALRM, lambda *_: os._exit(42))
                signal.alarm(2)
                try:
                    authority.__exit__(None, None, None)
                except BaseException:
                    os._exit(43)
                os._exit(0)
            _, status = os.waitpid(pid, 0)
            release.set()
            worker.join(timeout=5)
        authority.__exit__(None, None, None)
        if worker.is_alive() or not errors.empty():
            raise SystemExit(
                "parent publication failed after fork cleanup: "
                f"{errors.get_nowait()!r}" if not errors.empty() else "worker hung"
            )
        if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
            raise SystemExit(
                "child context cleanup touched inherited publication mutex: "
                f"status={status}"
            )


def main() -> None:
    test_read_snapshot_stays_coherent_during_goal_step_mutation()
    test_bounded_all_family_render_uses_one_snapshot_and_redacts_paths()
    test_publication_snapshot_blocks_every_source_writer_through_append()
    test_daily_appender_authority_is_scoped_and_single_use()
    test_publication_snapshot_blocks_spawned_process_writer_and_releases()
    test_uncertain_append_consumes_authority_before_effect()
    test_inode_open_registration_is_fork_atomic_for_both_append_paths()
    test_forked_child_cannot_retain_daily_append_authority()
    test_forked_child_cannot_retain_inode_lock_after_holder_crash()
    test_child_context_cleanup_skips_inherited_publication_mutex()
    print("PASS: daily plan source snapshot and publication fencing")


if __name__ == "__main__":
    main()
