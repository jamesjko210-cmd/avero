from __future__ import annotations

from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Event, Thread, current_thread
from typing import Any
from unittest import mock

from jarvis_v2.automations.jobs import build_daily_brief
from jarvis_v2.memory import obsidian as obsidian_module
from jarvis_v2.memory.store import (
    DecisionRecord,
    GoalRecord,
    MemoryRecord,
    PersonRecord,
    PreferenceRecord,
    SkillRecord,
    TaskRecord,
    profile_note_source_key,
)
from jarvis_v2.scripts.test_runtime import make_temp_runtime


TARGET_DATE = "2042-08-15"
OLD_STEP = "DAILY_BRIEF_SNAPSHOT_OLD_STEP"
NEW_STEP = "DAILY_BRIEF_SNAPSHOT_NEW_STEP"
SOURCE_FAMILIES = (
    "memory",
    "skill",
    "goal",
    "goal_step",
    "task",
    "pending_approval",
    "decision",
    "person",
    "preference",
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
    with TemporaryDirectory(prefix="jarvis-daily-brief-read-snapshot-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.add_memory(MemoryRecord("fixture", "anchor", "anchor"))
        goal_id = runtime.store.create_goal(GoalRecord("SNAPSHOT_RACE_GOAL"))
        old_step_id = runtime.store.add_goal_step(goal_id, OLD_STEP)
        if old_step_id is None:
            raise SystemExit("daily-brief snapshot fixture could not create its old step")

        step_query_reached = Event()
        mutation_finished = Event()
        synchronization_timed_out = Event()
        snapshots: Queue = Queue()
        errors: Queue = Queue()
        reader_name = "daily-brief-snapshot-reader"
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
                snapshots.put(runtime.store.read_daily_brief_snapshot())
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

        with mock.patch.object(runtime.store, "connect", side_effect=instrumented_connect):
            reader = Thread(target=read_snapshot, name=reader_name)
            mutator = Thread(target=replace_first_open_step, name="daily-brief-snapshot-mutator")
            reader.start()
            mutator.start()
            _join_threads((reader, mutator), errors, "daily-brief read snapshot race")

        if snapshots.empty() or synchronization_timed_out.is_set():
            raise SystemExit("daily-brief read snapshot race missed its forced boundary")
        snapshot = snapshots.get_nowait()
        after = runtime.store.read_daily_brief_snapshot()
        before_steps = [
            str(row["body"])
            for row in snapshot.open_steps
            if int(row["goal_id"]) == goal_id
        ]
        after_steps = [
            str(row["body"])
            for row in after.open_steps
            if int(row["goal_id"]) == goal_id
        ]
        if step_query_count[0] != 1 or before_steps != [OLD_STEP] or after_steps != [NEW_STEP]:
            raise SystemExit(
                "daily-brief read mixed goal and step generations: "
                f"query_count={step_query_count[0]} before={before_steps!r} "
                f"after={after_steps!r}"
            )


def _seed_all_bounded_sources(runtime, private_path: str) -> None:
    profile_record = MemoryRecord(
        "identity",
        "PROFILE_OWNED_MEMORY_MUST_NOT_RENDER",
        "PROFILE_OWNED_BODY_MUST_NOT_RENDER",
        "profile",
        1.0,
    )
    runtime.store.ensure_profile_note_memory_with_projection(
        profile_record,
        profile_note_source_key(
            profile_record.title,
            profile_record.body,
            profile_record.category,
        ),
    )
    runtime.store.add_memory(MemoryRecord("overflow", "MEMORY_OVERFLOW_CANARY", "overflow"))
    for index in range(8):
        body = (
            f"MEMORY_FAMILY_CANARY at {private_path} "
            + "x" * 200
            + " MEMORY_EXCERPT_TAIL"
            if index == 7
            else f"MEMORY_{index:02d}"
        )
        runtime.store.add_memory(MemoryRecord("brief", f"memory-{index:02d}", body))

    runtime.store.save_skill(SkillRecord("SKILL_OVERFLOW_CANARY", "overflow", "body"))
    for index in range(5):
        trigger = (
            f"SKILL_FAMILY_CANARY at {private_path} " + "x" * 2200 + " SKILL_CLIPPED_TAIL"
            if index == 4
            else f"skill trigger {index:02d}"
        )
        runtime.store.save_skill(SkillRecord(f"brief-skill-{index:02d}", trigger, "body"))

    overflow_goal = runtime.store.create_goal(GoalRecord("GOAL_OVERFLOW_CANARY"))
    runtime.store.add_goal_step(overflow_goal, "overflow step")
    for index in range(5):
        title = f"GOAL_FAMILY_CANARY at {private_path}" if index == 4 else f"GOAL_{index:02d}"
        goal_id = runtime.store.create_goal(GoalRecord(title))
        first = (
            f"FIRST_OPEN_STEP_CANARY at {private_path}"
            if index == 4
            else f"FIRST_STEP_{index:02d}"
        )
        if runtime.store.add_goal_step(goal_id, first) is None:
            raise SystemExit("daily-brief bounded fixture lost a first goal step")
        if index == 4 and runtime.store.add_goal_step(
            goal_id, "SECOND_OPEN_STEP_MUST_NOT_RENDER"
        ) is None:
            raise SystemExit("daily-brief bounded fixture lost its second goal step")

    for index in range(6):
        body = f"TASK_FAMILY_CANARY at {private_path}" if index == 0 else f"TASK_{index:02d}"
        runtime.store.add_task(TaskRecord(body, due=f"2042-08-{index + 1:02d}", priority="high"))
    runtime.store.add_task(TaskRecord("TASK_OVERFLOW_CANARY", priority="low"))

    runtime.store.add_pending_approval(
        "overflow-session", "APPROVAL_OVERFLOW_CANARY", "overflow_tool", "fixture"
    )
    for index in range(5):
        user_input = (
            f"APPROVAL_FAMILY_CANARY at {private_path}" if index == 4 else f"APPROVAL_{index:02d}"
        )
        runtime.store.add_pending_approval(
            f"session-{index}", user_input, f"approval_tool_{index}", "fixture"
        )

    runtime.store.add_decision(DecisionRecord("DECISION_OVERFLOW_CANARY"))
    for index in range(5):
        rationale = (
            f"DECISION_FAMILY_CANARY at {private_path}" if index == 4 else f"RATIONALE_{index:02d}"
        )
        runtime.store.add_decision(DecisionRecord(f"decision-{index:02d}", rationale))

    runtime.store.upsert_person(PersonRecord("PERSON_OVERFLOW_CANARY"))
    for index in range(5):
        name = "PERSON_FAMILY_CANARY" if index == 4 else f"person-{index:02d}"
        relation = f"collaborator at {private_path}" if index == 4 else "collaborator"
        runtime.store.upsert_person(PersonRecord(name, relation))

    for index in range(6):
        value = f"PREFERENCE_FAMILY_CANARY at {private_path}" if index == 0 else f"value-{index:02d}"
        runtime.store.set_preference(PreferenceRecord(f"key-{index:02d}", value, "brief"))
    runtime.store.set_preference(
        PreferenceRecord("zz-overflow", "PREFERENCE_OVERFLOW_CANARY", "zz-overflow")
    )


def test_bounded_all_family_render_uses_one_private_snapshot() -> None:
    with TemporaryDirectory(prefix="jarvis-daily-brief-bounded-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        private_path = str(root / "private" / "daily-brief-source.txt")
        _seed_all_bounded_sources(runtime, private_path)

        snapshot = runtime.store.read_daily_brief_snapshot()
        observed_lengths = tuple(
            len(rows)
            for rows in (
                snapshot.memories,
                snapshot.skills,
                snapshot.goals,
                snapshot.open_steps,
                snapshot.tasks,
                snapshot.approvals,
                snapshot.decisions,
                snapshot.people,
                snapshot.preferences,
            )
        )
        if (
            observed_lengths != (8, 5, 5, 5, 6, 5, 5, 5, 6)
            or snapshot.truncated_sources
            != ("memories", "skills", "goals", "tasks", "approvals", "decisions", "people", "preferences")
            or snapshot.clipped_sources != ("skills",)
        ):
            raise SystemExit(
                "daily-brief snapshot bounds changed: "
                f"lengths={observed_lengths!r} truncated={snapshot.truncated_sources!r} "
                f"clipped={snapshot.clipped_sources!r}"
            )

        expected_keys = (
            (snapshot.memories[0], {"category", "title", "body", "_clipped"}),
            (snapshot.skills[0], {"name", "trigger", "_clipped"}),
            (snapshot.goals[0], {"id", "title", "_clipped"}),
            (snapshot.open_steps[0], {"id", "goal_id", "body", "_clipped"}),
            (snapshot.tasks[0], {"id", "body", "due", "priority", "_clipped"}),
            (snapshot.approvals[0], {"id", "tool_name", "user_input", "_clipped"}),
            (snapshot.decisions[0], {"id", "title", "rationale", "_clipped"}),
            (snapshot.people[0], {"id", "name", "relation", "last_contact_at", "_clipped"}),
            (snapshot.preferences[0], {"category", "key", "value", "_clipped"}),
        )
        for row, keys in expected_keys:
            if set(row.keys()) != keys:
                raise SystemExit(f"daily-brief snapshot exposed private columns: {set(row.keys())!r}")

        original_connect = runtime.store.connect
        statements: list[str] = []
        connection_count = [0]

        def traced_connect():
            connection_count[0] += 1
            conn = original_connect()
            conn.set_trace_callback(statements.append)
            return conn

        with mock.patch.object(runtime.store, "connect", side_effect=traced_connect), mock.patch.object(
            runtime.store,
            "list_goal_steps",
            side_effect=AssertionError("daily brief performed a duplicate goal-step read"),
        ):
            brief = build_daily_brief(runtime.store, runtime.vault, target_date=TARGET_DATE)

        note = _daily_note_text(runtime)
        required = (
            "MEMORY_FAMILY_CANARY",
            "SKILL_FAMILY_CANARY",
            "GOAL_FAMILY_CANARY",
            "FIRST_OPEN_STEP_CANARY",
            "TASK_FAMILY_CANARY",
            "APPROVAL_FAMILY_CANARY",
            "DECISION_FAMILY_CANARY",
            "PERSON_FAMILY_CANARY",
            "PREFERENCE_FAMILY_CANARY",
            "<local-path>",
            "Source Limits",
        )
        forbidden = (
            private_path,
            str(root),
            "SECOND_OPEN_STEP_MUST_NOT_RENDER",
            "MEMORY_OVERFLOW_CANARY",
            "SKILL_OVERFLOW_CANARY",
            "GOAL_OVERFLOW_CANARY",
            "TASK_OVERFLOW_CANARY",
            "APPROVAL_OVERFLOW_CANARY",
            "DECISION_OVERFLOW_CANARY",
            "PERSON_OVERFLOW_CANARY",
            "PREFERENCE_OVERFLOW_CANARY",
            "PROFILE_OWNED_MEMORY_MUST_NOT_RENDER",
            "PROFILE_OWNED_BODY_MUST_NOT_RENDER",
            "MEMORY_EXCERPT_TAIL",
            "SKILL_CLIPPED_TAIL",
        )
        trace_counts = {
            "begin": sum("BEGIN IMMEDIATE" in item for item in statements),
            "memories": sum("FROM memories AS memory" in item for item in statements),
            "skills": sum("FROM skills" in item for item in statements),
            "goals": sum("FROM goals" in item for item in statements),
            "steps": sum("WITH ranked_steps AS" in item for item in statements),
            "tasks": sum("FROM tasks" in item for item in statements),
            "approvals": sum("FROM pending_approvals" in item for item in statements),
            "decisions": sum("FROM decisions" in item for item in statements),
            "people": sum("FROM people" in item for item in statements),
            "preferences": sum("FROM preferences" in item for item in statements),
        }
        if (
            connection_count[0] != 1
            or any(count != 1 for count in trace_counts.values())
            or any("SELECT *" in item.upper() for item in statements)
            or any(value not in brief or value not in note for value in required)
            or any(value in brief or value in note for value in forbidden)
            or note.count("\n## Jarvis Proactive Brief\n") != 1
        ):
            raise SystemExit(
                "daily brief did not render one bounded private snapshot: "
                f"connections={connection_count[0]} traces={trace_counts!r}"
            )


def _prepare_mutation_family(runtime, family: str) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if family == "goal_step":
        context["goal_id"] = runtime.store.create_goal(GoalRecord("FENCE_EXISTING_GOAL"))
    return context


def _apply_mutation_family(runtime, family: str, context: dict[str, Any], canary: str) -> int:
    if family == "memory":
        return runtime.store.add_memory(MemoryRecord("fence", canary, canary))
    if family == "skill":
        return runtime.store.save_skill(SkillRecord(canary, canary, "body"))
    if family == "goal":
        return runtime.store.create_goal(GoalRecord(canary))
    if family == "goal_step":
        row_id = runtime.store.add_goal_step(int(context["goal_id"]), canary)
        if row_id is None:
            raise RuntimeError("goal-step publication-fence mutation returned no id")
        return row_id
    if family == "task":
        return runtime.store.add_task(TaskRecord(canary))
    if family == "pending_approval":
        return runtime.store.add_pending_approval("fence-session", canary, "fence_tool", "fixture")
    if family == "decision":
        return runtime.store.add_decision(DecisionRecord(canary, canary))
    if family == "person":
        return runtime.store.upsert_person(PersonRecord(canary, "fixture"))
    if family == "preference":
        return runtime.store.set_preference(PreferenceRecord(canary, canary, "fence"))
    raise AssertionError(f"unknown daily-brief source family: {family}")


def _mutation_is_visible(runtime, family: str, row_id: int, canary: str) -> bool:
    getters = {
        "memory": runtime.store.get_memory,
        "skill": runtime.store.get_skill_by_id,
        "goal": runtime.store.get_goal,
        "goal_step": runtime.store.get_goal_step,
        "task": runtime.store.get_task,
        "pending_approval": runtime.store.get_approval,
        "decision": runtime.store.get_decision,
        "person": runtime.store.get_person,
        "preference": runtime.store.get_preference_by_id,
    }
    row = getters[family](row_id)
    if row is None:
        return False
    fields = {
        "memory": "title",
        "skill": "name",
        "goal": "title",
        "goal_step": "body",
        "task": "body",
        "pending_approval": "user_input",
        "decision": "title",
        "person": "name",
        "preference": "key",
    }
    return str(row[fields[family]]) == canary


def test_publication_snapshot_blocks_every_source_writer_through_append() -> None:
    for index, family in enumerate(SOURCE_FAMILIES):
        with TemporaryDirectory(prefix=f"jarvis-daily-brief-{family}-fence-") as temp:
            runtime = make_temp_runtime(Path(temp))
            context = _prepare_mutation_family(runtime, family)
            target_date = f"2042-09-{index + 1:02d}"
            canary = (
                "Publication Fence Person Mutation"
                if family == "person"
                else f"PUBLICATION_FENCE_{family.upper()}_MUTATION"
            )
            append_entered = Event()
            allow_append = Event()
            append_completed = Event()
            writer_attempted = Event()
            writer_finished = Event()
            writer_saw_effect = Event()
            outputs: Queue = Queue()
            mutation_results: Queue = Queue()
            errors: Queue = Queue()
            original_effect = obsidian_module._append_text_under_inode_lock

            def held_effect(*args, **kwargs):
                append_entered.set()
                if not allow_append.wait(timeout=5):
                    raise RuntimeError("daily-brief append was never released")
                result = original_effect(*args, **kwargs)
                append_completed.set()
                return result

            def build_brief() -> None:
                try:
                    outputs.put(build_daily_brief(runtime.store, runtime.vault, target_date=target_date))
                except BaseException as exc:
                    errors.put(exc)

            def mutate_source() -> None:
                try:
                    writer_attempted.set()
                    row_id = _apply_mutation_family(runtime, family, context, canary)
                    mutation_results.put(row_id)
                    if append_completed.is_set():
                        writer_saw_effect.set()
                except BaseException as exc:
                    errors.put(exc)
                finally:
                    writer_finished.set()

            with mock.patch.object(
                obsidian_module, "_append_text_under_inode_lock", side_effect=held_effect
            ):
                builder = Thread(target=build_brief, name=f"daily-brief-{family}-builder")
                builder.start()
                if not append_entered.wait(timeout=5):
                    allow_append.set()
                    _join_threads((builder,), errors, f"{family} publication fence")
                    raise SystemExit(f"{family} brief never entered its append effect")
                writer = Thread(target=mutate_source, name=f"daily-brief-{family}-writer")
                writer.start()
                attempted = writer_attempted.wait(timeout=5)
                committed_early = writer_finished.wait(timeout=0.2)
                allow_append.set()
                _join_threads((builder, writer), errors, f"{family} publication fence")

            if outputs.empty() or mutation_results.empty():
                raise SystemExit(f"{family} publication fence lost its result")
            brief = outputs.get_nowait()
            row_id = int(mutation_results.get_nowait())
            note = _daily_note_text(runtime, target_date)
            if (
                not attempted
                or committed_early
                or not append_completed.is_set()
                or not writer_saw_effect.is_set()
                or not _mutation_is_visible(runtime, family, row_id, canary)
                or canary in brief
                or canary in note
                or note.count("\n## Jarvis Proactive Brief\n") != 1
            ):
                raise SystemExit(
                    f"{family} writer escaped daily-brief publication custody: "
                    f"attempted={attempted} early={committed_early} "
                    f"effect_completed={append_completed.is_set()}"
                )


def test_scheduled_render_reuses_one_read_snapshot_without_publishing() -> None:
    with TemporaryDirectory(prefix="jarvis-daily-brief-scheduled-snapshot-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.add_memory(MemoryRecord("scheduled", "SCHEDULED_BRIEF_CANARY", "visible"))
        original_read = runtime.store.read_daily_brief_snapshot
        with mock.patch.object(
            runtime.store, "read_daily_brief_snapshot", wraps=original_read
        ) as read_snapshot, mock.patch.object(
            runtime.store,
            "daily_brief_publication_snapshot",
            side_effect=AssertionError("scheduled render entered direct publication custody"),
        ):
            brief = build_daily_brief(
                runtime.store,
                runtime.vault,
                scheduled_note_key="daily-brief-occurrence",
                scheduled_note_date=TARGET_DATE,
            )
        daily_dir = runtime.vault.root_path / "Daily"
        if (
            type(brief) is not str
            or "SCHEDULED_BRIEF_CANARY" not in brief
            or read_snapshot.call_count != 1
            or (daily_dir.is_dir() and any(daily_dir.iterdir()))
        ):
            raise SystemExit("scheduled daily brief did not reuse one non-publishing snapshot")


def main() -> None:
    test_read_snapshot_stays_coherent_during_goal_step_mutation()
    test_bounded_all_family_render_uses_one_private_snapshot()
    test_publication_snapshot_blocks_every_source_writer_through_append()
    test_scheduled_render_reuses_one_read_snapshot_without_publishing()
    print("PASS: daily brief source snapshot and publication fencing")


if __name__ == "__main__":
    main()
