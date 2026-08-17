from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from multiprocessing import get_context
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Barrier, Event, Thread, current_thread
from unittest import mock

from jarvis_v2.automations import scheduler as scheduler_module
from jarvis_v2.automations import telegram_control
from jarvis_v2.automations.jobs import GoalNudgeBuild, build_goal_nudge
from jarvis_v2.automations.scheduler import MORNING_BRIEF_ENV, Scheduler, iso
from jarvis_v2.memory import obsidian as obsidian_module
from jarvis_v2.memory import store as store_module
from jarvis_v2.memory.store import (
    GoalRecord,
    MemoryStore,
    OrganizedNoteEntryRecord,
)
from jarvis_v2.scripts.test_runtime import make_temp_runtime


OLD_STEP = "SNAPSHOT_OLD_FIRST_STEP"
NEW_STEP = "SNAPSHOT_NEW_FIRST_STEP"


def _process_add_goal_step(db_path: str, goal_id: int, pipe) -> None:
    store = MemoryStore(Path(db_path))
    pipe.send("attempting")
    try:
        pipe.send(("done", store.add_goal_step(goal_id, "PROCESS_FENCE_STEP")))
    except BaseException as exc:
        pipe.send(("error", type(exc).__name__))
    finally:
        pipe.close()


def _process_crash_while_holding_goal_fence(db_path: str, pipe) -> None:
    store = MemoryStore(Path(db_path))
    with store.goal_source_effect_fence():
        pipe.send("held")
        pipe.close()
        os._exit(0)


def _create_stale_goal(runtime, title: str, body: str) -> tuple[int, int]:
    goal_id = runtime.store.create_goal(GoalRecord(title=title))
    step_id = runtime.store.add_goal_step(goal_id, body)
    if step_id is None:
        raise SystemExit("goal-nudge fixture could not create its first step")
    with runtime.store.connect() as conn:
        conn.execute(
            "UPDATE goals SET updated_at = ? WHERE id = ?",
            ("2001-01-01T00:00:00Z", goal_id),
        )
    return goal_id, step_id


def test_goal_nudge_coalesces_only_normalized_duplicate_titles() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-nudge-duplicate-titles-") as temp:
        runtime = make_temp_runtime(Path(temp))
        older_spanish_id, _ = _create_stale_goal(
            runtime,
            "Learn Spanish",
            "choose a tutor",
        )
        newer_spanish_id, _ = _create_stale_goal(
            runtime,
            "  learn   SPANISH  ",
            "",
        )
        french_id, _ = _create_stale_goal(
            runtime,
            "Learn French",
            "pick a course",
        )

        build = build_goal_nudge(
            runtime.store,
            runtime.vault,
            scheduled_note_key="duplicate-title-render-only",
            include_source_manifest=True,
        )
        if not isinstance(build, GoalNudgeBuild):
            raise SystemExit(f"duplicate-title nudge did not return a build: {build!r}")
        expected_spanish = (
            f"- #{newer_spanish_id} (+1 duplicate-title goal) learn SPANISH "
            "| next: choose a tutor"
        )
        expected_french = f"- #{french_id} Learn French | next: pick a course"
        if (
            expected_spanish not in build.output
            or expected_french not in build.output
            or f"- #{older_spanish_id} Learn Spanish" in build.output
            or build.output.count("duplicate-title goal") != 1
        ):
            raise SystemExit(
                "goal nudge should coalesce only whitespace/case-normalized duplicate titles "
                "while preserving a useful existing next step: "
                f"{build.output!r}"
            )


def _due_goal_nudge(runtime, name: str) -> int:
    return runtime.store.upsert_job(
        name,
        1440,
        "goal_nudge",
        iso(datetime.now() - timedelta(minutes=5)),
    )


def _publication_rows(runtime):
    with runtime.store.connect() as conn:
        return list(
            conn.execute(
                "SELECT * FROM scheduled_note_publications ORDER BY publication_key"
            )
        )


def _run_due(scheduler: Scheduler) -> str:
    with mock.patch.dict(os.environ, {MORNING_BRIEF_ENV: ""}, clear=False):
        return scheduler.run_due_jobs()


def _accepted(message_id: int) -> dict[str, object]:
    return {"ok": True, "result": {"message_id": message_id}}


def _join_threads(threads: tuple[Thread, ...], errors: Queue, label: str) -> None:
    for thread in threads:
        thread.join(timeout=10)
    alive = [thread.name for thread in threads if thread.is_alive()]
    if alive:
        raise SystemExit(f"{label} threads did not finish: {alive}")
    if not errors.empty():
        raise SystemExit(f"{label} worker failed: {errors.get()!r}")


def test_build_goal_nudge_uses_one_sqlite_snapshot_during_step_mutation() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-nudge-snapshot-") as temp:
        runtime = make_temp_runtime(Path(temp))
        goal_id, old_step_id = _create_stale_goal(
            runtime,
            "SNAPSHOT_STALE_GOAL",
            OLD_STEP,
        )
        before = runtime.store.read_goal_nudge_snapshot()

        start = Barrier(3)
        reader_at_step_query = Event()
        mutation_finished = Event()
        synchronization_timed_out = Event()
        results: Queue = Queue()
        errors: Queue = Queue()
        mutation_results: Queue = Queue()
        reader_name = "goal-nudge-snapshot-reader"
        original_connect = runtime.store.connect

        def instrumented_connect():
            conn = original_connect()
            if current_thread().name == reader_name:
                def trace(statement: str) -> None:
                    if (
                        "WITH selected_goals AS" in statement
                        and not reader_at_step_query.is_set()
                    ):
                        reader_at_step_query.set()
                        if not mutation_finished.wait(timeout=5):
                            synchronization_timed_out.set()

                conn.set_trace_callback(trace)
            return conn

        def read_snapshot() -> None:
            try:
                start.wait(timeout=5)
                results.put(
                    build_goal_nudge(
                        runtime.store,
                        runtime.vault,
                        scheduled_note_key="snapshot-race-no-effect",
                        include_source_manifest=True,
                    )
                )
            except BaseException as exc:
                errors.put(exc)

        def mutate_steps() -> None:
            try:
                start.wait(timeout=5)
                if not reader_at_step_query.wait(timeout=5):
                    raise RuntimeError("reader never reached the second snapshot query")
                completed = runtime.store.complete_goal_step(old_step_id)
                new_step_id = runtime.store.add_goal_step(goal_id, NEW_STEP)
                if completed is None or completed["status"] != "done" or new_step_id is None:
                    raise RuntimeError("concurrent goal-step mutation did not commit")
                mutation_results.put(new_step_id)
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
            mutator = Thread(target=mutate_steps, name="goal-nudge-step-mutator")
            reader.start()
            mutator.start()
            start.wait(timeout=5)
            _join_threads((reader, mutator), errors, "goal-nudge snapshot race")

        if synchronization_timed_out.is_set() or mutation_results.empty():
            raise SystemExit("goal-nudge snapshot race did not cross the intended query boundary")
        build = results.get_nowait() if not results.empty() else None
        after = runtime.store.read_goal_nudge_snapshot()
        expected_line = f"- #{goal_id} SNAPSHOT_STALE_GOAL | next: {OLD_STEP}"
        if not isinstance(build, GoalNudgeBuild):
            raise SystemExit(f"goal nudge did not return its source manifest: {build!r}")
        if (
            build.source_manifest_digest != before.source_manifest_digest
            or build.source_manifest_digest == after.source_manifest_digest
            or expected_line not in build.output
            or NEW_STEP in build.output
        ):
            raise SystemExit(
                "goal nudge mixed its pre-mutation goal snapshot with the new first "
                f"open step: {build!r}"
            )


def test_stale_source_fails_before_note_and_linked_telegram_effects() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-nudge-stale-source-") as temp:
        runtime = make_temp_runtime(Path(temp))
        goal_id, old_step_id = _create_stale_goal(
            runtime,
            "STALE_SOURCE_GOAL",
            "STALE_SOURCE_OLD_STEP",
        )
        _due_goal_nudge(runtime, "Stale Source Goal Nudge")
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)

        start = Barrier(3)
        dispatch_reached = Event()
        mutation_finished = Event()
        synchronization_timed_out = Event()
        prepared_before_mutation = Event()
        outputs: Queue = Queue()
        errors: Queue = Queue()
        original_finalize = scheduler._finalize_claim_with_delivery

        def held_finalize(*args, **kwargs):
            finalized = original_finalize(*args, **kwargs)
            publications = _publication_rows(runtime)
            if len(publications) == 1 and publications[0]["state"] == "prepared":
                prepared_before_mutation.set()
            dispatch_reached.set()
            if not mutation_finished.wait(timeout=5):
                synchronization_timed_out.set()
            return finalized

        def run_scheduler() -> None:
            try:
                start.wait(timeout=5)
                outputs.put(_run_due(scheduler))
            except BaseException as exc:
                errors.put(exc)

        def mutate_after_composition() -> None:
            try:
                start.wait(timeout=5)
                if not dispatch_reached.wait(timeout=5):
                    raise RuntimeError("scheduler never reached scheduled-note dispatch")
                completed = runtime.store.complete_goal_step(old_step_id)
                new_step_id = runtime.store.add_goal_step(
                    goal_id,
                    "STALE_SOURCE_REPLACEMENT_STEP",
                )
                if completed is None or new_step_id is None:
                    raise RuntimeError("post-composition goal mutation did not commit")
            except BaseException as exc:
                errors.put(exc)
            finally:
                mutation_finished.set()

        with mock.patch.object(
            scheduler,
            "_finalize_claim_with_delivery",
            side_effect=held_finalize,
        ), mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            return_value=_accepted(8101),
        ) as send_owner:
            runner = Thread(target=run_scheduler, name="goal-nudge-stale-runner")
            mutator = Thread(
                target=mutate_after_composition,
                name="goal-nudge-stale-mutator",
            )
            runner.start()
            mutator.start()
            start.wait(timeout=5)
            _join_threads((runner, mutator), errors, "stale goal-nudge flow")
            if send_owner.call_count != 0:
                raise SystemExit("stale goal nudge reached owner Telegram")

        if (
            synchronization_timed_out.is_set()
            or not prepared_before_mutation.is_set()
            or outputs.empty()
        ):
            raise SystemExit("stale goal mutation did not occur after durable composition")
        publication_rows = _publication_rows(runtime)
        if len(publication_rows) != 1:
            raise SystemExit(f"stale goal nudge persisted the wrong publication count: {publication_rows}")
        publication = publication_rows[0]
        daily_path = runtime.vault.root_path / "Daily" / f"{publication['target_date']}.md"
        delivery_occurrence = str(publication["delivery_occurrence_key"] or "")
        deliveries = runtime.store.list_scheduled_deliveries_for_occurrence(
            delivery_occurrence
        )
        if (
            publication["state"] != "failed"
            or publication["error_code"] != "source_manifest_stale"
            or publication["payload"] is not None
            or publication["effect_started_at"] is not None
            or daily_path.exists()
            or not deliveries
            or any(
                row["state"] != "failed"
                or row["payload"] is not None
                or row["error_code"] != "blocked_by_note_publication_failure"
                for row in deliveries
            )
        ):
            raise SystemExit(
                "stale goal source did not fail closed and scrub its linked delivery: "
                f"{dict(publication)!r} {[dict(row) for row in deliveries]!r}"
            )


def test_unchanged_source_publishes_then_sends_once_with_opaque_manifest() -> None:
    private_title = "PRIVATE_GOAL_TITLE_CANARY_91d2"
    private_body = "PRIVATE_GOAL_BODY_CANARY_4a8f"
    with TemporaryDirectory(prefix="jarvis-goal-nudge-current-source-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        private_path = str(root / "private" / "goal-source.txt")
        _create_stale_goal(
            runtime,
            private_title,
            f"{private_body} at {private_path}",
        )
        _due_goal_nudge(runtime, "Current Source Goal Nudge")
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)

        effects: list[str] = []
        sent: list[str] = []
        original_append = runtime.vault.append_scheduled_daily_once_for_date

        def append_note(*args, **kwargs):
            result = original_append(*args, **kwargs)
            effects.append("note")
            return result

        def send_owner(text: str) -> dict[str, object]:
            effects.append("telegram")
            sent.append(text)
            return _accepted(8201)

        with mock.patch.object(
            runtime.vault,
            "append_scheduled_daily_once_for_date",
            side_effect=append_note,
        ), mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=send_owner,
        ):
            first_output = _run_due(scheduler)
            second_output = _run_due(scheduler)

        publications = _publication_rows(runtime)
        if len(publications) != 1:
            raise SystemExit(f"current goal nudge persisted the wrong publication count: {publications}")
        publication = publications[0]
        delivery_occurrence = str(publication["delivery_occurrence_key"] or "")
        deliveries = runtime.store.list_scheduled_deliveries_for_occurrence(
            delivery_occurrence
        )
        daily_path = runtime.vault.root_path / "Daily" / f"{publication['target_date']}.md"
        note_text = daily_path.read_text(encoding="utf-8") if daily_path.exists() else ""
        if (
            effects != ["note", "telegram"]
            or len(sent) != 1
            or publication["state"] != "published"
            or not deliveries
            or any(row["state"] != "accepted" for row in deliveries)
            or note_text.count(str(publication["publication_key"])) != 2
            or private_title not in note_text
            or private_body not in note_text
            or "[note published]" not in first_output
            or "accepted by Telegram API" not in first_output
            or second_output != "No jobs due."
        ):
            raise SystemExit(
                "unchanged goal source did not publish its note before one Telegram send: "
                f"effects={effects!r} publication={dict(publication)!r} "
                f"deliveries={[dict(row) for row in deliveries]!r}"
            )

        manifest_keys = {
            key for key in publication.keys() if str(key).startswith("source_manifest_")
        }
        manifest = {
            "source_manifest_kind": publication["source_manifest_kind"],
            "source_manifest_digest": publication["source_manifest_digest"],
        }
        manifest_surface = json.dumps(manifest, sort_keys=True)
        if (
            manifest_keys != {"source_manifest_kind", "source_manifest_digest"}
            or manifest["source_manifest_kind"] != "goal_nudge_v1"
            or re.fullmatch(r"[0-9a-f]{64}", str(manifest["source_manifest_digest"]))
            is None
            or any(
                secret in manifest_surface
                for secret in (private_title, private_body, private_path, str(root))
            )
        ):
            raise SystemExit(f"persisted goal-nudge manifest leaked source data: {manifest!r}")


def test_goal_mutation_waits_until_all_telegram_chunks_complete() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-nudge-delivery-fence-") as temp:
        runtime = make_temp_runtime(Path(temp))
        goal_id, old_step_id = _create_stale_goal(
            runtime,
            "DELIVERY_FENCE_STALE_GOAL",
            "DELIVERY_FENCE_OLD_STEP",
        )
        _due_goal_nudge(runtime, "Delivery Fence Goal Nudge")
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)

        send_started = Event()
        mutation_attempted = Event()
        mutation_finished = Event()
        mutation_saw_all_chunks = Event()
        errors: Queue = Queue()
        sent: list[str] = []

        def three_chunks(text: str, *, limit: int = 3900) -> list[str]:
            del limit
            first = max(1, len(text) // 3)
            second = max(first + 1, (len(text) * 2) // 3)
            return [text[:first], text[first:second], text[second:]]

        def send_owner(text: str) -> dict[str, object]:
            sent.append(text)
            send_started.set()
            if len(sent) == 1:
                if not mutation_attempted.wait(timeout=5):
                    raise RuntimeError("goal mutation was not attempted during Telegram send")
                if mutation_finished.wait(timeout=0.2):
                    raise RuntimeError("goal mutation escaped the multi-chunk source fence")
            return _accepted(8300 + len(sent))

        def mutate_before_send() -> None:
            try:
                if not send_started.wait(timeout=5):
                    raise RuntimeError("delivery never reached the Telegram effect")
                mutation_attempted.set()
                completed = runtime.store.complete_goal_step(old_step_id)
                new_step_id = runtime.store.add_goal_step(
                    goal_id,
                    "DELIVERY_FENCE_REPLACEMENT_STEP",
                )
                if completed is None or new_step_id is None:
                    raise RuntimeError("delivery-fence source mutation did not commit")
                if len(sent) == 3:
                    mutation_saw_all_chunks.set()
            except BaseException as exc:
                errors.put(exc)
            finally:
                mutation_finished.set()

        mutator = Thread(target=mutate_before_send, name="goal-nudge-delivery-mutator")
        mutator.start()
        with mock.patch.object(
            telegram_control,
            "_text_chunks",
            side_effect=three_chunks,
        ), mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=send_owner,
        ) as send_owner:
            output = _run_due(scheduler)
        _join_threads((mutator,), errors, "goal-nudge delivery source fence")

        publications = _publication_rows(runtime)
        if len(publications) != 1:
            raise SystemExit(f"delivery-fence flow lost its publication: {publications}")
        publication = publications[0]
        deliveries = runtime.store.list_scheduled_deliveries_for_occurrence(
            str(publication["delivery_occurrence_key"] or "")
        )
        ordered = sorted(deliveries, key=lambda row: int(row["chunk_index"]))
        if (
            not mutation_attempted.is_set()
            or not mutation_saw_all_chunks.is_set()
            or send_owner.call_count != 3
            or publication["state"] != "published"
            or len(ordered) != 3
            or any(
                row["state"] != "accepted"
                or row["payload"] is not None
                or row["error_code"] is not None
                for row in ordered
            )
            or "accepted by Telegram API" not in output
        ):
            raise SystemExit(
                "goal mutation was not serialized behind the complete Telegram occurrence: "
                f"publication={dict(publication)!r} "
                f"deliveries={[dict(row) for row in ordered]!r} output={output!r}"
            )


def test_wall_clock_staleness_transition_invalidates_quiet_scheduled_nudge() -> None:
    just_fresh_at = "2042-01-08T00:00:00Z"
    newly_stale_at = "2042-01-08T00:00:01Z"
    source_updated_at = "2042-01-01T00:00:00Z"
    with TemporaryDirectory(prefix="jarvis-goal-nudge-clock-fence-") as temp:
        runtime = make_temp_runtime(Path(temp))
        goal_id = runtime.store.create_goal(GoalRecord(title="CLOCK_EDGE_GOAL"))
        step_id = runtime.store.add_goal_step(goal_id, "CLOCK_EDGE_STEP")
        if step_id is None:
            raise SystemExit("clock-fence fixture could not create its goal step")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE goals SET updated_at = ? WHERE id = ?",
                (source_updated_at, goal_id),
            )
        before_goal = dict(runtime.store.get_goal(goal_id) or {})
        before_steps = [dict(row) for row in runtime.store.list_goal_steps(goal_id)]
        _due_goal_nudge(runtime, "Clock Fence Goal Nudge")
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)

        clock = {"now": just_fresh_at}
        crossed_at_dispatch = Event()
        original_dispatch = scheduler._dispatch_scheduled_note_occurrence

        def current_time() -> str:
            return clock["now"]

        def cross_clock(occurrence_key: str):
            publications = _publication_rows(runtime)
            if (
                len(publications) == 1
                and publications[0]["state"] == "prepared"
                and "No stale active goals" in str(publications[0]["payload"] or "")
            ):
                crossed_at_dispatch.set()
            clock["now"] = newly_stale_at
            return original_dispatch(occurrence_key)

        with mock.patch.object(
            store_module,
            "utc_now",
            side_effect=current_time,
        ), mock.patch.object(
            scheduler,
            "_dispatch_scheduled_note_occurrence",
            side_effect=cross_clock,
        ), mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            return_value=_accepted(8401),
        ) as send_owner:
            output = _run_due(scheduler)

        after_goal = dict(runtime.store.get_goal(goal_id) or {})
        after_steps = [dict(row) for row in runtime.store.list_goal_steps(goal_id)]
        publications = _publication_rows(runtime)
        if len(publications) != 1:
            raise SystemExit(f"clock-fence flow lost its publication: {publications}")
        publication = publications[0]
        daily_path = runtime.vault.root_path / "Daily" / f"{publication['target_date']}.md"
        with runtime.store.connect() as conn:
            delivery_count = int(
                conn.execute("SELECT COUNT(*) FROM scheduled_deliveries").fetchone()[0]
            )
        if (
            not crossed_at_dispatch.is_set()
            or before_goal != after_goal
            or before_steps != after_steps
            or publication["state"] != "failed"
            or publication["error_code"] != "source_manifest_stale"
            or publication["payload"] is not None
            or publication["effect_started_at"] is not None
            or daily_path.exists()
            or delivery_count != 0
            or send_owner.call_count != 0
            or "No stale active goals" not in output
        ):
            raise SystemExit(
                "clock-only fresh-to-stale transition did not invalidate the quiet nudge: "
                f"publication={dict(publication)!r} output={output!r}"
            )


def test_direct_goal_nudge_holds_mutation_until_date_bound_append_completes() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-nudge-direct-lock-") as temp:
        runtime = make_temp_runtime(Path(temp))
        goal_id, _ = _create_stale_goal(
            runtime,
            "DIRECT_LOCK_GOAL",
            "DIRECT_LOCK_ORIGINAL_STEP",
        )

        append_entered = Event()
        allow_append = Event()
        append_completed = Event()
        mutation_attempted = Event()
        mutation_finished = Event()
        mutation_saw_completed_append = Event()
        synchronization_timed_out = Event()
        outputs: Queue = Queue()
        errors: Queue = Queue()
        original_append = runtime.vault.append_daily_for_date

        def held_append(target_date: str, heading: str, body: str):
            append_entered.set()
            if not allow_append.wait(timeout=5):
                synchronization_timed_out.set()
            result = original_append(target_date, heading, body)
            append_completed.set()
            return result

        def build_direct_nudge() -> None:
            try:
                outputs.put(build_goal_nudge(runtime.store, runtime.vault))
            except BaseException as exc:
                errors.put(exc)

        def mutate_during_append() -> None:
            try:
                if not append_entered.wait(timeout=5):
                    raise RuntimeError(
                        "direct goal nudge never entered append_daily_for_date"
                    )
                mutation_attempted.set()
                step_id = runtime.store.add_goal_step(
                    goal_id,
                    "DIRECT_LOCK_CONCURRENT_STEP",
                )
                if step_id is None:
                    raise RuntimeError("direct-lock goal mutation did not commit")
                if append_completed.is_set():
                    mutation_saw_completed_append.set()
            except BaseException as exc:
                errors.put(exc)
            finally:
                mutation_finished.set()

        with mock.patch.object(
            runtime.vault,
            "append_daily_for_date",
            side_effect=held_append,
        ) as append_daily_for_date:
            builder = Thread(target=build_direct_nudge, name="goal-nudge-direct-builder")
            mutator = Thread(
                target=mutate_during_append,
                name="goal-nudge-direct-mutator",
            )
            builder.start()
            mutator.start()
            append_ready = append_entered.wait(timeout=5)
            mutation_waiting = mutation_attempted.wait(timeout=5)
            mutation_finished_early = mutation_finished.wait(timeout=0.2)
            allow_append.set()
            _join_threads((builder, mutator), errors, "direct goal-nudge publication lock")

        daily_path = runtime.vault.root_path / "Daily" / f"{datetime.now():%Y-%m-%d}.md"
        note_text = daily_path.read_text(encoding="utf-8") if daily_path.exists() else ""
        output = outputs.get_nowait() if not outputs.empty() else ""
        if (
            not append_ready
            or not mutation_waiting
            or synchronization_timed_out.is_set()
            or mutation_finished_early
            or not mutation_saw_completed_append.is_set()
            or append_daily_for_date.call_count != 1
            or "DIRECT_LOCK_ORIGINAL_STEP" not in str(output)
            or "DIRECT_LOCK_CONCURRENT_STEP" in str(output)
            or "DIRECT_LOCK_ORIGINAL_STEP" not in note_text
            or len(runtime.store.list_goal_steps(goal_id)) != 2
        ):
            raise SystemExit(
                "direct goal nudge did not hold goal mutation through its date-bound append: "
                f"append_ready={append_ready} mutation_waiting={mutation_waiting} "
                f"finished_early={mutation_finished_early} output={output!r}"
            )


def test_scheduled_note_holds_goal_mutation_through_actual_append() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-nudge-scheduled-effect-lock-") as temp:
        runtime = make_temp_runtime(Path(temp))
        goal_id, _ = _create_stale_goal(
            runtime,
            "SCHEDULED_EFFECT_LOCK_GOAL",
            "SCHEDULED_EFFECT_OLD_STEP",
        )
        _due_goal_nudge(runtime, "Scheduled Effect Lock Goal Nudge")
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)

        append_entered = Event()
        append_completed = Event()
        mutation_attempted = Event()
        mutation_finished = Event()
        mutation_saw_completed_append = Event()
        mutation_escaped_early = Event()
        errors: Queue = Queue()
        original_append = runtime.vault.append_scheduled_daily_once_for_date

        def held_append(*args, **kwargs):
            append_entered.set()
            if not mutation_attempted.wait(timeout=5):
                raise RuntimeError("goal mutation was not attempted during scheduled append")
            if mutation_finished.wait(timeout=0.2):
                mutation_escaped_early.set()
            result = original_append(*args, **kwargs)
            append_completed.set()
            return result

        def mutate_during_append() -> None:
            try:
                if not append_entered.wait(timeout=5):
                    raise RuntimeError("scheduled goal nudge never entered its note append")
                mutation_attempted.set()
                step_id = runtime.store.add_goal_step(
                    goal_id,
                    "SCHEDULED_EFFECT_NEW_STEP",
                )
                if step_id is None:
                    raise RuntimeError("scheduled-effect goal mutation did not commit")
                if append_completed.is_set():
                    mutation_saw_completed_append.set()
            except BaseException as exc:
                errors.put(exc)
            finally:
                mutation_finished.set()

        mutator = Thread(target=mutate_during_append, name="scheduled-effect-mutator")
        mutator.start()
        with mock.patch.object(
            runtime.vault,
            "append_scheduled_daily_once_for_date",
            side_effect=held_append,
        ) as append_note, mock.patch.object(
            scheduler,
            "_dispatch_occurrence",
            return_value=scheduler_module._DeliveryDispatch("prepared", "held for test"),
        ):
            _run_due(scheduler)
        _join_threads((mutator,), errors, "scheduled goal-nudge effect lock")

        publications = _publication_rows(runtime)
        if len(publications) != 1:
            raise SystemExit(f"scheduled-effect flow lost its publication: {publications}")
        publication = publications[0]
        daily_path = runtime.vault.root_path / "Daily" / f"{publication['target_date']}.md"
        note_text = daily_path.read_text(encoding="utf-8") if daily_path.exists() else ""
        if (
            append_note.call_count != 1
            or mutation_escaped_early.is_set()
            or not mutation_saw_completed_append.is_set()
            or publication["state"] != "published"
            or "SCHEDULED_EFFECT_OLD_STEP" not in note_text
            or "SCHEDULED_EFFECT_NEW_STEP" in note_text
            or len(runtime.store.list_goal_steps(goal_id)) != 2
        ):
            raise SystemExit(
                "scheduled note publication did not retain goal-source authority through append: "
                f"publication={dict(publication)!r} note={note_text!r}"
            )


def test_unwritten_retry_revalidates_source_before_later_publication() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-nudge-unwritten-retry-") as temp:
        runtime = make_temp_runtime(Path(temp))
        goal_id, old_step_id = _create_stale_goal(
            runtime,
            "UNWRITTEN_RETRY_GOAL",
            "UNWRITTEN_RETRY_OLD_STEP",
        )
        _due_goal_nudge(runtime, "Unwritten Retry Goal Nudge")
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)

        with mock.patch.object(
            obsidian_module,
            "_replace_text",
            side_effect=OSError("injected failure before note write"),
        ), mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            return_value=_accepted(8601),
        ) as first_send:
            _run_due(scheduler)
        publications = _publication_rows(runtime)
        if len(publications) != 1:
            raise SystemExit(f"unwritten retry lost its publication: {publications}")
        first = publications[0]
        daily_path = runtime.vault.root_path / "Daily" / f"{first['target_date']}.md"
        if (
            first["state"] != "prepared"
            or first["effect_started_at"] is None
            or daily_path.exists()
            or first_send.call_count != 0
        ):
            raise SystemExit(
                "pre-write failure did not preserve an evidence-pending retry: "
                f"{dict(first)!r}"
            )

        completed = runtime.store.complete_goal_step(old_step_id)
        replacement = runtime.store.add_goal_step(goal_id, "UNWRITTEN_RETRY_NEW_STEP")
        if completed is None or replacement is None:
            raise SystemExit("unwritten-retry source mutation did not commit")

        with mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            return_value=_accepted(8602),
        ) as retry_send:
            dispatch = Scheduler(
                runtime.store, runtime.vault, runtime.config
            )._dispatch_scheduled_note_occurrence(str(first["occurrence_key"]))
        current = runtime.store.get_scheduled_note_publication(
            str(first["publication_key"])
        )
        deliveries = runtime.store.list_scheduled_deliveries_for_occurrence(
            str(first["delivery_occurrence_key"] or "")
        )
        if (
            current is None
            or dispatch.state != "failed"
            or current["state"] != "failed"
            or current["error_code"] != "source_manifest_stale"
            or current["payload"] is not None
            or current["effect_started_at"] is not None
            or daily_path.exists()
            or retry_send.call_count != 0
            or any(row["state"] != "failed" or row["payload"] is not None for row in deliveries)
        ):
            raise SystemExit(
                "unwritten failed attempt bypassed later source freshness: "
                f"dispatch={dispatch!r} publication={dict(current) if current else None!r}"
            )


def test_goal_source_fence_covers_process_crash_nested_and_all_mutators() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-source-process-fence-") as temp:
        runtime = make_temp_runtime(Path(temp))
        goal_id = runtime.store.create_goal(GoalRecord(title="PROCESS_FENCE_GOAL"))
        step_id = runtime.store.add_goal_step(goal_id, "PROCESS_FENCE_ORIGINAL_STEP")
        if step_id is None:
            raise SystemExit("process-fence fixture could not create its goal step")

        context = get_context("fork")
        parent_pipe, child_pipe = context.Pipe(duplex=False)
        with runtime.store.goal_source_effect_fence():
            process = context.Process(
                target=_process_add_goal_step,
                args=(str(runtime.store.db_path), goal_id, child_pipe),
            )
            process.start()
            child_pipe.close()
            if not parent_pipe.poll(5) or parent_pipe.recv() != "attempting":
                raise SystemExit("forked goal mutator did not reach the process fence")
            if parent_pipe.poll(0.2):
                raise SystemExit("forked goal mutation escaped the process fence")
        if not parent_pipe.poll(5):
            raise SystemExit("forked goal mutation did not resume after fence release")
        process_result = parent_pipe.recv()
        process.join(timeout=5)
        parent_pipe.close()
        if (
            process.is_alive()
            or process.exitcode != 0
            or not isinstance(process_result, tuple)
            or process_result[0] != "done"
            or type(process_result[1]) is not int
        ):
            raise SystemExit(f"forked goal mutation failed after release: {process_result!r}")

        crash_parent, crash_child = context.Pipe(duplex=False)
        crashing = context.Process(
            target=_process_crash_while_holding_goal_fence,
            args=(str(runtime.store.db_path), crash_child),
        )
        crashing.start()
        crash_child.close()
        if not crash_parent.poll(5) or crash_parent.recv() != "held":
            raise SystemExit("crash-release child never acquired the goal fence")
        crashing.join(timeout=5)
        crash_parent.close()
        if crashing.is_alive() or crashing.exitcode != 0:
            raise SystemExit("crash-release child did not exit while holding the fence")
        if runtime.store.add_goal_step(goal_id, "AFTER_CRASH_RELEASE_STEP") is None:
            raise SystemExit("process death left the goal source fence stranded")

        with runtime.store.goal_source_effect_fence():
            with runtime.store.goal_source_effect_fence():
                nested = runtime.store.set_goal_status_result(goal_id, "paused")
        if not nested.changed:
            raise SystemExit("nested goal source fence was not reentrant")
        runtime.store.set_goal_status_result(goal_id, "active")

        organized = OrganizedNoteEntryRecord(
            kind="goal",
            goal=GoalRecord(title="ORGANIZED_FENCE_GOAL"),
        )
        actions = (
            ("create", lambda: runtime.store.create_goal(GoalRecord(title="CREATE_FENCE_GOAL"))),
            ("add_step", lambda: runtime.store.add_goal_step(goal_id, "THREAD_ADD_STEP")),
            ("complete_step", lambda: runtime.store.complete_goal_step_result(step_id)),
            ("set_status", lambda: runtime.store.set_goal_status_result(goal_id, "paused")),
            (
                "organized_goal",
                lambda: runtime.store.reserve_organized_note_batch(
                    "organize-note:v1:" + "a" * 64,
                    (organized,),
                ),
            ),
        )
        for label, action in actions:
            attempted = Event()
            finished = Event()
            errors: Queue = Queue()

            def run_action(action=action) -> None:
                attempted.set()
                try:
                    action()
                except BaseException as exc:
                    errors.put(exc)
                finally:
                    finished.set()

            with runtime.store.goal_source_effect_fence():
                worker = Thread(target=run_action, name=f"goal-source-{label}")
                worker.start()
                if not attempted.wait(timeout=5):
                    raise SystemExit(f"{label} mutator did not start")
                if finished.wait(timeout=0.1):
                    raise SystemExit(f"{label} mutator escaped the source fence")
            _join_threads((worker,), errors, f"goal source {label} fence")


def test_scheduled_and_ordinary_daily_writes_share_one_path_lock() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-nudge-shared-note-lock-") as temp:
        runtime = make_temp_runtime(Path(temp))
        target_date = "2042-06-01"
        runtime.vault.append_daily_for_date(target_date, "Initial", "INITIAL_BLOCK")

        scheduled_before_replace = Event()
        ordinary_attempted = Event()
        ordinary_opened = Event()
        ordinary_finished = Event()
        ordinary_escaped = Event()
        errors: Queue = Queue()
        original_replace = obsidian_module._replace_text
        original_append = obsidian_module._append_text_under_inode_lock

        def held_replace(*args, **kwargs):
            if current_thread().name == "scheduled-note-writer":
                scheduled_before_replace.set()
                if not ordinary_attempted.wait(timeout=5):
                    raise RuntimeError("ordinary append was not attempted during replacement")
                if ordinary_opened.wait(timeout=0.2):
                    ordinary_escaped.set()
            return original_replace(*args, **kwargs)

        def observed_append(*args, **kwargs):
            if current_thread().name == "ordinary-note-writer":
                ordinary_opened.set()
            return original_append(*args, **kwargs)

        def scheduled_write() -> None:
            try:
                runtime.vault.append_scheduled_daily_once_for_date(
                    target_date,
                    "b" * 64,
                    "Jarvis Goal Nudge",
                    "SCHEDULED_BLOCK",
                )
            except BaseException as exc:
                errors.put(exc)

        def ordinary_write() -> None:
            try:
                if not scheduled_before_replace.wait(timeout=5):
                    raise RuntimeError("scheduled writer never reached replacement")
                ordinary_attempted.set()
                runtime.vault.append_daily_for_date(
                    target_date,
                    "Concurrent",
                    "ORDINARY_BLOCK",
                )
            except BaseException as exc:
                errors.put(exc)
            finally:
                ordinary_finished.set()

        with mock.patch.object(
            obsidian_module,
            "_replace_text",
            side_effect=held_replace,
        ), mock.patch.object(
            obsidian_module,
            "_append_text_under_inode_lock",
            side_effect=observed_append,
        ):
            scheduled = Thread(target=scheduled_write, name="scheduled-note-writer")
            ordinary = Thread(target=ordinary_write, name="ordinary-note-writer")
            scheduled.start()
            ordinary.start()
            _join_threads((scheduled, ordinary), errors, "shared daily-note path lock")

        daily_path = runtime.vault.root_path / "Daily" / f"{target_date}.md"
        persisted = daily_path.read_text(encoding="utf-8")
        if (
            ordinary_escaped.is_set()
            or not ordinary_finished.is_set()
            or not ordinary_opened.is_set()
            or "SCHEDULED_BLOCK" not in persisted
            or "ORDINARY_BLOCK" not in persisted
        ):
            raise SystemExit(
                "scheduled replacement and ordinary append did not share one path lock: "
                f"persisted={persisted!r}"
            )

        poison_date = "2042-06-02"
        runtime.vault.append_daily_for_date(
            poison_date,
            "jarvis-scheduled-note-start:heading-poison",
            "jarvis-scheduled-note-end:body-poison",
        )
        _, appended, persisted_body = runtime.vault.append_scheduled_daily_once_for_date(
            poison_date,
            "c" * 64,
            "Jarvis Goal Nudge",
            "SAFE_SCHEDULED_BLOCK",
        )
        poison_path = runtime.vault.root_path / "Daily" / f"{poison_date}.md"
        poison_text = poison_path.read_text(encoding="utf-8")
        if (
            not appended
            or persisted_body != "SAFE_SCHEDULED_BLOCK"
            or "jarvis-scheduled-note-start:heading-poison" in poison_text
            or "jarvis-scheduled-note-end:body-poison" in poison_text
            or "jarvis&#45;scheduled&#45;note-start:heading-poison" not in poison_text
            or "SAFE_SCHEDULED_BLOCK" not in poison_text
        ):
            raise SystemExit(
                "ordinary daily content poisoned the scheduled-note marker namespace: "
                f"{poison_text!r}"
            )


def test_scheduled_goal_nudge_plan_refuses_missing_source_manifest() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-nudge-manifest-required-") as temp:
        runtime = make_temp_runtime(Path(temp))
        job_id = _due_goal_nudge(runtime, "Manifest Required Goal Nudge")
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        claimed = runtime.store.claim_job(
            job_id,
            "manifest-required-claim",
            300,
            due_before=iso(datetime.now()),
        )
        if claimed is None:
            raise SystemExit("manifest-required fixture could not claim its job")
        missing = scheduler._scheduled_note_plan(
            claimed,
            "MANIFEST_REQUIRED_BODY",
            delivery_plan=None,
            allow_disabled=False,
        )
        if missing is None:
            raise SystemExit("manifest-required fixture did not build its note plan")
        try:
            runtime.store._validate_scheduled_note_publication_plan(
                missing.as_store_plan()
            )
        except ValueError:
            pass
        else:
            raise SystemExit("scheduled goal nudge accepted a missing source manifest")

        snapshot = runtime.store.read_goal_nudge_snapshot()
        current = scheduler._scheduled_note_plan(
            claimed,
            "MANIFEST_REQUIRED_BODY",
            delivery_plan=None,
            allow_disabled=False,
            source_manifest_kind=snapshot.source_manifest_kind,
            source_manifest_digest=snapshot.source_manifest_digest,
        )
        if current is None:
            raise SystemExit("manifest-required fixture lost its valid note plan")
        normalized = runtime.store._validate_scheduled_note_publication_plan(
            current.as_store_plan()
        )
        if normalized["source_manifest_digest"] != snapshot.source_manifest_digest:
            raise SystemExit("valid goal-nudge source manifest did not survive validation")


def test_published_note_lease_completes_exact_delivery_after_source_change() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-nudge-occurrence-lease-") as temp:
        runtime = make_temp_runtime(Path(temp))
        goal_id, old_step_id = _create_stale_goal(
            runtime,
            "OCCURRENCE_LEASE_GOAL",
            "OCCURRENCE_LEASE_OLD_STEP",
        )
        _due_goal_nudge(runtime, "Occurrence Lease Goal Nudge")
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)

        def three_chunks(text: str, *, limit: int = 3900) -> list[str]:
            del limit
            first = max(1, len(text) // 3)
            second = max(first + 1, (len(text) * 2) // 3)
            return [text[:first], text[first:second], text[second:]]

        with mock.patch.object(
            telegram_control,
            "_text_chunks",
            side_effect=three_chunks,
        ), mock.patch.object(
            scheduler,
            "_dispatch_occurrence",
            return_value=scheduler_module._DeliveryDispatch("prepared", "held for retry"),
        ):
            _run_due(scheduler)
        publication = _publication_rows(runtime)[0]
        occurrence_key = str(publication["delivery_occurrence_key"] or "")
        deliveries = runtime.store.list_scheduled_deliveries_for_occurrence(occurrence_key)
        if publication["state"] != "published" or len(deliveries) != 3:
            raise SystemExit("occurrence-lease fixture did not publish and prepare three chunks")

        sent: list[str] = []

        def send_owner(text: str) -> dict[str, object]:
            sent.append(text)
            return _accepted(8700 + len(sent))

        with mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=send_owner,
        ):
            first_dispatch = scheduler._dispatch_delivery_row(deliveries[0])
        if first_dispatch.state != "accepted":
            raise SystemExit("occurrence-lease fixture did not accept its first chunk")

        completed = runtime.store.complete_goal_step(old_step_id)
        replacement = runtime.store.add_goal_step(goal_id, "OCCURRENCE_LEASE_NEW_STEP")
        if completed is None or replacement is None:
            raise SystemExit("occurrence-lease source mutation did not commit")

        with mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=send_owner,
        ):
            retry = scheduler._dispatch_occurrence(occurrence_key)
        current = runtime.store.list_scheduled_deliveries_for_occurrence(occurrence_key)
        sent_text = "".join(sent)
        if (
            retry.state != "accepted"
            or len(sent) != 3
            or any(row["state"] != "accepted" for row in current)
            or "OCCURRENCE_LEASE_OLD_STEP" not in sent_text
            or "OCCURRENCE_LEASE_NEW_STEP" in sent_text
        ):
            raise SystemExit(
                "published Goal Nudge occurrence did not complete its exact accepted prefix: "
                f"retry={retry!r} rows={[dict(row) for row in current]!r}"
            )


def test_legacy_goal_nudge_delivery_without_linked_manifest_fails_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-nudge-legacy-delivery-") as temp:
        runtime = make_temp_runtime(Path(temp))
        job_id = runtime.store.upsert_job(
            "Legacy Goal Nudge Delivery",
            1440,
            "goal_nudge",
            iso(datetime.now() + timedelta(hours=1)),
        )
        row = next(
            item for item in runtime.store.list_jobs() if int(item["id"]) == job_id
        )
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        plan = scheduler._delivery_plan(
            row,
            "LEGACY_UNBOUND_GOAL_NUDGE_PAYLOAD",
            datetime.now(),
        )
        runtime.store.prepare_scheduled_delivery_occurrence(
            job_id,
            str(row["name"]),
            str(row["job_type"]),
            plan.occurrence_key,
            list(plan.chunks),
        )
        with mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            return_value=_accepted(8501),
        ) as send_owner:
            dispatch = scheduler._dispatch_occurrence(plan.occurrence_key)
        deliveries = runtime.store.list_scheduled_deliveries_for_occurrence(
            plan.occurrence_key
        )
        if (
            dispatch.state != "failed"
            or send_owner.call_count != 0
            or len(deliveries) != 1
            or deliveries[0]["state"] != "failed"
            or deliveries[0]["payload"] is not None
            or deliveries[0]["error_code"] != "source_manifest_stale"
        ):
            raise SystemExit(
                "legacy unbound goal-nudge delivery bypassed the source-manifest fence: "
                f"dispatch={dispatch!r} rows={[dict(item) for item in deliveries]!r}"
            )


def main() -> None:
    test_goal_nudge_coalesces_only_normalized_duplicate_titles()
    test_build_goal_nudge_uses_one_sqlite_snapshot_during_step_mutation()
    test_stale_source_fails_before_note_and_linked_telegram_effects()
    test_unchanged_source_publishes_then_sends_once_with_opaque_manifest()
    test_goal_mutation_waits_until_all_telegram_chunks_complete()
    test_wall_clock_staleness_transition_invalidates_quiet_scheduled_nudge()
    test_direct_goal_nudge_holds_mutation_until_date_bound_append_completes()
    test_scheduled_note_holds_goal_mutation_through_actual_append()
    test_unwritten_retry_revalidates_source_before_later_publication()
    test_goal_source_fence_covers_process_crash_nested_and_all_mutators()
    test_scheduled_and_ordinary_daily_writes_share_one_path_lock()
    test_scheduled_goal_nudge_plan_refuses_missing_source_manifest()
    test_published_note_lease_completes_exact_delivery_after_source_change()
    test_legacy_goal_nudge_delivery_without_linked_manifest_fails_closed()
    print("PASS: goal nudge source snapshot and freshness fencing")


if __name__ == "__main__":
    main()
