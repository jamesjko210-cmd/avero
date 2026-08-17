from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Barrier, Event, Thread
from types import SimpleNamespace
from unittest import mock

from jarvis_v2.automations import scheduler as scheduler_module
from jarvis_v2.automations.scheduler import (
    MORNING_BRIEF_ENV,
    Scheduler,
    _payload_sha256,
    iso,
)
from jarvis_v2.memory import obsidian as obsidian_module
from jarvis_v2.scripts.test_runtime import make_temp_runtime


def _job(runtime, name: str, job_type: str = "daily_brief", *, due: bool = False):
    next_run = datetime.now() + (-timedelta(minutes=5) if due else timedelta(hours=1))
    job_id = runtime.store.upsert_job(name, 1440, job_type, iso(next_run))
    row = next(row for row in runtime.store.list_jobs() if int(row["id"]) == job_id)
    return job_id, row


def _job_row(runtime, job_id: int):
    return next(row for row in runtime.store.list_jobs() if int(row["id"]) == job_id)


def _history(runtime, job_id: int) -> list[dict[str, str]]:
    metadata = json.loads(_job_row(runtime, job_id)["metadata"] or "{}")
    history = metadata.get("run_history")
    return history if isinstance(history, list) else []


def _claim(runtime, scheduler: Scheduler, row, token: str, *, now: datetime | None = None):
    claimed = runtime.store.claim_job(
        int(row["id"]),
        token,
        scheduler.job_lease_seconds,
        now=now,
    )
    if claimed is None:
        raise SystemExit(f"scheduled-note fixture could not claim {row['name']}")
    return claimed


def _finalize_note(
    runtime,
    scheduler: Scheduler,
    claimed,
    lease_token: str,
    payload: str,
    *,
    completed_at: datetime | None = None,
    allow_disabled: bool = False,
    linked_telegram: bool = False,
):
    completed = completed_at or datetime.now()
    delivery_plan = (
        scheduler._delivery_plan(claimed, f"Scheduled report\n\n{payload}", completed)
        if linked_telegram
        else None
    )
    note_plan = scheduler._scheduled_note_plan(
        claimed,
        payload,
        delivery_plan=delivery_plan,
        allow_disabled=allow_disabled,
    )
    if note_plan is None:
        raise SystemExit("scheduled-note fixture did not produce a note plan")
    finalized = scheduler._finalize_claim_with_delivery(
        claimed,
        lease_token,
        completed,
        iso(completed + timedelta(days=1)),
        delivery_plan,
        note_plan,
        payload,
    )
    if not finalized:
        raise SystemExit("scheduled-note fixture did not finalize atomically")
    row = runtime.store.get_scheduled_note_publication(note_plan.publication_key)
    if row is None:
        raise SystemExit("scheduled-note finalization lost its publication row")
    return note_plan, delivery_plan, row


def _daily_path(runtime, target_date: str) -> Path:
    return runtime.vault.root_path / "Daily" / f"{target_date}.md"


def _marker_count(path: Path, publication_key: str) -> int:
    text = path.read_text(encoding="utf-8")
    return text.count(f"jarvis-scheduled-note-start:v1:{publication_key}:")


def _run_due(scheduler: Scheduler) -> str:
    with mock.patch.dict(os.environ, {MORNING_BRIEF_ENV: ""}, clear=False):
        return scheduler.run_due_jobs()


def _accepted(message_id: int = 7101) -> dict[str, object]:
    return {"ok": True, "result": {"message_id": message_id}}


def test_finalization_precedes_drain_payload_is_immutable_and_history_converges() -> None:
    original = "ORIGINAL_NOTE_PAYLOAD_CUSTODY"
    changed = "CHANGED_RETRY_SOURCE_MUST_NOT_WIN"
    with TemporaryDirectory(prefix="jarvis-note-finalize-first-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        job_id, row = _job(runtime, "Note Finalize First")
        claimed = _claim(runtime, scheduler, row, "note-finalize-first")
        completed = datetime.now()
        note_plan, _, publication = _finalize_note(
            runtime,
            scheduler,
            claimed,
            "note-finalize-first",
            original,
            completed_at=completed,
        )

        target = _daily_path(runtime, note_plan.target_date)
        history = _history(runtime, job_id)
        if (
            publication["state"] != "prepared"
            or publication["payload"] != original
            or publication["payload_sha256"] != _payload_sha256(original)
            or target.exists()
            or len(history) != 1
            or history[0].get("status") != "pending"
        ):
            raise SystemExit(
                "job finalization did not persist the immutable note before drain "
                f"or record pending history: {dict(publication)!r} {history!r}"
            )

        changed_plan = scheduler._scheduled_note_plan(
            claimed,
            changed,
            delivery_plan=None,
            allow_disabled=False,
        )
        if changed_plan is None or changed_plan.publication_key != note_plan.publication_key:
            raise SystemExit("changed retry source did not retain the occurrence publication identity")
        normalized = runtime.store._validate_scheduled_note_publication_plan(
            changed_plan.as_store_plan()
        )
        try:
            with runtime.store.connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                runtime.store._prepare_scheduled_note_publication_on_connection(
                    conn,
                    normalized,
                    now="2042-01-01T00:00:00Z",
                )
        except ValueError:
            pass
        else:
            raise SystemExit("changed retry source replaced an already persisted note payload")
        persisted = runtime.store.get_scheduled_note_publication(note_plan.publication_key)
        if persisted is None or persisted["payload"] != original or changed in json.dumps(dict(persisted)):
            raise SystemExit(f"immutable publication payload changed during retry: {persisted}")

        dispatch = scheduler._dispatch_scheduled_note_occurrence(note_plan.occurrence_key)
        if dispatch.state != "published" or not target.exists():
            raise SystemExit(f"persisted scheduled note did not drain: {dispatch!r}")
        text = target.read_text(encoding="utf-8")
        converged = _history(runtime, job_id)
        if (
            original not in text
            or changed in text
            or len(converged) != 1
            or converged[0].get("status") != "ok"
            or converged[0].get("occurrence_key") != history[0].get("occurrence_key")
        ):
            raise SystemExit(
                f"note drain changed payload or failed exact run-history convergence: {converged!r}"
            )


def test_crash_after_file_publish_stale_recovery_coalesces_exact_block() -> None:
    payload = "CRASH_RECOVERY_EXACT_BODY"
    with TemporaryDirectory(prefix="jarvis-note-crash-recovery-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        _, row = _job(runtime, "Note Crash Recovery")
        claimed_job = _claim(runtime, scheduler, row, "note-crash-job")
        note_plan, _, _ = _finalize_note(
            runtime,
            scheduler,
            claimed_job,
            "note-crash-job",
            payload,
        )

        attempt = "note-crash-after-file"
        claimed_note = runtime.store.claim_scheduled_note_publication(
            note_plan.publication_key,
            attempt,
        )
        if claimed_note is None:
            raise SystemExit("crash-recovery fixture could not claim its note")
        path, appended, persisted = runtime.vault.append_scheduled_daily_once_for_date(
            str(claimed_note["target_date"]),
            note_plan.publication_key,
            str(claimed_note["title"]),
            str(claimed_note["payload"]),
            effect_authority=lambda: (
                None
                if runtime.store.begin_scheduled_note_publication_effect(
                    note_plan.publication_key,
                    attempt,
                )
                else (_ for _ in ()).throw(RuntimeError("lost crash fixture claim"))
            ),
        )
        if not appended or persisted != payload or _marker_count(path, note_plan.publication_key) != 1:
            raise SystemExit("pre-crash file publication did not write one exact owned block")

        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_note_publications SET claimed_at = ? WHERE publication_key = ?",
                ("2000-01-01T00:00:00Z", note_plan.publication_key),
            )
        recovered = runtime.store.recover_stale_scheduled_note_publications(
            "2100-01-01T00:00:00Z"
        )
        if recovered.get("claimed_to_prepared") != 1:
            raise SystemExit(f"stale note claim did not recover to prepared: {recovered!r}")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_note_publications SET payload = ? WHERE publication_key = ?",
                ("CORRUPTED_AFTER_VISIBLE_WRITE", note_plan.publication_key),
            )
        dispatch = scheduler._dispatch_scheduled_note_occurrence(note_plan.occurrence_key)
        terminal = runtime.store.get_scheduled_note_publication(note_plan.publication_key)
        text = path.read_text(encoding="utf-8")
        if (
            dispatch.state != "published"
            or terminal is None
            or terminal["state"] != "published"
            or _marker_count(path, note_plan.publication_key) != 1
            or text.count(payload) != 1
        ):
            raise SystemExit(
                f"stale recovery did not coalesce the exact published block: {dispatch!r} {terminal}"
            )


def test_destination_size_failure_stays_pre_effect_and_terminalizes() -> None:
    with TemporaryDirectory(prefix="jarvis-note-size-before-effect-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        _, row = _job(runtime, "Note Size Before Effect")
        claimed_job = _claim(runtime, scheduler, row, "note-size-job")
        note_plan, _, _ = _finalize_note(
            runtime,
            scheduler,
            claimed_job,
            "note-size-job",
            "DESTINATION_SIZE_FAILURE_MUST_TERMINALIZE",
        )
        observed = []
        with mock.patch.object(obsidian_module, "MAX_SCHEDULED_NOTE_FILE_BYTES", 32):
            for _ in range(7):
                observed.append(
                    scheduler._dispatch_scheduled_note_occurrence(note_plan.occurrence_key)
                )
        terminal = runtime.store.get_scheduled_note_publication(note_plan.publication_key)
        if (
            terminal is None
            or terminal["state"] != "failed"
            or int(terminal["attempt_count"]) != 5
            or terminal["effect_started_at"] is not None
            or _daily_path(runtime, note_plan.target_date).exists()
            or observed[-1].state != "failed"
        ):
            raise SystemExit(
                "deterministic destination-size failure crossed the effect boundary or retried "
                f"without bound: {dict(terminal) if terminal is not None else None!r} {observed!r}"
            )


def test_wrong_valid_marker_stays_pre_effect_and_terminalizes() -> None:
    expected = "EXPECTED_EXACT_PUBLICATION_BODY"
    wrong = "WRONG_BUT_SELF_CONSISTENT_MARKER_BODY"
    with TemporaryDirectory(prefix="jarvis-note-wrong-marker-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        _, row = _job(runtime, "Note Wrong Marker")
        claimed_job = _claim(runtime, scheduler, row, "note-wrong-marker-job")
        note_plan, _, _ = _finalize_note(
            runtime,
            scheduler,
            claimed_job,
            "note-wrong-marker-job",
            expected,
        )
        wrong_path, appended, persisted = runtime.vault.append_scheduled_daily_once_for_date(
            note_plan.target_date,
            note_plan.publication_key,
            note_plan.title,
            wrong,
        )
        if not appended or persisted != wrong:
            raise SystemExit("wrong-marker fixture could not create valid conflicting evidence")
        observed = [
            scheduler._dispatch_scheduled_note_occurrence(note_plan.occurrence_key)
            for _ in range(7)
        ]
        terminal = runtime.store.get_scheduled_note_publication(note_plan.publication_key)
        text = wrong_path.read_text(encoding="utf-8")
        if (
            terminal is None
            or terminal["state"] != "failed"
            or int(terminal["attempt_count"]) != 5
            or terminal["effect_started_at"] is not None
            or expected in text
            or text.count(wrong) != 1
            or observed[-1].state != "failed"
        ):
            raise SystemExit(
                "valid conflicting marker crossed the effect boundary or retried without bound: "
                f"{dict(terminal) if terminal is not None else None!r} {observed!r}"
            )


def test_stale_replaced_note_claim_writes_nothing() -> None:
    with TemporaryDirectory(prefix="jarvis-note-stale-fence-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        _, row = _job(runtime, "Note Stale Fence")
        claimed_job = _claim(runtime, scheduler, row, "note-stale-job")
        note_plan, _, _ = _finalize_note(
            runtime,
            scheduler,
            claimed_job,
            "note-stale-job",
            "STALE_OWNER_MUST_NOT_WRITE",
        )
        stale_token = "stale-note-owner"
        stale = runtime.store.claim_scheduled_note_publication(
            note_plan.publication_key,
            stale_token,
        )
        if stale is None:
            raise SystemExit("stale-owner fixture could not claim its note")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_note_publications SET claimed_at = ? WHERE publication_key = ?",
                ("2000-01-01T00:00:00Z", note_plan.publication_key),
            )
        runtime.store.recover_stale_scheduled_note_publications("2100-01-01T00:00:00Z")
        replacement_token = "replacement-note-owner"
        replacement = runtime.store.claim_scheduled_note_publication(
            note_plan.publication_key,
            replacement_token,
        )
        if replacement is None:
            raise SystemExit("replacement note owner could not claim recovered work")

        target = _daily_path(runtime, note_plan.target_date)
        try:
            runtime.vault.append_scheduled_daily_once_for_date(
                str(stale["target_date"]),
                note_plan.publication_key,
                str(stale["title"]),
                str(stale["payload"]),
                effect_authority=lambda: (
                    None
                    if runtime.store.scheduled_note_publication_claim_is_live(
                        note_plan.publication_key,
                        stale_token,
                    )
                    else (_ for _ in ()).throw(RuntimeError("stale note claim rejected"))
                ),
            )
        except RuntimeError:
            pass
        else:
            raise SystemExit("stale/replaced note owner retained file-write authority")
        if target.exists():
            raise SystemExit("stale/replaced note claim wrote a destination file")
        if not runtime.store.finish_scheduled_note_publication(
            note_plan.publication_key,
            replacement_token,
            "failed",
            error_code="fixture_cleanup",
        ):
            raise SystemExit("replacement note claim could not be terminalized")


def test_delete_after_effect_boundary_preserves_truthful_publication() -> None:
    with TemporaryDirectory(prefix="jarvis-note-delete-effect-race-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        _, row = _job(runtime, "Note Delete Effect Race")
        claimed_job = _claim(runtime, scheduler, row, "note-delete-job")
        note_plan, _, _ = _finalize_note(
            runtime,
            scheduler,
            claimed_job,
            "note-delete-job",
            "DELETE_AFTER_EFFECT_MUST_STAY_TRUTHFUL",
        )
        original_publish = runtime.vault.append_scheduled_daily_once_for_date
        boundary_crossed = Event()
        release_publish = Event()

        def delayed_publish(*args, effect_authority=None, **kwargs):
            if effect_authority is None:
                raise AssertionError("delete-race publication lacked effect authority")
            effect_authority()
            boundary_crossed.set()
            if not release_publish.wait(timeout=5):
                raise AssertionError("delete-race publication was not released")
            return original_publish(*args, effect_authority=None, **kwargs)

        result: Queue[object] = Queue()

        def dispatch() -> None:
            try:
                result.put(
                    scheduler._dispatch_scheduled_note_occurrence(note_plan.occurrence_key)
                )
            except BaseException as exc:
                result.put(exc)

        with mock.patch.object(
            runtime.vault,
            "append_scheduled_daily_once_for_date",
            side_effect=delayed_publish,
        ):
            thread = Thread(target=dispatch)
            thread.start()
            try:
                if not boundary_crossed.wait(timeout=5):
                    raise SystemExit("delete race did not reach the fenced effect boundary")
                if not runtime.store.delete_job("Note Delete Effect Race"):
                    raise SystemExit("delete race could not delete the source job")
            finally:
                release_publish.set()
                thread.join(timeout=10)
        if thread.is_alive():
            raise SystemExit("delete race left its publisher thread running")
        dispatch_result = result.get_nowait()
        terminal = runtime.store.get_scheduled_note_publication(note_plan.publication_key)
        target = _daily_path(runtime, note_plan.target_date)
        if (
            isinstance(dispatch_result, BaseException)
            or getattr(dispatch_result, "state", "") != "published"
            or terminal is None
            or terminal["state"] != "published"
            or terminal["job_id"] is not None
            or not target.exists()
            or _marker_count(target, note_plan.publication_key) != 1
        ):
            raise SystemExit(
                "delete after the fenced effect boundary produced contradictory durable truth: "
                f"{dispatch_result!r} {dict(terminal) if terminal is not None else None!r}"
            )

        _, crash_row = _job(runtime, "Note Delete Crash Recovery")
        crash_claim = _claim(runtime, scheduler, crash_row, "note-delete-crash-job")
        crash_plan, _, _ = _finalize_note(
            runtime,
            scheduler,
            crash_claim,
            "note-delete-crash-job",
            "DELETE_CRASH_RECOVERY_EXACT_BODY",
        )
        crash_token = "note-delete-crash-effect"
        crash_publication = runtime.store.claim_scheduled_note_publication(
            crash_plan.publication_key,
            crash_token,
        )
        if crash_publication is None or not runtime.store.begin_scheduled_note_publication_effect(
            crash_plan.publication_key,
            crash_token,
        ):
            raise SystemExit("delete-crash fixture could not enter its effect phase")
        if not runtime.store.delete_job("Note Delete Crash Recovery"):
            raise SystemExit("delete-crash fixture could not delete its source job")
        crash_path, _, _ = runtime.vault.append_scheduled_daily_once_for_date(
            crash_plan.target_date,
            crash_plan.publication_key,
            crash_plan.title,
            crash_plan.payload,
        )
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_note_publications SET claimed_at = ? WHERE publication_key = ?",
                ("2000-01-01T00:00:00Z", crash_plan.publication_key),
            )
        recovered = runtime.store.recover_stale_scheduled_note_publications(
            "2100-01-01T00:00:00Z"
        )
        recovered_dispatch = scheduler._dispatch_scheduled_note_occurrence(
            crash_plan.occurrence_key
        )
        recovered_row = runtime.store.get_scheduled_note_publication(
            crash_plan.publication_key
        )
        if (
            recovered.get("claimed_to_prepared") != 1
            or recovered_dispatch.state != "published"
            or recovered_row is None
            or recovered_row["state"] != "published"
            or _marker_count(crash_path, crash_plan.publication_key) != 1
        ):
            raise SystemExit("deleted in-flight publication did not recover through exact marker replay")

        _, stale_row = _job(runtime, "Note Delete Stale Before Write")
        stale_job_claim = _claim(runtime, scheduler, stale_row, "note-delete-stale-job")
        stale_plan, _, _ = _finalize_note(
            runtime,
            scheduler,
            stale_job_claim,
            "note-delete-stale-job",
            "STALE_RECOVERY_BEFORE_WRITE_MUST_REPLAY",
        )
        stale_token = "note-delete-stale-effect"
        stale_publication = runtime.store.claim_scheduled_note_publication(
            stale_plan.publication_key,
            stale_token,
        )
        if stale_publication is None or not runtime.store.begin_scheduled_note_publication_effect(
            stale_plan.publication_key,
            stale_token,
        ):
            raise SystemExit("stale-before-write fixture could not enter its effect phase")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_note_publications SET claimed_at = ? WHERE publication_key = ?",
                ("2000-01-01T00:00:00Z", stale_plan.publication_key),
            )
        stale_recovered = runtime.store.recover_stale_scheduled_note_publications(
            "2100-01-01T00:00:00Z"
        )
        if not runtime.store.delete_job("Note Delete Stale Before Write"):
            raise SystemExit("stale-before-write fixture could not delete its source job")
        committed = runtime.store.get_scheduled_note_publication(stale_plan.publication_key)
        if (
            stale_recovered.get("claimed_to_prepared") != 1
            or committed is None
            or committed["state"] != "prepared"
            or committed["effect_started_at"] is None
            or int(committed["allow_disabled"]) != 1
            or committed["job_id"] is not None
        ):
            raise SystemExit(
                "stale recovery or deletion discarded irreversible publication custody: "
                f"{stale_recovered!r} {dict(committed) if committed is not None else None!r}"
            )
        stale_path, _, _ = runtime.vault.append_scheduled_daily_once_for_date(
            stale_plan.target_date,
            stale_plan.publication_key,
            stale_plan.title,
            stale_plan.payload,
        )
        if runtime.store.finish_scheduled_note_publication(
            stale_plan.publication_key,
            stale_token,
            "published",
            path_display=f"Daily/{stale_plan.target_date}.md",
            content_sha256=_payload_sha256(stale_plan.payload),
        ):
            raise SystemExit("recovered stale owner retained durable finalization authority")
        replacement_dispatch = scheduler._dispatch_scheduled_note_occurrence(
            stale_plan.occurrence_key
        )
        replacement_row = runtime.store.get_scheduled_note_publication(
            stale_plan.publication_key
        )
        if (
            replacement_dispatch.state != "published"
            or replacement_row is None
            or replacement_row["state"] != "published"
            or _marker_count(stale_path, stale_plan.publication_key) != 1
        ):
            raise SystemExit(
                "stale recovery before write did not converge through marker replay: "
                f"{replacement_dispatch!r} {replacement_row!r}"
            )


def test_concurrent_drainers_publish_one_block() -> None:
    with TemporaryDirectory(prefix="jarvis-note-concurrent-drain-") as temp:
        runtime = make_temp_runtime(Path(temp))
        first = Scheduler(runtime.store, runtime.vault, runtime.config)
        second = Scheduler(runtime.store, runtime.vault, runtime.config)
        _, row = _job(runtime, "Note Concurrent Drain")
        claimed_job = _claim(runtime, first, row, "note-concurrent-job")
        note_plan, _, _ = _finalize_note(
            runtime,
            first,
            claimed_job,
            "note-concurrent-job",
            "CONCURRENT_DRAIN_EXACTLY_ONCE",
        )
        start = Barrier(3)
        results: Queue[object] = Queue()

        def drain(scheduler: Scheduler) -> None:
            try:
                start.wait(timeout=5)
                results.put(
                    scheduler._dispatch_scheduled_note_occurrence(note_plan.occurrence_key)
                )
            except BaseException as exc:
                results.put(exc)

        threads = [Thread(target=drain, args=(scheduler,)) for scheduler in (first, second)]
        for thread in threads:
            thread.start()
        start.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=10)
        if any(thread.is_alive() for thread in threads):
            raise SystemExit("concurrent note drain left a thread running")
        observed = [results.get_nowait() for _ in range(results.qsize())]
        if len(observed) != 2 or any(isinstance(item, BaseException) for item in observed):
            raise SystemExit(f"concurrent note drain raised unexpectedly: {observed!r}")
        target = _daily_path(runtime, note_plan.target_date)
        terminal = runtime.store.get_scheduled_note_publication(note_plan.publication_key)
        if (
            terminal is None
            or terminal["state"] != "published"
            or not target.exists()
            or _marker_count(target, note_plan.publication_key) != 1
            or target.read_text(encoding="utf-8").count("CONCURRENT_DRAIN_EXACTLY_ONCE") != 1
        ):
            raise SystemExit(f"concurrent drain did not converge to one block: {observed!r}")


def test_paused_due_job_holds_then_resume_drains_and_manual_disabled_run_publishes() -> None:
    held_payload = "PAUSED_DUE_PUBLICATION_HELD"
    manual_payload = "DISABLED_RUN_JOB_NOW_PUBLICATION_ALLOWED"
    with TemporaryDirectory(prefix="jarvis-note-pause-resume-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        job_id, _ = _job(runtime, "Paused Due Note", "recent_file_digest", due=True)
        original_dispatch = scheduler._dispatch_scheduled_note_occurrence

        def pause_before_note_claim(occurrence_key: str):
            scheduler.pause_job("Paused Due Note")
            return original_dispatch(occurrence_key)

        sent: list[str] = []
        with mock.patch.object(
            scheduler,
            "_run_owned_job_type",
            return_value=held_payload,
        ), mock.patch.object(
            scheduler,
            "_dispatch_scheduled_note_occurrence",
            side_effect=pause_before_note_claim,
        ), mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or _accepted(),
        ):
            first_tick = _run_due(scheduler)

        with runtime.store.connect() as conn:
            note = conn.execute(
                "SELECT * FROM scheduled_note_publications WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if note is None:
            raise SystemExit(f"paused due run did not persist its publication: {first_tick!r}")
        target = _daily_path(runtime, str(note["target_date"]))
        deliveries = runtime.store.list_scheduled_deliveries_for_occurrence(
            str(note["delivery_occurrence_key"])
        )
        if (
            note["state"] != "prepared"
            or target.exists()
            or sent
            or not deliveries
            or any(row["state"] != "prepared" for row in deliveries)
        ):
            raise SystemExit(
                "paused due job did not hold both note publication and linked Telegram "
                f"delivery: {dict(note)!r} {deliveries!r} {first_tick!r}"
            )

        scheduler.resume_job("Paused Due Note")
        with mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or _accepted(7102),
        ):
            resumed_tick = _run_due(scheduler)
        resumed_note = runtime.store.get_scheduled_note_publication(str(note["publication_key"]))
        resumed_deliveries = runtime.store.list_scheduled_deliveries_for_occurrence(
            str(note["delivery_occurrence_key"])
        )
        if (
            resumed_note is None
            or resumed_note["state"] != "published"
            or not target.exists()
            or held_payload not in target.read_text(encoding="utf-8")
            or len(sent) != 1
            or any(row["state"] != "accepted" for row in resumed_deliveries)
        ):
            raise SystemExit(
                f"resume did not drain held publication before Telegram: {resumed_tick!r}"
            )

        manual_id, _ = _job(runtime, "Manual Disabled Note")
        scheduler.pause_job("Manual Disabled Note")
        with mock.patch.object(
            scheduler,
            "_run_owned_job_type",
            return_value=manual_payload,
        ):
            manual_output = scheduler.run_job_now("Manual Disabled Note")
        with runtime.store.connect() as conn:
            manual_note = conn.execute(
                "SELECT * FROM scheduled_note_publications WHERE job_id = ?",
                (manual_id,),
            ).fetchone()
        if (
            manual_note is None
            or int(manual_note["allow_disabled"]) != 1
            or manual_note["state"] != "published"
            or manual_payload not in _daily_path(runtime, str(manual_note["target_date"])).read_text(
                encoding="utf-8"
            )
            or "Ran Manual Disabled Note" not in manual_output
        ):
            raise SystemExit(
                f"run_job_now did not explicitly authorize disabled note publication: {manual_output!r}"
            )


def test_first_claim_date_is_retained_across_rollover() -> None:
    with TemporaryDirectory(prefix="jarvis-note-date-rollover-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        _, row = _job(runtime, "Note Date Rollover")
        first_claim = datetime(2042, 3, 4, 23, 59, 0)
        completed = first_claim + timedelta(minutes=2)
        claimed = _claim(
            runtime,
            scheduler,
            row,
            "note-date-rollover",
            now=first_claim,
        )
        note_plan, _, _ = _finalize_note(
            runtime,
            scheduler,
            claimed,
            "note-date-rollover",
            "FIRST_CLAIM_DATE_BODY",
            completed_at=completed,
        )
        if note_plan.target_date != first_claim.date().isoformat():
            raise SystemExit(f"note target date followed completion rollover: {note_plan!r}")
        dispatch = scheduler._dispatch_scheduled_note_occurrence(note_plan.occurrence_key)
        first_path = _daily_path(runtime, first_claim.date().isoformat())
        rollover_path = _daily_path(runtime, completed.date().isoformat())
        if (
            dispatch.state != "published"
            or not first_path.exists()
            or rollover_path.exists()
            or "FIRST_CLAIM_DATE_BODY" not in first_path.read_text(encoding="utf-8")
        ):
            raise SystemExit("scheduled note did not retain its first-claim date over rollover")


def test_job_type_reconfiguration_preserves_old_sink_and_next_occurrence_is_distinct() -> None:
    with TemporaryDirectory(prefix="jarvis-note-type-reconfigure-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        job_id, row = _job(runtime, "Note Type Reconfigure", "daily_brief")
        claimed = _claim(runtime, scheduler, row, "note-old-type")
        configured_next = iso(datetime.now() + timedelta(days=5))
        runtime.store.upsert_job(
            "Note Type Reconfigure",
            10080,
            "weekly_review",
            configured_next,
        )
        old_plan, _, _ = _finalize_note(
            runtime,
            scheduler,
            claimed,
            "note-old-type",
            "OLD_DAILY_OCCURRENCE",
        )
        after_old = _job_row(runtime, job_id)
        if (
            old_plan.job_type != "daily_brief"
            or old_plan.target_kind != "daily"
            or after_old["job_type"] != "weekly_review"
            or after_old["next_run_at"] != configured_next
        ):
            raise SystemExit("old occurrence was rebound to the reconfigured handler or schedule")
        if scheduler._dispatch_scheduled_note_occurrence(old_plan.occurrence_key).state != "published":
            raise SystemExit("old occurrence did not publish through its original daily sink")

        runtime.store.upsert_job(
            "Note Type Reconfigure",
            10080,
            "weekly_review",
            iso(datetime.now() - timedelta(minutes=1)),
        )
        new_claim = _claim(
            runtime,
            scheduler,
            _job_row(runtime, job_id),
            "note-new-type",
        )
        new_plan = scheduler._scheduled_note_plan(
            new_claim,
            "NEW_REFLECTION_OCCURRENCE",
            delivery_plan=None,
            allow_disabled=False,
        )
        if (
            new_plan is None
            or new_plan.job_type != "weekly_review"
            or new_plan.target_kind != "reflection"
            or new_plan.publication_key == old_plan.publication_key
            or new_plan.occurrence_key == old_plan.occurrence_key
        ):
            raise SystemExit("next reconfigured occurrence did not receive a distinct reflection identity")
        if not runtime.store.release_job_claim(job_id, "note-new-type"):
            raise SystemExit("reconfiguration fixture could not release its final claim")


def test_legacy_missing_origin_is_retired_without_rebinding_or_publication() -> None:
    with TemporaryDirectory(prefix="jarvis-note-legacy-origin-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        job_id, row = _job(runtime, "Legacy Note Origin", "daily_brief")
        legacy_key = "scheduler-occurrence:v1:" + hashlib.sha256(
            f"legacy:{job_id}".encode("utf-8")
        ).hexdigest()
        with runtime.store.connect() as conn:
            conn.execute(
                """
                UPDATE scheduled_jobs
                SET active_occurrence_key = ?,
                    active_occurrence_schedule_revision = NULL,
                    active_occurrence_next_run_at = NULL,
                    active_occurrence_started_at = NULL,
                    active_occurrence_job_type = NULL,
                    active_occurrence_schedule_identity_revision = NULL
                WHERE id = ?
                """,
                (legacy_key, job_id),
            )
        configured_next = iso(datetime.now() + timedelta(days=4))
        runtime.store.upsert_job(
            "Legacy Note Origin",
            10080,
            "weekly_review",
            configured_next,
        )
        claimed = _claim(
            runtime,
            scheduler,
            _job_row(runtime, job_id),
            "legacy-note-retire",
        )
        if claimed["active_occurrence_job_type"] != "legacy_unknown":
            raise SystemExit("legacy occurrence was rebound to the current configured job type")
        if scheduler._scheduled_note_plan(
            claimed,
            "LEGACY_PAYLOAD_MUST_NOT_PUBLISH",
            delivery_plan=None,
            allow_disabled=False,
        ) is not None:
            raise SystemExit("legacy occurrence without immutable origin prepared a note effect")
        if not runtime.store.mark_claimed_job_run(
            job_id,
            "legacy-note-retire",
            iso(datetime.now()),
            iso(datetime.now() + timedelta(days=1)),
            int(claimed["active_occurrence_schedule_revision"]),
            expected_occurrence_key=legacy_key,
            source_job_type="legacy_unknown",
        ):
            raise SystemExit("legacy occurrence could not be retired through the fenced path")
        retired = _job_row(runtime, job_id)
        with runtime.store.connect() as conn:
            note_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM scheduled_note_publications WHERE source_job_id = ?",
                    (job_id,),
                ).fetchone()[0]
            )
        if (
            retired["job_type"] != "weekly_review"
            or retired["next_run_at"] != configured_next
            or note_count != 0
            or list(runtime.vault.root_path.rglob("*LEGACY*"))
        ):
            raise SystemExit("legacy retirement altered the current schedule or published a note")


def test_empty_file_header_manual_text_and_digest_mutation_fail_closed() -> None:
    source_key = hashlib.sha256(b"scheduled-note-marker-integrity").hexdigest()
    target_date = "2042-04-05"
    heading = "Scheduled Integrity"
    body = "OWNED_BLOCK_BODY"
    with TemporaryDirectory(prefix="jarvis-note-marker-integrity-") as temp:
        runtime = make_temp_runtime(Path(temp))
        path = _daily_path(runtime, target_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        _, appended, persisted = runtime.vault.append_scheduled_daily_once_for_date(
            target_date,
            source_key,
            heading,
            body,
        )
        first = path.read_text(encoding="utf-8")
        if not appended or persisted != body or not first.startswith(f"# {target_date}\n\n"):
            raise SystemExit("empty scheduled-note destination did not receive its canonical header")

        manual = "\nMANUAL_TEXT_OUTSIDE_JARVIS_BLOCK\n"
        path.write_text(first + manual, encoding="utf-8")
        _, replay_appended, replayed = runtime.vault.append_scheduled_daily_once_for_date(
            target_date,
            source_key,
            heading,
            "CHANGED_RETRY_BODY_MUST_NOT_REPLACE",
        )
        replay_text = path.read_text(encoding="utf-8")
        if replay_appended or replayed != body or not replay_text.endswith(manual):
            raise SystemExit("exact replay did not preserve manual text outside the owned block")

        mutated = replay_text.replace(body, body + "_MUTATED", 1)
        path.write_text(mutated, encoding="utf-8")
        try:
            runtime.vault.append_scheduled_daily_once_for_date(
                target_date,
                source_key,
                heading,
                body,
            )
        except RuntimeError:
            pass
        else:
            raise SystemExit("inside-block digest mutation did not fail closed")
        if path.read_text(encoding="utf-8") != mutated or not mutated.endswith(manual):
            raise SystemExit("failed-closed digest check altered manual or owned text")

        malformed_cases = {
            "2042-04-06": (
                f"# 2042-04-06\n\n<!-- jarvis-scheduled-note-start:v1:{source_key}:bad-digest -->\n"
            ),
            "2042-04-07": (
                f"# 2042-04-07\n\n<!-- jarvis-scheduled-note-end:v1:{source_key}:"
                f"{hashlib.sha256(b'stray').hexdigest()} -->\n"
            ),
        }
        for malformed_date, malformed_text in malformed_cases.items():
            malformed_path = _daily_path(runtime, malformed_date)
            malformed_path.write_text(malformed_text, encoding="utf-8")
            try:
                runtime.vault.append_scheduled_daily_once_for_date(
                    malformed_date,
                    source_key,
                    heading,
                    body,
                )
            except RuntimeError:
                pass
            else:
                raise SystemExit("malformed scheduled-note control line did not fail closed")
            if malformed_path.read_text(encoding="utf-8") != malformed_text:
                raise SystemExit("malformed marker refusal altered the destination")

        crlf_date = "2042-04-08"
        crlf_key = hashlib.sha256(b"scheduled-note-crlf").hexdigest()
        crlf_path, _, _ = runtime.vault.append_scheduled_daily_once_for_date(
            crlf_date,
            crlf_key,
            heading,
            body,
        )
        crlf_text = crlf_path.read_text(encoding="utf-8").replace("\n", "\r\n")
        crlf_path.write_bytes(crlf_text.encode("utf-8"))
        try:
            runtime.vault.append_scheduled_daily_once_for_date(
                crlf_date,
                crlf_key,
                heading,
                body,
            )
        except RuntimeError:
            pass
        else:
            raise SystemExit("CRLF-mutated owned payload did not fail closed")
        if _marker_count(crlf_path, crlf_key) != 1:
            raise SystemExit("CRLF marker handling appended duplicate occurrence evidence")

        embedded_date = "2042-04-09"
        embedded_key = hashlib.sha256(b"scheduled-note-embedded-control").hexdigest()
        embedded_marker = (
            f"<!-- jarvis-scheduled-note-start:v1:{embedded_key}:"
            f"{hashlib.sha256(b'embedded').hexdigest()} -->"
        )
        embedded_path, _, embedded_body = runtime.vault.append_scheduled_daily_once_for_date(
            embedded_date,
            embedded_key,
            heading,
            f"User text {embedded_marker}",
        )
        embedded_text = embedded_path.read_text(encoding="utf-8")
        if embedded_marker in embedded_text or "jarvis&#45;scheduled&#45;note-start:" not in embedded_body:
            raise SystemExit("report content could impersonate a scheduled-note control marker")


def test_linked_telegram_waits_for_note_and_terminal_note_failure_blocks_delivery() -> None:
    with TemporaryDirectory(prefix="jarvis-note-telegram-dependency-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        _, row = _job(runtime, "Note Before Telegram", "recent_file_digest")
        claimed = _claim(runtime, scheduler, row, "note-before-telegram")
        note_plan, delivery_plan, _ = _finalize_note(
            runtime,
            scheduler,
            claimed,
            "note-before-telegram",
            "NOTE_MUST_PRECEDE_TELEGRAM",
            linked_telegram=True,
        )
        if delivery_plan is None:
            raise SystemExit("linked Telegram fixture did not prepare a delivery")
        delivery = runtime.store.list_scheduled_deliveries_for_occurrence(
            delivery_plan.occurrence_key
        )[0]
        premature = runtime.store.claim_scheduled_delivery(
            str(delivery["delivery_key"]),
            "premature-telegram-claim",
        )
        if premature is not None:
            raise SystemExit("linked Telegram claimed work before its note was published")
        sent: list[str] = []
        with mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or _accepted(7201),
        ):
            held = scheduler._dispatch_occurrence(delivery_plan.occurrence_key)
        if held.state != "prepared" or sent:
            raise SystemExit(f"linked Telegram sent before note publication: {held!r} {sent!r}")

        note_dispatch = scheduler._dispatch_scheduled_note_occurrence(note_plan.occurrence_key)
        with mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or _accepted(7202),
        ):
            delivered = scheduler._dispatch_occurrence(delivery_plan.occurrence_key)
        if note_dispatch.state != "published" or delivered.state != "accepted" or len(sent) != 1:
            raise SystemExit("linked Telegram did not become sendable after note publication")

        _, failing_row = _job(runtime, "Terminal Note Blocks Telegram", "recent_file_digest")
        failing_claim = _claim(runtime, scheduler, failing_row, "terminal-note-job")
        failing_note, failing_delivery, _ = _finalize_note(
            runtime,
            scheduler,
            failing_claim,
            "terminal-note-job",
            "TERMINAL_NOTE_FAILURE_BODY",
            linked_telegram=True,
        )
        if failing_delivery is None:
            raise SystemExit("terminal note fixture did not prepare linked Telegram")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_note_publications SET payload = ? WHERE publication_key = ?",
                ("CORRUPTED_NOTE_PAYLOAD", failing_note.publication_key),
            )
        failed_note = scheduler._dispatch_scheduled_note_occurrence(
            failing_note.occurrence_key
        )
        linked = runtime.store.list_scheduled_deliveries_for_occurrence(
            failing_delivery.occurrence_key
        )
        blocked_sent: list[str] = []
        with mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: blocked_sent.append(text) or _accepted(7203),
        ):
            failed_delivery = scheduler._dispatch_occurrence(failing_delivery.occurrence_key)
        if (
            failed_note.state != "failed"
            or failed_delivery.state != "failed"
            or blocked_sent
            or not linked
            or any(
                row["state"] != "failed"
                or row["payload"] is not None
                or row["error_code"] != "blocked_by_note_publication_failure"
                for row in linked
            )
        ):
            raise SystemExit(
                "terminal note failure did not atomically terminalize linked Telegram "
                f"delivery: {failed_note!r} {linked!r}"
            )


def test_durable_evidence_and_markers_exclude_privacy_canaries_paths_and_tokens() -> None:
    canary = "PRIVATE_NOTE_EVIDENCE_CANARY_7f31"
    attempt_token = "PRIVATE_NOTE_ATTEMPT_TOKEN_83c9"
    with TemporaryDirectory(prefix="jarvis-note-privacy-evidence-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        job_id, row = _job(runtime, "Note Privacy Evidence")
        claimed = _claim(runtime, scheduler, row, "note-privacy-job")
        local_path_canary = str(Path(temp) / "private" / "source.txt")
        payload = f"{canary}\nsecret={attempt_token}"
        note_plan, _, _ = _finalize_note(
            runtime,
            scheduler,
            claimed,
            "note-privacy-job",
            payload,
        )
        with mock.patch.object(
            scheduler_module.uuid,
            "uuid4",
            return_value=SimpleNamespace(hex=attempt_token),
        ):
            dispatch = scheduler._dispatch_scheduled_note_occurrence(note_plan.occurrence_key)
        terminal = runtime.store.get_scheduled_note_publication(note_plan.publication_key)
        if dispatch.state != "published" or terminal is None:
            raise SystemExit(f"privacy evidence fixture did not publish: {dispatch!r}")
        target = _daily_path(runtime, note_plan.target_date)
        marker_lines = "\n".join(
            line for line in target.read_text(encoding="utf-8").splitlines() if "jarvis-scheduled-note-" in line
        )
        evidence = json.dumps(
            {
                "receipt": dict(terminal),
                "history": _history(runtime, job_id),
                "metadata": json.loads(_job_row(runtime, job_id)["metadata"] or "{}"),
            },
            sort_keys=True,
            default=str,
        )
        forbidden = (canary, local_path_canary, str(Path(temp)), attempt_token)
        if any(value in evidence or value in marker_lines for value in forbidden):
            raise SystemExit(
                f"durable note evidence or ownership markers leaked private custody data: {evidence!r}"
            )
        if (
            terminal["payload"] is not None
            or terminal["attempt_token"] is not None
            or str(terminal["path_display"]).startswith(("/", "~"))
            or str(terminal["path_display"]) != f"Daily/{note_plan.target_date}.md"
            or canary not in target.read_text(encoding="utf-8")
        ):
            raise SystemExit(f"terminal note evidence retained unsafe custody fields: {dict(terminal)!r}")


def test_shared_vault_stores_get_distinct_opaque_occurrence_keys() -> None:
    with TemporaryDirectory(prefix="jarvis-note-shared-vault-") as temp:
        root = Path(temp)
        first = make_temp_runtime(root / "first")
        second = make_temp_runtime(root / "second")
        first_scheduler = Scheduler(first.store, first.vault, first.config)
        second_scheduler = Scheduler(second.store, first.vault, second.config)
        next_run = iso(datetime(2042, 6, 7, 8, 9, 10))
        first_id = first.store.upsert_job("Shared Vault Note", 1440, "daily_brief", next_run)
        second_id = second.store.upsert_job("Shared Vault Note", 1440, "daily_brief", next_run)
        claim_time = datetime(2042, 6, 7, 8, 10, 0)
        first_row = first.store.claim_job(
            first_id,
            "shared-vault-first",
            first_scheduler.job_lease_seconds,
            now=claim_time,
        )
        second_row = second.store.claim_job(
            second_id,
            "shared-vault-second",
            second_scheduler.job_lease_seconds,
            now=claim_time,
        )
        if first_row is None or second_row is None:
            raise SystemExit("shared-vault fixture could not claim both store occurrences")
        first_key, _ = first_scheduler._scheduled_note_publication(first_row, "daily_brief")
        second_key, _ = second_scheduler._scheduled_note_publication(second_row, "daily_brief")
        if first_key == second_key or not all(
            len(key) == 64 and set(key) <= set("0123456789abcdef")
            for key in (first_key, second_key)
        ):
            raise SystemExit("two stores sharing one vault did not receive distinct opaque keys")


def main() -> None:
    test_finalization_precedes_drain_payload_is_immutable_and_history_converges()
    test_crash_after_file_publish_stale_recovery_coalesces_exact_block()
    test_destination_size_failure_stays_pre_effect_and_terminalizes()
    test_wrong_valid_marker_stays_pre_effect_and_terminalizes()
    test_stale_replaced_note_claim_writes_nothing()
    test_delete_after_effect_boundary_preserves_truthful_publication()
    test_concurrent_drainers_publish_one_block()
    test_paused_due_job_holds_then_resume_drains_and_manual_disabled_run_publishes()
    test_first_claim_date_is_retained_across_rollover()
    test_job_type_reconfiguration_preserves_old_sink_and_next_occurrence_is_distinct()
    test_legacy_missing_origin_is_retired_without_rebinding_or_publication()
    test_empty_file_header_manual_text_and_digest_mutation_fail_closed()
    test_linked_telegram_waits_for_note_and_terminal_note_failure_blocks_delivery()
    test_durable_evidence_and_markers_exclude_privacy_canaries_paths_and_tokens()
    test_shared_vault_stores_get_distinct_opaque_occurrence_keys()
    print("Scheduler note publication custody smoke passed")


if __name__ == "__main__":
    main()
