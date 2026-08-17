from __future__ import annotations

import io
import json
import os
import urllib.error
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Barrier, Event, Lock, Thread
from unittest import mock

from jarvis_v2.agent.types import ToolResult
from jarvis_v2.automations import scheduler as scheduler_module
from jarvis_v2.automations import telegram_control
from jarvis_v2.automations.jobs import GoalNudgeBuild
from jarvis_v2.automations.scheduler import (
    MORNING_BRIEF_ENV,
    MORNING_BRIEF_INTERVAL_MINUTES,
    MORNING_BRIEF_JOB_NAME,
    MORNING_BRIEF_JOB_TYPE,
    Scheduler,
    _payload_sha256,
    _run_marker,
    iso,
)
from jarvis_v2.scripts.test_runtime import handle_runtime_case, make_temp_runtime


def _accepted(message_id: int = 101) -> dict:
    return {"ok": True, "result": {"message_id": message_id}}


def _goal_nudge_build(runtime, output: str) -> GoalNudgeBuild:
    snapshot = runtime.store.read_goal_nudge_snapshot()
    return GoalNudgeBuild(
        output=output,
        source_manifest_kind=snapshot.source_manifest_kind,
        source_manifest_digest=snapshot.source_manifest_digest,
    )


def _job(runtime, name: str, job_type: str, *, due: bool = False):
    when = datetime.now() + (-timedelta(minutes=5) if due else timedelta(hours=1))
    job_id = runtime.store.upsert_job(name, 1440, job_type, iso(when))
    return next(row for row in runtime.store.list_jobs() if int(row["id"]) == job_id)


def _rows(runtime, occurrence_key: str):
    return runtime.store.list_scheduled_deliveries_for_occurrence(occurrence_key)


def _prepare_plan(runtime, scheduler: Scheduler, row, payload: str, *, now: datetime | None = None):
    plan = scheduler._delivery_plan(row, payload, now or datetime.now())
    runtime.store.prepare_scheduled_delivery_occurrence(
        int(row["id"]),
        str(row["name"]),
        str(row["job_type"]),
        plan.occurrence_key,
        list(plan.chunks),
    )
    return plan


def _run_due(scheduler: Scheduler) -> str:
    with mock.patch.dict(os.environ, {MORNING_BRIEF_ENV: ""}, clear=False):
        return scheduler.run_due_jobs()


def _long_payload() -> str:
    return "\n".join(f"section {index:03d}: " + (chr(65 + index % 26) * 140) for index in range(96))


def test_ledger_state_flow_and_accepted_payload_is_cleared() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-state-flow-") as temp:
        runtime = make_temp_runtime(Path(temp))
        row = _job(runtime, "State Flow", "inbox_ingest")
        delivery_key = "state-flow:0"
        payload = "one Telegram API call"
        prepared = runtime.store.prepare_scheduled_delivery(
            delivery_key,
            int(row["id"]),
            str(row["name"]),
            str(row["job_type"]),
            payload,
            _payload_sha256(payload),
        )
        if prepared["state"] != "prepared":
            raise SystemExit(f"new delivery must start prepared: {dict(prepared)}")

        claimed = runtime.store.claim_scheduled_delivery(delivery_key, "attempt-state-flow")
        if claimed is None or claimed["state"] != "claimed" or int(claimed["attempt_count"]) != 0:
            raise SystemExit(f"delivery must atomically enter claimed: {claimed}")
        if not runtime.store.begin_scheduled_delivery_send(delivery_key, "attempt-state-flow"):
            raise SystemExit("claimed delivery did not enter sending")
        sending = runtime.store.get_scheduled_delivery(delivery_key)
        if sending is None or sending["state"] != "sending" or int(sending["attempt_count"]) != 1:
            raise SystemExit(f"delivery was not fenced as sending before network I/O: {sending}")
        if not runtime.store.finish_scheduled_delivery(
            delivery_key,
            "attempt-state-flow",
            "accepted",
            telegram_message_id=812,
        ):
            raise SystemExit("accepted delivery could not be finalized")
        accepted = runtime.store.get_scheduled_delivery(delivery_key)
        if accepted is None or accepted["state"] != "accepted":
            raise SystemExit(f"delivery did not finish accepted: {accepted}")
        if accepted["payload"] is not None or int(accepted["telegram_message_id"]) != 812:
            raise SystemExit(f"accepted row must clear payload and retain only its receipt: {dict(accepted)}")
        listed = runtime.registry.get("list_scheduled_jobs").handler({})
        if listed.metadata.get("scheduled_delivery_occurrences") != 1:
            raise SystemExit(f"scheduled-job status must expose occurrence counts: {listed.metadata}")
        if listed.metadata.get("scheduled_delivery_content_bearing_chunks") != 0:
            raise SystemExit(f"accepted status must report no retained payload chunks: {listed.metadata}")
        for expected in ("occurrences: 1", "accepted by API: 1", "content-bearing retry chunks: 0"):
            if expected not in listed.output:
                raise SystemExit(f"scheduled-job status missed ledger health {expected!r}: {listed.output}")


def test_attempt_budget_is_consumed_only_when_send_begins() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-attempt-budget-") as temp:
        runtime = make_temp_runtime(Path(temp))
        row = _job(runtime, "Attempt Budget", "inbox_ingest")
        key = "attempt-budget:0"
        payload = "send starts consume attempts"
        runtime.store.prepare_scheduled_delivery(
            key,
            int(row["id"]),
            str(row["name"]),
            str(row["job_type"]),
            payload,
            _payload_sha256(payload),
        )

        fresh = runtime.store.claim_scheduled_delivery(key, "fresh-claim")
        if fresh is None or int(fresh["attempt_count"]) != 0:
            raise SystemExit(f"fresh pre-I/O claim consumed attempt budget: {fresh}")
        if runtime.store.begin_scheduled_delivery_send(key, "wrong-token"):
            raise SystemExit("wrong attempt token entered sending")
        after_wrong_token = runtime.store.get_scheduled_delivery(key)
        if after_wrong_token is None or int(after_wrong_token["attempt_count"]) != 0:
            raise SystemExit(f"wrong attempt token consumed attempt budget: {after_wrong_token}")

        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_deliveries SET claimed_at = '2000-01-01T00:00:00Z' WHERE delivery_key = ?",
                (key,),
            )
        recovered = runtime.store.recover_stale_scheduled_deliveries(iso(datetime.now()))
        stale = runtime.store.get_scheduled_delivery(key)
        if (
            recovered["claimed_to_prepared"] != 1
            or stale is None
            or stale["state"] != "prepared"
            or int(stale["attempt_count"]) != 0
        ):
            raise SystemExit(f"stale pre-I/O claim consumed attempt budget: {recovered!r} {stale}")

        token = "concurrent-begin"
        reclaimed = runtime.store.claim_scheduled_delivery(key, token)
        if reclaimed is None or int(reclaimed["attempt_count"]) != 0:
            raise SystemExit(f"reclaimed pre-I/O row consumed attempt budget: {reclaimed}")
        start = Barrier(2)
        results: Queue[object] = Queue()

        def begin() -> None:
            try:
                start.wait(timeout=5)
                results.put(runtime.store.begin_scheduled_delivery_send(key, token))
            except BaseException as exc:
                results.put(exc)

        threads = [Thread(target=begin) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        if any(thread.is_alive() for thread in threads):
            raise SystemExit("concurrent send-begin smoke left a thread running")
        observed = [results.get_nowait() for _ in range(results.qsize())]
        sending = runtime.store.get_scheduled_delivery(key)
        if (
            any(isinstance(item, BaseException) for item in observed)
            or sorted(observed) != [False, True]
            or sending is None
            or sending["state"] != "sending"
            or int(sending["attempt_count"]) != 1
        ):
            raise SystemExit(f"concurrent send begin did not consume exactly one attempt: {observed!r} {sending}")

        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_deliveries SET sending_at = '2000-01-01T00:00:00Z' WHERE delivery_key = ?",
                (key,),
            )
        recovered = runtime.store.recover_stale_scheduled_deliveries(iso(datetime.now()))
        uncertain = runtime.store.get_scheduled_delivery(key)
        if (
            recovered["sending_to_uncertain"] != 1
            or uncertain is None
            or uncertain["state"] != "uncertain"
            or int(uncertain["attempt_count"]) != 1
        ):
            raise SystemExit(f"uncertain recovery lost the real send attempt: {recovered!r} {uncertain}")


def test_terminal_receipt_retention_is_bounded_without_pruning_live_work() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-retention-") as temp:
        runtime = make_temp_runtime(Path(temp))
        row = _job(runtime, "Retention Digest", "inbox_ingest")
        for label in ("accepted-old", "uncertain-old", "prepared-live"):
            runtime.store.prepare_scheduled_delivery(
                label,
                int(row["id"]),
                str(row["name"]),
                str(row["job_type"]),
                label,
                _payload_sha256(label),
            )
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_deliveries SET state='accepted', payload=NULL, finished_at='2000-01-01T00:00:00Z' WHERE delivery_key='accepted-old'"
            )
            conn.execute(
                "UPDATE scheduled_deliveries SET state='uncertain', payload=NULL, finished_at='2000-01-01T00:00:00Z' WHERE delivery_key='uncertain-old'"
            )
            conn.execute(
                "UPDATE scheduled_deliveries SET created_at='2000-01-01T00:00:00Z' WHERE delivery_key='prepared-live'"
            )
        deleted = runtime.store.prune_scheduled_deliveries(
            "2020-01-01T00:00:00Z",
            "2020-01-01T00:00:00Z",
            100,
        )
        if deleted != 2:
            raise SystemExit(f"retention should prune two old terminal receipts: {deleted}")
        remaining = runtime.store.get_scheduled_delivery("prepared-live")
        if remaining is None or remaining["state"] != "prepared" or remaining["payload"] != "prepared-live":
            raise SystemExit(f"retention pruned or altered live delivery work: {remaining}")


def test_corrupted_persisted_payload_is_scrubbed_before_network_io() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-payload-integrity-") as temp:
        runtime = make_temp_runtime(Path(temp))
        row = _job(runtime, "Payload Integrity", "inbox_ingest")
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        plan = _prepare_plan(runtime, scheduler, row, "original private payload SHOULD NOT APPEAR")
        first = _rows(runtime, plan.occurrence_key)[0]
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_deliveries SET payload = ? WHERE delivery_key = ?",
                ("CORRUPTED private payload SHOULD NOT APPEAR", first["delivery_key"]),
            )
        sent: list[str] = []
        with mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or _accepted(),
        ):
            dispatch = scheduler._dispatch_occurrence(plan.occurrence_key)
        if sent:
            raise SystemExit(f"corrupted scheduled payload crossed the network boundary: {sent!r}")
        if dispatch.state != "failed" or dispatch.output != "scheduled Telegram payload integrity check failed; nothing was sent":
            raise SystemExit(f"corrupted scheduled payload returned false dispatch truth: {dispatch!r}")
        persisted = _rows(runtime, plan.occurrence_key)
        if not persisted or any(row["state"] != "failed" or row["payload"] is not None for row in persisted):
            raise SystemExit(f"corrupted scheduled occurrence was not scrubbed and quarantined: {[dict(row) for row in persisted]}")
        if persisted[0]["error_code"] != "payload_integrity_failed":
            raise SystemExit(f"corrupted scheduled payload missed its bounded error code: {dict(persisted[0])}")
        detail = dispatch.output + json.dumps(
            [
                {
                    "state": item["state"],
                    "error_code": item["error_code"],
                    "payload_present": item["payload"] is not None,
                }
                for item in persisted
            ],
            sort_keys=True,
        )
        for forbidden in ("SHOULD NOT APPEAR", "CORRUPTED private", "original private"):
            if forbidden in detail:
                raise SystemExit(f"payload-integrity diagnostic leaked content {forbidden!r}: {detail}")

        null_job = _job(runtime, "Null Payload Integrity", "inbox_ingest")
        null_plan = _prepare_plan(runtime, scheduler, null_job, "second private payload SHOULD NOT APPEAR")
        null_row = _rows(runtime, null_plan.occurrence_key)[0]
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_deliveries SET payload = NULL WHERE delivery_key = ?",
                (null_row["delivery_key"],),
            )
        sent.clear()
        with mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or _accepted(),
        ):
            null_dispatch = scheduler._dispatch_occurrence(null_plan.occurrence_key)
        if sent or null_dispatch.state != "failed":
            raise SystemExit(f"NULL scheduled payload must fail before network I/O: {sent!r} {null_dispatch!r}")


def test_malformed_claim_timestamps_recover_without_stranding_or_replay() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-malformed-claim-time-") as temp:
        runtime = make_temp_runtime(Path(temp))
        row = _job(runtime, "Malformed Claim Time", "inbox_ingest")
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)

        claimed_plan = _prepare_plan(runtime, scheduler, row, "safe pre-I/O retry")
        claimed_row = _rows(runtime, claimed_plan.occurrence_key)[0]
        if runtime.store.claim_scheduled_delivery(claimed_row["delivery_key"], "null-claimed-at") is None:
            raise SystemExit("malformed claimed-at fixture could not claim its row")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_deliveries SET claimed_at = NULL WHERE delivery_key = ?",
                (claimed_row["delivery_key"],),
            )
        recovered = runtime.store.recover_stale_scheduled_deliveries("9999-12-31T23:59:59Z")
        recovered_row = runtime.store.get_scheduled_delivery(claimed_row["delivery_key"])
        if recovered["claimed_to_prepared"] != 1 or recovered_row is None or recovered_row["state"] != "prepared":
            raise SystemExit(f"NULL claimed_at row remained stranded: {recovered!r} {recovered_row}")
        if recovered_row["payload"] != "safe pre-I/O retry":
            raise SystemExit(f"pre-I/O recovery lost its retryable payload: {dict(recovered_row)}")

        sending_job = _job(runtime, "Malformed Sending Time", "inbox_ingest")
        sending_plan = _prepare_plan(runtime, scheduler, sending_job, "ambiguous in-flight payload SHOULD NOT APPEAR")
        sending_row = _rows(runtime, sending_plan.occurrence_key)[0]
        token = "null-sending-at"
        if runtime.store.claim_scheduled_delivery(sending_row["delivery_key"], token) is None:
            raise SystemExit("malformed sending-at fixture could not claim its row")
        if not runtime.store.begin_scheduled_delivery_send(sending_row["delivery_key"], token):
            raise SystemExit("malformed sending-at fixture could not enter sending")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_deliveries SET sending_at = NULL WHERE delivery_key = ?",
                (sending_row["delivery_key"],),
            )
        recovered = runtime.store.recover_stale_scheduled_deliveries("9999-12-31T23:59:59Z")
        uncertain_row = runtime.store.get_scheduled_delivery(sending_row["delivery_key"])
        if recovered["sending_to_uncertain"] != 1 or uncertain_row is None or uncertain_row["state"] != "uncertain":
            raise SystemExit(f"NULL sending_at row was not quarantined uncertain: {recovered!r} {uncertain_row}")
        if uncertain_row["payload"] is not None or uncertain_row["error_code"] != "stale_sending_recovered":
            raise SystemExit(f"ambiguous NULL sending_at row retained replayable content: {dict(uncertain_row)}")


def test_duplicate_occurrence_sends_once_under_concurrent_tickers() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-concurrent-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _job(runtime, "Concurrent Digest", "goal_nudge", due=True)
        schedulers = [Scheduler(runtime.store, runtime.vault, runtime.config) for _ in range(2)]
        start = Barrier(2)
        send_entered = Event()
        release_send = Event()
        send_lock = Lock()
        sent: list[str] = []
        results: Queue[object] = Queue()

        def send(text: str) -> dict:
            with send_lock:
                sent.append(text)
            send_entered.set()
            if not release_send.wait(3):
                raise RuntimeError("concurrent smoke did not release mocked send")
            return _accepted()

        def tick(scheduler: Scheduler) -> None:
            try:
                start.wait(timeout=3)
                results.put(_run_due(scheduler))
            except BaseException as exc:
                results.put(exc)

        with mock.patch.object(
            scheduler_module,
            "build_goal_nudge",
            return_value=_goal_nudge_build(runtime, "exactly once"),
        ), mock.patch.object(
            scheduler_module, "_send_owner_telegram", side_effect=send
        ):
            threads = [Thread(target=tick, args=(scheduler,)) for scheduler in schedulers]
            for thread in threads:
                thread.start()
            if not send_entered.wait(3):
                raise SystemExit("concurrent ticker smoke never reached mocked Telegram send")
            release_send.set()
            for thread in threads:
                thread.join(timeout=5)

        if any(thread.is_alive() for thread in threads):
            raise SystemExit("concurrent delivery ticker left a worker running")
        observed = [results.get_nowait() for _ in range(results.qsize())]
        failures = [item for item in observed if isinstance(item, BaseException)]
        if failures:
            raise SystemExit(f"concurrent delivery ticker failed: {failures!r}")
        if sent != ["🎯 Goal Nudge\n\nexactly once"]:
            raise SystemExit(f"duplicate scheduled occurrence must send exactly once: {sent!r}")
        with runtime.store.connect() as conn:
            rows = list(conn.execute("SELECT state, payload FROM scheduled_deliveries"))
        if len(rows) != 1 or rows[0]["state"] != "accepted" or rows[0]["payload"] is not None:
            raise SystemExit(f"concurrent occurrence should leave one scrubbed accepted row: {rows}")


def test_delivered_goal_nudge_preserves_active_reschedule_and_claimed_identity() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-active-reschedule-") as temp:
        runtime = make_temp_runtime(Path(temp))
        original_name = "Active Reschedule Delivery"
        original_type = "goal_nudge"
        refreshed_type = "weekly_review"
        sentinel_next_run = "2099-01-02T03:04:05"
        _job(runtime, original_name, original_type, due=True)
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        builder_entered = Event()
        release_builder = Event()
        sent: list[str] = []
        results: Queue[object] = Queue()

        def build_goal_nudge(*_args: object, **_kwargs: object) -> GoalNudgeBuild:
            builder_entered.set()
            if not release_builder.wait(timeout=5):
                raise RuntimeError("active-reschedule builder was not released")
            return _goal_nudge_build(
                runtime,
                "claimed occurrence survived its reschedule",
            )

        def run_owner() -> None:
            try:
                results.put(_run_due(scheduler))
            except BaseException as exc:
                results.put(exc)

        with mock.patch.object(scheduler_module, "build_goal_nudge", side_effect=build_goal_nudge), mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or _accepted(211),
        ):
            thread = Thread(target=run_owner)
            thread.start()
            try:
                if not builder_entered.wait(timeout=5):
                    raise SystemExit("active-reschedule smoke never reached the claimed goal-nudge builder")
                active = next(row for row in runtime.store.list_jobs() if row["name"] == original_name)
                if not active["lease_token"]:
                    raise SystemExit(f"active-reschedule smoke did not hold a job lease: {dict(active)}")
                runtime.store.upsert_job(original_name, 120, refreshed_type, sentinel_next_run)
            finally:
                release_builder.set()
                thread.join(timeout=5)

        if thread.is_alive():
            raise SystemExit("active-reschedule smoke left its scheduler thread running")
        completed = results.get(timeout=5)
        if isinstance(completed, BaseException) or f"Ran {original_name}" not in str(completed):
            raise SystemExit(f"active-reschedule owner did not finish its claimed occurrence: {completed!r}")
        expected_payload = "🎯 Goal Nudge\n\nclaimed occurrence survived its reschedule"
        if sent != [expected_payload]:
            raise SystemExit(f"active-reschedule occurrence was not sent exactly once: {sent!r}")

        with runtime.store.connect() as conn:
            deliveries = list(conn.execute("SELECT * FROM scheduled_deliveries ORDER BY chunk_index ASC"))
        if len(deliveries) != 1:
            raise SystemExit(f"active-reschedule occurrence did not commit exactly one chunk: {deliveries}")
        delivery = deliveries[0]
        if (
            delivery["state"] != "accepted"
            or delivery["payload"] is not None
            or delivery["job_name"] != original_name
            or delivery["job_type"] != original_type
        ):
            raise SystemExit(f"delivery ledger lost the original claimed job identity: {dict(delivery)}")

        current = next(row for row in runtime.store.list_jobs() if row["name"] == original_name)
        if (
            int(current["interval_minutes"]) != 120
            or current["job_type"] != refreshed_type
            or current["next_run_at"] != sentinel_next_run
        ):
            raise SystemExit(f"claimed occurrence overwrote the concurrent reschedule: {dict(current)}")
        if current["last_run_at"] is None:
            raise SystemExit(f"claimed occurrence did not record last_run_at: {dict(current)}")
        if current["lease_token"] is not None or current["lease_expires_at"] is not None:
            raise SystemExit(f"claimed occurrence did not clear its lease: {dict(current)}")

        replayed: list[str] = []
        with mock.patch.object(scheduler_module, "build_goal_nudge") as old_builder, mock.patch.object(
            scheduler_module, "build_weekly_review"
        ) as refreshed_builder, mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: replayed.append(text) or _accepted(212),
        ):
            second_tick = _run_due(Scheduler(runtime.store, runtime.vault, runtime.config))
        if old_builder.called or refreshed_builder.called or replayed:
            raise SystemExit(
                f"next tick reran the committed occurrence: old={old_builder.call_count} "
                f"refreshed={refreshed_builder.call_count} sent={replayed!r}"
            )
        if second_tick != "No jobs due.":
            raise SystemExit(f"next tick should leave the future reschedule idle: {second_tick!r}")


def test_active_outbox_occurrence_advances_across_pause_resume() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-pause-resume-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        base = datetime(2040, 2, 3, 4, 5, 6)
        due_at = iso(base - timedelta(minutes=5))

        for suffix, resume_before_completion in (
            ("Resume After Completion", False),
            ("Resume Before Completion", True),
        ):
            name = f"Outbox {suffix}"
            job_id = runtime.store.upsert_job(name, 1440, "inbox_ingest", due_at)
            token = f"outbox-{suffix.lower().replace(' ', '-')}"
            claimed = runtime.store.claim_job(job_id, token, 300, due_before=iso(base), now=base)
            if claimed is None:
                raise SystemExit(f"could not claim {name} fixture")
            plan = scheduler._delivery_plan(claimed, f"payload for {suffix}", base)
            if not runtime.store.set_job_enabled(name, False):
                raise SystemExit(f"could not pause active {name} fixture")
            if resume_before_completion and not runtime.store.set_job_enabled(name, True):
                raise SystemExit(f"could not resume active {name} fixture before completion")

            completed_at = base + timedelta(seconds=1)
            advanced_at = iso(base + timedelta(days=1))
            if not runtime.store.complete_claimed_job_with_scheduled_deliveries(
                job_id,
                token,
                iso(completed_at),
                advanced_at,
                plan.occurrence_key,
                list(plan.chunks),
                str(claimed["name"]),
                str(claimed["job_type"]),
                int(claimed["active_occurrence_schedule_revision"]),
                now=completed_at,
            ):
                raise SystemExit(f"active {name} fixture could not complete with its outbox")
            if not resume_before_completion and not runtime.store.set_job_enabled(name, True):
                raise SystemExit(f"could not resume completed {name} fixture")

            current = next(row for row in runtime.store.list_jobs() if int(row["id"]) == job_id)
            deliveries = _rows(runtime, plan.occurrence_key)
            if current["next_run_at"] != advanced_at or current["enabled"] != 1:
                raise SystemExit(f"pause/resume replayed the completed outbox occurrence: {dict(current)}")
            if len(deliveries) != 1 or deliveries[0]["state"] != "prepared":
                raise SystemExit(f"outbox completion did not atomically prepare one delivery: {deliveries}")
            if current["active_occurrence_key"] is not None or current["lease_token"] is not None:
                raise SystemExit(f"outbox pause/resume completion left active ownership behind: {dict(current)}")

        name = "Outbox Concurrent Real Reschedule"
        job_id = runtime.store.upsert_job(name, 1440, "inbox_ingest", due_at)
        token = "outbox-real-reschedule"
        claimed = runtime.store.claim_job(job_id, token, 300, due_before=iso(base), now=base)
        if claimed is None:
            raise SystemExit("could not claim outbox real-reschedule fixture")
        plan = scheduler._delivery_plan(claimed, "preserve the real reschedule", base)
        rescheduled_at = iso(base + timedelta(days=7))
        runtime.store.upsert_job(name, 120, "weekly_review", rescheduled_at)
        if not runtime.store.complete_claimed_job_with_scheduled_deliveries(
            job_id,
            token,
            iso(base + timedelta(seconds=1)),
            iso(base + timedelta(days=1)),
            plan.occurrence_key,
            list(plan.chunks),
            str(claimed["name"]),
            str(claimed["job_type"]),
            int(claimed["active_occurrence_schedule_revision"]),
            now=base + timedelta(seconds=1),
        ):
            raise SystemExit("outbox real-reschedule fixture could not complete")
        current = next(row for row in runtime.store.list_jobs() if int(row["id"]) == job_id)
        if current["next_run_at"] != rescheduled_at or current["job_type"] != "weekly_review":
            raise SystemExit(f"outbox completion overwrote a genuine concurrent reschedule: {dict(current)}")

        name = "Outbox Same Due Reschedule"
        job_id = runtime.store.upsert_job(name, 1440, "inbox_ingest", due_at)
        token = "outbox-same-due-reschedule"
        claimed = runtime.store.claim_job(job_id, token, 300, due_before=iso(base), now=base)
        if claimed is None:
            raise SystemExit("could not claim outbox same-due reschedule fixture")
        plan = scheduler._delivery_plan(claimed, "preserve the same-due reschedule", base)
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_jobs SET schedule_identity_revision = ? WHERE id = ?",
                ("-1", job_id),
            )
        runtime.store.upsert_job_with_metadata(
            name,
            180,
            "weekly_review",
            due_at,
            {"schedule_marker": "replacement"},
        )
        if not runtime.store.complete_claimed_job_with_scheduled_deliveries(
            job_id,
            token,
            iso(base + timedelta(seconds=1)),
            iso(base + timedelta(days=1)),
            plan.occurrence_key,
            list(plan.chunks),
            str(claimed["name"]),
            str(claimed["job_type"]),
            int(claimed["active_occurrence_schedule_revision"]),
            now=base + timedelta(seconds=1),
        ):
            raise SystemExit("outbox same-due reschedule fixture could not complete")
        current = next(row for row in runtime.store.list_jobs() if int(row["id"]) == job_id)
        metadata = json.loads(current["metadata"] or "{}")
        deliveries = _rows(runtime, plan.occurrence_key)
        if (
            current["next_run_at"] != due_at
            or current["interval_minutes"] != 180
            or current["job_type"] != "weekly_review"
            or type(current["schedule_identity_revision"]) is not int
            or current["schedule_identity_revision"] != -1
            or metadata.get("schedule_marker") != "replacement"
        ):
            raise SystemExit(f"outbox completion overwrote a same-due reschedule: {dict(current)}")
        if len(deliveries) != 1 or deliveries[0]["job_type"] != "inbox_ingest":
            raise SystemExit(f"same-due outbox lost its claimed source identity: {deliveries}")


def test_two_dispatchers_claim_one_prepared_chunk_once() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-dispatch-race-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        row = _job(runtime, "Dispatch Race", "inbox_ingest")
        plan = _prepare_plan(runtime, scheduler, row, "one prepared chunk")
        start = Barrier(2)
        entered = Event()
        release = Event()
        sent: list[str] = []
        results: Queue[object] = Queue()

        def send(text: str) -> dict:
            sent.append(text)
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("dispatch-race send was not released")
            return _accepted(202)

        def dispatch() -> None:
            try:
                start.wait(timeout=5)
                results.put(scheduler._dispatch_occurrence(plan.occurrence_key))
            except BaseException as exc:
                results.put(exc)

        with mock.patch.object(scheduler_module, "_send_owner_telegram", side_effect=send):
            threads = [Thread(target=dispatch) for _ in range(2)]
            for thread in threads:
                thread.start()
            if not entered.wait(timeout=5):
                raise SystemExit("dispatch-race smoke never entered the mocked API call")
            release.set()
            for thread in threads:
                thread.join(timeout=5)
        if any(thread.is_alive() for thread in threads):
            raise SystemExit("dispatch-race smoke left a thread running")
        observed = [results.get_nowait() for _ in range(results.qsize())]
        if any(isinstance(item, BaseException) for item in observed):
            raise SystemExit(f"dispatch-race smoke raised: {observed}")
        if sent != ["one prepared chunk"]:
            raise SystemExit(f"two dispatchers issued duplicate Telegram calls: {sent}")
        if _rows(runtime, plan.occurrence_key)[0]["state"] != "accepted":
            raise SystemExit("dispatch-race winner did not persist the accepted receipt")


def test_prepared_before_finalize_survives_restart_and_drains() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-restart-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        row = _job(runtime, "Restart Digest", "inbox_ingest", due=True)
        token = "restart-job-claim"
        claimed = runtime.store.claim_job(int(row["id"]), token, 300, due_before=iso(datetime.now()))
        if claimed is None:
            raise SystemExit("restart smoke could not claim its due job")
        completed_at = datetime.now()
        plan = scheduler._delivery_plan_for_output(claimed, "survives process restart", completed_at)
        if plan is None or not plan.chunks:
            raise SystemExit(f"restart smoke did not prepare a delivery plan: {plan}")
        finalized = runtime.store.complete_claimed_job_with_scheduled_deliveries(
            int(claimed["id"]),
            token,
            _run_marker(completed_at),
            iso(completed_at + timedelta(days=1)),
            plan.occurrence_key,
            list(plan.chunks),
            str(claimed["name"]),
            str(claimed["job_type"]),
            int(claimed["schedule_revision"]),
        )
        if not finalized:
            raise SystemExit("job finalization and outbox preparation were not committed together")
        prepared = _rows(runtime, plan.occurrence_key)
        if [item["state"] for item in prepared] != ["prepared"]:
            raise SystemExit(f"restart fixture should be durably prepared before dispatch: {prepared}")

        sent: list[str] = []
        restarted = Scheduler(runtime.store, runtime.vault, runtime.config)
        with mock.patch.object(
            scheduler_module, "_send_owner_telegram", side_effect=lambda text: sent.append(text) or _accepted()
        ):
            _run_due(restarted)
        if sent != ["📥 Inbox Ingest\n\nsurvives process restart"]:
            raise SystemExit(f"restart did not drain the prepared occurrence once: {sent!r}")
        if _rows(runtime, plan.occurrence_key)[0]["state"] != "accepted":
            raise SystemExit("restart drain did not finalize the prepared row")
        metadata = next(row for row in runtime.store.list_jobs() if row["name"] == "Restart Digest")["metadata"]
        parsed = json.loads(metadata or "{}")
        if parsed.get("last_delivery_status") != "accepted" or parsed.get("last_delivery_occurrence_key") != plan.occurrence_key:
            raise SystemExit(f"restart recovery did not reconcile content-free job delivery metadata: {parsed}")


def test_transient_rejection_recovers_after_restart_without_duplicate_delivery() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-network-restart-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        row = _job(runtime, "Transient Network Digest", "inbox_ingest")
        plan = _prepare_plan(runtime, scheduler, row, "retry once after deterministic HTTP 503")
        payload = str(plan.chunks[0]["payload"])
        attempts: list[tuple[str, str]] = []

        with mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: attempts.append(("before_restart", text))
            or {
                "ok": False,
                "error": "http_503",
                "http_status": 503,
                "retry_after": 0,
            },
        ):
            first = scheduler._dispatch_occurrence(plan.occurrence_key)
        rejected = _rows(runtime, plan.occurrence_key)[0]
        if (
            first.state != "rejected"
            or attempts != [("before_restart", payload)]
            or rejected["state"] != "rejected"
            or int(rejected["attempt_count"]) != 1
            or rejected["payload"] != payload
        ):
            raise SystemExit(
                f"transient pre-restart rejection was not durably retryable: {first} {attempts} {dict(rejected)}"
            )

        restarted_runtime = make_temp_runtime(root)
        restarted = Scheduler(
            restarted_runtime.store,
            restarted_runtime.vault,
            restarted_runtime.config,
        )
        with mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: attempts.append(("after_restart", text))
            or _accepted(9503),
        ):
            recovered_output = _run_due(restarted)
        recovered = _rows(restarted_runtime, plan.occurrence_key)[0]
        if attempts != [
            ("before_restart", payload),
            ("after_restart", payload),
        ]:
            raise SystemExit(f"daemon restart issued the wrong network attempts: {attempts}")
        if (
            recovered["state"] != "accepted"
            or int(recovered["attempt_count"]) != 2
            or int(recovered["telegram_message_id"]) != 9503
            or recovered["payload"] is not None
        ):
            raise SystemExit(f"restart recovery missed its accepted receipt: {dict(recovered)}")
        if "Recovered scheduled Telegram delivery: accepted by Telegram API." not in recovered_output:
            raise SystemExit(f"restart recovery did not report deterministic acceptance: {recovered_output}")

        second_restart_runtime = make_temp_runtime(root)
        replayed: list[str] = []
        with mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: replayed.append(text) or _accepted(9504),
        ):
            _run_due(
                Scheduler(
                    second_restart_runtime.store,
                    second_restart_runtime.vault,
                    second_restart_runtime.config,
                )
            )
        final = _rows(second_restart_runtime, plan.occurrence_key)[0]
        if replayed or final["state"] != "accepted" or int(final["attempt_count"]) != 2:
            raise SystemExit(
                f"accepted restart recovery replayed a duplicate delivery: {replayed} {dict(final)}"
            )


def test_recovered_morning_brief_updates_daily_receipt_metadata() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-morning-recovery-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        row = _job(runtime, MORNING_BRIEF_JOB_NAME, MORNING_BRIEF_JOB_TYPE, due=True)
        token = "morning-recovery-claim"
        claimed = runtime.store.claim_job(int(row["id"]), token, 300, due_before=iso(datetime.now()))
        if claimed is None:
            raise SystemExit("Morning Brief recovery smoke could not claim its occurrence")
        completed_at = datetime.now()
        plan = scheduler._delivery_plan(claimed, "recover this morning brief", completed_at)
        if not runtime.store.complete_claimed_job_with_scheduled_deliveries(
            int(claimed["id"]),
            token,
            _run_marker(completed_at),
            iso(completed_at + timedelta(days=1)),
            plan.occurrence_key,
            list(plan.chunks),
            str(claimed["name"]),
            str(claimed["job_type"]),
            int(claimed["schedule_revision"]),
        ):
            raise SystemExit("Morning Brief recovery smoke could not atomically prepare its outbox")
        sent: list[str] = []
        with mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or _accepted(303),
        ), mock.patch.object(Scheduler, "ensure_env_morning_brief", return_value=None):
            Scheduler(runtime.store, runtime.vault, runtime.config).run_due_jobs()
        if sent != ["recover this morning brief"]:
            raise SystemExit(f"recovered Morning Brief did not drain exactly once: {sent}")
        current = next(item for item in runtime.store.list_jobs() if item["name"] == MORNING_BRIEF_JOB_NAME)
        metadata = json.loads(current["metadata"] or "{}")
        if metadata.get("last_sent_date") != completed_at.date().isoformat():
            raise SystemExit(f"recovered Morning Brief missed daily receipt metadata: {metadata}")
        if metadata.get("last_delivery_status") != "accepted" or metadata.get("last_send_status") != "accepted":
            raise SystemExit(f"recovered Morning Brief missed accepted status reconciliation: {metadata}")


def test_send_exception_is_uncertain_and_never_auto_replays() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-uncertain-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        row = _job(runtime, "Uncertain Digest", "inbox_ingest")
        plan = _prepare_plan(runtime, scheduler, row, "ambiguous network outcome")
        with mock.patch.object(scheduler_module, "_send_owner_telegram", side_effect=RuntimeError("socket vanished")):
            dispatch = scheduler._dispatch_occurrence(plan.occurrence_key)
        uncertain = _rows(runtime, plan.occurrence_key)[0]
        if dispatch.state != "uncertain" or uncertain["state"] != "uncertain":
            raise SystemExit(f"exception after send claim must quarantine uncertain: {dispatch} {dict(uncertain)}")
        if uncertain["payload"] is not None:
            raise SystemExit(f"uncertain payload must be cleared to make replay impossible: {dict(uncertain)}")

        replayed: list[str] = []
        with mock.patch.object(
            scheduler_module, "_send_owner_telegram", side_effect=lambda text: replayed.append(text) or _accepted()
        ):
            _run_due(Scheduler(runtime.store, runtime.vault, runtime.config))
            _run_due(Scheduler(runtime.store, runtime.vault, runtime.config))
        if replayed or _rows(runtime, plan.occurrence_key)[0]["state"] != "uncertain":
            raise SystemExit(f"uncertain delivery must never auto-replay: {replayed!r}")


def test_api_acceptance_with_local_receipt_failure_never_replays() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-receipt-failure-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        row = _job(runtime, "Receipt Failure", "inbox_ingest")
        plan = _prepare_plan(runtime, scheduler, row, "accepted before local receipt failure")
        sent: list[str] = []
        original_finish = runtime.store.finish_scheduled_delivery

        def fail_receipt(*args, **kwargs):
            raise RuntimeError("synthetic local receipt failure")

        runtime.store.finish_scheduled_delivery = fail_receipt  # type: ignore[method-assign]
        try:
            with mock.patch.object(
                scheduler_module,
                "_send_owner_telegram",
                side_effect=lambda text: sent.append(text) or _accepted(404),
            ):
                dispatch = scheduler._dispatch_occurrence(plan.occurrence_key)
        finally:
            runtime.store.finish_scheduled_delivery = original_finish  # type: ignore[method-assign]
        if dispatch.state != "uncertain" or sent != ["accepted before local receipt failure"]:
            raise SystemExit(f"receipt failure should be visibly uncertain after one API call: {dispatch} {sent}")
        key = str(plan.chunks[0]["delivery_key"])
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_deliveries SET sending_at = '2000-01-01T00:00:00Z' WHERE delivery_key = ?",
                (key,),
            )
        replayed: list[str] = []
        with mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: replayed.append(text) or _accepted(),
        ):
            _run_due(Scheduler(runtime.store, runtime.vault, runtime.config))
        final = runtime.store.get_scheduled_delivery(key)
        if replayed or final is None or final["state"] != "uncertain" or final["payload"] is not None:
            raise SystemExit(f"failed local receipt must recover uncertain without a second send: {replayed} {final}")


def test_stale_claimed_recovers_but_stale_sending_quarantines_suffix() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-stale-claimed-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        row = _job(runtime, "Stale Claimed", "inbox_ingest")
        plan = _prepare_plan(runtime, scheduler, row, "safe pre-network recovery")
        key = str(plan.chunks[0]["delivery_key"])
        if runtime.store.claim_scheduled_delivery(key, "stale-claimed-token") is None:
            raise SystemExit("could not create stale claimed fixture")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_deliveries SET claimed_at = '2000-01-01T00:00:00Z' WHERE delivery_key = ?",
                (key,),
            )
        sent: list[str] = []
        with mock.patch.object(
            scheduler_module, "_send_owner_telegram", side_effect=lambda text: sent.append(text) or _accepted()
        ):
            _run_due(Scheduler(runtime.store, runtime.vault, runtime.config))
        recovered = _rows(runtime, plan.occurrence_key)[0]
        if sent != ["safe pre-network recovery"] or recovered["state"] != "accepted":
            raise SystemExit(f"stale claimed row should recover and send once: sent={sent!r} row={dict(recovered)}")

    with TemporaryDirectory(prefix="jarvis-delivery-stale-sending-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        row = _job(runtime, "Stale Sending", "inbox_ingest")
        plan = _prepare_plan(runtime, scheduler, row, _long_payload())
        first_key = str(plan.chunks[0]["delivery_key"])
        if runtime.store.claim_scheduled_delivery(first_key, "stale-sending-token") is None:
            raise SystemExit("could not claim stale sending fixture")
        if not runtime.store.begin_scheduled_delivery_send(first_key, "stale-sending-token"):
            raise SystemExit("could not enter stale sending fixture")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_deliveries SET sending_at = '2000-01-01T00:00:00Z' WHERE delivery_key = ?",
                (first_key,),
            )
        sent = []
        with mock.patch.object(
            scheduler_module, "_send_owner_telegram", side_effect=lambda text: sent.append(text) or _accepted()
        ):
            _run_due(Scheduler(runtime.store, runtime.vault, runtime.config))
        states = [item["state"] for item in _rows(runtime, plan.occurrence_key)]
        if sent or not states or states[0] != "uncertain" or any(state != "failed" for state in states[1:]):
            raise SystemExit(f"stale sending must quarantine and block every later chunk: sent={sent!r} states={states}")
        if any(item["payload"] is not None for item in _rows(runtime, plan.occurrence_key)[1:]):
            raise SystemExit("stale uncertain recovery must scrub every blocked suffix payload")


def test_rejection_retries_and_long_chunks_are_ordered_deterministically() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-chunks-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        row = _job(runtime, "Chunked Digest", "inbox_ingest")
        payload = _long_payload()
        first_plan = scheduler._delivery_plan(row, payload, datetime(2026, 7, 11, 9, 0))
        second_plan = scheduler._delivery_plan(row, payload, datetime(2026, 7, 11, 9, 0))
        if first_plan != second_plan or len(first_plan.chunks) < 3:
            raise SystemExit("long payload must split into at least three deterministic chunk rows")
        expected = [str(chunk["payload"]) for chunk in first_plan.chunks]
        runtime.store.prepare_scheduled_delivery_occurrence(
            int(row["id"]),
            str(row["name"]),
            str(row["job_type"]),
            first_plan.occurrence_key,
            list(first_plan.chunks),
        )

        attempted: list[str] = []
        sending_states: list[str] = []

        def reject_first(text: str) -> dict:
            attempted.append(text)
            current = _rows(runtime, first_plan.occurrence_key)
            sending_states.append(str(current[0]["state"]))
            return {"ok": False, "error": "telegram_rejected"}

        with mock.patch.object(scheduler_module, "DELIVERY_RETRY_SECONDS", 0), mock.patch.object(
            scheduler_module, "_send_owner_telegram", side_effect=reject_first
        ):
            rejected = scheduler._dispatch_occurrence(first_plan.occurrence_key)
        states = [item["state"] for item in _rows(runtime, first_plan.occurrence_key)]
        if rejected.state != "rejected" or attempted != expected[:1] or sending_states != ["sending"]:
            raise SystemExit(f"explicit rejection must stop after the first sending chunk: {rejected} {attempted!r}")
        if states[0] != "rejected" or any(state != "prepared" for state in states[1:]):
            raise SystemExit(f"later chunks must remain blocked behind rejection: {states}")

        accepted_attempts: list[str] = []
        with mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: accepted_attempts.append(text) or _accepted(len(accepted_attempts)),
        ):
            completed = scheduler._dispatch_occurrence(first_plan.occurrence_key)
        final_rows = _rows(runtime, first_plan.occurrence_key)
        if completed.state != "accepted" or accepted_attempts != expected:
            raise SystemExit(f"retry must accept each deterministic chunk exactly once and in order: {accepted_attempts!r}")
        if any(item["state"] != "accepted" or item["payload"] is not None for item in final_rows):
            raise SystemExit(f"accepted chunks must all be scrubbed: {[dict(item) for item in final_rows]}")


def test_partial_chunk_success_is_uncertain_and_blocks_remaining_chunks() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-partial-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        row = _job(runtime, "Partial Digest", "inbox_ingest")
        plan = _prepare_plan(runtime, scheduler, row, _long_payload())
        attempts: list[str] = []

        def partial(text: str) -> dict:
            attempts.append(text)
            if len(attempts) == 1:
                return _accepted()
            raise RuntimeError("connection lost after request began")

        with mock.patch.object(scheduler_module, "_send_owner_telegram", side_effect=partial):
            dispatch = scheduler._dispatch_occurrence(plan.occurrence_key)
        states = [item["state"] for item in _rows(runtime, plan.occurrence_key)]
        if dispatch.state != "uncertain" or states[:2] != ["accepted", "uncertain"]:
            raise SystemExit(f"partial chunk outcome must be accepted then uncertain: {dispatch} {states}")
        if any(state != "failed" for state in states[2:]):
            raise SystemExit(f"uncertain earlier chunk must quarantine every suffix chunk: {states}")
        if any(item["payload"] is not None for item in _rows(runtime, plan.occurrence_key)[1:]):
            raise SystemExit("uncertain chunk must scrub every blocked suffix payload")

        replayed: list[str] = []
        with mock.patch.object(
            scheduler_module, "_send_owner_telegram", side_effect=lambda text: replayed.append(text) or _accepted()
        ):
            scheduler._dispatch_occurrence(plan.occurrence_key)
            _run_due(Scheduler(runtime.store, runtime.vault, runtime.config))
        if replayed:
            raise SystemExit(f"uncertain chunk must block all later automatic sends: {replayed!r}")


def test_delivery_key_collision_fails_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-collision-") as temp:
        runtime = make_temp_runtime(Path(temp))
        row = _job(runtime, "Collision Digest", "inbox_ingest")
        original = "trusted payload"
        runtime.store.prepare_scheduled_delivery(
            "collision-key",
            int(row["id"]),
            str(row["name"]),
            str(row["job_type"]),
            original,
            _payload_sha256(original),
            occurrence_key="collision-occurrence",
        )
        try:
            runtime.store.prepare_scheduled_delivery(
                "collision-key",
                int(row["id"]),
                str(row["name"]),
                str(row["job_type"]),
                "hostile replacement",
                _payload_sha256("hostile replacement"),
                occurrence_key="collision-occurrence",
            )
        except ValueError as exc:
            if "collision" not in str(exc).casefold():
                raise SystemExit(f"key collision should fail with a safe diagnostic: {exc}")
        else:
            raise SystemExit("reused delivery key with different payload did not fail closed")
        stored = runtime.store.get_scheduled_delivery("collision-key")
        if stored is None or stored["state"] != "prepared" or stored["payload"] != original:
            raise SystemExit(f"collision attempt modified the trusted ledger row: {stored}")


def test_mid_batch_collision_rolls_back_enqueue_and_job_finalization() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-batch-rollback-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        row = _job(runtime, "Batch Rollback", "inbox_ingest", due=True)
        token = "batch-rollback-claim"
        claimed = runtime.store.claim_job(int(row["id"]), token, 300, due_before=iso(datetime.now()))
        if claimed is None:
            raise SystemExit("batch rollback smoke could not claim its due job")
        plan = scheduler._delivery_plan(claimed, _long_payload(), datetime.now())
        if len(plan.chunks) < 2:
            raise SystemExit("batch rollback smoke needs multiple chunks")
        second = plan.chunks[1]
        hostile_payload = "preexisting collision"
        now_text = "2000-01-01T00:00:00Z"
        with runtime.store.connect() as conn:
            conn.execute(
                """
                INSERT INTO scheduled_deliveries(
                    delivery_key, occurrence_key, chunk_index, chunk_count,
                    job_id, source_job_id, job_name, job_type, payload,
                    payload_sha256, state, next_attempt_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'prepared', ?, ?, ?)
                """,
                (
                    second["delivery_key"],
                    plan.occurrence_key,
                    second["chunk_index"],
                    second["chunk_count"],
                    int(claimed["id"]),
                    int(claimed["id"]),
                    str(claimed["name"]),
                    str(claimed["job_type"]),
                    hostile_payload,
                    _payload_sha256(hostile_payload),
                    now_text,
                    now_text,
                    now_text,
                ),
            )
        try:
            runtime.store.complete_claimed_job_with_scheduled_deliveries(
                int(claimed["id"]),
                token,
                _run_marker(datetime.now()),
                iso(datetime.now() + timedelta(days=1)),
                plan.occurrence_key,
                list(plan.chunks),
                str(claimed["name"]),
                str(claimed["job_type"]),
                int(claimed["schedule_revision"]),
            )
        except ValueError:
            pass
        else:
            raise SystemExit("mid-batch collision did not fail closed")
        rows = _rows(runtime, plan.occurrence_key)
        if len(rows) != 1 or int(rows[0]["chunk_index"]) != int(second["chunk_index"]):
            raise SystemExit(f"mid-batch collision did not roll back earlier inserts: {len(rows)}")
        current = next(item for item in runtime.store.list_jobs() if item["name"] == "Batch Rollback")
        if current["lease_token"] != token or current["last_run_at"] is not None:
            raise SystemExit("mid-batch collision advanced or released the claimed job")


def test_incomplete_occurrence_is_rejected_and_retries_are_bounded() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-incomplete-") as temp:
        runtime = make_temp_runtime(Path(temp))
        row = _job(runtime, "Incomplete Digest", "inbox_ingest")
        payload = "only one of three"
        try:
            runtime.store.prepare_scheduled_delivery(
                "incomplete:0",
                int(row["id"]),
                str(row["name"]),
                str(row["job_type"]),
                payload,
                _payload_sha256(payload),
                occurrence_key="incomplete",
                chunk_index=0,
                chunk_count=3,
            )
        except ValueError as exc:
            if "atomically" not in str(exc):
                raise SystemExit(f"incomplete occurrence refusal should name the atomic path: {exc}")
        else:
            raise SystemExit("single-row API committed an incomplete multi-chunk occurrence")
        if runtime.store.list_scheduled_deliveries_for_occurrence("incomplete"):
            raise SystemExit("incomplete occurrence refusal must leave no ledger rows")

        key = "bounded-retry:0"
        retry_payload = "bounded explicit rejection"
        runtime.store.prepare_scheduled_delivery(
            key,
            int(row["id"]),
            str(row["name"]),
            str(row["job_type"]),
            retry_payload,
            _payload_sha256(retry_payload),
        )
        for attempt in range(1, 6):
            token = f"bounded-attempt-{attempt}"
            claimed = runtime.store.claim_scheduled_delivery(key, token)
            if claimed is None or int(claimed["attempt_count"]) != attempt - 1:
                raise SystemExit(f"bounded rejection claim {attempt} consumed attempt budget: {claimed}")
            if not runtime.store.begin_scheduled_delivery_send(key, token):
                raise SystemExit(f"bounded rejection attempt {attempt} could not enter sending")
            sending = runtime.store.get_scheduled_delivery(key)
            if sending is None or int(sending["attempt_count"]) != attempt:
                raise SystemExit(f"bounded rejection send start {attempt} was not counted once: {sending}")
            if not runtime.store.finish_scheduled_delivery(
                key,
                token,
                "rejected",
                error_code="explicit_rejection",
                retry_after_seconds=0,
            ):
                raise SystemExit(f"bounded rejection attempt {attempt} could not finalize")
        terminal = runtime.store.get_scheduled_delivery(key)
        if terminal is None or terminal["state"] != "failed" or int(terminal["attempt_count"]) != 5:
            raise SystemExit(f"explicit rejection must stop at five attempts: {terminal}")
        if terminal["payload"] is not None or terminal["next_attempt_at"] is not None:
            raise SystemExit("terminal rejection must scrub payload and disable retry")
        if runtime.store.claim_scheduled_delivery(key, "sixth-attempt") is not None:
            raise SystemExit("terminal rejection accepted an unbounded sixth attempt")


def test_http_rejections_are_retryable_but_malformed_results_are_uncertain() -> None:
    body = io.BytesIO(b'{"ok":false,"parameters":{"retry_after":17},"description":"hostile/private text"}')
    error = urllib.error.HTTPError("https://example.invalid", 429, "rate limited", {}, body)
    with mock.patch.object(telegram_control.urllib.request, "urlopen", side_effect=error):
        result = telegram_control._api_call("synthetic-token", "sendMessage", {"chat_id": "1"}, 1)
    if result != {"ok": False, "error": "http_429", "http_status": 429, "retry_after": 17}:
        raise SystemExit(f"Telegram HTTP normalization must keep only bounded retry evidence: {result}")
    if "hostile" in str(result) or "private" in str(result):
        raise SystemExit("Telegram HTTP normalization leaked response description")

    with mock.patch.object(
        telegram_control.urllib.request,
        "urlopen",
        side_effect=urllib.error.URLError("hostile/private network detail"),
    ):
        network_unknown = telegram_control._api_call(
            "synthetic-token", "sendMessage", {"chat_id": "1"}, 1
        )
    if network_unknown != {"error": "network_outcome_unknown"} or "ok" in network_unknown:
        raise SystemExit(
            f"Telegram network ambiguity must not become a replayable rejection: {network_unknown}"
        )

    class _MalformedTelegramResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"{not-json"

    with mock.patch.object(
        telegram_control.urllib.request,
        "urlopen",
        return_value=_MalformedTelegramResponse(),
    ):
        malformed = telegram_control._api_call(
            "synthetic-token", "sendMessage", {"chat_id": "1"}, 1
        )
    if malformed != {"error": "malformed_response"} or "ok" in malformed:
        raise SystemExit(
            f"Telegram malformed acknowledgement must remain an unknown outcome: {malformed}"
        )

    with TemporaryDirectory(prefix="jarvis-delivery-http-policy-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        row = _job(runtime, "HTTP Policy", "inbox_ingest")
        retry_plan = _prepare_plan(runtime, scheduler, row, "retry the definite rejection")
        with mock.patch.object(scheduler_module, "_send_owner_telegram", return_value=result):
            retry = scheduler._dispatch_occurrence(retry_plan.occurrence_key)
        retry_row = _rows(runtime, retry_plan.occurrence_key)[0]
        if retry.state != "rejected" or retry_row["state"] != "rejected" or retry_row["next_attempt_at"] is None:
            raise SystemExit(f"definite HTTP 429 rejection must remain retryable: {retry} {dict(retry_row)}")

        other = _job(runtime, "Malformed Policy", "inbox_ingest")
        unknown_plan = _prepare_plan(runtime, scheduler, other, "do not replay malformed acknowledgement")
        with mock.patch.object(scheduler_module, "_send_owner_telegram", return_value={"ok": True}):
            unknown = scheduler._dispatch_occurrence(unknown_plan.occurrence_key)
        unknown_row = _rows(runtime, unknown_plan.occurrence_key)[0]
        if unknown.state != "uncertain" or unknown_row["state"] != "uncertain" or unknown_row["payload"] is not None:
            raise SystemExit(f"malformed Telegram result must quarantine without replay: {unknown} {dict(unknown_row)}")


def test_outbox_recovery_failure_does_not_starve_due_jobs() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-recovery-isolation-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _job(runtime, "Recovery Isolation", "goal_nudge", due=True)
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        sent: list[str] = []

        # This case isolates outbox-recovery failure handling. Real lease expiry,
        # renewal, and competing-owner safety are exercised in
        # smoke_test_scheduler_basics; using its background heartbeat here made
        # an unrelated recovery assertion depend on thread scheduling.
        class DeterministicOwnedLease:
            def __init__(self) -> None:
                self.started = False
                self.stopped = False
                self.authority_checks = 0

            def start(self) -> bool:
                self.started = True
                return True

            def require_authority(self) -> None:
                if not self.started or self.stopped:
                    raise RuntimeError("deterministic_test_lease_not_live")
                self.authority_checks += 1

            def stop(self) -> None:
                self.stopped = True

        heartbeat = DeterministicOwnedLease()
        with mock.patch.object(
            scheduler,
            "_recover_and_drain_scheduled_notes",
            side_effect=RuntimeError("/private/note recovery details must not leak"),
        ), mock.patch.object(
            scheduler,
            "_recover_and_drain_deliveries",
            side_effect=RuntimeError("/private/telegram recovery details must not leak"),
        ), mock.patch.object(
            scheduler_module,
            "build_goal_nudge",
            return_value=_goal_nudge_build(runtime, "due work still ran"),
        ), mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or _accepted(),
        ), mock.patch.object(
            scheduler,
            "_claim_heartbeat",
            return_value=heartbeat,
        ):
            output = _run_due(scheduler)
        if "Scheduled note outbox recovery failed: RuntimeError; due jobs continued." not in output:
            raise SystemExit(f"isolated note recovery failure missed safe bounded receipt: {output}")
        if "Scheduled Telegram outbox recovery failed: RuntimeError; due jobs continued." not in output:
            raise SystemExit(f"isolated recovery failure missed safe bounded receipt: {output}")
        if "Ran Recovery Isolation" not in output or sent != ["🎯 Goal Nudge\n\ndue work still ran"]:
            raise SystemExit(f"outbox recovery failure starved due job execution: {output} {sent}")
        for expected in (
            "list scheduled jobs",
            "setup check",
            "storage, model, or connector issue",
            "durable delivery receipt state",
            "do not manually rerun or resend",
            "outcome-unknown delivery",
        ):
            if expected not in output:
                raise SystemExit(f"outbox recovery failure missed guidance {expected!r}: {output}")
        if "/private" in output or "recovery details" in output or "must not leak" in output:
            raise SystemExit("outbox recovery failure leaked exception details")
        if (
            not heartbeat.started
            or not heartbeat.stopped
            or heartbeat.authority_checks < 2
        ):
            raise SystemExit(
                "outbox recovery isolation bypassed the owned-job authority contract: "
                f"started={heartbeat.started}, stopped={heartbeat.stopped}, "
                f"checks={heartbeat.authority_checks}"
            )


def test_morning_brief_schedule_and_on_demand_alias_send_once() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-morning-scheduled-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.upsert_job(
            MORNING_BRIEF_JOB_NAME,
            MORNING_BRIEF_INTERVAL_MINUTES,
            MORNING_BRIEF_JOB_TYPE,
            iso(datetime.now() - timedelta(minutes=5)),
        )
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        built: list[object] = []
        sent: list[str] = []
        with mock.patch.object(
            scheduler_module,
            "_build_live_daily_brief",
            side_effect=lambda config: built.append(config) or "scheduled morning brief",
        ), mock.patch.object(
            scheduler_module, "_send_owner_telegram", side_effect=lambda text: sent.append(text) or _accepted()
        ), mock.patch.object(
            Scheduler, "ensure_env_morning_brief", return_value=None
        ):
            scheduler.run_due_jobs()
            runtime.store.upsert_job(
                MORNING_BRIEF_JOB_NAME,
                MORNING_BRIEF_INTERVAL_MINUTES,
                MORNING_BRIEF_JOB_TYPE,
                iso(datetime.now() - timedelta(minutes=1)),
            )
            Scheduler(runtime.store, runtime.vault, runtime.config).run_due_jobs()
        if len(built) != 1 or sent != ["scheduled morning brief"]:
            raise SystemExit(f"Morning Brief once-per-day ledger guard failed: built={built} sent={sent}")

    aliases = ("run morning brief now", "push today's brief to my phone")
    for index, alias in enumerate(aliases):
        with TemporaryDirectory(prefix=f"jarvis-delivery-morning-alias-{index}-") as temp:
            runtime = make_temp_runtime(Path(temp))
            scheduled = runtime.registry.get("schedule_morning_brief").handler({"time": "7:30am"})
            if not scheduled.ok:
                raise SystemExit(f"could not schedule Morning Brief for alias {alias!r}: {scheduled}")
            built = []
            sent = []
            with mock.patch.object(
                scheduler_module,
                "_build_live_daily_brief",
                side_effect=lambda config: built.append(config) or f"alias morning brief {index}",
            ), mock.patch.object(
                scheduler_module, "_send_owner_telegram", side_effect=lambda text: sent.append(text) or _accepted()
            ):
                result = handle_runtime_case(runtime, alias, approved=True)
            if not result.verified or len(built) != 1 or sent != [f"alias morning brief {index}"]:
                raise SystemExit(f"on-demand alias must compose and send once: {alias!r} {result} {built} {sent}")


def test_morning_brief_job_recreation_reuses_daily_receipt() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-morning-recreate-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.upsert_job(
            MORNING_BRIEF_JOB_NAME,
            MORNING_BRIEF_INTERVAL_MINUTES,
            MORNING_BRIEF_JOB_TYPE,
            iso(datetime.now() - timedelta(minutes=5)),
        )
        built: list[object] = []
        sent: list[str] = []
        with mock.patch.object(
            scheduler_module,
            "_build_live_daily_brief",
            side_effect=lambda config: built.append(config) or "recreated morning brief",
        ), mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or _accepted(),
        ), mock.patch.object(Scheduler, "ensure_env_morning_brief", return_value=None):
            Scheduler(runtime.store, runtime.vault, runtime.config).run_due_jobs()
            if not runtime.store.delete_job(MORNING_BRIEF_JOB_NAME):
                raise SystemExit("Morning Brief recreation smoke could not delete the original job")
            runtime.store.upsert_job(
                MORNING_BRIEF_JOB_NAME,
                MORNING_BRIEF_INTERVAL_MINUTES,
                MORNING_BRIEF_JOB_TYPE,
                iso(datetime.now() - timedelta(minutes=1)),
            )
            second = Scheduler(runtime.store, runtime.vault, runtime.config).run_due_jobs()
        if len(built) != 1 or sent != ["recreated morning brief"]:
            raise SystemExit(f"recreated Morning Brief job must reuse the owner-day receipt: built={built} sent={sent}")
        if "skipped duplicate" not in second:
            raise SystemExit(f"recreated Morning Brief job should explain daily coalescing: {second}")


def test_recreated_morning_brief_recovers_post_finalize_crash() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-morning-recreate-crash-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.upsert_job(
            MORNING_BRIEF_JOB_NAME,
            MORNING_BRIEF_INTERVAL_MINUTES,
            MORNING_BRIEF_JOB_TYPE,
            iso(datetime.now() - timedelta(minutes=5)),
        )
        built: list[object] = []
        sent: list[str] = []
        with mock.patch.object(
            scheduler_module,
            "_build_live_daily_brief",
            side_effect=lambda config: built.append(config) or "recreated crash brief",
        ), mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or _accepted(),
        ), mock.patch.object(Scheduler, "ensure_env_morning_brief", return_value=None):
            Scheduler(runtime.store, runtime.vault, runtime.config).run_due_jobs()
            if not runtime.store.delete_job(MORNING_BRIEF_JOB_NAME):
                raise SystemExit("Morning Brief crash fixture could not delete the original job")
            runtime.store.upsert_job(
                MORNING_BRIEF_JOB_NAME,
                MORNING_BRIEF_INTERVAL_MINUTES,
                MORNING_BRIEF_JOB_TYPE,
                iso(datetime.now() - timedelta(minutes=1)),
            )
            recreated_scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
            original_finalize = recreated_scheduler._finalize_claim_with_delivery

            def finalize_then_crash(*args, **kwargs):
                if not original_finalize(*args, **kwargs):
                    raise SystemExit("Morning Brief crash fixture did not finalize before interruption")
                raise RuntimeError("simulated process stop after occurrence finalization")

            with mock.patch.object(
                recreated_scheduler,
                "_finalize_claim_with_delivery",
                side_effect=finalize_then_crash,
            ):
                recreated_scheduler.run_due_jobs()
        if len(built) != 1 or sent != ["recreated crash brief"]:
            raise SystemExit(f"crash recovery fixture duplicated work: built={built} sent={sent}")
        recreated = next(
            row
            for row in runtime.store.list_jobs()
            if str(row["name"]).casefold() == MORNING_BRIEF_JOB_NAME.casefold()
        )
        pending_history = json.loads(recreated["metadata"] or "{}").get("run_history") or []
        if len(pending_history) != 1 or pending_history[0].get("status") != "pending":
            raise SystemExit(f"post-finalize crash did not retain pending truth: {pending_history!r}")
        reconciliations = runtime.store.list_scheduled_delivery_history_reconciliations()
        if not any(
            str(row["occurrence_key"]) == pending_history[0].get("occurrence_key")
            for row in reconciliations
        ):
            raise SystemExit("recreated Morning Brief pending truth was not recoverable")
        Scheduler(runtime.store, runtime.vault, runtime.config)._recover_and_drain_deliveries()
        recovered = next(row for row in runtime.store.list_jobs() if int(row["id"]) == int(recreated["id"]))
        recovered_history = json.loads(recovered["metadata"] or "{}").get("run_history") or []
        if len(recovered_history) != 1 or recovered_history[0].get("status") != "ok":
            raise SystemExit(f"recreated Morning Brief pending truth did not recover: {recovered_history!r}")


def test_manual_non_morning_digest_does_not_push() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-manual-digest-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _job(runtime, "Manual Goal Nudge", "goal_nudge")
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        sent: list[str] = []
        with mock.patch.object(
            scheduler_module,
            "build_goal_nudge",
            return_value=_goal_nudge_build(runtime, "manual caller output"),
        ), mock.patch.object(
            scheduler_module, "_send_owner_telegram", side_effect=lambda text: sent.append(text) or _accepted()
        ):
            output = scheduler.run_job_now("Manual Goal Nudge")
        with runtime.store.connect() as conn:
            delivery_count = int(conn.execute("SELECT COUNT(*) FROM scheduled_deliveries").fetchone()[0])
        if "manual caller output" not in output or sent or delivery_count:
            raise SystemExit(
                f"manual non-Morning digest must return locally without push/outbox: output={output!r} sent={sent!r} rows={delivery_count}"
            )


def test_recent_file_digest_without_watched_dirs_is_quiet_but_configured_digest_delivers() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-file-digest-quiet-") as temp:
        root = Path(temp)
        quiet_runtime = make_temp_runtime(root / "quiet")
        quiet_config = replace(quiet_runtime.config, watched_dirs=())
        quiet_scheduler = Scheduler(
            quiet_runtime.store,
            quiet_runtime.vault,
            quiet_config,
        )
        _job(quiet_runtime, "Quiet File Digest", "recent_file_digest", due=True)
        quiet_sent: list[str] = []
        with mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: quiet_sent.append(text) or _accepted(),
        ):
            quiet_output = _run_due(quiet_scheduler)
        with quiet_runtime.store.connect() as conn:
            quiet_delivery_count = int(
                conn.execute("SELECT COUNT(*) FROM scheduled_deliveries").fetchone()[0]
            )
        if quiet_sent or quiet_delivery_count:
            raise SystemExit(
                "unconfigured scheduled file digest must not create or send Telegram work: "
                f"sent={quiet_sent!r} rows={quiet_delivery_count}"
            )
        if (
            "No watched directories configured." not in quiet_output
            or "quiet — nothing new, delivery skipped" not in quiet_output
        ):
            raise SystemExit(
                f"unconfigured scheduled file digest did not record a clean quiet run: {quiet_output!r}"
            )
        manual_output = quiet_scheduler._run_job_type("recent_file_digest")
        if (
            not isinstance(manual_output, ToolResult)
            or not manual_output.ok
            or "Set JARVIS_WATCHED_DIRS" not in manual_output.output
        ):
            raise SystemExit(
                f"manual file digest lost its setup guidance: {manual_output!r}"
            )

        active_runtime = make_temp_runtime(root / "active")
        watched = active_runtime.config.watched_dirs[0]
        watched.mkdir(parents=True, exist_ok=True)
        (watched / "proof.txt").write_text(
            "scheduled file digest delivery proof",
            encoding="utf-8",
        )
        _job(active_runtime, "Active File Digest", "recent_file_digest", due=True)
        active_scheduler = Scheduler(
            active_runtime.store,
            active_runtime.vault,
            active_runtime.config,
        )
        active_sent: list[str] = []
        with mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: active_sent.append(text) or _accepted(104),
        ):
            active_output = _run_due(active_scheduler)
        if (
            len(active_sent) != 1
            or "🗂 Recent File Digest" not in active_sent[0]
            or "proof.txt" not in active_sent[0]
            or "accepted by Telegram API" not in active_output
        ):
            raise SystemExit(
                "configured scheduled file digest did not retain one delivery: "
                f"sent={active_sent!r} output={active_output!r}"
            )


def test_pause_finalizes_started_run_without_sending_until_resume() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-pause-fence-") as temp:
        runtime = make_temp_runtime(Path(temp))
        row = _job(runtime, "Pause Fence", "goal_nudge", due=True)
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        handler_started = Event()
        release_handler = Event()
        sent: list[str] = []
        result: Queue[object] = Queue()

        def blocked_handler(*_args, **_kwargs) -> GoalNudgeBuild:
            handler_started.set()
            if not release_handler.wait(timeout=5):
                raise RuntimeError("pause fence smoke timed out")
            return _goal_nudge_build(runtime, "handler completed after pause")

        def run_tick() -> None:
            try:
                result.put(_run_due(scheduler))
            except BaseException as exc:
                result.put(exc)

        with mock.patch.object(scheduler_module, "build_goal_nudge", side_effect=blocked_handler), mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or _accepted(),
        ), mock.patch.object(Scheduler, "ensure_env_morning_brief", return_value=None):
            thread = Thread(target=run_tick, name="scheduler-pause-fence-smoke")
            thread.start()
            if not handler_started.wait(timeout=5):
                raise SystemExit("scheduled handler did not reach the deterministic pause point")
            paused = scheduler.pause_job(str(row["name"]))
            if paused != "Paused Pause Fence.":
                raise SystemExit(f"pause did not disable the active scheduled run: {paused!r}")
            release_handler.set()
            thread.join(timeout=5)
        if thread.is_alive():
            raise SystemExit("paused scheduler worker did not terminate")
        observed = result.get_nowait()
        if isinstance(observed, BaseException):
            raise SystemExit(f"paused scheduler worker raised unexpectedly: {observed!r}")
        current = next(item for item in runtime.store.list_jobs() if int(item["id"]) == int(row["id"]))
        with runtime.store.connect() as conn:
            deliveries = list(
                conn.execute(
                    "SELECT state, payload FROM scheduled_deliveries ORDER BY delivery_key"
                )
            )
        if sent or len(deliveries) != 1 or deliveries[0]["state"] != "prepared":
            raise SystemExit(
                f"pause did not hold the finalized run's pending delivery: "
                f"sent={sent!r} rows={[dict(item) for item in deliveries]!r}"
            )
        if (
            current["enabled"] != 0
            or current["lease_token"] is not None
            or current["last_run_at"] is None
            or current["active_occurrence_key"] is not None
        ):
            raise SystemExit(f"paused started owner did not finalize exactly once: {dict(current)}")
        if "Ran Pause Fence" not in str(observed) or "Skipped stale claim" in str(observed):
            raise SystemExit(f"paused started run lost its truthful completion: {observed!r}")

        with mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or _accepted(),
        ), mock.patch.object(Scheduler, "ensure_env_morning_brief", return_value=None):
            paused_tick = scheduler.run_due_jobs()
        if sent or "accepted by Telegram API" in paused_tick:
            raise SystemExit(f"paused job dispatched its held outbox: {paused_tick!r}")

        with mock.patch.object(
            scheduler_module,
            "build_goal_nudge",
            side_effect=AssertionError("resuming must not replay the completed handler"),
        ), mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or _accepted(),
        ), mock.patch.object(Scheduler, "ensure_env_morning_brief", return_value=None):
            if scheduler.resume_job(str(row["name"])) != "Resumed Pause Fence.":
                raise SystemExit("paused finalized run could not be resumed")
            resumed_tick = scheduler.run_due_jobs()
        with runtime.store.connect() as conn:
            final_delivery = conn.execute(
                "SELECT state, payload FROM scheduled_deliveries ORDER BY delivery_key"
            ).fetchone()
        if (
            len(sent) != 1
            or final_delivery is None
            or final_delivery["state"] != "accepted"
            or final_delivery["payload"] is not None
            or "accepted by Telegram API" not in resumed_tick
        ):
            raise SystemExit(
                f"resume did not drain the held delivery exactly once: "
                f"sent={sent!r} row={dict(final_delivery) if final_delivery else None} "
                f"output={resumed_tick!r}"
            )


def test_pause_between_delivery_claim_and_send_releases_pre_io_claim() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-pause-claimed-") as temp:
        runtime = make_temp_runtime(Path(temp))
        row = _job(runtime, "Pause Claimed Delivery", "goal_nudge", due=True)
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        real_begin = runtime.store.begin_scheduled_delivery_send
        sent: list[str] = []
        pauses: list[str] = []
        handler_calls: list[str] = []

        def pause_before_begin(delivery_key: str, attempt_token: str) -> bool:
            pauses.append(scheduler.pause_job(str(row["name"])))
            return real_begin(delivery_key, attempt_token)

        with mock.patch.object(
            scheduler_module,
            "build_goal_nudge",
            side_effect=lambda *_args, **_kwargs: handler_calls.append("built")
            or _goal_nudge_build(runtime, "claimed pause payload"),
        ), mock.patch.object(
            runtime.store,
            "begin_scheduled_delivery_send",
            side_effect=pause_before_begin,
        ), mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or _accepted(),
        ), mock.patch.object(Scheduler, "ensure_env_morning_brief", return_value=None):
            first_tick = scheduler.run_due_jobs()

        with runtime.store.connect() as conn:
            deliveries = list(
                conn.execute("SELECT * FROM scheduled_deliveries ORDER BY delivery_key")
            )
        if (
            pauses != ["Paused Pause Claimed Delivery."]
            or handler_calls != ["built"]
            or sent
            or len(deliveries) != 1
            or deliveries[0]["state"] != "prepared"
            or deliveries[0]["attempt_token"] is not None
            or deliveries[0]["claimed_at"] is not None
            or int(deliveries[0]["attempt_count"]) != 0
            or "durable Telegram outbox: prepared" not in first_tick
        ):
            raise SystemExit(
                f"pause stranded or dispatched the pre-I/O claim: pauses={pauses!r} "
                f"handlers={handler_calls!r} sent={sent!r} "
                f"rows={[dict(item) for item in deliveries]!r} output={first_tick!r}"
            )

        with mock.patch.object(
            scheduler_module,
            "build_goal_nudge",
            side_effect=AssertionError("resume must not replay the completed handler"),
        ), mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or _accepted(),
        ), mock.patch.object(Scheduler, "ensure_env_morning_brief", return_value=None):
            if scheduler.resume_job(str(row["name"])) != "Resumed Pause Claimed Delivery.":
                raise SystemExit("pre-I/O claimed delivery fixture could not resume")
            resumed_tick = scheduler.run_due_jobs()

        final = runtime.store.get_scheduled_delivery(str(deliveries[0]["delivery_key"]))
        if (
            len(sent) != 1
            or not sent[0].endswith("\n\nclaimed pause payload")
            or final is None
            or final["state"] != "accepted"
            or final["payload"] is not None
            or int(final["attempt_count"]) != 1
            or "accepted by Telegram API" not in resumed_tick
        ):
            raise SystemExit(
                f"resume did not drain the released pre-I/O claim once: "
                f"sent={sent!r} row={dict(final) if final else None} output={resumed_tick!r}"
            )


def test_false_tool_result_is_bounded_failed_retry_not_delivery() -> None:
    private_marker = "PRIVATE_MARKER_MUST_NOT_PERSIST"
    refusal = ToolResult(
        "ingest_inbox_notes",
        False,
        private_marker + ("x" * 20000),
        {"status": "refused", "reason": private_marker + ("y" * 20000)},
    )

    with TemporaryDirectory(prefix="jarvis-delivery-false-result-due-") as temp:
        runtime = make_temp_runtime(Path(temp))
        row = _job(runtime, "False Result Due", "inbox_ingest", due=True)
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        sent: list[str] = []
        with mock.patch.object(scheduler_module, "ingest_inbox_notes", return_value=refusal), mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or _accepted(),
        ), mock.patch.object(Scheduler, "ensure_env_morning_brief", return_value=None):
            output = scheduler.run_due_jobs()
        current = next(item for item in runtime.store.list_jobs() if int(item["id"]) == int(row["id"]))
        metadata = json.loads(current["metadata"] or "{}")
        history = metadata.get("run_history") or []
        with runtime.store.connect() as conn:
            delivery_count = int(conn.execute("SELECT COUNT(*) FROM scheduled_deliveries").fetchone()[0])
        if "Failed False Result Due: Job failed: inbox_ingest_failed. Retry scheduled." not in output:
            raise SystemExit(f"false scheduled ToolResult did not retain failure truth: {output!r}")
        for expected in (
            "list scheduled jobs",
            "setup check",
            "storage, model, or connector issue",
            "automatic retry",
        ):
            if expected not in output:
                raise SystemExit(f"false scheduled ToolResult missed guidance {expected!r}: {output!r}")
        if private_marker in output or private_marker in json.dumps(history) or len(json.dumps(history)) > 4096:
            raise SystemExit("false scheduled ToolResult leaked or persisted unbounded refusal content")
        if sent or delivery_count or not history or history[-1].get("status") != "failed":
            raise SystemExit(
                f"false scheduled ToolResult was delivered or recorded as success: sent={sent!r} rows={delivery_count} history={history!r}"
            )
        if current["last_run_at"] is None or current["active_occurrence_key"] is None:
            raise SystemExit(f"false scheduled ToolResult missed bounded retry state: {dict(current)}")

    with TemporaryDirectory(prefix="jarvis-delivery-false-result-now-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _job(runtime, "False Result Now", "inbox_ingest")
        with mock.patch.object(scheduler_module, "ingest_inbox_notes", return_value=refusal):
            result = runtime.registry.get("run_job_now").handler({"name": "False Result Now"})
        if result.ok or "Job failed: inbox_ingest_failed" not in result.output:
            raise SystemExit(f"run_job_now discarded structured handler failure: {result}")
        for expected in ("list scheduled jobs", "setup check", "automatic retry"):
            if expected not in result.output:
                raise SystemExit(f"run_job_now failure missed guidance {expected!r}: {result}")
        if private_marker in result.output or result.metadata.get("runs_scheduled_job"):
            raise SystemExit(f"run_job_now exposed refusal content or claimed success: {result}")


def test_pause_preserves_delivery_truth_after_send_started() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-pause-after-send-") as temp:
        runtime = make_temp_runtime(Path(temp))
        row = _job(runtime, "Pause After Send", "inbox_ingest")
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        plan = _prepare_plan(runtime, scheduler, row, "already crossing the external boundary")
        delivery = _rows(runtime, plan.occurrence_key)[0]
        key = str(delivery["delivery_key"])
        token = "pause-after-send-started"
        if runtime.store.claim_scheduled_delivery(key, token) is None:
            raise SystemExit("could not claim pause-after-send fixture")
        if not runtime.store.begin_scheduled_delivery_send(key, token):
            raise SystemExit("could not establish irreversible send boundary")
        if scheduler.pause_job(str(row["name"])) != "Paused Pause After Send.":
            raise SystemExit("could not pause after the send boundary")
        if not runtime.store.finish_scheduled_delivery(
            key,
            token,
            "accepted",
            telegram_message_id=919,
        ):
            raise SystemExit("pause erased a known post-send outcome")
        accepted = runtime.store.get_scheduled_delivery(key)
        if accepted is None or accepted["state"] != "accepted" or int(accepted["telegram_message_id"]) != 919:
            raise SystemExit(f"pause failed to preserve irreversible delivery truth: {accepted}")


def test_delete_job_terminalizes_pre_io_delivery_before_drain() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-delete-before-drain-") as temp:
        runtime = make_temp_runtime(Path(temp))
        row = _job(runtime, "Delete Before Drain", "inbox_ingest")
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        keys = [f"delete-before-drain-{state}" for state in ("prepared", "claimed", "rejected")]
        for key in keys:
            payload = f"must never send {key}"
            runtime.store.prepare_scheduled_delivery(
                key,
                int(row["id"]),
                str(row["name"]),
                str(row["job_type"]),
                payload,
                _payload_sha256(payload),
            )
        prepared_key, claimed_key, rejected_key = keys
        if runtime.store.claim_scheduled_delivery(claimed_key, "delete-before-drain-claim") is None:
            raise SystemExit("could not establish the pre-I/O claimed deletion fixture")
        rejected_token = "delete-before-drain-rejected"
        if runtime.store.claim_scheduled_delivery(rejected_key, rejected_token) is None:
            raise SystemExit("could not claim the retryable deletion fixture")
        if not runtime.store.begin_scheduled_delivery_send(rejected_key, rejected_token):
            raise SystemExit("could not begin the retryable deletion fixture")
        if not runtime.store.finish_scheduled_delivery(
            rejected_key,
            rejected_token,
            "rejected",
            error_code="retryable_fixture",
        ):
            raise SystemExit("could not establish the retryable rejected deletion fixture")

        if not runtime.store.delete_job(str(row["name"])):
            raise SystemExit("scheduled job deletion unexpectedly reported no change")

        sent: list[str] = []
        with mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or _accepted(),
        ):
            dispatches = [scheduler._dispatch_occurrence(key) for key in keys]
        persisted_rows = [runtime.store.get_scheduled_delivery(key) for key in keys]
        if sent:
            raise SystemExit(f"deleted job's prepared delivery crossed the network boundary: {sent!r}")
        if any(row is None for row in persisted_rows):
            raise SystemExit("job deletion erased the delivery receipt")
        for key, persisted in zip(keys, persisted_rows, strict=True):
            if (
                persisted["job_id"] is not None
                or persisted["state"] != "failed"
                or persisted["error_code"] != "job_deleted_before_send"
                or persisted["payload"] is not None
                or persisted["next_attempt_at"] is not None
                or persisted["attempt_token"] is not None
            ):
                raise SystemExit(
                    f"job deletion did not terminalize pre-I/O work safely: {dict(persisted)}"
                )
            if runtime.store.claim_scheduled_delivery(key, f"deleted-job-reclaim-{key}") is not None:
                raise SystemExit("deleted job's terminalized delivery could be reclaimed")
        if runtime.store.list_ready_scheduled_deliveries(iso(datetime.now()), 10):
            raise SystemExit("deleted job's terminalized delivery remained ready")
        if any(dispatch.state not in {"failed", "uncertain"} for dispatch in dispatches):
            raise SystemExit(f"deleted occurrence reported false delivery success: {dispatches!r}")


def test_delete_job_preserves_in_flight_and_final_delivery_truth() -> None:
    with TemporaryDirectory(prefix="jarvis-delivery-delete-in-flight-") as temp:
        runtime = make_temp_runtime(Path(temp))
        row = _job(runtime, "Delete In Flight", "inbox_ingest")
        job_id = int(row["id"])

        def prepare(label: str) -> str:
            payload = f"{label} payload"
            runtime.store.prepare_scheduled_delivery(
                label,
                job_id,
                str(row["name"]),
                str(row["job_type"]),
                payload,
                _payload_sha256(payload),
            )
            return label

        sending_key = prepare("delete-preserve-sending")
        rejected_key = prepare("delete-terminalize-rejected")
        accepted_key = prepare("delete-preserve-accepted")
        uncertain_key = prepare("delete-preserve-uncertain")

        sending_token = "delete-preserve-sending-token"
        if runtime.store.claim_scheduled_delivery(sending_key, sending_token) is None:
            raise SystemExit("could not claim in-flight deletion fixture")
        if not runtime.store.begin_scheduled_delivery_send(sending_key, sending_token):
            raise SystemExit("could not establish the in-flight send boundary")

        rejected_token = "delete-terminalize-rejected-token"
        if runtime.store.claim_scheduled_delivery(rejected_key, rejected_token) is None:
            raise SystemExit("could not claim orphaned rejected-send fixture")
        if not runtime.store.begin_scheduled_delivery_send(rejected_key, rejected_token):
            raise SystemExit("could not establish orphaned rejected-send boundary")

        accepted_token = "delete-preserve-accepted-token"
        if runtime.store.claim_scheduled_delivery(accepted_key, accepted_token) is None:
            raise SystemExit("could not claim accepted deletion fixture")
        if not runtime.store.begin_scheduled_delivery_send(accepted_key, accepted_token):
            raise SystemExit("could not begin accepted deletion fixture")
        if not runtime.store.finish_scheduled_delivery(
            accepted_key,
            accepted_token,
            "accepted",
            telegram_message_id=920,
        ):
            raise SystemExit("could not establish accepted deletion truth")

        uncertain_token = "delete-preserve-uncertain-token"
        if runtime.store.claim_scheduled_delivery(uncertain_key, uncertain_token) is None:
            raise SystemExit("could not claim uncertain deletion fixture")
        if not runtime.store.begin_scheduled_delivery_send(uncertain_key, uncertain_token):
            raise SystemExit("could not begin uncertain deletion fixture")
        if not runtime.store.finish_scheduled_delivery(
            uncertain_key,
            uncertain_token,
            "uncertain",
            error_code="outcome_unknown",
        ):
            raise SystemExit("could not establish uncertain deletion truth")

        if not runtime.store.delete_job(str(row["name"])):
            raise SystemExit("could not delete job with preserved delivery truth")
        after_delete = {
            key: runtime.store.get_scheduled_delivery(key)
            for key in (sending_key, rejected_key, accepted_key, uncertain_key)
        }
        if any(item is None or item["job_id"] is not None for item in after_delete.values()):
            raise SystemExit(f"job deletion lost orphan receipt linkage truth: {after_delete!r}")
        if after_delete[sending_key]["state"] != "sending":
            raise SystemExit(f"job deletion rewrote in-flight truth: {dict(after_delete[sending_key])}")
        if after_delete[rejected_key]["state"] != "sending":
            raise SystemExit(f"job deletion rewrote rejected in-flight truth: {dict(after_delete[rejected_key])}")
        if (
            after_delete[accepted_key]["state"] != "accepted"
            or int(after_delete[accepted_key]["telegram_message_id"]) != 920
        ):
            raise SystemExit(f"job deletion rewrote accepted truth: {dict(after_delete[accepted_key])}")
        if after_delete[uncertain_key]["state"] != "uncertain":
            raise SystemExit(f"job deletion rewrote uncertain truth: {dict(after_delete[uncertain_key])}")

        if not runtime.store.finish_scheduled_delivery(
            rejected_key,
            rejected_token,
            "rejected",
            error_code="telegram_rejected",
            retry_after_seconds=300,
        ):
            raise SystemExit("orphaned in-flight rejection could not be finalized")
        rejected = runtime.store.get_scheduled_delivery(rejected_key)
        if (
            rejected is None
            or rejected["state"] != "failed"
            or rejected["error_code"] != "job_deleted_before_retry"
            or rejected["payload"] is not None
            or rejected["next_attempt_at"] is not None
        ):
            raise SystemExit(f"orphaned rejection retained retryable content: {dict(rejected) if rejected else None}")

        if not runtime.store.finish_scheduled_delivery(
            sending_key,
            sending_token,
            "accepted",
            telegram_message_id=921,
        ):
            raise SystemExit("in-flight delivery could not record its known result after job deletion")
        finalized = runtime.store.get_scheduled_delivery(sending_key)
        if (
            finalized is None
            or finalized["state"] != "accepted"
            or int(finalized["telegram_message_id"]) != 921
        ):
            raise SystemExit(f"post-deletion in-flight result was not preserved: {finalized}")


def main() -> None:
    original_owner = os.environ.get("JARVIS_OWNER_TELEGRAM")
    os.environ["JARVIS_OWNER_TELEGRAM"] = "owner-smoke"
    try:
        test_ledger_state_flow_and_accepted_payload_is_cleared()
        test_attempt_budget_is_consumed_only_when_send_begins()
        test_terminal_receipt_retention_is_bounded_without_pruning_live_work()
        test_corrupted_persisted_payload_is_scrubbed_before_network_io()
        test_malformed_claim_timestamps_recover_without_stranding_or_replay()
        test_duplicate_occurrence_sends_once_under_concurrent_tickers()
        test_delivered_goal_nudge_preserves_active_reschedule_and_claimed_identity()
        test_active_outbox_occurrence_advances_across_pause_resume()
        test_two_dispatchers_claim_one_prepared_chunk_once()
        test_prepared_before_finalize_survives_restart_and_drains()
        test_transient_rejection_recovers_after_restart_without_duplicate_delivery()
        test_recovered_morning_brief_updates_daily_receipt_metadata()
        test_send_exception_is_uncertain_and_never_auto_replays()
        test_api_acceptance_with_local_receipt_failure_never_replays()
        test_stale_claimed_recovers_but_stale_sending_quarantines_suffix()
        test_rejection_retries_and_long_chunks_are_ordered_deterministically()
        test_partial_chunk_success_is_uncertain_and_blocks_remaining_chunks()
        test_delivery_key_collision_fails_closed()
        test_mid_batch_collision_rolls_back_enqueue_and_job_finalization()
        test_incomplete_occurrence_is_rejected_and_retries_are_bounded()
        test_http_rejections_are_retryable_but_malformed_results_are_uncertain()
        test_outbox_recovery_failure_does_not_starve_due_jobs()
        test_morning_brief_schedule_and_on_demand_alias_send_once()
        test_morning_brief_job_recreation_reuses_daily_receipt()
        test_recreated_morning_brief_recovers_post_finalize_crash()
        test_manual_non_morning_digest_does_not_push()
        test_recent_file_digest_without_watched_dirs_is_quiet_but_configured_digest_delivers()
        test_pause_finalizes_started_run_without_sending_until_resume()
        test_pause_between_delivery_claim_and_send_releases_pre_io_claim()
        test_false_tool_result_is_bounded_failed_retry_not_delivery()
        test_pause_preserves_delivery_truth_after_send_started()
        test_delete_job_terminalizes_pre_io_delivery_before_drain()
        test_delete_job_preserves_in_flight_and_final_delivery_truth()
    finally:
        if original_owner is None:
            os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        else:
            os.environ["JARVIS_OWNER_TELEGRAM"] = original_owner
    print("Scheduler delivery outbox smoke passed")


if __name__ == "__main__":
    main()
