from __future__ import annotations

import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from threading import Barrier, Thread
from unittest import mock

from jarvis_v2.automations import scheduler as scheduler_module
from jarvis_v2.automations.scheduler import Scheduler, _run_marker, iso
from jarvis_v2.memory import store as store_module
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.scripts import live_check


def _row(runtime, job_id: int):
    return next(row for row in runtime.store.list_jobs() if int(row["id"]) == job_id)


def _history(runtime, job_id: int) -> list[dict[str, object]]:
    metadata = json.loads(_row(runtime, job_id)["metadata"] or "{}")
    history = metadata.get("run_history")
    return history if isinstance(history, list) else []


def _claim(runtime, job_id: int, token: str, now: datetime):
    claimed = runtime.store.claim_job(job_id, token, 300, now=now)
    if claimed is None:
        raise SystemExit("run-history fixture could not claim its job")
    return claimed


def _event(scheduler: Scheduler, claimed, now: datetime, status: str, plan=None):
    return scheduler._run_history_event(claimed, now, status, plan)


def test_plain_finalization_and_history_roll_back_together() -> None:
    with TemporaryDirectory(prefix="jarvis-run-history-atomic-plain-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        base = datetime(2042, 1, 2, 3, 4, 5)
        job_id = runtime.store.upsert_job("Atomic Plain", 60, "atomic_plain", iso(base))
        claimed = _claim(runtime, job_id, "atomic-plain-token", base)
        completed = base + timedelta(seconds=1)
        event = _event(scheduler, claimed, completed, "ok")
        expected_event = {
            "occurrence_key": str(claimed["active_occurrence_key"]),
            "date": completed.date().isoformat(),
            "ran_at": _run_marker(completed),
            "status": "ok",
            "job_type": "atomic_plain",
            "schedule_identity_revision": int(
                claimed["active_occurrence_schedule_identity_revision"]
            ),
        }
        if event != expected_event:
            raise SystemExit(f"scheduler emitted a non-schema-3 run event: {event!r}")

        try:
            runtime.store.mark_claimed_job_run(
                job_id,
                "atomic-plain-token",
                _run_marker(completed),
                iso(base + timedelta(hours=1)),
                int(claimed["active_occurrence_schedule_revision"]),
                now=completed,
                run_history_event={**event, "occurrence_key": "morning:" + ("b" * 32)},
                expected_occurrence_key=str(claimed["active_occurrence_key"]),
                source_job_type=str(claimed["job_type"]),
            )
        except ValueError:
            pass
        else:
            raise SystemExit("a non-Morning job accepted a Morning occurrence identity")

        with mock.patch.object(
            store_module,
            "_job_metadata_with_run_event",
            side_effect=RuntimeError("injected metadata failure"),
        ):
            try:
                runtime.store.mark_claimed_job_run(
                    job_id,
                    "atomic-plain-token",
                    _run_marker(completed),
                    iso(base + timedelta(hours=1)),
                    int(claimed["active_occurrence_schedule_revision"]),
                    now=completed,
                    run_history_event=event,
                    expected_occurrence_key=str(claimed["active_occurrence_key"]),
                    source_job_type=str(claimed["job_type"]),
                )
            except RuntimeError:
                pass
            else:
                raise SystemExit("injected run-history failure should roll back finalization")

        rolled_back = _row(runtime, job_id)
        if rolled_back["last_run_at"] is not None or _history(runtime, job_id):
            raise SystemExit(f"plain finalization escaped its history transaction: {dict(rolled_back)}")
        if rolled_back["lease_token"] != "atomic-plain-token":
            raise SystemExit("rolled-back plain finalization lost its live lease")

        if not runtime.store.mark_claimed_job_run(
            job_id,
            "atomic-plain-token",
            _run_marker(completed),
            iso(base + timedelta(hours=1)),
            int(claimed["active_occurrence_schedule_revision"]),
            now=completed,
            run_history_event=event,
            expected_occurrence_key=str(claimed["active_occurrence_key"]),
            source_job_type=str(claimed["job_type"]),
        ):
            raise SystemExit("plain occurrence did not finalize after rollback")
        history = _history(runtime, job_id)
        if history != [expected_event] or _row(runtime, job_id)["last_run_at"] != event["ran_at"]:
            raise SystemExit(f"plain occurrence/history were not committed together: {history!r}")


def test_outbox_finalization_is_pending_then_converges_once() -> None:
    private_marker = "/private/tmp/DELIVERY_PAYLOAD_MUST_NOT_ENTER_HISTORY"
    with TemporaryDirectory(prefix="jarvis-run-history-atomic-outbox-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        base = datetime(2042, 2, 3, 4, 5, 6)
        job_id = runtime.store.upsert_job("Atomic Outbox", 1440, "inbox_ingest", iso(base))
        claimed = _claim(runtime, job_id, "atomic-outbox-token", base)
        plan = scheduler._delivery_plan(claimed, private_marker, base)
        completed = base + timedelta(seconds=1)
        event = _event(scheduler, claimed, completed, "pending", plan)

        with mock.patch.object(
            store_module,
            "_job_metadata_with_run_event",
            side_effect=RuntimeError("injected outbox metadata failure"),
        ):
            try:
                runtime.store.complete_claimed_job_with_scheduled_deliveries(
                    job_id,
                    "atomic-outbox-token",
                    _run_marker(completed),
                    iso(base + timedelta(days=1)),
                    plan.occurrence_key,
                    list(plan.chunks),
                    str(claimed["name"]),
                    str(claimed["job_type"]),
                    int(claimed["active_occurrence_schedule_revision"]),
                    now=completed,
                    run_history_event=event,
                    expected_occurrence_key=str(claimed["active_occurrence_key"]),
                )
            except RuntimeError:
                pass
            else:
                raise SystemExit("injected outbox history failure should roll back finalization")
        if (
            _row(runtime, job_id)["last_run_at"] is not None
            or runtime.store.list_scheduled_deliveries_for_occurrence(plan.occurrence_key)
            or _history(runtime, job_id)
        ):
            raise SystemExit("outbox/history finalization escaped its rollback boundary")

        if not runtime.store.complete_claimed_job_with_scheduled_deliveries(
            job_id,
            "atomic-outbox-token",
            _run_marker(completed),
            iso(base + timedelta(days=1)),
            plan.occurrence_key,
            list(plan.chunks),
            str(claimed["name"]),
            str(claimed["job_type"]),
            int(claimed["active_occurrence_schedule_revision"]),
            now=completed,
            run_history_event=event,
            expected_occurrence_key=str(claimed["active_occurrence_key"]),
        ):
            raise SystemExit("outbox occurrence did not finalize atomically")
        pending = _history(runtime, job_id)
        if pending != [event] or private_marker in json.dumps(pending):
            raise SystemExit(f"outbox finalization leaked content or missed pending custody: {pending!r}")

        sent: list[str] = []
        with mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or {"ok": True, "result": {"message_id": 7001}},
        ):
            dispatch = scheduler._dispatch_occurrence(plan.occurrence_key)
        if dispatch.state != "accepted" or _history(runtime, job_id)[0]["status"] != "pending":
            raise SystemExit("accepted API receipt should precede local history convergence")
        Scheduler(runtime.store, runtime.vault, runtime.config)._recover_and_drain_deliveries()
        converged = _history(runtime, job_id)
        if len(sent) != 1 or len(converged) != 1 or converged[0]["status"] != "ok":
            raise SystemExit(f"accepted occurrence did not converge exactly once: {converged!r}")
        if converged[0]["occurrence_key"] != event["occurrence_key"]:
            raise SystemExit("accepted convergence changed occurrence identity")
        scheduler._record_delivery_status(
            plan.occurrence_key,
            scheduler_module._DeliveryDispatch("sending", "stale concurrent observer"),
            completed,
        )
        metadata = json.loads(_row(runtime, job_id)["metadata"] or "{}")
        if metadata.get("last_delivery_status") != "accepted":
            raise SystemExit(f"transient observer regressed accepted delivery truth: {metadata!r}")
        scheduler._record_delivery_status(
            plan.occurrence_key,
            scheduler_module._DeliveryDispatch("rejected", "stale retryable rejection"),
            completed,
        )
        metadata = json.loads(_row(runtime, job_id)["metadata"] or "{}")
        if metadata.get("last_delivery_status") != "accepted" or _history(runtime, job_id)[0]["status"] != "ok":
            raise SystemExit(f"stale rejection regressed accepted delivery truth: {metadata!r}")


def test_inconsistent_chunk_counts_never_terminalize_success() -> None:
    with TemporaryDirectory(prefix="jarvis-run-history-inconsistent-chunks-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        base = datetime(2042, 2, 3, 5, 0, 0)
        job_id = runtime.store.upsert_job("Malformed Outbox", 1440, "inbox_ingest", iso(base))
        claimed = _claim(runtime, job_id, "malformed-outbox-token", base)
        plan = scheduler._delivery_plan(claimed, "x" * 8000, base)
        if len(plan.chunks) < 2:
            raise SystemExit("malformed outbox fixture did not produce multiple chunks")
        completed = base + timedelta(seconds=1)
        if not runtime.store.complete_claimed_job_with_scheduled_deliveries(
            job_id,
            "malformed-outbox-token",
            _run_marker(completed),
            iso(base + timedelta(days=1)),
            plan.occurrence_key,
            list(plan.chunks),
            str(claimed["name"]),
            str(claimed["job_type"]),
            int(claimed["active_occurrence_schedule_revision"]),
            now=completed,
            run_history_event=_event(scheduler, claimed, completed, "pending", plan),
            expected_occurrence_key=str(claimed["active_occurrence_key"]),
        ):
            raise SystemExit("malformed outbox fixture did not finalize")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_deliveries SET chunk_count = 1 "
                "WHERE occurrence_key = ? AND chunk_index = 0",
                (plan.occurrence_key,),
            )
            conn.execute(
                "UPDATE scheduled_deliveries SET state = 'accepted', payload = NULL "
                "WHERE occurrence_key = ?",
                (plan.occurrence_key,),
            )
        if any(
            str(row["occurrence_key"]) == plan.occurrence_key
            for row in runtime.store.list_scheduled_delivery_history_reconciliations()
        ):
            raise SystemExit("inconsistent accepted chunk counts entered terminal reconciliation")
        sent: list[str] = []
        with mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or {"ok": True},
        ):
            dispatch = scheduler._dispatch_occurrence(plan.occurrence_key)
        if dispatch.state != "failed" or sent:
            raise SystemExit(f"inconsistent chunk set reached Telegram dispatch: {dispatch!r}")
        scheduler._record_delivery_status(plan.occurrence_key, dispatch, completed)
        history = _history(runtime, job_id)
        if len(history) != 1 or history[0]["status"] != "failed":
            raise SystemExit(f"inconsistent chunk set terminalized as success: {history!r}")


def test_stale_sending_recovery_never_records_ok() -> None:
    with TemporaryDirectory(prefix="jarvis-run-history-uncertain-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        base = datetime.now() - timedelta(hours=1)
        job_id = runtime.store.upsert_job("Uncertain Outbox", 1440, "inbox_ingest", iso(base))
        claimed = _claim(runtime, job_id, "uncertain-job-token", base)
        plan = scheduler._delivery_plan(claimed, "ambiguous send", base)
        completed = base + timedelta(seconds=1)
        if not runtime.store.complete_claimed_job_with_scheduled_deliveries(
            job_id,
            "uncertain-job-token",
            _run_marker(completed),
            iso(base + timedelta(days=1)),
            plan.occurrence_key,
            list(plan.chunks),
            str(claimed["name"]),
            str(claimed["job_type"]),
            int(claimed["active_occurrence_schedule_revision"]),
            now=completed,
            run_history_event=_event(scheduler, claimed, completed, "pending", plan),
            expected_occurrence_key=str(claimed["active_occurrence_key"]),
        ):
            raise SystemExit("uncertain occurrence did not finalize")
        delivery_key = str(plan.chunks[0]["delivery_key"])
        if runtime.store.claim_scheduled_delivery(delivery_key, "uncertain-attempt") is None:
            raise SystemExit("uncertain fixture could not claim delivery")
        if not runtime.store.begin_scheduled_delivery_send(delivery_key, "uncertain-attempt"):
            raise SystemExit("uncertain fixture did not cross the send boundary")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_deliveries SET sending_at = ? WHERE delivery_key = ?",
                ("2000-01-01T00:00:00Z", delivery_key),
            )
        scheduler._recover_and_drain_deliveries()
        history = _history(runtime, job_id)
        if len(history) != 1 or history[0]["status"] != "failed":
            raise SystemExit(f"uncertain recovery retained or invented success: {history!r}")


def test_reconfiguration_preserves_identity_and_tampering_fails_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-run-history-identity-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        base = datetime(2042, 3, 1, 9, 0, 0)
        job_id = runtime.store.upsert_job("Identity Custody", 240, "inbox_ingest", iso(base))
        claimed = _claim(runtime, job_id, "identity-token", base)
        occurrence_type = str(claimed["active_occurrence_job_type"])
        occurrence_revision = int(claimed["active_occurrence_schedule_identity_revision"])
        reconfigured_due = base + timedelta(hours=2)
        runtime.store.upsert_job(
            "Identity Custody",
            60,
            "recent_file_digest",
            iso(reconfigured_due),
        )
        reconfigured = _row(runtime, job_id)
        completed = base + timedelta(seconds=1)
        event = _event(scheduler, reconfigured, completed, "ok")
        if (
            event["job_type"] != occurrence_type
            or event["schedule_identity_revision"] != occurrence_revision
            or reconfigured["job_type"] != "recent_file_digest"
            or int(reconfigured["schedule_identity_revision"]) == occurrence_revision
        ):
            raise SystemExit(f"reconfiguration reinterpreted claimed run evidence: {event!r}")

        poisoned_events = (
            {**event, "job_type": "recent_file_digest"},
            {**event, "schedule_identity_revision": occurrence_revision + 1},
        )
        for poisoned in poisoned_events:
            if runtime.store.mark_claimed_job_run(
                job_id,
                "identity-token",
                _run_marker(completed),
                iso(base + timedelta(hours=4)),
                int(claimed["active_occurrence_schedule_revision"]),
                now=completed,
                run_history_event=poisoned,
                expected_occurrence_key=str(claimed["active_occurrence_key"]),
                source_job_type=occurrence_type,
            ):
                raise SystemExit(f"store accepted run evidence with the wrong identity: {poisoned!r}")
        after_poison = _row(runtime, job_id)
        if (
            after_poison["last_run_at"] is not None
            or _history(runtime, job_id)
            or after_poison["lease_token"] != "identity-token"
        ):
            raise SystemExit("rejected identity evidence mutated the claimed occurrence")

        if not runtime.store.mark_claimed_job_run(
            job_id,
            "identity-token",
            _run_marker(completed),
            iso(base + timedelta(hours=4)),
            int(claimed["active_occurrence_schedule_revision"]),
            now=completed,
            run_history_event=event,
            expected_occurrence_key=str(claimed["active_occurrence_key"]),
            source_job_type=occurrence_type,
        ):
            raise SystemExit("immutable pre-reconfiguration evidence did not finalize")
        finalized = _row(runtime, job_id)
        if (
            _history(runtime, job_id) != [event]
            or finalized["job_type"] != "recent_file_digest"
            or finalized["next_run_at"] != iso(reconfigured_due)
        ):
            raise SystemExit(f"old evidence overwrote or adopted the new schedule: {dict(finalized)!r}")


def test_legacy_occurrence_retires_with_explicit_nonproof_identity() -> None:
    with TemporaryDirectory(prefix="jarvis-run-history-legacy-retire-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        base = datetime(2042, 3, 2, 8, 0, 0)
        job_id = runtime.store.upsert_job("Legacy Retire", 60, "daily_brief", iso(base))
        legacy_key = "scheduler-occurrence:v1:" + ("c" * 64)
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
        claimed = _claim(runtime, job_id, "legacy-retire-token", base)
        if (
            claimed["active_occurrence_job_type"] != "legacy_unknown"
            or claimed["active_occurrence_schedule_identity_revision"] != -1
        ):
            raise SystemExit(f"legacy occurrence did not receive its explicit sentinel: {dict(claimed)!r}")
        completed = base + timedelta(seconds=1)
        event = _event(scheduler, claimed, completed, "skipped")
        if event["job_type"] != "legacy_unknown" or event["schedule_identity_revision"] != -1:
            raise SystemExit(f"legacy receipt lost its nonproof identity: {event!r}")
        if not runtime.store.mark_claimed_job_run(
            job_id,
            "legacy-retire-token",
            _run_marker(completed),
            iso(base + timedelta(hours=1)),
            -1,
            now=completed,
            run_history_event=event,
            expected_occurrence_key=legacy_key,
            source_job_type="legacy_unknown",
        ):
            raise SystemExit("legacy occurrence could not retire through the typed history path")
        finalized = _row(runtime, job_id)
        if finalized["lease_token"] is not None or _history(runtime, job_id) != [event]:
            raise SystemExit(f"legacy occurrence retirement did not converge exactly once: {dict(finalized)!r}")


def test_partially_migrated_occurrence_retires_without_rebinding_identity() -> None:
    with TemporaryDirectory(prefix="jarvis-run-history-partial-legacy-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        base = datetime(2042, 3, 2, 8, 30, 0)
        job_id = runtime.store.upsert_job("Partial Legacy", 60, "daily_brief", iso(base))
        current = _row(runtime, job_id)
        legacy_key = "scheduler-occurrence:v1:" + ("d" * 64)
        original_started_at = iso(base - timedelta(minutes=1))
        with runtime.store.connect() as conn:
            conn.execute(
                """
                UPDATE scheduled_jobs
                SET active_occurrence_key = ?,
                    active_occurrence_schedule_revision = ?,
                    active_occurrence_next_run_at = ?,
                    active_occurrence_started_at = ?,
                    active_occurrence_job_type = ?,
                    active_occurrence_schedule_identity_revision = NULL
                WHERE id = ?
                """,
                (
                    legacy_key,
                    int(current["schedule_revision"]),
                    str(current["next_run_at"]),
                    original_started_at,
                    "daily_brief",
                    job_id,
                ),
            )
        claimed = _claim(runtime, job_id, "partial-legacy-token", base)
        if (
            claimed["active_occurrence_job_type"] != "legacy_unknown"
            or claimed["active_occurrence_schedule_identity_revision"] != -1
            or claimed["active_occurrence_started_at"] != original_started_at
        ):
            raise SystemExit(
                f"partially migrated occurrence was rebound or lost custody: {dict(claimed)!r}"
            )
        completed = base + timedelta(seconds=1)
        event = _event(scheduler, claimed, completed, "skipped")
        if not runtime.store.mark_claimed_job_run(
            job_id,
            "partial-legacy-token",
            _run_marker(completed),
            iso(base + timedelta(hours=1)),
            int(claimed["active_occurrence_schedule_revision"]),
            now=completed,
            run_history_event=event,
            expected_occurrence_key=legacy_key,
            source_job_type="legacy_unknown",
        ):
            raise SystemExit("partially migrated occurrence could not retire")
        if _history(runtime, job_id) != [event] or _row(runtime, job_id)["lease_token"] is not None:
            raise SystemExit("partially migrated occurrence did not retire exactly once")


def test_schema_two_history_is_archived_without_invented_provenance() -> None:
    with TemporaryDirectory(prefix="jarvis-run-history-schema-two-") as temp:
        runtime = make_temp_runtime(Path(temp))
        base = datetime(2042, 3, 2, 8, 45, 0)
        job_id = runtime.store.upsert_job("History Migration", 60, "daily_brief", iso(base))
        pending_key = "scheduled:scheduler-occurrence:v1:" + ("8" * 64)
        old_history = [
            {
                "date": "2042-03-01",
                "ran_at": "2042-03-01T08:45:00",
                "status": "ok",
                "private": "/\x55sers/example/private SHOULD NOT SURVIVE",
            },
            {
                "occurrence_key": "legacy:" + ("e" * 64),
                "date": "2042-03-02",
                "ran_at": "2042-03-02T07:45:00",
                "status": "failed",
                "job_type": "daily_brief",
                "schedule_identity_revision": 99,
            },
            {
                "occurrence_key": pending_key,
                "date": "2042-03-02",
                "ran_at": "2042-03-02T08:00:00",
                "status": "pending",
            },
            {"date": "malformed SHOULD NOT SURVIVE"},
        ]
        current_run = iso(base)
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_jobs SET last_run_at = ?, metadata = ? WHERE id = ?",
                (
                    current_run,
                    json.dumps({"run_history_schema": 2, "run_history": old_history}),
                    job_id,
                ),
            )
        event = {
            "occurrence_key": "legacy:" + ("f" * 64),
            "date": base.date().isoformat(),
            "ran_at": current_run,
            "status": "ok",
            "job_type": "daily_brief",
            "schedule_identity_revision": 1,
        }
        if not runtime.store.append_job_run_history_event(
            job_id,
            event,
            expected_last_run_at=current_run,
        ):
            raise SystemExit("schema-two fixture could not append its typed event")
        metadata = json.loads(_row(runtime, job_id)["metadata"] or "{}")
        legacy = metadata.get("run_history_legacy")
        if metadata.get("run_history") != [event] or metadata.get("run_history_schema") != 3:
            raise SystemExit(f"schema-two history entered current proof: {metadata!r}")
        if metadata.get("run_history_legacy_schema") != 2 or not isinstance(legacy, list):
            raise SystemExit(f"schema-two audit history was not retained: {metadata!r}")
        if len(legacy) != 3 or {item.get("status") for item in legacy} != {"ok", "failed", "pending"}:
            raise SystemExit(f"schema-two success/failure evidence was not preserved: {legacy!r}")
        serialized = json.dumps(legacy, sort_keys=True)
        for forbidden in ("job_type", "schedule_identity_revision", "private", "SHOULD NOT SURVIVE"):
            if forbidden in serialized:
                raise SystemExit(f"legacy archive invented or leaked {forbidden!r}: {legacy!r}")
        receipt_now = iso(base + timedelta(seconds=2))
        with runtime.store.connect() as conn:
            conn.execute(
                """
                INSERT INTO scheduled_deliveries (
                    delivery_key, occurrence_key, chunk_index, chunk_count,
                    job_id, source_job_id, job_name, job_type,
                    payload, payload_sha256, state, attempt_count,
                    next_attempt_at, created_at, updated_at
                ) VALUES (?, ?, 0, 1, ?, ?, ?, ?, NULL, ?, 'accepted', 1, NULL, ?, ?)
                """,
                (
                    "7" * 64,
                    pending_key,
                    job_id,
                    job_id,
                    "History Migration",
                    "daily_brief",
                    "6" * 64,
                    receipt_now,
                    receipt_now,
                ),
            )
        reconciliations = runtime.store.list_scheduled_delivery_history_reconciliations()
        if not any(str(row["occurrence_key"]) == pending_key for row in reconciliations):
            raise SystemExit("terminal outbox truth did not find archived pending history")
        if not runtime.store.converge_job_run_history(job_id, pending_key, "ok"):
            raise SystemExit("archived schema-two pending history did not converge")
        metadata = json.loads(_row(runtime, job_id)["metadata"] or "{}")
        legacy = metadata.get("run_history_legacy") or []
        pending = [item for item in legacy if item.get("occurrence_key") == pending_key]
        if len(pending) != 1 or pending[0].get("status") != "ok":
            raise SystemExit(f"archived pending history retained stale truth: {legacy!r}")
        if runtime.store.list_scheduled_delivery_history_reconciliations():
            raise SystemExit("terminal archived history remained queued for reconciliation")


def test_occurrence_history_is_unique_bounded_and_merge_safe() -> None:
    with TemporaryDirectory(prefix="jarvis-run-history-bounded-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        base = datetime(2042, 3, 2, 9, 0, 0)
        job_id = runtime.store.upsert_job("Bounded History", 60, "bounded_history", iso(base))
        expected_events: list[dict[str, object]] = []
        for index in range(70):
            now = base + timedelta(hours=index)
            runtime.store.upsert_job("Bounded History", 60, "bounded_history", iso(now))
            token = f"bounded-{index}"
            claimed = _claim(runtime, job_id, token, now)
            event = _event(scheduler, claimed, now, "ok")
            expected_events.append(event)
            if not runtime.store.mark_claimed_job_run(
                job_id,
                token,
                _run_marker(now),
                iso(now + timedelta(hours=1)),
                int(claimed["active_occurrence_schedule_revision"]),
                now=now,
                run_history_event=event,
                expected_occurrence_key=str(claimed["active_occurrence_key"]),
                source_job_type=str(claimed["job_type"]),
            ):
                raise SystemExit(f"bounded occurrence {index} did not finalize")

        target_key = str(expected_events[-1]["occurrence_key"])
        start = Barrier(5)

        def merge(index: int) -> None:
            start.wait(timeout=5)
            runtime.store.merge_job_metadata(job_id, {f"parallel_{index}": index})

        def converge() -> None:
            start.wait(timeout=5)
            runtime.store.converge_job_run_history(job_id, target_key, "ok")

        threads = [Thread(target=merge, args=(index,)) for index in range(4)] + [Thread(target=converge)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        if any(thread.is_alive() for thread in threads):
            raise SystemExit("bounded history concurrency fixture left a worker running")

        metadata = json.loads(_row(runtime, job_id)["metadata"] or "{}")
        history = metadata.get("run_history") or []
        keys = [event.get("occurrence_key") for event in history]
        if (
            metadata.get("run_history_schema") != 3
            or history != expected_events[-64:]
            or len(set(keys)) != 64
        ):
            raise SystemExit(f"occurrence history was duplicated, reordered, or unbounded: {keys!r}")
        if any(metadata.get(f"parallel_{index}") != index for index in range(4)):
            raise SystemExit(f"concurrent metadata merges were lost: {metadata!r}")
        if runtime.store.converge_job_run_history(
            job_id,
            str(expected_events[0]["occurrence_key"]),
            "failed",
        ):
            raise SystemExit("an evicted occurrence should not displace bounded recent history")


def test_live_check_cross_checks_delivery_truth_and_schema() -> None:
    today = datetime(2042, 4, 8, 10, 0, 0)
    uncertain_key = "scheduled:scheduler-occurrence:v1:" + ("a" * 64)
    accepted_key = "scheduled:scheduler-occurrence:v1:" + ("b" * 64)
    missing_key = "scheduled:scheduler-occurrence:v1:" + ("c" * 64)

    def key(label: str) -> str:
        return "legacy:" + hashlib.sha256(label.encode("utf-8")).hexdigest()

    def run_event(
        label: str,
        ran_at: datetime,
        job_type: str,
        schedule_identity_revision: int,
    ) -> dict[str, object]:
        return {
            "occurrence_key": key(label),
            "date": ran_at.date().isoformat(),
            "ran_at": iso(ran_at),
            "status": "ok",
            "job_type": job_type,
            "schedule_identity_revision": schedule_identity_revision,
        }

    rows: list[dict[str, object]] = []
    expected_schedules = list(live_check.SCHEDULED_JOB_EXPECTED_INTERVALS.items())
    for job_index, (job_type, interval_minutes) in enumerate(expected_schedules):
        identity_revision = 20 + job_index
        first_run = today - timedelta(days=7) + timedelta(minutes=interval_minutes)
        events = []
        cursor = first_run
        offset = 0
        while cursor <= today:
            events.append(
                run_event(
                    f"{job_index}:{offset}",
                    cursor,
                    job_type,
                    identity_revision,
                )
            )
            cursor += timedelta(minutes=interval_minutes)
            offset += 1
        rows.append(
            {
                "id": job_index + 1,
                "enabled": 1,
                "interval_minutes": interval_minutes,
                "job_type": job_type,
                "schedule_identity_revision": identity_revision,
                "metadata": json.dumps({"run_history_schema": 3, "run_history": events}),
            }
        )

    def check(
        fixtures: list[dict[str, object]],
        *,
        checked_at: datetime = today,
    ) -> tuple[bool | None, str]:
        def receipt(
            *,
            state: str,
            chunk_index: int = 0,
            chunk_count: int = 1,
        ) -> dict[str, object]:
            job_id = int(fixtures[inbox_index]["id"])
            return {
                "job_id": job_id,
                "source_job_id": job_id,
                "job_type": "inbox_ingest",
                "state": state,
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
            }

        class TruthStore:
            def __init__(self, _db_path: object) -> None:
                pass

            def list_jobs(self):
                return fixtures

            def list_scheduled_deliveries_for_occurrence(self, occurrence_key: str):
                if occurrence_key == uncertain_key:
                    return [receipt(state="uncertain")]
                if occurrence_key == accepted_key:
                    return [
                        receipt(state="accepted", chunk_index=index, chunk_count=2)
                        for index in range(2)
                    ]
                return []

        with mock.patch("jarvis_v2.memory.store.MemoryStore", TruthStore):
            return live_check._check_scheduled_job_streak(
                SimpleNamespace(db_path="unused"),
                now=checked_at,
            )

    inbox_index = [job_type for job_type, _interval in expected_schedules].index("inbox_ingest")
    events, invalid = live_check._job_history_events(rows[inbox_index])
    if len(events) != 42 or invalid != 0:
        raise SystemExit("live check could not retain and parse 42 four-hour Inbox Ingest runs")
    ok, detail = check(rows)
    if ok is not True or "7 consecutive complete day(s) proven" not in detail:
        raise SystemExit(f"42 Inbox Ingest occurrences did not prove seven days: {ok!r} {detail!r}")

    uncertain_rows = json.loads(json.dumps(rows))
    uncertain_metadata = json.loads(uncertain_rows[inbox_index]["metadata"])
    uncertain_metadata["run_history"][-1]["occurrence_key"] = uncertain_key
    uncertain_rows[inbox_index]["metadata"] = json.dumps(uncertain_metadata)
    ok, detail = check(uncertain_rows)
    if ok is not False or "failed scheduled-job run event" not in detail:
        raise SystemExit(f"live check trusted an uncertain delivery receipt: {ok!r} {detail!r}")

    accepted_rows = json.loads(json.dumps(rows))
    accepted_metadata = json.loads(accepted_rows[inbox_index]["metadata"])
    accepted_metadata["run_history"][-1].update(
        {"occurrence_key": accepted_key, "status": "pending"}
    )
    accepted_rows[inbox_index]["metadata"] = json.dumps(accepted_metadata)
    ok, detail = check(accepted_rows)
    if ok is not True:
        raise SystemExit(f"accepted delivery truth did not promote pending metadata: {ok!r} {detail!r}")

    class PartialTruthStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def list_jobs(self):
            return accepted_rows

        def list_scheduled_deliveries_for_occurrence(self, occurrence_key: str):
            if occurrence_key != accepted_key:
                return []
            job_id = int(accepted_rows[inbox_index]["id"])
            return [
                {
                    "job_id": job_id,
                    "source_job_id": job_id,
                    "job_type": "inbox_ingest",
                    "state": "accepted",
                    "chunk_index": 0,
                    "chunk_count": 2,
                }
            ]

    with mock.patch("jarvis_v2.memory.store.MemoryStore", PartialTruthStore):
        ok, detail = live_check._check_scheduled_job_streak(
            SimpleNamespace(db_path="unused"),
            now=today,
        )
    if ok is not False or "invalid run-history event" not in detail:
        raise SystemExit(f"partial accepted chunk set overclaimed delivery truth: {ok!r} {detail!r}")

    missing_rows = json.loads(json.dumps(rows))
    missing_metadata = json.loads(missing_rows[inbox_index]["metadata"])
    missing_metadata["run_history"][-1]["occurrence_key"] = missing_key
    missing_rows[inbox_index]["metadata"] = json.dumps(missing_metadata)
    ok, detail = check(missing_rows)
    if ok is not False or "invalid run-history event" not in detail:
        raise SystemExit(f"live check trusted delivery-backed history without receipts: {ok!r} {detail!r}")

    duplicate = dict(rows[1])
    duplicate_metadata = json.loads(str(duplicate["metadata"]))
    duplicate_metadata["run_history"].append(dict(duplicate_metadata["run_history"][-1]))
    duplicate["metadata"] = json.dumps(duplicate_metadata)
    events, invalid = live_check._job_history_events(duplicate)
    if len(events) != 7 or invalid != 1:
        raise SystemExit("live check accepted duplicate schema-3 occurrence history")

    old_schema = dict(rows[1])
    old_metadata = json.loads(str(old_schema["metadata"]))
    old_metadata["run_history_schema"] = 2
    old_schema["metadata"] = json.dumps(old_metadata)
    events, invalid = live_check._job_history_events(old_schema)
    if events or invalid != 1:
        raise SystemExit("live check accepted old-schema run-history evidence")

    typeless = dict(rows[1])
    typeless_metadata = json.loads(str(typeless["metadata"]))
    typeless_event = dict(typeless_metadata["run_history"][0])
    typeless_event.pop("job_type")
    typeless_event.pop("schedule_identity_revision")
    typeless_metadata["run_history"] = [typeless_event]
    typeless["metadata"] = json.dumps(typeless_metadata)
    events, invalid = live_check._job_history_events(typeless)
    if events or invalid != 1:
        raise SystemExit("live check accepted type-less schema-3 run-history evidence")

    reconfigured = json.loads(json.dumps(rows))
    reconfigured[inbox_index]["job_type"] = "inbox_ingest_reconfigured"
    reconfigured[inbox_index]["schedule_identity_revision"] = int(
        reconfigured[inbox_index]["schedule_identity_revision"]
    ) + 1
    ok, detail = check(reconfigured)
    if ok is not None or "schedule definition(s) missing or changed" not in detail:
        raise SystemExit(f"live check reinterpreted pre-reconfiguration evidence: {ok!r} {detail!r}")

    arbitrary = json.loads(json.dumps(rows))
    for index, row in enumerate(arbitrary):
        row["job_type"] = f"arbitrary_annual_{index}"
        row["interval_minutes"] = 525600
    ok, detail = check(arbitrary)
    if ok is not None or "schedule definition(s) missing or changed" not in detail:
        raise SystemExit(f"arbitrary annual inventory overclaimed scheduler proof: {ok!r} {detail!r}")

    local_timezone = datetime.now().astimezone().tzinfo
    aware_now = today.replace(tzinfo=local_timezone)
    ok, detail = check(rows, checked_at=aware_now)
    if ok is not True:
        raise SystemExit(f"timezone-aware proof check crashed or lost coverage: {ok!r} {detail!r}")


def test_run_history_dates_are_exact_and_timezone_stable() -> None:
    aware_event = {
        "occurrence_key": "legacy:" + ("9" * 64),
        "date": "2042-04-08",
        "ran_at": "2042-04-08T18:00:00+00:00",
        "status": "ok",
        "job_type": "daily_brief",
        "schedule_identity_revision": 7,
    }
    if store_module._scheduled_job_run_event(aware_event) != aware_event:
        raise SystemExit("store changed a valid timezone-aware run event")
    row = {
        "metadata": json.dumps({"run_history_schema": 3, "run_history": [aware_event]})
    }
    events, invalid = live_check._job_history_events(row)
    if len(events) != 1 or invalid != 0 or events[0][1].isoformat() != "2042-04-08":
        raise SystemExit(f"live reader rejected or shifted encoded timezone evidence: {events!r}")

    malformed = {**aware_event, "date": "2042-04-08T00:00:00"}
    try:
        store_module._scheduled_job_run_event(malformed)
    except ValueError:
        pass
    else:
        raise SystemExit("store accepted a datetime-shaped run-history date")
    malformed_row = {
        "metadata": json.dumps({"run_history_schema": 3, "run_history": [malformed]})
    }
    events, invalid = live_check._job_history_events(malformed_row)
    if events or invalid != 1:
        raise SystemExit("live reader accepted a datetime-shaped run-history date")


def main() -> None:
    test_plain_finalization_and_history_roll_back_together()
    test_outbox_finalization_is_pending_then_converges_once()
    test_inconsistent_chunk_counts_never_terminalize_success()
    test_stale_sending_recovery_never_records_ok()
    test_reconfiguration_preserves_identity_and_tampering_fails_closed()
    test_legacy_occurrence_retires_with_explicit_nonproof_identity()
    test_partially_migrated_occurrence_retires_without_rebinding_identity()
    test_schema_two_history_is_archived_without_invented_provenance()
    test_occurrence_history_is_unique_bounded_and_merge_safe()
    test_live_check_cross_checks_delivery_truth_and_schema()
    test_run_history_dates_are_exact_and_timezone_stable()
    print("scheduler run-history custody smoke passed")


if __name__ == "__main__":
    main()
