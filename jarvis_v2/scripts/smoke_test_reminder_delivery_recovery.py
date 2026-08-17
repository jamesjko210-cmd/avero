"""Local-only smoke coverage for durable Telegram reminder delivery."""

from __future__ import annotations

import json
import os
import stat
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

from jarvis_v2.automations import reminders as rem
from jarvis_v2.automations.telegram_control import TelegramCommandBridge


OWNER = "10001"


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@contextmanager
def _isolated_store() -> Iterator[Path]:
    old_module_file = rem.REMINDERS_FILE
    old_env_file = os.environ.get("JARVIS_REMINDERS_FILE")
    old_owner = os.environ.get("JARVIS_OWNER_TELEGRAM")
    with TemporaryDirectory(prefix="jarvis-reminder-delivery-") as temp_dir:
        reminders_file = Path(temp_dir) / "reminders.json"
        try:
            os.environ["JARVIS_REMINDERS_FILE"] = str(reminders_file)
            os.environ["JARVIS_OWNER_TELEGRAM"] = OWNER
            rem.REMINDERS_FILE = reminders_file
            yield reminders_file
        finally:
            rem.REMINDERS_FILE = old_module_file
            if old_env_file is None:
                os.environ.pop("JARVIS_REMINDERS_FILE", None)
            else:
                os.environ["JARVIS_REMINDERS_FILE"] = old_env_file
            if old_owner is None:
                os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
            else:
                os.environ["JARVIS_OWNER_TELEGRAM"] = old_owner


def _write_rows(reminders_file: Path, rows: list[dict]) -> None:
    reminders_file.write_text(json.dumps(rows), encoding="utf-8")


def _rows(reminders_file: Path) -> list[dict]:
    return json.loads(reminders_file.read_text(encoding="utf-8"))


def _bridge(send_func) -> TelegramCommandBridge:
    return TelegramCommandBridge(
        runtime_factory=lambda: None,
        send_func=send_func,
        fetch_func=lambda _token, _offset, _timeout: [],
    )


def _accepted(message_id: int) -> dict:
    return {"ok": True, "result": {"message_id": message_id}}


def test_legacy_id_migration_persists_unique_ids() -> None:
    with _isolated_store() as reminders_file:
        duplicate_id = str(uuid.uuid4())
        due = time.time() + 60
        _write_rows(
            reminders_file,
            [
                {"due": due, "message": "same legacy row", "chat_id": OWNER},
                {"due": due, "message": "same legacy row", "chat_id": OWNER},
                {"id": duplicate_id, "due": due, "message": "duplicate id one", "chat_id": OWNER},
                {"id": duplicate_id, "due": due, "message": "duplicate id two", "chat_id": OWNER},
            ],
        )

        claimed = rem.claim_due(now_epoch=due - 1, owner_chat_id=OWNER)
        persisted_ids = [str(row.get("id")) for row in _rows(reminders_file)]
        _check(not claimed, f"future legacy rows were claimed during migration: {claimed!r}")
        _check(len(persisted_ids) == 4, f"legacy rows were lost during migration: {persisted_ids!r}")
        _check(len(set(persisted_ids)) == 4, f"legacy rows did not receive distinct ids: {persisted_ids!r}")
        for reminder_id in persisted_ids:
            uuid.UUID(reminder_id)
        rem.claim_due(now_epoch=due - 1, owner_chat_id=OWNER)
        _check(
            [str(row.get("id")) for row in _rows(reminders_file)] == persisted_ids,
            "migrated reminder ids were not durably persisted",
        )


def test_concurrent_claim_due_has_one_network_eligible_claim() -> None:
    with _isolated_store():
        _check(rem.add_reminder(time.time() - 1, "claim once", OWNER), "could not seed reminder")
        barrier = threading.Barrier(2)

        def claim() -> list[dict]:
            barrier.wait()
            return rem.claim_due(owner_chat_id=OWNER)

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _index: claim(), range(2)))

        claims = [item for result in results for item in result]
        _check([len(result) for result in results].count(1) == 1, f"concurrent claims were not exclusive: {results!r}")
        _check(len(claims) == 1, f"expected one network-eligible claim, got {claims!r}")
        item = claims[0]
        _check(rem.begin_delivery(item["id"], item["token"]), "the sole claim was not network eligible")
        _check(not rem.begin_delivery(item["id"], item["token"]), "one claim became network eligible twice")


def test_claim_due_batch_limit_is_strict_and_ordered() -> None:
    with _isolated_store() as reminders_file:
        now = time.time()
        seeded: list[tuple[float, str]] = []
        due_offsets = (4, 1, 3, 1, 2, 1, 5)
        for index, offset in enumerate(due_offsets):
            due = now - offset
            result = rem.add_reminder_result(due, f"ordered-{index}", OWNER)
            _check(result.status == "stored", f"could not seed ordered reminder {index}: {result!r}")
            seeded.append((due, result.reminder_id))

        before_invalid = reminders_file.read_bytes()
        invalid_limits = (
            None,
            True,
            False,
            0,
            -1,
            1.0,
            "1",
            rem.REMINDER_DELIVERY_BATCH_LIMIT + 1,
        )
        for invalid_limit in invalid_limits:
            claims = rem.claim_due(  # type: ignore[arg-type]
                now_epoch=now,
                owner_chat_id=OWNER,
                limit=invalid_limit,
            )
            _check(not claims, f"invalid claim limit was accepted: {invalid_limit!r} -> {claims!r}")
            _check(
                reminders_file.read_bytes() == before_invalid,
                f"invalid claim limit changed reminder state: {invalid_limit!r}",
            )

        expected_ids = [
            reminder_id
            for _due, reminder_id in sorted(seeded, key=lambda item: (item[0], item[1]))
        ]
        claimed_ids: list[str] = []
        for _batch in range((len(expected_ids) + 2) // 3):
            claims = rem.claim_due(now_epoch=now, owner_chat_id=OWNER, limit=3)
            _check(1 <= len(claims) <= 3, f"claim batch escaped its explicit bound: {claims!r}")
            claimed_ids.extend(str(item["id"]) for item in claims)
        _check(claimed_ids == expected_ids, f"due reminders were not claimed by due/id order: {claimed_ids!r}")


def test_bridge_drains_large_backlog_in_bounded_batches() -> None:
    with _isolated_store() as reminders_file:
        now = time.time()
        total = rem.REMINDER_DELIVERY_BATCH_LIMIT * 2 + 3
        for index in reversed(range(total)):
            result = rem.add_reminder_result(
                now - total + index,
                f"batch-{index:02d}",
                OWNER,
            )
            _check(result.status == "stored", f"could not seed backlog reminder {index}: {result!r}")

        calls: list[str] = []

        def send(_chat_id: str, text: str, _markup=None) -> dict:
            calls.append(text)
            return _accepted(50_000 + len(calls))

        bridge = _bridge(send)
        expected_batch_sizes = [
            rem.REMINDER_DELIVERY_BATCH_LIMIT,
            rem.REMINDER_DELIVERY_BATCH_LIMIT,
            3,
        ]
        delivered = 0
        for expected_size in expected_batch_sizes:
            bridge._deliver_due_reminders()
            delivered += expected_size
            _check(
                len(calls) == delivered,
                f"bridge did not honor bounded reminder batch {expected_size}: {len(calls)}",
            )

        expected_calls = [f"⏰ batch-{index:02d}" for index in range(total)]
        _check(calls == expected_calls, f"large backlog drained out of due order: {calls!r}")
        rows = _rows(reminders_file)
        _check(len(rows) == total, f"bounded drain lost reminder receipts: {len(rows)}")
        _check(
            all(
                row.get("state") == "accepted"
                and row.get("attempt_count") == 1
                and "message" not in row
                and "token" not in row
                for row in rows
            ),
            f"bounded drain changed accepted receipt semantics: {rows!r}",
        )
        health = rem.delivery_health_status()
        _check(
            health.get("source_available") is True
            and health.get("status") == "healthy_batch"
            and health.get("claimed_count") == 3
            and health.get("started_count") == 3
            and health.get("api_accepted_count") == 3
            and health.get("durably_finalized_count") == 3
            and health.get("saturated") is False,
            f"latest bounded batch health receipt was not truthful: {health!r}",
        )
        health_payload = rem._delivery_health_file().read_bytes()
        _check(
            len(health_payload) <= rem.REMINDER_DELIVERY_HEALTH_MAX_BYTES,
            f"health snapshot escaped its byte cap: {len(health_payload)}",
        )
        _check(
            b"batch-" not in health_payload and OWNER.encode() not in health_payload,
            f"health snapshot retained reminder or owner content: {health_payload!r}",
        )


def test_delivery_health_snapshot_is_strict_content_free_and_stale_safe() -> None:
    with _isolated_store():
        missing = rem.delivery_health_status()
        _check(
            missing.get("status") == "unknown"
            and missing.get("reason") == "missing"
            and missing.get("source_available") is False,
            f"missing health snapshot claimed health: {missing!r}",
        )
        secret = "sk_" + "live_SUPERSECRET123"
        _check(
            rem.record_delivery_health(
                status="degraded",
                reason=secret,
                claimed_count=10_000,
                started_count=10_000,
                durably_finalized_count=10_000,
                uncertain_count=10_000,
                saturated=True,
            ),
            "content-free health snapshot could not be recorded",
        )
        payload = rem._delivery_health_file().read_bytes()
        _check(
            len(payload) <= rem.REMINDER_DELIVERY_HEALTH_MAX_BYTES
            and secret.encode() not in payload
            and b'"reason":"detail_suppressed"' in payload,
            f"health snapshot leaked or escaped its cap: {payload!r}",
        )
        health = rem.delivery_health_status()
        _check(
            health.get("status") == "degraded"
            and health.get("reason") == "detail_suppressed"
            and health.get("claimed_count") == rem.REMINDER_DELIVERY_BATCH_LIMIT
            and health.get("started_count") == rem.REMINDER_DELIVERY_BATCH_LIMIT
            and health.get("uncertain_count") == rem.REMINDER_DELIVERY_BATCH_LIMIT
            and health.get("content_in_receipt") is False,
            f"health snapshot caps or content boundary drifted: {health!r}",
        )

        injected = json.loads(payload.decode("utf-8"))
        injected["reason"] = secret
        rem._delivery_health_file().write_text(json.dumps(injected), encoding="utf-8")
        rejected = rem.delivery_health_status()
        _check(
            rejected.get("status") == "unknown"
            and rejected.get("reason") == "invalid_schema"
            and rejected.get("source_available") is False,
            f"unrecognized reason code was trusted: {rejected!r}",
        )

        injected["status"] = "healthy_idle"
        injected["reason"] = ""
        rem._delivery_health_file().write_text(json.dumps(injected), encoding="utf-8")
        inconsistent = rem.delivery_health_status()
        _check(
            inconsistent.get("status") == "unknown"
            and inconsistent.get("reason") == "invalid_schema"
            and inconsistent.get("source_available") is False,
            f"healthy status with failure counters was trusted: {inconsistent!r}",
        )

        _check(
            rem.record_delivery_health(
                status="healthy_idle",
                checked_at=time.time() - rem.REMINDER_DELIVERY_HEALTH_STALE_SECONDS - 1,
            ),
            "stale health fixture could not be recorded",
        )
        stale = rem.delivery_health_status()
        _check(
            stale.get("status") == "unknown"
            and stale.get("reason") == "stale"
            and stale.get("stale") is True
            and stale.get("last_status") == "healthy_idle",
            f"stale health snapshot was trusted: {stale!r}",
        )

        rem._delivery_health_file().write_bytes(b"{malformed-health")
        malformed = rem.delivery_health_status()
        _check(
            malformed.get("status") == "unknown"
            and malformed.get("reason") == "malformed"
            and malformed.get("source_available") is False,
            f"malformed health snapshot was trusted or repaired: {malformed!r}",
        )


def test_health_write_failure_does_not_change_delivery_semantics() -> None:
    with _isolated_store() as reminders_file:
        _check(rem.add_reminder(time.time() - 1, "health write failure", OWNER), "could not seed reminder")
        original_record = rem.record_delivery_health
        calls: list[str] = []
        try:
            rem.record_delivery_health = lambda **_kwargs: False  # type: ignore[assignment]
            bridge = _bridge(
                lambda _chat_id, text, _markup=None: calls.append(text) or _accepted(50_501)
            )
            bridge._deliver_due_reminders()
        finally:
            rem.record_delivery_health = original_record  # type: ignore[assignment]
        row = _rows(reminders_file)[0]
        _check(len(calls) == 1, f"health writer failure changed send count: {calls!r}")
        _check(
            row.get("state") == "accepted"
            and row.get("telegram_message_id") == 50_501
            and row.get("attempt_count") == 1,
            f"health writer failure changed durable send truth: {row!r}",
        )


def test_finalization_failure_is_health_outcome_unknown() -> None:
    with _isolated_store() as reminders_file:
        _check(rem.add_reminder(time.time() - 1, "finalization failure", OWNER), "could not seed reminder")
        original_finish = rem.finish_delivery
        calls: list[str] = []
        try:
            rem.finish_delivery = lambda *_args, **_kwargs: False  # type: ignore[assignment]
            bridge = _bridge(
                lambda _chat_id, text, _markup=None: calls.append(text) or _accepted(50_502)
            )
            bridge._deliver_due_reminders()
        finally:
            rem.finish_delivery = original_finish  # type: ignore[assignment]
        row = _rows(reminders_file)[0]
        health = rem.delivery_health_status()
        _check(len(calls) == 1 and row.get("state") == "sending", f"finalization fixture drifted: {row!r}")
        _check(
            health.get("status") == "outcome_unknown"
            and health.get("reason") == "finalization_failed"
            and health.get("api_accepted_count") == 1
            and health.get("durably_finalized_count") == 0
            and health.get("finalization_failed_count") == 1,
            f"finalization loss overclaimed delivery health: {health!r}",
        )


def test_begin_and_finish_are_token_fenced() -> None:
    with _isolated_store() as reminders_file:
        _check(rem.add_reminder(time.time() - 1, "fenced", OWNER), "could not seed reminder")
        item = rem.claim_due(owner_chat_id=OWNER)[0]
        wrong_token = str(uuid.uuid4())
        _check(not rem.begin_delivery(item["id"], wrong_token), "wrong token began delivery")
        _check(
            _rows(reminders_file)[0].get("attempt_count") == 0,
            "wrong begin token consumed an attempt",
        )
        _check(rem.begin_delivery(item["id"], item["token"]), "claim token could not begin delivery")
        _check(
            not rem.finish_delivery(item["id"], wrong_token, "accepted", telegram_message_id=201),
            "wrong token finished delivery",
        )
        _check(
            rem.finish_delivery(item["id"], item["token"], "accepted", telegram_message_id=202),
            "claim token could not finish delivery",
        )
        _check(
            not rem.finish_delivery(item["id"], item["token"], "accepted", telegram_message_id=203),
            "finished token was reusable",
        )
        row = _rows(reminders_file)[0]
        _check(row.get("state") == "accepted" and row.get("telegram_message_id") == 202, f"bad fenced receipt: {row!r}")


def test_concurrent_begin_increments_once() -> None:
    with _isolated_store() as reminders_file:
        _check(rem.add_reminder(time.time() - 1, "begin once", OWNER), "could not seed reminder")
        item = rem.claim_due(owner_chat_id=OWNER)[0]
        barrier = threading.Barrier(2)

        def begin() -> bool:
            barrier.wait()
            return rem.begin_delivery(item["id"], item["token"])

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _index: begin(), range(2)))

        row = _rows(reminders_file)[0]
        _check(results.count(True) == 1, f"concurrent begin was not exclusive: {results!r}")
        _check(row.get("state") == "sending", f"winning begin did not enter sending: {row!r}")
        _check(row.get("attempt_count") == 1, f"concurrent begin counted more than once: {row!r}")


def test_begin_recovers_one_time_directory_sync_failure_before_send() -> None:
    with _isolated_store() as reminders_file:
        _check(rem.add_reminder(time.time() - 1, "resync once", OWNER), "could not seed reminder")
        item = rem.claim_due(owner_chat_id=OWNER)[0]
        original_fsync = rem.os.fsync
        directory_calls = 0

        def fail_first_directory_sync(fd: int) -> None:
            nonlocal directory_calls
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                directory_calls += 1
                if directory_calls == 1:
                    raise OSError("synthetic directory sync failure")
            original_fsync(fd)

        rem.os.fsync = fail_first_directory_sync  # type: ignore[assignment]
        try:
            result = rem.begin_delivery_result(item["id"], item["token"])
        finally:
            rem.os.fsync = original_fsync  # type: ignore[assignment]

        row = _rows(reminders_file)[0]
        _check(result.network_eligible, f"verified published transition was not recovered: {result!r}")
        _check(result.reason == "directory_resynced", f"recovery reason was not explicit: {result!r}")
        _check(directory_calls == 2, f"directory durability was not retried exactly once: {directory_calls}")
        _check(
            row.get("state") == "sending"
            and row.get("token") == item["token"]
            and row.get("attempt_count") == 1,
            f"recovered begin did not preserve the exact send fence: {row!r}",
        )


def test_begin_compensates_persistent_directory_sync_failure_before_send() -> None:
    with _isolated_store() as reminders_file:
        _check(rem.add_reminder(time.time() - 1, "compensate before send", OWNER), "could not seed reminder")
        item = rem.claim_due(owner_chat_id=OWNER)[0]
        original_fsync = rem.os.fsync
        directory_calls = 0

        def fail_all_directory_syncs(fd: int) -> None:
            nonlocal directory_calls
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                directory_calls += 1
                raise OSError("synthetic persistent directory sync failure")
            original_fsync(fd)

        rem.os.fsync = fail_all_directory_syncs  # type: ignore[assignment]
        try:
            result = rem.begin_delivery_result(item["id"], item["token"])
        finally:
            rem.os.fsync = original_fsync  # type: ignore[assignment]

        row = _rows(reminders_file)[0]
        _check(not result.network_eligible, f"unproven transition reached the network boundary: {result!r}")
        _check(result.status == "unavailable", f"unproven compensation claimed retryability: {result!r}")
        _check(
            result.reason == "compensation_durability_unproven",
            f"unproven compensation reason was not explicit: {result!r}",
        )
        _check(directory_calls == 3, f"begin did not attempt durable compensation: {directory_calls}")
        _check(
            row.get("state") == "pending"
            and row.get("attempt_count") == 0
            and "token" not in row
            and "sending_at" not in row,
            f"unsent reminder was not compensated to pending: {row!r}",
        )

        calls: list[str] = []
        bridge = _bridge(lambda _chat_id, text, _markup=None: calls.append(text) or _accepted(205))
        bridge._deliver_due_reminders()
        accepted = _rows(reminders_file)[0]
        _check(len(calls) == 1, f"compensated reminder was not delivered exactly once: {calls!r}")
        _check(
            accepted.get("state") == "accepted"
            and accepted.get("telegram_message_id") == 205
            and accepted.get("attempt_count") == 1,
            f"compensated reminder did not finish with one real attempt: {accepted!r}",
        )


def test_begin_retries_transient_readback_failure_before_send() -> None:
    with _isolated_store() as reminders_file:
        _check(rem.add_reminder(time.time() - 1, "retry readback", OWNER), "could not seed reminder")
        item = rem.claim_due(owner_chat_id=OWNER)[0]
        original_fsync = rem.os.fsync
        original_read_locked = rem._read_locked
        directory_calls = 0
        read_calls = 0

        def fail_first_directory_sync(fd: int) -> None:
            nonlocal directory_calls
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                directory_calls += 1
                if directory_calls == 1:
                    raise OSError("synthetic directory sync failure")
            original_fsync(fd)

        def fail_first_reconciliation_read(path: Path):
            nonlocal read_calls
            read_calls += 1
            if read_calls == 2:
                raise rem.ReminderStoreUnavailable("synthetic_read_error")
            return original_read_locked(path)

        rem.os.fsync = fail_first_directory_sync  # type: ignore[assignment]
        rem._read_locked = fail_first_reconciliation_read  # type: ignore[assignment]
        try:
            result = rem.begin_delivery_result(item["id"], item["token"])
        finally:
            rem._read_locked = original_read_locked  # type: ignore[assignment]
            rem.os.fsync = original_fsync  # type: ignore[assignment]

        row = _rows(reminders_file)[0]
        _check(result.network_eligible, f"transient readback failure stranded the send: {result!r}")
        _check(read_calls == 3, f"readback was not retried exactly once: {read_calls}")
        _check(directory_calls == 2, f"directory sync was not recovered: {directory_calls}")
        _check(
            row.get("state") == "sending" and row.get("attempt_count") == 1,
            f"transient readback recovery changed send truth: {row!r}",
        )


def test_begin_compensates_exhausted_normalized_readback_failures() -> None:
    with _isolated_store() as reminders_file:
        _check(rem.add_reminder(time.time() - 1, "compensate readback", OWNER), "could not seed reminder")
        item = rem.claim_due(owner_chat_id=OWNER)[0]
        original_fsync = rem.os.fsync
        original_read_locked = rem._read_locked
        directory_calls = 0
        read_calls = 0

        def fail_first_directory_sync(fd: int) -> None:
            nonlocal directory_calls
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                directory_calls += 1
                if directory_calls == 1:
                    raise OSError("synthetic directory sync failure")
            original_fsync(fd)

        def fail_reconciliation_reads(path: Path):
            nonlocal read_calls
            read_calls += 1
            if read_calls >= 2:
                raise rem.ReminderStoreUnavailable("synthetic_read_error")
            return original_read_locked(path)

        rem.os.fsync = fail_first_directory_sync  # type: ignore[assignment]
        rem._read_locked = fail_reconciliation_reads  # type: ignore[assignment]
        try:
            result = rem.begin_delivery_result(item["id"], item["token"])
        finally:
            rem._read_locked = original_read_locked  # type: ignore[assignment]
            rem.os.fsync = original_fsync  # type: ignore[assignment]

        row = _rows(reminders_file)[0]
        _check(result.status == "retryable", f"durable compensation did not recover readback loss: {result!r}")
        _check(read_calls == 3, f"normalized readback retries were not bounded: {read_calls}")
        _check(directory_calls == 2, f"compensation was not directory-durable: {directory_calls}")
        _check(
            row.get("state") == "pending"
            and row.get("attempt_count") == 0
            and "token" not in row
            and "sending_at" not in row,
            f"exhausted readback did not durably compensate: {row!r}",
        )


def test_begin_file_sync_failure_preserves_claim_without_attempt() -> None:
    with _isolated_store() as reminders_file:
        _check(rem.add_reminder(time.time() - 1, "no publish", OWNER), "could not seed reminder")
        item = rem.claim_due(owner_chat_id=OWNER)[0]
        original_fsync = rem.os.fsync
        failed = False

        def fail_first_file_sync(fd: int) -> None:
            nonlocal failed
            if not failed and stat.S_ISREG(os.fstat(fd).st_mode):
                failed = True
                raise OSError("synthetic file sync failure")
            original_fsync(fd)

        rem.os.fsync = fail_first_file_sync  # type: ignore[assignment]
        try:
            result = rem.begin_delivery_result(item["id"], item["token"])
        finally:
            rem.os.fsync = original_fsync  # type: ignore[assignment]

        row = _rows(reminders_file)[0]
        _check(not result.network_eligible, f"pre-publication failure became network eligible: {result!r}")
        _check(
            row.get("state") == "claimed"
            and row.get("token") == item["token"]
            and row.get("attempt_count") == 0,
            f"pre-publication failure changed durable attempt truth: {row!r}",
        )


def test_exhausted_or_malformed_attempts_never_reach_sender() -> None:
    with _isolated_store() as reminders_file:
        now = time.time()
        _write_rows(
            reminders_file,
            [
                {
                    "id": str(uuid.uuid4()),
                    "due": now - 1,
                    "message": "exhausted",
                    "chat_id": OWNER,
                    "transport": "telegram",
                    "state": "rejected",
                    "attempt_count": rem.MAX_DELIVERY_ATTEMPTS,
                    "next_attempt_at": now - 1,
                },
                {
                    "id": str(uuid.uuid4()),
                    "due": now - 1,
                    "message": "malformed string",
                    "chat_id": OWNER,
                    "transport": "telegram",
                    "state": "pending",
                    "attempt_count": "not-a-count",
                },
                {
                    "id": str(uuid.uuid4()),
                    "due": now - 1,
                    "message": "malformed falsey",
                    "chat_id": OWNER,
                    "transport": "telegram",
                    "state": "pending",
                    "attempt_count": [],
                },
                {
                    "id": str(uuid.uuid4()),
                    "due": now - 1,
                    "message": "malformed fractional",
                    "chat_id": OWNER,
                    "transport": "telegram",
                    "state": "pending",
                    "attempt_count": 4.5,
                },
                {
                    "id": str(uuid.uuid4()),
                    "due": now - 1,
                    "message": "malformed empty",
                    "chat_id": OWNER,
                    "transport": "telegram",
                    "state": "pending",
                    "attempt_count": "",
                },
                {
                    "id": str(uuid.uuid4()),
                    "due": now - 1,
                    "message": "malformed unicode digit",
                    "chat_id": OWNER,
                    "transport": "telegram",
                    "state": "pending",
                    "attempt_count": "²",
                },
                {
                    "id": str(uuid.uuid4()),
                    "due": now - 1,
                    "message": "malformed huge integer",
                    "chat_id": OWNER,
                    "transport": "telegram",
                    "state": "pending",
                    "attempt_count": "9" * 5000,
                },
            ],
        )
        calls: list[str] = []
        bridge = _bridge(lambda _chat_id, text, _markup=None: calls.append(text) or _accepted(206))
        bridge._deliver_due_reminders()
        rows = _rows(reminders_file)
        _check(not calls, f"invalid attempt accounting reached Telegram: {calls!r}")
        _check(all(row.get("state") == "failed" for row in rows), f"invalid rows stayed active: {rows!r}")
        _check(
            {row.get("error_code") for row in rows} == {"retry_limit_reached", "invalid_attempt_count"},
            f"invalid attempt accounting was not explicit: {rows!r}",
        )
        _check(all("message" not in row and "token" not in row for row in rows), f"failed rows kept payloads: {rows!r}")


def test_retry_limit_counts_actual_delivery_starts() -> None:
    with _isolated_store() as reminders_file:
        _check(rem.add_reminder(time.time() - 1, "bounded retry", OWNER), "could not seed reminder")

        for expected_attempt in range(1, rem.MAX_DELIVERY_ATTEMPTS + 1):
            item = rem.claim_due(owner_chat_id=OWNER)[0]
            _check(
                item.get("attempt_count") == expected_attempt - 1,
                f"claim consumed attempt {expected_attempt}: {item!r}",
            )
            _check(rem.begin_delivery(item["id"], item["token"]), f"attempt {expected_attempt} did not begin")
            _check(
                _rows(reminders_file)[0].get("attempt_count") == expected_attempt,
                f"attempt {expected_attempt} was not counted at begin",
            )
            _check(
                rem.finish_delivery(item["id"], item["token"], "rejected", error_code="http_429"),
                f"attempt {expected_attempt} could not finish",
            )

            row = _rows(reminders_file)[0]
            expected_state = "failed" if expected_attempt == rem.MAX_DELIVERY_ATTEMPTS else "rejected"
            _check(row.get("state") == expected_state, f"wrong retry-limit state: {row!r}")

        _check(not rem.claim_due(owner_chat_id=OWNER), "retry limit allowed another claim")


def test_bridge_retries_explicit_429_then_accepts_receipt() -> None:
    with _isolated_store() as reminders_file:
        _check(rem.add_reminder(time.time() - 1, "retry me", OWNER), "could not seed reminder")
        calls: list[tuple[str, str]] = []
        responses = iter(
            [
                {"ok": False, "error": "http_429", "http_status": 429, "retry_after": 0},
                _accepted(42901),
            ]
        )

        def send(chat_id: str, text: str, _markup=None) -> dict:
            calls.append((chat_id, text))
            return next(responses)

        bridge = _bridge(send)
        bridge._deliver_due_reminders()
        rejected = _rows(reminders_file)[0]
        _check(rejected.get("state") == "rejected", f"429 was not retained for retry: {rejected!r}")
        _check(rejected.get("attempt_count") == 1 and "message" in rejected, f"retry payload was lost: {rejected!r}")
        bridge._deliver_due_reminders()
        accepted = _rows(reminders_file)[0]
        _check(len(calls) == 2, f"429 retry did not make exactly two send attempts: {calls!r}")
        _check(
            accepted.get("state") == "accepted" and accepted.get("telegram_message_id") == 42901,
            f"retry acceptance receipt was not stored: {accepted!r}",
        )
        _check(accepted.get("attempt_count") == 2, f"retry attempt count was wrong: {accepted!r}")


def test_bridge_401_is_permanent_without_retry() -> None:
    with _isolated_store() as reminders_file:
        _check(rem.add_reminder(time.time() - 1, "do not retry", OWNER), "could not seed reminder")
        calls: list[str] = []

        def send(_chat_id: str, text: str, _markup=None) -> dict:
            calls.append(text)
            return {"ok": False, "error": "http_401", "http_status": 401}

        bridge = _bridge(send)
        bridge._deliver_due_reminders()
        bridge._deliver_due_reminders()
        row = _rows(reminders_file)[0]
        _check(len(calls) == 1, f"permanent 401 was retried: {calls!r}")
        _check(row.get("state") == "failed" and row.get("error_code") == "http_401", f"bad 401 state: {row!r}")


def test_bridge_missing_message_id_is_uncertain_without_replay() -> None:
    with _isolated_store() as reminders_file:
        _check(rem.add_reminder(time.time() - 1, "unknown receipt", OWNER), "could not seed reminder")
        calls: list[str] = []

        def send(_chat_id: str, text: str, _markup=None) -> dict:
            calls.append(text)
            return {"ok": True, "result": {}}

        bridge = _bridge(send)
        bridge._deliver_due_reminders()
        bridge._deliver_due_reminders()
        row = _rows(reminders_file)[0]
        _check(len(calls) == 1, f"missing message id was replayed: {calls!r}")
        _check(
            row.get("state") == "uncertain" and row.get("error_code") == "missing_message_id",
            f"malformed ok:true response was not quarantined: {row!r}",
        )


def test_bridge_send_exception_is_uncertain_without_replay() -> None:
    with _isolated_store() as reminders_file:
        _check(rem.add_reminder(time.time() - 1, "exception receipt", OWNER), "could not seed reminder")
        calls: list[str] = []

        def send(_chat_id: str, text: str, _markup=None) -> dict:
            calls.append(text)
            raise RuntimeError("synthetic local send failure")

        bridge = _bridge(send)
        bridge._deliver_due_reminders()
        bridge._deliver_due_reminders()
        row = _rows(reminders_file)[0]
        _check(len(calls) == 1, f"send exception was replayed: {calls!r}")
        _check(
            row.get("state") == "uncertain" and row.get("error_code") == "send_exception",
            f"send exception was not quarantined: {row!r}",
        )


def test_stale_pre_io_claim_does_not_consume_attempt() -> None:
    with _isolated_store() as reminders_file:
        now = time.time()
        _write_rows(
            reminders_file,
            [
                {
                    "id": str(uuid.uuid4()),
                    "due": now - 60,
                    "message": "recover claim",
                    "chat_id": OWNER,
                    "transport": "telegram",
                    "state": "claimed",
                    "attempt_count": 0,
                    "claimed_at": now - rem.DELIVERY_CLAIM_STALE_SECONDS - 10,
                    "token": str(uuid.uuid4()),
                }
            ],
        )
        claims = rem.claim_due(owner_chat_id=OWNER)
        row = _rows(reminders_file)[0]
        _check(len(claims) == 1, f"stale pre-I/O claim was not reclaimed once: {claims!r}")
        _check(
            row.get("state") == "claimed" and row.get("attempt_count") == 0,
            f"stale pre-I/O claim changed the attempt budget: {row!r}",
        )


def test_stale_sending_becomes_uncertain_without_replay() -> None:
    with _isolated_store() as reminders_file:
        now = time.time()
        _write_rows(
            reminders_file,
            [
                {
                    "id": str(uuid.uuid4()),
                    "due": now - 60,
                    "message": "do not replay sending",
                    "chat_id": OWNER,
                    "transport": "telegram",
                    "state": "sending",
                    "attempt_count": 3,
                    "sending_at": now - rem.DELIVERY_CLAIM_STALE_SECONDS - 10,
                    "token": str(uuid.uuid4()),
                }
            ],
        )
        calls: list[str] = []
        bridge = _bridge(lambda _chat_id, text, _markup=None: calls.append(text) or _accepted(302))
        bridge._deliver_due_reminders()
        bridge._deliver_due_reminders()
        row = _rows(reminders_file)[0]
        _check(not calls, f"stale sending state reached sender: {calls!r}")
        _check(
            row.get("state") == "uncertain"
            and row.get("error_code") == "stale_sending_recovered"
            and row.get("attempt_count") == 3,
            f"stale sending state was not quarantined: {row!r}",
        )


def test_non_owner_and_unknown_transport_never_call_sender() -> None:
    with _isolated_store() as reminders_file:
        now = time.time()
        _write_rows(
            reminders_file,
            [
                {
                    "id": str(uuid.uuid4()),
                    "due": now - 1,
                    "message": "wrong owner",
                    "chat_id": "20002",
                    "transport": "telegram",
                    "state": "pending",
                    "attempt_count": 0,
                },
                {
                    "id": str(uuid.uuid4()),
                    "due": now - 1,
                    "message": "wrong transport",
                    "chat_id": OWNER,
                    "transport": "unknown-local-transport",
                    "state": "pending",
                    "attempt_count": 0,
                },
            ],
        )
        calls: list[str] = []
        bridge = _bridge(lambda _chat_id, text, _markup=None: calls.append(text) or _accepted(303))
        bridge._deliver_due_reminders()
        rows = _rows(reminders_file)
        _check(not calls, f"unsafe stored row reached sender: {calls!r}")
        _check(
            {row.get("error_code") for row in rows} == {"owner_mismatch", "unknown_transport"},
            f"unsafe rows were not quarantined with explicit reasons: {rows!r}",
        )
        _check(all(row.get("state") == "failed" for row in rows), f"unsafe rows remained active: {rows!r}")


def test_local_path_payload_is_redacted_and_terminal_message_scrubbed() -> None:
    with _isolated_store() as reminders_file:
        raw_path = "/tmp/jarvis-reminder-smoke-secret.txt"
        _check(rem.add_reminder(time.time() - 1, raw_path, OWNER), "could not seed path reminder")
        calls: list[tuple[str, str]] = []
        bridge = _bridge(
            lambda chat_id, text, _markup=None: calls.append((chat_id, text)) or _accepted(304)
        )
        bridge._deliver_due_reminders()
        row = _rows(reminders_file)[0]
        serialized = json.dumps({"calls": calls, "row": row})
        _check(len(calls) == 1 and calls[0][0] == OWNER, f"redacted payload was not sent once: {calls!r}")
        _check("<local-path>" in calls[0][1] and raw_path not in serialized, f"local path leaked: {serialized}")
        _check(row.get("state") == "accepted" and "message" not in row, f"terminal payload was not scrubbed: {row!r}")


def test_unreadable_store_never_reaches_sender_or_writer() -> None:
    with _isolated_store() as reminders_file:
        payload = b"{malformed-reminder-state"
        reminders_file.write_bytes(payload)
        calls: list[str] = []
        original_write = rem._write_locked
        write_calls: list[str] = []

        def forbidden_write(path, items):
            write_calls.append(str(path))
            raise AssertionError("unavailable reminder state reached the writer")

        rem._write_locked = forbidden_write  # type: ignore[assignment]
        try:
            bridge = _bridge(lambda _chat_id, text, _markup=None: calls.append(text) or _accepted(999))
            bridge._deliver_due_reminders()
        finally:
            rem._write_locked = original_write  # type: ignore[assignment]

        _check(not calls, f"unreadable reminder state reached Telegram sender: {calls!r}")
        _check(not write_calls, f"unreadable reminder state reached writer: {write_calls!r}")
        _check(reminders_file.read_bytes() == payload, "unreadable reminder bytes were not preserved")
        health = rem.delivery_health_status()
        _check(
            health.get("status") == "store_unavailable"
            and health.get("reason") == "malformed_json"
            and health.get("claimed_count") == 0
            and health.get("started_count") == 0,
            f"unreadable store did not leave a content-free outage receipt: {health!r}",
        )
        health_payload = rem._delivery_health_file().read_bytes()
        _check(
            payload not in health_payload
            and str(reminders_file).encode() not in health_payload,
            f"store outage health leaked raw reminder state or path: {health_payload!r}",
        )


def main() -> None:
    test_legacy_id_migration_persists_unique_ids()
    test_concurrent_claim_due_has_one_network_eligible_claim()
    test_claim_due_batch_limit_is_strict_and_ordered()
    test_bridge_drains_large_backlog_in_bounded_batches()
    test_delivery_health_snapshot_is_strict_content_free_and_stale_safe()
    test_health_write_failure_does_not_change_delivery_semantics()
    test_finalization_failure_is_health_outcome_unknown()
    test_begin_and_finish_are_token_fenced()
    test_concurrent_begin_increments_once()
    test_begin_recovers_one_time_directory_sync_failure_before_send()
    test_begin_compensates_persistent_directory_sync_failure_before_send()
    test_begin_retries_transient_readback_failure_before_send()
    test_begin_compensates_exhausted_normalized_readback_failures()
    test_begin_file_sync_failure_preserves_claim_without_attempt()
    test_exhausted_or_malformed_attempts_never_reach_sender()
    test_retry_limit_counts_actual_delivery_starts()
    test_bridge_retries_explicit_429_then_accepts_receipt()
    test_bridge_401_is_permanent_without_retry()
    test_bridge_missing_message_id_is_uncertain_without_replay()
    test_bridge_send_exception_is_uncertain_without_replay()
    test_stale_pre_io_claim_does_not_consume_attempt()
    test_stale_sending_becomes_uncertain_without_replay()
    test_non_owner_and_unknown_transport_never_call_sender()
    test_local_path_payload_is_redacted_and_terminal_message_scrubbed()
    test_unreadable_store_never_reaches_sender_or_writer()
    print("Reminder delivery recovery smoke passed")


if __name__ == "__main__":
    main()
