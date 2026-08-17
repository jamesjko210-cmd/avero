"""Owner-Telegram reminders & timers for Jarvis V2.

A reminder is stored locally with a due time and a message. The Telegram bridge
checks for due reminders on each poll and sends them through the owner Telegram
channel. Telegram API acceptance is the automatable proof; phone display remains
a live-user confirmation, unlike a macOS reminder that only fires on the Mac.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import fcntl

from jarvis_v2.tools.nl_datetime import _parse_time


REMINDERS_FILE: Path | None = None
MAX_MESSAGE_CHARS = 300
MAX_DELIVERY_ATTEMPTS = 5
MAX_RETRY_AFTER_SECONDS = 86_400
DELIVERY_CLAIM_STALE_SECONDS = 300
REMINDER_DELIVERY_BATCH_LIMIT = 10
REMINDER_DELIVERY_HEALTH_VERSION = 1
REMINDER_DELIVERY_HEALTH_MAX_BYTES = 4_096
REMINDER_DELIVERY_HEALTH_STALE_SECONDS = 300
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
_ACTIVE_STATES = frozenset({"pending", "claimed", "sending", "rejected"})
_TERMINAL_STATES = frozenset({"accepted", "failed", "uncertain", "cancelled"})
_DELIVERY_HEALTH_STATUSES = frozenset(
    {"healthy_idle", "healthy_batch", "degraded", "store_unavailable", "outcome_unknown"}
)
_DELIVERY_HEALTH_REASONS = frozenset(
    {
        "",
        "detail_suppressed",
        "finalization_failed",
        "invalid_encoding",
        "invalid_health_status",
        "invalid_limit",
        "invalid_now",
        "invalid_row",
        "invalid_schema",
        "invalid_size",
        "malformed",
        "malformed_json",
        "missing",
        "permanent_failure",
        "pre_send_blocked",
        "read_error",
        "retry_scheduled",
        "stale",
        "store_unavailable",
        "uncertain_delivery",
        "unknown",
    }
)
_DELIVERY_HEALTH_COUNT_KEYS = (
    "claimed_count",
    "started_count",
    "durably_finalized_count",
    "api_accepted_count",
    "rejected_count",
    "failed_count",
    "uncertain_count",
    "pre_send_blocked_count",
    "finalization_failed_count",
)
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_DURATION_AMOUNT_RE = r"\d+|an?|half(?:\s+an?)?"
_DURATION_NATURAL_AMOUNT_RE = r"an?|half(?:\s+an?)?"
_DURATION_WORD_UNIT_RE = r"seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?"
_DURATION_UNIT_RE = rf"{_DURATION_WORD_UNIT_RE}|s|m|h|d|w"
_DURATION_PART_RE = rf"(?:\d+\s*(?:{_DURATION_UNIT_RE})|(?:{_DURATION_NATURAL_AMOUNT_RE})\s+(?:{_DURATION_WORD_UNIT_RE}))"
_DURATION_RE = rf"{_DURATION_PART_RE}(?:\s*(?:and|,)?\s*{_DURATION_PART_RE})*"
_RECURRING_RE = re.compile(
    r"\b(?:recurring|repeat(?:ing)?|daily|weekly|monthly|yearly|annually)\b"
    r"|\bevery\s+(?:\d+\s*)?(?:second|sec|minute|min|hour|hr|day|weekday|week|month|year)s?\b"
    r"|\beach\s+(?:day|weekday|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)


class ReminderStoreUnavailable(RuntimeError):
    """Bounded signal that existing reminder state could not be read safely."""

    def __init__(self, reason: str) -> None:
        self.reason = _safe_error_code(reason, "read_error")
        super().__init__(self.reason)


class ReminderWriteUncertain(RuntimeError):
    """The new reminder file was published but directory durability is unproven."""


@dataclass(frozen=True)
class ReminderWriteResult:
    status: str
    reminder_id: str = ""
    reason: str = ""

    @property
    def published(self) -> bool:
        return self.status in {"stored", "published_uncertain"}


@dataclass(frozen=True)
class ReminderBeginResult:
    status: str
    reason: str = ""

    @property
    def network_eligible(self) -> bool:
        return self.status == "started"


@dataclass(frozen=True)
class ReminderClaimResult:
    status: str
    items: tuple[dict, ...] = ()
    reason: str = ""

    @property
    def store_available(self) -> bool:
        return self.status == "ok"


def _reminders_file() -> Path:
    if REMINDERS_FILE is not None:
        return REMINDERS_FILE
    raw = os.getenv("JARVIS_REMINDERS_FILE", "").strip()
    return Path(raw or Path.home() / ".jarvis_v3" / "telegram_reminders.json").expanduser()


def _delivery_health_file() -> Path:
    reminders_file = _reminders_file()
    return reminders_file.with_name(reminders_file.name + ".delivery-health.json")


def _thread_lock_for(path: Path) -> threading.RLock:
    key = os.path.abspath(os.fspath(path))
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _THREAD_LOCKS[key] = lock
        return lock


@contextmanager
def _locked_store() -> Iterator[Path]:
    reminders_file = _reminders_file()
    thread_lock = _thread_lock_for(reminders_file)
    with thread_lock:
        reminders_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file = reminders_file.with_name(reminders_file.name + ".lock")
        with lock_file.open("a+b") as process_lock:
            fcntl.flock(process_lock.fileno(), fcntl.LOCK_EX)
            try:
                yield reminders_file
            finally:
                fcntl.flock(process_lock.fileno(), fcntl.LOCK_UN)


def _finite_epoch(value: object) -> float | None:
    try:
        epoch = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return epoch if math.isfinite(epoch) else None


def _safe_error_code(value: object, default: str = "") -> str:
    text = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", str(value or "").strip())
    return (text[:128] or default)[:128]


def _delivery_health_reason(value: object, default: str = "") -> str:
    fallback = (
        default.strip()
        if type(default) is str and default.strip() in _DELIVERY_HEALTH_REASONS
        else ""
    )
    if type(value) is not str:
        return fallback or ("detail_suppressed" if value is not None else "")
    text = value.strip()
    if text in _DELIVERY_HEALTH_REASONS:
        return text
    return fallback or ("detail_suppressed" if text else "")


def _delivery_health_count(value: object) -> int:
    if type(value) is not int:
        return 0
    return max(0, min(value, REMINDER_DELIVERY_BATCH_LIMIT))


def _unknown_delivery_health(reason: str, *, stale: bool = False, age_seconds: int = 0) -> dict:
    return {
        "version": REMINDER_DELIVERY_HEALTH_VERSION,
        "status": "unknown",
        "reason": _delivery_health_reason(reason, "unknown"),
        "source_available": False,
        "stale": stale,
        "age_seconds": max(0, min(int(age_seconds), 86_400)),
        "content_in_receipt": False,
    }


def record_delivery_health(
    *,
    status: str,
    reason: str = "",
    checked_at: float | None = None,
    saturated: bool = False,
    **counts: int,
) -> bool:
    """Best-effort content-free latest-sweep receipt; never affects delivery truth."""
    normalized_status = str(status or "").strip().lower()
    if normalized_status not in _DELIVERY_HEALTH_STATUSES:
        normalized_status = "outcome_unknown"
        reason = "invalid_health_status"
    checked = time.time() if checked_at is None else _finite_epoch(checked_at)
    if checked is None:
        return False
    receipt = {
        "version": REMINDER_DELIVERY_HEALTH_VERSION,
        "checked_at": checked,
        "status": normalized_status,
        "reason": _delivery_health_reason(reason),
        "batch_limit": REMINDER_DELIVERY_BATCH_LIMIT,
        **{
            key: _delivery_health_count(counts.get(key))
            for key in _DELIVERY_HEALTH_COUNT_KEYS
        },
        "saturated": bool(saturated),
        "content_in_receipt": False,
    }
    payload = json.dumps(
        receipt,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > REMINDER_DELIVERY_HEALTH_MAX_BYTES:
        return False

    health_file = _delivery_health_file()
    thread_lock = _thread_lock_for(health_file)
    try:
        with thread_lock:
            health_file.parent.mkdir(parents=True, exist_ok=True)
            lock_file = health_file.with_name(health_file.name + ".lock")
            with lock_file.open("a+b") as process_lock:
                fcntl.flock(process_lock.fileno(), fcntl.LOCK_EX)
                tmp_fd, tmp_name = tempfile.mkstemp(
                    prefix=f".{health_file.name}.",
                    suffix=".tmp",
                    dir=health_file.parent,
                )
                tmp = Path(tmp_name)
                try:
                    with os.fdopen(tmp_fd, "wb") as handle:
                        tmp_fd = -1
                        handle.write(payload)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(tmp, health_file)
                    _sync_parent_directory(health_file)
                finally:
                    if tmp_fd >= 0:
                        os.close(tmp_fd)
                    try:
                        tmp.unlink()
                    except FileNotFoundError:
                        pass
                fcntl.flock(process_lock.fileno(), fcntl.LOCK_UN)
        return True
    except Exception:
        return False


def delivery_health_status(now_epoch: float | None = None) -> dict:
    """Read one strict content-free latest-sweep receipt without repairing it."""
    now = time.time() if now_epoch is None else _finite_epoch(now_epoch)
    if now is None:
        return _unknown_delivery_health("invalid_now")
    try:
        payload = _delivery_health_file().read_bytes()
    except FileNotFoundError:
        return _unknown_delivery_health("missing")
    except OSError:
        return _unknown_delivery_health("read_error")
    if not payload or len(payload) > REMINDER_DELIVERY_HEALTH_MAX_BYTES:
        return _unknown_delivery_health("invalid_size")
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return _unknown_delivery_health("malformed")
    expected_keys = {
        "version",
        "checked_at",
        "status",
        "reason",
        "batch_limit",
        *_DELIVERY_HEALTH_COUNT_KEYS,
        "saturated",
        "content_in_receipt",
    }
    if not isinstance(raw, dict) or set(raw) != expected_keys:
        return _unknown_delivery_health("invalid_schema")
    checked_at_value = _finite_epoch(raw.get("checked_at"))
    status = raw.get("status")
    reason = raw.get("reason")
    if (
        type(raw.get("version")) is not int
        or raw.get("version") != REMINDER_DELIVERY_HEALTH_VERSION
        or type(raw.get("checked_at")) not in {int, float}
        or checked_at_value is None
        or checked_at_value > now + 60
        or status not in _DELIVERY_HEALTH_STATUSES
        or type(reason) is not str
        or reason != _delivery_health_reason(reason)
        or type(raw.get("batch_limit")) is not int
        or raw.get("batch_limit") != REMINDER_DELIVERY_BATCH_LIMIT
        or type(raw.get("saturated")) is not bool
        or raw.get("content_in_receipt") is not False
        or any(
            type(raw.get(key)) is not int
            or raw[key] < 0
            or raw[key] > REMINDER_DELIVERY_BATCH_LIMIT
            for key in _DELIVERY_HEALTH_COUNT_KEYS
        )
    ):
        return _unknown_delivery_health("invalid_schema")
    claimed_count = raw["claimed_count"]
    started_count = raw["started_count"]
    durably_finalized_count = raw["durably_finalized_count"]
    api_accepted_count = raw["api_accepted_count"]
    disposition_count = (
        raw["rejected_count"]
        + raw["failed_count"]
        + raw["uncertain_count"]
    )
    accepted_finalized_count = durably_finalized_count - disposition_count
    if (
        raw["pre_send_blocked_count"] + started_count != claimed_count
        or raw["finalization_failed_count"] + durably_finalized_count != started_count
        or accepted_finalized_count < 0
        or api_accepted_count < accepted_finalized_count
        or api_accepted_count
        > accepted_finalized_count + raw["finalization_failed_count"]
        or raw["saturated"] is not (
            claimed_count == REMINDER_DELIVERY_BATCH_LIMIT
        )
    ):
        return _unknown_delivery_health("invalid_schema")
    if raw["finalization_failed_count"]:
        expected_status = "outcome_unknown"
        expected_reason = "finalization_failed"
    elif raw["pre_send_blocked_count"]:
        expected_status = "degraded"
        expected_reason = "pre_send_blocked"
    elif raw["uncertain_count"]:
        expected_status = "degraded"
        expected_reason = "uncertain_delivery"
    elif raw["failed_count"]:
        expected_status = "degraded"
        expected_reason = "permanent_failure"
    elif raw["rejected_count"]:
        expected_status = "degraded"
        expected_reason = "retry_scheduled"
    elif claimed_count:
        expected_status = "healthy_batch"
        expected_reason = ""
    else:
        expected_status = "healthy_idle"
        expected_reason = ""
    special_zero_count_state = (
        claimed_count == 0
        and (
            (
                status == "store_unavailable"
                and reason
                in {
                    "detail_suppressed",
                    "invalid_encoding",
                    "invalid_row",
                    "invalid_schema",
                    "malformed_json",
                    "read_error",
                    "store_unavailable",
                }
            )
            or (
                status == "outcome_unknown"
                and reason
                in {
                    "detail_suppressed",
                    "invalid_health_status",
                    "invalid_limit",
                    "invalid_now",
                    "unknown",
                }
            )
        )
    )
    if not special_zero_count_state and (
        status != expected_status
        or reason
        not in (
            {expected_reason, "detail_suppressed"}
            if expected_reason
            else {expected_reason}
        )
    ):
        return _unknown_delivery_health("invalid_schema")
    age_seconds = max(0, int(now - checked_at_value))
    if age_seconds > REMINDER_DELIVERY_HEALTH_STALE_SECONDS:
        stale = _unknown_delivery_health("stale", stale=True, age_seconds=age_seconds)
        stale["last_status"] = status
        return stale
    return {
        **raw,
        "source_available": True,
        "stale": False,
        "age_seconds": age_seconds,
    }


def _normalize_id(value: object, seen_ids: set[str], fallback_seed: str) -> tuple[str, bool]:
    changed = False
    try:
        reminder_id = str(uuid.UUID(str(value)))
    except (TypeError, ValueError, AttributeError):
        reminder_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"jarvis-v2-reminder:{fallback_seed}"))
        changed = True
    if reminder_id in seen_ids:
        reminder_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"jarvis-v2-reminder-duplicate:{fallback_seed}"))
        while reminder_id in seen_ids:
            fallback_seed += ":duplicate"
            reminder_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"jarvis-v2-reminder-duplicate:{fallback_seed}"))
        changed = True
    seen_ids.add(reminder_id)
    return reminder_id, changed


def _normalize_reminder(
    value: object,
    seen_ids: set[str],
    fallback_seed: str = "compatibility",
) -> tuple[dict | None, bool]:
    if not isinstance(value, dict):
        return None, True
    due = _finite_epoch(value.get("due"))
    if due is None:
        return None, True

    reminder_id, changed = _normalize_id(value.get("id"), seen_ids, fallback_seed)
    raw_state = str(value.get("state") or "pending").strip().lower()
    state = "rejected" if raw_state == "retry" else raw_state
    if state not in _ACTIVE_STATES | _TERMINAL_STATES:
        state = "failed"
        changed = True

    transport = str(value.get("transport") or "telegram").strip().lower()
    message = _display_message(value.get("message", ""))
    chat_id = str(value.get("chat_id") or "").strip()
    raw_attempt_count = value.get("attempt_count")
    attempt_count_invalid = False
    if "attempt_count" not in value:
        parsed_attempt_count = 0
    elif type(raw_attempt_count) is int:
        parsed_attempt_count = raw_attempt_count
    elif (
        isinstance(raw_attempt_count, str)
        and re.fullmatch(r"[0-9]+", raw_attempt_count.strip()) is not None
        and len(raw_attempt_count.strip()) <= len(str(MAX_DELIVERY_ATTEMPTS))
    ):
        parsed_attempt_count = int(raw_attempt_count.strip())
    else:
        parsed_attempt_count = 0
        attempt_count_invalid = True
    if parsed_attempt_count < 0 or parsed_attempt_count > MAX_DELIVERY_ATTEMPTS:
        attempt_count_invalid = True
    attempt_count = max(0, min(parsed_attempt_count, MAX_DELIVERY_ATTEMPTS))
    if attempt_count_invalid:
        changed = True

    item: dict = {
        "id": reminder_id,
        "due": due,
        "message": message,
        "chat_id": chat_id,
        "transport": transport,
        "state": state,
        "attempt_count": attempt_count,
    }
    for key in ("claimed_at", "sending_at", "next_attempt_at", "finished_at"):
        epoch = _finite_epoch(value.get(key))
        if epoch is not None:
            item[key] = epoch
    token = str(value.get("token") or value.get("attempt_token") or "").strip()
    if token:
        item["token"] = token[:128]
    error_code = _safe_error_code(value.get("error_code"))
    if error_code:
        item["error_code"] = error_code
    telegram_message_id = value.get("telegram_message_id")
    if isinstance(telegram_message_id, int) and not isinstance(telegram_message_id, bool):
        item["telegram_message_id"] = telegram_message_id

    if attempt_count_invalid and state in _ACTIVE_STATES:
        item["state"] = "uncertain" if state == "sending" else "failed"
        item["finished_at"] = time.time()
        item["error_code"] = "invalid_attempt_count"
        changed = True
    elif state in {"pending", "claimed", "rejected"} and attempt_count >= MAX_DELIVERY_ATTEMPTS:
        item["state"] = "failed"
        item["finished_at"] = time.time()
        item["error_code"] = "retry_limit_reached"
        changed = True

    if item["state"] == "claimed" and (not token or "claimed_at" not in item):
        item["state"] = "pending"
        item.pop("token", None)
        item.pop("claimed_at", None)
        changed = True
    elif item["state"] == "sending" and (not token or "sending_at" not in item):
        item["state"] = "uncertain"
        item["finished_at"] = time.time()
        item["error_code"] = "incomplete_sending_state"
        changed = True

    if item["state"] in _TERMINAL_STATES:
        item.pop("message", None)
        item.pop("token", None)
        item.pop("next_attempt_at", None)
        if item["state"] != "accepted":
            item.pop("telegram_message_id", None)
    elif item["state"] != "rejected":
        item.pop("next_attempt_at", None)

    normalized_source = dict(value)
    if normalized_source != item:
        changed = True
    return item, changed


def _read_locked(
    reminders_file: Path,
    *,
    owner_scope: str | None = None,
) -> tuple[list[dict], bool]:
    try:
        raw = reminders_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return [], False
    except UnicodeError as exc:
        raise ReminderStoreUnavailable("invalid_encoding") from exc
    except OSError as exc:
        raise ReminderStoreUnavailable("read_error") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReminderStoreUnavailable("malformed_json") from exc
    if not isinstance(data, list):
        raise ReminderStoreUnavailable("invalid_schema")
    items: list[dict] = []
    changed = False
    seen_ids: set[str] = set()
    for index, value in enumerate(data):
        if owner_scope is not None and isinstance(value, dict):
            row_owner = str(value.get("chat_id") or "").strip()
            if row_owner != owner_scope:
                preserved = dict(value)
                try:
                    seen_ids.add(str(uuid.UUID(str(preserved.get("id")))))
                except (TypeError, ValueError, AttributeError):
                    pass
                items.append(preserved)
                continue
        due_seed = value.get("due") if isinstance(value, dict) else "invalid"
        item, item_changed = _normalize_reminder(value, seen_ids, f"read:{index}:{due_seed}")
        if item is None:
            raise ReminderStoreUnavailable("invalid_row")
        changed = changed or item_changed
        items.append(item)
    return items, changed


def _recover_stale(
    items: list[dict],
    now_epoch: float,
    owner_chat_id: str | None = None,
) -> bool:
    changed = False
    stale_before = now_epoch - DELIVERY_CLAIM_STALE_SECONDS
    for item in items:
        if owner_chat_id is not None and str(item.get("chat_id") or "").strip() != owner_chat_id:
            continue
        if item.get("state") == "claimed" and float(item.get("claimed_at", now_epoch)) <= stale_before:
            item["state"] = "pending"
            item.pop("token", None)
            item.pop("claimed_at", None)
            item.pop("error_code", None)
            changed = True
        elif item.get("state") == "sending" and float(item.get("sending_at", now_epoch)) <= stale_before:
            item["state"] = "uncertain"
            item["finished_at"] = now_epoch
            item["error_code"] = "stale_sending_recovered"
            item.pop("message", None)
            item.pop("token", None)
            item.pop("next_attempt_at", None)
            changed = True
    return changed


def _sync_parent_directory(reminders_file: Path) -> None:
    directory_fd = os.open(reminders_file.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _serialized_items(items: list[dict]) -> bytes:
    return json.dumps(items, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def _write_locked(reminders_file: Path, items: list[dict]) -> None:
    payload = _serialized_items(items)
    fd, raw_tmp = tempfile.mkstemp(
        prefix=f".{reminders_file.name}.", suffix=".tmp", dir=reminders_file.parent
    )
    tmp = Path(raw_tmp)
    published = False
    try:
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, reminders_file)
            published = True
            _sync_parent_directory(reminders_file)
        except Exception as exc:
            if published:
                raise ReminderWriteUncertain("directory_sync_failed") from exc
            raise
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _load() -> list[dict]:
    """Compatibility read-only view of nonterminal reminders."""
    try:
        items, _ = _read_locked(_reminders_file())
        _recover_stale(items, time.time())
        return [dict(item) for item in items if item.get("state") in _ACTIVE_STATES]
    except Exception:
        return []


def _valid_reminder(value: object) -> bool:
    if not isinstance(value, dict) or "due" not in value:
        return False
    try:
        return math.isfinite(float(value.get("due", 0)))
    except (TypeError, ValueError):
        return False


def _save(items: list[dict]) -> bool:
    """Compatibility writer that replaces active rows and retains delivery receipts."""
    try:
        with _locked_store() as reminders_file:
            existing, _ = _read_locked(reminders_file)
            terminal = [item for item in existing if item.get("state") in _TERMINAL_STATES]
            seen_ids = {str(item["id"]) for item in terminal}
            replacement: list[dict] = []
            for index, value in enumerate(items):
                due_seed = value.get("due") if isinstance(value, dict) else "invalid"
                item, _ = _normalize_reminder(value, seen_ids, f"save:{index}:{due_seed}")
                if item is not None:
                    replacement.append(item)
            _write_locked(reminders_file, terminal + replacement)
        return True
    except Exception:
        return False


def add_reminder_result(due_epoch: float, message: str, chat_id: str) -> ReminderWriteResult:
    due = _finite_epoch(due_epoch)
    if due is None:
        return ReminderWriteResult("failed", reason="invalid_due")
    reminder_id = str(uuid.uuid4())
    try:
        with _locked_store() as reminders_file:
            items, _ = _read_locked(reminders_file)
            items.append({
                "id": reminder_id,
                "due": due,
                "message": _display_message(message),
                "chat_id": str(chat_id).strip(),
                "transport": "telegram",
                "state": "pending",
                "attempt_count": 0,
            })
            _write_locked(reminders_file, items)
        return ReminderWriteResult("stored", reminder_id)
    except ReminderWriteUncertain as exc:
        return ReminderWriteResult("published_uncertain", reminder_id, _safe_error_code(exc, "directory_sync_failed"))
    except Exception as exc:
        return ReminderWriteResult("failed", reason=_safe_error_code(type(exc).__name__, "write_error"))


def add_reminder(due_epoch: float, message: str, chat_id: str) -> bool:
    """Compatibility wrapper: true means the row was published, not durability-proven."""
    return add_reminder_result(due_epoch, message, chat_id).published


def _quarantine_unsafe(items: list[dict], owner_chat_id: str, now_epoch: float) -> bool:
    changed = False
    for item in items:
        if item.get("state") not in _ACTIVE_STATES:
            continue
        error_code = ""
        if item.get("transport") != "telegram":
            error_code = "unknown_transport"
        elif owner_chat_id and str(item.get("chat_id") or "") != owner_chat_id:
            error_code = "owner_mismatch"
        if error_code:
            item["state"] = "failed"
            item["finished_at"] = now_epoch
            item["error_code"] = error_code
            item.pop("message", None)
            item.pop("token", None)
            item.pop("next_attempt_at", None)
            changed = True
    return changed


def _delivery_view(item: dict, *, include_token: bool = False) -> dict:
    result = {
        "id": str(item["id"]),
        "due": float(item["due"]),
        "message": _display_message(item.get("message", "")),
        "chat_id": str(item.get("chat_id") or ""),
        "transport": "telegram",
        "state": str(item.get("state") or "pending"),
        "attempt_count": int(item.get("attempt_count") or 0),
    }
    if item.get("next_attempt_at") is not None:
        result["next_attempt_at"] = float(item["next_attempt_at"])
    if include_token:
        result["token"] = str(item["token"])
    return result


def claim_due_result(
    now_epoch: float | None = None,
    owner_chat_id: str = "",
    *,
    limit: int = REMINDER_DELIVERY_BATCH_LIMIT,
) -> ReminderClaimResult:
    """Atomically claim one bounded due batch with explicit store-availability truth."""
    if type(limit) is not int or limit < 1 or limit > REMINDER_DELIVERY_BATCH_LIMIT:
        return ReminderClaimResult("invalid", reason="invalid_limit")
    now = time.time() if now_epoch is None else _finite_epoch(now_epoch)
    if now is None:
        return ReminderClaimResult("invalid", reason="invalid_now")
    owner = str(owner_chat_id or "").strip()
    try:
        with _locked_store() as reminders_file:
            items, changed = _read_locked(reminders_file)
            changed = _recover_stale(items, now) or changed
            changed = _quarantine_unsafe(items, owner, now) or changed
            claimed: list[dict] = []
            for item in sorted(items, key=lambda row: (float(row["due"]), str(row["id"]))):
                state = item.get("state")
                ready_at = item.get("next_attempt_at", item["due"])
                if state not in {"pending", "rejected"} or float(ready_at) > now:
                    continue
                token = str(uuid.uuid4())
                item["state"] = "claimed"
                item["token"] = token
                item["claimed_at"] = now
                item.pop("next_attempt_at", None)
                item.pop("finished_at", None)
                item.pop("error_code", None)
                item.pop("telegram_message_id", None)
                claimed.append(_delivery_view(item, include_token=True))
                changed = True
                if len(claimed) >= limit:
                    break
            if changed:
                _write_locked(reminders_file, items)
            return ReminderClaimResult("ok", tuple(claimed))
    except ReminderStoreUnavailable as exc:
        return ReminderClaimResult("store_unavailable", reason=exc.reason)
    except Exception as exc:
        return ReminderClaimResult(
            "store_unavailable",
            reason=_safe_error_code(type(exc).__name__, "read_error"),
        )


def claim_due(
    now_epoch: float | None = None,
    owner_chat_id: str = "",
    *,
    limit: int = REMINDER_DELIVERY_BATCH_LIMIT,
) -> list[dict]:
    """Compatibility wrapper returning only one bounded fenced claim batch."""
    return list(
        claim_due_result(
            now_epoch=now_epoch,
            owner_chat_id=owner_chat_id,
            limit=limit,
        ).items
    )


def _exact_reminder_row(reminders_file: Path, reminder_id: str) -> dict | None:
    items, _ = _read_locked(reminders_file)
    for item in items:
        if item.get("id") == reminder_id:
            return item
    return None


def _read_expected_reminder(
    reminders_file: Path,
    reminder_id: str,
    expected: dict,
) -> tuple[bool, str]:
    reason = "readback_mismatch"
    for _attempt in range(2):
        try:
            if _exact_reminder_row(reminders_file, reminder_id) == expected:
                return True, ""
            reason = "readback_mismatch"
        except ReminderStoreUnavailable as exc:
            reason = exc.reason
        except Exception as exc:
            reason = _safe_error_code(type(exc).__name__, "readback_failed")
    return False, reason


def _compensate_unstarted_begin(
    reminders_file: Path,
    items: list[dict],
    reminder_id: str,
    token: str,
    prior_attempt_count: int,
) -> ReminderBeginResult:
    """Restore a published pre-send transition to retryable state."""
    rollback_items = [dict(item) for item in items]
    expected: dict | None = None
    for item in rollback_items:
        if (
            item.get("id") == reminder_id
            and item.get("state") == "sending"
            and item.get("token") == token
        ):
            item["state"] = "pending"
            item["attempt_count"] = prior_attempt_count
            item.pop("token", None)
            item.pop("claimed_at", None)
            item.pop("sending_at", None)
            item.pop("next_attempt_at", None)
            item.pop("error_code", None)
            expected = dict(item)
            break
    if expected is None:
        return ReminderBeginResult("unavailable", "compensation_target_changed")

    current_payload = _serialized_items(items)
    try:
        if reminders_file.read_bytes() != current_payload:
            return ReminderBeginResult("unavailable", "compensation_generation_changed")
    except Exception:
        return ReminderBeginResult("unavailable", "compensation_readback_failed")
    try:
        _write_locked(reminders_file, rollback_items)
    except ReminderWriteUncertain:
        return ReminderBeginResult("unavailable", "compensation_durability_unproven")
    except Exception:
        return ReminderBeginResult("unavailable", "compensation_write_failed")
    try:
        if reminders_file.read_bytes() != _serialized_items(rollback_items):
            return ReminderBeginResult("unavailable", "compensation_readback_mismatch")
    except Exception:
        return ReminderBeginResult("unavailable", "compensation_readback_failed")
    return ReminderBeginResult("retryable", "pre_send_compensated")


def begin_delivery_result(reminder_id: str, token: str) -> ReminderBeginResult:
    """Persist and verify the pre-network delivery fence."""
    reminder_id = str(reminder_id or "").strip()
    token = str(token or "").strip()
    if not reminder_id or not token:
        return ReminderBeginResult("not_claimed", "invalid_fence")
    try:
        with _locked_store() as reminders_file:
            items, changed = _read_locked(reminders_file)
            for item in items:
                if (
                    item.get("id") == reminder_id
                    and item.get("state") == "claimed"
                    and item.get("token") == token
                ):
                    prior_attempt_count = int(item.get("attempt_count") or 0)
                    item["state"] = "sending"
                    item["sending_at"] = time.time()
                    item["attempt_count"] = prior_attempt_count + 1
                    expected = dict(item)
                    try:
                        _write_locked(reminders_file, items)
                    except ReminderWriteUncertain:
                        matches, reason = _read_expected_reminder(
                            reminders_file,
                            reminder_id,
                            expected,
                        )
                        if not matches:
                            compensated = _compensate_unstarted_begin(
                                reminders_file,
                                items,
                                reminder_id,
                                token,
                                prior_attempt_count,
                            )
                            if compensated.status == "retryable":
                                return compensated
                            return ReminderBeginResult("unavailable", f"sending_{reason}")
                        try:
                            _sync_parent_directory(reminders_file)
                        except Exception:
                            return _compensate_unstarted_begin(
                                reminders_file,
                                items,
                                reminder_id,
                                token,
                                prior_attempt_count,
                            )
                        return ReminderBeginResult("started", "directory_resynced")
                    return ReminderBeginResult("started")
            if changed:
                _write_locked(reminders_file, items)
            return ReminderBeginResult("not_claimed", "fence_mismatch")
    except ReminderStoreUnavailable as exc:
        return ReminderBeginResult("unavailable", exc.reason)
    except Exception as exc:
        return ReminderBeginResult("unavailable", _safe_error_code(type(exc).__name__, "write_error"))


def begin_delivery(reminder_id: str, token: str) -> bool:
    """Compatibility wrapper: true grants this claim one network attempt."""
    return begin_delivery_result(reminder_id, token).network_eligible


def finish_delivery(
    reminder_id: str,
    token: str,
    disposition: str,
    error_code: str = "",
    retry_after: float | None = None,
    telegram_message_id: int | None = None,
) -> bool:
    """Finish a token-fenced send with an explicit delivery disposition."""
    dispositions = {
        "accepted": "accepted",
        "sent": "accepted",
        "delivered": "accepted",
        "rejected": "rejected",
        "retry": "rejected",
        "retryable": "rejected",
        "failed": "failed",
        "permanent": "failed",
        "uncertain": "uncertain",
        "unknown": "uncertain",
    }
    requested = dispositions.get(str(disposition or "").strip().lower())
    if requested is None:
        raise ValueError("invalid reminder delivery disposition")
    if telegram_message_id is not None and requested != "accepted":
        raise ValueError("only accepted reminder deliveries may store a Telegram receipt")
    if requested == "accepted" and (
        not isinstance(telegram_message_id, int)
        or isinstance(telegram_message_id, bool)
        or telegram_message_id <= 0
    ):
        raise ValueError("invalid Telegram message id")
    if requested == "rejected":
        if retry_after is None:
            retry_seconds = 0
        else:
            retry_value = _finite_epoch(retry_after)
            if retry_value is None:
                raise ValueError("invalid reminder retry delay")
            retry_seconds = max(0, min(int(retry_value), MAX_RETRY_AFTER_SECONDS))
    else:
        retry_seconds = 0

    reminder_id = str(reminder_id or "").strip()
    token = str(token or "").strip()
    if not reminder_id or not token:
        return False
    now = time.time()
    try:
        with _locked_store() as reminders_file:
            items, changed = _read_locked(reminders_file)
            for item in items:
                if (
                    item.get("id") != reminder_id
                    or item.get("state") != "sending"
                    or item.get("token") != token
                ):
                    continue
                effective = requested
                safe_error = _safe_error_code(error_code)
                if requested == "rejected" and int(item.get("attempt_count") or 0) >= MAX_DELIVERY_ATTEMPTS:
                    effective = "failed"
                    safe_error = safe_error or "retry_limit_reached"
                item["state"] = effective
                item["finished_at"] = now
                item.pop("token", None)
                if effective == "rejected":
                    item["next_attempt_at"] = now + retry_seconds
                    item.pop("sending_at", None)
                else:
                    item.pop("message", None)
                    item.pop("next_attempt_at", None)
                    if effective == "accepted" and telegram_message_id is not None:
                        item["telegram_message_id"] = telegram_message_id
                    else:
                        item.pop("telegram_message_id", None)
                if safe_error:
                    item["error_code"] = safe_error
                else:
                    item.pop("error_code", None)
                _write_locked(reminders_file, items)
                return True
            if changed:
                _write_locked(reminders_file, items)
            return False
    except Exception:
        return False


def _cancel_pre_network_item(item: dict, now_epoch: float) -> dict:
    preview = _delivery_view(item)
    item["state"] = "cancelled"
    item["finished_at"] = now_epoch
    item.pop("message", None)
    item.pop("token", None)
    item.pop("claimed_at", None)
    item.pop("next_attempt_at", None)
    item.pop("error_code", None)
    return preview


def _normalize_reminder_selector(value: object) -> str:
    selector = str(value or "").strip().lower()
    if selector.startswith("#"):
        selector = selector[1:]
    selector = selector.replace("-", "")
    return selector if re.fullmatch(r"[0-9a-f]{8,32}", selector) else ""


def cancel_pending_item(owner_chat_id: str, selector: object) -> tuple[str, list[dict], bool]:
    """Cancel one uniquely selected reminder without racing its send boundary."""
    owner = str(owner_chat_id or "").strip()
    if not owner:
        return "owner_missing", [], False
    normalized_selector = _normalize_reminder_selector(selector)
    if not normalized_selector:
        return "invalid_selector", [], True
    now = time.time()
    preview: dict | None = None
    try:
        with _locked_store() as reminders_file:
            items, _ = _read_locked(reminders_file, owner_scope=owner)
            matches = [
                item
                for item in items
                if item.get("state") in _ACTIVE_STATES
                and str(item.get("chat_id") or "") == owner
                and str(item.get("id") or "").replace("-", "").lower().startswith(normalized_selector)
            ]
            if len(matches) != 1:
                return ("ambiguous" if matches else "not_found"), [], True
            item = matches[0]
            if item.get("state") == "sending":
                return "in_flight", [], True
            preview = _cancel_pre_network_item(item, now)
            _write_locked(reminders_file, items)
            return "cancelled", [preview], True
    except ReminderWriteUncertain:
        return "published_uncertain", [preview] if preview is not None else [], False
    except Exception:
        return "storage_error", [], False


def cancel_pending_items_status(owner_chat_id: str = "") -> tuple[list[dict], bool, int, str]:
    """Cancel pre-network reminders and count sends already past the safe boundary."""
    owner = str(owner_chat_id or "").strip()
    if not owner:
        return [], False, 0, "owner_missing"
    now = time.time()
    cancelled: list[dict] = []
    in_flight_count = 0
    try:
        with _locked_store() as reminders_file:
            items, _ = _read_locked(reminders_file, owner_scope=owner)
            in_flight_count = sum(
                1
                for item in items
                if item.get("state") == "sending"
                and str(item.get("chat_id") or "") == owner
            )
            for item in items:
                if item.get("state") not in {"pending", "claimed", "rejected"}:
                    continue
                if str(item.get("chat_id") or "") != owner:
                    continue
                cancelled.append(_cancel_pre_network_item(item, now))
            if cancelled:
                _write_locked(reminders_file, items)
            status = "cancelled" if cancelled else ("in_flight" if in_flight_count else "empty")
            return cancelled, True, in_flight_count, status
    except ReminderWriteUncertain:
        return cancelled, False, in_flight_count, "published_uncertain"
    except Exception:
        return [], False, 0, "storage_error"


def cancel_pending_items(owner_chat_id: str = "") -> tuple[list[dict], bool]:
    """Compatibility wrapper returning cancelled previews plus write status."""
    cancelled, saved, _, _ = cancel_pending_items_status(owner_chat_id)
    return cancelled, saved


def cancel_pending(owner_chat_id: str = "") -> int:
    """Cancel pending/pre-network reminders for one owner, returning the count."""
    cancelled, saved = cancel_pending_items(owner_chat_id)
    return len(cancelled) if saved else 0


def pending_reminders(owner_chat_id: str = "") -> list[dict]:
    """Return active reminder previews without exposing delivery tokens."""
    items, _, _ = pending_reminders_status(owner_chat_id)
    return items


def pending_reminders_status(owner_chat_id: str = "") -> tuple[list[dict], bool, str]:
    """Return reminder previews plus bounded source-availability truth."""
    owner = str(owner_chat_id or "").strip()
    if not owner:
        return [], False, "owner_missing"
    now = time.time()
    try:
        items, _ = _read_locked(_reminders_file())
        _recover_stale(items, now)
        visible = [
            _delivery_view(item)
            for item in items
            if item.get("state") in _ACTIVE_STATES
            and str(item.get("chat_id") or "") == owner
        ]
        return sorted(visible, key=lambda row: (float(row["due"]), str(row["id"]))), True, ""
    except ReminderStoreUnavailable as exc:
        return [], False, exc.reason
    except Exception:
        return [], False, "read_error"


def pop_due(now_epoch: float | None = None) -> list[dict]:
    """Compatibility shim: destructively pop due rows without delivery fencing."""
    now = time.time() if now_epoch is None else _finite_epoch(now_epoch)
    if now is None:
        return []
    try:
        with _locked_store() as reminders_file:
            items, changed = _read_locked(reminders_file)
            due_ids = {
                str(item["id"])
                for item in items
                if item.get("state") in {"pending", "rejected"}
                and float(item.get("next_attempt_at", item["due"])) <= now
            }
            due = [dict(item) for item in items if str(item["id"]) in due_ids]
            if due_ids or changed:
                _write_locked(reminders_file, [item for item in items if str(item["id"]) not in due_ids])
            return sorted(due, key=lambda row: (float(row["due"]), str(row["id"])))
    except Exception:
        return []


def _clean_message(text: str) -> str:
    t = text.strip()
    t = re.sub(r"\bremind me\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(?:ping|notify) me\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bwake me(?: up)?\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bset (a )?timer( for)?\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bset (?:an? )?alarm( for)?\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(?:an? )?alarm(?:\s+for)?\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(a )?timer(?:\s+for)?\b", "", t, flags=re.IGNORECASE)
    t = re.sub(rf"\bin\s+{_DURATION_RE}\b", "", t, flags=re.IGNORECASE)
    t = re.sub(rf"\b{_DURATION_RE}\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bat\s+\d{1,2}(?::\d{2})?\s*(am|pm)?\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b\d{1,2}(?::\d{2})?\s*(am|pm)\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(at\s+)?(noon|midnight)\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(today|tonight|tomorrow|(?:next|on)\s+(?:" + "|".join(_WEEKDAYS) + r")|" + "|".join(_WEEKDAYS) + r")\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*(to|that|about)\b\s+", "", t.strip(), flags=re.IGNORECASE)
    t = re.sub(r"\b(to|that|about)\b\s*$", "", t.strip(), flags=re.IGNORECASE)
    return " ".join(t.split()).strip(" ,.-")


def _parse_duration_delta(low: str) -> timedelta | None:
    match = (
        re.search(rf"\bin\s+(?P<duration>{_DURATION_RE})\b", low)
        or re.search(rf"\b(?:for|timer for|timer of)\s+(?P<duration>{_DURATION_RE})\b", low)
        or re.search(rf"\btimer\s+(?P<duration>{_DURATION_RE})\b", low)
        or re.search(rf"\b(?P<duration>{_DURATION_RE})\s+timer\b", low)
    )
    if not match:
        return None
    try:
        delta = timedelta()
        part_re = rf"(?:(\d+)\s*({_DURATION_UNIT_RE})|({_DURATION_NATURAL_AMOUNT_RE})\s+({_DURATION_WORD_UNIT_RE}))"
        for part in re.finditer(part_re, match.group("duration"), re.IGNORECASE):
            raw_amount = (part.group(1) or part.group(3) or "").lower().strip()
            if raw_amount.startswith("half"):
                n = 0.5
            elif raw_amount in {"a", "an"}:
                n = 1
            else:
                n = int(raw_amount)
            unit = (part.group(2) or part.group(4) or "").lower()
            if unit.startswith(("second", "sec")) or unit == "s":
                delta += timedelta(seconds=n)
            elif unit.startswith(("hour", "hr")) or unit == "h":
                delta += timedelta(hours=n)
            elif unit.startswith("day") or unit == "d":
                delta += timedelta(days=n)
            elif unit.startswith("week") or unit == "w":
                delta += timedelta(weeks=n)
            else:
                delta += timedelta(minutes=n)
    except OverflowError:
        return None
    return delta if delta.total_seconds() > 0 else None


def is_recurring_request(text: str) -> bool:
    return bool(_RECURRING_RE.search(str(text or "")))


def has_local_path(value: object) -> bool:
    return bool(LOCAL_PATH_RE.search(str(value or "")))


def _display_message(value: object, *, limit: int = MAX_MESSAGE_CHARS) -> str:
    text = "" if value is None else str(value).strip()
    text = " ".join(text.split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def parse_when(text: str, now: datetime | None = None) -> tuple[datetime, str] | None:
    """Parse a reminder/timer request into (due_datetime, message), or None."""
    now = now or datetime.now().astimezone()
    low = " ".join(text.lower().split())
    due: datetime | None = None

    duration_delta = _parse_duration_delta(low)
    if duration_delta is not None:
        try:
            due = now + duration_delta
        except OverflowError:
            return None
    else:
        hm = _parse_time(low)
        if hm is not None:
            day = now
            if re.search(r"\btomorrow\b", low):
                day = now + timedelta(days=1)
            else:
                for name, idx in _WEEKDAYS.items():
                    if re.search(rf"\b{name}\b", low):
                        ahead = (idx - now.weekday()) % 7
                        day = now + timedelta(days=ahead)
                        break
            candidate = day.replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)
            if candidate <= now and not re.search(r"\b(tomorrow|" + "|".join(_WEEKDAYS) + r")\b", low):
                candidate += timedelta(days=1)  # "remind me at 7am" when it's already past -> next day
            due = candidate

    if due is None:
        return None
    message = _clean_message(text) or "⏰ Reminder!"
    return due, message
