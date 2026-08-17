from __future__ import annotations

import json
import hashlib
import os
import re
import secrets
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Event, Lock, Thread
from typing import Any, Callable

from jarvis_v2.agent.types import ToolResult
from jarvis_v2.automations.jobs import (
    GoalNudgeBuild,
    build_daily_brief,
    build_goal_nudge,
    build_weekly_review,
)
from jarvis_v2.config import JarvisConfig, load_config
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import MemoryStore
from jarvis_v2.tools.ingest import build_recent_file_digest, ingest_inbox_notes
from jarvis_v2.tools.next_step import make_next_step_tools
from jarvis_v2.tools.state import make_state_tools

MORNING_BRIEF_JOB_NAME = "Morning Brief"
MORNING_BRIEF_JOB_TYPE = "morning_brief_telegram"
MORNING_BRIEF_INTERVAL_MINUTES = 1440
MORNING_BRIEF_DEFAULT_HHMM = "09:00"
MORNING_BRIEF_ENV = "JARVIS_MORNING_BRIEF"
LEGACY_MORNING_BRIEF_ENV = "JARVIS_MORNING_BRIEF_HHMM"
JOB_RUN_HISTORY_KEY = "run_history"
JOB_RUN_HISTORY_LIMIT = 64
JOB_LEASE_SECONDS = 300.0
JOB_HEARTBEAT_SECONDS = 60.0
JOB_FAILURE_RETRY_SECONDS = 300
STATE_SNAPSHOT_RECEIPT_STALE_SECONDS = 60 * 60
DELIVERY_CLAIM_SECONDS = 120
DELIVERY_RETRY_SECONDS = 300
DELIVERY_DRAIN_LIMIT = 16
DELIVERY_MAX_CHUNKS = 32
DELIVERY_MAX_PAYLOAD_CHARS = 131072
DELIVERY_TERMINAL_RETENTION_DAYS = 30
DELIVERY_UNCERTAIN_RETENTION_DAYS = 90
NOTE_PUBLICATION_CLAIM_SECONDS = 120
NOTE_PUBLICATION_DRAIN_LIMIT = 16
SCHEDULER_RECOVERY_INSPECTION = (
    "Inspect `list scheduled jobs`, run `setup check`, and repair any reported "
    "storage, model, or connector issue before the automatic retry."
)
SCHEDULER_STATUS_INSPECTION = (
    "Inspect `list scheduled jobs`, run `setup check`, and repair any reported "
    "storage, model, or connector issue before trusting scheduler status."
)
SCHEDULER_DELIVERY_RECEIPT_INSPECTION = (
    "Inspect the durable delivery receipt state in `list scheduled jobs`; "
    "do not manually rerun or resend an outcome-unknown delivery."
)

# Digest jobs whose output should be pushed to the owner's Telegram when the
# ticker fires them. Without this, run_due_jobs returned their text to the
# daemon log — real reports that nobody ever saw. Morning Brief is absent
# because it delivers itself (with per-day dedup) inside _run_job_type.
DELIVERED_JOB_TITLES: dict[str, str] = {
    "daily_brief": "📋 Daily Brief",
    "goal_nudge": "🎯 Goal Nudge",
    "weekly_review": "🗓 Weekly Review",
    "inbox_ingest": "📥 Inbox Ingest",
    "recent_file_digest": "🗂 Recent File Digest",
    # state_snapshot is deliberately absent: its job is writing local
    # continuity notes (Current Context / Mission Control), not reporting.
    # Delivering it would push 4 near-identical messages a day.
}

# When a job's output contains its marker, there is nothing new to report —
# skip the Telegram push instead of waking the owner for an empty digest.
# (The job still runs, still writes its vault note, still reschedules.)
QUIET_OUTPUT_MARKERS: dict[str, tuple[str, ...]] = {
    "goal_nudge": ("- No stale active goals.",),
    "inbox_ingest": ("Obsidian Inbox has no ingestible notes.",),
    "recent_file_digest": (
        "No recent files found.",
        "No watched directories configured.",
    ),
}
SCHEDULED_NOTE_JOB_TYPES = frozenset(
    {"daily_brief", "goal_nudge", "weekly_review", "recent_file_digest"}
)
SCHEDULED_NOTE_TITLES: dict[str, str] = {
    "daily_brief": "Jarvis Proactive Brief",
    "goal_nudge": "Jarvis Goal Nudge",
    "weekly_review": "Weekly Review",
    "recent_file_digest": "Jarvis Recent File Digest",
}
_HHMM_RE = re.compile(r"^\s*(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>am|pm)?\s*$", re.IGNORECASE)


def iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def _run_marker(dt: datetime) -> str:
    return dt.isoformat(timespec="microseconds")


def _scheduler_recovery_guidance(
    *,
    automatic_retry: bool = True,
    delivery_receipt_may_be_unknown: bool = False,
) -> str:
    guidance = (
        SCHEDULER_RECOVERY_INSPECTION
        if automatic_retry
        else SCHEDULER_STATUS_INSPECTION
    )
    if delivery_receipt_may_be_unknown:
        guidance = f"{guidance} {SCHEDULER_DELIVERY_RECEIPT_INSPECTION}"
    return guidance


def _positive_seconds(value: object, fallback: float, minimum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(parsed, minimum)


def parse_hhmm(value: object) -> tuple[int, int] | None:
    text = str(value or "").strip().lower()
    match = _HHMM_RE.match(text)
    if not match:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or "0")
    ampm = match.group("ampm")
    if minute > 59:
        return None
    if ampm:
        if hour < 1 or hour > 12:
            return None
        if ampm == "am":
            hour = 0 if hour == 12 else hour
        else:
            hour = 12 if hour == 12 else hour + 12
    elif hour > 23:
        return None
    return hour, minute


def format_hhmm(hour: int, minute: int) -> str:
    return f"{hour:02d}:{minute:02d}"


def next_daily_run(hour: int, minute: int, now: datetime | None = None) -> datetime:
    base = now or datetime.now()
    candidate = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= base:
        candidate += timedelta(days=1)
    return candidate


def _job_metadata(row: Any | None) -> dict[str, Any]:
    if row is None:
        return {}
    try:
        raw = row["metadata"]
    except (KeyError, IndexError):
        return {}
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _job_occurrence_schedule_revision(row: Any) -> int:
    try:
        value = row["active_occurrence_schedule_revision"]
        if not isinstance(value, bool) and isinstance(value, int):
            return value
    except (KeyError, IndexError, TypeError):
        pass
    return int(row["schedule_revision"])


def _job_occurrence_schedule_identity_revision(row: Any) -> int:
    try:
        value = row["active_occurrence_schedule_identity_revision"]
        if not isinstance(value, bool) and isinstance(value, int) and value >= 0:
            return value
        if value == -1 and _job_occurrence_type(row) == "legacy_unknown":
            return value
    except (KeyError, IndexError, TypeError):
        pass
    value = row["schedule_identity_revision"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError("scheduled_occurrence_identity_revision_invalid")
    return value


def _job_occurrence_type(row: Any) -> str:
    try:
        value = row["active_occurrence_job_type"]
        active_key = str(row["active_occurrence_key"] or "")
    except (KeyError, IndexError, TypeError):
        value = None
        active_key = ""
    if type(value) is str and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value):
        return value
    if re.fullmatch(r"scheduler-occurrence:v1:[0-9a-f]{64}", active_key):
        raise RuntimeError("scheduled_occurrence_job_type_invalid")
    current = str(row["job_type"] or "")
    if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", current) is None:
        raise RuntimeError("scheduled_job_type_invalid")
    return current


def _safe_projection_path_display(value: object) -> str:
    if type(value) is not str or not 0 < len(value) <= 4096:
        return ""
    if value.startswith(("/", "\\")) or "\\" in value:
        return ""
    if any(char in value for char in ("\n", "\r", "\x00")):
        return ""
    if ".." in value.split("/"):
        return ""
    return value


def _safe_projection_digest(value: object) -> str:
    return value if type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) else ""


def _classify_job_run_status(output: object) -> str:
    text = str(output or "").casefold()
    if "reconciled as" in text:
        return "reconciled"
    if "held for review" in text or "receipt identity collision" in text:
        return "held"
    if (
        "not sent:" in text
        or "unknown job type:" in text
        or "delivery failed:" in text
        or "job failed:" in text
        or "outcome is unknown" in text
        or "retry limit" in text
        or "automatic retries stopped" in text
    ):
        return "failed"
    if (
        "already sent today" in text
        or "already accepted by telegram api today" in text
        or "delivery skipped" in text
        or "nothing to deliver" in text
    ):
        return "skipped"
    if "quiet" in text and "nothing new" in text:
        return "skipped"
    return "ok"


def _durable_job_run_status(output: object) -> str:
    status = _classify_job_run_status(output)
    if status == "reconciled":
        return "skipped"
    if status == "held":
        return "failed"
    return status


def _safe_job_failure_code(exc: Exception) -> str:
    code = type(exc).__name__
    if len(code) <= 64 and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", code):
        return code
    return "JobHandlerError"


def _delivery_key_part(value: object) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8", errors="replace")).hexdigest()[:32]


def _payload_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_telegram_message_id(result: object) -> int | None:
    if not isinstance(result, dict):
        return None
    payload = result.get("result")
    if not isinstance(payload, dict):
        return None
    value = payload.get("message_id")
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _telegram_failure_policy(result: object) -> tuple[str, str, int]:
    if not isinstance(result, dict) or result.get("ok") is not False:
        return "uncertain", "malformed_response", 0
    nested = result.get("results")
    if isinstance(nested, list) and any(isinstance(item, dict) and item.get("ok") is True for item in nested):
        return "uncertain", "partial_acceptance", 0
    raw_error = str(result.get("error") or "telegram_rejected").strip().casefold()
    error_code = _safe_telegram_delivery_error(raw_error)
    try:
        http_status = int(result.get("http_status"))
    except (TypeError, ValueError, OverflowError):
        http_status = 0
    try:
        retry_after = int(result.get("retry_after"))
    except (TypeError, ValueError, OverflowError):
        retry_after = DELIVERY_RETRY_SECONDS
    retry_after = max(0, min(retry_after, 86400))
    permanent_errors = {
        "missing_or_invalid_owner",
        "no token",
        "invalid token",
        "no_token",
        "invalid_token",
    }
    if raw_error in permanent_errors or http_status in {400, 401, 403, 404}:
        return "failed", error_code, 0
    if http_status == 429 or http_status >= 500:
        return "rejected", error_code, retry_after
    return "rejected", error_code, DELIVERY_RETRY_SECONDS


@dataclass(frozen=True)
class _DeliveryPlan:
    occurrence_key: str
    chunks: tuple[dict[str, Any], ...]
    existing: bool = False


@dataclass(frozen=True)
class _DeliveryDispatch:
    state: str
    output: str


def _delivery_receipt_set_state(rows: object, occurrence_key: str) -> str:
    """Classify one structurally complete, identity-consistent delivery receipt set."""
    if not isinstance(rows, (list, tuple)) or not rows:
        return "invalid"
    expected_count: int | None = None
    expected_job_id: int | None = None
    expected_job_id_set = False
    expected_source_job_id: int | None = None
    expected_job_type: str | None = None
    indexes: set[int] = set()
    states: set[str] = set()
    for row in rows:
        try:
            row_occurrence = row["occurrence_key"]
            index = row["chunk_index"]
            count = row["chunk_count"]
            job_id = row["job_id"]
            source_job_id = row["source_job_id"]
            job_type = row["job_type"]
            state = row["state"]
        except (KeyError, IndexError, TypeError):
            return "invalid"
        if (
            row_occurrence != occurrence_key
            or type(index) is not int
            or type(count) is not int
            or index < 0
            or count < 1
            or count > DELIVERY_MAX_CHUNKS
            or index >= count
            or index in indexes
            or type(source_job_id) is not int
            or source_job_id < 1
            or type(job_type) is not str
            or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", job_type) is None
            or state
            not in {"prepared", "claimed", "sending", "accepted", "rejected", "uncertain", "failed"}
            or (expected_count is not None and count != expected_count)
            or (expected_job_id_set and job_id != expected_job_id)
            or (expected_source_job_id is not None and source_job_id != expected_source_job_id)
            or (expected_job_type is not None and job_type != expected_job_type)
        ):
            return "invalid"
        is_morning_receipt = job_type == MORNING_BRIEF_JOB_TYPE and occurrence_key.startswith("morning:")
        if is_morning_receipt:
            if job_id is not None and (type(job_id) is not int or job_id < 1):
                return "invalid"
        elif type(job_id) is not int or job_id < 1 or source_job_id != job_id:
            return "invalid"
        expected_count = count
        expected_job_id = job_id
        expected_job_id_set = True
        expected_source_job_id = source_job_id
        expected_job_type = job_type
        indexes.add(index)
        states.add(state)
    if expected_count is None or len(rows) != expected_count or indexes != set(range(expected_count)):
        return "invalid"
    if "uncertain" in states:
        return "uncertain"
    if "failed" in states:
        return "failed"
    if "rejected" in states:
        return "rejected"
    if states == {"accepted"}:
        return "accepted"
    return "pending"


@dataclass(frozen=True)
class _ScheduledNotePlan:
    publication_key: str
    occurrence_key: str
    delivery_occurrence_key: str
    effect_index: int
    job_id: int
    source_job_id: int
    job_name: str
    job_type: str
    target_kind: str
    target_date: str
    title: str
    payload: str
    payload_sha256: str
    source_manifest_kind: str
    source_manifest_digest: str
    allow_disabled: bool

    def as_store_plan(self) -> dict[str, Any]:
        return {
            "publication_key": self.publication_key,
            "occurrence_key": self.occurrence_key,
            "delivery_occurrence_key": self.delivery_occurrence_key or None,
            "effect_index": self.effect_index,
            "job_id": self.job_id,
            "source_job_id": self.source_job_id,
            "job_name": self.job_name,
            "job_type": self.job_type,
            "target_kind": self.target_kind,
            "target_date": self.target_date,
            "title": self.title,
            "payload": self.payload,
            "payload_sha256": self.payload_sha256,
            "source_manifest_kind": self.source_manifest_kind,
            "source_manifest_digest": self.source_manifest_digest,
            "allow_disabled": self.allow_disabled,
        }


@dataclass(frozen=True)
class _ScheduledNoteDispatch:
    state: str
    output: str


@dataclass(frozen=True)
class _JobHandlerResult:
    ok: bool
    output: str
    failure_code: str = ""
    source_manifest_kind: str = ""
    source_manifest_digest: str = ""


def _job_handler_result(value: object, job_type: str) -> _JobHandlerResult:
    if isinstance(value, GoalNudgeBuild):
        if (
            value.source_manifest_kind != "goal_nudge_v1"
            or re.fullmatch(r"[0-9a-f]{64}", value.source_manifest_digest) is None
        ):
            return _JobHandlerResult(False, "", "invalid_goal_nudge_source_manifest")
        return _JobHandlerResult(
            True,
            value.output,
            source_manifest_kind=value.source_manifest_kind,
            source_manifest_digest=value.source_manifest_digest,
        )
    if isinstance(value, ToolResult):
        if type(value.ok) is not bool:
            return _JobHandlerResult(False, "", "invalid_tool_result")
        if value.ok is False:
            safe_type = re.sub(r"[^a-z0-9_]+", "_", str(job_type).casefold()).strip("_")
            code = f"{safe_type[:48]}_failed" if safe_type else "scheduled_handler_failed"
            return _JobHandlerResult(False, "", code[:64])
        return _JobHandlerResult(True, str(value.output or ""))
    return _JobHandlerResult(True, str(value or ""))


def _row_next_run_hhmm(row: Any) -> str:
    try:
        next_run = datetime.fromisoformat(str(row["next_run_at"]))
    except (TypeError, ValueError):
        return ""
    return format_hhmm(next_run.hour, next_run.minute)


def job_enabled(row: Any) -> bool:
    try:
        value = row["enabled"]
    except Exception:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    return False


def _job_explicitly_disabled(row: Any) -> bool:
    try:
        value = row["enabled"]
    except Exception:
        return False
    return isinstance(value, int) and not isinstance(value, bool) and value == 0


def _configured_morning_brief_env() -> tuple[str | None, str]:
    if MORNING_BRIEF_ENV in os.environ:
        raw = os.environ.get(MORNING_BRIEF_ENV, "")
        text = str(raw or "").strip()
        if not text:
            return None, MORNING_BRIEF_ENV
        return text, MORNING_BRIEF_ENV
    legacy = str(os.getenv(LEGACY_MORNING_BRIEF_ENV, "") or "").strip()
    if legacy:
        return legacy, LEGACY_MORNING_BRIEF_ENV
    return MORNING_BRIEF_DEFAULT_HHMM, "default"


def _build_live_daily_brief(config: JarvisConfig) -> str:
    from jarvis_v2.tools.brief_tools import make_brief_tools

    tool = {tool.name: tool for tool in make_brief_tools(config)}["daily_briefing"]
    return tool.handler({}).output


def _send_owner_telegram(text: str) -> dict:
    from jarvis_v2.automations import telegram_control

    owner = telegram_control._owner_chat_id()
    if not owner:
        return {"ok": False, "error": "missing_or_invalid_owner"}
    return telegram_control.send_message(owner, text)


def _safe_telegram_delivery_error(value: object) -> str:
    text = str(value or "telegram_send_failed").strip()
    if not text:
        return "telegram_send_failed"
    if len(text) > 64 or any(marker in text for marker in ("/", "\\", "\n", "\r", "\t")):
        return "telegram_send_failed"
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", text):
        return "telegram_send_failed"
    return text


def _telegram_delivery_failure_message(value: object, *, prefix: str) -> str:
    code = _safe_telegram_delivery_error(value)
    if code == "missing_or_invalid_owner":
        body = (
            "owner Telegram chat is not configured. Set JARVIS_OWNER_TELEGRAM, "
            "then run setup check or channel status."
        )
    else:
        body = (
            "Telegram delivery failed. Run setup check or channel status, verify the bot token "
            "and owner chat, then retry."
        )
    diagnostic = "" if code == "telegram_send_failed" else f" Diagnostic: {code}."
    return f"{prefix}: {body}{diagnostic}"


class _JobLeaseHeartbeat:
    def __init__(
        self,
        store: MemoryStore,
        job_id: int,
        lease_token: str,
        *,
        lease_seconds: float,
        heartbeat_seconds: float,
        initial_lease_expires_at: object,
    ) -> None:
        self.store = store
        self.job_id = job_id
        self.lease_token = lease_token
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self._stop = Event()
        self._ownership_lost = Event()
        self._deadline_lock = Lock()
        self._last_confirmed_deadline = time.monotonic() + self._remaining_initial_lease(
            initial_lease_expires_at
        )
        self._thread: Thread | None = None

    def _remaining_initial_lease(self, value: object) -> float:
        try:
            expires_at = datetime.fromisoformat(str(value))
            now = datetime.now(tz=expires_at.tzinfo) if expires_at.tzinfo else datetime.now()
            return max(0.0, min((expires_at - now).total_seconds(), self.lease_seconds))
        except (TypeError, ValueError):
            return 0.0

    def _confirm_renewal(self, renewal_started_at: float) -> None:
        with self._deadline_lock:
            self._last_confirmed_deadline = renewal_started_at + self.lease_seconds

    def _remaining_confirmed_authority(self) -> float:
        with self._deadline_lock:
            remaining = self._last_confirmed_deadline - time.monotonic()
        if remaining <= 0:
            self._ownership_lost.set()
            return 0.0
        return remaining

    def _renew(self) -> bool | None:
        renewal_started_at = time.monotonic()
        try:
            renewed = self.store.renew_job_claim(
                self.job_id,
                self.lease_token,
                self.lease_seconds,
            )
        except Exception:
            # A transient storage error is inconclusive only while a previously
            # confirmed lease is still live. Past that deadline, another
            # scheduler may claim the job, so fail closed.
            self._remaining_confirmed_authority()
            return None
        if not renewed:
            self._ownership_lost.set()
        else:
            # The store computes its deadline after this attempt begins. Using
            # the attempt start makes this local authority window conservative:
            # it can expire before, but never after, the confirmed DB lease.
            self._confirm_renewal(renewal_started_at)
        return renewed

    def start(self) -> bool:
        if self._renew() is False:
            return False
        self._thread = Thread(
            target=self._run,
            name=f"jarvis-job-lease-{self.job_id}",
            daemon=True,
        )
        self._thread.start()
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            remaining = self._remaining_confirmed_authority()
            if remaining <= 0:
                return
            if self._stop.wait(min(self.heartbeat_seconds, remaining)):
                return
            if self._renew() is False:
                return

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(max(self.heartbeat_seconds * 2, 0.1), 1.0))

    @property
    def ownership_lost(self) -> bool:
        if not self._ownership_lost.is_set():
            self._remaining_confirmed_authority()
        return self._ownership_lost.is_set()

    def require_authority(self) -> None:
        if self.ownership_lost:
            raise RuntimeError("scheduler_job_lease_authority_lost")
        if self._renew() is not True:
            raise RuntimeError("scheduler_job_lease_authority_lost")


class Scheduler:
    def __init__(
        self,
        store: MemoryStore,
        vault: ObsidianVault,
        config: JarvisConfig | None = None,
        *,
        job_lease_seconds: float | None = None,
        job_heartbeat_seconds: float | None = None,
    ):
        self.store = store
        self.vault = vault
        self.config = config
        self.job_lease_seconds = _positive_seconds(
            JOB_LEASE_SECONDS if job_lease_seconds is None else job_lease_seconds,
            JOB_LEASE_SECONDS,
            0.05,
        )
        requested_heartbeat = _positive_seconds(
            JOB_HEARTBEAT_SECONDS if job_heartbeat_seconds is None else job_heartbeat_seconds,
            JOB_HEARTBEAT_SECONDS,
            0.01,
        )
        self.job_heartbeat_seconds = min(requested_heartbeat, self.job_lease_seconds / 3)

    def schedule_job(self, name: str, interval_minutes: int, job_type: str) -> int:
        next_run = iso(datetime.now() + timedelta(minutes=interval_minutes))
        return self.store.upsert_job(name, interval_minutes, job_type, next_run)

    def schedule_job_if_missing(
        self,
        name: str,
        interval_minutes: int,
        job_type: str,
    ) -> tuple[Any, bool]:
        next_run = iso(datetime.now() + timedelta(minutes=interval_minutes))
        return self.store.create_job_if_missing(
            name,
            interval_minutes,
            job_type,
            next_run,
        )

    def _morning_brief_row(self):
        for row in self.store.list_jobs():
            if str(row["name"]).casefold() == MORNING_BRIEF_JOB_NAME.casefold():
                return row
        return None

    def _merge_job_metadata(self, job_id: int, updates: dict[str, Any]) -> bool:
        return self.store.merge_job_metadata(job_id, updates)

    def _record_job_run_history(self, row: Any, now_dt: datetime, output: object) -> bool:
        try:
            job_id = int(row["id"])
        except (TypeError, ValueError, KeyError, IndexError):
            return False

        status = _classify_job_run_status(output)
        event = self._run_history_event(row, now_dt, status)
        return self.store.append_job_run_history_event(
            job_id,
            expected_last_run_at=_run_marker(now_dt),
            event=event,
        )

    def _run_history_occurrence_key(
        self,
        row: Any,
        plan: _DeliveryPlan | None = None,
    ) -> str:
        if plan is not None:
            return plan.occurrence_key
        active_occurrence_key = str(row["active_occurrence_key"] or "")
        if re.fullmatch(r"scheduler-occurrence:v1:[0-9a-f]{64}", active_occurrence_key):
            return active_occurrence_key
        material = f"{int(row['id'])}:{_job_occurrence_schedule_revision(row)}:{row['next_run_at']}"
        return f"scheduled:{int(row['id'])}:{_delivery_key_part(material)}"

    def _run_history_event(
        self,
        row: Any,
        now_dt: datetime,
        status: str,
        plan: _DeliveryPlan | None = None,
    ) -> dict[str, Any]:
        return {
            "occurrence_key": self._run_history_occurrence_key(row, plan),
            "date": now_dt.date().isoformat(),
            "ran_at": _run_marker(now_dt),
            "status": status,
            "job_type": _job_occurrence_type(row),
            "schedule_identity_revision": _job_occurrence_schedule_identity_revision(row),
        }

    def _scheduled_note_publication(self, row: Any | None, job_type: str) -> tuple[str, str]:
        if job_type not in SCHEDULED_NOTE_JOB_TYPES or row is None:
            return "", ""
        try:
            job_id = int(row["id"])
            job_name = str(row["name"])
            active_occurrence = str(row["active_occurrence_key"] or "")
            active_started_at = str(row["active_occurrence_started_at"] or "")
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            raise RuntimeError("scheduled_note_occurrence_identity_unavailable") from exc
        if (
            job_id < 1
            or not job_name
            or _job_occurrence_type(row) != job_type
            or re.fullmatch(r"scheduler-occurrence:v1:[0-9a-f]{64}", active_occurrence) is None
        ):
            raise RuntimeError("scheduled_note_occurrence_identity_invalid")
        try:
            target_date = datetime.fromisoformat(active_started_at).date().isoformat()
        except (TypeError, ValueError) as exc:
            raise RuntimeError("scheduled_note_occurrence_date_invalid") from exc
        store_identity = self.store.store_instance_identity()
        material = json.dumps(
            {
                "version": 1,
                "store_identity": store_identity,
                "job_id": job_id,
                "job_name": job_name,
                "job_type": job_type,
                "occurrence_key": active_occurrence,
                "target_date": target_date,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest(), target_date

    def _scheduled_note_plan(
        self,
        row: Any,
        output: str,
        *,
        delivery_plan: _DeliveryPlan | None,
        allow_disabled: bool,
        source_manifest_kind: str = "",
        source_manifest_digest: str = "",
    ) -> _ScheduledNotePlan | None:
        job_type = _job_occurrence_type(row)
        if job_type not in SCHEDULED_NOTE_JOB_TYPES:
            return None
        publication_key, target_date = self._scheduled_note_publication(row, job_type)
        payload = self.vault.scheduled_note_payload(output)
        occurrence_key = str(row["active_occurrence_key"] or "")
        delivery_occurrence_key = ""
        if delivery_plan is not None:
            delivery_occurrence_key = delivery_plan.occurrence_key
        return _ScheduledNotePlan(
            publication_key=publication_key,
            occurrence_key=occurrence_key,
            delivery_occurrence_key=delivery_occurrence_key,
            effect_index=0,
            job_id=int(row["id"]),
            source_job_id=int(row["id"]),
            job_name=str(row["name"]),
            job_type=job_type,
            target_kind="reflection" if job_type == "weekly_review" else "daily",
            target_date=target_date,
            title=SCHEDULED_NOTE_TITLES[job_type],
            payload=payload,
            payload_sha256=_payload_sha256(payload),
            source_manifest_kind=source_manifest_kind,
            source_manifest_digest=source_manifest_digest,
            allow_disabled=allow_disabled,
        )

    def _converge_run_history_output(
        self,
        row: Any,
        plan: _DeliveryPlan,
        output: object,
    ) -> None:
        try:
            self.store.converge_job_run_history(
                int(row["id"]),
                plan.occurrence_key,
                _durable_job_run_status(output),
            )
        except Exception:
            return

    def schedule_morning_brief(self, hhmm: str, now: datetime | None = None, *, source: str = "manual") -> tuple[int, str] | None:
        parsed = parse_hhmm(hhmm)
        if not parsed:
            return None
        hour, minute = parsed
        normalized = format_hhmm(hour, minute)
        next_run = iso(next_daily_run(hour, minute, now=now))
        row = self.store.upsert_job_with_metadata(
            MORNING_BRIEF_JOB_NAME,
            MORNING_BRIEF_INTERVAL_MINUTES,
            MORNING_BRIEF_JOB_TYPE,
            next_run,
            {
                "configured_time_hhmm": normalized,
                "schedule_source": source,
                "once_per_day_guard": True,
                "disabled_by_env": False,
                "resumed_by_operator": False,
            },
        )
        return int(row["id"]), normalized

    def ensure_env_morning_brief(self, now: datetime | None = None) -> None:
        raw, source = _configured_morning_brief_env()
        if raw is None:
            row = self._morning_brief_row()
            if row is None:
                return
            metadata = _job_metadata(row)
            if _job_explicitly_disabled(row) and metadata.get("disabled_by_env") is not True:
                return
            self.store.set_job_enabled(
                MORNING_BRIEF_JOB_NAME,
                False,
                metadata_updates={
                    "disabled_by_env": True,
                    "schedule_source": source,
                    "resumed_by_operator": False,
                },
                expected_schedule_revision=int(row["schedule_revision"]),
            )
            return
        parsed = parse_hhmm(raw)
        if not parsed:
            return
        hour, minute = parsed
        normalized = format_hhmm(hour, minute)
        row = self._morning_brief_row()
        disabled_by_env = (
            _job_metadata(row).get("disabled_by_env") is True if row is not None else False
        )
        resumed_by_operator = (
            _job_metadata(row).get("resumed_by_operator") is True if row is not None else False
        )
        if row is not None and _job_explicitly_disabled(row) and not disabled_by_env:
            return
        next_run_is_future = False
        if row is not None:
            try:
                next_run_is_future = datetime.fromisoformat(str(row["next_run_at"])) > (now or datetime.now())
            except (TypeError, ValueError):
                next_run_is_future = False
        if (
            row is not None
            and job_enabled(row)
            and _row_next_run_hhmm(row) == normalized
            and (next_run_is_future or not resumed_by_operator)
        ):
            merged = self.store.merge_job_metadata(
                int(row["id"]),
                {
                    "configured_time_hhmm": normalized,
                    "schedule_source": source,
                    "once_per_day_guard": True,
                    "disabled_by_env": False,
                    "resumed_by_operator": False,
                },
                expected_next_run_at=str(row["next_run_at"]),
            )
            if merged:
                return
            row = self._morning_brief_row()
            if row is None:
                return
            disabled_by_env = _job_metadata(row).get("disabled_by_env") is True
            resumed_by_operator = _job_metadata(row).get("resumed_by_operator") is True
            if _job_explicitly_disabled(row) and not disabled_by_env:
                return
        if row is None:
            self.schedule_morning_brief(raw, now=now, source=source)
            return
        self.store.reconcile_job_schedule_if_revision(
            int(row["id"]),
            int(row["schedule_revision"]),
            MORNING_BRIEF_INTERVAL_MINUTES,
            MORNING_BRIEF_JOB_TYPE,
            iso(next_daily_run(hour, minute, now=now)),
            {
                "configured_time_hhmm": normalized,
                "schedule_source": source,
                "once_per_day_guard": True,
                "disabled_by_env": False,
                "resumed_by_operator": False,
            },
            allow_disabled=disabled_by_env,
        )

    def list_jobs(self) -> str:
        rows = self.store.list_jobs()
        if not rows:
            return "No scheduled jobs."
        lines = []
        for row in rows:
            status = "on" if job_enabled(row) else "off"
            lines.append(f"- #{row['id']} {row['name']} [{row['job_type']}, {status}] next: {row['next_run_at']}")
        return "\n".join(lines)

    def pause_job(self, name: str) -> str:
        return (
            f"Paused {name}."
            if self.store.set_job_enabled(
                name,
                False,
                remove_metadata_keys=("disabled_by_env", "resumed_by_operator"),
            )
            else f"No job found named {name}."
        )

    def resume_job(self, name: str) -> str:
        return (
            f"Resumed {name}."
            if self.store.set_job_enabled(
                name,
                True,
                metadata_updates={"resumed_by_operator": True},
                remove_metadata_keys=("disabled_by_env",),
                metadata_on_state_change_only=True,
            )
            else f"No job found named {name}."
        )

    def delete_job(self, name: str) -> str:
        return f"Deleted {name}." if self.store.delete_job(name) else f"No job found named {name}."

    def _claim_job(self, row: Any, *, due_before: str | None = None) -> tuple[Any, str] | None:
        lease_token = uuid.uuid4().hex
        claimed_row = self.store.claim_job(
            int(row["id"]),
            lease_token,
            self.job_lease_seconds,
            due_before=due_before,
        )
        if claimed_row is None:
            return None
        return claimed_row, lease_token

    def _claim_heartbeat(self, row: Any, lease_token: str) -> _JobLeaseHeartbeat:
        return _JobLeaseHeartbeat(
            self.store,
            int(row["id"]),
            lease_token,
            lease_seconds=self.job_lease_seconds,
            heartbeat_seconds=self.job_heartbeat_seconds,
            initial_lease_expires_at=row["lease_expires_at"],
        )

    def _release_claim(self, row: Any, lease_token: str) -> bool:
        try:
            return self.store.release_job_claim(
                int(row["id"]),
                lease_token,
            )
        except Exception:
            return False

    def _run_owned_job_type(
        self,
        heartbeat: _JobLeaseHeartbeat,
        job_type: str,
        *,
        row: Any,
        now: datetime,
        lease_token: str,
    ) -> str | ToolResult | GoalNudgeBuild:
        heartbeat.require_authority()
        kwargs: dict[str, Any] = {
            "row": row,
            "now": now,
            "lease_authority": heartbeat.require_authority,
        }
        if job_type in {"conversation_compaction", "state_snapshot"}:
            kwargs["lease_token"] = lease_token
        output = self._run_job_type(job_type, **kwargs)
        heartbeat.require_authority()
        return output

    def _delivery_occurrence_key(self, row: Any, now_dt: datetime) -> str:
        job_id = int(row["id"])
        job_type = _job_occurrence_type(row)
        if job_type == MORNING_BRIEF_JOB_TYPE:
            occurrence = now_dt.date().isoformat()
            return f"morning:{_delivery_key_part(occurrence)}"
        active_occurrence_key = str(row["active_occurrence_key"] or "")
        if re.fullmatch(r"scheduler-occurrence:v1:[0-9a-f]{64}", active_occurrence_key):
            return f"scheduled:{active_occurrence_key}"
        occurrence = f"{_job_occurrence_schedule_revision(row)}:{row['next_run_at']}"
        return f"scheduled:{job_id}:{_delivery_key_part(occurrence)}"

    def _run_receipted_state_snapshot(self, row: Any, lease_token: str) -> str:
        try:
            job_id = int(row["id"])
            occurrence_key = str(row["active_occurrence_key"] or "")
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            raise RuntimeError("state_snapshot_occurrence_identity_unavailable") from exc
        if not re.fullmatch(r"scheduler-occurrence:v1:[0-9a-f]{64}", occurrence_key):
            raise RuntimeError("state_snapshot_occurrence_identity_invalid")

        def require_publication_authority() -> None:
            if not self.store.job_claim_is_live(job_id, lease_token):
                raise RuntimeError("state_snapshot_lease_lost")

        if not lease_token or not self.store.renew_job_claim(
            job_id,
            lease_token,
            self.job_lease_seconds,
        ):
            raise RuntimeError("state_snapshot_lease_lost")

        cutoff = datetime.now(timezone.utc) - timedelta(seconds=STATE_SNAPSHOT_RECEIPT_STALE_SECONDS)
        running_before = cutoff.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        self.store.recover_stale_auto_mutation_receipts(running_before, limit=100)
        preparation = self.store.prepare_auto_mutation_receipts(
            occurrence_key,
            [
                (
                    "export_state_snapshot",
                    {},
                    {
                        "projection": "state_snapshot_bundle",
                        "occurrence_key": occurrence_key,
                    },
                )
            ],
        )
        if preparation.status == "REQUEST_TOKEN_COLLISION":
            return (
                "State Snapshot occurrence held due to a receipt identity collision; "
                "no automatic rewrite was attempted."
            )
        if preparation.status not in {"PREPARED", "RESUMED"} or len(preparation.receipts) != 1:
            raise RuntimeError("state_snapshot_receipt_prepare_failed")
        receipt = preparation.receipts[0]
        if receipt.receipt_id is None:
            raise RuntimeError("state_snapshot_receipt_missing")
        if receipt.status == "COMPLETED_COALESCED":
            return (
                f"State Snapshot occurrence already completed (receipt #{receipt.receipt_id}); "
                "no projection was rewritten."
            )
        if receipt.status == "RUNNING_COALESCED":
            raise RuntimeError("state_snapshot_receipt_running")
        if receipt.status == "UNCERTAIN_BLOCKED":
            return (
                f"State Snapshot occurrence held for review (receipt #{receipt.receipt_id}, "
                f"status {receipt.status.lower()}); no automatic rewrite was attempted."
            )
        if receipt.status in {"RECONCILED_APPLIED", "RECONCILED_NOT_APPLIED"}:
            disposition = "already applied" if receipt.status == "RECONCILED_APPLIED" else "not applied"
            return (
                f"State Snapshot occurrence reconciled as {disposition} "
                f"(receipt #{receipt.receipt_id}); no automatic rewrite was attempted."
            )
        if receipt.status not in {"PREPARED", "PREPARED_RESUME"}:
            raise RuntimeError("state_snapshot_receipt_unavailable")

        claim = self.store.claim_auto_mutation_receipt(receipt.receipt_id, occurrence_key)
        if claim.status != "CLAIMED" or claim.receipt_id is None or not claim.run_token:
            if claim.status == "COMPLETED_COALESCED":
                return (
                    f"State Snapshot occurrence already completed (receipt #{receipt.receipt_id}); "
                    "no projection was rewritten."
                )
            if claim.status == "RUNNING_COALESCED":
                raise RuntimeError("state_snapshot_receipt_running")
            if claim.status == "UNCERTAIN_BLOCKED":
                return (
                    f"State Snapshot occurrence held for review (receipt #{receipt.receipt_id}, "
                    f"status {claim.status.lower()}); no automatic rewrite was attempted."
                )
            if claim.status in {"RECONCILED_APPLIED", "RECONCILED_NOT_APPLIED"}:
                disposition = "already applied" if claim.status == "RECONCILED_APPLIED" else "not applied"
                return (
                    f"State Snapshot occurrence reconciled as {disposition} "
                    f"(receipt #{receipt.receipt_id}); no automatic rewrite was attempted."
                )
            raise RuntimeError("state_snapshot_receipt_claim_failed")

        handler_started = False
        try:
            if not self.store.renew_job_claim(job_id, lease_token, self.job_lease_seconds):
                raise RuntimeError("state_snapshot_lease_lost")
            _, export_state_snapshot = make_state_tools(
                self.store,
                self.vault,
                effect_authority=require_publication_authority,
            )
            handler_started = True
            state_result = export_state_snapshot({})
            if (
                state_result.tool_name != "export_state_snapshot"
                or type(state_result.ok) is not bool
                or state_result.ok is not True
            ):
                raise RuntimeError("state_snapshot_current_context_failed")

            if not self.store.renew_job_claim(job_id, lease_token, self.job_lease_seconds):
                raise RuntimeError("state_snapshot_lease_lost")
            next_step_tools = make_next_step_tools(
                self.store,
                self.vault,
                effect_authority=require_publication_authority,
            )
            mission_result = next_step_tools[7]({})
            if (
                mission_result.tool_name != "export_mission_control"
                or type(mission_result.ok) is not bool
                or mission_result.ok is not True
            ):
                raise RuntimeError("state_snapshot_mission_control_failed")

            safe_output = f"State Snapshot occurrence completed (receipt #{claim.receipt_id})."
            audit_metadata = {
                "handler_invoked": True,
                "executed_handler": True,
                "writes_files": True,
                "writes_notes": True,
                "reads_personal_data": True,
                "reads_private_data": True,
                "scheduled_occurrence": True,
                "authorizes_execution": False,
                "authorizes_completion_claim": False,
                "approval_granted": False,
            }
            path_display = _safe_projection_path_display(state_result.metadata.get("path_display"))
            mission_path_display = _safe_projection_path_display(
                mission_result.metadata.get("path_display")
            )
            content_sha256 = _safe_projection_digest(state_result.metadata.get("content_sha256"))
            source_revision = _safe_projection_digest(state_result.metadata.get("source_revision"))
            if path_display:
                audit_metadata["path_display"] = path_display
            if mission_path_display:
                audit_metadata["mission_control_path_display"] = mission_path_display
            if content_sha256:
                audit_metadata["content_sha256"] = content_sha256
            if source_revision:
                audit_metadata["source_revision"] = source_revision
            if not self.store.renew_job_claim(job_id, lease_token, self.job_lease_seconds):
                raise RuntimeError("state_snapshot_lease_lost")
            run_id = self.store.complete_auto_mutation_receipt(
                receipt_id=claim.receipt_id,
                run_token=claim.run_token,
                session_id=f"scheduler:{job_id}",
                risk="LOCAL_SAFE",
                ok=True,
                output=safe_output,
                metadata=audit_metadata,
            )
            return (
                f"State Snapshot occurrence completed (receipt #{claim.receipt_id}, "
                f"audit run #{run_id})."
            )
        except Exception as exc:
            try:
                uncertainty_recorded = self.store.mark_auto_mutation_uncertain(
                    claim.receipt_id,
                    claim.run_token,
                )
            except Exception:
                uncertainty_recorded = False
            if not uncertainty_recorded:
                raise RuntimeError("state_snapshot_uncertainty_record_failed") from exc
            if handler_started:
                return (
                    f"State Snapshot occurrence held for review (receipt #{claim.receipt_id}, "
                    "status uncertain); no automatic rewrite will be attempted."
                )
            raise

    def _existing_delivery_plan(self, occurrence_key: str) -> _DeliveryPlan | None:
        rows = self.store.list_scheduled_deliveries_for_occurrence(occurrence_key)
        if not rows:
            return None
        return _DeliveryPlan(occurrence_key=occurrence_key, chunks=(), existing=True)

    def _delivery_plan(self, row: Any, payload: str, now_dt: datetime) -> _DeliveryPlan:
        occurrence_key = self._delivery_occurrence_key(row, now_dt)
        existing = self._existing_delivery_plan(occurrence_key)
        if existing is not None:
            return existing
        from jarvis_v2.automations import telegram_control

        safe_payload = telegram_control._safe_outbound_text(payload).strip()
        if not safe_payload:
            raise ValueError("empty scheduled Telegram payload")
        if len(safe_payload) > DELIVERY_MAX_PAYLOAD_CHARS:
            raise ValueError("scheduled Telegram payload exceeds the durable outbox limit")
        bodies = telegram_control._text_chunks(safe_payload)
        if not bodies or len(bodies) > DELIVERY_MAX_CHUNKS:
            raise ValueError("scheduled Telegram payload exceeds the chunk limit")
        chunks: list[dict[str, Any]] = []
        for index, body in enumerate(bodies):
            chunks.append(
                {
                    "delivery_key": f"{occurrence_key}:{index:02d}",
                    "payload": body,
                    "payload_sha256": _payload_sha256(body),
                    "chunk_index": index,
                    "chunk_count": len(bodies),
                }
            )
        return _DeliveryPlan(occurrence_key=occurrence_key, chunks=tuple(chunks))

    def _delivery_plan_for_output(self, row: Any, output: str, now_dt: datetime) -> _DeliveryPlan | None:
        job_type = _job_occurrence_type(row)
        title = DELIVERED_JOB_TITLES.get(job_type)
        if not title:
            return None
        text = str(output or "").strip()
        if not text:
            return None
        markers = QUIET_OUTPUT_MARKERS.get(job_type, ())
        if any(marker in text for marker in markers):
            return None
        return self._delivery_plan(row, f"{title}\n\n{text}", now_dt)

    def _finalize_claim_with_delivery(
        self,
        row: Any,
        lease_token: str,
        completed_at: datetime,
        next_run: str,
        plan: _DeliveryPlan | None,
        note_plan: _ScheduledNotePlan | None = None,
        output: object = "",
    ) -> bool:
        status = "pending" if plan is not None or note_plan is not None else _durable_job_run_status(output)
        event = self._run_history_event(row, completed_at, status, plan)
        active_occurrence_key = str(row["active_occurrence_key"] or "")
        source_job_type = _job_occurrence_type(row)
        note_rows = [note_plan.as_store_plan()] if note_plan is not None else None
        if plan is None or plan.existing:
            return self.store.mark_claimed_job_run(
                int(row["id"]),
                lease_token,
                _run_marker(completed_at),
                next_run,
                _job_occurrence_schedule_revision(row),
                run_history_event=event,
                expected_occurrence_key=active_occurrence_key,
                source_job_type=source_job_type,
                scheduled_note_publications=note_rows,
            )
        return self.store.complete_claimed_job_with_scheduled_deliveries(
            int(row["id"]),
            lease_token,
            _run_marker(completed_at),
            next_run,
            plan.occurrence_key,
            list(plan.chunks),
            str(row["name"]),
            source_job_type,
            _job_occurrence_schedule_revision(row),
            run_history_event=event,
            expected_occurrence_key=active_occurrence_key,
            scheduled_note_publications=note_rows,
        )

    def _delivery_occurrence_state(self, occurrence_key: str) -> str:
        rows = self.store.list_scheduled_deliveries_for_occurrence(occurrence_key)
        if not rows:
            return "missing"
        states = [str(row["state"]) for row in rows]
        if all(state == "accepted" for state in states):
            return "accepted"
        if "uncertain" in states:
            return "uncertain"
        if "rejected" in states:
            return "rejected"
        if "failed" in states:
            return "failed"
        if "sending" in states:
            return "sending"
        if "claimed" in states:
            return "claimed"
        return "prepared"

    def _dispatch_scheduled_note_row(self, row: Any) -> _ScheduledNoteDispatch:
        publication_key = str(row["publication_key"])
        attempt_token = uuid.uuid4().hex
        claimed = self.store.claim_scheduled_note_publication(publication_key, attempt_token)
        if claimed is None:
            current = self.store.get_scheduled_note_publication(publication_key)
            state = str(current["state"]) if current is not None else "missing"
            return _ScheduledNoteDispatch(state, f"scheduled note coalesced ({state})")
        if claimed["effect_started_at"] is not None:
            try:
                evidence = self.vault.scheduled_note_publication_evidence(
                    str(claimed["target_kind"]),
                    str(claimed["target_date"]),
                    publication_key,
                    str(claimed["title"]),
                )
                if evidence is not None:
                    evidence_path, persisted = evidence
                    persisted_sha256 = _payload_sha256(persisted)
                    if secrets.compare_digest(
                        persisted_sha256,
                        str(claimed["payload_sha256"] or ""),
                    ):
                        path_display = _safe_projection_path_display(
                            evidence_path.relative_to(self.vault.root_path).as_posix()
                        )
                        if path_display and self.store.finish_scheduled_note_publication(
                            publication_key,
                            attempt_token,
                            "published",
                            path_display=path_display,
                            content_sha256=persisted_sha256,
                        ):
                            return _ScheduledNoteDispatch(
                                "published",
                                "scheduled note publication reconciled from durable evidence",
                            )
            except Exception:
                pass
        payload = str(claimed["payload"] or "")
        if not payload or not secrets.compare_digest(
            _payload_sha256(payload),
            str(claimed["payload_sha256"] or ""),
        ):
            if claimed["effect_started_at"] is None:
                self.store.finish_scheduled_note_publication(
                    publication_key,
                    attempt_token,
                    "failed",
                    error_code="payload_integrity_failed",
                )
                return _ScheduledNoteDispatch("failed", "scheduled note payload integrity failed")
            self.store.finish_scheduled_note_publication(
                publication_key,
                attempt_token,
                "retryable",
                error_code="publication_evidence_pending",
            )
            return _ScheduledNoteDispatch(
                "prepared",
                "scheduled note publication awaits durable evidence reconciliation",
            )

        def require_claim() -> None:
            if not self.store.begin_scheduled_note_publication_effect(
                publication_key,
                attempt_token,
            ):
                raise RuntimeError("scheduled_note_claim_lost")

        try:
            target_kind = str(claimed["target_kind"])
            source_bound = str(claimed["job_type"]) == "goal_nudge"
            args = (
                str(claimed["target_date"]),
                publication_key,
                str(claimed["title"]),
                payload,
            )
            source_fence = (
                self.store.goal_source_effect_fence() if source_bound else nullcontext()
            )
            with source_fence:
                if target_kind == "daily":
                    path, _, persisted = self.vault.append_scheduled_daily_once_for_date(
                        *args,
                        effect_authority=require_claim,
                    )
                elif target_kind == "reflection":
                    path, _, persisted = self.vault.append_scheduled_reflection_once_for_date(
                        *args,
                        effect_authority=require_claim,
                    )
                else:
                    raise RuntimeError("scheduled_note_target_kind_invalid")
            if not secrets.compare_digest(_payload_sha256(persisted), str(claimed["payload_sha256"])):
                raise RuntimeError("scheduled_note_persisted_payload_mismatch")
            path_display = _safe_projection_path_display(
                path.relative_to(self.vault.root_path).as_posix()
            )
            if not path_display:
                raise RuntimeError("scheduled_note_path_evidence_invalid")
            finalized = self.store.finish_scheduled_note_publication(
                publication_key,
                attempt_token,
                "published",
                path_display=path_display,
                content_sha256=_payload_sha256(persisted),
            )
        except Exception as exc:
            try:
                self.store.finish_scheduled_note_publication(
                    publication_key,
                    attempt_token,
                    "retryable",
                    error_code=_safe_job_failure_code(exc),
                )
            except Exception:
                pass
            current = self.store.get_scheduled_note_publication(publication_key)
            state = str(current["state"]) if current is not None else "missing"
            return _ScheduledNoteDispatch(
                state,
                "scheduled note publication failed safely" if state == "failed" else "scheduled note publication retry queued",
            )
        if not finalized:
            return _ScheduledNoteDispatch("uncertain", "scheduled note receipt finalization was uncertain")
        return _ScheduledNoteDispatch("published", "scheduled note published")

    def _record_scheduled_note_status(self, row: Any, dispatch: _ScheduledNoteDispatch) -> None:
        try:
            job_id = int(row["job_id"])
        except (TypeError, ValueError):
            return
        delivery_occurrence = str(row["delivery_occurrence_key"] or "")
        history_occurrence = delivery_occurrence or str(row["occurrence_key"])
        updates = {
            "last_note_publication_status": dispatch.state,
            "last_note_publication_status_at": iso(datetime.now()),
            "last_note_publication_occurrence_key": str(row["occurrence_key"]),
        }
        try:
            if dispatch.state == "published" and not delivery_occurrence:
                if not self.store.converge_job_run_history(
                    job_id,
                    history_occurrence,
                    "ok",
                    updates=updates,
                ):
                    self.store.merge_job_metadata(job_id, updates)
            elif dispatch.state == "failed":
                if not self.store.converge_job_run_history(
                    job_id,
                    history_occurrence,
                    "failed",
                    updates=updates,
                ):
                    self.store.merge_job_metadata(job_id, updates)
            elif dispatch.state == "published":
                self.store.merge_job_metadata(job_id, updates)
        except Exception:
            return

    def _dispatch_scheduled_note_occurrence(self, occurrence_key: str) -> _ScheduledNoteDispatch:
        rows = self.store.list_scheduled_note_publications_for_occurrence(occurrence_key)
        if not rows:
            return _ScheduledNoteDispatch("missing", "scheduled note was not durably prepared")
        for row in rows:
            state = str(row["state"])
            if state == "published":
                continue
            if state == "failed":
                dispatch = _ScheduledNoteDispatch("failed", "scheduled note reached terminal failure")
                self._record_scheduled_note_status(row, dispatch)
                return dispatch
            dispatch = self._dispatch_scheduled_note_row(row)
            current = self.store.get_scheduled_note_publication(str(row["publication_key"])) or row
            self._record_scheduled_note_status(current, dispatch)
            if dispatch.state != "published":
                return dispatch
        return _ScheduledNoteDispatch("published", "scheduled note published")

    def _dispatch_delivery_row(self, row: Any) -> _DeliveryDispatch:
        delivery_key = str(row["delivery_key"])
        attempt_token = uuid.uuid4().hex
        claimed = self.store.claim_scheduled_delivery(delivery_key, attempt_token)
        if claimed is None:
            current = self.store.get_scheduled_delivery(delivery_key)
            state = str(current["state"]) if current is not None else "missing"
            if current is not None and str(current["error_code"] or "") == "payload_integrity_failed":
                return _DeliveryDispatch(
                    "failed",
                    "scheduled Telegram payload integrity check failed; nothing was sent",
                )
            return _DeliveryDispatch(state, f"scheduled Telegram chunk coalesced ({state})")
        source_bound = str(claimed["job_type"]) == "goal_nudge"
        source_fence = self.store.goal_source_effect_fence() if source_bound else nullcontext()
        network_started = False
        try:
            with source_fence:
                if not self.store.begin_scheduled_delivery_send(delivery_key, attempt_token):
                    released = self.store.release_scheduled_delivery_claim(
                        delivery_key, attempt_token
                    )
                    if released:
                        return _DeliveryDispatch(
                            "prepared",
                            "scheduled Telegram chunk was held before network I/O",
                        )
                    current = self.store.get_scheduled_delivery(delivery_key)
                    state = str(current["state"]) if current is not None else "missing"
                    return _DeliveryDispatch(
                        state,
                        f"scheduled Telegram chunk was not dispatched ({state})",
                    )
                network_started = True
                result = _send_owner_telegram(str(claimed["payload"]))
        except Exception:
            if not network_started:
                released = self.store.release_scheduled_delivery_claim(
                    delivery_key, attempt_token
                )
                if released:
                    return _DeliveryDispatch(
                        "prepared",
                        "scheduled Telegram chunk was held before network I/O",
                    )
                current = self.store.get_scheduled_delivery(delivery_key)
                state = str(current["state"]) if current is not None else "missing"
                return _DeliveryDispatch(
                    state,
                    f"scheduled Telegram chunk was not dispatched ({state})",
                )
            try:
                self.store.finish_scheduled_delivery(
                    delivery_key,
                    attempt_token,
                    "uncertain",
                    error_code="send_exception",
                )
            except Exception:
                pass
            return _DeliveryDispatch(
                "uncertain",
                "Telegram outcome is unknown. It may have been accepted; Jarvis will not automatically resend.",
            )
        if isinstance(result, dict) and result.get("ok") is True:
            message_id = _safe_telegram_message_id(result)
            if message_id is None:
                try:
                    self.store.finish_scheduled_delivery(
                        delivery_key,
                        attempt_token,
                        "uncertain",
                        error_code="missing_message_id",
                    )
                except Exception:
                    pass
                return _DeliveryDispatch(
                    "uncertain",
                    "Telegram returned a malformed acceptance receipt; Jarvis will not automatically resend.",
                )
            try:
                finalized = self.store.finish_scheduled_delivery(
                    delivery_key,
                    attempt_token,
                    "accepted",
                    telegram_message_id=message_id,
                )
            except Exception:
                finalized = False
            if finalized:
                return _DeliveryDispatch("accepted", "accepted by Telegram API")
            return _DeliveryDispatch(
                "uncertain",
                "Telegram accepted the request but its local receipt is uncertain; Jarvis will not automatically resend.",
            )
        failure_state, error_code, retry_after = _telegram_failure_policy(result)
        if failure_state == "uncertain":
            try:
                self.store.finish_scheduled_delivery(
                    delivery_key,
                    attempt_token,
                    "uncertain",
                    error_code=error_code,
                )
            except Exception:
                pass
            return _DeliveryDispatch(
                "uncertain",
                "Telegram outcome is unknown. It may have been accepted; Jarvis will not automatically resend.",
            )
        try:
            finalized = self.store.finish_scheduled_delivery(
                delivery_key,
                attempt_token,
                failure_state,
                error_code=error_code,
                retry_after_seconds=retry_after,
            )
        except Exception:
            finalized = False
        if not finalized:
            return _DeliveryDispatch(
                "uncertain",
                "Telegram rejection could not be recorded safely; Jarvis will not automatically resend.",
            )
        current = self.store.get_scheduled_delivery(delivery_key)
        if current is not None and str(current["state"]) == "failed":
            return _DeliveryDispatch(
                "failed",
                _telegram_delivery_failure_message(error_code, prefix="delivery failed")
                + " Automatic retries stopped.",
            )
        return _DeliveryDispatch(
            "rejected",
            _telegram_delivery_failure_message(error_code, prefix="delivery failed") + " Retry scheduled.",
        )

    def _dispatch_occurrence(self, occurrence_key: str) -> _DeliveryDispatch:
        rows = self.store.list_scheduled_deliveries_for_occurrence(occurrence_key)
        if not rows:
            return _DeliveryDispatch("missing", "scheduled Telegram delivery was not durably prepared")
        source_bound = str(rows[0]["job_type"]) == "goal_nudge"
        source_fence = self.store.goal_source_effect_fence() if source_bound else nullcontext()
        with source_fence:
            receipt_state = _delivery_receipt_set_state(rows, occurrence_key)
            if receipt_state == "invalid":
                return _DeliveryDispatch(
                    "failed", "scheduled Telegram delivery receipts are inconsistent"
                )
            outputs: list[str] = []
            for row in rows:
                state = str(row["state"])
                if state == "accepted":
                    continue
                if state == "uncertain":
                    return _DeliveryDispatch(
                        "uncertain",
                        "Telegram outcome is unknown. It may have been accepted; Jarvis will not automatically resend.",
                    )
                if state == "failed":
                    return _DeliveryDispatch(
                        "failed",
                        "Telegram delivery reached a terminal failure; automatic retries stopped.",
                    )
                dispatch = self._dispatch_delivery_row(row)
                outputs.append(dispatch.output)
                if dispatch.state != "accepted":
                    return _DeliveryDispatch(dispatch.state, dispatch.output)
            return _DeliveryDispatch(
                "accepted", "; ".join(outputs) or "accepted by Telegram API"
            )

    def _record_delivery_status(
        self,
        occurrence_key: str,
        dispatch: _DeliveryDispatch,
        now_dt: datetime,
    ) -> None:
        rows = self.store.list_scheduled_deliveries_for_occurrence(occurrence_key)
        if not rows:
            return
        actual_state = _delivery_receipt_set_state(rows, occurrence_key)
        if actual_state == "invalid":
            actual_state = "failed"
        terminal_states = {"accepted", "uncertain", "rejected", "failed"}
        if actual_state in terminal_states:
            dispatch = _DeliveryDispatch(actual_state, dispatch.output)
        elif dispatch.state not in terminal_states:
            return
        first = rows[0]
        job_id: int | None = None
        if str(first["job_type"]) == MORNING_BRIEF_JOB_TYPE:
            current = self._morning_brief_row()
            if current is not None:
                job_id = int(current["id"])
        elif first["job_id"] is not None:
            job_id = int(first["job_id"])
        if job_id is None:
            return
        updates: dict[str, Any] = {
            "last_delivery_status": dispatch.state,
            "last_delivery_status_at": iso(now_dt),
            "last_delivery_occurrence_key": occurrence_key,
            "last_delivery_chunk_count": len(rows),
            "last_delivery_content_stored": any(row["payload"] is not None for row in rows),
        }
        if str(first["job_type"]) == MORNING_BRIEF_JOB_TYPE and dispatch.state == "accepted":
            updates.update(
                {
                    "last_sent_date": now_dt.date().isoformat(),
                    "last_sent_at": iso(now_dt),
                    "last_send_status": "accepted",
                    "delivery_occurrence_key": occurrence_key,
                }
            )
        try:
            run_status = {
                "accepted": "ok",
                "uncertain": "failed",
                "rejected": "failed",
                "failed": "failed",
                "missing": "failed",
            }.get(dispatch.state)
            converged = (
                self.store.converge_job_run_history(
                    job_id,
                    occurrence_key,
                    run_status,
                    updates=updates,
                )
                if run_status is not None
                else False
            )
            if not converged:
                self.store.merge_job_metadata(job_id, updates)
        except Exception:
            return

    def _reconcile_delivery_history(self) -> None:
        try:
            rows = self.store.list_scheduled_delivery_history_reconciliations(100)
        except Exception:
            return
        for row in rows:
            state = str(row["delivery_state"])
            try:
                self._record_delivery_status(
                    str(row["occurrence_key"]),
                    _DeliveryDispatch(state, "durable delivery history reconciliation"),
                    datetime.now(),
                )
            except Exception:
                continue

    def _reconcile_scheduled_note_history(self) -> None:
        try:
            rows = self.store.list_scheduled_note_history_reconciliations(100)
        except Exception:
            return
        for row in rows:
            state = str(row["state"])
            if state not in {"published", "failed"}:
                continue
            try:
                self._record_scheduled_note_status(
                    row,
                    _ScheduledNoteDispatch(
                        state,
                        "durable scheduled note history reconciliation",
                    ),
                )
            except Exception:
                continue

    def _recover_and_drain_scheduled_notes(self) -> list[str]:
        now_utc = datetime.utcnow().replace(microsecond=0)
        stale_before = (
            now_utc - timedelta(seconds=NOTE_PUBLICATION_CLAIM_SECONDS)
        ).isoformat() + "Z"
        recovered = self.store.recover_stale_scheduled_note_publications(
            stale_before,
            limit=100,
        )
        lines: list[str] = []
        recovered_count = sum(
            int(recovered.get(key, 0))
            for key in ("claimed_to_prepared", "claimed_to_failed")
        )
        if recovered_count:
            lines.append(
                f"Recovered {recovered_count} stale scheduled note publication claim(s)."
            )
        self._reconcile_scheduled_note_history()
        rows = self.store.list_ready_scheduled_note_publications(
            NOTE_PUBLICATION_DRAIN_LIMIT
        )
        seen: set[str] = set()
        for row in rows:
            occurrence_key = str(row["occurrence_key"])
            if occurrence_key in seen:
                continue
            seen.add(occurrence_key)
            try:
                dispatch = self._dispatch_scheduled_note_occurrence(occurrence_key)
                lines.append(f"Recovered scheduled note publication: {dispatch.output}.")
            except Exception as exc:
                lines.append(
                    "Scheduled note recovery failed for one occurrence: "
                    f"{_safe_job_failure_code(exc)}; other work continued. "
                    f"{_scheduler_recovery_guidance()}"
                )
        return lines

    def _recover_and_drain_outboxes(self) -> list[str]:
        lines = self._recover_and_drain_scheduled_notes()
        lines.extend(self._recover_and_drain_deliveries())
        return lines

    def _recover_and_drain_deliveries(self) -> list[str]:
        now_utc = datetime.utcnow().replace(microsecond=0)
        stale_before = (now_utc - timedelta(seconds=DELIVERY_CLAIM_SECONDS)).isoformat() + "Z"
        now_text = now_utc.isoformat() + "Z"
        recovered = self.store.recover_stale_scheduled_deliveries(stale_before)
        lines: list[str] = []
        if recovered.get("sending_to_uncertain", 0):
            for occurrence_key in recovered.get("uncertain_occurrence_keys", []):
                self._record_delivery_status(
                    str(occurrence_key),
                    _DeliveryDispatch(
                        "uncertain",
                        "Telegram outcome is unknown; no automatic resend.",
                    ),
                    datetime.now(),
                )
            lines.append(
                f"Held {recovered['sending_to_uncertain']} scheduled Telegram delivery chunk(s) "
                f"with unknown outcome; no automatic resend. "
                f"{_scheduler_recovery_guidance(automatic_retry=False, delivery_receipt_may_be_unknown=True)}"
            )
        self._reconcile_delivery_history()
        terminal_before = (now_utc - timedelta(days=DELIVERY_TERMINAL_RETENTION_DAYS)).isoformat() + "Z"
        uncertain_before = (now_utc - timedelta(days=DELIVERY_UNCERTAIN_RETENTION_DAYS)).isoformat() + "Z"
        pruned = self.store.prune_scheduled_deliveries(terminal_before, uncertain_before, 100)
        if pruned:
            lines.append(f"Pruned {pruned} expired scheduled Telegram delivery receipt chunk(s).")
        rows = self.store.list_ready_scheduled_deliveries(now_text, DELIVERY_DRAIN_LIMIT)
        seen: set[str] = set()
        for row in rows:
            occurrence_key = str(row["occurrence_key"])
            if occurrence_key in seen:
                continue
            seen.add(occurrence_key)
            try:
                dispatch = self._dispatch_occurrence(occurrence_key)
                self._record_delivery_status(occurrence_key, dispatch, datetime.now())
                lines.append(f"Recovered scheduled Telegram delivery: {dispatch.output}.")
            except Exception as exc:
                lines.append(
                    f"Scheduled Telegram recovery failed for one occurrence: "
                    f"{_safe_job_failure_code(exc)}; other work continued. "
                    f"{_scheduler_recovery_guidance(delivery_receipt_may_be_unknown=True)}"
                )
        return lines

    def _morning_brief_result(self, row: Any, plan: _DeliveryPlan, now_dt: datetime) -> str:
        dispatch = self._dispatch_occurrence(plan.occurrence_key)
        self._record_delivery_status(plan.occurrence_key, dispatch, now_dt)
        if dispatch.state == "accepted":
            try:
                self._merge_job_metadata(
                    int(row["id"]),
                    {
                        "last_sent_date": now_dt.date().isoformat(),
                        "last_sent_at": iso(now_dt),
                        "last_send_status": "accepted",
                        "delivery_occurrence_key": plan.occurrence_key,
                    },
                )
            except Exception:
                pass
            if plan.existing:
                return f"Morning Brief already accepted by Telegram API today ({now_dt.date().isoformat()}); skipped duplicate."
            return "Morning Brief accepted by Telegram API."
        if dispatch.state == "uncertain":
            return "Morning Brief Telegram outcome is unknown. It may have been accepted; Jarvis will not automatically resend."
        if dispatch.state == "rejected":
            return dispatch.output.replace("delivery failed:", "Morning Brief not sent:", 1)
        if dispatch.state == "failed":
            return dispatch.output.replace("delivery failed:", "Morning Brief not sent:", 1)
        return f"Morning Brief queued in the durable Telegram outbox ({dispatch.state})."

    def _prepare_morning_brief(self, row: Any, now_dt: datetime) -> tuple[_DeliveryPlan | None, str]:
        occurrence_key = self._delivery_occurrence_key(row, now_dt)
        existing = self._existing_delivery_plan(occurrence_key)
        if existing is not None:
            return existing, "Morning Brief already has a durable Telegram delivery receipt."
        sent_date = now_dt.date().isoformat()
        if _job_metadata(row).get("last_sent_date") == sent_date:
            return None, (
                f"Morning Brief already accepted by Telegram API today ({sent_date}) under the legacy guard; "
                "skipped duplicate during outbox migration."
            )
        from jarvis_v2.automations import telegram_control

        if not telegram_control._owner_chat_id():
            return None, _telegram_delivery_failure_message(
                "missing_or_invalid_owner",
                prefix="Morning Brief not sent",
            )
        brief = _build_live_daily_brief(self.config or load_config())
        return self._delivery_plan(row, brief, now_dt), "Morning Brief prepared for durable Telegram delivery."

    def run_job_now(self, name: str) -> str:
        rows = [row for row in self.store.list_jobs() if row["name"].lower() == name.lower()]
        if not rows:
            return f"No job found named {name}."
        claimed = self._claim_job(rows[0])
        if claimed is None:
            return f"Job {rows[0]['name']} is already running; run skipped."
        row, lease_token = claimed
        heartbeat = self._claim_heartbeat(row, lease_token)
        if not heartbeat.start():
            self._release_claim(row, lease_token)
            return f"Skipped stale claim for {row['name']}; job was not started."

        run_at = datetime.now()
        try:
            plan: _DeliveryPlan | None = None
            note_plan: _ScheduledNotePlan | None = None
            source_job_type = _job_occurrence_type(row)
            if source_job_type == MORNING_BRIEF_JOB_TYPE:
                heartbeat.require_authority()
                plan, output = self._prepare_morning_brief(row, run_at)
                heartbeat.require_authority()
                result = _JobHandlerResult(True, output)
            else:
                raw_output = self._run_owned_job_type(
                    heartbeat,
                    source_job_type,
                    row=row,
                    now=run_at,
                    lease_token=lease_token,
                )
                result = _job_handler_result(raw_output, source_job_type)
                output = result.output
                if result.ok:
                    if source_job_type in SCHEDULED_NOTE_JOB_TYPES:
                        output = self.vault.scheduled_note_payload(output)
                    note_plan = self._scheduled_note_plan(
                        row,
                        output,
                        delivery_plan=None,
                        allow_disabled=True,
                        source_manifest_kind=result.source_manifest_kind,
                        source_manifest_digest=result.source_manifest_digest,
                    )
            completed_at = datetime.now()
            next_run = self._next_run_after(row, completed_at)
        except Exception:
            stale_claim = heartbeat.ownership_lost
            self._release_claim(row, lease_token)
            heartbeat.stop()
            if stale_claim:
                return (
                    f"Skipped stale claim for {row['name']}; run was not finalized. "
                    f"{_scheduler_recovery_guidance(automatic_retry=False)}"
                )
            raise

        if not result.ok:
            retry_at = iso(completed_at + timedelta(seconds=JOB_FAILURE_RETRY_SECONDS))
            failure = f"Job failed: {result.failure_code}."
            try:
                recorded = self.store.mark_claimed_job_run(
                    int(row["id"]),
                    lease_token,
                    _run_marker(completed_at),
                    retry_at,
                    _job_occurrence_schedule_revision(row),
                    complete_occurrence=False,
                    run_history_event=self._run_history_event(row, completed_at, "failed"),
                    expected_occurrence_key=str(row["active_occurrence_key"] or ""),
                    source_job_type=source_job_type,
                )
            except Exception:
                recorded = False
            finally:
                heartbeat.stop()
            suffix = " Retry scheduled." if recorded else " Lease recovery pending."
            return (
                f"Failed {row['name']}: {failure}{suffix} "
                f"{_scheduler_recovery_guidance()}"
            )

        try:
            heartbeat.require_authority()
            finalized = self._finalize_claim_with_delivery(
                row,
                lease_token,
                completed_at,
                next_run,
                plan,
                note_plan,
                output,
            )
        except Exception:
            finalized = False
        finally:
            heartbeat.stop()
        if not finalized:
            return (
                f"Skipped stale claim for {row['name']}; run was not finalized. "
                f"{_scheduler_recovery_guidance(automatic_retry=False)}"
            )
        if note_plan is not None:
            note_dispatch = self._dispatch_scheduled_note_occurrence(
                note_plan.occurrence_key
            )
            if note_dispatch.state != "published":
                return (
                    f"Ran {row['name']}, but its scheduled note is held in the durable outbox "
                    f"({note_dispatch.state}). {_scheduler_recovery_guidance()}"
                )
        if source_job_type == MORNING_BRIEF_JOB_TYPE and plan is not None:
            output = self._morning_brief_result(row, plan, run_at)
            self._converge_run_history_output(row, plan, output)
        return f"Ran {row['name']}:\n{str(output)[:1200]}"

    def run_due_jobs(self) -> str:
        tick_at = datetime.now()
        self.ensure_env_morning_brief(now=tick_at)
        due_before = iso(tick_at)
        candidates = self.store.due_jobs(due_before)
        outputs: list[str] = []
        try:
            outputs.extend(self._recover_and_drain_scheduled_notes())
        except Exception as exc:
            outputs.append(
                f"Scheduled note outbox recovery failed: {_safe_job_failure_code(exc)}; "
                f"due jobs continued. {_scheduler_recovery_guidance()}"
            )
        try:
            outputs.extend(self._recover_and_drain_deliveries())
        except Exception as exc:
            outputs.append(
                f"Scheduled Telegram outbox recovery failed: {_safe_job_failure_code(exc)}; "
                f"due jobs continued. "
                f"{_scheduler_recovery_guidance(delivery_receipt_may_be_unknown=True)}"
            )
        for candidate in candidates:
            try:
                claimed = self._claim_job(candidate, due_before=due_before)
            except Exception as exc:
                outputs.append(
                    f"Failed {candidate['name']}: Job claim failed: "
                    f"{_safe_job_failure_code(exc)}. {_scheduler_recovery_guidance()}"
                )
                continue
            if claimed is None:
                continue
            row, lease_token = claimed
            heartbeat = self._claim_heartbeat(row, lease_token)
            if not heartbeat.start():
                self._release_claim(row, lease_token)
                outputs.append(
                    f"Skipped stale claim for {row['name']}. "
                    f"{_scheduler_recovery_guidance(automatic_retry=False)}"
                )
                continue

            run_at = datetime.now()
            try:
                plan: _DeliveryPlan | None = None
                note_plan: _ScheduledNotePlan | None = None
                source_job_type = _job_occurrence_type(row)
                if source_job_type == MORNING_BRIEF_JOB_TYPE:
                    heartbeat.require_authority()
                    plan, output = self._prepare_morning_brief(row, run_at)
                    heartbeat.require_authority()
                    result = _JobHandlerResult(True, output)
                else:
                    raw_output = self._run_owned_job_type(
                        heartbeat,
                        source_job_type,
                        row=row,
                        now=run_at,
                        lease_token=lease_token,
                    )
                    result = _job_handler_result(raw_output, source_job_type)
                    output = result.output
                    if result.ok and source_job_type in SCHEDULED_NOTE_JOB_TYPES:
                        output = self.vault.scheduled_note_payload(output)
                    plan = self._delivery_plan_for_output(row, output, run_at) if result.ok else None
                    if result.ok:
                        note_plan = self._scheduled_note_plan(
                            row,
                            output,
                            delivery_plan=plan,
                            allow_disabled=False,
                            source_manifest_kind=result.source_manifest_kind,
                            source_manifest_digest=result.source_manifest_digest,
                        )
                completed_at = datetime.now()
                next_run = self._next_run_after(row, completed_at)
            except Exception as exc:
                if heartbeat.ownership_lost:
                    heartbeat.stop()
                    outputs.append(
                        f"Skipped stale claim for {row['name']}. "
                        f"{_scheduler_recovery_guidance(automatic_retry=False)}"
                    )
                    continue
                failed_at = datetime.now()
                retry_at = iso(failed_at + timedelta(seconds=JOB_FAILURE_RETRY_SECONDS))
                failure = f"Job failed: {_safe_job_failure_code(exc)}."
                try:
                    released = self.store.mark_claimed_job_run(
                        int(row["id"]),
                        lease_token,
                        _run_marker(failed_at),
                        retry_at,
                        _job_occurrence_schedule_revision(row),
                        complete_occurrence=False,
                        run_history_event=self._run_history_event(row, failed_at, "failed"),
                        expected_occurrence_key=str(row["active_occurrence_key"] or ""),
                        source_job_type=source_job_type,
                    )
                except Exception:
                    released = False
                finally:
                    heartbeat.stop()
                retry_status = " Retry scheduled." if released else " Lease recovery pending."
                outputs.append(
                    f"Failed {row['name']}: {failure}{retry_status} "
                    f"{_scheduler_recovery_guidance()}"
                )
                continue

            if not result.ok:
                retry_at = iso(completed_at + timedelta(seconds=JOB_FAILURE_RETRY_SECONDS))
                failure = f"Job failed: {result.failure_code}."
                try:
                    released = self.store.mark_claimed_job_run(
                        int(row["id"]),
                        lease_token,
                        _run_marker(completed_at),
                        retry_at,
                        _job_occurrence_schedule_revision(row),
                        complete_occurrence=False,
                        run_history_event=self._run_history_event(row, completed_at, "failed"),
                        expected_occurrence_key=str(row["active_occurrence_key"] or ""),
                        source_job_type=source_job_type,
                    )
                except Exception:
                    released = False
                finally:
                    heartbeat.stop()
                retry_status = " Retry scheduled." if released else " Lease recovery pending."
                outputs.append(
                    f"Failed {row['name']}: {failure}{retry_status} "
                    f"{_scheduler_recovery_guidance()}"
                )
                continue

            try:
                heartbeat.require_authority()
                finalized = self._finalize_claim_with_delivery(
                    row,
                    lease_token,
                    completed_at,
                    next_run,
                    plan,
                    note_plan,
                    output,
                )
            except Exception:
                finalized = False
            finally:
                heartbeat.stop()
            if not finalized:
                outputs.append(
                    f"Skipped stale claim for {row['name']}. "
                    f"{_scheduler_recovery_guidance(automatic_retry=False)}"
                )
                continue
            try:
                source_fence = (
                    self.store.goal_source_effect_fence()
                    if source_job_type == "goal_nudge"
                    else nullcontext()
                )
                with source_fence:
                    note_delivery = ""
                    if note_plan is not None:
                        note_dispatch = self._dispatch_scheduled_note_occurrence(
                            note_plan.occurrence_key
                        )
                        if note_dispatch.state != "published":
                            outputs.append(
                                f"Ran {row['name']}: scheduled note held in durable outbox "
                                f"({note_dispatch.state}). {_scheduler_recovery_guidance()}\n"
                                f"{str(output)[:1200]}"
                            )
                            continue
                        note_delivery = " [note published]"
                    if source_job_type == MORNING_BRIEF_JOB_TYPE and plan is not None:
                        output = self._morning_brief_result(row, plan, run_at)
                        self._converge_run_history_output(row, plan, output)
                        delivery = ""
                    elif plan is not None:
                        dispatch = self._dispatch_occurrence(plan.occurrence_key)
                        self._record_delivery_status(
                            plan.occurrence_key, dispatch, completed_at
                        )
                        if dispatch.state == "accepted":
                            delivery = " [accepted by Telegram API]"
                        elif dispatch.state == "uncertain":
                            delivery = (
                                " [Telegram outcome unknown; no automatic resend. "
                                f"{_scheduler_recovery_guidance(automatic_retry=False, delivery_receipt_may_be_unknown=True)}]"
                            )
                        elif dispatch.state == "rejected":
                            delivery = f" [{dispatch.output}]"
                        elif dispatch.state == "failed":
                            delivery = f" [{dispatch.output}]"
                        else:
                            delivery = f" [durable Telegram outbox: {dispatch.state}]"
                    else:
                        delivery = self._delivery_skip_suffix(source_job_type, output)
                    outputs.append(
                        f"Ran {row['name']}:{note_delivery}{delivery}\n{str(output)[:1200]}"
                    )
            except Exception as exc:
                outputs.append(
                    f"Ran {row['name']}: status recording failed: "
                    f"{_safe_job_failure_code(exc)}. "
                    f"{_scheduler_recovery_guidance(automatic_retry=False, delivery_receipt_may_be_unknown=True)}"
                )
        return "\n\n".join(outputs) if outputs else "No jobs due."

    def _delivery_skip_suffix(self, job_type: str, output: str) -> str:
        title = DELIVERED_JOB_TITLES.get(job_type)
        if not title:
            return ""
        text = (output or "").strip()
        if not text:
            return " [nothing to deliver]"
        markers = QUIET_OUTPUT_MARKERS.get(job_type, ())
        if any(marker in text for marker in markers):
            return " [quiet — nothing new, delivery skipped]"
        return " [durable Telegram delivery was not prepared]"

    def _next_run_after(self, row, now_dt: datetime) -> str:
        if row["job_type"] == MORNING_BRIEF_JOB_TYPE:
            try:
                scheduled = datetime.fromisoformat(str(row["next_run_at"]))
                return iso(next_daily_run(scheduled.hour, scheduled.minute, now=now_dt))
            except (TypeError, ValueError):
                return iso(now_dt + timedelta(minutes=MORNING_BRIEF_INTERVAL_MINUTES))
        return iso(now_dt + timedelta(minutes=int(row["interval_minutes"])))

    def _run_job_type(
        self,
        job_type: str,
        *,
        row: Any | None = None,
        now: datetime | None = None,
        lease_token: str = "",
        lease_authority: Callable[[], None] | None = None,
    ) -> str | ToolResult | GoalNudgeBuild:
        require_authority = lease_authority or (lambda: None)
        require_authority()
        scheduled_note_key, scheduled_note_date = self._scheduled_note_publication(
            row,
            job_type,
        )
        if job_type == "daily_brief":
            return build_daily_brief(
                self.store,
                self.vault,
                effect_authority=require_authority,
                scheduled_note_key=scheduled_note_key,
                scheduled_note_date=scheduled_note_date,
            )
        if job_type == "goal_nudge":
            return build_goal_nudge(
                self.store,
                self.vault,
                effect_authority=require_authority,
                scheduled_note_key=scheduled_note_key,
                scheduled_note_date=scheduled_note_date,
                include_source_manifest=True,
            )
        if job_type == "weekly_review":
            return build_weekly_review(
                self.store,
                self.vault,
                effect_authority=require_authority,
                scheduled_note_key=scheduled_note_key,
                scheduled_note_date=scheduled_note_date,
            )
        if job_type == "inbox_ingest":
            scheduled_job_claim = (
                (int(row["id"]), lease_token)
                if row is not None and lease_token
                else None
            )
            return ingest_inbox_notes(
                self.store,
                self.vault,
                effect_authority=require_authority,
                scheduled_job_claim=scheduled_job_claim,
            )
        if job_type == "recent_file_digest":
            return build_recent_file_digest(
                self.vault,
                self.config,
                effect_authority=require_authority,
                scheduled_note_key=scheduled_note_key,
                scheduled_note_date=scheduled_note_date,
            )
        if job_type == "state_snapshot":
            if row is None:
                raise RuntimeError("state_snapshot_scheduled_row_required")
            return self._run_receipted_state_snapshot(row, lease_token)
        if job_type == "conversation_compaction":
            from jarvis_v2.automations.compaction import build_conversation_compaction

            metadata = _job_metadata(row)
            try:
                watermark = int(metadata.get("last_compacted_id") or 0)
            except (TypeError, ValueError):
                watermark = 0
            config = self.config or load_config()
            # No now= override: message created_at stamps are UTC, so the
            # keep-recent cutoff must be computed in UTC, not local time.
            output, new_watermark = build_conversation_compaction(
                self.store,
                self.vault,
                config,
                last_compacted_id=watermark,
                job_id=int(row["id"]) if row is not None else None,
                lease_token=lease_token,
            )
            if new_watermark is not None and row is not None:
                require_authority()
                merged = self._merge_job_metadata(
                    int(row["id"]),
                    {
                        "last_compacted_id": int(new_watermark),
                        "last_compacted_at": iso(now or datetime.now()),
                    },
                )
                if not merged:
                    raise RuntimeError("conversation compaction watermark cache could not be updated")
            if output.startswith("Compaction skipped: model unavailable"):
                raise RuntimeError("conversation compaction model unavailable")
            return output
        if job_type == MORNING_BRIEF_JOB_TYPE:
            raise RuntimeError("Morning Brief delivery must run through the durable scheduler outbox")
        return f"Unknown job type: {job_type}"
