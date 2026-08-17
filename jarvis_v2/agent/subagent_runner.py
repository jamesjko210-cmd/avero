"""Durable, custody-verified execution for pure internal subagent work."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import json
import multiprocessing
import re
import secrets
import struct
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Protocol

from jarvis_v2.memory.store import MemoryStore, SubagentAssignmentClaim, SubagentTaskEnqueue


MAX_ENVELOPE_BYTES = 1_048_576
MAX_RESULT_BYTES = 1_048_576
MAX_HANDLER_TIMEOUT_SECONDS = 600.0
DEFAULT_EPHEMERAL_MAX_ITEMS = 1_024
DEFAULT_EPHEMERAL_MAX_BYTES = 64 * 1_048_576
DEFAULT_RESULT_RETENTION_MS = 300_000
HANDLER_PROCESS_STOP_GRACE_SECONDS = 0.25
HANDLER_PROCESS_CLEANUP_LEASE_MS = 2_000
REQUEST_PREFIX = b"jarvis-v2/subagent-request/v1\0"
RESULT_PREFIX = b"jarvis-v2/subagent-result/v1\0"
CODE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}")


def _bounded_code(value: object, *, field: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    if CODE_RE.fullmatch(text) is None:
        raise ValueError(f"invalid {field}")
    return text


def _canonical_json_bytes(value: Any, *, limit: int) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("value must be finite JSON-compatible data") from exc
    if len(encoded) > limit:
        raise ValueError("encoded value exceeds the bounded size")
    return encoded


def _encode_envelope(prefix: bytes, header: dict[str, Any], payload: Any, *, limit: int) -> bytes:
    header_bytes = _canonical_json_bytes(header, limit=16_384)
    payload_bytes = _canonical_json_bytes(payload, limit=limit)
    envelope = prefix + struct.pack(">I", len(header_bytes)) + header_bytes + payload_bytes
    if len(envelope) > limit:
        raise ValueError("envelope exceeds the bounded size")
    return envelope


def _decode_envelope(envelope: bytes, prefix: bytes, *, limit: int) -> tuple[dict[str, Any], Any]:
    if type(envelope) is not bytes or len(envelope) > limit or not envelope.startswith(prefix):
        raise ValueError("invalid bounded envelope")
    offset = len(prefix)
    if len(envelope) < offset + 4:
        raise ValueError("truncated envelope")
    header_size = struct.unpack(">I", envelope[offset : offset + 4])[0]
    header_start = offset + 4
    payload_start = header_start + header_size
    if header_size > 16_384 or payload_start > len(envelope):
        raise ValueError("invalid envelope header")
    try:
        header = json.loads(envelope[header_start:payload_start].decode("ascii"))
        payload = json.loads(envelope[payload_start:].decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid envelope JSON") from exc
    if type(header) is not dict:
        raise ValueError("envelope header must be an object")
    return header, payload


def request_envelope(
    *, kind: str, handler_id: str, handler_version: int, payload: Any
) -> tuple[str, bytes]:
    kind = _bounded_code(kind, field="kind")
    handler_id = _bounded_code(handler_id, field="handler id")
    if type(handler_version) is not int or handler_version < 1:
        raise ValueError("handler version must be positive")
    envelope = _encode_envelope(
        REQUEST_PREFIX,
        {
            "handler_id": handler_id,
            "handler_version": handler_version,
            "kind": kind,
            "payload_codec": "json-v1",
        },
        payload,
        limit=MAX_ENVELOPE_BYTES,
    )
    return hashlib.sha256(envelope).hexdigest(), envelope


def result_envelope(
    *,
    task_id: str,
    request_digest: str,
    kind: str,
    handler_id: str,
    handler_version: int,
    result_type: str,
    result: Any,
) -> tuple[str, bytes]:
    envelope = _encode_envelope(
        RESULT_PREFIX,
        {
            "handler_id": _bounded_code(handler_id, field="handler id"),
            "handler_version": handler_version,
            "kind": _bounded_code(kind, field="kind"),
            "request_digest": request_digest,
            "result_type": _bounded_code(result_type, field="result type"),
            "task_id": _bounded_code(task_id, field="task id"),
        },
        result,
        limit=MAX_RESULT_BYTES,
    )
    return hashlib.sha256(envelope).hexdigest(), envelope


def decode_result_envelope(envelope: bytes) -> tuple[dict[str, Any], Any]:
    return _decode_envelope(envelope, RESULT_PREFIX, limit=MAX_RESULT_BYTES)


CustodyMode = Literal["ephemeral", "authoritative_rehydrating"]


@dataclass(frozen=True)
class CustodyRead:
    status: Literal["found", "missing", "unavailable"]
    envelope: bytes | None = None
    authoritative: bool = False


class PayloadCustodian(Protocol):
    mode: CustodyMode

    async def put_if_absent(self, request_digest: str, envelope: bytes) -> str: ...

    async def fetch(self, request_digest: str) -> CustodyRead: ...

    async def discard(self, request_digest: str) -> bool: ...

    async def list_digests(self) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class SinkLookup:
    status: Literal["missing", "found", "unavailable"]
    request_digest: str = ""
    kind: str = ""
    handler_id: str = ""
    handler_version: int = 0
    result_digest: str = ""
    result_type: str = ""


@dataclass(frozen=True)
class ResultRecord:
    task_id: str
    request_digest: str
    kind: str
    handler_id: str
    handler_version: int
    result_digest: str
    result_type: str
    envelope: bytes


def _result_record_matches_contract(record: ResultRecord) -> bool:
    if not hmac.compare_digest(
        hashlib.sha256(record.envelope).hexdigest(), record.result_digest
    ):
        return False
    try:
        header, _result = decode_result_envelope(record.envelope)
    except ValueError:
        return False
    return (
        header.get("task_id") == record.task_id
        and header.get("request_digest") == record.request_digest
        and header.get("kind") == record.kind
        and header.get("handler_id") == record.handler_id
        and header.get("handler_version") == record.handler_version
        and header.get("result_type") == record.result_type
    )


class ResultSink(Protocol):
    async def lookup(self, task_id: str) -> SinkLookup: ...

    async def commit(self, record: ResultRecord) -> str: ...

    async def stage(self, record: ResultRecord) -> str: ...

    async def publish(self, task_id: str, result_digest: str) -> str: ...

    async def discard(self, task_id: str, result_digest: str) -> bool: ...

    async def get_record(self, task_id: str) -> ResultRecord | None: ...


class InMemoryPayloadCustodian:
    def __init__(
        self,
        mode: CustodyMode = "ephemeral",
        *,
        max_items: int = DEFAULT_EPHEMERAL_MAX_ITEMS,
        max_bytes: int = DEFAULT_EPHEMERAL_MAX_BYTES,
    ) -> None:
        if mode not in {"ephemeral", "authoritative_rehydrating"}:
            raise ValueError("invalid custody mode")
        if type(max_items) is not int or not 1 <= max_items <= 1_000_000:
            raise ValueError("invalid custody item capacity")
        if type(max_bytes) is not int or not MAX_ENVELOPE_BYTES <= max_bytes <= 2**40:
            raise ValueError("invalid custody byte capacity")
        self.mode = mode
        self.max_items = max_items
        self.max_bytes = max_bytes
        self._items: dict[str, bytes] = {}
        self._bytes = 0
        self.available = True
        self._lock = asyncio.Lock()

    async def put_if_absent(self, request_digest: str, envelope: bytes) -> str:
        if not self.available:
            return "unavailable"
        if not hmac.compare_digest(hashlib.sha256(envelope).hexdigest(), request_digest):
            return "digest_mismatch"
        async with self._lock:
            existing = self._items.get(request_digest)
            if existing is None:
                stored = bytes(envelope)
                if len(self._items) >= self.max_items or self._bytes + len(stored) > self.max_bytes:
                    return "capacity"
                self._items[request_digest] = stored
                self._bytes += len(stored)
                return "stored"
            return "existing" if hmac.compare_digest(existing, envelope) else "collision"

    async def fetch(self, request_digest: str) -> CustodyRead:
        if not self.available:
            return CustodyRead("unavailable", authoritative=self.mode == "authoritative_rehydrating")
        async with self._lock:
            envelope = self._items.get(request_digest)
        if envelope is None:
            return CustodyRead("missing", authoritative=self.mode == "authoritative_rehydrating")
        return CustodyRead("found", bytes(envelope), self.mode == "authoritative_rehydrating")

    async def discard(self, request_digest: str) -> bool:
        async with self._lock:
            envelope = self._items.pop(request_digest, None)
            if envelope is None:
                return False
            self._bytes -= len(envelope)
            return True

    async def list_digests(self) -> tuple[str, ...]:
        async with self._lock:
            return tuple(self._items)


class InMemoryResultSink:
    def __init__(
        self,
        *,
        max_items: int = DEFAULT_EPHEMERAL_MAX_ITEMS,
        max_bytes: int = DEFAULT_EPHEMERAL_MAX_BYTES,
    ) -> None:
        if type(max_items) is not int or not 1 <= max_items <= 1_000_000:
            raise ValueError("invalid result item capacity")
        if type(max_bytes) is not int or not MAX_RESULT_BYTES <= max_bytes <= 2**40:
            raise ValueError("invalid result byte capacity")
        self.max_items = max_items
        self.max_bytes = max_bytes
        self._items: dict[str, ResultRecord] = {}
        self._staged: dict[str, ResultRecord] = {}
        self._bytes = 0
        self.available = True
        self._lock = asyncio.Lock()

    @staticmethod
    def _record_matches(existing: ResultRecord, record: ResultRecord) -> bool:
        return (
            existing.request_digest == record.request_digest
            and existing.kind == record.kind
            and existing.handler_id == record.handler_id
            and existing.handler_version == record.handler_version
            and existing.result_digest == record.result_digest
            and existing.result_type == record.result_type
            and hmac.compare_digest(existing.envelope, record.envelope)
        )

    @staticmethod
    def _record_is_bound(record: ResultRecord) -> bool:
        return _result_record_matches_contract(record)

    def _has_capacity(self, record: ResultRecord) -> bool:
        return (
            len(set(self._items) | set(self._staged)) < self.max_items
            and self._bytes + len(record.envelope) <= self.max_bytes
        )

    async def lookup(self, task_id: str) -> SinkLookup:
        if not self.available:
            return SinkLookup("unavailable")
        async with self._lock:
            item = self._items.get(task_id)
        if item is None:
            return SinkLookup("missing")
        return SinkLookup(
            "found",
            item.request_digest,
            item.kind,
            item.handler_id,
            item.handler_version,
            item.result_digest,
            item.result_type,
        )

    async def commit(self, record: ResultRecord) -> str:
        if not self.available:
            return "unavailable"
        if not self._record_is_bound(record):
            return "digest_mismatch"
        async with self._lock:
            existing = self._items.get(record.task_id)
            if existing is not None:
                return "existing_match" if self._record_matches(existing, record) else "collision"
            staged = self._staged.get(record.task_id)
            if staged is not None:
                if not self._record_matches(staged, record):
                    return "collision"
                self._items[record.task_id] = staged
                del self._staged[record.task_id]
                return "committed"
            if not self._has_capacity(record):
                return "capacity"
            self._items[record.task_id] = record
            self._bytes += len(record.envelope)
            return "committed"

    async def stage(self, record: ResultRecord) -> str:
        if not self.available:
            return "unavailable"
        if not self._record_is_bound(record):
            return "digest_mismatch"
        async with self._lock:
            published = self._items.get(record.task_id)
            if published is not None:
                return (
                    "existing_match"
                    if self._record_matches(published, record)
                    else "collision"
                )
            existing = self._staged.get(record.task_id)
            if existing is None:
                if not self._has_capacity(record):
                    return "capacity"
                self._staged[record.task_id] = record
                self._bytes += len(record.envelope)
                return "staged"
            return "existing_match" if self._record_matches(existing, record) else "collision"

    async def publish(self, task_id: str, result_digest: str) -> str:
        if not self.available:
            return "unavailable"
        async with self._lock:
            existing = self._items.get(task_id)
            if existing is not None:
                status = (
                    "existing_match"
                    if hmac.compare_digest(existing.result_digest, result_digest)
                    else "collision"
                )
                staged = self._staged.get(task_id)
                if status == "existing_match" and staged is not None and self._record_matches(
                    existing, staged
                ):
                    del self._staged[task_id]
                    self._bytes -= len(staged.envelope)
                return status
            staged = self._staged.get(task_id)
            if staged is None:
                return "missing"
            if not hmac.compare_digest(staged.result_digest, result_digest):
                return "collision"
            self._items[task_id] = staged
            del self._staged[task_id]
            return "published"

    async def discard(self, task_id: str, result_digest: str) -> bool:
        async with self._lock:
            removed = False
            for items in (self._staged, self._items):
                record = items.get(task_id)
                if record is not None and hmac.compare_digest(
                    record.result_digest, result_digest
                ):
                    del items[task_id]
                    self._bytes -= len(record.envelope)
                    removed = True
            return removed

    async def get_record(self, task_id: str) -> ResultRecord | None:
        async with self._lock:
            return self._items.get(task_id)


RunnerHandler = Callable[[Any], Awaitable[Any]]


def _handler_process_main(run: RunnerHandler, payload: Any, sender: Any) -> None:
    """Execute one pure handler and return only a bounded, detail-free message."""
    try:
        result = asyncio.run(run(payload))
        try:
            encoded = _canonical_json_bytes(result, limit=MAX_RESULT_BYTES)
        except ValueError:
            message = b"\x02"
        else:
            message = b"\x01" + encoded
    except BaseException:
        message = b"\x00"
    finally:
        payload = None
    try:
        sender.send_bytes(message)
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        sender.close()


def _handler_process_context() -> multiprocessing.context.BaseContext:
    return multiprocessing.get_context("spawn")


async def _stop_handler_process(
    process: multiprocessing.Process, *, allow_graceful_exit: bool = False
) -> bool:
    """Boundedly reap a handler process, escalating from terminate to kill."""
    if not process.is_alive():
        await asyncio.to_thread(process.join, 0)
        return True
    if allow_graceful_exit:
        await asyncio.to_thread(process.join, HANDLER_PROCESS_STOP_GRACE_SECONDS)
        if not process.is_alive():
            return True
    process.terminate()
    await asyncio.to_thread(process.join, HANDLER_PROCESS_STOP_GRACE_SECONDS)
    if process.is_alive():
        kill = getattr(process, "kill", None)
        if callable(kill):
            kill()
        else:
            process.terminate()
        await asyncio.to_thread(process.join, HANDLER_PROCESS_STOP_GRACE_SECONDS)
    return not process.is_alive()


@dataclass(frozen=True)
class HandlerSpec:
    kind: str
    handler_id: str
    version: int
    result_type: str
    timeout_seconds: float
    run: RunnerHandler
    validate: Callable[[Any], bool] = lambda _payload: True
    pure: bool = True
    replay_safe: bool = True

    def __post_init__(self) -> None:
        _bounded_code(self.kind, field="kind")
        _bounded_code(self.handler_id, field="handler id")
        _bounded_code(self.result_type, field="result type")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("handler version must be positive")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or self.timeout_seconds <= 0
            or self.timeout_seconds > MAX_HANDLER_TIMEOUT_SECONDS
        ):
            raise ValueError("handler timeout is invalid")
        if self.pure is not True or self.replay_safe is not True:
            raise ValueError("production subagent handlers must be pure and replay safe")
        if not inspect.iscoroutinefunction(self.run):
            raise ValueError("production subagent handlers must be async functions")
        if (
            not inspect.isfunction(self.run)
            or not self.run.__module__
            or not self.run.__qualname__
            or "<locals>" in self.run.__qualname__
            or self.run.__name__ == "<lambda>"
        ):
            raise ValueError("production subagent handlers must be importable module-level functions")
        if not callable(self.validate):
            raise ValueError("production subagent handler validator must be callable")


class HandlerRegistry:
    def __init__(self, specs: list[HandlerSpec] | tuple[HandlerSpec, ...]) -> None:
        items: dict[str, HandlerSpec] = {}
        for spec in specs:
            if spec.kind in items:
                raise ValueError(f"duplicate subagent handler kind: {spec.kind}")
            items[spec.kind] = spec
        self._items = items

    def get(self, kind: str) -> HandlerSpec | None:
        return self._items.get(kind)


@dataclass(frozen=True)
class RunnerOutcome:
    status: str
    worker_id: str = ""
    task_id: str = ""
    assignment_id: str = ""
    error_code: str = ""


class DurableSubagentRunner:
    def __init__(
        self,
        *,
        store: MemoryStore,
        custodian: PayloadCustodian,
        result_sink: ResultSink,
        handlers: HandlerRegistry,
        worker_count: int = 10,
        lease_ms: int = 30_000,
        renew_interval_ms: int = 5_000,
        result_retention_ms: int = DEFAULT_RESULT_RETENTION_MS,
        runtime_id: str | None = None,
    ) -> None:
        if type(worker_count) is not int or not 1 <= worker_count <= 10:
            raise ValueError("runner worker count is out of range")
        if type(lease_ms) is not int or not 100 <= lease_ms <= 600_000:
            raise ValueError("runner lease is out of range")
        if type(renew_interval_ms) is not int or not 10 <= renew_interval_ms < lease_ms:
            raise ValueError("runner renewal interval is out of range")
        if (
            type(result_retention_ms) is not int
            or not 100 <= result_retention_ms <= 86_400_000
        ):
            raise ValueError("runner result retention is out of range")
        self.store = store
        self.custodian = custodian
        self.result_sink = result_sink
        self.handlers = handlers
        self.worker_count = worker_count
        self.lease_ms = lease_ms
        self.renew_interval_ms = renew_interval_ms
        self.result_retention_ms = result_retention_ms
        self.runtime_id = runtime_id or f"runner-{secrets.token_hex(12)}"
        _bounded_code(self.runtime_id, field="runtime id")
        self._generations: dict[str, int] = {}
        self._active_processes: dict[
            multiprocessing.Process, SubagentAssignmentClaim
        ] = {}
        self._active_worker_tasks: dict[
            asyncio.Task[RunnerOutcome], SubagentAssignmentClaim
        ] = {}
        self._started = False
        self._stopping = False

    async def submit(
        self,
        *,
        kind: str,
        payload: Any,
        priority: int = 0,
        max_attempts: int = 3,
    ) -> SubagentTaskEnqueue:
        spec = self.handlers.get(kind)
        if spec is None:
            raise ValueError("subagent handler kind is not registered")
        try:
            payload_valid = spec.validate(payload) is True
        except Exception:
            payload_valid = False
        if not payload_valid:
            raise ValueError("subagent payload does not match the registered handler contract")
        digest, envelope = request_envelope(
            kind=kind,
            handler_id=spec.handler_id,
            handler_version=spec.version,
            payload=payload,
        )
        preflight = self.store.get_subagent_task_ephemeral_status_by_request_digest(
            digest
        )
        if preflight.status == "FOUND" and preflight.state in {
            "succeeded",
            "failed",
            "uncertain",
        }:
            return self.store.enqueue_subagent_task(
                request_digest=digest,
                kind=kind,
                priority=priority,
                replay_policy="safe",
                max_attempts=max_attempts,
            )
        try:
            custody = await self.custodian.put_if_absent(digest, envelope)
        except Exception as exc:
            raise RuntimeError("payload custody unavailable") from exc
        if custody not in {"stored", "existing"}:
            raise RuntimeError(f"payload custody refused enqueue: {custody}")
        enqueue_args = {
            "request_digest": digest,
            "kind": kind,
            "priority": priority,
            "replay_policy": "safe",
            "max_attempts": max_attempts,
        }
        try:
            queued = self.store.enqueue_subagent_task(**enqueue_args)
        except BaseException as first_error:
            try:
                queued = self.store.enqueue_subagent_task(**enqueue_args)
            except BaseException:
                try:
                    durable = self.store.get_subagent_task_ephemeral_status_by_request_digest(
                        digest
                    )
                except BaseException:
                    durable = None
                if custody == "stored" and (
                    durable is None or durable.status == "NOT_FOUND"
                ):
                    await self.custodian.discard(digest)
                if durable is not None and durable.status == "FOUND":
                    exact = (
                        durable.kind == kind
                        and durable.priority == priority
                        and durable.replay_policy == "safe"
                        and durable.max_attempts == max_attempts
                    )
                    if durable.state in {"succeeded", "failed", "uncertain"}:
                        await self.custodian.discard(digest)
                    return SubagentTaskEnqueue(
                        "EXISTING" if exact else "COLLISION",
                        durable.task_id,
                        durable.state,
                    )
                raise first_error
        if queued.task_state in {"succeeded", "failed", "uncertain"}:
            await self.custodian.discard(digest)
        return queued

    async def start(self) -> dict[str, int]:
        self._stopping = False
        recovery = self.store.recover_expired_subagent_assignments(
            allow_running_requeue=self.custodian.mode == "authoritative_rehydrating"
        )
        self.store.configure_subagent_workers(self.worker_count)
        self._generations = {}
        self._refresh_worker_generations()
        self._started = True
        return recovery

    def _refresh_worker_generations(self) -> None:
        self.store.recover_expired_subagent_assignments(
            allow_running_requeue=self.custodian.mode == "authoritative_rehydrating"
        )
        for index in range(1, self.worker_count + 1):
            worker_id = f"subagent-{index:02d}"
            if worker_id in self._generations:
                continue
            generation = self.store.activate_subagent_worker(worker_id, self.runtime_id)
            if generation is not None:
                self._generations[worker_id] = generation

    async def stop(self) -> None:
        self._stopping = True
        worker_tasks = tuple(self._active_worker_tasks)
        for task in worker_tasks:
            if not task.done():
                task.cancel()
        if worker_tasks:
            await asyncio.gather(*worker_tasks, return_exceptions=True)
        for process, claim in list(self._active_processes.items()):
            if not await _stop_handler_process(process):
                self._quarantine_unreaped(claim)
        for worker_id, generation in list(self._generations.items()):
            self.store.deactivate_subagent_worker(worker_id, self.runtime_id, generation)
        self._generations.clear()
        self._started = False

    async def run_once(self) -> list[RunnerOutcome]:
        if not self._started:
            await self.start()
        else:
            self._refresh_worker_generations()
        await self._discard_unreferenced_payloads()
        coroutines = [self._run_worker(worker_id, generation) for worker_id, generation in self._generations.items()]
        if not coroutines:
            return [RunnerOutcome("NO_WORKERS")]
        outcomes = list(await asyncio.gather(*coroutines))
        for outcome in outcomes:
            if outcome.status == "WORKER_LOST":
                self._generations.pop(outcome.worker_id, None)
        return outcomes

    async def _discard_unreferenced_payloads(self) -> None:
        try:
            digests = await self.custodian.list_digests()
        except BaseException:
            return
        for request_digest in digests:
            try:
                status = self.store.get_subagent_task_ephemeral_status_by_request_digest(
                    request_digest
                )
                if status.status == "NOT_FOUND" or status.state in {
                    "succeeded",
                    "failed",
                    "uncertain",
                }:
                    await self.custodian.discard(request_digest)
            except BaseException:
                continue

    async def serve(self, stop_event: asyncio.Event, *, poll_seconds: float = 0.5) -> None:
        if poll_seconds <= 0 or poll_seconds > 30:
            raise ValueError("runner poll interval is out of range")
        if not self._started:
            await self.start()
        cycle: asyncio.Task[list[RunnerOutcome]] | None = None
        stop_wait: asyncio.Task[bool] | None = None
        try:
            while not stop_event.is_set():
                cycle = asyncio.create_task(self.run_once())
                stop_wait = asyncio.create_task(stop_event.wait())
                done, _ = await asyncio.wait({cycle, stop_wait}, return_when=asyncio.FIRST_COMPLETED)
                if stop_wait in done and stop_event.is_set() and not cycle.done():
                    self._stopping = True
                    cycle.cancel()
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(cycle, return_exceptions=True),
                            timeout=2.0,
                        )
                    except asyncio.TimeoutError:
                        pass
                    await asyncio.gather(stop_wait, return_exceptions=True)
                    stop_wait = None
                    break
                stop_wait.cancel()
                await asyncio.gather(stop_wait, return_exceptions=True)
                stop_wait = None
                await cycle
                if not stop_event.is_set():
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=poll_seconds)
                    except asyncio.TimeoutError:
                        pass
        finally:
            self._stopping = True
            if stop_wait is not None:
                stop_wait.cancel()
                await asyncio.gather(stop_wait, return_exceptions=True)
            await self.stop()
            if cycle is not None:
                if not cycle.done():
                    cycle.cancel()
                await asyncio.gather(cycle, return_exceptions=True)

    async def _reject_before_start(
        self, claim: SubagentAssignmentClaim, error_code: str
    ) -> RunnerOutcome:
        transition = self.store.reject_subagent_assignment_before_start(
            claim.assignment_id,
            claim.worker_id,
            self.runtime_id,
            claim.worker_generation,
            claim.lease_token,
            error_code=error_code,
        )
        return RunnerOutcome(
            transition.status,
            claim.worker_id,
            claim.task_id,
            claim.assignment_id,
            error_code,
        )

    async def _release_before_start(
        self, claim: SubagentAssignmentClaim, error_code: str
    ) -> RunnerOutcome:
        transition = self.store.release_subagent_assignment_before_start(
            claim.assignment_id,
            claim.worker_id,
            self.runtime_id,
            claim.worker_generation,
            claim.lease_token,
            error_code=error_code,
        )
        return RunnerOutcome(
            transition.status,
            claim.worker_id,
            claim.task_id,
            claim.assignment_id,
            error_code,
        )

    async def _run_worker(self, worker_id: str, generation: int) -> RunnerOutcome:
        if self._stopping:
            return RunnerOutcome("STOPPING", worker_id)
        if not self.store.heartbeat_subagent_worker(worker_id, self.runtime_id, generation):
            return RunnerOutcome("WORKER_LOST", worker_id)
        claim = self.store.claim_subagent_assignment(
            worker_id,
            self.runtime_id,
            generation,
            lease_ms=self.lease_ms,
        )
        if claim.status != "CLAIMED":
            return RunnerOutcome(claim.status, worker_id)

        owner_task = asyncio.current_task()
        if owner_task is not None:
            self._active_worker_tasks[owner_task] = claim
        try:
            try:
                return await self._run_claim(claim)
            except asyncio.CancelledError:
                if not self._stopping:
                    raise
                rejected = await self._reject_before_start(
                    claim, "service_shutdown"
                )
                if rejected.status == "REJECTED":
                    return rejected
                status = self.store.get_subagent_assignment_status(
                    claim.assignment_id
                )
                if status.status == "FOUND" and status.state == "running":
                    return await self._finalize_failed(claim, "service_shutdown")
                return RunnerOutcome(
                    status.state.upper()
                    if status.status == "FOUND"
                    else rejected.status,
                    claim.worker_id,
                    claim.task_id,
                    claim.assignment_id,
                    "service_shutdown",
                )
        finally:
            try:
                await asyncio.shield(self._discard_terminal_payload(claim))
            finally:
                if owner_task is not None:
                    self._active_worker_tasks.pop(owner_task, None)

    async def _discard_terminal_payload(
        self, claim: SubagentAssignmentClaim
    ) -> None:
        try:
            status = self.store.get_subagent_task_ephemeral_status(claim.task_id)
            if (
                status.status == "FOUND"
                and status.state in {"succeeded", "failed", "uncertain"}
                and hmac.compare_digest(status.request_digest, claim.request_digest)
            ):
                await self.custodian.discard(claim.request_digest)
        except BaseException:
            pass

    async def _mark_result_available(
        self,
        claim: SubagentAssignmentClaim,
        result_digest: str,
    ) -> str:
        try:
            transition = self.store.mark_subagent_result_available(
                claim.task_id,
                claim.request_digest,
                result_digest,
            )
        except BaseException:
            try:
                status = self.store.get_subagent_task_ephemeral_status(claim.task_id)
            except BaseException:
                return "unknown"
            if (
                status.status == "FOUND"
                and status.state == "succeeded"
                and hmac.compare_digest(status.request_digest, claim.request_digest)
                and hmac.compare_digest(status.result_digest, result_digest)
            ):
                return status.result_state
            return "unknown"
        if transition.status in {"AVAILABLE", "ALREADY_AVAILABLE"}:
            return "available"
        return transition.result_state or "unknown"

    async def _release_retired_result(
        self,
        claim: SubagentAssignmentClaim,
        result_digest: str,
        result_state: str,
    ) -> None:
        try:
            await self.result_sink.discard(claim.task_id, result_digest)
        except BaseException:
            return
        try:
            self.store.mark_subagent_result_released(
                claim.task_id,
                claim.request_digest,
                result_digest,
                result_state,
            )
        except BaseException:
            pass

    async def _retire_unavailable_result(
        self,
        claim: SubagentAssignmentClaim,
        result_digest: str,
    ) -> bool:
        try:
            transition = self.store.mark_subagent_result_unavailable(
                claim.task_id,
                claim.request_digest,
                result_digest,
            )
        except BaseException:
            try:
                status = self.store.get_subagent_task_ephemeral_status(claim.task_id)
            except BaseException:
                return False
            if not (
                status.status == "FOUND"
                and status.result_state == "unavailable"
                and hmac.compare_digest(status.request_digest, claim.request_digest)
                and hmac.compare_digest(status.result_digest, result_digest)
            ):
                return False
            transition = status
        try:
            await self.result_sink.discard(claim.task_id, result_digest)
        except BaseException:
            return False
        if transition.result_state == "unavailable":
            try:
                self.store.mark_subagent_result_released(
                    claim.task_id,
                    claim.request_digest,
                    result_digest,
                    "unavailable",
                )
            except BaseException:
                pass
            return True
        return False

    async def _run_claim(self, claim: SubagentAssignmentClaim) -> RunnerOutcome:

        spec = self.handlers.get(claim.kind)
        if spec is None:
            return await self._reject_before_start(claim, "handler_unknown")
        try:
            custody = await self.custodian.fetch(claim.request_digest)
        except Exception:
            return await self._release_before_start(claim, "custodian_unavailable")
        if custody.status == "unavailable":
            return await self._release_before_start(claim, "custodian_unavailable")
        if custody.status != "found" or custody.envelope is None:
            return await self._reject_before_start(claim, "payload_missing")
        if not hmac.compare_digest(hashlib.sha256(custody.envelope).hexdigest(), claim.request_digest):
            return await self._reject_before_start(claim, "payload_digest_mismatch")
        try:
            header, payload = _decode_envelope(custody.envelope, REQUEST_PREFIX, limit=MAX_ENVELOPE_BYTES)
        except ValueError:
            return await self._reject_before_start(claim, "payload_invalid")
        if (
            header.get("kind") != claim.kind
            or header.get("handler_id") != spec.handler_id
            or header.get("handler_version") != spec.version
            or header.get("payload_codec") != "json-v1"
        ):
            return await self._reject_before_start(claim, "handler_contract_mismatch")
        try:
            payload_valid = self.handlers.get(claim.kind) is spec and spec.validate(payload) is True
        except Exception:
            payload_valid = False
        if not payload_valid:
            return await self._reject_before_start(claim, "payload_invalid")

        try:
            existing = await self.result_sink.lookup(claim.task_id)
        except Exception:
            return await self._release_before_start(claim, "sink_unavailable")
        if existing.status == "unavailable":
            return await self._release_before_start(claim, "sink_unavailable")
        if existing.status == "found":
            try:
                existing_record = await self.result_sink.get_record(claim.task_id)
            except Exception:
                return await self._release_before_start(claim, "sink_unavailable")
            if (
                existing.request_digest != claim.request_digest
                or existing.kind != claim.kind
                or existing.handler_id != spec.handler_id
                or existing.handler_version != spec.version
                or not existing.result_digest
                or existing.result_type != spec.result_type
                or existing_record is None
                or not _result_record_matches_contract(existing_record)
                or existing_record.task_id != claim.task_id
                or existing_record.request_digest != claim.request_digest
                or existing_record.kind != claim.kind
                or existing_record.handler_id != spec.handler_id
                or existing_record.handler_version != spec.version
                or existing_record.result_type != spec.result_type
                or not hmac.compare_digest(
                    existing_record.result_digest, existing.result_digest
                )
            ):
                return await self._reject_before_start(claim, "sink_collision")
            try:
                transition = self.store.adopt_subagent_result(
                    claim.assignment_id,
                    claim.worker_id,
                    self.runtime_id,
                    claim.worker_generation,
                    claim.lease_token,
                    result_digest=existing.result_digest,
                    result_type=existing.result_type,
                    result_retention_ms=self.result_retention_ms,
                )
            except BaseException:
                transition = None
            transition_status = transition.status if transition is not None else "OUTCOME_UNKNOWN"
            if transition is None or transition.status == "LEASE_LOST":
                status = self.store.get_subagent_assignment_status(claim.assignment_id)
                if (
                    status.status == "FOUND"
                    and status.state == "succeeded"
                    and status.started is False
                    and hmac.compare_digest(status.result_digest, existing.result_digest)
                    and status.result_type == existing.result_type
                ):
                    transition_status = "ALREADY_ADOPTED"
                elif transition is None:
                    return await self._release_before_start(
                        claim, "adoption_outcome_unknown"
                    )
            if transition_status not in {"ADOPTED", "ALREADY_ADOPTED"}:
                return RunnerOutcome(
                    transition_status,
                    claim.worker_id,
                    claim.task_id,
                    claim.assignment_id,
                )
            retention_state = await self._mark_result_available(
                claim, existing.result_digest
            )
            if retention_state != "available":
                if retention_state == "expired":
                    await self._release_retired_result(
                        claim, existing.result_digest, "expired"
                    )
                else:
                    await self._retire_unavailable_result(
                        claim, existing.result_digest
                    )
                return RunnerOutcome(
                    "RESULT_EXPIRED" if retention_state == "expired" else "RESULT_UNAVAILABLE",
                    claim.worker_id,
                    claim.task_id,
                    claim.assignment_id,
                    "result_expired"
                    if retention_state == "expired"
                    else "sink_publication_unavailable",
                )
            return RunnerOutcome(
                transition_status,
                claim.worker_id,
                claim.task_id,
                claim.assignment_id,
            )

        started = self.store.start_subagent_assignment(
            claim.assignment_id,
            claim.worker_id,
            self.runtime_id,
            claim.worker_generation,
            claim.lease_token,
        )
        if started.status != "STARTED":
            return RunnerOutcome(started.status, worker_id, claim.task_id, claim.assignment_id)
        return await self._execute_started(claim, spec, deepcopy(payload))

    async def _execute_started(
        self, claim: SubagentAssignmentClaim, spec: HandlerSpec, payload: Any
    ) -> RunnerOutcome:
        context = _handler_process_context()
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_handler_process_main,
            args=(spec.run, payload, sender),
            name=f"jarvis-{claim.worker_id}-{claim.assignment_id}",
            daemon=True,
        )
        try:
            process.start()
        except Exception:
            receiver.close()
            sender.close()
            return await self._finalize_failed(claim, "handler_start_error")
        self._active_processes[process] = claim
        sender.close()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + float(spec.timeout_seconds)
        next_renewal = loop.time() + self.renew_interval_ms / 1000.0
        try:
            message: bytes | None = None
            while message is None:
                now = loop.time()
                remaining = deadline - now
                if remaining <= 0:
                    reaped, lease_owned = await self._reap_owned_handler_process(
                        claim, process
                    )
                    if not reaped:
                        return self._quarantine_unreaped(claim)
                    if not lease_owned:
                        return RunnerOutcome(
                            "LEASE_LOST",
                            claim.worker_id,
                            claim.task_id,
                            claim.assignment_id,
                            "lease_lost",
                        )
                    return await self._finalize_failed(claim, "handler_timeout")
                if now >= next_renewal:
                    renewed = self.store.renew_subagent_assignment(
                        claim.assignment_id,
                        claim.worker_id,
                        self.runtime_id,
                        claim.worker_generation,
                        claim.lease_token,
                        lease_ms=self.lease_ms,
                    )
                    if not renewed:
                        reaped = await _stop_handler_process(process)
                        if not reaped:
                            return self._quarantine_unreaped(claim)
                        return RunnerOutcome(
                            "LEASE_LOST",
                            claim.worker_id,
                            claim.task_id,
                            claim.assignment_id,
                            "lease_lost",
                        )
                    next_renewal = loop.time() + self.renew_interval_ms / 1000.0
                    continue
                if receiver.poll():
                    try:
                        message = receiver.recv_bytes(MAX_RESULT_BYTES + 1)
                    except (EOFError, OSError):
                        message = b"\x00"
                    break
                if not process.is_alive():
                    await asyncio.sleep(0)
                    if receiver.poll():
                        try:
                            message = receiver.recv_bytes(MAX_RESULT_BYTES + 1)
                        except (EOFError, OSError):
                            message = b"\x00"
                    else:
                        message = b"\x00"
                    break
                until_renewal = max(0.0, next_renewal - loop.time())
                await asyncio.sleep(min(remaining, until_renewal, 0.02))
            reaped, lease_owned = await self._reap_owned_handler_process(
                claim, process, allow_graceful_exit=True
            )
            if not reaped:
                return self._quarantine_unreaped(claim)
            if not lease_owned:
                return RunnerOutcome(
                    "LEASE_LOST",
                    claim.worker_id,
                    claim.task_id,
                    claim.assignment_id,
                    "lease_lost",
                )
            if loop.time() >= deadline:
                return await self._finalize_failed(claim, "handler_timeout")
            if not message or message[:1] == b"\x00":
                return await self._finalize_failed(claim, "handler_error")
            if message[:1] != b"\x01":
                return await self._finalize_failed(claim, "result_invalid")
            try:
                result = json.loads(message[1:].decode("ascii"))
            except (UnicodeError, json.JSONDecodeError):
                return await self._finalize_failed(claim, "result_invalid")

            try:
                result_digest, envelope = result_envelope(
                    task_id=claim.task_id,
                    request_digest=claim.request_digest,
                    kind=claim.kind,
                    handler_id=spec.handler_id,
                    handler_version=spec.version,
                    result_type=spec.result_type,
                    result=result,
                )
            except ValueError:
                return await self._finalize_failed(claim, "result_invalid")
            record = ResultRecord(
                task_id=claim.task_id,
                request_digest=claim.request_digest,
                kind=claim.kind,
                handler_id=spec.handler_id,
                handler_version=spec.version,
                result_digest=result_digest,
                result_type=spec.result_type,
                envelope=envelope,
            )
            if loop.time() >= deadline:
                return await self._finalize_failed(claim, "handler_timeout")
            renewed = self.store.renew_subagent_assignment(
                claim.assignment_id,
                claim.worker_id,
                self.runtime_id,
                claim.worker_generation,
                claim.lease_token,
                lease_ms=self.lease_ms,
            )
            if not renewed:
                return RunnerOutcome(
                    "LEASE_LOST",
                    claim.worker_id,
                    claim.task_id,
                    claim.assignment_id,
                    "lease_lost",
                )
            try:
                sink_status = await self.result_sink.stage(record)
            except Exception:
                sink_status = "outcome_unknown"
            if sink_status not in {"staged", "existing_match"}:
                try:
                    await self.result_sink.discard(claim.task_id, result_digest)
                except Exception:
                    pass
                if sink_status == "collision":
                    error_code = "sink_collision"
                elif sink_status == "outcome_unknown":
                    error_code = "sink_outcome_unknown"
                else:
                    error_code = "sink_unavailable"
                quarantine = self.store.quarantine_subagent_assignment(
                    claim.assignment_id,
                    claim.worker_id,
                    self.runtime_id,
                    claim.worker_generation,
                    claim.lease_token,
                    error_code=error_code,
                )
                return RunnerOutcome(
                    quarantine.status,
                    claim.worker_id,
                    claim.task_id,
                    claim.assignment_id,
                    error_code,
                )
            if loop.time() >= deadline:
                try:
                    await self.result_sink.discard(claim.task_id, result_digest)
                except Exception:
                    pass
                return await self._finalize_failed(claim, "handler_timeout")
            renewed = self.store.renew_subagent_assignment(
                claim.assignment_id,
                claim.worker_id,
                self.runtime_id,
                claim.worker_generation,
                claim.lease_token,
                lease_ms=self.lease_ms,
            )
            if not renewed:
                try:
                    await self.result_sink.discard(claim.task_id, result_digest)
                except Exception:
                    pass
                return RunnerOutcome(
                    "LEASE_LOST",
                    claim.worker_id,
                    claim.task_id,
                    claim.assignment_id,
                    "lease_lost",
                )
            try:
                transition = self.store.finalize_subagent_assignment(
                    claim.assignment_id,
                    claim.worker_id,
                    self.runtime_id,
                    claim.worker_generation,
                    claim.lease_token,
                    outcome="succeeded",
                    result_digest=result_digest,
                    result_type=spec.result_type,
                    result_retention_ms=self.result_retention_ms,
                )
            except BaseException:
                transition = None
            durable_success = transition is not None and transition.status == "SUCCEEDED"
            transition_status = transition.status if transition is not None else "OUTCOME_UNKNOWN"
            if transition is None or transition.status == "LEASE_LOST":
                try:
                    status = self.store.get_subagent_assignment_status(
                        claim.assignment_id
                    )
                except BaseException:
                    return RunnerOutcome(
                        "FINALIZE_OUTCOME_UNKNOWN",
                        claim.worker_id,
                        claim.task_id,
                        claim.assignment_id,
                        "finalize_outcome_unknown",
                    )
                if (
                    status.status == "FOUND"
                    and status.state == "succeeded"
                    and status.started is True
                    and hmac.compare_digest(status.result_digest, result_digest)
                    and status.result_type == spec.result_type
                ):
                    durable_success = True
                    transition_status = "ALREADY_SUCCEEDED"
                elif transition is None and status.status == "FOUND" and status.state == "running":
                    try:
                        quarantine = self.store.quarantine_subagent_assignment(
                            claim.assignment_id,
                            claim.worker_id,
                            self.runtime_id,
                            claim.worker_generation,
                            claim.lease_token,
                            error_code="finalize_outcome_unknown",
                        )
                    except BaseException:
                        return RunnerOutcome(
                            "FINALIZE_OUTCOME_UNKNOWN",
                            claim.worker_id,
                            claim.task_id,
                            claim.assignment_id,
                            "finalize_outcome_unknown",
                        )
                    if quarantine.status != "UNCERTAIN":
                        return RunnerOutcome(
                            quarantine.status,
                            claim.worker_id,
                            claim.task_id,
                            claim.assignment_id,
                            "finalize_outcome_unknown",
                        )
                    transition_status = quarantine.status
            if not durable_success:
                try:
                    await self.result_sink.discard(claim.task_id, result_digest)
                except Exception:
                    pass
                return RunnerOutcome(
                    transition_status,
                    claim.worker_id,
                    claim.task_id,
                    claim.assignment_id,
                )
            try:
                publish_status = await self.result_sink.publish(
                    claim.task_id, result_digest
                )
            except Exception:
                publish_status = "outcome_unknown"
            if publish_status not in {"published", "existing_match"}:
                await self._retire_unavailable_result(claim, result_digest)
                return RunnerOutcome(
                    "RESULT_UNAVAILABLE",
                    claim.worker_id,
                    claim.task_id,
                    claim.assignment_id,
                    "sink_publish_unavailable",
                )
            retention_state = await self._mark_result_available(claim, result_digest)
            if retention_state != "available":
                if retention_state == "expired":
                    await self._release_retired_result(
                        claim, result_digest, "expired"
                    )
                else:
                    await self._retire_unavailable_result(claim, result_digest)
                return RunnerOutcome(
                    "RESULT_EXPIRED" if retention_state == "expired" else "RESULT_UNAVAILABLE",
                    claim.worker_id,
                    claim.task_id,
                    claim.assignment_id,
                    "result_expired"
                    if retention_state == "expired"
                    else "result_state_unavailable",
                )
            return RunnerOutcome(
                transition_status,
                claim.worker_id,
                claim.task_id,
                claim.assignment_id,
            )
        except asyncio.CancelledError:
            reaped, lease_owned = await asyncio.shield(
                self._reap_owned_handler_process(claim, process)
            )
            if not reaped:
                return self._quarantine_unreaped(claim)
            if not lease_owned:
                return RunnerOutcome(
                    "LEASE_LOST",
                    claim.worker_id,
                    claim.task_id,
                    claim.assignment_id,
                    "lease_lost",
                )
            return await self._finalize_failed(
                claim, "service_shutdown" if self._stopping else "handler_cancelled"
            )
        finally:
            if process.is_alive():
                reaped, _lease_owned = await asyncio.shield(
                    self._reap_owned_handler_process(claim, process)
                )
                if not reaped:
                    self._quarantine_unreaped(claim)
            if not process.is_alive():
                self._active_processes.pop(process, None)
                process.close()
            receiver.close()
            payload = None

    async def _reap_owned_handler_process(
        self,
        claim: SubagentAssignmentClaim,
        process: multiprocessing.Process,
        *,
        allow_graceful_exit: bool = False,
    ) -> tuple[bool, bool]:
        """Reap a child while retaining enough lease for terminal custody."""
        cleanup_lease_ms = max(self.lease_ms, HANDLER_PROCESS_CLEANUP_LEASE_MS)
        try:
            lease_owned = self.store.renew_subagent_assignment(
                claim.assignment_id,
                claim.worker_id,
                self.runtime_id,
                claim.worker_generation,
                claim.lease_token,
                lease_ms=cleanup_lease_ms,
            )
        except BaseException:
            lease_owned = False
        reaped = await _stop_handler_process(
            process, allow_graceful_exit=allow_graceful_exit
        )
        if not reaped or not lease_owned:
            return reaped, lease_owned
        try:
            lease_owned = self.store.renew_subagent_assignment(
                claim.assignment_id,
                claim.worker_id,
                self.runtime_id,
                claim.worker_generation,
                claim.lease_token,
                lease_ms=cleanup_lease_ms,
            )
        except BaseException:
            lease_owned = False
        return reaped, lease_owned

    def _quarantine_unreaped(
        self, claim: SubagentAssignmentClaim
    ) -> RunnerOutcome:
        quarantine = self.store.quarantine_subagent_assignment(
            claim.assignment_id,
            claim.worker_id,
            self.runtime_id,
            claim.worker_generation,
            claim.lease_token,
            error_code="handler_process_unreaped",
        )
        return RunnerOutcome(
            quarantine.status,
            claim.worker_id,
            claim.task_id,
            claim.assignment_id,
            "handler_process_unreaped",
        )

    async def _finalize_failed(
        self, claim: SubagentAssignmentClaim, error_code: str
    ) -> RunnerOutcome:
        try:
            transition = self.store.finalize_subagent_assignment(
                claim.assignment_id,
                claim.worker_id,
                self.runtime_id,
                claim.worker_generation,
                claim.lease_token,
                outcome="failed",
                error_code=error_code,
            )
        except BaseException:
            transition = None
        if transition is None or transition.status == "LEASE_LOST":
            try:
                status = self.store.get_subagent_assignment_status(
                    claim.assignment_id
                )
            except BaseException:
                return RunnerOutcome(
                    "FINALIZE_OUTCOME_UNKNOWN",
                    claim.worker_id,
                    claim.task_id,
                    claim.assignment_id,
                    "finalize_outcome_unknown",
                )
            if (
                status.status == "FOUND"
                and status.state == "failed"
                and status.error_code == error_code
            ):
                return RunnerOutcome(
                    "ALREADY_FAILED",
                    claim.worker_id,
                    claim.task_id,
                    claim.assignment_id,
                    error_code,
                )
            if transition is None and status.status == "FOUND" and status.state == "running":
                try:
                    quarantine = self.store.quarantine_subagent_assignment(
                        claim.assignment_id,
                        claim.worker_id,
                        self.runtime_id,
                        claim.worker_generation,
                        claim.lease_token,
                        error_code="finalize_outcome_unknown",
                    )
                except BaseException:
                    quarantine = None
                return RunnerOutcome(
                    quarantine.status if quarantine is not None else "FINALIZE_OUTCOME_UNKNOWN",
                    claim.worker_id,
                    claim.task_id,
                    claim.assignment_id,
                    "finalize_outcome_unknown",
                )
        if transition is None:
            return RunnerOutcome(
                "FINALIZE_OUTCOME_UNKNOWN",
                claim.worker_id,
                claim.task_id,
                claim.assignment_id,
                "finalize_outcome_unknown",
            )
        return RunnerOutcome(
            transition.status,
            claim.worker_id,
            claim.task_id,
            claim.assignment_id,
            error_code,
        )
