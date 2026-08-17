"""Run the private local durable subagent worker service."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import re
import secrets
import signal
import stat
import struct
import time
from pathlib import Path
from typing import Any

from jarvis_v2.agent.subagent_runner import (
    DurableSubagentRunner,
    HandlerRegistry,
    HandlerSpec,
    InMemoryPayloadCustodian,
    InMemoryResultSink,
    decode_result_envelope,
)
from jarvis_v2.config import load_config
from jarvis_v2.memory.store import MemoryStore


MAX_REQUEST_BYTES = 1_048_576
MAX_RESPONSE_BYTES = 1_048_576
WIRE_VERSION = 1
RESULT_RETENTION_SWEEP_SECONDS = 0.1
SOCKET_LOCK_MARKER = "jarvis-v2/subagent-socket-lock/v1"
SOCKET_LOCK_SLOT_BYTES = 512
SOCKET_LOCK_SLOT_COUNT = 2


def _valid_text_metrics_payload(payload: Any) -> bool:
    return (
        type(payload) is dict
        and set(payload) == {"text"}
        and type(payload.get("text")) is str
        and len(payload["text"].encode("utf-8")) <= 524_288
    )


async def _text_metrics(payload: Any) -> dict[str, int]:
    text = payload["text"]
    return {
        "characters": len(text),
        "lines": len(text.splitlines()) or 1,
        "words": len(text.split()),
    }


def build_runner_handlers() -> HandlerRegistry:
    return HandlerRegistry(
        [
            HandlerSpec(
                "text_metrics",
                "jarvis.text_metrics",
                1,
                "text_metrics",
                30.0,
                _text_metrics,
                _valid_text_metrics_payload,
            )
        ]
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Jarvis's private local pure-computation subagent worker service."
    )
    parser.add_argument("--workers", type=int, default=10, help="Worker slots (1-10; default: 10).")
    parser.add_argument(
        "--socket",
        type=Path,
        default=None,
        help="Private Unix socket path (default: <Jarvis data dir>/subagent-runner.sock).",
    )
    return parser


def _socket_lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _decode_socket_lock_slot(
    raw: bytes,
) -> tuple[int, tuple[int, int] | None, str] | None:
    if len(raw) != SOCKET_LOCK_SLOT_BYTES:
        return None
    payload_size = struct.unpack(">I", raw[:4])[0]
    if payload_size <= 0 or payload_size > SOCKET_LOCK_SLOT_BYTES - 36:
        return None
    expected_digest = raw[4:36]
    encoded = raw[36 : 36 + payload_size]
    if not hashlib.sha256(encoded).digest() == expected_digest:
        return None
    try:
        payload = json.loads(encoded.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError):
        return None
    if (
        type(payload) is not dict
        or set(payload)
        != {
            "creation_token",
            "generation",
            "marker",
            "socket_device",
            "socket_inode",
            "version",
        }
        or payload.get("marker") != SOCKET_LOCK_MARKER
        or type(payload.get("version")) is not int
        or payload["version"] != 1
        or type(payload.get("generation")) is not int
        or payload["generation"] < 1
        or type(payload.get("creation_token")) is not str
        or re.fullmatch(r"[0-9a-f]{24}", payload["creation_token"]) is None
    ):
        return None
    device = payload.get("socket_device")
    inode = payload.get("socket_inode")
    if device is None and inode is None:
        return payload["generation"], None, payload["creation_token"]
    if (
        type(device) is not int
        or type(inode) is not int
        or device < 0
        or inode <= 0
    ):
        return None
    return payload["generation"], (device, inode), payload["creation_token"]


def _read_socket_lock(
    lock_fd: int,
) -> tuple[bool, tuple[int, int] | None, int, str]:
    valid: list[tuple[int, tuple[int, int] | None, str]] = []
    for slot in range(SOCKET_LOCK_SLOT_COUNT):
        raw = os.pread(
            lock_fd,
            SOCKET_LOCK_SLOT_BYTES,
            slot * SOCKET_LOCK_SLOT_BYTES,
        )
        decoded = _decode_socket_lock_slot(raw)
        if decoded is not None:
            valid.append(decoded)
    if not valid:
        return False, None, 0, ""
    generation, identity, creation_token = max(valid, key=lambda item: item[0])
    return True, identity, generation, creation_token


def _pwrite_all(fd: int, payload: bytes, offset: int) -> None:
    written = 0
    while written < len(payload):
        count = os.pwrite(fd, payload[written:], offset + written)
        if count <= 0:
            raise OSError("short socket lock journal write")
        written += count


def _write_socket_lock(
    lock_fd: int,
    identity: tuple[int, int] | None,
    *,
    creation_token: str | None = None,
) -> None:
    valid, _previous_identity, previous_generation, previous_token = _read_socket_lock(
        lock_fd
    )
    token = previous_token if valid else creation_token
    if token is None or re.fullmatch(r"[0-9a-f]{24}", token) is None:
        raise RuntimeError("subagent runner socket ownership token is invalid")
    generation = previous_generation + 1
    encoded = json.dumps(
        {
            "creation_token": token,
            "generation": generation,
            "marker": SOCKET_LOCK_MARKER,
            "socket_device": identity[0] if identity is not None else None,
            "socket_inode": identity[1] if identity is not None else None,
            "version": 1,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    if len(encoded) > SOCKET_LOCK_SLOT_BYTES - 36:
        raise RuntimeError("subagent runner socket ownership journal is too large")
    slot = (generation - 1) % SOCKET_LOCK_SLOT_COUNT
    framed = (
        struct.pack(">I", len(encoded))
        + hashlib.sha256(encoded).digest()
        + encoded
    ).ljust(SOCKET_LOCK_SLOT_BYTES, b"\0")
    _pwrite_all(lock_fd, framed, slot * SOCKET_LOCK_SLOT_BYTES)
    os.fsync(lock_fd)


def _record_socket_identity(lock_fd: int, identity: tuple[int, int]) -> None:
    _write_socket_lock(lock_fd, identity)


def _initialize_socket_lock_file(lock_path: Path) -> None:
    creation_token = secrets.token_hex(12)
    temp_path = lock_path.with_name(f".jvl-{creation_token}")
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    temp_fd = os.open(temp_path, flags, 0o600)
    linked = False
    try:
        os.fchmod(temp_fd, 0o600)
        _write_socket_lock(temp_fd, None, creation_token=creation_token)
        _fsync_directory(lock_path.parent)
        try:
            os.link(temp_path, lock_path, follow_symlinks=False)
            linked = True
            _fsync_directory(lock_path.parent)
        except FileExistsError:
            pass
    finally:
        os.close(temp_fd)
        try:
            temp_path.unlink()
            _fsync_directory(lock_path.parent)
        except FileNotFoundError:
            pass
    if not linked and not lock_path.exists() and not lock_path.is_symlink():
        raise RuntimeError("subagent runner socket ownership lock creation raced")


def _acquire_socket_ownership(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    lock_path = _socket_lock_path(path)
    if not lock_path.exists() and not lock_path.is_symlink():
        try:
            _initialize_socket_lock_file(lock_path)
        except OSError as exc:
            raise RuntimeError(
                "subagent runner socket ownership lock is unavailable"
            ) from exc
    try:
        lock_fd = os.open(lock_path, flags)
    except OSError as exc:
        raise RuntimeError("subagent runner socket ownership lock is unavailable") from exc
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            raise RuntimeError("subagent runner socket path is already occupied") from exc
        lock_stat = os.fstat(lock_fd)
        path_stat = lock_path.lstat()
        if (
            not stat.S_ISREG(lock_stat.st_mode)
            or (lock_stat.st_dev, lock_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
            or lock_stat.st_uid != os.geteuid()
            or stat.S_IMODE(lock_stat.st_mode) != 0o600
        ):
            raise RuntimeError("subagent runner socket ownership lock is invalid")
        valid_lock, _identity, _generation, creation_token = _read_socket_lock(
            lock_fd
        )
        if not valid_lock:
            raise RuntimeError("subagent runner socket ownership lock is invalid")
        if lock_stat.st_nlink == 2:
            creation_path = lock_path.with_name(f".jvl-{creation_token}")
            creation_stat = creation_path.lstat()
            if (
                (creation_stat.st_dev, creation_stat.st_ino)
                != (lock_stat.st_dev, lock_stat.st_ino)
                or not stat.S_ISREG(creation_stat.st_mode)
                or creation_stat.st_uid != os.geteuid()
                or stat.S_IMODE(creation_stat.st_mode) != 0o600
            ):
                raise RuntimeError("subagent runner socket ownership lock is invalid")
            creation_path.unlink()
            _fsync_directory(path.parent)
            lock_stat = os.fstat(lock_fd)
        if lock_stat.st_nlink != 1:
            raise RuntimeError("subagent runner socket ownership lock is invalid")
        if path.exists() or path.is_symlink():
            valid_lock, expected, _generation, _creation_token = _read_socket_lock(
                lock_fd
            )
            current = path.lstat()
            if (
                not valid_lock
                or expected is None
                or not stat.S_ISSOCK(current.st_mode)
                or (current.st_dev, current.st_ino) != expected
            ):
                raise RuntimeError("subagent runner socket path is already occupied")
            path.unlink()
            _fsync_directory(path.parent)
        _write_socket_lock(lock_fd, None)
        return lock_fd
    except BaseException:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(lock_fd)
        raise


def _release_socket_ownership(lock_fd: int, *, clear: bool) -> None:
    try:
        if clear:
            _write_socket_lock(lock_fd, None)
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


async def _wait_for_result(
    store: MemoryStore,
    sink: InMemoryResultSink,
    task_id: str,
    *,
    timeout_seconds: float = 35.0,
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    while True:
        response = await _result_status(store, sink, task_id)
        if response["state"] not in {"active", "result_publishing"}:
            return response
        if loop.time() >= deadline:
            return {
                "version": WIRE_VERSION,
                "ok": False,
                "task_id": task_id,
                "state": "timeout",
            }
        await asyncio.sleep(0.02)


def _base_response(task_id: str, *, ok: bool, state: str) -> dict[str, Any]:
    return {
        "version": WIRE_VERSION,
        "ok": ok,
        "task_id": task_id,
        "state": state,
    }


async def _mark_result_bytes_released(
    store: MemoryStore,
    sink: InMemoryResultSink,
    transition: Any,
) -> None:
    try:
        await sink.discard(transition.task_id, transition.result_digest)
    except BaseException:
        return
    try:
        store.mark_subagent_result_released(
            transition.task_id,
            transition.request_digest,
            transition.result_digest,
            transition.result_state,
        )
    except BaseException:
        pass


async def _mark_result_unavailable(
    store: MemoryStore,
    sink: InMemoryResultSink,
    status: Any,
) -> None:
    try:
        transition = store.mark_subagent_result_unavailable(
            status.task_id,
            status.request_digest,
            status.result_digest,
        )
    except BaseException:
        return
    if transition.status in {"UNAVAILABLE", "ALREADY_UNAVAILABLE"}:
        await _mark_result_bytes_released(store, sink, transition)


async def _result_status(
    store: MemoryStore,
    sink: InMemoryResultSink,
    task_id: str,
) -> dict[str, Any]:
    try:
        status = store.get_subagent_task_ephemeral_status(task_id)
    except (TypeError, ValueError):
        return _base_response(task_id, ok=False, state="invalid_task_id")
    if status.status == "NOT_FOUND":
        return _base_response(task_id, ok=False, state="not_found")
    if status.state in {"queued", "running"}:
        return _base_response(task_id, ok=False, state="active")
    if status.state in {"failed", "uncertain"}:
        response = _base_response(task_id, ok=False, state=status.state)
        response["error_code"] = status.error_code
        return response
    if status.state != "succeeded":
        return _base_response(task_id, ok=False, state="control_state_invalid")
    if status.result_state == "publishing":
        return _base_response(task_id, ok=False, state="result_publishing")
    if status.result_state == "acknowledged":
        return _base_response(task_id, ok=False, state="result_acknowledged")
    if status.result_state == "expired":
        return _base_response(task_id, ok=False, state="result_expired")
    if status.result_state == "restart_unavailable":
        return _base_response(task_id, ok=False, state="result_unavailable_after_restart")
    if status.result_state in {"none", "unavailable"}:
        return _base_response(task_id, ok=False, state="result_unavailable")
    if status.result_state != "available":
        return _base_response(task_id, ok=False, state="control_state_invalid")
    now_ms = int(time.time() * 1000)
    if status.result_expires_ms is None or status.result_expires_ms <= now_ms:
        expired = store.expire_due_subagent_results(checked_at_ms=now_ms, limit=100)
        for transition in expired:
            await _mark_result_bytes_released(store, sink, transition)
        return _base_response(task_id, ok=False, state="result_expired")
    record = await sink.get_record(task_id)
    if record is None:
        await _mark_result_unavailable(store, sink, status)
        return _base_response(task_id, ok=False, state="result_unavailable")
    try:
        header, result = decode_result_envelope(record.envelope)
    except ValueError:
        await _mark_result_unavailable(store, sink, status)
        return _base_response(task_id, ok=False, state="result_unavailable")
    valid = (
        record.task_id == status.task_id
        and record.request_digest == status.request_digest
        and record.kind == status.kind
        and record.result_type == status.result_type
        and record.result_digest == status.result_digest
        and header.get("task_id") == status.task_id
        and header.get("request_digest") == status.request_digest
        and header.get("kind") == status.kind
        and header.get("result_type") == status.result_type
    )
    if not valid:
        await _mark_result_unavailable(store, sink, status)
        return _base_response(task_id, ok=False, state="result_unavailable")
    response = _base_response(task_id, ok=True, state="result_available")
    response["result"] = result
    return response


async def _ack_result(
    store: MemoryStore,
    sink: InMemoryResultSink,
    task_id: str,
) -> dict[str, Any]:
    try:
        status = store.get_subagent_task_ephemeral_status(task_id)
    except (TypeError, ValueError):
        return _base_response(task_id, ok=False, state="invalid_task_id")
    if status.status == "NOT_FOUND":
        return _base_response(task_id, ok=False, state="not_found")
    if status.state != "succeeded" or not status.result_digest:
        return _base_response(task_id, ok=False, state="ack_refused")
    transition = store.acknowledge_subagent_result(
        status.task_id,
        status.request_digest,
        status.result_digest,
    )
    if transition.status in {
        "ACKNOWLEDGED",
        "ALREADY_ACKNOWLEDGED",
        "EXPIRED",
    }:
        await _mark_result_bytes_released(store, sink, transition)
    if transition.status in {"ACKNOWLEDGED", "ALREADY_ACKNOWLEDGED"}:
        return _base_response(task_id, ok=True, state="result_acknowledged")
    if transition.status == "EXPIRED":
        return _base_response(task_id, ok=False, state="result_expired")
    return _base_response(task_id, ok=False, state="ack_refused")


async def _retention_maintenance(
    store: MemoryStore,
    sink: InMemoryResultSink,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        expired = store.expire_due_subagent_results(limit=100)
        for transition in expired:
            await _mark_result_bytes_released(store, sink, transition)
        for transition in store.list_subagent_result_releases(limit=100):
            await _mark_result_bytes_released(store, sink, transition)
        try:
            await asyncio.wait_for(
                stop_event.wait(), timeout=RESULT_RETENTION_SWEEP_SECONDS
            )
        except asyncio.TimeoutError:
            pass


async def _send(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    if len(encoded) > MAX_RESPONSE_BYTES:
        encoded = b'{"ok":false,"state":"response_too_large","task_id":"","version":1}'
    writer.write(encoded + b"\n")
    await writer.drain()


async def _serve_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    runner: DurableSubagentRunner,
    store: MemoryStore,
    sink: InMemoryResultSink,
    stop_event: asyncio.Event,
) -> None:
    try:
        raw = await reader.readline()
        if not raw or len(raw) > MAX_REQUEST_BYTES or not raw.endswith(b"\n"):
            await _send(writer, _base_response("", ok=False, state="invalid_request"))
            return
        try:
            request = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            await _send(writer, _base_response("", ok=False, state="invalid_json"))
            return
        if type(request) is not dict:
            await _send(writer, _base_response("", ok=False, state="invalid_request"))
            return
        if type(request.get("version")) is not int or request["version"] != WIRE_VERSION:
            await _send(writer, _base_response("", ok=False, state="unsupported_version"))
            return
        operation = request.get("operation")
        if operation in {"status", "ack_result"}:
            task_id = request.get("task_id")
            if type(task_id) is not str:
                await _send(writer, _base_response("", ok=False, state="invalid_task_id"))
                return
            if operation == "status":
                await _send(writer, await _result_status(store, sink, task_id))
            else:
                await _send(writer, await _ack_result(store, sink, task_id))
            return
        if operation != "submit":
            await _send(writer, _base_response("", ok=False, state="unsupported_operation"))
            return
        if stop_event.is_set():
            await _send(writer, _base_response("", ok=False, state="service_stopping"))
            return
        if request.get("kind") != "text_metrics" or not _valid_text_metrics_payload(request.get("payload")):
            await _send(writer, _base_response("", ok=False, state="invalid_payload"))
            return
        try:
            queued = await runner.submit(
                kind="text_metrics",
                payload=request["payload"],
                priority=0,
                max_attempts=3,
            )
        except (ValueError, RuntimeError):
            await _send(writer, _base_response("", ok=False, state="enqueue_failed"))
            return
        if queued.status not in {"ENQUEUED", "EXISTING"}:
            await _send(
                writer,
                _base_response(
                    queued.task_id, ok=False, state=queued.status.lower()
                ),
            )
            return
        await _send(writer, await _wait_for_result(store, sink, queued.task_id))
    finally:
        writer.close()
        await writer.wait_closed()


async def _run_owned_service(
    *,
    workers: int,
    socket_path: Path,
    stop_event: asyncio.Event,
    socket_lock_fd: int,
) -> None:
    config = load_config()
    store = MemoryStore(config.db_path)
    store.init()
    custody = InMemoryPayloadCustodian("ephemeral")
    sink = InMemoryResultSink()
    runner = DurableSubagentRunner(
        store=store,
        custodian=custody,
        result_sink=sink,
        handlers=build_runner_handlers(),
        worker_count=workers,
    )
    client_tasks: set[asyncio.Task[None]] = set()

    def accept_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.create_task(
            _serve_client(reader, writer, runner, store, sink, stop_event)
        )
        client_tasks.add(task)
        task.add_done_callback(client_tasks.discard)

    bind_path = socket_path.parent / f".jv-{secrets.token_hex(4)}"
    previous_umask = os.umask(0o077)
    try:
        server = await asyncio.start_unix_server(
            accept_client,
            path=str(bind_path),
            limit=MAX_REQUEST_BYTES + 1,
        )
    finally:
        os.umask(previous_umask)
    try:
        bind_stat = bind_path.lstat()
        if not stat.S_ISSOCK(bind_stat.st_mode):
            raise RuntimeError("subagent runner private bind is not a socket")
        published_socket = (bind_stat.st_dev, bind_stat.st_ino)
        _record_socket_identity(socket_lock_fd, published_socket)
        os.link(bind_path, socket_path, follow_symlinks=False)
        socket_stat = socket_path.lstat()
        if (
            not stat.S_ISSOCK(socket_stat.st_mode)
            or (socket_stat.st_dev, socket_stat.st_ino)
            != (bind_stat.st_dev, bind_stat.st_ino)
        ):
            raise RuntimeError("subagent runner socket publication changed during startup")
        bind_path.unlink()
    except BaseException:
        server.close()
        await asyncio.wait_for(server.wait_closed(), timeout=2.0)
        try:
            current_socket = socket_path.lstat()
            if (current_socket.st_dev, current_socket.st_ino) == (
                bind_stat.st_dev,
                bind_stat.st_ino,
            ):
                socket_path.unlink()
        except (FileNotFoundError, UnboundLocalError):
            pass
        try:
            current_bind = bind_path.lstat()
            if (current_bind.st_dev, current_bind.st_ino) == (
                bind_stat.st_dev,
                bind_stat.st_ino,
            ):
                bind_path.unlink()
        except (FileNotFoundError, UnboundLocalError):
            pass
        raise
    owned_socket: tuple[int, int] | None = published_socket
    store.reconcile_subagent_results_after_ephemeral_restart()
    runner_task: asyncio.Task[None] | None = None
    retention_task: asyncio.Task[None] | None = None
    stop_wait: asyncio.Task[bool] | None = None
    primary_error: BaseException | None = None
    runner_failure: BaseException | None = None
    cleanup_error: BaseException | None = None

    def observe_task(task: asyncio.Task[Any]) -> BaseException | None:
        if not task.done():
            return None
        try:
            task.result()
        except BaseException as exc:
            return exc
        return None

    try:
        runner_task = asyncio.create_task(runner.serve(stop_event))
        retention_task = asyncio.create_task(
            _retention_maintenance(store, sink, stop_event)
        )
        stop_wait = asyncio.create_task(stop_event.wait())
        await asyncio.wait(
            {runner_task, retention_task, stop_wait},
            return_when=asyncio.FIRST_COMPLETED,
        )
    except BaseException as exc:
        primary_error = exc
    finally:
        try:
            server.close()
        except BaseException as exc:
            cleanup_error = exc
        stop_event.set()

        if stop_wait is not None:
            stop_wait.cancel()
            try:
                await asyncio.gather(stop_wait, return_exceptions=True)
            except BaseException as exc:
                cleanup_error = cleanup_error or exc

        if runner_task is not None:
            cancelled_for_cleanup = False
            try:
                _done, pending = await asyncio.wait({runner_task}, timeout=2.0)
                if pending:
                    cancelled_for_cleanup = runner_task.cancel()
                await asyncio.gather(runner_task, return_exceptions=True)
                observed = observe_task(runner_task)
                if observed is not None and not (
                    cancelled_for_cleanup and isinstance(observed, asyncio.CancelledError)
                ):
                    runner_failure = observed
            except BaseException as exc:
                cleanup_error = cleanup_error or exc

        if retention_task is not None:
            cancelled_for_cleanup = False
            try:
                _done, pending = await asyncio.wait({retention_task}, timeout=2.0)
                if pending:
                    cancelled_for_cleanup = retention_task.cancel()
                await asyncio.gather(retention_task, return_exceptions=True)
                observed = observe_task(retention_task)
                if observed is not None and not (
                    cancelled_for_cleanup
                    and isinstance(observed, asyncio.CancelledError)
                ):
                    runner_failure = runner_failure or observed
            except BaseException as exc:
                cleanup_error = cleanup_error or exc

        tasks = set(client_tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
                for task in tasks:
                    observe_task(task)
            except BaseException as exc:
                cleanup_error = cleanup_error or exc

        try:
            await asyncio.wait_for(server.wait_closed(), timeout=2.0)
        except BaseException as exc:
            cleanup_error = cleanup_error or exc
        finally:
            if owned_socket is not None:
                try:
                    socket_stat = socket_path.lstat()
                    if (
                        stat.S_ISSOCK(socket_stat.st_mode)
                        and (socket_stat.st_dev, socket_stat.st_ino) == owned_socket
                    ):
                        socket_path.unlink()
                except FileNotFoundError:
                    pass
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc

    if runner_failure is not None:
        raise runner_failure
    if primary_error is not None:
        raise primary_error
    if cleanup_error is not None:
        raise cleanup_error


async def run_service(*, workers: int, socket_path: Path, stop_event: asyncio.Event) -> None:
    socket_lock_fd = _acquire_socket_ownership(socket_path)
    try:
        await _run_owned_service(
            workers=workers,
            socket_path=socket_path,
            stop_event=stop_event,
            socket_lock_fd=socket_lock_fd,
        )
    finally:
        clear = not socket_path.exists() and not socket_path.is_symlink()
        if clear:
            _fsync_directory(socket_path.parent)
        _release_socket_ownership(socket_lock_fd, clear=clear)


def main() -> None:
    args = _parser().parse_args()
    if not 1 <= args.workers <= 10:
        raise SystemExit("--workers must be between 1 and 10")
    config = load_config()
    socket_path = args.socket or (config.data_dir / "subagent-runner.sock")
    stop_event = asyncio.Event()

    async def run() -> None:
        loop = asyncio.get_running_loop()
        for name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    loop.add_signal_handler(sig, stop_event.set)
                except (NotImplementedError, RuntimeError):
                    pass
        print("Jarvis V2 private subagent runner active. Press Ctrl+C to stop.", flush=True)
        await run_service(workers=args.workers, socket_path=socket_path, stop_event=stop_event)
        print("Jarvis V2 private subagent runner stopped.", flush=True)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
