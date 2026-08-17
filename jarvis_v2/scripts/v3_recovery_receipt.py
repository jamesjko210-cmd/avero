"""Content-free receipts for supervised V3 recovery observations.

This module never changes network, service, scheduler, or reboot state.  The
public API and CLI accept only paths and the requested proof phase; every fact
that can prove a transition is derived through bounded, read-only observers.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import plistlib
import pwd
import re
import selectors
import secrets
import shlex
import signal
import stat
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from jarvis_v2.scripts import v3_active_launchagent_contracts as active_contracts


RECEIPT_VERSION = 3
MAX_BOOT_IDENTITY_BYTES = 512
MAX_MANIFEST_BYTES = 64 * 1024
MAX_CONTRACT_BYTES = 128 * 1024
MAX_ENV_BYTES = 256 * 1024
MAX_STATE_BYTES = 64 * 1024
MAX_OBSERVER_OUTPUT_BYTES = 64 * 1024
OBSERVER_TERM_GRACE_SECONDS = 0.25
OBSERVER_KILL_GRACE_SECONDS = 1.0
MAX_FUTURE_SKEW_SECONDS = 300
ACTIVE_CONTRACT_DIGEST_FIELD = "contract_set_sha256"
NETWORK_PROBE_TARGET = "1.1.1.1"
SCHEDULER_LABEL = "com.jarvis-v3.scheduler"
SOURCE_COMMIT_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
SERVICE_NAME_RE = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
UTC_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
BOOT_IDENTITY_RE = re.compile(r"\{\s*sec\s*=\s*[0-9]+,\s*usec\s*=\s*[0-9]+\s*\}")
PYTHON_EXECUTABLE = "/opt/homebrew/bin/python3"
ALLOWED_LOADED_ENVIRONMENT_EXTRAS = frozenset({"XPC_SERVICE_NAME"})
STATE_KEYS = {
    "version", "receipt_id", "receipt_contract_manifest_digest",
    "active_contract_manifest_digest", "active_contract_set_digest",
    "source_commit", "env_content_digest", "kind", "state", "created_at", "finalized_at",
    "boot_hash_salt", "scheduler_disabled", "expected_services",
    "observations", "boundary", "verdict", "integrity_digest",
}
OBSERVATION_KEYS = {
    "ordinal", "phase", "observed_at", "boot_session_hash", "network_reachable",
    "scheduler_disabled", "service_states",
}
CONTRACT_MANIFEST = {
    "name": "jarvis-v3-supervised-recovery-proof",
    "version": RECEIPT_VERSION,
    "active_contract_digest_field": ACTIVE_CONTRACT_DIGEST_FIELD,
    "proof_kinds": {
        "network": ["prepared", "network-down", "network-up"],
        "reboot": ["preboot", "postboot"],
    },
    "required_invariants": [
        "clean-source-head-stable",
        "manifest-and-contract-set-stable",
        "selected-environment-content-stable",
        "exact-installed-contracts-running",
        "scheduler-disabled",
        "expected-service-processes-present",
        "known-network-state",
        "os-derived-boot-identity",
        "ordered-observations",
        "single-finalization",
    ],
    "boundaries": {
        "changes_network_state": False,
        "reboots_computer": False,
        "starts_or_restarts_services": False,
        "sends_messages": False,
        "approves_actions": False,
        "reads_private_payloads": False,
    },
}


class RecoveryReceiptError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ContractEvidence:
    manifest_digest: str
    contract_set_digest: str
    expected_services: tuple[str, ...]
    artifacts: tuple[tuple[str, bytes], ...]
    env_content_digest: str


@dataclass(frozen=True)
class SystemObservation:
    source_commit: str
    manifest_digest: str
    contract_set_digest: str
    env_content_digest: str
    expected_services: tuple[str, ...]
    boot_identity: str
    network_reachable: bool
    scheduler_disabled: bool
    service_states: dict[str, bool]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _canonical_json_document(value: object) -> bytes:
    return _canonical_json(value) + b"\n"


def _load_canonical_json(raw: bytes, code: str) -> object:
    duplicate = False

    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        nonlocal duplicate
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                duplicate = True
            result[key] = value
        return result

    def reject_constant(_value: str) -> object:
        raise ValueError("non-finite JSON constant")

    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError, ValueError, TypeError) as exc:
        raise RecoveryReceiptError(code) from exc
    if duplicate or raw != _canonical_json_document(value):
        raise RecoveryReceiptError(code)
    return value


def contract_manifest_digest() -> str:
    return hashlib.sha256(_canonical_json(CONTRACT_MANIFEST)).hexdigest()


def _utc_now() -> str:
    return _utc_datetime().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _utc_datetime() -> datetime:
    return datetime.now(timezone.utc)


def _validate_source_commit(value: object) -> str:
    if type(value) is not str or SOURCE_COMMIT_RE.fullmatch(value) is None:
        raise RecoveryReceiptError("invalid_source_commit")
    return value


def _validate_digest(value: object, code: str) -> str:
    if type(value) is not str or DIGEST_RE.fullmatch(value) is None:
        raise RecoveryReceiptError(code)
    return value


def _parse_utc_timestamp(value: object) -> datetime:
    if type(value) is not str or UTC_TIMESTAMP_RE.fullmatch(value) is None:
        raise RecoveryReceiptError("state_corrupt")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise RecoveryReceiptError("state_corrupt") from exc


def active_source_commit(source_root: Path | None = None) -> str:
    root = source_root or active_contracts.PROJECT_ROOT
    head = _run_observer(
        ["git", "-C", os.fspath(root), "rev-parse", "HEAD"],
        "source_state_unavailable",
        timeout=10,
        max_output_bytes=4096,
    )
    status = _run_observer(
        ["git", "-C", os.fspath(root), "status", "--porcelain", "--untracked-files=normal"],
        "source_state_unavailable",
        timeout=10,
        max_output_bytes=MAX_OBSERVER_OUTPUT_BYTES,
    )
    if head.returncode != 0 or status.returncode != 0 or head.stderr or status.stderr:
        raise RecoveryReceiptError("source_state_unavailable")
    if status.stdout:
        raise RecoveryReceiptError("source_worktree_not_clean")
    return _validate_source_commit(head.stdout.strip())


def _open_owner_directory(path: Path, code: str) -> int:
    if not path.is_absolute():
        raise RecoveryReceiptError(code)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open("/", flags)
        for component in path.parts[1:]:
            if component in {"", ".", ".."}:
                raise OSError("invalid component")
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        info = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o077
        ):
            raise RecoveryReceiptError(code)
        return descriptor
    except RecoveryReceiptError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise RecoveryReceiptError(code) from exc


def _validate_state_path(value: Path | str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.name or len(os.fsencode(path)) > 2048:
        raise RecoveryReceiptError("invalid_state_path")
    parent_fd = _open_owner_directory(path.parent, "unsafe_state_parent")
    os.close(parent_fd)
    return path


def _normalize_services(values: Sequence[str]) -> tuple[str, ...]:
    expected = tuple(sorted(contract.name for contract in active_contracts.SERVICE_CONTRACTS))
    normalized = tuple(sorted(values))
    if normalized != expected or any(SERVICE_NAME_RE.fullmatch(item) is None for item in normalized):
        raise RecoveryReceiptError("service_evidence_mismatch")
    return normalized


def _normalize_service_states(
    expected_services: Sequence[str], service_states: dict[str, bool]
) -> dict[str, bool]:
    if type(service_states) is not dict or set(service_states) != set(expected_services):
        raise RecoveryReceiptError("service_evidence_mismatch")
    if any(type(value) is not bool for value in service_states.values()):
        raise RecoveryReceiptError("service_outcome_unknown")
    if not all(service_states.values()):
        raise RecoveryReceiptError("expected_service_process_not_present")
    return {name: service_states[name] for name in sorted(service_states)}


def _hash_boot_identity(raw_identity: str, salt: str) -> str:
    if type(raw_identity) is not str:
        raise RecoveryReceiptError("boot_identity_unknown")
    encoded = raw_identity.encode("utf-8", errors="strict")
    if not encoded or len(encoded) > MAX_BOOT_IDENTITY_BYTES or "\x00" in raw_identity:
        raise RecoveryReceiptError("invalid_boot_identity")
    return hashlib.sha256(
        b"jarvis-v3-boot-session-v2\0" + salt.encode("ascii") + b"\0" + encoded
    ).hexdigest()


def _open_owner_regular(
    path: Path, *, max_bytes: int, code: str, private_parent: bool = True,
    allow_empty: bool = False,
) -> tuple[int, os.stat_result]:
    if not path.is_absolute() or not path.name:
        raise RecoveryReceiptError(code)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd: int | None = None
    try:
        parent_fd = os.open("/", directory_flags)
        for component in path.parent.parts[1:]:
            if component in {"", ".", ".."}:
                raise OSError("invalid component")
            next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        parent_info = os.fstat(parent_fd)
        if (
            parent_info.st_uid != os.geteuid()
            or not stat.S_ISDIR(parent_info.st_mode)
            or (private_parent and parent_info.st_mode & 0o077)
        ):
            raise RecoveryReceiptError(f"unsafe_{code}_parent")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path.name, flags, dir_fd=parent_fd)
        info = os.fstat(fd)
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or (info.st_dev, info.st_ino) != (named.st_dev, named.st_ino)
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o077
            or info.st_nlink != 1
            or (not allow_empty and info.st_size < 1)
            or info.st_size > max_bytes
        ):
            os.close(fd)
            raise RecoveryReceiptError(f"unsafe_{code}")
        return fd, info
    except RecoveryReceiptError:
        raise
    except OSError as exc:
        raise RecoveryReceiptError(f"{code}_unavailable") from exc
    finally:
        if parent_fd is not None:
            os.close(parent_fd)


def _read_bounded_fd(fd: int, limit: int, code: str) -> bytes:
    try:
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        value = b"".join(chunks)
    except OSError as exc:
        raise RecoveryReceiptError(f"{code}_unavailable") from exc
    if len(value) > limit:
        raise RecoveryReceiptError(f"{code}_too_large")
    return value


def _read_stable_owner_regular(
    path: Path, *, max_bytes: int, code: str, private_parent: bool = True,
    allow_empty: bool = False,
) -> bytes:
    fd, before = _open_owner_regular(
        path, max_bytes=max_bytes, code=code, private_parent=private_parent,
        allow_empty=allow_empty,
    )
    try:
        payload = _read_bounded_fd(fd, max_bytes, code)
        after = os.fstat(fd)
        named_after = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise RecoveryReceiptError(f"{code}_unstable") from exc
    finally:
        os.close(fd)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or (after.st_dev, after.st_ino) != (named_after.st_dev, named_after.st_ino)
        or stat.S_ISLNK(named_after.st_mode)
    ):
        raise RecoveryReceiptError(f"{code}_unstable")
    return payload


def _validate_active_payload(payload: bytes, contract: active_contracts.ServiceContract) -> str:
    try:
        value = plistlib.loads(payload)
    except Exception as exc:
        raise RecoveryReceiptError("active_contract_invalid") from exc
    if type(value) is not dict:
        raise RecoveryReceiptError("active_contract_invalid")
    environment = value.get("EnvironmentVariables")
    if type(environment) is not dict:
        raise RecoveryReceiptError("active_contract_invalid")
    env_path = environment.get(active_contracts.ENV_FILE_KEY)
    if type(env_path) is not str or not Path(env_path).is_absolute():
        raise RecoveryReceiptError("active_contract_invalid")
    if environment.get(active_contracts.SCHEDULER_ENABLE_KEY) != "0":
        raise RecoveryReceiptError("active_contract_invalid")
    try:
        active_contracts._validate_env_file(Path(env_path))
        inert_template = active_contracts._load_valid_template(
            active_contracts.PROJECT_ROOT, contract
        )
    except active_contracts.ContractBuildError as exc:
        raise RecoveryReceiptError("active_contract_provenance_invalid") from exc
    expected = active_contracts._active_payload(inert_template, env_path)
    expected_bytes = plistlib.dumps(expected, fmt=plistlib.FMT_XML, sort_keys=True)
    if value != expected or not secrets.compare_digest(payload, expected_bytes):
        raise RecoveryReceiptError("active_contract_invalid")
    return env_path


def validate_active_contract_manifest(manifest_path: Path | str) -> ContractEvidence:
    path = Path(manifest_path)
    try:
        raw = _read_stable_owner_regular(
            path, max_bytes=MAX_MANIFEST_BYTES, code="manifest"
        )
        manifest = _load_canonical_json(raw, "manifest_invalid")
    except RecoveryReceiptError:
        raise
    expected_keys = {
        "schema_version", "kind", "offline_only", "scheduler_activation_present",
        "artifact_count", "contract_set_sha256", "log_contract", "artifacts",
    }
    if type(manifest) is not dict or set(manifest) != expected_keys:
        raise RecoveryReceiptError("manifest_invalid")
    contracts_by_name = {contract.filename: contract for contract in active_contracts.SERVICE_CONTRACTS}
    artifacts = manifest.get("artifacts")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("kind") != "jarvis_v3_offline_active_launchagent_contracts"
        or manifest.get("offline_only") is not True
        or manifest.get("scheduler_activation_present") is not False
        or manifest.get("log_contract") != active_contracts._log_contract_manifest()
        or manifest.get("artifact_count") != len(contracts_by_name)
        or type(artifacts) is not list
        or len(artifacts) != len(contracts_by_name)
    ):
        raise RecoveryReceiptError("manifest_invalid")
    expected_artifact_order = [
        contract.filename for contract in active_contracts.SERVICE_CONTRACTS
    ]
    if [item.get("name") if type(item) is dict else None for item in artifacts] != expected_artifact_order:
        raise RecoveryReceiptError("manifest_invalid")
    expected_env: str | None = None
    receipts: list[dict[str, str]] = []
    payloads: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    for artifact in artifacts:
        if type(artifact) is not dict or set(artifact) != {"name", "sha256"}:
            raise RecoveryReceiptError("manifest_invalid")
        name = artifact.get("name")
        digest = _validate_digest(artifact.get("sha256"), "manifest_invalid")
        if name not in contracts_by_name or name in seen:
            raise RecoveryReceiptError("manifest_invalid")
        seen.add(name)
        artifact_path = path.parent / name
        payload = _read_stable_owner_regular(
            artifact_path, max_bytes=MAX_CONTRACT_BYTES, code="artifact"
        )
        if not secrets.compare_digest(hashlib.sha256(payload).hexdigest(), digest):
            raise RecoveryReceiptError("artifact_digest_mismatch")
        selected_env = _validate_active_payload(payload, contracts_by_name[name])
        if expected_env is None:
            expected_env = selected_env
        elif selected_env != expected_env:
            raise RecoveryReceiptError("active_contract_env_mismatch")
        receipts.append({"name": name, "sha256": digest})
        payloads.append((name, payload))
    if set(seen) != set(contracts_by_name):
        raise RecoveryReceiptError("manifest_invalid")
    calculated_set = hashlib.sha256(_canonical_json(receipts)).hexdigest()
    declared_set = _validate_digest(manifest.get("contract_set_sha256"), "manifest_invalid")
    if not secrets.compare_digest(calculated_set, declared_set):
        raise RecoveryReceiptError("contract_set_digest_mismatch")
    assert expected_env is not None
    env_payload = _read_stable_owner_regular(
        Path(expected_env),
        max_bytes=MAX_ENV_BYTES,
        code="active_contract_env",
        private_parent=False,
        allow_empty=True,
    )
    env_content_digest = hashlib.sha256(
        b"jarvis-v3-selected-env-v1\0" + env_payload
    ).hexdigest()
    return ContractEvidence(
        manifest_digest=hashlib.sha256(raw).hexdigest(),
        contract_set_digest=calculated_set,
        expected_services=tuple(sorted(contract.name for contract in active_contracts.SERVICE_CONTRACTS)),
        artifacts=tuple(payloads),
        env_content_digest=env_content_digest,
    )


def _installed_launchagent_root() -> Path:
    try:
        account = pwd.getpwuid(os.geteuid())
    except (KeyError, OSError) as exc:
        raise RecoveryReceiptError("installed_contract_root_invalid") from exc
    raw_home = getattr(account, "pw_dir", None)
    if type(raw_home) is not str:
        raise RecoveryReceiptError("installed_contract_root_invalid")
    home = Path(raw_home)
    if not home.is_absolute() or not home.name:
        raise RecoveryReceiptError("installed_contract_root_invalid")
    try:
        info = os.stat(home, follow_symlinks=False)
    except (OSError, ValueError) as exc:
        raise RecoveryReceiptError("installed_contract_root_invalid") from exc
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.geteuid()
    ):
        raise RecoveryReceiptError("installed_contract_root_invalid")
    return home / "Library" / "LaunchAgents"


def _verify_installed_contracts(evidence: ContractEvidence) -> None:
    root = _installed_launchagent_root()
    for name, expected in evidence.artifacts:
        installed = _read_stable_owner_regular(
            root / name,
            max_bytes=MAX_CONTRACT_BYTES,
            code="installed_contract",
            private_parent=False,
        )
        if not secrets.compare_digest(installed, expected):
            raise RecoveryReceiptError("installed_contract_mismatch")


def _verify_log_custody() -> None:
    """Verify only metadata for the precreated owner-only daemon log targets."""

    try:
        root = active_contracts._account_log_directory()
    except active_contracts.ContractBuildError as exc:
        raise RecoveryReceiptError("log_custody_invalid") from exc
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd: int | None = None
    try:
        parent_fd = os.open("/", directory_flags)
        for component in root.parts[1:]:
            if component in {"", ".", ".."}:
                raise OSError("invalid log directory component")
            next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        directory_info = os.fstat(parent_fd)
        named_directory = os.stat(root, follow_symlinks=False)
        if (
            not stat.S_ISDIR(directory_info.st_mode)
            or stat.S_ISLNK(named_directory.st_mode)
            or directory_info.st_uid != os.geteuid()
            or stat.S_IMODE(directory_info.st_mode) != active_contracts.LOG_DIRECTORY_MODE
            or (directory_info.st_dev, directory_info.st_ino)
            != (named_directory.st_dev, named_directory.st_ino)
        ):
            raise OSError("unsafe log directory")
        file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        for contract in active_contracts.SERVICE_CONTRACTS:
            descriptor = os.open(contract.log_filename, file_flags, dir_fd=parent_fd)
            try:
                opened = os.fstat(descriptor)
                named = os.stat(
                    contract.log_filename,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or stat.S_ISLNK(named.st_mode)
                    or opened.st_uid != os.geteuid()
                    or stat.S_IMODE(opened.st_mode) != active_contracts.LOG_FILE_MODE
                    or opened.st_nlink != 1
                    or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
                ):
                    raise OSError("unsafe log file")
            finally:
                os.close(descriptor)
        named_directory_after = os.stat(root, follow_symlinks=False)
        if (
            stat.S_ISLNK(named_directory_after.st_mode)
            or (directory_info.st_dev, directory_info.st_ino)
            != (named_directory_after.st_dev, named_directory_after.st_ino)
        ):
            raise OSError("unstable log directory")
    except (OSError, ValueError) as exc:
        raise RecoveryReceiptError("log_custody_invalid") from exc
    finally:
        if parent_fd is not None:
            os.close(parent_fd)


def _observer_group_alive(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_observer_group_exit(
    process: subprocess.Popen[bytes], process_group_id: int, deadline: float
) -> bool:
    while True:
        process.poll()
        if not _observer_group_alive(process_group_id):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)


def _terminate_observer_group(process: subprocess.Popen[bytes]) -> None:
    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    term_deadline = time.monotonic() + OBSERVER_TERM_GRACE_SECONDS
    if not _wait_observer_group_exit(process, process_group_id, term_deadline):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        _wait_observer_group_exit(
            process,
            process_group_id,
            time.monotonic() + OBSERVER_KILL_GRACE_SECONDS,
        )
    try:
        process.wait(timeout=OBSERVER_KILL_GRACE_SECONDS)
    except (OSError, subprocess.SubprocessError):
        pass


def _run_observer(
    command: list[str],
    code: str,
    timeout: float = 5.0,
    max_output_bytes: int = MAX_OBSERVER_OUTPUT_BYTES,
) -> subprocess.CompletedProcess[str]:
    process: subprocess.Popen[bytes] | None = None
    selector: selectors.BaseSelector | None = None
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        assert process.stdout is not None and process.stderr is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        chunks: dict[str, list[bytes]] = {"stdout": [], "stderr": []}
        total = 0
        deadline = time.monotonic() + timeout
        while selector.get_map():
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                raise RecoveryReceiptError(code)
            events = selector.select(remaining_time)
            if not events:
                raise RecoveryReceiptError(code)
            for key, _mask in events:
                chunk = os.read(key.fileobj.fileno(), min(8192, max_output_bytes - total + 1))
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                total += len(chunk)
                if total > max_output_bytes:
                    raise RecoveryReceiptError(code)
                chunks[key.data].append(chunk)
        remaining_time = deadline - time.monotonic()
        if remaining_time <= 0:
            raise RecoveryReceiptError(code)
        returncode = process.wait(timeout=remaining_time)
        if _observer_group_alive(process.pid):
            raise RecoveryReceiptError(code)
        stdout = b"".join(chunks["stdout"]).decode("utf-8", errors="strict")
        stderr = b"".join(chunks["stderr"]).decode("utf-8", errors="strict")
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)
    except RecoveryReceiptError:
        if process is not None:
            _terminate_observer_group(process)
        raise
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        if process is not None:
            _terminate_observer_group(process)
        raise RecoveryReceiptError(code) from exc
    except BaseException:
        # The observer owns a fresh process group.  Cleanup must also run for
        # unexpected selector/runtime failures and cancellation; otherwise a
        # child or descendant could outlive an aborted proof attempt.
        if process is not None:
            _terminate_observer_group(process)
        raise
    finally:
        if selector is not None:
            selector.close()
        if process is not None:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()


def observe_boot_identity() -> str:
    result = _run_observer(
        ["/usr/sbin/sysctl", "-n", "kern.boottime"],
        "boot_identity_unknown",
        max_output_bytes=1024,
    )
    identity = result.stdout.strip()
    if (
        result.returncode != 0
        or result.stderr
        or len(identity.encode("utf-8")) > MAX_BOOT_IDENTITY_BYTES
        or BOOT_IDENTITY_RE.fullmatch(identity) is None
    ):
        raise RecoveryReceiptError("boot_identity_unknown")
    return identity


def observe_network_reachability() -> bool:
    result = _run_observer(
        ["/usr/sbin/scutil", "-r", NETWORK_PROBE_TARGET], "network_outcome_unknown", timeout=4.0
    )
    outcome = result.stdout.strip()
    if result.returncode != 0 or result.stderr:
        raise RecoveryReceiptError("network_outcome_unknown")
    if outcome == "Reachable":
        return True
    if outcome == "Not Reachable":
        return False
    raise RecoveryReceiptError("network_outcome_unknown")


def _launchctl_target(label: str) -> str:
    return f"gui/{os.getuid()}/{label}"


def _running_service_pid(result: subprocess.CompletedProcess[str]) -> int:
    if result.returncode != 0 or result.stderr:
        raise RecoveryReceiptError("service_outcome_unknown")
    states = re.findall(r"(?m)^\s*state\s*=\s*([^\s]+)\s*$", result.stdout)
    pids = re.findall(r"(?m)^\s*pid\s*=\s*([0-9]+)\s*$", result.stdout)
    if len(states) != 1 or states[0] != "running" or len(pids) != 1:
        raise RecoveryReceiptError("service_outcome_unknown")
    pid = int(pids[0])
    if pid <= 0:
        raise RecoveryReceiptError("service_outcome_unknown")
    return pid


def _launchctl_flat_block(output: str, name: str) -> tuple[str, ...] | None:
    """Extract one flat launchctl-print block, rejecting ambiguous structure."""

    lines = output.splitlines()
    starts = [
        index
        for index, line in enumerate(lines)
        if line.strip() == f"{name} = {{"
    ]
    if len(starts) != 1:
        return None
    body: list[str] = []
    for line in lines[starts[0] + 1 :]:
        stripped = line.strip()
        if stripped == "}":
            return tuple(body)
        if stripped.endswith("{") or stripped.endswith("}"):
            return None
        if stripped:
            body.append(stripped)
    return None


def _loaded_service_contract(
    result: subprocess.CompletedProcess[str],
) -> tuple[int, tuple[str, tuple[str, ...], str, tuple[tuple[str, str], ...]]]:
    pid = _running_service_pid(result)
    arguments_block = _launchctl_flat_block(result.stdout, "arguments")
    environment_block = _launchctl_flat_block(result.stdout, "environment")
    program_matches = re.findall(
        r"(?m)^\s*program\s*=\s*([^\r\n]+?)\s*$",
        result.stdout,
    )
    working_matches = re.findall(
        r"(?m)^\s*working directory\s*=\s*([^\r\n]+?)\s*$",
        result.stdout,
    )
    if (
        arguments_block is None
        or not arguments_block
        or any("=>" in item for item in arguments_block)
        or environment_block is None
        or len(program_matches) != 1
        or len(working_matches) != 1
    ):
        raise RecoveryReceiptError("service_contract_mismatch")

    environment: dict[str, str] = {}
    for item in environment_block:
        key, separator, value = item.partition("=>")
        key = key.strip()
        value = value.strip()
        if not separator or not key or not value or key in environment:
            raise RecoveryReceiptError("service_contract_mismatch")
        environment[key] = value
    required_keys = (
        active_contracts.DAEMON_ENABLE_KEY,
        active_contracts.ENV_FILE_KEY,
        active_contracts.SCHEDULER_ENABLE_KEY,
    )
    if any(key not in environment for key in required_keys):
        raise RecoveryReceiptError("service_contract_mismatch")
    return pid, (
        program_matches[0],
        tuple(arguments_block),
        working_matches[0],
        tuple(sorted(environment.items())),
    )


def _expected_loaded_service_contracts(
    evidence: ContractEvidence,
) -> dict[str, tuple[str, tuple[str, ...], str, tuple[tuple[str, str], ...]]]:
    expected_filenames = {contract.filename for contract in active_contracts.SERVICE_CONTRACTS}
    artifact_names = [name for name, _payload in evidence.artifacts]
    if (
        len(artifact_names) != len(expected_filenames)
        or set(artifact_names) != expected_filenames
        or evidence.expected_services
        != tuple(sorted(contract.name for contract in active_contracts.SERVICE_CONTRACTS))
    ):
        raise RecoveryReceiptError("service_contract_evidence_mismatch")
    artifacts = dict(evidence.artifacts)
    expectations: dict[
        str,
        tuple[str, tuple[str, ...], str, tuple[tuple[str, str], ...]],
    ] = {}
    for contract in active_contracts.SERVICE_CONTRACTS:
        payload = artifacts.get(contract.filename)
        try:
            value = plistlib.loads(payload) if type(payload) is bytes else None
        except Exception as exc:
            raise RecoveryReceiptError("service_contract_evidence_mismatch") from exc
        if type(value) is not dict:
            raise RecoveryReceiptError("service_contract_evidence_mismatch")
        arguments = value.get("ProgramArguments")
        working_directory = value.get("WorkingDirectory")
        environment = value.get("EnvironmentVariables")
        if (
            arguments != [PYTHON_EXECUTABLE, "-m", contract.module]
            or type(working_directory) is not str
            or not working_directory
            or type(environment) is not dict
            or environment.get(active_contracts.DAEMON_ENABLE_KEY) != "1"
            or type(environment.get(active_contracts.ENV_FILE_KEY)) is not str
            or not Path(environment[active_contracts.ENV_FILE_KEY]).is_absolute()
            or environment.get(active_contracts.SCHEDULER_ENABLE_KEY) != "0"
            or any(
                type(key) is not str
                or not key
                or type(item) is not str
                or not item
                for key, item in environment.items()
            )
        ):
            raise RecoveryReceiptError("service_contract_evidence_mismatch")
        expectations[contract.name] = (
            arguments[0],
            tuple(arguments),
            working_directory,
            tuple(sorted(environment.items())),
        )
    return expectations


def _loaded_service_contract_matches(
    loaded: tuple[str, tuple[str, ...], str, tuple[tuple[str, str], ...]],
    expected: tuple[str, tuple[str, ...], str, tuple[tuple[str, str], ...]],
    *,
    label: str,
) -> bool:
    if loaded[:3] != expected[:3]:
        return False
    loaded_environment = dict(loaded[3])
    expected_environment = dict(expected[3])
    if (
        len(loaded_environment) != len(loaded[3])
        or len(expected_environment) != len(expected[3])
    ):
        return False
    loaded_extra_keys = set(loaded_environment) - set(expected_environment)
    if loaded_extra_keys - ALLOWED_LOADED_ENVIRONMENT_EXTRAS:
        return False
    if (
        "XPC_SERVICE_NAME" in loaded_environment
        and loaded_environment["XPC_SERVICE_NAME"] != label
    ):
        return False
    return all(
        loaded_environment.get(key) == value
        for key, value in expected_environment.items()
    )


def _verify_service_process(pid: int, module: str) -> None:
    result = _run_observer(
        ["/bin/ps", "-ww", "-p", str(pid), "-o", "args="],
        "service_process_unknown",
    )
    if result.returncode != 0 or result.stderr:
        raise RecoveryReceiptError("service_process_unknown")
    try:
        arguments = shlex.split(result.stdout.strip())
    except ValueError as exc:
        raise RecoveryReceiptError("service_process_unknown") from exc
    if (
        len(arguments) != 3
        or arguments[0] != PYTHON_EXECUTABLE
        or arguments[1] != "-m"
        or arguments[2] != module
    ):
        raise RecoveryReceiptError("service_process_mismatch")


def observe_service_states(evidence: ContractEvidence) -> dict[str, bool]:
    states: dict[str, bool] = {}
    seen_pids: set[int] = set()
    expectations = _expected_loaded_service_contracts(evidence)
    for contract in active_contracts.SERVICE_CONTRACTS:
        target = _launchctl_target(contract.label)
        first = _run_observer(
            ["/bin/launchctl", "print", target],
            "service_outcome_unknown",
        )
        pid, first_loaded_contract = _loaded_service_contract(first)
        if not _loaded_service_contract_matches(
            first_loaded_contract,
            expectations[contract.name],
            label=contract.label,
        ):
            raise RecoveryReceiptError("service_contract_mismatch")
        if pid in seen_pids:
            raise RecoveryReceiptError("service_process_mismatch")
        _verify_service_process(pid, contract.module)
        second = _run_observer(
            ["/bin/launchctl", "print", target],
            "service_outcome_unknown",
        )
        second_pid, second_loaded_contract = _loaded_service_contract(second)
        if second_pid != pid:
            raise RecoveryReceiptError("service_process_changed")
        if second_loaded_contract != first_loaded_contract:
            raise RecoveryReceiptError("service_contract_changed")
        if not _loaded_service_contract_matches(
            second_loaded_contract,
            expectations[contract.name],
            label=contract.label,
        ):
            raise RecoveryReceiptError("service_contract_mismatch")
        _verify_service_process(pid, contract.module)
        seen_pids.add(pid)
        states[contract.name] = True
    return states


def observe_scheduler_disabled() -> bool:
    result = _run_observer(
        ["/bin/launchctl", "print", _launchctl_target(SCHEDULER_LABEL)],
        "scheduler_outcome_unknown",
    )
    combined = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode == 0:
        return False
    exact_missing = combined == "Could not find service"
    detailed_missing = re.fullmatch(
        r'(?:Bad request\.\s*)?Could not find service "com\.jarvis-v3\.scheduler" '
        r'in domain for user gui:\s*[0-9]+\.?',
        combined,
    ) is not None
    if exact_missing or detailed_missing:
        return True
    raise RecoveryReceiptError("scheduler_outcome_unknown")


def observe_system(manifest_path: Path | str) -> SystemObservation:
    commit = active_source_commit()
    evidence = validate_active_contract_manifest(manifest_path)
    _verify_installed_contracts(evidence)
    _verify_log_custody()
    services = _normalize_service_states(
        evidence.expected_services, observe_service_states(evidence)
    )
    scheduler_disabled = observe_scheduler_disabled()
    if scheduler_disabled is not True:
        raise RecoveryReceiptError("scheduler_not_disabled")
    network = observe_network_reachability()
    if type(network) is not bool:
        raise RecoveryReceiptError("network_outcome_unknown")
    boot_identity = observe_boot_identity()
    if active_source_commit() != commit:
        raise RecoveryReceiptError("source_state_changed")
    return SystemObservation(
        source_commit=commit,
        manifest_digest=evidence.manifest_digest,
        contract_set_digest=evidence.contract_set_digest,
        env_content_digest=evidence.env_content_digest,
        expected_services=evidence.expected_services,
        boot_identity=boot_identity,
        network_reachable=network,
        scheduler_disabled=True,
        service_states=services,
    )


def _integrity_digest(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("integrity_digest", None)
    return hashlib.sha256(b"jarvis-v3-recovery-receipt-v3\0" + _canonical_json(unsigned)).hexdigest()


def _seal(payload: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(payload)
    sealed["integrity_digest"] = _integrity_digest(sealed)
    return sealed


def _validate_payload(payload: object) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) != STATE_KEYS:
        raise RecoveryReceiptError("state_corrupt")
    if payload.get("version") != RECEIPT_VERSION:
        raise RecoveryReceiptError("state_version_mismatch")
    digest = payload.get("integrity_digest")
    if type(digest) is not str or not secrets.compare_digest(digest, _integrity_digest(payload)):
        raise RecoveryReceiptError("state_integrity_failed")
    if payload.get("receipt_contract_manifest_digest") != contract_manifest_digest():
        raise RecoveryReceiptError("contract_manifest_mismatch")
    _validate_digest(payload.get("active_contract_manifest_digest"), "active_contract_manifest_invalid")
    _validate_digest(payload.get("active_contract_set_digest"), "active_contract_set_invalid")
    _validate_digest(payload.get("env_content_digest"), "environment_content_invalid")
    _validate_source_commit(payload.get("source_commit"))
    if type(payload.get("receipt_id")) is not str or re.fullmatch(
        r"[0-9a-f]{32}", payload["receipt_id"]
    ) is None:
        raise RecoveryReceiptError("state_corrupt")
    created_at = _parse_utc_timestamp(payload.get("created_at"))
    future_limit = _utc_datetime() + timedelta(seconds=MAX_FUTURE_SKEW_SECONDS)
    if created_at > future_limit:
        raise RecoveryReceiptError("state_timestamp_in_future")
    kind = payload.get("kind")
    if kind not in CONTRACT_MANIFEST["proof_kinds"] or payload.get("state") not in {"prepared", "finalized"}:
        raise RecoveryReceiptError("state_corrupt")
    expected_raw = payload.get("expected_services")
    if type(expected_raw) is not list:
        raise RecoveryReceiptError("state_corrupt")
    expected = _normalize_services(expected_raw)
    observations = payload.get("observations")
    if type(observations) is not list or not observations or payload.get("scheduler_disabled") is not True:
        raise RecoveryReceiptError("state_corrupt")
    salt = payload.get("boot_hash_salt")
    if type(salt) is not str or re.fullmatch(r"[0-9a-f]{32}", salt) is None:
        raise RecoveryReceiptError("state_corrupt")
    phases = CONTRACT_MANIFEST["proof_kinds"][kind]
    if len(observations) > len(phases):
        raise RecoveryReceiptError("state_corrupt")
    first_boot: str | None = None
    last_observed_at = created_at
    for ordinal, observation in enumerate(observations):
        if type(observation) is not dict or set(observation) != OBSERVATION_KEYS:
            raise RecoveryReceiptError("state_corrupt")
        boot_hash = observation.get("boot_session_hash")
        if (
            observation.get("ordinal") != ordinal
            or observation.get("phase") != phases[ordinal]
            or type(boot_hash) is not str
            or DIGEST_RE.fullmatch(boot_hash) is None
            or observation.get("scheduler_disabled") is not True
        ):
            raise RecoveryReceiptError("state_corrupt")
        _normalize_service_states(expected, observation.get("service_states"))
        observed_at = _parse_utc_timestamp(observation.get("observed_at"))
        if observed_at < last_observed_at or observed_at > future_limit:
            raise RecoveryReceiptError("state_corrupt")
        last_observed_at = observed_at
        network = observation.get("network_reachable")
        if type(network) is not bool:
            raise RecoveryReceiptError("network_outcome_unknown")
        if ordinal == 0:
            first_boot = boot_hash
            if network is not True:
                raise RecoveryReceiptError("state_corrupt")
        elif kind == "network":
            if boot_hash != first_boot or network is not (observation["phase"] == "network-up"):
                raise RecoveryReceiptError("contradictory_stored_evidence")
        elif boot_hash == first_boot or network is not True:
            raise RecoveryReceiptError("contradictory_stored_evidence")
    finalized = len(observations) == len(phases)
    if finalized != (payload.get("state") == "finalized") or finalized != (payload.get("verdict") == "proven"):
        raise RecoveryReceiptError("state_corrupt")
    finalized_raw = payload.get("finalized_at")
    if finalized:
        finalized_at = _parse_utc_timestamp(finalized_raw)
        if finalized_at < last_observed_at or finalized_at > future_limit:
            raise RecoveryReceiptError("state_corrupt")
    elif finalized_raw is not None or payload.get("verdict") != "pending":
        raise RecoveryReceiptError("state_corrupt")
    if payload.get("boundary") != CONTRACT_MANIFEST["boundaries"]:
        raise RecoveryReceiptError("boundary_contract_mismatch")
    return payload


def _open_state_file(parent_fd: int, name: str, flags: int, mode: int = 0o600) -> int:
    fd = os.open(name, flags | getattr(os, "O_NOFOLLOW", 0), mode, dir_fd=parent_fd)
    info = os.fstat(fd)
    if (
        not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
        or info.st_mode & 0o077 or info.st_nlink != 1
    ):
        os.close(fd)
        raise RecoveryReceiptError("unsafe_state_permissions")
    return fd


@contextmanager
def _state_parent(path: Path) -> Iterator[tuple[int, str]]:
    parent_fd = _open_owner_directory(path.parent, "unsafe_state_parent")
    try:
        yield parent_fd, path.name
    finally:
        os.close(parent_fd)


def _require_parent_path_stable(path: Path, parent_fd: int) -> None:
    reopened = _open_owner_directory(path, "state_parent_changed")
    try:
        expected = os.fstat(parent_fd)
        current = os.fstat(reopened)
    finally:
        os.close(reopened)
    if (expected.st_dev, expected.st_ino) != (current.st_dev, current.st_ino):
        raise RecoveryReceiptError("state_parent_changed")


def _state_identity(parent_fd: int, name: str) -> tuple[int, int, int, int, int]:
    try:
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise RecoveryReceiptError("state_changed") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o077
        or info.st_nlink != 1
    ):
        raise RecoveryReceiptError("state_changed")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_mode)


def _require_state_identity(
    parent_fd: int,
    name: str,
    expected: tuple[int, int, int, int, int],
) -> None:
    if _state_identity(parent_fd, name) != expected:
        raise RecoveryReceiptError("state_changed")


@contextmanager
def _state_lock(parent_fd: int, name: str) -> Iterator[None]:
    lock_name = f".{name}.lock"
    try:
        fd = _open_state_file(parent_fd, lock_name, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        raise RecoveryReceiptError("state_lock_unavailable") from exc
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _read_state(parent_fd: int, name: str) -> dict[str, Any]:
    try:
        fd = _open_state_file(parent_fd, name, os.O_RDONLY)
    except FileNotFoundError as exc:
        raise RecoveryReceiptError("state_not_found") from exc
    except OSError as exc:
        raise RecoveryReceiptError("state_unavailable") from exc
    before = os.fstat(fd)
    try:
        content = _read_bounded_fd(fd, MAX_STATE_BYTES, "state")
        after = os.fstat(fd)
        named_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise RecoveryReceiptError("state_unavailable") from exc
    finally:
        os.close(fd)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or (after.st_dev, after.st_ino) != (named_after.st_dev, named_after.st_ino)
        or stat.S_ISLNK(named_after.st_mode)
    ):
        raise RecoveryReceiptError("state_unstable")
    payload = _load_canonical_json(content, "state_corrupt")
    return _validate_payload(payload)


def _write_state(
    parent_fd: int,
    name: str,
    payload: dict[str, Any],
    *,
    create: bool,
    expected_identity: tuple[int, int, int, int, int] | None = None,
) -> tuple[int, int, int, int, int]:
    data = _canonical_json_document(_seal(payload))
    temp_name = f".{name}.{secrets.token_hex(16)}.tmp"
    temp_exists = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(temp_name, flags, 0o600, dir_fd=parent_fd)
        temp_exists = True
        try:
            os.fchmod(fd, 0o600)
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short state write")
                view = view[written:]
            os.fsync(fd)
            temp_info = os.fstat(fd)
        finally:
            os.close(fd)
        if create:
            try:
                os.link(
                    temp_name,
                    name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise RecoveryReceiptError("state_already_exists") from exc
            os.unlink(temp_name, dir_fd=parent_fd)
            temp_exists = False
        else:
            if expected_identity is None:
                raise RecoveryReceiptError("state_changed")
            _require_state_identity(parent_fd, name, expected_identity)
            os.replace(
                temp_name,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            temp_exists = False
        os.fsync(parent_fd)
        committed_identity = _state_identity(parent_fd, name)
        if committed_identity != (
            temp_info.st_dev,
            temp_info.st_ino,
            temp_info.st_size,
            temp_info.st_mtime_ns,
            temp_info.st_mode,
        ):
            raise RecoveryReceiptError("state_changed")
        return committed_identity
    except RecoveryReceiptError:
        raise
    except OSError as exc:
        raise RecoveryReceiptError("state_write_failed") from exc
    finally:
        if temp_exists:
            try:
                os.unlink(temp_name, dir_fd=parent_fd)
            except OSError:
                pass


def _observation(ordinal: int, phase: str, boot_hash: str, observed: SystemObservation) -> dict[str, Any]:
    return {
        "ordinal": ordinal,
        "phase": phase,
        "observed_at": _utc_now(),
        "boot_session_hash": boot_hash,
        "network_reachable": observed.network_reachable,
        "scheduler_disabled": observed.scheduler_disabled,
        "service_states": observed.service_states,
    }


def _validate_current_binding(payload: dict[str, Any], observed: SystemObservation) -> None:
    if payload["source_commit"] != observed.source_commit:
        raise RecoveryReceiptError("source_commit_mismatch")
    if payload["active_contract_manifest_digest"] != observed.manifest_digest:
        raise RecoveryReceiptError("active_contract_manifest_mismatch")
    if payload["active_contract_set_digest"] != observed.contract_set_digest:
        raise RecoveryReceiptError("active_contract_set_mismatch")
    if payload["env_content_digest"] != observed.env_content_digest:
        raise RecoveryReceiptError("environment_content_mismatch")
    if tuple(payload["expected_services"]) != observed.expected_services:
        raise RecoveryReceiptError("service_evidence_mismatch")


def prepare_receipt(
    state_path: Path | str, *, kind: str, active_contract_manifest: Path | str
) -> dict[str, Any]:
    path = _validate_state_path(state_path)
    if kind not in CONTRACT_MANIFEST["proof_kinds"]:
        raise RecoveryReceiptError("invalid_proof_kind")
    with _state_parent(path) as (parent_fd, name), _state_lock(parent_fd, name):
        try:
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise RecoveryReceiptError("state_already_exists")
        observed = observe_system(active_contract_manifest)
        if observed.network_reachable is not True:
            raise RecoveryReceiptError("baseline_network_unavailable")
        services = _normalize_services(observed.expected_services)
        states = _normalize_service_states(services, observed.service_states)
        if observed.scheduler_disabled is not True:
            raise RecoveryReceiptError("scheduler_not_disabled")
        salt = secrets.token_hex(16)
        boot_hash = _hash_boot_identity(observed.boot_identity, salt)
        phase = "prepared" if kind == "network" else "preboot"
        normalized_observed = SystemObservation(
            source_commit=observed.source_commit,
            manifest_digest=observed.manifest_digest,
            contract_set_digest=observed.contract_set_digest,
            env_content_digest=observed.env_content_digest,
            expected_services=services,
            boot_identity=observed.boot_identity,
            network_reachable=True,
            scheduler_disabled=True,
            service_states=states,
        )
        payload = {
            "version": RECEIPT_VERSION,
            "receipt_id": secrets.token_hex(16),
            "receipt_contract_manifest_digest": contract_manifest_digest(),
            "active_contract_manifest_digest": observed.manifest_digest,
            "active_contract_set_digest": observed.contract_set_digest,
            "env_content_digest": observed.env_content_digest,
            "source_commit": observed.source_commit,
            "kind": kind,
            "state": "prepared",
            "created_at": _utc_now(),
            "finalized_at": None,
            "boot_hash_salt": salt,
            "scheduler_disabled": True,
            "expected_services": list(services),
            "observations": [_observation(0, phase, boot_hash, normalized_observed)],
            "boundary": dict(CONTRACT_MANIFEST["boundaries"]),
            "verdict": "pending",
        }
        _validate_payload(_seal(payload))
        _require_parent_path_stable(path.parent, parent_fd)
        committed_identity = _write_state(parent_fd, name, payload, create=True)
        _require_parent_path_stable(path.parent, parent_fd)
        _require_state_identity(parent_fd, name, committed_identity)
    return _public_summary(payload, observed)


def record_observation(
    state_path: Path | str, *, phase: str, active_contract_manifest: Path | str
) -> dict[str, Any]:
    path = _validate_state_path(state_path)
    with _state_parent(path) as (parent_fd, name), _state_lock(parent_fd, name):
        payload = _read_state(parent_fd, name)
        original_state_identity = _state_identity(parent_fd, name)
        if payload["state"] == "finalized":
            raise RecoveryReceiptError("receipt_replay_blocked")
        observed = observe_system(active_contract_manifest)
        _validate_current_binding(payload, observed)
        phases = CONTRACT_MANIFEST["proof_kinds"][payload["kind"]]
        observations = list(payload["observations"])
        ordinal = len(observations)
        if ordinal >= len(phases) or phase != phases[ordinal]:
            raise RecoveryReceiptError("observation_out_of_order")
        boot_hash = _hash_boot_identity(observed.boot_identity, payload["boot_hash_salt"])
        first_boot = observations[0]["boot_session_hash"]
        if payload["kind"] == "network":
            if boot_hash != first_boot:
                raise RecoveryReceiptError("wrong_boot_transition")
            expected_network = phase == "network-up"
        else:
            if boot_hash == first_boot:
                raise RecoveryReceiptError("wrong_boot_transition")
            expected_network = True
        if observed.network_reachable is not expected_network:
            raise RecoveryReceiptError("contradictory_network_evidence")
        _normalize_service_states(payload["expected_services"], observed.service_states)
        observations.append(_observation(ordinal, phase, boot_hash, observed))
        payload["observations"] = observations
        if len(observations) == len(phases):
            payload["state"] = "finalized"
            payload["finalized_at"] = _utc_now()
            payload["verdict"] = "proven"
        _validate_payload(_seal(payload))
        _require_parent_path_stable(path.parent, parent_fd)
        _require_state_identity(parent_fd, name, original_state_identity)
        committed_identity = _write_state(
            parent_fd,
            name,
            payload,
            create=False,
            expected_identity=original_state_identity,
        )
        _require_parent_path_stable(path.parent, parent_fd)
        _require_state_identity(parent_fd, name, committed_identity)
    return _public_summary(payload, observed)


def inspect_receipt(
    state_path: Path | str, *, active_contract_manifest: Path | str
) -> dict[str, Any]:
    path = _validate_state_path(state_path)
    with _state_parent(path) as (parent_fd, name), _state_lock(parent_fd, name):
        payload = _read_state(parent_fd, name)
        original_state_identity = _state_identity(parent_fd, name)
        observed = observe_system(active_contract_manifest)
        _validate_current_binding(payload, observed)
        _require_parent_path_stable(path.parent, parent_fd)
        _require_state_identity(parent_fd, name, original_state_identity)
    return _public_summary(payload, observed)


def _public_summary(payload: dict[str, Any], observed: SystemObservation) -> dict[str, Any]:
    return {
        "ok": True,
        "proof_kind": payload["kind"],
        "proof_state": payload["state"],
        "verdict": payload["verdict"],
        "observation_count": len(payload["observations"]),
        "expected_service_count": len(payload["expected_services"]),
        "scheduler_disabled": payload["scheduler_disabled"] is True and observed.scheduler_disabled is True,
        "contract_bound": payload["receipt_contract_manifest_digest"] == contract_manifest_digest(),
        "active_contract_bound": (
            payload["active_contract_manifest_digest"] == observed.manifest_digest
            and payload["active_contract_set_digest"] == observed.contract_set_digest
        ),
        "source_bound": payload["source_commit"] == observed.source_commit,
        "authorizes_execution": False,
        "authorizes_approval": False,
        "external_side_effects": False,
        "source_commit": observed.source_commit,
        "active_contract_manifest_sha256": observed.manifest_digest,
        "contract_set_sha256": observed.contract_set_digest,
    }


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise RecoveryReceiptError("invalid_cli_arguments")


def _build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(prog="v3_recovery_receipt", add_help=True)
    commands = parser.add_subparsers(dest="command", required=True, parser_class=_SafeArgumentParser)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--state", required=True)
    prepare.add_argument("--kind", required=True, choices=("network", "reboot"))
    prepare.add_argument("--active-contract-manifest", required=True)
    observe = commands.add_parser("observe")
    observe.add_argument("--state", required=True)
    observe.add_argument("--phase", required=True)
    observe.add_argument("--active-contract-manifest", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--state", required=True)
    inspect.add_argument("--active-contract-manifest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _build_parser().parse_args(argv)
        if args.command == "prepare":
            result = prepare_receipt(
                args.state, kind=args.kind, active_contract_manifest=args.active_contract_manifest
            )
        elif args.command == "observe":
            result = record_observation(
                args.state, phase=args.phase, active_contract_manifest=args.active_contract_manifest
            )
        else:
            result = inspect_receipt(
                args.state, active_contract_manifest=args.active_contract_manifest
            )
    except RecoveryReceiptError as exc:
        print(json.dumps({"ok": False, "error": exc.code}, sort_keys=True, separators=(",", ":")))
        return 2
    except Exception:
        print('{"error":"internal_error","ok":false}')
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
