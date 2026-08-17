"""Inert adapter for one separately supplied credential-scanner executable.

The adapter neither acquires nor chooses a scanner. It binds one absolute local
source file by SHA-256, then launches the current Python interpreter by pathname
and delivers those exact source bytes through a small content-free protocol
without a shell. The interpreter's open descriptor and named path are checked
before and after launch, but execution is not bound to that descriptor; the
current interpreter remains a trusted runtime dependency. The adapter also
revalidates the immutable candidate afterward. The child receives a scrubbed
credential/proxy environment and a neutral working directory, but retains the
caller's OS identity and host filesystem/network access. Kernel-enforced network
and filesystem denial are not attested, so the scanner must be separately reviewed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import sys
import time
from typing import Sequence

from jarvis_v2.scripts.public_release_candidate import (
    CANDIDATE_PROFILE_PREVIEW,
    CANDIDATE_PROFILE_PUBLICATION,
    CANDIDATE_PROFILES,
    CandidateBuildError,
    _candidate_state_fd,
    _candidate_tree_has_finalized_modes_fd,
    _open_candidate_root,
    _publication_metadata,
)
from jarvis_v2.scripts.public_release_preflight import (
    PreflightConfigError,
    _scan_release_tree_fd,
    extra_deny_literals_from_env,
)


SCANNER_PROTOCOL_SCHEMA = "jarvis-independent-credential-scan"
SCANNER_PROTOCOL_VERSION = 1
SCANNER_PROTOCOL_ARGUMENT = "--jarvis-credential-scan-v1"
SCANNER_SOURCE_CONTRACT_LINE = b"# jarvis-independent-credential-scanner:1"
SCANNER_BOOTSTRAP = """import os
import sys

candidate_fd = int(sys.argv[1])
protocol_argument = sys.argv[2]
source = sys.stdin.buffer.read()
os.fchdir(candidate_fd)
os.close(candidate_fd)
sys.argv = ["-", protocol_argument, "."]
namespace = {"__name__": "__main__", "__file__": "<reviewed-scanner-source>"}
exec(compile(source, "<reviewed-scanner-source>", "exec"), namespace, namespace)
"""
EXPECTED_SHA256_RE = re.compile(r"[0-9a-f]{64}")
MAX_EXECUTABLE_PATH_BYTES = 1_024
MAX_SCANNER_EXECUTABLE_BYTES = 64 * 1024 * 1024
MAX_INTERPRETER_BYTES = 128 * 1024 * 1024
MAX_SCANNER_OUTPUT_BYTES = 64 * 1024
MAX_FINDING_COUNT = 1_000_000
SCANNER_TIMEOUT_SECONDS = 60.0
SCANNER_TERMINATE_GRACE_SECONDS = 0.5
SCANNER_CLEANUP_ATTEST_SECONDS = 3.0
SCANNER_NEUTRAL_DIRECTORY = os.path.realpath("/var/empty")
_OPEN_EXECUTABLE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
_SCANNER_RESULT_KEYS = frozenset({"schema", "version", "clean", "finding_count"})


class IndependentCredentialScanError(ValueError):
    """A content-free adapter failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class IndependentCredentialScanReport:
    clean: bool
    finding_count: int
    scanner_sha256: str
    interpreter_preflight_descriptor_sha256: str
    candidate_profile: str

    def summary_json_payload(self) -> dict[str, object]:
        return {
            "ok": self.clean,
            "clean": self.clean,
            "finding_count": self.finding_count,
            "scanner_protocol_schema": SCANNER_PROTOCOL_SCHEMA,
            "scanner_protocol_version": SCANNER_PROTOCOL_VERSION,
            "scanner_sha256": self.scanner_sha256,
            "candidate_profile": self.candidate_profile,
            "candidate_valid": True,
            "candidate_stable": True,
            "candidate_finalized_immutable": True,
            "scanner_identity_revalidated": True,
            "scanner_source_bytes_delivered_to_bootstrap_exactly": True,
            "scanner_path_used_for_child_execution": False,
            "scanner_source_contract": "isolated_python_stdin_v1",
            "scanner_source_format_marker_validated": True,
            "candidate_access_bound_to_open_descriptor": True,
            "candidate_descriptor_inherited_by_child": True,
            "interpreter_preflight_descriptor_sha256": (
                self.interpreter_preflight_descriptor_sha256
            ),
            "interpreter_named_path_matched_open_descriptor_before_launch": True,
            "interpreter_open_descriptor_stable_after_launch": True,
            "interpreter_named_path_matched_open_descriptor_after_launch": True,
            "interpreter_execution_bound_to_hashed_descriptor": False,
            "interpreter_execution_identity_attested": False,
            "trusted_runtime_dependency": "current_python_interpreter_named_path",
            "scanner_acquired": False,
            "scanner_version_chosen": False,
            "shell_used": False,
            "credential_environment_inherited": False,
            "proxy_environment_inherited": False,
            "network_environment_inherited": False,
            "working_directory_inherited": False,
            "neutral_launch_working_directory_used": True,
            "scanner_working_directory_is_candidate_descriptor": True,
            "os_process_identity_inherited": True,
            "host_filesystem_access_available": True,
            "host_network_access_available": True,
            "kernel_network_denial_attested": False,
            "kernel_filesystem_write_denial_attested": False,
            "candidate_mutation_observed": False,
            "candidate_write_absence_attested": False,
            "external_scanner_write_absence_attested": False,
            "paths_included": False,
            "private_content_included": False,
            "scanner_output_included": False,
            "adapter_writes_files": False,
            "adapter_changes_permissions": False,
            "publishes": False,
            "authorizes_publication": False,
        }


@dataclass(frozen=True)
class _ExecutableCustody:
    path: Path
    fd: int
    opened_stat: os.stat_result
    sha256: str
    program: bytes


@dataclass(frozen=True)
class _InterpreterCustody:
    path: Path
    fd: int
    opened_stat: os.stat_result
    sha256: str


def _stat_token(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        getattr(value, "st_flags", 0),
    )


def _read_and_hash_executable_fd(fd: int) -> tuple[bytes, str]:
    try:
        opened = os.fstat(fd)
        if opened.st_size < 1 or opened.st_size > MAX_SCANNER_EXECUTABLE_BYTES:
            raise IndependentCredentialScanError("scanner_executable_invalid")
        os.lseek(fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        content = bytearray()
        remaining = opened.st_size
        while remaining:
            chunk = os.read(fd, min(65_536, remaining))
            if not chunk:
                raise IndependentCredentialScanError("scanner_executable_invalid")
            digest.update(chunk)
            content.extend(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise IndependentCredentialScanError("scanner_executable_invalid")
        if _stat_token(os.fstat(fd)) != _stat_token(opened):
            raise IndependentCredentialScanError("scanner_executable_invalid")
        return bytes(content), digest.hexdigest()
    except IndependentCredentialScanError:
        raise
    except OSError:
        raise IndependentCredentialScanError("scanner_executable_invalid") from None


def _hash_interpreter_fd(fd: int) -> str:
    try:
        opened = os.fstat(fd)
        if opened.st_size < 1 or opened.st_size > MAX_INTERPRETER_BYTES:
            raise IndependentCredentialScanError("interpreter_invalid")
        os.lseek(fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        remaining = opened.st_size
        while remaining:
            chunk = os.read(fd, min(65_536, remaining))
            if not chunk:
                raise IndependentCredentialScanError("interpreter_invalid")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise IndependentCredentialScanError("interpreter_invalid")
        if _stat_token(os.fstat(fd)) != _stat_token(opened):
            raise IndependentCredentialScanError("interpreter_invalid")
        return digest.hexdigest()
    except IndependentCredentialScanError:
        raise
    except OSError:
        raise IndependentCredentialScanError("interpreter_invalid") from None


def _open_scanner_executable(path: str | Path, expected_sha256: str) -> _ExecutableCustody:
    if type(expected_sha256) is not str or not EXPECTED_SHA256_RE.fullmatch(expected_sha256):
        raise IndependentCredentialScanError("scanner_sha256_invalid")
    try:
        raw = Path(path)
        encoded = os.fsencode(raw)
        resolved = raw.resolve()
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeError):
        raise IndependentCredentialScanError("scanner_executable_invalid") from None
    if (
        not raw.is_absolute()
        or not encoded
        or len(encoded) > MAX_EXECUTABLE_PATH_BYTES
        or resolved != raw
    ):
        raise IndependentCredentialScanError("scanner_executable_invalid")
    fd: int | None = None
    try:
        fd = os.open(raw, _OPEN_EXECUTABLE_FLAGS)
        opened = os.fstat(fd)
        named = os.stat(raw, follow_symlinks=False)
        mode = stat.S_IMODE(opened.st_mode)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _stat_token(opened) != _stat_token(named)
            or opened.st_nlink != 1
            or opened.st_uid not in {0, os.geteuid()}
            or not mode & stat.S_IXUSR
            or mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            or opened.st_mode & (stat.S_ISUID | stat.S_ISGID)
        ):
            raise IndependentCredentialScanError("scanner_executable_invalid")
        program, actual_sha256 = _read_and_hash_executable_fd(fd)
        stable_opened = os.fstat(fd)
        stable_named = os.stat(raw, follow_symlinks=False)
        if (
            _stat_token(stable_opened) != _stat_token(opened)
            or _stat_token(stable_named) != _stat_token(opened)
        ):
            raise IndependentCredentialScanError("scanner_executable_invalid")
        if actual_sha256 != expected_sha256:
            raise IndependentCredentialScanError("scanner_sha256_mismatch")
        lines = program.splitlines()
        if (
            len(lines) < 2
            or not lines[0].startswith(b"#!")
            or lines[1] != SCANNER_SOURCE_CONTRACT_LINE
            or b"\x00" in program
        ):
            raise IndependentCredentialScanError("scanner_source_contract_invalid")
        result = _ExecutableCustody(raw, fd, opened, actual_sha256, program)
        fd = None
        return result
    except IndependentCredentialScanError:
        raise
    except (OSError, ValueError):
        raise IndependentCredentialScanError("scanner_executable_invalid") from None
    finally:
        if fd is not None:
            os.close(fd)


def _open_python_interpreter() -> _InterpreterCustody:
    fd: int | None = None
    try:
        path = Path(sys.executable).resolve(strict=True)
        encoded = os.fsencode(path)
        if not path.is_absolute() or not encoded or len(encoded) > MAX_EXECUTABLE_PATH_BYTES:
            raise IndependentCredentialScanError("interpreter_invalid")
        fd = os.open(path, _OPEN_EXECUTABLE_FLAGS)
        opened = os.fstat(fd)
        named = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _stat_token(opened) != _stat_token(named)
            or opened.st_nlink < 1
            or not stat.S_IMODE(opened.st_mode) & stat.S_IXUSR
            or opened.st_mode & (stat.S_ISUID | stat.S_ISGID)
        ):
            raise IndependentCredentialScanError("interpreter_invalid")
        digest = _hash_interpreter_fd(fd)
        stable_opened = os.fstat(fd)
        stable_named = os.stat(path, follow_symlinks=False)
        if (
            _stat_token(stable_opened) != _stat_token(opened)
            or _stat_token(stable_named) != _stat_token(opened)
        ):
            raise IndependentCredentialScanError("interpreter_invalid")
        result = _InterpreterCustody(path, fd, opened, digest)
        fd = None
        return result
    except IndependentCredentialScanError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeError):
        raise IndependentCredentialScanError("interpreter_invalid") from None
    finally:
        if fd is not None:
            os.close(fd)


def _scanner_environment() -> dict[str, str]:
    return {
        "HOME": "/var/empty",
        "LANG": "C",
        "LC_ALL": "C",
        "NO_PROXY": "*",
        "PATH": "/usr/bin:/bin",
        "TMPDIR": "/var/empty",
        "no_proxy": "*",
    }


def _process_group_exists(group_id: int) -> bool | None:
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return None
    return True


def _terminate_process_group(process: subprocess.Popen[bytes]) -> bool:
    cleanup_known = True
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError:
        cleanup_known = False
        if process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
    try:
        process.wait(timeout=SCANNER_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    group_exists = _process_group_exists(process.pid)
    if group_exists is None:
        cleanup_known = False
        group_exists = process.poll() is None
    if group_exists:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            cleanup_known = False
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass
        try:
            process.wait(timeout=SCANNER_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            cleanup_known = False
    deadline = time.monotonic() + SCANNER_CLEANUP_ATTEST_SECONDS
    while True:
        remaining_group = _process_group_exists(process.pid)
        if remaining_group is False:
            break
        if remaining_group is None:
            cleanup_known = False
            break
        if time.monotonic() >= deadline:
            cleanup_known = False
            break
        time.sleep(0.01)
    if process.poll() is None:
        cleanup_known = False
    return cleanup_known


def _run_scanner_bounded_impl(
    command: Sequence[str],
    program: bytes,
    *,
    pass_fds: Sequence[int] = (),
) -> tuple[int, bytes, bytes]:
    try:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_scanner_environment(),
            cwd=SCANNER_NEUTRAL_DIRECTORY,
            shell=False,
            close_fds=True,
            pass_fds=tuple(pass_fds),
            start_new_session=True,
        )
    except OSError:
        raise IndependentCredentialScanError("scanner_start_failed") from None
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    selector = selectors.DefaultSelector()
    output_streams = {
        process.stdout.fileno(): (process.stdout, bytearray()),
        process.stderr.fileno(): (process.stderr, bytearray()),
    }
    input_fd = process.stdin.fileno()
    input_offset = 0
    deadline = time.monotonic() + SCANNER_TIMEOUT_SECONDS
    try:
        for stream, _ in output_streams.values():
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ)
        os.set_blocking(input_fd, False)
        selector.register(process.stdin, selectors.EVENT_WRITE)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise IndependentCredentialScanError("scanner_timeout")
            events = selector.select(min(0.25, remaining))
            if not events:
                continue
            for key, _ in events:
                if key.fd == input_fd:
                    try:
                        written = os.write(input_fd, program[input_offset : input_offset + 16_384])
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        raise IndependentCredentialScanError("scanner_start_failed") from None
                    input_offset += written
                    if input_offset == len(program):
                        selector.unregister(process.stdin)
                        process.stdin.close()
                    continue
                stream, output = output_streams[key.fd]
                try:
                    chunk = os.read(stream.fileno(), 65_536)
                except OSError:
                    raise IndependentCredentialScanError("scanner_output_invalid") from None
                if not chunk:
                    selector.unregister(stream)
                    continue
                output.extend(chunk)
                if len(output) > MAX_SCANNER_OUTPUT_BYTES:
                    raise IndependentCredentialScanError("scanner_output_limit")
        try:
            return_code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            raise IndependentCredentialScanError("scanner_timeout") from None
        return return_code, bytes(output_streams[process.stdout.fileno()][1]), bytes(
            output_streams[process.stderr.fileno()][1]
        )
    finally:
        selector.close()
        # A scanner can exit after leaving descendants behind. Always reap its
        # private process group, including on an otherwise valid result.
        cleanup_ok = _terminate_process_group(process)
        if not process.stdin.closed:
            process.stdin.close()
        process.stdout.close()
        process.stderr.close()
        if not cleanup_ok:
            raise IndependentCredentialScanError("scanner_cleanup_unknown") from None


def _run_scanner_bounded(
    command: Sequence[str],
    program: bytes,
    *,
    pass_fds: Sequence[int] = (),
) -> tuple[int, bytes, bytes]:
    """Run one child while removing primary/cleanup exception chains."""

    failure_code: str | None = None
    try:
        return _run_scanner_bounded_impl(
            command,
            program,
            pass_fds=pass_fds,
        )
    except IndependentCredentialScanError as error:
        failure_code = error.code
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError):
        failure_code = "scanner_execution_failed"
    if failure_code is None:  # pragma: no cover - defensive type narrowing
        failure_code = "scanner_execution_failed"
    raise IndependentCredentialScanError(failure_code)


def _parse_scanner_result(stdout: bytes, stderr: bytes, return_code: int) -> tuple[bool, int]:
    if stderr or not stdout or len(stdout) > MAX_SCANNER_OUTPUT_BYTES or b"\x00" in stdout:
        raise IndependentCredentialScanError("scanner_output_invalid")
    duplicate = False

    def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        nonlocal duplicate
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                duplicate = True
            result[key] = value
        return result

    try:
        payload = json.loads(stdout.decode("utf-8"), object_pairs_hook=strict_object)
    except (UnicodeDecodeError, ValueError):
        raise IndependentCredentialScanError("scanner_output_invalid") from None
    if duplicate or type(payload) is not dict or frozenset(payload) != _SCANNER_RESULT_KEYS:
        raise IndependentCredentialScanError("scanner_output_invalid")
    clean = payload.get("clean")
    finding_count = payload.get("finding_count")
    if (
        payload.get("schema") != SCANNER_PROTOCOL_SCHEMA
        or type(payload.get("version")) is not int
        or payload["version"] != SCANNER_PROTOCOL_VERSION
        or type(clean) is not bool
        or type(finding_count) is not int
        or finding_count < 0
        or finding_count > MAX_FINDING_COUNT
    ):
        raise IndependentCredentialScanError("scanner_output_invalid")
    canonical = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if stdout != canonical:
        raise IndependentCredentialScanError("scanner_output_invalid")
    if clean:
        if finding_count != 0 or return_code != 0:
            raise IndependentCredentialScanError("scanner_result_inconsistent")
    elif finding_count < 1 or return_code != 1:
        raise IndependentCredentialScanError("scanner_result_inconsistent")
    return clean, finding_count


def _executable_stable(custody: _ExecutableCustody) -> bool:
    try:
        opened_before = os.fstat(custody.fd)
        named_before = os.stat(custody.path, follow_symlinks=False)
        _, digest = _read_and_hash_executable_fd(custody.fd)
        opened_after = os.fstat(custody.fd)
        named_after = os.stat(custody.path, follow_symlinks=False)
    except (OSError, IndependentCredentialScanError):
        return False
    expected = _stat_token(custody.opened_stat)
    return (
        _stat_token(opened_before) == expected
        and _stat_token(named_before) == expected
        and _stat_token(opened_after) == expected
        and _stat_token(named_after) == expected
        and digest == custody.sha256
    )


def _interpreter_stable(custody: _InterpreterCustody) -> bool:
    try:
        opened_before = os.fstat(custody.fd)
        named_before = os.stat(custody.path, follow_symlinks=False)
        digest = _hash_interpreter_fd(custody.fd)
        opened_after = os.fstat(custody.fd)
        named_after = os.stat(custody.path, follow_symlinks=False)
    except (OSError, IndependentCredentialScanError):
        return False
    expected = _stat_token(custody.opened_stat)
    return (
        _stat_token(opened_before) == expected
        and _stat_token(named_before) == expected
        and _stat_token(opened_after) == expected
        and _stat_token(named_after) == expected
        and digest == custody.sha256
    )


def _candidate_root_token_stable(
    path: Path,
    root_fd: int,
    opened: os.stat_result,
) -> bool:
    try:
        current_fd = os.fstat(root_fd)
        current_named = os.stat(path, follow_symlinks=False)
    except OSError:
        return False
    token = _stat_token(opened)
    return (
        stat.S_ISDIR(current_fd.st_mode)
        and stat.S_ISDIR(current_named.st_mode)
        and _stat_token(current_fd) == token
        and _stat_token(current_named) == token
    )


def _scan_candidate_independently_impl(
    candidate_root: str | Path,
    scanner_executable: str | Path,
    expected_scanner_sha256: str,
    *,
    profile: str = CANDIDATE_PROFILE_PUBLICATION,
    environ: dict[str, str] | None = None,
) -> IndependentCredentialScanReport:
    if type(profile) is not str or profile not in CANDIDATE_PROFILES:
        raise IndependentCredentialScanError("candidate_profile_invalid")
    try:
        extra_deny_literals = extra_deny_literals_from_env(environ)
    except PreflightConfigError:
        raise IndependentCredentialScanError("private_review_invalid") from None
    opened = _open_candidate_root(candidate_root)
    if opened is None:
        raise IndependentCredentialScanError("candidate_invalid")
    candidate_path, candidate_fd, candidate_stat = opened
    custody: _ExecutableCustody | None = None
    interpreter: _InterpreterCustody | None = None
    try:
        before = _candidate_state_fd(candidate_fd, profile=profile)
        if before is None:
            raise IndependentCredentialScanError("candidate_invalid")
        if profile == CANDIDATE_PROFILE_PUBLICATION:
            try:
                _publication_metadata(before.blobs, extra_deny_literals)
            except CandidateBuildError:
                raise IndependentCredentialScanError("candidate_not_publication_ready") from None
        preflight_before = _scan_release_tree_fd(
            candidate_fd,
            extra_deny_literals=extra_deny_literals,
        )
        if (
            not preflight_before.ok
            or not _candidate_tree_has_finalized_modes_fd(candidate_fd)
            or not _candidate_root_token_stable(candidate_path, candidate_fd, candidate_stat)
        ):
            raise IndependentCredentialScanError("candidate_invalid")

        custody = _open_scanner_executable(scanner_executable, expected_scanner_sha256)
        interpreter = _open_python_interpreter()
        return_code, stdout, stderr = _run_scanner_bounded(
            (
                os.fspath(interpreter.path),
                "-I",
                "-S",
                "-B",
                "-c",
                SCANNER_BOOTSTRAP,
                str(candidate_fd),
                SCANNER_PROTOCOL_ARGUMENT,
            ),
            custody.program,
            pass_fds=(candidate_fd,),
        )

        if not _executable_stable(custody):
            raise IndependentCredentialScanError("scanner_identity_changed")
        if not _interpreter_stable(interpreter):
            raise IndependentCredentialScanError("interpreter_identity_changed")
        clean, finding_count = _parse_scanner_result(stdout, stderr, return_code)
        after = _candidate_state_fd(candidate_fd, profile=profile)
        preflight_after = _scan_release_tree_fd(
            candidate_fd,
            extra_deny_literals=extra_deny_literals,
        )
        if (
            after is None
            or before != after
            or not preflight_after.ok
            or not _candidate_tree_has_finalized_modes_fd(candidate_fd)
            or not _candidate_root_token_stable(candidate_path, candidate_fd, candidate_stat)
        ):
            raise IndependentCredentialScanError("candidate_changed")
        return IndependentCredentialScanReport(
            clean=clean,
            finding_count=finding_count,
            scanner_sha256=custody.sha256,
            interpreter_preflight_descriptor_sha256=interpreter.sha256,
            candidate_profile=profile,
        )
    finally:
        if custody is not None:
            os.close(custody.fd)
        if interpreter is not None:
            os.close(interpreter.fd)
        os.close(candidate_fd)


def scan_candidate_independently(
    candidate_root: str | Path,
    scanner_executable: str | Path,
    expected_scanner_sha256: str,
    *,
    profile: str = CANDIDATE_PROFILE_PUBLICATION,
    environ: dict[str, str] | None = None,
) -> IndependentCredentialScanReport:
    """Run the adapter while removing all internal exception object chains."""

    failure_code: str | None = None
    try:
        return _scan_candidate_independently_impl(
            candidate_root,
            scanner_executable,
            expected_scanner_sha256,
            profile=profile,
            environ=environ,
        )
    except IndependentCredentialScanError as error:
        failure_code = error.code
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError):
        failure_code = "independent_scan_failed"
    if failure_code is None:  # pragma: no cover - defensive type narrowing
        failure_code = "independent_scan_failed"
    raise IndependentCredentialScanError(failure_code)


def _failure_payload(code: str) -> dict[str, object]:
    return {
        "ok": False,
        "error_code": code,
        "paths_included": False,
        "private_content_included": False,
        "scanner_output_included": False,
        "scanner_source_bytes_delivered_to_bootstrap_exactly": False,
        "scanner_path_used_for_child_execution": False,
        "candidate_access_bound_to_open_descriptor": False,
        "candidate_descriptor_inherited_by_child": False,
        "interpreter_named_path_matched_open_descriptor_before_launch": False,
        "interpreter_open_descriptor_stable_after_launch": False,
        "interpreter_named_path_matched_open_descriptor_after_launch": False,
        "interpreter_execution_bound_to_hashed_descriptor": False,
        "interpreter_execution_identity_attested": False,
        "trusted_runtime_dependency": "current_python_interpreter_named_path",
        "scanner_acquired": False,
        "scanner_version_chosen": False,
        "shell_used": False,
        "credential_environment_inherited": False,
        "proxy_environment_inherited": False,
        "network_environment_inherited": False,
        "kernel_network_denial_attested": False,
        "kernel_filesystem_write_denial_attested": False,
        "adapter_writes_files": False,
        "publishes": False,
        "authorizes_publication": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one hash-bound local credential scanner through the content-free "
            "Jarvis scanner protocol."
        )
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--scanner", required=True)
    parser.add_argument("--scanner-sha256", required=True)
    parser.add_argument(
        "--profile",
        choices=sorted(CANDIDATE_PROFILES),
        default=CANDIDATE_PROFILE_PUBLICATION,
    )
    args = parser.parse_args(argv)
    try:
        report = scan_candidate_independently(
            args.candidate,
            args.scanner,
            args.scanner_sha256,
            profile=args.profile,
        )
    except IndependentCredentialScanError as exc:
        print(json.dumps(_failure_payload(exc.code), sort_keys=True, separators=(",", ":")))
        return 2
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError):
        print(
            json.dumps(
                _failure_payload("independent_scan_failed"),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    print(json.dumps(report.summary_json_payload(), sort_keys=True, separators=(",", ":")))
    return 0 if report.clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
