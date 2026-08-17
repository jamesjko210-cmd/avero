from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import traceback
from unittest import mock
import uuid

from jarvis_v2.scripts import independent_credential_scan as scan_module
from jarvis_v2.scripts.independent_credential_scan import (
    IndependentCredentialScanError,
    SCANNER_PROTOCOL_ARGUMENT,
    SCANNER_PROTOCOL_SCHEMA,
    _run_scanner_bounded,
    main as scan_main,
    scan_candidate_independently,
)
from jarvis_v2.scripts.public_release_candidate import build_public_candidate


PROCESS_CLEANUP_PIPE_DEADLINE_SECONDS = 5.0


def _git(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", os.fspath(root), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise SystemExit("independent scanner Git fixture failed")
    return result.stdout


def _source_fixture(root: Path) -> None:
    package = root / "jarvis_v2"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "module.py").write_text("VALUE = 'synthetic public source'\n", encoding="utf-8")
    _git(root, "init", "--quiet")
    _git(root, "add", "-A")
    _git(
        root,
        "-c",
        "user.name=Synthetic Maintainer",
        "-c",
        "user.email=maintainer@example.com",
        "commit",
        "--quiet",
        "-m",
        "synthetic scanner candidate",
    )


def _remove_candidate(candidate: Path) -> None:
    if not candidate.exists():
        return
    for path in sorted(candidate.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o700)
    candidate.chmod(0o700)
    shutil.rmtree(candidate)


@contextlib.contextmanager
def _preview_candidate():
    with TemporaryDirectory(prefix="jarvis-independent-source-", dir="/private/tmp") as temp:
        source = Path(temp)
        _source_fixture(source)
        candidate = Path("/private/tmp") / f"jarvis-independent-candidate-{uuid.uuid4().hex}"
        try:
            report = build_public_candidate(source, candidate)
            if not report.ok or not report.candidate_finalized_immutable:
                raise SystemExit("independent scanner candidate fixture was not finalized")
            yield candidate
        finally:
            _remove_candidate(candidate)


def _scanner_payload(clean: bool, finding_count: int) -> str:
    return json.dumps(
        {
            "clean": clean,
            "finding_count": finding_count,
            "schema": SCANNER_PROTOCOL_SCHEMA,
            "version": 1,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _write_scanner(
    root: Path,
    *,
    clean: bool = True,
    finding_count: int = 0,
    mutate_candidate: bool = False,
    malformed_output: str | None = None,
    stderr_text: str = "",
) -> tuple[Path, str]:
    scanner = root / f"scanner-{uuid.uuid4().hex}"
    payload = malformed_output if malformed_output is not None else _scanner_payload(clean, finding_count)
    body = f"""#!{sys.executable}
# jarvis-independent-credential-scanner:1
import json
import os
import pathlib
import sys

expected_environment = {{
    "HOME": "/var/empty",
    "LANG": "C",
    "LC_ALL": "C",
    "NO_PROXY": "*",
    "PATH": "/usr/bin:/bin",
    "TMPDIR": "/var/empty",
    "no_proxy": "*",
}}
if (
    len(sys.argv) != 3
    or sys.argv[1] != {SCANNER_PROTOCOL_ARGUMENT!r}
    or sys.argv[2] != "."
    or any(os.environ.get(key) != value for key, value in expected_environment.items())
    or any(
        key.lower().endswith(("token", "secret", "password", "credential"))
        for key in os.environ
    )
):
    print("protocol or environment mismatch", file=sys.stderr)
    raise SystemExit(2)
candidate = pathlib.Path(sys.argv[2])
if {mutate_candidate!r}:
    target = candidate / "jarvis_v2" / "module.py"
    target.chmod(0o600)
    target.write_text("changed by synthetic scanner\\n", encoding="utf-8")
if {stderr_text!r}:
    print({stderr_text!r}, file=sys.stderr)
print({payload!r})
raise SystemExit({0 if clean else 1})
"""
    scanner.write_text(body, encoding="utf-8")
    scanner.chmod(0o500)
    digest = hashlib.sha256(scanner.read_bytes()).hexdigest()
    return scanner, digest


def _snapshot(root: Path) -> tuple[tuple[str, int, int, int, str], ...]:
    rows: list[tuple[str, int, int, int, str]] = []
    for path in sorted((root, *root.rglob("*")), key=lambda item: os.fspath(item)):
        info = path.lstat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
        rows.append(
            (
                "." if path == root else path.relative_to(root).as_posix(),
                stat.S_IMODE(info.st_mode),
                info.st_size,
                info.st_mtime_ns,
                digest,
            )
        )
    return tuple(rows)


def _expect_error(candidate: Path, scanner: Path, digest: str, code: str) -> None:
    try:
        scan_candidate_independently(
            candidate,
            scanner,
            digest,
            profile="preview",
            environ={},
        )
    except IndependentCredentialScanError as exc:
        if exc.code != code:
            raise SystemExit(f"independent scanner error code drifted: {exc.code}")
    else:
        raise SystemExit(f"independent scanner accepted {code} fixture")


def test_clean_and_finding_receipts_are_content_free_and_inert() -> None:
    with _preview_candidate() as candidate, TemporaryDirectory(
        prefix="jarvis-independent-scanner-", dir="/private/tmp"
    ) as temp:
        root = Path(temp)
        for clean, count in ((True, 0), (False, 3)):
            scanner, digest = _write_scanner(root, clean=clean, finding_count=count)
            candidate_before = _snapshot(candidate)
            scanner_before = _snapshot(scanner)
            report = scan_candidate_independently(
                candidate,
                scanner,
                digest,
                profile="preview",
                environ={},
            )
            summary = report.summary_json_payload()
            if report.clean is not clean or report.finding_count != count or summary.get("ok") is not clean:
                raise SystemExit(f"independent scanner result drifted: {summary}")
            for key in (
                "scanner_acquired",
                "scanner_version_chosen",
                "shell_used",
                "credential_environment_inherited",
                "proxy_environment_inherited",
                "network_environment_inherited",
                "working_directory_inherited",
                "kernel_network_denial_attested",
                "kernel_filesystem_write_denial_attested",
                "candidate_mutation_observed",
                "candidate_write_absence_attested",
                "external_scanner_write_absence_attested",
                "interpreter_execution_bound_to_hashed_descriptor",
                "interpreter_execution_identity_attested",
                "paths_included",
                "private_content_included",
                "scanner_output_included",
                "adapter_writes_files",
                "adapter_changes_permissions",
                "publishes",
                "authorizes_publication",
            ):
                if summary.get(key) is not False:
                    raise SystemExit(f"independent scanner overstated its boundary: {summary}")
            if (
                summary.get("candidate_valid") is not True
                or summary.get("candidate_stable") is not True
                or summary.get("scanner_identity_revalidated") is not True
                or summary.get("scanner_source_bytes_delivered_to_bootstrap_exactly") is not True
                or summary.get("scanner_path_used_for_child_execution") is not False
                or summary.get("scanner_source_contract") != "isolated_python_stdin_v1"
                or summary.get("scanner_source_format_marker_validated") is not True
                or summary.get("candidate_access_bound_to_open_descriptor") is not True
                or summary.get("candidate_descriptor_inherited_by_child") is not True
                or summary.get("interpreter_named_path_matched_open_descriptor_before_launch")
                is not True
                or summary.get("interpreter_open_descriptor_stable_after_launch") is not True
                or summary.get("interpreter_named_path_matched_open_descriptor_after_launch")
                is not True
                or not isinstance(summary.get("interpreter_preflight_descriptor_sha256"), str)
                or len(summary["interpreter_preflight_descriptor_sha256"]) != 64
                or summary.get("trusted_runtime_dependency")
                != "current_python_interpreter_named_path"
                or "interpreter_identity_revalidated" in summary
                or "interpreter_sha256" in summary
                or summary.get("neutral_launch_working_directory_used") is not True
                or summary.get("scanner_working_directory_is_candidate_descriptor") is not True
                or summary.get("os_process_identity_inherited") is not True
                or summary.get("host_filesystem_access_available") is not True
                or summary.get("host_network_access_available") is not True
                or summary.get("scanner_sha256") != digest
            ):
                raise SystemExit(f"independent scanner lost its proof bindings: {summary}")
            if _snapshot(candidate) != candidate_before or _snapshot(scanner) != scanner_before:
                raise SystemExit("independent scanner adapter mutated candidate or executable")

            cli_output = io.StringIO()
            with contextlib.redirect_stdout(cli_output):
                cli_code = scan_main(
                    [
                        "--candidate",
                        os.fspath(candidate),
                        "--scanner",
                        os.fspath(scanner),
                        "--scanner-sha256",
                        digest,
                        "--profile",
                        "preview",
                    ]
                )
            rendered = cli_output.getvalue()
            cli_payload = json.loads(rendered)
            if (
                cli_code != (0 if clean else 1)
                or cli_payload.get("clean") is not clean
                or cli_payload.get("finding_count") != count
                or os.fspath(candidate) in rendered
                or os.fspath(scanner) in rendered
            ):
                raise SystemExit(f"independent scanner CLI receipt drifted: {cli_payload}")


def test_executable_custody_protocol_and_candidate_mutation_fail_closed() -> None:
    with _preview_candidate() as candidate, TemporaryDirectory(
        prefix="jarvis-independent-invalid-", dir="/private/tmp"
    ) as temp:
        root = Path(temp)
        scanner, digest = _write_scanner(root)
        _expect_error(candidate, scanner, "0" * 64, "scanner_sha256_mismatch")

        link = root / "scanner-link"
        link.symlink_to(scanner)
        _expect_error(candidate, link, digest, "scanner_executable_invalid")

        malformed, malformed_digest = _write_scanner(root, malformed_output='{"clean":true}')
        _expect_error(candidate, malformed, malformed_digest, "scanner_output_invalid")

        noisy, noisy_digest = _write_scanner(root, stderr_text="private scanner detail")
        _expect_error(candidate, noisy, noisy_digest, "scanner_output_invalid")

        invalid_source = root / "invalid-source-contract"
        invalid_source.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        invalid_source.chmod(0o500)
        invalid_digest = hashlib.sha256(invalid_source.read_bytes()).hexdigest()
        _expect_error(
            candidate,
            invalid_source,
            invalid_digest,
            "scanner_source_contract_invalid",
        )

        mutating, mutating_digest = _write_scanner(root, mutate_candidate=True)
        _expect_error(candidate, mutating, mutating_digest, "candidate_changed")


def test_verified_program_bytes_survive_scanner_path_substitution() -> None:
    with _preview_candidate() as candidate, TemporaryDirectory(
        prefix="jarvis-independent-swap-", dir="/private/tmp"
    ) as temp:
        root = Path(temp)
        scanner, digest = _write_scanner(root, clean=True, finding_count=0)
        replacement, replacement_digest = _write_scanner(root, clean=False, finding_count=7)
        if replacement_digest == digest:
            raise SystemExit("scanner swap fixture did not create distinct programs")
        backup = root / "verified-scanner-backup"
        real_run = scan_module._run_scanner_bounded
        observed: dict[str, object] = {}

        def swap_then_run(command, program, **kwargs):
            observed["program_sha256"] = hashlib.sha256(program).hexdigest()
            observed["command_contains_scanner_path"] = os.fspath(scanner) in command
            scanner.rename(backup)
            replacement.rename(scanner)
            try:
                result = real_run(command, program, **kwargs)
                observed["result"] = result
                return result
            finally:
                scanner.rename(replacement)
                backup.rename(scanner)

        try:
            with mock.patch.object(
                scan_module,
                "_run_scanner_bounded",
                side_effect=swap_then_run,
            ):
                scan_candidate_independently(
                    candidate,
                    scanner,
                    digest,
                    profile="preview",
                    environ={},
                )
        except IndependentCredentialScanError as exc:
            if exc.code != "scanner_identity_changed":
                raise SystemExit(f"scanner path-swap failure drifted: {exc.code}")

        result = observed.get("result")
        if (
            observed.get("program_sha256") != digest
            or observed.get("command_contains_scanner_path") is not False
            or not isinstance(result, tuple)
            or result[0] != 0
            or json.loads(result[1]).get("clean") is not True
        ):
            raise SystemExit(f"scanner execution was not bound to verified bytes: {observed}")


def test_candidate_path_replacement_fails_closed() -> None:
    with _preview_candidate() as candidate, TemporaryDirectory(
        prefix="jarvis-independent-candidate-swap-", dir="/private/tmp"
    ) as temp:
        root = Path(temp)
        clean_payload = _scanner_payload(True, 0)
        finding_payload = _scanner_payload(False, 1)
        scanner = root / "candidate-binding-scanner"
        scanner.write_text(
            "\n".join(
                (
                    f"#!{sys.executable}",
                    "# jarvis-independent-credential-scanner:1",
                    "import pathlib",
                    "import sys",
                    "candidate = pathlib.Path(sys.argv[2])",
                    "original = (candidate / 'jarvis_v2' / 'module.py').read_text(encoding='utf-8')",
                    f"print({clean_payload!r} if 'synthetic public source' in original else {finding_payload!r})",
                    "raise SystemExit(0 if 'synthetic public source' in original else 1)",
                    "",
                )
            ),
            encoding="utf-8",
        )
        scanner.chmod(0o500)
        digest = hashlib.sha256(scanner.read_bytes()).hexdigest()
        backup = candidate.with_name(f"jarvis-independent-backup-{uuid.uuid4().hex}")
        real_run = scan_module._run_scanner_bounded
        observed: dict[str, object] = {}

        def replace_candidate_during_scan(command, program, **kwargs):
            inherited_fds = tuple(kwargs.get("pass_fds", ()))
            observed["candidate_path_in_command"] = os.fspath(candidate) in command
            observed["one_descriptor_passed"] = (
                len(inherited_fds) == 1 and str(inherited_fds[0]) in command
            )
            candidate.rename(backup)
            (candidate / "jarvis_v2").mkdir(parents=True, mode=0o700)
            (candidate / "jarvis_v2" / "module.py").write_text(
                "VALUE = 'alternate path tree'\n",
                encoding="utf-8",
            )
            try:
                result = real_run(command, program, **kwargs)
                observed["result"] = result
                return result
            finally:
                shutil.rmtree(candidate)
                backup.rename(candidate)

        try:
            with mock.patch.object(
                scan_module,
                "_run_scanner_bounded",
                side_effect=replace_candidate_during_scan,
            ):
                scan_candidate_independently(
                    candidate,
                    scanner,
                    digest,
                    profile="preview",
                    environ={},
                )
        except IndependentCredentialScanError as exc:
            if exc.code != "candidate_changed":
                raise SystemExit(f"candidate path replacement failure drifted: {exc.code}")
        else:
            raise SystemExit("candidate path replacement passed the stable-root proof")
        result = observed.get("result")
        if (
            not isinstance(result, tuple)
            or result[0] != 0
            or json.loads(result[1]).get("clean") is not True
            or observed.get("candidate_path_in_command") is not False
            or observed.get("one_descriptor_passed") is not True
        ):
            raise SystemExit("scanner followed the replacement pathname instead of the candidate fd")


def test_interpreter_named_path_and_descriptor_are_revalidated_without_execution_attestation() -> None:
    with _preview_candidate() as candidate, TemporaryDirectory(
        prefix="jarvis-independent-interpreter-", dir="/private/tmp"
    ) as temp:
        scanner, digest = _write_scanner(Path(temp))
        with mock.patch.object(scan_module, "_interpreter_stable", return_value=False):
            _expect_error(
                candidate,
                scanner,
                digest,
                "interpreter_identity_changed",
            )


def test_sensitive_error_causes_and_tracebacks_are_suppressed() -> None:
    private_output = "private-scanner-output-sentinel-7731"
    with _preview_candidate() as candidate, TemporaryDirectory(
        prefix="jarvis-independent-errors-", dir="/private/tmp"
    ) as temp:
        root = Path(temp)
        malformed, digest = _write_scanner(root, malformed_output=private_output)
        try:
            scan_candidate_independently(
                candidate,
                malformed,
                digest,
                profile="preview",
                environ={},
            )
        except IndependentCredentialScanError as exc:
            rendered = "".join(traceback.format_exception(exc))
            if (
                exc.code != "scanner_output_invalid"
                or exc.__cause__ is not None
                or exc.__context__ is not None
                or private_output in rendered
                or os.fspath(candidate) in rendered
                or os.fspath(malformed) in rendered
            ):
                raise SystemExit("scanner output escaped through a sanitized adapter error")
        else:
            raise SystemExit("malformed private scanner output unexpectedly passed")

        missing = root / "private-missing-scanner-path-sentinel"
        try:
            scan_candidate_independently(
                candidate,
                missing,
                "0" * 64,
                profile="preview",
                environ={},
            )
        except IndependentCredentialScanError as exc:
            rendered = "".join(traceback.format_exception(exc))
            if (
                exc.__cause__ is not None
                or exc.__context__ is not None
                or os.fspath(missing) in rendered
            ):
                raise SystemExit("scanner path escaped through a sanitized adapter error")
        else:
            raise SystemExit("missing scanner path unexpectedly passed")


def _wait_for_cleanup_pipe_byte(
    read_fd: int,
    *,
    process_group_id: int,
    label: str,
) -> None:
    deadline = time.monotonic() + PROCESS_CLEANUP_PIPE_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        try:
            value = os.read(read_fd, 1)
        except BlockingIOError:
            time.sleep(0.01)
            continue
        if value == b"R":
            return
        break
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass
    raise SystemExit(f"{label} did not publish its ready byte")


def _assert_cleanup_pipe_reaches_eof(
    read_fd: int,
    *,
    process_group_id: int,
    label: str,
) -> None:
    deadline = time.monotonic() + PROCESS_CLEANUP_PIPE_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        try:
            value = os.read(read_fd, 1)
        except BlockingIOError:
            time.sleep(0.01)
            continue
        if value == b"":
            return
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass
    raise SystemExit(f"{label} retained an executable pipe holder after cleanup")


def test_output_limit_and_timeout_stop_the_entire_scanner_process_group() -> None:
    parent_script = """
import subprocess
import sys
import time

write_fd = int(sys.argv[1])
child = subprocess.Popen([
    sys.executable,
    "-c",
    (
        "import os,signal,sys,time; "
        "write_fd=int(sys.argv[1]); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "os.write(write_fd,b'R'); "
        "time.sleep(60)"
    ),
    str(write_fd),
], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
   close_fds=True, pass_fds=(write_fd,))
sys.stdin.buffer.read()
if sys.argv[2] == "overflow":
    print("X" * 100000, flush=True)
if sys.argv[2] == "success":
    print("bounded success", flush=True)
    raise SystemExit(0)
time.sleep(60)
""".strip()
    real_popen = subprocess.Popen
    for mode in ("overflow", "timeout", "success"):
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
        synthetic_process: subprocess.Popen[bytes] | None = None
        write_fd_open = True
        try:

            def synthetic_popen(_command, **kwargs):
                nonlocal synthetic_process
                inherited_fds = tuple(kwargs.get("pass_fds", ()))
                kwargs["pass_fds"] = (*inherited_fds, write_fd)
                synthetic_process = real_popen(
                    [sys.executable, "-B", "-c", parent_script, str(write_fd), mode],
                    **kwargs,
                )
                _wait_for_cleanup_pipe_byte(
                    read_fd,
                    process_group_id=synthetic_process.pid,
                    label=f"scanner {mode} descendant",
                )
                return synthetic_process

            timeout = 0.2 if mode == "timeout" else 2.0
            with mock.patch.object(
                scan_module.subprocess,
                "Popen",
                side_effect=synthetic_popen,
            ), mock.patch.object(scan_module, "SCANNER_TIMEOUT_SECONDS", timeout):
                try:
                    result = _run_scanner_bounded(("/synthetic/scanner",), b"synthetic program")
                except IndependentCredentialScanError as exc:
                    expected = (
                        "scanner_output_limit" if mode == "overflow" else "scanner_timeout"
                    )
                    allowed_codes = {"scanner_cleanup_unknown"}
                    if mode != "success":
                        allowed_codes.add(expected)
                    if exc.code not in allowed_codes:
                        raise SystemExit(f"scanner process-group failure drifted: {exc.code}")
                    result = None
                else:
                    if mode != "success":
                        raise SystemExit(f"scanner {mode} fixture unexpectedly completed")
                    if result != (0, b"bounded success\n", b""):
                        raise SystemExit(f"successful scanner output drifted: {result}")

            if synthetic_process is None:
                raise SystemExit("scanner cleanup fixture did not launch")
            os.close(write_fd)
            write_fd_open = False
            _assert_cleanup_pipe_reaches_eof(
                read_fd,
                process_group_id=synthetic_process.pid,
                label=f"scanner {mode} descendant",
            )
        finally:
            if write_fd_open:
                os.close(write_fd)
            os.close(read_fd)


def test_unknown_process_group_cleanup_fails_closed() -> None:
    child_script = "import sys; sys.stdin.buffer.read(); print('bounded success')"
    real_popen = subprocess.Popen

    def synthetic_popen(_command, **kwargs):
        return real_popen([sys.executable, "-B", "-c", child_script], **kwargs)

    with mock.patch.object(
        scan_module.subprocess,
        "Popen",
        side_effect=synthetic_popen,
    ), mock.patch.object(
        scan_module,
        "_terminate_process_group",
        return_value=False,
    ):
        try:
            _run_scanner_bounded(("/synthetic/scanner",), b"synthetic program")
        except IndependentCredentialScanError as exc:
            if exc.code != "scanner_cleanup_unknown":
                raise SystemExit(f"unknown cleanup failure drifted: {exc.code}")
        else:
            raise SystemExit("unknown scanner process-group cleanup was accepted")


def test_cleanup_unknown_overrides_primary_child_failures_without_chains() -> None:
    child_script = """
import sys
import time

sys.stdin.buffer.read()
if sys.argv[1] == "overflow":
    print("X" * 100000, flush=True)
time.sleep(60)
""".strip()
    real_popen = subprocess.Popen
    real_cleanup = scan_module._terminate_process_group

    def cleanup_then_report_unknown(process):
        real_cleanup(process)
        return False

    for mode in ("timeout", "overflow"):
        def synthetic_popen(_command, **kwargs):
            return real_popen(
                [sys.executable, "-B", "-c", child_script, mode],
                **kwargs,
            )

        timeout = 0.2 if mode == "timeout" else 2.0
        with mock.patch.object(
            scan_module.subprocess,
            "Popen",
            side_effect=synthetic_popen,
        ), mock.patch.object(
            scan_module,
            "_terminate_process_group",
            side_effect=cleanup_then_report_unknown,
        ), mock.patch.object(
            scan_module,
            "SCANNER_TIMEOUT_SECONDS",
            timeout,
        ):
            try:
                _run_scanner_bounded(
                    ("/synthetic/scanner",),
                    b"synthetic program",
                    pass_fds=(),
                )
            except IndependentCredentialScanError as exc:
                if (
                    exc.code != "scanner_cleanup_unknown"
                    or exc.__cause__ is not None
                    or exc.__context__ is not None
                ):
                    raise SystemExit(
                        f"{mode} plus unknown cleanup did not fail chain-free: {exc.code}"
                    )
            else:
                raise SystemExit(f"{mode} plus unknown cleanup was accepted")


def test_cli_failure_is_path_and_content_free() -> None:
    hidden_candidate = "/private/tmp/private-independent-candidate-owner-marker"
    hidden_scanner = "/private/tmp/private-independent-scanner-contact-marker"
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = scan_main(
            [
                "--candidate",
                hidden_candidate,
                "--scanner",
                hidden_scanner,
                "--scanner-sha256",
                "not-a-digest",
                "--profile",
                "preview",
            ]
        )
    rendered = output.getvalue()
    payload = json.loads(rendered)
    if code != 2 or payload.get("ok") is not False:
        raise SystemExit(f"independent scanner CLI did not fail safely: {payload}")
    if hidden_candidate in rendered or hidden_scanner in rendered:
        raise SystemExit("independent scanner CLI leaked a supplied path")
    for key in (
        "paths_included",
        "private_content_included",
        "scanner_output_included",
        "scanner_source_bytes_delivered_to_bootstrap_exactly",
        "scanner_path_used_for_child_execution",
        "candidate_access_bound_to_open_descriptor",
        "candidate_descriptor_inherited_by_child",
        "interpreter_named_path_matched_open_descriptor_before_launch",
        "interpreter_open_descriptor_stable_after_launch",
        "interpreter_named_path_matched_open_descriptor_after_launch",
        "interpreter_execution_bound_to_hashed_descriptor",
        "interpreter_execution_identity_attested",
        "scanner_acquired",
        "scanner_version_chosen",
        "shell_used",
        "credential_environment_inherited",
        "kernel_network_denial_attested",
        "kernel_filesystem_write_denial_attested",
        "adapter_writes_files",
        "publishes",
        "authorizes_publication",
    ):
        if payload.get(key) is not False:
            raise SystemExit(f"independent scanner CLI boundary drifted: {payload}")
    if Path(hidden_candidate).exists() or Path(hidden_scanner).exists():
        raise SystemExit("independent scanner CLI created a supplied path")


def main() -> None:
    test_clean_and_finding_receipts_are_content_free_and_inert()
    test_executable_custody_protocol_and_candidate_mutation_fail_closed()
    test_verified_program_bytes_survive_scanner_path_substitution()
    test_candidate_path_replacement_fails_closed()
    test_interpreter_named_path_and_descriptor_are_revalidated_without_execution_attestation()
    test_sensitive_error_causes_and_tracebacks_are_suppressed()
    test_output_limit_and_timeout_stop_the_entire_scanner_process_group()
    test_unknown_process_group_cleanup_fails_closed()
    test_cleanup_unknown_overrides_primary_child_failures_without_chains()
    test_cli_failure_is_path_and_content_free()
    print("independent credential scanner smoke passed")


if __name__ == "__main__":
    main()
