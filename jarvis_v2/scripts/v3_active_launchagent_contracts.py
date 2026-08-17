"""Build supervised V3 service contracts without activating them.

This module has one deliberately narrow effect: it materializes validated plist
files in a new directory below ``/private/tmp``.  It does not inspect environment
file contents and has no process-control or service-management capability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import pwd
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path("/private/tmp")
MANIFEST_NAME = "manifest.json"
DAEMON_ENABLE_KEY = "JARVIS_V3_ENABLE_DAEMONS"
ENV_FILE_KEY = "JARVIS_V3_ENV"
SCHEDULER_ENABLE_KEY = "JARVIS_V3_ENABLE_SCHEDULER"
EXPECTED_PATH = "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
DIRECTORY_MODE = 0o700
FILE_MODE = 0o600
MAX_TEMPLATE_BYTES = 64 * 1024
LOG_DIRECTORY_RELATIVE = Path("Library") / "Logs" / "JarvisV3"
LOG_DIRECTORY_MODE = 0o700
LOG_FILE_MODE = 0o600


@dataclass(frozen=True)
class ServiceContract:
    name: str
    module: str
    log_filename: str
    throttle_interval: int | None = None

    @property
    def filename(self) -> str:
        return f"com.jarvis-v3.{self.name}.plist"

    @property
    def label(self) -> str:
        return f"com.jarvis-v3.{self.name}"

    @property
    def log_path(self) -> str:
        return os.fspath(_account_log_directory() / self.log_filename)


SERVICE_CONTRACTS = (
    ServiceContract("telegram", "jarvis_v2.scripts.run_telegram_control", "telegram.log"),
    ServiceContract("imessage", "jarvis_v2.scripts.run_imessage_control", "imessage.log"),
    ServiceContract(
        "dashboard",
        "jarvis_v2.scripts.run_status_server",
        "dashboard.log",
        throttle_interval=30,
    ),
)

_ALLOWED_REASON_CODES = frozenset(
    {
        "env_not_absolute",
        "env_missing_or_unreadable",
        "env_not_regular",
        "env_symlink",
        "env_not_owner_only",
        "env_wrong_owner",
        "env_unstable",
        "output_not_absolute",
        "output_outside_private_tmp",
        "output_parent_invalid",
        "output_exists",
        "template_invalid",
        "template_unstable",
        "account_home_invalid",
        "write_failed",
    }
)


class ContractBuildError(RuntimeError):
    """Bounded failure that never renders a supplied local path or file content."""

    def __init__(self, reason_code: str):
        bounded = reason_code if reason_code in _ALLOWED_REASON_CODES else "write_failed"
        self.reason_code = bounded
        super().__init__(bounded)


def _fail(reason_code: str) -> NoReturn:
    raise ContractBuildError(reason_code)


def _same_entry(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _account_log_directory() -> Path:
    """Return the canonical inert log target without creating any directory."""

    try:
        account = pwd.getpwuid(os.geteuid())
    except (KeyError, OSError):
        _fail("account_home_invalid")
    raw_home = getattr(account, "pw_dir", None)
    if type(raw_home) is not str:
        _fail("account_home_invalid")
    home = Path(raw_home)
    if not home.is_absolute() or not home.name:
        _fail("account_home_invalid")
    try:
        info = os.stat(home, follow_symlinks=False)
    except (OSError, ValueError):
        _fail("account_home_invalid")
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.geteuid()
    ):
        _fail("account_home_invalid")
    return home / LOG_DIRECTORY_RELATIVE


def _log_contract_manifest() -> dict[str, Any]:
    return {
        "base": "account_home",
        "relative_directory": LOG_DIRECTORY_RELATIVE.as_posix(),
        "directory_mode": f"{LOG_DIRECTORY_MODE:04o}",
        "file_mode": f"{LOG_FILE_MODE:04o}",
        "no_follow_required": True,
        "precreate_required": True,
    }


def _open_stable_regular(path: Path, *, error_prefix: str) -> tuple[int, os.stat_result]:
    if not path.is_absolute() or not path.name:
        _fail(f"{error_prefix}_missing_or_unreadable" if error_prefix == "env" else "template_invalid")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    try:
        parent_fd = os.open("/", directory_flags)
        for component in path.parent.parts[1:]:
            if component in ("", ".", ".."):
                raise OSError("invalid path component")
            next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
    except OSError:
        try:
            os.close(parent_fd)
        except (OSError, UnboundLocalError):
            pass
        _fail(f"{error_prefix}_missing_or_unreadable" if error_prefix == "env" else "template_invalid")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        os.close(parent_fd)
        _fail(f"{error_prefix}_missing_or_unreadable" if error_prefix == "env" else "template_invalid")
    if stat.S_ISLNK(named.st_mode):
        os.close(parent_fd)
        _fail(f"{error_prefix}_symlink" if error_prefix == "env" else "template_invalid")
    if not stat.S_ISREG(named.st_mode):
        os.close(parent_fd)
        _fail(f"{error_prefix}_not_regular" if error_prefix == "env" else "template_invalid")
    try:
        descriptor = os.open(path.name, flags, dir_fd=parent_fd)
    except OSError:
        os.close(parent_fd)
        _fail(f"{error_prefix}_missing_or_unreadable" if error_prefix == "env" else "template_invalid")
    try:
        opened = os.fstat(descriptor)
        after = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        os.close(descriptor)
        os.close(parent_fd)
        _fail(f"{error_prefix}_unstable")
    os.close(parent_fd)
    if (
        not stat.S_ISREG(opened.st_mode)
        or not _same_entry(named, opened)
        or not _same_entry(opened, after)
    ):
        os.close(descriptor)
        _fail(f"{error_prefix}_unstable")
    return descriptor, opened


def _validate_env_file(path: Path) -> str:
    if not path.is_absolute():
        _fail("env_not_absolute")
    descriptor, opened = _open_stable_regular(path, error_prefix="env")
    try:
        if opened.st_uid != os.geteuid():
            _fail("env_wrong_owner")
        if opened.st_mode & 0o077:
            _fail("env_not_owner_only")
        if opened.st_nlink != 1:
            _fail("env_unstable")
    finally:
        os.close(descriptor)
    return os.fspath(path)


def _validate_output_path(path: Path) -> None:
    if not path.is_absolute():
        _fail("output_not_absolute")
    try:
        path.relative_to(OUTPUT_ROOT)
    except ValueError:
        _fail("output_outside_private_tmp")
    if path == OUTPUT_ROOT:
        _fail("output_outside_private_tmp")
    if os.path.lexists(path):
        _fail("output_exists")


def _open_output_parent(path: Path) -> tuple[int, str]:
    """Open the output parent component-by-component without following links."""

    try:
        relative_parent = path.parent.relative_to(OUTPUT_ROOT)
    except ValueError:
        _fail("output_outside_private_tmp")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        parent_fd = os.open(OUTPUT_ROOT, flags)
    except OSError:
        _fail("output_parent_invalid")
    try:
        for component in relative_parent.parts:
            if component in ("", ".", ".."):
                _fail("output_parent_invalid")
            next_fd = os.open(component, flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        if not stat.S_ISDIR(os.fstat(parent_fd).st_mode):
            _fail("output_parent_invalid")
    except ContractBuildError:
        os.close(parent_fd)
        raise
    except OSError:
        os.close(parent_fd)
        _fail("output_parent_invalid")
    return parent_fd, path.name


def _expected_template(contract: ServiceContract) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "Label": contract.label,
        "ProgramArguments": [
            "/opt/homebrew/bin/python3",
            "-m",
            contract.module,
        ],
        "WorkingDirectory": os.fspath(PROJECT_ROOT),
        "EnvironmentVariables": {
            "PATH": EXPECTED_PATH,
            "PYTHONPATH": os.fspath(PROJECT_ROOT),
        },
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProcessType": "Background",
        "StandardOutPath": contract.log_path,
        "StandardErrorPath": contract.log_path,
    }
    if contract.throttle_interval is not None:
        expected["ThrottleInterval"] = contract.throttle_interval
    return expected


def _load_valid_template(template_root: Path, contract: ServiceContract) -> dict[str, Any]:
    path = template_root / contract.filename
    descriptor, opened = _open_stable_regular(path, error_prefix="template")
    try:
        if opened.st_size < 1 or opened.st_size > MAX_TEMPLATE_BYTES:
            _fail("template_invalid")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            try:
                value = plistlib.load(handle)
            except Exception:
                _fail("template_invalid")
        try:
            named_after = os.stat(path, follow_symlinks=False)
            opened_after = os.fstat(descriptor)
        except OSError:
            _fail("template_unstable")
        if not _same_entry(named_after, opened_after):
            _fail("template_unstable")
    finally:
        os.close(descriptor)
    if value != _expected_template(contract):
        _fail("template_invalid")
    return value


def _active_payload(template: dict[str, Any], env_file: str) -> dict[str, Any]:
    active = dict(template)
    environment = dict(template["EnvironmentVariables"])
    environment[DAEMON_ENABLE_KEY] = "1"
    environment[ENV_FILE_KEY] = env_file
    # Override any launchd-manager/global inheritance. Non-scheduler services
    # must carry an explicit disabled value rather than merely omitting the key.
    environment[SCHEDULER_ENABLE_KEY] = "0"
    active["EnvironmentVariables"] = environment
    return active


def _write_new_file(directory_fd: int, name: str, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(name, flags, FILE_MODE, dir_fd=directory_fd)
    try:
        os.fchmod(descriptor, FILE_MODE)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail("write_failed")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _manifest(artifacts: list[tuple[str, bytes]]) -> dict[str, Any]:
    artifact_receipts = [
        {"name": name, "sha256": hashlib.sha256(payload).hexdigest()}
        for name, payload in artifacts
    ]
    contract_set_payload = json.dumps(
        artifact_receipts,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": 2,
        "kind": "jarvis_v3_offline_active_launchagent_contracts",
        "offline_only": True,
        "scheduler_activation_present": False,
        "artifact_count": len(artifacts),
        "contract_set_sha256": hashlib.sha256(contract_set_payload).hexdigest(),
        "log_contract": _log_contract_manifest(),
        "artifacts": artifact_receipts,
    }


def build_active_launchagent_contracts(
    *,
    env_file: Path,
    output_dir: Path,
    template_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Validate inert templates and materialize active contracts offline."""

    selected_env = _validate_env_file(env_file)
    _validate_output_path(output_dir)
    artifacts: list[tuple[str, bytes]] = []
    for contract in SERVICE_CONTRACTS:
        template = _load_valid_template(template_root, contract)
        active = _active_payload(template, selected_env)
        payload = plistlib.dumps(active, fmt=plistlib.FMT_XML, sort_keys=True)
        artifacts.append((contract.filename, payload))
    manifest = _manifest(artifacts)
    manifest_payload = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )

    created = False
    completed = False
    parent_fd: int | None = None
    directory_fd: int | None = None
    try:
        parent_fd, output_name = _open_output_parent(output_dir)
        os.mkdir(output_name, DIRECTORY_MODE, dir_fd=parent_fd)
        created = True
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        if hasattr(os, "O_CLOEXEC"):
            directory_flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        directory_fd = os.open(output_name, directory_flags, dir_fd=parent_fd)
        os.fchmod(directory_fd, DIRECTORY_MODE)
        for name, payload in artifacts:
            _write_new_file(directory_fd, name, payload)
        _write_new_file(directory_fd, MANIFEST_NAME, manifest_payload)
        os.fsync(directory_fd)
        completed = True
    except ContractBuildError:
        raise
    except FileExistsError:
        _fail("output_exists")
    except OSError:
        _fail("write_failed")
    finally:
        if directory_fd is not None:
            if created and not completed:
                for name in (MANIFEST_NAME, *(item[0] for item in reversed(artifacts))):
                    try:
                        os.unlink(name, dir_fd=directory_fd)
                    except OSError:
                        pass
            os.close(directory_fd)
        if created and not completed and parent_fd is not None:
            try:
                os.rmdir(output_dir.name, dir_fd=parent_fd)
            except OSError:
                pass
        if parent_fd is not None:
            os.close(parent_fd)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build inactive-on-disk V3 LaunchAgent contracts under /private/tmp."
    )
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = build_active_launchagent_contracts(
            env_file=args.env_file,
            output_dir=args.output_dir,
        )
    except ContractBuildError as exc:
        print(json.dumps({"ok": False, "reason": exc.reason_code}, sort_keys=True))
        return 2
    print(json.dumps({"ok": True, **manifest}, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
