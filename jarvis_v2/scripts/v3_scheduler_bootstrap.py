"""Fail-closed V3 scheduler timezone bootstrap.

This module is selected by inert offline V3 continuity contracts, but no
installed or live service is wired to it.  If invoked directly, it checks both
V3 scheduler gates before inspecting the bundled runtime.  It then selects and
verifies the bundle's exact Asia/Seoul timezone for both Python ``zoneinfo``
and libc before importing the existing scheduler runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import stat
import sys
import time
import unicodedata
import zoneinfo

from jarvis_v2.scripts.daemon_gate import (
    require_v3_daemon_enable,
    require_v3_scheduler_enable,
)


ZONE_KEY = "Asia/Seoul"
PYTHON_PREFIX_NAME = "python"
INTERPRETER_DIRECTORY_NAME = "bin"
INTERPRETER_NAME = "python3"
TZPATH_RELATIVE = Path("share/zoneinfo")
TZIF_RELATIVE = TZPATH_RELATIVE / "Asia/Seoul"
MAX_TZIF_BYTES = 1024 * 1024
MAX_EXECUTABLE_PATH_BYTES = 4096
TARGET_SCHEDULER_MODULE = "jarvis_v2.scripts.run_scheduler"
BOOTSTRAP_FAILURE_EXIT = 5
BOOTSTRAP_FAILURE_MESSAGE = (
    "Jarvis V3 scheduler timezone bootstrap stopped before scheduler invocation. "
    "The bundled Asia/Seoul runtime is unavailable, invalid, or the import "
    "boundary is not clean. The bootstrap did not construct or run the scheduler."
)
_EXPECTED_OFFSET = timedelta(hours=9)
_PROBE_UTC = (
    datetime(2026, 1, 1, tzinfo=timezone.utc),
    datetime(2026, 7, 1, tzinfo=timezone.utc),
)
_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_DIRECTORY", 0)
)


class SchedulerTimezoneBootstrapError(RuntimeError):
    """Path-free refusal raised before scheduler import."""

    def __init__(self) -> None:
        super().__init__("scheduler_timezone_bootstrap_failed")


def _fail() -> None:
    raise SchedulerTimezoneBootstrapError()


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


@dataclass(frozen=True, repr=False)
class _HeldNode:
    parent_fd: int
    name: str
    fd: int
    token: tuple[int, ...]


@dataclass(frozen=True, repr=False)
class _TimezoneCustody:
    tzpath: Path
    tzif: Path
    tzif_fd: int
    nodes: tuple[_HeldNode, ...]


def _validate_directory(info: os.stat_result, *, sealed: bool) -> None:
    if not stat.S_ISDIR(info.st_mode) or info.st_nlink < 1:
        _fail()
    if sealed and (
        info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o500
    ):
        _fail()


def _open_directory_node(
    parent_fd: int,
    name: str,
    *,
    sealed: bool,
) -> _HeldNode:
    descriptor: int | None = None
    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        opened = os.fstat(descriptor)
        if _stat_token(named) != _stat_token(opened):
            _fail()
        _validate_directory(opened, sealed=sealed)
        result = _HeldNode(parent_fd, name, descriptor, _stat_token(opened))
        descriptor = None
        return result
    except SchedulerTimezoneBootstrapError:
        raise
    except OSError:
        _fail()
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _open_file_node(
    parent_fd: int,
    name: str,
    *,
    mode: int,
    executable: bool,
    bounded_tzif: bool,
) -> _HeldNode:
    descriptor: int | None = None
    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(name, _READ_FLAGS, dir_fd=parent_fd)
        opened = os.fstat(descriptor)
        if (
            _stat_token(named) != _stat_token(opened)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != mode
            or opened.st_mode & (stat.S_ISUID | stat.S_ISGID)
            or (executable and not stat.S_IMODE(opened.st_mode) & stat.S_IXUSR)
            or (bounded_tzif and not 44 <= opened.st_size <= MAX_TZIF_BYTES)
        ):
            _fail()
        result = _HeldNode(parent_fd, name, descriptor, _stat_token(opened))
        descriptor = None
        return result
    except SchedulerTimezoneBootstrapError:
        raise
    except OSError:
        _fail()
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _close_nodes(nodes: list[_HeldNode] | tuple[_HeldNode, ...]) -> bool:
    failed = False
    for node in reversed(nodes):
        try:
            os.close(node.fd)
        except OSError:
            failed = True
    return not failed


def _verify_nodes(nodes: tuple[_HeldNode, ...]) -> bool:
    try:
        for node in nodes:
            named = os.stat(node.name, dir_fd=node.parent_fd, follow_symlinks=False)
            opened = os.fstat(node.fd)
            if _stat_token(named) != node.token or _stat_token(opened) != node.token:
                return False
    except OSError:
        return False
    return True


def _open_timezone_custody() -> _TimezoneCustody:
    try:
        executable_text = os.fspath(sys.executable)
        executable_bytes = executable_text.encode("utf-8", errors="strict")
        executable = Path(executable_text)
    except (TypeError, ValueError, UnicodeError):
        _fail()
    if (
        not executable_text
        or len(executable_bytes) > MAX_EXECUTABLE_PATH_BYTES
        or "\x00" in executable_text
        or unicodedata.normalize("NFKC", executable_text) != executable_text
        or any(
            unicodedata.category(character).startswith("C")
            for character in executable_text
        )
        or not executable_text.startswith("/")
        or executable_text.startswith("//")
        or executable_text != os.path.normpath(executable_text)
        or not executable.is_absolute()
        or any(part in {".", ".."} for part in executable.parts)
        or executable.name != INTERPRETER_NAME
        or executable.parent.name != INTERPRETER_DIRECTORY_NAME
        or executable.parent.parent.name != PYTHON_PREFIX_NAME
        or executable.parent.parent.parent.name != "runtime"
    ):
        _fail()

    prefix = executable.parent.parent
    tzpath = prefix / TZPATH_RELATIVE
    tzif = prefix / TZIF_RELATIVE
    nodes: list[_HeldNode] = []
    root_fd: int | None = None
    try:
        root_fd = os.open("/", _DIRECTORY_FLAGS)
        root_info = os.fstat(root_fd)
        _validate_directory(root_info, sealed=False)
        root = _HeldNode(root_fd, ".", root_fd, _stat_token(root_info))
        nodes.append(root)
        parent_fd = root_fd
        root_fd = None
        parts = executable.parts[1:]
        sealed_start = len(parts) - 5
        prefix_fd: int | None = None
        for index, name in enumerate(parts[:-1]):
            node = _open_directory_node(
                parent_fd,
                name,
                sealed=index >= sealed_start,
            )
            nodes.append(node)
            parent_fd = node.fd
            if index == len(parts) - 3:
                prefix_fd = node.fd
        interpreter = _open_file_node(
            parent_fd,
            parts[-1],
            mode=0o500,
            executable=True,
            bounded_tzif=False,
        )
        nodes.append(interpreter)
        if prefix_fd is None:
            _fail()
        share = _open_directory_node(prefix_fd, "share", sealed=True)
        nodes.append(share)
        zoneinfo_directory = _open_directory_node(share.fd, "zoneinfo", sealed=True)
        nodes.append(zoneinfo_directory)
        asia = _open_directory_node(zoneinfo_directory.fd, "Asia", sealed=True)
        nodes.append(asia)
        tzif_node = _open_file_node(
            asia.fd,
            "Seoul",
            mode=0o400,
            executable=False,
            bounded_tzif=True,
        )
        nodes.append(tzif_node)
        header = os.read(tzif_node.fd, 5)
        if len(header) != 5 or header[:4] != b"TZif" or header[4:5] not in {b"\x00", b"2", b"3", b"4"}:
            _fail()
        os.lseek(tzif_node.fd, 0, os.SEEK_SET)
        if not _verify_nodes(tuple(nodes)):
            _fail()
        return _TimezoneCustody(tzpath, tzif, tzif_node.fd, tuple(nodes))
    except SchedulerTimezoneBootstrapError:
        raise
    except OSError:
        _fail()
    finally:
        if sys.exc_info()[0] is not None:
            _close_nodes(nodes)
            if root_fd is not None:
                try:
                    os.close(root_fd)
                except OSError:
                    pass


def _has_expected_offsets(zone: zoneinfo.ZoneInfo) -> bool:
    for instant in _PROBE_UTC:
        if instant.astimezone(zone).utcoffset() != _EXPECTED_OFFSET:
            return False
    return True


def _libc_matches(zone: zoneinfo.ZoneInfo) -> bool:
    for instant in _PROBE_UTC:
        epoch = instant.timestamp()
        expected = instant.astimezone(zone)
        observed = time.localtime(epoch)
        if (
            (observed.tm_year, observed.tm_mon, observed.tm_mday)
            != (expected.year, expected.month, expected.day)
            or (observed.tm_hour, observed.tm_min, observed.tm_sec)
            != (expected.hour, expected.minute, expected.second)
            or getattr(observed, "tm_gmtoff", None) != 32_400
        ):
            return False
    return True


def _restore_timezone(
    previous_tzpath: tuple[str, ...],
    previous_tz_present: bool,
    previous_tz: str | None,
) -> None:
    try:
        zoneinfo.reset_tzpath(previous_tzpath)
        zoneinfo.ZoneInfo.clear_cache()
        if previous_tz_present and previous_tz is not None:
            os.environ["TZ"] = previous_tz
        else:
            os.environ.pop("TZ", None)
        time.tzset()
    except (OSError, RuntimeError, ValueError):
        pass


def _select_bundled_timezone() -> None:
    custody: _TimezoneCustody | None = None
    previous_tzpath = tuple(zoneinfo.TZPATH)
    previous_tz_present = "TZ" in os.environ
    previous_tz = os.environ.get("TZ")
    changed = False
    failed = False
    try:
        custody = _open_timezone_custody()
        duplicate_fd: int | None = None
        try:
            duplicate_fd = os.dup(custody.tzif_fd)
            stream = os.fdopen(duplicate_fd, "rb")
            duplicate_fd = None
        finally:
            if duplicate_fd is not None:
                try:
                    os.close(duplicate_fd)
                except OSError:
                    pass
        with stream:
            exact_zone = zoneinfo.ZoneInfo.from_file(stream, key=ZONE_KEY)
        if not _verify_nodes(custody.nodes):
            _fail()

        changed = True
        zoneinfo.reset_tzpath((os.fspath(custody.tzpath),))
        zoneinfo.ZoneInfo.clear_cache()
        os.environ["TZ"] = f":{os.fspath(custody.tzif)}"
        time.tzset()
        if tuple(zoneinfo.TZPATH) != (os.fspath(custody.tzpath),):
            _fail()
        if (
            not _verify_nodes(custody.nodes)
            or not _has_expected_offsets(exact_zone)
            or not _libc_matches(exact_zone)
        ):
            _fail()
    except SchedulerTimezoneBootstrapError:
        failed = True
    except (OSError, RuntimeError, TypeError, ValueError, OverflowError, zoneinfo.ZoneInfoNotFoundError):
        failed = True
    finally:
        close_ok = custody is None or _close_nodes(custody.nodes)
    if failed or not close_ok:
        if changed:
            _restore_timezone(previous_tzpath, previous_tz_present, previous_tz)
        raise SchedulerTimezoneBootstrapError() from None


def _run_scheduler() -> None:
    from jarvis_v2.scripts import run_scheduler

    run_scheduler.main()


def main() -> None:
    require_v3_daemon_enable("scheduler")
    require_v3_scheduler_enable()
    if TARGET_SCHEDULER_MODULE in sys.modules:
        print(BOOTSTRAP_FAILURE_MESSAGE, flush=True)
        raise SystemExit(BOOTSTRAP_FAILURE_EXIT)
    try:
        _select_bundled_timezone()
    except SchedulerTimezoneBootstrapError:
        print(BOOTSTRAP_FAILURE_MESSAGE, flush=True)
        raise SystemExit(BOOTSTRAP_FAILURE_EXIT) from None
    _run_scheduler()


if __name__ == "__main__":
    main()
